"""Build labelled instances for one split.

For every issue this clones the repository (once), reads the files the gold
patch touches as they were before the fix, maps the patch onto AST nodes and
records the candidate files (direct imports, and BM25 top-k for eval splits).

    python experiments/build_data.py --split lite
    python experiments/build_data.py --split train --workers 6

Output: data/processed/<split>.jsonl and results/data/<split>_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from astprune.data.candidates import MAX_CANDIDATE_CHARS, FileRetriever, ModuleIndex, is_test_path
from astprune.data.datasets import PROCESSED_DIR, SPLITS, load_raw, processed_path
from astprune.data.labels import align_edit, edit_matches_file, gold_for_file, parse_patch
from astprune.data.repos import PROJECT_ROOT, Repo, RepoError
from astprune.parse import parse_source

MAX_IMPORT_FILES = 15
BM25_TOP_K = 20


def process_instance(repo: Repo, retriever: FileRetriever | None, row: dict) -> dict:
    rec = {
        "instance_id": row["instance_id"], "repo": row["repo"], "base_commit": row["base_commit"],
        "issue": row["problem_statement"] or "", "status": "ok",
        "gold_functions": [], "gold_nodes": [], "gold_lines": {},
        "oracle_files": [], "import_files": [], "bm25_files": [],
    }
    commit = row["base_commit"]
    if not repo.ensure_commit(commit):
        rec["status"] = "skipped:missing_commit"
        return rec
    edits = parse_patch(row["patch"] or "")
    if not edits:
        rec["status"] = "skipped:no_python_edit"
        return rec

    tree = repo.tree(commit)
    index = ModuleIndex(list(tree))
    imports: list[str] = []
    parse_fallbacks = Counter()
    for path, edit in sorted(edits.items()):
        source = repo.read(commit, path)
        if source is None:
            rec["status"] = "skipped:file_missing"
            return rec
        if len(source) > MAX_CANDIDATE_CHARS:
            rec["oversized_gold_file"] = True
        pf = parse_source(source, path)
        parse_fallbacks[pf.parser] += 1
        rec["gold_file_nodes"] = rec.get("gold_file_nodes", 0) + len(pf.nodes)
        aligned = align_edit(pf, edit)
        if aligned is None or not edit_matches_file(pf, aligned):
            rec["status"] = "skipped:patch_mismatch"
            return rec
        rec["realigned_hunks"] = rec.get("realigned_hunks", 0) + (aligned is not edit)
        edit = aligned
        gold = gold_for_file(pf, edit)
        rec["oracle_files"].append(path)
        rec["gold_lines"][path] = gold.lines
        for idx in gold.node_idx:
            node = pf.nodes[idx]
            entry = {"path": path, "name": node.name, "kind": node.kind, "start": node.start,
                     "end": node.end, "sig_end": node.sig_end, "tokens": node.tokens_full}
            rec["gold_nodes"].append(entry)
            if node.kind in ("function", "method"):
                rec["gold_functions"].append(entry)
        for imp in index.direct_imports(path, source):
            if imp not in imports and not is_test_path(imp):
                imports.append(imp)
    for imp in imports:
        if len(rec["import_files"]) >= MAX_IMPORT_FILES:
            break
        if imp in rec["oracle_files"]:
            continue
        text = repo.read(commit, imp)
        if text is not None and len(text) <= MAX_CANDIDATE_CHARS:
            rec["import_files"].append(imp)
    rec["parsers"] = dict(parse_fallbacks)
    if retriever is not None:
        rec["bm25_files"] = retriever.rank(commit, rec["issue"], top_k=BM25_TOP_K)
    return rec


def process_repo(repo_name: str, rows: list[dict], with_bm25: bool) -> list[dict]:
    try:
        repo = Repo(repo_name)
    except (RepoError, OSError, TimeoutError) as err:
        return [{"instance_id": r["instance_id"], "repo": repo_name, "base_commit": r["base_commit"],
                 "status": "skipped:clone_failed", "error": str(err)[:200]} for r in rows]
    retriever = FileRetriever(repo, max_cache=15_000) if with_bm25 else None
    out = []
    with repo:
        for row in rows:
            try:
                out.append(process_instance(repo, retriever, row))
            except RepoError as err:
                out.append({"instance_id": row["instance_id"], "repo": repo_name,
                            "base_commit": row["base_commit"], "status": "skipped:git_error",
                            "error": str(err)[:200]})
    return out


def summarise(split: str, records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    with_fn = [r for r in ok if r["gold_functions"]]
    n_fn = [len(r["gold_functions"]) for r in with_fn]
    parsers = Counter()
    for r in ok:
        parsers.update(r.get("parsers", {}))
    recall_at = {}
    if any(r["bm25_files"] for r in ok):
        for k in (5, 10, 20):
            hits = [all(p in r["bm25_files"][:k] for p in r["oracle_files"]) for r in ok]
            recall_at[f"all_gold_files_in_top{k}"] = round(sum(hits) / max(len(hits), 1), 4)
    return {
        "split": split,
        "instances_total": len(records),
        "instances_ok": len(ok),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "instances_with_gold_function": len(with_fn),
        "instances_without_gold_function": len(ok) - len(with_fn),
        "gold_functions_total": sum(n_fn),
        "gold_functions_per_instance_mean": round(sum(n_fn) / max(len(n_fn), 1), 3),
        "gold_functions_per_instance_hist": dict(sorted(Counter(min(n, 6) for n in n_fn).items())),
        "oracle_files_per_instance_mean": round(sum(len(r["oracle_files"]) for r in ok) / max(len(ok), 1), 3),
        "import_files_per_instance_mean": round(sum(len(r["import_files"]) for r in ok) / max(len(ok), 1), 3),
        "gold_file_parsers": dict(parsers),
        "nodes_per_gold_file_mean": round(
            sum(r.get("gold_file_nodes", 0) for r in ok) / max(sum(len(r["oracle_files"]) for r in ok), 1), 1),
        "instances_without_gold_lines": sum(not any(r["gold_lines"].values()) for r in ok),
        "bm25_file_recall": recall_at,
        "instances_with_oversized_gold_file": sum(bool(r.get("oversized_gold_file")) for r in ok),
        "instances_with_realigned_hunks": sum(r.get("realigned_hunks", 0) > 0 for r in ok),
        "repos": dict(sorted(Counter(r["repo"] for r in ok).items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--workers", type=int, default=4, help="repositories processed in parallel")
    ap.add_argument("--limit", type=int, default=0, help="only the first N instances (for smoke tests)")
    args = ap.parse_args()

    rows = load_raw(args.split)
    if args.limit:
        rows = rows[: args.limit]
    role = SPLITS[args.split][2]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)
    print(f"{args.split}: {len(rows)} instances in {len(by_repo)} repositories", flush=True)

    records: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_repo, name, repo_rows, role != "train"): name
                   for name, repo_rows in sorted(by_repo.items(), key=lambda kv: -len(kv[1]))}
        for fut in as_completed(futures):
            got = fut.result()
            records.extend(got)
            ok = sum(r["status"] == "ok" for r in got)
            print(f"  {futures[fut]}: {ok}/{len(got)} ok  ({time.time() - started:.0f}s)", flush=True)

    records.sort(key=lambda r: r["instance_id"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with processed_path(args.split).open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    stats = summarise(args.split, records)
    stats_dir = PROJECT_ROOT / "results" / "data"
    stats_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    (stats_dir / f"{args.split}{suffix}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "repos"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
