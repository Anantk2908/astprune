"""How recall and latency change as the candidate set grows from 5 to 10 to 20 files.

    python experiments/scaling.py --split lite --methods ast_trained ast_bm25 ast_unixcoder chunk_bm25

Recall comes from the finished runs in results/runs (so run evaluate.py for the
bm25_5, bm25_10 and bm25_20 settings first). Latency is measured here on a
fixed sample of issues with every cache emptied before each issue, since the
evaluation runs reuse embeddings across issues and would look faster than a
cold call really is.

Output: results/scaling/<split>.json and results/scaling/<split>.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time

import torch

from astprune.data.datasets import load_processed
from astprune.data.repos import PROJECT_ROOT
from astprune.evaluation import FileLoader, candidate_paths, load_runs
from astprune.metrics import original_tokens
from astprune.parse import parse_source
from astprune.registry import Registry, load_config

KS = (5, 10, 20)


def recall_from_runs(split: str, method: str, k: int, budget_label: str) -> tuple[float | None, int]:
    rows = load_runs(split, f"bm25_{k}", [method]).get(method, [])
    values = [r["gold_fn_recall"] for r in rows if r["budget"] == budget_label and r["gold_fn_recall"] is not None]
    return (sum(values) / len(values) if values else None), len(values)


def clear_caches(registry: Registry) -> None:
    for scorer in registry._scorers.values():
        scorer._memo.clear()
        inner = scorer.scorer
        for part in getattr(inner, "scorers", [inner]):
            target = getattr(part, "scorer", part)
            if hasattr(target, "encoder"):
                target.encoder.clear_cache()
            if hasattr(target, "_cache"):
                target._cache.clear()
    for encoder in registry._encoders.values():
        encoder.clear_cache()


def time_method(method, registry: Registry, issue: str, sources: list[tuple[str, str]], budget: float) -> dict:
    """One cold call: parse, score, select and render, each timed."""
    clear_caches(registry)
    t0 = time.perf_counter()
    files = [parse_source(src, path) for path, src in sources]
    t1 = time.perf_counter()
    prepared = method.prepare(issue, files)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    method.prune(prepared, files, int(budget * original_tokens(files)))
    t3 = time.perf_counter()
    return {"parse_ms": (t1 - t0) * 1e3, "score_ms": (t2 - t1) * 1e3, "select_ms": (t3 - t2) * 1e3,
            "total_ms": (t3 - t0) * 1e3, "n_files": len(files), "n_nodes": sum(len(f.nodes) for f in files)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--methods", nargs="+", default=["ast_trained", "ast_bm25", "ast_unixcoder", "chunk_bm25"])
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--sample", type=int, default=60, help="issues used for timing")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="suffix for the output file, for example cpu")
    args = ap.parse_args()

    config = load_config()
    registry = Registry(config, device=args.device)
    methods = registry.methods(args.methods)
    records = [r for r in load_processed(args.split) if len(r["bm25_files"]) >= max(KS)]
    random.Random(0).shuffle(records)
    records = records[: args.sample]
    budget_label = f"r{args.budget:.2f}"

    loader = FileLoader()
    out = {"split": args.split, "budget": args.budget, "timing_issues": len(records),
           "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != "cpu" else None,
           "methods": {}, "parameters": {}}

    # Warm up models so the first timed issue does not pay for CUDA start-up.
    warm = records[0]
    repo = loader.repo(warm["repo"])
    warm_sources = [(p, repo.read(warm["base_commit"], p) or "") for p in candidate_paths(warm, "bm25_5")]
    for method in methods.values():
        time_method(method, registry, warm["issue"], warm_sources, args.budget)

    timings: dict[str, dict[int, list[dict]]] = {m: {k: [] for k in KS} for m in methods}
    for i, rec in enumerate(records, start=1):
        repo = loader.repo(rec["repo"])
        for k in KS:
            sources = [(p, repo.read(rec["base_commit"], p) or "") for p in candidate_paths(rec, f"bm25_{k}")]
            for name, method in methods.items():
                timings[name][k].append(time_method(method, registry, rec["issue"], sources, args.budget))
        if i % 10 == 0:
            print(f"  timed {i}/{len(records)} issues", flush=True)
    loader.close()

    for name in methods:
        points = []
        for k in KS:
            rows = timings[name][k]
            recall, n = recall_from_runs(args.split, name, k, budget_label)
            totals = sorted(r["total_ms"] for r in rows)
            points.append({
                "k": k, "gold_fn_recall": recall, "recall_issues": n,
                "ms_per_issue": statistics.mean(totals),
                "ms_per_issue_median": statistics.median(totals),
                "ms_per_issue_p90": totals[int(0.9 * (len(totals) - 1))],
                "ms_per_file": statistics.mean(r["total_ms"] / r["n_files"] for r in rows),
                "score_ms": statistics.mean(r["score_ms"] for r in rows),
                "select_ms": statistics.mean(r["select_ms"] for r in rows),
                "parse_ms": statistics.mean(r["parse_ms"] for r in rows),
                "nodes": statistics.mean(r["n_nodes"] for r in rows),
            })
        out["methods"][name] = points
    for key, encoder in registry._encoders.items():
        out["parameters"][key] = encoder.n_parameters

    folder = PROJECT_ROOT / "results" / "scaling"
    folder.mkdir(parents=True, exist_ok=True)
    stem = args.split + (f"_{args.tag}" if args.tag else "")
    (folder / f"{stem}.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = [f"# Scaling with candidate files ({args.split}, {int(args.budget * 100)}% budget, {out['device']})", "",
             f"Latency: mean over {len(records)} issues, cold caches, includes parsing. Recall: all issues in the split.", "",
             "| method | files | gold-function recall % | ms per issue | median | p90 | ms per file | score ms | select ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, points in out["methods"].items():
        for p in points:
            recall = f"{p['gold_fn_recall'] * 100:.1f}" if p["gold_fn_recall"] is not None else "n/a"
            lines.append(f"| {name} | {p['k']} | {recall} | {p['ms_per_issue']:.0f} | {p['ms_per_issue_median']:.0f} | "
                         f"{p['ms_per_issue_p90']:.0f} | {p['ms_per_file']:.1f} | {p['score_ms']:.0f} | {p['select_ms']:.0f} |")
    (folder / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
