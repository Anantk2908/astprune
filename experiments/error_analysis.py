"""Collect and group the cases where a method lost a gold function.

    python experiments/error_analysis.py --split lite --methods ast_trained ast_bm25

For every issue where a gold function is not kept in full, this reruns the
method on that issue and records why it may have happened: how big the
function is next to the budget, where the scorer ranked it, and whether the
issue text names it at all. Each case gets one category from simple rules
(listed in CATEGORIES, first match wins). The rules only sort cases into
piles. docs/error_analysis.md has the cases we then read by hand.

Output:
  results/error_analysis/<split>_<setting>_<budget>_<method>.jsonl   one row per lost gold function
  results/error_analysis/categories.json                            counts per method
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from astprune.data.datasets import load_processed
from astprune.data.repos import PROJECT_ROOT
from astprune.evaluation import FileLoader, candidate_paths
from astprune.metrics import original_tokens
from astprune.registry import Registry, load_config
from astprune.scorers.bm25 import tokenize
from astprune.select import DROP, FULL, SIG

CATEGORIES = [
    "file not among candidates",
    "function too large for the budget",
    "near miss (next in line or signature kept)",
    "issue does not name the function",
    "named in the issue but outranked",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def issue_mentions(issue: str, qualified_name: str, path: str) -> dict:
    """Does the issue text contain the function name, its class, or its file name?"""
    words = set(_IDENT.findall(issue))
    lowered = {w.lower() for w in words}
    parts = qualified_name.split(".")
    func = parts[-1]
    cls = parts[-2] if len(parts) > 1 else None
    module = path.rsplit("/", 1)[-1].removesuffix(".py")
    return {
        "function": func in words or (len(func) > 4 and func.lower() in lowered),
        "class": bool(cls) and (cls in words or cls.lower() in lowered),
        "module": module in words or path in issue,
    }


def categorise(case: dict) -> str:
    if case["state"] == "file_missing":
        return CATEGORIES[0]
    if case["function_tokens"] > 0.5 * case["budget_tokens"]:
        return CATEGORIES[1]
    if case["state"] == "signature" or case["rank"] <= case["rank_cutoff"] + 3:
        return CATEGORIES[2]
    if not case["mentions"]["function"]:
        return CATEGORIES[3]
    return CATEGORIES[4]


def analyse(method_name: str, method, records: list[dict], setting: str, budget: float) -> list[dict]:
    loader = FileLoader()
    cases: list[dict] = []
    for rec in sorted(records, key=lambda r: (r["repo"], r["base_commit"])):
        if not rec["gold_functions"]:
            continue
        files = loader.load(rec, candidate_paths(rec, setting), {})
        if not files:
            continue
        by_path = {pf.path: pf for pf in files}
        budget_tokens = int(budget * original_tokens(files))
        scores = method.prepare(rec["issue"], files)
        states = method.select_states(scores, files, budget_tokens)
        flat = np.concatenate([np.asarray(scores[pf.path], dtype=np.float64) for pf in files])
        order = np.argsort(-flat, kind="stable")
        rank_of = np.empty(len(flat), dtype=np.int64)
        rank_of[order] = np.arange(1, len(flat) + 1)
        offsets, offset = {}, 0
        for pf in files:
            offsets[pf.path] = offset
            offset += len(pf.nodes)
        kept_ranks = [int(rank_of[offsets[pf.path] + n.idx]) for pf in files for n in pf.nodes
                      if states[pf.path][n.idx] == FULL]
        cutoff = max(kept_ranks) if kept_ranks else 0
        issue_idents = len(set(_IDENT.findall(rec["issue"])))
        for gold in rec["gold_functions"]:
            pf = by_path.get(gold["path"])
            case = {
                "instance_id": rec["instance_id"], "method": method_name, "setting": setting, "budget": budget,
                "path": gold["path"], "function": gold["name"], "function_tokens": gold["tokens"],
                "budget_tokens": budget_tokens, "n_candidates": len(flat), "n_gold_functions": len(rec["gold_functions"]),
                "issue_tokens": len(tokenize(rec["issue"])), "issue_identifiers": issue_idents,
                "mentions": issue_mentions(rec["issue"], gold["name"], gold["path"]),
            }
            if pf is None:
                case.update(state="file_missing", rank=None, rank_cutoff=cutoff)
            else:
                node = next((n for n in pf.nodes if n.start == gold["start"]), None)
                if node is None:
                    continue
                state = states[pf.path][node.idx]
                if state == FULL:
                    continue
                case.update(state={SIG: "signature", DROP: "dropped"}[state],
                            rank=int(rank_of[offsets[pf.path] + node.idx]), rank_cutoff=cutoff)
            case["category"] = categorise(case)
            case["issue_head"] = rec["issue"].strip()[:400]
            cases.append(case)
    loader.close()
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--setting", default="oracle_imports")
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--methods", nargs="+", default=["ast_trained", "ast_bm25"])
    args = ap.parse_args()

    registry = Registry(load_config())
    records = load_processed(args.split)
    folder = PROJECT_ROOT / "results" / "error_analysis"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "setting": args.setting, "budget": args.budget,
               "categories": CATEGORIES, "counts": {}, "gold_functions_total": sum(len(r["gold_functions"]) for r in records)}
    for name in args.methods:
        cases = analyse(name, registry.method(name), records, args.setting, args.budget)
        path = folder / f"{args.split}_{args.setting}_r{args.budget:.2f}_{name}.jsonl"
        with path.open("w") as fh:
            for case in cases:
                fh.write(json.dumps(case) + "\n")
        counts = {c: sum(case["category"] == c for case in cases) for c in CATEGORIES}
        summary["counts"][name] = counts
        print(f"{name}: {len(cases)} lost gold functions -> {path.name}")
        for category, n in counts.items():
            print(f"   {n:4d}  {category}")
    (folder / "categories.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
