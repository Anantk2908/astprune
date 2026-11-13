"""Build tokenized training examples from the labelled train splits.

    python experiments/build_train_examples.py --workers 6

Reads data/processed/train.jsonl and gym.jsonl (see build_data.py), drops any
repository that also appears in the dev or test splits, and writes
data/processed/train_examples/. The leakage check is saved to
results/data/leakage_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from astprune.data.datasets import load_processed
from astprune.data.examples import build_repo_examples, examples_dir, save_examples
from astprune.data.repos import PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "gym"])
    ap.add_argument("--held-out", nargs="+", default=["dev", "lite", "verified"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only N issues per repository (smoke test)")
    ap.add_argument("--name", default="train_examples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    held_out_repos: dict[str, set[str]] = {}
    for split in args.held_out:
        held_out_repos[split] = {r["repo"] for r in load_processed(split, only_ok=False)}
    forbidden = set().union(*held_out_repos.values())

    by_repo: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    counts = Counter()
    for split in args.splits:
        for rec in load_processed(split):
            counts[f"{split}_ok"] += 1
            if rec["repo"] in forbidden:
                counts["dropped_repo_overlap"] += 1
                continue
            # SWE-Gym and SWE-bench train share two repositories; keep one copy of an issue.
            key = (rec["repo"], rec["base_commit"], tuple(sorted((g["path"], g["start"]) for g in rec["gold_nodes"])))
            if key in seen:
                counts["dropped_duplicate"] += 1
                continue
            seen.add(key)
            by_repo[rec["repo"]].append(rec)

    train_repos = set(by_repo)
    leakage = {
        "train_repos": sorted(train_repos),
        "held_out_repos": {k: sorted(v) for k, v in held_out_repos.items()},
        "overlap_after_filter": sorted(train_repos & forbidden),
        "dropped_for_overlap": counts["dropped_repo_overlap"],
        "dropped_duplicates": counts["dropped_duplicate"],
    }
    assert not leakage["overlap_after_filter"], "a training repository is also in dev/test"
    out = PROJECT_ROOT / "results" / "data"
    out.mkdir(parents=True, exist_ok=True)
    if not args.limit:
        (out / "leakage_check.json").write_text(json.dumps(leakage, indent=2) + "\n")

    if args.limit:
        by_repo = {k: v[: args.limit] for k, v in by_repo.items()}
    total = sum(len(v) for v in by_repo.values())
    print(f"{total} issues in {len(by_repo)} repositories ({dict(counts)})", flush=True)

    parts = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_repo_examples, name, recs, args.seed): name
                   for name, recs in sorted(by_repo.items(), key=lambda kv: -len(kv[1]))}
        for fut in as_completed(futures):
            part = fut.result()
            parts.append(part)
            print(f"  {part['repo']}: {len(part['examples'])} examples, "
                  f"{len(part['node_tokens'])} node texts ({time.time() - started:.0f}s)", flush=True)

    stats = save_examples(parts, examples_dir(args.name),
                          {"splits": args.splits, "seed": args.seed, "issues_in": total, **counts})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
