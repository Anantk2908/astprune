"""Run pruning methods on a split and write per-issue metrics.

    python experiments/evaluate.py --split dev --methods untrained
    python experiments/evaluate.py --split lite --methods ast_trained ast_bm25 --settings oracle_imports

``--methods`` takes method names or the group names from config.yaml
(untrained, trained, ablations). Runs that already exist are skipped, so the
command can be repeated after adding a method. Results land in
results/runs/<split>/<setting>/<method>.jsonl; summarise them with
experiments/summarize.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import torch

from astprune.data.datasets import load_processed
from astprune.data.repos import PROJECT_ROOT
from astprune.evaluation import SETTINGS, evaluate, evaluate_parallel
from astprune.registry import Registry, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "lite", "verified"])
    ap.add_argument("--methods", nargs="+", default=["untrained"])
    ap.add_argument("--settings", nargs="+", default=None, choices=SETTINGS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only the first N issues (smoke test)")
    ap.add_argument("--sample", type=int, default=0,
                    help="run on a fixed random sample of N issues (seed 0), for methods too slow for the full split")
    ap.add_argument("--force", action="store_true", help="redo runs that already exist")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes; each loads its own copy of any model (about 0.6 GB of GPU memory)")
    args = ap.parse_args()

    config = load_config(args.config)
    names: list[str] = []
    for item in args.methods:
        names.extend(config["methods"].get(item, [item]))
    settings = args.settings or config["settings"]

    records = load_processed(args.split)
    if args.limit:
        records = records[: args.limit]
    if args.sample:
        import random

        records = sorted(records, key=lambda r: r["instance_id"])
        records = random.Random(0).sample(records, min(args.sample, len(records)))
    split = args.split if not args.limit else f"{args.split}_limit{args.limit}"

    print(f"{split}: {len(records)} issues, settings={settings}, methods={names}", flush=True)
    started = time.time()
    budgets = config["budgets"]
    if args.workers > 1:
        evaluate_parallel(records, names, config, split, settings, budgets["relative"], budgets["fixed"],
                          workers=args.workers, force=args.force, device=args.device)
    else:
        methods = Registry(config, device=args.device).methods(names)
        evaluate(records, methods, split, settings, budgets["relative"], budgets["fixed"], force=args.force)

    manifest = {
        "split": split, "settings": settings, "methods": names, "n_issues": len(records),
        "random_sample": args.sample or None,
        "config": config, "seconds": round(time.time() - started, 1),
        "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = PROJECT_ROOT / "results" / "runs" / split / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
