"""Fine-tune the ranker.

    python experiments/train_ranker.py --run-name main
    python experiments/train_ranker.py --run-name unfiltered --max-gold-functions 0
    python experiments/train_ranker.py --run-name main --promote     # copy its best checkpoint to checkpoints/ranker/best

Needs data/processed/train_examples (build_train_examples.py) and the dev
split (build_data.py --split dev). Model selection uses the dev split only.
Checkpoints go to checkpoints/ranker/<run-name>/, the run record to
results/training/<run-name>.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from astprune.data.datasets import load_processed
from astprune.data.examples import examples_dir, load_examples
from astprune.data.repos import PROJECT_ROOT
from astprune.train import TrainConfig, train

CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "ranker"


def main() -> None:
    defaults = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--examples", default="train_examples")
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--epochs", type=float, default=defaults.epochs)
    ap.add_argument("--batch-issues", type=int, default=defaults.batch_issues)
    ap.add_argument("--same-file-negatives", type=int, default=defaults.same_file_negatives)
    ap.add_argument("--import-negatives", type=int, default=defaults.import_negatives)
    ap.add_argument("--accumulate", type=int, default=defaults.accumulate)
    ap.add_argument("--lr", type=float, default=defaults.lr)
    ap.add_argument("--temperature", type=float, default=defaults.temperature)
    ap.add_argument("--max-gold-functions", type=int, default=defaults.max_gold_functions)
    ap.add_argument("--max-per-repo", type=int, default=defaults.max_per_repo)
    ap.add_argument("--evals-per-epoch", type=int, default=defaults.evals_per_epoch)
    ap.add_argument("--dev-limit", type=int, default=0, help="use only N dev issues (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run of the same name from its last dev check")
    ap.add_argument("--promote", action="store_true",
                    help="do not train; copy this run's best checkpoint to checkpoints/ranker/best")
    args = ap.parse_args()

    run_dir = CKPT_ROOT / args.run_name
    if args.promote:
        if not (run_dir / "best").exists():
            sys.exit(f"no checkpoint at {run_dir / 'best'}")
        target = CKPT_ROOT / "best"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run_dir / "best", target)
        (target / "PROMOTED_FROM.txt").write_text(args.run_name + "\n")
        print(f"promoted {args.run_name} -> {target}")
        return

    cfg = TrainConfig(
        run_name=args.run_name, seed=args.seed, epochs=args.epochs, batch_issues=args.batch_issues,
        same_file_negatives=args.same_file_negatives, import_negatives=args.import_negatives,
        accumulate=args.accumulate, lr=args.lr, temperature=args.temperature,
        max_gold_functions=args.max_gold_functions, max_per_repo=args.max_per_repo,
        evals_per_epoch=args.evals_per_epoch,
    )
    examples, nodes, queries = load_examples(examples_dir(args.examples))
    dev = load_processed("dev")
    if args.dev_limit:
        dev = dev[: args.dev_limit]
    print(f"run {args.run_name}: {len(examples)} examples, {len(dev)} dev issues", flush=True)
    summary = train(cfg, examples, nodes, queries, dev, run_dir, resume=args.resume)

    out = PROJECT_ROOT / "results" / "training"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.run_name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy(run_dir / "train_log.jsonl", out / f"{args.run_name}_log.jsonl")
    print(json.dumps({"best": summary["best"], "train_minutes": summary["train_minutes"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
