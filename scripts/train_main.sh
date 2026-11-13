#!/usr/bin/env bash
# Train the main ranker. Safe to rerun: it continues from the last dev check.
cd "$(dirname "$0")/.."
.venv/bin/python experiments/train_ranker.py --run-name main --epochs 2 --evals-per-epoch 3 --resume >> logs/train_main.log 2>&1
echo "exit=$?" >> logs/train_main.log
