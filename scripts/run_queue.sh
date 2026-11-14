#!/usr/bin/env bash
# Everything that needs the GPU after training, one job at a time.
# Safe to rerun: finished evaluation runs are skipped and training resumes.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/queue.log; }
ev() { $PY experiments/evaluate.py "$@"; }

# If an evaluation from an earlier queue is still running, let it finish first.
while pgrep -f "experiments/evaluate[.]py" > /dev/null; do sleep 20; done
log "start"
for split in lite verified; do
  ev --split $split --methods ast_trained ast_hybrid ast_unixcoder --workers 2 > logs/eval_${split}_gpu.log 2>&1
  log "$split main encoder methods exit=$?"
  ev --split $split --methods chunk_unixcoder ast_trained_nosig ast_trained_sig15 ast_trained_sig30 ast_trained_density ast_trained_perfile --settings oracle oracle_imports bm25_5 --workers 2 > logs/eval_${split}_gpu2.log 2>&1
  log "$split chunk encoder and ablations exit=$?"
done
log "MAIN_TEST_DONE"

# Third-party baselines, only if their smoke test passes.
if [ -s results/runs/lite/oracle_imports/swe_pruner.jsonl ] || $PY scripts/smoke_external.py swe_pruner > logs/smoke_swe_pruner.log 2>&1; then
  log "swe_pruner smoke ok"
  ev --split lite --methods swe_pruner --settings oracle oracle_imports > logs/eval_lite_swe_pruner.log 2>&1; log "lite swe_pruner exit=$?"
else log "swe_pruner smoke FAILED"; fi
# The two LongCodeZip modes are smoke-tested in separate processes: together they
# load two copies of the scoring model and run out of GPU memory.
if [ -s results/runs/lite/oracle/longcodezip_rank.jsonl ] || $PY scripts/smoke_external.py longcodezip_rank > logs/smoke_longcodezip_rank.log 2>&1; then
  log "longcodezip_rank smoke ok"
  ev --split lite --methods longcodezip_rank --settings oracle --sample 100 > logs/eval_lite_lcz_rank.log 2>&1; log "lite longcodezip_rank (100 issues) exit=$?"
else log "longcodezip_rank smoke FAILED"; fi
if [ -s results/runs/lite/oracle/longcodezip.jsonl ] || $PY scripts/smoke_external.py longcodezip > logs/smoke_longcodezip_full.log 2>&1; then
  log "longcodezip full smoke ok"
  timeout 3h $PY experiments/evaluate.py --split lite --methods longcodezip --settings oracle --sample 100 > logs/eval_lite_lcz_full.log 2>&1; log "lite longcodezip full (100 issues) exit=$?"
else log "longcodezip full smoke FAILED"; fi
log "EXTERNAL_DONE"

[ -s results/scaling/lite.json ] || { $PY experiments/scaling.py --split lite > logs/scaling.log 2>&1; log "scaling gpu exit=$?"; }
[ -s results/error_analysis/categories.json ] || { $PY experiments/error_analysis.py --split lite > logs/error_analysis.log 2>&1; log "error analysis exit=$?"; }
ev --split lite --methods ast_ppl --settings oracle oracle_imports > logs/eval_lite_ppl.log 2>&1; log "lite ppl exit=$?"
[ -s results/downstream/qwen7b_4bit/summary.json ] || { $PY experiments/downstream.py --split lite --n 100 --tag qwen7b_4bit > logs/downstream_7b.log 2>&1; log "downstream 7B exit=$?"; }
log "CORE_DONE"
for run in seed1:1:5 seed2:2:5 unfiltered:0:0; do
  IFS=: read name seed maxgold <<< "$run"
  $PY experiments/train_ranker.py --run-name $name --seed $seed --max-gold-functions $maxgold --epochs 2 --evals-per-epoch 3 --resume > logs/train_$name.log 2>&1; log "train $name exit=$?"
  ev --split dev --methods "ast_trained@$name" --settings oracle_imports > logs/eval_dev_$name.log 2>&1
  ev --split lite --methods "ast_trained@$name" --settings oracle_imports > logs/eval_lite_$name.log 2>&1; log "eval $name exit=$?"
done
log "SEEDS_DONE"
# CPU latency last, with nothing else running, so the timings are clean.
[ -s results/scaling/lite_cpu.json ] || { $PY experiments/scaling.py --split lite --device cpu --sample 10 --tag cpu > logs/scaling_cpu.log 2>&1; log "scaling cpu exit=$?"; }
ev --split verified --methods swe_pruner ast_ppl --settings oracle_imports > logs/eval_verified_external.log 2>&1; log "verified swe_pruner and ppl exit=$?"
log "QUEUE_DONE"
