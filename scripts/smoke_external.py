"""Check the two third-party baselines on a few dev issues before a long run.

Exits non-zero if an adapter crashes, returns nothing, or goes over budget.
"""

import sys
import time
import traceback

from astprune.data.datasets import load_processed
from astprune.evaluation import FileLoader, candidate_paths
from astprune.metrics import evaluate_instance, original_tokens, output_tokens
from astprune.registry import Registry, load_config

names = sys.argv[1:] or ["swe_pruner", "longcodezip_rank", "longcodezip"]
records = [r for r in sorted(load_processed("dev"), key=lambda r: r["instance_id"]) if r["gold_functions"]][::40][:4]
registry = Registry(load_config())
loader = FileLoader()
failed = False
for name in names:
    try:
        method = registry.method(name)
        if name == "swe_pruner":
            from astprune.baselines.swe_pruner import verify_weights

            model, _ = method._ensure()
            check = verify_weights(model, method.checkpoint)
            print(f"{name}: weights {check['tensors_in_file']} tensors, different {len(check['different'])}, "
                  f"missing {len(check['not_in_model'])}", flush=True)
            failed |= bool(check["different"] or check["not_in_model"])
        for rec in records:
            files = loader.load(rec, candidate_paths(rec, "oracle"), {})
            budget = int(0.25 * original_tokens(files))
            t = time.time()
            ctx = method.prune(method.prepare(rec["issue"], files), files, budget)
            row = evaluate_instance(files, rec["gold_functions"], rec["gold_lines"], ctx)
            used = output_tokens(ctx)
            ok = 0 < used <= budget
            failed |= not ok
            print(f"{name}: {rec['instance_id']} {'ok' if ok else 'BAD'} tokens {used}/{budget} "
                  f"fn recall {row['gold_fn_recall']:.2f} line recall {row['gold_line_recall']} {time.time() - t:.1f}s",
                  flush=True)
    except Exception:  # noqa: BLE001 - report and carry on with the next adapter
        failed = True
        print(f"{name}: FAILED", flush=True)
        traceback.print_exc(limit=5)
loader.close()
sys.exit(1 if failed else 0)
