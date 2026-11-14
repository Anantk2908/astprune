r"""Write report/numbers.tex: every reported number as a LaTeX macro.

    python experiments/make_report_numbers.py

In the report, \res{lite:oracle_imports:ast_trained:r0.10:gold_fn_recall}
prints that mean in percent, with :lo and :hi for the interval ends, and
\dif{lite:oracle_imports:ast_trained:ast_bm25:r0.10} prints a paired
difference (also with :lo, :hi, :p). \stat{...} holds counts and timings.
A key that does not exist prints "??" so a missing result is visible.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from astprune.data.repos import PROJECT_ROOT

SUMMARY = PROJECT_ROOT / "results" / "summary"
OUT = PROJECT_ROOT / "report" / "numbers.tex"
lines: list[str] = []


def put(prefix: str, key: str, value) -> None:
    lines.append(f"\\expandafter\\def\\csname {prefix}:{key}\\endcsname{{{value}}}")


def pct(text: str, signed: bool = False) -> str:
    value = float(text) * 100
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def summaries() -> None:
    for path in sorted(SUMMARY.glob("*.csv")):
        paired = path.stem.endswith("_paired")
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if paired:
                    if r["metric"] != "gold_fn_recall" or r["diff"] == "":
                        continue
                    key = f"{r['split']}:{r['setting']}:{r['a']}:{r['b']}:{r['budget']}"
                    put("d", key, pct(r["diff"], signed=True))
                    put("d", key + ":lo", pct(r["lo"], signed=True))
                    put("d", key + ":hi", pct(r["hi"], signed=True))
                    put("d", key + ":p", "$<$0.001" if float(r["p"]) < 0.001 else f"{float(r['p']):.3f}")
                    put("d", key + ":n", r["n"])
                elif r["mean"] != "":
                    key = f"{r['split']}:{r['setting']}:{r['method']}:{r['budget']}:{r['metric']}"
                    if r["metric"].endswith("_ms_per_file"):
                        put("n", key, f"{float(r['mean']):.1f}")
                    else:
                        put("n", key, pct(r["mean"]))
                        if r["lo"] != "":
                            put("n", key + ":lo", pct(r["lo"]))
                            put("n", key + ":hi", pct(r["hi"]))
                    put("n", key + ":n", r["n"])


def stats() -> None:
    for split in ("lite", "verified", "dev", "train", "gym"):
        path = PROJECT_ROOT / "results" / "data" / f"{split}_stats.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        for k in ("instances_total", "instances_ok", "instances_with_gold_function", "gold_functions_total",
                  "instances_without_gold_lines", "instances_with_realigned_hunks"):
            if k in d:
                put("s", f"{split}:{k}", f"{d[k]:,}")
        put("s", f"{split}:gold_per_issue", f"{d['gold_functions_per_instance_mean']:.2f}")
        put("s", f"{split}:nodes_per_file", f"{d.get('nodes_per_gold_file_mean', 0):.0f}")
        put("s", f"{split}:repos", len(d["repos"]))
        for k, v in d.get("bm25_file_recall", {}).items():
            put("s", f"{split}:{k}", f"{v * 100:.1f}")
        for k, v in d.get("status_counts", {}).items():
            put("s", f"{split}:{k.replace('skipped:', 'skip_')}", f"{v:,}")
    path = PROJECT_ROOT / "data" / "processed" / "train_examples" / "meta.json"
    if path.exists():
        d = json.loads(path.read_text())
        for k in ("examples", "node_texts", "repos", "dropped_duplicate"):
            put("s", f"examples:{k}", f"{d[k]:,}")
    for path in sorted((PROJECT_ROOT / "results" / "training").glob("*.json")):
        d = json.loads(path.read_text())
        run = path.stem
        put("s", f"train:{run}:minutes", f"{d['train_minutes']:.0f}")
        put("s", f"train:{run}:examples", f"{d['examples_used']:,}")
        put("s", f"train:{run}:steps", f"{d['total_batches']:,}")
        put("s", f"train:{run}:best_step", f"{d['best']['step']:,}")
        put("s", f"train:{run}:params_m", f"{d['parameters'] / 1e6:.0f}")
        for h in (d["dev_history"][0], d["best"]):
            tag = "zero" if h["step"] == 0 else "best"
            for k, v in h.items():
                if k.startswith("recall_r"):
                    put("s", f"train:{run}:{tag}:{k.split('_r')[1]}", f"{v * 100:.1f}")
    for path in sorted((PROJECT_ROOT / "results" / "scaling").glob("*.json")):
        d = json.loads(path.read_text())
        for method, points in d["methods"].items():
            for p in points:
                key = f"scaling:{path.stem}:{method}:{p['k']}"
                put("s", key + ":ms", f"{p['ms_per_issue']:.0f}")
                put("s", key + ":ms_file", f"{p['ms_per_file']:.1f}")
                put("s", key + ":score_ms", f"{p['score_ms']:.0f}")
                put("s", key + ":select_ms", f"{p['select_ms']:.0f}")
                if p["gold_fn_recall"] is not None:
                    put("s", key + ":recall", f"{p['gold_fn_recall'] * 100:.1f}")
        put("s", f"scaling:{path.stem}:issues", d["timing_issues"])
    path = PROJECT_ROOT / "results" / "error_analysis" / "categories.json"
    if path.exists():
        d = json.loads(path.read_text())
        for method, counts in d["counts"].items():
            total = sum(counts.values())
            put("s", f"errors:{method}:total", total)
            for i, c in enumerate(d["categories"]):
                put("s", f"errors:{method}:{i}", counts[c])
                put("s", f"errors:{method}:{i}:pct", f"{100 * counts[c] / max(total, 1):.0f}")
    for path in sorted((PROJECT_ROOT / "results" / "downstream").glob("*/summary.json")):
        d = json.loads(path.read_text())
        tag = path.parent.name
        for cond, s in d["conditions"].items():
            put("s", f"down:{tag}:{cond}:n", s["n"])
            put("s", f"down:{tag}:{cond}:prompt", f"{s['mean_prompt_tokens']:,.0f}")
            for m in ("has_edit", "search_found", "edits_gold_function", "has_undefined_symbol"):
                put("s", f"down:{tag}:{cond}:{m}", f"{s[m]['mean'] * 100:.1f}")
                put("s", f"down:{tag}:{cond}:{m}:lo", f"{s[m]['lo'] * 100:.1f}")
                put("s", f"down:{tag}:{cond}:{m}:hi", f"{s[m]['hi'] * 100:.1f}")
    path = PROJECT_ROOT / "results" / "demo_reliability.json"
    if path.exists():
        d = json.loads(path.read_text())
        put("s", "demo:runs", d["runs"])
        put("s", "demo:ok", d["succeeded"])
        for k in ("server_ready_seconds", "first_call_seconds", "repeat_call_seconds"):
            for q, v in d.get(k, {}).items():
                put("s", f"demo:{k}:{q}", f"{v:.2f}")


def main() -> None:
    summaries()
    stats()
    header = [
        "% Generated by experiments/make_report_numbers.py. Do not edit.",
        r"\newcommand{\res}[1]{\ifcsname n:#1\endcsname\csname n:#1\endcsname\else\textbf{??}\fi}",
        r"\newcommand{\dif}[1]{\ifcsname d:#1\endcsname\csname d:#1\endcsname\else\textbf{??}\fi}",
        r"\newcommand{\stat}[1]{\ifcsname s:#1\endcsname\csname s:#1\endcsname\else\textbf{??}\fi}",
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(header + lines) + "\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} with {len(lines)} values")


if __name__ == "__main__":
    sys.exit(main())
