"""Turn per-issue rows into tables with confidence intervals.

    python experiments/summarize.py --split dev
    python experiments/summarize.py --split lite --settings oracle_imports

Writes, per setting:
  results/summary/<split>_<setting>.csv          mean and 95% CI per method, budget and metric
  results/summary/<split>_<setting>_paired.csv   paired differences for the comparisons in config.yaml
  results/summary/<split>_<setting>.md           the main table, readable
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

from astprune.data.repos import PROJECT_ROOT
from astprune.evaluation import SETTINGS, load_runs
from astprune.registry import load_config
from astprune.stats import mean_ci, paired_diff

SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"

# metric -> needs a gold function to be defined for the issue
ISSUE_METRICS = {
    "gold_fn_recall": True,
    "all_gold": True,
    "sig_recall": True,
    "gold_line_recall": False,
    "gold_token_precision": True,
    "compression": False,
}


def budget_sort_key(label: str) -> tuple[int, float]:
    return (0 if label.startswith("r") else 1, float(label[1:]))


def index_rows(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """budget -> instance_id -> row"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        out[row["budget"]][row["instance_id"]] = row
    return out


def summarise_setting(split: str, setting: str, config: dict) -> tuple[list[dict], list[dict]]:
    runs = {m: index_rows(rows) for m, rows in load_runs(split, setting).items()}
    boot = config["bootstrap"]
    kw = dict(n_resamples=boot["resamples"], seed=boot["seed"], level=boot["level"])
    table: list[dict] = []
    for method, by_budget in sorted(runs.items()):
        for budget in sorted(by_budget, key=budget_sort_key):
            rows = list(by_budget[budget].values())
            for metric, needs_gold in ISSUE_METRICS.items():
                values = [r[metric] for r in rows
                          if r[metric] is not None and (not needs_gold or r["n_gold_functions"] > 0)]
                ci = mean_ci(values, **kw)
                table.append({"split": split, "setting": setting, "method": method, "budget": budget,
                              "metric": metric, **ci})
            checked = sum(r["parse_checked"] for r in rows)
            extras = {
                "parse_rate": sum(r["parse_ok"] for r in rows) / checked if checked else None,
                "over_budget_rate": sum(r["over_budget"] for r in rows) / len(rows),
                "prep_ms_per_file": sum(r["prep_ms"] / r["n_files"] for r in rows) / len(rows),
                "prune_ms_per_file": sum(r["prune_ms"] / r["n_files"] for r in rows) / len(rows),
            }
            for metric, value in extras.items():
                table.append({"split": split, "setting": setting, "method": method, "budget": budget,
                              "metric": metric, "mean": value, "lo": None, "hi": None, "n": len(rows)})

    paired: list[dict] = []
    for family, pairs in config.get("comparisons", {}).items():
        for a, b in pairs:
            if a not in runs or b not in runs:
                continue
            for budget in sorted(set(runs[a]) & set(runs[b]), key=budget_sort_key):
                for metric in ("gold_fn_recall", "gold_line_recall"):
                    ids = sorted(
                        i for i in set(runs[a][budget]) & set(runs[b][budget])
                        if runs[a][budget][i][metric] is not None and runs[b][budget][i][metric] is not None
                        and (metric != "gold_fn_recall" or runs[a][budget][i]["n_gold_functions"] > 0)
                    )
                    res = paired_diff([runs[a][budget][i][metric] for i in ids],
                                      [runs[b][budget][i][metric] for i in ids], **kw)
                    paired.append({"split": split, "setting": setting, "family": family, "a": a, "b": b,
                                   "budget": budget, "metric": metric, **res})
    return table, paired


def write_csv(path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})


def fmt(cell: dict | None, pct: bool = True) -> str:
    if cell is None or cell["mean"] is None:
        return "n/a"
    scale = 100.0 if pct else 1.0
    if cell["lo"] is None:
        return f"{cell['mean'] * scale:.1f}"
    return f"{cell['mean'] * scale:.1f} [{cell['lo'] * scale:.1f}, {cell['hi'] * scale:.1f}]"


def markdown(split: str, setting: str, table: list[dict], paired: list[dict]) -> str:
    cells = {(r["method"], r["budget"], r["metric"]): r for r in table}
    methods = sorted({r["method"] for r in table})
    budgets = sorted({r["budget"] for r in table}, key=budget_sort_key)
    lines = [f"# {split} / {setting}", ""]
    for metric, title in [("gold_fn_recall", "Gold-function recall, % (95% CI)"),
                          ("gold_line_recall", "Gold-line recall, % (95% CI)"),
                          ("all_gold", "All-gold rate, %"),
                          ("sig_recall", "Signature recall, %"),
                          ("gold_token_precision", "Gold-token precision, %"),
                          ("compression", "Actual compression, % of original tokens"),
                          ("parse_rate", "Pruned views that parse, %")]:
        n = max((cells[(m, b, metric)]["n"] for m in methods for b in budgets if (m, b, metric) in cells), default=0)
        lines += [f"## {title}", "", f"Issues: up to {n}", "",
                  "| method | " + " | ".join(budgets) + " |", "|---|" + "---|" * len(budgets)]
        for m in methods:
            lines.append(f"| {m} | " + " | ".join(fmt(cells.get((m, b, metric))) for b in budgets) + " |")
        lines.append("")
    if paired:
        lines += ["## Paired differences (A minus B, percentage points, 95% CI)", "",
                  "| family | A | B | metric | budget | diff | p | n |", "|---|---|---|---|---|---|---|---|"]
        for r in paired:
            if r["diff"] is None:
                continue
            lines.append(
                f"| {r['family']} | {r['a']} | {r['b']} | {r['metric']} | {r['budget']} | "
                f"{r['diff'] * 100:+.1f} [{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}] | {r['p']:.4f} | {r['n']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--settings", nargs="+", default=None, choices=SETTINGS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    config = load_config(args.config)
    for setting in args.settings or config["settings"]:
        table, paired = summarise_setting(args.split, setting, config)
        if not table:
            continue
        write_csv(SUMMARY_DIR / f"{args.split}_{setting}.csv", table)
        write_csv(SUMMARY_DIR / f"{args.split}_{setting}_paired.csv", paired)
        text = markdown(args.split, setting, table, paired)
        (SUMMARY_DIR / f"{args.split}_{setting}.md").write_text(text + "\n")
        if not args.quiet:
            print(text)


if __name__ == "__main__":
    sys.exit(main())
