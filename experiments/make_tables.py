"""Write the tables used in the README, report and slides from the summary CSVs.

    python experiments/summarize.py --split lite
    python experiments/make_tables.py --split lite

Output in results/tables/: one Markdown and one LaTeX file per table. The
report includes the LaTeX files directly, so no number is typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from astprune.data.repos import PROJECT_ROOT
from astprune.display import LABELS, SETTING_LABELS, label, sort_key

SUMMARY = PROJECT_ROOT / "results" / "summary"
OUT = PROJECT_ROOT / "results" / "tables"


def load(split: str, setting: str, paired: bool = False) -> list[dict]:
    path = SUMMARY / f"{split}_{setting}{'_paired' if paired else ''}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def cell(rows: dict, method: str, budget: str, metric: str, ci: bool = False) -> str:
    r = rows.get((method, budget, metric))
    if r is None or r["mean"] == "":
        return "n/a"
    mean = float(r["mean"]) * 100
    if ci and r["lo"] != "":
        return f"{mean:.1f} [{float(r['lo']) * 100:.1f}, {float(r['hi']) * 100:.1f}]"
    return f"{mean:.1f}"


def write(name: str, header: list[str], body: list[list[str]], caption: str, bold_row: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in body]
    (OUT / f"{name}.md").write_text(f"{caption}\n\n" + "\n".join(md) + "\n")
    esc = lambda t: t.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")  # noqa: E731
    tex = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    for i, row in enumerate(body):
        cells = [esc(c) for c in row]
        if i == bold_row:
            cells = [r"\textbf{" + c + "}" for c in cells]
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote results/tables/{name}.md and .tex ({len(body)} rows)")


def index(rows: list[dict]) -> dict:
    return {(r["method"], r["budget"], r["metric"]): r for r in rows}


def main_table(split: str, setting: str, methods: list[str] | None) -> None:
    rows = index(load(split, setting))
    present = sorted({m for m, _, _ in rows if m in LABELS and (methods is None or m in methods)}, key=sort_key)
    if not present:
        return
    n = max(int(r["n"]) for (m, b, k), r in rows.items() if k == "gold_fn_recall" and b == "r0.25")
    header = ["Method", "Recall @10%", "Recall @25%", "Recall @50%", "Line recall @25%",
              "All-gold @25%", "Sig. recall @25%", "Precision @25%", "Parses @25%"]
    body = []
    for m in present:
        body.append([label(m)] + [cell(rows, m, b, "gold_fn_recall", ci=True) for b in ("r0.10", "r0.25", "r0.50")]
                    + [cell(rows, m, "r0.25", k) for k in ("gold_line_recall", "all_gold", "sig_recall",
                                                           "gold_token_precision", "parse_rate")])
    bold = present.index("ast_trained") if "ast_trained" in present else None
    write(f"{split}_{setting}_main", header, body,
          f"Gold-function recall with 95% bootstrap intervals and other metrics, in percent. "
          f"{split}, {SETTING_LABELS[setting].lower()}, {n} issues with a gold function.", bold)


def settings_table(split: str, methods: list[str] | None) -> None:
    settings = [s for s in SETTING_LABELS if load(split, s)]
    tables = {s: index(load(split, s)) for s in settings}
    present = sorted({m for t in tables.values() for m, _, _ in t if m in LABELS and (methods is None or m in methods)},
                     key=sort_key)
    if not present:
        return
    header = ["Method"] + [SETTING_LABELS[s] for s in settings]
    body = [[label(m)] + [cell(tables[s], m, "r0.25", "gold_fn_recall") for s in settings] for m in present]
    write(f"{split}_settings", header, body,
          f"Gold-function recall (%) at a 25% budget under each candidate-file setting, {split}.",
          present.index("ast_trained") if "ast_trained" in present else None)


def paired_table(split: str, setting: str) -> None:
    rows = [r for r in load(split, setting, paired=True) if r["metric"] == "gold_fn_recall" and r["diff"] != ""]
    if not rows:
        return
    header = ["Question", "A", "B", "Budget", "A minus B (pp)", "95% interval", "p", "Issues"]
    body = []
    for r in rows:
        budget = f"{int(float(r['budget'][1:]) * 100)}%" if r["budget"].startswith("r") else f"{r['budget'][1:]} tok"
        body.append([r["family"].split("_")[0], label(r["a"]), label(r["b"]), budget,
                     f"{float(r['diff']) * 100:+.1f}", f"[{float(r['lo']) * 100:+.1f}, {float(r['hi']) * 100:+.1f}]",
                     f"{float(r['p']):.3f}" if float(r["p"]) >= 0.001 else "<0.001", r["n"]])
    write(f"{split}_{setting}_paired", header, body,
          f"Paired bootstrap differences in gold-function recall, {split}, {SETTING_LABELS[setting].lower()}.")


def ablation_table(split: str, setting: str) -> None:
    rows = index(load(split, setting))
    variants = sorted({m for m, _, _ in rows if m.startswith(("ast_trained", "ast_bm25", "ast_hybrid"))}, key=sort_key)
    if len(variants) < 3:
        return
    header = ["Selector", "Recall @10%", "Recall @25%", "Recall @50%", "Sig. recall @10%", "Sig. recall @25%", "Line recall @25%"]
    body = [[label(m)] + [cell(rows, m, b, "gold_fn_recall") for b in ("r0.10", "r0.25", "r0.50")]
            + [cell(rows, m, "r0.10", "sig_recall"), cell(rows, m, "r0.25", "sig_recall"),
               cell(rows, m, "r0.25", "gold_line_recall")] for m in variants]
    write(f"{split}_{setting}_ablation", header, body,
          f"Selector ablations (%), {split}, {SETTING_LABELS[setting].lower()}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--methods", nargs="+", default=None, help="limit the main tables to these methods")
    args = ap.parse_args()
    for setting in ("oracle_imports", "oracle", "bm25_5"):
        main_table(args.split, setting, args.methods)
    settings_table(args.split, args.methods)
    paired_table(args.split, "oracle_imports")
    paired_table(args.split, "bm25_5")
    ablation_table(args.split, "oracle_imports")


if __name__ == "__main__":
    sys.exit(main())
