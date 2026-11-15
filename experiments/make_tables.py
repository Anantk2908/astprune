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
    esc = lambda t: t.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("±", r"$\pm$")  # noqa: E731
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


def full_split(rows: dict) -> set[str]:
    """Methods that ran on (nearly) every issue. Runs on a sample go in the external table instead."""
    ns = {m: int(r["n"]) for (m, b, k), r in rows.items() if k == "gold_fn_recall" and b == "r0.25" and r["n"]}
    top = max(ns.values(), default=0)
    return {m for m, n in ns.items() if n >= 0.9 * top}


def main_table(split: str, setting: str, methods: list[str] | None) -> None:
    rows = index(load(split, setting))
    full = full_split(rows)
    present = sorted({m for m, _, _ in rows if m in LABELS and m in full and (methods is None or m in methods)},
                     key=sort_key)
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
    sampled = {m for t in tables.values() for m, _, _ in t} - set().union(*(full_split(t) for t in tables.values()))
    present = sorted({m for t in tables.values() for m, _, _ in t
                      if m in LABELS and m not in sampled and (methods is None or m in methods)}, key=sort_key)
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
    variants = sorted({m for m, _, _ in rows if m.startswith(("ast_trained", "ast_bm25", "ast_hybrid")) and "@" not in m},
                      key=sort_key)  # other training runs (name@run) have their own table
    if len(variants) < 3:
        return
    header = ["Selector", "Recall @10%", "Recall @25%", "Recall @50%", "Sig. recall @10%", "Sig. recall @25%", "Line recall @25%"]
    body = [[label(m)] + [cell(rows, m, b, "gold_fn_recall") for b in ("r0.10", "r0.25", "r0.50")]
            + [cell(rows, m, "r0.10", "sig_recall"), cell(rows, m, "r0.25", "sig_recall"),
               cell(rows, m, "r0.25", "gold_line_recall")] for m in variants]
    write(f"{split}_{setting}_ablation", header, body,
          f"Selector ablations (%), {split}, {SETTING_LABELS[setting].lower()}.")


EXTERNAL = ["swe_pruner", "longcodezip_rank", "longcodezip", "ast_ppl"]


def external_table(split: str) -> None:
    """Third-party baselines next to our ranker, on exactly the issues each baseline ran on."""
    from astprune.evaluation import load_runs
    from astprune.registry import load_config
    from astprune.stats import paired_diff

    boot = load_config(None)["bootstrap"]
    kw = dict(n_resamples=boot["resamples"], seed=boot["seed"], level=boot["level"])
    header = ["Method", "Files", "Issues", "Recall @10%", "Recall @25%", "Recall @50%",
              "Ours minus them @25% (pp)", "Line recall @25%", "Parses @25%", "Seconds per file"]
    body = []
    for setting in ("oracle", "oracle_imports"):
        runs = load_runs(split, setting)
        if "ast_trained" not in runs:
            continue
        ours = {(r["budget"], r["instance_id"]): r for r in runs["ast_trained"]}
        for m in EXTERNAL:
            if m not in runs:
                continue
            theirs = {(r["budget"], r["instance_id"]): r for r in runs[m]}

            def pair(budget, metric):
                ids = sorted(i for b, i in theirs if b == budget and (budget, i) in ours
                             and theirs[(b, i)][metric] is not None and ours[(b, i)][metric] is not None
                             and (metric != "gold_fn_recall" or theirs[(b, i)]["n_gold_functions"] > 0))
                return ([theirs[(budget, i)][metric] for i in ids], [ours[(budget, i)][metric] for i in ids])

            row = [label(m), SETTING_LABELS[setting]]
            a, _ = pair("r0.25", "gold_fn_recall")
            row.append(str(len(a)))
            for budget in ("r0.10", "r0.25", "r0.50"):
                a, b = pair(budget, "gold_fn_recall")
                row.append(f"{100 * sum(a) / len(a):.1f} / {100 * sum(b) / len(b):.1f}" if a else "n/a")
            a, b = pair("r0.25", "gold_fn_recall")
            d = paired_diff(b, a, **kw)
            row.append(f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]")
            a, b = pair("r0.25", "gold_line_recall")
            row.append(f"{100 * sum(a) / len(a):.1f} / {100 * sum(b) / len(b):.1f}" if a else "n/a")
            rows25 = [r for (bud, _), r in theirs.items() if bud == "r0.25"]
            checked = sum(r["parse_checked"] for r in rows25)
            row.append(f"{100 * sum(r['parse_ok'] for r in rows25) / checked:.1f}" if checked else "n/a")
            row.append(f"{sum((r['prep_ms'] + r['prune_ms']) / r['n_files'] for r in rows25) / len(rows25) / 1000:.2f}")
            body.append(row)
        rows25 = [r for (bud, _), r in ours.items() if bud == "r0.25"]
        body.append([label("ast_trained"), SETTING_LABELS[setting], str(sum(r["n_gold_functions"] > 0 for r in rows25)),
                     "", "", "", "", "", "100.0" if all(r["parse_ok"] == r["parse_checked"] for r in rows25) else "",
                     f"{sum((r['prep_ms'] + r['prune_ms']) / r['n_files'] for r in rows25) / len(rows25) / 1000:.2f}"])
    if len(body) > 2:
        write(f"{split}_external", header, body,
              f"Third-party baselines on {split}, and LongCodeZip's ranking criterion on our nodes (AST + LLM perplexity), "
              "each next to our trained ranker on exactly the issues it ran on. "
              "Recall cells are theirs / ours, in percent. The difference is paired over the same issues, "
              "with a 95% bootstrap interval. Seconds per file include model scoring, on our laptop GPU.")


def downstream_table(split: str) -> None:
    """The downstream check (experiments/downstream.py), one table per model tag."""
    import json

    for path in sorted((PROJECT_ROOT / "results" / "downstream").glob("*/summary.json")):
        s = json.loads(path.read_text())
        if s.get("split") != split:
            continue
        rows = [json.loads(line) for line in (path.parent / "rows.jsonl").open()]
        fits = {r["instance_id"] for r in rows if r["condition"] == "full"}
        paired = {p["b"]: p for p in s["paired"] if p["metric"] == "edits_gold_function"}
        header = ["Context", "Issues", "Prompt tokens", "Edits that apply", "Edits a gold function",
                  "Ours minus this (pp)", "Calls an undefined name"]
        body = []
        for cond, c in s["conditions"].items():
            def ci(m):
                return f"{100 * c[m]['mean']:.1f} [{100 * c[m]['lo']:.1f}, {100 * c[m]['hi']:.1f}]"
            p = paired.get(cond)
            body.append(["Whole gold files, no pruning" if cond == "full" else label(cond), str(c["n"]),
                         f"{c['mean_prompt_tokens']:,.0f}", f"{100 * c['search_found']['mean']:.1f}",
                         ci("edits_gold_function"),
                         f"{100 * p['diff']:+.1f} [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}]" if p else "",
                         f"{100 * c['has_undefined_symbol']['mean']:.1f}"])
        matched = [r["prompt_tokens"] for r in rows if r["condition"] == "ast_trained" and r["instance_id"] in fits]
        note = (f" Whole files were only given when they were short enough, on {len(fits)} issues; there the "
                f"trained ranker's prompts average {sum(matched) / len(matched):,.0f} tokens.") if matched else ""
        model = s["model"].split("/")[-1] + (" in 4-bit" if s.get("four_bit") else "")
        write(f"{split}_downstream_{path.parent.name}", header, body,
              f"Downstream check on {split}: {model} reads the issue and one view of the gold files and writes "
              f"SEARCH/REPLACE edits. Budget {100 * s['budget']:.0f}% of the file tokens, at most "
              f"{s['max_budget_tokens']:,} tokens. Greedy decoding, no tests run. In percent of issues, except "
              "\"edits that apply\", the share of edits whose SEARCH text is found in the file. Intervals are 95% "
              "bootstrap intervals, and the difference is paired over the same issues." + note)


SEED_RUNS = [("main", "ast_trained", "Seed 0 (the main run)"), ("seed1", "ast_trained@seed1", "Seed 1"),
             ("seed2", "ast_trained@seed2", "Seed 2"),
             ("unfiltered", "ast_trained@unfiltered", "Seed 0, patches of any size")]


def seeds_table(split: str, setting: str = "oracle_imports") -> None:
    """Training variance: the main run, two more seeds, and a run without the patch-size filter."""
    import json
    import statistics

    if split == "dev":  # dev is already in the table as the selection split
        return

    dev, test = index(load("dev", setting)), index(load(split, setting))
    cols = [(dev, "r0.10"), (dev, "r0.25"), (test, "r0.10"), (test, "r0.25"), (test, "r0.50")]

    def val(rows, method, budget):
        r = rows.get((method, budget, "gold_fn_recall"))
        return 100 * float(r["mean"]) if r and r["mean"] != "" else None

    body, seeds = [], []
    for run, method, name in SEED_RUNS:
        info = PROJECT_ROOT / "results" / "training" / f"{run}.json"
        vals = [val(rows, method, b) for rows, b in cols]
        if not info.exists() or all(v is None for v in vals[2:]):  # not evaluated on this test split
            continue
        t = json.loads(info.read_text())
        body.append([name, f"{t['examples_used']:,}", f"{t['best']['step']:,}"]
                    + [f"{v:.1f}" if v is not None else "n/a" for v in vals])
        if run != "unfiltered":
            seeds.append(vals)
    if len(seeds) >= 2:
        row = [f"Mean ± sd of the {len(seeds)} seeds", "", ""]
        for i in range(len(cols)):
            xs = [v[i] for v in seeds if v[i] is not None]
            row.append(f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) >= 2 else "n/a")
        body.insert(len(seeds), row)
    if len(body) > 1:
        name = split.capitalize()
        write(f"{split}_seeds", ["Training run", "Examples", "Best step", "Dev @10%", "Dev @25%",
                                 f"{name} @10%", f"{name} @25%", f"{name} @50%"], body,
              f"Training variance. Gold-function recall (%) of the trained ranker, gold files + direct imports, "
              f"on dev (where the checkpoint is chosen) and on {split}. Same data and settings in every run except "
              "the seed, and in the last row the filter that drops training patches touching more than 5 functions.")


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
    external_table(args.split)
    downstream_table(args.split)
    seeds_table(args.split)


if __name__ == "__main__":
    sys.exit(main())
