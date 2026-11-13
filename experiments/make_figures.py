"""Draw the figures from the summary tables.

    python experiments/summarize.py --split lite      # writes results/summary/*.csv
    python experiments/make_figures.py --split lite   # writes results/figures/*.png and *.pdf

Every figure is built only from files under results/, so anyone can redraw
them without rerunning an experiment. Each figure has a table twin: the CSV it
was drawn from, and the matching .md file in results/summary/.

Colours come from one fixed method -> colour map, so a method keeps its colour
in every figure. The palette is a published colour-blind-safe categorical set;
marker shapes repeat the identity for print and greyscale.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from astprune.data.repos import PROJECT_ROOT  # noqa: E402

SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#9a9992"

# method -> (label, colour, marker). Order here is legend order.
METHODS: dict[str, tuple[str, str, str]] = {
    "ast_trained": ("AST + trained ranker (ours)", "#2a78d6", "o"),
    "ast_bm25": ("AST + BM25", "#eb6834", "s"),
    "ast_unixcoder": ("AST + zero-shot UniXcoder", "#1baf7a", "^"),
    "chunk_bm25": ("Chunk RAG (BM25)", "#eda100", "D"),
    "longcodezip": ("LongCodeZip", "#e87ba4", "v"),
    "skeleton": ("Skeleton (Agentless-style)", "#008300", "P"),
    "line_window": ("Line window", "#6250d6", "X"),
    "swe_pruner": ("SWE-Pruner", "#e34948", "*"),
}
# Shown in grey when present: reference points, not competitors.
REFERENCE = {"ast_random": ("AST + random order", "h")}

SETTING_TITLES = {
    "oracle": "Gold files only",
    "oracle_imports": "Gold files + direct imports",
    "bm25_5": "BM25 top-5 files",
    "bm25_10": "BM25 top-10 files",
    "bm25_20": "BM25 top-20 files",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 7,
})


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def num(text: str | None) -> float | None:
    return float(text) if text not in (None, "") else None


def save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/figures/{name}.png")


def recall_vs_budget(split: str, settings: list[str], metric: str, ylabel: str, name: str,
                     methods: list[str] | None = None) -> None:
    """Small multiples, one panel per candidate setting, shared y axis."""
    tables = {s: read_csv(SUMMARY_DIR / f"{split}_{s}.csv") for s in settings}
    settings = [s for s in settings if tables[s]]
    if not settings:
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(3.6 * len(settings) + 0.4, 3.9), sharey=True)
    axes = [axes] if len(settings) == 1 else list(axes)
    wanted = methods or list(METHODS)
    drawn: dict[str, object] = {}
    for ax, setting in zip(axes, settings):
        rows = [r for r in tables[setting] if r["metric"] == metric and r["budget"].startswith("r")]
        present = {r["method"] for r in rows}
        series = [(m, *REFERENCE[m], NEUTRAL) for m in REFERENCE if m in present]
        series += [(m, METHODS[m][0], METHODS[m][2], METHODS[m][1]) for m in wanted if m in present]
        ends: list[tuple[float, str]] = []
        n_series = len(series)
        for k, (method, label, marker, colour) in enumerate(series):
            pts = sorted((float(r["budget"][1:]) * 100, num(r["mean"]), num(r["lo"]), num(r["hi"]))
                         for r in rows if r["method"] == method and num(r["mean"]) is not None)
            if not pts:
                continue
            # Small sideways offset so the intervals of different methods do not sit on top of each other.
            dx = (k - (n_series - 1) / 2) * 0.55
            xs = [p[0] + dx for p in pts]
            ys = [p[1] * 100 for p in pts]
            lo = [(p[1] - p[2]) * 100 for p in pts]
            hi = [(p[3] - p[1]) * 100 for p in pts]
            is_ours = method == "ast_trained"
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=colour, elinewidth=1.0, alpha=0.55, zorder=2)
            (line,) = ax.plot(xs, ys, color=colour, marker=marker, markeredgecolor=SURFACE, markeredgewidth=1.5,
                              linewidth=2.6 if is_ours else 1.8, zorder=4 if is_ours else 3, label=label)
            drawn.setdefault(label, line)
            ends.append((ys[-1], method))
        ax.set_title(SETTING_TITLES.get(setting, setting), loc="left")
        ax.set_xticks([10, 25, 50])
        ax.set_xticklabels(["10%", "25%", "50%"])
        ax.set_xlim(4, 56)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Token budget (share of candidate tokens)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(ylabel)
    fig.legend(drawn.values(), drawn.keys(), loc="lower center", ncol=min(4, len(drawn)),
               bbox_to_anchor=(0.5, -0.14 if len(drawn) > 4 else -0.08), fontsize=8.5)
    fig.suptitle(f"{ylabel} against budget ({split}, bars are 95% bootstrap intervals over issues)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, name)


def paired_forest(split: str, setting: str, name: str) -> None:
    """Paired differences with intervals: one row per comparison and budget."""
    rows = [r for r in read_csv(SUMMARY_DIR / f"{split}_{setting}_paired.csv")
            if r["metric"] == "gold_fn_recall" and r["budget"].startswith("r") and r["diff"] not in ("", None)]
    if not rows:
        return
    labels, diffs, los, his = [], [], [], []
    for r in rows:
        a = METHODS.get(r["a"], (r["a"],))[0].replace(" (ours)", "")
        b = METHODS.get(r["b"], (r["b"],))[0]
        labels.append(f"{a}  vs  {b}  @ {int(float(r['budget'][1:]) * 100)}%")
        diffs.append(float(r["diff"]) * 100)
        los.append(float(r["lo"]) * 100)
        his.append(float(r["hi"]) * 100)
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(rows) + 1.3))
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.0, zorder=1)
    for y, d, lo, hi in zip(ys, diffs, los, his):
        clear = lo > 0 or hi < 0
        colour = METHODS["ast_trained"][1] if clear else NEUTRAL
        ax.plot([lo, hi], [y, y], color=colour, linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.plot([d], [y], marker="o", color=colour, markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=8, zorder=3)
        ax.text(max(his) + 1.5, y, f"{d:+.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Difference in gold-function recall, percentage points (A minus B)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Paired differences, {SETTING_TITLES.get(setting, setting).lower()} ({split}).\n"
                 "Blue: the 95% interval excludes zero. Grey: it does not.", loc="left", fontsize=9.5)
    save(fig, name)


def scaling(split: str, name: str, methods: list[str]) -> None:
    """Recall and latency as the candidate set grows. Two panels, never two y axes on one."""
    path = PROJECT_ROOT / "results" / "scaling" / f"{split}.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for method in methods:
        if method not in data["methods"]:
            continue
        label, colour, marker = METHODS[method]
        pts = data["methods"][method]
        ks = [p["k"] for p in pts]
        style = dict(color=colour, marker=marker, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax1.plot(ks, [p["gold_fn_recall"] * 100 for p in pts], **style)
        ax2.plot(ks, [p["ms_per_issue"] for p in pts], **style)
    for ax in (ax1, ax2):
        ax.set_xticks([5, 10, 20])
        ax.set_xlabel("Candidate files (BM25 top-k)")
        ax.grid(axis="x", visible=False)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Gold-function recall (%)")
    ax1.set_title(f"Recall at {int(data['budget'] * 100)}% budget", loc="left")
    ax2.set_ylabel("Milliseconds per issue")
    ax2.set_ylim(bottom=0)
    ax2.set_title("Time to rank and prune", loc="left")
    ax1.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, name)


def training_curve(run: str, name: str) -> None:
    path = PROJECT_ROOT / "results" / "training" / f"{run}.json"
    if not path.exists():
        return
    history = json.loads(path.read_text())["dev_history"]
    colour = METHODS["ast_trained"][1]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    keys = [k for k in history[0] if k.startswith("recall_r")]
    for key, marker in zip(keys, ["o", "s", "^"]):
        ys = [h[key] * 100 for h in history]
        ax.plot([h["epoch"] for h in history], ys, color=colour, marker=marker, markeredgecolor=SURFACE,
                markeredgewidth=1.5, alpha=1.0 if key == keys[0] else 0.55)
        ax.annotate(f"{int(float(key.split('_r')[1]) * 100)}% budget", (history[-1]["epoch"], ys[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Training epochs (0 = zero-shot UniXcoder)")
    ax.set_ylabel("Dev gold-function recall (%)")
    ax.set_title("Dev recall during fine-tuning", loc="left")
    ax.set_xlim(right=history[-1]["epoch"] * 1.22 + 0.05)
    ax.grid(axis="x", visible=False)
    save(fig, name)


def error_categories(name: str) -> None:
    path = PROJECT_ROOT / "results" / "error_analysis" / "categories.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    systems = list(data["counts"])
    categories = data["categories"]
    fig, axes = plt.subplots(1, len(systems), figsize=(4.3 * len(systems), 0.34 * len(categories) + 1.4), sharey=True)
    axes = [axes] if len(systems) == 1 else list(axes)
    for ax, system in zip(axes, systems):
        counts = [data["counts"][system].get(c, 0) for c in categories]
        ys = list(range(len(categories)))[::-1]
        ax.barh(ys, counts, height=0.55, color=METHODS.get(system, ("", NEUTRAL))[1])
        for y, c in zip(ys, counts):
            ax.text(c + 0.2, y, str(c), va="center", fontsize=8.5, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels(categories, fontsize=8.5)
        ax.set_title(f"{METHODS.get(system, (system,))[0]} (n={sum(counts)})", loc="left", fontsize=9.5)
        ax.set_xlabel("Dropped gold functions")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--run", default="main", help="training run for the dev-recall curve")
    args = ap.parse_args()
    s = args.split
    recall_vs_budget(s, ["oracle", "oracle_imports", "bm25_5"], "gold_fn_recall",
                     "Gold-function recall (%)", f"{s}_recall_vs_budget")
    recall_vs_budget(s, ["oracle", "oracle_imports", "bm25_5"], "gold_line_recall",
                     "Gold-line recall (%)", f"{s}_line_recall_vs_budget")
    paired_forest(s, "oracle_imports", f"{s}_paired_differences")
    scaling(s, f"{s}_scaling", ["ast_trained", "ast_bm25", "ast_unixcoder", "chunk_bm25"])
    training_curve(args.run, "training_dev_recall")
    error_categories("error_categories")


if __name__ == "__main__":
    sys.exit(main())
