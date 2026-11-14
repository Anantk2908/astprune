"""Create and run notebooks/analysis.ipynb.

    python notebooks/build_notebook.py

The notebook only reads files under results/. It rebuilds the summary tables
and every figure, so it doubles as a check that the committed results are
enough to reproduce the analysis without a GPU or the datasets.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
cells: list = []


def md(text: str) -> None:
    cells.append(nbformat.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbformat.v4.new_code_cell(text.strip()))


md("""
# Analysis: learned AST pruning for token-budgeted code context

Everything here is computed from the per-issue result files in `results/runs/`.
Run all cells to rebuild the tables in `results/summary/` and `results/tables/` and the figures in `results/figures/`.
No GPU and no dataset download are needed.
""")
code("""
import json, subprocess, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
def run(*args):
    out = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    print(out.stdout[-1500:] or out.stderr[-1500:])
pd.set_option("display.width", 200)
""")
md("## 1. Rebuild summaries, tables and figures from the per-issue rows")
code("""
for split in ("dev", "lite", "verified"):
    run("experiments/summarize.py", "--split", split, "--quiet")
    run("experiments/make_tables.py", "--split", split)
run("experiments/make_figures.py", "--split", "lite")
run("experiments/make_figures.py", "--split", "verified")
""")
md("## 2. Main result: SWE-bench Lite, gold files plus direct imports")
code("""
display(Markdown((ROOT / "results/tables/lite_oracle_imports_main.md").read_text()))
display(Image(str(ROOT / "results/figures/lite_recall_vs_budget.png")))
""")
md("## 3. Is the difference real? Paired bootstrap over issues")
code("""
display(Markdown((ROOT / "results/tables/lite_oracle_imports_paired.md").read_text()))
display(Image(str(ROOT / "results/figures/lite_paired_differences.png")))
""")
md("## 4. The granularity-neutral view: gold-line recall")
code("""
display(Image(str(ROOT / "results/figures/lite_line_recall_vs_budget.png")))
""")
md("## 5. Second test set: SWE-bench Verified")
code("""
display(Markdown((ROOT / "results/tables/verified_oracle_imports_main.md").read_text()))
""")
md("## 6. Every candidate-file setting at a 25% budget")
code("""
for split in ("lite", "verified"):
    display(Markdown((ROOT / f"results/tables/{split}_settings.md").read_text()))
""")
md("## 7. Selector ablations")
code("""
display(Markdown((ROOT / "results/tables/lite_oracle_imports_ablation.md").read_text()))
""")
md("## 8. Per-repository breakdown (computed here from the raw rows)")
code("""
def per_repo(split, setting, methods, budget="r0.25"):
    frames = []
    for m in methods:
        path = ROOT / f"results/runs/{split}/{setting}/{m}.jsonl"
        if path.exists():
            frames.append(pd.read_json(path, lines=True).query("budget == @budget").assign(method=m))
    df = pd.concat(frames).dropna(subset=["gold_fn_recall"])
    table = df.pivot_table(index="repo", columns="method", values="gold_fn_recall", aggfunc="mean") * 100
    table["issues"] = df[df.method == methods[0]].groupby("repo").size()
    return table.round(1)

per_repo("lite", "oracle_imports", ["ast_trained", "ast_bm25", "ast_unixcoder", "chunk_bm25"])
""")
md("## 9. Training curve, scaling and error categories")
code("""
for name in ("training_dev_recall", "lite_scaling", "error_categories"):
    path = ROOT / f"results/figures/{name}.png"
    if path.exists():
        display(Image(str(path)))
    else:
        print("not built yet:", path.name)
""")

nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE.parent)}}).execute()
nbformat.write(nb, HERE / "analysis.ipynb")
print("wrote notebooks/analysis.ipynb")
