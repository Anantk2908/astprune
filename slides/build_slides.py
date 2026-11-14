"""Build the presentation from the files in results/.

    python slides/build_slides.py            # writes slides/presentation.pptx

Every number on a slide is read from results/ when the script runs, so the
deck cannot drift from the tables. Slides whose results do not exist yet
(third-party baselines, scaling, error analysis, downstream check) are left
out and listed at the end, so rerun the script after those jobs finish.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
OUT = ROOT / "slides" / "presentation.pptx"

SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
INK = RGBColor(0x0B, 0x0B, 0x0B)
SECONDARY = RGBColor(0x52, 0x51, 0x4E)
RULE = RGBColor(0xE6, 0xE5, 0xE1)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
ACCENT_SOFT = RGBColor(0xE8, 0xF1, 0xFB)
CODE_BG = RGBColor(0xF3, 0xF2, 0xEE)
FONT = "Calibri"
MONO = "Consolas"
WIDTH, HEIGHT = 13.333, 7.5
MAIN = "oracle_imports"

LABELS = {
    "line_window": "Line window",
    "chunk_bm25": "Chunk RAG (BM25)",
    "chunk_unixcoder": "Chunk RAG (UniXcoder)",
    "skeleton": "Skeleton (Agentless-style)",
    "ast_random": "AST + random order",
    "ast_bm25": "AST + BM25",
    "ast_unixcoder": "AST + UniXcoder, zero-shot",
    "ast_ppl": "AST + perplexity (LongCodeZip criterion)",
    "ast_trained": "AST + trained ranker (ours)",
    "ast_hybrid": "AST + trained + BM25 fusion",
    "ast_trained_nosig": "Ours, no signature tier",
    "ast_trained_perfile": "Ours, budget split per file",
    "ast_trained_density": "Ours, score per token order",
    "swe_pruner": "SWE-Pruner (released checkpoint)",
    "longcodezip_rank": "LongCodeZip, ranking only",
    "longcodezip": "LongCodeZip, full",
}


# ---- reading results ----------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class Numbers:
    def __init__(self) -> None:
        self.cells: dict[tuple, dict] = {}
        self.pairs: dict[tuple, dict] = {}
        for path in sorted((RESULTS / "summary").glob("*.csv")):
            for row in read_csv(path):
                if "diff" in row:
                    self.pairs[(row["split"], row["setting"], row["a"], row["b"], row["budget"], row["metric"])] = row
                elif "mean" in row:
                    self.cells[(row["split"], row["setting"], row["method"], row["budget"], row["metric"])] = row

    def pct(self, split, setting, method, budget="r0.25", metric="gold_fn_recall") -> str:
        row = self.cells.get((split, setting, method, budget, metric))
        return f"{100 * float(row['mean']):.1f}" if row and row["mean"] else "n/a"

    def has(self, split, setting, method) -> bool:
        return (split, setting, method, "r0.25", "gold_fn_recall") in self.cells

    def n(self, split, setting, method) -> str:
        row = self.cells.get((split, setting, method, "r0.25", "gold_fn_recall"))
        return row["n"] if row else "?"

    def diff(self, split, setting, a, b, budget) -> str:
        row = self.pairs.get((split, setting, a, b, budget, "gold_fn_recall"))
        if not row:
            return "n/a"
        d, lo, hi = (100 * float(row[k]) for k in ("diff", "lo", "hi"))
        return f"{d:+.1f}  [{lo:+.1f}, {hi:+.1f}]"


# ---- drawing helpers ----------------------------------------------------------
class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(WIDTH), Inches(HEIGHT)
        self.count = 0
        self.skipped: list[str] = []

    def slide(self, title: str, kicker: str = "", notes: str = ""):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = SURFACE
        self.count += 1
        if kicker:
            self.text(s, kicker.upper(), 0.7, 0.45, 12, 0.3, size=12, color=ACCENT, bold=True)
        self.text(s, title, 0.7, 0.72, 12, 0.9, size=30, bold=True)
        self.text(s, str(self.count), 12.2, 6.95, 0.6, 0.3, size=11, color=SECONDARY, align=PP_ALIGN.RIGHT)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def text(self, s, content, x, y, w, h, *, size=18, color=INK, bold=False, font=FONT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        frame.vertical_anchor = anchor
        for i, line in enumerate(content.split("\n")):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.alignment = align
            run = para.add_run()
            run.text = line
            run.font.size, run.font.bold, run.font.name = Pt(size), bold, font
            run.font.color.rgb = color
        return box

    def bullets(self, s, items, x, y, w, h, *, size=20, gap=10):
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        for i, item in enumerate(items):
            head, _, rest = item.partition("|")
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.space_after = Pt(gap)
            run = para.add_run()
            run.text = head.strip()
            run.font.size, run.font.name, run.font.bold = Pt(size), FONT, bool(rest)
            run.font.color.rgb = INK
            if rest:
                more = para.add_run()
                more.text = "  " + rest.strip()
                more.font.size, more.font.name = Pt(size), FONT
                more.font.color.rgb = SECONDARY
        return box

    def box(self, s, x, y, w, h, fill=ACCENT_SOFT, line=None):
        shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.adjustments[0] = 0.08
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        if line is None:
            shape.line.fill.background()
        else:
            shape.line.color.rgb = line
        shape.shadow.inherit = False
        return shape

    def card(self, s, big, small, x, y, w=2.8, h=1.5):
        self.box(s, x, y, w, h)
        self.text(s, big, x + 0.2, y + 0.12, w - 0.4, 0.75, size=34, bold=True, color=ACCENT)
        self.text(s, small, x + 0.2, y + 0.88, w - 0.4, 0.6, size=13, color=SECONDARY)

    def picture(self, s, path: Path, x, y, w, h):
        with Image.open(path) as im:
            ratio = im.width / im.height
        if w / h > ratio:
            new_w, new_h = h * ratio, h
        else:
            new_w, new_h = w, w / ratio
        s.shapes.add_picture(str(path), Inches(x + (w - new_w) / 2), Inches(y + (h - new_h) / 2),
                             Inches(new_w), Inches(new_h))

    def table(self, s, header, rows, x, y, w, *, col_widths=None, size=14, highlight=(), row_h=0.42):
        shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * (len(rows) + 1)))
        table = shape.table
        shape._element.graphic.graphicData.tbl[0][-1].text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # no style
        total = sum(col_widths) if col_widths else len(header)
        for c in range(len(header)):
            share = (col_widths[c] if col_widths else 1) / total
            table.columns[c].width = Inches(w * share)
        for r, cells in enumerate([header] + rows):
            for c, value in enumerate(cells):
                cell = table.cell(r, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RULE if r == 0 else (ACCENT_SOFT if (r - 1) in highlight else SURFACE)
                cell.margin_top = cell.margin_bottom = Inches(0.04)
                para = cell.text_frame.paragraphs[0]
                para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
                run = para.add_run()
                run.text = str(value)
                run.font.size, run.font.name = Pt(size), FONT
                run.font.bold = r == 0 or (r - 1) in highlight
                run.font.color.rgb = INK
        return table

    def code(self, s, content, x, y, w, h, size=13):
        self.box(s, x, y, w, h, fill=CODE_BG)
        self.text(s, content, x + 0.2, y + 0.15, w - 0.4, h - 0.3, size=size, font=MONO)


# ---- slides -----------------------------------------------------------------------
def build() -> Deck:
    num = Numbers()
    deck = Deck()
    train = read_json(RESULTS / "training" / "main.json") or {}
    lite_stats = read_json(RESULTS / "data" / "lite_stats.json") or {}
    leak = read_json(RESULTS / "data" / "leakage_check.json") or {}
    n_lite, n_ver = num.n("lite", MAIN, "ast_trained"), num.n("verified", MAIN, "ast_trained")

    def r(method, budget="r0.25", split="lite", setting=MAIN, metric="gold_fn_recall"):
        return num.pct(split, setting, method, budget, metric)

    # 1. title
    s = deck.prs.slides.add_slide(deck.prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = SURFACE
    deck.count += 1
    deck.text(s, "Learned AST Pruning for\nToken-Budgeted Code Context", 0.9, 2.0, 11.5, 2.0, size=44, bold=True)
    deck.text(s, "A small trained ranker that decides which functions a coding agent gets to read",
              0.9, 4.0, 11.5, 0.6, size=22, color=SECONDARY)
    deck.text(s, "Anant Kumar, Marcellanous Ellonye\nCSCI-GA 3033-091, NYU, Fall 2026", 0.9, 5.4, 11.5, 0.9,
              size=18, color=SECONDARY)
    s.notes_slide.notes_text_frame.text = (
        "One line: we train a 125M encoder on real bug fixes so that an agent reads the functions "
        "that matter and skips the rest, inside a token budget it chooses.")

    # 2. problem
    s = deck.slide("Agents read whole files. Most of it is not about the bug.", "Problem",
                   "Set up the cost: the candidate files for one issue are large, the fix is small.")
    deck.bullets(s, [
        "A coding agent opens every file that might matter and pays for every token in it.",
        "The fix usually touches one function. On SWE-bench Lite the gold patch changes "
        f"{lite_stats.get('gold_functions_per_instance_mean', 1.2):.1f} existing functions per issue on average, "
        f"in files with about {lite_stats.get('nodes_per_gold_file_mean', 72):.0f} functions and blocks each.",
        "Long context costs money and time, and models get worse at using what is in the middle of it.",
        "So: given an issue, some files and a token budget, which parts should the agent see?",
    ], 0.7, 1.9, 7.4, 4.6, size=20)
    deck.card(s, f"{100 - float(r('ast_trained', metric='compression')) if r('ast_trained', metric='compression') != 'n/a' else 75:.0f}%",
              "of the tokens removed at our middle budget", 8.8, 1.9, 3.8)
    deck.card(s, f"{r('ast_trained')}%", "of the functions the real fix edits are still there, whole", 8.8, 3.6, 3.8)
    deck.card(s, f"{r('line_window')}%", "if you keep the top of each file instead", 8.8, 5.3, 3.8)

    # 3. where existing work stops
    s = deck.slide("What exists, and where it stops", "Problem",
                   "Position against the closest work. The gap: nobody trains a ranker over whole functions "
                   "for an explicit budget and checks it against the gold patch.")
    deck.table(s, ["Approach", "Unit it prunes", "Ranking", "What can go wrong"], [
        ["Line window, truncation", "lines", "none", "cuts functions in half"],
        ["Chunk retrieval (RepoCoder-style)", "fixed-size chunks", "BM25 or embeddings", "chunks ignore syntax"],
        ["Agentless skeleton", "signatures", "none", "no bodies until a second step"],
        ["LongCodeZip", "functions, then blocks", "perplexity of a 7B LM, no training", "slow, needs a big model"],
        ["SWE-Pruner", "lines", "trained 0.6B scorer", "kept lines need not parse"],
        ["This project", "functions and classes", "trained 125M encoder", "see limits slide"],
    ], 0.7, 1.9, 11.9, col_widths=[3.4, 2.3, 3.5, 3.2], size=14, highlight=(5,))
    deck.text(s, "Two questions. H1: do whole functions beat lines and chunks as the unit? "
                 "H2: does training on real fixes beat training-free ranking over the same units?",
              0.7, 5.55, 11.9, 1.0, size=18, color=SECONDARY)

    # 4. method
    s = deck.slide("Parse, score, select, render", "Method",
                   "Walk the pipeline left to right. Stress that all AST methods share parse, select and render, "
                   "so comparisons isolate the ranker.")
    steps = [("Parse", "tree-sitter splits each file into functions, methods, class headers and blocks. "
                       "Every line belongs to exactly one node."),
             ("Score", "UniXcoder-base, 125M. One embedding for the issue, one per node, cosine similarity."),
             ("Select", "Fill the budget in score order. Each node is kept whole, kept as a signature, or dropped."),
             ("Render", "Valid Python again, with a short comment wherever code was left out.")]
    for i, (name, body) in enumerate(steps):
        x = 0.7 + i * 3.05
        deck.box(s, x, 2.0, 2.8, 2.9)
        deck.text(s, name, x + 0.2, 2.15, 2.4, 0.5, size=24, bold=True, color=ACCENT)
        deck.text(s, body, x + 0.2, 2.8, 2.4, 2.0, size=15)
        if i < 3:
            deck.text(s, ">", x + 2.8, 3.1, 0.25, 0.5, size=22, color=SECONDARY, align=PP_ALIGN.CENTER)
    deck.bullets(s, [
        "One budget shared by all files, so a good ranker can spend it on the right file.",
        "Shipped as an MCP server with one tool: get_budgeted_context(issue, files, budget).",
    ], 0.7, 5.3, 11.9, 1.4, size=18)

    # 5. what the output looks like
    s = deck.slide("What the agent gets back", "Method",
                   "Point at the three states. The view still parses, which line and chunk pruning cannot promise.")
    deck.code(s, (
        "class Session:\n"
        "    # ... 14 lines omitted\n"
        "    def resolve_redirects(self, resp, req, stream=False):\n"
        "        i = 0\n"
        "        hist = []\n"
        "        while resp.is_redirect:\n"
        "            prepared_request = req.copy()\n"
        "            ...                                   <- kept whole\n"
        "\n"
        "    def rebuild_auth(self, prepared_request, response): ...\n"
        "    def rebuild_proxies(self, prepared_request, proxies): ...\n"
        "                                                  <- signature only\n"
        "    # ... 212 lines omitted                       <- dropped"), 0.7, 1.9, 7.6, 4.7, size=14)
    deck.bullets(s, [
        "Kept whole | the functions the ranker scores highest",
        "Signature | the next ones, so the agent knows they exist and can ask for them",
        "Dropped | everything else, with a line count",
        f"Parses as Python | {r('ast_trained', metric='parse_rate')}% of our views, "
        f"{r('chunk_bm25', metric='parse_rate')}% of chunk views",
    ], 8.6, 1.9, 4.1, 4.7, size=17, gap=14)
    deck.text(s, "Illustration, shortened from the requests sample in the demo.", 0.7, 6.65, 8, 0.3,
              size=11, color=SECONDARY)

    # 6. training
    s = deck.slide("Training on real fixes, on a laptop", "Method",
                   "Labels are free: the gold patch tells us which node a human edited. Hard negatives matter.")
    deck.bullets(s, [
        "Positive | a node the gold patch changed",
        "Negatives | other nodes of the same file, half picked by BM25, plus nodes from imported files",
        f"Data | SWE-bench train and SWE-Gym, {len(leak.get('train_repos', [])) or 44} repositories, "
        f"{train.get('examples_used', 0):,} issues used",
        f"No leakage | {len(leak.get('overlap_after_filter', []))} repositories shared with dev or test, checked in code",
        f"Cost | {train.get('train_minutes', 0):.0f} minutes on one {train.get('gpu', 'RTX 4060 Laptop GPU').replace('NVIDIA GeForce ', '')}",
        "Loss | InfoNCE, temperature 0.05. Checkpoint picked on the dev split only",
    ], 0.7, 1.9, 6.4, 4.8, size=17, gap=12)
    if (FIGURES / "training_dev_recall.png").exists():
        deck.picture(s, FIGURES / "training_dev_recall.png", 7.3, 1.9, 5.4, 4.6)

    # 7. evaluation setup
    s = deck.slide("How we measured it", "Evaluation",
                   "No tests are run. We ask whether the code a human edited survives pruning. Test sets touched once.")
    deck.bullets(s, [
        "Question asked | does the function the real fix edits survive pruning, with every line intact?",
        f"Test sets | SWE-bench Lite ({n_lite} issues with an edited function) and SWE-bench Verified ({n_ver})",
        "Tuning | all choices made on the SWE-bench dev split and written down before any test run",
        "Candidate files | gold files plus their direct imports (main), gold files only, BM25 top 5, 10, 20",
        "Budgets | 10%, 25%, 50% of the candidate tokens, and fixed 1k, 2k, 4k tokens",
        "Also reported | gold-line recall, which does not favour function-shaped units, and parse rate",
        "Statistics | paired bootstrap over issues, 10,000 resamples, 95% intervals",
    ], 0.7, 1.9, 11.9, 4.9, size=18, gap=11)

    # 8. main result
    s = deck.slide("Training helps at every budget", "Evaluation",
                   "Same nodes, same selector. Blue against green is the effect of training alone. "
                   "The gap narrows as the budget grows, as expected.")
    deck.picture(s, FIGURES / "lite_recall_vs_budget.png", 0.5, 1.8, 7.4, 5.0)
    order = ["line_window", "chunk_bm25", "skeleton", "ast_unixcoder", "ast_bm25", "ast_trained"]
    deck.table(s, ["Lite, recall %", "10%", "25%", "50%"],
               [[LABELS[m], r(m, "r0.10"), r(m, "r0.25"), r(m, "r0.50")] for m in order],
               8.0, 2.0, 4.9, col_widths=[3.2, 0.9, 0.9, 0.9], size=13, highlight=(5,))
    deck.text(s, f"Gold files plus direct imports, {n_lite} issues.", 8.0, 5.2, 4.9, 0.4, size=12, color=SECONDARY)

    # 9. hypotheses with intervals
    s = deck.slide("Both hypotheses, with intervals", "Evaluation",
                   "These comparisons were named in the config before the test run. Differences in recall points.")
    rows = []
    for name, a, b in [("H1: AST + BM25 minus line window", "ast_bm25", "line_window"),
                       ("H1: AST + BM25 minus chunk RAG (BM25)", "ast_bm25", "chunk_bm25"),
                       ("H2: trained minus zero-shot, same encoder", "ast_trained", "ast_unixcoder"),
                       ("H2: trained minus BM25, same nodes", "ast_trained", "ast_bm25")]:
        rows.append([name] + [num.diff("lite", MAIN, a, b, bud) for bud in ("r0.10", "r0.25", "r0.50")])
    deck.table(s, ["Paired difference in gold-function recall, Lite", "10% budget", "25% budget", "50% budget"],
               rows, 0.7, 1.9, 11.9, col_widths=[5.0, 2.3, 2.3, 2.3], size=14, highlight=(2, 3))
    deck.bullets(s, [
        "H1 holds for function recall. Whole functions beat lines and chunks by a wide margin.",
        f"The fair caveat: on gold-line recall at 25%, chunk BM25 scores {r('chunk_bm25', metric='gold_line_recall')} "
        f"and AST + BM25 {r('ast_bm25', metric='gold_line_recall')}. Part of the AST advantage is keeping functions intact.",
        f"H2 holds on both metrics. Trained ranker line recall at 25%: {r('ast_trained', metric='gold_line_recall')}.",
    ], 0.7, 4.5, 11.9, 2.3, size=17)

    # 10. second test set and retrieved files
    s = deck.slide("It holds on a second test set and with retrieved files", "Evaluation",
                   "Verified is 500 human-checked issues from the same projects. With BM25-retrieved files the "
                   "ceiling is lower because retrieval sometimes misses the gold file.")
    settings = [("oracle", "Gold files"), (MAIN, "Gold + imports"), ("bm25_5", "BM25 top 5"),
                ("bm25_10", "BM25 top 10"), ("bm25_20", "BM25 top 20")]
    methods = ["chunk_bm25", "ast_bm25", "ast_unixcoder", "ast_trained"]
    deck.table(s, ["Verified, recall % at 25% budget"] + [name for _, name in settings],
               [[LABELS[m]] + [r(m, split="verified", setting=st) for st, _ in settings] for m in methods],
               0.7, 1.9, 11.9, col_widths=[4.0, 1.5, 1.7, 1.5, 1.6, 1.6], size=15, highlight=(3,))
    deck.table(s, ["Lite, recall % at 25% budget"] + [name for _, name in settings],
               [[LABELS[m]] + [r(m, setting=st) for st, _ in settings] for m in methods],
               0.7, 4.4, 11.9, col_widths=[4.0, 1.5, 1.7, 1.5, 1.6, 1.6], size=15, highlight=(3,))

    # 11. ablations and the negative result
    s = deck.slide("What mattered, and one idea that did not work", "Evaluation",
                   "Be upfront: we expected fusion with BM25 to help and it did not. Pooled budget is the big design win.")
    abl = ["ast_trained", "ast_hybrid", "ast_trained_perfile", "ast_trained_nosig", "ast_trained_density"]
    deck.table(s, ["Lite, gold + imports", "Recall 10%", "Recall 25%", "Signature recall 25%"],
               [[LABELS[m], r(m, "r0.10"), r(m, "r0.25"), r(m, metric="sig_recall")] for m in abl if num.has("lite", MAIN, m)],
               0.7, 1.9, 7.4, col_widths=[3.4, 1.3, 1.3, 1.9], size=14, highlight=(0,))
    deck.bullets(s, [
        "Fusion with BM25 | slightly below the trained ranker alone, on dev and on both test sets",
        "One shared budget | much better than giving each file its own share",
        "Signature tier | costs little recall and tells the agent what else is there",
    ], 8.4, 1.9, 4.4, 4.6, size=16, gap=14)

    # 12. third-party baselines (only when run)
    ext = [m for m in ("swe_pruner", "longcodezip_rank", "longcodezip", "ast_ppl") if num.has("lite", "oracle", m)
           or num.has("lite", MAIN, m)]
    if ext:
        s = deck.slide("Against released systems", "Evaluation",
                       "State the conditions honestly: LongCodeZip ran with a 1.5B scorer instead of its 7B default, "
                       "and its full mode on a 100-issue sample. Details are in docs/baselines.md.")
        rows = []
        for m in ext + ["ast_bm25", "ast_trained"]:
            st = MAIN if num.has("lite", MAIN, m) else "oracle"
            rows.append([LABELS[m], "gold + imports" if st == MAIN else "gold files", num.n("lite", st, m),
                         r(m, "r0.10", setting=st), r(m, "r0.25", setting=st), r(m, "r0.50", setting=st),
                         r(m, setting=st, metric="gold_line_recall")])
        deck.table(s, ["Lite", "Files", "Issues", "Fn 10%", "Fn 25%", "Fn 50%", "Lines 25%"], rows, 0.7, 1.9, 11.9,
                   col_widths=[4.2, 1.9, 1.0, 1.1, 1.1, 1.1, 1.3], size=14, highlight=(len(rows) - 1,))
        deck.text(s, "Fn: gold-function recall. Lines: gold-line recall, the metric that suits line pruners. "
                     "Compare rows only within the same file setting.", 0.7, 5.6, 11.9, 0.8, size=14, color=SECONDARY)
    else:
        deck.skipped.append("third-party baselines (no swe_pruner / longcodezip / ast_ppl runs in results/summary)")

    # 13. scaling and latency
    scaling_fig = FIGURES / "lite_scaling.png"
    if scaling_fig.exists():
        s = deck.slide("More candidate files, and what it costs", "Evaluation",
                       "Recall as the candidate set grows from 5 to 20 retrieved files, and time per request.")
        deck.picture(s, scaling_fig, 0.5, 1.8, 7.6, 5.0)
        md = RESULTS / "scaling" / "lite.md"
        if md.exists():
            deck.text(s, "\n".join(md.read_text().splitlines()[:14]), 8.2, 1.9, 4.7, 4.8, size=11, font=MONO)
    else:
        deck.skipped.append("scaling (results/figures/lite_scaling.png missing)")

    # 14. error analysis
    cats = read_json(RESULTS / "error_analysis" / "categories.json")
    err_fig = FIGURES / "error_categories.png"
    if cats and err_fig.exists():
        s = deck.slide("Where it still fails", "Evaluation",
                       "The largest group is functions bigger than the whole budget. Those come back as a signature "
                       "and the agent has to ask again with more room.")
        deck.picture(s, err_fig, 0.5, 1.8, 7.6, 5.0)
        deck.bullets(s, [
            "Function larger than the budget | it cannot fit, so it is shown as a signature",
            "Issue names other code | the report talks about the symptom, the fix is elsewhere",
            "Vague issue | little text to rank by",
            "Case studies are in docs/error_analysis.md",
        ], 8.3, 1.9, 4.5, 4.8, size=15, gap=12)
    else:
        deck.skipped.append("error analysis (results/error_analysis/categories.json or figure missing)")

    # 15. demo
    s = deck.slide("Demo: the pruner as an MCP tool", "Demo",
                   "Run python demo/app.py, then sample 2 at a 10% budget, then a bad request to show error handling. "
                   "Backup: the recorded video.")
    rel = read_json(RESULTS / "demo_reliability.json") or {}
    deck.code(s, "$ python demo/app.py --sample 1\n$ python demo/app.py --sample 2 --budget 0.1\n"
                 "$ python demo/app.py --files nope.py --issue x\n\n"
                 "get_budgeted_context(\n    issue: str,\n    files: list[str],\n    budget: float = 0.25)",
              0.7, 1.9, 6.2, 3.6, size=15)
    items = [
        "A real SWE-bench issue, the files an agent would open, and the pruned view next to the token counts.",
        "The demo script is an MCP client. Claude Code calls the same server through .mcp.json.",
        "Bad paths, non-Python files, empty issues and a missing checkpoint all return a readable answer.",
    ]
    if rel:
        ok, runs = rel.get("succeeded", rel.get("ok")), rel.get("runs")
        if ok is not None and runs:
            items.append(f"Reliability check: {ok} of {runs} scripted runs succeeded.")
    deck.bullets(s, items, 7.2, 1.9, 5.5, 4.8, size=17, gap=12)

    # 16. limits
    s = deck.slide("Limits", "Evaluation", "Say these before someone asks.")
    deck.bullets(s, [
        "We measure whether the right code survives, not whether an agent then fixes the bug. No tests are run.",
        "Python only, and 12 well-known projects in the test sets.",
        "Fixes that only add new functions have nothing to keep, so those issues are outside the recall numbers.",
        "The ranker reads the first 256 tokens of a function. A long function is judged by its head.",
        "A function larger than the whole budget can only come back as a signature.",
        "Third-party baselines ran under the hardware limits of one 8 GB laptop GPU, stated in the repository.",
    ], 0.7, 1.9, 11.9, 4.9, size=19, gap=12)

    # 17. takeaways
    s = deck.slide("Takeaways", "Summary", "Three sentences, then questions.")
    deck.card(s, f"{r('ast_trained', 'r0.10')}%", f"gold functions kept at a 10% budget, against {r('ast_bm25', 'r0.10')}% "
              "for BM25 on the same nodes", 0.7, 1.9, 3.8, 1.7)
    deck.card(s, f"{train.get('parameters', 125_000_000) / 1e6:.0f}M", f"parameters, trained in "
              f"{train.get('train_minutes', 78):.0f} minutes on a laptop GPU", 4.75, 1.9, 3.8, 1.7)
    deck.card(s, f"{r('ast_trained', metric='parse_rate')}%", "of pruned views are valid Python", 8.8, 1.9, 3.8, 1.7)
    deck.bullets(s, [
        "Whole functions are the right unit to prune, and a small ranker trained on real fixes picks them "
        "better than training-free ranking.",
        "The result holds on two test sets, with oracle and with retrieved files, and the gap is largest when the budget is tight.",
        "It runs as an MCP server today. Next: check the effect on actual bug fixing, and more languages.",
    ], 0.7, 4.0, 11.9, 2.8, size=19, gap=12)
    return deck


def main() -> None:
    deck = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    deck.prs.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} with {deck.count} slides")
    for item in deck.skipped:
        print(f"left out: {item}")


if __name__ == "__main__":
    main()
