import numpy as np
import pytest

from astprune.baselines.chunk_rag import ChunkRAG, make_chunks
from astprune.baselines.line_window import LineWindow
from astprune.baselines.lines import prefix_fill
from astprune.metrics import evaluate_instance, original_tokens, output_tokens, parses
from astprune.parse import parse_source
from astprune.pruners import ASTPruner, SkeletonPruner
from astprune.scorers.bm25 import BM25NodeScorer
from astprune.scorers.simple import HybridScorer, RandomScorer
from astprune.stats import mean_ci, paired_diff

ISSUE = "Shape.area returns 0 for every shape, it should use the sides"


def gold_for(pf, name):
    node = next(n for n in pf.nodes if n.name == name)
    return [{"path": pf.path, "name": name, "kind": node.kind, "start": node.start,
             "end": node.end, "sig_end": node.sig_end, "tokens": node.tokens_full}]


def all_methods():
    return [
        LineWindow(), ChunkRAG(chunk_tokens=40), SkeletonPruner(),
        ASTPruner(BM25NodeScorer()), ASTPruner(RandomScorer()),
        ASTPruner(BM25NodeScorer(), name="ast_bm25_nosig", sig_tier=False),
        ASTPruner(BM25NodeScorer(), name="ast_bm25_perfile", per_file=True),
        ASTPruner(HybridScorer([BM25NodeScorer(), RandomScorer()]), name="ast_hybrid"),
    ]


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 1.0])
def test_every_method_respects_the_budget(sample_source, fraction):
    files = [parse_source(sample_source, "a.py"), parse_source(sample_source.replace("Shape", "Form"), "b.py")]
    budget = int(original_tokens(files) * fraction)
    for method in all_methods():
        ctx = method.prune(method.prepare(ISSUE, files), files, budget)
        assert output_tokens(ctx) <= budget, method.name
        for pf in files:  # kept lines are real line numbers of that file
            assert ctx[pf.path].kept_lines <= set(range(1, len(pf.lines) + 1)), method.name


def test_full_budget_keeps_everything(sample_source):
    files = [parse_source(sample_source, "a.py")]
    budget = original_tokens(files) + 50
    for method in all_methods():
        ctx = method.prune(method.prepare(ISSUE, files), files, budget)
        row = evaluate_instance(files, gold_for(files[0], "Shape.area"), {"a.py": [32]}, ctx, {"a.py"})
        assert row["gold_fn_recall"] == 1.0 and row["gold_line_recall"] == 1.0, method.name


def test_ast_bm25_finds_the_named_method_at_a_tight_budget(sample_source):
    files = [parse_source(sample_source, "a.py")]
    method = ASTPruner(BM25NodeScorer())
    budget = int(original_tokens(files) * 0.3)
    ctx = method.prune(method.prepare(ISSUE, files), files, budget)
    row = evaluate_instance(files, gold_for(files[0], "Shape.area"), {}, ctx, {"a.py"})
    assert row["gold_fn_recall"] == 1.0
    assert row["sig_recall"] == 1.0
    assert row["parse_ok"] == row["parse_checked"] == 1
    assert 0 < row["gold_token_precision"] <= 1
    assert row["compression"] <= 0.3


def test_line_window_keeps_a_prefix(sample_source):
    files = [parse_source(sample_source, "a.py")]
    method = LineWindow()
    ctx = method.prune(method.prepare(ISSUE, files), files, int(original_tokens(files) * 0.3))
    kept = sorted(ctx["a.py"].kept_lines)
    assert kept == list(range(1, len(kept) + 1))


def test_chunks_cover_the_file_and_ignore_structure(sample_source):
    pf = parse_source(sample_source, "a.py")
    chunks = make_chunks(pf, chunk_tokens=40)
    covered = [i for lo, hi in chunks for i in range(lo, hi + 1)]
    assert covered == list(range(1, len(pf.lines) + 1))
    assert len(chunks) > 3


def test_partial_function_does_not_count_as_kept(sample_source):
    pf = parse_source(sample_source, "a.py")
    node = next(n for n in pf.nodes if n.name == "top_level")
    order = [("a.py", i) for i in range(node.start, node.end)]  # all but the last line
    order = [x for x in order if pf.lines[x[1] - 1].strip()][:-1]
    ctx = prefix_fill([pf], order, 10_000)
    row = evaluate_instance([pf], gold_for(pf, "top_level"), {}, ctx, {"a.py"})
    assert row["gold_fn_recall"] == 0.0
    assert row["sig_recall"] == 1.0


def test_gold_function_in_a_missing_file_counts_as_lost(sample_source):
    pf = parse_source(sample_source, "a.py")
    gold = [{"path": "elsewhere.py", "name": "f", "kind": "function", "start": 1, "end": 3, "sig_end": 1, "tokens": 9}]
    method = SkeletonPruner()
    ctx = method.prune(None, [pf], 10_000)
    row = evaluate_instance([pf], gold, {"elsewhere.py": [2]}, ctx, {"a.py"})
    assert row["gold_fn_recall"] == 0.0 and row["gold_line_recall"] == 0.0 and row["all_gold"] == 0.0


def test_issue_without_gold_function_has_no_function_metrics(sample_source):
    pf = parse_source(sample_source, "a.py")
    ctx = SkeletonPruner().prune(None, [pf], 10_000)
    row = evaluate_instance([pf], [], {"a.py": [5]}, ctx, {"a.py"})
    assert row["gold_fn_recall"] is None and row["gold_line_recall"] == 1.0


def test_parses_helper():
    assert parses("x = 1\n") and not parses("def broken(:\n")


def test_bootstrap_intervals():
    ci = mean_ci([1.0] * 50 + [0.0] * 50)
    assert ci["mean"] == 0.5 and 0.35 < ci["lo"] < 0.5 < ci["hi"] < 0.65 and ci["n"] == 100
    rng = np.random.default_rng(0)
    a = rng.random(200)
    same = paired_diff(a, a)
    assert same["diff"] == 0.0 and same["p"] == 1.0
    better = paired_diff(a + 0.2, a + rng.normal(0, 0.05, 200))
    assert better["lo"] > 0 and better["p"] < 0.01
    assert mean_ci([])["mean"] is None
    with pytest.raises(ValueError):
        paired_diff([1, 2], [1])


def test_oracle_candidates_do_not_list_gold_files_first():
    from astprune.evaluation import candidate_paths

    rec = {"oracle_files": ["pkg/zeta.py"], "import_files": ["pkg/alpha.py", "pkg/mid.py"],
           "bm25_files": ["pkg/mid.py", "pkg/zeta.py", "pkg/alpha.py"]}
    assert candidate_paths(rec, "oracle_imports") == ["pkg/alpha.py", "pkg/mid.py", "pkg/zeta.py"]
    assert candidate_paths(rec, "oracle") == ["pkg/zeta.py"]
    assert candidate_paths(rec, "bm25_5") == rec["bm25_files"]  # retrieval order is kept
