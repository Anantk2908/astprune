import ast

import numpy as np
import pytest

from astprune.parse import parse_source
from astprune.select import DROP, FULL, SIG, render, render_file, rendered_tokens, select
from astprune.tokens import count_tokens


def scores_favouring(pf, *names, base=0.0):
    s = np.full(len(pf.nodes), base)
    for rank, name in enumerate(names):
        idx = next(n.idx for n in pf.nodes if n.name == name)
        s[idx] = 100.0 - rank
    return {pf.path: s}


def test_full_budget_round_trips_the_file(sample_source):
    pf = parse_source(sample_source, "sample.py")
    out = render_file(pf, [FULL] * len(pf.nodes))
    assert out.text == sample_source
    assert out.kept_lines == set(range(1, len(pf.lines) + 1))


def test_nothing_kept_renders_empty(sample_source):
    pf = parse_source(sample_source, "sample.py")
    out = render_file(pf, [DROP] * len(pf.nodes))
    assert out.text == "" and out.kept_lines == set()


@pytest.mark.parametrize("fraction", [0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("sig_tier", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_never_over_budget_and_always_parses(sample_source, fraction, sig_tier, seed):
    pf = parse_source(sample_source, "sample.py")
    rng = np.random.default_rng(seed)
    scores = {pf.path: rng.normal(size=len(pf.nodes))}
    budget = int(count_tokens(sample_source) * fraction)
    states = select([pf], scores, budget, sig_tier=sig_tier, sig_reserve=0.2 if sig_tier else 0.0)
    ctx = render([pf], states)
    assert rendered_tokens(ctx) <= budget
    ast.parse(ctx[pf.path].text)  # raises if the pruned view is not valid Python
    if not sig_tier:
        fn_states = {states[pf.path][n.idx] for n in pf.nodes if n.kind in ("function", "method")}
        assert SIG not in fn_states


def test_top_ranked_node_is_kept_whole(sample_source):
    pf = parse_source(sample_source, "sample.py")
    target = next(n for n in pf.nodes if n.name == "top_level")
    states = select([pf], scores_favouring(pf, "top_level"), target.tokens_full + 40)
    assert states[pf.path][target.idx] == FULL
    out = render_file(pf, states[pf.path])
    assert set(range(target.start, target.start + 4)) <= out.kept_lines
    assert "lines omitted" in out.text


def test_kept_method_brings_its_class_line(sample_source):
    pf = parse_source(sample_source, "sample.py")
    target = next(n for n in pf.nodes if n.name == "Shape.Meta.describe")
    states = select([pf], scores_favouring(pf, "Shape.Meta.describe"), target.tokens_full + 60)
    text = render_file(pf, states[pf.path]).text
    assert "class Shape:" in text and "class Meta:" in text and "def describe(self):" in text
    ast.parse(text)


def test_signature_only_class_gets_a_placeholder_body(sample_source):
    pf = parse_source(sample_source, "sample.py")
    states = [DROP] * len(pf.nodes)
    states[next(n.idx for n in pf.nodes if n.name == "Shape")] = SIG
    text = render_file(pf, states).text
    tree = ast.parse(text)
    assert isinstance(tree.body[0], ast.ClassDef)


def test_signature_tier_shows_def_line_and_ellipsis(sample_source):
    pf = parse_source(sample_source, "sample.py")
    states = [DROP] * len(pf.nodes)
    node = next(n for n in pf.nodes if n.name == "decorated")
    states[node.idx] = SIG
    out = render_file(pf, states)
    assert "@decorator" in out.text and ") -> int:" in out.text and "    ..." in out.text
    assert "return inner() + y" not in out.text
    assert out.kept_lines == set(range(node.start, node.sig_end + 1))


def test_sig_reserve_trades_full_bodies_for_signatures(sample_source):
    pf = parse_source(sample_source, "sample.py")
    rng = np.random.default_rng(0)
    scores = {pf.path: rng.normal(size=len(pf.nodes))}
    budget = int(count_tokens(sample_source) * 0.5)
    plain = select([pf], scores, budget, sig_reserve=0.0)[pf.path]
    reserved = select([pf], scores, budget, sig_reserve=0.5)[pf.path]
    assert reserved.count(SIG) >= plain.count(SIG)


def test_pooled_budget_goes_to_the_better_file(sample_source):
    a = parse_source(sample_source, "a.py")
    b = parse_source(sample_source, "b.py")
    scores = {"a.py": np.full(len(a.nodes), -5.0), "b.py": np.full(len(b.nodes), 5.0)}
    states = select([a, b], scores, count_tokens(sample_source) // 2)
    assert not any(states["a.py"]) and any(states["b.py"])


def test_density_order_prefers_small_nodes_when_scores_tie(sample_source):
    pf = parse_source(sample_source, "sample.py")
    scores = {pf.path: np.zeros(len(pf.nodes))}
    states = select([pf], scores, 60, order="density", sig_tier=False)[pf.path]
    kept = [pf.nodes[i].tokens_full for i, s in enumerate(states) if s == FULL]
    dropped = [pf.nodes[i].tokens_full for i, s in enumerate(states) if s == DROP]
    assert kept and max(kept) <= max(dropped)


def test_bad_arguments_are_rejected(sample_source):
    pf = parse_source(sample_source, "sample.py")
    scores = {pf.path: np.zeros(len(pf.nodes))}
    with pytest.raises(ValueError):
        select([pf], scores, -1)
    with pytest.raises(ValueError):
        select([pf], scores, 100, sig_reserve=1.5)
    with pytest.raises(ValueError):
        select([pf], {pf.path: np.zeros(3)}, 100)
    with pytest.raises(ValueError):
        select([pf], scores, 100, order="nope")


def test_budget_equal_to_the_file_returns_the_file(sample_source):
    pf = parse_source(sample_source, "sample.py")
    rng = np.random.default_rng(3)
    states = select([pf], {pf.path: rng.normal(size=len(pf.nodes))}, pf.n_tokens)
    assert render([pf], states)[pf.path].text == sample_source


def test_class_with_only_a_comment_kept_still_parses():
    src = (
        "class Field:\n"
        "    # The comment below sits between two methods.\n"
        "    def first(self):\n"
        "        return 1\n"
        "\n"
        "    # HACK: explain second\n"
        "\n"
        "    def second(self):\n"
        "        return 2\n"
    )
    pf = parse_source(src, "f.py")
    comment_blocks = [n for n in pf.nodes if n.kind == "block"]
    assert comment_blocks, "expected the comment between methods to be its own block"
    states = [DROP] * len(pf.nodes)
    states[comment_blocks[0].idx] = FULL
    text = render_file(pf, states).text
    ast.parse(text)
    assert "class Field:" in text


def test_real_world_shaped_file_always_parses_under_random_selection():
    body = ",\n".join(f"    'k{i}': {i}" for i in range(70))
    src = (
        "import os\n\n"
        f"DEFAULTS = {{\n{body}\n}}  # type: dict\n\n\n"
        "class A:\n    x = 1\n\n    # note\n\n    def f(self):\n        return 1\n\n"
        "    # trailing note\n\n\n"
        "class B(A):\n    def g(self):\n        return 2\n"
    )
    pf = parse_source(src, "w.py")
    total = count_tokens(src)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        for fraction in (0.05, 0.15, 0.3, 0.6):
            states = select([pf], {pf.path: rng.normal(size=len(pf.nodes))}, int(total * fraction),
                            sig_reserve=0.2 if seed % 2 else 0.0)
            ast.parse(render([pf], states)[pf.path].text)
