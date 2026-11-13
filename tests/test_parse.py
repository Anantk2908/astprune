from astprune.parse import MAX_BLOCK_LINES, parse_source


def by_name(pf):
    return {n.name: n for n in pf.nodes}


def test_every_line_belongs_to_exactly_one_node(sample_source):
    pf = parse_source(sample_source, "sample.py")
    covered = []
    for node in pf.nodes:
        covered.extend(range(node.start, node.end + 1))
    assert covered == list(range(1, len(pf.lines) + 1))


def test_node_kinds_and_names(sample_source):
    nodes = by_name(parse_source(sample_source, "sample.py"))
    assert nodes["top_level"].kind == "function"
    assert nodes["decorated"].kind == "function"
    assert nodes["Shape"].kind == "class"
    assert nodes["Shape.area"].kind == "method"
    assert nodes["Shape.name"].kind == "method"
    assert nodes["Shape.Meta"].kind == "class"
    assert nodes["Shape.Meta.describe"].kind == "method"
    assert nodes["fetch"].kind == "function"
    assert "inner" not in nodes and "decorated.inner" not in nodes  # nested defs stay inside


def test_decorators_and_multiline_signature(sample_source):
    pf = parse_source(sample_source, "sample.py")
    node = by_name(pf)["decorated"]
    assert pf.lines[node.start - 1].strip() == "@decorator"
    assert pf.lines[node.sig_end - 1].strip() == ") -> int:"
    assert pf.signature_text(node).splitlines()[-1] == "    ..."


def test_parents_point_at_enclosing_classes(sample_source):
    pf = parse_source(sample_source, "sample.py")
    nodes = by_name(pf)
    assert nodes["Shape.area"].parents == (nodes["Shape"].idx,)
    assert nodes["Shape.Meta.describe"].parents == (nodes["Shape"].idx, nodes["Shape.Meta"].idx)
    assert nodes["top_level"].parents == ()


def test_class_attribute_between_methods_is_a_block(sample_source):
    pf = parse_source(sample_source, "sample.py")
    blocks = [n for n in pf.nodes if n.kind == "block" and n.parents]
    assert any('label = "shape"' in pf.text(n) for n in blocks)


def test_one_liners_have_no_signature_tier(sample_source):
    nodes = by_name(parse_source(sample_source, "sample.py"))
    assert not nodes["one_liner"].has_signature
    assert not nodes["Empty"].has_signature
    assert nodes["top_level"].has_signature


def test_docstring_and_token_counts(sample_source):
    nodes = by_name(parse_source(sample_source, "sample.py"))
    assert nodes["top_level"].docstring == "Add two numbers."
    assert nodes["top_level"].tokens_full > nodes["top_level"].tokens_sig > 0


def test_syntax_error_does_not_raise_and_keeps_good_parts():
    src = "def good():\n    return 1\n\n\ndef broken(:\n    return 2\n\n\ndef also_good():\n    return 3\n"
    pf = parse_source(src, "bad.py")
    assert pf.had_errors
    assert pf.parser in ("tree-sitter-partial", "ast")
    names = {n.name for n in pf.nodes}
    assert "good" in names
    covered = sum(n.end - n.start + 1 for n in pf.nodes)
    assert covered == len(pf.lines)


def test_python2_source_falls_back_without_crashing():
    src = "def old():\n    print 'hello'\n    return 1\n"
    pf = parse_source(src, "py2.py")
    assert "old" in {n.name for n in pf.nodes}


def test_empty_and_comment_only_files():
    assert len(parse_source("", "empty.py").nodes) == 1
    pf = parse_source("# just a comment\n", "c.py")
    assert pf.nodes[0].kind == "block"


def test_long_constant_runs_are_split_at_statements():
    src = "\n".join(f"X{i} = {i}" for i in range(3 * MAX_BLOCK_LINES)) + "\n"
    pf = parse_source(src, "consts.py")
    assert len(pf.nodes) >= 3
    assert all(n.end - n.start + 1 <= MAX_BLOCK_LINES + 1 for n in pf.nodes)


def test_trailing_comment_never_splits_a_statement():
    body = ",\n".join(f"    'key{i}': {i}" for i in range(60))
    src = f"import os\n\nSETTINGS = {{\n{body}\n}}  # type: Dict[str, Any]\n\nOTHER = 1\n"
    pf = parse_source(src, "consts.py")
    owner = {n.idx for n in pf.nodes if n.start <= 3 <= n.end or n.start <= 64 <= n.end}
    assert len(owner) == 1  # the opening and closing brace are in the same node


def test_two_statements_on_one_line_stay_together():
    src = "\n".join(f"a{i} = {i}; b{i} = {i}" for i in range(100)) + "\n"
    pf = parse_source(src, "pairs.py")
    covered = [i for n in pf.nodes for i in range(n.start, n.end + 1)]
    assert covered == list(range(1, len(pf.lines) + 1))
