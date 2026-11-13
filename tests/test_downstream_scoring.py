"""The static checks used by experiments/downstream.py, tested without any model."""

import importlib.util
from pathlib import Path

from astprune.parse import parse_source

_spec = importlib.util.spec_from_file_location(
    "downstream", Path(__file__).resolve().parents[1] / "experiments" / "downstream.py")
downstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(downstream)


def answer(search: str, replace: str) -> str:
    return f"a.py\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def gold(pf, name):
    node = next(n for n in pf.nodes if n.name == name)
    return [{"path": pf.path, "name": name, "start": node.start, "end": node.end}]


def test_parse_edits_reads_every_block():
    text = "some chatter\n" + answer("x = 1", "x = 2") + "more\n" + answer("y = 1", "y = 2")
    assert downstream.parse_edits(text) == [("x = 1", "x = 2"), ("y = 1", "y = 2")]
    assert downstream.parse_edits("no edits here") == []


def test_locate_ignores_indentation_and_blank_lines(sample_source):
    pf = parse_source(sample_source, "a.py")
    assert downstream.locate("total = a + b\n\nreturn total", pf) == (10, 11)
    assert downstream.locate("this line is not in the file", pf) is None
    assert downstream.locate("   \n", pf) is None


def test_called_names_finds_bare_and_self_calls():
    calls, defined, ok = downstream.called_names(
        "    value = helper(x)\n    self.refresh()\n    other.method()\n    np.zeros(3)\n")
    assert ok and calls == {"helper", "refresh"} and "value" in defined
    assert downstream.called_names("def broken(:")[2] is False


def test_edit_in_gold_function_with_known_symbols(sample_source):
    pf = parse_source(sample_source, "a.py")
    row = downstream.score_answer(answer("        return 0", "        return self.sides * len(self.label)"),
                                  [pf], gold(pf, "Shape.area"), repo_names=set())
    assert row["n_edits"] == 1 and row["search_found"] == 1.0
    assert row["edits_gold_function"] == 1.0 and row["has_undefined_symbol"] == 0.0


def test_made_up_function_is_flagged(sample_source):
    pf = parse_source(sample_source, "a.py")
    row = downstream.score_answer(answer("        return 0", "        return compute_polygon_area(self)"),
                                  [pf], gold(pf, "Shape.area"), repo_names={"something_else"})
    assert row["undefined_symbols"] == ["compute_polygon_area"] and row["has_undefined_symbol"] == 1.0
    known = downstream.score_answer(answer("        return 0", "        return compute_polygon_area(self)"),
                                    [pf], gold(pf, "Shape.area"), repo_names={"compute_polygon_area"})
    assert known["has_undefined_symbol"] == 0.0


def test_edit_outside_gold_function_and_hallucinated_search(sample_source):
    pf = parse_source(sample_source, "a.py")
    elsewhere = downstream.score_answer(answer("    total = a + b", "    total = a - b"), [pf], gold(pf, "Shape.area"), set())
    assert elsewhere["search_found"] == 1.0 and elsewhere["edits_gold_function"] == 0.0
    invented = downstream.score_answer(answer("    area = width * height", "    area = 1"), [pf], gold(pf, "Shape.area"), set())
    assert invented["search_found"] == 0.0 and invented["edits_gold_function"] == 0.0
    nothing = downstream.score_answer("I cannot fix this.", [pf], gold(pf, "Shape.area"), set())
    assert nothing["has_edit"] == 0.0 and nothing["n_edits"] == 0
