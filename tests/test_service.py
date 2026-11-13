import ast

import pytest

from astprune.service import PruningError, PruningService

ISSUE = "Shape.area returns 0 for every shape, it should use the sides"


@pytest.fixture
def workspace(tmp_path, sample_source):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "shapes.py").write_text(sample_source)
    (tmp_path / "pkg" / "other.py").write_text(sample_source.replace("Shape", "Form").replace("area", "size"))
    (tmp_path / "broken.py").write_text("def good():\n    return 1\n\n\ndef broken(:\n    return 2\n")
    (tmp_path / "notes.md").write_text("# Notes\n" + "some text here\n" * 50)
    return tmp_path


@pytest.fixture
def service(workspace):
    return PruningService(root=workspace, scorer="bm25")


def test_prunes_to_the_budget_and_keeps_the_named_method(service):
    out = service.get_budgeted_context(ISSUE, ["pkg/shapes.py", "pkg/other.py"], 0.3)
    assert out["kept_tokens"] <= out["budget_tokens"] < out["original_tokens"]
    assert out["ranking"] == "bm25" and out["warnings"] == []
    shapes = next(f for f in out["files"] if f["path"] == "pkg/shapes.py")
    assert "Shape.area" in shapes["kept_full"]
    assert "### pkg/shapes.py" in out["context"] and "def area(self):" in out["context"]
    assert 0 < out["saved_percent"] <= 100 and out["latency_ms"] > 0


def test_each_python_section_is_valid_python(service):
    out = service.get_budgeted_context(ISSUE, ["pkg/shapes.py", "pkg/other.py"], 0.4)
    for section in out["context"].split("### ")[1:]:
        _, _, code = section.partition("\n")
        ast.parse(code)


def test_budget_above_one_is_a_token_count(service):
    out = service.get_budgeted_context(ISSUE, ["pkg/shapes.py"], 120)
    assert out["budget_tokens"] == 120 and out["kept_tokens"] <= 120


def test_missing_file_is_reported_not_fatal(service):
    out = service.get_budgeted_context(ISSUE, ["pkg/shapes.py", "pkg/nope.py"], 0.5)
    assert len(out["files"]) == 1
    assert any("pkg/nope.py" in w and "not found" in w for w in out["warnings"])


def test_path_outside_the_workspace_is_refused(service):
    out = service.get_budgeted_context(ISSUE, ["pkg/shapes.py", "../../etc/passwd", "/etc/hostname"], 0.5)
    assert [f["path"] for f in out["files"]] == ["pkg/shapes.py"]
    assert sum("outside the workspace" in w for w in out["warnings"]) == 2


def test_no_readable_file_is_a_clear_error(service):
    with pytest.raises(PruningError, match="none of the files could be read"):
        service.get_budgeted_context(ISSUE, ["nope.py"], 0.5)


def test_empty_issue_returns_a_skeleton_with_a_note(service):
    out = service.get_budgeted_context("   ", ["pkg/shapes.py"], 0.3)
    assert out["ranking"] == "skeleton"
    assert any("issue text is empty" in w for w in out["warnings"])
    assert out["kept_tokens"] <= out["budget_tokens"]
    assert out["files"][0]["kept_signature"]


def test_file_with_syntax_errors_still_works(service):
    out = service.get_budgeted_context("good returns the wrong value", ["broken.py"], 0.9)
    assert any("syntax errors" in w for w in out["warnings"])
    assert "def good():" in out["context"]


def test_non_python_file_falls_back_to_first_lines(service):
    out = service.get_budgeted_context(ISSUE, ["notes.md", "pkg/shapes.py"], 0.3)
    assert any("not a Python file" in w for w in out["warnings"])
    assert "# Notes" in out["context"]
    assert out["kept_tokens"] <= out["budget_tokens"]


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), "lots"])
def test_bad_budgets_are_rejected(service, budget):
    with pytest.raises(PruningError, match="budget"):
        service.get_budgeted_context(ISSUE, ["pkg/shapes.py"], budget)


@pytest.mark.parametrize("files", [[], None, "pkg/shapes.py"])
def test_bad_file_lists_are_rejected(service, files):
    with pytest.raises(PruningError, match="files"):
        service.get_budgeted_context(ISSUE, files, 0.5)


def test_too_many_files_is_rejected(service):
    with pytest.raises(PruningError, match="too many files"):
        service.get_budgeted_context(ISSUE, ["pkg/shapes.py"] * 51, 0.5)


def test_missing_checkpoint_falls_back_to_bm25(workspace):
    svc = PruningService(root=workspace, scorer="auto", checkpoint=workspace / "no_such_checkpoint").load()
    assert svc.scorer_name == "bm25"
    assert any("Using BM25" in note for note in svc.startup_notes)
    out = svc.get_budgeted_context(ISSUE, ["pkg/shapes.py"], 0.3)
    assert any("Using BM25" in w for w in out["warnings"])


def test_unknown_scorer_is_rejected(workspace):
    with pytest.raises(PruningError, match="unknown scorer"):
        PruningService(root=workspace, scorer="magic")
