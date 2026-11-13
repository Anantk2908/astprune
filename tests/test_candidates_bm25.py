from collections import Counter

import numpy as np

from astprune.data.candidates import ModuleIndex, is_test_path, list_imports
from astprune.parse import parse_source
from astprune.scorers.bm25 import BM25NodeScorer, bm25_scores, tokenize

PATHS = [
    "lib/pkg/__init__.py",
    "lib/pkg/core.py",
    "lib/pkg/utils/__init__.py",
    "lib/pkg/utils/text.py",
    "lib/pkg/views.py",
    "other/core.py",
    "setup.py",
]


def test_tokenize_splits_identifiers():
    toks = tokenize("QuerySet.get_or_create(HTTPResponse)")
    for expected in ["queryset", "query", "set", "get_or_create", "get", "create", "httpresponse", "http", "response"]:
        assert expected in toks


def test_bm25_ranks_the_matching_document_first():
    docs = [Counter(tokenize(t)) for t in ["def parse_date(value)", "def render_template(name)", "x = 1"]]
    scores = bm25_scores(tokenize("date parsing fails"), docs)
    assert int(np.argmax(scores)) == 0
    assert scores[2] == 0.0
    assert bm25_scores(["anything"], []).shape == (0,)


def test_node_scorer_returns_one_score_per_node(sample_source):
    pf = parse_source(sample_source, "sample.py")
    scores = BM25NodeScorer().score("area of a shape is wrong", [pf])
    assert scores[pf.path].shape == (len(pf.nodes),)
    best = pf.nodes[int(np.argmax(scores[pf.path]))]
    assert best.name == "Shape.area"


def test_imports_absolute_relative_and_src_layout():
    index = ModuleIndex(PATHS)
    source = (
        "import os\n"
        "import pkg.core\n"
        "from pkg.utils import text\n"
        "from . import views\n"
        "from .utils.text import clean\n"
        "from numpy import array\n"
    )
    found = index.direct_imports("lib/pkg/models.py", source)
    assert found == ["lib/pkg/core.py", "lib/pkg/utils/text.py", "lib/pkg/utils/__init__.py", "lib/pkg/views.py"]


def test_imports_survive_python2_syntax():
    source = "import pkg.core\nprint 'old'\nfrom pkg.utils import text\n"
    mods = [m for _, m, _ in list_imports(source)]
    assert "pkg.core" in mods and "pkg.utils" in mods


def test_test_paths_are_recognised():
    assert is_test_path("tests/test_x.py")
    assert is_test_path("pkg/tests/helpers.py")
    assert is_test_path("pkg/test_models.py")
    assert is_test_path("conftest.py")
    assert not is_test_path("pkg/models.py")
    assert not is_test_path("pkg/contest.py")
    assert not is_test_path("pkg/latest/core.py")
