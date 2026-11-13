"""Which files does a pruner get to look at for an issue?

Two ways to pick them, as in the proposal:

* oracle: the files the gold patch edits, plus the files those import directly
* retrieved: the top-k files from a BM25 search over the whole repository
"""

from __future__ import annotations

import ast
import posixpath
import re
from collections import Counter

import numpy as np

from ..parse import quiet_ast_parse
from ..scorers.bm25 import bm25_scores, tokenize

# Files above this size are data tables or generated code (pydicom ships DICOM
# dictionaries of up to 2.7 MB as .py files). No agent would read one whole, so
# they are never offered as retrieved or imported candidates. The largest
# hand-written files in the benchmark repositories are well below this.
MAX_CANDIDATE_CHARS = 400_000

_TEST_PATH = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")
_IMPORT_LINE = re.compile(r"^\s*(?:from\s+(\.*)([\w.]*)\s+import\s+(.+)|import\s+(.+))$")


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search(path))


# --------------------------------------------------------------------------
# direct imports
# --------------------------------------------------------------------------

def _imports_via_ast(source: str) -> list[tuple[int, str, list[str]]] | None:
    try:
        tree = quiet_ast_parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    found: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((0, alias.name, []))
        elif isinstance(node, ast.ImportFrom):
            found.append((node.level, node.module or "", [a.name for a in node.names]))
    return found


def _imports_via_regex(source: str) -> list[tuple[int, str, list[str]]]:
    found: list[tuple[int, str, list[str]]] = []
    for line in source.split("\n"):
        m = _IMPORT_LINE.match(line)
        if not m:
            continue
        if m.group(4):
            for part in m.group(4).split(","):
                name = part.strip().split(" as ")[0].strip()
                if name:
                    found.append((0, name, []))
        else:
            names = [p.strip().split(" as ")[0].strip() for p in m.group(3).strip("()").split(",")]
            found.append((len(m.group(1)), m.group(2), [n for n in names if n and n != "\\"]))
    return found


def list_imports(source: str) -> list[tuple[int, str, list[str]]]:
    """``(relative level, module, imported names)`` for every import statement, in file order."""
    found = _imports_via_ast(source)
    return found if found is not None else _imports_via_regex(source)


class ModuleIndex:
    """Resolves dotted module names to files of one repository snapshot."""

    def __init__(self, paths: list[str]):
        self.paths = set(paths)
        # "matplotlib.axes._base" must resolve even though the file lives under
        # "lib/", so index every dotted suffix of every path.
        self._by_suffix: dict[str, list[str]] = {}
        for path in paths:
            stem = path[:-3]
            if stem.endswith("/__init__"):
                stem = stem[: -len("/__init__")]
            elif stem == "__init__":
                continue
            parts = stem.split("/")
            for i in range(len(parts)):
                self._by_suffix.setdefault(".".join(parts[i:]), []).append(path)

    def resolve_absolute(self, module: str, importer: str) -> str | None:
        hits = self._by_suffix.get(module)
        if not hits:
            return None
        # Prefer the file closest to the repository root, then one that shares
        # the importer's top-level directory.
        top = importer.split("/")[0]
        return min(hits, key=lambda p: (p.count("/"), not p.startswith(top + "/"), p))

    def resolve_relative(self, level: int, module: str, importer: str) -> str | None:
        base = posixpath.dirname(importer)
        for _ in range(level - 1):
            base = posixpath.dirname(base)
        stem = posixpath.join(base, *module.split(".")) if module else base
        for candidate in (stem + ".py", posixpath.join(stem, "__init__.py")):
            if candidate in self.paths:
                return candidate
        return None

    def direct_imports(self, importer: str, source: str) -> list[str]:
        """Repository files that ``importer`` imports, in order of first appearance."""
        out: list[str] = []

        def add(path: str | None) -> bool:
            if path and path != importer and path not in out:
                out.append(path)
            return path is not None

        for level, module, names in list_imports(source):
            if level:
                resolved_any = False
                for name in names:  # "from . import views" names a sibling module
                    sub = f"{module}.{name}" if module else name
                    resolved_any |= add(self.resolve_relative(level, sub, importer))
                if module or not resolved_any:
                    add(self.resolve_relative(level, module, importer))
            else:
                for name in names:  # "from pkg import module"
                    add(self.resolve_absolute(f"{module}.{name}", importer))
                add(self.resolve_absolute(module, importer))
        return out


# --------------------------------------------------------------------------
# BM25 file retrieval
# --------------------------------------------------------------------------

class FileRetriever:
    """BM25 over the ``.py`` files of a repository snapshot.

    Term counts are cached by git blob id, so issues from the same repository
    only pay for the files that changed between their commits.
    """

    def __init__(self, repo, max_cache: int = 60_000):
        self.repo = repo
        self.max_cache = max_cache
        self._tf_cache: dict[str, Counter] = {}
        self._too_big: set[str] = set()   # blob ids over MAX_CANDIDATE_CHARS

    def _tf(self, path: str, sha: str) -> Counter:
        tf = self._tf_cache.get(sha)
        if tf is None:
            text = self.repo.read_blob(sha) or ""
            if len(text) > MAX_CANDIDATE_CHARS:
                self._too_big.add(sha)
                text = ""
            tf = Counter(tokenize(path + "\n" + text))
            if len(self._tf_cache) >= self.max_cache:
                self._tf_cache.clear()
            self._tf_cache[sha] = tf
        return tf

    def rank(self, commit: str, issue: str, top_k: int = 20, include_tests: bool = False) -> list[str]:
        tree = self.repo.tree(commit)
        paths = [p for p in sorted(tree) if include_tests or not is_test_path(p)]
        if not paths:
            return []
        tfs = [self._tf(p, tree[p]) for p in paths]
        scores = bm25_scores(tokenize(issue), tfs)
        ranked = [paths[i] for i in np.argsort(-scores, kind="stable") if tree[paths[i]] not in self._too_big]
        return ranked[:top_k]
