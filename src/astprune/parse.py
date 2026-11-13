"""Split a Python file into the units the pruner ranks.

Every line of the file belongs to exactly one node, so a file can always be
rebuilt from its nodes. Node kinds:

* ``function``  a top-level def, decorators included
* ``method``    a def directly inside a class body
* ``class``     the class header: the ``class X(...):`` line(s), the docstring
                and anything else before the first method
* ``block``     a run of other statements (imports, constants, class
                attributes that sit between methods)

Functions nested inside functions stay part of their parent. Defs hidden
inside ``if``/``try`` at module level stay part of the surrounding block.

tree-sitter is the main parser. If it reports syntax errors we try Python's
own ``ast`` module, and if that fails too we keep tree-sitter's partial tree,
which is still correct for the parts of the file outside the broken region.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass, field

import tree_sitter_python as tsp
from tree_sitter import Language, Parser

from .tokens import count_tokens, count_tokens_batch

# Gaps between defs longer than this are split at statement boundaries so one
# huge run of constants does not become a single all-or-nothing node.
MAX_BLOCK_LINES = 40

_LANGUAGE = Language(tsp.language())
_PARSER = Parser(_LANGUAGE)


@dataclass(slots=True)
class Node:
    idx: int                      # position in file order
    path: str
    kind: str                     # function | method | class | block
    name: str                     # qualified name, e.g. "Foo.bar"
    start: int                    # first line, 1-based, inclusive
    end: int                      # last line, 1-based, inclusive
    sig_end: int | None = None    # last line of the signature, None if there is no signature tier
    body_indent: int | None = None
    parents: tuple[int, ...] = () # idx of enclosing class nodes, outermost first
    docstring: str = ""
    tokens_full: int = 0
    tokens_sig: int = 0

    @property
    def has_signature(self) -> bool:
        """True when a signature-only view exists and is smaller than the full node."""
        # body_indent is None for one-liners such as "def f(): return 1".
        return self.sig_end is not None and self.body_indent is not None and self.sig_end < self.end


@dataclass
class ParsedFile:
    path: str
    lines: list[str]
    nodes: list[Node]
    parser: str                   # tree-sitter | ast | tree-sitter-partial | none
    had_errors: bool
    _line_tokens: list[int] | None = field(default=None, repr=False)
    _n_tokens: int | None = field(default=None, repr=False)
    # Scratch space for scorers that want to cache per-file work (BM25 term counts).
    memo: dict = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        """Sum of node token counts. A close upper bound on ``n_tokens``, free to compute."""
        return sum(n.tokens_full for n in self.nodes)

    @property
    def n_tokens(self) -> int:
        """Exact token count of the whole file. Budgets are fractions of this."""
        if self._n_tokens is None:
            self._n_tokens = count_tokens("\n".join(self.lines) + "\n")
        return self._n_tokens

    @property
    def line_tokens(self) -> list[int]:
        """Token count of each line (index 0 is line 1). Computed on first use."""
        if self._line_tokens is None:
            self._line_tokens = count_tokens_batch([ln + "\n" for ln in self.lines])
        return self._line_tokens

    def text(self, node: Node) -> str:
        return "\n".join(self.lines[node.start - 1 : node.end])

    def signature_text(self, node: Node) -> str:
        """The signature-only rendering of a node (same text the selector emits)."""
        if not node.has_signature:
            return self.text(node)
        sig = self.lines[node.start - 1 : node.sig_end]
        if node.kind != "class" and node.body_indent is not None:
            sig = sig + [" " * node.body_indent + "..."]
        return "\n".join(sig)


# --------------------------------------------------------------------------
# Intermediate form shared by the tree-sitter and ast front ends
# --------------------------------------------------------------------------

@dataclass
class _Def:
    kind: str                     # "function" or "class"
    name: str
    start: int
    end: int
    sig_end: int
    body_indent: int | None
    docstring: str
    children: list["_Def"] = field(default_factory=list)
    stmt_starts: list[int] = field(default_factory=list)  # non-def statements in a class body


# ---- tree-sitter front end ------------------------------------------------

def _ts_docstring(block) -> str:
    if block is None:
        return ""
    for child in block.children:
        if child.type == "comment":
            continue
        if child.type == "expression_statement" and child.children and child.children[0].type == "string":
            raw = child.children[0].text.decode("utf-8", errors="replace")
            return raw.strip("rRbBuUfF").strip("\"'").strip()
        break
    return ""


def _ts_unwrap(node):
    """Return (definition node, outer node) for defs, decorated or not, else (None, node)."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None and inner.type in ("function_definition", "class_definition"):
            return inner, node
        return None, node
    if node.type in ("function_definition", "class_definition"):
        return node, node
    return None, node


def _ts_def(inner, outer) -> _Def:
    name_node = inner.child_by_field_name("name")
    name = name_node.text.decode("utf-8", errors="replace") if name_node is not None else "<anonymous>"
    body = inner.child_by_field_name("body")
    start = outer.start_point[0] + 1
    end = outer.end_point[0] + 1
    # The signature ends on the line of the ':' that opens the body.
    sig_end = inner.start_point[0] + 1
    for child in inner.children:
        if body is not None and child.start_byte >= body.start_byte:
            break
        if child.type == ":":
            sig_end = child.end_point[0] + 1
    body_indent = None
    first_stmt = None
    if body is not None:
        first_stmt = next((c for c in body.children if c.type != "comment"), None)
    if first_stmt is not None and first_stmt.start_point[0] + 1 > sig_end:
        body_indent = first_stmt.start_point[1]
    else:
        sig_end = end  # one-liner such as "def f(): return 1"
    d = _Def(
        kind="function" if inner.type == "function_definition" else "class",
        name=name, start=start, end=end, sig_end=sig_end,
        body_indent=body_indent, docstring=_ts_docstring(body),
    )
    if d.kind == "class" and body is not None and body_indent is not None:
        d.children, d.stmt_starts = _ts_members(body.children)
    return d


def _ts_members(children) -> tuple[list[_Def], list[int]]:
    defs: list[_Def] = []
    stmt_starts: list[int] = []
    for child in children:
        inner, outer = _ts_unwrap(child)
        if inner is None:
            stmt_starts.append(child.start_point[0] + 1)
        else:
            defs.append(_ts_def(inner, outer))
    return defs, stmt_starts


def _parse_tree_sitter(source: str) -> tuple[list[_Def], list[int], bool]:
    tree = _PARSER.parse(source.encode("utf-8", errors="replace"))
    defs, stmt_starts = _ts_members(tree.root_node.children)
    return defs, stmt_starts, tree.root_node.has_error


# ---- ast front end (fallback) ----------------------------------------------

_AST_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _ast_def(node) -> _Def:
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    end = node.end_lineno or node.lineno
    first = node.body[0]
    if first.lineno > node.lineno:
        sig_end, body_indent = first.lineno - 1, first.col_offset
    else:
        sig_end, body_indent = end, None
    d = _Def(
        kind="class" if isinstance(node, ast.ClassDef) else "function",
        name=node.name, start=start, end=end, sig_end=sig_end, body_indent=body_indent,
        docstring=(ast.get_docstring(node, clean=False) or "").strip(),
    )
    if d.kind == "class" and body_indent is not None:
        d.children, d.stmt_starts = _ast_members(node.body)
    return d


def _ast_members(body) -> tuple[list[_Def], list[int]]:
    defs: list[_Def] = []
    stmt_starts: list[int] = []
    for stmt in body:
        if isinstance(stmt, _AST_DEFS):
            defs.append(_ast_def(stmt))
        else:
            stmt_starts.append(stmt.lineno)
    return defs, stmt_starts


def quiet_ast_parse(source: str) -> ast.Module:
    """``ast.parse`` without the SyntaxWarnings old code triggers (bad escapes and the like)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(source)


def _parse_ast(source: str) -> tuple[list[_Def], list[int]]:
    return _ast_members(quiet_ast_parse(source).body)


# ---- turning defs into a line partition -------------------------------------

class _Builder:
    def __init__(self, path: str, lines: list[str]):
        self.path = path
        self.lines = lines
        self.nodes: list[Node] = []

    def _blank(self, lo: int, hi: int) -> bool:
        return all(not self.lines[i - 1].strip() for i in range(lo, hi + 1))

    def _add(self, **kw) -> Node:
        node = Node(idx=len(self.nodes), path=self.path, **kw)
        self.nodes.append(node)
        return node

    def _gap(self, lo: int, hi: int, stmt_starts: list[int], parents: tuple[int, ...], prefix: str) -> None:
        """Lines lo..hi hold no def. Make block nodes, or glue blank lines to the previous node."""
        if lo > hi:
            return
        if self.nodes and self._blank(lo, hi):
            self.nodes[-1].end = hi
            return
        cuts = [s for s in stmt_starts if lo < s <= hi]
        chunk_start = lo
        last_cut = None
        for cut in cuts + [hi + 1]:
            if cut - chunk_start > MAX_BLOCK_LINES and last_cut is not None and last_cut > chunk_start:
                self._block(chunk_start, last_cut - 1, parents, prefix)
                chunk_start = last_cut
            last_cut = cut
        self._block(chunk_start, hi, parents, prefix)

    def _block(self, lo: int, hi: int, parents: tuple[int, ...], prefix: str) -> None:
        if self.nodes and self._blank(lo, hi):
            self.nodes[-1].end = hi
            return
        self._add(kind="block", name=f"{prefix}<block:{lo}>", start=lo, end=hi, parents=parents)

    def level(self, defs: list[_Def], lo: int, hi: int, stmt_starts: list[int],
              parents: tuple[int, ...], prefix: str) -> None:
        cursor = lo
        for d in sorted(defs, key=lambda x: x.start):
            if d.start < cursor or d.end > hi:
                continue  # overlaps something already placed; leave it inside the gap
            self._gap(cursor, d.start - 1, stmt_starts, parents, prefix)
            if d.kind == "function":
                self._add(
                    kind="method" if parents else "function", name=prefix + d.name,
                    start=d.start, end=d.end, sig_end=d.sig_end, body_indent=d.body_indent,
                    parents=parents, docstring=d.docstring,
                )
            else:
                self._class(d, parents, prefix)
            cursor = d.end + 1
        self._gap(cursor, hi, stmt_starts, parents, prefix)

    def _class(self, d: _Def, parents: tuple[int, ...], prefix: str) -> None:
        members = [c for c in sorted(d.children, key=lambda x: x.start) if c.start > d.sig_end]
        header_end = members[0].start - 1 if members else d.end
        header = self._add(
            kind="class", name=prefix + d.name, start=d.start, end=header_end,
            sig_end=d.sig_end, body_indent=d.body_indent, parents=parents, docstring=d.docstring,
        )
        if members:
            self.level(members, header_end + 1, d.end, d.stmt_starts,
                       parents + (header.idx,), prefix + d.name + ".")


def parse_source(source: str, path: str = "<memory>") -> ParsedFile:
    """Parse Python source into a :class:`ParsedFile`. Never raises on bad syntax."""
    lines = source.split("\n")
    parser_used, had_errors = "tree-sitter", False
    try:
        defs, stmt_starts, had_errors = _parse_tree_sitter(source)
    except (ValueError, RecursionError):
        defs, stmt_starts, had_errors, parser_used = [], [], True, "none"
    if had_errors:
        try:
            defs, stmt_starts = _parse_ast(source)
            parser_used = "ast"
        except (SyntaxError, ValueError, RecursionError):
            if parser_used != "none":
                parser_used = "tree-sitter-partial"

    builder = _Builder(path, lines)
    builder.level(defs, 1, len(lines), stmt_starts, (), "")
    nodes = builder.nodes
    if not nodes:  # empty file
        nodes = [Node(idx=0, path=path, kind="block", name="<block:1>", start=1, end=len(lines))]

    pf = ParsedFile(path=path, lines=lines, nodes=nodes, parser=parser_used, had_errors=had_errors)
    full = count_tokens_batch([pf.text(n) + "\n" for n in nodes])
    sig = count_tokens_batch([pf.signature_text(n) + "\n" if n.has_signature else "" for n in nodes])
    for node, tf, tsig in zip(nodes, full, sig):
        node.tokens_full = tf
        node.tokens_sig = tsig if node.has_signature else tf
    return pf


def node_text_for_scoring(pf: ParsedFile, node: Node, max_lines: int = 60) -> str:
    """Text shown to a scorer: file path, qualified name, then the start of the node.

    Many functions have no docstring, so the signature and first body lines
    carry the signal. Encoders truncate this further to their own limit.
    """
    body = "\n".join(pf.lines[node.start - 1 : min(node.end, node.start - 1 + max_lines)])
    return f"# {pf.path} :: {node.name}\n{body}"
