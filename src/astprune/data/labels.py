"""Turn a gold patch into labels on the pre-fix file.

A node is *gold* when the patch changes it:

* a non-blank line inside the node is removed or rewritten, or
* new lines are inserted into the node's body, or
* a decorator is added right above it.

New functions or classes that the patch adds between existing nodes do not
make any existing node gold. They did not exist before the fix, so there is
nothing for a pruner to keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from ..parse import Node, ParsedFile

_DEF_STARTS = ("def ", "async def ", "class ", "@")


@dataclass
class Insertion:
    after: int        # pre-fix line the new text goes after (0 = top of file)
    first: str        # first non-blank added line, unstripped
    last: str         # last non-blank added line, unstripped


@dataclass
class FileEdit:
    """What a patch does to one pre-fix file, in pre-fix line numbers."""

    path: str
    removed: dict[int, str] = field(default_factory=dict)   # line number -> text
    insertions: list[Insertion] = field(default_factory=list)
    # One entry per hunk: (first pre-fix line, the pre-fix lines the hunk shows).
    # Used to re-locate hunks whose line numbers are off, see ``align_edit``.
    hunks: list[tuple[int, list[str]]] = field(default_factory=list)


def parse_patch(patch: str) -> dict[str, FileEdit]:
    """Edits per existing ``.py`` file. New files, deleted files and non-Python files are skipped."""
    try:
        patch_set = PatchSet(patch)
    except (UnidiffParseError, ValueError):
        return {}
    edits: dict[str, FileEdit] = {}
    for pfile in patch_set:
        if pfile.is_added_file or pfile.is_removed_file or not pfile.source_file:
            continue
        path = pfile.source_file
        if path.startswith("a/"):
            path = path[2:]
        if not path.endswith(".py"):
            continue
        edit = FileEdit(path=path)
        for hunk in pfile:
            prev_source = hunk.source_start - 1  # last pre-fix line passed so far
            run: list[str] = []                  # current run of added lines
            run_is_rewrite = False               # run sits next to removed lines
            last_was_removed = False

            def close_run() -> None:
                # A run of added lines with no removed line next to it is a pure insertion.
                nonblank = [ln for ln in run if ln.strip()]
                if nonblank and not run_is_rewrite:
                    edit.insertions.append(Insertion(prev_source, nonblank[0], nonblank[-1]))

            edit.hunks.append((hunk.source_start, [ln.value.rstrip("\r\n") for ln in hunk
                                                   if ln.is_context or ln.is_removed]))
            for line in hunk:
                if line.is_added:
                    if not run:
                        run_is_rewrite = last_was_removed
                    run.append(line.value.rstrip("\r\n"))
                elif line.is_removed:
                    if run:
                        run_is_rewrite = True
                        close_run()
                        run = []
                    edit.removed[line.source_line_no] = line.value.rstrip("\r\n")
                    prev_source = line.source_line_no
                    last_was_removed = True
                elif line.is_context:
                    if run:
                        close_run()
                        run = []
                    prev_source = line.source_line_no
                    last_was_removed = False
            if run:
                close_run()
        if edit.removed or edit.insertions:
            edits[path] = edit
    return edits


def _hunk_offset(lines: list[str], start: int, block: list[str]) -> int | None:
    """How far a hunk has to move to sit on matching text, or None if it fits nowhere."""
    if not block:
        return 0
    want = [b.rstrip() for b in block]
    last_start = len(lines) - len(block) + 1

    def fits(at: int) -> bool:
        return 1 <= at <= last_start and all(lines[at - 1 + i].rstrip() == w for i, w in enumerate(want))

    for delta in range(0, len(lines) + 1):
        for signed in ((delta, -delta) if delta else (0,)):
            if fits(start + signed):
                return signed
    return None


def align_edit(pf: ParsedFile, edit: FileEdit) -> FileEdit | None:
    """Shift each hunk onto the text it describes.

    A few SWE-bench patches only apply to their base commit with a line offset
    (``git apply`` tolerates that). We do the same search ``patch`` does: try
    the stated position first, then the nearest position where the hunk's
    context and removed lines match the file. Returns None if a hunk fits nowhere.
    """
    bounds = []
    for start, block in edit.hunks:
        offset = _hunk_offset(pf.lines, start, block)
        if offset is None:
            return None
        bounds.append((start, start + max(len(block), 1) - 1, offset))
    if all(offset == 0 for _, _, offset in bounds):
        return edit

    def shift(line: int) -> int:
        for lo, hi, offset in bounds:
            if lo - 1 <= line <= hi:
                return line + offset
        return line

    return FileEdit(
        path=edit.path,
        removed={shift(n): text for n, text in edit.removed.items()},
        insertions=[Insertion(shift(i.after), i.first, i.last) for i in edit.insertions],
        hunks=[(start + offset, block) for (start, block), (_, _, offset) in zip(edit.hunks, bounds)],
    )


def edit_matches_file(pf: ParsedFile, edit: FileEdit) -> bool:
    """True when the patch's removed lines are really at those line numbers in the file."""
    for line_no, text in edit.removed.items():
        if line_no < 1 or line_no > len(pf.lines) or pf.lines[line_no - 1].strip() != text.strip():
            return False
    return all(ins.after <= len(pf.lines) for ins in edit.insertions)


@dataclass
class GoldFile:
    path: str
    node_idx: list[int]       # every changed node
    function_idx: list[int]   # changed nodes that are functions or methods
    lines: list[int]          # pre-fix lines an agent must see: changed lines and insertion anchors


def node_at(pf: ParsedFile, line: int) -> Node | None:
    """The node that owns a 1-based line number (binary search over the partition)."""
    lo, hi = 0, len(pf.nodes) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        node = pf.nodes[mid]
        if line < node.start:
            hi = mid - 1
        elif line > node.end:
            lo = mid + 1
        else:
            return node
    return None


def _indent_of(text: str) -> int:
    return len(text) - len(text.lstrip())


def _insertion_target(pf: ParsedFile, ins: Insertion) -> Node | None:
    n_lines = len(pf.lines)
    before = node_at(pf, ins.after) if ins.after >= 1 else None
    after = node_at(pf, ins.after + 1) if ins.after + 1 <= n_lines else None
    added_indent = _indent_of(ins.first)
    adds_definition = ins.first.lstrip().startswith(_DEF_STARTS)

    if before is not None:
        own_indent = _indent_of(pf.lines[before.start - 1])
        if before.kind in ("function", "method"):
            if added_indent > own_indent:
                return before  # deeper than the def line, so it is in that body
        elif before.kind == "class":
            if added_indent > own_indent and not (adds_definition and added_indent == before.body_indent):
                return before  # new attribute or docstring text, not a new method
        elif added_indent == own_indent and not adds_definition:
            return before      # new statement in a run of statements

    # "@decorator" added directly above an existing def changes that def.
    if after is not None and after.start == ins.after + 1 and ins.last.lstrip().startswith("@"):
        if after.kind in ("function", "method", "class"):
            return after
    return None


def gold_for_file(pf: ParsedFile, edit: FileEdit) -> GoldFile:
    """Map one file's edits onto its nodes."""
    changed: set[int] = set()
    lines: set[int] = set()

    for line in edit.removed:
        node = node_at(pf, line)
        if node is not None and pf.lines[line - 1].strip():
            changed.add(node.idx)
            lines.add(line)

    for ins in edit.insertions:
        target = _insertion_target(pf, ins)
        if target is None:
            continue
        changed.add(target.idx)
        anchors = [a for a in (ins.after, ins.after + 1)
                   if target.start <= a <= target.end and pf.lines[a - 1].strip()]
        if not anchors:  # insertion sits after trailing blank lines: anchor on the last code line
            anchors = [max((i for i in range(target.start, min(ins.after, target.end) + 1)
                            if pf.lines[i - 1].strip()), default=target.start)]
        lines.update(anchors)

    ordered = sorted(changed)
    return GoldFile(
        path=pf.path,
        node_idx=ordered,
        function_idx=[i for i in ordered if pf.nodes[i].kind in ("function", "method")],
        lines=sorted(lines),
    )
