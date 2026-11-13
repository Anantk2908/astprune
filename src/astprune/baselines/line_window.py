"""Simplest baseline: the first lines of each file until the budget runs out."""

from __future__ import annotations

from ..parse import ParsedFile
from ..types import PrunedContext
from .lines import LineOrder, prefix_fill


class LineWindow:
    """Every file is cut at the same fraction of its length, measured in tokens."""

    name = "line_window"

    def prepare(self, issue: str, files: list[ParsedFile]) -> LineOrder:
        keyed: list[tuple[float, int, int, str]] = []
        for file_pos, pf in enumerate(files):
            total = sum(pf.line_tokens) or 1
            seen = 0
            for i, tokens in enumerate(pf.line_tokens, start=1):
                seen += tokens
                keyed.append((seen / total, file_pos, i, pf.path))
        keyed.sort()
        return [(path, line) for _, _, line, path in keyed]

    def prune(self, order: LineOrder, files: list[ParsedFile], budget: int) -> PrunedContext:
        return prefix_fill(files, order, budget)
