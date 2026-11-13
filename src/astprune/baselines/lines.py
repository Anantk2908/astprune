"""Shared machinery for baselines that keep lines instead of AST nodes.

A line baseline only has to say in which order it wants lines. The longest
prefix of that order that fits the budget is kept. Kept lines are shown in
file order with nothing in between, so these baselines pay no tokens for
omission markers (our AST view does pay for its markers).
"""

from __future__ import annotations

from ..parse import ParsedFile
from ..tokens import count_tokens
from ..types import PrunedContext, PrunedFile

LineOrder = list[tuple[str, int]]  # (path, 1-based line number), most wanted first

MAX_REFINE_ROUNDS = 4


def context_from_lines(files: list[ParsedFile], kept: dict[str, set[int]]) -> PrunedContext:
    ctx: PrunedContext = {}
    for pf in files:
        lines = sorted(kept.get(pf.path, ()))
        ctx[pf.path] = PrunedFile(
            path=pf.path, text="\n".join(pf.lines[i - 1] for i in lines), kept_lines=set(lines)
        )
    return ctx


def _exact_tokens(files: list[ParsedFile], order: LineOrder, k: int) -> int:
    kept: dict[str, list[int]] = {}
    for path, line in order[:k]:
        kept.setdefault(path, []).append(line)
    total = 0
    for pf in files:
        lines = kept.get(pf.path)
        if lines:
            total += count_tokens("\n".join(pf.lines[i - 1] for i in sorted(lines)) + "\n")
    return total


def prefix_fill(files: list[ParsedFile], order: LineOrder, budget: int) -> PrunedContext:
    """Keep the longest prefix of ``order`` whose rendering fits in ``budget`` tokens."""
    by_path = {pf.path: pf for pf in files}
    costs = [by_path[path].line_tokens[line - 1] for path, line in order]

    def extend(k: int, room: int) -> int:
        """Longest extension of prefix ``k`` whose per-line costs sum to at most ``room``."""
        while k < len(order) and costs[k] <= room:
            room -= costs[k]
            k += 1
        return k

    # Per-line counts slightly overestimate the joined text, so a prefix chosen
    # with them is safe. A few rounds of "measure exactly, spend the slack"
    # then bring it up to the real limit without a full search.
    lo = extend(0, budget)
    exact = _exact_tokens(files, order, lo)
    while lo > 0 and exact > budget:  # rare: joined text came out longer than the estimate
        lo = int(lo * 0.9)
        exact = _exact_tokens(files, order, lo)
    for _ in range(MAX_REFINE_ROUNDS):
        grown = extend(lo, budget - exact)
        if grown == lo:
            break
        grown_exact = _exact_tokens(files, order, grown)
        if grown_exact > budget:
            break
        lo, exact = grown, grown_exact
    kept: dict[str, set[int]] = {}
    for path, line in order[:lo]:
        kept.setdefault(path, set()).add(line)
    return context_from_lines(files, kept)
