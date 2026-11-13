"""Small shared types.

Every pruning method, ours and the baselines, returns the same thing: for each
file, the text it would show the agent and the set of original line numbers
that survived word for word. Metrics only look at this, so no method gets a
custom scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .parse import ParsedFile


@dataclass
class PrunedFile:
    path: str
    text: str
    kept_lines: set[int] = field(default_factory=set)  # 1-based original line numbers


PrunedContext = dict[str, PrunedFile]


class Pruner(Protocol):
    """Anything that can shrink a set of files to a token budget for one issue."""

    name: str

    def prune(self, issue: str, files: list[ParsedFile], budget: int) -> PrunedContext:
        ...
