"""Pruning methods that work on AST nodes.

Every method here and in ``baselines/`` has the same two-step shape so the
evaluation loop can score an issue once and reuse the work for every budget:

    prepared = method.prepare(issue, files)
    context  = method.prune(prepared, files, budget)
"""

from __future__ import annotations

import numpy as np

from .parse import ParsedFile
from .select import FULL, SIG, States, _Budget, fit_exactly, render, select
from .types import PrunedContext


class ASTPruner:
    """Rank AST nodes with a scorer, then fill the budget (our method and its ablations)."""

    def __init__(self, scorer, name: str | None = None, *, sig_tier: bool = True,
                 sig_reserve: float = 0.0, order: str = "score", temperature: float = 1.0,
                 per_file: bool = False):
        self.scorer = scorer
        self.name = name or f"ast_{scorer.name}"
        self.sig_tier = sig_tier
        self.sig_reserve = sig_reserve
        self.order = order
        self.temperature = temperature
        self.per_file = per_file

    def prepare(self, issue: str, files: list[ParsedFile]) -> dict[str, np.ndarray]:
        return self.scorer.score(issue, files)

    def select_states(self, scores: dict[str, np.ndarray], files: list[ParsedFile], budget: int) -> States:
        opts = dict(sig_tier=self.sig_tier, sig_reserve=self.sig_reserve,
                    order=self.order, temperature=self.temperature)
        if not self.per_file:
            return select(files, scores, budget, **opts)
        # Ablation: each file gets a share of the budget in proportion to its size.
        total = sum(pf.total_tokens for pf in files) or 1
        states: States = {}
        for pf in files:
            share = int(budget * pf.total_tokens / total)
            states.update(select([pf], {pf.path: scores[pf.path]}, share, **opts))
        return states

    def prune(self, scores: dict[str, np.ndarray], files: list[ParsedFile], budget: int) -> PrunedContext:
        return render(files, self.select_states(scores, files, budget))


class SkeletonPruner:
    """Agentless-style skeleton: every signature first, then full bodies in file order.

    It uses the same nodes and renderer as our method but never looks at the
    issue, so it shows what code structure buys without any ranking.
    """

    name = "skeleton"

    def prepare(self, issue: str, files: list[ParsedFile]) -> None:
        return None

    def prune(self, prepared: None, files: list[ParsedFile], budget: int) -> PrunedContext:
        b = _Budget(files)
        in_order = [(pf.path, node.idx) for pf in files for node in pf.nodes]
        for path, idx in in_order:
            if b.files[path].nodes[idx].has_signature and b.spent + b.cost_to_raise(path, idx, SIG) <= budget:
                b.raise_to(path, idx, SIG)
        for path, idx in in_order:
            if b.spent + b.cost_to_raise(path, idx, FULL) <= budget:
                b.raise_to(path, idx, FULL)
        return render(files, fit_exactly(files, b.states, in_order, budget, sig_tier=True))
