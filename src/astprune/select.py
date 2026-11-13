"""Fill a token budget with ranked nodes and render what is left.

Each node ends up in one of three states:

* ``FULL``  shown as is
* ``SIG``   only the signature, with ``...`` as the body
* ``DROP``  left out; a run of dropped nodes becomes one ``# ... N lines omitted`` comment

The rendered view is always valid Python as long as the input file was, since
nothing is ever cut in the middle of a function.
"""

from __future__ import annotations

import math

import numpy as np

from .parse import Node, ParsedFile
from .tokens import count_tokens
from .types import PrunedContext, PrunedFile

DROP, SIG, FULL = 0, 1, 2

# Upper bound on the cost of one "# ... 1234 lines omitted" line. Charged once
# per kept node and once per file, which covers the worst case where every kept
# node sits between two dropped runs.
MARKER_TOKENS = 9

# After the greedy fill, this many of the best remaining nodes are tried one by
# one against the exact token count to use up slack left by the upper bound.
TOP_UP_TRIES = 12

States = dict[str, list[int]]


def _marker(n_lines: int, indent: int = 0) -> str:
    return f"{' ' * indent}# ... {n_lines} lines omitted"


def _state_cost(node: Node, state: int) -> int:
    if state == FULL:
        return node.tokens_full
    if state == SIG:
        return node.tokens_sig
    return 0


class _Budget:
    """Tracks estimated spend while nodes are switched on, class lines and markers included."""

    def __init__(self, files: list[ParsedFile]):
        self.files = {pf.path: pf for pf in files}
        self.states: States = {pf.path: [DROP] * len(pf.nodes) for pf in files}
        self.n_kept = {pf.path: 0 for pf in files}
        self.spent = 0

    def cost_to_raise(self, path: str, idx: int, state: int) -> int:
        pf, states = self.files[path], self.states[path]
        node = pf.nodes[idx]
        if states[idx] >= state:
            return 0
        cost = _state_cost(node, state) - _state_cost(node, states[idx])
        if states[idx] == DROP:
            cost += MARKER_TOKENS
            if self.n_kept[path] == 0:
                cost += MARKER_TOKENS
        for p in node.parents:  # a method needs its class line
            if states[p] == DROP:
                cost += pf.nodes[p].tokens_sig + MARKER_TOKENS
        return cost

    def raise_to(self, path: str, idx: int, state: int) -> None:
        self.spent += self.cost_to_raise(path, idx, state)
        states = self.states[path]
        for p in self.files[path].nodes[idx].parents:
            if states[p] == DROP:
                states[p] = SIG
                self.n_kept[path] += 1
        if states[idx] == DROP and state != DROP:
            self.n_kept[path] += 1
        states[idx] = max(states[idx], state)


def _ranking(files: list[ParsedFile], scores: dict[str, np.ndarray], order: str,
             temperature: float) -> list[tuple[str, int]]:
    items: list[tuple[float, str, int]] = []
    for pf in files:
        s = np.asarray(scores[pf.path], dtype=np.float64)
        if len(s) != len(pf.nodes):
            raise ValueError(f"{pf.path}: got {len(s)} scores for {len(pf.nodes)} nodes")
        for node in pf.nodes:
            items.append((float(s[node.idx]), pf.path, node.idx))
    if order == "density" and items:
        # Treat softmax(score / T) as the chance a node is needed and rank by
        # chance per token, the usual greedy rule for a knapsack.
        raw = np.array([it[0] for it in items]) / max(temperature, 1e-8)
        prob = np.exp(raw - raw.max())
        prob /= prob.sum()
        by_path = {pf.path: pf for pf in files}
        items = [
            (p / max(by_path[path].nodes[idx].tokens_full, 1), path, idx)
            for p, (_, path, idx) in zip(prob, items)
        ]
    elif order not in ("score", "density"):
        raise ValueError(f"unknown order: {order}")
    # Stable tie-break on (path, idx) keeps runs reproducible.
    items.sort(key=lambda it: (-it[0], it[1], it[2]))
    return [(path, idx) for _, path, idx in items]


def select(
    files: list[ParsedFile],
    scores: dict[str, np.ndarray],
    budget: int,
    *,
    sig_tier: bool = True,
    sig_reserve: float = 0.0,
    order: str = "score",
    temperature: float = 1.0,
) -> States:
    """Choose a state for every node so the rendered view fits in ``budget`` tokens.

    Passes over the nodes in rank order:

    1. keep nodes whole while they fit in ``(1 - sig_reserve) * budget``
    2. spend what is left on signatures of the next-ranked nodes
    3. use any remainder to keep more nodes whole

    ``sig_reserve=0`` is plain greedy with a signature fill at the end.
    ``sig_tier=False`` turns signatures off, so nodes are either whole or gone.

    Passes 1 to 3 use a cheap upper bound on the rendered size. Afterwards the
    result is checked against the real token count, so the returned selection
    never renders above ``budget``.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not 0.0 <= sig_reserve < 1.0:
        raise ValueError("sig_reserve must be in [0, 1)")
    ranking = _ranking(files, scores, order, temperature)
    b = _Budget(files)

    full_cap = budget if not sig_tier else math.floor(budget * (1.0 - sig_reserve))
    for path, idx in ranking:
        if b.spent + b.cost_to_raise(path, idx, FULL) <= full_cap:
            b.raise_to(path, idx, FULL)
    if sig_tier:
        for path, idx in ranking:
            node = b.files[path].nodes[idx]
            if node.has_signature and b.states[path][idx] == DROP:
                if b.spent + b.cost_to_raise(path, idx, SIG) <= budget:
                    b.raise_to(path, idx, SIG)
        for path, idx in ranking:
            if b.states[path][idx] != FULL and b.spent + b.cost_to_raise(path, idx, FULL) <= budget:
                b.raise_to(path, idx, FULL)

    return fit_exactly(files, b.states, ranking, budget, sig_tier=sig_tier)


def fit_exactly(files: list[ParsedFile], states: States, ranking: list[tuple[str, int]],
                budget: int, *, sig_tier: bool = True) -> States:
    """Make a selection fit the real token count, then use up leftover room.

    ``ranking`` lists ``(path, node idx)`` from most to least wanted. Nodes are
    dropped from the back if the rendering is over budget, and added from the
    front while there is slack.
    """
    by_path = {pf.path: pf for pf in files}
    file_tokens = {pf.path: _file_tokens(pf, states[pf.path]) for pf in files}
    total = sum(file_tokens.values())

    # Safety net: token counts are not exactly additive across joins.
    while total > budget:
        for path, idx in reversed(ranking):
            if states[path][idx] != DROP and not _has_kept_child(by_path[path], states[path], idx):
                states[path][idx] = DROP
                file_tokens[path] = _file_tokens(by_path[path], states[path])
                total = sum(file_tokens.values())
                break
        else:
            break

    # Exact top-up: the upper bound usually leaves a little room.
    tries = 0
    for path, idx in ranking:
        if tries >= TOP_UP_TRIES or budget - total < MARKER_TOKENS:
            break
        current = states[path][idx]
        if current == FULL:
            continue
        pf = by_path[path]
        node = pf.nodes[idx]
        targets = [FULL] + ([SIG] if sig_tier and node.has_signature and current == DROP else [])
        for target in targets:
            if _state_cost(node, target) - _state_cost(node, current) > budget - total:
                continue
            tries += 1
            trial = list(states[path])
            trial[idx] = target
            for p in node.parents:
                trial[p] = max(trial[p], SIG)
            new_tokens = _file_tokens(pf, trial)
            if total - file_tokens[path] + new_tokens <= budget:
                states[path] = trial
                total += new_tokens - file_tokens[path]
                file_tokens[path] = new_tokens
                break
    return states


def _file_tokens(pf: ParsedFile, states: list[int]) -> int:
    text = render_file(pf, states).text
    return count_tokens(text + "\n") if text else 0


def _has_kept_child(pf: ParsedFile, states: list[int], idx: int) -> bool:
    return any(states[n.idx] != DROP and idx in n.parents for n in pf.nodes)


def render_file(pf: ParsedFile, states: list[int]) -> PrunedFile:
    """Render one file under the given node states."""
    states = list(states)
    for node in pf.nodes:  # never emit a method without its class line
        if states[node.idx] != DROP:
            for p in node.parents:
                if states[p] == DROP:
                    states[p] = SIG
    if not any(states):
        return PrunedFile(path=pf.path, text="", kept_lines=set())

    out: list[str] = []
    kept: set[int] = set()
    pending = 0  # dropped lines not yet reported by a marker
    # Classes whose line was emitted and whose body is still open:
    # [last line of the class, body indent, has the body got a real statement yet]
    open_classes: list[list] = []

    def close_classes(before_line: int) -> None:
        # A class body made only of comments is a syntax error, so give it "...".
        while open_classes and open_classes[-1][0] < before_line:
            _, indent, has_stmt = open_classes.pop()
            if not has_stmt:
                out.append(" " * indent + "...")

    def class_last_line(node: Node) -> int:
        end = node.end
        for other in pf.nodes[node.idx + 1 :]:
            if node.idx not in other.parents:
                break
            end = other.end
        return end

    def emit(lo: int, hi: int) -> None:
        out.extend(pf.lines[lo - 1 : hi])
        kept.update(range(lo, hi + 1))

    for node in pf.nodes:
        state = states[node.idx]
        if state == DROP:
            pending += node.end - node.start + 1
            continue
        close_classes(node.start)
        first = pf.lines[node.start - 1]
        if pending:
            out.append(_marker(pending, len(first) - len(first.lstrip())))
            pending = 0
        for oc in open_classes:  # every class still open contains this node
            oc[2] = True
        if node.kind == "class":
            if state == FULL:
                emit(node.start, node.end)
                rest = pf.lines[node.sig_end : node.end]
                has_stmt = any(ln.strip() and not ln.strip().startswith("#") for ln in rest)
            else:
                emit(node.start, node.sig_end)
                pending += node.end - node.sig_end
                has_stmt = False
            if node.body_indent is not None:
                open_classes.append([class_last_line(node), node.body_indent, has_stmt])
        elif state == SIG and node.has_signature:
            emit(node.start, node.sig_end)
            out.append(" " * (node.body_indent or 0) + "...")
        else:
            emit(node.start, node.end)

    close_classes(len(pf.lines) + 1)
    if pending:
        out.append(_marker(pending))
    return PrunedFile(path=pf.path, text="\n".join(out), kept_lines=kept)


def render(files: list[ParsedFile], states: States) -> PrunedContext:
    return {pf.path: render_file(pf, states[pf.path]) for pf in files}


def rendered_tokens(ctx: PrunedContext) -> int:
    """Tokens in the pruned view. Files with nothing kept cost nothing."""
    return sum(count_tokens(p.text + "\n") for p in ctx.values() if p.text)
