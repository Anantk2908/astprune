"""Metrics for one issue, computed the same way for every method.

A method hands back, per file, the text it would show and the set of original
lines that survived. Everything here is derived from those two things plus the
gold labels, so a line pruner and an AST pruner are judged by identical code.

A gold function counts as kept only if every non-blank line of it survived.
"""

from __future__ import annotations

from .parse import ParsedFile, quiet_ast_parse
from .tokens import count_tokens
from .types import PrunedContext


def _nonblank(pf: ParsedFile, start: int, end: int) -> list[int]:
    return [i for i in range(start, end + 1) if pf.lines[i - 1].strip()]


def original_tokens(files: list[ParsedFile]) -> int:
    """Token count of the unpruned candidate files. Budgets are fractions of this."""
    return sum(pf.n_tokens for pf in files)


def output_tokens(ctx: PrunedContext) -> int:
    return sum(count_tokens(p.text + "\n") for p in ctx.values() if p.text)


def parses(text: str) -> bool:
    try:
        quiet_ast_parse(text)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def evaluate_instance(
    files: list[ParsedFile],
    gold_functions: list[dict],
    gold_lines: dict[str, list[int]],
    ctx: PrunedContext,
    parseable_originals: set[str] | None = None,
) -> dict:
    """Metrics for one issue.

    ``gold_functions`` and ``gold_lines`` describe the whole gold patch. A gold
    function in a file that is not among the candidates simply counts as lost,
    which is what happens when file retrieval misses.

    ``parseable_originals`` lists the candidate files that are valid Python
    before pruning. Only those are used for the parse-rate, since a pruner
    cannot be blamed for a file that was already broken.
    """
    by_path = {pf.path: pf for pf in files}
    kept = {path: p.kept_lines for path, p in ctx.items()}

    full_hits, sig_hits = [], []
    gold_token_total = 0
    gold_spans: dict[str, set[int]] = {}
    for fn in gold_functions:
        pf = by_path.get(fn["path"])
        lines_kept = kept.get(fn["path"], set())
        if pf is None:
            full_hits.append(False)
            sig_hits.append(False)
            continue
        body = _nonblank(pf, fn["start"], fn["end"])
        sig_end = fn["sig_end"] if fn["sig_end"] is not None else fn["end"]
        sig = _nonblank(pf, fn["start"], sig_end)
        full = all(i in lines_kept for i in body)
        full_hits.append(full)
        sig_hits.append(full or all(i in lines_kept for i in sig))
        gold_spans.setdefault(fn["path"], set()).update(range(fn["start"], fn["end"] + 1))

    for path, span in gold_spans.items():
        pf = by_path[path]
        gold_token_total += sum(pf.line_tokens[i - 1] for i in span if i in kept.get(path, set()))

    n_gold_lines = sum(len(v) for v in gold_lines.values())
    gold_lines_kept = sum(1 for path, lines in gold_lines.items() for i in lines if i in kept.get(path, set()))

    out_tokens = output_tokens(ctx)
    orig_tokens = original_tokens(files)

    parse_checked = parse_ok = 0
    for path, pruned in ctx.items():
        if not pruned.text.strip():
            continue
        if parseable_originals is not None and path not in parseable_originals:
            continue
        parse_checked += 1
        parse_ok += parses(pruned.text)

    n = len(gold_functions)
    return {
        "n_gold_functions": n,
        "gold_fn_recall": sum(full_hits) / n if n else None,
        "all_gold": float(all(full_hits)) if n else None,
        "sig_recall": sum(sig_hits) / n if n else None,
        "gold_line_recall": gold_lines_kept / n_gold_lines if n_gold_lines else None,
        "gold_token_precision": min(gold_token_total / out_tokens, 1.0) if out_tokens else 0.0,
        "output_tokens": out_tokens,
        "original_tokens": orig_tokens,
        "compression": out_tokens / orig_tokens if orig_tokens else 0.0,
        "parse_checked": parse_checked,
        "parse_ok": parse_ok,
    }
