"""SWE-Pruner (Wang et al., 2026) as a baseline, using the authors' released checkpoint.

SWE-Pruner is a 0.6B model (Qwen3-Reranker backbone with a CRF head) that
scores every token of a file against a query. The authors average token
scores into line scores and keep lines above a threshold.

A threshold gives no control over the output size, and our comparison needs
every method at the same budget. So we take the model's line scores exactly as
the authors compute them and keep the highest-scoring lines until the budget
is full. A function counts as kept only if all of its lines survive, which is
strict for a line pruner. Gold-line recall is the metric that treats it on
its own terms.

Install (kept out of requirements.txt, it is only needed for this baseline):

    pip install --no-deps swe-pruner
    huggingface-cli download ayanami-kitasan/code-pruner --local-dir checkpoints/swe_pruner

The authors run the backbone with flash-attention. That package has no wheel
for our torch build, so we switch the backbone to PyTorch's built-in scaled
dot-product attention. The maths is the same, only the kernel differs.
"""

from __future__ import annotations

import functools
from pathlib import Path

from ..data.repos import PROJECT_ROOT
from ..parse import ParsedFile
from ..scorers.encoder import issue_text
from ..types import PrunedContext
from .lines import LineOrder, prefix_fill

DEFAULT_CHECKPOINT = "checkpoints/swe_pruner"
MAX_QUERY_CHARS = 2000   # about what our encoder sees of the issue (512 tokens)
MAX_LENGTH = 8192        # the authors' context length


def _load(checkpoint: Path):
    try:
        import swe_pruner.swepruner as swepruner_module
        from swe_pruner import prune_wrapper
        from swe_pruner.model_structure import TokenScorer
    except ImportError as err:  # pragma: no cover - depends on the optional install
        raise ImportError("SWE-Pruner is not installed. Run: pip install --no-deps swe-pruner") from err
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(
            f"No SWE-Pruner checkpoint at {checkpoint}. Run: huggingface-cli download "
            f"ayanami-kitasan/code-pruner --local-dir {checkpoint}"
        )
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        swepruner_module.TokenScorer = functools.partial(TokenScorer, attn_implementation="sdpa")
    model = prune_wrapper.SwePrunerForCodePruning.from_pretrained(str(checkpoint))
    return model, prune_wrapper


class SWEPrunerBaseline:
    name = "swe_pruner"

    def __init__(self, config: dict | None = None):
        path = (config or {}).get("swe_pruner", {}).get("checkpoint", DEFAULT_CHECKPOINT)
        self.checkpoint = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
        self._model = None
        self._wrapper = None

    def _ensure(self):
        if self._model is None:
            self._model, self._wrapper = _load(self.checkpoint)
        return self._model, self._wrapper

    def line_scores(self, query: str, code: str) -> dict[int, float]:
        """Line scores for one file, following ``SwePrunerForCodePruning.prune`` up to its thresholding."""
        model, w = self._ensure()
        tok = model.tokenizer
        formatted = w.format_instruction(None, query)
        # Same room calculation as the authors: the chat prefix and suffix they
        # wrap around the input take about 60 tokens.
        room = MAX_LENGTH - 80 - w.estimate_token_count(formatted, tok)
        if room < 100:
            return {}
        if w.estimate_token_count(code, tok) > room:
            results = []
            for text, start, end in w.split_code_into_chunks(code, tok, chunk_max_tokens=room, overlap_tokens=50):
                _, scores, offsets = model._process_single_chunk(
                    query, text, tok, max_length=MAX_LENGTH, instruction=model.instruction)
                results.append((scores, offsets, start, end))
            scores, offsets = w.merge_token_scores_from_chunks(code, results)
        else:
            _, scores, offsets = model._process_single_chunk(
                query, code, tok, max_length=MAX_LENGTH, instruction=model.instruction)
        return w.aggregate_token_scores_to_lines(code, scores, offsets)

    def prepare(self, issue: str, files: list[ParsedFile]) -> LineOrder:
        query = issue_text(issue)[:MAX_QUERY_CHARS]
        scored: list[tuple[float, int, int, str]] = []
        for file_pos, pf in enumerate(files):
            if "swe_pruner" not in pf.memo or pf.memo["swe_pruner"][0] != query:
                code = "\n".join(pf.lines)
                pf.memo["swe_pruner"] = (query, self.line_scores(query, code) if code.strip() else {})
            per_line = pf.memo["swe_pruner"][1]
            for i in range(1, len(pf.lines) + 1):
                scored.append((per_line.get(i, 0.0), file_pos, i, pf.path))
        # Highest score first. Ties keep file order.
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))
        return [(path, line) for _, _, line, path in scored]

    def prune(self, order: LineOrder, files: list[ParsedFile], budget: int) -> PrunedContext:
        return prefix_fill(files, order, budget)

