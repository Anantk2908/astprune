"""LongCodeZip (Shi et al., ASE 2025) as a baseline.

LongCodeZip is training-free. It splits a file into function-level chunks and
ranks them by how much each chunk lowers the perplexity of the query under a
code LLM. A second stage then drops low-value line blocks inside the kept
functions with a knapsack.

Two variants, both driven by the authors' released package:

``longcodezip_rank``
    Their chunking and their conditional-perplexity ranking, then chunks are
    taken in rank order until our token budget is full. This is the coarse
    stage (``rank_only=True`` in their API) under exact budget control.

``longcodezip``
    Their full two-stage ``compress_code_file``. Its own budget is a soft
    target in the scorer model's tokens, so we ask for a target, measure the
    result with our tokenizer and lower the target until it fits.

Install (kept out of requirements.txt because its pinned tree-sitter conflicts
with ours and it is not needed for anything else):

    pip install --no-deps git+https://github.com/YerbaPage/LongCodeZip.git loguru

The scorer LLM is set in experiments/config.yaml (``longcodezip.model``). The
paper uses a 7B model. On an 8 GB GPU we use Qwen2.5-Coder-1.5B-Instruct, which
is a deviation and is reported as such.
"""

from __future__ import annotations

import hashlib

from ..parse import ParsedFile
from ..scorers.encoder import issue_text
from ..tokens import count_tokens
from ..types import PrunedContext
from .lines import LineOrder, context_from_lines, prefix_fill

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_QUERY_CHARS = 2000   # about what our encoder sees of the issue (512 tokens)
MAX_CHUNK_CHARS = 12_000  # very long chunks are cut for scoring only, never for output
MAX_RETRIES = 4


def _load(model_name: str):
    try:
        from longcodezip import LongCodeZip
    except ImportError as err:  # pragma: no cover - depends on the optional install
        raise ImportError(
            "LongCodeZip is not installed. Run: "
            "pip install --no-deps git+https://github.com/YerbaPage/LongCodeZip.git loguru"
        ) from err
    try:
        from loguru import logger

        logger.remove()  # the library logs every step at debug level
    except ImportError:
        pass
    import torch

    if torch.cuda.is_available():
        return LongCodeZip(model_name=model_name)
    # CPU fallback, only sensible for smoke tests.
    return LongCodeZip(model_name=model_name, device_map="cpu", model_config={"torch_dtype": torch.float32})


def locate_chunks(pf: ParsedFile, chunks: list[str]) -> list[tuple[int, int] | None]:
    """Line span of each chunk in the file, or None when a chunk cannot be found.

    Chunks come back as substrings of the source in file order, so a forward
    search is enough. Whitespace at the edges is ignored.
    """
    source = "\n".join(pf.lines)
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for chunk in chunks:
        body = chunk.strip("\n")
        at = source.find(body, cursor) if body.strip() else -1
        if at < 0:
            at = source.find(body) if body.strip() else -1
        if at < 0:
            spans.append(None)
            continue
        first = source.count("\n", 0, at) + 1
        last = first + body.count("\n")
        spans.append((first, last))
        cursor = at + len(body)
    return spans


def lines_of_output(pf: ParsedFile, text: str) -> set[int]:
    """Which original lines appear, in order, in a compressed text."""
    kept: set[int] = set()
    cursor = 0
    stripped = [ln.rstrip() for ln in pf.lines]
    for out_line in text.split("\n"):
        want = out_line.rstrip()
        if not want.strip():
            continue
        for i in range(cursor, len(stripped)):
            if stripped[i] == want:
                kept.add(i + 1)
                cursor = i + 1
                break
    return kept


class LongCodeZipBaseline:
    def __init__(self, mode: str = "rank", model_name: str = DEFAULT_MODEL, name: str | None = None):
        if mode not in ("rank", "full"):
            raise ValueError("mode must be 'rank' or 'full'")
        self.mode = mode
        self.model_name = model_name
        self.name = name or ("longcodezip_rank" if mode == "rank" else "longcodezip")
        self._lcz = None
        self._ppl_cache: dict[bytes, float] = {}

    @property
    def lcz(self):
        if self._lcz is None:
            self._lcz = _load(self.model_name)
        return self._lcz

    # -- stage 1: the authors' relevance score ---------------------------------
    def _relevance(self, chunk: str, query: str) -> float:
        """LongCodeZip's chunk score: the drop in query perplexity when the chunk comes first.

        ``get_condition_ppl`` also computes the perplexity of the query alone on
        every call. That term is the same for all chunks of an issue, so we
        compute only the conditional part with their own primitives. The ranking
        is identical.
        """
        key = hashlib.blake2b(f"{query}\0{chunk}".encode("utf-8", "replace"), digest_size=12).digest()
        if key not in self._ppl_cache:
            context = chunk[:MAX_CHUNK_CHARS] + "\n\n"
            result = self.lcz.get_ppl(
                text=context + query, granularity="line", condition_mode="prefix",
                condition_pos_id=self.lcz.get_token_length(context, add_special_tokens=True),
            )
            if len(self._ppl_cache) > 200_000:
                self._ppl_cache.clear()
            self._ppl_cache[key] = -float(result["ppl"])
        return self._ppl_cache[key]

    def prepare(self, issue: str, files: list[ParsedFile]):
        query = issue_text(issue)[:MAX_QUERY_CHARS]
        if self.mode == "full":
            return query
        scored: list[tuple[float, str, int, int]] = []
        leftovers: LineOrder = []
        for pf in files:
            chunks = self.lcz.split_code_by_functions("\n".join(pf.lines), language="python")
            covered: set[int] = set()
            for chunk, span in zip(chunks, locate_chunks(pf, chunks)):
                if span is None or not chunk.strip():
                    continue
                scored.append((self._relevance(chunk, query), pf.path, span[0], span[1]))
                covered.update(range(span[0], span[1] + 1))
            leftovers.extend((pf.path, i) for i in range(1, len(pf.lines) + 1) if i not in covered)
        scored.sort(key=lambda s: -s[0])
        order: LineOrder = [(path, i) for _, path, lo, hi in scored for i in range(lo, hi + 1)]
        return order + leftovers  # lines no chunk covers come last

    # -- budget -----------------------------------------------------------------
    def prune(self, prepared, files: list[ParsedFile], budget: int) -> PrunedContext:
        if self.mode == "rank":
            return prefix_fill(files, prepared, budget)
        return self._prune_full(prepared, files, budget)

    def _prune_full(self, query: str, files: list[ParsedFile], budget: int) -> PrunedContext:
        total = sum(pf.n_tokens for pf in files) or 1
        rate = budget / total
        best: dict[str, set[int]] = {}
        for _ in range(MAX_RETRIES):
            kept: dict[str, set[int]] = {}
            for pf in files:
                code = "\n".join(pf.lines)
                if not code.strip():
                    continue
                result = self.lcz.compress_code_file(
                    code=code, query=query, instruction="", rate=max(rate, 0.01),
                    language="python", context_budget="+0", rank_only=False,
                )
                kept[pf.path] = lines_of_output(pf, result["compressed_code"])
            ctx = context_from_lines(files, kept)
            used = sum(count_tokens(p.text + "\n") for p in ctx.values() if p.text)
            if used <= budget:
                return ctx
            best = kept
            rate *= 0.85 * budget / used  # their target is soft, so aim lower and retry
        # Still over after the retries: trim from the end of each file until it fits.
        order: LineOrder = sorted(((path, i) for path, lines in best.items() for i in lines),
                                  key=lambda x: (x[1], x[0]))
        return prefix_fill(files, order, budget)


def build_longcodezip(name: str, config: dict) -> LongCodeZipBaseline:
    model = config.get("longcodezip", {}).get("model", DEFAULT_MODEL)
    mode = "rank" if name.endswith("_rank") else "full"
    return LongCodeZipBaseline(mode=mode, model_name=model, name=name)
