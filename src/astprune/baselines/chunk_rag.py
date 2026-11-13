"""Standard retrieval baseline: fixed-size text chunks ranked against the issue.

Chunks are runs of whole lines of about ``chunk_tokens`` tokens. They ignore
code structure, so a chunk can start or stop in the middle of a function.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..parse import ParsedFile
from ..scorers.bm25 import bm25_scores, tokenize
from ..types import PrunedContext
from .lines import LineOrder, prefix_fill

CHUNK_TOKENS = 256


def make_chunks(pf: ParsedFile, chunk_tokens: int = CHUNK_TOKENS) -> list[tuple[int, int]]:
    """``(first line, last line)`` of each chunk, covering the whole file."""
    chunks: list[tuple[int, int]] = []
    start, size = 1, 0
    for i, tokens in enumerate(pf.line_tokens, start=1):
        if size and size + tokens > chunk_tokens:
            chunks.append((start, i - 1))
            start, size = i, 0
        size += tokens
    if start <= len(pf.lines):
        chunks.append((start, len(pf.lines)))
    return chunks


class ChunkRAG:
    """Rank chunks with BM25 or with a text encoder, keep the best until the budget is full."""

    def __init__(self, encoder=None, chunk_tokens: int = CHUNK_TOKENS, name: str | None = None):
        self.encoder = encoder
        self.chunk_tokens = chunk_tokens
        self.name = name or ("chunk_bm25" if encoder is None else f"chunk_{encoder.name}")

    def prepare(self, issue: str, files: list[ParsedFile]) -> LineOrder:
        spans: list[tuple[str, int, int]] = []
        texts: list[str] = []
        tfs: list[Counter] = []
        for pf in files:
            key = ("chunks", self.chunk_tokens)
            if key not in pf.memo:
                chunks = make_chunks(pf, self.chunk_tokens)
                pf.memo[key] = (chunks, [f"# {pf.path}\n" + "\n".join(pf.lines[lo - 1 : hi]) for lo, hi in chunks])
            chunks, chunk_texts = pf.memo[key]
            spans.extend((pf.path, lo, hi) for lo, hi in chunks)
            texts.extend(chunk_texts)
            if self.encoder is None:
                tf_key = ("chunk_tf", self.chunk_tokens)
                if tf_key not in pf.memo:
                    pf.memo[tf_key] = [Counter(tokenize(t)) for t in chunk_texts]
                tfs.extend(pf.memo[tf_key])
        if not spans:
            return []
        if self.encoder is None:
            scores = bm25_scores(tokenize(issue), tfs)
        else:
            scores = self.encoder.similarity(issue, texts)
        order: LineOrder = []
        for j in np.argsort(-np.asarray(scores), kind="stable"):
            path, lo, hi = spans[j]
            order.extend((path, line) for line in range(lo, hi + 1))
        return order

    def prune(self, order: LineOrder, files: list[ParsedFile], budget: int) -> PrunedContext:
        return prefix_fill(files, order, budget)
