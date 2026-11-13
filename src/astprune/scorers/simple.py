"""Two small scorers: a random floor and a rank fusion of other scorers."""

from __future__ import annotations

import hashlib

import numpy as np

from ..parse import ParsedFile


class RandomScorer:
    """Random order, fixed per issue. The floor any real scorer has to beat."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, issue: str, files: list[ParsedFile]) -> dict[str, np.ndarray]:
        digest = hashlib.blake2b(issue.encode("utf-8", "replace"), digest_size=8).digest()
        rng = np.random.default_rng([self.seed, int.from_bytes(digest, "little")])
        return {pf.path: rng.random(len(pf.nodes)) for pf in files}


class HybridScorer:
    """Reciprocal rank fusion: each scorer votes ``1 / (k + rank)`` for every node.

    Only ranks are used, so scorers on different scales (BM25 and cosine) mix
    without any tuning.
    """

    def __init__(self, scorers: list, k: int = 60, name: str | None = None):
        self.scorers = scorers
        self.k = k
        self.name = name or "rrf_" + "+".join(s.name for s in scorers)

    def score(self, issue: str, files: list[ParsedFile]) -> dict[str, np.ndarray]:
        return self.fuse([s.score(issue, files) for s in self.scorers], files)

    def fuse(self, all_scores: list[dict[str, np.ndarray]], files: list[ParsedFile]) -> dict[str, np.ndarray]:
        sizes = [len(pf.nodes) for pf in files]
        fused = np.zeros(sum(sizes), dtype=np.float64)
        for scores in all_scores:
            flat = np.concatenate([np.asarray(scores[pf.path], dtype=np.float64) for pf in files])
            ranks = np.empty(len(flat), dtype=np.int64)
            ranks[np.argsort(-flat, kind="stable")] = np.arange(1, len(flat) + 1)
            fused += 1.0 / (self.k + ranks)
        out: dict[str, np.ndarray] = {}
        offset = 0
        for pf, size in zip(files, sizes):
            out[pf.path] = fused[offset : offset + size]
            offset += size
        return out
