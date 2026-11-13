"""BM25 over code, used three ways: ranking files, ranking AST nodes and ranking text chunks.

Written out by hand (about 40 lines) so that term counts can be cached per
git blob and reused across issues from the same repository.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..parse import ParsedFile, node_text_for_scoring

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

K1 = 1.2
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lower-cased identifiers plus their snake_case and camelCase parts.

    ``QuerySet.get_or_create`` gives ``queryset, query, set, get_or_create, get, or, create``.
    Keeping the whole identifier rewards exact name matches, the parts catch
    issues that describe the code in plain words.
    """
    out: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        if len(lowered) < 2:
            continue
        out.append(lowered)
        parts = [p.lower() for chunk in word.split("_") if chunk for p in _CAMEL.findall(chunk)]
        if len(parts) > 1:
            out.extend(p for p in parts if len(p) > 1)
    return out


def bm25_scores(query_tokens: list[str], doc_tfs: list[Counter], k1: float = K1, b: float = B) -> np.ndarray:
    """BM25 score of each document for the query (Lucene idf, never negative)."""
    n_docs = len(doc_tfs)
    scores = np.zeros(n_docs, dtype=np.float64)
    if n_docs == 0:
        return scores
    terms = set(query_tokens)
    doc_lens = np.array([sum(tf.values()) for tf in doc_tfs], dtype=np.float64)
    avg_len = doc_lens.mean() if doc_lens.sum() > 0 else 1.0
    norm = k1 * (1.0 - b + b * doc_lens / avg_len)
    for term in terms:
        tf = np.fromiter((d.get(term, 0) for d in doc_tfs), dtype=np.float64, count=n_docs)
        df = int((tf > 0).sum())
        if df == 0:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        scores += idf * tf * (k1 + 1.0) / (tf + norm)
    return scores


class BM25NodeScorer:
    """Scores AST nodes against the issue with BM25. The corpus is the issue's candidate nodes."""

    name = "bm25"

    def score(self, issue: str, files: list[ParsedFile]) -> dict[str, np.ndarray]:
        docs: list[Counter] = []
        for pf in files:
            if "bm25_node_tf" not in pf.memo:
                # Whole node text, not just its head: BM25 has no length limit.
                pf.memo["bm25_node_tf"] = [
                    Counter(tokenize(node_text_for_scoring(pf, node, max_lines=10**6))) for node in pf.nodes
                ]
            docs.extend(pf.memo["bm25_node_tf"])
        flat = bm25_scores(tokenize(issue), docs)
        out: dict[str, np.ndarray] = {}
        offset = 0
        for pf in files:
            out[pf.path] = flat[offset : offset + len(pf.nodes)]
            offset += len(pf.nodes)
        return out
