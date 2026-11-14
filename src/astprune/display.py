"""Names used for methods in tables and figures, in one place so they never drift apart."""

from __future__ import annotations

# method key -> label, in the order tables list them
LABELS: dict[str, str] = {
    "line_window": "Line window",
    "chunk_bm25": "Chunk RAG (BM25)",
    "chunk_unixcoder": "Chunk RAG (UniXcoder, zero-shot)",
    "skeleton": "Skeleton (Agentless-style)",
    "swe_pruner": "SWE-Pruner (released checkpoint)",
    "longcodezip_rank": "LongCodeZip (coarse stage)",
    "longcodezip": "LongCodeZip (full)",
    "ast_random": "AST + random order",
    "ast_bm25": "AST + BM25",
    "ast_unixcoder": "AST + UniXcoder, zero-shot",
    "ast_ppl": "AST + LLM perplexity",
    "ast_trained": "AST + trained ranker (ours)",
    "ast_hybrid": "AST + trained ranker + BM25 (ours)",
}

# selector ablations: suffix -> what changed
VARIANT_LABELS: dict[str, str] = {
    "nosig": "no signature tier",
    "sig15": "15% of budget reserved for signatures",
    "sig30": "30% of budget reserved for signatures",
    "density": "rank by probability per token",
    "perfile": "budget split per file",
}

SETTING_LABELS: dict[str, str] = {
    "oracle": "Gold files only",
    "oracle_imports": "Gold files + direct imports",
    "bm25_5": "BM25 top-5 files",
    "bm25_10": "BM25 top-10 files",
    "bm25_20": "BM25 top-20 files",
}


def label(method: str) -> str:
    if method in LABELS:
        return LABELS[method]
    for suffix, text in VARIANT_LABELS.items():
        if method.endswith("_" + suffix):
            base = method[: -len(suffix) - 1]
            return f"{LABELS.get(base, base)}, {text}"
    return method


def sort_key(method: str) -> tuple[int, str]:
    order = list(LABELS)
    for i, key in enumerate(order):
        if method == key:
            return (i * 10, method)
        if method.startswith(key + "_"):
            return (i * 10 + 1, method)
    return (10_000, method)
