"""Training-free relevance from a code LLM's perplexity.

This is the ranking criterion of LongCodeZip's coarse stage (Shi et al., ASE
2025), written from scratch: a piece of code is relevant to a query if reading
the code first makes the query easier to predict. For a node ``c`` and issue
``q`` the score is the negative mean token loss of ``q`` given ``c``, which is
a monotone transform of the conditional perplexity LongCodeZip ranks by.

It is not the authors' implementation. It applies their criterion to our AST
nodes and our selector, so the only thing that differs from our trained ranker
in the comparison is where the score comes from: a general code LLM with no
training on bug fixes, against a small encoder trained on them.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from ..parse import ParsedFile, node_text_for_scoring
from .encoder import issue_text

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


class PerplexityNodeScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 max_query_tokens: int = 256, max_node_tokens: int = 384,
                 batch_tokens: int = 12_000, name: str = "ppl"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.n_parameters = sum(p.numel() for p in self.model.parameters())
        self.max_query_tokens = max_query_tokens
        self.max_node_tokens = max_node_tokens
        self.batch_tokens = batch_tokens
        self._separator = self.tokenizer.encode("\n\n", add_special_tokens=False)
        self._pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self._cache: dict[bytes, float] = {}

    def _ids(self, texts: list[str], limit: int) -> list[list[int]]:
        enc = self.tokenizer(texts, add_special_tokens=False, truncation=True, max_length=limit)
        return enc["input_ids"]

    @torch.no_grad()
    def _query_loss(self, node_ids: list[list[int]], query_ids: list[int]) -> list[float]:
        """Mean loss of the query tokens after each node, lower meaning the node helps more."""
        losses = [0.0] * len(node_ids)
        order = sorted(range(len(node_ids)), key=lambda i: -len(node_ids[i]))
        tail = self._separator + query_ids
        start = 0
        while start < len(order):
            width = len(node_ids[order[start]]) + len(tail)
            size = max(1, min(len(order) - start, self.batch_tokens // width))
            rows = order[start : start + size]
            ids = torch.full((len(rows), width), self._pad, dtype=torch.long)
            mask = torch.zeros((len(rows), width), dtype=torch.long)
            spans = []
            for r, i in enumerate(rows):
                seq = node_ids[i] + tail
                ids[r, : len(seq)] = torch.tensor(seq)
                mask[r, : len(seq)] = 1
                spans.append((len(seq) - len(query_ids), len(seq)))
            ids, mask = ids.to(self.device), mask.to(self.device)
            logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
            token_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none"
            ).view(len(rows), width - 1)
            for r, i in enumerate(rows):
                lo, hi = spans[r]  # query tokens sit at positions lo..hi-1, predicted at lo-1..hi-2
                losses[i] = token_loss[r, lo - 1 : hi - 1].mean().item()
            start += size
        return losses

    def score(self, issue: str, files: list[ParsedFile]) -> dict[str, np.ndarray]:
        query = issue_text(issue)
        query_ids = self._ids([query], self.max_query_tokens)[0]
        texts = [node_text_for_scoring(pf, node) for pf in files for node in pf.nodes]
        flat = np.zeros(len(texts), dtype=np.float64)
        if query_ids and texts:
            query_key = hashlib.blake2b(query.encode("utf-8", "replace"), digest_size=8).digest()
            keys = [query_key + hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=12).digest() for t in texts]
            todo = [i for i, k in enumerate(keys) if k not in self._cache]
            if todo:
                unique: dict[bytes, int] = {}
                for i in todo:
                    unique.setdefault(keys[i], i)
                rows = list(unique.values())
                node_ids = self._ids([texts[i] for i in rows], self.max_node_tokens)
                if len(self._cache) > 400_000:
                    self._cache.clear()
                for i, loss in zip(rows, self._query_loss(node_ids, query_ids)):
                    self._cache[keys[i]] = -loss
            flat = np.array([self._cache[k] for k in keys], dtype=np.float64)
        out: dict[str, np.ndarray] = {}
        offset = 0
        for pf in files:
            out[pf.path] = flat[offset : offset + len(pf.nodes)]
            offset += len(pf.nodes)
        return out
