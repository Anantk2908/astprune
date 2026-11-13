"""Bi-encoder scorer built on UniXcoder.

The same class serves two roles. Loaded from ``microsoft/unixcoder-base`` it is
the zero-shot ablation. Loaded from one of our checkpoints it is the trained
ranker. Nothing else differs between the two, which is the point of the
comparison.

The issue and each node are encoded separately and compared by cosine
similarity, so node embeddings can be cached and reused.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from ..parse import ParsedFile, node_text_for_scoring

BASE_MODEL = "microsoft/unixcoder-base"
MAX_QUERY_TOKENS = 512
MAX_NODE_TOKENS = 256


def issue_text(issue: str) -> str:
    """What the encoder sees of an issue: the title and opening text.

    Truncation to the token limit happens in the tokenizer. Collapsing blank
    runs first lets more of a traceback fit.
    """
    lines = [ln.rstrip() for ln in issue.strip().splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out)


class InputBuilder:
    """Token ids in UniXcoder's encoder-only format: ``<s> <encoder-only> </s> text </s>``.

    Kept apart from the model so the training-data builder can tokenize in
    worker processes without loading any weights.
    """

    def __init__(self, model_name_or_path: str = BASE_MODEL):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tok = self.tokenizer
        mode = tok.convert_tokens_to_ids("<encoder-only>")
        has_mode = mode is not None and mode != tok.unk_token_id
        self.prefix = [tok.cls_token_id, mode, tok.sep_token_id] if has_mode else [tok.cls_token_id]
        self.suffix = [tok.sep_token_id]
        self.pad_id = tok.pad_token_id

    def token_ids(self, text: str, max_tokens: int) -> list[int]:
        return self.token_ids_batch([text], max_tokens)[0]

    def token_ids_batch(self, texts: list[str], max_tokens: int) -> list[list[int]]:
        room = max_tokens - len(self.prefix) - len(self.suffix)
        enc = self.tokenizer(texts, add_special_tokens=False, truncation=True, max_length=room)
        return [self.prefix + ids + self.suffix for ids in enc["input_ids"]]

    def pad(self, batch: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(x) for x in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, x in enumerate(batch):
            ids[i, : len(x)] = torch.as_tensor(x, dtype=torch.long)
            mask[i, : len(x)] = 1
        return ids.to(device), mask.to(device)


class Encoder:
    """Text to unit-length vectors."""

    def __init__(self, model_name_or_path: str = BASE_MODEL, device: str | None = None,
                 batch_tokens: int = 24_000, name: str | None = None):
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = InputBuilder(model_name_or_path)
        self.tokenizer = self.inputs.tokenizer
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device).eval()
        self.batch_tokens = batch_tokens
        self.name = name or model_name_or_path.rstrip("/").split("/")[-1]
        self.n_parameters = sum(p.numel() for p in self.model.parameters())
        self._cache: dict[bytes, np.ndarray] = {}
        self._cache_limit = 300_000

    def clear_cache(self) -> None:
        """Forget cached embeddings. Must be called whenever the weights change."""
        self._cache.clear()

    def token_ids(self, text: str, max_tokens: int) -> list[int]:
        return self.inputs.token_ids(text, max_tokens)

    def pad(self, batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs.pad(batch, self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pooled, L2-normalised embeddings. Used by training as well as inference."""
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    @torch.no_grad()
    def embed(self, texts: list[str], max_tokens: int) -> np.ndarray:
        """Embeddings for ``texts`` (rows are unit vectors). Results are cached by content."""
        dim = self.model.config.hidden_size
        out = np.zeros((len(texts), dim), dtype=np.float32)
        keys = [hashlib.blake2b(f"{max_tokens}\0{t}".encode("utf-8", "replace"), digest_size=12).digest()
                for t in texts]
        todo: dict[bytes, list[int]] = {}
        for i, key in enumerate(keys):
            hit = self._cache.get(key)
            if hit is not None:
                out[i] = hit
            else:
                todo.setdefault(key, []).append(i)
        if not todo:
            return out
        todo_keys = list(todo)
        todo_ids = self.inputs.token_ids_batch([texts[todo[k][0]] for k in todo_keys], max_tokens)
        pending = list(zip(todo_keys, todo_ids))
        pending.sort(key=lambda kv: len(kv[1]), reverse=True)  # similar lengths share a batch
        if len(self._cache) + len(pending) > self._cache_limit:
            self._cache.clear()
        start = 0
        while start < len(pending):
            width = len(pending[start][1])
            size = max(1, min(len(pending) - start, self.batch_tokens // max(width, 1)))
            chunk = pending[start : start + size]
            ids, mask = self.pad([x for _, x in chunk])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                vecs = self.forward(ids, mask).cpu().numpy().astype(np.float32)
            for (key, _), vec in zip(chunk, vecs):
                self._cache[key] = vec
                for row in todo[key]:
                    out[row] = vec
            start += size
        return out

    def similarity(self, issue: str, texts: list[str], max_tokens: int = MAX_NODE_TOKENS) -> np.ndarray:
        """Cosine similarity between the issue and each text."""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        query = self.embed([issue_text(issue)], MAX_QUERY_TOKENS)[0]
        return self.embed(texts, max_tokens) @ query


class EncoderNodeScorer:
    """Scores AST nodes by cosine similarity to the issue."""

    def __init__(self, encoder: Encoder, name: str | None = None):
        self.encoder = encoder
        self.name = name or encoder.name

    def score(self, issue: str, files: list[ParsedFile]) -> dict[str, np.ndarray]:
        texts = [node_text_for_scoring(pf, node) for pf in files for node in pf.nodes]
        flat = self.encoder.similarity(issue, texts)
        out: dict[str, np.ndarray] = {}
        offset = 0
        for pf in files:
            out[pf.path] = flat[offset : offset + len(pf.nodes)]
            offset += len(pf.nodes)
        return out
