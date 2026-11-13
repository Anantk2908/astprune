"""Training examples for the ranker.

One example per issue:

* a query: the issue text
* positives: the nodes the gold patch changed (functions, methods, class headers)
* hard negatives: other nodes of the same files, half picked by BM25 (the ones
  a lexical ranker would confuse with the answer) and half at random
* import negatives: nodes from files the edited files import directly

Texts are tokenized once here and stored as uint16 arrays, so training never
waits on a tokenizer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from ..parse import ParsedFile, node_text_for_scoring, parse_source
from ..scorers.bm25 import BM25NodeScorer
from ..scorers.encoder import BASE_MODEL, MAX_NODE_TOKENS, MAX_QUERY_TOKENS, InputBuilder, issue_text
from .candidates import MAX_CANDIDATE_CHARS
from .datasets import PROCESSED_DIR
from .repos import Repo, RepoError

POSITIVE_KINDS = ("function", "method", "class")
MAX_SAME_FILE_NEGATIVES = 32
MAX_IMPORT_NEGATIVES = 16
MAX_IMPORT_FILES = 3
MIN_NEGATIVE_TOKENS = 8      # skips blank runs and one-line fillers


def examples_dir(name: str = "train_examples") -> Path:
    return PROCESSED_DIR / name


def build_repo_examples(repo_name: str, records: list[dict], seed: int = 0) -> dict:
    """Examples for all issues of one repository. Runs in a worker process."""
    inputs = InputBuilder(BASE_MODEL)
    bm25 = BM25NodeScorer()
    rng = random.Random(f"{seed}:{repo_name}")
    texts: list[str] = []           # node texts, deduplicated
    text_index: dict[str, int] = {}
    queries: list[str] = []
    examples: list[dict] = []

    def text_id(text: str) -> int:
        if text not in text_index:
            text_index[text] = len(texts)
            texts.append(text)
        return text_index[text]

    try:
        repo = Repo(repo_name)
    except (RepoError, OSError):
        return {"repo": repo_name, "examples": [], "node_tokens": [], "query_tokens": []}

    with repo:
        for rec in records:
            gold = {(g["path"], g["start"]) for g in rec["gold_nodes"]}
            wanted = {(g["path"], g["start"]) for g in rec["gold_nodes"] if g["kind"] in POSITIVE_KINDS}
            if not wanted:
                continue
            files: list[ParsedFile] = []
            for path in rec["oracle_files"]:
                source = repo.read(rec["base_commit"], path)
                if source is not None and len(source) <= MAX_CANDIDATE_CHARS:
                    files.append(parse_source(source, path))
            if not files:
                continue
            scores = bm25.score(rec["issue"], files)
            pos_texts: list[str] = []
            candidates: list[tuple[float, str]] = []
            for pf in files:
                for node in pf.nodes:
                    key = (pf.path, node.start)
                    if key in wanted:
                        pos_texts.append(node_text_for_scoring(pf, node))
                    elif key not in gold and node.tokens_full >= MIN_NEGATIVE_TOKENS:
                        candidates.append((float(scores[pf.path][node.idx]), node_text_for_scoring(pf, node)))
            if not pos_texts:
                continue
            # Only texts that end up in an example get an id, so nothing unused is stored.
            taken = set(pos_texts)
            candidates = [(s, t) for s, t in candidates if t not in taken]
            candidates.sort(key=lambda st: -st[0])
            half = MAX_SAME_FILE_NEGATIVES // 2
            lexical = [t for _, t in candidates[:half]]
            rest = [t for _, t in candidates[half:]]
            rng.shuffle(rest)
            same_texts = list(dict.fromkeys(lexical + rest[: MAX_SAME_FILE_NEGATIVES - len(lexical)]))
            pos = [text_id(t) for t in dict.fromkeys(pos_texts)]
            pos_set = set(pos)
            neg_same = [text_id(t) for t in same_texts]

            neg_import: list[int] = []
            import_files = list(rec["import_files"])
            rng.shuffle(import_files)
            for path in import_files[:MAX_IMPORT_FILES]:
                source = repo.read(rec["base_commit"], path)
                if source is None or len(source) > MAX_CANDIDATE_CHARS:
                    continue
                pf = parse_source(source, path)
                pool = [n for n in pf.nodes if n.kind in POSITIVE_KINDS and n.tokens_full >= MIN_NEGATIVE_TOKENS]
                rng.shuffle(pool)
                per_file = MAX_IMPORT_NEGATIVES // MAX_IMPORT_FILES + 1
                neg_import.extend(text_id(node_text_for_scoring(pf, n)) for n in pool[:per_file])
            neg_import = [t for t in dict.fromkeys(neg_import) if t not in pos_set][:MAX_IMPORT_NEGATIVES]

            examples.append({
                "instance_id": rec["instance_id"], "repo": repo_name, "query": len(queries),
                "pos": pos, "neg_same": neg_same, "neg_import": neg_import,
                "n_gold_functions": len(rec["gold_functions"]),
            })
            queries.append(issue_text(rec["issue"]))

    def tokenize(items: list[str], limit: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for start in range(0, len(items), 512):
            for ids in inputs.token_ids_batch(items[start : start + 512], limit):
                out.append(np.asarray(ids, dtype=np.uint16))
        return out

    return {"repo": repo_name, "examples": examples,
            "node_tokens": tokenize(texts, MAX_NODE_TOKENS),
            "query_tokens": tokenize(queries, MAX_QUERY_TOKENS)}


class TokenStore:
    """Many short token sequences in one flat array."""

    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        self.flat = flat
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.flat[self.offsets[i] : self.offsets[i + 1]]

    @classmethod
    def from_arrays(cls, arrays: list[np.ndarray]) -> "TokenStore":
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([len(a) for a in arrays], out=offsets[1:])
        flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.uint16)
        return cls(flat, offsets)

    def save(self, prefix: Path) -> None:
        np.save(f"{prefix}_flat.npy", self.flat)
        np.save(f"{prefix}_offsets.npy", self.offsets)

    @classmethod
    def load(cls, prefix: Path) -> "TokenStore":
        return cls(np.load(f"{prefix}_flat.npy", mmap_mode="r"), np.load(f"{prefix}_offsets.npy"))


def save_examples(parts: list[dict], out_dir: Path, meta: dict) -> dict:
    """Merge per-repository results into one store on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    node_arrays: list[np.ndarray] = []
    query_arrays: list[np.ndarray] = []
    examples: list[dict] = []
    for part in sorted(parts, key=lambda p: p["repo"]):
        node_base, query_base = len(node_arrays), len(query_arrays)
        node_arrays.extend(part["node_tokens"])
        query_arrays.extend(part["query_tokens"])
        for ex in part["examples"]:
            examples.append({**ex, "query": ex["query"] + query_base,
                             "pos": [i + node_base for i in ex["pos"]],
                             "neg_same": [i + node_base for i in ex["neg_same"]],
                             "neg_import": [i + node_base for i in ex["neg_import"]]})
    TokenStore.from_arrays(node_arrays).save(out_dir / "nodes")
    TokenStore.from_arrays(query_arrays).save(out_dir / "queries")
    with (out_dir / "examples.jsonl").open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    stats = {**meta, "examples": len(examples), "node_texts": len(node_arrays),
             "repos": len({e["repo"] for e in examples}),
             "mean_positives": round(float(np.mean([len(e["pos"]) for e in examples])), 3) if examples else 0,
             "mean_same_file_negatives": round(float(np.mean([len(e["neg_same"]) for e in examples])), 2) if examples else 0,
             "mean_import_negatives": round(float(np.mean([len(e["neg_import"]) for e in examples])), 2) if examples else 0}
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def load_examples(out_dir: Path) -> tuple[list[dict], TokenStore, TokenStore]:
    examples = [json.loads(line) for line in (out_dir / "examples.jsonl").open()]
    return examples, TokenStore.load(out_dir / "nodes"), TokenStore.load(out_dir / "queries")
