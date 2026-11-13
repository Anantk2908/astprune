"""Where each split comes from, and the processed-instance format on disk."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .repos import PROJECT_ROOT

# Keep Hugging Face downloads inside the project so the setup is self-contained.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "hf"))

PROCESSED_DIR = Path(os.environ.get("ASTPRUNE_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))

# split key -> (Hugging Face dataset, split name, role)
SPLITS: dict[str, tuple[str, str, str]] = {
    "lite": ("princeton-nlp/SWE-bench_Lite", "test", "test"),
    "verified": ("princeton-nlp/SWE-bench_Verified", "test", "test"),
    "dev": ("princeton-nlp/SWE-bench", "dev", "dev"),
    "train": ("princeton-nlp/SWE-bench", "train", "train"),
    "gym": ("SWE-Gym/SWE-Gym", "train", "train"),
}


def load_raw(split: str) -> list[dict]:
    """Rows of a split with only the fields we use. ``hints_text`` is left out on
    purpose: it holds comments written after the issue was filed."""
    from datasets import load_dataset

    name, hf_split, _ = SPLITS[split]
    ds = load_dataset(name, split=hf_split)
    keep = ["instance_id", "repo", "base_commit", "problem_statement", "patch"]
    ds = ds.select_columns(keep)
    return [dict(row) for row in ds]


def processed_path(split: str) -> Path:
    return PROCESSED_DIR / f"{split}.jsonl"


def load_processed(split: str, only_ok: bool = True) -> list[dict]:
    """Instances written by ``experiments/build_data.py``."""
    path = processed_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python experiments/build_data.py --split {split}"
        )
    rows = [json.loads(line) for line in path.open()]
    return [r for r in rows if r["status"] == "ok"] if only_ok else rows
