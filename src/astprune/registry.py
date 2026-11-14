"""Builds methods from their names, so experiment configs can stay plain lists.

Encoders are loaded once and shared: ``ast_trained`` and all of its selector
ablations use the same model and the same cached scores.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .baselines.chunk_rag import ChunkRAG
from .baselines.line_window import LineWindow
from .data.repos import PROJECT_ROOT
from .evaluation import CachedScorer
from .pruners import ASTPruner, SkeletonPruner
from .scorers.bm25 import BM25NodeScorer
from .scorers.simple import HybridScorer, RandomScorer

DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "config.yaml"

# Selector variants, appended to an ast_<scorer> name: ast_trained_nosig and so on.
VARIANTS: dict[str, dict] = {
    "nosig": {"sig_tier": False},
    "sig15": {"sig_reserve": 0.15},
    "sig30": {"sig_reserve": 0.30},
    "density": {"order": "density"},
    "perfile": {"per_file": True},
}

# Softmax temperature for the density ordering. Matches the InfoNCE temperature.
DENSITY_TEMPERATURE = 0.05


def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())


class Registry:
    def __init__(self, config: dict | None = None, device: str | None = None):
        self.config = config or load_config()
        self.device = device
        self._encoders: dict[str, object] = {}
        self._scorers: dict[str, CachedScorer] = {}

    # -- shared pieces ------------------------------------------------------
    def encoder(self, key: str):
        """``key`` is "unixcoder" (zero-shot weights) or "trained" (our checkpoint)."""
        if key not in self._encoders:
            from .scorers.encoder import Encoder

            models = self.config["models"]
            if key == "unixcoder":
                source = models["base_encoder"]
            elif key.startswith("trained@"):  # another training run, e.g. trained@seed1
                source = str(PROJECT_ROOT / "checkpoints" / "ranker" / key.split("@", 1)[1] / "best")
                if not Path(source).exists():
                    raise FileNotFoundError(f"no checkpoint at {source}")
            elif key == "trained":
                source = models["trained_checkpoint"]
                if not Path(source).is_absolute():
                    source = str(PROJECT_ROOT / source)
                if not Path(source).exists():
                    raise FileNotFoundError(
                        f"No trained checkpoint at {source}. Train one with "
                        "experiments/train_ranker.py or point models.trained_checkpoint at one."
                    )
            else:
                raise KeyError(key)
            self._encoders[key] = Encoder(source, device=self.device, name=key)
        return self._encoders[key]

    def scorer(self, key: str) -> CachedScorer:
        if key not in self._scorers:
            if key == "bm25":
                base = BM25NodeScorer()
            elif key == "random":
                base = RandomScorer()
            elif key in ("unixcoder", "trained") or key.startswith("trained@"):
                from .scorers.encoder import EncoderNodeScorer

                base = EncoderNodeScorer(self.encoder(key), name=key)
            elif key == "ppl":
                from .scorers.perplexity import DEFAULT_MODEL, PerplexityNodeScorer

                opts = self.config.get("perplexity", {})
                base = PerplexityNodeScorer(
                    opts.get("model", DEFAULT_MODEL), device=self.device,
                    max_query_tokens=opts.get("max_query_tokens", 256),
                    max_node_tokens=opts.get("max_node_tokens", 384),
                )
            elif key == "hybrid":
                base = HybridScorer([self.scorer("bm25"), self.scorer("trained")], name="hybrid")
            elif key == "hybrid_zs":
                base = HybridScorer([self.scorer("bm25"), self.scorer("unixcoder")], name="hybrid_zs")
            else:
                raise KeyError(f"unknown scorer: {key}")
            self._scorers[key] = CachedScorer(base)
        return self._scorers[key]

    # -- methods --------------------------------------------------------------
    def method(self, name: str):
        if name == "line_window":
            return LineWindow()
        if name == "skeleton":
            return SkeletonPruner()
        if name == "chunk_bm25":
            return ChunkRAG()
        if name in ("chunk_unixcoder", "chunk_trained"):
            return ChunkRAG(encoder=self.encoder(name.split("_", 1)[1]), name=name)
        if name == "longcodezip" or name.startswith("longcodezip_"):
            from .baselines.longcodezip import build_longcodezip

            return build_longcodezip(name, self.config)
        if name == "swe_pruner":
            from .baselines.swe_pruner import SWEPrunerBaseline

            return SWEPrunerBaseline(self.config)
        if name.startswith("ast_"):
            rest = name[len("ast_"):]
            variant = None
            for suffix in VARIANTS:
                if rest.endswith("_" + suffix):
                    rest, variant = rest[: -len(suffix) - 1], suffix
                    break
            opts = dict(self.config.get("selector", {}))
            opts.update(VARIANTS.get(variant, {}))
            if opts.get("order") == "density":
                opts["temperature"] = DENSITY_TEMPERATURE
            return ASTPruner(self.scorer(rest), name=name, **opts)
        raise KeyError(f"unknown method: {name}")

    def methods(self, names: list[str]) -> dict[str, object]:
        return {name: self.method(name) for name in names}
