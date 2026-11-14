"""The pruner as one callable: issue text and file paths in, budgeted view out.

The MCP server and the demo both go through :class:`PruningService`, so what
is shown live is the same code the experiments evaluate.

Expected problems are handled here instead of crashing the caller:

* a file that does not exist or sits outside the workspace is reported and skipped
* a file with syntax errors is still pruned (tree-sitter keeps the good parts)
* a file that is not Python is cut to its first lines
* an empty issue gives a structure-only skeleton, since there is nothing to rank by
* a missing checkpoint falls back to BM25 ranking
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from .baselines.line_window import LineWindow
from .data.repos import PROJECT_ROOT
from .parse import ParsedFile, parse_source
from .pruners import ASTPruner, SkeletonPruner
from .scorers.bm25 import BM25NodeScorer
from .scorers.simple import HybridScorer
from .select import DROP, FULL, SIG, render_file
from .tokens import count_tokens

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ranker" / "best"
MAX_FILES = 50
MAX_FILE_BYTES = 2_000_000
SCORERS = ("auto", "trained", "hybrid", "bm25")


class PruningError(ValueError):
    """A problem with the request that the caller can fix. The message is shown as is."""


class PruningService:
    def __init__(self, root: str | Path | None = None, scorer: str | None = None,
                 checkpoint: str | Path | None = None, device: str | None = None):
        self.root = Path(root or os.environ.get("ASTPRUNE_ROOT") or Path.cwd()).resolve()
        self.checkpoint = Path(checkpoint or os.environ.get("ASTPRUNE_CHECKPOINT") or DEFAULT_CHECKPOINT)
        self.device = device or os.environ.get("ASTPRUNE_DEVICE") or None
        wanted = (scorer or os.environ.get("ASTPRUNE_SCORER") or "auto").lower()
        if wanted not in SCORERS:
            raise PruningError(f"unknown scorer '{wanted}', expected one of {', '.join(SCORERS)}")
        self.requested_scorer = wanted
        self.startup_notes: list[str] = []
        self._method: ASTPruner | None = None
        self.scorer_name = ""
        self.device_name = "cpu"

    # -- model loading ----------------------------------------------------------
    def load(self) -> "PruningService":
        """Load the ranker. Safe to call more than once."""
        if self._method is not None:
            return self
        bm25 = BM25NodeScorer()
        wanted = self.requested_scorer
        scorer = bm25
        self.scorer_name = "bm25"
        if wanted in ("auto", "trained", "hybrid"):
            try:
                if not self.checkpoint.exists():
                    raise FileNotFoundError(f"no checkpoint at {self.checkpoint}")
                from .scorers.encoder import Encoder, EncoderNodeScorer

                encoder = Encoder(str(self.checkpoint), device=self.device, name="trained")
                trained = EncoderNodeScorer(encoder, name="trained")
                encoder.embed(["warm up"], 16)  # pay for CUDA start-up here, not on the first request
                encoder.clear_cache()
                self.device_name = encoder.device
                if wanted == "hybrid":  # rank fusion with BM25, kept as an option
                    scorer, self.scorer_name = HybridScorer([bm25, trained], name="hybrid"), "hybrid"
                else:  # auto and trained. On the dev split the trained ranker alone beat the fusion.
                    scorer, self.scorer_name = trained, "trained"
            except Exception as err:  # noqa: BLE001 - any load failure must leave a working service
                self.startup_notes.append(
                    f"Trained ranker not available ({type(err).__name__}: {err}). Using BM25 ranking."
                )
        self._method = ASTPruner(scorer, name=f"ast_{self.scorer_name}")
        return self

    # -- request handling ---------------------------------------------------------
    def _resolve(self, path_text: str) -> Path:
        candidate = Path(path_text)
        full = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if full != self.root and self.root not in full.parents:
            raise PermissionError(f"outside the workspace {self.root}")
        return full

    def _read(self, paths: list[str], warnings: list[str]) -> tuple[list[ParsedFile], list[ParsedFile]]:
        python_files: list[ParsedFile] = []
        other_files: list[ParsedFile] = []
        seen: set[str] = set()
        for path_text in paths:
            if not isinstance(path_text, str) or not path_text.strip():
                warnings.append("Skipped an empty file path.")
                continue
            try:
                full = self._resolve(path_text.strip())
                shown = str(full.relative_to(self.root))
                if shown in seen:
                    continue
                if not full.is_file():
                    raise FileNotFoundError("file not found")
                if full.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError(f"larger than {MAX_FILE_BYTES // 1_000_000} MB")
                source = full.read_bytes().decode("utf-8", errors="replace")
            except (OSError, ValueError) as err:
                warnings.append(f"Skipped {path_text}: {err}.")
                continue
            seen.add(shown)
            pf = parse_source(source, shown)
            if full.suffix != ".py":
                warnings.append(f"{shown} is not a Python file. Showing its first lines instead of ranked functions.")
                other_files.append(pf)
            else:
                if pf.had_errors:
                    warnings.append(f"{shown} has syntax errors. Pruned with the {pf.parser} parser, check the result.")
                python_files.append(pf)
        return python_files, other_files

    def get_budgeted_context(self, issue: str, files: list[str], budget: float = 0.25) -> dict:
        """Prune ``files`` for ``issue``.

        ``budget`` up to 1 is a share of the files' tokens (0.25 keeps a quarter).
        Above 1 it is a token count.
        """
        started = time.perf_counter()
        self.load()
        if not isinstance(files, (list, tuple)) or not files:
            raise PruningError("files must be a non-empty list of paths")
        if len(files) > MAX_FILES:
            raise PruningError(f"too many files ({len(files)}), the limit is {MAX_FILES}")
        try:
            budget = float(budget)
        except (TypeError, ValueError):
            raise PruningError("budget must be a number") from None
        if not math.isfinite(budget) or budget <= 0:
            raise PruningError("budget must be above 0: a share such as 0.25, or a token count such as 2000")

        warnings = list(self.startup_notes)
        python_files, other_files = self._read(list(files), warnings)
        all_files = python_files + other_files
        if not all_files:
            raise PruningError("none of the files could be read: " + " ".join(warnings[len(self.startup_notes):]))

        original = sum(pf.n_tokens for pf in all_files)
        budget_tokens = int(budget * original) if budget <= 1 else int(budget)
        issue = (issue or "").strip()

        # Non-Python files get a slice of the budget in proportion to their size.
        py_tokens = sum(pf.n_tokens for pf in python_files)
        py_budget = int(budget_tokens * py_tokens / original) if original else 0
        views: dict[str, dict] = {}
        if python_files:
            if issue:
                scores = self._method.prepare(issue, python_files)
                states = self._method.select_states(scores, python_files, py_budget)
                mode = self.scorer_name
            else:
                warnings.append("The issue text is empty, so nothing could be ranked. "
                                "Returning signatures first, then bodies in file order.")
                states = SkeletonPruner().select_states(python_files, py_budget)
                mode = "skeleton"
            for pf in python_files:
                pruned = render_file(pf, states[pf.path])
                node_states = states[pf.path]
                views[pf.path] = {
                    "path": pf.path, "text": pruned.text, "original_tokens": pf.n_tokens,
                    "kept_tokens": count_tokens(pruned.text + "\n") if pruned.text else 0,
                    "kept_full": [n.name for n in pf.nodes if node_states[n.idx] == FULL and n.kind != "block"],
                    "kept_signature": [n.name for n in pf.nodes
                                       if node_states[n.idx] == SIG and n.kind in ("function", "method")],
                    "dropped": sum(1 for n in pf.nodes if node_states[n.idx] == DROP and n.kind != "block"),
                    "parser": pf.parser,
                }
        else:
            mode = "line_window"
        if other_files:
            window = LineWindow()
            ctx = window.prune(window.prepare(issue, other_files), other_files, budget_tokens - py_budget)
            for pf in other_files:
                text = ctx[pf.path].text
                views[pf.path] = {"path": pf.path, "text": text, "original_tokens": pf.n_tokens,
                                  "kept_tokens": count_tokens(text + "\n") if text else 0,
                                  "kept_full": [], "kept_signature": [], "dropped": 0, "parser": "none"}

        ordered = [views[pf.path] for pf in all_files]
        kept = sum(v["kept_tokens"] for v in ordered)
        parts = []
        for v in ordered:
            if v["text"]:
                parts.append(f"### {v['path']} ({v['kept_tokens']} of {v['original_tokens']} tokens shown)\n{v['text']}")
            else:
                parts.append(f"### {v['path']} (0 of {v['original_tokens']} tokens shown, nothing ranked high enough)")
        return {
            "context": "\n\n".join(parts),
            "original_tokens": original,
            "kept_tokens": kept,
            "budget_tokens": budget_tokens,
            "saved_percent": round(100.0 * (1.0 - kept / original), 1) if original else 0.0,
            "ranking": mode,
            "device": self.device_name,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "files": [{k: v for k, v in view.items() if k != "text"} for view in ordered],
            "warnings": warnings,
        }

