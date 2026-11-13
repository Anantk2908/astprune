"""Evaluation loop shared by every experiment script.

For each issue: load the candidate files once, let each method score the issue
once, then prune at every budget and compute metrics. Results go to one JSONL
file per (split, setting, method) so methods can be added later without
rerunning the rest.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

from .data.repos import PROJECT_ROOT, Repo
from .metrics import evaluate_instance, original_tokens, parses
from .parse import ParsedFile, parse_source

RUNS_DIR = PROJECT_ROOT / "results" / "runs"

SETTINGS = ("oracle", "oracle_imports", "bm25_5", "bm25_10", "bm25_20")


def candidate_paths(rec: dict, setting: str) -> list[str]:
    """Files a method may look at for this issue under ``setting``.

    Oracle settings return paths in alphabetical order. Listing the gold files
    first would leak the answer to any method that fills in file order (the
    skeleton baseline does). BM25 settings keep retrieval rank order, which is
    information a real agent would also have.
    """
    if setting == "oracle":
        return sorted(rec["oracle_files"])
    if setting == "oracle_imports":
        return sorted(set(rec["oracle_files"]) | set(rec["import_files"]))
    if setting.startswith("bm25_"):
        return list(rec["bm25_files"][: int(setting.split("_")[1])])
    raise ValueError(f"unknown setting: {setting}")


def budget_label(kind: str, value: float) -> str:
    return f"r{value:.2f}" if kind == "relative" else f"f{int(value)}"


class CachedScorer:
    """Remembers a scorer's output for the last few (issue, files) pairs.

    Several selector variants share one scorer. With this wrapper the encoder
    runs once per issue and setting, not once per variant.
    """

    def __init__(self, scorer, size: int = 8):
        self.scorer = scorer
        self.name = scorer.name
        self._memo: OrderedDict = OrderedDict()
        self._size = size

    def score(self, issue: str, files: list[ParsedFile]):
        key = (hash(issue), tuple(pf.path for pf in files), tuple(len(pf.lines) for pf in files))
        if key not in self._memo:
            self._memo[key] = self.scorer.score(issue, files)
            if len(self._memo) > self._size:
                self._memo.popitem(last=False)
        return self._memo[key]


class FileLoader:
    """Parses candidate files from the repository cache, keeping one git process per repo."""

    def __init__(self):
        self._repos: dict[str, Repo] = {}

    def repo(self, name: str) -> Repo:
        if name not in self._repos:
            self._repos[name] = Repo(name)
        return self._repos[name]

    def load(self, rec: dict, paths: list[str], cache: dict[str, ParsedFile]) -> list[ParsedFile]:
        repo = self.repo(rec["repo"])
        files = []
        for path in paths:
            if path not in cache:
                source = repo.read(rec["base_commit"], path)
                if source is None:
                    continue
                cache[path] = parse_source(source, path)
            files.append(cache[path])
        return files

    def close(self) -> None:
        for repo in self._repos.values():
            repo.close()
        self._repos.clear()


def run_path(split: str, setting: str, method: str) -> Path:
    return RUNS_DIR / split / setting / f"{method}.jsonl"


def evaluate(
    records: list[dict],
    methods: dict[str, object],
    split: str,
    settings: list[str],
    relative_budgets: list[float],
    fixed_budgets: list[int],
    force: bool = False,
    progress: bool = True,
    out_dir_for=run_path,
) -> None:
    """Run ``methods`` on ``records`` and write per-issue rows.

    A (setting, method) pair whose output file already exists is skipped unless
    ``force`` is set.
    """
    todo = {(s, m) for s in settings for m in methods
            if force or not out_dir_for(split, s, m).exists()}
    if not todo:
        print("nothing to do, all requested runs exist (use --force to redo them)")
        return
    handles = {}
    for setting, name in sorted(todo):
        path = out_dir_for(split, setting, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        handles[(setting, name)] = path.with_suffix(".jsonl.tmp").open("w")

    loader = FileLoader()
    records = sorted(records, key=lambda r: (r["repo"], r["base_commit"], r["instance_id"]))
    started = time.time()
    try:
        for i, rec in enumerate(records, start=1):
            file_cache: dict[str, ParsedFile] = {}
            parse_ok_cache: dict[str, bool] = {}
            for setting in settings:
                wanted = [m for m in methods if (setting, m) in todo]
                if not wanted:
                    continue
                files = loader.load(rec, candidate_paths(rec, setting), file_cache)
                if not files:
                    continue
                for pf in files:
                    if pf.path not in parse_ok_cache:
                        parse_ok_cache[pf.path] = parses("\n".join(pf.lines))
                parseable = {p for p, ok in parse_ok_cache.items() if ok}
                total = original_tokens(files)
                budgets = [("relative", r, int(r * total)) for r in relative_budgets]
                budgets += [("fixed", f, f) for f in fixed_budgets if total > f]
                for name in wanted:
                    method = methods[name]
                    t0 = time.perf_counter()
                    prepared = method.prepare(rec["issue"], files)
                    prep_ms = (time.perf_counter() - t0) * 1000.0
                    for kind, value, budget in budgets:
                        t1 = time.perf_counter()
                        ctx = method.prune(prepared, files, budget)
                        prune_ms = (time.perf_counter() - t1) * 1000.0
                        row = evaluate_instance(files, rec["gold_functions"], rec["gold_lines"], ctx, parseable)
                        row.update(
                            instance_id=rec["instance_id"], repo=rec["repo"], setting=setting, method=name,
                            budget=budget_label(kind, value), budget_tokens=budget,
                            over_budget=row["output_tokens"] > budget,
                            n_files=len(files), n_nodes=sum(len(pf.nodes) for pf in files),
                            prep_ms=round(prep_ms, 2), prune_ms=round(prune_ms, 2),
                        )
                        handles[(setting, name)].write(json.dumps(row) + "\n")
            if progress and (i % 25 == 0 or i == len(records)):
                print(f"  {i}/{len(records)} issues  ({time.time() - started:.0f}s)", flush=True)
    finally:
        loader.close()
        for fh in handles.values():
            fh.close()
    for setting, name in todo:  # only finished runs get their final name
        path = out_dir_for(split, setting, name)
        path.with_suffix(".jsonl.tmp").replace(path)


def _run_shard(args) -> None:
    """Worker entry point: build the methods in this process and evaluate one shard."""
    shard, records, names, config, device, split, settings, relative, fixed, todo = args
    from .registry import Registry  # imported here: registry imports this module

    methods = Registry(config, device=device).methods(names)

    def shard_path(sp: str, setting: str, method: str) -> Path:
        return run_path(sp, setting, method).with_suffix(f".shard{shard}.jsonl")

    for setting, name in todo:  # a stale shard file would make the run look finished
        shard_path(split, setting, name).unlink(missing_ok=True)
    wanted_settings = [s for s in settings if any(t[0] == s for t in todo)]
    wanted_methods = {n: m for n, m in methods.items() if any(t[1] == n for t in todo)}
    evaluate(records, wanted_methods, split, wanted_settings, relative, fixed,
             force=True, progress=shard == 0, out_dir_for=shard_path)


def evaluate_parallel(records: list[dict], names: list[str], config: dict, split: str, settings: list[str],
                      relative_budgets: list[float], fixed_budgets: list[int], workers: int,
                      force: bool = False, device: str | None = None) -> None:
    """Same as :func:`evaluate`, with the issues split across worker processes.

    Issues are sorted by repository and cut into contiguous shards, so each
    worker mostly stays inside a few repositories and its caches stay warm.
    Every worker loads its own copy of any model it needs.
    """
    import multiprocessing as mp

    todo = sorted((s, m) for s in settings for m in names if force or not run_path(split, s, m).exists())
    if not todo:
        print("nothing to do, all requested runs exist (use --force to redo them)")
        return
    records = sorted(records, key=lambda r: (r["repo"], r["base_commit"], r["instance_id"]))
    workers = max(1, min(workers, len(records)))
    bounds = [round(i * len(records) / workers) for i in range(workers + 1)]
    jobs = [(k, records[bounds[k]:bounds[k + 1]], names, config, device, split, settings,
             relative_budgets, fixed_budgets, todo) for k in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        pool.map(_run_shard, jobs)
    for setting, name in todo:
        final = run_path(split, setting, name)
        final.parent.mkdir(parents=True, exist_ok=True)
        with final.with_suffix(".jsonl.tmp").open("w") as out:
            for k in range(workers):
                shard = final.with_suffix(f".shard{k}.jsonl")
                if shard.exists():
                    out.write(shard.read_text())
                    shard.unlink()
        final.with_suffix(".jsonl.tmp").replace(final)


def load_runs(split: str, setting: str, methods: list[str] | None = None) -> dict[str, list[dict]]:
    """Per-issue rows of every finished run for a split and setting, keyed by method."""
    folder = RUNS_DIR / split / setting
    out: dict[str, list[dict]] = {}
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.jsonl")):
        if ".shard" in path.name:
            continue  # leftover of an interrupted parallel run
        if methods is None or path.stem in methods:
            out[path.stem] = [json.loads(line) for line in path.open()]
    return out
