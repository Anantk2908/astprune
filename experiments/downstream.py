"""Stretch experiment: does the pruned context change what a local code model writes?

    python experiments/downstream.py --split lite --n 100
    python experiments/downstream.py --split lite --n 100 --model Qwen/Qwen2.5-Coder-1.5B-Instruct --no-4bit --tag qwen1.5b

For a fixed sample of issues, a local model gets the issue and one view of the
gold file (the full file, or the file pruned by one of the methods) and writes
SEARCH/REPLACE edits. No tests are run. Two things are measured statically:

* does an edit land in a function the gold patch changed
* undefined-symbol rate: does the new code call a function that exists
  neither in the repository, nor in the file, nor in the edit itself

Everything runs locally. Output goes to results/downstream/<tag>/.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import random
import re
import sys
import textwrap
import time

import torch

from astprune.data.datasets import load_processed
from astprune.data.repos import PROJECT_ROOT, Repo
from astprune.evaluation import FileLoader, candidate_paths
from astprune.metrics import original_tokens
from astprune.parse import ParsedFile, quiet_ast_parse
from astprune.registry import Registry, load_config
from astprune.stats import mean_ci, paired_diff
from astprune.types import PrunedFile

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SYSTEM_PROMPT = (
    "You fix bugs in Python repositories. You are given an issue and code from the repository. "
    "Some code has been left out. Left-out parts are marked with a comment such as "
    "'# ... 40 lines omitted' or with a function body of '...'."
)
# The format is described after the code, next to where the answer starts. An
# example edit in the prompt gets copied word for word by small models.
USER_TEMPLATE = (
    "Issue:\n{issue}\n\nCode:\n{context}\n\n"
    "Write the edits that fix this issue. Use this format for each edit and write nothing else:\n\n"
    "<path of the file>\n<<<<<<< SEARCH\n<a few lines copied exactly from the code above>\n=======\n"
    "<the lines that should replace them>\n>>>>>>> REPLACE\n\n"
    "Only edit code that is shown above."
)

_BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)
_DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.M)
_IDENT = re.compile(r"[A-Za-z_]\w*")
BUILTIN_NAMES = set(dir(builtins))


# --------------------------------------------------------------------------
# reading the model's answer
# --------------------------------------------------------------------------

def parse_edits(answer: str) -> list[tuple[str, str]]:
    """(search text, replace text) for every well-formed edit block."""
    return [(s, r) for s, r in _BLOCK.findall(answer.replace("\r\n", "\n"))]


def locate(search: str, pf: ParsedFile) -> tuple[int, int] | None:
    """Lines of the original file that a SEARCH block matches, ignoring indentation and blank lines."""
    want = [ln.strip() for ln in search.split("\n") if ln.strip()]
    if not want:
        return None
    have = [(i, ln.strip()) for i, ln in enumerate(pf.lines, start=1) if ln.strip()]
    for start in range(len(have) - len(want) + 1):
        if all(have[start + k][1] == w for k, w in enumerate(want)):
            return have[start][0], have[start + len(want) - 1][0]
    return None


def called_names(code: str) -> tuple[set[str], set[str], bool]:
    """Names called as ``f(...)`` or ``self.f(...)``, names the code defines itself, and whether it parsed."""
    tree = None
    body = textwrap.dedent(code)
    for candidate in (body, "def _wrapper_():\n" + textwrap.indent(body, "    ")):
        try:
            tree = quiet_ast_parse(candidate)
            break
        except (SyntaxError, ValueError, RecursionError):
            continue
    if tree is None:
        return set(), set(), False
    calls: set[str] = set()
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.add(fn.id)
            elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in ("self", "cls"):
                calls.add(fn.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
    calls.discard("_wrapper_")
    return calls, defined, True


class RepoSymbols:
    """Every function and class name defined anywhere in a repository snapshot."""

    def __init__(self):
        self._by_blob: dict[str, frozenset[str]] = {}

    def names(self, repo: Repo, commit: str) -> set[str]:
        out: set[str] = set()
        for path, sha in repo.tree(commit).items():
            if sha not in self._by_blob:
                self._by_blob[sha] = frozenset(_DEF.findall(repo.read_blob(sha) or ""))
            out |= self._by_blob[sha]
        return out


def score_answer(answer: str, files: list[ParsedFile], gold_functions: list[dict], repo_names: set[str]) -> dict:
    edits = parse_edits(answer)
    file_idents: set[str] = set()
    for pf in files:
        file_idents.update(_IDENT.findall("\n".join(pf.lines)))
    found = hits_gold = 0
    undefined: set[str] = set()
    n_calls = 0
    unparseable = 0
    for search, replace in edits:
        span = None
        for pf in files:
            span = locate(search, pf)
            if span:
                found += 1
                if any(g["path"] == pf.path and g["start"] <= span[1] and span[0] <= g["end"] for g in gold_functions):
                    hits_gold += 1
                break
        calls, defined, ok = called_names(replace)
        if not ok:
            unparseable += 1
            continue
        known = BUILTIN_NAMES | repo_names | file_idents | defined
        n_calls += len(calls)
        undefined |= {c for c in calls if c not in known}
    return {
        "n_edits": len(edits),
        "has_edit": float(bool(edits)),
        "search_found": found / len(edits) if edits else 0.0,
        "edits_gold_function": float(hits_gold > 0),
        "n_calls": n_calls,
        "undefined_symbols": sorted(undefined),
        "has_undefined_symbol": float(bool(undefined)),
        "replace_unparseable": unparseable,
    }


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def load_generator(model_name: str, four_bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs: dict = {"dtype": torch.float16}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not four_bit:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model.eval()


@torch.no_grad()
def generate(tokenizer, model, issue: str, context: str, max_new_tokens: int) -> tuple[str, int, float]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(issue=issue, context=context)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    seconds = time.perf_counter() - started
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, int(inputs["input_ids"].shape[1]), seconds


def context_text(files: list[ParsedFile], ctx: dict[str, PrunedFile]) -> str:
    return "\n\n".join(f"### {pf.path}\n{ctx[pf.path].text}" for pf in files if ctx[pf.path].text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--n", type=int, default=100, help="issues sampled with a fixed seed")
    ap.add_argument("--methods", nargs="+", default=["ast_trained", "ast_bm25", "chunk_bm25", "line_window"])
    ap.add_argument("--setting", default="oracle")
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--max-budget-tokens", type=int, default=4000, help="cap on the pruned view")
    ap.add_argument("--max-full-tokens", type=int, default=12000, help="full file is only tried below this size")
    ap.add_argument("--max-issue-chars", type=int, default=4000)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-4bit", action="store_true", help="load in fp16 instead of 4-bit")
    ap.add_argument("--tag", default="qwen7b_4bit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "results" / "downstream" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {(r["instance_id"], r["condition"]) for r in map(json.loads, rows_path.open())}

    records = [r for r in load_processed(args.split) if r["gold_functions"]]
    random.Random(args.seed).shuffle(records)
    records = records[: args.n]
    registry = Registry(load_config())
    methods = registry.methods(args.methods)
    tokenizer, model = load_generator(args.model, not args.no_4bit)
    loader, symbols = FileLoader(), RepoSymbols()

    with rows_path.open("a") as fh:
        for i, rec in enumerate(records, start=1):
            files = loader.load(rec, candidate_paths(rec, args.setting), {})
            if not files:
                continue
            total = original_tokens(files)
            budget = min(int(args.budget * total), args.max_budget_tokens)
            repo_names = None
            conditions: dict[str, dict[str, PrunedFile]] = {}
            if total <= args.max_full_tokens:
                conditions["full"] = {pf.path: PrunedFile(pf.path, "\n".join(pf.lines)) for pf in files}
            for name, method in methods.items():
                conditions[name] = method.prune(method.prepare(rec["issue"], files), files, budget)
            for name, ctx in conditions.items():
                if (rec["instance_id"], name) in done:
                    continue
                if repo_names is None:
                    repo_names = symbols.names(loader.repo(rec["repo"]), rec["base_commit"])
                try:
                    answer, prompt_tokens, seconds = generate(
                        tokenizer, model, rec["issue"][: args.max_issue_chars], context_text(files, ctx),
                        args.max_new_tokens)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    fh.write(json.dumps({"instance_id": rec["instance_id"], "condition": name, "error": "oom"}) + "\n")
                    continue
                row = score_answer(answer, files, rec["gold_functions"], repo_names)
                row.update(instance_id=rec["instance_id"], condition=name, prompt_tokens=prompt_tokens,
                           file_tokens=total, budget_tokens=budget, seconds=round(seconds, 2), answer=answer)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(f"  {i}/{len(records)} issues", flush=True)
    loader.close()
    summarise(out_dir, args)


def summarise(out_dir, args) -> None:
    rows = [json.loads(line) for line in (out_dir / "rows.jsonl").open()]
    rows = [r for r in rows if "error" not in r]
    by_cond: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], {})[r["instance_id"]] = r
    metrics = ["has_edit", "search_found", "edits_gold_function", "has_undefined_symbol"]
    summary = {"model": args.model, "four_bit": not args.no_4bit, "split": args.split, "setting": args.setting,
               "budget": args.budget, "max_budget_tokens": args.max_budget_tokens, "conditions": {}, "paired": []}
    for cond, items in by_cond.items():
        vals = list(items.values())
        summary["conditions"][cond] = {
            "n": len(vals),
            "mean_prompt_tokens": sum(v["prompt_tokens"] for v in vals) / len(vals),
            "mean_seconds": sum(v["seconds"] for v in vals) / len(vals),
            **{m: mean_ci([v[m] for v in vals]) for m in metrics},
        }
    if "ast_trained" in by_cond:
        for cond, items in by_cond.items():
            if cond == "ast_trained":
                continue
            ids = sorted(set(items) & set(by_cond["ast_trained"]))
            for m in ("edits_gold_function", "has_undefined_symbol"):
                res = paired_diff([by_cond["ast_trained"][i][m] for i in ids], [items[i][m] for i in ids])
                summary["paired"].append({"a": "ast_trained", "b": cond, "metric": m, **res})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# Downstream check ({args.model}, {args.split}, {args.setting})", "",
             "| context | issues | prompt tokens | edit found in file % | edits a gold function % | calls an undefined symbol % |",
             "|---|---|---|---|---|---|"]
    for cond, s in summary["conditions"].items():
        cell = lambda m: f"{s[m]['mean'] * 100:.1f} [{s[m]['lo'] * 100:.1f}, {s[m]['hi'] * 100:.1f}]"  # noqa: E731
        lines.append(f"| {cond} | {s['n']} | {s['mean_prompt_tokens']:.0f} | {cell('search_found')} | "
                     f"{cell('edits_gold_function')} | {cell('has_undefined_symbol')} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
