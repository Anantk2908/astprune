# Progress log

One entry per phase: what ran, what broke, the numbers. Newest at the bottom.

## Phase 0. Setup (Nov 13)

- `.venv` with Python 3.12.3. torch 2.15.0+cu132 sees the RTX 4060. transformers 5.23.0, datasets 5.2.0, mcp 2.3.0, tree-sitter 0.26.1, tiktoken 0.14.0.
- `git init`, `.gitignore`, folders, plan copied to `docs/PLAN.md`.
- Agent definitions in `.claude/agents/` (phase-auditor, builder), both with `model: opus`.

## Phase 1. Parser, token counter, selector (Nov 13)

- `src/astprune/parse.py`, `tokens.py`, `select.py`, `types.py`.
- Every line of a file belongs to exactly one node, so a file at 100% budget renders back byte for byte. Tested.
- Selector uses a cheap upper bound on cost, then checks the real token count and tops up. Tested: never over budget, pruned view always parses.
- Problem found: tiktoken's batch call starts a thread pool per call and was the main cost of evaluation. Replaced with a plain loop.

## Phase 2. Data and labels (Nov 13)

- Datasets from Hugging Face. SWE-bench train has 35 repositories (the proposal said about 37). SWE-Gym has 11.
- Repo overlap checked in code: train vs test = none, train vs dev = none, dev vs test = none. SWE-Gym and SWE-bench train share pandas and conan, duplicates are dropped when examples are built.
- Bare clones with `--filter=blob:limit=2m`. The 12 test repos took about 2 minutes.
- Problem found: 4 Lite patches (all sympy) have hunk line numbers that are off against `base_commit` (git apply tolerates offsets). Added hunk realignment, the same search `patch` does. After that 300/300 Lite issues label cleanly, 8 of them needed realignment.

| split | issues | labelled | with a gold function | gold functions | mean per issue |
|---|---|---|---|---|---|
| lite | 300 | 300 | 274 | 317 | 1.16 |
| verified | 500 | 500 | 458 | 836 | 1.83 |
| dev | 225 | 223 | 209 | 809 | 3.87 |
| gym | 2438 | 2353 | 2256 | 8011 | 3.55 |

- Issues with no gold function (the patch only touches module-level code, class attributes, or adds new functions) are left out of the function metrics. They still count for gold-line recall.
- tree-sitter parsed all gold files except 2 in Lite and 2 in Verified, which went through the `ast` fallback.
- BM25 file retrieval, share of issues with every gold file in the top k (test files excluded): Lite 69.0 / 78.3 / 84.7 at k = 5 / 10 / 20. Verified 62.8 / 73.6 / 83.2. Dev 52.9 / 64.1 / 74.9.
