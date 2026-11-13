# Progress log

One entry per phase: what ran, what broke, the numbers. Newest at the bottom.
Numbers here are copied from files under `results/`. If they ever disagree, the files win.

## Phase 0. Setup (Nov 13)

- `.venv` with Python 3.12.3. torch 2.15.0+cu132 sees the RTX 4060. Exact versions are in `requirements.txt`.
- `git init`, `.gitignore`, folders, plan copied to `docs/PLAN.md`.
- Agent definitions in `.claude/agents/` (phase-auditor, builder), both with `model: opus`. They are only picked up after a session restart, so the first audit ran as a general-purpose agent with `model: opus` and the same role text.

## Phase 1. Parser, token counter, selector (Nov 13)

- `src/astprune/parse.py`, `tokens.py`, `select.py`, `types.py`.
- Every line of a file belongs to exactly one node. Rendering with every node kept gives the file back byte for byte.
- The selector uses a cheap upper bound on cost, then checks the real token count and tops up.
- Problem found: tiktoken's batch call starts a thread pool per call and was the main cost of evaluation. Replaced with a plain loop.
- Problems found by the audit and fixed (see the audit entry below): pruned views that did not parse, and a 100% budget that did not return the whole file.

## Phase 2. Data and labels (Nov 13)

- Datasets from Hugging Face. SWE-bench train has 35 repositories (the proposal said about 37). SWE-Gym has 11.
- Repo overlap is checked in code by `experiments/build_train_examples.py`, which writes `results/data/leakage_check.json` and refuses to continue on any overlap. Train vs test, train vs dev and dev vs test share no repository. SWE-Gym and SWE-bench train share pandas and conan, duplicates are dropped when examples are built. Lite and Verified share 93 issues, which is expected: both are subsets of the SWE-bench test set.
- Bare clones with `--filter=blob:limit=2m`. The 12 test repos took about 2 minutes, all 53 repos take about 12 GB. One training repo (JohnSnowLabs/spark-nlp, 3 issues) did not finish cloning and was skipped.
- Problem found: some patches have hunk line numbers that are off against `base_commit` (git apply tolerates offsets). In Lite that is 8 patches. 4 were rejected by our check, and the other 4 would have matched at the wrong place and been mislabelled without anyone noticing (for example pytest-5103 would have marked `visit_BinOp` where the fix is in `visit_Call_35`). Verified has 11 such patches. Added hunk realignment, the same search `patch` does. All 300 Lite and 500 Verified issues now label cleanly.
- Problem found: pydicom ships data dictionaries of up to 2.7 MB as `.py` files and BM25 retrieved them for 54 dev issues, which made evaluation crawl. Files over 400,000 characters are no longer offered as retrieved or imported candidates. No gold file in dev, Lite or Verified is that large.

| split | issues | labelled | with a gold function | gold functions | mean per issue | no gold lines at all | realigned patches | nodes per gold file |
|---|---|---|---|---|---|---|---|---|
| lite | 300 | 300 | 274 | 317 | 1.16 | 14 | 8 | 72.2 |
| verified | 500 | 500 | 458 | 836 | 1.82 | 17 | 11 | 68.2 |
| dev | 225 | 223 | 209 | 810 | 3.88 | 0 | 1 | 39.8 |

- Train and SWE-Gym counts are added below once their rebuild with the corrected label rules finishes. Before the label fixes: SWE-Gym 2,353 of 2,438 labelled, SWE-bench train 16,837 of 19,008 labelled (2,117 patches touch no Python file, 51 patch mismatches, 3 clone failures).
- Issues with no gold function (the patch only touches module-level code or class attributes, or only adds new functions) are left out of the function metrics. Most of them still have gold lines and count for gold-line recall. The "no gold lines at all" column counts the ones that do not, because the patch only adds new code. Those are in no recall metric.
- tree-sitter parsed all gold files except 2 in Lite and 2 in Verified, which went through the `ast` fallback.
- BM25 file retrieval, share of issues with every gold file in the top 5 / 10 / 20 (test files and oversized files excluded): Lite 69.0 / 78.3 / 84.3. Verified 63.2 / 73.4 / 83.4. Dev 52.5 / 64.6 / 75.3.

## Audit of phases 0 to 2 (Nov 13)

Run by an opus subagent, read-only. Result: FAIL with 1 blocker, since fixed. What it found and what was done:

| Finding | Severity | Fix |
|---|---|---|
| Pruned view did not always parse: a long statement could be split at a trailing comment (`}  # type: ...`), and a class could be left with a comment as its only body | blocker | `parse.py` only cuts where something starts its own line. `select.py` no longer treats a comment-only block as a class body. Regression tests added. Rechecked on 577 real files and 840 random selections: 0 parse failures, 0 over budget |
| `select()` with a budget equal to the file size dropped nodes, because marker overhead was charged even when nothing is omitted | should-fix | If the unpruned files fit the budget, everything is kept. Tested |
| Labels missed a function when a multi-line decorator was added above it, when a blank line was replaced by code, and when new lines were appended right before a rewritten neighbour. About 0.6% of training issues and 1 dev issue. The audit's independent oracle (apply the patch, diff function text) agreed with our labels on 300/300 Lite and 500/500 Verified issues | should-fix | All three cases handled in `labels.py` with tests. Dev gained one gold function (809 to 810). Lite and Verified counts did not change |
| Gold lines missed for an attribute added to a class-level block that starts with a blank line, and for a continuation line added inside a multi-line import | nit | Fixed with tests |
| PROGRESS said issues without a gold function still count for gold-line recall. Many have no gold lines at all | should-fix | Stats now report the count (table above), wording corrected |
| Stats lacked nodes per file, leakage file was not committed, no `requirements.txt`, no checkboxes in the plan, stale BM25 numbers, an em-dash in the plan | should-fix / nit | All added or corrected |

Everything downstream of labels and parsing (the dev evaluation and the training examples) was thrown away and rebuilt after these fixes.
