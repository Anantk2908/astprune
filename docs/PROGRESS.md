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
- Bare clones with `--filter=blob:limit=2m`. The 12 test repos took about 2 minutes, all 53 repos take about 15 GB. One training repo (JohnSnowLabs/spark-nlp, 3 issues) did not finish cloning and was skipped.
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

## Phase 3. Untrained methods and the evaluation harness (Nov 13)

- Methods: line window, chunk RAG (BM25 and zero-shot UniXcoder), Agentless-style skeleton, AST + random, AST + BM25, AST + zero-shot UniXcoder. Harness in `src/astprune/evaluation.py`, tables from `experiments/summarize.py` and `make_tables.py`.
- Problem found: with several worker processes each holding the encoder, the 8 GB card overflowed into shared memory and everything slowed down about tenfold. Encoder methods now run in one process, CPU methods run in parallel without the GPU.
- Problem found: the skeleton baseline scored 86% at the 50% budget on dev. Candidate files were listed gold files first, and the skeleton fills bodies in file order, so it was being handed the answer. Oracle settings now list files alphabetically. The skeleton dropped to 40%.
- Deviation from the plan: the plan asked for a first Lite figure in this phase. No test-set run was made until the choices below were frozen.
- Dev results are in `results/tables/dev_*.md`.

## Phase 4. Training (Nov 13)

- 17,325 examples from 44 repositories (SWE-bench train 16,840 labelled issues, SWE-Gym 2,353, 75 duplicates dropped, issues without a function or class positive dropped). 14,534 after dropping patches that touch more than 5 functions. At most 1,500 issues per repository per epoch, so pandas does not dominate.
- UniXcoder-base, InfoNCE, temperature 0.05, 8 issues per batch with 1 positive, 8 same-file negatives (half picked by BM25) and 3 import negatives each, lr 2e-5, 2 epochs, mixed precision.
- Problem found: this batch needs 12 GB without gradient checkpointing and 3.7 GB with it. Enabled.
- Problem found: a laptop restart killed the first run. Training now saves its full state at every dev check and resumes with `--resume`.
- Run `main`: 78 minutes on the RTX 4060. Dev recall (gold files + imports) at 10% / 25% budget went from 26.2 / 48.5 before training to 60.1 / 79.1 at step 2,700, the best check. Log in `results/training/main.json`.

## Choices frozen on dev before any test run (Nov 13)

Everything below was decided from `results/tables/dev_*.md`. The commit that adds this section is the last one before the first run on Lite or Verified.

| Choice | Decision | Dev evidence (recall at 10% / 25%, gold files + imports) |
|---|---|---|
| Checkpoint | run `main`, step 2,700 | best mean dev recall of 7 checks |
| Scorer shipped in the server | trained ranker alone | 60.1 / 79.1, against 57.0 / 74.3 for rank fusion with BM25 |
| Budget | pooled over candidate files | 60.1 / 79.1, against 34.8 / 53.4 when split per file |
| Signature tier | on, nothing reserved | 60.1 / 79.1 with it, 60.2 / 79.3 without, and signature recall 66.9 against 60.2 at 10% |
| Order | by score | 60.1 / 79.1, against 59.4 / 78.5 for probability per token |

The rank fusion with BM25 was our own suggested improvement to the proposal. It did not help and is reported as a negative result.

## Status note, Nov 14, about 09:50 (written before a laptop restart)

Done and committed:
- Lite and Verified test runs for all our methods and ablations. Tables, figures, README tables and `report/numbers.tex` rebuilt from them.
- SWE-Pruner (released checkpoint) on Lite, gold files and gold files plus imports, all 300 issues. Took 4.5 hours.
- LongCodeZip ranking-only mode on a fixed 100-issue sample of Lite (`--sample 100`, gold files only).
- `slides/build_slides.py` and `slides/presentation.pptx` (14 slides, 3 more appear when their results exist).
- Report section 7 (discussion, limitations, TA feedback, future work, conclusion).

Running when the laptop was restarted:
- LongCodeZip full mode on the same 100-issue sample, 3 hour cap. It writes to a `.tmp` file that is lost if the run is killed, so the rows finished so far were copied to `results/partial/lite_oracle_longcodezip_full_partial.jsonl`. The two LongCodeZip modes must be smoke-tested in separate processes, together they run out of GPU memory.

To continue after a restart:
1. `cd` to the project and run `setsid nohup scripts/run_queue.sh > /dev/null 2>&1 &`. Finished steps are skipped. Progress is in `logs/queue.log`.
2. Queue order from here: LongCodeZip full (100 issues, 3 h cap), scaling, CPU scaling, error analysis, perplexity baseline on Lite, downstream check with the 7B model, then `CORE_DONE`. After that: perplexity on dev, seeds `seed1`, `seed2` and the unfiltered-training run, SWE-Pruner and perplexity on Verified.

Still to write once those results exist:
- `report/sections/06_analysis.tex` (must define `\label{sec:external}`): third-party baselines with their exact conditions, perplexity baseline, scaling and latency, error analysis with case studies, downstream check, seed variance.
- `docs/baselines.md` and `docs/error_analysis.md` (the README links to both).
- Then: `summarize.py`, `make_tables.py`, `make_figures.py` for lite, verified and dev, `make_report_numbers.py`, `scripts/fill_docs.py`, `slides/build_slides.py`, `experiments/demo_reliability.py --runs 20`, `notebooks/build_notebook.py`, full `pytest`, tick the boxes in `docs/PLAN.md`.

Finding to report as it is: SWE-Pruner prunes lines, so on gold-function recall it is far below our method (54.6 against 94.3 at the 25% budget, gold files plus imports), but on gold-line recall it is level with ours in that setting (87.6 against 87.8) and better with gold files only at the 10% budget (58.9 against 45.9). Its views parse as Python in under 25% of cases.

Left for the team: put the 484 MB checkpoint on a release page and fill the link in the README, add the GitHub remote and push, record the backup demo video.

## Nov 14, evening: after the third restart

- The laptop's GPU driver crashed around 12:07 (a sticky CUDA error, the same time as the screen glitches), which also killed the CPU latency run. After the reboot the queue was trimmed and restarted. Dev perplexity runs were dropped, since nothing is tuned on them.
- LongCodeZip full mode hit its 3 hour cap after 42 of its 100 sampled issues. Those 42 (40 with a gold function) are kept as the run, with a manifest saying so. Its coarse stage finished all 100.
- New table `results/tables/lite_external.md` compares each third-party system with our ranker on exactly the issues it ran on. Sampled runs are kept out of the main tables and figures, since their averages are over different issues.
- Error analysis categories were rewritten. The first version had a "near miss" pile based on the rank of the last kept node, but tiny low-ranked nodes fill the last few tokens of the budget, so that rank says little. The rules are now: too large, only the signature fit, outranked with or without the issue naming the function. This also showed the README was wrong to call "too large for the budget" the most common miss. It is rare at 25%. Fixed.
- Written: `docs/baselines.md`, `docs/error_analysis.md`, `report/sections/06_analysis.tex`. Slides now have 17 slides.
- Fresh-clone check passed: clone into a scratch folder, new venv, `pip install -r requirements.txt && pip install -e .`, then `pytest` (all green) and `python demo/app.py --scorer bm25` on CPU.
- Error analysis at the 10% budget added (56 losses for ours, 129 for BM25). It corrected one claim: signature-only losses are more common for BM25 at 25% but level at 10%.

## Nov 14, night: perplexity baseline and downstream check

- AST + LLM perplexity (LongCodeZip's criterion on our nodes, Qwen2.5-Coder-1.5B) finished on all of Lite in both oracle settings. It is the best training-free ranker at the 10% and 25% budgets (82.1% at 25% with imports, BM25 77.1%), and the trained ranker is +12.2 [+7.7, +16.8] points ahead of it at 25%. Scoring takes about 1.8 s per file against 26 ms for ours. Added to the README, the results section of the report and slides 8 and 9. The error analysis still compares against BM25 and now says why.
- Downstream check with Qwen2.5-Coder-7B-Instruct in 4-bit on 100 Lite issues, gold files only, 25% budget capped at 4,000 tokens: an edit lands in a gold function for 35% of issues with our view, 29% with AST + BM25, 31% with chunk BM25, 23% with a line window, and 45% with whole files on the 64 issues where they fit (at three times the prompt). Only the gap to the line window is outside the noise (+12 [+1, +23]). Reported as a weak static proxy. New table `results/tables/lite_downstream_qwen7b_4bit.md`, report subsection `sec:downstream`, slide 15.
- Report: fixed doubled punctuation after paragraph headings and a 16 pt overfull table. Still 9 pages, no undefined references.
- Queue now on the extra training runs (seed1, seed2, unfiltered), then CPU latency and Verified runs for SWE-Pruner and perplexity.
