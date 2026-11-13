# Plan: Learned AST Pruning for Token-Budgeted Code Context (full project build)

## Status

Ticked when the phase exit criteria are met and the phase has been audited. Details per phase are in `docs/PROGRESS.md`.

- [x] Phase 0. Setup
- [x] Phase 1. Parser, token counter, selector
- [x] Phase 2. Data and labels
- [ ] Phase 3. Untrained methods and the evaluation harness
- [ ] Phase 4. Train the ranker
- [ ] Phase 5. External baselines
- [ ] Phase 6. Scaling, latency, error analysis
- [ ] Phase 7. MCP server and demo
- [ ] Phase 8. Stretch: downstream effect
- [ ] Phase 9. README, tutorial, notebook, reproduction check
- [ ] Phase 10. Report and slides

## Context

Course project for CSCI-GA 3033-091 (Anant Kumar, Marcellanous Ellonye). Final proposal (Oct 30) promises: a small trained ranker (UniXcoder-base, InfoNCE) that picks which functions/classes of a Python file an agent sees under a token budget, evaluated on SWE-bench Lite without running tests, compared against training-free methods, and shipped as an MCP server with a live demo.

Where things stand today (Nov 13): the project folder only has the proposal PDFs, the rubric PDF and the slides. No code, no git repo, no .venv. The proposal timeline had the parser and untrained baselines done by Nov 4, so we are behind and this plan does everything from scratch. Report and repo are due **Dec 15, 2026**, late = zero.

Grading: Presentation 30, Demo 30, Repo 15, Report 15, Proposal 10 (lost entirely if the final project drifts from the proposal). So everything below stays inside the proposal. Additions are extra evidence, never a change of topic.

Machine: WSL2, RTX 4060 Laptop 8 GB, 16 cores, 11 GB RAM, 890 GB disk, Python 3.12, git, pdflatex + bibtex (no IEEEtran/acmart class installed). Not installed: gh, uv, ollama, docker.

## Assumptions to confirm when you review this

1. Your message cut off at "also you will be making". I am assuming you meant the report and slides. If it was something else, tell me.
2. I will write the final report (IEEE LaTeX, PDF) and the presentation deck (.pptx) as well as the repo. Strike phases 10 if you want to write those yourselves.
3. Git: `git init` locally and commit at the end of every phase. I will not push anywhere. `gh` is not installed, so you add the GitHub remote and push when ready.
4. The rubric PDF and course slides stay out of the git repo (gitignored). Your own proposal PDFs go in `docs/proposal/`.

## Things I would change or add to the proposal

All small, all inside the proposed topic. Each one closes a hole a TA could poke at.

| # | Change | Why |
|---|--------|-----|
| 1 | Use the SWE-bench `dev` split (225 issues, 6 repos) as a validation set for every tuning decision | Rubric asks for "separation of evaluation data from development decisions". Lite stays untouched until the final run |
| 2 | Add **gold-line recall** (share of edited pre-fix lines that survive) next to gold-function recall | The proposal rule "line pruners only count if all lines survive" can look stacked against line and chunk methods. A granularity-neutral metric makes H1 credible |
| 3 | Add zero-shot UniXcoder (same encoder, no fine-tuning) as the main training ablation, next to BM25 | Cleanest answer to RQ2: only the weights differ |
| 4 | Add a hybrid scorer (reciprocal rank fusion of BM25 + trained ranker) and a random-ranking floor | Costs nothing. Hybrid is often the real winner, and if it is we should say so |
| 5 | Pooled budget across candidate files as the primary setup, per-file budget as an ablation | Proposal is ambiguous. Pooled lets a good ranker spend budget on the right file |
| 6 | Report a parseable-output rate (`ast.parse` on the pruned view) | Backs the "never mid-function" claim with a number |
| 7 | SWE-bench Verified (500 issues) as a second test set, Lite stays primary | Same 12 repos so no extra cloning. More issues means tighter CIs, and multi-function patches make the all-gold rate meaningful |
| 8 | Cite SWEzze (arXiv 2603.28119, Mar 2026) in related work | Trained compressor for issue resolution, very close to us. Better we bring it up than a TA does |
| 9 | Selector variants as ablations: no signature tier, and score-per-token greedy | Shows which part of the selector actually helps |
| 10 | Demo agent = Claude Code calling our MCP server through `.mcp.json`, with a scripted MCP client CLI as the reliable path | A real agent, zero API code to maintain, and a fallback that cannot hang on stage |

Verified while planning:
- SWE-bench on HF: train 19k, dev 225, test 2.29k. SWE-Gym: 2,438 issues, 11 repos, MIT.
- SWE-Pruner checkpoint is released: `ayanami-kitasan/code-pruner` (0.6B), should fit in 8 GB.
- LongCodeZip: `pip install git+https://github.com/YerbaPage/LongCodeZip.git`, MIT, default scorer is Qwen2.5-Coder-7B. On 8 GB we run it 4-bit or swap in Qwen2.5-Coder-1.5B and say so.
- UniXcoder-base: 1026 positions, matches the proposal's 1,024 token limit.
- MCP Python SDK is on v2 now (`MCPServer` class, Python 3.11+). I will check the installed package's API and pin the version instead of trusting old FastMCP snippets.

## Repo layout

Follows the rubric's suggested structure. Demo imports from `src/`, nothing is duplicated.

```
README.md  requirements.txt  pyproject.toml  .env.example  .gitignore  .mcp.json
src/astprune/
  parse.py          tree-sitter nodes (function, method, class header, module block), ast fallback
  tokens.py         one token counter used by every method (tiktoken cl100k_base)
  select.py         budget fill: full / signature / drop, renders valid Python
  scorers/          bm25.py  encoder.py (zero-shot + trained)  hybrid.py  random.py
  baselines/        line_window.py  chunk_rag.py  skeleton.py  longcodezip.py  swe_pruner.py
  data/             repos.py (bare clones, file at commit)  labels.py (patch -> gold nodes)
                    candidates.py (oracle files + direct imports, BM25 top-k files)
  train.py          InfoNCE training loop
  metrics.py  stats.py (paired bootstrap)
  server.py         MCP server: get_budgeted_context(issue, files, budget)
demo/               app.py (MCP client CLI)  sample_inputs.json  sample_repo/
experiments/        config.yaml  build_labels.py  train_ranker.py  evaluate.py
                    scaling.py  downstream.py  make_figures.py
results/            metrics JSON/CSV, figures, per-issue outputs, run manifests
notebooks/          analysis.ipynb (regenerates every figure from results/)
docs/               PLAN.md  PROGRESS.md  architecture.md  tutorial.md  ta_feedback.md
                    error_analysis.md  proposal/
tests/              unit, integration, MCP smoke, demo smoke
data/README.md      how to get the data, nothing large committed
report/             IEEE LaTeX source + PDF
slides/             build_slides.py + .pptx
.claude/agents/     phase-auditor.md  builder.md
```

## Subagents

You asked that any agent's YAML says opus before it is spun up. So, in Phase 0, before any spawn:

- `.claude/agents/phase-auditor.md` with frontmatter `model: opus`, read-only tools. Runs after each phase: rereads `docs/PLAN.md`, checks the phase exit criteria by running the tests and opening the result files, and flags any sentence in README/report that the files in `results/` do not back up.
- `.claude/agents/builder.md` with frontmatter `model: opus`. Used only for side work that can run while the GPU is busy (baseline adapters, docs drafts).

Every Agent call also passes `model: "opus"` explicitly. If the new agent files are not picked up without a restart, I use the built-in general-purpose agent with `model: "opus"` and the same role prompt. I will keep agent use light: the GPU is one 8 GB card, so most work is serial anyway.

## Phases

After each phase: run `pytest`, tick the box in `docs/PLAN.md`, append to `docs/PROGRESS.md` (what ran, what broke, numbers), git commit, auditor pass, then reread the plan before starting the next phase.

### Phase 0. Setup
- `python3 -m venv .venv`, upgrade pip, install torch (CUDA wheel), transformers, datasets, tree-sitter + tree-sitter-python, rank-bm25, tiktoken, unidiff, mcp, pytest, matplotlib, pandas, pyyaml, rich. Freeze exact versions into `requirements.txt`.
- `git init`, `.gitignore` (.venv, caches, clones, checkpoints, course PDFs), copy this plan to `docs/PLAN.md` with checkboxes, write the two agent files, save your writing-style preference to memory.
- Exit: `python -c "import torch; print(torch.cuda.is_available())"` prints True inside .venv.

### Phase 1. Parser, token counter, selector
- `parse.py`: nodes are top-level functions, methods, class headers (signature + docstring + class attributes) and contiguous module-level blocks (imports, constants). Nested functions stay inside their parent. Decorators belong to their node. Falls back to `ast` when tree-sitter reports errors, and counts how often.
- `select.py`: greedy fill in score order. Top nodes whole, next ones as signature (`def f(...): ...`), rest dropped with a one-line `# ... N lines omitted` marker. A kept method always keeps its class line. Output keeps file order.
- Exit: unit tests pass, including the invariants "never over budget", "output parses", "round trip at 100% budget equals the original file".

### Phase 2. Data and labels
- Download SWE-bench (train, dev, test), SWE-bench Lite, Verified, SWE-Gym from HF.
- Bare-clone the repos (12 test, 6 dev, about 48 train) into a gitignored cache and read pre-fix files with `git show <base_commit>:<path>`. Fetch a missing commit by SHA, else skip and count it. Fallback if cloning is too slow: `princeton-nlp/SWE-bench_oracle`, which ships oracle file text.
- `labels.py`: parse the gold patch with `unidiff`, map changed pre-fix lines to the innermost node. Positives = changed existing nodes. Added functions and non-Python edits are excluded, as in the proposal. Issues with no gold function are dropped from the function metrics and the count is reported.
- `candidates.py`: oracle files plus direct imports (resolved from import statements), and BM25 top-k files over the repo at base commit for k = 5, 10, 20.
- Leakage check in code: assert zero repo overlap between train (SWE-bench train + SWE-Gym) and the test repos, drop anything that overlaps, write the lists to `results/leakage_check.json`.
- Exit: label stats table (issues, gold functions per issue, nodes per file, parse-failure rate) written to `results/`, tests for label mapping on hand-made diffs pass.

### Phase 3. Untrained methods and the evaluation harness (proposal "must-have" tier)
- Methods: line window, chunk RAG (256-token chunks, BM25 and embedding scorers), Agentless-style skeleton, AST + BM25, AST + zero-shot UniXcoder, AST + random.
- `evaluate.py` driven by `experiments/config.yaml`: settings (oracle, retrieved), budgets (10/25/50% of candidate tokens, plus fixed 1k/2k/4k on issues that exceed them, with the surviving issue count reported).
- Metrics: gold-function recall (primary), all-gold rate, signature recall, gold-token precision, actual compression, gold-line recall, parseable rate, latency per file.
- `stats.py`: paired bootstrap over issues, 10k resamples, 95% CI for every method pair. H1 and H2 comparisons are named up front in the config, the rest are labelled exploratory.
- Exit: first recall-vs-budget figure on the dev split and on Lite oracle for the untrained methods. This alone is a complete fallback result, as the proposal says.

### Phase 4. Train the ranker
- Bi-encoder with shared UniXcoder-base weights, mean pooling, InfoNCE (temperature 0.05). Query = issue title + opening text incl. traceback. Node text = parent class, signature, docstring, first body lines. Negatives = other nodes from the same file (hard), nodes from direct imports, plus in-batch.
- Mixed precision, gradient accumulation, lr 2e-5, 2 to 3 epochs. Rough estimate on the 4060: 1 to 2 hours per epoch. Runs in the background with logs and checkpoints saved.
- Model selection on the dev split only (recall at 25% budget). Then one final Lite + Verified run.
- Ablations: filtered vs unfiltered training patches (drop patches touching many functions), selector variants, per-file vs pooled budget, hybrid BM25 + trained, zero-shot vs trained. Two extra seeds if GPU time allows, to show training variance.
- Exit: checkpoint saved, results tables with CIs for trained vs untrained in both settings. If trained does not beat BM25, that gets reported as the finding, with a breakdown of which issue types each scorer wins on.

### Phase 5. External baselines
- LongCodeZip through its own package, both `rank_only` and full two-stage, with the perplexity model that fits in 8 GB (recorded in config).
- SWE-Pruner's released checkpoint, adapted to a budget by keeping its top-scored lines until the budget is hit.
- Both are time-boxed. If one cannot run on this GPU, or is too slow for all settings, I run it on the oracle setting only (or a fixed random subset) and write that limit down. No silent skipping.
- Exit: both appear in the main results table or have a documented reason why not.

### Phase 6. Scaling, latency, error analysis
- Candidate set 5 to 10 to 20 BM25 files: recall and latency at each size. ms per file on GPU and CPU, parameter counts.
- Error analysis on at least 20 dropped-gold cases for our method and 20 for the strongest baseline, grouped by cause (vague issue, fix far from the code the issue names, function too big for the budget, other). Three become case studies. Written to `docs/error_analysis.md`.

### Phase 7. MCP server and demo (30% of the grade, so it gets real testing)
- `server.py`: tool `get_budgeted_context(issue, files, budget)`, model loaded once at startup, CPU fallback, returns pruned text plus token stats. Clear errors for missing files, bad budgets, unparseable files (falls back to ast, then to a line window with a warning) and empty issues (returns the skeleton view with a note).
- `demo/app.py`: a real MCP client that starts the server over stdio, runs a sample SWE-bench issue from `sample_inputs.json`, and prints full vs pruned side by side with token counts, kept/signature/dropped table and latency. One command to launch.
- `.mcp.json` so Claude Code can call the tool live as the agent.
- Reliability evidence: the smoke test runs the demo path 20 times and records success count and latency percentiles in `results/demo_reliability.json`.
- Exit: demo smoke test green, bad-input tests green, `docs/demo_script.md` with the 3 to 4 minute sequence and a note on recording a backup video (you record it, I cannot).

### Phase 8. Stretch: downstream effect
- Qwen2.5-Coder-7B-Instruct 4-bit (transformers + bitsandbytes so it stays inside .venv, 1.5B if the 7B does not fit with long prompts) writes a search/replace patch from each method's context on a fixed subset of Lite at 25% budget.
- Metrics: does the patch edit a gold function, and undefined-symbol rate (static check of called names against the repo). No tests run, as proposed.
- Time-boxed. Cut first if anything upstream slips.

### Phase 9. README, tutorial, notebook, reproduction check
- README written plainly: what it is, results table with the real numbers, install, run the demo, rerun experiments, hardware and time costs, limits, troubleshooting. Short sentences, no em-dashes, no hype words. A grep for the em-dash character over README, docs and report must come back empty.
- `docs/tutorial.md`, `docs/architecture.md`, `docs/ta_feedback.md` (TA asked for related literature and a metrics plan, and where each was addressed), `data/README.md`, `.env.example`.
- `notebooks/analysis.ipynb` regenerates every figure from `results/`.
- Exit: clone the repo into a scratch folder, make a fresh venv, follow the README commands exactly, demo and tests work.

### Phase 10. Report and slides
- Report in IEEE conference format (IEEEtran.cls fetched from CTAN into `report/`), sections mapped to the rubric: related work with critical comparison, methodology, results with CIs, ablations, case studies, limitations, TA feedback log. Every number is pulled from `results/` files.
- Slides built with python-pptx around the three graded blocks: problem statement, contributions, evaluation. About 12 slides.
- Final auditor pass against the rubric PDF, page by page.

## Rules I will hold to

- Lite and Verified test numbers are produced once, after choices are frozen on dev.
- Every number in README, report and slides comes from a file in `results/` written by a script in the repo. Nothing typed by hand.
- Negative results are reported as they are. The proposal already says matching BM25 is a valid finding.
- Seeds, model versions, prompts and configs are saved with each run.
- No secrets committed. No API keys needed anywhere.

## Verification

- `pytest` (unit + integration + MCP smoke + demo smoke) green after every phase.
- `python experiments/evaluate.py --config experiments/config.yaml` reproduces the main table. `python experiments/make_figures.py` rebuilds the figures.
- `python demo/app.py` runs end to end on the sample input, and Claude Code can call `get_budgeted_context` through `.mcp.json`.
- Fresh-clone test in Phase 9.
- Auditor agent report after each phase saved in `docs/PROGRESS.md`.

## Main risks

| Risk | What I do |
|------|-----------|
| Cloning about 60 repos is slow or commits are missing | Run in background, fetch by SHA, fall back to SWE-bench_oracle text |
| Training slower than estimated | Cap node length and negatives per issue, train on a subset first, untrained comparison already stands alone |
| LongCodeZip or SWE-Pruner do not run on 8 GB | Smaller scorer model or subset, documented |
| 11 GB RAM in WSL | Stream datasets, encode in batches, cache embeddings to disk |
| MCP SDK v2 API differs from older examples | Read the installed package, pin version, smoke test |
| Running out of calendar time | Order above is priority order. Phase 8 goes first, then extra seeds, then Verified |
