# Tutorial

From a fresh clone to a pruned view of your own files, then to rerunning the experiments. Every command here was run on our machine (WSL2 Ubuntu, Python 3.12, RTX 4060 Laptop GPU with 8 GB).

## 1. What you need

| Thing | Needed for | Notes |
|---|---|---|
| Python 3.11+ and git | everything | |
| About 3 GB of disk | install | mostly PyTorch |
| A CUDA GPU | training, fast evaluation | optional for the demo. The demo runs on CPU with BM25 ranking, or with the trained ranker at a few seconds per call |
| The trained checkpoint (484 MB) | trained ranking | see section 3. Without it the server uses BM25 and tells you |
| 2 GB more disk and network | rebuilding the evaluation data | 15 GB if you also rebuild the training data |
| Money | nothing | no paid API is called anywhere |

## 2. Install

```bash
git clone <this repository> && cd <folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q            # about two minutes, should end with all tests passing
```

There is nothing secret to configure. `.env.example` lists the optional settings (which folder file paths are relative to, which ranking to use, where caches go). Copy it to `.env` only if you want to change a default, and export the variables in your shell or MCP client config.

## 3. The checkpoint

Either download it from the repository's release page and unpack it so that `checkpoints/ranker/best/config.json` exists, or train it (section 7). Check it:

```bash
python -c "from astprune.service import PruningService; s = PruningService().load(); print(s.scorer_name, s.device_name, s.startup_notes)"
```

You want `trained cuda []`. If you see `bm25 cpu ['Trained ranker not available ...']` the checkpoint was not found, and everything below still works with BM25 ranking.

## 4. Run the demo

```bash
python demo/app.py --list        # the bundled sample issues
python demo/app.py               # sample 1 at a 25% budget
```

What you should see for sample 1 (an issue from the `requests` library, lines trimmed to fit):

```
Files the agent wants to read: requests/sessions.py, requests/compat.py, requests/cookies.py,
requests/models.py
╭─────────── Full files vs pruned view ────────────╮
│ 14,200 tokens if the agent reads the files whole │
│ 3,550 tokens in the pruned view (budget 3,550)   │
│ 10,650 tokens saved (75.0%)                      │
╰──────────────────────────────────────────────────╯
                                              Per file
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━
┃ file                 ┃ tokens ┃ shown ┃ kept in full                  ┃ signature only ┃ dropp
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━
│ requests/sessions.py │  4,215 │ 2,906 │ SessionRedirectMixin,         │              1 │
│                      │        │       │ SessionRedirectMixin.resolve… │                │
│                      │        │       │ Session, Session.__init__ (+4 │                │
│                      │        │       │ more)                         │                │
│ requests/compat.py   │    802 │     0 │ -                             │              0 │
│ requests/cookies.py  │  3,632 │   227 │ MockRequest.__init__,         │              3 │
│                      │        │       │ MockRequest.get_origin_req_h… │                │
│                      │        │       │ MockRequest.add_unredirected… │                │
│                      │        │       │ MockResponse.__init__         │                │
│ requests/models.py   │  5,551 │   417 │ Response.__init__,            │              0 │
│                      │        │       │ Response.is_redirect          │                │
└──────────────────────┴────────┴───────┴───────────────────────────────┴────────────────┴──────
Function the real fix changed: requests/sessions.py::SessionRedirectMixin.resolve_redirects -> k
in full
ranking: trained on cuda | server ready in 13.1s | pruning took 3188 ms
```

Below that the script prints the first 60 lines of the pruned code. The exact functions listed can differ slightly with BM25 ranking. The first number of milliseconds is larger on a cold start. Repeat calls to a running server take around 50 ms for these files.

Things to try:

```bash
python demo/app.py --sample 1 --budget 0.1      # a tenth of the tokens, the fixed function is still kept
python demo/app.py --sample 2 --budget 0.1      # the function is bigger than the whole budget: signature only
python demo/app.py --sample 2 --budget 0.25     # with more room it is kept in full
python demo/app.py --sample 1 --budget 2000     # a budget above 1 is a token count
python demo/app.py --sample 1 --scorer bm25     # no model at all
python demo/app.py --sample 1 --json            # the raw tool result
```

### Your own code

```bash
python demo/app.py --root /path/to/your/repo --files pkg/models.py pkg/views.py \
    --issue "Saving a model with an empty name raises KeyError in validate()"
```

`--issue-file issue.txt` reads the issue from a file. Paths are relative to `--root`, and files outside it are refused.

## 5. Use it from an agent over MCP

The server is `python -m astprune.server` and speaks MCP over stdio. `.mcp.json` in the repository root registers it for Claude Code:

1. Start Claude Code in the repository root. The `astprune` server shows as connected, with the tool `get_budgeted_context`.
2. Paste the prompt from `demo/agent_prompt.md`.
3. The agent calls the tool and gets back the pruned code with token counts.

For another MCP client, register this command and set `ASTPRUNE_ROOT` to the repository the agent works on:

```json
{"command": "/path/to/.venv/bin/python", "args": ["-m", "astprune.server"], "env": {"ASTPRUNE_ROOT": "/path/to/repo"}}
```

The tool returns `context` (the pruned code, one section per file), `original_tokens`, `kept_tokens`, `budget_tokens`, per-file lists of what was kept in full and as signatures, and `warnings`.

What happens on bad input:

| Input | Result |
|---|---|
| A file that does not exist, or lies outside the workspace | skipped, named in `warnings`, the other files are still processed |
| No readable file at all | tool error: "none of the files could be read" |
| Budget of 0, negative, or not a number | tool error that says what a valid budget looks like |
| Empty issue text | signatures first, then bodies in file order, with a warning |
| A file with syntax errors | still pruned, with a warning to check the result |
| A file that is not Python | its first lines, with a warning |
| No checkpoint | BM25 ranking, with a note |

In all of these the server keeps running. `tests/test_server_smoke.py` checks this over a real MCP connection.

## 6. Look at the results without rerunning anything

```bash
python notebooks/build_notebook.py
```

This rebuilds `results/summary/`, `results/tables/` and `results/figures/` from the per-issue rows in `results/runs/` and writes `notebooks/analysis.ipynb`. It takes about a minute and needs no GPU and no data download.

## 7. Rerun the experiments

Each step is safe to repeat. Finished evaluation runs are skipped, training resumes.

```bash
# data for evaluation: clones 18 repositories, about 5 minutes
python experiments/build_data.py --split dev
python experiments/build_data.py --split lite
python experiments/build_data.py --split verified

# data for training: clones 44 more repositories, about 45 minutes
python experiments/build_data.py --split gym
python experiments/build_data.py --split train
python experiments/build_train_examples.py      # also writes the leakage check

# train (80 minutes on an RTX 4060) and select the checkpoint
scripts/train_main.sh
python experiments/train_ranker.py --run-name main --promote

# evaluate on the validation split, then on the test sets
python experiments/evaluate.py --split dev --methods line_window chunk_bm25 skeleton ast_random ast_bm25 --workers 6 --device cpu
python experiments/evaluate.py --split dev --methods ast_trained ast_hybrid ast_unixcoder chunk_unixcoder
python experiments/evaluate.py --split lite --methods untrained trained --workers 2
python experiments/evaluate.py --split verified --methods untrained trained --workers 2

# tables and figures
for s in dev lite verified; do
  python experiments/summarize.py --split $s --quiet
  python experiments/make_tables.py --split $s
done
python experiments/make_figures.py --split lite
python scripts/fill_docs.py        # copies the tables into README.md and docs/

# the rest: scaling, error analysis, perplexity baseline, downstream check, extra seeds
scripts/run_queue.sh
```

Method names you can pass to `--methods` are listed in `experiments/config.yaml` and built in `src/astprune/registry.py`. A selector variant is a suffix, for example `ast_trained_nosig` or `ast_bm25_perfile`. Another training run is `ast_trained@<run name>`.

Tips for an 8 GB GPU:

- Run encoder methods with at most `--workers 2`. Each worker holds its own model.
- Run CPU-only methods with `--device cpu --workers 6` at the same time if you like, they do not touch the GPU.
- If the GPU is over-full on Windows or WSL it does not crash, it becomes about ten times slower. `nvidia-smi` showing memory at the limit is the sign.

## 8. Known limits

See the "Limits" section of the README. The short version: Python only, recall of the right code is measured and not test pass rate, and a function larger than the budget cannot be kept.

## 9. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| "Using BM25 ranking" in the output | no checkpoint at `checkpoints/ranker/best` | section 3, or set `ASTPRUNE_CHECKPOINT` |
| Demo: "Could not talk to the MCP server" | package not installed in this environment | `pip install -e .` |
| Demo: "The server refused the request" | bad budget or no readable file | the message says which |
| `data/processed/<split>.jsonl not found` | data not built | `python experiments/build_data.py --split <split>` |
| "nothing to do, all requested runs exist" | results are already there | add `--force` to redo them |
| A clone hangs | very large repository, slow network | stop and rerun, finished clones are kept |
| Training stops (laptop slept or restarted) | | `scripts/train_main.sh` again |
| `torch.OutOfMemoryError` | another program holds GPU memory | close it, or lower `--workers` |
| Leftover `*.tmp` files under `results/runs/` | an evaluation was interrupted | rerun the same command, they are overwritten |
