# Demo script (about 4 minutes)

Rehearse this once on the machine you will present from. Everything runs locally, so no network is needed during the demo.

## Before the slot

```bash
source .venv/bin/activate
python demo/app.py --list          # samples are there
python demo/app.py --sample 1      # warm run, also checks the GPU and checkpoint
nvidia-smi                         # nothing else is using the GPU
```

Have two terminals open: one for `demo/app.py`, one with Claude Code started in this folder (it picks up `.mcp.json` and shows the `astprune` server as connected).

## Sequence

1. **Say who it is for (20 s).** "A coding agent opens a 2,000-line file to fix one function. It pays for all 2,000 lines. Our tool gives it the functions that matter for this issue, within a token budget."
2. **Scripted run (60 s).** `python demo/app.py --sample 1`. Point at: tokens before and after, the "kept in full" column, the line that says the function the real fix changed was kept.
3. **Tighter budget (40 s).** `python demo/app.py --sample 1 --budget 0.1`. Same issue, a tenth of the tokens, and the function the fix changed is still there in full.
4. **A case where it falls short (40 s).** `python demo/app.py --sample 2 --budget 0.1`. The function that needs the fix is about 700 tokens and the whole budget is about 450, so it cannot fit and comes back as a signature only. Say that this is the most common kind of miss in our error analysis. Then run `python demo/app.py --sample 2 --budget 0.25`: the function is kept in full. This is the recovery path an agent has, ask again with a larger budget.
5. **Live agent (60 s).** In Claude Code, paste the prompt from `demo/agent_prompt.md`. The agent calls `get_budgeted_context` instead of reading the files. Point at the tool call and the token counts in its result.
6. **Bad input (20 s).** `python demo/app.py --root demo/sample_repo --files does_not_exist.py --issue "x"`. The server answers with a clear message and stays up.

## If something breaks

- No GPU or no checkpoint: add `--scorer bm25`. The server also falls back to BM25 by itself and says so in a note.
- The server does not start: `pip install -e .` in the active environment, then retry.
- Claude Code does not show the server: restart it from the repository root so it reads `.mcp.json`.
- Anything else: play the backup recording and say that it is a recording.

## Backup recording

Record steps 2 to 6 once with a screen recorder after the final rehearsal and keep the file next to the slides. A live demo is graded higher than a recording, so only use it if the live run fails.

## What is live and what is not

- Live: parsing, ranking with the trained model, selection, the MCP round trip, the agent's tool call.
- Not shown live: training (78 minutes on the RTX 4060) and the benchmark runs. Their outputs are in `results/`.
- Reliability numbers quoted in the talk come from `results/demo_reliability.json`.
