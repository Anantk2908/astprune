# Baselines: what we ran and how

Every method gets the same candidate files, the same budget and the same token counter (tiktoken `cl100k_base`). A gold function counts as kept only if all of its lines survive. That rule is strict for methods that prune lines, so every table also reports gold-line recall, which does not care about function boundaries.

## Our own baselines

These are implemented in this repository and run on every issue of every split.

| Name in the tables | What it does | Code |
|---|---|---|
| Line window | Keeps the top of each file until the budget is full. What an agent gets when it reads files with a size cap | `src/astprune/baselines/line_window.py` |
| Chunk RAG (BM25) | Cuts files into 256-token chunks and keeps the chunks BM25 ranks highest. The RepoCoder-style setup | `src/astprune/baselines/chunk_rag.py` |
| Chunk RAG (UniXcoder, zero-shot) | Same chunks, ranked by the untrained encoder | same file |
| Skeleton (Agentless-style) | Signatures of every function first, then bodies in file order. No ranking | `src/astprune/pruners.py` |
| AST + random order | Our parser and selector with a random ranking. A floor | `src/astprune/scorers/simple.py` |
| AST + BM25 | Our parser and selector, nodes ranked by BM25 against the issue | `src/astprune/scorers/bm25.py` |
| AST + UniXcoder, zero-shot | Our parser and selector, the same encoder as ours but without our training | `src/astprune/scorers/encoder.py` |
| AST + LLM perplexity | Our parser and selector, nodes ranked by LongCodeZip's criterion: how much a node lowers the perplexity of the issue under a code LLM (Qwen2.5-Coder-1.5B-Instruct). Our own implementation, so it runs on every issue | `src/astprune/scorers/perplexity.py` |
| AST + trained ranker + BM25 | Rank fusion of BM25 and our ranker. We expected it to help, and it did not | `src/astprune/scorers/simple.py` |

The AST baselines share everything with our method except the ranking, so the differences between them are differences in ranking alone.

## Third-party systems

Both are run through the authors' released code. Each needed small changes to run on our machine. The changes are listed below, and none of them touches how the systems rank code.

### SWE-Pruner

- **Package and weights:** `swe-pruner` 0.1.1, checkpoint `ayanami-kitasan/code-pruner` (0.6B, Qwen3-Reranker backbone with a CRF head).
- **Budget:** the released system keeps lines whose score is above a threshold, which gives no control over output size. We take the line scores exactly as the authors compute them and keep the highest-scoring lines until the budget is full.
- **Changes needed to run it:**
  - The authors use flash-attention, which has no wheel for our torch build, so the backbone uses PyTorch's built-in attention instead. Same maths, different kernel.
  - The released model class skips a setup step that transformers 5 needs, so a small subclass calls it.
  - We checked that all 325 weight tensors load and match the checkpoint.
- **What ran:** SWE-bench Lite, all 300 issues, with gold files and with gold files plus imports. About 4.3 to 6.8 seconds per file on our GPU, 4.5 hours in total.
- **Adapter:** `src/astprune/baselines/swe_pruner.py`.

### LongCodeZip

- **Package:** `longcodezip` 0.1.0. The paper's default scorer is a 7B model. On an 8 GB GPU we use Qwen2.5-Coder-1.5B-Instruct. This is a real deviation, and a stronger scorer may do better than what we report.
- **Two variants:**
  - *Coarse stage* (`longcodezip_rank`): their function-level chunking and perplexity ranking (`rank_only=True` in their API), then chunks in rank order until our budget is full. Exact budget control.
  - *Full* (`longcodezip`): their two-stage `compress_code_file`. Its budget is a soft target in the scorer's tokens, so we ask for a target, measure the result with our tokenizer and lower the target until it fits, up to four tries, then trim. It lands well under the budget on average (6.4%, 16.2% and 37.0% of the tokens at the 10%, 25% and 50% budgets), which costs it some recall.
- **What ran:** a fixed random sample of 100 Lite issues (`--sample 100`, seed 0), gold files only.
  - The coarse stage finished all 100. 95 of them have a gold function.
  - The full variant takes 30 to 60 seconds per file and was stopped by a 3-hour cap after 42 issues, 40 of them with a gold function. The sample order is random, so these 42 are still a random subset of Lite. Only issues with every budget finished are kept, see `results/runs/lite/manifests/longcodezip_full_partial.json`.
  - With gold files plus imports there are about ten files per issue, which would have taken days, so neither variant ran in that setting.
- **Adapter:** `src/astprune/baselines/longcodezip.py`.

AST + LLM perplexity, in the table above, covers the gap: LongCodeZip's ranking criterion with the same 1.5B model, on all issues and in every setting we ran it in.

## Results on the same issues

Because the third-party runs cover different sets of issues, this table puts each one next to our ranker on exactly the issues it ran on. Recall cells are theirs / ours.

<!-- table:lite_external -->
Third-party baselines on lite, and LongCodeZip's ranking criterion on our nodes (AST + LLM perplexity), each next to our trained ranker on exactly the issues it ran on. Recall cells are theirs / ours, in percent. The difference is paired over the same issues, with a 95% bootstrap interval. Seconds per file include model scoring, on our laptop GPU.

| Method | Files | Issues | Recall @10% | Recall @25% | Recall @50% | Ours minus them @25% (pp) | Line recall @25% | Parses @25% | Seconds per file |
|---|---|---|---|---|---|---|---|---|---|
| SWE-Pruner (released checkpoint) | Gold files only | 274 | 15.1 / 49.5 | 26.3 / 74.9 | 43.6 / 88.0 | +48.6 [+42.7, +54.6] | 73.0 / 69.3 | 0.0 | 6.81 |
| LongCodeZip (coarse stage) | Gold files only | 95 | 32.2 / 50.7 | 52.5 / 71.1 | 69.1 / 86.8 | +18.6 [+7.0, +30.0] | 65.8 / 67.8 | 17.0 | 7.69 |
| LongCodeZip (full) | Gold files only | 40 | 28.1 / 57.5 | 41.2 / 78.8 | 56.7 / 91.2 | +37.5 [+21.2, +53.8] | 58.1 / 74.7 | 4.8 | 40.79 |
| AST + LLM perplexity | Gold files only | 274 | 43.6 / 49.5 | 63.0 / 74.9 | 81.0 / 88.0 | +11.9 [+6.3, +17.5] | 59.6 / 69.3 | 100.0 | 4.27 |
| AST + trained ranker (ours) | Gold files only | 274 |  |  |  |  |  | 100.0 | 0.16 |
| SWE-Pruner (released checkpoint) | Gold files + direct imports | 274 | 32.0 / 81.8 | 54.6 / 94.3 | 81.0 / 98.0 | +39.8 [+33.9, +45.7] | 87.6 / 87.8 | 14.7 | 4.29 |
| AST + LLM perplexity | Gold files + direct imports | 274 | 70.3 / 81.8 | 82.1 / 94.3 | 90.0 / 98.0 | +12.2 [+7.7, +16.8] | 78.9 / 87.8 | 100.0 | 1.76 |
| AST + trained ranker (ours) | Gold files + direct imports | 274 |  |  |  |  |  | 100.0 | 0.03 |
<!-- /table -->

How to read it:

- On gold-function recall our ranker is ahead of every third-party system at every budget, by wide margins.
- On gold-line recall, SWE-Pruner is level with our ranker when imports are included. With gold files only it is ahead at the 10% budget (58.9% against 45.9%, from `results/summary/lite_oracle_paired.csv`). So it is good at finding the edited lines. It does not keep the functions around them whole, and its output almost never parses as Python.
- LongCodeZip's coarse stage is the closest in spirit to our method, since it also ranks whole functions. On the same 95 issues it is about level with AST + BM25 at the 10% and 25% budgets (32.2% and 52.5%, against 34.4% and 51.6% for BM25) and below it at 50%. The full variant scores lower than its coarse stage, partly because it leaves part of the budget unused. Keep in mind that both ran with a 1.5B scorer instead of the 7B one from the paper.
- AST + LLM perplexity is LongCodeZip's ranking criterion on our nodes, with the same 1.5B model, on every issue. It is the best training-free ranker we ran at the 10% and 25% budgets, and our ranker is still about 12 points ahead at 25% with either set of files. It is also roughly 25 to 60 times slower per file (the last column).

## Installing them

Both are kept out of `requirements.txt`, since they are only needed for these runs and LongCodeZip pins an older tree-sitter:

```bash
pip install --no-deps swe-pruner
huggingface-cli download ayanami-kitasan/code-pruner --local-dir checkpoints/swe_pruner
pip install --no-deps git+https://github.com/YerbaPage/LongCodeZip.git loguru
python scripts/smoke_external.py swe_pruner          # four issues, prints recall and time
python scripts/smoke_external.py longcodezip_rank    # run the two LongCodeZip modes separately,
python scripts/smoke_external.py longcodezip         # together they do not fit in 8 GB
```
