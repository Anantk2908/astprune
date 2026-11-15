Downstream check on lite: Qwen2.5-Coder-7B-Instruct in 4-bit reads the issue and one view of the gold files and writes SEARCH/REPLACE edits. Budget 25% of the file tokens, at most 4,000 tokens. Greedy decoding, no tests run. In percent of issues, except "edits that apply", the share of edits whose SEARCH text is found in the file. Intervals are 95% bootstrap intervals, and the difference is paired over the same issues. Whole files were only given when they were short enough, on 64 issues; there the trained ranker's prompts average 1,561 tokens.

| Context | Issues | Prompt tokens | Edits that apply | Edits a gold function | Ours minus this (pp) | Calls an undefined name |
|---|---|---|---|---|---|---|
| Whole gold files, no pruning | 64 | 4,676 | 62.5 | 45.3 [32.8, 57.8] | -9.4 [-21.9, +3.1] | 1.6 |
| AST + trained ranker (ours) | 100 | 2,587 | 50.0 | 35.0 [26.0, 44.0] |  | 1.0 |
| AST + BM25 | 100 | 2,586 | 48.0 | 29.0 [20.0, 38.0] | +6.0 [-3.0, +15.0] | 0.0 |
| Chunk RAG (BM25) | 100 | 2,582 | 55.5 | 31.0 [22.0, 40.0] | +4.0 [-6.0, +14.0] | 0.0 |
| Line window | 100 | 2,580 | 41.0 | 23.0 [15.0, 31.0] | +12.0 [+1.0, +23.0] | 3.0 |
