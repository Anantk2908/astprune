Gold-function recall (%) at a 25% budget under each candidate-file setting, lite.

| Method | Gold files only | Gold files + direct imports | BM25 top-5 files | BM25 top-10 files | BM25 top-20 files |
|---|---|---|---|---|---|
| Line window | 17.2 | 17.2 | 12.2 | 14.4 | 15.9 |
| Chunk RAG (BM25) | 29.7 | 55.5 | 40.5 | 48.8 | 59.7 |
| Chunk RAG (UniXcoder, zero-shot) | 19.3 | 46.8 | 32.2 | n/a | n/a |
| Skeleton (Agentless-style) | 12.5 | 6.8 | 33.2 | 43.4 | 54.2 |
| SWE-Pruner (released checkpoint) | 26.3 | 54.6 | n/a | n/a | n/a |
| AST + random order | 19.8 | 20.7 | 14.3 | 17.8 | 18.9 |
| AST + BM25 | 58.0 | 77.1 | 50.8 | 60.3 | 67.1 |
| AST + UniXcoder, zero-shot | 47.7 | 64.0 | 48.3 | 54.3 | 61.3 |
| AST + LLM perplexity | 63.0 | 82.1 | n/a | n/a | n/a |
| AST + trained ranker (ours) | 74.9 | 94.3 | 68.4 | 76.1 | 83.4 |
| AST + trained ranker + BM25 (ours) | 73.5 | 92.9 | 66.7 | 75.4 | 82.5 |
