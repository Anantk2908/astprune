Paired bootstrap differences in gold-function recall, lite, gold files + direct imports.

| Question | A | B | Budget | A minus B (pp) | 95% interval | p | Issues |
|---|---|---|---|---|---|---|---|
| H1 | AST + BM25 | Line window | 10% | +56.1 | [+49.6, +62.7] | <0.001 | 274 |
| H1 | AST + BM25 | Line window | 25% | +60.0 | [+53.3, +66.4] | <0.001 | 274 |
| H1 | AST + BM25 | Line window | 50% | +47.6 | [+40.8, +54.3] | <0.001 | 274 |
| H1 | AST + BM25 | Line window | 1000 tok | +40.3 | [+34.5, +46.2] | <0.001 | 271 |
| H1 | AST + BM25 | Line window | 2000 tok | +49.6 | [+43.5, +55.6] | <0.001 | 268 |
| H1 | AST + BM25 | Line window | 4000 tok | +54.1 | [+47.6, +60.5] | <0.001 | 262 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 10% | +24.5 | [+18.5, +30.6] | <0.001 | 274 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 25% | +21.6 | [+15.9, +27.5] | <0.001 | 274 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 50% | +10.4 | [+5.7, +15.3] | <0.001 | 274 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 1000 tok | +21.6 | [+16.0, +27.4] | <0.001 | 271 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 2000 tok | +27.0 | [+21.1, +33.1] | <0.001 | 268 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 4000 tok | +21.4 | [+15.8, +27.0] | <0.001 | 262 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 10% | +24.2 | [+18.2, +30.1] | <0.001 | 274 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 25% | +17.2 | [+11.7, +23.0] | <0.001 | 274 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 50% | +13.7 | [+8.2, +19.1] | <0.001 | 274 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 1000 tok | +14.9 | [+10.1, +19.6] | <0.001 | 271 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 2000 tok | +17.0 | [+11.7, +22.4] | <0.001 | 268 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 4000 tok | +18.7 | [+13.2, +24.3] | <0.001 | 262 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 10% | +35.1 | [+29.3, +41.0] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 25% | +30.3 | [+24.8, +36.0] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 50% | +14.5 | [+10.5, +18.8] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 1000 tok | +32.1 | [+26.4, +38.0] | <0.001 | 271 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 2000 tok | +36.4 | [+30.3, +42.5] | <0.001 | 268 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 4000 tok | +36.8 | [+30.9, +42.7] | <0.001 | 262 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 10% | +20.8 | [+14.8, +26.9] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 25% | +17.2 | [+12.4, +22.1] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 50% | +7.3 | [+4.2, +10.6] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 1000 tok | +18.1 | [+12.0, +24.1] | <0.001 | 271 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 2000 tok | +19.5 | [+13.2, +25.8] | <0.001 | 268 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 4000 tok | +23.3 | [+17.3, +29.3] | <0.001 | 262 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 10% | +18.6 | [+13.7, +23.5] | <0.001 | 274 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 25% | +15.8 | [+11.7, +20.1] | <0.001 | 274 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 50% | +6.9 | [+4.2, +9.9] | <0.001 | 274 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 1000 tok | +17.2 | [+12.0, +22.4] | <0.001 | 271 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 2000 tok | +16.7 | [+11.6, +22.1] | <0.001 | 268 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 4000 tok | +18.2 | [+13.1, +23.3] | <0.001 | 262 |
| H2 | AST + trained ranker (ours) | AST + LLM perplexity | 10% | +11.4 | [+6.2, +16.5] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + LLM perplexity | 25% | +12.2 | [+7.7, +16.8] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + LLM perplexity | 50% | +8.0 | [+4.9, +11.3] | <0.001 | 274 |
| H2 | AST + trained ranker (ours) | AST + LLM perplexity | 1000 tok | +6.6 | [+0.9, +12.1] | 0.021 | 271 |
| H2 | AST + trained ranker (ours) | AST + LLM perplexity | 2000 tok | +8.6 | [+3.4, +13.9] | 0.001 | 268 |
| H2 | AST + trained ranker (ours) | AST + LLM perplexity | 4000 tok | +12.6 | [+7.7, +17.6] | <0.001 | 262 |
| external | AST + trained ranker (ours) | SWE-Pruner (released checkpoint) | 10% | +49.7 | [+43.4, +55.8] | <0.001 | 274 |
| external | AST + trained ranker (ours) | SWE-Pruner (released checkpoint) | 25% | +39.8 | [+33.9, +45.7] | <0.001 | 274 |
| external | AST + trained ranker (ours) | SWE-Pruner (released checkpoint) | 50% | +17.0 | [+12.6, +21.5] | <0.001 | 274 |
| external | AST + trained ranker (ours) | SWE-Pruner (released checkpoint) | 1000 tok | +39.3 | [+33.1, +45.4] | <0.001 | 271 |
| external | AST + trained ranker (ours) | SWE-Pruner (released checkpoint) | 2000 tok | +45.7 | [+39.4, +52.0] | <0.001 | 268 |
| external | AST + trained ranker (ours) | SWE-Pruner (released checkpoint) | 4000 tok | +47.1 | [+41.0, +53.2] | <0.001 | 262 |
