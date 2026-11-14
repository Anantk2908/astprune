Paired bootstrap differences in gold-function recall, verified, gold files + direct imports.

| Question | A | B | Budget | A minus B (pp) | 95% interval | p | Issues |
|---|---|---|---|---|---|---|---|
| H1 | AST + BM25 | Line window | 10% | +51.6 | [+46.7, +56.4] | <0.001 | 458 |
| H1 | AST + BM25 | Line window | 25% | +56.7 | [+51.7, +61.5] | <0.001 | 458 |
| H1 | AST + BM25 | Line window | 50% | +40.9 | [+35.8, +46.0] | <0.001 | 458 |
| H1 | AST + BM25 | Line window | 1000 tok | +36.6 | [+32.4, +41.0] | <0.001 | 455 |
| H1 | AST + BM25 | Line window | 2000 tok | +45.4 | [+40.9, +50.0] | <0.001 | 450 |
| H1 | AST + BM25 | Line window | 4000 tok | +51.5 | [+46.8, +56.1] | <0.001 | 440 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 10% | +24.2 | [+19.7, +28.6] | <0.001 | 458 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 25% | +19.5 | [+14.8, +24.0] | <0.001 | 458 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 50% | +12.1 | [+8.2, +15.9] | <0.001 | 458 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 1000 tok | +20.6 | [+16.7, +24.6] | <0.001 | 455 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 2000 tok | +24.6 | [+20.3, +29.0] | <0.001 | 450 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 4000 tok | +22.8 | [+18.4, +27.0] | <0.001 | 440 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 10% | +22.3 | [+18.0, +26.6] | <0.001 | 458 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 25% | +22.4 | [+18.0, +26.8] | <0.001 | 458 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 50% | +15.9 | [+11.8, +20.0] | <0.001 | 458 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 1000 tok | +14.6 | [+11.0, +18.3] | <0.001 | 455 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 2000 tok | +19.4 | [+15.4, +23.4] | <0.001 | 450 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 4000 tok | +20.5 | [+16.3, +24.7] | <0.001 | 440 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 10% | +36.4 | [+32.0, +40.8] | <0.001 | 458 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 25% | +26.0 | [+22.2, +29.8] | <0.001 | 458 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 50% | +14.9 | [+11.9, +17.9] | <0.001 | 458 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 1000 tok | +28.7 | [+24.6, +32.8] | <0.001 | 455 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 2000 tok | +35.4 | [+31.0, +39.7] | <0.001 | 450 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 4000 tok | +35.5 | [+31.1, +39.9] | <0.001 | 440 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 10% | +26.7 | [+22.4, +31.0] | <0.001 | 458 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 25% | +19.1 | [+15.5, +22.9] | <0.001 | 458 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 50% | +11.3 | [+8.6, +14.1] | <0.001 | 458 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 1000 tok | +18.1 | [+13.7, +22.5] | <0.001 | 455 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 2000 tok | +24.9 | [+20.3, +29.5] | <0.001 | 450 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 4000 tok | +26.2 | [+21.9, +30.5] | <0.001 | 440 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 10% | +22.0 | [+18.5, +25.6] | <0.001 | 458 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 25% | +17.7 | [+14.5, +21.0] | <0.001 | 458 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 50% | +9.4 | [+7.1, +11.8] | <0.001 | 458 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 1000 tok | +16.8 | [+13.2, +20.6] | <0.001 | 455 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 2000 tok | +19.4 | [+15.8, +23.1] | <0.001 | 450 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 4000 tok | +21.0 | [+17.2, +24.9] | <0.001 | 440 |
