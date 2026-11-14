Paired bootstrap differences in gold-function recall, dev, gold files + direct imports.

| Question | A | B | Budget | A minus B (pp) | 95% interval | p | Issues |
|---|---|---|---|---|---|---|---|
| H1 | AST + BM25 | Line window | 10% | +36.9 | [+31.0, +43.0] | <0.001 | 209 |
| H1 | AST + BM25 | Line window | 25% | +41.8 | [+34.1, +49.3] | <0.001 | 209 |
| H1 | AST + BM25 | Line window | 50% | +39.2 | [+31.9, +46.5] | <0.001 | 209 |
| H1 | AST + BM25 | Line window | 1000 tok | +21.6 | [+16.6, +26.8] | <0.001 | 208 |
| H1 | AST + BM25 | Line window | 2000 tok | +29.6 | [+23.9, +35.3] | <0.001 | 205 |
| H1 | AST + BM25 | Line window | 4000 tok | +38.0 | [+31.4, +44.6] | <0.001 | 199 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 10% | +22.4 | [+17.0, +27.9] | <0.001 | 209 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 25% | +21.3 | [+15.3, +27.3] | <0.001 | 209 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 50% | +18.3 | [+13.1, +23.5] | <0.001 | 209 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 1000 tok | +16.8 | [+12.1, +21.6] | <0.001 | 208 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 2000 tok | +18.3 | [+13.4, +23.4] | <0.001 | 205 |
| H1 | AST + BM25 | Chunk RAG (BM25) | 4000 tok | +25.7 | [+20.0, +31.5] | <0.001 | 199 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 10% | +16.1 | [+11.7, +20.7] | <0.001 | 209 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 25% | +27.4 | [+22.0, +32.8] | <0.001 | 209 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 50% | +26.7 | [+20.9, +32.5] | <0.001 | 209 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 1000 tok | +6.7 | [+3.3, +10.2] | <0.001 | 208 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 2000 tok | +15.3 | [+10.8, +19.9] | <0.001 | 205 |
| H1 | AST + UniXcoder, zero-shot | Chunk RAG (UniXcoder, zero-shot) | 4000 tok | +21.5 | [+16.5, +26.6] | <0.001 | 199 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 10% | +33.9 | [+28.1, +39.7] | <0.001 | 209 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 25% | +30.8 | [+25.4, +36.2] | <0.001 | 209 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 50% | +18.2 | [+14.0, +22.5] | <0.001 | 209 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 1000 tok | +25.2 | [+19.9, +30.5] | <0.001 | 208 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 2000 tok | +30.1 | [+24.8, +35.7] | <0.001 | 205 |
| H2 | AST + trained ranker (ours) | AST + UniXcoder, zero-shot | 4000 tok | +31.8 | [+25.8, +37.9] | <0.001 | 199 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 10% | +18.8 | [+13.5, +24.0] | <0.001 | 209 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 25% | +18.4 | [+13.8, +23.0] | <0.001 | 209 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 50% | +13.2 | [+9.2, +17.3] | <0.001 | 209 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 1000 tok | +12.0 | [+6.3, +17.7] | <0.001 | 208 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 2000 tok | +21.0 | [+15.4, +26.8] | <0.001 | 205 |
| H2 | AST + trained ranker (ours) | AST + BM25 | 4000 tok | +20.4 | [+14.8, +26.2] | <0.001 | 199 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 10% | +15.6 | [+11.4, +20.0] | <0.001 | 209 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 25% | +13.5 | [+9.7, +17.5] | <0.001 | 209 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 50% | +9.6 | [+6.4, +12.9] | <0.001 | 209 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 1000 tok | +8.8 | [+4.7, +13.0] | <0.001 | 208 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 2000 tok | +16.5 | [+11.6, +21.4] | <0.001 | 205 |
| H2 | AST + trained ranker + BM25 (ours) | AST + BM25 | 4000 tok | +17.9 | [+13.5, +22.6] | <0.001 | 199 |
