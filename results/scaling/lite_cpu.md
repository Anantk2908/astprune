# Scaling with candidate files (lite, 25% budget, cpu)

Latency: mean over 10 issues, cold caches, includes parsing. Recall: all issues in the split.

| method | files | gold-function recall % | ms per issue | median | p90 | ms per file | score ms | select ms |
|---|---|---|---|---|---|---|---|---|
| ast_trained | 5 | 68.4 | 40769 | 37398 | 66548 | 8153.8 | 40596 | 76 |
| ast_trained | 10 | 76.1 | 75878 | 72625 | 93198 | 7587.8 | 75590 | 103 |
| ast_trained | 20 | 83.4 | 137709 | 131877 | 167792 | 6885.4 | 137173 | 169 |
| ast_bm25 | 5 | 50.8 | 185 | 155 | 264 | 36.9 | 61 | 31 |
| ast_bm25 | 10 | 60.3 | 344 | 325 | 462 | 34.4 | 127 | 36 |
| ast_bm25 | 20 | 67.1 | 684 | 629 | 869 | 34.2 | 263 | 61 |
| ast_unixcoder | 5 | 48.3 | 40829 | 37342 | 66250 | 8165.7 | 40705 | 30 |
| ast_unixcoder | 10 | 54.3 | 75819 | 72545 | 93264 | 7581.9 | 75580 | 61 |
| ast_unixcoder | 20 | 61.3 | 137622 | 131349 | 167772 | 6881.1 | 137143 | 96 |
| chunk_bm25 | 5 | 40.5 | 216 | 183 | 294 | 43.2 | 105 | 19 |
| chunk_bm25 | 10 | 48.8 | 434 | 387 | 553 | 43.4 | 216 | 35 |
| chunk_bm25 | 20 | 59.7 | 831 | 689 | 1186 | 41.6 | 416 | 60 |
