# Scaling with candidate files (lite, 25% budget, cuda)

Latency: mean over 60 issues, cold caches, includes parsing. Recall: all issues in the split.

| method | files | gold-function recall % | ms per issue | median | p90 | ms per file | score ms | select ms |
|---|---|---|---|---|---|---|---|---|
| ast_trained | 5 | 68.4 | 827 | 778 | 1178 | 165.4 | 678 | 68 |
| ast_trained | 10 | 76.1 | 1428 | 1431 | 2069 | 142.8 | 1170 | 100 |
| ast_trained | 20 | 83.4 | 2591 | 2583 | 3331 | 129.5 | 2117 | 171 |
| ast_bm25 | 5 | 50.8 | 161 | 137 | 258 | 32.1 | 57 | 24 |
| ast_bm25 | 10 | 60.3 | 306 | 280 | 431 | 30.6 | 111 | 37 |
| ast_bm25 | 20 | 67.1 | 575 | 546 | 850 | 28.7 | 222 | 56 |
| ast_unixcoder | 5 | 48.3 | 771 | 748 | 1070 | 154.1 | 661 | 30 |
| ast_unixcoder | 10 | 54.3 | 1399 | 1394 | 1985 | 139.9 | 1179 | 62 |
| ast_unixcoder | 20 | 61.3 | 2539 | 2529 | 3296 | 126.9 | 2141 | 100 |
| chunk_bm25 | 5 | 40.5 | 198 | 181 | 313 | 39.5 | 97 | 16 |
| chunk_bm25 | 10 | 48.8 | 380 | 359 | 560 | 38.0 | 194 | 30 |
| chunk_bm25 | 20 | 59.7 | 720 | 654 | 1155 | 36.0 | 366 | 56 |
