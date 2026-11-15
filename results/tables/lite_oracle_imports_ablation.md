Selector ablations (%), lite, gold files + direct imports.

| Selector | Recall @10% | Recall @25% | Recall @50% | Sig. recall @10% | Sig. recall @25% | Line recall @25% |
|---|---|---|---|---|---|---|
| AST + BM25 | 60.9 | 77.1 | 90.7 | 67.2 | 81.5 | 76.2 |
| AST + BM25, no signature tier | 60.9 | 77.9 | 91.2 | 60.9 | 77.9 | 76.9 |
| AST + trained ranker (ours) | 81.8 | 94.3 | 98.0 | 87.6 | 95.4 | 87.8 |
| AST + trained ranker (ours), rank by probability per token | 79.0 | 90.9 | 98.0 | 86.9 | 94.0 | 84.7 |
| AST + trained ranker (ours), no signature tier | 81.8 | 94.7 | 98.0 | 81.8 | 94.7 | 88.6 |
| AST + trained ranker (ours), budget split per file | 49.5 | 74.9 | 88.0 | 68.5 | 88.6 | 69.3 |
| AST + trained ranker (ours), 15% of budget reserved for signatures | 80.1 | 92.2 | 97.1 | 94.3 | 98.4 | 85.4 |
| AST + trained ranker (ours), 30% of budget reserved for signatures | 78.1 | 92.2 | 97.1 | 96.5 | 99.6 | 85.5 |
| AST + trained ranker + BM25 (ours) | 79.6 | 92.9 | 97.6 | 84.1 | 93.6 | 87.0 |
