Selector ablations (%), dev, gold files + direct imports.

| Selector | Recall @10% | Recall @25% | Recall @50% | Sig. recall @10% | Sig. recall @25% | Line recall @25% |
|---|---|---|---|---|---|---|
| AST + BM25 | 41.3 | 60.8 | 78.0 | 48.6 | 66.5 | 61.8 |
| AST + BM25, no signature tier | 41.4 | 61.3 | 78.2 | 41.4 | 61.3 | 62.3 |
| AST + BM25, budget split per file | 23.8 | 40.0 | 66.2 | 40.1 | 60.3 | 42.0 |
| AST + BM25, 15% of budget reserved for signatures | 37.4 | 60.9 | 76.2 | 69.5 | 87.6 | 62.8 |
| AST + trained ranker (ours) | 60.1 | 79.1 | 91.2 | 66.9 | 83.0 | 70.5 |
| AST + trained ranker (ours), rank by probability per token | 59.4 | 78.5 | 91.7 | 69.1 | 85.2 | 69.4 |
| AST + trained ranker (ours), no signature tier | 60.2 | 79.3 | 91.3 | 60.2 | 79.3 | 70.5 |
| AST + trained ranker (ours), budget split per file | 34.8 | 53.4 | 78.5 | 53.7 | 72.8 | 48.3 |
| AST + trained ranker (ours), 15% of budget reserved for signatures | 56.1 | 76.5 | 90.2 | 81.7 | 94.3 | 70.1 |
| AST + trained ranker (ours), 30% of budget reserved for signatures | 53.0 | 73.1 | 89.1 | 88.0 | 98.5 | 67.7 |
| AST + trained ranker + BM25 (ours) | 57.0 | 74.3 | 87.6 | 62.6 | 79.8 | 68.2 |
| AST + trained ranker + BM25 (ours), no signature tier | 57.2 | 74.8 | 87.8 | 57.2 | 74.8 | 68.5 |
