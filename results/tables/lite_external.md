Third-party baselines on lite, each next to our trained ranker on exactly the issues it ran on. Recall cells are theirs / ours, in percent. The difference is paired over the same issues, with a 95% bootstrap interval. Seconds per file include model scoring, on our laptop GPU.

| Method | Files | Issues | Recall @10% | Recall @25% | Recall @50% | Ours minus them @25% (pp) | Line recall @25% | Parses @25% | Seconds per file |
|---|---|---|---|---|---|---|---|---|---|
| SWE-Pruner (released checkpoint) | Gold files only | 274 | 15.1 / 49.5 | 26.3 / 74.9 | 43.6 / 88.0 | +48.6 [+42.7, +54.6] | 73.0 / 69.3 | 0.0 | 6.81 |
| LongCodeZip (coarse stage) | Gold files only | 95 | 32.2 / 50.7 | 52.5 / 71.1 | 69.1 / 86.8 | +18.6 [+7.0, +30.0] | 65.8 / 67.8 | 17.0 | 7.69 |
| LongCodeZip (full) | Gold files only | 40 | 28.1 / 57.5 | 41.2 / 78.8 | 56.7 / 91.2 | +37.5 [+21.2, +53.8] | 58.1 / 74.7 | 4.8 | 40.79 |
| AST + trained ranker (ours) | Gold files only | 274 |  |  |  |  |  | 100.0 | 0.16 |
| SWE-Pruner (released checkpoint) | Gold files + direct imports | 274 | 32.0 / 81.8 | 54.6 / 94.3 | 81.0 / 98.0 | +39.8 [+33.9, +45.7] | 87.6 / 87.8 | 14.7 | 4.29 |
| AST + trained ranker (ours) | Gold files + direct imports | 274 |  |  |  |  |  | 100.0 | 0.03 |
