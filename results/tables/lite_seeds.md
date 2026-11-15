Training variance. Gold-function recall (%) of the trained ranker, gold files + direct imports, on dev (where the checkpoint is chosen) and on lite. Same data and settings in every run except the seed, and in the last row the filter that drops training patches touching more than 5 functions.

| Training run | Examples | Best step | Dev @10% | Dev @25% | Lite @10% | Lite @25% | Lite @50% |
|---|---|---|---|---|---|---|---|
| Seed 0 (the main run) | 14,534 | 2,700 | 60.1 | 79.1 | 81.8 | 94.3 | 98.0 |
| Seed 1 | 14,534 | 1,620 | 59.5 | 78.4 | 82.2 | 94.2 | 97.6 |
| Seed 2 | 14,534 | 2,700 | 60.6 | 79.1 | 81.9 | 94.5 | 98.5 |
| Mean ± sd of the 3 seeds |  |  | 60.1 ± 0.5 | 78.9 ± 0.4 | 82.0 ± 0.2 | 94.3 ± 0.2 | 98.1 ± 0.5 |
| Seed 0, patches of any size | 17,325 | 3,200 | 58.6 | 78.9 | 82.3 | 94.5 | 98.2 |
