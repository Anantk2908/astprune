# Downstream check (Qwen/Qwen2.5-Coder-7B-Instruct, lite, oracle)

| context | issues | prompt tokens | edit found in file % | edits a gold function % | calls an undefined symbol % |
|---|---|---|---|---|---|
| full | 64 | 4676 | 62.5 [50.0, 73.4] | 45.3 [32.8, 57.8] | 1.6 [0.0, 4.7] |
| ast_trained | 100 | 2587 | 50.0 [40.0, 60.0] | 35.0 [26.0, 44.0] | 1.0 [0.0, 3.0] |
| ast_bm25 | 100 | 2586 | 48.0 [38.0, 58.0] | 29.0 [20.0, 38.0] | 0.0 [0.0, 0.0] |
| chunk_bm25 | 100 | 2582 | 55.5 [45.5, 65.0] | 31.0 [22.0, 40.0] | 0.0 [0.0, 0.0] |
| line_window | 100 | 2580 | 41.0 [32.0, 51.0] | 23.0 [15.0, 31.0] | 3.0 [0.0, 7.0] |
