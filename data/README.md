# Data

Nothing large is committed. Everything under `data/` except this file is rebuilt by scripts and ignored by git.

## Sources

| Split key | Hugging Face dataset | Split | Used for |
|---|---|---|---|
| `lite` | `princeton-nlp/SWE-bench_Lite` | test | main test set (300 issues, 12 repos) |
| `verified` | `princeton-nlp/SWE-bench_Verified` | test | second test set (500 issues, same 12 repos) |
| `dev` | `princeton-nlp/SWE-bench` | dev | validation, all tuning (225 issues, 6 repos) |
| `train` | `princeton-nlp/SWE-bench` | train | training (19,008 issues, 35 repos) |
| `gym` | `SWE-Gym/SWE-Gym` | train | training (2,438 issues, 11 repos) |

All of these are public. The repositories themselves are cloned from GitHub.

## Rebuild

```bash
python experiments/build_data.py --split dev        # about 1 minute once the repos are cloned
python experiments/build_data.py --split lite
python experiments/build_data.py --split verified
python experiments/build_data.py --split gym        # only needed for training
python experiments/build_data.py --split train      # only needed for training, clones 35 repos
python experiments/build_train_examples.py          # only needed for training
```

What you get:

- `data/repos/` bare git clones, about 12 GB for all splits (about 2 GB for the three evaluation splits). Blobs over 2 MB are left on the server.
- `data/cache/hf/` the Hugging Face cache.
- `data/processed/<split>.jsonl` one record per issue: issue text, gold functions and gold lines, candidate files.
- `data/processed/train_examples/` tokenized training examples.
- `results/data/<split>_stats.json` counts for each split (committed).
- `results/data/leakage_check.json` the train / dev / test repository lists and their overlap (committed).

## How labels are made

For each issue we read the files the gold patch touches as they were at `base_commit` and split them into nodes (functions, methods, class headers, runs of other statements). A node is gold if the patch removes or rewrites a line in it, inserts lines into its body, or adds a decorator to it. New functions the patch adds are not labels, since they do not exist before the fix. Code is in `src/astprune/data/labels.py`, tests in `tests/test_labels.py`.

## Things to know

- `hints_text` is never read. It contains comments written after the issue was filed.
- A few patches only apply to `base_commit` with a line offset. Hunks are moved to where their context matches, the same way `patch` does it. Issues where a hunk matches nowhere are skipped and counted.
- Issues whose patch touches no Python file are skipped (2,117 in SWE-bench train, mostly docs and Cython).
- One training repository (JohnSnowLabs/spark-nlp, 3 issues) did not finish cloning in our run and was skipped.
- Files over 400,000 characters are never offered as retrieved or imported candidates. These are data tables shipped as `.py` files, for example pydicom's DICOM dictionaries. No gold file in dev, Lite or Verified is that large.
- Test files are excluded from BM25 file retrieval. Gold patches never touch them, because SWE-bench keeps test changes in a separate field.
