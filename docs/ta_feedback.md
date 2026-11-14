# TA feedback and what we did with it

Our TA discussion took place between the initial proposal (Oct 15) and the final proposal (Oct 30). The TA approved the topic and asked for two things. Both went into the final proposal and both shaped the finished project.

| What the TA asked for | Where it went in the final proposal | Where it is in the finished project |
|---|---|---|
| Add related literature. The first proposal compared only against Agentless, RepoCoder, RepoBench and LLMLingua | New "Related work" section: retrieval and localization, prompt compression (LongCodeZip), learned pruners for agents (SWE-Pruner, LaMR), and how we differ | Report, Related Work section. We also added SWEzze (arXiv 2603.28119), which appeared in March 2026 and is close to our setting. Two of the cited systems became baselines: a perplexity ranker that follows LongCodeZip's criterion (`src/astprune/scorers/perplexity.py`), and adapters for the official LongCodeZip and SWE-Pruner releases (`src/astprune/baselines/`) |
| Add a metrics plan. The first proposal had one metric, function-level recall at budget | New "Metrics and comparison" table with seven metrics, the rule that every method gets the same files and the same budget, bootstrap intervals, and the scaling experiment | All seven metrics are implemented in `src/astprune/metrics.py` and reported in `results/tables/`. Intervals and paired tests are in `src/astprune/stats.py`. The scaling experiment is `experiments/scaling.py` |

## Things we changed beyond what was asked

These came out of thinking about the metrics plan while building it. None of them changes the topic.

- **A validation split.** All tuning uses the SWE-bench dev split (225 issues from 6 repositories that appear in neither train nor test). The test sets were run once, after the choices were written down and committed (`docs/PROGRESS.md`, "Choices frozen on dev").
- **Gold-line recall.** The proposal counts a function as kept only if all of its lines survive. That is strict for line and chunk methods, so we also report the share of edited lines that survive, which treats every method the same way.
- **A parse-rate metric.** The proposal claims our output is never cut mid-function. We check it: every pruned file is passed through `ast.parse`.
- **SWE-bench Verified as a second test set.** Lite has 300 issues and mostly one gold function per issue. Verified has 500 and more multi-function patches, which tightens the intervals and makes the all-gold rate meaningful.
- **A negative result we report.** We expected that fusing BM25 with the trained ranker would help. On dev it scored below the trained ranker alone, so the shipped default is the trained ranker and the fusion is listed as an ablation.

## Feedback we could not fully act on

- The final proposal lists LongCodeZip and SWE-Pruner's released checkpoint as baselines. The adapters for both are written and their helper code is tested. Whether the official packages were run, and on which settings, is stated in the README and report next to the results.
