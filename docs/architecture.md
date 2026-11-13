# Architecture

How the pieces fit, and why they are shaped the way they are.

## The path of one request

```
issue text + file paths + budget
        |
        v
   parse.py          tree-sitter splits each file into nodes
        |            (function, method, class header, block of other statements)
        v
   scorers/          one score per node against the issue
        |            bm25.py, encoder.py (zero-shot or trained), perplexity.py, simple.py (random, rank fusion)
        v
   select.py         fill the token budget in score order
        |            each node ends up FULL, SIG (signature only) or DROP
        v
   render            original lines for kept nodes, "..." for signature bodies,
                     one "# ... N lines omitted" comment per dropped run
```

`service.py` wraps this path for one caller. `server.py` exposes it as the MCP tool `get_budgeted_context(issue, files, budget)`. `demo/app.py` is an MCP client that calls the server. The experiments call the same `parse`, `scorers` and `select` modules through `pruners.py`, so the demo and the numbers come from one implementation.

## Modules

| Module | Job |
|---|---|
| `tokens.py` | The one token counter (tiktoken `cl100k_base`). Budgets and compression only compare across methods if everyone is measured the same way. |
| `parse.py` | File to nodes. Every line belongs to exactly one node, so a file can be rebuilt from its nodes. |
| `select.py` | Budget fill and rendering. |
| `pruners.py` | `ASTPruner` (scorer + selector) and the Agentless-style `SkeletonPruner`. |
| `scorers/` | Things that rank nodes. All have `score(issue, files) -> {path: array}`. |
| `baselines/` | Methods that do not use our nodes: line window, chunk RAG, and adapters for LongCodeZip and SWE-Pruner. |
| `data/` | Bare git clones, gold labels from patches, candidate files, training examples. |
| `metrics.py`, `stats.py` | Per-issue metrics and bootstrap intervals. |
| `evaluation.py`, `registry.py` | The loop that runs methods over issues, and the name to method table. |
| `train.py` | InfoNCE fine-tuning of the ranker. |
| `service.py`, `server.py` | The pruner as a callable, and as an MCP server. |

## Design choices

**Lines are partitioned, not just indexed.** Each line has one owner node. That gives three things for free: a 100% budget returns the file byte for byte, omitted regions can be reported exactly, and a patch line maps to one node when labels are built.

**What is a node.** Top-level functions and methods are nodes. A class is split into its header (the `class` line, docstring and leading attributes) and its methods, because a patch usually changes one method and keeping a whole class to show one method wastes the budget. Statements that sit between definitions become block nodes, cut at statement boundaries when a run is longer than 40 lines. Nested functions stay inside their parent, since half a closure is not useful.

**Three states.** FULL and DROP are obvious. SIG keeps the `def` line and replaces the body with `...`. The idea is that an agent that sees a signature knows the function exists and can ask for it, which should lower the chance it invents one. Signatures cost budget that could have gone to full bodies, so this is a trade-off and it is measured as an ablation, not assumed.

**The rendered view is valid Python.** A method is never shown without its class line, a class shown without any member gets a `...` body, and omission markers are comments, which Python ignores for indentation. The evaluation checks this with `ast.parse` and reports the rate.

**Budget is enforced on the real output.** The selector first uses a cheap upper bound (node token counts plus an allowance per omission marker). Then it renders, counts the real tokens, drops the lowest-ranked nodes if it is over, and uses up leftover room with the best remaining nodes. Our view pays for its own markers. The line-based baselines are rendered with no markers at all, so if anything the accounting favours them.

**All methods return the same thing.** A method returns, per file, the text it would show and the set of original line numbers that survived. Metrics are computed from that alone. A gold function counts as kept only if every non-blank line of it survived, for every method.

**Two-step methods.** `prepare(issue, files)` does the expensive part once (scoring), `prune(prepared, files, budget)` is cheap and runs once per budget. This is why adding a budget costs almost nothing.

**Bi-encoder, not cross-encoder.** The issue and each node are embedded separately and compared by cosine. A cross-encoder would likely rank better, but it needs one forward pass per (issue, node) pair. With a bi-encoder, node embeddings can be cached across budgets and across issues that share files.

**The trained and zero-shot rankers are the same class.** `Encoder` is loaded either from `microsoft/unixcoder-base` or from our checkpoint. Nothing else differs, so the gap between them is the effect of training.

**The MCP server never crashes on a bad request.** Missing files, files outside the workspace, non-Python files, syntax errors and an empty issue all produce a result with a warning. Bad budgets and an empty file list produce an MCP tool error with a message the calling model can act on. If no checkpoint is present the server starts with BM25 ranking and says so.

## Where time goes

For one issue with about ten candidate files, parsing and token counting take tens of milliseconds, the selector a few tens of milliseconds, and scoring depends on the scorer: BM25 is milliseconds, the encoder is one batched forward pass over the nodes. Measured numbers are in `results/scaling/`.

## What would change at larger scale

- More files per call: scoring is linear in the number of nodes. Node embeddings depend only on the file content, so a long-running server can cache them per file hash and pay only for the issue embedding and a matrix product.
- Larger files: nodes are truncated to 256 tokens for the encoder, so a very long function is judged by its head. BM25 sees the whole node.
- More users: the server holds one model and handles one call at a time. A shared deployment would batch requests on the GPU or run on CPU, where BM25 alone is a reasonable fallback.
- Other languages: only `parse.py` is Python-specific. tree-sitter has grammars for most languages, the rest of the pipeline works on nodes.
