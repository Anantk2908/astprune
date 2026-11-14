# Error analysis

This looks at every gold function that a method failed to keep in full. It covers our trained ranker and AST + BM25, the strongest training-free method on function recall. The setting is SWE-bench Lite with gold files plus their imports. The 25% budget is the main one, and the 10% budget adds more cases.

## How the cases were collected

`experiments/error_analysis.py` reruns each method on every issue where it lost a gold function. For each lost function it records:

- the function's size and the budget
- where the scorer ranked it among all candidate nodes
- whether it was dropped or shown as a signature
- whether the issue text names the function, its class or its module

Each case then gets one cause, by these rules in order:

1. **File not among candidates.** The file with the fix was not passed in. This cannot happen with gold files, only with retrieved ones.
2. **Function too large for the budget.** The function alone takes more than half the budget.
3. **Only its signature fit.** It was reached, but by then its body no longer fit, so it is shown as a signature.
4. **Outranked, issue does not name it.** Other code ranked higher and used up the budget, and the issue never mentions the function by name.
5. **Outranked, issue names it.** Same, although the issue does mention it.

The rules only sort cases into piles. We then read through the piles: all of our ranker's cases at 25% and a good part of the rest. Every case is listed in the tables at the end, and the raw rows are in `results/error_analysis/`.

## Counts

<!-- table:lite_r0.25_errors -->
Lost gold functions by cause, lite, gold files plus imports, 25% budget (317 gold functions in all).

| Cause | AST + trained ranker (ours) | AST + BM25 |
|---|---|---|
| file not among candidates | 0 | 0 |
| function too large for the budget | 2 | 2 |
| only its signature fit | 2 | 14 |
| outranked, issue does not name it | 12 | 53 |
| outranked, issue names it | 3 | 9 |
| total | 19 | 78 |
<!-- /table -->

<!-- table:lite_r0.10_errors -->
<!-- /table -->

What stands out:

- **Most misses, for both methods, are functions the issue never names.** Those issues describe a symptom (an error, a wrong result, a server that does not restart), and the fix sits in code with different names. This is also where training makes the biggest difference. At the 25% budget BM25 loses 53 such functions. Our ranker keeps 45 of them, and loses 12 of its own.
- **Functions too large for the budget are rare at 25%** and more common at 10%. In those cases the ranker often puts the function first, so this is a limit of the budget, not of the ranking.
- **Signature-only cases are much more common for BM25.** It tends to put large, word-heavy nodes first, which use up the budget before the gold function's turn.
- **Naming the function does not guarantee a hit.** A few issues mention the gold function in passing next to many others, and other code that looks more relevant ranks above it.

## Case studies

### 1. The symptom is far from the fix, and training helps

`django__django-15388`. The issue says that the development server stops restarting after the user adds `BASE_DIR` to `TEMPLATES[0]['DIRS']` in settings. The fix is in `template_changed` in `django/template/autoreload.py`, a function the issue never names.

The words in the issue (server, restart, settings, DIRS) share little with the function's text, so BM25 ranks it 37th of 79 nodes and drops it at the 25% budget. Our ranker keeps it in full. This is the most common pattern behind the gap between the two methods.

### 2. Our ranker misses, the issue gives no handle

`sympy__sympy-13146`, "Exponent doesn't fully simplify". The user computes `simplify(res.evalf(5))` and gets `-0.5*x**2.5 + 0.5*x**2.5` instead of 0. The fix is in `AssocOp._eval_evalf` in `sympy/core/operations.py`. It makes `evalf` always rebuild the sum from its evaluated terms, so that equal terms cancel.

The issue names neither the class, the method nor the module. The candidate files hold 576 nodes, many of them about simplification and evaluation, and our ranker places the gold function 184th, below what fits in the 30,665-token budget. Finding it needs knowledge of how SymPy evaluates expressions, which a 125M encoder reading the first 256 tokens of each function does not have. This issue is also the third demo sample.

### 3. Ranked first, too large to keep

`django__django-15851`. The issue says that `dbshell` passes extra parameters after the database name, which `psql` ignores. The candidate files are small and the 25% budget is 166 tokens. The gold function, `DatabaseClient.settings_to_cmd_args_env`, is 359 tokens.

Our ranker puts it first of 9 nodes. It cannot fit, so the selector shows its signature. This is the intended behaviour: the agent sees exactly which function to open, and the MCP tool's description tells it to call again with a larger budget. In our numbers it still counts as a miss.

## What this suggests

- A function that is too large for the budget could be shown as its signature plus its first lines, instead of the signature alone. That keeps the most likely fix site partly visible at small budgets.
- Symptom-style issues are where the ranker gains most and where it still fails most. More training data with that pattern, or giving the ranker more than the head of each function, are the obvious next steps.
- None of the misses at these budgets came from parsing. Every gold function was found as a node.

## All cases

<!-- table:lite_r0.25_error_cases_ast_trained -->
Every gold function AST + trained ranker (ours) lost, lite, 25% budget.

| Issue | Function | Its tokens | Budget | Shown as | Score rank | Cause |
|---|---|---|---|---|---|---|
| astropy__astropy-14182 | `RST.__init__` | 20 | 4511 | dropped | 56 of 125 | outranked, issue names it |
| django__django-14997 | `Expressions.rename_table_references` | 77 | 405 | dropped | 11 of 45 | outranked, issue does not name it |
| django__django-12113 | `DatabaseCreation.test_db_signature` | 112 | 795 | signature | 5 of 29 | only its signature fit |
| django__django-13265 | `MigrationAutodetector._detect_changes` | 763 | 6685 | dropped | 51 of 255 | outranked, issue does not name it |
| django__django-15851 | `DatabaseClient.settings_to_cmd_args_env` | 359 | 166 | signature | 1 of 9 | function too large for the budget |
| matplotlib__matplotlib-24149 | `Axes._convert_dx` | 443 | 75053 | dropped | 172 of 1430 | outranked, issue names it |
| pydata__xarray-4493 | `as_variable` | 634 | 30582 | dropped | 283 of 736 | outranked, issue does not name it |
| pylint-dev__pylint-7114 | `expand_modules` | 770 | 507 | dropped | 1 of 15 | function too large for the budget |
| scikit-learn__scikit-learn-25638 | `is_multilabel` | 528 | 5805 | dropped | 20 of 67 | outranked, issue does not name it |
| scikit-learn__scikit-learn-25638 | `type_of_target` | 1565 | 5805 | dropped | 16 of 67 | outranked, issue does not name it |
| scikit-learn__scikit-learn-25570 | `ColumnTransformer._hstack` | 422 | 17062 | dropped | 77 of 344 | outranked, issue names it |
| sympy__sympy-12419 | `MatrixElement._eval_derivative` | 111 | 8759 | dropped | 82 of 242 | outranked, issue does not name it |
| sympy__sympy-12419 | `Identity._entry` | 29 | 8759 | dropped | 108 of 242 | outranked, issue does not name it |
| sympy__sympy-13915 | `Mul.flatten` | 4708 | 39196 | dropped | 229 of 967 | outranked, issue does not name it |
| sympy__sympy-11870 | `sinc._eval_rewrite_as_sin` | 19 | 45197 | dropped | 253 of 1454 | outranked, issue does not name it |
| sympy__sympy-18087 | `Factors.__init__` | 976 | 48555 | dropped | 446 of 1076 | outranked, issue does not name it |
| sympy__sympy-20322 | `Mul._eval_is_integer` | 161 | 44579 | signature | 136 of 1043 | only its signature fit |
| sympy__sympy-13146 | `AssocOp._eval_evalf` | 563 | 30665 | dropped | 184 of 576 | outranked, issue does not name it |
| sympy__sympy-12236 | `PolynomialRing.from_FractionField` | 73 | 5233 | dropped | 69 of 274 | outranked, issue does not name it |
<!-- /table -->

The same lists for AST + BM25 and for the 10% budget are in `results/tables/lite_r0.25_error_cases_ast_bm25.md`, `results/tables/lite_r0.10_error_cases_ast_trained.md` and `results/tables/lite_r0.10_error_cases_ast_bm25.md`.
