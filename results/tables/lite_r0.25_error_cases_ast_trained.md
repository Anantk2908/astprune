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
