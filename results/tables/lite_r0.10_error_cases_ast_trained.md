Every gold function AST + trained ranker (ours) lost, lite, 10% budget.

| Issue | Function | Its tokens | Budget | Shown as | Score rank | Cause |
|---|---|---|---|---|---|---|
| astropy__astropy-14365 | `_line_type` | 619 | 2288 | dropped | 6 of 157 | outranked, issue does not name it |
| astropy__astropy-14182 | `RST.__init__` | 20 | 1804 | dropped | 56 of 125 | outranked, issue names it |
| django__django-14787 | `_multi_decorate` | 298 | 139 | signature | 2 of 12 | function too large for the budget |
| django__django-16229 | `BoundField.as_widget` | 170 | 2207 | dropped | 26 of 340 | outranked, issue does not name it |
| django__django-11283 | `update_proxy_model_permissions` | 256 | 134 | dropped | 1 of 18 | function too large for the budget |
| django__django-14997 | `Expressions.rename_table_references` | 77 | 162 | dropped | 11 of 45 | outranked, issue does not name it |
| django__django-11797 | `Exact.process_rhs` | 108 | 6246 | dropped | 48 of 813 | outranked, issue does not name it |
| django__django-17087 | `FunctionTypeSerializer.serialize` | 227 | 1444 | signature | 9 of 195 | only its signature fit |
| django__django-12184 | `RegexPattern.match` | 107 | 2528 | dropped | 14 of 325 | outranked, issue does not name it |
| django__django-15814 | `Query.deferred_to_data` | 828 | 7818 | dropped | 40 of 959 | outranked, issue does not name it |
| django__django-12113 | `DatabaseCreation.test_db_signature` | 112 | 318 | dropped | 5 of 29 | outranked, issue does not name it |
| django__django-11630 | `check_all_models` | 528 | 426 | signature | 1 of 40 | function too large for the budget |
| django__django-15819 | `Command.handle_inspection` | 1495 | 804 | signature | 2 of 58 | function too large for the budget |
| django__django-13265 | `MigrationAutodetector._detect_changes` | 763 | 2674 | dropped | 51 of 255 | outranked, issue does not name it |
| django__django-15851 | `DatabaseClient.settings_to_cmd_args_env` | 359 | 66 | signature | 1 of 9 | function too large for the budget |
| django__django-15388 | `template_changed` | 53 | 735 | dropped | 9 of 79 | outranked, issue does not name it |
| django__django-16820 | `CreateModel.reduce` | 962 | 3233 | signature | 22 of 374 | only its signature fit |
| django__django-16046 | `format` | 821 | 365 | signature | 1 of 41 | function too large for the budget |
| matplotlib__matplotlib-25433 | `FigureBase.delaxes` | 296 | 19655 | dropped | 172 of 1268 | outranked, issue does not name it |
| matplotlib__matplotlib-24149 | `Axes._convert_dx` | 443 | 30021 | dropped | 172 of 1430 | outranked, issue names it |
| matplotlib__matplotlib-24265 | `use` | 812 | 1764 | signature | 8 of 102 | only its signature fit |
| matplotlib__matplotlib-26020 | `_tick_only` | 66 | 5209 | dropped | 24 of 340 | outranked, issue names it |
| mwaskom__seaborn-3010 | `PolyFit.__call__` | 28 | 59 | signature | 1 of 9 | only its signature fit |
| psf__requests-2674 | `HTTPAdapter.send` | 795 | 3048 | signature | 17 of 308 | only its signature fit |
| pydata__xarray-4248 | `inline_variable_array_repr` | 127 | 3260 | dropped | 25 of 235 | outranked, issue does not name it |
| pydata__xarray-4493 | `as_variable` | 634 | 12232 | dropped | 283 of 736 | outranked, issue does not name it |
| pylint-dev__pylint-6506 | `_config_initialization` | 748 | 730 | dropped | 2 of 66 | function too large for the budget |
| pylint-dev__pylint-7114 | `expand_modules` | 770 | 203 | dropped | 1 of 15 | function too large for the budget |
| pytest-dev__pytest-5495 | `_compare_eq_sequence` | 226 | 500 | dropped | 3 of 53 | outranked, issue does not name it |
| pytest-dev__pytest-5413 | `ExceptionInfo.__str__` | 57 | 1073 | signature | 8 of 144 | only its signature fit |
| pytest-dev__pytest-7168 | `_format_repr_exception` | 106 | 81 | signature | 1 of 13 | function too large for the budget |
| pytest-dev__pytest-7220 | `Node._repr_failure_py` | 369 | 5435 | dropped | 40 of 502 | outranked, issue does not name it |
| scikit-learn__scikit-learn-10949 | `check_array` | 2227 | 1848 | dropped | 1 of 90 | function too large for the budget |
| scikit-learn__scikit-learn-25638 | `is_multilabel` | 528 | 2322 | dropped | 20 of 67 | outranked, issue does not name it |
| scikit-learn__scikit-learn-25638 | `type_of_target` | 1565 | 2322 | dropped | 16 of 67 | function too large for the budget |
| scikit-learn__scikit-learn-25570 | `ColumnTransformer._hstack` | 422 | 6825 | dropped | 77 of 344 | outranked, issue names it |
| sphinx-doc__sphinx-8801 | `get_class_members` | 589 | 5082 | dropped | 25 of 443 | outranked, issue does not name it |
| sympy__sympy-16792 | `CodeGen.routine` | 1386 | 7084 | signature | 21 of 520 | only its signature fit |
| sympy__sympy-21627 | `Abs.eval` | 798 | 21178 | dropped | 239 of 1433 | outranked, issue does not name it |
| sympy__sympy-17139 | `_TR56` | 532 | 25673 | signature | 65 of 1581 | only its signature fit |
| sympy__sympy-13031 | `MutableSparseMatrix.col_join` | 424 | 15891 | signature | 68 of 1055 | only its signature fit |
| sympy__sympy-13031 | `MutableSparseMatrix.row_join` | 463 | 15891 | dropped | 88 of 1055 | outranked, issue does not name it |
| sympy__sympy-12419 | `MatrixElement._eval_derivative` | 111 | 3503 | dropped | 82 of 242 | outranked, issue does not name it |
| sympy__sympy-12419 | `Identity._entry` | 29 | 3503 | dropped | 108 of 242 | outranked, issue does not name it |
| sympy__sympy-13895 | `Integer._eval_power` | 1052 | 16964 | dropped | 92 of 1098 | outranked, issue does not name it |
| sympy__sympy-13915 | `Mul.flatten` | 4708 | 15678 | dropped | 229 of 967 | outranked, issue does not name it |
| sympy__sympy-11870 | `sinc._eval_rewrite_as_sin` | 19 | 18078 | dropped | 253 of 1454 | outranked, issue does not name it |
| sympy__sympy-21379 | `Mod.eval` | 1471 | 16256 | signature | 39 of 1089 | only its signature fit |
| sympy__sympy-18087 | `Factors.__init__` | 976 | 19422 | dropped | 446 of 1076 | outranked, issue does not name it |
| sympy__sympy-18087 | `Factors.as_expr` | 168 | 19422 | dropped | 57 of 1076 | outranked, issue does not name it |
| sympy__sympy-15346 | `_futrig` | 703 | 17094 | dropped | 87 of 1281 | outranked, issue does not name it |
| sympy__sympy-20322 | `Mul._eval_is_integer` | 161 | 17831 | dropped | 136 of 1043 | outranked, issue does not name it |
| sympy__sympy-18621 | `BlockDiagMatrix.blocks` | 81 | 3611 | dropped | 27 of 341 | outranked, issue names it |
| sympy__sympy-13146 | `AssocOp._eval_evalf` | 563 | 12266 | dropped | 184 of 576 | outranked, issue does not name it |
| sympy__sympy-12236 | `PolynomialRing.from_FractionField` | 73 | 2093 | dropped | 69 of 274 | outranked, issue does not name it |
| sympy__sympy-11897 | `LatexPrinter._needs_mul_brackets` | 228 | 12042 | signature | 83 of 676 | only its signature fit |
