"""Scorers rank AST nodes against an issue. All share one method:

    score(issue: str, files: list[ParsedFile]) -> dict[path, np.ndarray]

with one score per node, higher meaning more relevant.
"""
