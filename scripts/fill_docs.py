"""Copy generated tables into README.md and docs between marker comments.

    python scripts/fill_docs.py

A marker pair looks like  <!-- table:lite_oracle_imports_main -->  ...  <!-- /table -->
and is filled from results/tables/<name>.md. Numbers in the docs therefore
always come from result files. Rerun after any experiment changes.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PATTERN = re.compile(r"(<!-- table:([\w.-]+) -->)(.*?)(<!-- /table -->)", re.S)


def fill(path: Path) -> int:
    text = path.read_text()
    missing = []

    def repl(match: re.Match) -> str:
        source = TABLES / f"{match.group(2)}.md"
        if not source.exists():
            missing.append(match.group(2))
            return match.group(0)
        return f"{match.group(1)}\n{source.read_text().strip()}\n{match.group(4)}"

    new_text, n = PATTERN.subn(repl, text)
    if new_text != text:
        path.write_text(new_text)
    for name in missing:
        print(f"  {path.name}: no table named {name} yet")
    return n


if __name__ == "__main__":
    total = 0
    for doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        if doc.exists():
            total += fill(doc)
    print(f"filled {total} table blocks")
    sys.exit(0)
