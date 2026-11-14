"""Export a few benchmark issues as self-contained demo inputs.

    python demo/make_samples.py --split lite --ids django__django-11099 sympy__sympy-13480

For each issue this copies the gold file and a few of the files it imports, as
they were before the fix, into demo/sample_repo/<issue id>/ and adds an entry
to demo/sample_inputs.json. Needs the data cache (see data/README.md). The
exported samples are committed, so running the demo does not need the cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from astprune.data.datasets import load_processed
from astprune.data.repos import Repo

DEMO = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--imports", type=int, default=3, help="imported files to include per issue")
    ap.add_argument("--notes", nargs="*", default=[], help="one short note per id, shown as the sample title suffix")
    args = ap.parse_args()

    by_id = {r["instance_id"]: r for r in load_processed(args.split)}
    samples = []
    for k, instance_id in enumerate(args.ids):
        rec = by_id[instance_id]
        files = sorted(rec["oracle_files"]) + rec["import_files"][: args.imports]
        root = DEMO / "sample_repo" / instance_id
        with Repo(rec["repo"]) as repo:
            for path in files:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(repo.read(rec["base_commit"], path) or "")
        title = next((ln.strip() for ln in rec["issue"].splitlines() if ln.strip()), instance_id)[:90]
        samples.append({
            "id": instance_id,
            "title": title + (f" ({args.notes[k]})" if k < len(args.notes) else ""),
            "source": f"SWE-bench {args.split}, {rec['repo']} at {rec['base_commit'][:10]}",
            "issue": rec["issue"],
            "root": f"sample_repo/{instance_id}",
            "files": files,
            "gold": [f"{g['path']}::{g['name']}" for g in rec["gold_functions"]],
        })
        print(f"{instance_id}: {len(files)} files -> {root.relative_to(DEMO)}")
    (DEMO / "sample_inputs.json").write_text(json.dumps(samples, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
