"""Run the demo path many times and record how often it works and how long it takes.

    python experiments/demo_reliability.py --runs 20

Each run starts a fresh MCP server (so start-up and model loading are
included), sends one bundled sample at a random budget, and checks the answer.
A few deliberately bad requests are mixed in to confirm the server reports
them and keeps going. Output: results/demo_reliability.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

from mcp import Client, StdioServerParameters

from astprune.data.repos import PROJECT_ROOT

DEMO = PROJECT_ROOT / "demo"


async def one_run(sample: dict, budget: float, scorer: str | None) -> dict:
    env = dict(os.environ, ASTPRUNE_ROOT=str(DEMO / sample["root"]))
    if scorer:
        env["ASTPRUNE_SCORER"] = scorer
    params = StdioServerParameters(command=sys.executable, args=["-m", "astprune.server"], env=env)
    t0 = time.perf_counter()
    async with Client(params, read_timeout_seconds=180) as client:
        await client.list_tools()
        ready = time.perf_counter() - t0
        t1 = time.perf_counter()
        good = await client.call_tool("get_budgeted_context",
                                      {"issue": sample["issue"], "files": sample["files"], "budget": budget})
        first_call = time.perf_counter() - t1
        t2 = time.perf_counter()
        again = await client.call_tool("get_budgeted_context",
                                       {"issue": sample["issue"], "files": sample["files"], "budget": budget})
        second_call = time.perf_counter() - t2
        bad = await client.call_tool("get_budgeted_context", {"issue": "x", "files": ["missing.py"], "budget": -1})
        after_bad = await client.call_tool("get_budgeted_context",
                                           {"issue": sample["issue"], "files": sample["files"][:1], "budget": 0.5})
    data = good.structured_content or {}
    ok = (not good.is_error and not again.is_error and bad.is_error and not after_bad.is_error
          and data.get("kept_tokens", 10**9) <= data.get("budget_tokens", -1))
    return {"ok": bool(ok), "ready_s": ready, "first_call_s": first_call, "second_call_s": second_call,
            "server_ms": data.get("latency_ms"), "ranking": data.get("ranking"), "device": data.get("device"),
            "sample": sample["id"], "budget": budget}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--scorer", default=None)
    args = ap.parse_args()
    samples = json.loads((DEMO / "sample_inputs.json").read_text())
    rng = random.Random(0)
    runs = []
    for i in range(args.runs):
        sample = samples[i % len(samples)]
        budget = rng.choice([0.1, 0.25, 0.5])
        try:
            runs.append(asyncio.run(one_run(sample, budget, args.scorer)))
        except Exception as err:  # noqa: BLE001 - a crash is exactly what this script is counting
            runs.append({"ok": False, "error": f"{type(err).__name__}: {err}", "sample": sample["id"], "budget": budget})
        print(f"  run {i + 1}/{args.runs}: {'ok' if runs[-1]['ok'] else 'FAILED'}", flush=True)
    good = [r for r in runs if r["ok"]]

    def stats(key: str) -> dict:
        values = sorted(r[key] for r in good)
        if not values:
            return {}
        return {"median": round(statistics.median(values), 3), "p90": round(values[int(0.9 * (len(values) - 1))], 3),
                "max": round(values[-1], 3)}

    out = {"runs": len(runs), "succeeded": len(good), "ranking": good[0]["ranking"] if good else None,
           "device": good[0]["device"] if good else None,
           "server_ready_seconds": stats("ready_s"), "first_call_seconds": stats("first_call_s"),
           "repeat_call_seconds": stats("second_call_s"),
           "what_a_run_checks": "start server, list tools, prune a sample, prune it again, send a bad request "
                                "and get an error result, prune once more",
           "failures": [r for r in runs if not r["ok"]]}
    path = PROJECT_ROOT / "results" / "demo_reliability.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
