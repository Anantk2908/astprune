"""Live demo: an MCP client asks the pruning server for a budgeted view of some files.

    python demo/app.py                         # first sample issue, 25% budget
    python demo/app.py --list                  # show the bundled samples
    python demo/app.py --sample 2 --budget 0.1
    python demo/app.py --root ~/code/myrepo --files pkg/a.py pkg/b.py --issue "crash when x is None"
    python demo/app.py --scorer bm25           # no GPU and no checkpoint needed

This script holds no pruning logic. It starts `python -m astprune.server` as a
subprocess, talks to it over MCP exactly like an agent would, and prints what
came back: tokens before and after, what was kept, and the pruned code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import Client, StdioServerParameters
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

DEMO_DIR = Path(__file__).resolve().parent
SAMPLES_FILE = DEMO_DIR / "sample_inputs.json"
TOOL = "get_budgeted_context"
CALL_TIMEOUT_S = 180
console = Console()


def load_samples() -> list[dict]:
    if not SAMPLES_FILE.exists():
        return []
    return json.loads(SAMPLES_FILE.read_text())


def list_samples(samples: list[dict]) -> None:
    table = Table(title="Bundled sample issues")
    for col in ("#", "id", "title", "files"):
        table.add_column(col)
    for i, s in enumerate(samples, start=1):
        table.add_row(str(i), s["id"], s["title"], str(len(s["files"])))
    console.print(table)


def request_from_args(args, samples: list[dict]) -> dict:
    """Work out what to ask the server: a bundled sample or the user's own files."""
    if args.files:
        if args.issue_file:
            issue = Path(args.issue_file).read_text()
        else:
            issue = args.issue or ""
        return {"id": "custom", "title": "your files", "issue": issue, "files": args.files,
                "root": str(Path(args.root or ".").resolve()), "gold": []}
    if not samples:
        sys.exit(f"No samples found at {SAMPLES_FILE}. Pass --files and --issue to use your own.")
    if not 1 <= args.sample <= len(samples):
        sys.exit(f"--sample must be between 1 and {len(samples)} (see --list)")
    sample = dict(samples[args.sample - 1])
    sample["root"] = str((DEMO_DIR / sample["root"]).resolve())
    return sample


async def ask_server(request: dict, budget: float, scorer: str | None) -> tuple[dict | None, str, float]:
    """One MCP round trip. Returns (result, error message, seconds until the server was ready)."""
    env = dict(os.environ, ASTPRUNE_ROOT=request["root"])
    if scorer:
        env["ASTPRUNE_SCORER"] = scorer
    params = StdioServerParameters(command=sys.executable, args=["-m", "astprune.server"], env=env)
    started = time.perf_counter()
    async with Client(params, read_timeout_seconds=CALL_TIMEOUT_S) as client:
        tools = await client.list_tools()
        ready_s = time.perf_counter() - started
        if TOOL not in [t.name for t in tools.tools]:
            return None, f"the server does not offer {TOOL}", ready_s
        result = await client.call_tool(
            TOOL, {"issue": request["issue"], "files": request["files"], "budget": budget})
        if result.is_error:
            message = result.content[0].text if result.content else "unknown error"
            return None, message, ready_s
        return result.structured_content, "", ready_s


def gold_status(gold: list[str], files: list[dict]) -> list[tuple[str, str]]:
    """For samples where we know which function the real fix changed, say what happened to it."""
    by_path = {f["path"]: f for f in files}
    out = []
    for item in gold:
        path, _, name = item.partition("::")
        info = by_path.get(path)
        if info is None:
            out.append((item, "[red]file missing[/red]"))
        elif name in info["kept_full"]:
            out.append((item, "[green]kept in full[/green]"))
        elif name in info["kept_signature"]:
            out.append((item, "[yellow]signature only[/yellow]"))
        else:
            out.append((item, "[red]dropped[/red]"))
    return out


def show(request: dict, data: dict, ready_s: float, show_lines: int) -> None:
    saved = data["original_tokens"] - data["kept_tokens"]
    console.print(Panel(
        f"[bold]{data['original_tokens']:,}[/bold] tokens if the agent reads the files whole\n"
        f"[bold]{data['kept_tokens']:,}[/bold] tokens in the pruned view "
        f"(budget {data['budget_tokens']:,})\n"
        f"[bold]{saved:,}[/bold] tokens saved ({data['saved_percent']}%)",
        title="Full files vs pruned view", expand=False))

    table = Table(title="Per file")
    table.add_column("file")
    table.add_column("tokens", justify="right")
    table.add_column("shown", justify="right")
    table.add_column("kept in full")
    table.add_column("signature only", justify="right")
    table.add_column("dropped", justify="right")
    for f in data["files"]:
        names = ", ".join(f["kept_full"][:4]) + (f" (+{len(f['kept_full']) - 4} more)" if len(f["kept_full"]) > 4 else "")
        table.add_row(f["path"], f"{f['original_tokens']:,}", f"{f['kept_tokens']:,}",
                      names or "-", str(len(f["kept_signature"])), str(f["dropped"]))
    console.print(table)

    for item, status in gold_status(request.get("gold", []), data["files"]):
        console.print(f"Function the real fix changed: [bold]{item}[/bold] -> {status}")
    for warning in data["warnings"]:
        console.print(f"[yellow]note:[/yellow] {warning}")
    console.print(f"[dim]ranking: {data['ranking']} on {data['device']} | server ready in {ready_s:.1f}s | "
                  f"pruning took {data['latency_ms']:.0f} ms[/dim]")

    if show_lines:
        lines = data["context"].split("\n")
        shown = "\n".join(lines[:show_lines])
        console.print(Panel(Syntax(shown, "python", line_numbers=False, word_wrap=False),
                            title=f"Pruned view (first {min(show_lines, len(lines))} of {len(lines)} lines)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the bundled samples and exit")
    ap.add_argument("--sample", type=int, default=1, help="which bundled sample to run (default 1)")
    ap.add_argument("--budget", type=float, default=0.25,
                    help="up to 1: share of the files' tokens. Above 1: a token count (default 0.25)")
    ap.add_argument("--issue", help="issue text for your own files")
    ap.add_argument("--issue-file", help="read the issue text from a file")
    ap.add_argument("--files", nargs="+", help="your own files, relative to --root")
    ap.add_argument("--root", help="repository root for --files (default: current folder)")
    ap.add_argument("--scorer", choices=["auto", "hybrid", "trained", "bm25"],
                    help="ranking to use (default: auto, which falls back to bm25 without a checkpoint)")
    ap.add_argument("--show-lines", type=int, default=60, help="lines of the pruned view to print, 0 for none")
    ap.add_argument("--json", action="store_true", help="print the raw tool result as JSON")
    args = ap.parse_args()

    samples = load_samples()
    if args.list:
        list_samples(samples)
        return 0
    request = request_from_args(args, samples)

    console.print(Panel(request["issue"].strip()[:700] or "(no issue text)",
                        title=f"Issue: {request['title']}", expand=False))
    console.print(f"Files the agent wants to read: {', '.join(request['files'])}")
    try:
        with console.status("Starting the MCP server and loading the ranker..."):
            data, error, ready_s = asyncio.run(ask_server(request, args.budget, args.scorer))
    except KeyboardInterrupt:
        console.print("[yellow]Stopped.[/yellow]")
        return 130
    except Exception as err:  # noqa: BLE001 - the demo must end with a readable message, not a traceback
        root = err.exceptions[0] if isinstance(err, BaseExceptionGroup) and err.exceptions else err
        console.print(f"[red]Could not talk to the MCP server:[/red] {type(root).__name__}: {root}")
        console.print("Check that the package is installed in this environment: pip install -e .")
        return 1
    if data is None:
        console.print(f"[red]The server refused the request:[/red] {error}")
        return 2
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    show(request, data, ready_s, args.show_lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
