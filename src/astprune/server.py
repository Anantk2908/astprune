"""MCP server exposing the pruner as one tool.

    python -m astprune.server

Speaks MCP over stdio, so any MCP client can start it as a subprocess. Settings
come from environment variables (see .env.example):

    ASTPRUNE_ROOT        folder that file paths are relative to (default: where the server starts)
    ASTPRUNE_SCORER      auto | hybrid | trained | bm25 (default: auto)
    ASTPRUNE_CHECKPOINT  trained ranker folder (default: checkpoints/ranker/best)
    ASTPRUNE_DEVICE      cuda | cpu (default: cuda if available)

stdout carries the protocol, so everything else is logged to stderr.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Quiet libraries that like to print, before they are imported.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from .service import PruningError, PruningService  # noqa: E402

INSTRUCTIONS = (
    "Use get_budgeted_context instead of reading whole Python files when you are working on a bug "
    "report or feature request. Pass the issue text and the files you would have opened. You get back "
    "the functions most likely to matter in full, others as signatures, and a note for what was left "
    "out. If the function you need is only shown as a signature or is missing, call again with a "
    "larger budget or read that file directly."
)

mcp = MCPServer("astprune", instructions=INSTRUCTIONS)
_service: PruningService | None = None


def service() -> PruningService:
    global _service
    if _service is None:
        _service = PruningService().load()
    return _service


@mcp.tool(structured_output=True)
def get_budgeted_context(issue: str, files: list[str], budget: float = 0.25) -> dict[str, Any]:
    """Return a token-budgeted view of Python files, ranked for one issue.

    Args:
        issue: The bug report or task description. Used to decide which functions matter.
        files: Paths of the files to look at, relative to the workspace root.
        budget: Up to 1 it is a share of the files' tokens (0.25 keeps a quarter).
            Above 1 it is a token count (2000 keeps about 2000 tokens).

    Returns a dict with ``context`` (the pruned code, one section per file),
    token counts before and after, which functions were kept in full or as
    signatures, and ``warnings`` for anything that was skipped or looked wrong.
    """
    try:
        return service().get_budgeted_context(issue, files, budget)
    except PruningError as err:
        # ToolError is the SDK's way to say "the request was wrong": the client
        # gets the message as an error result and the server keeps running.
        raise ToolError(str(err)) from None


def main() -> None:
    try:
        svc = service()  # load the model before accepting calls, so the first call is not slow
    except PruningError as err:
        print(f"astprune: {err}", file=sys.stderr)
        raise SystemExit(2) from None
    print(f"astprune: ready (ranking={svc.scorer_name}, device={svc.device_name}, root={svc.root})",
          file=sys.stderr, flush=True)
    for note in svc.startup_notes:
        print(f"astprune: {note}", file=sys.stderr, flush=True)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
