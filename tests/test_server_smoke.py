"""Starts the real MCP server as a subprocess and talks to it over stdio.

Uses BM25 ranking so the test needs no GPU and no checkpoint.
"""

import asyncio
import os
import sys

import pytest
from mcp import Client, StdioServerParameters

ISSUE = "Shape.area returns 0 for every shape, it should use the sides"


def server_params(root) -> StdioServerParameters:
    env = dict(os.environ, ASTPRUNE_ROOT=str(root), ASTPRUNE_SCORER="bm25")
    return StdioServerParameters(command=sys.executable, args=["-m", "astprune.server"], env=env)


@pytest.fixture
def workspace(tmp_path, sample_source):
    (tmp_path / "shapes.py").write_text(sample_source)
    return tmp_path


def test_server_lists_and_runs_the_tool(workspace):
    async def scenario():
        async with Client(server_params(workspace)) as client:
            tools = await client.list_tools()
            assert [t.name for t in tools.tools] == ["get_budgeted_context"]
            schema = tools.tools[0].input_schema
            assert set(schema["required"]) == {"issue", "files"}

            result = await client.call_tool(
                "get_budgeted_context", {"issue": ISSUE, "files": ["shapes.py"], "budget": 0.3})
            assert not result.is_error
            data = result.structured_content
            assert data["kept_tokens"] <= data["budget_tokens"] < data["original_tokens"]
            assert "def area(self):" in data["context"]
            return data

    asyncio.run(scenario())


def test_server_survives_bad_requests(workspace):
    async def scenario():
        async with Client(server_params(workspace)) as client:
            bad_budget = await client.call_tool(
                "get_budgeted_context", {"issue": ISSUE, "files": ["shapes.py"], "budget": -5})
            assert bad_budget.is_error
            assert "budget must be above 0" in bad_budget.content[0].text

            no_files = await client.call_tool(
                "get_budgeted_context", {"issue": ISSUE, "files": ["missing.py"], "budget": 0.3})
            assert no_files.is_error
            assert "none of the files could be read" in no_files.content[0].text

            wrong_type = await client.call_tool(
                "get_budgeted_context", {"issue": ISSUE, "files": "shapes.py"})
            assert wrong_type.is_error

            empty_issue = await client.call_tool(
                "get_budgeted_context", {"issue": "", "files": ["shapes.py"], "budget": 0.3})
            assert not empty_issue.is_error
            assert empty_issue.structured_content["ranking"] == "skeleton"

            # After four odd requests the server still answers a normal one.
            ok = await client.call_tool(
                "get_budgeted_context", {"issue": ISSUE, "files": ["shapes.py"], "budget": 200})
            assert not ok.is_error and ok.structured_content["kept_tokens"] <= 200

    asyncio.run(scenario())
