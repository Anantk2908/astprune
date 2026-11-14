# Prompt for the live agent step

Start Claude Code (or any MCP client) in the repository root so it loads `.mcp.json`. Then paste:

```
I am working on the bug described in demo/sample_inputs.json, first entry (field "issue").
The code is under demo/sample_repo/<id of that entry>/. Do not read those source files directly.
Call the astprune tool get_budgeted_context with the issue text, the files listed in that entry
(prefixed with the sample_repo path), and budget 0.25.
Then tell me: how many tokens the full files have, how many you received,
and which function you would edit to fix the issue.
```

The server resolves paths relative to the folder it was started in, which for `.mcp.json` is the repository root.
