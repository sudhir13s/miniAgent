# Tools

Tools are what an agent can do. List them by name on the profile; MCP servers
add whole sets of them.

## Built-in tools

| Name | Does |
|------|------|
| `calculator` | Safe arithmetic, e.g. `(2+3)*4` |
| `timestamp` | Current UTC time |
| `word_count` | Words, characters, lines |
| `text_transform` | upper / lower / title / reverse / strip |
| `json_formatter` | Parse and pretty-print JSON |
| `uuid_generator` | UUID v1 or v4 |
| `echo` | Echo the input (testing) |
| `file_reader` | Read a file under the working directory |
| `web_fetch` | Fetch a URL's text |
| `weather` | Current weather via wttr.in, no key needed |

## Your own tool: `@tool`

```python
from typing import Literal
from miniagent import build_agent
from miniagent.tool import tool

@tool
async def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
        unit: Temperature unit for the answer.
    """
    return f"{city}: 21 degrees {unit}"

agent = build_agent(profile, tools={"get_weather": get_weather})  # profile.tools lists it
```

- **Schema from type hints and the docstring.** Defaults make a parameter optional.
- **Arguments are validated** before your function runs; bad ones go back to the model.
- **Sync functions run in a thread**, so they never block the event loop.
- **Non-string results are sent as JSON.**
- **Options:** `@tool(name="lookup", description="...", timeout_seconds=10)`.

For full control, subclass `BaseTool`: give it a `ToolSchema` (name, description,
JSON Schema parameters) and implement `async invoke(**kwargs) -> str`.

## How tool calls run

- **Concurrently:** several calls in one turn run at once; results keep the
  model's order.
- **Failures go back to the model**, which can correct itself. The run continues.

| What went wrong | The model sees |
|-----------------|----------------|
| Unknown tool | `Tool 'x' is not registered with this agent.` |
| Tool raised `ToolInvocationError` | `Tool 'x' failed: <your message>` |
| Tool raised anything else | `Tool 'x' failed: KeyError: 'id'` |
| Took longer than `timeout_seconds` (default 60) | `Tool 'x' timed out after 60 seconds.` |
| Returned an empty string (not an error) | `(no output)` |

## MCP servers

Any [Model Context Protocol](https://modelcontextprotocol.io) server can supply
tools. Install the extra: `pip install -e ".[mcp]"`.

Declare servers once, then let agents pick them:

```yaml
mcp_servers:
  - name: github
    command: github-mcp-server         # stdio: started as a subprocess
    args: [stdio]
    env: {GITHUB_TOKEN: "${GITHUB_TOKEN}"}
    name_prefix: gh_
    tool_filter: [create_issue, search_issues]
  - name: docs
    url: https://docs.example.com/mcp  # Streamable HTTP
    headers: {Authorization: "Bearer ${DOCS_TOKEN}"}
    timeout_seconds: 30

agents:
  - name: triager
    model: {id: openai/gpt-4o}
    toolsets: [github, docs]
```

```python
from miniagent import build_agent
from miniagent.profile import AgentProfile, MCPServerConfig
from miniagent.tool.mcp import MCPToolset
from miniagent.tool.toolset import open_toolsets

toolsets = [MCPToolset.from_config(server) for server in MCPServerConfig.from_yaml("agents.yaml")]
async with open_toolsets(toolsets):          # connects all, always closes all
    profile = AgentProfile.from_yaml("agents.yaml", name="triager")
    agent = build_agent(profile, toolsets=toolsets)
    final_state = await agent.run(state)
```

- **Secrets stay out of the file:** `${VAR}` is read from the environment when
  the toolset is built (`MCPToolset.from_config`). A missing variable is an error naming it, never its value.
- **The REST API connects servers for you** at startup and closes them at shutdown.
- **Tools are re-listed every run**, so server-side changes need no restart.
- **Same tool name from two sources fails the run.** Fix it with `name_prefix` or `tool_filter`.
- **In code without YAML:** `MCPToolset.stdio(name, StdioServerParameters(...))`
  or `MCPToolset.http(name, url, headers=...)`.
- **A working example** of both transports is in [`agents.example.yaml`](../../agents.example.yaml).

**Your own tool source:** subclass `BaseToolset` and implement `async list_tools()`.
