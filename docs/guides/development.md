# Development

## Setup and checks

```bash
uv sync --extra dev --extra mcp     # or: pip install -e ".[dev,mcp]"
uv run pytest                       # no API keys needed; model calls are mocked
uv run ruff check src tests
uv run ruff format src tests
uv run mypy src/miniagent           # strict
```

Coverage must stay at or above 90% (`pytest --cov=miniagent`).

## Package layout

Each package owns one feature; `core/` is the shared kernel every feature uses.

| Package | Holds |
|---------|-------|
| `core/` | `Message`, `State`, the exception hierarchy, naming rules |
| `profile/` | `AgentProfile` (with `from_yaml`), `ModelConfig`, templates, `MCPServerConfig`, YAML file reading |
| `llm/` | `BaseLLMProvider`, `LiteLLMProvider`, default-provider registry |
| `agent/` | `BaseAgent`, `Agent` (the loop), tool invocation, `build_agent` |
| `tool/` | `BaseTool`, `@tool`, built-in tools, `BaseToolset`, MCP (`tool/mcp/`) |
| `memory/` | Context window policies, session stores |
| `workflow/` | Sequential, parallel, loop and graph workflows, registry |
| `api/` | FastAPI app, routes, lifespan (MCP connections) |

- **`base.py`** holds the abstract base class in `llm`, `agent`, `tool` and
  `workflow`. `memory`'s are in `context_window.py` and `session_store.py`;
  `BaseToolset` is in `tool/toolset.py`.
- **Features import `core` and each other's public API**, never internals.

## Where each contract is defined

Names and structures are defined once and imported, never retyped as strings.
To change one, edit its file; the tests beside it pin the public values.

| Contract | Defined in | Change it when |
|----------|------------|----------------|
| Message structure (tool calls, tool results, reasoning) | `core/message.py` | A message needs a new field |
| Names LiteLLM expects (request parameters, message and tool-call keys) | `llm/litellm_spec.py` | LiteLLM's format changes |
| Streaming event names and payloads | `api/stream_events.py` | The REST stream gains or changes an event |
| Server environment variables and defaults | `api/settings.py` | A setting is added or renamed |

- **`llm/litellm_format.py`** is the only code that converts to and from LiteLLM's format.
- **LiteLLM replies are read through LiteLLM's own types** (`ModelResponse`,
  `ModelResponseStream`, `Message`), so a mistyped attribute fails the type check.
- **A tool's limits and timeouts** are named constants at the top of its file.

## Tests

- **`tests/unit/<package>/test_<module>.py`** mostly mirrors `src/miniagent/<package>/<module>.py`.
- **`tests/integration/`** runs components together: workflows, the API, YAML
  round-trips, a real MCP subprocess, sessions on SQLite.
- **MCP tests skip** when the `mcp` extra is not installed.
