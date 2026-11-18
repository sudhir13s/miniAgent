# REST API

A FastAPI server over your `agents.yaml`. Interactive docs at `/docs`.

```bash
python -m miniagent                      # 0.0.0.0:8000
```

| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `AGENTS_CONFIG_PATH` | `agents.yaml` | Agent and MCP server config |
| `SESSION_DB_PATH` | unset | SQLite file for sessions; unset keeps them in memory |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |
| `LOG_LEVEL` / `RELOAD` | `info` / `false` | Uvicorn settings |
| `CORS_ORIGINS` | unset | Browser origins allowed to call the API |

All of these are defined in one file, `miniagent/api/settings.py`.

On startup the server connects every MCP server in the config, and closes them
on shutdown.

**Calling from a browser app:** cross-origin requests are blocked until you list
the origins, e.g. `CORS_ORIGINS=http://localhost:3000,https://app.example.com`.

- **`*`** allows any origin; fine locally, logged as a warning, avoid in production.
- **No credentials:** browsers never send cookies cross-origin to the API.
- **Not needed** for server-to-server calls, `curl`, or the `/docs` page.

## Endpoints

| Method & path | Does |
|---------------|------|
| `GET /agents` | List agents (name, description, model) |
| `GET /agents/{name}` | One agent's prompt, tools and iteration limit |
| `POST /agents/{name}/run` | Run an agent on a message |
| `POST /agents/{name}/run/stream` | Same, streaming the reply as it is written (SSE) |
| `GET /workflows` | List registered workflows |
| `POST /workflows/{name}/run` | Run a workflow, wait for the result |
| `POST /workflows/{name}/run/stream` | Run a workflow, stream progress (SSE) |
| `GET /health` | `{"status": "ok", "version": "…"}` |

## Run an agent

```json
POST /agents/researcher/run
{"message": "Explain quantum entanglement.", "variables": {}, "session_id": "user-42:chat-7"}
```

```json
200 OK
{"agent_name": "researcher", "response": "Quantum entanglement is…",
 "step_count": 3, "status": "completed", "session_id": "user-42:chat-7"}
```

- **`session_id`** is optional; with it, the conversation continues (see [memory.md](memory.md)).
- **`status`** is `completed` once the run finishes; a failed run returns an error instead.
- **`response`** is the last answer, or `"No response generated."` when the run
  ended without one (it hit `max_iterations` while still calling tools).
- **Errors:** `404` unknown agent or invalid agent configuration · `422` invalid
  input · `500` run failed · `503` session store failed.
- **`GET /agents` and `GET /agents/{name}`** return `503` when the configuration
  file cannot be loaded.

## Stream an agent

`POST /agents/{name}/run/stream` takes the same body and returns Server-Sent Events:

```text
event: text
data: {"delta": "Quantum "}

event: tool_call
data: {"id": "c1", "name": "web_fetch", "arguments": {"url": "https://…"}}

event: tool_result
data: {"id": "c1", "name": "web_fetch", "content": "…", "is_error": false}

event: completed
data: {"agent_name": "researcher", "response": "Quantum entanglement is…", "step_count": 3, …}
```

- **`completed`** carries the same fields as `/run`; the session is saved just before it.
- **Failures mid-run** arrive as `event: error` with a `detail`; a `404` or a
  session `503` is returned before streaming starts.
- **In a browser,** read it with `fetch()` and a stream reader (`EventSource` only does GET).

## Run a workflow

Workflows are registered in code first (see [workflows.md](workflows.md)).

```json
POST /workflows/pipeline/run
{"message": "Summarise the latest AI research.", "variables": {}, "max_steps": 200}
```

The response holds `workflow_name`, `status`, `step_count`, `variables` and
`message_count`.

`/run/stream` sends Server-Sent Events. It reports progress, not tokens.

| Event | Data |
|-------|------|
| `started` | `workflow_name` |
| `completed` | The same fields as the `/run` response |
| `error` | `workflow_name`, `detail` |
