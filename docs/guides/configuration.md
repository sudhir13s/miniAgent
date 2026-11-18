# Configuration

Agents are defined in YAML (or in code) and validated by Pydantic when loaded.

## Agents in YAML

```yaml
agents:
  - name: assistant              # Required. Letters, digits, '-', '_'. Unique per file.
    description: "General-purpose assistant."
    system_prompt: "You are a helpful AI."
    template: general-assistant  # Optional persona; fields you set win.
    model:
      id: "openai/gpt-4o"        # Required. "<vendor>/<model>".
      temperature: 0.7
      max_tokens: 4096
      extra_params: {stop: ["DONE"]}   # Passed straight to LiteLLM.
    tools: [web_fetch, calculator]     # Built-in or custom tool names.
    toolsets: [github]                 # MCP servers, see tools.md.
    max_iterations: 10                 # 1–100.
    max_context_tokens: 32000          # Optional; see memory.md.
```

```python
from miniagent import AgentProfile, build_agent

profiles = AgentProfile.from_yaml("agents.yaml")                   # every agent
assistant = AgentProfile.from_yaml("agents.yaml", name="assistant")  # just one
agent = build_agent(assistant)
```

- **Loading lives on the class**, like Hugging Face's `from_pretrained`:
  `AgentProfile.from_yaml` / `from_dict`, and `MCPServerConfig.from_yaml` / `from_dict`.
- **`from_dict`** takes the same shape as the file (`{"agents": [...]}`); for a
  single agent in code, use `AgentProfile(name=..., model=ModelConfig(id=...), ...)`.
- **Unknown keys are rejected**, in the agent and in `model`; so are repeated names.
- **Every tool name must resolve** to a built-in or a tool passed to `build_agent`.

## Model settings

| Field | Default | Notes |
|-------|---------|-------|
| `id` | required | `"openai/gpt-4o"`, `"anthropic/claude-sonnet-4-6"`, `"gemini/gemini-2.0-flash"`, `"groq/…"`, `"bedrock/…"` |
| `temperature` | `0.7` | 0.0–2.0 |
| `max_tokens` | `4096` | Reply length limit |
| `top_p` / `top_k` | `1.0` / unset | `top_k` is provider-dependent |
| `extra_params` | `{}` | Any LiteLLM option: stop sequences, `response_format`, `num_retries`, prompt caching |

**API keys** come from each vendor's environment variable (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …), read by LiteLLM.

**Claude extended thinking** is an `extra_params` setting:

```yaml
model:
  id: "anthropic/claude-sonnet-4-6"
  temperature: 1.0               # Required by Anthropic when thinking is on.
  max_tokens: 16000
  extra_params:
    thinking: {type: enabled, budget_tokens: 10000}
```

The reasoning lands in `message.reasoning`, and thinking blocks are re-sent on
tool turns automatically.

**Another gateway:** subclass `BaseLLMProvider` and install it with
`miniagent.llm.set_default_provider(...)`.

## Profile templates

A template is a reusable persona: prompt, default tools, iteration budget. It
never sets the model.

| Template (`miniagent.profile.templates`) | YAML name | Tools |
|------------------------------------------|-----------|-------|
| `GENERAL_ASSISTANT` | `general-assistant` | — |
| `TOOL_ASSISTANT` | `tool-assistant` | calculator, timestamp, weather |
| `WEB_RESEARCHER` | `web-researcher` | web_fetch, file_reader |
| `TECHNICAL_WRITER` | `technical-writer` | — |
| `PYTHON_DEVELOPER` | `python-developer` | file_reader |
| `CODE_REVIEWER` | `code-reviewer` | file_reader |
| `MATH_SOLVER` | `math-solver` | calculator |
| `TASK_PLANNER` | `task-planner` | — |
| `DOCUMENT_SUMMARIZER` | `document-summarizer` | file_reader, word_count |

```python
from miniagent import AgentProfile, ModelConfig, ProfileTemplate
from miniagent.profile.templates import WEB_RESEARCHER

scout = AgentProfile(name="scout", template=WEB_RESEARCHER, model=ModelConfig(id="openai/gpt-4o-mini"))

SUPPORT = ProfileTemplate(name="support-agent", system_prompt="You are friendly support.", tools=("web_fetch",))
profiles = AgentProfile.from_yaml("agents.yaml", templates=[SUPPORT])  # usable by name in YAML
```

- **Precedence:** your fields, then the template, then defaults.
- **An explicit `tools` list replaces** the template's tools.
