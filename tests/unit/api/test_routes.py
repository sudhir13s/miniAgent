"""Comprehensive tests for all Phase-2 API routes.

Covers:
  - GET /agents, GET /agents/{name}, POST /agents/{name}/run
  - GET /workflows, POST /workflows/{name}/run, POST /workflows/{name}/run/stream
  - GET /health, GET / (redirect)
  - app factory (create_app)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from miniagent.api.app import create_app
from miniagent.api.dependencies import get_workflow_registry, reset_profile_cache
from miniagent.core.exceptions import AgentExecutionError, ConfigurationError
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_profile_cache()
    yield
    reset_profile_cache()


def _state_with_ai(content: str = "AI response") -> State:
    s = State()
    s = s.add_message(Message(role=MessageRole.AI, content=content))
    s = s.increment_step()
    return s


def _mock_profiles(names: list[str] | None = None):
    names = names or ["agent-a", "agent-b"]
    profiles = []
    for name in names:
        p = MagicMock()
        p.name = name
        p.description = f"Description of {name}"
        p.model.id = "openai/gpt-4o"
        p.system_prompt = "You are helpful."
        p.tools = []
        p.max_iterations = 10
        profiles.append(p)
    return profiles


# ── Meta routes ───────────────────────────────────────────────────────────────


class TestHealthRoute:
    def test_returns_200_with_status_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_returns_version(self, client):
        resp = client.get("/health")
        assert "version" in resp.json()
        assert resp.json()["version"] == "0.1.0"


class TestRootRedirect:
    def test_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert resp.headers["location"] == "/docs"


# ── Agent routes ──────────────────────────────────────────────────────────────


class TestListAgents:
    def test_returns_200_with_agents(self, client):
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            return_value=_mock_profiles(),
        ):
            resp = client.get("/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["name"] == "agent-a"
        assert data[0]["model"] == "openai/gpt-4o"

    def test_returns_503_on_config_error(self, client):
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            side_effect=ConfigurationError("Config missing"),
        ):
            resp = client.get("/agents")
        assert resp.status_code == 503

    def test_returns_empty_list_when_no_agents(self, client):
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            return_value=[],
        ):
            resp = client.get("/agents")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetAgent:
    def test_returns_200_for_known_agent(self, client):
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            return_value=_mock_profiles(["my-agent"]),
        ):
            resp = client.get("/agents/my-agent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "my-agent"
        assert data["system_prompt"] == "You are helpful."
        assert data["max_iterations"] == 10

    def test_returns_404_for_unknown_agent(self, client):
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            return_value=_mock_profiles(["other"]),
        ):
            resp = client.get("/agents/nonexistent")
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]

    def test_returns_503_on_config_error(self, client):
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            side_effect=ConfigurationError("no file"),
        ):
            resp = client.get("/agents/x")
        assert resp.status_code == 503


class TestExecuteAgent:
    def test_returns_200_with_ai_response(self, client):
        mock_agent = MagicMock()
        mock_agent.profile.system_prompt = "Be helpful."
        mock_agent.run = AsyncMock(return_value=_state_with_ai("Hello!"))

        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            return_value=mock_agent,
        ):
            resp = client.post(
                "/agents/my-agent/run",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_name"] == "my-agent"
        assert data["response"] == "Hello!"
        assert data["step_count"] == 1

    def test_passes_variables_to_state(self, client):
        captured: list[State] = []

        async def _run(state: State) -> State:
            captured.append(state)
            return _state_with_ai("ok")

        mock_agent = MagicMock()
        mock_agent.profile.system_prompt = "sys"
        mock_agent.run = _run

        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            return_value=mock_agent,
        ):
            client.post(
                "/agents/bot/run",
                json={"message": "go", "variables": {"ctx": "test"}},
            )
        assert captured[0].variables["ctx"] == "test"

    def test_returns_404_when_agent_not_found(self, client):
        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            side_effect=ConfigurationError("not found"),
        ):
            resp = client.post("/agents/bad/run", json={"message": "hi"})
        assert resp.status_code == 404

    def test_returns_500_on_execution_error(self, client):
        mock_agent = MagicMock()
        mock_agent.profile.system_prompt = "sys"
        mock_agent.run = AsyncMock(side_effect=AgentExecutionError("LLM failed"))

        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            return_value=mock_agent,
        ):
            resp = client.post("/agents/bot/run", json={"message": "hi"})
        assert resp.status_code == 500
        assert "LLM failed" in resp.json()["detail"]

    def test_placeholder_response_when_no_ai_messages(self, client):
        mock_agent = MagicMock()
        mock_agent.profile.system_prompt = "sys"
        mock_agent.run = AsyncMock(return_value=State())

        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            return_value=mock_agent,
        ):
            resp = client.post("/agents/bot/run", json={"message": "hi"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "No response generated."


# ── Workflow routes ────────────────────────────────────────────────────────────


def _register_mock_workflow(name: str, final_state: State | None = None) -> MagicMock:
    """Register a mock workflow in the global registry and return it."""
    registry = get_workflow_registry()
    # Unregister if already registered (test cleanup may lag)
    if name in registry:
        registry.unregister(name)

    wf = MagicMock()
    wf.run = AsyncMock(return_value=final_state or State())
    registry.register(name, wf)
    return wf


def _unregister(name: str) -> None:
    registry = get_workflow_registry()
    if name in registry:
        registry.unregister(name)


class TestListWorkflows:
    def setup_method(self):
        _unregister("wf-list-a")
        _unregister("wf-list-b")

    def teardown_method(self):
        _unregister("wf-list-a")
        _unregister("wf-list-b")

    def test_returns_registered_workflow_names(self, client):
        _register_mock_workflow("wf-list-a")
        _register_mock_workflow("wf-list-b")
        resp = client.get("/workflows")
        assert resp.status_code == 200
        names = [w["name"] for w in resp.json()]
        assert "wf-list-a" in names
        assert "wf-list-b" in names

    def test_empty_when_none_registered(self, client):
        resp = client.get("/workflows")
        assert resp.status_code == 200
        # May have other workflows from other tests — just check it's a list
        assert isinstance(resp.json(), list)


class TestRunWorkflow:
    WF_NAME = "wf-run-test"

    def setup_method(self):
        _unregister(self.WF_NAME)

    def teardown_method(self):
        _unregister(self.WF_NAME)

    def test_returns_200_with_final_state(self, client):
        final = State()
        final = final.add_message(Message(role=MessageRole.AI, content="done"))
        final = final.increment_step()
        _register_mock_workflow(self.WF_NAME, final)

        resp = client.post(f"/workflows/{self.WF_NAME}/run", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflow_name"] == self.WF_NAME
        assert data["step_count"] == 1
        assert data["message_count"] == 1

    def test_seeds_message_into_state(self, client):
        captured: list[State] = []

        async def _run(state: State) -> State:
            captured.append(state)
            return state

        wf = MagicMock()
        wf.run = _run
        registry = get_workflow_registry()
        registry.register(self.WF_NAME, wf)

        client.post(f"/workflows/{self.WF_NAME}/run", json={"message": "start"})
        assert any(m.content == "start" for m in captured[0].messages)

    def test_returns_404_for_unknown_workflow(self, client):
        resp = client.post("/workflows/does-not-exist/run", json={})
        assert resp.status_code == 404

    def test_returns_500_on_workflow_error(self, client):
        from miniagent.core.exceptions import WorkflowExecutionError

        wf = MagicMock()
        wf.run = AsyncMock(side_effect=WorkflowExecutionError("boom"))
        registry = get_workflow_registry()
        registry.register(self.WF_NAME, wf)

        resp = client.post(f"/workflows/{self.WF_NAME}/run", json={})
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]

    def test_max_steps_passed_to_state(self, client):
        captured: list[State] = []

        async def _run(state: State) -> State:
            captured.append(state)
            return state

        wf = MagicMock()
        wf.run = _run
        registry = get_workflow_registry()
        registry.register(self.WF_NAME, wf)

        client.post(f"/workflows/{self.WF_NAME}/run", json={"max_steps": 42})
        assert captured[0].max_steps == 42


class TestStreamWorkflow:
    WF_NAME = "wf-stream-test"

    def setup_method(self):
        _unregister(self.WF_NAME)

    def teardown_method(self):
        _unregister(self.WF_NAME)

    def test_returns_sse_content_type(self, client):
        _register_mock_workflow(self.WF_NAME)
        resp = client.post(f"/workflows/{self.WF_NAME}/run/stream", json={})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_emits_started_and_completed_events(self, client):
        _register_mock_workflow(self.WF_NAME)
        resp = client.post(f"/workflows/{self.WF_NAME}/run/stream", json={})
        assert resp.status_code == 200
        body = resp.text
        assert "event: started" in body
        assert "event: completed" in body

    def test_completed_event_has_the_same_fields_as_the_run_response(self, client):
        _register_mock_workflow(self.WF_NAME)
        run = client.post(f"/workflows/{self.WF_NAME}/run", json={}).json()
        stream = client.post(f"/workflows/{self.WF_NAME}/run/stream", json={}).text

        started, completed = (
            json.loads(line.removeprefix("data: "))
            for line in stream.splitlines()
            if line.startswith("data: ")
        )
        assert started == {"workflow_name": self.WF_NAME}
        assert completed.keys() == run.keys()
        assert completed["workflow_name"] == self.WF_NAME

    def test_emits_error_event_on_failure(self, client):
        wf = MagicMock()
        wf.run = AsyncMock(side_effect=RuntimeError("stream failure"))
        registry = get_workflow_registry()
        registry.register(self.WF_NAME, wf)

        resp = client.post(f"/workflows/{self.WF_NAME}/run/stream", json={})
        assert resp.status_code == 200
        assert "event: error" in resp.text
        assert "stream failure" in resp.text

    def test_returns_404_for_unknown_workflow(self, client):
        resp = client.post("/workflows/unknown-stream/run/stream", json={})
        assert resp.status_code == 404


# ── App factory ───────────────────────────────────────────────────────────────


class TestAppFactory:
    def test_creates_fastapi_app(self):
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_title(self):
        app = create_app()
        assert app.title == "miniAgent API"

    def test_all_routers_registered(self):
        app = create_app()
        paths = {r.path for r in app.routes}
        assert "/health" in paths
        assert "/agents" in paths
        assert "/workflows" in paths


# ── CORS ──────────────────────────────────────────────────────────────────────

_ALLOWED_ORIGIN = "http://localhost:3000"
_OTHER_ORIGIN = "http://evil.example"


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


class TestCors:
    def test_should_send_no_cors_headers_when_origins_unset(self, monkeypatch):
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        client = TestClient(create_app())

        response = client.get("/health", headers={"Origin": _ALLOWED_ORIGIN})

        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_should_allow_only_listed_origins(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", f" {_ALLOWED_ORIGIN} , http://other.local ")
        client = TestClient(create_app())

        allowed = _preflight(client, _ALLOWED_ORIGIN)
        rejected = _preflight(client, _OTHER_ORIGIN)

        assert allowed.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN
        assert "access-control-allow-origin" not in rejected.headers

    def test_should_never_allow_credentials(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", _ALLOWED_ORIGIN)
        client = TestClient(create_app())

        response = _preflight(client, _ALLOWED_ORIGIN)

        assert "access-control-allow-credentials" not in response.headers

    def test_should_allow_any_origin_when_wildcard_set(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "*")
        client = TestClient(create_app())

        response = client.get("/health", headers={"Origin": _OTHER_ORIGIN})

        assert response.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in response.headers

    def test_should_send_no_cors_headers_when_origins_blank(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", " , ")
        client = TestClient(create_app())

        response = client.get("/health", headers={"Origin": _ALLOWED_ORIGIN})

        assert "access-control-allow-origin" not in response.headers
