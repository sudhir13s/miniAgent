"""Integration tests for the FastAPI layer (Phase 4.7).

Uses FastAPI's TestClient with mocked LLM calls — no live API or real agents.
Tests the full HTTP stack: routing, serialisation, error handling.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from miniagent.api.app import create_app
from miniagent.api.dependencies import get_workflow_registry, reset_profile_cache
from miniagent.core.exceptions import AgentExecutionError, ConfigurationError
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_caches() -> Any:
    reset_profile_cache()
    yield
    reset_profile_cache()


def _make_mock_profiles(names: list[str] | None = None) -> list[Any]:
    names = names or ["agent-x"]
    profiles = []
    for name in names:
        p = MagicMock()
        p.name = name
        p.description = f"Desc of {name}"
        p.model.id = "openai/gpt-4o"
        p.system_prompt = "You are helpful."
        p.tools = []
        p.max_iterations = 5
        profiles.append(p)
    return profiles


def _state_with_ai(content: str = "AI reply") -> State:
    s = State()
    s = s.add_message(Message(role=MessageRole.AI, content=content))
    return s.increment_step()


def _register_wf(name: str, final: State | None = None) -> MagicMock:
    reg = get_workflow_registry()
    if name in reg:
        reg.unregister(name)
    wf = MagicMock()
    wf.run = AsyncMock(return_value=final or State())
    reg.register(name, wf)
    return wf


def _unregister(name: str) -> None:
    reg = get_workflow_registry()
    if name in reg:
        reg.unregister(name)


# ---------------------------------------------------------------------------
# Agent routes
# ---------------------------------------------------------------------------


class TestAgentExecuteIntegration:
    def test_run_returns_200_with_ai_response(self, client: TestClient) -> None:
        mock_agent = MagicMock()
        mock_agent.profile.system_prompt = "Be helpful."
        mock_agent.run = AsyncMock(return_value=_state_with_ai("Hello!"))

        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            return_value=mock_agent,
        ):
            resp = client.post(
                "/agents/my-agent/run",
                json={"message": "Hi there"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_name"] == "my-agent"
        assert data["response"] == "Hello!"
        assert data["step_count"] == 1

    def test_run_404_when_agent_not_found(self, client: TestClient) -> None:
        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            side_effect=ConfigurationError("not found"),
        ):
            resp = client.post(
                "/agents/ghost/run",
                json={"message": "hi"},
            )

        assert resp.status_code == 404

    def test_run_500_on_agent_execution_error(self, client: TestClient) -> None:
        mock_agent = MagicMock()
        mock_agent.profile.system_prompt = "sys"
        mock_agent.run = AsyncMock(side_effect=AgentExecutionError("LLM timed out"))

        with patch(
            "miniagent.api.routes.agents.get_agent_by_name",
            return_value=mock_agent,
        ):
            resp = client.post("/agents/bad-agent/run", json={"message": "go"})

        assert resp.status_code == 500
        assert "LLM timed out" in resp.json()["detail"]

    def test_run_passes_variables_to_state(self, client: TestClient) -> None:
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
                json={"message": "go", "variables": {"ctx": "value"}},
            )

        assert captured[0].variables["ctx"] == "value"


class TestListAgentsIntegration:
    def test_returns_registered_agents(self, client: TestClient) -> None:
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            return_value=_make_mock_profiles(["alpha", "beta"]),
        ):
            resp = client.get("/agents")

        assert resp.status_code == 200
        names = [a["name"] for a in resp.json()]
        assert "alpha" in names
        assert "beta" in names

    def test_returns_503_on_config_error(self, client: TestClient) -> None:
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            side_effect=ConfigurationError("no config"),
        ):
            resp = client.get("/agents")

        assert resp.status_code == 503

    def test_get_single_agent_returns_detail(self, client: TestClient) -> None:
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            return_value=_make_mock_profiles(["my-agent"]),
        ):
            resp = client.get("/agents/my-agent")

        assert resp.status_code == 200
        assert resp.json()["name"] == "my-agent"

    def test_get_unknown_agent_returns_404(self, client: TestClient) -> None:
        with patch(
            "miniagent.api.routes.agents.get_agent_profiles",
            return_value=_make_mock_profiles(["other"]),
        ):
            resp = client.get("/agents/missing")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Workflow routes
# ---------------------------------------------------------------------------


class TestWorkflowRunIntegration:
    WF = "wf-int-run"

    def setup_method(self) -> None:
        _unregister(self.WF)

    def teardown_method(self) -> None:
        _unregister(self.WF)

    def test_run_returns_200_with_state_summary(self, client: TestClient) -> None:
        final = State()
        final = final.add_message(Message(role=MessageRole.AI, content="done"))
        final = final.increment_step()
        _register_wf(self.WF, final)

        resp = client.post(f"/workflows/{self.WF}/run", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["workflow_name"] == self.WF
        assert data["step_count"] == 1
        assert data["message_count"] == 1

    def test_run_404_for_unknown_workflow(self, client: TestClient) -> None:
        resp = client.post("/workflows/nonexistent-wf/run", json={})
        assert resp.status_code == 404

    def test_run_seeds_message(self, client: TestClient) -> None:
        captured: list[State] = []

        async def _run(state: State) -> State:
            captured.append(state)
            return state

        wf = MagicMock()
        wf.run = _run
        reg = get_workflow_registry()
        reg.register(self.WF, wf)

        client.post(f"/workflows/{self.WF}/run", json={"message": "kick off"})

        assert any(m.content == "kick off" for m in captured[0].messages)

    def test_run_500_on_workflow_error(self, client: TestClient) -> None:
        from miniagent.core.exceptions import WorkflowExecutionError

        wf = MagicMock()
        wf.run = AsyncMock(side_effect=WorkflowExecutionError("exploded"))
        reg = get_workflow_registry()
        reg.register(self.WF, wf)

        resp = client.post(f"/workflows/{self.WF}/run", json={})

        assert resp.status_code == 500
        assert "exploded" in resp.json()["detail"]


class TestWorkflowStreamIntegration:
    WF = "wf-int-stream"

    def setup_method(self) -> None:
        _unregister(self.WF)

    def teardown_method(self) -> None:
        _unregister(self.WF)

    def test_stream_returns_sse_content_type(self, client: TestClient) -> None:
        _register_wf(self.WF)
        resp = client.post(f"/workflows/{self.WF}/run/stream", json={})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_started_and_completed(self, client: TestClient) -> None:
        _register_wf(self.WF)
        resp = client.post(f"/workflows/{self.WF}/run/stream", json={})
        assert "event: started" in resp.text
        assert "event: completed" in resp.text

    def test_stream_emits_error_event_on_failure(self, client: TestClient) -> None:
        wf = MagicMock()
        wf.run = AsyncMock(side_effect=RuntimeError("stream boom"))
        reg = get_workflow_registry()
        reg.register(self.WF, wf)

        resp = client.post(f"/workflows/{self.WF}/run/stream", json={})
        assert "event: error" in resp.text
        assert "stream boom" in resp.text

    def test_stream_404_for_unknown_workflow(self, client: TestClient) -> None:
        resp = client.post("/workflows/unknown-stream-wf/run/stream", json={})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Meta routes
# ---------------------------------------------------------------------------


class TestMetaRoutesIntegration:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_redirects_to_docs(self, client: TestClient) -> None:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert resp.headers["location"] == "/docs"
