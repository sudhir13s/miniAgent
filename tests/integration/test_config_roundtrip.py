"""Integration tests for loading agents.example.yaml end to end.

Loads agents.example.yaml, verifies all profiles and MCP servers parse
correctly, and builds a runnable agent from every profile. No LLM calls are
made and no MCP server is started.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from miniagent.agent import Agent, build_agent
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.mcp_server_config import MCPServerConfig
from miniagent.profile.templates import WEB_RESEARCHER
from tests.unit.tool.fakes import FakeToolset

# Path to the example YAML at the project root
_EXAMPLE_YAML = Path(__file__).parent.parent.parent / "agents.example.yaml"

_ENVIRONMENT_REFERENCE = re.compile(r"\$\{\w+\}")
# A token-looking word left after removing ${VAR} references is a literal secret.
_TOKEN_LIKE_WORD = re.compile(r"[A-Za-z0-9_\-]{20,}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load_profiles() -> dict[str, AgentProfile]:
    return {profile.name: profile for profile in AgentProfile.from_yaml(_EXAMPLE_YAML)}


def _load_servers() -> dict[str, MCPServerConfig]:
    servers = MCPServerConfig.from_yaml(_EXAMPLE_YAML)
    return {server.name: server for server in servers}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    def test_example_yaml_exists(self) -> None:
        assert _EXAMPLE_YAML.exists(), f"agents.example.yaml not found at {_EXAMPLE_YAML}"

    def test_loads_all_profiles(self) -> None:
        profiles = AgentProfile.from_yaml(_EXAMPLE_YAML)
        names = [profile.name for profile in profiles]
        assert len(names) == len(set(names)), "Duplicate agent names in example YAML"
        assert len(profiles) >= 3

    def test_researcher_profile_fields(self) -> None:
        researcher = _load_profiles()["researcher"]
        assert researcher.model.id == "openai/gpt-4o-mini"
        assert researcher.model.temperature == 0.3
        assert "web_fetch" in researcher.tools

    def test_writer_profile_fields(self) -> None:
        writer = _load_profiles()["writer"]
        assert writer.model.id == "openai/gpt-4o-mini"
        assert writer.tools == []

    def test_agents_mix_model_vendors(self) -> None:
        vendors = {p.model.id.split("/")[0] for p in _load_profiles().values()}
        assert {"openai", "anthropic", "gemini"} <= vendors

    def test_template_profile_resolves(self) -> None:
        gemini = _load_profiles()["gemini-researcher"]
        assert gemini.template is WEB_RESEARCHER
        assert gemini.model.id == "gemini/gemini-2.0-flash"
        assert gemini.system_prompt == WEB_RESEARCHER.system_prompt

    def test_extended_thinking_profile_passes_thinking_to_litellm(self) -> None:
        reasoner = _load_profiles()["claude-reasoner"]
        assert reasoner.model.id.startswith("anthropic/")
        assert reasoner.model.extra_params["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }

    def test_build_agent_for_every_profile(self) -> None:
        """Every example profile's tools and toolsets resolve to a runnable agent.

        Each declared MCP server is stood in for by an unconnected fake
        toolset of the same name, so the test needs neither the mcp extra
        nor a running server.
        """
        toolsets = [FakeToolset(name, []) for name in _load_servers()]
        for profile in _load_profiles().values():
            agent = build_agent(profile, toolsets=toolsets)
            assert isinstance(agent, Agent)
            assert sorted(agent.tools) == sorted(profile.tools)
            assert [toolset.name for toolset in agent.toolsets] == profile.toolsets

    def test_mcp_servers_cover_both_transports(self) -> None:
        servers = _load_servers()
        assert {server.transport for server in servers.values()} == {"stdio", "http"}
        assert servers["filesystem"].command == "npx"
        assert servers["github"].url == "https://api.githubcopilot.com/mcp/"

    def test_mcp_servers_build_unconnected_toolsets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("mcp", reason="MCP support needs the 'mcp' extra")
        from miniagent.tool.mcp import MCPToolset

        monkeypatch.setenv("GITHUB_TOKEN", "placeholder-for-test")
        toolsets = [MCPToolset.from_config(server) for server in _load_servers().values()]
        assert [toolset.name for toolset in toolsets] == ["filesystem", "github"]

    def test_every_profile_toolset_is_a_declared_server(self) -> None:
        declared = set(_load_servers())
        used = {name for profile in _load_profiles().values() for name in profile.toolsets}
        assert used, "Example YAML should show at least one agent using MCP toolsets"
        assert used <= declared

    def test_mcp_secrets_are_environment_references(self) -> None:
        """The example never holds a literal secret, only ${VAR} placeholders."""
        github = _load_servers()["github"]
        assert github.headers == {"Authorization": "Bearer ${GITHUB_TOKEN}"}
        for server in _load_servers().values():
            for value in [*server.headers.values(), *server.env.values()]:
                literal_part = _ENVIRONMENT_REFERENCE.sub("", value)
                assert not _TOKEN_LIKE_WORD.search(literal_part), f"{server.name}: literal secret"

    def test_from_dict_round_trip(self) -> None:
        """AgentProfile.from_dict() produces valid AgentProfile objects."""
        raw = {
            "agents": [
                {
                    "name": "test-openai",
                    "model": {"id": "openai/gpt-4o"},
                    "system_prompt": "You are a test assistant.",
                    "max_iterations": 3,
                },
                {
                    "name": "test-anthropic",
                    "model": {
                        "id": "anthropic/claude-sonnet-4-6",
                        "temperature": 0.3,
                    },
                    "system_prompt": "You are precise.",
                },
            ]
        }
        profiles = AgentProfile.from_dict(raw)
        assert [p.name for p in profiles] == ["test-openai", "test-anthropic"]
        assert profiles[0].model.id == "openai/gpt-4o"
        assert profiles[1].model.id == "anthropic/claude-sonnet-4-6"
