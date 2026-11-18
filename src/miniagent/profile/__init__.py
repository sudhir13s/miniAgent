"""Agent configuration: profiles, model settings, templates, MCP servers.

Quick import::

    from miniagent.profile import AgentProfile, MCPServerConfig, ModelConfig

    profiles = AgentProfile.from_yaml("agents.yaml")
    servers = MCPServerConfig.from_yaml("agents.yaml")
"""

from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.mcp_server_config import MCPServerConfig
from miniagent.profile.model_config import ModelConfig
from miniagent.profile.template import ProfileTemplate

__all__ = ["AgentProfile", "MCPServerConfig", "ModelConfig", "ProfileTemplate"]
