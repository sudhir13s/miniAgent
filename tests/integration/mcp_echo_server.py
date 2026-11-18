"""A tiny MCP server that the stdio integration test launches as a subprocess."""

from mcp.server import MCPServer

server = MCPServer("echo")


@server.tool()
def shout(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


if __name__ == "__main__":
    server.run("stdio")
