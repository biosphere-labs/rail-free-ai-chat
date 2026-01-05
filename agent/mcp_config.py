import os
from pathlib import Path


def get_mcp_config() -> dict:
    """Get MCP server configuration.

    Returns a dict compatible with MultiServerMCPClient.
    Servers are configured based on available environment variables.
    """
    config = {}

    # Filesystem MCP server - always available
    # Use HOST_HOME if running in Docker, otherwise use actual home
    host_home = os.environ.get("HOST_HOME")
    if host_home:
        # Running in Docker - use mounted host home
        home_dir = host_home
    else:
        home_dir = str(Path.home())
    cwd = os.getcwd()

    config["filesystem"] = {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            cwd,
            home_dir,
        ],
        "transport": "stdio",
    }

    # Brave Search MCP server - requires API key
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if brave_api_key:
        config["brave_search"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": brave_api_key},
            "transport": "stdio",
        }

    return config


def get_enabled_servers() -> list[str]:
    """Get list of enabled MCP server names."""
    config = get_mcp_config()
    return list(config.keys())
