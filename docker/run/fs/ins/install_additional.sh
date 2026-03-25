#!/bin/bash
set -e

# install playwright - moved to install A0
# bash /ins/install_playwright.sh "$@"

# searxng - moved to base image
# bash /ins/install_searxng.sh "$@"

# ─── SanMarcSoft additions ───

# mcp2cli — token-efficient MCP server access for all agents
/opt/venv/bin/pip install mcp2cli && ln -sf /opt/venv/bin/mcp2cli /usr/local/bin/mcp2cli