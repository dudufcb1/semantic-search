#!/bin/bash
# Start script for FastMCP server

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the FastMCP server using uv
exec uv run fastmcp run

