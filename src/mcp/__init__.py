"""
MCP Package for Progressive Schema Disclosure

This package implements the MCP (Model Context Protocol) pattern for
reducing context window usage when working with large database schemas.

Components:
- MCPServer: Schema registry with discovery methods
- sandbox: Structured validation for safe query execution

Usage:
    from src.mcp import MCPServer, StructuredSandbox

    # Initialize server from queries file
    server = MCPServer.from_queries_file("data/weaviate-gorilla.json")

    # Create sandbox for validation
    sandbox = StructuredSandbox(server)
"""

from src.mcp.server import MCPServer
from src.mcp.sandbox import StructuredSandbox, ValidationResult

__all__ = [
    "MCPServer",
    "StructuredSandbox",
    "ValidationResult",
]
