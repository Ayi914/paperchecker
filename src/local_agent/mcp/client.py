from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPStdioClient:
    """Manage one MCP stdio session and expose discovery/call primitives."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self._parameters = StdioServerParameters(command=command, args=args, env=env)
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPStdioClient":
        read, write = await self._stack.enter_async_context(stdio_client(self._parameters))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    async def list_tools(self):
        if self.session is None:
            raise RuntimeError("MCP client is not connected")
        return (await self.session.list_tools()).tools

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        if self.session is None:
            raise RuntimeError("MCP client is not connected")
        return await self.session.call_tool(name, arguments=arguments)
