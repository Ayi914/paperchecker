from collections.abc import Iterable

from langchain_core.tools import BaseTool


class ToolRegistry:
    """Keep tool names unique and expose tools to LangGraph."""

    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def extend(self, tools: Iterable[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())
