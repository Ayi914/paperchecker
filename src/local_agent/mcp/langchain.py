import asyncio
import json
import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from local_agent.mcp.client import MCPStdioClient


def build_stdio_tools(command: str, args: list[str], env: dict[str, str]) -> list[BaseTool]:
    """Discover MCP stdio tools and expose them as sync LangChain tools."""
    return asyncio.run(_build_stdio_tools(command, args, env))


async def _build_stdio_tools(command: str, args: list[str], env: dict[str, str]) -> list[BaseTool]:
    _log(f"discover start command={command!r} args={args!r} env_keys={sorted(env)}")
    async with MCPStdioClient(command, args, env=env) as client:
        mcp_tools = await client.list_tools()
    _log(f"discover ok tools={[tool.name for tool in mcp_tools]}")

    return [_make_tool(command, args, env, tool) for tool in mcp_tools]


def _make_tool(command: str, args: list[str], env: dict[str, str], mcp_tool: Any) -> BaseTool:
    name = f"mcp_{mcp_tool.name}"
    schema = getattr(mcp_tool, "inputSchema", None) or {}
    args_schema = _args_schema(name, schema)

    def _call(**kwargs: Any) -> str:
        return asyncio.run(_call_mcp_tool(command, args, env, mcp_tool.name, kwargs))

    return StructuredTool.from_function(
        func=_call,
        name=name,
        description=mcp_tool.description or f"MCP tool {mcp_tool.name}",
        args_schema=args_schema,
    )


async def _call_mcp_tool(
    command: str, args: list[str], env: dict[str, str], name: str, arguments: dict[str, Any]
) -> str:
    started = time.perf_counter()
    _log(f"call start name={name!r} arguments={json.dumps(arguments, ensure_ascii=False)}")
    async with MCPStdioClient(command, args, env=env) as client:
        result = await client.call_tool(name, arguments)
    formatted = _format_call_result(result)
    elapsed = time.perf_counter() - started
    _log(f"call ok name={name!r} elapsed_s={elapsed:.3f} result_preview={_preview(formatted)!r}")
    return formatted


def _args_schema(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        annotation = _annotation_for(field_schema)
        default = ... if field_name in required else None
        fields[field_name] = (
            annotation,
            Field(default, description=field_schema.get("description")),
        )

    return create_model(
        f"{tool_name.title().replace('_', '')}Args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _annotation_for(field_schema: dict[str, Any]) -> Any:
    schema_type = field_schema.get("type")
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return str


def _format_call_result(result: Any) -> str:
    content = getattr(result, "content", None)
    if not content:
        return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)

    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
            continue
        if text == "":
            continue
        parts.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    if parts:
        return "\n".join(parts)
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


def _log(message: str) -> None:
    print(f"[mcp] {message}", flush=True)


def _preview(text: str, limit: int = 500) -> str:
    normalized = text.replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."
