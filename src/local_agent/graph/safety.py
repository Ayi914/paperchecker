import hashlib
import json
import math
from collections.abc import Callable
from html import escape
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    recent_tool_fingerprints: list[str]
    loop_detected: bool
    recovery_attempted: bool
    last_error: str | None


def tool_call_fingerprint(name: str, args: object) -> str:
    canonical = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{name}\n{canonical}".encode()).hexdigest()


def wrap_tool_output(source: str, raw_output: object, max_chars: int, status: str = "ok") -> str:
    if max_chars <= 0:
        raise ValueError("tool output max_chars must be positive")
    if isinstance(raw_output, str):
        raw = raw_output
    else:
        try:
            raw = json.dumps(raw_output, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            raw = str(raw_output)
    original_chars = len(raw)
    truncated = original_chars > max_chars
    if truncated:
        raw = raw[:max_chars] + f"\n[tool output truncated: original_chars={original_chars}, kept_chars={max_chars}]"
    attributes = {
        "source": source,
        "trust_level": "untrusted",
        "status": status,
        "truncated": str(truncated).lower(),
        "original_chars": str(original_chars),
    }
    rendered = " ".join(f'{key}="{escape(value, quote=True)}"' for key, value in attributes.items())
    return f"<tool_output {rendered}>\n{escape(raw)}\n</tool_output>"


def estimate_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for message in messages:
        content = message.content if isinstance(message.content, str) else json.dumps(message.content, default=str)
        total += max(1, math.ceil(len(content) / 2)) + 8
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            total += math.ceil(len(json.dumps(tool_calls, default=str)) / 2)
    return total


def prepare_model_input(
    system_message: BaseMessage,
    history: list[BaseMessage],
    *,
    token_budget: int,
    recent_turns: int,
    tool_output_max_chars: int,
    token_counter: Callable[[list[BaseMessage]], int] | None = None,
) -> tuple[list[BaseMessage], dict[str, object]]:
    if token_budget <= 0 or recent_turns <= 0:
        raise ValueError("token budget and recent turns must be positive")
    start = _recent_turn_start(history, recent_turns)
    selected = [_truncate_tool_message(item, tool_output_max_chars) for item in history[start:]]
    count = token_counter or estimate_tokens
    model_input = [system_message, *selected]
    removed = start
    while len(selected) > 1 and count(model_input) > token_budget:
        next_human = next((i for i, item in enumerate(selected[1:], 1) if item.type == "human"), None)
        if next_human is None:
            break
        removed += next_human
        selected = selected[next_human:]
        model_input = [system_message, *selected]
    tokens = count(model_input)
    return model_input, {
        "estimated_tokens": tokens,
        "token_budget": token_budget,
        "messages_removed": removed,
        "over_budget": tokens > token_budget,
        "counter": "model" if token_counter else "fallback",
    }


def _recent_turn_start(history: list[BaseMessage], recent_turns: int) -> int:
    seen = 0
    for index in range(len(history) - 1, -1, -1):
        if history[index].type == "human":
            seen += 1
            if seen == recent_turns:
                return index
    return 0


def _truncate_tool_message(message: BaseMessage, max_chars: int) -> BaseMessage:
    if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
        return message
    if message.content.startswith("<tool_output ") or len(message.content) <= max_chars:
        return message
    marker = f"\n[tool output truncated for model input: original_chars={len(message.content)}, kept_chars={max_chars}]"
    return message.model_copy(update={"content": message.content[:max_chars] + marker})
