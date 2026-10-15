import asyncio
import json
import os
import shlex
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
import typer
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool

from local_agent.config import Settings, load_agent_yaml
from local_agent.graph import build_graph
from local_agent.llm import build_chat_model
from local_agent.llm.runtime import managed_model
from local_agent.mcp import build_stdio_tools
from local_agent.rag.store import index_documents, make_retrieval_tool, retrieve_knowledge
from local_agent.terminal import TerminalUI, answer_text
from local_agent.tools import BUILTIN_TOOLS, ToolRegistry

cli = typer.Typer(add_completion=False, help="本地论文助手：终端提问、检索与工具调用。")


def build_sqlite_checkpointer(path: Path):
    """Create the optional LangGraph SQLite checkpointer."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite checkpointing requires langgraph-checkpoint-sqlite. "
            "Run `uv sync --extra dev --extra rag` after updating dependencies."
        ) from exc

    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    setup = getattr(checkpointer, "setup", None)
    if setup is not None:
        setup()
    return checkpointer


@cli.command()
def chat(
    prompt: Annotated[
        str | None,
        typer.Argument(help="提问内容；省略时在终端进入交互模式，批处理使用配置中的问题。"),
    ] = None,
    thread_id: Annotated[
        str | None,
        typer.Option(help="会话 ID，用于恢复已保存的对话。"),
    ] = None,
    no_checkpoint: Annotated[
        bool,
        typer.Option("--no-checkpoint", help="关闭本次运行的 SQLite 会话记录。"),
    ] = False,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="进入连续提问模式。")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="只输出最终答案。")] = False,
    save: Annotated[bool, typer.Option("--save/--no-save", help="保存精简 Markdown 答案。")] = True,
    start_model: Annotated[
        bool, typer.Option("--start-model/--no-start-model", help="自动启动本地 vLLM 模型服务。")
    ] = True,
) -> None:
    """向本地模型提问，或进入连续对话。"""
    project_config = load_agent_yaml()
    agent_config = project_config.get("agent", {})
    run_config = project_config.get("run", {})
    interactive = interactive or (prompt is None and sys.stdin.isatty())
    user_prompt = prompt
    if user_prompt is None and not interactive:
        user_prompt = sys.stdin.read().strip() or run_config.get("user_prompt")
    if not interactive and not str(user_prompt or "").strip():
        raise typer.BadParameter("请提供问题，或设置 config/agent.yaml 的 run.user_prompt。")
    ui = TerminalUI(quiet=quiet)
    ui.status("本地论文助手 · 正在准备模型和工具…")

    load_dotenv(override=False)
    settings = Settings()
    with managed_model(settings, ui.status, auto_start=start_model):
        registry = ToolRegistry(BUILTIN_TOOLS)
        rag_config = project_config.get("rag", {})
        if settings.index_dir.exists() and any(settings.index_dir.iterdir()):
            registry.register(
                make_retrieval_tool(
                    settings.embedding_model,
                    settings.index_dir,
                    top_k=int(rag_config.get("top_k", 4)),
                    knowledge_dir=settings.knowledge_dir,
                    chunk_size=int(rag_config.get("chunk_size", 600)),
                    chunk_overlap=int(rag_config.get("chunk_overlap", 100)),
                    candidate_k=int(rag_config.get("candidate_k", 20)),
                    min_vector_score=float(rag_config.get("min_vector_score", 0.3)),
                    tokenizer_mode=str(rag_config.get("jieba_mode", "accurate")),
                    stopwords_path=Path(str(rag_config.get("stopwords_path", "data/stopwords/zh.txt"))),
                    user_dict_path=Path(str(rag_config.get("user_dict_path", "data/stopwords/domain_terms.txt"))),
                )
            )

        mcp_config = project_config.get("mcp", {})
        if mcp_config.get("brave_search_enabled", False):
            registry.extend(_build_brave_search_tools(mcp_config))
        if mcp_config.get("brave_answers_enabled", False):
            registry.register(_build_brave_answers_tool(mcp_config))

        tools = registry.all()
        ui.status(f"已加载 {len(tools)} 个工具。")
        checkpoint_enabled = bool(run_config.get("checkpoint_enabled", getattr(settings, "checkpoint_enabled", True)))
        checkpointer = (
            None
            if no_checkpoint or not checkpoint_enabled
            else build_sqlite_checkpointer(settings.checkpoint_db)
        )
        graph = build_graph(
            build_chat_model(settings),
            tools,
            system_prompt=agent_config.get(
                "system_prompt", "你是本地知识助手。必要时调用工具，并如实说明来源与限制。"
            ),
            checkpointer=checkpointer,
            event_handler=ui.event,
            model_input_token_budget=int(agent_config.get("model_input_token_budget", 4096)),
            recent_turns_to_keep=int(agent_config.get("recent_turns_to_keep", 4)),
            tool_output_max_chars=int(agent_config.get("tool_output_max_chars", 300)),
            repeated_tool_call_limit=int(agent_config.get("repeated_tool_call_limit", 3)),
        )
        active_thread_id = thread_id or (
            f"cli-{uuid4().hex[:12]}" if interactive
            else run_config.get("thread_id") or settings.thread_id
        )
        graph_config = {
            "recursion_limit": int(agent_config.get("recursion_limit", 12)),
            "configurable": {"thread_id": active_thread_id},
        }
        history: list[BaseMessage] = []
        if interactive:
            ui.status("输入问题并回车；/help 查看命令，/new 开始新会话，/exit 退出。")
        try:
            while True:
                if user_prompt is None:
                    try:
                        user_prompt = ui.read_input()
                    except (EOFError, typer.Abort):
                        break
                user_prompt = str(user_prompt).strip()
                if interactive and user_prompt in {"/exit", "/quit"}:
                    break
                if interactive and user_prompt == "/help":
                    ui.status("/new：清空当前会话；/exit 或 Ctrl+D：退出；支持直接粘贴单行问题。")
                    user_prompt = None
                    continue
                if interactive and user_prompt == "/new":
                    history = []
                    active_thread_id = f"cli-{uuid4().hex[:12]}"
                    graph_config["configurable"]["thread_id"] = active_thread_id
                    ui.status("已开始新会话。")
                    user_prompt = None
                    continue
                if not user_prompt:
                    user_prompt = None
                    if interactive:
                        continue
                    break
                messages = [HumanMessage(content=user_prompt)]
                if checkpointer is None:
                    messages = history + messages
                result = graph.invoke({"messages": messages}, graph_config)
                last = result["messages"][-1]
                if isinstance(last, AIMessage) and not last.tool_calls:
                    answer = answer_text(last.content) or "模型未返回可展示的最终答案。"
                else:
                    answer = "本轮因重复工具调用而停止，未生成最终答案。请调整问题后重试。"
                ui.show_answer(answer)
                if save:
                    write_output(
                        user_prompt, answer, Path(run_config.get("output_dir", "output")),
                        thread_id=active_thread_id if checkpointer is not None else None,
                    )
                if not interactive:
                    break
                history = result["messages"] if checkpointer is None else []
                user_prompt = None
        finally:
            ui.end_line()
            if checkpointer is not None:
                checkpointer.conn.close()


@cli.command("setup")
def setup(
    model_dir: Annotated[Path | None, typer.Option(help="已有 AWQ 模型目录。")] = None,
    download_model: Annotated[bool, typer.Option(help="下载 Qwen3-14B-AWQ，约 9.4 GB。")] = False,
) -> None:
    """首次配置或更换模型，不启动 GPU 服务。"""
    from local_agent.bootstrap import main as bootstrap_main

    args = []
    if model_dir is not None:
        args.extend(["--model-dir", str(model_dir)])
    if download_model:
        args.append("--download-model")
    raise typer.Exit(bootstrap_main(args))


@cli.command("doctor")
def doctor() -> None:
    """检查依赖、GPU 和模型配置，不启动模型。"""
    from local_agent.doctor import main as doctor_main

    raise typer.Exit(doctor_main([]))


@cli.command("rag-index")
def rag_index() -> None:
    """为本地论文目录建立检索索引。"""
    settings = Settings()
    project_config = load_agent_yaml()
    rag_config = project_config.get("rag", {})
    count = index_documents(
        settings.knowledge_dir,
        settings.embedding_model,
        settings.index_dir,
        chunk_size=int(rag_config.get("chunk_size", 600)),
        chunk_overlap=int(rag_config.get("chunk_overlap", 100)),
    )
    typer.echo(f"已索引 {count} 个文本片段，目录：{settings.index_dir}")


@cli.command("rag-search")
def rag_search(query: str) -> None:
    """直接检索已建立索引的本地论文。"""
    settings = Settings()
    project_config = load_agent_yaml()
    top_k = int(project_config.get("rag", {}).get("top_k", 4))
    rag_config = project_config.get("rag", {})
    hits = retrieve_knowledge(
        query,
        settings.embedding_model,
        settings.index_dir,
        top_k,
        knowledge_dir=settings.knowledge_dir,
        chunk_size=int(rag_config.get("chunk_size", 600)),
        chunk_overlap=int(rag_config.get("chunk_overlap", 100)),
        candidate_k=int(rag_config.get("candidate_k", 20)),
        min_vector_score=float(rag_config.get("min_vector_score", 0.3)),
        tokenizer_mode=str(rag_config.get("jieba_mode", "accurate")),
        stopwords_path=Path(str(rag_config.get("stopwords_path", "data/stopwords/zh.txt"))),
        user_dict_path=Path(str(rag_config.get("user_dict_path", "data/stopwords/domain_terms.txt"))),
    )
    for index, hit in enumerate(hits, start=1):
        typer.echo(
            f"[{index}] {hit['source']} chunk_id={hit['chunk_id']} page={hit.get('page')} score={hit['score']:.4f} "
            f"vector={hit['vector_score']:.4f} bm25={hit['bm25_score']:.4f}"
        )
        typer.echo(hit["text"])
        typer.echo("")


@cli.command("bench-concurrency")
def bench_concurrency(
    concurrency: Annotated[int, typer.Option(help="Number of concurrent requests.")] = 4,
    output_tokens: Annotated[int, typer.Option(help="max_tokens for each request.")] = 256,
) -> None:
    """对已启动的模型服务进行并发测试。"""
    asyncio.run(_bench_concurrency(concurrency, output_tokens))


def _build_brave_answers_tool(mcp_config: dict):
    api_key_env = str(mcp_config.get("brave_search_api_key_env", "BRAVE_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Brave Answers API is enabled but {api_key_env} is not set")

    model = str(mcp_config.get("brave_answers_model", "brave"))
    endpoint = str(
        mcp_config.get(
            "brave_answers_endpoint",
            "https://api.search.brave.com/res/v1/chat/completions",
        )
    )

    def brave_answer_search(query: str) -> str:
        """Use Brave Answers API for web-grounded answers when current internet information is needed."""
        payload = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": query}],
        }
        typer.echo(
            "[brave_answers] request "
            f"endpoint={endpoint!r} body={json.dumps(payload, ensure_ascii=False)}"
        )
        with httpx.Client(timeout=60, trust_env=True) as client:
            response = client.post(
                endpoint,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-Subscription-Token": api_key,
                },
                json=payload,
            )
        typer.echo(
            "[brave_answers] response "
            f"status={response.status_code} preview={response.text[:500]!r}"
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if choices:
            content = ((choices[0].get("message") or {}).get("content") or "").strip()
            if content:
                return content
        return json.dumps(data, ensure_ascii=False)

    return StructuredTool.from_function(
        func=brave_answer_search,
        name="brave_answer_search",
        description=(
            "Search the live web with Brave Answers API and return a grounded answer. "
            "Use this for current facts, admissions dates, deadlines, news, and questions "
            "that require internet lookup. The input must be the user's full search question."
        ),
    )


def _build_brave_search_tools(mcp_config: dict) -> list:
    api_key_env = str(mcp_config.get("brave_search_api_key_env", "BRAVE_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Brave Search MCP is enabled but {api_key_env} is not set")

    command = str(mcp_config.get("brave_search_command", "npx"))
    args = shlex.split(str(mcp_config.get("brave_search_args", "-y @modelcontextprotocol/server-brave-search")))
    env = {api_key_env: api_key}
    proxy_env_present = False
    for env_name in (
        "PATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "NODE_USE_ENV_PROXY",
    ):
        if value := os.environ.get(env_name):
            env[env_name] = value
            if "proxy" in env_name.lower():
                proxy_env_present = True
    if proxy_env_present:
        env.setdefault("NODE_USE_ENV_PROXY", "1")
    tools = build_stdio_tools(command, args, env=env)
    return tools


def write_output(
    prompt: str,
    answer: str,
    output_dir: Path,
    thread_id: str | None = None,
    model_inputs: list[list[BaseMessage]] | None = None,
    final_messages: list[BaseMessage] | None = None,
    model_input_stats: list[dict[str, object]] | None = None,
    model_context_window: int | None = None,
    max_output_tokens: int | None = None,
) -> Path:
    now = datetime.now(UTC).astimezone()
    path = output_dir / f"{now:%m}" / f"{now:%d}" / f"{now:%H%M%S_%f}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [f"# Prompt\n\n{prompt}"]
    if thread_id is not None:
        sections.append(f"# Thread\n\n`{thread_id}`")
    # 历史参数保留兼容，但不再把模型输入、图状态和调用统计写入用户答案。
    sections.append(f"# Output\n\n{answer_text(answer)}")
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def _format_context_usage(
    stats: list[dict[str, object]],
    *,
    model_context_window: int | None = None,
    max_output_tokens: int | None = None,
) -> str:
    if not stats:
        return "No model invocations were recorded."
    budget = int(stats[0]["token_budget"])
    actual_values = [
        int(item["actual_input_tokens"])
        for item in stats
        if item.get("actual_input_tokens") is not None
    ]
    estimated_peak = max(int(item["estimated_tokens"]) for item in stats)
    lines = []
    if model_context_window is not None:
        lines.append(f"- Model context window: **{model_context_window} tokens**")
    lines.append(f"- Agent input context limit: **{budget} tokens**")
    if max_output_tokens is not None:
        lines.append(f"- Configured maximum output: **{max_output_tokens} tokens**")
    if actual_values:
        actual_peak = max(actual_values)
        lines.append(
            f"- Peak actual input context used: **{actual_peak} tokens "
            f"({actual_peak / budget:.1%})**"
        )
    else:
        lines.append("- Peak actual input context used: **unavailable**")
    lines.extend([
        (f"- Peak estimated input context: **{estimated_peak} tokens "
         f"({estimated_peak / budget:.1%})**"),
        "",
        "| Invocation | Actual input | Estimated input | Limit | Actual usage | Messages removed | Over limit | Counter |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |",
    ])
    for index, item in enumerate(stats, start=1):
        tokens = int(item["estimated_tokens"])
        item_budget = int(item["token_budget"])
        actual = item.get("actual_input_tokens")
        actual_text = str(actual) if actual is not None else "unavailable"
        usage_text = f"{int(actual) / item_budget:.1%}" if actual is not None else "unavailable"
        lines.append(
            f"| {index} | {actual_text} | {tokens} | {item_budget} | {usage_text} | "
            f"{item['messages_removed']} | {'yes' if item['over_budget'] else 'no'} | "
            f"{item['counter']} |"
        )
    return "\n".join(lines)


def _format_model_inputs(model_inputs: list[list[BaseMessage]]) -> str:
    if not model_inputs:
        return "No model invocations were recorded."
    blocks = [
        ("This is the exact message list passed to `bound_model.invoke(...)`. "
         "It includes the system prompt plus any messages restored from the LangGraph checkpoint.")
    ]
    for index, messages in enumerate(model_inputs, start=1):
        blocks.append(f"## LLM Invocation {index} Input\n\n" + _format_messages(messages))
    return "\n\n".join(blocks)


def _format_final_messages(messages: list[BaseMessage]) -> str:
    return (
        "This is the graph state after the run finished. It is not a separate LLM input. "
        "When there is only one model invocation and no new tool call, it will look almost the same "
        "as the last LLM input, except it also includes the final AI response.\n\n"
        + _format_messages(messages)
    )


def _format_messages(messages: list[BaseMessage]) -> str:
    if not messages:
        return "No messages."
    parts = []
    for index, message in enumerate(messages, start=1):
        role = getattr(message, "type", type(message).__name__)
        content = getattr(message, "content", "")
        parts.append(f"### Message {index}: {role}\n\n{_format_message_content(content)}")
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            parts.append(
                "Parsed structured `tool_calls` on this AI message:\n\n"
                "```json\n" + json.dumps(tool_calls, ensure_ascii=False, indent=2, default=str) + "\n```"
            )
    return "\n\n".join(parts)


def _format_message_content(content) -> str:
    if isinstance(content, str):
        return content or "(empty)"
    return "```json\n" + json.dumps(content, ensure_ascii=False, indent=2, default=str) + "\n```"


async def _bench_concurrency(concurrency: int, output_tokens: int) -> None:
    settings = Settings()
    prompts = [
        f"你是第 {i + 1} 路请求。请用中文简要说明本地 Agent 四并发测试的目的，并输出 5 条编号。"
        for i in range(concurrency)
    ]
    async with httpx.AsyncClient(base_url=settings.base_url, timeout=120, trust_env=False) as client:
        results = await asyncio.gather(
            *[_stream_chat(client, settings, prompt, output_tokens) for prompt in prompts],
            return_exceptions=True,
        )

    ok = [result for result in results if isinstance(result, dict)]
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        for error in errors:
            typer.echo(f"ERROR {type(error).__name__}: {error}")
    if not ok:
        raise typer.Exit(code=1)

    ttfts = [float(result["ttft_s"]) for result in ok]
    totals = [float(result["total_s"]) for result in ok]
    tokens = [int(result["tokens"]) for result in ok]
    typer.echo(f"requests_ok={len(ok)} requests_error={len(errors)}")
    typer.echo(f"ttft_p50_s={statistics.median(ttfts):.3f} ttft_max_s={max(ttfts):.3f}")
    typer.echo(f"e2e_p50_s={statistics.median(totals):.3f} e2e_max_s={max(totals):.3f}")
    typer.echo(f"generated_tokens_total={sum(tokens)} approx_tokens_per_s={sum(tokens) / max(totals):.2f}")


async def _stream_chat(
    client: httpx.AsyncClient, settings: Settings, prompt: str, output_tokens: int
) -> dict[str, float | int]:
    started = time.perf_counter()
    first_token_at: float | None = None
    token_events = 0
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.temperature,
        "max_tokens": output_tokens,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    async with client.stream("POST", "/chat/completions", json=payload, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            token_events += 1
            if first_token_at is None:
                first_token_at = time.perf_counter()
    ended = time.perf_counter()
    return {
        "ttft_s": (first_token_at or ended) - started,
        "total_s": ended - started,
        "tokens": token_events,
    }


def main() -> None:
    # 保持完整子命令，同时兼容 local-agent "问题" 的快捷入口。
    commands = {"chat", "rag-index", "rag-search", "bench-concurrency", "doctor", "setup"}
    args = sys.argv[1:]
    if not args or (args[0] not in commands and args[0] not in {"--help", "-h"}):
        args.insert(0, "chat")
    try:
        cli(args=args)
    except KeyboardInterrupt:
        typer.echo("\n已中断。", err=True)
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001 -- CLI 边界统一展示运行错误
        typer.secho(f"运行失败：{exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
