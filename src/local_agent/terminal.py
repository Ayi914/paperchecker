"""轻量终端展示：过程写 stderr，最终答案写 stdout。"""

import re
import sys

import typer


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str) or isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def answer_text(content) -> str:
    text = content_text(content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 未闭合的思考区段不能成为最终答案。
    text = text.split("<think>", 1)[0]
    return text.strip()


class TerminalUI:
    input_prompt = ">>> "

    def read_input(self) -> str:
        """真实终端使用 readline 处理编辑和重绘；管道保持普通输入。"""
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                import readline  # noqa: F401 -- 导入即启用 input 的行编辑
            except ImportError:
                pass
            typer.echo()
            return input(self.input_prompt)
        return typer.prompt("", prompt_suffix=self.input_prompt, err=True)

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.pending = ""
        self.answer = ""
        self.thinking = False
        self.tool_round = False
        self.open_line = False
        self.reasoning_started = False
        self.round = 0

    def status(self, text: str) -> None:
        if not self.quiet:
            self.end_line()
            typer.secho(text, fg=typer.colors.CYAN, err=True)

    def end_line(self) -> None:
        if self.open_line:
            typer.echo(err=True)
            self.open_line = False

    def stream_text(self, text: str) -> None:
        if text and not self.quiet:
            # 清除终端控制字符，避免外部文本触发终端转义序列。
            text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
            sys.stderr.write(text)
            sys.stderr.flush()
            self.open_line = True

    def feed(self, text: str, final: bool = False) -> None:
        self.pending += text
        while self.pending:
            marker = "</think>" if self.thinking else "<think>"
            position = self.pending.find(marker)
            if position >= 0:
                self.emit(self.pending[:position])
                self.pending = self.pending[position + len(marker) :]
                self.thinking = not self.thinking
                if self.thinking:
                    self.status("模型思考（服务返回）")
                else:
                    self.end_line()
                continue
            keep = 0
            if not final:
                for size in range(1, min(len(marker), len(self.pending) + 1)):
                    if self.pending.endswith(marker[:size]):
                        keep = size
            ready = self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            self.emit(ready)
            break

    def emit(self, text: str) -> None:
        if self.thinking or self.tool_round:
            self.stream_text(text)
        else:
            self.answer += text

    def event(self, kind: str, data: dict) -> None:
        if kind == "model_start":
            self.round += 1
            self.pending = self.answer = ""
            self.thinking = self.tool_round = False
            self.reasoning_started = False
            self.status(f"\n执行过程 · 第 {self.round} 次模型调用")
        elif kind == "model_chunk":
            chunk = data["chunk"]
            reasoning = (getattr(chunk, "additional_kwargs", {}) or {}).get("reasoning_content")
            if reasoning:
                if not self.reasoning_started:
                    self.status("模型思考（服务返回）")
                    self.reasoning_started = True
                self.stream_text(str(reasoning))
            if getattr(chunk, "tool_call_chunks", None) and not self.tool_round:
                self.tool_round = True
                self.stream_text(self.answer)
                self.answer = ""
            self.feed(content_text(chunk.content))
        elif kind == "model_end":
            self.feed("", final=True)
            response = data["response"]
            if getattr(response, "tool_calls", None) and not self.tool_round:
                self.stream_text(self.answer)
                self.answer = ""
            self.end_line()
        elif kind == "tool_start":
            self.status(f"→ 调用工具：{data['name']}")
        elif kind == "tool_end":
            self.status(f"{'✓ 完成' if data['status'] == 'ok' else '✗ 失败'}：{data['name']}")
        elif kind == "loop_guard":
            self.status(
                "重复工具调用已阻止，正在尝试恢复。"
                if not data["final"]
                else "重复工具调用已阻止，本轮停止。"
            )
        elif kind == "context_warning":
            self.status("上下文超过输入预算，已保留当前问题和必要工具记录。")

    def show_answer(self, answer: str) -> None:
        self.end_line()
        if not self.quiet:
            typer.secho("\n最终答案", fg=typer.colors.GREEN, err=True)
        typer.echo(answer)
