"""CLI 所拥有的本地 vLLM 生命周期。"""

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from dotenv import dotenv_values

from local_agent.config import Settings


def runtime_environment() -> dict[str, str]:
    """沿用 .env 的 vLLM 参数；实际环境变量优先，不执行 shell 代码。"""
    values = {key: value for key, value in dotenv_values(".env").items() if value is not None}
    return {**values, **os.environ}


def model_ready(client: httpx.Client, settings: Settings) -> bool:
    try:
        response = client.get(
            settings.base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )
    except httpx.TransportError:
        return False
    if response.status_code in {502, 503, 504}:
        return False
    if response.is_error:
        raise RuntimeError(
            f"模型服务检查失败（HTTP {response.status_code}），请检查地址和 API key。"
        )
    try:
        ids = [item["id"] for item in response.json()["data"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("模型服务未返回有效的 /v1/models 响应。") from exc
    if settings.model not in ids:
        raise RuntimeError(f"服务未提供配置中的模型 {settings.model!r}，可用模型：{ids}。")
    return True


def require_local_gpu(env: dict[str, str]) -> None:
    if env.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        raise RuntimeError("当前进程没有可见 GPU，请在分配了 GPU 的计算节点启动 CLI。")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("当前节点没有可用 NVIDIA GPU，请切换到 GPU 计算节点启动 CLI。") from exc
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("无法访问 NVIDIA GPU，请检查 GPU 分配或驱动。")


def vllm_command(settings: Settings, env: dict[str, str]) -> list[str]:
    address = urlparse(settings.base_url)
    if address.scheme != "http" or address.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("远程模型服务无法连接；CLI 只能自动启动本机 HTTP 模型服务。")
    if address.path.rstrip("/") != "/v1":
        raise RuntimeError("自动启动要求 LOCAL_AGENT_BASE_URL 使用 http://127.0.0.1:端口/v1。")
    model_path = Path(settings.model).expanduser()
    if (
        model_path.is_absolute() or settings.model.startswith(("models/", "./", "../", "~/"))
    ) and not model_path.is_dir():
        raise RuntimeError("模型目录不存在；请运行 bash scripts/setup.sh 选择或下载模型。")
    host = address.hostname
    port = address.port or 80
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError as exc:
        raise RuntimeError(
            f"模型端口 {port} 已被占用或无法绑定，请检查服务或修改 LOCAL_AGENT_BASE_URL。"
        ) from exc
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        settings.model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        env.get("VLLM_DTYPE", "half"),
        "--gpu-memory-utilization",
        env.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-model-len",
        env.get("VLLM_MAX_MODEL_LEN", "16384"),
        "--max-num-seqs",
        env.get("VLLM_MAX_NUM_SEQS", "4"),
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    quantization = env.get("VLLM_QUANTIZATION", "awq")
    if quantization and quantization.lower() != "none":
        command.extend(["--quantization", quantization])
    if settings.api_key:
        command.extend(["--api-key", settings.api_key])
    return command


def stop_process(process: subprocess.Popen) -> None:
    """结束自己创建的进程组，包括 vLLM 的引擎子进程。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@contextmanager
def managed_model(
    settings: Settings,
    status: Callable[[str], None],
    *,
    auto_start: bool = True,
) -> Iterator[None]:
    """先检查服务，必要时启动；启动失败、中断与正常退出均清理。"""
    process = None
    with httpx.Client(timeout=2, trust_env=False) as client:
        if model_ready(client, settings):
            status("模型服务已就绪，复用现有服务。")
            yield
            return
        if not auto_start:
            raise RuntimeError("模型服务未就绪；去掉 --no-start-model 可让 CLI 自动启动 vLLM。")
        env = runtime_environment()
        command = vllm_command(settings, env)
        require_local_gpu(env)
        try:
            startup_timeout = float(env.get("LOCAL_AGENT_MODEL_STARTUP_TIMEOUT", "1200"))
        except ValueError as exc:
            raise RuntimeError("LOCAL_AGENT_MODEL_STARTUP_TIMEOUT 必须是正数秒数。") from exc
        if not 0 < startup_timeout < float("inf"):
            raise RuntimeError("LOCAL_AGENT_MODEL_STARTUP_TIMEOUT 必须是正数秒数。")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        log_path = Path("logs") / f"cli-vllm-{uuid4().hex[:12]}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        status(f"正在启动本地 vLLM，首次加载模型需要等待。日志：{log_path}")
        try:
            with log_path.open("x", encoding="utf-8") as log:
                log_path.chmod(0o600)
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                started = time.monotonic()
                last_report = started
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"vLLM 启动失败（退出码 {process.returncode}），请查看 {log_path}。"
                        )
                    if model_ready(client, settings):
                        status("模型已就绪，开始对话。")
                        break
                    elapsed = time.monotonic() - started
                    if elapsed >= startup_timeout:
                        raise RuntimeError(
                            f"等待模型就绪超时（{startup_timeout:g} 秒），请查看 {log_path}。"
                        )
                    if time.monotonic() - last_report >= 10:
                        status(f"正在加载模型，已等待 {int(elapsed)} 秒…")
                        last_report = time.monotonic()
                    time.sleep(1)
                yield
        finally:
            if process is not None:
                status("正在关闭本次启动的模型服务…")
                stop_process(process)
