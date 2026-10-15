"""环境诊断：不启动 vLLM，不下载模型，不输出密钥。"""

import argparse
import importlib.metadata
import platform
import sys

from dotenv import load_dotenv

from local_agent.bootstrap import model_is_configured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查论文助手的环境，不启动模型。")
    parser.add_argument(
        "--environment-only", action="store_true", help="只检查依赖，不要求 GPU 或模型。"
    )
    args = parser.parse_args(argv)
    load_dotenv(override=False)
    failures = []
    print(f"系统：{platform.system()} {platform.machine()}；Python {platform.python_version()}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        failures.append("初版安装脚本支持 Linux x86_64。")
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        failures.append("Python 需要 3.11 或 3.12；请重新运行安装脚本。")
    expected = {"torch": "2.8.0", "transformers": "4.56.2", "vllm": "0.10.2"}
    for package, version in expected.items():
        try:
            actual = importlib.metadata.version(package)
            print(f"{package}：{actual}")
            if actual.split("+")[0] != version:
                failures.append(f"{package} 版本不匹配，请运行 bash scripts/install.sh。")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"缺少 {package}，请运行 bash scripts/install.sh。")
    for module in ("torch", "langchain_chroma", "langchain_huggingface", "pysqlite3"):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 -- 动态库与可选依赖失败类型不同
            failures.append(f"无法加载 {module}：{type(exc).__name__}；请检查安装日志或重装环境。")
    if not args.environment_only:
        try:
            import torch

            if torch.cuda.is_available():
                memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU：{torch.cuda.get_device_name(0)}，显存 {memory:.1f} GiB")
                if memory < 23:
                    failures.append(
                        "当前默认模型在 A30 24 GB 上验证；此 GPU 显存较小，不能保证加载成功。"
                    )
            else:
                failures.append("当前进程没有可用 CUDA GPU，请检查驱动或切换到 GPU 计算节点。")
        except Exception as exc:  # noqa: BLE001 -- 诊断动态库/CUDA，避免栈追踪
            failures.append(f"GPU 检查失败：{type(exc).__name__}。")
        if not model_is_configured():
            failures.append("模型尚未配置或文件不完整；请运行 bash scripts/setup.sh。")
    for error in failures:
        print(f"✗ {error}", file=sys.stderr)
    if failures:
        return 1
    print("✓ 检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
