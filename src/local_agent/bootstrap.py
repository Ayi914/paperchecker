"""首次使用的配置和模型向导；不会启动 GPU 服务。"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values, set_key

MODEL_REPO = "Qwen/Qwen3-14B-AWQ"
MODEL_DIR = Path("models/Qwen3-14B-AWQ")


def init_config() -> None:
    path = Path(".env")
    if not path.exists():
        with path.open("x", encoding="utf-8") as handle:
            handle.write(Path(".env.example").read_text(encoding="utf-8"))
        path.chmod(0o600)
        print("已创建本地配置 .env。", file=sys.stderr)
    for directory in ("logs", "output", "data/knowledge", "data/indexes", "data/checkpoints"):
        Path(directory).mkdir(parents=True, exist_ok=True)


def validate_model_dir(path: Path) -> None:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"{path} 不是模型目录：需要 config.json 和模型权重。")
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError("无法读取模型 config.json，请检查模型是否完整。") from exc
    if not isinstance(config, dict):
        raise ValueError("模型 config.json 必须是 JSON 对象。")  # noqa: TRY004 -- 文件内容校验错误
    shards = list(path.glob("*.safetensors"))
    if not shards or any(shard.stat().st_size == 0 for shard in shards):
        raise ValueError("模型权重缺失或为空，请重新运行模型下载向导。")
    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("权重索引不完整，请重新下载模型。") from exc
        if not names or any(not (path / name).is_file() for name in names):
            raise ValueError("模型权重分片未下载完整，请重新运行模型下载向导。")
    if not (path / "tokenizer.json").is_file():
        raise ValueError("缺少 tokenizer.json，请重新下载完整模型。")
    quantization = config.get("quantization_config", {})
    if quantization.get("quant_method") != "awq":
        raise ValueError("初版默认使用 AWQ 模型，请选择 Qwen3-14B-AWQ 的完整目录。")


def configured_model() -> str:
    return (
        os.environ.get("LOCAL_AGENT_MODEL") or dotenv_values(".env").get("LOCAL_AGENT_MODEL") or ""
    )


def model_is_configured() -> bool:
    model = configured_model()
    if not model:
        return False
    path = Path(model).expanduser()
    if path.exists() or path.is_absolute() or model.startswith(("models/", "./", "../", "~/")):
        try:
            validate_model_dir(path)
        except ValueError:
            return False
    # 高级用户可以手工配置 Hugging Face 模型 ID 或远程服务的 served-model-name。
    return True


def save_model(path: Path) -> None:
    validate_model_dir(path)
    set_key(".env", "LOCAL_AGENT_MODEL", str(path.expanduser().resolve()), quote_mode="always")
    Path(".env").chmod(0o600)
    print("模型目录已配置。现在运行 bash scripts/start.sh 即可提问。", file=sys.stderr)


def download_model() -> Path:
    from huggingface_hub import snapshot_download

    print("正在下载 Qwen3-14B-AWQ（权重约 9.4 GB），中断后可重复运行继续下载。", file=sys.stderr)
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
    validate_model_dir(MODEL_DIR)
    return MODEL_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地论文助手首次配置：选择模型，不启动模型服务。")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-dir", type=Path, help="使用已有的 Qwen3-14B-AWQ 模型目录。")
    group.add_argument("--download-model", action="store_true", help="下载默认模型（约 9.4 GB）。")
    group.add_argument("--skip-model", action="store_true", help="只创建配置，稍后选择模型。")
    group.add_argument("--check", action="store_true", help="只检查模型是否已配置，不写入文件。")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if model_is_configured() else 1
    try:
        init_config()
        if args.skip_model:
            print("环境配置完成；首次启动时会继续引导选择模型。", file=sys.stderr)
            return 0
        if args.model_dir:
            save_model(args.model_dir.expanduser())
        elif args.download_model:
            save_model(download_model())
        elif model_is_configured():
            print("现有模型配置已保留。", file=sys.stderr)
        elif not sys.stdin.isatty():
            print(
                "尚未选择模型；请运行 bash scripts/setup.sh，或指定 --model-dir /模型目录。",
                file=sys.stderr,
            )
        else:
            print(
                "\n请选择模型来源：\n  1. 已有模型文件\n  2. 下载 Qwen3-14B-AWQ（约 9.4 GB）\n  3. 稍后配置",
                file=sys.stderr,
            )
            choice = input("选择 [1/2/3]：").strip()
            if choice == "1":
                save_model(Path(input("模型目录：").strip()).expanduser())
            elif choice == "2":
                save_model(download_model())
            elif choice == "3":
                print("稍后运行 bash scripts/setup.sh 配置模型。", file=sys.stderr)
            else:
                raise ValueError("请输入 1、2 或 3，然后重新运行 bash scripts/setup.sh。")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n配置已中断，可重新运行 bash scripts/setup.sh。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 -- 向导边界给出可读错误
        print(
            f"配置未完成：{exc}\n请检查网络或目录，然后重试 bash scripts/setup.sh。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
