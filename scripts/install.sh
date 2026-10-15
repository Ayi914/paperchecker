#!/usr/bin/env bash
# 可从任何工作目录运行；仅安装到项目 .venv，不需要 sudo。
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
用法：bash scripts/install.sh [--model-dir 模型目录 | --download-model | --skip-model]
安装锁定版本的 Python 环境及论文检索依赖，然后进入模型配置向导。
已有 .env 不会被覆盖。模型下载约 9.4 GB，仅选择下载后才执行。
HELP
  exit 0
fi
installer_args=("$@")
model_options=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "--model-dir 后需要填写模型目录。" >&2
        exit 2
      fi
      model_options=$((model_options + 1))
      shift 2 ;;
    --download-model|--skip-model)
      model_options=$((model_options + 1))
      shift ;;
    *) echo "未知选项：$1。查看用法：bash scripts/install.sh --help" >&2; exit 2 ;;
  esac
done
if [[ "$model_options" -gt 1 ]]; then
  echo "--model-dir、--download-model 和 --skip-model 只能选择一个。" >&2
  exit 2
fi
set -- "${installer_args[@]}"
trap 'echo "安装未完成。请检查上方错误；可重新运行安装脚本继续。现有模型和对话文件不会被删除。" >&2' ERR
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "初版安装脚本仅支持 Linux x86_64，请使用 Linux NVIDIA GPU 环境。" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=name --format=csv,noheader; then
  echo "当前未检测到 NVIDIA GPU。可安装环境，但启动本地模型前需切换到 GPU 节点或安装驱动。" >&2
fi
mkdir -p logs
install_log="logs/install-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$install_log") 2>&1
printf '%s\n' "安装日志：$install_log"
printf '%s\n' "[1/4] 准备 uv 与 Python 3.11（不修改系统 Python）…"
if [[ -x "$project_dir/.tools/bin/uv" ]]; then
  uv_bin="$project_dir/.tools/bin/uv"
elif command -v uv >/dev/null; then
  uv_bin=$(command -v uv)
else
  if ! command -v curl >/dev/null; then
    echo "缺少 curl；请使用系统软件管理器安装 curl，或先按 README 安装 uv。" >&2
    exit 1
  fi
  installer_file=$(mktemp)
  trap 'rm -f -- "$installer_file"' EXIT
  curl --fail --location --show-error --retry 2 https://astral.sh/uv/0.12.10/install.sh -o "$installer_file"
  UV_UNMANAGED_INSTALL="$project_dir/.tools/bin" sh "$installer_file"
  uv_bin="$project_dir/.tools/bin/uv"
fi
"$uv_bin" python install 3.11
printf '%s\n' "[2/4] 安装固定版本依赖，首次下载可能需要较长时间和数 GB 空间…"
"$uv_bin" sync --frozen --python 3.11 --extra rag --no-dev --inexact
printf '%s\n' "[3/4] 检查 Python 依赖与动态库…"
"$project_dir/.venv/bin/python" -m local_agent.doctor --environment-only
printf '%s\n' "[4/4] 配置模型…"
"$project_dir/.venv/bin/python" -m local_agent.bootstrap "$@"
printf '%s\n' "安装完成。启动：bash scripts/start.sh" "检查环境：bash scripts/doctor.sh"
