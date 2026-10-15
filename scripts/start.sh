#!/usr/bin/env bash
# 可从任何目录调用；直接使用已安装环境，不在每次启动时联网同步依赖。
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "尚未安装环境。请先运行：bash scripts/install.sh" >&2
  exit 1
fi
case "${1:-}" in
  --help|-h|doctor|setup|rag-index|rag-search|bench-concurrency) ;;
  *)
    if ! "$python_bin" -m local_agent.bootstrap --check; then
      "$python_bin" -m local_agent.bootstrap
      if ! "$python_bin" -m local_agent.bootstrap --check; then
        echo "模型尚未配置。请运行 bash scripts/setup.sh 后再启动。" >&2
        exit 1
      fi
    fi
    ;;
esac
exec "$python_bin" -m local_agent.app "$@"
