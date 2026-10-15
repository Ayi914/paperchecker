#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
  echo "请先运行：bash scripts/install.sh" >&2
  exit 1
fi
exec .venv/bin/python -m local_agent.bootstrap "$@"
