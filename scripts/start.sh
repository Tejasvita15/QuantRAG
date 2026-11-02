#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Create the environment first: python3.11 scripts/setup.py" >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.venv/huggingface}"
cd "$PROJECT_ROOT"
exec "$PYTHON" -m streamlit run app.py --server.address 127.0.0.1 "$@"
