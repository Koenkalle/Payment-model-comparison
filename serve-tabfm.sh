#!/usr/bin/env bash
# Start with a persistent TabFM runtime and cache. First-time setup: docs/tabfm.md.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tabfm_python="${PAYMENT_TABFM_PYTHON:-$HOME/.local/share/payment-model-comparison/tabfm-env/bin/python}"
if [[ ! -x "$tabfm_python" ]]; then
  echo "TabFM Python environment not found: $tabfm_python" >&2
  echo "See docs/tabfm.md, or set PAYMENT_TABFM_PYTHON to a prepared Python 3.11+ interpreter." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$project_root/artifacts/tabfm/hf-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
cd -- "$project_root"
exec "$tabfm_python" "$project_root/serve.py" "$@"
