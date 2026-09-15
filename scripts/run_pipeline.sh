#!/usr/bin/env sh
set -eu

if [ -n "${PYTHON:-}" ]; then
    python_bin="$PYTHON"
elif [ -x ".venv/bin/python" ]; then
    python_bin=".venv/bin/python"
else
    python_bin="python3"
fi

exec "$python_bin" -m scripts.run_pipeline "$@"
