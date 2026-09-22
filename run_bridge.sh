#!/usr/bin/env bash
#
# Starts the VoiceLK TTS Bridge on Linux without Docker.
#
# On first run it creates a virtual environment beside the service and installs the
# speech stack from the CPU wheel index. Afterwards it just starts the API.
#
#   ./run_bridge.sh                 # start on the configured port
#   PORT=8001 ./run_bridge.sh       # somewhere else
#   RELOAD=1 ./run_bridge.sh        # restart on code changes, for development
#   SKIP_INSTALL=1 ./run_bridge.sh  # skip the dependency check

set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$BRIDGE_DIR/.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

cd "$BRIDGE_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
fi

PYTHON="$VENV_DIR/bin/python"

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    echo "Installing dependencies (CPU torch index: $TORCH_INDEX)"
    "$PYTHON" -m pip install --extra-index-url "$TORCH_INDEX" \
        -r requirements-ml.txt -r requirements.txt
fi

ARGS=(-m uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1)
if [ "${RELOAD:-0}" = "1" ]; then
    ARGS+=(--reload)
fi

echo "Starting the TTS bridge on http://$HOST:$PORT"
exec "$PYTHON" "${ARGS[@]}"
