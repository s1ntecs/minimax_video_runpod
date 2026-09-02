#!/usr/bin/env bash
set -Eeuo pipefail

export COMFYUI_DIR="${COMFYUI_DIR:-/opt/comfyui-h3}"
export COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"
export COMFYUI_PORT="${COMFYUI_PORT:-8188}"
export MODEL_ROOT="${MODEL_ROOT:-/runpod-volume/models}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/runpod-volume/outputs}"
export INPUT_ROOT="${INPUT_ROOT:-/tmp/minimax-h3/input}"

if [[ ! -d /runpod-volume ]]; then
  echo "[startup] /runpod-volume is missing. Attach a RunPod Network Volume." >&2
  exit 1
fi
if [[ ! -w /runpod-volume ]]; then
  echo "[startup] /runpod-volume is not writable." >&2
  exit 1
fi

mkdir -p "$MODEL_ROOT" "$OUTPUT_ROOT" "$INPUT_ROOT"
# Inputs are job-local scratch data. Never leave files from a previous worker boot.
rm -rf "${INPUT_ROOT:?}"/*

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  python /app/download_models.py
fi

cd "$COMFYUI_DIR"
python main.py \
  --listen "$COMFYUI_HOST" \
  --port "$COMFYUI_PORT" \
  --disable-auto-launch \
  --extra-model-paths-config /app/extra_model_paths.yaml \
  --output-directory "$OUTPUT_ROOT" \
  --input-directory "$INPUT_ROOT" \
  > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!

cleanup() {
  kill "$COMFY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python - <<'PY'
import os
import sys
import time
import requests

host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
port = os.environ.get("COMFYUI_PORT", "8188")
base = f"http://{host}:{port}"
deadline = time.time() + int(os.environ.get("COMFYUI_STARTUP_TIMEOUT", "240"))
last_error = None

while time.time() < deadline:
    try:
        stats = requests.get(f"{base}/system_stats", timeout=3)
        node = requests.get(f"{base}/object_info/MiniMaxH3ReferenceToVideo", timeout=3)
        if stats.ok and node.ok:
            payload = node.json()
            if "MiniMaxH3ReferenceToVideo" in payload:
                print("[startup] ComfyUI is ready and MiniMaxH3ReferenceToVideo is registered")
                sys.exit(0)
            last_error = f"Ref2VA node absent from object_info: {str(payload)[:1000]}"
        else:
            last_error = f"system_stats={stats.status_code}, object_info={node.status_code}"
    except Exception as exc:
        last_error = repr(exc)
    time.sleep(1)

print(f"[startup] ComfyUI failed readiness: {last_error}", file=sys.stderr)
print("[startup] Last ComfyUI log lines:", file=sys.stderr)
try:
    print(open("/tmp/comfyui.log", "r", errors="replace").read()[-16000:], file=sys.stderr)
except Exception:
    pass
sys.exit(1)
PY

cd /app
exec python /app/handler.py
