#!/usr/bin/env bash
set -Eeuo pipefail

export COMFYUI_DIR="${COMFYUI_DIR:-/opt/comfyui-baked}"
export COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"
export COMFYUI_PORT="${COMFYUI_PORT:-8188}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/runpod-volume/outputs}"
export INPUT_ROOT="${INPUT_ROOT:-/runpod-volume/inputs}"

mkdir -p "$OUTPUT_ROOT" "$INPUT_ROOT" "${MODEL_ROOT:-/runpod-volume/models}"

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
import os, sys, time, requests
host = os.environ.get('COMFYUI_HOST', '127.0.0.1')
port = os.environ.get('COMFYUI_PORT', '8188')
url = f'http://{host}:{port}/system_stats'
deadline = time.time() + int(os.environ.get('COMFYUI_STARTUP_TIMEOUT', '180'))
while time.time() < deadline:
    try:
        r = requests.get(url, timeout=2)
        if r.ok:
            print('[startup] ComfyUI is ready')
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
print('[startup] ComfyUI failed to start. Last log lines:', file=sys.stderr)
try:
    print(open('/tmp/comfyui.log', 'r', errors='replace').read()[-12000:], file=sys.stderr)
except Exception:
    pass
sys.exit(1)
PY

exec python /app/handler.py
