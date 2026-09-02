# syntax=docker/dockerfile:1.7

# Pinned RunPod CUDA 13 / PyTorch stack. Do not replace with :latest.
FROM runpod/comfyui:1.4.7-cuda13.0

USER root
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    COMFYUI_DIR=/opt/comfyui-baked \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    MODEL_ROOT=/runpod-volume/models \
    OUTPUT_ROOT=/runpod-volume/outputs \
    INPUT_ROOT=/runpod-volume/inputs

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --constraint /opt/comfyui-runtime-constraints.txt -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY scripts/start.sh /app/start.sh
COPY scripts/download_models.py /app/download_models.py
COPY extra_model_paths.yaml /app/extra_model_paths.yaml
COPY test_input.json /app/test_input.json

RUN chmod +x /app/start.sh \
    && python -m py_compile /app/handler.py /app/download_models.py

# Build-time checks that catch a changed/incompatible base image immediately.
RUN python - <<'PY'
from pathlib import Path
import torch
root = Path('/opt/comfyui-baked')
assert root.joinpath('main.py').is_file(), 'ComfyUI main.py not found in pinned base image'
major, minor = map(int, torch.__version__.split('+')[0].split('.')[:2])
assert (major, minor) >= (2, 10), f'Expected torch >=2.10, got {torch.__version__}'
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
PY

CMD ["/app/start.sh"]
