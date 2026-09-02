# syntax=docker/dockerfile:1.7

# CUDA 13 / PyTorch 2.10 base for Blackwell. The ComfyUI bundled in this image
# is intentionally NOT used for H3: it is older than the current H3 Ref2VA nodes.
FROM runpod/comfyui:1.4.7-cuda13.0

ARG COMFYUI_COMMIT=12d5279438bfefc058a269eae805ceab6047777f

ENTRYPOINT []
USER root
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/h3-venv/bin:${PATH} \
    COMFYUI_DIR=/opt/comfyui-h3 \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    MODEL_ROOT=/runpod-volume/models \
    INPUT_ROOT=/tmp/minimax-h3/input \
    OUTPUT_ROOT=/runpod-volume/outputs \
    HF_MODEL_REVISION=dc559027db79c174125df4d827db55cd11178860 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY requirements.txt constraints.txt /app/

# Isolate our Python packages from the interactive ComfyUI environment shipped
# in the base image, while reusing its tested CUDA-matched Torch installation.
RUN python -m venv --system-site-packages /opt/h3-venv \
    && /opt/h3-venv/bin/python -m pip install --no-cache-dir --upgrade "pip==25.2"

# Fetch an immutable ComfyUI source commit. v0.34.0 resolves to this SHA.
RUN COMFYUI_COMMIT="${COMFYUI_COMMIT}" /opt/h3-venv/bin/python - <<'PY'
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

commit = os.environ["COMFYUI_COMMIT"]
dest = Path("/opt/comfyui-h3")
url = f"https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/{commit}"

with tempfile.TemporaryDirectory() as tmp:
    archive = Path(tmp) / "comfyui.tar.gz"
    urllib.request.urlretrieve(url, archive)
    extract_root = Path(tmp) / "extract"
    extract_root.mkdir()
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_root, filter="data")
    roots = [p for p in extract_root.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Unexpected ComfyUI archive layout: {roots}")
    shutil.copytree(roots[0], dest)
(dest / ".pinned_commit").write_text(commit + "\n", encoding="utf-8")
PY

# Keep the base image's Torch/CUDA pins, add the exact dependency set required
# by the pinned ComfyUI release, and bound packages with known breaking majors.
RUN /opt/h3-venv/bin/python -m pip install --no-cache-dir \
      --constraint /opt/comfyui-runtime-constraints.txt \
      --constraint /app/constraints.txt \
      -r /opt/comfyui-h3/requirements.txt \
      -r /app/requirements.txt \
    && /opt/h3-venv/bin/python -m pip check

COPY handler.py /app/handler.py
COPY scripts/start.sh /app/start.sh
COPY scripts/download_models.py /app/download_models.py
COPY extra_model_paths.yaml /app/extra_model_paths.yaml

RUN chmod +x /app/start.sh \
    && python -m py_compile /app/handler.py /app/download_models.py

# Fail the Docker build if the CUDA stack or the H3 Ref2VA node is missing.
RUN python - <<'PY'
from pathlib import Path
import torch

root = Path('/opt/comfyui-h3')
assert root.joinpath('main.py').is_file(), 'Pinned ComfyUI main.py is missing'
assert root.joinpath('.pinned_commit').read_text().strip() == '12d5279438bfefc058a269eae805ceab6047777f'
source = root.joinpath('comfy_extras/nodes_minimax_h3.py').read_text(encoding='utf-8')
assert 'class MiniMaxH3ReferenceToVideo' in source, 'MiniMaxH3ReferenceToVideo is missing'
assert 'io.Autogrow.Input("ref_images"' in source, 'Expected Ref2VA autogrow API is missing'
major, minor = map(int, torch.__version__.split('+')[0].split('.')[:2])
assert (major, minor) >= (2, 10), f'Expected torch >=2.10, got {torch.__version__}'
assert torch.version.cuda and torch.version.cuda.startswith('13.'), f'Expected CUDA 13 torch build, got {torch.version.cuda}'
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
PY

# Imports the full ComfyUI node graph during the image build. This catches
# dependency/import regressions before a GPU worker is ever provisioned.
RUN cd /opt/comfyui-h3 && timeout 300 python main.py --quick-test-for-ci --cpu

CMD ["/app/start.sh"]
