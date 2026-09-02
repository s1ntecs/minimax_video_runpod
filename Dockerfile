# syntax=docker/dockerfile:1.7

# Immutable linux/amd64 CUDA 13 / PyTorch 2.10 base for Blackwell.
# Tag kept for readability; digest prevents the base from changing underneath us.
# The ComfyUI bundled in this image is intentionally NOT used for H3.
FROM runpod/comfyui:1.4.7-cuda13.0@sha256:bad26aad809a442a0d2674827d58c03f95686d0ea6d0d0e0cbebacd787488797

ENTRYPOINT []
USER root
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/h3-venv/bin:${PATH} \
    COMFYUI_COMMIT=12d5279438bfefc058a269eae805ceab6047777f \
    COMFYUI_DIR=/opt/comfyui-h3 \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    MODEL_ROOT=/runpod-volume/models \
    INPUT_ROOT=/tmp/minimax-h3/input \
    OUTPUT_ROOT=/runpod-volume/outputs \
    HF_MODEL_REVISION=dc559027db79c174125df4d827db55cd11178860 \
    MAX_INLINE_OUTPUT_MB=6 \
    MAX_INPUT_MB=512 \
    MAX_INPUT_FILES=20 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY requirements.txt constraints.txt /app/

# Isolate our Python packages from the interactive ComfyUI environment shipped
# in the base image, while reusing its tested CUDA-matched Torch installation.
RUN python -m venv --system-site-packages /opt/h3-venv \
    && /opt/h3-venv/bin/python -m pip install --no-cache-dir --upgrade "pip==25.2"

# Fetch immutable ComfyUI source. The pinned commit is the v0.34.0 release.
RUN /opt/h3-venv/bin/python - <<'PY'
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

# Keep the base image's Torch/CUDA pins, add the dependency set required by the
# pinned ComfyUI release, and bound packages with known breaking majors.
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
    && python -m py_compile /app/handler.py /app/download_models.py \
    && cd /app \
    && python -c "import handler; from runpod.serverless.utils import download_files_from_urls; print('handler import OK')"

# Fail the Docker build if CUDA or the exact H3 Ref2VA API implementation is missing.
RUN python - <<'PY'
import os
from pathlib import Path
import torch

root = Path('/opt/comfyui-h3')
assert root.joinpath('main.py').is_file(), 'Pinned ComfyUI main.py is missing'
assert root.joinpath('.pinned_commit').read_text().strip() == os.environ['COMFYUI_COMMIT']
source = root.joinpath('comfy_extras/nodes_minimax_h3.py').read_text(encoding='utf-8')
assert 'class MiniMaxH3ReferenceToVideo' in source, 'MiniMaxH3ReferenceToVideo is missing'
assert 'io.Autogrow.Input("ref_images"' in source, 'Expected Ref2VA autogrow API is missing'
assert 'prefix="ref_image_", min=0, max=9' in source, 'Unexpected Ref2VA image API'
assert 'prefix="ref_video_", min=0, max=3' in source, 'Unexpected Ref2VA video API'
assert 'prefix="ref_video_audio_", min=0, max=3' in source, 'Unexpected Ref2VA video-audio API'
assert 'prefix="ref_audio_", min=0, max=3' in source, 'Unexpected Ref2VA audio API'
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
