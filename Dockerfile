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
# The RunPod base exposes /usr/bin/python3.12 but intentionally has no `python`
# command until its own entrypoint runs, so use the absolute interpreter here.
RUN /usr/bin/python3.12 -m venv --system-site-packages /opt/h3-venv \
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
COPY scripts/build_request.py /app/build_request.py
COPY scripts/validate_comfy_runtime.py /app/validate_comfy_runtime.py
COPY extra_model_paths.yaml /app/extra_model_paths.yaml
COPY workflows /app/workflows

RUN chmod +x /app/start.sh /app/build_request.py /app/validate_comfy_runtime.py \
    && python -m py_compile /app/handler.py /app/download_models.py /app/build_request.py /app/validate_comfy_runtime.py \
    && cd /app \
    && python -c "import handler; from runpod.serverless.utils import download_files_from_urls; print('handler import OK')"

# Validate the bundled API-format workflows and their exact sampling recipes.
RUN python - <<'PY'
import json
from pathlib import Path

root = Path('/app/workflows')
quality = json.loads((root / 'ref2va_quality_2ref.api.json').read_text(encoding='utf-8'))
turbo = json.loads((root / 'ref2va_turbo_4step_2ref.api.json').read_text(encoding='utf-8'))

for name, graph in [('quality', quality), ('turbo', turbo)]:
    assert isinstance(graph, dict) and graph, f'{name} workflow is empty'
    classes = {node.get('class_type') for node in graph.values() if isinstance(node, dict)}
    required = {
        'MiniMaxH3ReferenceToVideo', 'UNETLoader', 'CLIPLoader', 'VAELoader',
        'SamplerCustomAdvanced', 'CreateVideo', 'SaveVideo', 'LoadImage'
    }
    assert required <= classes, f'{name} workflow is missing nodes: {required - classes}'
    assert graph['127']['inputs']['unet_name'] == 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'
    assert graph['128']['inputs']['clip_name'] == 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
    assert graph['137']['inputs']['image'] == 'reference_1.png'
    assert graph['139']['inputs']['image'] == 'reference_2.png'
    assert graph['136']['inputs']['ref_images.ref_image_0'] == ['137', 0]
    assert graph['136']['inputs']['ref_images.ref_image_1'] == ['139', 0]

assert quality['123']['inputs']['sampler_name'] == 'res_multistep'
assert quality['124']['inputs']['scheduler'] == 'simple'
assert quality['124']['inputs']['steps'] == 20
assert not any(node.get('class_type') == 'LoraLoaderModelOnly' for node in quality.values())

assert turbo['123']['inputs']['sampler_name'] == 'euler'
assert turbo['124']['inputs']['scheduler'] == 'simple'
assert turbo['124']['inputs']['steps'] == 4
assert turbo['140']['class_type'] == 'LoraLoaderModelOnly'
assert turbo['140']['inputs']['lora_name'] == 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'
assert turbo['140']['inputs']['strength_model'] == 1.0
assert turbo['141']['class_type'] == 'MiniMaxH3SigmaShift'
assert turbo['141']['inputs']['shift_video'] == 12.0
assert turbo['141']['inputs']['shift_audio'] == 3.0
assert turbo['124']['inputs']['model'] == ['141', 0]
assert turbo['126']['inputs']['model'] == ['141', 0]
print('bundled Ref2VA workflows OK')
PY

# Fail the Docker build if CUDA or the exact H3 Ref2VA source API is missing.
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

# Import-only CI smoke test, then a real CPU ComfyUI HTTP server that proves
# every class_type referenced by the bundled workflows is actually registered.
RUN cd /opt/comfyui-h3 && timeout 300 python main.py --quick-test-for-ci --cpu
RUN python /app/validate_comfy_runtime.py

CMD ["/app/start.sh"]
