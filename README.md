# MiniMax H3 Ref2VA — RunPod Serverless

Production-oriented RunPod Serverless worker for MiniMax H3 **Ref2VA** on Blackwell GPUs (RTX 5090 / RTX PRO 6000 class) using native ComfyUI H3 nodes.

## Why this stack

- Base image is pinned to `runpod/comfyui:1.4.7-cuda13.0` instead of `latest`.
- CUDA/PyTorch come from the pinned RunPod image and are not reinstalled by this repo.
- H3 Ref2VA uses `minimax_h3_ref2va_pruned_int8_convrot.safetensors`.
- Text encoder uses `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`.
- The optional fast path uses the dedicated `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` LoRA.
- No H3 custom node is installed: current ComfyUI has native MiniMax H3 / Ref2VA support.
- Model files live on a RunPod Network Volume instead of inside the Docker image.

## Files downloaded to the Network Volume

The worker bootstraps these files from `Comfy-Org/MiniMax-H3` on first start:

```text
/runpod-volume/models/
├── diffusion_models/
│   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
├── vae/
│   ├── minimax_h3_video_vae_fp16.safetensors
│   └── minimax_h3_audio_vae_fp32.safetensors
└── loras/
    └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

Downloads are skipped when a non-empty file already exists, so the large weights are paid for only on the first Network Volume initialization.

## RunPod deployment

1. Create a RunPod Network Volume with enough free space (recommend at least 80 GB to leave room for outputs/cache).
2. Create a Serverless endpoint from this GitHub repository/branch.
3. Attach the Network Volume at `/runpod-volume`.
4. Use a CUDA-13-capable Blackwell GPU for the intended fast profile.
5. Do **not** override the container command; the image runs `/app/start.sh`.

Optional environment variables:

```text
HF_TOKEN=<only needed if Hugging Face access requires it>
INFERENCE_TIMEOUT=1800
COMFYUI_STARTUP_TIMEOUT=180
MAX_INLINE_OUTPUT_MB=18
SKIP_MODEL_DOWNLOAD=0
```

## Request format

The worker intentionally accepts **ComfyUI API-format workflows**, not UI-format workflow JSON. Export from ComfyUI using `Save (API Format)` / the API workflow export.

```json
{
  "input": {
    "files": [
      {
        "name": "reference_1.png",
        "url": "https://example.com/reference_1.png"
      }
    ],
    "workflow": {
      "...": "ComfyUI API-format nodes"
    },
    "timeout": 1800,
    "inline_output": false
  }
}
```

Each entry in `files` supports either `url` or `base64`. The saved filename is what your workflow's `LoadImage` node must reference.

## Output

The handler waits for the ComfyUI prompt to complete and returns discovered video/image/audio outputs with their Network Volume path, byte size, node id, and filename. Set `inline_output=true` only for small files; outputs larger than `MAX_INLINE_OUTPUT_MB` are not base64-inlined.

## Reliability choices

The container fails early when:

- the pinned base image no longer contains `/opt/comfyui-baked/main.py`;
- PyTorch is older than 2.10;
- the PyTorch build is not CUDA 13;
- a required model download is empty/incomplete;
- ComfyUI does not become healthy before the startup timeout;
- ComfyUI rejects the submitted workflow;
- inference exceeds the configured timeout.

`transformers`, `huggingface-hub`, Torch, CUDA libraries and ComfyUI dependencies are deliberately **not re-pinned or upgraded here**. They remain the mutually tested set baked into the pinned RunPod ComfyUI image; installing a second dependency stack on top is a common source of H3 startup/import failures.

## Local build smoke test

```bash
docker build --platform linux/amd64 -t minimax-h3-ref2va-runpod .
```

The Docker build performs Python compilation plus CUDA/PyTorch/base-layout assertions. A real H3 inference test still requires a compatible NVIDIA GPU and the model files; a CPU-only CI build cannot prove GPU kernel/runtime correctness.
