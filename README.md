# MiniMax H3 Ref2VA — RunPod Serverless

Production-oriented RunPod Serverless worker for MiniMax H3 **Ref2VA**, optimized for the CUDA 13 / Blackwell path and designed to fail early when versions or model files are wrong.

For the complete copy/paste build and deployment procedure, read **[RUNBOOK.md](RUNBOOK.md)**.

## Pinned runtime

The deployment intentionally avoids floating core components:

```text
RunPod base tag:    runpod/comfyui:1.4.7-cuda13.0
RunPod base digest: sha256:bad26aad809a442a0d2674827d58c03f95686d0ea6d0d0e0cbebacd787488797
PyTorch/CUDA:       base-image CUDA 13 pins (PyTorch 2.10 path)
ComfyUI core:       v0.34.0 / 12d5279438bfefc058a269eae805ceab6047777f
RunPod Python SDK:  1.12.0
H3 model revision:  dc559027db79c174125df4d827db55cd11178860
```

The Dockerfile uses **tag + digest**, so even if the Docker Hub tag is later moved, this build still resolves the same linux/amd64 base manifest. The ComfyUI bundled inside the RunPod image is **not** used as the H3 core. The Docker build installs the pinned H3-compatible ComfyUI source into `/opt/comfyui-h3` while reusing the base image's tested CUDA/Torch stack.

## Models on the Network Volume

First startup resolves these files against the pinned `Comfy-Org/MiniMax-H3` revision:

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

Downloads are protected by a Network Volume file lock so concurrent cold starts cannot corrupt the same model file. Once verified, the worker writes:

```text
/runpod-volume/models/.minimax_h3_ref2va_manifest.json
```

## Ready API workflows

The repository includes two already-exported **ComfyUI API-format** graphs:

```text
workflows/ref2va_quality_2ref.api.json
  20 steps · res_multistep · simple · no Turbo LoRA

workflows/ref2va_turbo_4step_2ref.api.json
  4 steps · euler · simple · Ref2VA Turbo LoRA 1.0 · H3 shift 12/3
```

Both examples expect:

```text
reference_1.png -> <Picture 1>
reference_2.png -> <Picture 2>
```

The Docker build parses both graphs and asserts their model names, reference wiring, sampler, scheduler, step count, Turbo LoRA and sigma shifts. See `workflows/README.md` for the editable node IDs.

You can build a complete queue payload without copying JSON by hand:

```bash
python scripts/build_request.py \
  --mode turbo \
  --ref1 "https://example.com/ref1.jpg" \
  --ref2 "https://example.com/ref2.jpg" \
  --prompt "Use <Picture 1> and <Picture 2> as references. ..." \
  --seconds 5 \
  --seed 42 \
  --output request.json
```

## Reliability checks

The image or worker deliberately fails instead of silently continuing when:

- ComfyUI dependencies cannot resolve against the CUDA-specific Torch pins;
- the RunPod SDK handler utilities do not import;
- a bundled API workflow is malformed or its sampling recipe changes unexpectedly;
- the pinned ComfyUI source does not contain the expected Ref2VA API;
- the ComfyUI node graph cannot import during the CPU build smoke test;
- `/runpod-volume` is missing or read-only;
- the required model files are missing;
- CUDA is unavailable;
- `MiniMaxH3ReferenceToVideo` is not registered after ComfyUI startup;
- the submitted workflow is invalid;
- inference exceeds its timeout;
- ComfyUI completes without a saved output.

## Job isolation

H3 is treated as one full-GPU job per worker. `concurrency_modifier` is fixed to `1`; scale horizontally by adding workers instead of running multiple H3 jobs on one GPU.

Each RunPod job gets its own local input directory. Exact input filenames in the API workflow are rewritten to that job directory, and save/output prefixes are scoped to the job ID **and save node ID**. Persistent results therefore land under:

```text
/runpod-volume/outputs/<JOB_ID>/...
```

URL inputs use RunPod SDK 1.12's hardened SSRF-safe downloader. Its streamed download cap is automatically tied to `MAX_INPUT_MB` (512 MiB by default) instead of using the SDK's much larger generic default.

## Build

```bash
docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  -t minimax-h3-ref2va:local \
  --load \
  .
```

See [RUNBOOK.md](RUNBOOK.md) before publishing or deploying the image.

## Request contract

The worker accepts a **ComfyUI API-format workflow**:

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
      "...": "complete ComfyUI API-format workflow"
    },
    "timeout": 1800,
    "inline_output": false
  }
}
```

Each `files` entry supports exactly one of `url` or `base64`. Prefer URLs for real reference video/audio payloads because RunPod queue requests have gateway payload limits.

The pinned Ref2VA node supports up to 9 reference images, 3 reference videos, 3 reference-video audio tracks, and 3 standalone audio references. `MAX_INPUT_FILES=20` covers the full set.

## Output

The response contains the RunPod job/prompt IDs and saved output metadata. Large files remain on the Network Volume and can be accessed through RunPod's S3-compatible Network Volume API.

For very small outputs, `inline_output=true` can include base64. The container default is `MAX_INLINE_OUTPUT_MB=6`, deliberately conservative because base64 increases payload size and queue-based `/run` requests/results have tighter gateway limits than raw files on the Network Volume.

## Important acceptance test

A successful Docker build proves dependency/import/startup compatibility, but it cannot prove CUDA kernel execution or visual reference adherence without a real GPU. Before increasing the endpoint beyond one worker, run a real Ref2VA generation and visually confirm that the result actually follows the references. The exact procedure is in [RUNBOOK.md](RUNBOOK.md).
