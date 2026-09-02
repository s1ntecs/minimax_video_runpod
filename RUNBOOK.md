# MiniMax H3 Ref2VA RunPod Serverless — Build & Deployment Runbook

This is the step-by-step path to build, publish, deploy, and validate the worker in this repository.

## 0. What is pinned

The worker deliberately avoids floating core versions:

- Base image: `runpod/comfyui:1.4.7-cuda13.0` (used for its CUDA 13 / PyTorch 2.10 runtime, not for its bundled ComfyUI core).
- ComfyUI: v0.34.0 commit `12d5279438bfefc058a269eae805ceab6047777f`.
- RunPod Python SDK: `1.12.0`.
- MiniMax H3 model repository revision: `dc559027db79c174125df4d827db55cd11178860`.
- Ref2VA diffusion model: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`.
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`.
- Video VAE: `minimax_h3_video_vae_fp16.safetensors`.
- Audio VAE: `minimax_h3_audio_vae_fp32.safetensors`.
- Turbo LoRA: `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`.

Do not change these pins before the first successful GPU smoke test.

## 1. Prerequisites

You need:

- Docker with Buildx.
- A Docker registry account (Docker Hub is the simplest path).
- A RunPod account and API key.
- A RunPod Network Volume. Use **100 GB** for the first deployment so there is comfortable room for models and generated outputs.
- An NVIDIA Blackwell GPU for the intended CUDA 13 fast path. Start with one worker and one GPU. RTX 5090-class 32 GB or a larger Blackwell GPU is the intended target.
- A Hugging Face token is optional for the currently public model repository but can be set as `HF_TOKEN`.

## 2. Check out the code

Until the hardening PR is merged:

```bash
git clone https://github.com/s1ntecs/minimax_video_runpod.git
cd minimax_video_runpod
git fetch origin
git checkout fix/h3-ref2va-production-hardening
```

After the PR is merged:

```bash
git checkout main
git pull --ff-only
```

## 3. Build locally first

The image is Linux AMD64. Build with plain progress so dependency failures are visible:

```bash
docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  -t minimax-h3-ref2va:local \
  --load \
  .
```

A successful build verifies, without a GPU:

- the pinned ComfyUI source can be fetched;
- its dependencies resolve without replacing the CUDA-matched Torch stack;
- `pip check` passes;
- `handler.py` and `download_models.py` compile;
- the RunPod 1.12 hardened URL downloader imports;
- Torch is at least 2.10 and is a CUDA 13 build;
- the pinned source contains `MiniMaxH3ReferenceToVideo`;
- expected Ref2VA dynamic image/video inputs exist;
- the complete ComfyUI node graph passes `--quick-test-for-ci --cpu`.

If the build fails, **do not deploy it**. Fix the build rather than bypassing an assertion.

## 4. Publish the exact image

Replace `<DOCKERHUB_USER>` with your Docker Hub username:

```bash
docker login

docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  -t <DOCKERHUB_USER>/minimax-h3-ref2va:2026-09-02 \
  --push \
  .
```

Use a versioned tag, not `latest`, for the first validation. Record the digest printed after push. After the GPU smoke test succeeds, prefer deploying by immutable digest.

## 5. Create the RunPod Network Volume

In RunPod:

1. Open **Storage**.
2. Click **New Network Volume**.
3. Pick a datacenter that also has your target GPU.
4. Create a **100 GB** volume such as `minimax-h3-models`.

Serverless mounts a Network Volume at the fixed path:

```text
/runpod-volume
```

The worker persists:

```text
/runpod-volume/models/
/runpod-volume/outputs/
/runpod-volume/.locks/
```

## 6. Create the Serverless endpoint

Create a **Queue-based** Serverless endpoint from the image from step 4.

Recommended initial validation settings:

```text
GPU count:          1
Workers min:        0
Workers max:        1
Allowed CUDA:       13.0
Execution timeout:  3600 seconds
FlashBoot:          enabled if available
Network Volume:     attach the volume from step 5
```

Use one worker during the first validation. The worker code forces concurrency to **1 job per GPU worker**; scale horizontally with more workers after validation.

### Environment variables

Recommended values:

```text
INFERENCE_TIMEOUT=1800
MAX_INFERENCE_TIMEOUT=3600
COMFYUI_STARTUP_TIMEOUT=240
MAX_INLINE_OUTPUT_MB=6
MAX_INPUT_MB=512
MAX_INPUT_FILES=20
SKIP_MODEL_DOWNLOAD=0
```

Optional:

```text
HF_TOKEN=<your Hugging Face token>
```

Do not override `COMFYUI_DIR`, `COMFYUI_COMMIT`, `HF_MODEL_REVISION`, `MODEL_ROOT`, `INPUT_ROOT`, or `OUTPUT_ROOT` for the first validated deployment.

## 7. First worker boot

Startup is deliberately strict:

1. Verify `/runpod-volume` exists and is writable.
2. Acquire `/runpod-volume/.locks/minimax-h3-ref2va.lock` so simultaneous cold starts cannot corrupt model files.
3. Resolve required weights against the pinned Hugging Face revision.
4. Write `/runpod-volume/models/.minimax_h3_ref2va_manifest.json`.
5. Start pinned ComfyUI v0.34.0.
6. Wait for `/system_stats`.
7. Verify `/object_info/MiniMaxH3ReferenceToVideo`.
8. Start the RunPod worker.
9. Run SDK fitness checks for CUDA, model files, and the Ref2VA node.

If the volume contains weights from the previous unpinned worker but no new manifest, the first hardened boot intentionally re-resolves them against the pinned revision. Later boots reuse the verified files.

Healthy startup logs include:

```text
[models] model lock acquired
[models] manifest ready: /runpod-volume/models/.minimax_h3_ref2va_manifest.json
[startup] ComfyUI is ready and MiniMaxH3ReferenceToVideo is registered
```

## 8. Get a Ref2VA API workflow

The request must contain a **ComfyUI API-format workflow**, not a normal UI-format workflow.

### Quality / baseline

A public API-format H3 R2V workflow is available at:

`https://github.com/TheTerrasque/minimax-h3-frontend/blob/main/resources/workflows_api/video_minimax_h3_r2v.api.json`

It already uses the same Ref2VA model, Qwen encoder, both VAEs, `MiniMaxH3ReferenceToVideo`, and native `SaveVideo`.

Change each `LoadImage` / `LoadVideo` filename to exactly match the corresponding `input.files[].name` value.

Example workflow input:

```json
{
  "class_type": "LoadImage",
  "inputs": {
    "image": "reference_1.png"
  }
}
```

Then the request must contain a file named exactly `reference_1.png`. The worker rewrites that exact filename to a job-isolated input path before queueing the workflow.

### Turbo Ref2VA 4-step

Use ModelTC's Ref2VA Turbo workflow:

`https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/example_workflows/video_minimax_h3_ref2v_lightx2v_turbo.json`

Open it in current/pinned ComfyUI and export **API Format**. Before using it confirm:

```text
Base model: minimax_h3_ref2va_pruned_int8_convrot.safetensors
LoRA:       minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
Steps:      4
Task node:  MiniMaxH3ReferenceToVideo
```

Do not substitute an FL2VA Turbo LoRA for the Ref2VA path.

## 9. Request wrapper

Prefer URL inputs. Queue-based `/run` has a **10 MB request/response payload limit**, while `/runsync` has **20 MB**. Large base64 reference images/video/audio can be rejected by the gateway before the handler runs.

Example:

```json
{
  "input": {
    "files": [
      {
        "name": "reference_1.png",
        "url": "https://your-public-host.example/reference_1.png"
      },
      {
        "name": "reference_2.png",
        "url": "https://your-public-host.example/reference_2.png"
      }
    ],
    "workflow": {
      "PASTE_THE_COMPLETE_COMFYUI_API_WORKFLOW_HERE": true
    },
    "timeout": 1800,
    "inline_output": false
  }
}
```

Each file accepts exactly one of `url` or `base64`. URL downloads use RunPod SDK 1.12's hardened downloader with SSRF protections and a streamed size cap.

Current Ref2VA supports up to:

```text
9 reference images
3 reference videos
3 reference-video audio tracks
3 standalone reference audio tracks
```

The worker's `MAX_INPUT_FILES=20` covers the full 18-file Ref2VA set.

## 10. Send the first request

```bash
export RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"
export ENDPOINT_ID="YOUR_ENDPOINT_ID"
```

Save the complete payload as `request.json`, then use async `/run`:

```bash
curl --request POST \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data @request.json
```

Poll the returned job ID:

```bash
curl --request GET \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/YOUR_JOB_ID" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

Health:

```bash
curl --request GET \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/health" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

For H3, async `/run` is the safer default because generation can exceed the convenient synchronous wait window.

## 11. Output behavior and payload limits

Every save node is automatically scoped to the RunPod job ID. Persistent output path:

```text
/runpod-volume/outputs/<JOB_ID>/...
```

RunPod queue payload limits matter:

```text
/run:     10 MB
/runsync: 20 MB
```

Base64 adds roughly one third to the raw file size, plus JSON overhead. The container therefore defaults to:

```text
MAX_INLINE_OUTPUT_MB=6
```

With `inline_output=true`, only sufficiently small outputs are embedded. For normal H3 videos, prefer `inline_output=false` and retrieve the persistent file through the Network Volume S3 API or copy it to your own object storage.

Network Volume S3 path mapping:

```text
/runpod-volume/outputs/<JOB_ID>/file.mp4
        ->
s3://<NETWORK_VOLUME_ID>/outputs/<JOB_ID>/file.mp4
```

## 12. First GPU smoke-test checklist

Do not increase `workers max` until:

- Docker build passes without bypassing checks.
- Logs show the pinned model manifest.
- Logs show `MiniMaxH3ReferenceToVideo is registered`.
- `/health` shows the endpoint can accept work.
- A real request reaches `COMPLETED`.
- The returned video visibly follows the references rather than only the text prompt.
- Audio/video decoding succeeds.
- Output exists under `/runpod-volume/outputs/<JOB_ID>/`.
- A second worker boot reuses verified model files rather than downloading them again.

The visual reference-adherence check is mandatory: a graph can technically finish even when its reference wiring is wrong.

## 13. Common failures

### `/runpod-volume is missing`

Attach the Network Volume in the Serverless endpoint Advanced settings.

### `MiniMaxH3ReferenceToVideo is missing`

Do not redirect the worker to the old ComfyUI baked into the RunPod image. Use this repository's image unchanged.

### `Workflow validation failed`

Usually the workflow is UI-format instead of API-format, a model filename differs, or a dynamic reference input is not connected correctly.

### References appear ignored

Verify the `MiniMaxH3ReferenceToVideo` API node contains keys such as:

```text
ref_images.ref_image_0
ref_images.ref_image_1
```

and each `LoadImage` filename exactly matches `input.files[].name`.

### CUDA OOM

The worker already enforces one job per GPU. Reduce resolution/duration, use `ref_image_size="match"` instead of `max`, and for the accelerated path verify the dedicated Ref2VA 4-step LoRA/workflow. If it still does not fit, use a larger-VRAM Blackwell GPU.

### Models download every boot

Verify this persists on the volume:

```text
/runpod-volume/models/.minimax_h3_ref2va_manifest.json
```

### Job completes but no output is returned

The API workflow needs a real save node such as `SaveVideo`. The worker deliberately treats "completed with no saved output" as an error.

### Inline/base64 response disappears or gateway rejects it

Set `inline_output=false`. Retrieve the file from Network Volume/S3. Do not raise the inline limit just to force a video through the JSON response.

## 14. After the first successful GPU generation

Only after the real smoke test:

1. Pin the published Docker image by digest in RunPod.
2. Increase `workers max` if needed.
3. Keep per-worker concurrency at 1.
4. Choose a production output transport: Network Volume S3 or your own object store for large videos.
5. Benchmark quality mode vs Ref2VA Turbo 4-step using the same seed/references before making Turbo the default.

## 15. What the repository can and cannot prove before deployment

The Docker build performs strong dependency/import/startup validation, but it cannot prove CUDA kernel execution or visual reference adherence without an actual NVIDIA GPU and loaded model files. The final acceptance test is the real Ref2VA generation in step 12.
