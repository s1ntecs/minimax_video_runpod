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

Do not change any of these pins before the first successful GPU smoke test.

## 1. Prerequisites

You need:

- Docker with Buildx.
- A Docker registry account (Docker Hub is the simplest path).
- A RunPod account and API key.
- A RunPod Network Volume. Use **100 GB** for the first deployment so there is comfortable room for all models and generated outputs.
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

After the PR is merged, use:

```bash
git checkout main
git pull --ff-only
```

## 3. Build locally first

The image is Linux AMD64. Build with plain progress so dependency failures are visible in full:

```bash
docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  -t minimax-h3-ref2va:local \
  --load \
  .
```

A successful build already verifies all of the following without a GPU:

- the pinned ComfyUI source can be fetched;
- its Python dependencies resolve without replacing the CUDA-matched Torch stack;
- `pip check` passes;
- `handler.py` and `download_models.py` compile;
- the RunPod 1.12 URL downloader imports successfully;
- Torch is at least 2.10 and is a CUDA 13 build;
- the pinned ComfyUI source contains `MiniMaxH3ReferenceToVideo`;
- the expected dynamic Ref2VA image/video inputs are present;
- the complete ComfyUI node graph passes `--quick-test-for-ci --cpu`.

If this build fails, **do not deploy the image**. Fix the build first rather than bypassing an assertion.

## 4. Publish the exact image you built

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

Use a versioned tag such as `2026-09-02`. Do not deploy `latest` for the initial production validation.

Record the digest printed by Buildx after the push. Once the first GPU smoke test succeeds, prefer deploying the image by immutable digest rather than by a mutable tag.

## 5. Create the RunPod Network Volume

In RunPod:

1. Open **Storage**.
2. Click **New Network Volume**.
3. Pick a datacenter that also has the target GPU available.
4. Create a **100 GB** volume, for example `minimax-h3-models`.

For Serverless, RunPod mounts this volume at the fixed path:

```text
/runpod-volume
```

The worker stores persistent data as:

```text
/runpod-volume/models/
/runpod-volume/outputs/
/runpod-volume/.locks/
```

Do not change the mount path in the worker.

## 6. Create the Serverless endpoint

Create a **Queue-based** Serverless endpoint from the Docker image pushed in step 4.

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

Use only one worker during the first validation. After a real Ref2VA job succeeds, increase `workers max` for horizontal scaling.

The worker itself forces concurrency to **1 job per GPU worker**. Do not try to increase in-worker concurrency for H3; scale with additional workers instead.

### Environment variables

Recommended values:

```text
INFERENCE_TIMEOUT=1800
MAX_INFERENCE_TIMEOUT=3600
COMFYUI_STARTUP_TIMEOUT=240
MAX_INLINE_OUTPUT_MB=18
MAX_INPUT_MB=512
MAX_INPUT_FILES=20
SKIP_MODEL_DOWNLOAD=0
```

Optional:

```text
HF_TOKEN=<your Hugging Face token>
```

Do not override `COMFYUI_DIR`, `COMFYUI_COMMIT`, `HF_MODEL_REVISION`, `MODEL_ROOT`, `INPUT_ROOT`, or `OUTPUT_ROOT` for the first validated deployment.

## 7. What happens on the first worker boot

The startup sequence is intentionally strict:

1. Verify `/runpod-volume` exists and is writable.
2. Acquire `/runpod-volume/.locks/minimax-h3-ref2va.lock` so two cold-starting workers cannot download the same weights simultaneously.
3. Resolve all required weights against the pinned Hugging Face revision.
4. Write `/runpod-volume/models/.minimax_h3_ref2va_manifest.json`.
5. Start pinned ComfyUI v0.34.0.
6. Wait for `/system_stats`.
7. Verify `/object_info/MiniMaxH3ReferenceToVideo` exists.
8. Start the RunPod Serverless worker.
9. Run SDK worker fitness checks, including CUDA availability and required model files.

If the Network Volume already contains weights from the older unpinned worker but does not contain the new manifest, the first hardened boot deliberately re-resolves the files against the pinned model revision. Later boots reuse the verified files.

Healthy startup logs must include lines similar to:

```text
[models] model lock acquired
[models] manifest ready: /runpod-volume/models/.minimax_h3_ref2va_manifest.json
[startup] ComfyUI is ready and MiniMaxH3ReferenceToVideo is registered
```

## 8. Get a Ref2VA API workflow

The request must contain a **ComfyUI API-format workflow**, not a normal UI-format workflow export.

### Quality / baseline workflow

A public API-format H3 R2V workflow is available here:

`https://github.com/TheTerrasque/minimax-h3-frontend/blob/main/resources/workflows_api/video_minimax_h3_r2v.api.json`

It already uses:

- `MiniMaxH3ReferenceToVideo`;
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`;
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`;
- both H3 VAEs;
- native `SaveVideo`.

Before sending it, change every `LoadImage` / `LoadVideo` filename in the workflow to the same plain filename that you provide in `input.files`.

For example, if the workflow has:

```json
{
  "class_type": "LoadImage",
  "inputs": {
    "image": "reference_1.png"
  }
}
```

then the request must include a file named exactly `reference_1.png`.

The worker automatically moves that file into a job-isolated input folder and rewrites exact filename references inside the workflow.

### Turbo Ref2VA 4-step workflow

For the accelerated path use the official ModelTC Ref2VA Turbo workflow:

`https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/example_workflows/video_minimax_h3_ref2v_lightx2v_turbo.json`

Open it in the same pinned/current ComfyUI generation, then export **API Format**. Confirm before using it:

```text
Base model: minimax_h3_ref2va_pruned_int8_convrot.safetensors
LoRA:       minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
Steps:      4
Task node:  MiniMaxH3ReferenceToVideo
```

Do not substitute an FL2VA Turbo LoRA for this Ref2VA path.

## 9. Minimal request wrapper

Suppose your API workflow contains two `LoadImage` nodes named `reference_1.png` and `reference_2.png`.

Build the request JSON like this:

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
    "inline_output": true
  }
}
```

`files` accepts either `url` or `base64`, never both for the same entry. URL downloads use the RunPod 1.12 hardened downloader with its SSRF protections and size cap.

The current Ref2VA node supports up to:

```text
9 reference images
3 reference videos
3 reference-video audio tracks
3 standalone reference audio tracks
```

The worker default `MAX_INPUT_FILES=20` covers the full 18-file Ref2VA set with a small safety margin.

## 10. Send the first request

Set local shell variables:

```bash
export RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"
export ENDPOINT_ID="YOUR_ENDPOINT_ID"
```

Put the complete request in `request.json`, then submit asynchronously:

```bash
curl --request POST \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data @request.json
```

The response contains a job ID. Poll it with:

```bash
curl --request GET \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/YOUR_JOB_ID" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

Check endpoint health with:

```bash
curl --request GET \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/health" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## 11. Output behavior

Every save node's output prefix is automatically scoped to the RunPod job ID, so outputs do not collide across workers.

Persistent output path:

```text
/runpod-volume/outputs/<JOB_ID>/...
```

If `inline_output=true`, files up to `MAX_INLINE_OUTPUT_MB` are returned as base64 in the job output. The default is 18 MB to keep a safety margin below Serverless response-size limits.

If a result is too large for inline output, the response still includes its persistent Network Volume path. RunPod's S3-compatible Network Volume API maps it to:

```text
s3://<NETWORK_VOLUME_ID>/outputs/<JOB_ID>/...
```

For a production application that regularly returns large videos, use the Network Volume S3 API or your own object storage instead of large base64 responses.

## 12. First GPU smoke-test checklist

Do not increase `workers max` until all of these are true:

- Docker build completed without bypassing checks.
- Worker logs show the pinned model manifest.
- Worker logs show `MiniMaxH3ReferenceToVideo is registered`.
- `/health` shows a worker able to accept work.
- A real request reaches `COMPLETED`.
- The returned video actually uses the reference identities/content, not just the text prompt.
- Audio/video decode succeeds.
- Output appears under `/runpod-volume/outputs/<JOB_ID>/`.
- A second request reuses the existing model files instead of downloading them again.

The identity/reference-adherence check matters: a workflow can technically complete while reference connections are wrong. Visually verify the first result before calling the deployment production-ready.

## 13. Common failures

### `/runpod-volume is missing`

The Serverless endpoint does not have the Network Volume attached. Attach it in endpoint Advanced settings.

### `MiniMaxH3ReferenceToVideo is missing` or readiness fails

Do not point the worker back to the old bundled ComfyUI. Use the image from this repository unchanged. Check that the Docker build used the hardening branch/main after merge.

### `Workflow validation failed`

Most often the JSON is UI-format rather than API-format, a model filename differs, or a dynamic reference input is not connected correctly. Export API Format from a current ComfyUI and compare the node names with step 8.

### Reference images appear ignored

Verify the API workflow itself contains keys such as:

```text
ref_images.ref_image_0
ref_images.ref_image_1
```

inside the `MiniMaxH3ReferenceToVideo` node, and verify each referenced `LoadImage` filename exactly matches an entry in `input.files`.

### CUDA OOM

The worker already enforces one job per GPU. First reduce generation resolution/duration and use `ref_image_size="match"` rather than `max`. For the fast mode, verify the dedicated 4-step Ref2VA Turbo workflow is being used. If the workload still does not fit, move to a higher-VRAM Blackwell GPU.

### Worker keeps downloading models

Check that this file persists on the attached Network Volume:

```text
/runpod-volume/models/.minimax_h3_ref2va_manifest.json
```

If it disappears between boots, the endpoint is not using the intended persistent volume.

### Job completes but no output is returned

The workflow must contain a real save/output node such as `SaveVideo`. The worker intentionally treats "completed with no saved output" as an error.

## 14. After the first successful GPU generation

Only after the real GPU smoke test:

1. Pin the published Docker image by digest in RunPod.
2. Increase `workers max` as needed.
3. Keep per-worker concurrency at 1.
4. Decide whether production outputs should use inline base64, RunPod Network Volume S3, or another object store.
5. Benchmark quality mode vs Ref2VA Turbo 4-step using the same seed and references before making Turbo the default.

## 15. What this repository can and cannot prove before deployment

The Docker build performs strong static/import/startup validation, but it cannot prove CUDA kernel execution or H3 visual reference adherence without an actual NVIDIA GPU and the model files loaded. The final acceptance test is therefore the real Ref2VA GPU generation in step 12. Do not remove that validation step.
