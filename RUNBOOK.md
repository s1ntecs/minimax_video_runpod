# MiniMax H3 Ref2VA RunPod Serverless — exact build & launch guide

This is the end-to-end procedure for the worker in this repository. Follow it in order for the first deployment. Do not upgrade individual CUDA, Torch, ComfyUI, RunPod SDK, or H3 model pins until one real GPU Ref2VA generation has passed.

## 1. What is locked

The build uses immutable/controlled core versions:

```text
RunPod base tag:
  runpod/comfyui:1.4.7-cuda13.0

RunPod linux/amd64 base digest:
  sha256:bad26aad809a442a0d2674827d58c03f95686d0ea6d0d0e0cbebacd787488797

ComfyUI:
  v0.34.0
  commit 12d5279438bfefc058a269eae805ceab6047777f

RunPod Python SDK:
  1.12.0

MiniMax H3 model repository revision:
  dc559027db79c174125df4d827db55cd11178860
```

The `FROM` line uses **tag + digest**. The RunPod image contributes the tested CUDA 13 / PyTorch 2.10 stack. Its older bundled ComfyUI is not used for H3; the Docker build installs the pinned H3-compatible ComfyUI source into `/opt/comfyui-h3`.

Required H3 files are resolved onto the Network Volume:

```text
/runpod-volume/models/diffusion_models/
  minimax_h3_ref2va_pruned_int8_convrot.safetensors

/runpod-volume/models/text_encoders/
  qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

/runpod-volume/models/vae/
  minimax_h3_video_vae_fp16.safetensors
  minimax_h3_audio_vae_fp32.safetensors

/runpod-volume/models/loras/
  minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

## 2. Prerequisites

Install/have:

- Docker with Buildx.
- A Docker registry account, for example Docker Hub.
- RunPod account and API key.
- A RunPod Network Volume. Use **100 GB** for the first deployment.
- A Blackwell GPU for the intended CUDA 13 path, for example RTX 5090 / RTX PRO 6000 if available in your selected RunPod datacenter.
- Python 3 locally only if you want to use the included request builder. It has no third-party dependencies.

A Hugging Face token is currently optional for the public files, but the worker supports `HF_TOKEN` if access requirements change.

## 3. Clone the hardened PR branch

Until PR #2 is merged:

```bash
git clone https://github.com/s1ntecs/minimax_video_runpod.git
cd minimax_video_runpod
git fetch origin
git checkout fix/h3-ref2va-production-hardening
```

Verify:

```bash
git status
git branch --show-current
```

Expected branch:

```text
fix/h3-ref2va-production-hardening
```

After PR #2 is merged, use `main` instead.

## 4. Build the container locally

Run from the repository root:

```bash
docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  -t minimax-h3-ref2va:local \
  --load \
  .
```

Do not deploy if this command fails.

The Docker build intentionally tests more than syntax. It verifies:

```text
base image digest is fixed
pinned ComfyUI source downloads
ComfyUI dependencies resolve against the CUDA/Torch constraints
pip check passes
handler imports
RunPod 1.12 hardened downloader imports
bundled quality workflow parses and stays 20-step res_multistep/simple
bundled Turbo workflow parses and stays 4-step Euler/simple + Ref2VA LoRA + 12/3 shift
MiniMaxH3ReferenceToVideo exists in the pinned ComfyUI source
Ref2VA dynamic limits are 9 images / 3 videos / 3 paired video audios / 3 audios
Torch >= 2.10
Torch CUDA build is 13.x
full ComfyUI node graph passes --quick-test-for-ci --cpu
```

If an upstream dependency becomes incompatible, the intended behavior is for the **image build to fail**, rather than discovering the break only after paying for a GPU worker.

## 5. Push the exact image to Docker Hub

Log in:

```bash
docker login
```

Replace `<DOCKERHUB_USER>`:

```bash
docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  -t <DOCKERHUB_USER>/minimax-h3-ref2va:2026-09-02 \
  --push \
  .
```

Use a versioned tag for the first deployment. Do not use `latest`.

Keep the digest printed by Buildx after the push. After the real GPU smoke test succeeds, configure RunPod to use your published image by digest for maximum reproducibility.

## 6. Create the RunPod Network Volume

In RunPod:

1. Open **Storage**.
2. Create a new Network Volume.
3. Choose a datacenter where your target GPU is also available.
4. Use **100 GB** initially.
5. A practical name is `minimax-h3-models`.

Serverless mounts a Network Volume at the fixed path:

```text
/runpod-volume
```

This worker persists:

```text
/runpod-volume/models/
/runpod-volume/outputs/
/runpod-volume/.locks/
```

Inputs are deliberately **not** shared. They live on the worker's local container disk under `/tmp/minimax-h3/input/<JOB_ID>/` and are removed after each job.

If you intend to retrieve output through RunPod's S3-compatible Network Volume API, choose a datacenter where that API is supported.

## 7. Create the RunPod Serverless endpoint

Create a **Queue-based Serverless endpoint** using the image pushed in step 5.

For the first validation use:

```text
GPU per worker: 1
Minimum workers: 0
Maximum workers: 1
Network Volume: attach the volume from step 6
Execution timeout: allow up to 3600 seconds
```

Choose the Blackwell GPU you intend to run in production.

Do not raise maximum workers yet. First let one worker initialize the volume and pass a real generation. RunPod explicitly warns about concurrent writes to the same Network Volume; this worker also serializes model initialization with a volume lock, but the safest bootstrap is still one worker.

### Environment variables

Recommended first deployment:

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
HF_TOKEN=<your token>
```

Do not override these pinned/internal paths or revisions on the first deployment:

```text
COMFYUI_DIR
COMFYUI_COMMIT
HF_MODEL_REVISION
MODEL_ROOT
INPUT_ROOT
OUTPUT_ROOT
```

The startup script derives RunPod SDK's streamed URL cap from `MAX_INPUT_MB`, so the default 512 MiB per-file limit is enforced while downloading rather than only after download.

## 8. What happens during the first worker startup

The worker performs this sequence:

1. Confirms `/runpod-volume` exists and is writable.
2. Acquires `/runpod-volume/.locks/minimax-h3-ref2va.lock`.
3. Resolves all five required H3 files against the fixed Hugging Face revision.
4. Writes a provenance manifest:

```text
/runpod-volume/models/.minimax_h3_ref2va_manifest.json
```

5. Starts pinned ComfyUI v0.34.0.
6. Waits for `/system_stats`.
7. Calls `/object_info/MiniMaxH3ReferenceToVideo` and refuses to start the worker if the node is absent.
8. Starts RunPod Serverless.
9. RunPod worker fitness checks verify CUDA, required model files, and the live Ref2VA node.

Healthy logs include:

```text
[models] model lock acquired
[models] manifest ready: /runpod-volume/models/.minimax_h3_ref2va_manifest.json
[startup] ComfyUI is ready and MiniMaxH3ReferenceToVideo is registered
```

If the volume contains files from the older pre-hardening worker but has no pinned manifest, the hardened worker intentionally resolves them once against the immutable revision before trusting them.

## 9. Use the ready-made API workflows

You do **not** need to manually export a ComfyUI workflow for the normal 2-image case.

Bundled quality graph:

```text
workflows/ref2va_quality_2ref.api.json
```

Recipe:

```text
20 steps
res_multistep
simple scheduler
no Turbo LoRA
0.5 MP / 16:9 default
ref_image_size=match
```

Bundled fast graph:

```text
workflows/ref2va_turbo_4step_2ref.api.json
```

Recipe:

```text
Ref2VA Turbo LoRA v0.1 strength 1.0
4 steps
Euler
simple scheduler
MiniMax H3 video/audio shift 12/3
0.5 MP / 16:9 default, approximately the 960x544 Turbo training scale
ref_image_size=match
```

Both expect exactly:

```text
reference_1.png -> <Picture 1>
reference_2.png -> <Picture 2>
```

For 1 image, 3+ images, reference video, or reference audio, export/construct another API-format graph and keep the native dynamic keys exactly, for example:

```text
ref_images.ref_image_0
ref_images.ref_image_1
ref_videos.ref_video_0
ref_video_audios.ref_video_audio_0
ref_audios.ref_audio_0
```

The native maximum is 9 images + 3 videos + 3 paired video soundtracks + 3 standalone audio references.

## 10. Build `request.json` automatically

The simplest path is the included script.

### Turbo request

```bash
python scripts/build_request.py \
  --mode turbo \
  --ref1 "https://YOUR_PUBLIC_HOST/reference1.jpg" \
  --ref2 "https://YOUR_PUBLIC_HOST/reference2.jpg" \
  --prompt "Use <Picture 1> and <Picture 2> as visual references. Preserve their identity and appearance. Generate a coherent cinematic shot with natural subject motion and camera movement." \
  --seconds 5 \
  --seed 42 \
  --timeout 1800 \
  --output request.json
```

### Quality request

Use the same command with:

```text
--mode quality
```

The script embeds the complete API workflow into `request.json` and sets:

```text
node 138: prompt
node 129: seed
node 132: duration
reference_1.png URL
reference_2.png URL
```

It defaults to `inline_output=false`, which is correct for normal video output.

## 11. Send the RunPod request

Set:

```bash
export RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"
export ENDPOINT_ID="YOUR_ENDPOINT_ID"
```

Submit asynchronously:

```bash
curl --request POST \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data @request.json
```

The response contains the RunPod job ID.

Poll it:

```bash
curl --request GET \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/YOUR_JOB_ID" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

Endpoint health:

```bash
curl --request GET \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/health" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

Use async `/run` for H3 instead of relying on a long synchronous connection.

## 12. Request and response size rules

Prefer public `http(s)` URLs for reference images/video/audio.

RunPod queue gateway limits are important:

```text
/run request/response payload:     10 MB
/runsync request/response payload: 20 MB
```

Base64 expands raw data by roughly one third, so large reference media should not be placed directly in the request JSON.

For the same reason, the worker defaults to:

```text
MAX_INLINE_OUTPUT_MB=6
```

Normal H3 videos should use:

```json
"inline_output": false
```

## 13. Output location

Every save node is rewritten to a unique path using both RunPod job ID and ComfyUI save-node ID. This avoids collisions across jobs and across multiple save nodes in one workflow.

Persistent files land under:

```text
/runpod-volume/outputs/<JOB_ID>/...
```

For production video delivery, retrieve/copy those files through the Network Volume S3-compatible API or your own object storage.

Network Volume mapping is:

```text
/runpod-volume/outputs/<JOB_ID>/file.mp4
        ->
s3://<NETWORK_VOLUME_ID>/outputs/<JOB_ID>/file.mp4
```

## 14. Mandatory first GPU smoke test

Before increasing `maximum workers`, verify all of these:

- Docker image built without bypassing any assertion.
- Worker startup produced the model manifest.
- Logs show `MiniMaxH3ReferenceToVideo is registered`.
- `/health` shows a worker can accept jobs.
- Turbo request reaches `COMPLETED`.
- Quality request reaches `COMPLETED`.
- Output MP4 opens and contains synchronized audio/video.
- **The generated video actually follows `<Picture 1>` / `<Picture 2>` visually.** A technically completed graph is not sufficient if references were wired incorrectly.
- Output exists under the correct job directory on the Network Volume.
- A later worker boot reuses the verified model files rather than resolving them from scratch.

After this passes, increase worker count as required. Keep in-worker concurrency at **1**; this repository intentionally scales H3 horizontally instead of trying to place two full generations on one GPU.

## 15. Common failures

### Docker build fails during `pip check` or ComfyUI quick test

Do not remove the check. The version set is no longer compatible with the pinned runtime. Inspect the first dependency/import error in the build log.

### Docker cannot pull the base digest

Make sure the build target is:

```text
linux/amd64
```

and that Docker can access Docker Hub. Do not silently replace the digest with `latest`.

### `/runpod-volume is missing`

The Network Volume is not attached to the Serverless endpoint.

### `MiniMaxH3ReferenceToVideo is missing`

The worker is not running the pinned ComfyUI source expected by this repository. Do not redirect `COMFYUI_DIR` to the old baked ComfyUI.

### `Workflow validation failed`

Check the returned `node_errors`. Typical causes are a wrong model filename, UI-format JSON instead of API-format JSON, or malformed dynamic reference keys.

If using the included 2-ref workflows, do not rename `reference_1.png` / `reference_2.png` inside only one side of the request. The request builder keeps them consistent automatically.

### References appear ignored

Inspect the API workflow's `MiniMaxH3ReferenceToVideo` node. For two images it must contain:

```text
ref_images.ref_image_0 -> LoadImage reference_1.png
ref_images.ref_image_1 -> LoadImage reference_2.png
```

Also make the prompt explicitly use `<Picture 1>` and `<Picture 2>`.

### Turbo looks broken

Turbo is not "quality workflow with steps changed to 4". The complete matched recipe is required:

```text
minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
strength 1.0
Euler
simple
4 steps
MiniMaxH3SigmaShift video=12 audio=3
```

Use the bundled Turbo workflow instead of manually changing individual fields.

### CUDA OOM

The worker already forces concurrency to one. First reduce resolution/duration and keep `ref_image_size=match`. If the job still does not fit, use a larger-VRAM Blackwell GPU.

### A URL input larger than 512 MiB fails

That is the configured per-file protection. Raise `MAX_INPUT_MB` only if you intentionally need larger reference media. The startup script automatically applies the same byte limit to the RunPod hardened downloader.

### Worker resolves model files every boot

Verify this persists:

```text
/runpod-volume/models/.minimax_h3_ref2va_manifest.json
```

If it disappears, the endpoint is not using the intended persistent Network Volume.

### Job finishes but response has no saved output

The graph needs a save node such as `SaveVideo`. The worker intentionally treats a completed graph with no saved output as an error.

### Gateway rejects inline output

Set:

```json
"inline_output": false
```

and retrieve the persistent file from Network Volume/S3. Do not raise the inline threshold simply to force a video through JSON.

## 16. What is still impossible to prove before a real GPU run

The repository now performs extensive deterministic build/import/workflow/startup checks, but a CPU Docker build cannot prove:

```text
actual Blackwell CUDA kernel execution
real VRAM headroom for your chosen resolution/duration/reference count
visual identity/reference adherence
real audio quality in the Ref2VA Turbo v0.1 path
```

The final acceptance criterion is therefore the real GPU smoke test in step 14. Keep PR #2 as draft until that test succeeds.
