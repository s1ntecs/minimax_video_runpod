# Bundled MiniMax H3 Ref2VA API workflows

These files are already in **ComfyUI API format** and can be placed directly in the RunPod request under `input.workflow`.

## `ref2va_quality_2ref.api.json`

Baseline quality path:

```text
Ref2VA INT8 ConvRot
sampler: res_multistep
scheduler: simple
steps: 20
ref_image_size: match
resolution preset: 0.5 MP / 16:9
```

This is based on the current native ComfyUI H3 Ref2VA graph and intentionally does not use the Turbo LoRA.

## `ref2va_turbo_4step_2ref.api.json`

Accelerated Ref2VA path using the dedicated LightX2V / ModelTC adapter:

```text
Ref2VA INT8 ConvRot
LoRA: minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
LoRA strength: 1.0
MiniMax H3 sigma shift: video 12 / audio 3
sampler: euler
scheduler: simple
steps: 4
ref_image_size: match
resolution preset: 0.5 MP / 16:9 (approximately 960x544)
```

Do not change only the step count to 4. The Turbo recipe is a matched set: dedicated Ref2VA LoRA + Euler + simple + 4 steps + H3 12/3 sigma shifts.

## Reference files

Both bundled examples use exactly two image references:

```text
reference_1.png -> <Picture 1>
reference_2.png -> <Picture 2>
```

Submit files using those names:

```json
"files": [
  {"name": "reference_1.png", "url": "https://.../first.png"},
  {"name": "reference_2.png", "url": "https://.../second.png"}
]
```

The worker moves them into a per-job directory and rewrites the exact filenames in the workflow automatically.

## What to edit

The most useful API nodes are:

```text
138 -> prompt (`inputs.value`)
129 -> seed (`inputs.noise_seed`)
132 -> duration in seconds (`inputs.value`)
115 -> target aspect/megapixels
136 -> Ref2VA dynamic reference inputs
```

If you need 1 or 3+ references, export a corresponding API-format graph from the pinned/current ComfyUI and preserve the dynamic keys exactly:

```text
ref_images.ref_image_0
ref_images.ref_image_1
...
```

The native Ref2VA node supports up to 9 images, 3 videos, 3 paired video soundtracks, and 3 standalone audios.
