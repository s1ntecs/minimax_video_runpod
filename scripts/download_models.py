#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "Comfy-Org/MiniMax-H3"
ROOT = Path(os.environ.get("MODEL_ROOT", "/runpod-volume/models"))

FILES = [
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    for filename in FILES:
        target = ROOT / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > 1024 * 1024:
            print(f"[models] present: {filename} ({target.stat().st_size / 2**30:.2f} GiB)")
            continue
        print(f"[models] downloading: {REPO}/{filename}")
        hf_hub_download(
            repo_id=REPO,
            filename=filename,
            local_dir=str(ROOT),
            token=token,
        )
        if not target.is_file() or target.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"Model download failed validation: {target}")
        print(f"[models] ready: {filename} ({target.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
