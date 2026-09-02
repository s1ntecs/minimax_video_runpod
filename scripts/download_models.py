#!/usr/bin/env python3
import fcntl
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Comfy-Org/MiniMax-H3"
REVISION = os.environ.get(
    "HF_MODEL_REVISION",
    "dc559027db79c174125df4d827db55cd11178860",
)
ROOT = Path(os.environ.get("MODEL_ROOT", "/runpod-volume/models"))
LOCK_PATH = Path(os.environ.get("MODEL_LOCK", "/runpod-volume/.locks/minimax-h3-ref2va.lock"))
MANIFEST_PATH = ROOT / ".minimax_h3_ref2va_manifest.json"
MIN_VALID_BYTES = 1024 * 1024

FILES = [
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
]


def _manifest_revision() -> str | None:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("revision")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_VALID_BYTES


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    # Multiple Serverless workers may cold-start against the same Network Volume.
    # Serialize model initialization so two workers never write the same LFS file.
    with LOCK_PATH.open("a+") as lock_file:
        print(f"[models] waiting for volume lock: {LOCK_PATH}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print("[models] model lock acquired")

        revision_changed = _manifest_revision() not in (None, REVISION)
        manifest_files: dict[str, dict[str, int | str]] = {}

        for filename in FILES:
            target = ROOT / filename
            target.parent.mkdir(parents=True, exist_ok=True)

            if _is_valid(target) and not revision_changed:
                print(f"[models] present: {filename} ({target.stat().st_size / 2**30:.2f} GiB)")
            else:
                if target.exists() and not _is_valid(target):
                    target.unlink()
                print(f"[models] downloading: {REPO}@{REVISION}/{filename}")
                downloaded = Path(
                    hf_hub_download(
                        repo_id=REPO,
                        filename=filename,
                        revision=REVISION,
                        local_dir=str(ROOT),
                        token=token,
                        force_download=revision_changed,
                    )
                )
                if not _is_valid(target):
                    raise RuntimeError(
                        f"Model download failed validation: expected {target}, got {downloaded}"
                    )
                print(f"[models] ready: {filename} ({target.stat().st_size / 2**30:.2f} GiB)")

            manifest_files[filename] = {
                "bytes": target.stat().st_size,
                "path": str(target),
            }

        temp_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
        temp_manifest.write_text(
            json.dumps(
                {"repo": REPO, "revision": REVISION, "files": manifest_files},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp_manifest.replace(MANIFEST_PATH)
        print(f"[models] manifest ready: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
