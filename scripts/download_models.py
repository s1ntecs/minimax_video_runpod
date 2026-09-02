#!/usr/bin/env python3
import fcntl
import hashlib
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

# Immutable LFS metadata for the exact artifacts used by this worker.
# Sizes/hashes are the published Hugging Face pointers for these files.
FILES: dict[str, dict[str, int | str]] = {
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors": {
        "bytes": 20_970_379_616,
        "sha256": "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
    },
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": {
        "bytes": 15_687_142_551,
        "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    },
    "vae/minimax_h3_video_vae_fp16.safetensors": {
        "bytes": 5_207_808_496,
        "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    },
    "vae/minimax_h3_audio_vae_fp32.safetensors": {
        "bytes": 605_254_808,
        "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    },
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": {
        "bytes": 1_956_193_000,
        "sha256": "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
    },
}


def _read_manifest() -> dict:
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _size_matches(path: Path, expected_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_trusts_file(manifest: dict, filename: str, spec: dict[str, int | str]) -> bool:
    if manifest.get("repo") != REPO or manifest.get("revision") != REVISION:
        return False
    item = (manifest.get("files") or {}).get(filename)
    if not isinstance(item, dict):
        return False
    return (
        item.get("bytes") == spec["bytes"]
        and item.get("sha256") == spec["sha256"]
    )


def _verify_hash(path: Path, expected_sha: str, label: str) -> None:
    print(f"[models] sha256: {label}")
    actual = _sha256(path)
    if actual != expected_sha:
        raise RuntimeError(
            f"SHA256 mismatch for {label}: expected {expected_sha}, got {actual}"
        )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    # Multiple Serverless workers may cold-start against the same Network Volume.
    # Serialize initialization so they never mutate the same large file together.
    with LOCK_PATH.open("a+") as lock_file:
        print(f"[models] waiting for volume lock: {LOCK_PATH}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print("[models] model lock acquired")

        old_manifest = _read_manifest()
        new_manifest_files: dict[str, dict[str, int | str]] = {}

        for filename, spec in FILES.items():
            target = ROOT / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            expected_bytes = int(spec["bytes"])
            expected_sha = str(spec["sha256"])

            trusted = _manifest_trusts_file(old_manifest, filename, spec)
            if _size_matches(target, expected_bytes) and trusted:
                print(
                    f"[models] present + manifest verified: {filename} "
                    f"({target.stat().st_size / 2**30:.2f} GiB)"
                )
            elif _size_matches(target, expected_bytes):
                # Migration path for a volume initialized by an older worker:
                # avoid redownloading ~44 GB if the bytes already match exactly.
                _verify_hash(target, expected_sha, filename)
                print(f"[models] existing file cryptographically verified: {filename}")
            else:
                if target.exists():
                    print(
                        f"[models] deleting invalid-size file: {filename} "
                        f"({target.stat().st_size} != {expected_bytes})"
                    )
                    target.unlink()

                print(f"[models] downloading pinned file: {REPO}@{REVISION}/{filename}")
                downloaded = Path(
                    hf_hub_download(
                        repo_id=REPO,
                        filename=filename,
                        revision=REVISION,
                        local_dir=str(ROOT),
                        token=token,
                        force_download=True,
                    )
                )
                if not _size_matches(target, expected_bytes):
                    raise RuntimeError(
                        f"Model size validation failed for {filename}: "
                        f"expected {expected_bytes}, got "
                        f"{target.stat().st_size if target.exists() else 'missing'}; "
                        f"hf_hub_download returned {downloaded}"
                    )
                try:
                    _verify_hash(target, expected_sha, filename)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                print(f"[models] ready: {filename} ({target.stat().st_size / 2**30:.2f} GiB)")

            new_manifest_files[filename] = {
                "bytes": expected_bytes,
                "sha256": expected_sha,
                "path": str(target),
            }

        temp_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
        temp_manifest.write_text(
            json.dumps(
                {
                    "repo": REPO,
                    "revision": REVISION,
                    "files": new_manifest_files,
                },
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
