import base64
import copy
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import runpod
import torch
from runpod.serverless.utils import download_files_from_urls

HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
PORT = os.environ.get("COMFYUI_PORT", "8188")
BASE = f"http://{HOST}:{PORT}"
INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/tmp/minimax-h3/input"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/runpod-volume/outputs"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/runpod-volume/models"))
DEFAULT_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "1800"))
MAX_TIMEOUT = int(os.environ.get("MAX_INFERENCE_TIMEOUT", "3600"))
MAX_INLINE_MB = int(os.environ.get("MAX_INLINE_OUTPUT_MB", "18"))
MAX_INPUT_MB = int(os.environ.get("MAX_INPUT_MB", "512"))
# Ref2VA supports 9 images + 3 videos + 3 video soundtracks + 3 standalone audios.
MAX_INPUT_FILES = int(os.environ.get("MAX_INPUT_FILES", "20"))
MIN_MODEL_BYTES = 1024 * 1024

REQUIRED_MODELS = [
    MODEL_ROOT / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    MODEL_ROOT / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    MODEL_ROOT / "vae/minimax_h3_video_vae_fp16.safetensors",
    MODEL_ROOT / "vae/minimax_h3_audio_vae_fp32.safetensors",
    MODEL_ROOT / "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
]


def _safe_segment(value: str, fallback: str = "job") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (cleaned or fallback)[:96]


def _safe_name(name: str) -> str:
    clean = Path(name).name
    if not clean or clean in {".", ".."} or name != clean or "/" in name or "\\" in name:
        raise ValueError(f"Invalid input file name: {name!r}; use a plain basename only")
    return clean


def _check_size(path: Path, label: str) -> None:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{label} is empty")
    if size > MAX_INPUT_MB * 1024 * 1024:
        raise ValueError(f"{label} exceeds MAX_INPUT_MB={MAX_INPUT_MB}")


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: _replace_strings(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_strings(v, replacements) for v in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _is_save_node(node: dict[str, Any]) -> bool:
    class_type = str(node.get("class_type", "")).lower()
    return "save" in class_type or "combine" in class_type


def _is_output_prefix_key(key: str) -> bool:
    return key in {"filename_prefix", "filename", "output_prefix"}


def _scope_output_prefixes(workflow: dict[str, Any], job_id: str) -> None:
    for node in workflow.values():
        if not isinstance(node, dict) or not _is_save_node(node):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if _is_output_prefix_key(str(key)) and isinstance(value, str):
                leaf = Path(value.replace("\\", "/")).name or "output"
                inputs[key] = f"{job_id}/{leaf}"


def _prepare_workflow(workflow: dict[str, Any], replacements: dict[str, str], job_id: str) -> dict[str, Any]:
    prepared = _replace_strings(copy.deepcopy(workflow), replacements)
    _scope_output_prefixes(prepared, job_id)
    return prepared


def _materialize_inputs(job_id: str, items: list[dict[str, Any]]) -> tuple[list[str], dict[str, str], Path]:
    if len(items) > MAX_INPUT_FILES:
        raise ValueError(f"Too many input files: {len(items)} > MAX_INPUT_FILES={MAX_INPUT_FILES}")

    job_input = INPUT_ROOT / job_id
    shutil.rmtree(job_input, ignore_errors=True)
    job_input.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    replacements: dict[str, str] = {}

    url_items: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every input.files entry must be an object")
        original_name = str(item.get("name", ""))
        name = _safe_name(original_name)
        if name in names:
            raise ValueError(f"Duplicate input file name: {name}")
        names.append(name)
        replacements[name] = f"{job_id}/{name}"

        has_url = "url" in item
        has_base64 = "base64" in item
        if has_url == has_base64:
            raise ValueError(f"Input {name!r} must contain exactly one of 'url' or 'base64'")
        if has_url:
            url = str(item["url"])
            if not url.startswith(("https://", "http://")):
                raise ValueError(f"Input {name!r} URL must be http(s)")
            url_items.append((name, url))
            continue

        payload = str(item["base64"])
        if payload.startswith("data:"):
            if "," not in payload:
                raise ValueError(f"Invalid data URL for {name!r}")
            payload = payload.split(",", 1)[1]
        estimated = (len(payload) * 3) // 4
        if estimated > MAX_INPUT_MB * 1024 * 1024:
            raise ValueError(f"Input {name!r} exceeds MAX_INPUT_MB={MAX_INPUT_MB}")
        target = job_input / name
        try:
            target.write_bytes(base64.b64decode(payload, validate=True))
        except Exception as exc:
            raise ValueError(f"Invalid base64 for input {name!r}: {exc}") from exc
        _check_size(target, f"Input {name!r}")

    if url_items:
        urls = [url for _, url in url_items]
        downloaded = download_files_from_urls(job_id, urls)
        if len(downloaded) != len(url_items) or any(path is None for path in downloaded):
            raise RuntimeError("One or more URL inputs failed to download")
        for (name, _), source_str in zip(url_items, downloaded, strict=True):
            source = Path(source_str)
            if not source.is_file():
                raise RuntimeError(f"RunPod downloader returned a missing file for {name!r}: {source}")
            _check_size(source, f"Input {name!r}")
            shutil.move(str(source), job_input / name)

    return names, replacements, job_input


def _queue(workflow: dict[str, Any]) -> str:
    response = requests.post(f"{BASE}/prompt", json={"prompt": workflow}, timeout=30)
    if not response.ok:
        raise RuntimeError(f"ComfyUI /prompt failed: {response.status_code} {response.text[:4000]}")
    data = response.json()
    if data.get("node_errors"):
        raise RuntimeError("Workflow validation failed: " + json.dumps(data["node_errors"], ensure_ascii=False)[:10000])
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {data}")
    return str(prompt_id)


def _wait(prompt_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = requests.get(f"{BASE}/history/{prompt_id}", timeout=15)
        response.raise_for_status()
        history = response.json()
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI execution failed: " + json.dumps(status, ensure_ascii=False)[:12000])
            if status.get("completed") is True:
                return item
        time.sleep(1)
    try:
        requests.post(f"{BASE}/interrupt", timeout=5)
    except Exception:
        pass
    raise TimeoutError(f"Inference exceeded {timeout_s}s")


def _collect(history_item: dict[str, Any], inline: bool) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    output_root = OUTPUT_ROOT.resolve()
    outputs = history_item.get("outputs", {})
    for node_id, node in outputs.items():
        if not isinstance(node, dict):
            continue
        for bucket in ("videos", "gifs", "images", "audio"):
            entries = node.get(bucket, []) or []
            if isinstance(entries, dict):
                entries = [entries]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                filename = entry.get("filename")
                if not filename:
                    continue
                subfolder = str(entry.get("subfolder") or "")
                path = (OUTPUT_ROOT / subfolder / str(filename)).resolve()
                if output_root not in path.parents and path != output_root:
                    continue
                result: dict[str, Any] = {
                    "node_id": str(node_id),
                    "kind": bucket,
                    "filename": str(filename),
                    "subfolder": subfolder,
                    "path": str(path),
                }
                if path.is_file():
                    size = path.stat().st_size
                    result["bytes"] = size
                    if inline and size <= MAX_INLINE_MB * 1024 * 1024:
                        result["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
                found.append(result)
    return found


def _validated_timeout(value: Any) -> int:
    timeout_s = int(value if value is not None else DEFAULT_TIMEOUT)
    if timeout_s < 30 or timeout_s > MAX_TIMEOUT:
        raise ValueError(f"timeout must be between 30 and {MAX_TIMEOUT} seconds")
    return timeout_s


@runpod.serverless.register_fitness_check
def check_h3_worker() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    missing = [str(path) for path in REQUIRED_MODELS if not path.is_file() or path.stat().st_size < MIN_MODEL_BYTES]
    if missing:
        raise RuntimeError("Required H3 model files are missing: " + ", ".join(missing))
    response = requests.get(f"{BASE}/object_info/MiniMaxH3ReferenceToVideo", timeout=5)
    response.raise_for_status()
    if "MiniMaxH3ReferenceToVideo" not in response.json():
        raise RuntimeError("ComfyUI MiniMaxH3ReferenceToVideo node is not registered")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    raw_job_id = str(job.get("id") or uuid.uuid4().hex)
    job_id = _safe_segment(raw_job_id, "job")
    data = job.get("input") or {}
    sdk_job_dir = Path("/app/jobs") / job_id
    job_input: Path | None = None

    try:
        if not isinstance(data, dict):
            raise ValueError("job.input must be an object")
        workflow = data.get("workflow")
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("input.workflow must be a non-empty ComfyUI API-format workflow object")
        timeout_s = _validated_timeout(data.get("timeout"))
        files = data.get("files") or []
        if not isinstance(files, list):
            raise ValueError("input.files must be an array")

        names, replacements, job_input = _materialize_inputs(job_id, files)
        prepared_workflow = _prepare_workflow(workflow, replacements, job_id)
        prompt_id = _queue(prepared_workflow)
        history = _wait(prompt_id, timeout_s)
        outputs = _collect(history, bool(data.get("inline_output", False)))
        if not outputs:
            raise RuntimeError("ComfyUI completed but no saved outputs were found in prompt history")

        return {
            "job_id": raw_job_id,
            "prompt_id": prompt_id,
            "input_files": names,
            "outputs": outputs,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return {
            "job_id": raw_job_id,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        if job_input is not None:
            shutil.rmtree(job_input, ignore_errors=True)
        shutil.rmtree(sdk_job_dir, ignore_errors=True)


if __name__ == "__main__":
    # H3 Ref2VA is a full-GPU workload. Never run two jobs concurrently inside
    # one worker; scale horizontally with more workers instead.
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda current: 1,
    })
