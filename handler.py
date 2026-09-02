import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import runpod

HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
PORT = os.environ.get("COMFYUI_PORT", "8188")
BASE = f"http://{HOST}:{PORT}"
INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/runpod-volume/inputs"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/runpod-volume/outputs"))
DEFAULT_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "1800"))
MAX_INLINE_MB = int(os.environ.get("MAX_INLINE_OUTPUT_MB", "18"))


def _safe_name(name: str) -> str:
    name = Path(name).name
    if not name or name in {".", ".."}:
        raise ValueError("Invalid file name")
    return name


def _materialize_inputs(items: list[dict[str, Any]]) -> list[str]:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for item in items:
        name = _safe_name(str(item.get("name", "")))
        target = INPUT_ROOT / name
        if "base64" in item:
            payload = str(item["base64"])
            if payload.startswith("data:"):
                payload = payload.split(",", 1)[1]
            target.write_bytes(base64.b64decode(payload, validate=True))
        elif "url" in item:
            with requests.get(str(item["url"]), stream=True, timeout=120) as r:
                r.raise_for_status()
                with target.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
        else:
            raise ValueError(f"Input {name!r} must contain 'url' or 'base64'")
        if target.stat().st_size == 0:
            raise ValueError(f"Input {name!r} is empty")
        names.append(name)
    return names


def _queue(workflow: dict[str, Any]) -> str:
    response = requests.post(f"{BASE}/prompt", json={"prompt": workflow}, timeout=30)
    if not response.ok:
        raise RuntimeError(f"ComfyUI /prompt failed: {response.status_code} {response.text[:4000]}")
    data = response.json()
    if data.get("node_errors"):
        raise RuntimeError("Workflow validation failed: " + json.dumps(data["node_errors"], ensure_ascii=False)[:8000])
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {data}")
    return str(prompt_id)


def _wait(prompt_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = requests.get(f"{BASE}/history/{prompt_id}", timeout=15)
        response.raise_for_status()
        history = response.json()
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI execution failed: " + json.dumps(status, ensure_ascii=False)[:10000])
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
    outputs = history_item.get("outputs", {})
    for node_id, node in outputs.items():
        for bucket in ("videos", "gifs", "images", "audio"):
            for entry in node.get(bucket, []) or []:
                filename = entry.get("filename")
                if not filename:
                    continue
                subfolder = entry.get("subfolder") or ""
                path = (OUTPUT_ROOT / subfolder / filename).resolve()
                if OUTPUT_ROOT.resolve() not in path.parents and path != OUTPUT_ROOT.resolve():
                    continue
                result: dict[str, Any] = {
                    "node_id": node_id,
                    "kind": bucket,
                    "filename": filename,
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


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    data = job.get("input") or {}
    workflow = data.get("workflow")
    if not isinstance(workflow, dict) or not workflow:
        return {"error": "input.workflow must be a ComfyUI API-format workflow object"}
    try:
        names = _materialize_inputs(data.get("files") or [])
        prompt_id = _queue(workflow)
        history = _wait(prompt_id, int(data.get("timeout", DEFAULT_TIMEOUT)))
        outputs = _collect(history, bool(data.get("inline_output", False)))
        return {
            "prompt_id": prompt_id,
            "input_files": names,
            "outputs": outputs,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
