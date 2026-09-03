#!/usr/bin/env python3
"""Build-time live ComfyUI API smoke test for bundled workflow class types."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", "/opt/comfyui-h3"))
WORKFLOW_DIR = Path(os.environ.get("WORKFLOW_DIR", "/app/workflows"))
HOST = "127.0.0.1"
PORT = 8191
BASE = f"http://{HOST}:{PORT}"
TIMEOUT = 180


def get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def workflow_classes() -> set[str]:
    result: set[str] = set()
    files = sorted(WORKFLOW_DIR.glob("*.api.json"))
    if not files:
        raise RuntimeError(f"No bundled API workflows found in {WORKFLOW_DIR}")
    for path in files:
        graph = json.loads(path.read_text(encoding="utf-8"))
        for node in graph.values():
            if isinstance(node, dict) and node.get("class_type"):
                result.add(str(node["class_type"]))
    return result


def main() -> None:
    log_path = Path("/tmp/comfyui-build-api.log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "main.py",
                "--cpu",
                "--listen",
                HOST,
                "--port",
                str(PORT),
                "--disable-auto-launch",
            ],
            cwd=COMFYUI_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        deadline = time.monotonic() + TIMEOUT
        last_error = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"ComfyUI exited early with code {process.returncode}")
            try:
                get_json(f"{BASE}/system_stats")
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        else:
            raise TimeoutError(f"ComfyUI API did not become ready: {last_error!r}")

        missing: list[str] = []
        for class_type in sorted(workflow_classes()):
            encoded = urllib.parse.quote(class_type, safe="")
            try:
                payload = get_json(f"{BASE}/object_info/{encoded}")
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                missing.append(class_type)
                continue
            if class_type not in payload:
                missing.append(class_type)

        if missing:
            raise RuntimeError("Pinned ComfyUI is missing bundled workflow nodes: " + ", ".join(missing))

        ref = get_json(f"{BASE}/object_info/MiniMaxH3ReferenceToVideo")
        if "MiniMaxH3ReferenceToVideo" not in ref:
            raise RuntimeError("MiniMaxH3ReferenceToVideo object_info is unavailable")

        print(f"Live ComfyUI API OK; validated {len(workflow_classes())} workflow class types")
    except Exception:
        try:
            print("--- ComfyUI build smoke log ---", file=sys.stderr)
            print(log_path.read_text(encoding="utf-8", errors="replace")[-20000:], file=sys.stderr)
        except Exception:
            pass
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
