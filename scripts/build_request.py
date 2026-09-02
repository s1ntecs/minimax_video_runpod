#!/usr/bin/env python3
"""Build a complete RunPod request.json from a bundled Ref2VA API workflow."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    "quality": ROOT / "workflows/ref2va_quality_2ref.api.json",
    "turbo": ROOT / "workflows/ref2va_turbo_4step_2ref.api.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MiniMax H3 Ref2VA RunPod request JSON")
    parser.add_argument("--mode", choices=sorted(WORKFLOWS), default="turbo")
    parser.add_argument("--ref1", required=True, help="Public http(s) URL for <Picture 1>")
    parser.add_argument("--ref2", required=True, help="Public http(s) URL for <Picture 2>")
    parser.add_argument("--prompt", required=True, help="MiniMax H3 prompt; use <Picture 1>/<Picture 2> tags")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--inline-output", action="store_true")
    parser.add_argument("--output", default="request.json")
    return parser.parse_args()


def validate_url(url: str, name: str) -> None:
    if not url.startswith(("https://", "http://")):
        raise SystemExit(f"{name} must be a public http(s) URL")


def main() -> None:
    args = parse_args()
    validate_url(args.ref1, "--ref1")
    validate_url(args.ref2, "--ref2")
    if not 0.25 <= args.seconds <= 15.0:
        raise SystemExit("--seconds must be between 0.25 and 15.0 for this ready-made H3 example")
    if not 30 <= args.timeout <= 3600:
        raise SystemExit("--timeout must be between 30 and 3600")

    workflow = json.loads(WORKFLOWS[args.mode].read_text(encoding="utf-8"))
    workflow["138"]["inputs"]["value"] = args.prompt
    workflow["132"]["inputs"]["value"] = args.seconds
    workflow["129"]["inputs"]["noise_seed"] = args.seed

    payload = {
        "input": {
            "files": [
                {"name": "reference_1.png", "url": args.ref1},
                {"name": "reference_2.png", "url": args.ref2},
            ],
            "workflow": workflow,
            "timeout": args.timeout,
            "inline_output": bool(args.inline_output),
        }
    }

    destination = Path(args.output)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination} using {args.mode} workflow")


if __name__ == "__main__":
    main()
