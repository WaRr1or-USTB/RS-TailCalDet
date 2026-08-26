"""Submit the generated showcase image to a running RS-CalVision API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = "demo_showcase/generated/images/demo_scene_grouped/big_demo_scene_grouped_000001.jpg"
LABEL_PATH = "demo_showcase/generated/labels/demo_scene_grouped/big_demo_scene_grouped_000001.txt"


def request_json(url: str, data: dict | None = None) -> dict:
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RS-CalVision dense showcase detection.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if not (PROJECT_ROOT / IMAGE_PATH).is_file() or not (PROJECT_ROOT / LABEL_PATH).is_file():
        raise FileNotFoundError("Showcase files are missing. Run build_demo_mosaic.py first.")

    run = request_json(
        f"{args.api.rstrip('/')}/runs",
        {
            "method_id": "rs-tailcaldet-v1",
            "image_path": IMAGE_PATH,
            "label_path": LABEL_PATH,
            "protocols": ["strict_25", "official_overall", "official_by_category", "group_macro"],
        },
    )
    run_id = run["run_id"]
    print(f"run_id={run_id}")
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        current = request_json(f"{args.api.rstrip('/')}/runs/{run_id}")
        status = current["status"]
        print(f"status={status}", flush=True)
        if status == "succeeded":
            print(f"workbench={args.api.rstrip('/')}/workbench/?run={run_id}#workbench")
            return 0
        if status in {"failed", "cancelled"}:
            raise RuntimeError(current.get("error") or f"Run ended with status={status}")
        time.sleep(2)
    raise TimeoutError(f"Run did not finish within {args.timeout} seconds")


if __name__ == "__main__":
    raise SystemExit(main())
