from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MODEL_PATH = Path("/app/models/best.pt")
THRESHOLDS_PATH = Path("/app/models/class_thresholds.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RS-TailCalDet competition inference entrypoint")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing input images")
    parser.add_argument("--output", required=True, type=Path, help="Directory for result.json")
    return parser.parse_args()


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is required but is not available inside the container")
    torch.cuda.set_device(0)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)


def discover_images(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {input_dir}")
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_images(paths: List[Path]):
    from PIL import Image

    loaded = []
    for path in paths:
        with Image.open(path) as source:
            loaded.append((path, source.convert("RGB")))
    return loaded


def build_image_record(path: Path, image, objects: List[dict], run_end_timestamp: int) -> dict:
    return {
        "image_id": path.stem,
        "file_name": path.name,
        "width": image.width,
        "height": image.height,
        "run_end_timestamp": run_end_timestamp,
        "objects": objects,
    }


def write_result(output_dir: Path, images: List[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    temporary_path = output_dir / "result.json.tmp"
    payload = {"status": "success", "images": images}
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(result_path)
    return result_path


def main() -> None:
    args = parse_args()
    check_gpu()

    from detector import Detector

    image_paths = discover_images(args.input)
    loaded_images = load_images(image_paths)
    detector = Detector(MODEL_PATH, THRESHOLDS_PATH)

    records = []
    try:
        for index, (path, image) in enumerate(loaded_images, start=1):
            objects = detector.predict(image)
            run_end_timestamp = time.time_ns() // 1_000_000
            records.append(build_image_record(path, image, objects, run_end_timestamp))
            print(
                f"image={index}/{len(loaded_images)} file={path.name} objects={len(objects)} "
                f"run_end_timestamp={run_end_timestamp}",
                flush=True,
            )
    finally:
        for _, image in loaded_images:
            image.close()

    result_path = write_result(args.output, records)
    print(f"result={result_path}", flush=True)


if __name__ == "__main__":
    main()
