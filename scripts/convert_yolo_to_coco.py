"""Convert data_v2 (YOLO format) to COCO detection JSON for D-FINE training/eval.

Usage:
    python scripts/convert_yolo_to_coco.py --split train --output third_party/D-FINE-master/dataset/custom/train.json
    python scripts/convert_yolo_to_coco.py --split val   --output third_party/D-FINE-master/dataset/custom/val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data_v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 4: "SU-35", 5: "C-130", 6: "C-17", 7: "C-5",
    8: "F-16", 9: "TU-160", 10: "E-3", 11: "B-52", 12: "P-3C", 13: "B-1B", 14: "E-8",
    15: "TU-22", 16: "F-15", 17: "KC-135", 18: "F-22", 19: "FA-18", 20: "TU-95",
    21: "KC-10", 22: "SU-34", 23: "SU-24", 24: "FSC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def image_path_for(stem: str, split: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = DATA / f"images/{split}" / (stem + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no image for {stem}")


def main() -> int:
    args = parse_args()
    img_dir = DATA / f"images/{args.split}"
    lbl_dir = DATA / f"labels/{args.split}"

    images = []
    annotations = []
    ann_id = 1
    for label_file in sorted(lbl_dir.glob("*.txt")):
        img_path = image_path_for(label_file.stem, args.split)
        with Image.open(img_path) as im:
            width, height = im.size
        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), *(float(p) for p in parts[1:5])
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            box_w = w * width
            box_h = h * height
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                # D-FINE uses category_id directly as a 0-based index when
                # remap_mscoco_category is False (its custom-dataset path).
                "category_id": cls,
                "bbox": [round(x1, 2), round(y1, 2), round(box_w, 2), round(box_h, 2)],
                "area": round(box_w * box_h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cls, "name": CLASS_NAMES[cls]} for cls in range(25)
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco), encoding="utf-8")
    print(
        f"split={args.split} images={len(images)} annotations={len(annotations)} "
        f"-> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
