"""Extract ship crop samples from data_v2 train for visual class-discrimination review.

For classes HM(0)/LQS(1) extracts ALL instances; for QHS(2)/MS(3) extracts a
deterministic stratified sample. Crops are padded, upscaled (smaller targets get
bigger upscale factors), and written to <output>/cls_<id>/ for vision review.

Usage:
    python scripts/extract_ship_samples.py --output tmp_ship_samples [--max-ms 8] [--max-qhs 8]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data_v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALL_CLASSES = (0, 1)  # HM, LQS
SAMPLE_CLASSES = (2, 3)  # QHS, MS
UPSCALE = {0: 1.5, 1: 2.0, 2: 3.0, 3: 2.5}  # class -> upscale factor for readability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp_ship_samples")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-ms", type=int, default=8)
    parser.add_argument("--max-qhs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def image_path_for(label_path: Path, img_dir: Path) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = img_dir / (label_path.stem + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no image found for {label_path.name}")


def main() -> None:
    args = parse_args()
    split_dir = DATA / f"labels/{args.split}"
    img_dir = DATA / f"images/{args.split}"
    out_root = PROJECT_ROOT / args.output
    rng = random.Random(args.seed)

    wanted: dict[int, list[tuple[Path, tuple[float, float, float, float]]]] = {c: [] for c in ALL_CLASSES + SAMPLE_CLASSES}

    label_files = sorted(split_dir.glob("*.txt"))
    for label_file in label_files:
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls not in wanted:
                continue
            cx, cy, w, h = (float(p) for p in parts[1:5])
            wanted[cls].append((label_file, (cx, cy, w, h)))

    # Stratified sampling for QHS/MS: keep spread across source images.
    for cls in SAMPLE_CLASSES:
        by_image: dict[str, list] = {}
        for label_file, box in wanted[cls]:
            by_image.setdefault(label_file.stem, []).append((label_file, box))
        sampled: list[tuple[Path, tuple[float, float, float, float]]] = []
        image_keys = sorted(by_image)
        rng.shuffle(image_keys)
        limit = args.max_ms if cls == 3 else args.max_qhs
        for key in image_keys:
            if len(sampled) >= limit:
                break
            sampled.append(by_image[key][0])
        wanted[cls] = sampled

    for cls, items in wanted.items():
        if not items:
            print(f"cls {cls}: no samples", flush=True)
            continue
        cls_dir = out_root / f"cls_{cls}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for index, (label_file, (cx, cy, w, h)) in enumerate(items):
            image_path = image_path_for(label_file, img_dir)
            with Image.open(image_path) as source:
                W, H = source.size
            pad = 0.06
            x1 = max(0, int((cx - w / 2 - w * pad) * W))
            y1 = max(0, int((cy - h / 2 - h * pad) * H))
            x2 = min(W, int((cx + w / 2 + w * pad) * W))
            y2 = min(H, int((cy + h / 2 + h * pad) * H))
            with Image.open(image_path) as source:
                crop = source.convert("RGB").crop((x1, y1, x2, y2))
            upscale = UPSCALE.get(cls, 1.5)
            if upscale != 1.0:
                crop = crop.resize(
                    (int(crop.width * upscale), int(crop.height * upscale)),
                    Image.Resampling.BILINEAR,
                )
            out_path = cls_dir / f"{index:03d}_{label_file.stem}.png"
            crop.save(out_path)
        print(f"cls {cls}: {len(items)} crops -> {cls_dir}", flush=True)


if __name__ == "__main__":
    main()
