"""Convert HRSC2016 ship samples into YOLO labels and merge into data_v2 train.

Mapping (vision-confirmed, see docs):
  HRSC 100000005/100000006/100000013 -> HM (class 0, carrier)
  HRSC 100000010                   -> LQS (class 1, amphibious ship)

Only objects of the mapped classes are kept; other objects in the same image are
ignored (images serve as HM/LQS positive samples). Images are copied (bmp -> jpg)
into data_v2/images/train with an `hrsc_` prefix.

Usage (on the server, where HRSC lives):
    python scripts/convert_hrsc_to_yolo.py \
        --hrsc /root/autodl-tmp/data/hrsc2016_extracted/HRSC2016/FullDataSet \
        --data-root data_v2
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_MAP: Dict[str, int] = {
    "100000005": 0,  # HM
    "100000006": 0,  # HM
    "100000013": 0,  # HM
    "100000010": 1,  # LQS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrsc", default="/root/autodl-tmp/data/hrsc2016_extracted/HRSC2016/FullDataSet")
    parser.add_argument("--data-root", default="data_v2")
    parser.add_argument("--min-objects", type=int, default=1, help="min mapped objects per image to keep")
    parser.add_argument(
        "--skip-mixed",
        action="store_true",
        help="Skip images that also contain unmapped objects (default: keep them, only mapped objects are labeled).",
    )
    return parser.parse_args()


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    ann_dir = Path(args.hrsc) / "Annotations"
    img_dir = Path(args.hrsc) / "AllImages"
    data_root = Path(args.data_root)
    out_img = data_root / "images/train"
    out_lbl = data_root / "labels/train"

    converted = 0
    kept_images = 0
    skipped_mixed = 0
    for xml_path in sorted(glob.glob(str(ann_dir / "*.xml"))):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            continue
        img_name = root.findtext("Img_FileName")
        img_path = img_dir / f"{img_name}.bmp"
        if not img_path.exists():
            continue

        mapped: List[Tuple[int, Tuple[float, float, float, float]]] = []
        has_other = False
        for obj in root.iter("HRSC_Object"):
            cid = obj.findtext("Class_ID")
            if cid not in CLASS_MAP:
                has_other = True
                continue
            x1 = fnum(obj.findtext("box_xmin"))
            y1 = fnum(obj.findtext("box_ymin"))
            x2 = fnum(obj.findtext("box_xmax"))
            y2 = fnum(obj.findtext("box_ymax"))
            if None in (x1, y1, x2, y2):
                continue
            mapped.append((CLASS_MAP[cid], (x1, y1, x2, y2)))

        if len(mapped) < args.min_objects:
            continue
        if has_other and args.skip_mixed:
            # Image mixes mapped and unmapped objects: skip to avoid unlabeled targets.
            skipped_mixed += 1
            continue

        with Image.open(img_path) as im:
            width, height = im.size
            im = im.convert("RGB")

        out_name = f"hrsc_{img_name}.jpg"
        im.save(out_img / out_name, quality=95)
        lines = []
        for cls, (x1, y1, x2, y2) in mapped:
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out_lbl / (out_name.rsplit(".", 1)[0] + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        converted += 1
        kept_images += 1

    print(f"kept images: {kept_images} (skipped mixed-object images: {skipped_mixed})", flush=True)

    # Class stats
    from collections import Counter

    counts = Counter()
    for lbl in out_lbl.glob("hrsc_*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            counts[int(line.split()[0])] += 1
    print(f"added objects: HM={counts[0]} LQS={counts[1]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
