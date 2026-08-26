"""Diagnose FSC (launcher) misses on the big-image evaluation.

Reads the edge-audit matching output (final model, frozen config) and profiles
every FSC ground-truth box: matched or missed, box size, whether it sits on a
mosaic seam, and which big image it belongs to. This tells us WHY launchers are
missed (too small / seam-clipped / specific scenes) before we pick an improvement.

Usage:
    python scripts/analyze_fsc_failures.py [--edge-csv runs/detect/edge_audit/val_scene_grouped/edge_targets.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {24: "FSC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edge-csv",
        default="runs/detect/edge_audit/val_scene_grouped/edge_targets.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = PROJECT_ROOT / args.edge_csv
    rows: List[Dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["cls"]) == 24:
                rows.append(r)

    total = len(rows)
    matched = [r for r in rows if r["matched"] == "1"]
    missed = [r for r in rows if r["matched"] == "0"]
    print(f"FSC GT total={total}  TP={len(matched)}  FN={len(missed)}  recall={len(matched)/total:.4f}", flush=True)

    def box_size(r: Dict) -> tuple:
        x1, y1, x2, y2 = (float(v) for v in json.loads(r["box"]))
        return x2 - x1, y2 - y1

    print("\n=== FN box size (pixels) ===", flush=True)
    sizes = [box_size(r) for r in missed]
    for w, h in sorted(sizes):
        print(f"  {w:8.1f} x {h:8.1f}  area={w*h:10.0f}  short={min(w, h):7.1f}", flush=True)

    print("\n=== FN distribution by image ===", flush=True)
    by_image: Dict[str, int] = {}
    for r in missed:
        by_image[r["image"]] = by_image.get(r["image"], 0) + 1
    for image, count in sorted(by_image.items(), key=lambda kv: -kv[1]):
        print(f"  {image}: {count}", flush=True)

    print("\n=== seam-clipped FN ===", flush=True)
    seam = [r for r in missed if r["is_edge"] == "1"]
    print(f"  {len(seam)}/{len(missed)} FN are on mosaic seams", flush=True)

    print("\n=== TP box size (pixels), first 15 ===", flush=True)
    for r in sorted(matched, key=lambda r: box_size(r)[0] * box_size(r)[1])[:15]:
        w, h = box_size(r)
        print(f"  {w:8.1f} x {h:8.1f}  area={w*h:10.0f}  short={min(w, h):7.1f}", flush=True)

    print("\n=== size threshold sweep (area / short side) ===", flush=True)
    for short_th in (30, 40, 50, 60, 80):
        fn_removed = sum(1 for r in missed if min(box_size(r)) < short_th)
        tp_removed = sum(1 for r in matched if min(box_size(r)) < short_th)
        print(
            f"  exclude short<{short_th:3d}px: FN -{fn_removed}  TP -{tp_removed}  "
            f"=> recall {len(matched) - tp_removed}/{total - tp_removed - fn_removed} = "
            f"{(len(matched) - tp_removed) / max(1, total - tp_removed - fn_removed):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
