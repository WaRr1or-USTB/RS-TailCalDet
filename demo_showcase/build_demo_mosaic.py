"""Build a balanced, tile-aligned 10000x10000 showcase mosaic from data_v2."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_big_mosaic_dataset import (
    grid_positions,
    load_samples,
    load_yaml,
    paste_sample,
    resolve_project_path,
)

Image.MAX_IMAGE_PIXELS = None

SHOWCASE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = SHOWCASE_ROOT / "generated"
PREVIEW_PATH = SHOWCASE_ROOT / "preview" / "demo_mosaic_preview.jpg"
FRONTEND_PREVIEW_PATH = PROJECT_ROOT / "frontend" / "public" / "demo" / "demo_mosaic_preview.jpg"
SPLIT_NAME = "demo_scene_grouped"
BIG_IMAGE = OUTPUT_ROOT / "images" / SPLIT_NAME / f"big_{SPLIT_NAME}_000001.jpg"
BIG_LABEL = OUTPUT_ROOT / "labels" / SPLIT_NAME / f"big_{SPLIT_NAME}_000001.txt"
MANIFEST_PATH = OUTPUT_ROOT / "manifests" / f"{SPLIT_NAME}_manifest.jsonl"
CELL_SIZE = 800
GROUP_QUOTAS = {"ship": 60, "aircraft": 60, "vehicle": 24}


def broad_group(class_id: int) -> str:
    if class_id <= 3:
        return "ship"
    if class_id == 24:
        return "vehicle"
    return "aircraft"


def choose_round_robin(samples: list, count: int) -> list:
    """Prefer low-density tiles and rotate across fine classes."""
    buckets: dict[int, list] = defaultdict(list)
    for sample in samples:
        buckets[sample.labels[0].cls].append(sample)
    for bucket in buckets.values():
        bucket.sort(key=lambda sample: (len(sample.labels), sample.image.name))

    class_ids = sorted(buckets)
    offsets = Counter()
    chosen = []
    while len(chosen) < count:
        progressed = False
        for class_id in class_ids:
            bucket = buckets[class_id]
            offset = offsets[class_id]
            if offset < len(bucket):
                chosen.append(bucket[offset])
                offsets[class_id] += 1
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            for class_id in class_ids:
                offsets[class_id] = 0
    return chosen


def interleave_groups(grouped: dict[str, list]) -> list:
    pattern = (
        "ship", "aircraft", "vehicle", "ship", "aircraft", "ship",
        "aircraft", "vehicle", "ship", "aircraft", "ship", "aircraft",
    )
    offsets = Counter()
    ordered = []
    while len(ordered) < sum(GROUP_QUOTAS.values()):
        for group in pattern:
            offset = offsets[group]
            if offset < len(grouped[group]):
                ordered.append(grouped[group][offset])
                offsets[group] += 1
    return ordered


def main() -> int:
    source_data = resolve_project_path("data_v2/dataset.yaml")
    config = load_yaml(source_data)
    samples = load_samples(source_data, config, "val", 0)
    by_group: dict[str, list] = defaultdict(list)
    for sample in samples:
        if sample.labels:
            by_group[broad_group(sample.labels[0].cls)].append(sample)

    selected = {
        group: choose_round_robin(by_group[group], quota)
        for group, quota in GROUP_QUOTAS.items()
    }
    ordered = interleave_groups(selected)
    positions = grid_positions(10000, 10000, CELL_SIZE)
    if len(ordered) != len(positions):
        raise RuntimeError(f"Expected {len(positions)} showcase cells, got {len(ordered)}")

    canvas = Image.new("RGB", (10000, 10000), (20, 25, 27))
    label_lines: list[str] = []
    placements: list[dict] = []
    for sample, (offset_x, offset_y) in zip(ordered, positions):
        transformed, placement = paste_sample(
            canvas, sample, offset_x, offset_y, CELL_SIZE, 10000, 10000
        )
        placement["broad_group"] = broad_group(sample.labels[0].cls)
        label_lines.extend(transformed)
        placements.append(placement)

    BIG_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    BIG_LABEL.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(BIG_IMAGE, quality=92)
    BIG_LABEL.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    group_targets = Counter()
    for sample in ordered:
        for label in sample.labels:
            group_targets[broad_group(label.cls)] += 1
    record = {
        "variant": "balanced_showcase",
        "split": SPLIT_NAME,
        "big_image": str(BIG_IMAGE),
        "big_label": str(BIG_LABEL),
        "big_size": [10000, 10000],
        "cell_size": CELL_SIZE,
        "source_images": len(ordered),
        "source_cells_by_group": GROUP_QUOTAS,
        "targets": len(label_lines),
        "targets_by_group": dict(group_targets),
        "placements": placements,
    }
    MANIFEST_PATH.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "mosaic_summary.json").write_text(
        json.dumps(record | {"placements": len(placements)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    preview = canvas.copy()
    preview.thumbnail((1800, 1800), resampling)
    preview.save(PREVIEW_PATH, quality=88, optimize=True)
    preview.save(FRONTEND_PREVIEW_PATH, quality=88, optimize=True)

    manifest = {
        "purpose": "RS-CalVision UI and live-detection showcase only",
        "data_type": "balanced proxy mosaic assembled from data_v2 validation crops",
        "not_a_continuous_satellite_scene": True,
        "big_image": str(BIG_IMAGE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "big_label": str(BIG_LABEL.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "preview": str(PREVIEW_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "size": [10000, 10000],
        "grid": [12, 12],
        "cell_size": CELL_SIZE,
        "source_cells": len(ordered),
        "source_cells_by_group": GROUP_QUOTAS,
        "labels": len(label_lines),
        "labels_by_group": dict(group_targets),
        "seed": 2026,
        "note": "Cells align with the audited 800-pixel inference grid; vehicle crops are deterministically reused because only seven exist in the validation split.",
    }
    (SHOWCASE_ROOT / "demo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"demo_image={BIG_IMAGE}")
    print(f"demo_label={BIG_LABEL}")
    print(f"source_cells_by_group={GROUP_QUOTAS}")
    print(f"labels_by_group={dict(group_targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
