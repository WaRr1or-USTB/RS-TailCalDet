"""Build leakage-safe broad-detection and crop-expert datasets from the official 25-class split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


FINE_NAMES = (
    "HM", "LQS", "QHS", "MS", "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5", "A5_F-16",
    "A6_TU-160", "A7_E-3", "A8_B-52", "A9_P-3C", "A10_B-1B", "A11_E-8", "A12_TU-22",
    "A13_F-15", "A14_KC-135", "A15_F-22", "A16_FA-18", "A17_TU-95", "A18_KC-10",
    "A19_SU-34", "A20_SU-24", "FSC",
)
BROAD_NAMES = ("ship", "aircraft", "vehicle")
BACKGROUND = "ZZ_BACKGROUND"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def fine_to_broad(class_id: int) -> int:
    if 0 <= class_id <= 3:
        return 0
    if 4 <= class_id <= 23:
        return 1
    if class_id == 24:
        return 2
    raise ValueError(f"fine class id outside [0, 24]: {class_id}")


@dataclass(frozen=True)
class Label:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float


@dataclass(frozen=True)
class CropCandidate:
    expert: str
    class_name: str
    image_path: Path
    box: tuple[int, int, int, int]
    source_kind: str
    source_index: int
    context: float


def read_labels(path: Path) -> list[Label]:
    if not path.exists():
        return []
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"expected 5 YOLO fields at {path}:{line_number}")
        label = Label(int(fields[0]), *(float(value) for value in fields[1:]))
        fine_to_broad(label.class_id)
        if not all(0.0 <= value <= 1.0 for value in (label.cx, label.cy, label.width, label.height)):
            raise ValueError(f"normalized coordinate outside [0, 1] at {path}:{line_number}")
        labels.append(label)
    return labels


def crop_box(label: Label, image_width: int, image_height: int, context: float) -> tuple[int, int, int, int]:
    """Return a clipped square crop around a normalized YOLO box."""
    object_width = max(2.0, label.width * image_width)
    object_height = max(2.0, label.height * image_height)
    side = max(object_width, object_height) * context
    cx, cy = label.cx * image_width, label.cy * image_height
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(image_width, int(round(cx + side / 2)))
    bottom = min(image_height, int(round(cy + side / 2)))
    if right <= left or bottom <= top:
        raise ValueError("degenerate crop after clipping")
    return left, top, right, bottom


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def link_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unknown link mode: {mode}")


def find_images(directory: Path, max_images: int) -> list[Path]:
    images = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return images[:max_images] if max_images else images


def random_background_boxes(
    labels: list[Label], width: int, height: int, count: int, rng: random.Random
) -> list[tuple[int, int, int, int]]:
    object_boxes = [crop_box(label, width, height, 1.0) for label in labels]
    typical_side = int(round(max((max(b[2] - b[0], b[3] - b[1]) for b in object_boxes), default=min(width, height) / 8)))
    typical_side = max(32, min(typical_side, min(width, height)))
    boxes = []
    for _ in range(count * 30):
        if len(boxes) >= count:
            break
        left = rng.randint(0, max(0, width - typical_side))
        top = rng.randint(0, max(0, height - typical_side))
        candidate = (left, top, left + typical_side, top + typical_side)
        if all(box_iou(candidate, object_box) <= 0.05 for object_box in object_boxes):
            boxes.append(candidate)
    return boxes


def prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def write_broad_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"path: {path.parent.as_posix()}\ntrain: images/train\nval: images/val\nnc: 3\n"
        "names:\n  0: ship\n  1: aircraft\n  2: vehicle\n",
        encoding="utf-8",
    )


def build_split(args, split: str, output: Path, rng: random.Random) -> tuple[list[CropCandidate], dict]:
    source_images = args.source / "images" / split
    source_labels = args.source / "labels" / split
    images = find_images(source_images, args.max_images)
    candidates: list[CropCandidate] = []
    counts = {"images": len(images), "objects": 0, "broad_labels": [0, 0, 0]}

    for image_path in images:
        labels = read_labels(source_labels / f"{image_path.stem}.txt")
        destination = output / "broad" / "images" / split / image_path.name
        link_image(image_path, destination, args.link_mode)
        mapped_lines = []
        with Image.open(image_path) as image:
            width, height = image.size
            for index, label in enumerate(labels):
                broad_id = fine_to_broad(label.class_id)
                counts["objects"] += 1
                counts["broad_labels"][broad_id] += 1
                mapped_lines.append(f"{broad_id} {label.cx:g} {label.cy:g} {label.width:g} {label.height:g}")
                contexts = args.train_contexts if split == "train" else (args.val_context,)
                for context in contexts:
                    candidates.append(
                        CropCandidate(
                            BROAD_NAMES[broad_id], FINE_NAMES[label.class_id], image_path,
                            crop_box(label, width, height, context), "positive", index, context,
                        )
                    )
                for expert in BROAD_NAMES:
                    if expert != BROAD_NAMES[broad_id]:
                        candidates.append(
                            CropCandidate(
                                expert, BACKGROUND, image_path,
                                crop_box(label, width, height, args.negative_context),
                                "cross_category", index, args.negative_context,
                            )
                        )
            for background_index, box in enumerate(
                random_background_boxes(labels, width, height, args.random_backgrounds_per_image, rng)
            ):
                for expert in BROAD_NAMES:
                    candidates.append(
                        CropCandidate(expert, BACKGROUND, image_path, box, "random_background", background_index, 1.0)
                    )
        label_destination = output / "broad" / "labels" / split / f"{image_path.stem}.txt"
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.write_text("\n".join(mapped_lines) + ("\n" if mapped_lines else ""), encoding="utf-8")
    return candidates, counts


def sample_negatives(candidates: list[CropCandidate], ratio: float, rng: random.Random) -> list[CropCandidate]:
    selected = []
    for expert in BROAD_NAMES:
        positives = [item for item in candidates if item.expert == expert and item.class_name != BACKGROUND]
        negatives = [item for item in candidates if item.expert == expert and item.class_name == BACKGROUND]
        maximum = int(round(len(positives) * ratio))
        if maximum and len(negatives) > maximum:
            negatives = rng.sample(negatives, maximum)
        selected.extend(positives)
        selected.extend(negatives)
    return sorted(
        selected,
        key=lambda item: (item.expert, item.class_name, str(item.image_path), item.source_kind, item.source_index, item.context),
    )


def write_crops(candidates: list[CropCandidate], output: Path, split: str, jpeg_quality: int) -> list[dict]:
    manifest = []
    current_path, current_image = None, None
    try:
        for sequence, item in enumerate(candidates):
            if item.image_path != current_path:
                if current_image is not None:
                    current_image.close()
                current_path = item.image_path
                current_image = Image.open(current_path).convert("RGB")
            context_tag = str(item.context).replace(".", "p")
            filename = (
                f"{item.image_path.stem}__{item.source_kind}"
                f"{item.source_index:04d}__ctx{context_tag}__{sequence:06d}.jpg"
            )
            destination = output / "experts" / item.expert / split / item.class_name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            current_image.crop(item.box).save(destination, quality=jpeg_quality, subsampling=0)
            manifest.append(
                {
                    "split": split,
                    "expert": item.expert,
                    "class_name": item.class_name,
                    "crop": destination.relative_to(output).as_posix(),
                    "source_image": item.image_path.as_posix(),
                    "source_kind": item.source_kind,
                    "source_index": item.source_index,
                    "context": item.context,
                    "box_xyxy": list(item.box),
                }
            )
    finally:
        if current_image is not None:
            current_image.close()
    return manifest


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project / "data_v2")
    parser.add_argument("--output", type=Path, default=project / "data_hierarchical")
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--train-contexts", type=float, nargs="+", default=(1.15, 1.5, 2.0))
    parser.add_argument("--val-context", type=float, default=1.5)
    parser.add_argument("--negative-context", type=float, default=1.5)
    parser.add_argument("--negative-positive-ratio", type=float, default=4.0)
    parser.add_argument("--random-backgrounds-per-image", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-images", type=int, default=0, help="Smoke-test limit per split; zero uses every image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    if args.negative_positive_ratio < 0:
        raise ValueError("--negative-positive-ratio must be non-negative")
    prepare_output(args.output)
    write_broad_yaml(args.output / "broad" / "dataset.yaml")
    rng = random.Random(args.seed)
    summary = {"source": str(args.source), "output": str(args.output), "seed": args.seed, "splits": {}}
    all_manifest = []
    for split in ("train", "val"):
        candidates, counts = build_split(args, split, args.output, rng)
        selected = sample_negatives(candidates, args.negative_positive_ratio, rng)
        manifest = write_crops(selected, args.output, split, args.jpeg_quality)
        all_manifest.extend(manifest)
        counts["expert_crops"] = len(manifest)
        counts["expert_crop_counts"] = {
            expert: sum(item["expert"] == expert for item in manifest) for expert in BROAD_NAMES
        }
        summary["splits"][split] = counts
    manifest_fields = (
        "split", "expert", "class_name", "crop", "source_image", "source_kind", "source_index", "context", "box_xyxy"
    )
    with (args.output / "expert_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(all_manifest)
    summary["fine_names"] = list(FINE_NAMES)
    summary["broad_names"] = list(BROAD_NAMES)
    summary["background_class"] = BACKGROUND
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
