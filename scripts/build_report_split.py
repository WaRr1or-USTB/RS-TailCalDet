"""Build a leakage-resistant 60/10/30 report split from data_v2.

The locked 30% validation split is drawn only from the original competition
samples. External HRSC images are train-only. Derived ``fsc_aug*`` images are
used only when their source image is assigned to the training split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "calibration", "validation")


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    original_split: str
    kind: str
    group_key: str
    class_counts: Tuple[int, ...]
    parent_stem: str | None = None


@dataclass(frozen=True)
class GroupStats:
    key: str
    sample_indices: Tuple[int, ...]
    image_count: int
    class_counts: Tuple[int, ...]


class DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_v2")
    parser.add_argument("--output-dir", default="data_v2/report_split_v16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--calibration-ratio", type=float, default=0.10)
    parser.add_argument("--validation-ratio", type=float, default=0.30)
    parser.add_argument("--mar20-block-size", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=512)
    parser.add_argument("--skip-content-hash", action="store_true")
    parser.add_argument("--allow-unresolved-derived", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_ratios(ratios: Sequence[float]) -> None:
    if any(value <= 0 for value in ratios):
        raise ValueError("all split ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/calibration/validation ratios must sum to 1")
    if ratios[2] < 0.30:
        raise ValueError("locked validation ratio must be at least 0.30")


def derived_parent_stem(stem: str) -> str | None:
    match = re.match(r"^(?:fsc_)?aug\d+_(.+)$", stem, flags=re.IGNORECASE)
    return match.group(1) if match else None


def sample_kind(stem: str) -> str:
    if stem.lower().startswith("hrsc_"):
        return "external_train_only"
    if derived_parent_stem(stem) is not None:
        return "derived"
    return "official"


def scene_group_key(stem: str, mar20_block_size: int = 20) -> str:
    if mar20_block_size <= 0:
        raise ValueError("mar20_block_size must be positive")

    parent = derived_parent_stem(stem)
    if parent:
        stem = parent

    match = re.match(r"^MAR20_(\d+)$", stem, flags=re.IGNORECASE)
    if match:
        block = (int(match.group(1)) - 1) // mar20_block_size
        return f"mar20:{block:06d}"

    match = re.match(r"^((?:01|02)-PAN-\d{8}-[^-]+-[^-]+-L\d+)", stem, flags=re.IGNORECASE)
    if match:
        return "sat:" + match.group(1).lower()

    match = re.match(
        r"^([EW]\d+(?:\.\d+)?_[NS]\d+(?:\.\d+)?_\d{8}_L[12]A\d+)",
        stem,
        flags=re.IGNORECASE,
    )
    if match:
        return "sat:" + match.group(1).lower()

    if stem.lower().startswith("fsc_") and "-lv" in stem.lower():
        location = re.split(r"-lv", stem, maxsplit=1, flags=re.IGNORECASE)[0]
        return "fsc:" + location.lower()

    generic = re.sub(r"_crop\d+$", "", stem, flags=re.IGNORECASE)
    return "sample:" + generic.lower()


def label_path_for_image(image: Path, data_root: Path) -> Path:
    relative = image.relative_to(data_root)
    parts = list(relative.parts)
    try:
        image_idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"image is not under an images directory: {image}") from exc
    parts[image_idx] = "labels"
    return (data_root / Path(*parts)).with_suffix(".txt")


def read_class_counts(label: Path, nc: int) -> Tuple[int, ...]:
    if not label.exists():
        raise FileNotFoundError(f"missing label: {label}")
    counts = [0] * nc
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            raise ValueError(f"invalid YOLO label at {label}:{line_no}: {raw}")
        cls = int(float(parts[0]))
        if not 0 <= cls < nc:
            raise ValueError(f"class {cls} outside [0,{nc - 1}] at {label}:{line_no}")
        counts[cls] += 1
    return tuple(counts)


def load_dataset_config(data_root: Path) -> Tuple[int, Dict[int, str]]:
    cfg_path = data_root / "dataset.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml

        cfg = yaml.safe_load(text)
        nc = int(cfg["nc"])
        raw_names = cfg["names"]
        if isinstance(raw_names, list):
            names = dict(enumerate(map(str, raw_names)))
        else:
            names = {int(key): str(value) for key, value in raw_names.items()}
    except ImportError:
        nc_match = re.search(r"(?m)^nc:\s*(\d+)\s*$", text)
        if not nc_match:
            raise ValueError(f"could not parse nc from {cfg_path}")
        nc = int(nc_match.group(1))
        names = {
            int(match.group(1)): match.group(2).strip().strip("'\"")
            for match in re.finditer(r"(?m)^\s{2}(\d+):\s*(.+?)\s*$", text)
        }
    if set(names) != set(range(nc)):
        raise ValueError(f"dataset names do not cover class IDs 0..{nc - 1}: {cfg_path}")
    return nc, names


def load_samples(data_root: Path, nc: int, mar20_block_size: int) -> List[Sample]:
    samples: List[Sample] = []
    seen_paths = set()
    for original_split in ("train", "val"):
        image_dir = data_root / "images" / original_split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"missing image directory: {image_dir}")
        for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            resolved = image.resolve()
            if resolved in seen_paths:
                raise ValueError(f"duplicate image path: {resolved}")
            seen_paths.add(resolved)
            label = label_path_for_image(resolved, data_root)
            stem = image.stem
            kind = sample_kind(stem)
            samples.append(
                Sample(
                    image=resolved,
                    label=label.resolve(),
                    original_split=original_split,
                    kind=kind,
                    group_key=scene_group_key(stem, mar20_block_size),
                    class_counts=read_class_counts(label, nc),
                    parent_stem=derived_parent_stem(stem),
                )
            )
    return samples


def merge_exact_duplicate_groups(samples: Sequence[Sample]) -> Dict[str, str]:
    official_keys = sorted({sample.group_key for sample in samples if sample.kind == "official"})
    dsu = DisjointSet(official_keys)
    digest_owner: Dict[str, str] = {}
    for sample in samples:
        if sample.kind != "official":
            continue
        digest = hashlib.sha256(sample.image.read_bytes()).hexdigest()
        owner = digest_owner.setdefault(digest, sample.group_key)
        dsu.union(owner, sample.group_key)
    return {key: dsu.find(key) for key in official_keys}


def build_groups(samples: Sequence[Sample], group_alias: Mapping[str, str], nc: int) -> List[GroupStats]:
    indices: Dict[str, List[int]] = defaultdict(list)
    counts: Dict[str, List[int]] = defaultdict(lambda: [0] * nc)
    for idx, sample in enumerate(samples):
        if sample.kind != "official":
            continue
        key = group_alias.get(sample.group_key, sample.group_key)
        indices[key].append(idx)
        for cls, value in enumerate(sample.class_counts):
            counts[key][cls] += value
    return [
        GroupStats(
            key=key,
            sample_indices=tuple(indices[key]),
            image_count=len(indices[key]),
            class_counts=tuple(counts[key]),
        )
        for key in sorted(indices)
    ]


def assignment_score(
    groups: Sequence[GroupStats], assignment: Mapping[str, str], ratios: Sequence[float]
) -> float:
    total_images = sum(group.image_count for group in groups)
    nc = len(groups[0].class_counts)
    total_classes = [sum(group.class_counts[c] for group in groups) for c in range(nc)]
    image_counts = {name: 0 for name in SPLIT_NAMES}
    class_counts = {name: [0] * nc for name in SPLIT_NAMES}
    for group in groups:
        split = assignment[group.key]
        image_counts[split] += group.image_count
        for cls, value in enumerate(group.class_counts):
            class_counts[split][cls] += value

    score = 0.0
    for split_idx, split in enumerate(SPLIT_NAMES):
        target_images = total_images * ratios[split_idx]
        score += 5.0 * ((image_counts[split] - target_images) / max(target_images, 1.0)) ** 2
        for cls, total in enumerate(total_classes):
            if total == 0:
                continue
            target = total * ratios[split_idx]
            score += ((class_counts[split][cls] - target) / max(target, 1.0)) ** 2 / nc

    validation_ratio = image_counts["validation"] / total_images
    if validation_ratio < ratios[2]:
        score += 1000.0 * (ratios[2] - validation_ratio) ** 2 + 100.0
    for split in ("calibration", "validation"):
        for cls, total in enumerate(total_classes):
            if total > 0 and class_counts[split][cls] == 0:
                score += 20.0
    return score


def greedy_assignment(groups: Sequence[GroupStats], ratios: Sequence[float], rng: random.Random) -> Dict[str, str]:
    nc = len(groups[0].class_counts)
    total_images = sum(group.image_count for group in groups)
    total_classes = [sum(group.class_counts[c] for group in groups) for c in range(nc)]
    targets_images = [total_images * ratio for ratio in ratios]
    targets_classes = [[total * ratio for total in total_classes] for ratio in ratios]
    current_images = [0] * len(SPLIT_NAMES)
    current_classes = [[0] * nc for _ in SPLIT_NAMES]

    def rarity(group: GroupStats) -> float:
        return sum(value / max(total_classes[cls], 1) for cls, value in enumerate(group.class_counts))

    ordered = sorted(groups, key=lambda group: (-(rarity(group) + rng.random() * 0.03), -group.image_count, group.key))
    assignment: Dict[str, str] = {}
    for group in ordered:
        options = []
        for split_idx, split in enumerate(SPLIT_NAMES):
            image_need = (targets_images[split_idx] - current_images[split_idx]) / max(targets_images[split_idx], 1.0)
            class_need = 0.0
            active = 0
            for cls, value in enumerate(group.class_counts):
                if value <= 0 or targets_classes[split_idx][cls] <= 0:
                    continue
                remaining = targets_classes[split_idx][cls] - current_classes[split_idx][cls]
                class_need += min(value, max(remaining, 0.0)) / targets_classes[split_idx][cls]
                active += 1
            if active:
                class_need /= active
            overshoot = max(
                0.0,
                (current_images[split_idx] + group.image_count - targets_images[split_idx])
                / max(targets_images[split_idx], 1.0),
            )
            benefit = 2.0 * image_need + 4.0 * class_need - 8.0 * overshoot + rng.random() * 1e-6
            options.append((benefit, split_idx, split))
        _, chosen_idx, chosen = max(options)
        assignment[group.key] = chosen
        current_images[chosen_idx] += group.image_count
        for cls, value in enumerate(group.class_counts):
            current_classes[chosen_idx][cls] += value
    return assignment


def ensure_minimum_validation(
    groups: Sequence[GroupStats], assignment: Dict[str, str], minimum_ratio: float
) -> Dict[str, str]:
    total = sum(group.image_count for group in groups)
    current = sum(group.image_count for group in groups if assignment[group.key] == "validation")
    if current / total >= minimum_ratio:
        return assignment
    candidates = sorted(
        (group for group in groups if assignment[group.key] != "validation"),
        key=lambda group: (group.image_count, group.key),
    )
    for group in candidates:
        assignment[group.key] = "validation"
        current += group.image_count
        if current / total >= minimum_ratio:
            break
    return assignment


def assign_groups(
    groups: Sequence[GroupStats], ratios: Sequence[float], seed: int, restarts: int = 512
) -> Dict[str, str]:
    if not groups:
        raise ValueError("no official groups were found")
    validate_ratios(ratios)
    if restarts <= 0:
        raise ValueError("restarts must be positive")
    best_assignment = None
    best_score = math.inf
    for restart in range(restarts):
        rng = random.Random(seed + restart * 1_000_003)
        candidate = greedy_assignment(groups, ratios, rng)
        candidate = ensure_minimum_validation(groups, candidate, ratios[2])
        score = assignment_score(groups, candidate, ratios)
        if score < best_score:
            best_assignment, best_score = candidate, score
    assert best_assignment is not None
    return best_assignment


def resolve_final_assignments(
    samples: Sequence[Sample], group_alias: Mapping[str, str], official_assignment: Mapping[str, str]
) -> Tuple[Dict[int, str], List[int], List[str]]:
    official_stems = {sample.image.stem: idx for idx, sample in enumerate(samples) if sample.kind == "official"}
    assignments: Dict[int, str] = {}
    excluded: List[int] = []
    unresolved: List[str] = []
    for idx, sample in enumerate(samples):
        if sample.kind == "official":
            key = group_alias.get(sample.group_key, sample.group_key)
            assignments[idx] = official_assignment[key]
        elif sample.kind == "external_train_only":
            assignments[idx] = "train"
        else:
            parent_idx = official_stems.get(sample.parent_stem or "")
            if parent_idx is None:
                unresolved.append(sample.image.name)
                continue
            parent_split = assignments.get(parent_idx)
            if parent_split is None:
                parent = samples[parent_idx]
                key = group_alias.get(parent.group_key, parent.group_key)
                parent_split = official_assignment[key]
            if parent_split == "train":
                assignments[idx] = "train"
            else:
                excluded.append(idx)
    return assignments, excluded, unresolved


def write_manifest(path: Path, images: Sequence[Path]) -> str:
    text = "".join(f"{image.as_posix()}\n" for image in sorted(images))
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def yaml_text(output_dir: Path, nc: int, names: Mapping[int, str]) -> str:
    lines = [
        "# Generated by scripts/build_report_split.py",
        f"path: {output_dir.as_posix()}",
        "train: train.txt",
        "val: calibration.txt",
        "test: validation.txt",
        "calibration: calibration.txt",
        "validation: validation.txt",
        "",
        f"nc: {nc}",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {names[idx]}" for idx in range(nc))
    return "\n".join(lines) + "\n"


def summarize_counts(samples: Sequence[Sample], indices: Iterable[int], nc: int) -> Dict[str, object]:
    selected = list(indices)
    class_counts = [0] * nc
    kind_counts = Counter()
    for idx in selected:
        sample = samples[idx]
        kind_counts[sample.kind] += 1
        for cls, value in enumerate(sample.class_counts):
            class_counts[cls] += value
    return {
        "images": len(selected),
        "objects": sum(class_counts),
        "class_counts": class_counts,
        "sample_kinds": dict(sorted(kind_counts.items())),
    }


def main() -> int:
    args = parse_args()
    ratios = (args.train_ratio, args.calibration_ratio, args.validation_ratio)
    validate_ratios(ratios)
    data_root = resolve_project_path(args.data_root)
    output_dir = resolve_project_path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_dir}; use --overwrite to refresh explicit files")
    output_dir.mkdir(parents=True, exist_ok=True)

    nc, names = load_dataset_config(data_root)
    samples = load_samples(data_root, nc, args.mar20_block_size)
    if args.skip_content_hash:
        group_alias = {sample.group_key: sample.group_key for sample in samples if sample.kind == "official"}
    else:
        group_alias = merge_exact_duplicate_groups(samples)
    groups = build_groups(samples, group_alias, nc)
    official_assignment = assign_groups(groups, ratios, args.seed, args.restarts)
    assignments, excluded, unresolved = resolve_final_assignments(samples, group_alias, official_assignment)
    if unresolved and not args.allow_unresolved_derived:
        preview = ", ".join(unresolved[:5])
        raise ValueError(
            f"{len(unresolved)} derived images could not be linked to an official source; "
            f"examples: {preview}; fix provenance or use --allow-unresolved-derived"
        )
    if unresolved:
        unresolved_set = set(unresolved)
        for idx, sample in enumerate(samples):
            if sample.image.name in unresolved_set:
                assignments[idx] = "train"

    split_indices = {
        split: [idx for idx, assigned in assignments.items() if assigned == split]
        for split in SPLIT_NAMES
    }
    official_indices = [idx for idx, sample in enumerate(samples) if sample.kind == "official"]
    official_total = len(official_indices)
    official_split_counts = {
        split: sum(1 for idx in split_indices[split] if samples[idx].kind == "official")
        for split in SPLIT_NAMES
    }
    if official_split_counts["validation"] / official_total + 1e-12 < args.validation_ratio:
        raise AssertionError("locked validation fell below the requested official-sample ratio")

    manifest_hashes = {}
    for split in SPLIT_NAMES:
        manifest_hashes[split] = write_manifest(
            output_dir / f"{split}.txt", [samples[idx].image for idx in split_indices[split]]
        )
    dataset_yaml = yaml_text(output_dir, nc, names)
    (output_dir / "dataset_report.yaml").write_text(dataset_yaml, encoding="utf-8")

    with (output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "original_split",
                "sample_kind",
                "scene_group",
                "assigned_split",
                "objects",
                "class_counts",
            ]
        )
        excluded_set = set(excluded)
        for idx, sample in enumerate(samples):
            assigned = "excluded_derived" if idx in excluded_set else assignments.get(idx, "unresolved")
            writer.writerow(
                [
                    sample.image.as_posix(),
                    sample.original_split,
                    sample.kind,
                    group_alias.get(sample.group_key, sample.group_key),
                    assigned,
                    sum(sample.class_counts),
                    json.dumps(sample.class_counts),
                ]
            )

    group_sets = {
        split: {
            group_alias.get(samples[idx].group_key, samples[idx].group_key)
            for idx in split_indices[split]
            if samples[idx].kind == "official"
        }
        for split in SPLIT_NAMES
    }
    overlap = {
        "train_calibration": sorted(group_sets["train"] & group_sets["calibration"]),
        "train_validation": sorted(group_sets["train"] & group_sets["validation"]),
        "calibration_validation": sorted(group_sets["calibration"] & group_sets["validation"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"scene leakage detected: {overlap}")

    summary = {
        "data_root": data_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "seed": args.seed,
        "requested_ratios": dict(zip(SPLIT_NAMES, ratios)),
        "mar20_block_size": args.mar20_block_size,
        "official_images": official_total,
        "official_split_counts": official_split_counts,
        "official_split_ratios": {
            split: official_split_counts[split] / official_total for split in SPLIT_NAMES
        },
        "external_train_only_images": sum(sample.kind == "external_train_only" for sample in samples),
        "derived_images": sum(sample.kind == "derived" for sample in samples),
        "excluded_derived_images": len(excluded),
        "unresolved_derived_images": unresolved,
        "scene_groups": len(groups),
        "scene_overlap": overlap,
        "splits": {
            split: summarize_counts(samples, split_indices[split], nc) for split in SPLIT_NAMES
        },
        "class_names": names,
        "manifest_sha256": manifest_hashes,
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"dataset_yaml={output_dir / 'dataset_report.yaml'}")
    print("REPORT_SPLIT=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
