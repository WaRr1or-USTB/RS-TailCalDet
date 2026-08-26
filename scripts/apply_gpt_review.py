"""Apply GPT review results to data_v2 labels.

Ship candidates confirmed as mislabeled get their class ID changed; FSC
candidates confirmed as missed get an added label row (prediction box converted
to normalized YOLO format). Everything else is left untouched.

Mapping is resolved ONCE up front (never rebuilt while files are being edited):
  - HM(0)/LQS(1): crop index == collection order over sorted label files.
  - QHS(2)/MS(3): crop index == sampled order (same rng/seed logic as
    scripts/extract_ship_samples.py: shuffle of sorted source-image keys, first
    box of each image, until the per-class limit).

Original label files are backed up to <output-dir>/backup_labels before editing,
and every change is logged to docs/data_review_log.csv.

Usage:
    python scripts/apply_gpt_review.py            # dry run, prints the plan
    python scripts/apply_gpt_review.py --apply    # execute the plan
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data_v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SHIP_CLASS_NAMES = {0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 24: "FSC"}
SAMPLE_CLASSES = (2, 3)  # classes whose crop index follows the sampled order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", default="docs/gpt_review_result.json")
    parser.add_argument("--ship-csv", default="docs/candidate_mislabel_ships.csv")
    parser.add_argument("--audit-dir", default="runs/detect/data_audit")
    parser.add_argument("--output-dir", default="runs/detect/review_applied")
    parser.add_argument("--log", default="docs/data_review_log.csv")
    parser.add_argument("--apply", action="store_true", help="execute changes (default is dry run)")
    return parser.parse_args()


def collection_order(split: str, cls: int) -> List[Tuple[Path, int, str]]:
    """(label_path, row_index, original_line) for every box of cls, in file/row order."""
    result: List[Tuple[Path, int, str]] = []
    label_dir = DATA / f"labels/{split}"
    for label_file in sorted(label_dir.glob("*.txt")):
        for row_index, line in enumerate(label_file.read_text(encoding="utf-8").splitlines()):
            parts = line.split()
            if len(parts) == 5 and int(parts[0]) == cls:
                result.append((label_file, row_index, line))
    return result


def sampled_order(split: str, cls: int, limit: int, seed: int = 2026) -> List[Tuple[Path, int, str]]:
    """Same sampling as extract_ship_samples: per-image first box, shuffled image keys."""
    by_image: Dict[str, List[Tuple[Path, int, str]]] = {}
    for label_file, row_index, line in collection_order(split, cls):
        by_image.setdefault(label_file.stem, []).append((label_file, row_index, line))
    rng = random.Random(seed)
    image_keys = sorted(by_image)
    rng.shuffle(image_keys)
    sampled: List[Tuple[Path, int, str]] = []
    for key in image_keys:
        if len(sampled) >= limit:
            break
        sampled.append(by_image[key][0])
    return sampled


def build_index(split: str, cls: int) -> List[Tuple[Path, int, str]]:
    if cls in SAMPLE_CLASSES:
        limits = {2: 8, 3: 8}
        return sampled_order(split, cls, limits[cls])
    return collection_order(split, cls)


def main() -> int:
    args = parse_args()
    review = json.loads((PROJECT_ROOT / args.review).read_text(encoding="utf-8"))
    ship_csv_rows = list(csv.DictReader((PROJECT_ROOT / args.ship_csv).open(encoding="utf-8")))

    plan: List[Dict] = []

    # ---- Ship mislabels: resolve exact (label_path, row) once, up front ----
    for case in review["ship_candidates"]:
        if case["verdict"] != "confirmed_mislabel":
            continue
        case_id = case["case_id"]
        csv_row = ship_csv_rows[int(case_id.split("-")[1]) - 1]
        crop_rel = csv_row["crop_image"]
        crop_name = Path(crop_rel).name
        index = int(crop_name.split("_")[0])
        cls = int(crop_rel.split("/")[-2].split("_")[1])
        new_cls = int(case["suggested_class"])

        entries = build_index("train", cls)
        if index >= len(entries):
            print(f"{case_id}: crop index {index} out of range for class {cls}", flush=True)
            continue
        label_path, row_index, old_line = entries[index]
        parts = old_line.split()
        parts[0] = str(new_cls)
        plan.append({
            "case_id": case_id,
            "kind": "modify_class",
            "label_path": label_path,
            "row": row_index,
            "old_line": old_line,
            "new_line": " ".join(parts),
            "evidence": case["evidence"],
        })

    # ---- FSC confirmed misses: append prediction box ----
    candidates = []
    csv_path = PROJECT_ROOT / args.audit_dir / "train" / "candidates.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            candidates = list(csv.DictReader(f))
    candidates_by_index = {}
    for cand in candidates:
        if cand["type"] != "candidate_miss":
            continue
        try:
            candidates_by_index[int(Path(cand["crop"]).name.split("_")[0])] = cand
        except (ValueError, IndexError):
            continue
    # FSC crop indexes from the task book: FSC-03=0051(AGZ), FSC-09=0053(TG-N24.21), FSC-10=0052(AGZ)
    fsc_map = {"FSC-03": 51, "FSC-09": 53, "FSC-10": 52}
    from PIL import Image

    for case in review["fsc_candidates"]:
        if case["verdict"] != "confirmed_miss":
            continue
        case_id = case["case_id"]
        index = fsc_map.get(case_id)
        cand = candidates_by_index.get(index) if index is not None else None
        if cand is None:
            print(f"{case_id}: candidate index {index} not found", flush=True)
            continue
        stem = Path(cand["image"]).stem
        box = [float(v) for v in json.loads(cand["box"])]
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            candidate = DATA / "images/train" / (stem + suffix)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            print(f"{case_id}: image not found for {stem}", flush=True)
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        cx = (box[0] + box[2]) / 2 / width
        cy = (box[1] + box[3]) / 2 / height
        w = (box[2] - box[0]) / width
        h = (box[3] - box[1]) / height
        plan.append({
            "case_id": case_id,
            "kind": "add_missed_target",
            "label_path": DATA / "labels/train" / f"{stem}.txt",
            "row": -1,
            "old_line": "-",
            "new_line": f"24 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}",
            "evidence": case["evidence"],
        })

    # ---- Show plan ----
    print("=== PLAN ===", flush=True)
    for item in plan:
        label_name = item["label_path"].name
        if item["kind"] == "modify_class":
            print(f"{item['case_id']}: MODIFY {label_name} row {item['row']}: {item['old_line']} -> {item['new_line']}", flush=True)
        else:
            print(f"{item['case_id']}: APPEND {label_name}: {item['new_line']}", flush=True)
    if not args.apply:
        print("\nDry run — no changes made. Re-run with --apply to execute.", flush=True)
        return 0

    # ---- Execute: backup each touched file once, then apply ----
    out_root = PROJECT_ROOT / args.output_dir
    backup_dir = out_root / "backup_labels"
    backup_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    touched: set = set()
    edits: Dict[Path, List[Tuple[int, str]]] = {}
    for item in plan:
        touched.add(item["label_path"])
        edits.setdefault(item["label_path"], []).append((item["row"], item["new_line"]))

    for label_path in sorted(touched):
        shutil.copy2(label_path, backup_dir / label_path.name)

    log_rows: List[Dict] = []
    for item in plan:
        label_path = item["label_path"]
        if item["kind"] == "modify_class":
            lines = label_path.read_text(encoding="utf-8").splitlines()
            lines[item["row"]] = item["new_line"]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:  # add_missed_target
            with label_path.open("a", encoding="utf-8") as f:
                f.write(item["new_line"] + "\n")
        log_rows.append({
            "case_id": item["case_id"],
            "action": item["kind"],
            "split": "train",
            "label_file": label_path.name,
            "row": item["row"],
            "before": item["old_line"],
            "after": item["new_line"],
            "reviewer": "GPT",
            "evidence": item["evidence"],
        })

    log_path = PROJECT_ROOT / args.log
    append_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        if append_header:
            writer.writeheader()
        writer.writerows(log_rows)

    summary = {
        "changes": len(log_rows),
        "modified_labels": len(touched),
        "backup_dir": str(backup_dir.relative_to(PROJECT_ROOT)),
        "log": str(log_path.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
