"""Build the official annotation-feedback CSV and refresh the team review list.

Sources:
  - docs/data_review_log.csv           : 9 confirmed fixes (high confidence)
  - docs/spotcheck_confirmed.json      : manually verified spot-checks (high confidence)
  - runs/detect/data_audit_regen/*/candidates.csv : fresh candidates (with box coords)

Outputs:
  - docs/annotation_feedback_official.csv  (image, problem_type, classes, box, evidence, confidence)
  - docs/data_review_pending.csv           (team review list, now with box coordinates)

Usage:
    python scripts/build_feedback_file.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data_v2"
AUDIT_DIR = PROJECT_ROOT / "runs/detect/data_audit_regen"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 4: "SU-35", 5: "C-130", 6: "C-17", 7: "C-5",
    8: "F-16", 9: "TU-160", 10: "E-3", 11: "B-52", 12: "P-3C", 13: "B-1B", 14: "E-8",
    15: "TU-22", 16: "F-15", 17: "KC-135", 18: "F-22", 19: "FA-18", 20: "TU-95",
    21: "KC-10", 22: "SU-34", 23: "SU-24", 24: "FSC",
}
SHIP_OR_VEHICLE = frozenset(range(0, 4)) | {24}


def image_path_for(stem: str, split: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = DATA / f"images/{split}" / (stem + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no image for {stem} ({split})")


def image_size(stem: str, split: str) -> Tuple[int, int]:
    with Image.open(image_path_for(stem, split)) as im:
        return im.size


def yolo_line_to_pixel_box(line: str, width: int, height: int) -> str:
    parts = line.split()
    cx, cy, w, h = (float(p) for p in parts[1:5])
    x1 = round((cx - w / 2) * width, 1)
    y1 = round((cy - h / 2) * height, 1)
    x2 = round((cx + w / 2) * width, 1)
    y2 = round((cy + h / 2) * height, 1)
    return f"[{x1}, {y1}, {x2}, {y2}]"


def read_candidates(split: str) -> List[Dict]:
    path = AUDIT_DIR / split / "candidates.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_pixel_box(raw: str) -> List[float]:
    return [float(v) for v in raw.strip("[]").split(",")]


def main() -> int:
    rows: List[Dict] = []

    # ---- Layer 1: confirmed fixes from review log ----
    log_path = PROJECT_ROOT / "docs/data_review_log.csv"
    with log_path.open(encoding="utf-8") as f:
        log_rows = list(csv.DictReader(f))
    for entry in log_rows:
        stem = Path(entry["label_file"]).stem
        split = entry["split"]
        width, height = image_size(stem, split)
        if entry["action"] == "modify_class":
            old_cls = int(entry["before"].split()[0])
            new_cls = int(entry["after"].split()[0])
            rows.append({
                "image": f"{stem}.jpg",
                "split": split,
                "problem_type": "mislabel",
                "annotated_class": CLASS_NAMES[old_cls],
                "suggested_class": CLASS_NAMES[new_cls],
                "box": yolo_line_to_pixel_box(entry["after"], width, height),
                "confidence": "high",
                "evidence": entry["evidence"],
            })
        else:  # add_missed_target
            rows.append({
                "image": f"{stem}.jpg",
                "split": split,
                "problem_type": "miss",
                "annotated_class": "",
                "suggested_class": CLASS_NAMES[int(entry["after"].split()[0])],
                "box": yolo_line_to_pixel_box(entry["after"], width, height),
                "confidence": "high",
                "evidence": entry["evidence"],
            })

    # ---- Layer 2: spot-check confirmed cases ----
    spot_path = PROJECT_ROOT / "docs/spotcheck_confirmed.json"
    if spot_path.exists():
        spot = json.loads(spot_path.read_text(encoding="utf-8"))
        for case in spot:
            rows.append({
                "image": f"{case['image_stem']}.jpg",
                "split": case["split"],
                "problem_type": case["problem_type"],
                "annotated_class": case.get("annotated_class", ""),
                "suggested_class": case.get("suggested_class", ""),
                "box": case["box"],
                "confidence": "high",
                "evidence": case["evidence"],
            })

    # Boxes of already-confirmed fixes: exclude overlapping candidates from review.
    confirmed_by_image: Dict[str, List[List[float]]] = {}
    for r in rows:
        if r["confidence"] == "high":
            confirmed_by_image.setdefault(r["image"], []).append(parse_pixel_box(r["box"]))

    # ---- Layer 3: fresh candidates (ship/vehicle only, for team review) ----
    pending: List[Dict] = []
    for split in ("train", "val"):
        for cand in read_candidates(split):
            pred_cls = int(cand["cls_pred"])
            gt_cls = int(cand["cls_gt"]) if cand["cls_gt"] != "-1" else None
            if not (pred_cls in SHIP_OR_VEHICLE or (gt_cls is not None and gt_cls in SHIP_OR_VEHICLE)):
                continue
            cand_box = parse_pixel_box(cand["box"])
            if any(
                box_iou(cand_box, fixed) > 0.5
                for fixed in confirmed_by_image.get(cand["image"], [])
            ):
                continue  # already confirmed & fixed
            if cand["type"] == "candidate_miss":
                evidence = f"模型以置信度 {cand['conf']} 检出 {CLASS_NAMES[pred_cls]} 目标但无标注，疑似漏标"
            else:
                evidence = (
                    f"标注为 {CLASS_NAMES[gt_cls]}，模型预测 {CLASS_NAMES[pred_cls]}"
                    f"（置信度 {cand['conf']}），疑似错标"
                )
            rows.append({
                "image": cand["image"],
                "split": split,
                "problem_type": "miss" if cand["type"] == "candidate_miss" else "mislabel",
                "annotated_class": CLASS_NAMES[gt_cls] if gt_cls is not None else "",
                "suggested_class": CLASS_NAMES[pred_cls],
                "box": cand["box"],
                "confidence": "candidate",
                "evidence": evidence,
            })
            pending.append({
                "priority": "HIGH" if (pred_cls in SHIP_OR_VEHICLE or (gt_cls is not None and gt_cls in SHIP_OR_VEHICLE)) else "MED",
                "split": split,
                "type": cand["type"],
                "cls_gt": CLASS_NAMES[gt_cls] if gt_cls is not None else "-",
                "cls_pred": CLASS_NAMES[pred_cls],
                "conf": cand["conf"],
                "iou": cand["iou"],
                "image": cand["image"],
                "box": cand["box"],
            })

    # ---- Write feedback file ----
    fieldnames = ["image", "split", "problem_type", "annotated_class", "suggested_class", "box", "confidence", "evidence"]
    out_path = PROJECT_ROOT / "docs/annotation_feedback_official.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Write team review list ----
    pending_path = PROJECT_ROOT / "docs/data_review_pending.csv"
    with pending_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pending[0].keys()))
        writer.writeheader()
        writer.writerows(pending)

    high = sum(1 for r in rows if r["confidence"] == "high")
    cand = sum(1 for r in rows if r["confidence"] == "candidate")
    print(json.dumps({
        "feedback_rows": len(rows),
        "high_confidence": high,
        "candidate": cand,
        "pending_rows": len(pending),
        "feedback_file": str(out_path.relative_to(PROJECT_ROOT)),
        "pending_file": str(pending_path.relative_to(PROJECT_ROOT)),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
