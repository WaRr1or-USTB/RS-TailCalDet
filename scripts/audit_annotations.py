"""Audit data_v2 annotations with the final model: find candidate missed/mislabeled targets.

For every image in a split, run single-image inference and compare with ground truth:
  - matched        : prediction overlaps a GT of the same class -> fine
  - candidate_mislabel : prediction overlaps a GT (IoU >= 0.5, FSC 0.35) but the classes differ
  - candidate_miss : confident prediction (conf >= --miss-conf) with no overlapping GT (IoU < 0.3)

Crops of all candidates are saved for manual/GPT review.

Usage:
    python scripts/audit_annotations.py --split val
    python scripts/audit_annotations.py --split train --miss-conf 0.35
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from evaluate_official_thresholds import (
    FSC_CLASS_ID,
    GroundTruth,
    Prediction,
    box_iou_xyxy,
    label_path_for_image,
    load_ground_truths,
    load_yaml,
    resolve_data_root,
    resolve_split_sources,
    setup_ultralytics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("runs/detect/yolo26x_domain_finetune_lr1e4_e40/weights/best.pt")
DEFAULT_DATA = Path("data_v2/dataset.yaml")
DEFAULT_ULTRALYTICS_ROOT = Path("ultralytics-main")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SHIP_IDS = frozenset(range(0, 4))
VEHICLE_IDS = frozenset({FSC_CLASS_ID})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--ultralytics-root", default=str(DEFAULT_ULTRALYTICS_ROOT))
    parser.add_argument("--split", default="val", help="train or val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--pred-conf", type=float, default=0.10, help="inference conf threshold")
    parser.add_argument("--pred-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--miss-conf", type=float, default=0.35, help="min conf of candidate missed targets")
    parser.add_argument("--output-dir", default="runs/detect/data_audit")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--no-crops", action="store_true", help="Skip saving review crops (only CSV output).")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def required_iou_for(gt_cls: int) -> float:
    return 0.35 if gt_cls == FSC_CLASS_ID else 0.50


def iou_to_gt(pred_box, gts: Sequence[GroundTruth]) -> Tuple[float, Optional[int], int]:
    """Return (best_iou, best_gt_index, best_gt_class) over unthresholded IoU."""
    best_iou = -1.0
    best_index: Optional[int] = None
    best_cls = -1
    for index, gt in enumerate(gts):
        iou = box_iou_xyxy(pred_box, gt.box)
        if iou > best_iou:
            best_iou = iou
            best_index = index
            best_cls = gt.cls
    return best_iou, best_index, best_cls


def main() -> int:
    args = parse_args()
    model_path = resolve_project_path(args.model)
    data_yaml = resolve_project_path(args.data)
    ultralytics_root = resolve_project_path(args.ultralytics_root)
    out_root = resolve_project_path(Path(args.output_dir) / f"{args.split}")

    setup_ultralytics(ultralytics_root)
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    cfg = load_yaml(data_yaml)
    data_root = resolve_data_root(data_yaml, cfg)
    image_paths = resolve_split_sources(data_yaml, cfg, args.split)

    out_root.mkdir(parents=True, exist_ok=True)
    crops_miss = out_root / "crops_miss"
    crops_mislbl = out_root / "crops_mislbl"
    if not args.no_crops:
        crops_miss.mkdir(exist_ok=True)
        crops_mislbl.mkdir(exist_ok=True)

    rows: List[Dict] = []
    matched = 0
    missed_gt = 0
    processed = 0
    for start in range(0, len(image_paths), args.batch):
        batch_paths = image_paths[start : start + args.batch]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.imgsz,
            conf=args.pred_conf,
            iou=args.pred_iou,
            max_det=args.max_det,
            batch=len(batch_paths),
            device=args.device,
            verbose=False,
            save=False,
        )
        for image_path, result in zip(batch_paths, results):
            with Image.open(image_path) as im:
                width, height = im.size
            gts = load_ground_truths(label_path_for_image(image_path, data_root, args.split), width, height)
            boxes = result.boxes
            preds: List[Prediction] = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                confs = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                for box, conf, cls in zip(xyxy, confs, classes):
                    preds.append(Prediction(0, 0, int(cls), float(conf), tuple(float(v) for v in box)))

            # GT bookkeeping per image
            gt_matched = [False] * len(gts)
            for pred in preds:
                best_iou, best_index, best_cls = iou_to_gt(pred.box, gts)
                if best_index is not None and best_iou >= required_iou_for(best_cls):
                    gt_matched[best_index] = True
                    if pred.cls == best_cls:
                        matched += 1
                    else:
                        row = {
                            "type": "candidate_mislabel",
                            "image": image_path.name,
                            "cls_gt": best_cls,
                            "cls_pred": pred.cls,
                            "conf": round(pred.conf, 4),
                            "iou": round(best_iou, 4),
                            "box": [round(v, 1) for v in pred.box],
                            "crop": "",
                        }
                        rows.append(row)
                        if not args.no_crops:
                            crop = crop_box(image_path, gts[best_index].box)
                            target = crops_mislbl / f"{len(rows):04d}_{image_path.stem}.png"
                            crop.save(target)
                            rows[-1]["crop"] = str(target.relative_to(PROJECT_ROOT))
                elif best_iou < 0.30 and pred.conf >= args.miss_conf:
                    row = {
                        "type": "candidate_miss",
                        "image": image_path.name,
                        "cls_gt": -1,
                        "cls_pred": pred.cls,
                        "conf": round(pred.conf, 4),
                        "iou": round(best_iou, 4),
                        "box": [round(v, 1) for v in pred.box],
                        "crop": "",
                    }
                    rows.append(row)
                    if not args.no_crops:
                        crop = crop_box(image_path, pred.box)
                        target = crops_miss / f"{len(rows):04d}_{image_path.stem}.png"
                        crop.save(target)
                        rows[-1]["crop"] = str(target.relative_to(PROJECT_ROOT))

            missed_gt += sum(1 for m in gt_matched if not m)
            processed += 1
            if args.print_every and processed % args.print_every == 0:
                print(f"processed={processed}/{len(image_paths)} candidates={len(rows)}", flush=True)

    rows.sort(key=lambda r: (-r["conf"], r["type"], r["image"]))

    csv_path = out_root / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "images": processed,
        "gt_total": missed_gt + sum(1 for r in rows if r["type"] == "candidate_mislabel"),
        "candidate_miss": sum(1 for r in rows if r["type"] == "candidate_miss"),
        "candidate_mislabel": sum(1 for r in rows if r["type"] == "candidate_mislabel"),
        "candidate_miss_ship": sum(1 for r in rows if r["type"] == "candidate_miss" and r["cls_pred"] in SHIP_IDS),
        "candidate_miss_vehicle": sum(1 for r in rows if r["type"] == "candidate_miss" and r["cls_pred"] in VEHICLE_IDS),
        "candidates_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "note": "train split results include model memorization effects; val split results are cleaner evidence.",
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def crop_box(image_path: Path, box: Sequence[float], pad: float = 0.08) -> Image.Image:
    with Image.open(image_path) as im:
        width, height = im.size
        x1 = max(0, int(box[0] - (box[2] - box[0]) * pad))
        y1 = max(0, int(box[1] - (box[3] - box[1]) * pad))
        x2 = min(width, int(box[2] + (box[2] - box[0]) * pad))
        y2 = min(height, int(box[3] + (box[3] - box[1]) * pad))
        return im.convert("RGB").crop((x1, y1, x2, y2))


if __name__ == "__main__":
    sys.exit(main())
