"""Audit detection recall of targets clipped by mosaic seams / image edges in big images.

Big images are mosaics of source crops pasted at placement offsets. A ground-truth
box whose visible region crosses a placement boundary line (a seam, possibly the
outer canvas edge) is an "edge target". The official-like matching policy
(IoU 0.5 / FSC 0.35, exact class match) is applied, and recall is reported
separately for edge and non-edge targets.

Usage:
    python scripts/audit_edge_targets.py --split val_scene_grouped [--device 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from evaluate_big_images_sliding import predict_big_image, resolve_project_path
from evaluate_official_thresholds import (
    FSC_CLASS_ID,
    GroundTruth,
    box_iou_xyxy,
    label_path_for_image,
    load_ground_truths,
    load_yaml,
    resolve_data_root,
    resolve_split_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = Path("data_big_v1/manifests")
EDGE_TOLERANCE_PX = 2.0
BIG_SIZE = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val_scene_grouped")
    parser.add_argument("--output-dir", default="runs/detect/edge_audit")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--model",
        default="runs/detect/yolo26x_domain_finetune_lr1e4_e40/weights/best.pt",
        help="Model weights to audit with (e.g. the HRSC retrained model).",
    )
    return parser.parse_args()


def build_infer_args(device: str, model: str = "runs/detect/yolo26x_domain_finetune_lr1e4_e40/weights/best.pt") -> Namespace:
    """Namespace for predict_big_image, matching the final frozen solution."""
    args = Namespace()
    args.model = model
    args.data = "data_big_v1/dataset.yaml"
    args.ultralytics_root = "ultralytics-main"
    args.split = "val"
    args.imgsz = 1024
    args.tile_size = 800
    args.stride = 800
    args.pred_conf = 0.001
    args.pred_iou = 0.70
    args.global_iou = 0.70
    args.max_det = 300
    args.batch = 1
    args.device = device
    args.half = False
    return args


def required_iou_for(gt_cls: int) -> float:
    return 0.35 if gt_cls == FSC_CLASS_ID else 0.50


def load_seam_lines(manifest_path: Path) -> Tuple[List[float], List[float]]:
    """Collect all placement boundary x/y coordinates from the manifest."""
    xs: set = {0.0, float(BIG_SIZE)}
    ys: set = {0.0, float(BIG_SIZE)}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for placement in record.get("placements", []):
            ox, oy = placement["offset"]
            pw, ph = placement["pasted_size"]
            xs.add(float(ox))
            xs.add(float(ox + pw))
            ys.add(float(oy))
            ys.add(float(oy + ph))
    return sorted(xs), sorted(ys)


def is_edge_target(box: Tuple[float, float, float, float], seams_x: List[float], seams_y: List[float]) -> bool:
    x1, y1, x2, y2 = box
    for line in seams_x:
        if x1 - EDGE_TOLERANCE_PX < line < x2 + EDGE_TOLERANCE_PX:
            return True
    for line in seams_y:
        if y1 - EDGE_TOLERANCE_PX < line < y2 + EDGE_TOLERANCE_PX:
            return True
    return False


def match_predictions(
    gts: Sequence[GroundTruth], predictions: Sequence[GroundTruth]
) -> List[bool]:
    """Official-like matching (exact class + IoU 0.5/0.35); returns gt matched flags."""
    ordered = sorted(predictions, key=lambda p: (-p.conf, p.pred_index))
    matched = [False] * len(gts)
    for pred in ordered:
        best_iou = -1.0
        best_gt = -1
        for gt_index, gt in enumerate(gts):
            if matched[gt_index]:
                continue
            if pred.cls != gt.cls:
                continue
            iou = box_iou_xyxy(pred.box, gt.box)
            if iou + 1e-12 < required_iou_for(gt.cls):
                continue
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_gt >= 0:
            matched[best_gt] = True
    return matched


def main() -> int:
    args = parse_args()
    from evaluate_big_images_sliding import setup_ultralytics

    setup_ultralytics(resolve_project_path(Path("ultralytics-main")))
    from ultralytics import YOLO

    model = YOLO(str(resolve_project_path(Path(args.model))))
    data_yaml = resolve_project_path(Path("data_big_v1/dataset.yaml"))
    cfg = load_yaml(data_yaml)
    data_root = resolve_data_root(data_yaml, cfg)
    image_paths = resolve_split_sources(data_yaml, cfg, args.split)
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    infer_args = build_infer_args(args.device, args.model)
    out_root = resolve_project_path(Path(args.output_dir) / args.split)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_root = resolve_project_path(DEFAULT_MANIFEST_DIR)
    manifest_name = f"{args.split}_manifest.jsonl"

    rows: List[Dict] = []
    summary = {"split": args.split, "images": 0, "edge_gt": 0, "edge_tp": 0, "edge_fn": 0,
               "normal_gt": 0, "normal_tp": 0, "normal_fn": 0}

    for image_index, image_path in enumerate(image_paths):
        predictions, stats = predict_big_image(model, image_path, image_index, infer_args)
        label_path = label_path_for_image(image_path, data_root, args.split)
        width, height = stats["width"], stats["height"]
        # load_ground_truths already converts normalized YOLO labels to pixel xyxy.
        gts = load_ground_truths(label_path, width, height)
        seams_x, seams_y = load_seam_lines(manifest_root / manifest_name)
        edge_flags = [is_edge_target(gt.box, seams_x, seams_y) for gt in gts]
        matched = match_predictions(gts, predictions)

        for gt, flag, is_matched in zip(gts, edge_flags, matched):
            rows.append({
                "image": image_path.name,
                "cls": gt.cls,
                "is_edge": int(flag),
                "matched": int(is_matched),
                "box": [round(v, 1) for v in gt.box],
            })
            if flag:
                summary["edge_gt"] += 1
                if is_matched:
                    summary["edge_tp"] += 1
                else:
                    summary["edge_fn"] += 1
            else:
                summary["normal_gt"] += 1
                if is_matched:
                    summary["normal_tp"] += 1
                else:
                    summary["normal_fn"] += 1
        summary["images"] += 1
        print(f"image {image_index + 1}/{len(image_paths)} edge_fn={summary['edge_fn']}", flush=True)

    summary["edge_recall"] = summary["edge_tp"] / summary["edge_gt"] if summary["edge_gt"] else None
    summary["normal_recall"] = summary["normal_tp"] / summary["normal_gt"] if summary["normal_gt"] else None
    summary["edge_detail_csv"] = str((out_root / "edge_targets.csv").relative_to(PROJECT_ROOT))

    with (out_root / "edge_targets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
