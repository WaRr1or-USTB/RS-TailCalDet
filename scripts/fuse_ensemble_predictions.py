"""Ensemble fusion of two models on big images (official rules allow multi-model fusion).

Strategy: run both models with the frozen sliding-window config; for each class,
a prediction from model A and a prediction from model B with IoU >= --fuse-iou are
the same object -> keep the higher-confidence one. Predictions seen by only one
model are dropped (suppresses false positives). Then run the official matching
protocol and dump per-GT matched/size rows for the official-metric script.

Usage:
    python scripts/fuse_ensemble_predictions.py \
        --model-a runs/detect/yolo26x_hrsc_ft_e40/weights/best.pt \
        --model-b runs/detect/yolo26x_domain_finetune_lr1e4_e40/weights/best.pt \
        --data data_big_v1/dataset.yaml --split val_scene_grouped --device 0 \
        --output-dir runs/detect/ensemble_scene_grouped
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from evaluate_big_images_sliding import (
    Candidate,
    Prediction,
    build_protocol_metrics,
    class_aware_nms,
    load_class_thresholds,
    predict_big_image,
    resolve_project_path,
    setup_ultralytics,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, help="Primary model (HRSC retrained).")
    parser.add_argument("--model-b", required=True, help="Secondary model (baseline).")
    parser.add_argument("--data", default="data_big_v1/dataset.yaml")
    parser.add_argument("--split", default="val_scene_grouped")
    parser.add_argument("--output-dir", default="runs/detect/ensemble_scene_grouped")
    parser.add_argument("--device", default="0")
    parser.add_argument("--fuse-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=800)
    parser.add_argument("--stride", type=int, default=800)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--pred-iou", type=float, default=0.70)
    parser.add_argument("--global-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--max-fdr", type=float, default=0.20)
    parser.add_argument(
        "--class-thresholds",
        default="",
        help="Per-class confidence thresholds JSON applied after fusion (same as single-model eval).",
    )
    return parser.parse_args()


def required_iou_for(gt_cls: int) -> float:
    return 0.35 if gt_cls == FSC_CLASS_ID else 0.50


def fuse(preds_a: Sequence[Prediction], preds_b: Sequence[Prediction], iou_threshold: float) -> List[Prediction]:
    """Keep same-object detections agreed by both models (per class, IoU matched)."""
    fused: List[Prediction] = []
    used_b: set = set()
    ordered_a = sorted(preds_a, key=lambda p: -p.conf)
    for pa in ordered_a:
        best_iou = -1.0
        best_index = -1
        for i, pb in enumerate(preds_b):
            if i in used_b or pb.cls != pa.cls:
                continue
            iou = box_iou_xyxy(pa.box, pb.box)
            if iou > best_iou:
                best_iou = iou
                best_index = i
        if best_index >= 0 and best_iou >= iou_threshold:
            used_b.add(best_index)
            pb = preds_b[best_index]
            winner = pa if pa.conf >= pb.conf else pb
            fused.append(winner)
    return fused


def match_gts(
    gts: Sequence[GroundTruth], predictions: Sequence[Prediction]
) -> List[bool]:
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
    setup_ultralytics(resolve_project_path(Path("ultralytics-main")))
    from ultralytics import YOLO

    model_a = YOLO(str(resolve_project_path(Path(args.model_a))))
    model_b = YOLO(str(resolve_project_path(Path(args.model_b))))

    data_yaml = resolve_project_path(Path(args.data))
    cfg = load_yaml(data_yaml)
    data_root = resolve_data_root(data_yaml, cfg)
    image_paths = resolve_split_sources(data_yaml, cfg, args.split)

    infer = argparse.Namespace(
        imgsz=args.imgsz, tile_size=args.tile_size, stride=args.stride,
        pred_conf=args.pred_conf, pred_iou=args.pred_iou, global_iou=args.global_iou,
        max_det=args.max_det, batch=1, device=args.device, half=False, tta=False,
    )

    out_root = resolve_project_path(Path(args.output_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    gt_by_image: List[List[GroundTruth]] = []
    all_preds: List[Prediction] = []
    edge_rows: List[Dict] = []
    for image_index, image_path in enumerate(image_paths):
        preds_a, stats_a = predict_big_image(model_a, image_path, image_index, infer)
        preds_b, stats_b = predict_big_image(model_b, image_path, image_index, infer)
        print(
            f"image {image_index + 1}/{len(image_paths)}: A={len(preds_a)} B={len(preds_b)}",
            flush=True,
        )
        fused = fuse(preds_a, preds_b, args.fuse_iou)
        # Merge near-duplicate detections the two models did not IoU-match (box drift).
        fused = class_aware_nms(
            [Candidate(cls=p.cls, conf=p.conf, box=p.box) for p in fused],
            args.global_iou,
        )
        fused = [
            Prediction(image_index, i, c.cls, c.conf, c.box)
            for i, c in enumerate(fused)
        ]
        for pred in fused:
            all_preds.append(
                Prediction(image_index, len(all_preds), pred.cls, pred.conf, pred.box)
            )

        label_path = label_path_for_image(image_path, data_root, args.split)
        width, height = stats_a["width"], stats_a["height"]
        gts = load_ground_truths(label_path, width, height)
        gt_by_image.append(gts)
        matched = match_gts(gts, fused)
        for gt, is_matched in zip(gts, matched):
            x1, y1, x2, y2 = gt.box
            edge_rows.append({
                "image": image_path.name,
                "cls": gt.cls,
                "is_edge": 0,
                "matched": int(is_matched),
                "box": json.dumps([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]),
            })

    class_names = {cls: str(cls) for cls in range(25)}
    if args.class_thresholds:
        thresholds = load_class_thresholds(resolve_project_path(Path(args.class_thresholds)))
        all_preds = [p for p in all_preds if p.conf + 1e-12 >= thresholds.get(p.cls, 0.0)]
        print(f"applied class thresholds, kept {len(all_preds)} fused predictions", flush=True)
    protocol = build_protocol_metrics(
        gt_by_image=gt_by_image,
        predictions=all_preds,
        threshold=0.001,
        target_recall=args.target_recall,
        max_fdr=args.max_fdr,
        class_names=class_names,
    )
    (out_root / "protocol_metrics.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_root / "edge_targets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()))
        writer.writeheader()
        writer.writerows(edge_rows)

    strict = protocol["strict_25_subclass"]["overall"]
    print(
        f"strict: tp={strict['tp']} fp={strict['fp']} fn={strict['fn']} "
        f"recall={strict['recall']:.4f} fdr={strict['fdr']:.4f}",
        flush=True,
    )
    for cat, m in protocol["official_by_category"]["categories"].items():
        print(f"{cat}: recall={m['recall']:.4f} fdr={m['fdr']:.4f}", flush=True)
    print(f"outputs -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
