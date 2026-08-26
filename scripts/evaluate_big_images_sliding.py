from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from evaluate_official_thresholds import (
    FSC_CLASS_ID,
    GroundTruth,
    Prediction,
    box_iou_xyxy,
    evaluate_thresholds,
    label_path_for_image,
    load_ground_truths,
    load_yaml,
    make_thresholds,
    resolve_data_root,
    resolve_split_sources,
    setup_ultralytics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt")
DEFAULT_DATA = Path("data_big_report_validation_s2026/dataset.yaml")
DEFAULT_ULTRALYTICS_ROOT = Path("ultralytics-main")
DEFAULT_CLASS_THRESHOLDS = Path("runs/detect/report_phase1_thresholds_r925_s2026/class_thresholds.json")
DEFAULT_OUTPUT_DIR = Path("runs/detect/report_recheck_val_scene_grouped")
BROAD_CATEGORIES = {
    "ship": frozenset(range(0, 4)),
    "aircraft": frozenset(range(4, 24)),
    "vehicle": frozenset({FSC_CLASS_ID}),
}
Image.MAX_IMAGE_PIXELS = None

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True)
class Candidate:
    cls: int
    conf: float
    box: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sliding-window inference and official-like threshold evaluation for pseudo big images."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--ultralytics-root", default=str(DEFAULT_ULTRALYTICS_ROOT))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=800)
    parser.add_argument("--stride", type=int, default=800)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--pred-iou", type=float, default=0.70)
    parser.add_argument("--global-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per tile before global NMS.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported by the device.")
    parser.add_argument(
        "--broad-fusion-gain",
        type=float,
        default=None,
        help="Override HierarchicalDetect broad-to-fine logit fusion gain; omitted keeps the checkpoint value.",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=None,
        help="Lowest swept confidence threshold. Defaults to --pred-conf.",
    )
    parser.add_argument("--threshold-stop", type=float, default=0.950)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--max-fdr", type=float, default=0.20)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--max-images", type=int, default=0, help="Evaluate only the first N big images. 0 means all.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--csv-name", default="threshold_metrics.csv")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--per-image-csv-name", default="per_image_times.csv")
    parser.add_argument("--protocols-name", default="protocol_metrics.json")
    parser.add_argument("--v16-csv-name", default="v16_category_metrics.csv")
    parser.add_argument(
        "--class-thresholds",
        default=str(DEFAULT_CLASS_THRESHOLDS),
        help="Optional class-threshold JSON from optimize_class_thresholds.py.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable flip test-time augmentation in model.predict (allowed by official rules).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print image/tile counts without loading the model.")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def require_numpy() -> Any:
    if np is None:
        raise RuntimeError("numpy is required for sliding-window inference. Install it in the inference environment.")
    return np


def load_class_thresholds(path: Path) -> Dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_thresholds = raw.get("thresholds", raw)
    if not isinstance(raw_thresholds, dict):
        raise ValueError(f"class thresholds must be a JSON object: {path}")

    thresholds: Dict[int, float] = {}
    for raw_cls, raw_value in raw_thresholds.items():
        value = raw_value.get("threshold") if isinstance(raw_value, dict) else raw_value
        if value is None:
            continue
        cls = int(raw_cls)
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"class {cls} threshold must be between 0 and 1, got {threshold}")
        thresholds[cls] = threshold
    return thresholds


def apply_class_thresholds(predictions: Sequence[Prediction], thresholds: Dict[int, float]) -> List[Prediction]:
    if not thresholds:
        return list(predictions)
    return [prediction for prediction in predictions if prediction.conf + 1e-12 >= thresholds.get(prediction.cls, 0.0)]


def validate_args(args: argparse.Namespace) -> None:
    if args.threshold_start is None:
        args.threshold_start = args.pred_conf
    if args.threshold_start + 1e-12 < args.pred_conf:
        raise ValueError("--threshold-start cannot be lower than --pred-conf.")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.batch <= 0:
        raise ValueError("--batch must be positive.")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive.")
    for name in ("pred_conf", "pred_iou", "global_iou", "target_recall", "max_fdr"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.broad_fusion_gain is not None and args.broad_fusion_gain < 0:
        raise ValueError("--broad-fusion-gain must be non-negative.")


def tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_positions(width: int, height: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    xs = tile_starts(width, tile_size, stride)
    ys = tile_starts(height, tile_size, stride)
    return [(x, y) for y in ys for x in xs]


def box_iou_vector(box: Any, boxes: Any) -> Any:
    numpy = require_numpy()
    x1 = numpy.maximum(box[0], boxes[:, 0])
    y1 = numpy.maximum(box[1], boxes[:, 1])
    x2 = numpy.minimum(box[2], boxes[:, 2])
    y2 = numpy.minimum(box[3], boxes[:, 3])
    inter = numpy.maximum(0.0, x2 - x1) * numpy.maximum(0.0, y2 - y1)
    area_box = numpy.maximum(0.0, box[2] - box[0]) * numpy.maximum(0.0, box[3] - box[1])
    area_boxes = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return numpy.divide(inter, union, out=numpy.zeros_like(inter), where=union > 0.0)


def nms_numpy(boxes: Any, scores: Any, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = box_iou_vector(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    return keep


def class_aware_nms(candidates: Sequence[Candidate], iou_threshold: float) -> List[Candidate]:
    if not candidates:
        return []
    numpy = require_numpy()
    boxes = numpy.asarray([candidate.box for candidate in candidates], dtype=numpy.float32)
    scores = numpy.asarray([candidate.conf for candidate in candidates], dtype=numpy.float32)
    classes = numpy.asarray([candidate.cls for candidate in candidates], dtype=numpy.int32)
    keep_indices: List[int] = []
    for cls in sorted(numpy.unique(classes).tolist()):
        cls_indices = numpy.where(classes == cls)[0]
        cls_keep_local = nms_numpy(boxes[cls_indices], scores[cls_indices], iou_threshold)
        keep_indices.extend(int(cls_indices[idx]) for idx in cls_keep_local)
    keep_indices.sort(key=lambda idx: (-candidates[idx].conf, candidates[idx].cls, candidates[idx].box))
    return [candidates[idx] for idx in keep_indices]


def maybe_cuda_synchronize(device: str) -> None:
    if str(device).lower() == "cpu":
        return
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clip_box(box: Sequence[float], offset_x: int, offset_y: int, width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = min(max(float(box[0]) + offset_x, 0.0), float(width))
    y1 = min(max(float(box[1]) + offset_y, 0.0), float(height))
    x2 = min(max(float(box[2]) + offset_x, 0.0), float(width))
    y2 = min(max(float(box[3]) + offset_y, 0.0), float(height))
    return x1, y1, x2, y2


def collect_tile_predictions(
    result,
    offset_x: int,
    offset_y: int,
    tile_width: int,
    width: int,
    height: int,
    raw_candidates: List[Candidate],
    flipped: bool = False,
) -> None:
    """Append a tile's predictions to raw_candidates, optionally un-flipping x coords."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    for box, conf, cls in zip(xyxy, confs, classes):
        if flipped:
            x1, y1, x2, y2 = box
            box = (tile_width - x2, y1, tile_width - x1, y2)  # undo horizontal flip
        full_box = clip_box(box, offset_x, offset_y, width, height)
        if full_box[2] <= full_box[0] or full_box[3] <= full_box[1]:
            continue
        raw_candidates.append(Candidate(cls=int(cls), conf=float(conf), box=full_box))


def predict_big_image(model, image_path: Path, image_index: int, args: argparse.Namespace) -> Tuple[List[Prediction], Dict]:
    numpy = require_numpy()
    read_started = time.perf_counter()
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    data_read_seconds = time.perf_counter() - read_started

    width, height = image.size
    tta_enabled = getattr(args, "tta", False)
    maybe_cuda_synchronize(args.device)
    started = time.perf_counter()
    raw_candidates: List[Candidate] = []
    try:
        positions = tile_positions(width, height, args.tile_size, args.stride)
        for start in range(0, len(positions), args.batch):
            batch_positions = positions[start : start + args.batch]
            batch_tiles = []
            for offset_x, offset_y in batch_positions:
                crop = image.crop((offset_x, offset_y, offset_x + args.tile_size, offset_y + args.tile_size))
                batch_tiles.append(crop)

            # Original orientation
            results = model.predict(
                source=batch_tiles,
                imgsz=args.imgsz,
                conf=args.pred_conf,
                iou=args.pred_iou,
                max_det=args.max_det,
                batch=len(batch_tiles),
                device=args.device,
                half=args.half,
                verbose=False,
                save=False,
            )
            for result, (offset_x, offset_y), tile in zip(results, batch_positions, batch_tiles):
                collect_tile_predictions(result, offset_x, offset_y, tile.width, width, height, raw_candidates)

            # TTA: horizontal-flip inference (YOLO26 end2end does not support built-in augment)
            if tta_enabled:
                flipped_tiles = [tile.transpose(Image.FLIP_LEFT_RIGHT) for tile in batch_tiles]
                results_f = model.predict(
                    source=flipped_tiles,
                    imgsz=args.imgsz,
                    conf=args.pred_conf,
                    iou=args.pred_iou,
                    max_det=args.max_det,
                    batch=len(flipped_tiles),
                    device=args.device,
                    half=args.half,
                    verbose=False,
                    save=False,
                )
                for result, (offset_x, offset_y), tile in zip(results_f, batch_positions, batch_tiles):
                    collect_tile_predictions(
                        result, offset_x, offset_y, tile.width, width, height, raw_candidates, flipped=True
                    )
    finally:
        image.close()

    after_nms = class_aware_nms(raw_candidates, args.global_iou)
    predictions = [
        Prediction(
            image_index=image_index,
            pred_index=pred_index,
            cls=candidate.cls,
            conf=candidate.conf,
            box=candidate.box,
        )
        for pred_index, candidate in enumerate(after_nms)
    ]
    maybe_cuda_synchronize(args.device)
    elapsed = time.perf_counter() - started
    stats = {
        "image": str(image_path),
        "width": width,
        "height": height,
        "tiles": len(positions),
        "raw_candidates": len(raw_candidates),
        "predictions_after_nms": len(predictions),
        "data_read_seconds": data_read_seconds,
        "seconds": elapsed,
        "end_to_end_seconds": data_read_seconds + elapsed,
    }
    return predictions, stats


def compact_row(row: Optional[Dict]) -> Optional[Dict]:
    if row is None:
        return None
    return {
        "threshold": row["threshold"],
        "tp": row["tp"],
        "fp": row["fp"],
        "fn": row["fn"],
        "predictions": row["prediction_count"],
        "recall": row["recall"],
        "fdr": row["fdr"],
        "precision": row["precision"],
    }


def normalize_class_names(raw_names: Any, nc: int) -> Dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(cls): str(name) for cls, name in raw_names.items()}
    if isinstance(raw_names, list):
        return {cls: str(name) for cls, name in enumerate(raw_names)}
    return {cls: str(cls) for cls in range(nc)}


def protocol_metric_row(
    tp: int,
    fp: int,
    total_gt: int,
    threshold: float,
    target_recall: float,
    max_fdr: float,
) -> Dict:
    predictions = tp + fp
    fn = total_gt - tp
    recall = tp / total_gt if total_gt else 0.0
    precision = tp / predictions if predictions else 0.0
    fdr = fp / predictions if predictions else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ground_truth": total_gt,
        "predictions": predictions,
        "recall": recall,
        "precision": precision,
        "fdr": fdr,
        "meets_recall_requirement": recall >= target_recall,
        "meets_fdr_requirement": fdr <= max_fdr,
        "meets_requirements": recall >= target_recall and fdr <= max_fdr,
    }


def evaluate_matching_protocol(
    gt_by_image: Sequence[Sequence[GroundTruth]],
    predictions: Sequence[Prediction],
    threshold: float,
    target_recall: float,
    max_fdr: float,
    require_exact_class: bool,
    allowed_classes: Optional[frozenset] = None,
) -> Dict:
    filtered_gt_by_image: List[List[GroundTruth]] = []
    for image_gts in gt_by_image:
        filtered_gt_by_image.append(
            [gt for gt in image_gts if allowed_classes is None or gt.cls in allowed_classes]
        )

    selected_predictions = [
        prediction
        for prediction in predictions
        if prediction.conf + 1e-12 >= threshold
        and (allowed_classes is None or prediction.cls in allowed_classes)
    ]
    selected_predictions.sort(key=lambda pred: (-pred.conf, pred.image_index, pred.pred_index))

    matched_by_image: List[set] = [set() for _ in filtered_gt_by_image]
    tp = 0
    fp = 0
    for prediction in selected_predictions:
        image_gts = filtered_gt_by_image[prediction.image_index]
        matched = matched_by_image[prediction.image_index]
        best_iou = -1.0
        best_gt_index: Optional[int] = None
        for gt_index, gt in enumerate(image_gts):
            if gt_index in matched:
                continue
            if require_exact_class and prediction.cls != gt.cls:
                continue
            iou = box_iou_xyxy(prediction.box, gt.box)
            required_iou = 0.35 if gt.cls == FSC_CLASS_ID else 0.50
            if iou + 1e-12 < required_iou:
                continue
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is None:
            fp += 1
        else:
            matched.add(best_gt_index)
            tp += 1

    total_gt = sum(len(image_gts) for image_gts in filtered_gt_by_image)
    return protocol_metric_row(tp, fp, total_gt, threshold, target_recall, max_fdr)


def build_protocol_metrics(
    gt_by_image: Sequence[Sequence[GroundTruth]],
    predictions: Sequence[Prediction],
    threshold: float,
    target_recall: float,
    max_fdr: float,
    class_names: Dict[int, str],
) -> Dict:
    official_overall = evaluate_matching_protocol(
        gt_by_image,
        predictions,
        threshold,
        target_recall,
        max_fdr,
        require_exact_class=False,
    )
    official_by_category = {
        category: evaluate_matching_protocol(
            gt_by_image,
            predictions,
            threshold,
            target_recall,
            max_fdr,
            require_exact_class=False,
            allowed_classes=classes,
        )
        for category, classes in BROAD_CATEGORIES.items()
    }
    strict_overall = evaluate_matching_protocol(
        gt_by_image,
        predictions,
        threshold,
        target_recall,
        max_fdr,
        require_exact_class=True,
    )
    strict_by_subclass = {
        str(cls): {
            "name": class_names.get(cls, str(cls)),
            **evaluate_matching_protocol(
                gt_by_image,
                predictions,
                threshold,
                target_recall,
                max_fdr,
                require_exact_class=True,
                allowed_classes=frozenset({cls}),
            ),
        }
        for cls in sorted(class_names)
    }
    v16_category_macro = build_v16_category_macro(strict_by_subclass)
    return {
        "evaluation_threshold": threshold,
        "official_overall": {
            "description": "Three broad categories merged; prediction class is ignored during matching.",
            **official_overall,
        },
        "official_by_category": {
            "description": "Ship, aircraft, and vehicle are evaluated separately; subclass is ignored within each category.",
            "categories": official_by_category,
        },
        "strict_25_subclass": {
            "description": "Prediction and ground-truth subclass IDs must match exactly.",
            "overall": strict_overall,
            "by_subclass": strict_by_subclass,
        },
        "v16_category_macro": v16_category_macro,
        "category_class_ids": {
            category: sorted(classes) for category, classes in BROAD_CATEGORIES.items()
        },
        "iou_policy": {
            "vehicle_ground_truth": 0.35,
            "other_ground_truth": 0.50,
        },
    }


def build_v16_category_macro(strict_by_subclass: Dict[str, Dict]) -> Dict:
    """Build the V1.6 equal-weight subtype metrics used for category ranking."""
    categories: Dict[str, Dict] = {}
    for category, class_ids in BROAD_CATEGORIES.items():
        expected_ids = sorted(class_ids)
        subtype_rows = [strict_by_subclass[str(cls)] for cls in expected_ids]
        evaluated_rows = [row for row in subtype_rows if int(row["ground_truth"]) > 0]
        missing_ids = [cls for cls, row in zip(expected_ids, subtype_rows) if int(row["ground_truth"]) == 0]
        categories[category] = {
            "aggregation": "Unweighted arithmetic mean across subtype Recall and FDR values with ground truth.",
            "expected_subclass_count": len(expected_ids),
            "evaluated_subclass_count": len(evaluated_rows),
            "missing_ground_truth_class_ids": missing_ids,
            "is_complete": not missing_ids,
            "macro_recall": (
                sum(float(row["recall"]) for row in evaluated_rows) / len(evaluated_rows)
                if evaluated_rows
                else None
            ),
            "macro_fdr": (
                sum(float(row["fdr"]) for row in evaluated_rows) / len(evaluated_rows)
                if evaluated_rows
                else None
            ),
            "macro_precision": (
                sum(float(row["precision"]) for row in evaluated_rows) / len(evaluated_rows)
                if evaluated_rows
                else None
            ),
            "subclasses": [
                {
                    "class_id": cls,
                    "name": row["name"],
                    "ground_truth": row["ground_truth"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "recall": row["recall"],
                    "fdr": row["fdr"],
                    "precision": row["precision"],
                }
                for cls, row in zip(expected_ids, subtype_rows)
            ],
        }
    return {
        "description": (
            "V1.6 category score: each subtype is matched with its exact class ID, then subtype Recall and FDR "
            "are averaged with equal weight within ship, aircraft, and vehicle."
        ),
        "categories": categories,
    }


def build_v16_ranking_inputs(v16_category_macro: Dict, per_image_stats: Sequence[Dict]) -> Dict:
    categories = v16_category_macro["categories"]
    times = [float(stat["seconds"]) for stat in per_image_stats]
    return {
        "description": (
            "Six V1.6 category metrics plus timing evidence. The document requires seven ranks but does not define "
            "one aggregate timeliness scalar, so total, mean, and maximum measured inference time are all reported."
        ),
        "ship_recall": categories["ship"]["macro_recall"],
        "ship_fdr": categories["ship"]["macro_fdr"],
        "aircraft_recall": categories["aircraft"]["macro_recall"],
        "aircraft_fdr": categories["aircraft"]["macro_fdr"],
        "vehicle_recall": categories["vehicle"]["macro_recall"],
        "vehicle_fdr": categories["vehicle"]["macro_fdr"],
        "timing": {
            "images": len(times),
            "total_seconds": sum(times),
            "mean_seconds": sum(times) / len(times) if times else None,
            "max_seconds": max(times) if times else None,
        },
    }


def build_summary(rows: Sequence[Dict], per_image_stats: Sequence[Dict], args: argparse.Namespace, metadata: Dict) -> Dict:
    feasible = [row for row in rows if row["meets_requirements"]]
    under_fdr = [row for row in rows if row["fdr"] <= args.max_fdr]
    over_recall = [row for row in rows if row["recall"] >= args.target_recall]

    best_feasible = max(feasible, key=lambda row: (row["recall"], -row["fdr"], row["threshold"])) if feasible else None
    best_recall_under_fdr = (
        max(under_fdr, key=lambda row: (row["recall"], -row["fdr"], row["threshold"])) if under_fdr else None
    )
    best_fdr_with_recall = (
        min(over_recall, key=lambda row: (row["fdr"], -row["recall"], -row["threshold"])) if over_recall else None
    )

    feasible_threshold_range = None
    if feasible:
        thresholds = [row["threshold"] for row in feasible]
        feasible_threshold_range = {"min": min(thresholds), "max": max(thresholds), "count": len(thresholds)}

    times = [stat["seconds"] for stat in per_image_stats]
    max_seconds = max(times) if times else 0.0
    mean_seconds = sum(times) / len(times) if times else 0.0
    meets_time_requirement = max_seconds <= args.time_limit if times else False
    return {
        "images": len(per_image_stats),
        "ground_truth": metadata["ground_truth"],
        "candidate_predictions": metadata["candidate_predictions"],
        "predictions_after_nms": metadata["predictions_after_nms"],
        "predictions_after_class_thresholds": metadata.get("predictions_after_class_thresholds"),
        "vehicle_classes": [FSC_CLASS_ID],
        "vehicle_iou": 0.35,
        "other_iou": 0.50,
        "target_recall": args.target_recall,
        "max_fdr": args.max_fdr,
        "time_limit_seconds": args.time_limit,
        "meets_requirements": bool(feasible),
        "meets_time_requirement": meets_time_requirement,
        "meets_all_requirements": bool(feasible) and meets_time_requirement,
        "class_threshold_result": compact_row(rows[0]) if metadata.get("class_threshold_filter_enabled") and rows else None,
        "best_feasible": compact_row(best_feasible),
        "best_recall_under_fdr_20": compact_row(best_recall_under_fdr),
        "best_fdr_with_recall_85": compact_row(best_fdr_with_recall),
        "feasible_threshold_range": feasible_threshold_range,
        "timing": {
            "max_seconds": max_seconds,
            "mean_seconds": mean_seconds,
            "min_seconds": min(times) if times else 0.0,
            "images_over_limit": sum(1 for value in times if value > args.time_limit),
        },
        "metadata": metadata,
    }


def write_threshold_csv(rows: Sequence[Dict], path: Path) -> None:
    fieldnames = [
        "threshold",
        "tp",
        "fp",
        "fn",
        "ground_truth",
        "predictions",
        "recall",
        "precision",
        "fdr",
        "meets_requirements",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "threshold": f"{row['threshold']:.6f}",
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "ground_truth": row["gt_total"],
                    "predictions": row["prediction_count"],
                    "recall": f"{row['recall']:.8f}",
                    "precision": f"{row['precision']:.8f}",
                    "fdr": f"{row['fdr']:.8f}",
                    "meets_requirements": int(row["meets_requirements"]),
                }
            )


def write_per_image_csv(rows: Sequence[Dict], path: Path) -> None:
    fieldnames = [
        "image",
        "width",
        "height",
        "tiles",
        "ground_truth",
        "raw_candidates",
        "predictions_after_nms",
        "data_read_seconds",
        "seconds",
        "end_to_end_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "width": row["width"],
                    "height": row["height"],
                    "tiles": row["tiles"],
                    "ground_truth": row["ground_truth"],
                    "raw_candidates": row["raw_candidates"],
                    "predictions_after_nms": row["predictions_after_nms"],
                    "data_read_seconds": f"{row['data_read_seconds']:.6f}",
                    "seconds": f"{row['seconds']:.6f}",
                    "end_to_end_seconds": f"{row['end_to_end_seconds']:.6f}",
                }
            )


def write_v16_category_csv(v16_category_macro: Dict, v16_ranking_inputs: Dict, path: Path) -> None:
    fieldnames = [
        "row_type",
        "category",
        "class_id",
        "class_name",
        "ground_truth",
        "tp",
        "fp",
        "fn",
        "recall",
        "fdr",
        "precision",
        "expected_subclass_count",
        "evaluated_subclass_count",
        "missing_ground_truth_class_ids",
        "is_complete",
        "timing_metric",
        "timing_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for category, metrics in v16_category_macro["categories"].items():
            for subtype in metrics["subclasses"]:
                writer.writerow(
                    {
                        "row_type": "subclass",
                        "category": category,
                        "class_id": subtype["class_id"],
                        "class_name": subtype["name"],
                        "ground_truth": subtype["ground_truth"],
                        "tp": subtype["tp"],
                        "fp": subtype["fp"],
                        "fn": subtype["fn"],
                        "recall": f"{subtype['recall']:.8f}",
                        "fdr": f"{subtype['fdr']:.8f}",
                        "precision": f"{subtype['precision']:.8f}",
                    }
                )
            writer.writerow(
                {
                    "row_type": "category_macro",
                    "category": category,
                    "recall": (
                        f"{metrics['macro_recall']:.8f}" if metrics["macro_recall"] is not None else ""
                    ),
                    "fdr": f"{metrics['macro_fdr']:.8f}" if metrics["macro_fdr"] is not None else "",
                    "precision": (
                        f"{metrics['macro_precision']:.8f}" if metrics["macro_precision"] is not None else ""
                    ),
                    "expected_subclass_count": metrics["expected_subclass_count"],
                    "evaluated_subclass_count": metrics["evaluated_subclass_count"],
                    "missing_ground_truth_class_ids": ";".join(
                        str(cls) for cls in metrics["missing_ground_truth_class_ids"]
                    ),
                    "is_complete": int(metrics["is_complete"]),
                }
            )
        timing = v16_ranking_inputs["timing"]
        for metric_name in ("total_seconds", "mean_seconds", "max_seconds"):
            value = timing[metric_name]
            writer.writerow(
                {
                    "row_type": "timing",
                    "ground_truth": timing["images"],
                    "timing_metric": metric_name,
                    "timing_seconds": f"{value:.8f}" if value is not None else "",
                }
            )


def main() -> int:
    args = parse_args()
    validate_args(args)

    model_path = resolve_project_path(args.model)
    data_yaml = resolve_project_path(args.data)
    ultralytics_root = resolve_project_path(args.ultralytics_root)
    output_dir = resolve_project_path(args.output_dir)
    class_thresholds_path = resolve_project_path(args.class_thresholds) if args.class_thresholds else None
    if not model_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"model not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
    if class_thresholds_path is not None and not class_thresholds_path.exists():
        raise FileNotFoundError(f"class thresholds JSON not found: {class_thresholds_path}")

    cfg = load_yaml(data_yaml)
    class_names = normalize_class_names(cfg.get("names"), int(cfg.get("nc", 0)))
    data_root = resolve_data_root(data_yaml, cfg)
    image_paths = resolve_split_sources(data_yaml, cfg, args.split)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    thresholds = make_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    class_thresholds = load_class_thresholds(class_thresholds_path) if class_thresholds_path is not None else {}

    planned_tiles = 0
    if args.dry_run:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                width, height = image.size
            planned_tiles += len(tile_positions(width, height, args.tile_size, args.stride))
        print(f"data={data_yaml}", flush=True)
        print(f"split={args.split} images={len(image_paths)} planned_tiles={planned_tiles}", flush=True)
        print("dry_run=ok", flush=True)
        return 0

    setup_ultralytics(ultralytics_root)
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if args.broad_fusion_gain is not None:
        head = model.model.model[-1]
        if not hasattr(head, "broad_fusion_gain"):
            raise ValueError("--broad-fusion-gain requires a HierarchicalDetect checkpoint")
        head.broad_fusion_gain = args.broad_fusion_gain
    gt_by_image: List[List[GroundTruth]] = []
    predictions: List[Prediction] = []
    per_image_stats: List[Dict] = []
    missing_label_files = 0

    print(f"model={model_path}", flush=True)
    print(f"data={data_yaml}", flush=True)
    print(f"split={args.split} images={len(image_paths)}", flush=True)
    print(
        f"sliding_window=tile_size={args.tile_size} stride={args.stride} "
        f"pred_conf={args.pred_conf} tile_iou={args.pred_iou} global_iou={args.global_iou}",
        flush=True,
    )

    for image_index, image_path in enumerate(image_paths):
        image_predictions, stats = predict_big_image(model, image_path, image_index, args)
        label_path = label_path_for_image(image_path, data_root, args.split)
        if not label_path.exists():
            missing_label_files += 1
        image_gts = load_ground_truths(label_path, stats["width"], stats["height"])
        stats["ground_truth"] = len(image_gts)
        gt_by_image.append(image_gts)
        predictions.extend(image_predictions)
        per_image_stats.append(stats)
        print(
            f"image={image_index + 1}/{len(image_paths)} tiles={stats['tiles']} "
            f"gt={stats['ground_truth']} raw={stats['raw_candidates']} "
            f"nms={stats['predictions_after_nms']} seconds={stats['seconds']:.3f}",
            flush=True,
        )

    predictions_after_nms = len(predictions)
    if class_thresholds:
        predictions = apply_class_thresholds(predictions, class_thresholds)
        print(
            f"class_thresholds={class_thresholds_path} kept={len(predictions)} "
            f"dropped={predictions_after_nms - len(predictions)}",
            flush=True,
        )

    rows = evaluate_thresholds(
        gt_by_image=gt_by_image,
        predictions=predictions,
        thresholds=thresholds,
        target_recall=args.target_recall,
        max_fdr=args.max_fdr,
    )
    metadata = {
        "model": str(model_path),
        "data": str(data_yaml),
        "data_root": str(data_root),
        "split": args.split,
        "images": len(image_paths),
        "ground_truth": sum(len(gts) for gts in gt_by_image),
        "candidate_predictions": sum(stat["raw_candidates"] for stat in per_image_stats),
        "predictions_after_nms": predictions_after_nms,
        "predictions_after_class_thresholds": len(predictions),
        "missing_label_files": missing_label_files,
        "imgsz": args.imgsz,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "pred_conf": args.pred_conf,
        "pred_iou": args.pred_iou,
        "global_iou": args.global_iou,
        "max_det_per_tile": args.max_det,
        "batch": args.batch,
        "half": args.half,
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
        "class_thresholds": str(class_thresholds_path) if class_thresholds_path is not None else None,
        "class_threshold_filter_enabled": bool(class_thresholds),
        "broad_fusion_gain": args.broad_fusion_gain,
        "iou_policy": {"class_24_FSC": 0.35, "other_classes": 0.50},
        "class_match_required": True,
        "timing_definition": "From completed image read/decode to final postprocessed predictions; excludes data reading.",
    }
    summary = build_summary(rows, per_image_stats, args, metadata)
    selected = summary["class_threshold_result"] if class_thresholds else summary["best_feasible"]
    if selected is None:
        selected = summary["best_recall_under_fdr_20"]
    protocol_threshold = selected["threshold"] if selected is not None else args.threshold_start
    protocol_metrics = build_protocol_metrics(
        gt_by_image=gt_by_image,
        predictions=predictions,
        threshold=protocol_threshold,
        target_recall=args.target_recall,
        max_fdr=args.max_fdr,
        class_names=class_names,
    )
    v16_ranking_inputs = build_v16_ranking_inputs(protocol_metrics["v16_category_macro"], per_image_stats)
    protocol_metrics["v16_ranking_inputs"] = v16_ranking_inputs
    summary["protocol_metrics"] = protocol_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    summary_path = output_dir / args.summary_name
    per_image_csv_path = output_dir / args.per_image_csv_name
    protocols_path = output_dir / args.protocols_name
    v16_csv_path = output_dir / args.v16_csv_name
    write_threshold_csv(rows, csv_path)
    write_per_image_csv(per_image_stats, per_image_csv_path)
    write_v16_category_csv(protocol_metrics["v16_category_macro"], v16_ranking_inputs, v16_csv_path)
    summary["csv"] = str(csv_path)
    summary["per_image_csv"] = str(per_image_csv_path)
    summary["protocol_metrics_json"] = str(protocols_path)
    summary["v16_category_metrics_csv"] = str(v16_csv_path)
    summary["summary"] = str(summary_path)
    protocols_path.write_text(json.dumps(protocol_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"csv={csv_path}", flush=True)
    print(f"per_image_csv={per_image_csv_path}", flush=True)
    print(f"protocol_metrics={protocols_path}", flush=True)
    print(f"v16_category_metrics_csv={v16_csv_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(f"meets_requirements={summary['meets_requirements']}", flush=True)
    print(f"meets_time_requirement={summary['meets_time_requirement']}", flush=True)
    print(f"best_feasible={summary['best_feasible']}", flush=True)
    print(f"official_overall={protocol_metrics['official_overall']}", flush=True)
    print(f"official_by_category={protocol_metrics['official_by_category']['categories']}", flush=True)
    print(f"strict_25_subclass={protocol_metrics['strict_25_subclass']['overall']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
