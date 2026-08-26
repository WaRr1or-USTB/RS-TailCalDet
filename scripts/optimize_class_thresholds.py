from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from evaluate_big_images_sliding import predict_big_image
from evaluate_official_thresholds import (
    FSC_CLASS_ID,
    GroundTruth,
    Prediction,
    box_iou_xyxy,
    iou_threshold_for_class,
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
DEFAULT_DATA = Path("data_big_report_calibration_s2026/dataset.yaml")
DEFAULT_ULTRALYTICS_ROOT = Path("ultralytics-main")
DEFAULT_OUTPUT_DIR = Path("runs/detect/report_threshold_calibration_recheck")
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class ClassCurvePoint:
    threshold: float
    tp: int
    fp: int
    fn: int
    gt: int
    predictions: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize per-class confidence thresholds for big-image sliding-window predictions."
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
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--base-threshold", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--min-recall", type=float, default=0.94)
    parser.add_argument(
        "--min-class-recall",
        action="append",
        default=[],
        metavar="CLASS_ID=RECALL",
        help="Minimum recall floor for a class, e.g. 24=0.85. May be repeated.",
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--min-fdr-improvement", type=float, default=1e-6)
    parser.add_argument("--tta", action="store_true", help="Use flip TTA inference (matches evaluate_big_images_sliding).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--class-metrics-name", default="class_metrics.csv")
    parser.add_argument("--trace-name", default="optimization_trace.csv")
    parser.add_argument("--thresholds-name", default="class_thresholds.json")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.batch <= 0:
        raise ValueError("--batch must be positive.")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive.")
    for name in ("pred_conf", "pred_iou", "global_iou"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if not 0.0 <= args.base_threshold <= 1.0:
        raise ValueError("--base-threshold must be between 0 and 1.")
    if args.base_threshold + 1e-12 < args.pred_conf:
        raise ValueError("--base-threshold cannot be lower than --pred-conf.")
    if args.base_threshold > args.threshold_stop:
        raise ValueError("--base-threshold must be <= --threshold-stop.")
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive.")
    if not 0.0 <= args.min_recall <= 1.0:
        raise ValueError("--min-recall must be between 0 and 1.")
    if args.max_rounds < 0:
        raise ValueError("--max-rounds must be non-negative.")


def parse_min_class_recalls(values: Sequence[str], classes: Sequence[int]) -> Dict[int, float]:
    allowed_classes = set(classes)
    floors: Dict[int, float] = {}
    for value in values:
        try:
            class_text, recall_text = value.split("=", 1)
            class_id = int(class_text)
            recall = float(recall_text)
        except ValueError as exc:
            raise ValueError(f"invalid --min-class-recall value: {value!r}; expected CLASS_ID=RECALL") from exc
        if class_id not in allowed_classes:
            raise ValueError(f"class ID {class_id} is outside the dataset class range")
        if not 0.0 <= recall <= 1.0:
            raise ValueError(f"minimum recall for class {class_id} must be between 0 and 1")
        floors[class_id] = recall
    return floors


def normalize_names(raw_names, nc: int) -> Dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    return {idx: str(idx) for idx in range(nc)}


def metric_row(tp: int, fp: int, gt_total: int) -> Dict:
    predictions = tp + fp
    fn = gt_total - tp
    recall = tp / gt_total if gt_total else 0.0
    precision = tp / predictions if predictions else 0.0
    fdr = fp / predictions if predictions else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_total": gt_total,
        "prediction_count": predictions,
        "recall": recall,
        "precision": precision,
        "fdr": fdr,
    }


def evaluate_class_at_threshold(
    cls: int,
    threshold: float,
    gt_by_image: Sequence[Sequence[GroundTruth]],
    predictions_by_class: Dict[int, List[Prediction]],
    gt_index_by_image_class: Sequence[Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]],
    gt_count_by_class: Dict[int, int],
) -> ClassCurvePoint:
    matched_by_image: List[set] = [set() for _ in gt_by_image]
    tp = 0
    fp = 0
    selected = 0
    for prediction in predictions_by_class.get(cls, []):
        if prediction.conf + 1e-12 < threshold:
            continue
        selected += 1
        candidates = gt_index_by_image_class[prediction.image_index].get(cls, [])
        matched = matched_by_image[prediction.image_index]
        best_iou = 0.0
        best_gt_index: Optional[int] = None
        for gt_index, gt_box in candidates:
            if gt_index in matched:
                continue
            iou = box_iou_xyxy(prediction.box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold_for_class(cls):
            matched.add(best_gt_index)
            tp += 1
        else:
            fp += 1
    gt = gt_count_by_class.get(cls, 0)
    return ClassCurvePoint(threshold=threshold, tp=tp, fp=fp, fn=gt - tp, gt=gt, predictions=selected)


def build_class_curves(
    classes: Sequence[int],
    thresholds: Sequence[float],
    gt_by_image: Sequence[Sequence[GroundTruth]],
    predictions: Sequence[Prediction],
) -> Tuple[Dict[int, List[ClassCurvePoint]], Dict[int, int]]:
    gt_count_by_class: Dict[int, int] = {cls: 0 for cls in classes}
    gt_index_by_image_class: List[Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]] = []
    for image_gts in gt_by_image:
        by_class: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
        for gt_index, gt in enumerate(image_gts):
            by_class.setdefault(gt.cls, []).append((gt_index, gt.box))
            gt_count_by_class[gt.cls] = gt_count_by_class.get(gt.cls, 0) + 1
        gt_index_by_image_class.append(by_class)

    predictions_by_class: Dict[int, List[Prediction]] = {cls: [] for cls in classes}
    for prediction in sorted(predictions, key=lambda pred: (-pred.conf, pred.image_index, pred.pred_index)):
        predictions_by_class.setdefault(prediction.cls, []).append(prediction)

    curves: Dict[int, List[ClassCurvePoint]] = {}
    for cls in classes:
        curves[cls] = [
            evaluate_class_at_threshold(
                cls=cls,
                threshold=threshold,
                gt_by_image=gt_by_image,
                predictions_by_class=predictions_by_class,
                gt_index_by_image_class=gt_index_by_image_class,
                gt_count_by_class=gt_count_by_class,
            )
            for threshold in thresholds
        ]
    return curves, gt_count_by_class


def closest_threshold_index(thresholds: Sequence[float], value: float) -> int:
    best_index = 0
    best_distance = float("inf")
    for idx, threshold in enumerate(thresholds):
        distance = abs(threshold - value)
        if distance < best_distance:
            best_index = idx
            best_distance = distance
    return best_index


def combine_curve_points(curves: Dict[int, List[ClassCurvePoint]], indices: Dict[int, int]) -> Dict:
    tp = sum(curves[cls][idx].tp for cls, idx in indices.items())
    fp = sum(curves[cls][idx].fp for cls, idx in indices.items())
    gt_total = sum(curves[cls][idx].gt for cls, idx in indices.items())
    return metric_row(tp, fp, gt_total)


def meets_class_recall_floors(
    curves: Dict[int, List[ClassCurvePoint]],
    indices: Dict[int, int],
    min_class_recalls: Dict[int, float],
) -> bool:
    for cls, min_recall in min_class_recalls.items():
        point = curves[cls][indices[cls]]
        if point.gt and point.tp / point.gt + 1e-12 < min_recall:
            return False
    return True


def is_better(candidate: Dict, current: Dict, min_fdr_improvement: float) -> bool:
    if candidate["fdr"] < current["fdr"] - min_fdr_improvement:
        return True
    if abs(candidate["fdr"] - current["fdr"]) <= min_fdr_improvement:
        if candidate["recall"] > current["recall"] + 1e-12:
            return True
        if abs(candidate["recall"] - current["recall"]) <= 1e-12:
            return candidate["precision"] > current["precision"] + 1e-12
    return False


def optimize_thresholds(
    classes: Sequence[int],
    thresholds: Sequence[float],
    curves: Dict[int, List[ClassCurvePoint]],
    base_threshold: float,
    min_recall: float,
    min_class_recalls: Dict[int, float],
    max_rounds: int,
    min_fdr_improvement: float,
) -> Tuple[Dict[int, int], Dict, List[Dict]]:
    base_index = closest_threshold_index(thresholds, base_threshold)
    current_indices = {cls: base_index for cls in classes}
    for cls, min_class_recall in min_class_recalls.items():
        feasible_indices = [
            idx
            for idx, point in enumerate(curves[cls])
            if not point.gt or point.tp / point.gt + 1e-12 >= min_class_recall
        ]
        if not feasible_indices:
            base_point = curves[cls][base_index]
            base_recall = base_point.tp / base_point.gt if base_point.gt else 0.0
            raise ValueError(
                f"class {cls} cannot satisfy recall floor {min_class_recall:.6f}; "
                f"recall at base threshold is {base_recall:.6f}"
            )
        current_indices[cls] = feasible_indices[-1]
    current_metrics = combine_curve_points(curves, current_indices)
    trace = [
        {
            "round": 0,
            "class_id": None,
            "threshold": thresholds[base_index],
            **current_metrics,
        }
    ]
    if current_metrics["recall"] + 1e-12 < min_recall:
        return current_indices, current_metrics, trace
    if not meets_class_recall_floors(curves, current_indices, min_class_recalls):
        raise AssertionError("class recall floor initialization failed")

    for round_index in range(1, max_rounds + 1):
        best_indices = None
        best_metrics = current_metrics
        best_cls = None
        best_threshold = None
        for cls in classes:
            start_index = current_indices[cls] + 1
            for idx in range(start_index, len(thresholds)):
                candidate_indices = dict(current_indices)
                candidate_indices[cls] = idx
                candidate_metrics = combine_curve_points(curves, candidate_indices)
                if candidate_metrics["recall"] + 1e-12 < min_recall:
                    continue
                if not meets_class_recall_floors(curves, candidate_indices, min_class_recalls):
                    continue
                if is_better(candidate_metrics, best_metrics, min_fdr_improvement):
                    best_indices = candidate_indices
                    best_metrics = candidate_metrics
                    best_cls = cls
                    best_threshold = thresholds[idx]
        if best_indices is None:
            break
        current_indices = best_indices
        current_metrics = best_metrics
        trace.append(
            {
                "round": round_index,
                "class_id": best_cls,
                "threshold": best_threshold,
                **current_metrics,
            }
        )
    return current_indices, current_metrics, trace


def class_metrics_rows(
    classes: Sequence[int],
    names: Dict[int, str],
    curves: Dict[int, List[ClassCurvePoint]],
    base_indices: Dict[int, int],
    optimized_indices: Dict[int, int],
) -> List[Dict]:
    rows: List[Dict] = []
    for cls in classes:
        base = curves[cls][base_indices[cls]]
        optimized = curves[cls][optimized_indices[cls]]
        rows.append(
            {
                "class_id": cls,
                "class_name": names.get(cls, str(cls)),
                "gt": optimized.gt,
                "base_threshold": base.threshold,
                "base_tp": base.tp,
                "base_fp": base.fp,
                "base_fn": base.fn,
                "base_predictions": base.predictions,
                "optimized_threshold": optimized.threshold,
                "optimized_tp": optimized.tp,
                "optimized_fp": optimized.fp,
                "optimized_fn": optimized.fn,
                "optimized_predictions": optimized.predictions,
                "delta_tp": optimized.tp - base.tp,
                "delta_fp": optimized.fp - base.fp,
                "delta_fn": optimized.fn - base.fn,
            }
        )
    rows.sort(key=lambda row: (row["delta_fp"], -row["delta_tp"], row["class_id"]))
    return rows


def write_class_metrics_csv(rows: Sequence[Dict], path: Path) -> None:
    fieldnames = [
        "class_id",
        "class_name",
        "gt",
        "base_threshold",
        "base_tp",
        "base_fp",
        "base_fn",
        "base_predictions",
        "optimized_threshold",
        "optimized_tp",
        "optimized_fp",
        "optimized_fn",
        "optimized_predictions",
        "delta_tp",
        "delta_fp",
        "delta_fn",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_trace_csv(rows: Sequence[Dict], path: Path) -> None:
    fieldnames = ["round", "class_id", "threshold", "tp", "fp", "fn", "gt_total", "prediction_count", "recall", "precision", "fdr"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    validate_args(args)

    model_path = resolve_project_path(args.model)
    data_yaml = resolve_project_path(args.data)
    ultralytics_root = resolve_project_path(args.ultralytics_root)
    output_dir = resolve_project_path(args.output_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")

    cfg = load_yaml(data_yaml)
    data_root = resolve_data_root(data_yaml, cfg)
    nc = int(cfg.get("nc", 0))
    classes = list(range(nc))
    min_class_recalls = parse_min_class_recalls(args.min_class_recall, classes)
    names = normalize_names(cfg.get("names"), nc)
    image_paths = resolve_split_sources(data_yaml, cfg, args.split)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    thresholds = make_thresholds(args.base_threshold, args.threshold_stop, args.threshold_step)
    if args.base_threshold not in thresholds:
        thresholds = sorted(set(thresholds + [args.base_threshold]))

    setup_ultralytics(ultralytics_root)
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    gt_by_image: List[List[GroundTruth]] = []
    predictions: List[Prediction] = []
    per_image_stats: List[Dict] = []
    missing_label_files = 0

    print(f"model={model_path}", flush=True)
    print(f"data={data_yaml}", flush=True)
    print(f"split={args.split} images={len(image_paths)}", flush=True)
    print(
        f"collecting_predictions=tile_size={args.tile_size} stride={args.stride} "
        f"pred_conf={args.pred_conf} pred_iou={args.pred_iou} global_iou={args.global_iou}",
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
            f"image={image_index + 1}/{len(image_paths)} gt={stats['ground_truth']} "
            f"raw={stats['raw_candidates']} nms={stats['predictions_after_nms']} "
            f"seconds={stats['seconds']:.3f}",
            flush=True,
        )

    print(f"building_class_curves=classes={len(classes)} thresholds={len(thresholds)}", flush=True)
    curves, gt_count_by_class = build_class_curves(classes, thresholds, gt_by_image, predictions)
    base_index = closest_threshold_index(thresholds, args.base_threshold)
    base_indices = {cls: base_index for cls in classes}
    base_metrics = combine_curve_points(curves, base_indices)
    optimized_indices, optimized_metrics, trace = optimize_thresholds(
        classes=classes,
        thresholds=thresholds,
        curves=curves,
        base_threshold=args.base_threshold,
        min_recall=args.min_recall,
        min_class_recalls=min_class_recalls,
        max_rounds=args.max_rounds,
        min_fdr_improvement=args.min_fdr_improvement,
    )
    class_rows = class_metrics_rows(classes, names, curves, base_indices, optimized_indices)

    class_thresholds = {
        str(cls): {
            "name": names.get(cls, str(cls)),
            "threshold": curves[cls][optimized_indices[cls]].threshold,
        }
        for cls in classes
    }
    summary = {
        "model": str(model_path),
        "data": str(data_yaml),
        "data_root": str(data_root),
        "split": args.split,
        "images": len(image_paths),
        "ground_truth": sum(len(gts) for gts in gt_by_image),
        "predictions_after_nms": len(predictions),
        "missing_label_files": missing_label_files,
        "base_threshold": args.base_threshold,
        "min_recall": args.min_recall,
        "min_class_recalls": {str(cls): recall for cls, recall in min_class_recalls.items()},
        "threshold_step": args.threshold_step,
        "vehicle_classes": [FSC_CLASS_ID],
        "vehicle_iou": 0.35,
        "other_iou": 0.50,
        "baseline": base_metrics,
        "optimized": optimized_metrics,
        "improvement": {
            "delta_tp": optimized_metrics["tp"] - base_metrics["tp"],
            "delta_fp": optimized_metrics["fp"] - base_metrics["fp"],
            "delta_fn": optimized_metrics["fn"] - base_metrics["fn"],
            "delta_recall": optimized_metrics["recall"] - base_metrics["recall"],
            "delta_precision": optimized_metrics["precision"] - base_metrics["precision"],
            "delta_fdr": optimized_metrics["fdr"] - base_metrics["fdr"],
        },
        "thresholds": class_thresholds,
        "trace": trace,
        "metadata": {
            "imgsz": args.imgsz,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "pred_conf": args.pred_conf,
            "pred_iou": args.pred_iou,
            "global_iou": args.global_iou,
            "max_det_per_tile": args.max_det,
            "batch": args.batch,
            "half": args.half,
            "gt_count_by_class": {str(cls): gt_count_by_class.get(cls, 0) for cls in classes},
            "per_image_timing": {
                "max_seconds": max((row["seconds"] for row in per_image_stats), default=0.0),
                "mean_seconds": sum(row["seconds"] for row in per_image_stats) / len(per_image_stats)
                if per_image_stats
                else 0.0,
                "min_seconds": min((row["seconds"] for row in per_image_stats), default=0.0),
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_name
    class_metrics_path = output_dir / args.class_metrics_name
    trace_path = output_dir / args.trace_name
    thresholds_path = output_dir / args.thresholds_name
    summary["summary"] = str(summary_path)
    summary["class_metrics_csv"] = str(class_metrics_path)
    summary["trace_csv"] = str(trace_path)
    summary["class_thresholds_json"] = str(thresholds_path)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    thresholds_path.write_text(
        json.dumps(
            {
                "base_threshold": args.base_threshold,
                "min_recall": args.min_recall,
                "min_class_recalls": {str(cls): recall for cls, recall in min_class_recalls.items()},
                "thresholds": class_thresholds,
                "optimized": optimized_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_class_metrics_csv(class_rows, class_metrics_path)
    write_trace_csv(trace, trace_path)

    print(f"baseline={base_metrics}", flush=True)
    print(f"optimized={optimized_metrics}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(f"class_thresholds={thresholds_path}", flush=True)
    print(f"class_metrics_csv={class_metrics_path}", flush=True)
    print(f"trace_csv={trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
