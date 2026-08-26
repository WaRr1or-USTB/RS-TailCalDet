from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics_config"

DEFAULT_MODEL = Path("runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt")
DEFAULT_DATA = Path("data_v2/report_split_v16/dataset_report.yaml")
DEFAULT_ULTRALYTICS_ROOT = Path("ultralytics-main")
DEFAULT_OUTPUT_DIR = Path("runs/detect/report_model_official_threshold_eval")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FSC_CLASS_ID = 24


@dataclass(frozen=True)
class GroundTruth:
    cls: int
    box: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    image_index: int
    pred_index: int
    cls: int
    conf: float
    box: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YOLO predictions with Challenge Cup official-like TP/FP/FN "
            "matching and sweep confidence thresholds."
        )
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--ultralytics-root", default=str(DEFAULT_ULTRALYTICS_ROOT))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--pred-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--csv-name", default="threshold_metrics.csv")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print progress every N images during prediction. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yaml(path: Path) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the dataset YAML.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {path}")
    return data


def resolve_data_root(data_yaml: Path, cfg: Dict) -> Path:
    root_value = cfg.get("path", data_yaml.parent)
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_sources(data_yaml: Path, cfg: Dict, split: str) -> List[Path]:
    if split not in cfg:
        raise KeyError(f"Split '{split}' was not found in {data_yaml}")

    data_root = resolve_data_root(data_yaml, cfg)
    split_value = cfg[split]
    raw_sources: Sequence[str]
    if isinstance(split_value, (list, tuple)):
        raw_sources = [str(item) for item in split_value]
    else:
        raw_sources = [str(split_value)]

    image_paths: List[Path] = []
    for raw_source in raw_sources:
        source = Path(raw_source).expanduser()
        if not source.is_absolute():
            source = data_root / source
        source = source.resolve()

        if source.is_dir():
            found = sorted(
                p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            image_paths.extend(found)
        elif source.is_file() and source.suffix.lower() == ".txt":
            image_paths.extend(read_image_list(source, data_root))
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(source)
        else:
            raise FileNotFoundError(f"Split source does not exist or has no supported images: {source}")

    unique_paths = sorted(dict.fromkeys(image_paths))
    if not unique_paths:
        raise FileNotFoundError(f"No images found for split '{split}' in {data_yaml}")
    return unique_paths


def read_image_list(list_path: Path, data_root: Path) -> List[Path]:
    image_paths: List[Path] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = data_root / path
            image_paths.append(path.resolve())
    return image_paths


def label_path_for_image(image_path: Path, data_root: Path, split: str) -> Path:
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return (data_root / "labels" / split / image_path.with_suffix(".txt").name).resolve()


def yolo_xywhn_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    return x1, y1, x2, y2


def load_ground_truths(label_path: Path, image_width: int, image_height: int) -> List[GroundTruth]:
    if not label_path.exists():
        return []

    targets: List[GroundTruth] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_no}: {raw}")
            cls = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:5])
            box = yolo_xywhn_to_xyxy(x_center, y_center, width, height, image_width, image_height)
            targets.append(GroundTruth(cls=cls, box=box))
    return targets


def setup_ultralytics(ultralytics_root: Path) -> None:
    if not ultralytics_root.exists():
        raise FileNotFoundError(f"ultralytics root not found: {ultralytics_root}")
    sys.path.insert(0, str(ultralytics_root))
    YOLO_CONFIG_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))


def collect_predictions(
    model_path: Path,
    image_paths: Sequence[Path],
    data_root: Path,
    split: str,
    args: argparse.Namespace,
) -> Tuple[List[List[GroundTruth]], List[Prediction], int]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    source = [str(path) for path in image_paths]
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.pred_conf,
        iou=args.pred_iou,
        max_det=args.max_det,
        batch=args.batch,
        device=args.device,
        stream=True,
        verbose=False,
        save=False,
    )

    gt_by_image: List[List[GroundTruth]] = []
    predictions: List[Prediction] = []
    missing_label_files = 0

    for image_index, result in enumerate(results):
        image_path = Path(result.path).resolve()
        image_height, image_width = result.orig_shape[:2]
        label_path = label_path_for_image(image_path, data_root, split)
        if not label_path.exists():
            missing_label_files += 1
        gt_by_image.append(load_ground_truths(label_path, image_width, image_height))

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            for pred_index, (box, conf, cls) in enumerate(zip(xyxy, confs, classes)):
                predictions.append(
                    Prediction(
                        image_index=image_index,
                        pred_index=pred_index,
                        cls=int(cls),
                        conf=float(conf),
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    )
                )

        processed = image_index + 1
        if args.print_every > 0 and processed % args.print_every == 0:
            print(
                f"predicted_images={processed}/{len(image_paths)} "
                f"candidate_predictions={len(predictions)}",
                flush=True,
            )

    if len(gt_by_image) != len(image_paths):
        raise RuntimeError(f"Expected {len(image_paths)} prediction results, got {len(gt_by_image)}")
    return gt_by_image, predictions, missing_label_files


def box_iou_xyxy(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def iou_threshold_for_class(cls: int) -> float:
    return 0.35 if cls == FSC_CLASS_ID else 0.50


def apply_prediction(
    prediction: Prediction,
    gt_by_image: Sequence[Sequence[GroundTruth]],
    matched_by_image: Sequence[set],
) -> bool:
    candidates = gt_by_image[prediction.image_index]
    matched = matched_by_image[prediction.image_index]
    best_iou = 0.0
    best_gt_index: Optional[int] = None

    for gt_index, gt in enumerate(candidates):
        if gt_index in matched:
            continue
        if gt.cls != prediction.cls:
            continue
        iou = box_iou_xyxy(prediction.box, gt.box)
        if iou > best_iou:
            best_iou = iou
            best_gt_index = gt_index

    if best_gt_index is None:
        return False

    threshold = iou_threshold_for_class(prediction.cls)
    if best_iou < threshold:
        return False

    matched.add(best_gt_index)
    return True


def make_thresholds(start: float, stop: float, step: float) -> List[float]:
    start_d = Decimal(str(start))
    stop_d = Decimal(str(stop))
    step_d = Decimal(str(step))
    if step_d <= 0:
        raise ValueError("--threshold-step must be positive.")
    if start_d > stop_d:
        raise ValueError("--threshold-start must be <= --threshold-stop.")

    thresholds: List[float] = []
    current = start_d
    while current <= stop_d:
        thresholds.append(float(current))
        current += step_d
    return thresholds


def build_metric_row(
    threshold: float,
    tp: int,
    fp: int,
    total_gt: int,
    target_recall: float,
    max_fdr: float,
) -> Dict:
    fn = total_gt - tp
    prediction_count = tp + fp
    recall = tp / total_gt if total_gt else 0.0
    precision = tp / prediction_count if prediction_count else 0.0
    fdr = fp / prediction_count if prediction_count else 0.0
    meets_requirements = recall >= target_recall and fdr <= max_fdr
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_total": total_gt,
        "prediction_count": prediction_count,
        "recall": recall,
        "precision": precision,
        "fdr": fdr,
        "meets_requirements": meets_requirements,
    }


def evaluate_thresholds(
    gt_by_image: Sequence[Sequence[GroundTruth]],
    predictions: Sequence[Prediction],
    thresholds: Sequence[float],
    target_recall: float,
    max_fdr: float,
) -> List[Dict]:
    ordered_thresholds = sorted(thresholds)
    total_gt = sum(len(gts) for gts in gt_by_image)
    sorted_predictions = sorted(
        predictions,
        key=lambda pred: (-pred.conf, pred.image_index, pred.pred_index),
    )
    matched_by_image: List[set] = [set() for _ in gt_by_image]
    tp = 0
    fp = 0
    pred_cursor = 0
    rows_desc: List[Dict] = []

    for threshold in reversed(ordered_thresholds):
        while pred_cursor < len(sorted_predictions) and sorted_predictions[pred_cursor].conf >= threshold:
            is_tp = apply_prediction(sorted_predictions[pred_cursor], gt_by_image, matched_by_image)
            if is_tp:
                tp += 1
            else:
                fp += 1
            pred_cursor += 1
        rows_desc.append(build_metric_row(threshold, tp, fp, total_gt, target_recall, max_fdr))

    return list(reversed(rows_desc))


def compact_row(row: Optional[Dict]) -> Optional[Dict]:
    if row is None:
        return None
    keys = ("threshold", "tp", "fp", "fn", "gt_total", "prediction_count", "recall", "precision", "fdr")
    compact = {key: row[key] for key in keys}
    compact["meets_requirements"] = row["meets_requirements"]
    return compact


def build_summary(rows: Sequence[Dict], args: argparse.Namespace, metadata: Dict) -> Dict:
    feasible = [row for row in rows if row["meets_requirements"]]
    under_fdr = [row for row in rows if row["fdr"] <= args.max_fdr]
    over_recall = [row for row in rows if row["recall"] >= args.target_recall]

    best_feasible = None
    if feasible:
        best_feasible = max(feasible, key=lambda row: (row["recall"], -row["fdr"], row["threshold"]))

    best_recall_under_fdr = None
    if under_fdr:
        best_recall_under_fdr = max(under_fdr, key=lambda row: (row["recall"], -row["fdr"], row["threshold"]))

    best_fdr_with_recall = None
    if over_recall:
        best_fdr_with_recall = min(over_recall, key=lambda row: (row["fdr"], -row["recall"], -row["threshold"]))

    feasible_threshold_range = None
    if feasible:
        feasible_thresholds = [row["threshold"] for row in feasible]
        feasible_threshold_range = {
            "min": min(feasible_thresholds),
            "max": max(feasible_thresholds),
            "count": len(feasible_thresholds),
        }

    return {
        "meets_requirements": bool(feasible),
        "best_feasible": compact_row(best_feasible),
        "best_recall_under_fdr_20": compact_row(best_recall_under_fdr),
        "best_fdr_with_recall_85": compact_row(best_fdr_with_recall),
        "feasible_threshold_range": feasible_threshold_range,
        "target_recall": args.target_recall,
        "max_fdr": args.max_fdr,
        "metadata": metadata,
    }


def write_csv(rows: Sequence[Dict], path: Path) -> None:
    fieldnames = [
        "threshold",
        "tp",
        "fp",
        "fn",
        "gt_total",
        "prediction_count",
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
                    "gt_total": row["gt_total"],
                    "prediction_count": row["prediction_count"],
                    "recall": f"{row['recall']:.8f}",
                    "precision": f"{row['precision']:.8f}",
                    "fdr": f"{row['fdr']:.8f}",
                    "meets_requirements": int(row["meets_requirements"]),
                }
            )


def write_json(data: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.threshold_start is None:
        args.threshold_start = args.pred_conf
    if args.threshold_start + 1e-12 < args.pred_conf:
        raise ValueError("--threshold-start cannot be lower than --pred-conf because lower-confidence boxes were not collected.")
    if not 0.0 <= args.pred_conf <= 1.0:
        raise ValueError("--pred-conf must be between 0 and 1.")
    if not 0.0 <= args.pred_iou <= 1.0:
        raise ValueError("--pred-iou must be between 0 and 1.")
    if not 0.0 <= args.target_recall <= 1.0:
        raise ValueError("--target-recall must be between 0 and 1.")
    if not 0.0 <= args.max_fdr <= 1.0:
        raise ValueError("--max-fdr must be between 0 and 1.")


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
    image_paths = resolve_split_sources(data_yaml, cfg, args.split)
    thresholds = make_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)

    setup_ultralytics(ultralytics_root)
    print(f"model={model_path}", flush=True)
    print(f"data={data_yaml}", flush=True)
    print(f"split={args.split} images={len(image_paths)}", flush=True)
    print(
        "matching=class_id_required, class_24_iou_0.35, other_classes_iou_0.50",
        flush=True,
    )
    print(
        f"candidate_collection=conf>={args.pred_conf} nms_iou={args.pred_iou} max_det={args.max_det}",
        flush=True,
    )

    gt_by_image, predictions, missing_label_files = collect_predictions(
        model_path=model_path,
        image_paths=image_paths,
        data_root=data_root,
        split=args.split,
        args=args,
    )
    total_gt = sum(len(gts) for gts in gt_by_image)
    print(
        f"collected_images={len(gt_by_image)} gt_total={total_gt} "
        f"candidate_predictions={len(predictions)} missing_label_files={missing_label_files}",
        flush=True,
    )

    rows = evaluate_thresholds(
        gt_by_image=gt_by_image,
        predictions=predictions,
        thresholds=thresholds,
        target_recall=args.target_recall,
        max_fdr=args.max_fdr,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    summary_path = output_dir / args.summary_name
    metadata = {
        "model": str(model_path),
        "data": str(data_yaml),
        "data_root": str(data_root),
        "split": args.split,
        "images": len(image_paths),
        "gt_total": total_gt,
        "candidate_predictions": len(predictions),
        "missing_label_files": missing_label_files,
        "imgsz": args.imgsz,
        "pred_conf": args.pred_conf,
        "pred_iou": args.pred_iou,
        "max_det": args.max_det,
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
        "iou_policy": {
            "class_24_FSC": 0.35,
            "other_classes": 0.50,
        },
        "class_match_required": True,
    }
    summary = build_summary(rows, args, metadata)

    write_csv(rows, csv_path)
    write_json(summary, summary_path)
    print(f"csv={csv_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(f"meets_requirements={summary['meets_requirements']}", flush=True)
    print(f"best_feasible={summary['best_feasible']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
