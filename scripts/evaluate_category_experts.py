"""Evaluate category-wise experts on unknown, potentially mixed-category big images.

The HRSC model supplies ship (0-3) and vehicle (24) detections. The old160
model supplies aircraft (4-23) detections. Both models see every image; routing
is performed per prediction, never per image.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

from evaluate_big_images_sliding import (
    Candidate,
    build_protocol_metrics,
    build_summary,
    build_v16_ranking_inputs,
    class_aware_nms,
    collect_tile_predictions,
    load_class_thresholds,
    maybe_cuda_synchronize,
    normalize_class_names,
    resolve_project_path,
    setup_ultralytics,
    tile_positions,
    write_per_image_csv,
    write_threshold_csv,
    write_v16_category_csv,
)
from evaluate_official_thresholds import (
    GroundTruth,
    Prediction,
    box_iou_xyxy,
    evaluate_thresholds,
    label_path_for_image,
    load_ground_truths,
    load_yaml,
    resolve_data_root,
    resolve_split_sources,
)


HRSC_EXPERT_CLASSES = frozenset({0, 1, 2, 3, 24})
OLD160_EXPERT_CLASSES = frozenset(range(4, 24))


@dataclass(frozen=True)
class RoutedPrediction:
    prediction: Prediction
    source: str
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-hrsc", default="runs/detect/yolo26x_hrsc_ft_e40/weights/best.pt")
    parser.add_argument(
        "--model-old160",
        default="runs/detect/yolo26x_domain_finetune_lr1e4_e40/weights/best.pt",
    )
    parser.add_argument("--thresholds-hrsc", default="runs/detect/final_thresholds_hrsc.json")
    parser.add_argument(
        "--thresholds-old160",
        default="runs/detect/yolo26x_domain_finetune_lr1e4_e40_class_thresholds_r945/class_thresholds.json",
    )
    parser.add_argument("--data", default="data_big_v1/dataset.yaml")
    parser.add_argument("--ultralytics-root", default="ultralytics-main")
    parser.add_argument("--split", default="val_scene_grouped")
    parser.add_argument("--output-dir", default="runs/detect/category_experts_val_scene_grouped")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=800)
    parser.add_argument("--stride", type=int, default=800)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--pred-iou", type=float, default=0.70)
    parser.add_argument("--global-iou", type=float, default=0.70)
    parser.add_argument("--cross-class-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--max-fdr", type=float, default=0.20)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def route_expert_predictions(
    hrsc_predictions: Sequence[Prediction],
    old160_predictions: Sequence[Prediction],
    hrsc_thresholds: Dict[int, float],
    old160_thresholds: Dict[int, float],
) -> List[RoutedPrediction]:
    routed: List[RoutedPrediction] = []
    for prediction in hrsc_predictions:
        threshold = hrsc_thresholds.get(prediction.cls, 0.0)
        if prediction.cls in HRSC_EXPERT_CLASSES and prediction.conf + 1e-12 >= threshold:
            routed.append(RoutedPrediction(prediction, "hrsc", threshold))
    for prediction in old160_predictions:
        threshold = old160_thresholds.get(prediction.cls, 0.0)
        if prediction.cls in OLD160_EXPERT_CLASSES and prediction.conf + 1e-12 >= threshold:
            routed.append(RoutedPrediction(prediction, "old160", threshold))
    return routed


def resolve_overlapping_predictions(
    predictions: Sequence[RoutedPrediction], iou_threshold: float
) -> List[RoutedPrediction]:
    """Suppress near-identical cross-class/model boxes using raw model confidence."""
    ordered = sorted(
        predictions,
        key=lambda item: (
            -item.prediction.conf,
            item.prediction.cls,
            item.source,
        ),
    )
    kept: List[RoutedPrediction] = []
    for candidate in ordered:
        if any(box_iou_xyxy(candidate.prediction.box, item.prediction.box) >= iou_threshold for item in kept):
            continue
        kept.append(candidate)
    return kept


def write_predictions_csv(rows: Sequence[Dict], path: Path) -> None:
    fieldnames = ["image", "source", "class_id", "confidence", "threshold", "x1", "y1", "x2", "y2"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_two_models_big_image(model_hrsc, model_old160, image_path: Path, image_index: int, args):
    """Read and tile one image once, then run both experts on the shared tile batches."""
    read_started = time.perf_counter()
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    data_read_seconds = time.perf_counter() - read_started
    width, height = image.size
    positions = tile_positions(width, height, args.tile_size, args.stride)
    raw_hrsc: List[Candidate] = []
    raw_old160: List[Candidate] = []

    maybe_cuda_synchronize(args.device)
    started = time.perf_counter()
    try:
        for start in range(0, len(positions), args.batch):
            batch_positions = positions[start : start + args.batch]
            batch_tiles = [
                image.crop((x, y, x + args.tile_size, y + args.tile_size))
                for x, y in batch_positions
            ]
            for model, candidates in ((model_hrsc, raw_hrsc), (model_old160, raw_old160)):
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
                    collect_tile_predictions(
                        result,
                        offset_x,
                        offset_y,
                        tile.width,
                        width,
                        height,
                        candidates,
                    )
            for tile in batch_tiles:
                tile.close()
    finally:
        image.close()

    hrsc_after_nms = class_aware_nms(raw_hrsc, args.global_iou)
    old160_after_nms = class_aware_nms(raw_old160, args.global_iou)
    hrsc_predictions = [
        Prediction(image_index, index, candidate.cls, candidate.conf, candidate.box)
        for index, candidate in enumerate(hrsc_after_nms)
    ]
    old160_predictions = [
        Prediction(image_index, index, candidate.cls, candidate.conf, candidate.box)
        for index, candidate in enumerate(old160_after_nms)
    ]
    maybe_cuda_synchronize(args.device)
    seconds = time.perf_counter() - started
    return hrsc_predictions, old160_predictions, {
        "width": width,
        "height": height,
        "tiles": len(positions),
        "raw_candidates": len(raw_hrsc) + len(raw_old160),
        "data_read_seconds": data_read_seconds,
        "seconds": seconds,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.cross_class_iou <= 1.0:
        raise ValueError("--cross-class-iou must be between 0 and 1")

    model_hrsc_path = resolve_project_path(args.model_hrsc)
    model_old160_path = resolve_project_path(args.model_old160)
    thresholds_hrsc_path = resolve_project_path(args.thresholds_hrsc)
    thresholds_old160_path = resolve_project_path(args.thresholds_old160)
    data_yaml = resolve_project_path(args.data)
    output_dir = resolve_project_path(args.output_dir)
    for path in (model_hrsc_path, model_old160_path, thresholds_hrsc_path, thresholds_old160_path, data_yaml):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = load_yaml(data_yaml)
    class_names = normalize_class_names(cfg.get("names"), int(cfg.get("nc", 0)))
    data_root = resolve_data_root(data_yaml, cfg)
    image_paths = resolve_split_sources(data_yaml, cfg, args.split)
    hrsc_thresholds = load_class_thresholds(thresholds_hrsc_path)
    old160_thresholds = load_class_thresholds(thresholds_old160_path)

    setup_ultralytics(resolve_project_path(args.ultralytics_root))
    from ultralytics import YOLO

    model_hrsc = YOLO(str(model_hrsc_path))
    model_old160 = YOLO(str(model_old160_path))
    infer = argparse.Namespace(
        imgsz=args.imgsz,
        tile_size=args.tile_size,
        stride=args.stride,
        pred_conf=args.pred_conf,
        pred_iou=args.pred_iou,
        global_iou=args.global_iou,
        max_det=args.max_det,
        batch=args.batch,
        device=args.device,
        half=args.half,
        tta=False,
    )

    gt_by_image: List[List[GroundTruth]] = []
    predictions: List[Prediction] = []
    prediction_rows: List[Dict] = []
    per_image_stats: List[Dict] = []

    print(f"model_hrsc={model_hrsc_path}", flush=True)
    print(f"model_old160={model_old160_path}", flush=True)
    print(f"split={args.split} images={len(image_paths)}", flush=True)
    for image_index, image_path in enumerate(image_paths):
        wall_started = time.perf_counter()
        hrsc_predictions, old160_predictions, inference_stats = predict_two_models_big_image(
            model_hrsc, model_old160, image_path, image_index, infer
        )
        routed = route_expert_predictions(
            hrsc_predictions,
            old160_predictions,
            hrsc_thresholds,
            old160_thresholds,
        )
        routed_before_conflict = len(routed)
        routed = resolve_overlapping_predictions(routed, args.cross_class_iou)

        image_predictions: List[Prediction] = []
        for item in routed:
            source = item.prediction
            prediction = Prediction(
                image_index=image_index,
                pred_index=len(predictions) + len(image_predictions),
                cls=source.cls,
                conf=source.conf,
                box=source.box,
            )
            image_predictions.append(prediction)
            x1, y1, x2, y2 = prediction.box
            prediction_rows.append(
                {
                    "image": image_path.name,
                    "source": item.source,
                    "class_id": prediction.cls,
                    "confidence": f"{prediction.conf:.8f}",
                    "threshold": f"{item.threshold:.8f}",
                    "x1": f"{x1:.3f}",
                    "y1": f"{y1:.3f}",
                    "x2": f"{x2:.3f}",
                    "y2": f"{y2:.3f}",
                }
            )
        predictions.extend(image_predictions)

        label_path = label_path_for_image(image_path, data_root, args.split)
        image_gts = load_ground_truths(label_path, inference_stats["width"], inference_stats["height"])
        gt_by_image.append(image_gts)
        inference_seconds = inference_stats["seconds"]
        wall_seconds = time.perf_counter() - wall_started
        per_image_stats.append(
            {
                "image": str(image_path),
                "width": inference_stats["width"],
                "height": inference_stats["height"],
                "tiles": inference_stats["tiles"],
                "ground_truth": len(image_gts),
                "raw_candidates": inference_stats["raw_candidates"],
                "predictions_after_nms": len(image_predictions),
                "data_read_seconds": inference_stats["data_read_seconds"],
                "seconds": inference_seconds,
                "end_to_end_seconds": wall_seconds,
            }
        )
        print(
            f"image={image_index + 1}/{len(image_paths)} hrsc={len(hrsc_predictions)} "
            f"old160={len(old160_predictions)} routed={routed_before_conflict} "
            f"kept={len(image_predictions)} seconds={inference_seconds:.3f} "
            f"end_to_end={wall_seconds:.3f}",
            flush=True,
        )

    rows = evaluate_thresholds(
        gt_by_image=gt_by_image,
        predictions=predictions,
        thresholds=[args.pred_conf],
        target_recall=args.target_recall,
        max_fdr=args.max_fdr,
    )
    metadata = {
        "ground_truth": sum(len(gts) for gts in gt_by_image),
        "candidate_predictions": sum(row["raw_candidates"] for row in per_image_stats),
        "predictions_after_nms": len(predictions),
        "predictions_after_class_thresholds": len(predictions),
        "class_threshold_filter_enabled": True,
        "strategy": "HRSC classes 0-3,24; old160 classes 4-23; raw-confidence cross-class suppression",
        "model_hrsc": str(model_hrsc_path),
        "model_old160": str(model_old160_path),
        "thresholds_hrsc": str(thresholds_hrsc_path),
        "thresholds_old160": str(thresholds_old160_path),
        "cross_class_iou": args.cross_class_iou,
        "timing_definition": "One shared image read/tile stream, two model passes, and merged postprocess; image read excluded.",
    }
    summary = build_summary(rows, per_image_stats, args, metadata)
    end_to_end_times = [float(row["end_to_end_seconds"]) for row in per_image_stats]
    summary["end_to_end_timing"] = {
        "description": "One image read, shared tile preparation, both model passes, and merge/output preparation.",
        "max_seconds": max(end_to_end_times) if end_to_end_times else 0.0,
        "mean_seconds": sum(end_to_end_times) / len(end_to_end_times) if end_to_end_times else 0.0,
        "images_over_limit": sum(value > args.time_limit for value in end_to_end_times),
    }
    protocol = build_protocol_metrics(
        gt_by_image,
        predictions,
        args.pred_conf,
        args.target_recall,
        args.max_fdr,
        class_names,
    )
    ranking = build_v16_ranking_inputs(protocol["v16_category_macro"], per_image_stats)
    protocol["v16_ranking_inputs"] = ranking
    summary["protocol_metrics"] = protocol
    summary["metadata"] = metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    write_threshold_csv(rows, output_dir / "threshold_metrics.csv")
    write_per_image_csv(per_image_stats, output_dir / "per_image_times.csv")
    write_v16_category_csv(protocol["v16_category_macro"], ranking, output_dir / "v16_category_metrics.csv")
    write_predictions_csv(prediction_rows, output_dir / "predictions.csv")
    (output_dir / "protocol_metrics.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"output_dir={output_dir}", flush=True)
    print(f"official_overall={protocol['official_overall']}", flush=True)
    print(f"official_by_category={protocol['official_by_category']['categories']}", flush=True)
    print(f"strict_25_subclass={protocol['strict_25_subclass']['overall']}", flush=True)
    print(f"timing={summary['timing']}", flush=True)
    print(f"end_to_end_timing={summary['end_to_end_timing']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
