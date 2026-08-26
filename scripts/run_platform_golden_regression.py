"""Run real-checkpoint golden regression against the audited legacy pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_big_images_sliding import apply_class_thresholds, build_protocol_metrics, predict_big_image  # noqa: E402
from evaluate_official_thresholds import load_ground_truths  # noqa: E402
from rs_calvision.domain import to_primitive  # noqa: E402
from rs_calvision.evaluation import EvaluationAdapter  # noqa: E402
from rs_calvision.golden import compare_detections, compare_metric_values  # noqa: E402
from rs_calvision.inference import BigImageInferenceAdapter, ModelRuntimeAdapter  # noqa: E402
from rs_calvision.manifest import MethodRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="At least three project-local large images")
    parser.add_argument("--method", default="rs-tailcaldet-v1")
    parser.add_argument("--output", default="runs/platform/golden-regression.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.images) < 3:
        raise ValueError("golden regression requires at least three representative images")
    case_names = [value.lower() for value in args.images]
    if not any("scene_grouped" in value or "mixed" in value for value in case_names):
        raise ValueError("golden regression requires a scene_grouped or mixed case")
    if not any("sparse" in value for value in case_names):
        raise ValueError("golden regression requires a sparse case")
    method = MethodRegistry().get(args.method)
    settings = method.inference_settings
    runtime = ModelRuntimeAdapter(method)
    runtime.load()
    legacy_args = SimpleNamespace(
        tile_size=settings.tile_size, stride=settings.stride, batch=settings.batch,
        imgsz=settings.input_size, pred_conf=settings.pred_conf, pred_iou=0.70,
        global_iou=settings.global_nms_iou, max_det=settings.max_det,
        device=settings.device, half=settings.precision == "fp16", tta=False,
    )
    rows = []
    legacy_all = []
    legacy_gt_by_image = []
    platform_all = []
    platform_ground_truths = []
    for index, raw_path in enumerate(args.images):
        path = (PROJECT_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        path.relative_to(PROJECT_ROOT)
        legacy, legacy_stats = predict_big_image(runtime._model, path, index, legacy_args)
        legacy_final = apply_class_thresholds(legacy, method.thresholds)
        platform = BigImageInferenceAdapter(method, runtime).infer(path, f"golden-{index}")
        legacy_all.extend(legacy_final)
        platform_all.extend(platform.final_detections)
        label_parts = list(path.parts)
        if "images" in label_parts:
            label_parts[len(label_parts) - 1 - label_parts[::-1].index("images")] = "labels"
        label_path = Path(*label_parts).with_suffix(".txt")
        if label_path.is_file():
            legacy_gt_by_image.append(load_ground_truths(label_path, platform.image.width, platform.image.height))
            platform_ground_truths.extend(EvaluationAdapter(method).load_yolo_labels(label_path, platform.image.image_id, platform.image.width, platform.image.height))
        else:
            legacy_gt_by_image.append([])
        comparison = compare_detections(legacy_final, platform.final_detections)
        comparison.update({
            "image": str(path),
            "raw_count_legacy": legacy_stats["raw_candidates"],
            "raw_count_platform": len(platform.raw_detections),
            "after_nms_count_legacy": legacy_stats["predictions_after_nms"],
            "after_nms_count_platform": len(platform.after_global_nms),
        })
        comparison["passed"] = comparison["passed"] and comparison["raw_count_legacy"] == comparison["raw_count_platform"] and comparison["after_nms_count_legacy"] == comparison["after_nms_count_platform"]
        rows.append(comparison)
    metric_regression = None
    if platform_ground_truths:
        legacy_metrics = build_protocol_metrics(
            legacy_gt_by_image, legacy_all, 0.0, 0.85, 0.20, method.class_names
        )
        platform_metrics = EvaluationAdapter(method).evaluate(platform_all, platform_ground_truths)
        legacy_subset = {
            "official_overall": legacy_metrics["official_overall"],
            "official_by_category": legacy_metrics["official_by_category"]["categories"],
            "strict_25": legacy_metrics["strict_25_subclass"]["overall"],
        }
        platform_subset = {
            "official_overall": to_primitive(platform_metrics["official_overall"].overall),
            "official_by_category": to_primitive(platform_metrics["official_by_category"].broad_categories),
            "strict_25": to_primitive(platform_metrics["strict_25"].overall),
        }
        comparable_keys = {"tp", "fp", "fn", "ground_truth", "predictions", "recall", "precision", "fdr"}
        legacy_subset["official_overall"] = {key: value for key, value in legacy_subset["official_overall"].items() if key in comparable_keys}
        legacy_subset["strict_25"] = {key: value for key, value in legacy_subset["strict_25"].items() if key in comparable_keys}
        legacy_subset["official_by_category"] = {
            category: {key: value for key, value in metrics.items() if key in comparable_keys}
            for category, metrics in legacy_subset["official_by_category"].items()
        }
        differences = compare_metric_values(legacy_subset, platform_subset)
        metric_regression = {"passed": not differences, "differences": differences}
    report = {
        "method_id": method.method_id,
        "checkpoint_hash": method.raw["base_detector"]["checkpoint_hash"],
        "threshold_hash": method.raw["threshold"]["hash"],
        "passed": all(row["passed"] for row in rows) and (metric_regression is None or metric_regression["passed"]),
        "cases": rows,
        "metric_regression": metric_regression,
    }
    output = (PROJECT_ROOT / args.output).resolve()
    output.relative_to(PROJECT_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
