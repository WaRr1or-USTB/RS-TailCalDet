"""Evaluation adapters for the four project scoring views."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .domain import (
    BBox,
    CountMetrics,
    EvaluationProtocol,
    FinalDetection,
    GroundTruth,
    MatchResult,
    MetricResult,
)
from .manifest import MethodManifest


def box_iou(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / (union + 1e-9)


def count_metrics(tp: int, fp: int, total_gt: int) -> CountMetrics:
    predictions = tp + fp
    return CountMetrics(
        tp=tp,
        fp=fp,
        fn=total_gt - tp,
        ground_truth=total_gt,
        predictions=predictions,
        recall=tp / total_gt if total_gt else 0.0,
        precision=tp / predictions if predictions else 0.0,
        fdr=fp / predictions if predictions else 0.0,
    )


class EvaluationAdapter:
    def __init__(self, manifest: MethodManifest):
        self.manifest = manifest
        policy = manifest.raw["evaluation"]["iou_rules"]
        self.iou_by_class = {24: float(policy["class_24_FSC"])}
        self.default_iou = float(policy["other_classes"])

    def load_yolo_labels(
        self, label_path: str | Path, image_id: str, width: int, height: int
    ) -> tuple[GroundTruth, ...]:
        path = Path(label_path)
        ground_truths: list[GroundTruth] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 YOLO label fields")
            class_id = int(values[0])
            center_x, center_y, box_width, box_height = map(float, values[1:])
            x1 = (center_x - box_width / 2.0) * width
            y1 = (center_y - box_height / 2.0) * height
            x2 = (center_x + box_width / 2.0) * width
            y2 = (center_y + box_height / 2.0) * height
            ground_truths.append(
                GroundTruth(
                    ground_truth_id=f"gt-{image_id}-{line_number:06d}",
                    image_id=image_id,
                    class_id=class_id,
                    class_name=self.manifest.class_names[class_id],
                    broad_class=self.manifest.broad_by_class[class_id],
                    bbox_global_xyxy=BBox(x1, y1, x2, y2),
                )
            )
        return tuple(ground_truths)

    def _match(
        self,
        protocol_id: str,
        predictions: Sequence[FinalDetection],
        ground_truths: Sequence[GroundTruth],
        require_exact_class: bool,
        allowed_classes: frozenset[int] | None = None,
    ) -> tuple[CountMetrics, tuple[MatchResult, ...]]:
        selected_gt = [
            item for item in ground_truths
            if allowed_classes is None or item.class_id in allowed_classes
        ]
        selected_predictions = [
            item for item in predictions
            if allowed_classes is None or item.class_id in allowed_classes
        ]
        selected_predictions.sort(key=lambda item: (-item.confidence, item.image_id, item.detection_id))
        gt_by_image: dict[str, list[GroundTruth]] = defaultdict(list)
        for item in selected_gt:
            gt_by_image[item.image_id].append(item)
        matched: dict[str, set[int]] = defaultdict(set)
        results: list[MatchResult] = []
        tp = 0
        for prediction in selected_predictions:
            best_index: int | None = None
            best_iou = -1.0
            required_for_best: float | None = None
            for index, truth in enumerate(gt_by_image[prediction.image_id]):
                if index in matched[prediction.image_id]:
                    continue
                if require_exact_class and prediction.class_id != truth.class_id:
                    continue
                iou = box_iou(prediction.bbox_global_xyxy, truth.bbox_global_xyxy)
                required = self.iou_by_class.get(truth.class_id, self.default_iou)
                if iou + 1e-12 >= required and iou > best_iou:
                    best_index, best_iou, required_for_best = index, iou, required
            if best_index is None:
                results.append(MatchResult(protocol_id, prediction.image_id, prediction.detection_id, None, prediction.class_id, None, None, "fp"))
                continue
            matched[prediction.image_id].add(best_index)
            truth = gt_by_image[prediction.image_id][best_index]
            tp += 1
            results.append(MatchResult(protocol_id, prediction.image_id, prediction.detection_id, truth.ground_truth_id, truth.class_id, best_iou, required_for_best, "tp"))
        matched_ids = {
            gt_by_image[image_id][index].ground_truth_id
            for image_id, indices in matched.items() for index in indices
        }
        for truth in selected_gt:
            if truth.ground_truth_id not in matched_ids:
                results.append(MatchResult(protocol_id, truth.image_id, None, truth.ground_truth_id, truth.class_id, None, self.iou_by_class.get(truth.class_id, self.default_iou), "fn"))
        return count_metrics(tp, len(selected_predictions) - tp, len(selected_gt)), tuple(results)

    def evaluate(
        self,
        predictions: Sequence[FinalDetection],
        ground_truths: Sequence[GroundTruth],
        protocols: Iterable[str] | None = None,
    ) -> dict[str, MetricResult]:
        requested = tuple(protocols or (item.value for item in EvaluationProtocol))
        strict_overall, strict_matches = self._match("strict_25", predictions, ground_truths, True)
        per_class: dict[str, CountMetrics] = {}
        for class_id, name in self.manifest.class_names.items():
            metrics, _ = self._match("strict_25", predictions, ground_truths, True, frozenset({class_id}))
            per_class[f"{class_id}:{name}"] = metrics
        broad_ids: dict[str, frozenset[int]] = {}
        for broad in sorted(set(self.manifest.broad_by_class.values())):
            broad_ids[broad] = frozenset(key for key, value in self.manifest.broad_by_class.items() if value == broad)
        broad_metrics: dict[str, CountMetrics] = {}
        for broad, class_ids in broad_ids.items():
            broad_metrics[broad], _ = self._match("official_by_category", predictions, ground_truths, False, class_ids)
        official_overall, official_matches = self._match("official_overall", predictions, ground_truths, False)
        macro: dict[str, Mapping[str, object]] = {}
        for broad, class_ids in broad_ids.items():
            rows = [per_class[f"{class_id}:{self.manifest.class_names[class_id]}"] for class_id in sorted(class_ids)]
            evaluated = [row for row in rows if row.ground_truth > 0]
            missing = [class_id for class_id, row in zip(sorted(class_ids), rows) if row.ground_truth == 0]
            macro[broad] = {
                "aggregation": "unweighted_mean_across_subclasses_with_ground_truth",
                "expected_subclass_count": len(rows),
                "evaluated_subclass_count": len(evaluated),
                "missing_ground_truth_class_ids": missing,
                "is_complete": not missing,
                "macro_recall": sum(row.recall for row in evaluated) / len(evaluated) if evaluated else None,
                "macro_fdr": sum(row.fdr for row in evaluated) / len(evaluated) if evaluated else None,
                "macro_precision": sum(row.precision for row in evaluated) / len(evaluated) if evaluated else None,
            }
        all_results = {
            "strict_25": MetricResult("strict_25", "Strict 25-subclass", True, strict_overall, per_class=per_class, matches=strict_matches, metadata=self._metadata()),
            "official_overall": MetricResult("official_overall", "Official overall", True, official_overall, matches=official_matches, metadata=self._metadata()),
            "official_by_category": MetricResult("official_by_category", "Official by broad category", True, None, broad_categories=broad_metrics, metadata=self._metadata()),
            "group_macro": MetricResult("group_macro", "Group macro", True, None, per_class=per_class, group_macro=macro, metadata=self._metadata()),
            "ultralytics_val": MetricResult("ultralytics_val", "Ultralytics val", False, None, unavailable_reason="No Ultralytics validation artifact was supplied; this protocol is not inferred from count matching."),
        }
        unknown = set(requested).difference(all_results)
        if unknown:
            raise ValueError(f"unknown evaluation protocols: {sorted(unknown)}")
        return {key: all_results[key] for key in requested}

    def _metadata(self) -> Mapping[str, object]:
        evaluation = self.manifest.raw["evaluation"]
        return {
            "iou_policy": {"default": self.default_iou, "per_ground_truth_class": {str(key): value for key, value in self.iou_by_class.items()}},
            "authority_source": evaluation["iou_rules"]["source"],
            "authority_verified": bool(evaluation["iou_rules"]["authority_verified"]),
        }
