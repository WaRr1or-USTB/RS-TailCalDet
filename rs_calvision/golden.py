"""Golden-regression comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .domain import FinalDetection


@dataclass(frozen=True, slots=True)
class GoldenTolerance:
    confidence: float = 1e-6
    coordinate: float = 1e-3
    metric: float = 1e-9


def normalized_detection(item: Any) -> tuple[int, float, tuple[float, float, float, float]]:
    if isinstance(item, FinalDetection):
        return item.class_id, item.confidence, item.bbox_global_xyxy.as_xyxy()
    if hasattr(item, "cls"):
        class_id = int(item.cls)
        confidence = float(item.conf)
        box = tuple(float(value) for value in item.box)
    else:
        class_id = int(item[0])
        confidence = float(item[1])
        box = tuple(float(value) for value in item[2])
    return class_id, confidence, box  # type: ignore[return-value]


def compare_detections(
    legacy: Sequence[Any], platform: Sequence[Any], tolerance: GoldenTolerance = GoldenTolerance()
) -> dict[str, Any]:
    legacy_rows = [normalized_detection(item) for item in legacy]
    platform_rows = [normalized_detection(item) for item in platform]
    differences: list[dict[str, Any]] = []
    if len(legacy_rows) != len(platform_rows):
        differences.append({"field": "count", "legacy": len(legacy_rows), "platform": len(platform_rows)})
    for index, (left, right) in enumerate(zip(legacy_rows, platform_rows)):
        if left[0] != right[0]:
            differences.append({"index": index, "field": "class_id", "legacy": left[0], "platform": right[0]})
        if abs(left[1] - right[1]) > tolerance.confidence:
            differences.append({"index": index, "field": "confidence", "legacy": left[1], "platform": right[1]})
        for coordinate, (legacy_value, platform_value) in enumerate(zip(left[2], right[2])):
            if abs(legacy_value - platform_value) > tolerance.coordinate:
                differences.append({"index": index, "field": f"bbox[{coordinate}]", "legacy": legacy_value, "platform": platform_value})
    return {
        "passed": not differences,
        "legacy_count": len(legacy_rows),
        "platform_count": len(platform_rows),
        "tolerance": {"confidence": tolerance.confidence, "coordinate": tolerance.coordinate},
        "differences": differences,
    }


def compare_metric_values(
    legacy: Any, platform: Any, path: str = "metrics", tolerance: float = 1e-9
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(legacy, dict) and isinstance(platform, dict):
        for key in sorted(set(legacy) | set(platform)):
            if key not in legacy or key not in platform:
                differences.append({"path": f"{path}.{key}", "legacy": legacy.get(key), "platform": platform.get(key)})
            else:
                differences.extend(compare_metric_values(legacy[key], platform[key], f"{path}.{key}", tolerance))
    elif isinstance(legacy, (int, float)) and isinstance(platform, (int, float)):
        if abs(float(legacy) - float(platform)) > tolerance:
            differences.append({"path": path, "legacy": legacy, "platform": platform})
    elif legacy != platform:
        differences.append({"path": path, "legacy": legacy, "platform": platform})
    return differences
