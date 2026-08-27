from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Candidate:
    cls: int
    conf: float
    box: Box


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


def tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_positions(width: int, height: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    return [
        (x, y)
        for y in tile_starts(height, tile_size, stride)
        for x in tile_starts(width, tile_size, stride)
    ]


def clip_box(box: Sequence[float], offset_x: int, offset_y: int, width: int, height: int) -> Box:
    x1 = min(max(float(box[0]) + offset_x, 0.0), float(width))
    y1 = min(max(float(box[1]) + offset_y, 0.0), float(height))
    x2 = min(max(float(box[2]) + offset_x, 0.0), float(width))
    y2 = min(max(float(box[3]) + offset_y, 0.0), float(height))
    return x1, y1, x2, y2


def box_iou(left: Box, right: Box) -> float:
    inter_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    inter_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = inter_width * inter_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _nms_numpy(boxes, scores, iou_threshold: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        other_boxes = boxes[rest]
        x1 = np.maximum(boxes[current, 0], other_boxes[:, 0])
        y1 = np.maximum(boxes[current, 1], other_boxes[:, 1])
        x2 = np.minimum(boxes[current, 2], other_boxes[:, 2])
        y2 = np.minimum(boxes[current, 3], other_boxes[:, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        current_area = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
            0.0, boxes[current, 3] - boxes[current, 1]
        )
        other_areas = np.maximum(0.0, other_boxes[:, 2] - other_boxes[:, 0]) * np.maximum(
            0.0, other_boxes[:, 3] - other_boxes[:, 1]
        )
        union = current_area + other_areas - intersection
        ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)
        order = rest[ious <= iou_threshold]
    return keep


def class_aware_nms(candidates: Sequence[Candidate], iou_threshold: float) -> List[Candidate]:
    if np is not None and candidates:
        boxes = np.asarray([candidate.box for candidate in candidates], dtype=np.float32)
        scores = np.asarray([candidate.conf for candidate in candidates], dtype=np.float32)
        classes = np.asarray([candidate.cls for candidate in candidates], dtype=np.int32)
        keep_indices: List[int] = []
        for class_id in sorted(np.unique(classes).tolist()):
            class_indices = np.where(classes == class_id)[0]
            class_keep = _nms_numpy(boxes[class_indices], scores[class_indices], iou_threshold)
            keep_indices.extend(int(class_indices[index]) for index in class_keep)
        keep_indices.sort(
            key=lambda index: (-candidates[index].conf, candidates[index].cls, candidates[index].box)
        )
        return [candidates[index] for index in keep_indices]

    kept: List[Candidate] = []
    class_ids = sorted({candidate.cls for candidate in candidates})
    for class_id in class_ids:
        pending = sorted(
            (candidate for candidate in candidates if candidate.cls == class_id),
            key=lambda candidate: (-candidate.conf, candidate.box),
        )
        while pending:
            current = pending.pop(0)
            kept.append(current)
            pending = [candidate for candidate in pending if box_iou(current.box, candidate.box) <= iou_threshold]
    return sorted(kept, key=lambda candidate: (-candidate.conf, candidate.cls, candidate.box))


def apply_class_thresholds(
    candidates: Iterable[Candidate], thresholds: Dict[int, float]
) -> List[Candidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.conf + 1e-12 >= thresholds.get(candidate.cls, 0.0)
    ]
