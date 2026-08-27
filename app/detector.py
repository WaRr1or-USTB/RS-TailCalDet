from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
from ultralytics import YOLO

from inference_core import Candidate, apply_class_thresholds, class_aware_nms, clip_box, load_class_thresholds, tile_positions


TILE_SIZE = 800
STRIDE = 800
MODEL_IMAGE_SIZE = 1024
PREDICTION_CONFIDENCE = 0.001
PREDICTION_IOU = 0.70
GLOBAL_NMS_IOU = 0.70
MAX_DETECTIONS_PER_TILE = 300
EXPECTED_CLASS_COUNT = 25


def normalize_names(raw_names) -> Dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(class_id): str(name) for class_id, name in raw_names.items()}
    return {class_id: str(name) for class_id, name in enumerate(raw_names)}


class Detector:
    def __init__(self, model_path: Path, thresholds_path: Path) -> None:
        self.thresholds = load_class_thresholds(thresholds_path)
        expected_ids = set(range(EXPECTED_CLASS_COUNT))
        if set(self.thresholds) != expected_ids:
            missing = sorted(expected_ids - set(self.thresholds))
            extra = sorted(set(self.thresholds) - expected_ids)
            raise ValueError(f"threshold class IDs must be 0..24; missing={missing}, extra={extra}")

        self.model = YOLO(str(model_path))
        self.class_names = normalize_names(self.model.names)
        if set(self.class_names) != expected_ids:
            raise ValueError(f"model must expose exactly 25 classes, got IDs {sorted(self.class_names)}")

        torch.backends.cudnn.benchmark = True

    def predict(self, image: Image.Image) -> List[dict]:
        width, height = image.size
        raw_candidates: List[Candidate] = []

        for offset_x, offset_y in tile_positions(width, height, TILE_SIZE, STRIDE):
            tile = image.crop((offset_x, offset_y, offset_x + TILE_SIZE, offset_y + TILE_SIZE))
            result = self.model.predict(
                source=[tile],
                imgsz=MODEL_IMAGE_SIZE,
                conf=PREDICTION_CONFIDENCE,
                iou=PREDICTION_IOU,
                max_det=MAX_DETECTIONS_PER_TILE,
                batch=1,
                device=0,
                half=False,
                verbose=False,
                save=False,
            )[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box, confidence, class_id in zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
            ):
                full_box = clip_box(box, offset_x, offset_y, width, height)
                if full_box[2] <= full_box[0] or full_box[3] <= full_box[1]:
                    continue
                raw_candidates.append(
                    Candidate(cls=int(class_id), conf=float(confidence), box=full_box)
                )

        after_nms = class_aware_nms(raw_candidates, GLOBAL_NMS_IOU)
        final_candidates = apply_class_thresholds(after_nms, self.thresholds)
        torch.cuda.synchronize()

        return [
            {
                "category_id": candidate.cls,
                "category_name": self.class_names[candidate.cls],
                "score": round(candidate.conf, 8),
                "bbox": [round(value, 4) for value in candidate.box],
            }
            for candidate in final_candidates
        ]
