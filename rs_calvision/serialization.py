"""Prediction artifact serialization and lightweight structural validation."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .domain import (
    BBox,
    FinalDetection,
    InferenceResult,
    PredictionArtifact,
    SCHEMA_VERSION,
    TimingResult,
    to_primitive,
    utc_now,
)
from .manifest import MethodManifest


PREDICTION_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "platform" / "prediction.schema.json"


def build_presentation_summary(prediction: dict, timing: dict | None) -> dict:
    """Build the compact, data-backed summary used by the report workbench."""
    final_detections = prediction.get("final_detections", [])
    broad_counts = Counter(item["broad_class"] for item in final_detections)
    return {
        "image": prediction["image"],
        "tile_count": prediction["tile_plan"]["tile_count"],
        "stage_counts": prediction["stage_counts"],
        "broad_class_counts": {
            "ship": broad_counts.get("ship", 0),
            "aircraft": broad_counts.get("aircraft", 0),
            "vehicle": broad_counts.get("vehicle", 0),
        },
        "timing": {
            "algorithm_pipeline_seconds": timing.get("algorithm_pipeline_seconds") if timing else None,
            "strict_e2e_seconds": timing.get("strict_e2e_seconds") if timing else None,
        },
    }


def bbox_dict(box: BBox) -> dict[str, float]:
    return {
        "x1": float(box.x1),
        "y1": float(box.y1),
        "x2": float(box.x2),
        "y2": float(box.y2),
        "width": float(box.width),
        "height": float(box.height),
    }


def raw_detection_dict(detection) -> dict[str, Any]:
    return {
        "id": detection.detection_id,
        "image_id": detection.image_id,
        "tile_id": detection.tile_id,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox_tile": bbox_dict(detection.bbox_tile_xyxy),
        "stage": detection.stage.value,
    }


def mapped_detection_dict(detection) -> dict[str, Any]:
    return {
        "id": detection.detection_id,
        "image_id": detection.image_id,
        "tile_id": detection.tile_id,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox_tile": bbox_dict(detection.bbox_tile_xyxy),
        "bbox_global": bbox_dict(detection.bbox_global_xyxy),
        "stage": detection.stage.value,
    }


def final_detection_dict(detection: FinalDetection) -> dict[str, Any]:
    box = detection.bbox_global_xyxy
    return {
        "id": detection.detection_id,
        "image_id": detection.image_id,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "broad_class": detection.broad_class,
        "confidence": detection.confidence,
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
        "width": box.width,
        "height": box.height,
        "bbox_global": bbox_dict(box),
        "source_tile_ids": list(detection.source_tile_ids),
        "source_detection_ids": list(detection.source_detection_ids),
        "stage": detection.stage.value,
        "kept_by_global_nms": detection.kept_by_global_nms,
        "kept_by_class_threshold": detection.kept_by_class_threshold,
    }


class PredictionSerializer:
    def build_artifact(
        self,
        result: InferenceResult,
        method: MethodManifest,
        created_at: str | None = None,
    ) -> PredictionArtifact:
        inference = method.inference_settings
        return PredictionArtifact(
            schema_version=SCHEMA_VERSION,
            run={
                "run_id": result.run_id,
                "created_at": created_at or utc_now(),
                "method_id": method.method_id,
                "method_version": method.version,
            },
            image=result.image,
            model={
                "method_display_name": method.display_name,
                "base_detector": method.raw["base_detector"]["family"],
                "checkpoint_path": str(method.raw["base_detector"]["checkpoint_path"]),
                "checkpoint_hash": str(method.raw["base_detector"]["checkpoint_hash"]),
                "end2end": bool(method.raw["base_detector"]["end2end"]),
                "inference_branch": method.raw["base_detector"]["inference_branch"],
                "detection_scales": list(method.raw["base_detector"]["detection_scales"]),
                "precision": inference.precision,
                "device": inference.device,
            },
            threshold={
                "file": str(method.raw["threshold"]["file"]),
                "hash": str(method.raw["threshold"]["hash"]),
                "per_class_values": {str(key): value for key, value in method.thresholds.items()},
            },
            tile_plan={
                "tile_size": inference.tile_size,
                "stride": inference.stride,
                "network_input_size": inference.input_size,
                "tile_count": len(result.tiles),
                "tile_transform": method.raw["inference"]["tile_transform"],
                "padding": any(tile.padding for tile in result.tiles),
                "global_class_wise_nms_iou": inference.global_nms_iou,
                "tile_iou_nms_active": False,
            },
            tiles=result.tiles,
            raw_detections=result.raw_detections,
            mapped_detections=result.mapped_detections,
            after_global_nms=result.after_global_nms,
            final_detections=result.final_detections,
            stage_counts=result.stage_counts,
            timing=result.timing,
        )

    def to_dict(self, artifact: PredictionArtifact) -> dict[str, Any]:
        data = {
            "schema_version": artifact.schema_version,
            "run": dict(artifact.run),
            "image": to_primitive(artifact.image),
            "model": to_primitive(artifact.model),
            "threshold": to_primitive(artifact.threshold),
            "tile_plan": to_primitive(artifact.tile_plan),
            "tiles": [to_primitive(tile) for tile in artifact.tiles],
            "raw_detections": [raw_detection_dict(item) for item in artifact.raw_detections],
            "mapped_detections": [mapped_detection_dict(item) for item in artifact.mapped_detections],
            "after_global_nms": [final_detection_dict(item) for item in artifact.after_global_nms],
            "final_detections": [final_detection_dict(item) for item in artifact.final_detections],
            "stage_counts": dict(artifact.stage_counts),
            "timing": to_primitive(artifact.timing),
        }
        self.validate_structure(data)
        self.validate_json_schema(data)
        return data

    @staticmethod
    def validate_json_schema(data: Mapping[str, Any]) -> None:
        """Apply the formal schema when the declared runtime dependency is present."""
        try:
            import jsonschema
        except ImportError:
            return
        schema = json.loads(PREDICTION_SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)

    @staticmethod
    def validate_structure(data: Mapping[str, Any]) -> None:
        required = {
            "schema_version", "run", "image", "model", "threshold", "tile_plan", "tiles",
            "raw_detections", "mapped_detections", "after_global_nms", "final_detections",
            "stage_counts", "timing",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"prediction artifact missing keys: {sorted(missing)}")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported prediction schema: {data['schema_version']}")
        counts = data["stage_counts"]
        expected_counts = {
            "raw": len(data["raw_detections"]),
            "mapped": len(data["mapped_detections"]),
            "after_global_nms": len(data["after_global_nms"]),
            "final": len(data["final_detections"]),
        }
        if any(int(counts.get(key, -1)) != value for key, value in expected_counts.items()):
            raise ValueError("stage_counts do not match detection arrays")
        for detection in data["final_detections"]:
            if not detection["kept_by_global_nms"] or not detection["kept_by_class_threshold"]:
                raise ValueError("final detection must pass NMS and class threshold")

    def write(self, artifact: PredictionArtifact, path: Path) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        data = self.to_dict(artifact)
        encoding_started = time.perf_counter()
        json.dumps(data, ensure_ascii=False, indent=2)
        encoding_seconds = time.perf_counter() - encoding_started
        data["timing"]["serialization_seconds"] = encoding_seconds
        data["timing"]["strict_e2e_seconds"] += encoding_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return data, time.perf_counter() - started

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        PredictionSerializer.validate_structure(data)
        return data
