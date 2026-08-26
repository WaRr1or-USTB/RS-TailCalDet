"""Headless large-image inference with explicit, observable pipeline stages."""

from __future__ import annotations

import sys
import struct
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from PIL import Image

from .domain import (
    BBox,
    CancellationCheck,
    DetectionStage,
    EventCallback,
    FinalDetection,
    ImageInfo,
    InferenceResult,
    MappedDetection,
    PipelineEvent,
    PipelineStage,
    RawDetection,
    TileInfo,
    TimingResult,
)
from .manifest import MethodManifest, PROJECT_ROOT, sha256_file


class InferenceCancelled(RuntimeError):
    """Raised cooperatively when a queued or running job is cancelled."""


@dataclass(frozen=True)
class RuntimeDetection:
    class_id: int
    confidence: float
    bbox: BBox


class RuntimeProtocol(Protocol):
    def load(self) -> None: ...

    def predict_tiles(
        self, tiles: Sequence[Image.Image]
    ) -> Sequence[Sequence[RuntimeDetection]]: ...


class ModelRuntimeAdapter:
    """Lazy, single-instance adapter around the project's Ultralytics runtime."""

    def __init__(self, manifest: MethodManifest):
        self.manifest = manifest
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self.manifest.verify_assets()
            source_root = PROJECT_ROOT / "ultralytics-main"
            source_text = str(source_root)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(str(self.manifest.checkpoint_path))

    def predict_tiles(
        self, tiles: Sequence[Image.Image]
    ) -> Sequence[Sequence[RuntimeDetection]]:
        self.load()
        settings = self.manifest.inference_settings
        results = self._model.predict(
            source=list(tiles),
            imgsz=settings.input_size,
            conf=settings.pred_conf,
            iou=0.7,
            max_det=settings.max_det,
            batch=len(tiles),
            device=settings.device,
            half=settings.precision == "fp16",
            verbose=False,
            save=False,
        )
        converted: list[list[RuntimeDetection]] = []
        for result in results:
            current: list[RuntimeDetection] = []
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for box, confidence, class_id in zip(xyxy, confidences, classes):
                    current.append(
                        RuntimeDetection(
                            class_id=int(class_id),
                            confidence=float(confidence),
                            bbox=BBox(*(float(value) for value in box)),
                        )
                    )
            converted.append(current)
        return converted


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Match the competition script's border-covering sliding-window plan."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    border = length - tile_size
    if starts[-1] != border:
        starts.append(border)
    return starts


def plan_tiles(
    width: int, height: int, tile_size: int, stride: int, input_size: int | None = None
) -> list[TileInfo]:
    tiles: list[TileInfo] = []
    index = 0
    for y in tile_starts(height, tile_size, stride):
        for x in tile_starts(width, tile_size, stride):
            tiles.append(
                TileInfo(
                    tile_id=f"tile-{index:06d}",
                    index=index,
                    offset_x=x,
                    offset_y=y,
                    tile_width=min(tile_size, width - x),
                    tile_height=min(tile_size, height - y),
                    network_input_width=input_size or tile_size,
                    network_input_height=input_size or tile_size,
                    scale_x=min(tile_size, width - x) / float(input_size or tile_size),
                    scale_y=min(tile_size, height - y) / float(input_size or tile_size),
                    is_boundary_tile=(x + tile_size >= width or y + tile_size >= height),
                )
            )
            index += 1
    return tiles


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _box_iou(left: BBox, right: BBox) -> float:
    # Preserve the float32 arithmetic used by the legacy NumPy implementation.
    width = _f32(max(0.0, _f32(min(_f32(left.x2), _f32(right.x2)) - max(_f32(left.x1), _f32(right.x1)))))
    height = _f32(max(0.0, _f32(min(_f32(left.y2), _f32(right.y2)) - max(_f32(left.y1), _f32(right.y1)))))
    intersection = _f32(width * height)
    left_area = _f32(_f32(_f32(left.x2) - _f32(left.x1)) * _f32(_f32(left.y2) - _f32(left.y1)))
    right_area = _f32(_f32(_f32(right.x2) - _f32(right.x1)) * _f32(_f32(right.y2) - _f32(right.y1)))
    denominator = _f32(_f32(_f32(left_area + right_area) - intersection) + 1e-9)
    return _f32(intersection / denominator)


def classwise_hard_nms(
    detections: Sequence[MappedDetection], iou_threshold: float
) -> tuple[list[int], dict[int, list[int]]]:
    """Return legacy-equivalent kept indices and suppression lineage."""
    kept: list[int] = []
    lineage: dict[int, list[int]] = {}
    classes = sorted({detection.class_id for detection in detections})
    for class_id in classes:
        indices = [
            index
            for index, detection in enumerate(detections)
            if detection.class_id == class_id
        ]
        if not indices:
            continue
        # NumPy's argsort()[::-1] used by the audited script reverses input order
        # for equal scores, hence the index tie-breaker below.
        order = sorted(indices, key=lambda index: (_f32(detections[index].confidence), index), reverse=True)
        while order:
            global_keep = order[0]
            kept.append(global_keep)
            lineage[global_keep] = [global_keep]
            if len(order) == 1:
                break
            remaining = order[1:]
            suppressed = [index for index in remaining if _box_iou(detections[global_keep].bbox_global_xyxy, detections[index].bbox_global_xyxy) > iou_threshold]
            lineage[global_keep].extend(suppressed)
            order = [index for index in remaining if index not in set(suppressed)]
    kept.sort(
        key=lambda index: (
            -detections[index].confidence,
            detections[index].class_id,
            detections[index].bbox_global_xyxy.x1,
            detections[index].bbox_global_xyxy.y1,
        )
    )
    return kept, lineage


def _emit(
    callback: EventCallback | None,
    run_id: str,
    stage: PipelineStage,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if callback is not None:
        progress_by_stage = {
            PipelineStage.IMAGE_LOADING: 0.05,
            PipelineStage.IMAGE_LOADED: 0.10,
            PipelineStage.TILE_PLANNING: 0.12,
            PipelineStage.TILING_READY: 0.15,
            PipelineStage.INFERENCE_STARTED: 0.18,
            PipelineStage.COORDINATE_MAPPING: 0.72,
            PipelineStage.GLOBAL_NMS: 0.76,
            PipelineStage.THRESHOLD_FILTERING: 0.79,
        }
        progress = progress_by_stage.get(stage, 0.0)
        if stage == PipelineStage.INFERENCE_PROGRESS and payload:
            total = max(1, int(payload.get("total_tiles", 1)))
            progress = 0.18 + 0.52 * int(payload.get("current_tile", 0)) / total
        callback(PipelineEvent.create(run_id, stage, progress, message, payload))


def _check_cancelled(check: CancellationCheck | None) -> None:
    if check is not None and check():
        raise InferenceCancelled("Inference cancelled by request")


class BigImageInferenceAdapter:
    """Stable headless inference API for one large optical remote-sensing image."""

    def __init__(self, manifest: MethodManifest, runtime: RuntimeProtocol | None = None):
        self.manifest = manifest
        self.runtime = runtime or ModelRuntimeAdapter(manifest)

    def infer(
        self,
        image_path: str | Path,
        run_id: str,
        event_callback: EventCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> InferenceResult:
        strict_start = time.perf_counter()
        settings = self.manifest.inference_settings
        path = Path(image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        _check_cancelled(cancellation_check)
        _emit(event_callback, run_id, PipelineStage.IMAGE_LOADING, "Loading source image")
        image_read_start = time.perf_counter()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image_read_seconds = time.perf_counter() - image_read_start
        image_info = ImageInfo(
            image_id=path.stem,
            filename=path.name,
            channels=3,
            file_size=path.stat().st_size,
            path=str(path),
            sha256=sha256_file(path),
            width=image.width,
            height=image.height,
            dataset="data_big_v1" if "data_big_v1" in path.parts else None,
            split=path.parent.name if "data_big_v1" in path.parts else None,
            proxy_big_image="data_big_v1" in path.parts,
        )
        _emit(event_callback, run_id, PipelineStage.IMAGE_LOADED, "Source image loaded", {"width": image.width, "height": image.height})

        stage_start = time.perf_counter()
        _emit(event_callback, run_id, PipelineStage.TILE_PLANNING, "Planning image tiles")
        tiles = plan_tiles(
            image.width, image.height, settings.tile_size, settings.stride, settings.input_size
        )
        tiling_seconds = time.perf_counter() - stage_start
        _emit(event_callback, run_id, PipelineStage.TILING_READY, "Tile plan ready", {"tile_count": len(tiles)})

        _check_cancelled(cancellation_check)
        load_start = time.perf_counter()
        self.runtime.load()
        model_loading_seconds = time.perf_counter() - load_start

        _emit(
            event_callback,
            run_id,
            PipelineStage.INFERENCE_STARTED,
            "Running tile inference",
            {"tile_count": len(tiles)},
        )
        model_seconds = 0.0
        mapping_seconds = 0.0
        raw: list[RawDetection] = []
        mapped: list[MappedDetection] = []
        for batch_start in range(0, len(tiles), settings.batch):
            _check_cancelled(cancellation_check)
            batch = tiles[batch_start : batch_start + settings.batch]
            crops = [
                image.crop((tile.offset_x, tile.offset_y, tile.offset_x + tile.tile_width, tile.offset_y + tile.tile_height))
                for tile in batch
            ]
            predict_start = time.perf_counter()
            predictions = self.runtime.predict_tiles(crops)
            model_seconds += time.perf_counter() - predict_start
            if len(predictions) != len(batch):
                raise RuntimeError("Runtime returned a different result count than tile count")
            map_start = time.perf_counter()
            for tile, tile_predictions in zip(batch, predictions):
                for prediction in tile_predictions:
                    local = prediction.bbox
                    if local.width <= 0.0 or local.height <= 0.0:
                        continue
                    raw_id = f"raw-{len(raw):08d}"
                    raw_detection = RawDetection(
                        detection_id=raw_id,
                        image_id=image_info.image_id,
                        tile_id=tile.tile_id,
                        class_id=prediction.class_id,
                        class_name=self.manifest.class_names[prediction.class_id],
                        confidence=prediction.confidence,
                        bbox_tile_xyxy=local,
                    )
                    raw.append(raw_detection)
                    global_box = BBox(
                        x1=max(0.0, min(float(image.width), local.x1 + tile.offset_x)),
                        y1=max(0.0, min(float(image.height), local.y1 + tile.offset_y)),
                        x2=max(0.0, min(float(image.width), local.x2 + tile.offset_x)),
                        y2=max(0.0, min(float(image.height), local.y2 + tile.offset_y)),
                    )
                    mapped.append(
                        MappedDetection(
                            detection_id=f"mapped-{len(mapped):08d}",
                            image_id=image_info.image_id,
                            tile_id=tile.tile_id,
                            class_id=prediction.class_id,
                            class_name=self.manifest.class_names[prediction.class_id],
                            confidence=prediction.confidence,
                            bbox_tile_xyxy=local,
                            bbox_global_xyxy=global_box,
                        )
                    )
            mapping_seconds += time.perf_counter() - map_start
            _emit(
                event_callback,
                run_id,
                PipelineStage.INFERENCE_PROGRESS,
                "Tile batch completed",
                {
                    "current_tile": min(batch_start + len(batch), len(tiles)),
                    "total_tiles": len(tiles),
                    "current_batch": batch_start // settings.batch + 1,
                    "detections_generated": len(raw),
                    "elapsed_seconds": time.perf_counter() - strict_start,
                },
            )

        _emit(
            event_callback,
            run_id,
            PipelineStage.COORDINATE_MAPPING,
            "Mapped tile detections to image coordinates",
            {"detection_count": len(mapped)},
        )
        _check_cancelled(cancellation_check)
        _emit(event_callback, run_id, PipelineStage.GLOBAL_NMS, "Applying global NMS")
        nms_start = time.perf_counter()
        kept_indices, lineage = classwise_hard_nms(
            mapped, settings.global_nms_iou
        )
        after_nms: list[FinalDetection] = []
        for index in kept_indices:
            source_indices = lineage[index]
            source_raw_ids = tuple(raw[value].detection_id for value in source_indices)
            source_tile_ids = tuple(
                dict.fromkeys(mapped[value].tile_id for value in source_indices)
            )
            detection = mapped[index]
            threshold = self.manifest.thresholds[detection.class_id]
            after_nms.append(
                FinalDetection(
                    detection_id=f"nms-{len(after_nms):08d}",
                    image_id=image_info.image_id,
                    class_id=detection.class_id,
                    class_name=self.manifest.class_names[detection.class_id],
                    broad_class=self.manifest.broad_by_class[detection.class_id],
                    confidence=detection.confidence,
                    bbox_global_xyxy=detection.bbox_global_xyxy,
                    stage=DetectionStage.AFTER_GLOBAL_NMS,
                    source_tile_ids=source_tile_ids,
                    source_detection_ids=source_raw_ids,
                    kept_by_global_nms=True,
                    kept_by_class_threshold=detection.confidence + 1e-12 >= threshold,
                )
            )
        nms_seconds = time.perf_counter() - nms_start
        _emit(
            event_callback,
            run_id,
            PipelineStage.GLOBAL_NMS,
            "Global NMS completed",
            {"detections_before_nms": len(mapped), "detections_after_nms": len(after_nms)},
        )

        _emit(
            event_callback,
            run_id,
            PipelineStage.THRESHOLD_FILTERING,
            "Applying per-class confidence thresholds",
        )
        threshold_start = time.perf_counter()
        final = tuple(
            replace(detection, stage=DetectionStage.FINAL)
            for detection in after_nms
            if detection.kept_by_class_threshold
        )
        threshold_seconds = time.perf_counter() - threshold_start
        _emit(
            event_callback,
            run_id,
            PipelineStage.THRESHOLD_FILTERING,
            "Class threshold filtering completed",
            {"detections_before_threshold": len(after_nms), "detections_final": len(final)},
        )
        algorithm_seconds = (
            tiling_seconds
            + model_seconds
            + mapping_seconds
            + nms_seconds
            + threshold_seconds
        )
        timing = TimingResult(
            image_read_seconds=image_read_seconds,
            tiling_preprocess_seconds=tiling_seconds,
            model_inference_seconds=model_seconds,
            coordinate_mapping_seconds=mapping_seconds,
            global_nms_seconds=nms_seconds,
            threshold_filter_seconds=threshold_seconds,
            algorithm_pipeline_seconds=algorithm_seconds,
            strict_e2e_seconds=time.perf_counter() - strict_start,
        )
        return InferenceResult(
            run_id=run_id,
            image=image_info,
            tiles=tuple(tiles),
            raw_detections=tuple(raw),
            mapped_detections=tuple(mapped),
            after_global_nms=tuple(after_nms),
            final_detections=final,
            timing=timing,
            stage_counts={
                "raw": len(raw),
                "mapped": len(mapped),
                "after_global_nms": len(after_nms),
                "final": len(final),
            },
        )
