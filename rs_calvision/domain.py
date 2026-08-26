"""Framework-independent domain schemas for RS-CalVision.

These dataclasses deliberately do not depend on FastAPI or Pydantic.  They are
JSON-compatible through :func:`to_primitive` and can be wrapped by any transport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_primitive(value: Any) -> Any:
    """Recursively convert domain values into JSON-compatible primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_primitive(item) for item in value]
    return value


class DetectionStage(str, Enum):
    RAW = "raw"
    MAPPED = "mapped"
    AFTER_GLOBAL_NMS = "after_global_nms"
    FINAL = "final"


class PipelineStage(str, Enum):
    QUEUED = "queued"
    IMAGE_LOADING = "image_loading"
    IMAGE_LOADED = "image_loaded"
    TILE_PLANNING = "tile_planning"
    TILING_READY = "tiling_ready"
    INFERENCE_STARTED = "inference_started"
    INFERENCE_PROGRESS = "inference_progress"
    COORDINATE_MAPPING = "coordinate_mapping"
    GLOBAL_NMS = "global_nms"
    THRESHOLD_FILTERING = "threshold_filtering"
    PREDICTION_SERIALIZATION = "prediction_serialization"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationProtocol(str, Enum):
    STRICT_25 = "strict_25"
    OFFICIAL_OVERALL = "official_overall"
    OFFICIAL_BY_CATEGORY = "official_by_category"
    GROUP_MACRO = "group_macro"
    ULTRALYTICS_VAL = "ultralytics_val"


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid xyxy box: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True, slots=True)
class ImageInfo:
    image_id: str
    filename: str
    width: int
    height: int
    channels: int
    file_size: int
    path: str | None = None
    source: str | None = None
    sha256: str | None = None
    dataset: str | None = None
    split: str | None = None
    proxy_big_image: bool = False


@dataclass(frozen=True, slots=True)
class TileInfo:
    tile_id: str
    index: int
    offset_x: int
    offset_y: int
    tile_width: int
    tile_height: int
    network_input_width: int
    network_input_height: int
    scale_x: float
    scale_y: float
    is_boundary_tile: bool
    padding: bool = False


@dataclass(frozen=True, slots=True)
class RawDetection:
    detection_id: str
    image_id: str
    tile_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_tile_xyxy: BBox
    stage: DetectionStage = DetectionStage.RAW


@dataclass(frozen=True, slots=True)
class MappedDetection:
    detection_id: str
    image_id: str
    tile_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_tile_xyxy: BBox
    bbox_global_xyxy: BBox
    stage: DetectionStage = DetectionStage.MAPPED


@dataclass(frozen=True, slots=True)
class FinalDetection:
    detection_id: str
    image_id: str
    class_id: int
    class_name: str
    broad_class: str
    confidence: float
    bbox_global_xyxy: BBox
    source_tile_ids: tuple[str, ...]
    source_detection_ids: tuple[str, ...]
    stage: DetectionStage
    kept_by_global_nms: bool
    kept_by_class_threshold: bool


@dataclass(frozen=True, slots=True)
class GroundTruth:
    ground_truth_id: str
    image_id: str
    class_id: int
    class_name: str
    broad_class: str
    bbox_global_xyxy: BBox


@dataclass(frozen=True, slots=True)
class MatchResult:
    protocol_id: str
    image_id: str
    detection_id: str | None
    ground_truth_id: str | None
    class_id: int | None
    iou: float | None
    iou_threshold: float | None
    outcome: str


@dataclass(frozen=True, slots=True)
class CountMetrics:
    tp: int
    fp: int
    fn: int
    ground_truth: int
    predictions: int
    recall: float
    precision: float
    fdr: float


@dataclass(frozen=True, slots=True)
class MetricResult:
    protocol_id: str
    label: str
    available: bool
    overall: CountMetrics | None
    per_class: Mapping[str, CountMetrics] = field(default_factory=dict)
    broad_categories: Mapping[str, CountMetrics] = field(default_factory=dict)
    group_macro: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    matches: tuple[MatchResult, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TimingResult:
    image_read_seconds: float = 0.0
    tiling_preprocess_seconds: float = 0.0
    model_inference_seconds: float = 0.0
    coordinate_mapping_seconds: float = 0.0
    global_nms_seconds: float = 0.0
    threshold_filter_seconds: float = 0.0
    serialization_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    algorithm_pipeline_seconds: float = 0.0
    strict_e2e_seconds: float = 0.0
    web_task_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    input_size: int
    tile_size: int
    stride: int
    pred_conf: float
    max_det: int
    global_nms_iou: float
    batch: int
    precision: str
    device: str


@dataclass(frozen=True, slots=True)
class RunConfig:
    method_id: str
    image_path: str
    label_path: str | None
    inference: InferenceSettings
    evaluation_protocols: tuple[str, ...]
    dataset: str | None = None
    split: str | None = None
    proxy_big_image: bool = False


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    created_at: str
    updated_at: str
    status: RunStatus
    method_id: str
    method_version: str
    image: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    threshold: Mapping[str, Any]
    code_version: str | None
    device: str
    precision: str
    tile_config: Mapping[str, Any]
    evaluation_protocols: tuple[str, ...]
    artifact_index: Mapping[str, str]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    run_id: str
    stage: PipelineStage
    timestamp: str
    progress: float
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        run_id: str,
        stage: PipelineStage,
        progress: float,
        message: str,
        payload: Mapping[str, Any] | None = None,
    ) -> "PipelineEvent":
        return cls(
            run_id=run_id,
            stage=stage,
            timestamp=utc_now(),
            progress=min(max(float(progress), 0.0), 1.0),
            message=message,
            payload=payload or {},
        )


EventCallback = Callable[[PipelineEvent], None]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class InferenceResult:
    run_id: str
    image: ImageInfo
    tiles: tuple[TileInfo, ...]
    raw_detections: tuple[RawDetection, ...]
    mapped_detections: tuple[MappedDetection, ...]
    after_global_nms: tuple[FinalDetection, ...]
    final_detections: tuple[FinalDetection, ...]
    timing: TimingResult
    stage_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    schema_version: str
    run: Mapping[str, Any]
    image: ImageInfo
    model: Mapping[str, Any]
    threshold: Mapping[str, Any]
    tile_plan: Mapping[str, Any]
    tiles: tuple[TileInfo, ...]
    raw_detections: tuple[RawDetection, ...]
    mapped_detections: tuple[MappedDetection, ...]
    after_global_nms: tuple[FinalDetection, ...]
    final_detections: tuple[FinalDetection, ...]
    stage_counts: Mapping[str, int]
    timing: TimingResult

