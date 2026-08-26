"""RS-CalVision headless inference, evaluation, and artifact core."""

from .domain import (
    BBox,
    FinalDetection,
    GroundTruth,
    ImageInfo,
    MappedDetection,
    MetricResult,
    PipelineEvent,
    RawDetection,
    RunConfig,
    RunManifest,
    TileInfo,
    TimingResult,
)
from .manifest import MethodManifest, MethodRegistry

__all__ = [
    "BBox",
    "FinalDetection",
    "GroundTruth",
    "ImageInfo",
    "MappedDetection",
    "MethodManifest",
    "MethodRegistry",
    "MetricResult",
    "PipelineEvent",
    "RawDetection",
    "RunConfig",
    "RunManifest",
    "TileInfo",
    "TimingResult",
]

__version__ = "0.1.0"
