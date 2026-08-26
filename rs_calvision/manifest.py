"""Trusted method manifest and model registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .domain import InferenceSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "configs" / "platform"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_project_asset(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"asset leaves active project: {relative_path}") from exc
    return path


@dataclass(frozen=True, slots=True)
class MethodManifest:
    raw: Mapping[str, Any]
    source_path: Path

    @property
    def method_id(self) -> str:
        return str(self.raw["method"]["id"])

    @property
    def display_name(self) -> str:
        return str(self.raw["method"]["display_name"])

    @property
    def version(self) -> str:
        return str(self.raw["method"]["version"])

    @property
    def checkpoint_path(self) -> Path:
        return resolve_project_asset(str(self.raw["base_detector"]["checkpoint_path"]))

    @property
    def threshold_path(self) -> Path:
        return resolve_project_asset(str(self.raw["threshold"]["file"]))

    @property
    def class_names(self) -> dict[int, str]:
        return {int(row["id"]): str(row["name"]) for row in self.raw["classes"]["fine_classes"]}

    @property
    def broad_by_class(self) -> dict[int, str]:
        return {int(key): str(value) for key, value in self.raw["classes"]["mappings"]["fine_to_broad"].items()}

    @property
    def thresholds(self) -> dict[int, float]:
        return {int(key): float(value) for key, value in self.raw["threshold"]["per_class_values"].items()}

    @property
    def inference_settings(self) -> InferenceSettings:
        cfg = self.raw["inference"]
        return InferenceSettings(
            input_size=int(cfg["input_size"]),
            tile_size=int(cfg["tile_size"]),
            stride=int(cfg["stride"]),
            pred_conf=float(cfg["pred_conf"]),
            max_det=int(cfg["max_det"]),
            global_nms_iou=float(cfg["global_nms_iou"]),
            batch=int(cfg["batch"]),
            precision=str(cfg["precision"]),
            device=str(cfg["device"]),
        )

    def verify_assets(self, verify_hashes: bool = True) -> None:
        for path in (self.checkpoint_path, self.threshold_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not verify_hashes:
            return
        checkpoint_hash = sha256_file(self.checkpoint_path)
        threshold_hash = sha256_file(self.threshold_path)
        if checkpoint_hash != str(self.raw["base_detector"]["checkpoint_hash"]).upper():
            raise ValueError("trusted checkpoint hash mismatch")
        if threshold_hash != str(self.raw["threshold"]["hash"]).upper():
            raise ValueError("trusted threshold hash mismatch")


class MethodRegistry:
    """Read-only registry of trusted, repository-owned method manifests."""

    def __init__(self, manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> None:
        self.manifest_dir = manifest_dir.resolve()
        self._methods: dict[str, MethodManifest] = {}
        self.reload()

    def reload(self) -> None:
        methods: dict[str, MethodManifest] = {}
        if not self.manifest_dir.is_dir():
            raise FileNotFoundError(self.manifest_dir)
        for path in sorted(self.manifest_dir.glob("*.json")):
            if path.name.endswith(".schema.json") or path.name == "prediction.schema.json":
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            if "method" not in raw or "base_detector" not in raw:
                continue
            manifest = MethodManifest(raw=raw, source_path=path)
            if manifest.method_id in methods:
                raise ValueError(f"duplicate method id: {manifest.method_id}")
            methods[manifest.method_id] = manifest
        if not methods:
            raise ValueError(f"no method manifests found in {self.manifest_dir}")
        self._methods = methods

    def list(self) -> tuple[MethodManifest, ...]:
        return tuple(self._methods[key] for key in sorted(self._methods))

    def get(self, method_id: str) -> MethodManifest:
        try:
            return self._methods[method_id]
        except KeyError as exc:
            raise KeyError(f"unregistered method: {method_id}") from exc

