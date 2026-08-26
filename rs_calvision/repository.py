"""Filesystem-backed run and artifact repository."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from .domain import PipelineEvent, RunStatus, to_primitive, utc_now
from .manifest import PROJECT_ROOT, MethodManifest


DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "platform"


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(to_primitive(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class RunRepository:
    def __init__(self, root: Path = DEFAULT_RUN_ROOT):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def run_dir(self, run_id: str) -> Path:
        if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
            raise ValueError("invalid run id")
        return self.root / run_id

    def create(
        self,
        method: MethodManifest,
        image_path: Path,
        label_path: Path | None,
        protocols: tuple[str, ...],
    ) -> dict[str, Any]:
        run_id = f"run-{uuid.uuid4().hex}"
        now = utc_now()
        settings = method.inference_settings
        data: dict[str, Any] = {
            "run_id": run_id,
            "created_at": now,
            "updated_at": now,
            "status": RunStatus.QUEUED.value,
            "method_id": method.method_id,
            "method_version": method.version,
            "image": {"path": str(image_path)},
            "label": {"path": str(label_path)} if label_path else None,
            "checkpoint": {
                "path": method.raw["base_detector"]["checkpoint_path"],
                "sha256": method.raw["base_detector"]["checkpoint_hash"],
            },
            "threshold": {
                "path": method.raw["threshold"]["file"],
                "sha256": method.raw["threshold"]["hash"],
            },
            "code_version": code_version(),
            "device": settings.device,
            "precision": settings.precision,
            "tile_config": {
                "tile_size": settings.tile_size,
                "stride": settings.stride,
                "input_size": settings.input_size,
                "global_nms_iou": settings.global_nms_iou,
            },
            "evaluation_protocols": list(protocols),
            "artifact_index": {},
            "error": None,
        }
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(directory / "run.json", data)
        _atomic_json(directory / "config.json", {
            "method_id": method.method_id,
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path else None,
            "inference": to_primitive(settings),
            "evaluation_protocols": list(protocols),
        })
        _atomic_json(directory / "environment.json", {
            "python_version": sys.version,
            "executable": sys.executable,
            "os_name": os.name,
            "code_version": data["code_version"],
            "device": settings.device,
            "precision": settings.precision,
        })
        _atomic_json(directory / "method_manifest.snapshot.json", dict(method.raw))
        (directory / "logs.txt").touch()
        (directory / "exports").mkdir(exist_ok=True)
        for name in ("config.json", "environment.json", "method_manifest.snapshot.json", "logs.txt"):
            data = self.register_artifact(run_id, name, directory / name)
        return data

    def get(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "run.json"
        if not path.is_file():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self.root.glob("run-*/run.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        rows.sort(key=lambda row: row["created_at"], reverse=True)
        return rows[: max(0, min(limit, 1000))]

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            data = self.get(run_id)
            data.update(changes)
            data["updated_at"] = utc_now()
            _atomic_json(self.run_dir(run_id) / "run.json", data)
            return data

    def register_artifact(self, run_id: str, name: str, path: Path) -> dict[str, Any]:
        with self._lock:
            data = self.get(run_id)
            artifact_index = dict(data.get("artifact_index", {}))
            artifact_index[name] = str(path.resolve().relative_to(self.run_dir(run_id).resolve()))
            return self.update(run_id, artifact_index=artifact_index)

    def write_json_artifact(self, run_id: str, name: str, data: Mapping[str, Any]) -> Path:
        path = self.run_dir(run_id) / name
        _atomic_json(path, data)
        self.register_artifact(run_id, name, path)
        return path

    def append_event(self, event: PipelineEvent) -> None:
        path = self.run_dir(event.run_id) / "events.jsonl"
        line = json.dumps(to_primitive(event), ensure_ascii=False) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        path = self.run_dir(run_id) / "events.jsonl"
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[max(0, after):] if line.strip()]

    def request_cancel(self, run_id: str) -> None:
        marker = self.run_dir(run_id) / "cancel.requested"
        marker.touch(exist_ok=True)

    def cancellation_requested(self, run_id: str) -> bool:
        return (self.run_dir(run_id) / "cancel.requested").exists()

    def resolve_artifact(self, run_id: str, name: str) -> Path:
        data = self.get(run_id)
        relative = data.get("artifact_index", {}).get(name)
        if relative is None:
            raise KeyError(name)
        path = (self.run_dir(run_id) / relative).resolve()
        path.relative_to(self.run_dir(run_id).resolve())
        return path
