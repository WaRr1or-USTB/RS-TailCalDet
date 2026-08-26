"""Asynchronous single-GPU execution service."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from .domain import PipelineEvent, PipelineStage, RunStatus, to_primitive
from .evaluation import EvaluationAdapter
from .inference import BigImageInferenceAdapter, InferenceCancelled, ModelRuntimeAdapter
from .manifest import MethodRegistry, PROJECT_ROOT
from .repository import RunRepository
from .serialization import PredictionSerializer


class QueueFullError(RuntimeError):
    pass


class PlatformService:
    """One bounded queue and one worker, intentionally serializing GPU work."""

    def __init__(
        self,
        registry: MethodRegistry | None = None,
        repository: RunRepository | None = None,
        max_queue_size: int = 8,
        start_worker: bool = True,
    ):
        self.registry = registry or MethodRegistry()
        self.repository = repository or RunRepository()
        self.serializer = PredictionSerializer()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue_size)
        self._submitted_at: dict[str, float] = {}
        self._runtime_by_method: dict[str, ModelRuntimeAdapter] = {}
        self._worker = threading.Thread(target=self._worker_loop, name="rs-calvision-gpu-worker", daemon=True)
        if start_worker:
            self._worker.start()

    @staticmethod
    def _trusted_input_path(value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError("input path must stay inside the active project") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def submit(
        self,
        method_id: str,
        image_path: str | Path,
        label_path: str | Path | None = None,
        protocols: tuple[str, ...] = ("strict_25", "official_overall", "official_by_category", "group_macro"),
    ) -> dict[str, Any]:
        method = self.registry.get(method_id)
        supported = set(method.raw["evaluation"]["supported_protocols"])
        unknown = set(protocols).difference(supported)
        if unknown:
            raise ValueError(f"unsupported evaluation protocols: {sorted(unknown)}")
        image = self._trusted_input_path(image_path)
        label = self._trusted_input_path(label_path) if label_path else None
        run = self.repository.create(method, image, label, protocols)
        run_id = run["run_id"]
        self.repository.append_event(PipelineEvent.create(run_id, PipelineStage.QUEUED, 0.0, "Run queued"))
        self._submitted_at[run_id] = time.perf_counter()
        try:
            self._queue.put_nowait(run_id)
        except queue.Full as exc:
            self.repository.update(run_id, status=RunStatus.FAILED.value, error="GPU queue is full")
            raise QueueFullError("GPU queue is full") from exc
        return self.repository.get(run_id)

    def cancel(self, run_id: str) -> dict[str, Any]:
        run = self.repository.get(run_id)
        if run["status"] in {RunStatus.SUCCEEDED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}:
            return run
        self.repository.request_cancel(run_id)
        return self.repository.update(run_id, cancellation_requested=True)

    def _event(self, event: PipelineEvent) -> None:
        self.repository.append_event(event)

    def _worker_loop(self) -> None:
        while True:
            run_id = self._queue.get()
            if run_id is None:
                self._queue.task_done()
                return
            try:
                self._execute(run_id)
            finally:
                self._submitted_at.pop(run_id, None)
                self._queue.task_done()

    def _execute(self, run_id: str) -> None:
        run = self.repository.get(run_id)
        if self.repository.cancellation_requested(run_id):
            self._mark_cancelled(run_id)
            return
        method = self.registry.get(run["method_id"])
        self.repository.update(run_id, status=RunStatus.RUNNING.value)
        runtime = self._runtime_by_method.setdefault(method.method_id, ModelRuntimeAdapter(method))
        adapter = BigImageInferenceAdapter(method, runtime)
        try:
            result = adapter.infer(
                run["image"]["path"], run_id,
                event_callback=self._event,
                cancellation_check=lambda: self.repository.cancellation_requested(run_id),
            )
            self._event(PipelineEvent.create(run_id, PipelineStage.PREDICTION_SERIALIZATION, 0.80, "Writing prediction artifact"))
            artifact = self.serializer.build_artifact(result, method, created_at=run["created_at"])
            prediction_path = self.repository.run_dir(run_id) / "predictions.json"
            _, serialization_seconds = self.serializer.write(artifact, prediction_path)
            self.repository.register_artifact(run_id, "predictions.json", prediction_path)

            evaluation_seconds = 0.0
            if run.get("label"):
                self._event(PipelineEvent.create(run_id, PipelineStage.EVALUATION_STARTED, 0.88, "Evaluating predictions"))
                evaluation_start = time.perf_counter()
                evaluator = EvaluationAdapter(method)
                truths = evaluator.load_yolo_labels(
                    run["label"]["path"], result.image.image_id, result.image.width, result.image.height
                )
                metrics = evaluator.evaluate(result.final_detections, truths, tuple(run["evaluation_protocols"]))
                evaluation_seconds = time.perf_counter() - evaluation_start
                self.repository.write_json_artifact(run_id, "metrics.json", to_primitive(metrics))
                self._event(PipelineEvent.create(run_id, PipelineStage.EVALUATION_COMPLETED, 0.96, "Evaluation completed"))

            web_task_seconds = time.perf_counter() - self._submitted_at.get(run_id, time.perf_counter())
            timing = replace(
                result.timing,
                serialization_seconds=serialization_seconds,
                evaluation_seconds=evaluation_seconds,
                strict_e2e_seconds=result.timing.strict_e2e_seconds + serialization_seconds,
                web_task_seconds=web_task_seconds,
            )
            self.repository.write_json_artifact(run_id, "timing.json", to_primitive(timing))
            self.repository.update(run_id, status=RunStatus.SUCCEEDED.value)
            self._event(PipelineEvent.create(run_id, PipelineStage.COMPLETED, 1.0, "Run completed"))
        except InferenceCancelled:
            self._mark_cancelled(run_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log_path = self.repository.run_dir(run_id) / "error.log"
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            self.repository.register_artifact(run_id, "error.log", log_path)
            self.repository.update(run_id, status=RunStatus.FAILED.value, error=error)
            self._event(PipelineEvent.create(run_id, PipelineStage.FAILED, 1.0, error))

    def _mark_cancelled(self, run_id: str) -> None:
        self.repository.update(run_id, status=RunStatus.CANCELLED.value)
        self._event(PipelineEvent.create(run_id, PipelineStage.CANCELLED, 1.0, "Run cancelled"))

    def stop(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=10)
