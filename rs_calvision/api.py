"""Minimal FastAPI transport for the RS-CalVision backend."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .domain import RunStatus
from .evaluation import EvaluationAdapter
from .manifest import PROJECT_ROOT
from .serialization import build_presentation_summary
from .service import PlatformService, QueueFullError


class RunRequest(BaseModel):
    method_id: str = "rs-tailcaldet-v1"
    image_path: str
    label_path: str | None = None
    protocols: list[str] = Field(default_factory=lambda: ["strict_25", "official_overall", "official_by_category", "group_macro"])


def create_app(service: PlatformService | None = None) -> FastAPI:
    app = FastAPI(title="RS-CalVision API", version="0.1.0")
    backend = service or PlatformService()
    app.state.backend = backend

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/system")
    def system() -> dict:
        return {
            "platform": "RS-CalVision",
            "execution": "single_gpu_single_worker",
            "run_repository": str(backend.repository.root),
        }

    @app.get("/methods")
    def methods() -> list[dict]:
        return [dict(item.raw) for item in backend.registry.list()]

    @app.get("/models")
    def models() -> list[dict]:
        return [
            {
                "method_id": item.method_id,
                "display_name": item.display_name,
                "base_detector": item.raw["base_detector"],
            }
            for item in backend.registry.list()
        ]

    @app.get("/classes")
    def classes(method_id: str = "rs-tailcaldet-v1") -> dict:
        try:
            return dict(backend.registry.get(method_id).raw["classes"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/methods/{method_id}")
    def method(method_id: str) -> dict:
        try:
            return dict(backend.registry.get(method_id).raw)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs", status_code=202)
    def submit(request: RunRequest) -> dict:
        try:
            return backend.submit(request.method_id, request.image_path, request.label_path, tuple(request.protocols))
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runs/upload", status_code=202)
    async def upload(
        image: UploadFile = File(...),
        method_id: str = Form("rs-tailcaldet-v1"),
        protocols: str = Form("strict_25,official_overall,official_by_category,group_macro"),
    ) -> dict:
        upload_root = PROJECT_ROOT / "runs" / "platform" / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        safe_name = Path(image.filename or "image.bin").name
        target = upload_root / f"{uuid.uuid4().hex}-{safe_name}"
        with target.open("wb") as handle:
            while chunk := await image.read(1024 * 1024):
                handle.write(chunk)
        try:
            return backend.submit(method_id, target, None, tuple(item.strip() for item in protocols.split(",") if item.strip()))
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs")
    def runs(limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
        return backend.repository.list(limit)

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> dict:
        try:
            return backend.repository.get(run_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict:
        try:
            return backend.cancel(run_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/events")
    def events(run_id: str, after: int = Query(0, ge=0)) -> list[dict]:
        try:
            backend.repository.get(run_id)
            return backend.repository.events(run_id, after)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/image")
    def source_image(run_id: str) -> FileResponse:
        try:
            run_data = backend.repository.get(run_id)
            path = backend._trusted_input_path(run_data["image"]["path"])
            return FileResponse(path)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/events/stream")
    async def event_stream(run_id: str, after: int = Query(0, ge=0)) -> StreamingResponse:
        try:
            backend.repository.get(run_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def generate():
            cursor = after
            terminal = {RunStatus.SUCCEEDED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
            while True:
                rows = backend.repository.events(run_id, cursor)
                for row in rows:
                    yield f"data: {json.dumps(row, ensure_ascii=False)}\n\n"
                cursor += len(rows)
                if backend.repository.get(run_id)["status"] in terminal and not rows:
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str) -> FileResponse:
        try:
            path = backend.repository.resolve_artifact(run_id, name)
            return FileResponse(path)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def read_json_artifact(run_id: str, name: str) -> dict:
        try:
            path = backend.repository.resolve_artifact(run_id, name)
            return json.loads(path.read_text(encoding="utf-8"))
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/predictions")
    def predictions(run_id: str) -> dict:
        return read_json_artifact(run_id, "predictions.json")

    @app.get("/runs/{run_id}/metrics")
    def metrics(run_id: str) -> dict:
        return read_json_artifact(run_id, "metrics.json")

    @app.get("/runs/{run_id}/timing")
    def timing(run_id: str) -> dict:
        return read_json_artifact(run_id, "timing.json")

    @app.get("/runs/{run_id}/presentation")
    def presentation(run_id: str) -> dict:
        prediction = read_json_artifact(run_id, "predictions.json")
        try:
            timing_data = read_json_artifact(run_id, "timing.json")
        except HTTPException:
            timing_data = None
        return build_presentation_summary(prediction, timing_data)

    @app.get("/runs/{run_id}/ground-truth")
    def ground_truth(run_id: str) -> dict:
        try:
            run_data = backend.repository.get(run_id)
            label = run_data.get("label")
            if not label:
                raise FileNotFoundError("run has no ground-truth label")
            predictions_data = read_json_artifact(run_id, "predictions.json")
            image = predictions_data["image"]
            method_manifest = backend.registry.get(run_data["method_id"])
            truths = EvaluationAdapter(method_manifest).load_yolo_labels(
                label["path"], image["image_id"], image["width"], image["height"]
            )
            return {
                "image_id": image["image_id"],
                "ground_truths": [
                    {
                        "id": item.ground_truth_id,
                        "class_id": item.class_id,
                        "class_name": item.class_name,
                        "broad_class": item.broad_class,
                        "bbox_global": {
                            "x1": item.bbox_global_xyxy.x1,
                            "y1": item.bbox_global_xyxy.y1,
                            "x2": item.bbox_global_xyxy.x2,
                            "y2": item.bbox_global_xyxy.y2,
                            "width": item.bbox_global_xyxy.width,
                            "height": item.bbox_global_xyxy.height,
                        },
                    }
                    for item in truths
                ],
            }
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/workbench", StaticFiles(directory=frontend_dist, html=True), name="workbench")

    return app


app = create_app()
