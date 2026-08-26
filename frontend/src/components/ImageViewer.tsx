import {forwardRef, useEffect, useImperativeHandle, useRef, useState} from "react";
import OpenSeadragon from "openseadragon";
import {Crosshair, Maximize2, Minus, Plus, Scan} from "lucide-react";

import type {ViewerDetection} from "../types";

export interface ImageViewerHandle {
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  fullScreen: () => void;
}

interface ImageViewerProps {
  imageUrl: string;
  imageWidth: number;
  detections: ViewerDetection[];
  selectedId: string | null;
  onSelect: (detection: ViewerDetection | null) => void;
  onReady?: () => void;
  onError?: (message: string) => void;
}

const ACCENT = "#e24d33";
const GT_COLOR = "#e7e5e4";

export const ImageViewer = forwardRef<ImageViewerHandle, ImageViewerProps>(function ImageViewer(
  {imageUrl, imageWidth, detections, selectedId, onSelect, onReady, onError},
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const detectionsRef = useRef(detections);
  const selectedIdRef = useRef(selectedId);
  const onSelectRef = useRef(onSelect);
  const [zoomPercent, setZoomPercent] = useState(100);
  const [imageState, setImageState] = useState<"loading" | "ready" | "error">("loading");

  detectionsRef.current = detections;
  selectedIdRef.current = selectedId;
  onSelectRef.current = onSelect;

  const draw = () => {
    const viewer = viewerRef.current;
    const canvas = canvasRef.current;
    if (!viewer || !canvas || !viewer.world.getItemCount()) return;
    const width = viewer.container.clientWidth;
    const height = viewer.container.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    for (const detection of detectionsRef.current) {
      const start = viewer.viewport.imageToViewerElementCoordinates(
        new OpenSeadragon.Point(detection.bbox.x1, detection.bbox.y1),
      );
      const end = viewer.viewport.imageToViewerElementCoordinates(
        new OpenSeadragon.Point(detection.bbox.x2, detection.bbox.y2),
      );
      const boxWidth = end.x - start.x;
      const boxHeight = end.y - start.y;
      if (end.x < 0 || end.y < 0 || start.x > width || start.y > height || boxWidth < 0.5 || boxHeight < 0.5) {
        continue;
      }
      const selected = detection.id === selectedIdRef.current;
      context.globalAlpha = selected ? 1 : Math.max(0.34, detection.confidence ?? 0.72);
      context.strokeStyle = selected ? "#fafaf9" : detection.stage === "ground_truth" ? GT_COLOR : ACCENT;
      context.lineWidth = selected ? 2.2 : 1.25;
      context.setLineDash(detection.stage === "ground_truth" ? [5, 4] : []);
      context.strokeRect(start.x, start.y, boxWidth, boxHeight);

      if (selected) {
        const label = `${detection.className}${detection.confidence === null ? "" : `  ${(detection.confidence * 100).toFixed(1)}%`}`;
        context.font = "600 12px 'Cascadia Code', monospace";
        const labelWidth = context.measureText(label).width + 14;
        const labelY = Math.max(4, start.y - 24);
        context.fillStyle = "rgba(17, 18, 16, 0.92)";
        context.fillRect(start.x, labelY, labelWidth, 21);
        context.globalAlpha = 1;
        context.fillStyle = "#fafaf9";
        context.fillText(label, start.x + 7, labelY + 14);
      }
    }
    context.globalAlpha = 1;
    context.setLineDash([]);
  };

  useEffect(() => {
    if (!hostRef.current) return;
    setImageState("loading");
    const viewer = OpenSeadragon({
      element: hostRef.current,
      tileSources: {type: "image", url: imageUrl},
      showNavigationControl: false,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      navigatorSizeRatio: 0.16,
      navigatorMaintainSizeRatio: true,
      animationTime: 0.7,
      blendTime: 0.1,
      springStiffness: 14,
      maxZoomPixelRatio: 5,
      visibilityRatio: 0.2,
      constrainDuringPan: true,
      gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true},
    });
    viewerRef.current = viewer;
    viewer.addHandler("open", () => {
      setImageState("ready");
      draw();
      onReady?.();
    });
    viewer.addHandler("open-failed", (event) => {
      setImageState("error");
      onError?.(event.message || "影像加载失败");
    });
    viewer.addHandler("animation", draw);
    viewer.addHandler("animation", () => {
      const homeZoom = viewer.viewport.getHomeZoom();
      if (homeZoom > 0) setZoomPercent(Math.round((viewer.viewport.getZoom(true) / homeZoom) * 100));
    });
    viewer.addHandler("resize", draw);
    viewer.addHandler("canvas-click", (event) => {
      if (!event.quick) return;
      const point = viewer.viewport.viewerElementToImageCoordinates(event.position);
      const hit = detectionsRef.current
        .filter(
          (item) =>
            point.x >= item.bbox.x1 &&
            point.x <= item.bbox.x2 &&
            point.y >= item.bbox.y1 &&
            point.y <= item.bbox.y2,
        )
        .sort((left, right) => left.bbox.width * left.bbox.height - right.bbox.width * right.bbox.height)[0];
      onSelectRef.current(hit ?? null);
    });
    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [imageUrl]);

  useEffect(draw, [detections, selectedId, imageWidth]);

  useImperativeHandle(ref, () => ({
    zoomIn: () => viewerRef.current?.viewport.zoomBy(1.5),
    zoomOut: () => viewerRef.current?.viewport.zoomBy(0.67),
    fit: () => viewerRef.current?.viewport.goHome(),
    fullScreen: () => viewerRef.current?.setFullScreen(true),
  }));

  return (
    <div className={`viewer-shell is-${imageState}`} aria-label="超大幅面遥感影像查看器">
      <div ref={hostRef} className="viewer-host" />
      <canvas ref={canvasRef} className="bbox-canvas" aria-hidden="true" />
      {imageState !== "ready" && <div className="viewer-loading" role="status">{imageState === "error" ? "影像加载失败" : "正在载入完整影像"}</div>}
      <div className="viewer-readout"><span>完整影像</span><strong>{zoomPercent}%</strong></div>
      <div className="viewer-controls" aria-label="影像视图控制">
        <button type="button" onClick={() => viewerRef.current?.viewport.zoomBy(1.5)} title="放大影像">
          <Plus size={18} /><span>放大</span>
        </button>
        <button type="button" onClick={() => viewerRef.current?.viewport.zoomBy(0.67)} title="缩小影像">
          <Minus size={18} /><span>缩小</span>
        </button>
        <button type="button" onClick={() => viewerRef.current?.viewport.goHome()} title="显示完整影像">
          <Scan size={18} /><span>完整适配</span>
        </button>
        <button type="button" onClick={() => onSelectRef.current(null)} title="清除当前选择">
          <Crosshair size={18} /><span>清除选择</span>
        </button>
        <button type="button" onClick={() => viewerRef.current?.setFullScreen(true)} title="进入全屏模式">
          <Maximize2 size={18} /><span>全屏查看</span>
        </button>
      </div>
    </div>
  );
});
