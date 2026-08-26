import {useCallback, useDeferredValue, useEffect, useMemo, useRef, useState} from "react";
import {AlertTriangle, Check, ChevronDown, CloudUpload, LoaderCircle, RefreshCw, Satellite} from "lucide-react";

import {api} from "./api";
import {FilterPanel} from "./components/FilterPanel";
import {ImageViewer, type ImageViewerHandle} from "./components/ImageViewer";
import {Inspector} from "./components/Inspector";
import {LandingPage} from "./components/LandingPage";
import {StageSwitch} from "./components/StageSwitch";
import {detectionsForStage, filterDetections} from "./model";
import type {
  ClassDefinition,
  GroundTruthRecord,
  MetricsArtifact,
  PredictionArtifact,
  PresentationSummary,
  RunRecord,
  TimingArtifact,
  ViewerDetection,
  ViewStage,
} from "./types";

type BroadFilter = "all" | "ship" | "aircraft" | "vehicle";

interface RunBundle {
  run: RunRecord;
  prediction: PredictionArtifact;
  metrics: MetricsArtifact | null;
  timing: TimingArtifact | null;
  groundTruth: GroundTruthRecord[];
  presentation: PresentationSummary;
}

const statusLabels: Record<RunRecord["status"], string> = {
  queued: "等待中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function StatusMark({status}: {status: RunRecord["status"]}) {
  const icon = status === "succeeded" ? <Check size={13} /> : status === "failed" ? <AlertTriangle size={13} /> : <LoaderCircle size={13} />;
  return <span className={`status-mark ${status}`}>{icon}{statusLabels[status]}</span>;
}

export default function App() {
  const reportMode = new URLSearchParams(window.location.search).get("mode") === "report";
  const viewerRef = useRef<ImageViewerHandle>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [classes, setClasses] = useState<ClassDefinition[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<RunBundle | null>(null);
  const [stage, setStage] = useState<ViewStage>("final");
  const [confidence, setConfidence] = useState(0);
  const deferredConfidence = useDeferredValue(confidence);
  const [broadClass, setBroadClass] = useState<BroadFilter>("all");
  const [selectedClasses, setSelectedClasses] = useState<Set<number>>(new Set());
  const [selectedDetection, setSelectedDetection] = useState<ViewerDetection | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "empty" | "error">("loading");
  const [message, setMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const [surface, setSurface] = useState<"landing" | "workbench">(() => reportMode || window.location.hash === "#workbench" ? "workbench" : "landing");

  const refreshRuns = useCallback(async () => {
    const nextRuns = await api.listRuns();
    setRuns(nextRuns);
    return nextRuns;
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([refreshRuns(), api.getClasses()])
      .then(([nextRuns, nextClasses]) => {
        if (!active) return;
        setClasses(nextClasses);
        const queryRun = new URLSearchParams(window.location.search).get("run");
        const initial = nextRuns.find((item) => item.run_id === queryRun) ?? nextRuns.find((item) => item.status === "succeeded") ?? nextRuns[0];
        if (initial) setSelectedRunId(initial.run_id);
        else setState("empty");
      })
      .catch((error: Error) => {
        if (!active) return;
        setMessage(error.message);
        setState("error");
      });
    return () => {
      active = false;
    };
  }, [refreshRuns]);

  useEffect(() => {
    if (!selectedRunId) return;
    let active = true;
    let timer: number | undefined;
    const load = async () => {
      try {
        const run = await api.getRun(selectedRunId);
        setRuns((current) => current.map((item) => (item.run_id === run.run_id ? run : item)));
        if (run.status === "queued" || run.status === "running") {
          if (active) {
            setBundle(null);
            setState("loading");
            setMessage(run.status === "queued" ? "任务正在等待 GPU worker" : "正在执行切片推理与全图后处理");
            timer = window.setTimeout(load, 1200);
          }
          return;
        }
        if (run.status !== "succeeded") {
          throw new Error(run.error || `任务状态为${statusLabels[run.status]}`);
        }
        const [prediction, metricsResult, timingResult, groundTruthResult, presentation] = await Promise.all([
          api.getPredictions(selectedRunId),
          api.getMetrics(selectedRunId).catch(() => null),
          api.getTiming(selectedRunId).catch(() => null),
          api.getGroundTruth(selectedRunId).catch(() => []),
          api.getPresentation(selectedRunId),
        ]);
        if (!active) return;
        setBundle({run, prediction, metrics: metricsResult, timing: timingResult, groundTruth: groundTruthResult, presentation});
        setSelectedDetection(null);
        setStage("final");
        setState("ready");
        const query = new URLSearchParams({run: run.run_id});
        if (reportMode) query.set("mode", "report");
        window.history.replaceState(null, "", `${window.location.pathname}?${query.toString()}#workbench`);
      } catch (error) {
        if (!active) return;
        setBundle(null);
        setMessage(error instanceof Error ? error.message : "无法读取运行产物");
        setState("error");
      }
    };
    setState("loading");
    setMessage("正在读取预测产物");
    void load();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [selectedRunId]);

  const stageDetections = useMemo(
    () => (bundle ? detectionsForStage(bundle.prediction, bundle.groundTruth, stage) : []),
    [bundle, stage],
  );
  const runScope = useMemo(() => {
    const groups = new Set(stageDetections.map((item) => item.broadClass));
    if (groups.size !== 1) return null;
    const only = groups.values().next().value;
    return only === "ship" ? "当前运行仅含舰船" : only === "aircraft" ? "当前运行仅含飞机" : "当前运行仅含车辆";
  }, [stageDetections]);
  const filteredDetections = useMemo(
    () => filterDetections(stageDetections, deferredConfidence, broadClass, selectedClasses),
    [stageDetections, deferredConfidence, broadClass, selectedClasses],
  );

  useEffect(() => {
    if (selectedDetection && !filteredDetections.some((item) => item.id === selectedDetection.id)) {
      setSelectedDetection(null);
    }
  }, [filteredDetections, selectedDetection]);

  const toggleClass = (classId: number) => {
    setSelectedClasses((current) => {
      const next = new Set(current);
      if (next.has(classId)) next.delete(classId);
      else next.add(classId);
      return next;
    });
  };

  const handleUpload = async (file?: File) => {
    if (!file) return;
    setUploading(true);
    try {
      const run = await api.uploadImage(file);
      setRuns((current) => [run, ...current]);
      setSelectedRunId(run.run_id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "上传失败");
      setState("error");
    } finally {
      setUploading(false);
      if (uploadRef.current) uploadRef.current.value = "";
    }
  };

  const selectedRun = runs.find((item) => item.run_id === selectedRunId);

  if (surface === "landing") {
    return <LandingPage onEnter={() => {window.location.hash = "workbench"; setSurface("workbench");}} />;
  }

  return (
    <div className={`app-shell ${reportMode ? "report-mode" : ""}`}>
      <header className="topbar">
        <button type="button" className="brand-block" onClick={() => {window.location.hash = ""; setSurface("landing");}} title="返回平台首页">
          <span className="brand-mark"><Satellite size={19} strokeWidth={1.7} /></span>
          <div><strong>RS-CalVision</strong><span>超大幅面光学遥感检测工作台</span></div>
        </button>

        <div className="run-control">
          <label htmlFor="run-select">{reportMode ? "技术路线展示" : "运行记录"}</label>
          <div className="select-wrap">
            <select id="run-select" value={selectedRunId ?? ""} onChange={(event) => setSelectedRunId(event.target.value)}>
              {runs.map((run) => <option value={run.run_id} key={run.run_id}>{run.run_id.slice(0, 16)} / {statusLabels[run.status]}</option>)}
            </select>
            <ChevronDown size={15} />
          </div>
          {selectedRun && <StatusMark status={selectedRun.status} />}
        </div>

        <div className="top-actions" aria-hidden={reportMode || undefined}>
          <button type="button" className="icon-button" onClick={() => void refreshRuns()} title="刷新运行记录"><RefreshCw size={17} /></button>
          <input ref={uploadRef} className="sr-only" type="file" accept="image/*" onChange={(event) => void handleUpload(event.target.files?.[0])} />
          <button type="button" className="upload-button" disabled={uploading} onClick={() => uploadRef.current?.click()}>
            {uploading ? <LoaderCircle className="spin" size={16} /> : <CloudUpload size={16} />}
            {uploading ? "正在上传" : "上传影像"}
          </button>
        </div>
      </header>

      {state === "ready" && bundle ? (
        <main className="workspace">
          <FilterPanel
            classes={classes}
            detections={stageDetections}
            confidence={confidence}
            broadClass={broadClass}
            selectedClasses={selectedClasses}
            onConfidenceChange={setConfidence}
            onBroadClassChange={setBroadClass}
            onToggleClass={toggleClass}
            onClearClasses={() => setSelectedClasses(new Set())}
          />

          <section className="canvas-column">
            <div className="canvas-toolbar">
              <div className="image-identity">
                <strong>{bundle.prediction.image.filename}</strong>
                <span>{bundle.prediction.image.width.toLocaleString()} × {bundle.prediction.image.height.toLocaleString()} px</span>
                {runScope && <em className="run-scope">{runScope}</em>}
              </div>
              <StageSwitch value={stage} gtAvailable={bundle.groundTruth.length > 0} onChange={(value) => {setStage(value); setSelectedDetection(null);}} />
              <div className="visible-count"><span>可见检测</span><strong>{filteredDetections.length}</strong></div>
            </div>
            <ImageViewer
              ref={viewerRef}
              imageUrl={api.imageUrl(bundle.run.run_id)}
              imageWidth={bundle.prediction.image.width}
              detections={filteredDetections}
              selectedId={selectedDetection?.id ?? null}
              onSelect={setSelectedDetection}
              onError={(error) => {setMessage(error); setState("error");}}
            />
            <footer className="pipeline-strip" aria-label="检测技术路线">
              <div><span>01</span><strong>影像接入</strong><small>{bundle.presentation.image.width.toLocaleString()} × {bundle.presentation.image.height.toLocaleString()}</small></div>
              <div><span>02</span><strong>智能切片</strong><small>{bundle.presentation.tile_count} 个切片</small></div>
              <div><span>03</span><strong>全图映射</strong><small>{bundle.presentation.stage_counts.mapped} 个候选框</small></div>
              <div><span>04</span><strong>全图抑制</strong><small>{bundle.presentation.stage_counts.after_global_nms} 个保留框</small></div>
              <div><span>05</span><strong>逐类校准</strong><small>{bundle.presentation.stage_counts.final} 个最终目标</small></div>
            </footer>
          </section>

          <Inspector selected={selectedDetection} prediction={bundle.prediction} metrics={bundle.metrics} timing={bundle.timing} presentation={bundle.presentation} />
        </main>
      ) : (
        <main className="state-stage">
          {state === "loading" && <><LoaderCircle className="spin" size={28} /><strong>{message || "正在连接 RS-CalVision"}</strong><p>任务状态和预测产物会自动刷新。</p></>}
          {state === "empty" && <><CloudUpload size={30} /><strong>还没有可查看的运行</strong><p>上传一张遥感影像，系统会创建异步 GPU 推理任务。</p><button type="button" onClick={() => uploadRef.current?.click()}>上传影像</button></>}
          {state === "error" && <><AlertTriangle size={30} /><strong>工作台无法读取当前运行</strong><p>{message}</p><button type="button" onClick={() => window.location.reload()}>重新连接</button></>}
        </main>
      )}
    </div>
  );
}
