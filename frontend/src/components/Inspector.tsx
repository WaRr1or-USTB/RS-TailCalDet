import {AnimatePresence, motion, useReducedMotion} from "motion/react";
import {Box, Clock3, Crosshair, Database, Layers3} from "lucide-react";

import type {MetricsArtifact, PredictionArtifact, PresentationSummary, TimingArtifact, ViewerDetection} from "../types";

const formatPercent = (value?: number) => (value === undefined ? "-" : `${(value * 100).toFixed(2)}%`);
const formatSeconds = (value?: number | null) => (value == null ? "-" : `${value.toFixed(2)} 秒`);
const broadNames = {ship: "舰船", aircraft: "飞机", vehicle: "车辆"};
const stageNames = {final: "最终结果", after_global_nms: "全图抑制后", mapped: "全图映射", ground_truth: "标注真值"};

interface InspectorProps {
  selected: ViewerDetection | null;
  prediction: PredictionArtifact;
  metrics: MetricsArtifact | null;
  timing: TimingArtifact | null;
  presentation: PresentationSummary;
}

export function Inspector({selected, prediction, metrics, timing, presentation}: InspectorProps) {
  const reduceMotion = useReducedMotion();
  const strict = metrics?.strict_25?.overall;
  return (
    <aside className="inspector" aria-label="检测详情">
      <div className="panel-title">
        <div>
          <span>检测详情</span>
          <strong>{selected ? selected.className : "未选择"}</strong>
        </div>
        <Crosshair size={17} />
      </div>

      <AnimatePresence mode="wait" initial={false}>
        {selected ? (
          <motion.div
            key={selected.id}
            className="inspector-selection"
            initial={reduceMotion ? false : {opacity: 0, x: 10}}
            animate={{opacity: 1, x: 0}}
            exit={reduceMotion ? undefined : {opacity: 0, x: -8}}
            transition={{duration: 0.18}}
          >
            <div className="selection-class">
              <span>{String(selected.classId).padStart(2, "0")}</span>
              <div>
                <strong>{selected.className}</strong>
                <small>{broadNames[selected.broadClass]}</small>
              </div>
              <b>{selected.confidence === null ? "GT" : `${(selected.confidence * 100).toFixed(1)}%`}</b>
            </div>

            <dl className="detail-list">
              <div><dt>检测编号</dt><dd>{selected.id}</dd></div>
              <div><dt>结果阶段</dt><dd>{stageNames[selected.stage]}</dd></div>
              <div><dt>左上坐标</dt><dd>{selected.bbox.x1.toFixed(1)}, {selected.bbox.y1.toFixed(1)}</dd></div>
              <div><dt>右下坐标</dt><dd>{selected.bbox.x2.toFixed(1)}, {selected.bbox.y2.toFixed(1)}</dd></div>
              <div><dt>宽 × 高</dt><dd>{selected.bbox.width.toFixed(1)} × {selected.bbox.height.toFixed(1)}</dd></div>
              <div><dt>来源切片</dt><dd>{selected.sourceTileIds.length ? selected.sourceTileIds.join(", ") : "-"}</dd></div>
              <div><dt>来源链路</dt><dd>{selected.sourceDetectionIds.length || "-"}</dd></div>
            </dl>
          </motion.div>
        ) : (
          <motion.div key="empty" className="run-overview" initial={{opacity: 0}} animate={{opacity: 1}}>
            <div className="overview-heading"><Box size={20} strokeWidth={1.5} /><div><strong>当前影像检测概览</strong><span>点击检测框可继续追溯单个目标</span></div></div>
            <div className="broad-summary">
              <div><span>舰船</span><strong>{presentation.broad_class_counts.ship}</strong></div>
              <div><span>飞机</span><strong>{presentation.broad_class_counts.aircraft}</strong></div>
              <div><span>车辆</span><strong>{presentation.broad_class_counts.vehicle}</strong></div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="metric-cluster">
        <h2><Database size={15} /> 当前影像严格25类评价</h2>
        <div className="metric-pair">
          <div><span>召回率</span><strong>{formatPercent(strict?.recall)}</strong></div>
          <div><span>虚警率</span><strong>{formatPercent(strict?.fdr)}</strong></div>
        </div>
        <div className="count-row">
          <span>正确检出 {strict?.tp ?? "-"}</span>
          <span>虚警 {strict?.fp ?? "-"}</span>
          <span>漏检 {strict?.fn ?? "-"}</span>
        </div>
      </div>

      <div className="metric-cluster">
        <h2><Clock3 size={15} /> 性能</h2>
        <dl className="compact-stats">
          <div><dt>算法流水线</dt><dd>{formatSeconds(timing?.algorithm_pipeline_seconds)}</dd></div>
          <div><dt>严格端到端</dt><dd>{formatSeconds(timing?.strict_e2e_seconds)}</dd></div>
          <div><dt>平台任务</dt><dd>{formatSeconds(timing?.web_task_seconds)}</dd></div>
        </dl>
      </div>

      <div className="metric-cluster method-block">
        <h2><Layers3 size={15} /> 方法配置</h2>
        <dl className="compact-stats">
          <div><dt>方法</dt><dd>{prediction.model.method_display_name}</dd></div>
          <div><dt>基础检测器</dt><dd>{prediction.model.base_detector}</dd></div>
          <div><dt>切片</dt><dd>{prediction.tile_plan.tile_size} / {prediction.tile_plan.stride}</dd></div>
          <div><dt>全图 NMS</dt><dd>{prediction.tile_plan.global_class_wise_nms_iou.toFixed(2)}</dd></div>
        </dl>
      </div>
    </aside>
  );
}
