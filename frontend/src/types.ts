export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type ViewStage = "final" | "after_global_nms" | "mapped" | "ground_truth";

export interface RunRecord {
  run_id: string;
  created_at: string;
  updated_at: string;
  status: RunStatus;
  method_id: string;
  method_version: string;
  image: {path: string};
  label: {path: string} | null;
  artifact_index: Record<string, string>;
  error: string | null;
}

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

export interface FinalDetection {
  id: string;
  image_id: string;
  class_id: number;
  class_name: string;
  broad_class: "ship" | "aircraft" | "vehicle";
  confidence: number;
  bbox_global: BBox;
  source_tile_ids: string[];
  source_detection_ids: string[];
  stage: "after_global_nms" | "final";
  kept_by_global_nms: boolean;
  kept_by_class_threshold: boolean;
}

export interface MappedDetection {
  id: string;
  image_id: string;
  tile_id: string;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_tile: BBox;
  bbox_global: BBox;
  stage: "mapped";
}

export interface PredictionArtifact {
  schema_version: string;
  run: {run_id: string; created_at: string; method_id: string; method_version: string};
  image: {
    image_id: string;
    filename: string;
    width: number;
    height: number;
    channels: number;
    file_size: number;
    dataset: string | null;
    split: string | null;
    proxy_big_image: boolean;
  };
  model: {
    method_display_name: string;
    base_detector: string;
    checkpoint_hash: string;
    end2end: boolean;
    inference_branch: string;
    detection_scales: string[];
    precision: string;
    device: string;
  };
  threshold: {per_class_values: Record<string, number>};
  tile_plan: {
    tile_size: number;
    stride: number;
    network_input_size: number;
    tile_count: number;
    global_class_wise_nms_iou: number;
  };
  mapped_detections: MappedDetection[];
  after_global_nms: FinalDetection[];
  final_detections: FinalDetection[];
  stage_counts: Record<"raw" | "mapped" | "after_global_nms" | "final", number>;
}

export interface GroundTruthRecord {
  id: string;
  class_id: number;
  class_name: string;
  broad_class: "ship" | "aircraft" | "vehicle";
  bbox_global: BBox;
}

export interface CountMetrics {
  tp: number;
  fp: number;
  fn: number;
  ground_truth: number;
  predictions: number;
  recall: number;
  precision: number;
  fdr: number;
}

export interface MetricsArtifact {
  strict_25?: {available: boolean; overall: CountMetrics; metadata: Record<string, unknown>};
  official_overall?: {available: boolean; overall: CountMetrics; metadata: Record<string, unknown>};
}

export interface TimingArtifact {
  algorithm_pipeline_seconds: number;
  strict_e2e_seconds: number;
  web_task_seconds: number | null;
  model_inference_seconds: number;
}

export interface PresentationSummary {
  image: PredictionArtifact["image"];
  tile_count: number;
  stage_counts: Record<"raw" | "mapped" | "after_global_nms" | "final", number>;
  broad_class_counts: Record<"ship" | "aircraft" | "vehicle", number>;
  timing: {
    algorithm_pipeline_seconds: number | null;
    strict_e2e_seconds: number | null;
  };
}

export interface ViewerDetection {
  id: string;
  classId: number;
  className: string;
  broadClass: "ship" | "aircraft" | "vehicle";
  confidence: number | null;
  bbox: BBox;
  sourceTileIds: string[];
  sourceDetectionIds: string[];
  stage: ViewStage;
}

export interface ClassDefinition {
  id: number;
  name: string;
}
