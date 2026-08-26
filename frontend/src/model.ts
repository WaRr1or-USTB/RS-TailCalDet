import type {GroundTruthRecord, PredictionArtifact, ViewerDetection, ViewStage} from "./types";

export const broadClassFor = (classId: number): "ship" | "aircraft" | "vehicle" => {
  if (classId <= 3) return "ship";
  if (classId === 24) return "vehicle";
  return "aircraft";
};

export function detectionsForStage(
  prediction: PredictionArtifact,
  groundTruth: GroundTruthRecord[],
  stage: ViewStage,
): ViewerDetection[] {
  if (stage === "ground_truth") {
    return groundTruth.map((item) => ({
      id: item.id,
      classId: item.class_id,
      className: item.class_name,
      broadClass: item.broad_class,
      confidence: null,
      bbox: item.bbox_global,
      sourceTileIds: [],
      sourceDetectionIds: [],
      stage,
    }));
  }
  if (stage === "mapped") {
    return prediction.mapped_detections.map((item) => ({
      id: item.id,
      classId: item.class_id,
      className: item.class_name,
      broadClass: broadClassFor(item.class_id),
      confidence: item.confidence,
      bbox: item.bbox_global,
      sourceTileIds: [item.tile_id],
      sourceDetectionIds: [item.id],
      stage,
    }));
  }
  const source = stage === "final" ? prediction.final_detections : prediction.after_global_nms;
  return source.map((item) => ({
    id: item.id,
    classId: item.class_id,
    className: item.class_name,
    broadClass: item.broad_class,
    confidence: item.confidence,
    bbox: item.bbox_global,
    sourceTileIds: item.source_tile_ids,
    sourceDetectionIds: item.source_detection_ids,
    stage,
  }));
}

export function filterDetections(
  detections: ViewerDetection[],
  confidence: number,
  broadClass: "all" | "ship" | "aircraft" | "vehicle",
  selectedClasses: Set<number>,
): ViewerDetection[] {
  return detections.filter(
    (item) =>
      (item.confidence === null || item.confidence >= confidence) &&
      (broadClass === "all" || item.broadClass === broadClass) &&
      (selectedClasses.size === 0 || selectedClasses.has(item.classId)),
  );
}
