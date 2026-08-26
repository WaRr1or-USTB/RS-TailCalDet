/* Local visual-QA fixture. It never participates in production builds. */
import {createReadStream, readFileSync, statSync} from "node:fs";
import {createServer} from "node:http";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const projectRoot = resolve(frontendRoot, "..");
const mockPort = Number(process.env.RS_CALVISION_MOCK_PORT || 8000);
const imagePath = join(projectRoot, "demo_showcase", "generated", "images", "demo_scene_grouped", "big_demo_scene_grouped_000001.jpg");
const labelPath = join(projectRoot, "demo_showcase", "generated", "labels", "demo_scene_grouped", "big_demo_scene_grouped_000001.txt");
const classNames = [
  "HM", "LQS", "QHS", "MS", "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5", "A5_F-16",
  "A6_TU-160", "A7_E-3", "A8_B-52", "A9_P-3C", "A10_B-1B", "A11_E-8", "A12_TU-22",
  "A13_F-15", "A14_KC-135", "A15_F-22", "A16_FA-18", "A17_TU-95", "A18_KC-10", "A19_SU-34",
  "A20_SU-24", "FSC",
];
const broadClass = (id) => (id <= 3 ? "ship" : id === 24 ? "vehicle" : "aircraft");
const boxes = readFileSync(labelPath, "utf8").trim().split(/\r?\n/).map((line, index) => {
  const [classId, cx, cy, width, height] = line.split(/\s+/).map(Number);
  const x1 = (cx - width / 2) * 10000;
  const y1 = (cy - height / 2) * 10000;
  const x2 = (cx + width / 2) * 10000;
  const y2 = (cy + height / 2) * 10000;
  return {
    id: `qa-${String(index).padStart(5, "0")}`,
    image_id: "big_val_sparse_000001",
    class_id: classId,
    class_name: classNames[classId],
    broad_class: broadClass(classId),
    confidence: 0.98 - (index % 8) * 0.025,
    bbox_global: {x1, y1, x2, y2, width: x2 - x1, height: y2 - y1},
    source_tile_ids: [`tile-${String(index % 169).padStart(6, "0")}`],
    source_detection_ids: [`raw-${String(index).padStart(8, "0")}`],
    stage: "final",
    kept_by_global_nms: true,
    kept_by_class_threshold: true,
  };
});
const mapped = boxes.map((item) => ({
  id: item.id.replace("qa", "mapped"), image_id: item.image_id, tile_id: item.source_tile_ids[0],
  class_id: item.class_id, class_name: item.class_name, confidence: item.confidence,
  bbox_tile: item.bbox_global, bbox_global: item.bbox_global, stage: "mapped",
}));
const run = {
  run_id: "ui-qa-run", created_at: "2026-08-25T09:00:00Z", updated_at: "2026-08-25T09:00:09Z",
  status: "succeeded", method_id: "rs-tailcaldet-v1", method_version: "1.0.0",
  image: {path: imagePath}, label: {path: labelPath}, artifact_index: {"predictions.json": "predictions.json"}, error: null,
};
const prediction = {
  schema_version: "1.0.0", run: {run_id: run.run_id, created_at: run.created_at, method_id: run.method_id, method_version: run.method_version},
  image: {image_id: "big_val_sparse_000001", filename: "big_val_sparse_000001.jpg", width: 10000, height: 10000, channels: 3, file_size: statSync(imagePath).size, dataset: "data_big_v1", split: "val_sparse", proxy_big_image: true},
  model: {method_display_name: "UI-QA Fixture", base_detector: "YOLO26x", checkpoint_hash: "QA", end2end: true, inference_branch: "one_to_one", detection_scales: ["P3/8", "P4/16", "P5/32"], precision: "fp32", device: "0"},
  threshold: {per_class_values: Object.fromEntries(classNames.map((_, id) => [String(id), id === 24 ? 0.15 : 0.1]))},
  tile_plan: {tile_size: 800, stride: 800, network_input_size: 1024, tile_count: 169, global_class_wise_nms_iou: 0.7},
  mapped_detections: mapped, after_global_nms: boxes.map((item) => ({...item, stage: "after_global_nms"})), final_detections: boxes,
  stage_counts: {raw: boxes.length, mapped: boxes.length, after_global_nms: boxes.length, final: boxes.length},
};
const groundTruths = boxes.map(({confidence, source_tile_ids, source_detection_ids, stage, kept_by_global_nms, kept_by_class_threshold, ...item}) => item);

const json = (response, body, status = 200) => {
  response.writeHead(status, {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"});
  response.end(JSON.stringify(body));
};

createServer((request, response) => {
  const url = new URL(request.url, "http://127.0.0.1:8000");
  if (url.pathname === "/health") return json(response, {status: "ok"});
  if (url.pathname === "/classes") return json(response, {fine_classes: classNames.map((name, id) => ({id, name}))});
  if (url.pathname === "/runs") return json(response, [run]);
  if (url.pathname === `/runs/${run.run_id}`) return json(response, run);
  if (url.pathname === `/runs/${run.run_id}/predictions`) return json(response, prediction);
  if (url.pathname === `/runs/${run.run_id}/metrics`) return json(response, {strict_25: {available: true, overall: {tp: 55, fp: 2, fn: 1, ground_truth: 56, predictions: 57, recall: 55 / 56, precision: 55 / 57, fdr: 2 / 57}, metadata: {fixture: true}}});
  if (url.pathname === `/runs/${run.run_id}/timing`) return json(response, {algorithm_pipeline_seconds: 7.27, strict_e2e_seconds: 8.29, web_task_seconds: 8.31, model_inference_seconds: 7.25});
  if (url.pathname === `/runs/${run.run_id}/presentation`) return json(response, {
    image: prediction.image,
    tile_count: prediction.tile_plan.tile_count,
    stage_counts: prediction.stage_counts,
    broad_class_counts: boxes.reduce((counts, item) => ({...counts, [item.broad_class]: counts[item.broad_class] + 1}), {ship: 0, aircraft: 0, vehicle: 0}),
    timing: {algorithm_pipeline_seconds: 7.27, strict_e2e_seconds: 8.29},
  });
  if (url.pathname === `/runs/${run.run_id}/ground-truth`) return json(response, {image_id: prediction.image.image_id, ground_truths: groundTruths});
  if (url.pathname === `/runs/${run.run_id}/image`) {
    response.writeHead(200, {"Content-Type": "image/jpeg", "Content-Length": statSync(imagePath).size});
    return createReadStream(imagePath).pipe(response);
  }
  return json(response, {detail: "Not found"}, 404);
}).listen(mockPort, "127.0.0.1", () => console.log(`RS-CalVision UI-QA fixture at http://127.0.0.1:${mockPort}`));
