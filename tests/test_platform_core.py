import json
import sys
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_big_images_sliding import tile_positions
from rs_calvision.domain import BBox, DetectionStage, FinalDetection, GroundTruth, PipelineStage
from rs_calvision.evaluation import EvaluationAdapter
from rs_calvision.golden import compare_detections
from rs_calvision.inference import (
    BigImageInferenceAdapter,
    RuntimeDetection,
    classwise_hard_nms,
    plan_tiles,
)
from rs_calvision.manifest import MethodRegistry
from rs_calvision.serialization import PredictionSerializer, build_presentation_summary


class FakeRuntime:
    def __init__(self):
        self.calls = 0
        self.loads = 0

    def load(self):
        self.loads += 1

    def predict_tiles(self, tiles):
        rows = []
        for _ in tiles:
            if self.calls == 0:
                rows.append([
                    RuntimeDetection(0, 0.9, BBox(150, 100, 250, 200)),
                    RuntimeDetection(24, 0.003, BBox(300, 300, 350, 350)),
                ])
            else:
                rows.append([RuntimeDetection(0, 0.8, BBox(50, 100, 150, 200))])
            self.calls += 1
        return rows


class PlatformCoreTests(unittest.TestCase):
    def test_presentation_summary_uses_persisted_detection_artifacts(self):
        prediction = {
            "image": {"filename": "demo.jpg", "width": 10000, "height": 10000},
            "tile_plan": {"tile_count": 169},
            "stage_counts": {"raw": 12, "mapped": 12, "after_global_nms": 8, "final": 6},
            "final_detections": [
                {"broad_class": "ship"},
                {"broad_class": "ship"},
                {"broad_class": "aircraft"},
            ],
        }
        summary = build_presentation_summary(prediction, {"algorithm_pipeline_seconds": 7.2, "strict_e2e_seconds": 8.3})
        self.assertEqual(summary["broad_class_counts"], {"ship": 2, "aircraft": 1, "vehicle": 0})
        self.assertEqual(summary["stage_counts"]["final"], 6)
        self.assertEqual(summary["tile_count"], 169)

    @classmethod
    def setUpClass(cls):
        cls.method = MethodRegistry().get("rs-tailcaldet-v1")

    def test_manifest_freezes_audited_method_identity(self):
        self.assertEqual(self.method.display_name, "RS-TailCalDet")
        self.assertEqual(self.method.raw["base_detector"]["family"], "YOLO26x")
        self.assertTrue(self.method.raw["base_detector"]["end2end"])
        self.assertEqual(self.method.thresholds[24], 0.004)
        self.assertEqual(self.method.thresholds[0], 0.002)

    def test_tile_plan_matches_legacy_border_coverage(self):
        for width, height in ((640, 480), (1600, 1600), (1700, 901), (801, 800)):
            current = [(tile.offset_x, tile.offset_y) for tile in plan_tiles(width, height, 800, 800, 1024)]
            self.assertEqual(current, tile_positions(width, height, 800, 800))

    def test_global_nms_matches_audited_classwise_order(self):
        from rs_calvision.domain import MappedDetection
        boxes = [
            (0, 0.9, BBox(0, 0, 100, 100)),
            (0, 0.8, BBox(2, 2, 98, 98)),
            (1, 0.7, BBox(2, 2, 98, 98)),
            (0, 0.6, BBox(200, 200, 250, 250)),
        ]
        mapped = [MappedDetection(f"m{i}", "img", f"t{i}", cls, str(cls), conf, box, box) for i, (cls, conf, box) in enumerate(boxes)]
        kept, lineage = classwise_hard_nms(mapped, 0.7)
        current = [(mapped[index].class_id, mapped[index].confidence, mapped[index].bbox_global_xyxy.as_xyxy()) for index in kept]
        expected = [(0, 0.9, (0, 0, 100, 100)), (1, 0.7, (2, 2, 98, 98)), (0, 0.6, (200, 200, 250, 250))]
        self.assertTrue(compare_detections(expected, current)["passed"])
        self.assertEqual(lineage[0], [0, 1])

    def test_headless_inference_preserves_all_stages_and_lineage(self):
        runtime = FakeRuntime()
        path = PROJECT_ROOT / "data_v2" / "images" / "train" / "01-PAN-20240423-029-368-L00000011028-CCD29_4_crop2.jpg"
        events = []
        result = BigImageInferenceAdapter(self.method, runtime).infer(path, "test-run", event_callback=events.append)
        self.assertEqual(runtime.loads, 1)
        self.assertEqual(result.stage_counts, {"raw": 2, "mapped": 2, "after_global_nms": 2, "final": 1})
        self.assertEqual(len(result.final_detections[0].source_detection_ids), 1)
        self.assertEqual(result.final_detections[0].stage, DetectionStage.FINAL)
        data = PredictionSerializer().to_dict(PredictionSerializer().build_artifact(result, self.method))
        self.assertEqual(data["final_detections"][0]["width"], 100.0)
        self.assertEqual(data["stage_counts"]["raw"], 2)
        stages = {event.stage for event in events}
        self.assertTrue({PipelineStage.IMAGE_LOADING, PipelineStage.TILE_PLANNING, PipelineStage.INFERENCE_PROGRESS, PipelineStage.COORDINATE_MAPPING, PipelineStage.GLOBAL_NMS, PipelineStage.THRESHOLD_FILTERING}.issubset(stages))
        progress = next(event for event in events if event.stage == PipelineStage.INFERENCE_PROGRESS)
        self.assertEqual(progress.payload["total_tiles"], 1)
        nms_completed = [event for event in events if event.stage == PipelineStage.GLOBAL_NMS][-1]
        self.assertEqual(nms_completed.payload["detections_before_nms"], 2)

    def test_evaluation_protocols_do_not_collapse(self):
        prediction = FinalDetection(
            "p1", "img", 1, "LQS", "ship", 0.9, BBox(0, 0, 10, 10),
            ("t1",), ("r1",), DetectionStage.FINAL, True, True,
        )
        truth = GroundTruth("g1", "img", 0, "HM", "ship", BBox(0, 0, 10, 10))
        results = EvaluationAdapter(self.method).evaluate([prediction], [truth])
        self.assertEqual(results["strict_25"].overall.tp, 0)
        self.assertEqual(results["official_overall"].overall.tp, 1)
        self.assertEqual(results["official_by_category"].broad_categories["ship"].tp, 1)
        self.assertFalse(results["ultralytics_val"].available)
        self.assertFalse(results["strict_25"].metadata["authority_verified"])


if __name__ == "__main__":
    unittest.main()
