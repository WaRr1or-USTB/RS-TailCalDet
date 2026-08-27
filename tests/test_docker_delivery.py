import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from inference_core import Candidate, apply_class_thresholds, class_aware_nms, load_class_thresholds, tile_positions
from main import build_image_record, discover_images, write_result


class FakeImage:
    width = 10000
    height = 10000


class DockerDeliveryTests(unittest.TestCase):
    def test_tile_grid_aligns_the_last_boundary(self):
        positions = tile_positions(10000, 10000, 800, 800)
        self.assertEqual(len(positions), 169)
        self.assertEqual(positions[0], (0, 0))
        self.assertEqual(positions[-1], (9200, 9200))

    def test_nms_is_class_aware(self):
        candidates = [
            Candidate(cls=0, conf=0.9, box=(0.0, 0.0, 10.0, 10.0)),
            Candidate(cls=0, conf=0.8, box=(0.5, 0.5, 10.5, 10.5)),
            Candidate(cls=1, conf=0.7, box=(0.5, 0.5, 10.5, 10.5)),
        ]
        kept = class_aware_nms(candidates, 0.70)
        self.assertEqual([(item.cls, item.conf) for item in kept], [(0, 0.9), (1, 0.7)])

    def test_per_class_thresholds_are_applied(self):
        candidates = [
            Candidate(cls=0, conf=0.49, box=(0.0, 0.0, 1.0, 1.0)),
            Candidate(cls=1, conf=0.49, box=(2.0, 2.0, 3.0, 3.0)),
        ]
        kept = apply_class_thresholds(candidates, {0: 0.5, 1: 0.4})
        self.assertEqual([candidate.cls for candidate in kept], [1])

    def test_threshold_file_has_all_25_classes(self):
        thresholds = load_class_thresholds(PROJECT_ROOT / "models" / "class_thresholds.json")
        self.assertEqual(set(thresholds), set(range(25)))

    def test_result_json_matches_official_top_level_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            record = build_image_record(
                Path("example.png"),
                FakeImage(),
                [{"category_id": 0, "category_name": "HM", "score": 0.9, "bbox": [1, 2, 3, 4]}],
                123456789,
            )
            result_path = write_result(output_dir, [record])
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["images"][0]["image_id"], "example")
        self.assertEqual(payload["images"][0]["run_end_timestamp"], 123456789)
        self.assertEqual(payload["images"][0]["objects"][0]["bbox"], [1, 2, 3, 4])

    def test_input_discovery_is_non_recursive_and_filters_extensions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "b.PNG").write_bytes(b"")
            (input_dir / "a.jpg").write_bytes(b"")
            (input_dir / "note.txt").write_text("ignored", encoding="utf-8")
            nested = input_dir / "nested"
            nested.mkdir()
            (nested / "hidden.jpg").write_bytes(b"")
            images = discover_images(input_dir)

        self.assertEqual([path.name for path in images], ["a.jpg", "b.PNG"])


if __name__ == "__main__":
    unittest.main()
