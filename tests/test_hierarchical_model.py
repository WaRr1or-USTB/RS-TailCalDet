import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_hierarchical_dataset import Label, box_iou, crop_box, fine_to_broad


class HierarchicalDatasetTests(unittest.TestCase):
    def test_official_fine_to_broad_mapping(self):
        self.assertEqual([fine_to_broad(index) for index in range(25)], [0] * 4 + [1] * 20 + [2])

    def test_crop_is_square_until_clipped(self):
        box = crop_box(Label(4, 0.5, 0.5, 0.1, 0.2), 1000, 800, 1.5)
        self.assertEqual(box[2] - box[0], box[3] - box[1])

    def test_crop_clips_at_image_boundary(self):
        box = crop_box(Label(24, 0.01, 0.02, 0.2, 0.1), 100, 80, 2.0)
        self.assertEqual(box[0], 0)
        self.assertEqual(box[1], 0)
        self.assertLessEqual(box[2], 100)
        self.assertLessEqual(box[3], 80)

    def test_box_iou(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


if __name__ == "__main__":
    unittest.main()
