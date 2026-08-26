import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_big_images_sliding import build_protocol_metrics, build_v16_category_macro
from evaluate_official_thresholds import GroundTruth, Prediction


def metric(name, ground_truth, recall, fdr):
    tp = round(ground_truth * recall)
    predictions = tp if fdr == 0 else round(tp / (1 - fdr))
    return {
        "name": name,
        "ground_truth": ground_truth,
        "tp": tp,
        "fp": predictions - tp,
        "fn": ground_truth - tp,
        "recall": recall,
        "fdr": fdr,
        "precision": 1 - fdr,
    }


class V16ScoringTests(unittest.TestCase):
    def test_ship_macro_is_an_equal_weight_subtype_average(self):
        strict = {str(cls): metric(str(cls), 1, 1.0, 0.0) for cls in range(25)}
        strict.update(
            {
                "0": metric("HM", 10, 0.2, 0.5),
                "1": metric("LQS", 20, 0.4, 0.25),
                "2": metric("QHS", 30, 0.6, 0.75),
                "3": metric("MS", 40, 0.8, 0.0),
            }
        )

        ship = build_v16_category_macro(strict)["categories"]["ship"]

        self.assertAlmostEqual(ship["macro_recall"], 0.5)
        self.assertAlmostEqual(ship["macro_fdr"], 0.375)
        self.assertEqual(ship["evaluated_subclass_count"], 4)
        self.assertTrue(ship["is_complete"])

    def test_missing_subtype_is_reported_not_scored_as_zero(self):
        strict = {str(cls): metric(str(cls), 1, 1.0, 0.0) for cls in range(25)}
        strict["0"] = metric("HM", 0, 0.0, 0.0)
        strict["1"] = metric("LQS", 10, 0.4, 0.2)
        strict["2"] = metric("QHS", 10, 0.6, 0.4)
        strict["3"] = metric("MS", 10, 0.8, 0.6)

        ship = build_v16_category_macro(strict)["categories"]["ship"]

        self.assertAlmostEqual(ship["macro_recall"], 0.6)
        self.assertEqual(ship["missing_ground_truth_class_ids"], [0])
        self.assertFalse(ship["is_complete"])

    def test_v16_macro_requires_exact_subclass_match(self):
        class_names = {cls: str(cls) for cls in range(25)}
        gt_by_image = [[GroundTruth(cls=0, box=(0.0, 0.0, 10.0, 10.0))]]
        predictions = [
            Prediction(image_index=0, pred_index=0, cls=1, conf=0.9, box=(0.0, 0.0, 10.0, 10.0))
        ]

        metrics = build_protocol_metrics(
            gt_by_image=gt_by_image,
            predictions=predictions,
            threshold=0.001,
            target_recall=0.85,
            max_fdr=0.20,
            class_names=class_names,
        )

        self.assertEqual(metrics["official_by_category"]["categories"]["ship"]["tp"], 1)
        ship = metrics["v16_category_macro"]["categories"]["ship"]
        self.assertEqual(ship["macro_recall"], 0.0)
        self.assertEqual(ship["subclasses"][0]["tp"], 0)


if __name__ == "__main__":
    unittest.main()
