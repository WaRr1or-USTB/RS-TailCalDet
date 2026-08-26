import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_category_experts import (
    resolve_overlapping_predictions,
    route_expert_predictions,
)
from evaluate_category_experts import RoutedPrediction
from evaluate_official_thresholds import Prediction


def prediction(cls, conf, box=(0.0, 0.0, 10.0, 10.0), index=0):
    return Prediction(image_index=0, pred_index=index, cls=cls, conf=conf, box=box)


class CategoryExpertTests(unittest.TestCase):
    def test_routes_hrsc_ship_and_vehicle_and_old160_aircraft(self):
        routed = route_expert_predictions(
            [prediction(0, 0.8), prediction(5, 0.9), prediction(24, 0.7)],
            [prediction(0, 0.9), prediction(5, 0.8), prediction(24, 0.9)],
            {0: 0.1, 24: 0.15},
            {5: 0.2},
        )
        self.assertEqual([(item.source, item.prediction.cls) for item in routed], [("hrsc", 0), ("hrsc", 24), ("old160", 5)])

    def test_applies_each_experts_own_thresholds(self):
        routed = route_expert_predictions(
            [prediction(2, 0.29), prediction(24, 0.15)],
            [prediction(5, 0.79), prediction(6, 0.14)],
            {2: 0.30, 24: 0.15},
            {5: 0.80, 6: 0.14},
        )
        self.assertEqual([item.prediction.cls for item in routed], [24, 6])

    def test_overlap_keeps_higher_raw_confidence(self):
        ship = RoutedPrediction(prediction(0, 0.60), "hrsc", 0.10)
        aircraft = RoutedPrediction(prediction(5, 0.90), "old160", 0.80)
        kept = resolve_overlapping_predictions([aircraft, ship], 0.70)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].prediction.cls, 5)

    def test_non_overlapping_predictions_are_preserved(self):
        first = RoutedPrediction(prediction(0, 0.8), "hrsc", 0.1)
        second = RoutedPrediction(prediction(5, 0.9, box=(20.0, 20.0, 30.0, 30.0)), "old160", 0.8)
        self.assertEqual(len(resolve_overlapping_predictions([first, second], 0.70)), 2)


if __name__ == "__main__":
    unittest.main()
