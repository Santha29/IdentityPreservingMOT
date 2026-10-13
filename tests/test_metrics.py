"""Unit tests for MOT metrics calculation (IDF1, MOTA, HOTA)."""

import unittest
import numpy as np
from src.evaluation.metrics import MOTMetricsCalculator


class TestMOTMetrics(unittest.TestCase):

    def setUp(self):
        self.calculator = MOTMetricsCalculator(iou_threshold=0.50)

    def test_perfect_tracking_metrics(self):
        # 10 frames, 2 targets perfectly tracked without ID switches
        ground_truth = {}
        tracker_results = {}

        for f in range(1, 11):
            box1 = np.array([10.0 + f * 2, 20.0, 50.0 + f * 2, 100.0], dtype=np.float32)
            box2 = np.array([200.0 - f * 2, 50.0, 240.0 - f * 2, 140.0], dtype=np.float32)

            ground_truth[f] = [
                {"id": 1, "bbox": box1},
                {"id": 2, "bbox": box2}
            ]
            tracker_results[f] = [
                {"id": 1, "bbox": box1, "score": 1.0},
                {"id": 2, "bbox": box2, "score": 1.0}
            ]

        metrics = self.calculator.evaluate(ground_truth, tracker_results)

        self.assertAlmostEqual(metrics["MOTA"], 1.0, places=4)
        self.assertAlmostEqual(metrics["IDF1"], 1.0, places=4)
        self.assertAlmostEqual(metrics["HOTA"], 1.0, places=4)
        self.assertEqual(metrics["IDSW"], 0)
        self.assertEqual(metrics["FP"], 0)
        self.assertEqual(metrics["FN"], 0)

    def test_id_switch_detection(self):
        # Frame 1: Trk 1 tracks GT 1
        # Frame 2: Trk 2 tracks GT 1 (An ID Switch occurred!)
        box = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float32)
        ground_truth = {
            1: [{"id": 1, "bbox": box}],
            2: [{"id": 1, "bbox": box}]
        }
        tracker_results = {
            1: [{"id": 10, "bbox": box, "score": 1.0}],
            2: [{"id": 20, "bbox": box, "score": 1.0}]  # Swapped tracker ID
        }

        metrics = self.calculator.evaluate(ground_truth, tracker_results)
        self.assertEqual(metrics["IDSW"], 1)
        self.assertLess(metrics["IDF1"], 1.0)


if __name__ == "__main__":
    unittest.main()
