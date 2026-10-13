"""MOTChallenge Benchmark Formatter and Evaluator.

Supports MOT16/17/20 standard format:
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .metrics import MOTMetricsCalculator


class MOTEvaluator:
    """Evaluates tracking outputs against ground-truth files or data structures."""

    def __init__(self, iou_threshold: float = 0.50):
        self.calculator = MOTMetricsCalculator(iou_threshold=iou_threshold)

    @staticmethod
    def export_mot_format(
        tracker_results: Dict[int, List[Dict[str, Any]]],
        output_filepath: str
    ):
        """Export tracker results to standard MOT challenge CSV format."""
        rows = []
        for frame in sorted(tracker_results.keys()):
            for trk in tracker_results[frame]:
                box = trk["bbox"]  # [x1, y1, x2, y2]
                w = max(0.0, box[2] - box[0])
                h = max(0.0, box[3] - box[1])
                score = trk.get("score", 1.0)
                rows.append([
                    int(frame),
                    int(trk["id"]),
                    float(box[0]),
                    float(box[1]),
                    float(w),
                    float(h),
                    float(score),
                    -1, -1, -1
                ])

        df = pd.DataFrame(rows, columns=[
            "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"
        ])
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        df.to_csv(output_filepath, index=False, header=False)

    @staticmethod
    def load_mot_format(filepath: str) -> Dict[int, List[Dict[str, Any]]]:
        """Load standard MOT challenge annotations or tracker results."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, header=None, names=[
            "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"
        ])

        results = {}
        for _, row in df.iterrows():
            f = int(row["frame"])
            if f not in results:
                results[f] = []
            x1 = float(row["bb_left"])
            y1 = float(row["bb_top"])
            x2 = x1 + float(row["bb_width"])
            y2 = y1 + float(row["bb_height"])
            results[f].append({
                "id": int(row["id"]),
                "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
                "score": float(row["conf"])
            })
        return results

    def evaluate_run(
        self,
        ground_truth: Dict[int, List[Dict[str, Any]]],
        tracker_results: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Evaluate tracking results against ground truth."""
        return self.calculator.evaluate(ground_truth, tracker_results)
