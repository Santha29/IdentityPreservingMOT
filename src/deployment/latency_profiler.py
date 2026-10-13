"""Micro-benchmark and Latency Profiling Suite for Real-Time Embedded Systems.

Measures stage-by-stage execution latency:
1. Kalman Filter Prediction
2. Occlusion Detection & Overlap Computation
3. Cost Matrix Construction (IoU + Appearance + Gating)
4. Hungarian Assignment (Scipy / Linear Sum Assignment)
5. Kalman Filter Joseph-Form Update
6. Appearance Memory & Gallery Updates
7. End-to-end P50, P95, P99 frame latency distributions
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any


class LatencyProfiler:
    """Profiles tracking runtime with microsecond precision."""

    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.budget_ms = 1000.0 / target_fps
        self.stage_times: Dict[str, List[float]] = {
            "kalman_predict": [],
            "occlusion_detect": [],
            "cost_matrix_calc": [],
            "hungarian_matching": [],
            "kalman_update": [],
            "appearance_update": [],
            "total_frame": [],
        }

    def record_stage(self, stage_name: str, duration_ms: float):
        if stage_name in self.stage_times:
            self.stage_times[stage_name].append(duration_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Compute statistical breakdown across all recorded frames."""
        summary = {}
        for stage, times in self.stage_times.items():
            if len(times) == 0:
                continue
            arr = np.array(times)
            summary[stage] = {
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "max_ms": float(np.max(arr)),
            }

        total_times = np.array(self.stage_times["total_frame"]) if len(self.stage_times["total_frame"]) > 0 else np.array([0.1])
        mean_total = np.mean(total_times)

        summary["system"] = {
            "frames_profiled": len(total_times),
            "mean_fps": float(1000.0 / max(0.01, mean_total)),
            "budget_ms": self.budget_ms,
            "budget_compliance_ratio": float(np.mean(total_times <= self.budget_ms)),
            "is_real_time": bool(mean_total <= self.budget_ms)
        }
        return summary

    def format_table(self) -> str:
        """Format profiling results into a readable text table."""
        summary = self.get_summary()
        lines = [
            "+" + "-" * 32 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+",
            f"| {'Pipeline Stage':<30} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Max (ms)':<10} |",
            "+" + "-" * 32 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+",
        ]
        for stage in ["kalman_predict", "occlusion_detect", "cost_matrix_calc",
                      "hungarian_matching", "kalman_update", "appearance_update", "total_frame"]:
            if stage in summary:
                s = summary[stage]
                lines.append(f"| {stage:<30} | {s['mean_ms']:<10.3f} | {s['p95_ms']:<10.3f} | {s['max_ms']:<10.3f} |")

        lines.append("+" + "-" * 32 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
        sys = summary.get("system", {})
        lines.append(f"| Mean FPS: {sys.get('mean_fps', 0):<6.1f} | Budget: {sys.get('budget_ms', 0):.1f}ms | Real-Time: {sys.get('is_real_time', False)!s:<5} |")
        lines.append("+" + "-" * 70 + "+")
        return "\n".join(lines)
