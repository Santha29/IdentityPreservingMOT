"""Occlusion and Crowd Density Stress Testing Suite.

Executes controlled stress experiments across:
1. Increasing Crowd Densities (e.g. N = 5 to 50 persons)
2. Increasing Occlusion Severities (Sparse -> Moderate -> Dense Bottleneck)
3. Direct comparative evaluation: IP-MOT vs Baseline SORT
4. Exports tabular metrics and visualization plots
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from ..simulation.crowded_gate_sim import CrowdedGateSimulator
from ..tracker.ip_mot_tracker import IPMOTTracker
from ..tracker.baseline_sort import BaselineSortTracker
from .metrics import MOTMetricsCalculator


class OcclusionStressTester:
    """Automated benchmark harness for occlusion and crowd density stress testing."""

    def __init__(
        self,
        densities: Optional[List[int]] = None,
        occlusion_severities: Optional[List[float]] = None,
        num_frames: int = 150,
        random_seed: int = 42
    ):
        self.densities = densities if densities is not None else [5, 10, 15, 20, 25, 30, 40]
        self.occlusion_severities = occlusion_severities if occlusion_severities is not None else [0.25, 0.50, 0.75]
        self.num_frames = num_frames
        self.random_seed = random_seed
        self.calculator = MOTMetricsCalculator(iou_threshold=0.50)

    def run_single_experiment(
        self,
        num_pedestrians: int,
        occlusion_severity: float,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Run a single test scenario comparing IP-MOT and Baseline SORT on identical data."""
        # 1. Generate scenario
        sim = CrowdedGateSimulator(
            num_pedestrians=num_pedestrians,
            num_frames=self.num_frames,
            occlusion_severity=occlusion_severity,
            random_seed=seed
        )
        dataset = sim.generate_dataset()
        gt = dataset["ground_truth"]
        dets = dataset["detections"]

        # 2. Run IP-MOT
        ip_tracker = IPMOTTracker(
            det_high_thresh=0.60,
            det_low_thresh=0.20,
            appearance_weight=0.50,
            reid_appearance_thresh=0.35,
            max_lost_age=45
        )
        ip_tracker.reset()

        ip_results: Dict[int, List[Dict[str, Any]]] = {}
        ip_latencies = []

        for f in range(1, self.num_frames + 1):
            frame_dets = dets[f]
            t0 = time.perf_counter()
            tracks = ip_tracker.update(
                boxes_xyxy=frame_dets["boxes"],
                scores=frame_dets["scores"],
                features=frame_dets["features"]
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            ip_latencies.append(lat_ms)

            ip_results[f] = [{
                "id": t.track_id,
                "bbox": t.get_box_xyxy(),
                "score": t.score
            } for t in tracks]

        ip_metrics = self.calculator.evaluate(gt, ip_results)
        ip_metrics["mean_latency_ms"] = float(np.mean(ip_latencies))
        ip_metrics["fps"] = float(1000.0 / max(0.1, np.mean(ip_latencies)))

        # 3. Run Baseline SORT
        sort_tracker = BaselineSortTracker(
            max_age=10,
            min_hits=3,
            iou_threshold=0.30
        )
        sort_tracker.reset()

        sort_results: Dict[int, List[Dict[str, Any]]] = {}
        sort_latencies = []

        for f in range(1, self.num_frames + 1):
            frame_dets = dets[f]
            # Baseline SORT only considers detections above high threshold
            mask = frame_dets["scores"] >= 0.50
            boxes = frame_dets["boxes"][mask] if len(frame_dets["boxes"]) > 0 else np.empty((0, 4))

            t0 = time.perf_counter()
            tracks = sort_tracker.update(boxes_xyxy=boxes)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            sort_latencies.append(lat_ms)

            sort_results[f] = [{
                "id": t.track_id,
                "bbox": t.get_box_xyxy(),
                "score": 1.0
            } for t in tracks]

        sort_metrics = self.calculator.evaluate(gt, sort_results)
        sort_metrics["mean_latency_ms"] = float(np.mean(sort_latencies))
        sort_metrics["fps"] = float(1000.0 / max(0.1, np.mean(sort_latencies)))

        return {
            "density": num_pedestrians,
            "occlusion_severity": occlusion_severity,
            "ip_mot": ip_metrics,
            "baseline_sort": sort_metrics
        }

    def run_density_sweep(
        self,
        fixed_occlusion_severity: float = 0.50,
        output_dir: str = "assets"
    ) -> pd.DataFrame:
        """Sweep crowd density and record metrics degradation."""
        records = []
        os.makedirs(output_dir, exist_ok=True)

        for n in self.densities:
            res = self.run_single_experiment(
                num_pedestrians=n,
                occlusion_severity=fixed_occlusion_severity,
                seed=self.random_seed
            )
            rec = {
                "Density": n,
                "Occlusion_Severity": fixed_occlusion_severity,
                # IP-MOT
                "IP_MOT_IDF1": res["ip_mot"]["IDF1"] * 100.0,
                "IP_MOT_MOTA": res["ip_mot"]["MOTA"] * 100.0,
                "IP_MOT_HOTA": res["ip_mot"]["HOTA"] * 100.0,
                "IP_MOT_IDSW": res["ip_mot"]["IDSW"],
                "IP_MOT_Latency_ms": res["ip_mot"]["mean_latency_ms"],
                "IP_MOT_FPS": res["ip_mot"]["fps"],
                # Baseline SORT
                "SORT_IDF1": res["baseline_sort"]["IDF1"] * 100.0,
                "SORT_MOTA": res["baseline_sort"]["MOTA"] * 100.0,
                "SORT_HOTA": res["baseline_sort"]["HOTA"] * 100.0,
                "SORT_IDSW": res["baseline_sort"]["IDSW"],
                "SORT_Latency_ms": res["baseline_sort"]["mean_latency_ms"],
                "SORT_FPS": res["baseline_sort"]["fps"],
            }
            records.append(rec)

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_dir, "stress_test_density_results.csv"), index=False)
        self.plot_density_curves(df, output_path=os.path.join(output_dir, "stress_test_curves.png"))
        return df

    def plot_density_curves(self, df: pd.DataFrame, output_path: str = "assets/stress_test_curves.png"):
        """Generate high-resolution 4-panel comparison plots."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)

        densities = df["Density"]

        # 1. IDF1 Comparison (Key Metric for Identity Preservation)
        ax = axes[0, 0]
        ax.plot(densities, df["IP_MOT_IDF1"], "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="IP-MOT (Ours)")
        ax.plot(densities, df["SORT_IDF1"], "s--", color="#d62728", linewidth=2.2, markersize=7, label="Baseline SORT")
        ax.set_title("Identity Preservation (IDF1) vs Crowd Density", fontsize=12, fontweight="bold")
        ax.set_xlabel("Crowd Density (Pedestrians in Gate)", fontsize=10)
        ax.set_ylabel("IDF1 Score (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10, loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)

        # 2. ID Switches Comparison
        ax = axes[0, 1]
        ax.plot(densities, df["IP_MOT_IDSW"], "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="IP-MOT (Ours)")
        ax.plot(densities, df["SORT_IDSW"], "s--", color="#d62728", linewidth=2.2, markersize=7, label="Baseline SORT")
        ax.set_title("ID Switches vs Crowd Density (Lower is Better)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Crowd Density (Pedestrians in Gate)", fontsize=10)
        ax.set_ylabel("Total ID Switches", fontsize=10)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)

        # 3. HOTA Comparison (Higher Order Tracking Accuracy)
        ax = axes[1, 0]
        ax.plot(densities, df["IP_MOT_HOTA"], "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="IP-MOT (Ours)")
        ax.plot(densities, df["SORT_HOTA"], "s--", color="#d62728", linewidth=2.2, markersize=7, label="Baseline SORT")
        ax.set_title("HOTA Accuracy vs Crowd Density", fontsize=12, fontweight="bold")
        ax.set_xlabel("Crowd Density (Pedestrians in Gate)", fontsize=10)
        ax.set_ylabel("HOTA Score (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10, loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)

        # 4. Processing Latency & Real-Time Performance
        ax = axes[1, 1]
        ax.plot(densities, df["IP_MOT_Latency_ms"], "o-", color="#2ca02c", linewidth=2.2, markersize=6, label="IP-MOT Latency (ms)")
        ax.plot(densities, df["SORT_Latency_ms"], "s--", color="#7f7f7f", linewidth=2.0, markersize=6, label="SORT Latency (ms)")
        ax.axhline(y=33.33, color="orange", linestyle=":", label="30 FPS Threshold (33.3ms)")
        ax.set_title("Inference Latency vs Crowd Density (Bounded Latency)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Crowd Density (Pedestrians in Gate)", fontsize=10)
        ax.set_ylabel("Latency per Frame (ms)", fontsize=10)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)

        fig.suptitle("IP-MOT vs Baseline SORT: Dense Occlusion & Crowd Stress Analysis", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close(fig)
