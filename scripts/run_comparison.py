"""Side-by-Side Comparison Script: IP-MOT vs Baseline SORT.

Runs both trackers on identical challenging crowded gate scenarios with dense occlusions.
Outputs side-by-side metric tables and optionally generates a side-by-side comparative video.
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.crowded_gate_sim import CrowdedGateSimulator
from src.simulation.video_renderer import VideoRenderer
from src.tracker.ip_mot_tracker import IPMOTTracker
from src.tracker.baseline_sort import BaselineSortTracker
from src.evaluation.metrics import MOTMetricsCalculator


def main():
    parser = argparse.ArgumentParser(description="Side-by-Side MOT Comparison.")
    parser.add_argument("--pedestrians", type=int, default=24, help="Number of pedestrians in gate")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to simulate")
    parser.add_argument("--occlusion_severity", type=float, default=0.60, help="Occlusion severity (0.0 to 1.0)")
    parser.add_argument("--export_video", action="store_true", default=True, help="Export side-by-side MP4 video")
    parser.add_argument("--output_video", type=str, default="assets/comparison_demo.mp4", help="Video output path")
    args = parser.parse_args()

    print("=" * 80)
    print("        IP-MOT vs BASELINE SORT: HEAD-TO-HEAD COMPARISON")
    print(f"        Pedestrians: {args.pedestrians} | Frames: {args.frames} | Occlusion: {args.occlusion_severity}")
    print("=" * 80)

    # 1. Generate Dataset
    sim = CrowdedGateSimulator(
        num_pedestrians=args.pedestrians,
        num_frames=args.frames,
        occlusion_severity=args.occlusion_severity,
        random_seed=42
    )
    dataset = sim.generate_dataset()
    gt = dataset["ground_truth"]
    dets = dataset["detections"]

    # 2. Trackers
    ip_tracker = IPMOTTracker(
        det_high_thresh=0.60,
        det_low_thresh=0.20,
        appearance_weight=0.50,
        reid_appearance_thresh=0.35,
        max_lost_age=50
    )
    ip_tracker.reset()

    sort_tracker = BaselineSortTracker(
        max_age=10,
        min_hits=3,
        iou_threshold=0.30
    )
    sort_tracker.reset()

    renderer = VideoRenderer(width=dataset["width"], height=dataset["height"])
    calc = MOTMetricsCalculator(iou_threshold=0.50)

    ip_results = {}
    sort_results = {}
    ip_latencies = []
    sort_latencies = []

    # Video Writer for side-by-side comparison
    writer = None
    if args.export_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
        # Side-by-side width = width * 2, scaled down to 1920x540 or similar for display
        out_w = dataset["width"] * 2
        out_h = dataset["height"]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, 30.0, (out_w, out_h))

    print("\nExecuting frame-by-frame causal tracking...")
    for f in range(1, args.frames + 1):
        frame_dets = dets[f]

        # IP-MOT update
        t0 = time.perf_counter()
        ip_tracks = ip_tracker.update(
            boxes_xyxy=frame_dets["boxes"],
            scores=frame_dets["scores"],
            features=frame_dets["features"]
        )
        ip_lat = (time.perf_counter() - t0) * 1000.0
        ip_latencies.append(ip_lat)

        ip_results[f] = [{
            "id": t.track_id,
            "bbox": t.get_box_xyxy(),
            "score": t.score
        } for t in ip_tracks]

        # SORT update
        mask = frame_dets["scores"] >= 0.50
        sort_boxes = frame_dets["boxes"][mask] if len(frame_dets["boxes"]) > 0 else np.empty((0, 4))
        t0 = time.perf_counter()
        sort_tracks = sort_tracker.update(boxes_xyxy=sort_boxes)
        sort_lat = (time.perf_counter() - t0) * 1000.0
        sort_latencies.append(sort_lat)

        sort_results[f] = [{
            "id": t.track_id,
            "bbox": t.get_box_xyxy(),
            "score": 1.0
        } for t in sort_tracks]

        if writer is not None:
            # Render IP-MOT frame
            ip_frame = renderer.render_frame(
                frame_idx=f,
                tracks=ip_tracks,
                ground_truth=gt.get(f, []),
                hud_info={
                    "tracker_name": "IP-MOT (Ours: Fused Re-ID + Occlusion Shield)",
                    "latency_ms": ip_lat,
                    "fps": 1000.0 / max(0.1, ip_lat),
                    "id_switches": 0  # Running count updated at end
                }
            )

            # Render Baseline SORT frame
            sort_frame = renderer.render_frame(
                frame_idx=f,
                tracks=sort_tracks,
                ground_truth=gt.get(f, []),
                hud_info={
                    "tracker_name": "Baseline SORT (Bewley et al., Pure IoU)",
                    "latency_ms": sort_lat,
                    "fps": 1000.0 / max(0.1, sort_lat),
                    "id_switches": 0
                }
            )

            # Concatenate horizontally
            combined = np.hstack([ip_frame, sort_frame])
            writer.write(combined)

    if writer is not None:
        writer.release()
        print(f"[Success] Side-by-side comparative video exported to: {args.output_video}")

    # Compute final metrics
    ip_m = calc.evaluate(gt, ip_results)
    sort_m = calc.evaluate(gt, sort_results)

    print("\n" + "=" * 65)
    print(f"{'Metric':<25} | {'Baseline SORT':<16} | {'IP-MOT (Ours)':<16}")
    print("=" * 65)
    print(f"{'IDF1 (Identity F1)':<25} | {sort_m['IDF1']*100:<15.2f}% | {ip_m['IDF1']*100:<15.2f}%")
    print(f"{'ID Switches (IDSW)':<25} | {sort_m['IDSW']:<16} | {ip_m['IDSW']:<16}")
    print(f"{'HOTA Accuracy':<25} | {sort_m['HOTA']*100:<15.2f}% | {ip_m['HOTA']*100:<15.2f}%")
    print(f"{'MOTA Accuracy':<25} | {sort_m['MOTA']*100:<15.2f}% | {ip_m['MOTA']*100:<15.2f}%")
    print(f"{'Identity Precision (IDP)':<25} | {sort_m['IDP']*100:<15.2f}% | {ip_m['IDP']*100:<15.2f}%")
    print(f"{'Identity Recall (IDR)':<25} | {sort_m['IDR']*100:<15.2f}% | {ip_m['IDR']*100:<15.2f}%")
    print(f"{'Mostly Tracked (MT)':<25} | {sort_m['MT_ratio']*100:<15.2f}% | {ip_m['MT_ratio']*100:<15.2f}%")
    print(f"{'Mostly Lost (ML)':<25} | {sort_m['ML_ratio']*100:<15.2f}% | {ip_m['ML_ratio']*100:<15.2f}%")
    print(f"{'Mean Latency (ms)':<25} | {np.mean(sort_latencies):<15.2f}ms | {np.mean(ip_latencies):<15.2f}ms")
    print(f"{'Throughput (FPS)':<25} | {1000.0/np.mean(sort_latencies):<15.0f} FPS | {1000.0/np.mean(ip_latencies):<15.0f} FPS")
    print("=" * 65)

    idsw_reduction = max(0.0, (sort_m['IDSW'] - ip_m['IDSW']) / max(1, sort_m['IDSW'])) * 100.0
    idf1_boost = (ip_m['IDF1'] - sort_m['IDF1']) * 100.0
    print(f"\n[Key Achievement] ID Switch Reduction: -{idsw_reduction:.1f}%")
    print(f"[Key Achievement] IDF1 Improvement:    +{idf1_boost:.1f}% absolute points")


if __name__ == "__main__":
    main()
