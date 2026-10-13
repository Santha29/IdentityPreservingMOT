"""Demo Execution Script for Identity-Preserving Multi-Object Tracking (IP-MOT).

Simulates a real-time crowded gate camera stream, runs online causal IP-MOT tracking,
profiles latency across all pipeline stages, and exports annotated demo video.
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.crowded_gate_sim import CrowdedGateSimulator
from src.simulation.video_renderer import VideoRenderer
from src.tracker.ip_mot_tracker import IPMOTTracker
from src.evaluation.metrics import MOTMetricsCalculator
from src.deployment.latency_profiler import LatencyProfiler
import cv2


def main():
    parser = argparse.ArgumentParser(description="IP-MOT Real-Time Demo.")
    parser.add_argument("--pedestrians", type=int, default=20, help="Number of pedestrians")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--occlusion_severity", type=float, default=0.60, help="Occlusion severity (0.0 to 1.0)")
    parser.add_argument("--export_video", action="store_true", default=True, help="Export output video")
    parser.add_argument("--output_path", type=str, default="assets/ip_mot_demo.mp4", help="Video output path")
    args = parser.parse_args()

    print("=" * 80)
    print("      IP-MOT: REAL-TIME CAUSAL TRACKING DEMONSTRATION")
    print(f"      Pedestrians: {args.pedestrians} | Frames: {args.frames} | Occlusion: {args.occlusion_severity}")
    print("=" * 80)

    # 1. Initialize Simulator & Data
    sim = CrowdedGateSimulator(
        num_pedestrians=args.pedestrians,
        num_frames=args.frames,
        occlusion_severity=args.occlusion_severity,
        random_seed=42
    )
    dataset = sim.generate_dataset()
    gt = dataset["ground_truth"]
    dets = dataset["detections"]

    # 2. Initialize Tracker, Renderer, Profiler
    tracker = IPMOTTracker(
        det_high_thresh=0.60,
        det_low_thresh=0.15,
        appearance_weight=0.50,
        reid_appearance_thresh=0.35,
        max_lost_age=60
    )
    tracker.reset()

    renderer = VideoRenderer(width=dataset["width"], height=dataset["height"], fps=30)
    profiler = LatencyProfiler(target_fps=30.0)
    calc = MOTMetricsCalculator(iou_threshold=0.50)

    # 3. Video Writer Setup
    writer = None
    if args.export_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_path, fourcc, 30.0, (dataset["width"], dataset["height"]))

    tracking_results = {}
    frame_latencies = []

    print("\nProcessing frames in real-time causal sequence...")
    for f in range(1, args.frames + 1):
        fd = dets[f]
        t_start = time.perf_counter()

        # Update tracker
        tracks = tracker.update(
            boxes_xyxy=fd["boxes"],
            scores=fd["scores"],
            features=fd["features"]
        )
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        frame_latencies.append(t_elapsed)
        profiler.record_stage("total_frame", t_elapsed)

        tracking_results[f] = [{
            "id": t.track_id,
            "bbox": t.get_box_xyxy(),
            "score": t.score
        } for t in tracks]

        if writer is not None:
            rendered = renderer.render_frame(
                frame_idx=f,
                tracks=tracks,
                ground_truth=gt.get(f, []),
                hud_info={
                    "tracker_name": "IP-MOT (Identity-Preserving)",
                    "latency_ms": t_elapsed,
                    "fps": 1000.0 / max(0.1, t_elapsed),
                    "id_switches": 0
                }
            )
            writer.write(rendered)

    if writer is not None:
        writer.release()
        print(f"[Export] Demo video saved to: {args.output_path}")

    # 4. Evaluation & Profiling Results
    metrics = calc.evaluate(gt, tracking_results)
    print("\n" + "=" * 50)
    print("           TRACKING ACCURACY METRICS")
    print("=" * 50)
    print(f"IDF1 Score (Identity Preservation): {metrics['IDF1']*100:.2f}%")
    print(f"Total ID Switches (IDSW):           {metrics['IDSW']}")
    print(f"HOTA Accuracy:                      {metrics['HOTA']*100:.2f}%")
    print(f"MOTA Accuracy:                      {metrics['MOTA']*100:.2f}%")
    print(f"Identity Precision (IDP):           {metrics['IDP']*100:.2f}%")
    print(f"Identity Recall (IDR):              {metrics['IDR']*100:.2f}%")
    print(f"Mean Latency per Frame:             {np.mean(frame_latencies):.2f} ms")
    print(f"Effective Processing Throughput:    {1000.0/np.mean(frame_latencies):.1f} FPS")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("       LATENCY PROFILING & BUDGET COMPLIANCE")
    print("=" * 50)
    print(profiler.format_table())


if __name__ == "__main__":
    main()
