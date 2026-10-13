"""CLI Script to Execute Occlusion and Crowd Density Stress Testing.

Sweeps pedestrian density from low (5) to high (40) crowded gate conditions,
evaluating both IP-MOT and Baseline SORT across standard MOT metrics (IDF1, HOTA, MOTA, IDSW, Latency).
"""

import os
import sys
import argparse
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.stress_test import OcclusionStressTester


def main():
    parser = argparse.ArgumentParser(description="Run MOT Occlusion & Crowd Density Stress Testing.")
    parser.add_argument("--config", type=str, default="configs/stress_test_config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="assets", help="Directory for output curves and CSV")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames per scenario")
    args = parser.parse_args()

    densities = [5, 10, 15, 20, 25, 30, 40]
    occlusion_severity = 0.50

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f).get("stress_test", {})
            densities = cfg.get("densities", densities)
            occlusion_severity = cfg.get("occlusion_severities", [0.50])[1] if len(cfg.get("occlusion_severities", [])) > 1 else 0.50
            args.frames = cfg.get("num_frames", args.frames)

    print("=" * 80)
    print("      IDENTITY-PRESERVING MULTI-OBJECT TRACKING (IP-MOT) STRESS TEST")
    print(f"      Pedestrian Densities: {densities}")
    print(f"      Occlusion Severity: {occlusion_severity} | Frames per Run: {args.frames}")
    print("=" * 80)

    tester = OcclusionStressTester(
        densities=densities,
        num_frames=args.frames,
        random_seed=42
    )

    print("\nRunning Density Sweep Benchmark (IP-MOT vs Baseline SORT)...")
    df = tester.run_density_sweep(fixed_occlusion_severity=occlusion_severity, output_dir=args.output_dir)

    print("\nBenchmark Execution Complete! Results Table:")
    print("-" * 92)
    print(f"{'Density':<8} | {'IP-MOT IDF1':<12} | {'SORT IDF1':<10} | {'IP-MOT IDSW':<12} | {'SORT IDSW':<10} | {'IP-MOT HOTA':<12} | {'SORT HOTA':<10} | {'FPS':<6}")
    print("-" * 92)
    for _, row in df.iterrows():
        print(f"{int(row['Density']):<8} | "
              f"{row['IP_MOT_IDF1']:<11.1f}% | "
              f"{row['SORT_IDF1']:<9.1f}% | "
              f"{int(row['IP_MOT_IDSW']):<12} | "
              f"{int(row['SORT_IDSW']):<10} | "
              f"{row['IP_MOT_HOTA']:<11.1f}% | "
              f"{row['SORT_HOTA']:<9.1f}% | "
              f"{row['IP_MOT_FPS']:<6.0f}")
    print("-" * 92)

    plot_path = os.path.join(args.output_dir, "stress_test_curves.png")
    csv_path = os.path.join(args.output_dir, "stress_test_density_results.csv")
    print(f"\n[Artifact] Comparison Curves saved to: {plot_path}")
    print(f"[Artifact] Metrics CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
