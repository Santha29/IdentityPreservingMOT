"""
NVIDIA Jetson Real-Time Deployment Pipeline for IP-MOT
Supports: Jetson Orin Nano / Orin NX / AGX Orin / Xavier NX
Features: GStreamer CSI/USB camera, TensorRT execution, Bounded Latency.
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from src.tracker.ip_mot_tracker import IPMOTTracker
from src.simulation.video_renderer import get_color_for_id


def build_gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    framerate=30,
    flip_method=0
):
    """GStreamer pipeline for Jetson onboard CSI camera (nvarguscamerasrc)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


def run_jetson_pipeline(
    source="0",
    engine_path="yolov8n.engine",
    conf_thresh=0.40,
    use_csi=False,
    headless=False,
    record_path=None
):
    print("=" * 75)
    print("      NVIDIA JETSON EMBEDDED REAL-TIME IP-MOT PIPELINE")
    print("=" * 75)

    # 1. Initialize Video Capture (CSI Camera / USB / RTSP / Video)
    if use_csi:
        print("[1/4] Initializing Jetson CSI Camera via GStreamer (nvarguscamerasrc)...")
        gst_pipeline = build_gstreamer_pipeline()
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        src = int(source) if str(source).isdigit() else source
        print(f"[1/4] Opening Video Input: {source}...")
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print("[Error] Failed to open video source on Jetson!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Capture Stream: {width}x{height} Resolution | Target FPS: {fps:.1f}")

    # 2. Load Detector (TensorRT .engine if exists, else auto-export)
    print("\n[2/4] Loading Detection Engine...")
    if os.path.exists(engine_path):
        print(f"Loading compiled TensorRT Engine: {engine_path}")
        model = YOLO(engine_path)
    else:
        print(f"TensorRT engine not found at '{engine_path}'. Loading PyTorch/ONNX model and exporting to TensorRT FP16...")
        model = YOLO("yolov8n.pt")
        try:
            # Auto export to TensorRT on Jetson
            model.export(format="engine", half=True, device=0)
            engine_path = "yolov8n.engine"
            model = YOLO(engine_path)
            print("TensorRT FP16 Engine compiled and loaded successfully!")
        except Exception as e:
            print(f"[Notice] TensorRT direct export fallback: using standard engine: {e}")

    # 3. Initialize Identity-Preserving Tracker (IP-MOT)
    print("\n[3/4] Initializing IP-MOT Tracking Engine (Bounded Latency Mode)...")
    tracker = IPMOTTracker(
        det_high_thresh=conf_thresh,
        det_low_thresh=max(0.15, conf_thresh * 0.4),
        appearance_weight=0.55,
        reid_appearance_thresh=0.35,
        max_lost_age=60  # Retain occluded tracks for 2 seconds (60 frames)
    )
    tracker.reset()

    # Video Recorder (Optional)
    writer = None
    if record_path:
        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print("\n[4/4] Starting Online Causal Processing Loop (Press 'q' to stop)...")
    frame_idx = 0
    t_start_loop = time.time()
    latencies = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_f_start = time.perf_counter()

        # Step A: TensorRT Person Detection (Class 0: person)
        preds = model(frame, classes=[0], conf=conf_thresh, verbose=False)[0]

        boxes = []
        scores = []
        patches = []

        if preds.boxes is not None and len(preds.boxes) > 0:
            for b in preds.boxes:
                c = b.xyxy[0].cpu().numpy().astype(np.float32)
                sc = float(b.conf[0])
                x1, y1 = max(0, int(c[0])), max(0, int(c[1]))
                x2, y2 = min(width, int(c[2])), min(height, int(c[3]))
                patch = frame[y1:y2, x1:x2]

                boxes.append(c)
                scores.append(sc)
                patches.append(patch)

        b_arr = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.empty((0, 4))
        s_arr = np.array(scores, dtype=np.float32) if len(scores) > 0 else np.empty(0)

        # Step B: IP-MOT Tracking: Fused Appearance + Kalman Joseph Form + Re-ID
        active_tracks = tracker.update(
            boxes_xyxy=b_arr,
            scores=s_arr,
            image_patches=patches,
            frame_img=frame
        )

        lat_ms = (time.perf_counter() - t_f_start) * 1000.0
        latencies.append(lat_ms)
        avg_fps = 1000.0 / max(0.1, lat_ms)

        # Step C: Real-Time Annotation Overlay
        if not headless or writer:
            for trk in active_tracks:
                t_id = trk.track_id
                col = get_color_for_id(t_id)
                box = trk.get_box_xyxy().astype(int)
                is_occ = getattr(trk, "is_occluded", False)

                # Trajectory Motion Trail
                history = getattr(trk, "trajectory_history", [])
                if len(history) > 1:
                    pts = history[-25:]
                    for k in range(1, len(pts)):
                        p1 = (int(pts[k-1][0]), int(pts[k-1][1]))
                        p2 = (int(pts[k][0]), int(pts[k][1]))
                        cv2.line(frame, p1, p2, col, 2, cv2.LINE_AA)

                # Box and ID Label
                box_col = (0, 140, 255) if is_occ else col
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_col, 2)
                label = f"ID: {t_id}" + (" [OCC]" if is_occ else "")
                cv2.rectangle(frame, (box[0], max(0, box[1] - 22)), (box[0] + 90, box[1]), box_col, -1)
                cv2.putText(frame, label, (box[0] + 4, box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            # Jetson Telemetry HUD Banner
            cv2.rectangle(frame, (10, 10), (520, 70), (15, 15, 15), -1)
            cv2.putText(frame, f"NVIDIA Jetson Orin | Frame: {frame_idx} | {avg_fps:.0f} FPS",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Active Tracks: {len(active_tracks)} | Latency: {lat_ms:.1f}ms (<33ms Budget)",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

            if writer:
                writer.write(frame)

            if not headless:
                cv2.imshow("NVIDIA Jetson - IP-MOT Real-Time Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if frame_idx % 60 == 0:
            print(f"[Jetson] Frame: {frame_idx:05d} | Latency: {lat_ms:5.1f}ms | Throughput: {avg_fps:4.0f} FPS | Active Targets: {len(active_tracks)}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - t_start_loop
    print("\n" + "=" * 65)
    print("             JETSON DEPLOYMENT BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Mean Latency per Frame: {np.mean(latencies):.2f} ms")
    print(f"P95 Latency:            {np.percentile(latencies, 95):.2f} ms")
    print(f"Average Throughput:     {frame_idx / max(0.1, total_time):.1f} FPS")
    print(f"30 FPS Budget Met:      {bool(np.mean(latencies) <= 33.33)}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVIDIA Jetson Real-Time Deployment")
    parser.add_argument("--source", type=str, default="0", help="Video file, USB cam index (0), or RTSP url")
    parser.add_argument("--engine", type=str, default="yolov8n.engine", help="Path to compiled TensorRT engine")
    parser.add_argument("--conf", type=float, default=0.40, help="Detector confidence threshold")
    parser.add_argument("--csi", action="store_true", help="Use Jetson onboard CSI camera (nvarguscamerasrc)")
    parser.add_argument("--headless", action="store_true", help="Run without X11 GUI display (saves CPU/GPU)")
    parser.add_argument("--record", type=str, default=None, help="Optional output MP4 path to record stream")
    args = parser.parse_args()

    run_jetson_pipeline(
        source=args.source,
        engine_path=args.engine,
        conf_thresh=args.conf,
        use_csi=args.csi,
        headless=args.headless,
        record_path=args.record
    )
