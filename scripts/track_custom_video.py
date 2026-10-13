"""
Universal Real-World Video Person Tracker (IP-MOT)
Works on ANY custom video (.mp4, .avi, .mov, .mkv) or live webcam.
Preserves person identities across dense occlusions.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from src.tracker.ip_mot_tracker import IPMOTTracker
from src.simulation.video_renderer import get_color_for_id


def process_custom_video(
    video_path: str,
    output_path: str = "assets/my_tracked_output.mp4",
    conf_thresh: float = 0.35,
    max_frames: int = None
):
    if not str(video_path).isdigit() and not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        print("Please check the file name and path.")
        return

    print("=" * 70)
    print("      UNIVERSAL IP-MOT REAL-WORLD VIDEO TRACKER")
    print(f"      Input Video:  {video_path}")
    print(f"      Output Video: {output_path}")
    print("=" * 70)

    source = int(video_path) if str(video_path).isdigit() else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source '{video_path}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Details: {width}x{height} Resolution | {fps:.1f} FPS | Total Frames: {total_frames}")

    scale_factor = max(0.5, min(width, height) / 720.0)
    font_scale = 0.55 * scale_factor
    box_thick = max(2, int(2 * scale_factor))

    print("\nLoading YOLOv8 AI Detector & IP-MOT Algorithm...")
    model = YOLO("yolov8n.pt")
    tracker = IPMOTTracker(
        det_high_thresh=conf_thresh,
        det_low_thresh=max(0.15, conf_thresh * 0.4),
        appearance_weight=0.55,
        reid_appearance_thresh=0.35,
        max_lost_age=60
    )
    tracker.reset()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    t_start = time.time()
    all_seen_ids = set()

    print("\nProcessing frames in real-time causal sequence...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        t_f_start = time.perf_counter()

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

        active_tracks = tracker.update(
            boxes_xyxy=b_arr,
            scores=s_arr,
            image_patches=patches,
            frame_img=frame
        )

        lat_ms = (time.perf_counter() - t_f_start) * 1000.0

        for trk in active_tracks:
            t_id = trk.track_id
            all_seen_ids.add(t_id)
            color = get_color_for_id(t_id)
            box = trk.get_box_xyxy().astype(int)
            is_occ = getattr(trk, "is_occluded", False)

            history = getattr(trk, "trajectory_history", [])
            if len(history) > 1:
                pts = history[-30:]
                for k in range(1, len(pts)):
                    p1 = (int(pts[k-1][0]), int(pts[k-1][1]))
                    p2 = (int(pts[k][0]), int(pts[k][1]))
                    cv2.line(frame, p1, p2, color, max(1, int(box_thick * 0.8)), cv2.LINE_AA)

            box_col = (0, 140, 255) if is_occ else color
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_col, box_thick)

            label = f"ID: {t_id}" + (" [OCC]" if is_occ else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(frame, (box[0], max(0, box[1] - th - 8)),
                          (box[0] + tw + 8, box[1]), box_col, -1)
            cv2.putText(frame, label, (box[0] + 4, box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(scale_factor)), cv2.LINE_AA)

        hud_h = int(65 * scale_factor)
        hud_w = int(480 * scale_factor)
        cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.putText(frame, f"IP-MOT | Frame: {frame_idx}/{total_frames} | {1000.0/max(0.1, lat_ms):.0f} FPS",
                    (20, int(35 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale_factor, (255, 255, 255), 1)
        cv2.putText(frame, f"Active: {len(active_tracks)} | Total Unique IDs: {len(all_seen_ids)}",
                    (20, int(60 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale_factor, (0, 255, 120), 2)

        out.write(frame)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"Frame {frame_idx}/{total_frames} | Active: {len(active_tracks)} | Unique IDs: {len(all_seen_ids)} | {lat_ms:.1f}ms")

    cap.release()
    out.release()
    total_time = time.time() - t_start

    print("\n" + "=" * 55)
    print("             TRACKING FINISHED!")
    print("=" * 55)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Unique Persons:   {len(all_seen_ids)}")
    print(f"Average Speed:          {frame_idx / max(0.1, total_time):.1f} FPS")
    print(f"Saved Output Video to:  {output_path}")
    print("=" * 55)

    os.system(f"start {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal IP-MOT Video Tracker")
    parser.add_argument("--video", type=str, required=True, help="Path to video file or 0 for webcam")
    parser.add_argument("--output", type=str, default="assets/my_tracked_output.mp4", help="Output path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    process_custom_video(args.video, args.output, args.conf, args.frames)
