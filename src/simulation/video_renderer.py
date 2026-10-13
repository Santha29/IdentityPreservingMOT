"""Video Renderer and Visualizer for Multi-Object Tracking.

Renders:
- Bounding boxes color-coded by Track ID
- Motion trajectory trails
- Mutual occlusion alerts and inter-object links
- Real-time performance Heads-Up Display (HUD): Frame, FPS, Latency, Active Tracks, ID Switches
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any


# Distinct palette for visualizing track IDs
COLOR_PALETTE = [
    (230, 25, 75),   # Red
    (60, 180, 75),   # Green
    (255, 225, 25),  # Yellow
    (0, 130, 200),   # Blue
    (245, 130, 48),  # Orange
    (145, 30, 180),  # Purple
    (70, 240, 240),  # Cyan
    (240, 50, 230),  # Magenta
    (210, 245, 60),  # Lime
    (250, 190, 212), # Pink
    (0, 128, 128),   # Teal
    (220, 190, 255), # Lavender
    (170, 110, 40),  # Brown
    (255, 250, 200), # Beige
    (128, 0, 0),     # Maroon
    (170, 255, 195), # Mint
    (128, 128, 0),   # Olive
    (255, 215, 180), # Coral
    (0, 0, 128),     # Navy
    (128, 128, 128), # Grey
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Return BGR color tuple for a given track ID."""
    idx = (track_id * 3 + 7) % len(COLOR_PALETTE)
    rgb = COLOR_PALETTE[idx]
    # Convert RGB to BGR for OpenCV
    return (rgb[2], rgb[1], rgb[0])


class VideoRenderer:
    """Renders video frames and tracking overlays."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        trail_length: int = 30
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.trail_length = trail_length

    def create_background(self) -> np.ndarray:
        """Render a realistic access gate / corridor background."""
        img = np.full((self.height, self.width, 3), 32, dtype=np.uint8)

        # Floor gradient
        for y in range(self.height):
            factor = 0.8 + 0.4 * (y / self.height)
            img[y, :, :] = (int(38 * factor), int(42 * factor), int(46 * factor))

        # Gate boundaries / corridor lines
        gate_y = int(self.height * 0.5)
        corridor_half = 140
        cv2.line(img, (0, gate_y - corridor_half), (self.width, gate_y - corridor_half), (70, 75, 80), 2)
        cv2.line(img, (0, gate_y + corridor_half), (self.width, gate_y + corridor_half), (70, 75, 80), 2)

        # Gate turnstiles / vertical pillars in the center
        gate_x = int(self.width * 0.5)
        cv2.line(img, (gate_x, 0), (gate_x, gate_y - 70), (100, 100, 110), 3)
        cv2.line(img, (gate_x, gate_y + 70), (gate_x, self.height), (100, 100, 110), 3)

        # Gate opening marker
        cv2.circle(img, (gate_x, gate_y - 70), 8, (0, 200, 255), -1)
        cv2.circle(img, (gate_x, gate_y + 70), 8, (0, 200, 255), -1)

        # Floor grid markings
        for x in range(0, self.width, 100):
            cv2.line(img, (x, gate_y - corridor_half), (x, gate_y + corridor_half), (48, 52, 56), 1)

        cv2.putText(img, "ACCESS GATE CORRIDOR (CHOKE POINT)", (gate_x - 170, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 145, 150), 1, cv2.LINE_AA)
        return img

    def render_frame(
        self,
        frame_idx: int,
        tracks: list,
        detections: Optional[dict] = None,
        ground_truth: Optional[list] = None,
        hud_info: Optional[dict] = None,
        base_img: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Render complete tracking visualization for a frame.

        Args:
            frame_idx: Frame index
            tracks: List of active track objects (STrack or SortTrack)
            detections: Optional dict with detection bounding boxes
            ground_truth: Optional list of GT targets
            hud_info: Dictionary containing telemetry (FPS, Latency, IDSW, etc.)
            base_img: Optional custom background image

        Returns:
            rendered: (H, W, 3) BGR image
        """
        if base_img is not None:
            canvas = base_img.copy()
        else:
            canvas = self.create_background()

        # 1. Render Ground Truth Silhouettes (subtle dashed/faded representation)
        if ground_truth:
            for gt in ground_truth:
                box = gt["bbox"].astype(int)
                color = gt.get("color", (100, 100, 100))
                # Faint person silhouette body
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (55, 60, 65), 1)

        # 2. Render Motion Trails
        for trk in tracks:
            t_id = trk.track_id
            color = get_color_for_id(t_id)
            history = getattr(trk, "trajectory_history", getattr(trk, "history", []))
            if len(history) > 1:
                pts = history[-self.trail_length:]
                for k in range(1, len(pts)):
                    pt1 = (int(pts[k-1][0]), int(pts[k-1][1]))
                    pt2 = (int(pts[k][0]), int(pts[k][1]))
                    # Increasing thickness towards recent
                    thick = max(1, int(1 + 2 * (k / len(pts))))
                    cv2.line(canvas, pt1, pt2, color, thick, cv2.LINE_AA)

        # 3. Render Mutual Occlusion Links
        occluded_tracks = [t for t in tracks if getattr(t, "is_occluded", False)]
        if len(occluded_tracks) >= 2:
            for i in range(len(occluded_tracks)):
                for j in range(i + 1, len(occluded_tracks)):
                    t1 = occluded_tracks[i]
                    t2 = occluded_tracks[j]
                    c1 = t1.get_center().astype(int)
                    c2 = t2.get_center().astype(int)
                    dist = np.linalg.norm(c1 - c2)
                    if dist < 85.0:
                        # Draw warning dash link
                        cv2.line(canvas, tuple(c1), tuple(c2), (0, 0, 255), 1, cv2.LINE_AA)

        # 4. Render Track Bounding Boxes and Badges
        for trk in tracks:
            t_id = trk.track_id
            box = trk.get_box_xyxy().astype(int)
            color = get_color_for_id(t_id)
            is_occ = getattr(trk, "is_occluded", False)
            hits = getattr(trk, "hits", 1)

            # Bounding box border
            line_thickness = 2 if not is_occ else 3
            box_color = color if not is_occ else (0, 140, 255)  # Orange alert if occluded
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), box_color, line_thickness)

            # Label badge background
            label = f"ID: {t_id}"
            if is_occ:
                label += " [OCC]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

            badge_y1 = max(0, box[1] - text_h - 6)
            badge_y2 = box[1]
            badge_x1 = box[0]
            badge_x2 = box[0] + text_w + 8

            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
            cv2.putText(canvas, label, (badge_x1 + 4, badge_y2 - 3), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

            # Velocity vector arrow
            vel = getattr(trk, "get_velocity", lambda: np.zeros(2))()
            if np.linalg.norm(vel) > 0.5:
                cx = int((box[0] + box[2]) * 0.5)
                cy = int((box[1] + box[3]) * 0.5)
                arrow_tip = (int(cx + vel[0] * 6.0), int(cy + vel[1] * 6.0))
                cv2.arrowedLine(canvas, (cx, cy), arrow_tip, color, 2, tipLength=0.35)

        # 5. Heads-Up Display (HUD) Telemetry Overlay
        self._render_hud(canvas, frame_idx, tracks, hud_info)
        return canvas

    def _render_hud(self, canvas: np.ndarray, frame_idx: int, tracks: list, hud_info: Optional[dict]):
        """Render sleek semi-transparent telemetry dashboard on upper area."""
        h, w, _ = canvas.shape

        # Semi-transparent HUD overlay banner
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (15, 12), (w - 15, 68), (18, 20, 24), -1)
        cv2.addWeighted(hud_bg, 0.82, canvas, 0.18, 0, canvas)
        cv2.rectangle(canvas, (15, 12), (w - 15, 68), (70, 75, 82), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        c_white = (240, 240, 240)
        c_cyan = (235, 205, 50)
        c_orange = (50, 160, 255)
        c_green = (80, 225, 100)

        # Section 1: Frame & Tracks
        cv2.putText(canvas, f"FRAME: {frame_idx:03d}", (30, 38), font, 0.55, c_white, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"ACTIVE TRACKS: {len(tracks)}", (30, 58), font, 0.55, c_cyan, 1, cv2.LINE_AA)

        # Section 2: Occlusions
        n_occ = sum(1 for t in tracks if getattr(t, "is_occluded", False))
        cv2.putText(canvas, f"MUTUAL OCCLUSIONS: {n_occ}", (260, 38), font, 0.55, c_orange, 1, cv2.LINE_AA)
        tracker_name = hud_info.get("tracker_name", "IP-MOT") if hud_info else "IP-MOT"
        cv2.putText(canvas, f"ALGORITHM: {tracker_name}", (260, 58), font, 0.55, c_green, 1, cv2.LINE_AA)

        # Section 3: Performance Telemetry
        latency = hud_info.get("latency_ms", 1.5) if hud_info else 1.5
        fps = hud_info.get("fps", 1000.0 / max(0.1, latency)) if hud_info else 60.0
        idsw = hud_info.get("id_switches", 0) if hud_info else 0
        idf1 = hud_info.get("idf1", None) if hud_info else None

        cv2.putText(canvas, f"LATENCY: {latency:.2f} ms ({fps:.0f} FPS)", (560, 38), font, 0.55, c_white, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"ID SWITCHES: {idsw}", (560, 58), font, 0.55, (80, 80, 255) if idsw > 2 else c_white, 1, cv2.LINE_AA)

        if idf1 is not None:
            cv2.putText(canvas, f"IDF1 SCORE: {idf1:.1%}", (850, 48), font, 0.65, (100, 240, 120), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "IDENTITY PRESERVATION: ACTIVE", (850, 48), font, 0.50, (120, 220, 140), 1, cv2.LINE_AA)
