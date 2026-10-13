"""Baseline SORT Tracker (Bewley et al., 2016).

Standard Simple Online and Realtime Tracking baseline:
- Standard Kalman filter
- Pure IoU cost matrix
- Single-stage Hungarian assignment
- No appearance embeddings
- No occlusion-aware covariance inflation or velocity damping
- No long-term re-identification
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from ..motion.kalman_filter import KalmanFilterAdaptive
from ..motion.motion_geometry import compute_pairwise_iou, xyxy_to_tlwh, xyah_to_xyxy, xyxy_to_xyah
from scipy.optimize import linear_sum_assignment


class SortTrack:
    """Track representation for Baseline SORT."""
    _count = 0

    def __init__(self, bbox_xyxy: np.ndarray, kalman_filter: KalmanFilterAdaptive):
        SortTrack._count += 1
        self.track_id = SortTrack._count
        self.kf = kalman_filter
        self.mean, self.covariance = self.kf.initiate(xyxy_to_xyah(bbox_xyxy))
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history = []

    def predict(self):
        # Standard prediction without occlusion handling
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance, is_occluded=False)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy: np.ndarray):
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, xyxy_to_xyah(bbox_xyxy))
        self.hits += 1
        self.time_since_update = 0

    def get_box_xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def get_box_tlwh(self) -> np.ndarray:
        return xyxy_to_tlwh(self.get_box_xyxy())


class BaselineSortTracker:
    """Standard SORT tracker for comparative ablation."""

    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 3,
        iou_threshold: float = 0.30
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.kf = KalmanFilterAdaptive()
        self.tracks: List[SortTrack] = []
        self.frame_id = 0
        self.last_latency_ms: float = 0.0

    def reset(self):
        SortTrack._count = 0
        self.tracks.clear()
        self.frame_id = 0
        self.last_latency_ms = 0.0

    def update(
        self,
        boxes_xyxy: np.ndarray,
        scores: Optional[np.ndarray] = None,
        *args,
        **kwargs
    ) -> List[SortTrack]:
        """Update tracker with detections."""
        start_time = time.perf_counter()
        self.frame_id += 1
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)

        # 1. Predict
        for trk in self.tracks:
            trk.predict()

        # 2. Compute IoU cost matrix
        if len(self.tracks) > 0 and len(boxes_xyxy) > 0:
            track_boxes = np.array([t.get_box_xyxy() for t in self.tracks], dtype=np.float32)
            iou_matrix = compute_pairwise_iou(track_boxes, boxes_xyxy)
            cost_matrix = 1.0 - iou_matrix

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= (1.0 - self.iou_threshold):
                    self.tracks[r].update(boxes_xyxy[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

            unmatched_dets = [j for j in range(len(boxes_xyxy)) if j not in matched_dets]
        else:
            unmatched_dets = list(range(len(boxes_xyxy)))

        # 3. Create new tracks
        for j in unmatched_dets:
            new_trk = SortTrack(boxes_xyxy[j], self.kf)
            self.tracks.append(new_trk)

        # 4. Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        self.last_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Return confirmed tracks
        return [t for t in self.tracks if (t.hits >= self.min_hits or self.frame_id <= self.min_hits) and t.time_since_update == 0]
