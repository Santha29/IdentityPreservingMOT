"""Track Representation and Lifecycle Management.

Defines:
- TrackState: Enum for track lifecycle (Tentative, Confirmed, Lost, Deleted)
- Detection: Container for single-frame bounding box, confidence, and feature
- STrack: Comprehensive track state containing Kalman filter, appearance gallery,
  and trajectory history.
"""

import numpy as np
from enum import Enum
from typing import Optional, List, Tuple
from ..motion.kalman_filter import KalmanFilterAdaptive
from ..motion.motion_geometry import xyah_to_xyxy, xyxy_to_xyah, xyxy_to_tlwh
from ..appearance.appearance_gallery import AppearanceGallery


class TrackState(Enum):
    """Lifecycle states of a tracked target."""
    TENTATIVE = 1
    CONFIRMED = 2
    OCCLUDED_LOST = 3
    DELETED = 4


class Detection:
    """Represents a bounding box observation from detector in a single frame."""

    def __init__(
        self,
        box_xyxy: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None,
        visibility: float = 1.0,
        crop: Optional[np.ndarray] = None
    ):
        """Initialize detection.

        Args:
            box_xyxy: [x1, y1, x2, y2]
            score: Confidence score in (0, 1]
            feature: L2-normalized appearance vector
            visibility: Estimated visible ratio in [0, 1]
            crop: Optional RGB image crop
        """
        self.box_xyxy = np.asarray(box_xyxy, dtype=np.float32)
        self.box_xyah = xyxy_to_xyah(self.box_xyxy)
        self.score = float(score)
        self.feature = feature
        self.visibility = float(visibility)
        self.crop = crop

    def get_center(self) -> np.ndarray:
        return np.array([
            (self.box_xyxy[0] + self.box_xyxy[2]) * 0.5,
            (self.box_xyxy[1] + self.box_xyxy[3]) * 0.5
        ], dtype=np.float32)


class STrack:
    """Single Object Track with Adaptive Kalman Filter and Appearance Memory."""

    _count = 0

    def __init__(
        self,
        detection: Detection,
        max_exemplars: int = 8,
        ema_alpha: float = 0.90,
        kalman_filter: Optional[KalmanFilterAdaptive] = None
    ):
        """Initialize a new track from an initial detection.

        Args:
            detection: Detection initiating this track
            max_exemplars: Size of exemplar buffer in gallery
            ema_alpha: Momentum factor for appearance smoothing
            kalman_filter: Shared or dedicated Kalman filter instance
        """
        STrack._count += 1
        self.track_id = STrack._count
        self.state = TrackState.TENTATIVE

        self.kalman_filter = kalman_filter if kalman_filter is not None else KalmanFilterAdaptive()
        self.mean, self.covariance = self.kalman_filter.initiate(detection.box_xyah)

        self.appearance_gallery = AppearanceGallery(
            base_ema_alpha=ema_alpha,
            max_exemplars=max_exemplars
        )
        if detection.feature is not None:
            self.appearance_gallery.update(
                detection.feature,
                visibility=detection.visibility,
                confidence_score=detection.score,
                is_occluded=False
            )

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.score = detection.score
        self.is_occluded = False
        self.crowd_density = 0.0
        self.visibility = detection.visibility

        # Store recent trajectory points: list of (xc, yc)
        cx = (detection.box_xyxy[0] + detection.box_xyxy[2]) * 0.5
        cy = (detection.box_xyxy[1] + detection.box_xyxy[3]) * 0.5
        self.trajectory_history: List[Tuple[float, float]] = [(cx, cy)]
        self.max_trajectory_length = 60

    @classmethod
    def reset_counter(cls, start_val: int = 0):
        cls._count = start_val

    def predict(self):
        """Predict forward one time step with occlusion-aware damping."""
        self.mean, self.covariance = self.kalman_filter.predict(
            self.mean,
            self.covariance,
            is_occluded=self.is_occluded or (self.state == TrackState.OCCLUDED_LOST)
        )
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        detection: Detection,
        is_occluded: bool = False,
        crowd_density: float = 0.0
    ):
        """Update track with associated detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            detection.box_xyah,
            confidence_score=detection.score
        )

        self.is_occluded = is_occluded
        self.crowd_density = crowd_density
        self.visibility = detection.visibility
        self.score = detection.score

        # Update appearance gallery (handles occlusion freezing internally)
        if detection.feature is not None:
            self.appearance_gallery.update(
                detection.feature,
                visibility=detection.visibility,
                confidence_score=detection.score,
                is_occluded=is_occluded
            )

        self.hits += 1
        self.time_since_update = 0

        # Record trajectory
        box = self.get_box_xyxy()
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        self.trajectory_history.append((float(cx), float(cy)))
        if len(self.trajectory_history) > self.max_trajectory_length:
            self.trajectory_history.pop(0)

    def re_activate(self, detection: Detection, is_occluded: bool = False):
        """Re-activate a lost track with a newly associated detection."""
        self.update(detection, is_occluded=is_occluded)
        self.state = TrackState.CONFIRMED

    def mark_confirmed(self):
        self.state = TrackState.CONFIRMED

    def mark_lost(self):
        self.state = TrackState.OCCLUDED_LOST
        self.is_occluded = True

    def mark_deleted(self):
        self.state = TrackState.DELETED

    def get_box_xyxy(self) -> np.ndarray:
        """Get current estimated bounding box in [x1, y1, x2, y2]."""
        return xyah_to_xyxy(self.mean[:4])

    def get_box_tlwh(self) -> np.ndarray:
        """Get current bounding box in [x, y, w, h]."""
        return xyxy_to_tlwh(self.get_box_xyxy())

    def get_velocity(self) -> np.ndarray:
        """Get estimated 2D velocity vector [vx, vy]."""
        return self.mean[4:6]

    def get_center(self) -> np.ndarray:
        """Get estimated 2D center [xc, yc]."""
        return self.mean[:2]

    def get_last_center(self) -> np.ndarray:
        """Get center from last update before prediction."""
        if len(self.trajectory_history) > 0:
            return np.array(self.trajectory_history[-1], dtype=np.float32)
        return self.get_center()
