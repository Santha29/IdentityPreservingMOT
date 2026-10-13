"""Motion modeling, Kalman filtering, and spatial geometry."""
from .kalman_filter import KalmanFilterAdaptive
from .motion_geometry import (
    xywh_to_xyxy,
    xyxy_to_xywh,
    xyxy_to_tlwh,
    tlwh_to_xyxy,
    compute_iou,
    compute_giou,
    compute_diou,
    compute_pairwise_iou,
    velocity_direction_gating
)

__all__ = [
    "KalmanFilterAdaptive",
    "xywh_to_xyxy",
    "xyxy_to_tlwh",
    "tlwh_to_xyxy",
    "xyxy_to_xywh",
    "compute_iou",
    "compute_giou",
    "compute_diou",
    "compute_pairwise_iou",
    "velocity_direction_gating"
]
