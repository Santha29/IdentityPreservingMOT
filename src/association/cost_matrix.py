"""Association Cost Matrix Computation with Appearance-Motion Fusion and Gating.

Implements:
1. Cosine Appearance Distance
2. IoU / GIoU / DIoU Geometric Distance
3. Mahalanobis Motion Distance with Chi-square gating
4. Directional Velocity Consistency Gating
5. Adaptive weighting between appearance and motion under crowd density
"""

import numpy as np
from typing import List, Tuple, Optional
from ..motion.motion_geometry import compute_pairwise_iou, compute_diou, velocity_direction_gating
from ..motion.kalman_filter import CHI2_THRESHOLD_95, CHI2_THRESHOLD_99


INFTY_COST = 1e5


def compute_iou_cost(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    use_diou: bool = False
) -> np.ndarray:
    """Compute (N, M) cost matrix based on bounding box IoU / DIoU.
    Cost = 1.0 - IoU, bounded in [0.0, 1.0].

    Args:
        track_boxes: (N, 4) in [x1, y1, x2, y2]
        det_boxes: (M, 4) in [x1, y1, x2, y2]
        use_diou: If True, uses DIoU distance
    """
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)

    if use_diou:
        diou = compute_diou(track_boxes, det_boxes)
        cost = 1.0 - diou  # Range [0, 2]
        cost = np.clip(cost, 0.0, 2.0)
    else:
        iou = compute_pairwise_iou(track_boxes, det_boxes)
        cost = 1.0 - iou
        cost = np.clip(cost, 0.0, 1.0)

    return cost.astype(np.float32)


def compute_fused_cost_matrix(
    tracks: list,
    detections: list,
    appearance_weight: float = 0.50,
    motion_gate_threshold: float = CHI2_THRESHOLD_95,
    appearance_gate_threshold: float = 0.45,
    use_velocity_gating: bool = True
) -> np.ndarray:
    """Compute fused appearance and spatial/motion cost matrix with hard gating.

    C_ij = (1 - w_app) * d_spatial(i, j) + w_app * d_app(i, j)
    Gated pairs are set to INFTY_COST.

    Args:
        tracks: List of active track objects
        detections: List of detection objects
        appearance_weight: Weight lambda_app in [0, 1] for appearance term
        motion_gate_threshold: Mahalanobis Chi-square threshold
        appearance_gate_threshold: Maximum cosine distance allowed
        use_velocity_gating: Whether to filter out opposing velocity matches

    Returns:
        cost_matrix: (N, M) matrix of association costs
    """
    N = len(tracks)
    M = len(detections)

    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    track_boxes = np.array([t.get_box_xyxy() for t in tracks], dtype=np.float32)
    det_boxes = np.array([d.box_xyxy for d in detections], dtype=np.float32)

    # 1. Spatial IoU Cost Matrix: (N, M)
    iou_cost = compute_iou_cost(track_boxes, det_boxes)

    # 2. Appearance Cosine Distance Matrix: (N, M)
    app_cost = np.ones((N, M), dtype=np.float32)
    det_features = np.array([d.feature for d in detections if d.feature is not None], dtype=np.float32)

    has_features = len(det_features) == M
    if has_features:
        for i, track in enumerate(tracks):
            if track.appearance_gallery.ema_feature is not None:
                app_cost[i, :] = track.appearance_gallery.compute_cosine_distance(det_features)

    # 3. Dynamic appearance weight adaptation:
    # If tracks are crowded, raise appearance weight to disambiguate tight groupings
    avg_crowd = np.mean([t.crowd_density for t in tracks])
    if avg_crowd > 1.0:
        effective_app_weight = min(0.85, appearance_weight + 0.15 * min(avg_crowd, 4.0) / 4.0)
    else:
        effective_app_weight = appearance_weight

    # Fused cost
    fused_cost = (1.0 - effective_app_weight) * iou_cost + effective_app_weight * app_cost

    # 4. Hard Gating:
    # Gating A: Mahalanobis distance from Kalman filter
    det_xyah = np.array([d.box_xyah for d in detections], dtype=np.float32)
    for i, track in enumerate(tracks):
        maha_dists = track.kalman_filter.mahalanobis_distance(
            track.mean, track.covariance, det_xyah
        )
        gate_mask = maha_dists > motion_gate_threshold
        fused_cost[i, gate_mask] = INFTY_COST

    # Gating B: Appearance gating threshold
    if has_features:
        # Strict appearance gating applies when spatial overlap is weak;
        # if boxes strongly overlap (IoU >= 0.4, iou_cost <= 0.6), allow match unless appearance is completely incompatible (> 0.70)
        app_gate_mask = (app_cost > appearance_gate_threshold) & (iou_cost > 0.60) | (app_cost > 0.70)
        fused_cost[app_gate_mask] = INFTY_COST

    # Gating C: Velocity directional gating
    if use_velocity_gating:
        track_vels = np.array([t.get_velocity() for t in tracks], dtype=np.float32)
        track_pos = np.array([t.get_last_center() for t in tracks], dtype=np.float32)
        det_pos = np.array([d.get_center() for d in detections], dtype=np.float32)

        dir_mask = velocity_direction_gating(track_vels, track_pos, det_pos, min_speed=1.5, cos_threshold=-0.3)
        fused_cost[~dir_mask] = INFTY_COST

    return fused_cost.astype(np.float32)
