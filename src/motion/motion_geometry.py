"""Bounding-box geometry, IoU variants, and motion gating functions.

Provides conversions across representations:
- xyah / xywh: [center_x, center_y, aspect_ratio (w/h), height]
- xyxy: [x1, y1, x2, y2]
- tlwh: [top_left_x, top_left_y, width, height]
"""

import numpy as np
from typing import Tuple, Union


def xyxy_to_tlwh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    bbox = np.asarray(bbox, dtype=np.float32)
    tlwh = np.empty_like(bbox)
    tlwh[..., 0] = bbox[..., 0]
    tlwh[..., 1] = bbox[..., 1]
    tlwh[..., 2] = np.maximum(0.0, bbox[..., 2] - bbox[..., 0])
    tlwh[..., 3] = np.maximum(0.0, bbox[..., 3] - bbox[..., 1])
    return tlwh


def tlwh_to_xyxy(tlwh: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    tlwh = np.asarray(tlwh, dtype=np.float32)
    xyxy = np.empty_like(tlwh)
    xyxy[..., 0] = tlwh[..., 0]
    xyxy[..., 1] = tlwh[..., 1]
    xyxy[..., 2] = tlwh[..., 0] + tlwh[..., 2]
    xyxy[..., 3] = tlwh[..., 1] + tlwh[..., 3]
    return xyxy


def xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x_center, y_center, aspect_ratio (w/h), height]."""
    bbox = np.asarray(bbox, dtype=np.float32)
    w = np.maximum(1e-4, bbox[..., 2] - bbox[..., 0])
    h = np.maximum(1e-4, bbox[..., 3] - bbox[..., 1])
    xc = bbox[..., 0] + 0.5 * w
    yc = bbox[..., 1] + 0.5 * h
    a = w / h
    xyah = np.empty_like(bbox)
    xyah[..., 0] = xc
    xyah[..., 1] = yc
    xyah[..., 2] = a
    xyah[..., 3] = h
    return xyah


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Convert [x_center, y_center, aspect_ratio (w/h), height] to [x1, y1, x2, y2]."""
    xyah = np.asarray(xyah, dtype=np.float32)
    xc = xyah[..., 0]
    yc = xyah[..., 1]
    a = xyah[..., 2]
    h = xyah[..., 3]
    w = np.maximum(1e-4, a * h)

    xyxy = np.empty_like(xyah)
    xyxy[..., 0] = xc - 0.5 * w
    xyxy[..., 1] = yc - 0.5 * h
    xyxy[..., 2] = xc + 0.5 * w
    xyxy[..., 3] = yc + 0.5 * h
    return xyxy


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Alias for xyah_to_xyxy or standard [xc, yc, w, h] to xyxy."""
    xywh = np.asarray(xywh, dtype=np.float32)
    xc = xywh[..., 0]
    yc = xywh[..., 1]
    w = xywh[..., 2]
    h = xywh[..., 3]
    xyxy = np.empty_like(xywh)
    xyxy[..., 0] = xc - 0.5 * w
    xyxy[..., 1] = yc - 0.5 * h
    xyxy[..., 2] = xc + 0.5 * w
    xyxy[..., 3] = yc + 0.5 * h
    return xyxy


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x_center, y_center, width, height]."""
    xyxy = np.asarray(xyxy, dtype=np.float32)
    w = np.maximum(1e-4, xyxy[..., 2] - xyxy[..., 0])
    h = np.maximum(1e-4, xyxy[..., 3] - xyxy[..., 1])
    xc = xyxy[..., 0] + 0.5 * w
    yc = xyxy[..., 1] + 0.5 * h
    xywh = np.empty_like(xyxy)
    xywh[..., 0] = xc
    xywh[..., 1] = yc
    xywh[..., 2] = w
    xywh[..., 3] = h
    return xywh


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_pairwise_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix of shape (N, M) for boxes in format [x1, y1, x2, y2]."""
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersections
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0.0, a_max=None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-6)

    return inter / union


def compute_giou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise Generalized IoU (GIoU) matrix of shape (N, M).
    GIoU = IoU - (C - Union) / C, where C is the area of the smallest enclosing box.
    Range: [-1.0, 1.0].
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    iou = compute_pairwise_iou(boxes1, boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt_inter = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_inter = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_inter = np.clip(rb_inter - lt_inter, a_min=0.0, a_max=None)
    inter = wh_inter[:, :, 0] * wh_inter[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter

    # Convex hull / enclosing box
    lt_c = np.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = np.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = np.clip(rb_c - lt_c, a_min=0.0, a_max=None)
    area_c = np.maximum(wh_c[:, :, 0] * wh_c[:, :, 1], 1e-6)

    giou = iou - (area_c - union) / area_c
    return giou


def compute_diou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise Distance-IoU (DIoU) matrix of shape (N, M).
    DIoU = IoU - rho^2(b1, b2) / c^2, where rho is Euclidean distance between centers,
    and c is diagonal length of the smallest enclosing box.
    Range: [-1.0, 1.0].
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    iou = compute_pairwise_iou(boxes1, boxes2)

    # Centers
    c1_x = (boxes1[:, 0] + boxes1[:, 2]) * 0.5
    c1_y = (boxes1[:, 1] + boxes1[:, 3]) * 0.5
    c2_x = (boxes2[:, 0] + boxes2[:, 2]) * 0.5
    c2_y = (boxes2[:, 1] + boxes2[:, 3]) * 0.5

    d_sq = (c1_x[:, None] - c2_x[None, :]) ** 2 + (c1_y[:, None] - c2_y[None, :]) ** 2

    # Enclosing box diagonal squared
    lt_c = np.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = np.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    diag_sq = (rb_c[:, :, 0] - lt_c[:, :, 0]) ** 2 + (rb_c[:, :, 1] - lt_c[:, :, 1]) ** 2
    diag_sq = np.maximum(diag_sq, 1e-6)

    diou = iou - (d_sq / diag_sq)
    return diou


def velocity_direction_gating(
    track_vels: np.ndarray,
    track_positions: np.ndarray,
    det_positions: np.ndarray,
    min_speed: float = 1.5,
    cos_threshold: float = -0.3
) -> np.ndarray:
    """Gate out matches where detection displacement vector sharply contradicts track velocity.
    Prevents ID switches when two targets cross each other heading in opposite directions.
    
    Args:
        track_vels: (N, 2) array of [vx, vy]
        track_positions: (N, 2) array of [xc, yc]
        det_positions: (M, 2) array of [det_xc, det_yc]
        min_speed: Minimum speed threshold to enforce directional gating
        cos_threshold: Cosine similarity threshold below which match is rejected
        
    Returns:
        mask: (N, M) boolean array where True indicates match is valid, False indicates direction mismatch
    """
    N = len(track_vels)
    M = len(det_positions)
    if N == 0 or M == 0:
        return np.ones((N, M), dtype=bool)

    speeds = np.linalg.norm(track_vels, axis=1)  # (N,)
    # Displacement vectors from track position to detection: (N, M, 2)
    disp = det_positions[None, :, :] - track_positions[:, None, :]  # (N, M, 2)
    disp_norms = np.linalg.norm(disp, axis=2)  # (N, M)

    valid_mask = np.ones((N, M), dtype=bool)

    # For tracks with significant speed and non-trivial displacements
    for i in range(N):
        if speeds[i] >= min_speed:
            v = track_vels[i] / (speeds[i] + 1e-6)  # unit velocity
            for j in range(M):
                if disp_norms[i, j] >= 2.0:
                    d = disp[i, j] / (disp_norms[i, j] + 1e-6)
                    cos_sim = np.dot(v, d)
                    if cos_sim < cos_threshold:
                        valid_mask[i, j] = False

    return valid_mask
