"""Occlusion Detector and Crowding Estimator.

Analyzes spatial overlaps among bounding boxes to estimate:
1. Pairwise occlusion indices (Inter-Object Occlusion / IOI)
2. Visibility scores for each target
3. Mutual occlusion flags to freeze appearance updates
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ..motion.motion_geometry import compute_pairwise_iou


class OcclusionDetector:
    """Detects mutual occlusion among bounding boxes and tracks."""

    def __init__(
        self,
        iou_occlusion_threshold: float = 0.30,
        containment_threshold: float = 0.50,
        crowd_radius_pixels: float = 80.0
    ):
        """Initialize the occlusion detector.

        Args:
            iou_occlusion_threshold: IoU above which mutual occlusion is declared.
            containment_threshold: Fraction of smaller box covered by larger box.
            crowd_radius_pixels: Distance within which nearby targets contribute to crowd density.
        """
        self.iou_occlusion_threshold = iou_occlusion_threshold
        self.containment_threshold = containment_threshold
        self.crowd_radius_pixels = crowd_radius_pixels

    def compute_mutual_occlusions(
        self,
        boxes: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mutual occlusion metrics for a set of bounding boxes [x1, y1, x2, y2].

        Args:
            boxes: (N, 4) array of bounding boxes
            scores: Optional (N,) array of confidence scores

        Returns:
            is_occluded: (N,) boolean mask indicating whether box is occluded
            visibility_scores: (N,) float array in [0.0, 1.0] estimating visible fraction
            crowd_density: (N,) float array representing local spatial congestion
        """
        N = len(boxes)
        if N <= 1:
            return (
                np.zeros(N, dtype=bool),
                np.ones(N, dtype=np.float32),
                np.zeros(N, dtype=np.float32)
            )

        boxes = np.asarray(boxes, dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas = np.maximum(areas, 1e-4)

        # Pairwise IoU
        iou_matrix = compute_pairwise_iou(boxes, boxes)
        np.fill_diagonal(iou_matrix, 0.0)

        # Pairwise Intersection Areas
        lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
        rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
        wh = np.clip(rb - lt, a_min=0.0, a_max=None)
        inter_area = wh[:, :, 0] * wh[:, :, 1]
        np.fill_diagonal(inter_area, 0.0)

        # Containment ratio: inter_area / min(area_i, area_j)
        min_areas = np.minimum(areas[:, None], areas[None, :])
        containment = inter_area / min_areas

        # Max overlap experienced by each box
        max_iou = np.max(iou_matrix, axis=1)
        max_containment = np.max(containment, axis=1)

        is_occluded = (max_iou >= self.iou_occlusion_threshold) | (max_containment >= self.containment_threshold)

        # Visibility estimate: 1.0 - fraction covered
        # Summing overlapping intersection areas (capped at 1.0)
        fraction_covered = np.clip(np.sum(inter_area, axis=1) / areas, 0.0, 0.95)
        visibility_scores = 1.0 - fraction_covered

        # If detection scores are given, lower confidence further reduces visibility estimate
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32)
            visibility_scores = np.minimum(visibility_scores, scores)

        # Crowd density: count of neighbors within spatial radius
        centers = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) * 0.5,
            (boxes[:, 1] + boxes[:, 3]) * 0.5
        ])
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        neighbor_counts = np.sum(dists < self.crowd_radius_pixels, axis=1)
        crowd_density = neighbor_counts.astype(np.float32)

        return is_occluded, visibility_scores.astype(np.float32), crowd_density
