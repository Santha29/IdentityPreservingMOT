"""Cascaded Association Engine for Dense Occlusion and Re-Identification.

Executes a 3-Stage Cascaded Matching Pipeline:
- Stage 1: Confirmed Active Tracks vs High-Confidence Detections (Fused Appearance + IoU + Motion Gating)
- Stage 2: Remaining Active Tracks vs Low-Confidence Detections (Occlusion Recovery via IoU)
- Stage 3: Lost/Occluded Tracks vs Unmatched High-Confidence Detections (Appearance-Driven Re-Association)
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from .cost_matrix import compute_fused_cost_matrix, compute_iou_cost, INFTY_COST
from .matching import linear_assignment, verify_graph_topological_consistency
from ..motion.kalman_filter import CHI2_THRESHOLD_95, CHI2_THRESHOLD_99


class CascadeAssociator:
    """Manages multi-stage cascaded data association."""

    def __init__(
        self,
        appearance_weight: float = 0.50,
        match_thresh_stage1: float = 0.70,
        match_thresh_stage2: float = 0.50,
        reid_appearance_thresh: float = 0.35,
        max_lost_age: int = 60,
        motion_gate_thresh: float = CHI2_THRESHOLD_95,
        use_topological_consistency: bool = True
    ):
        """Initialize cascade associator hyperparameters.

        Args:
            appearance_weight: Lambda weighting appearance vs motion in Stage 1.
            match_thresh_stage1: Max cost threshold for high-confidence match.
            match_thresh_stage2: Max cost threshold for low-confidence occlusion recovery.
            reid_appearance_thresh: Max cosine distance allowed for reviving lost track identity.
            max_lost_age: Maximum frames a lost track is remembered before deletion.
            motion_gate_thresh: Mahalanobis chi-square gating threshold.
            use_topological_consistency: Whether to check spatial order consistency.
        """
        self.appearance_weight = appearance_weight
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2
        self.reid_appearance_thresh = reid_appearance_thresh
        self.max_lost_age = max_lost_age
        self.motion_gate_thresh = motion_gate_thresh
        self.use_topological_consistency = use_topological_consistency

    def associate(
        self,
        active_tracks: list,
        lost_tracks: list,
        high_detections: list,
        low_detections: list
    ) -> Dict[str, Any]:
        """Run the 3-stage association pipeline.

        Returns:
            Dict containing:
                'stage1_matches': List of (track, detection)
                'stage2_matches': List of (track, detection)
                'reid_matches': List of (lost_track, detection)
                'unmatched_active_tracks': List of track objects
                'unmatched_lost_tracks': List of track objects
                'unmatched_high_dets': List of detection objects
        """
        # =====================================================================
        # STAGE 1: Active Tracks vs High-Confidence Detections
        # =====================================================================
        if len(active_tracks) > 0 and len(high_detections) > 0:
            cost_matrix_1 = compute_fused_cost_matrix(
                tracks=active_tracks,
                detections=high_detections,
                appearance_weight=self.appearance_weight,
                motion_gate_threshold=self.motion_gate_thresh,
                appearance_gate_threshold=0.45,
                use_velocity_gating=True
            )
            raw_matches_1, u_tracks_idx_1, u_dets_idx_1 = linear_assignment(
                cost_matrix_1, thresh=self.match_thresh_stage1
            )
            if self.use_topological_consistency and len(raw_matches_1) > 0:
                raw_matches_1 = verify_graph_topological_consistency(
                    raw_matches_1, active_tracks, high_detections
                )
                # Re-compute unmatched
                matched_t_set = set(raw_matches_1[:, 0]) if len(raw_matches_1) > 0 else set()
                matched_d_set = set(raw_matches_1[:, 1]) if len(raw_matches_1) > 0 else set()
                u_tracks_idx_1 = np.array([i for i in range(len(active_tracks)) if i not in matched_t_set], dtype=int)
                u_dets_idx_1 = np.array([j for j in range(len(high_detections)) if j not in matched_d_set], dtype=int)

            stage1_matches = [(active_tracks[i], high_detections[j]) for i, j in raw_matches_1]
            rem_active_tracks = [active_tracks[i] for i in u_tracks_idx_1]
            rem_high_dets = [high_detections[j] for j in u_dets_idx_1]
        else:
            stage1_matches = []
            rem_active_tracks = list(active_tracks)
            rem_high_dets = list(high_detections)

        # =====================================================================
        # STAGE 2: Remaining Active Tracks vs Low-Confidence Detections (Occlusion Recovery)
        # =====================================================================
        if len(rem_active_tracks) > 0 and len(low_detections) > 0:
            track_boxes_2 = np.array([t.get_box_xyxy() for t in rem_active_tracks], dtype=np.float32)
            det_boxes_2 = np.array([d.box_xyxy for d in low_detections], dtype=np.float32)
            cost_matrix_2 = compute_iou_cost(track_boxes_2, det_boxes_2)

            raw_matches_2, u_tracks_idx_2, _ = linear_assignment(
                cost_matrix_2, thresh=self.match_thresh_stage2
            )
            stage2_matches = [(rem_active_tracks[i], low_detections[j]) for i, j in raw_matches_2]
            final_unmatched_active = [rem_active_tracks[i] for i in u_tracks_idx_2]
        else:
            stage2_matches = []
            final_unmatched_active = list(rem_active_tracks)

        # =====================================================================
        # STAGE 3: Lost / Occluded Tracks vs Unmatched High-Confidence Detections (Re-ID)
        # =====================================================================
        reid_matches = []
        rem_lost_tracks = list(lost_tracks)
        final_unmatched_high_dets = list(rem_high_dets)

        if len(lost_tracks) > 0 and len(rem_high_dets) > 0:
            # Build Appearance Re-ID Cost Matrix
            cost_matrix_3 = np.ones((len(lost_tracks), len(rem_high_dets)), dtype=np.float32)
            det_features_3 = np.array([d.feature for d in rem_high_dets if d.feature is not None], dtype=np.float32)

            if len(det_features_3) == len(rem_high_dets):
                for i, l_track in enumerate(lost_tracks):
                    if l_track.appearance_gallery.ema_feature is not None:
                        dists = l_track.appearance_gallery.compute_cosine_distance(det_features_3)
                        cost_matrix_3[i, :] = dists

                # Relaxed spatial search window gating based on elapsed frames
                for i, l_track in enumerate(lost_tracks):
                    center_t = l_track.get_center()
                    elapsed_frames = l_track.time_since_update
                    # Maximum plausible travel distance in pixels (e.g. max_speed 15 px/frame)
                    max_search_dist = 60.0 + 15.0 * elapsed_frames
                    for j, det in enumerate(rem_high_dets):
                        center_d = det.get_center()
                        dist_px = np.linalg.norm(center_t - center_d)
                        if dist_px > max_search_dist:
                            cost_matrix_3[i, j] = INFTY_COST

                raw_matches_3, u_lost_idx_3, u_dets_idx_3 = linear_assignment(
                    cost_matrix_3, thresh=self.reid_appearance_thresh
                )
                reid_matches = [(lost_tracks[i], rem_high_dets[j]) for i, j in raw_matches_3]
                rem_lost_tracks = [lost_tracks[i] for i in u_lost_idx_3]
                final_unmatched_high_dets = [rem_high_dets[j] for j in u_dets_idx_3]

        return {
            "stage1_matches": stage1_matches,
            "stage2_matches": stage2_matches,
            "reid_matches": reid_matches,
            "unmatched_active_tracks": final_unmatched_active,
            "unmatched_lost_tracks": rem_lost_tracks,
            "unmatched_high_dets": final_unmatched_high_dets,
        }
