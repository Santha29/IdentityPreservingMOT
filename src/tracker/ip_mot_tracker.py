"""Identity-Preserving Multi-Object Tracker (IP-MOT).

A real-time, online causal tracking system engineered for dense crowd occlusions.
Features:
- Adaptive Kalman filtering with velocity damping
- Mutual occlusion detection and appearance update freezing
- 3-Stage Cascaded Data Association (High-Conf -> Low-Conf Occlusion -> Long-Lost Re-ID)
- Anti-ID Switch safeguards and bounded latency execution
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple

from .track import STrack, TrackState, Detection
from ..motion.kalman_filter import KalmanFilterAdaptive
from ..appearance.occlusion_detector import OcclusionDetector
from ..appearance.feature_extractor import AppearanceFeatureExtractor
from ..association.cascade_associator import CascadeAssociator


class IPMOTTracker:
    """Identity-Preserving Multi-Object Tracker for crowded gates and dense occlusion."""

    def __init__(
        self,
        det_high_thresh: float = 0.60,
        det_low_thresh: float = 0.15,
        match_thresh_stage1: float = 0.70,
        match_thresh_stage2: float = 0.50,
        reid_appearance_thresh: float = 0.35,
        appearance_weight: float = 0.50,
        n_init_confirmation: int = 3,
        max_lost_age: int = 60,
        iou_occlusion_thresh: float = 0.30,
        use_deep_reid: bool = False,
        onnx_model_path: Optional[str] = None
    ):
        """Initialize tracker with tuned hyper-parameters.

        Args:
            det_high_thresh: Minimum score for Stage 1 detections
            det_low_thresh: Minimum score for Stage 2 occluded detections
            match_thresh_stage1: Cost threshold for high-confidence matching
            match_thresh_stage2: Cost threshold for low-confidence matching
            reid_appearance_thresh: Maximum appearance distance to revive lost ID
            appearance_weight: Weight of appearance vs spatial cost
            n_init_confirmation: Consecutive hits needed to confirm a tentative track
            max_lost_age: Maximum frames a lost track is retained
            iou_occlusion_thresh: IoU overlap threshold declaring mutual occlusion
        """
        self.det_high_thresh = det_high_thresh
        self.det_low_thresh = det_low_thresh
        self.n_init_confirmation = n_init_confirmation
        self.max_lost_age = max_lost_age

        self.kalman_filter = KalmanFilterAdaptive()
        self.occlusion_detector = OcclusionDetector(iou_occlusion_threshold=iou_occlusion_thresh)
        self.feature_extractor = AppearanceFeatureExtractor(
            feature_dim=128,
            onnx_model_path=onnx_model_path,
            use_deep_reid=use_deep_reid
        )

        self.associator = CascadeAssociator(
            appearance_weight=appearance_weight,
            match_thresh_stage1=match_thresh_stage1,
            match_thresh_stage2=match_thresh_stage2,
            reid_appearance_thresh=reid_appearance_thresh,
            max_lost_age=max_lost_age
        )

        self.frame_id = 0
        self.tracked_tracks: List[STrack] = []    # Confirmed active tracks
        self.tentative_tracks: List[STrack] = []  # Tentative unconfirmed tracks
        self.lost_tracks: List[STrack] = []       # Occluded / disappeared tracks
        self.deleted_tracks: List[STrack] = []

        # Profiling stats
        self.last_latency_ms: float = 0.0

    def reset(self):
        """Reset internal tracker state."""
        STrack.reset_counter()
        self.frame_id = 0
        self.tracked_tracks.clear()
        self.tentative_tracks.clear()
        self.lost_tracks.clear()
        self.deleted_tracks.clear()
        self.last_latency_ms = 0.0

    def update(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
        image_patches: Optional[List[np.ndarray]] = None,
        frame_img: Optional[np.ndarray] = None
    ) -> List[STrack]:
        """Process a single video frame causal observation.

        Args:
            boxes_xyxy: (M, 4) array of detection boxes
            scores: (M,) array of detection scores
            features: Optional (M, D) pre-extracted appearance features
            image_patches: Optional list of M image patch crops
            frame_img: Optional full video frame (H, W, 3)

        Returns:
            active_confirmed_tracks: List of currently tracked STrack objects
        """
        start_time = time.perf_counter()
        self.frame_id += 1

        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        M = len(boxes_xyxy)

        # -------------------------------------------------------------
        # 1. Detection Pre-processing & Occlusion Estimation
        # -------------------------------------------------------------
        if M > 0:
            is_occ_det, vis_det, crowd_det = self.occlusion_detector.compute_mutual_occlusions(
                boxes_xyxy, scores=scores
            )
        else:
            is_occ_det = np.zeros(0, dtype=bool)
            vis_det = np.zeros(0, dtype=np.float32)
            crowd_det = np.zeros(0, dtype=np.float32)

        # Extract features if not pre-supplied
        if features is None:
            feat_list = []
            for i in range(M):
                if image_patches is not None and i < len(image_patches):
                    patch = image_patches[i]
                elif frame_img is not None:
                    b = boxes_xyxy[i].astype(int)
                    b[0] = max(0, b[0])
                    b[1] = max(0, b[1])
                    b[2] = min(frame_img.shape[1], b[2])
                    b[3] = min(frame_img.shape[0], b[3])
                    patch = frame_img[b[1]:b[3], b[0]:b[2]]
                else:
                    patch = None
                feat_list.append(self.feature_extractor.extract_patch_feature(patch))
            features = np.array(feat_list, dtype=np.float32) if len(feat_list) > 0 else np.empty((0, 128))

        # Partition detections into High-confidence and Low-confidence
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []

        for i in range(M):
            score = float(scores[i])
            det = Detection(
                box_xyxy=boxes_xyxy[i],
                score=score,
                feature=features[i] if i < len(features) else None,
                visibility=float(vis_det[i]),
                crop=image_patches[i] if image_patches and i < len(image_patches) else None
            )
            if score >= self.det_high_thresh:
                high_dets.append(det)
            elif score >= self.det_low_thresh:
                low_dets.append(det)

        # -------------------------------------------------------------
        # 2. Kalman State Prediction for all active & lost tracks
        # -------------------------------------------------------------
        for track in self.tracked_tracks:
            track.predict()
        for track in self.tentative_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        # -------------------------------------------------------------
        # 3. Cascaded Association
        # -------------------------------------------------------------
        # Candidates for primary matching: tracked + tentative
        all_active_candidates = self.tracked_tracks + self.tentative_tracks

        assoc_res = self.associator.associate(
            active_tracks=all_active_candidates,
            lost_tracks=self.lost_tracks,
            high_detections=high_dets,
            low_detections=low_dets
        )

        stage1_matches = assoc_res["stage1_matches"]
        stage2_matches = assoc_res["stage2_matches"]
        reid_matches = assoc_res["reid_matches"]
        unmatched_active = assoc_res["unmatched_active_tracks"]
        unmatched_lost = assoc_res["unmatched_lost_tracks"]
        unmatched_high_dets = assoc_res["unmatched_high_dets"]

        # -------------------------------------------------------------
        # 4. Process Matches & Updates
        # -------------------------------------------------------------
        # Process Stage 1 Matches (High-Confidence)
        for track, det in stage1_matches:
            # Check mutual occlusion against other matched detections
            track.update(
                detection=det,
                is_occluded=det.visibility < 0.70,
                crowd_density=track.crowd_density
            )
            if track in self.tentative_tracks:
                if track.hits >= self.n_init_confirmation:
                    track.mark_confirmed()
                    self.tentative_tracks.remove(track)
                    self.tracked_tracks.append(track)
            elif track not in self.tracked_tracks:
                self.tracked_tracks.append(track)

        # Process Stage 2 Matches (Occlusion Recovery via Low-Confidence)
        for track, det in stage2_matches:
            track.update(
                detection=det,
                is_occluded=True,  # Low-confidence match is treated as occluded!
                crowd_density=track.crowd_density
            )
            if track in self.tentative_tracks:
                if track.hits >= self.n_init_confirmation:
                    track.mark_confirmed()
                    self.tentative_tracks.remove(track)
                    self.tracked_tracks.append(track)
            elif track not in self.tracked_tracks:
                self.tracked_tracks.append(track)

        # Process Stage 3 Matches (Re-Identification of Lost Tracks)
        for lost_track, det in reid_matches:
            lost_track.re_activate(det, is_occluded=det.visibility < 0.70)
            if lost_track in self.lost_tracks:
                self.lost_tracks.remove(lost_track)
            self.tracked_tracks.append(lost_track)

        # -------------------------------------------------------------
        # 5. Track Lifecycle Transition
        # -------------------------------------------------------------
        # Handle unmatched active tracks -> transition to Lost/Occluded
        newly_lost_tracks = []
        for track in unmatched_active:
            if track in self.tracked_tracks:
                self.tracked_tracks.remove(track)
                track.mark_lost()
                newly_lost_tracks.append(track)
            elif track in self.tentative_tracks:
                # Tentative track unconfirmed and lost -> delete immediately
                self.tentative_tracks.remove(track)
                track.mark_deleted()
                self.deleted_tracks.append(track)

        # Handle existing unmatched lost tracks -> check age against max_lost_age
        surviving_lost = []
        for track in unmatched_lost:
            if track.time_since_update > self.max_lost_age:
                track.mark_deleted()
                self.deleted_tracks.append(track)
            else:
                surviving_lost.append(track)

        # Preserve both surviving and newly lost tracks in memory
        self.lost_tracks = surviving_lost + newly_lost_tracks

        # Initiate new tentative tracks from remaining unmatched high detections
        for det in unmatched_high_dets:
            new_track = STrack(
                detection=det,
                kalman_filter=self.kalman_filter
            )
            if self.n_init_confirmation <= 1:
                new_track.mark_confirmed()
                self.tracked_tracks.append(new_track)
            else:
                self.tentative_tracks.append(new_track)

        # Mutual occlusion check on final confirmed tracks
        if len(self.tracked_tracks) > 1:
            active_boxes = np.array([t.get_box_xyxy() for t in self.tracked_tracks], dtype=np.float32)
            is_occ_active, vis_active, density_active = self.occlusion_detector.compute_mutual_occlusions(active_boxes)
            for i, t in enumerate(self.tracked_tracks):
                t.is_occluded = is_occ_active[i]
                t.visibility = vis_active[i]
                t.crowd_density = density_active[i]

        self.last_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Return only confirmed tracks that were updated or recently observed
        return [t for t in self.tracked_tracks if t.state == TrackState.CONFIRMED]
