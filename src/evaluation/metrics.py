"""Comprehensive MOT Metrics Suite: IDF1, MOTA, HOTA, IDSW, and Track Quality.

Implements standard benchmark evaluation metrics:
1. IDF1, IDP, IDR (Ristani et al., 2016) - Global identity preservation metrics
2. MOTA, MOTP, FP, FN, IDSW (Bernardin & Stiefelhagen, 2008) - CLEAR MOT metrics
3. HOTA, DetA, AssA (Luiten et al., 2020) - Higher Order Tracking Accuracy
4. MT, PT, ML, Frag - Track fragmentation & coverage statistics
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any
from ..motion.motion_geometry import compute_pairwise_iou


class MOTMetricsCalculator:
    """Calculates CLEAR MOT, ID metrics (IDF1), and HOTA from frame-by-frame tracker results."""

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truth: Dict[int, List[Dict[str, Any]]],
        tracker_results: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Compute complete metric suite.

        Args:
            ground_truth: Dict mapping frame_id -> list of dicts with 'id' and 'bbox' ([x1, y1, x2, y2])
            tracker_results: Dict mapping frame_id -> list of dicts with 'id' and 'bbox'

        Returns:
            metrics_dict: Dictionary containing IDF1, MOTA, HOTA, IDSW, MT, PT, ML, etc.
        """
        all_frames = sorted(set(list(ground_truth.keys()) + list(tracker_results.keys())))

        # -------------------------------------------------------------
        # CLEAR MOT Calculation (MOTA, MOTP, FP, FN, IDSW, Frag)
        # -------------------------------------------------------------
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_frag = 0
        iou_sum = 0.0
        total_tp = 0

        # Mapping for ID switches: last assigned tracker_id for each gt_id
        gt_to_trk_last: Dict[int, int] = {}
        # Track active gt presences for MT/PT/ML
        gt_frame_counts: Dict[int, int] = {}
        gt_tracked_counts: Dict[int, int] = {}

        # Global matching matrix for IDF1: (all_gt_ids x all_trk_ids)
        all_gt_ids = set()
        all_trk_ids = set()
        for f in all_frames:
            for g in ground_truth.get(f, []):
                all_gt_ids.add(g["id"])
                gt_frame_counts[g["id"]] = gt_frame_counts.get(g["id"], 0) + 1
            for t in tracker_results.get(f, []):
                all_trk_ids.add(t["id"])

        gt_id_list = sorted(list(all_gt_ids))
        trk_id_list = sorted(list(all_trk_ids))
        gt_id_to_idx = {g_id: i for i, g_id in enumerate(gt_id_list)}
        trk_id_to_idx = {t_id: j for j, t_id in enumerate(trk_id_list)}

        global_overlap = np.zeros((len(gt_id_list), len(trk_id_list)), dtype=int)

        for f in all_frames:
            gts = ground_truth.get(f, [])
            trks = tracker_results.get(f, [])

            total_gt += len(gts)

            if len(gts) == 0 and len(trks) == 0:
                continue

            if len(gts) == 0:
                total_fp += len(trks)
                continue

            if len(trks) == 0:
                total_fn += len(gts)
                continue

            gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32)
            trk_boxes = np.array([t["bbox"] for t in trks], dtype=np.float32)

            iou_mat = compute_pairwise_iou(gt_boxes, trk_boxes)

            matched_gt = set()
            matched_trk = set()
            current_frame_matches = []

            # 1. CLEAR MOT Rule: Preserve valid matches from frame t-1
            # If (g_id, t_id) was matched previously and still valid (IoU >= threshold), keep it!
            for r, g in enumerate(gts):
                g_id = g["id"]
                if g_id in gt_to_trk_last:
                    prev_trk_id = gt_to_trk_last[g_id]
                    # Find if prev_trk_id exists in current frame
                    for c, t in enumerate(trks):
                        if t["id"] == prev_trk_id and c not in matched_trk:
                            if iou_mat[r, c] >= self.iou_threshold:
                                matched_gt.add(r)
                                matched_trk.add(c)
                                current_frame_matches.append((r, c, False))  # False = not an ID switch
                                break

            # 2. Hungarian Assignment on remaining unmatched GTs and tracks
            rem_gts = [r for r in range(len(gts)) if r not in matched_gt]
            rem_trks = [c for c in range(len(trks)) if c not in matched_trk]

            if len(rem_gts) > 0 and len(rem_trks) > 0:
                sub_iou = iou_mat[np.ix_(rem_gts, rem_trks)]
                sub_cost = 1.0 - sub_iou
                sub_row, sub_col = linear_sum_assignment(sub_cost)

                for sr, sc in zip(sub_row, sub_col):
                    r = rem_gts[sr]
                    c = rem_trks[sc]
                    if iou_mat[r, c] >= self.iou_threshold:
                        matched_gt.add(r)
                        matched_trk.add(c)
                        g_id = gts[r]["id"]
                        t_id = trks[c]["id"]
                        # Check if this GT had a prior different assignment
                        is_idsw = (g_id in gt_to_trk_last and gt_to_trk_last[g_id] != t_id)
                        current_frame_matches.append((r, c, is_idsw))

            # 3. Accumulate stats
            for r, c, is_idsw in current_frame_matches:
                total_tp += 1
                iou_sum += iou_mat[r, c]

                g_id = gts[r]["id"]
                t_id = trks[c]["id"]

                g_idx = gt_id_to_idx[g_id]
                t_idx = trk_id_to_idx[t_id]
                global_overlap[g_idx, t_idx] += 1

                gt_tracked_counts[g_id] = gt_tracked_counts.get(g_id, 0) + 1

                if is_idsw:
                    total_idsw += 1
                gt_to_trk_last[g_id] = t_id

            frame_fn = len(gts) - len(matched_gt)
            frame_fp = len(trks) - len(matched_trk)
            total_fn += frame_fn
            total_fp += frame_fp

        # Compute MOTA and MOTP
        mota = 1.0 - (total_fn + total_fp + total_idsw) / max(1, total_gt)
        motp = (iou_sum / max(1, total_tp)) if total_tp > 0 else 0.0

        # -------------------------------------------------------------
        # IDF1 Calculation (Global Bipartite Match)
        # -------------------------------------------------------------
        if len(gt_id_list) > 0 and len(trk_id_list) > 0:
            # Maximizing global overlap is minimizing negative overlap
            row_ind, col_ind = linear_sum_assignment(-global_overlap)
            id_tp = sum(global_overlap[r, c] for r, c in zip(row_ind, col_ind))
        else:
            id_tp = 0

        total_detections = sum(len(tracker_results.get(f, [])) for f in all_frames)
        id_fp = total_detections - id_tp
        id_fn = total_gt - id_tp

        idp = id_tp / max(1, id_tp + id_fp)
        idr = id_tp / max(1, id_tp + id_fn)
        idf1 = (2.0 * id_tp) / max(1, 2 * id_tp + id_fp + id_fn)

        # -------------------------------------------------------------
        # Track Trajectory Quality (MT, PT, ML)
        # -------------------------------------------------------------
        mt = 0
        pt = 0
        ml = 0
        for g_id, total_f in gt_frame_counts.items():
            tracked_f = gt_tracked_counts.get(g_id, 0)
            ratio = tracked_f / max(1, total_f)
            if ratio >= 0.80:
                mt += 1
            elif ratio >= 0.20:
                pt += 1
            else:
                ml += 1

        total_trajectories = max(1, len(gt_frame_counts))
        mt_ratio = mt / total_trajectories
        ml_ratio = ml / total_trajectories

        # -------------------------------------------------------------
        # HOTA Approximation (DetA and AssA)
        # -------------------------------------------------------------
        # DetA = TP / (TP + FP + FN)
        deta = total_tp / max(1, total_tp + total_fp + total_fn)
        # AssA: based on global identity association overlap
        assa = id_tp / max(1, total_tp) if total_tp > 0 else 0.0
        hota = np.sqrt(max(0.0, deta * assa))

        return {
            "IDF1": float(idf1),
            "IDP": float(idp),
            "IDR": float(idr),
            "MOTA": float(mota),
            "MOTP": float(motp),
            "HOTA": float(hota),
            "DetA": float(deta),
            "AssA": float(assa),
            "IDSW": int(total_idsw),
            "FP": int(total_fp),
            "FN": int(total_fn),
            "TP": int(total_tp),
            "GT": int(total_gt),
            "MT": int(mt),
            "ML": int(ml),
            "MT_ratio": float(mt_ratio),
            "ML_ratio": float(ml_ratio),
        }
