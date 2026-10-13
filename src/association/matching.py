"""Bipartite Matching and Hungarian Assignment with Graph Consistency Checks."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve linear sum assignment problem using Hungarian algorithm and filter out matches above threshold.

    Args:
        cost_matrix: (N, M) matrix of costs
        thresh: Maximum acceptable cost

    Returns:
        matches: (K, 2) array of matched indices [row_idx, col_idx]
        unmatched_a: (U_a,) array of unmatched row indices
        unmatched_b: (U_b,) array of unmatched col indices
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    matches_arr = np.array(matches, dtype=int) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    return matches_arr, np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)


def greedy_assignment(
    cost_matrix: np.ndarray,
    thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast greedy bipartite assignment for bounded-latency fallback under extreme density.
    Time complexity O(N * M * log(NM)) or O(min(N, M) * NM).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    N, M = cost_matrix.shape
    used_rows = set()
    used_cols = set()
    matches = []

    # Flatten and sort
    flat_indices = np.argsort(cost_matrix, axis=None)
    for idx in flat_indices:
        r = idx // M
        c = idx % M
        cost = cost_matrix[r, c]
        if cost > thresh:
            break
        if r not in used_rows and c not in used_cols:
            used_rows.add(r)
            used_cols.add(c)
            matches.append((r, c))
            if len(used_rows) == N or len(used_cols) == M:
                break

    unmatched_a = [r for r in range(N) if r not in used_rows]
    unmatched_b = [c for c in range(M) if c not in used_cols]
    matches_arr = np.array(matches, dtype=int) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    return matches_arr, np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)


def verify_graph_topological_consistency(
    matches: np.ndarray,
    tracks: list,
    detections: list
) -> np.ndarray:
    """Verify that matched pairs preserve topological order (relative horizontal ordering)
    for pairs that were previously walking side-by-side in close proximity.
    Prevents instantaneous identity swaps when people walk shoulder-to-shoulder.

    Args:
        matches: (K, 2) array of matched [track_idx, det_idx]
        tracks: List of track objects
        detections: List of detection objects

    Returns:
        filtered_matches: (K', 2) array of verified matches
    """
    if len(matches) < 2:
        return matches

    valid_matches = list(matches)
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            t_idx1, d_idx1 = matches[i]
            t_idx2, d_idx2 = matches[j]

            t1_box = tracks[t_idx1].get_box_xyxy()
            t2_box = tracks[t_idx2].get_box_xyxy()
            d1_box = detections[d_idx1].box_xyxy
            d2_box = detections[d_idx2].box_xyxy

            # Centers
            t1_cx = (t1_box[0] + t1_box[2]) * 0.5
            t2_cx = (t2_box[0] + t2_box[2]) * 0.5
            d1_cx = (d1_box[0] + d1_box[2]) * 0.5
            d2_cx = (d2_box[0] + d2_box[2]) * 0.5

            # Only enforce topology for targets traveling in the same direction (e.g. walking side-by-side)
            v1 = tracks[t_idx1].get_velocity()
            v2 = tracks[t_idx2].get_velocity()
            vel_dot = float(np.dot(v1, v2))

            if vel_dot > 0.5 and abs(t1_cx - t2_cx) < 50.0 and abs(d1_cx - d2_cx) < 50.0:
                track_order = (t1_cx < t2_cx)
                det_order = (d1_cx < d2_cx)

                if track_order != det_order:
                    # Swapped! Check appearance similarities
                    # If appearance of match 1 is worse than match 2, check if swap is invalid
                    t1_app_dist = 1.0
                    t2_app_dist = 1.0
                    if detections[d_idx1].feature is not None:
                        t1_app_dist = tracks[t_idx1].appearance_gallery.compute_cosine_distance(
                            detections[d_idx1].feature[None, :]
                        )[0]
                    if detections[d_idx2].feature is not None:
                        t2_app_dist = tracks[t_idx2].appearance_gallery.compute_cosine_distance(
                            detections[d_idx2].feature[None, :]
                        )[0]

                    # If either appearance match is weak (> 0.35), remove suspicious match to avoid swap
                    if t1_app_dist > 0.35 and (t_idx1, d_idx1) in valid_matches:
                        valid_matches.remove((t_idx1, d_idx1))
                    if t2_app_dist > 0.35 and (t_idx2, d_idx2) in valid_matches:
                        valid_matches.remove((t_idx2, d_idx2))

    return np.array(valid_matches, dtype=int) if len(valid_matches) > 0 else np.empty((0, 2), dtype=int)
