"""Unit tests for Hungarian assignment and Association Cost Matrix."""

import unittest
import numpy as np
from src.association.matching import linear_assignment, greedy_assignment
from src.motion.motion_geometry import compute_pairwise_iou, velocity_direction_gating


class TestAssociation(unittest.TestCase):

    def test_linear_assignment_optimal(self):
        # 3x3 cost matrix with clear optimal diagonal
        cost_matrix = np.array([
            [0.1, 0.9, 0.8],
            [0.7, 0.2, 0.9],
            [0.8, 0.6, 0.15]
        ], dtype=np.float32)

        matches, u_a, u_b = linear_assignment(cost_matrix, thresh=0.5)
        self.assertEqual(len(matches), 3)
        self.assertEqual(len(u_a), 0)
        self.assertEqual(len(u_b), 0)

        # Matched pairs should be (0,0), (1,1), (2,2)
        for r, c in matches:
            self.assertEqual(r, c)

    def test_linear_assignment_gating_threshold(self):
        # Matrix where some elements exceed threshold
        cost_matrix = np.array([
            [0.1, 0.9],
            [0.9, 0.95]
        ], dtype=np.float32)

        matches, u_a, u_b = linear_assignment(cost_matrix, thresh=0.5)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 0)
        self.assertEqual(matches[0][1], 0)
        self.assertIn(1, u_a)
        self.assertIn(1, u_b)

    def test_greedy_assignment_fallback(self):
        cost_matrix = np.array([
            [0.05, 0.8],
            [0.9, 0.1]
        ], dtype=np.float32)

        matches, u_a, u_b = greedy_assignment(cost_matrix, thresh=0.5)
        self.assertEqual(len(matches), 2)

    def test_velocity_direction_gating(self):
        # Track 0 moving right (+vx)
        track_vels = np.array([[5.0, 0.0]], dtype=np.float32)
        track_pos = np.array([[100.0, 100.0]], dtype=np.float32)

        # Det 0 is to the right (consistent)
        # Det 1 is to the left (opposing)
        det_pos = np.array([[120.0, 100.0], [80.0, 100.0]], dtype=np.float32)

        mask = velocity_direction_gating(track_vels, track_pos, det_pos, min_speed=1.5, cos_threshold=-0.2)
        self.assertTrue(mask[0, 0])   # Moving right to right -> valid
        self.assertFalse(mask[0, 1])  # Moving right but det is left -> gated out!


if __name__ == "__main__":
    unittest.main()
