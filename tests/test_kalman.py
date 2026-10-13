"""Unit tests for Adaptive Kalman Filter."""

import unittest
import numpy as np
from src.motion.kalman_filter import KalmanFilterAdaptive, CHI2_THRESHOLD_95


class TestKalmanFilterAdaptive(unittest.TestCase):

    def setUp(self):
        self.kf = KalmanFilterAdaptive(
            std_weight_position=0.05,
            std_weight_velocity=0.00625,
            velocity_damping_factor=0.90,
            covariance_inflation_rate=1.10
        )

    def test_initiate_and_predict(self):
        measurement = np.array([100.0, 200.0, 0.5, 120.0], dtype=np.float32)
        mean, cov = self.kf.initiate(measurement)

        self.assertEqual(len(mean), 8)
        self.assertEqual(cov.shape, (8, 8))
        np.testing.assert_allclose(mean[:4], measurement, rtol=1e-5)

        # Predict step
        pred_mean, pred_cov = self.kf.predict(mean, cov, is_occluded=False)
        self.assertEqual(pred_mean.shape, (8,))
        self.assertEqual(pred_cov.shape, (8, 8))

    def test_occlusion_damping_and_inflation(self):
        measurement = np.array([100.0, 200.0, 0.5, 120.0], dtype=np.float32)
        mean, cov = self.kf.initiate(measurement)
        # Give non-zero velocity
        mean[4] = 10.0  # vx
        mean[5] = -5.0  # vy

        # Normal prediction vs Occluded prediction
        normal_mean, normal_cov = self.kf.predict(mean, cov, is_occluded=False)
        occ_mean, occ_cov = self.kf.predict(mean, cov, is_occluded=True)

        # In occluded prediction, velocity is damped
        self.assertLess(abs(occ_mean[4]), abs(normal_mean[4]))
        self.assertLess(abs(occ_mean[5]), abs(normal_mean[5]))

        # Spatial covariance should be strictly inflated
        self.assertGreater(occ_cov[0, 0], normal_cov[0, 0])
        self.assertGreater(occ_cov[1, 1], normal_cov[1, 1])

    def test_update_and_mahalanobis_gating(self):
        measurement1 = np.array([100.0, 200.0, 0.5, 120.0], dtype=np.float32)
        mean, cov = self.kf.initiate(measurement1)
        mean_pred, cov_pred = self.kf.predict(mean, cov)

        # Near observation (should pass Chi-Square gate)
        near_obs = np.array([101.0, 202.0, 0.5, 119.0], dtype=np.float32)
        d2_near = self.kf.mahalanobis_distance(mean_pred, cov_pred, near_obs[None, :])
        self.assertLess(d2_near[0], CHI2_THRESHOLD_95)

        # Far observation (should fail gate)
        far_obs = np.array([500.0, 800.0, 0.5, 120.0], dtype=np.float32)
        d2_far = self.kf.mahalanobis_distance(mean_pred, cov_pred, far_obs[None, :])
        self.assertGreater(d2_far[0], CHI2_THRESHOLD_95)


if __name__ == "__main__":
    unittest.main()
