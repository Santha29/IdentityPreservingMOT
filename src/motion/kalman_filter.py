"""Adaptive Kalman Filter for Multi-Object Tracking with Occlusion Handling.

Mathematical Formulation:
State vector x in R^8: [xc, yc, a, h, v_xc, v_yc, v_a, v_h]^T
where:
  (xc, yc) is bounding box center,
  a is aspect ratio (w / h),
  h is bounding box height,
  (v_xc, v_yc, v_a, v_h) are corresponding velocities.

Measurement vector z in R^4: [xc, yc, a, h]^T

Key Occlusion-Aware Features:
1. Detection score-adaptive measurement covariance R(s)
2. Velocity damping during occlusion (prevents linear drift overshoots)
3. Covariance inflation during occluded propagation
4. Joseph-form covariance update for numerical symmetry and positive-definiteness
5. Chi-square Mahalanobis distance gating for outlier rejection
"""

import numpy as np
import scipy.linalg
from typing import Tuple, List, Optional


# 0.95 and 0.99 quantiles of Chi-square distribution for degree of freedom 4
CHI2_THRESHOLD_95 = 9.4877
CHI2_THRESHOLD_99 = 13.2767


class KalmanFilterAdaptive:
    """Occlusion-aware Kalman filter for 2D bounding box tracking."""

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20.0,
        std_weight_velocity: float = 1.0 / 160.0,
        velocity_damping_factor: float = 0.95,
        covariance_inflation_rate: float = 1.05,
    ):
        """Initialize filter matrices and noise weights.

        Args:
            std_weight_position: Positional process noise weight relative to height.
            std_weight_velocity: Velocity process noise weight relative to height.
            velocity_damping_factor: Damping factor gamma applied to velocities when unobserved.
            covariance_inflation_rate: Inflation multiplier for covariance during occlusions.
        """
        self._ndim = 4
        self._dt = 1.0

        # State transition matrix F (8x8)
        self._motion_mat = np.eye(2 * self._ndim, 2 * self._ndim)
        for i in range(self._ndim):
            self._motion_mat[i, self._ndim + i] = self._dt

        # Measurement projection matrix H (4x8)
        self._update_mat = np.eye(self._ndim, 2 * self._ndim)

        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self.velocity_damping_factor = velocity_damping_factor
        self.covariance_inflation_rate = covariance_inflation_rate

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create track from an unassociated measurement [xc, yc, a, h].

        Returns:
            mean: (8,) vector
            covariance: (8, 8) positive definite matrix
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        is_occluded: bool = False,
        damping_factor: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict state mean and covariance forward one time step.

        Args:
            mean: Current state mean (8,)
            covariance: Current state covariance (8, 8)
            is_occluded: If True, applies velocity damping and covariance inflation
            damping_factor: Optional override for velocity damping

        Returns:
            mean: Predicted state mean (8,)
            covariance: Predicted state covariance (8, 8)
        """
        h = max(mean[3], 1.0)
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # State projection
        mean_pred = np.dot(self._motion_mat, mean)
        cov_pred = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        if is_occluded:
            # Damp velocity to prevent overshoot while occluded
            gamma = damping_factor if damping_factor is not None else self.velocity_damping_factor
            mean_pred[4:8] *= gamma

            # Inflate spatial covariance to broaden search window
            cov_pred[:4, :4] *= self.covariance_inflation_rate

        # Ensure symmetry
        cov_pred = 0.5 * (cov_pred + cov_pred.T)
        return mean_pred, cov_pred

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        confidence_score: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space R^4.
        R is adaptively inflated when detection confidence is lower.

        Args:
            mean: State mean (8,)
            covariance: State covariance (8, 8)
            confidence_score: Detection confidence in [0, 1]

        Returns:
            proj_mean: (4,)
            proj_cov: (4, 4) innovation covariance S
        """
        h = max(mean[3], 1.0)
        conf = np.clip(confidence_score, 0.05, 1.0)

        # Scale measurement standard deviation inversely with confidence
        scale = 1.0 / (conf ** 1.5)
        std = [
            self._std_weight_position * h * scale,
            self._std_weight_position * h * scale,
            1e-1 * scale,
            self._std_weight_position * h * scale,
        ]
        meas_cov = np.diag(np.square(std))

        proj_mean = np.dot(self._update_mat, mean)
        proj_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + meas_cov
        proj_cov = 0.5 * (proj_cov + proj_cov.T)
        return proj_mean, proj_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence_score: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with observed measurement using Joseph-form covariance update.

        Args:
            mean: State mean (8,)
            covariance: State covariance (8, 8)
            measurement: Measurement [xc, yc, a, h] (4,)
            confidence_score: Detection confidence

        Returns:
            updated_mean: (8,)
            updated_covariance: (8, 8)
        """
        proj_mean, proj_cov = self.project(mean, covariance, confidence_score)

        # Kalman gain K = P * H^T * S^-1
        chol_factor, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - proj_mean
        updated_mean = mean + np.dot(kalman_gain, innovation)

        # Joseph form: P = (I - KH) * P * (I - KH)^T + K * R * K^T
        I_KH = np.eye(2 * self._ndim) - np.dot(kalman_gain, self._update_mat)
        
        # Measurement noise R
        h = max(mean[3], 1.0)
        scale = 1.0 / (np.clip(confidence_score, 0.05, 1.0) ** 1.5)
        std = [
            self._std_weight_position * h * scale,
            self._std_weight_position * h * scale,
            1e-1 * scale,
            self._std_weight_position * h * scale,
        ]
        R = np.diag(np.square(std))

        updated_cov = (
            np.linalg.multi_dot((I_KH, covariance, I_KH.T))
            + np.linalg.multi_dot((kalman_gain, R, kalman_gain.T))
        )
        updated_cov = 0.5 * (updated_cov + updated_cov.T)

        return updated_mean, updated_cov

    def mahalanobis_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute squared Mahalanobis distance between a predicted track and candidate detections.

        Args:
            mean: State mean (8,)
            covariance: State covariance (8, 8)
            measurements: (M, 4) matrix of candidate detections
            confidence_scores: Optional (M,) array of detection scores

        Returns:
            d2: (M,) array of squared Mahalanobis distances
        """
        measurements = np.asarray(measurements, dtype=np.float32)
        if len(measurements) == 0:
            return np.empty((0,), dtype=np.float32)

        proj_mean, proj_cov = self.project(mean, covariance, confidence_score=1.0)
        cholesky_factor = np.linalg.cholesky(proj_cov)
        diff = measurements - proj_mean
        z = scipy.linalg.solve_triangular(cholesky_factor, diff.T, lower=True, check_finite=False, overwrite_b=False)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    def mahalanobis_gate_mask(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        threshold: float = CHI2_THRESHOLD_95
    ) -> np.ndarray:
        """Return boolean mask of detections within Chi-Square gating threshold."""
        d2 = self.mahalanobis_distance(mean, covariance, measurements)
        return d2 <= threshold
