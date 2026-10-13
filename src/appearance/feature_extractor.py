"""Appearance Feature Extractor for Identity-Preserving Multi-Object Tracking.

Supports:
1. Spatial Multi-Stripe Color/Texture Descriptor (Fast, lightweight, invariant to scale)
2. ONNX Runtime Deep ReID Embedding Extractor (e.g., OSNet / MobileNetV2)
3. Synthetic / Simulated Identity Feature Generator for benchmark stress testing
"""

import numpy as np
import cv2
from typing import Optional, List, Union


class AppearanceFeatureExtractor:
    """Extracts discriminative appearance embeddings for person re-identification."""

    def __init__(
        self,
        feature_dim: int = 128,
        onnx_model_path: Optional[str] = None,
        use_deep_reid: bool = False
    ):
        """Initialize feature extractor.

        Args:
            feature_dim: Dimension of the appearance feature vector.
            onnx_model_path: Path to ONNX ReID model if available.
            use_deep_reid: Whether to use ONNX runtime if model is present.
        """
        self.feature_dim = feature_dim
        self.use_deep_reid = use_deep_reid
        self.onnx_model_path = onnx_model_path
        self._session = None

        if self.use_deep_reid and self.onnx_model_path is not None:
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(self.onnx_model_path)
                self._input_name = self._session.get_inputs()[0].name
            except Exception as e:
                print(f"[Warning] Failed to load ONNX model: {e}. Falling back to spatial descriptor.")
                self._session = None

    def extract_patch_feature(self, patch: np.ndarray) -> np.ndarray:
        """Extract L2-normalized feature vector from an RGB image crop of a detected person.

        Uses a 3-stripe spatial HSV color histogram descriptor:
        - Top stripe: Head / upper torso
        - Middle stripe: Torso / jacket
        - Bottom stripe: Legs / trousers

        Args:
            patch: (H, W, 3) BGR or RGB image patch

        Returns:
            feature: (feature_dim,) unit L2 normalized vector
        """
        if patch is None or patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            # Fallback zero/random vector
            v = np.ones(self.feature_dim, dtype=np.float32)
            return v / np.linalg.norm(v)

        if self._session is not None:
            return self._extract_onnx(patch)

        # Standard Spatial Multi-Stripe Color & Gradient Descriptor
        h, w, _ = patch.shape
        resized = cv2.resize(patch, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Split into 3 vertical stripes
        stripes = [
            hsv[0:42, :, :],    # Upper
            hsv[42:85, :, :],   # Middle
            hsv[85:128, :, :]   # Lower
        ]

        feat_parts = []
        for stripe in stripes:
            # Hue histogram: 16 bins, Saturation: 8 bins, Value: 8 bins
            hist_h = cv2.calcHist([stripe], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([stripe], [1], None, [8], [0, 256])
            hist_v = cv2.calcHist([stripe], [2], None, [8], [0, 256])
            part = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            norm = np.linalg.norm(part)
            if norm > 0:
                part /= norm
            feat_parts.append(part)

        # Also add full-body texture/edge energy
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist_edge = cv2.calcHist([ang.astype(np.float32)], [0], None, [32], [0, 360]).flatten()
        norm_edge = np.linalg.norm(hist_edge)
        if norm_edge > 0:
            hist_edge /= norm_edge
        feat_parts.append(hist_edge)

        full_vec = np.concatenate(feat_parts)  # length = 16+8+8 = 32 * 3 = 96 + 32 = 128
        if len(full_vec) != self.feature_dim:
            if len(full_vec) < self.feature_dim:
                full_vec = np.pad(full_vec, (0, self.feature_dim - len(full_vec)))
            else:
                full_vec = full_vec[:self.feature_dim]

        norm = np.linalg.norm(full_vec)
        if norm > 0:
            full_vec /= norm
        return full_vec.astype(np.float32)

    def _extract_onnx(self, patch: np.ndarray) -> np.ndarray:
        """Run deep ReID inference through ONNX runtime session."""
        resized = cv2.resize(patch, (128, 256))
        # ImageNet normalization: (x - mean) / std
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (rgb - mean) / std
        inp = np.transpose(inp, (2, 0, 1))  # [3, 256, 128]
        inp = np.expand_dims(inp, axis=0)  # [1, 3, 256, 128]

        outputs = self._session.run(None, {self._input_name: inp})
        feat = outputs[0].flatten().astype(np.float32)
        if len(feat) != self.feature_dim:
            if len(feat) < self.feature_dim:
                feat = np.pad(feat, (0, self.feature_dim - len(feat)))
            else:
                feat = feat[:self.feature_dim]
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat

    def generate_synthetic_feature(
        self,
        ground_truth_id: int,
        visibility: float = 1.0,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """Generate realistic synthetic feature vector for a given person ID.

        Used for rigorous stress-testing where ground-truth trajectories and simulated
        camera feeds are benchmarked under controlled occlusion.

        Args:
            ground_truth_id: Deterministic integer ID of the person
            visibility: Estimated visibility (1.0 = fully visible, 0.2 = heavily occluded)
            noise_level: Gaussian observation noise magnitude

        Returns:
            feature: (feature_dim,) unit L2 normalized vector
        """
        rng = np.random.RandomState(ground_truth_id * 10007 + 42)
        base_feature = rng.randn(self.feature_dim).astype(np.float32)
        base_feature /= np.linalg.norm(base_feature)

        # If partially occluded, add noise proportional to occlusion degree
        occ_noise_mag = (1.0 - np.clip(visibility, 0.0, 1.0)) * 0.45 + noise_level
        obs_noise = np.random.randn(self.feature_dim).astype(np.float32) * occ_noise_mag
        observed = base_feature + obs_noise
        norm = np.linalg.norm(observed)
        if norm > 0:
            observed /= norm
        return observed
