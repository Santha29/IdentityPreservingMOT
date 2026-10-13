"""Appearance Gallery and Feature Memory Buffer.

Implements:
1. Quality-weighted Exponential Moving Average (EMA) feature updating
2. Occlusion-aware update freezing to prevent appearance drift into neighbors
3. FIFO Exemplar Cache for long-term re-identification across disappearances
4. Cosine distance computation with multi-exemplar minimum distance
"""

import numpy as np
from typing import List, Optional


class AppearanceGallery:
    """Maintains a discriminative appearance profile for an individual track."""

    def __init__(
        self,
        base_ema_alpha: float = 0.90,
        max_exemplars: int = 8,
        min_update_visibility: float = 0.60,
        min_update_score: float = 0.65
    ):
        """Initialize appearance gallery.

        Args:
            base_ema_alpha: Momentum weight for EMA update (higher = more stable).
            max_exemplars: Maximum number of clean exemplars to store in FIFO cache.
            min_update_visibility: Minimum visibility required to accept appearance update.
            min_update_score: Minimum detection confidence required to update appearance.
        """
        self.base_ema_alpha = base_ema_alpha
        self.max_exemplars = max_exemplars
        self.min_update_visibility = min_update_visibility
        self.min_update_score = min_update_score

        self.ema_feature: Optional[np.ndarray] = None
        self.exemplars: List[np.ndarray] = []
        self.update_count: int = 0
        self.frozen_updates_count: int = 0

    def update(
        self,
        new_feature: np.ndarray,
        visibility: float = 1.0,
        confidence_score: float = 1.0,
        is_occluded: bool = False
    ) -> bool:
        """Update appearance gallery if observation meets quality and non-occlusion criteria.

        Args:
            new_feature: (D,) unit L2 normalized feature vector
            visibility: Estimated visibility in [0, 1]
            confidence_score: Detection confidence in [0, 1]
            is_occluded: Whether mutual occlusion was flagged

        Returns:
            updated: True if appearance was updated, False if update was frozen
        """
        if new_feature is None:
            return False

        # Normalize feature defensively
        feat_norm = np.linalg.norm(new_feature)
        if feat_norm > 0:
            new_feature = new_feature / feat_norm
        else:
            return False

        # First observation always initializes the gallery
        if self.ema_feature is None:
            self.ema_feature = new_feature.copy()
            self.exemplars.append(new_feature.copy())
            self.update_count += 1
            return True

        # CRITICAL OCCLUSION SHIELD:
        # If occluded or low visibility or low confidence, FREEZE appearance update
        # to avoid feature contamination from the occluding body.
        if is_occluded or visibility < self.min_update_visibility or confidence_score < self.min_update_score:
            self.frozen_updates_count += 1
            return False

        # Dynamic momentum: Clean, highly-visible observations carry more weight
        # alpha in [base_ema_alpha, 0.98]
        dynamic_alpha = self.base_ema_alpha + (1.0 - self.base_ema_alpha) * (1.0 - visibility * confidence_score)
        dynamic_alpha = np.clip(dynamic_alpha, 0.70, 0.98)

        # EMA update
        updated_ema = dynamic_alpha * self.ema_feature + (1.0 - dynamic_alpha) * new_feature
        updated_ema /= np.linalg.norm(updated_ema)
        self.ema_feature = updated_ema

        # Cache clean exemplar if sufficiently distinct from existing exemplars
        if len(self.exemplars) < self.max_exemplars:
            self.exemplars.append(new_feature.copy())
        else:
            # Check maximum similarity to current exemplars
            sims = [np.dot(ex, new_feature) for ex in self.exemplars]
            if min(sims) < 0.95:  # Diverse appearance (e.g. angle change)
                # FIFO replacement
                self.exemplars.pop(0)
                self.exemplars.append(new_feature.copy())

        self.update_count += 1
        return True

    def compute_cosine_distance(self, candidate_features: np.ndarray) -> np.ndarray:
        """Compute cosine distances between this gallery and candidate feature vectors.
        Distance = 1.0 - Cosine_Similarity, in range [0, 2].

        Uses minimum distance across EMA feature and stored clean exemplars.

        Args:
            candidate_features: (M, D) array of unit candidate features

        Returns:
            distances: (M,) array of minimum cosine distances
        """
        candidate_features = np.asarray(candidate_features, dtype=np.float32)
        M = len(candidate_features)
        if M == 0 or self.ema_feature is None:
            return np.ones(M, dtype=np.float32)

        # Distance to EMA
        ema_sim = np.dot(candidate_features, self.ema_feature)  # (M,)

        # Distance to exemplars
        if len(self.exemplars) > 0:
            ex_mat = np.stack(self.exemplars, axis=0)  # (K, D)
            ex_sims = np.dot(candidate_features, ex_mat.T)  # (M, K)
            max_ex_sim = np.max(ex_sims, axis=1)  # (M,)
            best_sim = np.maximum(ema_sim, max_ex_sim)
        else:
            best_sim = ema_sim

        # Cosine distance = 1 - sim, bounded [0, 2]
        distances = np.clip(1.0 - best_sim, 0.0, 2.0)
        return distances.astype(np.float32)
