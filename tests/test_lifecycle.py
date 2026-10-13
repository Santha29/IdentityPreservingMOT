"""Unit tests for Track Lifecycle and Occlusion Appearance Freezing."""

import unittest
import numpy as np
from src.tracker.track import STrack, TrackState, Detection
from src.appearance.appearance_gallery import AppearanceGallery


class TestTrackLifecycle(unittest.TestCase):

    def setUp(self):
        STrack.reset_counter()

    def test_appearance_gallery_occlusion_freezing(self):
        gallery = AppearanceGallery(base_ema_alpha=0.90, min_update_visibility=0.60)
        feat1 = np.ones(128, dtype=np.float32)
        feat1 /= np.linalg.norm(feat1)

        # Initial clean observation initializes gallery
        res = gallery.update(feat1, visibility=1.0, confidence_score=0.95, is_occluded=False)
        self.assertTrue(res)
        self.assertEqual(gallery.update_count, 1)

        # Occluded observation with noisy/different feature
        feat_occ = np.zeros(128, dtype=np.float32)
        feat_occ[0] = 1.0

        # Should FREEZE update when is_occluded=True
        res_frozen = gallery.update(feat_occ, visibility=0.40, confidence_score=0.50, is_occluded=True)
        self.assertFalse(res_frozen)
        self.assertEqual(gallery.frozen_updates_count, 1)

        # Verify EMA did NOT drift towards the occluding feature
        cos_sim = np.dot(gallery.ema_feature, feat1)
        self.assertAlmostEqual(cos_sim, 1.0, places=4)

    def test_strack_lifecycle_transitions(self):
        box = np.array([50.0, 50.0, 100.0, 150.0], dtype=np.float32)
        det = Detection(box_xyxy=box, score=0.90)

        track = STrack(detection=det)
        self.assertEqual(track.state, TrackState.TENTATIVE)

        # Confirm
        track.mark_confirmed()
        self.assertEqual(track.state, TrackState.CONFIRMED)

        # Mark lost under occlusion
        track.mark_lost()
        self.assertEqual(track.state, TrackState.OCCLUDED_LOST)
        self.assertTrue(track.is_occluded)

        # Re-activate upon reappearance
        det2 = Detection(box_xyxy=box + 5.0, score=0.85)
        track.re_activate(det2)
        self.assertEqual(track.state, TrackState.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
