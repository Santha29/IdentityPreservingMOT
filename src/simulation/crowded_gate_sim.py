"""Crowded Gate Multi-Person Simulator with Realistic Mutual Occlusions.

Simulates complex pedestrian dynamics in a narrow choke point / access gate:
1. Bi-directional pedestrian traffic (crossing paths, tight groups, opposing flows)
2. Realistic mutual occlusions (front-back occlusion, walking in tandem, side-by-side)
3. Partial/Complete detection drops and confidence degradation under occlusion
4. Distinctive ground-truth appearance signatures per person
5. Outputs standard MOTChallenge ground-truth and detection formats
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from ..appearance.feature_extractor import AppearanceFeatureExtractor


class PedestrianAgent:
    """Simulated pedestrian walking through a gate corridor."""

    def __init__(
        self,
        gt_id: int,
        start_x: float,
        start_y: float,
        vx: float,
        vy: float,
        width: float = 48.0,
        height: float = 120.0,
        start_frame: int = 1,
        total_frames: int = 200,
        feature_dim: int = 128
    ):
        self.gt_id = gt_id
        self.x = start_x
        self.y = start_y
        self.vx = vx
        self.vy = vy
        self.width = width
        self.height = height
        self.start_frame = start_frame
        self.end_frame = start_frame + total_frames
        self.is_active = False

        # Inherent clothing/color identity feature
        rng = np.random.RandomState(gt_id * 10007 + 13)
        feat = rng.randn(feature_dim).astype(np.float32)
        self.base_feature = feat / np.linalg.norm(feat)

        # Color representation (BGR)
        self.color = (
            int(rng.randint(60, 255)),
            int(rng.randint(60, 255)),
            int(rng.randint(60, 255))
        )

    def step(self, current_frame: int, gate_bounds: Tuple[float, float, float, float]):
        """Advance position with kinematic model and subtle human motion jitter."""
        if current_frame < self.start_frame or current_frame > self.end_frame:
            self.is_active = False
            return

        self.is_active = True
        # Slight pacing jitter
        jitter_x = np.random.normal(0.0, 0.3)
        jitter_y = np.random.normal(0.0, 0.2)

        self.x += self.vx + jitter_x
        self.y += self.vy + jitter_y

    def get_bbox_xyxy(self) -> np.ndarray:
        return np.array([
            self.x - self.width * 0.5,
            self.y - self.height * 0.5,
            self.x + self.width * 0.5,
            self.y + self.height * 0.5
        ], dtype=np.float32)


class CrowdedGateSimulator:
    """Generates synthetic crowded gate scenarios with customizable density and occlusion levels."""

    def __init__(
        self,
        num_pedestrians: int = 20,
        num_frames: int = 180,
        frame_width: int = 1280,
        frame_height: int = 720,
        occlusion_severity: float = 0.5,  # [0.0 = sparse, 1.0 = extreme bottleneck]
        detector_noise_sigma: float = 2.0,
        detector_miss_rate: float = 0.05,
        feature_dim: int = 128,
        random_seed: int = 42
    ):
        self.num_pedestrians = num_pedestrians
        self.num_frames = num_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.occlusion_severity = occlusion_severity
        self.detector_noise_sigma = detector_noise_sigma
        self.detector_miss_rate = detector_miss_rate
        self.feature_dim = feature_dim
        self.random_seed = random_seed

        np.random.seed(random_seed)
        self.agents: List[PedestrianAgent] = []
        self._initialize_scenario()

    def _initialize_scenario(self):
        """Create structured pedestrian flows designed to produce intense crossings and occlusions."""
        self.agents.clear()
        gate_x = self.frame_width * 0.5
        gate_y = self.frame_height * 0.5

        # Corridor gate width tightens with occlusion severity
        corridor_height = max(100.0, 280.0 * (1.2 - 0.7 * self.occlusion_severity))

        for i in range(1, self.num_pedestrians + 1):
            direction = 1 if (i % 2 == 1) else -1  # 1: Left-to-Right, -1: Right-to-Left
            
            # Start positions
            if direction == 1:
                start_x = np.random.uniform(50.0, 200.0)
                vx = np.random.uniform(3.5, 6.0)
            else:
                start_x = np.random.uniform(self.frame_width - 200.0, self.frame_width - 50.0)
                vx = -np.random.uniform(3.5, 6.0)

            # Funneling towards central gate
            start_y = gate_y + np.random.uniform(-corridor_height * 0.5, corridor_height * 0.5)
            # Slight drift towards center line
            vy = np.random.uniform(-0.6, 0.6)

            # Staggered entry times to create clusters and crossing pairs
            # With high occlusion severity, multiple pairs spawn at the exact same frame!
            pair_group = i // 2
            start_frame = int(1 + pair_group * max(5, int(25 * (1.1 - self.occlusion_severity))))

            total_frames = int(abs(self.frame_width - 150) / abs(vx)) + 20

            # Scale height by y-position (perspective effect: closer = lower = larger)
            persp_scale = 0.85 + 0.35 * (start_y / self.frame_height)
            w = 52.0 * persp_scale
            h = 135.0 * persp_scale

            agent = PedestrianAgent(
                gt_id=i,
                start_x=start_x,
                start_y=start_y,
                vx=vx,
                vy=vy,
                width=w,
                height=h,
                start_frame=start_frame,
                total_frames=total_frames,
                feature_dim=self.feature_dim
            )
            self.agents.append(agent)

    def generate_dataset(self) -> Dict[str, Any]:
        """Simulate all frames and generate ground-truth and noisy detections.

        Returns:
            Dict containing:
                'frames': List of frame indices
                'ground_truth': Dict[frame_id -> List[Dict(id, bbox, visibility)]]
                'detections': Dict[frame_id -> Dict(boxes, scores, features, gt_ids)]
                'metadata': Simulation parameters
        """
        gt_data = {}
        det_data = {}

        for frame in range(1, self.num_frames + 1):
            active_agents = []
            for agent in self.agents:
                agent.step(frame, (0, 0, self.frame_width, self.frame_height))
                if agent.is_active:
                    # Check if within frame bounds
                    box = agent.get_bbox_xyxy()
                    if (box[2] > 20 and box[0] < self.frame_width - 20 and
                        box[3] > 20 and box[1] < self.frame_height - 20):
                        active_agents.append(agent)

            # Compute mutual overlaps among active agents
            frame_gt_list = []
            frame_det_boxes = []
            frame_det_scores = []
            frame_det_features = []
            frame_det_gt_ids = []

            K = len(active_agents)
            boxes_all = np.array([a.get_bbox_xyxy() for a in active_agents], dtype=np.float32) if K > 0 else np.empty((0, 4))

            # Determine visibility based on mutual occlusion
            visibilities = np.ones(K, dtype=np.float32)
            if K > 1:
                for a_idx in range(K):
                    for b_idx in range(K):
                        if a_idx == b_idx:
                            continue
                        # B occluding A: if B is in front (lower y center) and overlaps A
                        cy_a = (boxes_all[a_idx, 1] + boxes_all[a_idx, 3]) * 0.5
                        cy_b = (boxes_all[b_idx, 1] + boxes_all[b_idx, 3]) * 0.5

                        # Calculate intersection
                        x1 = max(boxes_all[a_idx, 0], boxes_all[b_idx, 0])
                        y1 = max(boxes_all[a_idx, 1], boxes_all[b_idx, 1])
                        x2 = min(boxes_all[a_idx, 2], boxes_all[b_idx, 2])
                        y2 = min(boxes_all[a_idx, 3], boxes_all[b_idx, 3])
                        inter_w = max(0.0, x2 - x1)
                        inter_h = max(0.0, y2 - y1)
                        inter_area = inter_w * inter_h
                        area_a = (boxes_all[a_idx, 2] - boxes_all[a_idx, 0]) * (boxes_all[a_idx, 3] - boxes_all[a_idx, 1])

                        if area_a > 0 and inter_area > 0 and cy_b > cy_a:
                            # B is closer to camera (lower in frame) than A, so B occludes A!
                            fraction_occ = inter_area / area_a
                            visibilities[a_idx] = max(0.05, visibilities[a_idx] - fraction_occ)

            for idx, agent in enumerate(active_agents):
                vis = float(visibilities[idx])
                box = boxes_all[idx].copy()

                frame_gt_list.append({
                    "id": agent.gt_id,
                    "bbox": box,
                    "visibility": vis,
                    "color": agent.color
                })

                # Simulated detection generation:
                # If heavily occluded (vis < 0.20), probability of detector miss is very high!
                prob_miss = self.detector_miss_rate + (1.0 - vis) * 0.65
                if np.random.rand() < prob_miss and vis < 0.35:
                    # Dropped detection due to heavy occlusion
                    continue

                # Add detector noise to bounding box
                noise = np.random.normal(0.0, self.detector_noise_sigma, size=4)
                det_box = box + noise

                # Detection score drops with occlusion
                base_score = np.random.uniform(0.85, 0.98)
                score = base_score * (0.35 + 0.65 * vis)
                score = float(np.clip(score, 0.15, 0.99))

                # Feature extraction: base feature + occlusion corruption
                feature = agent.base_feature.copy()
                if vis < 0.85:
                    # Realistic perturbation: noise norm is proportional to occlusion degree
                    sigma = ((1.0 - vis) * 0.35) / np.sqrt(self.feature_dim)
                    feature += np.random.normal(0.0, sigma, size=self.feature_dim).astype(np.float32)
                    feature /= np.linalg.norm(feature)

                frame_det_boxes.append(det_box)
                frame_det_scores.append(score)
                frame_det_features.append(feature)
                frame_det_gt_ids.append(agent.gt_id)

            gt_data[frame] = frame_gt_list
            det_data[frame] = {
                "boxes": np.array(frame_det_boxes, dtype=np.float32) if len(frame_det_boxes) > 0 else np.empty((0, 4), dtype=np.float32),
                "scores": np.array(frame_det_scores, dtype=np.float32) if len(frame_det_scores) > 0 else np.empty((0,), dtype=np.float32),
                "features": np.array(frame_det_features, dtype=np.float32) if len(frame_det_features) > 0 else np.empty((0, self.feature_dim), dtype=np.float32),
                "gt_ids": frame_det_gt_ids
            }

        return {
            "num_frames": self.num_frames,
            "width": self.frame_width,
            "height": self.frame_height,
            "ground_truth": gt_data,
            "detections": det_data,
            "agents": self.agents,
        }
