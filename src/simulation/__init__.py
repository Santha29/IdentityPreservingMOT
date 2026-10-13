"""Simulation and video rendering package."""
from .crowded_gate_sim import CrowdedGateSimulator, PedestrianAgent
from .video_renderer import VideoRenderer, get_color_for_id

__all__ = [
    "CrowdedGateSimulator",
    "PedestrianAgent",
    "VideoRenderer",
    "get_color_for_id",
]
