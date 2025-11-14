"""
Training module for wildfire environment using RLlib MAPPO
"""

from train_marllib.environment import ENV_CONFIG
from train_marllib.wildfire_rllib_wrapper import WildfireRLlibEnv

__all__ = [
    "ENV_CONFIG",
    "WildfireRLlibEnv",
]
