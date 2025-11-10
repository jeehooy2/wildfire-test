"""
Training module for wildfire environment using RLlib MAPPO
"""

from train_rllib.environment import ENV_CONFIG
from train_rllib.wildfire_rllib_wrapper_new import WildfireRLlibEnv

__all__ = [
    "ENV_CONFIG",
    "WildfireRLlibEnv",
]
