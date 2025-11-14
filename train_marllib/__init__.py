"""
Training module for wildfire environment using RLlib MAPPO
"""

from train_marllib.environment import ENV_CONFIG

# Conditional imports to avoid dependency issues
try:
    from train_marllib.wildfire_marllib_wrapper import WildfireRLlibEnv
    _has_rllib_wrapper = True
except ImportError:
    _has_rllib_wrapper = False

try:
    from train_marllib.wildfire_marllib_wrapper import WildfireMARLlibEnv
    _has_marllib_wrapper = True
except ImportError:
    _has_marllib_wrapper = False

__all__ = [
    "ENV_CONFIG",
]

if _has_rllib_wrapper:
    __all__.append("WildfireRLlibEnv")

if _has_marllib_wrapper:
    __all__.append("WildfireMARLlibEnv")
