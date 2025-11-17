"""
Environment Configuration for MAPPO Training

This module defines the environment configuration for training MAPPO agents
on the wildfire suppression task. The configuration includes environment
parameters, agent composition, fire dynamics, and reward settings.
"""

# =============================================================================
# BASE ENVIRONMENT CONFIGURATION
# =============================================================================

ENV_CONFIG = {
    # Grid and episode settings
    "size":22,                     # Grid size (25x25)
    "num_agents": 6,                # Total number of agents
    "max_steps": 300,              # Maximum steps per episode
    "initial_fire_size": 3,         # Initial fire area (3x3)

    # Agent type configuration (must sum to num_agents)
    "num_helicopters": 2,           # Fast, high efficiency
    "num_trucks": 4,                # Medium speed, high efficiency
    "num_crews": 0,                 # Slow, low efficiency

    # Agent start position (각 에이전트를 다른 위치에 배치)
    # Grid corners for 4 agents: top-left, top-right, bottom-left, bottom-right
    "agent_start_positions": ((18, 1), (20, 3), (19, 1), (20, 1), (20, 2), (19, 2)),
    # "agent_start_positions": ((20, 1),) * 9,

    # Fire dynamics parameters
    # These control how fire spreads and decays
    "alpha": 0.05,                  # Fire spread probability parameter
    "beta": 0.9,                    # Fire intensity parameter
    "delta_beta": 0.54,             # Fire decay/burnout parameter

    # Observation settings
    "partial_obs": False,           # Full observability (recommended to start)
    "agent_view_size": 10,          # Only used if partial_obs=True

    # Reward settings
    "cooperative_reward": False,     # All agents share same reward (MAPPO standard)
    "selfishness_weight": 0.2,      # Only used if cooperative_reward=False

    # Reward shaping (supports multiple reward functions)
    "reward_shaping": "individual",    # Options: "cooperative", "ramadan", "individual", "individual2", None
    "reward_shaping_config": None,      # None = use defaults from reward_functions.py

    # Rendering
    "render_mode": "rgb_array",     # Required for GIF generation
}