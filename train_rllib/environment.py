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
    "size":17,                     # Grid size (25x25)
    "num_agents": 6,                # Total number of agents
    "max_steps": 300,              # Maximum steps per episode
    "initial_fire_size": 2,         # Initial fire area (3x3)

    # Agent type configuration (must sum to num_agents)
    "num_helicopters": 2,           # Fast, high efficiency
    "num_trucks": 2,                # Medium speed, high efficiency
    "num_crews": 2,                 # Slow, low efficiency

    # Agent start position
    "agent_start_positions": ((15, 1),) * 6,
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


# ENV_CONFIG = {
#     "num_agents": 2,
#         "size": 17,
#         "initial_fire_size": 2,
#         "cooperative_reward": False,  # 다중 에이전트 학습 (각자 보상)
#         "max_steps": 300,
#         "agent_start_positions": ((1, 1), (15, 15)),
#         "log_selfish_region_metrics": True,
#         "selfish_region_xmin": [7, 13],
#         "selfish_region_xmax": [9, 15],
#         "selfish_region_ymin": [7, 1],
#         "selfish_region_ymax": [9, 3],
#         "delta_beta": 0.7,     # ← 행동이 화재 소멸에 강하게 영향 주도록
#         "beta": 0.99,          # 기본값 유지 가능
#         "alpha": 0.05,         # 기본값 유지 가능(확산)
# }