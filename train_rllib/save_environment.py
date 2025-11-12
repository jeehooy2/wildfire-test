# run4
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
    "reward_shaping": "individual2",    # Options: "cooperative", "ramadan", "individual", "individual2", None
    "reward_shaping_config": None,      # None = use defaults from reward_functions.py

    # Rendering
    "render_mode": "rgb_array",     # Required for GIF generation
}

# run3
ENV_CONFIG = {
    # Grid and episode settings
    "size":17,                     # Grid size (25x25)
    "num_agents": 3,                # Total number of agents
    "max_steps": 300,              # Maximum steps per episode
    "initial_fire_size": 2,         # Initial fire area (3x3)

    # Agent type configuration (must sum to num_agents)
    "num_helicopters": 1,           # Fast, high efficiency
    "num_trucks": 1,                # Medium speed, high efficiency
    "num_crews": 1,                 # Slow, low efficiency

    # Agent start position
    "agent_start_positions": ((15, 1),) * 3,
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
    "reward_shaping": "individual2",    # Options: "cooperative", "ramadan", "individual", "individual2", None
    "reward_shaping_config": None,      # None = use defaults from reward_functions.py

    # Rendering
    "render_mode": "rgb_array",     # Required for GIF generation
}