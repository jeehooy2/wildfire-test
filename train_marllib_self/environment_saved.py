# (mappo) rtest0_obs_size_9
ENV_CONFIG = {
    # Grid and episode settings
    "size":22,                     # Grid size (25x25)
    "num_agents": 8,                # Total number of agents
    "max_steps": 300,              # Maximum steps per episode
    "initial_fire_size": 3,         # Initial fire area (3x3)

    # Agent type configuration (must sum to num_agents)
    "num_helicopters": 2,           # Fast, high efficiency
    "num_trucks": 6,                # Medium speed, high efficiency
    "num_crews": 0,                 # Slow, low efficiency

    # Agent start position (각 에이전트를 다른 위치에 배치)
    # Grid corners for 4 agents: top-left, top-right, bottom-left, bottom-right
    "agent_start_positions": ((17, 1), (20, 4), (18, 1), (20, 3), (19, 1), (20, 1), (20, 2), (19, 2)),
    # "agent_start_positions": ((20, 1),) * 9,

    # Fire dynamics parameters
    # These control how fire spreads and decays
    "alpha": 0.05,                  # Fire spread probability parameter
    "beta": 0.9,                    # Fire intensity parameter
    "delta_beta": 0.54,             # Fire decay/burnout parameter

    # Observation settings
    "partial_obs": False,            # Full observability (recommended to start)
    "agent_view_size": 10,         # Only used if partial_obs=True

    # Reward settings
    "cooperative_reward": False,     # All agents share same reward (MAPPO standard)
    "selfishness_weight": 0.2,      # Only used if cooperative_reward=False

    # Reward shaping (supports multiple reward functions)
    "reward_shaping": "individual",    # Options: "cooperative", "ramadan", "individual", "individual2", None
    "reward_shaping_config": None,      # None = use defaults from reward_functions.py

    # Rendering
    "render_mode": "rgb_array",     # Required for GIF generation
}

# (mappo) run_test_fraction_speeds_size_40
ENV_CONFIG = {
    # Grid and episode settings
    "size":42,                     # Grid size (25x25)
    "num_agents": 9,                # Total number of agents
    "max_steps": 300,              # Maximum steps per episode
    "initial_fire_size": 3,         # Initial fire area (3x3)

    # Agent type configuration (must sum to num_agents)
    "num_helicopters": 3,           # Fast, high efficiency
    "num_trucks": 6,                # Medium speed, high efficiency
    "num_crews": 0,                 # Slow, low efficiency

    # Agent start position (각 에이전트를 다른 위치에 배치)
    # Grid corners for 4 agents: top-left, top-right, bottom-left, bottom-right
    "agent_start_positions": ((37, 1), (38, 1), (40, 3), (39, 1), (38, 2), (39, 3), (40, 1), (40, 2), (39, 2)),
    # "agent_start_positions": ((20, 1),) * 9,

    # Fire dynamics parameters
    # These control how fire spreads and decays
    "alpha": 0.05,                  # Fire spread probability parameter
    "beta": 0.9,                    # Fire intensity parameter
    "delta_beta": 0.54,             # Fire decay/burnout parameter

    # Observation settings
    "partial_obs": True,           # Full observability (recommended to start)
    "agent_view_size": 5,         # Only used if partial_obs=True

    # Reward settings
    "cooperative_reward": False,     # All agents share same reward (MAPPO standard)
    "selfishness_weight": 0.2,      # Only used if cooperative_reward=False

    # Reward shaping (supports multiple reward functions)
    "reward_shaping": "individual",    # Options: "cooperative", "ramadan", "individual", "individual2", None
    "reward_shaping_config": None,      # None = use defaults from reward_functions.py

    # Rendering
    "render_mode": "rgb_array",     # Required for GIF generation
}

# (mappo) run_test_fraction_speeds
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
    "partial_obs": True,           # Full observability (recommended to start)
    "agent_view_size": 5,         # Only used if partial_obs=True

    # Reward settings
    "cooperative_reward": False,     # All agents share same reward (MAPPO standard)
    "selfishness_weight": 0.2,      # Only used if cooperative_reward=False

    # Reward shaping (supports multiple reward functions)
    "reward_shaping": "individual",    # Options: "cooperative", "ramadan", "individual", "individual2", None
    "reward_shaping_config": None,      # None = use defaults from reward_functions.py

    # Rendering
    "render_mode": "rgb_array",     # Required for GIF generation
}

# (mappo) run_test_global_state_flag
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
    "agent_view_size": 5,         # Only used if partial_obs=True

    # Reward settings
    "cooperative_reward": False,     # All agents share same reward (MAPPO standard)
    "selfishness_weight": 0.2,      # Only used if cooperative_reward=False

    # Reward shaping (supports multiple reward functions)
    "reward_shaping": "individual",    # Options: "cooperative", "ramadan", "individual", "individual2", None
    "reward_shaping_config": None,      # None = use defaults from reward_functions.py

    # Rendering
    "render_mode": "rgb_array",     # Required for GIF generation
}

# run14
ENV_CONFIG = {
    # Grid and episode settings
    "size":17,                     # Grid size (25x25)
    "num_agents": 2,                # Total number of agents
    "max_steps": 300,              # Maximum steps per episode
    "initial_fire_size": 2,         # Initial fire area (3x3)

    # Agent type configuration (must sum to num_agents)
    "num_helicopters": 2,           # Fast, high efficiency
    "num_trucks": 0,                # Medium speed, high efficiency
    "num_crews": 0,                 # Slow, low efficiency

    # Agent start position (각 에이전트를 다른 위치에 배치)
    # Grid corners for 4 agents: top-left, top-right, bottom-left, bottom-right
    "agent_start_positions": ((15, 1), (1, 15)),
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

# run13
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