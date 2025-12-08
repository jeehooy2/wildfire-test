"""
Reward functions for wildfire suppression environment

This module provides three reward calculation strategies:
1. Cooperative: All agents receive same shared reward
2. Ramadan: Cumulative reward based on preserved/extinguished/burned trees
3. Individual: Each agent gets reward based on their contribution plus shared component
"""

import numpy as np


def cooperative_reward(
    trees_to_fire_state,
    trees_extinguished_by_agents,
    num_agents,
    r_extinguish=1.0,
    r_new_fire=0.5, 
):
    """
    Cooperative reward function - all agents receive same reward

    R = r_e * T_e - r_n * T_n

    Where:
    - T_e: Trees extinguished this step
    - T_n: New trees that caught fire this step
    - r_e: Reward weight for extinguishing fires
    - r_n: Penalty weight for new fires

    Parameters
    ----------
    trees_to_fire_state : list
        List of trees that caught fire this step
    trees_extinguished_by_agents : int
        Total number of trees extinguished by agents this step
    num_agents : int
        Number of agents in environment
    r_extinguish : float
        Reward coefficient for extinguishing trees
    r_new_fire : float
        Penalty coefficient for new fires

    Returns
    -------
    dict
        Dictionary mapping agent indices to rewards
    """
    T_e = trees_extinguished_by_agents
    T_n = len(trees_to_fire_state)

    # Compute cooperative reward
    reward = r_extinguish * T_e - r_new_fire * T_n

    # All agents get same reward
    return {str(i): reward for i in range(num_agents)}


def ramadan_reward(
    trees_preserved,
    cumulative_extinguished,
    cumulative_burned,
    cumulative_actions,
    num_agents,
    a=0.1,
    b=1.0,
    c=0.5,
    d=0.01,
):
    """
    Ramadan (2024) cumulative reward function

    R = a*T_p + b*T_e - c*T_b - d*A_total

    Where (all cumulative over episode):
    - T_p: Preserved trees (trees never affected by fire during episode)
    - T_e: Successfully extinguished trees (cumulative trees rescued by agents)
    - T_b: Lost trees (cumulative trees burned by fire)
    - A_total: Total number of actions taken by all agents
    - (a, b, c, d): Configurable coefficients

    Parameters
    ----------
    trees_preserved : int
        Number of trees that never caught fire
    cumulative_extinguished : int
        Total trees extinguished so far
    cumulative_burned : int
        Total trees burned so far
    cumulative_actions : int
        Total actions taken by all agents so far
    num_agents : int
        Number of agents in environment
    a : float
        Weight for preserved trees
    b : float
        Weight for extinguished trees
    c : float
        Weight for burned trees
    d : float
        Weight for action count

    Returns
    -------
    dict
        Dictionary mapping agent indices to rewards
    """
    # Compute cumulative Ramadan reward
    cumulative_reward = (
        a * trees_preserved +
        b * cumulative_extinguished -
        c * cumulative_burned -
        d * cumulative_actions
    )

    # All agents get same cumulative reward
    return {str(i): cumulative_reward for i in range(num_agents)}

def individual_reward(
    trees_to_fire_state,
    trees_to_burnt_state,
    agent_tree_extinguished,
    num_agents,
    r_extinguish=1.0,
    r_new_fire=0.5,
    r_new_burnt=0.3,
    r_shared=0.1,
):
    """
    Individual reward function based on per-agent contribution

    R_i = r_e * T_e_i + r_shared * (r_e * T_e_global - r_n * T_n)

    Where (per time step):
    - T_e_i: Trees extinguished by this specific agent this step
    - T_e_global: Total trees extinguished by all agents this step
    - T_n: New trees that caught fire this step
    - r_e: Reward weight for extinguishing fires
    - r_n: Penalty weight for new fires
    - r_shared: Weight for shared/cooperative component (0-1)

    Parameters
    ----------
    trees_to_fire_state : list
        List of trees that caught fire this step
    agent_tree_extinguished : dict
        Dictionary mapping agent index to 1 or 0 (extinguished or not)
    num_agents : int
        Number of agents in environment
    r_extinguish : float
        Reward coefficient for extinguishing trees
    r_new_fire : float
        Penalty coefficient for new fires
    r_shared : float
        Weight for shared reward component (0-1)

    Returns
    -------
    dict
        Dictionary mapping agent indices to rewards
    """
    T_e = sum(agent_tree_extinguished.values())
    T_n = len(trees_to_fire_state)
    T_b = len(trees_to_burnt_state)
    shared_reward = r_extinguish * T_e - r_new_fire * T_n - r_new_burnt * T_b

    rewards = {}
    for i in range(num_agents):
        # Individual contribution: trees this agent extinguished
        extinguished_fire = agent_tree_extinguished.get(i, 0)
        individual_reward = r_extinguish * extinguished_fire

        # Combined reward: individual + shared component
        rewards[str(i)] = individual_reward + r_shared * shared_reward

    return rewards

# def individual_reward(
#     trees_to_fire_state,
#     extinguished_per_agent,
#     num_agents,
#     r_extinguish=1.0,
#     r_new_fire=0.5,
#     r_shared=0.3,
# ):
#     """
#     Individual reward function based on per-agent contribution

#     R_i = r_e * T_e_i + r_shared * (r_e * T_e_global - r_n * T_n)

#     Where (per time step):
#     - T_e_i: Trees extinguished by this specific agent this step
#     - T_e_global: Total trees extinguished by all agents this step
#     - T_n: New trees that caught fire this step
#     - r_e: Reward weight for extinguishing fires
#     - r_n: Penalty weight for new fires
#     - r_shared: Weight for shared/cooperative component (0-1)

#     Parameters
#     ----------
#     trees_to_fire_state : list
#         List of trees that caught fire this step
#     extinguished_per_agent : dict
#         Dictionary mapping agent index to number of trees they extinguished
#     num_agents : int
#         Number of agents in environment
#     r_extinguish : float
#         Reward coefficient for extinguishing trees
#     r_new_fire : float
#         Penalty coefficient for new fires
#     r_shared : float
#         Weight for shared reward component (0-1)

#     Returns
#     -------
#     dict
#         Dictionary mapping agent indices to rewards
#     """
#     T_n = len(trees_to_fire_state)
#     T_e_global = sum(extinguished_per_agent.values())

#     rewards = {}
#     for i in range(num_agents):
#         # Individual contribution: trees this agent extinguished
#         T_e_individual = extinguished_per_agent.get(i, 0)
#         individual_reward = r_extinguish * T_e_individual

#         # Combined reward: individual + shared component
#         rewards[str(i)] = individual_reward + r_shared * (r_extinguish * T_e_global - r_new_fire * T_n)

#     return rewards


def individual2_reward(
    trees_extinguished,
    trees_burnt,
    trees_to_fire,
    agent_tree_extinguished,
    agent_on_fire_tree,
    num_agents,
    num_agents_moved,
    alpha=0.3,
):
    """
    Individual2 reward function with differentiated rewards per agent

    R_i = alpha * (1.0 * T_e - 0.5 * T_b) + (1 - alpha) * (1.0 * T_e_individual + 1.0 * above_tree_on_fire)

    Where (per time step):
    - T_e: Total trees extinguished this step (on_fire -> healthy)
    - T_o: New trees that caught fire this step (healthy -> on_fire) [tracked but not used in reward]
    - T_b: Total trees burnt this step (on_fire -> burnt)
    - extinguished_fire: Whether this agent's tree was extinguished (0 or 1)
    - above_tree_on_fire: Whether this agent is on a burning tree (0 or 1)
    - num_agents_moved: Number of agents that actually moved this step (position changed, excluding STILL and agents with speed<1 that didn't accumulate enough to move)
    - alpha: Weight for shared/cooperative component (0-1)

    Parameters
    ----------
    trees_extinguished : int
        Total number of trees extinguished this step
    trees_burnt : int
        Total number of trees burnt this step
    trees_to_fire : int
        Total number of trees that caught fire this step (tracked but not used)
    agent_tree_extinguished : dict
        Dictionary mapping agent index to whether their tree was extinguished (0 or 1)
    agent_on_fire_tree : dict
        Dictionary mapping agent index to whether they are on a burning tree (0 or 1)
    num_agents : int
        Number of agents in environment
    num_agents_moved : int
        Number of agents that actually moved this step (position changed)
    alpha : float
        Weight for shared reward component (0-1), default 0.3

    Returns
    -------
    dict
        Dictionary mapping agent indices to rewards
    """
    # Shared component: cooperative part of the reward
    T_e = trees_extinguished
    T_b = trees_burnt
    C = num_agents_moved
    shared_reward = 1.0 * T_e - 0.5 * T_b
    # shared_reward = 1.0 * T_e - 0.5 * T_b - 0.1 * C

    # Calculate reward for each agent
    rewards = {}
    for i in range(num_agents):
        # Individual component
        extinguished_fire = agent_tree_extinguished.get(i, 0)
        above_tree_on_fire = agent_on_fire_tree.get(i, 0)
        individual_component = 1.0 * above_tree_on_fire + 1.0 * extinguished_fire

        # Combined reward
        rewards[str(i)] = alpha * shared_reward + (1 - alpha) * individual_component

    return rewards


# Default configurations for each reward function
DEFAULT_CONFIGS = {
    "cooperative": {
        "r_extinguish": 1.0,
        "r_new_fire": 0.5,
    },
    "ramadan": {
        "a": 0.1,   # Weight for preserved trees
        "b": 1.0,   # Weight for extinguished trees
        "c": 0.5,   # Weight for burned trees
        "d": 0.01,  # Weight for action count
    },
    "individual": {
        "r_extinguish": 1.0,
        "r_new_fire": 0.5,
        # "r_shared": 0.3,
    },
    "individual2": {
        "alpha": 0.3,  # Weight for shared/cooperative component
    },
}


def get_reward_function(reward_type):
    """
    Get reward function by name

    Parameters
    ----------
    reward_type : str
        Type of reward function: "cooperative", "ramadan", "individual", or "individual2"

    Returns
    -------
    callable
        The reward function

    Raises
    ------
    ValueError
        If reward_type is not recognized
    """
    reward_functions = {
        "cooperative": cooperative_reward,
        "ramadan": ramadan_reward,
        "individual": individual_reward,
        "individual2": individual2_reward,
    }

    if reward_type not in reward_functions:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available types: {list(reward_functions.keys())}"
        )

    return reward_functions[reward_type]
