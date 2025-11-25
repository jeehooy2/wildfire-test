"""
Reward functions for wildfire suppression environment

This module provides three reward calculation strategies:
1. Cooperative: All agents receive same shared reward
2. Ramadan: Cumulative reward based on preserved/extinguished/burned trees
3. Individual: Each agent gets reward based on their contribution plus shared component
"""

import numpy as np


# def cooperative_reward(
#     trees_to_fire_state,
#     trees_to_burnt_state,
#     trees_to_healthy_state,
#     agent_tree_extinguished,
#     agent_on_fire_tree,
#     num_agents,
#     is_episode_end=False,
#     current_step=0,
#     max_steps=100,
#     total_healthy_trees=0,
#     total_trees=1,
#     w_extinguish=2.0,
#     w_on_fire=0.3,
#     w_new_fire=0.2,
#     w_burnt=0.5,
#     w_episode_health=10.0,
#     w_episode_speed=1.0,
#     w_healthy_ratio=2.0,
#     w_time_penalty=0.01,
# ):
#     """
#     Cooperative reward function with three objectives:
#     1. 불 진화 및 건강한 나무 비율 유지 (주요 목표)
#     2. 정규화된 시간 패널티로 빠른 종료 유도 (효율성)
#     3. 에피소드 종료 시 최종 성과 보너스 (장기 목표)

#     R = r_step + r_time_penalty(step) + r_episode_end

#     Where:
#     r_step: 매 스텝의 즉각적 신호 (진화, 새 불, healthy ratio 등)
#     r_time_penalty(step): 진행도 기반 시간 비용 = -w_time_penalty * (step/max_steps)
#                          → 에피소드 초반: 낮은 페널티, 후반: 높은 페널티
#     r_episode_end: 에피소드 마지막에만 제공되는 보너스 (목표 달성도)
#     -------
#     dict
#         Dictionary mapping agent indices to rewards
#     """
#     T_e = len(trees_to_healthy_state)
#     T_n = len(trees_to_fire_state)
#     T_b = len(trees_to_burnt_state)
#     agents_on_fire = sum(1 for v in agent_on_fire_tree.values() if v == 1)

#     # ===== 1. 매 스텝 리워드 (즉각적 피드백) =====
#     # 건강한 나무 비율 보상 (지속적 긍정 신호)
#     healthy_ratio_reward = 0.0
#     if total_trees > 0:
#         healthy_ratio_reward = w_healthy_ratio * (total_healthy_trees / total_trees)

#     # # 시간 페널티 (매 스텝마다 적용 - 빠른 종료 유도)
#     # normalized_progress = current_step / max_steps if max_steps > 0 else 0.0
#     # time_penalty = w_time_penalty * normalized_progress

#     shared_reward = (
#         w_extinguish * T_e +
#         w_on_fire * agents_on_fire +
#         healthy_ratio_reward -
#         w_new_fire * T_n -
#         w_burnt * T_b
#     )

#     # ===== 3. 에이전트별 리워드 계산 =====
#     rewards = {}
#     for i in range(num_agents):
#         extinguished_fire = agent_tree_extinguished.get(i, 0)
#         on_fire = agent_on_fire_tree.get(i, 0)

#         # 개별 기여도 (해당 에이전트가 한 일)
#         individual_reward = w_extinguish * extinguished_fire + w_on_fire * on_fire

#         # 최종 리워드 = 공동 리워드 + 개별 리워드 + 에피소드 보너스
#         # 70% 공동 협력, 30% 개별 기여 + 에피소드 종료 보너스는 모두에게 동등하게
#         rewards[str(i)] = 0.8 * shared_reward + 0.2 * individual_reward

#     return rewards


def cooperative_reward(
    trees_to_fire_state,
    trees_to_burnt_state,
    trees_to_healthy_state,
    agent_tree_extinguished,
    agent_on_fire_tree,
    num_agents,
    is_episode_end=False,
    current_step=0,
    max_steps=100,
    total_healthy_trees=0,
    total_trees=1,
    w_extinguish=2.0,
    w_shared=0.7,
    w_new_fire=0.2,
    w_new_burnt=0.5,
    w_episode_health=10.0,
    w_episode_speed=1.0,
    w_healthy_ratio=2.0,
    w_time_penalty=0.01,
):
    """
    Cooperative reward function with three objectives:
    1. 불 진화 및 건강한 나무 비율 유지 (주요 목표)
    2. 정규화된 시간 패널티로 빠른 종료 유도 (효율성)
    3. 에피소드 종료 시 최종 성과 보너스 (장기 목표)

    R = r_step + r_time_penalty(step) + r_episode_end

    Where:
    r_step: 매 스텝의 즉각적 신호 (진화, 새 불, healthy ratio 등)
    r_time_penalty(step): 진행도 기반 시간 비용 = -w_time_penalty * (step/max_steps)
                         → 에피소드 초반: 낮은 페널티, 후반: 높은 페널티
    r_episode_end: 에피소드 마지막에만 제공되는 보너스 (목표 달성도)
    -------
    dict
        Dictionary mapping agent indices to rewards
    """
    T_e = len(trees_to_healthy_state)
    T_n = len(trees_to_fire_state)
    T_b = len(trees_to_burnt_state)

    healthy_ratio_reward = 0.0
    if total_trees > 0:
        healthy_ratio_reward = w_healthy_ratio * (total_healthy_trees / total_trees)

    # # 시간 페널티 (매 스텝마다 적용 - 빠른 종료 유도)
    # normalized_progress = current_step / max_steps if max_steps > 0 else 0.0
    # time_penalty = w_time_penalty * normalized_progress

    shared_reward = (
        w_extinguish * T_e +
        w_on_fire * agents_on_fire +
        healthy_ratio_reward -
        w_new_fire * T_n -
        w_burnt * T_b
    )

    # ===== 3. 에이전트별 리워드 계산 =====
    rewards = {}
    for i in range(num_agents):
        extinguished_fire = agent_tree_extinguished.get(i, 0)
        on_fire = agent_on_fire_tree.get(i, 0)

        # 개별 기여도 (해당 에이전트가 한 일)
        individual_reward = w_extinguish * extinguished_fire + w_on_fire * on_fire

        # 최종 리워드 = 공동 리워드 + 개별 리워드 + 에피소드 보너스
        # 70% 공동 협력, 30% 개별 기여 + 에피소드 종료 보너스는 모두에게 동등하게
        rewards[str(i)] = 0.8 * shared_reward + 0.2 * individual_reward

    return rewards

    T_e = sum(agent_tree_extinguished.values())
    T_n = len(trees_to_fire_state)
    T_b = len(trees_to_burnt_state)

    healthy_ratio_reward = 0.0
    if total_trees > 0:
        healthy_ratio_reward = w_healthy_ratio * (total_healthy_trees / total_trees)

    # Shared component: team performance
    shared_reward = r_extinguish * T_e + healthy_ratio_reward - r_new_fire * T_n - r_new_burnt * T_b

    rewards = {}
    for i in range(num_agents):
        # Individual contribution: trees this agent extinguished
        extinguished_fire = agent_tree_extinguished.get(i, 0)
        individual_reward = r_extinguish * extinguished_fire

        rewards[str(i)] = r_shared * shared_reward + (1 - r_shared) * individual_reward

    return rewards



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
    r_new_fire=0.3,
    r_new_burnt=0.5,
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
    agent_tree_extinguished : dict (이번 스텝에 진화한 나무 개수)
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


def cooperative2_reward(
    trees_to_fire_state,
    trees_to_burnt_state,
    agent_tree_extinguished,
    num_agents,
    r_extinguish=1.0,
    r_new_fire=0.15,
    r_new_burnt=0.25,
    r_shared=0.5,
    normalize_shared=True,
    current_step=0,
    max_steps=100,
    r_time_penalty=0.1,
):
    """
    Individual reward with balanced cooperative focus + time penalty
    (공동보상 비율을 높이면서도 안정적인 학습을 위한 개선 버전)

    시간 패널티: 에이전트가 산불 진화를 지연시키는 것을 방지
    - 에피소드 초반: 낮은 패널티
    - 에피소드 후반: 높은 패널티

    R_i = r_shared * (shared_reward / num_agents) + (1 - r_shared) * individual_reward - time_penalty

    Parameters
    ----------
    current_step : int
        Current step number in the episode
    max_steps : int
        Maximum number of steps in an episode
    r_time_penalty : float
        Time penalty weight (higher = stronger penalty for taking longer)

    Returns
    -------
    dict
        Dictionary mapping agent indices to rewards
    """
    T_e = sum(agent_tree_extinguished.values())
    T_n = len(trees_to_fire_state)
    T_b = len(trees_to_burnt_state)

    # Shared component: team performance
    shared_reward = r_extinguish * T_e - r_new_fire * T_n - r_new_burnt * T_b

    # Time penalty: 진행도 기반 (step / max_steps)
    # 진행도가 높을수록 페널티 증가 → 빠른 종료 유도
    # normalized_progress = current_step / max_steps if max_steps > 0 else 0.0
    # time_penalty = r_time_penalty * normalized_progress

    # Normalize shared reward by number of agents to prevent explosion
    # if normalize_shared and num_agents > 0:
    #     shared_reward = shared_reward / num_agents

    rewards = {}
    for i in range(num_agents):
        # Individual contribution: trees this agent extinguished
        extinguished_fire = agent_tree_extinguished.get(i, 0)
        individual_reward = r_extinguish * extinguished_fire

        # Combined reward: 50% shared + 50% individual - time penalty
        # 시간 패널티는 모든 에이전트에게 동등하게 적용
        rewards[str(i)] = r_shared * shared_reward + (1 - r_shared) * individual_reward 

    return rewards


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


def cooperative3_reward(
    trees_preserved,
    trees_to_fire_state,
    trees_to_burnt_state,
    trees_to_healthy_state,
    agent_tree_extinguished,
    agent_on_fire_tree,
    num_agents,
    is_episode_end=False,
    current_step=0,
    max_steps=100,
    total_healthy_trees=0,
    total_trees=1,
    w_extinguish=1.0,
    w_healthy=1.0,
    w_new_fire=1.0,
    w_new_burnt=5.0,
    w_time_penalty=0.1, #0.02
):
    """
    Cooperative reward function (v3) with improved positive signal balance

    Design philosophy:
    1. 주요 목표: 건강한 나무 비율 최대화 (지속적 양의 피드백)
    2. 부차 목표: 에피소드 길이 최소화 (효율성)
    3. 특징: Sparse 양의 보상 문제 해결

    """
    # 이번 스텝 통계
    T_p = trees_preserved / total_trees #보존된 나무 비율 (untouched by fire or agent)
    T_h = total_healthy_trees / total_trees if total_trees > 0 else 0.0
    T_e = len(trees_to_healthy_state) / total_trees  # 성공적으로 진화된 나무 비율
    T_n = len(trees_to_fire_state) / total_trees if total_trees > 0 else 0.0     # 새로 불타는 나무 비율
    T_b = len(trees_to_burnt_state) / total_trees if total_trees > 0 else 0.0    # 소실된 나무 비율

    # 3) 공동 보상 계산
    # shared_reward = (
    #     1.0 * T_e +
    #     0.5 * T_h -
    #     1.0 * T_n -
    #     2.0 * T_b
    # )

    shared_reward = (
        0.5 * T_p +
        10.0 * T_e -
        1.0 * T_n -
        5.0 * T_b
    )

    # shared_reward = (
    #     0.2 * T_h +
    #     10.0 * T_e -
    #     1.0 * T_n -
    #     5.0 * T_b
    # )

    # 4) 시간 페널티 (진행도 기반: 초반은 낮고 후반이 높음)
    # → 빠른 에피소드 종료를 유도하지만 처음부터는 압박하지 않음
    normalized_progress = current_step / max_steps if max_steps > 0 else 0.0
    time_penalty = w_time_penalty * normalized_progress

    rewards = {}
    # 5) 에이전트별 보상 할당
    for i in range(num_agents):
        extinguished_fire = agent_tree_extinguished.get(i, 0)
        individual_reward = 1.0 * extinguished_fire
        
        # 80% 공동, 20% 개별 기여 (신용 배분 개선)
        agent_reward = (
            0.8 * shared_reward +
            0.2 * individual_reward -
            time_penalty
        )
        
        rewards[str(i)] = agent_reward
    
    return rewards


# Default configurations for each reward function
DEFAULT_CONFIGS = {
    "cooperative": {
    },
    "ramadan": {
        "a": 0.1,   # Weight for preserved trees
        "b": 1.0,   # Weight for extinguished trees
        "c": 0.5,   # Weight for burned trees
        "d": 0.01,  # Weight for action count
    },
    "individual": {
    },
    "individual2": {
        "alpha": 0.3,  # Weight for shared/cooperative component
    },
    "cooperative2": {

    },
    "cooperative3": {
    },
}


 # "r_shared": 0.3,

def get_reward_function(reward_type):
    """
    Get reward function by name

    Parameters
    ----------
    reward_type : str
        Type of reward function: "cooperative", "ramadan", "individual", "individual2", "cooperative2", or "cooperative3"

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
        "cooperative2": cooperative2_reward,
        "cooperative3": cooperative3_reward,
    }

    if reward_type not in reward_functions:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available types: {list(reward_functions.keys())}"
        )

    return reward_functions[reward_type]
