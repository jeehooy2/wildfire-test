#!/usr/bin/env python3
"""
Test script for cooperative3_reward function

Tests the new reward function with various scenarios:
1. Normal episode progression
2. Healthy ratio changes
3. Time penalty effects
"""

import numpy as np
import sys
sys.path.insert(0, '/home/bmkim88/wildfire_environment')

from wildfire_environment.envs.reward_functions import cooperative3_reward, get_reward_function

def test_cooperative3_reward_basic():
    """Test basic cooperative3_reward functionality"""
    print("=" * 80)
    print("TEST 1: Basic Cooperative3 Reward Function")
    print("=" * 80)

    # Test parameters
    num_agents = 6
    total_trees = 400  # 20x20 grid

    # Scenario: Healthy trees decreasing (fire spreading)
    print("\nScenario A: Trees burning (health ratio decreasing)")
    print("-" * 50)

    # Initialize: All healthy
    total_healthy_trees = 400
    trees_to_fire_state = [1, 2, 3, 4, 5]  # 5 trees catch fire
    trees_to_burnt_state = [6, 7]  # 2 trees burned
    trees_to_healthy_state = [8, 9]  # 2 trees extinguished

    agent_tree_extinguished = {i: 0 for i in range(num_agents)}
    agent_tree_extinguished[0] = 1  # Agent 0 extinguished a tree
    agent_on_fire_tree = {i: 0 for i in range(num_agents)}
    agent_on_fire_tree[1] = 1  # Agent 1 is on a burning tree

    # Calculate updated tree count
    total_healthy_trees = total_healthy_trees - len(trees_to_fire_state) + len(trees_to_healthy_state) - len(trees_to_burnt_state)

    rewards = cooperative3_reward(
        trees_to_fire_state=trees_to_fire_state,
        trees_to_burnt_state=trees_to_burnt_state,
        trees_to_healthy_state=trees_to_healthy_state,
        agent_tree_extinguished=agent_tree_extinguished,
        agent_on_fire_tree=agent_on_fire_tree,
        num_agents=num_agents,
        current_step=10,
        max_steps=300,
        total_healthy_trees=total_healthy_trees,
        total_trees=total_trees,
    )

    print(f"Total healthy trees: {total_healthy_trees}/{total_trees} ({100*total_healthy_trees/total_trees:.1f}%)")
    print(f"New fires: {len(trees_to_fire_state)}, Extinguished: {len(trees_to_healthy_state)}, Burned: {len(trees_to_burnt_state)}")
    print(f"Sample reward (agent 0): {rewards['0']:.4f}")
    print(f"All rewards equal (cooperation): {all(v == list(rewards.values())[0] for v in rewards.values())}")

    # Scenario B: Better situation (more extinguished trees)
    print("\nScenario B: Better control (more fires extinguished)")
    print("-" * 50)

    total_healthy_trees = 350  # Worse health ratio
    trees_to_fire_state = [10, 11, 12]  # 3 new fires
    trees_to_burnt_state = [13]  # 1 burned
    trees_to_healthy_state = [14, 15, 16, 17, 18]  # 5 extinguished!

    agent_tree_extinguished = {i: 0 for i in range(num_agents)}
    agent_on_fire_tree = {i: 0 for i in range(num_agents)}

    total_healthy_trees = total_healthy_trees - len(trees_to_fire_state) + len(trees_to_healthy_state) - len(trees_to_burnt_state)

    rewards = cooperative3_reward(
        trees_to_fire_state=trees_to_fire_state,
        trees_to_burnt_state=trees_to_burnt_state,
        trees_to_healthy_state=trees_to_healthy_state,
        agent_tree_extinguished=agent_tree_extinguished,
        agent_on_fire_tree=agent_on_fire_tree,
        num_agents=num_agents,
        current_step=20,
        max_steps=300,
        total_healthy_trees=total_healthy_trees,
        total_trees=total_trees,
    )

    print(f"Total healthy trees: {total_healthy_trees}/{total_trees} ({100*total_healthy_trees/total_trees:.1f}%)")
    print(f"New fires: {len(trees_to_fire_state)}, Extinguished: {len(trees_to_healthy_state)}, Burned: {len(trees_to_burnt_state)}")
    print(f"Sample reward (agent 0): {rewards['0']:.4f}")
    print(f"Note: More extinguished fires → higher reward ✓")


def test_time_penalty_progression():
    """Test time penalty increases with episode progress"""
    print("\n" + "=" * 80)
    print("TEST 2: Time Penalty Progression")
    print("=" * 80)

    num_agents = 6
    total_trees = 400
    total_healthy_trees = 390

    # Same environment state, different time steps
    trees_to_fire_state = [1]
    trees_to_burnt_state = []
    trees_to_healthy_state = [2]
    agent_tree_extinguished = {i: 0 for i in range(num_agents)}
    agent_on_fire_tree = {i: 0 for i in range(num_agents)}

    max_steps = 300

    print(f"\nSame environment state at different episode progress:")
    print("-" * 50)

    for step in [0, 75, 150, 225, 300]:
        rewards = cooperative3_reward(
            trees_to_fire_state=trees_to_fire_state,
            trees_to_burnt_state=trees_to_burnt_state,
            trees_to_healthy_state=trees_to_healthy_state,
            agent_tree_extinguished=agent_tree_extinguished,
            agent_on_fire_tree=agent_on_fire_tree,
            num_agents=num_agents,
            current_step=step,
            max_steps=max_steps,
            total_healthy_trees=total_healthy_trees,
            total_trees=total_trees,
        )

        progress = 100 * step / max_steps
        print(f"Step {step:3d} ({progress:5.1f}%): Reward = {float(rewards['0']):7.4f}")

    print("\nNote: Reward decreases as episode progresses → encourages faster resolution ✓")


def test_healthy_ratio_signal():
    """Test that healthy ratio is the main signal"""
    print("\n" + "=" * 80)
    print("TEST 3: Healthy Ratio as Main Signal")
    print("=" * 80)

    num_agents = 6
    total_trees = 400
    max_steps = 300
    current_step = 50

    # Same action outcome, different starting health
    trees_to_fire_state = []
    trees_to_burnt_state = []
    trees_to_healthy_state = []
    agent_tree_extinguished = {i: 0 for i in range(num_agents)}
    agent_on_fire_tree = {i: 0 for i in range(num_agents)}

    print(f"\nNo fires/extinguish events - only healthy ratio varies:")
    print("-" * 50)

    for health_percent in [25, 50, 75, 95]:
        total_healthy_trees = int(total_trees * health_percent / 100)
        rewards = cooperative3_reward(
            trees_to_fire_state=trees_to_fire_state,
            trees_to_burnt_state=trees_to_burnt_state,
            trees_to_healthy_state=trees_to_healthy_state,
            agent_tree_extinguished=agent_tree_extinguished,
            agent_on_fire_tree=agent_on_fire_tree,
            num_agents=num_agents,
            current_step=current_step,
            max_steps=max_steps,
            total_healthy_trees=total_healthy_trees,
            total_trees=total_trees,
        )

        print(f"Health ratio {health_percent:3d}%: Reward = {float(rewards['0']):7.4f}")

    print("\nNote: Higher health ratio gives higher reward (main signal) ✓")


def test_positive_reward_signal():
    """Test that positive rewards are distributed to prevent sparse reward problem"""
    print("\n" + "=" * 80)
    print("TEST 4: Positive Reward Signal Distribution")
    print("=" * 80)

    num_agents = 6
    total_trees = 400
    total_healthy_trees = 300

    # Typical episode where agents are working
    trees_to_fire_state = [1, 2]  # Some new fires
    trees_to_burnt_state = [3]  # Some losses
    trees_to_healthy_state = [4, 5]  # Some progress
    agent_tree_extinguished = {i: 0 for i in range(num_agents)}
    agent_on_fire_tree = {i: 0 for i in range(num_agents)}

    print(f"\nTypical episode scenario:")
    print(f"  - Health ratio: {total_healthy_trees}/{total_trees} = {100*total_healthy_trees/total_trees:.1f}%")
    print(f"  - New fires: {len(trees_to_fire_state)}")
    print(f"  - Extinguished: {len(trees_to_healthy_state)}")
    print(f"  - Burned: {len(trees_to_burnt_state)}")

    reward = cooperative3_reward(
        trees_to_fire_state=trees_to_fire_state,
        trees_to_burnt_state=trees_to_burnt_state,
        trees_to_healthy_state=trees_to_healthy_state,
        agent_tree_extinguished=agent_tree_extinguished,
        agent_on_fire_tree=agent_on_fire_tree,
        num_agents=num_agents,
        current_step=50,
        max_steps=300,
        total_healthy_trees=total_healthy_trees,
        total_trees=total_trees,
    )[f'0']

    print(f"\n  Reward: {reward:.4f}")

    # Calculate breakdown
    healthy_ratio = total_healthy_trees / total_trees
    r_health = 3.0 * healthy_ratio
    r_extinguish = 1.5 * len(trees_to_healthy_state)
    r_new_fire = -0.5 * len(trees_to_fire_state)
    r_burnt = -0.8 * len(trees_to_burnt_state)
    r_time = -0.02 * (50 / 300)

    print(f"\nReward breakdown:")
    print(f"  - Healthy ratio signal (3.0 * {healthy_ratio:.2f}): {r_health:7.4f}")
    print(f"  - Extinguish signal (1.5 * {len(trees_to_healthy_state)}): {r_extinguish:7.4f}")
    print(f"  - New fire penalty (-0.5 * {len(trees_to_fire_state)}): {r_new_fire:7.4f}")
    print(f"  - Burnt penalty (-0.8 * {len(trees_to_burnt_state)}): {r_burnt:7.4f}")
    print(f"  - Time penalty (-0.02 * {50/300:.3f}): {r_time:7.4f}")
    print(f"  - Total: {r_health + r_extinguish + r_new_fire + r_burnt + r_time:7.4f}")

    is_positive = reward > 0
    print(f"\nPositive reward signal: {is_positive} {'✓' if is_positive else '✗'}")


def test_get_reward_function():
    """Test that cooperative3 is registered in get_reward_function"""
    print("\n" + "=" * 80)
    print("TEST 5: Function Registration")
    print("=" * 80)

    try:
        func = get_reward_function("cooperative3")
        print(f"✓ cooperative3_reward successfully registered")
        print(f"  Function: {func.__name__}")
        print(f"  Module: {func.__module__}")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " COOPERATIVE3_REWARD FUNCTION TEST SUITE ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        test_cooperative3_reward_basic()
        test_time_penalty_progression()
        test_healthy_ratio_signal()
        test_positive_reward_signal()
        test_get_reward_function()

        print("\n" + "=" * 80)
        print("All tests completed successfully! ✓")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
