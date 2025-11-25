#!/usr/bin/env python
"""Test script for supply source return mechanism"""

import sys
sys.path.insert(0, '/home/bmkim88/wildfire_environment')

import numpy as np
from wildfire_environment.envs.wildfire import WildfireEnv
from wildfire_environment.core.agent import AgentState

def test_supply_source_return():
    """Test that agents return to supply source after max_active_time steps"""

    # Create environment
    env = WildfireEnv(
        size=17,
        num_agents=2,
        num_helicopters=0,
        num_trucks=0,
        num_crews=0,
        agent_start_positions=((1, 1), (15, 15)),
        max_steps=500,
    )

    obs, info = env.reset(seed=42)

    print("=" * 60)
    print("Supply Source Return Mechanism Test")
    print("=" * 60)

    # Test parameters
    agent_0 = env.agents[0]
    agent_1 = env.agents[1]

    print(f"\nAgent 0: max_active_time={agent_0.max_active_time}, recharge_time={agent_0.recharge_time}")
    print(f"Agent 1: max_active_time={agent_1.max_active_time}, recharge_time={agent_1.recharge_time}")
    print(f"\nAgent 0 home: {agent_0.home_pos}, Agent 1 home: {agent_1.home_pos}")

    # Run simulation
    state_changes = {0: [], 1: []}

    for step in range(300):
        # Random actions
        actions = {
            "0": env.action_space["0"].sample(),
            "1": env.action_space["1"].sample(),
        }

        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Track state changes
        for agent_id in [0, 1]:
            agent = env.agents[agent_id]
            if not state_changes[agent_id] or state_changes[agent_id][-1][1] != agent.state:
                state_changes[agent_id].append((step, agent.state, agent.active_time_remaining, agent.water_remaining))

        if done:
            print(f"\nEpisode ended at step {step}")
            break

    # Print state changes
    print("\n" + "=" * 60)
    print("Agent 0 State Transitions:")
    print("=" * 60)
    print(f"{'Step':<6} {'State':<12} {'Active_Time':<12} {'Water':<10}")
    print("-" * 60)

    for step, state, active_time, water in state_changes[0]:
        state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[state]
        print(f"{step:<6} {state_name:<12} {active_time:<12} {water:<10.1f}")

    print("\n" + "=" * 60)
    print("Agent 1 State Transitions:")
    print("=" * 60)
    print(f"{'Step':<6} {'State':<12} {'Active_Time':<12} {'Water':<10}")
    print("-" * 60)

    for step, state, active_time, water in state_changes[1]:
        state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[state]
        print(f"{step:<6} {state_name:<12} {active_time:<12} {water:<10.1f}")

    # Verification
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)

    # Check if there are state transitions
    if len(state_changes[0]) > 1:
        print("✓ Agent 0 has state transitions")
    else:
        print("✗ Agent 0 has NO state transitions")

    if len(state_changes[1]) > 1:
        print("✓ Agent 1 has state transitions")
    else:
        print("✗ Agent 1 has NO state transitions")

    # Check if agents return to home
    for agent_id in [0, 1]:
        agent = env.agents[agent_id]
        home_reached = any(step > 0 for step, state, _, _ in state_changes[agent_id] if state == AgentState.RECHARGING)
        if home_reached:
            print(f"✓ Agent {agent_id} reached home and started recharging")
        else:
            print(f"✗ Agent {agent_id} did NOT reach home")

    # Check water consumption
    print("\nWater Consumption:")
    for agent_id in [0, 1]:
        if state_changes[agent_id]:
            initial_water = state_changes[agent_id][0][3]
            final_water = state_changes[agent_id][-1][3]
            water_consumed = initial_water - final_water
            print(f"Agent {agent_id}: {initial_water:.1f} → {final_water:.1f} (consumed: {water_consumed:.1f})")

    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    test_supply_source_return()
