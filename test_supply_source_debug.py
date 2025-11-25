#!/usr/bin/env python
"""Debug test script for supply source return mechanism"""

import sys
sys.path.insert(0, '/home/bmkim88/wildfire_environment')

import numpy as np
from wildfire_environment.envs.wildfire import WildfireEnv
from wildfire_environment.core.agent import AgentState

def test_supply_source_debug():
    """Debug supply source mechanism"""

    # Create environment
    env = WildfireEnv(
        size=17,
        num_agents=2,
        num_helicopters=0,
        num_trucks=0,
        num_crews=0,
        agent_start_positions=((1, 1), (15, 15)),
        max_steps=500,
        initial_fire_size=3,
    )

    obs, info = env.reset(seed=42)

    print("=" * 60)
    print("Supply Source Return Mechanism Debug Test")
    print("=" * 60)

    agent_0 = env.agents[0]
    agent_1 = env.agents[1]

    print(f"\nAgent 0:")
    print(f"  Position: {agent_0.pos}")
    print(f"  Home: {agent_0.home_pos}")
    print(f"  State: {agent_0.state} (0=ACTIVE, 1=RETURNING, 2=RECHARGING)")
    print(f"  Max Active Time: {agent_0.max_active_time}")
    print(f"  Active Time Remaining: {agent_0.active_time_remaining}")
    print(f"  Water: {agent_0.water_remaining}/{agent_0.max_water}")

    print(f"\nAgent 1:")
    print(f"  Position: {agent_1.pos}")
    print(f"  Home: {agent_1.home_pos}")
    print(f"  State: {agent_1.state}")
    print(f"  Max Active Time: {agent_1.max_active_time}")
    print(f"  Active Time Remaining: {agent_1.active_time_remaining}")
    print(f"  Water: {agent_1.water_remaining}/{agent_1.max_water}")

    # Run for specific number of steps and print status
    print("\n" + "=" * 60)
    print("Simulation Progress (every 20 steps):")
    print("=" * 60)

    for step in range(300):
        # Random actions
        actions = {
            "0": env.action_space["0"].sample(),
            "1": env.action_space["1"].sample(),
        }

        obs, reward, terminated, truncated, info = env.step(actions)

        if (step + 1) % 30 == 0 or step < 5:
            print(f"\nStep {step + 1}:")
            print(f"  Agent 0: State={agent_0.state}, Pos={agent_0.pos}, Home={agent_0.home_pos}, Active_Time={agent_0.active_time_remaining}, Water={agent_0.water_remaining:.1f}")
            print(f"  Agent 1: State={agent_1.state}, Pos={agent_1.pos}, Home={agent_1.home_pos}, Active_Time={agent_1.active_time_remaining}, Water={agent_1.water_remaining:.1f}")

        done = terminated or truncated
        if done:
            print(f"\nEpisode ended at step {step + 1}")
            break

    print("\n" + "=" * 60)
    print("Final Status:")
    print("=" * 60)

    print(f"\nAgent 0:")
    print(f"  State: {agent_0.state} (0=ACTIVE, 1=RETURNING, 2=RECHARGING)")
    print(f"  Position: {agent_0.pos} (home: {agent_0.home_pos})")
    print(f"  Active Time: {agent_0.active_time_remaining}/{agent_0.max_active_time}")
    print(f"  Water: {agent_0.water_remaining:.1f}/{agent_0.max_water}")

    print(f"\nAgent 1:")
    print(f"  State: {agent_1.state}")
    print(f"  Position: {agent_1.pos} (home: {agent_1.home_pos})")
    print(f"  Active Time: {agent_1.active_time_remaining}/{agent_1.max_active_time}")
    print(f"  Water: {agent_1.water_remaining:.1f}/{agent_1.max_water}")

if __name__ == "__main__":
    test_supply_source_debug()
