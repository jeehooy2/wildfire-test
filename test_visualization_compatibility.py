#!/usr/bin/env python
"""Test compatibility of visualize_environment.py with updated wildfire_environment"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from wildfire_environment.core.agent import AgentState

print("=" * 80)
print("Visualization Compatibility Test")
print("=" * 80)

# Test 1: Check if WildfireEnv can be created with ENV_CONFIG
print("\n[Test 1] Creating environment with ENV_CONFIG...")
try:
    env = WildfireEnv(**ENV_CONFIG)
    print("✓ Environment created successfully")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    sys.exit(1)

# Test 2: Check if env.reset() works
print("\n[Test 2] Testing env.reset()...")
try:
    obs_dict, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  - Observations: {len(obs_dict)} agents")
    print(f"  - Agent IDs: {list(obs_dict.keys())}")
except Exception as e:
    print(f"✗ Failed to reset environment: {e}")
    sys.exit(1)

# Test 3: Check agent properties
print("\n[Test 3] Checking agent properties...")
try:
    for agent in env.agents:
        print(f"  Agent {agent.index}:")
        print(f"    - Type: {agent.type}")
        print(f"    - State: {agent.state} (0=ACTIVE, 1=RETURNING, 2=RECHARGING)")
        print(f"    - Home: {agent.home_pos}")
        print(f"    - Position: {agent.pos}")
        print(f"    - Max Active Time: {agent.max_active_time}")
        print(f"    - Water: {agent.water_remaining:.1f}/{agent.max_water}")
    print("✓ All agents have supply source properties")
except Exception as e:
    print(f"✗ Failed to check agent properties: {e}")
    sys.exit(1)

# Test 4: Run a few steps with random actions
print("\n[Test 4] Running simulation with random actions...")
try:
    done = False
    step = 0
    max_steps = 50

    while not done and step < max_steps:
        actions = {}
        for agent_id in obs_dict.keys():
            action = env.action_space[agent_id].sample()
            actions[agent_id] = action

        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}:")
            for i, agent in enumerate(env.agents):
                state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[agent.state]
                print(f"    Agent {i}: {state_name}, Water={agent.water_remaining:.1f}")

        step += 1

    print(f"✓ Simulation ran successfully ({step} steps)")
except Exception as e:
    print(f"✗ Failed to run simulation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check rendering
print("\n[Test 5] Testing rendering...")
try:
    frame = env.render(mode='rgb_array')
    if frame is not None:
        print(f"✓ Rendering successful")
        print(f"  - Frame shape: {frame.shape}")
        print(f"  - Frame dtype: {frame.dtype}")
    else:
        print("✗ Rendering returned None")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to render: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check action space
print("\n[Test 6] Checking action space...")
try:
    print(f"✓ Action space information:")
    for agent_id in env.action_space.spaces.keys():
        action_space = env.action_space[agent_id]
        print(f"  - Agent {agent_id}: {action_space}")
except Exception as e:
    print(f"✗ Failed to check action space: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All compatibility tests passed!")
print("=" * 80)
print("\nVisualize_environment.py should work correctly with the updated")
print("wildfire_environment. You can run it with:")
print("\n  python train_marllib_self/visualize_environment.py --episodes 3 --seed 42")
print("=" * 80)
