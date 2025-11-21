#!/usr/bin/env python3
"""
Test script to verify partial_obs=True works correctly in new_evaluation.py
테스트하는 항목:
  1. partial_obs=True로 환경 생성
  2. 관찰 공간 크기 확인
  3. 에피소드 실행 및 관찰 수집
  4. 관찰 형태 및 값 검증
"""

import sys
import os
from pathlib import Path

# Set up project root
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv


def test_partial_obs_basic():
    """Test basic partial_obs environment creation"""
    print("=" * 80)
    print("TEST 1: Basic partial_obs environment creation")
    print("=" * 80)

    # Create environment with partial_obs=True
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_config['partial_obs'] = True
    env_config['agent_view_size'] = 10

    try:
        env = WildfireEnv(**env_config)
        print("✓ Environment created successfully with partial_obs=True")

        # Check observation space
        print(f"\nObservation Space:")
        for agent_id, obs_space in env.observation_space.items():
            print(f"  Agent {agent_id}: {obs_space}")

        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_partial_obs_reset_and_obs():
    """Test environment reset and observation collection with partial_obs"""
    print("\n" + "=" * 80)
    print("TEST 2: Environment reset and observation collection")
    print("=" * 80)

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_config['partial_obs'] = True
    env_config['agent_view_size'] = 10

    try:
        env = WildfireEnv(**env_config)

        # Reset environment
        obs_dict, info = env.reset(seed=42)
        print("✓ Environment reset successfully")

        # Check observations
        print(f"\nObservations from reset:")
        for agent_id, obs in obs_dict.items():
            print(f"  Agent {agent_id}:")
            print(f"    Shape: {obs.shape}")
            print(f"    dtype: {obs.dtype}")
            print(f"    Min: {np.min(obs):.4f}, Max: {np.max(obs):.4f}")
            print(f"    Contains NaN: {np.isnan(obs).any()}")
            print(f"    Contains Inf: {np.isinf(obs).any()}")

            # Verify observation matches observation space
            obs_space = env.observation_space[agent_id]
            if obs.shape != obs_space.shape:
                print(f"    ✗ Shape mismatch! Expected {obs_space.shape}, got {obs.shape}")
                return False

            if not obs_space.contains(obs):
                print(f"    ✗ Observation not in observation space!")
                print(f"      Space: low={obs_space.low.shape}, high={obs_space.high.shape}")
                return False

        print("\n✓ All observations valid")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_partial_obs_step():
    """Test environment step with partial_obs"""
    print("\n" + "=" * 80)
    print("TEST 3: Environment step and action execution")
    print("=" * 80)

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_config['partial_obs'] = True
    env_config['agent_view_size'] = 10
    env_config['max_steps'] = 50

    try:
        env = WildfireEnv(**env_config)
        obs_dict, _ = env.reset(seed=42)

        print(f"Initial observations: {len(obs_dict)} agents")

        # Step environment multiple times
        for step in range(5):
            actions = {}
            for agent_id in obs_dict.keys():
                actions[agent_id] = env.action_space[agent_id].sample()

            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            # Validate observations
            for agent_id, obs in obs_dict.items():
                obs_space = env.observation_space[agent_id]

                if not obs_space.contains(obs):
                    print(f"✗ Step {step}, Agent {agent_id}: Observation out of bounds!")
                    return False

                if obs.shape != obs_space.shape:
                    print(f"✗ Step {step}, Agent {agent_id}: Shape mismatch!")
                    return False

                if np.isnan(obs).any() or np.isinf(obs).any():
                    print(f"✗ Step {step}, Agent {agent_id}: Invalid values (NaN/Inf)!")
                    return False

            print(f"  Step {step + 1}: ✓ All observations valid")

        print("\n✓ Environment step executed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_partial_obs_full_episode():
    """Test full episode with partial_obs"""
    print("\n" + "=" * 80)
    print("TEST 4: Full episode execution")
    print("=" * 80)

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_config['partial_obs'] = True
    env_config['agent_view_size'] = 10
    env_config['max_steps'] = 100

    try:
        env = WildfireEnv(**env_config)
        obs_dict, _ = env.reset(seed=42)

        step = 0
        done = False

        while not done and step < env_config['max_steps']:
            actions = {}
            for agent_id in obs_dict.keys():
                actions[agent_id] = env.action_space[agent_id].sample()

            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            step += 1

            # Validate every 10 steps
            if step % 10 == 0:
                valid = True
                for agent_id, obs in obs_dict.items():
                    if not env.observation_space[agent_id].contains(obs):
                        valid = False
                        break

                if valid:
                    print(f"  Step {step}: ✓ Valid")
                else:
                    print(f"  Step {step}: ✗ Invalid observation")
                    return False

        print(f"\n✓ Full episode completed ({step} steps)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_partial_obs_vs_full_obs():
    """Compare partial_obs vs full_obs observation sizes"""
    print("\n" + "=" * 80)
    print("TEST 5: Compare partial_obs vs full_obs")
    print("=" * 80)

    try:
        # Full observation
        env_config_full = {k: v for k, v in ENV_CONFIG.items()}
        env_config_full['partial_obs'] = False
        env_full = WildfireEnv(**env_config_full)

        # Partial observation
        env_config_partial = {k: v for k, v in ENV_CONFIG.items()}
        env_config_partial['partial_obs'] = True
        env_config_partial['agent_view_size'] = 10
        env_partial = WildfireEnv(**env_config_partial)

        obs_full, _ = env_full.reset(seed=42)
        obs_partial, _ = env_partial.reset(seed=42)

        first_agent = list(obs_full.keys())[0]

        print(f"Grid size: {env_full.grid_size}")
        print(f"Grid size without walls: {env_full.grid_size_without_walls}")
        print(f"obs_depth (num_agents + tree states): {env_full.obs_depth}")
        print(f"Agents: {env_full.num_agents}")

        full_size = obs_full[first_agent].shape[0]
        partial_size = obs_partial[first_agent].shape[0]

        print(f"\nFull observation size: {full_size}")
        print(f"Partial observation size (view_size=10): {partial_size}")
        print(f"Reduction: {100 * (1 - partial_size / full_size):.1f}%")

        if partial_size < full_size:
            print(f"\n✓ Partial obs is smaller than full obs (as expected)")
            return True
        else:
            print(f"\n✗ Partial obs is NOT smaller than full obs!")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("Testing partial_obs=True functionality")
    print("=" * 80 + "\n")

    results = []

    # Run all tests
    results.append(("Basic creation", test_partial_obs_basic()))
    results.append(("Reset & observation", test_partial_obs_reset_and_obs()))
    results.append(("Step execution", test_partial_obs_step()))
    results.append(("Full episode", test_partial_obs_full_episode()))
    results.append(("Partial vs Full", test_partial_obs_vs_full_obs()))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
