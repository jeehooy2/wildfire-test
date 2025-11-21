#!/usr/bin/env python3
"""
Test script to verify new_evaluation.py works correctly with partial_obs=True

이 스크립트는:
  1. partial_obs=True로 환경 생성
  2. new_evaluation.py의 run_evaluation_episode 함수를 사용
  3. 관찰이 올바르게 처리되는지 확인
"""

import sys
import os
from pathlib import Path

# Set up project root
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from train_marllib_self.new_evaluation import (
    run_evaluation_episode,
    get_agent_policy_name,
)


def test_evaluation_with_partial_obs():
    """Test evaluation episode with partial_obs=True"""
    print("=" * 80)
    print("Testing evaluation episode with partial_obs=True")
    print("=" * 80)

    # Environment with partial_obs=True
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_config['partial_obs'] = True
    env_config['agent_view_size'] = 10
    env_config['max_steps'] = 100

    print(f"\nEnvironment config:")
    print(f"  partial_obs: {env_config['partial_obs']}")
    print(f"  agent_view_size: {env_config['agent_view_size']}")
    print(f"  max_steps: {env_config['max_steps']}")

    try:
        env = WildfireEnv(**env_config)
        print(f"\n✓ Environment created successfully")

        # Get agent info
        num_helicopters = sum(1 for agent in env.agents if agent.type == "helicopter")
        num_trucks = sum(1 for agent in env.agents if agent.type == "truck")
        num_crews = sum(1 for agent in env.agents if agent.type == "crew")

        print(f"\nAgent types:")
        print(f"  Helicopters: {num_helicopters}")
        print(f"  Trucks: {num_trucks}")
        print(f"  Crews: {num_crews}")

        # Create policy mapping function
        def create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews):
            def mapping_fn(agent_id):
                try:
                    agent_idx = int(agent_id)
                    if 0 <= agent_idx < len(env.agents):
                        agent_obj = env.agents[agent_idx]
                        return get_agent_policy_name(agent_obj, num_helicopters, num_trucks, num_crews)
                except (ValueError, TypeError):
                    pass
                return 'helicopter_policy'
            return mapping_fn

        policy_mapping_fn = create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews)

        # Run evaluation episode WITHOUT trained policies (random actions)
        print(f"\nRunning evaluation episode with random policy...")
        metrics = run_evaluation_episode(
            env, None, policy_mapping_fn, seed=42, max_steps=env_config['max_steps']
        )

        print(f"\n✓ Evaluation episode completed successfully")
        print(f"\nMetrics:")
        print(f"  Healthy trees: {metrics['healthy_trees_ratio']*100:.2f}%")
        print(f"  Burnt trees: {metrics['burnt_trees_ratio']*100:.2f}%")
        print(f"  Episode length: {metrics['episode_length']} steps")
        print(f"  Healthy counts tracked: {len(metrics['healthy_counts'])} steps")
        print(f"  Burnt counts tracked: {len(metrics['burnt_counts'])} steps")

        # Verify metrics are valid
        if not (0 <= metrics['healthy_trees_ratio'] <= 1):
            print(f"\n✗ Invalid healthy_trees_ratio: {metrics['healthy_trees_ratio']}")
            return False

        if not (0 <= metrics['burnt_trees_ratio'] <= 1):
            print(f"\n✗ Invalid burnt_trees_ratio: {metrics['burnt_trees_ratio']}")
            return False

        if metrics['episode_length'] <= 0:
            print(f"\n✗ Invalid episode_length: {metrics['episode_length']}")
            return False

        if len(metrics['healthy_counts']) == 0 or len(metrics['burnt_counts']) == 0:
            print(f"\n✗ No tree counts tracked")
            return False

        print(f"\n✓ All metrics are valid")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_partial_vs_full_obs():
    """Compare evaluation with partial_obs vs full_obs"""
    print("\n" + "=" * 80)
    print("Comparing evaluation with partial_obs vs full_obs")
    print("=" * 80)

    try:
        # Full observation
        env_config_full = {k: v for k, v in ENV_CONFIG.items()}
        env_config_full['partial_obs'] = False
        env_config_full['max_steps'] = 50

        # Partial observation
        env_config_partial = {k: v for k, v in ENV_CONFIG.items()}
        env_config_partial['partial_obs'] = True
        env_config_partial['agent_view_size'] = 10
        env_config_partial['max_steps'] = 50

        print(f"\nRunning evaluation with full_obs...")
        env_full = WildfireEnv(**env_config_full)
        num_helicopters = sum(1 for agent in env_full.agents if agent.type == "helicopter")
        num_trucks = sum(1 for agent in env_full.agents if agent.type == "truck")
        num_crews = sum(1 for agent in env_full.agents if agent.type == "crew")

        def create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews):
            def mapping_fn(agent_id):
                try:
                    agent_idx = int(agent_id)
                    if 0 <= agent_idx < len(env.agents):
                        agent_obj = env.agents[agent_idx]
                        return get_agent_policy_name(agent_obj, num_helicopters, num_trucks, num_crews)
                except (ValueError, TypeError):
                    pass
                return 'helicopter_policy'
            return mapping_fn

        policy_mapping_fn_full = create_policy_mapping_fn(env_full, num_helicopters, num_trucks, num_crews)
        metrics_full = run_evaluation_episode(
            env_full, None, policy_mapping_fn_full, seed=42, max_steps=50
        )

        print(f"  ✓ Full obs - Length: {metrics_full['episode_length']} steps")

        print(f"\nRunning evaluation with partial_obs...")
        env_partial = WildfireEnv(**env_config_partial)
        policy_mapping_fn_partial = create_policy_mapping_fn(env_partial, num_helicopters, num_trucks, num_crews)
        metrics_partial = run_evaluation_episode(
            env_partial, None, policy_mapping_fn_partial, seed=42, max_steps=50
        )

        print(f"  ✓ Partial obs - Length: {metrics_partial['episode_length']} steps")

        # Compare results
        print(f"\nComparison:")
        print(f"  Full obs healthy: {metrics_full['healthy_trees_ratio']*100:.2f}%")
        print(f"  Partial obs healthy: {metrics_partial['healthy_trees_ratio']*100:.2f}%")
        print(f"  Difference: {abs(metrics_full['healthy_trees_ratio'] - metrics_partial['healthy_trees_ratio'])*100:.2f}%")

        print(f"\n✓ Both full_obs and partial_obs work correctly")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_evaluation_episodes():
    """Test multiple evaluation episodes with partial_obs"""
    print("\n" + "=" * 80)
    print("Testing multiple evaluation episodes with partial_obs")
    print("=" * 80)

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_config['partial_obs'] = True
    env_config['agent_view_size'] = 10
    env_config['max_steps'] = 50

    try:
        env = WildfireEnv(**env_config)
        num_helicopters = sum(1 for agent in env.agents if agent.type == "helicopter")
        num_trucks = sum(1 for agent in env.agents if agent.type == "truck")
        num_crews = sum(1 for agent in env.agents if agent.type == "crew")

        def create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews):
            def mapping_fn(agent_id):
                try:
                    agent_idx = int(agent_id)
                    if 0 <= agent_idx < len(env.agents):
                        agent_obj = env.agents[agent_idx]
                        return get_agent_policy_name(agent_obj, num_helicopters, num_trucks, num_crews)
                except (ValueError, TypeError):
                    pass
                return 'helicopter_policy'
            return mapping_fn

        policy_mapping_fn = create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews)

        print(f"\nRunning 5 evaluation episodes...")
        metrics_list = []

        for ep in range(5):
            metrics = run_evaluation_episode(
                env, None, policy_mapping_fn, seed=42 + ep, max_steps=50
            )
            metrics_list.append(metrics)
            print(f"  Episode {ep+1}: Healthy={metrics['healthy_trees_ratio']*100:.1f}%, Burnt={metrics['burnt_trees_ratio']*100:.1f}%, Steps={metrics['episode_length']}")

        # Compute summary
        healthy_ratios = [m['healthy_trees_ratio'] for m in metrics_list]
        burnt_ratios = [m['burnt_trees_ratio'] for m in metrics_list]

        print(f"\nSummary:")
        print(f"  Healthy: {np.mean(healthy_ratios)*100:.2f}% ± {np.std(healthy_ratios)*100:.2f}%")
        print(f"  Burnt: {np.mean(burnt_ratios)*100:.2f}% ± {np.std(burnt_ratios)*100:.2f}%")

        print(f"\n✓ Multiple episodes completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("Testing new_evaluation.py with partial_obs=True")
    print("=" * 80 + "\n")

    results = []

    # Run all tests
    results.append(("Basic evaluation", test_evaluation_with_partial_obs()))
    results.append(("Partial vs Full", test_evaluation_partial_vs_full_obs()))
    results.append(("Multiple episodes", test_multiple_evaluation_episodes()))

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
        print("\n결론: new_evaluation.py에서 partial_obs=True가 제대로 작동합니다.")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
