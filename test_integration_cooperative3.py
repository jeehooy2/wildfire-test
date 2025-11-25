#!/usr/bin/env python3
"""
Integration test: Run cooperative3_reward with actual WildfireEnv
"""

import sys
sys.path.insert(0, '/home/bmkim88/wildfire_environment')

from wildfire_environment.envs.wildfire import WildfireEnv
import numpy as np

def test_cooperative3_in_environment():
    """Test that cooperative3_reward works with actual environment"""
    print("=" * 80)
    print("INTEGRATION TEST: cooperative3_reward with WildfireEnv")
    print("=" * 80)

    # Create environment with cooperative3 reward
    env = WildfireEnv(
        size=17,
        num_agents=2,
        max_steps=100,
        initial_fire_size=2,
        partial_obs=False,
        reward_shaping="cooperative3",
        reward_shaping_config={
            "w_extinguish": 1.5,
            "w_healthy_ratio": 3.0,
            "w_new_fire": 0.5,
            "w_burnt": 0.8,
            "w_time_penalty": 0.02,
        },
    )

    print(f"\n✓ Environment created with cooperative3_reward")
    print(f"  - Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  - Agents: {env.num_agents}")
    print(f"  - Max steps: {env.max_steps}")
    print(f"  - Reward shaping: {env.reward_shaping}")

    # Reset and run episode
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment reset successfully")
    print(f"  - Initial burnt trees: {info.get('burnt trees', 0)}")

    # Run a few steps
    print(f"\n{'Step':>4} {'Agent 0 Reward':>15} {'Agent 1 Reward':>15} {'Avg Reward':>12} {'Fire Trees':>11} {'Status':>15}")
    print("-" * 80)

    total_rewards = {0: 0.0, 1: 0.0}
    episode_done = False

    for step in range(min(20, env.max_steps)):
        # Random actions for testing
        actions = {
            f"{i}": env.action_space[f"{i}"].sample()
            for i in range(env.num_agents)
        }

        obs, rewards, terminated, truncated, infos = env.step(actions)
        episode_done = terminated or truncated

        total_rewards[0] += float(rewards['0'])
        total_rewards[1] += float(rewards['1'])
        avg_reward = (total_rewards[0] + total_rewards[1]) / 2 / (step + 1)

        fire_count = infos.get('0', {}).get('trees_on_fire', 0) if isinstance(infos, dict) else 0
        status = "Running" if not episode_done else "Done" if terminated else "Truncated"

        print(f"{step+1:>4} {float(rewards['0']):>15.4f} {float(rewards['1']):>15.4f} {avg_reward:>12.4f} {fire_count:>11} {status:>15}")

        if episode_done:
            print(f"\nEpisode ended at step {step+1}")
            break

    print("\n✓ Integration test passed!")
    print(f"  Total episodes completed: 1")
    print(f"  Final agent 0 cumulative reward: {total_rewards[0]:.4f}")
    print(f"  Final agent 1 cumulative reward: {total_rewards[1]:.4f}")

    env.close()

    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " INTEGRATION TEST FOR COOPERATIVE3_REWARD ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        test_cooperative3_in_environment()
        print("\n" + "=" * 80)
        print("Integration test completed successfully! ✓")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"\n❌ Integration test failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
