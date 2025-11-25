"""Test script for water/suppressant system with multiple waypoints"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

# Mock gym with gymnasium for compatibility
try:
    import gym
except ImportError:
    import gymnasium as gym
    sys.modules['gym'] = gym
    sys.modules['gym.spaces'] = gym.spaces

from wildfire_environment.envs.wildfire import WildfireEnv


def test_water_system_multi_waypoint():
    """Test agents visiting multiple fire locations before returning to base"""

    # Create environment with 2 agents
    env = WildfireEnv(
        size=17,
        num_agents=2,
        agent_start_positions=((1, 1), (1, 1)),
        agent_colors=("red", "blue"),
        max_steps=500,
        render_mode="rgb_array",
        alpha=0.05,  # Fire spread rate
        beta=0.95,   # Fire persistence (lower = easier to extinguish)
        delta_beta=0.3,  # Agent suppression effectiveness
    )

    # Reset environment
    obs, info = env.reset()

    # Define multiple waypoints for each agent
    # Agent 0 (red): visits 3 fire locations before returning
    waypoints_agent_0 = [
        (8, 4),   # First fire location (top middle)
        (13, 8),  # Second fire location (right middle) - changed to avoid collision
        (8, 13),  # Third fire location (bottom middle) - changed to avoid collision
    ]

    # Agent 1 (blue): visits 3 different fire locations (non-overlapping paths)
    waypoints_agent_1 = [
        (4, 8),   # First fire location (left middle)
        (8, 2),   # Second fire location (top middle) - changed to avoid collision
        (13, 13), # Third fire location (bottom right) - changed to avoid collision
    ]

    current_waypoint = {0: 0, 1: 0}  # Track current waypoint index for each agent
    frames = []

    print("=" * 70)
    print("Testing Water/Suppressant System with Multiple Waypoints")
    print("=" * 70)
    print(f"Agent 0 (red) starts at {env.agents[0].pos}")
    print(f"Agent 1 (blue) starts at {env.agents[1].pos}")
    print(f"\nAgent 0 max_active_time: {env.agents[0].max_active_time}")
    print(f"Agent 0 max_water: {env.agents[0].max_water}")
    print(f"Agent 0 water_consumption_rate: {env.agents[0].water_consumption_rate}")
    print("=" * 70)

    for step in range(1000):
        actions = {}

        for agent_idx, agent in enumerate(env.agents):
            # Check if agent needs a new waypoint
            if agent.state == 0:  # ACTIVE
                # Check if agent reached current waypoint (convert to arrays for proper comparison)
                agent_pos_array = np.array(agent.pos)
                reached_target = False

                if agent.target_pos is not None:
                    target_pos_array = np.array(agent.target_pos)
                    reached_target = np.array_equal(agent_pos_array, target_pos_array)

                if agent.target_pos is None or reached_target:
                    waypoints = waypoints_agent_0 if agent_idx == 0 else waypoints_agent_1

                    if current_waypoint[agent_idx] < len(waypoints):
                        # Set next waypoint
                        target = waypoints[current_waypoint[agent_idx]]
                        action_idx = (target[1] - 1) * 15 + (target[0] - 1)
                        actions[str(agent_idx)] = action_idx + 1
                        current_waypoint[agent_idx] += 1
                        print(f"\nStep {step}: Agent {agent_idx} heading to waypoint {current_waypoint[agent_idx]}: {target}")
                    else:
                        # All waypoints visited, wait
                        actions[str(agent_idx)] = 0
                else:
                    # Continue moving to current target
                    actions[str(agent_idx)] = 0
            else:
                # Agent is RETURNING or RECHARGING
                actions[str(agent_idx)] = 0

        # Step environment
        obs, rewards, terminated, truncated, infos = env.step(actions)

        # Render and save frame
        img = env.render()
        frames.append(img)

        # Print status every 10 steps
        if step % 10 == 0:
            print(f"\n--- Step {step} ---")
            for agent in env.agents:
                tree_at_pos = env.helper_grid.get(*agent.pos)
                on_fire = tree_at_pos.type == "tree" and tree_at_pos.state == 1
                state_names = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}
                print(f"Agent {agent.index}: Pos={agent.pos}, Target={agent.target_pos}, State={state_names[agent.state]}, "
                      f"Water={agent.water_remaining:.1f}/{agent.max_water:.1f}, "
                      f"Active={agent.active_time_remaining}/{agent.max_active_time}, "
                      f"OnFire={on_fire}")

        # Check for mission completion or episode end
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            break

    # Save key frames as PNG
    print("\n" + "=" * 70)
    print("Saving test results...")
    print("=" * 70)

    # Save frames at key points
    key_frames = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    combined_width = sum(frames[i].shape[1] for i in key_frames)
    combined_height = max(frames[i].shape[0] for i in key_frames)

    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    x_offset = 0

    for idx, frame_idx in enumerate(key_frames):
        frame = frames[frame_idx]
        h, w = frame.shape[:2]
        combined_img[:h, x_offset:x_offset+w, :] = frame
        x_offset += w

    result_img = Image.fromarray(combined_img)
    result_img.save("test_water_system_multi_waypoint.png")
    print(f"✓ Saved test_water_system_multi_waypoint.png")

    # Create GIF
    print("\nCreating GIF animation...")
    gif_frames = [Image.fromarray(f) for f in frames[::2]]  # Every other frame
    gif_frames[0].save(
        "test_water_system_multi_waypoint.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=300,  # 100ms per frame
        loop=0
    )
    print(f"✓ Saved test_water_system_multi_waypoint.gif ({len(gif_frames)} frames)")

    # Final statistics
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)
    for agent in env.agents:
        print(f"Agent {agent.index}:")
        print(f"  - Final position: {agent.pos}")
        print(f"  - Water remaining: {agent.water_remaining:.1f}/{agent.max_water:.1f}")
        print(f"  - Active time remaining: {agent.active_time_remaining}/{agent.max_active_time}")
        print(f"  - State: {['ACTIVE', 'RETURNING', 'RECHARGING'][agent.state]}")

    print(f"\nBurnt trees: {env.burnt_trees}")
    print(f"Trees on fire: {env.trees_on_fire}")
    print("=" * 70)
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_water_system_multi_waypoint()
