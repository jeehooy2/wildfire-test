"""Create animated GIF demo of the full system"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

try:
    import gym
except ImportError:
    import gymnasium as gym
    sys.modules['gym'] = gym
    sys.modules['gym.spaces'] = gym.spaces

from wildfire_environment.envs.wildfire import WildfireEnv


def create_demo_gif():
    """Create a demo GIF showing the full system in action"""
    print("Creating demo GIF...")

    # Create environment
    env = WildfireEnv(
        size=17,
        num_agents=2,
        agent_start_positions=((1, 1), (15, 15)),
        agent_colors=("red", "blue"),
        max_steps=200,
        render_mode="rgb_array",
    )

    # Reset environment
    obs, info = env.reset(seed=42)

    # Set short times for demo
    for a in env.agents:
        a.max_active_time = 15
        a.active_time_remaining = 15
        a.recharge_time = 10

    print("Running simulation...")

    # Mission plan with time offsets:
    # Agent 0 (red): goes to (12, 4) first
    # Agent 1 (blue): goes to (4, 12) after a delay
    # After recharge, they get new missions with different timings

    frames = []
    max_steps = 100

    # Track mission states
    agent_0_mission_sent = False
    agent_1_mission_sent = False
    agent_0_recharged = False
    agent_1_recharged = False
    agent_0_mission_2_sent = False
    agent_1_mission_2_sent = False

    # Render initial state before any step
    frames.append(Image.fromarray(env.render()))

    for step in range(max_steps):
        agent_0 = env.agents[0]
        agent_1 = env.agents[1]

        # Step 0: Send Agent 0 (red) to (12, 4)
        if step == 0:
            target_0 = (12, 4)
            action_0 = 1 + (target_0[1] - 1) * env.grid_size_without_walls + (target_0[0] - 1)
            actions = {"0": action_0, "1": 0}  # Agent 1 waits
            print(f"[Step {step}] Agent 0 -> {target_0}")
            agent_0_mission_sent = True

        # Step 7: Send Agent 1 (blue) to (4, 12) with delay
        elif step == 7 and not agent_1_mission_sent:
            target_1 = (4, 12)
            action_1 = 1 + (target_1[1] - 1) * env.grid_size_without_walls + (target_1[0] - 1)
            actions = {"0": 0, "1": action_1}  # Agent 0 continues
            print(f"[Step {step}] Agent 1 -> {target_1}")
            agent_1_mission_sent = True

        # Check if Agent 0 completed recharge and ready for new mission
        elif (agent_0.state == 0 and
              agent_0.active_time_remaining == agent_0.max_active_time and
              np.array_equal(agent_0.pos, agent_0.home_pos) and
              agent_0_mission_sent and not agent_0_recharged):
            print(f"[Step {step}] Agent 0 recharged and ready!")
            agent_0_recharged = True
            # Send Agent 0 to new target (8, 13)
            target_0_new = (8, 13)
            action_0 = 1 + (target_0_new[1] - 1) * env.grid_size_without_walls + (target_0_new[0] - 1)
            actions = {"0": action_0, "1": 0}
            print(f"[Step {step}] Agent 0 -> {target_0_new} (Mission 2)")
            agent_0_mission_2_sent = True

        # Check if Agent 1 completed recharge (with 3 step delay after Agent 0)
        elif (agent_1.state == 0 and
              agent_1.active_time_remaining == agent_1.max_active_time and
              np.array_equal(agent_1.pos, agent_1.home_pos) and
              agent_1_mission_sent and not agent_1_recharged and
              agent_0_mission_2_sent):
            # Wait 3 more steps before sending Agent 1
            if step >= 3:
                print(f"[Step {step}] Agent 1 recharged and ready!")
                agent_1_recharged = True
                # Send Agent 1 to new target (13, 8)
                target_1_new = (13, 8)
                action_1 = 1 + (target_1_new[1] - 1) * env.grid_size_without_walls + (target_1_new[0] - 1)
                actions = {"0": 0, "1": action_1}
                print(f"[Step {step}] Agent 1 -> {target_1_new} (Mission 2)")
                agent_1_mission_2_sent = True
            else:
                actions = {"0": 0, "1": 0}

        else:
            # Default: both agents continue with current targets (WAIT)
            actions = {"0": 0, "1": 0}

        # Execute step
        obs, rewards, terminated, truncated, infos = env.step(actions)

        # Render after step
        frames.append(Image.fromarray(env.render()))

        # Print status every 5 steps
        if step % 5 == 0 or step < 10:
            state_names = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}
            print(f"  Step {step}: A0={agent_0.pos}[{state_names[agent_0.state]}] A1={agent_1.pos}[{state_names[agent_1.state]}]")

        if terminated or truncated:
            print(f"Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
            break

    print(f"Captured {len(frames)} frames")

    # Save as GIF
    print("Saving GIF...")
    frames[0].save(
        'wildfire_demo.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame = 10fps
        loop=0
    )

    print("✓ GIF saved as 'wildfire_demo.gif'")


if __name__ == "__main__":
    create_demo_gif()
