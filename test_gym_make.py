"""
간단한 gym.make 테스트
"""
import gym
import wildfire_environment

print("=== Gym.make 직접 테스트 ===")

config = {
    "size": 22,
    "num_agents": 4,
    "max_steps": 300,
    "initial_fire_size": 3,
    "num_helicopters": 2,
    "num_trucks": 2,
    "num_crews": 0,
    "agent_start_positions": ((1, 1), (1, 20), (20, 1), (20, 20)),
    "alpha": 0.05,
    "beta": 0.9,
    "delta_beta": 0.54,
    "partial_obs": False,
    "agent_view_size": 10,
    "cooperative_reward": False,
    "selfishness_weight": 0.2,
    "reward_shaping": "individual",
    "reward_shaping_config": None,
    "render_mode": "rgb_array",
}

print(f"Config: {config}")
print("\n gym.make 호출 시작...")

try:
    env = gym.make("wildfire-v0", **config)
    print("✅ gym.make 성공!")
    print(f"Environment: {env}")
    print(f"Num agents: {env.num_agents}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
