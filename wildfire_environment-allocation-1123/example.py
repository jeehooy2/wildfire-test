"""
wildfire-environment 기본 사용 예제

이 스크립트는 wildfire 환경을 생성하고 무작위 액션으로 시뮬레이션을 실행합니다.
"""

import gym
import wildfire_environment

# 환경 생성
env = gym.make("wildfire-v0", 
    num_agents=2,
    size=17,
    initial_fire_size=3,
    cooperative_reward=False,
    log_selfish_region_metrics=True,
    selfish_region_xmin=[7, 13],
    selfish_region_xmax=[9, 15],
    selfish_region_ymin=[7, 1],
    selfish_region_ymax=[9, 3],
)

print("=" * 60)
print("Wildfire Environment 시뮬레이션 시작")
print("=" * 60)
print(f"에이전트 수: {env.num_agents}")
print(f"그리드 크기: {env.width} x {env.height}")
print(f"최대 스텝 수: {env.max_steps}")
print("=" * 60)

# 환경 초기화
observation, info = env.reset(seed=42)

episode_count = 0
total_steps = 0

# 시뮬레이션 실행
for step in range(1000):
    # 무작위 액션 샘플링
    action = env.action_space.sample()
    
    # 환경 스텝 실행
    observation, reward, done, info = env.step(action)
    total_steps += 1
    
    # 에피소드가 끝나면 재시작
    if done:
        episode_count += 1
        print(f"에피소드 {episode_count} 완료 (총 스텝: {total_steps})")
        observation, info = env.reset()

print("=" * 60)
print(f"시뮬레이션 완료!")
print(f"총 에피소드 수: {episode_count}")
print(f"총 스텝 수: {total_steps}")
print("=" * 60)

env.close()

