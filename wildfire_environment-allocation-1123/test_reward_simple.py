"""
보상 시스템 테스트 스크립트

wildfire 환경에서 보상이 제대로 발생하는지 확인
"""

import gym
import wildfire_environment
import numpy as np

def test_basic_reward():
    """기본적인 보상 발생 테스트"""
    print("=" * 60)
    print("Wildfire 환경 보상 테스트")
    print("=" * 60)

    # 환경 생성
    env_config = {
        "num_agents": 2,
        "size": 17,
        "initial_fire_size": 3,
        "cooperative_reward": False,
        "max_steps": 50,
        "agent_start_positions": ((8, 8), (9, 9)),
        "delta_beta": 0.7,
        "beta": 0.99,
        "alpha": 0.05,
        "use_action_mask": False,  # 일단 마스킹 없이 테스트
    }

    env = gym.make("wildfire-v0", **env_config)

    print(f"\n환경 설정:")
    print(f"  에이전트 수: {env.num_agents}")
    print(f"  그리드 크기: {env.grid_size}")
    print(f"  최대 스텝: {env.max_steps}")
    print(f"  delta_beta: {env.delta_beta}")

    # 에피소드 실행
    num_episodes = 5

    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\n{'='*60}")
        print(f"에피소드 {episode + 1}/{num_episodes}")
        print(f"  초기 불 타는 나무: {env.trees_on_fire}")
        print(f"  초기 탄 나무: {env.burnt_trees}")

        episode_rewards = {str(i): 0.0 for i in range(env.num_agents)}
        step_count = 0
        done = False

        while not done and step_count < env.max_steps:
            # 랜덤 액션 수행
            actions = {str(i): env.action_space[str(i)].sample() for i in range(env.num_agents)}

            obs, rewards, done, infos = env.step(actions)
            step_count += 1

            # 보상 누적
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward

            # 상세 정보 출력 (처음 10 스텝만)
            if step_count <= 10:
                print(f"\n  Step {step_count}:")
                print(f"    불 타는 나무: {env.trees_on_fire}")
                print(f"    탄 나무: {env.burnt_trees}")
                for agent_id in range(env.num_agents):
                    agent_id_str = str(agent_id)
                    print(f"    Agent {agent_id}: reward={rewards[agent_id_str]:.3f}, "
                          f"info={infos.get(agent_id_str, {})}")

        print(f"\n  에피소드 종료:")
        print(f"    총 스텝: {step_count}")
        print(f"    최종 불 타는 나무: {env.trees_on_fire}")
        print(f"    최종 탄 나무: {env.burnt_trees}")
        print(f"    누적 보상:")
        for agent_id, total_reward in episode_rewards.items():
            print(f"      Agent {agent_id}: {total_reward:.3f}")

    env.close()
    print(f"\n{'='*60}")
    print("테스트 완료")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_basic_reward()
