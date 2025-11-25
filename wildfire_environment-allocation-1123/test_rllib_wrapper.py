"""
RLlib wrapper 테스트 스크립트

WildfireRLlibEnv가 보상을 제대로 전달하는지 확인
"""

from wildfire_rllib_wrapper import WildfireRLlibEnv

def test_rllib_wrapper():
    """RLlib wrapper 테스트"""
    print("=" * 60)
    print("RLlib Wrapper 테스트")
    print("=" * 60)

    # 환경 설정
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
        "use_action_mask": False,
    }

    env = WildfireRLlibEnv(env_config)

    print(f"\n환경 정보:")
    print(f"  에이전트 수: {env.num_agents}")
    print(f"  액션 스페이스: {env.action_space}")
    print(f"  관찰 스페이스 키: {list(env.observation_space.keys())}")

    # 에피소드 실행
    num_episodes = 3

    for episode in range(num_episodes):
        obs, infos = env.reset()
        print(f"\n{'='*60}")
        print(f"에피소드 {episode + 1}/{num_episodes}")
        print(f"  초기 관찰 키: {list(obs.keys())}")
        print(f"  초기 infos 키: {list(infos.keys())}")

        episode_rewards = {i: 0.0 for i in range(env.num_agents)}
        step_count = 0
        done = False

        while not done and step_count < 20:  # 처음 20 스텝만
            # 랜덤 액션
            actions = env.action_space_sample()

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            step_count += 1

            # 보상 누적
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward

            # 상세 정보 출력 (처음 5 스텝만)
            if step_count <= 5:
                print(f"\n  Step {step_count}:")
                print(f"    Rewards: {rewards}")
                print(f"    Terminated: {terminateds}")
                print(f"    Truncated: {truncateds}")
                for agent_id in range(env.num_agents):
                    if agent_id in infos:
                        print(f"    Agent {agent_id} info: {infos[agent_id]}")

            # 종료 확인
            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        print(f"\n  에피소드 종료:")
        print(f"    총 스텝: {step_count}")
        print(f"    누적 보상:")
        for agent_id, total_reward in episode_rewards.items():
            print(f"      Agent {agent_id}: {total_reward:.3f}")

    env.close()
    print(f"\n{'='*60}")
    print("테스트 완료 - Wrapper는 정상 작동")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_rllib_wrapper()
