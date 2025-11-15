"""
wrapper.py에 대한 상세 테스트 스크립트
"""

import sys
import numpy as np
from wrapper import WildfireMultiAgentWrapper, WildfireMARLlibWrapper

def test_basic_functionality():
    """기본 기능 테스트"""
    print("=" * 70)
    print("테스트 1: 기본 기능 테스트")
    print("=" * 70)

    env_config = {
        'num_agents': 2,
        'size': 11,
        'max_steps': 30,
        'cooperative_reward': False,
        'agent_start_positions': ((1, 1), (9, 9)),
    }

    env = WildfireMultiAgentWrapper(env_config)

    # 속성 확인
    print(f"✓ 에이전트 수: {env.num_agents}")
    print(f"✓ 에이전트 IDs: {env.get_agent_ids()}")
    print(f"✓ observation_space 타입: {type(env.observation_space)}")
    print(f"✓ action_space 타입: {type(env.action_space)}")

    # reset 테스트
    obs = env.reset(seed=42)
    print(f"✓ reset() 반환 타입: {type(obs)}")
    print(f"✓ observation keys: {list(obs.keys())}")
    print(f"✓ observation key 타입: {type(list(obs.keys())[0])}")
    assert isinstance(list(obs.keys())[0], int), "키가 int가 아닙니다!"

    # step 테스트
    actions = {0: 0, 1: 0}  # STILL action
    next_obs, rewards, dones, infos = env.step(actions)

    print(f"✓ step() 반환 값 개수: 4")
    print(f"✓ rewards: {rewards}")
    print(f"✓ dones 키: {list(dones.keys())}")
    assert "__all__" in dones, "__all__ 키가 없습니다!"
    print(f"✓ infos keys: {list(infos.keys())}")

    env.close()
    print("✓ 기본 기능 테스트 통과!\n")
    return True


def test_episode_execution():
    """전체 에피소드 실행 테스트"""
    print("=" * 70)
    print("테스트 2: 전체 에피소드 실행 테스트")
    print("=" * 70)

    env_config = {
        'num_agents': 2,
        'size': 11,
        'max_steps': 50,
        'cooperative_reward': True,
        'agent_start_positions': ((5, 5), (6, 6)),
        'initial_fire_size': 1,
    }

    env = WildfireMultiAgentWrapper(env_config)
    obs = env.reset(seed=123)

    total_rewards = {i: 0.0 for i in range(env.num_agents)}
    step_count = 0

    for step in range(100):
        # 랜덤 액션
        actions = env.action_space_sample()
        obs, rewards, dones, infos = env.step(actions)

        for i in range(env.num_agents):
            total_rewards[i] += rewards[i]

        step_count += 1

        if dones["__all__"]:
            print(f"✓ 에피소드 종료: step={step_count}")
            print(f"✓ 누적 보상: {total_rewards}")
            print(f"✓ 최종 info (agent 0): {infos[0]}")
            break

    env.close()
    print("✓ 에피소드 실행 테스트 통과!\n")
    return True


def test_different_agent_types():
    """다양한 에이전트 타입 테스트"""
    print("=" * 70)
    print("테스트 3: 다양한 에이전트 타입 테스트")
    print("=" * 70)

    # Helicopter, Truck, Crew 혼합
    env_config = {
        'num_agents': 3,
        'size': 11,
        'max_steps': 30,
        'num_helicopters': 1,
        'num_trucks': 1,
        'num_crews': 1,
        'agent_start_positions': ((1, 1), (5, 5), (9, 9)),
    }

    env = WildfireMultiAgentWrapper(env_config)

    print(f"✓ 에이전트 수: {env.num_agents}")
    print(f"✓ 에이전트 타입: Helicopter(1), Truck(1), Crew(1)")

    obs = env.reset(seed=42)

    # 에이전트별 속성 확인
    for i, agent in enumerate(env.env.agents):
        agent_type = type(agent).__name__
        speed = getattr(agent, 'speed', 1.0)
        efficiency = getattr(agent, 'efficiency', 1.0)
        print(f"✓ Agent {i}: {agent_type}, speed={speed}, efficiency={efficiency}")

    # 몇 스텝 실행
    for _ in range(5):
        actions = env.action_space_sample()
        obs, rewards, dones, infos = env.step(actions)
        if dones["__all__"]:
            break

    env.close()
    print("✓ 다양한 에이전트 타입 테스트 통과!\n")
    return True


def test_reward_shaping():
    """Reward shaping 테스트"""
    print("=" * 70)
    print("테스트 4: Reward Shaping 테스트")
    print("=" * 70)

    reward_types = ['cooperative', 'individual', 'individual2']

    for reward_type in reward_types:
        print(f"\n[{reward_type} reward 테스트]")

        env_config = {
            'num_agents': 2,
            'size': 11,
            'max_steps': 30,
            'agent_start_positions': ((5, 5), (6, 6)),
            'reward_shaping': reward_type,
        }

        env = WildfireMultiAgentWrapper(env_config)
        obs = env.reset(seed=42)

        # 몇 스텝 실행하여 보상 확인
        step_rewards = []
        for step in range(10):
            actions = env.action_space_sample()
            obs, rewards, dones, infos = env.step(actions)
            step_rewards.append(rewards)

            if dones["__all__"]:
                break

        print(f"  ✓ {len(step_rewards)} 스텝 실행")
        print(f"  ✓ 첫 번째 스텝 보상: {step_rewards[0]}")

        env.close()

    print("\n✓ Reward shaping 테스트 통과!\n")
    return True


def test_marllib_wrapper():
    """MARLlib wrapper 전용 기능 테스트"""
    print("=" * 70)
    print("테스트 5: MARLlib Wrapper 전용 기능 테스트")
    print("=" * 70)

    env_config = {
        'num_agents': 3,
        'size': 11,
        'max_steps': 30,
        'agent_start_positions': ((1, 1), (5, 5), (9, 9)),
    }

    env = WildfireMARLlibWrapper(env_config)

    # MARLlib 전용 속성 확인
    print(f"✓ agents: {env.agents}")
    print(f"✓ possible_agents: {env.possible_agents}")
    print(f"✓ n_agents: {env.n_agents}")

    obs = env.reset(seed=42)

    # state() 메서드 테스트
    state = env.state()
    print(f"✓ state shape: {state.shape}")
    print(f"✓ state_space: {env.state_space}")
    print(f"✓ state dtype: {state.dtype}")

    # get_observation_space / get_action_space 테스트
    obs_space_0 = env.get_observation_space(0)
    action_space_0 = env.get_action_space(0)
    print(f"✓ observation_space[0]: {obs_space_0}")
    print(f"✓ action_space[0]: {action_space_0}")

    # 전체 space 조회
    all_obs_space = env.get_observation_space()
    all_action_space = env.get_action_space()
    print(f"✓ get_observation_space() 반환 타입: {type(all_obs_space)}")
    print(f"✓ get_action_space() 반환 타입: {type(all_action_space)}")

    env.close()
    print("✓ MARLlib wrapper 테스트 통과!\n")
    return True


def test_space_compatibility():
    """Space 호환성 테스트 (RLlib/MARLlib)"""
    print("=" * 70)
    print("테스트 6: Space 호환성 테스트")
    print("=" * 70)

    env_config = {
        'num_agents': 2,
        'size': 11,
        'max_steps': 30,
        'agent_start_positions': ((1, 1), (9, 9)),
    }

    env = WildfireMultiAgentWrapper(env_config)

    # observation_space 샘플링 테스트
    obs_sample = env.observation_space_sample()
    print(f"✓ observation_space_sample() keys: {list(obs_sample.keys())}")
    print(f"✓ observation_space_sample()[0] shape: {obs_sample[0].shape}")

    # action_space 샘플링 테스트
    action_sample = env.action_space_sample()
    print(f"✓ action_space_sample() keys: {list(action_sample.keys())}")
    print(f"✓ action_space_sample() values: {action_sample}")

    # space contains 테스트
    obs = env.reset(seed=42)
    for i in range(env.num_agents):
        assert env._single_obs_space.contains(obs[i]), f"observation {i}가 space에 포함되지 않습니다!"
    print(f"✓ 모든 observation이 space에 포함됨")

    # action 테스트
    actions = env.action_space_sample()
    for i in range(env.num_agents):
        assert env._single_action_space.contains(actions[i]), f"action {i}가 space에 포함되지 않습니다!"
    print(f"✓ 모든 action이 space에 포함됨")

    env.close()
    print("✓ Space 호환성 테스트 통과!\n")
    return True


def test_key_conversion():
    """키 변환 테스트 (int <-> str)"""
    print("=" * 70)
    print("테스트 7: 키 변환 테스트")
    print("=" * 70)

    env_config = {
        'num_agents': 2,
        'size': 11,
        'max_steps': 30,
        'agent_start_positions': ((1, 1), (9, 9)),
    }

    env = WildfireMultiAgentWrapper(env_config)

    # reset 키 확인
    obs = env.reset(seed=42)
    print(f"✓ reset() observation keys: {list(obs.keys())}")
    print(f"✓ 키 타입: {type(list(obs.keys())[0])}")
    assert all(isinstance(k, int) for k in obs.keys()), "observation 키가 int가 아닙니다!"

    # step 키 확인
    actions = {0: 0, 1: 0}
    next_obs, rewards, dones, infos = env.step(actions)

    print(f"✓ step() observation keys: {list(next_obs.keys())}")
    print(f"✓ step() rewards keys: {list(rewards.keys())}")
    print(f"✓ step() infos keys: {list(infos.keys())}")

    assert all(isinstance(k, int) for k in next_obs.keys()), "observation 키가 int가 아닙니다!"
    assert all(isinstance(k, int) for k in rewards.keys()), "rewards 키가 int가 아닙니다!"
    assert all(isinstance(k, int) for k in infos.keys()), "infos 키가 int가 아닙니다!"

    env.close()
    print("✓ 키 변환 테스트 통과!\n")
    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 70)
    print("WILDFIRE WRAPPER 상세 테스트 시작")
    print("=" * 70 + "\n")

    tests = [
        test_basic_functionality,
        test_episode_execution,
        test_different_agent_types,
        test_reward_shaping,
        test_marllib_wrapper,
        test_space_compatibility,
        test_key_conversion,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result, None))
        except Exception as e:
            print(f"✗ {test_func.__name__} 실패: {e}\n")
            results.append((test_func.__name__, False, str(e)))

    # 결과 요약
    print("=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  에러: {error}")

    print("\n" + "=" * 70)
    print(f"결과: {passed}/{total} 테스트 통과")
    print("=" * 70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
