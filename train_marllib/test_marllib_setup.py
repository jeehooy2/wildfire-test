"""
MARLlib MAPPO 설정 테스트 스크립트

이 스크립트는 MARLlib 환경이 제대로 설정되어 있는지 확인합니다.

실행:
conda activate marllib-x86
python train_rllib/test_marllib_setup.py
"""

import sys

def test_imports():
    """필수 라이브러리 import 테스트"""
    print("=" * 60)
    print("1. Import 테스트")
    print("=" * 60)

    try:
        import gym
        print("✓ gym")
    except ImportError as e:
        print(f"✗ gym: {e}")
        return False

    try:
        import wildfire_environment
        print("✓ wildfire_environment")
    except ImportError as e:
        print(f"✗ wildfire_environment: {e}")
        return False

    try:
        from marllib import marl
        print("✓ marllib")
    except ImportError as e:
        print(f"✗ marllib: {e}")
        print("\n해결책: conda activate marllib-x86")
        return False

    try:
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
        print("✓ ray.rllib")
    except ImportError as e:
        print(f"✗ ray.rllib: {e}")
        return False

    return True


def test_wrapper():
    """환경 래퍼 테스트"""
    print("\n" + "=" * 60)
    print("2. 환경 래퍼 테스트")
    print("=" * 60)

    try:
        from train_marllib.wildfire_marllib_wrapper import WildfireMARLlibEnv
        from train_marllib.environment import ENV_CONFIG

        print("환경 생성 중...")
        env = WildfireMARLlibEnv(ENV_CONFIG)

        print(f"\n✓ 환경 생성 성공")
        print(f"  - 에이전트 수: {env.num_agents}")
        print(f"  - 관찰 공간: {env.observation_space}")
        print(f"  - 행동 공간: {env.action_space}")
        print(f"  - 상태 공간: {env.state_space}")

        # Reset 테스트
        print("\nReset 테스트...")
        obs = env.reset()
        print(f"✓ Reset 성공")
        print(f"  - Observations: {len(obs)} agents")
        print(f"  - Observation shape: {obs[0].shape}")

        # Global state 테스트
        print("\nGlobal state 테스트...")
        state = env.state()
        print(f"✓ Global state 접근 성공")
        print(f"  - State shape: {state.shape}")

        # Step 테스트
        print("\nStep 테스트...")
        actions = {i: env.action_space.sample() for i in range(env.num_agents)}
        obs, rewards, dones, infos = env.step(actions)
        print(f"✓ Step 성공")
        print(f"  - Rewards: {rewards}")
        print(f"  - Done: {dones}")
        print(f"  - Info contains state: {'state' in infos[0]}")

        env.close()
        return True

    except Exception as e:
        print(f"✗ 환경 래퍼 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_marllib_integration():
    """MARLlib 통합 테스트"""
    print("\n" + "=" * 60)
    print("3. MARLlib 통합 테스트")
    print("=" * 60)

    try:
        from marllib import marl

        print("✓ MARLlib MAPPO 알고리즘 사용 가능")

        # 알고리즘 초기화 테스트
        print("\nMAPPO 초기화 테스트...")
        mappo = marl.algos.mappo(hyperparam_source="common")
        print("✓ MAPPO 초기화 성공")

        return True

    except Exception as e:
        print(f"✗ MARLlib 통합 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("MARLlib MAPPO 설정 테스트")
    print("=" * 60 + "\n")

    # 테스트 실행
    test1 = test_imports()
    if not test1:
        print("\n❌ Import 테스트 실패")
        sys.exit(1)

    test2 = test_wrapper()
    if not test2:
        print("\n❌ 환경 래퍼 테스트 실패")
        sys.exit(1)

    test3 = test_marllib_integration()
    if not test3:
        print("\n❌ MARLlib 통합 테스트 실패")
        sys.exit(1)

    # 모든 테스트 통과
    print("\n" + "=" * 60)
    print("🎉 모든 테스트 통과!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. 학습 시작:")
    print("   python train_rllib/train_marllib_mappo.py --timesteps 1000000")
    print("\n2. 학습 모니터링:")
    print("   tensorboard --logdir ~/ray_results/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
