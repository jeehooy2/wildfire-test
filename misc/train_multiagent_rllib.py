"""
RLlib을 사용한 다중 에이전트 PPO 학습

이 스크립트는 Ray RLlib을 사용하여 wildfire 환경에서
다중 에이전트를 학습시킵니다.

설치 필요:
pip install "ray[rllib]>=2.7.0" torch

실행:
python train_multiagent_rllib.py

API 버전:
- 최신 RLlib API (Ray 2.0+) 사용
- from ray.rllib.algorithms.ppo import PPOConfig  # ✅ 신버전
- 구버전 (ray.rllib.agents) 사용 X

# uv run python train_multiagent_rllib.py --iterations 1000 --checkpoint-freq 10
"""

import os
import ray
import logging
from ray.rllib.algorithms.ppo import PPOConfig  # 신버전 API (Ray 2.0+)
from ray.tune.registry import register_env
from misc.wildfire_rllib_wrapper import WildfireRLlibEnv

# Ray 및 RLlib 로깅 레벨 설정
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)


def train_rllib_multiagent(
    num_iterations=100,
    checkpoint_freq=10,
    save_dir="./rllib_checkpoints"
):
    """
    RLlib PPO로 다중 에이전트 학습

    Parameters
    ----------
    num_iterations : int
        학습 반복 횟수
    checkpoint_freq : int
        체크포인트 저장 빈도
    save_dir : str
        체크포인트 저장 디렉토리
    """

    # 체크포인트 디렉토리를 절대 경로로 변환
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Ray 초기화 (메트릭 에이전트 비활성화로 경고 제거)
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,  # Ray 내부 로깅도 ERROR만 표시
            _metrics_export_port=None,  # 메트릭 exporter 비활성화
            _system_config={
                "metrics_report_interval_ms": 0,  # 메트릭 리포트 비활성화
            }
        )

    # 환경 등록
    def env_creator(env_config):
        # base_env = WildfireRLlibEnv(env_config)
        # # RLlib의 호환성 래퍼로 감싸서 새 Gymnasium API 지원
        # return MultiAgentEnvCompatibility(base_env)
        return WildfireRLlibEnv(env_config)

    register_env("wildfire-ma", env_creator)

    # 환경 설정
    env_config = {
        "num_agents": 2,
        "size": 17,
        "initial_fire_size": 3,
        "cooperative_reward": False,  # 다중 에이전트 학습 (각자 보상)
        "max_steps": 300,
        "agent_start_positions": ((1, 1), (15, 15)),
        "log_selfish_region_metrics": True,
        "selfish_region_xmin": [7, 13],
        "selfish_region_xmax": [9, 15],
        "selfish_region_ymin": [7, 1],
        "selfish_region_ymax": [9, 3],
        "delta_beta": 0.7,     # ← 행동이 화재 소멸에 강하게 영향 주도록
        "beta": 0.99,          # 기본값 유지 가능
        "alpha": 0.05,         # 기본값 유지 가능(확산)
    }

    # 테스트 환경 생성하여 에이전트 수 확인
    test_env = env_creator(env_config)
    num_agents = test_env.num_agents
    test_env.close()

    print("=" * 60)
    print("RLlib 다중 에이전트 PPO 학습")
    print("=" * 60)
    print(f"에이전트 수: {num_agents}")
    print(f"학습 반복 횟수: {num_iterations}")
    print(f"체크포인트 저장: {save_dir}")
    print("=" * 60)

    # PPO 설정
    config = (
        PPOConfig()
        .environment(
            env="wildfire-ma",
            env_config=env_config
        )
        .framework("torch")
        .training(
            lr=3e-4,
            gamma=0.99,
            train_batch_size_per_learner=512,  # 새 API
        )
        .env_runners(
            num_env_runners=1,  # 로컬에서만 실행
            num_envs_per_env_runner=1,
        )
        .learners(
            num_learners=1,  # 로컬 learner 사용
            num_gpus_per_learner=0,  # CPU 학습 (GPU 없을 경우)
        )
        .resources(
            num_gpus=0,  # CPU만 사용
        )
        .multi_agent(
            # 간단한 정책 설정: 모든 에이전트가 같은 정책 공유
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )

    # 알고리즘 생성
    algo = config.build()

    print("\n학습 시작...")
    print("(Ctrl+C로 중단 가능)\n")

    # 학습 루프
    best_reward = float('-inf')
    try:
        for i in range(num_iterations):
            result = algo.train()

            # 진행 상황 출력
            # RLlib 새 API stack에서는 metrics가 env_runners 아래에 있음
            env_runners = result.get("env_runners", {})
            episode_reward_mean = env_runners.get("episode_return_mean", 0)
            episode_len_mean = env_runners.get("episode_len_mean", 0)

            print(f"Iteration {i + 1}/{num_iterations}:")
            print(f"  평균 보상: {episode_reward_mean:.2f}")
            print(f"  평균 에피소드 길이: {episode_len_mean:.1f}")

            # 체크포인트 저장
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_path = algo.save(save_dir)
                print(f"  ✓ 체크포인트 저장: {checkpoint_path}")

                # 최고 성능 모델 별도 저장
                if episode_reward_mean > best_reward:
                    best_reward = episode_reward_mean
                    best_checkpoint = algo.save(os.path.join(save_dir, "best"))
                    print(f"  ⭐ 최고 성능 모델 저장: {best_checkpoint}")

            print()

    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")

    # 최종 체크포인트 저장
    final_checkpoint = algo.save(os.path.join(save_dir, "final"))
    print("\n" + "=" * 60)
    print(f"최종 모델 저장: {final_checkpoint}")
    print("=" * 60)

    algo.stop()
    ray.shutdown()

    return final_checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLlib 다중 에이전트 학습")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="학습 반복 횟수 (기본값: 100)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="체크포인트 저장 빈도 (기본값: 10)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./rllib_checkpoints",
        help="체크포인트 저장 디렉토리"
    )

    args = parser.parse_args()

    checkpoint_path = train_rllib_multiagent(
        num_iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        save_dir=args.save_dir
    )

    print("\n학습 완료!")
    print(f"모델 위치: {checkpoint_path}")
    print("\n다음 명령으로 시뮬레이션 실행:")
    print(f"python simulate_trained_model.py --checkpoint {checkpoint_path}")
