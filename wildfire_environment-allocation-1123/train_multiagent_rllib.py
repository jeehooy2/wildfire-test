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
from ray.rllib.algorithms.ppo import PPOConfig  # 신버전 API (Ray 2.0+)
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from wildfire_rllib_wrapper import WildfireRLlibEnv
from masked_model import MaskedCategoricalTorchModel


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

    # Ray 초기화
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env={"excludes": ['.git/']})

    # Register custom masked model
    ModelCatalog.register_custom_model("masked_categorical", MaskedCategoricalTorchModel)

    # 환경 등록
    def env_creator(env_config):
        return WildfireRLlibEnv(env_config)

    register_env("wildfire-ma", env_creator)

    # 환경 설정
    env_config = {
        "num_agents": 2,
        "size": 17,
        "initial_fire_size": 4,
        "cooperative_reward": False,  # 다중 에이전트 학습 (각자 보상)
        "max_steps": 200,  # 50: 에피소드가 빨리 끝나서 보상 metrics 확인 가능
        "agent_start_positions": ((1, 1), (15, 15)),
        "log_selfish_region_metrics": False,
        "selfish_region_xmin": [7, 13],
        "selfish_region_xmax": [9, 15],
        "selfish_region_ymin": [7, 1],
        "selfish_region_ymax": [9, 3],
        "delta_beta": 0.7,     # ← 행동이 화재 소멸에 강하게 영향 주도록
        "beta": 0.99,          # 기본값 유지 가능
        "alpha": 0.05,         # 기본값 유지 가능(확산)
        # Action masking configuration
        "use_action_mask": True,  # Disable action masking for now to debug
        "max_masked_actions": 64,  # Top-K candidate actions
        "target_switch_penalty": 0.01,  # Penalty for changing targets
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
            env_config=env_config,
            disable_env_checking=True,  # env checking 비활성화로 빠른 시작
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,  # TODO
            enable_env_runner_and_connector_v2=False  # TODO
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            train_batch_size=2000,  # 작은 배치로 빠른 테스트
            model={
                "fcnet_hiddens": [128, 128],  # 작은 네트워크로 빠른 학습
            }
        )
        .env_runners(
            num_env_runners=8,  # 2개로 줄여서 빠른 초기화
            sample_timeout_s=None,
            num_envs_per_env_runner=4,
            rollout_fragment_length='auto',
            batch_mode="complete_episodes",  # 에피소드가 완료될 때까지 기다림
        )
        .resources(
            num_gpus=1,
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
            # env_runners 아래에 metrics가 있음
            env_runners = result.get("env_runners", {})
            episode_reward_mean = env_runners.get("episode_reward_mean", None)
            episode_reward_max = env_runners.get("episode_reward_max", None)
            episode_reward_min = env_runners.get("episode_reward_min", None)
            episode_len_mean = env_runners.get("episode_len_mean", None)
            episodes_this_iter = env_runners.get("episodes_this_iter", 0)

            # 샘플링 통계
            num_env_steps = result.get("num_env_steps_sampled_this_iter", 0)

            # 학습 통계
            learner_info = result.get("info", {}).get("learner", {}).get("shared_policy", {}).get("learner_stats", {})
            policy_loss = learner_info.get("policy_loss", 0)
            vf_loss = learner_info.get("vf_loss", 0)

            print(f"Iteration {i + 1}/{num_iterations}:")
            if episode_reward_mean is not None:
                print(f"  평균 보상: {episode_reward_mean:.2f} (min: {episode_reward_min:.2f}, max: {episode_reward_max:.2f})")
                print(f"  평균 에피소드 길이: {episode_len_mean:.1f}")
                print(f"  완료된 에피소드 수: {episodes_this_iter}")
                print(f"  Policy Loss: {policy_loss:.4f}, Value Loss: {vf_loss:.4f}")
            else:
                print(f"  평균 보상: 아직 에피소드 완료 없음")
            print(f"  샘플링된 스텝: {num_env_steps}")

            # 체크포인트 저장
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_path = algo.save(save_dir)
                print(f"  ✓ 체크포인트 저장: {checkpoint_path}")

                # 최고 성능 모델 별도 저장
                if episode_reward_mean is not None and episode_reward_mean > best_reward:
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
