"""
RLlib을 사용한 다중 에이전트 PPO 학습 - 체크포인트에서 이어서 학습

이 스크립트는 기존에 저장된 체크포인트에서 모델을 복원하고
추가로 학습을 이어서 진행합니다.

설치 필요:
pip install "ray[rllib]>=2.7.0" torch

실행 예시:
# 체크포인트에서 50 iteration 추가 학습
python train_rllib/train_rllib_mappo_resume.py \
    --checkpoint ./train_rllib/experiments/run6_individual_agent3_iter100 \
    --iterations 50
    --save-dir ./train_rllib/experiments/run6_resumed

# 체크포인트에서 50 iteration 추가 학습
python train_rllib/train_rllib_mappo_resume.py \
    --checkpoint ./train_rllib/experiments/run1/checkpoint_000100 \
    --iterations 50 \
    --checkpoint-freq 10 \
    --save-dir ./train_rllib/experiments/run1_resumed

API 버전:
- 최신 RLlib API (Ray 2.0+) 사용
- Algorithm.from_checkpoint() 메서드 사용
"""

import os
import warnings
import ray
import logging
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from train_rllib.wildfire_rllib_wrapper_new import WildfireRLlibEnv
from train_rllib.environment import ENV_CONFIG

# 모든 deprecation 경고 억제
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Ray 및 RLlib 로깅 레벨 설정
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)


def resume_training(
    checkpoint_path,
    num_iterations=100,
    checkpoint_freq=10,
    save_dir="./train_rllib/experiments_resumed"
):
    """
    체크포인트에서 복원하여 PPO 학습 이어하기

    Parameters
    ----------
    checkpoint_path : str
        복원할 체크포인트 경로 (예: ./experiments/run1/checkpoint_000100)
    num_iterations : int
        추가로 학습할 반복 횟수
    checkpoint_freq : int
        체크포인트 저장 빈도
    save_dir : str
        새로운 체크포인트 저장 디렉토리
    """

    # 체크포인트 경로 확인
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    # 저장 디렉토리 생성
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Ray 초기화
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            _metrics_export_port=None,
            _system_config={
                "metrics_report_interval_ms": 0,
            }
        )

    # 환경 등록 (체크포인트 복원 전에 필요)
    def env_creator(env_config):
        return WildfireRLlibEnv(env_config)

    register_env("wildfire-ma", env_creator)

    print("=" * 60)
    print("RLlib 다중 에이전트 PPO 이어서 학습")
    print("=" * 60)
    print(f"체크포인트: {checkpoint_path}")
    print(f"추가 학습 반복: {num_iterations}")
    print(f"새 저장 디렉토리: {save_dir}")
    print("=" * 60)

    # 체크포인트에서 알고리즘 복원
    print("\n체크포인트 복원 중...")
    algo = PPO.from_checkpoint(checkpoint_path)
    print("✓ 체크포인트 복원 완료\n")

    # 현재 iteration 확인 (복원된 모델이 몇 iteration까지 학습했는지)
    try:
        # RLlib 3.0+ API
        initial_iteration = algo.training_iteration
    except AttributeError:
        # 이전 버전 fallback
        initial_iteration = algo.iteration

    print(f"현재 iteration: {initial_iteration}")
    print(f"목표 iteration: {initial_iteration + num_iterations}\n")

    print("학습 시작...")
    print("(Ctrl+C로 중단 가능)\n")

    # 학습 루프
    best_reward = float('-inf')
    try:
        for i in range(num_iterations):
            result = algo.train()

            # 진행 상황 출력
            env_runners = result.get("env_runners", {})
            episode_reward_mean = env_runners.get("episode_return_mean", 0)
            episode_len_mean = env_runners.get("episode_len_mean", 0)

            # 실제 iteration 번호 계산
            current_iter = initial_iteration + i + 1

            print(f"Iteration {current_iter} (추가 {i + 1}/{num_iterations}):")
            print(f"  평균 보상: {episode_reward_mean:.2f}")
            print(f"  평균 에피소드 길이: {episode_len_mean:.1f}")

            # 체크포인트 저장
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_save_path = algo.save(save_dir)
                # print(f"  ✓ 체크포인트 저장: {checkpoint_save_path}")

                # 최고 성능 모델 별도 저장
                if episode_reward_mean > best_reward:
                    best_reward = episode_reward_mean
                    best_checkpoint = algo.save(os.path.join(save_dir, "best"))
                    # print(f"  ⭐ 최고 성능 모델 저장: {best_checkpoint}")

            print()

    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")

    # 최종 체크포인트 저장
    final_checkpoint = algo.save(os.path.join(save_dir, "final"))
    print("\n" + "=" * 60)
    print(f"최종 iteration: {initial_iteration + num_iterations}")
    # print(f"최종 모델 저장: {final_checkpoint}")
    print("=" * 60)

    algo.stop()
    ray.shutdown()

    return final_checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLlib 체크포인트에서 이어서 학습")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="복원할 체크포인트 경로 (예: ./experiments/run1/checkpoint_000100)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="추가로 학습할 반복 횟수 (기본값: 100)"
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
        default="./train_rllib/experiments_resumed",
        help="새 체크포인트 저장 디렉토리"
    )

    args = parser.parse_args()

    checkpoint_path = resume_training(
        checkpoint_path=args.checkpoint,
        num_iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        save_dir=args.save_dir
    )

    print("\n학습 완료!")
    print(f"\n💡 사용 팁:")
    print("  - 다시 이어서 학습하려면: --checkpoint {최신 체크포인트 경로}")
    print("  - 최고 성능 모델 사용: --checkpoint {save_dir}/best/checkpoint_XXXXXX")
