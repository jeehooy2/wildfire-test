"""
MARLlib을 사용한 다중 에이전트 MAPPO 학습

이 스크립트는 MARLlib을 사용하여 wildfire 환경에서
다중 에이전트를 학습시킵니다.

설치 필요:
pip install marllib

실행 예시:
# 기본 실행
python train_marllib_self/train_marllib_mappo_new.py

# 파라미터 조정 실행
python train_marllib_self/train_marllib_mappo_new.py --timesteps 10000000 --checkpoint-freq 100
"""

# === PYTHONPATH 설정 (필요시 이 블록 전체 삭제 가능) ===
import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# === PYTHONPATH 설정 끝 ===

import os
import warnings
import argparse
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from train_marllib_self.environment import ENV_CONFIG
from train_marllib_self.wildfire_marllib_wrapper import MARLlibWildfireEnv

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


def train_marllib_mappo(
    timesteps_total=10000000,
    episode_reward_mean=2000,
    checkpoint_freq=50,
    num_gpus=0,
    num_workers=2,
    share_policy='all',
    local_mode=True,
    core_arch='mlp',
    encode_layer='128-128'
):
    """
    MARLlib MAPPO로 다중 에이전트 학습

    Parameters
    ----------
    timesteps_total : int
        총 학습 타임스텝 수
    episode_reward_mean : float
        목표 평균 에피소드 보상
    checkpoint_freq : int
        체크포인트 저장 빈도 (에피소드 기준)
    num_gpus : int
        사용할 GPU 수 (0이면 CPU만 사용)
    num_workers : int
        병렬 워커 수
    share_policy : str
        정책 공유 방식 ('all', 'group', 'individual')
    local_mode : bool
        로컬 모드 실행 여부 (디버깅용)
    core_arch : str
        신경망 아키텍처 ('mlp', 'gru', 'lstm')
    encode_layer : str
        인코더 레이어 구조 (예: '128-128')
    """

    print("=" * 60)
    print("MARLlib 다중 에이전트 MAPPO 학습")
    print("=" * 60)
    print(f"총 타임스텝: {timesteps_total}")
    print(f"목표 평균 보상: {episode_reward_mean}")
    print(f"체크포인트 저장 빈도: {checkpoint_freq}")
    print(f"GPU 수: {num_gpus}")
    print(f"워커 수: {num_workers}")
    print(f"정책 공유: {share_policy}")
    print(f"신경망 구조: {core_arch} ({encode_layer})")
    print("=" * 60)

    # MARLlib 환경 레지스트리에 등록
    # add_new_env.py의 방식을 따름
    ENV_REGISTRY["wildfire-ma"] = MARLlibWildfireEnv

    # 환경 초기화
    # MARLlib은 map_name을 자동으로 env_config에 추가
    env = marl.make_env(
        environment_name="wildfire-ma",
        map_name="wildfire",
        force_coop=False,  # cooperative_reward 설정에 따름
        **ENV_CONFIG
    )

    # MAPPO 알고리즘 선택
    # hyperparam_source 옵션:
    # - "common": 일반적인 하이퍼파라미터
    # - "test": 테스트용 (빠른 실행)
    # - "mpe": MPE 환경 최적화
    # - "smac": SMAC 환경 최적화
    mappo = marl.algos.mappo(hyperparam_source="common")

    # 모델 커스터마이징
    # core_arch: 'mlp', 'gru', 'lstm'
    # encode_layer: 인코더 레이어 구조 (예: '128-128', '256-256')
    model = marl.build_model(
        env,
        mappo,
        {
            "core_arch": core_arch,
            "encode_layer": encode_layer
        }
    )

    print("\n학습 시작...")
    print("(Ctrl+C로 중단 가능)\n")

    # 학습 시작
    # stop: 학습 중단 조건
    # share_policy 옵션:
    # - 'all': 모든 에이전트가 하나의 정책 공유
    # - 'group': 그룹별로 정책 공유
    # - 'individual': 각 에이전트가 독립 정책
    mappo.fit(
        env,
        model,
        stop={
            'episode_reward_mean': episode_reward_mean,
            'timesteps_total': timesteps_total
        },
        local_mode=local_mode,
        num_gpus=num_gpus,
        num_workers=num_workers,
        share_policy=share_policy,
        checkpoint_freq=checkpoint_freq
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MARLlib 다중 에이전트 MAPPO 학습")

    # 학습 관련 파라미터
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000000,
        help="총 학습 타임스텝 수 (기본값: 10000000)"
    )
    parser.add_argument(
        "--target-reward",
        type=float,
        default=2000,
        help="목표 평균 에피소드 보상 (기본값: 2000)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="체크포인트 저장 빈도 (기본값: 50)"
    )

    # 리소스 관련 파라미터
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="사용할 GPU 수 (기본값: 0, 0이면 CPU만 사용)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="병렬 워커 수 (기본값: 2)"
    )

    # 정책 공유 관련 파라미터
    parser.add_argument(
        "--share-policy",
        type=str,
        default='all',
        choices=['all', 'group', 'individual'],
        help="정책 공유 방식 (기본값: all)"
    )

    # 모델 구조 관련 파라미터
    parser.add_argument(
        "--core-arch",
        type=str,
        default='mlp',
        choices=['mlp', 'gru', 'lstm'],
        help="신경망 아키텍처 (기본값: mlp)"
    )
    parser.add_argument(
        "--encode-layer",
        type=str,
        default='128-128',
        help="인코더 레이어 구조 (기본값: 128-128)"
    )

    # 실행 모드
    parser.add_argument(
        "--no-local-mode",
        action='store_true',
        help="로컬 모드 비활성화 (분산 실행)"
    )

    args = parser.parse_args()

    train_marllib_mappo(
        timesteps_total=args.timesteps,
        episode_reward_mean=args.target_reward,
        checkpoint_freq=args.checkpoint_freq,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        share_policy=args.share_policy,
        local_mode=not args.no_local_mode,
        core_arch=args.core_arch,
        encode_layer=args.encode_layer
    )
