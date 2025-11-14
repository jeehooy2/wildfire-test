"""
MARLlib을 사용한 다중 에이전트 MAPPO (CTDE) 학습

이 스크립트는 MARLlib을 사용하여 wildfire 환경에서
진정한 MAPPO (Centralized Training with Decentralized Execution)로 학습합니다.

RLlib의 기본 PPO는 Independent PPO(IPPO)이지만,
MARLlib의 MAPPO는 Centralized Critic을 사용하여 진정한 CTDE를 구현합니다.

설치 필요:
conda activate marllib-x86

실행 예시:
# 기본 실행
cd /Users/jeehooy2/Desktop/wildfire_environment-aiblue
conda activate marllib-x86
python train_marllib/train_marllib_mappo.py --iterations 100 --checkpoint-freq 10

# 실험 이름 지정
python train_marllib/train_marllib_mappo.py --iterations 1000 --checkpoint-freq 50 --exp-name mappo_run1
"""

import os
import sys
import warnings
import argparse

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_marllib.environment import ENV_CONFIG

# 경고 억제
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


def train_marllib_mappo(
    num_iterations=100,
    checkpoint_freq=10,
    exp_name="mappo_wildfire",
    num_workers=2,
    num_gpus=0,
    restore_path=None,
):
    """
    MARLlib MAPPO로 다중 에이전트 학습 (진정한 CTDE)

    Parameters
    ----------
    num_iterations : int
        학습 반복 횟수 (iterations)
    checkpoint_freq : int
        체크포인트 저장 빈도 (iterations 단위)
    exp_name : str
        실험 이름 (결과 저장 폴더명)
    num_workers : int
        병렬 worker 수
    num_gpus : int
        사용할 GPU 수
    restore_path : str
        재개할 체크포인트 경로
    """

    # MARLlib import (조건부 import로 환경 확인 시 에러 방지)
    try:
        from marllib import marl
        from ray import tune
        import ray
    except ImportError as e:
        print("=" * 60)
        print("❌ MARLlib을 찾을 수 없습니다!")
        print("=" * 60)
        print("다음 명령으로 marllib-x86 환경을 활성화하세요:")
        print("  conda activate marllib-x86")
        print("\nMARLlib이 설치되지 않은 경우:")
        print("  pip install git+https://github.com/Replicable-MARL/MARLlib.git")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("MARLlib MAPPO (CTDE) 학습")
    print("=" * 60)
    print(f"환경 설정:")
    print(f"  그리드 크기: {ENV_CONFIG['size']}x{ENV_CONFIG['size']}")
    print(f"  에이전트 수: {ENV_CONFIG['num_agents']}")
    print(f"  - Helicopters: {ENV_CONFIG.get('num_helicopters', 0)}")
    print(f"  - Trucks: {ENV_CONFIG.get('num_trucks', 0)}")
    print(f"  - Crews: {ENV_CONFIG.get('num_crews', 0)}")
    print(f"  Max steps: {ENV_CONFIG['max_steps']}")
    print(f"\n학습 설정:")
    print(f"  학습 반복 횟수: {num_iterations} iterations")
    print(f"  체크포인트 빈도: {checkpoint_freq} iterations")
    print(f"  실험 이름: {exp_name}")
    print(f"  Workers: {num_workers}")
    print(f"  GPUs: {num_gpus}")
    if restore_path:
        print(f"  재개 경로: {restore_path}")
    print("=" * 60)

    # Step 1: 환경 등록
    print("\n[1/4] 환경 등록 중...")

    # MARLlib 환경 등록을 위한 설정
    from marllib.envs.base_env import ENV_REGISTRY
    from train_marllib.wildfire_marllib_wrapper import WildfireMARLlibEnv

    # 정책 매핑 정보 생성 (시나리오별로)
    num_helicopters = ENV_CONFIG.get("num_helicopters", 0)
    num_trucks = ENV_CONFIG.get("num_trucks", 0)
    num_crews = ENV_CONFIG.get("num_crews", 0)

    if num_helicopters > 0 or num_trucks > 0 or num_crews > 0:
        # 이질적 에이전트: 3개 정책
        policy_mapping_dict = {
            "custom": {  # map_name과 일치해야 함
                "description": "heterogeneous agents (helicopter, truck, crew)",
                "team_prefix": ("helicopter_", "truck_", "crew_"),
                "all_agents_one_policy": False,
                "one_agent_one_policy": False,
            }
        }
    else:
        # 동질적 에이전트: 1개 정책
        policy_mapping_dict = {
            "custom": {  # map_name과 일치해야 함
                "description": "homogeneous agents",
                "team_prefix": ("agent_",),
                "all_agents_one_policy": True,
                "one_agent_one_policy": False,
            }
        }

    # Wildfire 환경을 MARLlib에 등록
    if "wildfire" not in ENV_REGISTRY:
        # 환경 생성 함수
        def env_creator(env_config):
            return WildfireMARLlibEnv(env_config)

        # 환경 정보 딕셔너리
        env_info = {
            "wildfire": {
                "env_args": ENV_CONFIG,
                "scenario": ["custom"],  # 시나리오 이름
            }
        }

        # 환경 등록
        ENV_REGISTRY["wildfire"] = {
            "creator": env_creator,
            "info": env_info,
            "policy_mapping": policy_mapping_dict,
        }

        print("✓ Wildfire 환경 등록 완료")

    # Step 2: 환경 생성
    print("\n[2/4] 환경 생성 중...")

    # MARLlib make_env을 사용하지 않고 직접 생성
    # (커스텀 환경이므로)
    env = WildfireMARLlibEnv(ENV_CONFIG)

    # local_mode 결정 (나중에 Ray 초기화 시 사용)
    # True로 설정하면 Redis 없이 실행 (단일 프로세스, 느림)
    # False로 설정하면 Redis 사용 (멀티 프로세스, 빠름, macOS에서 오류 가능)
    use_local_mode_for_env = True  # macOS Redis 문제 회피

    # MARLlib이 요구하는 env_config_dict 형식 (run_cc.py 기준)
    env_config_dict = {
        "env": "wildfire",  # 환경 이름 (짧은 버전)
        "env_name": "wildfire",  # 환경 이름 (전체)
        "env_args": {
            "map_name": "custom",
            **ENV_CONFIG  # 전체 환경 설정 포함
        },
        "num_agents": ENV_CONFIG["num_agents"],
        "episode_limit": ENV_CONFIG["max_steps"],
        "policy_mapping_info": policy_mapping_dict,  # 정책 매핑 정보 추가
        "seed": 0,  # 랜덤 시드
        "local_mode": use_local_mode_for_env,  # Ray 로컬 모드
        "share_policy": "group" if (num_helicopters > 0 or num_trucks > 0 or num_crews > 0) else "all",
        # run_cc.py에서 요구하는 추가 필드
        "algorithm": "mappo",  # 알고리즘 이름
        "num_gpus_per_worker": 0,  # worker당 GPU 수
        "num_gpus": num_gpus,  # 전체 GPU 수
        "num_workers": num_workers,  # worker 수
        "framework": "torch",  # 프레임워크
        "evaluation_interval": 50,  # 평가 빈도
        # stop 조건
        "stop_reward": 1000000,  # 보상 목표 (매우 큰 값)
        "stop_timesteps": 10000000,  # 최대 timesteps
        "stop_iters": num_iterations,  # 최대 iterations
        # restore 경로
        "restore_path": {
            "model_path": restore_path if restore_path else "",
        },
    }

    print(f"✓ 환경 생성 완료")
    print(f"  관찰 공간: {env.observation_space}")
    print(f"  행동 공간: {env.action_space}")
    print(f"  상태 공간 (global): {env.state_space}")

    # Step 3: MAPPO 알고리즘 초기화
    print("\n[3/4] MAPPO 알고리즘 초기화 중...")

    mappo = marl.algos.mappo(hyperparam_source="common")

    print("✓ MAPPO 초기화 완료")
    print("  알고리즘: MAPPO (Multi-Agent PPO with Centralized Critic)")
    print("  학습 방식: CTDE (Centralized Training, Decentralized Execution)")

    # Step 4: 모델 생성
    print("\n[4/4] 에이전트 모델 생성 중...")

    model_config = {
        "core_arch": "mlp",           # MLP 네트워크
        "encode_layer": "128-256",    # 인코더 레이어 구조
        "fc_layer": "256-256",        # Fully connected 레이어
        "activation": "relu",          # 활성화 함수
    }

    model = marl.build_model(
        environment=(env, env_config_dict),  # 파라미터 이름: environment
        algorithm=mappo,                      # 파라미터 이름: algorithm
        model_preference=model_config         # 파라미터 이름: model_preference
    )

    print("✓ 모델 생성 완료")
    print(f"  아키텍처: {model_config['core_arch'].upper()}")
    print(f"  인코더: {model_config['encode_layer']}")
    print(f"  FC 레이어: {model_config['fc_layer']}")

    # Step 5: 정책 공유 방식 결정
    if num_helicopters > 0 or num_trucks > 0 or num_crews > 0:
        share_policy = "group"  # 타입별로 정책 공유
        print(f"\n정책 공유 방식: {share_policy}")
        print(f"  - Helicopter 정책: 에이전트 0-{num_helicopters-1}")
        print(f"  - Truck 정책: 에이전트 {num_helicopters}-{num_helicopters+num_trucks-1}")
        if num_crews > 0:
            print(f"  - Crew 정책: 에이전트 {num_helicopters+num_trucks}-{ENV_CONFIG['num_agents']-1}")
    else:
        share_policy = "all"  # 모든 에이전트가 같은 정책
        print(f"\n정책 공유 방식: {share_policy} (모든 에이전트)")

    # Step 6: 결과 저장 디렉토리 설정
    # train_marllib/results 디렉토리에 저장
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        exp_name
    )
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("학습 시작...")
    print("(Ctrl+C로 중단 가능)")
    print("=" * 60)
    print(f"📁 결과 저장: {results_dir}")
    print("=" * 60 + "\n")

    # Ray 초기화는 MARLlib.fit()이 자동으로 처리하도록 함
    # 직접 초기화하면 Redis 문제 발생 가능

    # Iteration 기반 학습
    # MARLlib의 fit()은 내부적으로 iteration을 관리
    # training_iteration으로 중단 조건 설정
    try:
        mappo.fit(
            env=(env, env_config_dict),
            model=model,
            stop={
                'training_iteration': num_iterations,  # iteration 기반
            },
            share_policy=share_policy,
            local_mode=False,
            num_gpus=num_gpus,
            num_workers=num_workers,
            checkpoint_freq=checkpoint_freq,
            checkpoint_end=True,
            restore_path=restore_path,
            # 결과 저장 디렉토리 지정
            local_dir=results_dir,
            # 추가 학습 설정
            config={
                "lr": 5e-4,
                "gamma": 0.95,
                "clip_param": 0.3,
                "train_batch_size": 1024,
                "sgd_minibatch_size": 128,
                "num_sgd_iter": 10,
                "framework": "torch",
            }
        )

        print("\n" + "=" * 60)
        print("🎉 학습 완료!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⚠️  학습이 사용자에 의해 중단되었습니다.")
        print("=" * 60)
        if ray.is_initialized():
            ray.shutdown()

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 학습 중 오류 발생: {e}")
        print("=" * 60)
        if ray.is_initialized():
            ray.shutdown()
        raise

    finally:
        # Ray 정리
        if ray.is_initialized():
            print("\nRay 종료 중...")
            ray.shutdown()
            print("✓ Ray 종료 완료")

    # 결과 위치 안내
    print(f"\n📁 결과 저장 위치:")
    print(f"  {results_dir}")
    print("\n다음 단계:")
    print("1. TensorBoard로 학습 곡선 확인:")
    print(f"   tensorboard --logdir {results_dir}")
    print("2. 체크포인트로 시뮬레이션 실행")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MARLlib MAPPO (CTDE) 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 100 iterations 학습
  python train_rllib/train_marllib_mappo.py --iterations 100

  # GPU 사용
  python train_rllib/train_marllib_mappo.py --iterations 1000 --num-gpus 1

  # 체크포인트에서 재개
  python train_rllib/train_marllib_mappo.py --iterations 500 --restore ~/ray_results/mappo_wildfire/checkpoint_000100
        """
    )
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
        help="체크포인트 저장 빈도 (기본값: 10 iterations)"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="mappo_wildfire",
        help="실험 이름 (기본값: mappo_wildfire)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="병렬 worker 수 (기본값: 2)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="사용할 GPU 수 (기본값: 0, CPU만 사용)"
    )
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="재개할 체크포인트 경로"
    )

    args = parser.parse_args()

    train_marllib_mappo(
        num_iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        exp_name=args.exp_name,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        restore_path=args.restore,
    )
