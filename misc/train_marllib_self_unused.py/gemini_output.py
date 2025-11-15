"""
MARLLib를 사용한 다중 에이전트 MAPPO 학습

이 스크립트는 Ray MARLLib를 사용하여 wildfire 환경에서
다중 에이전트를 학습시킵니다.

RLlib PPOConfig 대신 MARLLib의 API를 사용하도록 변환되었습니다.

설치 필요:
pip install "ray[rllib]>=2.7.0" torch
pip install git+https://github.com/Replicable-MARL/MARLlib.git

실행:
python gemini_output.py --iterations 10 --checkpoint-freq 10 --save-dir ./train_marllib/experiments/run_marllib

"""

import os
import warnings
import ray
import logging
from marllib import marl  # PPOConfig 대신 MARLLib 임포트
from ray.tune.registry import register_env
from marllib.envs.base_env import ENV_REGISTRY
from train_marllib_self.new_wrapper import WildfireRLlibEnv
from train_marllib_self.environment import ENV_CONFIG

# 모든 deprecation 경고 억제
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Ray 및 RLlib 로깅 레벨 설정
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)


def train_marllib_multiagent(
    num_iterations=100,
    checkpoint_freq=10,
    save_dir="./train_rllib/experiments"
):
    """
    MARLLib MAPPO로 다중 에이전트 학습

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
    
    # MARLLib는 Ray를 내부적으로 초기화하지만, 명시적으로 호출하여 로깅 레벨 설정
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            _metrics_export_port=None,
            _system_config={
                "metrics_report_interval_ms": 0,
            }
        )

    # --- 1. 환경 등록 (기존과 동일) ---
    def env_creator(env_config):
        return WildfireRLlibEnv(env_config)

    register_env("wildfire-ma", env_creator)
    env_name = "wildfire-ma"

    # 환경 설정 (train_rllib/environment.py의 ENV_CONFIG 사용)
    env_config = ENV_CONFIG

    # --- 2. MARLLib를 위한 환경 정보 추출 (핵심 변경 사항) ---
    # MARLLib는 설정 시 observation/action space가 명시적으로 필요합니다.
    test_env = env_creator(env_config)
    num_agents = test_env.num_agents
    
    # RLlib MultiAgentEnv는 .observation_space, .action_space 속성이
    # {agent_id: space} 형태의 딕셔너리를 반환합니다.
    observation_spaces = test_env.observation_space
    action_spaces = test_env.action_space
    test_env.close()

    # TODO
    ENV_REGISTRY["ma-wildfire"] = WildfireRLlibEnv
    # initialize env
    # config_path = os.path.join(os.path.dirname(__file__), "envs/base_env/config/ma-wildfire.yaml")
    env = marl.make_env(environment_name="ma-wildfire", map_name="wildfire")

    # env = (WildfireRLlibEnv(env_config), env_config)

    # 이질적 에이전트 설정 추출
    num_helicopters = env_config.get("num_helicopters", 0)
    num_trucks = env_config.get("num_trucks", 0)
    num_crews = env_config.get("num_crews", 0)

    print("=" * 60)
    print("MARLLib 다중 에이전트 MAPPO 학습")
    print("=" * 60)
    print(f"에이전트 수: {num_agents}")
    print(f"학습 반복 횟수: {num_iterations}")
    print(f"체크포인트 저장: {save_dir}")
    print("=" * 60)

    # --- 3. MARLLib 정책 매핑 정보 생성 (핵심 변경 사항) ---
    
    # 기존 policy_mapping_fn
    def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id < num_helicopters:
            return "helicopter_policy"
        elif agent_id < num_helicopters + num_trucks:
            return "truck_policy"
        else:
            return "crew_policy"

    # MARLLib의 'policies' 딕셔너리 생성
    # 각 정책이 사용할 Observation/Action Space를 지정해야 함
    policies = {}
    if num_helicopters > 0:
        # 헬리콥터 정책 (0번 에이전트의 space 사용)
        policies["helicopter_policy"] = (None, observation_spaces[0], action_spaces[0], {})
    if num_trucks > 0:
        # 트럭 정책 (첫 번째 트럭 에이전트의 space 사용)
        policies["truck_policy"] = (None, observation_spaces[num_helicopters], action_spaces[num_helicopters], {})
    if num_crews > 0:
        # 크루 정책 (첫 번째 크루 에이전트의 space 사용)
        policies["crew_policy"] = (None, observation_spaces[num_helicopters + num_trucks], action_spaces[num_helicopters + num_trucks], {})

    # MARLLib가 요구하는 최종 policy_mapping_info
    policy_mapping_info = {
        "policy_mapping_fn": policy_mapping_fn,
        "policies": policies
    }
    
    # MARLLib 알고리즘 빌더에 전달할 환경 정보
    marllib_env_config = {
        "env_name": env_name,
        "env_config": env_config,
        "policy_mapping_info": policy_mapping_info,
        "observation_spaces": observation_spaces,
        "action_spaces": action_spaces
    }

    # --- 4. MARLLib 알고리즘 설정 (핵심 변경 사항) ---
    # PPOConfig 빌더 대신 marl.algos.mappo() 사용
    config = marl.algos.mappo(hyperparam_source="common")
    # customize model
    # model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

    # config = marl.algos.mappo(
    #     marllib_env_config,  # 위에서 정의한 환경/정책 정보
        
    #     # 하이퍼파라미터 (기존 PPOConfig 값 매핑)
    #     framework="torch",
    #     lr=5e-4,
    #     gamma=0.95,
    #     clip_param=0.3,
        
    #     # RLlib 2.x의 'train_batch_size_per_learner'는 1.x의 'train_batch_size'와 다릅니다.
    #     # MARLLib는 'train_batch_size' (총 스텝 수)를 사용합니다. 4000은 일반적인 값입니다.
    #     train_batch_size=4000, 
        
    #     num_workers=2,  # PPOConfig의 num_env_runners
    #     num_envs_per_worker=2, # PPOConfig의 num_envs_per_env_runner
    #     num_gpus=0
    # )
    

    # --- 5. MARLLib 학습 실행 (핵심 변경 사항) ---
    # 수동 학습 루프 대신 marl.run() 사용

    # 중지 기준 (iterations)
    stop = {
        "training_iteration": num_iterations
    }

    print("\nMARLLib 학습 시작...")
    print("(Ctrl+C로 중단 가능)\n")

    try:
        results = marl.fit(
            env,
            model,
            stop=stop,
            local_dir=save_dir,  # 체크포인트 저장 위치
            name="mappo_wildfire_run", # Ray Tune 실험 이름
            checkpoint_freq=checkpoint_freq, # 체크포인트 저장 빈도
            verbose=1 # 0(조용히), 1(결과 요약), 2(상세)
        )

        
        best_logdir = results.get_best_logdir()
        print("\n" + "=" * 60)
        print(f"최종 모델 저장 위치 (Best): {best_logdir}")
        print("=" * 60)
        
        final_checkpoint_path = best_logdir # MARLLib는 가장 좋은 체크포인트를 반환

    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
        final_checkpoint_path = None
    
    finally:
        ray.shutdown()

    return final_checkpoint_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARLLib 다중 에이전트 학습")
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
        default="./train_rllib/experiments",
        help="체크포인트 저장 디렉토리"
    )

    args = parser.parse_args()

    checkpoint_path = train_marllib_multiagent(
        num_iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        save_dir=args.save_dir
    )

    if checkpoint_path:
        print("\n학습 완료!")
        print(f"모델 위치: {checkpoint_path}")
        print("\n다음 명령으로 시뮬레이션 실행:")
        print(f"python simulate_trained_model.py --checkpoint {checkpoint_path}")
    else:
        print("\n학습이 완료되지 않았거나 중단되었습니다.")