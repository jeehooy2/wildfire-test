"""
학습된 MAPPO 모델과 랜덤 에이전트 비교 평가

checkpoint_000200에서 모델을 로드하여 성능 평가
"""

import sys
import os
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from train_marllib_self.new_wrapper import WildfireRLlibEnv
from train_marllib_self.environment import ENV_CONFIG
from ray.rllib.agents.ppo import PPOTrainer
import matplotlib
matplotlib.use('Agg')  # 백엔드를 non-interactive로 설정
import matplotlib.pyplot as plt


def evaluate_agent(env, policy=None, num_episodes=10, agent_type="trained"):
    """
    에이전트 평가

    Parameters
    ----------
    env : WildfireRLlibEnv
        평가할 환경
    policy : Policy or None
        학습된 policy (None이면 랜덤)
    num_episodes : int
        평가할 에피소드 수
    agent_type : str
        "trained" 또는 "random"

    Returns
    -------
    dict
        평가 통계
    """

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # 액션 선택
            if policy is None:
                # 랜덤 액션
                actions = {i: env._single_action_space.sample() for i in range(env.num_agents)}
            else:
                # 학습된 policy 사용
                actions = {}
                for agent_id in range(env.num_agents):
                    action = policy.compute_single_action(
                        obs[agent_id],
                        policy_id="shared_policy"
                    )
                    actions[agent_id] = action

            # 스텝 실행
            obs, rewards, dones, infos = env.step(actions)

            # 리워드 합산 (모든 에이전트의 평균)
            episode_reward += np.mean(list(rewards.values()))
            episode_length += 1
            done = dones["__all__"]

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"{agent_type.capitalize()} Agent - Episode {ep+1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {episode_length}")

    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "all_rewards": episode_rewards,
        "all_lengths": episode_lengths
    }

    return stats


def compare_agents(checkpoint_path, num_episodes=10):
    """
    학습된 에이전트와 랜덤 에이전트 비교
    """

    print("=" * 80)
    print("MAPPO Model Evaluation - Trained vs Random")
    print("=" * 80)

    # 환경 등록
    ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnv
    COOP_ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnv

    # 환경 생성 (평가용)
    print("\n환경 초기화 중...")
    from wildfire_environment.envs import WildfireEnv
    env_config = {k: v for k, v in ENV_CONFIG.items() if k != 'map_name'}
    if 'agent_start_positions' in env_config and isinstance(env_config['agent_start_positions'], str):
        import ast
        env_config['agent_start_positions'] = ast.literal_eval(env_config['agent_start_positions'])
    if 'reward_shaping_config' in env_config and env_config['reward_shaping_config'] == 'None':
        env_config['reward_shaping_config'] = None

    eval_env = WildfireRLlibEnv(env_config)

    # 체크포인트에서 모델 로드
    print(f"\n체크포인트 로드 중: {checkpoint_path}")

    try:
        import ray
        ray.init(local_mode=True, ignore_reinit_error=True)

        from marllib.marl.algos.core.CC.mappo import MAPPOTrainer

        # Trainer 복원 (RLlib 방식)
        config = {
            "framework": "torch",
            "num_workers": 0,  # 평가용이므로 worker 불필요
            "num_gpus": 0,
        }

        trainer = MAPPOTrainer(config=config)
        trainer.restore(checkpoint_path)
        policy = trainer.get_policy("shared_policy")

        print("✓ 모델 로드 완료!")

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("\n대안: 통계 파일에서 정보 읽기...")
        import traceback
        traceback.print_exc()
        return

    # 1. 학습된 에이전트 평가
    print("\n" + "=" * 80)
    print("학습된 에이전트 평가")
    print("=" * 80)
    trained_stats = evaluate_agent(eval_env, policy, num_episodes, "trained")

    # 2. 랜덤 에이전트 평가
    print("\n" + "=" * 80)
    print("랜덤 에이전트 평가")
    print("=" * 80)
    random_stats = evaluate_agent(eval_env, None, num_episodes, "random")

    # 3. 결과 비교
    print("\n" + "=" * 80)
    print("평가 결과 비교")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Trained':<20} {'Random':<20} {'Improvement':<15}")
    print("-" * 85)

    # 평균 리워드
    improvement = ((trained_stats['mean_reward'] - random_stats['mean_reward']) /
                   abs(random_stats['mean_reward']) * 100 if random_stats['mean_reward'] != 0 else float('inf'))
    print(f"{'Mean Reward':<30} {trained_stats['mean_reward']:>8.2f} ± {trained_stats['std_reward']:<8.2f} "
          f"{random_stats['mean_reward']:>8.2f} ± {random_stats['std_reward']:<8.2f} "
          f"{improvement:>+6.1f}%")

    # 평균 에피소드 길이
    length_diff = random_stats['mean_length'] - trained_stats['mean_length']
    print(f"{'Mean Episode Length':<30} {trained_stats['mean_length']:>8.1f} ± {trained_stats['std_length']:<8.1f} "
          f"{random_stats['mean_length']:>8.1f} ± {random_stats['std_length']:<8.1f} "
          f"{length_diff:>+6.1f}")

    # 4. 시각화
    print("\n그래프 생성 중...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 리워드 비교
    axes[0].boxplot([trained_stats['all_rewards'], random_stats['all_rewards']],
                     labels=['Trained', 'Random'])
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Episode Rewards Comparison')
    axes[0].grid(True, alpha=0.3)

    # 에피소드 길이 비교
    axes[1].boxplot([trained_stats['all_lengths'], random_stats['all_lengths']],
                     labels=['Trained', 'Random'])
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Episode Length Comparison')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    output_path = Path(checkpoint_path).parent / "evaluation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 그래프 저장: {output_path}")

    # 통계 저장
    stats_path = Path(checkpoint_path).parent / "evaluation_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MAPPO Model Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Episodes: {num_episodes}\n\n")

        f.write("Trained Agent:\n")
        f.write(f"  Mean Reward: {trained_stats['mean_reward']:.2f} ± {trained_stats['std_reward']:.2f}\n")
        f.write(f"  Mean Length: {trained_stats['mean_length']:.1f} ± {trained_stats['std_length']:.1f}\n\n")

        f.write("Random Agent:\n")
        f.write(f"  Mean Reward: {random_stats['mean_reward']:.2f} ± {random_stats['std_reward']:.2f}\n")
        f.write(f"  Mean Length: {random_stats['mean_length']:.1f} ± {random_stats['std_length']:.1f}\n\n")

        f.write(f"Improvement: {improvement:+.1f}%\n")

    print(f"✓ 통계 저장: {stats_path}")

    print("\n평가 완료!")


if __name__ == '__main__':
    # 체크포인트 경로
    checkpoint_path = "/home/bmkim88/wildfire_environment/exp_results/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_e09f3_00000_0_2025-11-15_15-58-07/checkpoint_000200"

    # 평가 실행 (10 에피소드)
    compare_agents(checkpoint_path, num_episodes=10)
