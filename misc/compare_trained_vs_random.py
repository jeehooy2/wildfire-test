"""
학습된 정책과 랜덤 정책 비교

이 스크립트는 학습된 RLlib 모델과 랜덤 정책을 비교하여
학습 효과를 검증합니다.
"""

import os
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from misc.wildfire_rllib_wrapper import WildfireRLlibEnv


def evaluate_policy(algo, env_config, num_episodes=10, policy_type="trained", seed=42):
    """
    정책을 평가하고 통계를 반환

    Parameters
    ----------
    algo : Algorithm or None
        RLlib 알고리즘 (None이면 랜덤 정책)
    env_config : dict
        환경 설정
    num_episodes : int
        평가할 에피소드 수
    policy_type : str
        정책 타입 (trained or random)
    seed : int
        랜덤 시드

    Returns
    -------
    dict
        평가 결과 통계
    """
    # 환경 생성
    env = WildfireRLlibEnv(env_config)

    episode_rewards = []
    episode_lengths = []
    agent_rewards = {i: [] for i in range(env.num_agents)}

    for ep in range(num_episodes):
        # seed 설정 (재현 가능하도록)
        episode_seed = seed + ep
        obs, info = env.reset(seed=episode_seed)

        episode_reward = 0
        episode_length = 0
        agent_episode_rewards = {i: 0 for i in range(env.num_agents)}

        done = False
        while not done:
            if policy_type == "trained" and algo is not None:
                # 학습된 정책 사용 (새로운 RLlib API)
                import torch

                # RLModule 가져오기
                rl_module = algo.get_module("shared_policy")

                # 관찰을 배치로 변환
                obs_batch = {
                    "obs": torch.tensor(np.array([obs[i] for i in range(env.num_agents)]), dtype=torch.float32)
                }

                # forward_inference로 액션 계산
                with torch.no_grad():
                    output = rl_module.forward_inference(obs_batch)
                    # output 구조 확인 - 일반적으로 "action_dist_inputs"나 직접 액션이 있음
                    if "action_dist" in output:
                        action_dist = output["action_dist"]
                        action_tensor = action_dist.mode() if hasattr(action_dist, 'mode') else action_dist.sample()
                    elif "actions" in output:
                        action_tensor = output["actions"]
                    else:
                        # action_dist_inputs로부터 액션 계산
                        from ray.rllib.models.torch.torch_distributions import TorchCategorical
                        logits = output.get("action_dist_inputs", output)
                        if isinstance(logits, dict):
                            logits = logits.get("logits", list(logits.values())[0])
                        action_dist = TorchCategorical(logits=logits)
                        action_tensor = action_dist.mode() if hasattr(action_dist, 'mode') else torch.argmax(logits, dim=-1)

                    actions_array = action_tensor.cpu().numpy()

                # 에이전트별로 액션 분리
                actions = {i: int(actions_array[i]) for i in range(env.num_agents)}
            else:
                # 랜덤 정책 사용
                actions = {i: env._action_space[i].sample() for i in range(env.num_agents)}

            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            # 보상 누적
            for agent_id in range(env.num_agents):
                agent_episode_rewards[agent_id] += rewards[agent_id]
            episode_reward += sum(rewards.values())
            episode_length += 1

            # 종료 확인
            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        for agent_id in range(env.num_agents):
            agent_rewards[agent_id].append(agent_episode_rewards[agent_id])

        print(f"  Episode {ep+1}/{num_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")

    env.close()

    # 통계 계산
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "agent_mean_rewards": {i: np.mean(agent_rewards[i]) for i in range(env.num_agents)},
        "episode_rewards": episode_rewards,
    }

    return stats


def main(checkpoint_path, num_episodes=10, seed=42):
    """
    메인 함수: 학습된 모델과 랜덤 정책을 비교

    Parameters
    ----------
    checkpoint_path : str
        학습된 모델의 체크포인트 경로
    num_episodes : int
        각 정책당 평가할 에피소드 수
    seed : int
        랜덤 시드 (재현성을 위해)
    """
    print("=" * 70)
    print("학습된 정책 vs 랜덤 정책 비교")
    print("=" * 70)
    print(f"체크포인트: {checkpoint_path}")
    print(f"에피소드 수: {num_episodes}")
    print(f"시드: {seed}")
    print("=" * 70)

    # Ray 초기화
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 환경 등록
    def env_creator(env_config):
        return WildfireRLlibEnv(env_config)

    register_env("wildfire-ma", env_creator)

    # 환경 설정 (학습 시와 동일하게)
    env_config = {
        "num_agents": 2,
        "size": 17,
        "initial_fire_size": 3,
        "cooperative_reward": False,
        "max_steps": 300,
        "agent_start_positions": ((1, 1), (15, 15)),
        "log_selfish_region_metrics": True,
        "selfish_region_xmin": [7, 13],
        "selfish_region_xmax": [9, 15],
        "selfish_region_ymin": [7, 1],
        "selfish_region_ymax": [9, 3],
        "delta_beta": 0.7,
        "beta": 0.99,
        "alpha": 0.05,
    }

    # 학습된 모델 로드
    print("\n학습된 모델 로딩 중...")
    checkpoint_path = os.path.abspath(checkpoint_path)
    algo = PPO.from_checkpoint(checkpoint_path)
    print("✓ 모델 로드 완료")

    # 1. 학습된 정책 평가
    print("\n" + "=" * 70)
    print("1. 학습된 정책 평가")
    print("=" * 70)
    trained_stats = evaluate_policy(
        algo, env_config, num_episodes=num_episodes,
        policy_type="trained", seed=seed
    )

    # 2. 랜덤 정책 평가
    print("\n" + "=" * 70)
    print("2. 랜덤 정책 평가")
    print("=" * 70)
    random_stats = evaluate_policy(
        None, env_config, num_episodes=num_episodes,
        policy_type="random", seed=seed
    )

    # 결과 비교
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)

    print(f"\n평균 보상:")
    print(f"  학습된 정책: {trained_stats['mean_reward']:.2f} ± {trained_stats['std_reward']:.2f}")
    print(f"  랜덤 정책:   {random_stats['mean_reward']:.2f} ± {random_stats['std_reward']:.2f}")
    print(f"  개선도:      {trained_stats['mean_reward'] - random_stats['mean_reward']:.2f} "
          f"({((trained_stats['mean_reward'] - random_stats['mean_reward']) / abs(random_stats['mean_reward']) * 100):.1f}%)")

    print(f"\n보상 범위:")
    print(f"  학습된 정책: [{trained_stats['min_reward']:.2f}, {trained_stats['max_reward']:.2f}]")
    print(f"  랜덤 정책:   [{random_stats['min_reward']:.2f}, {random_stats['max_reward']:.2f}]")

    print(f"\n평균 에피소드 길이:")
    print(f"  학습된 정책: {trained_stats['mean_length']:.1f}")
    print(f"  랜덤 정책:   {random_stats['mean_length']:.1f}")

    print(f"\n에이전트별 평균 보상:")
    for agent_id in range(len(trained_stats['agent_mean_rewards'])):
        print(f"  Agent {agent_id}:")
        print(f"    학습된 정책: {trained_stats['agent_mean_rewards'][agent_id]:.2f}")
        print(f"    랜덤 정책:   {random_stats['agent_mean_rewards'][agent_id]:.2f}")
        print(f"    개선도:      {trained_stats['agent_mean_rewards'][agent_id] - random_stats['agent_mean_rewards'][agent_id]:.2f}")

    # 에피소드별 보상 비교
    print(f"\n에피소드별 보상 비교:")
    print(f"  {'Episode':<10} {'Trained':<15} {'Random':<15} {'Difference':<15}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    for i in range(num_episodes):
        trained_r = trained_stats['episode_rewards'][i]
        random_r = random_stats['episode_rewards'][i]
        diff = trained_r - random_r
        print(f"  {i+1:<10} {trained_r:<15.2f} {random_r:<15.2f} {diff:<15.2f}")

    print("\n" + "=" * 70)

    if trained_stats['mean_reward'] > random_stats['mean_reward']:
        improvement = trained_stats['mean_reward'] - random_stats['mean_reward']
        print(f"✓ 학습이 성공적입니다! 평균 {improvement:.2f}의 보상 개선")
    else:
        print("✗ 학습된 정책이 랜덤 정책보다 성능이 낮습니다.")

    print("=" * 70)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="학습된 정책과 랜덤 정책 비교")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./rllib_checkpoints/best",
        help="학습된 모델 체크포인트 경로"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="각 정책당 평가할 에피소드 수"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (재현성을 위해)"
    )

    args = parser.parse_args()

    main(args.checkpoint, args.episodes, args.seed)
