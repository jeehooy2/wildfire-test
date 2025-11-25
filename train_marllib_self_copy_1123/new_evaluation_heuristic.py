"""
Evaluation script for heuristic policies on wildfire suppression task

new_evaluation.py와 동일한 평가 항목을 다음 2가지 heuristic에 대해 계산합니다:
1. Nearest fire greedy 로직 (heuristic_nearest_fire.py)
2. Random policy (각 에이전트가 무작위 행동 수행)

각 에피소드에 대해 다음 지표들을 계산합니다:
  1. 평균 healthy (green) trees 비율
  2. 최종 피해 면적 (burnt trees 비율)
  3. 총 진화 시간 (에피소드 길이 in steps)

실행 예시:
python train_marllib_self/new_evaluation_heuristic.py \\
    --episodes 10 \\
    --seed 42

# 커스텀 출력 디렉토리
python train_marllib_self/new_evaluation_heuristic.py \\
    --episodes 20 \\
    --seed 100 \\
    --output-dir ./evaluation_results/

# 부분 관찰 모드로 평가하려면 environment.py에서 partial_obs=True로 설정하세요
"""

import sys
import os
from pathlib import Path

# Set up project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import json
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from train_marllib_self.heuristic_nearest_fire import select_heuristic_action


# ============================================================================
# Episode Evaluation Functions
# ============================================================================

def run_nearest_fire_episode(env, env_config, seed, max_steps=300):
    """
    Run a single evaluation episode with nearest fire greedy heuristic
    and collect metrics

    Returns:
        metrics: dict with keys:
            - healthy_trees_ratio: Average healthy trees ratio
            - burnt_trees_ratio: Average burnt trees ratio (damage)
            - episode_length: Number of steps taken
            - healthy_counts: List of healthy tree counts per step
            - burnt_counts: List of burnt tree counts per step
    """
    # Reset environment
    obs_dict, _ = env.reset(seed=seed)

    partial_obs = env_config.get('partial_obs', False)
    done = False
    step = 0

    healthy_counts = []
    burnt_counts = []

    while not done and step < max_steps:
        # Get current tree state
        grid = env.grid

        # Count healthy (green) and burnt (brown) trees
        healthy_count = 0
        burnt_count = 0
        total_trees = 0

        for i in range(grid.height):
            for j in range(grid.width):
                cell = grid.get(j, i)
                if cell is not None and cell.type == "tree":
                    total_trees += 1
                    if cell.state == 0:  # Green/healthy tree
                        healthy_count += 1
                    elif cell.state == 2:  # Brown/burnt tree
                        burnt_count += 1

        healthy_counts.append(healthy_count)
        burnt_counts.append(burnt_count)

        # Select actions using nearest fire greedy heuristic
        actions = {}
        for agent_id in obs_dict.keys():
            obs = obs_dict[agent_id]
            action = select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs)
            actions[agent_id] = action

        # Step environment
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated
        step += 1

    # Calculate metrics at episode end
    grid = env.grid
    healthy_count = 0
    burnt_count = 0
    total_trees = 0

    for i in range(grid.height):
        for j in range(grid.width):
            cell = grid.get(j, i)
            if cell is not None and cell.type == "tree":
                total_trees += 1
                if cell.state == 0:
                    healthy_count += 1
                elif cell.state == 2:
                    burnt_count += 1

    # Compute ratios
    if total_trees > 0:
        healthy_ratio = healthy_count / total_trees
        burnt_ratio = burnt_count / total_trees
    else:
        healthy_ratio = 0.0
        burnt_ratio = 0.0

    return {
        'healthy_trees_ratio': healthy_ratio,
        'burnt_trees_ratio': burnt_ratio,
        'episode_length': step,
        'healthy_counts': healthy_counts,
        'burnt_counts': burnt_counts,
    }


def run_random_policy_episode(env, env_config, seed, max_steps=300):
    """
    Run a single evaluation episode with random policy
    and collect metrics

    Returns:
        metrics: dict (same structure as run_nearest_fire_episode)
    """
    # Reset environment
    obs_dict, _ = env.reset(seed=seed)

    done = False
    step = 0

    healthy_counts = []
    burnt_counts = []

    while not done and step < max_steps:
        # Get current tree state
        grid = env.grid

        # Count healthy (green) and burnt (brown) trees
        healthy_count = 0
        burnt_count = 0
        total_trees = 0

        for i in range(grid.height):
            for j in range(grid.width):
                cell = grid.get(j, i)
                if cell is not None and cell.type == "tree":
                    total_trees += 1
                    if cell.state == 0:  # Green/healthy tree
                        healthy_count += 1
                    elif cell.state == 2:  # Brown/burnt tree
                        burnt_count += 1

        healthy_counts.append(healthy_count)
        burnt_counts.append(burnt_count)

        # Select random actions for all agents
        actions = {}
        for agent_id in obs_dict.keys():
            action = env.action_space[agent_id].sample()
            actions[agent_id] = action

        # Step environment
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated
        step += 1

    # Calculate metrics at episode end
    grid = env.grid
    healthy_count = 0
    burnt_count = 0
    total_trees = 0

    for i in range(grid.height):
        for j in range(grid.width):
            cell = grid.get(j, i)
            if cell is not None and cell.type == "tree":
                total_trees += 1
                if cell.state == 0:
                    healthy_count += 1
                elif cell.state == 2:
                    burnt_count += 1

    # Compute ratios
    if total_trees > 0:
        healthy_ratio = healthy_count / total_trees
        burnt_ratio = burnt_count / total_trees
    else:
        healthy_ratio = 0.0
        burnt_ratio = 0.0

    return {
        'healthy_trees_ratio': healthy_ratio,
        'burnt_trees_ratio': burnt_ratio,
        'episode_length': step,
        'healthy_counts': healthy_counts,
        'burnt_counts': burnt_counts,
    }


# ============================================================================
# Evaluation Summary Function
# ============================================================================

def print_evaluation_summary(policy_name, metrics_list):
    """Print evaluation results summary for a policy"""
    healthy_ratios = [m['healthy_trees_ratio'] for m in metrics_list]
    burnt_ratios = [m['burnt_trees_ratio'] for m in metrics_list]
    episode_lengths = [m['episode_length'] for m in metrics_list]

    print(f"\n{'='*80}")
    print(f"{policy_name} - Evaluation Results")
    print(f"{'='*80}")

    print(f"\nHealthy Trees Ratio (Avg ± Std):")
    print(f"  Mean: {np.mean(healthy_ratios)*100:.2f}% ± {np.std(healthy_ratios)*100:.2f}%")
    print(f"  Min:  {np.min(healthy_ratios)*100:.2f}%")
    print(f"  Max:  {np.max(healthy_ratios)*100:.2f}%")

    print(f"\nBurnt Trees / Damage Area (Avg ± Std):")
    print(f"  Mean: {np.mean(burnt_ratios)*100:.2f}% ± {np.std(burnt_ratios)*100:.2f}%")
    print(f"  Min:  {np.min(burnt_ratios)*100:.2f}%")
    print(f"  Max:  {np.max(burnt_ratios)*100:.2f}%")

    print(f"\nExtinguishing Time (Avg ± Std):")
    print(f"  Mean: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"  Min:  {np.min(episode_lengths):.1f} steps")
    print(f"  Max:  {np.max(episode_lengths):.1f} steps")

    return {
        'healthy_trees_mean': float(np.mean(healthy_ratios)),
        'healthy_trees_std': float(np.std(healthy_ratios)),
        'burnt_trees_mean': float(np.mean(burnt_ratios)),
        'burnt_trees_std': float(np.std(burnt_ratios)),
        'episode_length_mean': float(np.mean(episode_lengths)),
        'episode_length_std': float(np.std(episode_lengths)),
    }


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main(num_episodes=10, seed=42, output_dir=None):
    """
    Evaluate heuristic policies on wildfire suppression task

    Parameters:
        num_episodes: Number of evaluation episodes for each policy
        seed: Random seed for reproducibility
        output_dir: Output directory for results

    Note:
        partial_obs는 environment.py의 ENV_CONFIG['partial_obs']에서 읽어옵니다.
    """

    print("=" * 80)
    print("Heuristic Policies - Wildfire Suppression Task Evaluation")
    print("=" * 80)

    # Environment configuration (partial_obs는 ENV_CONFIG에서 직접 읽어옴)
    env_config = {k: v for k, v in ENV_CONFIG.items()}

    # Create output directory
    if output_dir is None:
        output_dir = "train_marllib_self/evaluation/heuristic"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n설정:")
    print(f"  - 에피소드 수: {num_episodes}")
    print(f"  - 시작 시드: {seed}")
    print(f"  - 부분 관찰: {env_config['partial_obs']}")
    print(f"  - 그리드 크기: {env_config['size']}x{env_config['size']}")
    print(f"  - 에이전트 수: {env_config['num_agents']}")
    print(f"  - 최대 스텝: {env_config['max_steps']}")
    print(f"  - 출력 디렉토리: {output_dir}")

    # Create environments for each policy
    print("\n환경 생성 중...")
    env_nearest_fire = WildfireEnv(**env_config)
    env_random = WildfireEnv(**env_config)

    print(f"✓ 환경 생성 완료")

    # ============================================================================
    # Evaluate Nearest Fire Greedy Policy
    # ============================================================================
    print(f"\nNearest Fire Greedy 정책 평가")
    print("-" * 80)

    nearest_fire_metrics = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"Episode {ep+1}/{num_episodes} (seed={episode_seed})...", end="", flush=True)

        metrics = run_nearest_fire_episode(
            env_nearest_fire, env_config, episode_seed,
            max_steps=env_config['max_steps']
        )
        nearest_fire_metrics.append(metrics)

        print(f" ✓")
        print(f"  - Healthy: {metrics['healthy_trees_ratio']*100:.1f}%")
        print(f"  - Burnt:   {metrics['burnt_trees_ratio']*100:.1f}%")
        print(f"  - Length:  {metrics['episode_length']} steps")

    nearest_fire_summary = print_evaluation_summary("Nearest Fire Greedy", nearest_fire_metrics)

    # ============================================================================
    # Evaluate Random Policy
    # ============================================================================
    print(f"\nRandom 정책 평가")
    print("-" * 80)

    random_metrics = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"Episode {ep+1}/{num_episodes} (seed={episode_seed})...", end="", flush=True)

        metrics = run_random_policy_episode(
            env_random, env_config, episode_seed,
            max_steps=env_config['max_steps']
        )
        random_metrics.append(metrics)

        print(f" ✓")
        print(f"  - Healthy: {metrics['healthy_trees_ratio']*100:.1f}%")
        print(f"  - Burnt:   {metrics['burnt_trees_ratio']*100:.1f}%")
        print(f"  - Length:  {metrics['episode_length']} steps")

    random_summary = print_evaluation_summary("Random", random_metrics)

    # ============================================================================
    # Save Results
    # ============================================================================
    partial_obs_value = env_config['partial_obs']
    results_file = Path(output_dir) / f"evaluation_results_partial_obs_{partial_obs_value}.json"
    report_file = Path(output_dir) / f"evaluation_report_partial_obs_{partial_obs_value}.txt"

    results = {
        'num_episodes': num_episodes,
        'seed': seed,
        'environment_config': env_config,
        'nearest_fire_greedy': {
            'metrics': nearest_fire_metrics,
            'summary': nearest_fire_summary,
        },
        'random': {
            'metrics': random_metrics,
            'summary': random_summary,
        },
    }

    # Save to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Save detailed report
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Heuristic Policies - Wildfire Suppression Task Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Environment Configuration:\n")
        for key, value in env_config.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Nearest Fire Greedy Policy Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Healthy Trees Ratio (%):\n")
        healthy_ratios = [m['healthy_trees_ratio'] for m in nearest_fire_metrics]
        f.write(f"  Mean: {np.mean(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Std:  {np.std(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Min:  {np.min(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Max:  {np.max(healthy_ratios)*100:.2f}%\n\n")

        f.write("Burnt Trees Ratio (%) - Damage Area:\n")
        burnt_ratios = [m['burnt_trees_ratio'] for m in nearest_fire_metrics]
        f.write(f"  Mean: {np.mean(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Std:  {np.std(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Min:  {np.min(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Max:  {np.max(burnt_ratios)*100:.2f}%\n\n")

        f.write("Episode Length (steps):\n")
        episode_lengths = [m['episode_length'] for m in nearest_fire_metrics]
        f.write(f"  Mean: {np.mean(episode_lengths):.1f}\n")
        f.write(f"  Std:  {np.std(episode_lengths):.1f}\n")
        f.write(f"  Min:  {np.min(episode_lengths):.1f}\n")
        f.write(f"  Max:  {np.max(episode_lengths):.1f}\n\n")

        f.write("Per-Episode Details:\n")
        f.write("-" * 80 + "\n")
        for i, metrics in enumerate(nearest_fire_metrics):
            f.write(f"Episode {i+1} (seed={seed+i}):\n")
            f.write(f"  Healthy: {metrics['healthy_trees_ratio']*100:.2f}%\n")
            f.write(f"  Burnt:   {metrics['burnt_trees_ratio']*100:.2f}%\n")
            f.write(f"  Length:  {metrics['episode_length']} steps\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Random Policy Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Healthy Trees Ratio (%):\n")
        healthy_ratios = [m['healthy_trees_ratio'] for m in random_metrics]
        f.write(f"  Mean: {np.mean(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Std:  {np.std(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Min:  {np.min(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Max:  {np.max(healthy_ratios)*100:.2f}%\n\n")

        f.write("Burnt Trees Ratio (%) - Damage Area:\n")
        burnt_ratios = [m['burnt_trees_ratio'] for m in random_metrics]
        f.write(f"  Mean: {np.mean(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Std:  {np.std(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Min:  {np.min(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Max:  {np.max(burnt_ratios)*100:.2f}%\n\n")

        f.write("Episode Length (steps):\n")
        episode_lengths = [m['episode_length'] for m in random_metrics]
        f.write(f"  Mean: {np.mean(episode_lengths):.1f}\n")
        f.write(f"  Std:  {np.std(episode_lengths):.1f}\n")
        f.write(f"  Min:  {np.min(episode_lengths):.1f}\n")
        f.write(f"  Max:  {np.max(episode_lengths):.1f}\n\n")

        f.write("Per-Episode Details:\n")
        f.write("-" * 80 + "\n")
        for i, metrics in enumerate(random_metrics):
            f.write(f"Episode {i+1} (seed={seed+i}):\n")
            f.write(f"  Healthy: {metrics['healthy_trees_ratio']*100:.2f}%\n")
            f.write(f"  Burnt:   {metrics['burnt_trees_ratio']*100:.2f}%\n")
            f.write(f"  Length:  {metrics['episode_length']} steps\n")

    print(f"✓ Report saved to: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate heuristic policies on wildfire suppression task"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes for each policy (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: train_marllib_self/evaluation/heuristic)"
    )

    args = parser.parse_args()

    main(args.episodes, args.seed, args.output_dir)
