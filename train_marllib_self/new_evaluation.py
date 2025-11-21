"""
Evaluation script for trained MARL models on wildfire suppression task

여러 알고리즘(MAPPO, PPO, MAA2C 등)으로 학습한 모델을 평가합니다.
각 에피소드에 대해 다음 지표들을 계산합니다:
  1. 평균 healthy (green) trees 비율
  2. 최종 피해 면적 (burnt trees 비율)
  3. 총 진화 시간 (에피소드 길이 in steps)

실행 예시:
ㅌ \\
    --experiment train_marllib_self/experiments/mappo/run13_fixed \\
    --checkpoint 100 \\
    --episodes 10 \\
    --seed 42

또는 checkpoint 번호 지정하지 않으면 최신 checkpoint 자동 선택:
python train_marllib_self/new_evaluation.py \\
    --experiment train_marllib_self/experiments/mappo/run13_fixed \\
    --episodes 10 \\
    --seed 42
"""

import sys
import os
from pathlib import Path

# Set up project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pickle
import torch
import re
import json
from pathlib import Path
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv


# ============================================================================
# Checkpoint Detection and Loading Utilities
# ============================================================================

def find_latest_checkpoint(directory_path):
    """
    Find the latest checkpoint in a Ray Tune directory structure

    Supports both RLlib (checkpoint-*) and RLac (best/final) formats.
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"  ❌ Directory not found: {directory_path}")
        return None

    if not directory.is_dir():
        # Already a checkpoint path
        if directory.name.startswith('checkpoint_') or directory.parent.name in ['best', 'final']:
            return str(directory_path)
        print(f"  ❌ Invalid path: {directory_path}")
        return None

    # Search for RLac format first (best/final directories)
    for checkpoint_dir in ['best', 'final']:
        checkpoint_path = directory / checkpoint_dir
        if checkpoint_path.exists():
            print(f"  ✓ Found RLac checkpoint: {checkpoint_dir}")
            return str(checkpoint_path)

    # Search for RLlib format (checkpoint_XXXXX)
    checkpoint_dirs = []
    for item in directory.rglob('checkpoint_*'):
        if item.is_dir():
            match = re.search(r'checkpoint_(\d+)', item.name)
            if match:
                checkpoint_number = int(match.group(1))
                checkpoint_dirs.append((checkpoint_number, item))

    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_num, latest_path = checkpoint_dirs[0]
        print(f"  ✓ Found RLlib checkpoint: checkpoint_{latest_num:06d}")
        return str(latest_path)

    print(f"  ❌ No checkpoint found in: {directory_path}")
    return None


def get_algorithm_from_path(checkpoint_path):
    """Extract algorithm name from checkpoint path"""
    path_parts = os.path.normpath(checkpoint_path).split(os.sep)

    # Look for algorithm name in path (mappo, ppo, maa2c, etc.)
    for part in path_parts:
        if part.lower() in ['mappo', 'ppo', 'maa2c', 'qmix', 'mappo_rl']:
            return part.lower()

    return 'unknown'


def load_heterogeneous_policies_rllib(checkpoint_path):
    """
    Load heterogeneous policies from RLlib checkpoint format

    RLlib stores policies under policy_<agent_type>_ keys in checkpoint-* files
    """
    checkpoint_dir = Path(checkpoint_path)

    # Find checkpoint-* files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
    checkpoint_files = [f for f in checkpoint_files if f.is_file() and not f.name.endswith('.tune_metadata')]

    if not checkpoint_files:
        return None

    # Sort and select the latest checkpoint
    checkpoint_files.sort(key=lambda x: int(x.name.split('-')[1]))
    checkpoint_file = checkpoint_files[-1]

    print(f"  Loading checkpoint file: {checkpoint_file.name}")

    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)

    policies_weights = {}

    if 'worker' in checkpoint_data:
        worker_bytes = checkpoint_data['worker']
        worker_data = pickle.loads(worker_bytes)

        if 'state' in worker_data:
            state = worker_data['state']

            # Map checkpoint keys to policy names
            policy_name_mappings = {
                'policy_helicopter_': 'helicopter_policy',
                'policy_truck_': 'truck_policy',
                'policy_crew_': 'crew_policy'
            }

            for checkpoint_key, canonical_name in policy_name_mappings.items():
                if checkpoint_key in state:
                    policy_state = state[checkpoint_key]
                    if 'weights' in policy_state:
                        policies_weights[canonical_name] = policy_state['weights']

            # Fallback: check for shared policy
            if 'shared_policy' in state and not policies_weights:
                policy_state = state['shared_policy']
                if 'weights' in policy_state:
                    return {'shared': policy_state['weights']}

    return policies_weights if policies_weights else None


def load_heterogeneous_policies_rlac(checkpoint_path):
    """
    Load heterogeneous policies from RLac checkpoint format (best/final)

    RLac stores policies under rl_module/{agent_type}/module_state.pkl
    """
    checkpoint_dir = Path(checkpoint_path)

    policies_weights = {}

    # Look for rl_module directory with agent policies
    rl_module_dir = checkpoint_dir / 'learner_group' / 'learner' / 'rl_module'

    if not rl_module_dir.exists():
        return None

    # Load each agent type's policy
    for agent_type in ['helicopter_policy', 'truck_policy', 'crew_policy']:
        agent_module_dir = rl_module_dir / agent_type

        if agent_module_dir.exists():
            module_state_file = agent_module_dir / 'module_state.pkl'

            if module_state_file.exists():
                try:
                    with open(module_state_file, 'rb') as f:
                        module_state = pickle.load(f)

                    if isinstance(module_state, dict):
                        policies_weights[agent_type] = module_state
                        print(f"  ✓ Loaded {agent_type} from RLac checkpoint")
                except Exception as e:
                    print(f"  ⚠ Failed to load {agent_type}: {e}")

    return policies_weights if policies_weights else None


def load_policies(checkpoint_path):
    """
    Load policies from checkpoint, supporting both RLlib and RLac formats
    """
    checkpoint_dir = Path(checkpoint_path)

    # Try RLac format first
    if checkpoint_dir.name in ['best', 'final']:
        print("  Attempting RLac format...")
        policies = load_heterogeneous_policies_rlac(checkpoint_path)
        if policies:
            return policies

    # Try RLlib format
    print("  Attempting RLlib format...")
    policies = load_heterogeneous_policies_rllib(checkpoint_path)
    if policies:
        return policies

    return None


# ============================================================================
# Policy Network Creation and Loading
# ============================================================================

def create_policy_network(obs_dim, action_dim, hidden_dim=256):
    """
    Create a policy network matching MARLlib's CentralizedCriticMLP architecture

    Supports weights from both MAPPO and PPO trained models.
    """

    class PolicyNetwork(torch.nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
            super().__init__()

            # Policy encoder
            self.p_encoder_layer0 = torch.nn.Linear(obs_dim, hidden_dim)
            self.p_encoder_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.p_branch = torch.nn.Linear(hidden_dim, action_dim)

            # Value function encoder
            self.vf_encoder_layer0 = torch.nn.Linear(obs_dim, hidden_dim)
            self.vf_encoder_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.vf_branch = torch.nn.Linear(hidden_dim, 1)

            # Centralized critic encoder (for MAPPO)
            self.cc_vf_encoder_layer0 = torch.nn.Linear(obs_dim, hidden_dim)
            self.cc_vf_encoder_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.cc_vf_branch = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x):
            """Forward pass for policy"""
            p_features = torch.relu(self.p_encoder_layer0(x))
            p_features = torch.relu(self.p_encoder_layer1(p_features))
            logits = self.p_branch(p_features)
            return logits

        def get_action(self, obs):
            """Select action greedily from observation"""
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = self.forward(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            return action

    return PolicyNetwork(obs_dim, action_dim, hidden_dim)


def load_marllib_weights_to_network(network, marllib_weights):
    """
    Load MARLlib checkpoint weights to policy network

    Handles weight mapping from MARLlib format to our simplified network
    """
    weight_mapping = {
        'p_encoder.encoder.0._model.0.weight': ('p_encoder_layer0', 'weight'),
        'p_encoder.encoder.0._model.0.bias': ('p_encoder_layer0', 'bias'),
        'p_encoder.encoder.1._model.0.weight': ('p_encoder_layer1', 'weight'),
        'p_encoder.encoder.1._model.0.bias': ('p_encoder_layer1', 'bias'),
        'p_branch._model.0.weight': ('p_branch', 'weight'),
        'p_branch._model.0.bias': ('p_branch', 'bias'),
    }

    loaded_count = 0
    skipped_count = 0

    print(f"    MARLlib weights keys: {list(marllib_weights.keys())[:10]}...")
    print(f"    Total keys in checkpoint: {len(marllib_weights)}")

    for marllib_key, (layer_name, param_type) in weight_mapping.items():
        if marllib_key not in marllib_weights:
            print(f"    ⚠ {marllib_key} 없음 (스킵)")
            skipped_count += 1
            continue

        marllib_weight = marllib_weights[marllib_key]
        layer = getattr(network, layer_name)

        target_param = getattr(layer, param_type)

        if target_param.shape != marllib_weight.shape:
            print(f"    ⚠ {marllib_key}: shape 불일치 {target_param.shape} vs {marllib_weight.shape} (스킵)")
            skipped_count += 1
            continue

        target_param.data = torch.FloatTensor(marllib_weight)
        print(f"    ✓ {marllib_key}: {target_param.shape} 로드 성공")
        loaded_count += 1

    print(f"    총 {loaded_count}개 가중치 로드 성공 ({skipped_count}개 스킵)")
    return loaded_count > 0


def get_agent_policy_name(agent, num_helicopters, num_trucks, num_crews):
    """Get policy name for an agent based on its type"""
    if hasattr(agent, 'type'):
        if agent.type == 'helicopter':
            return 'helicopter_policy'
        elif agent.type == 'truck':
            return 'truck_policy'
        elif agent.type == 'crew':
            return 'crew_policy'

    # Fallback based on index
    if agent.index < num_helicopters:
        return 'helicopter_policy'
    elif agent.index < num_helicopters + num_trucks:
        return 'truck_policy'
    else:
        return 'crew_policy'


# ============================================================================
# Episode Evaluation
# ============================================================================

def run_evaluation_episode(env, policy_networks, policy_mapping_fn, seed, max_steps=300):
    """
    Run a single evaluation episode and collect metrics

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

        # Select actions
        actions = {}
        for agent_id in obs_dict.keys():
            obs = obs_dict[agent_id]

            if policy_networks is not None and policy_mapping_fn is not None:
                policy_name = policy_mapping_fn(agent_id)

                if policy_name in policy_networks:
                    network = policy_networks[policy_name]
                    action = network.get_action(obs)
                else:
                    action = env.action_space[agent_id].sample()
            else:
                # Random policy if no trained networks
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
# Main Evaluation Function
# ============================================================================

def main(experiment_path, checkpoint_num=None, num_episodes=10, seed=42):
    """
    Evaluate trained model on wildfire suppression task

    Parameters:
        experiment_path: Path to experiment directory (containing checkpoints)
        checkpoint_num: Specific checkpoint number (if None, use latest)
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
    """

    # ============================================================================
    # Initialize all random seeds for reproducibility
    # ============================================================================
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set torch to deterministic mode for complete reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 80)
    print("Wildfire Suppression Task - Model Evaluation")
    print("=" * 80)

    # Validate and find checkpoint
    experiment_dir = Path(experiment_path)
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_path}")
        return

    # Find checkpoint
    checkpoint_path = None
    if checkpoint_num is not None:
        # Look for specific checkpoint
        for item in experiment_dir.rglob(f'checkpoint_{checkpoint_num:06d}'):
            if item.is_dir():
                checkpoint_path = str(item)
                break

        if checkpoint_path is None:
            print(f"❌ Checkpoint {checkpoint_num:06d} not found")
            return
    else:
        # Use latest checkpoint
        checkpoint_path = find_latest_checkpoint(experiment_path)
        if checkpoint_path is None:
            return

    print(f"\nCheckpoint: {checkpoint_path}")

    # Get algorithm name
    algorithm = get_algorithm_from_path(checkpoint_path)
    run_name = experiment_dir.name

    print(f"Algorithm: {algorithm}")
    print(f"Run: {run_name}")

    # Create output directory
    output_dir = Path(__file__).parent / "evaluation" / algorithm / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Create environment
    print("\nCreating environment...")
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    print(f"  Grid size: {env_config['size']}x{env_config['size']}")
    print(f"  Agents: {env.num_agents}")
    print(f"  Max steps: {env_config['max_steps']}")

    # Get agent configuration
    num_helicopters = sum(1 for agent in env.agents if agent.type == "helicopter")
    num_trucks = sum(1 for agent in env.agents if agent.type == "truck")
    num_crews = sum(1 for agent in env.agents if agent.type == "crew")

    print(f"Agent types:")
    print(f"  Helicopters: {num_helicopters}")
    print(f"  Trucks: {num_trucks}")
    print(f"  Crews: {num_crews}")

    # Create policy mapping function
    def create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews):
        def mapping_fn(agent_id):
            try:
                agent_idx = int(agent_id)
                if 0 <= agent_idx < len(env.agents):
                    agent_obj = env.agents[agent_idx]
                    return get_agent_policy_name(agent_obj, num_helicopters, num_trucks, num_crews)
            except (ValueError, TypeError):
                pass
            return 'helicopter_policy'
        return mapping_fn

    policy_mapping_fn = create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews)

    # Load policies
    print("\nLoading policies from checkpoint...")
    policy_networks = None

    try:
        policies_weights = load_policies(checkpoint_path)

        if policies_weights is not None:
            first_agent_id = list(env.observation_space.keys())[0]
            obs_dim = env.observation_space[first_agent_id].shape[0]
            action_dim = env.action_space[first_agent_id].n

            print(f"  Observation dim: {obs_dim}")
            print(f"  Action dim: {action_dim}")

            policy_networks = {}

            if 'shared' in policies_weights:
                # Shared policy
                policy_networks['shared'] = create_policy_network(obs_dim, action_dim)
                success = load_marllib_weights_to_network(
                    policy_networks['shared'],
                    policies_weights['shared']
                )
                if success:
                    print("  ✓ Loaded shared policy weights")
            else:
                # Heterogeneous policies
                for policy_name in ['helicopter_policy', 'truck_policy', 'crew_policy']:
                    if policy_name in policies_weights:
                        policy_networks[policy_name] = create_policy_network(obs_dim, action_dim)
                        success = load_marllib_weights_to_network(
                            policy_networks[policy_name],
                            policies_weights[policy_name]
                        )
                        if success:
                            print(f"  ✓ Loaded {policy_name} weights")
        else:
            print("  ⚠ Could not load policies - will use random policy")

    except Exception as e:
        print(f"  ❌ Error loading policies: {e}")
        import traceback
        traceback.print_exc()

    # Run evaluation episodes
    print(f"\nEvaluating on {num_episodes} episodes...")
    print("-" * 80)

    metrics_list = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"Episode {ep+1}/{num_episodes} (seed={episode_seed})...", end="", flush=True)

        metrics = run_evaluation_episode(
            env, policy_networks, policy_mapping_fn,
            episode_seed, max_steps=env_config['max_steps']
        )

        metrics_list.append(metrics)

        print(f" ✓")
        print(f"  - Healthy trees: {metrics['healthy_trees_ratio']*100:.1f}%")
        print(f"  - Burnt trees: {metrics['burnt_trees_ratio']*100:.1f}%")
        print(f"  - Episode length: {metrics['episode_length']} steps")

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    healthy_ratios = [m['healthy_trees_ratio'] for m in metrics_list]
    burnt_ratios = [m['burnt_trees_ratio'] for m in metrics_list]
    episode_lengths = [m['episode_length'] for m in metrics_list]

    print(f"\nHealthy Trees (Avg ± Std):")
    print(f"  Mean: {np.mean(healthy_ratios)*100:.2f}% ± {np.std(healthy_ratios)*100:.2f}%")

    print(f"\nBurnt Trees / Damage Area (Avg ± Std):")
    print(f"  Mean: {np.mean(burnt_ratios)*100:.2f}% ± {np.std(burnt_ratios)*100:.2f}%")

    print(f"\nExtinguishing Time (Avg ± Std):")
    print(f"  Mean: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")

    # Save results
    results = {
        'algorithm': algorithm,
        'run_name': run_name,
        'checkpoint_path': checkpoint_path,
        'num_episodes': num_episodes,
        'seed': seed,
        'environment_config': env_config,
        'metrics': metrics_list,
        'summary': {
            'healthy_trees_mean': float(np.mean(healthy_ratios)),
            'healthy_trees_std': float(np.std(healthy_ratios)),
            'burnt_trees_mean': float(np.mean(burnt_ratios)),
            'burnt_trees_std': float(np.std(burnt_ratios)),
            'episode_length_mean': float(np.mean(episode_lengths)),
            'episode_length_std': float(np.std(episode_lengths)),
        }
    }

    # Save to JSON
    results_file = output_dir / f"evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Save detailed report
    report_file = output_dir / f"evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Wildfire Suppression Task - Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Evaluation Episodes: {num_episodes}\n")
        f.write(f"Seed: {seed}\n\n")

        f.write("Environment Configuration:\n")
        for key, value in env_config.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Healthy Trees Ratio (%):\n")
        f.write(f"  Mean: {np.mean(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Std:  {np.std(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Min:  {np.min(healthy_ratios)*100:.2f}%\n")
        f.write(f"  Max:  {np.max(healthy_ratios)*100:.2f}%\n\n")

        f.write("Burnt Trees Ratio (%) - Damage Area:\n")
        f.write(f"  Mean: {np.mean(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Std:  {np.std(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Min:  {np.min(burnt_ratios)*100:.2f}%\n")
        f.write(f"  Max:  {np.max(burnt_ratios)*100:.2f}%\n\n")

        f.write("Episode Length (steps):\n")
        f.write(f"  Mean: {np.mean(episode_lengths):.1f}\n")
        f.write(f"  Std:  {np.std(episode_lengths):.1f}\n")
        f.write(f"  Min:  {np.min(episode_lengths):.1f}\n")
        f.write(f"  Max:  {np.max(episode_lengths):.1f}\n\n")

        f.write("Per-Episode Details:\n")
        f.write("-" * 80 + "\n")
        for i, metrics in enumerate(metrics_list):
            f.write(f"Episode {i+1} (seed={seed+i}):\n")
            f.write(f"  Healthy: {metrics['healthy_trees_ratio']*100:.2f}%\n")
            f.write(f"  Burnt:   {metrics['burnt_trees_ratio']*100:.2f}%\n")
            f.write(f"  Length:  {metrics['episode_length']} steps\n")

    print(f"✓ Report saved to: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trained MARL models on wildfire suppression task"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment directory (containing checkpoints)"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Specific checkpoint number (e.g., 100). If not specified, uses latest."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    main(args.experiment, args.checkpoint, args.episodes, args.seed)
