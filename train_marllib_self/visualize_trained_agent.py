"""
학습된 MAPPO 에이전트의 실제 동작을 GIF로 시각화

checkpoint_000200에서 모델을 로드하여 여러 시드로 에피소드 실행 및 GIF 생성
"""

import sys
import os
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pickle
import torch
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io


def load_policy_weights(checkpoint_path):
    """체크포인트에서 policy weights 직접 로드"""

    checkpoint_file = Path(checkpoint_path) / "checkpoint-50"

    print(f"  체크포인트 파일: {checkpoint_file.name}")

    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)

    # worker는 bytes로 저장되어 있음 (Ray 직렬화)
    if 'worker' in checkpoint_data:
        worker_bytes = checkpoint_data['worker']

        # Worker bytes를 언패킹
        worker_data = pickle.loads(worker_bytes)

        if 'state' in worker_data and 'shared_policy' in worker_data['state']:
            policy_state = worker_data['state']['shared_policy']

            if 'weights' in policy_state:
                print(f"  ✓ Policy weights 발견")
                return policy_state['weights']

    return None


def create_policy_network(obs_dim, action_dim):
    """간단한 policy network 생성 (MLP)"""

    class PolicyNetwork(torch.nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc_policy = torch.nn.Linear(hidden_dim, action_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            logits = self.fc_policy(x)
            return logits

        def get_action(self, obs):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = self.forward(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            return action

    return PolicyNetwork(obs_dim, action_dim)


def run_episode_and_render(env, policy_network, seed, max_steps=300):
    """에피소드 실행하고 프레임 수집"""

    # 환경 리셋
    obs_dict, _ = env.reset(seed=seed)

    frames = []
    done = False
    episode_reward = 0
    step = 0

    # 초기 프레임
    frame = env.render(mode='rgb_array')
    frames.append(frame)

    while not done and step < max_steps:
        # 각 에이전트의 액션 선택
        actions = {}
        for agent_id in range(env.num_agents):
            obs = obs_dict[str(agent_id)]
            if policy_network is not None:
                action = policy_network.get_action(obs)
            else:
                # 랜덤 액션 (비교용)
                action = env.action_space[str(agent_id)].sample()
            actions[str(agent_id)] = action

        # 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        # 리워드 합산
        episode_reward += np.mean(list(reward_dict.values()))

        # 프레임 저장
        frame = env.render(mode='rgb_array')
        frames.append(frame)

        step += 1

    print(f"  Seed {seed}: {step} steps, Total Reward: {episode_reward:.2f}")

    return frames, episode_reward, step


def create_gif(frames, output_path, fps=10):
    """프레임들을 GIF로 저장"""

    # PIL Image로 변환
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # GIF 저장
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000//fps,  # milliseconds per frame
        loop=0
    )

    print(f"  ✓ GIF 저장: {output_path}")


def visualize_trained_agent(checkpoint_path, seeds=[0, 42, 123], use_trained=True):
    """학습된 에이전트 시각화"""

    print("=" * 80)
    print(f"{'Trained' if use_trained else 'Random'} Agent Visualization")
    print("=" * 80)

    # 환경 생성
    print("\n환경 생성 중...")
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    print(f"✓ 환경 생성 완료")
    print(f"  Grid size: {env_config['size']}x{env_config['size']}")
    print(f"  Agents: {env.num_agents}")
    print(f"  Max steps: {env_config['max_steps']}")

    # Policy network 생성 및 weights 로드
    policy_network = None
    if use_trained:
        print(f"\n체크포인트 로드 시도...")

        try:
            # 체크포인트 데이터 로드
            policy_weights = load_policy_weights(checkpoint_path)

            if policy_weights is not None:
                # 관찰 공간 차원
                obs_dim = env.observation_space["0"].shape[0]
                action_dim = env.action_space["0"].n

                print(f"  Observation dim: {obs_dim}")
                print(f"  Action dim: {action_dim}")

                # Policy network 생성
                policy_network = create_policy_network(obs_dim, action_dim)

                # Weights 로드 시도
                try:
                    # MARLlib weights는 RLlib 형식 (nested dict)
                    # 필요한 레이어만 매칭
                    state_dict = policy_network.state_dict()

                    # Weights 구조 확인
                    print(f"  Checkpoint weights keys: {list(policy_weights.keys())[:5]}...")

                    # TorchPolicy는 '_model' 아래에 저장
                    if '_model' in policy_weights:
                        model_weights = policy_weights['_model']
                        policy_network.load_state_dict(model_weights, strict=False)
                        print("  ✓ Policy weights 로드 완료!")
                    else:
                        print("  ⚠ 예상 형식과 다름 - 학습된 weights 사용 불가")
                        print("  → 랜덤 초기화된 네트워크 사용")

                except Exception as e:
                    print(f"  ⚠ Weights 매칭 실패: {e}")
                    print("  → 랜덤 초기화된 네트워크 사용")
            else:
                print("  ⚠ Policy weights를 찾을 수 없습니다")
                print("  → 랜덤 액션 사용")

        except Exception as e:
            print(f"  ❌ 체크포인트 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            print("  → 랜덤 액션 사용")

    # 출력 디렉토리 생성
    output_dir = Path("/home/bmkim88/wildfire_environment/exp_results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 여러 시드로 실행 및 GIF 생성
    print(f"\n{'학습된' if use_trained else '랜덤'} 에이전트로 에피소드 실행 및 GIF 생성")
    print("-" * 80)

    all_rewards = []
    all_steps = []

    for seed in seeds:
        print(f"\nSeed {seed} 실행 중...")

        # 에피소드 실행
        frames, episode_reward, steps = run_episode_and_render(
            env, policy_network, seed
        )

        all_rewards.append(episode_reward)
        all_steps.append(steps)

        # GIF 생성
        gif_filename = f"{'trained' if use_trained else 'random'}_agent_seed_{seed}.gif"
        gif_path = output_dir / gif_filename
        create_gif(frames, gif_path, fps=10)

    # 통계 출력
    print("\n" + "=" * 80)
    print("통계")
    print("=" * 80)
    print(f"\n평균 리워드: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"평균 스텝: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")

    print(f"\n모든 GIF가 저장되었습니다: {output_dir}")

    # 통계 파일 저장
    stats_path = output_dir / f"{'trained' if use_trained else 'random'}_agent_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"{'Trained' if use_trained else 'Random'} Agent Visualization Stats\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Seeds: {seeds}\n\n")
        for i, seed in enumerate(seeds):
            f.write(f"Seed {seed}:\n")
            f.write(f"  Reward: {all_rewards[i]:.2f}\n")
            f.write(f"  Steps: {all_steps[i]}\n\n")
        f.write(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n")
        f.write(f"Average Steps: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}\n")

    print(f"✓ 통계 저장: {stats_path}")


if __name__ == '__main__':
    # checkpoint_path = "/home/bmkim88/wildfire_environment/exp_results/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_e09f3_00000_0_2025-11-15_15-58-07/checkpoint_000200"
    checkpoint_path = "/home/bmkim88/wildfire_environment/exp_results/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_bb4c2_00000_0_2025-11-16_01-22-35/checkpoint_000050"

    # 학습된 에이전트 시각화
    print("\n학습된 에이전트 시각화 중...")
    visualize_trained_agent(checkpoint_path, seeds=[0, 42, 123, 999], use_trained=True)

    # 비교를 위해 랜덤 에이전트도 시각화 (선택사항)
    # print("\n\n랜덤 에이전트 시각화 중...")
    # visualize_trained_agent(checkpoint_path, seeds=[0, 42, 123], use_trained=False)
