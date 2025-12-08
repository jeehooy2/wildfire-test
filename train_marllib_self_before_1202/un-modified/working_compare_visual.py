"""
MARLlib 학습된 정책과 랜덤 정책 시각적 비교

MARLlib MAPPO로 학습된 에이전트와 랜덤 에이전트의 행동을
side-by-side로 비교하는 GIF를 생성합니다.

실행 예시:
# run1 실험 결과 비교
python train_marllib_self/new_compare_visual4.py \
    --checkpoint train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_16fa4_00000_0_2025-11-16_05-07-03/checkpoint_000002 \
    --episodes 3 \
    --seed 42

# 출력 디렉토리 직접 지정
python train_marllib_self/new_compare_visual4.py --checkpoint <path> --output-dir ./my_gifs/
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
from PIL import Image, ImageDraw, ImageFont


def load_policy_weights(checkpoint_path):
    """체크포인트에서 policy weights 직접 로드"""

    checkpoint_dir = Path(checkpoint_path)

    # checkpoint-* 파일 찾기
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
    checkpoint_files = [f for f in checkpoint_files if f.is_file() and not f.name.endswith('.tune_metadata')]

    if not checkpoint_files:
        print(f"  ⚠ 경고: checkpoint-* 파일을 찾을 수 없습니다: {checkpoint_dir}")
        return None

    # 가장 최신 checkpoint 파일 선택 (번호가 높은 것)
    checkpoint_files.sort(key=lambda x: int(x.name.split('-')[1]))
    checkpoint_file = checkpoint_files[-1]

    print(f"  체크포인트 파일: {checkpoint_file.name}")

    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)

    # worker는 bytes로 저장되어 있음 (Ray 직렬화)
    if 'worker' in checkpoint_data:
        worker_bytes = checkpoint_data['worker']
        worker_data = pickle.loads(worker_bytes)

        if 'state' in worker_data and 'shared_policy' in worker_data['state']:
            policy_state = worker_data['state']['shared_policy']

            if 'weights' in policy_state:
                print(f"  ✓ Policy weights 발견")
                return policy_state['weights']

    print("  ⚠ 경고: 'worker' 또는 'weights'를 체크포인트에서 찾을 수 없습니다.")
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


def run_episode_and_render(env, policy_network, seed, max_steps=300, policy_type="trained"):
    """
    에피소드를 실행하고 프레임을 수집

    Parameters
    ----------
    env : WildfireEnv
        환경
    policy_network : PolicyNetwork or None
        학습된 정책 네트워크 (None이면 랜덤 정책)
    seed : int
        랜덤 시드
    max_steps : int
        최대 스텝 수
    policy_type : str
        정책 타입 (trained or random)

    Returns
    -------
    tuple
        (frames, episode_reward, episode_length)
    """
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
            if policy_type == "trained" and policy_network is not None:
                action = policy_network.get_action(obs)
            else:
                # 랜덤 액션
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

    return frames, episode_reward, step


def combine_frames_side_by_side(frames1, frames2, label1="Trained", label2="Random"):
    """
    두 프레임 시퀀스를 side-by-side로 결합

    Parameters
    ----------
    frames1 : list
        첫 번째 프레임 시퀀스
    frames2 : list
        두 번째 프레임 시퀀스
    label1 : str
        첫 번째 프레임 레이블
    label2 : str
        두 번째 프레임 레이블

    Returns
    -------
    list
        결합된 프레임 시퀀스
    """
    # 더 짧은 시퀀스에 맞춤
    min_len = min(len(frames1), len(frames2))
    frames1 = frames1[:min_len]
    frames2 = frames2[:min_len]

    combined_frames = []

    for f1, f2 in zip(frames1, frames2):
        # PIL Image로 변환
        img1 = Image.fromarray(f1)
        img2 = Image.fromarray(f2)

        # 크기 확인
        w1, h1 = img1.size
        w2, h2 = img2.size

        # 텍스트 레이블을 위한 여유 공간
        label_height = 30

        # 새로운 이미지 생성 (side-by-side)
        combined_width = w1 + w2 + 10  # 10px 여백
        combined_height = max(h1, h2) + label_height
        combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

        # 이미지 붙이기
        combined.paste(img1, (0, label_height))
        combined.paste(img2, (w1 + 10, label_height))

        # 레이블 추가
        draw = ImageDraw.Draw(combined)
        try:
            # 기본 폰트 사용
            font = ImageFont.load_default()
        except:
            font = None

        # 텍스트 추가
        draw.text((w1//2 - 30, 5), label1, fill=(0, 0, 0), font=font)
        draw.text((w1 + 10 + w2//2 - 30, 5), label2, fill=(0, 0, 0), font=font)

        # 구분선 추가
        draw.line([(w1 + 5, label_height), (w1 + 5, combined_height)], fill=(200, 200, 200), width=2)

        combined_frames.append(np.array(combined))

    return combined_frames


def save_as_gif(frames, filename, fps=10):
    """
    프레임을 GIF로 저장

    Parameters
    ----------
    frames : list
        프레임 리스트
    filename : str
        저장할 파일명
    fps : int
        초당 프레임 수
    """
    if not frames:
        print("저장할 프레임이 없습니다!")
        return

    images = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)

    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    file_size = os.path.getsize(filename) / 1024
    print(f"  ✓ 저장 완료: {filename} ({len(frames)} frames, {file_size:.1f} KB)")


def main(checkpoint_path, num_episodes=3, seed=42, output_dir=None):
    """
    메인 함수: 학습된 모델과 랜덤 정책을 시각적으로 비교

    Parameters
    ----------
    checkpoint_path : str
        MARLlib 학습된 모델의 체크포인트 경로
    num_episodes : int
        생성할 비교 GIF 수
    seed : int
        시작 랜덤 시드
    output_dir : str, optional
        GIF를 저장할 디렉토리. None이면 checkpoint 경로에서 run name을 추출하여
        train_marllib_self/results/mappo/{run_name}/ 에 저장
    """
    # output_dir이 지정되지 않은 경우, checkpoint 경로에서 run name 추출
    if output_dir is None:
        # 경로 예: train_marllib_self/experiments/mappo/run1/... -> run1
        path_parts = os.path.normpath(checkpoint_path).split(os.sep)
        if "mappo" in path_parts:
            mappo_idx = path_parts.index("mappo")
            if mappo_idx + 1 < len(path_parts):
                run_name = path_parts[mappo_idx + 1]
                output_dir = os.path.join("train_marllib_self", "results", "mappo", run_name)
            else:
                output_dir = "train_marllib_self/results/mappo/default"
        else:
            output_dir = "train_marllib_self/results/mappo/default"

    print("=" * 80)
    print("MARLlib 학습된 정책 vs 랜덤 정책 시각적 비교")
    print("=" * 80)
    print(f"체크포인트: {checkpoint_path}")
    print(f"에피소드 수: {num_episodes}")
    print(f"시작 시드: {seed}")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 80)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 환경 설정
    print("\n환경 생성 중...")
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_trained = WildfireEnv(**env_config)
    env_random = WildfireEnv(**env_config)

    print(f"✓ 환경 생성 완료")
    print(f"  Grid size: {env_config['size']}x{env_config['size']}")
    print(f"  Agents: {env_trained.num_agents}")
    print(f"  Max steps: {env_config['max_steps']}")

    # 정책 네트워크 생성 및 가중치 로드
    policy_network = None
    print(f"\n체크포인트 로드 시도...")
    try:
        policy_weights = load_policy_weights(checkpoint_path)

        if policy_weights is not None:
            obs_dim = env_trained.observation_space["0"].shape[0]
            action_dim = env_trained.action_space["0"].n

            print(f"  Observation dim: {obs_dim}")
            print(f"  Action dim: {action_dim}")

            # Policy network 생성
            policy_network = create_policy_network(obs_dim, action_dim)

            # Weights 구조 확인
            print(f"  Checkpoint weights keys: {list(policy_weights.keys())[:5]}...")

            # Weights 로드 시도
            try:
                policy_network.load_state_dict(policy_weights, strict=False)
                print("  ✓ Policy weights 로드 완료!")
            except Exception as load_error:
                print(f"  ⚠ 직접 로드 실패: {load_error}")
                print("  → 랜덤 초기화된 네트워크 사용")
            # if '_model' in policy_weights:
            #     model_weights = policy_weights['_model']
            #     policy_network.load_state_dict(model_weights, strict=False)
            #     print("  ✓ Policy weights 로드 완료!")
            # else:
            #     print("  ⚠ 예상 형식과 다름 - 랜덤 초기화된 네트워크 사용")

        else:
            print("  ⚠ Policy weights를 찾을 수 없습니다.")
            print("  → 랜덤 정책과의 비교를 진행하지 않습니다.")
            return

    except Exception as e:
        print(f"  ❌ 체크포인트 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        print("  → 스크립트를 중단합니다.")
        return

    # 시드별로 비교 에피소드 실행 및 GIF 생성
    print(f"\n학습된 정책 vs 랜덤 정책 비교 GIF 생성")
    print("-" * 80)

    all_rewards_trained = []
    all_rewards_random = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"\n[{ep+1}/{num_episodes}] 에피소드 {ep+1} 생성 중 (seed={episode_seed})...")

        # 학습된 정책 실행
        print("  - 학습된 정책 실행 중...")
        trained_frames, trained_reward, trained_length = run_episode_and_render(
            env_trained, policy_network, episode_seed,
            max_steps=env_config['max_steps'], policy_type="trained"
        )
        print(f"    보상: {trained_reward:.2f}, 길이: {trained_length}")

        # 랜덤 정책 실행 (같은 시드로 동일한 초기 상태)
        print("  - 랜덤 정책 실행 중...")
        random_frames, random_reward, random_length = run_episode_and_render(
            env_random, None, episode_seed,
            max_steps=env_config['max_steps'], policy_type="random"
        )
        print(f"    보상: {random_reward:.2f}, 길이: {random_length}")

        all_rewards_trained.append(trained_reward)
        all_rewards_random.append(random_reward)

        # side-by-side 결합
        print("  - 프레임 결합 중...")
        combined_frames = combine_frames_side_by_side(
            trained_frames, random_frames,
            label1=f"Trained (R={trained_reward:.1f})",
            label2=f"Random (R={random_reward:.1f})"
        )

        # GIF로 저장
        output_path = os.path.join(output_dir, f"comparison_ep{ep+1:02d}_seed{episode_seed}.gif")
        save_as_gif(combined_frames, output_path, fps=10)

    # 최종 통계 출력
    print("\n" + "=" * 80)
    print("최종 통계 (Trained vs Random)")
    print("=" * 80)
    print(f"\n평균 리워드 (Trained): {np.mean(all_rewards_trained):.2f} ± {np.std(all_rewards_trained):.2f}")
    print(f"평균 리워드 (Random): {np.mean(all_rewards_random):.2f} ± {np.std(all_rewards_random):.2f}")

    print(f"\n모든 GIF가 저장되었습니다: {output_dir}")

    # 통계 파일 저장
    stats_path = os.path.join(output_dir, f"comparison_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Trained vs Random Agent Visualization Stats\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Seeds: {[seed + i for i in range(num_episodes)]}\n\n")
        for i in range(num_episodes):
            episode_seed = seed + i
            f.write(f"Episode {i+1} (Seed {episode_seed}):\n")
            f.write(f"  Trained Reward: {all_rewards_trained[i]:.2f}\n")
            f.write(f"  Random Reward: {all_rewards_random[i]:.2f}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"Avg Trained Reward: {np.mean(all_rewards_trained):.2f} ± {np.std(all_rewards_trained):.2f}\n")
        f.write(f"Avg Random Reward: {np.mean(all_rewards_random):.2f} ± {np.std(all_rewards_random):.2f}\n")

    print(f"✓ 통계 저장: {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARLlib 학습된 정책과 랜덤 정책 시각적 비교")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="MARLlib 체크포인트 경로 (예: train_marllib_self/experiments/mappo/run1/.../checkpoint_000050)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="생성할 비교 GIF 수 (기본값: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="시작 랜덤 시드 (기본값: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="GIF를 저장할 디렉토리 (기본값: train_marllib_self/results/mappo/{run_name}/)"
    )

    args = parser.parse_args()

    main(args.checkpoint, args.episodes, args.seed, args.output_dir)
