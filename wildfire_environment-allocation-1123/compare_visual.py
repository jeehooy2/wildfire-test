"""
학습된 정책과 랜덤 정책 시각적 비교

같은 시드를 사용하여 학습된 에이전트와 랜덤 에이전트의 행동을
side-by-side로 비교하는 GIF를 생성합니다.

$ uv run python compare_visual.py --checkpoint rllib_checkpoints/best --episodes 10 --seed 42
"""

import os
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from wildfire_rllib_wrapper import WildfireRLlibEnv
from masked_model import MaskedCategoricalTorchModel
from PIL import Image, ImageDraw, ImageFont
import gym
import wildfire_environment


def run_episode_and_render(algo, env_config, seed, policy_type="trained"):
    """
    에피소드를 실행하고 프레임을 수집

    Parameters
    ----------
    algo : Algorithm or None
        RLlib 알고리즘 (None이면 랜덤 정책)
    env_config : dict
        환경 설정
    seed : int
        랜덤 시드
    policy_type : str
        정책 타입 (trained or random)

    Returns
    -------
    tuple
        (frames, episode_reward, episode_length)
    """
    # render_mode를 추가한 환경 설정
    render_env_config = env_config.copy()
    render_env_config["render_mode"] = "rgb_array"
    render_env_config["render_selfish_region_boundaries"] = True

    # WildfireRLlibEnv 래퍼 사용
    env = WildfireRLlibEnv(render_env_config)

    obs, info = env.reset(seed=seed)

    frames = []
    episode_reward = 0
    episode_length = 0

    # 첫 프레임 렌더링 (내부 환경의 render 호출)
    frame = env.env.render()
    frames.append(frame)

    done = False
    while not done:
        if policy_type == "trained" and algo is not None:
            # 학습된 정책 사용 (compute_single_action API)
            actions = {}
            for agent_id in range(env.num_agents):
                action = algo.compute_single_action(
                    obs[agent_id],
                    policy_id="shared_policy",
                    explore=False  # Deterministic policy for evaluation
                )
                actions[agent_id] = action
        else:
            # 랜덤 정책 사용
            actions = {i: env._action_space[i].sample() for i in range(env.num_agents)}

        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        episode_reward += sum(rewards.values())
        episode_length += 1

        # 프레임 렌더링
        frame = env.env.render()
        frames.append(frame)

        # 종료 확인
        done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

    env.close()

    return frames, episode_reward, episode_length


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


def main(checkpoint_path, num_episodes=10, seed=42, output_dir="comparison_gifs"):
    """
    메인 함수: 학습된 모델과 랜덤 정책을 시각적으로 비교

    Parameters
    ----------
    checkpoint_path : str
        학습된 모델의 체크포인트 경로
    num_episodes : int
        생성할 비교 GIF 수
    seed : int
        시작 랜덤 시드
    output_dir : str
        GIF를 저장할 디렉토리
    """
    print("=" * 70)
    print("학습된 정책 vs 랜덤 정책 시각적 비교")
    print("=" * 70)
    print(f"체크포인트: {checkpoint_path}")
    print(f"에피소드 수: {num_episodes}")
    print(f"시작 시드: {seed}")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 70)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # Ray 초기화
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env={"excludes": ['.git/']})

    # Register custom masked model (필수: 체크포인트 로드 시 필요)
    ModelCatalog.register_custom_model("masked_categorical", MaskedCategoricalTorchModel)

    # 환경 등록
    def env_creator(env_config):
        return WildfireRLlibEnv(env_config)

    register_env("wildfire-ma", env_creator)

    # 환경 설정 (학습 시와 동일하게 - train_multiagent_rllib.py 참조)
    env_config = {
        "num_agents": 2,
        "size": 17,
        "initial_fire_size": 4,  # 학습 시 설정과 동일
        "cooperative_reward": False,
        "max_steps": 200,  # 학습 시 설정과 동일
        "agent_start_positions": ((1, 1), (15, 15)),
        "log_selfish_region_metrics": True,
        "selfish_region_xmin": [7, 13],
        "selfish_region_xmax": [9, 15],
        "selfish_region_ymin": [7, 1],
        "selfish_region_ymax": [9, 3],
        "delta_beta": 0.7,
        "beta": 0.99,
        "alpha": 0.05,
        # Action masking configuration
        "use_action_mask": True,  # 체크포인트와 동일하게 설정
        "max_masked_actions": 64,
        "target_switch_penalty": 0.01,
    }

    # 학습된 모델 로드
    print("\n학습된 모델 로딩 중...")
    checkpoint_path = os.path.abspath(checkpoint_path)
    algo = PPO.from_checkpoint(checkpoint_path)
    print("✓ 모델 로드 완료\n")

    # 각 에피소드에 대해 비교 GIF 생성
    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"[{ep+1}/{num_episodes}] 에피소드 {ep+1} 생성 중 (seed={episode_seed})...")

        # 학습된 정책 실행
        print("  - 학습된 정책 실행 중...")
        trained_frames, trained_reward, trained_length = run_episode_and_render(
            algo, env_config, episode_seed, policy_type="trained"
        )
        print(f"    보상: {trained_reward:.2f}, 길이: {trained_length}")

        # 랜덤 정책 실행
        print("  - 랜덤 정책 실행 중...")
        random_frames, random_reward, random_length = run_episode_and_render(
            None, env_config, episode_seed, policy_type="random"
        )
        print(f"    보상: {random_reward:.2f}, 길이: {random_length}")

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
        print()

    print("=" * 70)
    print(f"✓ 완료! {num_episodes}개의 비교 GIF가 '{output_dir}' 디렉토리에 저장되었습니다.")
    print("=" * 70)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="학습된 정책과 랜덤 정책 시각적 비교")
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
        help="생성할 비교 GIF 수"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="시작 랜덤 시드"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_gifs",
        help="GIF를 저장할 디렉토리"
    )

    args = parser.parse_args()

    main(args.checkpoint, args.episodes, args.seed, args.output_dir)
