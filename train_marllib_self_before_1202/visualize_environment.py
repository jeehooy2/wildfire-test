"""
Wildfire Environment 랜덤 액션 시각화

환경이 랜덤 액션에 어떻게 반응하는지 보여주는 GIF를 생성합니다.
이 스크립트는 학습된 정책 없이 순수 랜덤 액션으로 환경을 구동합니다.

추가된 시각화 기능:
  1. 물/억제제 게이지 (시안색): 에이전트 아래쪽 - 남은 물의 양 비율
  2. 급수원 마커 (파란색 상자): 에이전트의 홈 위치(급수원) 표시
  3. 에이전트 상태 패널 (우측): 각 에이전트의 상태, 물 양, 재충전 시간

상태별 색상 코드:
  - ACTIVE (활동 중): 녹색
  - RETURNING (귀환 중): 황색
  - RECHARGING (충전 중): 빨간색

실행 예시:
    python train_marllib_self/visualize_environment.py --episodes 3 --seed 42

또는:
    python train_marllib_self/visualize_environment.py --episodes 5 --seed 100 --output-dir ./my_gifs/
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from PIL import Image, ImageDraw, ImageFont
from wildfire_environment.core.constants import TILE_PIXELS


def run_episode_with_random_actions(env, seed, max_steps=300):
    """
    환경에서 랜덤 액션으로 에피소드 실행 및 프레임 수집

    Parameters
    ----------
    env : WildfireEnv
        환경
    seed : int
        랜덤 시드
    max_steps : int
        최대 스텝 수

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

    # 프레임에 시각화 정보 추가
    # 1. 각 에이전트의 물 게이지 추가
    for agent in env.agents:
        frame = render_water_gauge(frame, agent)
        frame = render_supply_source_marker(frame, agent)

    # 2. 에이전트 상태 패널 추가
    frame = render_agent_status_panel(frame, env.agents, step)

    frames.append(frame)

    while not done and step < max_steps:
        # 각 에이전트의 액션 선택 (모두 랜덤)
        actions = {}
        for agent_id in obs_dict.keys():
            # 랜덤 액션
            action = env.action_space[agent_id].sample()
            actions[agent_id] = action

        # 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        # 리워드 합산
        episode_reward += np.mean(list(reward_dict.values()))

        # 프레임 저장
        frame = env.render(mode='rgb_array')

        # 프레임에 시각화 정보 추가
        # 1. 각 에이전트의 물 게이지 추가
        for agent in env.agents:
            frame = render_water_gauge(frame, agent)
            frame = render_supply_source_marker(frame, agent)

        # 2. 에이전트 상태 패널 추가
        frame = render_agent_status_panel(frame, env.agents, step)

        frames.append(frame)

        step += 1

    return frames, episode_reward, step


def render_activity_gauge(frame, agent, tile_size=TILE_PIXELS):
    """
    에이전트의 활동 시간 게이지를 에이전트 위쪽에 렌더링

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agent : Agent
        에이전트 객체
    tile_size : int
        타일 크기 (픽셀)

    Returns
    -------
    numpy array
        게이지가 추가된 이미지 배열
    """
    if not hasattr(agent, 'max_active_time') or agent.max_active_time == 0:
        return frame

    # 활동 시간 비율
    fill_ratio = agent.active_time_remaining / agent.max_active_time

    # 게이지 위치 (에이전트 위쪽 2픽셀)
    pos = agent.pos
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 2

    # 게이지 배경 (진회색)
    frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]

    # 게이지 채우기 (시안색)
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = [0, 255, 255]

    return frame


def render_water_gauge(frame, agent, tile_size=TILE_PIXELS):
    """
    에이전트의 물/억제제 게이지를 에이전트 아래쪽에 렌더링

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agent : Agent
        에이전트 객체
    tile_size : int
        타일 크기 (픽셀)

    Returns
    -------
    numpy array
        게이지가 추가된 이미지 배열
    """
    if not hasattr(agent, 'max_water') or agent.max_water == 0:
        return frame

    # 물 양 비율
    fill_ratio = agent.water_remaining / agent.max_water

    # 게이지 위치 (에이전트 아래쪽 6픽셀)
    pos = agent.pos
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 6

    # 게이지 배경 (진회색)
    frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]

    # 게이지 채우기 (시안색)
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = [0, 255, 255]

    return frame


def render_supply_source_marker(frame, agent, tile_size=TILE_PIXELS):
    """
    급수원(홈 위치)을 프레임에 표시

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agent : Agent
        에이전트 객체
    tile_size : int
        타일 크기 (픽셀)

    Returns
    -------
    numpy array
        급수원 마커가 추가된 이미지 배열
    """
    if not hasattr(agent, 'home_pos'):
        return frame

    home_pos = agent.home_pos
    # 급수원 타일의 시작점
    home_x = home_pos[0] * tile_size
    home_y = home_pos[1] * tile_size

    # 급수원을 파란색 상자로 표시
    box_thickness = 2

    # 상단 테두리
    frame[max(0, home_y):min(frame.shape[0], home_y+box_thickness),
          max(0, home_x):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]
    # 하단 테두리
    frame[max(0, home_y+tile_size-box_thickness):min(frame.shape[0], home_y+tile_size),
          max(0, home_x):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]
    # 좌측 테두리
    frame[max(0, home_y):min(frame.shape[0], home_y+tile_size),
          max(0, home_x):min(frame.shape[1], home_x+box_thickness)] = [0, 0, 255]
    # 우측 테두리
    frame[max(0, home_y):min(frame.shape[0], home_y+tile_size),
          max(0, home_x+tile_size-box_thickness):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]

    return frame


def render_agent_status_panel(frame, agents, step):
    """
    에이전트들의 상태를 화면 우측에 패널로 표시

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agents : list
        에이전트 리스트
    step : int
        현재 스텝

    Returns
    -------
    numpy array
        상태 패널이 추가된 이미지 배열
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    # 우측 패널 위치
    panel_width = 200
    panel_height = frame.shape[0]
    panel_x = frame.shape[1] - panel_width
    panel_y = 0

    # 패널 배경 (반투명 검은색 효과를 위해 직접 처리)
    # 상태 텍스트 추가
    y_offset = 10

    # 스텝 표시
    draw.text((panel_x + 5, y_offset), f"Step: {step}", fill=(255, 255, 255), font=font)
    y_offset += 15

    # 각 에이전트 상태
    state_names = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}

    for agent_idx, agent in enumerate(agents):
        # 에이전트 번호
        agent_text = f"Agent {agent_idx}:"
        draw.text((panel_x + 5, y_offset), agent_text, fill=(255, 255, 255), font=font)
        y_offset += 12

        # 상태
        state = state_names.get(agent.state, "UNKNOWN")
        state_color = (0, 255, 0) if agent.state == 0 else (255, 255, 0) if agent.state == 1 else (255, 0, 0)
        draw.text((panel_x + 10, y_offset), f"State: {state}", fill=state_color, font=font)
        y_offset += 12

        # 물 게이지 (있을 경우)
        if hasattr(agent, 'max_water'):
            draw.text((panel_x + 10, y_offset), f"Water: {agent.water_remaining:.1f}/{agent.max_water}",
                     fill=(100, 200, 255), font=font)
            y_offset += 12

        # 재충전 시간 (RECHARGING 상태일 때만)
        if agent.state == 2 and hasattr(agent, 'recharge_time'):
            recharge_percent = int(100 * (agent.recharge_time - agent.recharge_time_remaining) / agent.recharge_time)
            draw.text((panel_x + 10, y_offset), f"Recharge: {agent.recharge_time_remaining}/{agent.recharge_time}",
                     fill=(255, 165, 0), font=font)
            y_offset += 12

        y_offset += 5  # 에이전트 사이의 간격

    return np.array(img)


def add_text_to_frame(frame, text, position=(10, 10)):
    """
    프레임에 텍스트 추가

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    text : str
        추가할 텍스트
    position : tuple
        텍스트 위치 (x, y)

    Returns
    -------
    numpy array
        텍스트가 추가된 이미지 배열
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    # 텍스트 추가 (검은색)
    draw.text(position, text, fill=(0, 0, 0), font=font)

    return np.array(img)


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


def main(num_episodes=3, seed=42, output_dir=None, fps=10):
    """
    메인 함수: 랜덤 액션으로 환경 시각화

    Parameters
    ----------
    num_episodes : int
        생성할 GIF 수 (기본값: 3)
    seed : int
        시작 랜덤 시드 (기본값: 42)
    output_dir : str, optional
        GIF를 저장할 디렉토리. None이면 train_marllib_self/ 에 저장
    fps : int
        GIF의 프레임 속도 (기본값: 10)
    """
    # output_dir이 지정되지 않은 경우
    if output_dir is None:
        output_dir = os.path.join("train_marllib_self", "results", "visualization")

    print("=" * 80)
    print("Wildfire Environment 랜덤 액션 시각화")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 에피소드 수: {num_episodes}")
    print(f"  - 시작 시드: {seed}")
    print(f"  - 출력 디렉토리: {output_dir}")
    print(f"  - FPS: {fps}")
    print("=" * 80)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 환경 설정
    print("\n환경 생성 중...")
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    print(f"✓ 환경 생성 완료")
    print(f"  Grid size: {env_config['size']}x{env_config['size']}")
    print(f"  Agents: {env.num_agents}")
    print(f"  Max steps: {env_config['max_steps']}")

    # 에이전트 구성 출력
    num_helicopters = sum(1 for agent in env.agents if agent.type == "helicopter")
    num_trucks = sum(1 for agent in env.agents if agent.type == "truck")
    num_crews = sum(1 for agent in env.agents if agent.type == "crew")

    print(f"\n에이전트 구성:")
    print(f"  - Helicopters: {num_helicopters}")
    print(f"  - Trucks: {num_trucks}")
    print(f"  - Crews: {num_crews}")

    # 각 에피소드별 GIF 생성
    print(f"\n랜덤 액션 시각화 GIF 생성")
    print("-" * 80)

    all_rewards = []
    all_lengths = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"\n[{ep+1}/{num_episodes}] 에피소드 {ep+1} 생성 중 (seed={episode_seed})...")

        # 랜덤 액션으로 에피소드 실행
        print("  - 에피소드 실행 중...")
        frames, episode_reward, episode_length = run_episode_with_random_actions(
            env, episode_seed,
            max_steps=env_config['max_steps']
        )
        print(f"    보상: {episode_reward:.2f}, 길이: {episode_length}")

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

        # 프레임에 정보 추가 (선택 사항)
        # frames_with_info = []
        # for i, frame in enumerate(frames):
        #     info_text = f"Step: {i}/{len(frames)-1} | Reward: {episode_reward:.1f}"
        #     frame_with_text = add_text_to_frame(frame, info_text)
        #     frames_with_info.append(frame_with_text)

        # GIF로 저장
        output_path = os.path.join(output_dir, f"random_episode_{ep+1:02d}_seed{episode_seed}.gif")
        save_as_gif(frames, output_path, fps=fps)

    # 최종 통계 출력
    print("\n" + "=" * 80)
    print("최종 통계 (Random Actions)")
    print("=" * 80)
    print(f"\n평균 리워드: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"평균 에피소드 길이: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}")
    print(f"최대 리워드: {np.max(all_rewards):.2f}")
    print(f"최소 리워드: {np.min(all_rewards):.2f}")

    print(f"\n✓ 모든 GIF가 저장되었습니다: {output_dir}")

    # 통계 파일 저장
    stats_path = os.path.join(output_dir, "random_visualization_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Wildfire Environment Random Actions Visualization Stats\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Environment Configuration:\n")
        f.write(f"  Grid size: {env_config['size']}x{env_config['size']}\n")
        f.write(f"  Max steps: {env_config['max_steps']}\n")
        f.write(f"  Initial fire size: {env_config['initial_fire_size']}\n")
        f.write(f"\nAgent Configuration:\n")
        f.write(f"  - Helicopters: {num_helicopters}\n")
        f.write(f"  - Trucks: {num_trucks}\n")
        f.write(f"  - Crews: {num_crews}\n")
        f.write(f"  - Total: {env.num_agents}\n")
        f.write(f"\nFire Dynamics:\n")
        f.write(f"  - Alpha (spread probability): {env_config['alpha']}\n")
        f.write(f"  - Beta (fire intensity): {env_config['beta']}\n")
        f.write(f"  - Delta beta (fire decay): {env_config['delta_beta']}\n\n")
        f.write(f"Visualization Parameters:\n")
        f.write(f"  - Number of episodes: {num_episodes}\n")
        f.write(f"  - Starting seed: {seed}\n")
        f.write(f"  - FPS: {fps}\n\n")
        f.write(f"Results:\n")
        for i in range(num_episodes):
            f.write(f"  Episode {i+1} (Seed {seed + i}):\n")
            f.write(f"    Reward: {all_rewards[i]:.2f}\n")
            f.write(f"    Length: {all_lengths[i]}\n")
        f.write(f"\nSummary:\n")
        f.write(f"  Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n")
        f.write(f"  Average Episode Length: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}\n")
        f.write(f"  Max Reward: {np.max(all_rewards):.2f}\n")
        f.write(f"  Min Reward: {np.min(all_rewards):.2f}\n")

    print(f"✓ 통계 저장: {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wildfire Environment 랜덤 액션 시각화")
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="생성할 GIF 수 (기본값: 3)"
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
        help="GIF를 저장할 디렉토리 (기본값: train_marllib_self/results/visualization/)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF의 프레임 속도 (기본값: 10)"
    )

    args = parser.parse_args()

    main(args.episodes, args.seed, args.output_dir, args.fps)
