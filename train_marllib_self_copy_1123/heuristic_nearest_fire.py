"""
휴리스틱 기반 산불 진화 정책: 가장 가까운 활화목으로 이동

학습된 MAPPO, PPO 등의 정책과 비교할 휴리스틱 정책입니다.
- partial_obs=False인 경우: 전체 그리드에서 각 에이전트와 가장 가까운 불타는 나무로 이동
- partial_obs=True인 경우: 에이전트 중심으로 partial_obs_size x partial_obs_size 관찰 공간에서 가장 가까운 불타는 나무로 이동
  (agent.py Line 112: self.partial_obs_size = 5)

partial_obs는 environment.py의 ENV_CONFIG['partial_obs']에서 읽어옵니다.

실행 예시:
# 기본 실행 (통계만 저장)
python train_marllib_self/heuristic_nearest_fire.py \
    --episodes 10 \
    --seed 42

# GIF 시각화 포함
python train_marllib_self/heuristic_nearest_fire.py \
    --episodes 5 \
    --seed 42 \
    --visualize

# 커스텀 출력 디렉토리 + 시각화
python train_marllib_self/heuristic_nearest_fire.py \
    --episodes 3 \
    --seed 100 \
    --visualize \
    --output-dir ./my_results/
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from PIL import Image


# ============================================================================
# [휴리스틱 액션 선택 함수]
# ============================================================================

def select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs):
    """
    휴리스틱 정책으로 액션 선택

    partial_obs=False: 전체 그리드에서 가장 가까운 활화목으로 이동
    partial_obs=True: 에이전트 중심으로 partial_obs_size x partial_obs_size 관찰 공간 내에서 가장 가까운 활화목으로 이동

    Note: wildfire.py와 일치하게 구현 (agent를 중심으로 half_view 범위)
    - agent.py Line 112: self.partial_obs_size = 5
    - wildfire.py Line 592: half_view = partial_view_size // 2 = 2
    - 따라서 5x5 영역 (-2 ~ +2)을 에이전트 중심으로 관찰

    Parameters
    ----------
    env : WildfireEnv
        환경
    agent_id : str
        에이전트 ID
    obs : np.ndarray
        에이전트의 관찰값
    obs_dict : dict
        모든 에이전트의 관찰값
    env_config : dict
        환경 설정
    partial_obs : bool
        부분 관찰 여부

    Returns
    -------
    int
        선택한 액션 (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
    """
    # 에이전트 위치 찾기
    try:
        agent_idx = int(agent_id)
        if agent_idx >= len(env.agents):
            return 4  # STAY
    except (ValueError, TypeError):
        return 4

    agent = env.agents[agent_idx]
    agent_pos = agent.pos  # (x, y)

    # 활화목 찾기
    grid = env.grid
    fire_positions = []

    if partial_obs:
        # 에이전트 중심으로 partial_obs_size x partial_obs_size 관찰 공간 내에서만 활화목 찾기
        # wildfire.py Line 592: half_view = partial_view_size // 2
        # agent.py Line 112: self.partial_obs_size = 5
        partial_view_size = agent.partial_obs_size
        half_view = partial_view_size // 2  # 5 // 2 = 2
        x_min = max(0, agent_pos[0] - half_view)
        x_max = min(grid.width - 1, agent_pos[0] + half_view)
        y_min = max(0, agent_pos[1] - half_view)
        y_max = min(grid.height - 1, agent_pos[1] + half_view)
    else:
        # 전체 그리드에서 활화목 찾기
        x_min = 0
        x_max = grid.width - 1
        y_min = 0
        y_max = grid.height - 1

    # 활화목 수집
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            cell = grid.get(x, y)
            # Tree 객체에서 state = 1이면 "on fire" 상태
            if cell and hasattr(cell, 'state') and cell.state == 1:
                fire_positions.append((x, y))

    # 활화목이 없으면 STAY
    if not fire_positions:
        return 4

    # 가장 가까운 활화목 찾기
    min_dist = float('inf')
    nearest_fire = None

    for fire_pos in fire_positions:
        dist = abs(fire_pos[0] - agent_pos[0]) + abs(fire_pos[1] - agent_pos[1])
        if dist < min_dist:
            min_dist = dist
            nearest_fire = fire_pos

    if nearest_fire is None:
        return 4

    # 가장 가까운 활화목 방향으로 이동
    return _move_towards(agent_pos, nearest_fire)


def _move_towards(from_pos, to_pos):
    """
    from_pos에서 to_pos 방향으로 한 칸 이동하는 액션 반환

    WildfireActions 매핑:
    - 0: STILL
    - 1: NORTH (y 감소)
    - 2: NORTH_EAST (x 증가, y 감소)
    - 3: EAST (x 증가)
    - 4: SOUTH_EAST (x 증가, y 증가)
    - 5: SOUTH (y 증가)
    - 6: SOUTH_WEST (x 감소, y 증가)
    - 7: WEST (x 감소)
    - 8: NORTH_WEST (x 감소, y 감소)

    Parameters
    ----------
    from_pos : tuple
        현재 위치 (x, y)
    to_pos : tuple
        목표 위치 (x, y)

    Returns
    -------
    int
        액션 (0: STILL, 1-8: 8방향 이동)
    """
    x_diff = to_pos[0] - from_pos[0]
    y_diff = to_pos[1] - from_pos[1]

    # 방향 결정
    if x_diff == 0 and y_diff == 0:
        return 0  # STILL

    # 8방향 이동
    if y_diff < 0:  # 위로 이동 (북쪽)
        if x_diff < 0:
            return 8  # NORTH_WEST
        elif x_diff > 0:
            return 2  # NORTH_EAST
        else:
            return 1  # NORTH
    elif y_diff > 0:  # 아래로 이동 (남쪽)
        if x_diff < 0:
            return 6  # SOUTH_WEST
        elif x_diff > 0:
            return 4  # SOUTH_EAST
        else:
            return 5  # SOUTH
    else:  # y_diff == 0
        if x_diff < 0:
            return 7  # WEST
        elif x_diff > 0:
            return 3  # EAST
        else:
            return 0  # STILL


# ============================================================================
# [휴리스틱 에이전트 실행]
# ============================================================================

def run_heuristic_episode(env, env_config, seed, max_steps=300, render=False):
    """
    휴리스틱 정책으로 에피소드 실행

    Parameters
    ----------
    env : WildfireEnv
        환경
    env_config : dict
        환경 설정
    seed : int
        랜덤 시드
    max_steps : int
        최대 스텝 수
    render : bool
        렌더링 여부 (GIF 생성용)

    Returns
    -------
    tuple
        (episode_reward, episode_length, episode_data) 또는
        (episode_reward, episode_length, episode_data, frames) if render=True
    """
    # 환경 리셋
    obs_dict, _ = env.reset(seed=seed)

    partial_obs = env_config.get('partial_obs', False)

    episode_reward = 0
    step = 0
    done = False

    episode_data = {
        'seed': seed,
        'rewards': [],
        'actions': {},
        'fire_counts': [],
        'partial_obs': partial_obs
    }

    frames = [] if render else None

    # 초기 프레임
    if render:
        frame = env.render(mode='rgb_array')
        frames.append(frame)

    while not done and step < max_steps:
        # 각 에이전트의 액션 선택 (휴리스틱)
        actions = {}

        for agent_id in obs_dict.keys():
            obs = obs_dict[agent_id]
            action = select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs)
            actions[agent_id] = action

            if agent_id not in episode_data['actions']:
                episode_data['actions'][agent_id] = []
            episode_data['actions'][agent_id].append(action)

        # 환경 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        # 리워드 합산
        step_reward = np.mean(list(reward_dict.values()))
        episode_reward += step_reward
        episode_data['rewards'].append(step_reward)

        # 활화목 개수 기록
        grid = env.grid
        fire_count = 0
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell and hasattr(cell, 'fire') and cell.fire > 0.5:
                    fire_count += 1
        episode_data['fire_counts'].append(fire_count)

        # 프레임 저장
        if render:
            frame = env.render(mode='rgb_array')
            frames.append(frame)

        step += 1

    if render:
        return episode_reward, step, episode_data, frames
    else:
        return episode_reward, step, episode_data


# ============================================================================
# [GIF 저장 함수]
# ============================================================================

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
    print(f"  ✓ GIF 저장 완료: {filename} ({len(frames)} frames, {file_size:.1f} KB)")


# ============================================================================
# [메인 함수]
# ============================================================================

def main(num_episodes=10, seed=42, output_dir=None, visualize=False):
    """
    휴리스틱 정책 기반 산불 진화 시뮬레이션 실행

    Parameters
    ----------
    num_episodes : int
        실행할 에피소드 수
    seed : int
        시작 랜덤 시드
    output_dir : str, optional
        결과 저장 디렉토리
    visualize : bool
        GIF 시각화 여부 (기본값: False)

    Note:
        partial_obs는 environment.py의 ENV_CONFIG['partial_obs']에서 읽어옵니다.
    """
    # 환경 설정 복사 (partial_obs는 ENV_CONFIG에서 읽어옴)
    env_config = {k: v for k, v in ENV_CONFIG.items()}

    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = "train_marllib_self/results/heuristic"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("휴리스틱 정책 기반 산불 진화 에이전트")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 에피소드 수: {num_episodes}")
    print(f"  - 시작 시드: {seed}")
    print(f"  - 부분 관찰: {env_config['partial_obs']}")
    print(f"  - 시각화: {visualize}")
    print(f"  - 그리드 크기: {env_config['size']}x{env_config['size']}")
    print(f"  - 에이전트 수: {env_config['num_agents']}")
    print(f"  - 최대 스텝: {env_config['max_steps']}")
    print(f"  - 출력 디렉토리: {output_dir}")
    print("=" * 80)

    # 환경 생성
    print("\n환경 생성 중...")
    env = WildfireEnv(**env_config)
    print(f"✓ 환경 생성 완료")

    # 에피소드 실행
    print(f"\n휴리스틱 정책 에피소드 실행")
    print("-" * 80)

    all_rewards = []
    all_lengths = []
    all_episode_data = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"\n[{ep+1}/{num_episodes}] 에피소드 {ep+1} 실행 중 (seed={episode_seed})...")

        result = run_heuristic_episode(
            env, env_config, episode_seed, max_steps=env_config['max_steps'], render=visualize
        )

        if visualize:
            episode_reward, episode_length, episode_data, frames = result
        else:
            episode_reward, episode_length, episode_data = result

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_episode_data.append(episode_data)

        print(f"  보상: {episode_reward:.2f}")
        print(f"  길이: {episode_length}")
        if episode_data['fire_counts']:
            print(f"  최종 활화목: {episode_data['fire_counts'][-1]}")

        # GIF 저장
        if visualize and frames:
            gif_filename = f"heuristic_ep{ep+1:02d}_seed{episode_seed}.gif"
            gif_path = os.path.join(output_dir, gif_filename)
            save_as_gif(frames, gif_path, fps=10)

    # 최종 통계 출력 및 저장
    print("\n" + "=" * 80)
    print("최종 통계 (휴리스틱 정책)")
    print("=" * 80)

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    avg_length = np.mean(all_lengths)

    print(f"\n평균 리워드: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"평균 에피소드 길이: {avg_length:.2f}")
    print(f"최고 리워드: {np.max(all_rewards):.2f}")
    print(f"최저 리워드: {np.min(all_rewards):.2f}")

    # 통계 파일 저장
    partial_obs_value = env_config['partial_obs']
    stats_filename = f"heuristic_stats_partial_obs_{partial_obs_value}.txt"
    stats_path = os.path.join(output_dir, stats_filename)

    with open(stats_path, 'w') as f:
        f.write("휴리스틱 정책 산불 진화 시뮬레이션 결과\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"설정:\n")
        f.write(f"  - 에피소드 수: {num_episodes}\n")
        f.write(f"  - 시작 시드: {seed}\n")
        f.write(f"  - 부분 관찰: {partial_obs_value}\n")
        f.write(f"  - 그리드 크기: {env_config['size']}x{env_config['size']}\n")
        f.write(f"  - 에이전트 수: {env_config['num_agents']}\n")
        f.write(f"  - 최대 스텝: {env_config['max_steps']}\n\n")
        f.write("-" * 80 + "\n")
        f.write("에피소드별 결과:\n")
        f.write("-" * 80 + "\n")

        for i, (reward, length) in enumerate(zip(all_rewards, all_lengths)):
            episode_seed = seed + i
            f.write(f"Episode {i+1} (Seed {episode_seed}):\n")
            f.write(f"  Reward: {reward:.2f}\n")
            f.write(f"  Length: {length}\n\n")

        f.write("-" * 80 + "\n")
        f.write("통계:\n")
        f.write("-" * 80 + "\n")
        f.write(f"평균 리워드: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"평균 에피소드 길이: {avg_length:.2f}\n")
        f.write(f"최고 리워드: {np.max(all_rewards):.2f}\n")
        f.write(f"최저 리워드: {np.min(all_rewards):.2f}\n")

    print(f"\n✓ 통계 저장: {stats_path}")

    # 결과 데이터 저장 (pickle)
    import pickle
    results_data = {
        'rewards': all_rewards,
        'lengths': all_lengths,
        'episode_data': all_episode_data,
        'config': env_config,
        'partial_obs': partial_obs_value
    }

    results_filename = f"heuristic_results_partial_obs_{partial_obs_value}.pkl"
    results_path = os.path.join(output_dir, results_filename)

    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)

    print(f"✓ 결과 데이터 저장: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="휴리스틱 정책 기반 산불 진화 에이전트")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="실행할 에피소드 수 (기본값: 10)"
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
        help="결과 저장 디렉토리 (기본값: train_marllib_self/results/heuristic)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="GIF 시각화 생성 여부"
    )

    args = parser.parse_args()

    main(args.episodes, args.seed, args.output_dir, args.visualize)
