"""
휴리스틱 기반 산불 진화 정책: 가장 가까운 활화목으로 이동 + 급수원 반환 메커니즘
+ 충돌 회피(Collision Avoidance) 로직 수정 (WildfireEnv 호환)

Phase 3/4 업데이트: wildfire.py의 급수원 반환 로직을 반영한 휴리스틱 정책입니다.

에이전트 상태별 행동:
1. RECHARGING 상태: 제자리 유지 (STILL)
2. RETURNING 상태: 급수원(home_pos)으로 이동
3. ACTIVE 상태: 가장 가까운 활화목으로 이동 (기존 로직)

충돌 회피 로직:
- env.agents 리스트를 직접 조회하여 다른 에이전트와의 좌표 충돌을 방지합니다.

실행 예시:
# 기본 실행
python train_marllib_self/heuristic_nearest_fire.py --episodes 10 --seed 42
"""

import sys
import os
import random
from pathlib import Path

# 프로젝트 경로 설정
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from wildfire_environment.core.constants import TILE_PIXELS
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# [헬퍼 함수: 이동 및 충돌 감지]
# ============================================================================

def get_action_vector(action):
    """액션 정수에 따른 (dx, dy) 반환"""
    # 0: STILL
    if action == 0: return (0, 0)
    # 1: NORTH (y 감소)
    if action == 1: return (0, -1)
    # 2: NORTH_EAST
    if action == 2: return (1, -1)
    # 3: EAST
    if action == 3: return (1, 0)
    # 4: SOUTH_EAST
    if action == 4: return (1, 1)
    # 5: SOUTH
    if action == 5: return (0, 1)
    # 6: SOUTH_WEST
    if action == 6: return (-1, 1)
    # 7: WEST
    if action == 7: return (-1, 0)
    # 8: NORTH_WEST
    if action == 8: return (-1, -1)
    return (0, 0)

def is_cell_occupied_by_agent(env, pos, self_agent):
    """
    특정 위치(pos)에 다른 에이전트가 있는지 확인
    (WildfireEnv/MultiGridEnv 호환 버전)
    """
    # 1. 그리드 범위 확인
    if not (0 <= pos[0] < env.grid.width and 0 <= pos[1] < env.grid.height):
        return True # 맵 밖은 이동 불가

    # 2. env.agents 리스트를 순회하며 위치 확인
    # WildfireEnv에서는 env.agents에 모든 에이전트 객체가 리스트로 저장되어 있음
    for other_agent in env.agents:
        # 자기 자신은 제외
        if other_agent is self_agent:
            continue
        
        # 위치 비교 (tuple로 변환하여 비교)
        other_pos = tuple(other_agent.pos) if isinstance(other_agent.pos, np.ndarray) else other_agent.pos
        check_pos = tuple(pos) if isinstance(pos, np.ndarray) else pos

        if other_pos == check_pos:
            return True
            
    return False

def get_best_valid_action(env, agent, target_pos):
    """
    충돌을 피하면서 target_pos로 가기 위한 최적의 액션을 반환.
    최단 경로가 막혀있으면 차선책(돌아가기)을 선택.
    """
    agent_pos = tuple(agent.pos) if isinstance(agent.pos, np.ndarray) else agent.pos
    
    # 가능한 모든 액션 (1~8)
    # 우선순위: 목표와의 거리가 줄어드는 액션 순서대로 정렬
    candidates = []
    for action in range(1, 9):
        dx, dy = get_action_vector(action)
        next_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
        
        # 목표까지의 거리 (Manhattan distance)
        dist = abs(target_pos[0] - next_pos[0]) + abs(target_pos[1] - next_pos[1])
        candidates.append((dist, action, next_pos))
    
    # 거리 오름차순 정렬 (가장 가까운 곳 부터 시도)
    candidates.sort(key=lambda x: x[0])
    
    # 유효한(빈) 칸 찾기
    for _, action, next_pos in candidates:
        if not is_cell_occupied_by_agent(env, next_pos, agent):
            return action
            
    # 모든 방향이 막혀있으면 STILL (0)
    return 0


# ============================================================================
# [휴리스틱 액션 선택 함수]
# ============================================================================

def select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs):
    """
    휴리스틱 정책으로 액션 선택 + 충돌 회피 적용
    """
    # 에이전트 객체 찾기
    try:
        agent_idx = int(agent_id)
        if agent_idx >= len(env.agents):
            return 0
    except (ValueError, TypeError):
        return 0

    agent = env.agents[agent_idx]
    agent_pos = tuple(agent.pos) if isinstance(agent.pos, np.ndarray) else agent.pos

    from wildfire_environment.core.agent import AgentState

    # ----------------------------------------------------------
    # 1. 상태에 따른 목표 설정
    # ----------------------------------------------------------
    target_pos = None

    # RECHARGING 상태: 제자리 유지
    if hasattr(agent, 'state') and agent.state == AgentState.RECHARGING:
        return 0  # STILL

    # RETURNING 상태: 급수원으로 이동
    if hasattr(agent, 'state') and agent.state == AgentState.RETURNING:
        if hasattr(agent, 'home_pos') and agent.home_pos is not None:
            target_pos = tuple(agent.home_pos)
        else:
            return 0

    # ACTIVE 상태: 활화목 탐색
    elif hasattr(agent, 'state') and agent.state == AgentState.ACTIVE:
        grid = env.grid
        fire_positions = []

        if partial_obs:
            partial_view_size = agent.partial_obs_size
            half_view = partial_view_size // 2
            x_min = max(0, agent_pos[0] - half_view)
            x_max = min(grid.width - 1, agent_pos[0] + half_view)
            y_min = max(0, agent_pos[1] - half_view)
            y_max = min(grid.height - 1, agent_pos[1] + half_view)
        else:
            x_min, x_max = 0, grid.width - 1
            y_min, y_max = 0, grid.height - 1

        # helper_grid를 사용하여 트리 정보를 가져옵니다 (env.grid에는 agent가 있을 수 있음)
        # wildfire.py에서는 나무 정보를 helper_grid에 저장합니다.
        check_grid = env.helper_grid if hasattr(env, 'helper_grid') else env.grid
        
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                cell = check_grid.get(x, y)
                if cell and hasattr(cell, 'state') and cell.state == 1: # 1: on fire
                    fire_positions.append((x, y))

        if not fire_positions:
            return 0

        # 가장 가까운 활화목 찾기
        min_dist = float('inf')
        nearest_fire = None

        for fire_pos in fire_positions:
            dist = abs(fire_pos[0] - agent_pos[0]) + abs(fire_pos[1] - agent_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_fire = fire_pos
        
        target_pos = nearest_fire

    # ----------------------------------------------------------
    # 2. 목표가 없으면 정지
    # ----------------------------------------------------------
    if target_pos is None:
        return 0

    # ----------------------------------------------------------
    # 3. 충돌 회피 이동 로직 적용
    # ----------------------------------------------------------
    # 목표 위치와 동일하다면 정지
    target_pos = tuple(target_pos)
    if target_pos == agent_pos:
        return 0

    # 단순히 방향만 구하는 것이 아니라, 실제 갈 수 있는 최적의 칸을 계산
    final_action = get_best_valid_action(env, agent, target_pos)
    
    return final_action


# ============================================================================
# [휴리스틱 에이전트 실행]
# ============================================================================

def run_heuristic_episode(env, env_config, seed, max_steps=300, render=False):
    """
    휴리스틱 정책으로 에피소드 실행
    """
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

    if render:
        frame = env.render(mode='rgb_array')
        for agent in env.agents:
            frame = render_activity_gauge(frame, agent)
            frame = render_water_gauge(frame, agent)
            frame = render_supply_source_marker(frame, agent)
        frame = render_agent_status_panel(frame, env.agents, step)
        frames.append(frame)

    while not done and step < max_steps:
        # 1. 에이전트 순서를 섞어서 행동 결정 (우선순위 편향 방지)
        agent_ids = list(obs_dict.keys())
        # random.shuffle(agent_ids) 

        actions = {}
        for agent_id in agent_ids:
            obs = obs_dict[agent_id]
            # 수정된 select_heuristic_action 함수 호출
            action = select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs)
            actions[agent_id] = action

            if agent_id not in episode_data['actions']:
                episode_data['actions'][agent_id] = []
            episode_data['actions'][agent_id].append(action)

        # 2. 환경 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        step_reward = np.mean(list(reward_dict.values()))
        episode_reward += step_reward
        episode_data['rewards'].append(step_reward)

        # 활화목 통계
        # helper_grid를 사용하는 것이 더 정확함 (WildfireEnv)
        check_grid = env.helper_grid if hasattr(env, 'helper_grid') else env.grid
        fire_count = 0
        for x in range(check_grid.width):
            for y in range(check_grid.height):
                cell = check_grid.get(x, y)
                if cell and hasattr(cell, 'state') and cell.state == 1:
                    fire_count += 1
        episode_data['fire_counts'].append(fire_count)

        if render:
            frame = env.render(mode='rgb_array')
            for agent in env.agents:
                frame = render_activity_gauge(frame, agent)
                frame = render_water_gauge(frame, agent)
                frame = render_supply_source_marker(frame, agent)
            frame = render_agent_status_panel(frame, env.agents, step)
            frames.append(frame)

        step += 1

    if render:
        return episode_reward, step, episode_data, frames
    else:
        return episode_reward, step, episode_data


# ============================================================================
# [시각화 함수]
# ============================================================================

def render_activity_gauge(frame, agent, tile_size=TILE_PIXELS):
    if not hasattr(agent, 'max_active_time') or agent.max_active_time == 0:
        return frame
    fill_ratio = agent.active_time_remaining / agent.max_active_time
    pos = agent.pos
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 2
    frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = [0, 255, 255]
    return frame

def render_water_gauge(frame, agent, tile_size=TILE_PIXELS):
    if not hasattr(agent, 'max_water') or agent.max_water == 0:
        return frame
    fill_ratio = agent.water_remaining / agent.max_water
    pos = agent.pos
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 6
    frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = [0, 255, 255]
    return frame

def render_supply_source_marker(frame, agent, tile_size=TILE_PIXELS):
    if not hasattr(agent, 'home_pos') or agent.home_pos is None:
        return frame
    home_pos = agent.home_pos
    home_x = home_pos[0] * tile_size
    home_y = home_pos[1] * tile_size
    box_thickness = 2
    frame[max(0, home_y):min(frame.shape[0], home_y+box_thickness),
          max(0, home_x):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]
    frame[max(0, home_y+tile_size-box_thickness):min(frame.shape[0], home_y+tile_size),
          max(0, home_x):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]
    frame[max(0, home_y):min(frame.shape[0], home_y+tile_size),
          max(0, home_x):min(frame.shape[1], home_x+box_thickness)] = [0, 0, 255]
    frame[max(0, home_y):min(frame.shape[0], home_y+tile_size),
          max(0, home_x+tile_size-box_thickness):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]
    return frame

def render_agent_status_panel(frame, agents, step):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    panel_width = 200
    panel_x = frame.shape[1] - panel_width
    y_offset = 10
    draw.text((panel_x + 5, y_offset), f"Step: {step}", fill=(255, 255, 255), font=font)
    y_offset += 15
    from wildfire_environment.core.agent import AgentState
    state_names = {AgentState.ACTIVE: "ACTIVE", AgentState.RETURNING: "RETURNING", AgentState.RECHARGING: "RECHARGING"}
    for agent_idx, agent in enumerate(agents):
        agent_text = f"Agent {agent_idx}:"
        draw.text((panel_x + 5, y_offset), agent_text, fill=(255, 255, 255), font=font)
        y_offset += 12
        state = state_names.get(agent.state, "UNKNOWN")
        state_color = (0, 255, 0) if agent.state == AgentState.ACTIVE else (255, 255, 0) if agent.state == AgentState.RETURNING else (255, 0, 0)
        draw.text((panel_x + 10, y_offset), f"State: {state}", fill=state_color, font=font)
        y_offset += 12
        if hasattr(agent, 'max_active_time'):
            draw.text((panel_x + 10, y_offset), f"Active: {agent.active_time_remaining}/{agent.max_active_time}", fill=(200, 200, 200), font=font)
            y_offset += 12
        if hasattr(agent, 'max_water'):
            draw.text((panel_x + 10, y_offset), f"Water: {agent.water_remaining:.1f}/{agent.max_water}", fill=(100, 200, 255), font=font)
            y_offset += 12
        if agent.state == AgentState.RECHARGING and hasattr(agent, 'recharge_time'):
            draw.text((panel_x + 10, y_offset), f"Recharge: {agent.recharge_time_remaining}/{agent.recharge_time}", fill=(255, 165, 0), font=font)
            y_offset += 12
        y_offset += 5
    return np.array(img)

def save_as_gif(frames, filename, fps=10):
    if not frames:
        print("저장할 프레임이 없습니다!")
        return
    images = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)
    file_size = os.path.getsize(filename) / 1024
    print(f"  ✓ GIF 저장 완료: {filename} ({len(frames)} frames, {file_size:.1f} KB)")


# ============================================================================
# [메인 함수]
# ============================================================================

def main(num_episodes=10, seed=42, output_dir=None, visualize=False):
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    if output_dir is None:
        output_dir = "train_marllib_self/results/heuristic"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("휴리스틱 정책 기반 산불 진화 에이전트 (충돌 회피 + WildfireEnv 호환)")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 에피소드 수: {num_episodes}")
    print(f"  - 시작 시드: {seed}")
    print(f"  - 부분 관찰: {env_config['partial_obs']}")
    print(f"  - 시각화: {visualize}")
    print(f"  - 그리드 크기: {env_config['size']}x{env_config['size']}")
    print(f"  - 에이전트 수: {env_config['num_agents']}")
    print(f"  - 출력 디렉토리: {output_dir}")
    print("=" * 80)

    print("\n환경 생성 중...")
    env = WildfireEnv(**env_config)
    print(f"✓ 환경 생성 완료")

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

        if visualize and frames:
            gif_filename = f"heuristic_ep{ep+1:02d}_seed{episode_seed}.gif"
            gif_path = os.path.join(output_dir, gif_filename)
            save_as_gif(frames, gif_path, fps=10)

    print("\n" + "=" * 80)
    print("최종 통계")
    print("=" * 80)

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    avg_length = np.mean(all_lengths)

    print(f"\n평균 리워드: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"평균 에피소드 길이: {avg_length:.2f}")

    partial_obs_value = env_config['partial_obs']
    stats_filename = f"heuristic_stats_partial_obs_{partial_obs_value}.txt"
    stats_path = os.path.join(output_dir, stats_filename)

    with open(stats_path, 'w') as f:
        f.write("휴리스틱 정책 산불 진화 시뮬레이션 결과\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"평균 리워드: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"평균 에피소드 길이: {avg_length:.2f}\n")

    print(f"\n✓ 통계 저장: {stats_path}")

    import pickle
    results_data = {
        'rewards': all_rewards,
        'lengths': all_lengths,
        'episode_data': all_episode_data,
        'config': env_config
    }
    results_filename = f"heuristic_results_partial_obs_{partial_obs_value}.pkl"
    results_path = os.path.join(output_dir, results_filename)
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"✓ 결과 데이터 저장: {results_path}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    main(args.episodes, args.seed, args.output_dir, args.visualize)