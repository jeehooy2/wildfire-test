import sys
import os
from pathlib import Path
import heapq # 최단 경로/거리를 찾기 위해 힙큐를 사용 (선택 사항)

# 프로젝트 경로 설정 (생략된 부분)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from wildfire_environment.core.constants import TILE_PIXELS
from wildfire_environment.core.agent import AgentState # AgentState 임포트
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# [유틸리티 함수]
# ============================================================================

def _get_neighbors(pos):
    """주어진 위치의 8방향 인접 타일 위치를 반환합니다."""
    x, y = pos
    neighbors = []
    # 액션 순서 (1: NORTH ~ 8: NORTH_WEST)와 일치시킬 필요는 없으나,
    # 일반적으로 x, y 변화량으로 정의합니다.
    # dx, dy: (0,-1) N, (1,-1) NE, (1,0) E, (1,1) SE, (0,1) S, (-1,1) SW, (-1,0) W, (-1,-1) NW
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbors.append((x + dx, y + dy))
    return neighbors

def _is_position_vacant(env, pos):
    """
    주어진 위치가 에이전트 이동 가능한지 확인 (경계 및 다른 에이전트 점유 여부).
    can_overlap=False 환경을 가정합니다.
    """
    x, y = pos
    grid = env.grid
    
    # 1. 경계 확인
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return False
        
    # 2. 다른 에이전트 점유 확인
    # env.agents 리스트를 순회하며 현재 위치에 다른 에이전트가 있는지 확인
    for other_agent in env.agents:
        # 자기 자신은 제외
        if other_agent.pos[0] == x and other_agent.pos[1] == y:
            # 현재 에이전트가 자기 자신의 위치를 체크하는 경우 (예: 목표 지점 확인 시)
            # 이 함수는 목표 지점에 '다른' 에이전트가 있는지 확인해야 하므로,
            # 현재 에이전트가 목표 지점에 있다면 True를 반환해야 하지만, 
            # 이 함수는 일반적으로 '비어있는지' 확인하기 위해 사용되므로,
            # 목표 지점에 에이전트가 있으면 False로 간주하는 것이 안전합니다.
            # 다만, 이 정책에서는 현재 위치(agent_pos)를 체크하지 않습니다.
            continue
            
        if np.array_equal(other_agent.pos, pos):
            # 다른 에이전트가 해당 위치에 있음
            return False
            
    # 3. 환경 내 다른 장애물 확인 (이 환경의 WildfireEnv에서는 에이전트를 제외한 장애물은 없다고 가정)
    
    return True

def _get_action_from_diff(x_diff, y_diff):
    """
    좌표 차이로부터 WildfireActions 액션 인덱스 반환
    """
    if x_diff == 0 and y_diff == 0:
        return 0  # STILL

    # y 감소 (북쪽)
    if y_diff < 0:
        if x_diff < 0: return 8  # NORTH_WEST
        if x_diff > 0: return 2  # NORTH_EAST
        return 1  # NORTH
    # y 증가 (남쪽)
    elif y_diff > 0:
        if x_diff < 0: return 6  # SOUTH_WEST
        if x_diff > 0: return 4  # SOUTH_EAST
        return 5  # SOUTH
    # y_diff == 0
    else:
        if x_diff < 0: return 7  # WEST
        if x_diff > 0: return 3  # EAST
        return 0 # STILL (이미 위에서 처리됨)


def _move_towards_safely(env, from_pos, to_pos):
    """
    안전하게 목표(to_pos) 방향으로 이동할 수 있는 액션 선택.
    주변 8방향 중 비어있고 목표에 가장 가까운 타일을 찾습니다.
    """
    # 1. 목표 방향으로의 직접 이동 시도
    x_diff = to_pos[0] - from_pos[0]
    y_diff = to_pos[1] - from_pos[1]

    # 한 스텝의 목표 위치 (직접적인 목표)
    target_x = from_pos[0] + np.sign(x_diff)
    target_y = from_pos[1] + np.sign(y_diff)
    
    direct_target_pos = (target_x, target_y)

    # 직접적인 목표 위치가 비어있다면 바로 이동
    if _is_position_vacant(env, direct_target_pos):
        return _get_action_from_diff(np.sign(x_diff), np.sign(y_diff))

    # 2. 직접 이동 불가 시, 주변 8방향 중 최적의 위치 탐색
    # (distance, action_index) 튜플을 저장할 리스트
    possible_moves = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue

            next_pos = (from_pos[0] + dx, from_pos[1] + dy)
            
            # 다음 위치가 비어있고 경계 내에 있는지 확인
            if _is_position_vacant(env, next_pos):
                # 8방향 이동 액션 계산
                action = _get_action_from_diff(dx, dy)
                
                # 목표까지의 맨해튼 거리 또는 유클리드 거리
                # 맨해튼 거리: dist = abs(next_pos[0] - to_pos[0]) + abs(next_pos[1] - to_pos[1])
                # 유클리드 거리: dist = np.sqrt((next_pos[0] - to_pos[0])**2 + (next_pos[1] - to_pos[1])**2)
                # 여기서는 간단히 맨해튼 거리를 사용 (유클리드도 가능)
                dist = abs(next_pos[0] - to_pos[0]) + abs(next_pos[1] - to_pos[1])

                possible_moves.append((dist, action))

    # 이동 가능한 위치가 있으면, 목표에 가장 가까운 곳으로 이동
    if possible_moves:
        # 거리가 가장 짧은(최소) 액션 선택
        possible_moves.sort(key=lambda x: x[0])
        return possible_moves[0][1] # 가장 짧은 거리의 액션 반환

    # 주변 8방향 모두 이동 불가 (에이전트에게 포위된 상황)
    return 0 # STILL


# ============================================================================
# [휴리스틱 액션 선택 함수]
# ============================================================================

def select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs):
    """
    휴리스틱 정책으로 액션 선택 (can_overlap=False에 대한 충돌 회피/탈출 로직 추가)
    """
    try:
        agent_idx = int(agent_id)
        if agent_idx >= len(env.agents):
            return 0  # STILL
    except (ValueError, TypeError):
        return 0

    agent = env.agents[agent_idx]
    agent_pos = tuple(agent.pos) if isinstance(agent.pos, np.ndarray) else agent.pos
    grid = env.grid

    # Phase 3/4: 에이전트 상태에 따른 휴리스틱 선택

    # 1. RECHARGING 상태: 제자리 유지
    if hasattr(agent, 'state') and agent.state == AgentState.RECHARGING:
        return 0  # STILL

    # 2. RETURNING 상태: 급수원으로 이동 또는 급수원 탈출
    if hasattr(agent, 'state') and agent.state == AgentState.RETURNING:
        if hasattr(agent, 'home_pos') and agent.home_pos is not None:
            home_pos = agent.home_pos
            
            # 급수원에 이미 도착한 경우 (recharge 상태가 아닐 때 = 충전 대기/완료 후)
            if agent_pos == home_pos:
                # 급수원 타일에서 벗어나기 위한 "탈출" 로직
                # 다른 에이전트에게 자리를 내주기 위해 가장 가까운 빈 타일로 이동
                
                min_dist = float('inf')
                best_escape_pos = None

                for next_pos in _get_neighbors(agent_pos):
                    # 경계 내에 있고, 다른 에이전트가 점유하고 있지 않은 타일 중
                    if _is_position_vacant(env, next_pos):
                        # 급수원 타일과 충돌하지 않는 비어있는 위치를 찾습니다.
                        # (거리 1이므로 모든 이웃 타일은 급수원 타일이 아님)
                        
                        # 가장 가까운 활화목까지의 거리를 최소화하는 탈출 경로를 찾는 것도 가능하나,
                        # 여기서는 단순히 탈출을 위해 비어있는 이웃 중 아무 곳이나 선택합니다.
                        # (가장 가까운 활화목을 목표로 하는 'ACTIVE' 상태로의 전환을 유도하기 위해)
                        
                        # 모든 이웃이 막혀있다면, 여기서 best_escape_pos는 None으로 남습니다.
                        # 여기서는 단순히 첫 번째 발견된 빈 타일로 이동합니다.
                        best_escape_pos = next_pos
                        break

                if best_escape_pos:
                    # 탈출 방향으로 이동
                    return _move_towards_safely(env, agent_pos, best_escape_pos)
                else:
                    # 주변이 모두 막혀 탈출 불가. 제자리 유지 (STILL)
                    return 0
            
            # 급수원으로 이동 중인 경우: 안전하게 이동
            else:
                return _move_towards_safely(env, agent_pos, home_pos)

        else:
            return 0  # STILL (home_pos 없음)

    # 3. ACTIVE 상태: 가장 가까운 활화목으로 이동
    
    # 활화목 찾기 (기존 로직 유지)
    fire_positions = []
    
    # ... (기존 활화목 탐색 로직 유지) ...
    if partial_obs:
        partial_view_size = agent.partial_obs_size
        half_view = partial_view_size // 2
        x_min = max(0, agent_pos[0] - half_view)
        x_max = min(grid.width - 1, agent_pos[0] + half_view)
        y_min = max(0, agent_pos[1] - half_view)
        y_max = min(grid.height - 1, agent_pos[1] + half_view)
    else:
        x_min = 0
        x_max = grid.width - 1
        y_min = 0
        y_max = grid.height - 1

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            cell = grid.get(x, y)
            if cell and hasattr(cell, 'state') and cell.state == 1:
                fire_positions.append((x, y))

    if not fire_positions:
        return 0 # 활화목이 없으면 STAY

    # 가장 가까운 활화목 찾기 (기존 로직 유지)
    min_dist = float('inf')
    nearest_fire = None

    for fire_pos in fire_positions:
        # 에이전트 위치와 활화목 위치가 같으면 목표에 도달한 것이므로, 이동할 필요 없음 (STILL)
        if fire_pos == agent_pos:
            return 0 
            
        dist = abs(fire_pos[0] - agent_pos[0]) + abs(fire_pos[1] - agent_pos[1])
        if dist < min_dist:
            min_dist = dist
            nearest_fire = fire_pos

    if nearest_fire is None:
        return 0

    # 가장 가까운 활화목 방향으로 안전하게 이동
    return _move_towards_safely(env, agent_pos, nearest_fire)


# ######################################################################
# NOTE: 기존 _move_towards 함수는 충돌 방지 로직이 없으므로,
# 새롭게 정의된 _move_towards_safely 함수를 사용하도록 변경합니다.
# 기존 _move_towards 함수는 사용하지 않으므로 삭제하거나 주석 처리합니다.
# ######################################################################

def _move_towards(from_pos, to_pos):
    """
    기존 _move_towards 함수는 충돌 방지 로직이 없으므로,
    can_overlap=False 환경에서는 사용하지 않고 _move_towards_safely를 사용합니다.
    (호환성을 위해 남겨두지만 실제로는 _move_towards_safely가 사용되어야 합니다.)
    """
    x_diff = to_pos[0] - from_pos[0]
    y_diff = to_pos[1] - from_pos[1]
    
    return _get_action_from_diff(np.sign(x_diff), np.sign(y_diff))


# ============================================================================
# [휴리스틱 에이전트 실행 및 기타 함수]
# ============================================================================

# ... (run_heuristic_episode, render_* 등의 기타 함수는 변경 없이 유지) ...

# run_heuristic_episode 내에서 select_heuristic_action을 호출하므로,
# 나머지 부분은 원본 코드를 그대로 사용하면 됩니다.
# 원본 코드에서 생략된 함수들을 다시 삽입하거나, 새 코드를 원본 파일에 덮어쓰세요.