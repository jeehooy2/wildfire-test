"""
Tree 객체의 state 값이 제대로 저장되고 있는지 확인
"""
import sys
import os
from pathlib import Path

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from wildfire_environment.core.agent import AgentState

def run_tree_debug():
    """Tree 객체의 state 값 확인"""

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    obs_dict, _ = env.reset(seed=42)

    print("=" * 100)
    print("Step 40: Agent 5가 홈에서 ACTIVE 상태 (활화목=46개)")
    print("=" * 100)

    # Step 40까지 실행
    for step in range(40):
        actions = {str(i): 0 for i in range(len(env.agents))}
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        if step >= 300:
            break

    # Step 40: 그리드 상태 확인
    print("\n현재 상태:")
    agent_5 = env.agents[5]
    print(f"Agent 5: pos={agent_5.pos}, home={agent_5.home_pos}, state={agent_5.state.name}")

    grid = env.grid

    # 전체 활화목 수 세기
    total_fires = 0
    fire_positions = []
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and hasattr(cell, 'state'):
                if cell.state == 1:
                    total_fires += 1
                    fire_positions.append((x, y))

    print(f"\n전체 활화목: {total_fires}개")
    print(f"활화목 위치 (처음 20개): {fire_positions[:20]}")

    # Agent 5의 위치에서 visible 범위 확인
    agent_pos = tuple(agent_5.pos)

    # full observability이므로 전체 그리드
    print(f"\nAgent 5 위치: {agent_pos}")
    print(f"전체 그리드 범위: x=0~{grid.width-1}, y=0~{grid.height-1}")

    # 가까운 활화목 찾기 (heuristic_nearest_fire.py 로직 재현)
    agent_pos_tuple = tuple(agent_5.pos)
    fire_distances = []

    for x in range(0, grid.width):
        for y in range(0, grid.height):
            cell = grid.get(x, y)
            # 정확히 heuristic_nearest_fire.py 코드 사용
            if cell and hasattr(cell, 'state') and cell.state == 1:
                dist = abs(x - agent_pos_tuple[0]) + abs(y - agent_pos_tuple[1])
                fire_distances.append((dist, (x, y), cell, cell.state))

    print(f"\nAgent 5에서 찾은 활화목 개수: {len(fire_distances)}개")
    if fire_distances:
        fire_distances.sort()
        print(f"가장 가까운 활화목:")
        for i in range(min(5, len(fire_distances))):
            dist, pos, cell, state = fire_distances[i]
            print(f"  거리={dist}, 위치={pos}, state={state}, cell={cell}")
    else:
        print("활화목이 없음!")

    # 그리드의 전체 상태 확인
    print(f"\n그리드 전체 상태 샘플 (5x5 영역):")
    for y in range(8, 13):
        row = []
        for x in range(8, 13):
            cell = grid.get(x, y)
            if cell is None:
                row.append(".")
            elif hasattr(cell, 'state'):
                if cell.state == 0:
                    row.append("H")  # Healthy tree
                elif cell.state == 1:
                    row.append("F")  # Fire
                elif cell.state == 2:
                    row.append("B")  # Burnt
                else:
                    row.append("?")
            else:
                row.append("X")  # Agent or other
        print(f"  y={y}: {' '.join(row)}")


if __name__ == "__main__":
    run_tree_debug()
