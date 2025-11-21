"""
휴리스틱 정책 디버깅 스크립트
에이전트의 액션과 위치 변화를 추적합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from train_marllib_self.heuristic_nearest_fire import select_heuristic_action

# 환경 설정
env_config = {k: v for k, v in ENV_CONFIG.items()}
partial_obs = env_config.get('partial_obs', False)

print("="*80)
print(f"디버깅 설정: partial_obs={partial_obs}")
print(f"그리드 크기: {env_config['size']}x{env_config['size']}")
print(f"에이전트 수: {env_config['num_agents']}")
print(f"시작 위치: {env_config['agent_start_positions']}")
print("="*80)

# 환경 생성
env = WildfireEnv(**env_config)
print(f"\n✓ 환경 생성 완료")

# 초기 상태
obs_dict, _ = env.reset(seed=42)

# 디버그: 환경 내부 상태 확인
print(f"\n환경 내부 상태:")
print(f"  env.trees_on_fire: {env.trees_on_fire}")
print(f"  env.unburnt_trees 길이: {len(env.unburnt_trees)}")
print(f"  env.initial_fire_size: {env.initial_fire_size}")

print(f"\n초기 상태:")
print(f"에이전트 수 (obs_dict): {len(obs_dict)}")
print(f"에이전트 ID: {list(obs_dict.keys())}")

# 에이전트 위치와 불 위치 출력
print(f"\n초기 에이전트 위치:")
for agent_id in obs_dict.keys():
    agent_idx = int(agent_id)
    agent = env.agents[agent_idx]
    print(f"  Agent {agent_id}: {agent.pos}")

print(f"\n초기 불 위치 (grid에서):")
grid = env.grid
fire_positions_grid = []
for x in range(grid.width):
    for y in range(grid.height):
        cell = grid.get(x, y)
        if cell and hasattr(cell, 'state') and cell.state == 1:
            fire_positions_grid.append((x, y))
print(f"  불 위치 (grid): {fire_positions_grid}")

print(f"\n초기 불 위치 (helper_grid에서):")
helper_grid = env.helper_grid
fire_positions_helper = []
for x in range(helper_grid.width):
    for y in range(helper_grid.height):
        cell = helper_grid.get(x, y)
        if cell and hasattr(cell, 'state') and cell.state == 1:
            fire_positions_helper.append((x, y))
print(f"  불 위치 (helper_grid): {fire_positions_helper}")

# 첫 몇 스텝 실행
max_steps = 5
for step in range(max_steps):
    print(f"\n{'='*80}")
    print(f"Step {step + 1}")
    print(f"{'='*80}")

    actions = {}

    for agent_id in obs_dict.keys():
        agent_idx = int(agent_id)
        agent = env.agents[agent_idx]
        current_pos = agent.pos

        obs = obs_dict[agent_id]
        action = select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs)
        actions[agent_id] = action

        print(f"\nAgent {agent_id}:")
        print(f"  현재 위치: {current_pos}")
        print(f"  선택된 액션: {action}")

        # 액션 이름 매핑
        action_names = {
            0: "STILL",
            1: "NORTH",
            2: "NORTH_EAST",
            3: "EAST",
            4: "SOUTH_EAST",
            5: "SOUTH",
            6: "SOUTH_WEST",
            7: "WEST",
            8: "NORTH_WEST"
        }
        print(f"  액션 이름: {action_names.get(action, 'UNKNOWN')}")

    # 환경 스텝 실행
    obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)

    print(f"\n스텝 후 에이전트 위치:")
    for agent_id in obs_dict.keys():
        agent_idx = int(agent_id)
        agent = env.agents[agent_idx]
        print(f"  Agent {agent_id}: {agent.pos}")

    print(f"\n리워드:")
    for agent_id in obs_dict.keys():
        print(f"  Agent {agent_id}: {reward_dict[agent_id]:.2f}")

    done = terminated or truncated
    if done:
        print("\n에피소드 종료!")
        break

print("\n" + "="*80)
print("디버깅 완료")
print("="*80)
