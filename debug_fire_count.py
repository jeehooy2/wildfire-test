"""
그리드의 활화목 개수 추적

에이전트가 멈춰있을 때 실제로 활화목이 있는지 확인
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

def count_fires(env):
    """그리드의 활화목 개수 반환"""
    fire_count = 0
    grid = env.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and hasattr(cell, 'state') and cell.state == 1:
                fire_count += 1
    return fire_count

def run_debug_fires():
    """활화목 개수 추적"""

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    obs_dict, _ = env.reset(seed=42)

    print("=" * 100)
    print("그리드 활화목 추적")
    print("=" * 100)

    step = 0
    agent_5_at_home = False
    agent_5_home_steps = []

    while step < 300:
        # 활화목 개수 세기
        fire_count = count_fires(env)

        # Agent 5 추적
        agent_5 = env.agents[5]
        at_home = np.array_equal(agent_5.pos, agent_5.home_pos)

        if at_home and agent_5.state == AgentState.ACTIVE:
            agent_5_at_home = True
            agent_5_home_steps.append({
                'step': step,
                'fires': fire_count,
                'active_time': agent_5.active_time_remaining,
                'water': agent_5.water_remaining
            })

        # 특정 범위의 스텝만 출력
        if 34 <= step <= 140:
            status = ""
            if at_home:
                status = f"[홈] 상태={agent_5.state.name}"
            print(f"Step {step}: 활화목={fire_count:2d}개, Agent5: pos={agent_5.pos}, "
                  f"home={agent_5.home_pos}, {status}")

        # 액션 선택 (모두 STILL)
        actions = {str(i): 0 for i in range(len(env.agents))}

        # 환경 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)

        step += 1
        done = terminated or truncated
        if done:
            break

    print("\n" + "=" * 100)
    print("Agent 5가 홈에서 ACTIVE 상태였던 스텝들")
    print("=" * 100)
    for info in agent_5_home_steps:
        print(f"Step {info['step']}: 활화목={info['fires']}개, 활동시간={info['active_time']}")


if __name__ == "__main__":
    run_debug_fires()
