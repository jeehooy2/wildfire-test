"""
휴리스틱 정책의 액션 선택 문제 디버깅

에이전트가 RECHARGING 상태에서 왜 움직이지 않는지 확인
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

def select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs, debug=False):
    """
    휴리스틱 정책으로 액션 선택 (디버깅 정보 포함)
    """
    try:
        agent_idx = int(agent_id)
        if agent_idx >= len(env.agents):
            return 0
    except (ValueError, TypeError):
        return 0

    agent = env.agents[agent_idx]
    agent_pos = tuple(agent.pos) if isinstance(agent.pos, np.ndarray) else agent.pos

    # RECHARGING 상태: 제자리 유지
    if hasattr(agent, 'state') and agent.state == AgentState.RECHARGING:
        if debug:
            print(f"    Agent {agent_id}: RECHARGING 상태 → STILL")
        return 0

    # RETURNING 상태: 급수원으로 이동
    if hasattr(agent, 'state') and agent.state == AgentState.RETURNING:
        if hasattr(agent, 'home_pos') and agent.home_pos is not None:
            if debug:
                print(f"    Agent {agent_id}: RETURNING 상태 → Home으로 이동 (pos={agent_pos}, home={agent.home_pos})")
            return 0  # _move_towards 호출 필요 (생략됨)
        else:
            return 0

    # 활화목 찾기 (생략 - 기본값 반환)
    if debug:
        print(f"    Agent {agent_id}: ACTIVE 상태 → 불 찾기")
    return 0


def run_heuristic_debug():
    """휴리스틱 정책 디버깅"""

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    obs_dict, _ = env.reset(seed=42)

    print("=" * 80)
    print("휴리스틱 정책 액션 선택 디버깅")
    print("=" * 80)

    partial_obs = env_config.get('partial_obs', False)
    step = 0
    recharging_steps = {i: 0 for i in range(len(env.agents))}

    while step < 300:
        # 각 에이전트의 액션 선택
        actions = {}

        print(f"\n[Step {step}]")
        for agent_id in obs_dict.keys():
            obs = obs_dict[agent_id]
            agent_idx = int(agent_id)
            agent = env.agents[agent_idx]

            # 현재 상태 출력
            print(f"  Agent {agent_id}: 상태={agent.state.name}, 위치={agent.pos}, "
                  f"활동={agent.active_time_remaining}, 물={agent.water_remaining:.1f}")

            action = select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs, debug=True)
            actions[agent_id] = action

            # RECHARGING 상태 추적
            if agent.state == AgentState.RECHARGING:
                recharging_steps[agent_idx] += 1

        # 환경 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)

        # RECHARGING 상태 종료 감지
        for i, agent in enumerate(env.agents):
            if agent.state != AgentState.RECHARGING and recharging_steps[i] > 0:
                print(f"  → Agent {i} RECHARGING 종료 (소요 {recharging_steps[i]} step)")
                recharging_steps[i] = 0

        step += 1
        done = terminated or truncated
        if done:
            break

        # 처음 50 스텝 이후는 10 스텝마다만 출력
        if step > 50 and step % 10 != 0:
            continue

    print("\n" + "=" * 80)
    print(f"에피소드 종료 (총 {step} step)")
    print("=" * 80)


if __name__ == "__main__":
    run_heuristic_debug()
