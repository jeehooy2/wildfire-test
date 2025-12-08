"""
급수원에서 멈춰있는 에이전트 상태 디버깅

각 스텝마다 에이전트의 상태, 위치, 상태 전환 조건을 상세히 추적
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

def run_debug_stuck():
    """에이전트가 멈춰있는 상태 추적"""

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    obs_dict, _ = env.reset(seed=42)

    print("=" * 100)
    print("급수원에서 멈춰있는 에이전트 추적")
    print("=" * 100)

    step = 0
    stuck_agents = {}  # {agent_id: {'start_step': , 'start_pos': , ...}}

    while step < 300:
        # 각 에이전트의 상태 확인
        print(f"\n[Step {step}]")

        for i, agent in enumerate(env.agents):
            # 홈에 있는지 확인
            at_home = np.array_equal(agent.pos, agent.home_pos)
            state_name = agent.state.name

            print(f"  Agent {i}: pos={agent.pos}, home={agent.home_pos}, "
                  f"at_home={at_home}, state={state_name}, "
                  f"active_time={agent.active_time_remaining}, water={agent.water_remaining:.1f}, "
                  f"recharge_time={agent.recharge_time_remaining}")

            # 급수원에 있고 RECHARGING이 아닌 상태 감지 (비정상)
            if at_home and state_name != "RECHARGING":
                if i not in stuck_agents:
                    stuck_agents[i] = {
                        'start_step': step,
                        'start_pos': tuple(agent.pos),
                        'states': []
                    }

                # 상태 기록
                stuck_agents[i]['states'].append({
                    'step': step,
                    'state': state_name,
                    'active_time': agent.active_time_remaining,
                    'water': agent.water_remaining
                })

                if state_name == "ACTIVE" and agent.active_time_remaining == 0:
                    print(f"    ⚠️  Agent {i}가 홈에서 ACTIVE 상태인데 활동 시간 = 0 (상태 전환 안 됨?)")

        # 액션 선택 (모두 STILL)
        actions = {str(i): 0 for i in range(len(env.agents))}

        # 환경 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)

        step += 1
        done = terminated or truncated
        if done:
            break

        # 너무 많이 출력되지 않도록 제한
        if step > 100 and step % 10 != 0:
            continue

    print("\n" + "=" * 100)
    print("멈춰있는 에이전트 요약")
    print("=" * 100)

    for agent_id, info in stuck_agents.items():
        if len(info['states']) > 10:  # 10 스텝 이상 멈춰있음
            print(f"\n❌ Agent {agent_id}: 급수원에서 {len(info['states'])} 스텝간 멈춰있음 (Step {info['start_step']}~)")
            # 처음 3개와 마지막 3개 상태 출력
            print("  초기 상태들:")
            for s in info['states'][:3]:
                print(f"    Step {s['step']}: {s['state']}, active_time={s['active_time']}, water={s['water']:.1f}")
            if len(info['states']) > 6:
                print("  ...")
            print("  최근 상태들:")
            for s in info['states'][-3:]:
                print(f"    Step {s['step']}: {s['state']}, active_time={s['active_time']}, water={s['water']:.1f}")


if __name__ == "__main__":
    run_debug_stuck()
