"""
에이전트 이동 문제 디버깅 스크립트

에이전트가 급수원에 도착한 후 왜 움직이지 않는지 확인
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
from wildfire_environment.core.agent import AgentState

def run_debug_episode():
    """에이전트 상태를 추적하면서 에피소드 실행"""

    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env = WildfireEnv(**env_config)

    obs_dict, _ = env.reset(seed=42)

    print("=" * 80)
    print("초기 에이전트 상태")
    print("=" * 80)
    for i, agent in enumerate(env.agents):
        print(f"Agent {i}:")
        print(f"  위치: {agent.pos}")
        print(f"  홈 위치: {agent.home_pos}")
        print(f"  상태: {agent.state.name}")
        print(f"  활동 시간: {agent.active_time_remaining}/{agent.max_active_time}")
        print(f"  물 양: {agent.water_remaining}/{agent.max_water}")
        print()

    # 에피소드 실행
    step_count = 0
    recharging_agents = {i: False for i in range(len(env.agents))}
    recharging_start_step = {}

    print("=" * 80)
    print("에피소드 실행 중...")
    print("=" * 80)

    while step_count < 300:
        # 각 에이전트의 액션 선택 (모두 STILL)
        actions = {str(i): 0 for i in range(len(env.agents))}

        # 환경 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)

        # 각 에이전트 상태 확인
        agent_status = []
        for i, agent in enumerate(env.agents):
            agent_status.append({
                'idx': i,
                'pos': tuple(agent.pos),
                'home': agent.home_pos,
                'state': agent.state.name,
                'active_time': agent.active_time_remaining,
                'water': agent.water_remaining,
                'recharge_time': agent.recharge_time_remaining if hasattr(agent, 'recharge_time_remaining') else 0
            })

            # RECHARGING 상태 진입 시점 추적
            if agent.state == AgentState.RECHARGING:
                if not recharging_agents[i]:
                    recharging_agents[i] = True
                    recharging_start_step[i] = step_count
                    print(f"\n⚠️  [Step {step_count}] Agent {i} RECHARGING 상태 진입")
                    print(f"    위치: {agent.pos}, 홈: {agent.home_pos}")
                    print(f"    재충전 시간: {agent.recharge_time_remaining}/{agent.recharge_time}")

        # RECHARGING 상태 종료 추적
        for i, agent in enumerate(env.agents):
            if agent.state != AgentState.RECHARGING and recharging_agents[i]:
                recharging_duration = step_count - recharging_start_step[i]
                print(f"\n✓ [Step {step_count}] Agent {i} RECHARGING 상태 종료 (소요 {recharging_duration} step)")
                print(f"    상태 전환: {agent.state.name}")
                print(f"    활동 시간 리셋: {agent.active_time_remaining}/{agent.max_active_time}")
                print(f"    물 리셋: {agent.water_remaining}/{agent.max_water}")
                recharging_agents[i] = False

        step_count += 1
        done = terminated or truncated
        if done:
            break

    print("\n" + "=" * 80)
    print(f"에피소드 종료 (총 {step_count} step)")
    print("=" * 80)

    print("\n최종 에이전트 상태:")
    for agent_info in agent_status:
        print(f"Agent {agent_info['idx']}: 위치={agent_info['pos']}, "
              f"상태={agent_info['state']}, "
              f"활동={agent_info['active_time']}, "
              f"물={agent_info['water']:.1f}")

if __name__ == "__main__":
    run_debug_episode()
