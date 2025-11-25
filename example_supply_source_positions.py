#!/usr/bin/env python
"""
급수원 위치 설정 예제 스크립트

이 스크립트는 각 에이전트의 급수원 위치를 다양한 방식으로 설정하는 방법을 보여줍니다.
"""

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from wildfire_environment.envs.wildfire import WildfireEnv
from wildfire_environment.core.agent import AgentState


def example_1_default_home_positions():
    """예제 1: 기본 동작 (시작 위치 = 급수원 위치)"""
    print("\n" + "=" * 80)
    print("예제 1: 기본 동작 (시작 위치 = 급수원 위치)")
    print("=" * 80)

    env = WildfireEnv(
        size=17,
        num_agents=2,
        agent_start_positions=((1, 1), (15, 15)),
    )

    obs, info = env.reset(seed=42)

    print("\n에이전트 급수원 위치:")
    for agent in env.agents:
        print(f"  Agent {agent.index}: start_pos={(agent.pos[0], agent.pos[1])}, home_pos={agent.home_pos}")

    # 100 스텝 실행 후 상태 확인
    for step in range(100):
        actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(2)}
        obs, reward, terminated, truncated, info = env.step(actions)

    print("\n100 스텝 후:")
    for agent in env.agents:
        state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[agent.state]
        print(f"  Agent {agent.index}: pos={agent.pos}, home={agent.home_pos}, state={state_name}")


def example_2_central_supply_station():
    """예제 2: 중앙 급수원 (모든 에이전트가 같은 위치)"""
    print("\n" + "=" * 80)
    print("예제 2: 중앙 급수원 (모든 에이전트가 같은 위치로 귀환)")
    print("=" * 80)

    env = WildfireEnv(
        size=17,
        num_agents=2,
        agent_start_positions=((1, 1), (15, 15)),
    )

    obs, info = env.reset(seed=42)

    # 급수원 위치를 중앙으로 변경
    supply_station = (8, 8)
    for agent in env.agents:
        agent.home_pos = supply_station

    print(f"\n설정된 급수원 위치: {supply_station}")
    print("에이전트 급수원 위치:")
    for agent in env.agents:
        print(f"  Agent {agent.index}: start_pos=(이전), home_pos={agent.home_pos}")

    # 150 스텝 실행 (RETURNING 상태 유도)
    for step in range(150):
        actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(2)}
        obs, reward, terminated, truncated, info = env.step(actions)

    print("\n150 스텝 후 (RETURNING 상태 도입):")
    for agent in env.agents:
        state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[agent.state]
        print(f"  Agent {agent.index}: pos={agent.pos}, home={agent.home_pos}, state={state_name}")


def example_3_different_supply_stations():
    """예제 3: 에이전트별 다른 급수원 위치"""
    print("\n" + "=" * 80)
    print("예제 3: 에이전트별 다른 급수원 위치")
    print("=" * 80)

    env = WildfireEnv(
        size=17,
        num_agents=4,
        agent_start_positions=((1, 1), (15, 1), (1, 15), (15, 15)),
        agent_colors=("red", "blue", "yellow", "green"),
    )

    obs, info = env.reset(seed=42)

    # 각 사분면에 급수원 배치
    supply_stations = [
        (2, 2),       # Agent 0: 좌상단
        (14, 2),      # Agent 1: 우상단
        (2, 14),      # Agent 2: 좌하단
        (14, 14),     # Agent 3: 우하단
    ]

    for i, agent in enumerate(env.agents):
        agent.home_pos = supply_stations[i]

    print("\n에이전트별 급수원 위치:")
    for agent in env.agents:
        print(f"  Agent {agent.index}: start_pos=(이전), home_pos={agent.home_pos}")

    # 200 스텝 실행
    for step in range(200):
        actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(4)}
        obs, reward, terminated, truncated, info = env.step(actions)

    print("\n200 스텝 후:")
    for agent in env.agents:
        state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[agent.state]
        print(f"  Agent {agent.index}: pos={agent.pos}, home={agent.home_pos}, state={state_name}, water={agent.water_remaining:.1f}")


def example_4_dynamic_position_change():
    """예제 4: 동적 급수원 위치 변경"""
    print("\n" + "=" * 80)
    print("예제 4: 동적 급수원 위치 변경 (중간에 위치 변경)")
    print("=" * 80)

    env = WildfireEnv(
        size=17,
        num_agents=2,
        agent_start_positions=((1, 1), (15, 15)),
    )

    obs, info = env.reset(seed=42)

    print("\n초기 급수원 위치:")
    for agent in env.agents:
        print(f"  Agent {agent.index}: home_pos={agent.home_pos}")

    # 50 스텝 실행
    print("\n처음 50 스텝 실행...")
    for step in range(50):
        actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(2)}
        obs, reward, terminated, truncated, info = env.step(actions)

    # 급수원 위치 변경
    print("\n급수원 위치 변경!")
    new_positions = [(8, 1), (8, 15)]
    for i, agent in enumerate(env.agents):
        print(f"  Agent {agent.index}: {agent.home_pos} → {new_positions[i]}")
        agent.home_pos = new_positions[i]

    # 50 스텝 더 실행
    print("\n다음 50 스텝 실행 (변경된 위치로 귀환)...")
    for step in range(50):
        actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(2)}
        obs, reward, terminated, truncated, info = env.step(actions)

    print("\n100 스텝 후 최종 상태:")
    for agent in env.agents:
        state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[agent.state]
        print(f"  Agent {agent.index}: pos={agent.pos}, home={agent.home_pos}, state={state_name}")


def example_5_supply_station_tracking():
    """예제 5: 급수원 도달 추적"""
    print("\n" + "=" * 80)
    print("예제 5: 급수원 도달 추적 및 통계")
    print("=" * 80)

    env = WildfireEnv(
        size=22,
        num_agents=3,
        agent_start_positions=((1, 1), (20, 1), (10, 10)),
        agent_colors=("red", "blue", "yellow"),
    )

    # 중앙 급수원 설정
    supply_station = (11, 11)
    for agent in env.agents:
        agent.home_pos = supply_station

    obs, info = env.reset(seed=42)

    print(f"\n급수원 위치: {supply_station}")

    # 추적 변수
    supply_visits = {i: 0 for i in range(3)}
    state_history = {i: [] for i in range(3)}

    print("\n에피소드 실행 중 (500 스텝)...")
    for step in range(500):
        actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(3)}
        obs, reward, terminated, truncated, info = env.step(actions)

        # 상태 추적
        for agent in env.agents:
            state = agent.state
            state_history[agent.index].append(state)

            # 급수원 도달 감지
            if agent.state == AgentState.RECHARGING and agent.pos == agent.home_pos:
                supply_visits[agent.index] += 1

        if terminated or truncated:
            break

    # 통계 출력
    print(f"\n총 실행 스텝: {step + 1}")
    print("\n에이전트별 급수원 도달 횟수:")
    for agent_id in range(3):
        visits = supply_visits[agent_id]
        # RECHARGING 상태 지속 시간 계산
        recharging_steps = sum(1 for s in state_history[agent_id] if s == AgentState.RECHARGING)
        print(f"  Agent {agent_id}: {visits}회 도달, {recharging_steps}스텝 재충전")

    print("\n에이전트별 상태 전이 횟수:")
    for agent_id in range(3):
        history = state_history[agent_id]
        transitions = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
        print(f"  Agent {agent_id}: {transitions}회 상태 전이")


def example_6_practical_setup():
    """예제 6: 실전 설정 (train_marllib_self 환경)"""
    print("\n" + "=" * 80)
    print("예제 6: 실전 설정 (train_marllib_self의 환경)")
    print("=" * 80)

    from train_marllib_self.environment import ENV_CONFIG

    env = WildfireEnv(**ENV_CONFIG)
    obs, info = env.reset(seed=42)

    print(f"\n환경: {ENV_CONFIG['size']}x{ENV_CONFIG['size']} 그리드, {len(env.agents)} 에이전트")

    # 모든 에이전트의 현재 급수원 위치
    print("\n기본 급수원 위치 (시작 위치와 동일):")
    for agent in env.agents:
        agent_type = agent.type
        print(f"  Agent {agent.index} ({agent_type}): pos={agent.pos}, home={agent.home_pos}")

    # 사용자 정의: 모든 에이전트를 그리드 중앙으로
    center = (11, 11)
    print(f"\n중앙 급수원으로 변경 → {center}")
    for agent in env.agents:
        agent.home_pos = center

    print("\n변경 후 급수원 위치:")
    for agent in env.agents:
        print(f"  Agent {agent.index}: home={agent.home_pos}")

    # 100 스텝 실행
    for step in range(100):
        actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(len(env.agents))}
        obs, reward, terminated, truncated, info = env.step(actions)

    print("\n100 스텝 후 에이전트 상태:")
    for agent in env.agents:
        state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[agent.state]
        print(f"  Agent {agent.index}: state={state_name}, water={agent.water_remaining:.1f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("급수원 위치 설정 예제")
    print("=" * 80)

    # 모든 예제 실행
    example_1_default_home_positions()
    example_2_central_supply_station()
    example_3_different_supply_stations()
    example_4_dynamic_position_change()
    example_5_supply_station_tracking()
    example_6_practical_setup()

    print("\n" + "=" * 80)
    print("✓ 모든 예제 완료")
    print("=" * 80)
