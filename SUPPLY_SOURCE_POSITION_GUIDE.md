# 급수원 위치 설정 가이드

## 개요

각 에이전트가 충전하기 위해 돌아가는 급수원(홈베이스) 위치를 설정하는 방법을 설명합니다.

## 현재 구현

### 기본 동작 (start_position = home_position)

```python
# wildfire_environment/envs/wildfire.py:507
a.home_pos = agent_start_pos[i]
```

**특징**:
- 에이전트의 시작 위치 = 급수원 위치
- 각 에이전트마다 다른 급수원 가능
- 가장 간단한 방식

**예시**:
```python
agent_start_positions = ((1, 1), (15, 15), (8, 8))
# Agent 0: home_pos = (1, 1)
# Agent 1: home_pos = (15, 15)
# Agent 2: home_pos = (8, 8)
```

---

## 🛠️ 급수원 위치 수정 방법

### 방법 1: 환경 생성 후 직접 수정 (권장)

**장점**: 가장 간단하고 유연함

```python
from wildfire_environment.envs.wildfire import WildfireEnv

# 환경 생성
env = WildfireEnv(
    size=17,
    num_agents=2,
    agent_start_positions=((1, 1), (15, 15)),
)

# 급수원 위치 변경
env.agents[0].home_pos = (1, 1)      # 에이전트 0의 급수원: 좌상단
env.agents[1].home_pos = (15, 15)    # 에이전트 1의 급수원: 우하단

# 또는 중앙에 통합 급수원
env.agents[0].hㅐ
# 환경 초기화
obs, info = env.reset()
```

**사용 사례**:
```python
# 모든 에이전트가 동일한 급수원으로 돌아가기
supply_station = (8, 8)
for agent in env.agents:
    agent.home_pos = supply_station
```

### 방법 2: 초기화 후 수정

```python
from wildfire_environment.envs.wildfire import WildfireEnv

env = WildfireEnv(
    size=17,
    num_agents=3,
    agent_start_positions=((1, 1), (8, 8), (15, 15)),
)

# Reset 후 급수원 수정
obs, info = env.reset(seed=42)

# 에이전트별 다른 급수원 설정
env.agents[0].home_pos = (1, 1)       # Agent 0: 좌상단
env.agents[1].home_pos = (8, 8)       # Agent 1: 중앙
env.agents[2].home_pos = (15, 15)     # Agent 2: 우하단

# 재설정 후 다시 reset (선택사항)
obs, info = env.reset(seed=42)
```

### 방법 3: 환경 설정 수정 (코드 변경)

wildfire.py의 `_gen_grid()` 메서드를 수정하여 급수원 위치를 별도로 설정:

**현재 코드**:
```python
# wildfire_environment/envs/wildfire.py:507
a.home_pos = agent_start_pos[i]
```

**수정 예시 1 - 급수원 위치 파라미터 추가**:

```python
# __init__에 파라미터 추가
def __init__(
    self,
    ...
    agent_start_positions=((1, 1), (15, 15)),
    agent_supply_positions=None,  # 새로 추가
    ...
):
    ...
    self.agent_supply_positions = agent_supply_positions or agent_start_positions
    ...

# _gen_grid에서 수정
for i, a in enumerate(self.agents):
    self.place_agent(a, pos=agent_start_pos[i])
    # 급수원 위치를 별도로 설정
    a.home_pos = self.agent_supply_positions[i]
```

**사용 방법**:
```python
env = WildfireEnv(
    size=17,
    num_agents=2,
    agent_start_positions=((1, 1), (15, 15)),
    agent_supply_positions=((8, 1), (8, 15)),  # 다른 급수원 위치
)
```

**수정 예시 2 - 통합 급수원 위치**:

```python
# __init__에 파라미터 추가
def __init__(
    self,
    ...
    central_supply_position=None,  # 새로 추가
    ...
):
    ...
    self.central_supply_position = central_supply_position
    ...

# _gen_grid에서 수정
for i, a in enumerate(self.agents):
    self.place_agent(a, pos=agent_start_pos[i])
    # 통합 급수원 또는 개별 급수원
    if self.central_supply_position is not None:
        a.home_pos = self.central_supply_position
    else:
        a.home_pos = agent_start_pos[i]
```

**사용 방법**:
```python
# 모든 에이전트가 중앙 급수원 사용
env = WildfireEnv(
    size=17,
    num_agents=2,
    agent_start_positions=((1, 1), (15, 15)),
    central_supply_position=(8, 8),  # 중앙의 통합 급수원
)
```

---

## 💡 실제 예제

### 예제 1: 기본 사용 (현재 시스템)

```python
from wildfire_environment.envs.wildfire import WildfireEnv

# 각 에이전트가 자신의 시작 위치를 급수원으로 사용
env = WildfireEnv(
    size=17,
    num_agents=2,
    agent_start_positions=((1, 1), (15, 15)),
)

obs, info = env.reset()

# Agent 0: home_pos = (1, 1)
# Agent 1: home_pos = (15, 15)
```

### 예제 2: 중앙 급수원

```python
from wildfire_environment.envs.wildfire import WildfireEnv

env = WildfireEnv(
    size=17,
    num_agents=2,
    agent_start_positions=((1, 1), (15, 15)),
)

obs, info = env.reset()

# 모든 에이전트가 중앙으로 돌아가기
supply_station = (8, 8)
for agent in env.agents:
    agent.home_pos = supply_station

# 이제 에이전트들이 (8, 8)로 돌아갈 것
```

### 예제 3: 각 구역별 급수원

```python
from wildfire_environment.envs.wildfire import WildfireEnv

env = WildfireEnv(
    size=17,
    num_agents=4,
    agent_start_positions=((1, 1), (15, 1), (1, 15), (15, 15)),
)

obs, info = env.reset()

# 각 사분면에 급수원 배치
supply_stations = [
    (2, 2),       # 좌상단
    (14, 2),      # 우상단
    (2, 14),      # 좌하단
    (14, 14),     # 우하단
]

for i, agent in enumerate(env.agents):
    agent.home_pos = supply_stations[i]
```

### 예제 4: 동적 급수원 위치 변경

```python
from wildfire_environment.envs.wildfire import WildfireEnv

env = WildfireEnv(
    size=17,
    num_agents=2,
    agent_start_positions=((1, 1), (15, 15)),
)

obs, info = env.reset()

# 초기 급수원
print(f"Agent 0 home: {env.agents[0].home_pos}")  # (1, 1)
print(f"Agent 1 home: {env.agents[1].home_pos}")  # (15, 15)

# 급수원 변경
env.agents[0].home_pos = (8, 1)
env.agents[1].home_pos = (8, 15)

print(f"Updated Agent 0 home: {env.agents[0].home_pos}")  # (8, 1)
print(f"Updated Agent 1 home: {env.agents[1].home_pos}")  # (8, 15)

# 이제 에피소드를 계속 실행하면 새 위치로 돌아감
```

### 예제 5: 임의의 급수원 배치

```python
from wildfire_environment.envs.wildfire import WildfireEnv

# 환경 생성
env = WildfireEnv(
    size=22,
    num_agents=6,
    agent_start_positions=(
        (1, 1),     # Agent 0
        (20, 1),    # Agent 1
        (1, 20),    # Agent 2
        (20, 20),   # Agent 3
        (11, 11),   # Agent 4
        (11, 1),    # Agent 5
    ),
)

obs, info = env.reset()

# 임의의 급수원 위치 설정
supply_map = {
    0: (3, 3),      # Agent 0: 좌상단 근처
    1: (19, 3),     # Agent 1: 우상단 근처
    2: (3, 19),     # Agent 2: 좌하단 근처
    3: (19, 19),    # Agent 3: 우하단 근처
    4: (11, 11),    # Agent 4: 중앙
    5: (11, 3),     # Agent 5: 상단 중앙
}

for agent_id, supply_pos in supply_map.items():
    env.agents[agent_id].home_pos = supply_pos

# 시뮬레이션 실행
for step in range(100):
    actions = {f"{i}": env.action_space[f"{i}"].sample() for i in range(6)}
    obs, reward, terminated, truncated, info = env.step(actions)

    if step % 50 == 0:
        for agent in env.agents:
            print(f"Agent {agent.index}: pos={agent.pos}, home={agent.home_pos}")
```

---

## 🔧 주의사항

### 1. 유효한 위치만 사용

```python
# 그리드 내부에만 배치 (벽 피하기)
# 17x17 그리드: (1,1)~(15,15) 유효
env.agents[0].home_pos = (8, 8)   # ✓ OK
env.agents[0].home_pos = (0, 0)   # ✗ 벽의 위치
env.agents[0].home_pos = (20, 20) # ✗ 범위 초과
```

### 2. 다중 에이전트가 같은 급수원 사용 가능

```python
# 여러 에이전트가 같은 급수원으로 귀환
supply_station = (8, 8)
for agent in env.agents:
    agent.home_pos = supply_station  # ✓ OK - 겹칠 수 있음
```

### 3. Reset 후에도 유지

```python
env.reset()  # 급수원 위치는 유지됨
obs, info = env.reset(seed=42)  # 급수원 위치는 유지됨
```

### 4. 동적 변경 가능

```python
# 에피소드 중에도 변경 가능 (다음 귀환부터 적용)
env.agents[0].home_pos = (10, 10)
obs, reward, done, info = env.step(actions)
```

---

## 📊 설정 옵션 비교

| 방법 | 복잡도 | 유연성 | 추천도 |
|------|--------|--------|--------|
| **방법 1: 직접 수정** | ⭐ 낮음 | ⭐⭐⭐⭐⭐ 매우 높음 | ⭐⭐⭐⭐⭐ |
| **방법 2: Reset 후 수정** | ⭐ 낮음 | ⭐⭐⭐⭐ 높음 | ⭐⭐⭐⭐ |
| **방법 3: 코드 수정** | ⭐⭐⭐ 중간 | ⭐⭐⭐⭐⭐ 매우 높음 | ⭐⭐⭐ |

---

## 🎯 권장사항

### 상황별 추천 방법

**1. 빠른 프로토타이핑 → 방법 1**
```python
env = WildfireEnv(...)
obs, info = env.reset()
env.agents[0].home_pos = (8, 1)
env.agents[1].home_pos = (8, 15)
```

**2. 실험/비교 → 방법 1 (함수화)**
```python
def setup_supply_stations(env, config):
    for i, pos in enumerate(config):
        env.agents[i].home_pos = pos

# 다양한 설정 시도
setup_supply_stations(env, [(1, 1), (15, 15)])
setup_supply_stations(env, [(8, 8), (8, 8)])  # 중앙
```

**3. 영구적 변경 → 방법 3**
wildfire.py 수정 후 재사용

---

## 📝 요약

**급수원 위치 수정**:

1. **현재 기본값**: `agent.home_pos = agent_start_pos[i]`
2. **가장 간단한 변경**:
   ```python
   agent.home_pos = (원하는_x, 원하는_y)
   ```
3. **가장 유연한 방법**: 방법 1 (환경 생성 후 직접 수정)

---

**버전**: 1.0
**마지막 업데이트**: 2025-11-23
**상태**: ✅ 완료
