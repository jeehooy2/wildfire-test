# 급수원 반환 메커니즘 구현 가이드

## 개요

`wildfire_environment`에 **급수원 반환 및 재충전 메커니즘**이 구현되었습니다. 에이전트는 일정 시간 활동하면 자동으로 급수원(홈베이스)으로 돌아가 물을 재충전합니다.

## 주요 특징

### 1. 에이전트 상태 (3가지)

```python
class AgentState(enum.IntEnum):
    ACTIVE = 0         # 활동 중 (불 진압)
    RETURNING = 1      # 급수원으로 돌아가는 중
    RECHARGING = 2     # 급수원에서 충전 중
```

### 2. 에이전트 속성

각 에이전트는 다음과 같은 속성을 가집니다:

```python
# Phase 3/4: 급수원 반환 및 재충전
self.home_pos = None                        # 급수원 위치
self.state = AgentState.ACTIVE              # 현재 상태

# 활동 시간 제약
self.max_active_time = 150                  # 최대 활동 시간 (스텝)
self.active_time_remaining = 150            # 남은 활동 시간

# 재충전 메커니즘
self.recharge_time = 20                     # 재충전 소요 시간 (스텝)
self.recharge_time_remaining = 0            # 남은 재충전 시간

# 물/소화제 시스템
self.max_water = 100.0                      # 최대 수량
self.water_remaining = 100.0                # 현재 수량
self.water_consumption_rate = 2.0           # 스텝당 소비량

# 목표 위치
self.target_pos = None
```

## 에이전트 타입별 설정

### Helicopter (헬리콥터)
- `max_active_time = 200` 스텝 (오래 활동 가능)
- `recharge_time = 15` 스텝 (빨리 충전)
- 빠른 속도와 높은 효율성으로 오래 활동

### Truck (소방차)
- `max_active_time = 150` 스텝 (표준)
- `recharge_time = 20` 스텝 (표준)
- 균형잡힌 활동과 재충전

### Crew (인력)
- `max_active_time = 100` 스텝 (자주 돌아옴)
- `recharge_time = 25` 스텝 (느린 충전)
- 느린 속도로 자주 재충전 필요

## 상태 전이 로직

### ACTIVE → RETURNING 조건

1. **활동 시간 만료**: `active_time_remaining == 0`
2. **물 부족**: `water_remaining <= 0`

### RETURNING → RECHARGING 조건

에이전트가 `home_pos`에 도착했을 때

### RECHARGING → ACTIVE 조건

`recharge_time_remaining == 0`일 때 (물이 자동으로 `max_water`로 재충전됨)

## Step 함수에서의 동작

### 1. 상태 관리 (Step 시작)

```python
if agent.state == AgentState.ACTIVE:
    # 활동 시간 감소
    active_time_remaining -= 1
    # 활동 시간 만료 → RETURNING
    if active_time_remaining == 0:
        state = RETURNING
        target_pos = home_pos
    # 물 부족 → RETURNING (강제)
    if water_remaining <= 0:
        state = RETURNING
        target_pos = home_pos

elif agent.state == AgentState.RETURNING:
    # 홈 도착 확인
    if pos == home_pos:
        state = RECHARGING
        recharge_time_remaining = recharge_time

elif agent.state == AgentState.RECHARGING:
    # 재충전 시간 감소
    recharge_time_remaining -= 1
    # 재충전 완료 → ACTIVE
    if recharge_time_remaining == 0:
        state = ACTIVE
        active_time_remaining = max_active_time
        water_remaining = max_water  # 물 재충전
```

### 2. 에이전트 이동

**RETURNING 상태**: 자동으로 홈 방향으로 이동 (8방향)
```python
if state == AgentState.RETURNING:
    # 홈 방향으로 한 칸 이동
    # 사용자 action 무시
```

**RECHARGING 상태**: 이동 금지
```python
elif state == AgentState.RECHARGING:
    # 제자리에서 대기
```

**ACTIVE 상태**: 사용자 action 대로 이동

### 3. 물 소비

```python
for agent in agents:
    tree_at_pos = helper_grid.get(*agent.pos)
    # 불타는 나무 위에서 물 소비
    if tree_at_pos.state == 1 and agent.water_remaining > 0:
        agent.water_remaining -= agent.water_consumption_rate
        suppression_active[pos] = True
```

## 사용 예제

### 환경 생성

```python
from wildfire_environment.envs.wildfire import WildfireEnv

env = WildfireEnv(
    size=17,
    num_agents=2,
    num_helicopters=1,  # 헬리콥터 추가
    num_trucks=1,       # 소방차 추가
    initial_fire_size=3,
    max_steps=500,
    agent_start_positions=((1, 1), (15, 15)),
)

obs, info = env.reset(seed=42)
```

### 에이전트 상태 모니터링

```python
for agent in env.agents:
    state_name = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}[agent.state]
    print(f"Agent {agent.index}:")
    print(f"  State: {state_name}")
    print(f"  Position: {agent.pos}")
    print(f"  Home: {agent.home_pos}")
    print(f"  Active Time: {agent.active_time_remaining}/{agent.max_active_time}")
    print(f"  Water: {agent.water_remaining}/{agent.max_water}")
```

## 타임라인 예시

```
Step 0~149:   Agent ACTIVE (불 진압)
              - active_time: 150 → 0
              - water 소비 (불 위에 있을 때)

Step 150:     active_time 만료 → RETURNING으로 전환
              - target_pos = home_pos

Step 151~157: Agent RETURNING (홈으로 귀환)
              - 자동으로 홈 방향으로 이동 (7칸 거리)

Step 158:     홈 도착 → RECHARGING으로 전환
              - recharge_time = 20

Step 159~178: Agent RECHARGING (재충전)
              - 제자리에서 대기
              - 물 회복 (step 179에 100.0으로 설정됨)

Step 179:     재충전 완료 → ACTIVE로 전환
              - active_time = 150 (재설정)
              - water = 100.0 (재충전)
```

## 주요 메서드 수정 사항

### `agent.py` - Agent.__init__()
- AgentState enum 추가
- 급수원 관련 속성 추가
- 각 에이전트 타입(Helicopter, Truck, Crew)에 맞춘 시간 설정

### `wildfire.py` - _gen_grid()
- `a.home_pos = agent_start_pos[i]` 추가

### `wildfire.py` - reset()
- 에이전트 상태 초기화 추가

### `wildfire.py` - step()
1. 상태 관리 로직 추가 (처음 부분)
2. RETURNING/RECHARGING 상태에서의 이동 처리
3. 물 소비 및 suppression_active 플래그 설정

## 테스트

`test_supply_source_debug.py`를 실행하여 동작을 확인할 수 있습니다:

```bash
python test_supply_source_debug.py
```

예상 결과:
- 에이전트가 150 스텝 동안 활동
- 불 위에 있을 때 물이 2씩 감소
- 150 스텝 후 RETURNING 상태로 전환
- 홈에 도착하면 RECHARGING 상태로 전환
- 20 스텝 후 물이 100으로 재충전되고 ACTIVE로 복귀

## 커스터마이징

에이전트 타입별로 활동 시간과 재충전 시간을 커스터마이징할 수 있습니다:

```python
# agent.py의 에이전트 클래스에서
self.max_active_time = 200      # 원하는 값으로 설정
self.recharge_time = 15         # 원하는 값으로 설정
```

또는 environment 생성 후:

```python
env.agents[0].max_active_time = 250
env.agents[0].recharge_time = 25
```

## 주의사항

1. **홈 위치 설정**: `agent_start_positions`가 홈으로 자동 설정됨
2. **상태 강제 변경**: 물이 0이 되면 즉시 RETURNING 상태로 전환
3. **대각선 이동**: RETURNING 중에는 8방향 대각선 이동 가능
4. **액션 무시**: RETURNING/RECHARGING 중에는 사용자 액션이 무시됨

---

**마지막 수정**: 2025-11-23
**작성자**: Claude Code
**상태**: ✅ 완료
