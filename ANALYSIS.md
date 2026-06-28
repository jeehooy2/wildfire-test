# Wildfire Environment 코드 분석 기술서

## 목차
1. [개요](#개요)
2. [코드베이스 구조](#코드베이스-구조)
3. [산불 확산 로직](#산불-확산-로직)
4. [에이전트 배치 로직](#에이전트-배치-로직)
5. [진화 효율 (소방 효율)](#진화-효율-소방-효율)
6. [보상 시스템](#보상-시스템)
7. [관찰 공간 및 상태 표현](#관찰-공간-및-상태-표현)
8. [주요 파라미터](#주요-파라미터)

---

## 개요

Wildfire Environment는 무인 항공기(UAV)가 확산되는 산불을 진압하는 시뮬레이션 환경입니다. 이 환경은 OpenAI Gym 인터페이스를 따르는 멀티 에이전트 강화학습 환경으로, 협력적 또는 이기적 에이전트 행동을 지원합니다.

### 주요 특징
- 멀티 에이전트 환경 (2개 이상의 UAV)
- 확률적 산불 확산 모델
- 에이전트의 소방 행동에 따른 동적 진화
- 협력 및 이기적 보상 구조 지원
- 완전/부분 관찰 가능성 지원

---

## 코드베이스 구조

```
wildfire_environment/
├── core/                    # 핵심 구성 요소
│   ├── agent.py            # 에이전트 클래스 및 액션 정의
│   ├── grid.py             # 그리드 표현 및 렌더링
│   ├── object.py           # 세계 오브젝트 (Tree, Wall 등)
│   ├── world.py            # 세계 정의 (WildfireWorld)
│   └── constants.py        # 상수 정의 (색상, 상태 등)
├── envs/
│   └── wildfire.py         # WildfireEnv 메인 환경
├── multigrid.py            # MultiGridEnv 기본 클래스
├── policy/                 # 정책 관련
│   └── base.py
└── utils/                  # 유틸리티
    ├── rendering.py        # 렌더링 함수
    ├── window.py           # 시각화 윈도우
    └── misc.py             # 기타 유틸리티
```

### 핵심 파일 설명

#### 1. `envs/wildfire.py` (985줄)
- **역할**: 산불 환경의 메인 클래스
- **주요 클래스**: `WildfireEnv`
- **핵심 메서드**:
  - `_gen_grid()`: 초기 그리드 생성 및 나무/에이전트 배치
  - `step()`: 환경 시뮬레이션 (에이전트 이동 + 산불 확산)
  - `neighbors_on_fire()`: 이웃한 불타는 나무 수 계산
  - `_get_obs()`: 에이전트 관찰 벡터 생성

#### 2. `core/agent.py` (500줄)
- **역할**: 에이전트 정의 및 액션
- **주요 클래스**:
  - `Agent`: 기본 에이전트
  - `WildfireActions`: 산불 환경 액션 (STILL, NORTH, EAST, SOUTH, WEST)

#### 3. `core/object.py` (414줄)
- **역할**: 그리드 오브젝트 정의
- **주요 클래스**:
  - `Tree`: 나무 오브젝트 (상태: healthy, on fire, burnt)
  - `Wall`: 벽 오브젝트

#### 4. `core/grid.py` (556줄)
- **역할**: 그리드 표현 및 렌더링
- **주요 메서드**:
  - `render()`: 그리드 시각화
  - `encode()`: 그리드 인코딩

---

## 산불 확산 로직

### 확산 메커니즘

산불 확산은 `envs/wildfire.py`의 `step()` 메서드 (786-839 라인)에서 구현됩니다.

#### 1. 나무 상태 전이

나무는 3가지 상태를 가집니다:
- **0 (healthy)**: 건강한 나무 (녹색)
- **1 (on fire)**: 불타는 나무 (주황색)
- **2 (burnt)**: 타버린 나무 (갈색)

#### 2. 상태 전이 확률

**건강한 나무 → 불타는 나무**
```python
# envs/wildfire.py:799-801
if np.random.rand() < 1 - (1 - self.alpha) ** self.neighbors_on_fire(pos):
    # 나무에 불이 붙음
```

수식:
```
P(healthy → on fire) = 1 - (1 - α)^n
```
- `α` (alpha): 산불 확산 파라미터 (기본값: 0.05)
- `n`: 4방향 이웃 중 불타는 나무의 개수 (0~4)

**확률 테이블:**
| 불타는 이웃 수 (n) | α=0.05 | α=0.1 | α=0.2 |
|------------------|---------|--------|--------|
| 0 | 0% | 0% | 0% |
| 1 | 5% | 10% | 20% |
| 2 | 9.75% | 19% | 36% |
| 3 | 14.26% | 27.1% | 48.8% |
| 4 | 18.55% | 34.39% | 59.04% |

**불타는 나무 → 타버린 나무**
```python
# envs/wildfire.py:811
if np.random.rand() < 1 - self.beta + c.agent_above * self.delta_beta:
    # 나무가 타버림
```

수식:
```
P(on fire → burnt) = 1 - β + δ_β × agent_above
```
- `β` (beta): 불이 계속 타는 확률 (기본값: 0.99)
- `δ_β` (delta_beta): 에이전트 진화 효과 (기본값: 0)
- `agent_above`: 에이전트가 나무 위에 있으면 1, 아니면 0

#### 3. 이웃 탐지 로직

```python
# envs/wildfire.py:660-689
def neighbors_on_fire(self, tree_pos) -> int:
    """4방향 이웃 중 불타는 나무 수 계산"""
    num = 0
    relative_pos = [
        np.array([1, 0]),   # 동
        np.array([-1, 0]),  # 서
        np.array([0, 1]),   # 남
        np.array([0, -1]),  # 북
    ]
    for r in relative_pos:
        neighbor_pos = tree_pos + r
        if neighbor_pos[0] >= 0 and neighbor_pos[0] < self.helper_grid.width:
            if neighbor_pos[1] >= 0 and neighbor_pos[1] < self.helper_grid.height:
                o = self.helper_grid.get(*neighbor_pos)
                if o is not None and o.type == "tree":
                    if o.state == 1:  # on fire
                        num += 1
    return num
```

**특징:**
- **4방향 연결성**: 대각선 이웃은 고려하지 않음
- **경계 처리**: 그리드 경계 밖은 이웃으로 계산하지 않음
- **동시 업데이트**: 모든 나무의 상태 전이를 먼저 계산한 후 일괄 업데이트하여 순서 의존성 제거

#### 4. 업데이트 순서

```python
# envs/wildfire.py:787-839
# 1단계: 전이할 나무 식별
trees_to_fire_state = []    # healthy → on fire
trees_to_burnt_state = []   # on fire → burnt

# 2단계: 모든 나무에 대해 전이 확률 계산
for c in self.unburnt_trees:
    # 확률 계산 및 전이 결정
    ...

# 3단계: 일괄 업데이트 (순서 의존성 제거)
for c in trees_to_fire_state:
    c.state = 1
    c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]

for c in trees_to_burnt_state:
    c.state = 2
    c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
```

이 접근 방식은 **동기적(synchronous) 업데이트**를 보장하여 업데이트 순서에 따른 편향을 방지합니다.

---

## 에이전트 배치 로직

### 초기 배치

#### 1. 그리드 생성 (`_gen_grid()`)

```python
# envs/wildfire.py:236-382
def _gen_grid(self, width, height, state=None):
    """그리드 및 초기 상태 생성"""

    # 1. 빈 그리드 생성
    self.grid = Grid(width, height, self.world)

    # 2. 벽 생성 (그리드 경계)
    self.grid.horz_wall(0, 0)                # 상단
    self.grid.horz_wall(0, height - 1)       # 하단
    self.grid.vert_wall(0, 0)                # 좌측
    self.grid.vert_wall(width - 1, 0)        # 우측
```

#### 2. 초기 화재 위치 결정

**상태가 지정되지 않은 경우 (무작위 생성):**

```python
# envs/wildfire.py:279-316
if self.initial_fire_size % 2 == 0:
    # 짝수 크기: 좌상단 모서리 무작위 선택
    top_left_corner = (
        random.randint(1, self.grid_size_without_walls - self.initial_fire_size),
        random.randint(1, self.grid_size_without_walls - self.initial_fire_size),
    )
    initial_fire = get_initial_fire_coordinates(
        *top_left_corner, self.grid_size, self.initial_fire_size
    )
else:
    # 홀수 크기: 중심 무작위 선택
    fire_square_center = (
        random.randint(
            1 + (self.initial_fire_size - 1) / 2,
            self.grid_size_without_walls - (self.initial_fire_size - 1) / 2
        ),
        ...
    )
    initial_fire = get_initial_fire_coordinates(
        *fire_square_center, self.grid_size, self.initial_fire_size
    )
```

**특징:**
- 화재는 정사각형 영역으로 시작
- 벽을 제외한 유효 영역에서만 생성 가능
- `initial_fire_size` × `initial_fire_size` 크기

#### 3. 에이전트 배치

```python
# envs/wildfire.py:378-381
for i, a in enumerate(self.agents):
    self.place_agent(a, pos=agent_start_pos[i])
    self.helper_grid.get(*agent_start_pos[i]).agent_above = True
```

**에이전트 위치 파라미터:**
- `agent_start_positions`: 초기 위치 튜플 (기본값: `((1,1), (15,15))`)
- 각 에이전트는 고정된 시작 위치를 가짐
- 위치는 (x, y) 좌표로 지정

#### 4. 나무 배치

```python
# envs/wildfire.py:349-368
num_healthy_trees = self.grid_size_without_walls**2 - len(initial_fire)

for _ in range(num_healthy_trees):
    tree_obj = Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"])
    self.place_obj(tree_obj)  # 무작위 빈 공간에 배치
```

**배치 전략:**
- 전체 나무 수 = (grid_size - 2)² (벽 제외)
- 초기 화재 영역을 제외한 모든 셀에 건강한 나무 배치
- `place_obj()`는 빈 공간을 무작위로 찾아 배치

### 동적 이동 (런타임)

#### 이동 메커니즘

```python
# envs/wildfire.py:637-658
def move_agent(self, i, next_pos):
    """에이전트를 새 위치로 이동"""

    # 1. 새 위치에 에이전트 추가
    self.grid.set(*next_pos, self.agents[i])

    # 2. 이전 위치에서 에이전트 제거, 나무 복원
    tree = self.helper_grid.get(*self.agents[i].pos)
    tree.agent_above = False
    self.grid.set(*self.agents[i].pos, tree)

    # 3. 속성 업데이트
    next_tree = self.helper_grid.get(*next_pos)
    next_tree.agent_above = True
    self.agents[i].pos = next_pos
```

#### 이동 규칙

```python
# envs/wildfire.py:760-784
order = np.random.permutation(len(actions))  # 무작위 순서
for i in order:
    if actions[i] == self.actions.STILL:
        continue
    if actions[i] == self.actions.NORTH:
        next_pos = self.agents[i].north_pos()
        next_cell = self.grid.get(*next_pos)
        if next_cell is None or next_cell.can_overlap():
            self.move_agent(i, next_pos)
    # SOUTH, EAST, WEST 동일
```

**특징:**
- **무작위 순서 실행**: 에이전트 간 순서 편향 방지
- **충돌 검사**: 벽과는 충돌 불가, 나무는 겹칠 수 있음
- **5가지 액션**: STILL (0), NORTH (1), EAST (2), SOUTH (3), WEST (4)

---

## 진화 효율 (소방 효율)

### 소방 메커니즘

#### 1. 기본 원리

에이전트가 불타는 나무 위에 위치하면 해당 나무가 타버릴 확률이 증가합니다.

```python
# envs/wildfire.py:811
P(on fire → burnt) = 1 - β + δ_β × agent_above
```

**파라미터:**
- `beta` (β): 에이전트 없을 때 불이 계속 타는 확률 (기본값: 0.99)
- `delta_beta` (δ_β): 에이전트 진화 효과 (기본값: 0)

#### 2. 효율 분석

**기본 소화 확률 (에이전트 없음):**
```
P(소화) = 1 - β = 1 - 0.99 = 0.01 (1%)
```

**에이전트 있을 때:**
```
P(소화) = 1 - β + δ_β = 1 - 0.99 + δ_β = 0.01 + δ_β
```

**예시:**
| δ_β 값 | 에이전트 없음 | 에이전트 있음 | 증가율 |
|--------|-------------|-------------|--------|
| 0.0 | 1% | 1% | 0% |
| 0.05 | 1% | 6% | 500% |
| 0.1 | 1% | 11% | 1000% |
| 0.2 | 1% | 21% | 2000% |
| 0.5 | 1% | 51% | 5000% |

#### 3. 평균 소화 시간

불이 완전히 꺼지는데 걸리는 평균 타임스텝:

**에이전트 없음:**
```
E[T_소화] = 1 / (1 - β) = 1 / 0.01 = 100 타임스텝
```

**에이전트 있음:**
```
E[T_소화] = 1 / (1 - β + δ_β)
```

**예시:**
| δ_β | 평균 소화 시간 | β=0.99 대비 개선 |
|-----|-------------|----------------|
| 0.0 | 100 steps | 0% |
| 0.05 | 16.67 steps | 83.3% |
| 0.1 | 9.09 steps | 90.9% |
| 0.2 | 4.76 steps | 95.2% |
| 0.5 | 1.96 steps | 98.0% |

#### 4. 소방 전략 시사점

**최적 에이전트 할당:**
- 에이전트는 **불타는 나무 위**에 배치되어야 효과적
- 여러 에이전트가 한 나무에 있어도 추가 효과 없음 (agent_above는 0 또는 1)
- 에이전트를 분산 배치하는 것이 효율적

**화재 경계 우선순위:**
```python
# envs/wildfire.py:715-734
def _on_fire_boundary(self, i, j):
    """화재 경계 확인: 건강한 이웃이 있는 불타는 나무"""
    for r in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        o = self.helper_grid.get(i + r[0], j + r[1])
        if o is not None and o.type == "tree" and o.state == 0:
            return True
    return False
```

경계의 불타는 나무를 우선 진압하면 확산을 효과적으로 차단할 수 있습니다.

---

## 보상 시스템

### 1. 협력 보상 (Cooperative Reward)

```python
# envs/wildfire.py:851-852
if self.cooperative_reward:
    agent_rewards -= 0.5 * len(trees_to_fire_state)
```

**특징:**
- 모든 에이전트가 동일한 보상 수령
- 새로 불붙은 나무 수에 비례하여 페널티
- 완전한 협력적 목표

**수식:**
```
r_i(t) = -0.5 × |새로 불붙은 나무|  (모든 i에 대해 동일)
```

### 2. 이기적 보상 (Selfish Reward)

```python
# envs/wildfire.py:854-862
for a in self.agents:
    agent_rewards[a.index] -= 0.5 * (
        num_trees_to_fire_state_sr[f"{a.index}"]
        + self.selfishness_weight * (
            len(trees_to_fire_state) - num_trees_to_fire_state_sr[f"{a.index}"]
        )
    )
```

**수식:**
```
r_i(t) = -0.5 × [n_i + λ × (n_total - n_i)]
```
- `n_i`: 에이전트 i의 이기적 영역에서 새로 불붙은 나무 수
- `n_total`: 전체 그리드에서 새로 불붙은 나무 수
- `λ` (selfishness_weight): 이기심 가중치 (기본값: 0.2, 범위: [0, 1))

**이기적 영역 정의:**
```python
# envs/wildfire.py:691-713
def in_selfish_region(self, i: int, j: int, region_index: int) -> bool:
    return (
        i >= self.selfish_xmin[region_index] and
        i <= self.selfish_xmax[region_index] and
        j >= self.selfish_ymin[region_index] and
        j <= self.selfish_ymax[region_index]
    )
```

### 3. 보상 구조 비교

| λ 값 | 의미 | 행동 특성 |
|------|------|----------|
| 0.0 | 완전 이기적 | 자신의 영역만 보호 |
| 0.2 | 약간 이기적 | 자신의 영역 우선, 공동 영역도 일부 고려 |
| 0.5 | 중립 | 균형잡힌 행동 |
| 0.99 | 거의 협력적 | 전체 화재 진압 중시 |
| 1.0 | 완전 협력적 | cooperative_reward=True와 동일 |

**보상 분해 예시 (λ=0.2):**
- 에이전트 1의 영역에서 5개 불붙음
- 에이전트 2의 영역에서 3개 불붙음
- 공동 영역에서 2개 불붙음

```
r_1 = -0.5 × [5 + 0.2 × (10 - 5)] = -0.5 × 6 = -3.0
r_2 = -0.5 × [3 + 0.2 × (10 - 3)] = -0.5 × 4.4 = -2.2
```

### 4. 종료 조건

```python
# envs/wildfire.py:842-847
if self.trees_on_fire == 0:
    terminated = True
    rewards = {f"{a.index}": 0 for a in self.agents}
elif self.step_count >= self.max_steps:
    truncated = True
    rewards = {f"{a.index}": 0 for a in self.agents}
```

**종료 시나리오:**
1. **성공**: 모든 불이 꺼짐 (`trees_on_fire == 0`)
2. **타임아웃**: 최대 스텝 도달 (`step_count >= max_steps`)

---

## 관찰 공간 및 상태 표현

### 1. 관찰 벡터 구조

```python
# envs/wildfire.py:115-116
self.obs_depth = self.num_agents + len(STATE_IDX_TO_COLOR_WILDFIRE)
```

**관찰 깊이 (obs_depth):**
- 3개 채널: 나무 상태 (healthy, on fire, burnt)
- 1개 채널: 벽
- n개 채널: 다른 에이전트 위치 (n = num_agents - 1)

**총 깊이 = 3 + 1 + (num_agents - 1) = 2 + num_agents**

### 2. 에이전트 중심 좌표

```python
# envs/wildfire.py:411-416
nc = [i - a.pos[0], j - a.pos[1]]
# 토로이달 좌표로 변환
if nc[0] < 0:
    nc[0] += self.grid_size_without_walls + 1
if nc[1] < 0:
    nc[1] += self.grid_size_without_walls + 1
```

**특징:**
- 각 에이전트는 자신을 중심으로 한 좌표계 사용
- 토로이달(toroidal) 변환: 음수 좌표를 양수로 변환
- 벽을 제외한 그리드 크기 사용

### 3. 원-핫 인코딩

```python
# envs/wildfire.py:418-424
if obj.type == "tree":
    agent_obs[a.index][obj.state, nc[1], nc[0]] = 1
elif obj.type == "wall":
    agent_obs[a.index][len(STATE_IDX_TO_COLOR_WILDFIRE), nc[1], nc[0]] = 1
```

각 (x, y) 위치에 대해 obs_depth 차원의 원-핫 벡터가 할당됩니다.

### 4. 관찰 벡터 차원

```python
# envs/wildfire.py:219-221
low = np.full(self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 0)
```

**총 차원:**
```
dim = obs_depth × (grid_size_without_walls + 1)² + 1
```

**예시 (기본 설정):**
- grid_size = 17
- grid_size_without_walls = 15
- num_agents = 2
- obs_depth = 2 + 2 = 4

```
dim = 4 × (15 + 1)² + 1 = 4 × 256 + 1 = 1025
```

마지막 +1은 정규화된 타임스텝:
```python
# envs/wildfire.py:448-451
agent_obs[a.index] = np.append(
    agent_obs[a.index].flatten(),
    np.array(self.step_count / self.max_steps, dtype=np.float32),
)
```

### 5. 상태 표현 (State)

```python
# envs/wildfire.py:454-493
def get_state(self):
    """환경의 전역 상태 반환"""
    s = np.zeros(
        (self.obs_depth + 1, self.grid_size, self.grid_size),
        dtype=np.float32,
    )
    # 나무, 벽, 에이전트 위치 인코딩
    ...
    return s.flatten() + [normalized_timestep]
```

**차원:**
```
state_dim = (obs_depth + 1) × grid_size² + 1
```

**관찰 vs 상태:**
- **관찰 (Observation)**: 에이전트 중심 좌표, 부분 관찰 가능
- **상태 (State)**: 절대 좌표, 전체 환경 정보

---

## 주요 파라미터

### 산불 동역학 파라미터

| 파라미터 | 기호 | 기본값 | 범위 | 설명 |
|---------|-----|-------|------|------|
| `alpha` | α | 0.05 | [0, 1] | 불 확산 확률 계수 |
| `beta` | β | 0.99 | [0, 1] | 불이 계속 타는 확률 |
| `delta_beta` | δ_β | 0 | [0, 1] | 에이전트 진화 효과 |

**권장 설정:**
- **쉬운 화재**: α=0.03, β=0.97, δ_β=0.1
- **보통 화재**: α=0.05, β=0.99, δ_β=0.05
- **어려운 화재**: α=0.1, β=0.995, δ_β=0.02

### 환경 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `size` | 17 | 그리드 한 변의 길이 (벽 포함) |
| `num_agents` | 2 | UAV 에이전트 수 |
| `initial_fire_size` | 1 | 초기 화재 영역 크기 (정사각형) |
| `max_steps` | 100 | 에피소드 최대 타임스텝 |
| `partial_obs` | False | 부분 관찰 가능성 활성화 |
| `agent_view_size` | 10 | 부분 관찰 시 시야 범위 |

### 보상 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `cooperative_reward` | False | {True, False} | 협력 보상 사용 여부 |
| `selfishness_weight` | 0.2 | [0, 1) | 이기심 가중치 λ |

### 에이전트 배치 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `agent_start_positions` | ((1,1), (15,15)) | 에이전트 초기 위치 튜플 |
| `agent_colors` | ("red", "blue") | 에이전트 색상 |
| `agent_groups` | None | 에이전트 그룹 (협력 단위) |

### 이기적 영역 파라미터

| 파라미터 | 설명 |
|---------|------|
| `selfish_region_xmin` | 이기적 영역 좌측 경계 리스트 |
| `selfish_region_xmax` | 이기적 영역 우측 경계 리스트 |
| `selfish_region_ymin` | 이기적 영역 상단 경계 리스트 |
| `selfish_region_ymax` | 이기적 영역 하단 경계 리스트 |

**예시:**
```python
env = WildfireEnv(
    selfish_region_xmin=[1, 9],
    selfish_region_xmax=[8, 15],
    selfish_region_ymin=[1, 1],
    selfish_region_ymax=[15, 15],
)
```
- 에이전트 0: (1,1)~(8,15) 영역에 이기적 관심
- 에이전트 1: (9,1)~(15,15) 영역에 이기적 관심

---

## 성능 및 최적화

### Helper Grid

```python
# envs/wildfire.py:370-371
self.helper_grid = self.grid.copy()
```

**목적:**
- 그리드는 한 셀에 하나의 오브젝트만 저장 가능
- 에이전트가 나무 위에 있을 때 나무 정보를 유지하기 위해 helper_grid 사용
- helper_grid는 에이전트를 포함하지 않고 나무만 저장

### 타일 캐싱

```python
# core/grid.py:14-15, 268-309
tile_cache = {}  # 클래스 레벨 캐시

if key in cls.tile_cache:
    return cls.tile_cache[key]
```

**최적화:**
- 렌더링된 타일을 캐싱하여 반복 렌더링 성능 향상
- 동일한 (오브젝트, 색상, 하이라이트) 조합 재사용

### 동기 업데이트

```python
# envs/wildfire.py:787-839
# 1. 전이 계산
for c in self.unburnt_trees:
    if condition:
        trees_to_fire_state.append(c)

# 2. 일괄 업데이트
for c in trees_to_fire_state:
    c.state = 1
```

**이점:**
- 순서 의존성 제거
- 공정한 확률 계산
- 병렬화 가능

---

## 확장 및 커스터마이징

### 1. 커스텀 액션 추가

```python
class ExtendedWildfireActions(enum.IntEnum):
    STILL = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4
    SPRAY_WATER = 5  # 새 액션
```

### 2. 커스텀 보상 함수

`step()` 메서드의 보상 계산 부분을 오버라이드:

```python
class CustomWildfireEnv(WildfireEnv):
    def step(self, actions):
        # ... 기존 로직 ...

        # 커스텀 보상
        for a in self.agents:
            distance_to_fire = self.compute_distance_to_nearest_fire(a.pos)
            rewards[a.index] += -0.1 * distance_to_fire
```

### 3. 다양한 화재 패턴

`_gen_grid()`를 수정하여 다양한 초기 화재 패턴 생성:

```python
# 링 모양 화재
def create_ring_fire(center, radius):
    fire_positions = []
    for angle in range(0, 360, 10):
        x = center[0] + radius * cos(radians(angle))
        y = center[1] + radius * sin(radians(angle))
        fire_positions.append((int(x), int(y)))
    return fire_positions
```

---

## 결론

Wildfire Environment는 다음과 같은 강점을 가진 멀티 에이전트 강화학습 환경입니다:

1. **현실적 동역학**: 확률적 산불 확산 모델이 실제 화재의 불확실성을 반영
2. **유연한 보상 구조**: 협력적/이기적 행동 연구 가능
3. **확장 가능성**: 모듈화된 설계로 커스터마이징 용이
4. **효율적 구현**: 캐싱 및 벡터화를 통한 성능 최적화

**활용 분야:**
- 멀티 에이전트 강화학습 알고리즘 벤치마킹
- 협력/경쟁 행동 연구
- 재난 대응 시뮬레이션
- 자원 할당 전략 학습

---

## 참조

### 주요 파일 위치

- 환경 정의: `envs/wildfire.py`
- 에이전트: `core/agent.py`
- 그리드: `core/grid.py`
- 오브젝트: `core/object.py`
- 상수: `core/constants.py`

### 핵심 메서드

- `WildfireEnv.step()`: 환경 시뮬레이션 (envs/wildfire.py:736)
- `WildfireEnv.neighbors_on_fire()`: 이웃 불 계산 (envs/wildfire.py:660)
- `WildfireEnv._gen_grid()`: 그리드 초기화 (envs/wildfire.py:236)
- `WildfireEnv.move_agent()`: 에이전트 이동 (envs/wildfire.py:637)
