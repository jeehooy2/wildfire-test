# 빠른 수렴을 위한 최적화 전략 (우선순위 순)

## 🎯 핵심 원칙
- **우선 partial_obs=True로 돌아가기** (가장 효과적)
- 보상함수 개선 (sparse reward 해결)
- 하이퍼파라미터 튜닝
- 네트워크 아키텍처 최적화

---

## 1️⃣ TIER 1: 즉시 적용 (가장 효과적)

### 1.1 `partial_obs=True` 복원 ⭐⭐⭐⭐⭐
**효과: 가장 크다**

```python
# environment.py
"partial_obs": True,            # ← False에서 True로 변경
"agent_view_size": 7,           # 5×5 → 7×7 (약간 증가)
```

**왜 효과적인가:**
- 관찰 공간 감소: 484 → 49 (약 10배 감소)
- 더 빠른 신호: 에이전트 움직임에 따라 자주 변화
- 학습 시간: 약 50-70% 감소
- 이미 검증된 설정

**trade-off:** 에이전트가 "근시안적"이 될 수 있지만, 대부분의 경우 문제 없음

---

### 1.2 시간 페널티 강화 ⭐⭐⭐⭐
**효과: 높음 (episode_len 단축 직접 효과)**

```python
# wildfire_environment/envs/reward_functions.py - cooperative3_reward 함수

# 현재 (line 502-515)
time_penalty = current_step / max_steps if max_steps > 0 else 0.0
agent_reward = (
    0.8 * shared_reward +
    0.2 * individual_reward -
    0.02 * time_penalty  # ← 너무 약함
)

# 변경
time_penalty = (current_step / max_steps) ** 1.5  # 후반부 가중
agent_reward = (
    0.8 * shared_reward +
    0.2 * individual_reward -
    0.2 * time_penalty  # ← 10배 강화
)
```

**효과:**
- 300 스텝에서: 0.02 → 0.2 (10배 증가)
- 후반부(250+ 스텝)에 더 강한 페널티
- 에피소드 길이 20-30% 단축

---

### 1.3 보상 스케일 최적화 ⭐⭐⭐⭐
**효과: 높음 (수렴 속도 개선)**

```python
# cooperative3_reward 함수 수정

# 현재
shared_reward = (
    0 * T_p +
    10.0 * T_e -      # 너무 크지도 작지도 않음
    1.0 * T_n -
    5.0 * T_b
)

# 개선안 1: 더 명확한 신호 (추천)
shared_reward = (
    20.0 * T_e -      # ← 보상 2배 (신호 강화)
    2.0 * T_n -       # ← 페널티 증가
    10.0 * T_b        # ← 페널티 증가
)

# 개선안 2: 건강한 나무 비율 명시적 반영
shared_reward = (
    20.0 * T_e +
    5.0 * (total_healthy_trees / total_trees) -  # 건강한 나무 많을수록 +
    2.0 * T_n -
    10.0 * T_b
)
```

**효과:**
- 더 명확한 그래디언트 → 빠른 정책 업데이트
- 수렴 속도 15-25% 개선

---

## 2️⃣ TIER 2: 중요한 하이퍼파라미터 조정

### 2.1 학습률(Learning Rate) 증가 ⭐⭐⭐
**현재: 0.0005 → 변경: 0.001 또는 0.002**

```python
# new_train_ppo.py (line 147)

# 현재
ppo = marl.algos.ippo(
    hyperparam_source="common",
    batch_episode=10,
    lr=0.0005,        # ← 작음
    clip_param=0.3
)

# 변경
ppo = marl.algos.ippo(
    hyperparam_source="common",
    batch_episode=10,
    lr=0.001,         # ← 2배 증가
    clip_param=0.2    # 약간 감소 (안정성 유지)
)
```

**효과:**
- 초반 학습 속도 가속
- 수렴 속도 20-30% 개선
- 주의: 너무 높으면 불안정해짐 (0.002 이상 비추천)

---

### 2.2 배치 에피소드 수 증가 ⭐⭐⭐
**현재: 10 → 변경: 20 또는 30**

```python
# new_train_ppo.py (line 146)

# 현재
ppo = marl.algos.ippo(
    batch_episode=10,  # 한 번에 10 에피소드 모음
)

# 변경 (추천)
ppo = marl.algos.ippo(
    batch_episode=20,  # ← 한 번에 20 에피소드 (더 안정적)
)
```

**효과:**
- 정책 업데이트 안정성 증가
- 학습 곡선 더 smooth
- 수렴 속도 약 10% 개선

**주의:**
- 배치 크기 = batch_episode × num_workers × max_steps
- 배치 크기가 커지면 업데이트 빈도 감소 (trade-off)

---

### 2.3 Clip Parameter 조정 ⭐⭐⭐
**현재: 0.3 → 변경: 0.2**

```python
# new_train_ppo.py (line 148)

ppo = marl.algos.ippo(
    clip_param=0.3,   # ← 현재 (좀 큼)
)

# 변경
ppo = marl.algos.ippo(
    clip_param=0.2,   # ← 더 보수적 (안정적)
)
```

**효과:**
- PPO의 신뢰 영역(trust region) 축소
- 더 작은 정책 업데이트 → 안정성
- 수렴 속도는 약간 느리지만 더 안정적

---

## 3️⃣ TIER 3: 고급 최적화 (선택사항)

### 3.1 워커(Worker) 수 증가 ⭐⭐
**현재: 4 → 변경: 8 또는 12**

```python
# new_train_ppo.py (line 188)

ppo.fit(
    env,
    model,
    num_workers=4,    # ← 현재
)

# 변경
ppo.fit(
    env,
    model,
    num_workers=8,    # ← 2배 (더 많은 샘플 생성)
)
```

**효과:**
- 병렬 샘플 수집: 더 많은 경험 데이터
- 수렴 속도 10-20% 개선
- **주의:** GPU 메모리 부족 가능 (4개 워커로 충분할 수 있음)

---

### 3.2 네트워크 아키텍처 최적화 ⭐⭐

```python
# new_train_ppo.py (line 156-159)

# 현재
model = marl.build_model(
    env,
    ppo,
    {
        "core_arch": "mlp",
        "encode_layer": "256-256"  # 256-256
    }
)

# 옵션 1: 더 큰 네트워크 (full obs에 맞춤)
{
    "core_arch": "mlp",
    "encode_layer": "512-512"     # 더 큼
}

# 옵션 2: 더 작은 네트워크 (빠른 학습, partial obs에 맞춤)
{
    "core_arch": "mlp",
    "encode_layer": "128-128"     # 더 작음
}
```

**partial_obs=True일 때:** `128-128` 또는 `256-256` 추천
**partial_obs=False일 때:** `512-512` 이상 필요할 수도 있음

---

## 4️⃣ TIER 4: 보상함수 교체 (극단적 옵션)

### 4.1 다른 보상함수 시도 ⭐⭐

```python
# environment.py (line 48)

# 현재
"reward_shaping": "cooperative3",

# 시도해볼 대안
"reward_shaping": "cooperative2",  # 더 단순한 보상
# 또는
"reward_shaping": "ramadan",       # 누적 기반 보상
```

**각 함수의 특징:**
- `cooperative2`: 개별+공동 보상 균형, 간단함
- `cooperative3`: 더 정교함, 하지만 튜닝 필요
- `ramadan`: 누적 기반, 에피소드 전체 봄

---

## 🚀 추천 실행 순서

### **빠른 수렴 (1-2일)**

```
1단계 (필수):
  ✅ partial_obs=True 복원
  ✅ agent_view_size=7 설정

2단계 (권장):
  ✅ 시간 페널티 강화 (0.02 → 0.2)
  ✅ 보상 스케일 2배 증가

3단계 (선택):
  ✅ 학습률 0.0005 → 0.001
  ✅ batch_episode 10 → 20
```

### **결과**
- 초기: 약 100-150 iteration에 수렴
- 개선후: 약 30-50 iteration에 수렴 **(약 3배 빠름)**

---

## 📊 비교표

| 전략 | 효과 | 구현 난이도 | 예상 개선 |
|------|------|-----------|---------|
| partial_obs=True 복원 | ⭐⭐⭐⭐⭐ | 쉬움 | **50-70%** |
| 시간 페널티 강화 | ⭐⭐⭐⭐ | 쉬움 | **20-30%** |
| 보상 스케일 증가 | ⭐⭐⭐⭐ | 쉬움 | **15-25%** |
| 학습률 증가 | ⭐⭐⭐ | 쉬움 | **20-30%** |
| batch_episode 증가 | ⭐⭐⭐ | 쉬움 | **10%** |
| 워커 수 증가 | ⭐⭐ | 쉬움 | **10-20%** |
| 네트워크 크기 조정 | ⭐⭐ | 중간 | **5-15%** |

---

## ⚠️ 주의사항

### 한 번에 너무 많이 바꾸지 않기
- 변수를 하나씩 바꾸고 테스트하기
- 어떤 변경이 효과적인지 파악 가능

### Partial Obs 복원의 트레이드오프
- **장점:** 빠른 수렴, 학습 효율 높음
- **단점:** 에이전트가 "근시안적" (먼 불을 못 봄)
  - 해결: `agent_view_size=7` (5×5가 아닌 7×7로 약간 확대)

### 안정성 vs 속도
- 속도만 추구하면 학습 불안정
- 추천: 시간 페널티만 강화하고 lr은 보수적으로 유지

---

## 🎓 최종 추천 설정

```python
# environment.py
ENV_CONFIG = {
    "partial_obs": True,           # ← 변경
    "agent_view_size": 7,          # ← 변경
    "reward_shaping": "cooperative3",
    # 나머지는 동일
}

# new_train_ppo.py
ppo = marl.algos.ippo(
    batch_episode=20,              # ← 10에서 변경
    lr=0.001,                      # ← 0.0005에서 변경
    clip_param=0.2                 # ← 0.3에서 변경
)

# reward_functions.py - cooperative3_reward
time_penalty = (current_step / max_steps) ** 1.5  # ← 변경
agent_reward = (
    0.8 * shared_reward +
    0.2 * individual_reward -
    0.2 * time_penalty             # ← 0.02에서 변경
)
```

**예상 결과:**
- 현재: 100-150 iteration
- 개선: 30-40 iteration **(약 3배 빠름)**

---

## 🔍 디버깅 팁

### 학습이 불안정해지면:
1. lr 조금 줄이기 (0.0005로 복원)
2. clip_param 늘리기 (0.3으로)
3. batch_episode 증가 (30으로)

### 여전히 느리면:
1. partial_obs=True 재확인
2. 보상 스케일 추가로 증가 (20배 → 40배)
3. 다른 보상함수 시도 (cooperative2)

