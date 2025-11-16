# new_compare_visual.py 이질적 에이전트 지원 업데이트

## 📋 개요

`new_compare_visual.py`를 이질적 에이전트(헬리콥터, 트럭, 승무원)를 지원하도록 업데이트했습니다.
학습된 각 정책별로 에이전트의 행동을 시각화하고 랜덤 정책과 비교할 수 있습니다.

---

## 🔄 주요 변경사항

### 1. **새로운 함수 추가**

#### `load_heterogeneous_policies(checkpoint_path)`
- **목적**: 이질적 정책들(helicopter_policy, truck_policy, crew_policy)을 체크포인트에서 로드
- **반환값**:
  - 이질적 정책: `{'helicopter_policy': weights, 'truck_policy': weights, ...}`
  - 단일 정책: `{'shared': weights}` (호환성)
- **특징**:
  - 각 정책의 가중치를 별도로 추출
  - 단일 정책 학습과의 호환성 유지

#### `get_agent_policy_name(agent_id, num_helicopters, num_trucks, num_crews)`
- **목적**: 에이전트 ID → 정책 이름 매핑
- **예시**:
  - agent_0, agent_1 → `helicopter_policy`
  - agent_2, agent_3 → `truck_policy`
  - agent_4+ → `crew_policy`

### 2. **함수 시그니처 수정**

#### `run_episode_and_render()`
**이전**:
```python
run_episode_and_render(env, policy_network, seed, max_steps=300, policy_type="trained")
```

**수정 후**:
```python
run_episode_and_render(env, policy_networks, policy_mapping_fn, seed,
                      max_steps=300, policy_type="trained")
```

**변경사항**:
- `policy_network` → `policy_networks` (딕셔너리)
- `policy_mapping_fn` 파라미터 추가 (정책 매핑 함수)
- 에이전트별로 적절한 정책 네트워크 선택

### 3. **main() 함수 업데이트**

#### 체크포인트 로드 로직
```python
# 이질적 정책 로드
policies_weights = load_heterogeneous_policies(checkpoint_path)

# 정책 네트워크 생성 및 가중치 로드
policy_networks = {}
for policy_name in ['helicopter_policy', 'truck_policy', 'crew_policy']:
    if policy_name in policies_weights:
        policy_networks[policy_name] = create_policy_network(obs_dim, action_dim)
        policy_networks[policy_name].load_state_dict(policies_weights[policy_name], strict=False)
```

#### 에피소드 실행
```python
# 학습된 정책 실행
trained_frames, trained_reward, trained_length = run_episode_and_render(
    env_trained, policy_networks, policy_mapping_fn, episode_seed,
    max_steps=env_config['max_steps'], policy_type="trained"
)

# 랜덤 정책 실행
random_frames, random_reward, random_length = run_episode_and_render(
    env_random, None, None, episode_seed,
    max_steps=env_config['max_steps'], policy_type="random"
)
```

### 4. **통계 저장 업데이트**

통계 파일에 에이전트 구성 정보 추가:
```
Agent Configuration:
  - Helicopters: 2
  - Trucks: 2
  - Crews: 0
```

---

## 🔧 코드 구조 비교

### 이전 (단일 정책)
```
Checkpoint
  └─ shared_policy (모든 에이전트 공유)

run_episode_and_render(env, policy_network, seed, ...)
  └─ 모든 에이전트 → policy_network.get_action(obs)
```

### 수정 후 (이질적 정책)
```
Checkpoint
  ├─ helicopter_policy
  ├─ truck_policy
  └─ crew_policy

run_episode_and_render(env, policy_networks, policy_mapping_fn, seed, ...)
  ├─ agent_0, agent_1 → policy_networks['helicopter_policy']
  ├─ agent_2, agent_3 → policy_networks['truck_policy']
  └─ agent_4+ → policy_networks['crew_policy']
```

---

## 📊 호환성

| 시나리오 | 지원 여부 |
|---------|---------|
| 이질적 정책 학습 | ✅ 완벽 지원 |
| 단일 정책 학습 | ✅ 호환성 유지 (shared_policy) |
| 부분적 정책 (일부만 로드) | ✅ 랜덤 액션으로 대체 |

---

## 🚀 사용 방법

### 기본 실행
```bash
python train_marllib_self/new_compare_visual.py \
    --checkpoint <checkpoint_path> \
    --episodes 3 \
    --seed 42
```

### 출력 디렉토리 지정
```bash
python train_marllib_self/new_compare_visual.py \
    --checkpoint <checkpoint_path> \
    --output-dir ./my_results/
```

### 출력 파일
- `comparison_ep01_seed42.gif`: 에피소드 1 (학습된 정책 vs 랜덤 정책)
- `comparison_ep02_seed43.gif`: 에피소드 2
- `comparison_ep03_seed44.gif`: 에피소드 3
- `comparison_stats.txt`: 통계 요약

---

## 📝 주석 처리된 코드

이전의 단일 정책 지원 코드는 모두 주석 처리되어 있습니다:
- `load_policy_weights()` (line 36-71)
- 이전 `run_episode_and_render()` (line 169-228)
- main의 정책 로드 부분 (line 503-546)
- main의 에피소드 실행 부분 (line 620-638)

필요시 주석을 제거하여 이전 코드로 복귀 가능합니다.

---

## ✨ 개선사항

1. **유연한 정책 매핑**: 에이전트 타입별로 서로 다른 정책 적용 가능
2. **호환성**: 단일 정책 학습과도 호환
3. **명확한 시각화**: 에이전트별 정책 할당이 명시적으로 드러남
4. **견고한 에러 처리**: 정책 로드 실패 시 랜덤 액션으로 대체
5. **상세한 로깅**: 어떤 정책이 로드되었는지 명확하게 표시

---

## 📚 관련 파일

- `new_train_mappo.py`: 이질적 정책으로 학습 (line 179-203)
- `new_wrapper.py`: 이질적 에이전트 환경 래퍼 (line 15-27)
- `environment.py`: 환경 설정 (에이전트 구성 정의)
- `validate_config.py`: 설정 검증 도구
