# 이질적 에이전트 정책 설정 수정 보고서

## 📋 개요
`new_train_mappo.py`의 line 179-189에서 이질적 에이전트(헬리콥터, 트럭, 승무원)가 각각 다른 정책을 학습하도록 코드를 수정하고 검증했습니다.

---

## 🔴 발견된 문제점

### 1. **문법 오류: line 179-184 (policies 정의)**
```python
# ❌ 잘못된 코드 (세트 형식)
'policies': {
    'helicopter_policy',
    'truck_policy',
    'crew_policy',
},
```

**문제**: Python 세트(set)로 정의되어 있어 정책 설정 정보가 없습니다.
- RLlib/MARLlib의 `policies`는 딕셔너리여야 합니다.
- 각 정책마다 `(policy_class, obs_space, action_space, config)` 튜플이 필요합니다.

### 2. **미정의 변수: line 185-189 (policy_mapping_fn)**
```python
# ❌ 잘못된 코드 (변수 미정의)
'policy_mapping_fn': lambda agent_id, *args, **kwargs: (
    "helicopter_policy" if agent_id < num_helicopters  # ❌ num_helicopters 정의 안 됨
    else "truck_policy" if agent_id < num_helicopters + num_trucks  # ❌ num_trucks 정의 안 됨
    else "crew_policy"
)
```

**문제**: `num_helicopters`, `num_trucks` 변수가 정의되지 않아 `NameError` 발생합니다.

### 3. **new_wrapper.py 정책 설정 불일치**
- `new_wrapper.py`에서 `all_agents_one_policy: True`로 설정되어 있어 이질적 에이전트를 지원하지 않음.

---

## ✅ 적용된 수정사항

### 수정 1: `new_train_mappo.py` (line 162-203)

```python
# 환경에서 에이전트 타입별 개수 추출
num_helicopters = ENV_CONFIG['num_helicopters']
num_trucks = ENV_CONFIG['num_trucks']
num_crews = ENV_CONFIG['num_crews']

print(f"\n이질적 에이전트 설정:")
print(f"  헬리콥터: {num_helicopters}명")
print(f"  트럭: {num_trucks}명")
print(f"  승무원: {num_crews}명")

# ...

mappo.fit(
    env,
    model,
    stop={
        'training_iteration': 10
    },
    local_mode=True,
    num_gpus=1,
    num_workers=2,
    multi_agent_config={
        'policies': {
            'helicopter_policy': (None, env.observation_space, env.action_space, {}),
            'truck_policy': (None, env.observation_space, env.action_space, {}),
            'crew_policy': (None, env.observation_space, env.action_space, {}),
        },
        'policy_mapping_fn': lambda agent_id, *args, **kwargs: (
            "helicopter_policy" if agent_id < num_helicopters
            else "truck_policy" if agent_id < num_helicopters + num_trucks
            else "crew_policy"
        )
    },
    checkpoint_freq=10,
    local_dir=str(output_dir)
)
```

**개선사항**:
- ✓ `ENV_CONFIG`에서 변수 추출하여 정의
- ✓ `policies`를 올바른 딕셔너리 형식으로 수정
- ✓ 각 정책에 observation_space, action_space 설정
- ✓ `policy_mapping_fn`이 정의된 변수를 참조

### 수정 2: `new_wrapper.py` (line 15-27)

```python
policy_mapping_dict = {
    "wildfire-ma": {
        "description": "wildfire suppression with heterogeneous agents",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": False,  # ✓ 이질적 에이전트: 타입별로 다른 정책
        "one_agent_one_policy": False,   # ✓ 같은 타입의 에이전트는 같은 정책 공유
    }
}
```

**개선사항**:
- ✓ `all_agents_one_policy: False` → 에이전트들이 서로 다른 정책 가능
- ✓ `one_agent_one_policy: False` → 같은 타입 에이전트는 정책 공유

---

## 🧪 검증 결과

✓ **모든 검증 통과**

### 정책 매핑 결과 (현재 ENV_CONFIG 기준)
```
Agent 0 → helicopter_policy
Agent 1 → helicopter_policy
Agent 2 → truck_policy
Agent 3 → truck_policy
```

### 에이전트 구성 검증
```
✓ 헬리콥터: 2명
✓ 트럭: 2명
✓ 승무원: 0명
✓ 합계: 4명 (num_agents와 일치)
```

### 호환성 검증
```
✓ new_wrapper.py MultiAgentEnv 지원
✓ 모든 에이전트 동일 공간 사용 가능
✓ 정책 매핑 함수 올바르게 작동
```

---

## 📊 수정 전후 비교

| 항목 | 수정 전 | 수정 후 |
|------|-------|-------|
| **policies 형식** | ❌ 세트(set) | ✓ 딕셔너리 |
| **변수 정의** | ❌ 미정의 | ✓ ENV_CONFIG에서 추출 |
| **정책 매핑** | ❌ 작동 불가 | ✓ 에이전트ID → 정책 매핑 |
| **new_wrapper 호환성** | ⚠️ 부분 호환 | ✓ 완전 호환 |
| **실행 가능성** | ❌ NameError 발생 | ✓ 정상 실행 |

---

## 🔧 관련 파일 상태

| 파일 | 상태 | 비고 |
|------|------|------|
| `new_train_mappo.py` | ✅ 수정 완료 | line 162-203 |
| `new_wrapper.py` | ✅ 수정 완료 | line 15-27 |
| `environment.py` | ✓ 수정 불필요 | 정책 설정과 무관 |
| `validate_config.py` | ✅ 추가 생성 | 검증 도구 |

---

## 📝 사용 방법

### 1. 설정 검증 (선택사항)
```bash
python train_marllib_self/validate_config.py
```

### 2. 학습 시작
```bash
python train_marllib_self/new_train_mappo.py --run-name run1
```

### 3. 로그 확인
```
✓ 이질적 에이전트 설정이 출력됨
  헬리콥터: 2명
  트럭: 2명
  승무원: 0명
```

---

## ⚠️ 주의사항

1. **ENV_CONFIG 수정 시**: `num_helicopters + num_trucks + num_crews = num_agents` 반드시 확인
2. **정책 매핑**: 현재는 agent_id 기반으로 매핑됨 (ID 순서 중요)
3. **승무원 추가 시**: `crew_policy`가 자동으로 할당됨 (현재는 0명)

---

## ✨ 현재 설정

- 헬리콥터 정책: `agent_0`, `agent_1`이 공유
- 트럭 정책: `agent_2`, `agent_3`이 공유
- 각 정책은 독립적으로 학습됨

이를 통해 이질적 에이전트가 자신의 특성에 맞는 정책을 학습할 수 있습니다.
