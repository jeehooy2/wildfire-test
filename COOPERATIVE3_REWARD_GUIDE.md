# Cooperative3 Reward Function Guide

## Overview

`cooperative3_reward`는 산불 진화 에이전트의 두 가지 주요 목표를 최적화하기 위해 설계된 개선된 협력 보상함수입니다:

1. **피해 면적 최소화**: 건강한 나무 비율 최대화 (에피소드 종료 시)
2. **진화 시간 최소화**: 에피소드 길이 최소화

## Design Philosophy

### 핵심 아이디어

기존 보상함수들의 문제점:
- **Sparse 양의 보상**: 진화에만 보상을 주므로, 진화가 없는 스텝에서는 음의 보상만 받음
- **과도한 음의 페널티**: 새 불이 나면 큰 페널티로 에이전트가 주눅 들 수 있음
- **건강한 나무 비율 무시**: 최종 피해를 직접 반영하지 않음

### 해결책: 건강한 나무 비율을 메인 신호로

```
R = w_healthy_ratio * (healthy_trees / total_trees)    ← 메인 신호 (항상 양수)
    + w_extinguish * T_e                               ← 즉각적 보상
    - w_new_fire * T_n                                 ← 새 불 페널티
    - w_burnt * T_b                                    ← 소실 페널티
    - w_time_penalty * (step / max_steps)              ← 시간 페널티
```

**장점**:
1. **지속적 양의 신호**: 모든 스텝에서 건강한 나무 비율 기반 양의 보상 제공
2. **균형잡힌 페널티**: 음의 페널티를 낮춰 에이전트가 주눅들지 않도록 함
3. **최종 목표 직접 반영**: healthy tree ratio가 곧 성능 지표
4. **협력 강화**: 모든 에이전트가 동일한 보상을 받아 협력 동기 부여

## Mathematical Formulation

### 보상 구성요소

#### 1. 건강한 나무 비율 신호 (Main Signal)
```
r_health = w_healthy_ratio * (healthy_trees / total_trees)
```

- **범위**: [0, w_healthy_ratio] (기본: [0, 3.0])
- **특징**:
  - 항상 양수 (더 많은 건강한 나무 → 더 높은 보상)
  - 모든 스텝에서 지속적으로 제공
  - 이 신호만으로도 건강한 정책을 학습할 수 있음

#### 2. 진화 보상 (Extinguish Signal)
```
r_extinguish = w_extinguish * (이번 스텝 진화된 나무 수)
```

- **범위**: [0, ∞) (이론적으로)
- **특징**: 불을 끈 것에 대한 즉각적 양의 피드백
- **기본값**: w_extinguish = 1.5

#### 3. 새 불 페널티 (New Fire Penalty)
```
r_new_fire = -w_new_fire * (이번 스텝 새로 불타는 나무 수)
```

- **범위**: (-∞, 0] (이론적으로)
- **특징**: 새로 불이 나는 것을 억제하지만 과도하지는 않음
- **기본값**: w_new_fire = 0.5 (진화 보상의 1/3 수준)

#### 4. 소실 페널티 (Burnt Tree Penalty)
```
r_burnt = -w_burnt * (이번 스텝 소실된 나무 수)
```

- **범위**: (-∞, 0]
- **특징**: 불타서 소실된 나무에 대한 페널티
- **기본값**: w_burnt = 0.8

#### 5. 시간 페널티 (Time Penalty)
```
r_time = -w_time_penalty * (current_step / max_steps)
```

- **범위**: [0, -w_time_penalty]
- **특징**:
  - 진행도 기반 (0% → 100%)
  - 초반에는 낮은 패널티, 후반으로 갈수록 증가
  - 빠른 에피소드 종료 유도 (효율성)
- **기본값**: w_time_penalty = 0.02

### 총 보상 계산

```
R_total = r_health + r_extinguish + r_new_fire + r_burnt + r_time
```

## Default Configuration

```python
{
    "w_extinguish": 1.5,      # 진화 보상 가중치
    "w_healthy_ratio": 3.0,   # 건강한 나무 비율 가중치 (메인 신호)
    "w_new_fire": 0.5,        # 새 불 페널티 가중치
    "w_burnt": 0.8,           # 소실 페널티 가중치
    "w_time_penalty": 0.02,   # 시간 페널티 가중치
}
```

## Usage Examples

### 1. 기본 사용법

```python
from wildfire_environment.envs.wildfire import WildfireEnv

env = WildfireEnv(
    size=22,
    num_agents=6,
    max_steps=300,
    reward_shaping="cooperative3",  # 새 보상함수 활성화
)
```

### 2. 커스텀 하이퍼파라미터

```python
env = WildfireEnv(
    size=22,
    num_agents=6,
    max_steps=300,
    reward_shaping="cooperative3",
    reward_shaping_config={
        "w_extinguish": 2.0,        # 진화를 더 강조
        "w_healthy_ratio": 5.0,     # 건강 유지를 더 강조
        "w_new_fire": 0.3,          # 새 불 페널티를 낮춤
        "w_burnt": 1.0,             # 소실 페널티를 높임
        "w_time_penalty": 0.03,     # 시간 패널티를 강화
    },
)
```

### 3. training_config에서 설정 (MARLlib)

```python
ENV_CONFIG = {
    "size": 22,
    "num_agents": 6,
    "max_steps": 300,
    "reward_shaping": "cooperative3",
    "reward_shaping_config": {
        "w_extinguish": 1.5,
        "w_healthy_ratio": 3.0,
        "w_new_fire": 0.5,
        "w_burnt": 0.8,
        "w_time_penalty": 0.02,
    },
}
```

## Expected Behavior

### 에피소드 진행 중

1. **초반 (0-100 steps)**
   - 건강한 나무 비율이 높음 → 높은 기본 보상
   - 시간 페널티가 작음 → 빠른 학습 시작
   - 에이전트들이 불 진화에 집중

2. **중반 (100-200 steps)**
   - 불이 계속 퍼질 수 있음 → 건강 비율 감소
   - 보상이 감소 → 에이전트가 더 효율적이 되어야 함
   - 시간 페널티 증가 → 빠른 해결 압박

3. **후반 (200-300 steps)**
   - 대부분의 불이 꺼졌거나 소실됨
   - 건강 비율로 최종 성과 평가
   - 높은 시간 페널티 → 신속한 종료 강제

### 학습 곡선 특성

- **초반**: 빠른 학습 (충분한 양의 보상)
- **중반**: 안정적 개선 (균형잡힌 보상)
- **후반**: 미세 조정 (최적화 단계)

## Comparison with Other Reward Functions

| 특성 | cooperative | individual2 | cooperative2 | **cooperative3** |
|------|-----------|-----------|------------|--------------|
| 양의 신호 지속성 | 낮음 | 낮음 | 중간 | **높음** |
| 협력 강화 | 낮음 | 매우 낮음 | 중간 | **높음** |
| 시간 효율성 반영 | 없음 | 없음 | 약함 | **중간** |
| 건강한 나무 직접 반영 | 없음 | 없음 | 없음 | **높음** |
| Sparse 보상 문제 | **심각** | **심각** | 중간 | **해결** |
| 학습 안정성 | 중간 | 낮음 | 중간 | **높음** |

## Hyperparameter Tuning Guide

### 목표: 더 빠른 진화
```python
reward_shaping_config={
    "w_extinguish": 2.5,          # ↑ 진화 보상 증가
    "w_new_fire": 0.3,           # ↓ 새 불 페널티 감소
}
```

### 목표: 건강한 나무 보존에 집중
```python
reward_shaping_config={
    "w_healthy_ratio": 5.0,       # ↑ 건강 신호 증가
    "w_burnt": 1.5,              # ↑ 소실 페널티 증가
}
```

### 목표: 빠른 에피소드 완료
```python
reward_shaping_config={
    "w_time_penalty": 0.05,       # ↑ 시간 페널티 증가
}
```

### 목표: 더 탐험적인 학습
```python
reward_shaping_config={
    "w_healthy_ratio": 2.0,       # ↓ 메인 신호를 약하게
    "w_new_fire": 0.3,            # ↓ 새 불 페널티 낮게
}
```

## Testing Results

### 단위 테스트
```
✓ Basic reward calculation
✓ Time penalty progression
✓ Healthy ratio as main signal
✓ Positive reward signal distribution
✓ Function registration
```

### 통합 테스트
```
✓ Environment creation with cooperative3
✓ Episode execution (20 steps)
✓ Stable reward generation
✓ Proper agent cooperation (동일 보상)
```

## Files Modified

1. **wildfire_environment/envs/reward_functions.py**
   - `cooperative3_reward()` 함수 추가
   - `DEFAULT_CONFIGS` 업데이트
   - `get_reward_function()` 업데이트

2. **wildfire_environment/envs/wildfire.py**
   - `_compute_shaped_rewards()` 메서드에 "cooperative3" 처리 로직 추가

## Test Files

- `test_cooperative3_reward.py`: 단위 테스트 (5개 테스트 케이스)
- `test_integration_cooperative3.py`: 통합 테스트 (WildfireEnv 통합)

## Future Improvements

1. **에피소드 종료 보너스**: 에피소드 마지막에 최종 건강 비율 보너스 추가
2. **적응형 가중치**: 학습 진행도에 따라 가중치 자동 조정
3. **지역 기반 보상**: 특정 영역의 건강한 나무에 더 높은 가중치
4. **멀티 목표 최적화**: Pareto 최적성 고려

## References

- Design inspired by: Multi-agent RL reward shaping best practices
- Environment: Wildfire Suppression MDP
- Framework: RLlib + MARLlib
