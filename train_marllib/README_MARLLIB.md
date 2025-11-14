# MARLlib MAPPO 학습 가이드

이 문서는 MARLlib을 사용하여 진정한 MAPPO (CTDE: Centralized Training with Decentralized Execution)로 학습하는 방법을 설명합니다.

## 🎯 주요 차이점

### RLlib 기본 PPO vs MARLlib MAPPO

| 구분 | RLlib PPO (기존) | MARLlib MAPPO (신규) |
|------|-----------------|---------------------|
| **알고리즘** | Independent PPO (IPPO) | MAPPO (CTDE) |
| **Critic** | 각 에이전트마다 독립적 | **Centralized Critic** |
| **Critic 입력** | 자신의 observation만 | **Global state (전체 상태)** |
| **학습 효과** | 다른 에이전트를 환경의 일부로 취급 | **다른 에이전트의 행동을 고려** |
| **협력 능력** | 제한적 | **강력** |
| **구현** | `train_rllib_mappo_new.py` | `train_marllib_mappo.py` |

## 📦 설치 및 환경 설정

```bash
# MARLlib 환경 활성화
conda activate marllib-x86

# 프로젝트 디렉토리로 이동
cd /Users/jeehooy2/Desktop/wildfire_environment-aiblue
```

## 🚀 학습 실행

### 기본 실행

```bash
# 100 iterations 학습
conda activate marllib-x86
python train_rllib/train_marllib_mappo.py --iterations 100
```

### 고급 설정

```bash
# 더 긴 학습 (1000 iterations)
python train_rllib/train_marllib_mappo.py \
    --iterations 1000 \
    --checkpoint-freq 50 \
    --exp-name mappo_long_run

# Worker 수 조정 (병렬 처리)
python train_rllib/train_marllib_mappo.py \
    --iterations 500 \
    --num-workers 4

# GPU 사용
python train_rllib/train_marllib_mappo.py \
    --iterations 1000 \
    --num-gpus 1
```

### 학습 재개 (체크포인트에서)

```bash
python train_rllib/train_marllib_mappo.py \
    --iterations 500 \
    --restore ~/ray_results/mappo_wildfire/checkpoint_000100
```

## 📊 학습 모니터링

### TensorBoard

```bash
# 학습 곡선 실시간 확인
tensorboard --logdir ~/ray_results/mappo_wildfire/
```

브라우저에서 `http://localhost:6006` 접속

### 주요 지표

- `episode_reward_mean`: 평균 에피소드 보상
- `episode_len_mean`: 평균 에피소드 길이
- `policy_loss`: 정책 손실
- `vf_loss`: **가치 함수 손실 (Centralized Critic)**

## 🔧 환경 설정 변경

`train_rllib/environment.py`에서 설정 수정:

```python
ENV_CONFIG = {
    "size": 22,                     # 그리드 크기
    "num_agents": 4,                # 전체 에이전트 수
    "max_steps": 300,               # 최대 스텝

    # 이질적 에이전트 설정
    "num_helicopters": 2,           # 빠르고 효율적
    "num_trucks": 2,                # 중간 속도, 높은 효율
    "num_crews": 0,                 # 느리고 낮은 효율

    # 관찰 설정
    "partial_obs": False,           # False = global, True = local
    "agent_view_size": 10,          # partial_obs=True일 때만 사용

    # 보상 설정
    "reward_shaping": "individual", # "cooperative", "individual", "individual2"
}
```

## 🏗️ 코드 구조

### 새로 추가된 파일

1. **`wildfire_marllib_wrapper.py`**
   - MARLlib 환경 래퍼
   - **`state()` 메소드로 global state 제공** (CTDE의 핵심)
   - RLlib `MultiAgentEnv` 기반

2. **`train_marllib_mappo.py`**
   - MARLlib MAPPO 학습 스크립트
   - Centralized Critic 사용
   - 이질적 에이전트 정책 분리 지원

### 기존 파일과의 관계

```
train_rllib/
├── environment.py              # 환경 설정 (공통)
├── wildfire_rllib_wrapper.py   # RLlib IPPO용 (기존)
├── wildfire_marllib_wrapper.py # MARLlib MAPPO용 (신규) ⭐
├── train_rllib_mappo_new.py    # RLlib IPPO 학습 (기존)
└── train_marllib_mappo.py      # MARLlib MAPPO 학습 (신규) ⭐
```

## 🎓 CTDE 작동 원리

### 1. Actor (Policy)
- **입력**: Local observation (각 에이전트의 부분 관찰)
- **출력**: Action
- **실행**: Decentralized (각자 독립적으로 행동 선택)

### 2. Critic (Value Function)
- **입력**: **Global state (전체 환경 상태)** ⭐
- **출력**: Value (상태 가치)
- **학습**: Centralized (모든 정보 활용)

### 3. 학습 과정

```python
# Actor: 자신의 관찰만 사용
action = policy(local_observation)

# Critic: 전체 상태 사용 (CTDE의 핵심!)
value = critic(global_state)  # wildfire_marllib_wrapper.state() 메소드

# Advantage 계산
advantage = reward + gamma * value_next - value

# 정책 업데이트
policy_loss = -advantage * log_prob(action)
```

## 📈 성능 비교

### Independent PPO (IPPO)
```python
# 각 에이전트의 Critic
value_agent0 = critic0(obs_agent0)  # 에이전트 0의 관찰만
value_agent1 = critic1(obs_agent1)  # 에이전트 1의 관찰만
# 문제: 다른 에이전트의 행동을 모름 → Credit assignment 어려움
```

### MAPPO (CTDE)
```python
# 모든 에이전트가 같은 Centralized Critic 사용
value = critic(global_state)  # 모든 에이전트 위치 + 모든 나무 상태
# 장점: 다른 에이전트의 행동을 고려 → 협력 학습 향상
```

## 🐛 문제 해결

### 1. MARLlib import 오류

```bash
# marllib-x86 환경이 활성화되어 있는지 확인
conda activate marllib-x86

# MARLlib 설치 확인
python -c "from marllib import marl; print('OK')"
```

### 2. Gymnasium vs Gym 충돌

- MARLlib은 기본 `gym`을 사용 (gymnasium 불필요)
- `wildfire_marllib_wrapper.py`는 `gym.spaces`만 사용

### 3. 메모리 부족

```bash
# Worker 수 줄이기
python train_rllib/train_marllib_mappo.py --num-workers 1

# 배치 크기 줄이기 (코드 수정 필요)
```

## 📚 추가 자료

- [MARLlib 공식 문서](https://marllib.readthedocs.io/)
- [MAPPO 논문](https://arxiv.org/abs/2103.01955)
- [RLlib 문서](https://docs.ray.io/en/latest/rllib/)

## ✅ 체크리스트

학습 시작 전:
- [ ] `conda activate marllib-x86` 실행
- [ ] `train_rllib/environment.py`에서 설정 확인
- [ ] 디스크 공간 충분한지 확인 (체크포인트 용량)
- [ ] TensorBoard 준비

학습 중:
- [ ] TensorBoard로 학습 곡선 모니터링
- [ ] `episode_reward_mean` 증가 확인
- [ ] 주기적으로 체크포인트 백업

학습 완료 후:
- [ ] 최종 체크포인트 저장 확인
- [ ] 학습 곡선 분석
- [ ] 시뮬레이션으로 정책 평가

## 🎉 요약

**MARLlib MAPPO를 사용하면:**
- ✅ Centralized Critic으로 더 강력한 협력 학습
- ✅ Global state를 활용한 정확한 가치 평가
- ✅ Credit assignment 문제 완화
- ✅ 더 안정적인 학습 곡선

**기존 RLlib IPPO와 비교:**
- 기존: 각 에이전트가 독립적으로 학습 (제한적 협력)
- 신규: 중앙화된 critic으로 협력 학습 강화 (진정한 MAPPO)
