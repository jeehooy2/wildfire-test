# 다중 에이전트 학습 가이드 (RLlib)

이 가이드는 Ray RLlib을 사용하여 wildfire 환경에서 다중 에이전트를 학습시키는 방법을 설명합니다.

## 🎯 개요

**제공된 파일:**
1. `wildfire_rllib_wrapper.py` - RLlib MultiAgentEnv 래퍼
2. `train_multiagent_rllib.py` - 학습 스크립트
3. `simulate_trained_model.py` - 학습된 모델 시뮬레이션

**특징:**
- ✅ 진정한 다중 에이전트 학습 (각 에이전트가 독립적인 보상)
- ✅ Parameter sharing (모든 에이전트가 하나의 정책 공유)
- ✅ 분산 학습 지원
- ✅ 체크포인트 자동 저장
- ✅ TensorBoard 로깅

## 📦 설치

### 1단계: Ray RLlib 설치

**중요: Ray 2.0 이상 버전 필요 (최신 API 사용)**

```bash
# Ray RLlib 최신 버전 설치
pip install "ray[rllib]>=2.7.0" torch
```

**선택사항:** TensorBoard (학습 모니터링)
```bash
pip install tensorboard
```

### 2단계: 설치 확인

```bash
python -c "import ray; from ray.rllib.algorithms.ppo import PPO; print(f'✓ RLlib 설치 완료 (Ray v{ray.__version__})')"
```

### ⚠️ API 버전 주의사항

이 프로젝트는 **최신 RLlib API (Ray 2.0+)** 를 사용합니다:

```python
# ✅ 올바른 방법 (신버전 - 우리가 사용)
from ray.rllib.algorithms.ppo import PPO, PPOConfig

# ❌ 구버전 (작동하지 않음)
from ray.rllib.agents.ppo import PPOTrainer
```

**구버전 코드를 발견했다면:** `RLLIB_API_VERSION.md` 참조

## 🚀 빠른 시작

### 1. 학습 실행

**기본 설정 (100 iterations):**
```bash
python train_multiagent_rllib.py
```

**커스텀 설정:**
```bash
python train_multiagent_rllib.py --iterations 200 --checkpoint-freq 20 --save-dir ./my_models
```

**학습 중 출력:**
```
===========================================================
RLlib 다중 에이전트 PPO 학습
===========================================================
에이전트 수: 2
학습 반복 횟수: 100
체크포인트 저장: ./rllib_checkpoints
===========================================================

학습 시작...

Iteration 1/100:
  평균 보상: -8.45
  평균 에피소드 길이: 100.0

Iteration 2/100:
  평균 보상: -7.23
  평균 에피소드 길이: 95.2

...

Iteration 10/100:
  평균 보상: -5.12
  평균 에피소드 길이: 87.3
  ✓ 체크포인트 저장: ./rllib_checkpoints/checkpoint_000010
  ⭐ 최고 성능 모델 저장: ./rllib_checkpoints/best/checkpoint_000010
```

**Ctrl+C로 언제든 중단 가능** - 진행 상황은 체크포인트에 저장됨

### 2. 학습된 모델로 시뮬레이션

**기본 시뮬레이션 (10 에피소드):**
```bash
python simulate_trained_model.py --checkpoint ./rllib_checkpoints/final
```

**GIF 저장:**
```bash
python simulate_trained_model.py --checkpoint ./rllib_checkpoints/best --save-gif --gif-filename my_trained_agent.gif
```

**무작위 정책과 비교:**
```bash
python simulate_trained_model.py --checkpoint ./rllib_checkpoints/best --episodes 20 --compare
```

**출력 예시:**
```
===========================================================
학습된 모델 시뮬레이션
===========================================================
체크포인트: ./rllib_checkpoints/best
에피소드 수: 10
GIF 저장: my_trained_agent.gif
===========================================================

모델 로드 중...
✓ 모델 로드 완료

시뮬레이션 시작...

에피소드 1:
  총 보상: -4.23
    - Agent 0: -2.15
    - Agent 1: -2.08
  스텝 수: 76

에피소드 2:
  총 보상: -3.87
    - Agent 0: -1.92
    - Agent 1: -1.95
  스텝 수: 68

...

===========================================================
시뮬레이션 결과
===========================================================
평균 총 보상: -4.12 ± 0.34
평균 에피소드 길이: 71.2 ± 5.8
최고 보상: -3.45
최저 보상: -4.89
===========================================================

GIF 생성 중: my_trained_agent.gif
✓ GIF 저장 완료: my_trained_agent.gif (76 프레임)
```

## ⚙️ 학습 설정 커스터마이징

### train_multiagent_rllib.py 수정

**환경 설정 변경:**
```python
env_config = {
    "num_agents": 4,              # 에이전트 수 증가
    "size": 25,                    # 그리드 크기 증가
    "initial_fire_size": 5,        # 초기 화재 크기
    "cooperative_reward": True,    # 협력 모드로 변경
    "max_steps": 200,              # 최대 스텝 증가
    "agent_start_positions": ((1, 1), (23, 23), (1, 23), (23, 1)),  # 4명일 경우
}
```

**PPO 하이퍼파라미터 조정:**
```python
config = (
    PPOConfig()
    .training(
        lr=1e-4,                   # 학습률 감소
        gamma=0.995,               # 할인율 증가 (장기 보상 중시)
        train_batch_size=8000,     # 배치 크기 증가
        sgd_minibatch_size=256,    # 미니배치 크기
        num_sgd_iter=20,           # SGD 반복 횟수
    )
    .rollouts(
        num_rollout_workers=4,     # 워커 수 증가 (병렬화)
    )
    .resources(
        num_gpus=1,                # GPU 사용
    )
)
```

**정책 분리 (각 에이전트가 독립적인 정책):**
```python
.multi_agent(
    policies={
        "policy_0": (...),
        "policy_1": (...),
    },
    policy_mapping_fn=lambda agent_id, **kwargs: f"policy_{agent_id}",
)
```

## 📊 학습 모니터링

### TensorBoard 사용

```bash
# 터미널 1: 학습 실행
python train_multiagent_rllib.py

# 터미널 2: TensorBoard 실행
tensorboard --logdir ~/ray_results
```

브라우저에서 `http://localhost:6006` 접속

**확인 가능한 지표:**
- Episode reward (에피소드 보상)
- Episode length (에피소드 길이)
- Policy loss (정책 손실)
- Value loss (가치 손실)
- Learning rate (학습률)

## 🔧 문제 해결

### 1. Ray 초기화 오류

```
RuntimeError: Maybe you called ray.init twice by accident?
```

**해결:**
```bash
# Ray 프로세스 종료
ray stop
```

### 2. 메모리 부족

**증상:** 학습 중 메모리 사용량이 계속 증가

**해결:**
```python
# rollout 워커 수 줄이기
.rollouts(num_rollout_workers=1)

# 배치 크기 줄이기
.training(train_batch_size=2000, sgd_minibatch_size=64)
```

### 3. 학습이 느림

**해결:**
```python
# 워커 수 증가 (CPU 코어 수에 맞춰)
.rollouts(num_rollout_workers=4)

# GPU 사용 (있다면)
.resources(num_gpus=1)
```

### 4. 체크포인트 로드 실패

```
FileNotFoundError: Checkpoint not found
```

**해결:**
```bash
# 전체 경로 사용
python simulate_trained_model.py --checkpoint /full/path/to/checkpoint_000100
```

## 📈 학습 결과 분석

### 좋은 학습 징후:
- ✅ 평균 보상이 점진적으로 증가
- ✅ 에피소드 길이가 감소 (효율적으로 불 끄기)
- ✅ 보상의 분산이 감소 (안정적인 성능)

### 나쁜 학습 징후:
- ❌ 보상이 진동하거나 발산
- ❌ 에피소드 길이가 max_steps에 계속 도달
- ❌ 무작위 정책보다 성능이 나쁨

**해결책:**
- 학습률 감소 (`lr=1e-4` 또는 `3e-5`)
- 배치 크기 증가
- 보상 함수 재설계

## 🎮 고급 사용법

### 1. 학습 재개

```python
# train_multiagent_rllib.py 수정
algo = PPO.from_checkpoint("./rllib_checkpoints/checkpoint_000050")
# 이후 algo.train() 계속 실행
```

### 2. 하이퍼파라미터 튜닝

Ray Tune 사용:
```python
from ray import tune

tune.run(
    "PPO",
    config={
        "env": "wildfire-ma",
        "lr": tune.grid_search([1e-4, 3e-4, 1e-3]),
        "gamma": tune.grid_search([0.95, 0.99, 0.995]),
    }
)
```

### 3. 다른 알고리즘 시도

```python
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.a3c import A3CConfig

# DQN
config = DQNConfig().environment(...)

# A3C
config = A3CConfig().environment(...)
```

## 📚 참고 자료

- [Ray RLlib 문서](https://docs.ray.io/en/latest/rllib/)
- [PPO 논문](https://arxiv.org/abs/1707.06347)
- [Multi-Agent RL 가이드](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-environments)

## 💡 팁

1. **먼저 짧게 학습** (`--iterations 10`)하여 모든 것이 작동하는지 확인
2. **best 체크포인트 사용** - 최고 성능 모델이 저장됨
3. **비교 옵션 사용** (`--compare`) - 학습 효과 확인
4. **GIF 저장** - 시각적으로 에이전트 행동 분석
5. **TensorBoard 모니터링** - 실시간 학습 진행 상황 확인
