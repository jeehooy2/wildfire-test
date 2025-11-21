# Discrete Action Space를 위한 MADDPG + Gumbel-Softmax 구현 가이드

## 개요

이 문서는 **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** 알고리즘을 **discrete action space**에 적용하는 방법을 설명합니다. 원래 MADDPG는 continuous action space만 지원하지만, **Gumbel-Softmax reparameterization trick**을 사용하여 discrete action을 미분 가능하게 처리할 수 있습니다.

---

## 핵심 개념

### 1. Gumbel-Softmax Reparameterization

**문제**: Discrete action 샘플링은 미분 불가능해서 정책 경사도(policy gradient)를 계산할 수 없습니다.

**해결책**: Gumbel-Softmax는 discrete distribution을 continuous relaxation으로 근사합니다.

#### 수식

```
g_i = -log(-log(U_i))              # Gumbel noise 생성 (U_i ~ Uniform(0,1))
s_i = softmax((logits_i + g_i) / τ)  # Soft sample (미분 가능)
y_i = hard_one_hot(argmax(s_i))    # Hard sample (discrete action)
```

#### 핵심 파라미터

- **τ (tau)**: Temperature parameter
  - τ >> 1: Uniform distribution에 가까움 (높은 탐색)
  - τ = 1: 기본값
  - τ << 1: One-hot에 가까움 (낮은 탐색, 높은 활용)

- **Hard vs Soft**
  - **Soft**: Training에 사용 (미분 가능, 확률 분포)
  - **Hard**: Inference/Evaluation에 사용 (discrete action 선택)

---

## 구현 구조

### 2. Actor Network (정책)

**입력**: Agent의 관찰 (observation)
**출력**: Logits 벡터 (각 discrete action마다 하나)

```python
# 예시: 9개 discrete actions
observations = [1.2, 0.5, -0.3, ...]  # obs_dim
actor_output = [0.8, -1.2, 2.1, 0.0, 1.5, -0.9, 0.2, 1.8, 0.4]  # 9 logits
```

**구현**:
```python
actor = DiscreteActorNetwork(
    obs_dim=50,
    num_actions=9,
    hidden_dims=[256, 256]
)

# Forward pass: observation -> logits
logits = actor(observation)  # Shape: (batch_size, 9)

# Gumbel-Softmax sampling
action_soft = actor.sample_action(observation, tau=1.0, hard=False)
action_hard = actor.sample_action(observation, tau=0.1, hard=True)
```

### 3. Critic Network (Q-function)

**입력**: 모든 에이전트의 observations + actions (one-hot encoded)
**출력**: Q-value (스칼라)

```python
critic = DiscreteCriticNetwork(
    obs_dim_list=[50, 50, 50],      # 3개 에이전트
    action_dim_list=[9, 9, 9],      # 각 9개 actions
    hidden_dims=[256, 256]
)

# Forward pass: [all_obs, all_actions] -> Q-value
obs_all = torch.cat([obs1, obs2, obs3], dim=1)  # (batch, 150)
actions_all = torch.cat([a1_onehot, a2_onehot, a3_onehot], dim=1)  # (batch, 27)
q_value = critic(obs_all, actions_all)  # (batch, 1)
```

### 4. Temperature Annealing

학습 진행에 따라 temperature를 천천히 감소시킵니다.

```python
scheduler = TemperatureScheduler(
    initial_tau=1.0,      # 초반: 높은 탐색
    final_tau=0.1,        # 후반: 높은 활용
    total_steps=1000000,
    anneal_type="linear"  # or "exponential", "cosine"
)

# 각 step마다
tau = scheduler.step()  # tau가 점진적으로 감소
```

---

## 사용 방법

### 기본 사용법

```python
from train_marllib_self.discrete_maddpg import (
    GumbelSoftmax,
    DiscreteActorNetwork,
    DiscreteCriticNetwork,
    TemperatureScheduler
)

# 1. Actor 생성
actor = DiscreteActorNetwork(
    obs_dim=50,
    num_actions=9,
    hidden_dims=[256, 256]
)

# 2. Action 샘플링
observation = torch.randn(32, 50)  # batch_size=32

# Training: soft action (미분 가능)
action_soft, log_prob = actor.get_action_and_log_prob(
    observation, tau=1.0, hard=False
)

# Inference: hard action (discrete)
action_hard = actor.sample_action(
    observation, tau=0.1, hard=True
)
```

### 멀티에이전트 wrapper 사용

```python
from train_marllib_self.discrete_maddpg import DiscreteMADDPGWrapper

# Wrapper 생성
wrapper = DiscreteMADDPGWrapper(
    num_agents=3,
    obs_dim=50,
    num_actions=9,
    use_temperature_annealing=True
)

# Action 샘플링
observations = [obs1, obs2, obs3]
actions, log_probs = wrapper.get_actions(observations, hard=False)

# Temperature scheduler 진행
wrapper.step()
```

---

## MADDPG 학습 루프

### 1. **Experience Collection** (온폴리시)

```python
# 각 에피소드:
for step in range(episode_length):
    # 모든 에이전트의 action 샘플링 (soft)
    logits = [actor_i(obs_i) for obs_i in observations]
    actions = [gumbel_softmax(logits_i, tau=tau, hard=False) for logits_i in logits]

    # 환경에서 step 실행
    next_obs, rewards, dones = env.step(actions)

    # Experience 저장 (replay buffer)
    # [obs, action, reward, next_obs, done]
    replay_buffer.add(obs, actions, rewards, next_obs, dones)
```

### 2. **Critic 업데이트**

```python
# Replay buffer에서 배치 샘플
batch = replay_buffer.sample()
obs, actions, rewards, next_obs, dones = batch

# Target Q-value 계산 (target network)
with torch.no_grad():
    # Next actions: hard sample 사용
    next_actions = [
        gumbel_softmax(actor_target_i(next_obs_i), tau=tau_final, hard=True)
        for obs_i in next_obs
    ]
    next_actions_onehot = [to_onehot(a) for a in next_actions]
    next_actions_concat = torch.cat(next_actions_onehot, dim=1)

    target_q = critic_target(concat_obs(next_obs), next_actions_concat)
    y = rewards + gamma * target_q * (1 - dones)

# Q-function 업데이트
q_pred = critic(concat_obs(obs), concat_actions(actions))
critic_loss = F.mse_loss(q_pred, y)
critic_loss.backward()
critic_optimizer.step()
```

### 3. **Actor 업데이트**

```python
# Critic으로부터 policy gradient 얻기
# Actions는 soft sample (미분 가능)
actions_soft = [
    gumbel_softmax(actor_i(obs_i), tau=tau, hard=False)
    for obs_i in observations
]
actions_soft_onehot = [to_onehot(a) for a in actions_soft]
actions_soft_concat = torch.cat(actions_soft_onehot, dim=1)

# Policy gradient (Critic은 detach)
q_value = critic(concat_obs(obs), actions_soft_concat)
actor_loss = -q_value.mean()  # Maximize Q-value

actor_loss.backward()
actor_optimizer.step()
```

### 4. **Soft Target Update**

```python
# Target network를 천천히 업데이트
with torch.no_grad():
    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
        target_param.data.copy_(tau_soft * param.data + (1 - tau_soft) * target_param.data)

    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(tau_soft * param.data + (1 - tau_soft) * target_param.data)
```

---

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|------|------|
| `gumbel_tau_initial` | 1.0 | 초기 Gumbel-Softmax temperature |
| `gumbel_tau_final` | 0.1 | 최종 Gumbel-Softmax temperature |
| `tau_soft` | 0.002 | Soft target update 계수 |
| `actor_lr` | 0.0005 | Actor 학습률 |
| `critic_lr` | 0.0005 | Critic 학습률 |
| `batch_size` | 32 | Replay buffer 배치 크기 |
| `hidden_dims` | [256, 256] | 신경망 숨겨진 층 크기 |
| `anneal_type` | "linear" | Temperature annealing 방식 |

---

## 주요 특징

### ✅ 장점

1. **미분 가능한 Discrete Action 샘플링**
   - Gradient를 통한 직접적인 정책 최적화 가능

2. **Temperature Annealing**
   - 초반 탐색 -> 후반 활용으로 자동 전환

3. **Straight-through Estimator**
   - Forward: One-hot (discrete action)
   - Backward: Soft probabilities (gradient)

4. **멀티에이전트 지원**
   - 중앙화된 Critic으로 다른 에이전트 정보 활용

### ⚠️ 주의사항

1. **Temperature 감소 중요**
   - τ를 너무 빨리 감소시키면 탐색 부족
   - 너무 천천히 감소시키면 수렴 지연

2. **One-hot Encoding**
   - Discrete action을 continuous representation으로 변환 필요
   - Action dimension이 크면 메모리 증가

3. **Log Probability 계산**
   - Soft sample의 log prob 계산에 주의
   - Hard sample은 argmax 기반으로 계산

---

## 코드 예제

### new_train_maddpg.py의 주요 함수

```python
# 1. Gumbel-Softmax 샘플링
def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-20):
    """Gumbel-Softmax reparameterization"""
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + eps) + eps)
    gumbel_logits = (logits + gumbel_noise) / tau
    y_soft = F.softmax(gumbel_logits, dim=-1)

    if hard:
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y

# 2. Discrete to continuous action 변환
def convert_discrete_to_continuous_action(discrete_action, num_actions):
    """One-hot encoding"""
    if isinstance(discrete_action, int):
        action = torch.zeros(num_actions)
        action[discrete_action] = 1.0
    else:
        action = F.one_hot(discrete_action, num_classes=num_actions).float()
    return action
```

---

## 실행 방법

### 기본 실행

```bash
cd /home/bmkim88/wildfire_environment

# MADDPG 학습 (Gumbel-Softmax 지원)
python train_marllib_self/new_train_maddpg.py --run-name discrete_maddpg_v1

# discrete_maddpg 모듈 테스트
python -m train_marllib_self.discrete_maddpg
```

### MARLlib과의 통합

```python
from marllib import marl
from train_marllib_self.new_wrapper import WildfireRLlibEnv
from train_marllib_self.discrete_maddpg import DiscreteMADDPGWrapper

# 환경과 알고리즘 설정
env = marl.make_env(environment_name="wildfire-ma", map_name="wildfire-ma")
maddpg = marl.algos.maddpg(hyperparam_source="common")
model = marl.build_model(env, maddpg, {"core_arch": "mlp", "encode_layer": "256-256"})

# Discrete action wrapper 통합
discrete_wrapper = DiscreteMADDPGWrapper(
    num_agents=env.num_agents,
    obs_dim=env.observation_space.shape[0],
    num_actions=env.action_space.n
)

# 학습
maddpg.fit(env, model, ...)
```

---

## 참고 논문

1. **Jang et al.** (2017). "Categorical Reparameterization with Gumbel-Softmax"
   ICLR 2017. [arXiv](https://arxiv.org/abs/1611.01144)

2. **Maddison et al.** (2017). "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables"
   ICLR 2017. [arXiv](https://arxiv.org/abs/1611.00712)

3. **Lowe et al.** (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
   NIPS 2017. [arXiv](https://arxiv.org/abs/1706.02275)

4. **Lillicrap et al.** (2016). "Continuous control with deep reinforcement learning"
   ICLR 2016 (DDPG). [arXiv](https://arxiv.org/abs/1509.02971)

---

## 트러블슈팅

### Q1: Loss가 수렴하지 않음

**원인**: Temperature가 너무 빨리 감소
**해결책**: `gumbel_tau_final`을 높이거나 `total_steps`를 늘리기

```python
scheduler = TemperatureScheduler(
    initial_tau=1.0,
    final_tau=0.3,      # 0.1 대신 0.3으로 변경
    total_steps=2000000 # 더 오래 학습
)
```

### Q2: Action이 항상 같은 것만 선택됨

**원인**: Temperature가 너무 낮거나 Actor가 수렴
**해결책**:
- Temperature annealing schedule 재조정
- Learning rate 변경
- Network 크기 조정

### Q3: Critic loss가 음수가 됨

**정상**: MSE loss가 항상 양수여야 함
**확인**: Q-value 계산에서 target 계산이 올바른지 검증

---

## 파일 구조

```
train_marllib_self/
├── new_train_maddpg.py           # Main training script (Gumbel-Softmax 통합)
├── discrete_maddpg.py            # Discrete MADDPG 모듈
├── new_wrapper.py                # Wildfire 환경 래퍼
├── environment.py                # 환경 설정
└── experiments/                  # 학습 결과 저장
    └── maddpg/
        └── {run_name}/
            ├── checkpoint_*
            ├── progress.csv
            └── logs/
```

---

## 라이선스 및 참고

이 구현은 MARLlib 프레임워크를 기반으로 하며, ICLR 2017 Gumbel-Softmax 논문의 방법론을 따릅니다.

---

마지막 수정: 2025-11-18
