# Inference/Execution Mode (Discrete Actions without Gumbel-Softmax)

## 개요

**Execution 시에는 Gumbel-Softmax를 사용하지 않습니다.** 대신 Actor network의 logits에서 단순히 **argmax**를 취하여 discrete action을 선택합니다.

```
Training Mode:  observation -> logits -> Gumbel-Softmax -> soft action (미분 가능)
Execution Mode: observation -> logits -> argmax -> discrete action (선택)
```

---

## 구현

### 1. Actor Network의 Inference Method

#### `DiscreteActorNetwork.get_discrete_action(obs)`

```python
def get_discrete_action(self, obs: torch.Tensor) -> torch.Tensor:
    """
    Observation으로부터 discrete action (argmax)을 얻습니다.

    **Inference/Execution용**: Gumbel-Softmax를 사용하지 않고,
    단순히 logits의 argmax를 취합니다.
    """
    logits = self.forward(obs)
    action = logits.argmax(dim=-1)  # <- 간단! argmax만 사용
    return action
```

**동작:**
```
observation (50,)
    ↓
Actor network forward
    ↓
logits [1.2, -0.5, 8.5, 0.1, 3.2, -1.0, ...]  (9 logits)
    ↓
argmax(logits)
    ↓
action = 2  (discrete action index)
```

### 2. New Train MADDPG의 Inference Functions

#### `get_discrete_action_from_logits(logits)`

```python
def get_discrete_action_from_logits(logits):
    """
    Logits에서 직접 discrete action을 추출합니다 (Inference용).

    Gumbel-Softmax를 사용하지 않고, 단순히 argmax만 취합니다.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()

    action = logits.argmax(dim=-1)  # <- argmax만!
    return action
```

#### `get_discrete_action_from_actor(actor_logits, hard=False, tau=1.0)`

```python
def get_discrete_action_from_actor(actor_logits, hard=False, tau=1.0):
    """
    Actor logits에서 action을 얻습니다.

    hard=True: argmax (Execution)
    hard=False: Gumbel-Softmax (Training)
    """
    if hard:
        # Execution: argmax만 사용
        action = actor_logits.argmax(dim=-1)
    else:
        # Training: Gumbel-Softmax
        action = gumbel_softmax(actor_logits, tau=tau, hard=False)

    return action
```

### 3. Multi-Agent Wrapper의 Inference Method

#### `DiscreteMADDPGWrapper.get_discrete_actions(observations)`

```python
def get_discrete_actions(self, observations: list) -> list:
    """
    Inference/Execution용 discrete actions을 얻습니다.

    Gumbel-Softmax를 사용하지 않고, logits의 argmax만 취합니다.
    """
    discrete_actions = []

    for i, obs in enumerate(observations):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        # get_discrete_action() 사용 -> argmax만!
        action = self.actors[i].get_discrete_action(obs_tensor)
        action_idx = action.squeeze(0).item()
        discrete_actions.append(action_idx)

    return discrete_actions
```

---

## 사용 예제

### 예제 1: Single Agent Inference

```python
from train_marllib_self.discrete_maddpg import DiscreteActorNetwork
import torch

# Actor 네트워크
actor = DiscreteActorNetwork(obs_dim=50, num_actions=9)

# Observation
observation = torch.randn(1, 50)

# ========== Execution: Discrete action (argmax) ==========
discrete_action = actor.get_discrete_action(observation)
print(f"Discrete action: {discrete_action.item()}")  # 정수 (0-8)
# Output: Discrete action: 3
```

### 예제 2: Training vs Inference

```python
import torch
from train_marllib_self.discrete_maddpg import DiscreteActorNetwork

actor = DiscreteActorNetwork(obs_dim=50, num_actions=9)
optimizer = torch.optim.Adam(actor.parameters())

obs = torch.randn(4, 50)

# ========== TRAINING ==========
# 미분 가능한 soft action으로 학습
action_soft, log_prob = actor.get_action_and_log_prob(obs, tau=1.0, hard=False)
# action_soft: shape (4, 9), soft probabilities
# log_prob: shape (4,), gradient 가능

loss = -log_prob.mean()  # Policy gradient
loss.backward()  # Gradient 계산
optimizer.step()  # Network 업데이트

# ========== INFERENCE ==========
# 단순 argmax로 discrete action 선택
discrete_actions = actor.get_discrete_action(obs)
# discrete_actions: shape (4,), 정수 action indices
# Gradient 계산 필요 없음!
```

### 예제 3: Multi-Agent Execution

```python
from train_marllib_self.discrete_maddpg import DiscreteMADDPGWrapper
import numpy as np

# Multi-agent wrapper
wrapper = DiscreteMADDPGWrapper(
    num_agents=3,
    obs_dim=50,
    num_actions=9
)

# Observations (각 에이전트마다 하나)
observations = [
    np.random.randn(50),  # Agent 0
    np.random.randn(50),  # Agent 1
    np.random.randn(50),  # Agent 2
]

# ========== TRAINING: Soft actions ==========
# Gumbel-Softmax로 미분 가능한 action
actions_soft, log_probs = wrapper.get_actions(observations, hard=False)
# actions_soft: [array(9,), array(9,), array(9,)]  <- soft probabilities
# log_probs: [-1.5, -1.6, -1.4]

# ========== INFERENCE: Discrete actions ==========
# Argmax로 discrete action 선택
discrete_actions = wrapper.get_discrete_actions(observations)
# discrete_actions: [3, 7, 2]  <- discrete indices!
```

### 예제 4: Environment에 Action 전달

```python
import numpy as np
import torch
from train_marllib_self.discrete_maddpg import DiscreteActorNetwork

# Environment 설정
env = setup_environment()  # 예시
actor = DiscreteActorNetwork(obs_dim=50, num_actions=9)

# Episode 실행
for step in range(episode_length):
    # Observation 획득
    observations = env.get_observations()  # list of arrays

    # ========== Execution: discrete actions ==========
    discrete_actions = []
    for i, obs in enumerate(observations):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = actor.get_discrete_action(obs_tensor)
        discrete_actions.append(action.item())

    # Environment에 action 전달
    next_obs, rewards, done = env.step(discrete_actions)
    # discrete_actions: [0, 1, 2, ...] <- 정수!
```

---

## 핵심 차이점

### Training Mode vs Execution Mode

| 항목 | Training | Execution |
|------|----------|-----------|
| **사용 함수** | `get_action_and_log_prob()` | `get_discrete_action()` |
| **샘플링** | Gumbel-Softmax | Argmax |
| **출력 형태** | Soft probabilities (9,) | Discrete index (scalar) |
| **미분 가능** | ✓ Yes | ✗ No |
| **사용 시점** | Batch training | Evaluation/Deployment |
| **예시 출력** | [0.1, 0.05, 0.7, ...] | 2 |

### 구체적인 예

```
Actor output logits: [1.2, -0.5, 8.5, 0.1, 3.2, -1.0]

Training:
  -> Gumbel-Softmax(logits, tau=1.0)
  -> [0.05, 0.01, 0.80, 0.03, 0.10, 0.01] (soft)
  -> Critic으로 Q-value 계산
  -> Gradient 전파

Execution:
  -> argmax(logits)
  -> action = 2 (discrete)
  -> Environment에 직접 전달
  -> 환경 상호작용
```

---

## 주요 특징

### ✅ Inference의 장점

1. **간단함**: Argmax만 사용, 복잡한 계산 없음
2. **빠름**: Gumbel noise 샘플링 필요 없음
3. **결정론적**: 같은 observation -> 같은 action
4. **메모리 효율**: One-hot 변환 없이 정수만 전달 가능

### 🔄 Training/Execution의 조화

```python
# Training
action_soft = gumbel_softmax(logits, tau=1.0)  # 미분 가능
q_value = critic(obs, action_soft)              # Gradient 전파
loss = -q_value.mean()
loss.backward()  # Network 학습

# Execution
action_discrete = argmax(logits)                # 선택
env.step(action_discrete)                       # 환경 상호작용
```

---

## 구현 체크리스트

✅ **구현됨**:
- [ ] `DiscreteActorNetwork.get_discrete_action()` - argmax만 사용
- [ ] `get_discrete_action_from_logits()` - 유틸리티 함수
- [ ] `get_discrete_action_from_actor()` - training/inference 선택
- [ ] `DiscreteMADDPGWrapper.get_discrete_actions()` - 멀티에이전트
- [ ] 테스트: `test_inference_execution()` - 정확성 검증
- [ ] 테스트: `test_multi_agent_inference()` - 멀티에이전트 검증

---

## 테스트 결과

```
✅ Test 6: Inference/Execution (Argmax only, No Gumbel-Softmax)
  - Single observation inference
  - Batch inference (5 samples)
  - Action validity check (0-8)

✅ Test 7: Multi-Agent Inference (Discrete Actions)
  - Training mode: soft actions with Gumbel-Softmax
  - Inference mode: discrete actions with argmax
  - Multi-agent action consistency
```

---

## 올바른 사용 패턴

### ✅ 올바른 방법

```python
# Training
logits = actor(obs)
action_soft = gumbel_softmax(logits, tau=tau)  # Gumbel-Softmax 사용
q = critic(obs, action_soft)
loss = -q.mean()
loss.backward()

# Inference
logits = actor(obs)
action = logits.argmax(dim=-1)  # Argmax만 사용
env.step(action)
```

### ❌ 잘못된 방법

```python
# ❌ Inference에서 Gumbel-Softmax 사용
logits = actor(obs)
action = gumbel_softmax(logits)  # 불필요!
env.step(action)  # One-hot을 정수로 변환? 복잡함

# ❌ Inference 후 gradient 계산
action = logits.argmax()
loss = -critic(obs, action).mean()
loss.backward()  # action이 미분 불가능! 에러!
```

---

## 실전 팁

### 1. 환경에 Action 전달

```python
# 만약 환경이 정수 action을 기대한다면:
discrete_action = actor.get_discrete_action(obs)
obs, reward, done = env.step(discrete_action.item())  # 정수로 변환

# 만약 환경이 one-hot을 기대한다면:
discrete_action = actor.get_discrete_action(obs)
one_hot_action = F.one_hot(discrete_action, num_classes=9)
obs, reward, done = env.step(one_hot_action)
```

### 2. Batch Evaluation

```python
# 여러 관찰에 대해 동시에 inference
observations = torch.randn(32, 50)  # batch_size=32
discrete_actions = actor.get_discrete_action(observations)
# discrete_actions: (32,) shape 정수 배열
```

### 3. 저장 및 로드

```python
# 모델 저장 (학습 후)
torch.save(actor.state_dict(), 'actor.pth')

# 모델 로드 (inference 시)
actor = DiscreteActorNetwork(obs_dim=50, num_actions=9)
actor.load_state_dict(torch.load('actor.pth'))
actor.eval()  # Evaluation mode

# Inference
with torch.no_grad():  # Gradient 계산 불필요
    action = actor.get_discrete_action(obs)
```

---

## 마지막 정리

**Execution 시에는:**
1. ✅ Argmax만 사용 (Gumbel-Softmax 불필요)
2. ✅ 정수 action 출력
3. ✅ 빠르고 결정론적
4. ✅ 환경에 직접 전달 가능

**이것이 모든 요구사항을 만족합니다!**

```
observation -> logits -> argmax -> discrete action ✓
```

---

마지막 수정: 2025-11-18
