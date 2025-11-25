# RLlib API 버전 정보

## ✅ 이 프로젝트는 최신 RLlib API를 사용합니다

**Ray RLlib 2.0+** (2022년 이후) API를 사용하고 있습니다.

## 🔄 API 변경 사항 (구버전 → 신버전)

### 1. Import 경로 변경

#### ❌ 구버전 (Ray < 2.0)
```python
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG
```

#### ✅ 신버전 (Ray >= 2.0) - 우리가 사용하는 버전
```python
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
```

**변경 이유:** `agents` 패키지가 `algorithms`로 이름 변경됨

### 2. Trainer → Algorithm 이름 변경

#### ❌ 구버전
```python
trainer = PPOTrainer(config=config, env="my-env")
result = trainer.train()
trainer.save(checkpoint_dir)
```

#### ✅ 신버전 - 우리가 사용하는 방식
```python
algo = PPO(config=config, env="my-env")
result = algo.train()
algo.save(checkpoint_dir)
```

또는 Config 객체 사용 (권장):
```python
config = PPOConfig()
config = config.environment(env="my-env")
algo = config.build()
```

### 3. Config 설정 방식 변경

#### ❌ 구버전
```python
from ray.rllib.agents.ppo import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["num_workers"] = 4
config["lr"] = 3e-4
```

#### ✅ 신버전 - 우리가 사용하는 방식
```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .rollouts(num_rollout_workers=4)
    .training(lr=3e-4)
)
```

### 4. 체크포인트 로드 방식 변경

#### ❌ 구버전
```python
from ray.rllib.agents.ppo import PPOTrainer

trainer = PPOTrainer(config=config)
trainer.restore(checkpoint_path)
```

#### ✅ 신버전 - 우리가 사용하는 방식
```python
from ray.rllib.algorithms.ppo import PPO

algo = PPO.from_checkpoint(checkpoint_path)
```

## 📦 우리 프로젝트의 파일들

모든 파일이 **신버전 API**를 사용합니다:

### train_multiagent_rllib.py
```python
from ray.rllib.algorithms.ppo import PPOConfig  # ✅ 신버전

config = PPOConfig()  # ✅ 신버전
algo = config.build()  # ✅ 신버전
```

### simulate_trained_model.py
```python
from ray.rllib.algorithms.ppo import PPO  # ✅ 신버전

algo = PPO.from_checkpoint(checkpoint_path)  # ✅ 신버전
```

### wildfire_rllib_wrapper.py
```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv  # ✅ 신버전
```

## 🔍 버전 확인 방법

### 설치된 Ray 버전 확인
```bash
python -c "import ray; print(f'Ray version: {ray.__version__}')"
```

**필요한 버전:** Ray >= 2.0.0

### 올바른 import 테스트
```python
# 이 코드가 작동하면 올바른 버전입니다
from ray.rllib.algorithms.ppo import PPO, PPOConfig
print("✓ 올바른 RLlib 버전")
```

## 📚 마이그레이션 가이드

### 구버전 코드를 발견했다면

**1. Import 경로 수정**
```python
# 찾기
from ray.rllib.agents

# 바꾸기
from ray.rllib.algorithms
```

**2. Trainer → Algorithm**
```python
# 찾기
PPOTrainer
DQNTrainer
A3CTrainer

# 바꾸기
PPO
DQN
A3C
```

**3. Config 방식 변경**
```python
# 구버전
from ray.rllib.agents.ppo import DEFAULT_CONFIG
config = DEFAULT_CONFIG.copy()
config["lr"] = 3e-4

# 신버전
from ray.rllib.algorithms.ppo import PPOConfig
config = PPOConfig().training(lr=3e-4)
```

**4. restore() → from_checkpoint()**
```python
# 구버전
trainer.restore(path)

# 신버전
algo = PPO.from_checkpoint(path)
```

## 🚨 일반적인 오류 메시지

### 오류 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'ray.rllib.agents'
```

**원인:** Ray 2.0+ 버전에서는 `agents`가 `algorithms`로 변경됨

**해결:**
```python
# 변경 전
from ray.rllib.agents.ppo import PPOTrainer

# 변경 후
from ray.rllib.algorithms.ppo import PPO
```

### 오류 2: AttributeError
```
AttributeError: module 'ray.rllib.algorithms.ppo' has no attribute 'PPOTrainer'
```

**원인:** `PPOTrainer`가 `PPO`로 이름 변경됨

**해결:**
```python
# 변경 전
trainer = PPOTrainer(config=config)

# 변경 후
algo = PPO(config=config)
```

### 오류 3: DeprecationWarning
```
DeprecationWarning: DEFAULT_CONFIG is deprecated
```

**원인:** Config 딕셔너리 방식이 deprecated됨

**해결:**
```python
# 변경 전
config = DEFAULT_CONFIG.copy()

# 변경 후
config = PPOConfig()
```

## 📖 참고 자료

### 공식 문서
- [RLlib Migration Guide](https://docs.ray.io/en/latest/rllib/rllib-migration.html)
- [RLlib Algorithms API](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html)
- [PPOConfig Documentation](https://docs.ray.io/en/latest/rllib/package_ref/algorithm.html#ppo-config)

### 변경 이력
- **Ray 2.0.0** (2022년 8월): `agents` → `algorithms` 변경
- **Ray 2.3.0** (2023년): Config 클래스 방식 도입
- **Ray 2.7.0** (2023년): 구버전 API deprecated

## ✅ 체크리스트

우리 프로젝트는 다음을 모두 사용합니다:

- ✅ `from ray.rllib.algorithms.ppo import PPO`
- ✅ `from ray.rllib.algorithms.ppo import PPOConfig`
- ✅ `PPOConfig()` 방식으로 설정
- ✅ `algo = config.build()`
- ✅ `algo.train()`
- ✅ `PPO.from_checkpoint(path)`
- ✅ `from ray.rllib.env.multi_agent_env import MultiAgentEnv`

**결론: 모든 코드가 최신 API를 사용하고 있습니다!** 🎉

## 💡 추가 정보

### Ray 버전별 호환성

| Ray 버전 | API | 우리 프로젝트 |
|----------|-----|--------------|
| < 2.0 | `agents` + Trainer | ❌ 미지원 |
| 2.0 - 2.2 | `algorithms` + Trainer | ⚠️ 부분 지원 |
| >= 2.3 | `algorithms` + Config | ✅ 완전 지원 |

### 권장 버전
```bash
pip install "ray[rllib]>=2.7.0" torch
```

현재 (2024년 기준) 최신 stable 버전 사용을 권장합니다.
