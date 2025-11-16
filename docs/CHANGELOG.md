# Changelog - MAPPO Training Integration with MARLlib

## 날짜: 2025-11-15

### 요약
wildfire-v0 환경을 MARLlib 프레임워크와 통합하여 MAPPO 알고리즘으로 학습 가능하도록 수정

---

## 0. 환경 설정 변경
```
wheel==0.38.0
protobuf==3.20.3
ray==1.8.0
ray[tune]==1.8.0
ray[rllib]==1.8.0
icecream==2.1.3
torch==1.9.0
pettingzoo==1.12.0
pettingzoo[mpe]==1.12.0
supersuit==3.2.0
numpy==1.21.0
importlib-metadata==4.13.0
gym==0.22.0
PyYAML
matplotlib
```

## 1. 프로젝트 파일 수정

### 1.1 `train_marllib_self/new_train_mappo.py`
**목적**: MARLlib을 사용한 MAPPO 학습 메인 스크립트

**주요 변경사항**:
- MARLlib 환경 등록:
  ```python
  ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnv
  COOP_ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnv
  ```

- 자동 YAML 설정 파일 생성 (`setup_marllib_config()` 함수):
  - MARLlib 패키지 내부에 `wildfire-ma.yaml` 자동 생성
  - 환경 설정을 ENV_CONFIG에서 읽어서 YAML 형식으로 변환

- MARLlib config 수정 (학습 시작 전):
  ```python
  if mappo.config_dict is not None:
      mappo.config_dict["opp_action_in_cc"] = False
      mappo.config_dict["global_state_flag"] = False
  ```
  **이유**: MARLlib이 요구하는 필수 설정값을 명시적으로 추가

### 1.2 `train_marllib_self/new_wrapper.py`
**목적**: wildfire-v0를 RLlib MultiAgentEnv 형식으로 래핑

**주요 변경사항**:

1. **환경 생성 방식 변경** (가장 중요):
   ```python
   # 이전: gym.make() 사용
   # self.env = gym.make("wildfire-v0", **clean_config)

   # 이후: 직접 생성
   from wildfire_environment.envs import WildfireEnv
   self.env = WildfireEnv(**clean_config)
   ```
   **이유**:
   - `gym.make()`는 자동으로 `OrderEnforcing` 등의 래퍼를 추가
   - 이 래퍼들은 구 Gym API(4-tuple)를 기대하지만 wildfire는 신 API(5-tuple) 사용
   - 직접 생성하여 래퍼 충돌 회피

2. **Config 전처리 강화**:
   ```python
   # map_name 제거 (wildfire env가 받지 않는 파라미터)
   clean_config = {k: v for k, v in env_config.items() if k != 'map_name'}

   # YAML에서 문자열로 저장된 튜플을 실제 튜플로 변환
   if isinstance(clean_config['agent_start_positions'], str):
       import ast
       clean_config['agent_start_positions'] = ast.literal_eval(...)

   # 문자열 "None"을 실제 None으로 변환
   if clean_config['reward_shaping_config'] == 'None':
       clean_config['reward_shaping_config'] = None
   ```

3. **Observation Space 구조 수정**:
   ```python
   # MARLlib은 Dict({"obs": Box(...)}) 형식 요구
   from gym.spaces import Dict as GymDict
   self._observation_space = GymDict({"obs": single_obs_space})
   ```

4. **Action Space 구조 수정**:
   ```python
   # Dict가 아닌 단일 Discrete space로 설정
   self._action_space = single_action_space  # Discrete(9)
   ```

5. **reset() 메서드 수정**:
   ```python
   # Gym 환경은 (obs, info) 튜플 반환 가능
   result = self.env.reset(seed=seed)
   if isinstance(result, tuple):
       obs_dict, info = result
   else:
       obs_dict = result

   # MARLlib 형식으로 래핑
   obs = {int(k): {"obs": v} for k, v in obs_dict.items()}
   ```

6. **step() 메서드 수정**:
   ```python
   # 키 변환: int → str (wildfire env용)
   env_actions = {str(i): action_dict[i] for i in range(self.num_agents)}

   # 환경 실행
   obs_dict, reward_dict, terminated, truncated, infos_dict = self.env.step(env_actions)

   # RLlib 형식으로 변환: str → int, MARLlib obs 래핑
   observations = {int(k): {"obs": v} for k, v in obs_dict.items()}
   rewards = {int(k): v for k, v in reward_dict.items()}
   ```

7. **policy_mapping_dict 추가**:
   ```python
   policy_mapping_dict = {
       "wildfire-ma": {
           "description": "cooperative wildfire suppression",
           "team_prefix": ("agent_",),
           "all_agents_one_policy": True,
           "one_agent_one_policy": True,
       }
   }
   ```

8. **get_env_info() 메서드 추가**:
   ```python
   def get_env_info(self):
       """MARLlib이 요구하는 환경 정보 반환"""
       env_info = {
           "space_obs": self.observation_space,
           "space_act": self.action_space,
           "num_agents": self.num_agents,
           "episode_limit": self.env.max_steps,
           "policy_mapping_info": policy_mapping_dict
       }
       return env_info
   ```

### 1.3 `train_marllib_self/environment.py`
**목적**: 환경 설정 파일

**주요 변경사항**:
```python
# agent_start_positions 수정
# 이전: 모든 에이전트가 (15, 1)에서 시작 → 초기화 무한 대기 발생
"agent_start_positions": ((1, 1), (1, 20), (20, 1), (20, 20)),
```
**이유**: 에이전트들이 격자 모서리에 분산 배치되어 초기화 충돌 방지

### 1.4 `wildfire_environment/utils/window.py`
**목적**: matplotlib import 실패 시 프로세스 종료 방지

**주요 변경사항**:
```python
# 이전: matplotlib import 실패 시 sys.exit(-1)
# 이후: 선택적 import로 변경
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed...")
```
**이유**: 학습 시 렌더링이 필요없으므로 matplotlib이 없어도 작동 가능하도록 수정

### 1.5 `wildfire_environment/envs/wildfire.py`
**임시 변경사항**: 디버그 출력 추가
```python
print("[DEBUG] WildfireEnv.__init__ 시작")
print("[DEBUG] 기본 파라미터 설정 완료")
# ... 등등
```
**TODO**: 학습이 안정적으로 작동하면 디버그 출력 제거 예정

---

## 2. MARLlib 라이브러리 소스 코드 수정

### ⚠️ 중요: 외부 라이브러리 수정
MARLlib 라이브러리의 소스 코드를 직접 수정했습니다. 다른 컴퓨터에서 사용 시 동일한 수정이 필요합니다.

### 2.1 `cc_mlp.py` (Centralized Critic MLP 모델)
**파일 경로**: `.venv/lib/python3.8/site-packages/marllib/marl/models/zoo/mlp/cc_mlp.py`

**수정 위치**: Line 57

**변경사항**:
```python
# 이전:
if self.custom_config["opp_action_in_cc"]:

# 이후:
if self.custom_config.get("opp_action_in_cc", False):
```

**이유**:
- `custom_config`에 `opp_action_in_cc` 키가 없으면 KeyError 발생
- `.get()` 메서드로 기본값 False 제공하여 에러 방지

### 2.2 `centralized_critic.py` (Centralized Critic Postprocessing)
**파일 경로**: `.venv/lib/python3.8/site-packages/marllib/marl/algos/utils/centralized_critic.py`

**수정 위치**: Lines 55-57

**변경사항**:
```python
# 이전:
opp_action_in_cc = custom_config["opp_action_in_cc"]
global_state_flag = custom_config["global_state_flag"]
mask_flag = custom_config["mask_flag"]

# 이후:
opp_action_in_cc = custom_config.get("opp_action_in_cc", False)
global_state_flag = custom_config.get("global_state_flag", False)
mask_flag = custom_config.get("mask_flag", False)
```

**이유**:
- config 키가 없을 때 발생하는 KeyError 방지
- 각 설정의 기본값을 False로 지정

### 2.3 `mappo.py` (MAPPO Loss Function)
**파일 경로**: `.venv/lib/python3.8/site-packages/marllib/marl/algos/core/CC/mappo.py`

**수정 위치**: Line 58

**변경사항**:
```python
# 이전:
opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]

# 이후:
opp_action_in_cc = policy.config["model"]["custom_model_config"].get("opp_action_in_cc", False)
```

**이유**:
- Policy 초기화 시 config에 키가 없으면 KeyError 발생
- `.get()` 메서드로 안전하게 접근

---

## 3. 자동 생성 파일

### 3.1 `wildfire-ma.yaml`
**파일 경로**: `.venv/lib/python3.8/site-packages/examples/config/env_config/wildfire-ma.yaml`

**생성 방법**: `new_train_mappo.py`의 `setup_marllib_config()` 함수가 자동 생성

**내용**:
```yaml
env: wildfire-ma

env_args:
  size: 22
  num_agents: 4
  max_steps: 300
  initial_fire_size: 32
  num_helicopters: 2
  num_trucks: 2
  num_crews: 0
  agent_start_positions: ((1, 1), (1, 20), (20, 1), (20, 20))
  alpha: 0.05
  beta: 0.2
  delta_beta: 0.1
  partial_obs: False
  agent_view_size: 7
  cooperative_reward: False
  selfishness_weight: 0.0
  reward_shaping: individual
  reward_shaping_config: None
  render_mode: None

mask_flag: False
global_state_flag: False
```

---

## 4. 해결된 주요 에러들

### 4.1 KeyError: 'opp_action_in_cc'
- **원인**: MARLlib 소스 코드에서 config dict 키에 직접 접근
- **해결**: MARLlib 소스 3개 파일 수정 (위 2.1, 2.2, 2.3 참조)

### 4.2 AttributeError: 'tuple' object has no attribute 'items'
- **원인**: `env.reset()`이 (obs, info) 튜플 반환하지만 dict로 가정
- **해결**: `new_wrapper.py`의 `reset()` 메서드에서 튜플 처리 추가

### 4.3 ValueError: too many values to unpack (expected 4)
- **원인**: Gym의 `OrderEnforcing` 래퍼가 4-tuple 기대, wildfire는 5-tuple 반환
- **해결**: `gym.make()` 대신 `WildfireEnv()` 직접 생성으로 래퍼 회피

### 4.4 ValueError: action_space not provided
- **원인**: action_space가 dict 형태로 제공되어 MARLlib과 충돌
- **해결**: 단일 `Discrete(9)` space로 변경

### 4.5 KeyError: 'wildfire-ma' in policy_mapping_info
- **원인**: `get_env_info()` 메서드 누락
- **해결**: `new_wrapper.py`에 `get_env_info()` 메서드 추가

### 4.6 sys.exit(-1) from matplotlib
- **원인**: `window.py`가 matplotlib 없으면 프로세스 종료
- **해결**: 선택적 import로 변경

### 4.7 환경 초기화 무한 대기
- **원인**: 모든 에이전트가 동일 위치 (15, 1)에서 시작
- **해결**: 에이전트를 격자 모서리로 분산 배치

---

## 5. 학습 결과 확인

### 현재 상태 (2025-11-15 16:03 기준)
- ✅ **Status**: RUNNING
- ✅ **Iteration**: 3
- ✅ **Timesteps**: 9,000 / 5,000,000
- ✅ **Mean Reward**: -71.664
- ✅ **Episode Length**: 147.75

### 학습 종료 조건
- Target reward: 1000 또는
- Max timesteps: 5,000,000

---

## 6. 다른 컴퓨터에서 사용 시 주의사항

### 필수 작업:
1. **MARLlib 소스 코드 수정** (섹션 2 참조)
   - `cc_mlp.py` (line 57)
   - `centralized_critic.py` (lines 55-57)
   - `mappo.py` (line 58)

2. **가상환경에서 실행**:
   ```bash
   source .venv/bin/activate
   python train_marllib_self/new_train_mappo.py
   ```

### 선택 작업:
- matplotlib 설치 (렌더링 필요 시):
  ```bash
  pip install matplotlib
  ```

---

## 7. TODO

- [ ] wildfire.py의 디버그 출력 제거
- [ ] 학습 완료 후 성능 평가
- [ ] 체크포인트에서 학습 재개 테스트
- [ ] 다른 알고리즘 (QMIX, VDN 등)과 비교

---

## 8. 파일 변경 요약

### 새로 생성된 파일:
- `train_marllib_self/new_train_mappo.py` - MAPPO 학습 스크립트
- `train_marllib_self/new_wrapper.py` - RLlib 환경 래퍼
- `.venv/.../wildfire-ma.yaml` - MARLlib 환경 설정 (자동 생성)

### 수정된 프로젝트 파일:
- `train_marllib_self/environment.py` - 에이전트 시작 위치 수정
- `wildfire_environment/utils/window.py` - matplotlib 에러 처리
- `wildfire_environment/envs/wildfire.py` - 디버그 출력 추가 (임시)

### 수정된 MARLlib 파일 (외부 라이브러리):
- `.venv/.../marllib/marl/models/zoo/mlp/cc_mlp.py`
- `.venv/.../marllib/marl/algos/utils/centralized_critic.py`
- `.venv/.../marllib/marl/algos/core/CC/mappo.py`

---

**작성자**: Claude Code
**날짜**: 2025-11-15
**버전**: 1.0
