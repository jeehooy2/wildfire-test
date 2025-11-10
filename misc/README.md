# wildfire-environment

산불 진화를 시뮬레이션하기 위한 gym 기반 다중 에이전트 환경. 산불 확산 프로세스와 여러 UAV를 사용한 진화 작업은 마르코프 결정 과정(MDP)으로 모델링.

> 🚀 **빠르게 시작하기**: [QUICKSTART.md](QUICKSTART.md) 또는 아래의 자동 설치 스크립트를 참조하세요!

## 특징

환경은 세 가지 유형의 에이전트를 지원합니다:
- **팀 에이전트** (단일 공유 보상)
- **그룹 에이전트** (각 그룹이 공유 보상을 가짐)
- **개별 에이전트** (개별 보상)

팀 에이전트를 위한 보상 함수는 전체 숲에 대해 동등한 선호도로 화재 확산을 방지하는 것을 목표로 하며, 그룹 또는 개별 에이전트 보상은 다른 지역보다 이기적 관심 지역(selfish regions)에서 확산을 방지하는 데 더 높은 선호도를 가집니다.

## 자동 설치 스크립트

가장 간단한 방법은 제공된 설치 스크립트를 사용하는 것입니다:

```bash
./install.sh
```

또는 수동으로 설치하려면 아래 안내를 따르세요.

## 시스템 요구사항

- Python 3.8 또는 3.9 (권장: 3.9)
- uv 또는 pip

## 설치 방법 (uv 사용 - 권장)

### 1. uv 가상 환경 생성

```bash
# Python 3.9로 가상 환경 생성 (3.8이 없는 경우)
uv venv --python 3.9

# 환경 활성화
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate  # Windows
```

### 2. pip 및 기본 도구 설치

gym 0.21.0과 호환되는 버전의 도구들을 설치합니다:

```bash
# uv를 사용하여 pip 설치
uv pip install pip

# 호환 가능한 버전으로 도구 다운그레이드
.venv/bin/pip install "pip<24.1" "setuptools<66" "wheel==0.38.4"
```

**중요**: gym 0.21.0은 최신 pip 버전과 호환성 문제가 있어 pip<24.1 버전이 필요합니다.

### 3. 의존성 설치

```bash
# requirements.txt의 패키지들 설치
.venv/bin/pip install -r requirements.txt
```

### 4. 패키지를 개발 모드로 설치

소스 코드를 직접 수정하면서 테스트할 수 있도록 개발 모드로 설치합니다:

```bash
# 개발 모드로 설치
.venv/bin/pip install -e .
```

## 빠른 설치 (한 번에 실행)

```bash
# 전체 설치 과정을 한 번에 실행
uv venv --python 3.9
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install pip
.venv/bin/pip install "pip<24.1" "setuptools<66" "wheel==0.38.4"
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## 기본 사용법

이 저장소는 gym 기반 환경을 제공합니다. 핵심 기여는 `WildfireEnv` 클래스로, `gym.Env`의 서브클래스입니다 (`MultiGridEnv` 클래스를 통해).

> 📹 **렌더링 및 GIF 저장**: [RENDERING_GUIDE.md](RENDERING_GUIDE.md)에서 시뮬레이션을 시각화하고 GIF로 저장하는 방법을 확인하세요!

### 예제 코드

```python
import gym
import wildfire_environment

env = gym.make("wildfire-v0", 
    num_agents=2,
    size=17,
    initial_fire_size=3,
    cooperative_reward=False,
    log_selfish_region_metrics=True,
    selfish_region_xmin=[7, 13],
    selfish_region_xmax=[9, 15],
    selfish_region_ymin=[7, 1],
    selfish_region_ymax=[9, 3],
    )
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset()
env.close()
```

### 예제 파일 실행

저장소에 포함된 `example.py` 파일을 실행할 수 있습니다:

```bash
# 가상 환경이 활성화된 상태에서
python example.py

# 또는 가상 환경을 직접 지정
.venv/bin/python example.py
```

예상 출력:
```
============================================================
Wildfire Environment 시뮬레이션 시작
============================================================
에이전트 수: 2
그리드 크기: 17 x 17
최대 스텝 수: 100
============================================================
에피소드 1 완료 (총 스텝: 100)
에피소드 2 완료 (총 스텝: 200)
...
============================================================
```

## 환경 설명

### Wildfire Environment

| 속성 | 설명 |
|------|------|
| Actions | Discrete |
| Agent Action Space | Discrete(5) |
| Observations | Discrete |
| Observability | 완전 관찰 가능 |
| Agent Observation Space | Box([0,...],[1,...],(에이전트 수에 따라 다름,),float32) |
| States | Discrete |
| State Space | Box([0,...],[1,...],(에이전트 수에 따라 다름,),float32) |
| Agents | 협력적 또는 비협력적 또는 그룹 |
| Number of Agents | >=1 |
| Termination Condition | 불타는 나무가 없을 때 |
| Truncation Steps | >=1 |
| Creation | gym.make("wildfire-v0") |

에이전트는 불타는 나무 위로 이동하여 난연제를 뿌립니다. 초기 화재는 무작위로 배치됩니다. 에이전트는 협력적(공유 보상) 또는 비협력적(개별/그룹 보상)일 수 있습니다. 비협력 에이전트는 그리드 내에서 이기적 관심 영역을 우선적으로 보호합니다.

## 주요 매개변수

- `num_agents`: UAV 에이전트의 수 (기본값: 2)
- `size`: 정사각형 그리드 세계의 한 변 (기본값: 17)
- `initial_fire_size`: 정사각형 모양의 초기 화재 지역의 한 변 (기본값: 1)
- `cooperative_reward`: 에이전트가 협력 보상을 사용하는지 여부 (기본값: False)
- `log_selfish_region_metrics`: 이기적 지역의 나무와 관련된 메트릭을 기록할지 여부 (기본값: False)
- `selfish_region_xmin/xmax/ymin/ymax`: 이기적 관심 지역의 경계 좌표 (리스트)
- `max_steps`: 에피소드의 최대 단계 수 (기본값: 100)
- `partial_obs`: 에이전트가 부분 관찰성을 가지는지 여부 (기본값: False)
