# Wildfire Environment 빠른 시작 가이드

이 가이드는 wildfire-environment를 빠르게 설치하고 실행하는 방법을 안내합니다.

## 사전 준비

- Python 3.8 또는 3.9가 설치되어 있어야 합니다
- uv 패키지 관리자가 설치되어 있어야 합니다

## 5분 안에 시작하기

### 1단계: 프로젝트 디렉토리로 이동

```bash
cd /path/to/wildfire_environment
```

### 2단계: 가상 환경 생성 및 설정

```bash
# Python 3.9로 가상 환경 생성
uv venv --python 3.9

# 환경 활성화 (macOS/Linux)
source .venv/bin/activate

# Windows의 경우
# .venv\Scripts\activate
```

### 3단계: 필요한 도구 설치

```bash
# pip 설치
uv pip install pip

# gym 0.21.0과 호환되는 버전으로 다운그레이드
.venv/bin/pip install "pip<24.1" "setuptools<66" "wheel==0.38.4"
```

### 4단계: 패키지 설치

```bash
# 의존성 설치
.venv/bin/pip install -r requirements.txt

# wildfire-environment 패키지 설치 (개발 모드)
.venv/bin/pip install -e .
```

### 5단계: 예제 실행

```bash
# 예제 스크립트 실행
.venv/bin/python example.py
```

## 한 줄 설치 스크립트 (전체)

```bash
uv venv --python 3.9 && \
source .venv/bin/activate && \
uv pip install pip && \
.venv/bin/pip install "pip<24.1" "setuptools<66" "wheel==0.38.4" && \
.venv/bin/pip install -r requirements.txt && \
.venv/bin/pip install -e . && \
.venv/bin/python example.py
```

## 설치 확인

Python 인터프리터에서 다음 코드를 실행하여 설치를 확인할 수 있습니다:

```python
import gym
import wildfire_environment

# 환경 생성
env = gym.make("wildfire-v0", num_agents=2, size=17)
print("✓ Wildfire environment 설치 성공!")
env.close()
```

## 자신만의 시뮬레이션 작성하기

```python
import gym
import wildfire_environment

# 환경 생성 및 설정
env = gym.make("wildfire-v0", 
    num_agents=2,                    # 에이전트 수
    size=17,                         # 그리드 크기
    initial_fire_size=3,            # 초기 화재 크기
    cooperative_reward=False,        # 협력 모드 여부
    max_steps=100,                   # 최대 스텝 수
)

# 환경 초기화
observation, info = env.reset(seed=42)

# 시뮬레이션 루프
for step in range(1000):
    # 행동 선택 (여기서는 무작위)
    action = env.action_space.sample()
    
    # 환경 스텝 실행
    observation, reward, done, info = env.step(action)
    
    # 에피소드 종료 시 재시작
    if done:
        observation, info = env.reset()

env.close()
```

## 문제 해결

### ImportError: No module named 'wildfire_environment'

해결 방법:
```bash
.venv/bin/pip install -e .
```

### gym 설치 실패

gym 0.21.0은 최신 pip와 호환성 문제가 있습니다. 다음을 확인하세요:
```bash
# pip 버전 확인
.venv/bin/pip --version

# pip<24.1이어야 합니다. 아니라면:
.venv/bin/pip install "pip<24.1"
```

### NumPy 버전 오류

NumPy 버전이 너무 높을 수 있습니다:
```bash
.venv/bin/pip install "numpy>=1.19.0,<1.24.0"
```

## 다음 단계

- [README.md](README.md)에서 상세한 문서 확인
- [example.py](example.py)에서 더 많은 예제 코드 확인
- 환경의 다양한 매개변수를 조정하여 실험해보기

## 추가 자료

- 완전한 문서: [README.md](README.md)
- 프로젝트 분석: [ANALYSIS.md](ANALYSIS.md)

