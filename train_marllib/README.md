# MARLlib MAPPO 학습

MARLlib을 사용한 진정한 MAPPO (CTDE) 학습을 위한 폴더입니다.

## 📂 구조

```
train_marllib/
├── train_marllib_mappo.py      # MAPPO 학습 스크립트
├── wildfire_marllib_wrapper.py # MARLlib 환경 래퍼
├── environment.py               # 환경 설정
├── test_marllib_setup.py        # 설정 테스트
├── results/                     # 학습 결과 저장 (자동 생성)
│   ├── mappo_wildfire/         # 기본 실험
│   ├── mappo_run1/             # 커스텀 실험 1
│   └── ...
└── README.md                    # 이 파일
```

## 🚀 학습 실행

### 기본 실행

```bash
# 1. 환경 활성화
conda activate marllib-x86

# 2. PYTHONPATH 설정
export PYTHONPATH=/Users/jeehooy2/Desktop/wildfire_environment-aiblue:$PYTHONPATH

# 3. 학습 시작 (100 iterations)
python train_marllib/train_marllib_mappo.py --iterations 100
```

### 실험 이름 지정

```bash
# 결과가 train_marllib/results/my_experiment/ 에 저장됨
python train_marllib/train_marllib_mappo.py \
    --iterations 500 \
    --exp-name my_experiment
```

## 📊 결과 확인

### 결과 저장 위치

학습 결과는 **자동으로** `train_marllib/results/<실험이름>/` 에 저장됩니다:

```
train_marllib/results/
└── mappo_wildfire/              # 실험 폴더
    ├── PPO_wildfire_xxxxx/      # Ray Tune 결과
    │   ├── checkpoint_000010/   # 체크포인트
    │   ├── checkpoint_000020/
    │   ├── events.out.tfevents  # TensorBoard 로그
    │   └── params.json
    └── ...
```

### TensorBoard로 모니터링

```bash
# 실험별로 확인
tensorboard --logdir train_marllib/results/mappo_wildfire/

# 모든 실험 비교
tensorboard --logdir train_marllib/results/
```

브라우저에서 `http://localhost:6006` 접속

## 🎯 주요 명령어

### 짧은 테스트 (10 iterations)
```bash
python train_marllib/train_marllib_mappo.py --iterations 10 --exp-name test
```

### 본격적인 학습 (100 iterations)
```bash
python train_marllib/train_marllib_mappo.py --iterations 100 --exp-name run1
```

### 긴 학습 (1000 iterations)
```bash
python train_marllib/train_marllib_mappo.py \
    --iterations 1000 \
    --checkpoint-freq 50 \
    --exp-name long_run
```

### 학습 재개
```bash
python train_marllib/train_marllib_mappo.py \
    --iterations 500 \
    --restore train_marllib/results/run1/PPO_wildfire_xxxxx/checkpoint_000100
```

## ⚙️ 환경 설정

`environment.py`에서 설정 수정:

```python
ENV_CONFIG = {
    "size": 22,                  # 그리드 크기
    "num_agents": 4,             # 에이전트 수
    "num_helicopters": 2,        # 헬리콥터 (빠름)
    "num_trucks": 2,             # 트럭 (중간)
    "num_crews": 0,              # 소방대원 (느림)
    "max_steps": 300,            # 최대 스텝
    "reward_shaping": "individual",
}
```

## 🔑 CTDE (Centralized Training, Decentralized Execution)

### 핵심 차이점

| RLlib IPPO | MARLlib MAPPO (CTDE) |
|------------|---------------------|
| 각 에이전트마다 독립적인 Critic | **Centralized Critic** |
| Local observation만 사용 | **Global state 사용** |
| 협력 학습 제한적 | **강력한 협력 학습** |

### 작동 원리

```python
# Actor: Decentralized (각자 행동)
action = policy(local_observation)

# Critic: Centralized (전체 상태로 평가)
value = critic(global_state)  # 모든 에이전트 위치 + 모든 나무 상태
```

## 📝 메모

- **결과 저장**: `train_marllib/results/` 에 자동 저장
- **Iteration 기반**: RLlib 기본 코드와 동일한 방식
- **체크포인트**: `--checkpoint-freq`로 빈도 조절
- **TensorBoard**: 학습 중에도 실시간 확인 가능

## 🐛 문제 해결

### Import 오류
```bash
export PYTHONPATH=/Users/jeehooy2/Desktop/wildfire_environment-aiblue:$PYTHONPATH
```

### MARLlib 환경
```bash
conda activate marllib-x86
```

### 디스크 공간
결과 파일이 클 수 있으므로 주기적으로 오래된 실험 삭제:
```bash
rm -rf train_marllib/results/old_experiment/
```
