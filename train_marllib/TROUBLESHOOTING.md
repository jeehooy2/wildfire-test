# MARLlib MAPPO 문제 해결 가이드

## 현재 문제: Ray Redis 초기화 오류

### 오류 메시지
```
redis.exceptions.ResponseError: Invalid argument '9223372036854775775' for CONFIG SET 'maxclients'
```

### 원인
- MARLlib이 사용하는 Ray 1.8.0 버전의 알려진 버그
- macOS에서 Redis 설정 시 잘못된 maxclients 값 (2^63-1) 시도
- Redis가 허용하는 최대값 (2^32-1)을 초과

### 해결 방법

#### 방법 1: Ray 버전 업그레이드 (권장하지 않음)
```bash
# MARLlib과 호환성 문제 발생 가능
conda activate marllib-x86
pip install ray[rllib]==2.0.0
```
⚠️ **주의**: MARLlib은 특정 Ray 버전에 의존하므로 업그레이드 시 다른 문제 발생 가능

#### 방법 2: Docker 사용
```bash
# Docker에서 Linux 환경으로 실행하면 문제 없음
docker run -it --rm -v $(pwd):/workspace python:3.8
# 컨테이너 내부에서 MARLlib 설치 및 실행
```

#### 방법 3: Linux 서버 사용
- 가능하면 Linux 서버에서 실행 (AWS, GCP, 로컬 리눅스 등)
- macOS의 Redis 관련 이슈가 없음

#### 방법 4: RLlib 직접 사용 (커스텀 CTDE 구현)
MARLlib 대신 RLlib를 직접 사용하여 centralized critic 구현:

1. **Centralized Critic 모델 정의**
   - Custom model에서 global state를 입력으로 받는 value function 구현
   - `model_config`에서 `use_critic=True`, `use_lstm=False` 설정

2. **학습 파일**: `train_rllib/train_rllib_mappo_custom.py` 참고
   - RLlib의 PPO를 기반으로 커스텀 모델 사용
   - 각 에이전트에 global state 전달

3. **장점**:
   - Ray 버전 문제 없음 (최신 Ray 사용 가능)
   - MARLlib 의존성 제거
   - 더 유연한 커스터마이징 가능

4. **단점**:
   - MARLlib의 검증된 MAPPO 구현 대신 직접 구현 필요
   - 디버깅이 더 어려울 수 있음

## 참고

### Ray local_mode의 한계
- Ray 1.8.0의 `local_mode=True`도 여전히 Redis 초기화 시도
- 단일 프로세스 모드에서도 내부적으로 Redis 사용
- 이는 Ray 1.8.0의 알려진 제한사항

### MARLlib 환경 확인
```bash
conda activate marllib-x86
python -c "import ray; print('Ray:', ray.__version__)"
python -c "import marllib; print('MARLlib:', marllib.__version__)"
```

### 대안 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| MARLlib | 검증된 MAPPO 구현, 논문 재현성 | macOS 호환성 문제 |
| RLlib 커스텀 | macOS 호환, 유연성 | 직접 구현 필요 |
| Docker/Linux | MARLlib 사용 가능 | 추가 환경 설정 필요 |

## 추천 순서
1. **Linux 환경 사용** (가능하면)
2. **RLlib 커스텀 구현** (macOS에서 빠른 테스트)
3. **Docker 사용** (MARLlib 정확한 재현)
