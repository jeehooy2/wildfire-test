# 다중 에이전트 강화학습 - 완전 가이드

wildfire 환경에서 RLlib을 사용한 다중 에이전트 강화학습 완전 가이드입니다.

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [파일 구조](#파일-구조)
3. [설치](#설치)
4. [사용법](#사용법)
5. [예제](#예제)
6. [문제 해결](#문제-해결)

## 🚀 빠른 시작

### 방법 1: 자동 스크립트 (권장)

```bash
# RLlib 설치
pip install "ray[rllib]" torch

# 빠른 시작 스크립트 실행 (테스트 학습 + 시뮬레이션)
./quick_start_multiagent.sh
```

### 방법 2: 수동 실행

```bash
# 1. RLlib 설치
pip install "ray[rllib]" torch

# 2. 환경 테스트
python test_rl_ready.py

# 3. 학습 (100 iterations)
python train_multiagent_rllib.py

# 4. 시뮬레이션
python simulate_trained_model.py --checkpoint ./rllib_checkpoints/best --save-gif
```

## 📁 파일 구조

```
wildfire_environment/
├── wildfire_rllib_wrapper.py      # RLlib 환경 래퍼
├── train_multiagent_rllib.py      # 다중 에이전트 학습
├── simulate_trained_model.py      # 학습 모델 시뮬레이션
├── test_rl_ready.py               # 환경 테스트
├── quick_start_multiagent.sh      # 자동 시작 스크립트
├── MULTIAGENT_TRAINING_GUIDE.md   # 상세 가이드
└── README_MULTIAGENT.md           # 이 파일
```

### 각 파일 설명

**wildfire_rllib_wrapper.py**
- wildfire-v0 환경을 RLlib MultiAgentEnv으로 변환
- Dict 공간 → RLlib 형식 변환 처리
- 다중 에이전트 인터페이스 제공

**train_multiagent_rllib.py**
- PPO 알고리즘으로 다중 에이전트 학습
- 자동 체크포인트 저장
- 최고 성능 모델 별도 저장
- TensorBoard 로깅 지원

**simulate_trained_model.py**
- 학습된 모델 로드 및 실행
- GIF 생성 기능
- 무작위 정책과 성능 비교
- 상세한 통계 출력

## 💻 설치

### 필수 요구사항

```bash
# Python 3.8 또는 3.9
python --version

# 기본 환경 (이미 설치되어 있어야 함)
pip install -r requirements.txt
pip install -e .
```

### RLlib 설치

**중요: Ray 2.0 이상 필요 (최신 API)**

```bash
# RLlib + PyTorch (권장)
pip install "ray[rllib]>=2.7.0" torch

# 또는 RLlib + TensorFlow
pip install "ray[rllib]>=2.7.0" tensorflow

# TensorBoard (선택사항, 학습 모니터링용)
pip install tensorboard
```

### 설치 확인

```bash
# RLlib 설치 및 버전 확인
python -c "import ray; from ray.rllib.algorithms.ppo import PPO; print(f'✓ RLlib 설치 성공 (Ray v{ray.__version__})')"
```

**참고:**
- ✅ **신버전 API** (Ray >= 2.0): `from ray.rllib.algorithms.ppo import PPO`
- ❌ **구버전 API** (Ray < 2.0): `from ray.rllib.agents.ppo import PPOTrainer`
- 자세한 내용: `RLLIB_API_VERSION.md` 참조

## 📖 사용법

### 1. 환경 테스트

먼저 환경이 제대로 작동하는지 확인:

```bash
python test_rl_ready.py
```

**예상 출력:**
```
✓ 환경 생성 성공
✓ 환경 리셋 성공
✓ 환경 스텝 실행 성공
환경 테스트 완료! RL 학습을 시작할 준비가 되었습니다.
```

### 2. 학습

**기본 학습 (100 iterations):**
```bash
python train_multiagent_rllib.py
```

**커스텀 설정:**
```bash
# 200 iterations, 20번마다 체크포인트 저장
python train_multiagent_rllib.py --iterations 200 --checkpoint-freq 20

# 저장 위치 변경
python train_multiagent_rllib.py --save-dir ./my_models
```

**학습 중 모니터링:**
```bash
# 터미널 1: 학습 실행
python train_multiagent_rllib.py

# 터미널 2: TensorBoard 실행
tensorboard --logdir ~/ray_results

# 브라우저에서 http://localhost:6006 접속
```

### 3. 시뮬레이션

**기본 시뮬레이션:**
```bash
python simulate_trained_model.py --checkpoint ./rllib_checkpoints/best
```

**GIF 생성:**
```bash
python simulate_trained_model.py \
    --checkpoint ./rllib_checkpoints/best \
    --save-gif \
    --gif-filename my_agents.gif \
    --fps 15
```

**성능 비교:**
```bash
python simulate_trained_model.py \
    --checkpoint ./rllib_checkpoints/best \
    --episodes 20 \
    --compare
```

**예상 출력:**
```
===========================================================
학습 모델 vs 무작위 정책 비교
===========================================================

학습된 모델:
  평균 보상: -4.23 ± 0.45

무작위 정책:
  평균 보상: -8.67 ± 1.23

성능 향상:
  4.44 (51.2%)
===========================================================
```

## 💡 예제

### 예제 1: 빠른 테스트 (10분)

```bash
# 짧은 학습으로 빠르게 테스트
python train_multiagent_rllib.py --iterations 10

# 결과 확인
python simulate_trained_model.py \
    --checkpoint ./rllib_checkpoints/final \
    --episodes 3 \
    --save-gif
```

### 예제 2: 완전한 학습 (1-2시간)

```bash
# 충분한 학습
python train_multiagent_rllib.py --iterations 200

# 최고 성능 모델로 시뮬레이션
python simulate_trained_model.py \
    --checkpoint ./rllib_checkpoints/best \
    --episodes 50 \
    --save-gif \
    --compare
```

### 예제 3: 협력 학습

`train_multiagent_rllib.py`에서 환경 설정 수정:

```python
env_config = {
    "num_agents": 2,
    "size": 17,
    "initial_fire_size": 3,
    "cooperative_reward": True,  # ← 협력 모드
    "max_steps": 100,
    "agent_start_positions": ((1, 1), (15, 15)),
}
```

### 예제 4: 더 많은 에이전트

```python
env_config = {
    "num_agents": 4,  # ← 4명의 에이전트
    "size": 25,
    "initial_fire_size": 5,
    "cooperative_reward": False,
    "max_steps": 200,
    "agent_start_positions": ((1, 1), (23, 23), (1, 23), (23, 1)),
}
```

## 🔧 문제 해결

### 문제 1: RLlib 설치 오류

```
ERROR: Failed building wheel for ray
```

**해결:**
```bash
# C++ 컴파일러 필요 (macOS)
xcode-select --install

# 재설치
pip install --upgrade pip
pip install "ray[rllib]" torch
```

### 문제 2: Ray 초기화 실패

```
RuntimeError: Maybe you called ray.init twice
```

**해결:**
```bash
ray stop
python train_multiagent_rllib.py
```

### 문제 3: 메모리 부족

```
MemoryError: Unable to allocate array
```

**해결:** `train_multiagent_rllib.py`에서 워커 수 줄이기
```python
.rollouts(num_rollout_workers=1)  # 2 → 1
```

### 문제 4: 학습 속도 느림

**해결:** 병렬화 증가 (CPU 코어 수에 맞춰)
```python
.rollouts(num_rollout_workers=4)
```

또는 GPU 사용:
```python
.resources(num_gpus=1)
```

### 문제 5: 체크포인트 찾을 수 없음

```
FileNotFoundError: Checkpoint not found
```

**해결:** 전체 경로 사용
```bash
# 체크포인트 목록 확인
ls -la ./rllib_checkpoints/

# 전체 경로 사용
python simulate_trained_model.py \
    --checkpoint /absolute/path/to/checkpoint_000100
```

## 📊 학습 결과 해석

### 좋은 결과:
- ✅ 평균 보상이 증가 추세
- ✅ 보상의 표준편차가 감소 (안정적)
- ✅ 무작위 정책보다 30% 이상 성능 향상

### 나쁜 결과:
- ❌ 보상이 진동하거나 발산
- ❌ 무작위 정책과 성능 차이 없음
- ❌ 학습 중 성능이 악화

**개선 방법:**
1. 학습률 감소: `lr=1e-4` 또는 `3e-5`
2. 더 오래 학습: `--iterations 300`
3. 배치 크기 조정: `train_batch_size=8000`
4. 환경 설정 변경 (보상 함수, 그리드 크기 등)

## 🎓 다음 단계

1. **MULTIAGENT_TRAINING_GUIDE.md** 읽기 - 상세 설명
2. **하이퍼파라미터 튜닝** - 학습 성능 최적화
3. **다른 알고리즘 시도** - DQN, A3C 등
4. **환경 커스터마이징** - 보상 함수, 그리드 크기 등

## 📚 참고 자료

- [Ray RLlib 공식 문서](https://docs.ray.io/en/latest/rllib/)
- [PPO 알고리즘 논문](https://arxiv.org/abs/1707.06347)
- [Multi-Agent RL 튜토리얼](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-environments)

## 🙋 FAQ

**Q: 학습에 시간이 얼마나 걸리나요?**
A: 설정에 따라 다르지만, 100 iterations는 일반적으로 30분~1시간 정도 걸립니다.

**Q: GPU가 필요한가요?**
A: 필수는 아니지만, GPU를 사용하면 학습 속도가 2-3배 빨라집니다.

**Q: 협력 학습과 경쟁 학습의 차이는?**
A:
- 협력: 모든 에이전트가 같은 보상 (공동 목표)
- 경쟁: 각 에이전트가 자신의 영역 우선 (이기적 보상)

**Q: 학습이 잘 안되는 것 같아요.**
A:
1. TensorBoard로 학습 곡선 확인
2. `--compare` 옵션으로 무작위 정책과 비교
3. 하이퍼파라미터 튜닝 시도
4. 더 오래 학습 (200+ iterations)

## ✅ 체크리스트

학습 전:
- [ ] RLlib 설치 완료
- [ ] `test_rl_ready.py` 성공
- [ ] 충분한 디스크 공간 (체크포인트용)

학습 중:
- [ ] TensorBoard 모니터링
- [ ] 보상이 증가하는지 확인
- [ ] 정기적으로 체크포인트 저장

학습 후:
- [ ] 최고 성능 모델 확인
- [ ] 시뮬레이션으로 성능 검증
- [ ] GIF로 시각화
- [ ] 무작위 정책과 비교

---

**즐거운 강화학습 되세요! 🚀**
