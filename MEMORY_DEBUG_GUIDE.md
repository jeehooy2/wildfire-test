# 메모리 누수 디버깅 가이드

## 빠른 체크 (권장 순서)

### 1단계: 간단한 메모리 모니터링
```bash
cd /home/bmkim88/wildfire_environment
python train_marllib_self/test_memory_leak.py --episodes 10 --steps 30
```

**출력 해석:**
- `에피소드당 평균 증가: 0.0001 MB` → 정상 ✓
- `에피소드당 평균 증가: 0.01 MB` → 약간의 누수 ⚠️
- `에피소드당 평균 증가: 0.1 MB 이상` → 심각한 누수 🚨

### 2단계: 각 메서드별 상세 분석
```bash
# reset() 메서드 메모리 분석
python train_marllib_self/analyze_env_memory.py --test reset --resets 50

# step() 메서드 메모리 분석
python train_marllib_self/analyze_env_memory.py --test step --episodes 10 --steps 30

# 원본 환경 (래핑 전) 메모리 분석
python train_marllib_self/analyze_env_memory.py --test raw
```

**출력 해석:**
```
[관찰 구조 분석]
  Agent helicopter_0:
    - dtype: float32
    - shape: (1024,)
    - size (bytes): 4.10 KB      ← 관찰 크기 확인
  ...
  총 크기: 40.96 KB
```

메모리가 계속 증가하면 누수 의심

### 3단계: 실시간 훈련 모니터링
```bash
# 터미널 1: PPO 훈련 시작
python train_marllib_self/new_train_ppo.py --run-name memory_test

# 터미널 2: 실시간 모니터링
python train_marllib_self/monitor_training.py --interval 5 \
  --log-file train_marllib_self/results/memory_log.csv
```

**출력:**
```
[11:37:20] MEM:  1234.5MB (+  50.5MB) ↑ | CPU:  45.2% | Threads: 12
[11:37:25] MEM:  1267.3MB (+  83.3MB) ↑ | CPU:  48.1% | Threads: 12
[11:37:30] MEM:  1289.2MB (+105.2MB) ↑ | CPU:  51.0% | Threads: 12
```

메모리가 선형으로 증가하면 누수 확정

---

## 메모리 누수 위치별 원인과 해결책

### 문제 1: 관찰 데이터 누적 (가장 흔함)

**증상:**
- reset() 호출마다 메모리 증가
- 에피소드 종료 후에도 메모리 해제 안됨

**원인 (new_wrapper.py:213-218 확인):**
```python
obs = {}
for idx, agent_id in enumerate(self.agents):
    if str(idx) in obs_dict:
        obs_array = np.asarray(obs_dict[str(idx)], dtype=np.float32)
        obs[agent_id] = {"obs": obs_array}  # ← 여기서 복사본 생성?
```

**확인 방법:**
```python
# step() 메서드에서 같은 배열 객체인지 확인
obs_dict, _, _, _ = self.env.step(env_actions)
obs_array1 = obs_dict[str(0)]

obs_dict2, _, _, _ = self.env.step(env_actions)
obs_array2 = obs_dict2[str(0)]

print(id(obs_array1), id(obs_array2))  # 다르면 새로운 복사본 생성
```

**해결책:**
```python
# 복사본 생성 최소화
obs_array = obs_dict[str(idx)]
if obs_array.dtype != np.float32:
    obs_array = obs_array.astype(np.float32)
obs[agent_id] = {"obs": obs_array}
```

### 문제 2: 환경 내부 누적 (캐싱)

**증상:**
- step() 호출마다 메모리 증가
- 관찰 데이터 자체는 작지만 계속 증가

**원인:**
wildfire.py의 step()이나 reset() 내부에서:
- 이전 프레임을 캐시로 저장
- 히스토리를 계속 누적

**확인 방법:**
```bash
python train_marllib_self/analyze_env_memory.py --test raw
```

결과에서 원본 환경도 누수가 보이면 `wildfire_environment/envs/wildfire.py` 확인 필요

### 문제 3: Ray 워커 메모리 누적

**증상:**
- 단일 프로세스(local_mode=True)에서는 정상
- 분산 모드(local_mode=False)에서 iteration 25에서 크래시
- 메모리 사용이 계속 증가

**원인:**
- 워커가 이전 에피소드 데이터를 해제하지 않음
- 워커 간 공유 메모리 문제

**확인 방법:**
```bash
# local_mode=True로 변경
python train_marllib_self/new_train_ppo.py --run-name local_mode_test
```

local_mode=True에서는 정상이면, Ray 워커 설정 문제

**해결책 (new_train_ppo.py 수정):**
```python
ppo.fit(
    env,
    model,
    stop={...},
    local_mode=True,  # 임시로 True로 테스트
    num_workers=1,
    checkpoint_freq=50,
    local_dir=str(output_dir),
)
```

---

## 상세 메모리 프로파일링 (고급)

### memory_profiler 사용
```bash
pip install memory-profiler
```

`analyze_env_memory.py`에 데코레이터 추가:
```python
from memory_profiler import profile

@profile
def test_step_memory_leak():
    # 메모리 사용량이 라인별로 표시됨
    ...
```

실행:
```bash
python -m memory_profiler analyze_env_memory.py --test step
```

### objgraph를 이용한 객체 추적
```bash
pip install objgraph
```

```python
import objgraph

# 특정 시점에서 객체 수 비교
objgraph.show_most_common_types(limit=10)
```

---

## 빠른 진단 체크리스트

```
[ ] 1. test_memory_leak.py 실행 → 에피소드당 증가량 확인
[ ] 2. analyze_env_memory.py --test reset → reset() 메모리 증가 확인
[ ] 3. analyze_env_memory.py --test step → step() 메모리 증가 확인
[ ] 4. new_train_ppo.py에서 local_mode=True로 테스트
[ ] 5. 메모리 누수 위치 특정
[ ] 6. new_wrapper.py or wildfire.py 수정
[ ] 7. 재테스트
```

---

## 일반적인 해결책

### 옵션 A: 즉시 적용 가능
```python
# new_train_ppo.py 수정
ppo.fit(
    env, model,
    stop={...},
    local_mode=True,        # 분산 모드 비활성화
    num_workers=1,          # 워커 1개
    checkpoint_freq=100,    # 체크포인트 빈도 감소
    num_gpus=0,             # GPU 비활성화 (메모리 절약)
    local_dir=str(output_dir),
)
```

### 옵션 B: 메모리 효율적 설정
```python
ppo.fit(
    env, model,
    stop={...},
    local_mode=False,
    num_workers=1,          # 워커 1개만
    num_gpus=1,
    batch_episode=5,        # 배치 크기 감소
    checkpoint_freq=50,
    keep_checkpoints_num=2, # 최근 2개 체크포인트만 유지
    local_dir=str(output_dir),
)
```
