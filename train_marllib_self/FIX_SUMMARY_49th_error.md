# MAPPO Training Fix Summary - iter 49 Error Resolution

## 🎯 문제 정의

**에러 발생:**
- **시점**: Iteration 49 완료 후 iter 50의 평가(evaluation) 단계
- **에러 메시지**: `TypeError: can't convert np.ndarray of type numpy.object_`
- **발생 위치**: RLlib의 `policy.set_state()` 중 optimizer state 복원 시

## 🔍 근본 원인 분석

### 문제의 원인
```
평가 워커(evaluation worker) 초기화 → 정책 구성 불일치 → optimizer state 손상
```

**구체적 원인:**
- `new_wrapper.py`의 `policy_mapping_dict`가 **고정값**
- `team_prefix: ("helicopter_", "truck_")` 하드코딩
- 평가 워커가 생성될 때, 동일한 정책 구성을 보장하지 못함
- 특정 정책의 optimizer state가 `None` 또는 비정상 상태로 복원됨
- `numpy.object_` 타입으로 변환되어 PyTorch가 처리 불가

## ✅ 해결 방법 (Method A)

### 1. new_wrapper.py 수정

**파일**: `train_marllib_self/new_wrapper.py`

#### 변경 1: 동적 policy_mapping_dict 함수 추가
```python
# Line 16-71: 새로운 get_policy_mapping_dict() 함수 추가
def get_policy_mapping_dict(num_helicopters, num_trucks, num_crews):
    """
    환경의 에이전트 구성에 따라 동적으로 policy_mapping_dict 생성
    모든 워커가 일관된 정책 구성을 받도록 보장
    """
    team_prefix = []
    if num_helicopters > 0:
        team_prefix.append("helicopter_")
    if num_trucks > 0:
        team_prefix.append("truck_")
    if num_crews > 0:
        team_prefix.append("crew_")

    return {
        "wildfire-ma": {
            "description": "wildfire suppression with heterogeneous agents",
            "team_prefix": tuple(team_prefix),
            "all_agents_one_policy": False,
            "one_agent_one_policy": False,
        }
    }
```

#### 변경 2: WildfireRLlibEnv.__init__에서 환경 설정 저장
**Lines 131-166**:
- `self.num_helicopters`, `self.num_trucks`, `self.num_crews` 저장
- agent_ids 생성 시 사용
- 나중에 `get_env_info()`에서 동적 생성에 필요

```python
self.num_helicopters = clean_config.get('num_helicopters', 0)
self.num_trucks = clean_config.get('num_trucks', 0)
self.num_crews = clean_config.get('num_crews', 0)
```

#### 변경 3: get_env_info()에서 동적 policy_mapping_dict 생성
**Lines 314-342**:
```python
def get_env_info(self):
    # 동적으로 policy_mapping_dict 생성
    dynamic_policy_mapping_dict = get_policy_mapping_dict(
        self.num_helicopters,
        self.num_trucks,
        self.num_crews
    )

    env_info = {
        "space_obs": self.observation_space,
        "space_act": self.action_space,
        "num_agents": self.num_agents,
        "episode_limit": self.env.max_steps,
        "policy_mapping_info": dynamic_policy_mapping_dict  # 동적 생성!
    }
    return env_info
```

### 2. new_train_mappo.py 수정

**파일**: `train_marllib_self/new_train_mappo.py`

**Lines 123-135**: TODO 주석 추가
- `marl.make_env()` 호출 시 자동으로 동적 policy_mapping_dict 생성됨을 명시
- 모든 워커가 일관된 정책을 받음을 보장

## 🧪 테스트 현황

### 테스트 방법
```bash
python train_marllib_self/new_train_mappo_resume.py \
  --restore /home/bmkim88/wildfire_environment/train_marllib_self/experiments/mappo/run8/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_2cd80_00000_0_2025-11-16_17-17-48/checkpoint_000040 \
  --run-name run8_resumed_fix_crew
```

### 진행 상황 (2025-11-16 23:38 기준)

**✅ 학습 시작 성공**
- checkpoint 40에서 정상 복구
- 평가 워커 생성 성공
- 정책 동기화 성공 (에러 없음!)

**현재 상태:**
- 백그라운드 PID: `ca8c84`
- 실시간 로그: `/tmp/test_resume.log`
- 결과 디렉토리: `train_marllib_self/experiments/mappo/run8_resumed_fix_crew/`

**테스트 목표:**
- ✅ iter 1 (=원래 iter 41) 완료
- ⏳ iter 10 (=원래 iter 50, 평가 단계) 통과 여부 확인 중

## 📋 파일 변경 사항

### 수정된 파일

1. **new_wrapper.py**
   - `TODO 49th iter 에러` 태그로 표시된 부분 참고
   - Lines 16-71: `get_policy_mapping_dict()` 함수 추가
   - Lines 131-166: `__init__` 수정
   - Lines 314-342: `get_env_info()` 수정

2. **new_train_mappo.py**
   - `TODO 49th iter 에러` 태그 추가 (Lines 123-135)
   - setup_marllib_config() 함수는 수정 없음 (이미 올바름)
   - resume 스크립트도 자동으로 적용됨

## 🔄 다음 단계

### 1. 테스트 결과 확인 (다음 세션에서)
```bash
# 최신 로그 확인
tail -50 /tmp/test_resume.log

# iteration 10 도달 확인
grep "iter.*10" train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv

# 또는 평가 단계 통과 확인
grep -i "evaluation\|evaluate" /tmp/test_resume.log | tail -20
```

### 2. 성공 기준
- ✅ iter 10(=iter 50)에서 에러 없이 평가 단계 완료
- ✅ 학습이 계속 진행됨 (iter 11, 12, ... 진행)
- ✅ `numpy.object_` 에러 발생 안 함

### 3. 추가 테스트 (필요시)
새로운 학습 시작 테스트:
```bash
python train_marllib_self/new_train_mappo.py --run-name test_new_training
```

## 💾 참고 정보

- **ENV_CONFIG**: `train_marllib_self/environment.py` (num_crews: 0)
- **YAML 설정**: `/.venv/lib/python3.8/site-packages/examples/config/env_config/wildfire-ma.yaml`
- **MARLlib 정책 생성**: `MARLlib/marllib/marl/algos/run_cc.py` (line 95-160 참고)

## 🎯 핵심 아이디어

> **동적 policy_mapping_dict 생성**
>
> 고정된 team_prefix 대신, 환경 설정(num_helicopters, num_trucks, num_crews)에 따라
> 실시간으로 policy_mapping_dict를 생성함으로써, 모든 워커(main trainer,
> evaluation workers, training workers)가 일관된 정책 구성을 받도록 보장.
>
> 이는 특히 정책이 0개인 경우(num_crews=0)나 나중에 에이전트 구성을 변경할 때
> 유용함.

---

**작성일**: 2025-11-16
**상태**: ✅ 수정 완료, 테스트 진행 중
