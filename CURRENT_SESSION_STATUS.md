# 현재 세션 상태 (2025-11-16 23:38 기준)

## ⏳ 현재 진행 중인 테스트

**백그라운드 학습 실행 중:**
```bash
# 실시간 로그 확인
tail -f /tmp/test_resume.log

# 최신 상태만 보기
tail -50 /tmp/test_resume.log
```

| 항목 | 값 |
|------|-----|
| **테스트 커맨드** | `python train_marllib_self/new_train_mappo_resume.py --restore checkpoint_000040 --run-name run8_resumed_fix_crew` |
| **시작 시간** | 2025-11-16 23:30:54 |
| **현재 경과 시간** | ~8분 |
| **테스트 목표** | iter 50(=iter 10 in resume)에서 에러 없이 평가 단계 통과 |
| **현재 상태** | ✅ 정상 진행 중 |

## 📍 주요 파일 위치

### 수정된 코드
```
train_marllib_self/new_wrapper.py       # TODO 49th iter 에러 관련 3곳 수정
train_marllib_self/new_train_mappo.py   # TODO 49th iter 에러 주석 추가
```

### 실시간 모니터링
```
/tmp/test_resume.log                    # 실시간 학습 로그
train_marllib_self/experiments/mappo/run8_resumed_fix_crew/
  └── mappo_mlp_wildfire-ma/
      ├── progress.csv                 # 학습 진행 통계
      ├── result.json                  # 최종 결과
      └── MAPPOTrainer*/checkpoint_*   # checkpoint 파일
```

### 문서
```
train_marllib_self/FIX_SUMMARY.md       # 상세 수정 사항 (이 파일!)
CURRENT_SESSION_STATUS.md               # 현재 상태 (지금 읽는 파일)
```

## 🔍 다음 세션에서 확인할 사항

### 1. 가장 먼저 확인할 것
```bash
# 학습이 여전히 진행 중인가?
ps aux | grep new_train_mappo_resume

# 최신 로그 확인
tail -100 /tmp/test_resume.log

# 에러 발생했는가?
tail -100 /tmp/test_resume.log | grep -i "error\|exception\|failed"
```

### 2. iteration 진행 상황 확인
```bash
# iter 10 (=iter 50)에 도달했는가?
tail -10 train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv

# 또는 정확히 grep
grep "^10," train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv
```

### 3. 평가 단계 통과 확인
```bash
# 평가 단계 로그 찾기
grep -i "evaluation\|evaluate" /tmp/test_resume.log | tail -30
```

### 4. 최종 성공 기준
- ✅ `numpy.object_` 에러 발생 안 함
- ✅ iter 10에서 에러 없이 통과
- ✅ 학습이 계속 진행 (iter 11, 12, ...)

## 📊 예상 일정

| 시간 | 예상 이벤트 |
|------|-----------|
| 23:38 | 현재 시간 (iter ~6-7 정도) |
| 00:00 | iter ~10-12 (평가 단계 통과 확인 가능) |
| 00:30 | iter ~20-25 (충분히 진행) |
| 01:00 | iter ~30+ (안정화) |

## 🎯 주요 수정 내용 요약

### Problem
```python
# 기존: 고정값
policy_mapping_dict = {
    "wildfire-ma": {
        "team_prefix": ("helicopter_", "truck_"),  # 하드코딩!
    }
}
```

### Solution
```python
# 새로운: 동적 생성
def get_policy_mapping_dict(num_helicopters, num_trucks, num_crews):
    team_prefix = []
    if num_helicopters > 0:
        team_prefix.append("helicopter_")
    if num_trucks > 0:
        team_prefix.append("truck_")
    if num_crews > 0:
        team_prefix.append("crew_")

    return {
        "wildfire-ma": {
            "team_prefix": tuple(team_prefix),  # 동적!
        }
    }
```

## 🚀 다음 단계 (테스트 완료 후)

1. **테스트 성공 시**
   ```bash
   # 새로운 학습 시작
   python train_marllib_self/new_train_mappo.py --run-name run9_fresh_test
   ```

2. **테스트 실패 시**
   - 로그 분석 (FIX_SUMMARY.md 참고)
   - 에러 메시지로 추가 원인 파악

## 📝 기억할 점

- **에러 원인**: evaluation worker 초기화 시 policy 불일치
- **해결책**: 동적 policy_mapping_dict 생성
- **핵심**: 모든 워커가 일관된 정책 구성을 받아야 함
- **상태**: ✅ 수정 완료, 테스트 진행 중

---

**마지막 업데이트**: 2025-11-16 23:38
**다음 확인 시간**: 가능한 빨리 (또는 00:00 경)
