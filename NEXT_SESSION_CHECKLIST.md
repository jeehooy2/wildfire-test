# 다음 Claude 세션 체크리스트

## 🚀 가장 먼저 할 일

```bash
# 1. 학습이 여전히 진행 중인지 확인
ps aux | grep new_train_mappo_resume | grep -v grep

# 2. 최신 로그 확인
tail -100 /tmp/test_resume.log

# 3. 에러 여부 확인
tail -100 /tmp/test_resume.log | grep -i "error\|exception\|failed"
```

## ✅ 테스트 성공 기준

### 가장 중요한 것:
1. **iter 10 (=iter 50 원본)에 도달했는가?**
   ```bash
   tail -10 train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv
   ```

2. **에러 없이 평가 단계를 통과했는가?**
   ```bash
   # numpy.object_ 에러가 있는지 확인
   grep "numpy.object_" /tmp/test_resume.log
   # 있으면 에러, 없으면 성공!
   ```

3. **학습이 계속 진행되었는가?**
   ```bash
   # iter 수가 증가하고 있는지 확인
   tail -5 train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv
   ```

## 📊 정확한 확인 방법

### Option 1: 간단한 확인 (30초)
```bash
# 최신 5줄 확인 (가장 최근 iteration)
tail -5 train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv | cut -d',' -f12

# 또는 grep으로 iter 10 확인
grep "^10," train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv
```

### Option 2: 상세 확인 (1분)
```bash
# 전체 진행 상황 보기
wc -l train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv

# 현재 iter 찾기
tail -1 train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv | cut -d',' -f12

# 에러 확인
tail -200 /tmp/test_resume.log | grep -i "error\|exception\|failed"
```

### Option 3: 완전 확인 (2분)
```bash
# 학습 완료 여부 및 최종 상태 확인
if [ -f train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/result.json ]; then
  echo "✅ 학습 완료!"
  cat train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/result.json
else
  echo "⏳ 학습 진행 중"
  tail -20 train_marllib_self/experiments/mappo/run8_resumed_fix_crew/mappo_mlp_wildfire-ma/progress.csv
fi
```

## 🎯 기대되는 결과

### ✅ 성공 시
- ✓ iter 1~10 이상 완료
- ✓ `numpy.object_` 에러 없음
- ✓ 평가 단계(iter 50=iter 10) 통과
- ✓ 학습이 계속 진행 (iter 11, 12, ...)

### ❌ 실패 시
- 같은 에러 반복
- → 추가 디버깅 필요 (FIX_SUMMARY.md 참고)

## 📍 핵심 파일 경로

```
train_marllib_self/FIX_SUMMARY.md              # 상세 설명
CURRENT_SESSION_STATUS.md                     # 현재 상태
NEXT_SESSION_CHECKLIST.md                     # 이 파일

실시간 로그: /tmp/test_resume.log
결과: train_marllib_self/experiments/mappo/run8_resumed_fix_crew/
```

## ⏰ 예상 시간

| 시점 | 예상 iteration |
|------|----------------|
| 작성 시점 (23:42) | iter 7-8 |
| 1시간 후 (00:42) | iter 15-20 |
| 2시간 후 (01:42) | iter 25-30 |

## 🔄 다음 테스트 (성공 시)

```bash
# 새로운 학습 시작
python train_marllib_self/new_train_mappo.py --run-name test_new_training

# 또는 더 오래 실행
python train_marllib_self/new_train_mappo.py --run-name test_long_training
```

## 📝 기억할 것

1. **수정 내용**: new_wrapper.py의 policy_mapping_dict → 동적 생성
2. **목표**: iter 50(=iter 10 resume)에서 numpy.object_ 에러 없이 통과
3. **상태**: 현재까지 정상! (iter 1~7 에러 없음)
4. **문서**: FIX_SUMMARY.md에 상세 내용 있음

---

**마지막 업데이트**: 2025-11-16 23:42:31
**다음 확인**: 가능한 빨리 또는 대기
**예상 소요**: 3~5분 (간단한 확인)
