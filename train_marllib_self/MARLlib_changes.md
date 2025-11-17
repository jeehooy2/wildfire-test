# MARLlib 필수 수정사항

## 문제 설명
49번째 iteration 이후 MAPPO 훈련 중 다음 오류 발생:
```
TypeError: can't convert np.ndarray of type numpy.object_.
The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
```

**원인**: MARLlib의 preprocessor에서 observation을 numpy array로 변환할 때 dtype을 지정하지 않아,
observation이 list 형태일 경우 numpy가 object dtype으로 자동 변환함.

---

## 필수 수정 파일

### 파일: `MARLlib/marllib/patch/rllib/models/preprocessors.py`

**위치**: Line 80-86 (check_shape 메서드)

**현재 코드**:
```python
def check_shape(self, observation: Any) -> None:
    """Checks the shape of the given observation."""
    if self._i % OBS_VALIDATION_INTERVAL == 0:
        # Convert lists to np.ndarrays.
        if type(observation) is list and isinstance(
                self._obs_space, gym.spaces.Box):
            observation = np.array(observation)  # ❌ dtype 미지정!
```

**수정 코드**:
```python
def check_shape(self, observation: Any) -> None:
    """Checks the shape of the given observation."""
    if self._i % OBS_VALIDATION_INTERVAL == 0:
        # Convert lists to np.ndarrays.
        if type(observation) is list and isinstance(
                self._obs_space, gym.spaces.Box):
            observation = np.array(observation, dtype=self._obs_space.dtype)  # ✅ dtype 명시
```

**변경사항**:
- `np.array(observation)` → `np.array(observation, dtype=self._obs_space.dtype)`
- observation space의 dtype을 명시적으로 지정하여 object dtype 생성 방지

---

## 임시 해결책 (현재 적용)

`train_marllib_self/new_wrapper.py`의 `reset()` 및 `step()` 메서드에서
observation을 명시적으로 float32로 변환하여 임시로 해결했습니다:

```python
# numpy object dtype 변환 에러 방지: observation을 명시적으로 float32로 변환
obs_array = np.asarray(obs_dict[str(idx)], dtype=np.float32)
obs[agent_id] = {"obs": obs_array}
```

이 임시 해결책은 observation space의 dtype이 float32인 경우에만 작동합니다.

---

## 추천사항

MARLlib 저장소에 PR을 제출하여 위 수정사항을 merge 받는 것을 권장합니다.
현재는 우리의 wrapper에서 dtype을 강제로 지정하여 해결했습니다.
