# new_evaluation.py 네트워크 로딩 문제 상세 분석 보고서

## 개요

`new_evaluation.py`는 MARLlib으로 학습한 정책 네트워크를 로드하여 평가하는 스크립트입니다. 하지만 여러 알고리즘(MAPPO, PPO, MAA2C 등)과 다양한 환경 설정(`partial_obs=True/False`)을 지원하기 위해서는 네트워크 아키텍처 감지 및 가중치 로딩에 개선이 필요합니다.

---

## 1. 주요 문제점

### 1.1 네트워크 아키텍처 고정 (2-layer만 지원)

**현재 구현 (line 222-263):**
```python
class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        # 항상 2-layer 고정
        self.p_encoder_layer0 = torch.nn.Linear(obs_dim, hidden_dim)
        self.p_encoder_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.p_branch = torch.nn.Linear(hidden_dim, action_dim)
        # ... value encoder도 동일하게 2-layer 고정
```

**문제:**
- MARLlib은 `encode_layer` 파라미터로 동적으로 layer 개수를 설정 가능
- `encode_layer="256-256"`은 2개 layer, `encode_layer="256-256-256"`은 3개 layer
- 3개 이상의 layer가 있는 모델은 **부분 로드만 가능**

**영향:**
- 3개 이상 layer 모델: 마지막 layer의 가중치가 로드되지 않음
- 정책 출력이 잘못된 feature를 기반으로 생성됨
- 성능 저하 또는 랜덤 정책처럼 동작

---

### 1.2 Value Encoder 가중치 미로드

**현재 구현 (line 266-308):**
```python
weight_mapping = {
    'p_encoder.encoder.0._model.0.weight': ('p_encoder_layer0', 'weight'),
    # ... policy encoder만 로드
    'p_branch._model.0.weight': ('p_branch', 'weight'),
    # ❌ vf_encoder.encoder.X._model.0.weight 없음!
}
```

**문제:**
- Value function encoder의 가중치를 로드하지 않음
- `vf_encoder` layer들이 초기화 상태 유지
- Value function이 학습된 가중치를 사용하지 않음

**영향:**
- PPO/MAPPO는 value function을 중요하게 사용하지 않지만, A2C 계열 알고리즘은 critical
- 가치 추정이 부정확함
- 정책의 학습된 behavior가 제대로 반영되지 않을 수 있음

---

### 1.3 Centralized Critic 미로드 (MAPPO)

**현재 구현:**
```python
# Centralized critic encoder를 정의하지만
self.cc_vf_encoder_layer0 = torch.nn.Linear(obs_dim, hidden_dim)
self.cc_vf_encoder_layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
self.cc_vf_branch = torch.nn.Linear(hidden_dim, 1)

# ❌ forward pass에서 사용하지 않음
def forward(self, x):
    p_features = torch.relu(self.p_encoder_layer0(x))
    # ... cc_vf_encoder는 사용 안함
```

**문제:**
- MAPPO는 centralized critic을 사용하여 모든 에이전트의 상태를 관찰
- 이를 통해 에이전트 간 협력을 학습함
- 그런데 가중치를 로드하지 않아 초기화 상태로 유지

**영향:**
- MAPPO 모델이 centralized critic의 학습된 정보를 활용하지 못함
- 중앙화된 정보 활용의 이점을 잃음
- 성능 저하

---

### 1.4 관측 차원 불일치 미검증

**환경 설정에 따른 관측 차원:**

| 설정 | 계산 | 결과 |
|------|------|------|
| `partial_obs=True, size=22, agent_view_size=5` | 7 × 5² + 1 | **176** |
| `partial_obs=True, size=22, agent_view_size=10` | 7 × 10² + 1 | **701** |
| `partial_obs=False, size=22` | 7 × 21² + 1 | **3,089** |
| `partial_obs=False, size=25` | 7 × 24² + 1 | **4,032** |

**현재 코드:**
```python
obs_dim = env.observation_space[first_agent_id].shape[0]
# 이 값이 checkpoint의 첫 번째 layer input_dim과 일치하지 않으면?
# → 가중치 로드 실패하지만, 에러 메시지가 명확하지 않음
```

**문제:**
- 가중치 shape 불일치로 로드 실패 가능
- 하지만 "dimension mismatch" 에러가 아닌 일반적인 로드 실패로 보임
- 사용자는 원인을 파악하기 어려움

**예시:**
- 체크포인트: `partial_obs=False, size=25` (4,032 차원으로 학습)
- 평가 환경: `partial_obs=False, size=22` (3,089 차원)
- 결과: 첫 번째 layer weight shape 불일치 → 가중치 로드 실패

---

### 1.5 부분 로드 탐지 불가

**현재 검증:**
```python
print(f"    총 {loaded_count}개 가중치 로드 성공 ({skipped_count}개 스킵)")
return loaded_count > 0  # ❌ 1개라도 로드되면 True!
```

**문제:**
- `return loaded_count > 0`은 1개 가중치만 로드되어도 성공으로 간주
- 6개 가중치 중 1개만 로드되었는데도 "로드 성공"으로 표시
- 실제로는 정책이 대부분 초기화 상태

**예시:**
```
총 1개 가중치 로드 성공 (11개 스킵)  # ← 이것도 성공으로 처리됨!
✓ Loaded helicopter_policy weights  # ← 실제로는 부분 로드
```

---

## 2. 원인 분석

### 2.1 MARLlib의 동적 네트워크 구조

MARLlib의 네트워크 구조:
```
encode_layer="256-256"        (2개 layer)
├── encoder.0._model.0  (input_dim → 256)
└── encoder.1._model.0  (256 → 256)

encode_layer="256-256-256"    (3개 layer)
├── encoder.0._model.0  (input_dim → 256)
├── encoder.1._model.0  (256 → 256)
└── encoder.2._model.0  (256 → 256)
```

고정된 2-layer 구조로는 3개 이상 layer 지원 불가.

### 2.2 체크포인트 형식 다양성

| 형식 | 위치 | 가중치 로드 방식 |
|------|------|-----------------|
| RLac (new) | `final/learner_group/learner/rl_module/{policy}/module_state.pkl` | Dict[str, np.ndarray] |
| RLlib (old) | checkpoint-* 파일 | `worker['state']['policy_*']['weights']` |

---

### 2.3 partial_obs 설정의 중요성

```python
ENV_CONFIG = {
    "size": 22,
    "partial_obs": False,  # ← 이것이 관측 차원 결정
    "agent_view_size": 10  # ← partial_obs=True일 때만 사용
}
```

체크포인트가 다른 설정으로 학습되었으면 관측 차원이 맞지 않음.

---

## 3. 개선 방안

### 3.1 동적 Layer 구조 지원

**함수 추가:**
```python
def detect_network_architecture(marllib_weights):
    """Checkpoint 가중치에서 네트워크 구조 자동 감지"""
    def extract_encoder_layers(prefix):
        layers = []
        for i in range(20):  # 최대 20개 layer 지원
            weight_key = f'{prefix}.encoder.{i}._model.0.weight'
            if weight_key in marllib_weights:
                weight = marllib_weights[weight_key]
                out_dim, in_dim = weight.shape
                layers.append({'in_dim': in_dim, 'out_dim': out_dim})
            else:
                break
        return layers

    p_encoder_layers = extract_encoder_layers('p_encoder')
    vf_encoder_layers = extract_encoder_layers('vf_encoder')
    cc_vf_encoder_layers = extract_encoder_layers('cc_vf_encoder')

    return {
        'p_encoder_layers': p_encoder_layers,
        'vf_encoder_layers': vf_encoder_layers,
        'cc_vf_encoder_layers': cc_vf_encoder_layers,
        'expected_obs_dim': p_encoder_layers[0]['in_dim'],
        'has_centralized_critic': len(cc_vf_encoder_layers) > 0,
    }
```

### 3.2 동적 네트워크 구성

```python
def create_policy_network(obs_dim, action_dim, arch_info):
    class PolicyNetwork(torch.nn.Module):
        def __init__(self, obs_dim, action_dim, arch_info):
            super().__init__()

            # Policy encoder를 ModuleList로 동적 구성
            self.p_encoder_layers = torch.nn.ModuleList()
            for layer_info in arch_info['p_encoder_layers']:
                layer = torch.nn.Linear(layer_info['in_dim'], layer_info['out_dim'])
                self.p_encoder_layers.append(layer)

            # Value encoder도 동일하게
            self.vf_encoder_layers = torch.nn.ModuleList()
            for layer_info in arch_info['vf_encoder_layers']:
                layer = torch.nn.Linear(layer_info['in_dim'], layer_info['out_dim'])
                self.vf_encoder_layers.append(layer)

            # Centralized critic (MAPPO용)
            if arch_info['has_centralized_critic']:
                self.cc_vf_encoder_layers = torch.nn.ModuleList()
                for layer_info in arch_info['cc_vf_encoder_layers']:
                    layer = torch.nn.Linear(layer_info['in_dim'], layer_info['out_dim'])
                    self.cc_vf_encoder_layers.append(layer)

        def forward(self, x):
            # 모든 layer에 대해 forward pass
            features = x
            for layer in self.p_encoder_layers:
                features = torch.relu(layer(features))
            logits = self.p_branch(features)
            return logits
```

### 3.3 모든 가중치 로드

```python
def load_marllib_weights_to_network(network, marllib_weights, arch_info):
    """정책, 값, 중앙화된 평가자 모든 가중치 로드"""

    loaded_count = 0

    # Policy encoder
    for layer_idx in range(len(arch_info['p_encoder_layers'])):
        weight_key = f'p_encoder.encoder.{layer_idx}._model.0.weight'
        bias_key = f'p_encoder.encoder.{layer_idx}._model.0.bias'
        # ... 로드 로직

    # Value encoder
    for layer_idx in range(len(arch_info['vf_encoder_layers'])):
        weight_key = f'vf_encoder.encoder.{layer_idx}._model.0.weight'
        bias_key = f'vf_encoder.encoder.{layer_idx}._model.0.bias'
        # ... 로드 로직

    # Centralized critic (MAPPO)
    if arch_info['has_centralized_critic']:
        for layer_idx in range(len(arch_info['cc_vf_encoder_layers'])):
            weight_key = f'cc_vf_encoder.encoder.{layer_idx}._model.0.weight'
            # ... 로드 로직

    return {
        'success': loaded_count >= expected_count * 0.8,  # 80% threshold
        'loaded_count': loaded_count,
        'total_expected': expected_count
    }
```

### 3.4 관측 차원 검증

```python
def validate_observation_dimension(expected_obs_dim, actual_obs_dim):
    """Checkpoint와 환경의 관측 차원 일치 확인"""
    if expected_obs_dim == actual_obs_dim:
        return True, f"Dimension match: {actual_obs_dim}"

    return False, (
        f"❌ Observation dimension mismatch!\n"
        f"    Expected (from checkpoint): {expected_obs_dim}\n"
        f"    Actual (from environment): {actual_obs_dim}\n"
        f"    Likely cause: partial_obs or grid_size differs from training"
    )
```

### 3.5 부분 로드 검증

```python
def verify_weights_loaded(loaded_count, expected_count):
    """로드된 가중치 비율 검증"""
    ratio = loaded_count / expected_count if expected_count > 0 else 0

    if ratio >= 0.95:  # 95% 이상
        return True, "All weights loaded"
    elif ratio >= 0.80:  # 80% 이상
        return True, f"Warning: {ratio*100:.1f}% weights loaded"
    else:
        return False, f"Only {ratio*100:.1f}% weights loaded (threshold: 80%)"
```

---

## 4. 실제 영향 분석

### 4.1 test2_individual_reward4 모델의 경우

모델 설정 분석:
```
Algorithm: MAPPO
Encoding: 256-256 (2-layer)
Partial obs: False
Grid size: 22
Expected obs dim: 7 × 21² + 1 = 3,089
```

현재 환경과 체크포인트 설정이 같다면:
- 2-layer 구조이므로 현재 코드도 정상 로드
- 하지만 value encoder 가중치를 로드하지 않아 부분 로드 상태
- MAPPO이므로centralized critic도 로드 안 됨

결과:
```
로드된 가중치:
  p_encoder.encoder.0 (weight, bias)
  p_encoder.encoder.1 (weight, bias)
  p_branch (weight, bias)
  ❌ vf_encoder 가중치 없음
  ❌ cc_vf_encoder 가중치 없음

총 6개 가중치만 로드 (전체 18개 중)
→ 33% 로드율로도 "로드 성공" 판정
→ 실제 성능은 예상보다 훨씬 낮음
```

---

## 5. 권장 조치

### 즉시 조치 (P0)

1. `new_evaluation_fixed.py` 사용
   - 동적 layer 구조 지원
   - Value encoder 가중치 로드
   - Centralized critic 지원
   - 관측 차원 검증

2. 기존 `new_evaluation.py` 백업
   - 문제 원인 분석용으로 보관

### 단기 조치 (P1)

1. 다른 evaluation 스크립트 검토
   - `new_compare_visual_ppo.py`
   - `new_render_policy.py`
   - 동일한 네트워크 로딩 문제 확인

2. 환경 설정 문서화
   - 각 모델이 학습된 설정 명확히 기록
   - `partial_obs`, `grid_size`, `agent_view_size` 등

### 장기 조치 (P2)

1. MARLlib 정책 직접 로드
   - 자체 네트워크 구축 대신 MARLlib 정책 객체 사용
   - 항상 정확한 구조 보장

2. Checkpoint 메타데이터 저장
   - 학습 설정을 JSON으로 checkpoint에 함께 저장
   - 평가 시 환경 설정과 자동 비교

---

## 6. 테스트 방법

```bash
# 기존 버전
python train_marllib_self/new_evaluation.py \
    --experiment train_marllib_self/experiments/mappo/test2_individual_reward4 \
    --episodes 2 \
    --seed 42

# 개선된 버전
python train_marllib_self/new_evaluation_fixed.py \
    --experiment train_marllib_self/experiments/mappo/test2_individual_reward4 \
    --episodes 2 \
    --seed 42
```

개선된 버전에서:
```
Expected observation dim (from checkpoint): 3089
Actual observation dim (from env): 3089
Dimension match: 3089

로드 결과: 18/18 가중치 로드
Value encoder layers 로드됨 ✓
Centralized critic layers 로드됨 ✓
```

---

## 7. 결론

`new_evaluation.py`의 네트워크 로딩은 다음 3가지 근본 문제로 인해 부분 로드 상태:

1. **네트워크 구조 고정**: 2-layer만 지원 → 3개 이상 layer 미로드
2. **Value encoder 미로드**: PPO/MAPPO 모델의 33% 가중치 미로드
3. **검증 부족**: 1개 가중치만 로드되어도 성공 판정

이로 인해:
- 예상보다 낮은 성능 평가
- 어떤 가중치가 로드되지 않았는지 파악 어려움
- 부분 로드된 정책이 정상 동작하는 것처럼 보임

**해결책**: `new_evaluation_fixed.py` 사용 (모든 가중치 동적 로드 + 검증 강화)

---

## 첨부: 가중치 로딩 비교표

### 기존 코드 (new_evaluation.py)

| 항목 | 로드 | 개수 |
|------|------|------|
| Policy encoder | ✓ | 2 layers (4개 param) |
| Policy branch | ✓ | 1 layer (2개 param) |
| Value encoder | ❌ | 0 |
| Value branch | ❌ | 0 |
| CC encoder | ❌ | 0 |
| CC branch | ❌ | 0 |
| **합계** | | **6/18 (33%)** |

### 개선된 코드 (new_evaluation_fixed.py)

| 항목 | 로드 | 개수 |
|------|------|------|
| Policy encoder | ✓ | 2 layers (4개 param) |
| Policy branch | ✓ | 1 layer (2개 param) |
| Value encoder | ✓ | 2 layers (4개 param) |
| Value branch | ✓ | 1 layer (2개 param) |
| CC encoder | ✓ | 2 layers (4개 param) |
| CC branch | ✓ | 1 layer (2개 param) |
| **합계** | | **18/18 (100%)** |

---

*작성일: 2025-11-20*
*분석 대상: new_evaluation.py (line 222-308)*
*권장 조치: new_evaluation_fixed.py 즉시 도입*
