"""
MARLlib 학습된 정책과 랜덤 정책 시각적 비교 (유연한 아키텍처)

MARLlib (MAPPO, MAA2C 등)로 학습된 이질적 에이전트(헬리콥터, 트럭, 승무원)와
랜덤 에이전트의 행동을 side-by-side로 비교하는 GIF를 생성합니다.

주요 기능:
- Checkpoint의 weight 구조를 자동으로 분석하여 적응형 network 구성
- Full observation과 partial observation 자동 감지
- 이질적 에이전트 지원
- Weight 로드 실패 시 상세 보고

실행 예시:
# Global state를 사용하는 경우 (full observation)
python train_marllib_self/new_compare_visual_flexible.py \
    --checkpoint train_marllib_self/experiments/mappo/run1 \
    --episodes 3 \
    --seed 42

# Partial observation을 사용하는 경우
python train_marllib_self/new_compare_visual_flexible.py \
    --checkpoint train_marllib_self/experiments/mappo/run_test_global_state_flag_partial_obs \
    --episodes 3 \
    --seed 42
"""

import sys
import os
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pickle
import torch
import re
from pathlib import Path
from train_marllib_self.environment import ENV_CONFIG
from wildfire_environment.envs import WildfireEnv
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# [Weight 구조 분석]
# ============================================================================
def analyze_weight_structure(marllib_weights):
    """
    Checkpoint의 weight 구조를 분석하여 network 구조를 파악

    Parameters
    ----------
    marllib_weights : dict
        MARLlib checkpoint에서 추출한 가중치 딕셔너리

    Returns
    -------
    dict
        분석 결과:
        {
            'input_dim': int,           # Policy encoder의 input dimension
            'hidden_dim': int,          # Hidden layer dimension
            'output_dim': int,          # Policy branch의 output dimension
            'cc_input_dim': int,        # Centralized critic encoder의 input dimension
            'is_partial_obs': bool,     # Partial observation 여부
            'encoder_layers': list,     # Encoder layer 구조
        }
    """
    analysis = {
        'input_dim': None,
        'hidden_dim': None,
        'output_dim': None,
        'cc_input_dim': None,
        'is_partial_obs': False,
        'encoder_layers': [],
    }

    # p_encoder.encoder.0._model.0.weight에서 input_dim과 hidden_dim 추출
    if 'p_encoder.encoder.0._model.0.weight' in marllib_weights:
        weight = marllib_weights['p_encoder.encoder.0._model.0.weight']
        hidden_dim, input_dim = weight.shape
        analysis['input_dim'] = input_dim
        analysis['hidden_dim'] = hidden_dim

    # p_branch._model.0.weight에서 output_dim 추출
    if 'p_branch._model.0.weight' in marllib_weights:
        weight = marllib_weights['p_branch._model.0.weight']
        output_dim, _ = weight.shape
        analysis['output_dim'] = output_dim

    # cc_vf_encoder.encoder.0._model.0.weight에서 cc_input_dim 추출
    if 'cc_vf_encoder.encoder.0._model.0.weight' in marllib_weights:
        weight = marllib_weights['cc_vf_encoder.encoder.0._model.0.weight']
        _, cc_input_dim = weight.shape
        analysis['cc_input_dim'] = cc_input_dim

        # input_dim과 cc_input_dim이 다르면 partial observation
        if analysis['input_dim'] != analysis['cc_input_dim']:
            analysis['is_partial_obs'] = True

    # Encoder layer 구조 추출
    encoder_layers = []
    for i in range(10):  # 최대 10개 layer 확인
        key = f'p_encoder.encoder.{i}._model.0.weight'
        if key in marllib_weights:
            weight = marllib_weights[key]
            out_dim, in_dim = weight.shape
            encoder_layers.append({'layer': i, 'in_dim': in_dim, 'out_dim': out_dim})
        else:
            break

    analysis['encoder_layers'] = encoder_layers

    return analysis


def print_weight_analysis(analysis):
    """Weight 분석 결과를 읽기 좋은 형식으로 출력"""
    print("\n  📊 Weight 구조 분석:")
    print(f"    - Input dimension: {analysis['input_dim']}")
    print(f"    - Hidden dimension: {analysis['hidden_dim']}")
    print(f"    - Output dimension: {analysis['output_dim']}")
    print(f"    - Centralized critic input dim: {analysis['cc_input_dim']}")
    print(f"    - Partial observation: {'예' if analysis['is_partial_obs'] else '아니오'}")
    print(f"    - Encoder layers: {len(analysis['encoder_layers'])}")
    for layer_info in analysis['encoder_layers']:
        print(f"      Layer {layer_info['layer']}: {layer_info['in_dim']} -> {layer_info['out_dim']}")


# ============================================================================
# [Checkpoint 자동 탐지]
# ============================================================================
def find_latest_checkpoint(directory_path):
    """
    주어진 디렉토리에서 가장 최신 checkpoint를 자동으로 찾음

    Ray Tune 표준 구조를 따르는 디렉토리에서 checkpoint_XXXXX 형식의
    가장 높은 번호의 checkpoint를 선택합니다.

    Parameters
    ----------
    directory_path : str
        탐색할 디렉토리 경로 (예: train_marllib_self/experiments/mappo/run4)

    Returns
    -------
    str or None
        찾은 checkpoint 경로, 없으면 None
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"  ❌ 디렉토리를 찾을 수 없습니다: {directory_path}")
        return None

    if not directory.is_dir():
        # 이미 checkpoint 파일 경로인 경우
        if directory.name.startswith('checkpoint_'):
            print(f"  ✓ 이미 checkpoint 경로입니다: {directory_path}")
            return str(directory_path)
        print(f"  ❌ 파일 경로입니다. 디렉토리를 입력해주세요: {directory_path}")
        return None

    # checkpoint 디렉토리 찾기 (checkpoint_XXXXX 패턴)
    checkpoint_dirs = []

    # 재귀적으로 모든 checkpoint_* 디렉토리 찾기
    for item in directory.rglob('checkpoint_*'):
        if item.is_dir():
            # checkpoint_XXXXX 패턴 확인
            match = re.search(r'checkpoint_(\d+)', item.name)
            if match:
                checkpoint_number = int(match.group(1))
                checkpoint_dirs.append((checkpoint_number, item))

    if not checkpoint_dirs:
        print(f"  ❌ checkpoint를 찾을 수 없습니다: {directory_path}")
        print(f"     다음 구조를 예상합니다:")
        print(f"     {directory_path}/mappo_mlp_wildfire-ma/MAPPOTrainer_.../checkpoint_XXXXX")
        return None

    # 가장 높은 번호의 checkpoint 선택
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint_num, latest_checkpoint_path = checkpoint_dirs[0]

    print(f"  ✓ 최신 checkpoint 발견: checkpoint_{latest_checkpoint_num:06d}")
    print(f"     경로: {latest_checkpoint_path}")

    return str(latest_checkpoint_path)


# ============================================================================
# [이질적 정책 지원]
# ============================================================================
def load_heterogeneous_policies(checkpoint_path):
    """
    체크포인트에서 이질적 정책(helicopter, truck, crew)을 로드

    Returns
    -------
    dict
        {'helicopter_policy': weights, 'truck_policy': weights, 'crew_policy': weights}
    """

    checkpoint_dir = Path(checkpoint_path)

    # checkpoint-* 파일 찾기
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
    checkpoint_files = [f for f in checkpoint_files if f.is_file() and not f.name.endswith('.tune_metadata')]

    if not checkpoint_files:
        print(f"  ⚠ 경고: checkpoint-* 파일을 찾을 수 없습니다: {checkpoint_dir}")
        return None

    # 가장 최신 checkpoint 파일 선택 (번호가 높은 것)
    checkpoint_files.sort(key=lambda x: int(x.name.split('-')[1]))
    checkpoint_file = checkpoint_files[-1]

    print(f"  체크포인트 파일: {checkpoint_file.name}")

    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)

    # worker는 bytes로 저장되어 있음 (Ray 직렬화)
    policies_weights = {}

    if 'worker' in checkpoint_data:
        worker_bytes = checkpoint_data['worker']
        worker_data = pickle.loads(worker_bytes)

        if 'state' in worker_data:
            state = worker_data['state']

            # 이질적 정책들 찾기
            # RLlib 체크포인트 형식: policy_<agent_type>_ (e.g., policy_helicopter_, policy_truck_, policy_crew_)
            policy_name_mappings = {
                'policy_helicopter_': 'helicopter_policy',
                'policy_truck_': 'truck_policy',
                'policy_crew_': 'crew_policy'
            }

            for checkpoint_key, canonical_name in policy_name_mappings.items():
                if checkpoint_key in state:
                    policy_state = state[checkpoint_key]
                    if 'weights' in policy_state:
                        policies_weights[canonical_name] = policy_state['weights']
                        print(f"  ✓ {canonical_name} weights 발견 (checkpoint key: {checkpoint_key})")
                    else:
                        print(f"  ⚠ {checkpoint_key} 상태는 있지만 weights 없음")

            # 호환성: shared_policy도 확인 (단일 정책 학습 시)
            if 'shared_policy' in state and not policies_weights:
                policy_state = state['shared_policy']
                if 'weights' in policy_state:
                    print(f"  ✓ shared_policy weights 발견 (모든 에이전트가 공유)")
                    return {'shared': policy_state['weights']}

    if not policies_weights:
        print("  ⚠ 정책을 찾을 수 없습니다.")
        if 'worker_data' in locals() and 'state' in worker_data:
            print(f"  사용 가능한 state 키: {list(worker_data['state'].keys())}")
        return None

    return policies_weights


# ============================================================================
# [적응형 Policy Network 생성]
# ============================================================================
def create_adaptive_policy_network(analysis):
    """
    Weight 분석 결과를 기반으로 적응형 network 생성

    Parameters
    ----------
    analysis : dict
        Weight 분석 결과

    Returns
    -------
    class
        PolicyNetwork 클래스
    """
    input_dim = analysis['input_dim']
    hidden_dim = analysis['hidden_dim']
    output_dim = analysis['output_dim']
    cc_input_dim = analysis['cc_input_dim']
    encoder_layers = analysis['encoder_layers']

    class AdaptivePolicyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # ===== Policy Encoder =====
            # Dynamic layer 구성
            self.p_encoder_layers = torch.nn.ModuleList()
            for layer_info in encoder_layers:
                layer = torch.nn.Linear(layer_info['in_dim'], layer_info['out_dim'])
                self.p_encoder_layers.append(layer)

            # ===== Policy Branch =====
            final_hidden_dim = encoder_layers[-1]['out_dim'] if encoder_layers else hidden_dim
            self.p_branch = torch.nn.Linear(final_hidden_dim, output_dim)

            # ===== Value Function Encoder =====
            self.vf_encoder_layers = torch.nn.ModuleList()
            for layer_info in encoder_layers:
                layer = torch.nn.Linear(layer_info['in_dim'], layer_info['out_dim'])
                self.vf_encoder_layers.append(layer)

            # ===== Value Function Branch =====
            self.vf_branch = torch.nn.Linear(final_hidden_dim, 1)

            # ===== Centralized Critic Encoder =====
            # cc_input_dim은 다를 수 있음 (partial observation인 경우)
            self.cc_encoder_layers = torch.nn.ModuleList()

            # cc_input_dim -> hidden_dim으로 변환하는 첫 번째 layer
            self.cc_encoder_layers.append(torch.nn.Linear(cc_input_dim, hidden_dim))

            # 나머지 encoder layer들은 동일한 구조
            for i in range(1, len(encoder_layers)):
                layer_info = encoder_layers[i]
                layer = torch.nn.Linear(layer_info['in_dim'], layer_info['out_dim'])
                self.cc_encoder_layers.append(layer)

            # ===== Centralized Critic Branch =====
            self.cc_vf_branch = torch.nn.Linear(final_hidden_dim, 1)

        def forward(self, x):
            """Policy network forward pass"""
            # Policy encoder
            features = x
            for layer in self.p_encoder_layers:
                features = torch.relu(layer(features))

            # Policy branch
            logits = self.p_branch(features)
            return logits

        def get_action(self, obs):
            """Select action from observation"""
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = self.forward(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            return action

    return AdaptivePolicyNetwork()


def load_adaptive_weights_to_network(network, marllib_weights):
    """
    MARLlib checkpoint 가중치를 적응형 network로 로드

    자동으로 layer 매칭을 수행하여 유연성 있게 가중치를 로드합니다.
    로드 실패 시 자세한 메시지를 출력합니다.
    """
    try:
        loaded_count = 0
        failed_count = 0
        skipped_count = 0
        failed_details = []

        # Policy encoder layers 로드
        encoder_layer_idx = 0
        for i in range(10):
            weight_key = f'p_encoder.encoder.{i}._model.0.weight'
            bias_key = f'p_encoder.encoder.{i}._model.0.bias'

            if weight_key not in marllib_weights:
                break

            if encoder_layer_idx < len(network.p_encoder_layers):
                try:
                    weight = marllib_weights[weight_key]
                    network.p_encoder_layers[encoder_layer_idx].weight.data = torch.FloatTensor(weight)
                    loaded_count += 1

                    if bias_key in marllib_weights:
                        bias = marllib_weights[bias_key]
                        network.p_encoder_layers[encoder_layer_idx].bias.data = torch.FloatTensor(bias)
                        loaded_count += 1
                    else:
                        skipped_count += 1

                except Exception as e:
                    print(f"    ✗ {weight_key}: 로드 실패 - {str(e)}")
                    failed_count += 1
                    failed_details.append({'key': weight_key, 'reason': str(e)})

                encoder_layer_idx += 1

        # Policy branch 로드
        if 'p_branch._model.0.weight' in marllib_weights:
            try:
                weight = marllib_weights['p_branch._model.0.weight']
                network.p_branch.weight.data = torch.FloatTensor(weight)
                loaded_count += 1

                if 'p_branch._model.0.bias' in marllib_weights:
                    bias = marllib_weights['p_branch._model.0.bias']
                    network.p_branch.bias.data = torch.FloatTensor(bias)
                    loaded_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"    ✗ p_branch._model.0.weight: 로드 실패 - {str(e)}")
                failed_count += 1
                failed_details.append({'key': 'p_branch._model.0.weight', 'reason': str(e)})
        else:
            skipped_count += 1

        # Value function encoder layers 로드
        vf_encoder_layer_idx = 0
        for i in range(10):
            weight_key = f'vf_encoder.encoder.{i}._model.0.weight'
            bias_key = f'vf_encoder.encoder.{i}._model.0.bias'

            if weight_key not in marllib_weights:
                break

            if vf_encoder_layer_idx < len(network.vf_encoder_layers):
                try:
                    weight = marllib_weights[weight_key]
                    network.vf_encoder_layers[vf_encoder_layer_idx].weight.data = torch.FloatTensor(weight)
                    loaded_count += 1

                    if bias_key in marllib_weights:
                        bias = marllib_weights[bias_key]
                        network.vf_encoder_layers[vf_encoder_layer_idx].bias.data = torch.FloatTensor(bias)
                        loaded_count += 1
                    else:
                        skipped_count += 1

                except Exception as e:
                    print(f"    ✗ {weight_key}: 로드 실패 - {str(e)}")
                    failed_count += 1
                    failed_details.append({'key': weight_key, 'reason': str(e)})

                vf_encoder_layer_idx += 1

        # Value function branch 로드
        if 'vf_branch._model.0.weight' in marllib_weights:
            try:
                weight = marllib_weights['vf_branch._model.0.weight']
                network.vf_branch.weight.data = torch.FloatTensor(weight)
                loaded_count += 1

                if 'vf_branch._model.0.bias' in marllib_weights:
                    bias = marllib_weights['vf_branch._model.0.bias']
                    network.vf_branch.bias.data = torch.FloatTensor(bias)
                    loaded_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"    ✗ vf_branch._model.0.weight: 로드 실패 - {str(e)}")
                failed_count += 1
                failed_details.append({'key': 'vf_branch._model.0.weight', 'reason': str(e)})
        else:
            skipped_count += 1

        # Centralized critic encoder layers 로드
        cc_encoder_layer_idx = 0
        for i in range(10):
            weight_key = f'cc_vf_encoder.encoder.{i}._model.0.weight'
            bias_key = f'cc_vf_encoder.encoder.{i}._model.0.bias'

            if weight_key not in marllib_weights:
                break

            if cc_encoder_layer_idx < len(network.cc_encoder_layers):
                try:
                    weight = marllib_weights[weight_key]
                    network.cc_encoder_layers[cc_encoder_layer_idx].weight.data = torch.FloatTensor(weight)
                    loaded_count += 1

                    if bias_key in marllib_weights:
                        bias = marllib_weights[bias_key]
                        network.cc_encoder_layers[cc_encoder_layer_idx].bias.data = torch.FloatTensor(bias)
                        loaded_count += 1
                    else:
                        skipped_count += 1

                except Exception as e:
                    print(f"    ✗ {weight_key}: 로드 실패 - {str(e)}")
                    failed_count += 1
                    failed_details.append({'key': weight_key, 'reason': str(e)})

                cc_encoder_layer_idx += 1

        # Centralized critic branch 로드
        if 'cc_vf_branch._model.0.weight' in marllib_weights:
            try:
                weight = marllib_weights['cc_vf_branch._model.0.weight']
                network.cc_vf_branch.weight.data = torch.FloatTensor(weight)
                loaded_count += 1

                if 'cc_vf_branch._model.0.bias' in marllib_weights:
                    bias = marllib_weights['cc_vf_branch._model.0.bias']
                    network.cc_vf_branch.bias.data = torch.FloatTensor(bias)
                    loaded_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"    ✗ cc_vf_branch._model.0.weight: 로드 실패 - {str(e)}")
                failed_count += 1
                failed_details.append({'key': 'cc_vf_branch._model.0.weight', 'reason': str(e)})
        else:
            skipped_count += 1

        # 결과 요약 출력
        if failed_count > 0:
            print(f"    ✓ {loaded_count}개 가중치 로드 성공 ({failed_count}개 실패, {skipped_count}개 스킵)")
            if failed_details:
                print(f"\n    📋 실패한 weight 목록:")
                for detail in failed_details:
                    print(f"      - {detail['key']}: {detail['reason']}")
        else:
            print(f"    ✓ {loaded_count}개 가중치 로드 성공 ({skipped_count}개 스킵)")

        return failed_count == 0

    except Exception as e:
        print(f"    ❌ 가중치 로드 중 예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# [에이전트별 정책 매핑]
# ============================================================================
def get_agent_policy_name(agent, num_helicopters, num_trucks, num_crews):
    """
    에이전트 객체로부터 정책 이름 반환

    Parameters
    ----------
    agent : Agent
        에이전트 객체 (Helicopter, Truck, Crew 등)
    num_helicopters : int
        헬리콥터 수
    num_trucks : int
        트럭 수
    num_crews : int
        승무원 수

    Returns
    -------
    str
        정책 이름 ('helicopter_policy', 'truck_policy', 'crew_policy')
    """
    # 에이전트 객체의 type 속성 확인
    if agent.type == 'helicopter':
        return 'helicopter_policy'
    elif agent.type == 'truck':
        return 'truck_policy'
    elif agent.type == 'crew':
        return 'crew_policy'
    else:
        # 기본값 (호환성)
        if agent.index < num_helicopters:
            return 'helicopter_policy'
        elif agent.index < num_helicopters + num_trucks:
            return 'truck_policy'
        else:
            return 'crew_policy'


def run_episode_and_render(env, policy_networks, policy_mapping_fn, seed,
                          max_steps=300, policy_type="trained", is_partial_obs=False):
    """
    에피소드를 실행하고 프레임을 수집 (이질적 정책 지원, 부분 관찰)

    Parameters
    ----------
    env : WildfireEnv
        환경
    policy_networks : dict or None
        정책 네트워크 딕셔너리
        - 학습된 정책: {'helicopter_policy': network, 'truck_policy': network, ...}
        - shared: {'shared': network}
        - 랜덤: None
    policy_mapping_fn : callable or None
        에이전트ID → 정책이름 매핑 함수
    seed : int
        랜덤 시드
    max_steps : int
        최대 스텝 수
    policy_type : str
        정책 타입 (trained or random)
    is_partial_obs : bool
        부분 관찰 여부

    Returns
    -------
    tuple
        (frames, episode_reward, episode_length)
    """
    # 환경 리셋
    obs_dict, _ = env.reset(seed=seed)

    frames = []
    done = False
    episode_reward = 0
    step = 0

    # 초기 프레임
    frame = env.render(mode='rgb_array')
    frames.append(frame)

    while not done and step < max_steps:
        # 각 에이전트의 액션 선택
        actions = {}
        # 문자열 프리픽스 기반 에이전트 ID
        for agent_id in obs_dict.keys():
            # Partial observation의 경우: obs_dict[agent_id]는 딕셔너리
            # 'obs': local observation, 'state': global state
            if is_partial_obs and isinstance(obs_dict[agent_id], dict):
                obs = obs_dict[agent_id]['obs']  # local observation만 사용
            else:
                obs = obs_dict[agent_id]

            if policy_type == "trained" and policy_networks is not None:
                # 이질적 정책 선택
                if policy_mapping_fn is not None:
                    policy_name = policy_mapping_fn(agent_id)

                    if policy_name in policy_networks:
                        network = policy_networks[policy_name]
                        action = network.get_action(obs)
                    else:
                        # 정책을 찾을 수 없으면 랜덤 액션
                        action = env.action_space[agent_id].sample()
                else:
                    # shared 정책 (호환성)
                    if 'shared' in policy_networks:
                        network = policy_networks['shared']
                        action = network.get_action(obs)
                    else:
                        action = env.action_space[agent_id].sample()
            else:
                # 랜덤 액션
                action = env.action_space[agent_id].sample()

            actions[agent_id] = action

        # 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        # 리워드 합산
        episode_reward += np.mean(list(reward_dict.values()))

        # 프레임 저장
        frame = env.render(mode='rgb_array')
        frames.append(frame)

        step += 1

    return frames, episode_reward, step


def combine_frames_side_by_side(frames1, frames2, label1="Trained", label2="Random"):
    """
    두 프레임 시퀀스를 side-by-side로 결합

    Parameters
    ----------
    frames1 : list
        첫 번째 프레임 시퀀스
    frames2 : list
        두 번째 프레임 시퀀스
    label1 : str
        첫 번째 프레임 레이블
    label2 : str
        두 번째 프레임 레이블

    Returns
    -------
    list
        결합된 프레임 시퀀스
    """
    # 더 짧은 시퀀스에 맞춤
    min_len = min(len(frames1), len(frames2))
    frames1 = frames1[:min_len]
    frames2 = frames2[:min_len]

    combined_frames = []

    for f1, f2 in zip(frames1, frames2):
        # PIL Image로 변환
        img1 = Image.fromarray(f1)
        img2 = Image.fromarray(f2)

        # 크기 확인
        w1, h1 = img1.size
        w2, h2 = img2.size

        # 텍스트 레이블을 위한 여유 공간
        label_height = 30

        # 새로운 이미지 생성 (side-by-side)
        combined_width = w1 + w2 + 10  # 10px 여백
        combined_height = max(h1, h2) + label_height
        combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

        # 이미지 붙이기
        combined.paste(img1, (0, label_height))
        combined.paste(img2, (w1 + 10, label_height))

        # 레이블 추가
        draw = ImageDraw.Draw(combined)
        try:
            # 기본 폰트 사용
            font = ImageFont.load_default()
        except:
            font = None

        # 텍스트 추가
        draw.text((w1//2 - 30, 5), label1, fill=(0, 0, 0), font=font)
        draw.text((w1 + 10 + w2//2 - 30, 5), label2, fill=(0, 0, 0), font=font)

        # 구분선 추가
        draw.line([(w1 + 5, label_height), (w1 + 5, combined_height)], fill=(200, 200, 200), width=2)

        combined_frames.append(np.array(combined))

    return combined_frames


def save_as_gif(frames, filename, fps=10):
    """
    프레임을 GIF로 저장

    Parameters
    ----------
    frames : list
        프레임 리스트
    filename : str
        저장할 파일명
    fps : int
        초당 프레임 수
    """
    if not frames:
        print("저장할 프레임이 없습니다!")
        return

    images = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)

    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    file_size = os.path.getsize(filename) / 1024
    print(f"  ✓ 저장 완료: {filename} ({len(frames)} frames, {file_size:.1f} KB)")


def main(checkpoint_path, num_episodes=3, seed=42, output_dir=None):
    """
    메인 함수: 학습된 모델과 랜덤 정책을 시각적으로 비교 (이질적 에이전트 + 유연한 아키텍처 지원)

    이질적 에이전트(헬리콥터, 트럭, 승무원)가 각각 다른 정책을 학습했을 때,
    부분 관찰 환경에서도 자동으로 적응하여 각 정책별로 행동을 시각화하고
    랜덤 정책과 비교합니다.

    Parameters
    ----------
    checkpoint_path : str
        MARLlib 학습된 모델의 체크포인트 경로 또는 checkpoint가 있는 디렉토리 경로
        (디렉토리인 경우 자동으로 최신 checkpoint를 선택)
    num_episodes : int
        생성할 비교 GIF 수 (기본값: 3)
    seed : int
        시작 랜덤 시드 (기본값: 42)
    output_dir : str, optional
        GIF를 저장할 디렉토리. None이면 checkpoint 경로에서 알고리즘과 run name을 추출하여
        train_marllib_self/results/{algo}/{run_name}/ 에 저장
    """
    # ============================================================================
    # Checkpoint 경로 검증: 디렉토리면 최신 checkpoint 자동 선택
    # ============================================================================
    checkpoint_path_validated = Path(checkpoint_path)
    if checkpoint_path_validated.is_dir():
        print(f"\n디렉토리 경로 감지: {checkpoint_path}")
        print(f"최신 checkpoint를 자동으로 찾는 중...")
        latest_checkpoint = find_latest_checkpoint(checkpoint_path)
        if latest_checkpoint is None:
            print("❌ checkpoint를 찾을 수 없습니다. 스크립트를 중단합니다.")
            return
        checkpoint_path = latest_checkpoint
    else:
        print(f"\n직접 지정된 checkpoint 경로: {checkpoint_path}")

    # Checkpoint 번호 추출
    checkpoint_num = None
    match = re.search(r'checkpoint_(\d+)', checkpoint_path)
    if match:
        checkpoint_num = int(match.group(1))

    # output_dir이 지정되지 않은 경우, checkpoint 경로에서 알고리즘과 run name 추출
    if output_dir is None:
        # 경로 예: train_marllib_self/experiments/mappo/run_test_global_state_flag_partial_obs/...
        path_parts = os.path.normpath(checkpoint_path).split(os.sep)
        algo_name = None
        run_name = None

        # experiments 디렉토리 다음의 알고리즘 이름 찾기 (mappo, maa2c 등)
        if "experiments" in path_parts:
            experiments_idx = path_parts.index("experiments")
            if experiments_idx + 1 < len(path_parts):
                algo_name = path_parts[experiments_idx + 1]
                if experiments_idx + 2 < len(path_parts):
                    run_name = path_parts[experiments_idx + 2]

        # run_name과 algo_name이 모두 있으면 사용, 없으면 기본값
        if algo_name and run_name:
            output_dir = os.path.join("train_marllib_self", "results", algo_name, run_name)
        else:
            output_dir = "train_marllib_self/results/default"

    print("=" * 80)
    print("MARLlib 학습된 정책 vs 랜덤 정책 시각적 비교 (유연한 아키텍처)")
    print("=" * 80)
    print(f"\n✓ 사용된 Checkpoint:")
    if checkpoint_num is not None:
        print(f"  - 번호: checkpoint_{checkpoint_num:06d}")
    print(f"  - 경로: {checkpoint_path}")
    print(f"\n설정:")
    print(f"  - 에피소드 수: {num_episodes}")
    print(f"  - 시작 시드: {seed}")
    print(f"  - 출력 디렉토리: {output_dir}")
    print("=" * 80)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 환경 설정
    print("\n환경 생성 중...")
    env_config = {k: v for k, v in ENV_CONFIG.items()}
    env_trained = WildfireEnv(**env_config)
    env_random = WildfireEnv(**env_config)

    print(f"✓ 환경 생성 완료")
    print(f"  Grid size: {env_config['size']}x{env_config['size']}")
    print(f"  Agents: {env_trained.num_agents}")
    print(f"  Max steps: {env_config['max_steps']}")
    print(f"  Partial observation: {env_config.get('partial_obs', False)}")

    # 에이전트 타입과 수 파악
    num_helicopters = sum(1 for agent in env_trained.agents if agent.type == "helicopter")
    num_trucks = sum(1 for agent in env_trained.agents if agent.type == "truck")
    num_crews = sum(1 for agent in env_trained.agents if agent.type == "crew")

    print(f"에이전트 구성:")
    print(f"  - Helicopters: {num_helicopters}")
    print(f"  - Trucks: {num_trucks}")
    print(f"  - Crews: {num_crews}")

    # 정책 매핑 함수 (에이전트 ID는 문자열 인덱스 "0", "1", "2" 등)
    # agent_id를 인덱스로 변환하여 env의 agents 리스트에서 에이전트 객체를 찾음
    def create_policy_mapping_fn(env, num_helicopters, num_trucks, num_crews):
        def mapping_fn(agent_id):
            try:
                agent_idx = int(agent_id)
                if 0 <= agent_idx < len(env.agents):
                    agent_obj = env.agents[agent_idx]
                    return get_agent_policy_name(agent_obj, num_helicopters, num_trucks, num_crews)
            except (ValueError, TypeError):
                pass
            return 'helicopter_policy'  # 기본값
        return mapping_fn

    policy_mapping_fn = create_policy_mapping_fn(env_trained, num_helicopters, num_trucks, num_crews)

    # 정책 네트워크 생성 및 가중치 로드
    policy_networks = None
    is_partial_obs = env_config.get('partial_obs', False)

    print(f"\n체크포인트 로드 시도...")
    try:
        policies_weights = load_heterogeneous_policies(checkpoint_path)

        if policies_weights is not None:
            # 첫 번째 정책의 weight 구조 분석
            first_policy_name = list(policies_weights.keys())[0]
            first_weights = policies_weights[first_policy_name]

            print(f"\n  분석 중인 정책: {first_policy_name}")
            analysis = analyze_weight_structure(first_weights)
            print_weight_analysis(analysis)

            # 적응형 network 생성
            policy_networks = {}

            if 'shared' in policies_weights:
                # 호환성: shared 정책 (모든 에이전트가 같은 정책)
                print("\n  [Shared Policy 로드 중]")
                policy_networks['shared'] = create_adaptive_policy_network(analysis)
                success = load_adaptive_weights_to_network(
                    policy_networks['shared'],
                    policies_weights['shared']
                )
                if success:
                    print("  ✓ {policy_name} weights 로드 완료!".format(policy_name="shared_policy"))
                else:
                    print("  ⚠ shared_policy 로드 부분 실패했지만 진행합니다")

            else:
                # 이질적 정책들
                for policy_name in ['helicopter_policy', 'truck_policy', 'crew_policy']:
                    if policy_name in policies_weights:
                        print(f"\n  [{policy_name} 로드 중]")
                        policy_networks[policy_name] = create_adaptive_policy_network(analysis)

                        # 가중치 로드
                        success = load_adaptive_weights_to_network(
                            policy_networks[policy_name],
                            policies_weights[policy_name]
                        )

                        if success:
                            print(f"  ✓ {policy_name} weights 로드 완료!")
                        else:
                            print(f"  ⚠ {policy_name} 로드 부분 실패했지만 진행합니다")

            if not policy_networks:
                print("  ⚠ 정책 네트워크를 로드하지 못했습니다.")
                return
        else:
            print("  ⚠ Policy weights를 찾을 수 없습니다.")
            return

    except Exception as e:
        print(f"  ❌ 체크포인트 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 시드별로 비교 에피소드 실행 및 GIF 생성
    print(f"\n학습된 정책 vs 랜덤 정책 비교 GIF 생성")
    print("-" * 80)

    all_rewards_trained = []
    all_rewards_random = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"\n[{ep+1}/{num_episodes}] 에피소드 {ep+1} 생성 중 (seed={episode_seed})...")

        # 학습된 정책 실행
        print("  - 학습된 정책 실행 중...")
        trained_frames, trained_reward, trained_length = run_episode_and_render(
            env_trained, policy_networks, policy_mapping_fn, episode_seed,
            max_steps=env_config['max_steps'], policy_type="trained",
            is_partial_obs=is_partial_obs
        )
        print(f"    보상: {trained_reward:.2f}, 길이: {trained_length}")

        # 랜덤 정책 실행 (같은 시드로 동일한 초기 상태)
        print("  - 랜덤 정책 실행 중...")
        random_frames, random_reward, random_length = run_episode_and_render(
            env_random, None, None, episode_seed,
            max_steps=env_config['max_steps'], policy_type="random",
            is_partial_obs=is_partial_obs
        )
        print(f"    보상: {random_reward:.2f}, 길이: {random_length}")

        all_rewards_trained.append(trained_reward)
        all_rewards_random.append(random_reward)

        # side-by-side 결합
        print("  - 프레임 결합 중...")
        combined_frames = combine_frames_side_by_side(
            trained_frames, random_frames,
            label1=f"Trained (R={trained_reward:.1f})",
            label2=f"Random (R={random_reward:.1f})"
        )

        # GIF로 저장
        output_path = os.path.join(output_dir, f"comparison_ep{ep+1:02d}_seed{episode_seed}.gif")
        save_as_gif(combined_frames, output_path, fps=10)

    # 최종 통계 출력
    print("\n" + "=" * 80)
    print("최종 통계 (Trained vs Random)")
    print("=" * 80)
    print(f"\n평균 리워드 (Trained): {np.mean(all_rewards_trained):.2f} ± {np.std(all_rewards_trained):.2f}")
    print(f"평균 리워드 (Random): {np.mean(all_rewards_random):.2f} ± {np.std(all_rewards_random):.2f}")

    print(f"\n모든 GIF가 저장되었습니다: {output_dir}")

    # 통계 파일 저장
    stats_path = os.path.join(output_dir, f"comparison_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Trained vs Random Agent Visualization Stats (Flexible Architecture)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Partial Observation: {is_partial_obs}\n")
        f.write(f"Agent Configuration:\n")
        f.write(f"  - Helicopters: {num_helicopters}\n")
        f.write(f"  - Trucks: {num_trucks}\n")
        f.write(f"  - Crews: {num_crews}\n")
        f.write(f"Seeds: {[seed + i for i in range(num_episodes)]}\n\n")
        for i in range(num_episodes):
            episode_seed = seed + i
            f.write(f"Episode {i+1} (Seed {episode_seed}):\n")
            f.write(f"  Trained Reward: {all_rewards_trained[i]:.2f}\n")
            f.write(f"  Random Reward: {all_rewards_random[i]:.2f}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"Avg Trained Reward: {np.mean(all_rewards_trained):.2f} ± {np.std(all_rewards_trained):.2f}\n")
        f.write(f"Avg Random Reward: {np.mean(all_rewards_random):.2f} ± {np.std(all_rewards_random):.2f}\n")

    print(f"✓ 통계 저장: {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARLlib 학습된 정책과 랜덤 정책 시각적 비교 (유연한 아키텍처)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="MARLlib 체크포인트 경로. 디렉토리(run 폴더)를 지정하면 자동으로 최신 checkpoint를 선택합니다. "
             "예시 1 (권장): train_marllib_self/experiments/mappo/run4 "
             "예시 2: train_marllib_self/experiments/maa2c/run1 "
             "예시 3 (부분 관찰): train_marllib_self/experiments/mappo/run_test_global_state_flag_partial_obs "
             "예시 4: train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/.../checkpoint_000050"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="생성할 비교 GIF 수 (기본값: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="시작 랜덤 시드 (기본값: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="GIF를 저장할 디렉토리 (기본값: train_marllib_self/results/{algo}/{run_name}/)"
    )

    args = parser.parse_args()

    main(args.checkpoint, args.episodes, args.seed, args.output_dir)
