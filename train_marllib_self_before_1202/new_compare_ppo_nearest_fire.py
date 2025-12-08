"""
MARLlib PPO 학습된 정책과 가장 가까운 활화목 휴리스틱 정책 시각적 비교

MARLlib PPO로 학습된 이질적 에이전트(헬리콥터, 트럭, 승무원)와
가장 가까운 활화목으로 이동하는 휴리스틱 에이전트의 행동을 side-by-side로 비교하는 GIF를 생성합니다.

주요 기능:
- Checkpoint의 weight 구조를 자동으로 분석하여 적응형 network 구성
- Full observation과 partial observation 자동 감지
- 이질적 에이전트 지원
- Centralized Critic 미사용 (PPO는 decentralized 구조)
- Weight 로드 실패 시 상세 보고
- 가장 가까운 활화목 휴리스틱과의 비교

실행 예시:
# Global state를 사용하는 경우 (full observation)
python train_marllib_self/new_compare_ppo_nearest_fire.py \
    --checkpoint train_marllib_self/experiments/ppo/run1 \
    --episodes 3 \
    --seed 42

# Partial observation을 사용하는 경우
python train_marllib_self/new_compare_ppo_nearest_fire.py \
    --checkpoint train_marllib_self/experiments/ppo/run_test_partial_obs \
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
from train_marllib_self.heuristic_nearest_fire import select_heuristic_action
from wildfire_environment.envs import WildfireEnv
from wildfire_environment.core.constants import TILE_PIXELS
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# [Weight 구조 분석]
# ============================================================================
def analyze_weight_structure(marllib_weights, is_partial_obs=False):
    """
    Checkpoint의 weight 구조를 분석하여 network 구조를 파악

    Parameters
    ----------
    marllib_weights : dict
        MARLlib checkpoint에서 추출한 가중치 딕셔너리
    is_partial_obs : bool, optional
        환경 설정에서의 partial observation 여부, by default False

    Returns
    -------
    dict
        분석 결과:
        {
            'input_dim': int,           # Policy encoder의 input dimension
            'hidden_dim': int,          # Hidden layer dimension
            'output_dim': int,          # Policy branch의 output dimension
            'is_partial_obs': bool,     # Partial observation 여부
            'encoder_layers': list,     # Encoder layer 구조
        }
    """
    analysis = {
        'input_dim': None,
        'hidden_dim': None,
        'output_dim': None,
        'is_partial_obs': is_partial_obs,
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
        탐색할 디렉토리 경로 (예: train_marllib_self/experiments/ppo/run1)

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
        print(f"     {directory_path}/ppo_mlp_wildfire-ma/PPOTrainer_.../checkpoint_XXXXX")
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
# [적응형 Policy Network 생성 (PPO용 - Centralized Critic 없음)]
# ============================================================================
def create_adaptive_policy_network(analysis):
    """
    Weight 분석 결과를 기반으로 적응형 network 생성 (PPO용)

    PPO는 centralized critic을 사용하지 않으므로, policy encoder와 value encoder만 구성합니다.

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

            # PPO에는 centralized critic이 없음 (decentralized 구조)

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
                action = torch.argmax(logits, dim=-1).item()
            return action

    return AdaptivePolicyNetwork()


def load_adaptive_weights_to_network(network, marllib_weights):
    """
    MARLlib checkpoint 가중치를 적응형 network로 로드 (PPO용)

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


def run_ppo_episode_and_render(env, policy_networks, policy_mapping_fn, seed,
                              max_steps=300, is_partial_obs=False):
    """
    PPO 정책으로 에피소드를 실행하고 프레임을 수집 (이질적 정책 지원, 부분 관찰)

    Parameters
    ----------
    env : WildfireEnv
        환경
    policy_networks : dict
        정책 네트워크 딕셔너리
        - {'helicopter_policy': network, 'truck_policy': network, ...}
        - {'shared': network}
    policy_mapping_fn : callable or None
        에이전트ID → 정책이름 매핑 함수
    seed : int
        랜덤 시드
    max_steps : int
        최대 스텝 수
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

    # 각 에이전트의 활동 시간 게이지 추가
    for agent in env.agents:
        frame = render_activity_gauge(frame, agent)

    # # 각 에이전트의 물 게이지 추가
    # for agent in env.agents:
    #     frame = render_water_gauge(frame, agent)

    # 각 에이전트의 급수원 마커 추가
    for agent in env.agents:
        frame = render_supply_source_marker(frame, agent)

    # # 에이전트 상태 패널 추가
    # frame = render_agent_status_panel(frame, env.agents, step)

    # Step 수 표시 추가
    frame = render_step_counter(frame, step)

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

            actions[agent_id] = action

        # 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        # 리워드 합산
        episode_reward += np.mean(list(reward_dict.values()))

        # 프레임 저장
        frame = env.render(mode='rgb_array')

        # 각 에이전트의 활동 시간 게이지 추가
        for agent in env.agents:
            frame = render_activity_gauge(frame, agent)

        # # 각 에이전트의 물 게이지 추가
        # for agent in env.agents:
        #     frame = render_water_gauge(frame, agent)

        # 각 에이전트의 급수원 마커 추가
        for agent in env.agents:
            frame = render_supply_source_marker(frame, agent)

        # 에이전트 상태 패널 추가
        # frame = render_agent_status_panel(frame, env.agents, step)

        frames.append(frame)

        step += 1

    return frames, episode_reward, step


def run_heuristic_episode_and_render(env, env_config, seed, max_steps=300, is_partial_obs=False):
    """
    휴리스틱 정책(가장 가까운 활화목)으로 에피소드를 실행하고 프레임을 수집

    Parameters
    ----------
    env : WildfireEnv
        환경
    env_config : dict
        환경 설정
    seed : int
        랜덤 시드
    max_steps : int
        최대 스텝 수
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

    # 각 에이전트의 활동 시간 게이지 추가
    for agent in env.agents:
        frame = render_activity_gauge(frame, agent)

    # # 각 에이전트의 물 게이지 추가
    # for agent in env.agents:
    #     frame = render_water_gauge(frame, agent)

    # 각 에이전트의 급수원 마커 추가
    for agent in env.agents:
        frame = render_supply_source_marker(frame, agent)

    # # 에이전트 상태 패널 추가
    # frame = render_agent_status_panel(frame, env.agents, step)

    # Step 수 표시 추가
    frame = render_step_counter(frame, step)

    frames.append(frame)

    while not done and step < max_steps:
        # 각 에이전트의 액션 선택
        actions = {}
        for agent_id in obs_dict.keys():
            obs = obs_dict[agent_id]
            action = select_heuristic_action(env, agent_id, obs, obs_dict, env_config, is_partial_obs)
            actions[agent_id] = action

        # 스텝 실행
        obs_dict, reward_dict, terminated, truncated, info_dict = env.step(actions)
        done = terminated or truncated

        # 리워드 합산
        episode_reward += np.mean(list(reward_dict.values()))

        # 프레임 저장
        frame = env.render(mode='rgb_array')

        # 각 에이전트의 활동 시간 게이지 추가
        for agent in env.agents:
            frame = render_activity_gauge(frame, agent)

        # 각 에이전트의 물 게이지 추가
        # for agent in env.agents:
        #     frame = render_water_gauge(frame, agent)

        # 각 에이전트의 급수원 마커 추가
        for agent in env.agents:
            frame = render_supply_source_marker(frame, agent)

        # 에이전트 상태 패널 추가
        # frame = render_agent_status_panel(frame, env.agents, step)

        frames.append(frame)

        step += 1

    return frames, episode_reward, step


def combine_frames_side_by_side(frames1, frames2, label1="PPO", label2="Heuristic"):
    """
    두 프레임 시퀀스를 side-by-side로 결합

    한쪽이 먼저 종료되어도 그 화면은 정지된 상태로 유지되고,
    나머지 한쪽은 에피소드가 끝날 때까지 진행됩니다.

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
    # 더 긴 시퀀스에 맞춤
    max_len = max(len(frames1), len(frames2))

    # 짧은 쪽의 마지막 프레임을 반복하여 길이 맞춤
    if len(frames1) < max_len:
        # frames1의 마지막 프레임 반복
        frames1 = list(frames1) + [frames1[-1]] * (max_len - len(frames1))
    if len(frames2) < max_len:
        # frames2의 마지막 프레임 반복
        frames2 = list(frames2) + [frames2[-1]] * (max_len - len(frames2))

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


def render_activity_gauge(frame, agent, tile_size=TILE_PIXELS):
    """
    에이전트의 활동 시간 게이지를 에이전트 위쪽에 렌더링

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agent : Agent
        에이전트 객체
    tile_size : int
        타일 크기 (픽셀)

    Returns
    -------
    numpy array
        게이지가 추가된 이미지 배열
    """
    if not hasattr(agent, 'max_active_time') or agent.max_active_time == 0:
        return frame

    # 활동 시간 비율
    fill_ratio = agent.active_time_remaining / agent.max_active_time

    # 게이지 위치 (에이전트 위쪽 2픽셀)
    pos = agent.pos
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 2

    # 게이지 배경 (진회색)
    frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]

    # 게이지 채우기 (시안색)
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = [0, 255, 255]

    return frame


def render_water_gauge(frame, agent, tile_size=TILE_PIXELS):
    """
    에이전트의 물/억제제 게이지를 에이전트 아래쪽에 렌더링

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agent : Agent
        에이전트 객체
    tile_size : int
        타일 크기 (픽셀)

    Returns
    -------
    numpy array
        게이지가 추가된 이미지 배열
    """
    if not hasattr(agent, 'max_water') or agent.max_water == 0:
        return frame

    # 물 양 비율
    fill_ratio = agent.water_remaining / agent.max_water

    # 게이지 위치 (에이전트 아래쪽 6픽셀)
    pos = agent.pos
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 6

    # 게이지 배경 (진회색)
    frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]

    # 게이지 채우기 (파란색)
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        frame[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = [0, 100, 255]

    return frame


def render_supply_source_marker(frame, agent, tile_size=TILE_PIXELS):
    """
    급수원(홈 위치)을 프레임에 파란색 상자로 표시

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agent : Agent
        에이전트 객체
    tile_size : int
        타일 크기 (픽셀)

    Returns
    -------
    numpy array
        급수원 마커가 추가된 이미지 배열
    """
    if not hasattr(agent, 'home_pos'):
        return frame

    home_pos = agent.home_pos
    # 급수원 타일의 시작점
    home_x = home_pos[0] * tile_size
    home_y = home_pos[1] * tile_size

    # 급수원을 파란색 상자로 표시
    box_thickness = 2

    # 상단 테두리
    frame[max(0, home_y):min(frame.shape[0], home_y+box_thickness),
          max(0, home_x):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]
    # 하단 테두리
    frame[max(0, home_y+tile_size-box_thickness):min(frame.shape[0], home_y+tile_size),
          max(0, home_x):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]
    # 좌측 테두리
    frame[max(0, home_y):min(frame.shape[0], home_y+tile_size),
          max(0, home_x):min(frame.shape[1], home_x+box_thickness)] = [0, 0, 255]
    # 우측 테두리
    frame[max(0, home_y):min(frame.shape[0], home_y+tile_size),
          max(0, home_x+tile_size-box_thickness):min(frame.shape[1], home_x+tile_size)] = [0, 0, 255]

    return frame


def render_step_counter(frame, step):
    """
    상단에 Step 수를 표시

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    step : int
        현재 스텝

    Returns
    -------
    numpy array
        Step 카운터가 추가된 이미지 배열
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    # 상단 중앙에 Step 표시
    step_text = f"Step: {step}"
    text_x = frame.shape[1] // 2 - 20
    text_y = 5

    draw.text((text_x, text_y), step_text, fill=(255, 255, 255), font=font)

    return np.array(img)


def render_agent_status_panel(frame, agents, step):
    """
    에이전트들의 상태를 화면 우측에 패널로 표시

    Parameters
    ----------
    frame : numpy array
        RGB 이미지 배열
    agents : list
        에이전트 리스트
    step : int
        현재 스텝

    Returns
    -------
    numpy array
        상태 패널이 추가된 이미지 배열
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    # 우측 패널 위치
    panel_width = 200
    panel_height = frame.shape[0]
    panel_x = frame.shape[1] - panel_width
    panel_y = 0

    # 패널 배경 (반투명 검은색 효과를 위해 직접 처리)
    # 상태 텍스트 추가
    y_offset = 10

    # 스텝 표시
    draw.text((panel_x + 5, y_offset), f"Step: {step}", fill=(255, 255, 255), font=font)
    y_offset += 15

    # 각 에이전트 상태
    state_names = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}

    for agent_idx, agent in enumerate(agents):
        # 에이전트 번호
        agent_text = f"Agent {agent_idx}:"
        draw.text((panel_x + 5, y_offset), agent_text, fill=(255, 255, 255), font=font)
        y_offset += 12

        # 상태
        state = state_names.get(agent.state, "UNKNOWN")
        state_color = (0, 255, 0) if agent.state == 0 else (255, 255, 0) if agent.state == 1 else (255, 0, 0)
        draw.text((panel_x + 10, y_offset), f"State: {state}", fill=state_color, font=font)
        y_offset += 12

        # 활동 시간 (있을 경우)
        if hasattr(agent, 'max_active_time'):
            time_percent = int(100 * agent.active_time_remaining / agent.max_active_time)
            draw.text((panel_x + 10, y_offset), f"Active: {agent.active_time_remaining}/{agent.max_active_time}",
                     fill=(200, 200, 200), font=font)
            y_offset += 12

        # 재충전 시간 (RECHARGING 상태일 때만)
        if agent.state == 2 and hasattr(agent, 'recharge_time'):
            recharge_percent = int(100 * (agent.recharge_time - agent.recharge_time_remaining) / agent.recharge_time)
            draw.text((panel_x + 10, y_offset), f"Recharge: {agent.recharge_time_remaining}/{agent.recharge_time}",
                     fill=(255, 165, 0), font=font)
            y_offset += 12

        # 급수원 위치
        if hasattr(agent, 'home_pos'):
            draw.text((panel_x + 10, y_offset), f"Home: {agent.home_pos}",
                     fill=(0, 255, 0), font=font)
            y_offset += 12

        y_offset += 5  # 에이전트 사이의 간격

    return np.array(img)


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
    메인 함수: 학습된 PPO 모델과 가장 가까운 활화목 휴리스틱을 시각적으로 비교

    이질적 에이전트(헬리콥터, 트럭, 승무원)가 각각 다른 정책을 학습했을 때,
    부분 관찰 환경에서도 자동으로 적응하여 각 정책별로 행동을 시각화하고
    휴리스틱 정책과 비교합니다.

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
        # 경로 예: train_marllib_self/experiments/ppo/run_test_partial_obs/...
        path_parts = os.path.normpath(checkpoint_path).split(os.sep)
        algo_name = None
        run_name = None

        # experiments 디렉토리 다음의 알고리즘 이름 찾기 (ppo, ppo_clip 등)
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
    print("MARLlib PPO 학습된 정책 vs 가장 가까운 활화목 휴리스틱 시각적 비교")
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
    env_ppo = WildfireEnv(**env_config)
    env_heuristic = WildfireEnv(**env_config)

    print(f"✓ 환경 생성 완료")
    print(f"  Grid size: {env_config['size']}x{env_config['size']}")
    print(f"  Agents: {env_ppo.num_agents}")
    print(f"  Max steps: {env_config['max_steps']}")
    print(f"  Partial observation: {env_config.get('partial_obs', False)}")

    # 에이전트 타입과 수 파악
    num_helicopters = sum(1 for agent in env_ppo.agents if agent.type == "helicopter")
    num_trucks = sum(1 for agent in env_ppo.agents if agent.type == "truck")
    num_crews = sum(1 for agent in env_ppo.agents if agent.type == "crew")

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

    policy_mapping_fn = create_policy_mapping_fn(env_ppo, num_helicopters, num_trucks, num_crews)

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
            analysis = analyze_weight_structure(first_weights, is_partial_obs=is_partial_obs)
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
                    print("  ✓ shared_policy weights 로드 완료!")
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
    print(f"\nPPO 정책 vs 휴리스틱 정책 비교 GIF 생성")
    print("-" * 80)

    all_rewards_ppo = []
    all_rewards_heuristic = []

    for ep in range(num_episodes):
        episode_seed = seed + ep
        print(f"\n[{ep+1}/{num_episodes}] 에피소드 {ep+1} 생성 중 (seed={episode_seed})...")

        # PPO 정책 실행
        print("  - PPO 정책 실행 중...")
        ppo_frames, ppo_reward, ppo_length = run_ppo_episode_and_render(
            env_ppo, policy_networks, policy_mapping_fn, episode_seed,
            max_steps=env_config['max_steps'],
            is_partial_obs=is_partial_obs
        )
        print(f"    보상: {ppo_reward:.2f}, 길이: {ppo_length}")

        # 휴리스틱 정책 실행 (같은 시드로 동일한 초기 상태)
        print("  - 휴리스틱 정책 실행 중...")
        heuristic_frames, heuristic_reward, heuristic_length = run_heuristic_episode_and_render(
            env_heuristic, env_config, episode_seed,
            max_steps=env_config['max_steps'],
            is_partial_obs=is_partial_obs
        )
        print(f"    보상: {heuristic_reward:.2f}, 길이: {heuristic_length}")

        all_rewards_ppo.append(ppo_reward)
        all_rewards_heuristic.append(heuristic_reward)

        # side-by-side 결합
        print("  - 프레임 결합 중...")
        combined_frames = combine_frames_side_by_side(
            ppo_frames, heuristic_frames,
            label1=f"PPO (R={ppo_reward:.1f})",
            label2=f"Heuristic (R={heuristic_reward:.1f})"
        )

        # GIF로 저장
        output_path = os.path.join(output_dir, f"comparison_ep{ep+1:02d}_seed{episode_seed}.gif")
        save_as_gif(combined_frames, output_path, fps=10)

    # 최종 통계 출력
    print("\n" + "=" * 80)
    print("최종 통계 (PPO vs Heuristic)")
    print("=" * 80)
    print(f"\n평균 리워드 (PPO): {np.mean(all_rewards_ppo):.2f} ± {np.std(all_rewards_ppo):.2f}")
    print(f"평균 리워드 (Heuristic): {np.mean(all_rewards_heuristic):.2f} ± {np.std(all_rewards_heuristic):.2f}")

    print(f"\n모든 GIF가 저장되었습니다: {output_dir}")

    # 통계 파일 저장
    stats_path = os.path.join(output_dir, f"comparison_stats_ppo_heuristic.txt")
    with open(stats_path, 'w') as f:
        f.write("PPO vs Nearest Fire Heuristic Agent Visualization Stats\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Algorithm: PPO (decentralized)\n")
        f.write(f"Comparison: PPO vs Nearest Fire Heuristic\n")
        f.write(f"Partial Observation: {is_partial_obs}\n")
        f.write(f"Agent Configuration:\n")
        f.write(f"  - Helicopters: {num_helicopters}\n")
        f.write(f"  - Trucks: {num_trucks}\n")
        f.write(f"  - Crews: {num_crews}\n")
        f.write(f"Seeds: {[seed + i for i in range(num_episodes)]}\n\n")
        for i in range(num_episodes):
            episode_seed = seed + i
            f.write(f"Episode {i+1} (Seed {episode_seed}):\n")
            f.write(f"  PPO Reward: {all_rewards_ppo[i]:.2f}\n")
            f.write(f"  Heuristic Reward: {all_rewards_heuristic[i]:.2f}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"Avg PPO Reward: {np.mean(all_rewards_ppo):.2f} ± {np.std(all_rewards_ppo):.2f}\n")
        f.write(f"Avg Heuristic Reward: {np.mean(all_rewards_heuristic):.2f} ± {np.std(all_rewards_heuristic):.2f}\n")

    print(f"✓ 통계 저장: {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARLlib PPO 학습된 정책과 가장 가까운 활화목 휴리스틱 정책 시각적 비교")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="MARLlib 체크포인트 경로. 디렉토리(run 폴더)를 지정하면 자동으로 최신 checkpoint를 선택합니다. "
             "예시 1 (권장): train_marllib_self/experiments/ppo/run1 "
             "예시 2: train_marllib_self/experiments/ppo/run_test_partial_obs "
             "예시 3: train_marllib_self/experiments/ppo/run1/ppo_mlp_wildfire-ma/.../checkpoint_000050"
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
