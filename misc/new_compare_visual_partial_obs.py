"""
MARLlib 학습된 정책과 랜덤 정책 시각적 비교 (Partial Observation 지원)

Partial Observation (agent_view_size)로 학습된 이질적 에이전트(헬리콥터, 트럭, 승무원)과
랜덤 에이전트의 행동을 side-by-side로 비교하는 GIF를 생성합니다.

각 에이전트 타입별로 학습된 정책을 로드하여 에이전트별로 적절한 정책을
적용하고, 학습된 정책과 랜덤 정책의 성능을 시각적으로 비교합니다.

실행 예시:
# Partial observation으로 학습된 모델 시각화
python train_marllib_self/new_compare_visual_partial_obs.py \
    --checkpoint train_marllib_self/experiments/mappo/run15_partial_obs \
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
        탐색할 디렉토리 경로 (예: train_marllib_self/experiments/mappo/run15_partial_obs)

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
# [이질적 정책 로드]
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
# [Policy Network - Partial Observation 지원]
# ============================================================================
def create_policy_network(obs_dim, action_dim):
    """
    MARLlib의 CentralizedCriticMLP 아키텍처 재구성

    Partial observation으로 학습되었을 때도 같은 구조를 사용합니다.
    관찰 차원만 다를 뿐 네트워크 구조는 동일합니다.

    Checkpoint 가중치 구조:
    - p_encoder.encoder.0._model.0.weight  (256, obs_dim)
    - p_encoder.encoder.1._model.0.weight  (256, 256)
    - p_branch._model.0.weight             (action_dim, 256)
    - vf_encoder.encoder.0._model.0.weight (256, obs_dim)
    - vf_encoder.encoder.1._model.0.weight (256, 256)
    - vf_branch._model.0.weight            (1, 256)
    - cc_vf_encoder.encoder.0._model.0.weight (256, obs_dim)
    - cc_vf_encoder.encoder.1._model.0.weight (256, 256)
    - cc_vf_branch._model.0.weight         (1, 256)
    """

    class PolicyNetwork(torch.nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
            super().__init__()

            # ===== Policy Encoder (observation -> feature) =====
            # Layer 0: obs_dim -> hidden_dim
            self.p_encoder_layer0_weight = torch.nn.Linear(obs_dim, hidden_dim)
            # Layer 1: hidden_dim -> hidden_dim
            self.p_encoder_layer1_weight = torch.nn.Linear(hidden_dim, hidden_dim)

            # ===== Policy Branch (feature -> action logits) =====
            self.p_branch_weight = torch.nn.Linear(hidden_dim, action_dim)

            # ===== Value Function Encoder =====
            # Layer 0: obs_dim -> hidden_dim
            self.vf_encoder_layer0_weight = torch.nn.Linear(obs_dim, hidden_dim)
            # Layer 1: hidden_dim -> hidden_dim
            self.vf_encoder_layer1_weight = torch.nn.Linear(hidden_dim, hidden_dim)

            # ===== Value Function Branch (feature -> scalar value) =====
            self.vf_branch_weight = torch.nn.Linear(hidden_dim, 1)

            # ===== Centralized Critic Encoder =====
            # Layer 0: obs_dim -> hidden_dim
            self.cc_vf_encoder_layer0_weight = torch.nn.Linear(obs_dim, hidden_dim)
            # Layer 1: hidden_dim -> hidden_dim
            self.cc_vf_encoder_layer1_weight = torch.nn.Linear(hidden_dim, hidden_dim)

            # ===== Centralized Critic Branch =====
            self.cc_vf_branch_weight = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x):
            """Policy network forward pass"""
            # Policy encoder: obs -> hidden features
            p_features = torch.relu(self.p_encoder_layer0_weight(x))
            p_features = torch.relu(self.p_encoder_layer1_weight(p_features))

            # Policy branch: hidden features -> action logits
            logits = self.p_branch_weight(p_features)
            return logits

        def get_action(self, obs):
            """Select action from observation"""
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = self.forward(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            return action

    return PolicyNetwork(obs_dim, action_dim)


def load_marllib_weights_to_network(network, marllib_weights):
    """
    MARLlib checkpoint 가중치를 재구성된 PolicyNetwork로 로드

    Parameters
    ----------
    network : PolicyNetwork
        로드할 네트워크
    marllib_weights : dict
        MARLlib checkpoint에서 추출한 가중치 딕셔너리

    Returns
    -------
    bool
        성공 여부
    """
    try:
        # 매핑 관계: MARLlib 키 -> 네트워크 파라미터
        weight_mapping = {
            'p_encoder.encoder.0._model.0.weight': 'p_encoder_layer0_weight.weight',
            'p_encoder.encoder.0._model.0.bias': 'p_encoder_layer0_weight.bias',
            'p_encoder.encoder.1._model.0.weight': 'p_encoder_layer1_weight.weight',
            'p_encoder.encoder.1._model.0.bias': 'p_encoder_layer1_weight.bias',
            'p_branch._model.0.weight': 'p_branch_weight.weight',
            'p_branch._model.0.bias': 'p_branch_weight.bias',
            'vf_encoder.encoder.0._model.0.weight': 'vf_encoder_layer0_weight.weight',
            'vf_encoder.encoder.0._model.0.bias': 'vf_encoder_layer0_weight.bias',
            'vf_encoder.encoder.1._model.0.weight': 'vf_encoder_layer1_weight.weight',
            'vf_encoder.encoder.1._model.0.bias': 'vf_encoder_layer1_weight.bias',
            'vf_branch._model.0.weight': 'vf_branch_weight.weight',
            'vf_branch._model.0.bias': 'vf_branch_weight.bias',
            'cc_vf_encoder.encoder.0._model.0.weight': 'cc_vf_encoder_layer0_weight.weight',
            'cc_vf_encoder.encoder.0._model.0.bias': 'cc_vf_encoder_layer0_weight.bias',
            'cc_vf_encoder.encoder.1._model.0.weight': 'cc_vf_encoder_layer1_weight.weight',
            'cc_vf_encoder.encoder.1._model.0.bias': 'cc_vf_encoder_layer1_weight.bias',
            'cc_vf_branch._model.0.weight': 'cc_vf_branch_weight.weight',
            'cc_vf_branch._model.0.bias': 'cc_vf_branch_weight.bias',
        }

        loaded_count = 0
        skipped_count = 0

        for marllib_key, network_key in weight_mapping.items():
            if marllib_key not in marllib_weights:
                print(f"    ⚠ {marllib_key} 없음 (스킵)")
                skipped_count += 1
                continue

            marllib_weight = marllib_weights[marllib_key]

            # 네트워크 파라미터에 접근 (예: p_encoder_layer0_weight.weight)
            parts = network_key.split('.')
            obj = network
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param_name = parts[-1]
            target_param = getattr(obj, param_name)

            # 크기가 맞지 않으면 스킵 (예: cc_vf_encoder layer 0)
            if target_param.shape != marllib_weight.shape:
                print(f"    ⚠ {marllib_key}: shape 불일치 {target_param.shape} vs {marllib_weight.shape} (스킵)")
                skipped_count += 1
                continue

            # 가중치 복사
            target_param.data = torch.FloatTensor(marllib_weight)
            loaded_count += 1

        if loaded_count > 0:
            print(f"    ✓ {loaded_count}개 가중치 로드 성공 ({skipped_count}개 스킵)")
            return True
        else:
            print(f"    ❌ 로드된 가중치 없음 (모두 스킵)")
            return False

    except Exception as e:
        print(f"    ❌ 가중치 로드 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# [Partial Observation 환경 생성]
# ============================================================================
def create_partial_obs_env(env_config, partial_obs=True, agent_view_size=10):
    """
    Partial observation 설정으로 환경 생성

    Parameters
    ----------
    env_config : dict
        기본 환경 설정
    partial_obs : bool
        Partial observation 활성화 여부
    agent_view_size : int
        에이전트 관찰 범위 (partial_obs=True일 때만 사용)

    Returns
    -------
    WildfireEnv
        생성된 환경
    """
    # 환경 설정 복사
    config = {k: v for k, v in env_config.items()}

    # Partial observation 설정
    config['partial_obs'] = partial_obs
    config['agent_view_size'] = agent_view_size

    env = WildfireEnv(**config)
    return env


# ============================================================================
# [에이전트 정책 매핑]
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


# ============================================================================
# [에피소드 실행 - Partial Observation 지원]
# ============================================================================
def run_episode_and_render(env, policy_networks, policy_mapping_fn, seed,
                          max_steps=300, policy_type="trained"):
    """
    에피소드를 실행하고 프레임을 수집 (이질적 정책 지원, Partial observation)

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

        for agent_id in obs_dict.keys():
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


# ============================================================================
# [프레임 결합 및 GIF 저장]
# ============================================================================
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


# ============================================================================
# [메인 함수 - Partial Observation 지원]
# ============================================================================
def main(checkpoint_path, num_episodes=3, seed=42, output_dir=None):
    """
    메인 함수: 학습된 모델과 랜덤 정책을 시각적으로 비교 (Partial observation 지원)

    이질적 에이전트(헬리콥터, 트럭, 승무원)가 각각 다른 정책을 학습했을 때,
    각 정책별로 행동을 시각화하고 랜덤 정책과 비교합니다.

    **중요**: 학습 시 partial_obs=True로 설정하여 학습한 모델을 시각화할 때는
    이 스크립트를 사용하세요. 이 스크립트는 자동으로 partial observation 환경을 생성합니다.

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
        GIF를 저장할 디렉토리. None이면 checkpoint 경로에서 run name을 추출하여
        train_marllib_self/results/mappo/{run_name}/ 에 저장
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

    # output_dir이 지정되지 않은 경우, checkpoint 경로에서 run name 추출
    if output_dir is None:
        # 경로 예: train_marllib_self/experiments/mappo/run15_partial_obs/... -> run15_partial_obs
        path_parts = os.path.normpath(checkpoint_path).split(os.sep)
        if "mappo" in path_parts:
            mappo_idx = path_parts.index("mappo")
            if mappo_idx + 1 < len(path_parts):
                run_name = path_parts[mappo_idx + 1]
                output_dir = os.path.join("train_marllib_self", "results", "mappo", run_name)
            else:
                output_dir = "train_marllib_self/results/mappo/default"
        else:
            output_dir = "train_marllib_self/results/mappo/default"

    print("=" * 80)
    print("MARLlib 학습된 정책 vs 랜덤 정책 시각적 비교")
    print("(Partial Observation 버전 - run15_partial_obs 용)")
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

    # ============================================================================
    # 환경 생성 (Partial Observation 활성화)
    # ============================================================================
    print("\n환경 생성 중 (Partial Observation 활성화)...")
    env_config = {k: v for k, v in ENV_CONFIG.items()}

    # 중요: Partial observation 활성화
    partial_obs = env_config.get('partial_obs', True)
    agent_view_size = env_config.get('agent_view_size', 10)

    print(f"  - Partial Observation: {partial_obs}")
    print(f"  - Agent View Size: {agent_view_size}")

    env_trained = create_partial_obs_env(env_config, partial_obs=partial_obs, agent_view_size=agent_view_size)
    env_random = create_partial_obs_env(env_config, partial_obs=partial_obs, agent_view_size=agent_view_size)

    print(f"✓ 환경 생성 완료")
    print(f"  Grid size: {env_config['size']}x{env_config['size']}")
    print(f"  Agents: {env_trained.num_agents}")
    print(f"  Max steps: {env_config['max_steps']}")

    # ============================================================================
    # 에이전트 정보 파악
    # ============================================================================
    num_helicopters = sum(1 for agent in env_trained.agents if agent.type == "helicopter")
    num_trucks = sum(1 for agent in env_trained.agents if agent.type == "truck")
    num_crews = sum(1 for agent in env_trained.agents if agent.type == "crew")

    print(f"에이전트 구성:")
    print(f"  - Helicopters: {num_helicopters}")
    print(f"  - Trucks: {num_trucks}")
    print(f"  - Crews: {num_crews}")

    # 정책 매핑 함수
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

    # ============================================================================
    # 정책 네트워크 생성 및 가중치 로드
    # ============================================================================
    policy_networks = None
    print(f"\n체크포인트 로드 시도...")
    try:
        policies_weights = load_heterogeneous_policies(checkpoint_path)

        if policies_weights is not None:
            # 첫 번째 에이전트로부터 observation/action 차원 추출
            first_agent_id = list(env_trained.observation_space.keys())[0]
            obs_dim = env_trained.observation_space[first_agent_id].shape[0]
            action_dim = env_trained.action_space[first_agent_id].n

            print(f"  Observation dim: {obs_dim} (Partial observation)")
            print(f"  Action dim: {action_dim}")

            # 정책 네트워크 생성
            policy_networks = {}

            if 'shared' in policies_weights:
                # 호환성: shared 정책 (모든 에이전트가 같은 정책)
                policy_networks['shared'] = create_policy_network(obs_dim, action_dim)
                try:
                    policy_networks['shared'].load_state_dict(
                        policies_weights['shared'], strict=False
                    )
                    print("  ✓ Shared policy weights 로드 완료!")
                except Exception as e:
                    print(f"  ⚠ Shared policy 로드 실패: {e}")
                    policy_networks = None
            else:
                # 이질적 정책들
                for policy_name in ['helicopter_policy', 'truck_policy', 'crew_policy']:
                    if policy_name in policies_weights:
                        policy_networks[policy_name] = create_policy_network(obs_dim, action_dim)

                        # 새로운 가중치 로드 함수 사용 (MARLlib 형식에 맞게)
                        success = load_marllib_weights_to_network(
                            policy_networks[policy_name],
                            policies_weights[policy_name]
                        )

                        if success:
                            print(f"  ✓ {policy_name} weights 로드 완료!")
                        else:
                            print(f"  ⚠ {policy_name} 로드 부분 성공 또는 실패")

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

    # ============================================================================
    # 시드별로 비교 에피소드 실행 및 GIF 생성
    # ============================================================================
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
            max_steps=env_config['max_steps'], policy_type="trained"
        )
        print(f"    보상: {trained_reward:.2f}, 길이: {trained_length}")

        # 랜덤 정책 실행 (같은 시드로 동일한 초기 상태)
        print("  - 랜덤 정책 실행 중...")
        random_frames, random_reward, random_length = run_episode_and_render(
            env_random, None, None, episode_seed,
            max_steps=env_config['max_steps'], policy_type="random"
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

    # ============================================================================
    # 최종 통계 출력 및 저장
    # ============================================================================
    print("\n" + "=" * 80)
    print("최종 통계 (Trained vs Random) - Partial Observation")
    print("=" * 80)
    print(f"\n평균 리워드 (Trained): {np.mean(all_rewards_trained):.2f} ± {np.std(all_rewards_trained):.2f}")
    print(f"평균 리워드 (Random): {np.mean(all_rewards_random):.2f} ± {np.std(all_rewards_random):.2f}")
    print(f"개선도: {np.mean(all_rewards_trained) - np.mean(all_rewards_random):.2f}")

    print(f"\n모든 GIF가 저장되었습니다: {output_dir}")

    # 통계 파일 저장
    stats_path = os.path.join(output_dir, f"comparison_stats_partial_obs.txt")
    with open(stats_path, 'w') as f:
        f.write("Trained vs Random Agent Visualization Stats\n")
        f.write("(Partial Observation - run15_partial_obs)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Environment Configuration:\n")
        f.write(f"  - Grid Size: {env_config['size']}x{env_config['size']}\n")
        f.write(f"  - Partial Observation: {partial_obs}\n")
        f.write(f"  - Agent View Size: {agent_view_size}\n")
        f.write(f"  - Max Steps: {env_config['max_steps']}\n")
        f.write(f"\nAgent Configuration:\n")
        f.write(f"  - Helicopters: {num_helicopters}\n")
        f.write(f"  - Trucks: {num_trucks}\n")
        f.write(f"  - Crews: {num_crews}\n")
        f.write(f"\nNetwork Configuration:\n")
        f.write(f"  - Observation Dim: {env_trained.observation_space[first_agent_id].shape[0]}\n")
        f.write(f"  - Action Dim: {env_trained.action_space[first_agent_id].n}\n")
        f.write(f"\nSeeds: {[seed + i for i in range(num_episodes)]}\n\n")
        for i in range(num_episodes):
            episode_seed = seed + i
            f.write(f"Episode {i+1} (Seed {episode_seed}):\n")
            f.write(f"  Trained Reward: {all_rewards_trained[i]:.2f}\n")
            f.write(f"  Random Reward: {all_rewards_random[i]:.2f}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"Avg Trained Reward: {np.mean(all_rewards_trained):.2f} ± {np.std(all_rewards_trained):.2f}\n")
        f.write(f"Avg Random Reward: {np.mean(all_rewards_random):.2f} ± {np.std(all_rewards_random):.2f}\n")
        f.write(f"Improvement: {np.mean(all_rewards_trained) - np.mean(all_rewards_random):.2f}\n")

    print(f"✓ 통계 저장: {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MARLlib 학습된 정책과 랜덤 정책 시각적 비교 (Partial Observation 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  # Partial observation으로 학습된 모델 시각화 (권장)
  python train_marllib_self/new_compare_visual_partial_obs.py \\
      --checkpoint train_marllib_self/experiments/mappo/run15_partial_obs \\
      --episodes 3 \\
      --seed 42
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="MARLlib 체크포인트 경로. 디렉토리(run 폴더)를 지정하면 자동으로 최신 checkpoint를 선택합니다. "
             "예시: train_marllib_self/experiments/mappo/run15_partial_obs"
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
        help="GIF를 저장할 디렉토리 (기본값: train_marllib_self/results/mappo/{run_name}/)"
    )

    args = parser.parse_args()

    main(args.checkpoint, args.episodes, args.seed, args.output_dir)
