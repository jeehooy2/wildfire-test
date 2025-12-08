"""
Train Discrete MADDPG agents on wildfire suppression environment

Discrete action space를 지원하는 MADDPG 구현:
- Action space를 Discrete(9)에서 Box(9,)로 변환
- Policy output을 logits로 해석
- Step에서 argmax로 discrete action으로 변환
"""

import os
os.environ["TUNE_REPORTER_INTERVAL_S"] = "60"

import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from train_marllib_self.new_wrapper import WildfireRLlibEnv
from train_marllib_self.environment import ENV_CONFIG
from train_marllib_self.new_wrapper_maddpg import WildfireRLlibEnvDiscreteToContinuous

import argparse
from datetime import datetime
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Type, Union
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.ddpg.ddpg_torch_policy import (
    TargetNetworkMixin,
    ComputeTDErrorMixin
)
from marllib.marl.algos.utils.centralized_Q import (
    CentralizedQValueMixin,
    centralized_critic_q,
    before_learn_on_batch
)
from marllib.marl.algos.core.IL.ddpg import (
    IDDPG_DEFAULT_CONFIG,
    IDDPGTorchPolicy,
    IDDPGTrainer,
    IDDPGTorchModel
)
from ray.rllib.models.catalog import ModelCatalog
import copy

torch, nn = try_import_torch()


# ============================================================================
# 1. Action Space Wrapper (Discrete -> Continuous)
# ============================================================================

# ============================================================================
# 2. Discrete MADDPG Model
# ============================================================================

class DiscreteMaddpgTorchModel(IDDPGTorchModel):
    """
    Discrete action space를 지원하는 MADDPG 모델.

    MADDPG 구조 (MARLlib 참조):
    - Policy network: obs -> logits (num_actions,)
    - Q-network: obs + actions -> Q-value
    - Twin Q-network (optional): obs + actions -> Q-value
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 policy_model_config=None, q_model_config=None, twin_q=False,
                 add_layer_norm=False, **kwargs):
        """
        Parameters
        ----------
        action_space : gym.Space
            Box action space (continuous, shape=(num_actions,))
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name,
                         policy_model_config, q_model_config, twin_q, add_layer_norm)

        # Discrete action 개수 추출 (Box space에서)
        self.num_actions = int(action_space.shape[0])
        print(f"  Discrete MADDPG Model initialized with {self.num_actions} actions")


# ============================================================================
# 3. Discrete MADDPG Loss Function
# ============================================================================

def discrete_maddpg_loss(policy: Policy, model: ModelV2,
                        dist_class: ActionDistribution,
                        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """
    Discrete MADDPG 손실 함수 (MARLlib MADDPG 기반).

    Continuous MADDPG와 동일하지만, action handling을 discrete에 맞게 조정.
    """
    CentralizedQValueMixin.__init__(policy)
    target_model = policy.target_models[model]

    i = 0
    state_batches = []
    while "state_in_{}".format(i) in train_batch:
        state_batches.append(train_batch["state_in_{}".format(i)])
        i += 1
    assert state_batches
    seq_lens = train_batch.get(SampleBatch.SEQ_LENS)

    twin_q = policy.config["twin_q"]
    gamma = policy.config["gamma"]
    n_step = policy.config["n_step"]
    use_huber = policy.config["use_huber"]
    huber_threshold = policy.config["huber_threshold"]
    l2_reg = policy.config["l2_reg"]

    input_dict = {
        "obs": train_batch[SampleBatch.CUR_OBS],
        "state": train_batch["state"],
        "is_training": True,
        "prev_actions": train_batch[SampleBatch.PREV_ACTIONS],
        "opponent_actions": train_batch["opponent_actions"],
        "prev_opponent_actions": train_batch["prev_opponent_actions"],
        "prev_rewards": train_batch[SampleBatch.PREV_REWARDS],
    }
    model_out_t, state_in_t = model(input_dict, state_batches, seq_lens)
    states_in_t = model.select_state(state_in_t, ["policy", "q", "twin_q"])

    input_dict_next = {
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "state": train_batch["new_state"],
        "is_training": True,
        "prev_actions": train_batch[SampleBatch.ACTIONS],
        "opponent_actions": train_batch["next_opponent_actions"],
        "prev_opponent_actions": train_batch["opponent_actions"],
        "prev_rewards": train_batch[SampleBatch.REWARDS],
    }

    target_model_out_tp1, target_state_in_tp1 = target_model(
        input_dict_next, state_batches, seq_lens)
    target_states_in_tp1 = target_model.select_state(target_state_in_tp1,
                                                     ["policy", "q", "twin_q"])

    # Policy network evaluation (logits for discrete actions)
    policy_t = model.get_policy_output(
        model_out_t, states_in_t["policy"], seq_lens)[0]

    policy_tp1 = target_model.get_policy_output(
        target_model_out_tp1, target_states_in_tp1["policy"], seq_lens)[0]

    # Discrete action handling: Convert logits to action probabilities
    # Use softmax on logits to get valid probabilities
    policy_tp1_probs = F.softmax(policy_tp1, dim=-1)

    # Q-net(s) evaluation
    q_t = model.get_cc_q_values(
        model_out_t, states_in_t["q"], seq_lens, train_batch[SampleBatch.ACTIONS])[0]

    # Q-values for current policy
    q_t_det_policy = model.get_cc_q_values(
        model_out_t, states_in_t["q"], seq_lens, policy_t)[0]
    q_t_det_policy = torch.squeeze(input=q_t_det_policy, axis=len(q_t_det_policy.shape) - 1)

    if twin_q:
        twin_q_t = model.get_twin_q_values(model_out_t, states_in_t["twin_q"], seq_lens,
                                           train_batch[SampleBatch.ACTIONS])[0]

    # Target q-net(s) evaluation
    q_tp1 = target_model.get_cc_q_values(
        target_model_out_tp1, target_states_in_tp1["q"], seq_lens, policy_tp1_probs)[0]

    if twin_q:
        twin_q_tp1 = target_model.get_twin_q_values(target_model_out_tp1, target_states_in_tp1["twin_q"], seq_lens,
                                                    policy_tp1_probs)[0]

    q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
    if twin_q:
        twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 = torch.min(q_tp1, twin_q_tp1)

    q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
    q_tp1_best_masked = \
        (1.0 - train_batch[SampleBatch.DONES].float()) * \
        q_tp1_best

    # Compute Bellman target
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
                           gamma ** n_step * q_tp1_best_masked).detach()

    # BURNIN
    from ray.rllib.utils.torch_utils import sequence_mask
    from ray.rllib.utils.numpy import PRIO_WEIGHTS

    B = state_batches[0].shape[0]
    T = q_t_selected.shape[0] // B
    seq_mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], T)
    burn_in = policy.config["burn_in"]
    if burn_in > 0 and burn_in < T:
        seq_mask[:, :burn_in] = False

    seq_mask = seq_mask.reshape(-1)
    num_valid = torch.sum(seq_mask)

    def reduce_mean_valid(t):
        return torch.sum(t[seq_mask]) / num_valid

    # Compute error
    if twin_q:
        from ray.rllib.agents.ddpg.ddpg_torch_policy import huber_loss, l2_loss

        td_error = q_t_selected - q_t_selected_target
        td_error = td_error * seq_mask
        twin_td_error = twin_q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold) \
                     + huber_loss(twin_td_error, huber_threshold)
        else:
            errors = 0.5 * \
                     (torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0))
    else:
        from ray.rllib.agents.ddpg.ddpg_torch_policy import huber_loss, l2_loss

        td_error = q_t_selected - q_t_selected_target
        td_error = td_error * seq_mask
        if use_huber:
            errors = huber_loss(td_error, huber_threshold)
        else:
            errors = 0.5 * torch.pow(td_error, 2.0)

    critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)
    actor_loss = -torch.mean(q_t_det_policy * seq_mask)

    # Add l2-regularization
    if l2_reg is not None:
        from ray.rllib.agents.ddpg.ddpg_torch_policy import l2_loss

        for name, var in model.policy_variables(as_dict=True).items():
            if "bias" not in name:
                actor_loss += (l2_reg * l2_loss(var))
        for name, var in model.q_variables(as_dict=True).items():
            if "bias" not in name:
                critic_loss += (l2_reg * l2_loss(var))

    # Store stats
    model.tower_stats["q_t"] = q_t * seq_mask[..., None]
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["td_error"] = td_error

    # Return two loss terms
    return actor_loss, critic_loss


# ============================================================================
# 4. Discrete MADDPG Policy and Trainer
# ============================================================================

def build_discrete_maddpg_models_and_action_dist(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, ActionDistribution]:
    """Discrete MADDPG 모델과 액션 분포 생성"""

    from ray.rllib.models.distributions import Deterministic

    num_outputs = int(np.product(obs_space.shape))
    from ray.rllib.models.torch.misc import normc_initializer
    from ray.rllib.agents.ddpg.ddpg_torch_policy import MODEL_DEFAULTS

    policy_model_config = MODEL_DEFAULTS.copy()
    policy_model_config.update(config["policy_model"])
    q_model_config = MODEL_DEFAULTS.copy()
    q_model_config.update(config["Q_model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=DiscreteMaddpgTorchModel,
        name="discrete_maddpg_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        add_layer_norm=(config.get("exploration_config", {}).get("type") == "ParameterNoise"),
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=DiscreteMaddpgTorchModel,
        name="discrete_maddpg_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        add_layer_norm=(config.get("exploration_config", {}).get("type") == "ParameterNoise"),
    )

    assert policy.model.get_initial_state() != [], \
        "Discrete MADDPG requires recurrent model!"

    from ray.rllib.models.distributions import Deterministic
    return policy.model, Deterministic


def vf_preds_fetches_discrete(policy, input_dict, state_batches, model, action_dist):
    """Value function 예측 (MADDPG는 deterministic이므로 empty)"""
    return dict()


# Discrete MADDPG Policy 클래스 생성
DiscreteMaddpgTorchPolicy = IDDPGTorchPolicy.with_updates(
    name="DiscreteMaddpgTorchPolicy",
    postprocess_fn=centralized_critic_q,
    extra_action_out_fn=vf_preds_fetches_discrete,
    make_model_and_action_dist=build_discrete_maddpg_models_and_action_dist,
    loss_fn=discrete_maddpg_loss,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        CentralizedQValueMixin
    ]
)


def get_policy_class_discrete(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    """정책 클래스 반환"""
    if config["framework"] == "torch":
        return DiscreteMaddpgTorchPolicy


def validate_config_discrete(config: TrainerConfigDict) -> None:
    """설정 검증"""
    config["replay_sequence_length"] = \
        config["burn_in"] + config["model"]["max_seq_len"]

    def f(batch, workers, config):
        policies = dict(workers.local_worker()
                        .foreach_trainable_policy(lambda p, i: (i, p)))
        return before_learn_on_batch(batch, policies,
                                     config["train_batch_size"])

    config["before_learn_on_batch"] = f


# Discrete MADDPG Trainer
DiscreteMaddpgTrainer = IDDPGTrainer.with_updates(
    name="DiscreteMaddpgTrainer",
    default_config=IDDPG_DEFAULT_CONFIG,
    default_policy=DiscreteMaddpgTorchPolicy,
    get_policy_class=get_policy_class_discrete,
    validate_config=validate_config_discrete,
    allow_unknown_subkeys=["Q_model", "policy_model"]
)


# ============================================================================
# 5. Setup and Training
# ============================================================================

def setup_marllib_config():
    """MARLlib 패키지 내부에 wildfire.yaml 설정 파일을 자동 생성"""
    import marllib
    marllib_path = Path(marllib.__file__).parent
    config_dir = marllib_path.parent / "examples" / "config" / "env_config"

    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_file = config_dir / "wildfire-ma.yaml"

    yaml_content = f"""# Wildfire environment configuration for MARLlib
# Auto-generated by new_train_maddpg.py

env: wildfire-ma

env_args:
  size: {ENV_CONFIG['size']}
  num_agents: {ENV_CONFIG['num_agents']}
  max_steps: {ENV_CONFIG['max_steps']}
  initial_fire_size: {ENV_CONFIG['initial_fire_size']}
  num_helicopters: {ENV_CONFIG['num_helicopters']}
  num_trucks: {ENV_CONFIG['num_trucks']}
  num_crews: {ENV_CONFIG['num_crews']}
  agent_start_positions: {ENV_CONFIG['agent_start_positions']}
  alpha: {ENV_CONFIG['alpha']}
  beta: {ENV_CONFIG['beta']}
  delta_beta: {ENV_CONFIG['delta_beta']}
  partial_obs: {ENV_CONFIG['partial_obs']}
  agent_view_size: {ENV_CONFIG['agent_view_size']}
  cooperative_reward: {ENV_CONFIG['cooperative_reward']}
  selfishness_weight: {ENV_CONFIG['selfishness_weight']}
  reward_shaping: {ENV_CONFIG['reward_shaping']}
  reward_shaping_config: {ENV_CONFIG['reward_shaping_config']}
  render_mode: {ENV_CONFIG['render_mode']}

mask_flag: False
global_state_flag: False
"""
    yaml_file.write_text(yaml_content)
    print(f"  Created/Updated config file: {yaml_file}")

    return str(yaml_file)


if __name__ == '__main__':
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='Train Discrete MADDPG on Wildfire Environment')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Run name for saving results (default: auto-generated)')
    args = parser.parse_args()

    # Run name 설정
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"discrete_maddpg_{timestamp}"

    # 출력 디렉토리 설정
    output_dir = Path(__file__).parent / "experiments" / "discrete_maddpg" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Wildfire Discrete MADDPG Training with Action Space Wrapper")
    print("=" * 80)
    print(f"\nRun name: {run_name}")
    print(f"Output directory: {output_dir}")

    # 환경 설정 출력
    print(f"\n환경 설정:")
    print(f"  Grid size: {ENV_CONFIG['size']}x{ENV_CONFIG['size']}")
    print(f"  Agents: {ENV_CONFIG['num_agents']}")
    print(f"  Max steps: {ENV_CONFIG['max_steps']}")
    print(f"  Helicopters: {ENV_CONFIG['num_helicopters']}")
    print(f"  Trucks: {ENV_CONFIG['num_trucks']}")
    print(f"  Crews: {ENV_CONFIG['num_crews']}")
    print(f"  Reward shaping: {ENV_CONFIG['reward_shaping']}")
    print(f"  Cooperative reward: {ENV_CONFIG['cooperative_reward']}")

    # ⭐ MARLlib에 wrapped 환경 등록 (discrete action을 continuous로 변환)
    ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnvDiscreteToContinuous
    COOP_ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnvDiscreteToContinuous

    # MARLlib 설정 파일 자동 생성
    print(f"\nMARLlib 설정 파일 생성 중...")
    config_path = setup_marllib_config()

    # 환경 초기화
    print(f"환경 초기화 중...")
    env_tuple = marl.make_env(
        environment_name="wildfire-ma",
        map_name="wildfire-ma",
        force_coop=False
    )

    # MADDPG 알고리즘 선택
    print(f"\nMADDPG 알고리즘 초기화 중...")
    maddpg = marl.algos.maddpg(
        hyperparam_source="common",
        batch_episode=10,
        actor_lr=0.0005,
        critic_lr=0.0005,
        learning_starts_episode=16,
        tau=0.002,
    )

    # 모델 빌드 (원본 환경 튜플로)
    print(f"모델 빌드 중...")
    model = marl.build_model(
        env_tuple,
        maddpg,
        {
            "core_arch": "mlp",
            "encode_layer": "256-256"
        }
    )

    # env_tuple은 이미 wrapped environment를 포함하고 있음
    # (WildfireRLlibEnvDiscreteToContinuous가 registry에 등록됨)
    env_wrapped, env_config = env_tuple

    # env_config에 필수 설정 추가
    if env_config is None:
        env_config = {}
    if "env_args" not in env_config:
        env_config["env_args"] = {}

    # centralized_critic_q가 필요한 설정들을 env_config에 추가
    env_config["opp_action_in_cc"] = False
    env_config["global_state_flag"] = False

    # MARLlib config에도 필수 설정 추가
    if maddpg.config_dict is not None:
        maddpg.config_dict["opp_action_in_cc"] = False
        maddpg.config_dict["global_state_flag"] = False
        maddpg.config_dict["local_dir"] = str(output_dir)
    else:
        maddpg.config_dict = {
            "opp_action_in_cc": False,
            "global_state_flag": False,
            "local_dir": str(output_dir)
        }

    # 학습 시작
    print("\n" + "=" * 80)
    print("학습 시작 (Discrete MADDPG with Action Space Wrapper)")
    print("=" * 80)
    print("\n특징:")
    print("  - Action space: Discrete(9) -> Continuous Box(9,)")
    print("  - Policy output: logits (unconstrained continuous values)")
    print("  - Step execution: argmax(logits) -> discrete action index")
    print("  - Critic input: logits or softmax(logits) probabilities")
    print("=" * 80 + "\n")

    try:
        maddpg.fit(
            env_tuple,  # 이미 wrapped environment를 포함하는 튜플
            model,
            stop={
                'episode_reward_mean': 50,
                'timesteps_total': 5000000,
            },
            local_mode=False,
            num_gpus=1,
            num_workers=2,
            share_policy='group',
            checkpoint_freq=10,
            local_dir=str(output_dir),
            evaluation_interval=None
        )

        print("\n" + "=" * 80)
        print("✓ 학습 완료!")
        print("=" * 80)
        print(f"결과 저장 위치: {output_dir}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR during training:")
        print(f"  Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        # 여전히 discrete MADDPG 에러라면 정보 제공
        if "Discrete" in str(e) or "action space" in str(e).lower():
            print("\n" + "=" * 80)
            print("NOTE: The action space wrapper has been successfully applied.")
            print("If you still see 'Discrete' action space errors, they may be from")
            print("a different part of the training pipeline.")
            print("=" * 80)
        raise
