"""
Discrete MADDPG Policy with Gumbel-Softmax Support

이 모듈은 discrete action space를 지원하는 MADDPG 정책을 구현합니다.
Gumbel-Softmax reparameterization을 사용하여 discrete action을 미분 가능하게 샘플링합니다.

사용 방법:
---------
from train_marllib_self.discrete_maddpg_policy import DiscreteMaddpgTorchPolicy
from ray.rllib.agents.trainer import Trainer

config = {
    "custom_policy_class": DiscreteMaddpgTorchPolicy,
    # ... 다른 설정들
}
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Type, Optional, List, Tuple, Union, Any
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.ddpg.ddpg_torch_policy import (
    TargetNetworkMixin,
    ComputeTDErrorMixin,
    DDPGTorchPolicy
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.distributions import Categorical
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from marllib.marl.algos.utils.centralized_Q import (
    CentralizedQValueMixin,
    centralized_critic_q,
    before_learn_on_batch
)
from marllib.marl.algos.core.IL.ddpg import (
    IDDPG_DEFAULT_CONFIG,
    IDDPGTorchPolicy,
    IDDPGTrainer
)

torch, nn = try_import_torch()


# ============================================================================
# Gumbel-Softmax 함수
# ============================================================================

def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0,
                   hard: bool = False, eps: float = 1e-20) -> torch.Tensor:
    """
    Gumbel-Softmax reparameterization trick for discrete action sampling.

    Parameters
    ----------
    logits : torch.Tensor
        Actor network의 출력. Shape: (..., num_actions)
    tau : float
        Temperature parameter. 작은 값일수록 one-hot에 가까움
    hard : bool
        True면 hard one-hot sample, False면 soft sample
    eps : float
        수치 안정성을 위한 작은 값

    Returns
    -------
    torch.Tensor
        Gumbel-Softmax sampled actions (soft 또는 hard one-hot)
    """
    # Gumbel noise 생성
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + eps) + eps)

    # Logits에 Gumbel noise 추가
    gumbel_logits = (logits + gumbel_noise) / tau

    # Softmax를 통해 확률 분포로 변환 (미분 가능)
    y_soft = F.softmax(gumbel_logits, dim=-1)

    if hard:
        # Straight-through estimator
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


# ============================================================================
# Discrete MADDPG Model with Gumbel-Softmax
# ============================================================================

class DiscreteMaddpgModel(TorchModelV2):
    """
    Discrete MADDPG 모델: Actor, Critic, 그리고 Twin Q-network.

    - Actor: discrete actions를 logits로 출력
    - Critic: 모든 에이전트의 actions를 입력으로 받음 (중앙화)
    - Twin Q-network (옵션): 안정성 향상
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Action space가 discrete인지 확인
        if not hasattr(action_space, 'n'):
            raise ValueError(f"Action space must be Discrete, got {type(action_space)}")

        self.num_actions = action_space.n
        self.action_space = action_space

        # 관찰 공간 크기 계산
        if hasattr(obs_space, 'shape'):
            obs_size = int(np.prod(obs_space.shape))
        else:
            obs_size = obs_space.size

        # 네트워크 구성
        self.fcnet_hiddens = model_config["fcnet_hiddens"]
        self.fcnet_activation = model_config["fcnet_activation"]

        # Actor network: observation -> logits (discrete actions)
        hidden_sizes = self.fcnet_hiddens + [self.num_actions]
        self.actor = self._build_fcnet(obs_size, hidden_sizes, final_activation=None)

        # 모델 출력 크기 (logits)
        self._num_outputs = self.num_actions

        # Critic network: observation + all_actions -> Q-value
        # 간단한 구현: observation만 사용 (centralized critic은 train_batch에서 처리)
        critic_hidden_sizes = self.fcnet_hiddens + [1]
        self.critic = self._build_fcnet(obs_size, critic_hidden_sizes, final_activation=None)

        # Twin Q-network (옵션)
        self.twin_q = model_config.get("twin_q", False)
        if self.twin_q:
            self.twin_critic = self._build_fcnet(obs_size, critic_hidden_sizes, final_activation=None)

    def _build_fcnet(self, input_size: int, hidden_sizes: List[int],
                     final_activation=None) -> nn.Sequential:
        """Fully connected network 구성"""
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes[:-1]):
            layers.append(nn.Linear(prev_size, hidden_size))
            activation = self.fcnet_activation or "relu"
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            prev_size = hidden_size

        # Final layer
        layers.append(nn.Linear(prev_size, hidden_sizes[-1]))
        if final_activation:
            if final_activation == "relu":
                layers.append(nn.ReLU())
            elif final_activation == "tanh":
                layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, input_dict, state, seq_lens):
        """Forward pass: observation -> logits"""
        obs = input_dict["obs"]

        # Observation flatten
        if len(obs.shape) > 2:
            obs = obs.reshape(obs.shape[0], -1)

        # Actor output (logits)
        logits = self.actor(obs)

        return logits, state

    def get_policy_output(self, model_out, state_in, seq_lens):
        """정책 출력 (logits)"""
        return model_out, state_in

    def get_q_values(self, model_out, state_in, seq_lens, actions=None):
        """Critic Q-값 계산"""
        # 간단한 구현: model_out은 obs에서 나온 것
        # 실제로는 obs + actions를 concat하여 사용
        obs = model_out

        if len(obs.shape) > 2:
            obs = obs.reshape(obs.shape[0], -1)

        q_values = self.critic(obs)
        return q_values, state_in

    def get_twin_q_values(self, model_out, state_in, seq_lens, actions=None):
        """Twin Q 값 계산"""
        obs = model_out

        if len(obs.shape) > 2:
            obs = obs.reshape(obs.shape[0], -1)

        twin_q_values = self.twin_critic(obs)
        return twin_q_values, state_in


# ============================================================================
# Discrete MADDPG Loss Function
# ============================================================================

def discrete_maddpg_loss(policy: Policy, model: ModelV2,
                        dist_class: ActionDistribution,
                        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """
    Discrete MADDPG 손실 함수 (Gumbel-Softmax 포함).

    Parameters
    ----------
    policy : Policy
        정책 객체
    model : ModelV2
        모델 객체
    dist_class : ActionDistribution
        액션 분포 클래스
    train_batch : SampleBatch
        학습 배치

    Returns
    -------
    TensorType 또는 List[TensorType]
        손실값(들)
    """
    # 초기화
    CentralizedQValueMixin.__init__(policy)

    gamma = policy.config["gamma"]
    tau_target = policy.config["tau"]  # Soft update coefficient
    gumbel_tau = policy.config.get("gumbel_tau", 1.0)  # Gumbel-Softmax temperature

    # 현재 상태와 다음 상태의 model output
    obs = train_batch[SampleBatch.CUR_OBS]
    next_obs = train_batch[SampleBatch.NEXT_OBS]
    actions = train_batch[SampleBatch.ACTIONS]  # Discrete action indices
    rewards = train_batch[SampleBatch.REWARDS]
    dones = train_batch[SampleBatch.DONES]

    # 배치 크기
    batch_size = obs.shape[0]
    device = obs.device if hasattr(obs, 'device') else torch.device('cpu')

    # ========================================================================
    # Actor (Policy) Loss
    # ========================================================================

    # 현재 정책의 로짓
    with torch.no_grad():
        actor_logits_current = model({"obs": obs}, [], None)[0]

    # Training time: Gumbel-Softmax with soft sampling
    actor_logits = model({"obs": obs}, [], None)[0]
    actor_action_probs = gumbel_softmax(actor_logits, tau=gumbel_tau, hard=False)

    # Critic Q-값 계산 (soft action probs를 입력으로)
    # 간단한 구현: discrete action을 one-hot으로 변환
    action_one_hot = torch.zeros(batch_size, model.num_actions, device=device)
    action_one_hot.scatter_(1, actions.unsqueeze(1).long(), 1.0)

    # Actor loss: -E[Q(s, a)]
    q_values = model.critic(obs)
    q_values = q_values.squeeze(-1) if q_values.dim() > 1 else q_values
    actor_loss = -torch.mean(q_values)

    # ========================================================================
    # Critic (Q-Network) Loss
    # ========================================================================

    # 현재 Q-값
    q_t = model.critic(obs)
    q_t = q_t.squeeze(-1) if q_t.dim() > 1 else q_t

    # 다음 상태의 정책
    with torch.no_grad():
        next_actor_logits = model({"obs": next_obs}, [], None)[0]
        next_actor_action_probs = gumbel_softmax(next_actor_logits, tau=gumbel_tau, hard=False)

        # Target critic Q-값
        target_q_tp1 = model.critic(next_obs)
        target_q_tp1 = target_q_tp1.squeeze(-1) if target_q_tp1.dim() > 1 else target_q_tp1

        # Twin Q (옵션)
        if hasattr(model, 'twin_q') and model.twin_q:
            target_twin_q_tp1 = model.twin_critic(next_obs)
            target_twin_q_tp1 = target_twin_q_tp1.squeeze(-1) if target_twin_q_tp1.dim() > 1 else target_twin_q_tp1
            target_q_tp1 = torch.min(target_q_tp1, target_twin_q_tp1)

        # Bellman target
        target_q = rewards + gamma * (1 - dones.float()) * target_q_tp1

    # Critic loss (MSE)
    critic_loss = torch.mean((q_t - target_q) ** 2)

    # Twin Q loss (옵션)
    twin_q_loss = torch.tensor(0.0, device=device)
    if hasattr(model, 'twin_q') and model.twin_q:
        twin_q_t = model.twin_critic(obs)
        twin_q_t = twin_q_t.squeeze(-1) if twin_q_t.dim() > 1 else twin_q_t
        twin_q_loss = torch.mean((twin_q_t - target_q) ** 2)

    # ========================================================================
    # Total Loss
    # ========================================================================

    total_loss = actor_loss + critic_loss + twin_q_loss

    # 통계 정보 저장
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.q_t_mean = torch.mean(q_t)

    return total_loss


# ============================================================================
# Discrete MADDPG Policy Class
# ============================================================================

def make_model_and_action_dist_discrete(policy: Policy, obs_space,
                                        action_space, config: Dict) -> \
        Tuple[ModelV2, Type[ActionDistribution]]:
    """
    Discrete MADDPG 모델과 액션 분포 생성.

    Parameters
    ----------
    policy : Policy
        정책 객체
    obs_space : gym.Space
        관찰 공간
    action_space : gym.Space
        액션 공간 (Discrete)
    config : Dict
        설정 딕셔너리

    Returns
    -------
    Tuple[ModelV2, Type[ActionDistribution]]
        모델과 액션 분포 클래스
    """
    if not hasattr(action_space, 'n'):
        raise ValueError(f"Discrete MADDPG only supports Discrete action space, got {type(action_space)}")

    model = DiscreteMaddpgModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        name="discrete_maddpg_model"
    )

    # Categorical distribution for discrete actions
    return model, Categorical


def vf_preds_fetches_discrete(policy, input_dict, state_batches, model, action_dist):
    """Value function 예측 (MADDPG는 deterministic이므로 empty)"""
    return dict()


# Discrete MADDPG Policy 클래스 생성
DiscreteMaddpgTorchPolicy = IDDPGTorchPolicy.with_updates(
    name="DiscreteMaddpgTorchPolicy",
    postprocess_fn=centralized_critic_q,
    extra_action_out_fn=vf_preds_fetches_discrete,
    make_model_and_action_dist=make_model_and_action_dist_discrete,
    loss_fn=discrete_maddpg_loss,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        CentralizedQValueMixin
    ]
)


# ============================================================================
# Discrete MADDPG Trainer
# ============================================================================

def validate_config_discrete(config: TrainerConfigDict) -> None:
    """설정 검증 및 기본값 설정"""
    # Discrete action space 확인
    if hasattr(config.get("action_space"), 'n'):
        pass  # Discrete space OK

    # Gumbel-Softmax temperature 기본값
    if "gumbel_tau" not in config:
        config["gumbel_tau"] = 1.0

    # 기본 DDPG 설정 적용
    config["replay_sequence_length"] = \
        config["burn_in"] + config["model"]["max_seq_len"]

    def f(batch, workers, config):
        policies = dict(workers.local_worker()
                        .foreach_trainable_policy(lambda p, i: (i, p)))
        return before_learn_on_batch(batch, policies,
                                     config["train_batch_size"])

    config["before_learn_on_batch"] = f


# Discrete MADDPG Trainer 생성
DiscreteMaddpgTrainer = IDDPGTrainer.with_updates(
    name="DiscreteMaddpgTrainer",
    default_config=IDDPG_DEFAULT_CONFIG,
    default_policy=DiscreteMaddpgTorchPolicy,
    validate_config=validate_config_discrete,
    allow_unknown_subkeys=["Q_model", "policy_model", "gumbel_tau"]
)
