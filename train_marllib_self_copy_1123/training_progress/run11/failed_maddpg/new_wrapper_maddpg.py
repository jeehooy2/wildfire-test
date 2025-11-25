"""
Discrete to Continuous Action Space Wrapper for MADDPG

Discrete action space를 continuous (Box) action space로 변환합니다.
MADDPG (continuous action 기반 알고리즘)가 discrete 환경에서 작동하도록 합니다.

메커니즘:
---------
1. 환경의 discrete action space (Discrete(9))를 Box(9,) (연속값)로 변환
2. step() 호출 시 argmax로 discrete action으로 변환하여 실제 환경에 전달
3. 관찰값은 래핑하지 않음 (그대로 전달)

사용 예시:
----------
from train_marllib_self.new_wrapper_maddpg import WildfireRLlibEnvDiscreteToContinuous
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

# MARLlib 환경 registry에 등록
ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnvDiscreteToContinuous
COOP_ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnvDiscreteToContinuous

# 이제 wrapped environment가 자동으로 사용됨
env_tuple = marl.make_env(environment_name="wildfire-ma", map_name="wildfire-ma")
# env_tuple[0].action_space는 Box(-1.0, 1.0, (9,), float32)
"""

import numpy as np
from typing import Dict, Tuple
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from train_marllib_self.new_wrapper import WildfireRLlibEnv


class DiscreteToContinuousWrapper(MultiAgentEnv):
    """
    Discrete action space를 continuous (Box) action space로 변환하는 래퍼.

    MADDPG 같은 continuous action space 기반 알고리즘이 discrete 환경에서
    작동하도록 변환합니다.

    Parameters
    ----------
    env : MultiAgentEnv
        원본 환경 (discrete action space)

    Attributes
    ----------
    env : MultiAgentEnv
        래핑된 환경
    _action_space_discrete : gym.Space
        원본 discrete action space
    _action_space : gym.Space
        변환된 continuous action space (Box)
    num_actions : int
        Discrete action의 개수
    """

    def __init__(self, env: MultiAgentEnv):
        """
        Parameters
        ----------
        env : MultiAgentEnv
            원본 환경

        Raises
        ------
        ValueError
            Action space가 Discrete이 아닌 경우
        """
        super().__init__()

        self.env = env

        # Original discrete action space 저장
        self._action_space_discrete = env.action_space

        # Action space가 Discrete인지 확인
        if not isinstance(self._action_space_discrete, spaces.Discrete):
            raise ValueError(
                f"Expected Discrete action space, got {type(self._action_space_discrete)}. "
                f"This wrapper only converts Discrete to Box (continuous)."
            )

        # Discrete action 개수
        self.num_actions = self._action_space_discrete.n

        # Continuous action space로 변환 (logits/확률값)
        # Bounded box: MADDPG가 요구하는 bounded action space
        # bounds: [-1, 1] (정규화된 범위)
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32
        )

        print(f"✓ Discrete to Continuous Action Space Wrapper")
        print(f"  Original: {self._action_space_discrete}")
        print(f"  Converted: {self._action_space}")

    def _convert_action(self, continuous_action: np.ndarray) -> int:
        """Continuous action (logits)을 discrete action으로 변환"""
        # Action bounds를 [-1, 1]로 clamp (MADDPG가 요구)
        bounded_action = np.clip(continuous_action, -1.0, 1.0)
        discrete_action = int(np.argmax(bounded_action))
        return discrete_action

    def reset(self, *, seed=None, options=None):
        """환경 리셋"""
        return self.env.reset(seed=seed, options=options)

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        환경 스텝 (continuous action을 discrete로 변환)

        Parameters
        ----------
        action_dict : Dict[str, np.ndarray]
            에이전트별 continuous action
            Key: 에이전트 ID
            Value: Shape (num_actions,) float array (logits/확률)

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            observations, rewards, dones, infos
        """
        # Continuous action을 discrete로 변환
        discrete_action_dict = {}

        for agent_id, continuous_action in action_dict.items():
            # numpy array로 변환 (필요시)
            if not isinstance(continuous_action, np.ndarray):
                continuous_action = np.array(continuous_action, dtype=np.float32)

            # Discrete action으로 변환
            discrete_action = self._convert_action(continuous_action)
            discrete_action_dict[agent_id] = discrete_action

        # 원본 환경에 discrete action 전달
        obs, rewards, dones, infos = self.env.step(discrete_action_dict)

        return obs, rewards, dones, infos

    @property
    def action_space(self) -> spaces.Box:
        """Continuous action space 반환"""
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        """Observation space 반환 (원본 그대로)"""
        return self.env.observation_space

    @property
    def num_agents(self) -> int:
        """에이전트 수 반환"""
        return self.env.num_agents

    @property
    def agents(self):
        """현재 에이전트 리스트"""
        return self.env.agents

    @property
    def possible_agents(self):
        """가능한 모든 에이전트 리스트"""
        return self.env.possible_agents

    def get_env_info(self):
        """MARLlib이 요구하는 환경 정보 (wrapped version)"""
        env_info = self.env.get_env_info()

        # Action space 업데이트 (continuous로 변환됨)
        env_info["space_act"] = self._action_space

        return env_info

    def close(self):
        """환경 종료"""
        self.env.close()

    def render(self, mode='human'):
        """렌더링"""
        return self.env.render(mode=mode)

    def action_space_sample(self, agent_ids=None):
        """액션 공간 샘플링 (continuous 샘플링)"""
        if agent_ids is None:
            agent_ids = self.agents

        # Continuous action space에서 샘플링
        return {agent_id: self.action_space.sample() for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids=None):
        """관찰 공간 샘플링"""
        if agent_ids is None:
            agent_ids = self.agents

        return self.env.observation_space_sample(agent_ids)

    def get_agent_ids(self):
        """에이전트 ID 반환"""
        return self.env.get_agent_ids()

    def get_action_space(self, agent_id=None):
        """액션 공간 반환 (continuous)"""
        return self._action_space

    def get_observation_space(self, agent_id=None):
        """관찰 공간 반환"""
        return self.env.get_observation_space(agent_id)

    def __getattr__(self, name):
        """원본 환경의 속성/메서드 위임"""
        return getattr(self.env, name)


class WildfireRLlibEnvDiscreteToContinuous(DiscreteToContinuousWrapper):
    """
    Discrete action space를 continuous로 변환하는 Wildfire 환경.

    RLlib workers가 이 클래스를 인스턴스화할 때 env_config를 받아서
    자동으로 원본 환경을 생성하고 wrapper를 적용합니다.

    MARLlib 환경 registry에 등록되어 사용됩니다:

    Examples
    --------
    from train_marllib_self.new_wrapper_maddpg import WildfireRLlibEnvDiscreteToContinuous
    from marllib.envs.base_env import ENV_REGISTRY
    from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

    # Registry 등록
    ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnvDiscreteToContinuous
    COOP_ENV_REGISTRY["wildfire-ma"] = WildfireRLlibEnvDiscreteToContinuous

    # 이제 make_env를 호출하면 자동으로 wrapped environment가 반환됨
    env_tuple = marl.make_env(environment_name="wildfire-ma", map_name="wildfire-ma")
    env, env_config = env_tuple
    # env.action_space는 이미 Box(-1.0, 1.0, (9,)) 형태
    """

    def __init__(self, env_config):
        """
        Parameters
        ----------
        env_config : dict
            환경 설정 (MARLlib에서 전달)
        """
        # 원본 환경 생성
        original_env = WildfireRLlibEnv(env_config)
        # Wrapper 초기화
        super().__init__(original_env)
