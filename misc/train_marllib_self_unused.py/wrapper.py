"""
Gym 0.20.0 compatible MARL wrapper for wildfire_environment using RLlib's MultiAgentEnv

This wrapper converts the wildfire-v0 environment to a format compatible with
gym 0.20.0, RLlib, and MARL training frameworks like MARLlib.
"""

import gym
import numpy as np
from gym import spaces
from collections import OrderedDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import wildfire_environment


class WildfireMultiAgentWrapper(MultiAgentEnv):
    """
    Gym 0.20.0 + RLlib MultiAgentEnv compatible wrapper for wildfire-v0 environment.

    This wrapper provides:
    - RLlib MultiAgentEnv interface
    - Dict observation and action spaces
    - Compatibility with MARL frameworks (MARLlib, RLlib)
    - Support for gym 0.20.0 reset/step signatures
    """

    def __init__(self, env_config=None):
        """
        Initialize the wrapper.

        Parameters
        ----------
        env_config : dict, optional
            Configuration for the wildfire environment.
            Common parameters:
            - size: int, grid size (default: 17)
            - num_agents: int, number of agents (default: 2)
            - max_steps: int, maximum episode steps (default: 100)
            - alpha: float, fire spread parameter (default: 0.05)
            - beta: float, fire persistence parameter (default: 0.9)
            - delta_beta: float, fire suppression parameter (default: 0.54)
            - cooperative_reward: bool, use cooperative rewards (default: False)
            - reward_shaping: str, reward shaping type (default: None)
            - num_helicopters: int, number of helicopter agents (default: 0)
            - num_trucks: int, number of truck agents (default: 0)
            - num_crews: int, number of crew agents (default: 0)
        """
        super().__init__()

        # 기본 설정
        if env_config is None:
            env_config = {}

        # wildfire-v0 환경 생성
        self.env = gym.make("wildfire-v0", **env_config)

        # 에이전트 수
        self.num_agents = self.env.num_agents
        self._agent_ids = set(range(self.num_agents))

        # RLlib MultiAgentEnv에서 요구하는 속성들
        self.agents = list(range(self.num_agents))
        self.possible_agents = list(range(self.num_agents))
        self._num_agents = self.num_agents

        # 기본 환경의 observation/action space 추출
        # wildfire 환경은 Dict space를 사용하며, 키가 문자열("0", "1", ...)
        base_obs_space = self.env.observation_space
        base_action_space = self.env.action_space

        # 단일 에이전트의 space 추출
        self._single_obs_space = base_obs_space["0"]
        self._single_action_space = base_action_space["0"]

        # RLlib의 MultiAgentEnv은 Dict space를 사용 (int 키)
        self._observation_space = spaces.Dict({
            i: self._single_obs_space for i in range(self._num_agents)
        })
        self._action_space = spaces.Dict({
            i: self._single_action_space for i in range(self._num_agents)
        })

        # 메타데이터
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'name': 'WildfireMultiAgent-v0'
        }

        print(f"[WildfireWrapper] 초기화 완료")
        print(f"  에이전트 수: {self._num_agents}")
        print(f"  관찰 공간: {self._single_obs_space}")
        print(f"  액션 공간: {self._single_action_space}")

    def reset(self, *, seed=None, options=None):
        """
        환경 리셋 (gym 0.20.0 + RLlib signature).

        Parameters
        ----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional options (e.g., initial state)

        Returns
        -------
        observations : dict
            {agent_id: observation} 형태의 딕셔너리 (int 키)
        """
        # wildfire env는 (obs, info) 반환
        if seed is not None:
            obs_dict, info = self.env.reset(seed=seed)
        else:
            obs_dict, info = self.env.reset()

        # 문자열 키("0", "1", ...)를 int로 변환
        observations = {int(k): v for k, v in obs_dict.items()}

        return observations

    def step(self, action_dict):
        """
        환경 스텝 실행 (RLlib MultiAgentEnv signature).

        Parameters
        ----------
        action_dict : dict
            {agent_id: action} 형태의 딕셔너리 (int 키)

        Returns
        -------
        observations : dict
            다음 상태 관찰 {agent_id: obs}
        rewards : dict
            각 에이전트의 보상 {agent_id: reward}
        dones : dict
            종료 상태 (RLlib 형식: {"__all__": bool})
        infos : dict
            추가 정보 {agent_id: info}
        """
        # int 키를 문자열로 변환하여 base env에 전달
        env_actions = {str(i): action_dict[i] for i in range(self._num_agents)}

        # wildfire env step: (obs, rewards, terminated, truncated, infos)
        obs_dict, reward_dict, terminated, truncated, infos_dict = self.env.step(env_actions)

        # 문자열 키를 int로 변환
        observations = {int(k): v for k, v in obs_dict.items()}
        rewards = {int(k): v for k, v in reward_dict.items()}
        infos = {int(k): v for k, v in infos_dict.items()}

        # RLlib MultiAgentEnv 형식: dones는 최소한 "__all__" 키를 포함해야 함
        done = terminated or truncated
        dones = {"__all__": done}

        return observations, rewards, dones, infos

    def render(self, mode='rgb_array'):
        """
        환경 렌더링.

        Parameters
        ----------
        mode : str
            렌더링 모드 ('human' 또는 'rgb_array')

        Returns
        -------
        img : np.ndarray or None
            rgb_array 모드일 경우 이미지 반환
        """
        return self.env.render(mode=mode)

    def close(self):
        """환경 종료"""
        if hasattr(self, 'env'):
            self.env.close()

    def seed(self, seed=None):
        """
        랜덤 시드 설정 (legacy gym interface).

        Parameters
        ----------
        seed : int, optional
            Random seed
        """
        # gym 0.20.0 이후에는 reset(seed=...)를 사용하지만
        # 하위 호환성을 위해 제공
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return [seed]

    # RLlib MultiAgentEnv 필수 메서드

    def action_space_sample(self, agent_ids=None):
        """
        액션 공간 샘플링 (RLlib 인터페이스).

        Parameters
        ----------
        agent_ids : list, optional
            샘플링할 에이전트 ID 리스트 (None이면 모든 에이전트)

        Returns
        -------
        actions : dict
            샘플링된 액션 딕셔너리
        """
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: self._single_action_space.sample() for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids=None):
        """
        관찰 공간 샘플링 (RLlib 인터페이스).

        Parameters
        ----------
        agent_ids : list, optional
            샘플링할 에이전트 ID 리스트 (None이면 모든 에이전트)

        Returns
        -------
        observations : dict
            샘플링된 관찰 딕셔너리
        """
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: self._single_obs_space.sample() for agent_id in agent_ids}

    def get_agent_ids(self):
        """에이전트 ID 집합 반환"""
        return self._agent_ids

    # RLlib이 필요로 하는 속성 프로퍼티

    @property
    def observation_space(self):
        """관찰 공간 (Dict space)"""
        return self._observation_space

    @property
    def action_space(self):
        """액션 공간 (Dict space)"""
        return self._action_space

    @property
    def unwrapped(self):
        """원본 환경 반환"""
        return self.env


# MARLlib 호환성을 위한 추가 래퍼
class WildfireMARLlibWrapper(WildfireMultiAgentWrapper):
    """
    MARLlib 전용 wrapper.

    MARLlib의 ENV_REGISTRY에 등록하기 위한 추가 인터페이스 제공.
    """

    def __init__(self, env_config=None):
        super().__init__(env_config)

        # MARLlib에서 필요로 하는 추가 속성들
        self.n_agents = self.num_agents

    def state(self):
        """
        전역 상태 반환 (CTDE 알고리즘용).

        Returns
        -------
        state : np.ndarray
            전역 상태 벡터
        """
        return self.env.get_state()

    @property
    def state_space(self):
        """전역 상태 공간"""
        # 환경의 state 차원 계산
        # obs_depth = num_agents + len(STATE_IDX_TO_COLOR_WILDFIRE)
        # state = obs_depth * (grid_size_without_walls + 1)^2 + 1 (time step)
        state_dim = self.env.obs_depth * ((self.env.grid_size_without_walls + 1) ** 2) + 1
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32
        )

    def get_observation_space(self, agent_id=None):
        """
        특정 에이전트의 관찰 공간 반환 (MARLlib 인터페이스).

        Parameters
        ----------
        agent_id : int, optional
            에이전트 ID (None이면 전체 Dict 반환)

        Returns
        -------
        space : gym.Space
            관찰 공간
        """
        if agent_id is None:
            return self._observation_space
        return self._observation_space[agent_id]

    def get_action_space(self, agent_id=None):
        """
        특정 에이전트의 액션 공간 반환 (MARLlib 인터페이스).

        Parameters
        ----------
        agent_id : int, optional
            에이전트 ID (None이면 전체 Dict 반환)

        Returns
        -------
        space : gym.Space
            액션 공간
        """
        if agent_id is None:
            return self._action_space
        return self._action_space[agent_id]


# 환경 생성 헬퍼 함수
def make_wildfire_env(env_config=None):
    """
    Wildfire MARL 환경 생성 헬퍼 함수.

    Parameters
    ----------
    env_config : dict, optional
        환경 설정

    Returns
    -------
    env : WildfireMultiAgentWrapper
        래핑된 환경

    Examples
    --------
    >>> env = make_wildfire_env({
    ...     'num_agents': 3,
    ...     'size': 17,
    ...     'max_steps': 100,
    ...     'cooperative_reward': True
    ... })
    """
    return WildfireMultiAgentWrapper(env_config)


def make_wildfire_marllib_env(env_config=None):
    """
    MARLlib용 Wildfire 환경 생성 헬퍼 함수.

    Parameters
    ----------
    env_config : dict, optional
        환경 설정

    Returns
    -------
    env : WildfireMARLlibWrapper
        MARLlib 호환 환경

    Examples
    --------
    >>> env = make_wildfire_marllib_env({
    ...     'num_agents': 3,
    ...     'size': 17,
    ...     'max_steps': 100,
    ...     'reward_shaping': 'individual2'
    ... })
    """
    return WildfireMARLlibWrapper(env_config)


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("Wildfire MARL Wrapper 테스트 (RLlib MultiAgentEnv)")
    print("=" * 60)

    # 기본 wrapper 테스트
    env_config = {
        'num_agents': 3,
        'size': 17,
        'max_steps': 50,
        'cooperative_reward': True,
        'alpha': 0.05,
        'beta': 0.9,
        'delta_beta': 0.54,
        'agent_start_positions': ((1, 1), (8, 8), (15, 15)),  # 3개 에이전트 위치
    }

    env = make_wildfire_env(env_config)

    print("\n[테스트 1] 환경 리셋")
    obs = env.reset(seed=42)
    print(f"  관찰 keys: {list(obs.keys())}")
    print(f"  관찰 타입: {type(list(obs.keys())[0])}")  # int 확인
    print(f"  관찰 shape (agent 0): {obs[0].shape}")

    print("\n[테스트 2] 랜덤 액션 스텝")
    actions = env.action_space_sample()
    print(f"  액션: {actions}")

    next_obs, rewards, dones, infos = env.step(actions)
    print(f"  보상: {rewards}")
    print(f"  종료: {dones['__all__']}")
    print(f"  info (agent 0): {infos[0]}")

    print("\n[테스트 3] 에피소드 실행")
    obs = env.reset(seed=123)
    total_rewards = {i: 0.0 for i in range(env.num_agents)}

    for step in range(10):
        actions = env.action_space_sample()
        obs, rewards, dones, infos = env.step(actions)

        for i in range(env.num_agents):
            total_rewards[i] += rewards[i]

        if dones["__all__"]:
            print(f"  에피소드 종료: step={step+1}")
            break

    print(f"  누적 보상: {total_rewards}")

    env.close()

    print("\n[테스트 4] MARLlib wrapper 테스트")
    marllib_env = make_wildfire_marllib_env(env_config)

    print(f"  agents: {marllib_env.agents}")
    print(f"  n_agents: {marllib_env.n_agents}")

    obs = marllib_env.reset(seed=42)
    state = marllib_env.state()
    print(f"  state shape: {state.shape}")
    print(f"  state space: {marllib_env.state_space}")

    # 개별 공간 조회 테스트
    obs_space_0 = marllib_env.get_observation_space(0)
    action_space_0 = marllib_env.get_action_space(0)
    print(f"  observation_space[0]: {obs_space_0}")
    print(f"  action_space[0]: {action_space_0}")

    marllib_env.close()

    print("\n[테스트 5] RLlib 속성 확인")
    env2 = make_wildfire_env(env_config)
    print(f"  observation_space type: {type(env2.observation_space)}")
    print(f"  action_space type: {type(env2.action_space)}")
    print(f"  observation_space: {env2.observation_space}")
    print(f"  action_space: {env2.action_space}")
    env2.close()

    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)
