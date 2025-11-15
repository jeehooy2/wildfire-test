"""
RLlib MultiAgentEnv 래퍼

wildfire-v0 환경을 RLlib의 MultiAgentEnv 형식으로 변환
"""

import gym
import wildfire_environment
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces
import gymnasium
import numpy as np 
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY





def convert_gym_space_to_gymnasium(gym_space):
    """gym.spaces를 gymnasium.spaces로 변환"""
    if isinstance(gym_space, spaces.Box):
        return gymnasium.spaces.Box(
            low=gym_space.low,
            high=gym_space.high,
            shape=gym_space.shape,
            dtype=gym_space.dtype
        )
    elif isinstance(gym_space, spaces.Discrete):
        return gymnasium.spaces.Discrete(n=gym_space.n)
    elif isinstance(gym_space, spaces.Dict):
        return gymnasium.spaces.Dict({
            k: convert_gym_space_to_gymnasium(v) for k, v in gym_space.spaces.items()
        })
    else:
        raise NotImplementedError(f"Unsupported space type: {type(gym_space)}")


class WildfireRLlibEnv(MultiAgentEnv):
    """RLlib용 Wildfire 다중 에이전트 환경 래퍼"""

    def __init__(self, env_config):
        """
        Parameters
        ----------
        env_config : dict
            환경 설정 딕셔너리
        """
        super().__init__()

        # wildfire-v0 환경 생성
        self.env = gym.make("wildfire-v0", **env_config)

        # 에이전트 수
        self._num_agents = self.env.num_agents
        self._agent_ids = set(range(self._num_agents))

        # RLlib 새 API에서 요구하는 속성들
        self.agents = list(range(self._num_agents))  # 현재 에피소드의 에이전트 ID 리스트
        self.possible_agents = list(range(self._num_agents))  # 가능한 모든 에이전트 ID 리스트

        # 액션/관찰 공간 설정 (RLlib 형식)
        # Dict 공간에서 단일 에이전트 공간 추출
        dict_action_space = self.env.action_space
        dict_obs_space = self.env.observation_space

        # 각 에이전트의 액션/관찰 공간 (모두 동일)
        # wildfire 환경은 키를 문자열로 사용 ("0", "1", ...)
        single_action_space = dict_action_space["0"]
        single_obs_space = dict_obs_space["0"]

        # gym spaces를 gymnasium spaces로 변환 (RLlib 새 API 호환성)
        single_action_space_gym = convert_gym_space_to_gymnasium(single_action_space)
        single_obs_space_gym = convert_gym_space_to_gymnasium(single_obs_space)

        # 단일 공간 저장 (샘플링용)
        self._single_action_space = single_action_space_gym
        self._single_obs_space = single_obs_space_gym

        # RLlib의 새 API stack은 dict 형태의 공간을 기대
        # 각 에이전트 ID를 키로 하는 dict 생성
        self._action_space = {i: single_action_space_gym for i in range(self._num_agents)}
        self._observation_space = {i: single_obs_space_gym for i in range(self._num_agents)}

        print(f"RLlib 래퍼 생성:")
        print(f"  에이전트 수: {self._num_agents}")
        print(f"  액션 공간: {single_action_space}")
        print(f"  관찰 공간: {single_obs_space}")

    def reset(self, *, seed=None, options=None):
        """환경 리셋"""
        obs_dict, info = self.env.reset(seed=seed)

        # wildfire env는 문자열 키("0", "1")를 사용, RLlib은 int 키를 사용
        # 문자열 키를 int로 변환
        obs = {int(k): v for k, v in obs_dict.items()}

        # RLlib MultiAgentEnv는 reset에서 (obs, infos) 형태 반환
        # infos는 각 에이전트별로
        infos = {int(k): info for k in obs_dict.keys()}

        return obs, infos

    def step(self, action_dict):
        """환경 스텝 실행"""
        # RLlib은 int 키를 사용하지만, wildfire env는 문자열 키를 기대함
        # int 키를 문자열로 변환
        env_actions = {str(i): action_dict[i] for i in range(self._num_agents)}

        # 환경 실행 (5-tuple 반환: obs, reward, terminated, truncated, infos)
        obs_dict, reward_dict, terminated, truncated, infos_env = self.env.step(env_actions)

        # 결과를 RLlib 형식으로 변환 (문자열 키를 int로)
        observations = {int(k): v for k, v in obs_dict.items()}
        rewards = {int(k): v for k, v in reward_dict.items()}

        # RLlib MultiAgentEnv 형식:
        # - terminateds: {agent_id: bool} 또는 "__all__": bool
        # - truncateds: {agent_id: bool} 또는 "__all__": bool
        # - infos: {agent_id: dict}
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}
        infos = {int(k): v for k, v in infos_env.items()}

        return observations, rewards, terminateds, truncateds, infos

    def action_space_sample(self, agent_ids=None):
        """액션 공간 샘플링"""
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: self._single_action_space.sample() for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids=None):
        """관찰 공간 샘플링"""
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: self._single_obs_space.sample() for agent_id in agent_ids}

    def get_agent_ids(self):
        """에이전트 ID 목록 반환"""
        return self._agent_ids

    def get_action_space(self, agent_id=None):
        """특정 에이전트의 액션 공간 반환"""
        if agent_id is None:
            return self._action_space
        return self._action_space[agent_id]

    def get_observation_space(self, agent_id=None):
        """특정 에이전트의 관찰 공간 반환"""
        if agent_id is None:
            return self._observation_space
        return self._observation_space[agent_id]

    # RLlib이 필요로 하는 속성들
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def num_agents(self):
        return self._num_agents

    def close(self):
        """환경 종료"""
        self.env.close()
