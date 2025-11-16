"""
RLlib MultiAgentEnv 래퍼

wildfire-v0 환경을 RLlib의 MultiAgentEnv 형식으로 변환
"""

import gym
import wildfire_environment
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces
import numpy as np
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

# Policy mapping dictionary for MARLlib
policy_mapping_dict = {
    "wildfire-ma": {
        "description": "cooperative wildfire suppression",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}

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

        # MARLlib에서 전달받은 env_config 정리
        # 1. map_name 제거 (wildfire env가 받지 않는 파라미터)
        clean_config = {k: v for k, v in env_config.items() if k != 'map_name'}

        # 2. 문자열로 전달된 agent_start_positions를 튜플로 변환
        if 'agent_start_positions' in clean_config and isinstance(clean_config['agent_start_positions'], str):
            import ast
            clean_config['agent_start_positions'] = ast.literal_eval(clean_config['agent_start_positions'])

        # 3. 문자열 "None"을 실제 None으로 변환
        if 'reward_shaping_config' in clean_config and clean_config['reward_shaping_config'] == 'None':
            clean_config['reward_shaping_config'] = None

        print(f"\n=== WildfireRLlibEnv 초기화 ===")
        print(f"  에이전트 수: {clean_config.get('num_agents', 'N/A')}")
        print(f"  Grid 크기: {clean_config.get('size', 'N/A')}")
        print(f"  Max steps: {clean_config.get('max_steps', 'N/A')}")
        print(f"=== 환경 생성 시작 ===\n")

        # wildfire-v0 환경 생성
        try:
            print(f"gym.make 호출 전...")
            # disable_env_checker를 사용하여 래퍼 자동 추가 방지
            import wildfire_environment
            from wildfire_environment.envs import WildfireEnv

            # 직접 환경 생성 (gym.make의 래퍼 자동 추가 회피)
            self.env = WildfireEnv(**clean_config)

            print(f"직접 환경 생성 완료!")
        except Exception as e:
            print(f"❌ ERROR in gym.make:")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 에이전트 수
        self._num_agents = self.env.num_agents
        self._agent_ids = set(range(self.num_agents))

        # RLlib 새 API에서 요구하는 속성들
        self.agents = list(range(self.num_agents))  # 현재 에피소드의 에이전트 ID 리스트
        self.possible_agents = list(range(self.num_agents))  # 가능한 모든 에이전트 ID 리스트

        # 액션/관찰 공간 설정 (RLlib 형식)
        # Dict 공간에서 단일 에이전트 공간 추출
        dict_action_space = self.env.action_space
        dict_obs_space = self.env.observation_space

        # 각 에이전트의 액션/관찰 공간 (모두 동일)
        # wildfire 환경은 키를 문자열로 사용 ("0", "1", ...)
        single_action_space = dict_action_space["0"]
        single_obs_space = dict_obs_space["0"]

        # 단일 공간 저장 (샘플링용)
        self._single_action_space = single_action_space
        self._single_obs_space = single_obs_space

        # MARLlib space definitions (단일 에이전트의 space, 모든 에이전트가 공유)
        from gym.spaces import Dict as GymDict
        self._observation_space = GymDict({"obs": single_obs_space})
        self._action_space = single_action_space

        print(f"RLlib 래퍼 생성:")
        print(f"  에이전트 수: {self.num_agents}")
        print(f"  액션 공간: {single_action_space}")
        print(f"  관찰 공간: {single_obs_space}")

    def reset(self, *, seed=None, options=None):
        """환경 리셋"""
        # Gym 환경은 (obs, info) 튜플을 반환
        result = self.env.reset(seed=seed)
        if isinstance(result, tuple):
            obs_dict, info = result
        else:
            obs_dict = result

        # wildfire env는 문자열 키("0", "1")를 사용, RLlib은 int 키를 사용
        # 문자열 키를 int로 변환하고 MARLlib 형식으로 래핑
        obs = {int(k): {"obs": v} for k, v in obs_dict.items()}

        return obs

    def step(self, action_dict):
        """환경 스텝 실행"""
        # RLlib은 int 키를 사용하지만, wildfire env는 문자열 키를 기대함
        # int 키를 문자열로 변환
        env_actions = {str(i): action_dict[i] for i in range(self.num_agents)}

        # 환경 실행 (5-tuple 반환: obs, reward, terminated, truncated, infos)
        # obs_dict, reward_dict, done, infos_env = self.env.step(env_actions)
        obs_dict, reward_dict, terminated, truncated, infos_dict = self.env.step(env_actions)

        # 결과를 RLlib 형식으로 변환 (문자열 키를 int로, MARLlib obs 형식으로 래핑)
        observations = {int(k): {"obs": v} for k, v in obs_dict.items()}
        rewards = {int(k): v for k, v in reward_dict.items()}
        infos = {int(k): v for k, v in infos_dict.items()}

        # dones = {"__all__": done}
        done = terminated or truncated
        dones = {"__all__": done}

        return observations, rewards, dones, infos
    
    def render(self, mode='rgb_array'):
        """
        img : np.ndarray or None
            rgb_array 모드일 경우 이미지 반환
        """
        return self.env.render(mode=mode)

    def action_space_sample(self, agent_ids=None):
        """액션 공간 샘플링"""
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: self._single_action_space.sample() for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids=None):
        """관찰 공간 샘플링"""
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: {"obs": self._single_obs_space.sample()} for agent_id in agent_ids}

    def get_agent_ids(self):
        """에이전트 ID 목록 반환"""
        return self._agent_ids

    def get_action_space(self, agent_id=None):
        """특정 에이전트의 액션 공간 반환 (모든 에이전트가 같은 space 공유)"""
        return self._action_space

    def get_observation_space(self, agent_id=None):
        """특정 에이전트의 관찰 공간 반환 (모든 에이전트가 같은 space 공유)"""
        return self._observation_space

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

    def get_env_info(self):
        """MARLlib이 요구하는 환경 정보 반환"""
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
