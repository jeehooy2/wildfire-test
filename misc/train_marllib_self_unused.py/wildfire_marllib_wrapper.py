"""
MARLlib MultiAgentEnv 래퍼

wildfire-v0 환경을 MARLlib의 MultiAgentEnv 형식으로 변환
add_new_env.py의 RLlibMAGym 클래스 구조를 참조하여 작성
"""

import gym
import wildfire_environment
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box, Discrete
import numpy as np
import time


# MARLlib을 위한 정책 매핑 정보
policy_mapping_dict = {
    "wildfire": {
        "description": "wildfire suppression with heterogeneous agents",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class MARLlibWildfireEnv(MultiAgentEnv):
    """MARLlib용 Wildfire 다중 에이전트 환경 래퍼"""

    def __init__(self, env_config):
        """
        Parameters
        ----------
        env_config : dict
            환경 설정 딕셔너리
            map_name은 MARLlib에서 자동으로 추가됨
        """
        super().__init__()

        # MARLlib은 map_name을 env_config에 포함시킴 (제거 필요)
        map_name = env_config.pop("map_name", None)

        # wildfire-v0 환경 생성
        self.env = gym.make("wildfire-v0", **env_config)

        # 에이전트 수
        self.num_agents = self.env.num_agents

        # 에이전트 ID는 문자열 형식으로 (MARLlib 규칙)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # 액션/관찰 공간 설정 (모든 에이전트가 동일한 공간 사용)
        # wildfire 환경에서 단일 에이전트 공간 추출
        dict_action_space = self.env.action_space
        dict_obs_space = self.env.observation_space

        # wildfire 환경은 키를 문자열로 사용 ("0", "1", ...)
        single_action_space = dict_action_space["0"]
        single_obs_space = dict_obs_space["0"]

        # MARLlib 형식: 단일 에이전트 공간 (모든 에이전트가 공유)
        # Discrete 액션 공간
        self.action_space = single_action_space

        # 관찰 공간을 Dict로 래핑 (MARLlib 규칙)
        # Box observation을 {"obs": Box} 형태로 변환
        if isinstance(single_obs_space, Box):
            self.observation_space = GymDict({
                "obs": Box(
                    low=single_obs_space.low.astype(np.float32),
                    high=single_obs_space.high.astype(np.float32),
                    shape=single_obs_space.shape,
                    dtype=np.float32
                )
            })
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(single_obs_space)}")

        # 환경 설정 저장 (map_name 복원)
        if map_name is not None:
            env_config["map_name"] = map_name
        self.env_config = env_config

        print(f"MARLlib 래퍼 생성:")
        print(f"  에이전트 수: {self.num_agents}")
        print(f"  액션 공간: {self.action_space}")
        print(f"  관찰 공간: {self.observation_space}")

    def reset(self, *, seed=None):
        """환경 리셋

        Returns
        -------
        obs : dict
            {agent_id: {"obs": observation}}
        """
        obs_dict, info = self.env.reset(seed=seed)

        # wildfire env는 문자열 키("0", "1")를 사용
        # MARLlib 형식으로 변환: int -> "agent_i" 형식
        obs = {}
        for i in range(self.num_agents):
            agent_id = self.agents[i]
            # observation을 {"obs": ...} 형태로 래핑
            obs[agent_id] = {"obs": np.array(obs_dict[str(i)], dtype=np.float32)}

        return obs

    def step(self, action_dict):
        """환경 스텝 실행

        Parameters
        ----------
        action_dict : dict
            {agent_id: action}

        Returns
        -------
        obs : dict
            {agent_id: {"obs": observation}}
        rewards : dict
            {agent_id: reward}
        dones : dict
            {"__all__": bool}
        infos : dict
            빈 딕셔너리 (MARLlib 호환성)
        """
        # MARLlib은 "agent_i" 형식의 키를 사용
        # wildfire env는 문자열 숫자 키("0", "1", ...)를 기대
        env_actions = {}
        for i in range(self.num_agents):
            agent_id = self.agents[i]
            if agent_id in action_dict:
                env_actions[str(i)] = action_dict[agent_id]

        # 환경 실행 (5-tuple 반환: obs, reward, terminated, truncated, infos)
        obs_dict, reward_dict, terminated, truncated, infos_env = self.env.step(env_actions)

        # MARLlib 형식으로 변환
        observations = {}
        rewards = {}
        for i in range(self.num_agents):
            agent_id = self.agents[i]
            # observation을 {"obs": ...} 형태로 래핑
            observations[agent_id] = {"obs": np.array(obs_dict[str(i)], dtype=np.float32)}
            rewards[agent_id] = float(reward_dict[str(i)])

        # MARLlib은 "__all__" 키로 에피소드 종료 여부 확인
        done = terminated or truncated
        dones = {"__all__": done}

        # infos는 빈 딕셔너리 반환 (MARLlib 호환성)
        return observations, rewards, dones, {}

    def close(self):
        """환경 종료"""
        self.env.close()

    def render(self, mode=None):
        """환경 렌더링 (선택적)"""
        # wildfire 환경이 render를 지원하면 호출
        if hasattr(self.env, 'render'):
            return self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        """환경 정보 반환 (MARLlib 필수 메서드)

        Returns
        -------
        env_info : dict
            환경 메타데이터
        """
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env_config.get("max_steps", 300),
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
