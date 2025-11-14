"""
MARLlib 환경 래퍼

wildfire-v0 환경을 MARLlib의 환경 형식으로 변환
MARLlib은 내부적으로 RLlib을 사용하므로 MultiAgentEnv 기반
"""

import gym
import numpy as np
import wildfire_environment
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces
from gym.spaces import Dict as GymDict, Box, Discrete


class WildfireMARLlibEnv(MultiAgentEnv):
    """MARLlib용 Wildfire 다중 에이전트 환경 래퍼 (CTDE 지원)"""

    def __init__(self, env_config):
        """
        Parameters
        ----------
        env_config : dict
            환경 설정 딕셔너리
        """
        super().__init__()

        # wildfire-v0 환경 생성 (wrapper 없이 직접 생성)
        # gym.make()는 자동으로 wrapper를 추가하므로 직접 생성
        import wildfire_environment
        from wildfire_environment.envs.wildfire import WildfireEnv
        self.env = WildfireEnv(**env_config)

        # 에이전트 수
        self.num_agents = self.env.num_agents
        self._agent_ids = set(range(self.num_agents))

        # MARLlib/RLlib에서 요구하는 속성들
        self.agents = list(range(self.num_agents))
        self.possible_agents = list(range(self.num_agents))

        # 액션/관찰 공간 설정
        dict_action_space = self.env.action_space
        dict_obs_space = self.env.observation_space

        # 단일 에이전트 공간 추출 (wildfire는 문자열 키 사용)
        single_action_space = dict_action_space["0"]
        single_obs_space = dict_obs_space["0"]

        # 공간 저장
        self.action_space = single_action_space

        # MARLlib은 observation_space가 Dict{"obs": Box} 형식이어야 함
        from gym.spaces import Dict as GymDict
        self.observation_space = GymDict({
            "obs": Box(
                low=single_obs_space.low,
                high=single_obs_space.high,
                shape=single_obs_space.shape,
                dtype=single_obs_space.dtype
            )
        })

        # MARLlib은 state() 메소드를 통해 global state를 받아 CTDE 구현
        # Global state 차원 계산
        self.state_space = self._get_state_space()

        print(f"MARLlib 래퍼 생성:")
        print(f"  에이전트 수: {self.num_agents}")
        print(f"  액션 공간: {self.action_space}")
        print(f"  관찰 공간 (local): {self.observation_space}")
        print(f"  상태 공간 (global): {self.state_space}")

    def _get_state_space(self):
        """
        Global state space 정의
        wildfire 환경의 get_state() 메소드가 반환하는 전체 상태
        """
        # wildfire.py의 get_state() 반환 형태:
        # (obs_depth + 1, grid_size, grid_size) flattened + normalized time
        obs_depth = self.env.obs_depth  # num_agents + len(STATE_IDX_TO_COLOR_WILDFIRE)
        grid_size = self.env.grid_size
        state_dim = obs_depth * (grid_size ** 2) + 1 + 1  # +1 for time, +1 for flatten

        return Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """환경 리셋"""
        obs_dict, info = self.env.reset(seed=seed)

        # 문자열 키를 int로 변환하고 MARLlib 형식으로 wrapping
        # MARLlib expects {"obs": observation_array}
        obs = {int(k): {"obs": v} for k, v in obs_dict.items()}

        return obs

    def step(self, action_dict):
        """
        환경 스텝 실행

        MARLlib MAPPO를 위해 각 에이전트의 info에 global state 추가
        """
        # int 키를 문자열로 변환 (wildfire env 형식)
        env_actions = {str(i): action_dict[i] for i in action_dict.keys()}

        # 환경 실행
        obs_dict, reward_dict, terminated, truncated, infos_env = self.env.step(env_actions)

        # Global state 가져오기 (CTDE의 Centralized Critic용)
        global_state = self.env.get_state()

        # 결과를 MARLlib 형식으로 변환
        # MARLlib expects {"obs": observation_array}
        observations = {int(k): {"obs": v} for k, v in obs_dict.items()}
        rewards = {int(k): v for k, v in reward_dict.items()}

        # 각 에이전트의 info에 global state 추가
        # MARLlib의 centralized critic은 이 state를 사용
        infos = {}
        for k, v in infos_env.items():
            agent_id = int(k)
            infos[agent_id] = v.copy() if isinstance(v, dict) else {}
            infos[agent_id]["state"] = global_state  # CTDE를 위한 global state

        # MARLlib/RLlib 형식
        dones = {"__all__": terminated or truncated}

        return observations, rewards, dones, infos

    def state(self):
        """
        Global state 반환 (MARLlib CTDE용)

        MARLlib의 centralized critic이 이 메소드를 호출하여
        전체 환경 상태를 받아 가치 함수를 평가합니다.
        """
        return self.env.get_state()

    def get_env_info(self):
        """
        MARLlib이 요구하는 환경 정보 반환
        """
        # 환경 설정에서 에이전트 타입 정보 가져오기
        num_helicopters = getattr(self.env, 'num_helicopters', 0)
        num_trucks = getattr(self.env, 'num_trucks', 0)
        num_crews = getattr(self.env, 'num_crews', 0)

        # 정책 매핑 정보 생성 (map_name별로)
        if num_helicopters > 0 or num_trucks > 0 or num_crews > 0:
            # 이질적 에이전트: 3개 정책
            policy_mapping_info = {
                "custom": {  # map_name과 일치
                    "description": "heterogeneous agents (helicopter, truck, crew)",
                    "team_prefix": ("helicopter_", "truck_", "crew_"),
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": False,
                }
            }
        else:
            # 동질적 에이전트: 1개 정책
            policy_mapping_info = {
                "custom": {  # map_name과 일치
                    "description": "homogeneous agents",
                    "team_prefix": ("agent_",),
                    "all_agents_one_policy": True,
                    "one_agent_one_policy": False,
                }
            }

        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.max_steps,
            "policy_mapping_info": policy_mapping_info,  # 시나리오별 정책 매핑
        }

    def _get_policy_mapping_info(self):
        """
        이질적 에이전트를 위한 정책 매핑 정보
        """
        # 환경 설정에서 에이전트 타입 정보 가져오기
        num_helicopters = getattr(self.env, 'num_helicopters', 0)
        num_trucks = getattr(self.env, 'num_trucks', 0)
        num_crews = getattr(self.env, 'num_crews', 0)

        if num_helicopters > 0 or num_trucks > 0 or num_crews > 0:
            # 3개의 정책 그룹
            policy_mapping_info = {
                "all_agents": list(range(self.num_agents)),
                "team_prefix": ("helicopter_", "truck_", "crew_"),
            }
        else:
            # 단일 정책 (모든 에이전트가 동일)
            policy_mapping_info = {
                "all_agents": list(range(self.num_agents)),
                "team_prefix": ("agent_",),
            }

        return policy_mapping_info

    def close(self):
        """환경 종료"""
        self.env.close()

    # RLlib 호환성을 위한 추가 속성
    @property
    def get_agent_ids(self):
        """에이전트 ID 목록 반환"""
        return self._agent_ids
