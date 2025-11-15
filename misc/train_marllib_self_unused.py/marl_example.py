from marllib import marl

# prepare env
env = marl.make_env(environment_name="mpe", map_name="wildfire")

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=False, num_gpus=0,
          num_workers=1, share_policy='group', checkpoint_freq=50)

# # MIT License

# # Copyright (c) 2023 Replicable-MARL

# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.

# """
# An example of integrating new tasks into MARLLib
# About ma-gym: https://github.com/koulanurag/ma-gym
# doc: https://github.com/koulanurag/ma-gym/wiki

# Learn how to transform the environment to be compatible with MARLlib:
# please refer to the paper: https://arxiv.org/abs/2210.13708

# Install ma-gym before use
# """

# import numpy as np
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from gym.spaces import Dict as GymDict, Box
# from marllib import marl
# from marllib.envs.base_env import ENV_REGISTRY
# import time
# """
# RLlib MultiAgentEnv 래퍼

# wildfire-v0 환경을 RLlib의 MultiAgentEnv 형식으로 변환
# """

# import gym
# # import wildfire_environment
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from gym import spaces
# import numpy as np
# from marllib import marl
# from marllib.envs.base_env import ENV_REGISTRY

# class WildfireRLlibEnv(MultiAgentEnv):
#     """RLlib용 Wildfire 다중 에이전트 환경 래퍼"""

#     def __init__(self, env_config):
#         """
#         Parameters
#         ----------
#         env_config : dict
#             환경 설정 딕셔너리
#         """
#         super().__init__()

#         # wildfire-v0 환경 생성
#         self.env = gym.make("wildfire-v0", **env_config)

#         # 에이전트 수
#         self._num_agents = self.env.num_agents
#         self._agent_ids = set(range(self.num_agents))

#         # RLlib 새 API에서 요구하는 속성들
#         self.agents = list(range(self.num_agents))  # 현재 에피소드의 에이전트 ID 리스트
#         self.possible_agents = list(range(self.num_agents))  # 가능한 모든 에이전트 ID 리스트

#         # 액션/관찰 공간 설정 (RLlib 형식)
#         # Dict 공간에서 단일 에이전트 공간 추출
#         dict_action_space = self.env.action_space
#         dict_obs_space = self.env.observation_space

#         # 각 에이전트의 액션/관찰 공간 (모두 동일)
#         # wildfire 환경은 키를 문자열로 사용 ("0", "1", ...)
#         single_action_space = dict_action_space["0"]
#         single_obs_space = dict_obs_space["0"]

#         # 단일 공간 저장 (샘플링용)
#         self._single_action_space = single_action_space
#         self._single_obs_space = single_obs_space

#         # RLlib의 새 API stack은 dict 형태의 공간을 기대
#         # 각 에이전트 ID를 키로 하는 dict 생성
#         self._action_space = {i: single_action_space for i in range(self.num_agents)}
#         self._observation_space = {i: single_obs_space for i in range(self.num_agents)}

#         print(f"RLlib 래퍼 생성:")
#         print(f"  에이전트 수: {self.num_agents}")
#         print(f"  액션 공간: {single_action_space}")
#         print(f"  관찰 공간: {single_obs_space}")

#     def reset(self, *, seed=None, options=None):
#         """환경 리셋"""
#         obs_dict = self.env.reset(seed=seed)

#         # wildfire env는 문자열 키("0", "1")를 사용, RLlib은 int 키를 사용
#         # 문자열 키를 int로 변환
#         obs = {int(k): v for k, v in obs_dict.items()}

#         return obs

#     def step(self, action_dict):
#         """환경 스텝 실행"""
#         # RLlib은 int 키를 사용하지만, wildfire env는 문자열 키를 기대함
#         # int 키를 문자열로 변환
#         env_actions = {str(i): action_dict[i] for i in range(self.num_agents)}

#         # 환경 실행 (5-tuple 반환: obs, reward, terminated, truncated, infos)
#         # obs_dict, reward_dict, done, infos_env = self.env.step(env_actions)
#         obs_dict, reward_dict, terminated, truncated, infos_dict = self.env.step(env_actions)

#         # 결과를 RLlib 형식으로 변환 (문자열 키를 int로)
#         observations = {int(k): v for k, v in obs_dict.items()}
#         rewards = {int(k): v for k, v in reward_dict.items()}
#         infos = {int(k): v for k, v in infos_dict.items()}

#         # dones = {"__all__": done}
#         done = terminated or truncated
#         dones = {"__all__": done}

#         return observations, rewards, dones, infos
    
#     def render(self, mode='rgb_array'):
#         """
#         img : np.ndarray or None
#             rgb_array 모드일 경우 이미지 반환
#         """
#         return self.env.render(mode=mode)

#     def action_space_sample(self, agent_ids=None):
#         """액션 공간 샘플링"""
#         if agent_ids is None:
#             agent_ids = self._agent_ids
#         return {agent_id: self._single_action_space.sample() for agent_id in agent_ids}

#     def observation_space_sample(self, agent_ids=None):
#         """관찰 공간 샘플링"""
#         if agent_ids is None:
#             agent_ids = self._agent_ids
#         return {agent_id: self._single_obs_space.sample() for agent_id in agent_ids}

#     def get_agent_ids(self):
#         """에이전트 ID 목록 반환"""
#         return self._agent_ids

#     def get_action_space(self, agent_id=None):
#         """특정 에이전트의 액션 공간 반환"""
#         if agent_id is None:
#             return self._action_space
#         return self._action_space[agent_id]

#     def get_observation_space(self, agent_id=None):
#         """특정 에이전트의 관찰 공간 반환"""
#         if agent_id is None:
#             return self._observation_space
#         return self._observation_space[agent_id]

#     # RLlib이 필요로 하는 속성들
#     @property
#     def action_space(self):
#         return self._action_space

#     @property
#     def observation_space(self):
#         return self._observation_space

#     @property
#     def num_agents(self):
#         return self._num_agents

#     def close(self):
#         """환경 종료"""
#         self.env.close()

#     def get_env_info(self):
#         """MARLlib이 요구하는 환경 정보 반환"""
#         env_info = {
#             "space_obs": self._observation_space,
#             "space_act": self._action_space,
#             "num_agents": self.num_agents,
#             "episode_limit": self.env.max_steps,
#             "policy_mapping_info": {
#                 "wildfire": {
#                     "description": "cooperative wildfire suppression",
#                     "team_prefix": ("agent_",),
#                     "all_agents_one_policy": True,
#                     "one_agent_one_policy": True,
#                 }
#             }
#         }
#         return env_info


# # register all scenario with env class
# REGISTRY = {}
# REGISTRY["Wildfire"] = WildfireRLlibEnv # ?? 어디서 import?

# # provide detailed information of each scenario
# # mostly for policy sharing
# policy_mapping_dict = {
#     "wildfire": {
#         "description": "cooperative wildfire suppression",
#         "team_prefix": ("agent_",),
#         "all_agents_one_policy": True,
#         "one_agent_one_policy": True,
#     }
# }

# # must inherited from MultiAgentEnv class
# class RLlibMAGym(MultiAgentEnv):

#     def __init__(self, env_config):
#         map = env_config["map_name"]
#         env_config.pop("map_name", None)

#         self.env = REGISTRY[map](**env_config)
#         # assume all agent same action/obs space
#         self.action_space = self.env.action_space[0]
#         self.observation_space = GymDict({"obs": Box(
#             low=0.0,
#             high=1.0,
#             shape=(self.env.observation_space[0].shape[0],),
#             dtype=np.dtype("float64"))})
#         self.agents = ["red_0", "blue_0"]
#         self.num_agents = len(self.agents)
#         env_config["map_name"] = map
#         self.env_config = env_config

#     def reset(self):
#         original_obs = self.env.reset()
#         obs = {}
#         for i, name in enumerate(self.agents):
#             obs[name] = {"obs": np.array(original_obs[i])}
#         return obs

#     def step(self, action_dict):
#         action_ls = [action_dict[key] for key in action_dict.keys()]
#         o, r, d, info = self.env.step(action_ls)
#         rewards = {}
#         obs = {}
#         for i, key in enumerate(action_dict.keys()):
#             rewards[key] = r[i]
#             obs[key] = {
#                 "obs": np.array(o[i])
#             }
#         dones = {"__all__": True if sum(d) == self.num_agents else False}
#         return obs, rewards, dones, {}

#     def close(self):
#         self.env.close()

#     def render(self, mode=None):
#         self.env.render()
#         time.sleep(0.05)
#         return True

#     def get_env_info(self):
#         env_info = {
#             "space_obs": self.observation_space,
#             "space_act": self.action_space,
#             "num_agents": self.num_agents,
#             "episode_limit": 100,
#             "policy_mapping_info": policy_mapping_dict
#         }
#         return env_info


# if __name__ == '__main__':
#     # register new env
#     ENV_REGISTRY["magym"] = RLlibMAGym
#     # initialize env
#     env = marl.make_env(environment_name="magym", map_name="Checkers", abs_path="../../examples/config/env_config/magym.yaml")
#     # pick mappo algorithms
#     mappo = marl.algos.mappo(hyperparam_source="test")
#     # customize model
#     model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
#     # start learning
#     mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
#               num_workers=2, share_policy='all', checkpoint_freq=50)