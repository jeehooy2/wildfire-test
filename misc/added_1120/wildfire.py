"""Defines the WildfireEnv class, which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire
"""

from collections import OrderedDict
import random
from gym.spaces import Box, Dict, Discrete
import numpy as np
from wildfire_environment.multigrid import MultiGridEnv
from wildfire_environment.core.world import WildfireWorld
from wildfire_environment.core.agent import WildfireActions, Agent, Helicopter, Truck, Crew
from wildfire_environment.core.object import Tree
from wildfire_environment.core.grid import Grid
from wildfire_environment.core.constants import (
    STATE_TO_IDX_WILDFIRE,
    TILE_PIXELS,
    STATE_IDX_TO_COLOR_WILDFIRE,
    COLORS,
)
from wildfire_environment.utils.window import Window
from wildfire_environment.utils.misc import (
    render_agent_tiles,
    get_initial_fire_coordinates,
)


class WildfireEnv(MultiGridEnv):
    """Grid environment which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire"""

    def __init__(
        self,
        alpha=0.05,
        beta=0.9,
        delta_beta=0.54,
        size=17,
        num_agents=2,
        agent_start_positions=((1, 1), (15, 15)),
        agent_colors=("red", "blue", "yellow"),
        agent_groups=None,
        agent_view_size=10,
        initial_fire_size=1,
        max_steps=100,
        partial_obs=False,
        actions_set=WildfireActions,
        render_mode="rgb_array",
        render_selfish_region_boundaries=False,
        cooperative_reward=False,
        selfishness_weight=0.2,
        log_selfish_region_metrics=False,
        selfish_region_xmin=None,
        selfish_region_xmax=None,
        selfish_region_ymin=None,
        selfish_region_ymax=None,
        reward_shaping=None,
        reward_shaping_config=None,
        num_helicopters=0,
        num_trucks=0,
        num_crews=0,
    ):
        """Create a WildfireEnv environment

        Parameters
        ----------
        alpha : float, optional
            parameter for the wildfire dynamics model, by default 0.05
        beta : float, optional
            parameter for the wildfire dynamics model, by default 0.99
        delta_beta : float, optional
            parameter for the wildfire dynamics model, by default 0
        size : int, optional
            side of the square gridworld, by default 17
        num_agents : int, optional
            number of UAV agents, by default 2
        agent_start_positions : tuple[tuple[int,int]], optional
            tuple of tuples containing the start positions of the agents, in order of agent index. By default ((1, 1), (15, 15))
        agent_colors : tuple[str,str], optional
            tuple of strings of color names of all agents (or groups if agents are grouped) in order of increasing index. All agents in a group have the same color. The strings should be keys in the COLORS dictionary in constants.py. Only applicable if cooperative_reward is False. Fully cooperative agents have light_blue color by default. By default self-interested agents have red and blue colors
        agent_groups : tuple[tuple], optional
            tuple of tuples containing the indices (in ascending order) of agents in each group. Only applicable if cooperative_reward is False. By default None
        agent_view_size : int, optional
            side of the square region visible to an agent with partial observability, by default 10. Only applicable if partial_obs is True
        initial_fire_size : int, optional
            side of the square shaped initial fire region, by default 1
        max_steps : int, optional
            maximum number of steps in an episode, by default 100
        partial_obs : bool, optional
            whether agents have partial observability, by default False
        actions_set : WildfireActions, optional
            action space of the agents. All agents have the same action space. By default WildfireActions.
        render_mode : str, optional
            mode of rendering the environment, by default "rgb_array"
        render_selfish_region_boundaries : bool, optional
            whether to render boundaries of selfish regions, by default False
        cooperative_reward : bool, optional
            whether the agents use a cooperative reward, by default False. If True, the agents are fully cooperative and receive the same reward.
        selfishness_weight : float, optional
            parameter to control selfishness in the Markov game reward functions, by default 0.2. Only applicable if cooperative_reward is False. Should be in the range [0,1).
        log_selfish_region_metrics : bool, optional
            whether to log metrics related to trees in selfish regions, by default False
        selfish_region_xmin : list, optional
            list containing x-coordinates of the left boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_xmax : list, optional
            list containing x-coordinates of the right boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_ymin : list, optional
            list containing y-coordinates of the top boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_ymax : list, optional
            list containing y-coordinates of the bottom boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        reward_shaping : str, optional
            reward shaping function to use. Options: None, "cooperative", "ramadan", "individual". By default None (uses base rewards from environment).
        reward_shaping_config : dict, optional
            configuration dictionary for reward shaping function. By default None (uses default config from reward_functions.py).
        num_helicopters : int, optional
            number of helicopter agents (fast, high efficiency), by default 0
        num_trucks : int, optional
            number of truck agents (medium speed, high efficiency), by default 0
        num_crews : int, optional
            number of crew agents (slow, low efficiency), by default 0
        """
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta
        self.num_agents = num_agents
        self.agent_start_positions = agent_start_positions
        self.agent_colors = agent_colors
        self.agent_groups = agent_groups
        if agent_groups:
            self.idx_to_group = {}
            for i, group in enumerate(agent_groups):
                for agent_index in group:
                    self.idx_to_group[agent_index] = i
        # observation vector of each agent is concatenation of obs_depth number of one-hot encodings, see paper for details. len(STATE_IDX_TO_COLOR_WILDFIRE) = the number of tree states
        self.obs_depth = self.num_agents + len(STATE_IDX_TO_COLOR_WILDFIRE)
        self.max_steps = max_steps
        self.world = WildfireWorld
        self.grid_size = size
        self.grid_size_without_walls = size - 2
        self.initial_fire_size = initial_fire_size
        self.burnt_trees = 0
        self.unburnt_trees = []
        self.trees_on_fire = 0
        self.cooperative_reward = cooperative_reward
        self.selfishness_weight = selfishness_weight
        if selfishness_weight < 0 or selfishness_weight >= 1:
            raise ValueError("Selfishness weight should be in the range [0,1).")
        self.render_selfish_region_boundaries = render_selfish_region_boundaries
        self.log_selfish_region_metrics = log_selfish_region_metrics
        if self.log_selfish_region_metrics:
            # initialize attributes for logging metrics related to trees in selfish regions
            self.selfish_xmin = np.array(selfish_region_xmin)
            self.selfish_xmax = np.array(selfish_region_xmax)
            self.selfish_ymin = np.array(selfish_region_ymin)
            self.selfish_ymax = np.array(selfish_region_ymax)
            # raise error if xmin > xmax or ymin > ymax
            if np.any(self.selfish_xmin > self.selfish_xmax) or np.any(
                self.selfish_ymin > self.selfish_ymax
            ):
                raise ValueError(
                    "Invalid selfish region. xmin should be less than or equal to xmax and ymin should be less than or equal to ymax for every selfish region."
                )
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))
            self.selfish_region_size = (
                self.selfish_xmax
                - self.selfish_xmin
                + np.ones(len(selfish_region_xmin))
            ) * (
                self.selfish_ymax
                - self.selfish_ymin
                + np.ones(len(selfish_region_ymin))
            )

        # 다양한 타입의 에이전트 생성 (Helicopter, Truck, Crew)
        if num_helicopters > 0 or num_trucks > 0 or num_crews > 0:
            # 타입별 에이전트 수 검증
            if num_helicopters + num_trucks + num_crews != num_agents:
                raise ValueError(
                    f"Sum of agent types ({num_helicopters + num_trucks + num_crews}) "
                    f"must equal num_agents ({num_agents})"
                )

            agents = []
            agent_idx = 0

            # Helicopter 생성 (red)
            for _ in range(num_helicopters):
                agents.append(Helicopter(
                    world=self.world,
                    index=agent_idx,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color="red",
                ))
                agent_idx += 1

            # Truck 생성 (yellow)
            for _ in range(num_trucks):
                agents.append(Truck(
                    world=self.world,
                    index=agent_idx,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color="yellow",
                ))
                agent_idx += 1

            # Crew 생성 (white)
            for _ in range(num_crews):
                agents.append(Crew(
                    world=self.world,
                    index=agent_idx,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color="white",
                ))
                agent_idx += 1

        elif self.cooperative_reward:
            # initialize cooperative agents (기본 Agent)
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color="light_blue",
                )
                for i in range(self.num_agents)
            ]
        elif self.agent_groups:
            # initialize self-interested agents with different colors
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color=self.agent_colors[self.idx_to_group[i]],
                )
                for i in range(self.num_agents)
            ]
        else:
            # initialize self-interested agents with different colors
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color=self.agent_colors[i],
                )
                for i in range(self.num_agents)
            ]

        super().__init__(
            agents=agents,
            grid_size=size,
            max_steps=max_steps,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=actions_set,
            world=self.world,
            render_mode=render_mode,
        )

        # Initialize movement accumulator for deterministic speed mechanics
        # !!! TODO: 나중에 꼭 체크!! !!!
        # Current implementation: Option 1 (Action Repetition)
        # - Agent repeats the same action until it actually moves
        # - Simple but has learning issues (delayed action effect)
        # Future options to consider:
        #   - Option 2: Frame skip at policy level (complex RLlib integration)
        #   - Option 3: Speed as movement cost (better for learning)
        for agent in self.agents:
            agent.movement_accumulator = 0.0
            agent.pending_action = None  # Action to repeat until movement occurs

        # Initialize reward shaping
        self.reward_shaping = reward_shaping
        if reward_shaping and reward_shaping.lower() != "none":
            from wildfire_environment.envs.reward_functions import get_reward_function, DEFAULT_CONFIGS
            self.reward_function = get_reward_function(reward_shaping)
            self.reward_config = reward_shaping_config or DEFAULT_CONFIGS.get(reward_shaping, {})
            # State tracking variables for reward shaping
            self.cumulative_extinguished = 0
            self.cumulative_actions = 0
            self.total_trees = (self.grid_size_without_walls) ** 2
        else:
            self.reward_function = None  # Use base rewards

        self.helper_grid = None
        self.observation_space = self._set_observation_space()
        self.action_space = Dict(
            {f"{a.index}": Discrete(len(self.actions)) for a in self.agents}
        )

    def _set_observation_space(self) -> Dict:
        """Set the observation space for each agent in the environment. All agents possess the same observation space

        Returns
        -------
        observation_space : dict
            dictionary where each key is an agent's index and the value is the observation space
            for that agent
        """
        # TODO: 관찰 공간 수정 (partial_obs) ###
        # observation vector of agent is the concatenation of obs_depth number of one-hot encodings where each encoding has grid_size_without_walls number of elements valued either 0 or 1. Additionally, the observation vector contains the normalized time step at the end
        # low = np.full(self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 0)
        # high = np.full(
        #     self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 1
        # )
        # observation_space = Dict(
        #     {
        #         f"{a.index}": Box(
        #             low=low,
        #             high=high,
        #             dtype=np.float32,
        #         )
        #         for a in self.agents
        #     }
        # )

        if self.partial_obs:
            # Partial observability: 에이전트 중심 partial_obs_size x partial_obs_size 그리드 (flattened)
            # obs_depth = num_agents + len(tree states)
            # Shape: obs_depth * partial_view_size * partial_view_size + 1 (timestep)
            partial_view_size = self.agents[0].partial_obs_size
            obs_size = self.obs_depth * (partial_view_size ** 2) + 1

            low = np.full(obs_size, 0)
            high = np.full(obs_size, 1)
            observation_space = Dict(
                {
                    f"{a.index}": Box(
                        low=low,
                        high=high,
                        dtype=np.float32,
                    )
                    for a in self.agents
                }
            )
        else:
            # Full observability: 전체 그리드 (기존 방식)
            low = np.full(self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 0)
            high = np.full(
                self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 1
            )
            observation_space = Dict(
                {
                    f"{a.index}": Box(
                        low=low,
                        high=high,
                        dtype=np.float32,
                    )
                    for a in self.agents
                }
            )
        ### 관찰 공간 수정 끝! ###

        return observation_space

    def _gen_grid(self, width, height, state=None):
        """Generate the grid for the environment

        Parameters
        ----------
        width : int
            width of the grid
        height : int
            height of the grid
        state : ndarray, optional
            specifies the initial state of the environment, by default None.
            If none, it is chosen uniformly at random from the assumed initial state distribution
        """
        self.grid = Grid(width, height, self.world)

        # generate the walls of the grid
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        agent_start_pos = []
        if state is not None:
            # store positions of agents and trees on fire as per the specified initial state.
            state = state[:-1].reshape(
                (
                    self.obs_depth + 1,
                    self.grid_size,
                    self.grid_size,
                ),
            )
            initial_fire = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if state[1, j, i]:
                        # i and j are swapped because the y-coordinate specifies the row, while the x-coordinate specifies the column.
                        initial_fire.append((i, j))
                    for o in self.agents:
                        index = o.index
                        if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + index, j, i]:
                            # i and j are swapped because the y-coordinate specifies the row, while the x-coordinate specifies the column.
                            agent_start_pos.append((i, j))

        else:
            # choose location of initial fire uniformly at random
            if self.initial_fire_size % 2 == 0:
                # for even sized initial fires, choose location of top left corner of fire region uniformly at random
                top_left_corner = (
                    self._rand_int(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                    self._rand_int(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *top_left_corner,
                    self.grid_size,
                    self.initial_fire_size,
                )
            else:
                # for odd sized initial fires, choose location of center of fire region uniformly at random
                fire_square_center = (
                    self._rand_int(
                        1 + ((self.initial_fire_size - 1) // 2),
                        self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) // 2),
                    ),
                    self._rand_int(
                        1 + ((self.initial_fire_size - 1) // 2),
                        self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) // 2),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *fire_square_center,
                    self.grid_size,
                    self.initial_fire_size,
                )
            # agent_start_pos is specified during environment initialization
            agent_start_pos = self.agent_start_positions

        for pos in initial_fire:
            region = "common"
            if self.log_selfish_region_metrics:
                # update count of trees on fire in selfish regions
                if self.agent_groups:
                    for i, _ in enumerate(self.agent_groups):
                        if self.in_selfish_region(pos[0], pos[1], i):
                            # selfish region is identified by the lowest index among indices of the corresponding group of selfish agents
                            region = f"{i}"
                            self.selfish_region_trees_on_fire[i] += 1
                            break
                else:
                    for a in self.agents:
                        if self.in_selfish_region(pos[0], pos[1], a.index):
                            # selfish region is identified by the lowest index among indices of the corresponding selfish agent
                            region = f"{a.index}"
                            self.selfish_region_trees_on_fire[a.index] += 1
                            break
            # insert tree on fire in grid
            self.put_obj(
                Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"], region=region),
                int(pos[0]),
                int(pos[1]),
            )

        # update counts of trees on fire and healthy trees
        self.trees_on_fire += self.initial_fire_size**2
        num_healthy_trees = self.grid_size_without_walls**2 - len(initial_fire)

        # insert healthy tree in grid
        for _ in range(num_healthy_trees):
            tree_obj = Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"])
            self.place_obj(tree_obj)
            if self.log_selfish_region_metrics:
                # check if tree is in a selfish region, and update region attribute of tree if it is
                if self.agent_groups:
                    for i, _ in enumerate(self.agent_groups):
                        if self.in_selfish_region(
                            *(tree_obj.pos), i  # pylint: disable=not-an-iterable
                        ):
                            tree_obj.region = f"{i}"
                            break
                else:
                    for a in self.agents:
                        if self.in_selfish_region(
                            *(tree_obj.pos), a.index  # pylint: disable=not-an-iterable
                        ):
                            tree_obj.region = f"{a.index}"
                            break

        # helper grid is a work around for grid being unable to store multiple objects at a single cell. It does not contain agents
        self.helper_grid = self.grid.copy()

        # create list of unburnt trees. initial state does not have burnt trees.
        for c in self.helper_grid.grid:
            if c is not None and c.type == "tree":
                self.unburnt_trees.append(c)

        # insert agents in grid
        for i, a in enumerate(self.agents):
            self.place_agent(a, pos=agent_start_pos[i])
            self.helper_grid.get(*agent_start_pos[i]).agent_above = True

    def _get_obs(self):
        """Get observation vectors of all agents in the environment.

        Returns
        -------
        agent_obs: list(ndarray)
            list of agent observations where the element at i^th list index is the observation vector for the agent with index i.
        """
        if self.partial_obs:
            return self._get_obs_partial()
        else:
            return self._get_obs_full()

    def _get_obs_full(self):
        """Get full observation (entire grid) for all agents."""
        # initialize list of observation vector of each agent
        agent_obs = [
            np.zeros(
                (
                    self.obs_depth,
                    self.grid_size_without_walls + 1,
                    self.grid_size_without_walls + 1,
                ),
                dtype=np.float32,
            )
            for _ in range(self.num_agents)
        ]

        # update walls and tree states in agent observations
        for obj in self.helper_grid.grid:
            if obj is None:
                continue
            i, j = obj.pos
            for a in self.agents:
                # convert to agent centered coordinates
                nc = [i - a.pos[0], j - a.pos[1]]
                # wrap around to get agent centered toroidal coordinates
                if nc[0] < 0:
                    nc[0] += self.grid_size_without_walls + 1
                if nc[1] < 0:
                    nc[1] += self.grid_size_without_walls + 1
                # update agent's observation. # switch x and y coordinates because the y-coordinate specifies the row, while the x-coordinate specifies the column
                if obj.type == "tree":
                    agent_obs[a.index][obj.state, nc[1], nc[0]] = 1
                elif obj.type == "wall":
                    agent_obs[a.index][
                        len(STATE_IDX_TO_COLOR_WILDFIRE), nc[1], nc[0]
                    ] = 1

        # for each agent, update other agents' positions in agent observations
        for a in self.agents:
            for o in self.agents:
                if o.index != a.index:
                    idx = o.index - int(np.heaviside(o.index - a.index, 0))
                    # convert to agent centered coordinates
                    nc = [
                        o.pos[0] - a.pos[0],
                        o.pos[1] - a.pos[1],
                    ]
                    # wrap around to get agent centered toroidal coordinates
                    if nc[0] < 0:
                        nc[0] += self.grid_size_without_walls + 1
                    if nc[1] < 0:
                        nc[1] += self.grid_size_without_walls + 1
                    agent_obs[a.index][
                        len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + idx,
                        nc[1],
                        nc[0],
                    ] = 1

        # flatten, and append normalized time step at the end of, each agent observation
        for a in self.agents:
            agent_obs[a.index] = np.append(
                agent_obs[a.index].flatten(),
                np.array(self.step_count / self.max_steps, dtype=np.float32),
            )
        return agent_obs

    def _get_obs_partial(self):
        """Get partial observation (centered grid around each agent) for all agents."""
        # TODO: partial obs 관찰 생성 ###
        agent_obs = []

        for a in self.agents:
            partial_view_size = a.partial_obs_size  # 에이전트의 partial_obs_size 사용 (기본값: 5)
            half_view = partial_view_size // 2  # 2 (5x5의 경우)

            # obs_depth = num_agents + num_tree_states
            obs = np.zeros(
                (
                    self.obs_depth,
                    partial_view_size,
                    partial_view_size,
                ),
                dtype=np.float32,
            )

            agent_x, agent_y = a.pos

            # 에이전트 중심으로 partial_view_size x partial_view_size 영역 추출
            for dx in range(-half_view, half_view + 1):
                for dy in range(-half_view, half_view + 1):
                    world_x = agent_x + dx
                    world_y = agent_y + dy

                    # 그리드 범위 체크 (벽 포함)
                    if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                        # partial_view_size x partial_view_size 그리드 내 로컬 좌표
                        local_x = dx + half_view
                        local_y = dy + half_view

                        # 이 위치의 객체 가져오기
                        obj = self.helper_grid.get(world_x, world_y)

                        if obj is not None:
                            if obj.type == "tree":
                                obs[obj.state, local_y, local_x] = 1
                            elif obj.type == "wall":
                                obs[
                                    len(STATE_IDX_TO_COLOR_WILDFIRE), local_y, local_x
                                ] = 1

            # partial_view_size 영역 내의 다른 에이전트 위치 추가
            for o in self.agents:
                if o.index != a.index:
                    other_x, other_y = o.pos

                    # 다른 에이전트가 뷰 내에 있는지 확인
                    dx = other_x - agent_x
                    dy = other_y - agent_y

                    if abs(dx) <= half_view and abs(dy) <= half_view:
                        local_x = dx + half_view
                        local_y = dy + half_view

                        idx = o.index - int(np.heaviside(o.index - a.index, 0))
                        obs[
                            len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + idx,
                            local_y,
                            local_x,
                        ] = 1

            # Flatten and append normalized time step
            obs = np.append(
                obs.flatten(),
                np.array(self.step_count / self.max_steps, dtype=np.float32),
            )
            agent_obs.append(obs)

        return agent_obs
        ### partial obs 관찰 생성 끝! ###

    def get_state(self):
        """Get the state representation of the environment.

        Returns
        -------
        ndarray
            state representation of the environment
        """
        # initialize array to store state vector
        s = np.zeros(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float32,
        )

        # update tree states and walls in state representation
        for o in self.helper_grid.grid:
            # switch x and y coordinates because the y-coordinate specifies the row, while the x-coordinate specifies the column
            if o.type == "tree":
                s[o.state, o.pos[1], o.pos[0]] = 1
            if o.type == "wall":
                s[len(STATE_IDX_TO_COLOR_WILDFIRE), o.pos[1], o.pos[0]] = 1

        # update agent positions in state representation
        for a in self.agents:
            s[
                len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + a.index,
                a.pos[1],
                a.pos[0],
            ] = 1

        # flatten, and append normalized time step at the end of, state representation
        s = np.append(
            s.flatten(),
            np.array(self.step_count / self.max_steps, dtype=np.float32),
        )
        return s

    def get_state_interpretation(self, state, print_interpretation=True):
        """Get human readable interpretation of the state of the environment

        Parameters
        ----------
        state : ndarray
            state representation of the environment
        print_interpretation : bool, optional
           whether to print the interpretation of the state, by default True. Human readable interpretation refers to printing the positions of trees on fire, agents, and the time step.

        Returns
        -------
        trees_on_fire : list[tuple[int,int]]
            list of tuples containing position coordinates (x,y) of trees on fire.
        time_step : float
            normalized time step of the episode at which time the state was recorded
        """
        time_step = state[-1]
        state = state[:-1].reshape(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
        )
        if print_interpretation:
            print("-------------------------------------------------------------")
            print("State interpretation:")
        trees_on_fire = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if state[1, j, i] == 1:
                    if print_interpretation:
                        print(f"Tree at position {(i,j)} is on fire.")
                    trees_on_fire.append((i, j))
                for o in self.agents:
                    index = o.index
                    if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + index, j, i] == 1:
                        if print_interpretation:
                            print(f"Agent {o.index} is at position {(i,j)}.")
        if print_interpretation:
            print(f"Time step: {time_step}")
            print("-------------------------------------------------------------")
        return trees_on_fire, time_step

    def construct_state(self, trees_on_fire, agent_pos, time_step: int):
        """Construct the state representation vector of the environment for given positions of trees on fire and agents

        Parameters
        ----------
        trees_on_fire : list
            list of tuples containing position coordinates (x,y) of trees on fire.
        agent_pos : list
            list of tuples containing position coordinates (x,y) of agents, in order of agent index.
        time_step : int
            normalized time step of the episode at which time the state was recorded.

        Returns
        -------
        state : ndarray
            state representation of the environment
        """
        state = np.zeros(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float32,
        )
        # update tree states and walls in state representation. There are no burnt trees in the state
        state[0, :, :] = 1
        for pos in trees_on_fire:
            state[1, pos[1], pos[0]] = 1
            state[0, pos[1], pos[0]] = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (
                    i == 0
                    or i == self.grid_size - 1
                    or j == 0
                    or j == self.grid_size - 1
                ):
                    state[len(STATE_IDX_TO_COLOR_WILDFIRE), j, i] = 1
                    state[0, j, i] = 0

        # update agent positions in state representation
        for i, pos in enumerate(agent_pos):
            state[
                len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + i,
                pos[1],
                pos[0],
            ] = 1

        # flatten, and append normalized time step at the end of, state representation
        state = np.append(
            state.flatten(),
            np.array(time_step / self.max_steps, dtype=np.float32),
        )
        return state

    def reset(self, seed=None, state=None):
        """Reset the state of the environment

        Parameters
        ----------
        seed : int, optional
            seed for random number generator, by default None
        state : ndarray, optional
            specifies the initial state of the environment upon reset, by default None.
            If none, initial state is chosen uniformly at random from initial state distribution.

        Returns
        -------
        obs : OrderedDict
            dictionary where each key is the agent index and the value is the observation vector for that agent.
        info : dict
            dictionary containing additional information about the environment.
            Here, it contains the number of burnt trees in the environment after reset
        """
        # reset environment attributes
        self.burnt_trees = 0
        self.trees_on_fire = 0
        self.unburnt_trees = []
        if self.log_selfish_region_metrics:
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))

        # reset movement accumulators for all agents
        for agent in self.agents:
            agent.movement_accumulator = 0.0
            agent.pending_action = None  # Clear pending actions on reset

        # reset reward shaping 상태 변수
        if self.reward_function is not None:
            self.cumulative_extinguished = 0
            self.cumulative_actions = 0

        # reset the grid
        # Temporarily set partial_obs to False to prevent parent's gen_obs() call
        # which would fail with our partial observation setup
        original_partial_obs = self.partial_obs
        self.partial_obs = False

        if state is not None:
            super().reset(seed=seed, state=state)
        else:
            super().reset(seed=seed)

        self.partial_obs = original_partial_obs

        # get agent observations (use our custom _get_obs() instead of parent's)
        agent_obs = self._get_obs()
        obs = OrderedDict({f"{a.index}": agent_obs[a.index] for a in self.agents})

        # create info dictionary
        info = {"burnt trees": self.burnt_trees}
        return obs, info

    def move_agent(self, i, next_pos):
        """Move agent to a new position in the grid

        Parameters
        ----------
        i : int
            index of agent to be moved
        next_pos : tuple[int, int]
            coordinates of new position

        Notes
        -----
        When an agent moves through cells (when speed > 1.0), the agent applies its efficiency
        to any burning trees it passes through (not just the destination).
        The trees along the path (excluding start, including destination) will have the same
        probability of being extinguished as if the agent ended there.
        """
        # 에이전트가 이동할 때 지나는 경로상의 모든 cell에 efficiency 적용
        # (특히 속도가 높아서 한 step에 여러 cell을 이동할 때)
        current_pos = np.array(self.agents[i].pos)
        next_pos_arr = np.array(next_pos)

        # 이동 경로 계산: 시작점 포함, 중간 지점, 도착점 포함
        path = self._get_movement_path(current_pos, next_pos_arr)

        # 이동 경로상의 모든 tree에 효율 적용 (시작점만 제외)
        # 중간 지점과 도착점 모두 agents_passed_through에 일관되게 기록
        agent_efficiency = getattr(self.agents[i], 'efficiency', 1.0)
        for path_pos in path[1:]:  # 시작점(인덱스 0) 제외, 중간 + 도착점 포함
            tree_on_path = self.helper_grid.get(*path_pos)
            if tree_on_path is not None and tree_on_path.type == "tree" and tree_on_path.state == 1:
                # 이 tree를 나중에 처리하기 위해 추적
                if not hasattr(tree_on_path, 'agents_passed_through'):
                    tree_on_path.agents_passed_through = []
                tree_on_path.agents_passed_through.append(agent_efficiency)

        # Update agent direction based on movement
        if self.agents[i].pos is not None:
            direction = np.array(next_pos) - np.array(self.agents[i].pos)
            # Find matching direction in dir_to_vec
            for dir_idx, vec in enumerate(self.agents[i].dir_to_vec):
                if np.array_equal(vec, direction):
                    self.agents[i].dir = dir_idx
                    break

        # add agent to grid in new position
        self.grid.set(*next_pos, self.agents[i])

        # get tree in agent's old position from helper grid and add tree to grid
        tree = self.helper_grid.get(*self.agents[i].pos)
        tree.agent_above = False
        self.grid.set(*self.agents[i].pos, tree)

        # update attributes
        next_tree = self.helper_grid.get(*next_pos)
        # agent_above flag은 더 이상 fire dynamics에서 사용되지 않음
        # (agents_passed_through가 모든 경로상 tree를 처리함)
        # 하지만 호환성을 위해 유지
        next_tree.agent_above = True
        self.agents[i].pos = next_pos

    def _get_movement_path(self, start_pos, end_pos):
        """
        Get all cells that an agent passes through when moving from start_pos to end_pos.
        Uses Bresenham-like line algorithm for diagonal movement.

        Parameters
        ----------
        start_pos : np.ndarray
            Starting position [x, y]
        end_pos : np.ndarray
            Ending position [x, y]

        Returns
        -------
        list of np.ndarray
            List of positions along the path (including start and end)

        Notes
        -----
        Caller should use path[1:] to exclude start position only
        for agents_passed_through tracking. Both intermediate and end positions are included.
        """
        # Simple implementation: for now, just return start and end
        # This can be improved to track all cells for diagonal movement
        if np.array_equal(start_pos, end_pos):
            return [start_pos]

        # For cardinal/diagonal moves (single step), return both positions
        # For multi-cell moves, we'd need Bresenham algorithm
        # For now, simple approach: all cells along the line
        path = [start_pos.copy()]

        # Get the direction
        diff = end_pos - start_pos
        steps = int(np.max(np.abs(diff)))

        if steps == 0:
            return [start_pos]

        # Generate intermediate points
        for step in range(1, steps + 1):
            intermediate = start_pos + (diff / steps) * step
            path.append(np.round(intermediate).astype(int))

        # Remove duplicates while preserving order
        unique_path = []
        for pos in path:
            if len(unique_path) == 0 or not np.array_equal(pos, unique_path[-1]):
                unique_path.append(pos)

        return unique_path

    def neighbors_on_fire(self, tree_pos) -> int:
        """Get the number of neighboring trees on fire for a given tree.
           Neighbors are adjacent trees in 8 directions (cardinal + diagonal). A tree can have at most 8 neighbors

        Parameters
        ----------
        tree_pos : tuple[int, int]
            position coordinates of the tree whose neighbors are to be checked

        Returns
        -------
        num : int
            the number of neighboring trees on fire
        """
        num = 0
        relative_pos = [
            np.array([1, 0]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([1, 1]),
            np.array([1, -1]),
            np.array([-1, 1]),
            np.array([-1, -1]),
        ]
        for r in relative_pos:
            neighbor_pos = tree_pos + r
            if neighbor_pos[0] >= 0 and neighbor_pos[0] < self.helper_grid.width:
                if neighbor_pos[1] >= 0 and neighbor_pos[1] < self.helper_grid.height:
                    o = self.helper_grid.get(*neighbor_pos)
                    if o is not None and o.type == "tree":
                        if o.state == 1:
                            num += 1
        return num

    def in_selfish_region(self, i: int, j: int, region_index: int) -> bool:
        """Check if given tree is in region of selfish interest with given index

        Parameters
        ----------
        i : int
            x-coordinate of tree position
        j : int
            y-coordinate of tree position
        region_index : int
            index of selfish region. Same as index of corresponding selfish agent (or group of selfish agents).

        Returns
        -------
        bool
            True, if tree is in the region of selfish interest. Otherwise, False
        """
        return (
            i >= self.selfish_xmin[region_index]
            and i <= self.selfish_xmax[region_index]
            and j >= self.selfish_ymin[region_index]
            and j <= self.selfish_ymax[region_index]
        )

    def _on_fire_boundary(self, i, j):
        """Check if the tree at position (i,j) is on the fire boundary. Tree is on the fire boundary if it has at least one neighbor that is healthy.

        Parameters
        ----------
        i : int
            x-coordinate of tree position
        j : int
            y-coordinate of tree position

        Returns
        -------
        bool
            True, if the tree at position (i,j) is on the fire boundary, otherwise False.
        """
        for r in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            o = self.helper_grid.get(i + r[0], j + r[1])
            if o is not None and o.type == "tree" and o.state == 0:
                return True
        return False

    def step(self, actions):
        """Take a step in the environment. Wildfire dynamics are propagated by one time step, and agents move according to their actions."""
        self.step_count += 1
        actions = list(actions.values())
        terminated = False
        truncated = False

        # Save initial positions to track which agents actually moved
        initial_positions = {i: tuple(agent.pos) for i, agent in enumerate(self.agents)}

        # 1) Move agents sequentially, in random order
        order = np.random.permutation(len(actions))
        for i in order:
            if actions[i] == self.actions.STILL:
                continue

            # Get agent speed (default 1.0 if not defined)
            agent_speed = self.agents[i].speed if hasattr(self.agents[i], 'speed') else 1.0

            # Deterministic movement accumulator system
            # Accumulate movement based on speed
            self.agents[i].movement_accumulator += agent_speed

            # Determine number of moves from accumulator (deterministic)
            num_moves = int(self.agents[i].movement_accumulator)
            self.agents[i].movement_accumulator -= num_moves

            # Example behavior:
            # - Speed 0.5: moves once every 2 steps (step 1: 0 moves, step 2: 1 move)
            # - Speed 1.0: moves once every step
            # - Speed 1.5: alternates 1 and 2 moves (step 1: 1 move, step 2: 2 moves, step 3: 1 move...)

            # Execute the moves
            for _ in range(num_moves):
                if actions[i] == self.actions.NORTH:
                    next_pos = self.agents[i].north_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)
                elif actions[i] == self.actions.SOUTH:
                    next_pos = self.agents[i].south_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)
                elif actions[i] == self.actions.EAST:
                    next_pos = self.agents[i].east_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)
                elif actions[i] == self.actions.WEST:
                    next_pos = self.agents[i].west_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)
                elif actions[i] == self.actions.NORTH_EAST:
                    next_pos = self.agents[i].north_east_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)
                elif actions[i] == self.actions.NORTH_WEST:
                    next_pos = self.agents[i].north_west_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)
                elif actions[i] == self.actions.SOUTH_EAST:
                    next_pos = self.agents[i].south_east_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)
                elif actions[i] == self.actions.SOUTH_WEST:
                    next_pos = self.agents[i].south_west_pos()
                    next_cell = self.grid.get(*next_pos)
                    if next_cell is None or next_cell.can_overlap():
                        self.move_agent(i, next_pos)

        # Count agents that actually moved (position changed)
        num_agents_moved = 0
        for i, agent in enumerate(self.agents):
            if tuple(agent.pos) != initial_positions[i]:
                num_agents_moved += 1

        # 2) Propagate wildfire dynamics by one time step
        trees_to_fire_state = []
        if self.log_selfish_region_metrics:
            num_trees_to_fire_state_sr = {f"{i}": 0 for i, _ in enumerate(self.selfish_xmin)}
        trees_to_burnt_state = []
        trees_to_healthy_state = []

        for c in self.unburnt_trees:
            if c.state == 0:
                pos = np.array(c.pos)
                # healthy -> on fire
                if np.random.rand() < 1 - (1 - self.alpha) ** self.neighbors_on_fire(pos):
                    trees_to_fire_state.append(c)
                    self.trees_on_fire += 1
                    if self.log_selfish_region_metrics and c.region != "common":
                        self.selfish_region_trees_on_fire[int(c.region)] += 1
                        num_trees_to_fire_state_sr[c.region] += 1

            if c.state == 1:
                # Calculate total efficiency of all agents that passed through or are on this tree
                total_efficiency = 0.0

                # 에이전트가 이동 경로에서 이 tree를 지나간 경우 또는 현재 이 tree 위에 있는 경우
                # (예: speed=1.5일 때 한 step에 2칸 이동하면서 중간 tree들과 도착점 모두에 efficiency 적용)
                if hasattr(c, 'agents_passed_through') and c.agents_passed_through:
                    total_efficiency += sum(c.agents_passed_through)
                    # Step 이후 누적값 초기화
                    c.agents_passed_through = []

                # Calculate probabilities
                # P(on fire -> healthy) = min(alpha * delta_beta, beta)
                p_healthy = min(total_efficiency * self.delta_beta, self.beta) if total_efficiency > 0 else 0.0
                # P(on fire -> burnt) = 1 - beta
                p_burnt = 1 - self.beta
                # P(on fire -> on fire) = beta - p_healthy (remaining probability)

                # Determine transition
                rand = np.random.rand()

                # on fire -> burnt (natural burning process)
                if rand < p_burnt:
                    trees_to_burnt_state.append(c)
                    self.burnt_trees += 1
                    self.trees_on_fire -= 1
                    if self.log_selfish_region_metrics and c.region != "common":
                        self.selfish_region_burnt_trees[int(c.region)] += 1
                        self.selfish_region_trees_on_fire[int(c.region)] -= 1
                # on fire -> healthy (agent extinguishes fire)
                elif rand < p_burnt + p_healthy:
                    trees_to_healthy_state.append(c)
                    self.trees_on_fire -= 1
                    if self.log_selfish_region_metrics and c.region != "common":
                        self.selfish_region_trees_on_fire[int(c.region)] -= 1
                # else: on fire -> on fire (remains burning)

        # 3) Apply updates to grid/helper_grid after loop
        for c in trees_to_fire_state:
            c.state = 1
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 1
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        for c in trees_to_healthy_state:
            c.state = 0
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 0
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        for c in trees_to_burnt_state:
            self.unburnt_trees.remove(c)
            c.state = 2
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 2
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        # 4) Episode termination check
        if self.trees_on_fire == 0:
            terminated = True
            rewards = {f"{a.index}": 0.0 for a in self.agents}
        elif self.step_count >= self.max_steps:
            truncated = True
            rewards = {f"{a.index}": 0.0 for a in self.agents}
        else:
            # 5) 보상 계산을 위한 상태 수집  ---------------------------------------------
            # (a) 크레딧 할당: 어떤 에이전트가 불을 껐는가?
            # agents_passed_through를 통해 경로상의 모든 tree들이 기록됨 (중간 + 도착점)
            extinguished_by_agent = np.zeros(self.num_agents, dtype=np.int32)

            for c in trees_to_healthy_state:
                if hasattr(c, 'agents_passed_through') and c.agents_passed_through:
                    # 경로상의 모든 efficiency 값에 대해 credit 부여
                    for a in self.agents:
                        agent_eff = getattr(a, 'efficiency', 1.0)
                        if agent_eff in c.agents_passed_through:
                            # 이 에이전트가 경로상에 있으면 1점 부여
                            extinguished_by_agent[a.index] += 1
                            # 중복 계산 방지: 해당 efficiency 제거
                            c.agents_passed_through.remove(agent_eff)

            new_fire_total = len(trees_to_fire_state)

            # (b) reward shaping 함수를 위한 추가 상태 계산
            if self.reward_function is not None:
                # 이번 스텝에 소화된 나무 수
                trees_extinguished_this_step = int(np.sum(extinguished_by_agent))
                # 누적 카운터 업데이트
                self.cumulative_extinguished += trees_extinguished_this_step
                self.cumulative_actions += self.num_agents
                # 한 번도 불이 안 붙은 나무 계산 (보존된 나무)
                trees_preserved = (self.total_trees - self.cumulative_extinguished
                                  - self.burnt_trees - self.trees_on_fire)
                # 에이전트별 소화 나무 수
                extinguished_per_agent = {i: int(extinguished_by_agent[i]) for i in range(self.num_agents)}

                # individual2 reward를 위한 추가 파라미터
                # agent_tree_extinguished: 각 에이전트가 진화한 나무가 있는지 (0 or 1)
                agent_tree_extinguished = {}
                for a in self.agents:
                    extinguished_here = 0
                    agent_eff = getattr(a, 'efficiency', 1.0)
                    # agents_passed_through에서 이 에이전트가 기여한 tree가 있는지 확인
                    for c in trees_to_healthy_state:
                        if hasattr(c, 'agents_passed_through') and agent_eff in c.agents_passed_through:
                            extinguished_here = 1
                            break
                    agent_tree_extinguished[a.index] = extinguished_here

                # agent_on_fire_tree: 각 에이전트가 불타는 나무 위에 있는지 (0 or 1)
                agent_on_fire_tree = {}
                for a in self.agents:
                    tree_at_pos = self.helper_grid.get(*a.pos)
                    if tree_at_pos and tree_at_pos.type == "tree" and tree_at_pos.state == 1:
                        agent_on_fire_tree[a.index] = 1
                    else:
                        agent_on_fire_tree[a.index] = 0

            # 6) 보상 계산  ---------------------------------------------
            if self.reward_function is not None:
                # reward shaping 함수 사용
                rewards = self._compute_shaped_rewards(
                    trees_to_fire_state=trees_to_fire_state,
                    trees_to_burnt_state=trees_to_burnt_state,
                    trees_extinguished_this_step=trees_extinguished_this_step,
                    new_fire_total=new_fire_total,
                    trees_preserved=trees_preserved,
                    extinguished_per_agent=extinguished_per_agent,
                    agent_tree_extinguished=agent_tree_extinguished,
                    agent_on_fire_tree=agent_on_fire_tree,
                    num_agents_moved=num_agents_moved,
                )
            else:
                # 기존 base rewards 사용
                # 즉시성 신호를 강화: 끈 불(=on fire -> healthy)에는 +, 새로 번 불에는 -
                # 하이퍼파라미터(필요시 조정)
                r_extinguish = 1.0           # 내가 서 있던 불이 꺼지면 +1
                r_new_fire = 0.5             # 새로 번 불 1그루당 -0.5(기본)
                selfish_w = float(self.selfishness_weight)  # 타구역 가중
                agent_rewards = np.zeros(self.num_agents, dtype=np.float32)

                if self.cooperative_reward:
                    # 협력: 모두 같은 보상(공동체 관점)
                    coop = r_extinguish * int(np.sum(extinguished_by_agent)) - r_new_fire * new_fire_total
                    agent_rewards[:] = coop
                else:
                    # 비협력: 내 이기구역에서 번 불은 강한 페널티, 타구역은 약화(selfish_w)
                    if self.log_selfish_region_metrics:
                        for a in self.agents:
                            sr = int(num_trees_to_fire_state_sr.get(f"{a.index}", 0))
                            other = new_fire_total - sr
                            agent_rewards[a.index] = (
                                + r_extinguish * extinguished_by_agent[a.index]
                                - r_new_fire * (sr + selfish_w * other)
                            )
                    else:
                        # 자기/타 구역 구분을 쓰지 않을 때: 전역 페널티만
                        for a in self.agents:
                            agent_rewards[a.index] = (
                                + r_extinguish * extinguished_by_agent[a.index]
                                - r_new_fire * new_fire_total
                            )

                rewards = {f"{a.index}": float(agent_rewards[a.index]) for a in self.agents}
                # ----------------------------------------------------------------

        # 7) Observations
        agent_obs = self._get_obs()
        next_obs = OrderedDict({f"{a.index}": agent_obs[a.index] for a in self.agents})

        # 8) Infos (디버깅용)
        infos = {}
        # new_fire_total이 위에서만 정의되므로, 종료/최대스텝에서의 안전 처리
        _new_fire_total = len(trees_to_fire_state)
        for a in self.agents:
            info = {
                "burnt_trees_total": self.burnt_trees,
                "trees_on_fire": self.trees_on_fire,
                "new_fire_total": _new_fire_total,
            }
            # 비종료 분기에서만 존재하는 로컬 변수들 안전 처리
            if 'extinguished_by_agent' in locals():
                info["extinguished_by_me"] = int(extinguished_by_agent[a.index])
            if self.log_selfish_region_metrics and 'num_trees_to_fire_state_sr' in locals():
                info["new_fire_in_my_sr"] = int(num_trees_to_fire_state_sr.get(f"{a.index}", 0))
            infos[f"{a.index}"] = info

        return next_obs, rewards, terminated, truncated, infos

    def _compute_shaped_rewards(
        self,
        trees_to_fire_state,
        trees_to_burnt_state,
        trees_extinguished_this_step,
        new_fire_total,
        trees_preserved,
        extinguished_per_agent,
        agent_tree_extinguished,
        agent_on_fire_tree,
        num_agents_moved,
    ):
        """
        reward shaping 함수를 사용하여 보상 계산

        Parameters
        ----------
        trees_to_fire_state : list
            이번 스텝에 새로 불이 붙은 나무 리스트
        trees_to_burnt_state : list
            이번 스텝에 소실된 나무 리스트
        trees_extinguished_this_step : int
            이번 스텝에 소화된 나무 수
        new_fire_total : int
            이번 스텝에 새로 불이 붙은 나무 수
        trees_preserved : int
            에피소드 동안 한 번도 불이 안 붙은 나무 수
        extinguished_per_agent : dict
            각 에이전트가 소화한 나무 수 {agent_id: count}
        agent_tree_extinguished : dict
            각 에이전트가 있는 나무가 진화되었는지 {agent_id: 0 or 1}
        agent_on_fire_tree : dict
            각 에이전트가 불타는 나무 위에 있는지 {agent_id: 0 or 1}
        num_agents_moved : int
            이번 스텝에 실제로 이동한 에이전트 수 (위치가 변경된 에이전트)

        Returns
        -------
        rewards : dict
            각 에이전트의 보상 {agent_id_str: reward}
        """
        if self.reward_shaping == "cooperative":
            # cooperative reward: 모든 에이전트가 같은 보상
            reward_dict = self.reward_function(
                trees_to_fire_state=trees_to_fire_state,
                trees_extinguished_by_agents=trees_extinguished_this_step,
                num_agents=self.num_agents,
                **self.reward_config
            )
            # 반환된 dict의 키를 문자열로 변환
            return {f"{k}": v for k, v in reward_dict.items()}

        elif self.reward_shaping == "ramadan":
            # ramadan reward: 누적 보상
            reward_dict = self.reward_function(
                trees_preserved=trees_preserved,
                cumulative_extinguished=self.cumulative_extinguished,
                cumulative_burned=self.burnt_trees,
                cumulative_actions=self.cumulative_actions,
                num_agents=self.num_agents,
                **self.reward_config
            )
            # 반환된 dict의 키를 문자열로 변환
            return {f"{k}": v for k, v in reward_dict.items()}

        elif self.reward_shaping == "individual":
            # individual reward: 각 에이전트의 기여도 기반 보상
            reward_dict = self.reward_function(
                trees_to_fire_state=trees_to_fire_state,
                trees_to_burnt_state=trees_to_burnt_state,
                agent_tree_extinguished=agent_tree_extinguished,
                num_agents=self.num_agents,
                **self.reward_config
            )
            # 반환된 dict의 키를 문자열로 변환
            return {f"{k}": v for k, v in reward_dict.items()}

        elif self.reward_shaping == "individual2":
            # individual2 reward: 차등 보상 (공동 보상 + 개별 보상)
            reward_dict = self.reward_function(
                trees_extinguished=trees_extinguished_this_step,
                trees_burnt=len(trees_to_burnt_state),
                trees_to_fire=len(trees_to_fire_state),
                agent_tree_extinguished=agent_tree_extinguished,
                agent_on_fire_tree=agent_on_fire_tree,
                num_agents=self.num_agents,
                num_agents_moved=num_agents_moved,
                **self.reward_config
            )
            # 반환된 dict의 키를 문자열로 변환
            return {f"{k}": v for k, v in reward_dict.items()}

        else:
            # 알 수 없는 reward shaping type (기본 보상으로 fallback)
            return {f"{a.index}": 0.0 for a in self.agents}

    def render(self, mode=None, close=False, highlight=False, tile_size=TILE_PIXELS):
        """Render the whole-grid human view

        Parameters
        ----------
        mode : str, optional
            the mode to render with, by default None. To avoid requiring mode to be passed everytime render is called, the variable render_mode set during env initialization is used. This argument does not affect the rendering of the environment.
        close : bool, optional
            close the rendering window, by default False. Only applicable if render_mode is "human"
        highlight : bool, optional
            highlight the cells visible to the agent, by default False
        tile_size : int, optional
            size of each tile in pixels, by default TILE_PIXELS (defined in constants.py)

        Returns
        -------
        img : ndarray
            image of the grid
        """
        if close:
            if self.window:
                self.window.close()
            return

        if self.render_mode == "human" and not self.window:
            self.window = Window("gym_multigrid")
            self.window.show(block=False)

        if highlight:
            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {
                (i, j): [] for i in range(self.width) for j in range(self.height)
            }

            for i, a in enumerate(self.agents):
                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = (
                    a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)
                )

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the grid
        if self.render_selfish_region_boundaries:
            # include selfish region boundaries in the render
            colors = [COLORS[color] for color in self.agent_colors]
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncahed_object_types,
                x_min=self.selfish_xmin,
                y_min=self.selfish_ymin,
                x_max=self.selfish_xmax,
                y_max=self.selfish_ymax,
                colors=colors,
            )
        else:
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncahed_object_types,
            )

        # Re-render the tiles containing agents to include trees below agent
        if self.render_selfish_region_boundaries:
            # include selfish region boundaries in the render
            for a in self.agents:
                img = render_agent_tiles(
                    img,
                    a,
                    self.helper_grid,
                    self.world,
                    x_min=self.selfish_xmin,
                    y_min=self.selfish_ymin,
                    x_max=self.selfish_xmax,
                    y_max=self.selfish_ymax,
                    colors=colors,
                )
        else:
            for a in self.agents:
                img = render_agent_tiles(img, a, self.helper_grid, self.world)

        if self.render_mode == "human":
            self.window.show_img(img)

        return img
