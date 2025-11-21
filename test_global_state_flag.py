"""
Test script to validate global_state_flag=True support in new_wrapper_global.py
"""

import sys
sys.path.insert(0, '/home/bmkim88/wildfire_environment')

import numpy as np
from train_marllib_self.new_wrapper_global import WildfireRLlibEnv

def test_global_state_flag():
    """Test that global_state_flag=True works correctly"""

    print("\n" + "="*70)
    print("Testing global_state_flag=True Support")
    print("="*70)

    # Create environment with test configuration
    # Note: num_helicopters + num_trucks + num_crews must equal num_agents
    env_config = {
        'num_agents': 3,
        'num_helicopters': 2,
        'num_trucks': 1,
        'size': 17,
        'max_steps': 100,
        'agent_view_size': 10,
        'partial_obs': False,
        'reward_shaping': None,
        'reward_shaping_config': None,
        'agent_start_positions': ((1, 1), (15, 15), (8, 8)),
    }

    try:
        print("\n1️⃣  Creating WildfireRLlibEnv with global_state support...")
        env = WildfireRLlibEnv(env_config)
        print("✅ Environment created successfully!\n")

        # Test 1: Check observation space structure
        print("2️⃣  Checking observation space structure...")
        obs_space = env.observation_space
        print(f"   Observation space type: {type(obs_space)}")
        print(f"   Observation space keys: {obs_space.spaces.keys()}")

        assert "obs" in obs_space.spaces, "Missing 'obs' in observation space"
        assert "state" in obs_space.spaces, "Missing 'state' in observation space"
        print("✅ Observation space has correct keys ('obs' and 'state')\n")

        # Test 2: Check global state size
        print("3️⃣  Checking global state size...")
        obs_depth = env.env.obs_depth
        grid_size = env.env.grid_size
        expected_state_size = (obs_depth + 1) * grid_size * grid_size + 1
        actual_state_size = obs_space.spaces["state"].shape[0]

        print(f"   obs_depth: {obs_depth}")
        print(f"   grid_size: {grid_size}")
        print(f"   Expected state size: {expected_state_size}")
        print(f"   Actual state size: {actual_state_size}")

        assert expected_state_size == actual_state_size, \
            f"State size mismatch: expected {expected_state_size}, got {actual_state_size}"
        print("✅ Global state size is correct\n")

        # Test 3: Reset and check observation format
        print("4️⃣  Testing reset() method...")
        obs_dict = env.reset()
        print(f"   Returned observation keys: {list(obs_dict.keys())}")
        print(f"   Number of agents: {len(obs_dict)}")

        # Check that each agent has correct observation structure
        for agent_id, obs in obs_dict.items():
            assert isinstance(obs, dict), f"Agent {agent_id} obs should be dict"
            assert "obs" in obs, f"Agent {agent_id} missing 'obs' key"
            assert "state" in obs, f"Agent {agent_id} missing 'state' key"

            obs_array = obs["obs"]
            state_array = obs["state"]

            print(f"\n   Agent {agent_id}:")
            print(f"     - local obs shape: {obs_array.shape}, dtype: {obs_array.dtype}")
            print(f"     - global state shape: {state_array.shape}, dtype: {state_array.dtype}")

            assert isinstance(obs_array, np.ndarray), f"Agent {agent_id} obs should be numpy array"
            assert isinstance(state_array, np.ndarray), f"Agent {agent_id} state should be numpy array"
            assert obs_array.dtype == np.float32, f"Agent {agent_id} obs dtype should be float32"
            assert state_array.dtype == np.float32, f"Agent {agent_id} state dtype should be float32"

        print("\n✅ Reset observation format is correct\n")

        # Test 4: Check that all agents have SAME global state
        print("5️⃣  Checking that all agents share the SAME global state...")
        agent_ids = list(obs_dict.keys())
        global_state_0 = obs_dict[agent_ids[0]]["state"]

        all_same = True
        for agent_id in agent_ids[1:]:
            if not np.array_equal(global_state_0, obs_dict[agent_id]["state"]):
                all_same = False
                print(f"   ❌ Agent {agent_id} has different state!")
                break

        if all_same:
            print(f"   All {len(agent_ids)} agents have identical global state")
            print("✅ Global state is correctly shared across all agents\n")
        else:
            print("❌ Global state is NOT shared correctly!")
            return False

        # Test 5: Step and check observation format
        print("6️⃣  Testing step() method...")
        action_dict = env.action_space_sample(agent_ids)
        obs_dict, rewards, dones, infos = env.step(action_dict)

        print(f"   Returned observation keys: {list(obs_dict.keys())}")

        # Check that each agent has correct observation structure after step
        for agent_id, obs in obs_dict.items():
            assert isinstance(obs, dict), f"Agent {agent_id} obs should be dict after step"
            assert "obs" in obs, f"Agent {agent_id} missing 'obs' key after step"
            assert "state" in obs, f"Agent {agent_id} missing 'state' key after step"

        # Check that all agents still have SAME global state after step
        agent_ids = list(obs_dict.keys())
        global_state_0 = obs_dict[agent_ids[0]]["state"]

        all_same = True
        for agent_id in agent_ids[1:]:
            if not np.array_equal(global_state_0, obs_dict[agent_id]["state"]):
                all_same = False
                break

        assert all_same, "Global state not shared correctly after step"
        print("✅ Step observation format and shared state are correct\n")

        # Test 6: Check get_env_info
        print("7️⃣  Checking get_env_info()...")
        env_info = env.get_env_info()
        print(f"   Environment info keys: {list(env_info.keys())}")
        print(f"   policy_mapping_info: {env_info['policy_mapping_info']}")

        assert "space_obs" in env_info, "Missing 'space_obs' in env_info"
        assert "space_act" in env_info, "Missing 'space_act' in env_info"
        assert "num_agents" in env_info, "Missing 'num_agents' in env_info"
        assert "episode_limit" in env_info, "Missing 'episode_limit' in env_info"
        assert "policy_mapping_info" in env_info, "Missing 'policy_mapping_info' in env_info"

        print("✅ get_env_info() returns all required fields\n")

        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print(f"  - Observation space has 'obs' and 'state' keys ✓")
        print(f"  - Global state size calculated correctly ✓")
        print(f"  - Reset returns correct observation format ✓")
        print(f"  - All agents share identical global state ✓")
        print(f"  - Step returns correct observation format ✓")
        print(f"  - get_env_info provides all required fields ✓")

        env.close()
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_global_state_flag()
    sys.exit(0 if success else 1)
