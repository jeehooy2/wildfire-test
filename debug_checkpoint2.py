import pickle
from pathlib import Path

checkpoint_file = Path('train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_16fa4_00000_0_2025-11-16_05-07-03/checkpoint_000002/checkpoint-2')

print(f'Checkpoint file: {checkpoint_file.name}')

with open(checkpoint_file, 'rb') as f:
    checkpoint_data = pickle.load(f)

if 'worker' in checkpoint_data:
    worker_bytes = checkpoint_data['worker']
    worker_data = pickle.loads(worker_bytes)

    if 'state' in worker_data:
        if 'shared_policy' in worker_data['state']:
            policy_state = worker_data['state']['shared_policy']
            print(f'Policy state keys: {list(policy_state.keys())}')
            print(f'Policy state type: {type(policy_state)}')

            # 각 키의 타입 확인
            for key in policy_state.keys():
                value = policy_state[key]
                print(f'  {key}: type={type(value).__name__}')

                # weights 키 상세 확인
                if key == 'weights':
                    if isinstance(value, dict):
                        print(f'    weights is dict with {len(value)} items')
                        print(f'    First 3 keys: {list(value.keys())[:3]}')
                        # 첫 번째 value의 타입 확인
                        if value:
                            first_key = list(value.keys())[0]
                            first_val = value[first_key]
                            print(f'    First value type: {type(first_val).__name__}')
