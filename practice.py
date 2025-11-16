import pickle
from pathlib import Path

checkpoint_dir = Path('train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_16fa4_00000_0_2025-11-16_05-07-03')
checkpoint_files = list(checkpoint_dir.glob('checkpoint-*'))
checkpoint_files = [f for f in checkpoint_files if f.is_file() and not f.name.endswith('.tune_metadata')]
checkpoint_files.sort(key=lambda x: int(x.name.split('-')[1]))
checkpoint_file = checkpoint_files[-1]

print(f'Checkpoint file: {checkpoint_file.name}')

with open(checkpoint_file, 'rb') as f:
    checkpoint_data = pickle.load(f)

print(f'Top-level keys: {list(checkpoint_data.keys())}')

if 'worker' in checkpoint_data:
    worker_bytes = checkpoint_data['worker']
    worker_data = pickle.loads(worker_bytes)
    print(f'Worker data keys: {list(worker_data.keys())}')
    
    if 'state' in worker_data:
        print(f'State keys: {list(worker_data["state"].keys())}')
        
        if 'shared_policy' in worker_data['state']:
            policy_state = worker_data['state']['shared_policy']
            print(f'Policy state keys: {list(policy_state.keys())}')