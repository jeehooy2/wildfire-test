질문: 현재 wildfire_environment/envs/wildfire.py에서 partial_obs=True로 설정하면, 각 에이전트의 observation space가 어떻게 달라져? 

목표: 
1. 각 에이전트의 observation space를 에이전트가 중앙에 있는 5x5 cell space로 설정하고 싶어. (필요하면 에이전트의 방향 등도 저장해야 하는지 확인해줘)
2. 대신 partial_obs=False면 현재 상태가 유지됐으면 좋겠어.
3. 추후 train_marllib_self/new_train_mappo.py를 실행했을 때 train_marllib_self/environment.py에서 partial_obs=True면 이렇게 축소된 observation space로 학습이 진행되었으면 좋겠어. 대신 centralized critic은 이 경우 어떤 정보를 전달받는지 알려줘.