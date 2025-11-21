학습한 MAPPO, PPO 등의 산불 진화 모델과 함께 비교할 목적으로 train_marllib_self/heuristic_nearest_fire.py를 만들어줘.

train_marllib_self/environment.py를 확인하고 해당 heuristic은 다음과 같은 내용을 수행해:
1. train_marllib_self/environment.py를 이용해서 시뮬레이션 환경을 만들고, 만약 partial_obs=False면 전체 그리드에서 각 에이전트와 가장 가까운 on fire tree로 이동하게 해줘.
2. 만약 partial_obs=True면 에이전트의 관찰 공간에서 각 에이전트와 가장 가까운 on fire tree로 이동하게 해줘.
3. episode 수, seed를 사용자가 결정할 수 있게 해주고, 결과를 results/heuristic 안에 저장해줘.
4. 나중에 new_evaluation.py에서 불러서 비교군으로 사용할 수 있게 행동을 선택하는 부분을 함수로 만들어줘. (nearest fire greedy)

(혹시 wildfire_environment와 environment.py 사용 방법을 알고 싶다면 new_compare_visual_mappo.py를 참고해)