현재 wildfire_environment/wildfire_environment와 wildfire_environment/train_marllib_self/new_train_mappo.py와 environment.py를 전부 분석해서 다음 두 가지 목표를 최적화할 수 있는 산불 진화 에이전트의 보상함수를 wildfire_environment/envs/reward_functions.py에 cooperative3_reward 이름으로 만들어줘.

1. 피해 면적 최소화 (에피소드 종료 시점에서 healthy tree의 비율 최대화)
2. 진화 시간 최소화 (에피소드 길이 최소화)

최대한 단순한 논리로 설명력이 좋은 보상함수를 만들어줬으면 좋겠어. 그리고 지금 양의 보상이 sparse 한 문제에 대응하기 위해 신경써줘 (음의 보상에 너무 치우치지 않도록, healthy tree ratio 등 다른 양의 보상으로 줄 수 있는 것도 고려해봐줘)

학습 효과가 있도록 진화 행위(extinguished_tree)에 대한 즉각적인 양의 보상도 포함해줘.