MAPPO, PPO 등의 MARLlib 알고리즘으로 학습한 산불 진화 모델과 비교할 대조군의 평가값을 계산하고 싶어.

train_marllib/new_evalution.py는 MARLlib 알고리즘으로 학습한 산불 진화 모델의 평가 스크립트야. 
이와 동일한 평가 항목을 다음 2가지 heuristic에 대해 계산해주는 평가 스크립트를 train_marllib/new_evaluation_heuristic.py에 작성해줘:

1. heuristic_nearest_fire.py에 있는 nearest fire greedy 로직
2. 각 에이전트가 random 행동 (아마 그냥 step() 등을 쓰면 됐던 것 같아)