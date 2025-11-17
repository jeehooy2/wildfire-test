1. 현재 new_compare_visual.py의 create_policy_network()에서 사용하는 nn 구조가 옳은지 평가해주고, 결과를 나타내기 위한 이런 nn 구조를 어떻게 설계하는지 알려줘.

2. line 405부터 잘 작동하는지 어떻게 확인해? 현재 150 iteration까지 학습했는데 (experiments/mappo/run13_fixed) run new_compare_visual.py의 결과가 랜덤 정책과 비슷해서 혹시 line 413과 같이 정책을 찾을 수 없어서 랜덤 액션을 한 게 아닌지 확인하고 싶어.

3. 전반적으로 학습 결과가 너무 안 좋은데 현재 학습 과정(new_train_mappo.py, new_wrapper.py)에서 어떤 문제가 있거나, 더 개선되어야 하는 부분이 없는지 궁금해.