reward_functions.py에 아래와 같은 individual2_reward 함수를 만들어줘. 적절한 파라미터를 wildfire.py에서 받고, 추후 environment.py에서 reward_shaping 지정을 통해 wildfire.py에서 이 함수를 이용해서 모델을 학습할 수 있도록 수정해줘.

에이전트별로 차등 reward 적용

보상 함수:
reward = alpha * (1.0 * T_e - 0.5 * T_b) + (1 - alpha) * (1.0 * T_e_individual + 1.0 * above_tree_on_fire)

공동 보상 파트:
alpha = 공동 보상 계수 (기본값: 0.3)
T_e = 이번 스텝에 성공적으로 진화된 나무 수 (on_fire -> healthy)
T_o = 이번 스텝에 새롭게 불타는 나무 수 (healthy -> on_fire) (일단 reward에 사용하지 않더라도 만들어줘)
T_b = 이번 스텝에 소실된 나무 수 (on_fire -> burnt)

개별 보상 파트:
T_e_individual = 내가(현재 에이전트) 있는 나무가 이번에 진화되었는지
above_tree_on_fire = 내가(현재 에이전트) 있는 나무가 불 타는 상태 (on_fire)