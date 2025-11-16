train_marllib_self/visualize_trained_agent.py에서 MARLlib로 학습시킨 모델을 읽고 시각화하는 방법을 참고해서 다음 역할을 하는 코드를 train_marllib_self/new_compare_visual4.py 파일에 새로 작성해줘:

현재 compare_visual_new.py에서는 RLlib로 학습시킨 모델을 대상으로 지정한 파일 경로에 있는 모델과 seed를 입력받아 이에 대해 랜덤한 행동과의 비교를 시각화해서 .gif 파일로 저장하는데 이와 동일한 역할을 수행해줬으면 좋겠어. 결과는 train_marllib_self/results/mappo/run_name 형태로 저장되었으면 좋겠어.