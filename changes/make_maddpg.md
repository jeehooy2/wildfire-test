다중 에이전트의 산불 진화 문제에 대해 여러 MARL 알고리즘 간 성능 비교가 목표야. (구체적으로 MAPPO, MADDPG, MADQN)

현재 train_marllib_self/new_train_mappo.py에 MARLlib를 이용해서 MAPPO 알고리즘을 학습시키는 코드를 완성했어. 이와 더불어서 MARLlib를 참고해서 MARLlib에 제공되는 maddpg 알고리즘을 학습하는 코드를 train_marllib_self/new_train_maddpg.py로 저장해줘. 
(기존 maddpg 코드가 있는지 확인하지 말고, 오로지 new_train_mappo.py랑 MARLlib만 확인해)

MADDPG는 continuous action space 전용이기 때문에 다음에 신경써줘. (혹시 MARLlib가 자동으로 discrete action space에 대해 MADDPG를 이용하면 continuous로 변환해주는지 확인도 부탁해)
Discrete Actions: Ensure the configuration handles the Gumbel-Softmax reparameterization trick automatically. If there are specific hyperparameters in MARLlib's algo_args (like use_gumbel, discretize_actions, or actions_are_logits) to force this behavior, please include them.