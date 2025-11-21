train_marllib_self/new_evaluation.py 파일에 다음 내용으로 학습된 모델의 평가 스크립트를 작성해줘. 
(만약 checkpoint에 저장된 weights의 사용 방법을 알고 싶다면 train_marllib_self/new_compare_visual_mappo.py를 참고해도 좋아)

평가 스크립트를 실행하기 위한 명령어 예시:
python train_marllib_self/new_evaluation.py \
    --experiment train_marllib_self/experiments/mappo/run13_fixed
    --checkpoint 100
    --episodes 10
    --seed 42

평가 스크립트 내용:
총 10 (기본값) 에피소드에 대해 다음 평가 지표들을 계산해.
1. 각 에피소드 종료 시점에서 평균 healthy (green) tree의 비율 = 1/E * sum_over_all_episodes(count(healthy) / count(all_trees))
    (여기서 E: number of episodes)
2. 최종 피해 면적: 각 에피소드 종료 시점에서 평균 burnt (brown) trees의 비율 1/E * sum_over_all_episodes(count(burnt) / count(all_trees))
3. 총 진화 시간 (steps): 평균 에피소드 길이 (에피소드 종료까지 걸린 step 수)


저장 폴더: train_marllib_self/evaluation/ 내 알고리즘/run_name의 폴더
(만약 사용하는 모델이 train_marllib_self/experiments/mappo/run13_fixed라면 train_marllib_self/evaluation/mappo/run13_fixed에 저장)