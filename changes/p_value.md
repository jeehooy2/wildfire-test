실험 비교를 위한 p값을 계산해주는 다음 코드를 train_marllib_self/p_value.py 생성해줘:

입력: evaluation/ppo/run22/evaluation_report.txt, evaluation/heuristic/evaluation_report_single_fire.txt

위 두 파일에(heuristic의 경우 Nearest Fire Greedy만) 나와있는 Per-Episode Details(총 50개의 에피소드)에서 Healthy 그리고 Length 각각에 대해 서로 유의한 차이가 있는지 P값을 계산해줘.

출력: Healthy와 Length의 p값