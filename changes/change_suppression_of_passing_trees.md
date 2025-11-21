현재 wildfire_environment/envs/wildfire.py를 다음과 같이 수정해줘:

현재 구조 및 코드를 최대한 변경하지 않는 선에서, 에이전트가 1 스텝에 2칸 이상 이동할 경우에 시작점과 도착점 사이의 tree (시작점과 도착점을 미포함한, 에이전트가 지나가는 tree)에 대해서도 도착점과 마찬가지로 진화 효과를 볼 수 있도록 별도의 리스트로 관리해서 이중 on fire인 tree의 경우, 도착점과 마찬가지로 에이전트의 진화 확률 P(on fire -> healthy)를 적용해줘.

그리고 이렇게 진화된 나무가 늘어났으니, 이를 line 1238 이후 보상을 위한 상태값 수집에도 적용해줘. (예: 2칸 이동해서 도착 tree와 사이 tree 모두 진화되었다면 해당 에이전트에 대해 extinguished_per_agent=2)

주의: 오로지 wildfire_environment/envs/wildfire.py 파일만 수정하고 다른 파일은 일체 수정하지 마!