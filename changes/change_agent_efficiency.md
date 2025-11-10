에이전트 efficiency에 따라 Crew처럼 efficiency가 alpha이면 다음과 같이 산불 확산이 이루어지도록 wildfire_environment/envs/wildfire.py를 수정해줘: 

불타는 나무 → 건강한 나무가 될 확률: P(on fire → healthy) = min(alpha * δ_β * (agent_above), β)

불타는 나무 → 불타는 나무가 될 확률: P(on fire → on fire) = β - min(alpha * δ_β * (agent_above), β)

불타는 나무 → 타버린 나무가 될 확률: P(on fire → burnt) = 1 - β

(δ_β: Agent의 진화 효과 (기본값: 0.54))

이때, 한 cell 위에 여러 에이전트가 있을 경우에는 각각의 efficiency의 합을 alpha로 사용해줘.