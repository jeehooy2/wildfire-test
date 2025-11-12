현재 wildfire.py에서는 Crew와 같이 속도가 느린 에이전트는 매 step에 action을 sample()하지만 실제 이동은 안 하기 때문에 학습이 어려울 것이라 판단했어. 따라서 프레임 스킵 기반 방식을 도입하려고 해. 다음 예시 코드에서 로직만 참고해서 실제 구현은 wildfire.py에 적합하게 수정해서 바꿔줘.

시간 스케일을 이용한 구현 방안 (프레임 스킵 기반)이 방식은 환경 스텝을 조정하여 모든 에이전트의 속도를 정수로 만듭니다. 1. 환경 기본 단위 정의 (Base Time Step)가장 느린 Crew의 속도를 기준으로 새로운 환경 스텝을 정의합니다.

Helicopter: 속도 15.0
Truck: 속도 5.0
Crew: 속도 1.0

이때, 1개의 Base Time Step을 Crew가 1 cell 이동하는 데 필요한 시간으로 정의합니다. 하지만, 사용자님의 요청에 따라 Helicopter가 한 step에 1 cell 이동하는 것을 기준으로 설정해야 합니다.

에이전트: Helicopter : Truck : Crew
속도 비율: 15 : 5 : 1
1 Cell 이동에 필요한 환경 스텝 (k): 15/15 : 15/5 : 15/1

정책 학습 측면에서 가장 효과적인 구현은 느린 에이전트에게 k 스텝 동안 정책 업데이트를 하지 않도록 강제하는 정책 수준 프레임 스킵(Policy-Level Frame Skip) 방식입니다.

A. 에이전트 속성 추가각 에이전트 클래스(Helicopter, Truck, Crew)에 execution_frequency 속성을 정의합니다.
class Helicopter(Agent):
    # ...
    self.execution_frequency = 1  # 매 환경 스텝마다 정책 선택

class Truck(Agent):
    # ...
    self.execution_frequency = 3  # 3 환경 스텝마다 정책 선택

class Crew(Agent):
    # ...
    self.execution_frequency = 15 # 15 환경 스텝마다 정책 선택
    self.last_action = WildfireActions.STILL # 직전 액션 저장용

B. WildfireEnv.step 함수 수정step 함수 내에서 self.step_count와 execution_frequency를 이용하여 정책 실행을 제어합니다.
    def step(self, actions):
        self.step_count += 1
        actions_list = list(actions.values())
        
        # 1) 에이전트별 액션 결정 (프레임 스킵 적용)
        effective_actions = {}
        for i, agent in enumerate(self.agents):
            k = agent.execution_frequency
            
            if self.step_count % k == 0:
                # 1. 정책 업데이트 주기: 새로운 액션 선택 및 저장
                agent.last_action = actions_list[i]
            # else: k 스텝이 아닐 경우, 이전 액션(agent.last_action)을 유지 (프레임 스킵)
            
            effective_actions[i] = agent.last_action

        # 2) 결정된 액션(effective_actions)을 사용하여 이동 실행
        order = np.random.permutation(len(self.agents))
        for i in order:
            agent = self.agents[i]
            chosen_action = effective_actions[i] # 프레임 스킵이 적용된 액션

            # 만약 Crew(k=15)의 경우, 14스텝 동안은 last_action이 반복됨
            if chosen_action == self.actions.STILL:
                continue

            # (이동 로직): chosen_action을 기반으로 1 cell 이동 로직 수행
            # Helicopter, Truck, Crew 모두 1회 Move 액션 시 1 cell 이동
            # ... (기존 이동 로직: next_pos 계산 및 self.move_agent(i, next_pos) 호출) ...
            
        # 3) Wildfire Dynamics Propagate
        # 환경 동역학은 매 self.step_count마다 1회 발생
        # ...

        # 4) Observation & Reward Calculation
        # ... (생략) ...
        return next_obs, rewards, terminated, truncated, infos
