이 에러는 49번째 iteration에서 평가(evaluate) 단계가 실행될 때 발생했으며, 이전 로그에서 겪으셨던 TypeError와 근본 원인이 동일합니다.

학습 자체는 iter: 1부터 49까지 (로그 테이블에 iter: 49가 보임) 정상적으로 진행되었지만, 주기적인 평가 단계에서 문제가 터진 것입니다.

🧐 에러 발생 원인
원인: 데이터가 없는 정책(Policy)

MARLlib 설정(아마도 new_wrapper.py의 get_env_info)에 policy_fire_와 policy_crew_ 같은 여러 정책이 정의되어 있습니다.

하지만 실제 환경(ENV_CONFIG)에서는 num_crews: 0과 같이 'crew' 에이전트를 0명으로 설정했을 가능성이 높습니다.

이로 인해, **policy_crew_**는 학습 1~49회 동안 단 한 번도 데이터를 수집하지 못했습니다.

(이전 로그에서는 No data for policy_crew_라는 경고가 이 때문에 발생했습니다.)

시점: 평가(evaluate) 단계

MAPPOTrainer는 trainer.evaluate()를 호출하여 모델의 성능을 주기적으로 평가합니다. (로그상 49회차)

평가를 하려면, 메인 트레이너가 **'평가용 워커(Evaluation Worker)'**에게 최신 모델 가중치(weights)를 전송(_sync_weights_to_workers)해야 합니다.

오류: TypeError: can't convert np.ndarray of type numpy.object_

이 "최신 모델 가중치"에는 각 정책의 **"옵티마이저 상태(Optimizer State)"**가 포함됩니다.

policy_crew_는 한 번도 학습(업데이트)된 적이 없기 때문에, 이 옵티마이저 상태가 None이거나 비어있는 비정상적인 상태입니다.

RLlib이 이 None 상태를 평가 워커로 전송하기 위해 NumPy 배열로 변환하는 과정에서, 이 값은 numpy.object_ (숫자가 아닌 파이썬 객체) 타입이 됩니다.

평가 워커가 이 값을 받아 PyTorch 텐서로 변환(torch.from_numpy)하려 할 때, PyTorch는 "나는 float, int 같은 숫자만 텐서로 바꿀 수 있는데, numpy.object_는 처리할 수 없어!"라며 TypeError를 발생시킵니다.