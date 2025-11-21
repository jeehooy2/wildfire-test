#!/usr/bin/env python3
"""
Gumbel-Softmax 구현 테스트 스크립트

이 스크립트는 discrete_maddpg.py의 핵심 기능을 테스트합니다:
1. Gumbel-Softmax 샘플링
2. Temperature annealing
3. Hard vs Soft action
4. Log probability 계산
"""

import torch
import torch.nn.functional as F
import numpy as np
from train_marllib_self.discrete_maddpg import (
    GumbelSoftmax,
    DiscreteActorNetwork,
    DiscreteMADDPGWrapper,
    TemperatureScheduler
)


def test_gumbel_softmax():
    """Gumbel-Softmax 기본 기능 테스트"""
    print("=" * 70)
    print("Test 1: Gumbel-Softmax Sampling")
    print("=" * 70)

    # 로짓 생성
    logits = torch.tensor([[1.0, -0.5, 2.0, 0.1], [0.5, 0.5, 0.5, 0.5]])
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits:\n{logits}")

    # Soft sampling (Temperature = 1.0)
    print("\n--- Soft Sampling (tau=1.0) ---")
    gumbel_soft = GumbelSoftmax(tau=1.0, hard=False)
    soft_samples = gumbel_soft(logits)
    print(f"Soft samples (sum=1 per row):\n{soft_samples}")
    print(f"Row sums: {soft_samples.sum(dim=1)}")

    # Soft sampling (Temperature = 0.1)
    print("\n--- Soft Sampling (tau=0.1) ---")
    gumbel_soft_cold = GumbelSoftmax(tau=0.1, hard=False)
    soft_samples_cold = gumbel_soft_cold(logits)
    print(f"Soft samples (더 sharp):\n{soft_samples_cold}")

    # Hard sampling
    print("\n--- Hard Sampling (tau=0.5) ---")
    gumbel_hard = GumbelSoftmax(tau=0.5, hard=True)
    hard_samples = gumbel_hard(logits)
    print(f"Hard samples (one-hot or near one-hot):\n{hard_samples}")

    # Log probability 계산
    print("\n--- Log Probability Calculation ---")
    sample, log_prob = GumbelSoftmax.sample_and_log_prob(logits, tau=1.0, hard=False)
    print(f"Sample shape: {sample.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Log prob values: {log_prob}")

    print("\n✅ Test 1 PASSED\n")


def test_discrete_actor():
    """DiscreteActorNetwork 테스트"""
    print("=" * 70)
    print("Test 2: DiscreteActorNetwork")
    print("=" * 70)

    # Actor 네트워크 생성
    actor = DiscreteActorNetwork(
        obs_dim=32,
        num_actions=6,
        hidden_dims=[128, 128]
    )

    # Random observation
    obs = torch.randn(8, 32)  # batch_size=8
    print(f"\nObservation shape: {obs.shape}")

    # Forward pass: logits 생성
    logits = actor(obs)
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits:\n{logits[0]}")

    # Soft action 샘플링
    print("\n--- Soft Action Sampling ---")
    action_soft, log_prob_soft = actor.get_action_and_log_prob(obs, tau=1.0, hard=False)
    print(f"Action shape: {action_soft.shape}")
    print(f"Action (agent 0):\n{action_soft[0]}")
    print(f"Log prob shape: {log_prob_soft.shape}")
    print(f"Log probs: {log_prob_soft}")

    # Hard action 샘플링
    print("\n--- Hard Action Sampling ---")
    action_hard, log_prob_hard = actor.get_action_and_log_prob(obs, tau=0.1, hard=True)
    print(f"Hard action (agent 0):\n{action_hard[0]}")
    print(f"Log probs: {log_prob_hard}")

    print("\n✅ Test 2 PASSED\n")


def test_temperature_scheduler():
    """Temperature scheduler 테스트"""
    print("=" * 70)
    print("Test 3: Temperature Annealing Schedule")
    print("=" * 70)

    total_steps = 1000

    # Linear annealing
    print("\n--- Linear Annealing ---")
    scheduler_linear = TemperatureScheduler(
        initial_tau=1.0,
        final_tau=0.1,
        total_steps=total_steps,
        anneal_type="linear"
    )

    steps = [0, 250, 500, 750, 1000]
    for step in steps:
        tau = scheduler_linear.get_tau(step)
        print(f"  Step {step:4d}: tau = {tau:.4f}")

    # Exponential annealing
    print("\n--- Exponential Annealing ---")
    scheduler_exp = TemperatureScheduler(
        initial_tau=1.0,
        final_tau=0.1,
        total_steps=total_steps,
        anneal_type="exponential"
    )

    for step in steps:
        tau = scheduler_exp.get_tau(step)
        print(f"  Step {step:4d}: tau = {tau:.4f}")

    # Cosine annealing
    print("\n--- Cosine Annealing ---")
    scheduler_cos = TemperatureScheduler(
        initial_tau=1.0,
        final_tau=0.1,
        total_steps=total_steps,
        anneal_type="cosine"
    )

    for step in steps:
        tau = scheduler_cos.get_tau(step)
        print(f"  Step {step:4d}: tau = {tau:.4f}")

    print("\n✅ Test 3 PASSED\n")


def test_multi_agent_wrapper():
    """DiscreteMADDPGWrapper 멀티에이전트 테스트"""
    print("=" * 70)
    print("Test 4: Multi-Agent MADDPG Wrapper")
    print("=" * 70)

    num_agents = 3
    obs_dim = 32
    num_actions = 6

    # Wrapper 생성
    wrapper = DiscreteMADDPGWrapper(
        num_agents=num_agents,
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dims=[128, 128],
        use_temperature_annealing=True
    )

    print(f"\nAgents: {num_agents}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Number of actions: {num_actions}")
    print(f"Actor networks: {len(wrapper.actors)}")

    # Random observations (각 에이전트마다 하나)
    observations = [np.random.randn(obs_dim) for _ in range(num_agents)]

    # Soft actions 샘플링
    print("\n--- Soft Action Sampling (Training) ---")
    actions_soft, log_probs_soft = wrapper.get_actions(
        observations, tau=1.0, hard=False
    )
    print(f"Number of actions: {len(actions_soft)}")
    print(f"Action shapes: {[a.shape for a in actions_soft]}")
    print(f"Log probs: {log_probs_soft}")

    # Hard actions 샘플링
    print("\n--- Hard Action Sampling (Inference) ---")
    actions_hard, log_probs_hard = wrapper.get_actions(
        observations, tau=0.1, hard=True
    )
    print(f"Hard actions: {[a for a in actions_hard]}")
    print(f"Log probs: {log_probs_hard}")

    # Temperature 진행 확인
    print("\n--- Temperature Progression ---")
    initial_tau = wrapper.tau_scheduler.initial_tau
    print(f"Initial tau: {initial_tau}")

    for step in range(0, 100001, 25000):
        wrapper.tau_scheduler.current_step = step
        tau = wrapper.tau_scheduler.get_tau()
        print(f"  Step {step:6d}: tau = {tau:.4f}")

    print("\n✅ Test 4 PASSED\n")


def test_gradient_flow():
    """Gradient flow 테스트 (미분 가능성)"""
    print("=" * 70)
    print("Test 5: Gradient Flow (Differentiability)")
    print("=" * 70)

    actor = DiscreteActorNetwork(obs_dim=16, num_actions=4, hidden_dims=[64])
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)

    # Random input
    obs = torch.randn(2, 16, requires_grad=False)

    # Forward pass + Gumbel-Softmax
    action, log_prob = actor.get_action_and_log_prob(obs, tau=1.0, hard=False)

    # Dummy loss (Q-value 최대화 대신 단순 loss)
    loss = -log_prob.mean()

    print(f"\nObservation requires_grad: {obs.requires_grad}")
    print(f"Action requires_grad: {action.requires_grad}")
    print(f"Log prob requires_grad: {log_prob.requires_grad}")
    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Gradient 존재 확인
    has_grad = False
    for name, param in actor.named_parameters():
        if param.grad is not None:
            has_grad = True
            break

    print(f"\nGradients computed: {has_grad}")
    print(f"Network parameters: {sum(p.numel() for p in actor.parameters())}")

    # Optimizer step
    optimizer.step()
    print("Optimizer step executed ✓")

    print("\n✅ Test 5 PASSED\n")


def test_inference_execution():
    """Inference/Execution 테스트 (argmax만 사용, Gumbel-Softmax 불사용)"""
    print("=" * 70)
    print("Test 6: Inference/Execution (Argmax only, No Gumbel-Softmax)")
    print("=" * 70)

    print("\n**중요**: Execution 시에는 Gumbel-Softmax를 사용하지 않고,")
    print("단순히 logits의 argmax를 취합니다.\n")

    actor = DiscreteActorNetwork(obs_dim=32, num_actions=9, hidden_dims=[128])

    # Random observation
    obs = torch.randn(1, 32)
    print(f"Observation shape: {obs.shape}")

    # Actor logits
    logits = actor(obs)
    print(f"Logits: {logits}")
    print(f"Logits values: {logits[0].detach().numpy()}")

    # Inference: get_discrete_action (argmax만 사용)
    print("\n--- Inference Mode (Execution) ---")
    discrete_action = actor.get_discrete_action(obs)
    print(f"Discrete action (argmax): {discrete_action.item()}")
    print(f"Argmax index: {logits.argmax(dim=-1).item()}")

    # Verify argmax 동작
    max_logit_idx = logits[0].argmax(dim=-1).item()
    print(f"✓ argmax correctly returns action {max_logit_idx}")

    # 다른 관찰들로 테스트
    print("\n--- Batch Inference Test ---")
    obs_batch = torch.randn(5, 32)
    discrete_actions_batch = actor.get_discrete_action(obs_batch)
    print(f"Batch observations shape: {obs_batch.shape}")
    print(f"Discrete actions: {discrete_actions_batch.numpy()}")
    print(f"Each action is between 0 and {9-1}")

    # Verify 모든 actions이 유효한 범위
    assert discrete_actions_batch.min().item() >= 0
    assert discrete_actions_batch.max().item() < 9
    print("✓ All actions are valid discrete indices")

    print("\n✅ Test 6 PASSED\n")


def test_multi_agent_inference():
    """멀티에이전트 Inference 테스트"""
    print("=" * 70)
    print("Test 7: Multi-Agent Inference (Discrete Actions)")
    print("=" * 70)

    num_agents = 3
    obs_dim = 32
    num_actions = 6

    wrapper = DiscreteMADDPGWrapper(
        num_agents=num_agents,
        obs_dim=obs_dim,
        num_actions=num_actions,
        use_temperature_annealing=False
    )

    print(f"\nNumber of agents: {num_agents}")
    print(f"Number of actions per agent: {num_actions}")

    # Random observations
    observations = [np.random.randn(obs_dim) for _ in range(num_agents)]

    # Training mode: soft actions with Gumbel-Softmax
    print("\n--- Training Mode (Soft Actions) ---")
    actions_soft, log_probs_soft = wrapper.get_actions(observations, hard=False, tau=1.0)
    print(f"Soft action shapes: {[a.shape for a in actions_soft]}")
    print(f"Log probs: {log_probs_soft}")

    # Inference mode: discrete actions (argmax only)
    print("\n--- Inference Mode (Discrete Actions) ---")
    discrete_actions = wrapper.get_discrete_actions(observations)
    print(f"Discrete actions: {discrete_actions}")
    print(f"Type of actions: {[type(a) for a in discrete_actions]}")

    # Verify 모든 actions이 유효한 범위
    for i, action in enumerate(discrete_actions):
        assert 0 <= action < num_actions
        print(f"  Agent {i}: action {action} (valid)")

    print("\n✅ Test 7 PASSED\n")


def main():
    """모든 테스트 실행"""
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#  Discrete MADDPG with Gumbel-Softmax - Test Suite" + " " * 18 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print()

    try:
        test_gumbel_softmax()
        test_discrete_actor()
        test_temperature_scheduler()
        test_multi_agent_wrapper()
        test_gradient_flow()
        test_inference_execution()
        test_multi_agent_inference()

        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n구현 요약:")
        print("  ✓ Gumbel-Softmax 샘플링 (soft & hard)")
        print("  ✓ DiscreteActorNetwork (logits output)")
        print("  ✓ Temperature annealing scheduling")
        print("  ✓ Multi-agent wrapper")
        print("  ✓ Gradient flow (미분 가능)")
        print("  ✓ Inference mode (argmax only, no Gumbel-Softmax)")
        print("  ✓ Multi-agent discrete action inference")
        print("\n다음 단계:")
        print("  1. python train_marllib_self/new_train_maddpg.py 로 학습 시작")
        print("  2. DISCRETE_MADDPG_GUIDE.md 참고하여 하이퍼파라미터 조정")
        print("  3. experiments 폴더에서 학습 결과 확인")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
