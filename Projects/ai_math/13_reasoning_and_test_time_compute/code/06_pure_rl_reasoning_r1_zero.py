"""
Pure Reinforcement Learning for Reasoning: DeepSeek-R1-Zero & GRPO
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> tuple[float, float, torch.Tensor]:
    """
    Computes group mean, std, and normalized advantages: A_i = (R_i - mean) / (std + eps).
    Uses population standard deviation (unbiased=False) as in standard GRPO.
    """
    mean = rewards.mean().item()
    std = torch.std(rewards, unbiased=False).item()
    advantages = (rewards - mean) / (std + eps)
    return mean, std, advantages


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    
    # Rewards: [1.1, 0.1, 1.0, 0.0]
    rewards = torch.tensor([1.10, 0.10, 1.00, 0.00], dtype=torch.float32)
    mean, std, advantages = compute_group_advantages(rewards)
    
    expected_mean = 0.550000
    expected_std = 0.502494
    expected_adv = torch.tensor([1.094541, -0.895534, 0.895534, -1.094541], dtype=torch.float32)
    
    assert math.isclose(mean, expected_mean, abs_tol=1e-5), f"Mean mismatch: {mean}"
    assert math.isclose(std, expected_std, abs_tol=1e-5), f"Std mismatch: {std}"
    assert torch.allclose(advantages, expected_adv, atol=1e-5), f"Advantages mismatch: {advantages}"
    
    print(f"Group Mean: {mean:.6f} -> EXACT MATCH {expected_mean}")
    print(f"Group Std:  {std:.6f} -> EXACT MATCH {expected_std}")
    print(f"Advantages: {advantages.tolist()} -> EXACT MATCH\n")


def rule_based_verifier(completion: str, target_answer: str) -> tuple[float, float, float]:
    """
    DeepSeek-R1-Zero Rule-Based Verifier:
    Returns (acc_reward, format_reward, total_reward).
    """
    # 1. Format verification
    format_pattern = r"^<think>(.*?)</think><answer>(.*?)</answer>$"
    match = re.match(format_pattern, completion, re.DOTALL)
    
    if match:
        format_reward = 0.1
        extracted_answer = match.group(2).strip()
    else:
        format_reward = 0.0
        # Fallback extraction for accuracy check
        ans_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        extracted_answer = ans_match.group(1).strip() if ans_match else completion.strip()
        
    # 2. Accuracy verification (symbolic/string match)
    if extracted_answer == target_answer.strip():
        acc_reward = 1.0
    else:
        acc_reward = 0.0
        
    total_reward = acc_reward + format_reward
    return acc_reward, format_reward, total_reward


def test_rule_based_verifier():
    print("--- 2. Testing Rule-Based Verifier on Rollouts ---")
    target = "15"
    
    o1 = "<think>2x + 7 = 37 => 2x = 30 => x = 15</think><answer>15</answer>"
    o2 = "<think>2x + 7 = 37 => 2x = 24 => x = 12</think><answer>12</answer>"
    o3 = "<answer>15</answer>" # missing think tag
    o4 = "I think it is 42 maybe?"
    
    r1 = rule_based_verifier(o1, target)
    r2 = rule_based_verifier(o2, target)
    r3 = rule_based_verifier(o3, target)
    r4 = rule_based_verifier(o4, target)
    
    assert r1 == (1.0, 0.1, 1.1)
    assert r2 == (0.0, 0.1, 0.1)
    assert r3 == (1.0, 0.0, 1.0)
    assert r4 == (0.0, 0.0, 0.0)
    
    print(f"Rollout 1 (Math + Tags): Reward = {r1[2]:.2f}")
    print(f"Rollout 2 (Wrong Math):  Reward = {r2[2]:.2f}")
    print(f"Rollout 3 (No Tags):     Reward = {r3[2]:.2f}")
    print(f"Rollout 4 (Garbled):     Reward = {r4[2]:.2f}")
    print("Rule-based scoring verified perfectly!\n")


def test_grpo_gradient_update():
    print("--- 3. Testing PyTorch GRPO Policy Update Simulation ---")
    torch.manual_seed(42)
    G = 4
    
    # 4 log-probabilities from old policy and new policy
    logp_old = torch.tensor([-2.5, -3.0, -2.8, -4.0], dtype=torch.float32)
    # Trainable log-probabilities under updated policy
    logp_new = torch.tensor([-2.5, -3.0, -2.8, -4.0], dtype=torch.float32, requires_grad=True)
    
    # Normalized advantages from Part 5
    advantages = torch.tensor([1.094541, -0.895534, 0.895534, -1.094541], dtype=torch.float32)
    
    # Compute probability ratios
    ratios = torch.exp(logp_new - logp_old) # all 1.0 initially
    
    # GRPO loss: - (1/G) * sum(ratio * A)
    loss = -torch.mean(ratios * advantages)
    loss.backward()
    
    # Check gradients:
    # dLoss / d(logp_new) = - (1/G) * A
    expected_grad = -advantages / G
    assert torch.allclose(logp_new.grad, expected_grad, atol=1e-5)
    
    # Rollout 1 has positive advantage, so gradient on loss is negative (increasing logp_new reduces loss)
    assert logp_new.grad[0] < 0, "Rollout 1 must be reinforced (negative loss gradient)"
    assert logp_new.grad[1] > 0, "Rollout 2 must be suppressed (positive loss gradient)"
    assert logp_new.grad[3] > 0, "Rollout 4 must be heavily suppressed"
    
    print(f"GRPO Loss: {loss.item():.6f}")
    print(f"Policy Gradients: {logp_new.grad.tolist()}")
    print("GRPO Gradient Update verified: Highest positive boost on Rollout 1!")
    print("All tests in Chapter 13.6 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    test_rule_based_verifier()
    test_grpo_gradient_update()
    print("🟢 Chapter 13.6 Verification 100% Complete.")
