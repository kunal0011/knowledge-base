"""
Group Relative Policy Optimization (GRPO) & Frontier Reasoning (DeepSeek-R1) Verification Suite
Module 11: Reinforcement Learning - Chapter 28

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Group mean (0.525000), standard deviation (0.476314)
   - Standardized group advantages [+0.997241, +0.997241, -0.892269, -1.102214]
   - Zero-sum group property verification: sum(A_i) == 0.000000 (< 1e-12)
   - Schulman unbiased non-negative KL divergence (0.004401) and net token objective (1.096745)
2. Deterministic Rule-Based Verifiers (Illustration 1):
   - Math accuracy verifier extracting \\boxed{...}
   - Formatting verifier validating <think> and <answer> XML tags
3. Group Advantage Calculation on G = 6 Completions (Illustration 2):
   - Exact mean (0.500000), variance (0.250000), standard deviation (0.500000)
   - Standardized advantages [+1.0, -1.0, +1.0, +1.0, -1.0, -1.0] and zero-sum check
4. Token-Level Clipped Surrogate Forward Pass (Illustration 3):
   - Positive advantage completion (surrogate = 1.066667)
   - Negative advantage completion (surrogate = -0.966667)
5. Schulman Non-Negative KL Divergence Suite (Illustration 4):
   - Exact validation across 4 regimes proving non-negativity and identifying naive log-ratio flaws
6. Memory Footprint Comparison: PPO vs GRPO on 8x H100 (Illustration 5):
   - 50.0% static VRAM savings (504 GB vs 252 GB for 14B; 1152 GB vs 576 GB for 32B)
7. PyTorch Vectorized GRPO Engine:
   - Full implementation of Critic-free group-relative advantage estimation
   - Vectorized Schulman KL divergence regularizer
   - End-to-end forward and backward gradient verification.
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' GRPO GROUP ADVANTAGES & SCHULMAN KL")
    print("=" * 70)

    # Candidate completion rewards: [o1, o2, o3, o4]
    rewards = np.array([1.0, 1.0, 0.1, 0.0])
    G = len(rewards)

    # 1. Group Mean & Variance
    mu_G = np.mean(rewards)
    # DeepSeek uses population standard deviation with small epsilon (1e-6)
    var_G = np.mean((rewards - mu_G)**2)
    sigma_G = np.sqrt(var_G)

    print(f"Group Mean mu_G:      {mu_G:.6f} (Expected: 0.525000)")
    print(f"Group Variance:       {var_G:.6f} (Expected: 0.226875)")
    print(f"Group Std sigma_G:    {sigma_G:.6f} (Expected: 0.476314)")

    assert np.isclose(mu_G, 0.525, atol=1e-6)
    assert np.isclose(var_G, 0.226875, atol=1e-6)
    assert np.isclose(sigma_G, 0.47631397, atol=1e-6)

    # 2. Standardized Advantages
    advs = (rewards - mu_G) / sigma_G
    print(f"Advantage A_1 (o1 correct): {advs[0]:.6f} (Expected: +0.997241)")
    print(f"Advantage A_2 (o2 correct): {advs[1]:.6f} (Expected: +0.997241)")
    print(f"Advantage A_3 (o3 wrong):   {advs[2]:.6f} (Expected: -0.892269)")
    print(f"Advantage A_4 (o4 broken):  {advs[3]:.6f} (Expected: -1.102214)")

    expected_advs = np.array([0.99724136, 0.99724136, -0.89226859, -1.10221414])
    assert np.allclose(advs, expected_advs, atol=1e-5)

    # Zero-sum property: sum of advantages must be identically 0
    sum_advs = np.sum(advs)
    print(f"Zero-Sum Advantage Check: {sum_advs:.12f} (Expected: 0.000000000000)")
    assert np.isclose(sum_advs, 0.0, atol=1e-12)

    # 3. Token PPO & Schulman KL on Completion o1
    pi_theta = 0.55
    pi_old = 0.50
    pi_ref = 0.50
    beta = 0.05
    eps = 0.20

    rho = pi_theta / pi_old  # 1.10
    A_1 = advs[0]

    # PPO surrogate
    surr1 = rho * A_1
    surr2 = np.clip(rho, 1.0 - eps, 1.0 + eps) * A_1
    ppo_surr = min(surr1, surr2)  # 1.10 * 0.997241 = 1.096965

    # Schulman KL: u = pi_ref / pi_theta -> u - ln(u) - 1
    u = pi_ref / pi_theta  # 10 / 11 ≈ 0.909091
    kl_schulman = u - np.log(u) - 1.0  # 0.0044005

    net_token_obj = ppo_surr - beta * kl_schulman  # 1.096965 - 0.05 * 0.004401 ≈ 1.096745

    print(f"PPO Ratio rho:        {rho:.4f} (Expected: 1.1000)")
    print(f"PPO Surrogate:        {ppo_surr:.6f} (Expected: 1.096965)")
    print(f"Schulman Token KL:    {kl_schulman:.6f} (Expected: 0.004401)")
    print(f"Net Token Objective:  {net_token_obj:.6f} (Expected: 1.096745)")

    assert np.isclose(ppo_surr, 1.0969655, atol=1e-5)
    assert np.isclose(kl_schulman, 0.0044005, atol=1e-5)
    assert np.isclose(net_token_obj, 1.09674547, atol=1e-5)
    print(">> SUCCESS: GRPO group advantages, zero-sum property, and Schulman KL verified to exact precision!\n")


# =====================================================================
# 2. Deterministic Rule-Based Verifiers (Math & Format)
# =====================================================================

def math_accuracy_verifier(completion: str, ground_truth: str) -> float:
    """
    Extracts the answer from \\boxed{...} and compares to ground truth.
    """
    match = re.search(r'\\boxed\{([^}]+)\}', completion)
    if not match:
        return 0.0
    extracted = match.group(1).strip()
    return 1.0 if extracted == ground_truth.strip() else 0.0


def format_verifier(completion: str) -> float:
    """
    Validates presence of <think> ... </think> and <answer> ... </answer> tags.
    """
    has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
    return 0.1 if (has_think and has_answer) else 0.0


def verify_rule_based_verifiers():
    print("=" * 70)
    print("2. VERIFYING DETERMINISTIC VERIFIER FUNCTIONS")
    print("=" * 70)

    correct_sample = "<think>2 + 3 * 4 = 2 + 12 = 14</think>\n<answer>\\boxed{14}</answer>"
    wrong_math_sample = "<think>2 + 3 * 4 = 5 * 4 = 20</think>\n<answer>\\boxed{20}</answer>"
    broken_format_sample = "The answer is 14."

    gt = "14"

    # Test Sample 1: Perfect
    r_acc_1 = math_accuracy_verifier(correct_sample, gt)
    r_fmt_1 = format_verifier(correct_sample)
    print(f"Sample 1: Accuracy = {r_acc_1:.1f}, Format = {r_fmt_1:.1f} | Total = {r_acc_1 + r_fmt_1:.1f}")
    assert r_acc_1 == 1.0 and r_fmt_1 == 0.1

    # Test Sample 2: Wrong math, valid format
    r_acc_2 = math_accuracy_verifier(wrong_math_sample, gt)
    r_fmt_2 = format_verifier(wrong_math_sample)
    print(f"Sample 2: Accuracy = {r_acc_2:.1f}, Format = {r_fmt_2:.1f} | Total = {r_acc_2 + r_fmt_2:.1f}")
    assert r_acc_2 == 0.0 and r_fmt_2 == 0.1

    # Test Sample 3: Broken format
    r_acc_3 = math_accuracy_verifier(broken_format_sample, gt)
    r_fmt_3 = format_verifier(broken_format_sample)
    print(f"Sample 3: Accuracy = {r_acc_3:.1f}, Format = {r_fmt_3:.1f} | Total = {r_acc_3 + r_fmt_3:.1f}")
    assert r_acc_3 == 0.0 and r_fmt_3 == 0.0

    print(">> SUCCESS: Rule-based accuracy and formatting verifiers function properly!\n")


# =====================================================================
# 3. PyTorch Vectorized GRPO Engine
# =====================================================================

def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes standardized group advantages across group dimension:
    rewards: shape (G,) or (B, G)
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False) + eps
    return (rewards - mean) / std


def compute_grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.05
) -> torch.Tensor:
    """
    Computes the DeepSeek-R1 GRPO Loss:
    Args:
        logprobs: Tensor of shape (G, T) from current policy
        old_logprobs: Tensor of shape (G, T) from rollout policy
        ref_logprobs: Tensor of shape (G, T) from frozen SFT model
        advantages: Tensor of shape (G,)
    """
    # Expand advantage across token dimension T: (G, 1)
    adv_expanded = advantages.unsqueeze(1)

    # Importance sampling ratio: rho_{i, t}
    ratio = torch.exp(logprobs - old_logprobs)

    # Clipped PPO surrogate
    surr1 = ratio * adv_expanded
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv_expanded
    ppo_loss = torch.min(surr1, surr2)

    # Schulman unbiased KL divergence: u - ln(u) - 1, where u = pi_ref / pi_theta
    # ln(u) = ref_logprob - logprob -> u = exp(ref_logprob - logprob)
    log_u = ref_logprobs - logprobs
    u = torch.exp(log_u)
    kl_penalty = u - log_u - 1.0

    # Net objective per token
    token_objective = ppo_loss - beta * kl_penalty

    # Maximize objective -> minimize negative loss averaged over group G and length T
    loss = -token_objective.mean()
    return loss


def verify_grpo_pytorch_engine():
    print("=" * 70)
    print("3. VERIFYING PYTORCH GRPO ENGINE & GRADIENT PROPAGATION")
    print("=" * 70)

    torch.manual_seed(42)
    G = 4  # Group size
    T = 6  # Sequence length (tokens)

    # Mock rewards from verifiers
    rewards = torch.tensor([1.0, 1.0, 0.1, 0.0], dtype=torch.float32)
    advs = compute_group_advantages(rewards)

    print(f"PyTorch Computed Advantages: {advs.numpy()}")
    assert torch.isclose(advs.sum(), torch.tensor(0.0), atol=1e-6)

    # Mock policy token logits
    old_logprobs = torch.full((G, T), -0.6931, dtype=torch.float32)
    ref_logprobs = torch.full((G, T), -0.6931, dtype=torch.float32)

    # Learnable policy logits
    logits = nn.Parameter(torch.full((G, T), -0.6931, dtype=torch.float32, requires_grad=True))

    loss = compute_grpo_loss(logits, old_logprobs, ref_logprobs, advs, epsilon=0.2, beta=0.05)
    print(f"Computed GRPO Initial Loss: {loss.item():.6f}")

    # Backward gradient pass
    loss.backward()
    assert logits.grad is not None
    print(f"Logit Gradients for Completion 1 (Positive Adv): {logits.grad[0, 0].item():.6f}")
    print(f"Logit Gradients for Completion 4 (Negative Adv): {logits.grad[3, 0].item():.6f}")

    # For positive advantage (o1), gradient of loss should be negative (pushing logits UP!)
    assert logits.grad[0, 0].item() < 0.0, "Positive advantage must have negative loss gradient to increase probability!"
    # For negative advantage (o4), gradient of loss should be positive (pushing logits DOWN!)
    assert logits.grad[3, 0].item() > 0.0, "Negative advantage must have positive loss gradient to decrease probability!"

    print(">> SUCCESS: GRPO correctly reinforces winning completions and penalizes losing completions without a critic!\n")


# =====================================================================
# 4. Solved Illustrations Verification Suite (Illustrations 2, 3, 4, 5)
# =====================================================================

def verify_illustration2_group_advantages():
    print("=" * 70)
    print("4. VERIFYING ILLUSTRATION 2: GRPO ADVANTAGES ON G = 6 COMPLETIONS")
    print("=" * 70)
    r = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    mu_G = np.mean(r)
    var_G = np.mean((r - mu_G)**2)
    std_G = np.sqrt(var_G)
    advs = (r - mu_G) / std_G

    print(f"Group Mean mu_G:      {mu_G:.6f} (Expected: 0.500000)")
    print(f"Group Variance:       {var_G:.6f} (Expected: 0.250000)")
    print(f"Group Std:            {std_G:.6f} (Expected: 0.500000)")
    print(f"Advantages:           {advs}")
    print(f"Zero-Sum Check:       {np.sum(advs):.12f} (Expected: 0.0)")

    assert np.isclose(mu_G, 0.5, atol=1e-6)
    assert np.isclose(var_G, 0.25, atol=1e-6)
    assert np.isclose(std_G, 0.5, atol=1e-6)
    assert np.allclose(advs, [1.0, -1.0, 1.0, 1.0, -1.0, -1.0], atol=1e-6)
    assert np.isclose(np.sum(advs), 0.0, atol=1e-12)
    print(">> SUCCESS: Illustration 2 group advantages verified to exact precision!\n")


def verify_illustration3_clipped_surrogate():
    print("=" * 70)
    print("5. VERIFYING ILLUSTRATION 3: TOKEN-LEVEL CLIPPED SURROGATE FORWARD PASS")
    print("=" * 70)
    eps_clip = 0.20

    # Completion 1: A1 = +1.0
    pi_old_1 = np.array([0.40, 0.50, 0.80])
    pi_new_1 = np.array([0.44, 0.65, 0.72])
    rho_1 = pi_new_1 / pi_old_1
    A1 = 1.0

    t1_1 = rho_1 * A1
    t2_1 = np.clip(rho_1, 1.0 - eps_clip, 1.0 + eps_clip) * A1
    surr_1 = np.minimum(t1_1, t2_1)
    mean_surr_1 = np.mean(surr_1)

    print(f"Comp 1 Token Ratios:     {rho_1}")
    print(f"Comp 1 Clipped Surr:     {surr_1} (Expected: [1.1, 1.2, 0.9])")
    print(f"Comp 1 Mean Surrogate:   {mean_surr_1:.6f} (Expected: 1.066667)")

    assert np.allclose(rho_1, [1.10, 1.30, 0.90], atol=1e-6)
    assert np.allclose(surr_1, [1.10, 1.20, 0.90], atol=1e-6)
    assert np.isclose(mean_surr_1, 3.20 / 3.0, atol=1e-6)

    # Completion 2: A2 = -1.0
    pi_old_2 = np.array([0.50, 0.50, 0.40])
    pi_new_2 = np.array([0.60, 0.35, 0.36])
    rho_2 = pi_new_2 / pi_old_2
    A2 = -1.0

    t1_2 = rho_2 * A2
    t2_2 = np.clip(rho_2, 1.0 - eps_clip, 1.0 + eps_clip) * A2
    surr_2 = np.minimum(t1_2, t2_2)
    mean_surr_2 = np.mean(surr_2)

    print(f"Comp 2 Token Ratios:     {rho_2}")
    print(f"Comp 2 Clipped Surr:     {surr_2} (Expected: [-1.2, -0.8, -0.9])")
    print(f"Comp 2 Mean Surrogate:   {mean_surr_2:.6f} (Expected: -0.966667)")

    assert np.allclose(rho_2, [1.20, 0.70, 0.90], atol=1e-6)
    assert np.allclose(surr_2, [-1.20, -0.80, -0.90], atol=1e-6)
    assert np.isclose(mean_surr_2, -2.90 / 3.0, atol=1e-6)

    pair_obj = 0.5 * (mean_surr_1 + mean_surr_2)
    print(f"Combined Pair Objective: {pair_obj:.6f} (Expected: 0.050000)")
    assert np.isclose(pair_obj, 0.05, atol=1e-6)
    print(">> SUCCESS: Illustration 3 clipped surrogate forward pass verified!\n")


def verify_illustration4_schulman_kl():
    print("=" * 70)
    print("6. VERIFYING ILLUSTRATION 4: SCHULMAN NON-NEGATIVE KL DIVERGENCE")
    print("=" * 70)
    cases = [
        ("Case 1 (Exact match)", 0.50, 0.50, 0.000000, 0.000000),
        ("Case 2 (Probability boost)", 0.60, 0.40, 0.072132, 0.405465),
        ("Case 3 (Probability drop)", 0.30, 0.60, 0.306853, -0.693147),
        ("Case 4 (Severe divergence)", 0.80, 0.20, 0.636294, 1.386294),
    ]

    for name, p_theta, p_ref, exp_schulman, exp_naive in cases:
        u = p_ref / p_theta
        ln_u = np.log(u)
        kl_schulman = u - ln_u - 1.0
        kl_naive = np.log(p_theta / p_ref)
        print(f"{name:28s} | Schulman: {kl_schulman:.6f} | Naive: {kl_naive:+.6f}")
        assert np.isclose(kl_schulman, exp_schulman, atol=1e-5)
        assert np.isclose(kl_naive, exp_naive, atol=1e-5)
        assert kl_schulman >= -1e-12

    print(">> SUCCESS: Illustration 4 Schulman KL verified to guarantee non-negativity!\n")


def verify_illustration5_memory_footprint():
    print("=" * 70)
    print("7. VERIFYING ILLUSTRATION 5: PPO VS GRPO VRAM FOOTPRINT ON 8X H100")
    print("=" * 70)
    # 14B model
    N_14 = 14.0  # billions
    ppo_static_14 = (16 + 16 + 2 + 2) * N_14
    grpo_static_14 = (16 + 2) * N_14
    ppo_per_gpu_14 = ppo_static_14 / 8.0
    grpo_per_gpu_14 = grpo_static_14 / 8.0

    print(f"14B Model - PPO Static:  {ppo_static_14:.1f} GB ({ppo_per_gpu_14:.2f} GB/GPU)")
    print(f"14B Model - GRPO Static: {grpo_static_14:.1f} GB ({grpo_per_gpu_14:.2f} GB/GPU)")
    print(f"14B VRAM Savings:        {ppo_static_14 - grpo_static_14:.1f} GB ({(ppo_static_14 - grpo_static_14)/ppo_static_14 * 100:.1f}%)")
    print(f"14B Dynamic Headroom:    PPO = {80.0 - ppo_per_gpu_14:.2f} GB, GRPO = {80.0 - grpo_per_gpu_14:.2f} GB")

    assert np.isclose(ppo_static_14, 504.0)
    assert np.isclose(grpo_static_14, 252.0)
    assert np.isclose(ppo_per_gpu_14, 63.0)
    assert np.isclose(grpo_per_gpu_14, 31.5)
    assert np.isclose((ppo_static_14 - grpo_static_14) / ppo_static_14, 0.50)

    # 32B model
    N_32 = 32.0  # billions
    ppo_static_32 = (16 + 16 + 2 + 2) * N_32
    grpo_static_32 = (16 + 2) * N_32
    ppo_per_gpu_32 = ppo_static_32 / 8.0
    grpo_per_gpu_32 = grpo_static_32 / 8.0

    print(f"32B Model - PPO Static:  {ppo_static_32:.1f} GB ({ppo_per_gpu_32:.2f} GB/GPU)")
    print(f"32B Model - GRPO Static: {grpo_static_32:.1f} GB ({grpo_per_gpu_32:.2f} GB/GPU)")

    assert np.isclose(ppo_static_32, 1152.0)
    assert np.isclose(grpo_static_32, 576.0)
    assert np.isclose(ppo_per_gpu_32, 144.0)
    assert np.isclose(grpo_per_gpu_32, 72.0)
    print(">> SUCCESS: Illustration 5 memory calculations verified with exact precision!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_rule_based_verifiers()
    verify_grpo_pytorch_engine()
    verify_illustration2_group_advantages()
    verify_illustration3_clipped_surrogate()
    verify_illustration4_schulman_kl()
    verify_illustration5_memory_footprint()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 28 (GRPO & DEEPSEEK-R1) VERIFICATIONS PASSED!")
    print("=" * 70)
