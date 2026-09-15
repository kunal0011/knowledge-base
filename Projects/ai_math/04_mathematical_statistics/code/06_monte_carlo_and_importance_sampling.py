"""
Chapter 4.6: Monte Carlo Methods & Importance Sampling
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Grid & ESS Calculation Verification.
2. Rare-Event Tail Expectation: Crude Monte Carlo vs. Importance Sampling.
3. Monte Carlo Estimation of Pi and O(1/sqrt(N)) Convergence.
4. Reinforcement Learning PPO Clipped Importance Sampling Surrogate in PyTorch.
"""

import numpy as np
import scipy.stats as stats
import torch
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Visual Grid Verification
# =====================================================================
def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 4.6 Part 5:
    Target: E_{N(0,1)}[X^2 * 1_{X >= 3}]
    Proposal: q(x) = N(3, 1) => w(x) = exp(-3x + 4.5)
    Samples: [3.0, 3.5, 2.4, 4.0]
    """
    samples = np.array([3.0, 3.5, 2.4, 4.0], dtype=np.float64)

    # 1. Integrand h(x)
    h_vals = np.where(samples >= 3.0, samples ** 2, 0.0)
    expected_h = np.array([9.0000, 12.2500, 0.0000, 16.0000])

    # 2. Importance Weights w(x) = exp(-3x + 4.5)
    w_vals = np.exp(-3.0 * samples + 4.5)
    expected_w = np.array([0.011109, 0.002479, 0.067206, 0.000553])

    # 3. Products h * w
    products = h_vals * w_vals
    expected_products = np.array([0.099981, 0.030368, 0.000000, 0.008848])

    # 4. Final IS Estimate
    i_hat_is = float(np.mean(products))
    expected_i_hat = 0.034799

    # 5. Effective Sample Size ESS
    sum_w = float(np.sum(w_vals))
    sum_w2 = float(np.sum(w_vals ** 2))
    ess = (sum_w ** 2) / sum_w2
    expected_ess = 1.424

    print("--- Part 5: Visual Grid Verification ---")
    print(f"Samples x:           {samples}")
    print(f"h(x) evaluations:    {h_vals} (Expected: {expected_h})")
    print(f"Weights w(x):        {w_vals}")
    print(f"Weighted terms h*w:  {products}")
    print(f"IS Estimate I_hat:   {i_hat_is:.6f} (Expected: ~{expected_i_hat:.6f})")
    print(f"Effective Sample ESS:{ess:.4f} (Expected: ~{expected_ess:.3f})")

    assert np.allclose(h_vals, expected_h, atol=1e-4)
    assert np.allclose(w_vals, expected_w, atol=1e-5)
    assert np.allclose(products, expected_products, atol=1e-5)
    assert abs(i_hat_is - expected_i_hat) < 1e-5
    assert abs(ess - expected_ess) < 0.01
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Crude MC vs. Importance Sampling Variance Comparison
# =====================================================================
def verify_mc_vs_is_variance(num_trials: int = 2000, n_samples: int = 1000):
    """
    Compares the variance of Crude Monte Carlo against Importance Sampling
    when estimating the rare tail integral:
    I = int_3^inf x^2 * (1 / sqrt(2*pi)) * exp(-x^2 / 2) dx.
    """
    np.random.seed(42)

    # 1. Crude Monte Carlo: sample from N(0, 1)
    # Most samples will be < 3 (probability 99.865%), resulting in high estimator variance
    crude_estimates = []
    for _ in range(num_trials):
        x_crude = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
        h_crude = np.where(x_crude >= 3.0, x_crude ** 2, 0.0)
        crude_estimates.append(np.mean(h_crude))

    # 2. Importance Sampling: sample from N(3, 1)
    is_estimates = []
    for _ in range(num_trials):
        x_is = np.random.normal(loc=3.0, scale=1.0, size=n_samples)
        h_is = np.where(x_is >= 3.0, x_is ** 2, 0.0)
        w_is = np.exp(-3.0 * x_is + 4.5)
        is_estimates.append(np.mean(h_is * w_is))

    mean_crude = float(np.mean(crude_estimates))
    var_crude = float(np.var(crude_estimates))

    mean_is = float(np.mean(is_estimates))
    var_is = float(np.var(is_estimates))

    variance_reduction_ratio = var_crude / var_is

    print("\n--- Crude Monte Carlo vs. Importance Sampling (Rare Event) ---")
    print(f"Trials: {num_trials} | Samples per Trial: {n_samples}")
    print(f"Crude MC Mean:    {mean_crude:.6f} | Variance: {var_crude:.8e}")
    print(f"Importance Mean:  {mean_is:.6f} | Variance: {var_is:.8e}")
    print(f"Variance Reduction: {variance_reduction_ratio:.2f}x")

    # Both estimators are unbiased (agree on the mean)
    assert abs(mean_crude - mean_is) < 0.003
    # Importance sampling has vastly lower variance (> 20x reduction)
    assert variance_reduction_ratio > 20.0
    print("✓ Importance Sampling massive variance reduction confirmed!")


# =====================================================================
# 3. Monte Carlo Estimation of Pi
# =====================================================================
def verify_pi_estimation():
    """
    Estimates pi via uniform random sampling in the unit square [-1, 1]^2.
    """
    np.random.seed(42)
    sample_sizes = [1_000, 10_000, 100_000, 1_000_000]

    print("\n--- Monte Carlo Estimation of Pi ---")
    for n in sample_sizes:
        pts = np.random.uniform(-1.0, 1.0, size=(n, 2))
        inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 1.0
        pi_hat = 4.0 * np.mean(inside)
        error = abs(pi_hat - math.pi)
        theo_se = math.sqrt(math.pi * (4.0 - math.pi)) / math.sqrt(n)
        print(f"N = {n:7d} | Pi Estimate = {pi_hat:.6f} | Abs Error = {error:.6f} | Theo SE = {theo_se:.6f}")
        assert error < 3.0 * theo_se


# =====================================================================
# 4. PPO Clipped Importance Sampling Surrogate in PyTorch
# =====================================================================
def verify_ppo_clipped_surrogate():
    """
    Verifies that the PPO clipped objective bounds importance weight explosion:
    L_clip(theta) = min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t)
    """
    epsilon = 0.2  # PPO clip parameter (range [0.8, 1.2])

    # Case 1: Positive advantage with explosive importance weight (policy became much more likely)
    advantage_pos = torch.tensor([2.0])
    ratio_exploding = torch.tensor([2.5])  # r_t = 2.5 > 1 + eps

    surr1_pos = ratio_exploding * advantage_pos  # 2.5 * 2.0 = 5.0
    surr2_pos = torch.clamp(ratio_exploding, 1.0 - epsilon, 1.0 + epsilon) * advantage_pos  # 1.2 * 2.0 = 2.4
    ppo_loss_pos = torch.min(surr1_pos, surr2_pos)

    # Case 2: Negative advantage with collapsing importance weight
    advantage_neg = torch.tensor([-3.0])
    ratio_collapsed = torch.tensor([0.4])  # r_t = 0.4 < 1 - eps

    surr1_neg = ratio_collapsed * advantage_neg  # 0.4 * -3.0 = -1.2
    surr2_neg = torch.clamp(ratio_collapsed, 1.0 - epsilon, 1.0 + epsilon) * advantage_neg  # 0.8 * -3.0 = -2.4
    ppo_loss_neg = torch.min(surr1_neg, surr2_neg)

    print("\n--- Reinforcement Learning PPO Clipped Importance Sampling ---")
    print(f"Positive Advantage (A = 2.0), Exploding Weight (r = 2.5):")
    print(f"  Unclipped Loss: {surr1_pos.item():.2f} | Clipped PPO Loss: {ppo_loss_pos.item():.2f}")
    print(f"Negative Advantage (A = -3.0), Collapsed Weight (r = 0.4):")
    print(f"  Unclipped Loss: {surr1_neg.item():.2f} | Clipped PPO Loss: {ppo_loss_neg.item():.2f}")

    assert abs(ppo_loss_pos.item() - 2.4) < 1e-5  # Clamped by 1 + eps = 1.2
    assert abs(ppo_loss_neg.item() - (-2.4)) < 1e-5  # Clamped by 1 - eps = 0.8
    print("✓ PPO clipped importance weight stabilization verified!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 4.6 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_mc_vs_is_variance()
    verify_pi_estimation()
    verify_ppo_clipped_surrogate()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 4.6 PASSED CLEANLY!")
    print("=================================================================")
