"""
Chapter 4.4: Maximum A Posteriori (MAP) & Regularization Links (L1/L2)
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Grid Verification (MLE vs. Gaussian MAP vs. Laplace MAP).
2. Bayesian Linear Regression Gaussian Conjugacy Verification.
3. Soft-Thresholding Operator & Exact L1 Sparsity Verification.
4. Adam vs. AdamW Decoupled Weight Decay Experiment.
"""

import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Visual Grid Verification
# =====================================================================
def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 4.4 Part 5:
    x = [1.0, 2.0], y = [3.0, 5.0], sigma^2 = 1.0.
    Model: y_hat = w * x
    """
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([3.0, 5.0], dtype=np.float64)
    sigma2 = 1.0
    sigma0_2 = 0.5  # Gaussian prior variance
    b_scale = 0.5   # Laplace prior scale

    # 1. Sufficient Statistics
    sum_x2 = float(np.sum(x ** 2))  # 5.0
    sum_xy = float(np.sum(x * y))   # 13.0

    # 2. Analytical Closed Forms
    w_mle = sum_xy / sum_x2  # 13.0 / 5.0 = 2.6000
    w_map_gauss = sum_xy / (sum_x2 + sigma2 / sigma0_2)  # 13.0 / (5.0 + 2.0) = 13/7 = 1.857143
    # For Laplace: d/dw [0.5 * (5 w^2 - 26 w) + (1/b) * w] = 5 w - 13 + 2 = 5 w - 11 = 0
    w_map_laplace = (sum_xy - sigma2 / b_scale) / sum_x2  # (13.0 - 2.0) / 5.0 = 11/5 = 2.2000

    print("--- Part 5: Visual Grid Verification ---")
    print(f"Sum x^2:           {sum_x2:.4f} (Expected: 5.0000)")
    print(f"Sum x*y:           {sum_xy:.4f} (Expected: 13.0000)")
    print(f"Analytical w_MLE:  {w_mle:.4f} (Expected: 2.6000)")
    print(f"Gaussian MAP w:    {w_map_gauss:.6f} (Expected: 1.857143)")
    print(f"Laplace MAP w:     {w_map_laplace:.4f} (Expected: 2.2000)")

    assert abs(w_mle - 2.6000) < 1e-7
    assert abs(w_map_gauss - 13.0 / 7.0) < 1e-6
    assert abs(w_map_laplace - 2.2000) < 1e-7

    # 3. Numerical Optimization Verification
    def loss_data(w):
        return 0.5 * np.sum((y - w * x) ** 2)

    def loss_gauss_map(w):
        return loss_data(w) + (w ** 2) / (2.0 * sigma0_2)

    def loss_laplace_map(w):
        return loss_data(w) + np.abs(w) / b_scale

    opt_gauss = opt.minimize_scalar(loss_gauss_map, bounds=(0.0, 5.0), method='bounded')
    opt_laplace = opt.minimize_scalar(loss_laplace_map, bounds=(0.0, 5.0), method='bounded')

    print(f"Scipy Opt Gauss MAP:   {opt_gauss.x:.6f}")
    print(f"Scipy Opt Laplace MAP: {opt_laplace.x:.6f}")

    assert abs(opt_gauss.x - w_map_gauss) < 1e-5
    assert abs(opt_laplace.x - w_map_laplace) < 1e-5
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Bayesian Linear Regression Gaussian Conjugacy Verification
# =====================================================================
def verify_bayesian_conjugacy():
    """
    Verifies that the Bayesian Gaussian posterior mean m_N matches the
    L2 Ridge regression analytical solution exactly.
    """
    np.random.seed(42)
    n = 100
    dim = 5
    x = np.random.randn(n, dim)
    true_w = np.array([1.5, -2.0, 0.0, 0.5, -1.0])
    noise_sigma = 0.5
    y = x @ true_w + np.random.normal(0.0, noise_sigma, size=n)

    # Prior parameters: w ~ N(0, (1/alpha) * I)
    alpha = 2.0  # prior precision = 1 / sigma0^2
    beta = 1.0 / (noise_sigma ** 2)  # likelihood precision = 1 / sigma^2
    lambda_ridge = alpha / beta

    # 1. Posterior Covariance & Mean
    xt_x = x.T @ x
    xt_y = x.T @ y
    s_n_inv = alpha * np.eye(dim) + beta * xt_x
    s_n = np.linalg.inv(s_n_inv)
    m_n = beta * (s_n @ xt_y)

    # 2. Closed-form Ridge Regression solution: (X^T X + lambda I)^{-1} X^T y
    w_ridge = np.linalg.inv(xt_x + lambda_ridge * np.eye(dim)) @ xt_y

    print("\n--- Bayesian Linear Regression vs. Ridge Regression ---")
    print(f"Posterior Mean m_N:    {m_n}")
    print(f"Ridge Estimator w_hat: {w_ridge}")

    assert np.allclose(m_n, w_ridge, atol=1e-10)
    print("✓ Bayesian posterior mode perfectly matches Ridge regression estimator!")


# =====================================================================
# 3. Soft-Thresholding Operator & Exact L1 Sparsity
# =====================================================================
def soft_threshold(w: np.ndarray, lambd: float) -> np.ndarray:
    """Computes S_lambda(w) = sign(w) * max(0, |w| - lambda)."""
    return np.sign(w) * np.maximum(0.0, np.abs(w) - lambd)


def verify_soft_thresholding():
    """
    Verifies that the 1D L1 MAP soft-thresholding operator snaps values
    within [-lambda, lambda] to exact zero.
    """
    lambd = 0.5
    w_inputs = np.array([-2.0, -0.8, -0.4, 0.0, 0.3, 0.9, 2.5])
    expected_outputs = np.array([-1.5, -0.3, 0.0, 0.0, 0.0, 0.4, 2.0])

    outputs = soft_threshold(w_inputs, lambd)

    print("\n--- Soft-Thresholding Operator (Exact L1 Sparsity) ---")
    print(f"Inputs:   {w_inputs}")
    print(f"Outputs:  {outputs}")
    print(f"Expected: {expected_outputs}")

    assert np.allclose(outputs, expected_outputs, atol=1e-10)
    # Check that inside [-0.5, 0.5], output is strictly 0.0
    assert outputs[2] == 0.0
    assert outputs[3] == 0.0
    assert outputs[4] == 0.0
    print("✓ Soft-thresholding exact zero clamping verified!")


# =====================================================================
# 4. Adam vs. AdamW Decoupled Weight Decay Experiment
# =====================================================================
def verify_adam_vs_adamw():
    """
    Demonstrates why L2 regularization fails in Adam:
    Parameters with large gradient updates have their L2 weight decay suppressed,
    whereas AdamW applies scale-invariant decoupled shrinkage.
    """
    torch.manual_seed(42)

    # Two independent weights:
    # w1 will experience large gradients (e.g. 10.0)
    # w2 will experience tiny gradients (e.g. 0.1)
    w_adam = nn.Parameter(torch.tensor([5.0, 5.0]))
    w_adamw = nn.Parameter(torch.tensor([5.0, 5.0]))

    # Target gradient scaling factors
    grad_scale = torch.tensor([10.0, 0.1])

    # 1. Adam with L2 regularization (weight_decay passed to Adam acts on gradient)
    opt_adam = torch.optim.Adam([w_adam], lr=0.1, weight_decay=0.2)
    # 2. AdamW with decoupled weight decay
    opt_adamw = torch.optim.AdamW([w_adamw], lr=0.1, weight_decay=0.2)

    for _ in range(20):
        # Step Adam
        opt_adam.zero_grad()
        # Simulated dummy loss producing scaled gradient
        loss_adam = torch.sum(w_adam * grad_scale)
        loss_adam.backward()
        opt_adam.step()

        # Step AdamW
        opt_adamw.zero_grad()
        loss_adamw = torch.sum(w_adamw * grad_scale)
        loss_adamw.backward()
        opt_adamw.step()

    print("\n--- Adam (L2 Penalty) vs. AdamW (Decoupled Weight Decay) ---")
    print(f"Initial Weights: [5.0, 5.0]")
    print(f"Adam final weights:  w1 (large grad) = {w_adam[0].item():.4f}, w2 (small grad) = {w_adam[1].item():.4f}")
    print(f"AdamW final weights: w1 (large grad) = {w_adamw[0].item():.4f}, w2 (small grad) = {w_adamw[1].item():.4f}")

    # In AdamW, weight decay is applied as w = w - lr * wd * w directly, preserving proper shrinkage
    assert w_adamw[0].item() < w_adam[0].item()
    print("✓ AdamW decoupled weight decay behavior verified!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 4.4 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_bayesian_conjugacy()
    verify_soft_thresholding()
    verify_adam_vs_adamw()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 4.4 PASSED CLEANLY!")
    print("=================================================================")
