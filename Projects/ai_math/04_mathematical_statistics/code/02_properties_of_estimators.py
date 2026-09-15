"""
Chapter 4.2: Properties of Estimators (Bias, Variance, Consistency, Cramér-Rao)
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Verification (Estimators 1, 2, 3).
2. Analytical & Empirical Optimal Shrinkage Factor (The L2 Weight Decay Link).
3. Cramér-Rao Lower Bound Verification for Bernoulli Estimator.
4. Uniform Distribution Super-Efficiency (O(1/N^2) Variance Scaling).
5. Natural Gradient Descent & Empirical Fisher Information Matrix Preconditioning.
"""

import numpy as np
import scipy.stats as stats
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Numerical Verification
# =====================================================================
def verify_part5_visual_grid(num_trials: int = 200_000):
    """
    Validates the exact mathematical values from Chapter 4.2 Part 5:
    X ~ N(mu = 2.0, sigma^2 = 4.0), N = 4.
    Estimator 1: mu_hat_1 = (1/4) sum(X_i)
    Estimator 2: mu_hat_2 = X_1
    Estimator 3: mu_hat_3 = 0.80 * mu_hat_1
    """
    np.random.seed(42)
    mu = 2.0
    sigma2 = 4.0
    sigma = math.sqrt(sigma2)
    n = 4

    # Theoretical values
    crlb = sigma2 / n  # 4.0 / 4 = 1.0000

    theo_bias_1 = 0.0
    theo_var_1 = 1.0000
    theo_mse_1 = 1.0000
    theo_eff_1 = 1.0000

    theo_bias_2 = 0.0
    theo_var_2 = 4.0000
    theo_mse_2 = 4.0000
    theo_eff_2 = 0.2500

    theo_bias_3 = 0.80 * mu - mu  # -0.4000
    theo_var_3 = (0.80 ** 2) * 1.0000  # 0.6400
    theo_mse_3 = (theo_bias_3 ** 2) + theo_var_3  # 0.16 + 0.64 = 0.8000

    # Monte Carlo simulation
    samples = np.random.normal(loc=mu, scale=sigma, size=(num_trials, n))

    est1 = np.mean(samples, axis=1)
    est2 = samples[:, 0]
    est3 = 0.80 * est1

    emp_bias_1 = float(np.mean(est1) - mu)
    emp_var_1 = float(np.var(est1, ddof=1))
    emp_mse_1 = float(np.mean((est1 - mu) ** 2))

    emp_bias_2 = float(np.mean(est2) - mu)
    emp_var_2 = float(np.var(est2, ddof=1))
    emp_mse_2 = float(np.mean((est2 - mu) ** 2))

    emp_bias_3 = float(np.mean(est3) - mu)
    emp_var_3 = float(np.var(est3, ddof=1))
    emp_mse_3 = float(np.mean((est3 - mu) ** 2))

    print("--- Part 5: Visual Grid Numerical Verification ---")
    print(f"CRLB Theoretical Minimum Variance: {crlb:.4f}")
    print(f"Estimator 1 (Sample Mean):")
    print(f"  Bias: {emp_bias_1:.4f} (Theo: {theo_bias_1:.4f}) | Var: {emp_var_1:.4f} (Theo: {theo_var_1:.4f}) | MSE: {emp_mse_1:.4f} (Theo: {theo_mse_1:.4f})")
    print(f"Estimator 2 (First Observation):")
    print(f"  Bias: {emp_bias_2:.4f} (Theo: {theo_bias_2:.4f}) | Var: {emp_var_2:.4f} (Theo: {theo_var_2:.4f}) | MSE: {emp_mse_2:.4f} (Theo: {theo_mse_2:.4f})")
    print(f"Estimator 3 (0.80 Shrinkage):")
    print(f"  Bias: {emp_bias_3:.4f} (Theo: {theo_bias_3:.4f}) | Var: {emp_var_3:.4f} (Theo: {theo_var_3:.4f}) | MSE: {emp_mse_3:.4f} (Theo: {theo_mse_3:.4f})")

    # Assertions
    assert abs(emp_bias_1 - theo_bias_1) < 0.01
    assert abs(emp_var_1 - theo_var_1) < 0.01
    assert abs(emp_mse_1 - theo_mse_1) < 0.01

    assert abs(emp_bias_2 - theo_bias_2) < 0.02
    assert abs(emp_var_2 - theo_var_2) < 0.03
    assert abs(emp_mse_2 - theo_mse_2) < 0.03

    assert abs(emp_bias_3 - theo_bias_3) < 0.01
    assert abs(emp_var_3 - theo_var_3) < 0.01
    assert abs(emp_mse_3 - theo_mse_3) < 0.01

    # Verify key conceptual insight: Biased Estimator 3 has LOWER MSE than Unbiased MVUE Estimator 1
    assert emp_mse_3 < emp_mse_1
    print("✓ Part 5 visual grid numbers and shrinkage superiority verified!")


# =====================================================================
# 2. Analytical & Empirical Optimal Shrinkage Factor
# =====================================================================
def verify_optimal_shrinkage():
    """
    Shows that for estimator mu_hat(alpha) = alpha * X_bar:
    MSE(alpha) = (alpha - 1)^2 * mu^2 + alpha^2 * (sigma^2 / N).
    The analytical minimizer is alpha* = mu^2 / (mu^2 + sigma^2 / N).
    For mu=2, sigma^2=4, N=4 => alpha* = 4 / (4 + 1) = 0.80.
    """
    mu = 2.0
    sigma2 = 4.0
    n = 4
    var_xbar = sigma2 / n  # 1.0

    # Analytical minimizer
    alpha_star = (mu ** 2) / (mu ** 2 + var_xbar)  # 4.0 / (4.0 + 1.0) = 0.80

    alphas = np.linspace(0.5, 1.2, 71)
    analytical_mses = [(a - 1.0) ** 2 * (mu ** 2) + (a ** 2) * var_xbar for a in alphas]
    min_idx = np.argmin(analytical_mses)
    best_alpha = alphas[min_idx]

    print("\n--- Optimal Shrinkage Factor (Weight Decay Connection) ---")
    print(f"Analytical Optimal alpha*: {alpha_star:.4f}")
    print(f"Grid Minimum alpha:        {best_alpha:.4f}")
    print(f"MSE at alpha = 1.0 (Unbiased): {analytical_mses[np.argmin(np.abs(alphas - 1.0))]:.4f}")
    print(f"MSE at alpha = 0.8 (Optimal):  {analytical_mses[min_idx]:.4f}")

    assert abs(best_alpha - 0.80) < 0.015
    assert analytical_mses[min_idx] < analytical_mses[np.argmin(np.abs(alphas - 1.0))]
    print("✓ Analytical optimal shrinkage factor of 0.80 verified!")


# =====================================================================
# 3. Cramér-Rao Lower Bound for Bernoulli Trials
# =====================================================================
def verify_bernoulli_crlb(num_trials: int = 100_000):
    """
    Verifies that for X ~ Bernoulli(p), CRLB = p * (1 - p) / N,
    and the sample mean achieves exactly 100% efficiency.
    """
    np.random.seed(42)
    p = 0.30
    n = 50
    theo_crlb = p * (1.0 - p) / n  # 0.30 * 0.70 / 50 = 0.004200

    # Draw Bernoulli samples
    samples = np.random.binomial(n=1, p=p, size=(num_trials, n))
    p_hat = np.mean(samples, axis=1)

    emp_mean = float(np.mean(p_hat))
    emp_var = float(np.var(p_hat, ddof=1))
    emp_efficiency = theo_crlb / emp_var

    print("\n--- Bernoulli Cramér-Rao Lower Bound Verification ---")
    print(f"True Success Probability p:  {p:.4f}")
    print(f"Sample Size N:               {n}")
    print(f"Theoretical CRLB:            {theo_crlb:.6f}")
    print(f"Empirical Estimator Var:     {emp_var:.6f}")
    print(f"Empirical Efficiency:        {emp_efficiency * 100:.2f}% (Expected: 100.0%)")

    assert abs(emp_mean - p) < 0.001
    assert abs(emp_var - theo_crlb) < 0.00015
    assert abs(emp_efficiency - 1.0) < 0.02
    print("✓ Bernoulli sample proportion MVUE 100% efficiency verified!")


# =====================================================================
# 4. Uniform Distribution Super-Efficiency (O(1/N^2))
# =====================================================================
def verify_uniform_super_efficiency(num_trials: int = 50_000):
    """
    Verifies that for X ~ Uniform(0, theta), the unbiased estimator
    theta_hat = ((N + 1) / N) * max(X_i) has Var = theta^2 / (N * (N + 2)),
    scaling as O(1/N^2) rather than the standard parametric O(1/N) rate.
    """
    np.random.seed(42)
    theta = 5.0
    sample_sizes = [10, 20, 40, 80]

    print("\n--- Uniform Super-Efficiency Test (O(1/N^2) Scaling) ---")
    prev_var = None
    for n in sample_sizes:
        samples = np.random.uniform(low=0.0, high=theta, size=(num_trials, n))
        x_max = np.max(samples, axis=1)
        theta_hat = ((n + 1.0) / n) * x_max

        emp_mean = float(np.mean(theta_hat))
        emp_var = float(np.var(theta_hat, ddof=1))
        theo_var = (theta ** 2) / (n * (n + 2.0))

        ratio_str = ""
        if prev_var is not None:
            # When N doubles, Var should drop by approximately 4x
            decay_factor = prev_var / emp_var
            ratio_str = f"| Decay factor when N doubled: {decay_factor:.2f}x (Super-efficiency: ~4.0x)"

        print(f"N = {n:2d} | Mean = {emp_mean:.4f} | Emp Var = {emp_var:.6f} | Theo Var = {theo_var:.6f} {ratio_str}")
        assert abs(emp_mean - theta) < 0.02
        assert abs(emp_var - theo_var) / theo_var < 0.05
        prev_var = emp_var

    print("✓ O(1/N^2) super-efficiency confirmed!")


# =====================================================================
# 5. Natural Gradient vs. Euclidean Gradient Step
# =====================================================================
def verify_natural_gradient():
    """
    Computes standard gradient and Fisher-preconditioned Natural Gradient
    for a toy 2-parameter logistic classification model to demonstrate
    Riemannian invariant step adjustment.
    """
    np.random.seed(42)
    n_samples = 200
    dim = 2
    x = np.random.randn(n_samples, dim)
    true_w = np.array([1.5, -2.0])
    logits = x @ true_w
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(np.float64)

    # Current parameter guess
    w = np.array([0.2, 0.1])
    p_hat = 1.0 / (1.0 + np.exp(-x @ w))

    # Standard Euclidean gradient: g = (1/N) * X^T (p_hat - y)
    residuals = p_hat - y
    grad_euclidean = (1.0 / n_samples) * (x.T @ residuals)

    # Empirical Fisher Information Matrix: F = (1/N) sum_i p_i (1 - p_i) x_i x_i^T
    weights = p_hat * (1.0 - p_hat)
    fisher_info = (1.0 / n_samples) * (x.T @ (weights[:, np.newaxis] * x))

    # Natural gradient: g_nat = F^{-1} * g
    fisher_inv = np.linalg.inv(fisher_info)
    grad_natural = fisher_inv @ grad_euclidean

    print("\n--- Natural Gradient Descent vs. Euclidean Gradient ---")
    print(f"Fisher Information Matrix F:\n{fisher_info}")
    print(f"Euclidean Gradient Vector:     {grad_euclidean}")
    print(f"Natural Gradient Vector F^-1 g: {grad_natural}")

    # Fisher information must be strictly symmetric positive definite
    eigvals = np.linalg.eigvalsh(fisher_info)
    assert np.all(eigvals > 0), "Fisher matrix must be positive definite"
    assert np.allclose(fisher_info, fisher_info.T), "Fisher matrix must be symmetric"
    print("✓ Fisher Information Matrix properties & Natural Gradient verified!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 4.2 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_optimal_shrinkage()
    verify_bernoulli_crlb()
    verify_uniform_super_efficiency()
    verify_natural_gradient()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 4.2 PASSED CLEANLY!")
    print("=================================================================")
