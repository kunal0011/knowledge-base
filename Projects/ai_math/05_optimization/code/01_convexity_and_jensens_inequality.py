"""
Chapter 5.1: Convex Sets, Convex Functions & Jensen's Inequality
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Grid & Jensen Gap Variance Identity.
2. First-Order Convexity Supporting Hyperplane Verification.
3. AM-GM Inequality Verification via Concavity of Logarithm.
4. Variational Autoencoder (VAE) ELBO Bound & KL Slack Verification.
5. Neural Network Permutation Symmetry Non-Convexity Proof in PyTorch.
"""

import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Visual Grid Verification
# =====================================================================
def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 5.1 Part 5:
    f(x) = x^2, states = [1.0, 3.0, 6.0], weights = [0.50, 0.30, 0.20]
    """
    x = np.array([1.0, 3.0, 6.0], dtype=np.float64)
    weights = np.array([0.50, 0.30, 0.20], dtype=np.float64)

    # 1. Expectation E[X]
    exp_x = float(np.sum(weights * x))  # 2.6000
    # 2. f(E[X])
    f_exp_x = exp_x ** 2  # 6.7600
    # 3. E[f(X)]
    f_x = x ** 2  # [1.0, 9.0, 36.0]
    exp_f_x = float(np.sum(weights * f_x))  # 10.4000
    # 4. Jensen Gap
    jensen_gap = exp_f_x - f_exp_x  # 3.6400

    # 5. Variance of X: Var(X) = E[X^2] - (E[X])^2
    var_x = float(np.sum(weights * (x - exp_x) ** 2))

    print("--- Part 5: Visual Grid Verification ---")
    print(f"Expectation E[X]:         {exp_x:.4f} (Expected: 2.6000)")
    print(f"f(E[X]):                  {f_exp_x:.4f} (Expected: 6.7600)")
    print(f"E[f(X)]:                  {exp_f_x:.4f} (Expected: 10.4000)")
    print(f"Jensen Gap:               {jensen_gap:.4f} (Expected: 3.6400)")
    print(f"Distribution Variance:    {var_x:.4f} (Expected: 3.6400)")

    assert abs(exp_x - 2.6000) < 1e-7
    assert abs(f_exp_x - 6.7600) < 1e-7
    assert abs(exp_f_x - 10.4000) < 1e-7
    assert abs(jensen_gap - 3.6400) < 1e-7
    assert abs(jensen_gap - var_x) < 1e-7
    assert f_exp_x <= exp_f_x

    # 6. First-Order Under-estimator at x0 = 2.0: T(y) = 4y - 4
    x0 = 2.0
    test_y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f_y = test_y ** 2
    tangent_y = 4.0 * test_y - 4.0
    diff = f_y - tangent_y

    print(f"Tangent Verification diff (f(y) - T(y)): {diff}")
    assert np.all(diff >= -1e-9)
    print("✓ Part 5 visual grid and Jensen variance identity validated!")


# =====================================================================
# 2. First-Order Convexity Supporting Hyperplane Verification
# =====================================================================
def verify_first_order_convexity(num_pairs: int = 5000):
    """
    Verifies that f(y) >= f(x) + grad_f(x)^T (y - x) holds globally
    for a positive definite quadratic form f(x) = 0.5 * x^T A x.
    """
    np.random.seed(42)
    dim = 3
    # Construct random positive definite matrix A
    m = np.random.randn(dim, dim)
    a = m.T @ m + np.eye(dim)

    def f(v):
        return 0.5 * float(v.T @ a @ v)

    def grad_f(v):
        return a @ v

    violations = 0
    for _ in range(num_pairs):
        pt_x = np.random.randn(dim)
        pt_y = np.random.randn(dim)

        f_x = f(pt_x)
        f_y = f(pt_y)
        g_x = grad_f(pt_x)

        linear_approx = f_x + float(g_x.T @ (pt_y - pt_x))
        if f_y < linear_approx - 1e-9:
            violations += 1

    print("\n--- First-Order Convexity Characterization ---")
    print(f"Pairs Tested: {num_pairs} | Violations for Convex Quadratic: {violations}")
    assert violations == 0
    print("✓ First-order tangent supporting hyperplane verified strictly!")


# =====================================================================
# 3. AM-GM Inequality Verification via Jensen's Inequality
# =====================================================================
def verify_am_gm_inequality(num_trials: int = 5000):
    """
    Verifies that (1/N) sum a_i >= (prod a_i)^(1/N) across random positive vectors.
    """
    np.random.seed(42)
    n = 8
    violations = 0
    min_diff = float('inf')

    for _ in range(num_trials):
        a_vec = np.random.uniform(0.1, 10.0, size=n)
        am = float(np.mean(a_vec))
        gm = float(np.exp(np.mean(np.log(a_vec))))

        diff = am - gm
        if diff < -1e-9:
            violations += 1
        if diff < min_diff:
            min_diff = diff

    print("\n--- AM-GM Inequality Verification ---")
    print(f"Trials: {num_trials} | Vector Dimension: {n} | Violations: {violations}")
    print(f"Minimum (AM - GM) Difference: {min_diff:.8e} (Must be >= 0)")

    assert violations == 0
    assert min_diff >= -1e-9
    print("✓ AM-GM inequality verified via Jensen's inequality!")


# =====================================================================
# 4. VAE ELBO Variational Gap & Jensen Slack Verification
# =====================================================================
def verify_elbo_variational_gap():
    """
    Verifies that for a 1D latent variable model:
    ln p(x) >= E_q[ ln p(x, z) - ln q(z) ] = ELBO,
    and that the slack gap ln p(x) - ELBO equals D_KL(q(z) || p(z | x)).
    """
    np.random.seed(42)

    # Model: p(z) = N(0, 1), p(x | z) = N(z, 0.5^2)
    # Observed data: x = 1.2
    x_obs = 1.2
    sigma_x = 0.5

    def p_z(z_val):
        return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (z_val ** 2))

    def p_x_given_z(z_val):
        return (1.0 / (math.sqrt(2.0 * math.pi) * sigma_x)) * np.exp(-0.5 * ((x_obs - z_val) / sigma_x) ** 2)

    def p_joint(z_val):
        return p_x_given_z(z_val) * p_z(z_val)

    # 1. True Marginal log p(x) via numerical quadrature
    marginal_p_x, _ = integrate.quad(p_joint, -10.0, 10.0)
    log_marginal = math.log(marginal_p_x)

    # 2. Variational distribution q(z) = N(mu_q, sigma_q^2)
    mu_q = 0.8
    sigma_q = 0.6

    def q_z(z_val):
        return (1.0 / (math.sqrt(2.0 * math.pi) * sigma_q)) * np.exp(-0.5 * ((z_val - mu_q) / sigma_q) ** 2)

    def elbo_integrand(z_val):
        qz = q_z(z_val)
        pj = p_joint(z_val)
        # q(z) * [ ln p(x, z) - ln q(z) ]
        return qz * (np.log(pj) - np.log(qz))

    elbo_val, _ = integrate.quad(elbo_integrand, -10.0, 10.0)

    # 3. True KL divergence D_KL(q(z) || p(z | x))
    def kl_integrand(z_val):
        qz = q_z(z_val)
        p_post = p_joint(z_val) / marginal_p_x
        return qz * (np.log(qz) - np.log(p_post))

    kl_slack, _ = integrate.quad(kl_integrand, -10.0, 10.0)

    print("\n--- VAE ELBO Variational Gap (Jensen's Inequality) ---")
    print(f"True Log Marginal ln p(x): {log_marginal:.6f}")
    print(f"Variational Lower Bound ELBO:{elbo_val:.6f}")
    print(f"Slack Gap (ln p(x) - ELBO): {log_marginal - elbo_val:.6f}")
    print(f"Exact KL Divergence D_KL:   {kl_slack:.6f}")

    assert elbo_val <= log_marginal
    assert abs((log_marginal - elbo_val) - kl_slack) < 1e-5
    print("✓ Jensen ELBO lower bound and exact KL slack confirmed!")


# =====================================================================
# 5. Neural Network Permutation Symmetry Non-Convexity Proof
# =====================================================================
def verify_neural_network_nonconvexity():
    """
    Demonstrates that neural network loss surfaces are strictly non-convex
    due to hidden unit permutation symmetries.
    """
    torch.manual_seed(42)

    # 1-hidden layer ReLU net on XOR-like symmetry: x in {-2, -1, 1, 2}, y in {1, 0, 0, 1}
    x_train = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = torch.tensor([[1.0], [0.0], [0.0], [1.0]])

    # Model A achieves exact zero loss
    w1_a = torch.tensor([[1.0], [-1.0]])
    b1_a = torch.tensor([[-1.0], [-1.0]])
    w2_a = torch.tensor([[1.0, 1.0]])

    def compute_loss(w1, b1, w2):
        hidden = torch.relu(x_train @ w1.T + b1.T)
        out = hidden @ w2.T
        return float(torch.mean((out - y_train) ** 2).item())

    loss_a = compute_loss(w1_a, b1_a, w2_a)

    # Permute hidden units (swap neuron 1 and neuron 2)
    w1_b = w1_a[[1, 0], :]
    b1_b = b1_a[[1, 0], :]
    w2_b = w2_a[:, [1, 0]]
    loss_b = compute_loss(w1_b, b1_b, w2_b)

    # Compute midpoint in parameter space
    w1_mid = 0.5 * (w1_a + w1_b)
    b1_mid = 0.5 * (b1_a + b1_b)
    w2_mid = 0.5 * (w2_a + w2_b)
    loss_mid = compute_loss(w1_mid, b1_mid, w2_mid)

    print("\n--- Neural Network Non-Convexity (Permutation Symmetry) ---")
    print(f"Loss at Model A:        {loss_a:.6f}")
    print(f"Loss at Permuted Model B:{loss_b:.6f} (Must match Model A exactly)")
    print(f"Loss at Midpoint Model: {loss_mid:.6f}")

    # Permuted model computes identical function
    assert abs(loss_a - loss_b) < 1e-7
    # For a convex function, f(0.5 A + 0.5 B) <= 0.5 f(A) + 0.5 f(B) = loss_a
    # If loss_mid > loss_a, convexity is strictly violated!
    assert loss_mid > loss_a
    print(f"✓ Strictly Non-Convex! Midpoint loss ({loss_mid:.4f}) > Endpoint loss ({loss_a:.4f})!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 5.1 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_first_order_convexity()
    verify_am_gm_inequality()
    verify_elbo_variational_gap()
    verify_neural_network_nonconvexity()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 5.1 PASSED CLEANLY!")
    print("=================================================================")
