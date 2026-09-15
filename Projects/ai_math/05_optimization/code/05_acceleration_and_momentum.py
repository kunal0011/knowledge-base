"""
Chapter 5.5: Acceleration & Momentum (Polyak Momentum & Nesterov NAG)
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid cell-by-cell numerical verification.
2. Spectral analysis & contraction factor on ill-conditioned quadratics (kappa = 81).
3. Valley navigation benchmarks on the Rosenbrock Banana function.
4. PyTorch parity verification with Sutskever's NAG reformulation.
"""

import math
import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 1. Part 5 Visual Grid Numerical Verification
# =====================================================================
def verify_visual_grid():
    """
    Validates hand calculations on f(x_1, x_2) = 10*x_1^2 + x_2^2
    with alpha = 0.08, beta = 0.5, x_0 = [1.0, 1.0]^T.
    """
    def grad_f(x):
        return np.array([20.0 * x[0], 2.0 * x[1]], dtype=np.float64)

    alpha = 0.08
    beta = 0.5
    x0 = np.array([1.0, 1.0], dtype=np.float64)

    # 1. Standard GD
    x_gd = x0.copy()
    g0 = grad_f(x_gd)
    x_gd_1 = x_gd - alpha * g0
    g1 = grad_f(x_gd_1)
    x_gd_2 = x_gd_1 - alpha * g1

    assert np.allclose(x_gd_1, [-0.6000, 0.8400], atol=1e-5), f"GD 1 failed: {x_gd_1}"
    assert np.allclose(x_gd_2, [0.3600, 0.7056], atol=1e-5), f"GD 2 failed: {x_gd_2}"

    # 2. Polyak Heavy-Ball
    x_poly = x0.copy()
    v_poly = np.zeros(2, dtype=np.float64)

    # Iteration 1
    v_poly_1 = beta * v_poly + alpha * grad_f(x_poly)
    x_poly_1 = x_poly - v_poly_1
    assert np.allclose(v_poly_1, [1.6000, 0.1600], atol=1e-5), f"Polyak v1 failed: {v_poly_1}"
    assert np.allclose(x_poly_1, [-0.6000, 0.8400], atol=1e-5), f"Polyak x1 failed: {x_poly_1}"

    # Iteration 2
    v_poly_2 = beta * v_poly_1 + alpha * grad_f(x_poly_1)
    x_poly_2 = x_poly_1 - v_poly_2
    assert np.allclose(v_poly_2, [-0.1600, 0.2144], atol=1e-5), f"Polyak v2 failed: {v_poly_2}"
    assert np.allclose(x_poly_2, [-0.4400, 0.6256], atol=1e-5), f"Polyak x2 failed: {x_poly_2}"

    # 3. Nesterov Accelerated Gradient (Lookahead formulation)
    x_nag = x0.copy()
    v_nag = np.zeros(2, dtype=np.float64)

    # Iteration 1
    x_look_0 = x_nag - beta * v_nag
    v_nag_1 = beta * v_nag + alpha * grad_f(x_look_0)
    x_nag_1 = x_nag - v_nag_1
    assert np.allclose(v_nag_1, [1.6000, 0.1600], atol=1e-5), f"NAG v1 failed: {v_nag_1}"
    assert np.allclose(x_nag_1, [-0.6000, 0.8400], atol=1e-5), f"NAG x1 failed: {x_nag_1}"

    # Iteration 2
    x_look_1 = x_nag_1 - beta * v_nag_1
    assert np.allclose(x_look_1, [-1.4000, 0.7600], atol=1e-5), f"NAG x_look_1 failed: {x_look_1}"
    g_look_1 = grad_f(x_look_1)
    assert np.allclose(g_look_1, [-28.0000, 1.5200], atol=1e-5), f"NAG g_look_1 failed: {g_look_1}"
    v_nag_2 = beta * v_nag_1 + alpha * g_look_1
    x_nag_2 = x_nag_1 - v_nag_2
    assert np.allclose(v_nag_2, [-1.4400, 0.2016], atol=1e-5), f"NAG v2 failed: {v_nag_2}"
    assert np.allclose(x_nag_2, [0.8400, 0.6384], atol=1e-5), f"NAG x2 failed: {x_nag_2}"

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Standard GD:       x_1 = {x_gd_1.tolist()}, x_2 = {x_gd_2.tolist()}")
    print(f"Polyak Heavy-Ball: v_1 = {v_poly_1.tolist()}, x_1 = {x_poly_1.tolist()}")
    print(f"                   v_2 = {v_poly_2.tolist()}, x_2 = {x_poly_2.tolist()}")
    print(f"Nesterov NAG:      x_look_1 = {x_look_1.tolist()}, g_look_1 = {g_look_1.tolist()}")
    print(f"                   v_2 = {v_nag_2.tolist()}, x_2 = {x_nag_2.tolist()}")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Spectral Analysis & Contraction on Ill-Conditioned Quadratic
# =====================================================================
def verify_quadratic_spectral_contraction():
    """
    Quadratic f(x) = 1/2 x^T A x with L = 81, mu = 1 (kappa = 81).
    Validates that Polyak momentum converges at rate rho = 0.80
    versus GD rate rho = 40/41 = 0.97561.
    """
    L = 81.0
    mu = 1.0
    kappa = L / mu

    rho_gd_theory = (kappa - 1.0) / (kappa + 1.0)
    rho_poly_theory = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)

    assert abs(rho_gd_theory - (80.0 / 82.0)) < 1e-6
    assert abs(rho_poly_theory - 0.80) < 1e-6

    # Optimal hyperparameters
    alpha_gd = 2.0 / (L + mu)
    beta_poly = ((math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)) ** 2
    alpha_poly = 4.0 / ((math.sqrt(L) + math.sqrt(mu)) ** 2)

    assert abs(beta_poly - 0.6400) < 1e-6
    assert abs(alpha_poly - 0.0400) < 1e-6

    A = np.diag([L, mu])
    x0 = np.array([1.0, 1.0], dtype=np.float64)

    # Run GD for 41 steps
    x_gd = x0.copy()
    for _ in range(41):
        x_gd -= alpha_gd * (A @ x_gd)

    # Run Polyak for 41 steps
    x_poly = x0.copy()
    v_poly = np.zeros(2, dtype=np.float64)
    for _ in range(41):
        v_poly = beta_poly * v_poly + alpha_poly * (A @ x_poly)
        x_poly -= v_poly

    error_ratio_gd = np.linalg.norm(x_gd) / np.linalg.norm(x0)
    error_ratio_poly = np.linalg.norm(x_poly) / np.linalg.norm(x0)

    print("\n--- Ill-Conditioned Quadratic (kappa = 81) Convergence at 41 Steps ---")
    print(f"Standard GD Contraction Factor:     {rho_gd_theory:.5f} | Error Ratio at 41 steps: {error_ratio_gd:.5f}")
    print(f"Polyak Momentum Contraction Factor: {rho_poly_theory:.5f} | Error Ratio at 41 steps: {error_ratio_poly:.5e}")

    # Polyak error ratio should be <= 1e-2 (5.6e-3), whereas GD error ratio is ~ 0.36
    assert error_ratio_poly < 0.01
    assert error_ratio_poly < 0.02 * error_ratio_gd
    print("✓ Spectral acceleration property strictly confirmed!")


# =====================================================================
# 3. Rosenbrock Banana Valley Benchmark
# =====================================================================
def verify_rosenbrock_navigation():
    """
    Tests GD vs. Polyak vs. NAG on the curved Rosenbrock valley:
    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    Global minimum at (1, 1) where f(1, 1) = 0.
    """
    def rosenbrock(p):
        x, y = p[0], p[1]
        return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    def rosenbrock_grad(p):
        x, y = p[0], p[1]
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x ** 2)
        dy = 200.0 * (y - x ** 2)
        return np.array([dx, dy], dtype=np.float64)

    start = np.array([-1.2, 1.0], dtype=np.float64)
    steps = 2000
    lr = 0.001
    beta = 0.9

    # Standard GD
    p_gd = start.copy()
    for _ in range(steps):
        p_gd -= lr * rosenbrock_grad(p_gd)
    loss_gd = rosenbrock(p_gd)

    # Polyak Momentum
    p_poly = start.copy()
    v_poly = np.zeros(2, dtype=np.float64)
    for _ in range(steps):
        v_poly = beta * v_poly + lr * rosenbrock_grad(p_poly)
        p_poly -= v_poly
    loss_poly = rosenbrock(p_poly)

    # Nesterov NAG
    p_nag = start.copy()
    v_nag = np.zeros(2, dtype=np.float64)
    for _ in range(steps):
        p_look = p_nag - beta * v_nag
        v_nag = beta * v_nag + lr * rosenbrock_grad(p_look)
        p_nag -= v_nag
    loss_nag = rosenbrock(p_nag)

    print("\n--- Rosenbrock Banana Valley Optimization (2000 steps) ---")
    print(f"Standard GD Final Loss:       {loss_gd:.6f} at ({p_gd[0]:.4f}, {p_gd[1]:.4f})")
    print(f"Polyak Momentum Final Loss:   {loss_poly:.6f} at ({p_poly[0]:.4f}, {p_poly[1]:.4f})")
    print(f"Nesterov NAG Final Loss:      {loss_nag:.6f} at ({p_nag[0]:.4f}, {p_nag[1]:.4f})")

    assert loss_poly < loss_gd, "Polyak should significantly outperform standard GD on Rosenbrock"
    assert loss_nag < loss_gd, "NAG should significantly outperform standard GD on Rosenbrock"
    print("✓ Rosenbrock valley navigation superiority verified!")


# =====================================================================
# 4. PyTorch Parity with Sutskever Reformulation
# =====================================================================
def verify_pytorch_sutskever_parity():
    """
    Verifies that Sutskever's single-pass reformulation of NAG matches
    PyTorch's torch.optim.SGD(..., nesterov=True) identically.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    dim = 5
    lr = 0.05
    momentum = 0.9

    # Synthetic non-convex objective weights
    W = np.random.randn(dim, dim)
    A = W.T @ W + 0.5 * np.eye(dim)

    # Scratch Sutskever implementation
    theta_scratch = np.ones(dim, dtype=np.float64)
    v_scratch = np.zeros(dim, dtype=np.float64)

    # PyTorch parameter
    theta_torch = nn.Parameter(torch.ones(dim, dtype=torch.float64))
    optimizer = torch.optim.SGD([theta_torch], lr=lr, momentum=momentum, nesterov=True)
    A_torch = torch.tensor(A, dtype=torch.float64)

    for step in range(20):
        # 1. Scratch update (Sutskever formulation):
        # g_t = A @ theta + sin(theta)
        grad_scratch = A @ theta_scratch + np.sin(theta_scratch)
        v_scratch = momentum * v_scratch + grad_scratch
        theta_scratch -= lr * (grad_scratch + momentum * v_scratch)

        # 2. PyTorch update
        optimizer.zero_grad()
        loss = 0.5 * torch.matmul(theta_torch, torch.matmul(A_torch, theta_torch)) - torch.sum(torch.cos(theta_torch))
        loss.backward()
        optimizer.step()

        # Compare step-by-step
        diff = np.max(np.abs(theta_scratch - theta_torch.detach().numpy()))
        assert diff < 1e-7, f"Discrepancy at step {step}: max diff = {diff}"

    print("\n--- PyTorch Parity Verification ---")
    print(f"Scratch Sutskever Final Parameters: {theta_scratch[:3]}")
    print(f"PyTorch NAG Final Parameters:       {theta_torch.detach().numpy()[:3]}")
    print("✓ PyTorch C++ nesterov implementation matches Sutskever reformulation to float64 precision!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 5.5 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_quadratic_spectral_contraction()
    verify_rosenbrock_navigation()
    verify_pytorch_sutskever_parity()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 5.5 PASSED CLEANLY!")
    print("=" * 65)
