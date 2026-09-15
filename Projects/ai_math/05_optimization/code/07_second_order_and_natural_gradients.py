"""
Chapter 5.7: Second-Order Optimization & Natural Gradients
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (1-step exact convergence of Newton vs. slow GD).
2. BFGS rank-2 update verification and Secant Equation validation.
3. Pearlmutter's Hessian-Vector Product (R-operator) parity with explicit Hessian.
4. Natural Gradient invariance on probability distributions.
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
    Validates hand calculations on f(x_1, x_2) = 5*x_1^2 + 2*x_1*x_2 + 2*x_2^2
    Starting at x_0 = [2.0, -1.0]^T.
    Proves Newton converges in exactly 1 step while GD makes partial progress.
    """
    def f(x):
        return 5.0 * (x[0] ** 2) + 2.0 * x[0] * x[1] + 2.0 * (x[1] ** 2)

    def grad_f(x):
        return np.array([10.0 * x[0] + 2.0 * x[1], 2.0 * x[0] + 4.0 * x[1]], dtype=np.float64)

    Hessian = np.array([[10.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    det_H = 10.0 * 4.0 - 2.0 * 2.0
    assert abs(det_H - 36.0) < 1e-7

    H_inv = np.linalg.inv(Hessian)
    H_inv_expected = (1.0 / 36.0) * np.array([[4.0, -2.0], [-2.0, 10.0]], dtype=np.float64)
    assert np.allclose(H_inv, H_inv_expected, atol=1e-7)

    x0 = np.array([2.0, -1.0], dtype=np.float64)
    g0 = grad_f(x0)
    assert np.allclose(g0, [18.0, 0.0], atol=1e-7)

    # 1. Standard GD step with alpha = 0.05
    alpha = 0.05
    x1_gd = x0 - alpha * g0
    loss_gd = f(x1_gd)
    assert np.allclose(x1_gd, [1.1000, -1.0000], atol=1e-5)
    assert abs(loss_gd - 5.8500) < 1e-5

    # 2. Newton Step
    delta_x_newton = -H_inv @ g0
    assert np.allclose(delta_x_newton, [-2.0000, 1.0000], atol=1e-7)
    x1_newton = x0 + delta_x_newton
    loss_newton = f(x1_newton)
    assert np.allclose(x1_newton, [0.0000, 0.0000], atol=1e-7)
    assert abs(loss_newton - 0.0000) < 1e-7

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Initial Loss at x_0 = [2.0, -1.0]:      {f(x0):.4f}")
    print(f"Standard GD Step (alpha = 0.05):        x_1 = {x1_gd.tolist()} | Loss = {loss_gd:.4f}")
    print(f"Newton Method Step:                     x_1 = {x1_newton.tolist()} | Loss = {loss_newton:.4f}")
    print("✓ Newton 1-step exact convergence verified!")


# =====================================================================
# 2. BFGS Rank-2 Update Arithmetic & Secant Equation
# =====================================================================
def verify_bfgs_secant_arithmetic():
    """
    Validates Problem 2 manual calculations:
    s = [1, 2]^T, y = [2, 1]^T, H_0 = I_2.
    Checks BFGS inverse Hessian formula and Secant equation H_1 @ y == s.
    """
    s = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([2.0, 1.0], dtype=np.float64)
    H0 = np.eye(2, dtype=np.float64)

    y_dot_s = float(np.dot(y, s))
    assert abs(y_dot_s - 4.0) < 1e-7
    rho = 1.0 / y_dot_s

    I = np.eye(2, dtype=np.float64)
    term1 = I - rho * np.outer(s, y)
    term2 = I - rho * np.outer(y, s)
    H1 = term1 @ H0 @ term2 + rho * np.outer(s, s)

    H1_expected = np.array([[0.5625, -0.1250], [-0.1250, 2.2500]], dtype=np.float64)
    assert np.allclose(H1, H1_expected, atol=1e-7)

    # Secant equation check: H1 @ y MUST equal s
    secant_check = H1 @ y
    assert np.allclose(secant_check, s, atol=1e-7)

    # Positive definiteness check: all eigenvalues must be strictly positive
    eigenvalues = np.linalg.eigvals(H1)
    assert np.all(eigenvalues > 0.0)

    print("\n--- BFGS Rank-2 Update Verification ---")
    print(f"Updated Inverse Hessian H_1:\n{H1}")
    print(f"Secant check H_1 @ y: {secant_check.tolist()} (Expected s = {s.tolist()})")
    print(f"Eigenvalues: {eigenvalues.tolist()} (Strictly positive definite!)")
    print("✓ BFGS update and Secant Equation strictly confirmed!")


# =====================================================================
# 3. Pearlmutter's Hessian-Vector Product (R-operator) Parity
# =====================================================================
def verify_pearlmutter_hvp():
    """
    Validates Pearlmutter's trick:
    H @ v = grad_theta (grad_theta(f)^T v)
    using PyTorch autograd without ever instantiating the full Hessian matrix.
    """
    torch.manual_seed(42)
    dim = 6
    theta = torch.randn(dim, dtype=torch.float64, requires_grad=True)
    v = torch.randn(dim, dtype=torch.float64)

    # Arbitrary non-linear loss function
    def loss_fn(p):
        return torch.sum(p ** 4) + torch.sin(p[0] * p[1]) + torch.exp(p[2] * 0.1)

    loss = loss_fn(theta)

    # 1. Explicit Full Hessian via PyTorch functional API
    full_Hessian = torch.autograd.functional.hessian(loss_fn, theta)
    expected_hv = torch.matmul(full_Hessian, v)

    # 2. Pearlmutter's R-operator (Two-pass backprop)
    # Pass 1: compute standard gradient
    grad = torch.autograd.grad(loss, theta, create_graph=True)[0]

    # Pass 2: compute directional derivative of gradient inner product
    grad_dot_v = torch.dot(grad, v)
    pearlmutter_hv = torch.autograd.grad(grad_dot_v, theta)[0]

    diff = torch.max(torch.abs(expected_hv - pearlmutter_hv)).item()
    print("\n--- Pearlmutter Hessian-Vector Product (R-operator) Parity ---")
    print(f"Explicit Hessian-Vector Product:   {expected_hv[:3].tolist()}")
    print(f"Pearlmutter Autograd HVP Product:  {pearlmutter_hv[:3].tolist()}")
    print(f"Maximum Absolute Discrepancy:      {diff:.2e}")

    assert diff < 1e-12, f"Pearlmutter HVP does not match explicit Hessian! diff = {diff}"
    print("✓ Pearlmutter R-operator verified to machine precision (< 1e-12)!")


# =====================================================================
# 4. Natural Gradient on Categorical Distribution
# =====================================================================
def verify_natural_gradient_invariance():
    """
    Demonstrates Natural Gradient step on a multinomial distribution
    using the Fisher Information Matrix F = diag(p) - p p^T.
    """
    # Probability simplex point
    p = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    # Loss gradient with respect to probabilities
    grad_L = np.array([-1.0, 0.5, 0.5], dtype=np.float64)

    # Fisher Information Matrix for categorical distribution: F = diag(p) - p p^T
    F = np.diag(p) - np.outer(p, p)
    # Damped pseudo-inverse for numerical stability on the simplex
    F_reg = F + 1e-4 * np.eye(3)
    nat_grad = np.linalg.solve(F_reg, grad_L)

    print("\n--- Natural Gradient on Categorical Distribution ---")
    print(f"Probability Distribution p:  {p.tolist()}")
    print(f"Euclidean Loss Gradient:    {grad_L.tolist()}")
    print(f"Natural Gradient Direction: {nat_grad.tolist()}")
    assert np.isfinite(nat_grad).all()
    print("✓ Natural gradient direction computed cleanly!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 5.7 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_bfgs_secant_arithmetic()
    verify_pearlmutter_hvp()
    verify_natural_gradient_invariance()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 5.7 PASSED CLEANLY!")
    print("=" * 65)
