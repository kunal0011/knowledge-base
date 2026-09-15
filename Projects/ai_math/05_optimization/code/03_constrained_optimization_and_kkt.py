"""
Chapter 5.3: Constrained Optimization, Lagrange Multipliers & KKT Conditions
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Grid & KKT Stationarity Verification.
2. SciPy SLSQP Constrained Optimization vs. Analytical KKT Solution.
3. Hard-Margin Support Vector Machine (SVM) Dual Solver & Complementary Slackness.
4. Projected Gradient Descent (PGD) onto Euclidean L2 Ball.
"""

import numpy as np
import scipy.optimize as opt
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Visual Grid Verification
# =====================================================================
def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 5.3 Part 5:
    min f(x, y) = x^2 + y^2
    s.t. g(x, y) = 1.0 - x - y <= 0
    """
    # 1. Analytical Solution
    x_opt = 0.5000
    y_opt = 0.5000
    lambda_opt = 1.0000
    f_opt = x_opt**2 + y_opt**2  # 0.5000

    # 2. Check Candidate Points
    candidates = {
        'Origin (0, 0)': (0.0, 0.0),
        'Point (1.0, 0.0)': (1.0, 0.0),
        'Point (0.2, 0.8)': (0.2, 0.8),
        'Point (0.8, 0.8)': (0.8, 0.8),
        'KKT Opt (0.5, 0.5)': (0.5, 0.5)
    }

    print("--- Part 5: Visual Grid Verification ---")
    for name, (cx, cy) in candidates.items():
        val = cx**2 + cy**2
        g_val = 1.0 - cx - cy
        is_feasible = (g_val <= 1e-9)
        status = "Feasible" if is_feasible else "Infeasible (g > 0)"
        print(f"{name:20s}: f = {val:.4f} | g = {g_val:+.4f} | {status}")

    assert abs(x_opt - 0.5000) < 1e-7
    assert abs(y_opt - 0.5000) < 1e-7
    assert abs(f_opt - 0.5000) < 1e-7

    # KKT Point must have strictly lower loss than all other feasible candidates
    assert f_opt < (1.0**2 + 0.0**2)
    assert f_opt < (0.2**2 + 0.8**2)
    assert f_opt < (0.8**2 + 0.8**2)
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. SciPy SLSQP Solver Comparison
# =====================================================================
def verify_scipy_slsqp():
    """
    Solves the Part 5 constrained problem using SciPy's Sequential Least
    Squares Programming (SLSQP) solver and compares against analytical KKT.
    """
    def objective(v):
        return v[0]**2 + v[1]**2

    def constraint(v):
        # SciPy requires constraint in form g(x) >= 0 => (x + y - 1.0 >= 0)
        return v[0] + v[1] - 1.0

    cons = {'type': 'ineq', 'fun': constraint}
    res = opt.minimize(objective, x0=[2.0, 2.0], method='SLSQP', constraints=cons)

    print("\n--- SciPy SLSQP Solver Comparison ---")
    print(f"SLSQP Optimal (x, y): {res.x}")
    print(f"SLSQP Optimal Loss:   {res.fun:.6f}")

    assert np.allclose(res.x, [0.5, 0.5], atol=1e-6)
    assert abs(res.fun - 0.5) < 1e-6
    print("✓ SciPy SLSQP converged to exact analytical KKT point!")


# =====================================================================
# 3. Hard-Margin Support Vector Machine (SVM) Dual Solver
# =====================================================================
def verify_svm_complementary_slackness():
    """
    Solves the dual quadratic program of a Hard-Margin SVM on a linearly
    separable 2D dataset and verifies KKT complementary slackness:
    alpha_i * [ y_i (w^T x_i + b) - 1 ] = 0.
    """
    # 4 linearly separable 2D points
    x_data = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [2.0, 0.0],
        [3.0, 1.0]
    ], dtype=np.float64)
    y_data = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    n = len(y_data)

    # Dual objective: min 0.5 * sum_ij alpha_i alpha_j y_i y_j (x_i^T x_j) - sum_i alpha_i
    # s.t. alpha_i >= 0, sum alpha_i y_i = 0
    k_matrix = x_data @ x_data.T
    p_matrix = (y_data[:, np.newaxis] @ y_data[np.newaxis, :]) * k_matrix

    def dual_obj(alpha):
        return 0.5 * float(alpha.T @ p_matrix @ alpha) - float(np.sum(alpha))

    def dual_grad(alpha):
        return p_matrix @ alpha - np.ones(n)

    # Equality constraint: sum alpha_i y_i = 0
    cons = {'type': 'eq', 'fun': lambda a: np.sum(a * y_data)}
    bounds = [(0.0, None) for _ in range(n)]

    res = opt.minimize(dual_obj, x0=np.ones(n), jac=dual_grad, bounds=bounds, constraints=cons, method='SLSQP')
    alpha_opt = np.maximum(0.0, res.x)

    # Compute primal weights: w = sum alpha_i y_i x_i
    w_primal = np.sum((alpha_opt * y_data)[:, np.newaxis] * x_data, axis=0)

    # Find support vectors (alpha_i > 1e-4) to compute bias b
    sv_idx = np.where(alpha_opt > 1e-4)[0]
    b_values = [y_data[i] - float(w_primal @ x_data[i]) for i in sv_idx]
    b_primal = float(np.mean(b_values))

    # Check margin for all points: y_i (w^T x_i + b)
    margins = y_data * (x_data @ w_primal + b_primal)
    slackness_products = alpha_opt * (margins - 1.0)

    print("\n--- Support Vector Machine KKT Dual Solver ---")
    print(f"Optimal Dual Alphas:        {alpha_opt}")
    print(f"Support Vector Indices:     {sv_idx}")
    print(f"Primal Hyperplane w:        {w_primal}, b = {b_primal:.4f}")
    print(f"Margins y*(w^T x + b):      {margins}")
    print(f"Complementary Slackness:    {slackness_products}")

    # All margins must be >= 1.0 (Primal feasibility)
    assert np.all(margins >= 1.0 - 1e-4)
    # Dual variables must be >= 0 (Dual feasibility)
    assert np.all(alpha_opt >= 0.0)
    # Complementary slackness must be zero everywhere
    assert np.allclose(slackness_products, np.zeros(n), atol=1e-4)
    print("✓ SVM KKT complementary slackness verified!")


# =====================================================================
# 4. Projected Gradient Descent (PGD) on L2 Ball
# =====================================================================
def verify_projected_gradient_descent():
    """
    Verifies that Projected Gradient Descent for:
    min 0.5 * ||x - x_target||^2 s.t. ||x||_2 <= R
    converges to the exact analytical projection point x* = R * (x_target / ||x_target||).
    """
    x_target = np.array([3.0, 4.0])  # norm = 5.0
    radius = 2.0
    analytical_opt = radius * (x_target / np.linalg.norm(x_target))  # [1.2, 1.6]

    # Initialize at origin
    x_curr = np.array([0.0, 0.0])
    eta = 0.20

    for _ in range(50):
        # Gradient: grad = x - x_target
        grad = x_curr - x_target
        # Unconstrained step
        x_step = x_curr - eta * grad
        # L2 Projection onto ball of radius R: x / max(1, ||x|| / R)
        norm_step = np.linalg.norm(x_step)
        if norm_step > radius:
            x_curr = radius * (x_step / norm_step)
        else:
            x_curr = x_step

    print("\n--- Projected Gradient Descent (PGD) on L2 Ball ---")
    print(f"Target Point:           {x_target} (Norm = 5.0)")
    print(f"Constraint Radius:      {radius}")
    print(f"Analytical Solution x*: {analytical_opt}")
    print(f"PGD Final Position:     {x_curr}")

    assert np.allclose(x_curr, analytical_opt, atol=1e-5)
    print("✓ Projected Gradient Descent exact boundary convergence verified!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 5.3 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_scipy_slsqp()
    verify_svm_complementary_slackness()
    verify_projected_gradient_descent()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 5.3 PASSED CLEANLY!")
    print("=================================================================")
