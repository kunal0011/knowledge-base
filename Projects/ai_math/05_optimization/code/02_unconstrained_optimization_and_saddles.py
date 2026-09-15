"""
Chapter 5.2: Unconstrained Optimization, Stationary Points & Saddle Points
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Grid & Saddle Escape Verification.
2. Perturbed Gradient Descent Saddle Point Escape Simulation.
3. Rosenbrock Banana Function Condition Number & Ill-Conditioned Ravine.
4. Monkey Saddle Degeneracy & Higher-Order Directional Curvature.
5. Sharpness-Aware Minimization (SAM) Flat Minima Preference in PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Visual Grid Verification
# =====================================================================
def f_toy(x, y):
    return x**2 - y**2 + 0.25 * (y**4)


def grad_f_toy(x, y):
    return np.array([2.0 * x, -2.0 * y + y**3])


def hessian_f_toy(x, y):
    return np.array([
        [2.0, 0.0],
        [0.0, -2.0 + 3.0 * (y**2)]
    ])


def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 5.2 Part 5:
    f(x, y) = x^2 - y^2 + 0.25 y^4
    """
    # 1. Critical points classification
    crit_pts = {
        'A': (0.0, 0.0),
        'B': (0.0, math.sqrt(2.0)),
        'C': (0.0, -math.sqrt(2.0))
    }

    print("--- Part 5: Critical Points Classification ---")
    for name, (cx, cy) in crit_pts.items():
        grad = grad_f_toy(cx, cy)
        h = hessian_f_toy(cx, cy)
        eigvals = np.linalg.eigvalsh(h)
        val = f_toy(cx, cy)
        print(f"Point {name}: ({cx:.4f}, {cy:.4f}) | f = {val:.4f} | Eigvals: {eigvals}")
        assert np.allclose(grad, [0.0, 0.0], atol=1e-7)

    # Point A is saddle (+2, -2)
    eig_a = np.linalg.eigvalsh(hessian_f_toy(0.0, 0.0))
    assert eig_a[0] < 0 < eig_a[1]
    # Points B and C are local minima (+2, +4)
    eig_b = np.linalg.eigvalsh(hessian_f_toy(0.0, math.sqrt(2.0)))
    assert np.all(eig_b > 0)
    assert abs(f_toy(0.0, math.sqrt(2.0)) - (-1.0)) < 1e-7

    # 2. Trace 3 Gradient Descent steps from (0.20, 0.10) with eta = 0.10
    eta = 0.10
    x_t, y_t = 0.20, 0.10
    expected_x = [0.2000, 0.1600, 0.1280, 0.1024]
    expected_y = [0.1000, 0.1199, 0.1437, 0.1721]

    print("\n--- Part 5: Gradient Descent Trajectory Escaping Saddle ---")
    for step in range(4):
        val = f_toy(x_t, y_t)
        grad = grad_f_toy(x_t, y_t)
        print(f"Iter {step:1d} | x = {x_t:.4f} (Exp: {expected_x[step]:.4f}) | y = {y_t:.4f} (Exp: {expected_y[step]:.4f}) | Loss = {val:.4f}")
        assert abs(x_t - expected_x[step]) < 1e-3
        assert abs(y_t - expected_y[step]) < 1e-3

        # Update
        x_t = x_t - eta * grad[0]
        y_t = y_t - eta * grad[1]

    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Perturbed Gradient Descent Saddle Escape Simulation
# =====================================================================
def verify_saddle_escape():
    """
    Shows that vanilla GD initialized at (0, 0) is frozen, but adding
    a tiny perturbation allows it to escape along the negative curvature direction.
    """
    eta = 0.05
    n_iters = 150

    # Case 1: Unperturbed vanilla GD at exact saddle
    pos_unperturbed = np.array([0.0, 0.0])
    for _ in range(n_iters):
        g = grad_f_toy(pos_unperturbed[0], pos_unperturbed[1])
        pos_unperturbed -= eta * g

    assert np.allclose(pos_unperturbed, [0.0, 0.0])  # Still stuck at (0, 0)!

    # Case 2: Perturbed GD with random perturbation 1e-4
    np.random.seed(42)
    pos_perturbed = np.array([0.0, 0.0]) + np.random.normal(0.0, 1e-4, size=2)
    for _ in range(n_iters):
        g = grad_f_toy(pos_perturbed[0], pos_perturbed[1])
        pos_perturbed -= eta * g

    final_loss = f_toy(pos_perturbed[0], pos_perturbed[1])
    print("\n--- Saddle Escape Simulation ---")
    print(f"Unperturbed GD final pos: {pos_unperturbed} | Loss = {f_toy(pos_unperturbed[0], pos_unperturbed[1]):.4f} (Stuck at saddle)")
    print(f"Perturbed GD final pos:   {pos_perturbed} | Loss = {final_loss:.4f} (Escaped to minimum!)")

    assert abs(final_loss - (-1.0)) < 0.01  # Successfully reached global minimum f = -1
    print("✓ Perturbed gradient descent saddle point escape confirmed!")


# =====================================================================
# 3. Rosenbrock Banana Function Condition Number Analysis
# =====================================================================
def rosenbrock(x, y):
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2


def verify_rosenbrock_conditioning():
    """
    Computes the Hessian of the Rosenbrock function at the minimum (1, 1)
    and verifies the extreme condition number kappa ~ 2500.
    """
    # At (1, 1):
    # d^2 f / dx^2 = 2 - 400 y + 1200 x^2 = 2 - 400 + 1200 = 802
    # d^2 f / dx dy = -400 x = -400
    # d^2 f / dy^2 = 200
    h = np.array([
        [802.0, -400.0],
        [-400.0, 200.0]
    ])

    eigvals = np.linalg.eigvalsh(h)
    lambda_min = eigvals[0]
    lambda_max = eigvals[1]
    kappa = lambda_max / lambda_min

    print("\n--- Rosenbrock Banana Function Conditioning ---")
    print(f"Hessian Matrix at (1, 1):\n{h}")
    print(f"Eigenvalues: lambda_min = {lambda_min:.4f}, lambda_max = {lambda_max:.4f}")
    print(f"Condition Number kappa = L / mu: {kappa:.2f}")

    assert abs(lambda_min - 0.399) < 0.05
    assert abs(lambda_max - 1001.60) < 0.05
    assert kappa > 2000.0
    print("✓ Rosenbrock extreme condition number verified!")


# =====================================================================
# 4. Monkey Saddle Degeneracy & Higher-Order Directional Curvature
# =====================================================================
def monkey_saddle(x, y):
    return x**3 - 3.0 * x * (y**2)


def verify_monkey_saddle():
    """
    Verifies that at (0, 0), the gradient and Hessian are both zero,
    and tests the 3-fold cyclic ascending/descending structure.
    """
    # At (0, 0)
    # grad = [3x^2 - 3y^2, -6xy] = [0, 0]
    # hess = [[6x, -6y], [-6y, -6x]] = [[0, 0], [0, 0]]
    r = 1.0
    # Sample along 6 angles: theta = k * pi / 3
    # f(r, theta) = r^3 * cos(3 theta)
    angles = np.array([0, 1, 2, 3, 4, 5]) * (math.pi / 3.0)
    values = []
    for th in angles:
        x = r * math.cos(th)
        y = r * math.sin(th)
        val = monkey_saddle(x, y)
        values.append(val)

    print("\n--- Monkey Saddle Degeneracy Test ---")
    print(f"Angles k * pi/3: {angles}")
    print(f"Function values: {values}")

    # Should alternate +1, -1, +1, -1, +1, -1 (3 peaks, 3 valleys)
    expected = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert np.allclose(values, expected, atol=1e-7)
    print("✓ Monkey saddle 3-valley higher-order structure confirmed!")


# =====================================================================
# 5. Sharpness-Aware Minimization (SAM) Flat Minima Preference
# =====================================================================
def verify_sam_flat_minima():
    """
    Demonstrates that SAM avoids sharp minima by evaluating loss
    at the worst-case epsilon-perturbed neighbor:
    eps* = rho * grad / ||grad||_2.
    """
    torch.manual_seed(42)

    # 1D loss landscape with a sharp minimum at x = 1 and a flat minimum at x = -2:
    # Sharp: f_sharp(x) = 50 * (x - 1)^2
    # Flat:  f_flat(x) = 0.5 * (x + 2)^2
    def loss_landscape(w):
        return torch.where(w > 0, 50.0 * (w - 1.0)**2, 0.5 * (w + 2.0)**2)

    rho = 0.20  # SAM perturbation radius

    # Evaluate sharpness at x = 1.0 vs x = -2.0
    w_sharp = torch.tensor([1.0], requires_grad=True)
    w_flat = torch.tensor([-2.0], requires_grad=True)

    # Gradient of loss w.r.t w + eps
    # In neighborhood w in [1 - rho, 1 + rho], max loss is 50 * (0.2)^2 = 2.0
    # In neighborhood w in [-2 - rho, -2 + rho], max loss is 0.5 * (0.2)^2 = 0.02
    loss_sharp_worst = loss_landscape(w_sharp + rho)
    loss_flat_worst = loss_landscape(w_flat + rho)

    print("\n--- Sharpness-Aware Minimization (SAM) ---")
    print(f"Sharp Minimum Nominal Loss at x=1.0:  {loss_landscape(w_sharp).item():.4f}")
    print(f"Flat Minimum Nominal Loss at x=-2.0: {loss_landscape(w_flat).item():.4f}")
    print(f"Sharp Worst-Case SAM Loss (rho=0.2): {loss_sharp_worst.item():.4f}")
    print(f"Flat Worst-Case SAM Loss (rho=0.2):  {loss_flat_worst.item():.4f}")

    # SAM strongly penalizes the sharp minimum (100x higher perturbed loss)
    assert loss_sharp_worst.item() > 50.0 * loss_flat_worst.item()
    print("✓ SAM flatness preference strictly verified!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 5.2 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_saddle_escape()
    verify_rosenbrock_conditioning()
    verify_monkey_saddle()
    verify_sam_flat_minima()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 5.2 PASSED CLEANLY!")
    print("=================================================================")
