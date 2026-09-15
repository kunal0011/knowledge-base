"""
Chapter 2.1: Differential Calculus Foundations & Taylor Expansions
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Analytical vs. Numerical (Finite Difference) Gradients and Hessians.
2. Part 5 'AI by Hand' Numerical Verification (Polynomial Objective).
3. Empirical Taylor Convergence Rates (O(||h||^2) for 1st-order, O(||h||^3) for 2nd-order).
4. First-Order (GD) vs. Second-Order (Newton's Method) Optimization Step.
5. ReLU Non-Differentiability & Subgradient Supporting Hyperplane Check.
6. PyTorch Autograd Functional Hessian & Gradient Cross-Verification.
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Objective: Polynomial Function & Derivatives
# =====================================================================
def f_poly(x: np.ndarray) -> float:
    """
    f(x1, x2) = 2*x1^2 + x1*x2 + 3*x2^2 - 4*x1 - 2*x2 + 5
    """
    x1, x2 = x[0], x[1]
    return float(2 * x1**2 + x1 * x2 + 3 * x2**2 - 4 * x1 - 2 * x2 + 5)


def grad_poly(x: np.ndarray) -> np.ndarray:
    """
    nabla f(x) = [4*x1 + x2 - 4,  x1 + 6*x2 - 2]^T
    """
    x1, x2 = x[0], x[1]
    g1 = 4.0 * x1 + x2 - 4.0
    g2 = x1 + 6.0 * x2 - 2.0
    return np.array([g1, g2], dtype=np.float64)


def hessian_poly(x: np.ndarray) -> np.ndarray:
    """
    H = [[4, 1],
         [1, 6]]
    """
    return np.array([[4.0, 1.0], [1.0, 6.0]], dtype=np.float64)


# =====================================================================
# 2. General Finite Difference Estimators
# =====================================================================
def numerical_gradient(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes numerical gradient via central finite difference:
    df/dxi ~= (f(x + eps*ei) - f(x - eps*ei)) / (2*eps)
    """
    grad = np.zeros_like(x, dtype=np.float64)
    n = len(x)
    for i in range(n):
        e_i = np.zeros_like(x)
        e_i[i] = eps
        f_plus = func(x + e_i)
        f_minus = func(x - e_i)
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def numerical_hessian(func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Computes numerical Hessian via second-order central differences:
    d^2f / dxi dxj ~= (f(x + eps*ei + eps*ej) - f(x + eps*ei - eps*ej)
                     - f(x - eps*ei + eps*ej) + f(x - eps*ei - eps*ej)) / (4*eps^2)
    """
    n = len(x)
    H = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        e_i = np.zeros_like(x)
        e_i[i] = eps
        for j in range(n):
            e_j = np.zeros_like(x)
            e_j[j] = eps
            f_pp = func(x + e_i + e_j)
            f_pm = func(x + e_i - e_j)
            f_mp = func(x - e_i + e_j)
            f_mm = func(x - e_i - e_j)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
    return H


# =====================================================================
# 3. Non-Linear Test Function for Asymptotic Convergence
# =====================================================================
def f_exp(x: np.ndarray) -> float:
    """
    f(x1, x2) = exp(x1 + 2*x2)
    """
    return float(np.exp(x[0] + 2.0 * x[1]))


def grad_exp(x: np.ndarray) -> np.ndarray:
    val = np.exp(x[0] + 2.0 * x[1])
    return np.array([val, 2.0 * val], dtype=np.float64)


def hessian_exp(x: np.ndarray) -> np.ndarray:
    val = np.exp(x[0] + 2.0 * x[1])
    return np.array([[val, 2.0 * val], [2.0 * val, 4.0 * val]], dtype=np.float64)


def taylor_expansion_orders(func, grad_fn, hess_fn, x0: np.ndarray, h: np.ndarray):
    """
    Computes T0, T1, T2 approximations and the true value at x0 + h.
    """
    f0 = func(x0)
    g0 = grad_fn(x0)
    H0 = hess_fn(x0)

    t0 = f0
    t1 = f0 + np.dot(g0, h)
    t2 = f0 + np.dot(g0, h) + 0.5 * float(h.T @ H0 @ h)
    f_true = func(x0 + h)

    return t0, t1, t2, f_true


# =====================================================================
# 4. Optimization Step Comparison: GD vs Newton
# =====================================================================
def run_optimization_step_comparison():
    """
    Compares 1 step of Gradient Descent vs 1 step of Newton's method
    on an anisotropic quadratic bowl: f(x) = 0.5 * x^T A x - b^T x
    """
    A = np.array([[10.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    b = np.array([4.0, -1.0], dtype=np.float64)

    # Global analytical optimum: A x* = b => x* = A^-1 b
    x_opt = np.linalg.solve(A, b)

    x0 = np.array([2.0, -3.0], dtype=np.float64)

    def quad_loss(x):
        return 0.5 * float(x.T @ A @ x) - float(b.T @ x)

    # Gradient: nabla f(x) = A x - b
    g0 = A @ x0 - b
    # Hessian: H = A
    H0 = A

    # Optimal learning rate for GD: 2 / (lambda_max + lambda_min)
    eigenvalues = np.linalg.eigvalsh(A)
    eta = 2.0 / (np.max(eigenvalues) + np.min(eigenvalues))

    # 1 Step of Gradient Descent
    x_gd = x0 - eta * g0
    loss_gd = quad_loss(x_gd)

    # 1 Step of Newton's Method: Delta x = - H^-1 g0
    delta_newton = -np.linalg.solve(H0, g0)
    x_newton = x0 + delta_newton
    loss_newton = quad_loss(x_newton)

    return {
        "x_opt": x_opt,
        "x0": x0,
        "loss_0": quad_loss(x0),
        "x_gd": x_gd,
        "loss_gd": loss_gd,
        "x_newton": x_newton,
        "loss_newton": loss_newton,
        "newton_exact_match": np.allclose(x_newton, x_opt),
    }


# =====================================================================
# 5. ReLU Subgradient Supporting Hyperplane Verification
# =====================================================================
def verify_relu_subgradients():
    """
    Verifies that for any g in [0, 1], the supporting hyperplane
    y(x) = f(0) + g * (x - 0) = g * x lies completely below f(x) = max(0, x).
    """
    x_test = np.linspace(-3.0, 3.0, 500)
    f_relu = np.maximum(0.0, x_test)

    candidate_subgradients = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_valid = True

    for g in candidate_subgradients:
        plane = g * x_test
        # Supporting hyperplane condition: f(x) >= plane for all x
        is_supporting = np.all(f_relu >= plane - 1e-9)
        if not is_supporting:
            all_valid = False

    # Invalid subgradient outside [0, 1], e.g., g = 1.2 or g = -0.2
    invalid_g_high = 1.2
    invalid_g_low = -0.2
    assert not np.all(f_relu >= invalid_g_high * x_test - 1e-9)
    assert not np.all(f_relu >= invalid_g_low * x_test - 1e-9)

    return all_valid


# =====================================================================
# 6. PyTorch Functional Cross-Verification
# =====================================================================
def pytorch_autograd_cross_check():
    """
    Cross-verifies hand-derived gradient and Hessian against PyTorch's
    torch.autograd.functional.
    """
    def torch_poly(x):
        return 2.0 * x[0] ** 2 + x[0] * x[1] + 3.0 * x[1] ** 2 - 4.0 * x[0] - 2.0 * x[1] + 5.0

    x_torch = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

    # PyTorch gradient
    grad_torch = torch.autograd.functional.jacobian(torch_poly, x_torch)
    # PyTorch Hessian
    hess_torch = torch.autograd.functional.hessian(torch_poly, x_torch)

    expected_grad = grad_poly(np.array([1.0, 2.0]))
    expected_hess = hessian_poly(np.array([1.0, 2.0]))

    grad_match = np.allclose(grad_torch.detach().numpy(), expected_grad)
    hess_match = np.allclose(hess_torch.detach().numpy(), expected_hess)

    return grad_match and hess_match


# =====================================================================
# Main Test Suite Execution
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 2.1: Differential Calculus & Taylor Expansions — Tests")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Exact Numerical Verification
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Numerical Walkthrough]")
    x0 = np.array([1.0, 2.0], dtype=np.float64)
    h = np.array([0.1, -0.2], dtype=np.float64)

    t0, t1, t2, f_true = taylor_expansion_orders(f_poly, grad_poly, hessian_poly, x0, h)

    print(f"  Expansion point x0: {x0}")
    print(f"  Displacement h:     {h}")
    print(f"  Target point x0+h:  {x0 + h}")
    print(f"  T0 (0th order):     {t0:.6f}  (Expected: 13.000000)")
    print(f"  T1 (1st order):     {t1:.6f}  (Expected: 11.000000)")
    print(f"  T2 (2nd order):     {t2:.6f}  (Expected: 11.120000)")
    print(f"  True value f(x0+h): {f_true:.6f}  (Expected: 11.120000)")

    assert np.isclose(t0, 13.0), f"T0 mismatch: {t0}"
    assert np.isclose(t1, 11.0), f"T1 mismatch: {t1}"
    assert np.isclose(t2, 11.12), f"T2 mismatch: {t2}"
    assert np.isclose(f_true, 11.12), f"True value mismatch: {f_true}"
    assert np.isclose(t2, f_true), "For quadratic polynomials, T2 must be EXACT!"
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: Numerical vs Analytical Derivatives
    # -------------------------------------------------------------
    print("\n[Test 2: Finite Difference vs Analytical Derivatives]")
    num_g = numerical_gradient(f_poly, x0)
    ana_g = grad_poly(x0)
    print(f"  Analytical Gradient: {ana_g}")
    print(f"  Numerical Gradient:  {num_g}")
    assert np.allclose(num_g, ana_g, atol=1e-5), "Gradient mismatch!"

    num_H = numerical_hessian(f_poly, x0)
    ana_H = hessian_poly(x0)
    print(f"  Analytical Hessian:\n{ana_H}")
    print(f"  Numerical Hessian:\n{num_H}")
    assert np.allclose(num_H, ana_H, atol=1e-4), "Hessian mismatch!"
    print("  -> Gradient and Hessian verified via central differences!")

    # -------------------------------------------------------------
    # Test 3: Asymptotic Taylor Convergence Rates on Non-Linear Function
    # -------------------------------------------------------------
    print("\n[Test 3: Asymptotic Error Scaling (Non-Linear Exponential Surface)]")
    x0_exp = np.array([0.0, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)  # unit vector

    step_scales = [1e-1, 1e-2, 1e-3, 1e-4]
    errors_1st = []
    errors_2nd = []

    print(f"  {'Step Norm ||h||':<18} | {'1st-Order Error':<18} | {'2nd-Order Error':<18} | {'Error Ratio (1st/2nd)'}")
    print("  " + "-" * 75)
    for s in step_scales:
        h_vec = s * direction
        _, t1_val, t2_val, val_true = taylor_expansion_orders(
            f_exp, grad_exp, hessian_exp, x0_exp, h_vec
        )
        err1 = abs(val_true - t1_val)
        err2 = abs(val_true - t2_val)
        errors_1st.append(err1)
        errors_2nd.append(err2)
        ratio = err1 / (err2 + 1e-15)
        print(f"  {s:<18.1e} | {err1:<18.6e} | {err2:<18.6e} | {ratio:.2f}x")

    # Compute empirical order of convergence: slope = Delta log(err) / Delta log(s)
    log_s = np.log(step_scales)
    slope_1st, _ = np.polyfit(log_s, np.log(errors_1st), 1)
    slope_2nd, _ = np.polyfit(log_s, np.log(errors_2nd), 1)

    print(f"\n  Empirical Convergence Rate (1st-Order Taylor): O(||h||^{slope_1st:.2f})  [Expected: ~2.00]")
    print(f"  Empirical Convergence Rate (2nd-Order Taylor): O(||h||^{slope_2nd:.2f})  [Expected: ~3.00]")
    assert np.isclose(slope_1st, 2.0, atol=0.1), f"Unexpected 1st-order slope: {slope_1st}"
    assert np.isclose(slope_2nd, 3.0, atol=0.1), f"Unexpected 2nd-order slope: {slope_2nd}"
    print("  -> Taylor remainder asymptotic rates mathematically confirmed!")

    # -------------------------------------------------------------
    # Test 4: Gradient Descent vs. Newton's Method
    # -------------------------------------------------------------
    print("\n[Test 4: Optimization Mechanics — GD vs Newton's Method]")
    opt_res = run_optimization_step_comparison()
    print(f"  True Global Minimum x*:   {opt_res['x_opt']}")
    print(f"  Starting Point x0:        {opt_res['x0']} (Loss: {opt_res['loss_0']:.4f})")
    print(f"  After 1 GD Step:          {opt_res['x_gd']} (Loss: {opt_res['loss_gd']:.4f})")
    print(f"  After 1 Newton Step:      {opt_res['x_newton']} (Loss: {opt_res['loss_newton']:.4f})")
    print(f"  Newton Exact Match to x*? {opt_res['newton_exact_match']}")
    assert opt_res["newton_exact_match"], "Newton's method must solve quadratic loss in 1 step!"
    assert opt_res["loss_newton"] < opt_res["loss_gd"], "Newton step must outperform 1 GD step!"
    print("  -> Newton 1-step exact convergence confirmed!")

    # -------------------------------------------------------------
    # Test 5: ReLU Subgradient Supporting Condition
    # -------------------------------------------------------------
    print("\n[Test 5: ReLU Subdifferential & Supporting Hyperplanes]")
    relu_subgrad_valid = verify_relu_subgradients()
    print(f"  All g in [0, 1] satisfy subgradient definition? {relu_subgrad_valid}")
    assert relu_subgrad_valid, "ReLU subgradient condition failed!"
    print("  -> Subgradient supporting property confirmed!")

    # -------------------------------------------------------------
    # Test 6: PyTorch Autograd Functional Cross-Check
    # -------------------------------------------------------------
    print("\n[Test 6: PyTorch Autograd Functional Cross-Verification]")
    torch_passed = pytorch_autograd_cross_check()
    print(f"  PyTorch Autograd Jacobian & Hessian match analytical? {torch_passed}")
    assert torch_passed, "PyTorch cross-check failed!"
    print("  -> Analytical derivatives match PyTorch autograd exactly!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 2.1 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
