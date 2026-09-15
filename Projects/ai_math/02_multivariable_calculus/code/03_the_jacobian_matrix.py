"""
Chapter 2.3: The Jacobian Matrix
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' 2D Non-Linear Layer Jacobian & Taylor Walkthrough.
2. Numerical Finite Difference vs. Analytical Jacobian Matrix.
3. The Softmax Layer Jacobian: Diag(s) - s s^T, Symmetry, Zero-Sum Rows.
4. PyTorch Autograd Functional Jacobian Cross-Verification.
5. Vector-Jacobian Product (VJP) vs. Jacobian-Vector Product (JVP).
6. Polar-to-Cartesian Coordinate Transformation Determinant det(J) = r.
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Objective Function & Analytical Jacobian
# =====================================================================
def f_nonlinear(x: np.ndarray) -> np.ndarray:
    """
    f1(x1, x2) = x1^2 + 2*x1*x2
    f2(x1, x2) = 3*x1*x2^2 - x2
    """
    x1, x2 = x[0], x[1]
    y1 = x1**2 + 2.0 * x1 * x2
    y2 = 3.0 * x1 * x2**2 - x2
    return np.array([y1, y2], dtype=np.float64)


def jacobian_nonlinear(x: np.ndarray) -> np.ndarray:
    """
    J = [[2*x1 + 2*x2,     2*x1     ],
         [3*x2^2,          6*x1*x2 - 1]]
    """
    x1, x2 = x[0], x[1]
    j11 = 2.0 * x1 + 2.0 * x2
    j12 = 2.0 * x1
    j21 = 3.0 * x2**2
    j22 = 6.0 * x1 * x2 - 1.0
    return np.array([[j11, j12], [j21, j22]], dtype=np.float64)


# =====================================================================
# 2. General Numerical Finite Difference Jacobian Estimator
# =====================================================================
def numerical_jacobian(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes numerical Jacobian via central difference:
    J[:, j] ~= (func(x + eps*e_j) - func(x - eps*e_j)) / (2*eps)
    """
    n = len(x)
    y0 = func(x)
    m = len(y0)
    J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        e_j = np.zeros_like(x)
        e_j[j] = eps
        f_plus = func(x + e_j)
        f_minus = func(x - e_j)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


# =====================================================================
# 3. Softmax Function & Analytical Jacobian
# =====================================================================
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax: s_i = exp(z_i - max(z)) / sum(exp(z - max(z)))
    """
    shift_z = z - np.max(z)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z)


def jacobian_softmax(z: np.ndarray) -> np.ndarray:
    """
    J = Diag(s) - s s^T
    """
    s = softmax(z)
    diag_s = np.diag(s)
    outer_s = np.outer(s, s)
    return diag_s - outer_s


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 2.3: The Jacobian Matrix — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Numerical Verification
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' 2D Non-Linear Layer]")
    x0 = np.array([1.0, 2.0], dtype=np.float64)
    h = np.array([0.01, -0.02], dtype=np.float64)

    y0 = f_nonlinear(x0)
    J0 = jacobian_nonlinear(x0)
    det_J0 = np.linalg.det(J0)
    delta_y_lin = J0 @ h
    y_pred = y0 + delta_y_lin
    y_true = f_nonlinear(x0 + h)

    print(f"  Input Base Point x0:        {x0}")
    print(f"  Output Base Vector y0:      {y0} (Expected: [5.0, 10.0])")
    print(f"  Jacobian Matrix J(x0):\n{J0}")
    print(f"  det(J(x0)):                 {det_J0:.4f} (Expected: 42.0)")
    print(f"  Linear Shift J @ h:         {delta_y_lin} (Expected: [0.02, -0.10])")
    print(f"  Predicted Output y_pred:    {y_pred} (Expected: [5.02, 9.90])")
    print(f"  True Perturbed Output:      {y_true} (Expected: [5.0197, 9.898812])")

    assert np.allclose(y0, [5.0, 10.0]), f"y0 mismatch: {y0}"
    assert np.allclose(J0, [[6.0, 2.0], [12.0, 11.0]]), f"J0 mismatch: {J0}"
    assert np.isclose(det_J0, 42.0), f"det(J) mismatch: {det_J0}"
    assert np.allclose(delta_y_lin, [0.02, -0.10]), f"delta_y mismatch: {delta_y_lin}"
    assert np.allclose(y_pred, [5.02, 9.90]), f"y_pred mismatch: {y_pred}"
    assert np.isclose(y_true[0], 5.0197), f"y_true[0] mismatch: {y_true[0]}"
    assert np.isclose(y_true[1], 9.898812), f"y_true[1] mismatch: {y_true[1]}"
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: Numerical Finite-Difference vs Analytical Jacobian
    # -------------------------------------------------------------
    print("\n[Test 2: Numerical Finite Difference vs Analytical Jacobian]")
    J_num = numerical_jacobian(f_nonlinear, x0)
    print(f"  Analytical Jacobian:\n{J0}")
    print(f"  Numerical Jacobian:\n{J_num}")
    assert np.allclose(J_num, J0, atol=1e-5), "Numerical Jacobian mismatch!"
    print("  -> Numerical central-difference Jacobian matches analytical J!")

    # -------------------------------------------------------------
    # Test 3: The Softmax Layer Jacobian & Structural Properties
    # -------------------------------------------------------------
    print("\n[Test 3: Softmax Layer Jacobian Properties]")
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    probs = softmax(logits)
    J_soft = jacobian_softmax(logits)

    print(f"  Logits z:                   {logits}")
    print(f"  Probabilities s:            {probs}")
    print(f"  Softmax Jacobian J:\n{J_soft}")

    # Property 1: Symmetry J == J^T
    is_symmetric = np.allclose(J_soft, J_soft.T)
    print(f"  Is Softmax Jacobian symmetric? {is_symmetric}")
    assert is_symmetric, "Softmax Jacobian must be symmetric!"

    # Property 2: Row and Column sums equal zero (J @ 1 = 0)
    row_sums = np.sum(J_soft, axis=1)
    col_sums = np.sum(J_soft, axis=0)
    print(f"  Row sums of J:              {row_sums}")
    print(f"  Col sums of J:              {col_sums}")
    assert np.allclose(row_sums, 0.0, atol=1e-12), "Row sums must be zero!"
    assert np.allclose(col_sums, 0.0, atol=1e-12), "Col sums must be zero!"

    # Numerical verification of Softmax Jacobian
    J_soft_num = numerical_jacobian(softmax, logits)
    assert np.allclose(J_soft, J_soft_num, atol=1e-5), "Softmax numerical mismatch!"
    print("  -> Softmax Jacobian Diag(s) - s s^T and zero-sum verified!")

    # -------------------------------------------------------------
    # Test 4: PyTorch Functional Jacobian Cross-Check
    # -------------------------------------------------------------
    print("\n[Test 4: PyTorch Autograd Functional Cross-Check]")
    def torch_nonlinear(x):
        y1 = x[0] ** 2 + 2.0 * x[0] * x[1]
        y2 = 3.0 * x[0] * x[1] ** 2 - x[1]
        return torch.stack([y1, y2])

    x_torch = torch.tensor([1.0, 2.0], dtype=torch.float64)
    J_torch = torch.autograd.functional.jacobian(torch_nonlinear, x_torch)

    print(f"  PyTorch Jacobian:\n{J_torch.numpy()}")
    assert np.allclose(J_torch.numpy(), J0), "PyTorch Jacobian mismatch!"

    # Softmax PyTorch check
    z_torch = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    def torch_soft(z):
        return torch.softmax(z, dim=-1)

    J_soft_torch = torch.autograd.functional.jacobian(torch_soft, z_torch)
    assert np.allclose(J_soft_torch.numpy(), J_soft), "PyTorch Softmax mismatch!"
    print("  -> PyTorch Autograd functional Jacobian matches analytical exactly!")

    # -------------------------------------------------------------
    # Test 5: Vector-Jacobian Product (VJP) vs Jacobian-Vector Product (JVP)
    # -------------------------------------------------------------
    print("\n[Test 5: VJP (Reverse-Mode) and JVP (Forward-Mode) Invariance]")
    v_upstream = np.array([0.5, -1.5], dtype=np.float64)  # 1 x 2 upstream gradient
    u_direction = np.array([0.2, 0.8], dtype=np.float64)  # 2 x 1 input direction

    # JVP: J @ u
    jvp = J0 @ u_direction
    # VJP: v @ J
    vjp = v_upstream @ J0

    # Adjoint Identity: <v, J u> == <v J, u>
    left_side = np.dot(v_upstream, jvp)
    right_side = np.dot(vjp, u_direction)
    print(f"  Left Side  <v, J u>:       {left_side:.6f}")
    print(f"  Right Side <v J, u>:       {right_side:.6f}")
    assert np.isclose(left_side, right_side), "Adjoint adjointness identity violated!"
    print("  -> Adjoint identity <v, J u> = <v J, u> verified!")

    # -------------------------------------------------------------
    # Test 6: Polar-to-Cartesian Transformation Determinant
    # -------------------------------------------------------------
    print("\n[Test 6: Polar Transformation det(J) = r]")
    for r_test in [1.5, 3.2, 7.8]:
        theta_test = np.random.uniform(0, 2 * np.pi)
        # J_polar = [[cos(theta), -r*sin(theta)], [sin(theta), r*cos(theta)]]
        J_polar = np.array([
            [np.cos(theta_test), -r_test * np.sin(theta_test)],
            [np.sin(theta_test), r_test * np.cos(theta_test)],
        ])
        det_polar = np.linalg.det(J_polar)
        assert np.isclose(det_polar, r_test), f"Polar det mismatch: {det_polar} vs {r_test}"

    print("  -> Polar coordinate transformation det(J) = r confirmed across sweeps!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 2.3 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
