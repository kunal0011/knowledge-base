"""
Chapter 2.4: The Hessian Matrix & Curvature Analysis
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Curvature Analysis (Point A Positive Definite vs. Point B Saddle).
2. Clairaut's Theorem Symmetry Verification across Random Points.
3. Pearlmutter's Trick: Matrix-Free Hessian-Vector Products (HVP).
4. Ravine Condition Number & Critical Learning Rate Threshold Divergence.
5. PyTorch Autograd Functional Hessian & HVP Cross-Verification.
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Objective Function, Gradient & Hessian
# =====================================================================
def f_cubic(x: np.ndarray) -> float:
    """
    f(x1, x2) = x1^3 - 3*x1 + 2*x1*x2 + 2*x2^2
    """
    x1, x2 = x[0], x[1]
    return float(x1**3 - 3.0 * x1 + 2.0 * x1 * x2 + 2.0 * x2**2)


def grad_cubic(x: np.ndarray) -> np.ndarray:
    """
    nabla f = [3*x1^2 - 3 + 2*x2,  2*x1 + 4*x2]^T
    """
    x1, x2 = x[0], x[1]
    g1 = 3.0 * x1**2 - 3.0 + 2.0 * x2
    g2 = 2.0 * x1 + 4.0 * x2
    return np.array([g1, g2], dtype=np.float64)


def hessian_cubic(x: np.ndarray) -> np.ndarray:
    """
    H = [[6*x1,  2],
         [2,     4]]
    """
    x1 = x[0]
    return np.array([[6.0 * x1, 2.0], [2.0, 4.0]], dtype=np.float64)


# =====================================================================
# 2. Pearlmutter's Matrix-Free Hessian-Vector Product (HVP)
# =====================================================================
def numerical_hvp(grad_fn, x: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes H(x) * v via central difference of the gradient:
    H(x) * v ~= (nabla f(x + eps*v) - nabla f(x - eps*v)) / (2*eps)
    """
    g_plus = grad_fn(x + eps * v)
    g_minus = grad_fn(x - eps * v)
    return (g_plus - g_minus) / (2.0 * eps)


# =====================================================================
# 3. Ravine Divergence Simulation
# =====================================================================
def simulate_ravine_step(x_init: np.ndarray, lr: float, steps: int = 5):
    """
    Simulates gradient descent on f(x1, x2) = 50*x1^2 + 0.5*x2^2
    H = diag([100, 1]), lambda_max = 100 => eta_max = 2/100 = 0.02
    """
    x = x_init.copy()
    trajectory = [x.copy()]
    for _ in range(steps):
        # grad = [100*x1, 1*x2]
        grad = np.array([100.0 * x[0], 1.0 * x[1]], dtype=np.float64)
        x = x - lr * grad
        trajectory.append(x.copy())
    return trajectory


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 2.4: The Hessian Matrix & Curvature Analysis — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Point A (Positive Definite Bowl)
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 Point A — Upward Parabolic Bowl Analysis]")
    xA = np.array([1.0, 2.0], dtype=np.float64)
    gA = grad_cubic(xA)
    HA = hessian_cubic(xA)
    trA = float(np.trace(HA))
    detA = float(np.linalg.det(HA))
    eigvalsA = np.linalg.eigvalsh(HA)  # Sorted ascending
    lambda1_A = eigvalsA[1]  # larger
    lambda2_A = eigvalsA[0]  # smaller
    kappaA = lambda1_A / lambda2_A

    print(f"  Point A Coordinates:       {xA}")
    print(f"  Gradient nabla f(xA):      {gA} (Expected: [4.0, 10.0])")
    print(f"  Hessian Matrix H(xA):\n{HA}")
    print(f"  Trace Tr(H):               {trA:.4f} (Expected: 10.0)")
    print(f"  Determinant det(H):        {detA:.4f} (Expected: 20.0)")
    print(f"  lambda_1 (Larger):         {lambda1_A:.6f} (Expected: ~7.236068)")
    print(f"  lambda_2 (Smaller):        {lambda2_A:.6f} (Expected: ~2.763932)")
    print(f"  Condition Number kappa:    {kappaA:.6f} (Expected: ~2.618034)")

    assert np.allclose(gA, [4.0, 10.0]), f"Gradient mismatch: {gA}"
    assert np.allclose(HA, [[6.0, 2.0], [2.0, 4.0]]), f"Hessian mismatch: {HA}"
    assert np.isclose(trA, 10.0), f"Trace mismatch: {trA}"
    assert np.isclose(detA, 20.0), f"Determinant mismatch: {detA}"
    assert np.isclose(lambda1_A, 5.0 + np.sqrt(5.0)), f"lambda1 mismatch: {lambda1_A}"
    assert np.isclose(lambda2_A, 5.0 - np.sqrt(5.0)), f"lambda2 mismatch: {lambda2_A}"
    assert np.isclose(kappaA, (3.0 + np.sqrt(5.0)) / 2.0), f"kappa mismatch: {kappaA}"
    assert np.all(eigvalsA > 0), "Point A must be strictly Positive Definite!"
    print("  -> Point A exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: Part 5 "AI by Hand" Point B (Saddle Point)
    # -------------------------------------------------------------
    print("\n[Test 2: Part 5 Point B — Hyperbolic Saddle Analysis]")
    xB = np.array([-1.0, 2.0], dtype=np.float64)
    gB = grad_cubic(xB)
    HB = hessian_cubic(xB)
    trB = float(np.trace(HB))
    detB = float(np.linalg.det(HB))
    eigvalsB = np.linalg.eigvalsh(HB)
    lambda1_B = eigvalsB[1]  # positive
    lambda2_B = eigvalsB[0]  # negative

    print(f"  Point B Coordinates:       {xB}")
    print(f"  Gradient nabla f(xB):      {gB} (Expected: [4.0, 6.0])")
    print(f"  Hessian Matrix H(xB):\n{HB}")
    print(f"  Trace Tr(H):               {trB:.4f} (Expected: -2.0)")
    print(f"  Determinant det(H):        {detB:.4f} (Expected: -28.0)")
    print(f"  lambda_1 (Positive):       {lambda1_B:.6f} (Expected: ~+4.385165)")
    print(f"  lambda_2 (Negative):       {lambda2_B:.6f} (Expected: ~-6.385165)")

    assert np.allclose(gB, [4.0, 6.0]), f"Gradient mismatch: {gB}"
    assert np.allclose(HB, [[-6.0, 2.0], [2.0, 4.0]]), f"Hessian mismatch: {HB}"
    assert np.isclose(trB, -2.0), f"Trace mismatch: {trB}"
    assert np.isclose(detB, -28.0), f"Determinant mismatch: {detB}"
    assert np.isclose(lambda1_B, -1.0 + np.sqrt(29.0)), f"lambda1 mismatch: {lambda1_B}"
    assert np.isclose(lambda2_B, -1.0 - np.sqrt(29.0)), f"lambda2 mismatch: {lambda2_B}"
    assert lambda1_B > 0 and lambda2_B < 0, "Point B must have mixed eigenvalue signs (Saddle)!"
    print("  -> Point B saddle geometry MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 3: Clairaut's Symmetry across 50 Random Evaluation Points
    # -------------------------------------------------------------
    print("\n[Test 3: Clairaut's Symmetry H == H^T Check]")
    np.random.seed(123)
    max_asymm = 0.0
    for _ in range(50):
        x_rand = np.random.uniform(-10.0, 10.0, size=2)
        H_rand = hessian_cubic(x_rand)
        asymm = np.max(np.abs(H_rand - H_rand.T))
        if asymm > max_asymm:
            max_asymm = asymm

    print(f"  Max Asymmetry ||H - H^T|| across 50 points: {max_asymm:.6e}")
    assert np.isclose(max_asymm, 0.0, atol=1e-15), "Hessian is not symmetric!"
    print("  -> Clairaut's Theorem verified: Hessian is symmetric everywhere!")

    # -------------------------------------------------------------
    # Test 4: Pearlmutter's HVP (Hessian-Vector Product) Trick
    # -------------------------------------------------------------
    print("\n[Test 4: Pearlmutter's Matrix-Free HVP Verification]")
    v_test = np.array([3.0, -2.0], dtype=np.float64)
    # Direct matrix multiplication
    hvp_direct = HA @ v_test
    # Matrix-free finite-difference directional derivative of gradient
    hvp_matrix_free = numerical_hvp(grad_cubic, xA, v_test, eps=1e-6)

    print(f"  Direction Vector v:        {v_test}")
    print(f"  Direct H(xA) @ v:          {hvp_direct}")
    print(f"  Matrix-Free Pearlmutter:   {hvp_matrix_free}")
    err_hvp = np.max(np.abs(hvp_direct - hvp_matrix_free))
    print(f"  HVP Approximation Error:   {err_hvp:.6e}")
    assert err_hvp < 1e-5, f"HVP error too large: {err_hvp}"
    print("  -> Pearlmutter's trick matches direct matrix multiplication!")

    # -------------------------------------------------------------
    # Test 5: Ravine Condition Number & Step Size Divergence Threshold
    # -------------------------------------------------------------
    print("\n[Test 5: Anisotropic Ravine Divergence Threshold]")
    # lambda_max = 100 => 2/lambda_max = 0.02
    x_start = np.array([1.0, 1.0], dtype=np.float64)

    # Subcritical learning rate: eta = 0.015 (< 0.02) -> Converges
    traj_stable = simulate_ravine_step(x_start, lr=0.015, steps=5)
    # Supercritical learning rate: eta = 0.025 (> 0.02) -> Diverges / Explodes
    traj_unstable = simulate_ravine_step(x_start, lr=0.025, steps=5)

    print(f"  Stable Trajectory (eta = 0.015 < 0.02):")
    for step, pt in enumerate(traj_stable):
        print(f"    Step {step}: x1 = {pt[0]:+10.6f}, x2 = {pt[1]:+10.6f}")

    print(f"  Unstable Trajectory (eta = 0.025 > 0.02):")
    for step, pt in enumerate(traj_unstable):
        print(f"    Step {step}: x1 = {pt[0]:+10.6f}, x2 = {pt[1]:+10.6f}")

    assert abs(traj_stable[-1][0]) < abs(x_start[0]), "Stable run should shrink x1!"
    assert abs(traj_unstable[-1][0]) > abs(x_start[0]), "Unstable run must explode x1!"
    print("  -> Critical step size threshold eta < 2/lambda_max mathematically confirmed!")

    # -------------------------------------------------------------
    # Test 6: PyTorch Autograd Functional Hessian & HVP Cross-Check
    # -------------------------------------------------------------
    print("\n[Test 6: PyTorch Autograd Functional Hessian & HVP Cross-Check]")
    def torch_cubic(x):
        return x[0] ** 3 - 3.0 * x[0] + 2.0 * x[0] * x[1] + 2.0 * x[1] ** 2

    x_torch_A = torch.tensor([1.0, 2.0], dtype=torch.float64)
    v_torch = torch.tensor([3.0, -2.0], dtype=torch.float64)

    # PyTorch full Hessian
    H_torch = torch.autograd.functional.hessian(torch_cubic, x_torch_A)
    # PyTorch HVP
    _, hvp_torch = torch.autograd.functional.hvp(torch_cubic, x_torch_A, v_torch)

    assert np.allclose(H_torch.numpy(), HA), "PyTorch Hessian mismatch!"
    assert np.allclose(hvp_torch.numpy(), hvp_direct), "PyTorch HVP mismatch!"
    print(f"  PyTorch Hessian matches analytical HA? True")
    print(f"  PyTorch HVP matches analytical H @ v?  True")
    print("  -> PyTorch functional Autograd matches analytical derivations exactly!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 2.4 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
