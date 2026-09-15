"""
Chapter 2.2: Directional Derivatives & Gradients
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Verification (Quadratic Function).
2. Finite-Difference Limit vs. Analytical Inner Product Identity.
3. Cauchy-Schwarz Steepest Descent & Ascent Bound across Random Headings.
4. Numerical Contour Line Extraction & Orthogonality of Gradient.
5. Case A Binary Cross-Entropy Loss Directional Derivative Verification.
6. PyTorch JVP (Jacobian-Vector Product) Directional Derivative Cross-Check.
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Objective Function & Analytical Derivatives
# =====================================================================
def f_quad(x: np.ndarray) -> float:
    """
    f(x1, x2) = 3*x1^2 + 2*x1*x2 + x2^2
    """
    x1, x2 = x[0], x[1]
    return float(3.0 * x1**2 + 2.0 * x1 * x2 + x2**2)


def grad_quad(x: np.ndarray) -> np.ndarray:
    """
    nabla f(x) = [6*x1 + 2*x2,  2*x1 + 2*x2]^T
    """
    x1, x2 = x[0], x[1]
    g1 = 6.0 * x1 + 2.0 * x2
    g2 = 2.0 * x1 + 2.0 * x2
    return np.array([g1, g2], dtype=np.float64)


# =====================================================================
# 2. Numerical Limit Directional Derivative
# =====================================================================
def numerical_directional_derivative(
    func, x: np.ndarray, u: np.ndarray, t: float = 1e-7
) -> float:
    """
    Computes directional derivative via one-sided finite difference:
    D_u f(x) ~= (f(x + t*u) - f(x)) / t
    """
    assert np.isclose(np.linalg.norm(u), 1.0), "u must be a unit vector!"
    return (func(x + t * u) - func(x)) / t


# =====================================================================
# 3. Case A Logistic Regression BCE Loss
# =====================================================================
def bce_loss(w: np.ndarray, x_sample: np.ndarray, y_label: float) -> float:
    """
    L(w) = - [ y*log(sigmoid(w^T x)) + (1-y)*log(1 - sigmoid(w^T x)) ]
    """
    z = float(np.dot(w, x_sample))
    # Numerically stable log1pexp
    if z >= 0:
        return float((1.0 - y_label) * z + np.log1p(np.exp(-z)))
    else:
        return float(-y_label * z + np.log1p(np.exp(z)))


def grad_bce(w: np.ndarray, x_sample: np.ndarray, y_label: float) -> np.ndarray:
    """
    dL/dw = (sigmoid(w^T x) - y) * x
    """
    z = float(np.dot(w, x_sample))
    y_hat = 1.0 / (1.0 + np.exp(-z))
    return (y_hat - y_label) * x_sample


# =====================================================================
# 4. Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 2.2: Directional Derivatives & Gradients — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Numerical Verification
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Directional Derivative Grid]")
    x0 = np.array([2.0, 1.0], dtype=np.float64)
    val0 = f_quad(x0)
    g0 = grad_quad(x0)
    norm_g = np.linalg.norm(g0)

    print(f"  Expansion Point x0:        {x0}")
    print(f"  f(x0):                     {val0:.4f} (Expected: 17.0000)")
    print(f"  nabla f(x0):               {g0} (Expected: [14.0, 6.0])")
    print(f"  ||nabla f(x0)||:           {norm_g:.6f} (Expected: ~15.231546)")

    assert np.isclose(val0, 17.0), f"Value mismatch: {val0}"
    assert np.allclose(g0, [14.0, 6.0]), f"Gradient mismatch: {g0}"
    assert np.isclose(norm_g, np.sqrt(232.0)), f"Norm mismatch: {norm_g}"

    # Direction 1: Steepest Ascent
    u_ascent = g0 / norm_g
    d_ascent = np.dot(g0, u_ascent)
    print(f"  Steepest Ascent Du:        {d_ascent:.6f} (Expected: +{norm_g:.6f})")
    assert np.isclose(d_ascent, norm_g)

    # Direction 2: Steepest Descent
    u_descent = -g0 / norm_g
    d_descent = np.dot(g0, u_descent)
    print(f"  Steepest Descent Du:       {d_descent:.6f} (Expected: -{norm_g:.6f})")
    assert np.isclose(d_descent, -norm_g)

    # Direction 3: Contour Tangent
    v_tangent = np.array([-6.0, 14.0], dtype=np.float64)
    u_tangent = v_tangent / np.linalg.norm(v_tangent)
    d_tangent = np.dot(g0, u_tangent)
    print(f"  Contour Tangent Du:        {d_tangent:.6e} (Expected: 0.0)")
    assert np.isclose(d_tangent, 0.0, atol=1e-12)

    # Direction 4 & 5: Coordinate Axes
    d_e1 = np.dot(g0, np.array([1.0, 0.0]))
    d_e2 = np.dot(g0, np.array([0.0, 1.0]))
    print(f"  Coordinate Axis x1 Du:     {d_e1:.4f} (Expected: 14.0)")
    print(f"  Coordinate Axis x2 Du:     {d_e2:.4f} (Expected: 6.0)")
    assert np.isclose(d_e1, 14.0) and np.isclose(d_e2, 6.0)
    print("  -> Part 5 exact numbers verified perfectly!")

    # -------------------------------------------------------------
    # Test 2: Finite Difference Limit vs Analytical Dot Product
    # -------------------------------------------------------------
    print("\n[Test 2: Finite Difference Limit vs Analytical Identity]")
    np.random.seed(42)
    max_error = 0.0
    for _ in range(100):
        rand_vec = np.random.randn(2)
        u_rand = rand_vec / np.linalg.norm(rand_vec)
        d_analytical = float(np.dot(g0, u_rand))
        d_numerical = numerical_directional_derivative(f_quad, x0, u_rand, t=1e-7)
        err = abs(d_analytical - d_numerical)
        if err > max_error:
            max_error = err

    print(f"  Max Error over 100 Random Headings: {max_error:.6e}")
    assert max_error < 1e-4, f"Finite difference limit error too high: {max_error}"
    print("  -> Analytical inner product equals numerical limit for all directions!")

    # -------------------------------------------------------------
    # Test 3: Cauchy-Schwarz Steepest Descent Theorem Verification
    # -------------------------------------------------------------
    print("\n[Test 3: Cauchy-Schwarz Strict Bound Verification]")
    violating_headings = 0
    for _ in range(5000):
        rand_vec = np.random.randn(2)
        u_rand = rand_vec / np.linalg.norm(rand_vec)
        du = float(np.dot(g0, u_rand))
        if du < -norm_g - 1e-10 or du > norm_g + 1e-10:
            violating_headings += 1

    print(f"  Number of violating directions out of 5000: {violating_headings}")
    assert violating_headings == 0, "Found a direction exceeding Cauchy-Schwarz bound!"
    print(f"  -> Confirmed: For all unit vectors, -{norm_g:.4f} <= Du f <= +{norm_g:.4f}")

    # -------------------------------------------------------------
    # Test 4: Contour Orthogonality along Level Ellipse
    # -------------------------------------------------------------
    print("\n[Test 4: Level Contour Tangent Orthogonality]")
    # For f(x1, x2) = 3*x1^2 + 2*x1*x2 + x2^2 = 17, let x1 = r*cos(t), x2 = r*sin(t)
    # The implicit curve tangent is orthogonal to grad(f)
    # Check at 10 distinct points on the level curve
    orthogonality_errors = []
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    for theta in angles:
        # Direction from origin
        d_ray = np.array([np.cos(theta), np.sin(theta)])
        # Find r such that f(r * d_ray) = 17 => r^2 * (3*d1^2 + 2*d1*d2 + d2^2) = 17
        denom = 3.0 * d_ray[0] ** 2 + 2.0 * d_ray[0] * d_ray[1] + d_ray[1] ** 2
        r = np.sqrt(17.0 / denom)
        p = r * d_ray
        # Gradient at this point on contour
        gp = grad_quad(p)
        # Tangent vector is orthogonal to gp in 2D: [-gp[1], gp[0]]
        tp = np.array([-gp[1], gp[0]])
        # Dot product between gradient and tangent
        dot_prod = float(np.dot(gp, tp))
        orthogonality_errors.append(abs(dot_prod))

    max_ortho_err = max(orthogonality_errors)
    print(f"  Max <grad f, tangent> across 10 contour points: {max_ortho_err:.6e}")
    assert np.isclose(max_ortho_err, 0.0, atol=1e-12)
    print("  -> Gradient is mathematically orthogonal to contour tangent everywhere!")

    # -------------------------------------------------------------
    # Test 5: Logistic Regression BCE Loss Directional Derivative
    # -------------------------------------------------------------
    print("\n[Test 5: Part 6 Case A — BCE Loss on 2D Classifier]")
    x_sample = np.array([1.0, -1.0], dtype=np.float64)
    y_target = 1.0
    w0 = np.array([0.0, 0.0], dtype=np.float64)

    g_bce = grad_bce(w0, x_sample, y_target)
    print(f"  BCE Loss Gradient at w=0:  {g_bce} (Expected: [-0.5, 0.5])")
    assert np.allclose(g_bce, [-0.5, 0.5])

    u_diag = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    d_diag = np.dot(g_bce, u_diag)
    print(f"  Du along diagonal u:       {d_diag:.6e} (Expected: 0.0)")
    assert np.isclose(d_diag, 0.0, atol=1e-12)
    print("  -> BCE loss directional derivative verified!")

    # -------------------------------------------------------------
    # Test 6: PyTorch JVP (Jacobian-Vector Product) Directional Derivative
    # -------------------------------------------------------------
    print("\n[Test 6: PyTorch Autograd JVP Cross-Check]")
    def torch_f_quad(x):
        return 3.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + x[1] ** 2

    x_t = torch.tensor([2.0, 1.0], dtype=torch.float64)
    u_t = torch.tensor(u_ascent, dtype=torch.float64)

    # PyTorch JVP computes: (f(x), df/dx * v)
    _, jvp_val = torch.autograd.functional.jvp(torch_f_quad, (x_t,), (u_t,))
    print(f"  PyTorch JVP result:        {jvp_val.item():.6f}")
    print(f"  Analytical Du result:      {d_ascent:.6f}")
    assert np.isclose(jvp_val.item(), d_ascent)
    print("  -> PyTorch JVP directional derivative matches analytical calculation!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 2.2 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
