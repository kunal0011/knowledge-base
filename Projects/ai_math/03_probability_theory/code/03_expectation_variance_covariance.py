"""
Chapter 3.3: Expectation, Variance, Covariance & Covariance Matrices
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' 2D Discrete Joint Distribution & Covariance Matrix Walkthrough.
2. Zero Covariance with Deterministic Dependence (X ~ U[-1, 1], Y = X^2).
3. ZCA / PCA Whitening: Transforming Arbitrary Covariance to Identity Matrix I.
4. Kaiming vs. Improper Weight Initialization Variance Propagation Across 20 Layers.
5. Mini-Batch Gradient Variance Reduction (Var ~ 1/B).
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Discrete Joint PMF Analysis
# =====================================================================
def analyze_part5_discrete_joint():
    # Support: X1 in {1, 2}, X2 in {1, 3}
    x1_vals = np.array([1.0, 2.0])
    x2_vals = np.array([1.0, 3.0])

    # Joint PMF: rows correspond to x1, cols to x2
    p_joint = np.array([
        [0.10, 0.30],  # x1 = 1
        [0.40, 0.20],  # x1 = 2
    ])

    # 1. Marginals
    p_x1 = np.sum(p_joint, axis=1)  # [0.4, 0.6]
    p_x2 = np.sum(p_joint, axis=0)  # [0.5, 0.5]

    # 2. Expectations
    mean_x1 = float(np.sum(x1_vals * p_x1))  # 1.6
    mean_x2 = float(np.sum(x2_vals * p_x2))  # 2.0

    # 3. Second Moments & Variances
    mean_x1_sq = float(np.sum((x1_vals**2) * p_x1))  # 2.8
    mean_x2_sq = float(np.sum((x2_vals**2) * p_x2))  # 5.0

    var_x1 = mean_x1_sq - mean_x1**2  # 0.24
    var_x2 = mean_x2_sq - mean_x2**2  # 1.0

    # 4. Cross-moment E[X1 X2]
    # Outer product of grids
    grid_x1, grid_x2 = np.meshgrid(x1_vals, x2_vals, indexing="ij")
    mean_x1_x2 = float(np.sum(grid_x1 * grid_x2 * p_joint))  # 3.0

    # 5. Covariance & Correlation
    cov_x1_x2 = mean_x1_x2 - mean_x1 * mean_x2  # -0.20
    corr_x1_x2 = cov_x1_x2 / np.sqrt(var_x1 * var_x2)

    # 6. Covariance Matrix
    cov_matrix = np.array([[var_x1, cov_x1_x2], [cov_x1_x2, var_x2]])

    det_cov = float(np.linalg.det(cov_matrix))
    tr_cov = float(np.trace(cov_matrix))

    return {
        "p_x1": p_x1,
        "p_x2": p_x2,
        "mean_x1": mean_x1,
        "mean_x2": mean_x2,
        "var_x1": var_x1,
        "var_x2": var_x2,
        "mean_x1_x2": mean_x1_x2,
        "cov_x1_x2": cov_x1_x2,
        "corr_x1_x2": corr_x1_x2,
        "cov_matrix": cov_matrix,
        "det_cov": det_cov,
        "tr_cov": tr_cov,
    }


# =====================================================================
# 2. Uncorrelated but Non-Linear Dependent Test
# =====================================================================
def verify_uncorrelated_dependent(num_samples: int = 500_000):
    """
    X ~ Uniform[-1, 1], Y = X^2
    """
    np.random.seed(42)
    x = np.random.uniform(-1.0, 1.0, size=num_samples)
    y = x**2

    # Sample covariance
    cov_xy = float(np.cov(x, y)[0, 1])
    return cov_xy


# =====================================================================
# 3. ZCA / PCA Whitening Implementation
# =====================================================================
def zca_whitening(X: np.ndarray) -> np.ndarray:
    """
    Applies ZCA whitening: Z = (X - mu) @ (Q Lambda^-1/2 Q^T)
    """
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    sigma = np.cov(X_centered, rowvar=False)

    # Eigendecomposition
    eigvals, Q = np.linalg.eigh(sigma)
    # Inverse square root of eigenvalues
    inv_sqrt_lambda = np.diag(1.0 / np.sqrt(eigvals + 1e-12))
    # Whitening matrix W = Q @ inv_sqrt_lambda @ Q^T
    W_zca = Q @ inv_sqrt_lambda @ Q.T

    Z = X_centered @ W_zca
    return Z


# =====================================================================
# 4. Kaiming vs. Improper Weight Initialization Variance Propagation
# =====================================================================
def simulate_variance_propagation(depth: int = 20, dim: int = 256):
    """
    Propagates random Gaussian input through 20 layers of Linear + ReLU:
    1. Kaiming Var(W) = 2 / dim  -> Preserves variance ~ 1.0
    2. Under-scaled Var(W) = 1 / dim -> Decays variance as (1/2)^depth
    """
    np.random.seed(42)
    x_init = np.random.randn(500, dim)

    # Simulation 1: Kaiming He Init
    x_kaiming = x_init.copy()
    kaiming_vars = [float(np.var(x_kaiming))]
    for _ in range(depth):
        W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        x_kaiming = np.maximum(0.0, x_kaiming @ W)
        kaiming_vars.append(float(np.var(x_kaiming)))

    # Simulation 2: Under-scaled Init
    x_bad = x_init.copy()
    bad_vars = [float(np.var(x_bad))]
    for _ in range(depth):
        W = np.random.randn(dim, dim) * np.sqrt(1.0 / dim)
        x_bad = np.maximum(0.0, x_bad @ W)
        bad_vars.append(float(np.var(x_bad)))

    return kaiming_vars, bad_vars


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 3.3: Expectation, Variance & Covariance — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Discrete Joint Distribution
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' 2D Covariance Matrix]")
    res = analyze_part5_discrete_joint()
    print(f"  P(X1):                     {res['p_x1']} (Expected: [0.4, 0.6])")
    print(f"  P(X2):                     {res['p_x2']} (Expected: [0.5, 0.5])")
    print(f"  E[X1]:                     {res['mean_x1']:.4f} (Expected: 1.6000)")
    print(f"  E[X2]:                     {res['mean_x2']:.4f} (Expected: 2.0000)")
    print(f"  Var(X1):                   {res['var_x1']:.4f} (Expected: 0.2400)")
    print(f"  Var(X2):                   {res['var_x2']:.4f} (Expected: 1.0000)")
    print(f"  E[X1 X2]:                  {res['mean_x1_x2']:.4f} (Expected: 3.0000)")
    print(f"  Cov(X1, X2):               {res['cov_x1_x2']:.4f} (Expected: -0.2000)")
    print(f"  Correlation rho_12:        {res['corr_x1_x2']:.6f} (Expected: ~-0.408248)")
    print(f"  Covariance Matrix Sigma:\n{res['cov_matrix']}")
    print(f"  det(Sigma):                {res['det_cov']:.4f} (Expected: 0.2000)")

    assert np.allclose(res["p_x1"], [0.4, 0.6])
    assert np.allclose(res["p_x2"], [0.5, 0.5])
    assert np.isclose(res["mean_x1"], 1.6)
    assert np.isclose(res["mean_x2"], 2.0)
    assert np.isclose(res["var_x1"], 0.24)
    assert np.isclose(res["var_x2"], 1.0)
    assert np.isclose(res["mean_x1_x2"], 3.0)
    assert np.isclose(res["cov_x1_x2"], -0.20)
    assert np.isclose(res["corr_x1_x2"], -0.20 / np.sqrt(0.24))
    assert np.isclose(res["det_cov"], 0.20)
    assert res["det_cov"] > 0 and res["tr_cov"] > 0, "Sigma must be Positive Definite!"
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: Uncorrelated but Non-Linear Dependent Test
    # -------------------------------------------------------------
    print("\n[Test 2: Uncorrelated but Fully Dependent (X ~ U[-1, 1], Y = X^2)]")
    cov_val = verify_uncorrelated_dependent(num_samples=500_000)
    print(f"  Sample Covariance Cov(X, X^2): {cov_val:.6e} (Expected: ~0.0)")
    assert abs(cov_val) < 1e-3, "Covariance should be virtually zero!"
    print("  -> Uncorrelatedness does NOT imply independence confirmed!")

    # -------------------------------------------------------------
    # Test 3: ZCA Whitening Transformation Test
    # -------------------------------------------------------------
    print("\n[Test 3: ZCA Whitening Cov(Z) == Identity Matrix Check]")
    np.random.seed(99)
    # Generate correlated 3D Gaussian
    A_mix = np.array([[2.0, 1.0, -0.5], [1.0, 3.0, 0.8], [-0.5, 0.8, 1.5]])
    X_raw = np.random.randn(100_000, 3) @ A_mix + np.array([10.0, -5.0, 2.0])

    Z_whitened = zca_whitening(X_raw)
    cov_whitened = np.cov(Z_whitened, rowvar=False)

    print(f"  Whitened Covariance Matrix:\n{cov_whitened}")
    assert np.allclose(cov_whitened, np.eye(3), atol=0.02)
    print("  -> ZCA Whitening successfully transformed covariance to Identity!")

    # -------------------------------------------------------------
    # Test 4: Kaiming vs. Under-scaled Weight Initialization
    # -------------------------------------------------------------
    print("\n[Test 4: Kaiming (2/n_in) vs. Under-scaled (1/n_in) Variance Propagation]")
    k_vars, b_vars = simulate_variance_propagation(depth=20, dim=256)

    print(f"  Initial Input Variance:     {k_vars[0]:.4f}")
    print(f"  Layer 20 Kaiming Variance:  {k_vars[-1]:.4f} (Stable energy preservation)")
    print(f"  Layer 20 Bad Init Variance: {b_vars[-1]:.6e} (Exponential collapse to 0!)")

    assert k_vars[-1] > 0.1 and k_vars[-1] < 10.0, "Kaiming variance should remain O(1)!"
    assert b_vars[-1] < 1e-4, "Under-scaled variance should collapse to zero!"
    print("  -> Mathematical necessity of He/Kaiming 2/n_in factor verified!")

    # -------------------------------------------------------------
    # Test 5: Mini-batch Variance Reduction (Var ~ 1/B)
    # -------------------------------------------------------------
    print("\n[Test 5: Mini-Batch Gradient Variance Scaling]")
    # Generate individual noisy gradient vectors
    true_grad = np.array([2.0, -1.0])
    noise_sigma = 4.0
    N_trials = 20_000

    var_single_list = []
    var_batch16_list = []

    for _ in range(N_trials):
        # Batch size 1
        g1 = true_grad + np.random.randn(2) * noise_sigma
        var_single_list.append(g1)
        # Batch size 16
        g16 = true_grad + np.mean(np.random.randn(16, 2) * noise_sigma, axis=0)
        var_batch16_list.append(g16)

    emp_var_b1 = np.var(np.array(var_single_list)[:, 0])
    emp_var_b16 = np.var(np.array(var_batch16_list)[:, 0])
    ratio = emp_var_b1 / emp_var_b16

    print(f"  Variance with B=1:         {emp_var_b1:.4f} (Expected: ~16.0)")
    print(f"  Variance with B=16:        {emp_var_b16:.4f} (Expected: ~1.0)")
    print(f"  Variance Reduction Ratio:  {ratio:.2f}x (Expected: ~16.00x)")

    assert np.isclose(ratio, 16.0, atol=1.5)
    print("  -> Mini-batch gradient variance scales strictly as 1/B confirmed!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 3.3 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
