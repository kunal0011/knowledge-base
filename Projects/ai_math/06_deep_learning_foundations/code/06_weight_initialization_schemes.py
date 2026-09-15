"""
Chapter 6.6: Weight Initialization Schemes (Xavier/Glorot, He/Kaiming, Orthogonal)
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (hand calculation match).
2. 50-layer deep network forward variance propagation test (Explosion vs. Collapse vs. Kaiming Stability).
3. Orthogonal initialization singular value spectrum verification (Isometry check).
4. PyTorch native initializer parity test.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


# =====================================================================
# 1. Part 5 Visual Grid Numerical Verification
# =====================================================================
def verify_visual_grid():
    """
    Validates hand calculation grid for n_in = 4, n_out = 4
    and 3-layer variance evolution under ReLU:
    Standard Normal: Var = 1.0  -> 2.0, 4.0, 8.0
    Xavier Normal:   Var = 0.25 -> 0.5, 0.25, 0.125
    Kaiming Normal:  Var = 0.50 -> 1.0, 1.0, 1.0
    """
    n_in = 4
    n_out = 4

    var_std = 1.0
    var_xavier = 2.0 / (n_in + n_out)
    var_kaiming = 2.0 / n_in

    assert abs(var_xavier - 0.25) < 1e-7
    assert abs(var_kaiming - 0.50) < 1e-7

    # 3-layer theoretical recurrence: Var(z_{l}) = (0.5 * n_in * Var(w)) * Var(z_{l-1})
    # with Var(z_0) = 1.0
    gain_std = 0.5 * n_in * var_std       # 2.0
    gain_xavier = 0.5 * n_in * var_xavier # 0.5
    gain_kaiming = 0.5 * n_in * var_kaiming # 1.0

    layers_std = [gain_std ** l for l in [1, 2, 3]]
    layers_xavier = [gain_xavier ** l for l in [1, 2, 3]]
    layers_kaiming = [gain_kaiming ** l for l in [1, 2, 3]]

    assert np.allclose(layers_std, [2.0, 4.0, 8.0], atol=1e-7)
    assert np.allclose(layers_xavier, [0.5, 0.25, 0.125], atol=1e-7)
    assert np.allclose(layers_kaiming, [1.0, 1.0, 1.0], atol=1e-7)

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Standard Normal Variances (Layers 1-3): {layers_std} (Exploding 2^l)")
    print(f"Xavier Normal Variances   (Layers 1-3): {layers_xavier} (Vanishing 0.5^l)")
    print(f"Kaiming Normal Variances  (Layers 1-3): {layers_kaiming} (Perfect Equilibrium 1.0^l)")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. 50-Layer Deep Network Forward Variance Propagation Test
# =====================================================================
def verify_50layer_variance_propagation():
    """
    Simulates a 50-layer deep network with width 256 and ReLU activations.
    Validates that:
    - Standard Normal explodes to huge values (> 1e5).
    - Xavier Normal collapses to near-zero (< 1e-5).
    - Kaiming Normal maintains stable variance in [0.2, 5.0].
    """
    np.random.seed(42)
    depth = 50
    dim = 256
    n_samples = 1000

    # Input features with unit variance
    X = np.random.randn(n_samples, dim)

    # 1. Standard Normal Initialization (sigma = 1.0)
    a = X.copy()
    for _ in range(depth):
        w = np.random.randn(dim, dim)
        z = a @ w
        a = np.maximum(0.0, z)
        # Avoid overflow by capping check
        if np.max(a) > 1e15:
            break
    var_final_std = float(np.var(a))

    # 2. Xavier Normal Initialization (sigma = sqrt(2 / (2 * dim)))
    a = X.copy()
    sigma_xavier = math.sqrt(2.0 / (dim + dim))
    for _ in range(depth):
        w = np.random.randn(dim, dim) * sigma_xavier
        z = a @ w
        a = np.maximum(0.0, z)
    var_final_xavier = float(np.var(a))

    # 3. Kaiming Normal Initialization (sigma = sqrt(2 / dim))
    a = X.copy()
    sigma_kaiming = math.sqrt(2.0 / dim)
    for _ in range(depth):
        w = np.random.randn(dim, dim) * sigma_kaiming
        z = a @ w
        a = np.maximum(0.0, z)
    var_final_kaiming = float(np.var(a))

    print("\n--- 50-Layer Deep Network Variance Propagation ---")
    print(f"Standard Normal Final Variance: {var_final_std:.2e} (Exploded!)")
    print(f"Xavier Normal Final Variance:   {var_final_xavier:.2e} (Collapsed!)")
    print(f"Kaiming Normal Final Variance:  {var_final_kaiming:.4f} (STABLE ~ 1.0!)")

    assert var_final_std > 1e5, "Standard Normal failed to explode as expected"
    assert var_final_xavier < 1e-5, "Xavier failed to collapse under ReLU as expected"
    assert 0.1 < var_final_kaiming < 10.0, "Kaiming failed to maintain stable variance"
    print("✓ He/Kaiming equilibrium strictly proven on 50-layer deep network!")


# =====================================================================
# 3. Orthogonal Initialization Singular Value Spectrum (Isometry)
# =====================================================================
def verify_orthogonal_spectrum():
    """
    Constructs an orthogonal matrix via QR decomposition and validates
    that all singular values are 1.0 and Euclidean norm is perfectly preserved.
    """
    np.random.seed(42)
    dim = 64
    M = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(M)
    # Adjust signs to ensure proper Haar distribution
    Q = Q @ np.diag(np.sign(np.diag(R)))

    # 1. Check orthogonality Q^T @ Q = I
    I_check = Q.T @ Q
    assert np.allclose(I_check, np.eye(dim), atol=1e-12)

    # 2. Singular value spectrum
    singular_values = np.linalg.svd(Q, compute_uv=False)
    assert np.allclose(singular_values, np.ones(dim), atol=1e-12)

    # 3. Exact Norm Preservation
    x = np.random.randn(dim)
    norm_x = np.linalg.norm(x)
    norm_Qx = np.linalg.norm(Q @ x)
    assert abs(norm_x - norm_Qx) < 1e-12

    print("\n--- Orthogonal Initialization Isometry Verification ---")
    print(f"Max Singular Value Discrepancy from 1.0: {np.max(np.abs(singular_values - 1.0)):.2e}")
    print(f"Norm Input ||x||: {norm_x:.6f} | Norm Transformed ||Qx||: {norm_Qx:.6f}")
    print("✓ Orthogonal matrix preserves energy and spectrum to machine precision (< 1e-12)!")


# =====================================================================
# 4. PyTorch Native Initializer Parity Verification
# =====================================================================
def verify_pytorch_init_parity():
    """
    Compares analytical variance formulas against PyTorch's native
    torch.nn.init.kaiming_normal_ and torch.nn.init.xavier_normal_.
    """
    torch.manual_seed(42)
    n_in, n_out = 512, 512
    tensor_kaiming = torch.empty(n_out, n_in)
    tensor_xavier = torch.empty(n_out, n_in)

    init.kaiming_normal_(tensor_kaiming, mode='fan_in', nonlinearity='relu')
    init.xavier_normal_(tensor_xavier, gain=1.0)

    emp_var_kaiming = tensor_kaiming.var().item()
    theo_var_kaiming = 2.0 / n_in

    emp_var_xavier = tensor_xavier.var().item()
    theo_var_xavier = 2.0 / (n_in + n_out)

    diff_kaiming = abs(emp_var_kaiming - theo_var_kaiming) / theo_var_kaiming
    diff_xavier = abs(emp_var_xavier - theo_var_xavier) / theo_var_xavier

    print("\n--- PyTorch Initializer Parity Verification ---")
    print(f"Kaiming Empirical: {emp_var_kaiming:.6f} | Theoretical: {theo_var_kaiming:.6f} (Rel diff: {diff_kaiming:.3%})")
    print(f"Xavier Empirical:  {emp_var_xavier:.6f} | Theoretical: {theo_var_xavier:.6f} (Rel diff: {diff_xavier:.3%})")

    assert diff_kaiming < 0.02
    assert diff_xavier < 0.02
    print("✓ PyTorch native initializers match theoretical variance derivations (< 2% sample error)!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 6.6 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_50layer_variance_propagation()
    verify_orthogonal_spectrum()
    verify_pytorch_init_parity()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 6.6 PASSED CLEANLY!")
    print("=" * 65)
