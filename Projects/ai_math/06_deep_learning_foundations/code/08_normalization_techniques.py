"""
Chapter 6.8: Normalization Techniques (BatchNorm, LayerNorm, RMSNorm)
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (BatchNorm, LayerNorm, RMSNorm).
2. Closed-form analytical backward pass parity checks for LayerNorm and RMSNorm against PyTorch autograd.
3. Scale invariance verification on activations and gradients.
4. Parity check against native torch.nn.LayerNorm and torch.nn.RMSNorm.
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
    Validates hand calculation grid for 2x3 input matrix:
    X = [[1.0, 3.0, 5.0],
         [2.0, 4.0, 6.0]]
    under BatchNorm, LayerNorm, and RMSNorm (gamma = 1, beta = 0, eps = 0).
    """
    X = np.array([
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0]
    ], dtype=np.float64)

    # 1. BatchNorm (Column-wise across B = 2)
    mu_bn = np.mean(X, axis=0)
    var_bn = np.mean((X - mu_bn) ** 2, axis=0)
    std_bn = np.sqrt(var_bn)
    X_bn = (X - mu_bn) / std_bn

    assert np.allclose(mu_bn, [1.5, 3.5, 5.5], atol=1e-7)
    assert np.allclose(var_bn, [0.25, 0.25, 0.25], atol=1e-7)
    expected_bn = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    assert np.allclose(X_bn, expected_bn, atol=1e-7)

    # 2. LayerNorm (Row-wise across D = 3)
    mu_ln = np.mean(X, axis=1, keepdims=True)
    var_ln = np.mean((X - mu_ln) ** 2, axis=1, keepdims=True)
    std_ln = np.sqrt(var_ln)
    X_ln = (X - mu_ln) / std_ln

    assert np.allclose(mu_ln.squeeze(), [3.0, 4.0], atol=1e-7)
    assert np.allclose(var_ln.squeeze(), [8.0 / 3.0, 8.0 / 3.0], atol=1e-7)
    expected_ln_row = np.array([-math.sqrt(1.5), 0.0, math.sqrt(1.5)])
    assert np.allclose(X_ln[0], expected_ln_row, atol=1e-7)
    assert np.allclose(X_ln[1], expected_ln_row, atol=1e-7)

    # 3. RMSNorm (Row-wise without mean centering)
    rms = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True))
    X_rms = X / rms

    assert abs(rms[0, 0] - math.sqrt(35.0 / 3.0)) < 1e-7
    assert abs(rms[1, 0] - math.sqrt(56.0 / 3.0)) < 1e-7
    assert np.allclose(X_rms[0], [1.0 / math.sqrt(35/3), 3.0 / math.sqrt(35/3), 5.0 / math.sqrt(35/3)], atol=1e-7)

    print("--- Part 5 Visual Grid Verification ---")
    print(f"BatchNorm Output:\n{X_bn}")
    print(f"LayerNorm Output:\n{np.round(X_ln, 4)}")
    print(f"RMSNorm Output:\n{np.round(X_rms, 4)}")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Analytical Backward Pass Parity for LayerNorm and RMSNorm
# =====================================================================
def verify_layernorm_rmsnorm_backward():
    """
    Validates the single-line analytical vector derivatives:
    LayerNorm: dL/dx = 1/sigma * [ g - mean(g) - x_hat * mean(g * x_hat) ]
    RMSNorm:   dL/dx = 1/RMS   * [ g - x_bar * mean(g * x_bar) ]
    against PyTorch autograd.
    """
    torch.manual_seed(42)
    B, D = 4, 8
    eps = 1e-5

    # --- 1. LayerNorm Backward Check ---
    X_torch = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    gamma_torch = torch.ones(D, dtype=torch.float64)
    beta_torch = torch.zeros(D, dtype=torch.float64)

    # PyTorch forward and backward
    y_ln = torch.nn.functional.layer_norm(X_torch, (D,), weight=gamma_torch, bias=beta_torch, eps=eps)
    loss_ln = (y_ln ** 2).sum()
    loss_ln.backward()
    grad_ln_autograd = X_torch.grad.numpy()

    # Analytical formula
    X_np = X_torch.detach().numpy()
    mu = np.mean(X_np, axis=1, keepdims=True)
    var = np.var(X_np, axis=1, keepdims=True)
    std = np.sqrt(var + eps)
    x_hat = (X_np - mu) / std

    # Upstream gradient G = 2 * y_ln
    G = 2.0 * y_ln.detach().numpy()
    g = G * gamma_torch.numpy()

    mean_g = np.mean(g, axis=1, keepdims=True)
    mean_gx = np.mean(g * x_hat, axis=1, keepdims=True)
    grad_ln_analytical = (1.0 / std) * (g - mean_g - x_hat * mean_gx)

    diff_ln = np.max(np.abs(grad_ln_autograd - grad_ln_analytical))
    assert diff_ln < 1e-12, f"LayerNorm analytical gradient mismatch: {diff_ln}"

    # --- 2. RMSNorm Backward Check ---
    X_torch2 = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    rmsnorm_module = nn.RMSNorm(D, eps=eps, dtype=torch.float64)

    y_rms = rmsnorm_module(X_torch2)
    loss_rms = (y_rms ** 2).sum()
    loss_rms.backward()
    grad_rms_autograd = X_torch2.grad.numpy()

    # Analytical formula for RMSNorm:
    X_np2 = X_torch2.detach().numpy()
    rms2 = np.sqrt(np.mean(X_np2 ** 2, axis=1, keepdims=True) + eps)
    x_bar = X_np2 / rms2
    G2 = 2.0 * y_rms.detach().numpy()
    g2 = G2 * rmsnorm_module.weight.detach().numpy()

    mean_gx_bar = np.mean(g2 * x_bar, axis=1, keepdims=True)
    grad_rms_analytical = (1.0 / rms2) * (g2 - x_bar * mean_gx_bar)

    diff_rms = np.max(np.abs(grad_rms_autograd - grad_rms_analytical))
    assert diff_rms < 1e-12, f"RMSNorm analytical gradient mismatch: {diff_rms}"

    print("\n--- Analytical Backward Pass Verification ---")
    print(f"LayerNorm Max Gradient Discrepancy: {diff_ln:.2e}")
    print(f"RMSNorm Max Gradient Discrepancy:   {diff_rms:.2e}")
    print("✓ Closed-form vectorized backward passes verified to float64 machine precision (< 1e-12)!")


# =====================================================================
# 3. Scale Invariance Verification
# =====================================================================
def verify_scale_invariance():
    """
    Demonstrates that LayerNorm and RMSNorm outputs are completely invariant
    to arbitrary positive scalar scaling of inputs (e.g. x -> 10^4 * x).
    """
    x = np.random.randn(1, 16) * 2.0
    scale = 10000.0
    x_scaled = x * scale

    # LayerNorm
    def layernorm_fn(inp, eps=1e-5):
        mu = np.mean(inp, axis=-1, keepdims=True)
        var = np.var(inp, axis=-1, keepdims=True)
        return (inp - mu) / np.sqrt(var + eps)

    # RMSNorm
    def rmsnorm_fn(inp, eps=1e-5):
        rms = np.sqrt(np.mean(inp ** 2, axis=-1, keepdims=True) + eps)
        return inp / rms

    diff_ln = np.max(np.abs(layernorm_fn(x) - layernorm_fn(x_scaled)))
    diff_rms = np.max(np.abs(rmsnorm_fn(x) - rmsnorm_fn(x_scaled)))

    print("\n--- Scale Invariance Verification ---")
    print(f"LayerNorm Discrepancy under 10,000x input scaling: {diff_ln:.2e}")
    print(f"RMSNorm Discrepancy under 10,000x input scaling:   {diff_rms:.2e}")

    assert diff_ln < 1e-4
    assert diff_rms < 1e-4
    print("✓ Scale invariance property strictly confirmed!")


# =====================================================================
# 4. PyTorch Native Module Parity
# =====================================================================
def verify_pytorch_module_parity():
    """
    Verifies that scratch LayerNorm and RMSNorm match native PyTorch modules
    to float64 machine precision (< 1e-12).
    """
    X = torch.randn(3, 16, dtype=torch.float64)

    # LayerNorm parity
    ln_scratch = (X - X.mean(dim=-1, keepdim=True)) / torch.sqrt(X.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
    ln_torch = nn.LayerNorm(16, eps=1e-5, dtype=torch.float64, elementwise_affine=False)(X)
    diff_ln = torch.max(torch.abs(ln_scratch - ln_torch)).item()

    # RMSNorm parity
    rms_scratch = X / torch.sqrt(torch.mean(X ** 2, dim=-1, keepdim=True) + 1e-5)
    rms_torch = nn.RMSNorm(16, eps=1e-5, dtype=torch.float64)(X)
    diff_rms = torch.max(torch.abs(rms_scratch - rms_torch)).item()

    print("\n--- PyTorch Module Parity ---")
    print(f"LayerNorm Module Match Discrepancy: {diff_ln:.2e}")
    print(f"RMSNorm Module Match Discrepancy:   {diff_rms:.2e}")

    assert diff_ln < 1e-12
    assert diff_rms < 1e-12
    print("✓ Native PyTorch LayerNorm and RMSNorm match scratch derivations (< 1e-12)!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 6.8 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_layernorm_rmsnorm_backward()
    verify_scale_invariance()
    verify_pytorch_module_parity()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 6.8 PASSED CLEANLY!")
    print("=" * 65)
