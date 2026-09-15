"""
Modern Core Blocks: Pre-RMSNorm, SwiGLU & Residual Scaling Verification Suite
Module 12: Modern LLM Architectures - Chapter 02

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - RMSNorm forward pass: y = [0.942809, -0.235702, 2.828427, -0.942809]
   - SwiGLU gated activation: Out = 6.017634 (< 1e-6)
2. PyTorch RMSNorm Module:
   - Implementation and scale invariance verification: RMSNorm(10 * x) == RMSNorm(x)
3. PyTorch SwiGLU Feed-Forward Network:
   - Three-matrix projection and element-wise Swish gating
   - Backward autograd gradient check across full modern Transformer block.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' RMSNORM & SWIGLU CALCULATIONS")
    print("=" * 70)

    # --- A. RMSNorm Hand Verification ---
    x = np.array([2.0, -1.0, 3.0, -2.0])
    gamma = np.array([1.0, 0.5, 2.0, 1.0])

    mean_sq = np.mean(x**2)
    rms_val = np.sqrt(mean_sq)
    x_norm = x / rms_val
    y = x_norm * gamma

    print(f"Mean Square:     {mean_sq:.6f} (Expected: 4.500000)")
    print(f"RMS Divisor:     {rms_val:.6f} (Expected: 2.121320)")
    print(f"RMSNorm Output:  {y}")

    expected_y = np.array([0.94280904, -0.23570226, 2.82842712, -0.94280904])
    assert np.isclose(mean_sq, 4.5, atol=1e-6)
    assert np.isclose(rms_val, np.sqrt(4.5), atol=1e-6)
    assert np.allclose(y, expected_y, atol=1e-5)
    print(">> SUCCESS: RMSNorm hand calculations verified to exact precision!")

    # --- B. SwiGLU Hand Verification ---
    z = np.array([1.0, -2.0])
    W_gate = np.array([[2.0, 0.0],
                       [0.0, 0.5]])
    W_up = np.array([[0.5, 1.0],
                     [-1.0, 2.0]])
    W_down = np.array([[1.0],
                       [2.0]])

    g = z @ W_gate  # [2.0, -1.0]
    u = z @ W_up    # [2.5, -3.0]

    # Swish(z) = z / (1 + exp(-z))
    swish_g = g / (1.0 + np.exp(-g))
    h = swish_g * u
    out = (h @ W_down)[0]

    print(f"Gate Projection:   {g} (Expected: [2.0, -1.0])")
    print(f"Up Projection:     {u} (Expected: [2.5, -3.0])")
    print(f"Swish Gate Output: {swish_g}")
    print(f"Gated Product h:   {h}")
    print(f"SwiGLU Final Out:  {out:.6f} (Expected: 6.017634)")

    assert np.allclose(g, [2.0, -1.0], atol=1e-6)
    assert np.allclose(u, [2.5, -3.0], atol=1e-6)
    assert np.isclose(out, 6.01763392, atol=1e-5)
    print(">> SUCCESS: SwiGLU gating and forward pass verified to exact precision!\n")


# =====================================================================
# 2. PyTorch Vectorized RMSNorm Module
# =====================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def verify_rmsnorm_module():
    print("=" * 70)
    print("2. VERIFYING PYTORCH RMSNORM MODULE & SCALE INVARIANCE")
    print("=" * 70)

    dim = 64
    batch_size = 8
    seq_len = 16

    norm = RMSNorm(dim=dim)
    x = torch.randn(batch_size, seq_len, dim)

    # 1. Output shape check
    out = norm(x)
    assert out.shape == (batch_size, seq_len, dim)

    # 2. Scale Invariance Test: RMSNorm(alpha * x) == RMSNorm(x)
    alpha = 15.7
    x_scaled = alpha * x
    out_orig = norm(x)
    out_scaled = norm(x_scaled)

    max_diff = torch.max(torch.abs(out_orig - out_scaled)).item()
    print(f"Max difference under {alpha}x input scaling: {max_diff:.8f}")
    assert max_diff < 1e-5, f"RMSNorm must be scale invariant, max diff {max_diff}"
    print(">> SUCCESS: RMSNorm scale invariance verified!\n")


# =====================================================================
# 3. PyTorch SwiGLU MLP & Full Modern Transformer Block
# =====================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020; LLaMA-style).
    """
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x @ w_gate) * (x @ w_up) @ w_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModernTransformerBlock(nn.Module):
    """
    Modern Pre-RMSNorm Transformer Layer with SwiGLU MLP.
    """
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        # Dummy linear self-attention projection for gradient check
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN 1: Attention branch
        h = x + self.attn_proj(self.norm1(x))
        # Pre-LN 2: SwiGLU MLP branch
        out = h + self.mlp(self.norm2(h))
        return out


def verify_transformer_block():
    print("=" * 70)
    print("3. VERIFYING FULL PRE-RMSNORM + SWIGLU TRANSFORMER BLOCK")
    print("=" * 70)

    d_model = 64
    d_ffn = int(8 / 3 * d_model)  # 8/3 d parameter matching rule

    block = ModernTransformerBlock(d_model=d_model, d_ffn=d_ffn)
    x = torch.randn(4, 12, d_model, requires_grad=True)

    out = block(x)
    assert out.shape == (4, 12, d_model)

    # Backward gradient check
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    print(">> SUCCESS: Modern Transformer Block executed forward and backward pass flawlessly!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_rmsnorm_module()
    verify_transformer_block()
    print("=" * 70)
    print("ALL MODULE 12 CHAPTER 02 (MODERN CORE BLOCKS) VERIFICATIONS PASSED!")
    print("=" * 70)
