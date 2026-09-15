"""
FlashAttention & Online Softmax Verification Suite
Module 12: Modern LLM Architectures - Chapter 05

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Local statistics for Chunk 1: m1 = 1.0, d1 = 2.0, O1 = [1.0, 2.0]
   - Local statistics for Chunk 2: m2 = 3.0, d2 = 1.367879, O2 = [2.462118, -0.462118]
   - Online Softmax merge: m_new = 3.0, d_new = 1.638549, O_new = [2.220587, -0.055403]
   - Perfect equivalence with standard global multi-pass attention (< 1e-5)
2. Complete Python/NumPy Tiled FlashAttention Implementation:
   - Zero O(L^2) intermediate attention matrix materialization
   - Configurable block sizes (B_r, B_c)
   - Exact numerical equivalence with PyTorch scaled_dot_product_attention (< 1e-5).
"""

import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' ONLINE SOFTMAX CALCULATIONS")
    print("=" * 70)

    # Inputs:
    q = np.array([1.0, 1.0])
    k1, v1 = np.array([1.0, 0.0]), np.array([2.0, 0.0])
    k2, v2 = np.array([0.0, 1.0]), np.array([0.0, 4.0])
    k3, v3 = np.array([2.0, 0.0]), np.array([1.0, 1.0])
    k4, v4 = np.array([1.0, 2.0]), np.array([3.0, -1.0])

    # --- Chunk 1 ---
    s1 = np.dot(q, k1)  # 1.0
    s2 = np.dot(q, k2)  # 1.0
    m1 = max(s1, s2)    # 1.0
    d1 = np.exp(s1 - m1) + np.exp(s2 - m1)  # 2.0
    O1 = (np.exp(s1 - m1) * v1 + np.exp(s2 - m1) * v2) / d1

    print(f"Chunk 1: m1 = {m1:.4f}, d1 = {d1:.4f}, O1 = [{O1[0]:.4f}, {O1[1]:.4f}]")
    assert np.isclose(m1, 1.0)
    assert np.isclose(d1, 2.0)
    assert np.allclose(O1, [1.0, 2.0])

    # --- Chunk 2 ---
    s3 = np.dot(q, k3)  # 2.0
    s4 = np.dot(q, k4)  # 3.0
    m2 = max(s3, s4)    # 3.0
    d2 = np.exp(s3 - m2) + np.exp(s4 - m2)  # exp(-1) + exp(0) ≈ 1.367879
    O2 = (np.exp(s3 - m2) * v3 + np.exp(s4 - m2) * v4) / d2

    print(f"Chunk 2: m2 = {m2:.4f}, d2 = {d2:.4f}, O2 = [{O2[0]:.4f}, {O2[1]:.4f}]")
    assert np.isclose(m2, 3.0)
    assert np.isclose(d2, 1.0 + np.exp(-1.0))
    assert np.allclose(O2, [2.46211716, -0.46211716], atol=1e-5)

    # --- Online Merge ---
    m_new = max(m1, m2)  # 3.0
    d_new = d1 * np.exp(m1 - m_new) + d2 * np.exp(m2 - m_new)  # 2 * exp(-2) + d2 * 1
    O_new = (d1 * np.exp(m1 - m_new) * O1 + d2 * np.exp(m2 - m_new) * O2) / d_new

    print(f"Online Merge: m_new = {m_new:.4f}, d_new = {d_new:.6f}, O_new = [{O_new[0]:.6f}, {O_new[1]:.6f}]")
    assert np.isclose(m_new, 3.0)
    assert np.isclose(d_new, 1.6385493, atol=1e-5)

    # --- Standard Global Multi-Pass Softmax Check ---
    S_global = np.array([s1, s2, s3, s4])
    m_global = np.max(S_global)
    exp_global = np.exp(S_global - m_global)
    d_global = np.sum(exp_global)
    P_global = exp_global / d_global
    V_matrix = np.array([v1, v2, v3, v4])
    O_global = P_global @ V_matrix

    print(f"Global Multi-Pass: d_global = {d_global:.6f}, O_global = [{O_global[0]:.6f}, {O_global[1]:.6f}]")

    assert np.isclose(d_new, d_global, atol=1e-6)
    assert np.allclose(O_new, O_global, atol=1e-5)
    print(">> SUCCESS: Online Softmax matches global multi-pass attention down to machine precision!\n")


# =====================================================================
# 2. Complete Tiled FlashAttention Implementation (NumPy)
# =====================================================================

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, B_r: int = 16, B_c: int = 16) -> np.ndarray:
    """
    Tiled FlashAttention forward pass computing exact attention in blocks without materializing (L, L) matrix.
    Q, K, V: shape (L, d)
    """
    L, d = Q.shape
    O = np.zeros((L, d), dtype=np.float32)
    m = np.full((L,), -np.inf, dtype=np.float32)
    denominator = np.zeros((L,), dtype=np.float32)

    scale = 1.0 / np.sqrt(d)

    # Outer loop over KV blocks
    for j in range(0, L, B_c):
        K_j = K[j : j + B_c]  # (B_c, d)
        V_j = V[j : j + B_c]  # (B_c, d)

        # Inner loop over Query blocks
        for i in range(0, L, B_r):
            Q_i = Q[i : i + B_r]  # (B_r, d)

            # Local block logits: (B_r, B_c)
            S_ij = (Q_i @ K_j.T) * scale

            # Row-wise max of local block
            m_ij = np.max(S_ij, axis=-1)  # (B_r,)

            # Updated running max
            m_new = np.maximum(m[i : i + B_r], m_ij)

            # Local exponentials with updated max
            P_tilde = np.exp(S_ij - m_new[:, None])  # (B_r, B_c)
            d_local = np.sum(P_tilde, axis=-1)       # (B_r,)

            # Rescale factor for previous accumulation
            alpha = np.exp(m[i : i + B_r] - m_new)

            # Update denominator
            d_new = denominator[i : i + B_r] * alpha + d_local

            # Update output: O_new = (O_old * d_old * alpha + P_tilde @ V_j) / d_new
            O_prev = O[i : i + B_r]
            d_prev = denominator[i : i + B_r][:, None]

            O_updated = (O_prev * d_prev * alpha[:, None] + P_tilde @ V_j) / d_new[:, None]

            # Save state
            O[i : i + B_r] = O_updated
            m[i : i + B_r] = m_new
            denominator[i : i + B_r] = d_new

    return O


def verify_flash_attention_tiling():
    print("=" * 70)
    print("2. VERIFYING TILED FLASHATTENTION VS PYTORCH EXACT ATTENTION")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    L = 64
    d = 16
    B_r = 16
    B_c = 16

    Q = np.random.randn(L, d).astype(np.float32)
    K = np.random.randn(L, d).astype(np.float32)
    V = np.random.randn(L, d).astype(np.float32)

    # 1. Tiled FlashAttention
    O_flash = flash_attention_forward(Q, K, V, B_r=B_r, B_c=B_c)

    # 2. PyTorch Native Scaled Dot Product Attention
    Q_t = torch.tensor(Q).unsqueeze(0).unsqueeze(0)  # (1, 1, L, d)
    K_t = torch.tensor(K).unsqueeze(0).unsqueeze(0)
    V_t = torch.tensor(V).unsqueeze(0).unsqueeze(0)

    O_torch = F.scaled_dot_product_attention(Q_t, K_t, V_t).squeeze().numpy()

    max_error = np.max(np.abs(O_flash - O_torch))
    print(f"Max absolute discrepancy between FlashAttention and PyTorch: {max_error:.8f}")

    assert max_error < 1e-5, f"Discrepancy {max_error} exceeds numerical tolerance!"
    print(">> SUCCESS: Tiled FlashAttention matches PyTorch exact attention (< 1e-5)!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_flash_attention_tiling()
    print("=" * 70)
    print("ALL MODULE 12 CHAPTER 05 (FLASHATTENTION & ONLINE SOFTMAX) VERIFICATIONS PASSED!")
    print("=" * 70)
