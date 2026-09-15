"""
Rotary Position Embedding (RoPE) & Long-Context Scaling Verification Suite
Module 12: Modern LLM Architectures - Chapter 03

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Rotated 2D query and key vectors: q_rot = [0.877583, 0.479426], k_rot = [-1.994990, 0.141474]
   - Direct dot product and relative distance formula match: -1.682942 (< 1e-6)
2. Production PyTorch RoPE implementation (LLaMA-style complex exponential freqs_cis):
   - Shift-invariance property: <RoPE(q, m+k), RoPE(k, n+k)> == <RoPE(q, m), RoPE(k, n)>
3. NTK-Aware context scaling verification: b' = b * s^(d / (d-2))
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' ROPE ROTATIONS & RELATIVE INVARIANCE")
    print("=" * 70)

    # Inputs:
    theta = 0.50
    m, n = 1, 3
    q = np.array([1.0, 0.0])
    k = np.array([0.0, 2.0])

    # 1. Rotation Matrix at Position m = 1
    angle_m = m * theta
    R_m = np.array([[np.cos(angle_m), -np.sin(angle_m)],
                    [np.sin(angle_m),  np.cos(angle_m)]])
    q_rot = R_m @ q

    # 2. Rotation Matrix at Position n = 3
    angle_n = n * theta
    R_n = np.array([[np.cos(angle_n), -np.sin(angle_n)],
                    [np.sin(angle_n),  np.cos(angle_n)]])
    k_rot = R_n @ k

    print(f"q_rot (pos {m}): [{q_rot[0]:.6f}, {q_rot[1]:.6f}] (Expected: [0.877583, 0.479426])")
    print(f"k_rot (pos {n}): [{k_rot[0]:.6f}, {k_rot[1]:.6f}] (Expected: [-1.994990, 0.141474])")

    assert np.allclose(q_rot, [0.87758256, 0.47942554], atol=1e-5)
    assert np.allclose(k_rot, [-1.99498997, 0.14147414], atol=1e-5)

    # 3. Direct Dot Product:
    dot_direct = np.dot(q_rot, k_rot)
    print(f"Direct Rotated Dot Product: {dot_direct:.6f} (Expected: -1.682942)")
    assert np.isclose(dot_direct, -1.682942, atol=1e-5)

    # 4. Relative Distance Formulation: Delta = m - n = -2
    delta = m - n
    delta_angle = delta * theta  # -1.0 rad
    sym_term = q[0] * k[0] + q[1] * k[1]    # 0.0
    skew_term = q[0] * k[1] - q[1] * k[0]   # 2.0
    dot_rel = sym_term * np.cos(delta_angle) + skew_term * np.sin(delta_angle)

    print(f"Relative Formula Result:    {dot_rel:.6f} (Expected: -1.682942)")
    assert np.isclose(dot_rel, -1.682942, atol=1e-5)
    assert np.isclose(dot_direct, dot_rel, atol=1e-12)

    print(">> SUCCESS: Rotated dot product identically equals relative distance formula (< 1e-12)!\n")


# =====================================================================
# 2. Production PyTorch RoPE Module (LLaMA Style)
# =====================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis = cos + i*sin).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex tensor
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors using complex numbers.
    xq, xk: (B, seq_len, n_heads, head_dim)
    """
    # Reshape into complex numbers (pairs of 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Slice frequencies to sequence length
    freqs_cis = freqs_cis[: xq.shape[1], :].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)

    # Multiply by complex exponentials (rotation)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def verify_rope_pytorch_invariance():
    print("=" * 70)
    print("2. VERIFYING PYTORCH ROPE SHIFT INVARIANCE ON BATCH TENSORS")
    print("=" * 70)

    dim = 16
    max_len = 100
    freqs_cis = precompute_freqs_cis(dim=dim, end=max_len, theta=10000.0)

    # Create random query at position m=10, key at position n=15 (relative distance = -5)
    torch.manual_seed(42)
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    # Pair 1: m = 10, n = 15
    q_rot_1, _ = apply_rotary_emb(q, q, freqs_cis[10:11])
    _, k_rot_1 = apply_rotary_emb(k, k, freqs_cis[15:16])
    score_1 = torch.sum(q_rot_1 * k_rot_1).item()

    # Pair 2: Shift both by +20 -> m = 30, n = 35 (same relative distance = -5)
    q_rot_2, _ = apply_rotary_emb(q, q, freqs_cis[30:31])
    _, k_rot_2 = apply_rotary_emb(k, k, freqs_cis[35:36])
    score_2 = torch.sum(q_rot_2 * k_rot_2).item()

    # Pair 3: Shift both by +50 -> m = 60, n = 65 (same relative distance = -5)
    q_rot_3, _ = apply_rotary_emb(q, q, freqs_cis[60:61])
    _, k_rot_3 = apply_rotary_emb(k, k, freqs_cis[65:66])
    score_3 = torch.sum(q_rot_3 * k_rot_3).item()

    print(f"Score at positions (10, 15): {score_1:.8f}")
    print(f"Score at positions (30, 35): {score_2:.8f}")
    print(f"Score at positions (60, 65): {score_3:.8f}")

    assert np.isclose(score_1, score_2, atol=1e-6), "Shift invariance violated between pair 1 & 2!"
    assert np.isclose(score_1, score_3, atol=1e-6), "Shift invariance violated between pair 1 & 3!"
    print(">> SUCCESS: RoPE shift invariance verified across arbitrary sequence positions!\n")


# =====================================================================
# 3. NTK-Aware Context Scaling Verification
# =====================================================================

def verify_ntk_scaling():
    print("=" * 70)
    print("3. VERIFYING NTK-AWARE CONTEXT SCALING MATHEMATICS")
    print("=" * 70)

    # Section 6 Illustration 1 parameters:
    b = 10000.0
    scale = 4.0
    d = 64

    exponent = d / (d - 2)  # 64 / 62 ≈ 1.032258
    b_scaled = b * (scale ** exponent)

    print(f"Original Base:     {b:.1f}")
    print(f"Scaling Factor s:  {scale:.1f}x")
    print(f"Exponent d/(d-2):  {exponent:.6f}")
    print(f"Scaled Base b':    {b_scaled:.2f} (Expected: 41829.37)")

    assert np.isclose(b_scaled, 41829.37, atol=1e-1)
    print(">> SUCCESS: NTK-Aware scaled base frequency matches analytical formula!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_rope_pytorch_invariance()
    verify_ntk_scaling()
    print("=" * 70)
    print("ALL MODULE 12 CHAPTER 03 (ROPE & LONG CONTEXT) VERIFICATIONS PASSED!")
    print("=" * 70)
