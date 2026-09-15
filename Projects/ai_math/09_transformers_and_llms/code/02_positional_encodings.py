"""
Chapter 9.2: Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi)
===================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification (2D RoPE rotation)
2. Vectorized RoPE (rotate_half) vs explicit SO(2) block-diagonal matrix parity
3. RoPE Relative Shift Invariance Theorem (<R_{m+s} q, R_{n+s} k> == <R_m q, R_n k>)
4. Sinusoidal linear projection offset property
5. ALiBi slope calculation and attention score biasing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid Hand Arithmetic Verification
# ----------------------------------------------------------------------
def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid RoPE Hand Arithmetic Verification ---")
    theta = np.pi / 4.0 # 45 degrees
    
    # Position m = 1 (45 deg)
    R1 = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=torch.float64)
    
    # Position n = 2 (90 deg)
    R2 = torch.tensor([
        [np.cos(2*theta), -np.sin(2*theta)],
        [np.sin(2*theta),  np.cos(2*theta)]
    ], dtype=torch.float64)
    
    q = torch.tensor([1.0, 0.0], dtype=torch.float64)
    k = torch.tensor([0.0, 1.0], dtype=torch.float64)
    
    q_rot = torch.matmul(R1, q)
    k_rot = torch.matmul(R2, k)
    
    q_rot_exp = torch.tensor([np.sqrt(2)/2, np.sqrt(2)/2], dtype=torch.float64)
    k_rot_exp = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    
    assert torch.allclose(q_rot, q_rot_exp), "q_rot mismatch!"
    assert torch.allclose(k_rot, k_rot_exp), "k_rot mismatch!"
    
    score = torch.dot(q_rot, k_rot).item()
    score_exp = -np.sqrt(2) / 2
    print(f"Computed Score: {score:.6f} | Expected: {score_exp:.6f}")
    assert abs(score - score_exp) < 1e-12, "Score mismatch!"
    
    # Check relative shift directly: q^T * R_{n - m} * k = q^T * R_1 * k
    relative_score = torch.dot(q, torch.matmul(R1, k)).item()
    assert abs(score - relative_score) < 1e-12, "Relative shift invariance check failed!"
    print("✓ Part 5 Visual Grid RoPE hand calculations verified!\n")

# ----------------------------------------------------------------------
# 2. Vectorized RoPE Implementation vs Explicit SO(2) Matrices
# ----------------------------------------------------------------------
def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input.
    x: (..., d) -> returns [-x2, x1, -x4, x3, ...] or [-x[d/2:], x[:d/2]]
    Here we implement standard LLaMA pair-wise rotation: [-x[..., 1::2], x[..., 0::2]]
    """
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vectorized(x, cos, sin):
    # x: (B, T, d), cos, sin: (1, T, d)
    return (x * cos) + (rotate_half(x) * sin)

def test_rope_vectorized_parity():
    print("--- 2. Vectorized RoPE vs Explicit SO(2) Matrix Parity ---")
    d = 4 # 2 pairs
    T = 3
    base = 10000.0
    
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d)) # (d/2,)
    t = torch.arange(T, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq) # (T, d/2)
    
    # Shape for vectorized rotate_half: cat(freqs, freqs)
    emb = torch.cat((freqs, freqs), dim=-1) # (T, d)
    cos = emb.cos().unsqueeze(0) # (1, T, d)
    sin = emb.sin().unsqueeze(0) # (1, T, d)
    
    x = torch.randn(1, T, d, dtype=torch.float64)
    x_vec = apply_rotary_pos_emb_vectorized(x, cos, sin)
    
    # Explicit SO(2) block diagonal construction
    x_explicit = torch.zeros_like(x)
    for pos in range(T):
        # pair 1: channels (0, 2), pair 2: channels (1, 3) under this splitting
        theta1 = freqs[pos, 0].item()
        theta2 = freqs[pos, 1].item()
        
        # In split-half convention, coordinates are (x[0], x[2]) and (x[1], x[3])
        # x_rot[0] = x[0]*cos(th1) - x[2]*sin(th1)
        # x_rot[2] = x[0]*sin(th1) + x[2]*cos(th1)
        x_explicit[0, pos, 0] = x[0, pos, 0] * np.cos(theta1) - x[0, pos, 2] * np.sin(theta1)
        x_explicit[0, pos, 2] = x[0, pos, 0] * np.sin(theta1) + x[0, pos, 2] * np.cos(theta1)
        
        x_explicit[0, pos, 1] = x[0, pos, 1] * np.cos(theta2) - x[0, pos, 3] * np.sin(theta2)
        x_explicit[0, pos, 3] = x[0, pos, 1] * np.sin(theta2) + x[0, pos, 3] * np.cos(theta2)
        
    discrepancy = torch.max(torch.abs(x_vec - x_explicit)).item()
    print(f"Max Discrepancy (Vectorized vs Explicit): {discrepancy:.2e}")
    assert discrepancy < 1e-12, "RoPE vectorized implementation mismatch!"
    print("✓ Vectorized RoPE (rotate_half) matches analytical SO(2) rotation to < 1e-12!\n")

# ----------------------------------------------------------------------
# 3. RoPE Relative Shift Invariance Theorem
# ----------------------------------------------------------------------
def test_rope_relative_shift_invariance():
    print("--- 3. RoPE Relative Shift Invariance Theorem ---")
    d = 64
    base = 10000.0
    
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    
    def get_rotary(pos):
        freqs = pos * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)
    
    q = torch.randn(1, 1, d, dtype=torch.float64)
    k = torch.randn(1, 1, d, dtype=torch.float64)
    
    # Position pair (m, n) = (10, 25)
    cos_m, sin_m = get_rotary(10)
    cos_n, sin_n = get_rotary(25)
    
    q_10 = apply_rotary_pos_emb_vectorized(q, cos_m, sin_m)
    k_25 = apply_rotary_pos_emb_vectorized(k, cos_n, sin_n)
    score_10_25 = torch.sum(q_10 * k_25).item()
    
    # Shift both positions by +50 -> (m', n') = (60, 75)
    cos_mp, sin_mp = get_rotary(60)
    cos_np, sin_np = get_rotary(75)
    
    q_60 = apply_rotary_pos_emb_vectorized(q, cos_mp, sin_mp)
    k_75 = apply_rotary_pos_emb_vectorized(k, cos_np, sin_np)
    score_60_75 = torch.sum(q_60 * k_75).item()
    
    discrepancy = abs(score_10_25 - score_60_75)
    print(f"Score at (pos 10, pos 25): {score_10_25:.8f}")
    print(f"Score at (pos 60, pos 75): {score_60_75:.8f}")
    print(f"Discrepancy across temporal shift of 50 steps: {discrepancy:.2e}")
    assert discrepancy < 1e-12, "RoPE relative shift invariance violated!"
    print("✓ RoPE Relative Shift Invariance confirmed to machine precision (< 1e-12)!\n")

# ----------------------------------------------------------------------
# 4. Sinusoidal Linear Projection Property
# ----------------------------------------------------------------------
def test_sinusoidal_linear_projection():
    print("--- 4. Sinusoidal Linear Projection Property ---")
    d_model = 16
    T = 100
    
    # Generate sinusoidal table
    pe = torch.zeros(T, d_model, dtype=torch.float64)
    pos = torch.arange(0, T).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    
    # Check linear projection for fixed offset k = 5, channel 0 (omega_0 = 1.0)
    k_offset = 5
    omega_0 = div_term[0].item()
    
    # Rotation matrix M_k
    M_k = np.array([
        [np.cos(omega_0 * k_offset), np.sin(omega_0 * k_offset)],
        [-np.sin(omega_0 * k_offset), np.cos(omega_0 * k_offset)]
    ])
    
    for p in [10, 25, 40]:
        v_p = pe[p, 0:2].numpy()
        v_p_plus_k = pe[p + k_offset, 0:2].numpy()
        # Analytical projection
        v_pred = v_p @ M_k.T
        assert np.allclose(v_pred, v_p_plus_k, atol=1e-12), "Sinusoidal projection mismatch!"
        
    print("✓ Sinusoidal Linear Projection Property confirmed: PE_{pos+k} = M_k * PE_{pos}!\n")

# ----------------------------------------------------------------------
# 5. ALiBi Attention Slopes and Matrix Biasing
# ----------------------------------------------------------------------
def get_alibi_slopes(num_heads):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if np.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_pow_2 = 2 ** int(np.floor(np.log2(num_heads)))
        return (get_slopes_power_of_2(closest_pow_2) + 
                get_alibi_slopes(2 * closest_pow_2)[0::2][:num_heads - closest_pow_2])

def test_alibi():
    print("--- 5. ALiBi Attention Slopes and Matrix Biasing ---")
    h = 8
    slopes = get_alibi_slopes(h)
    expected_slopes = [2**(-i) for i in range(1, 9)]
    print(f"Computed ALiBi Slopes for h=8: {slopes}")
    assert np.allclose(slopes, expected_slopes), "ALiBi slopes mismatch!"
    
    # Construct ALiBi bias matrix for T = 4
    T = 4
    pos_i = torch.arange(T).unsqueeze(1) # (T, 1)
    pos_j = torch.arange(T).unsqueeze(0) # (1, T)
    rel_dist = -(pos_i - pos_j).clamp(min=0).float() # lower triangular distances
    
    # For head 0 (slope = 0.5):
    bias_h0 = slopes[0] * rel_dist
    bias_h0_expected = torch.tensor([
        [ 0.0,  0.0,  0.0,  0.0],
        [-0.5,  0.0,  0.0,  0.0],
        [-1.0, -0.5,  0.0,  0.0],
        [-1.5, -1.0, -0.5,  0.0]
    ], dtype=torch.float64)
    
    assert torch.allclose(bias_h0.to(torch.float64), bias_h0_expected), "ALiBi bias matrix mismatch!"
    print("✓ ALiBi slope calculation and distance penalty matrix verified!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 9.2 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_rope_vectorized_parity()
    test_rope_relative_shift_invariance()
    test_sinusoidal_linear_projection()
    test_alibi()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 9.2 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
