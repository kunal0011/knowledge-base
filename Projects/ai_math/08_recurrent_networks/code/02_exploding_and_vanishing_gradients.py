"""
Chapter 8.2: The Exploding and Vanishing Gradient Problem in RNNs
================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Verification (Toy vanishing, exploding & norm clipping)
2. Directional distortion: Norm clipping vs Value clipping
3. Scratch implementation of Pascanu's Norm Clipping vs PyTorch clip_grad_norm_
4. Vanishing vs Exploding Gradient Dynamics across sequence lengths T in [2..50]
5. Orthogonal vs Gaussian initialization gradient preservation test
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid Hand Arithmetic Verification ---")
    # Case A: Vanishing scalar recurrence
    w_vanish = 0.5
    grad_vanish = w_vanish ** 4
    assert np.isclose(grad_vanish, 0.0625), f"Vanishing mismatch: {grad_vanish}"
    print(f"Vanishing gradient (w=0.5, T=4): {grad_vanish} (Expected: 0.0625)")
    
    # Case B: Exploding scalar recurrence
    w_explode = 2.0
    grad_explode = w_explode ** 4
    assert np.isclose(grad_explode, 16.0), f"Exploding mismatch: {grad_explode}"
    print(f"Exploding gradient (w=2.0, T=4): {grad_explode} (Expected: 16.0)")
    
    # Norm clipping
    g = np.array([6.0, 8.0], dtype=np.float64)
    theta = 2.0
    g_norm = np.linalg.norm(g)
    assert np.isclose(g_norm, 10.0), f"Norm mismatch: {g_norm}"
    
    g_clipped = theta * (g / g_norm)
    g_clipped_expected = np.array([1.2, 1.6], dtype=np.float64)
    print(f"Clipped gradient: {g_clipped} | Expected: {g_clipped_expected}")
    assert np.allclose(g_clipped, g_clipped_expected), "Clipped gradient mismatch!"
    assert np.isclose(np.linalg.norm(g_clipped), 2.0), "Clipped norm mismatch!"
    print("✓ Part 5 Visual Grid hand calculations verified!\n")

def test_clipping_directional_distortion():
    print("--- 2. Directional Distortion: Norm Clipping vs Value Clipping ---")
    g = np.array([100.0, 1.0], dtype=np.float64)
    
    # 1. Norm clipping (theta = 5.0)
    g_norm = np.linalg.norm(g)
    g_norm_clipped = 5.0 * (g / g_norm)
    cos_sim_norm = np.dot(g, g_norm_clipped) / (g_norm * np.linalg.norm(g_norm_clipped))
    
    # 2. Value clipping ([-5.0, 5.0])
    g_val_clipped = np.clip(g, -5.0, 5.0) # [5.0, 1.0]
    cos_sim_val = np.dot(g, g_val_clipped) / (g_norm * np.linalg.norm(g_val_clipped))
    angle_val_deg = np.degrees(np.arccos(np.clip(cos_sim_val, -1.0, 1.0)))
    
    print(f"Norm-Clipped Cosine Similarity:  {cos_sim_norm:.6f} (Angle: 0.00°)")
    print(f"Value-Clipped Cosine Similarity: {cos_sim_val:.6f} (Angle: {angle_val_deg:.2f}°)")
    
    assert np.isclose(cos_sim_norm, 1.0), "Norm clipping must perfectly preserve direction!"
    assert cos_sim_val < 0.99, "Value clipping should alter direction!"
    assert np.isclose(g_val_clipped, np.array([5.0, 1.0])).all(), "Value clip mismatch!"
    print("✓ Illustration 2 confirmed: Value clipping distorts gradient direction by > 10°!\n")

def clip_grad_norm_scratch(parameters, max_norm):
    """
    Scratch implementation of Pascanu's norm clipping across an iterable of tensors
    """
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
    total_norm = np.sqrt(total_norm_sq)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm

def test_pytorch_clipping_parity():
    print("--- 3. Scratch Norm Clipping vs PyTorch clip_grad_norm_ Parity ---")
    torch.manual_seed(42)
    # Create two identical sets of parameter tensors with gradients
    params_scratch = [nn.Parameter(torch.randn(10, 10, dtype=torch.float64)) for _ in range(3)]
    params_pt = [nn.Parameter(p.clone()) for p in params_scratch]
    
    # Assign identical gradients
    for p_s, p_pt in zip(params_scratch, params_pt):
        grad = torch.randn_like(p_s) * 50.0
        p_s.grad = grad.clone()
        p_pt.grad = grad.clone()
        
    max_norm = 2.5
    
    norm_s = clip_grad_norm_scratch(params_scratch, max_norm)
    norm_pt = clip_grad_norm_(params_pt, max_norm).item()
    
    print(f"Scratch Computed Total Norm: {norm_s:.6f} | PyTorch: {norm_pt:.6f}")
    assert abs(norm_s - norm_pt) < 1e-10, "Total norm computation discrepancy!"
    
    max_diff = 0.0
    for p_s, p_pt in zip(params_scratch, params_pt):
        diff = torch.max(torch.abs(p_s.grad - p_pt.grad)).item()
        max_diff = max(max_diff, diff)
        
    print(f"Max Gradient Discrepancy after Clipping: {max_diff:.2e}")
    assert max_diff < 1e-10, "Clipped gradient values mismatch!"
    print("✓ Parity with PyTorch torch.nn.utils.clip_grad_norm_ established to < 1e-10!\n")

def test_gradient_dynamics_simulation():
    print("--- 4. Vanishing vs Exploding Gradient Dynamics Simulation ---")
    T_list = [2, 5, 10, 20, 50]
    d_h = 16
    
    # Case 1: Sub-unitary matrix (sigma_max = 0.6) -> Vanishing
    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.randn(d_h, d_h))
    W_vanish = Q * 0.6 # singular values all 0.6
    
    # Case 2: Super-unitary matrix (sigma_max = 1.4) -> Exploding
    W_explode = Q * 1.4 # singular values all 1.4
    
    print(f"{'Horizon T':>10} | {'Vanishing Norm':>18} | {'Exploding Norm':>18}")
    print("-" * 52)
    
    vanish_norms = []
    explode_norms = []
    
    for T in T_list:
        # Compute W^T
        W_T_vanish = np.linalg.matrix_power(W_vanish, T)
        W_T_explode = np.linalg.matrix_power(W_explode, T)
        
        # Test vector of norm 1.0
        v = np.ones(d_h) / np.sqrt(d_h)
        norm_v = np.linalg.norm(W_T_vanish @ v)
        norm_e = np.linalg.norm(W_T_explode @ v)
        
        vanish_norms.append(norm_v)
        explode_norms.append(norm_e)
        print(f"{T:>10} | {norm_v:>18.4e} | {norm_e:>18.4e}")
        
    # Check that vanishing norm at T=50 is practically 0
    assert vanish_norms[-1] < 1e-10, f"Vanishing failed: {vanish_norms[-1]}"
    # Check that exploding norm at T=50 is huge
    assert explode_norms[-1] > 1e+6, f"Exploding failed: {explode_norms[-1]}"
    print("✓ Empirical verification of Theorem 1 and Theorem 2 confirmed!\n")

def test_orthogonal_initialization():
    print("--- 5. Orthogonal vs Gaussian Initialization Long-Term Gradient Flow ---")
    d_h = 32
    T = 40
    
    # 1. Random Gaussian initialization (std = 1 / sqrt(d_h))
    W_gauss = np.random.randn(d_h, d_h) / np.sqrt(d_h)
    
    # 2. Orthogonal initialization
    W_orth, _ = np.linalg.qr(np.random.randn(d_h, d_h))
    
    v = np.ones(d_h) / np.sqrt(d_h) # ||v|| = 1.0
    
    W_T_gauss = np.linalg.matrix_power(W_gauss, T)
    W_T_orth = np.linalg.matrix_power(W_orth, T)
    
    norm_gauss = np.linalg.norm(W_T_gauss @ v)
    norm_orth = np.linalg.norm(W_T_orth @ v)
    
    print(f"Gaussian Initialization Norm after T={T} steps:   {norm_gauss:.6e}")
    print(f"Orthogonal Initialization Norm after T={T} steps: {norm_orth:.6f}")
    
    # Gaussian weights deviate wildly from 1.0 (either decaying or exploding depending on spectrum)
    assert abs(norm_gauss - 1.0) > 0.5, f"Gaussian should deviate from unit norm, got {norm_gauss}"
    # Orthogonal initialization preserves unit norm exactly to machine precision
    assert np.isclose(norm_orth, 1.0, atol=1e-5), f"Orthogonal norm must remain ~ 1.0, got {norm_orth}"
    print("✓ Orthogonal initialization strictly preserves unit norm across 40 steps!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 8.2 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_clipping_directional_distortion()
    test_pytorch_clipping_parity()
    test_gradient_dynamics_simulation()
    test_orthogonal_initialization()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 8.2 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
