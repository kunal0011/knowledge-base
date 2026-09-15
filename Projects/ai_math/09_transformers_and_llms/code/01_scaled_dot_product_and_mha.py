"""
Chapter 9.1: Scaled Dot-Product & Multi-Head Attention (MHA)
============================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification (Head 1 step-by-step)
2. Variance Inflation Theorem empirical verification (Var(q^T k) = d_k)
3. Softmax gradient saturation demonstration
4. Causal masking correctness
5. Full parity with PyTorch's torch.nn.MultiheadAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid Hand Arithmetic Verification
# ----------------------------------------------------------------------
def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid Hand Arithmetic Verification ---")
    X = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ], dtype=torch.float64)
    
    W1_Q = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float64)
    
    W1_K = torch.tensor([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ], dtype=torch.float64)
    
    W1_V = torch.tensor([
        [2.0, 0.0],
        [0.0, 2.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float64)
    
    # 1. Projections
    Q1 = torch.matmul(X, W1_Q)
    K1 = torch.matmul(X, W1_K)
    V1 = torch.matmul(X, W1_V)
    
    Q1_exp = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    K1_exp = torch.tensor([[1.0, 2.0], [1.0, 0.0]], dtype=torch.float64)
    V1_exp = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    
    assert torch.allclose(Q1, Q1_exp), "Q1 mismatch!"
    assert torch.allclose(K1, K1_exp), "K1 mismatch!"
    assert torch.allclose(V1, V1_exp), "V1 mismatch!"
    
    # 2. Scaled Dot Products (d_k = 2)
    d_k = 2.0
    S1 = torch.matmul(Q1, K1.t()) / np.sqrt(d_k)
    S1_exp = torch.tensor([
        [2.0 / np.sqrt(2), 2.0 / np.sqrt(2)],
        [4.0 / np.sqrt(2), 0.0 / np.sqrt(2)]
    ], dtype=torch.float64)
    assert torch.allclose(S1, S1_exp), "S1 mismatch!"
    
    # 3. Softmax Attention Weights
    A1 = F.softmax(S1, dim=-1)
    A1_exp = torch.tensor([
        [0.5, 0.5],
        [np.exp(4.0/np.sqrt(2)) / (np.exp(4.0/np.sqrt(2)) + 1.0),
         1.0 / (np.exp(4.0/np.sqrt(2)) + 1.0)]
    ], dtype=torch.float64)
    assert torch.allclose(A1, A1_exp), "A1 mismatch!"
    
    # 4. Values Aggregation
    O1 = torch.matmul(A1, V1)
    O1_exp = torch.matmul(A1_exp, V1_exp)
    print("Computed Output O1:\n", O1.numpy())
    assert torch.allclose(O1, O1_exp), "O1 mismatch!"
    print("✓ Part 5 Visual Grid Multi-Head Attention hand calculations verified!\n")

# ----------------------------------------------------------------------
# 2. Variance Inflation Theorem Verification
# ----------------------------------------------------------------------
def test_variance_inflation():
    print("--- 2. Variance Inflation Theorem Empirical Verification ---")
    np.random.seed(42)
    num_samples = 50000
    
    for d_k in [16, 64, 256]:
        q = np.random.randn(num_samples, d_k)
        k = np.random.randn(num_samples, d_k)
        
        dot_products = np.sum(q * k, axis=1)
        var_unscaled = np.var(dot_products)
        
        scaled_dot_products = dot_products / np.sqrt(d_k)
        var_scaled = np.var(scaled_dot_products)
        
        print(f"d_k = {d_k:>3} | Unscaled Var: {var_unscaled:>8.2f} (Expected: {d_k:>3}) | Scaled Var: {var_scaled:.4f} (Expected: 1.0000)")
        assert abs(var_unscaled - d_k) / d_k < 0.05, f"Unscaled variance deviated for d_k={d_k}"
        assert abs(var_scaled - 1.0) < 0.05, f"Scaled variance deviated for d_k={d_k}"
        
    print("✓ Variance Inflation Theorem strictly confirmed: Var(q^T k) = d_k and Var(q^T k / sqrt(d_k)) = 1.0!\n")

# ----------------------------------------------------------------------
# 3. Softmax Gradient Vanishing Demonstration
# ----------------------------------------------------------------------
def test_softmax_saturation():
    print("--- 3. Softmax Gradient Saturation Demonstration ---")
    # Unscaled large logits
    s_unscaled = torch.tensor([30.0, 10.0], dtype=torch.float64, requires_grad=True)
    a_unscaled = F.softmax(s_unscaled, dim=0)
    a_unscaled[0].backward()
    grad_unscaled = s_unscaled.grad[0].item()
    
    # Scaled logits (divided by sqrt(100) = 10)
    s_scaled = torch.tensor([3.0, 1.0], dtype=torch.float64, requires_grad=True)
    a_scaled = F.softmax(s_scaled, dim=0)
    a_scaled[0].backward()
    grad_scaled = s_scaled.grad[0].item()
    
    ratio = grad_scaled / grad_unscaled
    print(f"Unscaled Softmax Gradient: {grad_unscaled:.4e}")
    print(f"Scaled Softmax Gradient:   {grad_scaled:.4f}")
    print(f"Gradient Recovery Ratio:   {ratio:.2e}x larger!")
    
    assert grad_unscaled < 1e-7, "Unscaled gradient should have vanished!"
    assert grad_scaled > 0.05, "Scaled gradient should be active!"
    print("✓ Softmax gradient saturation proof verified!\n")

# ----------------------------------------------------------------------
# 4. MultiHeadAttention Scratch Implementation & Causal Masking
# ----------------------------------------------------------------------
class MultiHeadAttentionScratch(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads!"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, causal=False):
        """
        q, k, v: (B, T, d_model)
        """
        B, T, _ = q.shape
        
        # 1. Linear projections and reshape to (B, num_heads, T, d_k)
        Q = self.W_q(q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if causal:
            # Upper triangular mask filled with -inf
            mask = torch.triu(torch.full((T, T), float('-inf'), device=q.device), diagonal=1)
            scores = scores + mask
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V) # (B, num_heads, T, d_k)
        
        # 3. Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), attn_weights

def test_causal_masking():
    print("--- 4. Causal Masking Verification ---")
    mha = MultiHeadAttentionScratch(d_model=16, num_heads=2)
    x = torch.randn(2, 4, 16)
    out, attn_weights = mha(x, x, x, causal=True)
    
    # Verify upper triangle of attention matrix is strictly 0.0
    upper_tri = torch.triu(attn_weights, diagonal=1)
    max_leakage = torch.max(upper_tri).item()
    print(f"Max Causal Leakage to Future Tokens: {max_leakage:.2e}")
    assert max_leakage == 0.0, "Causal attention leaked information to future tokens!"
    print("✓ Causal masking strictly prevents future information leakage!\n")

# ----------------------------------------------------------------------
# 5. Parity with PyTorch torch.nn.MultiheadAttention
# ----------------------------------------------------------------------
def test_pytorch_parity():
    print("--- 5. End-to-End Parity with PyTorch nn.MultiheadAttention ---")
    B, T, d_model, num_heads = 2, 5, 16, 4
    torch.manual_seed(42)
    
    pt_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, bias=False, batch_first=True).to(torch.float64)
    scratch_mha = MultiHeadAttentionScratch(d_model=d_model, num_heads=num_heads).to(torch.float64)
    
    # Copy weights
    # PyTorch in_proj_weight has shape (3 * d_model, d_model)
    in_proj = pt_mha.in_proj_weight.detach()
    scratch_mha.W_q.weight.data.copy_(in_proj[:d_model])
    scratch_mha.W_k.weight.data.copy_(in_proj[d_model:2*d_model])
    scratch_mha.W_v.weight.data.copy_(in_proj[2*d_model:])
    scratch_mha.W_o.weight.data.copy_(pt_mha.out_proj.weight.detach())
    
    x_pt = torch.randn(B, T, d_model, dtype=torch.float64, requires_grad=True)
    x_scratch = x_pt.clone().detach().requires_grad_(True)
    
    # Forward passes
    out_pt, _ = pt_mha(x_pt, x_pt, x_pt, need_weights=False)
    out_scratch, _ = scratch_mha(x_scratch, x_scratch, x_scratch, causal=False)
    
    disc_fwd = torch.max(torch.abs(out_pt - out_scratch)).item()
    print(f"Max Discrepancy Forward Pass: {disc_fwd:.2e}")
    assert disc_fwd < 1e-12, "Forward pass mismatch!"
    
    # Backward passes
    loss_pt = out_pt.sum()
    loss_scratch = out_scratch.sum()
    
    loss_pt.backward()
    loss_scratch.backward()
    
    disc_bwd = torch.max(torch.abs(x_pt.grad - x_scratch.grad)).item()
    print(f"Max Discrepancy Backward Pass: {disc_bwd:.2e}")
    assert disc_bwd < 1e-12, "Backward gradient mismatch!"
    print("✓ Full parity with PyTorch nn.MultiheadAttention established to < 1e-12!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 9.1 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_variance_inflation()
    test_softmax_saturation()
    test_causal_masking()
    test_pytorch_parity()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 9.1 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
