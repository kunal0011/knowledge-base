"""
Chapter 9.4: Modern Efficiency (MQA, GQA, FlashAttention, MoE, KV-Cache)
========================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification (GQA grouping)
2. GroupedQueryAttention (GQA) with KV-Cache vs Non-Cached Parity
3. Online Softmax (FlashAttention tiling algorithm) exact numerical parity
4. Sparse Mixture-of-Experts (MoE) Top-k routing and load-balancing loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid GQA Hand Arithmetic Verification
# ----------------------------------------------------------------------
def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid GQA Hand Arithmetic Verification ---")
    # 4 Query heads, 2 KV groups, d_k = 2
    q1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    q2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    q3 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    q4 = torch.tensor([2.0, 0.0], dtype=torch.float64)
    
    k1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    v1 = torch.tensor([3.0, 0.0], dtype=torch.float64)
    
    k2 = torch.tensor([0.0, 2.0], dtype=torch.float64)
    v2 = torch.tensor([0.0, 4.0], dtype=torch.float64)
    
    d_k = 2.0
    # Group 1: q1 and q2 attend to k1, v1
    # For single token key, softmax([s]) = 1.0
    o1 = 1.0 * v1
    o2 = 1.0 * v1
    
    # Group 2: q3 and q4 attend to k2, v2
    o3 = 1.0 * v2
    o4 = 1.0 * v2
    
    O_concat = torch.cat([o1, o2, o3, o4])
    O_expected = torch.tensor([3.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 4.0], dtype=torch.float64)
    
    print(f"Computed Concatenated GQA Output: {O_concat.numpy()}")
    print(f"Expected Concatenated GQA Output: {O_expected.numpy()}")
    assert torch.allclose(O_concat, O_expected), "GQA hand output mismatch!"
    print("✓ Part 5 Visual Grid GQA hand calculations verified!\n")

# ----------------------------------------------------------------------
# 2. Grouped-Query Attention (GQA) & KV-Cache Implementation
# ----------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        """
        x: (B, T, d_model)
        kv_cache: tuple of (K_cache, V_cache) each of shape (B, num_kv_heads, T_past, d_k)
        """
        B, T, _ = x.shape
        
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)       # (B, H_q, T, d_k)
        k_new = self.W_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2) # (B, H_kv, T, d_k)
        v_new = self.W_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2) # (B, H_kv, T, d_k)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k_new], dim=2)
            v = torch.cat([v_cache, v_new], dim=2)
        else:
            k, v = k_new, v_new
            
        new_cache = (k, v)
        
        # Repeat/expand KV heads to match Q heads
        # (B, H_kv, 1, T_total, d_k) -> (B, H_kv, num_queries_per_kv, T_total, d_k) -> (B, H_q, T_total, d_k)
        T_total = k.shape[2]
        k_expanded = k.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1).contiguous().view(B, self.num_heads, T_total, self.d_k)
        v_expanded = v.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1).contiguous().view(B, self.num_heads, T_total, self.d_k)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / np.sqrt(self.d_k) # (B, H_q, T, T_total)
        
        # If evaluating sequence without cache or if causal is required:
        # In cached generation, q is (B, H_q, 1, d_k), so it attends to all T_total past tokens (no future tokens exist in cache).
        # In non-cached generation with T tokens, we apply a causal mask:
        if T == T_total and T > 1:
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), dtype=scores.dtype, device=scores.device), diagonal=1)
            scores = scores + causal_mask
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_expanded) # (B, H_q, T, d_k)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), new_cache

def test_gqa_and_kv_cache():
    print("--- 2. GQA with KV-Cache vs Non-Cached Generation Parity ---")
    B, d_model = 2, 32
    num_heads, num_kv_heads = 8, 2 # 4 query heads per KV group
    
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads).to(torch.float64)
    
    # Sequence of 5 tokens
    x_full = torch.randn(B, 5, d_model, dtype=torch.float64)
    
    # 1. Non-cached full forward pass (with causal masking)
    out_full, _ = gqa(x_full)
    
    # 2. Step-by-step cached generation
    # Token 0
    out_0, cache = gqa(x_full[:, 0:1])
    # Token 1
    out_1, cache = gqa(x_full[:, 1:2], kv_cache=cache)
    # Token 2
    out_2, cache = gqa(x_full[:, 2:3], kv_cache=cache)
    # Token 3
    out_3, cache = gqa(x_full[:, 3:4], kv_cache=cache)
    # Token 4
    out_4, cache = gqa(x_full[:, 4:5], kv_cache=cache)
    
    out_cached = torch.cat([out_0, out_1, out_2, out_3, out_4], dim=1)
    
    discrepancy = torch.max(torch.abs(out_full - out_cached)).item()
    print(f"Max Discrepancy (Cached vs Non-Cached Causal): {discrepancy:.2e}")
    assert discrepancy < 1e-12, "KV-Cache output deviates from full attention!"
    print("✓ GQA with incremental KV-Cache exactly matches full non-cached causal attention (< 1e-12)!\n")

# ----------------------------------------------------------------------
# 3. Online Softmax / FlashAttention Tiling Algorithm
# ----------------------------------------------------------------------
def flash_attention_online_softmax(Q, K, V, block_size=4):
    """
    Pure Python implementation of FlashAttention Online Softmax tiling.
    Q, K, V: (T, d) in float64
    """
    T, d = Q.shape
    d_scale = 1.0 / np.sqrt(d)
    
    O = torch.zeros_like(Q)
    l = torch.zeros(T, dtype=torch.float64) # running sum of exponentials
    m = torch.full((T,), float('-inf'), dtype=torch.float64) # running max
    
    num_blocks = (T + block_size - 1) // block_size
    
    for j in range(num_blocks):
        k_start = j * block_size
        k_end = min(k_start + block_size, T)
        
        K_j = K[k_start:k_end] # (B_c, d)
        V_j = V[k_start:k_end] # (B_c, d)
        
        for i in range(num_blocks):
            q_start = i * block_size
            q_end = min(q_start + block_size, T)
            
            Q_i = Q[q_start:q_end] # (B_r, d)
            
            # Local unnormalized scores for block (B_r, B_c)
            S_ij = torch.matmul(Q_i, K_j.t()) * d_scale
            
            # Local row-wise max
            m_ij, _ = torch.max(S_ij, dim=-1)
            
            # New running max
            m_new = torch.maximum(m[q_start:q_end], m_ij)
            
            # Rescale previous sum and compute new block sum
            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
            l_block = torch.sum(P_ij, dim=-1)
            
            l_new = torch.exp(m[q_start:q_end] - m_new) * l[q_start:q_end] + l_block
            
            # Update output: O = (O * l_old * exp(m_old - m_new) + P_ij * V_j) / l_new
            O_prev = O[q_start:q_end]
            alpha_prev = (torch.exp(m[q_start:q_end] - m_new) * l[q_start:q_end]).unsqueeze(-1)
            O_new = (O_prev * alpha_prev + torch.matmul(P_ij, V_j)) / l_new.unsqueeze(-1)
            
            O[q_start:q_end] = O_new
            l[q_start:q_end] = l_new
            m[q_start:q_end] = m_new
            
    return O

def test_online_softmax():
    print("--- 3. FlashAttention Online Softmax Tiling Verification ---")
    T, d = 16, 8
    torch.manual_seed(42)
    Q = torch.randn(T, d, dtype=torch.float64)
    K = torch.randn(T, d, dtype=torch.float64)
    V = torch.randn(T, d, dtype=torch.float64)
    
    # 1. Standard full-matrix attention
    scores_full = torch.matmul(Q, K.t()) / np.sqrt(d)
    attn_full = F.softmax(scores_full, dim=-1)
    O_full = torch.matmul(attn_full, V)
    
    # 2. Tiled online softmax with block_size = 4
    O_online = flash_attention_online_softmax(Q, K, V, block_size=4)
    
    discrepancy = torch.max(torch.abs(O_full - O_online)).item()
    print(f"Max Discrepancy (Standard Attention vs Online Softmax): {discrepancy:.2e}")
    assert discrepancy < 1e-12, "Online softmax deviated from standard attention!"
    print("✓ Online Softmax (FlashAttention tiling) exact numerical parity confirmed (< 1e-12)!\n")

# ----------------------------------------------------------------------
# 4. Sparse Mixture of Experts (MoE) Implementation
# ----------------------------------------------------------------------
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        B, T, d = x.shape
        x_flat = x.view(-1, d) # (N, d)
        
        # Gating router logits
        logits = self.router(x_flat) # (N, E)
        
        # Top-k selection
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1) # (N, k)
        topk_weights = F.softmax(topk_logits, dim=-1)                      # (N, k)
        
        # Compute auxiliary load-balancing loss: aux = E * sum(f_i * P_i)
        router_probs = F.softmax(logits, dim=-1) # (N, E)
        P_i = router_probs.mean(dim=0)           # (E,)
        
        mask = torch.zeros_like(logits).scatter_(1, topk_indices, 1.0)
        f_i = mask.mean(dim=0)                   # (E,)
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        
        # Dispatch to experts
        y_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                expert_mask = (topk_indices[:, k] == e)
                if expert_mask.any():
                    tokens = x_flat[expert_mask]
                    weight = topk_weights[expert_mask, k].unsqueeze(-1)
                    y_flat[expert_mask] += weight * self.experts[e](tokens)
                    
        return y_flat.view(B, T, d), aux_loss

def test_sparse_moe():
    print("--- 4. Sparse Mixture-of-Experts (MoE) Verification ---")
    B, T, d_model, d_ff = 2, 8, 16, 32
    moe = SparseMoELayer(d_model=d_model, d_ff=d_ff, num_experts=4, top_k=2)
    
    x = torch.randn(B, T, d_model, requires_grad=True)
    out, aux_loss = moe(x)
    
    assert out.shape == (B, T, d_model), f"Output shape mismatch: {out.shape}"
    assert aux_loss.item() > 0.0, "Auxiliary load loss must be positive!"
    
    total_loss = out.sum() + 0.1 * aux_loss
    total_loss.backward()
    
    assert x.grad is not None and moe.router.weight.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"MoE Output Shape: {out.shape} | Auxiliary Load Loss: {aux_loss.item():.4f}")
    print("✓ Sparse MoE forward, top-2 routing, and load-balancing loss verified!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 9.4 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_gqa_and_kv_cache()
    test_online_softmax()
    test_sparse_moe()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 9.4 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
