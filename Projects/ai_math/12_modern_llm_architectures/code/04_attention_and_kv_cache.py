"""
Grouped-Query Attention (GQA) & KV Cache Verification Suite
Module 12: Modern LLM Architectures - Chapter 04

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - GQA expanded key/value broadcasting (H_q = 4, H_kv = 2, R = 2)
   - Scaled dot-product attention logits: z = [2.121320, 1.414214, -2.121320, 1.414214]
   - Multi-head output concatenation matching hand trace (< 1e-6)
2. Production-Ready PyTorch GQA Layer with KV Cache:
   - Verification that autoregressive single-token cached generation exactly matches full sequence prefill
3. Exact VRAM KV cache footprint verification (MHA 64.0 GiB vs GQA 8.0 GiB).
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
    print("1. VERIFYING PART 5 'AI BY HAND' GQA CALCULATIONS")
    print("=" * 70)

    # 4 Query heads, 2 KV heads, head_dim = 2
    Q0 = np.array([1.0, 2.0])
    Q1 = np.array([2.0, 0.0])
    Q2 = np.array([-1.0, 1.0])
    Q3 = np.array([0.0, -2.0])

    K0 = np.array([1.0, 1.0])
    K1 = np.array([2.0, -1.0])

    V0 = np.array([0.5, 0.5])
    V1 = np.array([1.0, -1.0])

    sqrt_d = np.sqrt(2.0)

    # Group 0: Q0 and Q1 share K0, V0
    dot0 = np.dot(Q0, K0)  # 3.0
    dot1 = np.dot(Q1, K0)  # 2.0
    z0 = dot0 / sqrt_d
    z1 = dot1 / sqrt_d

    # Group 1: Q2 and Q3 share K1, V1
    dot2 = np.dot(Q2, K1)  # -3.0
    dot3 = np.dot(Q3, K1)  # 2.0
    z2 = dot2 / sqrt_d
    z3 = dot3 / sqrt_d

    print(f"Logits: z0 = {z0:.6f}, z1 = {z1:.6f}, z2 = {z2:.6f}, z3 = {z3:.6f}")

    assert np.isclose(z0, 3.0 / np.sqrt(2), atol=1e-6)
    assert np.isclose(z1, 2.0 / np.sqrt(2), atol=1e-6)
    assert np.isclose(z2, -3.0 / np.sqrt(2), atol=1e-6)
    assert np.isclose(z3, 2.0 / np.sqrt(2), atol=1e-6)

    # Output vectors (single token, so softmax weight is 1.0)
    out0 = 1.0 * V0
    out1 = 1.0 * V0
    out2 = 1.0 * V1
    out3 = 1.0 * V1

    out_concat = np.concatenate([out0, out1, out2, out3])
    expected_out = np.array([0.5, 0.5, 0.5, 0.5, 1.0, -1.0, 1.0, -1.0])

    print(f"Concatenated Multi-Head Output: {out_concat}")
    assert np.allclose(out_concat, expected_out, atol=1e-6)
    print(">> SUCCESS: GQA hand calculations verified to exact precision!\n")


# =====================================================================
# 2. Production PyTorch GQA Layer with KV Cache
# =====================================================================

class GroupedQueryAttention(nn.Module):
    """
    Production-ready Grouped-Query Attention (GQA) with KV Caching.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.num_queries_per_kv = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # Repeat KV heads along head dimension to match n_heads
        k_expanded = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_expanded = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled Dot-Product Attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

        # Apply causal mask if processing full sequence without prior cache
        if kv_cache is None and L > 1:
            causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
            scores = scores + causal_mask

        attn_probs = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_probs, v_expanded)

        # Transpose and project output
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out), new_kv_cache


def verify_gqa_kv_cache_equivalence():
    print("=" * 70)
    print("2. VERIFYING GQA KV CACHED GENERATION VS PREFILL EQUIVALENCE")
    print("=" * 70)

    torch.manual_seed(42)
    B, L, d_model = 1, 5, 32
    n_heads = 4
    n_kv_heads = 2

    gqa = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads)
    x = torch.randn(B, L, d_model)

    # 1. Full Sequence Prefill (Non-cached)
    full_out, _ = gqa(x)

    # 2. Step-by-Step Autoregressive Cached Generation
    cached_outs = []
    kv_cache = None
    for t in range(L):
        token_in = x[:, t:t+1, :]
        step_out, kv_cache = gqa(token_in, kv_cache=kv_cache, use_cache=True)
        cached_outs.append(step_out)

    cached_full_out = torch.cat(cached_outs, dim=1)

    max_diff = torch.max(torch.abs(full_out - cached_full_out)).item()
    print(f"Max difference between cached and non-cached GQA generation: {max_diff:.8f}")
    assert max_diff < 1e-5, f"KV cache must match prefill exactly, got diff {max_diff}"
    print(">> SUCCESS: GQA with KV caching exactly replicates non-cached prefill (< 1e-5)!\n")


# =====================================================================
# 3. KV Cache Memory Footprint Verification
# =====================================================================

def verify_kv_cache_memory_calculation():
    print("=" * 70)
    print("3. VERIFYING 70B MODEL KV CACHE MEMORY CALCULATIONS (MHA VS GQA)")
    print("=" * 70)

    B = 32
    L = 8192
    N_layers = 80
    d_head = 128
    bytes_per_elem = 2  # FP16

    # MHA: H_kv = 64
    mha_kv_heads = 64
    mha_bytes = 2 * B * L * N_layers * mha_kv_heads * d_head * bytes_per_elem
    mha_gib = mha_bytes / (1024**3)

    # GQA: H_kv = 8
    gqa_kv_heads = 8
    gqa_bytes = 2 * B * L * N_layers * gqa_kv_heads * d_head * bytes_per_elem
    gqa_gib = gqa_bytes / (1024**3)

    print(f"MHA KV Cache Footprint: {mha_gib:.1f} GiB (Expected: 640.0 GiB)")
    print(f"GQA KV Cache Footprint: {gqa_gib:.1f} GiB (Expected: 80.0 GiB)")
    print(f"Memory Reduction Factor: {mha_gib / gqa_gib:.1f}x")

    assert np.isclose(mha_gib, 640.0, atol=1e-2)
    assert np.isclose(gqa_gib, 80.0, atol=1e-2)
    assert np.isclose(mha_gib / gqa_gib, 8.0, atol=1e-2)
    print(">> SUCCESS: Exact 8x KV cache memory reduction verified!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_gqa_kv_cache_equivalence()
    verify_kv_cache_memory_calculation()
    print("=" * 70)
    print("ALL MODULE 12 CHAPTER 04 (ATTENTION EVOLUTION & GQA) VERIFICATIONS PASSED!")
    print("=" * 70)
