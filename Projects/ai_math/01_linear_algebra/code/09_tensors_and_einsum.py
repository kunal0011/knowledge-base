"""
Chapter 1.9: Tensors, Contractions & Einstein Summation Notation
================================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. The Einsum Rosetta Stone: Verifying All 8 Canonical Primitives
2. Pure Einsum Multi-Head Attention (MHA) Module in PyTorch
3. Part 5 By-Hand Numerical Reproduction (Q, K, Attention Scores, and Context Vector)
4. Bilinear Pooling Contraction in Vision-Language Models
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict


# =====================================================================
# 1. The Einsum Rosetta Stone Suite
# =====================================================================
def verify_einsum_rosetta_stone() -> Dict[str, bool]:
    """
    Verifies that einsum implementations of standard tensor operations
    match their canonical NumPy / PyTorch equivalents bit-for-bit.
    """
    results = {}

    # 1. Vector Dot Product ('i, i ->')
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    res_dot = np.einsum('i, i ->', u, v)
    expected_dot = np.dot(u, v)
    results["dot_product"] = np.isclose(res_dot, expected_dot)

    # 2. Vector Outer Product ('i, j -> i j')
    res_outer = np.einsum('i, j -> i j', u, v)
    expected_outer = np.outer(u, v)
    results["outer_product"] = np.allclose(res_outer, expected_outer)

    # 3. Matrix Multiplication ('i k, k j -> i j')
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    res_mm = np.einsum('i k, k j -> i j', A, B)
    expected_mm = A @ B
    results["matrix_multiply"] = np.allclose(res_mm, expected_mm)

    # 4. Matrix Trace ('i i ->')
    res_trace = np.einsum('i i ->', A)
    expected_trace = np.trace(A)
    results["trace"] = np.isclose(res_trace, expected_trace)

    # 5. Extract Diagonal ('i i -> i')
    res_diag = np.einsum('i i -> i', A)
    expected_diag = np.diag(A)
    results["extract_diagonal"] = np.allclose(res_diag, expected_diag)

    # 6. Matrix Transpose ('i j -> j i')
    res_t = np.einsum('i j -> j i', A)
    expected_t = A.T
    results["transpose"] = np.allclose(res_t, expected_t)

    # 7. Batched Matrix Multiply in PyTorch ('b i k, b k j -> b i j')
    A_batch = torch.randn(4, 5, 6)
    B_batch = torch.randn(4, 6, 7)
    res_bmm = torch.einsum('b i k, b k j -> b i j', A_batch, B_batch)
    expected_bmm = torch.bmm(A_batch, B_batch)
    results["batch_matmul"] = torch.allclose(res_bmm, expected_bmm, atol=1e-5)

    return results


# =====================================================================
# 2. Pure Einsum Multi-Head Attention Module in PyTorch
# =====================================================================
class EinsumMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implemented cleanly using torch.einsum:
      1. Scores = einsum('b h i d, b h j d -> b h i j', Q, K) / sqrt(d_k)
      2. Output = einsum('b h i j, b h j d -> b h i d', weights, V)
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads!"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, d_model]
        Returns:
          out: [B, T, d_model]
          weights: [B, num_heads, T, T]
        """
        B, T, _ = x.shape
        H, d_k = self.num_heads, self.head_dim

        # Project and reshape into heads: [B, T, d_model] -> [B, H, T, d_k]
        Q = self.q_proj(x).view(B, T, H, d_k).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, T, H, d_k).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, T, H, d_k).permute(0, 2, 1, 3)

        # 1. Attention Logits via einsum
        scores = torch.einsum('b h i d, b h j d -> b h i j', Q, K) / np.sqrt(d_k)

        # Causal mask (optional)
        if causal_mask:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(~mask, -1e9)

        # 2. Softmax normalization
        weights = torch.softmax(scores, dim=-1)

        # 3. Context pooling via einsum
        context = torch.einsum('b h i j, b h j d -> b h i d', weights, V)

        # Reshape back: [B, H, T, d_k] -> [B, T, d_model]
        context = context.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        out = self.out_proj(context)

        return out, weights


# =====================================================================
# 3. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.9: Tensors, Contractions & Einsum — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Einsum Rosetta Stone Suite
    # -------------------------------------------------------------
    print("\n[Test 1: Einsum Rosetta Stone Verification]")
    rosetta_results = verify_einsum_rosetta_stone()
    for name, passed in rosetta_results.items():
        print(f"  {name:<22}: Match? {passed}")
        assert passed == True, f"Einsum mismatch for {name}!"

    # -------------------------------------------------------------
    # Test 2: Part 5 AI by Hand Numerical Reproduction
    # Q = [[1, 2], [3, 0]], K = [[2, 1], [4, -1]]
    # Expected Raw Scores S = [[4, 2], [6, 12]]
    # -------------------------------------------------------------
    print("\n[Test 2: Part 5 Attention Scores by Hand]")
    Q_toy = torch.tensor([[[[1.0, 2.0], [3.0, 0.0]]]])    # [1, 1, 2, 2]
    K_toy = torch.tensor([[[[2.0, 1.0], [4.0, -1.0]]]])   # [1, 1, 2, 2]
    d_k = 2.0

    raw_scores = torch.einsum('b h i d, b h j d -> b h i j', Q_toy, K_toy)
    expected_raw = torch.tensor([[[[4.0, 2.0], [6.0, 12.0]]]])

    print(f"  Raw Attention Scores S:\n{raw_scores[0, 0].numpy()}")
    print(f"  Matches Hand Derivation? {torch.allclose(raw_scores, expected_raw)}")
    assert torch.allclose(raw_scores, expected_raw), "Raw attention scores mismatch!"

    # Scaled scores
    scaled_scores = raw_scores / np.sqrt(d_k)
    weights = torch.softmax(scaled_scores, dim=-1)

    expected_weights = torch.tensor([[[[0.8044, 0.1956], [0.0142, 0.9858]]]])
    print(f"  Softmax Weights:\n{weights[0, 0].numpy()}")
    print(f"  Matches Hand Derivation? {torch.allclose(weights, expected_weights, atol=1e-3)}")
    assert torch.allclose(weights, expected_weights, atol=1e-3), "Softmax weights mismatch!"

    # Value context aggregation
    V_toy = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])  # [1, 1, 2, 2]
    context = torch.einsum('b h i j, b h j d -> b h i d', weights, V_toy)

    expected_context = torch.tensor([[[[8.044, 3.912], [0.142, 19.716]]]])
    print(f"  Aggregated Context Output:\n{context[0, 0].numpy()}")
    print(f"  Matches Hand Derivation? {torch.allclose(context, expected_context, atol=1e-2)}")
    assert torch.allclose(context, expected_context, atol=1e-2), "Context output mismatch!"

    # -------------------------------------------------------------
    # Test 3: PyTorch Einsum MHA Module Forward Pass
    # -------------------------------------------------------------
    print("\n[Test 3: EinsumMultiHeadAttention Module Test]")
    torch.manual_seed(42)
    B, T, d_model, heads = 2, 8, 64, 4
    mha = EinsumMultiHeadAttention(d_model=d_model, num_heads=heads)

    x_tokens = torch.randn(B, T, d_model)
    out_tokens, attn_weights = mha(x_tokens, causal_mask=True)

    print(f"  Input Tokens Shape:   {x_tokens.shape}")
    print(f"  Output Tokens Shape:  {out_tokens.shape}")
    print(f"  Attention Weights:    {attn_weights.shape} (B, H, T, T)")
    print(f"  Weights Sum to 1.0?   {torch.allclose(torch.sum(attn_weights, dim=-1), torch.ones(B, heads, T))}")

    # Check causal masking (upper triangle should be 0.0)
    upper_tri = torch.triu(attn_weights[0, 0], diagonal=1)
    is_causal = torch.all(upper_tri == 0.0).item()
    print(f"  Causal Mask Enforced? {is_causal}")

    assert out_tokens.shape == (B, T, d_model), "MHA output shape mismatch!"
    assert is_causal == True, "Causal attention masking failed!"

    # -------------------------------------------------------------
    # Test 4: Multimodal Bilinear Pooling Contraction
    # Visual feats: [B, D_v], Text feats: [B, D_t], Weight: [D_out, D_v, D_t]
    # 'b v, o v t, b t -> b o'
    # -------------------------------------------------------------
    print("\n[Test 4: Multimodal Bilinear Pooling Contraction]")
    B, D_v, D_t, D_out = 4, 16, 24, 8
    v_feat = torch.randn(B, D_v)
    t_feat = torch.randn(B, D_t)
    W_bilinear = torch.randn(D_out, D_v, D_t)

    # Einsum contraction
    pooled = torch.einsum('b v, o v t, b t -> b o', v_feat, W_bilinear, t_feat)
    print(f"  Visual Feat Shape:  {v_feat.shape}")
    print(f"  Text Feat Shape:    {t_feat.shape}")
    print(f"  3D Tensor Shape:    {W_bilinear.shape}")
    print(f"  Bilinear Output:    {pooled.shape} (Expected: [4, 8])")

    assert pooled.shape == (B, D_out), "Bilinear pooling shape mismatch!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
