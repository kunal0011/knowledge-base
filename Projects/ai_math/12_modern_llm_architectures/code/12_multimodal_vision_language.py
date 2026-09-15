"""
Multimodal Vision-Language Architectures (MLP Projectors & Perceiver Resampler)
Implementation and Verification Suite
Module 12: Modern LLM Architectures & Engineering from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    Z_v = torch.tensor([
        [1.0, 2.0],
        [0.0, 1.0]
    ], dtype=torch.float32)
    
    W1 = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float32)
    b1 = torch.zeros(2, dtype=torch.float32)
    
    W2 = torch.tensor([
        [ 0.5, 1.0],
        [-0.5, 0.0]
    ], dtype=torch.float32)
    b2 = torch.zeros(2, dtype=torch.float32)
    
    # Step 1: Layer 1
    U = F.relu(torch.matmul(Z_v, W1) + b1)
    assert torch.allclose(U, Z_v, atol=1e-5)
    print(f"Layer 1 intermediate U:\n{U.numpy()} -> EXACT MATCH")
    
    # Step 2: Layer 2
    H_v = torch.matmul(U, W2) + b2
    expected_H_v = torch.tensor([
        [-0.5, 1.0],
        [-0.5, 0.0]
    ], dtype=torch.float32)
    assert torch.allclose(H_v, expected_H_v, atol=1e-5)
    print(f"Projected vision tokens H_v:\n{H_v.numpy()} -> EXACT MATCH")
    
    # Step 3: Prefix concatenation with text prompt
    H_t = torch.tensor([[3.0, -1.0]], dtype=torch.float32)
    H_seq = torch.cat([H_v, H_t], dim=0) # (3, 2)
    expected_H_seq = torch.tensor([
        [-0.5,  1.0],
        [-0.5,  0.0],
        [ 3.0, -1.0]
    ], dtype=torch.float32)
    assert torch.allclose(H_seq, expected_H_seq, atol=1e-5)
    print(f"Concatenated multimodal sequence H_seq (3 tokens):\n{H_seq.numpy()} -> EXACT MATCH")
    
    # Step 4: 3x3 Causal Mask check
    causal_mask = torch.triu(torch.full((3, 3), float('-inf')), diagonal=1)
    assert causal_mask[2, 0] == 0.0 and causal_mask[2, 1] == 0.0 and causal_mask[2, 2] == 0.0
    print("Causal attention mask verified: Text token causally attends to all vision patches.\n")


def verify_illustration_compression():
    print("--- 2. Verifying Illustration 1 Context Window Impact ---")
    H, W, p = 672, 672, 14
    tokens_vit = (H // p) * (W // p) # 48 * 48 = 2304
    assert tokens_vit == 2304
    
    K = 64
    compression_ratio = tokens_vit / K # 36.0
    assert math.isclose(compression_ratio, 36.0, abs_tol=1e-5)
    
    ctx = 4096
    rem_vit = ctx - tokens_vit # 1792
    rem_perceiver = ctx - K    # 4032
    
    pct_vit = (rem_vit / ctx) * 100.0
    pct_perceiver = (rem_perceiver / ctx) * 100.0
    assert math.isclose(pct_vit, 43.75, abs_tol=1e-3)
    assert math.isclose(pct_perceiver, 98.4375, abs_tol=1e-3)
    print(f"ViT tokens: {tokens_vit} -> Context remaining: {rem_vit} tokens ({pct_vit:.2f}%)")
    print(f"Perceiver Resampler tokens: {K} -> Context remaining: {rem_perceiver} tokens ({pct_perceiver:.2f}%) -> EXACT MATCH\n")


class MLPProjector(nn.Module):
    """LLaVA-style 2-layer MLP multimodal projector with GELU."""
    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_dim, text_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(text_dim, text_dim)
        
    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(z_v)))


class PerceiverResampler(nn.Module):
    """
    Flamingo-style Perceiver Resampler.
    Uses K learnable query vectors to cross-attend to arbitrary visual tokens.
    """
    def __init__(self, num_queries: int, vision_dim: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.latents = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.proj_k = nn.Linear(vision_dim, embed_dim)
        self.proj_v = nn.Linear(vision_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_v: (B, N_v, vision_dim)
        Returns:
            bottleneck_tokens: (B, K, embed_dim)
        """
        B = z_v.shape[0]
        # Expand queries across batch
        q = self.latents.unsqueeze(0).expand(B, -1, -1) # (B, K, embed_dim)
        k = self.proj_k(z_v) # (B, N_v, embed_dim)
        v = self.proj_v(z_v) # (B, N_v, embed_dim)
        
        out, _ = self.attn(q, k, v)
        return self.norm(out + q)


def test_multimodal_pipeline():
    print("--- 3. Testing Full Multimodal Vision-Language Forward & Backward Pipeline ---")
    torch.manual_seed(42)
    B = 2
    N_v = 16 # Visual patches
    N_t = 8  # Text tokens
    d_v = 32 # Vision encoder dim
    d_t = 64 # LLM embedding dim
    
    # 1. Projector Test
    projector = MLPProjector(vision_dim=d_v, text_dim=d_t)
    z_v = torch.randn(B, N_v, d_v, requires_grad=True)
    h_v = projector(z_v)
    assert h_v.shape == (B, N_v, d_t), f"Projector shape mismatch: {h_v.shape}"
    print(f"MLP Projector output shape: {h_v.shape} (Aligned with text dim {d_t})")
    
    # 2. Sequence Concatenation
    h_t = torch.randn(B, N_t, d_t, requires_grad=True)
    h_seq = torch.cat([h_v, h_t], dim=1) # (B, N_v + N_t, d_t)
    assert h_seq.shape == (B, N_v + N_t, d_t)
    
    # 3. Perceiver Resampler Test
    resampler = PerceiverResampler(num_queries=8, vision_dim=d_v, embed_dim=d_t, num_heads=4)
    bottleneck = resampler(z_v)
    assert bottleneck.shape == (B, 8, d_t), f"Resampler shape mismatch: {bottleneck.shape}"
    print(f"Perceiver Resampler bottleneck shape: {bottleneck.shape} (Compressed from {N_v} to 8 tokens)")
    
    # 4. Backward pass gradient check
    loss = h_seq.sum() + bottleneck.sum()
    loss.backward()
    
    assert z_v.grad is not None, "Vision features failed to receive gradients"
    assert h_t.grad is not None, "Text embeddings failed to receive gradients"
    assert projector.linear1.weight.grad is not None
    assert resampler.latents.grad is not None
    print("End-to-end multimodal gradient backpropagation successfully verified.")
    print("All multimodal tests passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_illustration_compression()
    test_multimodal_pipeline()
    print("🟢 Chapter 12.12 Verification 100% Complete.")
