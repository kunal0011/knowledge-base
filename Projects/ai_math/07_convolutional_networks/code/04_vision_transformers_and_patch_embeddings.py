"""
Chapter 7.4: Vision Transformers (ViT) & Patch Embeddings
=========================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Verification (Hand arithmetic vs Code)
2. Mathematical equivalence: Conv2d patch projection vs Unfold + Linear GEMM
3. Full VisionTransformer forward & backward pass
4. 2D Bicubic position embedding interpolation test (224x224 -> 384x384)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid Hand Arithmetic Verification
# ----------------------------------------------------------------------
def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid Verification ---")
    # 4x4 image
    X = torch.tensor([
        [1.0, 2.0, 0.0, 1.0],
        [3.0, 0.0, 2.0, 1.0],
        [1.0, 1.0, 4.0, 0.0],
        [0.0, 2.0, 1.0, 3.0]
    ], dtype=torch.float64)
    
    # Extract 2x2 patches manually
    p1 = X[0:2, 0:2].reshape(-1) # [1, 2, 3, 0]
    p2 = X[0:2, 2:4].reshape(-1) # [0, 1, 2, 1]
    p3 = X[2:4, 0:2].reshape(-1) # [1, 1, 0, 2]
    p4 = X[2:4, 2:4].reshape(-1) # [4, 0, 1, 3]
    
    x_p = torch.stack([p1, p2, p3, p4]) # (4, 4)
    x_p_expected = torch.tensor([
        [1.0, 2.0, 3.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 0.0, 2.0],
        [4.0, 0.0, 1.0, 3.0]
    ], dtype=torch.float64)
    assert torch.allclose(x_p, x_p_expected), "Flattened patch extraction mismatch!"
    
    # Projection matrix E (4, 2)
    E = torch.tensor([
        [ 1.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ], dtype=torch.float64)
    
    # Projected patches x_p * E
    proj = torch.matmul(x_p, E)
    proj_expected = torch.tensor([
        [ 4.0, -1.0],
        [ 1.0,  0.0],
        [-1.0,  3.0],
        [ 2.0,  2.0]
    ], dtype=torch.float64)
    assert torch.allclose(proj, proj_expected), "Patch projection mismatch!"
    
    # Prepend [CLS] token [0.0, 1.0]
    cls_token = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    z_unposed = torch.cat([cls_token, proj], dim=0) # (5, 2)
    
    # Position embeddings (5, 2)
    E_pos = torch.tensor([
        [ 0.5,  0.5],
        [ 0.1, -0.1],
        [-0.2,  0.2],
        [ 0.3, -0.3],
        [-0.4,  0.4]
    ], dtype=torch.float64)
    
    z_0 = z_unposed + E_pos
    z_0_expected = torch.tensor([
        [ 0.5,  1.5],
        [ 4.1, -1.1],
        [ 0.8,  0.2],
        [-0.7,  2.7],
        [ 1.6,  2.4]
    ], dtype=torch.float64)
    
    print("Computed Embedded Sequence z_0:\n", z_0.numpy())
    assert torch.allclose(z_0, z_0_expected), "Embedded sequence mismatch!"
    print("✓ Part 5 Visual Grid ViT sequence embedding hand arithmetic verified!\n")

# ----------------------------------------------------------------------
# 2. Mathematical Equivalence: Conv2d vs Unfold + Linear Patch Projection
# ----------------------------------------------------------------------
def test_patch_projection_equivalence():
    print("--- 2. Conv2d vs Unfold + Linear Patch Projection Equivalence ---")
    B, C, H, W = 2, 3, 32, 32
    P = 4
    D = 64
    
    x = torch.randn(B, C, H, W, dtype=torch.float64)
    
    # 1. Method A: 2D Convolution
    conv_proj = nn.Conv2d(C, D, kernel_size=P, stride=P, bias=True).to(torch.float64)
    y_conv = conv_proj(x) # (B, D, H/P, W/P)
    y_conv_flat = y_conv.flatten(2).transpose(1, 2) # (B, N, D)
    
    # 2. Method B: Unfold + Linear with matching weights
    # Unfold extracts sliding patches of shape (B, C * P * P, N)
    unfold = nn.Unfold(kernel_size=P, stride=P)
    x_unfold = unfold(x).transpose(1, 2) # (B, N, C * P * P)
    
    # Weight of conv is (D, C, P, P) -> flatten to (D, C * P * P)
    W_flat = conv_proj.weight.reshape(D, -1)
    b_flat = conv_proj.bias
    
    y_unfold = torch.matmul(x_unfold, W_flat.t()) + b_flat # (B, N, D)
    
    discrepancy = torch.max(torch.abs(y_conv_flat - y_unfold)).item()
    print(f"Max Absolute Discrepancy (Conv2d vs Unfold+Linear): {discrepancy:.2e}")
    assert discrepancy < 1e-12, "Patch projection equivalence failed!"
    print("✓ Strict mathematical equivalence of Conv2d and Unfold+Linear patch projection confirmed!\n")

# ----------------------------------------------------------------------
# 3. Full Vision Transformer Implementation & Verification
# ----------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        # Pre-LN MSA
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=64, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, N, D)
        
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, D)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        # Classify based solely on [CLS] token at index 0
        logits = self.head(x[:, 0])
        return logits

def test_vit_pipeline():
    print("--- 3. Full VisionTransformer Forward & Backward Pass ---")
    model = SimpleViT(img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=64, depth=2, num_heads=4)
    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    
    logits = model(x)
    assert logits.shape == (4, 10), f"Logits shape mismatch: {logits.shape}"
    
    loss = logits.sum()
    loss.backward()
    
    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert model.cls_token.grad is not None
    assert model.pos_embed.grad is not None
    print(f"ViT Output Logits Shape: {logits.shape} | Loss Gradient Norm: {x.grad.norm().item():.4f}")
    print("✓ Full VisionTransformer forward and backward gradients strictly verified!\n")

# ----------------------------------------------------------------------
# 4. 2D Bicubic Position Embedding Interpolation Test
# ----------------------------------------------------------------------
def interpolate_pos_embed(pos_embed, new_grid_size):
    """
    pos_embed: (1, N + 1, D) where N = old_grid_size^2
    new_grid_size: int or tuple (H_new, W_new)
    """
    cls_token = pos_embed[:, :1]
    spatial_pos = pos_embed[:, 1:]
    
    N, D = spatial_pos.shape[1], spatial_pos.shape[2]
    old_grid = int(N ** 0.5)
    assert old_grid * old_grid == N, "Spatial patches must form a square grid!"
    
    # Reshape (1, old_grid, old_grid, D) -> (1, D, old_grid, old_grid)
    spatial_pos = spatial_pos.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    
    # Bicubic interpolation
    new_h, new_w = (new_grid_size, new_grid_size) if isinstance(new_grid_size, int) else new_grid_size
    interpolated = F.interpolate(spatial_pos, size=(new_h, new_w), mode='bicubic', align_corners=False)
    
    # Permute back: (1, D, new_h, new_w) -> (1, new_h * new_w, D)
    interpolated = interpolated.permute(0, 2, 3, 1).flatten(1, 2)
    
    new_pos_embed = torch.cat((cls_token, interpolated), dim=1)
    return new_pos_embed

def test_pos_embed_interpolation():
    print("--- 4. 2D Bicubic Position Embedding Interpolation Test ---")
    D = 64
    old_grid = 14 # 14x14 = 196 patches (224x224 with P=16)
    new_grid = 24 # 24x24 = 576 patches (384x384 with P=16)
    
    pos_embed_orig = torch.randn(1, 196 + 1, D)
    pos_embed_new = interpolate_pos_embed(pos_embed_orig, new_grid)
    
    print(f"Original pos_embed shape: {pos_embed_orig.shape}")
    print(f"Interpolated pos_embed shape: {pos_embed_new.shape}")
    
    assert pos_embed_new.shape == (1, 576 + 1, D), f"Interpolated shape mismatch: {pos_embed_new.shape}"
    # Check that [CLS] token is preserved exactly
    assert torch.allclose(pos_embed_new[:, :1], pos_embed_orig[:, :1]), "[CLS] position token was modified!"
    print("✓ 2D Bicubic position embedding interpolation confirmed!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 7.4 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_patch_projection_equivalence()
    test_vit_pipeline()
    test_pos_embed_interpolation()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 7.4 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
