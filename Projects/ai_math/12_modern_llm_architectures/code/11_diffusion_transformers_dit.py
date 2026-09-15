"""
Diffusion Transformers (DiT), Patchification & adaLN-Zero Conditioning
Implementation and Verification Suite
Module 12: Modern LLM Architectures & Engineering from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    x = torch.tensor([2.0, 4.0], dtype=torch.float32)
    
    # Step 1: LayerNorm statistics
    mean = x.mean()
    std = torch.sqrt(torch.mean((x - mean) ** 2)) # population std for d=2
    assert math.isclose(mean.item(), 3.000000, abs_tol=1e-5)
    assert math.isclose(std.item(), 1.000000, abs_tol=1e-5)
    
    x_norm = (x - mean) / std
    expected_x_norm = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    assert torch.allclose(x_norm, expected_x_norm, atol=1e-5)
    print(f"LayerNorm normalized: {x_norm.tolist()} -> EXACT MATCH [-1.0, 1.0]")
    
    # Step 2: adaLN Modulation
    gamma1 = torch.tensor([0.20, -0.10], dtype=torch.float32)
    beta1 = torch.tensor([0.50, 0.50], dtype=torch.float32)
    alpha1 = torch.tensor([0.10, 0.10], dtype=torch.float32)
    
    x_hat = (1.0 + gamma1) * x_norm + beta1
    expected_x_hat = torch.tensor([-0.700000, 1.400000], dtype=torch.float32)
    assert torch.allclose(x_hat, expected_x_hat, atol=1e-5)
    print(f"Modulated input x_hat: {x_hat.tolist()} -> EXACT MATCH [-0.70, 1.40]")
    
    # Step 3: Gated Attention Update
    mha_out = torch.tensor([3.0, -1.0], dtype=torch.float32)
    delta_x = alpha1 * mha_out
    x_prime = x + delta_x
    expected_x_prime = torch.tensor([2.300000, 3.900000], dtype=torch.float32)
    assert torch.allclose(x_prime, expected_x_prime, atol=1e-5)
    print(f"Gated block output x_prime: {x_prime.tolist()} -> EXACT MATCH [2.30, 3.90]")
    
    # Step 4: Zero-Init Identity Check
    alpha_zero = torch.tensor([0.0, 0.0], dtype=torch.float32)
    x_init = x + alpha_zero * mha_out
    assert torch.equal(x_init, x), "Zero-init identity check failed!"
    print("Zero-init identity preservation verified: x_init == x exactly.\n")


def patchify(z: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Splits (B, C, H, W) into (B, T, patch_size * patch_size * C).
    """
    B, C, H, W = z.shape
    p = patch_size
    assert H % p == 0 and W % p == 0
    h, w = H // p, W // p
    
    # Reshape into patches
    z = z.view(B, C, h, p, w, p)
    # Permute to (B, h, w, p, p, C)
    z = z.permute(0, 2, 4, 3, 5, 1).contiguous()
    # Flatten to (B, T, d_patch)
    patches = z.view(B, h * w, p * p * C)
    return patches


def unpatchify(patches: torch.Tensor, C: int, H: int, W: int, patch_size: int) -> torch.Tensor:
    """
    Reconstructs (B, C, H, W) from (B, T, patch_size * patch_size * C).
    """
    B, T, d_patch = patches.shape
    p = patch_size
    h, w = H // p, W // p
    assert T == h * w
    assert d_patch == p * p * C
    
    patches = patches.view(B, h, w, p, p, C)
    patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = patches.view(B, C, H, W)
    return z


def verify_patchification_invertibility():
    print("--- 2. Verifying Patchification & Unpatchification Invertibility ---")
    B, C, H, W = 2, 4, 32, 32
    p = 2
    z = torch.randn(B, C, H, W)
    
    patches = patchify(z, patch_size=p)
    expected_T = (H // p) * (W // p) # 16 * 16 = 256
    expected_dim = C * p * p         # 4 * 4 = 16
    assert patches.shape == (B, expected_T, expected_dim)
    
    z_reconstructed = unpatchify(patches, C, H, W, patch_size=p)
    assert torch.allclose(z, z_reconstructed, atol=1e-6), "Unpatchify failed to perfectly invert patchify!"
    print(f"Latent shape: {z.shape} -> Patches: {patches.shape} -> Reconstructed: {z_reconstructed.shape}")
    print("Exact invertible reconstruction verified!\n")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with adaLN-Zero conditioning.
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
        # Regresses 6 parameters: gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        # ZERO INITIALIZATION
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) visual patch tokens
            c: (B, D) conditioning vector (timestep + class embedding)
        """
        mod_params = self.adaLN_modulation(c) # (B, 6D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.chunk(6, dim=-1)
        
        # Modulated Self-Attention with gate
        norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm1, norm1, norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Modulated MLP with gate
        norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


def test_dit_block():
    print("--- 3. Testing PyTorch DiTBlock with adaLN-Zero ---")
    torch.manual_seed(42)
    B, T, D = 2, 16, 64
    num_heads = 4
    
    block = DiTBlock(hidden_dim=D, num_heads=num_heads)
    
    x = torch.randn(B, T, D, requires_grad=True)
    c = torch.randn(B, D, requires_grad=True)
    
    # 1. Zero-init check: Forward pass at initialization must equal input x exactly!
    out_init = block(x, c)
    assert torch.allclose(out_init, x, atol=1e-6), "DiTBlock failed zero-init identity test!"
    print("Theorem verified: Initialized DiTBlock is an exact identity mapping (out == x).")
    
    # 2. Backward pass gradient check
    # Perturb modulation weights slightly to simulate training
    nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
    out_trained = block(x, c)
    loss = out_trained.sum()
    loss.backward()
    
    assert x.grad is not None, "Input x did not receive gradients"
    assert c.grad is not None, "Conditioning vector c did not receive gradients"
    assert block.adaLN_modulation[-1].weight.grad is not None
    print("Backward pass verified: Gradients successfully propagated through adaLN-Zero paths.")
    print("All DiT tests passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_patchification_invertibility()
    test_dit_block()
    print("🟢 Chapter 12.11 Verification 100% Complete.")
