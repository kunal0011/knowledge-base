"""
Parameter-Efficient Fine-Tuning: LoRA, QLoRA & Weight Merging
Implementation and Verification Suite
Module 12: Modern LLM Architectures & Engineering from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    x = torch.tensor([2.0, 1.0], dtype=torch.float32)
    W0 = torch.tensor([
        [1.0,  2.0],
        [0.0, -1.0]
    ], dtype=torch.float32)
    
    r = 1
    alpha = 2.0
    scaling = alpha / r
    
    A = torch.tensor([[1.0, 0.5]], dtype=torch.float32) # (1, 2)
    B = torch.tensor([[0.5], [-0.5]], dtype=torch.float32) # (2, 1)
    
    # Step 1: Base output
    y0 = torch.matmul(W0, x)
    expected_y0 = torch.tensor([4.0, -1.0], dtype=torch.float32)
    assert torch.allclose(y0, expected_y0, atol=1e-5), f"y0 mismatch: {y0}"
    print(f"Base output y0: {y0.tolist()} -> EXACT MATCH [4.0, -1.0]")
    
    # Step 2: Adapter down-projection u = A x
    u = torch.matmul(A, x) # (1,)
    expected_u = torch.tensor([2.5], dtype=torch.float32)
    assert torch.allclose(u, expected_u, atol=1e-5), f"u mismatch: {u}"
    print(f"Down-projection u: {u.item():.4f} -> EXACT MATCH 2.5000")
    
    # Step 3: Adapter up-projection v = B u
    v = torch.matmul(B, u) # (2,)
    expected_v = torch.tensor([1.25, -1.25], dtype=torch.float32)
    assert torch.allclose(v, expected_v, atol=1e-5), f"v mismatch: {v}"
    print(f"Up-projection v: {v.tolist()} -> EXACT MATCH [1.25, -1.25]")
    
    # Step 4: Scale adapter delta_y = (alpha / r) * v
    delta_y = scaling * v
    expected_delta_y = torch.tensor([2.5, -2.5], dtype=torch.float32)
    assert torch.allclose(delta_y, expected_delta_y, atol=1e-5), f"delta_y mismatch: {delta_y}"
    print(f"Scaled delta_y: {delta_y.tolist()} -> EXACT MATCH [2.5, -2.5]")
    
    # Step 5: Final output y = y0 + delta_y
    y = y0 + delta_y
    expected_y = torch.tensor([6.5, -3.5], dtype=torch.float32)
    assert torch.allclose(y, expected_y, atol=1e-5), f"y mismatch: {y}"
    print(f"Total LoRA output y: {y.tolist()} -> EXACT MATCH [6.5, -3.5]")
    
    # Step 6: Weight merging verification
    delta_W = scaling * torch.matmul(B, A)
    expected_delta_W = torch.tensor([
        [ 1.0,  0.5],
        [-1.0, -0.5]
    ], dtype=torch.float32)
    assert torch.allclose(delta_W, expected_delta_W, atol=1e-5)
    
    W_merged = W0 + delta_W
    expected_W_merged = torch.tensor([
        [ 2.0,  2.5],
        [-1.0, -1.5]
    ], dtype=torch.float32)
    assert torch.allclose(W_merged, expected_W_merged, atol=1e-5)
    
    y_from_merged = torch.matmul(W_merged, x)
    assert torch.allclose(y_from_merged, expected_y, atol=1e-5)
    print(f"Fused weight matrix W_merged:\n{W_merged.numpy()}")
    print(f"Fused forward W_merged * x: {y_from_merged.tolist()} -> ZERO ERROR EQUIVALENCE!\n")


def verify_memory_illustration():
    print("--- 2. Verifying Illustration 1 Memory Footprints (70B Model) ---")
    N = 70.0 # Billion parameters
    
    # Full Fine-Tuning
    full_vram = (2.0 * N) + (2.0 * N) + (12.0 * N) # weights + grads + adamw
    assert math.isclose(full_vram, 1120.0, abs_tol=1e-5)
    
    # LoRA (r=16, 0.5% adapter)
    adapter_N = 0.005 * N # 0.35B
    lora_vram = (2.0 * N) + (2.0 * adapter_N) + (2.0 * adapter_N) + (12.0 * adapter_N)
    assert math.isclose(lora_vram, 145.6, abs_tol=1e-5)
    
    # QLoRA (NF4 = 0.55 bytes/param + 16-bit adapters)
    qlora_vram = (0.55 * N) + (2.0 * adapter_N) + (2.0 * adapter_N) + (12.0 * adapter_N)
    assert math.isclose(qlora_vram, 44.1, abs_tol=1e-5)
    
    print(f"Full Fine-Tuning VRAM: {full_vram:.1f} GB")
    print(f"LoRA FP16 VRAM:        {lora_vram:.1f} GB")
    print(f"QLoRA NF4 VRAM:        {qlora_vram:.1f} GB -> EXACT MATCH\n")


class LoRALinear(nn.Module):
    """
    Production-grade Low-Rank Adaptation (LoRA) Linear Layer with
    Dynamic Weight Merging and Unmerging.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 16, lora_alpha: float = 32.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Pre-trained base weight (Frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False
        
        # Trainable low-rank decomposition
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return F.linear(x, self.weight)
        else:
            # Separate path: base(x) + scaling * (x A^T B^T)
            base_out = F.linear(x, self.weight)
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

    def merge(self):
        """Fuses LoRA weights directly into the base weight for zero-latency inference."""
        if not self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Extracts LoRA weights from base weight for continued training."""
        if self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_w
            self.merged = False


def test_lora_linear():
    print("--- 3. Testing PyTorch LoRALinear Layer ---")
    torch.manual_seed(42)
    B, S, Din, Dout = 2, 4, 32, 64
    r = 8
    
    lora_layer = LoRALinear(in_features=Din, out_features=Dout, r=r, lora_alpha=16.0)
    
    # 1. Check trainable parameters
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
    expected_trainable = r * (Din + Dout)  # 8 * (32 + 64) = 768
    expected_frozen = Din * Dout          # 32 * 64 = 2048
    assert trainable_params == expected_trainable
    assert frozen_params == expected_frozen
    print(f"Trainable params: {trainable_params}, Frozen base params: {frozen_params}")
    
    # 2. Check forward pass before training (B is initialized to 0, so lora output == base output)
    x = torch.randn(B, S, Din)
    out_init = lora_layer(x)
    base_only = F.linear(x, lora_layer.weight)
    assert torch.allclose(out_init, base_only, atol=1e-6), "Zero-initial drift violated!"
    print("Zero-initial drift property verified: out_init == base_only")
    
    # 3. Simulate training update on LoRA weights
    nn.init.normal_(lora_layer.lora_B, std=0.02)
    out_unmerged = lora_layer(x)
    
    # 4. Merge weights and verify exact equivalence
    lora_layer.merge()
    out_merged = lora_layer(x)
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5), "Merged forward pass mismatch!"
    print("Weight merge verification: out_unmerged == out_merged to 1e-5")
    
    # 5. Unmerge weights and verify backpropagation
    lora_layer.unmerge()
    out_train = lora_layer(x)
    loss = out_train.sum()
    loss.backward()
    
    assert lora_layer.weight.grad is None, "Base weight should NOT receive gradients!"
    assert lora_layer.lora_A.grad is not None, "LoRA A must receive gradients"
    assert lora_layer.lora_B.grad is not None, "LoRA B must receive gradients"
    print("Gradient isolation verified: Only LoRA adapter received gradients.")
    print("All LoRA tests passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_memory_illustration()
    test_lora_linear()
    print("🟢 Chapter 12.8 Verification 100% Complete.")
