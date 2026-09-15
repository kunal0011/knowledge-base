"""
Chapter 7.3: Classic to Modern ConvNets (LeNet, AlexNet, VGG, ResNet, ConvNeXt)
================================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Verification (Toy Residual Block forward and backward)
2. VGG Receptive field and parameter factorization
3. ResNet BasicBlock and Bottleneck forward/backward verification
4. ConvNeXt block with depthwise 7x7 conv and LayerScale
5. The Gradient Highway Experiment: 50-layer PlainNet vs. 50-layer ResNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid Hand Arithmetic Verification
# ----------------------------------------------------------------------
def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid Residual Block Verification ---")
    x = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    
    W1 = torch.tensor([[1.0, -1.0],
                       [0.0,  2.0]], dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    
    W2 = torch.tensor([[ 0.5, 1.0],
                       [-1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    
    # Forward pass
    z1 = torch.matmul(W1, x) + b1
    a1 = F.relu(z1)
    F_x = torch.matmul(W2, a1) + b2
    y = F_x + x
    
    y_expected = torch.tensor([3.5, -4.0], dtype=torch.float64)
    print(f"Computed Output y: {y.detach().numpy()} | Expected: {y_expected.numpy()}")
    assert torch.allclose(y, y_expected), "Forward output mismatch!"
    
    # Backward pass with upstream dL/dy = [1.0, 2.0]
    grad_y = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y.backward(gradient=grad_y)
    
    dx_expected = torch.tensor([-0.5, 3.5], dtype=torch.float64)
    print(f"Computed Input Gradient dL/dx: {x.grad.numpy()} | Expected: {dx_expected.numpy()}")
    assert torch.allclose(x.grad, dx_expected), "Backward input gradient mismatch!"
    print("✓ Part 5 Visual Grid Residual Block forward and backward arithmetic strictly verified!\n")

# ----------------------------------------------------------------------
# 2. Receptive Field Factorization in VGG
# ----------------------------------------------------------------------
def test_vgg_factorization():
    print("--- 2. VGG Receptive Field & Parameter Factorization ---")
    C = 64
    # Single 5x5 Conv
    single_5x5 = nn.Conv2d(C, C, kernel_size=5, padding=2, bias=False)
    params_5x5 = sum(p.numel() for p in single_5x5.parameters())
    
    # Two stacked 3x3 Convs
    stacked_3x3 = nn.Sequential(
        nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
    )
    params_stacked = sum(p.numel() for p in stacked_3x3.parameters())
    
    expected_5x5 = 5 * 5 * C * C
    expected_stacked = 2 * (3 * 3 * C * C)
    savings = (params_5x5 - params_stacked) / params_5x5
    
    print(f"5x5 Params: {params_5x5} (Expected: {expected_5x5})")
    print(f"2x 3x3 Params: {params_stacked} (Expected: {expected_stacked})")
    print(f"Parameter Savings: {savings * 100:.1f}% (Expected: 28.0%)")
    
    assert params_5x5 == expected_5x5
    assert params_stacked == expected_stacked
    assert abs(savings - 0.28) < 1e-4
    print("✓ VGG receptive field parameter savings strictly confirmed!\n")

# ----------------------------------------------------------------------
# 3. ResNet BasicBlock and Bottleneck
# ----------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def test_resnet_blocks():
    print("--- 3. ResNet BasicBlock & Bottleneck Verification ---")
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    
    # Test BasicBlock identity shortcut
    block = BasicBlock(64, 64, stride=1)
    out = block(x)
    assert out.shape == (2, 64, 16, 16), f"Shape mismatch: {out.shape}"
    out.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    
    # Test Bottleneck with downsampling projection
    x.grad.zero_()
    bottleneck = Bottleneck(64, 32, stride=2) # 64 -> 32 -> 32 -> 128
    out_b = bottleneck(x)
    assert out_b.shape == (2, 128, 8, 8), f"Bottleneck shape mismatch: {out_b.shape}"
    out_b.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    print("✓ ResNet BasicBlock and Bottleneck forward and backward gradients verified!\n")

# ----------------------------------------------------------------------
# 4. ConvNeXt Block Architecture
# ----------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (Liu et al., 2022)
    Depthwise 7x7 -> LayerNorm -> 1x1 Conv (4x) -> GELU -> 1x1 Conv (project) -> LayerScale -> Residual Add
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # 1x1 conv via Linear
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # Permute (N, C, H, W) -> (N, H, W, C) for LayerNorm & Linear
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # Permute back to (N, C, H, W)
        x = input + x
        return x

def test_convnext_block():
    print("--- 4. ConvNeXt Block Verification ---")
    dim = 96
    block = ConvNeXtBlock(dim=dim)
    x = torch.randn(2, dim, 14, 14, requires_grad=True)
    out = block(x)
    assert out.shape == (2, dim, 14, 14), f"ConvNeXt output shape mismatch: {out.shape}"
    out.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    print("✓ ConvNeXt Block (7x7 depthwise + LayerNorm + GELU + LayerScale) verified!\n")

# ----------------------------------------------------------------------
# 5. The Gradient Highway Experiment (50 Layers: PlainNet vs ResNet)
# ----------------------------------------------------------------------
def test_gradient_highway():
    print("--- 5. The Gradient Highway Experiment (50-Layer Plain vs ResNet) ---")
    num_layers = 50
    dim = 32
    torch.manual_seed(42)
    
    # 50-Layer PlainNet (Stacked Linear layers without skip connections)
    plain_layers = []
    for _ in range(num_layers):
        linear = nn.Linear(dim, dim, bias=False)
        # Standard initialization
        nn.init.orthogonal_(linear.weight, gain=0.9)
        plain_layers.append(linear)
        plain_layers.append(nn.ReLU())
    plain_net = nn.Sequential(*plain_layers)
    
    # 50-Layer ResNet (Stacked Residual blocks with skip connections)
    class ResModule(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc1 = nn.Linear(d, d, bias=False)
            nn.init.orthogonal_(self.fc1.weight, gain=0.9)
        def forward(self, x):
            return x + F.relu(self.fc1(x))
            
    res_layers = [ResModule(dim) for _ in range(num_layers)]
    res_net = nn.Sequential(*res_layers)
    
    x_plain = torch.randn(10, dim, requires_grad=True)
    x_res = x_plain.clone().detach().requires_grad_(True)
    
    out_plain = plain_net(x_plain)
    out_res = res_net(x_res)
    
    loss_plain = out_plain.sum()
    loss_res = out_res.sum()
    
    loss_plain.backward()
    loss_res.backward()
    
    grad_norm_plain = x_plain.grad.norm().item()
    grad_norm_res = x_res.grad.norm().item()
    
    print(f"50-Layer PlainNet Input Gradient Norm: {grad_norm_plain:.6e}")
    print(f"50-Layer ResNet   Input Gradient Norm: {grad_norm_res:.6f}")
    
    # PlainNet gradient should have severely decayed (order of magnitude ~ 1e-3 or smaller)
    # ResNet gradient highway preserves strong, non-vanishing gradient
    assert grad_norm_plain < 1e-2, f"PlainNet should suffer gradient decay, got {grad_norm_plain}"
    assert grad_norm_res > 1.0, f"ResNet should preserve gradient flow, got {grad_norm_res}"
    print("✓ The Gradient Highway theorem confirmed: ResNet preserves gradient magnitude over 50 layers!")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 7.3 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_vgg_factorization()
    test_resnet_blocks()
    test_convnext_block()
    test_gradient_highway()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 7.3 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
