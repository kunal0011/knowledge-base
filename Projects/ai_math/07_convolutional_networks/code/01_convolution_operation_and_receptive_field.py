"""
Chapter 7.1: The Convolution Operation (Strides, Padding, Dilation, Receptive Field)
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (4x4 image with 3x3 Sobel kernel).
2. Pure NumPy im2col GEMM convolution engine.
3. Strict parity check against PyTorch F.conv2d across multi-channel inputs.
4. Analytical Receptive Field recurrence calculator (VGG network verification).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# 1. Part 5 Visual Grid Numerical Verification
# =====================================================================
def verify_visual_grid():
    """
    Validates hand calculation grid for 4x4 image with 3x3 kernel:
    X = [[1, 2, 0, 1],
         [0, 1, 3, 2],
         [2, 0, 1, 0],
         [1, 3, 2, 1]]
    K = [[1,  0, -1],
         [2,  0, -2],
         [1,  0, -1]]
    Valid padding (p=0, s=1) -> Y = [[-4, -1], [-2, +1]]
    Strided (p=0, s=2)       -> Y = [[-4]]
    """
    X = np.array([
        [1, 2, 0, 1],
        [0, 1, 3, 2],
        [2, 0, 1, 0],
        [1, 3, 2, 1]
    ], dtype=np.float64)

    K = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float64)

    # 1. Valid Convolution (s = 1, p = 0)
    H_out = 4 - 3 + 1
    W_out = 4 - 3 + 1
    Y_valid = np.zeros((H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        for j in range(W_out):
            patch = X[i : i + 3, j : j + 3]
            Y_valid[i, j] = np.sum(patch * K)

    expected_valid = np.array([[-4.0, -1.0], [-2.0, 1.0]])
    assert np.allclose(Y_valid, expected_valid, atol=1e-7)

    # 2. Strided Convolution (s = 2, p = 0)
    Y_strided = np.zeros((1, 1), dtype=np.float64)
    patch_00 = X[0:3, 0:3]
    Y_strided[0, 0] = np.sum(patch_00 * K)
    assert np.allclose(Y_strided, [[-4.0]], atol=1e-7)

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Valid Convolution (s=1, p=0) Output:\n{Y_valid}")
    print(f"Strided Convolution (s=2, p=0) Output:\n{Y_strided}")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. NumPy im2col GEMM Convolution Engine & PyTorch Parity
# =====================================================================
def im2col_indices(x, kh, kw, padding=1, stride=1):
    """
    Vectorized im2col implementation for extracting patches.
    """
    B, C, H, W = x.shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    cols = np.zeros((B, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride]

    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(C * kh * kw, B * out_h * out_w)
    return cols, out_h, out_w


def conv2d_im2col(x, w, b=None, padding=1, stride=1):
    """
    Executes 2D convolution via im2col and General Matrix Multiplication (GEMM).
    x: (B, C_in, H, W)
    w: (C_out, C_in, kh, kw)
    b: (C_out,)
    """
    B, C_in, H, W = x.shape
    C_out, _, kh, kw = w.shape

    cols, out_h, out_w = im2col_indices(x, kh, kw, padding, stride)
    w_row = w.reshape(C_out, -1)

    out = w_row @ cols
    if b is not None:
        out += b.reshape(-1, 1)

    out = out.reshape(C_out, B, out_h, out_w).transpose(1, 0, 2, 3)
    return out


def verify_im2col_conv2d_parity():
    """
    Tests scratch im2col conv2d against PyTorch F.conv2d across
    random multi-channel tensors with stride, padding, and bias.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    B, C_in, H, W = 2, 3, 8, 8
    C_out, kh, kw = 4, 3, 3
    stride = 2
    padding = 1

    x_np = np.random.randn(B, C_in, H, W)
    w_np = np.random.randn(C_out, C_in, kh, kw)
    b_np = np.random.randn(C_out)

    # Scratch im2col
    out_scratch = conv2d_im2col(x_np, w_np, b_np, padding=padding, stride=stride)

    # PyTorch F.conv2d
    x_torch = torch.tensor(x_np, dtype=torch.float64)
    w_torch = torch.tensor(w_np, dtype=torch.float64)
    b_torch = torch.tensor(b_np, dtype=torch.float64)

    out_torch = F.conv2d(x_torch, w_torch, b_torch, stride=stride, padding=padding).numpy()

    diff = np.max(np.abs(out_scratch - out_torch))
    print("\n--- im2col GEMM vs. PyTorch F.conv2d Parity ---")
    print(f"Scratch Output Shape: {out_scratch.shape} | PyTorch Shape: {out_torch.shape}")
    print(f"Max Absolute Discrepancy: {diff:.2e}")

    assert diff < 1e-12, f"im2col convolution discrepancy too high: {diff}"
    print("✓ Scratch im2col GEMM matches PyTorch C++ cuDNN kernel (< 1e-12)!")


# =====================================================================
# 3. Analytical Receptive Field Recurrence Calculator
# =====================================================================
def calculate_receptive_field(layers):
    """
    layers: list of (kernel_size, stride, dilation) tuples.
    Returns cumulative list of (receptive_field, jump_stride).
    """
    r = 1
    j = 1
    history = [(r, j)]

    for k, s, d in layers:
        k_eff = d * (k - 1) + 1
        r = r + (k_eff - 1) * j
        j = j * s
        history.append((r, j))

    return history


def verify_receptive_field_recurrence():
    """
    Validates Problem 1 theoretical calculations for 4-layer VGG:
    Layer 1: Conv2D (k=3, s=1) -> r=3, j=1
    Layer 2: Conv2D (k=3, s=1) -> r=5, j=1
    Layer 3: MaxPool (k=2, s=2) -> r=6, j=2
    Layer 4: Conv2D (k=3, s=1) -> r=10, j=2
    """
    vgg_layers = [
        (3, 1, 1), # Conv 1
        (3, 1, 1), # Conv 2
        (2, 2, 1), # MaxPool
        (3, 1, 1)  # Conv 3
    ]

    history = calculate_receptive_field(vgg_layers)

    assert history[1] == (3, 1)
    assert history[2] == (5, 1)
    assert history[3] == (6, 2)
    assert history[4] == (10, 2)

    print("\n--- Receptive Field Recurrence Verification ---")
    for l, (r, j) in enumerate(history):
        print(f"Layer {l}: Receptive Field r = {r}x{r} | Cumulative Stride j = {j}")
    print("✓ VGG network receptive field recurrence verified!")


# =====================================================================
# 4. Dilated Convolution Exponential Growth Verification
# =====================================================================
def verify_dilated_receptive_field():
    """
    Validates that a 3-layer dilated convolution network with d = [1, 2, 4]
    and k = 3 expands receptive field to 15x15, compared to 7x7 for standard conv.
    """
    # Standard conv (d = 1)
    std_layers = [(3, 1, 1), (3, 1, 1), (3, 1, 1)]
    rf_std = calculate_receptive_field(std_layers)[-1][0]
    assert rf_std == 7

    # Dilated conv (d = 1, 2, 4)
    dilated_layers = [(3, 1, 1), (3, 1, 2), (3, 1, 4)]
    rf_dilated = calculate_receptive_field(dilated_layers)[-1][0]
    # Layer 1: 1 + (3-1) = 3
    # Layer 2: 3 + (2*(3-1))*1 = 3 + 4 = 7
    # Layer 3: 7 + (4*(3-1))*1 = 7 + 8 = 15
    assert rf_dilated == 15

    print("\n--- Dilated Convolution Growth Verification ---")
    print(f"Standard 3-Layer Conv (d=1,1,1) Receptive Field: {rf_std}x{rf_std}")
    print(f"Dilated 3-Layer Conv  (d=1,2,4) Receptive Field: {rf_dilated}x{rf_dilated}")
    print("✓ Dilated convolution exponential receptive field expansion confirmed!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 7.1 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_im2col_conv2d_parity()
    verify_receptive_field_recurrence()
    verify_dilated_receptive_field()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 7.1 PASSED CLEANLY!")
    print("=" * 65)
