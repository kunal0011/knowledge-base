"""
Chapter 7.2: Backpropagation Through Convolutions and Pooling Layers
===================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Verification (Hand arithmetic vs Code)
2. Argmax mask routing and additive gradient accumulation for Max-Pooling
3. Average-Pooling uniform gradient distribution
4. Finite-difference numerical gradient checking
5. End-to-end parity with PyTorch autograd across multi-channel batched tensors
"""

import numpy as np
import torch
import torch.nn.functional as F

def conv2d_forward_naive(X, K, b, stride=1, padding=0):
    """
    X: (B, C_in, H_in, W_in)
    K: (C_out, C_in, K_h, K_w)
    b: (C_out,)
    """
    B, C_in, H_in, W_in = X.shape
    C_out, _, K_h, K_w = K.shape
    
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        X_padded = X
        
    H_pad, W_pad = X_padded.shape[2], X_padded.shape[3]
    H_out = (H_pad - K_h) // stride + 1
    W_out = (W_pad - K_w) // stride + 1
    
    Y = np.zeros((B, C_out, H_out, W_out), dtype=np.float64)
    
    for b_idx in range(B):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + K_h
                    w_start = w * stride
                    w_end = w_start + K_w
                    
                    patch = X_padded[b_idx, :, h_start:h_end, w_start:w_end]
                    Y[b_idx, c_out, h, w] = np.sum(patch * K[c_out]) + b[c_out]
                    
    cache = (X, K, b, stride, padding, X_padded)
    return Y, cache

def conv2d_backward_naive(dY, cache):
    """
    dY: (B, C_out, H_out, W_out)
    Returns:
    dX: (B, C_in, H_in, W_in)
    dK: (C_out, C_in, K_h, K_w)
    db: (C_out,)
    """
    X, K, b, stride, padding, X_padded = cache
    B, C_in, H_in, W_in = X.shape
    C_out, _, K_h, K_w = K.shape
    H_out, W_out = dY.shape[2], dY.shape[3]
    
    dX_padded = np.zeros_like(X_padded, dtype=np.float64)
    dK = np.zeros_like(K, dtype=np.float64)
    db = np.sum(dY, axis=(0, 2, 3))
    
    for b_idx in range(B):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + K_h
                    w_start = w * stride
                    w_end = w_start + K_w
                    
                    delta = dY[b_idx, c_out, h, w]
                    
                    # Accumulate dK: delta * X_patch
                    dK[c_out] += delta * X_padded[b_idx, :, h_start:h_end, w_start:w_end]
                    
                    # Accumulate dX_padded: delta * K
                    dX_padded[b_idx, :, h_start:h_end, w_start:w_end] += delta * K[c_out]
                    
    if padding > 0:
        dX = dX_padded[:, :, padding:-padding, padding:-padding]
    else:
        dX = dX_padded
        
    return dX, dK, db

def maxpool2d_forward_naive(X, pool_size=2, stride=2):
    """
    X: (B, C, H, W)
    """
    B, C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    Y = np.zeros((B, C, H_out, W_out), dtype=np.float64)
    argmax_mask = np.zeros_like(X, dtype=bool)
    
    for b in range(B):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    patch = X[b, c, h_start:h_end, w_start:w_end]
                    max_val = np.max(patch)
                    Y[b, c, h, w] = max_val
                    
                    # Find first occurrence in row-major order
                    flat_idx = np.argmax(patch)
                    local_r, local_c = np.unravel_index(flat_idx, patch.shape)
                    argmax_mask[b, c, h_start + local_r, w_start + local_c] = True
                    
    cache = (X, argmax_mask, pool_size, stride)
    return Y, cache

def maxpool2d_backward_naive(dY, cache):
    """
    dY: (B, C, H_out, W_out)
    """
    X, _, pool_size, stride = cache
    B, C, H, W = X.shape
    H_out, W_out = dY.shape[2], dY.shape[3]
    
    dX = np.zeros_like(X, dtype=np.float64)
    
    for b in range(B):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    patch = X[b, c, h_start:h_end, w_start:w_end]
                    flat_idx = np.argmax(patch)
                    local_r, local_c = np.unravel_index(flat_idx, patch.shape)
                    
                    dX[b, c, h_start + local_r, w_start + local_c] += dY[b, c, h, w]
                    
    return dX

def avgpool2d_forward_backward(X, dY, pool_size=2, stride=2):
    """
    Pure NumPy implementation of average pooling forward & backward
    """
    B, C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    Y = np.zeros((B, C, H_out, W_out), dtype=np.float64)
    dX = np.zeros_like(X, dtype=np.float64)
    norm_factor = pool_size * pool_size
    
    for b in range(B):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    Y[b, c, h, w] = np.mean(X[b, c, h_start:h_end, w_start:w_end])
                    dX[b, c, h_start:h_end, w_start:w_end] += dY[b, c, h, w] / norm_factor
                    
    return Y, dX

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 7.2 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. Part 5 Visual Grid Verification
    # -------------------------------------------------------------
    print("--- 1. Part 5 Visual Grid Verification ---")
    X_hand = np.array([[1, 2, 0],
                       [3, 1, 4],
                       [0, 2, 1]], dtype=np.float64).reshape(1, 1, 3, 3)
    
    K_hand = np.array([[2, -1],
                       [1,  3]], dtype=np.float64).reshape(1, 1, 2, 2)
    b_hand = np.array([0.0], dtype=np.float64)
    
    dY_hand = np.array([[1, -1],
                        [2,  0]], dtype=np.float64).reshape(1, 1, 2, 2)
    
    Y_pred, cache = conv2d_forward_naive(X_hand, K_hand, b_hand, stride=1, padding=0)
    dX_pred, dK_pred, db_pred = conv2d_backward_naive(dY_hand, cache)
    
    # Expected hand values
    Y_expected = np.array([[6, 17],
                           [11, 3]], dtype=np.float64).reshape(1, 1, 2, 2)
    
    dK_expected = np.array([[5, 4],
                            [2, 1]], dtype=np.float64).reshape(1, 1, 2, 2)
    
    dX_expected = np.array([[2, -3, 1],
                            [5,  0, -3],
                            [2,  6, 0]], dtype=np.float64).reshape(1, 1, 3, 3)
    
    print("Forward Output Y:\n", Y_pred.squeeze())
    assert np.allclose(Y_pred, Y_expected), "Forward output mismatch!"
    
    print("Weight Gradient dK:\n", dK_pred.squeeze())
    assert np.allclose(dK_pred, dK_expected), "dK mismatch!"
    
    print("Input Gradient dX:\n", dX_pred.squeeze())
    assert np.allclose(dX_pred, dX_expected), "dX mismatch!"
    print("✓ Part 5 Convolution Visual Grid hand calculations perfectly verified!")

    # -------------------------------------------------------------
    # 2. Max-Pooling Visual Grid & Argmax Routing
    # -------------------------------------------------------------
    print("\n--- 2. Max-Pooling Visual Grid Verification ---")
    Z_hand = np.array([
        [1, 5, 2, 3],
        [4, 2, 8, 1],
        [3, 7, 4, 6],
        [0, 2, 9, 5]
    ], dtype=np.float64).reshape(1, 1, 4, 4)
    
    P_pred, pool_cache = maxpool2d_forward_naive(Z_hand, pool_size=2, stride=2)
    P_expected = np.array([[5, 8], [7, 9]], dtype=np.float64).reshape(1, 1, 2, 2)
    assert np.allclose(P_pred, P_expected), "MaxPool forward mismatch!"
    
    dP_hand = np.array([[10, -4], [3, 7]], dtype=np.float64).reshape(1, 1, 2, 2)
    dZ_pred = maxpool2d_backward_naive(dP_hand, pool_cache)
    
    dZ_expected = np.array([
        [0, 10,  0, 0],
        [0,  0, -4, 0],
        [0,  3,  0, 0],
        [0,  0,  7, 0]
    ], dtype=np.float64).reshape(1, 1, 4, 4)
    
    print("MaxPool Input Gradient dZ:\n", dZ_pred.squeeze())
    assert np.allclose(dZ_pred, dZ_expected), "MaxPool dZ mismatch!"
    print("✓ Part 5 Max-Pooling Visual Grid hand calculations perfectly verified!")

    # -------------------------------------------------------------
    # 3. Overlapping Pooling Additive Gradient Accumulation
    # -------------------------------------------------------------
    print("\n--- 3. Overlapping Pooling Additive Gradient Accumulation ---")
    # Illustration 2 test: X = [1, 5, 3] with pool=2, stride=1, upstream=[4, 6]
    X_1d = np.array([[1, 5, 3]], dtype=np.float64).reshape(1, 1, 1, 3)
    P_1d, cache_1d = maxpool2d_forward_naive(X_1d, pool_size=2, stride=1)
    # Windows: [1, 5] -> 5; [5, 3] -> 5
    assert np.allclose(P_1d.squeeze(), [5, 5]), "1D maxpool forward mismatch!"
    
    dP_1d = np.array([[4, 6]], dtype=np.float64).reshape(1, 1, 1, 2)
    dX_1d = maxpool2d_backward_naive(dP_1d, cache_1d)
    dX_1d_expected = np.array([[0, 10, 0]], dtype=np.float64).reshape(1, 1, 1, 3)
    assert np.allclose(dX_1d, dX_1d_expected), "Overlapping pool accumulation mismatch!"
    print("✓ Overlapping pooling gradient additive accumulation verified (dX = [0, 10, 0])!")

    # -------------------------------------------------------------
    # 4. Finite-Difference Numerical Gradient Checking
    # -------------------------------------------------------------
    print("\n--- 4. Finite-Difference Numerical Gradient Checking ---")
    np.random.seed(42)
    X_test = np.random.randn(2, 2, 4, 4)
    K_test = np.random.randn(3, 2, 3, 3)
    b_test = np.random.randn(3)
    
    def loss_fn(X_in, K_in, b_in):
        Y, _ = conv2d_forward_naive(X_in, K_in, b_in, stride=1, padding=1)
        return 0.5 * np.sum(Y**2)
    
    Y_base, cache = conv2d_forward_naive(X_test, K_test, b_test, stride=1, padding=1)
    dY_base = Y_base.copy() # derivative of 0.5 * sum(Y^2) w.r.t Y is Y
    dX_analytic, dK_analytic, db_analytic = conv2d_backward_naive(dY_base, cache)
    
    eps = 1e-6
    # Check dX at random coordinate
    b_idx, c, r, col = 0, 1, 2, 2
    X_plus = X_test.copy(); X_plus[b_idx, c, r, col] += eps
    X_minus = X_test.copy(); X_minus[b_idx, c, r, col] -= eps
    grad_X_num = (loss_fn(X_plus, K_test, b_test) - loss_fn(X_minus, K_test, b_test)) / (2 * eps)
    err_X = abs(grad_X_num - dX_analytic[b_idx, c, r, col])
    print(f"Num grad dX: {grad_X_num:.6f} | Analytic: {dX_analytic[b_idx, c, r, col]:.6f} | Error: {err_X:.2e}")
    assert err_X < 1e-5, f"Numerical gradient check failed on dX: {err_X}"

    # Check dK at random coordinate
    co, ci, kr, kc = 1, 0, 1, 2
    K_plus = K_test.copy(); K_plus[co, ci, kr, kc] += eps
    K_minus = K_test.copy(); K_minus[co, ci, kr, kc] -= eps
    grad_K_num = (loss_fn(X_test, K_plus, b_test) - loss_fn(X_test, K_minus, b_test)) / (2 * eps)
    err_K = abs(grad_K_num - dK_analytic[co, ci, kr, kc])
    print(f"Num grad dK: {grad_K_num:.6f} | Analytic: {dK_analytic[co, ci, kr, kc]:.6f} | Error: {err_K:.2e}")
    assert err_K < 1e-5, f"Numerical gradient check failed on dK: {err_K}"
    print("✓ Finite-difference gradient checks passed cleanly!")

    # -------------------------------------------------------------
    # 5. End-to-End Parity with PyTorch Autograd
    # -------------------------------------------------------------
    print("\n--- 5. End-to-End Parity with PyTorch Autograd ---")
    B, C_in, C_out, H_in, W_in = 3, 4, 6, 7, 7
    K_size, stride, padding = 3, 2, 1
    
    X_pt = torch.randn(B, C_in, H_in, W_in, dtype=torch.float64, requires_grad=True)
    K_pt = torch.randn(C_out, C_in, K_size, K_size, dtype=torch.float64, requires_grad=True)
    b_pt = torch.randn(C_out, dtype=torch.float64, requires_grad=True)
    
    # Forward PyTorch
    Y_pt = F.conv2d(X_pt, K_pt, b_pt, stride=stride, padding=padding)
    dY_pt = torch.randn_like(Y_pt)
    
    # Backward PyTorch
    dX_pt, dK_pt, db_pt = torch.autograd.grad(Y_pt, (X_pt, K_pt, b_pt), grad_outputs=dY_pt)
    
    # Scratch forward & backward
    X_np = X_pt.detach().numpy()
    K_np = K_pt.detach().numpy()
    b_np = b_pt.detach().numpy()
    dY_np = dY_pt.detach().numpy()
    
    Y_scratch, cache_scratch = conv2d_forward_naive(X_np, K_np, b_np, stride=stride, padding=padding)
    dX_scratch, dK_scratch, db_scratch = conv2d_backward_naive(dY_np, cache_scratch)
    
    discrepancy_Y = np.max(np.abs(Y_scratch - Y_pt.detach().numpy()))
    discrepancy_dX = np.max(np.abs(dX_scratch - dX_pt.numpy()))
    discrepancy_dK = np.max(np.abs(dK_scratch - dK_pt.numpy()))
    discrepancy_db = np.max(np.abs(db_scratch - db_pt.numpy()))
    
    print(f"Max Discrepancy Forward Y:  {discrepancy_Y:.2e}")
    print(f"Max Discrepancy Backward dX: {discrepancy_dX:.2e}")
    print(f"Max Discrepancy Backward dK: {discrepancy_dK:.2e}")
    print(f"Max Discrepancy Backward db: {discrepancy_db:.2e}")
    
    assert discrepancy_Y < 1e-12, "Forward discrepancy too high!"
    assert discrepancy_dX < 1e-12, "dX discrepancy too high!"
    assert discrepancy_dK < 1e-12, "dK discrepancy too high!"
    assert discrepancy_db < 1e-12, "db discrepancy too high!"
    
    # Check Average Pooling Parity
    X_pool_pt = torch.randn(2, 3, 6, 6, dtype=torch.float64, requires_grad=True)
    Y_avg_pt = F.avg_pool2d(X_pool_pt, kernel_size=2, stride=2)
    dY_avg_pt = torch.randn_like(Y_avg_pt)
    (dX_avg_pt,) = torch.autograd.grad(Y_avg_pt, (X_pool_pt,), grad_outputs=dY_avg_pt)
    
    Y_avg_np, dX_avg_np = avgpool2d_forward_backward(X_pool_pt.detach().numpy(), dY_avg_pt.detach().numpy(), pool_size=2, stride=2)
    discrepancy_avg_Y = np.max(np.abs(Y_avg_np - Y_avg_pt.detach().numpy()))
    discrepancy_avg_dX = np.max(np.abs(dX_avg_np - dX_avg_pt.numpy()))
    print(f"Max Discrepancy AvgPool Y:  {discrepancy_avg_Y:.2e}")
    print(f"Max Discrepancy AvgPool dX: {discrepancy_avg_dX:.2e}")
    assert discrepancy_avg_Y < 1e-12 and discrepancy_avg_dX < 1e-12
    
    print("✓ Full parity with PyTorch autograd established to < 1e-12!")
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 7.2 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
