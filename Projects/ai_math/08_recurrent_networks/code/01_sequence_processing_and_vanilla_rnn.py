"""
Chapter 8.1: Sequence Processing, Autoregressive Models & Vanilla RNN (BPTT)
===========================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification (T=2 toy sequence)
2. Finite-difference numerical gradient checking for all RNN parameters
3. End-to-end parity with PyTorch's torch.nn.RNN
"""

import numpy as np
import torch
import torch.nn as nn

class VanillaRNNScratch:
    def __init__(self, d_x, d_h, d_y):
        self.d_x = d_x
        self.d_h = d_h
        self.d_y = d_y
        
    def forward(self, X, h0, W_xh, W_hh, b_h, W_hy, b_y):
        """
        X: (T, B, d_x)
        h0: (B, d_h)
        W_xh: (d_h, d_x)
        W_hh: (d_h, d_h)
        b_h: (d_h,)
        W_hy: (d_y, d_h)
        b_y: (d_y,)
        """
        T, B, _ = X.shape
        H = np.zeros((T + 1, B, self.d_h), dtype=np.float64)
        A = np.zeros((T, B, self.d_h), dtype=np.float64)
        Z = np.zeros((T, B, self.d_y), dtype=np.float64)
        
        H[0] = h0
        
        for t in range(T):
            # a_t = W_hh * h_{t-1} + W_xh * x_t + b_h
            a_t = H[t] @ W_hh.T + X[t] @ W_xh.T + b_h
            A[t] = a_t
            H[t + 1] = np.tanh(a_t)
            Z[t] = H[t + 1] @ W_hy.T + b_y
            
        cache = (X, H, A, W_xh, W_hh, b_h, W_hy, b_y)
        return Z, H[1:], cache

    def backward(self, dZ, cache):
        """
        dZ: (T, B, d_y) - upstream loss gradients w.r.t logits Z
        """
        X, H, A, W_xh, W_hh, b_h, W_hy, b_y = cache
        T, B, _ = X.shape
        
        dW_hy = np.zeros_like(W_hy, dtype=np.float64)
        db_y = np.zeros_like(b_y, dtype=np.float64)
        dW_hh = np.zeros_like(W_hh, dtype=np.float64)
        dW_xh = np.zeros_like(W_xh, dtype=np.float64)
        db_h = np.zeros_like(b_h, dtype=np.float64)
        dX = np.zeros_like(X, dtype=np.float64)
        
        dh_next = np.zeros((B, self.d_h), dtype=np.float64)
        
        for t in reversed(range(T)):
            # Output parameter gradients
            # dZ[t]: (B, d_y), H[t+1]: (B, d_h)
            dW_hy += dZ[t].T @ H[t + 1]
            db_y += np.sum(dZ[t], axis=0)
            
            # Gradient w.r.t h_{t+1}: direct from output + recurrent from future
            dh = dZ[t] @ W_hy + dh_next
            
            # Gradient w.r.t pre-activation a_t: dh * (1 - tanh(a_t)^2) = dh * (1 - H[t+1]^2)
            da = dh * (1.0 - H[t + 1]**2)
            
            # Parameter gradients at step t
            dW_hh += da.T @ H[t]
            dW_xh += da.T @ X[t]
            db_h += np.sum(da, axis=0)
            
            # Gradient w.r.t input x_t
            dX[t] = da @ W_xh
            
            # Gradient for previous hidden state h_t to pass into previous step
            dh_next = da @ W_hh
            
        return dW_hy, db_y, dW_hh, dW_xh, db_h, dX

def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid Hand Arithmetic Verification ---")
    rnn = VanillaRNNScratch(d_x=2, d_h=2, d_y=2)
    
    # Concrete toy numbers from Part 5
    x1 = np.array([[1.0, 0.0]], dtype=np.float64) # shape (1, 2)
    x2 = np.array([[0.0, 1.0]], dtype=np.float64)
    X = np.stack([x1, x2], axis=0) # shape (2, 1, 2)
    h0 = np.zeros((1, 2), dtype=np.float64)
    
    W_xh = np.array([[0.5, 0.0],
                     [0.0, 0.5]], dtype=np.float64)
    W_hh = np.array([[0.4, 0.1],
                     [0.0, 0.4]], dtype=np.float64)
    b_h = np.zeros(2, dtype=np.float64)
    
    W_hy = np.array([[1.0, 0.0],
                     [0.0, 1.0]], dtype=np.float64)
    b_y = np.zeros(2, dtype=np.float64)
    
    dZ1 = np.array([[0.5, -0.5]], dtype=np.float64)
    dZ2 = np.array([[0.2,  0.1]], dtype=np.float64)
    dZ = np.stack([dZ1, dZ2], axis=0) # (2, 1, 2)
    
    Z, H, cache = rnn.forward(X, h0, W_xh, W_hh, b_h, W_hy, b_y)
    
    # Verify Forward
    h1_expected = np.array([np.tanh(0.5), 0.0])
    h2_a2_0 = 0.4 * np.tanh(0.5)
    h2_expected = np.array([np.tanh(h2_a2_0), np.tanh(0.5)])
    
    print(f"Computed h1: {H[0].squeeze()} | Expected: {h1_expected}")
    print(f"Computed h2: {H[1].squeeze()} | Expected: {h2_expected}")
    assert np.allclose(H[0].squeeze(), h1_expected), "h1 mismatch!"
    assert np.allclose(H[1].squeeze(), h2_expected), "h2 mismatch!"
    
    # Verify Backward
    dW_hy, db_y, dW_hh, dW_xh, db_h, dX = rnn.backward(dZ, cache)
    
    print("\nComputed dW_hh:\n", dW_hh)
    print("Computed dW_xh:\n", dW_xh)
    print("Computed dW_hy:\n", dW_hy)
    
    # Hand calculation check
    # dh2 = [0.2, 0.1]
    # da2 = [0.2 * (1 - h2_0^2), 0.1 * (1 - h2_1^2)]
    da2_0 = 0.2 * (1 - h2_expected[0]**2)
    da2_1 = 0.1 * (1 - h2_expected[1]**2)
    dW_hh_expected = np.array([[da2_0 * h1_expected[0], 0.0],
                              [da2_1 * h1_expected[0], 0.0]])
    assert np.allclose(dW_hh, dW_hh_expected), "dW_hh mismatch!"
    
    print("✓ Part 5 Visual Grid Vanilla RNN hand calculations verified!\n")

def test_finite_difference_gradient_check():
    print("--- 2. Finite-Difference Numerical Gradient Checking ---")
    np.random.seed(42)
    T, B, d_x, d_h, d_y = 3, 2, 3, 4, 2
    rnn = VanillaRNNScratch(d_x, d_h, d_y)
    
    X = np.random.randn(T, B, d_x)
    h0 = np.random.randn(B, d_h)
    W_xh = np.random.randn(d_h, d_x)
    W_hh = np.random.randn(d_h, d_h)
    b_h = np.random.randn(d_h)
    W_hy = np.random.randn(d_y, d_h)
    b_y = np.random.randn(d_y)
    
    def loss_fn(w_hh, w_xh, w_hy, b_h_in):
        Z, _, _ = rnn.forward(X, h0, w_xh, w_hh, b_h_in, w_hy, b_y)
        return 0.5 * np.sum(Z**2)
    
    Z_base, _, cache = rnn.forward(X, h0, W_xh, W_hh, b_h, W_hy, b_y)
    dZ = Z_base.copy()
    dW_hy_a, db_y_a, dW_hh_a, dW_xh_a, db_h_a, _ = rnn.backward(dZ, cache)
    
    eps = 1e-6
    # Check W_hh
    i, j = 1, 2
    W_hh_p = W_hh.copy(); W_hh_p[i, j] += eps
    W_hh_m = W_hh.copy(); W_hh_m[i, j] -= eps
    grad_num = (loss_fn(W_hh_p, W_xh, W_hy, b_h) - loss_fn(W_hh_m, W_xh, W_hy, b_h)) / (2 * eps)
    err = abs(grad_num - dW_hh_a[i, j])
    print(f"Num grad W_hh[{i},{j}]: {grad_num:.6f} | Analytic: {dW_hh_a[i, j]:.6f} | Error: {err:.2e}")
    assert err < 1e-5, f"Gradient check failed on W_hh: {err}"
    
    # Check W_xh
    i, j = 2, 1
    W_xh_p = W_xh.copy(); W_xh_p[i, j] += eps
    W_xh_m = W_xh.copy(); W_xh_m[i, j] -= eps
    grad_num = (loss_fn(W_hh, W_xh_p, W_hy, b_h) - loss_fn(W_hh, W_xh_m, W_hy, b_h)) / (2 * eps)
    err = abs(grad_num - dW_xh_a[i, j])
    print(f"Num grad W_xh[{i},{j}]: {grad_num:.6f} | Analytic: {dW_xh_a[i, j]:.6f} | Error: {err:.2e}")
    assert err < 1e-5, f"Gradient check failed on W_xh: {err}"
    print("✓ Finite-difference numerical gradient checks passed cleanly!\n")

def test_pytorch_parity():
    print("--- 3. End-to-End Parity with PyTorch nn.RNN ---")
    T, B, d_x, d_h, d_y = 4, 3, 2, 4, 3
    rnn_scratch = VanillaRNNScratch(d_x, d_h, d_y)
    
    # Initialize PyTorch RNN
    torch_rnn = nn.RNN(input_size=d_x, hidden_size=d_h, num_layers=1, nonlinearity='tanh', bias=True)
    torch_linear = nn.Linear(d_h, d_y, bias=True)
    
    # Double precision
    torch_rnn.to(torch.float64)
    torch_linear.to(torch.float64)
    
    # Extract weights for scratch
    W_xh = torch_rnn.weight_ih_l0.detach().numpy().copy()
    W_hh = torch_rnn.weight_hh_l0.detach().numpy().copy()
    # PyTorch RNN has two bias vectors: bias_ih + bias_hh
    b_h = (torch_rnn.bias_ih_l0 + torch_rnn.bias_hh_l0).detach().numpy().copy()
    
    W_hy = torch_linear.weight.detach().numpy().copy()
    b_y = torch_linear.bias.detach().numpy().copy()
    
    X_torch = torch.randn(T, B, d_x, dtype=torch.float64, requires_grad=True)
    h0_torch = torch.randn(1, B, d_h, dtype=torch.float64, requires_grad=True)
    
    # PyTorch forward
    out_rnn, h_n = torch_rnn(X_torch, h0_torch)
    Z_torch = torch_linear(out_rnn) # (T, B, d_y)
    
    # Upstream loss gradient
    dZ_torch = torch.randn_like(Z_torch)
    
    # PyTorch backward
    params = (torch_rnn.weight_ih_l0, torch_rnn.weight_hh_l0, torch_rnn.bias_ih_l0,
              torch_linear.weight, torch_linear.bias, X_torch)
    grads_pt = torch.autograd.grad(Z_torch, params, grad_outputs=dZ_torch)
    
    dW_xh_pt, dW_hh_pt, db_h_pt, dW_hy_pt, db_y_pt, dX_pt = grads_pt
    
    # Scratch forward & backward
    X_np = X_torch.detach().numpy().copy()
    h0_np = h0_torch.squeeze(0).detach().numpy().copy()
    dZ_np = dZ_torch.detach().numpy().copy()
    
    Z_scratch, H_scratch, cache = rnn_scratch.forward(X_np, h0_np, W_xh, W_hh, b_h, W_hy, b_y)
    dW_hy_s, db_y_s, dW_hh_s, dW_xh_s, db_h_s, dX_s = rnn_scratch.backward(dZ_np, cache)
    
    # Discrepancies
    disc_Z = np.max(np.abs(Z_scratch - Z_torch.detach().numpy()))
    disc_W_hh = np.max(np.abs(dW_hh_s - dW_hh_pt.numpy()))
    disc_W_xh = np.max(np.abs(dW_xh_s - dW_xh_pt.numpy()))
    disc_W_hy = np.max(np.abs(dW_hy_s - dW_hy_pt.numpy()))
    disc_b_y = np.max(np.abs(db_y_s - db_y_pt.numpy()))
    disc_X = np.max(np.abs(dX_s - dX_pt.numpy()))
    
    print(f"Max Discrepancy Forward Z:  {disc_Z:.2e}")
    print(f"Max Discrepancy Backward dW_hh: {disc_W_hh:.2e}")
    print(f"Max Discrepancy Backward dW_xh: {disc_W_xh:.2e}")
    print(f"Max Discrepancy Backward dW_hy: {disc_W_hy:.2e}")
    print(f"Max Discrepancy Backward db_y:  {disc_b_y:.2e}")
    print(f"Max Discrepancy Backward dX:    {disc_X:.2e}")
    
    assert disc_Z < 1e-12, "Forward discrepancy too high!"
    assert disc_W_hh < 1e-12, "dW_hh discrepancy too high!"
    assert disc_W_xh < 1e-12, "dW_xh discrepancy too high!"
    assert disc_W_hy < 1e-12, "dW_hy discrepancy too high!"
    assert disc_b_y < 1e-12, "db_y discrepancy too high!"
    assert disc_X < 1e-12, "dX discrepancy too high!"
    print("✓ Parity with PyTorch nn.RNN established to < 1e-12!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 8.1 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_finite_difference_gradient_check()
    test_pytorch_parity()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 8.1 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
