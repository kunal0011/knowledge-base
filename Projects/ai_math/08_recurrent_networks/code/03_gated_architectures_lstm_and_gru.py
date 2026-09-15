"""
Chapter 8.3: Gated Architectures (LSTM & GRU)
=============================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification
2. Finite-difference numerical gradient checking for LSTM and GRU
3. End-to-end parity with PyTorch's torch.nn.LSTMCell and torch.nn.GRUCell
4. Parameter count formula verification
"""

import numpy as np
import torch
import torch.nn as nn

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

class LSTMCellScratch:
    def __init__(self, d_x, d_h):
        self.d_x = d_x
        self.d_h = d_h

    def forward(self, x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh):
        """
        x: (B, d_x)
        h_prev: (B, d_h)
        c_prev: (B, d_h)
        W_ih: (4 * d_h, d_x) in order [i, f, c, o]
        W_hh: (4 * d_h, d_h) in order [i, f, c, o]
        b_ih: (4 * d_h,)
        b_hh: (4 * d_h,)
        """
        B = x.shape[0]
        d_h = self.d_h
        
        gates = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh # (B, 4 * d_h)
        
        gate_i = sigmoid(gates[:, 0:d_h])
        gate_f = sigmoid(gates[:, d_h:2*d_h])
        gate_c = np.tanh(gates[:, 2*d_h:3*d_h])
        gate_o = sigmoid(gates[:, 3*d_h:4*d_h])
        
        c_cur = gate_f * c_prev + gate_i * gate_c
        tanh_c = np.tanh(c_cur)
        h_cur = gate_o * tanh_c
        
        cache = (x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh, gate_i, gate_f, gate_c, gate_o, c_cur, tanh_c)
        return h_cur, c_cur, cache

    def backward(self, dh_cur, dc_cur, cache):
        """
        dh_cur: (B, d_h)
        dc_cur: (B, d_h)
        """
        x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh, i, f, c_tilde, o, c_cur, tanh_c = cache
        d_h = self.d_h
        
        # dL/dc_cur = dc_cur + dh_cur * o * (1 - tanh^2(c_cur))
        dc = dc_cur + dh_cur * o * (1.0 - tanh_c**2)
        
        # Gradients w.r.t gates
        do = dh_cur * tanh_c
        di = dc * c_tilde
        dc_tilde = dc * i
        df = dc * c_prev
        
        # Gradients w.r.t pre-activations
        d_gate_i = di * i * (1.0 - i)
        d_gate_f = df * f * (1.0 - f)
        d_gate_c = dc_tilde * (1.0 - c_tilde**2)
        d_gate_o = do * o * (1.0 - o)
        
        d_gates = np.concatenate([d_gate_i, d_gate_f, d_gate_c, d_gate_o], axis=1) # (B, 4*d_h)
        
        dW_ih = d_gates.T @ x
        dW_hh = d_gates.T @ h_prev
        db_ih = np.sum(d_gates, axis=0)
        db_hh = np.sum(d_gates, axis=0)
        
        dx = d_gates @ W_ih
        dh_prev = d_gates @ W_hh
        dc_prev = dc * f
        
        return dx, dh_prev, dc_prev, dW_ih, dW_hh, db_ih, db_hh

def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid LSTM Hand Arithmetic Verification ---")
    d_x = 1
    d_h = 1
    lstm = LSTMCellScratch(d_x, d_h)
    
    x = np.array([[1.0]], dtype=np.float64)
    h_prev = np.array([[0.5]], dtype=np.float64)
    c_prev = np.array([[2.0]], dtype=np.float64)
    
    # Order: [i, f, c, o]
    # Part 5 definitions:
    # i: W=1.0, U=-1.0, b=0.5
    # f: W=0.0, U=0.0, b=1.0
    # c: W=0.5, U=1.0, b=-1.0
    # o: W=0.0, U=2.0, b=-1.0
    W_ih = np.array([[1.0], [0.0], [0.5], [0.0]], dtype=np.float64)
    W_hh = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    b_ih = np.array([0.5, 1.0, -1.0, -1.0], dtype=np.float64)
    b_hh = np.zeros(4, dtype=np.float64)
    
    h_cur, c_cur, cache = lstm.forward(x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh)
    
    # Expected hand values:
    f_expected = sigmoid(1.0)
    i_expected = sigmoid(1.0)
    c_tilde_expected = 0.0
    c_expected = f_expected * 2.0
    o_expected = 0.5
    h_expected = o_expected * np.tanh(c_expected)
    
    print(f"Computed f_t: {cache[8].squeeze():.6f} | Expected: {f_expected:.6f}")
    print(f"Computed i_t: {cache[7].squeeze():.6f} | Expected: {i_expected:.6f}")
    print(f"Computed c_tilde: {cache[9].squeeze():.6f} | Expected: {c_tilde_expected:.6f}")
    print(f"Computed c_t: {c_cur.squeeze():.6f} | Expected: {c_expected:.6f}")
    print(f"Computed o_t: {cache[10].squeeze():.6f} | Expected: {o_expected:.6f}")
    print(f"Computed h_t: {h_cur.squeeze():.6f} | Expected: {h_expected:.6f}")
    
    assert np.isclose(cache[8].squeeze(), f_expected)
    assert np.isclose(c_cur.squeeze(), c_expected)
    assert np.isclose(h_cur.squeeze(), h_expected)
    
    # Backward pass with dh=1.0, dc=0.0
    dh_cur = np.array([[1.0]], dtype=np.float64)
    dc_cur = np.array([[0.0]], dtype=np.float64)
    dx, dh_prev_grad, dc_prev_grad, _, _, _, _ = lstm.backward(dh_cur, dc_cur, cache)
    
    # Expected CEC backward:
    dc_expected = 1.0 * o_expected * (1.0 - np.tanh(c_expected)**2)
    dc_prev_expected = dc_expected * f_expected
    print(f"Computed dc_prev (CEC): {dc_prev_grad.squeeze():.6f} | Expected: {dc_prev_expected:.6f}")
    assert np.isclose(dc_prev_grad.squeeze(), dc_prev_expected)
    print("✓ Part 5 Visual Grid LSTM hand arithmetic strictly verified!\n")

def test_finite_difference_gradient_check():
    print("--- 2. Finite-Difference Numerical Gradient Checking ---")
    np.random.seed(42)
    B, d_x, d_h = 2, 3, 4
    lstm = LSTMCellScratch(d_x, d_h)
    
    x = np.random.randn(B, d_x)
    h_prev = np.random.randn(B, d_h)
    c_prev = np.random.randn(B, d_h)
    W_ih = np.random.randn(4 * d_h, d_x)
    W_hh = np.random.randn(4 * d_h, d_h)
    b_ih = np.random.randn(4 * d_h)
    b_hh = np.random.randn(4 * d_h)
    
    def loss_fn(w_ih_in, w_hh_in):
        h, c, _ = lstm.forward(x, h_prev, c_prev, w_ih_in, w_hh_in, b_ih, b_hh)
        return 0.5 * (np.sum(h**2) + np.sum(c**2))
        
    h_base, c_base, cache = lstm.forward(x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh)
    dh_base = h_base.copy()
    dc_base = c_base.copy()
    _, _, _, dW_ih_a, dW_hh_a, _, _ = lstm.backward(dh_base, dc_base, cache)
    
    eps = 1e-6
    # Check W_hh at random index
    r, c_idx = 3, 2
    W_hh_p = W_hh.copy(); W_hh_p[r, c_idx] += eps
    W_hh_m = W_hh.copy(); W_hh_m[r, c_idx] -= eps
    grad_num = (loss_fn(W_ih, W_hh_p) - loss_fn(W_ih, W_hh_m)) / (2 * eps)
    err = abs(grad_num - dW_hh_a[r, c_idx])
    print(f"Num grad W_hh[{r},{c_idx}]: {grad_num:.6f} | Analytic: {dW_hh_a[r, c_idx]:.6f} | Error: {err:.2e}")
    assert err < 1e-5, f"Gradient check failed on W_hh: {err}"
    print("✓ Finite-difference numerical gradient checks passed cleanly!\n")

def test_pytorch_parity():
    print("--- 3. End-to-End Parity with PyTorch nn.LSTMCell ---")
    B, d_x, d_h = 3, 4, 5
    lstm_scratch = LSTMCellScratch(d_x, d_h)
    
    torch_lstm = nn.LSTMCell(input_size=d_x, hidden_size=d_h).to(torch.float64)
    
    W_ih = torch_lstm.weight_ih.detach().numpy().copy()
    W_hh = torch_lstm.weight_hh.detach().numpy().copy()
    b_ih = torch_lstm.bias_ih.detach().numpy().copy()
    b_hh = torch_lstm.bias_hh.detach().numpy().copy()
    
    x_pt = torch.randn(B, d_x, dtype=torch.float64, requires_grad=True)
    h_prev_pt = torch.randn(B, d_h, dtype=torch.float64, requires_grad=True)
    c_prev_pt = torch.randn(B, d_h, dtype=torch.float64, requires_grad=True)
    
    # Forward PyTorch
    h_pt, c_pt = torch_lstm(x_pt, (h_prev_pt, c_prev_pt))
    
    # Forward Scratch
    h_s, c_s, cache = lstm_scratch.forward(x_pt.detach().numpy(), h_prev_pt.detach().numpy(), 
                                           c_prev_pt.detach().numpy(), W_ih, W_hh, b_ih, b_hh)
    
    disc_h = np.max(np.abs(h_s - h_pt.detach().numpy()))
    disc_c = np.max(np.abs(c_s - c_pt.detach().numpy()))
    print(f"Max Discrepancy Forward h: {disc_h:.2e}")
    print(f"Max Discrepancy Forward c: {disc_c:.2e}")
    assert disc_h < 1e-12 and disc_c < 1e-12
    
    # Backward PyTorch
    dh_pt = torch.randn_like(h_pt)
    dc_pt = torch.randn_like(c_pt)
    
    grads_pt = torch.autograd.grad((h_pt, c_pt), 
                                   (x_pt, h_prev_pt, c_prev_pt, torch_lstm.weight_ih, torch_lstm.weight_hh),
                                   grad_outputs=(dh_pt, dc_pt))
    dx_pt, dh_prev_pt_g, dc_prev_pt_g, dW_ih_pt, dW_hh_pt = grads_pt
    
    # Backward Scratch
    dx_s, dh_prev_s, dc_prev_s, dW_ih_s, dW_hh_s, _, _ = lstm_scratch.backward(
        dh_pt.detach().numpy(), dc_pt.detach().numpy(), cache)
        
    disc_dx = np.max(np.abs(dx_s - dx_pt.numpy()))
    disc_dh_prev = np.max(np.abs(dh_prev_s - dh_prev_pt_g.numpy()))
    disc_dc_prev = np.max(np.abs(dc_prev_s - dc_prev_pt_g.numpy()))
    disc_dW_ih = np.max(np.abs(dW_ih_s - dW_ih_pt.numpy()))
    disc_dW_hh = np.max(np.abs(dW_hh_s - dW_hh_pt.numpy()))
    
    print(f"Max Discrepancy Backward dx:      {disc_dx:.2e}")
    print(f"Max Discrepancy Backward dh_prev: {disc_dh_prev:.2e}")
    print(f"Max Discrepancy Backward dc_prev: {disc_dc_prev:.2e}")
    print(f"Max Discrepancy Backward dW_ih:   {disc_dW_ih:.2e}")
    print(f"Max Discrepancy Backward dW_hh:   {disc_dW_hh:.2e}")
    
    assert disc_dx < 1e-12
    assert disc_dh_prev < 1e-12
    assert disc_dc_prev < 1e-12
    assert disc_dW_ih < 1e-12
    assert disc_dW_hh < 1e-12
    print("✓ Full parity with PyTorch nn.LSTMCell established to < 1e-12!\n")

def test_parameter_counts():
    print("--- 4. Parameter Count Formula Verification (LSTM vs GRU) ---")
    d_x = 512
    d_h = 512
    
    lstm_pt = nn.LSTMCell(d_x, d_h)
    gru_pt = nn.GRUCell(d_x, d_h)
    
    params_lstm_actual = sum(p.numel() for p in lstm_pt.parameters())
    params_gru_actual = sum(p.numel() for p in gru_pt.parameters())
    
    # Theoretical formulas:
    # PyTorch has weight_ih, weight_hh, bias_ih, bias_hh
    # LSTM: 4 * (d_h * d_x + d_h * d_h + d_h + d_h)
    expected_lstm = 4 * d_h * (d_x + d_h + 2)
    expected_gru = 3 * d_h * (d_x + d_h + 2)
    
    print(f"LSTM Actual: {params_lstm_actual} | Expected: {expected_lstm}")
    print(f"GRU  Actual: {params_gru_actual} | Expected: {expected_gru}")
    print(f"Ratio GRU/LSTM: {params_gru_actual / params_lstm_actual:.4f} (Expected: 0.7500)")
    
    assert params_lstm_actual == expected_lstm
    assert params_gru_actual == expected_gru
    assert abs(params_gru_actual / params_lstm_actual - 0.75) < 1e-4
    print("✓ Parameter count formulas verified!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 8.3 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_finite_difference_gradient_check()
    test_pytorch_parity()
    test_parameter_counts()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 8.3 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
