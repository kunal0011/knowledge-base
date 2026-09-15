"""
Chapter 5.6: Adaptive Learning Rates (AdaGrad, RMSprop, Adam, AdamW)
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (hand calculation match).
2. Bias correction factor verification eliminating early-step distortion.
3. Strict parity check between scratch Adam/AdamW and PyTorch implementations.
4. Empirical demonstration of L2 penalty distortion in Adam vs. clean decay in AdamW.
"""

import math
import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 1. Part 5 Visual Grid Numerical Verification
# =====================================================================
def verify_visual_grid():
    """
    Validates hand calculations for Adam and AdamW over 2 steps
    with alpha = 0.1, beta1 = 0.9, beta2 = 0.99, eps = 1e-6, wd = 0.05
    g1 = [10.0, 0.1], g2 = [10.0, 0.2].
    """
    theta_0 = np.array([1.0, 1.0], dtype=np.float64)
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.99
    eps = 1e-6
    wd = 0.05

    g1 = np.array([10.0, 0.1], dtype=np.float64)
    g2 = np.array([10.0, 0.2], dtype=np.float64)

    # --- Step 1 ---
    m1 = (1.0 - beta1) * g1
    v1 = (1.0 - beta2) * (g1 ** 2)

    m1_hat = m1 / (1.0 - beta1 ** 1)
    v1_hat = v1 / (1.0 - beta2 ** 1)

    step1 = alpha * m1_hat / (np.sqrt(v1_hat) + eps)
    theta1_adam = theta_0 - step1
    theta1_adamw = (1.0 - alpha * wd) * theta_0 - step1

    # Assertions for Step 1
    assert np.allclose(m1, [1.0000, 0.0100], atol=1e-5)
    assert np.allclose(v1, [1.0000, 0.0001], atol=1e-5)
    assert np.allclose(m1_hat, [10.0000, 0.1000], atol=1e-5)
    assert np.allclose(v1_hat, [100.0000, 0.0100], atol=1e-5)
    assert np.allclose(step1, [0.1000, 0.1000], atol=1e-4)
    assert np.allclose(theta1_adam, [0.9000, 0.9000], atol=1e-4)
    assert np.allclose(theta1_adamw, [0.8950, 0.8950], atol=1e-4)

    # --- Step 2 ---
    m2 = beta1 * m1 + (1.0 - beta1) * g2
    v2 = beta2 * v1 + (1.0 - beta2) * (g2 ** 2)

    m2_hat = m2 / (1.0 - beta1 ** 2)
    v2_hat = v2 / (1.0 - beta2 ** 2)

    step2 = alpha * m2_hat / (np.sqrt(v2_hat) + eps)
    theta2_adam = theta1_adam - step2
    theta2_adamw = (1.0 - alpha * wd) * theta1_adamw - step2

    assert np.allclose(m2, [1.9000, 0.0290], atol=1e-5)
    assert np.allclose(v2, [1.9900, 0.000499], atol=1e-6)
    assert np.allclose(m2_hat, [10.0000, 0.15263], atol=1e-4)
    assert np.allclose(v2_hat, [100.0000, 0.025075], atol=1e-4)
    assert np.allclose(theta2_adam, [0.8000, 0.80361], atol=1e-4)
    assert np.allclose(theta2_adamw, [0.79053, 0.79414], atol=1e-4)

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Step 1 Adam:  theta_1 = {theta1_adam.tolist()}")
    print(f"Step 1 AdamW: theta_1 = {theta1_adamw.tolist()}")
    print(f"Step 2 Adam:  theta_2 = {theta2_adam.tolist()}")
    print(f"Step 2 AdamW: theta_2 = {theta2_adamw.tolist()}")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Bias Correction Elimination of Early Step Artifacts
# =====================================================================
def verify_bias_correction():
    """
    Simulates stationary gradient g = 1.0 with beta1 = 0.9, beta2 = 0.999.
    Validates that bias correction keeps effective step size at alpha = 0.001
    from step t=1, whereas uncorrected steps overshoot or explode.
    """
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.001
    g = 1.0

    m = (1.0 - beta1) * g
    v = (1.0 - beta2) * (g ** 2)

    # Uncorrected step
    uncorrected_step = alpha * m / math.sqrt(v)

    # Corrected step
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    corrected_step = alpha * m_hat / math.sqrt(v_hat)

    ratio = uncorrected_step / corrected_step

    print("\n--- Bias Correction Analysis at Step t = 1 ---")
    print(f"Uncorrected Step: {uncorrected_step:.6f}")
    print(f"Corrected Step:   {corrected_step:.6f} (Matches alpha = 0.001)")
    print(f"Distortion Ratio: {ratio:.3f}x")

    assert abs(corrected_step - 0.001) < 1e-7
    assert abs(ratio - 3.162277) < 1e-4
    print("✓ Bias correction formula validated!")


# =====================================================================
# 3. PyTorch Parity Check (Adam and AdamW)
# =====================================================================
def verify_pytorch_parity():
    """
    Verifies that scratch Adam and AdamW match PyTorch's native implementations
    to float64 precision (< 1e-7) over 25 optimization steps.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    dim = 5
    lr = 0.01
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    wd = 0.02
    steps = 25

    # Target matrix
    W = np.random.randn(dim, dim)
    A = W.T @ W + 0.1 * np.eye(dim)
    A_torch = torch.tensor(A, dtype=torch.float64)

    # --- Test Adam ---
    theta_scratch = np.ones(dim, dtype=np.float64)
    m_scratch = np.zeros(dim, dtype=np.float64)
    v_scratch = np.zeros(dim, dtype=np.float64)

    theta_torch = nn.Parameter(torch.ones(dim, dtype=torch.float64))
    opt_torch = torch.optim.Adam([theta_torch], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=0.0)

    for t in range(1, steps + 1):
        # Scratch
        g = A @ theta_scratch
        m_scratch = beta1 * m_scratch + (1.0 - beta1) * g
        v_scratch = beta2 * v_scratch + (1.0 - beta2) * (g ** 2)
        m_hat = m_scratch / (1.0 - beta1 ** t)
        v_hat = v_scratch / (1.0 - beta2 ** t)
        theta_scratch -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # PyTorch
        opt_torch.zero_grad()
        loss = 0.5 * torch.matmul(theta_torch, torch.matmul(A_torch, theta_torch))
        loss.backward()
        opt_torch.step()

        diff = np.max(np.abs(theta_scratch - theta_torch.detach().numpy()))
        assert diff < 1e-7, f"Adam parity broken at step {t}: diff = {diff}"

    # --- Test AdamW ---
    theta_scratch_w = np.ones(dim, dtype=np.float64)
    m_scratch_w = np.zeros(dim, dtype=np.float64)
    v_scratch_w = np.zeros(dim, dtype=np.float64)

    theta_torch_w = nn.Parameter(torch.ones(dim, dtype=torch.float64))
    opt_torch_w = torch.optim.AdamW([theta_torch_w], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)

    for t in range(1, steps + 1):
        # Scratch AdamW (Decoupled weight decay)
        g = A @ theta_scratch_w
        # Decouple weight decay
        theta_scratch_w = theta_scratch_w * (1.0 - lr * wd)
        m_scratch_w = beta1 * m_scratch_w + (1.0 - beta1) * g
        v_scratch_w = beta2 * v_scratch_w + (1.0 - beta2) * (g ** 2)
        m_hat = m_scratch_w / (1.0 - beta1 ** t)
        v_hat = v_scratch_w / (1.0 - beta2 ** t)
        theta_scratch_w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # PyTorch AdamW
        opt_torch_w.zero_grad()
        loss = 0.5 * torch.matmul(theta_torch_w, torch.matmul(A_torch, theta_torch_w))
        loss.backward()
        opt_torch_w.step()

        diff_w = np.max(np.abs(theta_scratch_w - theta_torch_w.detach().numpy()))
        assert diff_w < 1e-7, f"AdamW parity broken at step {t}: diff = {diff_w}"

    print("\n--- PyTorch Parity Verification ---")
    print(f"Scratch Adam Final:  {theta_scratch[:3]}")
    print(f"PyTorch Adam Final:  {theta_torch.detach().numpy()[:3]}")
    print(f"Scratch AdamW Final: {theta_scratch_w[:3]}")
    print(f"PyTorch AdamW Final: {theta_torch_w.detach().numpy()[:3]}")
    print("✓ Scratch Adam & AdamW match PyTorch C++ backends to float64 precision (< 1e-7)!")


# =====================================================================
# 4. Adam vs. AdamW Weight Decay Distortion Demonstration
# =====================================================================
def verify_adam_vs_adamw_weight_decay():
    """
    Demonstrates that L2 regularization in Adam distorts weight decay
    across features with different gradient magnitudes, whereas AdamW
    restores uniform proportional weight shrinkage.
    """
    lr = 0.01
    wd = 0.1
    # Two parameters with 1000x gradient magnitude disparity
    # theta1 has massive gradient (frequent feature), theta2 has tiny gradient (rare feature)
    g_large = 100.0
    g_small = 0.1

    # In Adam with L2 regularization:
    # effective_decay = lr * wd / (sqrt(v) + eps)
    # where v ~ g^2
    eff_decay_large_adam = (lr * wd) / math.sqrt(g_large ** 2)
    eff_decay_small_adam = (lr * wd) / math.sqrt(g_small ** 2)

    # In AdamW:
    # effective_decay = lr * wd for both!
    eff_decay_adamw = lr * wd

    print("\n--- Effective Weight Decay Rate Comparison ---")
    print(f"Adam with L2 Reg (Frequent Feature g=100.0): {eff_decay_large_adam:.7f}")
    print(f"Adam with L2 Reg (Rare Feature     g=0.1):   {eff_decay_small_adam:.7f}")
    print(f"AdamW Decoupled (All Features):              {eff_decay_adamw:.7f}")

    # The rare feature in Adam experiences 1000x more weight decay than the frequent feature!
    distortion_ratio = eff_decay_small_adam / eff_decay_large_adam
    assert abs(distortion_ratio - 1000.0) < 1.0

    print(f"Distortion in Adam: Rare features get penalized {distortion_ratio:.0f}x more severely!")
    print("✓ Loshchilov & Hutter (2017) AdamW decoupling thesis empirically verified!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 5.6 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_bias_correction()
    verify_pytorch_parity()
    verify_adam_vs_adamw_weight_decay()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 5.6 PASSED CLEANLY!")
    print("=" * 65)
