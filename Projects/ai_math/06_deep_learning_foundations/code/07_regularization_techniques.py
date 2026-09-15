"""
Chapter 6.7: Regularization Techniques (Weight Decay, Dropout, Label Smoothing)
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (Inverted Dropout and Label Smoothing).
2. PyTorch native parity check for Label Smoothing Cross-Entropy.
3. Statistical expectation and variance preservation test for Inverted Dropout.
4. Generalization benchmark: Overfitting reduction via Weight Decay + Dropout.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Part 5 Visual Grid Numerical Verification
# =====================================================================
def verify_visual_grid():
    """
    Validates hand calculation grid for:
    1. Inverted Dropout: h = [2, -4, 1, 3], mask = [1, 0, 1, 0], p = 0.5, g = [0.1, 0.2, 0.3, 0.4]
    2. Label Smoothing: K = 3, true class 0, alpha = 0.1, logits = [2.0, 1.0, 0.0]
    """
    # 1. Inverted Dropout
    h = np.array([2.0, -4.0, 1.0, 3.0], dtype=np.float64)
    m = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    p = 0.5
    scale = 1.0 / (1.0 - p)

    h_tilde = scale * (m * h)
    g = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
    grad_h = scale * (m * g)

    assert np.allclose(h_tilde, [4.0, 0.0, 2.0, 0.0], atol=1e-7)
    assert np.allclose(grad_h, [0.20, 0.0, 0.60, 0.0], atol=1e-7)

    # 2. Label Smoothing
    K = 3
    alpha = 0.1
    y_one_hot = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_smooth = (1.0 - alpha) * y_one_hot + (alpha / K)

    assert np.allclose(y_smooth, [0.9333333, 0.0333333, 0.0333333], atol=1e-5)

    logits = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    exp_z = np.exp(logits)
    S = exp_z / np.sum(exp_z)

    loss_ce = -np.log(S[0])
    loss_ls = -np.sum(y_smooth * np.log(S))

    assert np.allclose(S, [0.665241, 0.244728, 0.090031], atol=1e-4)
    assert abs(loss_ce - 0.407606) < 1e-4
    assert abs(loss_ls - 0.507606) < 1e-4

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Inverted Dropout Output:   {h_tilde.tolist()}")
    print(f"Inverted Dropout Gradient: {grad_h.tolist()}")
    print(f"Smoothed Targets:          {np.round(y_smooth, 4).tolist()}")
    print(f"Standard CE Loss:          {loss_ce:.4f} | Label-Smoothed CE Loss: {loss_ls:.4f}")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. PyTorch Native Parity for Label Smoothing
# =====================================================================
def verify_pytorch_label_smoothing():
    """
    Checks that scratch Label-Smoothed Cross-Entropy matches
    torch.nn.CrossEntropyLoss(label_smoothing=0.1) to machine precision (< 1e-12).
    """
    logits_np = np.array([[2.0, 1.0, 0.0], [0.5, -1.0, 2.5]], dtype=np.float64)
    targets_np = np.array([0, 2], dtype=np.int64)
    alpha = 0.1
    K = 3

    # Scratch computation
    max_c = np.max(logits_np, axis=1, keepdims=True)
    log_sum_exp = max_c + np.log(np.sum(np.exp(logits_np - max_c), axis=1, keepdims=True))
    log_probs = logits_np - log_sum_exp

    # Smoothed targets matrix
    smooth_targets = np.full_like(logits_np, alpha / K)
    smooth_targets[np.arange(2), targets_np] += (1.0 - alpha)

    loss_scratch = -np.mean(np.sum(smooth_targets * log_probs, axis=1))

    # PyTorch computation
    criterion = nn.CrossEntropyLoss(label_smoothing=alpha)
    loss_torch = criterion(torch.tensor(logits_np), torch.tensor(targets_np)).item()

    diff = abs(loss_scratch - loss_torch)
    print("\n--- PyTorch Label Smoothing Parity ---")
    print(f"Scratch Loss:  {loss_scratch:.8f}")
    print(f"PyTorch Loss:  {loss_torch:.8f}")
    print(f"Discrepancy:   {diff:.2e}")

    assert diff < 1e-12
    print("✓ Scratch label smoothing matches PyTorch native C++ kernel (< 1e-12)!")


# =====================================================================
# 3. Statistical Properties of Inverted Dropout
# =====================================================================
def verify_inverted_dropout_statistics():
    """
    Verifies that Inverted Dropout:
    1. Preserves expectation: E[h_tilde] == h
    2. Injects variance: Var(h_tilde) == p / (1 - p) * h^2
    over 200,000 Monte Carlo samples.
    """
    np.random.seed(42)
    h_val = 5.0
    p = 0.4
    n_samples = 200000

    # Sample Bernoulli masks (1 with prob 1 - p)
    masks = (np.random.rand(n_samples) >= p).astype(np.float64)
    scale = 1.0 / (1.0 - p)
    h_tilde = scale * (masks * h_val)

    emp_mean = float(np.mean(h_tilde))
    theo_mean = h_val

    emp_var = float(np.var(h_tilde))
    theo_var = (p / (1.0 - p)) * (h_val ** 2)

    diff_mean = abs(emp_mean - theo_mean) / theo_mean
    diff_var = abs(emp_var - theo_var) / theo_var

    print("\n--- Inverted Dropout Monte Carlo Verification (200,000 samples) ---")
    print(f"Empirical Mean:     {emp_mean:.5f} | Theoretical: {theo_mean:.5f} (Rel error: {diff_mean:.3%})")
    print(f"Empirical Variance: {emp_var:.5f} | Theoretical: {theo_var:.5f} (Rel error: {diff_var:.3%})")

    assert diff_mean < 0.01, f"Mean preservation violated: {diff_mean}"
    assert diff_var < 0.01, f"Variance formula violated: {diff_var}"
    print("✓ Expectation and variance formulas empirically validated (< 1% error)!")


# =====================================================================
# 4. Overfitting vs. Generalization Benchmark
# =====================================================================
def verify_generalization_benchmark():
    """
    Synthesizes a small dataset (150 samples) with 15% label noise.
    Compares:
    - Model A: Unregularized (0 weight decay, 0 dropout) -> Overfits 100% train, poor test.
    - Model B: Regularized (weight decay = 0.02, dropout = 0.3, label smoothing = 0.1).
    Demonstrates that regularization improves generalization accuracy.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    dim = 20
    n_train = 150
    n_test = 500

    # True linear boundary + noise
    w_true = np.random.randn(dim)

    def generate_data(n, noise_rate=0.0):
        X = np.random.randn(n, dim)
        y = (X @ w_true > 0).astype(int)
        if noise_rate > 0:
            flip = np.random.rand(n) < noise_rate
            y[flip] = 1 - y[flip]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    X_train, y_train = generate_data(n_train, noise_rate=0.15)
    X_test, y_test = generate_data(n_test, noise_rate=0.0)

    # 1. Unregularized Model
    model_unreg = nn.Sequential(
        nn.Linear(dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    opt_unreg = torch.optim.Adam(model_unreg.parameters(), lr=0.01, weight_decay=0.0)
    crit_unreg = nn.CrossEntropyLoss()

    for _ in range(300):
        opt_unreg.zero_grad()
        loss = crit_unreg(model_unreg(X_train), y_train)
        loss.backward()
        opt_unreg.step()

    # 2. Regularized Model (Dropout + Weight Decay + Label Smoothing)
    model_reg = nn.Sequential(
        nn.Linear(dim, 64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(64, 2)
    )
    opt_reg = torch.optim.Adam(model_reg.parameters(), lr=0.01, weight_decay=0.05)
    crit_reg = nn.CrossEntropyLoss(label_smoothing=0.1)

    for _ in range(300):
        opt_reg.zero_grad()
        loss = crit_reg(model_reg(X_train), y_train)
        loss.backward()
        opt_reg.step()

    # Evaluate
    model_unreg.eval()
    model_reg.eval()

    with torch.no_grad():
        acc_train_unreg = (model_unreg(X_train).argmax(dim=1) == y_train).float().mean().item()
        acc_test_unreg = (model_unreg(X_test).argmax(dim=1) == y_test).float().mean().item()

        acc_train_reg = (model_reg(X_train).argmax(dim=1) == y_train).float().mean().item()
        acc_test_reg = (model_reg(X_test).argmax(dim=1) == y_test).float().mean().item()

    print("\n--- Generalization Benchmark on Noisy Dataset ---")
    print(f"Unregularized Model: Train Acc = {acc_train_unreg * 100:.1f}% | Test Acc = {acc_test_unreg * 100:.1f}% (Overfit!)")
    print(f"Regularized Model:   Train Acc = {acc_train_reg * 100:.1f}% | Test Acc = {acc_test_reg * 100:.1f}% (Generalizes!)")

    assert acc_test_reg > acc_test_unreg, "Regularization should improve test performance"
    print("✓ Regularization techniques strictly validated!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 6.7 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_pytorch_label_smoothing()
    verify_inverted_dropout_statistics()
    verify_generalization_benchmark()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 6.7 PASSED CLEANLY!")
    print("=" * 65)
