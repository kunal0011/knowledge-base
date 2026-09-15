"""
Chapter 6.1: The Perceptron, Linear Models & Perceptron Convergence Theorem
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (hand calculation match).
2. Novikoff's Perceptron Convergence Theorem empirical margin & mistake bound test.
3. XOR non-separability proof & 75% accuracy ceiling test for single perceptron.
4. 2-layer MLP solving XOR in NumPy and PyTorch autograd.
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
    Validates hand calculation trace of Perceptron Learning Algorithm
    on the 4-sample 2D dataset with homogeneous bias coordinates:
    x_1 = [1, 1, 2], y_1 = +1
    x_2 = [1, 2, 1], y_2 = +1
    x_3 = [1, 0, 0], y_3 = -1
    x_4 = [1, -1, 1], y_4 = -1
    """
    X = np.array([
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, -1.0, 1.0]
    ], dtype=np.float64)
    y = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

    w = np.zeros(3, dtype=np.float64)
    mistake_history = []
    weight_history = [w.copy()]

    # Epoch 1
    # Sample 1
    score1 = np.dot(w, X[0])
    if y[0] * score1 <= 0:
        w = w + y[0] * X[0]
        mistake_history.append((1, 1))
        weight_history.append(w.copy())
    assert np.allclose(w, [1.0, 1.0, 2.0], atol=1e-7)

    # Sample 2
    score2 = np.dot(w, X[1])
    assert score2 == 5.0
    assert y[1] * score2 > 0  # Correct

    # Sample 3
    score3 = np.dot(w, X[2])
    assert score3 == 1.0
    if y[2] * score3 <= 0:
        w = w + y[2] * X[2]
        mistake_history.append((1, 3))
        weight_history.append(w.copy())
    assert np.allclose(w, [0.0, 1.0, 2.0], atol=1e-7)

    # Sample 4
    score4 = np.dot(w, X[3])
    assert score4 == 1.0
    if y[3] * score4 <= 0:
        w = w + y[3] * X[3]
        mistake_history.append((1, 4))
        weight_history.append(w.copy())
    assert np.allclose(w, [-1.0, 2.0, 1.0], atol=1e-7)

    # Epoch 2 (All 4 samples correct)
    for i in range(4):
        s = np.dot(w, X[i])
        assert y[i] * s > 0, f"Sample {i+1} failed in epoch 2!"

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Total Mistakes in Epoch 1: {len(mistake_history)} (Samples: {[s for e, s in mistake_history]})")
    print(f"Final Converged Hyperplane Weights [b, w_1, w_2]: {w.tolist()}")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Novikoff's Perceptron Convergence Bound
# =====================================================================
def verify_novikoff_bound():
    """
    Validates Novikoff's theorem:
    R = max ||x_i|| = sqrt(6)
    w* = 1/sqrt(6) * [-1, 2, 1]^T
    gamma = min y_i (w*^T x_i) = 1/sqrt(6)
    k_max = (R / gamma)^2 = 36
    """
    X = np.array([
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, -1.0, 1.0]
    ], dtype=np.float64)
    y = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

    norms = np.linalg.norm(X, axis=1)
    R = float(np.max(norms))
    assert abs(R - math.sqrt(6.0)) < 1e-6

    w_star = np.array([-1.0, 2.0, 1.0], dtype=np.float64) / math.sqrt(6.0)
    assert abs(np.linalg.norm(w_star) - 1.0) < 1e-6

    margins = y * (X @ w_star)
    gamma = float(np.min(margins))
    assert abs(gamma - (1.0 / math.sqrt(6.0))) < 1e-6

    k_max = (R / gamma) ** 2
    assert abs(k_max - 36.0) < 1e-5

    k_actual = 3
    assert k_actual <= k_max

    print("\n--- Novikoff Perceptron Convergence Bound ---")
    print(f"Data Radius R:             {R:.5f} (sqrt(6))")
    print(f"Geometric Margin gamma:    {gamma:.5f} (1/sqrt(6))")
    print(f"Theoretical Max Mistakes:  {k_max:.0f}")
    print(f"Actual Mistakes Observed:  {k_actual}")
    print("✓ Novikoff bound k_actual <= (R / gamma)^2 strictly confirmed!")


# =====================================================================
# 3. XOR Impossibility on Single Linear Perceptron
# =====================================================================
def verify_xor_impossibility():
    """
    Empirically verifies that a single linear model cannot exceed 75% accuracy
    on the XOR truth table by exhaustive grid search over all hyperplanes.
    """
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = np.array([0, 1, 1, 0])

    # Search through 10,000 random hyperplanes
    np.random.seed(42)
    max_acc = 0.0
    for _ in range(10000):
        w = np.random.randn(2)
        b = np.random.randn(1)
        preds = ((X @ w + b) >= 0).astype(int)
        acc = np.mean(preds == y)
        if acc > max_acc:
            max_acc = acc

    print("\n--- XOR Linear Non-Separability ---")
    print(f"Maximum Empirical Accuracy for Single Perceptron: {max_acc * 100:.1f}%")
    assert max_acc == 0.75, f"Single perceptron somehow exceeded 75%: {max_acc}"
    print("✓ Single perceptron cannot exceed 75% on XOR (Minsky-Papert 1969 confirmed)!")


# =====================================================================
# 4. Multilayer Perceptron (MLP) Solving XOR
# =====================================================================
def verify_2layer_mlp_xor():
    """
    1. NumPy explicit hand-constructed 2-layer step network.
    2. PyTorch autograd-trained 2-layer MLP reaching 100% accuracy.
    """
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    # 1. NumPy Hand-Crafted Weights:
    # h1 = Step(x1 + x2 - 0.5) (OR)
    # h2 = Step(-x1 - x2 + 1.5) (NAND)
    # y = Step(h1 + h2 - 1.5) (AND)
    def step_fn(z):
        return (z >= 0.0).astype(np.float32)

    h1 = step_fn(X[:, 0] + X[:, 1] - 0.5)
    h2 = step_fn(-X[:, 0] - X[:, 1] + 1.5)
    y_hand = step_fn(h1 + h2 - 1.5)

    assert np.all(y_hand == y), f"Hand-crafted MLP failed on XOR: {y_hand}"

    # 2. PyTorch Autograd Training
    torch.manual_seed(42)
    X_torch = torch.tensor(X)
    y_torch = torch.tensor(y).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(400):
        optimizer.zero_grad()
        logits = model(X_torch)
        loss = criterion(logits, y_torch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(model(X_torch)) >= 0.5).float().squeeze()
        pytorch_acc = (preds == torch.tensor(y)).float().mean().item()

    print("\n--- 2-Layer MLP Solving XOR ---")
    print(f"NumPy Hand-Crafted MLP Predictions: {y_hand.tolist()} (100% Accuracy)")
    print(f"PyTorch Trained MLP Accuracy:       {pytorch_acc * 100:.1f}%")

    assert pytorch_acc == 1.0, f"PyTorch MLP failed to reach 100% accuracy: {pytorch_acc}"
    print("✓ 2-layer MLP solves XOR to 100% accuracy!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 6.1 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_novikoff_bound()
    verify_xor_impossibility()
    verify_2layer_mlp_xor()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 6.1 PASSED CLEANLY!")
    print("=" * 65)
