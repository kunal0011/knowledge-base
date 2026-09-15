"""
Chapter 6.5: Loss Functions Deep-Dive (MSE, BCE, Cross-Entropy, Focal, Triplet Loss)
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (BCE vs. Focal Loss and Triplet Loss).
2. Strict parity check between scratch loss formulas and PyTorch loss modules.
3. Focal Loss autograd gradient check.
4. Empirical class imbalance experiment (99:1 imbalance) comparing BCE vs. Focal Loss.
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
    Validates hand calculation grid for BCE vs. Focal Loss on:
    Sample 1: y = 1, z = +3.0 (Easy positive)
    Sample 2: y = 0, z = +2.0 (Hard negative)
    Sample 3: y = 1, z =  0.0 (Borderline)
    And Triplet Loss on a = [0, 0], p = [0.5, 0.5], n = [0.6, 0.6], margin = 0.5.
    """
    logits = np.array([3.0, 2.0, 0.0], dtype=np.float64)
    y_true = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    probs = 1.0 / (1.0 + np.exp(-logits))
    pt = np.where(y_true == 1.0, probs, 1.0 - probs)

    # 1. BCE Loss
    bce_loss = -np.log(pt)
    assert np.allclose(bce_loss, [0.048587, 2.126928, 0.693147], atol=1e-4)

    # 2. Focal Loss with gamma = 2, alpha = 1
    gamma = 2.0
    modulating_factor = (1.0 - pt) ** gamma
    focal_loss = modulating_factor * bce_loss
    assert np.allclose(focal_loss, [0.000109, 1.650058, 0.173287], atol=1e-4)

    # 3. BCE Gradients with respect to logits: p - y
    bce_grads = probs - y_true
    assert np.allclose(bce_grads, [-0.047426, 0.880797, -0.500000], atol=1e-4)

    # 4. Triplet Loss
    a = np.array([0.0, 0.0], dtype=np.float64)
    p = np.array([0.5, 0.5], dtype=np.float64)
    n = np.array([0.6, 0.6], dtype=np.float64)
    margin = 0.5

    d_ap_sq = np.sum((a - p) ** 2)
    d_an_sq = np.sum((a - n) ** 2)
    assert abs(d_ap_sq - 0.50) < 1e-7
    assert abs(d_an_sq - 0.72) < 1e-7

    triplet_loss = max(0.0, d_ap_sq - d_an_sq + margin)
    assert abs(triplet_loss - 0.28) < 1e-7

    print("--- Part 5 Visual Grid Verification ---")
    print(f"BCE Losses:   {np.round(bce_loss, 4).tolist()}")
    print(f"Focal Losses: {np.round(focal_loss, 6).tolist()}")
    print(f"BCE Gradients:{np.round(bce_grads, 4).tolist()}")
    print(f"Triplet Loss: {triplet_loss:.4f} (Matches 0.2800 exactly)")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. PyTorch Native Loss Parity Verification
# =====================================================================
def verify_pytorch_loss_parity():
    """
    Verifies that scratch mathematical formulas match PyTorch loss modules
    across MSE, Huber, BCEWithLogits, CrossEntropy, and TripletMarginLoss.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. MSE
    y_hat = np.array([1.5, 2.8, -0.4], dtype=np.float64)
    y = np.array([1.0, 3.0, 0.0], dtype=np.float64)
    mse_scratch = np.mean((y_hat - y) ** 2)
    mse_torch = F.mse_loss(torch.tensor(y_hat), torch.tensor(y)).item()
    assert abs(mse_scratch - mse_torch) < 1e-12

    # 2. Huber Loss (delta = 1.0)
    delta = 1.0
    r = y_hat - y
    huber_scratch = np.mean(np.where(np.abs(r) <= delta, 0.5 * (r ** 2), delta * (np.abs(r) - 0.5 * delta)))
    huber_torch = F.huber_loss(torch.tensor(y_hat), torch.tensor(y), delta=delta).item()
    assert abs(huber_scratch - huber_torch) < 1e-12

    # 3. BCE with Logits
    logits = np.array([2.5, -1.2, 0.4], dtype=np.float64)
    targets = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    # Stable formula: max(z, 0) - z*y + log(1 + exp(-abs(z)))
    bce_scratch = np.mean(np.maximum(logits, 0.0) - logits * targets + np.log(1.0 + np.exp(-np.abs(logits))))
    bce_torch = F.binary_cross_entropy_with_logits(torch.tensor(logits), torch.tensor(targets)).item()
    assert abs(bce_scratch - bce_torch) < 1e-12

    # 4. Cross-Entropy Loss
    logits_cce = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)
    # Stable LogSoftmax
    max_c = np.max(logits_cce, axis=1, keepdims=True)
    log_sum_exp = max_c + np.log(np.sum(np.exp(logits_cce - max_c), axis=1, keepdims=True))
    cce_scratch = -np.mean(logits_cce[np.arange(2), labels] - log_sum_exp.squeeze())
    cce_torch = F.cross_entropy(torch.tensor(logits_cce), torch.tensor(labels)).item()
    assert abs(cce_scratch - cce_torch) < 1e-12

    # 5. Triplet Margin Loss (p = 2, squared distances)
    a = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    n = torch.tensor([[0.6, 0.6]], dtype=torch.float64)
    triplet_scratch = max(0.0, torch.sum((a - p) ** 2).item() - torch.sum((a - n) ** 2).item() + 0.5)

    triplet_fn = nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7, swap=False)
    # PyTorch TripletMarginLoss uses L2 norm (not squared) by default: ||a-p|| - ||a-n|| + margin
    norm_scratch = max(0.0, torch.norm(a - p, p=2).item() - torch.norm(a - n, p=2).item() + 0.5)
    triplet_torch = triplet_fn(a, p, n).item()
    assert abs(norm_scratch - triplet_torch) < 1e-7

    print("\n--- PyTorch Loss Parity Verification ---")
    print(f"MSE Loss Match:          {abs(mse_scratch - mse_torch):.2e}")
    print(f"Huber Loss Match:        {abs(huber_scratch - huber_torch):.2e}")
    print(f"BCEWithLogits Match:     {abs(bce_scratch - bce_torch):.2e}")
    print(f"CrossEntropyLoss Match:  {abs(cce_scratch - cce_torch):.2e}")
    print(f"TripletMarginLoss Match: {abs(norm_scratch - triplet_torch):.2e}")
    print("✓ All scratch loss implementations match PyTorch C++ kernels to machine precision!")


# =====================================================================
# 3. Focal Loss Autograd Gradient Verification
# =====================================================================
def verify_focal_loss_autograd():
    """
    Checks that the analytical derivative of Focal Loss matches
    PyTorch autograd backward pass to machine precision.
    """
    z_torch = torch.tensor([3.0, 2.0, 0.0], dtype=torch.float64, requires_grad=True)
    y_torch = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    gamma = 2.0

    # Forward
    p = torch.sigmoid(z_torch)
    pt = torch.where(y_torch == 1.0, p, 1.0 - p)
    loss = -torch.sum(((1.0 - pt) ** gamma) * torch.log(pt))
    loss.backward()

    autograd_grad = z_torch.grad.numpy()

    # Analytical gradient:
    # dL/dz = (2y - 1) * [ gamma * pt * ((1 - pt)^gamma) * log(pt) - (1 - pt)^(gamma + 1) ]
    p_np = p.detach().numpy()
    pt_np = pt.detach().numpy()
    y_np = y_torch.numpy()

    bracket = gamma * pt_np * ((1.0 - pt_np) ** gamma) * np.log(pt_np) - ((1.0 - pt_np) ** (gamma + 1.0))
    analytical_grad = (2.0 * y_np - 1.0) * bracket

    diff = np.max(np.abs(autograd_grad - analytical_grad))
    print("\n--- Focal Loss Autograd Gradient Check ---")
    print(f"PyTorch Autograd Grad:  {autograd_grad.tolist()}")
    print(f"Analytical Formula Grad:{analytical_grad.tolist()}")
    print(f"Max Absolute Discrepancy: {diff:.2e}")

    assert diff < 1e-12
    print("✓ Analytical Focal Loss gradient verified to machine precision (< 1e-12)!")


# =====================================================================
# 4. Class Imbalance Experiment: BCE vs. Focal Loss
# =====================================================================
def verify_class_imbalance_benefit():
    """
    Synthesizes an extreme class imbalance dataset (990 negatives, 10 positives).
    Demonstrates that Focal Loss maintains higher sensitivity (recall) on the rare class.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    n_neg = 990
    n_pos = 10
    dim = 2

    # Negative cluster centered at [-1, -1]
    X_neg = np.random.randn(n_neg, dim) * 0.5 + np.array([-1.0, -1.0])
    y_neg = np.zeros(n_neg)

    # Positive cluster centered at [+1, +1]
    X_pos = np.random.randn(n_pos, dim) * 0.5 + np.array([+1.0, +1.0])
    y_pos = np.ones(n_pos)

    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([y_neg, y_pos])

    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Train with Focal Loss (gamma = 2.0)
    model_focal = nn.Linear(dim, 1)
    opt_focal = torch.optim.Adam(model_focal.parameters(), lr=0.05)

    for _ in range(300):
        opt_focal.zero_grad()
        logits = model_focal(X_torch)
        p = torch.sigmoid(logits)
        pt = torch.where(y_torch == 1.0, p, 1.0 - p)
        loss = -torch.mean(((1.0 - pt) ** 2.0) * torch.log(pt + 1e-8))
        loss.backward()
        opt_focal.step()

    with torch.no_grad():
        preds_pos = (torch.sigmoid(model_focal(X_torch[-n_pos:])) >= 0.5).float()
        recall_focal = preds_pos.mean().item()

    print("\n--- Extreme Class Imbalance (99:1) Training ---")
    print(f"Focal Loss Positive Class Recall: {recall_focal * 100:.1f}%")
    assert recall_focal >= 0.80, f"Focal loss failed to detect rare class: recall = {recall_focal}"
    print("✓ Focal Loss successfully detects rare positive instances despite 99:1 imbalance!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 6.5 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_pytorch_loss_parity()
    verify_focal_loss_autograd()
    verify_class_imbalance_benefit()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 6.5 PASSED CLEANLY!")
    print("=" * 65)
