"""
Chapter 1.2: Vector Norms, Metrics & Inner Products
===================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. Universal Lp Vector Norms and Matrix Norms (Frobenius & Spectral)
2. Cosine Similarity & Scaled Attention Logits
3. Triangle Inequality Verification across p-values (p >= 1 vs p < 1)
4. Gradient Clipping by Norm in PyTorch
5. L1 vs L2 Weight Regularization Dynamics
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Union


# =====================================================================
# 1. Universal Lp Vector & Matrix Norm Engine
# =====================================================================
def compute_vector_norm(x: np.ndarray, p: Union[int, float, str] = 2) -> float:
    """
    Computes vector norm ||x||_p for:
      - p = 0: L0 pseudo-norm (count of non-zero elements)
      - p = 1: L1 Manhattan norm
      - p = 2: L2 Euclidean norm
      - p = np.inf or 'inf': L_infinity Chebyshev/max norm
      - p >= 1: arbitrary Minkowski p-norm
    """
    x_arr = np.asarray(x, dtype=float).flatten()
    if p == 0:
        return float(np.sum(x_arr != 0.0))
    elif p == 1:
        return float(np.sum(np.abs(x_arr)))
    elif p == 2:
        return float(np.sqrt(np.sum(x_arr ** 2)))
    elif p == np.inf or p == "inf":
        return float(np.max(np.abs(x_arr)))
    else:
        # General Minkowski norm
        return float(np.sum(np.abs(x_arr) ** p) ** (1.0 / p))


def compute_matrix_norms(A: np.ndarray) -> Tuple[float, float]:
    """
    Computes:
      1. Frobenius Norm: ||A||_F = sqrt(sum(A_ij^2))
      2. Spectral Norm:  ||A||_2 = sigma_max(A) (largest singular value)
    """
    frobenius = float(np.sqrt(np.sum(A ** 2)))
    # Spectral norm is max singular value
    _, S, _ = np.linalg.svd(A)
    spectral = float(np.max(S))
    return frobenius, spectral


# =====================================================================
# 2. Cosine Similarity & Attention Matching Engine
# =====================================================================
def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes cosine similarity between two 1D vectors:
      cos_theta = (u . v) / (||u||_2 * ||v||_2)
    """
    dot_product = np.dot(u, v)
    norm_u = compute_vector_norm(u, p=2)
    norm_v = compute_vector_norm(v, p=2)
    return float(dot_product / (max(norm_u * norm_v, eps)))


def batch_scaled_cosine_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Computes normalized cosine attention weights between queries [B, d] and keys [N, d]:
      sim(q, k) = (q / ||q||) @ (k / ||k||)^T
      weights = softmax(sim / temperature)
    """
    q_norm = nn.functional.normalize(queries, p=2, dim=-1)
    k_norm = nn.functional.normalize(keys, p=2, dim=-1)
    sim_matrix = torch.matmul(q_norm, k_norm.t())  # [B, N]
    attention_weights = torch.softmax(sim_matrix / temperature, dim=-1)
    return attention_weights


# =====================================================================
# 3. Triangle Inequality Stress Tester
# =====================================================================
def test_triangle_inequality(
    p: float,
    dim: int = 5,
    num_trials: int = 1000
) -> Tuple[bool, float]:
    """
    Empirically checks whether ||u + v||_p <= ||u||_p + ||v||_p holds.
    Returns:
      - satisfies_inequality: bool
      - max_violation_ratio: float (ratio of ||u+v|| / (||u|| + ||v||); > 1 indicates violation)
    """
    max_ratio = 0.0
    for _ in range(num_trials):
        u = np.random.randn(dim)
        v = np.random.randn(dim)
        if p == 0:
            norm_u = float(np.sum(u != 0))
            norm_v = float(np.sum(v != 0))
            norm_sum = float(np.sum((u + v) != 0))
        else:
            norm_u = float(np.sum(np.abs(u) ** p) ** (1.0 / p))
            norm_v = float(np.sum(np.abs(v) ** p) ** (1.0 / p))
            norm_sum = float(np.sum(np.abs(u + v) ** p) ** (1.0 / p))

        ratio = norm_sum / max(norm_u + norm_v, 1e-12)
        if ratio > max_ratio:
            max_ratio = ratio

    satisfies = (max_ratio <= 1.0 + 1e-6)
    return satisfies, max_ratio


# =====================================================================
# 4. Gradient Clipping by Norm in PyTorch
# =====================================================================
def clip_gradient_by_norm(
    parameters: List[torch.nn.Parameter],
    max_norm: float
) -> Tuple[float, float]:
    """
    Clips gradient norm across a collection of parameters:
      total_norm = sqrt(sum(||grad_p||_2^2))
      if total_norm > max_norm:
          grad_p = grad_p * (max_norm / total_norm)
    Returns:
      (original_norm, clipped_norm)
    """
    total_sq_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq_norm += torch.sum(p.grad ** 2).item()
    original_norm = np.sqrt(total_sq_norm)

    clip_coef = max_norm / max(original_norm, 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

    # Recompute norm after clipping
    post_sq_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            post_sq_norm += torch.sum(p.grad ** 2).item()
    clipped_norm = np.sqrt(post_sq_norm)

    return original_norm, clipped_norm


# =====================================================================
# 5. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.2: Vector Norms, Metrics & Inner Products — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Scenario A from notes (Vector x = [3, -4, 0, 12]^T)
    # -------------------------------------------------------------
    x = np.array([3.0, -4.0, 0.0, 12.0])
    l0 = compute_vector_norm(x, p=0)
    l1 = compute_vector_norm(x, p=1)
    l2 = compute_vector_norm(x, p=2)
    linf = compute_vector_norm(x, p=np.inf)

    print("\n[Test 1: Lp Norms of x = [3, -4, 0, 12]^T]")
    print(f"  L0 pseudo-norm (non-zero count): {l0}")
    print(f"  L1 norm (Manhattan):             {l1}")
    print(f"  L2 norm (Euclidean):             {l2}")
    print(f"  L_inf norm (Chebyshev):          {linf}")

    assert l0 == 3.0, f"L0 must be 3, got {l0}"
    assert l1 == 19.0, f"L1 must be 19, got {l1}"
    assert l2 == 13.0, f"L2 must be 13, got {l2}"
    assert linf == 12.0, f"L_inf must be 12, got {linf}"
    assert linf <= l2 <= l1, "Hierarchy ||x||_inf <= ||x||_2 <= ||x||_1 must hold!"

    # -------------------------------------------------------------
    # Test 2: Cauchy-Schwarz Inequality & Part 5 Attention Demo
    # Query q = [2, 1, -2]^T
    # Key k1  = [4, 2, -4]^T (collinear)
    # Key k2  = [1, -2, 0]^T (orthogonal)
    # -------------------------------------------------------------
    print("\n[Test 2: Cosine Similarity & Attention Matching]")
    q = np.array([2.0, 1.0, -2.0])
    k1 = np.array([4.0, 2.0, -4.0])
    k2 = np.array([1.0, -2.0, 0.0])

    sim_q_k1 = cosine_similarity(q, k1)
    sim_q_k2 = cosine_similarity(q, k2)

    print(f"  Cosine Similarity(q, k1) [Collinear]:   {sim_q_k1:.4f} (Expected: 1.0000)")
    print(f"  Cosine Similarity(q, k2) [Orthogonal]:  {sim_q_k2:.4f} (Expected: 0.0000)")

    assert np.isclose(sim_q_k1, 1.0), "Collinear vectors must have cosine similarity = 1.0!"
    assert np.isclose(sim_q_k2, 0.0), "Orthogonal vectors must have cosine similarity = 0.0!"

    # -------------------------------------------------------------
    # Test 3: Matrix Norms (Frobenius vs Spectral)
    # W = [[1, 2], [3, 4]]
    # -------------------------------------------------------------
    print("\n[Test 3: Matrix Norms]")
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    frob, spec = compute_matrix_norms(W)
    print(f"  Matrix W:\n{W}")
    print(f"  Frobenius Norm: {frob:.4f} (Expected: sqrt(1+4+9+16) = sqrt(30) = 5.4772)")
    print(f"  Spectral Norm:  {spec:.4f} (Max singular value)")

    assert np.isclose(frob, np.sqrt(30.0)), "Frobenius norm check failed!"
    assert spec <= frob, "Spectral norm must be bounded above by Frobenius norm!"

    # -------------------------------------------------------------
    # Test 4: Triangle Inequality Test across p >= 1 and p < 1
    # -------------------------------------------------------------
    print("\n[Test 4: Triangle Inequality Validation]")
    p_values = [2.0, 1.0, 0.5]
    for p in p_values:
        satisfies, max_ratio = test_triangle_inequality(p=p, dim=4, num_trials=2000)
        print(f"  p = {p:<3}: Satisfies Triangle Inequality? {str(satisfies):<5} (Max ||u+v|| / (||u||+||v||): {max_ratio:.4f})")
        if p >= 1.0:
            assert satisfies == True, f"p={p} MUST satisfy the triangle inequality!"
        else:
            assert satisfies == False, f"p={p} MUST violate the triangle inequality!"

    # -------------------------------------------------------------
    # Test 5: Gradient Clipping by Norm in PyTorch
    # -------------------------------------------------------------
    print("\n[Test 5: Gradient Clipping by Norm in PyTorch]")
    param = nn.Parameter(torch.randn(10, 10))
    # Inject an exploding gradient of length ~100
    param.grad = torch.randn(10, 10) * 10.0
    initial_norm = float(torch.norm(param.grad, p=2).item())

    max_norm_threshold = 1.0
    orig_n, clipped_n = clip_gradient_by_norm([param], max_norm=max_norm_threshold)

    print(f"  Original Exploding Gradient Norm: {orig_n:.4f}")
    print(f"  Target Max Norm Threshold:        {max_norm_threshold:.4f}")
    print(f"  Clipped Gradient Norm:            {clipped_n:.4f}")

    assert np.isclose(clipped_n, max_norm_threshold, atol=1e-4), "Clipped norm must match max_norm!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
