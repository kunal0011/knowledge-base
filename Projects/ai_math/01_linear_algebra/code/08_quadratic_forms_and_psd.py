"""
Chapter 1.8: Quadratic Forms & Positive Semi-Definite (PSD) Matrices
====================================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. Cholesky Decomposition from Scratch (A = L L^T)
2. Matrix Definiteness Classifier (PD, PSD, ND, NSD, Indefinite)
3. Multivariate Gaussian Sampler via VAE Reparameterization Trick
4. Newton's Method Optimization: Descent Direction vs Saddle-Point Ascent
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


# =====================================================================
# 1. Cholesky Decomposition from Scratch
# =====================================================================
def cholesky_decomposition_scratch(A: np.ndarray, tol: float = 1e-9) -> Optional[np.ndarray]:
    """
    Computes lower-triangular matrix L such that A = L @ L^T.
    Returns None if matrix is not strictly Positive Definite.
    """
    n, cols = A.shape
    assert n == cols, "Matrix must be square!"
    assert np.allclose(A, A.T), "Matrix must be symmetric!"

    L = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - s
                if val <= tol:
                    return None  # Non-positive pivot: not strictly positive definite!
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


# =====================================================================
# 2. Definiteness Classifier
# =====================================================================
def classify_definiteness(A: np.ndarray, tol: float = 1e-6) -> Tuple[str, Dict]:
    """
    Classifies a symmetric matrix A into:
      - Positive Definite
      - Positive Semi-Definite
      - Negative Definite
      - Negative Semi-Definite
      - Indefinite
    """
    assert np.allclose(A, A.T), "Matrix must be symmetric!"
    evals = np.linalg.eigvalsh(A)

    # Sylvester's leading principal minors
    n = A.shape[0]
    minors = [float(np.linalg.det(A[:k, :k])) for k in range(1, n + 1)]

    has_pos = np.any(evals > tol)
    has_neg = np.any(evals < -tol)
    has_zero = np.any(np.abs(evals) <= tol)

    if has_pos and not has_neg and not has_zero:
        classification = "Positive Definite"
    elif has_pos and not has_neg and has_zero:
        classification = "Positive Semi-Definite"
    elif has_neg and not has_pos and not has_zero:
        classification = "Negative Definite"
    elif has_neg and not has_pos and has_zero:
        classification = "Negative Semi-Definite"
    else:
        classification = "Indefinite"

    return classification, {
        "eigenvalues": evals,
        "leading_principal_minors": minors
    }


# =====================================================================
# 3. Multivariate Gaussian Sampler (VAE Reparameterization Trick)
# =====================================================================
def sample_multivariate_gaussian(
    mu: np.ndarray,
    Sigma: np.ndarray,
    num_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draws samples from N(mu, Sigma) using the reparameterization trick:
      x = mu + L @ epsilon,  epsilon ~ N(0, I)
    Returns:
      samples: [num_samples, d]
      sample_mean: [d]
      sample_cov: [d, d]
    """
    L = cholesky_decomposition_scratch(Sigma)
    assert L is not None, "Covariance matrix must be positive definite!"

    d = len(mu)
    # Sample standard uncorrelated white noise
    epsilon = np.random.randn(num_samples, d)  # [N, d]

    # Transform noise: x = mu + epsilon @ L^T
    samples = mu + epsilon @ L.T

    emp_mean = np.mean(samples, axis=0)
    emp_cov = np.cov(samples, rowvar=False)

    return samples, emp_mean, emp_cov


# =====================================================================
# 4. Newton's Method Descent Test
# =====================================================================
def test_newton_step(H: np.ndarray, grad: np.ndarray) -> Tuple[bool, float]:
    """
    Evaluates whether Newton step delta_w = -H^{-1} grad is a descent direction:
      Direction is descent iff grad^T @ delta_w < 0.
    """
    H_inv = np.linalg.inv(H)
    delta_w = -H_inv @ grad
    directional_derivative = float(np.dot(grad, delta_w))
    is_descent = (directional_derivative < 0.0)
    return is_descent, directional_derivative


# =====================================================================
# 5. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.8: Quadratic Forms & PSD Matrices — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Part 5 AI by Hand Cholesky Factorization
    # Sigma = [[4, 2], [2, 10]] -> L = [[2, 0], [1, 3]]
    # -------------------------------------------------------------
    Sigma_toy = np.array([
        [4.0,  2.0],
        [2.0, 10.0]
    ])
    L = cholesky_decomposition_scratch(Sigma_toy)
    expected_L = np.array([[2.0, 0.0], [1.0, 3.0]])

    print("\n[Test 1: Cholesky Factorization from Part 5]")
    print(f"  Covariance Matrix Sigma:\n{Sigma_toy}")
    print(f"  Computed Cholesky Factor L:\n{L}")
    print(f"  Matches Manual Derivation? {np.allclose(L, expected_L)}")
    print(f"  L @ L^T = Sigma Check:     {np.allclose(L @ L.T, Sigma_toy)}")

    assert np.allclose(L, expected_L), "Cholesky factor mismatch!"
    assert np.allclose(L @ L.T, Sigma_toy), "L @ L^T must reconstruct Sigma!"

    # -------------------------------------------------------------
    # Test 2: Definiteness Classification (Scenario A from notes)
    # A1 = [[2, -1], [-1, 2]] -> PD
    # A2 = [[1, 2], [2, 4]]   -> PSD
    # A3 = [[-3, 1], [1, -2]] -> ND
    # A4 = [[1, 3], [3, 1]]   -> Indefinite
    # -------------------------------------------------------------
    print("\n[Test 2: Definiteness Classification across 4 Matrices]")
    matrices = {
        "A1 (PD)":         np.array([[2.0, -1.0], [-1.0, 2.0]]),
        "A2 (PSD)":        np.array([[1.0,  2.0], [ 2.0, 4.0]]),
        "A3 (ND)":         np.array([[-3.0, 1.0], [ 1.0, -2.0]]),
        "A4 (Indefinite)": np.array([[1.0,  3.0], [ 3.0, 1.0]])
    }

    expected_labels = {
        "A1 (PD)": "Positive Definite",
        "A2 (PSD)": "Positive Semi-Definite",
        "A3 (ND)": "Negative Definite",
        "A4 (Indefinite)": "Indefinite"
    }

    for name, mat in matrices.items():
        label, info = classify_definiteness(mat)
        print(f"  Matrix {name:<16}: Result = {label:<22} (Minors: {np.round(info['leading_principal_minors'], 2)})")
        assert label == expected_labels[name], f"Classification mismatch for {name}: got {label}"

    # -------------------------------------------------------------
    # Test 3: VAE Reparameterization Gaussian Sampler
    # mu = [5, 3], Sigma = [[4, 2], [2, 10]]
    # -------------------------------------------------------------
    print("\n[Test 3: Multivariate Gaussian Sampling via Cholesky (10,000 samples)]")
    np.random.seed(42)
    mu_true = np.array([5.0, 3.0])
    samples, emp_mu, emp_sigma = sample_multivariate_gaussian(mu_true, Sigma_toy, num_samples=10000)

    print(f"  True Mean:       {mu_true}")
    print(f"  Empirical Mean:  {np.round(emp_mu, 3)}")
    print(f"  True Covariance:\n{Sigma_toy}")
    print(f"  Empirical Covariance:\n{np.round(emp_cov := emp_sigma, 3)}")

    assert np.allclose(emp_mu, mu_true, atol=0.1), "Sample mean did not converge to true mean!"
    assert np.allclose(emp_cov, Sigma_toy, atol=0.3), "Sample covariance did not converge to true covariance!"

    # -------------------------------------------------------------
    # Test 4: Newton Optimization: PD vs Indefinite Hessian
    # -------------------------------------------------------------
    print("\n[Test 4: Newton Step Descent vs Saddle Ascent]")
    H_pd = np.array([[4.0, 1.0], [1.0, 4.0]])        # PD Hessian (curving up everywhere)
    H_indef = np.array([[1.0, 3.0], [3.0, 1.0]])     # Indefinite Hessian (saddle: evals 4 and -2)

    # Gradient along positive curvature direction of H_pd
    grad_pd = np.array([1.0, 1.0])
    # Gradient along negative curvature direction of H_indef (v = [1, -1]^T, lambda = -2)
    grad_saddle = np.array([1.0, -1.0])

    is_desc_pd, d_pd = test_newton_step(H_pd, grad_pd)
    is_desc_indef, d_indef = test_newton_step(H_indef, grad_saddle)

    print(f"  PD Hessian (bowl):      Directional Derivative = {d_pd:+.4f} -> Guaranteed Descent? {is_desc_pd}")
    print(f"  Indefinite Hessian (saddle): Directional Derivative = {d_indef:+.4f} -> Guaranteed Descent? {is_desc_indef}")

    assert is_desc_pd == True, "PD Hessian must guarantee a descent direction (d < 0)!"
    assert is_desc_indef == False, "Along negative curvature, Newton step produces an ASCENT step (d > 0)!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
