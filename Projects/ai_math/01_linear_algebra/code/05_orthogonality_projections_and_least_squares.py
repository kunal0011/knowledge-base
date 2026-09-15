"""
Chapter 1.5: Orthogonality, Projections & Least Squares
=======================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. Modified Gram-Schmidt (MGS) QR Decomposition
2. Orthogonal Projection Matrix Generator & Invariant Checks (P^2 = P, P^T = P)
3. Ordinary Least Squares (OLS) Solver (Normal Equations vs QR Back-substitution)
4. Residual Orthogonality Verification (A^T e = 0)
5. Signal Norm Preservation across Deep Layers: Gaussian vs Orthogonal Initialization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


# =====================================================================
# 1. Modified Gram-Schmidt QR Decomposition
# =====================================================================
def modified_gram_schmidt_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs QR factorization A = Q @ R using Modified Gram-Schmidt (MGS).
    Input:  A of shape (m, n) with linearly independent columns (m >= n)
    Output: Q of shape (m, n) with orthonormal columns (Q^T Q = I_n)
            R of shape (n, n) upper-triangular
    """
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    V = A.astype(float).copy()

    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        assert R[i, i] > 1e-12, f"Column {i} is linearly dependent!"
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

    return Q, R


# =====================================================================
# 2. Orthogonal Projection Matrix
# =====================================================================
def compute_projection_matrix(A: np.ndarray) -> np.ndarray:
    """
    Computes orthogonal projection matrix onto C(A):
      P = A @ (A^T @ A)^{-1} @ A^T
    """
    # Using pseudoinverse for numerical safety: P = A @ pinv(A)
    AtA = A.T @ A
    AtA_inv = np.linalg.inv(AtA)
    P = A @ AtA_inv @ A.T
    return P


# =====================================================================
# 3. OLS Solvers (Normal Equations vs QR)
# =====================================================================
def solve_ols_normal(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves A x = b via Normal Equations:
      x_hat = (A^T A)^{-1} A^T b
    Returns:
      x_hat, projection p = A x_hat, residual e = b - p
    """
    x_hat = np.linalg.inv(A.T @ A) @ (A.T @ b)
    p = A @ x_hat
    e = b - p
    return x_hat, p, e


def solve_ols_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves A x = b via QR factorization:
      R x_hat = Q^T b  (solved by back-substitution)
    """
    Q, R = modified_gram_schmidt_qr(A)
    qty = Q.T @ b
    # Back-substitution for upper-triangular R
    n = R.shape[0]
    x_hat = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_hat[i] = (qty[i] - np.dot(R[i, i+1:], x_hat[i+1:])) / R[i, i]
    return x_hat


# =====================================================================
# 4. Deep Layer Norm Preservation: Gaussian vs Orthogonal Init
# =====================================================================
def simulate_deep_signal_propagation(
    num_layers: int = 50,
    dim: int = 64
) -> Tuple[float, float]:
    """
    Propagates a unit-norm vector through num_layers linear layers:
      Case 1: Standard Gaussian init (W ~ N(0, 1/dim))
      Case 2: Orthogonal init (W^T W = I)
    Returns:
      (final_norm_gaussian, final_norm_orthogonal)
    """
    x_gauss = torch.randn(1, dim)
    x_gauss = x_gauss / torch.norm(x_gauss)

    x_ortho = x_gauss.clone()

    for _ in range(num_layers):
        # Gaussian layer
        W_g = torch.randn(dim, dim) / np.sqrt(dim)
        x_gauss = torch.matmul(x_gauss, W_g)

        # Orthogonal layer
        W_o = torch.empty(dim, dim)
        nn.init.orthogonal_(W_o)
        x_ortho = torch.matmul(x_ortho, W_o)

    norm_gauss = torch.norm(x_gauss).item()
    norm_ortho = torch.norm(x_ortho).item()
    return norm_gauss, norm_ortho


# =====================================================================
# 5. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.5: Orthogonality & Projections — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Part 5 AI by Hand OLS Linear Regression
    # Data: (1, 1), (2, 2), (3, 2)
    # Expected: x_hat = [2/3, 1/2]^T = [0.6667, 0.5000]
    # -------------------------------------------------------------
    A_toy = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0]
    ])
    b_toy = np.array([1.0, 2.0, 2.0])

    x_norm, p_norm, e_norm = solve_ols_normal(A_toy, b_toy)
    x_qr = solve_ols_qr(A_toy, b_toy)

    print("\n[Test 1: OLS Regression from Part 5]")
    print(f"  Normal Equations Solution: {x_norm} (Expected: [0.6667, 0.5])")
    print(f"  QR Decomposition Solution: {x_qr}")
    print(f"  Solutions Match?           {np.allclose(x_norm, x_qr)}")
    print(f"  Residual Error Vector e:   {e_norm}")
    print(f"  Orthogonality A^T @ e:     {A_toy.T @ e_norm}")

    assert np.allclose(x_norm, [2/3, 1/2]), "OLS parameter estimate mismatch!"
    assert np.allclose(x_qr, x_norm), "QR and Normal Equations must match!"
    assert np.allclose(A_toy.T @ e_norm, 0.0), "Residual error must be orthogonal to A!"

    # -------------------------------------------------------------
    # Test 2: Projection Matrix Properties (Idempotence & Symmetry)
    # -------------------------------------------------------------
    P = compute_projection_matrix(A_toy)
    is_idempotent = np.allclose(P @ P, P)
    is_symmetric = np.allclose(P.T, P)
    trace_P = np.trace(P)

    print("\n[Test 2: Orthogonal Projection Matrix Invariants]")
    print(f"  P is Idempotent (P^2 = P)? {is_idempotent}")
    print(f"  P is Symmetric  (P^T = P)? {is_symmetric}")
    print(f"  Tr(P) = rank(A) = 2?       {np.isclose(trace_P, 2.0)}")

    assert is_idempotent == True, "P^2 must equal P!"
    assert is_symmetric == True, "P^T must equal P!"
    assert np.isclose(trace_P, 2.0), "Trace of projection matrix must equal subspace dimension!"

    # -------------------------------------------------------------
    # Test 3: Gram-Schmidt QR Factorization (Scenario A from notes)
    # a1 = [1, 1, 0]^T, a2 = [1, 0, 1]^T, a3 = [0, 1, 1]^T
    # -------------------------------------------------------------
    A_gs = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ])
    Q, R = modified_gram_schmidt_qr(A_gs)

    print("\n[Test 3: Gram-Schmidt QR Factorization]")
    print(f"  Q^T @ Q = I Check: {np.allclose(Q.T @ Q, np.eye(3))}")
    print(f"  Q @ R = A Check:   {np.allclose(Q @ R, A_gs)}")
    print(f"  R is Upper Triangular? {np.allclose(R, np.triu(R))}")

    assert np.allclose(Q.T @ Q, np.eye(3)), "Columns of Q must be orthonormal!"
    assert np.allclose(Q @ R, A_gs), "Q @ R must reconstruct original matrix A!"
    assert np.allclose(R, np.triu(R)), "R must be upper-triangular!"

    # -------------------------------------------------------------
    # Test 4: Deep Neural Network Signal Propagation (50 Layers)
    # -------------------------------------------------------------
    print("\n[Test 4: 50-Layer Deep Signal Norm Conservation]")
    norm_gauss, norm_ortho = simulate_deep_signal_propagation(num_layers=50, dim=64)
    print(f"  Input Vector Initial Norm:        1.0000")
    print(f"  Output Norm after 50 GAUSSIAN layers:    {norm_gauss:.4f}")
    print(f"  Output Norm after 50 ORTHOGONAL layers:  {norm_ortho:.4f}")

    assert np.isclose(norm_ortho, 1.0, atol=1e-3), "Orthogonal layers must preserve exact unit norm across 50 layers!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
