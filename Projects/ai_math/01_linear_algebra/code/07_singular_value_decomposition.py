"""
Chapter 1.7: Singular Value Decomposition (SVD)
================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. SVD from Scratch via Gramian Eigendecomposition (A = U Sigma V^T)
2. Eckart-Young-Mirsky Theorem Verification (Optimal Rank-k Approximation)
3. Moore-Penrose Pseudoinverse & 4 Penrose Axiom Verification
4. Deep Neural Network Weight Compression via Truncated SVD in PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict


# =====================================================================
# 1. SVD from Scratch
# =====================================================================
def compute_svd_scratch(A: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes SVD A = U @ Sigma @ V^T from scratch via eigendecomposition of A^T A.
    Returns:
      U: shape (m, r)
      Sigma: 1D array of singular values (length r)
      Vt: shape (r, n)
    """
    m, n = A.shape
    AtA = A.T @ A
    # Eigendecomposition of A^T A
    evals, V = np.linalg.eigh(AtA)

    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    V = V[:, idx]

    # Non-zero singular values
    positive_mask = evals > tol
    sigma = np.sqrt(np.maximum(evals[positive_mask], 0.0))
    r = len(sigma)

    V_r = V[:, :r]

    # Compute left singular vectors: u_i = (A v_i) / sigma_i
    U_r = np.zeros((m, r))
    for i in range(r):
        U_r[:, i] = (A @ V_r[:, i]) / sigma[i]

    return U_r, sigma, V_r.T


# =====================================================================
# 2. Eckart-Young Low-Rank Approximation
# =====================================================================
def eckart_young_approx(A: np.ndarray, k: int) -> Tuple[np.ndarray, float, float]:
    """
    Computes optimal rank-k approximation A_k via truncated SVD.
    Returns:
      A_k: rank-k matrix
      empirical_frobenius_error: ||A - A_k||_F
      theoretical_frobenius_error: sqrt(sum_{j=k+1}^r sigma_j^2)
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(S))

    A_k = (U[:, :k] * S[:k]) @ Vt[:k, :]
    empirical_err = float(np.linalg.norm(A - A_k, ord='fro'))
    theoretical_err = float(np.sqrt(np.sum(S[k:] ** 2))) if k < len(S) else 0.0

    return A_k, empirical_err, theoretical_err


# =====================================================================
# 3. Moore-Penrose Pseudoinverse & Penrose Conditions
# =====================================================================
def compute_pseudoinverse(A: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, Dict[str, bool]]:
    """
    Computes A^+ = V @ Sigma^+ @ U^T and checks the 4 Penrose conditions:
      1. A @ A^+ @ A = A
      2. A^+ @ A @ A^+ = A^+
      3. (A @ A^+)^T = A @ A^+
      4. (A^+ @ A)^T = A^+ @ A
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.zeros_like(S)
    mask = S > tol
    S_inv[mask] = 1.0 / S[mask]

    A_plus = (Vt.T * S_inv) @ U.T

    # Verify 4 Penrose Conditions
    cond1 = np.allclose(A @ A_plus @ A, A, atol=1e-5)
    cond2 = np.allclose(A_plus @ A @ A_plus, A_plus, atol=1e-5)
    cond3 = np.allclose((A @ A_plus).T, A @ A_plus, atol=1e-5)
    cond4 = np.allclose((A_plus @ A).T, A_plus @ A, atol=1e-5)

    checks = {
        "cond1_A_Aplus_A": cond1,
        "cond2_Aplus_A_Aplus": cond2,
        "cond3_A_Aplus_symmetric": cond3,
        "cond4_Aplus_A_symmetric": cond4
    }
    return A_plus, checks


# =====================================================================
# 4. PyTorch Weight Compression via SVD
# =====================================================================
def compress_linear_layer_svd(
    weight: torch.Tensor,
    rank: int
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Decomposes weight [d_out, d_in] into B [d_out, rank] and A [rank, d_in]
    using truncated SVD.
    Returns:
      B, A, param_savings_percent, relative_error
    """
    d_out, d_in = weight.shape
    U, S, V = torch.svd(weight)

    # Truncate to rank
    U_k = U[:, :rank]                  # [d_out, rank]
    S_k = torch.diag(S[:rank])         # [rank, rank]
    V_k = V[:, :rank]                  # [d_in, rank]

    # Decompose into B and A: B = U_k @ sqrt(S_k), A = sqrt(S_k) @ V_k^T
    sqrt_S = torch.diag(torch.sqrt(S[:rank]))
    B = torch.matmul(U_k, sqrt_S)      # [d_out, rank]
    A = torch.matmul(sqrt_S, V_k.t())  # [rank, d_in]

    orig_params = d_out * d_in
    comp_params = (d_out * rank) + (rank * d_in)
    param_savings = (1.0 - comp_params / orig_params) * 100.0

    weight_recon = torch.matmul(B, A)
    rel_error = (torch.norm(weight - weight_recon) / torch.norm(weight)).item()

    return B, A, param_savings, rel_error


# =====================================================================
# 5. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.7: Singular Value Decomposition (SVD) — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Part 5 AI by Hand SVD
    # A = [[3, 2], [2, 3], [2, -2]] -> singular values: 5 and 3
    # -------------------------------------------------------------
    A_toy = np.array([
        [3.0,  2.0],
        [2.0,  3.0],
        [2.0, -2.0]
    ])
    U, S, Vt = compute_svd_scratch(A_toy)

    print("\n[Test 1: SVD from Scratch (Part 5 Illustration)]")
    print(f"  Matrix A:\n{A_toy}")
    print(f"  Singular Values: {S} (Expected: [5.0, 3.0])")
    print(f"  Left Singular Vectors U:\n{U}")
    print(f"  Right Singular Vectors V:\n{Vt.T}")

    # Reconstruction test
    reconstructed = (U * S) @ Vt
    print(f"  A = U @ Sigma @ V^T Check: {np.allclose(reconstructed, A_toy)}")

    assert np.allclose(S, [5.0, 3.0]), "Singular values mismatch!"
    assert np.allclose(reconstructed, A_toy), "Reconstruction check failed!"
    assert np.allclose(U.T @ U, np.eye(2)), "U must have orthonormal columns!"
    assert np.allclose(Vt @ Vt.T, np.eye(2)), "V must have orthonormal columns!"

    # -------------------------------------------------------------
    # Test 2: Eckart-Young Rank-1 Approximation
    # -------------------------------------------------------------
    print("\n[Test 2: Eckart-Young-Mirsky Theorem Verification]")
    A_1, emp_err, theo_err = eckart_young_approx(A_toy, k=1)

    print(f"  Rank-1 Approximated Matrix A_1:\n{A_1}")
    print(f"  Empirical Frobenius Error:   {emp_err:.4f}")
    print(f"  Theoretical Frobenius Error: {theo_err:.4f} (Expected: sigma_2 = 3.0)")
    print(f"  Eckart-Young Error Match?    {np.isclose(emp_err, theo_err)}")

    assert np.isclose(emp_err, 3.0), f"Frobenius error must be exactly sigma_2=3.0, got {emp_err}"
    assert np.isclose(emp_err, theo_err), "Eckart-Young theorem failed!"

    # -------------------------------------------------------------
    # Test 3: Moore-Penrose Pseudoinverse & 4 Penrose Conditions
    # -------------------------------------------------------------
    print("\n[Test 3: Moore-Penrose Pseudoinverse & 4 Penrose Conditions]")
    A_plus, checks = compute_pseudoinverse(A_toy)

    print(f"  Computed Pseudoinverse A^+:\n{A_plus}")
    for cond_name, passed in checks.items():
        print(f"  Checking {cond_name:<25}: {passed}")
        assert passed == True, f"Penrose condition {cond_name} failed!"

    # -------------------------------------------------------------
    # Test 4: PyTorch Weight Compression Demo (512x512 Layer to Rank 32)
    # -------------------------------------------------------------
    print("\n[Test 4: PyTorch Linear Layer Compression via SVD]")
    torch.manual_seed(42)
    d_out, d_in, rank = 512, 512, 32

    # Synthesize realistic low-intrinsic-rank neural network weights (intrinsic rank = 32)
    true_latent = torch.randn(d_out, 32) @ torch.randn(32, d_in) + 0.001 * torch.randn(d_out, d_in)

    B, A_adapt, savings, rel_err = compress_linear_layer_svd(true_latent, rank=rank)

    print(f"  Original Parameter Count:  {d_out * d_in:,}")
    print(f"  Compressed Parameter Count:{(d_out * rank) + (rank * d_in):,}")
    print(f"  Parameter Savings:         {savings:.2f}% (Expected: 87.50%)")
    print(f"  Relative Reconstruction Err: {rel_err * 100:.4f}%")

    assert np.isclose(savings, 87.5), "Parameter savings calculation mismatch!"
    assert rel_err < 0.01, f"Reconstruction error on rank-32 weight should be < 1%, got {rel_err * 100:.2f}%!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
