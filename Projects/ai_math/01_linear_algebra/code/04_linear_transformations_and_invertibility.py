"""
Chapter 1.4: Linear Transformations & Invertibility
===================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. Gauss-Jordan Matrix Inversion from Scratch with Singularity Detection
2. Determinant & Trace Invariance under Similarity Transformations (Change of Basis)
3. Volume Scaling Factor Verification
4. Invertible Coupling Block (Normalizing Flow) in PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


# =====================================================================
# 1. Gauss-Jordan Matrix Inversion from Scratch
# =====================================================================
def invert_matrix_gauss_jordan(A: np.ndarray, tol: float = 1e-9) -> Optional[np.ndarray]:
    """
    Computes A^{-1} via augmented Gauss-Jordan elimination [A | I_n].
    Returns None if matrix is singular (det == 0).
    """
    M = A.astype(float).copy()
    n, cols = M.shape
    assert n == cols, "Matrix must be square to be invertible!"

    # Augment with identity matrix
    aug = np.hstack([M, np.eye(n)])

    for i in range(n):
        # Partial pivoting: find max entry in column i from row i downward
        max_row = i + np.argmax(np.abs(aug[i:, i]))
        if np.abs(aug[max_row, i]) < tol:
            return None  # Singular matrix detected!

        # Swap rows
        aug[[i, max_row]] = aug[[max_row, i]]

        # Scale pivot row to have leading 1
        pivot = aug[i, i]
        aug[i] /= pivot

        # Eliminate all other entries in column i
        for r in range(n):
            if r != i:
                factor = aug[r, i]
                aug[r] -= factor * aug[i]

    # Clean floating-point artifacts
    inv = aug[:, n:]
    inv[np.abs(inv) < tol] = 0.0
    return inv


# =====================================================================
# 2. Change of Basis & Invariants (Trace & Determinant)
# =====================================================================
def change_of_basis(A: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Computes the representation of operator A in new basis defined by columns of P:
      B = P^{-1} @ A @ P
    Verifies that trace and determinant are invariant:
      Tr(B) == Tr(A) and det(B) == det(A)
    """
    P_inv = invert_matrix_gauss_jordan(P)
    assert P_inv is not None, "Change of basis matrix P must be invertible!"

    B = P_inv @ A @ P

    tr_A, tr_B = np.trace(A), np.trace(B)
    det_A, det_B = np.linalg.det(A), np.linalg.det(B)

    invariants_hold = np.isclose(tr_A, tr_B) and np.isclose(det_A, det_B)
    return B, invariants_hold


# =====================================================================
# 3. Normalizing Flow Affine Coupling Block in PyTorch
# =====================================================================
class AffineCouplingLayer(nn.Module):
    """
    Invertible RealNVP-style Affine Coupling Layer:
      Splits input x = [x1, x2]
      Forward:
        z1 = x1
        z2 = x2 * exp(s(x1)) + t(x1)
      Log-Determinant of Jacobian:
        log|det(J)| = sum(s(x1))
      Inverse:
        x1 = z1
        x2 = (z2 - t(z1)) * exp(-s(z1))
    """
    def __init__(self, dim: int = 4):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        # Scale and translation neural networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, 16),
            nn.Tanh(),
            nn.Linear(16, dim - self.split_dim)
        )
        self.trans_net = nn.Sequential(
            nn.Linear(self.split_dim, 16),
            nn.ReLU(),
            nn.Linear(16, dim - self.split_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = x[:, :self.split_dim]
        x2 = x[:, self.split_dim:]

        s = self.scale_net(x1)
        t = self.trans_net(x1)

        z1 = x1
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([z1, z2], dim=-1)

        # Log determinant is sum of scale factors (diagonal of triangular Jacobian)
        log_det_J = torch.sum(s, dim=-1)
        return z, log_det_J

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z1 = z[:, :self.split_dim]
        z2 = z[:, self.split_dim:]

        s = self.scale_net(z1)
        t = self.trans_net(z1)

        x1 = z1
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=-1)
        return x


# =====================================================================
# 4. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.4: Linear Transformations & Invertibility — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Scenario A from notes (3x3 Matrix Inversion)
    # A = [[1, 0, 2], [2, -1, 3], [4, 1, 8]]
    # Expected det = 1, A^{-1} = [[-11, 2, 2], [-4, 0, 1], [6, -1, -1]]
    # -------------------------------------------------------------
    A = np.array([
        [1.0,  0.0, 2.0],
        [2.0, -1.0, 3.0],
        [4.0,  1.0, 8.0]
    ])
    A_inv = invert_matrix_gauss_jordan(A)
    det_A = np.linalg.det(A)

    expected_inv = np.array([
        [-11.0,  2.0,  2.0],
        [ -4.0,  0.0,  1.0],
        [  6.0, -1.0, -1.0]
    ])

    print("\n[Test 1: 3x3 Gauss-Jordan Matrix Inversion]")
    print(f"  Matrix A Determinant: {det_A:.4f} (Expected: 1.0)")
    print(f"  Computed Inverse A^{{-1}}:\n{A_inv}")
    print(f"  Matches Manual Derivation? {np.allclose(A_inv, expected_inv)}")
    print(f"  A @ A^{{-1}} = I Check:     {np.allclose(A @ A_inv, np.eye(3))}")

    assert np.isclose(det_A, 1.0), "Determinant must be 1.0!"
    assert np.allclose(A_inv, expected_inv), "Inverse mismatch with manual derivation!"
    assert np.allclose(A @ A_inv, np.eye(3)), "A @ A^{-1} must equal Identity!"

    # -------------------------------------------------------------
    # Test 2: Singular Matrix Detection (Scenario C)
    # M = [[1, 2], [2, 4]] -> Singular (det = 0)
    # -------------------------------------------------------------
    M = np.array([[1.0, 2.0], [2.0, 4.0]])
    M_inv = invert_matrix_gauss_jordan(M)

    print("\n[Test 2: Singular Matrix Detection]")
    print(f"  Matrix M:\n{M}")
    print(f"  Is Invertible? {M_inv is not None} (Expected: False / None)")
    assert M_inv is None, "Singular matrix must return None!"

    # -------------------------------------------------------------
    # Test 3: Change of Basis & Invariants (Scenario B)
    # A = diag(2, 5), P = [[1, -1], [1, 1]]
    # -------------------------------------------------------------
    A_diag = np.array([[2.0, 0.0], [0.0, 5.0]])
    P_rot = np.array([[1.0, -1.0], [1.0, 1.0]])
    B_rot, invariants_ok = change_of_basis(A_diag, P_rot)

    print("\n[Test 3: Change of Basis & Invariants (Trace & Det)]")
    print(f"  Original Matrix A:\n{A_diag}")
    print(f"  Similar Matrix B = P^{{-1}} A P:\n{B_rot}")
    print(f"  Tr(A) = {np.trace(A_diag):.2f}, Tr(B) = {np.trace(B_rot):.2f}")
    print(f"  det(A) = {np.linalg.det(A_diag):.2f}, det(B) = {np.linalg.det(B_rot):.2f}")
    print(f"  Invariants Preserved? {invariants_ok}")

    assert invariants_ok == True, "Trace and Determinant must be identical for similar matrices!"

    # -------------------------------------------------------------
    # Test 4: Part 5 AI by Hand Normalizing Flow Matrix Test
    # A_flow = [[2, 1], [0, 3]], x = [3, 2]^T -> z = [8, 6]^T
    # -------------------------------------------------------------
    A_flow = np.array([[2.0, 1.0], [0.0, 3.0]])
    x_flow = np.array([3.0, 2.0])
    z_flow = A_flow @ x_flow
    det_flow = np.linalg.det(A_flow)
    A_flow_inv = invert_matrix_gauss_jordan(A_flow)
    x_recon = A_flow_inv @ z_flow

    print("\n[Test 4: Part 5 AI by Hand Invertible Layer Demo]")
    print(f"  Input x:               {x_flow}")
    print(f"  Forward z = A x:       {z_flow} (Expected: [8, 6])")
    print(f"  Volume Scaling det(A): {det_flow:.2f} (Expected: 6.0)")
    print(f"  Reconstructed x:       {x_recon}")

    assert np.allclose(z_flow, [8.0, 6.0]), "Forward mapping failed!"
    assert np.isclose(det_flow, 6.0), "Determinant must be 6.0!"
    assert np.allclose(x_recon, x_flow), "Inverse reconstruction failed!"

    # -------------------------------------------------------------
    # Test 5: PyTorch Affine Coupling Block (RealNVP) Test
    # -------------------------------------------------------------
    print("\n[Test 5: PyTorch Invertible Coupling Block]")
    torch.manual_seed(42)
    dim = 6
    coupling = AffineCouplingLayer(dim=dim)

    # 10 random samples in R^6
    X_in = torch.randn(10, dim)
    Z_latent, log_det = coupling(X_in)
    X_recovered = coupling.inverse(Z_latent)

    recon_error = torch.max(torch.abs(X_recovered - X_in)).item()
    print(f"  Input Batch Shape:        {X_in.shape}")
    print(f"  Latent Batch Shape:       {Z_latent.shape}")
    print(f"  Mean Log-Determinant:     {torch.mean(log_det).item():.4f}")
    print(f"  Max Reconstruction Error: {recon_error:.6e}")

    assert recon_error < 1e-6, "Normalizing flow reconstruction must be exact!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
