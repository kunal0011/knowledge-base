"""
Chapter 1.6: Eigendecomposition & Spectral Theory
=================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. Symmetric Matrix Eigendecomposition & Spectral Reconstruction (A = Q Lambda Q^T)
2. Invariants: Trace Sum & Determinant Product Laws
3. Multiplicity & Defective Matrix Detection (AM vs GM)
4. RNN Dynamic Horizon Simulation (Spectral Radius: Exploding, Vanishing, and Stable Regimes)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List


# =====================================================================
# 1. Symmetric Eigendecomposition & Spectral Verification
# =====================================================================
def decompose_symmetric_matrix(A: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes eigenvalues and orthonormal eigenvectors for a real symmetric matrix A.
    Verifies:
      1. A = Q @ Lambda @ Q^T
      2. Q^T @ Q = I
      3. Tr(A) = sum(lambda_i)
      4. det(A) = prod(lambda_i)
    """
    assert np.allclose(A, A.T), "Matrix must be symmetric!"
    eigenvalues, Q = np.linalg.eigh(A)  # Guaranteed real for symmetric

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    Q = Q[:, idx]

    Lambda = np.diag(eigenvalues)
    reconstructed = Q @ Lambda @ Q.T

    return {
        "eigenvalues": eigenvalues,
        "Q": Q,
        "Lambda": Lambda,
        "reconstructed": reconstructed,
        "is_orthonormal": np.allclose(Q.T @ Q, np.eye(A.shape[0])),
        "is_reconstructed": np.allclose(reconstructed, A),
        "trace_matches": np.isclose(np.trace(A), np.sum(eigenvalues)),
        "det_matches": np.isclose(np.linalg.det(A), np.prod(eigenvalues))
    }


# =====================================================================
# 2. Defectiveness & Multiplicity Tester
# =====================================================================
def test_defectiveness(A: np.ndarray, tol: float = 1e-5) -> Tuple[bool, List[Dict]]:
    """
    Tests whether an n x n matrix A is defective:
      Defective iff exists an eigenvalue where Geometric Multiplicity (GM) < Algebraic Multiplicity (AM).
    """
    n = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)
    # Cluster close eigenvalues
    unique_evals = []
    for ev in eigenvalues:
        if not any(np.isclose(ev, u, atol=tol) for u in unique_evals):
            unique_evals.append(ev)

    is_defective = False
    report = []

    for u_ev in unique_evals:
        # Count algebraic multiplicity
        am = int(np.sum([np.isclose(ev, u_ev, atol=tol) for ev in eigenvalues]))

        # Geometric multiplicity: dim(N(A - lambda I)) = n - rank(A - lambda I)
        if np.isreal(u_ev):
            M = A - np.real(u_ev) * np.eye(n)
        else:
            M = A.astype(complex) - u_ev * np.eye(n)

        # SVD rank of M
        _, S, _ = np.linalg.svd(M)
        rank_M = int(np.sum(S > tol))
        gm = n - rank_M

        if gm < am:
            is_defective = True

        report.append({
            "eigenvalue": u_ev,
            "AM": am,
            "GM": gm,
            "is_defective_mode": (gm < am)
        })

    return is_defective, report


# =====================================================================
# 3. RNN Recurrent Horizon Simulator (Vanishing, Exploding, Stable)
# =====================================================================
def simulate_rnn_dynamics(
    W: np.ndarray,
    h0: np.ndarray,
    timesteps: int = 30
) -> Tuple[float, List[float]]:
    """
    Propagates h_t = W h_{t-1} across timesteps.
    Returns:
      spectral_radius: max |lambda_i|
      norms: list of ||h_t||_2 across time
    """
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(W))))
    h = h0.astype(float).copy()
    norms = [float(np.linalg.norm(h))]

    for _ in range(timesteps):
        h = W @ h
        norms.append(float(np.linalg.norm(h)))

    return spectral_radius, norms


# =====================================================================
# 4. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.6: Eigendecomposition & Spectral Theory — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Part 5 AI by Hand Symmetric Matrix
    # W = [[1.2, 0.4], [0.4, 0.6]] -> evals: 1.4 and 0.4
    # -------------------------------------------------------------
    W_toy = np.array([
        [1.2, 0.4],
        [0.4, 0.6]
    ])
    decomp = decompose_symmetric_matrix(W_toy)

    print("\n[Test 1: Symmetric Eigendecomposition from Part 5]")
    print(f"  Matrix W:\n{W_toy}")
    print(f"  Eigenvalues: {decomp['eigenvalues']} (Expected: [1.4, 0.4])")
    print(f"  Orthonormal Eigenvectors Q:\n{decomp['Q']}")
    print(f"  Q^T @ Q = I Check:         {decomp['is_orthonormal']}")
    print(f"  Q @ Lambda @ Q^T = W Check: {decomp['is_reconstructed']}")
    print(f"  Tr(W) == sum(evals) Check:  {decomp['trace_matches']} ({np.trace(W_toy):.2f})")
    print(f"  det(W) == prod(evals) Check:{decomp['det_matches']} ({np.linalg.det(W_toy):.4f})")

    assert np.allclose(decomp['eigenvalues'], [1.4, 0.4]), "Eigenvalues mismatch!"
    assert decomp['is_orthonormal'] == True, "Q must be strictly orthonormal!"
    assert decomp['is_reconstructed'] == True, "Spectral reconstruction failed!"
    assert decomp['trace_matches'] == True, "Trace must equal sum of eigenvalues!"
    assert decomp['det_matches'] == True, "Determinant must equal product of eigenvalues!"

    # -------------------------------------------------------------
    # Test 2: Scenario A from notes
    # A = [[3, 1], [1, 3]] -> evals: 4 and 2
    # -------------------------------------------------------------
    A_scenA = np.array([[3.0, 1.0], [1.0, 3.0]])
    decomp_A = decompose_symmetric_matrix(A_scenA)
    print("\n[Test 2: Scenario A Matrix A = [[3, 1], [1, 3]]]")
    print(f"  Eigenvalues: {decomp_A['eigenvalues']} (Expected: [4.0, 2.0])")
    assert np.allclose(decomp_A['eigenvalues'], [4.0, 2.0]), "Scenario A eigenvalues mismatch!"

    # -------------------------------------------------------------
    # Test 3: Defective Matrix Detection (Scenario B)
    # Shear M = [[2, 1], [0, 2]] -> AM = 2, GM = 1 -> Defective
    # -------------------------------------------------------------
    M_defective = np.array([[2.0, 1.0], [0.0, 2.0]])
    is_def, report = test_defectiveness(M_defective)

    print("\n[Test 3: Defective Matrix Test (Shear Matrix)]")
    print(f"  Matrix M:\n{M_defective}")
    print(f"  Is Matrix Defective? {is_def} (Expected: True)")
    for rep in report:
        print(f"  eval={rep['eigenvalue']:.1f}, AM={rep['AM']}, GM={rep['GM']}, Defective Mode={rep['is_defective_mode']}")

    assert is_def == True, "Shear matrix must be detected as defective!"
    assert report[0]['AM'] == 2 and report[0]['GM'] == 1, "AM must be 2, GM must be 1!"

    # -------------------------------------------------------------
    # Test 4: Complex Eigenvalues from Pure 90-degree Rotation (Scenario C)
    # R = [[0, -1], [1, 0]] -> evals = +/- 1j
    # -------------------------------------------------------------
    R_rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    evals_R = np.linalg.eigvals(R_rot)
    print("\n[Test 4: Rotation Matrix Complex Spectrum]")
    print(f"  Rotation Matrix R:\n{R_rot}")
    print(f"  Eigenvalues: {evals_R} (Expected: +/- 1j)")
    assert np.allclose(np.abs(evals_R), 1.0), "Complex eigenvalues of rotation must have magnitude 1.0!"
    assert any(np.iscomplex(evals_R)), "Eigenvalues of 90-deg rotation must be imaginary!"

    # -------------------------------------------------------------
    # Test 5: RNN Stability Simulation (Exploding vs Vanishing vs Stable)
    # -------------------------------------------------------------
    print("\n[Test 5: RNN Recurrent State Dynamics Simulation]")
    h0 = np.array([2.0, 1.0])

    # Case 1: Exploding (rho = 1.4)
    rho_1, norms_1 = simulate_rnn_dynamics(W_toy, h0, timesteps=15)
    # Case 2: Vanishing (rho = 0.6)
    W_vanish = np.array([[0.5, 0.1], [0.1, 0.5]])  # evals: 0.6, 0.4
    rho_2, norms_2 = simulate_rnn_dynamics(W_vanish, h0, timesteps=15)
    # Case 3: Stable Orthogonal (rho = 1.0)
    theta = np.pi / 4
    W_stable = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rho_3, norms_3 = simulate_rnn_dynamics(W_stable, h0, timesteps=15)

    print(f"  Regime 1: rho={rho_1:.2f} > 1.0 -> Initial Norm: {norms_1[0]:.2f}, Final Norm: {norms_1[-1]:.2f} (EXPLODED!)")
    print(f"  Regime 2: rho={rho_2:.2f} < 1.0 -> Initial Norm: {norms_2[0]:.2f}, Final Norm: {norms_2[-1]:.6f} (VANISHED!)")
    print(f"  Regime 3: rho={rho_3:.2f} = 1.0 -> Initial Norm: {norms_3[0]:.2f}, Final Norm: {norms_3[-1]:.2f} (STABLE!)")

    assert norms_1[-1] > 100.0, "State must explode when rho > 1!"
    assert norms_2[-1] < 1e-3, "State must vanish when rho < 1!"
    assert np.isclose(norms_3[-1], norms_3[0]), "State norm must remain invariant when rho = 1!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
