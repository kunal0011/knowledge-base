"""
Chapter 1.3: Matrix Algebra, Column Spaces & Systems of Linear Equations
========================================================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. Four-Way Matrix Multiplication Engine (Dot, Column, Row, Outer Product)
2. Extraction of Gilbert Strang's Four Fundamental Subspaces (C(A), N(A), C(A^T), N(A^T))
3. Orthogonality and Rank-Nullity Verification
4. Complete Linear System Solver (x = x_p + x_n)
5. Deep Learning Nullspace Invariance Demonstration in PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict


# =====================================================================
# 1. Four-Way Matrix Multiplication Engine
# =====================================================================
def multiply_four_ways(A: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes C = A @ B using all 4 fundamental perspectives:
      1. Dot Product View (row dot col)
      2. Column View (linear combinations of cols of A)
      3. Row View (linear combinations of rows of B)
      4. Outer Product View (sum of rank-1 matrices)
    """
    m, k1 = A.shape
    k2, p = B.shape
    assert k1 == k2, f"Dimension mismatch: A is ({m}, {k1}), B is ({k2}, {p})"
    k = k1

    # 1. Dot Product View
    C_dot = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            C_dot[i, j] = np.dot(A[i, :], B[:, j])

    # 2. Column View
    C_col = np.zeros((m, p))
    for j in range(p):
        col_combo = np.zeros(m)
        for r in range(k):
            col_combo += B[r, j] * A[:, r]
        C_col[:, j] = col_combo

    # 3. Row View
    C_row = np.zeros((m, p))
    for i in range(m):
        row_combo = np.zeros(p)
        for r in range(k):
            row_combo += A[i, r] * B[r, :]
        C_row[i, :] = row_combo

    # 4. Outer Product View
    C_outer = np.zeros((m, p))
    for r in range(k):
        col_r = A[:, r:r+1]   # [m, 1]
        row_r = B[r:r+1, :]   # [1, p]
        C_outer += col_r @ row_r

    return {
        "dot_product": C_dot,
        "column_view": C_col,
        "row_view": C_row,
        "outer_product": C_outer
    }


# =====================================================================
# 2. Helper: Reduced Row Echelon Form (RREF)
# =====================================================================
def compute_rref(matrix: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, List[int]]:
    """
    Computes RREF and returns pivot column indices.
    """
    M = matrix.astype(float).copy()
    rows, cols = M.shape
    pivot_cols = []
    current_row = 0

    for c in range(cols):
        if current_row >= rows:
            break
        max_r = current_row + np.argmax(np.abs(M[current_row:, c]))
        if np.abs(M[max_r, c]) < tol:
            continue

        M[[current_row, max_r]] = M[[max_r, current_row]]
        M[current_row] /= M[current_row, c]

        for r in range(rows):
            if r != current_row and np.abs(M[r, c]) > tol:
                M[r] -= M[r, c] * M[current_row]

        pivot_cols.append(c)
        current_row += 1

    M[np.abs(M) < tol] = 0.0
    return M, pivot_cols


# =====================================================================
# 3. Four Fundamental Subspaces Extractor
# =====================================================================
def find_four_fundamental_subspaces(A: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes bases for:
      - C(A): Column Space of A
      - N(A): Nullspace of A
      - C(A^T): Row Space of A
      - N(A^T): Left Nullspace of A
    """
    m, n = A.shape
    rref_A, pivot_cols = compute_rref(A)
    rank = len(pivot_cols)

    # 1. Column Space C(A): Original columns corresponding to pivot columns
    basis_C_A = A[:, pivot_cols]

    # 2. Row Space C(A^T): Non-zero rows of RREF(A) transposed into column vectors
    basis_C_AT = rref_A[:rank, :].T

    # 3. Nullspace N(A): Special solutions from free variables
    free_cols = [c for c in range(n) if c not in pivot_cols]
    nullspace_vectors = []
    for free_col in free_cols:
        x_n = np.zeros(n)
        x_n[free_col] = 1.0
        for r, p_col in enumerate(pivot_cols):
            x_n[p_col] = -rref_A[r, free_col]
        nullspace_vectors.append(x_n)

    basis_N_A = np.column_stack(nullspace_vectors) if nullspace_vectors else np.zeros((n, 0))

    # 4. Left Nullspace N(A^T): Nullspace of A^T
    rref_AT, pivot_cols_AT = compute_rref(A.T)
    rank_AT = len(pivot_cols_AT)
    free_cols_AT = [c for c in range(m) if c not in pivot_cols_AT]
    left_nullspace_vectors = []
    for free_col in free_cols_AT:
        y_n = np.zeros(m)
        y_n[free_col] = 1.0
        for r, p_col in enumerate(pivot_cols_AT):
            y_n[p_col] = -rref_AT[r, free_col]
        left_nullspace_vectors.append(y_n)

    basis_N_AT = np.column_stack(left_nullspace_vectors) if left_nullspace_vectors else np.zeros((m, 0))

    return {
        "rank": rank,
        "C_A": basis_C_A,
        "N_A": basis_N_A,
        "C_AT": basis_C_AT,
        "N_AT": basis_N_AT
    }


# =====================================================================
# 4. Linear System Solver (x = x_p + x_n)
# =====================================================================
def solve_linear_system(A: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Solves A x = b.
    Returns:
      is_consistent: bool
      x_particular: particular solution vector
      N_A: nullspace basis matrix (shape [n, n-r])
    """
    m, n = A.shape
    # Augmented matrix
    augmented = np.column_stack([A, b])
    rref_aug, pivots = compute_rref(augmented)

    # Check if last column has a pivot (inconsistent system)
    if (n in pivots):
        return False, np.zeros(n), np.zeros((n, 0))

    # Particular solution: set free variables to 0
    x_p = np.zeros(n)
    for r, p_col in enumerate(pivots):
        if p_col < n:
            x_p[p_col] = rref_aug[r, -1]

    # Subspaces
    subspaces = find_four_fundamental_subspaces(A)
    return True, x_p, subspaces["N_A"]


# =====================================================================
# 5. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.3: Matrix Algebra & Systems — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Four-Way Matrix Multiplication (Part 5 Illustration)
    # X = [[1, 2], [3, 0]], W = [[2, 1, 4], [0, 3, -1]]
    # -------------------------------------------------------------
    X = np.array([[1.0, 2.0], [3.0, 0.0]])
    W = np.array([[2.0, 1.0, 4.0], [0.0, 3.0, -1.0]])
    results = multiply_four_ways(X, W)

    expected = np.array([[2.0, 7.0, 2.0], [6.0, 3.0, 12.0]])
    print("\n[Test 1: Four Perspectives of Matrix Multiplication]")
    print(f"  Expected Output Matrix Y:\n{expected}")
    for name, res in results.items():
        print(f"  Checking {name:<14}: Match? {np.allclose(res, expected)}")
        assert np.allclose(res, expected), f"Multiplication mismatch in {name}!"

    # -------------------------------------------------------------
    # Test 2: Four Fundamental Subspaces (Scenario A from notes)
    # A = [[1, 2, 0, 1], [0, 1, 1, 0], [1, 3, 1, 1]]
    # -------------------------------------------------------------
    A = np.array([
        [1.0, 2.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 3.0, 1.0, 1.0]
    ])
    subspaces = find_four_fundamental_subspaces(A)
    m, n = A.shape
    r = subspaces["rank"]

    print("\n[Test 2: Gilbert Strang's Four Fundamental Subspaces]")
    print(f"  Matrix A dimensions: m={m}, n={n}")
    print(f"  Computed Rank r:     {r}")
    print(f"  dim(C(A))   = {subspaces['C_A'].shape[1]} (Expected: 2)")
    print(f"  dim(C(A^T)) = {subspaces['C_AT'].shape[1]} (Expected: 2)")
    print(f"  dim(N(A))   = {subspaces['N_A'].shape[1]} (Expected: 2)")
    print(f"  dim(N(A^T)) = {subspaces['N_AT'].shape[1]} (Expected: 1)")

    # Assert Rank-Nullity
    assert r + subspaces["N_A"].shape[1] == n, "Rank-Nullity theorem failed for N(A)!"
    assert r + subspaces["N_AT"].shape[1] == m, "Rank-Nullity theorem failed for N(A^T)!"

    # Assert Orthogonality: C(A^T) ⊥ N(A)
    ortho_check = subspaces["C_AT"].T @ subspaces["N_A"]
    print(f"  Orthogonality check C(A^T)^T @ N(A) (should be 0):\n{np.round(ortho_check, 4)}")
    assert np.allclose(ortho_check, 0.0), "Row Space and Nullspace must be strictly orthogonal!"

    # -------------------------------------------------------------
    # Test 3: Linear System Solver A x = b (Scenario B)
    # b = [3, 1, 4]^T (Consistent)
    # -------------------------------------------------------------
    b_consistent = np.array([3.0, 1.0, 4.0])
    is_solvable, x_p, N_A = solve_linear_system(A, b_consistent)

    print("\n[Test 3: Linear System Solvability (A x = b)]")
    print(f"  Is system solvable? {is_solvable}")
    print(f"  Particular solution x_p: {x_p}")
    assert is_solvable == True, "System must be consistent!"
    assert np.allclose(A @ x_p, b_consistent), "Particular solution does not satisfy A x_p = b!"

    # Test random homogeneous combinations: A (x_p + c * x_n) == b
    for _ in range(5):
        coeffs = np.random.randn(N_A.shape[1])
        x_gen = x_p + N_A @ coeffs
        residual = np.linalg.norm(A @ x_gen - b_consistent)
        assert np.isclose(residual, 0.0, atol=1e-5), f"General solution failed residual check: {residual}"
    print("  Random nullspace shifts (x_p + N(A)*c) all verified: A x = b strictly preserved!")

    # -------------------------------------------------------------
    # Test 4: Deep Learning Nullspace Invariance in PyTorch
    # -------------------------------------------------------------
    print("\n[Test 4: PyTorch Neural Network Nullspace Invariance]")
    # Linear layer: W is [2, 4]
    W_torch = torch.tensor(A, dtype=torch.float32)  # [3, 4]
    x_input = torch.tensor([1.5, -2.0, 0.5, 3.0], dtype=torch.float32)

    # Clean forward output
    y_clean = torch.matmul(W_torch, x_input)

    # Pick a non-zero vector from the nullspace N(A)
    null_perturbation = torch.tensor(subspaces["N_A"][:, 0] * 5.0, dtype=torch.float32)
    x_perturbed = x_input + null_perturbation

    y_perturbed = torch.matmul(W_torch, x_perturbed)
    diff = torch.norm(y_perturbed - y_clean).item()

    print(f"  Clean Output:      {y_clean.numpy()}")
    print(f"  Perturbed Output:  {y_perturbed.numpy()}")
    print(f"  Difference norm:   {diff:.6e}")
    assert np.isclose(diff, 0.0, atol=1e-5), "Perturbation along N(W) must yield EXACT zero output difference!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
