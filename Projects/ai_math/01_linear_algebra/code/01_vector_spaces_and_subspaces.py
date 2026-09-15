"""
Chapter 1.1: Vector Spaces, Subspaces & Basis
=============================================
NumPy and PyTorch Reference Implementations and Mathematical Verifications.

Topics covered:
1. Subspace Closure Verification (Closure under linear combinations & zero vector)
2. Reduced Row Echelon Form (RREF) from scratch with partial pivoting
3. Linear Independence, Rank, and Basis Extraction
4. Coordinate Solver in Arbitrary Bases
5. LoRA (Low-Rank Adaptation) Subspace Constraint Demonstration
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional


# =====================================================================
# 1. Subspace Closure Checker
# =====================================================================
def verify_subspace_closure(
    membership_fn,
    sample_generator_fn,
    dim: int,
    num_tests: int = 500,
    tol: float = 1e-6
) -> Tuple[bool, str]:
    """
    Verifies whether a given subset defined by `membership_fn`
    satisfies the two-step subspace test:
      1. Contains the zero vector: 0 in W.
      2. Closed under linear combinations: a*u + b*v in W for any u, v in W.
    """
    # 1. Zero vector test
    zero_vec = np.zeros(dim)
    if not membership_fn(zero_vec):
        return False, "Failed zero vector test: 0 not in W."

    # 2. Linear combination closure test using generator
    for _ in range(num_tests):
        u = sample_generator_fn()
        v = sample_generator_fn()
        if not membership_fn(u) or not membership_fn(v):
            continue
        a, b = np.random.randn(2) * 5.0
        combo = a * u + b * v
        if not membership_fn(combo):
            return False, f"Failed closure: {a:.2f}*u + {b:.2f}*v does not satisfy subspace condition."

    return True, f"Subspace axioms confirmed across {num_tests} random linear combinations."


# =====================================================================
# 2. Reduced Row Echelon Form (RREF) from Scratch
# =====================================================================
def compute_rref(A: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, List[int]]:
    """
    Computes the Reduced Row Echelon Form (RREF) of matrix A using
    Gauss-Jordan elimination with partial pivoting.
    
    Returns:
      rref_A: Matrix in RREF
      pivot_cols: List of 0-indexed column indices containing leading 1s
    """
    M = A.astype(float).copy()
    rows, cols = M.shape
    pivot_cols = []
    current_row = 0

    for c in range(cols):
        if current_row >= rows:
            break

        # Partial pivoting: find max absolute value in column c from current_row down
        max_r = current_row + np.argmax(np.abs(M[current_row:, c]))
        if np.abs(M[max_r, c]) < tol:
            # Column is practically zero in remaining rows; skip to next column
            continue

        # Swap current_row with max_r
        M[[current_row, max_r]] = M[[max_r, current_row]]

        # Scale pivot row so leading entry is 1.0
        M[current_row] = M[current_row] / M[current_row, c]

        # Eliminate column c in all other rows
        for r in range(rows):
            if r != current_row and np.abs(M[r, c]) > tol:
                M[r] = M[r] - M[r, c] * M[current_row]

        pivot_cols.append(c)
        current_row += 1

    # Clean up floating point noise near zero
    M[np.abs(M) < tol] = 0.0
    return M, pivot_cols


# =====================================================================
# 3. Linear Independence & Basis Extraction
# =====================================================================
def check_linear_independence(vectors: np.ndarray) -> Tuple[bool, int, List[int], np.ndarray]:
    """
    Given a matrix where vectors are COLUMNS (shape: [d, k]):
    Returns:
      - is_independent: bool (True if all columns are linearly independent)
      - rank: int (dimension of the column span)
      - pivot_indices: column indices that form a maximal linearly independent basis
      - rref_matrix: the RREF matrix
    """
    d, k = vectors.shape
    rref_matrix, pivot_cols = compute_rref(vectors)
    rank = len(pivot_cols)
    is_independent = (rank == k)
    return is_independent, rank, pivot_cols, rref_matrix


# =====================================================================
# 4. Coordinate Solver Relative to a Basis
# =====================================================================
def get_coordinates_in_basis(basis: np.ndarray, target: np.ndarray, tol: float = 1e-6) -> Optional[np.ndarray]:
    """
    Given basis vectors as COLUMNS of matrix B (shape [d, k]) and target vector (shape [d]),
    finds unique coordinates c such that B @ c = target.
    Returns None if target is outside span(B).
    """
    c, residuals, rank, s = np.linalg.lstsq(basis, target, rcond=None)
    reconstructed = basis @ c
    err = np.linalg.norm(reconstructed - target)
    if err > tol:
        return None
    return c


# =====================================================================
# 5. LoRA Subspace Demonstration in PyTorch
# =====================================================================
class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation (LoRA).
    y = W_0 x + (alpha / r) * (B @ A @ x)
    where:
      W_0: base weight matrix [d_out, d_in]
      A: down-projection matrix [r, d_in]
      B: up-projection matrix   [d_out, r]
    """
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # Base frozen weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # Trainable low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.1)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_out = torch.matmul(x, self.weight.t())
        lora_delta = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_delta, lora_delta


# =====================================================================
# 6. Verification Test Harness
# =====================================================================
def run_all_tests():
    print("================================================================")
    print("Chapter 1.1: Vector Spaces, Subspaces & Basis — Test Suite")
    print("================================================================")

    # -------------------------------------------------------------
    # Test 1: Subspace Verification
    # W1: 3x1 - 2x2 + x3 = 0  (Subspace: passes through origin)
    # W2: 3x1 - 2x2 + x3 = 4  (Affine plane: does NOT contain 0)
    # -------------------------------------------------------------
    plane1_membership = lambda x: np.isclose(3 * x[0] - 2 * x[1] + x[2], 0.0, atol=1e-5)
    # Parametric generator for plane 1: x3 = 2*x2 - 3*x1
    plane1_generator = lambda: (lambda x1, x2: np.array([x1, x2, 2*x2 - 3*x1]))(*np.random.randn(2)*3)

    plane2_membership = lambda x: np.isclose(3 * x[0] - 2 * x[1] + x[2], 4.0, atol=1e-5)
    # Parametric generator for plane 2: x3 = 4 + 2*x2 - 3*x1
    plane2_generator = lambda: (lambda x1, x2: np.array([x1, x2, 4.0 + 2*x2 - 3*x1]))(*np.random.randn(2)*3)

    is_sub1, msg1 = verify_subspace_closure(plane1_membership, plane1_generator, dim=3)
    is_sub2, msg2 = verify_subspace_closure(plane2_membership, plane2_generator, dim=3)

    print("\n[Test 1: Subspace Verification]")
    print(f"  Plane W1 (3x1 - 2x2 + x3 = 0): Is Subspace? {is_sub1} -> {msg1}")
    print(f"  Plane W2 (3x1 - 2x2 + x3 = 4): Is Subspace? {is_sub2} -> {msg2}")
    assert is_sub1 == True, "W1 must be a valid subspace!"
    assert is_sub2 == False, "W2 cannot be a subspace because 0 is not in W2!"

    # -------------------------------------------------------------
    # Test 2: Linear Independence & RREF Analysis
    # Matrix columns: v1 = [1, 0, 2]^T, v2 = [2, 1, 0]^T, v3 = [1, 1, -2]^T
    # -------------------------------------------------------------
    v1 = np.array([1.0, 0.0, 2.0])
    v2 = np.array([2.0, 1.0, 0.0])
    v3 = np.array([1.0, 1.0, -2.0])
    V = np.column_stack([v1, v2, v3])

    is_indep, rank, pivots, rref_V = check_linear_independence(V)
    print("\n[Test 2: Linear Independence & RREF of {v1, v2, v3}]")
    print(f"  Original Matrix V:\n{V}")
    print(f"  RREF of Matrix V:\n{rref_V}")
    print(f"  Linearly Independent: {is_indep}")
    print(f"  Subspace Rank (Dimension): {rank}")
    print(f"  Pivot Column Indices: {pivots} (Columns 0 and 1)")
    assert is_indep == False, "Set must be linearly dependent!"
    assert rank == 2, "Rank must be exactly 2!"
    assert pivots == [0, 1], "Pivots must be columns 0 and 1!"

    # Extract basis
    basis_W = V[:, pivots]
    print(f"  Extracted Basis for Span(V):\n{basis_W}")

    # -------------------------------------------------------------
    # Test 3: Coordinate representation & Membership Check
    # v3 should have unique coordinates [-1, 1] relative to basis [v1, v2]
    # b = [5, 3, 0]^T is outside Span(V) -> should return None
    # -------------------------------------------------------------
    print("\n[Test 3: Coordinate Representation & Span Membership]")
    coords_v3 = get_coordinates_in_basis(basis_W, v3)
    print(f"  Coordinates of v3 in basis [v1, v2]: {coords_v3}")
    assert coords_v3 is not None, "v3 must lie in span of basis!"
    assert np.allclose(coords_v3, [-1.0, 1.0]), "v3 = -1*v1 + 1*v2!"
    assert np.allclose(basis_W @ coords_v3, v3), "Reconstruction check failed!"

    b_outside = np.array([5.0, 3.0, 0.0])
    coords_b = get_coordinates_in_basis(basis_W, b_outside)
    print(f"  Coordinates of b=[5, 3, 0]^T in basis: {coords_b} (Expected: None)")
    assert coords_b is None, "b=[5, 3, 0]^T must lie outside span(basis_W)!"

    # -------------------------------------------------------------
    # Test 4: LoRA Subspace Dimension Constraint
    # Batch size N=200 inputs in R^512 passing through LoRA rank r=4
    # The output delta_W x lives in R^1024, but MUST have rank exactly 4!
    # -------------------------------------------------------------
    print("\n[Test 4: LoRA Subspace Rank Constraint in PyTorch]")
    d_in, d_out, r = 512, 1024, 4
    torch.manual_seed(42)
    lora = LoRALinear(in_features=d_in, out_features=d_out, r=r)

    # 200 diverse inputs
    X = torch.randn(200, d_in)
    total_out, lora_delta = lora(X)

    # Compute singular values of LoRA delta outputs
    delta_np = lora_delta.detach().numpy()
    _, S_delta, _ = np.linalg.svd(delta_np)
    effective_rank = int(np.sum(S_delta > 1e-4))

    print(f"  LoRA Ambient Output Dimension: {d_out}")
    print(f"  LoRA Configured Subspace Rank r: {r}")
    print(f"  Singular values (top 7): {np.round(S_delta[:7], 4)}")
    print(f"  Effective Output Subspace Rank: {effective_rank}")
    assert effective_rank == r, f"Output rank must be strictly bounded by r={r}!"

    print("\n>>> ALL MATHEMATICAL & COMPUTATIONAL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_all_tests()
