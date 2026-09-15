"""
Chapter 2.6: Matrix Calculus (Numerator & Denominator Layouts, Trace Tricks)
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Frobenius Least-Squares Matrix Gradient Walkthrough.
2. Numerical Finite-Difference Matrix Gradient Checker.
3. Linear Layer Forward & Backward Pass (dW = X^T G, dX = G W^T, db = sum(G)).
4. Matrix Log-Determinant Derivative nabla_X log(det(X)) = X^{-T}.
5. Quadratic Trace Identities: nabla_X Tr(A X B) = A^T B^T and nabla_X Tr(X^T A X) = (A + A^T) X.
6. PyTorch Autograd Cross-Check.
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Least-Squares Objective & Analytical Matrix Gradient
# =====================================================================
def ls_loss(X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    """
    L(W) = 0.5 * ||X W - Y||_F^2
    """
    R = X @ W - Y
    return float(0.5 * np.sum(R**2))


def grad_ls_W(X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    nabla_W L = X^T (X W - Y)
    """
    R = X @ W - Y
    return X.T @ R


# =====================================================================
# 2. General Matrix Finite-Difference Gradient Estimator
# =====================================================================
def numerical_matrix_gradient(func, W: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes numerical gradient of a scalar function f(W) with respect to matrix W:
    (df/dW)_{ij} ~= (f(W + eps*E_ij) - f(W - eps*E_ij)) / (2*eps)
    """
    G = np.zeros_like(W, dtype=np.float64)
    rows, cols = W.shape
    for i in range(rows):
        for j in range(cols):
            E_ij = np.zeros_like(W)
            E_ij[i, j] = eps
            f_plus = func(W + E_ij)
            f_minus = func(W - E_ij)
            G[i, j] = (f_plus - f_minus) / (2.0 * eps)
    return G


# =====================================================================
# 3. Dense Layer Forward and Backward Implementation
# =====================================================================
class DenseLayer:
    """
    Forward:  Y = X @ W + b
    Backward: dL/dW = X^T @ G_Y
              dL/dX = G_Y @ W^T
              dL/db = sum(G_Y, axis=0)
    """

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W.copy()
        self.b = b.copy()
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X.copy()
        return X @ self.W + self.b

    def backward(self, G_Y: np.ndarray):
        dW = self.X.T @ G_Y
        dX = G_Y @ self.W.T
        db = np.sum(G_Y, axis=0, keepdims=True)
        return dX, dW, db


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 2.6: Matrix Calculus & Trace Tricks — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Numerical Verification
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Least-Squares Matrix Gradient]")
    X0 = np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float64)
    W0 = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64)
    Y0 = np.array([[3.0, 8.0], [6.0, 7.0]], dtype=np.float64)

    Y_hat = X0 @ W0
    R0 = Y_hat - Y0
    loss0 = ls_loss(X0, W0, Y0)
    grad0 = grad_ls_W(X0, W0, Y0)

    print(f"  Forward Prediction Y_hat:\n{Y_hat}")
    print(f"  Residual Matrix R:\n{R0}")
    print(f"  Scalar Loss L0:            {loss0:.4f} (Expected: 1.5000)")
    print(f"  Matrix Gradient nabla_W L:\n{grad0}")

    assert np.allclose(Y_hat, [[2.0, 7.0], [6.0, 6.0]]), "Y_hat mismatch!"
    assert np.allclose(R0, [[-1.0, -1.0], [0.0, -1.0]]), "Residual R mismatch!"
    assert np.isclose(loss0, 1.50), f"Loss mismatch: {loss0}"
    assert np.allclose(grad0, [[-1.0, -4.0], [-2.0, -3.0]]), "Gradient mismatch!"
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: Numerical Matrix Gradient vs Analytical X^T R
    # -------------------------------------------------------------
    print("\n[Test 2: Finite Difference Matrix Gradient Check]")
    def loss_closure(W):
        return ls_loss(X0, W, Y0)

    grad_num = numerical_matrix_gradient(loss_closure, W0)
    print(f"  Analytical Gradient:\n{grad0}")
    print(f"  Numerical Gradient:\n{grad_num}")
    assert np.allclose(grad_num, grad0, atol=1e-5), "Matrix finite-difference mismatch!"
    print("  -> Matrix finite difference matches analytical X^T R!")

    # -------------------------------------------------------------
    # Test 3: Dense Layer Backward Pass Shape-Matching
    # -------------------------------------------------------------
    print("\n[Test 3: Dense Layer Forward & Backward Shape-Matching]")
    B, d_in, d_out = 4, 3, 5
    np.random.seed(42)
    X_batch = np.random.randn(B, d_in)
    W_dense = np.random.randn(d_in, d_out)
    b_dense = np.random.randn(1, d_out)
    G_Y = np.random.randn(B, d_out)

    layer = DenseLayer(W_dense, b_dense)
    Y_out = layer.forward(X_batch)
    dX, dW, db = layer.backward(G_Y)

    print(f"  X shape: {X_batch.shape} <===> dX shape: {dX.shape}")
    print(f"  W shape: {W_dense.shape} <===> dW shape: {dW.shape}")
    print(f"  b shape: {b_dense.shape} <===> db shape: {db.shape}")

    assert dX.shape == X_batch.shape, "dX shape mismatch!"
    assert dW.shape == W_dense.shape, "dW shape mismatch!"
    assert db.shape == b_dense.shape, "db shape mismatch!"
    print("  -> Shape-Matching Principle strictly satisfied for all layer parameters!")

    # -------------------------------------------------------------
    # Test 4: Matrix Log-Determinant Derivative nabla_X log(det(X)) = X^{-T}
    # -------------------------------------------------------------
    print("\n[Test 4: Log-Determinant Gradient nabla_X log(det(X)) = X^{-T}]")
    # Positive definite test matrix
    A_rand = np.random.randn(3, 3)
    X_spd = A_rand.T @ A_rand + np.eye(3)

    # Analytical: X^-T
    grad_logdet_ana = np.linalg.inv(X_spd).T

    # Numerical
    def logdet_func(M):
        sign, logdet = np.linalg.slogdet(M)
        return float(logdet)

    grad_logdet_num = numerical_matrix_gradient(logdet_func, X_spd)
    print(f"  Analytical X^{{-T}}:\n{grad_logdet_ana}")
    print(f"  Numerical Finite Difference:\n{grad_logdet_num}")
    assert np.allclose(grad_logdet_num, grad_logdet_ana, atol=1e-4), "Log-det gradient mismatch!"
    print("  -> Log-determinant derivative nabla_X log(det(X)) = X^{-T} verified!")

    # -------------------------------------------------------------
    # Test 5: Trace Product Identities: Tr(A X B) and Tr(X^T A X)
    # -------------------------------------------------------------
    print("\n[Test 5: Master Trace Identities Verification]")
    A_mat = np.random.randn(3, 3)
    B_mat = np.random.randn(3, 3)
    X_mat = np.random.randn(3, 3)

    # Identity 1: f(X) = Tr(A X B) => grad_X = A^T B^T
    def f_tr1(X):
        return float(np.trace(A_mat @ X @ B_mat))

    grad1_ana = A_mat.T @ B_mat.T
    grad1_num = numerical_matrix_gradient(f_tr1, X_mat)
    assert np.allclose(grad1_num, grad1_ana, atol=1e-5), "Identity 1 mismatch!"

    # Identity 2: f(X) = Tr(X^T A X) => grad_X = (A + A^T) X
    def f_tr2(X):
        return float(np.trace(X.T @ A_mat @ X))

    grad2_ana = (A_mat + A_mat.T) @ X_mat
    grad2_num = numerical_matrix_gradient(f_tr2, X_mat)
    assert np.allclose(grad2_num, grad2_ana, atol=1e-5), "Identity 2 mismatch!"
    print("  -> Trace identities nabla Tr(A X B) and nabla Tr(X^T A X) verified!")

    # -------------------------------------------------------------
    # Test 6: PyTorch Autograd Cross-Check on Least Squares
    # -------------------------------------------------------------
    print("\n[Test 6: PyTorch Autograd Least Squares Cross-Check]")
    X_torch = torch.tensor(X0, dtype=torch.float64)
    W_torch = torch.tensor(W0, dtype=torch.float64, requires_grad=True)
    Y_torch = torch.tensor(Y0, dtype=torch.float64)

    L_torch = 0.5 * torch.sum((X_torch @ W_torch - Y_torch) ** 2)
    L_torch.backward()

    torch_dW = W_torch.grad.numpy()
    print(f"  PyTorch W.grad:\n{torch_dW}")
    assert np.allclose(torch_dW, grad0), "PyTorch gradient mismatch!"
    print("  -> PyTorch Autograd matches hand-derived matrix calculus gradient!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 2.6 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
