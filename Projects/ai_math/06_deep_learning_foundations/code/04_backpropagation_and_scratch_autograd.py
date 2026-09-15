"""
Chapter 6.4: Full Backpropagation Derivation & Scratch Autograd Engine
Verification Suite and Experimental Tests.

Includes:
1. Part 5 Visual Grid numerical verification (hand calculation match).
2. Minimal production-grade Scratch Autograd Tensor Engine with reverse-mode DAG.
3. Numerical gradient checking (finite difference verification).
4. Strict parity test against PyTorch C++ autograd backward pass.
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
    Validates hand calculation grid for 2-layer network:
    Input: x = [1.0, 2.0]^T, target y = 1.0
    W1 = [[0.5, -0.5], [0.2, 0.8]], b1 = [0.1, -0.2]
    W2 = [[0.4, 0.6]], b2 = [0.1]
    Activation: ReLU on hidden layer, Linear on output layer.
    Loss: 0.5 * (y_hat - y)^2
    """
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = 1.0

    W1 = np.array([[0.5, -0.5], [0.2, 0.8]], dtype=np.float64)
    b1 = np.array([0.1, -0.2], dtype=np.float64)

    W2 = np.array([[0.4, 0.6]], dtype=np.float64)
    b2 = np.array([0.1], dtype=np.float64)

    # Forward Pass
    z1 = W1 @ x + b1
    a1 = np.maximum(0.0, z1)
    z2 = W2 @ a1 + b2
    y_hat = z2[0]
    loss = 0.5 * ((y_hat - y) ** 2)

    assert np.allclose(z1, [-0.4, 1.6], atol=1e-7)
    assert np.allclose(a1, [0.0, 1.6], atol=1e-7)
    assert abs(z2[0] - 1.06) < 1e-7
    assert abs(loss - 0.0018) < 1e-7

    # Backward Pass
    delta2 = np.array([y_hat - y], dtype=np.float64)  # [0.06]
    grad_W2 = np.outer(delta2, a1)
    grad_b2 = delta2.copy()

    delta1 = (W2.T @ delta2) * (z1 > 0.0).astype(np.float64)
    grad_W1 = np.outer(delta1, x)
    grad_b1 = delta1.copy()

    assert np.allclose(delta2, [0.06], atol=1e-7)
    assert np.allclose(grad_W2, [[0.0, 0.096]], atol=1e-7)
    assert np.allclose(grad_b2, [0.06], atol=1e-7)

    assert np.allclose(delta1, [0.0, 0.036], atol=1e-7)
    assert np.allclose(grad_W1, [[0.0, 0.0], [0.036, 0.072]], atol=1e-7)
    assert np.allclose(grad_b1, [0.0, 0.036], atol=1e-7)

    print("--- Part 5 Visual Grid Verification ---")
    print(f"Forward Pass:  z_1 = {z1.tolist()} | a_1 = {a1.tolist()} | y_hat = {y_hat:.4f} | Loss = {loss:.6f}")
    print(f"Backward Pass: delta_2 = {delta2.tolist()} | grad_W2 = {grad_W2.tolist()}")
    print(f"               delta_1 = {delta1.tolist()} | grad_W1 = {grad_W1.tolist()}")
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Minimal Production-Grade Scratch Autograd Engine
# =====================================================================
class Tensor:
    """
    A lightweight reverse-mode automatic differentiation Tensor
    with computational graph DAG tracking and topological sort backpropagation.
    """
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    @staticmethod
    def _unbroadcast(grad, shape):
        g = grad
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                g = g.sum(axis=i, keepdims=True)
        if g.shape != shape:
            g = g.reshape(shape)
        return g

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (other * -1.0)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0.0).astype(np.float64) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Build topological sort of the computational graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed gradient of loss sink with 1.0
        self.grad = np.ones_like(self.data)

        # Traverse graph in reverse topological order
        for node in reversed(topo):
            node._backward()


def verify_scratch_autograd_engine():
    """
    Executes the Part 5 model through our scratch Autograd Engine
    and verifies all gradients match hand calculations.
    """
    x = Tensor(np.array([[1.0, 2.0]]))
    y = Tensor(np.array([[1.0]]))

    W1 = Tensor(np.array([[0.5, 0.2], [-0.5, 0.8]]))  # Transposed for x @ W1
    b1 = Tensor(np.array([[0.1, -0.2]]))

    W2 = Tensor(np.array([[0.4], [0.6]]))
    b2 = Tensor(np.array([[0.1]]))

    # Forward
    z1 = (x @ W1) + b1
    a1 = z1.relu()
    z2 = (a1 @ W2) + b2
    diff = z2 - y
    loss = (diff * diff) * 0.5

    assert abs(loss.data[0, 0] - 0.0018) < 1e-7

    # Backward
    loss.backward()

    # W1.grad in scratch (matches Part 5 W1.T)
    expected_W1_grad = np.array([[0.0, 0.036], [0.0, 0.072]])
    expected_b1_grad = np.array([[0.0, 0.036]])
    expected_W2_grad = np.array([[0.0], [0.096]])
    expected_b2_grad = np.array([[0.06]])

    assert np.allclose(W1.grad, expected_W1_grad, atol=1e-7)
    assert np.allclose(b1.grad, expected_b1_grad, atol=1e-7)
    assert np.allclose(W2.grad, expected_W2_grad, atol=1e-7)
    assert np.allclose(b2.grad, expected_b2_grad, atol=1e-7)

    print("\n--- Scratch Autograd Engine Verification ---")
    print(f"Scratch W1 Gradient:\n{W1.grad}")
    print(f"Scratch W2 Gradient:\n{W2.grad}")
    print("✓ Scratch Autograd reverse-mode graph executed with exact mathematical match!")


# =====================================================================
# 3. Numerical Gradient Checking (Finite Differences)
# =====================================================================
def verify_numerical_gradient_checking():
    """
    Performs two-sided finite difference gradient check:
    grad_num = (f(theta + eps) - f(theta - eps)) / (2 * eps)
    verifying analytical gradients.
    """
    def forward_loss(W1, b1, W2, b2, x, y):
        z1 = W1 @ x + b1
        a1 = np.maximum(0.0, z1)
        z2 = W2 @ a1 + b2
        return 0.5 * ((z2[0] - y) ** 2)

    x = np.array([1.0, 2.0], dtype=np.float64)
    y = 1.0
    W1 = np.array([[0.5, -0.5], [0.2, 0.8]], dtype=np.float64)
    b1 = np.array([0.1, -0.2], dtype=np.float64)
    W2 = np.array([[0.4, 0.6]], dtype=np.float64)
    b2 = np.array([0.1], dtype=np.float64)

    eps = 1e-5
    analytical_grad_W1_22 = 0.0720

    W1_plus = W1.copy()
    W1_plus[1, 1] += eps
    loss_plus = forward_loss(W1_plus, b1, W2, b2, x, y)

    W1_minus = W1.copy()
    W1_minus[1, 1] -= eps
    loss_minus = forward_loss(W1_minus, b1, W2, b2, x, y)

    num_grad = (loss_plus - loss_minus) / (2.0 * eps)
    rel_error = abs(num_grad - analytical_grad_W1_22) / abs(analytical_grad_W1_22)

    print("\n--- Numerical Gradient Checking ---")
    print(f"Analytical Gradient (dL/dW1_22): {analytical_grad_W1_22:.6f}")
    print(f"Numerical Gradient  (Finite Diff):{num_grad:.6f}")
    print(f"Relative Error:                   {rel_error:.2e}")

    assert rel_error < 1e-6
    print("✓ Gradient check passed to < 1e-6 relative tolerance!")


# =====================================================================
# 4. PyTorch C++ Autograd Parity Verification
# =====================================================================
def verify_pytorch_parity():
    """
    Constructs an identical 2-layer network in PyTorch and verifies that
    PyTorch autograd matches scratch backprop to machine precision (< 1e-12).
    """
    x_torch = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    y_torch = torch.tensor([[1.0]], dtype=torch.float64)

    fc1 = nn.Linear(2, 2, bias=True, dtype=torch.float64)
    fc2 = nn.Linear(2, 1, bias=True, dtype=torch.float64)

    # Set exact weights
    with torch.no_grad():
        fc1.weight.copy_(torch.tensor([[0.5, -0.5], [0.2, 0.8]], dtype=torch.float64))
        fc1.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
        fc2.weight.copy_(torch.tensor([[0.4, 0.6]], dtype=torch.float64))
        fc2.bias.copy_(torch.tensor([0.1], dtype=torch.float64))

    # Forward
    h1 = F.relu(fc1(x_torch))
    y_pred = fc2(h1)
    loss = 0.5 * torch.sum((y_pred - y_torch) ** 2)

    # Backward
    loss.backward()

    # Compare with analytical hand calculations
    diff_w1 = np.max(np.abs(fc1.weight.grad.numpy() - np.array([[0.0, 0.0], [0.036, 0.072]])))
    diff_b1 = np.max(np.abs(fc1.bias.grad.numpy() - np.array([0.0, 0.036])))
    diff_w2 = np.max(np.abs(fc2.weight.grad.numpy() - np.array([[0.0, 0.096]])))
    diff_b2 = np.max(np.abs(fc2.bias.grad.numpy() - np.array([0.06])))

    print("\n--- PyTorch Autograd Parity ---")
    print(f"PyTorch fc1.weight.grad:\n{fc1.weight.grad.numpy()}")
    print(f"PyTorch fc2.weight.grad:\n{fc2.weight.grad.numpy()}")
    print(f"Max Absolute Discrepancy: {max(diff_w1, diff_b1, diff_w2, diff_b2):.2e}")

    assert max(diff_w1, diff_b1, diff_w2, diff_b2) < 1e-12
    print("✓ PyTorch C++ autograd backward pass verified to float64 machine precision (< 1e-12)!")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING CHAPTER 6.4 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    verify_visual_grid()
    verify_scratch_autograd_engine()
    verify_numerical_gradient_checking()
    verify_pytorch_parity()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 6.4 PASSED CLEANLY!")
    print("=" * 65)
