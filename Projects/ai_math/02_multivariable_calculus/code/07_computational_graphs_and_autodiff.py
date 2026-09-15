"""
Chapter 2.7: Computational Graphs & Automatic Differentiation (Reverse vs. Forward Mode)
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' 7-Node Non-Linear DAG Evaluation (Forward & Backward).
2. Pure-Python Scratch Reverse-Mode Autograd Engine (Topological Sort + Tape).
3. Pure-Python Forward-Mode Dual Number Engine (Algebraic Ring R[eps] / <eps^2>).
4. Dual Number (Forward) vs. Autograd (Reverse) vs. PyTorch Autograd Cross-Check.
5. Multi-Branch Gradient Accumulation (the necessity of += in adjoints).
"""

import math
import numpy as np
import torch


# =====================================================================
# 1. Pure-Python Reverse-Mode Autograd Engine (Micro-Autograd)
# =====================================================================
class Value:
    """
    A scalar autograd node tracking computational history on a DAG.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be scalar"
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Executes reverse-mode backpropagation via topological sort.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed the root output
        self.grad = 1.0

        # Traverse in reverse topological order
        for node in reversed(topo):
            node._backward()


# =====================================================================
# 2. Forward-Mode Dual Number Engine
# =====================================================================
class DualNumber:
    """
    Dual number x + eps * x_dot where eps^2 = 0.
    """

    def __init__(self, real: float, dual: float = 0.0):
        self.real = float(real)
        self.dual = float(dual)

    def __repr__(self):
        return f"DualNumber({self.real:.6f} + {self.dual:.6f}ε)"

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        return DualNumber(self.real + float(other), self.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # (a + b eps)(c + d eps) = ac + (ad + bc) eps
            return DualNumber(
                self.real * other.real, self.real * other.dual + self.dual * other.real
            )
        return DualNumber(self.real * float(other), self.dual * float(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power: float):
        # (x + eps*x_dot)^p = x^p + p * x^{p-1} * x_dot * eps
        return DualNumber(
            self.real**power, power * (self.real ** (power - 1)) * self.dual
        )

    def sin(self):
        return DualNumber(math.sin(self.real), math.cos(self.real) * self.dual)

    def exp(self):
        return DualNumber(math.exp(self.real), math.exp(self.real) * self.dual)


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 2.7: Computational Graphs & Autodiff — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" DAG Trace using Scratch Engine
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Scratch Autograd Evaluation]")
    x1 = Value(2.0, label="x1")
    x2 = Value(-1.0, label="x2")

    # Construct DAG
    v3 = x1 * x2
    v4 = x1.sin()
    v5 = v3 + v4
    v6 = v5**2
    v7 = v3.exp()
    loss = v6 + v7

    print(f"  Forward Pass:")
    print(f"    v1 (x1):         {x1.data:.6f} (Expected: 2.000000)")
    print(f"    v2 (x2):         {x2.data:.6f} (Expected: -1.000000)")
    print(f"    v3 (x1*x2):      {v3.data:.6f} (Expected: -2.000000)")
    print(f"    v4 (sin(x1)):    {v4.data:.6f} (Expected: 0.909297)")
    print(f"    v5 (v3+v4):      {v5.data:.6f} (Expected: -1.090703)")
    print(f"    v6 (v5^2):       {v6.data:.6f} (Expected: 1.189633)")
    print(f"    v7 (exp(v3)):    {v7.data:.6f} (Expected: 0.135335)")
    print(f"    v8 (Loss):       {loss.data:.6f} (Expected: 1.324968)")

    assert np.isclose(v3.data, -2.0)
    assert np.isclose(v4.data, math.sin(2.0))
    assert np.isclose(v5.data, -2.0 + math.sin(2.0))
    assert np.isclose(v6.data, (-2.0 + math.sin(2.0)) ** 2)
    assert np.isclose(v7.data, math.exp(-2.0))
    assert np.isclose(loss.data, (-2.0 + math.sin(2.0)) ** 2 + math.exp(-2.0))
    print("  -> Forward pass values MATCHED perfectly!")

    # Run Reverse-Mode Backpropagation
    loss.backward()

    print(f"\n  Backward Pass (Adjoints):")
    print(f"    v8_bar (Loss):   {loss.grad:.6f} (Expected: 1.000000)")
    print(f"    v6_bar:          {v6.grad:.6f} (Expected: 1.000000)")
    print(f"    v7_bar:          {v7.grad:.6f} (Expected: 1.000000)")
    print(f"    v5_bar:          {v5.grad:.6f} (Expected: ~-2.181405)")
    print(f"    v4_bar:          {v4.grad:.6f} (Expected: ~-2.181405)")
    print(f"    v3_bar:          {v3.grad:.6f} (Expected: ~-2.046070)")
    print(f"    x1_bar (dL/dx1): {x1.grad:.6f} (Expected: ~+2.953855)")
    print(f"    x2_bar (dL/dx2): {x2.grad:.6f} (Expected: ~-4.092140)")

    assert np.isclose(loss.grad, 1.0)
    assert np.isclose(v6.grad, 1.0)
    assert np.isclose(v7.grad, 1.0)
    assert np.isclose(v5.grad, 2.0 * v5.data)
    assert np.isclose(x1.grad, 2.953855, atol=1e-4)
    assert np.isclose(x2.grad, -4.092140, atol=1e-4)
    print("  -> Scratch Reverse-Mode Autograd adjoints MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: PyTorch Autograd Cross-Check
    # -------------------------------------------------------------
    print("\n[Test 2: PyTorch Autograd Cross-Check]")
    pt_x1 = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    pt_x2 = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)

    pt_loss = (pt_x1 * pt_x2 + torch.sin(pt_x1)) ** 2 + torch.exp(pt_x1 * pt_x2)
    pt_loss.backward()

    print(f"  PyTorch Loss:      {pt_loss.item():.6f}")
    print(f"  PyTorch pt_x1.grad: {pt_x1.grad.item():.6f}")
    print(f"  PyTorch pt_x2.grad: {pt_x2.grad.item():.6f}")

    assert np.isclose(pt_loss.item(), loss.data)
    assert np.isclose(pt_x1.grad.item(), x1.grad)
    assert np.isclose(pt_x2.grad.item(), x2.grad)
    print("  -> Scratch Autograd matches PyTorch autograd to machine precision!")

    # -------------------------------------------------------------
    # Test 3: Forward-Mode Dual Number Differentiation
    # -------------------------------------------------------------
    print("\n[Test 3: Forward-Mode Dual Number Engine Verification]")
    # Derivative w.r.t. x1: seed x1_dot = 1.0, x2_dot = 0.0
    d_x1 = DualNumber(2.0, 1.0)
    d_x2 = DualNumber(-1.0, 0.0)
    d_loss_x1 = (d_x1 * d_x2 + d_x1.sin()) ** 2 + (d_x1 * d_x2).exp()

    # Derivative w.r.t. x2: seed x1_dot = 0.0, x2_dot = 1.0
    d2_x1 = DualNumber(2.0, 0.0)
    d2_x2 = DualNumber(-1.0, 1.0)
    d_loss_x2 = (d2_x1 * d2_x2 + d2_x1.sin()) ** 2 + (d2_x1 * d2_x2).exp()

    print(f"  Forward Dual dL/dx1: {d_loss_x1.dual:.6f} (Expected: {x1.grad:.6f})")
    print(f"  Forward Dual dL/dx2: {d_loss_x2.dual:.6f} (Expected: {x2.grad:.6f})")

    assert np.isclose(d_loss_x1.real, loss.data)
    assert np.isclose(d_loss_x1.dual, x1.grad)
    assert np.isclose(d_loss_x2.dual, x2.grad)
    print("  -> Forward-Mode Dual Numbers produce identical derivatives with zero roundoff!")

    # -------------------------------------------------------------
    # Test 4: Gradient Accumulation (Fan-Out += Correctness)
    # -------------------------------------------------------------
    print("\n[Test 4: Multi-Branch Gradient Accumulation Check]")
    # f(a) = a + a + a => df/da = 3.0
    a = Value(5.0)
    b = a + a + a
    b.backward()
    print(f"  Branching sum: b = a + a + a (at a=5.0)")
    print(f"  a.grad: {a.grad:.1f} (Expected: 3.0)")
    assert np.isclose(a.grad, 3.0), "Gradient accumulation failed on multi-branch node!"
    print("  -> Multi-branch gradient accumulation (+=) verified!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 2.7 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
