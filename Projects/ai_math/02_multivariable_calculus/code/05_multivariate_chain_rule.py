"""
Chapter 2.5: Multivariate Chain Rule
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' 2-Layer Neural Chain Rule Walkthrough.
2. Symbolic Differentiation vs. Chain Rule J^T * grad vs. PyTorch Autograd.
3. ResNet Skip Connection Identity Gradient Preservation (Vanishing Gradient Solution).
4. Forward-Mode vs. Reverse-Mode Matrix Chain Associativity Verification.
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Functions: Layer 1, Layer 2, Composite
# =====================================================================
def layer1_forward(x: np.ndarray) -> np.ndarray:
    """
    u1 = x1^2 + 2*x2
    u2 = 3*x1*x2
    """
    x1, x2 = x[0], x[1]
    u1 = x1**2 + 2.0 * x2
    u2 = 3.0 * x1 * x2
    return np.array([u1, u2], dtype=np.float64)


def layer1_jacobian(x: np.ndarray) -> np.ndarray:
    """
    J_g = [[2*x1,  2   ],
           [3*x2,  3*x1]]
    """
    x1, x2 = x[0], x[1]
    return np.array([[2.0 * x1, 2.0], [3.0 * x2, 3.0 * x1]], dtype=np.float64)


def layer2_loss(u: np.ndarray) -> float:
    """
    L(u1, u2) = u1^2 + 2*u1*u2 + 4*u2
    """
    u1, u2 = u[0], u[1]
    return float(u1**2 + 2.0 * u1 * u2 + 4.0 * u2)


def layer2_grad(u: np.ndarray) -> np.ndarray:
    """
    grad_u L = [2*u1 + 2*u2,  2*u1 + 4]^T
    """
    u1, u2 = u[0], u[1]
    g1 = 2.0 * u1 + 2.0 * u2
    g2 = 2.0 * u1 + 4.0
    return np.array([g1, g2], dtype=np.float64)


def direct_symbolic_grad(x: np.ndarray) -> np.ndarray:
    """
    Direct expansion:
    dL/dx1 = 4*x1^3 + 8*x1*x2 + 18*x1^2*x2 + 12*x2^2 + 12*x2
    dL/dx2 = 4*x1^2 + 8*x2 + 6*x1^3 + 24*x1*x2 + 12*x1
    """
    x1, x2 = x[0], x[1]
    gx1 = 4.0 * x1**3 + 8.0 * x1 * x2 + 18.0 * x1**2 * x2 + 12.0 * x2**2 + 12.0 * x2
    gx2 = 4.0 * x1**2 + 8.0 * x2 + 6.0 * x1**3 + 24.0 * x1 * x2 + 12.0 * x1
    return np.array([gx1, gx2], dtype=np.float64)


# =====================================================================
# 2. ResNet Gradient Highway Simulation
# =====================================================================
def simulate_vanishing_gradient(num_layers: int = 15, weight_scale: float = 0.5):
    """
    Simulates gradient flow through a deep vanilla network vs ResNet.
    Vanilla: y = W_L * ... * W_1 * x
    ResNet:  y_{l+1} = y_l + W_l * y_l = (I + W_l) y_l
    """
    np.random.seed(42)
    dim = 8
    # Initialize small weight matrices (contractive mappings)
    weights = [np.random.randn(dim, dim) * weight_scale for _ in range(num_layers)]

    # Downstream loss gradient: v in R^dim
    v_upstream = np.ones(dim, dtype=np.float64)

    # Vanilla backpropagation: grad = v * W_L * W_{L-1} * ... * W_1
    g_vanilla = v_upstream.copy()
    vanilla_norms = [np.linalg.norm(g_vanilla)]
    for W in reversed(weights):
        g_vanilla = g_vanilla @ W
        vanilla_norms.append(np.linalg.norm(g_vanilla))

    # ResNet backpropagation: grad = v * (I + W_L) * (I + W_{L-1}) * ... * (I + W_1)
    g_resnet = v_upstream.copy()
    resnet_norms = [np.linalg.norm(g_resnet)]
    for W in reversed(weights):
        g_resnet = g_resnet @ (np.eye(dim) + W)
        resnet_norms.append(np.linalg.norm(g_resnet))

    return vanilla_norms, resnet_norms


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 2.5: Multivariate Chain Rule — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" 2-Layer Neural Network
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' 2-Layer Neural Chain Rule]")
    x0 = np.array([1.0, 2.0], dtype=np.float64)

    # Forward
    u0 = layer1_forward(x0)
    L0 = layer2_loss(u0)
    print(f"  Input x0:                  {x0}")
    print(f"  Hidden Activations u0:     {u0} (Expected: [5.0, 6.0])")
    print(f"  Scalar Loss L0:            {L0:.4f} (Expected: 109.0000)")

    assert np.allclose(u0, [5.0, 6.0]), f"u0 mismatch: {u0}"
    assert np.isclose(L0, 109.0), f"L0 mismatch: {L0}"

    # Backward Layer 2
    grad_u = layer2_grad(u0)
    print(f"  Upstream Gradient nabla_u: {grad_u} (Expected: [22.0, 14.0])")
    assert np.allclose(grad_u, [22.0, 14.0]), f"grad_u mismatch: {grad_u}"

    # Backward Layer 1 Jacobian
    J_g = layer1_jacobian(x0)
    print(f"  Layer 1 Jacobian J_g:\n{J_g}")
    assert np.allclose(J_g, [[2.0, 2.0], [6.0, 3.0]]), f"J_g mismatch: {J_g}"

    # Composite Chain Rule: grad_x = J_g^T @ grad_u
    grad_x_chain = J_g.T @ grad_u
    print(f"  Composite Gradient nabla_x:{grad_x_chain} (Expected: [128.0, 86.0])")
    assert np.allclose(grad_x_chain, [128.0, 86.0]), f"Chain grad mismatch: {grad_x_chain}"

    # Direct Symbolic Verification
    grad_x_sym = direct_symbolic_grad(x0)
    print(f"  Symbolic Ground Truth:     {grad_x_sym}")
    assert np.allclose(grad_x_chain, grad_x_sym), "Chain rule does not match symbolic algebra!"
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: PyTorch Autograd Cross-Check
    # -------------------------------------------------------------
    print("\n[Test 2: PyTorch Autograd Verification]")
    x_torch = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

    # Forward through same graph in PyTorch
    u1_t = x_torch[0] ** 2 + 2.0 * x_torch[1]
    u2_t = 3.0 * x_torch[0] * x_torch[1]
    loss_t = u1_t**2 + 2.0 * u1_t * u2_t + 4.0 * u2_t

    # Backward pass
    loss_t.backward()

    torch_grad = x_torch.grad.numpy()
    print(f"  PyTorch x.grad:            {torch_grad}")
    assert np.allclose(torch_grad, [128.0, 86.0]), "PyTorch autograd mismatch!"
    print("  -> PyTorch Autograd matches hand chain rule derivation exactly!")

    # -------------------------------------------------------------
    # Test 3: ResNet Skip Connection Gradient Highway Preservation
    # -------------------------------------------------------------
    print("\n[Test 3: ResNet Identity Shortcut vs. Vanishing Gradient]")
    num_layers = 12
    v_norms, r_norms = simulate_vanishing_gradient(num_layers=num_layers, weight_scale=0.15)

    print(f"  Layer Depth: {num_layers}")
    print(f"  Vanilla Gradient Norm at Input: {v_norms[-1]:.6e} (Vanished by factor of {v_norms[0]/v_norms[-1]:.1f}x)")
    print(f"  ResNet Gradient Norm at Input:  {r_norms[-1]:.6e} (Remained stable & non-vanishing)")

    assert v_norms[-1] < 1e-3, "Vanilla gradient should vanish!"
    assert r_norms[-1] > 0.5, "ResNet gradient highway should preserve magnitude!"
    print("  -> ResNet identity highway mathematically prevents vanishing gradients!")

    # -------------------------------------------------------------
    # Test 4: Forward-Mode vs Reverse-Mode Associativity
    # -------------------------------------------------------------
    print("\n[Test 4: Matrix Associativity & Computational Invariance]")
    # Chain of 4 random Jacobians
    np.random.seed(99)
    J1 = np.random.randn(4, 4)
    J2 = np.random.randn(4, 4)
    J3 = np.random.randn(4, 4)
    J4 = np.random.randn(4, 4)
    v = np.random.randn(4)

    # Forward-mode: J_total = (J4 @ (J3 @ (J2 @ J1)))
    J_total = J4 @ J3 @ J2 @ J1
    res_forward = v @ J_total

    # Reverse-mode: v @ J4 @ J3 @ J2 @ J1
    v_step = v @ J4
    v_step = v_step @ J3
    v_step = v_step @ J2
    res_reverse = v_step @ J1

    print(f"  Forward Grouping Result:   {res_forward[:2]}...")
    print(f"  Reverse Grouping Result:   {res_reverse[:2]}...")
    diff = np.max(np.abs(res_forward - res_reverse))
    print(f"  Max Associativity Diff:    {diff:.6e}")
    assert np.isclose(diff, 0.0, atol=1e-12), "Matrix associativity failed!"
    print("  -> Forward-mode and Reverse-mode evaluate identical mathematical functions!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 2.5 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
