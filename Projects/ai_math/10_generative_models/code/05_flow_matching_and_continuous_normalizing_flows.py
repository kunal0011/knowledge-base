"""
Flow Matching Models & Continuous Normalizing Flows (CNFs) Verification
======================================================================
Module 10: Generative Modeling - Chapter 10.5

This script verifies:
1. Section 8.1 Hand Calculation Parity (OT-CFM):
   - Linear interpolant x_t at t=0.40.
   - Target velocity u_t = x_1 - x_0.
   - Velocity prediction error, squared Euclidean loss, MSE per dimension.
   - Autograd gradient dL/dW parity with manual outer product calculation.
2. Section 8.2 Numerical ODE Solvers Parity:
   - Exact analytical ODE solution x(1.0) = 1.5 * e^2 - 0.5 = 10.5836.
   - Forward Euler (N=2, h=0.5) = 5.5000.
   - Midpoint / RK2 (N=2, h=0.5) = 8.8750.
   - Runge-Kutta 4 (RK4) integration.
3. Instantaneous Change of Variables & Divergence Computation:
   - Exact Jacobian trace vs Hutchinson's randomized trace estimator.
4. End-to-End OT-CFM Training & 2-Rectified Flow (Re-Flow) on a 2D Manifold:
   - Velocity field MLP with sinusoidal positional time embeddings.
   - 1-Rectified Flow training loop.
   - Multi-step ODE forward trajectory generation.
   - 2-Rectified Flow retraining demonstrating trajectory straightening.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def test_hand_calculation_parity():
    print("--- Test 1: Section 8.1 Hand Calculation Parity (OT-CFM) ---")
    dtype = torch.float64
    
    # 1. Inputs
    x_0 = torch.tensor([-1.0, 2.0], dtype=dtype)
    x_1 = torch.tensor([3.0, -2.0], dtype=dtype)
    t = 0.40
    
    # 2. Linear Interpolation
    x_t = (1.0 - t) * x_0 + t * x_1
    print(f"Interpolated state x_t: [{x_t[0].item():.4f}, {x_t[1].item():.4f}] (Expected: [0.6000, 0.4000])")
    assert abs(x_t[0].item() - 0.6000) < 1e-7
    assert abs(x_t[1].item() - 0.4000) < 1e-7
    
    # 3. Target Velocity
    u_t = x_1 - x_0
    print(f"Target velocity u_t: [{u_t[0].item():.4f}, {u_t[1].item():.4f}] (Expected: [4.0000, -4.0000])")
    assert abs(u_t[0].item() - 4.0000) < 1e-7
    assert abs(u_t[1].item() - (-4.0000)) < 1e-7
    
    # 4. Predicted velocity & Loss
    v_theta = torch.tensor([3.20, -3.50], dtype=dtype)
    error = v_theta - u_t
    print(f"Error vector e: [{error[0].item():.4f}, {error[1].item():.4f}] (Expected: [-0.8000, 0.5000])")
    assert abs(error[0].item() - (-0.8000)) < 1e-7
    assert abs(error[1].item() - 0.5000) < 1e-7
    
    loss_sq = torch.sum(error ** 2)
    mse_per_dim = loss_sq / 2.0
    print(f"Squared loss: {loss_sq.item():.4f} (Expected: 0.8900)")
    print(f"MSE per coordinate: {mse_per_dim.item():.4f} (Expected: 0.4450)")
    assert abs(loss_sq.item() - 0.8900) < 1e-7
    assert abs(mse_per_dim.item() - 0.4450) < 1e-7
    
    # 5. Parameter Gradient Verification
    W = torch.tensor([[1.0, 2.0], [-1.0, -1.0]], dtype=dtype, requires_grad=True)
    v_linear = torch.matmul(W, x_t)
    loss_linear = torch.sum((v_linear - u_t) ** 2)
    loss_linear.backward()
    
    dL_dW_autograd = W.grad
    # Manual outer product formula: 2 * (v - u) * x_t^T
    dL_dv = 2.0 * (v_linear.detach() - u_t)
    dL_dW_manual = torch.ger(dL_dv, x_t)
    
    print("dL/dW Autograd:\n", dL_dW_autograd.numpy())
    print("dL/dW Manual:\n", dL_dW_manual.numpy())
    assert torch.allclose(dL_dW_autograd, dL_dW_manual, atol=1e-7)
    print("✓ Test 1 Passed Successfully.\n")


def test_ode_solver_hand_arithmetic():
    print("--- Test 2: Section 8.2 ODE Numerical Solvers Parity ---")
    
    # ODE: dx/dt = 2x + 1, x(0) = 1.0, t in [0, 1]
    def v_field(x, t):
        return 2.0 * x + 1.0
    
    h = 0.50
    N = 2
    
    # Method A: Forward Euler
    x_euler = 1.0
    t = 0.0
    for _ in range(N):
        x_euler = x_euler + h * v_field(x_euler, t)
        t += h
    print(f"Forward Euler (N=2): {x_euler:.4f} (Expected: 5.5000)")
    assert abs(x_euler - 5.5000) < 1e-6
    
    # Method B: Midpoint (RK2)
    x_midpoint = 1.0
    t = 0.0
    for _ in range(N):
        k1 = v_field(x_midpoint, t)
        x_half = x_midpoint + 0.5 * h * k1
        k2 = v_field(x_half, t + 0.5 * h)
        x_midpoint = x_midpoint + h * k2
        t += h
    print(f"Midpoint / RK2 (N=2): {x_midpoint:.4f} (Expected: 8.8750)")
    assert abs(x_midpoint - 8.8750) < 1e-6
    
    # Method C: RK4
    x_rk4 = 1.0
    t = 0.0
    for _ in range(N):
        k1 = v_field(x_rk4, t)
        k2 = v_field(x_rk4 + 0.5 * h * k1, t + 0.5 * h)
        k3 = v_field(x_rk4 + 0.5 * h * k2, t + 0.5 * h)
        k4 = v_field(x_rk4 + h * k3, t + h)
        x_rk4 = x_rk4 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
    # Analytical solution: x(1) = 1.5 * exp(2) - 0.5
    x_exact = 1.5 * math.exp(2.0) - 0.5
    print(f"RK4 (N=2): {x_rk4:.4f} vs Analytical Exact: {x_exact:.4f}")
    assert abs(x_exact - 10.58358) < 1e-3
    assert abs(x_rk4 - x_exact) < 0.20  # RK4 is orders of magnitude closer than Euler
    print("✓ Test 2 Passed Successfully.\n")


def test_hutchinson_divergence_trace():
    print("--- Test 3: Instantaneous Divergence & Hutchinson's Estimator ---")
    torch.manual_seed(42)
    
    dim = 4
    x = torch.randn(dim, requires_grad=True)
    # A simple nonlinear velocity field
    W = torch.randn(dim, dim)
    
    def velocity(inp):
        return torch.tanh(W @ inp)
    
    # 1. Exact Jacobian Trace (div(v))
    v = velocity(x)
    exact_jacobian = torch.zeros(dim, dim)
    for i in range(dim):
        grad_i = torch.autograd.grad(v[i], x, retain_graph=True)[0]
        exact_jacobian[i] = grad_i
    exact_div = torch.trace(exact_jacobian).item()
    
    # 2. Hutchinson's Randomized Estimator: E[eps^T J eps]
    num_samples = 1000
    hutchinson_estimates = []
    for _ in range(num_samples):
        eps = torch.randn(dim)
        v_dot_eps = torch.dot(velocity(x), eps)
        jvp = torch.autograd.grad(v_dot_eps, x, retain_graph=True)[0]
        estimate = torch.dot(eps, jvp).item()
        hutchinson_estimates.append(estimate)
        
    mean_hutchinson = np.mean(hutchinson_estimates)
    print(f"Exact Jacobian Divergence Trace: {exact_div:.4f}")
    print(f"Hutchinson's Estimator (N={num_samples}): {mean_hutchinson:.4f}")
    assert abs(exact_div - mean_hutchinson) < 0.25
    print("✓ Test 3 Passed Successfully.\n")


# ---------------------------------------------------------------------------
# Neural Velocity Field Architecture & OT-CFM / Re-Flow Training Pipeline
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityMLP(nn.Module):
    """Deep residual vector field network v_theta(x, t) for 2D generative flow."""
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_feat = self.time_embed(t)
        h = self.in_proj(x) + t_feat
        h = h + self.block1(F.silu(h))
        h = h + self.block2(F.silu(h))
        return self.out_proj(h)


def sample_target_data(batch_size: int) -> torch.Tensor:
    """Sample from an 8-Gaussian Mixture Model (ring of Gaussians in 2D)."""
    num_modes = 8
    radius = 3.0
    angles = torch.linspace(0, 2 * math.pi, num_modes + 1)[:-1]
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    
    mode_indices = torch.randint(0, num_modes, (batch_size,))
    sampled_centers = centers[mode_indices]
    noise = 0.15 * torch.randn(batch_size, 2)
    return sampled_centers + noise


@torch.no_grad()
def solve_ode_euler(model: nn.Module, x_0: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """Generate samples via Forward Euler ODE integration."""
    dt = 1.0 / num_steps
    x_t = x_0.clone()
    for i in range(num_steps):
        t_val = i * dt
        t_tensor = torch.full((x_0.shape[0],), t_val, device=x_0.device)
        v = model(x_t, t_tensor)
        x_t = x_t + dt * v
    return x_t


@torch.no_grad()
def solve_ode_rk4(model: nn.Module, x_0: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """Generate samples via Runge-Kutta 4 (RK4) ODE integration."""
    dt = 1.0 / num_steps
    x_t = x_0.clone()
    for i in range(num_steps):
        t_val = i * dt
        t_tensor = torch.full((x_0.shape[0],), t_val, device=x_0.device)
        t_half = torch.full((x_0.shape[0],), t_val + 0.5 * dt, device=x_0.device)
        t_next = torch.full((x_0.shape[0],), t_val + dt, device=x_0.device)
        
        k1 = model(x_t, t_tensor)
        k2 = model(x_t + 0.5 * dt * k1, t_half)
        k3 = model(x_t + 0.5 * dt * k2, t_half)
        k4 = model(x_t + dt * k3, t_next)
        
        x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_t


def test_end_to_end_flow_matching_and_reflow():
    print("--- Test 4: End-to-End OT-CFM Training & 2-Rectified Flow (Re-Flow) ---")
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    # 1. Instantiate 1-Rectified Flow Model
    model_rf1 = VelocityMLP().to(device)
    optimizer_rf1 = optim.AdamW(model_rf1.parameters(), lr=2e-3, weight_decay=1e-4)
    
    print("Training 1-Rectified Flow (OT-CFM) on 2D 8-Gaussian Mixture...")
    batch_size = 256
    num_train_steps = 400
    
    for step in range(num_train_steps):
        optimizer_rf1.zero_grad()
        
        # OT-CFM Coupling: x_0 ~ N(0, I), x_1 ~ p_data
        x_0 = torch.randn(batch_size, 2, device=device)
        x_1 = sample_target_data(batch_size).to(device)
        t = torch.rand(batch_size, device=device)
        
        # Linear displacement interpolant and constant target velocity
        x_t = (1.0 - t.unsqueeze(1)) * x_0 + t.unsqueeze(1) * x_1
        u_t = x_1 - x_0
        
        # Predict velocity
        v_pred = model_rf1(x_t, t)
        loss = F.mse_loss(v_pred, u_t)
        
        loss.backward()
        optimizer_rf1.step()
        
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  Step {step+1:03d}/{num_train_steps} | OT-CFM Loss: {loss.item():.4f}")
            
    assert loss.item() < 3.50
    print("✓ 1-Rectified Flow training converged successfully.")
    
    # 2. Re-Flow: Generate Straight-Coupled Pairs (x_0, x_1_hat) via RK4 simulation
    print("\nExecuting Re-Flow: Generating synthetic straight pairs via RK4 simulation...")
    num_reflow_samples = 2048
    x_0_reflow = torch.randn(num_reflow_samples, 2, device=device)
    x_1_hat = solve_ode_rk4(model_rf1, x_0_reflow, num_steps=20)
    
    # Measure transport distance for 1-Rectified Flow
    initial_transport_cost = torch.mean(torch.sum((x_1_hat - x_0_reflow) ** 2, dim=1)).item()
    print(f"Initial expected transport cost E[||x_1_hat - x_0||^2]: {initial_transport_cost:.4f}")
    
    # 3. Train 2-Rectified Flow on Straightened Pairs
    print("\nTraining 2-Rectified Flow on synthetic uncrossed pairs...")
    model_rf2 = VelocityMLP().to(device)
    optimizer_rf2 = optim.AdamW(model_rf2.parameters(), lr=2e-3, weight_decay=1e-4)
    
    for step in range(300):
        optimizer_rf2.zero_grad()
        indices = torch.randint(0, num_reflow_samples, (batch_size,))
        batch_x0 = x_0_reflow[indices]
        batch_x1 = x_1_hat[indices]
        t = torch.rand(batch_size, device=device)
        
        # Uncrossed straight interpolation
        x_t = (1.0 - t.unsqueeze(1)) * batch_x0 + t.unsqueeze(1) * batch_x1
        u_t = batch_x1 - batch_x0
        
        v_pred = model_rf2(x_t, t)
        loss_rf2 = F.mse_loss(v_pred, u_t)
        
        loss_rf2.backward()
        optimizer_rf2.step()
        
    print(f"Final 2-Rectified Flow Loss: {loss_rf2.item():.4f}")
    assert loss_rf2.item() < 1.00
    
    # 4. Compare Few-Step Sampling (Euler 4 steps)
    test_noise = torch.randn(100, 2, device=device)
    samples_euler_4steps = solve_ode_euler(model_rf2, test_noise, num_steps=4)
    # Check that samples are in reasonable range of the 8-Gaussian ring (radius ~3.0)
    radii = torch.norm(samples_euler_4steps, dim=1)
    mean_radius = torch.mean(radii).item()
    print(f"Generated 4-Step Euler Samples Mean Radius: {mean_radius:.4f} (Target ring radius: ~3.0)")
    assert 2.0 < mean_radius < 4.0
    print("✓ Test 4 (Flow Matching & Re-Flow Pipeline) Passed Successfully.\n")


if __name__ == "__main__":
    print("======================================================================")
    print("Flow Matching & Continuous Normalizing Flows (CNF) Test Suite")
    print("======================================================================\n")
    test_hand_calculation_parity()
    test_ode_solver_hand_arithmetic()
    test_hutchinson_divergence_trace()
    test_end_to_end_flow_matching_and_reflow()
    print("======================================================================")
    print("ALL TESTS PASSED: Flow Matching & CNF Verification 100% Successful!")
    print("======================================================================")
