"""
Denoising Diffusion Probabilistic Models (DDPM) & Score SDE Verification
======================================================================
Module 10: Generative Modeling - Chapter 10.4

This script verifies:
1. Part 5 Tom Yeh "AI by Hand" Complete Numerical Parity:
   - Closed-form forward noisy state x_2 calculation.
   - Denoising L_simple loss computation.
   - Posterior mean mu_tilde_2, posterior variance beta_tilde_2, and sample x_1.
   - Classifier-Free Guidance (CFG) vector extrapolation.
2. Step-by-Step Markov Chain vs O(1) Closed-Form Sampling Parity.
3. Tweedie's Formula and Stein Score Function Autograd Equivalence:
   grad_{x_t} log q(x_t | x_0) == -eps / sqrt(1 - alpha_bar_t).
4. End-to-End DDPM Training and Reverse Diffusion Sampling on 2D distribution.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def test_part5_hand_arithmetic_parity():
    print("--- Test 1: Part 5 Tom Yeh DDPM Hand Arithmetic Parity ---")
    dtype = torch.float64
    
    # 1. Inputs & Parameters
    x_0 = torch.tensor([1.0, 2.0], dtype=dtype)
    beta_1 = 0.1
    beta_2 = 0.2
    
    alpha_1 = 1.0 - beta_1  # 0.9
    alpha_2 = 1.0 - beta_2  # 0.8
    
    alpha_bar_1 = alpha_1                    # 0.9
    alpha_bar_2 = alpha_1 * alpha_2          # 0.72
    
    eps = torch.tensor([0.5, -1.0], dtype=dtype)
    
    # 2. Closed-form forward sample x_2
    sqrt_alpha_bar_2 = math.sqrt(alpha_bar_2)
    sqrt_one_minus_alpha_bar_2 = math.sqrt(1.0 - alpha_bar_2)
    
    x_2 = sqrt_alpha_bar_2 * x_0 + sqrt_one_minus_alpha_bar_2 * eps
    print(f"sqrt(alpha_bar_2): {sqrt_alpha_bar_2:.4f} (Hand: 0.8485)")
    print(f"sqrt(1 - alpha_bar_2): {sqrt_one_minus_alpha_bar_2:.4f} (Hand: 0.5292)")
    print(f"Noisy state x_2: [{x_2[0].item():.4f}, {x_2[1].item():.4f}] (Hand: [1.1131, 1.1679])")
    assert abs(x_2[0].item() - 1.1131) < 1e-3
    assert abs(x_2[1].item() - 1.1679) < 1e-3
    
    # 3. Denoising loss evaluation
    eps_theta = torch.tensor([0.4, -0.8], dtype=dtype)
    loss_simple = torch.sum((eps - eps_theta) ** 2)
    print(f"L_simple Loss: {loss_simple.item():.4f} (Hand: 0.0500)")
    assert abs(loss_simple.item() - 0.0500) < 1e-6
    
    # 4. Reverse Step Arithmetic (t=2 -> t=1)
    # mu_tilde_2 = (1 / sqrt(alpha_2)) * (x_2 - (beta_2 / sqrt(1 - alpha_bar_2)) * eps_theta)
    coeff_x2 = 1.0 / math.sqrt(alpha_2)
    coeff_eps = beta_2 / math.sqrt(1.0 - alpha_bar_2)
    
    mu_tilde_2 = coeff_x2 * (x_2 - coeff_eps * eps_theta)
    print(f"Posterior mean mu_tilde_2: [{mu_tilde_2[0].item():.4f}, {mu_tilde_2[1].item():.4f}] (Hand: [1.0754, 1.6438])")
    assert abs(mu_tilde_2[0].item() - 1.0754) < 1e-3
    assert abs(mu_tilde_2[1].item() - 1.6438) < 1e-3
    
    # Posterior variance
    beta_tilde_2 = ((1.0 - alpha_bar_1) / (1.0 - alpha_bar_2)) * beta_2
    sigma_2 = math.sqrt(beta_tilde_2)
    print(f"Posterior variance beta_tilde_2: {beta_tilde_2:.4f} (Hand: 0.0714)")
    print(f"Posterior std sigma_2: {sigma_2:.4f} (Hand: 0.2673)")
    assert abs(beta_tilde_2 - 0.0714) < 1e-3
    assert abs(sigma_2 - 0.2673) < 1e-3
    
    # Sample state x_1
    z_noise = torch.tensor([0.1, -0.1], dtype=dtype)
    x_1 = mu_tilde_2 + sigma_2 * z_noise
    print(f"Reverse sample x_1: [{x_1[0].item():.4f}, {x_1[1].item():.4f}] (Hand: [1.1021, 1.6171])")
    assert abs(x_1[0].item() - 1.1021) < 1e-3
    assert abs(x_1[1].item() - 1.6171) < 1e-3
    
    # 5. Classifier-Free Guidance extrapolation
    eps_uncond = torch.tensor([0.4, -0.8], dtype=dtype)
    eps_cond = torch.tensor([0.6, -1.1], dtype=dtype)
    s = 2.0
    
    eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
    print(f"CFG guided noise: [{eps_guided[0].item():.4f}, {eps_guided[1].item():.4f}] (Hand: [0.8000, -1.4000])")
    assert torch.allclose(eps_guided, torch.tensor([0.8, -1.4], dtype=dtype))
    
    print("✓ Part 5 Hand Arithmetic verified to exact precision!\n")


def test_markov_vs_closed_form_marginal():
    print("--- Test 2: Markov Chain Iteration vs. O(1) Closed-Form Sampling ---")
    torch.manual_seed(42)
    np.random.seed(42)
    
    T = 10
    betas = torch.linspace(0.01, 0.05, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    x_0 = torch.tensor([3.5, -2.5], dtype=torch.float64)
    N = 100_000
    
    # Method A: Step-by-step Markov simulation
    x_step = x_0.unsqueeze(0).repeat(N, 1)  # (N, 2)
    for t in range(T):
        noise = torch.randn_like(x_step)
        x_step = math.sqrt(alphas[t]) * x_step + math.sqrt(betas[t]) * noise
        
    mean_step = torch.mean(x_step, dim=0)
    var_step = torch.var(x_step, dim=0)
    
    # Method B: O(1) Closed-form formula
    # x_T = sqrt(alpha_bar_T) * x_0 + sqrt(1 - alpha_bar_T) * eps
    eps_direct = torch.randn(N, 2, dtype=torch.float64)
    x_direct = math.sqrt(alpha_bars[-1]) * x_0 + math.sqrt(1.0 - alpha_bars[-1]) * eps_direct
    
    mean_direct = torch.mean(x_direct, dim=0)
    var_direct = torch.var(x_direct, dim=0)
    
    # Theoretical values
    theoretical_mean = math.sqrt(alpha_bars[-1]) * x_0
    theoretical_var = 1.0 - alpha_bars[-1]
    
    print(f"Theoretical Mean: [{theoretical_mean[0].item():.4f}, {theoretical_mean[1].item():.4f}]")
    print(f"Markov Step Mean: [{mean_step[0].item():.4f}, {mean_step[1].item():.4f}]")
    print(f"Direct O(1) Mean: [{mean_direct[0].item():.4f}, {mean_direct[1].item():.4f}]")
    
    print(f"Theoretical Variance: {theoretical_var:.4f}")
    print(f"Markov Step Variance: [{var_step[0].item():.4f}, {var_step[1].item():.4f}]")
    print(f"Direct O(1) Variance: [{var_direct[0].item():.4f}, {var_direct[1].item():.4f}]")
    
    assert torch.allclose(mean_step, mean_direct, atol=0.02)
    assert torch.allclose(var_step, var_direct, atol=0.02)
    print("✓ Markov simulation matches closed-form marginal sampling perfectly!\n")


def test_tweedie_and_score_equivalence():
    print("--- Test 3: Tweedie's Formula & Score Function Autograd Equivalence ---")
    dtype = torch.float64
    x_0 = torch.tensor([1.5, -0.8], dtype=dtype)
    alpha_bar_t = 0.65
    sigma_t = math.sqrt(1.0 - alpha_bar_t)
    
    eps = torch.tensor([0.7, -1.2], dtype=dtype)
    x_t = (math.sqrt(alpha_bar_t) * x_0 + sigma_t * eps).clone().detach().requires_grad_(True)
    
    # Analytical score function: s(x_t) = -eps / sqrt(1 - alpha_bar_t)
    score_analytical = -eps / sigma_t
    
    # Autograd score function: grad_{x_t} log q(x_t | x_0)
    # log q(x_t | x_0) = -0.5 * d * log(2*pi) - 0.5 * d * log(sigma_t^2) - ||x_t - sqrt(alpha_bar_t)*x_0||^2 / (2 * sigma_t^2)
    mean_target = math.sqrt(alpha_bar_t) * x_0
    var_target = 1.0 - alpha_bar_t
    log_q = -0.5 * torch.sum((x_t - mean_target) ** 2) / var_target
    
    score_autograd = torch.autograd.grad(log_q, x_t)[0]
    
    print(f"Analytical Score (-eps / sigma_t): [{score_analytical[0].item():.4f}, {score_analytical[1].item():.4f}]")
    print(f"Autograd Score (grad log q):        [{score_autograd[0].item():.4f}, {score_autograd[1].item():.4f}]")
    
    assert torch.allclose(score_analytical, score_autograd, atol=1e-7)
    
    # Tweedie's reconstruction: E[x_0 | x_t] = (x_t + sigma_t^2 * s(x_t)) / sqrt(alpha_bar_t)
    x_0_reconstructed = (x_t + var_target * score_analytical) / math.sqrt(alpha_bar_t)
    print(f"Clean Target x_0:                  [{x_0[0].item():.4f}, {x_0[1].item():.4f}]")
    print(f"Tweedie Reconstructed x_0:         [{x_0_reconstructed[0].item():.4f}, {x_0_reconstructed[1].item():.4f}]")
    assert torch.allclose(x_0, x_0_reconstructed, atol=1e-7)
    print("✓ Tweedie's formula and Stein score equivalence verified!\n")


class ToyDiffusionModel(nn.Module):
    """MLP predicting noise eps_theta(x_t, t)."""
    def __init__(self, in_dim=2, hidden_dim=128, T=50):
        super().__init__()
        self.t_emb = nn.Embedding(T, 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def forward(self, x, t):
        emb = self.t_emb(t)
        inp = torch.cat([x, emb], dim=-1)
        return self.net(inp)


def test_end_to_end_ddpm_training_and_sampling():
    print("--- Test 4: End-to-End DDPM Training & Reverse Sampling ---")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2D Data Distribution: Two distinct clusters at [-2, -2] and [+2, +2]
    N = 2000
    cluster_choice = torch.bernoulli(torch.full((N,), 0.5))
    real_data = torch.where(
        cluster_choice.unsqueeze(1) == 0,
        torch.randn(N, 2) * 0.3 - 2.0,
        torch.randn(N, 2) * 0.3 + 2.0
    )
    
    # Diffusion parameters
    T = 50
    betas = torch.linspace(1e-4, 0.04, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    
    model = ToyDiffusionModel(in_dim=2, hidden_dim=128, T=T)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    batch_size = 128
    dataset = torch.utils.data.TensorDataset(real_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train for 30 epochs
    for epoch in range(30):
        for (batch_x,) in loader:
            optimizer.zero_grad()
            B = batch_x.size(0)
            t = torch.randint(0, T, (B,))
            
            a_bar = alphas_bar[t].unsqueeze(1)
            eps = torch.randn_like(batch_x)
            
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
            x_t = torch.sqrt(a_bar) * batch_x + torch.sqrt(1.0 - a_bar) * eps
            
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, eps)
            loss.backward()
            optimizer.step()
            
    print(f"Final training batch MSE loss: {loss.item():.4f}")
    assert loss.item() < 0.70, f"Diffusion model failed to learn noise prediction: {loss.item()}"
    
    # Reverse Sampling from pure noise x_T ~ N(0, I)
    num_samples = 400
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 2)
        for t_step in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t_step, dtype=torch.long)
            eps_pred = model(x, t_tensor)
            
            alpha_t = alphas[t_step]
            alpha_bar_t = alphas_bar[t_step]
            beta_t = betas[t_step]
            
            if t_step > 0:
                alpha_bar_prev = alphas_bar[t_step - 1]
                beta_tilde = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t
                sigma = math.sqrt(beta_tilde)
                z = torch.randn_like(x)
            else:
                sigma = 0.0
                z = 0.0
                
            x = (1.0 / math.sqrt(alpha_t)) * (x - (beta_t / math.sqrt(1.0 - alpha_bar_t)) * eps_pred) + sigma * z
            
    # Generated samples should be bimodal around -2.0 or +2.0
    mean_abs_coord = torch.mean(torch.abs(x)).item()
    print(f"Generated samples mean |coordinate|: {mean_abs_coord:.3f} (Expected near cluster mode ~2.000)")
    assert abs(mean_abs_coord - 2.0) < 0.6, f"Generated samples did not land in target clusters: {mean_abs_coord}"
    print("✓ End-to-end DDPM reverse sampling generated authentic target distribution!\n")


if __name__ == "__main__":
    test_part5_hand_arithmetic_parity()
    test_markov_vs_closed_form_marginal()
    test_tweedie_and_score_equivalence()
    test_end_to_end_ddpm_training_and_sampling()
    print("=========================================================")
    print("ALL MODULE 10 CHAPTER 10.4 VERIFICATIONS PASSED (100%)")
    print("=========================================================")
