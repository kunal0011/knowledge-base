"""
Taxonomy of Generative Models & Density Transformation Verification
===================================================================
Module 10: Generative Modeling - Chapter 10.1

This script verifies:
1. Part 5 Tom Yeh "AI by Hand" Change of Variables Density Transformation:
   z ~ U[0, 2] -> x = z^2 => p_x(x) = 1 / (4 * sqrt(x))
   - Exact analytical point evaluations at x = 0.25, 1.0, 4.0.
   - Numerical integration proving probability conservation integral = 1.0.
   - Monte Carlo empirical density matching via Kolmogorov-Smirnov test.
2. Forward KL (Zero-Avoiding / Mode-Covering) vs Reverse KL (Zero-Forcing / Mode-Collapsing)
   Optimization on a bimodal Gaussian mixture:
   - Forward KL fits the mean (mu=0) with inflated variance (sigma^2=26).
   - Reverse KL collapses to a single mode (mu=-5 or +5, sigma=1).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate
from scipy import stats

def test_part5_change_of_variables():
    print("--- Test 1: Part 5 Tom Yeh Hand Arithmetic Verification ---")
    
    # Analytical density function
    def p_x(x):
        return 1.0 / (4.0 * np.sqrt(x))
    
    # Evaluated points from notes
    x_test = [0.25, 1.00, 4.00]
    expected_densities = [0.5000, 0.2500, 0.1250]
    
    for x_val, expected in zip(x_test, expected_densities):
        computed = p_x(x_val)
        print(f"Point x={x_val:.2f}: Analytical={computed:.4f}, Expected={expected:.4f}")
        assert abs(computed - expected) < 1e-6, f"Mismatch at x={x_val}: {computed} vs {expected}"
    
    # Numerical integration from 0 to 4
    # Note: 1 / (4 * sqrt(x)) has an integrable singularity at 0
    integral_val, abserr = integrate.quad(p_x, 0.0, 4.0)
    print(f"Integral of p_x(x) over [0, 4]: {integral_val:.10f} (Error estimate: {abserr:.2e})")
    assert abs(integral_val - 1.0) < 1e-8, f"Integral does not equal 1.0: {integral_val}"
    
    # Empirical verification via random sampling
    np.random.seed(42)
    N = 100_000
    z_samples = np.random.uniform(0.0, 2.0, size=N)
    x_samples = z_samples ** 2
    
    # CDF of x: F(x) = P(X <= x) = P(Z^2 <= x) = P(Z <= sqrt(x)) = sqrt(x) / 2 for x in [0, 4]
    def cdf_x(x):
        return np.clip(np.sqrt(np.clip(x, 0, 4)) / 2.0, 0.0, 1.0)
    
    ks_stat, p_val = stats.kstest(x_samples, cdf_x)
    print(f"Kolmogorov-Smirnov test against analytical CDF: KS-stat = {ks_stat:.4f}, p-value = {p_val:.4f}")
    assert p_val > 0.01, f"Empirical samples deviate significantly from theoretical distribution (p={p_val})"
    print("✓ Part 5 Change of Variables verified successfully!\n")


def test_forward_vs_reverse_kl():
    print("--- Test 2: Forward KL vs Reverse KL on Bimodal Target ---")
    # Target distribution: p(x) = 0.5 * N(-5, 1) + 0.5 * N(+5, 1)
    # Parametric model: q_theta(x) = N(mu, sigma^2)
    
    # 1. Forward KL: min_{mu, sigma} KL(p || q) = E_p[log p(x) - log q(x)]
    # Argmin is equivalent to max_{mu, sigma} E_p[log q(x)]
    # Analytical solution:
    # mu* = E_p[x] = 0.5(-5) + 0.5(5) = 0.0
    # Var_p(x) = E_p[x^2] - (E_p[x])^2 = 0.5(25 + 1) + 0.5(25 + 1) - 0 = 26.0
    # sigma* = sqrt(26) ≈ 5.0990
    
    # We verify via Monte Carlo optimization of E_p[-log q(x)]
    torch.manual_seed(42)
    N_samples = 50_000
    # Sample from p
    modes = torch.bernoulli(torch.full((N_samples,), 0.5))
    target_samples = torch.where(modes == 0, torch.randn(N_samples) - 5.0, torch.randn(N_samples) + 5.0)
    
    # Train q parameters for Forward KL
    mu_fwd = nn.Parameter(torch.tensor([2.0]))
    log_sigma_fwd = nn.Parameter(torch.tensor([0.0]))
    opt_fwd = optim.Adam([mu_fwd, log_sigma_fwd], lr=0.05)
    
    for step in range(500):
        opt_fwd.zero_grad()
        sigma = torch.exp(log_sigma_fwd)
        # -log q(x) = 0.5 * log(2*pi*sigma^2) + 0.5 * ((x - mu)/sigma)^2
        nll = 0.5 * math.log(2 * math.pi) + log_sigma_fwd + 0.5 * torch.mean(((target_samples - mu_fwd) / sigma) ** 2)
        nll.backward()
        opt_fwd.step()
        
    fwd_mu_opt = mu_fwd.item()
    fwd_sigma_opt = torch.exp(log_sigma_fwd).item()
    fwd_var_opt = fwd_sigma_opt ** 2
    print(f"Forward KL (Mode-Covering):")
    print(f"  Optimal mu: {fwd_mu_opt:.3f} (Theoretical: 0.000)")
    print(f"  Optimal sigma^2: {fwd_var_opt:.3f} (Theoretical: 26.000, sigma: {math.sqrt(26):.3f})")
    assert abs(fwd_mu_opt - 0.0) < 0.2, f"Forward KL mu expected ~0, got {fwd_mu_opt}"
    assert abs(fwd_var_opt - 26.0) < 1.5, f"Forward KL variance expected ~26, got {fwd_var_opt}"
    
    # 2. Reverse KL: min_{mu, sigma} KL(q || p) = E_q[log q(x) - log p(x)]
    # Target log p(x):
    def log_p(x):
        # p(x) = 0.5 * N(x; -5, 1) + 0.5 * N(x; 5, 1)
        # Using LogSumExp for numerical stability
        log_norm_const = -0.5 * math.log(2 * math.pi)
        log_comp1 = -0.5 * (x + 5.0) ** 2 + log_norm_const + math.log(0.5)
        log_comp2 = -0.5 * (x - 5.0) ** 2 + log_norm_const + math.log(0.5)
        return torch.logsumexp(torch.stack([log_comp1, log_comp2], dim=-1), dim=-1)

    # Initialize near positive mode (mu = 3.0)
    mu_rev = nn.Parameter(torch.tensor([3.0]))
    log_sigma_rev = nn.Parameter(torch.tensor([0.5]))
    opt_rev = optim.Adam([mu_rev, log_sigma_rev], lr=0.02)
    
    M = 4096
    for step in range(800):
        opt_rev.zero_grad()
        sigma = torch.exp(log_sigma_rev)
        # Reparameterization trick: x = mu + sigma * eps
        eps = torch.randn(M)
        q_samples = mu_rev + sigma * eps
        
        # log q(x)
        log_q = -0.5 * math.log(2 * math.pi) - log_sigma_rev - 0.5 * (eps ** 2)
        log_p_val = log_p(q_samples)
        
        # KL(q || p) = E_q[log q - log p]
        kl_rev = torch.mean(log_q - log_p_val)
        kl_rev.backward()
        opt_rev.step()
        
    rev_mu_opt = mu_rev.item()
    rev_sigma_opt = torch.exp(log_sigma_rev).item()
    print(f"\nReverse KL (Mode-Dropping / Zero-Forcing):")
    print(f"  Optimal mu: {rev_mu_opt:.3f} (Collapsed to single mode, theoretical: +5.0 or -5.0)")
    print(f"  Optimal sigma: {rev_sigma_opt:.3f} (Theoretical: 1.000)")
    
    assert abs(abs(rev_mu_opt) - 5.0) < 0.2, f"Reverse KL mu expected ~5.0, got {rev_mu_opt}"
    assert abs(rev_sigma_opt - 1.0) < 0.2, f"Reverse KL sigma expected ~1.0, got {rev_sigma_opt}"
    print("✓ Forward KL vs Reverse KL verified successfully!\n")


if __name__ == "__main__":
    test_part5_change_of_variables()
    test_forward_vs_reverse_kl()
    print("=========================================================")
    print("ALL MODULE 10 CHAPTER 10.1 VERIFICATIONS PASSED (100%)")
    print("=========================================================")
