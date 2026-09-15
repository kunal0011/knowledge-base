"""
Chapter 3.2: Random Variables (Discrete vs. Continuous), PMF, PDF, CDF
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Triangular Distribution Numerical & Analytical Verification.
2. Numerical Quadrature Verification: Total Area Normalization & Interval Probability.
3. 1D Change of Variables Transformation Verification (Analytical vs. Empirical).
4. Inverse Transform Sampling Engine (Exponential Distribution).
5. PyTorch VAE Reparameterization Trick & Gradient Flow Verification.
"""

import numpy as np
import scipy.integrate as integrate
import torch


# =====================================================================
# 1. Part 5 Triangular Distribution PDF & CDF
# =====================================================================
def triangular_pdf(x: float) -> float:
    """
    Triangular PDF on [0, 2]:
    f(x) = x if 0 <= x <= 1 else 2 - x if 1 < x <= 2 else 0
    """
    if 0.0 <= x <= 1.0:
        return float(x)
    elif 1.0 < x <= 2.0:
        return float(2.0 - x)
    return 0.0


def triangular_cdf(x: float) -> float:
    """
    Analytical CDF:
    F(x) = x^2 / 2 if 0 <= x <= 1 else 2x - x^2/2 - 1 if 1 < x <= 2
    """
    if x < 0.0:
        return 0.0
    elif 0.0 <= x <= 1.0:
        return float(x**2 / 2.0)
    elif 1.0 < x <= 2.0:
        return float(2.0 * x - (x**2) / 2.0 - 1.0)
    return 1.0


# Vectorized versions
v_tri_pdf = np.vectorize(triangular_pdf)
v_tri_cdf = np.vectorize(triangular_cdf)


# =====================================================================
# 2. Inverse Transform Sampling for Triangular Distribution
# =====================================================================
def sample_triangular(num_samples: int = 200_000) -> np.ndarray:
    """
    Inverse CDF of Triangular distribution:
    For u <= 0.5: F(x) = x^2 / 2 = u => x = sqrt(2u)
    For u > 0.5:  F(x) = 2x - x^2/2 - 1 = u => x = 2 - sqrt(2(1 - u))
    """
    np.random.seed(42)
    u = np.random.rand(num_samples)
    samples = np.where(u <= 0.5, np.sqrt(2.0 * u), 2.0 - np.sqrt(2.0 * (1.0 - u)))
    return samples


# =====================================================================
# 3. Inverse Transform Sampling for Exponential Distribution
# =====================================================================
def sample_exponential(rate: float, num_samples: int = 200_000) -> np.ndarray:
    """
    X = -log(U) / rate
    """
    np.random.seed(42)
    u = np.random.rand(num_samples)
    return -np.log(u) / rate


# =====================================================================
# 4. PyTorch VAE Reparameterization Trick Test
# =====================================================================
def vae_reparameterization_test():
    """
    z = mu + sigma * eps, where eps ~ N(0, I).
    Loss: L = 0.5 * sum(z^2)
    dL/dmu = z, dL/dsigma = z * eps
    """
    mu = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([0.5, 1.2, 0.8], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)  # Fixed noise for test

    z = mu + sigma * eps
    loss = 0.5 * torch.sum(z**2)
    loss.backward()

    # Analytical
    z_np = (mu + sigma * eps).detach().numpy()
    eps_np = eps.numpy()
    expected_grad_mu = z_np
    expected_grad_sigma = z_np * eps_np

    match_mu = np.allclose(mu.grad.numpy(), expected_grad_mu)
    match_sigma = np.allclose(sigma.grad.numpy(), expected_grad_sigma)

    return match_mu and match_sigma


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 3.2: Random Variables, PMF, PDF, CDF — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Exact Numerical Verification
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Triangular Distribution Walkthrough]")
    # PDF values
    f_05 = triangular_pdf(0.5)
    f_10 = triangular_pdf(1.0)
    f_15 = triangular_pdf(1.5)

    # CDF values
    F_05 = triangular_cdf(0.5)
    F_10 = triangular_cdf(1.0)
    F_15 = triangular_cdf(1.5)

    # Interval Probability
    p_interval = F_15 - F_05

    print(f"  f_X(0.5):                  {f_05:.6f} (Expected: 0.500000)")
    print(f"  F_X(0.5):                  {F_05:.6f} (Expected: 0.125000)")
    print(f"  f_X(1.0):                  {f_10:.6f} (Expected: 1.000000)")
    print(f"  F_X(1.0):                  {F_10:.6f} (Expected: 0.500000)")
    print(f"  f_X(1.5):                  {f_15:.6f} (Expected: 0.500000)")
    print(f"  F_X(1.5):                  {F_15:.6f} (Expected: 0.875000)")
    print(f"  P(0.5 < X <= 1.5):         {p_interval:.6f} (Expected: 0.750000)")

    assert np.isclose(f_05, 0.5)
    assert np.isclose(F_05, 0.125)
    assert np.isclose(f_10, 1.0)
    assert np.isclose(F_10, 0.5)
    assert np.isclose(f_15, 0.5)
    assert np.isclose(F_15, 0.875)
    assert np.isclose(p_interval, 0.75)
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: Numerical Integration & Normalization Check
    # -------------------------------------------------------------
    print("\n[Test 2: Numerical Quadrature Normalization Check]")
    total_area, _ = integrate.quad(triangular_pdf, 0.0, 2.0)
    print(f"  Total Integrated Area:     {total_area:.8f} (Expected: 1.00000000)")
    assert np.isclose(total_area, 1.0, atol=1e-12)

    # Integral between 0.5 and 1.5
    quad_interval, _ = integrate.quad(triangular_pdf, 0.5, 1.5)
    print(f"  Integrated Interval Area:  {quad_interval:.8f} (Expected: 0.75000000)")
    assert np.isclose(quad_interval, 0.75, atol=1e-12)
    print("  -> Integral normalization and interval probability confirmed!")

    # -------------------------------------------------------------
    # Test 3: Change of Variables Verification (Y = 2X + 1)
    # -------------------------------------------------------------
    print("\n[Test 3: 1D Change of Variables Verification]")
    # Y = 2X + 1. For y = 3.0 => x = 1.0 => f_Y(3.0) = f_X(1.0) * |dx/dy| = 1.0 * 0.5 = 0.5
    samples_x = sample_triangular(num_samples=500_000)
    samples_y = 2.0 * samples_x + 1.0

    # Empirical histogram estimate around y = 3.0
    delta = 0.05
    count_in_window = np.sum((samples_y >= 3.0 - delta) & (samples_y <= 3.0 + delta))
    empirical_density_y3 = count_in_window / (len(samples_y) * (2.0 * delta))

    analytical_density_y3 = triangular_pdf((3.0 - 1.0) / 2.0) * 0.5  # 1.0 * 0.5 = 0.5
    print(f"  Analytical f_Y(3.0):       {analytical_density_y3:.6f}")
    print(f"  Empirical Density f_Y(3.0):{empirical_density_y3:.6f}")
    assert np.isclose(empirical_density_y3, analytical_density_y3, atol=0.01)
    print("  -> Change of variables density formula verified empirically!")

    # -------------------------------------------------------------
    # Test 4: Inverse Transform Sampling for Exponential
    # -------------------------------------------------------------
    print("\n[Test 4: Inverse Transform Sampling (Exponential Distribution)]")
    rate = 2.5
    samples_exp = sample_exponential(rate=rate, num_samples=300_000)

    # Theoretical mean: 1/rate = 1/2.5 = 0.4
    # Theoretical variance: 1/rate^2 = 1/6.25 = 0.16
    emp_mean = np.mean(samples_exp)
    emp_var = np.var(samples_exp)

    print(f"  Rate lambda:               {rate}")
    print(f"  Theoretical Mean:          {1.0 / rate:.4f}  | Empirical: {emp_mean:.4f}")
    print(f"  Theoretical Variance:      {1.0 / (rate**2):.4f}  | Empirical: {emp_var:.4f}")

    assert np.isclose(emp_mean, 1.0 / rate, atol=0.01)
    assert np.isclose(emp_var, 1.0 / (rate**2), atol=0.01)
    print("  -> Inverse Transform Sampling matches theoretical moments!")

    # -------------------------------------------------------------
    # Test 5: PyTorch VAE Reparameterization Trick Test
    # -------------------------------------------------------------
    print("\n[Test 5: PyTorch VAE Reparameterization Trick Gradient Flow]")
    vae_passed = vae_reparameterization_test()
    print(f"  Analytical gradients match PyTorch Autograd? {vae_passed}")
    assert vae_passed, "Reparameterization trick gradient mismatch!"
    print("  -> VAE Reparameterization trick verified!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 3.2 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
