"""
Chapter 4.1: Statistical Inference Foundations & Sampling
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Manual Calculations & Student's t CI Verification.
2. Bessel's Correction Monte Carlo Simulation (Biased vs. Unbiased Variance).
3. Cochran's Theorem Independence Verification (Gaussian vs. Exponential).
4. Glivenko-Cantelli Theorem (Empirical Distribution Function Uniform Convergence).
5. Student's t Confidence Interval Empirical Coverage Rate.
"""

import numpy as np
import scipy.stats as stats
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Verification
# =====================================================================
def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 4.1 Part 5:
    x = [2.40, 3.10, 1.80, 2.90, 2.80]
    """
    x = np.array([2.40, 3.10, 1.80, 2.90, 2.80], dtype=np.float64)
    n = len(x)

    # 1. Sample Mean
    mean_val = float(np.mean(x))
    
    # 2. Deviations & Sum of Squares
    deviations = x - mean_val
    sum_devs = float(np.sum(deviations))
    sq_deviations = deviations ** 2
    ss = float(np.sum(sq_deviations))

    # 3. Variances
    var_biased = float(np.var(x, ddof=0))  # S_n^2
    var_unbiased = float(np.var(x, ddof=1))  # S_{n-1}^2
    std_unbiased = float(np.std(x, ddof=1))

    # 4. Standard Error of the Mean
    se = std_unbiased / math.sqrt(n)

    # 5. 95% Student's t Confidence Interval
    # df = n - 1 = 4, alpha = 0.05 => 0.975 quantile
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    margin_of_error = t_crit * se
    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error

    print("--- Part 5: Visual Grid Verification ---")
    print(f"Sample Vector:               {x}")
    print(f"Sample Mean:                 {mean_val:.4f} (Expected: 2.6000)")
    print(f"Sum of Linear Deviations:    {sum_devs:.6f} (Expected: 0.0000)")
    print(f"Sum of Squared Deviations:   {ss:.4f} (Expected: 1.0600)")
    print(f"Biased Variance S_n^2:       {var_biased:.4f} (Expected: 0.2120)")
    print(f"Unbiased Variance S^2:       {var_unbiased:.4f} (Expected: 0.2650)")
    print(f"Sample Std Dev S:            {std_unbiased:.5f} (Expected: 0.51478)")
    print(f"Standard Error SE:           {se:.5f} (Expected: 0.23022)")
    print(f"Student's t Critical (df=4): {t_crit:.4f} (Expected: 2.7764)")
    print(f"95% Confidence Interval:     [{ci_lower:.4f}, {ci_upper:.4f}] (Expected: [1.9609, 3.2391])")

    # Assertions
    assert abs(mean_val - 2.6000) < 1e-7
    assert abs(sum_devs) < 1e-7
    assert abs(ss - 1.0600) < 1e-7
    assert abs(var_biased - 0.2120) < 1e-7
    assert abs(var_unbiased - 0.2650) < 1e-7
    assert abs(std_unbiased - 0.514781) < 1e-5
    assert abs(se - 0.230217) < 1e-5
    assert abs(ci_lower - 1.9609) < 1e-3
    assert abs(ci_upper - 3.2391) < 1e-3
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Bessel's Correction Monte Carlo Simulation
# =====================================================================
def verify_bessels_correction(num_trials: int = 100_000):
    """
    Demonstrates that E[S_n^2] = ((N - 1) / N) * sigma^2,
    whereas E[S_{N-1}^2] = sigma^2 exactly.
    """
    np.random.seed(42)
    n = 5
    true_sigma2 = 4.0
    true_std = math.sqrt(true_sigma2)
    
    # Draw (num_trials, n) samples from N(0, sigma^2)
    samples = np.random.normal(loc=10.0, scale=true_std, size=(num_trials, n))
    
    # Biased variance (ddof=0) vs Unbiased variance (ddof=1)
    s_n_sq = np.var(samples, axis=1, ddof=0)
    s_unbiased_sq = np.var(samples, axis=1, ddof=1)
    
    mean_biased = float(np.mean(s_n_sq))
    mean_unbiased = float(np.mean(s_unbiased_sq))
    theoretical_biased = ((n - 1) / n) * true_sigma2  # (4 / 5) * 4.0 = 3.20
    
    print("\n--- Bessel's Correction Monte Carlo Simulation ---")
    print(f"True Population Variance sigma^2: {true_sigma2:.4f}")
    print(f"Sample Size N:                    {n}")
    print(f"Empirical Mean of S_n^2:          {mean_biased:.4f} (Theo: {theoretical_biased:.4f})")
    print(f"Empirical Mean of S^2:            {mean_unbiased:.4f} (Theo: {true_sigma2:.4f})")
    
    assert abs(mean_biased - theoretical_biased) < 0.02
    assert abs(mean_unbiased - true_sigma2) < 0.02
    print("✓ Bessel's correction unbiasedness verified across 100,000 trials!")


# =====================================================================
# 3. Cochran's Theorem Independence Test
# =====================================================================
def verify_cochrans_theorem():
    """
    Verifies that for Gaussian data, the sample mean X_bar and sample variance S^2
    are statistically independent (zero correlation and zero mutual dependence),
    whereas for non-Gaussian data (Exponential), they are strongly correlated.
    """
    np.random.seed(42)
    num_trials = 50_000
    n = 10

    # Case A: Gaussian samples
    gauss_samples = np.random.normal(loc=0.0, scale=1.0, size=(num_trials, n))
    gauss_means = np.mean(gauss_samples, axis=1)
    gauss_vars = np.var(gauss_samples, axis=1, ddof=1)
    gauss_corr = float(np.corrcoef(gauss_means, gauss_vars)[0, 1])

    # Case B: Exponential samples
    exp_samples = np.random.exponential(scale=2.0, size=(num_trials, n))
    exp_means = np.mean(exp_samples, axis=1)
    exp_vars = np.var(exp_samples, axis=1, ddof=1)
    exp_corr = float(np.corrcoef(exp_means, exp_vars)[0, 1])

    print("\n--- Cochran's Theorem Independence Verification ---")
    print(f"Gaussian Sample Corr(X_bar, S^2):    {gauss_corr:.5f} (Expected: ~0.00000)")
    print(f"Exponential Sample Corr(X_bar, S^2): {exp_corr:.5f} (Expected: > 0.60000)")

    assert abs(gauss_corr) < 0.015, "Sample mean and variance must be uncorrelated for Gaussian"
    assert exp_corr > 0.50, "Sample mean and variance are dependent for Exponential"
    print("✓ Cochran's theorem Gaussian independence confirmed!")


# =====================================================================
# 4. Glivenko-Cantelli Theorem (EDF Uniform Convergence)
# =====================================================================
def verify_glivenko_cantelli():
    """
    Tracks the supremum distance ||F_hat_N - F||_inf across increasing sample sizes N
    to verify Glivenko-Cantelli uniform convergence to 0.
    """
    np.random.seed(42)
    sample_sizes = [10, 100, 1_000, 10_000, 100_000]
    
    print("\n--- Glivenko-Cantelli Theorem (EDF Convergence) ---")
    prev_ks = float('inf')
    for n in sample_sizes:
        samples = np.random.normal(loc=0.0, scale=1.0, size=n)
        # scipy kstest computes sup |F_hat_N(x) - F(x)|
        ks_stat, _ = stats.kstest(samples, 'norm')
        theo_rate = 1.0 / math.sqrt(n)
        print(f"N = {n:6d} | Sup Distance ||F_hat - F||_inf = {ks_stat:.5f} | Theo O(1/sqrt(N)) = {theo_rate:.5f}")
        assert ks_stat < 3.0 * theo_rate


# =====================================================================
# 5. Student's t Confidence Interval Coverage Probability
# =====================================================================
def verify_confidence_interval_coverage(num_trials: int = 20_000):
    """
    Verifies that a 95% Student's t confidence interval contains the true
    population mean mu in exactly ~95% of repeated samples.
    """
    np.random.seed(42)
    n = 5
    true_mu = 5.0
    true_sigma = 2.0
    alpha = 0.05
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    
    samples = np.random.normal(loc=true_mu, scale=true_sigma, size=(num_trials, n))
    sample_means = np.mean(samples, axis=1)
    sample_stds = np.std(samples, axis=1, ddof=1)
    se_vals = sample_stds / math.sqrt(n)
    
    me = t_crit * se_vals
    lower_bounds = sample_means - me
    upper_bounds = sample_means + me
    
    # Check coverage: true_mu is inside [lower, upper]
    covered = (lower_bounds <= true_mu) & (true_mu <= upper_bounds)
    empirical_coverage = float(np.mean(covered))
    
    print("\n--- Student's t 95% Confidence Interval Coverage Test ---")
    print(f"Number of Simulated Trials: {num_trials}")
    print(f"Sample Size per Trial N:    {n}")
    print(f"Nominal Target Coverage:    0.9500 (95.0%)")
    print(f"Empirical Coverage Rate:    {empirical_coverage:.4f} ({empirical_coverage * 100:.2f}%)")
    
    assert abs(empirical_coverage - 0.9500) < 0.005
    print("✓ Student's t confidence interval exact coverage confirmed!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 4.1 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_bessels_correction()
    verify_cochrans_theorem()
    verify_glivenko_cantelli()
    verify_confidence_interval_coverage()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 4.1 PASSED CLEANLY!")
    print("=================================================================")
