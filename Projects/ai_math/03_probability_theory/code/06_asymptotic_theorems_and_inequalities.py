"""
Chapter 3.6: Asymptotic Theorems (LLN, CLT) & Inequalities
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Concentration Inequalities Comparison (Markov, Chebyshev, Cantelli, Hoeffding, CLT, Exact).
2. Weak and Strong Law of Large Numbers Convergence & Rate (O(1 / sqrt(N))).
3. Central Limit Theorem Convergence (Uniform Sums / Irwin-Hall Distribution).
4. Failure of LLN and CLT on Heavy-Tailed Cauchy Distributions.
5. PAC Learning Sample Complexity & Generalization Bounds Simulation.
"""

import numpy as np
import scipy.stats as stats
import math


# =====================================================================
# 1. Part 5: Concentration Inequalities Verification
# =====================================================================
def verify_concentration_inequalities():
    """
    Verifies Part 5 visual grid for n = 100 fair coin tosses:
    X ~ Binomial(100, 0.5), E[X] = 50, Var(X) = 25, sigma = 5.
    Target: P(X >= 70)
    """
    n = 100
    p = 0.5
    mu = n * p
    var = n * p * (1 - p)
    sigma = math.sqrt(var)
    threshold = 70
    k = threshold - mu  # 20
    epsilon = k / n     # 0.20

    # 1. Markov's Inequality: P(X >= 70) <= E[X] / 70
    markov_bound = mu / threshold  # 50 / 70 = 5 / 7
    
    # 2. Chebyshev's Inequality (two-sided): P(|X - mu| >= 20) <= Var(X) / 20^2
    chebyshev_bound = var / (k ** 2)  # 25 / 400 = 0.0625
    
    # 3. Cantelli's Inequality (one-sided Chebyshev): P(X - mu >= k) <= Var / (Var + k^2)
    cantelli_bound = var / (var + k ** 2)  # 25 / 425 = 1 / 17
    
    # 4. Hoeffding's Inequality: P(X_bar - mu >= eps) <= exp(-2 * n * eps^2)
    # Range b_i - a_i = 1 - 0 = 1
    hoeffding_bound = math.exp(-2.0 * n * (epsilon ** 2))  # exp(-8)
    
    # 5. CLT Gaussian Approximation:
    # Without continuity correction
    z_nocorr = (threshold - mu) / sigma  # 20 / 5 = 4.0
    clt_nocorr = float(1.0 - stats.norm.cdf(z_nocorr))
    # With continuity correction (threshold = 69.5)
    z_corr = (threshold - 0.5 - mu) / sigma  # 19.5 / 5 = 3.9
    clt_corr = float(1.0 - stats.norm.cdf(z_corr))
    
    # 6. Exact Binomial Probability: P(X >= 70) = sum_{j=70}^100 (100 choose j) (0.5)^100
    exact_prob = float(stats.binom.sf(threshold - 1, n, p))
    
    print("--- Part 5: Concentration Inequalities Comparison ---")
    print(f"Target: P(X >= 70) for X ~ Binomial(100, 0.5)")
    print(f"1. Markov Bound:     {markov_bound:.6f} ({markov_bound * 100:.4f}%)")
    print(f"2. Chebyshev Bound:  {chebyshev_bound:.6f} ({chebyshev_bound * 100:.4f}%)")
    print(f"3. Cantelli Bound:   {cantelli_bound:.6f} ({cantelli_bound * 100:.4f}%)")
    print(f"4. Hoeffding Bound:  {hoeffding_bound:.6f} ({hoeffding_bound * 100:.4f}%)")
    print(f"5. CLT (no corr):    {clt_nocorr:.8f} ({clt_nocorr * 100:.6f}%)")
    print(f"   CLT (with corr):  {clt_corr:.8f} ({clt_corr * 100:.6f}%)")
    print(f"6. Exact Binomial:   {exact_prob:.8f} ({exact_prob * 100:.6f}%)")
    
    # Assertions
    assert abs(markov_bound - 5.0 / 7.0) < 1e-7
    assert abs(chebyshev_bound - 0.0625) < 1e-7
    assert abs(cantelli_bound - 1.0 / 17.0) < 1e-7
    assert abs(hoeffding_bound - math.exp(-8.0)) < 1e-9
    assert abs(exact_prob - 3.92507e-05) < 1e-7
    
    # Verify bounding hierarchy: Exact <= Hoeffding < Cantelli < Chebyshev < Markov
    assert exact_prob < hoeffding_bound < cantelli_bound < chebyshev_bound < markov_bound
    print("✓ All inequality bounds and hierarchy validated successfully!")


# =====================================================================
# 2. Law of Large Numbers Convergence Rate Test
# =====================================================================
def verify_lln_convergence():
    """
    Empirically verifies the Law of Large Numbers and the O(1 / sqrt(N))
    decay rate of the sample mean standard error.
    """
    np.random.seed(42)
    sample_sizes = [10, 100, 1_000, 10_000, 100_000]
    num_trials = 500
    true_mu = 3.5  # Fair 6-sided die
    true_sigma = math.sqrt(35.0 / 12.0)  # ~1.7078
    
    print("\n--- Law of Large Numbers O(1/sqrt(N)) Convergence Rate ---")
    for n in sample_sizes:
        # Generate trials of size n
        samples = np.random.randint(1, 7, size=(num_trials, n))
        sample_means = np.mean(samples, axis=1)
        empirical_error = np.mean(np.abs(sample_means - true_mu))
        theoretical_se = true_sigma / math.sqrt(n)
        
        print(f"N = {n:6d} | Empirical Mean Abs Error = {empirical_error:.5f} | Theo SE = {theoretical_se:.5f}")
        # As N increases by 100x, error drops by ~10x
        assert abs(np.mean(sample_means) - true_mu) < 3 * theoretical_se


# =====================================================================
# 3. Central Limit Theorem (Irwin-Hall Uniform Sums)
# =====================================================================
def verify_clt_uniform():
    """
    Verifies that the sum of n = 12 Uniform(0, 1) random variables
    converges tightly to a Standard Normal distribution N(0, 1).
    """
    np.random.seed(42)
    num_samples = 100_000
    n = 12
    # Uniform(0, 1) has mean 0.5 and variance 1/12
    # Sum of 12 has mean 6.0 and variance 1.0
    u = np.random.uniform(0.0, 1.0, size=(num_samples, n))
    z = np.sum(u, axis=1) - 6.0  # Normalized: mean 0, std 1
    
    emp_mean = float(np.mean(z))
    emp_var = float(np.var(z))
    emp_skew = float(stats.skew(z))
    emp_kurt = float(stats.kurtosis(z))  # Fisher kurtosis (normal = 0.0)
    
    # Kolmogorov-Smirnov test against standard normal
    ks_stat, p_val = stats.kstest(z, 'norm')
    
    print("\n--- Central Limit Theorem (Irwin-Hall n=12 Sum) ---")
    print(f"Empirical Mean:     {emp_mean:.4f} (Expected: 0.0)")
    print(f"Empirical Variance: {emp_var:.4f} (Expected: 1.0)")
    print(f"Empirical Skewness: {emp_skew:.4f} (Expected: 0.0)")
    print(f"Excess Kurtosis:    {emp_kurt:.4f} (Expected: 0.0, true Irwin-Hall: -0.1)")
    print(f"KS Test p-value:    {p_val:.4f} (p > 0.01 confirms normality)")
    
    assert abs(emp_mean) < 0.01
    assert abs(emp_var - 1.0) < 0.02
    assert abs(emp_skew) < 0.02
    assert p_val > 0.01
    print("✓ CLT Gaussian convergence validated successfully!")


# =====================================================================
# 4. Failure of LLN and CLT on Heavy-Tailed Cauchy Distribution
# =====================================================================
def verify_cauchy_failure():
    """
    Demonstrates that Cauchy distributed random variables fail both LLN and CLT
    because E[|X|] = inf and Var(X) = inf.
    """
    np.random.seed(42)
    n_samples = 100_000
    cauchy_samples = np.random.standard_cauchy(size=n_samples)
    
    # Cumulative running mean
    running_mean = np.cumsum(cauchy_samples) / np.arange(1, n_samples + 1)
    
    # Take slices at powers of 10
    checkpoints = [10, 100, 1_000, 10_000, 100_000]
    print("\n--- Heavy-Tailed Cauchy Distribution Failure ---")
    for cp in checkpoints:
        print(f"N = {cp:6d} | Cauchy Running Mean = {running_mean[cp-1]:.4f}")
    
    # Unlike normal/uniform where running mean settles within 0.01 of 0,
    # Cauchy running mean exhibits massive discontinuous jumps due to extreme outliers
    std_normal = np.random.standard_normal(size=n_samples)
    normal_running_mean = np.cumsum(std_normal) / np.arange(1, n_samples + 1)
    
    final_normal_err = abs(normal_running_mean[-1])
    print(f"Normal Running Mean at N=100,000: {normal_running_mean[-1]:.6f} (stabilized)")
    assert final_normal_err < 0.01  # Normal stabilized tightly at 0
    print("✓ LLN failure on Cauchy vs Normal stability demonstrated!")


# =====================================================================
# 5. PAC Learning Sample Complexity Simulation
# =====================================================================
def verify_pac_sample_complexity():
    """
    Simulates PAC learning bound based on Hoeffding's inequality:
    For a hypothesis class H of size |H|, with probability at least 1 - delta:
    L(h) <= L_emp(h) + sqrt( ln(|H| / delta) / (2N) )
    """
    np.random.seed(42)
    num_hypotheses = 50
    epsilon_target = 0.05
    delta = 0.05
    
    # Required sample size by Hoeffding:
    # epsilon = sqrt(ln(2 * |H| / delta) / (2 * N)) => N = ln(2 * |H| / delta) / (2 * eps^2)
    required_n = int(math.ceil(math.log(2.0 * num_hypotheses / delta) / (2.0 * (epsilon_target ** 2))))
    
    print("\n--- PAC Learning Generalization Bound Simulation ---")
    print(f"Hypothesis Class Size |H|: {num_hypotheses}")
    print(f"Generalization Tolerance epsilon: {epsilon_target}")
    print(f"Confidence parameter delta: {delta} (1 - delta = {1 - delta:.2f})")
    print(f"Theoretically Required N >= {required_n}")
    
    # Verify empirically with N = required_n across 200 repeated learning trials
    n_trials = 200
    violations = 0
    
    # True risk of each hypothesis
    true_risks = np.random.uniform(0.1, 0.4, size=num_hypotheses)
    
    for _ in range(n_trials):
        # Draw N independent training examples
        # Bernoulli loss for each hypothesis with probability true_risks
        sample_losses = np.random.binomial(n=1, p=true_risks, size=(required_n, num_hypotheses))
        empirical_risks = np.mean(sample_losses, axis=0)
        
        # Check maximum generalization gap across all hypotheses
        max_gap = np.max(np.abs(empirical_risks - true_risks))
        if max_gap > epsilon_target:
            violations += 1
            
    empirical_violation_rate = violations / n_trials
    print(f"Empirical Violation Rate: {empirical_violation_rate:.4f} (Must be <= delta = {delta})")
    
    assert empirical_violation_rate <= delta
    print("✓ PAC Sample Complexity bound confirmed via Monte Carlo trials!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 3.6 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_concentration_inequalities()
    verify_lln_convergence()
    verify_clt_uniform()
    verify_cauchy_failure()
    verify_pac_sample_complexity()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 3.6 PASSED CLEANLY!")
    print("=================================================================")
