"""
Chapter 3.4: Parametric Distributions (Bernoulli, Categorical, Gaussian, Beta, Dirichlet)
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Bivariate Gaussian PDF & Log-Likelihood Evaluation.
2. SciPy & PyTorch Cross-Verification of Multivariate Normal Calculations.
3. Beta-Binomial Bayesian Updating Numerical Verification.
4. Dirichlet Distribution Simplex Concentration & Sparsity Analysis.
5. Gaussian Conditioning via Schur Complement (Analytical vs. Empirical Slicing).
"""

import numpy as np
import scipy.stats as stats
import torch


# =====================================================================
# 1. Part 5 Multivariate Gaussian Walkthrough
# =====================================================================
def evaluate_bivariate_gaussian(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
    """
    Computes PDF and log-likelihood analytically for a 2D Gaussian.
    """
    D = len(x)
    delta_x = x - mu
    det_Sigma = float(np.linalg.det(Sigma))
    inv_Sigma = np.linalg.inv(Sigma)

    # Mahalanobis distance
    mahalanobis_sq = float(delta_x.T @ inv_Sigma @ delta_x)

    # Normalizer
    norm_const = ((2.0 * np.pi) ** (D / 2.0)) * np.sqrt(det_Sigma)

    # PDF & Log-Likelihood
    pdf_val = float(np.exp(-0.5 * mahalanobis_sq) / norm_const)
    log_lik = float(-0.5 * mahalanobis_sq - np.log(norm_const))

    return {
        "delta_x": delta_x,
        "det_Sigma": det_Sigma,
        "inv_Sigma": inv_Sigma,
        "mahalanobis_sq": mahalanobis_sq,
        "norm_const": norm_const,
        "pdf_val": pdf_val,
        "log_lik": log_lik,
    }


# =====================================================================
# 2. Gaussian Conditioning via Schur Complement
# =====================================================================
def gaussian_conditioning(
    mu: np.ndarray, Sigma: np.ndarray, x2_observed: float, idx1: int = 0, idx2: int = 1
):
    """
    Computes conditional distribution X1 | X2 = x2_observed:
    mu_{1|2} = mu_1 + Sigma_12 * Sigma_22^-1 * (x2 - mu_2)
    Sigma_{1|2} = Sigma_11 - Sigma_12 * Sigma_22^-1 * Sigma_21
    """
    mu1 = mu[idx1]
    mu2 = mu[idx2]
    s11 = Sigma[idx1, idx1]
    s12 = Sigma[idx1, idx2]
    s21 = Sigma[idx2, idx1]
    s22 = Sigma[idx2, idx2]

    cond_mean = mu1 + (s12 / s22) * (x2_observed - mu2)
    cond_var = s11 - (s12 * s21) / s22

    return float(cond_mean), float(cond_var)


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 3.4: Parametric Distributions — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Bivariate Gaussian
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Bivariate Gaussian Walkthrough]")
    mu_0 = np.array([0.0, 1.0], dtype=np.float64)
    Sigma_0 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    x_test = np.array([1.0, 2.0], dtype=np.float64)

    res = evaluate_bivariate_gaussian(x_test, mu_0, Sigma_0)

    print(f"  Centered Difference Delta x: {res['delta_x']} (Expected: [1.0, 1.0])")
    print(f"  det(Sigma):                 {res['det_Sigma']:.4f} (Expected: 3.0000)")
    print(f"  Mahalanobis Distance Delta^2: {res['mahalanobis_sq']:.6f} (Expected: ~0.666667 = 2/3)")
    print(f"  Normalizer Z_norm:          {res['norm_const']:.6f} (Expected: ~10.882796 = 2*pi*sqrt(3))")
    print(f"  PDF Value p(x):             {res['pdf_val']:.6f} (Expected: ~0.065841)")
    print(f"  Log-Likelihood log p(x):    {res['log_lik']:.6f} (Expected: ~-2.720481)")

    assert np.allclose(res["delta_x"], [1.0, 1.0])
    assert np.isclose(res["det_Sigma"], 3.0)
    assert np.isclose(res["mahalanobis_sq"], 2.0 / 3.0)
    assert np.isclose(res["norm_const"], 2.0 * np.pi * np.sqrt(3.0))
    assert np.isclose(res["pdf_val"], np.exp(-1.0 / 3.0) / (2.0 * np.pi * np.sqrt(3.0)))
    assert np.isclose(res["log_lik"], -1.0 / 3.0 - np.log(2.0 * np.pi * np.sqrt(3.0)))
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: SciPy & PyTorch Cross-Verification
    # -------------------------------------------------------------
    print("\n[Test 2: SciPy & PyTorch Cross-Verification]")
    # SciPy
    scipy_pdf = float(stats.multivariate_normal.pdf(x_test, mean=mu_0, cov=Sigma_0))
    scipy_logpdf = float(stats.multivariate_normal.logpdf(x_test, mean=mu_0, cov=Sigma_0))

    # PyTorch
    pt_dist = torch.distributions.MultivariateNormal(
        loc=torch.tensor(mu_0), covariance_matrix=torch.tensor(Sigma_0)
    )
    pt_logpdf = float(pt_dist.log_prob(torch.tensor(x_test)).item())
    pt_pdf = float(torch.exp(torch.tensor(pt_logpdf)).item())

    print(f"  Hand-Derived PDF:  {res['pdf_val']:.8f}")
    print(f"  SciPy PDF:         {scipy_pdf:.8f}")
    print(f"  PyTorch PDF:       {pt_pdf:.8f}")
    print(f"  Hand-Derived LogP: {res['log_lik']:.8f}")
    print(f"  SciPy LogP:        {scipy_logpdf:.8f}")
    print(f"  PyTorch LogP:      {pt_logpdf:.8f}")

    assert np.isclose(scipy_pdf, res["pdf_val"])
    assert np.isclose(pt_pdf, res["pdf_val"])
    assert np.isclose(scipy_logpdf, res["log_lik"])
    assert np.isclose(pt_logpdf, res["log_lik"])
    print("  -> Exact agreement across Hand, SciPy, and PyTorch implementations!")

    # -------------------------------------------------------------
    # Test 3: Beta-Binomial Conjugacy Test
    # -------------------------------------------------------------
    print("\n[Test 3: Beta-Binomial Conjugacy Updating]")
    # Prior: Beta(10, 10) -> mean = 10 / 20 = 0.50
    alpha_prior, beta_prior = 10.0, 10.0
    # Data: 100 trials, 25 successes
    k_success, n_trials = 25, 100

    # Posterior
    alpha_post = alpha_prior + k_success  # 35
    beta_post = beta_prior + (n_trials - k_success)  # 85

    expected_mean = 35.0 / 120.0
    expected_var = (35.0 * 85.0) / ((120.0**2) * 121.0)

    beta_dist = stats.beta(alpha_post, beta_post)
    print(f"  Posterior Alpha:           {alpha_post} (Expected: 35)")
    print(f"  Posterior Beta:            {beta_post} (Expected: 85)")
    print(f"  Analytical Posterior Mean: {expected_mean:.6f} (Expected: ~0.291667)")
    print(f"  SciPy Beta Mean:           {beta_dist.mean():.6f}")
    print(f"  SciPy Beta Variance:       {beta_dist.var():.6f}")

    assert np.isclose(beta_dist.mean(), expected_mean)
    assert np.isclose(beta_dist.var(), expected_var)
    print("  -> Beta-Binomial conjugacy parameters confirmed!")

    # -------------------------------------------------------------
    # Test 4: Dirichlet Simplex Sparsity vs Uniformity
    # -------------------------------------------------------------
    print("\n[Test 4: Dirichlet Simplex Concentration Analysis]")
    np.random.seed(42)
    # Case A: alpha = [0.1, 0.1, 0.1] -> Sparse (near vertices, max prob ~ 1.0)
    samples_sparse = stats.dirichlet([0.1, 0.1, 0.1]).rvs(size=10_000)
    max_components = np.max(samples_sparse, axis=1)
    mean_max_prob = float(np.mean(max_components))

    # Case B: alpha = [10.0, 10.0, 10.0] -> Dense (near center, all ~ 0.33)
    samples_dense = stats.dirichlet([10.0, 10.0, 10.0]).rvs(size=10_000)
    dist_from_center = np.mean(np.abs(samples_dense - 1.0 / 3.0))

    print(f"  alpha=0.1: Mean Max Probability (Sparsity): {mean_max_prob:.4f} (Expected: > 0.90)")
    print(f"  alpha=10 : Mean Distance from Center (1/3):  {dist_from_center:.4f} (Expected: < 0.08)")

    assert mean_max_prob > 0.85, "alpha < 1 must produce sparse one-hot distributions!"
    assert dist_from_center < 0.10, "alpha >> 1 must concentrate around the center!"
    print("  -> Dirichlet concentration properties mathematically validated!")

    # -------------------------------------------------------------
    # Test 5: Gaussian Conditioning via Schur Complement
    # -------------------------------------------------------------
    print("\n[Test 5: Gaussian Conditioning & Schur Complement]")
    # Distribution from Test 1: mu = [0, 1], Sigma = [[2, 1], [1, 2]]
    # Condition on X2 = 2.0
    x2_obs = 2.0
    cond_mean_ana, cond_var_ana = gaussian_conditioning(mu_0, Sigma_0, x2_obs)

    # Analytical derivation:
    # mu_{1|2} = mu_1 + Sigma_12 * Sigma_22^-1 * (x2 - mu_2) = 0 + (1/2)*(2 - 1) = +0.5
    # Sigma_{1|2} = Sigma_11 - Sigma_12^2 / Sigma_22 = 2 - 1/2 = 1.5
    print(f"  Analytical Conditional Mean mu_{{1|2}}: {cond_mean_ana:.4f} (Expected: 0.5000)")
    print(f"  Analytical Conditional Var Sigma_{{1|2}}:{cond_var_ana:.4f} (Expected: 1.5000)")

    assert np.isclose(cond_mean_ana, 0.5)
    assert np.isclose(cond_var_ana, 1.5)

    # Empirical slice verification: sample 1,000,000 points and filter X2 ~= 2.0
    samples_joint = np.random.multivariate_normal(mu_0, Sigma_0, size=1_000_000)
    slice_mask = np.abs(samples_joint[:, 1] - x2_obs) < 0.05
    empirical_cond_samples = samples_joint[slice_mask, 0]

    emp_cond_mean = float(np.mean(empirical_cond_samples))
    emp_cond_var = float(np.var(empirical_cond_samples))
    print(f"  Empirical Slice Mean:                  {emp_cond_mean:.4f}")
    print(f"  Empirical Slice Var:                   {emp_cond_var:.4f}")

    assert np.isclose(emp_cond_mean, cond_mean_ana, atol=0.03)
    assert np.isclose(emp_cond_var, cond_var_ana, atol=0.05)
    print("  -> Schur complement Gaussian conditioning confirmed by Monte Carlo slicing!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 3.4 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
