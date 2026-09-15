"""
Chapter 3.7: Information Theory Foundations
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Verification (Entropy, Cross-Entropy, KL, JSD).
2. Gibbs' Inequality Verification via Random Dirichlet Simplex Sampling.
3. Mutual Information, Joint & Conditional Entropy Identities.
4. VAE Analytical Gaussian KL Divergence vs. Monte Carlo Integration.
5. Softmax Cross-Entropy Loss Analytical Gradient vs. PyTorch Autograd.
"""

import numpy as np
import scipy.stats as stats
import torch
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Verification
# =====================================================================
def shannon_entropy(p: np.ndarray) -> float:
    """Computes Shannon entropy H(P) in nats with 0 log 0 = 0."""
    p_safe = np.where(p > 0, p, 1.0)
    return float(-np.sum(np.where(p > 0, p * np.log(p_safe), 0.0)))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Computes Cross-Entropy H(P, Q) in nats."""
    assert np.all(q > 0), "Q must have support everywhere P > 0"
    return float(-np.sum(p * np.log(q)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes KL Divergence D_KL(P || Q) in nats."""
    assert np.all(q > 0), "Q must have non-zero support"
    p_safe = np.where(p > 0, p, 1.0)
    return float(np.sum(np.where(p > 0, p * np.log(p_safe / q), 0.0)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes Jensen-Shannon Divergence JSD(P || Q) in nats."""
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 3.7 Part 5:
    P = [0.60, 0.30, 0.10]
    Q_A = [0.50, 0.35, 0.15]
    Q_B = [0.10, 0.20, 0.70]
    """
    p = np.array([0.60, 0.30, 0.10], dtype=np.float64)
    q_a = np.array([0.50, 0.35, 0.15], dtype=np.float64)
    q_b = np.array([0.10, 0.20, 0.70], dtype=np.float64)

    h_p = shannon_entropy(p)
    h_p_qa = cross_entropy(p, q_a)
    h_p_qb = cross_entropy(p, q_b)

    kl_qa = kl_divergence(p, q_a)
    kl_qb = kl_divergence(p, q_b)

    jsd_qa = js_divergence(p, q_a)

    print("--- Part 5: Visual Grid Verification ---")
    print(f"H(P):         {h_p:.6f} nats (Expected: ~0.897946)")
    print(f"H(P, Q_A):    {h_p_qa:.6f} nats (Expected: ~0.920547)")
    print(f"H(P, Q_B):    {h_p_qb:.6f} nats (Expected: ~1.900049)")
    print(f"KL(P || Q_A): {kl_qa:.6f} nats (Expected: ~0.022601)")
    print(f"KL(P || Q_B): {kl_qb:.6f} nats (Expected: ~1.002103)")
    print(f"JSD(P || Q_A):{jsd_qa:.6f} nats (Expected: ~0.005756)")

    # Assertions for Part 5 arithmetic
    assert abs(h_p - 0.897946) < 1e-5
    assert abs(h_p_qa - 0.920547) < 1e-5
    assert abs(h_p_qb - 1.900049) < 1e-5
    assert abs(kl_qa - 0.022601) < 1e-5
    assert abs(kl_qb - 1.002103) < 1e-5
    assert abs(jsd_qa - 0.005756) < 1e-5

    # Fundamental identity: H(P, Q) = H(P) + KL(P || Q)
    assert abs(h_p_qa - (h_p + kl_qa)) < 1e-9
    assert abs(h_p_qb - (h_p + kl_qb)) < 1e-9
    print("✓ Part 5 visual grid and fundamental cross-entropy identity validated!")


# =====================================================================
# 2. Gibbs' Inequality Verification
# =====================================================================
def verify_gibbs_inequality(num_trials: int = 5000):
    """
    Verifies that D_KL(P || Q) >= 0 with equality iff P == Q
    across thousands of randomly sampled categorical distributions on Dirichlet(1, 1, 1, 1).
    """
    np.random.seed(42)
    k = 4
    alpha = np.ones(k)
    p_samples = np.random.dirichlet(alpha, size=num_trials)
    q_samples = np.random.dirichlet(alpha, size=num_trials)

    # Check non-negativity
    kl_values = np.sum(p_samples * np.log(p_samples / q_samples), axis=1)
    min_kl = np.min(kl_values)
    
    # Check identity on self
    self_kl_values = np.sum(p_samples * np.log(p_samples / p_samples), axis=1)
    max_self_kl = np.max(np.abs(self_kl_values))

    print("\n--- Gibbs' Inequality Monte Carlo Verification ---")
    print(f"Trials Tested: {num_trials}")
    print(f"Minimum KL(P || Q): {min_kl:.8f} (Must be >= 0)")
    print(f"Maximum |KL(P || P)|: {max_self_kl:.8e} (Must be == 0)")

    assert min_kl >= 0.0
    assert max_self_kl < 1e-12
    print("✓ Gibbs' inequality strictly verified!")


# =====================================================================
# 3. Mutual Information & Information Identities
# =====================================================================
def verify_mutual_information():
    """
    Verifies set-theoretic identities for a 3x4 discrete joint probability table:
    1. I(X; Y) = H(X) - H(X | Y)
    2. I(X; Y) = H(Y) - H(Y | X)
    3. I(X; Y) = H(X) + H(Y) - H(X, Y)
    4. I(X; Y) = D_KL(P_{X, Y} || P_X * P_Y)
    """
    np.random.seed(42)
    # Generate arbitrary valid 3x4 joint PMF
    raw_table = np.random.uniform(1.0, 5.0, size=(3, 4))
    p_xy = raw_table / np.sum(raw_table)

    # Marginals
    p_x = np.sum(p_xy, axis=1)  # shape (3,)
    p_y = np.sum(p_xy, axis=0)  # shape (4,)

    # Product of marginals P_X * P_Y
    p_indep = np.outer(p_x, p_y)

    # Entropies
    h_x = shannon_entropy(p_x)
    h_y = shannon_entropy(p_y)
    h_xy = float(-np.sum(p_xy * np.log(p_xy)))

    # Conditional entropies
    # H(Y | X) = sum_x P(x) H(Y | X = x) = - sum_{x, y} P(x, y) log P(y | x)
    # P(y | x) = P(x, y) / P(x)
    p_y_given_x = p_xy / p_x[:, np.newaxis]
    h_y_given_x = float(-np.sum(p_xy * np.log(p_y_given_x)))

    p_x_given_y = p_xy / p_y[np.newaxis, :]
    h_x_given_y = float(-np.sum(p_xy * np.log(p_x_given_y)))

    # Mutual Information via KL divergence: D_KL(P_XY || P_X * P_Y)
    mi_kl = float(np.sum(p_xy * np.log(p_xy / p_indep)))

    # Alternative formulas
    mi_1 = h_x - h_x_given_y
    mi_2 = h_y - h_y_given_x
    mi_3 = h_x + h_y - h_xy

    print("\n--- Mutual Information & Joint Entropy Decompositions ---")
    print(f"H(X):          {h_x:.6f} nats")
    print(f"H(Y):          {h_y:.6f} nats")
    print(f"H(X, Y):       {h_xy:.6f} nats")
    print(f"H(Y | X):      {h_y_given_x:.6f} nats")
    print(f"H(X | Y):      {h_x_given_y:.6f} nats")
    print(f"I(X; Y) [KL]:  {mi_kl:.6f} nats")
    print(f"H(X) - H(X|Y): {mi_1:.6f} nats")
    print(f"H(Y) - H(Y|X): {mi_2:.6f} nats")
    print(f"H(X)+H(Y)-H(XY): {mi_3:.6f} nats")

    assert abs(mi_kl - mi_1) < 1e-9
    assert abs(mi_kl - mi_2) < 1e-9
    assert abs(mi_kl - mi_3) < 1e-9
    assert mi_kl >= 0.0
    print("✓ All mutual information and conditional entropy identities verified!")


# =====================================================================
# 4. VAE Closed-Form Gaussian KL vs. Monte Carlo Integration
# =====================================================================
def verify_vae_gaussian_kl():
    """
    Compares the analytical closed-form formula used in VAE loss functions:
    D_KL(N(mu, diag(sigma^2)) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    against numerical Monte Carlo expectation.
    """
    np.random.seed(42)
    dim = 4
    mu = np.array([0.5, -1.0, 0.2, 0.0])
    log_var = np.array([0.1, -0.4, 0.5, -0.2])  # log(sigma^2)
    var = np.exp(log_var)
    std = np.sqrt(var)

    # 1. Closed-form analytical formula
    analytical_kl = float(-0.5 * np.sum(1.0 + log_var - mu**2 - var))

    # 2. Monte Carlo estimation: E_{z ~ q}[ log q(z) - log p(z) ]
    num_samples = 300_000
    eps = np.random.standard_normal(size=(num_samples, dim))
    z = mu + std * eps  # Reparameterization trick

    # Log q(z) under N(mu, diag(sigma^2))
    log_q = np.sum(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((z - mu) ** 2) / var, axis=1)
    # Log p(z) under N(0, I)
    log_p = np.sum(-0.5 * np.log(2.0 * np.pi) - 0.5 * (z ** 2), axis=1)

    mc_kl = float(np.mean(log_q - log_p))

    print("\n--- VAE Latent Gaussian KL Divergence Verification ---")
    print(f"Analytical Closed-Form KL: {analytical_kl:.6f} nats")
    print(f"Monte Carlo Estimated KL:  {mc_kl:.6f} nats (N={num_samples})")
    rel_error = abs(analytical_kl - mc_kl) / analytical_kl
    print(f"Relative Error:            {rel_error * 100:.3f}%")

    assert rel_error < 0.01  # Under 1% Monte Carlo error
    print("✓ VAE analytical KL divergence matches numerical expectation perfectly!")


# =====================================================================
# 5. Softmax Cross-Entropy Loss Gradient vs PyTorch Autograd
# =====================================================================
def verify_cross_entropy_gradient():
    """
    Verifies that the analytical gradient of cross-entropy loss with respect
    to pre-softmax logits z:
        d L / d z = softmax(z) - y
    matches PyTorch's automatic differentiation engine exactly.
    """
    torch.manual_seed(42)
    # Raw unnormalized logits
    z = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
    target_class = 0  # One-hot target y = [1, 0, 0]
    target_tensor = torch.tensor(target_class)

    # Compute PyTorch Cross-Entropy Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(z.unsqueeze(0), target_tensor.unsqueeze(0))
    loss.backward()

    # Manual analytical gradient: y_hat - y
    with torch.no_grad():
        y_hat = torch.softmax(z, dim=0)
        y_true = torch.tensor([1.0, 0.0, 0.0])
        analytical_grad = y_hat - y_true

    print("\n--- Cross-Entropy Gradient vs PyTorch Autograd ---")
    print(f"PyTorch Calculated Grad:   {z.grad.numpy()}")
    print(f"Manual Analytical Grad:    {analytical_grad.numpy()}")

    assert torch.allclose(z.grad, analytical_grad, atol=1e-7)
    print("✓ Cross-entropy loss gradient perfectly matched PyTorch autograd engine!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 3.7 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_gibbs_inequality()
    verify_mutual_information()
    verify_vae_gaussian_kl()
    verify_cross_entropy_gradient()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 3.7 PASSED CLEANLY!")
    print("=================================================================")
