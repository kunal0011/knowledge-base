"""
World Models & Dreamer Verification Suite
Module 11: Reinforcement Learning - Chapter 24

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - 1D Gaussian KL divergence matching PyTorch torch.distributions (< 1e-7)
   - 2-Step recursive lambda-return calculation in imagination (V2 = 14.0, V1 = 13.52)
2. PyTorch Vectorized Recurrent State-Space Model (RSSM):
   - GRU deterministic transition: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
   - Posterior q(z_t | h_t, x_t) and Prior p(z_t | h_t) with reparameterization trick
   - KL Balancing implementation with stop_gradient (alpha = 0.8)
3. Actor-Critic Imagination Rollout & Lambda-Return Engine:
   - H-step latent hallucination without environment interaction
   - Vectorized backward lambda-return computation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' WORLD MODEL & IMAGINATION CALCULATIONS")
    print("=" * 70)

    # --- 1. Gaussian KL Divergence ---
    mu_p, sigma_p = 0.0, 1.0
    mu_q, sigma_q = 0.8, 0.5

    # Hand calculation:
    term_log = np.log(sigma_p / sigma_q)               # ln(2.0) ≈ 0.693147
    term_quad = (sigma_q**2 + (mu_q - mu_p)**2) / (2.0 * sigma_p**2)  # (0.25 + 0.64) / 2 = 0.445000
    kl_hand = term_log + term_quad - 0.5              # 0.638147

    # PyTorch analytical check:
    q_dist = dist.Normal(torch.tensor(mu_q), torch.tensor(sigma_q))
    p_dist = dist.Normal(torch.tensor(mu_p), torch.tensor(sigma_p))
    kl_torch = dist.kl_divergence(q_dist, p_dist).item()

    print(f"Hand KL Divergence:    {kl_hand:.6f}")
    print(f"PyTorch KL Divergence: {kl_torch:.6f}")
    assert np.isclose(kl_hand, 0.63814718, atol=1e-6)
    assert np.isclose(kl_hand, kl_torch, atol=1e-6)
    print(">> SUCCESS: Gaussian KL divergence exactly matches PyTorch analytical ground truth!")

    # --- 2. Imagined 2-Step Lambda Return ---
    r1_hat, r2_hat = 2.0, 5.0
    v_s2, v_s3 = 8.0, 10.0
    gamma = 0.90
    lambda_val = 0.80

    # Step t = 2:
    V2_lambda = r2_hat + gamma * v_s3  # 5.0 + 0.9 * 10 = 14.0000
    print(f"Imagined V2_lambda: {V2_lambda:.4f} (Expected: 14.0000)")
    assert np.isclose(V2_lambda, 14.0, atol=1e-12)

    # Step t = 1:
    bracket = (1.0 - lambda_val) * v_s2 + lambda_val * V2_lambda  # 0.2 * 8.0 + 0.8 * 14.0 = 1.6 + 11.2 = 12.8
    V1_lambda = r1_hat + gamma * bracket                          # 2.0 + 0.9 * 12.8 = 2.0 + 11.52 = 13.5200
    print(f"Imagined V1_lambda: {V1_lambda:.4f} (Expected: 13.5200)")
    assert np.isclose(V1_lambda, 13.52, atol=1e-12)
    print(">> SUCCESS: Imagined multi-step lambda-returns verified to machine precision (< 1e-12)!\n")


# =====================================================================
# 2. PyTorch Vectorized Recurrent State-Space Model (RSSM)
# =====================================================================

class RSSM(nn.Module):
    """
    Recurrent State-Space Model with deterministic GRU state and stochastic Gaussian latent.
    """
    def __init__(self, action_dim=2, obs_dim=16, deter_dim=32, stoch_dim=16):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # Deterministic dynamics: h_t = GRUCell(concat(z_{t-1}, a_{t-1}), h_{t-1})
        self.rnn_cell = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        # Posterior q(z_t | h_t, x_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * stoch_dim)  # [mean, std_raw]
        )

        # Prior p(z_hat_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * stoch_dim)  # [mean, std_raw]
        )

    def get_dist(self, stats):
        mean, std_raw = torch.chunk(stats, 2, dim=-1)
        std = nn.functional.softplus(std_raw) + 0.1
        return dist.Normal(mean, std)

    def step(self, prev_h, prev_z, action, obs=None):
        """
        Single transition step:
        If obs is provided, computes posterior. Otherwise (in imagination), uses prior.
        """
        rnn_input = torch.cat([prev_z, action], dim=-1)
        h = self.rnn_cell(rnn_input, prev_h)

        # Prior
        prior_stats = self.prior_net(h)
        prior_dist = self.get_dist(prior_stats)

        if obs is not None:
            # Posterior
            post_input = torch.cat([h, obs], dim=-1)
            post_stats = self.posterior_net(post_input)
            post_dist = self.get_dist(post_stats)
            z = post_dist.rsample()  # Reparameterization trick
            return h, z, post_dist, prior_dist
        else:
            # Imagination rollout using prior
            z = prior_dist.rsample()
            return h, z, prior_dist

    def kl_balancing_loss(self, post_dist, prior_dist, alpha=0.8):
        """
        Dreamer KL balancing:
        0.8 * KL(sg[q] || p) + 0.2 * KL(q || sg[p])
        """
        # Detached distributions
        sg_post = dist.Normal(post_dist.mean.detach(), post_dist.stddev.detach())
        sg_prior = dist.Normal(prior_dist.mean.detach(), prior_dist.stddev.detach())

        kl_prior_train = dist.kl_divergence(sg_post, prior_dist).sum(dim=-1)
        kl_post_train = dist.kl_divergence(post_dist, sg_prior).sum(dim=-1)

        kl_balanced = alpha * kl_prior_train + (1.0 - alpha) * kl_post_train
        return kl_balanced.mean()


def verify_rssm_module():
    print("=" * 70)
    print("2. VERIFYING PYTORCH RSSM DYNAMICS & KL BALANCING")
    print("=" * 70)

    torch.manual_seed(42)
    rssm = RSSM(action_dim=2, obs_dim=16, deter_dim=32, stoch_dim=16)

    batch_size = 4
    h = torch.zeros(batch_size, 32)
    z = torch.zeros(batch_size, 16)
    a = torch.randn(batch_size, 2)
    obs = torch.randn(batch_size, 16)

    # 1. Posterior step with real observation
    next_h, next_z, post_dist, prior_dist = rssm.step(h, z, a, obs)
    assert next_h.shape == (batch_size, 32)
    assert next_z.shape == (batch_size, 16)

    # 2. KL balancing loss calculation
    kl_loss = rssm.kl_balancing_loss(post_dist, prior_dist, alpha=0.8)
    print(f"Computed Balanced KL Loss: {kl_loss.item():.4f}")
    assert kl_loss.item() > 0.0
    assert torch.isfinite(kl_loss)

    # 3. Differentiability check
    kl_loss.backward()
    for name, param in rssm.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
    print(">> SUCCESS: RSSM forward and backward gradient passes verified successfully!\n")


# =====================================================================
# 3. Actor-Critic Imagination Rollout & Lambda-Return Engine
# =====================================================================

def compute_lambda_returns(rewards, values, gamma=0.95, lambda_val=0.8):
    """
    Computes recursive generalized lambda-returns across an imagined rollout.
    Args:
        rewards: Tensor of shape (H, B)
        values: Tensor of shape (H + 1, B)
    Returns:
        returns: Tensor of shape (H, B)
    """
    H, B = rewards.shape
    returns = torch.zeros_like(rewards)
    last_return = values[H]  # Boundary condition V_{H}^lambda = v(s_H)

    for t in reversed(range(H)):
        next_val = values[t + 1]
        target = (1.0 - lambda_val) * next_val + lambda_val * last_return
        last_return = rewards[t] + gamma * target
        returns[t] = last_return

    return returns


def verify_imagination_rollout():
    print("=" * 70)
    print("3. VERIFYING IMAGINATION ROLLOUT & LAMBDA-RETURN RECURSION")
    print("=" * 70)

    # Test numerical equivalence of vectorized lambda-return with Section 5 hand calculation
    # H = 2, B = 1
    rewards = torch.tensor([[2.0], [5.0]])      # r1 = 2.0, r2 = 5.0
    values = torch.tensor([[4.0], [8.0], [10.0]]) # v1 = 4.0, v2 = 8.0, v3 = 10.0

    lambda_returns = compute_lambda_returns(rewards, values, gamma=0.90, lambda_val=0.80)

    v1_calc = lambda_returns[0, 0].item()
    v2_calc = lambda_returns[1, 0].item()

    print(f"Vectorized V2 Lambda-Return: {v2_calc:.4f} (Expected: 14.0000)")
    print(f"Vectorized V1 Lambda-Return: {v1_calc:.4f} (Expected: 13.5200)")

    assert np.isclose(v2_calc, 14.0, atol=1e-6)
    assert np.isclose(v1_calc, 13.52, atol=1e-6)
    print(">> SUCCESS: Vectorized lambda-return engine matches analytical derivation perfectly!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_rssm_module()
    verify_imagination_rollout()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 24 (WORLD MODELS & DREAMER) VERIFICATIONS PASSED!")
    print("=" * 70)
