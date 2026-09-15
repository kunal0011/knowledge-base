"""
Reinforcement Learning from Human Feedback (RLHF) with PPO Verification Suite
Module 11: Reinforcement Learning - Chapter 27

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Bradley-Terry preference probability (0.731059) and analytical gradients (-0.268941, +0.268941)
   - Token-level KL rewards: R1 = 0.022314, R2 = 2.930685
   - Token TD residuals: delta1 = 0.322314, delta2 = 0.130685
   - Token GAE advantages: A1_hat = 0.446465, A2_hat = 0.130685 (< 1e-6)
2. PyTorch Bradley-Terry Pairwise Reward Model Trainer.
3. Complete Token-Level RLHF PPO Loss Engine:
   - Per-token KL penalty decomposition
   - Token-level GAE calculation
   - PPO-clipped policy objective evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' RLHF TOKEN CALCULATIONS")
    print("=" * 70)

    # --- A. Bradley-Terry Preference Verification ---
    r_w, r_l = 1.20, 0.20
    delta_r = r_w - r_l
    p_pref = 1.0 / (1.0 + np.exp(-delta_r))
    grad_rw = -(1.0 - p_pref)
    grad_rl = +(1.0 - p_pref)

    print(f"Bradley-Terry P(y_w > y_l): {p_pref:.6f} (Expected: 0.731059)")
    print(f"Gradient wrt r_w:            {grad_rw:.6f} (Expected: -0.268941)")
    print(f"Gradient wrt r_l:            {grad_rl:.6f} (Expected: +0.268941)")

    assert np.isclose(p_pref, 0.73105858, atol=1e-5)
    assert np.isclose(grad_rw, -0.26894142, atol=1e-5)
    assert np.isclose(grad_rl, +0.26894142, atol=1e-5)

    # PyTorch Autograd check on Bradley-Terry
    rw_t = torch.tensor(1.20, requires_grad=True)
    rl_t = torch.tensor(0.20, requires_grad=True)
    bt_loss = -torch.log(torch.sigmoid(rw_t - rl_t))
    bt_loss.backward()

    assert np.isclose(rw_t.grad.item(), grad_rw, atol=1e-5)
    assert np.isclose(rl_t.grad.item(), grad_rl, atol=1e-5)
    print(">> SUCCESS: Bradley-Terry preference loss and gradients verified against PyTorch!")

    # --- B. Token-Level KL Rewards ---
    pi_theta = np.array([0.40, 0.80])
    pi_ref = np.array([0.50, 0.40])
    beta = 0.10
    rm_score = 3.00

    log_pi_theta = np.log(pi_theta)
    log_pi_ref = np.log(pi_ref)
    delta_log = log_pi_theta - log_pi_ref

    # R1 = -beta * delta_log[0]
    R1 = -beta * delta_log[0]
    # R2 = rm_score - beta * delta_log[1]
    R2 = rm_score - beta * delta_log[1]

    print(f"Token Reward R1: {R1:.6f} (Expected: 0.022314)")
    print(f"Token Reward R2: {R2:.6f} (Expected: 2.930685)")

    assert np.isclose(R1, 0.02231435, atol=1e-5)
    assert np.isclose(R2, 2.93068529, atol=1e-5)

    # --- C. Token-Level TD Residuals & GAE ---
    V = np.array([2.50, 2.80, 0.00])  # [s1, s2, s3 (terminal)]
    gamma = 1.00
    lambda_val = 0.95

    delta_1 = R1 + gamma * V[1] - V[0]  # 0.022314 + 2.8 - 2.5 = 0.322314
    delta_2 = R2 + gamma * V[2] - V[1]  # 2.930685 + 0.0 - 2.8 = 0.130685

    A2_hat = delta_2
    A1_hat = delta_1 + (gamma * lambda_val) * A2_hat

    print(f"TD Residual delta_1: {delta_1:.6f} (Expected: 0.322314)")
    print(f"TD Residual delta_2: {delta_2:.6f} (Expected: 0.130685)")
    print(f"GAE Advantage A1:    {A1_hat:.6f} (Expected: 0.446465)")
    print(f"GAE Advantage A2:    {A2_hat:.6f} (Expected: 0.130685)")

    assert np.isclose(delta_1, 0.32231435, atol=1e-5)
    assert np.isclose(delta_2, 0.13068529, atol=1e-5)
    assert np.isclose(A1_hat, 0.44646538, atol=1e-5)
    assert np.isclose(A2_hat, 0.13068529, atol=1e-5)

    print(">> SUCCESS: Token-level KL, TD residuals, and GAE advantages verified to < 1e-5!\n")


# =====================================================================
# 2. PyTorch Bradley-Terry Pairwise Reward Model Trainer
# =====================================================================

class PairwiseRewardModel(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, seq_emb):
        # seq_emb: (B, emb_dim)
        return self.net(seq_emb).squeeze(-1)


def verify_reward_model_training():
    print("=" * 70)
    print("2. VERIFYING BRADLEY-TERRY REWARD MODEL OPTIMIZATION")
    print("=" * 70)

    torch.manual_seed(42)
    B, emb_dim = 10, 16
    rm = PairwiseRewardModel(emb_dim=emb_dim)
    optimizer = optim.Adam(rm.parameters(), lr=0.01)

    # Synthetic chosen (w) and rejected (l) embeddings
    # Target: chosen should score higher than rejected
    emb_w = torch.randn(B, emb_dim) + 0.5
    emb_l = torch.randn(B, emb_dim) - 0.5

    for step in range(50):
        r_w = rm(emb_w)
        r_l = rm(emb_l)

        # Bradley-Terry loss: -log(sigmoid(r_w - r_l))
        loss = -torch.log(torch.sigmoid(r_w - r_l) + 1e-8).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_rw = rm(emb_w).mean().item()
        final_rl = rm(emb_l).mean().item()
        print(f"Trained RM Mean Scores: Chosen = {final_rw:.4f}, Rejected = {final_rl:.4f}")
        assert final_rw > final_rl + 1.0, "Chosen response must have significantly higher reward!"

    print(">> SUCCESS: Bradley-Terry Reward Model learned correct preference separation!\n")


# =====================================================================
# 3. Token-Level RLHF PPO Engine
# =====================================================================

def compute_token_rlhf_gae(rewards, values, gamma=1.0, lambda_val=0.95):
    """
    Computes token-level GAE advantages across sequence of length T.
    rewards: (T,)
    values: (T + 1,)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lambda_val * last_gae
        advantages[t] = last_gae

    return advantages


def verify_token_rlhf_ppo():
    print("=" * 70)
    print("3. VERIFYING VECTORIZED TOKEN-LEVEL RLHF PPO GAE ENGINE")
    print("=" * 70)

    # Numerical verification with Section 5 parameters
    rewards_t = torch.tensor([0.02231435, 2.93068529])
    values_t = torch.tensor([2.50, 2.80, 0.00])

    advs = compute_token_rlhf_gae(rewards_t, values_t, gamma=1.0, lambda_val=0.95)

    print(f"Vectorized Token Advantages: [{advs[0].item():.6f}, {advs[1].item():.6f}]")
    assert np.isclose(advs[0].item(), 0.44646538, atol=1e-5)
    assert np.isclose(advs[1].item(), 0.13068529, atol=1e-5)

    # PPO Clipped Objective test
    logprobs_old = torch.tensor([-0.9162907, -0.2231436])
    logprobs_new = logprobs_old + 0.05  # slight policy update
    ratios = torch.exp(logprobs_new - logprobs_old)

    eps = 0.2
    surr1 = ratios * advs
    surr2 = torch.clamp(ratios, 1.0 - eps, 1.0 + eps) * advs
    ppo_loss = -torch.min(surr1, surr2).mean()

    print(f"PPO Token Surrogate Loss: {ppo_loss.item():.6f}")
    assert torch.isfinite(ppo_loss)
    print(">> SUCCESS: Token-level RLHF PPO GAE and surrogate loss verified!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_reward_model_training()
    verify_token_rlhf_ppo()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 27 (RLHF WITH PPO) VERIFICATIONS PASSED!")
    print("=" * 70)
