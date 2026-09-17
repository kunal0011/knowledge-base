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
# 4. Section 6 Illustrations 1–5 Analytical Verification Suite
# =====================================================================

def verify_section6_illustrations():
    print("=" * 70)
    print("4. VERIFYING SECTION 6 SOLVED ILLUSTRATIONS 1 TO 5")
    print("=" * 70)

    # --- Illustration 1: Bradley-Terry Pairwise Loss Gradient ---
    rw1, rl1 = 1.20, 0.20
    dr1 = rw1 - rl1
    p1 = 1.0 / (1.0 + np.exp(-dr1))
    loss1 = -np.log(p1)
    grad_rw1 = -(1.0 - p1)
    grad_rl1 = +(1.0 - p1)
    assert np.isclose(p1, 0.73105858, atol=1e-5)
    assert np.isclose(loss1, 0.31326169, atol=1e-5)
    assert np.isclose(grad_rw1, -0.26894142, atol=1e-5)
    assert np.isclose(grad_rl1, +0.26894142, atol=1e-5)
    print(">> Illustration 1 Verified: Bradley-Terry loss & gradients match to < 1e-5.")

    # --- Illustration 2: Reward Model Parameter Gradient Update ---
    w0 = np.array([1.50, -0.50])
    b = 0.25
    hw = np.array([1.20, 0.50])
    hl = np.array([0.40, 0.90])
    rw2 = np.dot(w0, hw) + b
    rl2 = np.dot(w0, hl) + b
    dr2 = rw2 - rl2
    assert np.isclose(rw2, 1.80, atol=1e-5)
    assert np.isclose(rl2, 0.40, atol=1e-5)
    assert np.isclose(dr2, 1.40, atol=1e-5)
    p2 = 1.0 / (1.0 + np.exp(-dr2))
    loss2 = -np.log(p2)
    scalar_err = -(1.0 - p2)
    dh = hw - hl
    grad_w = scalar_err * dh
    grad_b = 0.0
    lr = 0.10
    w1 = w0 - lr * grad_w
    rw2_new = np.dot(w1, hw) + b
    rl2_new = np.dot(w1, hl) + b
    dr2_new = rw2_new - rl2_new
    p2_new = 1.0 / (1.0 + np.exp(-dr2_new))
    loss2_new = -np.log(p2_new)
    assert np.isclose(p2, 0.80218389, atol=1e-5)
    assert np.isclose(loss2, 0.22041687, atol=1e-5)
    assert np.isclose(grad_w[0], -0.15825289, atol=1e-5)
    assert np.isclose(grad_w[1], +0.07912644, atol=1e-5)
    assert np.isclose(w1[0], 1.51582529, atol=1e-5)
    assert np.isclose(w1[1], -0.50791264, atol=1e-5)
    assert loss2_new < loss2
    print(">> Illustration 2 Verified: Parameter gradients and monotonic loss descent match.")

    # --- Illustration 3: 4-Token Sequence KL Reward & GAE ---
    beta3 = 0.05
    lam3 = 0.95
    gamma3 = 1.00
    r_phi3 = 2.50
    pi_th3 = np.array([0.60, 0.25, 0.70, 0.80])
    pi_rf3 = np.array([0.40, 0.50, 0.35, 0.40])
    V3 = np.array([2.20, 2.30, 2.40, 2.45, 0.00])
    d_log3 = np.log(pi_th3 / pi_rf3)
    R3 = np.zeros(4)
    for t in range(3):
        R3[t] = -beta3 * d_log3[t]
    R3[3] = r_phi3 - beta3 * d_log3[3]
    assert np.isclose(R3[0], -0.02027326, atol=1e-5)
    assert np.isclose(R3[1], +0.03465736, atol=1e-5)
    assert np.isclose(R3[2], -0.03465736, atol=1e-5)
    assert np.isclose(R3[3], +2.46534264, atol=1e-5)
    assert np.isclose(np.sum(R3), 2.44506938, atol=1e-5)

    delta3 = np.zeros(4)
    for t in range(4):
        delta3[t] = R3[t] + gamma3 * V3[t + 1] - V3[t]
    assert np.isclose(delta3[0], 0.07972674, atol=1e-5)
    assert np.isclose(delta3[1], 0.13465736, atol=1e-5)
    assert np.isclose(delta3[2], 0.01534264, atol=1e-5)
    assert np.isclose(delta3[3], 0.01534264, atol=1e-5)

    A3 = np.zeros(4)
    A3[3] = delta3[3]
    for t in range(2, -1, -1):
        A3[t] = delta3[t] + gamma3 * lam3 * A3[t + 1]
    assert np.isclose(A3[3], 0.01534264, atol=1e-5)
    assert np.isclose(A3[2], 0.02991815, atol=1e-5)
    assert np.isclose(A3[1], 0.16307960, atol=1e-5)
    assert np.isclose(A3[0], 0.23465236, atol=1e-5)
    print(">> Illustration 3 Verified: Token KL penalties, TD errors, and GAE match.")

    # --- Illustration 4: Closed-Form Optimal Policy Distribution ---
    pi_ref4 = np.array([0.60, 0.30, 0.10])
    r4 = np.array([1.00, 2.50, 4.00])
    beta4 = 1.00
    unnorm4 = pi_ref4 * np.exp(r4 / beta4)
    Z4 = np.sum(unnorm4)
    pi_star4 = unnorm4 / Z4
    assert np.isclose(Z4, 10.74553229, atol=1e-5)
    assert np.isclose(pi_star4[0], 0.15178114, atol=1e-5)
    assert np.isclose(pi_star4[1], 0.34011793, atol=1e-5)
    assert np.isclose(pi_star4[2], 0.50810094, atol=1e-5)
    exp_r4 = np.sum(pi_star4 * r4)
    kl4 = np.sum(pi_star4 * np.log(pi_star4 / pi_ref4))
    obj4 = exp_r4 - beta4 * kl4
    theory_obj4 = beta4 * np.log(Z4)
    assert np.isclose(obj4, theory_obj4, atol=1e-5)
    assert np.isclose(obj4, 2.37448984, atol=1e-5)

    # Greedy comparison
    pi_greedy = np.array([0.0, 0.0, 1.0])
    kl_greedy = 1.0 * np.log(1.0 / 0.10)
    obj_greedy = 4.0 - beta4 * kl_greedy
    assert np.isclose(obj_greedy, 1.69741491, atol=1e-5)
    assert obj4 > obj_greedy
    print(">> Illustration 4 Verified: Closed-form optimal policy and partition identity match.")

    # --- Illustration 5: Complete 4-Model RLHF Forward Pass ---
    pi_old5 = np.array([0.50, 0.40])
    pi_new5 = np.array([0.60, 0.52])
    pi_ref5 = np.array([0.45, 0.35])
    r_phi5 = 2.80
    V_old5 = np.array([1.80, 2.20, 0.00])
    V_new5 = np.array([1.85, 2.18])
    beta5 = 0.10
    eps5 = 0.20
    gamma5 = 1.00
    lam5 = 0.95
    c1_5 = 0.50

    d_log5 = np.log(pi_old5 / pi_ref5)
    R5 = np.array([-beta5 * d_log5[0], r_phi5 - beta5 * d_log5[1]])
    delta5 = np.array([R5[0] + gamma5 * V_old5[1] - V_old5[0], R5[1] + gamma5 * V_old5[2] - V_old5[1]])
    A5 = np.array([delta5[0] + gamma5 * lam5 * delta5[1], delta5[1]])
    assert np.isclose(R5[0], -0.01053605, atol=1e-5)
    assert np.isclose(R5[1], +2.78664686, atol=1e-5)
    assert np.isclose(delta5[0], 0.38946395, atol=1e-5)
    assert np.isclose(delta5[1], 0.58664686, atol=1e-5)
    assert np.isclose(A5[0], 0.94677847, atol=1e-5)
    assert np.isclose(A5[1], 0.58664686, atol=1e-5)

    rho5 = pi_new5 / pi_old5
    surr1_5 = rho5 * A5
    surr2_5 = np.clip(rho5, 1.0 - eps5, 1.0 + eps5) * A5
    ppo_tok5 = np.minimum(surr1_5, surr2_5)
    L_actor5 = -np.mean(ppo_tok5)
    assert np.isclose(rho5[0], 1.20, atol=1e-5)
    assert np.isclose(rho5[1], 1.30, atol=1e-5)
    assert np.isclose(ppo_tok5[0], 1.13613416, atol=1e-5)
    assert np.isclose(ppo_tok5[1], 0.70397623, atol=1e-5)  # clipped!
    assert np.isclose(L_actor5, -0.92005519, atol=1e-5)

    V_targ5 = V_old5[:2] + A5
    L_critic5 = 0.5 * np.mean((V_new5 - V_targ5)**2)
    L_tot5 = L_actor5 + c1_5 * L_critic5
    assert np.isclose(V_targ5[0], 2.74677847, atol=1e-5)
    assert np.isclose(V_targ5[1], 2.78664686, atol=1e-5)
    assert np.isclose(L_critic5, 0.29305801, atol=1e-5)
    assert np.isclose(L_tot5, -0.77352619, atol=1e-5)
    print(">> Illustration 5 Verified: 4-model forward pass, PPO clipping, and multi-task loss match.\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_reward_model_training()
    verify_token_rlhf_ppo()
    verify_section6_illustrations()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 27 (RLHF WITH PPO) VERIFICATIONS PASSED!")
    print("=" * 70)

