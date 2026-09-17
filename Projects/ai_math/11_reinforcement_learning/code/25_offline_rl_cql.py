"""
Conservative Q-Learning (CQL) & Offline RL Verification Suite
Module 11: Reinforcement Learning - Chapter 25

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Softmax probabilities: p(a1) = 0.047426, p(a2) = 0.952574
   - Analytical and PyTorch Autograd CQL gradients: [-0.952574, +0.952574]
   - Conservative update: Q_new(a1) = 2.476287, Q_new(a2) = 4.523713 (< 1e-6)
2. Complete PyTorch Conservative Q-Learning (CQL) discrete agent.
3. Offline RL OOD Overestimation Experiment:
   - Demonstrates how standard unregularized Q-learning fails on unseen actions
   - Proves CQL forces OOD action-values down into a safe lower bound.
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
    print("1. VERIFYING PART 5 'AI BY HAND' CONSERVATIVE Q-LEARNING GRADIENTS")
    print("=" * 70)

    # Initial Q-values: [In-data a1, OOD a2]
    q_vals = np.array([2.0, 5.0])
    pi_beta = np.array([1.0, 0.0])
    alpha = 1.0
    eta = 0.5

    # 1. Softmax and Log-Sum-Exp Hand Calculation
    exp_q = np.exp(q_vals)
    sum_exp = np.sum(exp_q)
    logsumexp_val = np.log(sum_exp)
    softmax_p = exp_q / sum_exp

    print(f"Log-Sum-Exp:       {logsumexp_val:.6f} (Expected: 5.048615)")
    print(f"Softmax p(a1):     {softmax_p[0]:.6f} (Expected: 0.047426)")
    print(f"Softmax p(a2):     {softmax_p[1]:.6f} (Expected: 0.952574)")

    assert np.isclose(logsumexp_val, 5.048615, atol=1e-5)
    assert np.isclose(softmax_p[0], 0.047426, atol=1e-5)
    assert np.isclose(softmax_p[1], 0.952574, atol=1e-5)

    # 2. CQL Gradients: grad = p_softmax - pi_beta
    grad_cql = softmax_p - pi_beta
    print(f"CQL Gradient a1:   {grad_cql[0]:.6f} (Expected: -0.952574)")
    print(f"CQL Gradient a2:   {grad_cql[1]:.6f} (Expected: +0.952574)")

    assert np.isclose(grad_cql[0], -0.952574, atol=1e-5)
    assert np.isclose(grad_cql[1], 0.952574, atol=1e-5)

    # 3. Gradient Descent Step: Q_new = Q - eta * alpha * grad
    q_new = q_vals - eta * alpha * grad_cql
    print(f"Updated Q_new(a1): {q_new[0]:.6f} (Expected: 2.476287)")
    print(f"Updated Q_new(a2): {q_new[1]:.6f} (Expected: 4.523713)")

    assert np.isclose(q_new[0], 2.476287, atol=1e-5)
    assert np.isclose(q_new[1], 4.523713, atol=1e-5)

    # 4. PyTorch Autograd Verification
    q_tensor = torch.tensor([2.0, 5.0], dtype=torch.float32, requires_grad=True)
    cql_loss = torch.logsumexp(q_tensor, dim=0) - q_tensor[0]
    cql_loss.backward()

    torch_grads = q_tensor.grad.numpy()
    print(f"PyTorch Autograd Gradients: [{torch_grads[0]:.6f}, {torch_grads[1]:.6f}]")
    assert np.allclose(torch_grads, grad_cql, atol=1e-5)

    print(">> SUCCESS: Hand calculation perfectly verified against PyTorch Autograd (< 1e-5)!\n")


# =====================================================================
# 2. Complete PyTorch CQL Discrete Agent
# =====================================================================

class CQLQNetwork(nn.Module):
    def __init__(self, state_dim=4, n_actions=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class CQLAgent:
    """
    Conservative Q-Learning (CQL) agent for offline datasets.
    """
    def __init__(self, state_dim=4, n_actions=2, alpha=1.0, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.q_net = CQLQNetwork(state_dim, n_actions)
        self.target_net = CQLQNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def train_step(self, states, actions, rewards, next_states, dones):
        # 1. Standard Bellman TD Target
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q, _ = torch.max(next_q, dim=-1)
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        # 2. Predicted Q-values for actions taken in data
        q_values = self.q_net(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 3. Bellman MSE Loss
        td_loss = 0.5 * nn.functional.mse_loss(q_taken, targets)

        # 4. Conservative Regularizer: E_{s}[ logsumexp(Q(s, a)) - Q(s, a_data) ]
        cql_reg = torch.logsumexp(q_values, dim=1) - q_taken
        cql_loss = cql_reg.mean()

        total_loss = td_loss + self.alpha * cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return td_loss.item(), cql_loss.item()


# =====================================================================
# 3. Offline RL OOD Overestimation Experiment
# =====================================================================

def verify_cql_offline_learning():
    print("=" * 70)
    print("2. VERIFYING CQL VS UNREGULARIZED Q-LEARNING ON OFFLINE DATASET")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Synthetic offline dataset:
    # Action 0 is safe & frequent (reward = +2.0)
    # Action 1 is OOD (never taken in dataset!)
    n_samples = 200
    state_dim = 2
    states = torch.randn(n_samples, state_dim)
    actions = torch.zeros(n_samples, dtype=torch.long)  # ONLY action 0 exists!
    rewards = torch.full((n_samples,), 2.0, dtype=torch.float32)
    next_states = torch.randn(n_samples, state_dim)
    dones = torch.ones(n_samples, dtype=torch.float32)  # terminal step

    # Train Unregularized Agent (alpha = 0.0) vs Conservative Agent (alpha = 2.0)
    unreg_agent = CQLAgent(state_dim=state_dim, n_actions=2, alpha=0.0, lr=1e-2)
    cql_agent = CQLAgent(state_dim=state_dim, n_actions=2, alpha=2.0, lr=1e-2)

    # Introduce deliberate positive weight bias towards action 1 (OOD hallucination)
    with torch.no_grad():
        unreg_agent.q_net.net[-1].bias[1].fill_(5.0)
        cql_agent.q_net.net[-1].bias[1].fill_(5.0)

    for epoch in range(100):
        unreg_agent.train_step(states, actions, rewards, next_states, dones)
        cql_agent.train_step(states, actions, rewards, next_states, dones)

    # Evaluate learned Q-values on test state:
    test_s = torch.zeros(1, state_dim)
    with torch.no_grad():
        unreg_q = unreg_agent.q_net(test_s).numpy()[0]
        cql_q = cql_agent.q_net(test_s).numpy()[0]

    print(f"Unregularized Q-values: Q(s, a0_data) = {unreg_q[0]:.4f} | Q(s, a1_OOD) = {unreg_q[1]:.4f}")
    print(f"Conservative CQL values: Q(s, a0_data) = {cql_q[0]:.4f} | Q(s, a1_OOD) = {cql_q[1]:.4f}")

    # In unregularized RL, a1 remains falsely elevated (> a0) because no data forces it down!
    assert unreg_q[1] > unreg_q[0], "Unregularized agent should suffer from OOD overestimation"

    # In CQL, conservative loss forces a1 below a0, so the agent chooses the safe, data-backed action!
    assert cql_q[0] > cql_q[1], "CQL must successfully suppress OOD action value below in-data action value"
    print(">> SUCCESS: CQL successfully suppressed OOD hallucination and picked safe data-backed action!\n")


# =====================================================================
# 3. Section 6 Numerical Illustrations Verification
# =====================================================================

def verify_section6_illustrations():
    print("=" * 70)
    print("3. VERIFYING SECTION 6 NUMERICAL ILLUSTRATIONS")
    print("=" * 70)

    # Illustration 2: 3-action CQL penalty
    Q2 = np.array([4.0, 2.0, 8.0])
    N2 = np.array([100.0, 50.0, 0.0])
    pi_beta2 = N2 / np.sum(N2)
    exp_Q2 = np.exp(Q2)
    sum_exp2 = np.sum(exp_Q2)
    p_soft2 = exp_Q2 / sum_exp2
    grad_R2 = p_soft2 - pi_beta2
    delta_Q2 = - 0.1 * 1.0 * grad_R2
    Q_new2 = Q2 + delta_Q2

    assert np.isclose(np.log(sum_exp2), 8.020581, atol=1e-5)
    assert np.isclose(p_soft2[0], 0.017943, atol=1e-5)
    assert np.isclose(p_soft2[2], 0.979629, atol=1e-5)
    assert np.isclose(Q_new2[0], 4.064872, atol=1e-5)
    assert np.isclose(Q_new2[1], 2.033091, atol=1e-5)
    assert np.isclose(Q_new2[2], 7.902037, atol=1e-5)
    print(">> Illustration 2 (3-Action CQL Penalty): VERIFIED!")

    # Illustration 3: CQL-SAC Target Calculation
    r3 = 1.50
    gamma3 = 0.95
    alpha_sac3 = 0.20
    log_pi3 = -1.20
    min_q3 = min(12.00, 10.50)
    v_targ3 = min_q3 - alpha_sac3 * log_pi3
    y_bellman3 = r3 + gamma3 * v_targ3
    td_err3 = 11.00 - y_bellman3
    td_loss3 = 0.5 * (td_err3 ** 2)

    q_samples3 = np.array([8.0, 6.0, 13.0, 10.0])
    lse_sac3 = np.log(np.mean(np.exp(q_samples3)))
    cql_reg3 = lse_sac3 - 11.00
    total_loss3 = td_loss3 + 1.0 * cql_reg3

    assert np.isclose(v_targ3, 10.7400, atol=1e-4)
    assert np.isclose(y_bellman3, 11.7030, atol=1e-4)
    assert np.isclose(td_loss3, 0.247104, atol=1e-4)
    assert np.isclose(lse_sac3, 11.669554, atol=1e-4)
    assert np.isclose(total_loss3, 0.916658, atol=1e-4)
    print(">> Illustration 3 (CQL-SAC Soft Bellman Target): VERIFIED!")

    # Illustration 4: Standard vs CQL Error Propagation
    qd = 2.80
    qo = 4.50
    alpha4 = 2.0
    lr4 = 0.2
    y4 = 2.80

    for step in range(1, 6):
        td_e = qd - y4
        ed = np.exp(qd)
        eo = np.exp(qo)
        se = ed + eo
        pd = ed / se
        po = eo / se
        gd = td_e + alpha4 * (pd - 1.0)
        go = alpha4 * po
        qd = qd - lr4 * gd
        qo = qo - lr4 * go

    assert np.isclose(qd, 3.601508, atol=1e-4)
    assert np.isclose(qo, 3.223142, atol=1e-4)
    assert qd > qo, "At step 5, in-data action must exceed OOD action!"
    print(">> Illustration 4 (Standard vs CQL Error Propagation): VERIFIED!")

    # Illustration 5: TD3+BC Adaptive Regularization
    Q_batch5 = np.array([120.0, -80.0, 150.0, 50.0])
    mean_abs_Q5 = np.mean(np.abs(Q_batch5))
    lam5 = 2.50 / mean_abs_Q5
    diff5 = np.array([0.80, -0.40]) - np.array([0.50, -0.10])
    bc_loss5 = np.sum(diff5 ** 2)
    grad_bc5 = 2 * diff5
    rl_grad5 = lam5 * np.array([40.0, -20.0])
    total_actor_grad5 = rl_grad5 - grad_bc5

    assert np.isclose(mean_abs_Q5, 100.0, atol=1e-4)
    assert np.isclose(lam5, 0.0250, atol=1e-4)
    assert np.isclose(bc_loss5, 0.1800, atol=1e-4)
    assert np.allclose(total_actor_grad5, [0.40, 0.10], atol=1e-4)
    print(">> Illustration 5 (TD3+BC Adaptive Weight Normalization): VERIFIED!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_cql_offline_learning()
    verify_section6_illustrations()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 25 (OFFLINE RL & CQL) VERIFICATIONS PASSED!")
    print("=" * 70)
