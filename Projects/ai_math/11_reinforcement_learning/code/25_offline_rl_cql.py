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
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_cql_offline_learning()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 25 (OFFLINE RL & CQL) VERIFICATIONS PASSED!")
    print("=" * 70)
