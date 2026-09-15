"""
Mathematical Verification Script for Module 11.12: Distributional RL (C51 & QR-DQN)
===================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Exact C51 Categorical Projection algorithm:
     m = [0.1600, 0.6000, 0.2400]
   - Mass conservation: sum(m) = 1.0000
   - Mean conservation: E[m] = r + gamma * E[p'] = 5.4000
   - Cross-entropy loss: L = ln(3) = 1.0986122886681098
2. Generalized Vectorized C51 Projection Implementation (PyTorch & NumPy):
   - Full 51-atom batch projection test.
3. Quantile Regression (QR-DQN) Loss:
   - Asymmetric pinball Huber loss calculation across quantiles.
"""

import numpy as np
import torch
import torch.nn.functional as F

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 C51 Hand Calculation Verification ---")
    V_min = 0.0
    V_max = 10.0
    N = 3
    delta_z = (V_max - V_min) / (N - 1)
    atoms = np.linspace(V_min, V_max, N)
    
    r = 1.0
    gamma = 0.8
    p_next = np.array([0.2, 0.5, 0.3])
    
    # Run hand C51 projection
    m = np.zeros(N)
    for j in range(N):
        Tz = np.clip(r + gamma * atoms[j], V_min, V_max)
        bj = (Tz - V_min) / delta_z
        l = int(np.floor(bj))
        u = int(np.ceil(bj))
        if l == u:
            m[l] += p_next[j]
        else:
            m[l] += p_next[j] * (u - bj)
            m[u] += p_next[j] * (bj - l)
            
    print(f"  Atoms: {atoms}")
    print(f"  Next State Probabilities: {p_next}")
    print(f"  Projected Target Distribution m: {m}")
    
    expected_m = np.array([0.16, 0.60, 0.24])
    assert np.allclose(m, expected_m), f"Expected {expected_m}, got {m}"
    assert np.isclose(np.sum(m), 1.0000), "Projected distribution must sum to 1.0"
    
    # Verify exact mean conservation
    mean_m = np.dot(atoms, m)
    expected_mean = r + gamma * np.dot(atoms, p_next)
    print(f"  E[m] = {mean_m:.4f}, r + gamma*E[p'] = {expected_mean:.4f}")
    assert np.isclose(mean_m, 5.4000)
    assert np.isclose(mean_m, expected_mean)
    
    # Cross-entropy loss with uniform p = [1/3, 1/3, 1/3]
    p_pred = np.full(N, 1.0 / 3.0)
    ce_loss = -np.sum(m * np.log(p_pred))
    expected_ce = np.log(3.0)
    print(f"  Cross-Entropy Loss: {ce_loss:.6f}, ln(3) = {expected_ce:.6f}")
    assert np.isclose(ce_loss, expected_ce)
    
    print("  [PASSED] Part 5 C51 hand calculations verified to exact precision!\n")


def test_vectorized_c51_projection():
    print("--- Test 2: Vectorized C51 Batch Projection (PyTorch) ---")
    batch_size = 4
    N = 51
    V_min = -10.0
    V_max = 10.0
    delta_z = (V_max - V_min) / (N - 1)
    atoms = torch.linspace(V_min, V_max, N)
    
    gamma = 0.99
    rewards = torch.tensor([0.0, 1.0, -1.0, 5.0]).unsqueeze(1) # (B, 1)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0]).unsqueeze(1)    # (B, 1)
    
    # Random next-state distributions
    torch.manual_seed(42)
    next_p = F.softmax(torch.randn(batch_size, N), dim=1) # (B, N)
    
    # Vectorized Bellman projection
    Tz = rewards + (1.0 - dones) * gamma * atoms.unsqueeze(0) # (B, N)
    Tz = Tz.clamp(V_min, V_max)
    b = (Tz - V_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    
    # Correct handling for l == u
    l[(u > 0) * (l == u)] -= 1
    u[(l < (N - 1)) * (l == u)] += 1
    
    m = torch.zeros(batch_size, N)
    offset = torch.linspace(0, (batch_size - 1) * N, batch_size).unsqueeze(1).expand(batch_size, N).long()
    
    m.view(-1).index_add_(0, (l + offset).view(-1), (next_p * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_p * (b - l.float())).view(-1))
    
    # Verify every batch item sums to 1.0
    sums = m.sum(dim=1)
    print(f"  Batch Probability Sums: {sums.numpy()}")
    assert torch.allclose(sums, torch.ones(batch_size)), "All projected distributions must sum to 1.0"
    
    print("  [PASSED] Vectorized C51 projection verified!\n")


def test_qr_dqn_quantile_huber_loss():
    print("--- Test 3: Quantile Regression (QR-DQN) Loss Verification ---")
    # For N quantiles, target locations theta_target, predicted locations theta_pred
    N = 5
    taus = np.array([(2 * i - 1) / (2.0 * N) for i in range(1, N + 1)]) # [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"  Quantile fractions taus: {taus}")
    
    # Test pinball loss for scalar errors
    def quantile_huber_loss(u, tau, kappa=1.0):
        abs_u = abs(u)
        huber = 0.5 * (u ** 2) if abs_u <= kappa else kappa * (abs_u - 0.5 * kappa)
        weight = abs(tau - float(u < 0))
        return weight * huber
    
    # Verify asymmetric weighting:
    # If tau = 0.9 (90th percentile), an underestimation (u > 0) has weight 0.9,
    # while an overestimation (u < 0) has weight 0.1
    u_pos = 2.0
    u_neg = -2.0
    
    loss_pos = quantile_huber_loss(u_pos, tau=0.9, kappa=1.0)
    loss_neg = quantile_huber_loss(u_neg, tau=0.9, kappa=1.0)
    
    print(f"  tau=0.9: Loss on underestimation (u=+2.0) = {loss_pos:.4f}")
    print(f"  tau=0.9: Loss on overestimation  (u=-2.0) = {loss_neg:.4f}")
    
    # Huber of 2.0 with kappa=1 is 1 * (2 - 0.5) = 1.5
    # loss_pos = 0.9 * 1.5 = 1.35
    # loss_neg = 0.1 * 1.5 = 0.15
    assert np.isclose(loss_pos, 1.3500)
    assert np.isclose(loss_neg, 0.1500)
    assert np.isclose(loss_pos / loss_neg, 9.0) # 0.9 / 0.1 = 9x ratio!
    
    print("  [PASSED] QR-DQN asymmetric quantile loss verified!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.12 DISTRIBUTIONAL RL (C51 & QR-DQN) TESTS")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_vectorized_c51_projection()
    test_qr_dqn_quantile_huber_loss()
    
    print("=================================================================")
    print("ALL MODULE 11.12 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
