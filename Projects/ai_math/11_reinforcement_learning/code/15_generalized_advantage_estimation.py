"""
Mathematical Verification Script for Module 11.15: Generalized Advantage Estimation (GAE)
=========================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - TD residuals: delta_0 = 1.6000, delta_1 = 2.5000, delta_2 = 0.0000
   - Backward GAE(0.9, 0.8) advantages:
     A_2 = 0.0000, A_1 = 2.5000, A_0 = 3.4000
   - Intermediate position between lambda=0 (1.6) and lambda=1 (3.85).
2. Production-Grade Vectorized GAE Implementation:
   - Verifies lambda = 0.0 identically reproduces 1-step TD residuals delta_t.
   - Verifies lambda = 1.0 identically reproduces Monte Carlo advantages G_t - V(S_t).
   - Verifies multi-episode boundary reset with terminal masks.
3. Bias-Variance Trade-Off Sweep across lambda in [0.0, 1.0].
"""

import numpy as np
import torch

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 GAE Hand Calculation Verification ---")
    gamma = 0.90
    lam = 0.80
    
    # 3-step episode
    # R_1 = 1.0, R_2 = 2.0, R_3 = 5.0 (terminal)
    rewards = np.array([1.0, 2.0, 5.0])
    values = np.array([3.0, 4.0, 5.0, 0.0]) # V(S0), V(S1), V(S2), V(S3_term)
    
    # 1. Forward TD errors
    deltas = np.zeros(3)
    for t in range(3):
        deltas[t] = rewards[t] + gamma * values[t + 1] - values[t]
        
    print(f"  TD Residuals deltas: {deltas}")
    expected_deltas = np.array([1.6, 2.5, 0.0])
    assert np.allclose(deltas, expected_deltas), f"Expected {expected_deltas}, got {deltas}"
    
    # 2. Backward GAE recursion
    advantages = np.zeros(3)
    gae = 0.0
    for t in reversed(range(3)):
        gae = deltas[t] + (gamma * lam) * gae
        advantages[t] = gae
        
    print(f"  GAE(0.9, 0.8) Advantages: {advantages}")
    expected_advantages = np.array([3.4, 2.5, 0.0])
    assert np.allclose(advantages, expected_advantages), f"Expected {expected_advantages}, got {advantages}"
    
    # 3. Verify boundary conditions:
    # lambda = 0:
    adv_lam0 = deltas.copy()
    assert np.allclose(adv_lam0, [1.6, 2.5, 0.0])
    
    # lambda = 1.0 (Monte Carlo):
    # G = [6.85, 6.5, 5.0] -> G - V = [6.85 - 3.0, 6.5 - 4.0, 5.0 - 5.0] = [3.85, 2.5, 0.0]
    G = np.zeros(3)
    current_G = 0.0
    for t in reversed(range(3)):
        current_G = rewards[t] + gamma * current_G
        G[t] = current_G
    adv_lam1 = G - values[:3]
    print(f"  Monte Carlo Advantages (lambda=1): {adv_lam1}")
    assert np.allclose(adv_lam1, [3.85, 2.5, 0.0])
    
    # Check that GAE(0.8) for t=0 is between lambda=0 (1.6) and lambda=1 (3.85)
    assert adv_lam0[0] < advantages[0] < adv_lam1[0]
    print("  [PASSED] Part 5 GAE hand calculations verified to exact machine precision!\n")


def compute_gae_vectorized(rewards, values, dones, gamma, lam):
    """
    Standard production GAE computation used in PPO.
    rewards: (T, N)
    values:  (T + 1, N)
    dones:   (T, N)
    """
    T = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        next_val = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        
    return advantages


def test_gae_boundary_cases():
    print("--- Test 2: GAE Boundary Cases Verification (lambda=0 and lambda=1) ---")
    np.random.seed(42)
    T = 20
    gamma = 0.99
    
    rewards = np.random.normal(0.0, 1.0, size=(T,))
    values = np.random.normal(0.0, 1.0, size=(T + 1,))
    dones = np.zeros(T)
    dones[-1] = 1.0 # terminal at final step
    
    # 1. Lambda = 0.0 must match 1-step TD residuals delta_t
    gae_0 = compute_gae_vectorized(rewards, values, dones, gamma, lam=0.0)
    deltas = np.zeros(T)
    for t in range(T):
        next_val = values[t + 1] if not dones[t] else 0.0
        deltas[t] = rewards[t] + gamma * next_val - values[t]
        
    assert np.allclose(gae_0, deltas), "GAE(lambda=0) must match 1-step TD errors identically!"
    print("  GAE(lambda=0.0) matches 1-step TD errors identically!")
    
    # 2. Lambda = 1.0 must match Monte Carlo G_t - V(S_t)
    gae_1 = compute_gae_vectorized(rewards, values, dones, gamma, lam=1.0)
    G = np.zeros(T)
    current_G = 0.0
    for t in reversed(range(T)):
        current_G = rewards[t] + gamma * current_G * (1.0 - dones[t])
        G[t] = current_G
    mc_adv = G - values[:T]
    
    # For the last step where done=1, G[T-1] = rewards[T-1] -> G - V = R - V
    assert np.allclose(gae_1, mc_adv), "GAE(lambda=1) must match Monte Carlo returns G_t - V(S_t) identically!"
    print("  GAE(lambda=1.0) matches Monte Carlo advantages identically!")
    
    print("  [PASSED] GAE boundary mathematical identities confirmed!\n")


def test_gae_bias_variance_tradeoff():
    print("--- Test 3: GAE Bias-Variance Trade-Off Sweep ---")
    # Simulate noisy environment with an imperfect critic: V_approx(s) = V_true(s) + noise
    np.random.seed(42)
    T = 50
    gamma = 0.95
    n_trajectories = 500
    
    # True state values
    true_V = np.linspace(10.0, 0.0, T + 1)
    
    # Imperfect critic with systematic estimation error
    critic_bias = 2.0
    critic_noise_std = 0.5
    
    lambda_values = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    variances = []
    
    for lam in lambda_values:
        adv_trajectories = []
        for _ in range(n_trajectories):
            # Stochastic rewards around transitions
            rewards = true_V[:T] - gamma * true_V[1:] + np.random.normal(0, 1.0, size=T)
            values = true_V + np.random.normal(critic_bias, critic_noise_std, size=T + 1)
            values[-1] = 0.0
            dones = np.zeros(T)
            dones[-1] = 1.0
            
            adv = compute_gae_vectorized(rewards, values, dones, gamma, lam)
            adv_trajectories.append(adv[0])
            
        var_A0 = np.var(adv_trajectories)
        variances.append(var_A0)
        print(f"  lambda = {lam:4.2f} -> Empirical Variance of A_0: {var_A0:.4f}")
        
    # Variance must monotonically increase with lambda!
    for i in range(len(variances) - 1):
        assert variances[i] <= variances[i + 1] + 1e-4, "Variance must increase with lambda"
        
    print(f"  Variance ratio: lambda=1.0 is {variances[-1] / variances[0]:.2f}x higher than lambda=0.0!")
    print("  [PASSED] GAE bias-variance spectrum mathematically confirmed!\n")


def test_section_6_illustrations():
    print("--- Test 4: Section 6 Solved Illustrations Verification ---")
    
    # --- Illustration 1: Telescoping Sum & Numerical Cancellation ---
    gamma1 = 0.90
    R1 = np.array([1.0, 2.0, 5.0])
    V1 = np.array([3.0, 4.0, 5.0, 0.0])
    deltas1 = np.array([R1[t] + gamma1 * V1[t+1] - V1[t] for t in range(3)])
    expected_deltas1 = np.array([1.6, 2.5, 0.0])
    assert np.allclose(deltas1, expected_deltas1), f"Ill 1 deltas mismatch: {deltas1}"
    
    sum_td1 = deltas1[0] + gamma1 * deltas1[1] + (gamma1**2) * deltas1[2]
    G0 = R1[0] + gamma1 * R1[1] + (gamma1**2) * R1[2]
    assert np.isclose(sum_td1, 3.8500), f"Ill 1 TD sum expected 3.85, got {sum_td1}"
    assert np.isclose(G0 - V1[0], 3.8500), f"Ill 1 G0 - V0 expected 3.85, got {G0 - V1[0]}"
    print("  [PASSED] Illustration 1: Telescoping sum verified numerically (3.8500 == 3.8500).")
    
    # --- Illustration 2: 4-Step Trajectory Backward Recursion ---
    gamma2 = 0.99
    lam2 = 0.95
    gl2 = gamma2 * lam2
    R2 = np.array([1.5, -0.5, 2.0, 3.0])
    V2 = np.array([2.0, 2.5, 1.8, 3.2, 0.0])
    deltas2 = np.array([R2[t] + gamma2 * V2[t+1] - V2[t] for t in range(4)])
    expected_deltas2 = np.array([1.9750, -1.2180, 3.3680, -0.2000])
    assert np.allclose(deltas2, expected_deltas2, atol=1e-4), f"Ill 2 deltas mismatch: {deltas2}"
    
    gae2 = np.zeros(4)
    last = 0.0
    for t in reversed(range(4)):
        last = deltas2[t] + gl2 * last
        gae2[t] = last
    expected_gae2 = np.array([3.6422, 1.7727, 3.1799, -0.2000])
    assert np.allclose(gae2, expected_gae2, atol=1e-4), f"Ill 2 GAE mismatch: {gae2}"
    print(f"  [PASSED] Illustration 2: 4-step backward recursion verified {np.round(gae2, 4)}.")
    
    # --- Illustration 3: Comparison across lambdas ---
    for l, expected_A0 in [(0.0, 1.9750), (0.5, 2.1731), (0.95, 3.6422), (1.0, 3.8761)]:
        last = 0.0
        gae_l = np.zeros(4)
        for t in reversed(range(4)):
            last = deltas2[t] + (gamma2 * l) * last
            gae_l[t] = last
        assert np.isclose(gae_l[0], expected_A0, atol=1e-4), f"Ill 3 lambda={l} expected {expected_A0}, got {gae_l[0]}"
    # Verify exact match with Monte Carlo returns at lambda=1.0:
    G2 = np.zeros(4)
    curr_g = 0.0
    for t in reversed(range(4)):
        curr_g = R2[t] + gamma2 * curr_g
        G2[t] = curr_g
    mc_adv2 = G2 - V2[:4]
    assert np.allclose(mc_adv2, [3.8761, 1.9203, 3.1700, -0.2000], atol=1e-4)
    print("  [PASSED] Illustration 3: Lambda parameter sweep verified from 1-step TD to Monte Carlo.")
    
    # --- Illustration 4: Mini-Batch PPO Advantage Normalization ---
    adv4 = np.array([3.6422, 1.7727, 3.1799, -0.2000, -1.4500, 0.8500])
    mu4 = np.mean(adv4)
    assert np.isclose(mu4, 1.299133, atol=1e-5), f"Ill 4 mean mismatch: {mu4}"
    s4 = np.std(adv4, ddof=1)
    assert np.isclose(s4, 1.962568, atol=1e-5), f"Ill 4 sample std mismatch: {s4}"
    norm4 = (adv4 - mu4) / (s4 + 1e-8)
    expected_norm4 = np.array([1.1939, 0.2413, 0.9583, -0.7639, -1.4008, -0.2288])
    assert np.allclose(norm4, expected_norm4, atol=1e-4), f"Ill 4 normalized mismatch: {norm4}"
    assert np.isclose(np.mean(norm4), 0.0, atol=1e-7)
    assert np.isclose(np.std(norm4, ddof=1), 1.0, atol=1e-7)
    print("  [PASSED] Illustration 4: PPO advantage standardization verified (mean=0.0, std=1.0).")
    
    # --- Illustration 5: Episode Termination vs Truncation ---
    gamma5 = 0.95
    lam5 = 0.90
    gl5 = gamma5 * lam5
    R5 = np.array([1.0, 0.0, 2.0, -1.0, 3.0])
    V5 = np.array([1.5, 2.0, 1.0, 2.5, 3.0, 2.0])
    
    # Case A: Natural termination (d_5 = 1)
    deltas_A = np.array([R5[t] + gamma5 * (V5[t+1] if t < 4 else 0.0) - V5[t] for t in range(5)])
    gae_A = np.zeros(5)
    last = 0.0
    for t in reversed(range(5)):
        last = deltas_A[t] + gl5 * last
        gae_A[t] = last
    assert np.allclose(gae_A, [2.5632, 1.3605, 2.8192, -0.6500, 0.0000], atol=1e-4)
    
    # Case B: Time-limit truncation (d_5 = 0, bootstrap V(S5)=2.0)
    deltas_B = np.array([R5[t] + gamma5 * V5[t+1] - V5[t] for t in range(5)])
    gae_B = np.zeros(5)
    last = 0.0
    for t in reversed(range(5)):
        last = deltas_B[t] + gl5 * last
        gae_B[t] = last
    assert np.allclose(gae_B, [3.5785, 2.5480, 4.2082, 0.9745, 1.9000], atol=1e-4)
    
    # Difference shockwave decay:
    diff = gae_B - gae_A
    for t in range(5):
        theoretical_diff = (gamma5 * V5[-1]) * (gl5 ** (4 - t))
        assert np.isclose(diff[t], theoretical_diff, atol=1e-4)
    print("  [PASSED] Illustration 5: Termination vs truncation shockwave transmission confirmed.\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.15 GENERALIZED ADVANTAGE ESTIMATION (GAE) TESTS")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_gae_boundary_cases()
    test_gae_bias_variance_tradeoff()
    test_section_6_illustrations()
    
    print("=================================================================")
    print("ALL MODULE 11.15 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")

