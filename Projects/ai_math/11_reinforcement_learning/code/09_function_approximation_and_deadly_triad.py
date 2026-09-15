"""
Mathematical Verification Script for Module 11.9: Function Approximation & The Deadly Triad
============================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Exponential divergence of Tsitsiklis & Van Roy 2-state counterexample:
     w_0 = 1.0000, w_1 = 1.4000, w_2 = 1.9600, w_3 = 2.7440, w_4 = 3.8416, w_5 = 5.37824
2. Linear On-Policy TD(0) Convergence to Projected Bellman Fixed Point:
   - Verifies w_t -> w_TD = A^{-1} b = (Phi^T D (I - gamma P) Phi)^{-1} Phi^T D R.
3. Baird's Counterexample Divergence:
   - Proves that off-policy linear semi-gradient updates cause ||w_t|| to explode toward infinity.
"""

import numpy as np

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 Hand Calculation Verification (2-State Divergence) ---")
    gamma = 0.90
    alpha = 0.50
    
    phi_S1 = 1.0
    phi_S2 = 2.0
    
    w = 1.0000
    expected_weights = [1.0, 1.4000, 1.9600, 2.7440, 3.8416, 5.37824]
    
    print(f"  Iteration 0: w = {w:.5f}")
    assert np.isclose(w, expected_weights[0])
    
    for t in range(1, 6):
        # Target = 0 + gamma * (w * phi_S2)
        target = 0.0 + gamma * (w * phi_S2)
        v_est = w * phi_S1
        delta = target - v_est
        w = w + alpha * delta * phi_S1
        print(f"  Iteration {t}: Target = {target:.5f}, Delta = {delta:.5f}, New w = {w:.5f}")
        assert np.isclose(w, expected_weights[t]), f"At step {t}, expected {expected_weights[t]}, got {w}"
        
    print("  [PASSED] Part 5 hand calculation verified to exact precision!\n")


def test_linear_td_projected_fixed_point():
    print("--- Test 2: Linear TD(0) Convergence to Analytical Fixed Point ---")
    # 3-State Markov Chain: 0 -> 1 -> 2 -> 0 (cyclic)
    # Transition matrix P:
    P = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    R = np.array([1.0, 2.0, 3.0])
    gamma = 0.8
    
    # Feature matrix Phi: 3 states, 2 features
    Phi = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0]
    ])
    
    # Stationary distribution D: uniform 1/3 each
    d = np.array([1/3, 1/3, 1/3])
    D = np.diag(d)
    
    # Analytical A and b matrices
    I = np.eye(3)
    A = Phi.T @ D @ (I - gamma * P) @ Phi
    b = Phi.T @ D @ R
    
    w_TD_analytical = np.linalg.solve(A, b)
    print(f"  Analytical w_TD = {w_TD_analytical}")
    
    # Run online linear TD(0)
    w = np.zeros(2)
    np.random.seed(42)
    s = 0
    n_steps = 100000
    
    for t in range(1, n_steps + 1):
        alpha_t = 1.0 / (100.0 + t**0.6) # Robbins-Monro schedule
        r = R[s]
        next_s = np.random.choice(3, p=P[s])
        
        target = r + gamma * np.dot(Phi[next_s], w)
        v_s = np.dot(Phi[s], w)
        delta = target - v_s
        
        w += alpha_t * delta * Phi[s]
        s = next_s
        
    print(f"  Learned linear TD w = {w}")
    error = np.linalg.norm(w - w_TD_analytical)
    print(f"  L2 distance to analytical fixed point: {error:.4e}")
    assert error < 0.05, f"Expected error < 0.05, got {error}"
    print("  [PASSED] Linear TD(0) converged to projected Bellman fixed point!\n")


def test_bairds_counterexample_divergence():
    print("--- Test 3: Baird's Counterexample Off-Policy Divergence ---")
    # 7 states, feature dimension 14 (or standard 8 features)
    # Standard Baird: 7 states. States 1-6 can transition to any 1-6 with dashed, or 7 with solid.
    # Feature representation:
    # State 1-6: [2*e_s, 1] -> dim = 8
    # State 7: [e_s, 2]
    # Under behavior policy, solid action is taken with prob 1/7, dashed with 6/7.
    # Target policy: always solid action (always transitions to state 7).
    # All rewards R = 0. gamma = 0.99.
    
    n_states = 7
    d = 8 # 8 features
    
    Phi = np.zeros((n_states, d))
    for s in range(6):
        Phi[s, s] = 2.0
        Phi[s, 7] = 1.0
    Phi[6, :6] = 1.0
    Phi[6, 7] = 2.0
    
    gamma = 0.99
    alpha = 0.01
    
    # Initial weights: [1, 1, 1, 1, 1, 1, 10, 1]
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
    
    initial_norm = np.linalg.norm(w)
    
    np.random.seed(42)
    n_steps = 2000
    norm_history = []
    
    for step in range(n_steps):
        # Behavior policy: choose state s uniformly from {0, ..., 6}
        s = np.random.randint(n_states)
        # Action under target policy always transitions to state 7 (index 6)
        next_s = 6
        r = 0.0
        
        # Semi-gradient update for target policy:
        target = r + gamma * np.dot(Phi[next_s], w)
        delta = target - np.dot(Phi[s], w)
        w += alpha * delta * Phi[s]
        
        if step % 200 == 0:
            norm_history.append(np.linalg.norm(w))
            
    final_norm = np.linalg.norm(w)
    print(f"  Initial ||w||: {initial_norm:.2f}")
    print(f"  Final ||w|| after {n_steps} steps: {final_norm:.2e}")
    print(f"  Weight growth factor: {final_norm / initial_norm:.2e}x")
    
    # Baird's counterexample must diverge (norm strictly increases by orders of magnitude)
    assert final_norm > initial_norm * 50.0, f"Expected divergent growth (>50x), got {final_norm / initial_norm:.2f}x"
    print("  [PASSED] Baird's counterexample diverged as mathematically predicted!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.9 FUNCTION APPROXIMATION & DEADLY TRIAD TESTS")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_linear_td_projected_fixed_point()
    test_bairds_counterexample_divergence()
    
    print("=================================================================")
    print("ALL MODULE 11.9 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
