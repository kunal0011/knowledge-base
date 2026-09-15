"""
Dynamic Programming & The Banach Contraction Mapping Theorem Verification
========================================================================
Module 11: Reinforcement Learning - Chapter 11.4

This script verifies:
1. Part 5 Tom Yeh "AI by Hand" Exact Numerical Parity:
   - Value Iteration steps V_1, V_2, and V_3 on 3-state MDP.
   - Exact contraction ratios ||Delta_1|| / ||Delta_0|| == 0.5 == gamma.
   - ||Delta_2|| <= gamma * ||Delta_1||.
2. Full Implementation of Policy Iteration (Evaluation + Improvement).
3. Full Implementation of Value Iteration (Contraction sweep to epsilon).
4. Exact Equivalence: Proving Policy Iteration and Value Iteration find identical
   optimal value function V* and optimal policy pi*.
5. Strict Monotonicity Proof Check: V^{pi_{k+1}} >= V^{pi_k} everywhere.
"""

import math
import numpy as np

def test_part5_hand_arithmetic_parity():
    print("--- Test 1: Part 5 Tom Yeh Value Iteration Parity ---")
    gamma = 0.50
    
    # 3 states, 2 actions
    # Action 0: a1, Action 1: a2
    P_a1 = np.array([
        [0.7, 0.3, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    P_a2 = np.array([
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
        [0.4, 0.4, 0.2]
    ], dtype=np.float64)
    
    r_a1 = np.array([4.0, -1.0, 0.0], dtype=np.float64)
    r_a2 = np.array([1.0, 2.0, 6.0], dtype=np.float64)
    
    # V_0 = [0, 0, 0]
    V_0 = np.zeros(3, dtype=np.float64)
    
    # Step 1: V_1
    Q_0_a1 = r_a1 + gamma * (P_a1 @ V_0)
    Q_0_a2 = r_a2 + gamma * (P_a2 @ V_0)
    V_1 = np.maximum(Q_0_a1, Q_0_a2)
    expected_V1 = np.array([4.0, 2.0, 6.0], dtype=np.float64)
    print("Computed V_1:", V_1)
    assert np.allclose(V_1, expected_V1, atol=1e-14)
    
    delta_0 = np.max(np.abs(V_1 - V_0))
    print(f"||Delta_0||_inf: {delta_0:.4f} (Hand: 6.0000)")
    assert abs(delta_0 - 6.0) < 1e-14
    
    # Step 2: V_2
    Q_1_a1 = r_a1 + gamma * (P_a1 @ V_1)
    Q_1_a2 = r_a2 + gamma * (P_a2 @ V_1)
    V_2 = np.maximum(Q_1_a1, Q_1_a2)
    expected_V2 = np.array([5.7, 5.0, 7.8], dtype=np.float64)
    print("Computed V_2:", V_2)
    assert np.allclose(V_2, expected_V2, atol=1e-14)
    
    delta_1 = np.max(np.abs(V_2 - V_1))
    print(f"||Delta_1||_inf: {delta_1:.4f} (Hand: 3.0000)")
    assert abs(delta_1 - 3.0) < 1e-14
    
    # Contraction check 1
    ratio_1 = delta_1 / delta_0
    print(f"Contraction Ratio 1: {ratio_1:.4f} (Exact gamma: {gamma:.4f})")
    assert abs(ratio_1 - gamma) < 1e-14
    
    # Step 3: V_3
    Q_2_a1 = r_a1 + gamma * (P_a1 @ V_2)
    Q_2_a2 = r_a2 + gamma * (P_a2 @ V_2)
    V_3 = np.maximum(Q_2_a1, Q_2_a2)
    expected_V3 = np.array([6.745, 5.90, 8.92], dtype=np.float64)
    print("Computed V_3:", V_3)
    assert np.allclose(V_3, expected_V3, atol=1e-14)
    
    delta_2 = np.max(np.abs(V_3 - V_2))
    print(f"||Delta_2||_inf: {delta_2:.4f} (Hand: 1.1200)")
    assert abs(delta_2 - 1.1200) < 1e-14
    
    # Contraction check 2
    assert delta_2 <= gamma * delta_1 + 1e-14
    print("✓ Part 5 Hand Arithmetic verified to exact machine precision!\n")


def value_iteration(P, r, gamma, tol=1e-10):
    """
    P shape: (n_states, n_actions, n_states)
    r shape: (n_states, n_actions)
    """
    n_states, n_actions = r.shape
    v = np.zeros(n_states)
    
    while True:
        # Compute Q(s, a) for all s, a
        Q = r + gamma * np.einsum('san,n->sa', P, v)
        v_next = np.max(Q, axis=1)
        if np.max(np.abs(v_next - v)) < tol:
            break
        v = v_next
        
    policy = np.argmax(Q, axis=1)
    return v, policy


def policy_iteration(P, r, gamma):
    """
    Solves MDP using Policy Iteration.
    Returns: V_opt, pi_opt, list of evaluated value vectors per iteration.
    """
    n_states, n_actions = r.shape
    # Initial policy: all action 0
    pi = np.zeros(n_states, dtype=int)
    history_V = []
    
    while True:
        # 1. Policy Evaluation via exact matrix inversion
        # P_pi: (n_states, n_states), r_pi: (n_states,)
        P_pi = P[np.arange(n_states), pi, :]
        r_pi = r[np.arange(n_states), pi]
        
        A = np.eye(n_states) - gamma * P_pi
        v = np.linalg.solve(A, r_pi)
        history_V.append(v.copy())
        
        # 2. Policy Improvement
        Q = r + gamma * np.einsum('san,n->sa', P, v)
        pi_new = np.argmax(Q, axis=1)
        
        # 3. Termination check
        if np.array_equal(pi, pi_new):
            break
        pi = pi_new
        
    return v, pi, history_V


def test_dp_algorithms_parity_and_monotonicity():
    print("--- Test 2: Policy Iteration vs Value Iteration Parity & Monotonicity ---")
    gamma = 0.50
    
    # Assemble MDP tensors
    P_a1 = np.array([
        [0.7, 0.3, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    P_a2 = np.array([
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
        [0.4, 0.4, 0.2]
    ], dtype=np.float64)
    
    P = np.stack([P_a1, P_a2], axis=1) # (3, 2, 3)
    r = np.array([
        [4.0, 1.0],
        [-1.0, 2.0],
        [0.0, 6.0]
    ], dtype=np.float64)
    
    # 1. Run Value Iteration
    v_vi, pi_vi = value_iteration(P, r, gamma)
    print(f"Value Iteration V*:  [{v_vi[0]:.6f}, {v_vi[1]:.6f}, {v_vi[2]:.6f}]")
    print(f"Value Iteration pi*: {pi_vi} (0: a1, 1: a2)")
    
    # 2. Run Policy Iteration
    v_pi, pi_pi, hist_V = policy_iteration(P, r, gamma)
    print(f"Policy Iteration V*: [{v_pi[0]:.6f}, {v_pi[1]:.6f}, {v_pi[2]:.6f}]")
    print(f"Policy Iteration pi*:{pi_pi}")
    
    # Assert exact equivalence
    assert np.allclose(v_vi, v_pi, atol=1e-8), "Value Iteration and Policy Iteration values diverge"
    assert np.array_equal(pi_vi, pi_pi), "Value Iteration and Policy Iteration policies diverge"
    assert np.array_equal(pi_vi, [0, 1, 1]), f"Expected policy [a1, a2, a2] -> [0, 1, 1], got {pi_vi}"
    
    # 3. Check Policy Iteration Strict Monotonicity
    print(f"\nChecking Policy Iteration Monotonicity across {len(hist_V)} iterations:")
    for k in range(len(hist_V) - 1):
        diff = hist_V[k + 1] - hist_V[k]
        print(f"  Step {k} -> {k+1}: V_diff = [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}]")
        assert np.all(diff >= -1e-12), "Policy improvement failed monotonicity!"
        assert np.any(diff > 1e-6), "Policy improvement should strictly improve at least one state before convergence"
        
    print("✓ Policy Iteration vs Value Iteration parity & strict monotonicity verified!\n")


if __name__ == "__main__":
    test_part5_hand_arithmetic_parity()
    test_dp_algorithms_parity_and_monotonicity()
    print("=========================================================")
    print("ALL MODULE 11 CHAPTER 11.4 VERIFICATIONS PASSED (100%)")
    print("=========================================================")
