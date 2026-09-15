"""
The Bellman Equations: State-Value and Action-Value Functions Verification
========================================================================
Module 11: Reinforcement Learning - Chapter 11.3

This script verifies:
1. Part 5 Tom Yeh "AI by Hand" Exact Numerical Parity:
   - Induced matrix P_pi and reward r_pi.
   - Resolvent inverse matrix (I - gamma * P_pi)^(-1) matching [4/3, 2/3; 0, 2].
   - Value vector v_pi = [7/3, 1.0].
   - Action-value function Q_pi(s, a) across all 4 state-action pairs.
   - Strict consistency: sum_a pi(a|s) Q(s, a) == V(s) to < 10^-14.
2. Neumann Series Expansion Convergence Rate.
3. Iterative Bellman Expectation Operator Fixed Point Convergence.
4. Non-Linear Bellman Optimality Operator Fixed Point Iteration (Value Iteration)
   and strict policy dominance V^*(s) >= V^pi(s).
"""

import math
import numpy as np

def test_part5_hand_arithmetic_parity():
    print("--- Test 1: Part 5 Tom Yeh Bellman Matrix Inversion Parity ---")
    gamma = 0.50
    
    # Transition dynamics & Rewards
    P_a1 = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)
    P_a2 = np.array([[0.2, 0.8], [0.0, 1.0]], dtype=np.float64)
    
    r_a1 = np.array([1.0, -1.0], dtype=np.float64)
    r_a2 = np.array([2.0, 0.5], dtype=np.float64)
    
    # Policy: pi(a | s)
    pi = np.array([
        [0.5, 0.5],  # S1: a1, a2
        [0.0, 1.0]   # S2: a1, a2
    ], dtype=np.float64)
    
    # 1. Induced P_pi and r_pi
    P_pi = np.zeros((2, 2), dtype=np.float64)
    r_pi = np.zeros(2, dtype=np.float64)
    for s in range(2):
        P_pi[s] = pi[s, 0] * P_a1[s] + pi[s, 1] * P_a2[s]
        r_pi[s] = pi[s, 0] * r_a1[s] + pi[s, 1] * r_a2[s]
        
    expected_P_pi = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
    expected_r_pi = np.array([1.5, 0.5], dtype=np.float64)
    
    print("Computed Induced P_pi:\n", P_pi)
    print("Computed Induced r_pi: ", r_pi)
    assert np.allclose(P_pi, expected_P_pi, atol=1e-14)
    assert np.allclose(r_pi, expected_r_pi, atol=1e-14)
    
    # 2. Resolvent Matrix (I - gamma * P_pi)^(-1)
    A = np.eye(2) - gamma * P_pi
    expected_A = np.array([[0.75, -0.25], [0.0, 0.50]], dtype=np.float64)
    assert np.allclose(A, expected_A, atol=1e-14)
    
    A_inv = np.linalg.inv(A)
    expected_A_inv = np.array([[4.0 / 3.0, 2.0 / 3.0], [0.0, 2.0]], dtype=np.float64)
    print("Computed (I - gamma * P_pi)^(-1):\n", A_inv)
    assert np.allclose(A_inv, expected_A_inv, atol=1e-14)
    
    # 3. State-Value Vector v_pi
    v_pi = A_inv @ r_pi
    expected_v_pi = np.array([7.0 / 3.0, 1.0], dtype=np.float64)
    print(f"Computed v_pi: [{v_pi[0]:.4f}, {v_pi[1]:.4f}] (Hand: [7/3 = 2.3333, 1.0000])")
    assert np.allclose(v_pi, expected_v_pi, atol=1e-14)
    
    # 4. Action-Value Q_pi(s, a)
    Q = np.zeros((2, 2), dtype=np.float64)
    Q[0, 0] = r_a1[0] + gamma * np.dot(P_a1[0], v_pi)  # Q(S1, a1)
    Q[0, 1] = r_a2[0] + gamma * np.dot(P_a2[0], v_pi)  # Q(S1, a2)
    Q[1, 0] = r_a1[1] + gamma * np.dot(P_a1[1], v_pi)  # Q(S2, a1)
    Q[1, 1] = r_a2[1] + gamma * np.dot(P_a2[1], v_pi)  # Q(S2, a2)
    
    expected_Q = np.array([
        [6.1 / 3.0, 7.9 / 3.0],
        [-0.7 / 3.0, 1.0]
    ], dtype=np.float64)
    
    print("Computed Q_pi(s, a):\n", Q)
    print("Hand Expected Q_pi(s, a):\n", expected_Q)
    assert np.allclose(Q, expected_Q, atol=1e-14)
    
    # 5. Weighted Consistency: sum_a pi(a|s) Q(s, a) == V(s)
    v_reconstructed = np.sum(pi * Q, axis=1)
    print("Reconstructed V from Q:", v_reconstructed)
    assert np.allclose(v_reconstructed, v_pi, atol=1e-14)
    
    print("✓ Part 5 Hand Arithmetic verified to exact machine precision!\n")


def test_neumann_series_convergence():
    print("--- Test 2: Neumann Series Expansion Convergence ---")
    gamma = 0.5
    P = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
    M = gamma * P
    
    # Analytical exact inverse
    exact_inv = np.linalg.inv(np.eye(2) - M)
    
    # Power series accumulation
    running_sum = np.zeros((2, 2), dtype=np.float64)
    M_k = np.eye(2, dtype=np.float64)
    
    for k in range(12):
        running_sum += M_k
        M_k = M_k @ M
        err = np.max(np.abs(running_sum - exact_inv))
        theoretical_bound = (gamma ** (k + 1)) / (1.0 - gamma)
        if k in [0, 1, 2, 5, 10]:
            print(f"Step k={k:2d}: Norm Error = {err:.2e}, Bound = {theoretical_bound:.2e}")
        assert err <= theoretical_bound + 1e-15, f"Error exceeded Neumann bound at step {k}"
        
    print("✓ Neumann series exponential convergence verified!\n")


def test_bellman_expectation_iteration():
    print("--- Test 3: Iterative Bellman Expectation Fixed-Point Convergence ---")
    gamma = 0.50
    P_pi = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
    r_pi = np.array([1.5, 0.5], dtype=np.float64)
    v_exact = np.array([7.0 / 3.0, 1.0], dtype=np.float64)
    
    # Start from arbitrary initial value vector
    v = np.array([10.0, -5.0], dtype=np.float64)
    
    for it in range(30):
        v = r_pi + gamma * (P_pi @ v)
        
    diff = np.max(np.abs(v - v_exact))
    print(f"Iterated Value: [{v[0]:.8f}, {v[1]:.8f}]")
    print(f"Exact Value:    [{v_exact[0]:.8f}, {v_exact[1]:.8f}]")
    print(f"Residual Error after 30 steps: {diff:.2e}")
    assert diff < 1e-8, "Bellman expectation iteration failed to converge"
    print("✓ Bellman expectation fixed-point convergence confirmed!\n")


def test_bellman_optimality_operator():
    print("--- Test 4: Bellman Optimality Operator & Policy Dominance ---")
    gamma = 0.50
    P_a1 = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)
    P_a2 = np.array([[0.2, 0.8], [0.0, 1.0]], dtype=np.float64)
    r_a1 = np.array([1.0, -1.0], dtype=np.float64)
    r_a2 = np.array([2.0, 0.5], dtype=np.float64)
    
    P_all = [P_a1, P_a2]
    r_all = [r_a1, r_a2]
    
    v_opt = np.zeros(2, dtype=np.float64)
    
    # Fixed point iteration of Bellman Optimality Operator T*
    for it in range(50):
        # Q(s, a) = r(s, a) + gamma * P(s, a) @ v
        Q_s1_a1 = r_a1[0] + gamma * np.dot(P_a1[0], v_opt)
        Q_s1_a2 = r_a2[0] + gamma * np.dot(P_a2[0], v_opt)
        Q_s2_a1 = r_a1[1] + gamma * np.dot(P_a1[1], v_opt)
        Q_s2_a2 = r_a2[1] + gamma * np.dot(P_a2[1], v_opt)
        
        v_next = np.array([
            max(Q_s1_a1, Q_s1_a2),
            max(Q_s2_a1, Q_s2_a2)
        ])
        if np.max(np.abs(v_next - v_opt)) < 1e-12:
            break
        v_opt = v_next
        
    v_pi_part5 = np.array([7.0 / 3.0, 1.0], dtype=np.float64)
    print(f"Optimal Value V*: [{v_opt[0]:.4f}, {v_opt[1]:.4f}]")
    print(f"Policy Value V^pi:[{v_pi_part5[0]:.4f}, {v_pi_part5[1]:.4f}]")
    
    # Optimal value function must be strictly greater than or equal to arbitrary policy value everywhere
    assert np.all(v_opt >= v_pi_part5 - 1e-7), "Optimal value V* must dominate policy value V^pi"
    assert v_opt[0] > v_pi_part5[0] + 0.1, "V*(S1) should strictly exceed suboptimal policy value"
    print("✓ Bellman optimality contraction and policy dominance verified!\n")


if __name__ == "__main__":
    test_part5_hand_arithmetic_parity()
    test_neumann_series_convergence()
    test_bellman_expectation_iteration()
    test_bellman_optimality_operator()
    print("=========================================================")
    print("ALL MODULE 11 CHAPTER 11.3 VERIFICATIONS PASSED (100%)")
    print("=========================================================")
