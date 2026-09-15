"""
Mathematical Verification Script for Module 11.6: Temporal-Difference Learning TD(0)
=====================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Episode 1: V = [0.0000, 0.0000, 5.0000]
   - Episode 2: V = [0.0000, 2.2500, 7.5000]
   - Episode 3 (Step 1): V(S_1) = 1.0125
2. Sutton's Example 6.4 (The AB Batch Experiment):
   - Batch Monte Carlo converges to V(A) = 0.0000, V(B) = 0.7500
   - Batch TD(0) converges to V(A) = 0.7500, V(B) = 0.7500
3. Sutton's 5-State Random Walk:
   - Analytical Bellman solution: V* = [1/6, 2/6, 3/6, 4/6, 5/6]
   - TD(0) convergence and RMSE reduction across episodes.
"""

import numpy as np

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 Tom Yeh Hand Calculation Verification ---")
    gamma = 0.90
    alpha = 0.50
    
    # 3 states: 0: S1, 1: S2, 2: S3
    V = np.zeros(3)
    
    # Episode definition: transitions (s, r, next_s)
    # None indicates terminal state
    episode = [
        (0, 0.0, 1),
        (1, 0.0, 2),
        (2, 10.0, None)
    ]
    
    def run_td_step(V_current, s, r, next_s):
        next_v = 0.0 if next_s is None else V_current[next_s]
        target = r + gamma * next_v
        delta = target - V_current[s]
        V_new = V_current.copy()
        V_new[s] += alpha * delta
        return V_new, target, delta
    
    # Episode 1
    for s, r, next_s in episode:
        V, target, delta = run_td_step(V, s, r, next_s)
        
    print(f"  After Episode 1: V = {V}")
    assert np.isclose(V[0], 0.0), f"Expected V(S1)=0.0, got {V[0]}"
    assert np.isclose(V[1], 0.0), f"Expected V(S2)=0.0, got {V[1]}"
    assert np.isclose(V[2], 5.0), f"Expected V(S3)=5.0, got {V[2]}"
    
    # Episode 2
    for s, r, next_s in episode:
        V, target, delta = run_td_step(V, s, r, next_s)
        
    print(f"  After Episode 2: V = {V}")
    assert np.isclose(V[0], 0.0), f"Expected V(S1)=0.0, got {V[0]}"
    assert np.isclose(V[1], 2.25), f"Expected V(S2)=2.25, got {V[1]}"
    assert np.isclose(V[2], 7.5), f"Expected V(S3)=7.5, got {V[2]}"
    
    # Episode 3: Step 1 (S1 -> S2)
    s, r, next_s = episode[0]
    V, target, delta = run_td_step(V, s, r, next_s)
    print(f"  After Episode 3 (Step 1): V(S1) = {V[0]:.4f}, Target = {target:.4f}, Delta = {delta:.4f}")
    assert np.isclose(target, 2.025), f"Expected Target=2.025, got {target}"
    assert np.isclose(delta, 2.025), f"Expected Delta=2.025, got {delta}"
    assert np.isclose(V[0], 1.0125), f"Expected V(S1)=1.0125, got {V[0]}"
    
    print("  [PASSED] Part 5 hand calculations verified to exact precision!\n")


def test_sutton_ab_batch_experiment():
    print("--- Test 2: Sutton Example 6.4 (Batch TD vs Batch MC on AB) ---")
    # States: A = 0, B = 1
    # Episode 1: A -> 0, B -> 0
    # Episodes 2-7: B -> 1 (6 times)
    # Episode 8: B -> 0 (1 time)
    # Gamma = 1.0
    
    episodes = [
        [(0, 0.0, 1), (1, 0.0, None)],
        [(1, 1.0, None)],
        [(1, 1.0, None)],
        [(1, 1.0, None)],
        [(1, 1.0, None)],
        [(1, 1.0, None)],
        [(1, 1.0, None)],
        [(1, 0.0, None)]
    ]
    
    # 1. Batch Monte Carlo:
    # A appears once with return 0.0 -> V_MC(A) = 0.0
    # B appears 8 times: 6 times return 1.0, 2 times return 0.0 -> V_MC(B) = 6/8 = 0.75
    returns_A = [0.0]
    returns_B = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    
    V_MC_A = np.mean(returns_A)
    V_MC_B = np.mean(returns_B)
    
    print(f"  Batch MC: V(A) = {V_MC_A:.4f}, V(B) = {V_MC_B:.4f}")
    assert np.isclose(V_MC_A, 0.0)
    assert np.isclose(V_MC_B, 0.75)
    
    # 2. Batch TD(0):
    # Repeatedly update V using small alpha over all 8 episodes until convergence
    V_TD = np.zeros(2)
    alpha = 0.005
    gamma = 1.0
    
    for epoch in range(2000):
        delta_sum = np.zeros(2)
        for ep in episodes:
            for s, r, next_s in ep:
                next_val = 0.0 if next_s is None else V_TD[next_s]
                td_err = r + gamma * next_val - V_TD[s]
                delta_sum[s] += td_err
        V_TD += alpha * delta_sum
        
    print(f"  Batch TD: V(A) = {V_TD[0]:.4f}, V(B) = {V_TD[1]:.4f}")
    assert np.isclose(V_TD[0], 0.75, atol=1e-3), f"Expected V_TD(A)=0.75, got {V_TD[0]}"
    assert np.isclose(V_TD[1], 0.75, atol=1e-3), f"Expected V_TD(B)=0.75, got {V_TD[1]}"
    
    print("  [PASSED] Sutton AB example verified: Batch TD finds certainty-equivalence solution!\n")


def test_sutton_random_walk_convergence():
    print("--- Test 3: Sutton's 5-State Random Walk Convergence ---")
    # States: 0(A), 1(B), 2(C), 3(D), 4(E)
    # Undiscounted: gamma = 1.0
    # Left of A: terminal with R = 0
    # Right of E: terminal with R = 1
    # Equal probability left or right (0.5 each)
    # True values: [1/6, 2/6, 3/6, 4/6, 5/6]
    true_V = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    
    n_states = 5
    gamma = 1.0
    alpha = 0.1
    
    np.random.seed(42)
    n_runs = 50
    n_episodes = 100
    
    rmse_history = np.zeros(n_episodes)
    
    for run in range(n_runs):
        V = np.full(n_states, 0.5) # initialized to 0.5 like Sutton
        for ep in range(n_episodes):
            s = 2 # start in center state C
            while True:
                action = -1 if np.random.rand() < 0.5 else 1
                next_s = s + action
                if next_s == -1: # Left terminal
                    r = 0.0
                    target = r
                    V[s] += alpha * (target - V[s])
                    break
                elif next_s == 5: # Right terminal
                    r = 1.0
                    target = r
                    V[s] += alpha * (target - V[s])
                    break
                else:
                    r = 0.0
                    target = r + gamma * V[next_s]
                    V[s] += alpha * (target - V[s])
                    s = next_s
            rmse = np.sqrt(np.mean((V - true_V) ** 2))
            rmse_history[ep] += rmse
            
    rmse_history /= n_runs
    print(f"  Initial RMSE: {rmse_history[0]:.4f}")
    print(f"  Final RMSE after {n_episodes} episodes: {rmse_history[-1]:.4f}")
    print(f"  True Values: {true_V}")
    print(f"  Learned Values (sample): {np.round(V, 4)}")
    
    assert rmse_history[-1] < rmse_history[0], "TD(0) RMSE must decrease with training"
    assert rmse_history[-1] < 0.15, f"Expected final RMSE < 0.15, got {rmse_history[-1]}"
    print("  [PASSED] Sutton 5-State Random Walk convergence confirmed!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.6 TEMPORAL-DIFFERENCE LEARNING TEST SUITE")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_sutton_ab_batch_experiment()
    test_sutton_random_walk_convergence()
    
    print("=================================================================")
    print("ALL MODULE 11.6 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
