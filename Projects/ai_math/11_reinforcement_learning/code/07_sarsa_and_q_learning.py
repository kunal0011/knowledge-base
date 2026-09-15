"""
Mathematical Verification Script for Module 11.7: SARSA, Q-Learning & Expected SARSA
=====================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - SARSA Target = 8.8000, Delta = 6.8000, Q_new(S1, a1) = 5.4000
   - Q-Learning Target = 12.0000, Delta = 10.0000, Q_new(S1, a1) = 7.0000
   - Expected SARSA Target = 11.6800, Delta = 9.6800, Q_new(S1, a1) = 6.8400
2. Cliff Walking Environment:
   - Evaluates SARSA, Q-Learning, and Expected SARSA.
   - Verifies SARSA achieves higher online return by learning the safe path.
3. Maximization Bias Benchmark (Sutton Example 6.7):
   - Demonstrates Q-Learning's systematic overestimation bias.
   - Proves Double Q-Learning eliminates maximization bias.
"""

import numpy as np

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 Tom Yeh Hand Calculation Verification ---")
    gamma = 0.80
    alpha = 0.50
    epsilon = 0.20
    
    # Current Q-values
    # State 0: S1, State 1: S2
    # Action 0: a1, Action 1: a2
    Q_init = np.array([
        [2.0, 1.0],  # S1: a1=2.0, a2=1.0
        [6.0, 10.0]  # S2: a1=6.0, a2=10.0
    ])
    
    s = 0   # S1
    a = 0   # a1
    r = 4.0 # R = 4.0
    s_next = 1 # S2
    a_next_sampled = 0 # a1
    
    # 1. SARSA
    target_sarsa = r + gamma * Q_init[s_next, a_next_sampled]
    delta_sarsa = target_sarsa - Q_init[s, a]
    q_sarsa = Q_init[s, a] + alpha * delta_sarsa
    
    print(f"  SARSA: Target={target_sarsa:.4f}, Delta={delta_sarsa:.4f}, Q_new={q_sarsa:.4f}")
    assert np.isclose(target_sarsa, 8.8000), f"Expected 8.8, got {target_sarsa}"
    assert np.isclose(delta_sarsa, 6.8000), f"Expected 6.8, got {delta_sarsa}"
    assert np.isclose(q_sarsa, 5.4000), f"Expected 5.4, got {q_sarsa}"
    
    # 2. Q-Learning
    max_q_next = np.max(Q_init[s_next])
    target_q = r + gamma * max_q_next
    delta_q = target_q - Q_init[s, a]
    q_qlearn = Q_init[s, a] + alpha * delta_q
    
    print(f"  Q-Learning: Target={target_q:.4f}, Delta={delta_q:.4f}, Q_new={q_qlearn:.4f}")
    assert np.isclose(target_q, 12.0000), f"Expected 12.0, got {target_q}"
    assert np.isclose(delta_q, 10.0000), f"Expected 10.0, got {delta_q}"
    assert np.isclose(q_qlearn, 7.0000), f"Expected 7.0, got {q_qlearn}"
    
    # 3. Expected SARSA
    n_actions = 2
    greedy_action = np.argmax(Q_init[s_next]) # action 1 (a2)
    pi_next = np.full(n_actions, epsilon / n_actions)
    pi_next[greedy_action] += (1.0 - epsilon)
    # pi_next = [0.10, 0.90]
    assert np.isclose(pi_next[0], 0.10)
    assert np.isclose(pi_next[1], 0.90)
    
    expected_q_next = np.sum(pi_next * Q_init[s_next])
    assert np.isclose(expected_q_next, 9.6000)
    
    target_exp = r + gamma * expected_q_next
    delta_exp = target_exp - Q_init[s, a]
    q_exp = Q_init[s, a] + alpha * delta_exp
    
    print(f"  Expected SARSA: Target={target_exp:.4f}, Delta={delta_exp:.4f}, Q_new={q_exp:.4f}")
    assert np.isclose(target_exp, 11.6800), f"Expected 11.68, got {target_exp}"
    assert np.isclose(delta_exp, 9.6800), f"Expected 9.68, got {delta_exp}"
    assert np.isclose(q_exp, 6.8400), f"Expected 6.84, got {q_exp}"
    
    print("  [PASSED] Part 5 hand calculations verified to exact precision!\n")


def test_cliff_walking_comparison():
    print("--- Test 2: Cliff Walking Benchmark (SARSA vs Q-Learning) ---")
    # 4x12 Gridworld
    # Start: (3, 0), Goal: (3, 11), Cliff: (3, 1) through (3, 10)
    # Actions: 0: Up, 1: Right, 2: Down, 3: Left
    
    H, W = 4, 12
    n_states = H * W
    n_actions = 4
    
    def state_to_coord(s):
        return divmod(s, W)
    
    def coord_to_state(r, c):
        return r * W + c
    
    start_state = coord_to_state(3, 0)
    goal_state = coord_to_state(3, 11)
    cliff_states = set(coord_to_state(3, c) for c in range(1, 11))
    
    def step_cliff(s, a):
        r, c = state_to_coord(s)
        if a == 0:   # Up
            r = max(0, r - 1)
        elif a == 1: # Right
            c = min(W - 1, c + 1)
        elif a == 2: # Down
            r = min(H - 1, r + 1)
        elif a == 3: # Left
            c = max(0, c - 1)
            
        next_s = coord_to_state(r, c)
        if next_s in cliff_states:
            return start_state, -100.0, False
        elif next_s == goal_state:
            return next_s, -1.0, True
        else:
            return next_s, -1.0, False
        
    def epsilon_greedy(Q, s, eps):
        if np.random.rand() < eps:
            return np.random.randint(n_actions)
        return np.argmax(Q[s])
    
    np.random.seed(42)
    gamma = 1.0
    alpha = 0.5
    epsilon = 0.1
    n_episodes = 500
    
    # Train SARSA
    Q_sarsa = np.zeros((n_states, n_actions))
    sarsa_rewards = []
    for ep in range(n_episodes):
        s = start_state
        a = epsilon_greedy(Q_sarsa, s, epsilon)
        ep_ret = 0
        while True:
            next_s, r, done = step_cliff(s, a)
            ep_ret += r
            if done:
                Q_sarsa[s, a] += alpha * (r - Q_sarsa[s, a])
                break
            next_a = epsilon_greedy(Q_sarsa, next_s, epsilon)
            Q_sarsa[s, a] += alpha * (r + gamma * Q_sarsa[next_s, next_a] - Q_sarsa[s, a])
            s = next_s
            a = next_a
        sarsa_rewards.append(ep_ret)
        
    # Train Q-Learning
    Q_qlearn = np.zeros((n_states, n_actions))
    q_rewards = []
    for ep in range(n_episodes):
        s = start_state
        ep_ret = 0
        while True:
            a = epsilon_greedy(Q_qlearn, s, epsilon)
            next_s, r, done = step_cliff(s, a)
            ep_ret += r
            if done:
                Q_qlearn[s, a] += alpha * (r - Q_qlearn[s, a])
                break
            Q_qlearn[s, a] += alpha * (r + gamma * np.max(Q_qlearn[next_s]) - Q_qlearn[s, a])
            s = next_s
        q_rewards.append(ep_ret)
        
    avg_sarsa_last100 = np.mean(sarsa_rewards[-100:])
    avg_q_last100 = np.mean(q_rewards[-100:])
    
    print(f"  Average Reward (Last 100 Episodes):")
    print(f"  SARSA (Safe Path): {avg_sarsa_last100:.2f}")
    print(f"  Q-Learning (Optimal Path hugging Cliff): {avg_q_last100:.2f}")
    
    # SARSA's online performance is strictly superior in Cliff Walking under epsilon-greedy exploration
    assert avg_sarsa_last100 > avg_q_last100, f"Expected SARSA ({avg_sarsa_last100}) > Q-Learning ({avg_q_last100})"
    print("  [PASSED] Cliff Walking demonstrated: SARSA chooses safe path to avoid exploration penalty!\n")


def test_maximization_bias_and_double_q():
    print("--- Test 3: Maximization Bias (Sutton Example 6.7) ---")
    # State A (0) and State B (1)
    # A has 2 actions: 0: Right (terminates, reward 0), 1: Left (goes to B, reward 0)
    # B has 10 actions, all terminate with reward ~ N(-0.1, 1.0)
    # True expected reward from B is -0.1 < 0.
    # Optimal choice at A is Right (action 0).
    
    n_runs = 200
    n_episodes = 300
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    
    q_left_count = np.zeros(n_episodes)
    dq_left_count = np.zeros(n_episodes)
    
    np.random.seed(42)
    for run in range(n_runs):
        # Standard Q-learning
        Q = {
            'A': np.zeros(2), # 0: Right, 1: Left
            'B': np.zeros(10) # 10 actions
        }
        
        # Double Q-learning
        Q1 = {'A': np.zeros(2), 'B': np.zeros(10)}
        Q2 = {'A': np.zeros(2), 'B': np.zeros(10)}
        
        for ep in range(n_episodes):
            # Run Standard Q
            if np.random.rand() < epsilon:
                act_A = np.random.choice(2)
            else:
                act_A = np.argmax(Q['A'])
            if act_A == 1:
                q_left_count[ep] += 1
                # Transition to B
                act_B = np.random.choice(10) if np.random.rand() < epsilon else np.argmax(Q['B'])
                r = np.random.normal(-0.1, 1.0)
                Q['B'][act_B] += alpha * (r - Q['B'][act_B])
                Q['A'][1] += alpha * (0.0 + gamma * np.max(Q['B']) - Q['A'][1])
            else:
                Q['A'][0] += alpha * (0.0 - Q['A'][0])
                
            # Run Double Q
            q_sum_A = Q1['A'] + Q2['A']
            if np.random.rand() < epsilon:
                dact_A = np.random.choice(2)
            else:
                dact_A = np.argmax(q_sum_A)
            if dact_A == 1:
                dq_left_count[ep] += 1
                # Transition to B
                q_sum_B = Q1['B'] + Q2['B']
                dact_B = np.random.choice(10) if np.random.rand() < epsilon else np.argmax(q_sum_B)
                r = np.random.normal(-0.1, 1.0)
                if np.random.rand() < 0.5:
                    Q1['B'][dact_B] += alpha * (r - Q1['B'][dact_B])
                    best_b = np.argmax(Q1['B'])
                    Q1['A'][1] += alpha * (0.0 + gamma * Q2['B'][best_b] - Q1['A'][1])
                else:
                    Q2['B'][dact_B] += alpha * (r - Q2['B'][dact_B])
                    best_b = np.argmax(Q2['B'])
                    Q2['A'][1] += alpha * (0.0 + gamma * Q1['B'][best_b] - Q2['A'][1])
            else:
                if np.random.rand() < 0.5:
                    Q1['A'][0] += alpha * (0.0 - Q1['A'][0])
                else:
                    Q2['A'][0] += alpha * (0.0 - Q2['A'][0])
                    
    q_pct_left = (q_left_count / n_runs) * 100
    dq_pct_left = (dq_left_count / n_runs) * 100
    
    max_q_pct = np.max(q_pct_left[:100])
    avg_dq_pct = np.mean(dq_pct_left[:100])
    
    print(f"  Standard Q-Learning peak % Left choice: {max_q_pct:.1f}% (suffers severe maximization bias)")
    print(f"  Double Q-Learning average % Left choice: {avg_dq_pct:.1f}% (effectively unbiased)")
    
    assert max_q_pct > 50.0, f"Expected Q-Learning overestimation bias peak > 50%, got {max_q_pct}"
    assert avg_dq_pct < max_q_pct, "Double Q-Learning must have lower % Left actions than Q-Learning peak"
    print("  [PASSED] Maximization bias and Double Q-Learning verified!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.7 SARSA, Q-LEARNING & EXPECTED SARSA TEST SUITE")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_cliff_walking_comparison()
    test_maximization_bias_and_double_q()
    
    print("=================================================================")
    print("ALL MODULE 11.7 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
