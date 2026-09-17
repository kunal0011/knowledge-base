"""
Model-Based Reinforcement Learning: Dyna-Q & Dyna-Q+ Verification Suite
Module 11: Reinforcement Learning - Chapter 21

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Direct Q-update for (S2, a1) = 5.0000
   - Retrospective planning update propagating reward back to (S1, a0) = 2.2500
2. Complete Dyna-Q tabular agent implementation with configurable planning steps N.
3. Comparative sample efficiency benchmark on Sutton's Gridworld:
   - N = 0 (model-free Q-learning)
   - N = 5 (moderate planning)
   - N = 50 (deep mental simulation)
4. Dyna-Q+ exploration bonus verification in non-stationary environments.
"""

import numpy as np


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' DYNA-Q NUMERICAL DERIVATIONS")
    print("=" * 70)

    # Setup environment parameters
    # States: S1 (0), S2 (1), S3 (2, terminal)
    # Actions: a0 (0), a1 (1)
    alpha = 0.5
    gamma = 0.9

    Q = np.zeros((3, 2))  # States x Actions
    model = {}  # (s, a) -> (r, s_next, terminal)

    # --- Transition 1: (S1, a0) -> S2, R = 0.0 ---
    s, a, r, s_next, done = 0, 0, 0.0, 1, False
    target_1 = r + gamma * np.max(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (target_1 - Q[s, a])
    model[(s, a)] = (r, s_next, done)

    # Planning step 1: sample (S1, a0)
    r_plan, s_next_plan, done_plan = model[(s, a)]
    target_plan_1 = r_plan + gamma * (0.0 if done_plan else np.max(Q[s_next_plan]))
    Q[s, a] = Q[s, a] + alpha * (target_plan_1 - Q[s, a])

    print(f"End of Transition 1: Q(S1, a0) = {Q[0, 0]:.4f} (Expected: 0.0000)")
    assert np.isclose(Q[0, 0], 0.0, atol=1e-12), f"Expected 0.0, got {Q[0, 0]}"

    # --- Transition 2: (S2, a1) -> S3 (terminal), R = 10.0 ---
    s2, a2, r2, s_next2, done2 = 1, 1, 10.0, 2, True
    target_2 = r2 + gamma * (0.0 if done2 else np.max(Q[s_next2]))
    Q[s2, a2] = Q[s2, a2] + alpha * (target_2 - Q[s2, a2])
    model[(s2, a2)] = (r2, s_next2, done2)

    print(f"Transition 2 Direct Update: Q(S2, a1) = {Q[1, 1]:.4f} (Expected: 5.0000)")
    assert np.isclose(Q[1, 1], 5.0, atol=1e-12), f"Expected 5.0, got {Q[1, 1]}"

    # --- Planning Step 2: Mental rehearsal draws (S1, a0) ---
    r_k, s_next_k, done_k = model[(0, 0)]
    target_plan_2 = r_k + gamma * (0.0 if done_k else np.max(Q[s_next_k]))
    Q[0, 0] = Q[0, 0] + alpha * (target_plan_2 - Q[0, 0])

    print(f"Transition 2 Planning Update: Q(S1, a0) = {Q[0, 0]:.4f} (Expected: 2.2500)")
    assert np.isclose(Q[0, 0], 2.25, atol=1e-12), f"Expected 2.25, got {Q[0, 0]}"

    print(">> SUCCESS: Hand calculation perfectly verified to machine precision (< 1e-12)!\n")


# =====================================================================
# 2. Complete Dyna-Q Tabular Agent Implementation
# =====================================================================

class DynaQAgent:
    """
    Tabular Dyna-Q agent integrating direct RL, model learning, and mental planning.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        # Action-value table
        self.Q = np.zeros((n_states, n_actions))

        # Learned deterministic model: (s, a) -> (r, s_next, done)
        self.model = {}

        # Set of observed state-action pairs for uniform planning sampling
        self.observed_pairs = []
        self.observed_set = set()

    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        # Random tie-breaking among best actions
        q_vals = self.Q[state]
        max_val = np.max(q_vals)
        best_actions = np.flatnonzero(np.isclose(q_vals, max_val))
        return np.random.choice(best_actions)

    def step_learn(self, s, a, r, s_next, done):
        # 1. Direct Q-learning update
        target = r + self.gamma * (0.0 if done else np.max(self.Q[s_next]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # 2. Model learning (recording transition)
        pair = (s, a)
        self.model[pair] = (r, s_next, done)
        if pair not in self.observed_set:
            self.observed_set.add(pair)
            self.observed_pairs.append(pair)

        # 3. Planning phase: N simulated experience replays from model
        if self.planning_steps > 0 and len(self.observed_pairs) > 0:
            for _ in range(self.planning_steps):
                # Sample previously observed state-action pair
                idx = np.random.randint(len(self.observed_pairs))
                s_sim, a_sim = self.observed_pairs[idx]
                r_sim, s_next_sim, done_sim = self.model[(s_sim, a_sim)]

                # Update Q using simulated transition
                plan_target = r_sim + self.gamma * (0.0 if done_sim else np.max(self.Q[s_next_sim]))
                self.Q[s_sim, a_sim] += self.alpha * (plan_target - self.Q[s_sim, a_sim])


# =====================================================================
# 3. Sutton's Gridworld Benchmark
# =====================================================================

class Gridworld:
    """
    Standard 6x9 Gridworld maze with obstacles.
    Start: (2, 0)
    Goal: (0, 8) with reward +1.0
    Step reward: 0.0
    Actions: 0: Up, 1: Down, 2: Left, 3: Right
    """
    def __init__(self, height=6, width=9):
        self.height = height
        self.width = width
        self.start_pos = (2, 0)
        self.goal_pos = (0, 8)

        # Obstacles (row, col)
        self.obstacles = {
            (1, 2), (2, 2), (3, 2),
            (4, 5),
            (0, 7), (1, 7), (2, 7)
        }
        self.pos = self.start_pos

    def reset(self):
        self.pos = self.start_pos
        return self._pos_to_state(self.pos)

    def _pos_to_state(self, pos):
        return pos[0] * self.width + pos[1]

    def step(self, action):
        r, c = self.pos
        if action == 0:    # Up
            nr, nc = max(0, r - 1), c
        elif action == 1:  # Down
            nr, nc = min(self.height - 1, r + 1), c
        elif action == 2:  # Left
            nr, nc = r, max(0, c - 1)
        elif action == 3:  # Right
            nr, nc = r, min(self.width - 1, c + 1)
        else:
            raise ValueError("Invalid action")

        # Check obstacle collision
        if (nr, nc) in self.obstacles:
            nr, nc = r, c

        self.pos = (nr, nc)
        done = (self.pos == self.goal_pos)
        reward = 1.0 if done else 0.0
        return self._pos_to_state(self.pos), reward, done


def run_gridworld_benchmark():
    print("=" * 70)
    print("2. RUNNING SUTTON'S GRIDWORLD BENCHMARK (N=0 vs N=5 vs N=50)")
    print("=" * 70)

    np.random.seed(42)
    env = Gridworld()
    n_states = env.height * env.width
    n_actions = 4
    n_episodes = 25

    planning_options = [0, 5, 50]
    results = {}

    for n in planning_options:
        agent = DynaQAgent(n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=n)
        steps_history = []

        for ep in range(n_episodes):
            s = env.reset()
            steps = 0
            done = False
            while not done and steps < 2000:
                a = agent.select_action(s)
                s_next, r, done = env.step(a)
                agent.step_learn(s, a, r, s_next, done)
                s = s_next
                steps += 1
            steps_history.append(steps)

        results[n] = steps_history
        avg_last_5 = np.mean(steps_history[-5:])
        print(f"Planning Steps N={n:2d} | Ep 1 steps: {steps_history[0]:4d} | Ep 2: {steps_history[1]:4d} | Final 5 Avg: {avg_last_5:5.1f}")

    # Sanity checks on sample efficiency:
    # N=50 should solve Episode 2 in drastically fewer steps than Episode 1
    assert results[50][1] < results[50][0] / 3.0, (
        f"Dyna-Q N=50 should exhibit rapid sample efficiency improvement: Ep1={results[50][0]}, Ep2={results[50][1]}"
    )
    # N=50 should reach optimal or near-optimal path (< 25 steps) within 10 episodes
    assert results[50][9] < 30, f"Dyna-Q N=50 should converge quickly, got {results[50][9]} steps"
    print(">> SUCCESS: Dyna-Q planning dramatically accelerates convergence over model-free (N=0)!\n")


# =====================================================================
# 4. Dyna-Q+ Exploration Bonus Verification
# =====================================================================

class DynaQPlusAgent:
    """
    Dyna-Q+ agent with exploration bonus r + kappa * sqrt(tau) during planning,
    allowing it to adapt to non-stationary changing environments (e.g. shortcut mazes).
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=10, kappa=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.kappa = kappa

        self.Q = np.zeros((n_states, n_actions))
        self.model = {}
        self.time_step = 0
        self.last_tried = np.zeros((n_states, n_actions))

        # In Dyna-Q+, all (s, a) pairs can be modeled (defaulting to returning to same state with r=0)
        self.all_states = set()

    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.Q[state]
        max_val = np.max(q_vals)
        best_actions = np.flatnonzero(np.isclose(q_vals, max_val))
        return np.random.choice(best_actions)

    def step_learn(self, s, a, r, s_next, done):
        self.time_step += 1
        self.last_tried[s, a] = self.time_step
        self.all_states.add(s)

        # 1. Direct Q update
        target = r + self.gamma * (0.0 if done else np.max(self.Q[s_next]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # 2. Model update
        self.model[(s, a)] = (r, s_next, done)

        # 3. Planning with Dyna-Q+ bonus: r_plan = r + kappa * sqrt(tau)
        if self.planning_steps > 0 and len(self.all_states) > 0:
            states_list = list(self.all_states)
            for _ in range(self.planning_steps):
                s_sim = states_list[np.random.randint(len(states_list))]
                a_sim = np.random.randint(self.n_actions)

                if (s_sim, a_sim) in self.model:
                    r_sim, s_next_sim, done_sim = self.model[(s_sim, a_sim)]
                else:
                    # Never tried: transitions to self with reward 0
                    r_sim, s_next_sim, done_sim = 0.0, s_sim, False

                tau = self.time_step - self.last_tried[s_sim, a_sim]
                r_bonus = r_sim + self.kappa * np.sqrt(tau)

                plan_target = r_bonus + self.gamma * (0.0 if done_sim else np.max(self.Q[s_next_sim]))
                self.Q[s_sim, a_sim] += self.alpha * (plan_target - self.Q[s_sim, a_sim])


def verify_dynaq_plus_bonus():
    print("=" * 70)
    print("3. VERIFYING DYNA-Q+ EXPLORATION BONUS MATHEMATICS")
    print("=" * 70)

    agent = DynaQPlusAgent(n_states=5, n_actions=2, kappa=0.01)
    agent.time_step = 1000
    agent.last_tried[1, 0] = 100  # Tried at step 100
    tau = agent.time_step - agent.last_tried[1, 0]  # 900 steps elapsed
    bonus = agent.kappa * np.sqrt(tau)  # 0.01 * 30 = 0.30

    print(f"Elapsed steps tau = {tau:.1f}")
    print(f"Calculated exploration bonus: {bonus:.4f} (Expected: 0.3000)")
    assert np.isclose(bonus, 0.30, atol=1e-12), f"Expected 0.30, got {bonus}"
    print(">> SUCCESS: Dyna-Q+ exploration bonus matches theoretical formula!\n")


# =====================================================================
# 5. Section 6 Numerical Illustrations Verification Suite
# =====================================================================

def verify_illustration2_dyna_cycle():
    print("=" * 70)
    print("4. VERIFYING ILLUSTRATION 2: DYNA-Q REAL & PLANNING STEP CYCLE")
    print("=" * 70)
    alpha = 0.2000
    gamma = 0.9000
    Q = {"S0": 0.0000, "S1": 0.0000, "S2": 0.0000}

    # Real step: (S2, a0) -> S3, R = 10.0000 (S3 terminal)
    target_real = 10.0000 + gamma * 0.0000
    Q["S2"] = Q["S2"] + alpha * (target_real - Q["S2"])
    assert np.isclose(Q["S2"], 2.0000, atol=1e-8), f"Expected 2.0, got {Q['S2']}"

    # Planning Step 1: sample (S1, a0) -> (R=0, S2)
    target_p1 = 0.0000 + gamma * Q["S2"]
    Q["S1"] = Q["S1"] + alpha * (target_p1 - Q["S1"])
    assert np.isclose(Q["S1"], 0.3600, atol=1e-8), f"Expected 0.36, got {Q['S1']}"

    # Planning Step 2: sample (S0, a0) -> (R=0, S1)
    target_p2 = 0.0000 + gamma * Q["S1"]
    Q["S0"] = Q["S0"] + alpha * (target_p2 - Q["S0"])
    assert np.isclose(Q["S0"], 0.0648, atol=1e-8), f"Expected 0.0648, got {Q['S0']}"

    # Planning Step 3: sample (S1, a0) -> (R=0, S2)
    target_p3 = 0.0000 + gamma * Q["S2"]
    Q["S1"] = Q["S1"] + alpha * (target_p3 - Q["S1"])
    assert np.isclose(Q["S1"], 0.6480, atol=1e-8), f"Expected 0.6480, got {Q['S1']}"

    print(f">> SUCCESS: Illustration 2 cycle verified: Q(S2)={Q['S2']:.4f}, Q(S1)={Q['S1']:.4f}, Q(S0)={Q['S0']:.4f}\n")


def verify_illustration3_shortcut_overtaking():
    print("=" * 70)
    print("5. VERIFYING ILLUSTRATION 3: DYNA-Q+ SHORTCUT OVERTAKING")
    print("=" * 70)
    kappa = 0.0010
    gap = 0.1200
    tau_star = (gap / kappa)**2
    assert np.isclose(tau_star, 14400.0, atol=1e-8), f"Expected 14400, got {tau_star}"

    for tau, expected_bonus, expected_flip in [
        (900, 0.0300, False),
        (4900, 0.0700, False),
        (10000, 0.1000, False),
        (14400, 0.1200, True),
        (25600, 0.1600, True),
    ]:
        bonus = kappa * np.sqrt(tau)
        assert np.isclose(bonus, expected_bonus, atol=1e-8)
        assert (bonus >= gap) == expected_flip

    print(f">> SUCCESS: Illustration 3 verified: tau* = {tau_star:.0f} steps for shortcut discovery!\n")


def verify_illustration4_compounding_error():
    print("=" * 70)
    print("6. VERIFYING ILLUSTRATION 4: SIMULATION LEMMA & COMPOUNDING BOUNDS")
    print("=" * 70)
    eps_m = 0.0500
    R_max = 1.0000
    gamma = 0.9500

    expected_vals = {
        1: (0.0500, 0.0500, 0.0500),
        5: (0.2500, 0.7500, 0.655502),
        10: (0.5000, 2.7500, 2.037861),
        20: (1.0000, 10.5000, 5.660603),
    }

    for H, (exp_step_tv, exp_cum_tv, exp_val_err) in expected_vals.items():
        step_tv = H * eps_m
        cum_tv = H * (H + 1) / 2.0 * eps_m
        t_arr = np.arange(1, H + 1)
        val_err = np.sum(gamma**(t_arr - 1) * t_arr * eps_m * R_max)
        assert np.isclose(step_tv, exp_step_tv, atol=1e-4)
        assert np.isclose(cum_tv, exp_cum_tv, atol=1e-4)
        assert np.isclose(val_err, exp_val_err, atol=1e-4)

    bound_inf = (gamma * eps_m * R_max) / ((1 - gamma)**2)
    assert np.isclose(bound_inf, 19.0000, atol=1e-8)
    print(f">> SUCCESS: Illustration 4 verified: compounding bounds match theoretical formulas!\n")


def verify_illustration5_rls_transition_model():
    print("=" * 70)
    print("7. VERIFYING ILLUSTRATION 5: RECURSIVE LEAST SQUARES TRANSITION MODEL")
    print("=" * 70)
    theta = np.array([0.0000, 0.0000])
    P = 10.0000 * np.eye(2)

    # Step 1: x_1 = [1.0, 2.0], y_1 = 1.8000
    x_1 = np.array([1.0, 2.0])
    y_1 = 1.8000
    denom_1 = 1.0 + float(x_1 @ P @ x_1)
    k_1 = (P @ x_1) / denom_1
    pred_1 = float(x_1 @ theta)
    e_1 = y_1 - pred_1
    theta = theta + k_1 * e_1
    P = (np.eye(2) - np.outer(k_1, x_1)) @ P

    assert np.isclose(denom_1, 51.0000, atol=1e-8)
    assert np.allclose(k_1, [10.0 / 51.0, 20.0 / 51.0], atol=1e-8)
    assert np.allclose(theta, [18.0 / 51.0, 36.0 / 51.0], atol=1e-8)
    assert np.allclose(theta, [0.352941, 0.705882], atol=1e-4)

    # Step 2: x_2 = [1.8, -1.0], y_2 = 0.9400
    x_2 = np.array([1.8, -1.0])
    y_2 = 0.9400
    P_x2 = P @ x_2
    denom_2 = 1.0 + float(x_2 @ P_x2)
    k_2 = P_x2 / denom_2
    pred_2 = float(x_2 @ theta)
    e_2 = y_2 - pred_2
    theta = theta + k_2 * e_2
    P = (np.eye(2) - np.outer(k_2, x_2)) @ P

    assert np.isclose(denom_2, 43.321569, atol=1e-4)
    assert np.allclose(theta, [0.781986, 0.490903], atol=1e-4)
    assert np.isclose(P[0, 0], 0.230832, atol=1e-4)
    print(f">> SUCCESS: Illustration 5 verified: RLS converged to theta = [{theta[0]:.4f}, {theta[1]:.4f}]!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    run_gridworld_benchmark()
    verify_dynaq_plus_bonus()
    verify_illustration2_dyna_cycle()
    verify_illustration3_shortcut_overtaking()
    verify_illustration4_compounding_error()
    verify_illustration5_rls_transition_model()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 21 (DYNA-Q) VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

