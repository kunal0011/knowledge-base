"""
Monte Carlo Tree Search (MCTS) & AlphaZero Verification Suite
Module 11: Reinforcement Learning - Chapter 23

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Step-by-step PUCT score evaluation across Simulations 1 to 4
   - Accurate visit count tracking and mean Q-value updates (< 1e-6)
2. Complete Generic AlphaZero MCTS Engine:
   - MCTSNode with PUCT action selection
   - Expansion, dual-headed evaluation, and zero-sum backup
   - Policy extraction via temperature-scaled visit count distribution
3. Tic-Tac-Toe Zero-Sum Benchmark:
   - MCTS policy evaluated against a random opponent
   - Verification of 100% non-loss rate (all wins or draws)
"""

import numpy as np


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' ALPHAZERO PUCT MCTS CALCULATIONS")
    print("=" * 70)

    # Parameters from Section 5.1
    c_puct = 1.0
    prior_P = np.array([0.70, 0.30])  # [a1, a2]
    N = np.zeros(2)                   # visit counts
    W = np.zeros(2)                   # value sums
    Q = np.zeros(2)                   # mean values

    # --- Simulation 1 ---
    # N_total = 0, both actions unvisited. Selected by prior: a1
    v1 = 0.60
    selected_1 = 0  # a1
    N[selected_1] += 1
    W[selected_1] += v1
    Q[selected_1] = W[selected_1] / N[selected_1]
    print(f"Sim 1: Selected a1 | N=[{N[0]:.0f}, {N[1]:.0f}] | Q=[{Q[0]:.4f}, {Q[1]:.4f}]")
    assert np.isclose(Q[0], 0.60, atol=1e-6)

    # --- Simulation 2 ---
    n_total_2 = np.sum(N)  # 1
    sqrt_n2 = np.sqrt(n_total_2)  # 1.0
    u1_sim2 = c_puct * prior_P[0] * (sqrt_n2 / (1.0 + N[0]))  # 1.0 * 0.70 * 1 / 2 = 0.35
    u2_sim2 = c_puct * prior_P[1] * (sqrt_n2 / (1.0 + N[1]))  # 1.0 * 0.30 * 1 / 1 = 0.30
    score1_sim2 = Q[0] + u1_sim2  # 0.60 + 0.35 = 0.95
    score2_sim2 = Q[1] + u2_sim2  # 0.00 + 0.30 = 0.30

    print(f"Sim 2: Score(a1) = {score1_sim2:.4f} (Exp: 0.9500) | Score(a2) = {score2_sim2:.4f} (Exp: 0.3000)")
    assert np.isclose(score1_sim2, 0.95, atol=1e-6)
    assert np.isclose(score2_sim2, 0.30, atol=1e-6)

    # Select a1, evaluate v2 = +0.40
    v2 = 0.40
    selected_2 = 0
    N[selected_2] += 1
    W[selected_2] += v2
    Q[selected_2] = W[selected_2] / N[selected_2]  # (0.6 + 0.4) / 2 = 0.50
    print(f"Sim 2 Backup: N(a1) = {N[0]:.0f}, Q(a1) = {Q[0]:.4f} (Exp: 0.5000)")
    assert np.isclose(Q[0], 0.50, atol=1e-6)

    # --- Simulation 3 ---
    n_total_3 = np.sum(N)  # 2
    sqrt_n3 = np.sqrt(n_total_3)  # sqrt(2) ≈ 1.414214
    u1_sim3 = c_puct * prior_P[0] * (sqrt_n3 / (1.0 + N[0]))  # 0.70 * sqrt(2) / 3 ≈ 0.329983
    u2_sim3 = c_puct * prior_P[1] * (sqrt_n3 / (1.0 + N[1]))  # 0.30 * sqrt(2) / 1 ≈ 0.424264
    score1_sim3 = Q[0] + u1_sim3  # 0.50 + 0.329983 = 0.829983
    score2_sim3 = Q[1] + u2_sim3  # 0.00 + 0.424264 = 0.424264

    print(f"Sim 3: Score(a1) = {score1_sim3:.4f} (Exp: 0.8300) | Score(a2) = {score2_sim3:.4f} (Exp: 0.4243)")
    assert np.isclose(score1_sim3, 0.50 + 0.70 * np.sqrt(2)/3.0, atol=1e-6)
    assert np.isclose(score2_sim3, 0.30 * np.sqrt(2), atol=1e-6)

    # Select a1, evaluate v3 = -0.20
    v3 = -0.20
    selected_3 = 0
    N[selected_3] += 1
    W[selected_3] += v3
    Q[selected_3] = W[selected_3] / N[selected_3]  # (1.0 - 0.2) / 3 = 0.80 / 3 ≈ 0.266667
    print(f"Sim 3 Backup: N(a1) = {N[0]:.0f}, Q(a1) = {Q[0]:.4f} (Exp: 0.2667)")
    assert np.isclose(Q[0], 0.80 / 3.0, atol=1e-6)

    # --- Simulation 4 ---
    n_total_4 = np.sum(N)  # 3
    sqrt_n4 = np.sqrt(n_total_4)  # sqrt(3) ≈ 1.732051
    u1_sim4 = c_puct * prior_P[0] * (sqrt_n4 / (1.0 + N[0]))  # 0.70 * sqrt(3) / 4 ≈ 0.303109
    u2_sim4 = c_puct * prior_P[1] * (sqrt_n4 / (1.0 + N[1]))  # 0.30 * sqrt(3) / 1 ≈ 0.519615
    score1_sim4 = Q[0] + u1_sim4  # 0.266667 + 0.303109 = 0.569776
    score2_sim4 = Q[1] + u2_sim4  # 0.00 + 0.519615 = 0.519615

    print(f"Sim 4: Score(a1) = {score1_sim4:.4f} (Exp: 0.5698) | Score(a2) = {score2_sim4:.4f} (Exp: 0.5196)")
    assert np.isclose(score1_sim4, 0.80/3.0 + 0.70 * np.sqrt(3)/4.0, atol=1e-6)
    assert np.isclose(score2_sim4, 0.30 * np.sqrt(3), atol=1e-6)
    assert score1_sim4 > score2_sim4, "a1 should win selection in Sim 4"

    print(">> SUCCESS: Hand calculations for Simulations 1-4 verified to exact numerical precision!\n")


# =====================================================================
# 2. Complete Generic AlphaZero MCTS Engine
# =====================================================================

class MCTSNode:
    """
    Node in the AlphaZero MCTS search tree.
    """
    def __init__(self, state, player, parent=None, action_taken=None, prior=0.0):
        self.state = state
        self.player = player  # +1 (Player 1) or -1 (Player 2)
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct=1.414):
        best_score = -float('inf')
        best_action = None
        best_child = None

        total_visits = sum(child.visit_count for child in self.children.values())
        sqrt_total = np.sqrt(max(1, total_visits))

        for action, child in self.children.items():
            # In zero-sum, child q_value is from child.player's perspective.
            # To self.player, the value is -child.q_value!
            q_perspective = -child.q_value
            u_bonus = c_puct * child.prior * (sqrt_total / (1 + child.visit_count))
            score = q_perspective + u_bonus

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, legal_actions, action_priors, next_player):
        for action in legal_actions:
            prior = action_priors[action]
            next_state = self.state.apply_action(action, self.player)
            self.children[action] = MCTSNode(
                state=next_state,
                player=next_player,
                parent=self,
                action_taken=action,
                prior=prior
            )

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            # Value flips sign for opponent in zero-sum tree
            self.parent.backpropagate(-value)


# =====================================================================
# 3. Tic-Tac-Toe Game Environment & MCTS Benchmark
# =====================================================================

class TicTacToeState:
    """
    3x3 Tic-Tac-Toe board state representation.
    0: empty, 1: Player 1 (X), -1: Player 2 (O).
    """
    def __init__(self, board=None):
        if board is None:
            self.board = np.zeros((3, 3), dtype=int)
        else:
            self.board = np.copy(board)

    def get_legal_actions(self):
        empty_cells = np.argwhere(self.board == 0)
        return [r * 3 + c for r, c in empty_cells]

    def apply_action(self, action, player):
        r, c = divmod(action, 3)
        assert self.board[r, c] == 0
        new_board = np.copy(self.board)
        new_board[r, c] = player
        return TicTacToeState(new_board)

    def check_terminal(self):
        # Rows and cols
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return True, np.sign(np.sum(self.board[i, :]))
            if abs(np.sum(self.board[:, i])) == 3:
                return True, np.sign(np.sum(self.board[:, i]))
        # Diagonals
        diag1 = self.board[0, 0] + self.board[1, 1] + self.board[2, 2]
        if abs(diag1) == 3:
            return True, np.sign(diag1)
        diag2 = self.board[0, 2] + self.board[1, 1] + self.board[2, 0]
        if abs(diag2) == 3:
            return True, np.sign(diag2)
        # Draw check
        if len(self.get_legal_actions()) == 0:
            return True, 0  # Draw
        return False, 0


def mock_neural_evaluator(state, player):
    """
    Mock dual-headed network:
    Returns uniform prior over legal actions and simple heuristic rollout value.
    """
    legal = state.get_legal_actions()
    priors = {a: 1.0 / len(legal) for a in legal} if len(legal) > 0 else {}

    # Simple rollout evaluation:
    curr_s = TicTacToeState(state.board)
    curr_p = player
    while True:
        is_term, winner = curr_s.check_terminal()
        if is_term:
            # Value from perspective of 'player'
            return priors, winner * player
        legal_acts = curr_s.get_legal_actions()
        act = np.random.choice(legal_acts)
        curr_s = curr_s.apply_action(act, curr_p)
        curr_p = -curr_p


def run_mcts_search(root_state, current_player, n_simulations=100, c_puct=1.414):
    root = MCTSNode(root_state, current_player)
    legal = root_state.get_legal_actions()
    priors, _ = mock_neural_evaluator(root_state, current_player)
    root.expand(legal, priors, -current_player)

    for _ in range(n_simulations):
        node = root
        # 1. Selection
        while node.is_expanded():
            _, node = node.select_child(c_puct=c_puct)

        # 2. Evaluation & Terminal Check
        is_term, winner = node.state.check_terminal()
        if is_term:
            # Terminal value from node.player's perspective
            val = winner * node.player
        else:
            # 3. Expansion & Evaluation
            leg_acts = node.state.get_legal_actions()
            priors, sim_val = mock_neural_evaluator(node.state, node.player)
            node.expand(leg_acts, priors, -node.player)
            val = sim_val

        # 4. Backup
        node.backpropagate(val)

    # Return action with highest visit count
    action_visits = {action: child.visit_count for action, child in root.children.items()}
    best_action = max(action_visits, key=action_visits.get)
    return best_action, action_visits


def verify_mcts_tic_tac_toe():
    print("=" * 70)
    print("2. VERIFYING ALPHAZERO MCTS ON TIC-TAC-TOE VS RANDOM OPPONENT")
    print("=" * 70)

    np.random.seed(42)
    n_games = 10
    mcts_wins = 0
    draws = 0
    losses = 0

    for game in range(n_games):
        state = TicTacToeState()
        player = 1  # 1: MCTS (Player 1), -1: Random (Player 2)

        while True:
            is_term, winner = state.check_terminal()
            if is_term:
                if winner == 1:
                    mcts_wins += 1
                elif winner == 0:
                    draws += 1
                else:
                    losses += 1
                break

            if player == 1:
                action, _ = run_mcts_search(state, current_player=1, n_simulations=100)
            else:
                legal = state.get_legal_actions()
                action = np.random.choice(legal)

            state = state.apply_action(action, player)
            player = -player

    print(f"Results across {n_games} games: MCTS Wins: {mcts_wins}, Draws: {draws}, Losses: {losses}")
    assert losses == 0, f"AlphaZero MCTS should never lose to random play, but got {losses} losses!"
    print(">> SUCCESS: AlphaZero MCTS achieved 100% non-loss rate (0 losses)!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_mcts_tic_tac_toe()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 23 (MCTS & ALPHAZERO) VERIFICATIONS PASSED!")
    print("=" * 70)
