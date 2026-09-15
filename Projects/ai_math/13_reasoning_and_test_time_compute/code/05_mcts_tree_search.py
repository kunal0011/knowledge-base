"""
Monte Carlo Tree Search (MCTS) & PUCT for Language Model Reasoning
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
import numpy as np


def compute_puct(Q: float, P: float, N_edge: int, N_parent: int, c_puct: float = 2.0) -> tuple[float, float]:
    """
    Computes PUCT score: Q + c_puct * P * (sqrt(N_parent) / (1 + N_edge)).
    Returns (total_score, exploration_bonus U).
    """
    u = c_puct * P * (math.sqrt(N_parent) / (1.0 + N_edge))
    score = Q + u
    return score, u


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    c_puct = 2.0
    N_parent = 4 # 3 + 1
    
    # Action 1
    P1, N1, W1 = 0.70, 3, 1.80
    Q1 = W1 / N1 # 0.600000
    score1, u1 = compute_puct(Q1, P1, N1, N_parent, c_puct)
    
    assert math.isclose(Q1, 0.600000, abs_tol=1e-6)
    assert math.isclose(u1, 0.700000, abs_tol=1e-6)
    assert math.isclose(score1, 1.300000, abs_tol=1e-6)
    print(f"Action a1: Q={Q1:.4f}, U={u1:.4f}, PUCT={score1:.4f} -> EXACT MATCH 1.3000")
    
    # Action 2
    P2, N2, W2 = 0.30, 1, 0.90
    Q2 = W2 / N2 # 0.900000
    score2, u2 = compute_puct(Q2, P2, N2, N_parent, c_puct)
    
    assert math.isclose(Q2, 0.900000, abs_tol=1e-6)
    assert math.isclose(u2, 0.600000, abs_tol=1e-6)
    assert math.isclose(score2, 1.500000, abs_tol=1e-6)
    print(f"Action a2: Q={Q2:.4f}, U={u2:.4f}, PUCT={score2:.4f} -> EXACT MATCH 1.5000")
    
    # Selection
    assert score2 > score1, "Action a2 must be selected over a1"
    print("Selection decision: Action a2 selected (1.50 > 1.30) -> EXACT MATCH")
    
    # Backpropagation with leaf evaluation v = 1.0
    v = 1.0
    N2_new = N2 + 1
    W2_new = W2 + v
    Q2_new = W2_new / N2_new
    
    assert N2_new == 2
    assert math.isclose(W2_new, 1.900000, abs_tol=1e-6)
    assert math.isclose(Q2_new, 0.950000, abs_tol=1e-6)
    print(f"Post-Backprop Action a2: N={N2_new}, W={W2_new:.4f}, Q={Q2_new:.4f} -> EXACT MATCH 0.9500\n")


class MCTSNode:
    """A Node in the MCTS Language Reasoning Tree."""
    def __init__(self, state: str, parent=None, prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children: dict[str, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self, c_puct: float = 2.0):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        parent_visits = max(1, self.visit_count)
        for action, child in self.children.items():
            u = c_puct * child.prior * (math.sqrt(parent_visits) / (1.0 + child.visit_count))
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, actions_with_priors: list[tuple[str, float]]):
        for action, prior in actions_with_priors:
            next_state = f"{self.state} -> {action}"
            self.children[action] = MCTSNode(state=next_state, parent=self, prior=prior)

    def backpropagate(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(value)


def test_mcts_reasoning_search():
    print("--- 2. Testing Complete MCTS Reasoning Tree Search ---")
    root = MCTSNode(state="Question: What is 2 + 3 * 4?")
    
    # Expand root with candidate initial thoughts
    root.expand([
        ("Order of operations: multiply 3*4=12 first", 0.70),
        ("Left-to-right: add 2+3=5 first", 0.30)
    ])
    
    # Run 20 MCTS iterations
    for i in range(20):
        # 1. Selection
        action, child = root.select_child(c_puct=1.5)
        
        # 2. Evaluation via mock PRM / ground truth verifier
        if "multiply 3*4=12" in action:
            value = 1.0 # Mathematically correct step
        else:
            value = 0.0 # Incorrect step
            
        # 3. Backpropagation
        child.backpropagate(value)
        
    correct_child = root.children["Order of operations: multiply 3*4=12 first"]
    flawed_child = root.children["Left-to-right: add 2+3=5 first"]
    
    print(f"Correct Branch Visits: {correct_child.visit_count} (Q={correct_child.q_value:.2f})")
    print(f"Flawed Branch Visits:  {flawed_child.visit_count} (Q={flawed_child.q_value:.2f})")
    
    assert correct_child.visit_count > flawed_child.visit_count, "MCTS failed to allocate visits to correct branch"
    assert math.isclose(correct_child.q_value, 1.0, abs_tol=1e-5)
    print("MCTS successfully condensed search onto the correct mathematical derivation!")
    print("All tests in Chapter 13.5 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    test_mcts_reasoning_search()
    print("🟢 Chapter 13.5 Verification 100% Complete.")
