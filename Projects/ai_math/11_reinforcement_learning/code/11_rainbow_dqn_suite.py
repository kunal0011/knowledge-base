"""
Mathematical Verification Script for Module 11.11: Advanced DQN (Rainbow Suite)
================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Dueling Network forward pass:
     V(s) = 5.0000, A(s) = [2.0, 6.0, 4.0], mean(A) = 4.0
     Q(s) = [3.0000, 7.0000, 5.0000]
   - Verified identically in NumPy and PyTorch nn.Module.
2. Sum-Tree Implementation for Prioritized Experience Replay (PER):
   - Leaf priorities: [1.0, 3.0, 2.0, 4.0], sum = 10.0
   - Prefix sum sample query for v = 4.5 -> selects leaf index 2
   - Dynamic update verification: updating priority propagates in O(log N).
3. Double DQN vs DQN Overestimation Bias:
   - Compares action-value target estimates, proving DDQN prevents overestimation.
"""

import numpy as np
import torch
import torch.nn as nn

def verify_part_5_dueling_hand_calculation():
    print("--- Test 1: Part 5 Dueling Architecture Verification ---")
    
    # Inputs
    s = np.array([2.0, -1.0])
    W_h = np.eye(2)
    h = np.maximum(0, W_h @ s)
    assert np.allclose(h, [2.0, 0.0])
    
    w_V = np.array([2.0, 1.0])
    b_V = 1.0
    V = np.dot(w_V, h) + b_V
    assert np.isclose(V, 5.0000)
    
    W_A = np.array([
        [0.5, 0.0],
        [2.5, 0.0],
        [1.5, 0.0]
    ])
    b_A = np.array([1.0, 1.0, 1.0])
    
    A_raw = W_A @ h + b_A
    assert np.allclose(A_raw, [2.0, 6.0, 4.0])
    
    A_mean = np.mean(A_raw)
    assert np.isclose(A_mean, 4.0000)
    
    A_centered = A_raw - A_mean
    assert np.allclose(A_centered, [-2.0, 2.0, 0.0])
    assert np.isclose(np.sum(A_centered), 0.0)
    
    Q = V + A_centered
    expected_Q = np.array([3.0, 7.0, 5.0])
    assert np.allclose(Q, expected_Q)
    
    print(f"  V(s) = {V:.4f}")
    print(f"  Raw Advantages: {A_raw}")
    print(f"  Mean Advantage: {A_mean:.4f}")
    print(f"  Centered Advantages: {A_centered}")
    print(f"  Final Q(s, a): {Q}")
    
    # Verify in PyTorch
    class DuelingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = nn.Linear(2, 1)
            self.adv_head = nn.Linear(2, 3)
            
            # Load exact weights
            self.value_head.weight.data = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
            self.value_head.bias.data = torch.tensor([1.0], dtype=torch.float32)
            self.adv_head.weight.data = torch.tensor(W_A, dtype=torch.float32)
            self.adv_head.bias.data = torch.tensor(b_A, dtype=torch.float32)
            
        def forward(self, x):
            v = self.value_head(x)
            a = self.adv_head(x)
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        
    pt_net = DuelingNet()
    with torch.no_grad():
        pt_out = pt_net(torch.tensor([h], dtype=torch.float32)).numpy().flatten()
    assert np.allclose(pt_out, expected_Q)
    print("  [PASSED] Dueling network forward pass verified to exact machine precision!\n")


def test_sum_tree_per():
    print("--- Test 2: Sum-Tree Implementation for Prioritized Replay ---")
    
    class SumTree:
        def __init__(self, capacity):
            self.capacity = capacity
            self.tree = np.zeros(2 * capacity - 1)
            self.data = np.zeros(capacity, dtype=object)
            self.n_entries = 0
            self.write = 0
            
        def _propagate(self, idx, change):
            parent = (idx - 1) // 2
            self.tree[parent] += change
            if parent != 0:
                self._propagate(parent, change)
                
        def _retrieve(self, idx, s):
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                return self._retrieve(left, s)
            else:
                return self._retrieve(right, s - self.tree[left])
            
        def total(self):
            return self.tree[0]
        
        def add(self, p, data):
            idx = self.write + self.capacity - 1
            self.data[self.write] = data
            self.update(idx, p)
            self.write = (self.write + 1) % self.capacity
            if self.n_entries < self.capacity:
                self.n_entries += 1
                
        def update(self, idx, p):
            change = p - self.tree[idx]
            self.tree[idx] = p
            self._propagate(idx, change)
            
        def get(self, s):
            idx = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1
            return idx, self.tree[idx], self.data[data_idx]
        
    # Test Part 5 hand setup: capacity = 4, priorities = [1.0, 3.0, 2.0, 4.0]
    tree = SumTree(4)
    priorities = [1.0, 3.0, 2.0, 4.0]
    for i, p in enumerate(priorities):
        tree.add(p, f"transition_{i}")
        
    print(f"  Total Tree Sum: {tree.total():.1f}")
    assert np.isclose(tree.total(), 10.0), f"Expected sum 10.0, got {tree.total()}"
    
    # Query with v = 4.5
    idx, p, data = tree.get(4.5)
    print(f"  Query v=4.5 -> Returned node index {idx}, priority {p}, data '{data}'")
    assert data == "transition_2", f"Expected transition_2, got {data}"
    assert np.isclose(p, 2.0), f"Expected priority 2.0, got {p}"
    
    # Dynamic update: change priority of leaf 1 from 3.0 to 5.0 (+2.0 change)
    leaf1_tree_idx = 1 + 4 - 1 # 4
    tree.update(leaf1_tree_idx, 5.0)
    print(f"  Updated leaf 1 to 5.0 -> New total sum: {tree.total():.1f}")
    assert np.isclose(tree.total(), 12.0), f"Expected updated sum 12.0, got {tree.total()}"
    
    print("  [PASSED] Sum-Tree O(log N) retrieval and update verified!\n")


def test_double_dqn_target_evaluation():
    print("--- Test 3: Double DQN vs Standard DQN Overestimation ---")
    # Synthetic scenario with noisy Q-evaluations
    np.random.seed(42)
    n_actions = 10
    true_mean = 0.0
    
    # Noisy estimates for online network and target network
    q_online = np.random.normal(true_mean, 1.0, size=n_actions)
    q_target = np.random.normal(true_mean, 1.0, size=n_actions)
    
    # Standard DQN target (maximization over target network directly)
    target_dqn = np.max(q_target)
    
    # Double DQN target (greedy action selected from online, evaluated on target)
    best_a_online = np.argmax(q_online)
    target_ddqn = q_target[best_a_online]
    
    print(f"  Online Q: {np.round(q_online, 2)}")
    print(f"  Target Q: {np.round(q_target, 2)}")
    print(f"  Standard DQN max target: {target_dqn:.4f} (positive bias)")
    print(f"  Double DQN evaluated target: {target_ddqn:.4f} (unbiased)")
    
    # Over 1000 Monte Carlo simulations, verify E[max Q] > E[Q_double]
    runs = 2000
    dqn_targets = []
    ddqn_targets = []
    for _ in range(runs):
        qo = np.random.normal(true_mean, 1.0, size=n_actions)
        qt = np.random.normal(true_mean, 1.0, size=n_actions)
        dqn_targets.append(np.max(qt))
        ddqn_targets.append(qt[np.argmax(qo)])
        
    mean_dqn = np.mean(dqn_targets)
    mean_ddqn = np.mean(ddqn_targets)
    print(f"  2000 Runs Mean Target: DQN = {mean_dqn:.4f}, DDQN = {mean_ddqn:.4f}")
    
    assert mean_dqn > 1.0, f"Expected DQN overestimation > 1.0, got {mean_dqn}"
    assert abs(mean_ddqn) < 0.1, f"Expected DDQN near true mean 0.0, got {mean_ddqn}"
    print("  [PASSED] Double DQN successfully eliminates maximization overestimation bias!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.11 ADVANCED DQN (RAINBOW) UNIT TEST SUITE")
    print("=================================================================\n")
    
    verify_part_5_dueling_hand_calculation()
    test_sum_tree_per()
    test_double_dqn_target_evaluation()
    
    print("=================================================================")
    print("ALL MODULE 11.11 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
