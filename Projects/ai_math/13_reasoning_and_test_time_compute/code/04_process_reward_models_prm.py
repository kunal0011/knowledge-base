"""
Process Reward Models (PRMs) & Step-Level Value Estimation (Math-Shepherd)
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    
    # Step 1: Monte Carlo Rollout Values
    rollouts_s1 = [1, 1, 1, 0]
    rollouts_s2 = [0, 0, 0, 0]
    rollouts_s3 = [0, 0, 0, 0]
    
    v1 = sum(rollouts_s1) / len(rollouts_s1)
    v2 = sum(rollouts_s2) / len(rollouts_s2)
    v3 = sum(rollouts_s3) / len(rollouts_s3)
    
    assert math.isclose(v1, 0.750000, abs_tol=1e-6)
    assert math.isclose(v2, 0.000000, abs_tol=1e-6)
    assert math.isclose(v3, 0.000000, abs_tol=1e-6)
    print(f"MC Values: V(s1)={v1:.4f}, V(s2)={v2:.4f}, V(s3)={v3:.4f}")
    
    # Step 2: Derived step labels
    y = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    
    # Step 3: PRM Logits and Probabilities
    r = torch.tensor([1.50, -1.00, -2.00], dtype=torch.float32)
    p = torch.sigmoid(r)
    
    expected_p1 = 0.817574
    expected_p2 = 0.268941
    expected_p3 = 0.119203
    assert math.isclose(p[0].item(), expected_p1, abs_tol=1e-5)
    assert math.isclose(p[1].item(), expected_p2, abs_tol=1e-5)
    assert math.isclose(p[2].item(), expected_p3, abs_tol=1e-5)
    print(f"PRM Probabilities: p1={p[0]:.6f}, p2={p[1]:.6f}, p3={p[2]:.6f} -> EXACT MATCH")
    
    # Step 4: Binary cross-entropy losses per step
    l1 = -math.log(p[0].item())
    l2 = -math.log(1.0 - p[1].item())
    l3 = -math.log(1.0 - p[2].item())
    
    expected_l1 = 0.201413
    expected_l2 = 0.313262
    expected_l3 = 0.126928
    assert math.isclose(l1, expected_l1, abs_tol=1e-5)
    assert math.isclose(l2, expected_l2, abs_tol=1e-5)
    assert math.isclose(l3, expected_l3, abs_tol=1e-5)
    
    mean_loss = (l1 + l2 + l3) / 3.0
    expected_mean_loss = 0.213868
    assert math.isclose(mean_loss, expected_mean_loss, abs_tol=1e-5)
    
    # Verify with PyTorch binary_cross_entropy_with_logits
    py_loss = F.binary_cross_entropy_with_logits(r, y)
    assert math.isclose(py_loss.item(), expected_mean_loss, abs_tol=1e-5)
    print(f"PRM Loss: Manual={mean_loss:.6f}, PyTorch={py_loss.item():.6f} -> EXACT MATCH {expected_mean_loss}")
    
    # Step 5: Trajectory Aggregations
    r_prod = p[0].item() * p[1].item() * p[2].item()
    r_min = min(p[0].item(), p[1].item(), p[2].item())
    expected_r_prod = 0.026211
    expected_r_min = 0.119203
    assert math.isclose(r_prod, expected_r_prod, abs_tol=1e-5)
    assert math.isclose(r_min, expected_r_min, abs_tol=1e-5)
    print(f"Trajectory Aggregations: Product={r_prod:.6f}, Min={r_min:.6f} -> EXACT MATCH\n")


def verify_aggregation_comparison():
    print("--- 2. Verifying Illustration 1: Min vs Product Aggregation ---")
    proof_a = [0.70, 0.70, 0.70] # 3 steps
    proof_b = [0.90] * 20        # 20 steps
    
    prod_a = math.prod(proof_a)
    prod_b = math.prod(proof_b)
    
    min_a = min(proof_a)
    min_b = min(proof_b)
    
    assert prod_a > prod_b, "Product aggregation exhibits pathological length penalty"
    assert min_b > min_a, "Min aggregation correctly favors higher quality formal proof"
    print(f"Proof A (3 steps):  Product={prod_a:.4f}, Min={min_a:.4f}")
    print(f"Proof B (20 steps): Product={prod_b:.4f}, Min={min_b:.4f}")
    print("Verification success: Min aggregation eliminates trajectory length bias!\n")


class ProcessRewardModelHead(nn.Module):
    """
    Step-level reward model head that scores representation vectors
    at intermediate step delimiter tokens.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_embeddings: Tensor of shape (B, K, hidden_dim)
        Returns:
            step_logits: Tensor of shape (B, K)
        """
        return self.value_head(step_embeddings).squeeze(-1)


def test_prm_head_module():
    print("--- 3. Testing PyTorch PRM Value Head Module ---")
    torch.manual_seed(42)
    B, K, D = 2, 4, 32
    prm = ProcessRewardModelHead(hidden_dim=D)
    
    step_embeds = torch.randn(B, K, D, requires_grad=True)
    step_labels = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0]
    ])
    
    logits = prm(step_embeds)
    assert logits.shape == (B, K)
    
    loss = F.binary_cross_entropy_with_logits(logits, step_labels)
    loss.backward()
    
    assert step_embeds.grad is not None, "Gradients failed to flow back to step embeddings"
    print(f"PRM Forward Logits Shape: {logits.shape}")
    print(f"PRM Step-Level Loss: {loss.item():.4f}")
    print("PRM Module backward pass verified: Step-level credit assignment working cleanly.")
    print("All tests in Chapter 13.4 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_aggregation_comparison()
    test_prm_head_module()
    print("🟢 Chapter 13.4 Verification 100% Complete.")
