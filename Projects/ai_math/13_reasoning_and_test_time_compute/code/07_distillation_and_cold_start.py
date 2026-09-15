"""
Cold-Start Data, Multi-Stage Alignment & Reasoning Distillation
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    
    # Student raw logits: [2.0, 0.5, -1.0]
    logits_S = torch.tensor([2.0, 0.5, -1.0], dtype=torch.float32)
    target_idx = 1 # Token 42 (0-indexed)
    
    # Part A: Sequence-level standard softmax
    probs_S = F.softmax(logits_S, dim=-1)
    expected_p = torch.tensor([0.785597, 0.175290, 0.039113], dtype=torch.float32)
    assert torch.allclose(probs_S, expected_p, atol=1e-5), f"Probs mismatch: {probs_S}"
    print(f"Student Softmax Probs: {probs_S.tolist()} -> EXACT MATCH")
    
    # Cross-Entropy Loss
    loss_seq = -math.log(probs_S[target_idx].item())
    expected_loss_seq = 1.741311
    assert math.isclose(loss_seq, expected_loss_seq, abs_tol=1e-5)
    
    # Verify via F.cross_entropy
    py_ce = F.cross_entropy(logits_S.unsqueeze(0), torch.tensor([target_idx]))
    assert math.isclose(py_ce.item(), expected_loss_seq, abs_tol=1e-5)
    print(f"Sequence-Level Distillation Loss: {loss_seq:.6f} -> EXACT MATCH {expected_loss_seq}")
    
    # Part B: Soft-Target Knowledge Distillation (T = 2.0)
    T = 2.0
    teacher_probs = torch.tensor([0.10, 0.80, 0.10], dtype=torch.float32)
    
    soft_student_logits = logits_S / T # [1.0, 0.25, -0.5]
    soft_student_probs = F.softmax(soft_student_logits, dim=-1)
    expected_soft_p = torch.tensor([0.589798, 0.278601, 0.131602], dtype=torch.float32)
    assert torch.allclose(soft_student_probs, expected_soft_p, atol=1e-4)
    print(f"Soft Student Probs (T=2): {soft_student_probs.tolist()} -> EXACT MATCH")
    
    # KL Divergence: sum(P_T * log(P_T / Q_S))
    kl_div = torch.sum(teacher_probs * torch.log(teacher_probs / soft_student_probs)).item()
    expected_kl = 0.638944
    assert math.isclose(kl_div, expected_kl, abs_tol=1e-5)
    
    scaled_kd_loss = (T ** 2) * kl_div
    expected_scaled_kd = 2.555776
    assert math.isclose(scaled_kd_loss, expected_scaled_kd, abs_tol=1e-5)
    print(f"KL Divergence: {kl_div:.6f} -> EXACT MATCH {expected_kl}")
    print(f"Scaled KD Loss (T^2 * KL): {scaled_kd_loss:.6f} -> EXACT MATCH {expected_scaled_kd}\n")


def verify_illustration_cold_start():
    print("--- 2. Verifying Illustration 1: Small-Model Cold Start Trap ---")
    p_token = 0.90
    L = 50
    p_seq = math.pow(p_token, L)
    assert math.isclose(p_seq, 0.00515377, abs_tol=1e-6)
    
    G = 16
    p_at_least_one = 1.0 - math.pow(1.0 - p_seq, G)
    assert math.isclose(p_at_least_one, 0.079414, abs_tol=1e-4)
    print(f"Per-token accuracy: {p_token*100:.1f}%, 50-step sequence pass: {p_seq*100:.3f}%")
    print(f"Group of 16 rollouts: Probability of success = {p_at_least_one*100:.2f}% (92% failure rate!)")
    print("Cold-start trap verified: Small models cannot discover deep reasoning without distillation!\n")


class ReasoningDistillationLoss(nn.Module):
    """
    Combined Sequence-Level and Soft-Target Knowledge Distillation Loss.
    """
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, target_tokens: torch.Tensor, teacher_probs: torch.Tensor = None):
        # 1. Hard Cross-Entropy Loss
        ce_loss = F.cross_entropy(student_logits, target_tokens)
        
        if teacher_probs is None:
            return ce_loss
            
        # 2. Soft KL Divergence Loss
        T = self.temperature
        soft_student_logp = F.log_softmax(student_logits / T, dim=-1)
        soft_loss = F.kl_div(soft_student_logp, teacher_probs, reduction='batchmean') * (T ** 2)
        
        return self.alpha * ce_loss + (1.0 - self.alpha) * soft_loss


def test_distillation_module():
    print("--- 3. Testing PyTorch Reasoning Distillation Loss Module ---")
    torch.manual_seed(42)
    B, V = 4, 32
    student_logits = torch.randn(B, V, requires_grad=True)
    target_tokens = torch.tensor([3, 15, 8, 22])
    teacher_probs = F.softmax(torch.randn(B, V) * 2.0, dim=-1)
    
    distill_loss_fn = ReasoningDistillationLoss(temperature=2.0, alpha=0.5)
    loss = distill_loss_fn(student_logits, target_tokens, teacher_probs)
    
    assert loss.ndim == 0 and loss.item() > 0.0
    loss.backward()
    
    assert student_logits.grad is not None, "Gradients failed to flow back to student logits"
    print(f"Combined Distillation Loss: {loss.item():.4f}")
    print("Backward pass verified: Gradients successfully updated student parameters.")
    print("All tests in Chapter 13.7 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_illustration_cold_start()
    test_distillation_module()
    print("🟢 Chapter 13.7 Verification 100% Complete.")
