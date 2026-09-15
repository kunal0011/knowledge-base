"""
Chapter 9.6: Alignment & Post-Training (SFT, RLHF with PPO, DPO)
================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification (DPO Loss & Gradients)
2. SFT Prompt Masking Verification (ignore_index = -100)
3. Bradley-Terry Reward Model Loss Verification
4. End-to-end DPO mini-training loop: successfully flipping an inverted preference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid DPO Hand Arithmetic Verification
# ----------------------------------------------------------------------
def dpo_loss_formula(pi_w, pi_l, ref_w, ref_l, beta=0.5):
    """
    Computes DPO loss for scalar log-probabilities
    """
    ratio_w = pi_w - ref_w
    ratio_l = pi_l - ref_l
    delta_r = beta * (ratio_w - ratio_l)
    
    loss = -F.logsigmoid(delta_r)
    error_weight = torch.sigmoid(-delta_r)
    return loss, delta_r, error_weight

def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid DPO Hand Arithmetic Verification ---")
    beta = 0.5
    ref_w = torch.tensor(-2.0, dtype=torch.float64)
    ref_l = torch.tensor(-2.0, dtype=torch.float64)
    
    # Model mistakenly prefers y_l over y_w
    pi_w = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
    pi_l = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    
    loss, delta_r, error_weight = dpo_loss_formula(pi_w, pi_l, ref_w, ref_l, beta)
    
    # Expected hand values:
    delta_r_exp = 0.5 * ((-1.5 - (-2.0)) - (-1.0 - (-2.0))) # 0.5 * (0.5 - 1.0) = -0.25
    loss_exp = -np.log(1.0 / (1.0 + np.exp(0.25)))          # -log(0.437824) = 0.825965
    weight_exp = 1.0 / (1.0 + np.exp(-0.25))                # 0.562176
    
    print(f"Computed Margin Delta r: {delta_r.item():.6f} | Expected: {delta_r_exp:.6f}")
    print(f"Computed DPO Loss:      {loss.item():.6f} | Expected: {loss_exp:.6f}")
    print(f"Computed Error Weight:  {error_weight.item():.6f} | Expected: {weight_exp:.6f}")
    
    assert abs(delta_r.item() - delta_r_exp) < 1e-6
    assert abs(loss.item() - loss_exp) < 1e-5
    assert abs(error_weight.item() - weight_exp) < 1e-5
    
    # Backward pass
    loss.backward()
    grad_w_exp = -beta * weight_exp # -0.5 * 0.562176 = -0.281088
    grad_l_exp = +beta * weight_exp # +0.5 * 0.562176 = +0.281088
    
    print(f"Computed dL/d(pi_w): {pi_w.grad.item():.6f} | Expected: {grad_w_exp:.6f}")
    print(f"Computed dL/d(pi_l): {pi_l.grad.item():.6f} | Expected: {grad_l_exp:.6f}")
    
    assert abs(pi_w.grad.item() - grad_w_exp) < 1e-5
    assert abs(pi_l.grad.item() - grad_l_exp) < 1e-5
    print("✓ Part 5 Visual Grid DPO hand arithmetic verified!\n")

# ----------------------------------------------------------------------
# 2. SFT Prompt Masking Verification (ignore_index = -100)
# ----------------------------------------------------------------------
def test_sft_prompt_masking():
    print("--- 2. SFT Prompt Masking Verification (ignore_index = -100) ---")
    vocab_size = 10
    seq_len = 5
    
    logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
    
    # Target: prompt tokens (first 2) are -100, assistant tokens (last 3) are valid IDs
    targets = torch.tensor([[-100, -100, 3, 5, 2]])
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    
    # Verify that gradients for prompt positions 0 and 1 are STRICTLY 0.0
    prompt_grad_norm = logits.grad[0, 0:2].norm().item()
    assistant_grad_norm = logits.grad[0, 2:].norm().item()
    
    print(f"Prompt Positions (0-1) Gradient Norm:    {prompt_grad_norm:.2e}")
    print(f"Assistant Positions (2-4) Gradient Norm: {assistant_grad_norm:.4f}")
    
    assert prompt_grad_norm == 0.0, "Prompt tokens leaked gradient during SFT!"
    assert assistant_grad_norm > 0.0, "Assistant tokens must have active gradient!"
    print("✓ SFT prompt masking strictly verified: prompt tokens receive zero loss/gradient!\n")

# ----------------------------------------------------------------------
# 3. Bradley-Terry Reward Model Loss
# ----------------------------------------------------------------------
def test_bradley_terry_reward():
    print("--- 3. Bradley-Terry Reward Model Loss Verification ---")
    # Winner reward = 3.0, Loser reward = 1.0
    r_w = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    r_l = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    
    # Loss = -log(sigmoid(r_w - r_l))
    loss = -F.logsigmoid(r_w - r_l)
    loss_exp = -np.log(1.0 / (1.0 + np.exp(-2.0))) # -log(0.880797) = 0.126928
    
    print(f"Computed Bradley-Terry Loss: {loss.item():.6f} | Expected: {loss_exp:.6f}")
    assert abs(loss.item() - loss_exp) < 1e-5
    
    loss.backward()
    # dL/dr_w = -(1 - sigmoid(r_w - r_l)) = -(1 - 0.880797) = -0.119203
    assert abs(r_w.grad.item() - (-0.119203)) < 1e-5
    assert abs(r_l.grad.item() - (+0.119203)) < 1e-5
    print("✓ Bradley-Terry reward model forward and backward derivatives verified!\n")

# ----------------------------------------------------------------------
# 4. End-to-End DPO Mini-Training Loop (Flipping Inverted Preference)
# ----------------------------------------------------------------------
def test_dpo_training_flip():
    print("--- 4. End-to-End DPO Mini-Training Loop (Flipping Preferences) ---")
    torch.manual_seed(42)
    # Simple linear policy mapping input to 2 completions
    # We want policy to prefer candidate 0 (winner) over candidate 1 (loser)
    
    # Initial weights strongly favor candidate 1 (loser)
    policy_logits = nn.Parameter(torch.tensor([-1.0, 2.0], dtype=torch.float64))
    ref_logits = torch.tensor([0.0, 0.0], dtype=torch.float64) # frozen reference
    
    optimizer = torch.optim.SGD([policy_logits], lr=0.5)
    beta = 0.5
    
    print(f"Initial Policy Logits: Winner={policy_logits[0].item():.2f}, Loser={policy_logits[1].item():.2f} (INVERTED!)")
    
    for step in range(25):
        optimizer.zero_grad()
        
        log_pi = F.log_softmax(policy_logits, dim=0)
        log_ref = F.log_softmax(ref_logits, dim=0)
        
        pi_w = log_pi[0]
        pi_l = log_pi[1]
        ref_w = log_ref[0]
        ref_l = log_ref[1]
        
        loss, delta_r, _ = dpo_loss_formula(pi_w, pi_l, ref_w, ref_l, beta=beta)
        loss.backward()
        optimizer.step()
        
    final_pi = F.softmax(policy_logits, dim=0)
    print(f"Final Policy Logits:   Winner={policy_logits[0].item():.2f}, Loser={policy_logits[1].item():.2f}")
    print(f"Final Probabilities:   P(Winner)={final_pi[0].item():.4f}, P(Loser)={final_pi[1].item():.4f}")
    
    # Policy must now strongly favor the winner over the loser
    assert policy_logits[0] > policy_logits[1], "DPO failed to flip preference order!"
    assert final_pi[0] > 0.85, f"Winner probability should be > 85%, got {final_pi[0].item():.4f}"
    print("✓ DPO successfully reversed the inverted preference, driving P(Winner) > 85%!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 9.6 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_sft_prompt_masking()
    test_bradley_terry_reward()
    test_dpo_training_flip()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 9.6 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
