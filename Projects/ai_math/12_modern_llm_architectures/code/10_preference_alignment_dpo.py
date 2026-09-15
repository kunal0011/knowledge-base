"""
Post-Training II: Direct Preference Alignment (DPO & SimPO)
Implementation and Verification Suite
Module 12: Modern LLM Architectures & Engineering from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    beta_dpo = 0.5
    
    # Log probabilities
    policy_w = torch.tensor(-1.200000, dtype=torch.float32)
    policy_l = torch.tensor(-2.600000, dtype=torch.float32)
    ref_w = torch.tensor(-2.000000, dtype=torch.float32)
    ref_l = torch.tensor(-3.000000, dtype=torch.float32)
    
    # Step 1: Log-ratios
    ratio_w = policy_w - ref_w # +0.800000
    ratio_l = policy_l - ref_l # +0.400000
    assert math.isclose(ratio_w.item(), 0.800000, abs_tol=1e-5)
    assert math.isclose(ratio_l.item(), 0.400000, abs_tol=1e-5)
    print(f"Log-ratios: Winner={ratio_w.item():.4f}, Loser={ratio_l.item():.4f}")
    
    # Step 2: Implicit reward difference
    delta_r = beta_dpo * (ratio_w - ratio_l) # 0.5 * 0.4 = 0.200000
    assert math.isclose(delta_r.item(), 0.200000, abs_tol=1e-5)
    print(f"Implicit reward difference delta_r: {delta_r.item():.6f} -> EXACT MATCH 0.200000")
    
    # Step 3: Sigmoid probability
    prob = torch.sigmoid(delta_r)
    expected_prob = 0.549834
    assert math.isclose(prob.item(), expected_prob, abs_tol=1e-5)
    print(f"Sigmoid preference prob: {prob.item():.6f} -> EXACT MATCH {expected_prob}")
    
    # Step 4: DPO Loss
    loss_dpo = -F.logsigmoid(delta_r)
    expected_dpo_loss = 0.598139
    assert math.isclose(loss_dpo.item(), expected_dpo_loss, abs_tol=1e-5)
    print(f"DPO Loss: {loss_dpo.item():.6f} -> EXACT MATCH {expected_dpo_loss}")
    
    # Step 5: SimPO Trace
    len_w = 2.0
    len_l = 4.0
    gamma = 0.50
    beta_simpo = 1.0
    
    r_simpo_w = (beta_simpo / len_w) * policy_w # -0.600000
    r_simpo_l = (beta_simpo / len_l) * policy_l # -0.650000
    assert math.isclose(r_simpo_w.item(), -0.600000, abs_tol=1e-5)
    assert math.isclose(r_simpo_l.item(), -0.650000, abs_tol=1e-5)
    
    delta_simpo = (r_simpo_w - r_simpo_l) - gamma # (-0.60 - (-0.65)) - 0.50 = 0.05 - 0.50 = -0.450000
    assert math.isclose(delta_simpo.item(), -0.450000, abs_tol=1e-5)
    
    loss_simpo = -F.logsigmoid(delta_simpo)
    expected_simpo_loss = 0.943249
    assert math.isclose(loss_simpo.item(), expected_simpo_loss, abs_tol=1e-5)
    print(f"SimPO Loss: {loss_simpo.item():.6f} -> EXACT MATCH {expected_simpo_loss}\n")


def verify_verbosity_illustration():
    print("--- 2. Verifying Illustration 1: Verbosity Bias & SimPO Resolution ---")
    beta = 0.5
    # Short response (10 tokens, high quality)
    logp_pi_short = -5.0
    logp_ref_short = -10.0
    r_dpo_short = beta * (logp_pi_short - logp_ref_short) # 0.5 * 5.0 = 2.5
    
    # Verbose response (100 tokens, lower quality)
    logp_pi_verbose = -80.0
    logp_ref_verbose = -100.0
    r_dpo_verbose = beta * (logp_pi_verbose - logp_ref_verbose) # 0.5 * 20.0 = 10.0
    
    assert r_dpo_verbose > r_dpo_short, "DPO should exhibit verbosity bias in this setup"
    print(f"DPO Rewards: Short={r_dpo_short:.2f}, Verbose={r_dpo_verbose:.2f} (Pathological bias towards length)")
    
    # SimPO normalized rewards
    r_simpo_short = logp_pi_short / 10.0   # -0.50
    r_simpo_verbose = logp_pi_verbose / 100.0 # -0.80
    assert r_simpo_short > r_simpo_verbose, "SimPO should prefer short high-quality response"
    print(f"SimPO Rewards: Short={r_simpo_short:.2f}, Verbose={r_simpo_verbose:.2f} (Correctly favors high quality!)\n")


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss Module.
    Computes -log sigma(beta * (log(pi_w/ref_w) - log(pi_l/ref_l)))
    """
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = self.beta * (pi_logratios - ref_logratios)
        losses = -F.logsigmoid(logits)
        return losses.mean()


class SimPOLoss(nn.Module):
    """
    Simple Preference Optimization (SimPO) Loss Module.
    Reference-free, length-normalized, with target margin gamma.
    """
    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        chosen_lens: torch.Tensor,
        rejected_lens: torch.Tensor
    ) -> torch.Tensor:
        r_chosen = (self.beta / chosen_lens) * policy_chosen_logps
        r_rejected = (self.beta / rejected_lens) * policy_rejected_logps
        logits = (r_chosen - r_rejected) - self.gamma
        losses = -F.logsigmoid(logits)
        return losses.mean()


def test_dpo_and_simpo_modules():
    print("--- 3. Testing PyTorch DPO and SimPO Loss Modules ---")
    torch.manual_seed(42)
    B = 4
    
    # Simulate policy output with gradients enabled
    policy_w = torch.randn(B, requires_grad=True)
    policy_l = torch.randn(B, requires_grad=True)
    ref_w = torch.randn(B)
    ref_l = torch.randn(B)
    
    dpo = DPOLoss(beta=0.1)
    loss = dpo(policy_w, policy_l, ref_w, ref_l)
    assert loss.ndim == 0 and loss.item() > 0.0
    loss.backward()
    
    assert policy_w.grad is not None and policy_l.grad is not None
    # Gradients should have opposite signs: pushing w up, pushing l down
    assert torch.all(policy_w.grad < 0), "DPO gradient should pull chosen log-prob up (loss negative grad)"
    assert torch.all(policy_l.grad > 0), "DPO gradient should push rejected log-prob down"
    print("DPO backward pass verified: Gradients actively pull chosen UP and push rejected DOWN.")
    
    # Test SimPO
    simpo = SimPOLoss(beta=1.0, gamma=0.5)
    lens_w = torch.tensor([10, 15, 20, 25], dtype=torch.float32)
    lens_l = torch.tensor([30, 40, 50, 60], dtype=torch.float32)
    simpo_loss = simpo(policy_w, policy_l, lens_w, lens_l)
    assert simpo_loss.ndim == 0 and simpo_loss.item() > 0.0
    print("SimPO forward pass verified successfully.")
    print("All preference alignment tests passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_verbosity_illustration()
    test_dpo_and_simpo_modules()
    print("🟢 Chapter 12.10 Verification 100% Complete.")
