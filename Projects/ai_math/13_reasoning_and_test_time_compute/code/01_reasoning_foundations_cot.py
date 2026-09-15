"""
Foundations of Reasoning: System 1 vs. System 2 & Chain-of-Thought Dynamics
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    
    # Probabilities
    p_zA = 0.60
    p_zB = 0.30
    
    p_y1_given_zA = 0.90
    p_y2_given_zA = 0.10
    
    p_y1_given_zB = 0.20
    p_y2_given_zB = 0.80
    
    # Step 1: Joint probabilities
    joint_zA_y1 = p_zA * p_y1_given_zA
    joint_zA_y2 = p_zA * p_y2_given_zA
    joint_zB_y1 = p_zB * p_y1_given_zB
    joint_zB_y2 = p_zB * p_y2_given_zB
    
    assert math.isclose(joint_zA_y1, 0.540000, abs_tol=1e-6)
    assert math.isclose(joint_zA_y2, 0.060000, abs_tol=1e-6)
    assert math.isclose(joint_zB_y1, 0.060000, abs_tol=1e-6)
    assert math.isclose(joint_zB_y2, 0.240000, abs_tol=1e-6)
    
    print(f"Joint P(zA, y1) = {joint_zA_y1:.6f}, P(zA, y2) = {joint_zA_y2:.6f}")
    print(f"Joint P(zB, y1) = {joint_zB_y1:.6f}, P(zB, y2) = {joint_zB_y2:.6f} -> EXACT MATCH")
    
    # Step 2: Marginalization
    p_y1 = joint_zA_y1 + joint_zB_y1 # 0.600000
    p_y2 = joint_zA_y2 + joint_zB_y2 # 0.300000
    assert math.isclose(p_y1, 0.600000, abs_tol=1e-6)
    assert math.isclose(p_y2, 0.300000, abs_tol=1e-6)
    
    ratio = p_y1 / p_y2
    assert math.isclose(ratio, 2.000000, abs_tol=1e-6)
    print(f"Marginal P(y1) = {p_y1:.6f}, Marginal P(y2) = {p_y2:.6f} (Ratio: {ratio:.1f}x) -> EXACT MATCH")
    
    # Step 3: Self-Consistency Majority Vote
    samples = ["y1", "y1", "y2", "y1", "y1"]
    vote_counts = Counter(samples)
    winner, top_votes = vote_counts.most_common(1)[0]
    confidence = top_votes / len(samples)
    
    assert winner == "y1"
    assert math.isclose(confidence, 0.80, abs_tol=1e-5)
    print(f"Self-consistency vote winner: {winner} with {confidence*100:.1f}% consensus -> EXACT MATCH\n")


def simulate_parity_reasoning():
    print("--- 2. Verifying Parity Reasoning: System 1 (Direct) vs System 2 (CoT) ---")
    
    def direct_system1_parity(bits: list[int]) -> int:
        # A mock shallow linear sum without scratchpad: susceptible to parity saturation on long inputs
        # Models the inability of a single constant-depth step to compute unbounded XOR
        return sum(bits) % 2

    def cot_system2_parity(bits: list[int]) -> tuple[int, list[int]]:
        # Emits intermediate cumulative parity tokens: z_k = (z_{k-1} + b_k) % 2
        scratchpad = []
        running_parity = 0
        for b in bits:
            running_parity = (running_parity + b) % 2
            scratchpad.append(running_parity)
        return running_parity, scratchpad

    test_bits = [1, 0, 1, 1, 1, 0, 1, 1] # 6 ones -> parity 0
    ans_s1 = direct_system1_parity(test_bits)
    ans_s2, trace = cot_system2_parity(test_bits)
    
    assert ans_s1 == 0
    assert ans_s2 == 0
    assert trace == [1, 1, 0, 1, 0, 0, 1, 0]
    print(f"Input bits: {test_bits} (Sum = {sum(test_bits)})")
    print(f"CoT step-by-step trace: {trace}")
    print(f"Final Parity Answer: {ans_s2} -> Verified 100% Correct!\n")


class SelfConsistencySampler:
    """
    Simulates Self-Consistency (Wang et al., 2022) with Temperature Sampling.
    """
    def __init__(self, answer_dist: dict[str, float]):
        self.answers = list(answer_dist.keys())
        self.probs = torch.tensor(list(answer_dist.values()), dtype=torch.float32)

    def sample(self, num_samples: int = 10, temperature: float = 0.7) -> tuple[str, float]:
        logits = torch.log(self.probs) / temperature
        sampled_indices = torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples, replacement=True)
        sampled_answers = [self.answers[idx.item()] for idx in sampled_indices]
        
        counts = Counter(sampled_answers)
        winner, count = counts.most_common(1)[0]
        consensus = count / num_samples
        return winner, consensus


def test_self_consistency():
    print("--- 3. Testing Self-Consistency Sampler ---")
    torch.manual_seed(42)
    # True underlying marginal: y1: 0.60, y2: 0.30, y3: 0.10
    dist = {"y1": 0.60, "y2": 0.30, "y3": 0.10}
    sampler = SelfConsistencySampler(dist)
    
    winner, consensus = sampler.sample(num_samples=20, temperature=0.5)
    assert winner == "y1", f"Expected y1 to win majority vote, got {winner}"
    print(f"Self-Consistency over 20 rollouts: Winner={winner}, Consensus={consensus*100:.1f}%")
    print("All tests in Chapter 13.1 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    simulate_parity_reasoning()
    test_self_consistency()
    print("🟢 Chapter 13.1 Verification 100% Complete.")
