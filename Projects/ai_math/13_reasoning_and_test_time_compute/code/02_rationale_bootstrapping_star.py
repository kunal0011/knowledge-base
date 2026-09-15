"""
Rationale Bootstrapping: The Self-Taught Reasoner (STaR) & Verification Filtering
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    
    # Problems and ground truths
    problems = [
        {"id": "Q1", "target": 15, "pred_init": 15, "rationale_init": "5 + 10 = 15"},
        {"id": "Q2", "target": 42, "pred_init": 4.67, "rationale_init": "14 / 3 = 4.67"},
        {"id": "Q3", "target": 7,  "pred_init": 7,  "rationale_init": "2x=14 -> x=7"}
    ]
    
    # E-step Filtering
    filtered_data = []
    failed_problems = []
    
    for p in problems:
        if math.isclose(p["pred_init"], p["target"], abs_tol=1e-3):
            filtered_data.append((p["id"], p["rationale_init"], p["target"]))
        else:
            failed_problems.append(p)
            
    assert len(filtered_data) == 2, f"Expected 2 passed problems, got {len(filtered_data)}"
    assert len(failed_problems) == 1 and failed_problems[0]["id"] == "Q2"
    print(f"E-step: Filtered passed: {[x[0] for x in filtered_data]}, Failed: {[x['id'] for x in failed_problems]}")
    
    # Rationalization of Q2
    q2 = failed_problems[0]
    # Condition on ground truth hint
    rationalized_trace = "Distance = speed * time = 14 * 3 = 42"
    pred_from_hint = 42.0
    # Verification without hint
    assert math.isclose(pred_from_hint, q2["target"], abs_tol=1e-3)
    filtered_data.append((q2["id"], rationalized_trace, q2["target"]))
    assert len(filtered_data) == 3
    print("Rationalization rescue of Q2: Successfully added to training set.")
    
    # M-step Loss Verification
    # Probs: [0.50, 0.80, 0.70, 0.90]
    probs = [0.50, 0.80, 0.70, 0.90]
    losses = [-math.log(p) for p in probs]
    expected_losses = [0.693147, 0.223144, 0.356675, 0.105361]
    for l, exp_l in zip(losses, expected_losses):
        assert math.isclose(l, exp_l, abs_tol=1e-5)
        
    mean_loss = sum(losses) / len(losses)
    expected_mean_loss = 0.344582
    assert math.isclose(mean_loss, expected_mean_loss, abs_tol=1e-5)
    print(f"M-step Cross-Entropy Loss: {mean_loss:.6f} -> EXACT MATCH {expected_mean_loss}\n")


class MockReasoner:
    """Simulates a language model whose rationale capability improves over STaR iterations."""
    def __init__(self):
        # Learned knowledge base: mappings from problem key to (rationale, answer)
        self.learned_knowledge = {
            "Q1": ("5 + 10 = 15", 15),
            "Q3": ("2x + 1 = 15 => x = 7", 7)
        }

    def generate(self, question_id: str, hint: int | None = None):
        if hint is not None:
            # Rationalization: if given hint for Q2, backward-deduces proper equation
            if question_id == "Q2" and hint == 42:
                return "Distance = 14 * 3 = 42", 42
        # Default policy
        if question_id in self.learned_knowledge:
            return self.learned_knowledge[question_id]
        else:
            # Flawed default reasoning
            return "14 / 3 = 4.67", 4.67

    def train(self, dataset: list[tuple[str, str, int]]):
        for q_id, rationale, target in dataset:
            self.learned_knowledge[q_id] = (rationale, target)


def test_star_bootstrap_loop():
    print("--- 2. Testing Complete STaR Self-Taught Bootstrap Loop ---")
    dataset = [
        {"id": "Q1", "target": 15},
        {"id": "Q2", "target": 42},
        {"id": "Q3", "target": 7}
    ]
    
    model = MockReasoner()
    
    # Iteration 0: Baseline evaluation
    correct_iter0 = 0
    training_buffer = []
    for item in dataset:
        rationale, pred = model.generate(item["id"])
        if pred == item["target"]:
            correct_iter0 += 1
            training_buffer.append((item["id"], rationale, item["target"]))
        else:
            # Rationalization step
            rat_trace, rat_pred = model.generate(item["id"], hint=item["target"])
            if rat_pred == item["target"]:
                training_buffer.append((item["id"], rat_trace, item["target"]))
                
    acc_iter0 = (correct_iter0 / len(dataset)) * 100.0
    assert math.isclose(acc_iter0, 66.6666, abs_tol=1e-3)
    print(f"Iteration 0 Accuracy: {acc_iter0:.1f}% (2/3 solved unaided)")
    
    # M-Step: Train model on bootstrapped training buffer
    assert len(training_buffer) == 3
    model.train(training_buffer)
    print("M-step complete: Model fine-tuned on self-generated and rationalized traces.")
    
    # Iteration 1: Test without hints
    correct_iter1 = 0
    for item in dataset:
        rationale, pred = model.generate(item["id"])
        if pred == item["target"]:
            correct_iter1 += 1
            
    acc_iter1 = (correct_iter1 / len(dataset)) * 100.0
    assert acc_iter1 == 100.0
    print(f"Iteration 1 Accuracy: {acc_iter1:.1f}% (3/3 solved unaided!)")
    print("STaR Bootstrap Loop successfully verified: 66.7% -> 100.0% accuracy progression!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    test_star_bootstrap_loop()
    print("🟢 Chapter 13.2 Verification 100% Complete.")
