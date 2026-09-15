"""
Self-Correction, Backtracking & External Verification Loops (Python REPL Sandbox)
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
import torch
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    
    # Dot products: [3.0, -1.0, 4.0]
    dot_products = torch.tensor([3.0, -1.0, 4.0], dtype=torch.float32)
    attention_weights = F.softmax(dot_products, dim=-1)
    
    expected_weights = torch.tensor([0.267623, 0.004902, 0.727475], dtype=torch.float32)
    assert torch.allclose(attention_weights, expected_weights, atol=1e-5), f"Weights mismatch: {attention_weights}"
    
    a_quest = attention_weights[0].item()
    a_flawed = attention_weights[1].item()
    a_error = attention_weights[2].item()
    
    assert a_flawed < 0.005, "Flawed token should be suppressed to < 0.5%"
    assert a_error > 0.70, "Compiler error token should dominate attention (> 70%)"
    
    print(f"Attention to Question:       {a_quest*100:.2f}% (Retains objective)")
    print(f"Attention to Flawed Token:   {a_flawed*100:.2f}% (Suppressed to < 0.5%!)")
    print(f"Attention to Compiler Error: {a_error*100:.2f}% (Dominant focus)")
    print("Attention weight re-allocation verified with zero error!\n")


class PythonExecutionSandbox:
    """Simulates an isolated execution environment for verifying reasoning solutions."""
    def run_assertion(self, code_str: str) -> tuple[bool, str]:
        local_scope = {}
        try:
            exec(code_str, {}, local_scope)
            return True, "SUCCESS"
        except AssertionError as e:
            return False, f"AssertionError: {str(e)}"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}"


class GroundedSelfCorrectingAgent:
    """Simulates an agent that uses Python sandbox feedback to self-correct reasoning."""
    def __init__(self, sandbox: PythonExecutionSandbox):
        self.sandbox = sandbox

    def solve(self, problem_statement: str) -> tuple[int, list[str]]:
        history = []
        
        # Round 0: Flawed initial mental attempt
        code_round0 = """
# Compute 1^3 + 2^3 + 3^3
ans = 1 + 8 + 16 # Oops, 3^3 is not 16
assert ans == 36, f"Computed {ans} != expected 36"
"""
        history.append("Initial Attempt: ans = 1 + 8 + 16 = 25")
        success, feedback = self.sandbox.run_assertion(code_round0)
        history.append(f"Execution Feedback: {feedback}")
        
        if not success:
            # Round 1: Grounded correction triggered by feedback
            # "Wait! 3^3 is 27, not 16!"
            code_round1 = """
# Corrected: 3^3 = 27
ans = 1 + 8 + 27
assert ans == 36, f"Computed {ans} != expected 36"
"""
            history.append("Backtrack: Wait, 3^3 is 27, not 16! Revised: 1 + 8 + 27 = 36")
            success1, feedback1 = self.sandbox.run_assertion(code_round1)
            history.append(f"Execution Feedback: {feedback1}")
            assert success1 is True
            return 36, history

        return 25, history


def test_grounded_self_correction():
    print("--- 2. Testing Grounded Self-Correction with Execution Sandbox ---")
    sandbox = PythonExecutionSandbox()
    agent = GroundedSelfCorrectingAgent(sandbox)
    
    final_ans, history = agent.solve("Compute sum of cubes 1..3")
    assert final_ans == 36
    
    for entry in history:
        print(f"  [Agent Step] {entry}")
        
    print(f"\nFinal Grounded Answer: {final_ans} -> Verified 100% Correct!")
    print("All tests in Chapter 13.8 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    test_grounded_self_correction()
    print("🟢 Chapter 13.8 Verification 100% Complete.")
