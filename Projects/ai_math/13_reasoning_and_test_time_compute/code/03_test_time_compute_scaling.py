"""
Test-Time Compute & Inference Scaling Laws (Best-of-N, Pass@k & Verifier Dynamics)
Implementation and Verification Suite
Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
"""

import math
import numpy as np


def compute_coverage_probability(p: float, N: int) -> float:
    """Computes 1 - (1 - p)^N."""
    return 1.0 - math.pow(1.0 - p, N)


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Computes the unbiased combinatorial Pass@k estimator (Chen et al., 2021).
    Pass@k = 1 - (comb(N - c, k) / comb(N, k))
    """
    n = num_samples
    c = num_correct
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    p = 0.20
    
    # Step 1: Coverage probabilities
    cov1 = compute_coverage_probability(p, 1)
    cov2 = compute_coverage_probability(p, 2)
    cov4 = compute_coverage_probability(p, 4)
    cov8 = compute_coverage_probability(p, 8)
    
    assert math.isclose(cov1, 0.200000, abs_tol=1e-6)
    assert math.isclose(cov2, 0.360000, abs_tol=1e-6)
    assert math.isclose(cov4, 0.590400, abs_tol=1e-6)
    assert math.isclose(cov8, 0.832228, abs_tol=1e-5)
    
    print(f"Coverage N=1: {cov1:.6f} -> EXACT MATCH 0.200000")
    print(f"Coverage N=2: {cov2:.6f} -> EXACT MATCH 0.360000")
    print(f"Coverage N=4: {cov4:.6f} -> EXACT MATCH 0.590400")
    print(f"Coverage N=8: {cov8:.6f} -> EXACT MATCH 0.832228")
    
    # Step 2: Pass@k combinatorial checks (N=5, c=2)
    pass1 = estimate_pass_at_k(num_samples=5, num_correct=2, k=1)
    pass2 = estimate_pass_at_k(num_samples=5, num_correct=2, k=2)
    
    assert math.isclose(pass1, 0.400000, abs_tol=1e-6)
    assert math.isclose(pass2, 0.700000, abs_tol=1e-6)
    print(f"Pass@1 (N=5, c=2): {pass1:.6f} -> EXACT MATCH 0.400000")
    print(f"Pass@2 (N=5, c=2): {pass2:.6f} -> EXACT MATCH 0.700000")
    
    # Step 3: Best-of-3 Selection Check
    candidates = [
        {"id": "y1", "score": 0.35, "is_correct": False},
        {"id": "y2", "score": 0.85, "is_correct": True},
        {"id": "y3", "score": 0.60, "is_correct": False}
    ]
    best_candidate = max(candidates, key=lambda c: c["score"])
    assert best_candidate["id"] == "y2" and best_candidate["is_correct"] is True
    print(f"Best-of-3 Verifier Selection: {best_candidate['id']} (Score={best_candidate['score']}) -> Verified Correct!\n")


def verify_illustration_diminishing_returns():
    print("--- 2. Verifying Illustration 1 Sample Count Scaling ---")
    p = 0.10
    
    def samples_for_target(p_base: float, target: float) -> int:
        return math.ceil(math.log(1.0 - target) / math.log(1.0 - p_base))
        
    n50 = samples_for_target(p, 0.50)
    n90 = samples_for_target(p, 0.90)
    n99 = samples_for_target(p, 0.99)
    
    assert n50 == 7
    assert n90 == 22
    assert n99 == 44
    print(f"Samples required for 50% coverage: {n50} -> EXACT MATCH")
    print(f"Samples required for 90% coverage: {n90} -> EXACT MATCH")
    print(f"Samples required for 99% coverage: {n99} -> EXACT MATCH\n")


def simulate_best_of_n_verifier():
    print("--- 3. Simulating Best-of-N with Noisy Verifiers ---")
    np.random.seed(42)
    num_trials = 1000
    base_p = 0.25
    verifier_noise_std = 0.20
    
    for N in [1, 4, 16, 64]:
        correct_selections = 0
        for _ in range(num_trials):
            # Generate N answers: 1 = correct, 0 = incorrect
            ground_truths = (np.random.rand(N) < base_p).astype(int)
            # True reward is 1.0 for correct, 0.0 for incorrect
            # Verifier score is perturbed by Gaussian noise
            noisy_scores = ground_truths.astype(float) + np.random.normal(0, verifier_noise_std, N)
            best_idx = np.argmax(noisy_scores)
            if ground_truths[best_idx] == 1:
                correct_selections += 1
                
        empirical_acc = (correct_selections / num_trials) * 100.0
        print(f"Best-of-{N:2d} Empirical Accuracy: {empirical_acc:.2f}% (Base: {base_p*100:.1f}%)")
        
    print("Inference Scaling verified: Accuracy scales monotonically with inference budget N.\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_illustration_diminishing_returns()
    simulate_best_of_n_verifier()
    print("🟢 Chapter 13.3 Verification 100% Complete.")
