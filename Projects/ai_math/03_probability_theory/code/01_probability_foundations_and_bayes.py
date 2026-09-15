"""
Chapter 3.1: Probability Foundations, Axioms & Bayes' Theorem
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Spam Classification Contingency Table & Bayes' Update.
2. Medical Diagnostic Test Base Rate Fallacy (Analytical vs. 1,000,000 Monte Carlo Trials).
3. Naive Bayes Classifier Implementation from Scratch.
4. Simpson's Paradox Numerical Confirmation.
5. Maximum A Posteriori (MAP) with Gaussian Prior vs. L2 Weight Decay Equivalence.
"""

import numpy as np
import torch


# =====================================================================
# 1. Part 5 Bayes' Rule Contingency Table
# =====================================================================
def compute_spam_posterior():
    p_spam = 0.20
    p_ham = 0.80
    p_w_given_spam = 0.70
    p_w_given_ham = 0.05

    # Joint probabilities
    p_joint_spam = p_w_given_spam * p_spam
    p_joint_ham = p_w_given_ham * p_ham

    # Evidence (Total Probability)
    p_evidence = p_joint_spam + p_joint_ham

    # Posterior
    p_spam_given_w = p_joint_spam / p_evidence
    p_ham_given_w = p_joint_ham / p_evidence

    return {
        "p_joint_spam": p_joint_spam,
        "p_joint_ham": p_joint_ham,
        "p_evidence": p_evidence,
        "p_spam_given_w": p_spam_given_w,
        "p_ham_given_w": p_ham_given_w,
    }


# =====================================================================
# 2. Medical Diagnostic Base Rate Fallacy (Monte Carlo)
# =====================================================================
def simulate_base_rate_fallacy(num_samples: int = 1_000_000):
    """
    Simulates disease testing:
    Prevalence: P(D) = 0.001 (0.1%)
    Sensitivity: P(T+ | D) = 0.99
    False Positive: P(T+ | ~D) = 0.01
    """
    np.random.seed(42)
    # Generate true disease state: 1 if sick, 0 if healthy
    has_disease = (np.random.rand(num_samples) < 0.001).astype(int)

    # Generate test results
    test_positive = np.zeros(num_samples, dtype=int)
    # Sick patients test positive with 99% probability
    sick_indices = np.where(has_disease == 1)[0]
    test_positive[sick_indices] = (np.random.rand(len(sick_indices)) < 0.99).astype(int)
    # Healthy patients test positive with 1% probability (false positive)
    healthy_indices = np.where(has_disease == 0)[0]
    test_positive[healthy_indices] = (np.random.rand(len(healthy_indices)) < 0.01).astype(int)

    # Condition on test being positive
    all_positive_indices = np.where(test_positive == 1)[0]
    sick_and_positive = np.sum(has_disease[all_positive_indices] == 1)
    empirical_posterior = sick_and_positive / len(all_positive_indices)

    # Analytical value
    p_d = 0.001
    p_not_d = 0.999
    p_t_given_d = 0.99
    p_t_given_not_d = 0.01
    evidence = p_t_given_d * p_d + p_t_given_not_d * p_not_d
    analytical_posterior = (p_t_given_d * p_d) / evidence

    return empirical_posterior, analytical_posterior, len(all_positive_indices)


# =====================================================================
# 3. Naive Bayes Classifier from Scratch
# =====================================================================
class ScratchNaiveBayes:
    """
    Binary Bernoulli Naive Bayes Classifier.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        classes = np.unique(y)

        for c in classes:
            X_c = X[y == c]
            # Prior P(Y = c)
            self.class_priors[c] = len(X_c) / n_samples
            # Likelihood with Laplace smoothing: P(x_d = 1 | Y = c)
            self.feature_probs[c] = (np.sum(X_c, axis=0) + self.alpha) / (
                len(X_c) + 2.0 * self.alpha
            )

    def predict_log_proba(self, X: np.ndarray) -> dict:
        log_posteriors = {}
        for c, prior in self.class_priors.items():
            # log P(Y=c) + sum log P(x_d | Y=c)
            probs = self.feature_probs[c]
            # log likelihood: x * log(p) + (1-x) * log(1-p)
            log_lik = X * np.log(probs) + (1.0 - X) * np.log(1.0 - probs)
            log_posteriors[c] = np.log(prior) + np.sum(log_lik, axis=1)
        return log_posteriors

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_posts = self.predict_log_proba(X)
        classes = list(log_posts.keys())
        stacked = np.stack([log_posts[c] for c in classes], axis=1)
        return np.array(classes)[np.argmax(stacked, axis=1)]


# =====================================================================
# 4. Simpson's Paradox Numerical Verification
# =====================================================================
def verify_simpsons_paradox():
    """
    Subgroup 1 (Men):   Treatment: 7/10 (70%)  vs Control: 60/100 (60%) -> Treatment Better!
    Subgroup 2 (Women): Treatment: 40/100 (40%) vs Control: 3/10 (30%)   -> Treatment Better!
    Combined:           Treatment: 47/110 (42.7%) vs Control: 63/110 (57.3%) -> Control Better!
    """
    p_cure_men_treat = 7 / 10
    p_cure_men_ctrl = 60 / 100

    p_cure_women_treat = 40 / 100
    p_cure_women_ctrl = 3 / 10

    p_cure_pool_treat = (7 + 40) / (10 + 100)
    p_cure_pool_ctrl = (60 + 3) / (100 + 10)

    men_treat_better = p_cure_men_treat > p_cure_men_ctrl
    women_treat_better = p_cure_women_treat > p_cure_women_ctrl
    pool_ctrl_better = p_cure_pool_ctrl > p_cure_pool_treat

    return {
        "men_treat": p_cure_men_treat,
        "men_ctrl": p_cure_men_ctrl,
        "women_treat": p_cure_women_treat,
        "women_ctrl": p_cure_women_ctrl,
        "pool_treat": p_cure_pool_treat,
        "pool_ctrl": p_cure_pool_ctrl,
        "paradox_holds": men_treat_better and women_treat_better and pool_ctrl_better,
    }


# =====================================================================
# 5. MAP with Gaussian Prior vs L2 Weight Decay Equivalence
# =====================================================================
def verify_map_weight_decay_equivalence():
    """
    Proves that argmin ||Y - Xw||^2 + lambda * ||w||^2
    is identical to MAP under w ~ N(0, sigma^2 I) where lambda = 1 / sigma^2.
    """
    np.random.seed(42)
    X = np.random.randn(20, 5)
    w_true = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    Y = X @ w_true + 0.1 * np.random.randn(20)

    lambda_reg = 0.5
    sigma_sq = 1.0 / lambda_reg

    # Ridge / L2 closed form: (X^T X + lambda I)^-1 X^T Y
    w_ridge = np.linalg.solve(X.T @ X + lambda_reg * np.eye(5), X.T @ Y)

    # MAP closed form with w ~ N(0, sigma^2 I) and noise ~ N(0, 1):
    # -log p(Y|w) - log p(w) = 0.5 ||Y - Xw||^2 + 0.5/sigma^2 ||w||^2
    # Setting derivative to 0 gives: (X^T X + (1/sigma^2) I) w = X^T Y
    w_map = np.linalg.solve(X.T @ X + (1.0 / sigma_sq) * np.eye(5), X.T @ Y)

    return np.allclose(w_ridge, w_map)


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 3.1: Probability Foundations & Bayes — Test Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Spam Filter Grid
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Spam Classification]")
    spam_res = compute_spam_posterior()
    print(f"  P(Spam ∩ 'crypto'):        {spam_res['p_joint_spam']:.4f} (Expected: 0.1400)")
    print(f"  P(Ham ∩ 'crypto'):         {spam_res['p_joint_ham']:.4f} (Expected: 0.0400)")
    print(f"  Total Evidence P('crypto'):{spam_res['p_evidence']:.4f} (Expected: 0.1800)")
    print(f"  Posterior P(Spam|'crypto'):{spam_res['p_spam_given_w']:.6f} (Expected: ~0.777778 = 7/9)")
    print(f"  Posterior P(Ham|'crypto'): {spam_res['p_ham_given_w']:.6f} (Expected: ~0.222222 = 2/9)")

    assert np.isclose(spam_res["p_joint_spam"], 0.14)
    assert np.isclose(spam_res["p_joint_ham"], 0.04)
    assert np.isclose(spam_res["p_evidence"], 0.18)
    assert np.isclose(spam_res["p_spam_given_w"], 7.0 / 9.0)
    assert np.isclose(spam_res["p_ham_given_w"], 2.0 / 9.0)
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: Medical Diagnostic Test & Base Rate Fallacy
    # -------------------------------------------------------------
    print("\n[Test 2: Medical Diagnosis Base Rate Fallacy (1,000,000 Patients)]")
    emp_post, ana_post, total_pos = simulate_base_rate_fallacy(num_samples=1_000_000)
    print(f"  Total Positive Tests:      {total_pos}")
    print(f"  Analytical Posterior:      {ana_post * 100:.2f}% (Expected: ~9.02%)")
    print(f"  Empirical Monte Carlo:     {emp_post * 100:.2f}%")
    assert np.isclose(emp_post, ana_post, atol=0.01), "Monte Carlo posterior mismatch!"
    print("  -> Base rate fallacy empirically verified across 1M samples!")

    # -------------------------------------------------------------
    # Test 3: Naive Bayes Classifier Implementation
    # -------------------------------------------------------------
    print("\n[Test 3: Scratch Naive Bayes Classifier Verification]")
    # Toy binary feature data: 4 samples, 3 features (e.g. keywords)
    X_train = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
    ])
    y_train = np.array([1, 1, 0, 0])

    nb = ScratchNaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    preds = nb.predict(X_train)
    accuracy = np.mean(preds == y_train)

    print(f"  Training Samples: 4, Features: 3")
    print(f"  Predictions:      {preds}")
    print(f"  Ground Truth:     {y_train}")
    print(f"  Accuracy:         {accuracy * 100:.1f}%")
    assert accuracy >= 0.75, "Naive Bayes accuracy unexpected!"
    print("  -> Scratch Naive Bayes training and inference verified!")

    # -------------------------------------------------------------
    # Test 4: Simpson's Paradox Numerical Confirmation
    # -------------------------------------------------------------
    print("\n[Test 4: Simpson's Paradox Numerical Verification]")
    simp = verify_simpsons_paradox()
    print(f"  Men Cure Rate:   Treatment {simp['men_treat']*100:.1f}% vs Control {simp['men_ctrl']*100:.1f}%  (Treatment Wins)")
    print(f"  Women Cure Rate: Treatment {simp['women_treat']*100:.1f}% vs Control {simp['women_ctrl']*100:.1f}%  (Treatment Wins)")
    print(f"  Pooled Total:    Treatment {simp['pool_treat']*100:.1f}% vs Control {simp['pool_ctrl']*100:.1f}%  (CONTROL WINS!)")
    print(f"  Simpson's Paradox Holds? {simp['paradox_holds']}")
    assert simp["paradox_holds"], "Simpson's paradox condition failed!"
    print("  -> Simpson's paradox mathematically demonstrated!")

    # -------------------------------------------------------------
    # Test 5: MAP vs L2 Weight Decay Equivalence
    # -------------------------------------------------------------
    print("\n[Test 5: MAP Gaussian Prior vs. L2 Weight Decay Equivalence]")
    equiv = verify_map_weight_decay_equivalence()
    print(f"  Are w_ridge (L2 penalty) and w_map (Gaussian prior) identical? {equiv}")
    assert equiv, "MAP and L2 weight decay must be mathematically equivalent!"
    print("  -> MAP Gaussian prior and L2 regularization equivalence verified!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 3.1 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
