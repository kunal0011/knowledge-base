"""
Mathematical Verification Script for Module 11.17: Natural Policy Gradients
============================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Fisher Information Matrix: F = [[0.16, -0.16], [-0.16, 0.16]]
   - Damped Fisher Matrix: F_reg = [[0.20, -0.16], [-0.16, 0.20]]
   - Matrix inverse F_reg^{-1}: [[13.8889, 11.1111], [11.1111, 13.8889]]
   - Natural gradient direction: g_tilde = [+2.7778, -2.7778]
   - Scaling factor: beta = 0.0600
   - Final update: delta_theta = [+0.1667, -0.1667]
   - KL constraint satisfaction: 0.5 * delta_theta^T F_reg delta_theta = 0.0100
2. Invariance to Re-parameterization (Kakade's Theorem):
   - Proves natural gradient steps produce identical probability shifts under linear coordinate changes.
3. Analytical Gaussian Fisher Information Matrix:
   - Verifies Monte Carlo empirical FIM converges to analytical [[1/sigma^2, 0], [0, 2/sigma^2]].
"""

import numpy as np

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 Natural Policy Gradient Hand Calculation Verification ---")
    pi0 = 0.80
    pi1 = 0.20
    g = np.array([1.0, -1.0])
    epsilon = 0.01
    delta_damp = 0.04
    
    # 1. Scores
    s0 = np.array([1.0 - pi0, -pi1]) # [0.2, -0.2]
    s1 = np.array([-pi0, 1.0 - pi1]) # [-0.8, 0.8]
    assert np.allclose(s0, [0.2, -0.2])
    assert np.allclose(s1, [-0.8, 0.8])
    
    # 2. Fisher Matrix F
    F = pi0 * np.outer(s0, s0) + pi1 * np.outer(s1, s1)
    expected_F = np.array([
        [0.16, -0.16],
        [-0.16, 0.16]
    ])
    print(f"  Fisher Matrix F:\n{F}")
    assert np.allclose(F, expected_F)
    
    # 3. Damped F_reg
    F_reg = F + delta_damp * np.eye(2)
    expected_F_reg = np.array([
        [0.20, -0.16],
        [-0.16, 0.20]
    ])
    print(f"  Damped Fisher Matrix F_reg:\n{F_reg}")
    assert np.allclose(F_reg, expected_F_reg)
    
    # 4. Matrix Inverse
    det = np.linalg.det(F_reg)
    assert np.isclose(det, 0.0144)
    F_inv = np.linalg.inv(F_reg)
    expected_F_inv = np.array([
        [13.88888889, 11.11111111],
        [11.11111111, 13.88888889]
    ])
    print(f"  Inverse Metric F_reg^{{-1}}:\n{F_inv}")
    assert np.allclose(F_inv, expected_F_inv)
    
    # 5. Natural gradient direction g_tilde
    g_tilde = F_inv @ g
    expected_g_tilde = np.array([2.77777778, -2.77777778])
    print(f"  Natural Gradient Direction g_tilde: {g_tilde}")
    assert np.allclose(g_tilde, expected_g_tilde)
    
    # 6. Quadratic form & Scaling
    curvature = g @ g_tilde
    expected_curvature = 5.55555556
    print(f"  Curvature g^T F^{{-1}} g = {curvature:.6f}")
    assert np.isclose(curvature, expected_curvature)
    
    beta = np.sqrt(2 * epsilon / curvature)
    print(f"  Step Size beta = {beta:.6f}")
    assert np.isclose(beta, 0.0600000)
    
    delta_theta = beta * g_tilde
    expected_delta_theta = np.array([1.0 / 6.0, -1.0 / 6.0]) # [0.166667, -0.166667]
    print(f"  Final Natural Policy Gradient Step: {delta_theta}")
    assert np.allclose(delta_theta, expected_delta_theta)
    
    # 7. Check KL constraint
    actual_kl = 0.5 * delta_theta @ F_reg @ delta_theta
    print(f"  Resulting Local KL divergence: {actual_kl:.6f}")
    assert np.isclose(actual_kl, epsilon)
    
    print("  [PASSED] Part 5 hand calculations verified to exact machine precision!\n")


def test_invariance_to_reparameterization():
    print("--- Test 2: Invariance to Re-parameterization (Information Geometry) ---")
    # Softmax policy with parameters theta
    # Consider invertible linear re-parameterization: psi = M @ theta
    # Let M be an arbitrary non-singular matrix
    np.random.seed(42)
    theta = np.array([0.5, -0.3])
    
    # Random invertible transformation
    M = np.array([
        [2.0, 1.0],
        [1.0, 3.0]
    ])
    M_inv = np.linalg.inv(M)
    psi = M @ theta
    
    # Current probabilities
    def get_probs(th):
        exp_th = np.exp(th - np.max(th))
        return exp_th / np.sum(exp_th)
    
    pi_theta = get_probs(theta)
    pi_psi = get_probs(M_inv @ psi)
    assert np.allclose(pi_theta, pi_psi)
    
    # Synthetic gradient in theta space
    g_theta = np.array([0.4, -0.4])
    
    # Score vectors in theta-space
    s0 = np.array([1 - pi_theta[0], -pi_theta[1]])
    s1 = np.array([-pi_theta[0], 1 - pi_theta[1]])
    F_theta = pi_theta[0] * np.outer(s0, s0) + pi_theta[1] * np.outer(s1, s1) + 0.01 * np.eye(2)
    
    # Natural gradient update in theta space
    nat_step_theta = np.linalg.inv(F_theta) @ g_theta
    
    # Gradient in psi space via chain rule: g_psi = (d theta / d psi)^T g_theta = M^{-T} g_theta
    g_psi = M_inv.T @ g_theta
    
    # Fisher in psi space: F_psi = M^{-T} F_theta M^{-1}
    F_psi = M_inv.T @ F_theta @ M_inv
    
    # Natural gradient update in psi space
    nat_step_psi = np.linalg.inv(F_psi) @ g_psi
    
    # Map nat_step_psi back to theta coordinates: M^{-1} @ nat_step_psi
    mapped_step_theta = M_inv @ nat_step_psi
    
    print(f"  Natural step in theta space: {nat_step_theta}")
    print(f"  Mapped natural step from psi space: {mapped_step_theta}")
    assert np.allclose(nat_step_theta, mapped_step_theta), "Natural gradient must be invariant to coordinates!"
    
    print("  [PASSED] Invariance to coordinate re-parameterization confirmed!\n")


def test_gaussian_fisher_information():
    print("--- Test 3: Analytical 1D Gaussian Fisher Information Matrix ---")
    mu = 2.0
    sigma = 1.5
    
    # Analytical Fisher Information
    F_analytical = np.array([
        [1.0 / (sigma ** 2), 0.0],
        [0.0, 2.0 / (sigma ** 2)]
    ])
    print(f"  Analytical FIM (mu={mu}, sigma={sigma}):\n{F_analytical}")
    
    # Monte Carlo empirical FIM estimation
    np.random.seed(42)
    N_samples = 100000
    samples = np.random.normal(mu, sigma, size=N_samples)
    
    grad_mu = (samples - mu) / (sigma ** 2)
    grad_sigma = -1.0 / sigma + ((samples - mu) ** 2) / (sigma ** 3)
    
    F_empirical_00 = np.mean(grad_mu * grad_mu)
    F_empirical_01 = np.mean(grad_mu * grad_sigma)
    F_empirical_11 = np.mean(grad_sigma * grad_sigma)
    
    F_empirical = np.array([
        [F_empirical_00, F_empirical_01],
        [F_empirical_01, F_empirical_11]
    ])
    print(f"  Empirical FIM (N={N_samples}):\n{F_empirical}")
    
    assert np.allclose(F_empirical, F_analytical, atol=0.03)
    print("  [PASSED] Gaussian Fisher Information matrix matches analytical derivation!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.17 NATURAL POLICY GRADIENTS TEST SUITE")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_invariance_to_reparameterization()
    test_gaussian_fisher_information()
    
    print("=================================================================")
    print("ALL MODULE 11.17 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
