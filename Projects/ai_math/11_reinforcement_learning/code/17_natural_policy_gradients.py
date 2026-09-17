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


def test_illustration_1_gaussian_numerical():
    print("--- Test 4: Illustration 1 - 1D Gaussian Policy Numerical Walkthrough ---")
    mu, sigma = 1.5, 0.5
    F = np.array([[1.0 / (sigma**2), 0.0], [0.0, 2.0 / (sigma**2)]])
    expected_F = np.array([[4.0, 0.0], [0.0, 8.0]])
    assert np.allclose(F, expected_F)

    g = np.array([0.8, -0.4])
    epsilon = 0.02
    F_inv = np.linalg.inv(F)
    expected_F_inv = np.array([[0.25, 0.0], [0.0, 0.125]])
    assert np.allclose(F_inv, expected_F_inv)

    g_tilde = F_inv @ g
    expected_g_tilde = np.array([0.20, -0.05])
    assert np.allclose(g_tilde, expected_g_tilde)

    curv = g @ g_tilde
    assert np.isclose(curv, 0.18)

    beta = np.sqrt(2 * epsilon / curv)
    expected_beta = np.sqrt(0.04 / 0.18)
    assert np.isclose(beta, expected_beta)

    delta_theta = beta * g_tilde
    expected_delta_theta = expected_beta * expected_g_tilde
    assert np.allclose(delta_theta, expected_delta_theta)

    kl = 0.5 * delta_theta @ F @ delta_theta
    assert np.isclose(kl, epsilon)
    print(f"  Fisher Matrix F:\n{F}")
    print(f"  Natural gradient g_tilde: {g_tilde}")
    print(f"  Step size beta: {beta:.6f}")
    print(f"  Update delta_theta: {delta_theta}")
    print(f"  Verified KL divergence: {kl:.6f}")
    print("  [PASSED] Illustration 1 numerical walkthrough verified!\n")


def test_illustration_2_softmax_fim():
    print("--- Test 5: Illustration 2 - 2-Action Softmax Analytical & Numerical FIM ---")
    theta = np.array([np.log(3.0), 0.0])
    exp_th = np.exp(theta)
    pi = exp_th / np.sum(exp_th)
    expected_pi = np.array([0.75, 0.25])
    assert np.allclose(pi, expected_pi)

    # Analytical formula: F = diag(pi) - pi pi^T
    F_formula = np.diag(pi) - np.outer(pi, pi)
    expected_F = np.array([[0.1875, -0.1875], [-0.1875, 0.1875]])
    assert np.allclose(F_formula, expected_F)

    # Outer product of score functions:
    s0 = np.array([1 - pi[0], -pi[1]])  # [0.25, -0.25]
    s1 = np.array([-pi[0], 1 - pi[1]])  # [-0.75, 0.75]
    F_scores = pi[0] * np.outer(s0, s0) + pi[1] * np.outer(s1, s1)
    assert np.allclose(F_scores, expected_F)
    assert np.allclose(F_formula, F_scores)
    print(f"  Policy probabilities: {pi}")
    print(f"  Analytical FIM (diag(pi) - pi pi^T):\n{F_formula}")
    print("  [PASSED] Illustration 2 softmax FIM verified!\n")


def test_illustration_3_ill_conditioned_plateau():
    print("--- Test 6: Illustration 3 - Ill-Conditioned Plateau Natural vs Vanilla ---")
    F = np.array([[25.0, 0.0], [0.0, 0.04]])
    g = np.array([1.0, 0.2])
    epsilon = 0.01

    # Vanilla step constrained by same KL = 0.01:
    curv_vanilla = g @ F @ g
    assert np.isclose(curv_vanilla, 25.0016)
    alpha_vanilla = np.sqrt(2 * epsilon / curv_vanilla)
    delta_vanilla = alpha_vanilla * g
    J_vanilla = g @ delta_vanilla

    # Natural gradient step:
    F_inv = np.linalg.inv(F)
    g_tilde = F_inv @ g
    assert np.allclose(g_tilde, [0.04, 5.0])
    curv_nat = g @ g_tilde
    assert np.isclose(curv_nat, 1.04)
    beta_nat = np.sqrt(2 * epsilon / curv_nat)
    delta_nat = beta_nat * g_tilde
    J_nat = g @ delta_nat

    ratio = J_nat / J_vanilla
    print(f"  Vanilla Delta theta: {delta_vanilla}, J improvement: {J_vanilla:.6f}")
    print(f"  Natural Delta theta: {delta_nat}, J improvement: {J_nat:.6f}")
    print(f"  Ascent speedup ratio: {ratio:.4f}x")
    assert np.isclose(ratio, 4.90306027, atol=1e-5)
    print("  [PASSED] Illustration 3 ill-conditioned plateau verified!\n")


def test_illustration_4_empirical_fisher():
    print("--- Test 7: Illustration 4 - Empirical Fisher Estimation & Damped Inversion ---")
    g_batch = [
        np.array([1.0, 0.5]),
        np.array([-0.5, 1.0]),
        np.array([0.8, -0.4]),
        np.array([-0.6, -0.8]),
    ]
    G_sum = sum(np.outer(gi, gi) for gi in g_batch)
    expected_G_sum = np.array([[2.25, 0.16], [0.16, 2.05]])
    assert np.allclose(G_sum, expected_G_sum)

    F_emp = G_sum / 4.0
    expected_F_emp = np.array([[0.5625, 0.0400], [0.0400, 0.5125]])
    assert np.allclose(F_emp, expected_F_emp)

    lam = 0.001
    F_damp = F_emp + lam * np.eye(2)
    expected_F_damp = np.array([[0.5635, 0.0400], [0.0400, 0.5135]])
    assert np.allclose(F_damp, expected_F_damp)

    det = F_damp[0, 0] * F_damp[1, 1] - F_damp[0, 1] * F_damp[1, 0]
    assert np.isclose(det, 0.28775725)

    F_inv = (1.0 / det) * np.array([[F_damp[1, 1], -F_damp[0, 1]], [-F_damp[1, 0], F_damp[0, 0]]])
    assert np.allclose(F_inv, np.linalg.inv(F_damp))

    g_sample = np.array([0.5, -0.2])
    nat_step = F_inv @ g_sample
    expected_nat_step = np.array([0.92004632, -0.46115259])
    assert np.allclose(nat_step, expected_nat_step)
    print(f"  Empirical Fisher F_emp:\n{F_emp}")
    print(f"  Damped Fisher F_damp:\n{F_damp}")
    print(f"  Determinant: {det:.8f}")
    print(f"  Inverse metric F_inv:\n{F_inv}")
    print(f"  Natural direction: {nat_step}")
    print("  [PASSED] Illustration 4 empirical Fisher matrix verified!\n")


def test_illustration_5_continuous_gaussian():
    print("--- Test 8: Illustration 5 - Continuous Gaussian Natural Step & Coordinate Invariance ---")
    Sigma = np.array([[4.0, 1.0], [1.0, 2.0]])
    g = np.array([1.5, -0.5])
    epsilon = 0.05

    g_tilde = Sigma @ g
    assert np.allclose(g_tilde, [5.5, 0.5])
    curv = g @ Sigma @ g
    assert np.isclose(curv, 8.0)
    beta = np.sqrt(2 * epsilon / curv)
    delta_mu = beta * g_tilde
    expected_delta_mu = np.sqrt(0.10 / 8.0) * np.array([5.5, 0.5])
    assert np.allclose(delta_mu, expected_delta_mu)

    # Coordinate scaling C = diag(10, 1)
    C = np.diag([10.0, 1.0])
    C_inv = np.linalg.inv(C)
    Sigma_scaled = C @ Sigma @ C.T
    g_scaled = C_inv @ g
    g_tilde_scaled = Sigma_scaled @ g_scaled
    curv_scaled = g_scaled @ Sigma_scaled @ g_scaled
    assert np.isclose(curv_scaled, 8.0)
    beta_scaled = np.sqrt(2 * epsilon / curv_scaled)
    delta_mu_scaled = beta_scaled * g_tilde_scaled
    delta_mu_mapped = C_inv @ delta_mu_scaled
    assert np.allclose(delta_mu, delta_mu_mapped, atol=1e-15)

    print(f"  Original update Delta mu: {delta_mu}")
    print(f"  Scaled update Delta mu_tilde: {delta_mu_scaled}")
    print(f"  Mapped back C^{{-1}} Delta mu_tilde: {delta_mu_mapped}")
    print("  [PASSED] Illustration 5 continuous Gaussian natural step verified!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.17 NATURAL POLICY GRADIENTS TEST SUITE")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_invariance_to_reparameterization()
    test_gaussian_fisher_information()
    test_illustration_1_gaussian_numerical()
    test_illustration_2_softmax_fim()
    test_illustration_3_ill_conditioned_plateau()
    test_illustration_4_empirical_fisher()
    test_illustration_5_continuous_gaussian()
    
    print("=================================================================")
    print("ALL MODULE 11.17 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
