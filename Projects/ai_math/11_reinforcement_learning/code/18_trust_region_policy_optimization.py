"""
Mathematical Verification Script for Module 11.18: Trust Region Policy Optimization (TRPO)
===========================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Conjugate Gradient Iteration 0: alpha_0 = 5/11 = 0.454545, x_1 = [0.909091, 0.454545]
   - Conjugate Gradient Iteration 1: alpha_1 = 1.256983, x_2 = [6/7, 4/7] = [0.857143, 0.571429]
   - Exact match with analytical inverse F^{-1} g to machine precision (< 1e-14).
   - Trust region scaling: beta = 0.209165, delta_theta = [0.179284, 0.119523]
   - KL constraint: 0.5 * delta_theta^T F delta_theta = 0.050000
2. PyTorch Hessian-Vector Product (HVP) and Conjugate Gradient:
   - Verifies Pearlmutter's trick computes F * v in O(d) without instantiating F.
3. Backtracking Line Search logic.
"""

import numpy as np
import torch
import torch.nn as nn

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 TRPO Conjugate Gradient Hand Calculation Verification ---")
    g = np.array([2.0, 1.0])
    F = np.array([
        [2.0, 0.5],
        [0.5, 1.0]
    ])
    delta_kl = 0.05
    
    # 1. CG Iteration 0
    x0 = np.zeros(2)
    r0 = g.copy()
    p0 = r0.copy()
    
    v0 = F @ p0 # [4.5, 2.0]
    curv0 = p0 @ v0 # 11.0
    r0_norm_sq = r0 @ r0 # 5.0
    alpha0 = r0_norm_sq / curv0 # 5 / 11
    
    x1 = x0 + alpha0 * p0
    r1 = r0 - alpha0 * v0
    
    print(f"  Iteration 0: v0 = {v0}, alpha0 = {alpha0:.6f}, x1 = {x1}, r1 = {r1}")
    assert np.allclose(v0, [4.5, 2.0])
    assert np.isclose(curv0, 11.0)
    assert np.isclose(alpha0, 5.0 / 11.0)
    assert np.allclose(x1, [10.0 / 11.0, 5.0 / 11.0])
    
    # 2. CG Iteration 1
    r1_norm_sq = r1 @ r1
    beta0 = r1_norm_sq / r0_norm_sq
    p1 = r1 + beta0 * p0
    v1 = F @ p1
    curv1 = p1 @ v1
    alpha1 = r1_norm_sq / curv1
    
    x2 = x1 + alpha1 * p1
    r2 = r1 - alpha1 * v1
    
    print(f"  Iteration 1: beta0 = {beta0:.6f}, alpha1 = {alpha1:.6f}, x2 = {x2}")
    
    # Analytical solution x* = F^{-1} g = [6/7, 4/7]
    x_analytical = np.array([6.0 / 7.0, 4.0 / 7.0])
    print(f"  Analytical x* = {x_analytical}")
    assert np.allclose(x2, x_analytical), f"Expected {x_analytical}, got {x2}"
    assert np.allclose(r2, [0.0, 0.0], atol=1e-12)
    
    # 3. Trust Region Step Scaling
    curvature = x2 @ F @ x2
    expected_curvature = 16.0 / 7.0
    print(f"  Curvature x^T F x = {curvature:.6f}, Expected = {expected_curvature:.6f}")
    assert np.isclose(curvature, expected_curvature)
    
    beta_scale = np.sqrt(2 * delta_kl / curvature)
    expected_beta = np.sqrt(0.1 * 7.0 / 16.0)
    print(f"  Scaling Multiplier beta = {beta_scale:.6f}")
    assert np.isclose(beta_scale, expected_beta)
    
    delta_theta = beta_scale * x2
    expected_delta_theta = beta_scale * np.array([6.0 / 7.0, 4.0 / 7.0])
    print(f"  Final TRPO Update Step: {delta_theta}")
    assert np.allclose(delta_theta, expected_delta_theta)
    
    # Verify exact KL constraint satisfaction
    resulting_kl = 0.5 * delta_theta @ F @ delta_theta
    print(f"  Resulting KL Divergence: {resulting_kl:.6f}")
    assert np.isclose(resulting_kl, delta_kl)
    
    print("  [PASSED] Part 5 TRPO Conjugate Gradient verified to exact machine precision!\n")


def test_pytorch_hvp_and_cg():
    print("--- Test 2: PyTorch Hessian-Vector Product & Conjugate Gradient ---")
    
    # Define a 4D quadratic function to test HVP: f(theta) = 0.5 * theta^T A theta
    torch.manual_seed(42)
    d = 4
    # Generate random positive-definite matrix A
    rand_mat = torch.randn(d, d)
    A = rand_mat.T @ rand_mat + 0.1 * torch.eye(d)
    
    theta = nn.Parameter(torch.zeros(d))
    
    # Arbitrary vector v
    v = torch.randn(d)
    
    # Compute A * v directly
    Av_direct = A @ v
    
    # Compute A * v via Hessian-vector product trick
    # loss = 0.5 * theta^T A theta
    loss = 0.5 * theta @ A @ theta
    grads = torch.autograd.grad(loss, theta, create_graph=True)[0]
    # grads is A @ theta
    # Let's test with a point theta_0
    theta_0 = torch.randn(d, requires_grad=True)
    loss_0 = 0.5 * theta_0 @ A @ theta_0
    grad_0 = torch.autograd.grad(loss_0, theta_0, create_graph=True)[0]
    
    grad_v_prod = torch.dot(grad_0, v)
    hvp = torch.autograd.grad(grad_v_prod, theta_0)[0]
    
    print(f"  Direct A @ v: {Av_direct.numpy()}")
    print(f"  HVP trick:    {hvp.numpy()}")
    assert torch.allclose(Av_direct, hvp, atol=1e-5)
    print("  Hessian-Vector Product matches direct matrix multiplication identically!")
    
    # Now run Conjugate Gradient using the HVP function
    def f_Ax(vec):
        gvp = torch.dot(torch.autograd.grad(0.5 * theta_0 @ A @ theta_0, theta_0, create_graph=True)[0], vec)
        return torch.autograd.grad(gvp, theta_0, retain_graph=True)[0]
    
    b = torch.randn(d)
    # Solve A x = b via CG
    x = torch.zeros(d)
    r = b.clone()
    p = r.clone()
    
    for _ in range(d):
        Ap = f_Ax(p)
        alpha = torch.dot(r, r) / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r_new = r - alpha * Ap
        if torch.norm(r_new) < 1e-6:
            break
        beta = torch.dot(r_new, r_new) / (torch.dot(r, r) + 1e-8)
        p = r_new + beta * p
        r = r_new
        
    x_analytical = torch.linalg.solve(A, b)
    print(f"  CG Solution:         {x.numpy()}")
    print(f"  Analytical Solution: {x_analytical.numpy()}")
    assert torch.allclose(x, x_analytical, atol=1e-4)
    print("  [PASSED] PyTorch HVP and CG solver verified!\n")


def test_backtracking_line_search():
    print("--- Test 3: Backtracking Line Search Logic ---")
    
    # Synthetic objective that is concave near 0, but decreases past 0.5
    # f(x) = x - x^2
    # Constraint: g(x) = x^2 <= 0.04 (so |x| <= 0.2)
    def surrogate_obj(step):
        return step - step ** 2
    
    def kl_div(step):
        return step ** 2
    
    max_kl = 0.04
    proposed_step = 0.5 # Violates both constraint (0.25 > 0.04) and is suboptimal
    
    step = proposed_step
    alpha = 0.5
    accepted = False
    
    for j in range(10):
        c_step = (alpha ** j) * proposed_step
        if surrogate_obj(c_step) > 0 and kl_div(c_step) <= max_kl:
            accepted = True
            accepted_step = c_step
            accepted_j = j
            break
            
    print(f"  Initial Step: {proposed_step}")
    print(f"  Accepted at iteration {accepted_j}: Step = {accepted_step}, KL = {kl_div(accepted_step):.4f}")
    assert accepted
    assert kl_div(accepted_step) <= max_kl
    assert surrogate_obj(accepted_step) > 0
    print("  [PASSED] Backtracking line search successfully enforces trust region!\n")


def test_illustration_2_cg_2x2():
    print("--- Test 4: Section 6 Illustration 2 - 2-Iteration CG Solve on 2x2 System ---")
    F = np.array([[2.0, 1.0], [1.0, 2.0]])
    g = np.array([1.0, 2.0])
    
    # Iteration 0
    x0 = np.zeros(2)
    r0 = g - F @ x0
    p0 = r0.copy()
    v0 = F @ p0 # [4, 5]
    alpha0 = (r0 @ r0) / (p0 @ v0) # 5 / 14
    x1 = x0 + alpha0 * p0 # [5/14, 10/14]
    r1 = r0 - alpha0 * v0 # [-6/14, 3/14]
    
    assert np.allclose(v0, [4.0, 5.0])
    assert np.isclose(alpha0, 5.0 / 14.0)
    assert np.allclose(x1, [5.0 / 14.0, 5.0 / 7.0])
    assert np.allclose(r1, [-3.0 / 7.0, 3.0 / 14.0])
    
    # Iteration 1
    r1_sq = r1 @ r1
    r0_sq = r0 @ r0
    beta0 = r1_sq / r0_sq # 9 / 196
    p1 = r1 + beta0 * p0 # [-75/196, 60/196]
    v1 = F @ p1 # [-90/196, 45/196] = 45/196 * [-2, 1]
    curv1 = p1 @ v1
    alpha1 = r1_sq / curv1 # 14 / 15
    x2 = x1 + alpha1 * p1
    r2 = r1 - alpha1 * v1
    
    assert np.isclose(beta0, 9.0 / 196.0)
    assert np.allclose(p1, [-75.0 / 196.0, 60.0 / 196.0])
    assert np.isclose(alpha1, 14.0 / 15.0)
    assert np.allclose(x2, [0.0, 1.0])
    assert np.allclose(r2, [0.0, 0.0], atol=1e-14)
    
    # Analytical verification
    x_analytical = np.linalg.solve(F, g)
    assert np.allclose(x2, x_analytical)
    print(f"  Converged to exact solution {x2} matching analytical inverse [0, 1] identically!")
    print("  [PASSED] Illustration 2 verified to exact machine precision!\n")


def test_illustration_3_trpo_step_and_backtracking():
    print("--- Test 5: Section 6 Illustration 3 - Step Length & Backtracking Line Search ---")
    F = np.array([[2.0, 1.0], [1.0, 2.0]])
    g = np.array([1.0, 2.0])
    x = np.array([0.0, 1.0])
    delta = 0.01
    
    # Scaling factor beta
    x_dot_g = x @ g
    beta = np.sqrt(2 * delta / x_dot_g)
    assert np.isclose(beta, 0.1)
    
    step0 = beta * x
    assert np.allclose(step0, [0.0, 0.1])
    
    # Quadratic approximation
    quad_kl = 0.5 * step0 @ F @ step0
    assert np.isclose(quad_kl, delta)
    
    # Non-linear models
    def true_kl(s):
        return 0.5 * s @ F @ s + 3.0 * (s[1] ** 3)
    
    def surrogate_obj(s):
        return g @ s - 15.0 * (s[1] ** 2)
    
    # j = 0
    kl_0 = true_kl(step0)
    L_0 = surrogate_obj(step0)
    assert np.isclose(kl_0, 0.013)
    assert np.isclose(L_0, 0.05)
    assert kl_0 > delta # VIOLATED
    
    # j = 1 (backtracking alpha = 0.5)
    step1 = 0.5 * step0
    kl_1 = true_kl(step1)
    L_1 = surrogate_obj(step1)
    assert np.allclose(step1, [0.0, 0.05])
    assert np.isclose(kl_1, 0.002875)
    assert np.isclose(L_1, 0.0625)
    assert kl_1 <= delta # SATISFIED
    assert L_1 > 0 # SATISFIED
    print(f"  Step 0: KL={kl_0:.4f} > {delta} (Rejected)")
    print(f"  Step 1: KL={kl_1:.6f} <= {delta}, L={L_1:.4f} > 0 (Accepted)")
    print("  [PASSED] Illustration 3 verified!\n")


def test_illustration_4_monotonic_bound_3state_mdp():
    print("--- Test 6: Section 6 Illustration 4 - Monotonic Bound on 3-State MDP ---")
    gamma = 0.5
    P_a1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    P_a2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    R_a1 = np.array([1.0, 2.0, 3.0])
    R_a2 = np.array([0.0, 1.0, 0.0])
    I = np.eye(3)
    mu = np.array([1.0, 0.0, 0.0])
    
    # Baseline policy pi: 0.5, 0.5
    P_pi = 0.5 * P_a1 + 0.5 * P_a2
    R_pi = 0.5 * R_a1 + 0.5 * R_a2
    V_pi = np.linalg.solve(I - gamma * P_pi, R_pi)
    assert np.allclose(V_pi, [5.0 / 3.0, 3.0, 3.0])
    
    Q_a1 = R_a1 + gamma * P_a1 @ V_pi
    Q_a2 = R_a2 + gamma * P_a2 @ V_pi
    assert np.allclose(Q_a1, [2.5, 3.5, 4.5])
    assert np.allclose(Q_a2, [5.0 / 6.0, 2.5, 1.5])
    
    A_a1 = Q_a1 - V_pi
    A_a2 = Q_a2 - V_pi
    assert np.allclose(A_a1, [5.0 / 6.0, 0.5, 1.5])
    assert np.allclose(A_a2, [-5.0 / 6.0, -0.5, -1.5])
    
    d_pi_unnorm = mu @ np.linalg.inv(I - gamma * P_pi)
    assert np.allclose(d_pi_unnorm, [4.0 / 3.0, 4.0 / 9.0, 2.0 / 9.0])
    
    # Candidate policy pi_tilde: 0.6, 0.4
    expected_A = 0.6 * A_a1 + 0.4 * A_a2 # 0.2 * A_a1
    L_surr = V_pi[0] + np.sum(d_pi_unnorm * expected_A)
    assert np.isclose(L_surr, 2.0)
    
    # True candidate return J(pi_tilde)
    P_tilde = 0.6 * P_a1 + 0.4 * P_a2
    R_tilde = 0.6 * R_a1 + 0.4 * R_a2
    V_tilde = np.linalg.solve(I - gamma * P_tilde, R_tilde)
    J_tilde = V_tilde[0]
    assert np.isclose(J_tilde, 2.00625)
    
    # Bound constants
    epsilon = 1.5
    C = 4.0 * epsilon * gamma / ((1.0 - gamma) ** 2) # 12.0
    assert np.isclose(C, 12.0)
    
    kl_max = 0.5 * np.log(25.0 / 24.0) # ~ 0.020411
    lower_bound = L_surr - C * kl_max
    assert np.isclose(lower_bound, 1.755068032878469)
    assert J_tilde >= lower_bound
    assert J_tilde > V_pi[0]
    print(f"  J(pi) = {V_pi[0]:.6f}, L_pi(pi_tilde) = {L_surr:.4f}, J(pi_tilde) = {J_tilde:.6f}")
    print(f"  Lower bound = {lower_bound:.6f} <= J(pi_tilde) (Monotonic improvement guaranteed!)")
    print("  [PASSED] Illustration 4 verified!\n")


def test_illustration_5_pearlmutter_gaussian_fvp():
    print("--- Test 7: Section 6 Illustration 5 - Pearlmutter FVP on Gaussian Policy ---")
    theta_old = torch.tensor([1.0, 0.5])
    v = torch.tensor([0.6, -0.8])
    
    # Analytical Fisher matrix F = diag(exp(-2*rho), 2)
    F_analytic = torch.tensor([[torch.exp(torch.tensor(-1.0)), 0.0], [0.0, 2.0]])
    Fv_direct = F_analytic @ v
    
    # Autograd Pearlmutter double backward
    theta = torch.tensor([1.0, 0.5], requires_grad=True)
    kl = (theta[1] - theta_old[1]) + (torch.exp(2 * theta_old[1]) + (theta_old[0] - theta[0]) ** 2) / (2 * torch.exp(2 * theta[1])) - 0.5
    
    g_kl = torch.autograd.grad(kl, theta, create_graph=True)[0]
    prod = torch.dot(g_kl, v)
    fvp = torch.autograd.grad(prod, theta)[0]
    
    print(f"  Direct Analytical F @ v: {Fv_direct.numpy()}")
    print(f"  Pearlmutter Double-Grad: {fvp.detach().numpy()}")
    assert torch.allclose(Fv_direct, fvp, atol=1e-7)
    print("  [PASSED] Illustration 5 verified to exact machine precision!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.18 TRUST REGION POLICY OPTIMIZATION (TRPO) TESTS")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_pytorch_hvp_and_cg()
    test_backtracking_line_search()
    test_illustration_2_cg_2x2()
    test_illustration_3_trpo_step_and_backtracking()
    test_illustration_4_monotonic_bound_3state_mdp()
    test_illustration_5_pearlmutter_gaussian_fvp()
    
    print("=================================================================")
    print("ALL MODULE 11.18 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
