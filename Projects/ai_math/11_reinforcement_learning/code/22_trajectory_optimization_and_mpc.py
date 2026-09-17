"""
Trajectory Optimization & Model Predictive Control (MPC) Verification Suite
Module 11: Reinforcement Learning - Chapter 22

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Backward Riccati difference recursion: K1 = -0.8000, K0 = -14/19 ≈ -0.736842
   - Forward optimal state-control rollout: s1 = 50/19, s2 = 10/19
   - Total cumulative cost J* = 136.842105 == V0(s0) matching to < 1e-12.
2. Multi-dimensional Discrete-Time LQR Solver for general matrices (A, B, Q, R, Qf).
3. Cross-Entropy Method (CEM) sampling-based trajectory optimizer.
4. Closed-Loop Receding Horizon MPC vs. Open-Loop under stochastic disturbance (wind gust).
"""

import numpy as np


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' LQR RICCATI & ROLLOUT")
    print("=" * 70)

    # 1D System Parameters: s_{t+1} = a*s_t + b*u_t
    a = 1.0
    b = 1.0
    q = 2.0   # running state penalty
    r = 1.0   # running control penalty
    pH = 4.0  # terminal state penalty
    s0 = 10.0
    H = 2

    # --- Backward Riccati Pass ---
    # Terminal step t = 2:
    P2 = pH
    print(f"Terminal P2 = {P2:.4f} (Expected: 4.0000)")
    assert np.isclose(P2, 4.0, atol=1e-12)

    # Step t = 1:
    K1 = -(b * P2 * a) / (r + (b**2) * P2)
    P1 = q + (a**2) * P2 + a * P2 * b * K1
    print(f"Step t=1: K1 = {K1:.6f} (Expected: -0.800000), P1 = {P1:.6f} (Expected: 2.800000)")
    assert np.isclose(K1, -0.8, atol=1e-12)
    assert np.isclose(P1, 2.8, atol=1e-12)

    # Step t = 0:
    K0 = -(b * P1 * a) / (r + (b**2) * P1)
    P0 = q + (a**2) * P1 + a * P1 * b * K0
    expected_K0 = -14.0 / 19.0
    expected_P0 = 52.0 / 19.0
    print(f"Step t=0: K0 = {K0:.6f} (Expected: {expected_K0:.6f}), P0 = {P0:.6f} (Expected: {expected_P0:.6f})")
    assert np.isclose(K0, expected_K0, atol=1e-12)
    assert np.isclose(P0, expected_P0, atol=1e-12)

    # --- Forward Rollout ---
    # Step t = 0:
    u0 = K0 * s0
    ell_0 = 0.5 * q * (s0**2) + 0.5 * r * (u0**2)
    s1 = a * s0 + b * u0
    expected_u0 = -140.0 / 19.0
    expected_s1 = 50.0 / 19.0
    assert np.isclose(u0, expected_u0, atol=1e-12)
    assert np.isclose(s1, expected_s1, atol=1e-12)

    # Step t = 1:
    u1 = K1 * s1
    ell_1 = 0.5 * q * (s1**2) + 0.5 * r * (u1**2)
    s2 = a * s1 + b * u1
    expected_u1 = -40.0 / 19.0
    expected_s2 = 10.0 / 19.0
    assert np.isclose(u1, expected_u1, atol=1e-12)
    assert np.isclose(s2, expected_s2, atol=1e-12)

    # Step t = 2 (Terminal):
    ell_f = 0.5 * pH * (s2**2)
    expected_ell_f = 200.0 / 361.0
    assert np.isclose(ell_f, expected_ell_f, atol=1e-12)

    # Total Rollout Cost vs Bellman Riccati Cost:
    J_rollout = ell_0 + ell_1 + ell_f
    V0_riccati = 0.5 * P0 * (s0**2)
    expected_J = 2600.0 / 19.0

    print(f"Total Rollout Cost J_rollout = {J_rollout:.8f}")
    print(f"Riccati Cost V0(s0)          = {V0_riccati:.8f}")
    print(f"Analytical Expected          = {expected_J:.8f}")
    assert np.isclose(J_rollout, expected_J, atol=1e-12)
    assert np.isclose(V0_riccati, expected_J, atol=1e-12)
    assert np.isclose(J_rollout, V0_riccati, atol=1e-12)

    print(">> SUCCESS: Riccati backward recursion and forward rollout match analytical solution to < 1e-12!\n")


# =====================================================================
# 2. General Multi-Dimensional LQR Solver
# =====================================================================

def solve_lqr(A, B, Q, R, Qf, H):
    """
    Solves discrete-time finite horizon LQR via backward Riccati recursion.
    Returns:
        K_seq: list of feedback gain matrices [K_0, ..., K_{H-1}]
        P_seq: list of cost-to-go matrices [P_0, ..., P_H]
    """
    P_seq = [None] * (H + 1)
    K_seq = [None] * H

    P_seq[H] = np.copy(Qf)

    for t in reversed(range(H)):
        Pt1 = P_seq[t + 1]
        # K_t = -(R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A
        inv_term = np.linalg.inv(R + B.T @ Pt1 @ B)
        K_t = -inv_term @ (B.T @ Pt1 @ A)
        # P_t = Q + A^T P_{t+1} (A + B K_t)
        P_t = Q + A.T @ Pt1 @ (A + B @ K_t)

        K_seq[t] = K_t
        P_seq[t] = P_t

    return K_seq, P_seq


def verify_multidimensional_lqr():
    print("=" * 70)
    print("2. VERIFYING MULTI-DIMENSIONAL LQR SOLVER ON DOUBLE INTEGRATOR")
    print("=" * 70)

    # Double Integrator: state = [position, velocity]
    dt = 0.1
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.5 * dt**2],
                  [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    Qf = np.diag([20.0, 5.0])
    H = 20

    K_seq, P_seq = solve_lqr(A, B, Q, R, Qf, H)

    # Initial state: displaced position = 5.0, zero velocity
    s = np.array([[5.0], [0.0]])
    cost_accum = 0.0

    for t in range(H):
        u = K_seq[t] @ s
        step_cost = 0.5 * (s.T @ Q @ s + u.T @ R @ u)[0, 0]
        cost_accum += step_cost
        s = A @ s + B @ u

    terminal_cost = 0.5 * (s.T @ Qf @ s)[0, 0]
    total_cost = cost_accum + terminal_cost

    initial_s = np.array([[5.0], [0.0]])
    v0_cost = 0.5 * (initial_s.T @ P_seq[0] @ initial_s)[0, 0]

    print(f"Double Integrator: Final Position = {s[0,0]:.4f}, Final Velocity = {s[1,0]:.4f}")
    print(f"Total Rollout Cost: {total_cost:.6f} | Riccati V0: {v0_cost:.6f}")
    assert np.isclose(total_cost, v0_cost, atol=1e-10)
    assert np.abs(s[0, 0]) < 0.2, "Double integrator should stabilize near 0"
    print(">> SUCCESS: Multi-dimensional LQR successfully tested and verified!\n")


# =====================================================================
# 3. Cross-Entropy Method (CEM) Trajectory Optimizer
# =====================================================================

def cem_trajectory_optimization(s0, A, B, Q, R, Qf, H, n_samples=300, n_elites=30, n_iters=15):
    """
    Optimizes action trajectory sequence U in R^{H x d_a} using CEM.
    """
    np.random.seed(42)
    d_a = B.shape[1]

    # Prior: mean 0, std 5.0 (covering range [-10, 10])
    mean = np.zeros((H, d_a))
    std = 5.0 * np.ones((H, d_a))

    for iteration in range(n_iters):
        # Sample candidate action sequences: shape (n_samples, H, d_a)
        samples = np.random.normal(loc=mean, scale=std, size=(n_samples, H, d_a))
        costs = np.zeros(n_samples)

        for i in range(n_samples):
            s = np.copy(s0)
            cost_i = 0.0
            for t in range(H):
                u = samples[i, t:t+1].T
                cost_i += 0.5 * (s.T @ Q @ s + u.T @ R @ u)[0, 0]
                s = A @ s + B @ u
            cost_i += 0.5 * (s.T @ Qf @ s)[0, 0]
            costs[i] = cost_i

        # Elite selection
        elite_indices = np.argsort(costs)[:n_elites]
        elite_samples = samples[elite_indices]

        # Refit Gaussian
        new_mean = np.mean(elite_samples, axis=0)
        new_std = np.std(elite_samples, axis=0) + 1e-4  # ensure non-zero

        # Polyak smoothing
        mean = 0.7 * new_mean + 0.3 * mean
        std = 0.7 * new_std + 0.3 * std

    return mean


def verify_cem_optimizer():
    print("=" * 70)
    print("3. VERIFYING SAMPLING-BASED CEM TRAJECTORY OPTIMIZATION")
    print("=" * 70)

    # Compare CEM against exact analytical LQR on 1D system
    a = np.array([[1.0]])
    b = np.array([[1.0]])
    q = np.array([[2.0]])
    r = np.array([[1.0]])
    pH = np.array([[4.0]])
    s0 = np.array([[10.0]])
    H = 2

    # Exact LQR
    K_seq, _ = solve_lqr(a, b, q, r, pH, H)
    lqr_u0 = (K_seq[0] @ s0)[0, 0]
    s1 = a @ s0 + b @ np.array([[lqr_u0]])
    lqr_u1 = (K_seq[1] @ s1)[0, 0]

    # CEM Trajectory Optimization
    cem_plan = cem_trajectory_optimization(s0, a, b, q, r, pH, H, n_samples=500, n_elites=50, n_iters=20)
    cem_u0 = cem_plan[0, 0]
    cem_u1 = cem_plan[1, 0]

    print(f"Analytical Optimal Actions: u0 = {lqr_u0:.4f}, u1 = {lqr_u1:.4f}")
    print(f"CEM Optimized Actions:      u0 = {cem_u0:.4f}, u1 = {cem_u1:.4f}")

    # CEM should approximate analytical actions closely
    assert np.isclose(cem_u0, lqr_u0, atol=0.2), f"CEM u0 {cem_u0} should be close to {lqr_u0}"
    assert np.isclose(cem_u1, lqr_u1, atol=0.2), f"CEM u1 {cem_u1} should be close to {lqr_u1}"
    print(">> SUCCESS: Sampling-based CEM accurately converged near global optimal trajectory!\n")


# =====================================================================
# 4. Closed-Loop MPC vs. Open-Loop Under Disturbance (Wind Gust)
# =====================================================================

def verify_mpc_disturbance_rejection():
    print("=" * 70)
    print("4. VERIFYING CLOSED-LOOP MPC VS OPEN-LOOP DISTURBANCE REJECTION")
    print("=" * 70)

    # 1D setup from Section 6 Illustration 1
    a, b = 1.0, 1.0
    q, r, pH = 2.0, 1.0, 4.0
    s0 = 10.0
    H = 2
    K1 = -0.8000

    # Planned nominal trajectory:
    u0_nom = -140.0 / 19.0  # ≈ -7.3684
    s1_nom = 50.0 / 19.0   # ≈ 2.6316
    u1_nom = -40.0 / 19.0   # ≈ -2.1053

    # Wind gust at t = 1: delta = +3.0
    delta = 3.0
    s1_true = s1_nom + delta  # 5.6316

    # Open-Loop Controller: ignores true s1, blindly applies u1_nom
    s2_open = a * s1_true + b * u1_nom
    cost_f_open = 0.5 * pH * (s2_open**2)

    # Closed-Loop MPC Controller: senses s1_true and computes feedback u1 = K1 * s1_true
    u1_mpc = K1 * s1_true
    s2_mpc = a * s1_true + b * u1_mpc
    cost_f_mpc = 0.5 * pH * (s2_mpc**2)

    print(f"Wind Gust Delta = +{delta:.1f} injected at t=1 (s1_true = {s1_true:.4f})")
    print(f"Open-Loop:  u1 = {u1_nom:.4f} | s2_terminal = {s2_open:.4f} | Terminal Cost = {cost_f_open:.4f}")
    print(f"Closed-Loop MPC: u1 = {u1_mpc:.4f} | s2_terminal = {s2_mpc:.4f} | Terminal Cost = {cost_f_mpc:.4f}")

    # MPC should achieve drastically lower terminal error and cost
    assert s2_mpc < s2_open / 2.5, "MPC should reduce terminal error by > 2.5x"
    assert cost_f_mpc < cost_f_open / 8.0, "MPC should reduce terminal cost by > 8x"
    assert np.isclose(s2_open, 3.52631579, atol=1e-6)
    assert np.isclose(s2_mpc, 1.12631579, atol=1e-6)
    print(">> SUCCESS: MPC closed-loop feedback provides order-of-magnitude disturbance rejection!\n")


# =====================================================================
# 5. Section 6 Numerical Illustrations 2 - 5 Verifications
# =====================================================================

def verify_illustration2_lqr_backward():
    print("=" * 70)
    print("5. VERIFYING SECTION 6 ILLUSTRATION 2: 1D LQR BACKWARD RICCATI")
    print("=" * 70)
    A = 1.1
    B = 0.5
    Q = 1.0
    R = 0.1
    Qf = 2.0

    # Step t = 2:
    P2 = Qf
    assert np.isclose(P2, 2.0, atol=1e-12)

    # Step t = 1:
    denom1 = R + (B**2) * P2  # 0.1 + 0.25 * 2.0 = 0.6
    K1 = -(B * P2 * A) / denom1  # -1.1 / 0.6 = -11/6
    P1 = Q + (A**2) * P2 + A * P2 * B * K1  # 1.0 + 2.42 - 1.1*(11/6) = 421/300
    assert np.isclose(K1, -11.0 / 6.0, atol=1e-12)
    assert np.isclose(P1, 421.0 / 300.0, atol=1e-12)

    # Step t = 0:
    denom0 = R + (B**2) * P1  # 0.1 + 0.25 * (421/300) = 541/1200
    K0 = -(B * P1 * A) / denom0  # -(0.5 * 421/300 * 1.1) / (541/1200) = -4631/2705
    P0 = Q + (A**2) * P1 + A * P1 * B * K0  # 186191 / 135250
    assert np.isclose(K0, -4631.0 / 2705.0, atol=1e-12)
    assert np.isclose(P0, 186191.0 / 135250.0, atol=1e-12)

    print(f"P2 = {P2:.4f}")
    print(f"Step t=1: K1 = {K1:.6f} (-11/6), P1 = {P1:.6f} (421/300)")
    print(f"Step t=0: K0 = {K0:.6f} (-4631/2705), P0 = {P0:.6f} (186191/135250)")
    print(">> SUCCESS: Illustration 2 exact Riccati fractions and decimals verified!\n")


def verify_illustration3_pendulum_ilqr():
    print("=" * 70)
    print("6. VERIFYING SECTION 6 ILLUSTRATION 3: PENDULUM iLQR BACKWARD EXPANSION")
    print("=" * 70)
    dt = 0.1
    g, l, m = 10.0, 1.0, 1.0
    x_bar = np.array([np.pi / 6.0, 0.2])
    u_bar = 0.0

    fx = np.array([[1.0, dt], [-dt * (g / l) * np.cos(x_bar[0]), 1.0]])
    fu = np.array([[0.0], [dt / (m * (l**2))]])
    Q_pen = np.diag([2.0, 0.5])
    R_pen = np.array([[0.2]])
    Qf_pen = np.diag([5.0, 1.0])

    x_next = np.array([
        x_bar[0] + dt * x_bar[1],
        x_bar[1] - dt * (g / l) * np.sin(x_bar[0]) + dt * (1.0 / (m * (l**2))) * u_bar
    ])
    Vx_prime = Qf_pen @ x_next
    Vxx_prime = Qf_pen

    lx = Q_pen @ x_bar
    lu = R_pen @ np.array([u_bar])
    lxx = Q_pen
    luu = R_pen
    lux = np.zeros((1, 2))

    Qx = lx + fx.T @ Vx_prime
    Qu = lu + fu.T @ Vx_prime
    Qxx = lxx + fx.T @ Vxx_prime @ fx
    Quu = luu + fu.T @ Vxx_prime @ fu
    Qux = lux + fu.T @ Vxx_prime @ fx

    Quu_inv = np.linalg.inv(Quu)
    k = -Quu_inv @ Qu
    K = -Quu_inv @ Qux

    Vx = Qx + Qux.T @ k
    Vxx = Qxx - K.T @ Quu @ K
    dV = float(-0.5 * k.T @ Quu @ k)

    assert np.isclose(k[0], 1.0 / 7.0, atol=1e-12)
    assert np.isclose(Quu[0, 0], 0.21, atol=1e-12)
    assert np.isclose(Qu[0], -0.03, atol=1e-12)
    assert np.isclose(dV, -0.002142857142857142, atol=1e-12)

    print(f"Nominal State: theta = {x_bar[0]:.4f} rad, omega = {x_bar[1]:.4f} rad/s")
    print(f"Qu = {Qu[0]:.4f}, Quu = {Quu[0,0]:.4f}")
    print(f"Feedforward k = {k[0]:.6f} (exact 1/7)")
    print(f"Feedback K = [{K[0,0]:.6f}, {K[0,1]:.6f}]")
    print(f"Expected cost improvement dV = {dV:.6f}")
    print(">> SUCCESS: Illustration 3 pendulum iLQR expansion verified!\n")


def verify_illustration4_cem_sampling():
    print("=" * 70)
    print("7. VERIFYING SECTION 6 ILLUSTRATION 4: CEM SAMPLING & ELITE UPDATE")
    print("=" * 70)
    s0 = 2.0
    samples = np.array([
        [-1.2, -0.6],  # 1
        [-0.5, -0.2],  # 2
        [-1.5, -0.4],  # 3
        [-0.8, -1.0],  # 4
        [-1.0, -0.8],  # 5
        [-0.2, -0.5],  # 6
        [-1.4, -0.5],  # 7
        [-0.7, -0.3],  # 8
        [-1.3, -0.5],  # 9
        [-0.4, -0.8]   # 10
    ])
    costs = np.array([(s0 + u[0])**2 + u[0]**2 + 2.0 * (s0 + u[0] + u[1])**2 + u[1]**2 for u in samples])
    elite_idx = np.argsort(costs)[:3]
    assert np.array_equal(elite_idx, [8, 0, 6]), f"Expected elites [8, 0, 6] (1-based: 9, 1, 7), got {elite_idx}"

    elites = samples[elite_idx]
    mu_elite = np.mean(elites, axis=0)
    sigma_elite = np.std(elites, axis=0)

    assert np.isclose(mu_elite[0], -1.3, atol=1e-12)
    assert np.isclose(mu_elite[1], -1.6 / 3.0, atol=1e-12)

    beta = 0.7
    mu_prior = np.array([0.0, 0.0])
    sigma_prior = np.array([1.0, 1.0])
    mu_new = beta * mu_elite + (1 - beta) * mu_prior
    sigma_new = beta * sigma_elite + (1 - beta) * sigma_prior

    assert np.isclose(mu_new[0], -0.91, atol=1e-12)
    assert np.isclose(mu_new[1], -0.37333333333333335, atol=1e-12)

    print(f"Top 3 Elite candidates: {elite_idx + 1} with costs {costs[elite_idx]}")
    print(f"Elite mean: mu_elite = {mu_elite}")
    print(f"Updated Gaussian params: mu = {mu_new}, sigma = {sigma_new}")
    print(">> SUCCESS: Illustration 4 CEM trajectory elite selection and update verified!\n")


def verify_illustration5_receding_horizon():
    print("=" * 70)
    print("8. VERIFYING SECTION 6 ILLUSTRATION 5: RECEDING HORIZON MPC VS OPEN-LOOP")
    print("=" * 70)
    a, b = 1.0, 1.0
    q, r, qf = 2.0, 1.0, 4.0
    K_mpc = -14.0 / 19.0
    x0 = 10.0
    u0_ol = K_mpc * x0
    x1_nom = a * x0 + b * u0_ol
    u1_ol = -0.8 * x1_nom
    x2_nom = a * x1_nom + b * u1_ol
    w = [2.0, 2.0]

    # Open-loop execution under noise:
    x1_ol = a * x0 + b * u0_ol + w[0]
    x2_ol = a * x1_ol + b * u1_ol + w[1]
    cost_ol = 0.5 * (q * (x0**2) + r * (u0_ol**2)) + 0.5 * (q * (x1_ol**2) + r * (u1_ol**2)) + 0.5 * qf * (x2_ol**2)

    # Receding-horizon MPC execution under noise:
    x1_mpc = a * x0 + b * u0_ol + w[0]
    u1_mpc = K_mpc * x1_mpc
    x2_mpc = a * x1_mpc + b * u1_mpc + w[1]
    cost_mpc = 0.5 * (q * (x0**2) + r * (u0_ol**2)) + 0.5 * (q * (x1_mpc**2) + r * (u1_mpc**2)) + 0.5 * qf * (x2_mpc**2)

    assert np.isclose(x2_ol, 4.526315789473684, atol=1e-12)
    assert np.isclose(x2_mpc, 3.2188365650969527, atol=1e-12)
    assert np.isclose(cost_ol, 191.78947368421055, atol=1e-12)
    assert np.isclose(cost_mpc, 175.14361596767554, atol=1e-12)

    print(f"Open-Loop:  Final x2 = {x2_ol:.4f}, Total Realized Cost = {cost_ol:.4f}")
    print(f"Closed-Loop MPC: Final x2 = {x2_mpc:.4f}, Total Realized Cost = {cost_mpc:.4f}")
    print(">> SUCCESS: Illustration 5 receding horizon closed-loop execution verified!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_multidimensional_lqr()
    verify_cem_optimizer()
    verify_mpc_disturbance_rejection()
    verify_illustration2_lqr_backward()
    verify_illustration3_pendulum_ilqr()
    verify_illustration4_cem_sampling()
    verify_illustration5_receding_horizon()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 22 VERIFICATIONS (INCLUDING ALL 5 ILLUSTRATIONS) PASSED!")
    print("=" * 70)

