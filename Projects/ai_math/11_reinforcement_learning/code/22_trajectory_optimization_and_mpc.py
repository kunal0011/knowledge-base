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
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_multidimensional_lqr()
    verify_cem_optimizer()
    verify_mpc_disturbance_rejection()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 22 (TRAJECTORY OPT & MPC) VERIFICATIONS PASSED!")
    print("=" * 70)
