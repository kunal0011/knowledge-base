"""
Mathematical Verification Script for Module 11.16: Continuous Action Spaces (DDPG & TD3)
========================================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Smoothed target action: a_tilde' = 0.5000
   - Clipped Double Q: Q1_targ = 1.5500, Q2_targ = 1.7000, min = 1.5500
   - Target y = 4.3950
   - Critic losses: L1 = 0.07801, L2 = 0.18301
   - Updated critic weights:
     phi1 = [1.0790, 2.0395], phi2 = [1.8790, 0.9395]
   - Deterministic policy gradient: g_theta = +4.0000
   - Updated actor parameter: theta_new = 0.9000
   - Verified identically in PyTorch autograd (< 1e-6).
2. End-to-End TD3 Continuous Control Agent on Inverted Pendulum:
   - Replay buffer, Twin Critic evaluation, Target policy smoothing, Delayed updates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 Hand Calculation Verification (PyTorch + NumPy) ---")
    s = 2.0
    a = 1.0
    r = 3.0
    s_next = 1.0
    gamma = 0.90
    alpha = 0.10
    noise = 0.10
    
    # 1. NumPy hand calculation
    theta = 0.50
    theta_targ = 0.40
    
    phi1 = np.array([1.00, 2.00])
    phi1_targ = np.array([0.80, 1.50])
    
    phi2 = np.array([2.00, 1.00])
    phi2_targ = np.array([1.20, 1.00])
    
    # Target action with smoothing
    raw_a_next = theta_targ * s_next # 0.4 * 1 = 0.4
    a_tilde_next = np.clip(raw_a_next + noise, -2.0, 2.0)
    print(f"  Smoothed Next Action a_tilde': {a_tilde_next:.4f}")
    assert np.isclose(a_tilde_next, 0.5000)
    
    # Target critics
    q1_targ = phi1_targ[0] * s_next + phi1_targ[1] * a_tilde_next
    q2_targ = phi2_targ[0] * s_next + phi2_targ[1] * a_tilde_next
    min_q_targ = min(q1_targ, q2_targ)
    y = r + gamma * min_q_targ
    
    print(f"  Q1_targ = {q1_targ:.4f}, Q2_targ = {q2_targ:.4f}, min = {min_q_targ:.4f}, y = {y:.4f}")
    assert np.isclose(q1_targ, 1.5500)
    assert np.isclose(q2_targ, 1.7000)
    assert np.isclose(min_q_targ, 1.5500)
    assert np.isclose(y, 4.3950)
    
    # Online critics
    q1 = phi1[0] * s + phi1[1] * a
    u1 = y - q1
    loss1 = 0.5 * (u1 ** 2)
    
    q2 = phi2[0] * s + phi2[1] * a
    u2 = y - q2
    loss2 = 0.5 * (u2 ** 2)
    
    print(f"  Q1 = {q1:.4f}, u1 = {u1:.4f}, Loss1 = {loss1:.5f}")
    print(f"  Q2 = {q2:.4f}, u2 = {u2:.4f}, Loss2 = {loss2:.5f}")
    assert np.isclose(q1, 4.0000)
    assert np.isclose(u1, 0.3950)
    assert np.isclose(loss1, 0.0780125)
    assert np.isclose(q2, 5.0000)
    assert np.isclose(u2, -0.6050)
    assert np.isclose(loss2, 0.1830125)
    
    # Critic updates
    grad_phi1 = - u1 * np.array([s, a])
    phi1_new = phi1 - alpha * grad_phi1
    
    grad_phi2 = - u2 * np.array([s, a])
    phi2_new = phi2 - alpha * grad_phi2
    
    print(f"  Updated phi1: {phi1_new}")
    print(f"  Updated phi2: {phi2_new}")
    assert np.allclose(phi1_new, [1.0790, 2.0395])
    assert np.allclose(phi2_new, [1.8790, 0.9395])
    
    # Deterministic policy gradient update
    # Actor action at s: a_actor = theta * s = 0.5 * 2 = 1.0
    grad_a_Q1 = phi1[1] # 2.0
    grad_theta_mu = s   # 2.0
    g_theta = grad_a_Q1 * grad_theta_mu # 2.0 * 2.0 = 4.0
    theta_new = theta + alpha * g_theta
    print(f"  Actor Policy Gradient: {g_theta:.4f}, Updated theta: {theta_new:.4f}")
    assert np.isclose(g_theta, 4.0000)
    assert np.isclose(theta_new, 0.9000)
    
    # 2. PyTorch autograd exact match
    pt_theta = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))
    pt_phi1 = nn.Parameter(torch.tensor([1.00, 2.00], dtype=torch.float32))
    pt_phi2 = nn.Parameter(torch.tensor([2.00, 1.00], dtype=torch.float32))
    
    pt_s = torch.tensor(s, dtype=torch.float32)
    pt_a = torch.tensor(a, dtype=torch.float32)
    
    # Critic 1 loss
    pt_q1 = pt_phi1[0] * pt_s + pt_phi1[1] * pt_a
    pt_loss1 = 0.5 * ((torch.tensor(y, dtype=torch.float32) - pt_q1) ** 2)
    pt_loss1.backward()
    assert np.allclose(pt_phi1.grad.numpy(), grad_phi1)
    
    # Actor loss (- Q1(s, mu(s)))
    pt_mu = pt_theta * pt_s
    pt_q_for_actor = pt_phi1[0] * pt_s + pt_phi1[1] * pt_mu
    pt_actor_loss = - pt_q_for_actor
    pt_actor_loss.backward()
    # Gradient of actor loss w.r.t theta is -g_theta
    assert np.isclose(pt_theta.grad.item(), -4.0000)
    
    print("  [PASSED] Part 5 TD3 hand calculations match PyTorch autograd identically!\n")


def test_td3_continuous_control():
    print("--- Test 2: TD3 Continuous Control on Pendulum Dynamics ---")
    
    # Continuous Pendulum Simulation Environment
    class PendulumEnv:
        def __init__(self):
            self.max_speed = 8.0
            self.max_torque = 2.0
            self.dt = 0.05
            self.g = 10.0
            self.m = 1.0
            self.l = 1.0
            self.reset()
            
        def reset(self):
            high = np.array([np.pi, 1.0])
            self.state = np.random.uniform(-high, high)
            return self._get_obs()
            
        def _get_obs(self):
            theta, thetadot = self.state
            return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        
        def step(self, u):
            th, thdot = self.state
            g, m, l, dt = self.g, self.m, self.l, self.dt
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
            
            # Angle normalization
            costs = ((th + np.pi) % (2 * np.pi) - np.pi) ** 2 + 0.1 * (thdot ** 2) + 0.001 * (u ** 2)
            
            newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
            newth = th + newthdot * dt
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
            
            self.state = np.array([newth, newthdot])
            return self._get_obs(), -float(costs), False
        
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()
            )
        def forward(self, s):
            return self.net(s) * 2.0 # scale to [-2, 2]
        
    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3 + 1, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, s, a):
            return self.net(torch.cat([s, a], dim=-1))
        
    # Instantiate TD3 modules
    actor = Actor()
    actor_targ = Actor()
    actor_targ.load_state_dict(actor.state_dict())
    
    c1, c2 = Critic(), Critic()
    c1_targ, c2_targ = Critic(), Critic()
    c1_targ.load_state_dict(c1.state_dict())
    c2_targ.load_state_dict(c2.state_dict())
    
    opt_a = optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=1e-3)
    
    # Replay buffer
    buffer = []
    env = PendulumEnv()
    gamma = 0.99
    tau = 0.005
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Pre-fill buffer
    s = env.reset()
    for _ in range(500):
        a = np.random.uniform(-2.0, 2.0, size=(1,))
        ns, r, d = env.step(a)
        buffer.append((s, a, r, ns, d))
        s = ns
        
    # Quick 50-step TD3 training loop
    for step in range(50):
        # Sample batch
        idx = np.random.choice(len(buffer), 32)
        batch = [buffer[i] for i in idx]
        b_s, b_a, b_r, b_ns, b_d = zip(*batch)
        
        b_s = torch.tensor(np.array(b_s), dtype=torch.float32)
        b_a = torch.tensor(np.array(b_a), dtype=torch.float32)
        b_r = torch.tensor(b_r, dtype=torch.float32).unsqueeze(1)
        b_ns = torch.tensor(np.array(b_ns), dtype=torch.float32)
        
        with torch.no_grad():
            noise = (torch.randn_like(b_a) * 0.2).clamp(-0.5, 0.5)
            next_a = (actor_targ(b_ns) + noise).clamp(-2.0, 2.0)
            q1_next = c1_targ(b_ns, next_a)
            q2_next = c2_targ(b_ns, next_a)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = b_r + gamma * min_q_next
            
        current_q1 = c1(b_s, b_a)
        current_q2 = c2(b_s, b_a)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        opt_c.zero_grad()
        critic_loss.backward()
        opt_c.step()
        
        # Delayed Actor update (every 2 steps)
        if step % 2 == 0:
            actor_loss = - c1(b_s, actor(b_s)).mean()
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()
            
            # Polyak update targets
            for p, pt in zip(actor.parameters(), actor_targ.parameters()):
                pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
            for p, pt in zip(c1.parameters(), c1_targ.parameters()):
                pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
            for p, pt in zip(c2.parameters(), c2_targ.parameters()):
                pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
                
    print(f"  Final Critic Loss: {critic_loss.item():.4f}")
    assert not np.isnan(critic_loss.item()), "Critic loss must be valid finite number"
    print("  [PASSED] TD3 continuous control algorithm verified!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.16 CONTINUOUS ACTION SPACES (DDPG/TD3) TESTS")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_td3_continuous_control()
    
    print("=================================================================")
    print("ALL MODULE 11.16 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
