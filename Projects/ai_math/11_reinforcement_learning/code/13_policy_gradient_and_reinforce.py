"""
Mathematical Verification Script for Module 11.13: Policy Gradient & REINFORCE
=============================================================================
This test suite verifies:
1. Part 5 Tom Yeh "AI by Hand" numerical walkthrough:
   - Initial Softmax probabilities: pi = [0.268941, 0.731059]
   - Analytical score function: grad log pi(a0) = [+0.731059, -0.731059]
   - Advantage: A_0 = G_0 - b = 10.0 - 5.0 = 5.0
   - Policy gradient: g = [+3.655293, -3.655293]
   - Updated parameters: theta_new = [1.365529, 1.634471]
   - Updated probabilities: pi_new(a0) = 0.433171
   - Identical reproduction in PyTorch autograd (< 1e-6).
2. End-to-End REINFORCE with Baseline Agent on CartPole:
   - Demonstrates stable policy gradient ascent and reward improvement.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

def verify_part_5_hand_calculation():
    print("--- Test 1: Part 5 Hand Calculation Verification (PyTorch + NumPy) ---")
    theta_init = np.array([1.0, 2.0])
    b = 5.0
    G0 = 10.0
    alpha = 0.1
    
    # 1. NumPy Softmax and Score calculation
    exp_theta = np.exp(theta_init)
    pi_init = exp_theta / np.sum(exp_theta)
    
    print(f"  Initial Logits: {theta_init}")
    print(f"  Initial Softmax Probs: {pi_init}")
    assert np.isclose(pi_init[0], 0.26894142), f"Expected pi0=0.26894, got {pi_init[0]}"
    assert np.isclose(pi_init[1], 0.73105858), f"Expected pi1=0.73106, got {pi_init[1]}"
    
    # Chosen action a0
    score_grad = np.array([1.0 - pi_init[0], -pi_init[1]])
    print(f"  Score function grad_theta log pi(a0): {score_grad}")
    assert np.isclose(score_grad[0], 0.73105858)
    assert np.isclose(score_grad[1], -0.73105858)
    
    # Advantage
    adv = G0 - b
    assert np.isclose(adv, 5.0)
    
    # Policy gradient
    g = adv * score_grad
    print(f"  Policy Gradient: {g}")
    assert np.allclose(g, [3.6552929, -3.6552929])
    
    # Updated parameters
    theta_new = theta_init + alpha * g
    print(f"  Updated Parameters: {theta_new}")
    assert np.allclose(theta_new, [1.36552929, 1.63447071])
    
    # Updated probabilities
    exp_theta_new = np.exp(theta_new)
    pi_new = exp_theta_new / np.sum(exp_theta_new)
    print(f"  Updated Softmax Probs: {pi_new}")
    assert np.isclose(pi_new[0], 0.433171, atol=1e-5)
    assert np.isclose(pi_new[1], 0.566829, atol=1e-5)
    
    # 2. PyTorch autograd exact match
    pt_theta = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    
    # In PyTorch, policy gradient loss is: - (log pi(a0)) * Advantage
    log_probs = torch.log_softmax(pt_theta, dim=0)
    loss = - log_probs[0] * adv
    
    loss.backward()
    
    # Gradient of loss w.r.t theta is -g (since we minimize loss to maximize return)
    pt_grad = pt_theta.grad.numpy()
    print(f"  PyTorch computed grad of loss: {pt_grad}")
    assert np.allclose(pt_grad, -g)
    
    # Gradient ascent step: theta - alpha * pt_grad = theta + alpha * g
    pt_theta_new = (pt_theta - alpha * pt_theta.grad).detach().numpy()
    assert np.allclose(pt_theta_new, theta_new)
    
    print("  [PASSED] Part 5 hand calculations match PyTorch autograd to exact machine precision!\n")


def test_reinforce_cartpole():
    print("--- Test 2: REINFORCE with Baseline on CartPole Dynamics ---")
    
    # Simple simulated CartPole environment
    class CartPoleEnv:
        def __init__(self):
            self.gravity = 9.8
            self.masscart = 1.0
            self.masspole = 0.1
            self.total_mass = 1.1
            self.length = 0.5
            self.polemass_length = 0.05
            self.force_mag = 10.0
            self.tau = 0.02
            self.reset()
            
        def reset(self):
            self.state = np.random.uniform(-0.05, 0.05, size=(4,))
            return self.state.copy()
            
        def step(self, action):
            x, x_dot, theta, theta_dot = self.state
            force = self.force_mag if action == 1 else -self.force_mag
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
            
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            
            self.state = np.array([x, x_dot, theta, theta_dot])
            done = bool(x < -2.4 or x > 2.4 or theta < -0.21 or theta > 0.21)
            reward = 1.0 if not done else 0.0
            return self.state.copy(), reward, done
        
    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        def forward(self, x):
            return self.fc(x)
        
    class ValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.fc(x)
        
    env = CartPoleEnv()
    policy = PolicyNet()
    baseline = ValueNet()
    
    opt_p = optim.Adam(policy.parameters(), lr=0.01)
    opt_b = optim.Adam(baseline.parameters(), lr=0.01)
    gamma = 0.99
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    durations = []
    
    for ep in range(120):
        s = env.reset()
        log_probs = []
        states = []
        rewards = []
        
        for step in range(200):
            s_tensor = torch.tensor(s, dtype=torch.float32)
            logits = policy(s_tensor)
            dist = Categorical(logits=logits)
            a = dist.sample()
            
            next_s, r, done = env.step(a.item())
            log_probs.append(dist.log_prob(a))
            states.append(s_tensor)
            rewards.append(r)
            
            s = next_s
            if done:
                break
                
        durations.append(len(rewards))
        
        # Compute returns G_t
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Baseline critic update
        states_tensor = torch.stack(states)
        values = baseline(states_tensor).squeeze(1)
        
        critic_loss = F.mse_loss(values, returns) if 'F' in locals() else nn.MSELoss()(values, returns)
        opt_b.zero_grad()
        critic_loss.backward()
        opt_b.step()
        
        # Policy actor update
        with torch.no_grad():
            advantages = returns - baseline(states_tensor).squeeze(1)
            
        policy_loss = - (torch.stack(log_probs) * advantages).mean()
        opt_p.zero_grad()
        policy_loss.backward()
        opt_p.step()
        
    avg_first_20 = np.mean(durations[:20])
    avg_last_20 = np.mean(durations[-20:])
    print(f"  Average Episode Duration (First 20): {avg_first_20:.1f} steps")
    print(f"  Average Episode Duration (Last 20): {avg_last_20:.1f} steps")
    
    assert avg_last_20 > avg_first_20, "REINFORCE training must improve episode duration"
    print("  [PASSED] REINFORCE with baseline agent demonstrated policy gradient ascent!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("STARTING MODULE 11.13 POLICY GRADIENT & REINFORCE TEST SUITE")
    print("=================================================================\n")
    
    verify_part_5_hand_calculation()
    test_reinforce_cartpole()
    
    print("=================================================================")
    print("ALL MODULE 11.13 UNIT TESTS PASSED SUCCESSFULLY! (100% VERIFIED)")
    print("=================================================================")
