"""
Decision Transformer (DT) & Sequence Modeling RL Verification Suite
Module 11: Reinforcement Learning - Chapter 26

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - Modality embedding projections e_R = [1.0, 2.0], e_s = [2.0, -1.0]
   - Scaled dot-product attention logits [0.0, 3.535534]
   - Softmax attention weights [0.028318, 0.971682]
   - Context output Y2 = [0.985841, -0.457523]
   - Predicted action a1_hat = 0.070795 (< 1e-5)
2. Complete PyTorch Decision Transformer Architecture:
   - Causal multi-head self-attention
   - Modality projections and shared timestep positional encodings
   - Supervised trajectory action loss backward pass
3. Autoregressive Rollout Simulation with online Return-to-Go decrementing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' DECISION TRANSFORMER CALCULATIONS")
    print("=" * 70)

    # Inputs:
    rtg_1 = 10.0
    s_1 = 5.0

    # Projection weights:
    W_R = np.array([0.1, 0.2])
    W_s = np.array([0.4, -0.2])

    # 1. Embeddings:
    e_R = W_R * rtg_1  # [1.0, 2.0]
    e_s = W_s * s_1    # [2.0, -1.0]

    print(f"e_R: [{e_R[0]:.4f}, {e_R[1]:.4f}] (Expected: [1.0, 2.0])")
    print(f"e_s: [{e_s[0]:.4f}, {e_s[1]:.4f}] (Expected: [2.0, -1.0])")
    assert np.allclose(e_R, [1.0, 2.0], atol=1e-6)
    assert np.allclose(e_s, [2.0, -1.0], atol=1e-6)

    # 2. Attention Projections (W_Q = I, W_K = I, W_V = 0.5 * I):
    q2 = e_s
    k1 = e_R
    k2 = e_s
    v1 = 0.5 * e_R  # [0.5, 1.0]
    v2 = 0.5 * e_s  # [1.0, -0.5]

    # 3. Attention Logits for Query 2 (State token s1):
    dot_1 = np.dot(q2, k1)  # 2(1) + (-1)(2) = 0.0
    dot_2 = np.dot(q2, k2)  # 2(2) + (-1)(-1) = 5.0
    sqrt_dk = np.sqrt(2.0)

    z1 = dot_1 / sqrt_dk  # 0.0
    z2 = dot_2 / sqrt_dk  # 5.0 / sqrt(2) ≈ 3.535534

    print(f"Attention Logits: z1 = {z1:.6f}, z2 = {z2:.6f}")
    assert np.isclose(z1, 0.0, atol=1e-6)
    assert np.isclose(z2, 5.0 / np.sqrt(2.0), atol=1e-6)

    # 4. Softmax Weights:
    exp_z = np.exp([z1, z2])
    w = exp_z / np.sum(exp_z)
    print(f"Attention Weights: w1 = {w[0]:.6f} (Exp: 0.028318), w2 = {w[1]:.6f} (Exp: 0.971682)")
    assert np.isclose(w[0], 0.028318, atol=1e-5)
    assert np.isclose(w[1], 0.971682, atol=1e-5)

    # 5. Context Output Y2:
    Y2 = w[0] * v1 + w[1] * v2
    print(f"Context Y2: [{Y2[0]:.6f}, {Y2[1]:.6f}] (Expected: [0.985841, -0.457523])")
    assert np.isclose(Y2[0], 0.985841, atol=1e-5)
    assert np.isclose(Y2[1], -0.457523, atol=1e-5)

    # 6. Action Head Output:
    W_act = np.array([1.0, 2.0])
    pred_action = np.dot(W_act, Y2)
    print(f"Predicted Action a1_hat = {pred_action:.6f} (Expected: 0.070795)")
    assert np.isclose(pred_action, 0.070795, atol=1e-5)

    # 7. PyTorch Exact Equivalence:
    X_torch = torch.tensor([[1.0, 2.0], [2.0, -1.0]], dtype=torch.float32)  # [e_R, e_s]
    Q_torch = X_torch[1:2]  # query for token 2
    K_torch = X_torch
    V_torch = 0.5 * X_torch

    scores = torch.matmul(Q_torch, K_torch.T) / np.sqrt(2.0)
    attn_probs = F.softmax(scores, dim=-1)
    context_torch = torch.matmul(attn_probs, V_torch)
    action_torch = torch.matmul(context_torch, torch.tensor([1.0, 2.0]))

    print(f"PyTorch Generated Action: {action_torch.item():.6f}")
    assert np.isclose(action_torch.item(), pred_action, atol=1e-6)
    print(">> SUCCESS: Decision Transformer forward pass verified to exact precision!\n")


# =====================================================================
# 2. Complete PyTorch Decision Transformer Architecture
# =====================================================================

class DecisionTransformer(nn.Module):
    """
    Autoregressive Decision Transformer for conditional RL.
    """
    def __init__(self, state_dim, act_dim, hidden_dim=64, max_ep_len=1000, n_heads=2, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Modality embedding projections
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)

        # Timestep positional embedding
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        # Causal Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        # Action prediction head (from state representation)
        self.predict_action = nn.Linear(hidden_dim, act_dim)

    def forward(self, states, actions, rtgs, timesteps):
        """
        states: (B, K, state_dim)
        actions: (B, K, act_dim)
        rtgs: (B, K, 1)
        timesteps: (B, K)
        """
        B, K, _ = states.shape

        # 1. Project modalities + add shared timestep embedding
        time_emb = self.embed_timestep(timesteps)  # (B, K, hidden_dim)

        r_emb = self.embed_rtg(rtgs) + time_emb
        s_emb = self.embed_state(states) + time_emb
        a_emb = self.embed_action(actions) + time_emb

        # 2. Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        # Stack to (B, K, 3, hidden_dim) and reshape to (B, 3*K, hidden_dim)
        stacked_tokens = torch.stack((r_emb, s_emb, a_emb), dim=2)
        sequence = stacked_tokens.reshape(B, 3 * K, self.hidden_dim)

        # 3. Create causal mask (size 3K x 3K)
        seq_len = 3 * K
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(sequence.device)

        # 4. Pass through Transformer
        hidden_states = self.transformer(sequence, mask=causal_mask)

        # 5. Extract state token representations (index 1, 4, 7, ..., 3K-2) to predict actions
        state_indices = torch.arange(1, seq_len, 3, device=sequence.device)
        state_outputs = hidden_states[:, state_indices, :]  # (B, K, hidden_dim)

        # 6. Predict actions
        action_preds = self.predict_action(state_outputs)  # (B, K, act_dim)
        return action_preds


def verify_decision_transformer_model():
    print("=" * 70)
    print("2. VERIFYING PYTORCH DECISION TRANSFORMER FORWARD & BACKWARD PASS")
    print("=" * 70)

    torch.manual_seed(42)
    B, K = 4, 10
    state_dim = 4
    act_dim = 2

    model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, hidden_dim=32, max_ep_len=500)

    # Synthetic trajectory batch
    states = torch.randn(B, K, state_dim)
    actions = torch.randn(B, K, act_dim)
    rtgs = torch.randn(B, K, 1)
    timesteps = torch.arange(K).unsqueeze(0).repeat(B, 1)

    # Forward pass
    action_preds = model(states, actions, rtgs, timesteps)
    assert action_preds.shape == (B, K, act_dim)

    # Supervised MSE loss
    loss = F.mse_loss(action_preds, actions)
    print(f"Batch MSE Loss: {loss.item():.4f}")
    assert loss.item() > 0.0

    # Backward gradient check
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient in {name}"
    print(">> SUCCESS: Decision Transformer successfully executed forward and backward pass!\n")


# =====================================================================
# 3. Autoregressive Rollout Simulation with Online Return-to-Go
# =====================================================================

def simulate_autoregressive_dt_rollout():
    print("=" * 70)
    print("3. VERIFYING AUTOREGRESSIVE ROLLOUT & ONLINE RETURN-TO-GO DECREMENTING")
    print("=" * 70)

    torch.manual_seed(42)
    model = DecisionTransformer(state_dim=2, act_dim=1, hidden_dim=16, max_ep_len=100)

    # Prompt target return
    target_return = 100.0
    current_rtg = target_return
    current_state = np.array([1.0, 0.0])

    history_states = []
    history_actions = []
    history_rtgs = []
    history_timesteps = []

    # Roll out for 3 timesteps
    for t in range(3):
        history_states.append(current_state)
        history_rtgs.append([current_rtg])
        history_timesteps.append(t)

        # Prepare tensors with dummy zero action for current step
        s_tensor = torch.tensor(np.array([history_states]), dtype=torch.float32)
        # For actions, pad current step with zeros
        acts = history_actions + [np.zeros(1)]
        a_tensor = torch.tensor(np.array([acts]), dtype=torch.float32)
        r_tensor = torch.tensor(np.array([history_rtgs]), dtype=torch.float32)
        t_tensor = torch.tensor(np.array([history_timesteps]), dtype=torch.long)

        with torch.no_grad():
            pred_acts = model(s_tensor, a_tensor, r_tensor, t_tensor)
            action_t = pred_acts[0, -1].numpy()

        history_actions.append(action_t)

        # Mock environment transition: receives reward +10.0
        step_reward = 10.0
        current_state = current_state + 0.1 * action_t
        current_rtg -= step_reward

        print(f"Step t={t}: Executed a={action_t[0]:.4f} | Reward={step_reward:.1f} | Next RTG={current_rtg:.1f}")

    # RTG should decrement by exactly 3 * 10 = 30 -> 70.0
    assert np.isclose(current_rtg, 70.0), f"Expected final RTG 70.0, got {current_rtg}"
    print(">> SUCCESS: Autoregressive DT online rollout and RTG tracking verified!\n")


# =====================================================================
# 4. Section 6 Solved Illustrations Numerical Verification
# =====================================================================

def verify_section6_illustrations():
    print("=" * 70)
    print("4. VERIFYING SECTION 6 SOLVED ILLUSTRATIONS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Illustration 2: 3-step Episode Trajectory Tokenization
    # -------------------------------------------------------------
    r = np.array([2.0, 5.0, -1.0])
    R = np.zeros(3)
    R[2] = r[2]
    R[1] = r[1] + R[2]
    R[0] = r[0] + R[1]
    assert np.allclose(R, [6.0, 4.0, -1.0])

    s = np.array([[1.0, 0.0], [0.5, 1.5], [-1.0, 2.0]])
    a = np.array([[0.5], [-0.8], [1.2]])

    W_R = np.array([[0.5], [-0.2]])
    W_s = np.array([[0.4, 0.1], [-0.2, 0.3]])
    W_a = np.array([[0.6], [0.2]])
    pos = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

    e_R1 = W_R.flatten() * R[0] + pos[0]
    e_s1 = W_s @ s[0] + pos[0]
    e_a1 = W_a.flatten() * a[0] + pos[0]
    assert np.allclose(e_R1, [3.1, -1.1])
    assert np.allclose(e_s1, [0.5, -0.1])
    assert np.allclose(e_a1, [0.4, 0.2])

    e_R2 = W_R.flatten() * R[1] + pos[1]
    e_s2 = W_s @ s[1] + pos[1]
    e_a2 = W_a.flatten() * a[1] + pos[1]
    assert np.allclose(e_R2, [2.2, -0.6])
    assert np.allclose(e_s2, [0.55, 0.55])
    assert np.allclose(e_a2, [-0.28, 0.04])

    e_R3 = W_R.flatten() * R[2] + pos[2]
    e_s3 = W_s @ s[2] + pos[2]
    e_a3 = W_a.flatten() * a[2] + pos[2]
    assert np.allclose(e_R3, [-0.2, 0.5])
    assert np.allclose(e_s3, [0.1, 1.1])
    assert np.allclose(e_a3, [1.02, 0.54])
    print(">> Illustration 2 (Tokenization & Positional Embeddings): Verified!")

    # -------------------------------------------------------------
    # Illustration 3: Causal Masking & Self-Attention Matrix
    # -------------------------------------------------------------
    U = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],
        [2.0, -1.0]
    ])
    S_raw = U @ U.T
    scale = np.sqrt(2.0)
    S = S_raw / scale
    mask = np.triu(np.full((4, 4), -np.inf), k=1)
    S_masked = S + mask

    def softmax_masked(row):
        valid = row != -np.inf
        m = np.max(row[valid])
        exp_r = np.zeros_like(row)
        exp_r[valid] = np.exp(row[valid] - m)
        return exp_r / np.sum(exp_r[valid])

    A = np.zeros((4, 4))
    for i in range(4):
        A[i] = softmax_masked(S_masked[i])
    Y = A @ U

    assert np.allclose(A[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(A[1], [0.055807, 0.944193, 0.0, 0.0], atol=1e-5)
    assert np.allclose(A[2], [0.197776, 0.401112, 0.401112, 0.0], atol=1e-5)
    assert np.allclose(A[3], [0.101068, 0.005974, 0.049834, 0.843125], atol=1e-5)
    assert np.allclose(Y[1], [0.055807, 1.888386], atol=1e-5)
    print(">> Illustration 3 (Causal Self-Attention Matrix Pass): Verified!")

    # -------------------------------------------------------------
    # Illustration 4: 2-Step Rollout with RTG Decrementing
    # -------------------------------------------------------------
    W_R_i4 = np.array([[0.1], [0.1]])
    W_s_i4 = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_a_i4 = np.array([[1.0], [0.0]])
    W_act_i4 = np.array([[1.0, 1.0]])

    R1 = 20.0
    s1 = np.array([1.0, 0.0])
    e_R1 = (W_R_i4 * R1).flatten()
    e_s1 = W_s_i4 @ s1

    z = np.array([np.dot(e_s1, e_R1), np.dot(e_s1, e_s1)]) / scale
    exp_z = np.exp(z)
    w1 = exp_z / np.sum(exp_z)
    Y1 = w1[0] * e_R1 + w1[1] * e_s1
    a1 = (W_act_i4 @ Y1)[0]
    assert np.isclose(a1, 3.009285, atol=1e-5)

    r1 = 3.0
    R2 = R1 - r1
    s2 = np.array([0.0, 2.0])
    e_a1 = (W_a_i4 * a1).flatten()
    e_R2 = (W_R_i4 * R2).flatten()
    e_s2 = W_s_i4 @ s2

    tokens = np.array([e_R1, e_s1, e_a1, e_R2, e_s2])
    dots = tokens @ e_s2
    z5 = dots / scale
    exp_z5 = np.exp(z5)
    w2 = exp_z5 / np.sum(exp_z5)
    Y2 = w2 @ tokens
    a2 = (W_act_i4 @ Y2)[0]
    assert np.isclose(a2, 3.051953, atol=1e-5)
    print(">> Illustration 4 (DT Inference Action Sampling & RTG Decrement): Verified!")

    # -------------------------------------------------------------
    # Illustration 5: Stochastic Pitfall & Sub-trajectory Stitching
    # -------------------------------------------------------------
    Q_safe = 18.0
    Q_gamble = 0.1 * 100.0 + 0.9 * 0.0
    assert Q_safe == 18.0 and Q_gamble == 10.0
    dt_return = 0.1 * 100.0 + 0.9 * 0.0
    assert dt_return == 10.0
    drop = (dt_return - Q_safe) / Q_safe
    assert np.isclose(drop, -0.444444, atol=1e-5)
    print(">> Illustration 5 (Stitching vs Stochastic MDP Failure): Verified!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_decision_transformer_model()
    simulate_autoregressive_dt_rollout()
    verify_section6_illustrations()
    print("=" * 70)
    print("ALL MODULE 11 CHAPTER 26 (DECISION TRANSFORMER) VERIFICATIONS PASSED!")
    print("=" * 70)

