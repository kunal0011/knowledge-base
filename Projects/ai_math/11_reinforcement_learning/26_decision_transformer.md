# Chapter 26: Decision Transformers & Sequence Modeling RL

---

## 1. Intuition & 101 Motivation

For decades, reinforcement learning was framed exclusively through the lens of **Dynamic Programming and the Bellman Equation**:
$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$

This formulation requires temporal-difference bootstrapping, actor-critic architectures, discount factors, target networks, and replay buffers. When scaled to large neural networks, this paradigm often suffers from numerical instabilities—most notoriously the **Deadly Triad** (bootstrapping + function approximation + off-policy distribution shift).

In 2021, a revolutionary paradigm emerged: **Can we throw away the Bellman equation entirely and solve Reinforcement Learning as generic autoregressive Sequence Modeling?**

Enter the **Decision Transformer (DT)** (Chen et al., NeurIPS 2021):
- An entire RL episode is transcribed into a sequence of tokens:
  $$\tau = \big( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \big)$$
  where $\hat{R}_t = \sum_{t'=t}^T r_{t'}$ is the **Return-to-Go** (the sum of future rewards remaining in the episode).
- A standard causal autoregressive Transformer (identical to GPT) is trained on offline trajectories using simple **supervised regression / cross-entropy loss**.
- At test time, to generate expert behavior, we simply condition the Transformer on a high desired return:
  $$\text{Prompt: } \text{"Generate actions that achieve } \hat{R}_1 = 100.0\text{"}$$
- The Transformer autoregressively predicts the exact sequence of actions required to satisfy that return prompt!

```
Target Return R_1 ---> State s_1 ---> [ Transformer (GPT) ] ---> Action a_1
                                                                     |
Environment Step <---------------------------------------------------+
       |
       v
Real Reward r_1 & Next State s_2
       |
       v
New Return R_2 = R_1 - r_1 ---> State s_2 ---> [ Transformer ] ---> Action a_2
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Trajectory Representation & Return-to-Go

Let a collected episode of length $T$ consist of states $s_t \in \mathbb{R}^{d_s}$, actions $a_t \in \mathbb{R}^{d_a}$, and scalar stage rewards $r_t \in \mathbb{R}$.

The **Return-to-Go (RTG)** at timestep $t$ is the exact empirical sum of future rewards:
$$\hat{R}_t \triangleq \sum_{t'=t}^T r_{t'}$$

It satisfies the deterministic backward recurrence:
$$\hat{R}_t = r_t + \hat{R}_{t+1} \iff \hat{R}_{t+1} = \hat{R}_t - r_t$$

The trajectory is ordered chronologically into a sequence of triplets:
$$\tau = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \right)$$

To enable finite-length transformer processing, the model attends over a sliding context window of the most recent $K$ timesteps, resulting in a sequence of $3K$ tokens:
$$\mathbf{x} = \left[ \hat{R}_{t-K+1}, s_{t-K+1}, a_{t-K+1}, \dots, \hat{R}_t, s_t \right]$$

---

### 2.2 Token Embedding & Modality Encodings

Because the three modalities (returns, states, actions) have different physical dimensions, each modality is passed through a distinct linear projection matrix:

$$e_t^R = W_R \hat{R}_t + e_t^{\text{pos}}$$
$$e_t^s = W_s s_t + e_t^{\text{pos}}$$
$$e_t^a = W_a a_t + e_t^{\text{pos}}$$

where:
- $W_R \in \mathbb{R}^{d \times 1}$, $W_s \in \mathbb{R}^{d \times d_s}$, $W_a \in \mathbb{R}^{d \times d_a}$ project each token into a shared hidden dimension $d$.
- $e_t^{\text{pos}} \in \mathbb{R}^d$ is a learnable or sinusoidal **timestep positional embedding** corresponding to episode step $t$. (Critically, all three tokens belonging to the same timestep $t$ share the exact same positional embedding $e_t^{\text{pos}}$).

The input token sequence fed into the Transformer is:
$$\mathbf{E} = \left[ e_1^R, e_1^s, e_1^a, e_2^R, e_2^s, e_2^a, \dots, e_K^R, e_K^s \right] \in \mathbb{R}^{N \times d}$$

---

### 2.3 Causal Masking & Supervised Training Objective

The Transformer blocks use standard **multi-head causal self-attention**:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

where $M$ is the upper-triangular causal mask ($M_{i, j} = -\infty$ for $j > i$).

Under this causal structure:
- Token $e_t^s$ can attend to all past tokens and $\hat{R}_t$, but **cannot see future tokens or its own action $a_t$**.
- Token $e_t^s$ is directly used to predict action $a_t$ via a linear action head $W_{\text{act}}$:
  $$\hat{a}_t = W_{\text{act}} h_t^s$$
  where $h_t^s \in \mathbb{R}^d$ is the Transformer output representation corresponding to token $e_t^s$.

#### The Training Objective:
The entire architecture is trained via standard supervised regression (MSE for continuous actions, Cross-Entropy for discrete actions):

$$\mathcal{L}_{\text{DT}}(\theta) = \frac{1}{T} \sum_{t=1}^T \left\| a_t - \hat{a}_t \left( \hat{R}_{1:t}, s_{1:t}, a_{1:t-1} \right) \right\|_2^2$$

Notice what is absent:
- **No Bellman equations.**
- **No value bootstrapping.**
- **No policy gradient estimators.**
- **No discount factor $\gamma$.**
Training is as stable and scalable as training a standard GPT language model!

---

### 2.4 Test-Time Autoregressive Rollout Protocol

To deploy a trained Decision Transformer in an environment:
1. **Prompt Initialization:** The user chooses a target return $\hat{R}_1$ (e.g. the maximum return in the training dataset, or an ambitious expert target).
2. **Observe Initial State:** Receive $s_1$ from the environment.
3. **Action Generation:** Pass $[\hat{R}_1, s_1]$ to the Transformer. Predict $\hat{a}_1$.
4. **Environment Execution:** Execute $\hat{a}_1$, observe environment reward $r_1$ and next state $s_2$.
5. **Return-to-Go Update:** Decrement the return-to-go:
   $$\hat{R}_2 \leftarrow \hat{R}_1 - r_1$$
6. **Context Extension:** Append $(\hat{a}_1, \hat{R}_2, s_2)$ to the context window and repeat until episode termination.

---

## 3. Geometric & Physical Interpretation

### 3.1 Partitioning the Trajectory Manifold by Return
Why does conditioning on Return-to-Go work better than naive Behavior Cloning?

```
Trajectory Space
 ^
 |    [ Low Return Manifold: R = 10 ]  ---> Sub-optimal random wander
 |
 |    [ Medium Return: R = 50 ]        ---> Mediocre heuristics
 |
 |    [ High Return Manifold: R = 100] ---> Direct, optimal shortest path!
 |___________________________________________________> Action Space
```
- **Behavior Cloning (BC):** Averages across all trajectories, learning a blurry, mediocre policy that collides with walls.
- **Decision Transformer (DT):** Learns the entire joint distribution $p(\tau)$. Conditioning on $\hat{R}_{\text{target}} = 100$ slices the manifold, conditioning the model *strictly* on the subspace of optimal expert transitions!

---

## 4. Real-World Analogy: GPS Navigation

Consider using a GPS navigation app on your phone:
- **Standard RL (Q-Learning):** At an intersection, the GPS evaluates: *"Turning left gives an expected commute time of 28 minutes; turning right gives 35 minutes."*
- **Decision Transformer:** You tell the GPS: *"I need to reach the airport in exactly 20 minutes ($\hat{R}_{\text{target}}$). I am currently at 5th Avenue ($s_1$). What turns are consistent with that outcome?"* The GPS outputs the high-speed highway route!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete forward pass of a **1-Layer, 1-Head Decision Transformer** by hand on a 2-token context $[\hat{R}_1, s_1]$ predicting continuous action $\hat{a}_1$.

---

### 5.1 System & Model Parameters
- **Hidden Embedding Dimension:** $d = 2$
- **Input Tokens ($T = 1$):**
  - Return-to-Go: $\hat{R}_1 = [10.0]$ (scalar)
  - State: $s_1 = [5.0]$ (scalar)
- **Linear Projection Weights:**
  $$W_R = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}, \quad W_s = \begin{bmatrix} 0.4 \\ -0.2 \end{bmatrix}$$
  *(Zero positional embedding for simplicity: $e_t^{\text{pos}} = [0, 0]^T$)*
- **Self-Attention Weights ($d = 2, d_k = 2$):**
  $$W_Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad W_K = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad W_V = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}$$
- **Linear Action Head:**
  $$W_{\text{act}} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\hat{R}_1, s_1$ | `rtg_1, state_1` | Raw input return-to-go and environment state scalars |
| $e_1^R, e_1^s$ | `embed_R, embed_s` | Projected token embedding vectors in $\mathbb{R}^2$ |
| $X$ | `input_seq` | Matrix of input tokens $\begin{bmatrix} (e_1^R)^T \\ (e_1^s)^T \end{bmatrix} \in \mathbb{R}^{2 \times 2}$ |
| $Q, K, V$ | `q_mat, k_mat, v_mat` | Query, Key, and Value matrices ($X W_Q, X W_K, X W_V$) |
| $z_{2, 1}, z_{2, 2}$ | `attn_logits` | Scaled dot-product attention logits for token 2 ($s_1$) |
| $w_{2, 1}, w_{2, 2}$ | `attn_weights` | Softmax normalized attention probabilities for token 2 |
| $Y_2$ | `attn_out_2` | Attention context output for state token $s_1$ |
| $\hat{a}_1$ | `pred_action` | Final continuous action prediction: $W_{\text{act}} Y_2$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Token Embeddings
$$e_1^R = W_R \hat{R}_1 = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} \times 10.0 = \begin{bmatrix} \mathbf{1.0} \\ \mathbf{2.0} \end{bmatrix}$$
$$e_1^s = W_s s_1 = \begin{bmatrix} 0.4 \\ -0.2 \end{bmatrix} \times 5.0 = \begin{bmatrix} \mathbf{2.0} \\ \mathbf{-1.0} \end{bmatrix}$$

Sequence matrix $X \in \mathbb{R}^{2 \times 2}$:
$$X = \begin{bmatrix} (e_1^R)^T \\ (e_1^s)^T \end{bmatrix} = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & -1.0 \end{bmatrix}$$

#### Step 2: Linear Projections for $Q, K, V$
Since $W_Q = I$ and $W_K = I$:
$$Q = X = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & -1.0 \end{bmatrix}, \quad K = X = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & -1.0 \end{bmatrix}$$
Since $W_V = 0.5 \cdot I$:
$$V = 0.5 \cdot X = \begin{bmatrix} 0.5 & 1.0 \\ 1.0 & -0.5 \end{bmatrix}$$

For predicting action $a_1$, we look at the row corresponding to state token $s_1$ (row index 2):
- Query vector: $q_2 = \begin{bmatrix} 2.0 & -1.0 \end{bmatrix}$
- Key 1 (Return token): $k_1 = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}$
- Key 2 (State token): $k_2 = \begin{bmatrix} 2.0 & -1.0 \end{bmatrix}$
- Value 1: $v_1 = \begin{bmatrix} 0.5 & 1.0 \end{bmatrix}$
- Value 2: $v_2 = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix}$

#### Step 3: Dot-Product Attention Logits for State Token $s_1$
1. **Dot product with Return Token $e_1^R$:**
   $$q_2 \cdot k_1 = (2.0 \times 1.0) + (-1.0 \times 2.0) = 2.0 - 2.0 = \mathbf{0.0000}$$
2. **Dot product with State Token $e_1^s$:**
   $$q_2 \cdot k_2 = (2.0 \times 2.0) + (-1.0 \times -1.0) = 4.0 + 1.0 = \mathbf{5.0000}$$

Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414214$:
$$z_{2, 1} = \frac{0.0000}{1.414214} = \mathbf{0.0000}$$
$$z_{2, 2} = \frac{5.0000}{1.414214} \approx \mathbf{3.535534}$$

#### Step 4: Softmax Attention Weights
$$e^{z_{2, 1}} = e^{0.0} = \mathbf{1.0000}$$
$$e^{z_{2, 2}} = e^{3.535534} \approx \mathbf{34.313410}$$
$$\sum = 1.0000 + 34.313410 = \mathbf{35.313410}$$

$$w_{2, 1} = \frac{1.0000}{35.313410} \approx \mathbf{0.028318}$$
$$w_{2, 2} = \frac{34.313410}{35.313410} \approx \mathbf{0.971682}$$

#### Step 5: Attention Output Context $Y_2$
$$Y_2 = w_{2, 1} v_1 + w_{2, 2} v_2$$
$$= 0.028318 \begin{bmatrix} 0.5 & 1.0 \end{bmatrix} + 0.971682 \begin{bmatrix} 1.0 & -0.5 \end{bmatrix}$$
$$= \begin{bmatrix} 0.014159 & 0.028318 \end{bmatrix} + \begin{bmatrix} 0.971682 & -0.485841 \end{bmatrix}$$
$$= \begin{bmatrix} \mathbf{0.985841} & \mathbf{-0.457523} \end{bmatrix}$$

#### Step 6: Action Prediction via Linear Head $W_{\text{act}}$
$$\hat{a}_1 = W_{\text{act}} Y_2^T = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 0.985841 \\ -0.457523 \end{bmatrix}$$
$$= 1.0 \times 0.985841 + 2.0 \times (-0.457523) = 0.985841 - 0.915046 = \mathbf{0.070795}$$

---

### 5.4 Summary Visual Grid: Decision Transformer Forward Ledger

| Step | Operation | Components | Arithmetic Formulation | Output Value |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $e_1^R$ Embedding | $W_R \cdot 10.0$ | $[0.1, 0.2]^T \times 10$ | $[1.0, 2.0]^T$ |
| **2** | $e_1^s$ Embedding | $W_s \cdot 5.0$ | $[0.4, -0.2]^T \times 5$ | $[2.0, -1.0]^T$ |
| **3** | Attention Logits | $q_2 \cdot K^T / \sqrt{2}$ | $[2.0, -1.0] \cdot K^T / 1.4142$ | $[0.0000, 3.5355]$ |
| **4** | Softmax Weights | $\operatorname{softmax}(\mathbf{z})$ | $\exp(\mathbf{z}) / \sum \exp(\mathbf{z})$ | $[0.0283, 0.9717]$ |
| **5** | Context Vector | $\sum w_i v_i$ | $0.0283 v_1 + 0.9717 v_2$ | $[0.9858, -0.4575]$ |
| **6** | Action Head | $W_{\text{act}} Y_2$ | $1.0(0.9858) + 2.0(-0.4575)$ | $\mathbf{\hat{a}_1 = 0.0708}$ |

---

## 6. Solved Illustrations

### Illustration 1: Prompting Beyond the Data Horizon (Return Extrapolation)
**Problem:**
Suppose the highest episode return in our offline training dataset is $R_{\max} = 100.0$.
What happens if at test time we prompt the Decision Transformer with an extreme target return:
$$\hat{R}_{\text{target}} = 500.0$$
**Solution:**
Unlike Bellman value functions which might extrapolate linearly, Transformers are pattern matchers. An input token of $500.0$ is Out-of-Distribution (OOD) for the embedding layer $W_R$. In empirical studies (Chen et al., 2021), performance typically **saturates or slightly degrades** when prompted beyond the dataset maximum: the model generates actions corresponding to the highest return it saw in training ($R \approx 100$), rather than magically performing $5\times$ better. $\blacksquare$

---

## 7. Deep RL Connection & Modern Applications

- **Robotics Foundation Models (RT-1, RT-2, Octo, OpenVLA):**
  Google DeepMind's Robotic Transformer (RT-1 / RT-2) treats robot manipulation as vision-language-action sequence modeling, directly predicting end-effector positions from visual tokens and task descriptions.
- **Multi-Game Decision Transformers (Lee et al., NeurIPS 2022):**
  A single 200M parameter Transformer trained across 41 Atari games simultaneously, outperforming human players without game-specific fine-tuning.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical reproduction of token embeddings, dot-product attention weights, context vector $Y_2$, and predicted action $\hat{a}_1 = 0.070795$ matching to $< 10^{-6}$.
2. **Complete PyTorch Decision Transformer Architecture:**
   - Modality embedding projections with positional encoding.
   - Causal multi-head self-attention blocks.
   - Autoregressive inference rollout loop with online return-to-go decrementing.

See implementation in:
[`11_reinforcement_learning/code/26_decision_transformer.py`](./code/26_decision_transformer.py)
