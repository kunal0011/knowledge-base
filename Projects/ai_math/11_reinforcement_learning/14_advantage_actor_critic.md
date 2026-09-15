# Module 11.14: Advantage Actor-Critic (A2C & A3C)

---

## 1. Intuition & 101 Motivation

In Chapter 11.13, we derived the REINFORCE algorithm, which optimizes policies directly using sample trajectories. While REINFORCE is mathematically elegant and strictly unbiased, it suffers from a fatal flaw: **high variance**. Because it uses the full Monte Carlo return:
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$
the variance scales quadratically with the horizon $T$, requiring millions of rollouts to find clear gradient signals. Furthermore, REINFORCE cannot learn online or mid-episode; it must wait for the entire episode to terminate.

**Actor-Critic methods** eliminate this episodic restriction by replacing full Monte Carlo rollouts with **1-step bootstrapping**:
- **The Actor:** The parameterized policy $\pi_{\boldsymbol{\theta}}(a \mid s)$, which explores the environment and selects actions.
- **The Critic:** A parameterized state-value function $V_{\boldsymbol{\phi}}(s)$, which evaluates the quality of states by computing the 1-step Temporal-Difference (TD) error:
  $$\delta_t = R_{t+1} + \gamma V_{\boldsymbol{\phi}}(S_{t+1}) - V_{\boldsymbol{\phi}}(S_t)$$

The TD error $\delta_t$ serves as an immediate, low-variance estimate of the **Advantage function** $A(S_t, A_t)$. The Actor updates its policy weights at *every single time step* without waiting for the episode to end!

```
+-----------------------------------------------------------------------------------------+
|                               ACTOR-CRITIC ARCHITECTURE                                 |
|                                                                                         |
|                            +------------------------------+                             |
|                            |         Environment          |                             |
|                            +------------------------------+                             |
|                               ^ State S_t           | Reward R_{t+1}, Next State S_{t+1}|
|                               |                     v                                   |
|                  +------------+---------------------+------------+                      |
|                  |                                               |                      |
|                  v                                               v                      |
|       +---------------------+                         +---------------------+           |
|       |    ACTOR NETWORK    |                         |    CRITIC NETWORK   |           |
|       | pi_theta(a | S_t)   |                         |     V_phi(S_t)      |           |
|       +---------------------+                         +---------------------+           |
|                  | Action A_t                                    |                      |
|                  |                                               v                      |
|                  |                                    TD Error / Advantage:             |
|                  |                                    delta = R + gamma V' - V          |
|                  |                                               |                      |
|                  +<=========== Policy Gradient Update <==========+                      |
|                                grad = delta * grad log pi                               |
+-----------------------------------------------------------------------------------------+
```

### A3C vs. A2C: Asynchronous vs. Synchronous
- **A3C (Asynchronous Advantage Actor-Critic - Mnih et al., ICML 2016):** Multiple CPU worker threads interact with separate environment instances in parallel, asynchronously pushing gradient updates to a central parameter server without locks.
- **A2C (Advantage Actor-Critic):** A synchronous, deterministic alternative that waits for all parallel workers to complete their segments, batches their transitions into a single GPU tensor, and performs a single synchronized forward/backward pass. A2C matches or exceeds A3C's sample efficiency while fully utilizing modern GPU vectorization.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Policy Gradient with Advantage Function

Recall the general policy gradient theorem:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(A_t \mid S_t) \Psi_t \right]$$

Common choices for $\Psi_t$:
1. Trajectory Return: $\Psi_t = R(\tau)$ (high variance, unbiased)
2. Reward-to-Go: $\Psi_t = \sum_{k=t}^T \gamma^{k-t} R_{k+1}$ (lower variance, unbiased)
3. Action-Value Function: $\Psi_t = Q^{\pi}(S_t, A_t)$ (requires learning $Q$)
4. Advantage Function: $\Psi_t = A^{\pi}(S_t, A_t) \triangleq Q^\pi(S_t, A_t) - V^\pi(S_t)$ (**lowest variance, optimal baseline**)
5. 1-Step TD Error: $\Psi_t = \delta_t \triangleq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

#### Theorem: The TD Error is an Unbiased Advantage Estimator
If the critic $V$ equals the true state-value function $V^\pi$, the conditional expectation of the 1-step TD error $\delta_t$ given $(S_t = s, A_t = a)$ is **identically equal to the true Advantage** $A^\pi(s, a)$:

$$\mathbb{E}_{\mathcal{P}} \left[ \delta_t \mid S_t = s, A_t = a \right] = A^\pi(s, a)$$

#### Proof:
$$\mathbb{E} \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) - V^\pi(S_t) \mid S_t = s, A_t = a \right]$$
$$= \underbrace{\mathbb{E} \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a \right]}_{Q^\pi(s, a)} - V^\pi(s)$$
$$= Q^\pi(s, a) - V^\pi(s) = A^\pi(s, a) \quad \blacksquare$$

---

### 2.2 The Multi-Task Objective Function

In practice, the Actor and Critic share feature representations (e.g., convolutional or MLP hidden layers). The unified training loss combines three distinct objectives:

$$\mathcal{L}_{\text{total}}(\boldsymbol{\theta}, \boldsymbol{\phi}) \triangleq \mathcal{L}_{\text{policy}}(\boldsymbol{\theta}) + c_1 \mathcal{L}_{\text{value}}(\boldsymbol{\phi}) - c_2 \mathcal{H}(\pi_{\boldsymbol{\theta}}(\cdot \mid S_t))$$

where:
1. **Policy Loss (Actor):**
   $$\mathcal{L}_{\text{policy}}(\boldsymbol{\theta}) = - \log \pi_{\boldsymbol{\theta}}(A_t \mid S_t) \cdot \delta_t$$
   (The negative sign converts gradient ascent on expected return into gradient descent). Note: $\delta_t$ is detached from autograd when optimizing the Actor!
2. **Value Loss (Critic):**
   $$\mathcal{L}_{\text{value}}(\boldsymbol{\phi}) = \frac{1}{2} \delta_t^2 = \frac{1}{2} \left( R_{t+1} + \gamma V_{\boldsymbol{\phi}}(S_{t+1}) - V_{\boldsymbol{\phi}}(S_t) \right)^2$$
3. **Entropy Regularization ($\mathcal{H}$):**
   $$\mathcal{H}(\pi_{\boldsymbol{\theta}}(\cdot \mid S_t)) \triangleq - \sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid S_t) \log \pi_{\boldsymbol{\theta}}(a \mid S_t)$$
   Maximizing entropy (subtracting $c_2 \mathcal{H}$ from the loss) prevents the policy from prematurely collapsing into a deterministic suboptimal action, encouraging persistent exploration.
4. $c_1 \approx 0.5$ and $c_2 \approx 0.01$ are positive scaling coefficients.

---

### 2.3 Synchronous Batched Multi-Worker Execution (A2C)

Let $K$ be the number of parallel environment runners. At each rollout step:
1. Each environment $k \in \{1, \dots, K\}$ holds state $s_t^{(k)}$.
2. A single batched forward pass computes:
   $$\boldsymbol{\pi}_t = \text{Softmax}(\text{Actor}(\mathbf{S}_t)), \quad \mathbf{V}_t = \text{Critic}(\mathbf{S}_t)$$
3. Sample actions $a_t^{(k)} \sim \pi_t^{(k)}$ and step all $K$ environments synchronously in parallel, obtaining rewards $\mathbf{R}_{t+1}$ and next states $\mathbf{S}_{t+1}$.
4. Compute batched TD errors across all $K$ workers:
   $$\boldsymbol{\delta}_t = \mathbf{R}_{t+1} + \gamma (1 - \mathbf{d}) \mathbf{V}(S_{t+1}) - \mathbf{V}_t$$
5. Perform a single batched backpropagation update on the unified loss $\mathcal{L}_{\text{total}}$.

---

## 3. Geometric & Physical Interpretation: The Actor-Critic Landscape

Think of reinforcement learning as navigating a mountainous terrain at night:
- **REINFORCE:** You take 1,000 steps until you fall off a cliff or find a goldmine, then hike all the way back to the starting point to evaluate whether your first step was good.
- **Actor-Critic:**
  - The **Critic** acts as an altimeter: it maps out the altitude surface $V(s)$ in real time.
  - The **Actor** reads the local slope (gradient) relative to the altimeter reading: if taking action $A_t$ places you at a higher elevation than the Critic predicted ($\delta_t > 0$), you step in that direction immediately.

```
       Elevation V(s)
            ^                                     Critic's baseline surface
            |                                         /~~~~~\
            |                       * Actual R + gamma V' (Higher than expected!)
            |                      /
            |                     * Advantage delta > 0 (Actor encouraged)
            |                    /
            |            *------* Predicted V(s)
            +----------------------------------------------------> States
```

---

## 4. Real-World Analogy: The Jazz Improviser and Rhythm Section

Imagine a jazz trumpet soloist (The Actor) performing live with an experienced bassist and drummer (The Critic):
- The trumpeter plays spontaneous, exploratory musical lines $\pi_{\boldsymbol{\theta}}(a \mid s)$.
- The rhythm section maintains the harmonic groove and tempo, establishing the baseline $V_{\boldsymbol{\phi}}(s)$.
- When the trumpeter hits a daring note, the rhythm section immediately signals whether it resolved into harmonic brilliance ($\delta_t > 0$) or clashed discordantly ($\delta_t < 0$).
- The trumpeter doesn't wait until the 3-hour concert concludes to know if that note worked; the instantaneous harmonic feedback allows real-time musical adaptation mid-measure!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: Single Shared-Trunk Network
Let us trace an exact, cell-by-cell numerical forward and backward pass of an Actor-Critic step!

**Environment Transition:**
- State: $s = [1.0000, 0.0000]^\top$
- Chosen action: $a = 0$ ($a_0$)
- Observed reward: $r = 2.0000$
- Successor state: $s' = [0.0000, 1.0000]^\top$
- Done: $d = 0$ (non-terminal)
- Hyperparameters: $\gamma = 0.9000$, learning rate $\alpha = 0.1000$, $c_1 = 0.5000, c_2 = 0.0100$.

**Network Architecture:**
- **Shared Input Layer:** Identity trunk (inputs passed directly to heads: $\mathbf{h} = s$).
- **Actor Head (Policy):** Linear layer producing 2 logits for actions $\{a_0, a_1\}$:
  $$\mathbf{W}_{\pi} = \begin{bmatrix} 1.0000 & 0.0000 \\ 0.0000 & 1.0000 \end{bmatrix}, \quad \mathbf{b}_{\pi} = \begin{bmatrix} 0.0000 \\ 0.0000 \end{bmatrix}$$
- **Critic Head (Value):** Linear layer producing a single scalar state value:
  $$\mathbf{w}_V = [1.0000, 2.0000]^\top, \quad b_V = 0.0000$$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Value / Matrix |
| :--- | :--- | :--- |
| $s, s'$ | Current and Next State Vectors | $s = [1, 0]^\top, s' = [0, 1]^\top$ |
| $V(s), V(s')$ | Critic State Value Outputs | $\mathbf{w}_V^\top s + b_V$ |
| $\delta$ | TD Error / Advantage Estimate | $r + \gamma V(s') - V(s)$ |
| $\mathbf{z}$ | Actor Logits Vector | $\mathbf{W}_\pi s + \mathbf{b}_\pi$ |
| $\boldsymbol{\pi}$ | Action Probability Vector | $\text{Softmax}(\mathbf{z})$ |
| $\mathcal{H}$ | Policy Shannon Entropy | $-\sum \pi_i \ln(\pi_i)$ |
| $\mathcal{L}_{\text{policy}}$ | Actor Loss | $-\ln(\pi(a_0)) \cdot \delta$ |
| $\mathcal{L}_{\text{value}}$ | Critic Loss | $\frac{1}{2} \delta^2$ |
| $\mathcal{L}_{\text{total}}$ | Total Combined Loss | $\mathcal{L}_{\text{policy}} + 0.5 \mathcal{L}_{\text{value}} - 0.01 \mathcal{H}$ |

---

### 5.3 Step 1: Forward Pass on Critic (Compute TD Error $\delta$)

Compute current state value:
$$V(s) = \mathbf{w}_V^\top s + b_V = (1.0000 \times 1.0 + 2.0000 \times 0.0) + 0.0 = \mathbf{1.0000}$$

Compute successor state value:
$$V(s') = \mathbf{w}_V^\top s' + b_V = (1.0000 \times 0.0 + 2.0000 \times 1.0) + 0.0 = \mathbf{2.0000}$$

Compute 1-step TD Error (Advantage):
$$\delta = r + \gamma V(s') - V(s) = 2.0000 + 0.9000 \times 2.0000 - 1.0000$$
$$= 2.0000 + 1.8000 - 1.0000 = \mathbf{2.8000}$$

---

### 5.4 Step 2: Forward Pass on Actor (Compute Probabilities & Entropy)

Actor logits for state $s = [1.0, 0.0]^\top$:
$$\mathbf{z} = \mathbf{W}_\pi s + \mathbf{b}_\pi = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 1.0000 \\ 0.0000 \end{bmatrix}$$

Exponentiate logits:
$$e^{z_0} = e^{1.0000} \approx 2.71828, \quad e^{z_1} = e^{0.0000} = 1.00000$$
Sum of exponentials:
$$Z = 2.71828 + 1.00000 = 3.71828$$

Softmax probabilities:
$$\pi(a_0) = \frac{2.71828}{3.71828} \approx \mathbf{0.73106}$$
$$\pi(a_1) = \frac{1.00000}{3.71828} \approx \mathbf{0.26894}$$

Shannon Entropy $\mathcal{H}$:
$$\mathcal{H} = - \left[ \pi(a_0) \ln(\pi(a_0)) + \pi(a_1) \ln(\pi(a_1)) \right]$$
$$\ln(\pi(a_0)) = \ln(0.73106) \approx -0.31326$$
$$\ln(\pi(a_1)) = \ln(0.26894) \approx -1.31326$$
$$\mathcal{H} = - \left[ 0.73106 \times (-0.31326) + 0.26894 \times (-1.31326) \right]$$
$$= - \left[ -0.22901 - 0.35319 \right] = -[-0.58220] \approx \mathbf{0.58220}$$

---

### 5.5 Step 3: Compute Losses

1. **Policy Loss (Actor):**
   $$\mathcal{L}_{\text{policy}} = - \ln(\pi(a_0)) \cdot \delta = - (-0.31326) \times 2.8000 \approx \mathbf{+0.87713}$$
2. **Value Loss (Critic):**
   $$\mathcal{L}_{\text{value}} = \frac{1}{2} \delta^2 = \frac{1}{2} (2.8000)^2 = \frac{1}{2} (7.8400) = \mathbf{3.9200}$$
3. **Entropy Bonus:**
   $$- c_2 \mathcal{H} = - 0.0100 \times 0.58220 \approx \mathbf{-0.00582}$$

**Total Combined Loss:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}} - c_2 \mathcal{H}$$
$$= 0.87713 + 0.5000 \times 3.9200 - 0.00582 = 0.87713 + 1.9600 - 0.00582 = \mathbf{2.83131}$$

---

### 5.6 Step 4: Backward Pass (Gradient Derivations)

#### 1. Critic Weights Gradient ($\mathbf{w}_V$):
$$\frac{\partial \mathcal{L}_{\text{value}}}{\partial V(s)} = - \delta = - 2.8000$$
Since $V(s) = \mathbf{w}_V^\top s$, the gradient scaled by $c_1 = 0.5$:
$$\nabla_{\mathbf{w}_V} (c_1 \mathcal{L}_{\text{value}}) = c_1 \frac{\partial \mathcal{L}_{\text{value}}}{\partial V(s)} s = 0.5 \times (-2.8000) \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.4000} \\ \mathbf{0.0000} \end{bmatrix}$$

Updated Critic Weights ($\alpha = 0.1000$):
$$\mathbf{w}_{V, \text{new}} = \mathbf{w}_V - \alpha \nabla_{\mathbf{w}_V} = \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix} - 0.1000 \begin{bmatrix} -1.4000 \\ 0.0000 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.1400 \\ 2.0000 \end{bmatrix}}$$

#### 2. Actor Logits Gradient ($\mathbf{z}$):
From the policy loss $\mathcal{L}_{\text{policy}} = - \delta \ln \pi(a_0)$:
$$\frac{\partial \mathcal{L}_{\text{policy}}}{\partial z_k} = \delta \left( \pi(a_k) - \mathbb{I}(k = 0) \right)$$
- For $k = 0$: $2.8000 \times (0.73106 - 1.0) = 2.8000 \times (-0.26894) \approx \mathbf{-0.75303}$
- For $k = 1$: $2.8000 \times (0.26894 - 0.0) = 2.8000 \times (+0.26894) \approx \mathbf{+0.75303}$

Gradient with respect to Actor weight row 0 ($\mathbf{W}_{\pi, [0, :]}$):
$$\nabla_{\mathbf{W}_{\pi, [0, :]}} \mathcal{L}_{\text{policy}} = (-0.75303) s^\top = [-0.75303, \quad 0.0000]$$

Updated Actor Weight $W_{\pi, 00}$ ($\alpha = 0.1000$):
$$W_{\pi, 00, \text{new}} = 1.0000 - 0.1000 \times (-0.75303) = 1.0000 + 0.07530 = \mathbf{1.07530}$$
The logit for chosen action $a_0$ is reinforced from $1.0000$ to $1.0753$!

---

### 5.7 Visual Summary Tensor Grid

| Component | Value / Variable | Output / Loss | Gradient Component | Updated Parameter |
| :---: | :---: | :---: | :---: | :---: |
| **Critic $V(s)$** | $s = [1, 0]^\top$ | $V(s) = 1.0000$ | - | - |
| **Critic $V(s')$** | $s' = [0, 1]^\top$ | $V(s') = 2.0000$ | - | - |
| **TD Error $\delta$** | $2.0 + 0.9(2) - 1$ | $\mathbf{+2.8000}$ | - | - |
| **Critic Loss** | $0.5 \times 2.8^2$ | $3.9200$ | $\nabla_{w_{V, 0}} = -1.4000$ | $w_{V, 0}: 1.0 \to \mathbf{1.1400}$ |
| **Actor Probs** | Softmax($[1, 0]^\top$) | $\pi = [0.7311, 0.2689]$ | - | - |
| **Policy Loss** | $- \ln(0.7311) \times 2.8$ | $0.8771$ | $\nabla_{W_{\pi, 00}} = -0.7530$ | $W_{\pi, 00}: 1.0 \to \mathbf{1.0753}$ |
| **Entropy $\mathcal{H}$** | $-\sum \pi \ln \pi$ | $0.5822$ | Regularizer | Exploration Preserved |

---

## 6. Solved Illustrations

### Illustration 1: Why Bootstrapping Introduces Bias
**Problem:**
REINFORCE uses the true empirical return $G_t$, which has zero bias: $\mathbb{E}[G_t \mid S_t, A_t] = Q^\pi(S_t, A_t)$.
Actor-Critic replaces $G_t$ with $R_{t+1} + \gamma V_{\boldsymbol{\phi}}(S_{t+1})$.
Prove that if the critic is imperfect ($V_{\boldsymbol{\phi}} \neq V^\pi$), the policy gradient update becomes **biased**.

**Solution:**
Let $\epsilon(s) \triangleq V_{\boldsymbol{\phi}}(s) - V^\pi(s)$ be the critic's approximation error.
The expected TD error target is:
$$\mathbb{E} \left[ R_{t+1} + \gamma V_{\boldsymbol{\phi}}(S_{t+1}) \mid S_t = s, A_t = a \right]$$
$$= \mathbb{E} \left[ R_{t+1} + \gamma \left( V^\pi(S_{t+1}) + \epsilon(S_{t+1}) \right) \mid S_t = s, A_t = a \right]$$
$$= Q^\pi(s, a) + \gamma \mathbb{E} \left[ \epsilon(S_{t+1}) \mid S_t = s, A_t = a \right]$$
The policy gradient evaluated using this critic is:
$$\mathbf{g}_{\text{AC}} = \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) + \gamma \mathbb{E} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \mathbb{E}[\epsilon(S') \mid s, a] \right]$$
Unless $\mathbb{E}[\epsilon(S')]$ is independent of $a$ (or $\epsilon \equiv 0$), this extra term does not vanish.
**Conclusion:** Actor-Critic accepts non-zero initial bias in exchange for a massive reduction in variance ($\mathcal{O}(1)$ vs $\mathcal{O}(T)$), which makes neural network training practical! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **OpenAI Five & AlphaStar:** Large-scale distributed A2C/A3C variants trained multi-agent neural networks to defeat world-champion human esports teams in Dota 2 and StarCraft II.
- **LLM Token-Level Critics:** In reasoning models, token-level value critics provide intermediate credit assignment along mathematical deduction steps, pinpointing the exact reasoning step where a derivation went off track.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of the Part 5 hand calculations:
   - TD error $\delta = 2.8000$, Critic loss $3.9200$, Policy loss $0.8771$, Entropy $0.5822$
   - Parameter updates: $w_{V, 0} \to 1.1400$ and $W_{\pi, 00} \to 1.0753$ matching PyTorch autograd to $< 10^{-14}$.
2. A complete batched Advantage Actor-Critic (A2C) agent with parallel vector environments solving CartPole-v1.

See implementation in:
[`11_reinforcement_learning/code/14_advantage_actor_critic.py`](./code/14_advantage_actor_critic.py)
