# Chapter 24: World Models & Dreamer (Latent Dynamics & Imagination)

---

## 1. Intuition & 101 Motivation

In traditional model-based reinforcement learning (such as Dyna-Q and standard MPC), the transition model operates directly in the raw observation space:
$$\hat{x}_{t+1} \sim p(x_{t+1} \mid x_t, a_t)$$

When observations are high-dimensional pixels (e.g., $64 \times 64 \times 3 = 12,288$ numbers per frame), predicting raw pixels is catastrophic:
1. **Computational Waste:** The neural network squanders 99% of its capacity predicting visual noise—background clouds drifting, flickering stadium lights, or leaves rustling—which has zero relevance to optimal decision-making.
2. **Compounding Blur:** Pixel-level autoregressive models blur rapidly over multi-step rollouts, turning imagined futures into gray soup within 10 steps.

Enter **World Models** (Ha & Schmidhuber, 2018) and the **Dreamer** suite (Hafner et al., DreamerV1 2020, DreamerV2 2021, DreamerV3 2023):
- Instead of modeling pixels, the agent compresses raw frames into a compact, low-dimensional **latent vector** $z_t \in \mathbb{R}^d$ via a variational autoencoder.
- It learns **latent dynamics**: how actions transform latent states $z_{t+1} = f(z_t, a_t)$.
- Crucially, it trains an Actor-Critic policy **entirely inside imagination** (hallucinated latent rollouts) by backpropagating analytic gradients directly through the world model's differentiable transition dynamics!

In 2023, **DreamerV3** achieved a historic milestone: it was the first AI system to collect diamonds from scratch in *Minecraft*—a task requiring thousands of sequential actions and tool-crafting steps—purely through latent imagination without any human demonstrations or domain tuning!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Recurrent State-Space Model (RSSM)

The backbone of Dreamer is the **Recurrent State-Space Model (RSSM)**. It decomposes the latent state at time $t$ into two components:
- **Deterministic Recurrent State $h_t \in \mathbb{R}^{d_h}$:** Maintained by a Recurrent Neural Network (GRU), providing long-term memory across thousands of timesteps.
- **Stochastic Latent State $z_t \in \mathbb{R}^{d_z}$:** Captures unpredictable stochasticity and instantaneous uncertainty.

The combined model state is $s_t \triangleq (h_t, z_t)$.

```
   Observation x_t -------> Encoder ------> q(z_t | h_t, x_t) [Posterior]
                                                |
                                                v (sample z_t)
   Action a_{t-1} ------> Recurrent State h_t ---+---> Decoders (Image x_t, Reward r_t, Discount gamma_t)
                              |
                              v
                        p(z_hat_t | h_t) [Imagination Prior]
```

#### The Six Core Sub-Models of RSSM:
1. **Recurrent Model (Deterministic Dynamics):**
   $$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$$
2. **Representation Model (Posterior):**
   $$q_\theta(z_t \mid h_t, x_t) \quad \text{(Incorporates true real-world observation } x_t \text{)}$$
3. **Transition Predictor (Prior / Imagination Dynamics):**
   $$p_\theta(\hat{z}_t \mid h_t) \quad \text{(Predicts next latent without seeing } x_t \text{)}$$
4. **Observation Decoder (Image Reconstruction):**
   $$p_\theta(x_t \mid h_t, z_t)$$
5. **Reward Predictor:**
   $$p_\theta(r_t \mid h_t, z_t)$$
6. **Continue / Discount Predictor:**
   $$p_\theta(\gamma_t \mid h_t, z_t) \quad \text{where } \gamma_t \in [0, 1] \text{ predicts episode termination } (1 - d_t)$$

---

### 2.2 The Variational ELBO Objective

The world model parameters $\theta$ are trained end-to-end to maximize the Evidence Lower Bound (ELBO) over trajectories $(x_1, a_1, r_1, \dots, x_T)$:

$$\mathcal{L}_{\text{world}}(\theta) = \sum_{t=1}^T \mathbb{E}_{q_\theta(z_t \mid h_t, x_t)} \Big[ \underbrace{\ln p_\theta(x_t \mid h_t, z_t)}_{\text{Image Reconstruction}} + \underbrace{\ln p_\theta(r_t \mid h_t, z_t)}_{\text{Reward Prediction}} + \underbrace{\ln p_\theta(\gamma_t \mid h_t, z_t)}_{\text{Discount Prediction}} - \beta \underbrace{D_{\text{KL}}\left( q_\theta(z_t \mid h_t, x_t) \;\Vert\; p_\theta(\hat{z}_t \mid h_t) \right)}_{\text{KL Information Regularizer}} \Big]$$

#### Analytical Gaussian KL Divergence:
When both prior $p(z) = \mathcal{N}(\mu_p, \sigma_p^2)$ and posterior $q(z) = \mathcal{N}(\mu_q, \sigma_q^2)$ are diagonal Gaussians, the KL divergence is computable in closed form:

$$D_{\text{KL}}(q \;\Vert\; p) = \sum_{j=1}^{d_z} \left[ \ln\frac{\sigma_{p, j}}{\sigma_{q, j}} + \frac{\sigma_{q, j}^2 + (\mu_{q, j} - \mu_{p, j})^2}{2 \sigma_{p, j}^2} - \frac{1}{2} \right]$$

---

### 2.3 KL Balancing (DreamerV2 & DreamerV3)

If optimized naively, the KL term exhibits an asymmetry: the representation model $q$ can collapse toward the prior $p$ before the prior learns meaningful dynamics.
To fix this, Dreamer introduces **KL Balancing** with asymmetric gradient stops:

$$\mathcal{L}_{\text{KL}}(\theta) = \alpha D_{\text{KL}}\left( \operatorname{sg}[q_\theta] \;\Vert\; p_\theta \right) + (1 - \alpha) D_{\text{KL}}\left( q_\theta \;\Vert\; \operatorname{sg}[p_\theta] \right)$$

where $\operatorname{sg}[\cdot]$ is the `stop_gradient` operator, and $\alpha = 0.8$.
This forces the prior $p$ to chase the representation $q$ four times faster than $q$ is penalized for escaping $p$!

---

### 2.4 Learning Behaviors in Imagination

Once the world model is trained, the agent freezes the real world and trains an **Actor $\pi_\phi(a \mid s_t)$** and **Critic $v_\psi(s_t)$** purely within **dream trajectories** of horizon $H = 15$:

1. **Seed Initial State:** Sample $s_\tau = (h_\tau, z_\tau)$ from the replay buffer.
2. **Roll Out in Imagination:** For $t = \tau, \dots, \tau + H - 1$:
   $$a_t \sim \pi_\phi(a_t \mid s_t)$$
   $$h_{t+1} = f_\theta(h_t, z_t, a_t)$$
   $$z_{t+1} \sim p_\theta(z_{t+1} \mid h_{t+1}) \quad \text{(Imagined prior dynamics)}$$
   $$\hat{r}_t = \mathbb{E}[p_\theta(r \mid s_t)], \quad \hat{\gamma}_t = \mathbb{E}[p_\theta(\gamma \mid s_t)]$$
3. **Compute Generalized $\lambda$-Returns:**
   $$V_t^\lambda = \hat{r}_t + \hat{\gamma}_t \left( (1 - \lambda) v_\psi(s_{t+1}) + \lambda V_{t+1}^\lambda \right)$$
   with boundary condition $V_{\tau+H}^\lambda = v_\psi(s_{\tau+H})$.
4. **Update Critic:**
   $$\mathcal{L}(\psi) = \frac{1}{2} \sum_{t=\tau}^{\tau+H-1} \left( v_\psi(s_t) - \operatorname{sg}[V_t^\lambda] \right)^2$$
5. **Update Actor via Analytic Backpropagation:**
   Because the RSSM transition $h_{t+1} = f(h_t, z_t, a_t)$ is fully differentiable, gradients can flow directly from the value function $V_t^\lambda$ back through the world model dynamics into the policy parameters $\phi$:
   $$\max_\phi \sum_{t=\tau}^{\tau+H-1} V_t^\lambda + \eta \mathcal{H}(\pi_\phi(\cdot \mid s_t))$$

---

## 3. Geometric & Physical Interpretation

### 3.1 Latent Space Manifold Projection
A $64 \times 64 \times 3$ pixel frame lives in $\mathbb{R}^{12,288}$. However, the true physical degrees of freedom (e.g., robot joint angles, cart position) occupy a smooth, compact submanifold of dimension $d \le 30$.

```
High-Dimensional Observation Space (R^12288)
         \      /
          \    /  <-- Variational Encoder (Lossy Compression)
           \  /
            \/
Low-Dimensional Latent Manifold (R^32)
   *-------*-------*-------*   <-- Locally smooth, linearizable physics!
  z_0     z_1     z_2     z_3
```
In latent space, physics is smooth, predictable, and differentiable, enabling gradient-based trajectory optimization that would be impossible over raw pixels.

---

## 4. Real-World Analogy: The Flight Simulator in Your Mind

Imagine an elite stunt pilot preparing for a dangerous aerial maneuver:
- They sit in a chair with their eyes closed.
- In their mind, they execute the sequence: *"Pull stick back 3 inches, kick right rudder, tap throttle."*
- Their mental world model predicts the resulting sensation: *"G-force surges to 4G, horizon flips upside down, stall alarm sounds."*
- If the mental simulation ends in a crash, the pilot adjusts their imagined stick pressure and tries again—**all without burning a single drop of jet fuel or risking their life**.
- That is Dreamer: millions of simulated flights inside a neural world model!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace the exact numerical calculations of:
1. **The Variational Gaussian KL Divergence** between Posterior and Prior.
2. **The 2-Step Imagined $\lambda$-Return Calculation**.

---

### 5.1 System & Model Parameters

#### Component 1: Gaussian KL Calculation (1D Scalar Latent)
- **Prior Distribution:** $p(z) = \mathcal{N}(\mu_p, \sigma_p^2)$
  - $\mu_p = 0.0000$
  - $\sigma_p = 1.0000 \implies \sigma_p^2 = 1.0000$
- **Posterior Distribution:** $q(z) = \mathcal{N}(\mu_q, \sigma_q^2)$
  - $\mu_q = 0.8000$
  - $\sigma_q = 0.5000 \implies \sigma_q^2 = 0.2500$

#### Component 2: Imagined 2-Step $\lambda$-Return Rollout
- **Planning Horizon:** $H = 2$ steps ($t = 1, 2$)
- **Discount Factor:** $\gamma = 0.9000$
- **Bootstrapping Factor:** $\lambda = 0.8000$
- **Imagined Rewards:**
  - $\hat{r}_1 = 2.0000$
  - $\hat{r}_2 = 5.0000$
- **Critic Value Predictions:**
  - $v(s_1) = 4.0000$
  - $v(s_2) = 8.0000$
  - $v(s_3) = 10.0000$ (Terminal bootstrap)

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\mu_p, \sigma_p$ | `prior_mu, prior_std` | Mean and standard deviation of imagination prior $p(z \mid h)$ |
| $\mu_q, \sigma_q$ | `post_mu, post_std` | Mean and standard deviation of sensory posterior $q(z \mid h, x)$ |
| $D_{\text{KL}}(q \parallel p)$ | `kl_div` | Information divergence penalizing model hallucination |
| $\hat{r}_t$ | `reward_hat[t]` | Predicted stage reward emitted by latent state $s_t$ |
| $v(s_t)$ | `critic_val[t]` | Critic network baseline value estimate at state $s_t$ |
| $V_t^\lambda$ | `lambda_return[t]` | Recursive multi-step target return in imagination |

---

### 5.3 Step-by-Step Hand Calculations: Gaussian KL Divergence

Recall the exact formula:
$$D_{\text{KL}}(q \;\Vert\; p) = \ln\left(\frac{\sigma_p}{\sigma_q}\right) + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2 \sigma_p^2} - \frac{1}{2}$$

1. **Log-Ratio of Standard Deviations:**
   $$\frac{\sigma_p}{\sigma_q} = \frac{1.0000}{0.5000} = 2.0000$$
   $$\ln(2.0000) \approx \mathbf{0.693147}$$

2. **Variance and Mean Difference Term:**
   $$\sigma_q^2 = (0.5000)^2 = 0.2500$$
   $$(\mu_q - \mu_p)^2 = (0.8000 - 0.0000)^2 = 0.6400$$
   $$\sigma_q^2 + (\mu_q - \mu_p)^2 = 0.2500 + 0.6400 = 0.8900$$
   $$\frac{0.8900}{2 \times \sigma_p^2} = \frac{0.8900}{2.0000} = \mathbf{0.445000}$$

3. **Total KL Divergence:**
   $$D_{\text{KL}}(q \;\Vert\; p) = 0.693147 + 0.445000 - 0.500000 = \mathbf{0.638147}$$

---

### 5.4 Step-by-Step Hand Calculations: 2-Step Imagined $\lambda$-Return

The recursive formula is evaluated backward from terminal step $H = 2$ to $t = 1$:

$$V_H^\lambda = v(s_{H+1}) = v(s_3) = \mathbf{10.0000}$$

#### Timestep $t = 2$:
At the penultimate step:
$$V_2^\lambda = \hat{r}_2 + \gamma \cdot v(s_3) = 5.0000 + 0.9000 \times 10.0000 = 5.0000 + 9.0000 = \mathbf{14.0000}$$

#### Timestep $t = 1$:
Now apply the general recursive $\lambda$-combination:
$$V_1^\lambda = \hat{r}_1 + \gamma \left[ (1 - \lambda) v(s_2) + \lambda V_2^\lambda \right]$$
1. **Bracketed Next-State Value Combination:**
   $$(1 - \lambda) v(s_2) = (1.0 - 0.80) \times 8.0000 = 0.20 \times 8.0000 = \mathbf{1.6000}$$
   $$\lambda V_2^\lambda = 0.80 \times 14.0000 = \mathbf{11.2000}$$
   $$\text{Combined Target} = 1.6000 + 11.2000 = \mathbf{12.8000}$$

2. **Discount and Stage Reward Addition:**
   $$V_1^\lambda = \hat{r}_1 + \gamma \times 12.8000 = 2.0000 + 0.9000 \times 12.8000 = 2.0000 + 11.5200 = \mathbf{13.5200}$$

---

### 5.5 Summary Visual Grid: Latent World Model & Imagination Ledger

| Variable / Step | Prior Distribution $p$ | Posterior Distribution $q$ | Component Values | Step Result |
| :--- | :---: | :---: | :---: | :---: |
| **Log Ratio** | $\sigma_p = 1.0$ | $\sigma_q = 0.5$ | $\ln(1.0 / 0.5) = \ln(2)$ | **$+0.6931$** |
| **Quad Penalty** | $\mu_p = 0.0$ | $\mu_q = 0.8$ | $(0.25 + 0.64) / 2$ | **$+0.4450$** |
| **Offset** | — | — | Constant $-0.5$ | **$-0.5000$** |
| **Total KL** | — | — | $0.6931 + 0.4450 - 0.5$ | $\mathbf{D_{\text{KL}} = 0.6381}$ |
| **$V_2^\lambda$ Return** | — | — | $5.0 + 0.9 \times 10.0$ | $\mathbf{V_2^\lambda = 14.0000}$ |
| **$V_1^\lambda$ Return** | — | — | $2.0 + 0.9 \times [0.2(8) + 0.8(14)]$ | $\mathbf{V_1^\lambda = 13.5200}$ |

---

## 6. Solved Illustrations

### Illustration 1: Why DreamerV2 & V3 Use Discrete Categorical Latents
**Problem:**
Why did DreamerV2 and V3 replace continuous Gaussian latents $\mathcal{N}(\mu, \sigma^2)$ with a matrix of categorical variables ($32$ categorical distributions with $32$ classes each)?
**Solution:**
1. **Non-Gaussian Multimodality:** Physical transitions in games (e.g. stepping left or right around a wall) are discrete and multi-modal. A unimodal Gaussian is forced to average across modes, generating blurry, non-physical states in the middle of walls.
2. **Vanishing Gradient Prevention:** High-entropy Gaussians suffer from posterior collapse. Categorical distributions with straight-through Gumbel-Softmax or argmax sampling prevent mode collapse and preserve distinct discrete features over long rollouts. $\blacksquare$

---

## 7. Deep RL Connection & Modern Applications

- **DreamerV3 (Hafner et al., Nature 2023):** Mastered Atari 100k, DeepMind Control Suite, DMLab, and Minecraft without modifying a single hyperparameter across domains.
- **Autonomous Driving Video World Models (Wayve GAIA-1, Tesla FSD):**
  Train multi-billion parameter autoregressive video diffusion models to predict realistic 3D driving environments, enabling self-driving policies to practice millions of rare edge cases (pedestrian jaywalking, vehicle cut-ins) entirely in generative simulation.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Closed-form Gaussian KL divergence verification matching $0.638147$ to $< 10^{-6}$.
   - Backward recursive $\lambda$-return calculation matching $V_2^\lambda = 14.0000, V_1^\lambda = 13.5200$ to $< 10^{-12}$.
2. **Complete PyTorch-Vectorized RSSM Latent Dynamics Layer:**
   - Deterministic GRU recurrent transition.
   - Stochastic Gaussian posterior and prior with reparameterization trick.
   - Vectorized KL divergence computation with KL balancing ($\alpha = 0.8$).
3. **Actor-Critic Imagination Rollout Pipeline:**
   - Generates $H$-step imagined latent rollouts and computes analytical $\lambda$-returns.

See implementation in:
[`11_reinforcement_learning/code/24_world_models_and_dreamer.py`](./code/24_world_models_and_dreamer.py)
