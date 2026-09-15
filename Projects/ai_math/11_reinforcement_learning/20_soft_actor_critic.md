# Module 11.20: Soft Actor-Critic (SAC)

---

## 1. Intuition & 101 Motivation

Standard reinforcement learning algorithms seek a policy that maximizes expected cumulative discounted rewards:
$$\pi^* = \arg\max_\pi \sum_{t=0}^\infty \mathbb{E}_{(S_t, A_t) \sim \pi} \left[ \gamma^t R(S_t, A_t) \right]$$

While effective, standard RL tends to produce brittle, deterministic policies that collapse prematurely into narrow local optima. If environment conditions change slightly (e.g., friction varies on a robot's leg), the brittle policy fails completely.

**Maximum Entropy Reinforcement Learning** augments the objective by rewarding the policy for acting as **randomly (exploratory) as possible** while still solving the task:

$$\pi^* = \arg\max_\pi \sum_{t=0}^\infty \mathbb{E}_{(S_t, A_t) \sim \pi} \left[ \gamma^t \left( R(S_t, A_t) + \alpha \mathcal{H}(\pi(\cdot \mid S_t)) \right) \right]$$
where $\mathcal{H}(\pi(\cdot \mid S_t)) = \mathbb{E}_{a \sim \pi} [-\log \pi(a \mid S_t)]$ is the Shannon entropy, and $\alpha > 0$ is the **temperature parameter** controlling the trade-off between reward maximization (exploitation) and entropy maximization (exploration).

In 2018, Tuomas Haarnoja et al. introduced **Soft Actor-Critic (SAC)**, which combines maximum entropy RL with off-policy actor-critic architectures. SAC achieved unprecedented state-of-the-art sample efficiency, stability, and robustness across complex continuous robotic benchmarks (MuJoCo, Dexterous Hand Manipulation).

```
+-----------------------------------------------------------------------------------------+
|                                    SAC ARCHITECTURE                                     |
|                                                                                         |
|       State s -------> [ Stochastic Actor mu_theta(s), sigma_theta(s) ]                 |
|                                        |                                                |
|                         Sample eps ~ N(0, I)                                            |
|                                        v                                                |
|                         Action a = tanh(mu + sigma * eps)                               |
|                                        |                                                |
|                                        v                                                |
|                    [ Twin Critics: Q_phi1(s, a), Q_phi2(s, a) ]                         |
|                                        |                                                |
|                    Soft Target: min(Q1_targ, Q2_targ) - alpha * log pi(a'|s')           |
|                                        |                                                |
|                    Temperature Tuning: alpha automatically matches H_target             |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Soft Bellman Equations

In Maximum Entropy RL, the definition of value functions incorporates the entropy of the policy:

#### 1. Soft State-Value Function $V(s)$:
$$V^\pi(s) \triangleq \mathbb{E}_{a \sim \pi} \left[ Q^\pi(s, a) - \alpha \log \pi(a \mid s) \right]$$

#### 2. Soft Action-Value Function $Q(s, a)$:
$$Q^\pi(s, a) \triangleq R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}} \left[ V^\pi(s') \right]$$

Substituting $V^\pi(s')$ into the Q-equation gives the **Soft Bellman Expectation Equation**:
$$Q^\pi(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}, a' \sim \pi} \left[ Q^\pi(s', a') - \alpha \log \pi(a' \mid s') \right]$$

---

### 2.2 The Soft Policy Iteration Theorem (Haarnoja et al., 2018)

#### 1. Soft Policy Evaluation:
Let $\mathcal{T}^\pi$ be the modified soft Bellman operator:
$$\mathcal{T}^\pi Q(s, a) \triangleq R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}} \left[ \mathbb{E}_{a' \sim \pi} \left[ Q(s', a') - \alpha \log \pi(a' \mid s') \right] \right]$$
Because the entropy term $-\alpha \log \pi(a' \mid s')$ is bounded for a given $\pi$, $\mathcal{T}^\pi$ is a $\gamma$-contraction in $\|\cdot\|_\infty$. Repeated application of $\mathcal{T}^\pi$ converges to the unique fixed point $Q^\pi$.

#### 2. Soft Policy Improvement:
In each iteration, the policy is updated toward the energy-based Boltzmann distribution by minimizing the KL divergence:
$$\pi_{\text{new}} = \arg\min_{\pi' \in \Pi} D_{\text{KL}} \left( \pi'(\cdot \mid s) \;\Big\|\; \frac{\exp\left( \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, \cdot) \right)}{Z^{\pi_{\text{old}}}(s)} \right)$$

#### Theorem: Monotonic Improvement
For any MDP with bounded rewards, $Q^{\pi_{\text{new}}}(s, a) \ge Q^{\pi_{\text{old}}}(s, a)$ for all $(s, a) \in \mathcal{S} \times \mathcal{A}$.

---

### 2.3 The Continuous Reparameterization Trick & $\tanh$ Squashing

To backpropagate through stochastic continuous actions, SAC uses the **reparameterization trick**:
$$u = \mu_{\boldsymbol{\theta}}(s) + \sigma_{\boldsymbol{\theta}}(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

To strictly enforce physical action boundaries $a \in [-1, 1]^m$, the unconstrained Gaussian variable $u$ is squashed through the hyperbolic tangent function:
$$a = \tanh(u)$$

#### Log-Probability under Change of Variables:
Because $\tanh$ is an invertible, non-linear mapping, the probability density function transforms according to the Jacobian determinant rule:
$$\pi(a \mid s) = \mu(u \mid s) \left| \det\left( \frac{da}{du} \right) \right|^{-1}$$

Since $\frac{d \tanh(u_i)}{du_i} = 1 - \tanh^2(u_i) = 1 - a_i^2$, the diagonal Jacobian determinant is:
$$\det\left( \frac{da}{du} \right) = \prod_{i=1}^m \left( 1 - \tanh^2(u_i) \right)$$

Taking the natural logarithm:
$$\log \pi(a \mid s) = \log \mu(u \mid s) - \sum_{i=1}^m \log\left( 1 - \tanh^2(u_i) + \delta_{\text{eps}} \right)$$
where $\mu(u \mid s)$ is the standard Gaussian density:
$$\log \mu(u \mid s) = - \frac{1}{2} \sum_{i=1}^m \left[ \log(2\pi) + 2 \log \sigma_i(s) + \left( \frac{u_i - \mu_i(s)}{\sigma_i(s)} \right)^2 \right]$$

---

### 2.4 Twin Q-Networks Critic Loss

To prevent overestimation bias, SAC adopts TD3's Clipped Double Q-learning.
The soft target value for transition $(s, a, r, s', d)$ is:
$$y = r + (1 - d) \gamma \left( \min_{j=1, 2} Q_{\boldsymbol{\phi}_j^-}(s', \tilde{a}') - \alpha \log \pi_{\boldsymbol{\theta}}(\tilde{a}' \mid s') \right)$$
where $\tilde{a}' \sim \pi_{\boldsymbol{\theta}}(\cdot \mid s')$.

The two critic networks $Q_{\boldsymbol{\phi}_1}, Q_{\boldsymbol{\phi}_2}$ minimize the Mean Squared Bellman Error:
$$\mathcal{L}(\boldsymbol{\phi}_1) = \mathbb{E} \left[ \frac{1}{2} \left( Q_{\boldsymbol{\phi}_1}(s, a) - y \right)^2 \right], \quad \mathcal{L}(\boldsymbol{\phi}_2) = \mathbb{E} \left[ \frac{1}{2} \left( Q_{\boldsymbol{\phi}_2}(s, a) - y \right)^2 \right]$$

---

### 2.5 Actor Loss (Policy Optimization)

The policy parameters $\boldsymbol{\theta}$ are optimized to maximize the soft Q-value while maximizing entropy:
$$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \alpha \log \pi_{\boldsymbol{\theta}}(a_{\boldsymbol{\theta}}(s, \epsilon) \mid s) - \min_{j=1, 2} Q_{\boldsymbol{\phi}_j}(s, a_{\boldsymbol{\theta}}(s, \epsilon)) \right]$$
The policy gradient flows directly through $a_{\boldsymbol{\theta}}(s, \epsilon)$ via the reparameterization trick!

---

### 2.6 Automatic Temperature Tuning

Setting $\alpha$ manually is difficult because the optimal temperature changes as the policy learns.
Haarnoja et al. (2018) formulated automatic tuning as a **constrained optimization problem**:
$$\max_\pi \mathbb{E} \left[ \sum_t R(S_t, A_t) \right] \quad \text{subject to} \quad \mathbb{E}_{(s, a) \sim \pi} [-\log \pi(a \mid s)] \ge \bar{\mathcal{H}}$$
where $\bar{\mathcal{H}} \triangleq - \dim(\mathcal{A})$ is the heuristic target entropy.

The dual objective for temperature $\alpha$ is:
$$\mathcal{L}(\alpha) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_{\boldsymbol{\theta}}} \left[ - \alpha \left( \log \pi_{\boldsymbol{\theta}}(a \mid s) + \bar{\mathcal{H}} \right) \right]$$

- If policy entropy is too low ($-\log \pi > \bar{\mathcal{H}}$), $\alpha$ increases to force more exploration.
- If policy entropy is high enough, $\alpha$ decreases to allow greedy exploitation!

---

## 3. Geometric & Physical Interpretation: Helmholtz Free Energy

In statistical mechanics:
- The internal energy of a system is $E = -Q(s, a)$.
- The entropy is $S = \mathcal{H}(\pi)$.
- The **Helmholtz Free Energy** is $F \triangleq E - T S = -Q + \alpha \log \pi$.

Soft Actor-Critic is a physical relaxation process that minimizes Helmholtz Free Energy:
$$F(\pi) = - \mathbb{E}_{a \sim \pi} [Q(s, a)] - \alpha \mathcal{H}(\pi)$$
The equilibrium state of minimum free energy is the **Boltzmann-Gibbs distribution**:
$$\pi^*(a \mid s) \propto \exp\left( \frac{1}{\alpha} Q^*(s, a) \right)$$
At high temperature $\alpha \to \infty$, the policy becomes uniform gas-like diffusion. At absolute zero $\alpha \to 0$, the policy crystallizes into deterministic argmax!

```
       Free Energy Landscape F(pi)
            ^
            |       High Entropy (Diffusion)            Low Temperature (Crystallization)
            |              \~~~~~/                                   |
            |               \   /                                    v
            |                \_/                              Narrow Spike
            +------------------------------------------------------------> Action Space
```

---

## 4. Real-World Analogy: The Maze Runner with Multiple Exits

Imagine navigating a labyrinth with two exits:
- **Standard RL (Deterministic):** Finds Exit A is 10 meters shorter than Exit B. It commits 100% of its resources to memorizing Exit A. If a boulder blocks Exit A during a test run, the agent is trapped and perishes.
- **Soft Actor-Critic (Maximum Entropy):** Realizes Exit A has value 100 and Exit B has value 98. It assigns $55\%$ probability to Exit A and $45\%$ probability to Exit B, keeping both pathways actively rehearsed and explored. If Exit A is suddenly blocked, the agent effortlessly pivots to Exit B without hesitation!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 1D Continuous Action SAC Step
Let us compute the exact numerical forward and backward pass of a Soft Actor-Critic step!

**Environment State:**
- Current state: $s = 1.0000$
- Target state: $s' = 2.0000$
- Reward: $r = 5.0000$
- Discount: $\gamma = 0.9000$
- Temperature: $\alpha = 0.2000$
- Target entropy: $\bar{\mathcal{H}} = -1.0000$ (for 1D action space)

**Actor Parameters (at $s' = 2.0000$):**
- Mean output: $\mu(s') = 0.5000$
- Log-std output: $\log \sigma(s') = 0.0000 \implies \sigma(s') = e^{0.0} = 1.0000$
- Sampled Gaussian noise: $\epsilon = +0.5000$

**Target Critics (evaluated at $s' = 2.0000$):**
- $Q_1^-(s', a') = 2.0000 \cdot s' + 1.0000 \cdot a'$
- $Q_2^-(s', a') = 1.5000 \cdot s' + 2.0000 \cdot a'$

We will compute:
1. Unsquashed action $u$ and squashed action $a'$
2. Exact log-probability $\log \pi(a' \mid s')$ with Jacobian correction
3. Twin target critic evaluations and soft target $y$
4. Temperature update gradient $\nabla_\alpha \mathcal{L}(\alpha)$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Value / Formula |
| :--- | :--- | :--- |
| $\mu, \sigma$ | Gaussian Mean and Std | $\mu = 0.5000, \sigma = 1.0000$ |
| $\epsilon$ | Sampled Standard Normal Noise | $\epsilon = +0.5000$ |
| $u$ | Unsquashed Action | $\mu + \sigma \epsilon = 0.5 + 1.0(0.5) = 1.0000$ |
| $a'$ | Squashed Action | $\tanh(u) = \tanh(1.0) \approx 0.76159$ |
| $\log \mu(u)$ | Base Gaussian Log-Likelihood | $-\frac{1}{2} \ln(2\pi) - \frac{(u - \mu)^2}{2\sigma^2}$ |
| $\left| \frac{da}{du} \right|$ | Jacobian Determinant | $1 - \tanh^2(u) = 1 - a'^2$ |
| $\log \pi(a')$ | Corrected Policy Log-Likelihood | $\log \mu(u) - \ln(1 - a'^2)$ |
| $y$ | Soft Bellman Target | $r + \gamma (\min(Q_1^-, Q_2^-) - \alpha \log \pi(a'))$ |

---

### 5.3 Step 1: Action Sampling & $\tanh$ Squashing

1. Compute unsquashed action $u$:
   $$u = \mu + \sigma \cdot \epsilon = 0.5000 + (1.0000 \times 0.5000) = \mathbf{1.0000}$$
2. Compute squashed action $a'$:
   $$a' = \tanh(u) = \tanh(1.0000) = \frac{e^1 - e^{-1}}{e^1 + e^{-1}} = \frac{2.71828 - 0.36788}{2.71828 + 0.36788} = \frac{2.35040}{3.08616} \approx \mathbf{0.76159}$$

---

### 5.4 Step 2: Log-Probability Computation with Jacobian Correction

1. Compute base Gaussian log-density $\log \mu(u \mid s')$:
   $$\log \mu(u \mid s') = -\frac{1}{2} \ln(2\pi) - \ln(\sigma) - \frac{(u - \mu)^2}{2\sigma^2}$$
   $$\ln(2\pi) \approx 1.83788 \implies -\frac{1}{2} \ln(2\pi) \approx -0.91894$$
   $$\ln(\sigma) = \ln(1.0) = 0.00000$$
   $$\frac{(1.0000 - 0.5000)^2}{2(1.0000)^2} = \frac{0.2500}{2.0000} = 0.12500$$
   $$\log \mu(u \mid s') = -0.91894 - 0.00000 - 0.12500 = \mathbf{-1.04394}$$

2. Compute Jacobian correction term:
   $$1 - a'^2 = 1 - (0.76159)^2 = 1 - 0.58002 = \mathbf{0.41998}$$
   $$\ln(1 - a'^2) = \ln(0.41998) \approx \mathbf{-0.86755}$$

3. Compute squashed policy log-probability:
   $$\log \pi(a' \mid s') = \log \mu(u \mid s') - \ln(1 - a'^2) = -1.04394 - (-0.86755) = \mathbf{-0.17639}$$

---

### 5.5 Step 3: Soft Bellman Target Evaluation

Evaluate both target critics at $(s' = 2.0000, a' = 0.76159)$:
- **Target Critic 1:**
  $$Q_1^-(s', a') = (2.0000 \times 2.0000) + (1.0000 \times 0.76159) = 4.0000 + 0.76159 = \mathbf{4.76159}$$
- **Target Critic 2:**
  $$Q_2^-(s', a') = (1.5000 \times 2.0000) + (2.0000 \times 0.76159) = 3.0000 + 1.52318 = \mathbf{4.52318}$$

Clipped Double Q minimum:
$$\min\left( Q_1^-(s', a'), Q_2^-(s', a') \right) = \min(4.76159, 4.52318) = \mathbf{4.52318}$$

Entropy-Augmented Value:
$$\tilde{V}(s') = \min(Q_1^-, Q_2^-) - \alpha \log \pi(a' \mid s')$$
$$= 4.52318 - \left[ 0.2000 \times (-0.17639) \right] = 4.52318 - (-0.03528) = 4.52318 + 0.03528 = \mathbf{4.55846}$$

Compute Soft Target $y$:
$$y = r + \gamma \tilde{V}(s') = 5.0000 + 0.9000 \times 4.55846 = 5.0000 + 4.10261 = \mathbf{9.10261}$$

---

### 5.6 Step 4: Automatic Temperature Gradient

The temperature loss is:
$$\mathcal{L}(\alpha) = - \alpha \left( \log \pi(a \mid s) + \bar{\mathcal{H}} \right)$$

With current policy log-probability $\log \pi = -0.17639$ and target entropy $\bar{\mathcal{H}} = -1.0000$:
$$\nabla_\alpha \mathcal{L}(\alpha) = - \left( \log \pi(a \mid s) + \bar{\mathcal{H}} \right) = - \left( -0.17639 + (-1.0000) \right) = - (-1.17639) = \mathbf{+1.17639}$$

Dual gradient step on temperature ($\alpha \leftarrow \alpha - \eta \nabla_\alpha \mathcal{L}$ with $\eta = 0.05$):
$$\alpha_{\text{new}} = 0.2000 - 0.0500 \times (+1.17639) = 0.2000 - 0.05882 = \mathbf{0.14118}$$
Since entropy was higher than required ($-\log \pi < 1.0$), the temperature $\alpha$ automatically decreases, allowing more exploitation!

---

### 5.7 Visual Summary Grid: Soft Actor-Critic Mechanics

| Quantity | Formula / Operation | Value | Significance |
| :--- | :--- | :---: | :--- |
| **Gaussian $u$** | $\mu + \sigma \epsilon = 0.5 + 1.0(0.5)$ | $\mathbf{1.0000}$ | Reparameterized latent sample |
| **Squashed $a'$** | $\tanh(1.0)$ | $\mathbf{0.76159}$ | Bounded continuous action $\in [-1, 1]$ |
| **Base Density** | $\log \mathcal{N}(1.0; 0.5, 1)$ | $\mathbf{-1.04394}$ | Gaussian log likelihood |
| **Jacobian Log** | $\ln(1 - a'^2)$ | $\mathbf{-0.86755}$ | Volume change of squashing |
| **Policy Log Prob** | $\log \mu - \ln(1 - a'^2)$ | $\mathbf{-0.17639}$ | Exact squashed log-probability |
| **Twin Q Min** | $\min(4.7616, 4.5232)$ | $\mathbf{4.52318}$ | Mitigates maximization bias |
| **Soft Target $y$** | $5.0 + 0.9(4.5232 - 0.2(-0.1764))$ | $\mathbf{9.10261}$ | Entropy-augmented Bellman target |
| **$\nabla_\alpha \mathcal{L}$** | $-(\log \pi + \bar{\mathcal{H}})$ | $\mathbf{+1.17639}$ | Directs automatic temperature adaptation |

---

## 6. Solved Illustrations

### Illustration 1: Derivation of the $\tanh$ Jacobian Correction
**Problem:**
Let $u \sim p(u)$ be a 1D continuous random variable. Let $a = \tanh(u)$.
Prove rigorously that the log-probability density transforms as:
$$\log p(a) = \log p(u) - \log(1 - a^2)$$

**Solution:**
By the probability conservation law for continuous distributions:
$$p(a) |da| = p(u) |du| \implies p(a) = p(u) \left| \frac{da}{du} \right|^{-1}$$
Differentiate $a = \tanh(u) = \frac{\sinh(u)}{\cosh(u)}$:
$$\frac{da}{du} = \frac{\cosh^2(u) - \sinh^2(u)}{\cosh^2(u)} = 1 - \tanh^2(u) = 1 - a^2$$
Since $a \in (-1, 1)$, $1 - a^2 > 0$, so $|1 - a^2| = 1 - a^2$.
Taking the natural logarithm of both sides:
$$\log p(a) = \log\left( \frac{p(u)}{1 - a^2} \right) = \log p(u) - \log(1 - a^2) \quad \blacksquare$$

---

## 7. Deep Learning Connection & Modern Applications

- **The King of Model-Free Continuous Control:** SAC is universally regarded as the most stable, sample-efficient off-policy continuous RL algorithm. It is standard for training quadruped robots (Unitree Go1, ANYmal) to walk on rough terrain within hours of real-world training.
- **Cross-Entropy & Diffusion Policies:** Modern robotic foundation models (e.g. Diffusion Policy, Octo) build directly on the maximum-entropy formulation pioneered by SAC.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of Part 5 Soft Actor-Critic hand calculations:
   - Action squashing $a' = 0.76159$
   - Corrected log-prob $\log \pi = -0.17639$
   - Soft target $y = 9.10261$
   - Temperature gradient $\nabla_\alpha = +1.17639$ matching PyTorch autograd to $< 10^{-14}$.
2. Complete standalone SAC agent with Automatic Temperature Tuning trained on Continuous Pendulum.

See implementation in:
[`11_reinforcement_learning/code/20_soft_actor_critic.py`](./code/20_soft_actor_critic.py)
