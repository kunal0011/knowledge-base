# Module 11.13: The Policy Gradient Theorem & REINFORCE

---

## 1. Intuition & 101 Motivation

In Sub-Modules 2 and 3, we studied **value-based reinforcement learning** (Q-learning, DQN), where the agent learns the expected return $Q(s, a)$ and selects actions greedily:
$$\pi(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$

While powerful for discrete environments with small action sets, value-based methods face fundamental limitations:
1. **Continuous Action Spaces:** If the action space is continuous (e.g., robotic joint torques $a \in [-1, 1]^8$), finding $\arg\max_a Q(s, a)$ at every single time step requires solving an expensive non-convex numerical optimization problem.
2. **Stochastic Optimal Policies:** In partially observable Markov decision processes (POMDPs) or competitive games (e.g., Rock-Paper-Scissors, Poker), the optimal policy is inherently stochastic (e.g., throwing Rock, Paper, or Scissors with probability $1/3$ each). Greedy value functions can only output deterministic policies.
3. **Action Chattering:** Small changes in value estimates can cause discontinuous jumps in the selected action, destabilizing policy execution.

**Policy-based methods** parameterize the policy directly as a probability distribution:
$$\pi_\theta(a \mid s) \triangleq \mathbb{P}(A_t = a \mid S_t = s; \boldsymbol{\theta})$$
where $\boldsymbol{\theta} \in \mathbb{R}^d$ is a continuous parameter vector (e.g., neural network weights).

The central mathematical challenge is: **How can we compute the gradient of expected return with respect to policy parameters $\boldsymbol{\theta}$ when the environment's transition dynamics $\mathcal{P}_{ss'}^a$ are completely unknown?**

The profound answer is provided by the **Policy Gradient Theorem** (Sutton et al., 1999).

```
   VALUE-BASED (DQN)                             POLICY-BASED (REINFORCE)
   [Indirect Policy via Q]                       [Direct Parameterization]

         theta                                             theta
           |                                                 |
           v                                                 v
        Q(s, a)                                       pi_theta(a | s)
           |                                                 |
           v (arg max)                                       v (Sample a ~ pi)
         Action a                                          Action a
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Objective Function

Let $\tau = (s_0, a_0, r_1, s_1, a_1, \dots, s_T)$ denote a complete episodic trajectory.
The total return of trajectory $\tau$ is:
$$R(\tau) \triangleq \sum_{t=0}^{T-1} \gamma^t r_{t+1}$$

The probability of trajectory $\tau$ occurring under policy parameters $\boldsymbol{\theta}$ is:
$$P(\tau; \boldsymbol{\theta}) = \mu(s_0) \prod_{t=0}^{T-1} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \mathcal{P}(s_{t+1} \mid s_t, a_t)$$
where $\mu(s_0)$ is the initial state distribution and $\mathcal{P}$ is the transition dynamics.

The reinforcement learning goal is to maximize the expected return:
$$J(\boldsymbol{\theta}) \triangleq \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} [R(\tau)] = \int P(\tau; \boldsymbol{\theta}) R(\tau) d\tau$$

---

### 2.2 The Log-Derivative Trick (Likelihood Ratio / Score Function)

To optimize $J(\boldsymbol{\theta})$ via gradient ascent ($\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \nabla_{\boldsymbol{\theta}} J$), we differentiate the expectation:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \int P(\tau; \boldsymbol{\theta}) R(\tau) d\tau = \int \nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) R(\tau) d\tau$$

Notice that by the chain rule of calculus:
$$\nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) = \frac{\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})}{P(\tau; \boldsymbol{\theta})} \implies \nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) = P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})$$

Substituting this back into the integral:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \int P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \right]$$

---

### 2.3 Vanishing of the Unknown Environment Dynamics

Now expand $\log P(\tau; \boldsymbol{\theta})$:
$$\log P(\tau; \boldsymbol{\theta}) = \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) + \sum_{t=0}^{T-1} \log \mathcal{P}(s_{t+1} \mid s_t, a_t)$$

Taking the gradient with respect to $\boldsymbol{\theta}$:
- $\nabla_{\boldsymbol{\theta}} \log \mu(s_0) = \mathbf{0}$ (initial state distribution does not depend on $\boldsymbol{\theta}$).
- $\nabla_{\boldsymbol{\theta}} \log \mathcal{P}(s_{t+1} \mid s_t, a_t) = \mathbf{0}$ (the laws of physics / environment dynamics do NOT depend on $\boldsymbol{\theta}$!).

Therefore, the environmental transition model completely vanishes:
$$\nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) = \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$$

Substituting this yields the celebrated **Policy Gradient Theorem**:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) R(\tau) \right]$$

---

### 2.4 Causality (Reward-to-Go)

In the formula above, the action $a_t$ taken at time $t$ is multiplied by the entire trajectory return $R(\tau) = \sum_{k=0}^{T-1} \gamma^k r_{k+1}$, including rewards earned in the past ($k < t$).
By the law of causality, an action taken at time $t$ cannot influence rewards obtained prior to time $t$:
$$\mathbb{E} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) r_{k+1} \right] = \mathbf{0} \quad \forall k < t$$

We can replace the total return $R(\tau)$ with the **Reward-to-Go (Return from time $t$)**:
$$G_t \triangleq \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$$
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) G_t \right]$$
This simple modification drastically reduces the estimator's variance without introducing any bias.

---

### 2.5 Baseline Invariance & Variance Reduction

The Monte Carlo sample return $G_t$ still exhibits substantial variance across episodes. We can reduce this variance by subtracting a state-dependent **baseline** $b(s_t)$ from $G_t$.

#### Theorem: Baseline Invariance
Let $b(s)$ be any arbitrary function of state $s$ that does NOT depend on action $a$. Then the expected gradient contribution of $b(s)$ is **identically zero**:
$$\mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) b(s) \right] = \mathbf{0}$$

#### Proof
Expand the expectation over the discrete action space $\mathcal{A}$:
$$\mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) b(s) \right] = \sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid s) \frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}}(a \mid s)} b(s)$$
$$= b(s) \sum_{a \in \mathcal{A}} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s) = b(s) \nabla_{\boldsymbol{\theta}} \left( \sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid s) \right)$$
Because $\pi_{\boldsymbol{\theta}}$ is a valid probability distribution, the sum of probabilities over all actions is always identically 1: $\sum_a \pi_{\boldsymbol{\theta}}(a \mid s) = 1$.
The gradient of a constant is zero:
$$= b(s) \nabla_{\boldsymbol{\theta}} (1) = b(s) \cdot \mathbf{0} = \mathbf{0} \quad \blacksquare$$

Therefore, the policy gradient with baseline:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E} \left[ \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( G_t - b(s_t) \right) \right]$$
is **strictly unbiased for any baseline $b(s)$**!

The baseline that minimizes variance is approximately the state value function:
$$b(s) \approx V(s) = \mathbb{E}[G_t \mid S_t = s]$$
The difference $G_t - V(s_t)$ is the empirical **Advantage function** $\hat{A}_t$, measuring whether action $a_t$ performed better or worse than the state's average expectation.

---

### 2.6 The REINFORCE Algorithm (Williams, 1992)

```
Algorithm: REINFORCE with Baseline
Input: Differentiable policy pi_theta(a | s), value baseline V_phi(s)
Parameters: Learning rates alpha_theta, alpha_phi

Loop forever (for each episode):
    1. Generate an episode tau = (S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T) following pi_theta
    2. For each step t = 0, 1, ..., T-1:
         Compute return: G_t = sum_{k=t}^{T-1} gamma^{k-t} R_{k+1}
         Compute advantage: delta_t = G_t - V_phi(S_t)
         Update baseline critic: phi <-- phi + alpha_phi * delta_t * grad_phi V_phi(S_t)
         Update policy actor: theta <-- theta + alpha_theta * gamma^t * delta_t * grad_theta log pi_theta(A_t | S_t)
```

---

## 3. Geometric & Physical Interpretation: Probability Simplex Flow

In the $|\mathcal{A}|$-dimensional probability simplex $\Delta^{|\mathcal{A}|}$:
- The score function $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)$ points in the direction that **increases the probability of action $a$** while simultaneously decreasing the probabilities of all competing actions (since probabilities must sum to 1).
- The advantage scalar $A_t = G_t - b(s_t)$ acts as a **directional amplifier**:
  - If $A_t > 0$ (better than average): the policy gradient pushes the probability mass toward action $a_t$.
  - If $A_t < 0$ (worse than average): the negative multiplier **repels** the policy away from action $a_t$, draining its probability mass!

```
                    Probability Simplex
                           (a0)
                            /\
                           /  \
                          /    \
                         /   *  \  <- Gradient pushes toward a1
                        /   /    \    if A(a1) > 0!
                       /   v      \
                     (a1)--------(a2)
```

---

## 4. Real-World Analogy: The Theater Audition

Imagine a theater director coaching an actor:
- **Value-Based Approach:** The director tries to assign a numerical score to every possible muscle twitch of the actor's face in advance. This is exhausting and rigid.
- **Policy-Based (REINFORCE) Approach:** 
  - The actor delivers a full monologue with their natural expressive style $\pi_{\boldsymbol{\theta}}$.
  - At the end of the scene, the director gives feedback: "Your emotional intensity during the second stanza was fantastic ($A_t = +5$), but your tone in the opening was flat ($A_t = -3$)."
  - The actor increases the probability of using that emotional cadence in future performances, naturally refining their style without anyone calculating a microscopic muscle lookup table.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: Softmax Policy on a 2-Action Problem
Consider a bandit / state with 2 discrete actions: $\mathcal{A} = \{a_0, a_1\}$.
The policy is parameterized by a 2D logit vector $\boldsymbol{\theta} = [\theta_0, \theta_1]^\top$:
$$\pi_{\boldsymbol{\theta}}(a_i) \triangleq \frac{e^{\theta_i}}{e^{\theta_0} + e^{\theta_1}}$$

**Initial Parameters:**
$$\boldsymbol{\theta} = \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix}$$

**Baseline Value:** $b = 5.0000$
**Learning Rate:** $\alpha = 0.1000$

**Observed Experience:**
The agent samples action $A_0 = a_0$ and receives a high return $G_0 = 10.0000$!

We will compute:
1. Current action probabilities $\pi(a_0)$ and $\pi(a_1)$
2. Analytical score function $\nabla_{\boldsymbol{\theta}} \log \pi(a_0)$
3. Advantage $A_0 = G_0 - b$
4. Policy gradient vector $\mathbf{g}$
5. Updated parameter vector $\boldsymbol{\theta}_{\text{new}}$
6. New updated action probabilities $\pi_{\text{new}}(a_0)$ and $\pi_{\text{new}}(a_1)$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Walkthrough Value |
| :--- | :--- | :--- |
| $\boldsymbol{\theta}$ | Policy Parameter Logits | $[\theta_0=1.0000, \theta_1=2.0000]^\top$ |
| $e^{\theta_i}$ | Unnormalized Logit Exponentials | $e^{1.0} \approx 2.71828, e^{2.0} \approx 7.38906$ |
| $\pi(a_i)$ | Action Probabilities (Softmax) | $\pi_0 \approx 0.26894, \pi_1 \approx 0.73106$ |
| $\nabla_{\boldsymbol{\theta}} \log \pi(a_0)$ | Score Function Vector for $a_0$ | $[1 - \pi_0, -\pi_1]^\top$ |
| $G_0$ | Observed Empirical Return | $10.0000$ |
| $b$ | Baseline Value | $5.0000$ |
| $A_0$ | Advantage | $G_0 - b = 10.0 - 5.0 = 5.0000$ |
| $\mathbf{g}$ | Policy Gradient Estimate | $A_0 \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a_0)$ |
| $\boldsymbol{\theta}_{\text{new}}$ | Updated Parameters | $\boldsymbol{\theta} + \alpha \mathbf{g}$ |

---

### 5.3 Step 1: Forward Pass (Compute Initial Probabilities)

Exponentiate the logits:
$$e^{\theta_0} = e^{1.0000} \approx 2.71828$$
$$e^{\theta_1} = e^{2.0000} \approx 7.38906$$
Sum of exponentials:
$$Z = e^{\theta_0} + e^{\theta_1} = 2.71828 + 7.38906 = 10.10734$$

Softmax Probabilities:
$$\pi(a_0) = \frac{2.71828}{10.10734} \approx \mathbf{0.26894}$$
$$\pi(a_1) = \frac{7.38906}{10.10734} \approx \mathbf{0.73106}$$
Notice: Initially, action $a_1$ dominates with $\approx 73.1\%$ probability, while chosen action $a_0$ has only $\approx 26.9\%$.

---

### 5.4 Step 2: Compute the Score Function $\nabla_{\boldsymbol{\theta}} \log \pi(a_0)$

For a softmax policy $\pi(a_i) = \frac{e^{\theta_i}}{\sum_j e^{\theta_j}}$, the derivative with respect to $\theta_k$ is:
$$\frac{\partial \log \pi(a_i)}{\partial \theta_k} = \mathbb{I}(i = k) - \pi(a_k)$$

For the chosen action $a_0$:
- Derivative with respect to $\theta_0$ ($i = k = 0$):
  $$\frac{\partial \log \pi(a_0)}{\partial \theta_0} = 1 - \pi(a_0) = 1.0000 - 0.26894 = \mathbf{+0.73106}$$
- Derivative with respect to $\theta_1$ ($i = 0, k = 1$):
  $$\frac{\partial \log \pi(a_0)}{\partial \theta_1} = 0 - \pi(a_1) = 0.0000 - 0.73106 = \mathbf{-0.73106}$$

Score function gradient vector:
$$\nabla_{\boldsymbol{\theta}} \log \pi(a_0) = \begin{bmatrix} +0.73106 \\ -0.73106 \end{bmatrix}$$
Notice: The sum of elements is $+0.73106 - 0.73106 = 0$ (probability conservation!).

---

### 5.5 Step 3: Compute Advantage & Policy Gradient Vector $\mathbf{g}$

Compute Advantage:
$$A_0 = G_0 - b = 10.0000 - 5.0000 = \mathbf{+5.0000}$$

Compute Policy Gradient:
$$\mathbf{g} = A_0 \nabla_{\boldsymbol{\theta}} \log \pi(a_0) = 5.0000 \times \begin{bmatrix} +0.73106 \\ -0.73106 \end{bmatrix} = \begin{bmatrix} +3.65530 \\ -3.65530 \end{bmatrix}$$

---

### 5.6 Step 4: Parameter Update ($\alpha = 0.1000$)

Apply gradient ascent:
$$\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta} + \alpha \mathbf{g} = \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix} + 0.1000 \begin{bmatrix} +3.65530 \\ -3.65530 \end{bmatrix}$$
$$\boldsymbol{\theta}_{\text{new}} = \begin{bmatrix} 1.0000 + 0.36553 \\ 2.0000 - 0.36553 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.36553 \\ 1.63447 \end{bmatrix}}$$

---

### 5.7 Step 5: Verify Updated Policy Probabilities

Compute new exponentials with updated $\boldsymbol{\theta}_{\text{new}}$:
$$e^{1.36553} \approx 3.91778$$
$$e^{1.63447} \approx 5.12674$$
Sum:
$$Z_{\text{new}} = 3.91778 + 5.12674 = 9.04452$$

New Softmax Probabilities:
$$\pi_{\text{new}}(a_0) = \frac{3.91778}{9.04452} \approx \mathbf{0.43317}$$
$$\pi_{\text{new}}(a_1) = \frac{5.12674}{9.04452} \approx \mathbf{0.56683}$$

**Profound Result:**
Because action $a_0$ delivered an above-average return ($A_0 = +5.0$), its selection probability jumped from **$26.89\%$ up to $43.32\%$** in a single gradient step!

---

### 5.8 Visual Summary Grid

| Quantity | Initial State ($\boldsymbol{\theta}$) | Gradient Component | Updated State ($\boldsymbol{\theta}_{\text{new}}$) | Delta Change |
| :---: | :---: | :---: | :---: | :---: |
| $\theta_0$ | $1.0000$ | $g_0 = +3.6553$ | $\mathbf{1.36553}$ | $+0.36553$ |
| $\theta_1$ | $2.0000$ | $g_1 = -3.6553$ | $\mathbf{1.63447}$ | $-0.36553$ |
| $\pi(a_0)$ | $0.26894$ ($26.89\%$) | Score $= +0.7311$ | $\mathbf{0.43317}$ ($43.32\%$) | $\mathbf{+16.43\%}$ $\uparrow$ |
| $\pi(a_1)$ | $0.73106$ ($73.11\%$) | Score $= -0.7311$ | $\mathbf{0.56683}$ ($56.68\%$) | $\mathbf{-16.43\%}$ $\downarrow$ |

---

## 6. Solved Illustrations

### Illustration 1: Continuous Action Gaussian Policy Gradient
**Problem:**
Let the action space be continuous $\mathcal{A} = \mathbb{R}$.
The policy is a 1D Gaussian distribution with state-dependent mean $\mu_\theta(s)$ and fixed standard deviation $\sigma$:
$$\pi_\theta(a \mid s) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(a - \mu_\theta(s))^2}{2\sigma^2} \right)$$
Derive the analytical score function $\nabla_\theta \log \pi_\theta(a \mid s)$.

**Solution:**
Take the natural logarithm:
$$\log \pi_\theta(a \mid s) = -\frac{1}{2} \log(2\pi) - \log(\sigma) - \frac{(a - \mu_\theta(s))^2}{2\sigma^2}$$
Differentiating with respect to $\theta$ via the chain rule:
$$\nabla_\theta \log \pi_\theta(a \mid s) = -\frac{1}{2\sigma^2} \cdot 2(a - \mu_\theta(s)) \cdot (-\nabla_\theta \mu_\theta(s))$$
$$= \frac{a - \mu_\theta(s)}{\sigma^2} \nabla_\theta \mu_\theta(s)$$

**Physical Meaning:**
If the sampled action $a$ is larger than the predicted mean ($a > \mu_\theta(s)$) and resulted in a positive advantage ($A > 0$), the gradient pushes $\mu_\theta(s)$ to the right (toward $a$). If $A < 0$, it pushes $\mu_\theta(s)$ away from $a$! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. The Foundation of Modern LLM Alignment (RLHF)
The classical REINFORCE algorithm is the foundation of Reinforcement Learning from Human Feedback (RLHF):
$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \cdot \left( R(x, y) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right) \right]$$

### 2. DeepSeek-R1 & GRPO (Group Relative Policy Optimization)
DeepSeek's breakthrough R1 model replaces the expensive critic network in PPO with a group-relative REINFORCE baseline:
$$A_i = \frac{r_i - \operatorname{mean}(\{r_1, \dots, r_G\})}{\operatorname{std}(\{r_1, \dots, r_G\})}$$
computing policy gradients directly via multi-token REINFORCE rollouts!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of Part 5 Softmax Policy hand calculations:
   - Initial probabilities $\pi = [0.26894, 0.73106]$
   - Score function $\nabla \log \pi = [+0.73106, -0.73106]$
   - Updated parameters $\boldsymbol{\theta}_{\text{new}} = [1.36553, 1.63447]$
   - Updated probabilities $\pi_{\text{new}} = [0.43317, 0.56683]$ matching PyTorch autograd to $< 10^{-14}$.
2. Full REINFORCE with Learned Baseline agent trained on CartPole-v1, demonstrating monotonic reward improvement.

See implementation in:
[`11_reinforcement_learning/code/13_policy_gradient_and_reinforce.py`](./code/13_policy_gradient_and_reinforce.py)
