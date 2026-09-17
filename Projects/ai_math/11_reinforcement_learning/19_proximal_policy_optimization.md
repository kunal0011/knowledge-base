# Module 11.19: Proximal Policy Optimization (PPO)

---

## 1. Intuition & 101 Motivation

In Chapter 11.18, we studied **Trust Region Policy Optimization (TRPO)**, which provided rigorous mathematical guarantees of monotonic policy improvement by enforcing an average KL divergence constraint $\bar{D}_{\text{KL}}(\pi_{\text{old}} \parallel \pi) \le \delta$.

However, TRPO's heavy reliance on second-order optimization introduces severe practical drawbacks:
- It requires **Conjugate Gradient (CG)** loops and **Hessian-Vector Products**, adding architectural complexity.
- It cannot easily share parameters between the Actor and Critic trunks.
- It is incompatible with modern deep architectures like Transformers, LSTMs, and complex multi-head networks trained via standard backpropagation.

In 2017, John Schulman et al. at OpenAI introduced **Proximal Policy Optimization (PPO)**. PPO asked a simple question: *Can we achieve the stability and sample efficiency of TRPO using simple, first-order stochastic gradient descent (Adam)?*

The answer is the **PPO Clipped Surrogate Objective**: an elegant mathematical formulation that clips the probability ratio $r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$ to the interval $[1 - \epsilon, 1 + \epsilon]$, creating a zero-gradient safety plateau that physically prevents destructive policy updates.

Today, PPO is the undisputed **workhorse algorithm of modern deep reinforcement learning**, powering breakthroughs from OpenAI Five (Dota 2) to **Reinforcement Learning from Human Feedback (RLHF)** in ChatGPT and frontier reasoning models.

```
+-----------------------------------------------------------------------------------------+
|                                    PPO-CLIP OBJECTIVE                                   |
|                                                                                         |
|       Objective L^CLIP                                                                  |
|             ^                                                                           |
|             |          Positive Advantage (A > 0)                                       |
|  (1+eps)*A  +                 /-------------------------  (Zero gradient plateau!)      |
|             |                /                                                          |
|             |               /                                                           |
|             |              /                                                            |
|          0  +-------------+--------+--------+-------------> Probability Ratio r         |
|             |            1-eps    1.0      1+eps                                        |
|             |                                                                           |
|             |          Negative Advantage (A < 0)                                       |
|             |    --------------\                                                        |
|  (1-eps)*A  +  (Zero grad)      \                                                       |
|             |                    \                                                      |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Probability Ratio

Let $\pi_{\boldsymbol{\theta}_{\text{old}}}$ be the policy parameters used to collect rollout experience.
For any candidate parameter vector $\boldsymbol{\theta}$, the **probability ratio** at time step $t$ is:

$$r_t(\boldsymbol{\theta}) \triangleq \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$$

Notice that when $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}$, the ratio is identically unity: $r_t(\boldsymbol{\theta}_{\text{old}}) = 1.0$.

The unconstrained surrogate objective is:
$$L^{\text{CPI}}(\boldsymbol{\theta}) \triangleq \hat{\mathbb{E}}_t \left[ \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\boldsymbol{\theta}) \hat{A}_t \right]$$
where CPI stands for *Conservative Policy Iteration*. Without constraints, maximizing $L^{\text{CPI}}$ leads to excessively large steps and catastrophic policy collapse.

---

### 2.2 The Clipped Surrogate Objective ($L^{\text{CLIP}}$)

To penalize moves that take $r_t(\boldsymbol{\theta})$ far away from $1$, PPO modifies the objective using a **min-clip** operator:

$$L^{\text{CLIP}}(\boldsymbol{\theta}) \triangleq \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta}) \hat{A}_t, \quad \operatorname{clip}(r_t(\boldsymbol{\theta}), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
where $\epsilon$ is a hyperparameter (typically $\epsilon = 0.20$).

#### Mathematical Case Analysis:

1. **Case 1: Positive Advantage ($\hat{A}_t > 0$):**
   The chosen action yielded a higher return than average. We want to *increase* its probability ($\pi_{\boldsymbol{\theta}} \uparrow \implies r_t > 1$).
   - When $1 \le r_t \le 1 + \epsilon$: The step is within the trust region. The objective is $r_t \hat{A}_t$, providing a positive gradient pushing $\pi_{\boldsymbol{\theta}}$ upward.
   - When $r_t > 1 + \epsilon$: The update has already made the action sufficiently more likely. The objective is clipped to $(1 + \epsilon) \hat{A}_t$.
     $$\left. \nabla_{\boldsymbol{\theta}} L^{\text{CLIP}} \right|_{r_t > 1 + \epsilon} = \mathbf{0}$$
     The gradient vanishes! The optimizer cannot push the ratio any further, preventing overconfidence.

2. **Case 2: Negative Advantage ($\hat{A}_t < 0$):**
   The chosen action performed worse than expected. We want to *decrease* its probability ($\pi_{\boldsymbol{\theta}} \downarrow \implies r_t < 1$).
   - When $1 - \epsilon \le r_t \le 1$: The objective is $r_t \hat{A}_t$, providing a gradient that decreases $\pi_{\boldsymbol{\theta}}$.
   - When $r_t < 1 - \epsilon$: The action has already been sufficiently suppressed. Because $\hat{A}_t < 0$, multiplying by $(1 - \epsilon)$ gives a larger (less negative) number than multiplying by $r_t < 1 - \epsilon$. The $\min$ operator selects the lower bound $(1 - \epsilon) \hat{A}_t$.
     $$\left. \nabla_{\boldsymbol{\theta}} L^{\text{CLIP}} \right|_{r_t < 1 - \epsilon} = \mathbf{0}$$
     The gradient vanishes! The policy avoids over-correcting and destroying exploratory entropy.

#### The Pessimistic Bound:
Taking the minimum between the unclipped and clipped terms ensures that $L^{\text{CLIP}}(\boldsymbol{\theta})$ forms a **pessimistic lower bound**:
$$L^{\text{CLIP}}(\boldsymbol{\theta}) \le r_t(\boldsymbol{\theta}) \hat{A}_t$$
We only ignore the change when the objective would have become *better* than the bound, never when it becomes worse!

---

### 2.3 The Full Multi-Task Objective

In deep networks, Actor and Critic heads share underlying representation layers. The unified objective maximized via Adam is:

$$\mathcal{L}^{\text{PPO}}(\boldsymbol{\theta}, \boldsymbol{\phi}) \triangleq \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\boldsymbol{\theta}) - c_1 L_t^{\text{VF}}(\boldsymbol{\phi}) + c_2 \mathcal{H}(\pi_{\boldsymbol{\theta}}(\cdot \mid S_t)) \right]$$

where:
1. **Clipped Value Function Loss ($L^{\text{VF}}$):**
   Similar to the policy, value targets can also be clipped around the old value $V_{\boldsymbol{\phi}_{\text{old}}}$:
   $$L_t^{\text{VF}}(\boldsymbol{\phi}) = \max \left( (V_{\boldsymbol{\phi}}(S_t) - V_t^{\text{targ}})^2, \quad (\operatorname{clip}(V_{\boldsymbol{\phi}}(S_t), V_{\boldsymbol{\phi}_{\text{old}}}(S_t) - \epsilon_v, V_{\boldsymbol{\phi}_{\text{old}}}(S_t) + \epsilon_v) - V_t^{\text{targ}})^2 \right)$$
2. **Entropy Bonus ($\mathcal{H}$):**
   $$\mathcal{H}(\pi_{\boldsymbol{\theta}}(\cdot \mid S_t)) = - \sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid S_t) \log \pi_{\boldsymbol{\theta}}(a \mid S_t)$$
3. **Coefficients:** Typically $c_1 \in [0.5, 1.0]$ and $c_2 \in [0.01, 0.05]$.

---

### 2.4 Multi-Epoch Mini-Batch Optimization Loop

Unlike standard policy gradients (which can only perform 1 gradient step per rollout), PPO can perform **multiple epochs of mini-batch gradient descent** on the same collected data:

```
Algorithm: PPO with Clipped Objective
Parameters: Rollout horizon T, Actors N, Epochs K, Mini-batch size M, Clip eps=0.2

Loop forever:
    1. For actor = 1 to N:
         Run policy pi_theta_old for T timesteps in environment
         Compute GAE advantages A_1, ..., A_T using lambda and gamma
    2. Collect all N * T samples into dataset D
    3. Normalize advantages: A_hat = (A - mean(A)) / (std(A) + 1e-8)
    4. For epoch = 1 to K (e.g., K = 4 or 10):
         Shuffle dataset D into mini-batches of size M
         For each mini-batch:
             Compute ratio r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
             Compute L^CLIP(theta)
             Compute Value Loss L^VF(phi)
             Compute Entropy H
             Perform Adam gradient update on -L^PPO(theta, phi)
    5. Set theta_old <-- theta
```

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.19.1: PPO-Clip Surrogate Objective and Lower Bound Property

```
====================================================================================================
DERIVATION 11.19.1: PPO-Clip Surrogate Objective and Universal Pessimistic Lower Bound
====================================================================================================
Problem Statement:
Let π_{θ_old} be the rollout policy, π_θ a candidate policy, r_t(θ) ≜ π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)
the probability ratio, ϵ ∈ (0, 1) the clipping threshold, and Â_t ∈ ℝ the estimated advantage.
Define the PPO clipped surrogate objective:
    L^{CLIP}(θ) ≜ Ê_t [ min( r_t(θ) Â_t,  clip(r_t(θ), 1 - ϵ, 1 + ϵ) Â_t ) ]
Prove from first principles:
1. For every transition (s_t, a_t), every parameter vector θ, and every Â_t ∈ ℝ:
    L_t^{CLIP}(θ) ≤ r_t(θ) Â_t
   establishing that L^{CLIP}(θ) is a pointwise pessimistic lower bound on the unclipped surrogate L^{CPI}(θ).
2. Specifically, when Â_t > 0:
    L_t^{CLIP}(θ) = min( r_t(θ) Â_t, (1 + ϵ) Â_t ) ≤ r_t(θ) Â_t
   with equality on [0, 1 + ϵ] and strict inequality for r_t(θ) > 1 + ϵ.
3. Connect this property to Kakade & Langford's conservative policy improvement lower bound.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
In policy search, the expected discounted return of a candidate policy $\pi_{\boldsymbol{\theta}}$ is denoted $\eta(\pi_{\boldsymbol{\theta}}) \triangleq \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]$. Under the policy improvement theorem of Kakade & Langford (2002), the return of $\pi_{\boldsymbol{\theta}}$ can be expressed relative to a reference rollout policy $\pi_{\boldsymbol{\theta}_{\text{old}}}$:
$$\eta(\pi_{\boldsymbol{\theta}}) = \eta(\pi_{\boldsymbol{\theta}_{\text{old}}}) + \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=0}^\infty \gamma^t A_{\pi_{\boldsymbol{\theta}_{\text{old}}}}(s_t, a_t) \right]$$
Because sampling trajectories directly from $\pi_{\boldsymbol{\theta}}$ during optimization is impossible without interacting with the environment at every sub-step, Conservative Policy Iteration (CPI) replaces the state visitation distribution $\rho_{\pi_{\boldsymbol{\theta}}}$ with $\rho_{\pi_{\boldsymbol{\theta}_{\text{old}}}}$ and uses importance sampling over actions:
$$L^{\text{CPI}}(\boldsymbol{\theta}) \triangleq \hat{\mathbb{E}}_t \left[ \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\boldsymbol{\theta}) \hat{A}_t \right]$$
Maximizing $L^{\text{CPI}}(\boldsymbol{\theta})$ without constraint leads to catastrophic policy collapse when $r_t(\boldsymbol{\theta})$ deviates significantly from $1.0$. PPO replaces $L^{\text{CPI}}$ with the clipped surrogate:
$$L^{\text{CLIP}}(\boldsymbol{\theta}) \triangleq \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta})\hat{A}_t, \quad \operatorname{clip}(r_t(\boldsymbol{\theta}), 1 - \epsilon, 1 + \epsilon)\hat{A}_t \right) \right]$$
The mathematical goal is to prove that $L_t^{\text{CLIP}}(\boldsymbol{\theta}) \le r_t(\boldsymbol{\theta})\hat{A}_t$ universally across all $\hat{A}_t \in \mathbb{R}$ and all $r_t \in [0, \infty)$, providing an analytical lower bound that enforces conservative policy improvement without requiring second-order Hessian computation.

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Support Invariance (Common Support):** For all state-action pairs $(s_t, a_t)$ visited under $\pi_{\boldsymbol{\theta}_{\text{old}}}$, the reference policy has non-zero probability: $\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t) > 0$. The ratio $r_t(\boldsymbol{\theta}) \in [0, \infty)$ is well-defined.
2. **Finite Bounded Advantages:** The estimated advantage $\hat{A}_t$ is bounded: $|\hat{A}_t| \le A_{\max} < \infty$.
3. **Valid Clipping Hyperparameter:** The clipping threshold satisfies $\epsilon \in (0, 1)$, ensuring $0 < 1 - \epsilon < 1 < 1 + \epsilon < 2$.
4. **Probability Measure Normalization:** $\sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid s) = 1$ (discrete) or $\int_{\mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid s) da = 1$ (continuous) for all parameter vectors $\boldsymbol{\theta}$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
In TRPO (Schulman et al., 2015), monotonic improvement is guaranteed by bounding the true return from below using Kakade & Langford's theorem: $\eta(\pi) \ge L_{\pi_{\text{old}}}(\pi) - C \cdot D_{\text{KL}}^{\max}(\pi_{\text{old}}, \pi)$, where $C = \frac{4 \epsilon_{\text{rew}} \gamma}{(1 - \gamma)^2}$.
PPO's clipping mechanism constructs a first-order, pointwise lower bound on the unclipped surrogate $r_t \hat{A}_t$:
- If an action yields positive advantage ($\hat{A}_t > 0$), making it more likely ($r_t > 1$) increases expected return. However, if the optimizer increases $r_t$ beyond $1 + \epsilon$, the surrogate reward is clamped to $(1 + \epsilon)\hat{A}_t$. The optimizer is forbidden from claiming credit for large distribution shifts.
- If an action yields negative advantage ($\hat{A}_t < 0$), decreasing its probability ($r_t < 1$) improves performance. The reward is clamped at $(1 - \epsilon)\hat{A}_t$ for $r_t < 1 - \epsilon$.
- Crucially, if an update makes the policy catastrophically *worse* (such as increasing the probability $r_t > 1$ of a bad action with $\hat{A}_t < 0$, or decreasing the probability $r_t < 1$ of an outstanding action with $\hat{A}_t > 0$), the clipping term is *larger* than the unclipped term. The outer $\min$ operator explicitly chooses the unclipped term, exposing the optimizer to the full unmitigated penalty!
Thus, $L^{\text{CLIP}}$ acts as a pessimistic floor: it discounts overly optimistic improvements while fully preserving negative penalties.

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

We evaluate the transition-level objective:
$$L_t^{\text{CLIP}}(r_t, \hat{A}_t) \triangleq \min\left( r_t \hat{A}_t, \quad \operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon)\hat{A}_t \right)$$
where by definition:
$$\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) \triangleq \begin{cases} 1 - \epsilon & \text{if } r_t < 1 - \epsilon \\ r_t & \text{if } 1 - \epsilon \le r_t \le 1 + \epsilon \\ 1 + \epsilon & \text{if } r_t > 1 + \epsilon \end{cases}$$

We partition the analysis into three exhaustive cases based on the sign of the estimated advantage $\hat{A}_t$:

**Case 1: Positive Advantage ($\hat{A}_t > 0$)**
Let $\hat{A}_t > 0$. For any real numbers $X$ and $Y$, multiplying by positive scalar $\hat{A}_t$ distributes through the minimum:
$$\min(X \hat{A}_t, Y \hat{A}_t) = \hat{A}_t \cdot \min(X, Y)$$
Setting $X = r_t$ and $Y = \operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon)$:
$$L_t^{\text{CLIP}}(r_t, \hat{A}_t) = \hat{A}_t \cdot \min\left( r_t, \quad \operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) \right)$$

We examine the three sub-intervals of $r_t \in [0, \infty)$:
1. *Sub-interval 1a: $0 \le r_t < 1 - \epsilon$.*
   Here, $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) = 1 - \epsilon$.
   Since $r_t < 1 - \epsilon$, we have:
   $$\min(r_t, 1 - \epsilon) = r_t$$
   Multiplying by $\hat{A}_t$:
   $$L_t^{\text{CLIP}} = r_t \hat{A}_t$$
2. *Sub-interval 1b: $1 - \epsilon \le r_t \le 1 + \epsilon$.*
   Here, $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) = r_t$.
   Therefore:
   $$\min(r_t, r_t) = r_t$$
   Multiplying by $\hat{A}_t$:
   $$L_t^{\text{CLIP}} = r_t \hat{A}_t$$
3. *Sub-interval 1c: $r_t > 1 + \epsilon$.*
   Here, $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) = 1 + \epsilon$.
   Since $r_t > 1 + \epsilon$, we have:
   $$\min(r_t, 1 + \epsilon) = 1 + \epsilon$$
   Multiplying by $\hat{A}_t > 0$:
   $$L_t^{\text{CLIP}} = (1 + \epsilon)\hat{A}_t$$
   Because $r_t > 1 + \epsilon$ and $\hat{A}_t > 0$, multiplying by $\hat{A}_t$ yields:
   $$r_t \hat{A}_t > (1 + \epsilon)\hat{A}_t \implies (1 + \epsilon)\hat{A}_t < r_t \hat{A}_t$$

Combining Sub-intervals 1a, 1b, and 1c for $\hat{A}_t > 0$:
$$L_t^{\text{CLIP}}(r_t, \hat{A}_t) = \begin{cases} r_t \hat{A}_t & \text{if } 0 \le r_t \le 1 + \epsilon \\ (1 + \epsilon)\hat{A}_t & \text{if } r_t > 1 + \epsilon \end{cases}$$
Because $(1 + \epsilon)\hat{A}_t < r_t \hat{A}_t$ on $(1 + \epsilon, \infty)$ and $L_t^{\text{CLIP}} = r_t \hat{A}_t$ on $[0, 1 + \epsilon]$, we have:
$$L_t^{\text{CLIP}}(r_t, \hat{A}_t) \le r_t \hat{A}_t \quad \forall r_t \ge 0 \quad (\text{for } \hat{A}_t > 0)$$
with strict equality if and only if $r_t \le 1 + \epsilon$.

---

**Case 2: Negative Advantage ($\hat{A}_t < 0$)**
Let $\hat{A}_t < 0$, so $\hat{A}_t = - |\hat{A}_t|$.
For any real numbers $X, Y \in \mathbb{R}$:
$$\min(X \hat{A}_t, Y \hat{A}_t) = \min(- X |\hat{A}_t|, - Y |\hat{A}_t|) = - \max(X |\hat{A}_t|, Y |\hat{A}_t|) = - |\hat{A}_t| \max(X, Y) = \hat{A}_t \max(X, Y)$$
Setting $X = r_t$ and $Y = \operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon)$:
$$L_t^{\text{CLIP}}(r_t, \hat{A}_t) = \hat{A}_t \cdot \max\left( r_t, \quad \operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) \right)$$

We evaluate the three sub-intervals of $r_t \in [0, \infty)$:
1. *Sub-interval 2a: $0 \le r_t < 1 - \epsilon$.*
   Here, $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) = 1 - \epsilon$.
   Since $r_t < 1 - \epsilon$, we have:
   $$\max(r_t, 1 - \epsilon) = 1 - \epsilon$$
   Multiplying by $\hat{A}_t < 0$:
   $$L_t^{\text{CLIP}} = (1 - \epsilon)\hat{A}_t$$
   Now compare this to the unclipped term $r_t \hat{A}_t$:
   Since $r_t < 1 - \epsilon$, multiplying both sides by negative scalar $\hat{A}_t < 0$ reverses the inequality:
   $$r_t \hat{A}_t > (1 - \epsilon)\hat{A}_t \implies (1 - \epsilon)\hat{A}_t < r_t \hat{A}_t$$
   Therefore:
   $$L_t^{\text{CLIP}} = (1 - \epsilon)\hat{A}_t < r_t \hat{A}_t$$
2. *Sub-interval 2b: $1 - \epsilon \le r_t \le 1 + \epsilon$.*
   Here, $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) = r_t$.
   Therefore:
   $$\max(r_t, r_t) = r_t$$
   Multiplying by $\hat{A}_t$:
   $$L_t^{\text{CLIP}} = r_t \hat{A}_t$$
3. *Sub-interval 2c: $r_t > 1 + \epsilon$.*
   Here, $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon) = 1 + \epsilon$.
   Since $r_t > 1 + \epsilon$, we have:
   $$\max(r_t, 1 + \epsilon) = r_t$$
   Multiplying by $\hat{A}_t$:
   $$L_t^{\text{CLIP}} = r_t \hat{A}_t$$

Combining Sub-intervals 2a, 2b, and 2c for $\hat{A}_t < 0$:
$$L_t^{\text{CLIP}}(r_t, \hat{A}_t) = \begin{cases} (1 - \epsilon)\hat{A}_t & \text{if } 0 \le r_t < 1 - \epsilon \\ r_t \hat{A}_t & \text{if } r_t \ge 1 - \epsilon \end{cases}$$
Because $(1 - \epsilon)\hat{A}_t < r_t \hat{A}_t$ on $[0, 1 - \epsilon)$ and $L_t^{\text{CLIP}} = r_t \hat{A}_t$ on $[1 - \epsilon, \infty)$, we have:
$$L_t^{\text{CLIP}}(r_t, \hat{A}_t) \le r_t \hat{A}_t \quad \forall r_t \ge 0 \quad (\text{for } \hat{A}_t < 0)$$
with strict equality if and only if $r_t \ge 1 - \epsilon$.

---

**Case 3: Zero Advantage ($\hat{A}_t = 0$)**
When $\hat{A}_t = 0$:
$$r_t \hat{A}_t = 0 \quad \text{and} \quad \operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon)\hat{A}_t = 0$$
$$L_t^{\text{CLIP}}(r_t, 0) = \min(0, 0) = 0 = r_t \hat{A}_t$$
Hence $L_t^{\text{CLIP}} \le r_t \hat{A}_t$ holds with equality.

---

**Global Pessimistic Lower Bound Property:**
Combining Cases 1, 2, and 3, we have proved that for all $r_t \ge 0$ and all $\hat{A}_t \in \mathbb{R}$:
$$L_t^{\text{CLIP}}(r_t(\boldsymbol{\theta}), \hat{A}_t) \le r_t(\boldsymbol{\theta}) \hat{A}_t \quad \text{pointwise everywhere}$$

Taking the empirical expectation $\hat{\mathbb{E}}_t [\cdot] = \frac{1}{T} \sum_{t=1}^T (\cdot)$ over any finite batch of rollout transitions preserves the inequality by linearity and monotonicity of expectation:
$$L^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(r_t(\boldsymbol{\theta}), \hat{A}_t) \right] \le \hat{\mathbb{E}}_t \left[ r_t(\boldsymbol{\theta}) \hat{A}_t \right] = L^{\text{CPI}}(\boldsymbol{\theta})$$
This completes the first-principles proof that the clipped surrogate objective is a global pessimistic lower bound on the unclipped conservative policy iteration objective. $\blacksquare$

---

#### Derivation 11.19.2: Subgradient Analysis of Clipped Objective and Zero-Gradient Saturation Region

```
====================================================================================================
DERIVATION 11.19.2: Subgradient Analysis of Clipped Objective and Zero-Gradient Saturation Region
====================================================================================================
Problem Statement:
Let θ ∈ ℝ^d parameterize a differentiable stochastic policy π_θ(a | s). Analyze the generalized
gradient and Clarke subdifferential ∂_θ L_t^{CLIP}(θ) with respect to the parameter vector θ.
Prove from first principles:
1. The directional gradient follows the chain rule:
    ∇_θ L_t^{CLIP}(θ) = (∂ L_t^{CLIP} / ∂ r_t) ∇_θ r_t(θ) = (∂ L_t^{CLIP} / ∂ r_t) [ ∇_θ π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t) ]
2. The partial derivative with respect to r_t satisfies:
    ∂ L_t^{CLIP} / ∂ r_t = { Â_t  if (Â_t > 0 and r_t < 1 + ϵ) or (Â_t < 0 and r_t > 1 - ϵ)
                           { 0    if (Â_t > 0 and r_t > 1 + ϵ) or (Â_t < 0 and r_t < 1 - ϵ)
3. Thus, when the ratio escapes the trust region in the direction of improvement, ∇_θ L_t^{CLIP}(θ) = 0.
4. When the ratio escapes in the direction of degradation (destructive drift), the full non-zero restoring
   gradient Â_t ∇_θ r_t(θ) is preserved.
5. Characterize the Clarke subdifferential at the boundary kinks r_t = 1 ± ϵ.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
Consider the transition-level clipped surrogate objective $L_t^{\text{CLIP}}(\boldsymbol{\theta})$. Because $L_t^{\text{CLIP}}$ is composed of the continuous non-linear functions $\min$ and $\operatorname{clip}$, it is continuous and locally Lipschitz, but non-differentiable along the hyperplanes where $r_t(\boldsymbol{\theta}) = 1 - \epsilon$ and $r_t(\boldsymbol{\theta}) = 1 + \epsilon$.
The mathematical goal is to:
1. Formulate the gradient $\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta})$ in all open regions of parameter space.
2. Prove that $\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}} = \mathbf{0}$ whenever the policy moves too far in the direction that exploits the advantage estimate.
3. Prove that $\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}} \neq \mathbf{0}$ if the policy moves in the direction that worsens the objective, creating an asymmetric restoring force.
4. Formally define Clarke's generalized gradient $\partial_C L_t^{\text{CLIP}}(\boldsymbol{\theta})$ at the non-differentiable boundaries $r_t = 1 \pm \epsilon$.

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Differentiable Policy Parametrization:** The stochastic policy $\pi_{\boldsymbol{\theta}}(a \mid s)$ is continuously differentiable ($C^1$) with respect to $\boldsymbol{\theta} \in \mathbb{R}^d$ for all $(s, a)$.
2. **Fixed Advantage during Policy Step:** In accordance with standard policy gradient formulations, the rollout advantage $\hat{A}_t$ is computed from rollout data prior to policy updates and is treated as an independent scalar constant with respect to $\boldsymbol{\theta}$ ($\nabla_{\boldsymbol{\theta}} \hat{A}_t = \mathbf{0}$).
3. **Locally Lipschitz Continuity:** The composite map $\boldsymbol{\theta} \mapsto L_t^{\text{CLIP}}(r_t(\boldsymbol{\theta}), \hat{A}_t)$ is locally Lipschitz continuous on $\mathbb{R}^d$, guaranteeing the existence of Clarke's generalized gradient $\partial_C L_t^{\text{CLIP}}(\boldsymbol{\theta}) \neq \emptyset$.
4. **Non-Zero Reference Density:** $\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t) > 0$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
In an unconstrained policy gradient algorithm (such as REINFORCE or standard Actor-Critic), the objective gradient is $\hat{A}_t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$. If $\hat{A}_t > 0$, the gradient continues pulling $\boldsymbol{\theta}$ in the direction of increasing $\pi_{\boldsymbol{\theta}}(a_t \mid s_t)$ indefinitely, even across multiple mini-batch epochs on the same data.
In PPO:
- As soon as the parameter update pushes $\pi_{\boldsymbol{\theta}}(a_t \mid s_t)$ above $(1 + \epsilon)\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)$, the objective enters a horizontal plateau of constant value $(1 + \epsilon)\hat{A}_t$. The directional derivative along the ascent trajectory becomes identically zero ($\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}} = \mathbf{0}$). The optimizer encounters a "frictionless ceiling" and halts updates for this transition.
- Similarly, if $\hat{A}_t < 0$, as soon as the policy suppresses the action below $(1 - \epsilon)\pi_{\boldsymbol{\theta}_{\text{old}}}$, the objective hits a horizontal floor of value $(1 - \epsilon)\hat{A}_t$. The gradient becomes $\mathbf{0}$, preventing the policy from extinguishing exploration entropy.
- Crucially, if destructive cross-sample gradient interference pushes a bad action ($\hat{A}_t < 0$) upward to $r_t > 1 + \epsilon$, the objective is $r_t \hat{A}_t$, which has negative slope $\hat{A}_t < 0$. The gradient does *not* vanish; instead, it strongly pushes the policy back downward!

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Chain rule decomposition.*
The scalar objective $L_t^{\text{CLIP}}$ depends on the parameter vector $\boldsymbol{\theta}$ exclusively through the scalar probability ratio:
$$r_t(\boldsymbol{\theta}) \triangleq \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$$
Applying the multivariate chain rule:
$$\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \frac{\partial L_t^{\text{CLIP}}}{\partial r_t} \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta})$$
We first evaluate the gradient of the ratio $\nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta})$:
$$\nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \left( \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \right) = \frac{1}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$$
Using the identity $\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}} = \pi_{\boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}$:
$$\nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) = r_t(\boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$$

*Step 2: Differentiating $L_t^{\text{CLIP}}$ with respect to $r_t$.*
From Derivation 11.19.1, the piecewise form of $L_t^{\text{CLIP}}(r_t)$ is:

**For $\hat{A}_t > 0$:**
$$L_t^{\text{CLIP}}(r_t) = \begin{cases} r_t \hat{A}_t & \text{if } 0 \le r_t \le 1 + \epsilon \\ (1 + \epsilon)\hat{A}_t & \text{if } r_t > 1 + \epsilon \end{cases}$$
Differentiating with respect to $r_t$ in the interior of each region:
- On $r_t \in (0, 1 + \epsilon)$:
  $$\frac{\partial L_t^{\text{CLIP}}}{\partial r_t} = \frac{d}{dr_t}(r_t \hat{A}_t) = \hat{A}_t$$
- On $r_t \in (1 + \epsilon, \infty)$:
  $$\frac{\partial L_t^{\text{CLIP}}}{\partial r_t} = \frac{d}{dr_t}\left( (1 + \epsilon)\hat{A}_t \right) = 0$$

**For $\hat{A}_t < 0$:**
$$L_t^{\text{CLIP}}(r_t) = \begin{cases} (1 - \epsilon)\hat{A}_t & \text{if } 0 \le r_t < 1 - \epsilon \\ r_t \hat{A}_t & \text{if } r_t \ge 1 - \epsilon \end{cases}$$
Differentiating with respect to $r_t$ in the interior of each region:
- On $r_t \in (0, 1 - \epsilon)$:
  $$\frac{\partial L_t^{\text{CLIP}}}{\partial r_t} = \frac{d}{dr_t}\left( (1 - \epsilon)\hat{A}_t \right) = 0$$
- On $r_t \in (1 - \epsilon, \infty)$:
  $$\frac{\partial L_t^{\text{CLIP}}}{\partial r_t} = \frac{d}{dr_t}(r_t \hat{A}_t) = \hat{A}_t$$

*Step 3: Evaluating parameter gradients $\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta})$.*
Substituting $\frac{\partial L_t^{\text{CLIP}}}{\partial r_t}$ into the chain rule $\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}} = \frac{\partial L_t^{\text{CLIP}}}{\partial r_t} \nabla_{\boldsymbol{\theta}} r_t$:

1. **Active Positive Update Zone ($\hat{A}_t > 0, \; r_t < 1 + \epsilon$):**
   $$\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$$
   The gradient is non-zero, pushing the parameter in the direction of the policy gradient.

2. **Positive Saturation Zone ($\hat{A}_t > 0, \; r_t > 1 + \epsilon$):**
   $$\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta}) = 0 \cdot \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) = \mathbf{0}$$
   The gradient vanishes completely! Further updates receive zero contribution from this transition.

3. **Active Negative Update Zone ($\hat{A}_t < 0, \; r_t > 1 - \epsilon$):**
   $$\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$$
   The gradient is active, driving down the probability of this suboptimal action.

4. **Negative Saturation Zone ($\hat{A}_t < 0, \; r_t < 1 - \epsilon$):**
   $$\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta}) = 0 \cdot \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) = \mathbf{0}$$
   The gradient vanishes completely! The policy cannot be penalized further on this sample.

5. **Asymmetric Restoring Zone ($\hat{A}_t < 0, \; r_t > 1 + \epsilon$):**
   Suppose a negative advantage sample undergoes destructive drift such that $r_t > 1 + \epsilon$.
   As derived in Step 2, for $\hat{A}_t < 0$ and $r_t > 1 - \epsilon$, the derivative is $\frac{\partial L_t^{\text{CLIP}}}{\partial r_t} = \hat{A}_t$.
   Since $1 + \epsilon > 1 - \epsilon$, the point $r_t > 1 + \epsilon$ belongs strictly to $(1 - \epsilon, \infty)$.
   Therefore:
   $$\nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) \neq \mathbf{0}$$
   The objective does NOT saturate at zero! Instead, it retains the full negative gradient $\hat{A}_t \nabla_{\boldsymbol{\theta}} r_t$, generating an aggressive restorative force that actively suppresses the bad action back down toward $1.0$.

*Step 4: Clarke generalized subdifferential at the non-differentiable boundaries.*
At the kink boundaries $r_t = 1 + \epsilon$ (when $\hat{A}_t > 0$) and $r_t = 1 - \epsilon$ (when $\hat{A}_t < 0$), the classical derivative does not exist.
By Clarke's subdifferential theorem for continuous piecewise $C^1$ functions, the generalized gradient $\partial_C f(\mathbf{x})$ is the convex hull of the limits of gradients of sequences of smooth points converging to $\mathbf{x}$:
- For $\hat{A}_t > 0$ at $r_t(\boldsymbol{\theta}) = 1 + \epsilon$:
  The left limit is $\lim_{r_t \to (1+\epsilon)^-} \nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}} = \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta})$.
  The right limit is $\lim_{r_t \to (1+\epsilon)^+} \nabla_{\boldsymbol{\theta}} L_t^{\text{CLIP}} = \mathbf{0}$.
  Therefore, Clarke's generalized subdifferential is:
  $$\partial_C L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \operatorname{conv}\left\{ \mathbf{0}, \; \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) \right\} = \left\{ \alpha \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) \;\middle|\; \alpha \in [0, 1] \right\}$$
- For $\hat{A}_t < 0$ at $r_t(\boldsymbol{\theta}) = 1 - \epsilon$:
  The left limit is $\mathbf{0}$, and the right limit is $\hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta})$.
  Therefore:
  $$\partial_C L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \operatorname{conv}\left\{ \mathbf{0}, \; \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) \right\} = \left\{ \alpha \hat{A}_t \nabla_{\boldsymbol{\theta}} r_t(\boldsymbol{\theta}) \;\middle|\; \alpha \in [0, 1] \right\}$$

In standard automatic differentiation libraries (PyTorch autograd, JAX, TensorFlow), conditional branching is implemented via `torch.where(r <= 1 + eps, ...)`, which selects $\alpha = 1$ at the boundary point $r_t = 1 + \epsilon$, and selects $\alpha = 1$ at $r_t = 1 - \epsilon$. $\blacksquare$

---

#### Derivation 11.19.3: Adaptive KL Penalty PPO

```
====================================================================================================
DERIVATION 11.19.3: Dual Formulation and Update Rule for Adaptive KL Penalty PPO
====================================================================================================
Problem Statement:
In Trust Region Policy Optimization, consider the constrained policy improvement problem:
    max_θ Ê_t [ r_t(θ) Â_t ]   subject to   D̄_KL(π_{θ_old} || π_θ) ≤ d_targ
where d_targ > 0 is a specified target KL divergence.
PPO's unconstrained alternative maximizes the penalized surrogate objective:
    L^{KLPEN}(θ; β) ≜ Ê_t [ r_t(θ) Â_t ] - β D̄_KL(π_{θ_old} || π_θ)
Prove from first principles:
1. The penalized surrogate arises as the Lagrangian relaxation of the constrained problem with dual multiplier β ≥ 0.
2. The heuristic adaptive update rule:
    d_k ≜ D̄_KL(π_{θ_old} || π_{θ_k})
    β_{k+1} = { β_k / 2      if d_k < d_targ / 1.5
              { β_k × 2      if d_k > d_targ × 1.5
              { β_k          otherwise
   is an exact discretization of dual gradient ascent in the unconstrained log-multiplier space ψ = ln β.
3. This multiplicative update guarantees strict positivity β_k > 0 for all k ≥ 0 without projection,
   and prevents limit-cycle chattering via the hysteresis deadband [d_targ / 1.5, 1.5 d_targ].
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
While the clipped objective $L^{\text{CLIP}}$ is PPO's most common variant, Schulman et al. (2017) also introduced an alternative formulation using an adaptive KL penalty coefficient $\beta$.
The primary challenge in penalty methods is choosing $\beta$:
- If $\beta$ is too large, the penalty dominates, the policy takes miniscule updates, and learning stalls.
- If $\beta$ is too small, the policy drifts far outside the trust region, causing catastrophic performance collapse.
- If a fixed $\beta$ is used, the policy update size varies wildly across different phases of training (e.g. initial random exploration vs. fine-tuning).
The goal is to derive from first principles:
1. The Lagrangian duality between the hard KL-constrained optimization and the penalized surrogate.
2. The exact mathematical derivation showing that the update rule $\beta_{k+1} \in \{\beta_k / 2, \beta_k \times 2, \beta_k\}$ is a log-dual gradient ascent step with a hysteresis deadband.

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Differentiable Parameterization & Information Regularity:** The policy $\pi_{\boldsymbol{\theta}}$ is twice continuously differentiable ($C^2$) in $\boldsymbol{\theta}$. The average KL divergence $\bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}})$ is strictly convex in a neighborhood around $\boldsymbol{\theta}_{\text{old}}$ with positive-definite Fisher Information Matrix $\mathbf{F} \succ \mathbf{0}$.
2. **Strictly Positive Target:** The target divergence satisfies $d_{\text{targ}} > 0$ (typically $d_{\text{targ}} \in [0.003, 0.03]$).
3. **Initial Multiplier:** $\beta_0 > 0$ (typically $\beta_0 = 1.0$).
4. **Hysteresis Factor:** The deadband parameter $\kappa > 1$ (Schulman et al. set $\kappa = 1.5$).

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
Imagine an elastic tether connecting the candidate policy $\pi_{\boldsymbol{\theta}}$ to the anchor policy $\pi_{\boldsymbol{\theta}_{\text{old}}}$. The stiffness of the tether is $\beta$.
- If the policy update moves very little ($d_k < d_{\text{targ}} / 1.5$), the tether is excessively rigid, suffocating policy improvement. Halving $\beta$ softens the tether, permitting larger exploratory steps in subsequent epochs.
- If the policy update moves too far ($d_k > 1.5 d_{\text{targ}}$), the tether is too slack, allowing dangerous policy drift. Doubling $\beta$ stiffens the tether, pulling the policy back toward the safe trust region.
- The deadband $[d_{\text{targ}} / 1.5, 1.5 d_{\text{targ}}]$ acts as a mechanical damper: when the divergence is within acceptable bounds, $\beta$ remains unchanged. Without this deadband, $\beta$ would constantly oscillate between doubling and halving (bang-bang chattering), destabilizing optimization.

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Lagrangian relaxation of the constrained trust region problem.*
The primal constrained optimization problem is:
$$\max_{\boldsymbol{\theta}} L^{\text{CPI}}(\boldsymbol{\theta}) \quad \text{subject to} \quad \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}}) \le d_{\text{targ}}$$
where $L^{\text{CPI}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ r_t(\boldsymbol{\theta}) \hat{A}_t \right]$ and $\bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}}) = \hat{\mathbb{E}}_t \left[ D_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}}(\cdot \mid s_t) \parallel \pi_{\boldsymbol{\theta}}(\cdot \mid s_t)) \right]$.

Rewriting as an equivalent minimization problem:
$$\min_{\boldsymbol{\theta}} \left( - L^{\text{CPI}}(\boldsymbol{\theta}) \right) \quad \text{subject to} \quad \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}}) - d_{\text{targ}} \le 0$$
Introducing the Lagrange multiplier $\beta \ge 0$, the Lagrangian function is:
$$\mathcal{L}(\boldsymbol{\theta}, \beta) \triangleq - L^{\text{CPI}}(\boldsymbol{\theta}) + \beta \left( \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}}) - d_{\text{targ}} \right)$$
Negating back to maximization form:
$$\max_{\boldsymbol{\theta}} \left[ L^{\text{CPI}}(\boldsymbol{\theta}) - \beta \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}}) \right] + \beta d_{\text{targ}}$$
For any fixed $\beta$, the term $+\beta d_{\text{targ}}$ does not depend on $\boldsymbol{\theta}$. Thus, maximizing the Lagrangian over $\boldsymbol{\theta}$ is algebraically identical to maximizing the penalized surrogate:
$$L^{\text{KLPEN}}(\boldsymbol{\theta}; \beta) \triangleq \hat{\mathbb{E}}_t \left[ r_t(\boldsymbol{\theta}) \hat{A}_t \right] - \beta \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}})$$

*Step 2: Dual function and subgradient.*
The Lagrange dual function $g(\beta)$ is defined as:
$$g(\beta) \triangleq \min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}, \beta) = \mathcal{L}(\boldsymbol{\theta}^*(\beta), \beta)$$
where $\boldsymbol{\theta}^*(\beta) \triangleq \arg\max_{\boldsymbol{\theta}} L^{\text{KLPEN}}(\boldsymbol{\theta}; \beta)$.
By Danskin's Theorem, because $\mathcal{L}(\boldsymbol{\theta}, \beta)$ is affine in $\beta$, the derivative of the dual function with respect to $\beta$ is simply the partial derivative evaluated at the optimal primal point:
$$\frac{dg}{d\beta} = \left. \frac{\partial \mathcal{L}}{\partial \beta} \right|_{\boldsymbol{\theta}^*(\beta)} = \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}^*(\beta)}) - d_{\text{targ}} = d(\beta) - d_{\text{targ}}$$
To maximize the concave dual function $g(\beta)$ over $\beta \ge 0$, standard dual gradient ascent applies the update:
$$\beta_{k+1} = \left[ \beta_k + \alpha_{\text{dual}} (d_k - d_{\text{targ}}) \right]_+$$
where $[x]_+ \triangleq \max(0, x)$.
Notice the sign of the dual gradient:
- If $d_k > d_{\text{targ}}$: The KL divergence exceeded the budget. The dual gradient $(d_k - d_{\text{targ}}) > 0$ is positive, so $\beta$ must increase to penalize divergence more heavily.
- If $d_k < d_{\text{targ}}$: The KL divergence is well within budget. The dual gradient $(d_k - d_{\text{targ}}) < 0$ is negative, so $\beta$ must decrease to allow larger steps.

*Step 3: Reparameterization in log-dual space.*
In practice, standard additive dual gradient ascent has two fatal flaws:
1. It requires an artificial projection $[\cdot]_+$ to prevent $\beta \le 0$. If $\beta = 0$, the penalty vanishes entirely and the policy immediately destabilizes.
2. The divergence $d(\beta)$ scales inversely with $\beta^2$ (from the second-order Taylor expansion $d \approx \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \approx \frac{1}{2\beta^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}$). An additive step size $\alpha_{\text{dual}}$ cannot maintain stable step magnitudes across different scales of $\beta$.

To resolve this, we reparameterize the multiplier in **logarithmic space**:
$$\beta = \exp(\psi) \iff \psi = \ln \beta \quad \text{where } \psi \in (-\infty, \infty)$$
Because $\exp(\psi) > 0$ for all $\psi \in \mathbb{R}$, strict positivity $\beta > 0$ is guaranteed without projection!

Applying the chain rule to compute the gradient of the dual function with respect to $\psi$:
$$\frac{dg}{d\psi} = \frac{dg}{d\beta} \cdot \frac{d\beta}{d\psi} = (d_k - d_{\text{targ}}) \cdot \exp(\psi) = \beta_k (d_k - d_{\text{targ}})$$

Performing gradient ascent in the unconstrained $\psi$-space with step size $\eta_\psi$:
$$\psi_{k+1} = \psi_k + \eta_\psi \cdot \operatorname{sign}(d_k - d_{\text{targ}})$$
Exponentiating both sides:
$$\beta_{k+1} = \exp(\psi_{k+1}) = \exp\left( \psi_k + \eta_\psi \operatorname{sign}(d_k - d_{\text{targ}}) \right) = \exp(\psi_k) \cdot \exp\left( \eta_\psi \operatorname{sign}(d_k - d_{\text{targ}}) \right)$$
$$\beta_{k+1} = \beta_k \cdot \left( e^{\eta_\psi} \right)^{\operatorname{sign}(d_k - d_{\text{targ}})}$$

Setting the log step size to $\eta_\psi = \ln 2 \approx 0.693147$:
$$e^{\eta_\psi} = e^{\ln 2} = 2.0$$
Then:
- If $d_k > d_{\text{targ}}$: $\operatorname{sign}(d_k - d_{\text{targ}}) = +1 \implies \beta_{k+1} = \beta_k \cdot 2^{+1} = \beta_k \times 2$.
- If $d_k < d_{\text{targ}}$: $\operatorname{sign}(d_k - d_{\text{targ}}) = -1 \implies \beta_{k+1} = \beta_k \cdot 2^{-1} = \beta_k / 2$.

*Step 4: Derivation of the hysteresis deadband.*
If $\beta$ updates whenever $d_k \neq d_{\text{targ}}$, small stochastic fluctuations in mini-batch sampling will cause $\beta$ to alternate between doubling and halving at every iteration.
To enforce asymptotic Lyapunov stability, we define an indifference deadband around $d_{\text{targ}}$ parameterized by hysteresis factor $\kappa = 1.5$:
$$\mathcal{I}_{\text{stable}} \triangleq \left[ \frac{d_{\text{targ}}}{\kappa}, \; \kappa \cdot d_{\text{targ}} \right] = \left[ \frac{d_{\text{targ}}}{1.5}, \; 1.5 \cdot d_{\text{targ}} \right]$$
The three operational regimes become:
$$\beta_{k+1} = \begin{cases}
\beta_k / 2 & \text{if } d_k < d_{\text{targ}} / 1.5 \quad (\text{constraint too tight; encourage larger updates}) \\
\beta_k \times 2 & \text{if } d_k > 1.5 \cdot d_{\text{targ}} \quad (\text{constraint violated; enforce strict conservatism}) \\
\beta_k & \text{if } d_{\text{targ}} / 1.5 \le d_k \le 1.5 \cdot d_{\text{targ}} \quad (\text{stable trust region; maintain current penalty})
\end{cases}$$

This completes the first-principles derivation of PPO's adaptive KL penalty update rule as an exact discretization of log-dual gradient ascent. $\blacksquare$

---

## 3. Geometric & Physical Interpretation: The Safety Plateaus

In the 1D slice along any parameter direction:
- Unconstrained policy gradients form an unbounded linear slope: $\nabla_\theta J$ pulls the parameters uphill forever.
- PPO's clipping operator introduces **two horizontal plateaus** that bracket the objective:
  - An upper ceiling at $(1 + \epsilon)\hat{A}$ for good actions.
  - A lower floor at $(1 - \epsilon)\hat{A}$ for bad actions.
- As soon as the parameter vector reaches the edge of the trust region ($|r_t - 1| > \epsilon$), the objective flattens completely. The directional derivative vanishes ($\nabla L = \mathbf{0}$), acting like a **frictionless brake** that stops the optimizer from drifting outside the trust zone.

```
       Surrogate Objective
              ^
              |       /-------------- Upper Plateau (grad = 0)
              |      /
              |     /   Active Slope (grad > 0)
              |    /
  ------------+---/------------------> r_t
   (grad = 0) |  1-eps   1.0   1+eps
```

---

## 4. Real-World Analogy: The Speed Governor on a Performance Car

Imagine driving a supercar equipped with an intelligent speed governor:
- You are trying to maintain the optimal racing speed on a dangerous mountain track.
- If you are moving at $90\%$ of optimal speed, the throttle responds linearly to your foot, helping you accelerate.
- If you press the accelerator too aggressively and hit $121\%$ of optimal speed ($r > 1.20$), the governor smoothly cuts engine power to zero ($\nabla_\theta = 0$). You cannot accelerate into a fatal crash, no matter how hard you press the pedal!
- Conversely, if you hit a patch of ice and slide below $80\%$ speed, the system regulates brake pressure to prevent the wheels from locking.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: The Four Canonical Cases of PPO-Clip
To understand every branch of the min-clip operator down to the exact decimal digit, let us evaluate a mini-batch of 4 transitions with clip parameter $\epsilon = 0.2000 \implies [1 - \epsilon, 1 + \epsilon] = [0.8000, 1.2000]$:

- **Sample 1 (Positive Advantage, Moderate Ratio):**
  Advantage $\hat{A}_1 = +2.0000$, Ratio $r_1 = 1.1000$ (Action improved moderately).
- **Sample 2 (Positive Advantage, Excessive Ratio):**
  Advantage $\hat{A}_2 = +2.0000$, Ratio $r_2 = 1.3500$ (Action probability exploded!).
- **Sample 3 (Negative Advantage, Moderate Ratio):**
  Advantage $\hat{A}_3 = -1.5000$, Ratio $r_3 = 0.9000$ (Action suppressed moderately).
- **Sample 4 (Negative Advantage, Excessive Ratio):**
  Advantage $\hat{A}_4 = -1.5000$, Ratio $r_4 = 0.6500$ (Action probability collapsed!).

We will compute:
1. The unclipped objective: $T_{\text{unclip}} = r \hat{A}$
2. The clipped ratio: $\tilde{r} = \operatorname{clip}(r, 0.8, 1.2)$
3. The clipped objective: $T_{\text{clip}} = \tilde{r} \hat{A}$
4. The final PPO objective: $L^{\text{CLIP}} = \min(T_{\text{unclip}}, T_{\text{clip}})$
5. The effective gradient signal $\frac{\partial L^{\text{CLIP}}}{\partial r}$ (Active vs Clipped).

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Walkthrough Meaning |
| :--- | :--- | :--- |
| $\epsilon$ | Clipping Threshold | $\epsilon = 0.2000 \implies [0.8000, 1.2000]$ |
| $r_t$ | Probability Ratio | $\frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$ |
| $\hat{A}_t$ | Estimated Advantage | $\hat{A} > 0$ (good action), $\hat{A} < 0$ (bad action) |
| $T_{\text{unclip}}$ | Unclipped Surrogate | $r_t \cdot \hat{A}_t$ |
| $\tilde{r}_t$ | Clipped Ratio | $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon)$ |
| $T_{\text{clip}}$ | Clipped Surrogate | $\tilde{r}_t \cdot \hat{A}_t$ |
| $L^{\text{CLIP}}$ | PPO-Clip Value | $\min(T_{\text{unclip}}, T_{\text{clip}})$ |
| Gradient | Status of $\frac{\partial L}{\partial r}$ | $\hat{A}$ if unclipped; $0.0$ if on plateau |

---

### 5.3 Step-by-Step Hand Calculations: All 4 Cases

#### Sample 1: $\hat{A}_1 = +2.0000, \quad r_1 = 1.1000$
1. Unclipped term:
   $$T_{\text{unclip}} = r_1 \hat{A}_1 = 1.1000 \times (+2.0000) = \mathbf{+2.2000}$$
2. Clipped ratio:
   $$\tilde{r}_1 = \operatorname{clip}(1.1000, 0.8000, 1.2000) = \mathbf{1.1000} \quad (\text{Not clipped})$$
3. Clipped term:
   $$T_{\text{clip}} = 1.1000 \times (+2.0000) = \mathbf{+2.2000}$$
4. Final Objective:
   $$L_1^{\text{CLIP}} = \min(+2.2000, +2.2000) = \mathbf{+2.2000}$$
5. Gradient: $\frac{\partial L}{\partial r_1} = \hat{A}_1 = \mathbf{+2.0000}$ (**ACTIVE GRADIENT** $\checkmark$).

#### Sample 2: $\hat{A}_2 = +2.0000, \quad r_2 = 1.3500$
1. Unclipped term:
   $$T_{\text{unclip}} = r_2 \hat{A}_2 = 1.3500 \times (+2.0000) = \mathbf{+2.7000}$$
2. Clipped ratio:
   $$\tilde{r}_2 = \operatorname{clip}(1.3500, 0.8000, 1.2000) = \mathbf{1.2000} \quad (\text{CLIPPED!})$$
3. Clipped term:
   $$T_{\text{clip}} = 1.2000 \times (+2.0000) = \mathbf{+2.4000}$$
4. Final Objective:
   $$L_2^{\text{CLIP}} = \min(+2.7000, +2.4000) = \mathbf{+2.4000}$$
5. Gradient: The $\min$ chose $T_{\text{clip}}$, which is constant with respect to $r$ ($r > 1.2$).
   $$\frac{\partial L}{\partial r_2} = \mathbf{0.0000} \quad (\textbf{CLIPPED TO ZERO! Safe plateau reached})$$

#### Sample 3: $\hat{A}_3 = -1.5000, \quad r_3 = 0.9000$
1. Unclipped term:
   $$T_{\text{unclip}} = r_3 \hat{A}_3 = 0.9000 \times (-1.5000) = \mathbf{-1.3500}$$
2. Clipped ratio:
   $$\tilde{r}_3 = \operatorname{clip}(0.9000, 0.8000, 1.2000) = \mathbf{0.9000} \quad (\text{Not clipped})$$
3. Clipped term:
   $$T_{\text{clip}} = 0.9000 \times (-1.5000) = \mathbf{-1.3500}$$
4. Final Objective:
   $$L_3^{\text{CLIP}} = \min(-1.3500, -1.3500) = \mathbf{-1.3500}$$
5. Gradient: $\frac{\partial L}{\partial r_3} = \hat{A}_3 = \mathbf{-1.5000}$ (**ACTIVE GRADIENT** $\checkmark$).

#### Sample 4: $\hat{A}_4 = -1.5000, \quad r_4 = 0.6500$
1. Unclipped term:
   $$T_{\text{unclip}} = r_4 \hat{A}_4 = 0.6500 \times (-1.5000) = \mathbf{-0.9750}$$
2. Clipped ratio:
   $$\tilde{r}_4 = \operatorname{clip}(0.6500, 0.8000, 1.2000) = \mathbf{0.8000} \quad (\text{CLIPPED!})$$
3. Clipped term:
   $$T_{\text{clip}} = 0.8000 \times (-1.5000) = \mathbf{-1.2000}$$
4. Final Objective:
   $$L_4^{\text{CLIP}} = \min(-0.9750, -1.2000) = \mathbf{-1.2000}$$
   *Crucial Observation:* Because $\hat{A}_4 < 0$, $-1.2000 < -0.9750$, so the $\min$ correctly picks the lower (more pessimistic) value $-1.2000$!
5. Gradient: Because the $\min$ selected $T_{\text{clip}}$ where $\tilde{r} = 0.80$ is fixed:
   $$\frac{\partial L}{\partial r_4} = \mathbf{0.0000} \quad (\textbf{CLIPPED TO ZERO! Avoids excessive penalty})$$

---

### 5.4 Summary Visual Grid: PPO-Clip Decision Matrix

| Sample | Advantage $\hat{A}$ | Ratio $r$ | Unclipped $r\hat{A}$ | Clipped $\tilde{r}\hat{A}$ | Final $L^{\text{CLIP}}$ | Selected by Min | Gradient $\frac{\partial L}{\partial r}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $+2.0000$ | $1.1000$ | $+2.2000$ | $+2.2000$ | $\mathbf{+2.2000}$ | Both | $\mathbf{+2.0000}$ (Active) |
| **2** | $+2.0000$ | $1.3500$ | $+2.7000$ | $+2.4000$ | $\mathbf{+2.4000}$ | Clipped | $\mathbf{0.0000}$ (Clipped) |
| **3** | $-1.5000$ | $0.9000$ | $-1.3500$ | $-1.3500$ | $\mathbf{-1.3500}$ | Both | $\mathbf{-1.5000}$ (Active) |
| **4** | $-1.5000$ | $0.6500$ | $-0.9750$ | $-1.2000$ | $\mathbf{-1.2000}$ | Clipped | $\mathbf{0.0000}$ (Clipped) |

**Batch Mean PPO-Clip Objective:**
$$\bar{L}^{\text{CLIP}} = \frac{2.2000 + 2.4000 - 1.3500 - 1.2000}{4} = \frac{2.0500}{4} = \mathbf{+0.5125}$$

Every single arithmetic operation matches PyTorch execution to exact machine precision!

---

## 6. Solved Illustrations

### Illustration 1: Why the $\min$ Operator is Mandatory: Concrete Counterexamples and Safety Failure Analysis

**Problem Statement:**
A common question in reinforcement learning is: *Why can't we simply define the clipped surrogate objective as $\tilde{L}(\boldsymbol{\theta}) = \operatorname{clip}(r_t(\boldsymbol{\theta}), 1 - \epsilon, 1 + \epsilon) \hat{A}_t$ without the outer $\min$ operator?*
Provide concrete numerical counterexamples proving that removing the outer $\min$ operator leads to two catastrophic failure modes:
1. Complete loss of gradient signal when a catastrophic policy update dramatically worsens performance on a bad action.
2. Complete loss of gradient signal when an update severely over-suppresses a good action.

**Numerical Solution & Analysis:**

Let the clipping threshold be $\epsilon = 0.2000$, establishing the trust interval $[1 - \epsilon, 1 + \epsilon] = [0.8000, 1.2000]$.

#### Counterexample 1: Catastrophic Over-Selection of a Bad Action ($\hat{A}_t < 0, r_t > 1 + \epsilon$)
Suppose an action performed very poorly during rollouts, with estimated advantage:
$$\hat{A}_t = -10.0000$$
During multi-epoch mini-batch training, suppose cross-sample gradient interference causes the policy to mistakenly *increase* the probability of this disastrous action, resulting in an exploded probability ratio:
$$r_t(\boldsymbol{\theta}) = 2.0000 \quad (\text{The action became twice as likely!})$$

Let us evaluate the candidate objectives:
1. **Unclipped Surrogate ($T_{\text{unclip}}$):**
   $$T_{\text{unclip}} = r_t \hat{A}_t = 2.0000 \times (-10.0000) = \mathbf{-20.0000}$$
2. **Clipped Surrogate without $\min$ ($\tilde{L}$):**
   $$\tilde{r}_t = \operatorname{clip}(2.0000, 0.8000, 1.2000) = 1.2000$$
   $$\tilde{L}(\boldsymbol{\theta}) = \tilde{r}_t \hat{A}_t = 1.2000 \times (-10.0000) = \mathbf{-12.0000}$$
3. **PPO-Clip with $\min$ ($L^{\text{CLIP}}$):**
   $$L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \min(T_{\text{unclip}}, \tilde{L}) = \min(-20.0000, -12.0000) = \mathbf{-20.0000}$$

Now examine the resulting gradient signals with respect to the probability ratio $r_t$:
- **Under naive clipping $\tilde{L}$:**
  Because $r_t = 2.0000 > 1.2000$, $\tilde{r}_t$ is pegged to the constant $1.2000$ in this neighborhood.
  $$\frac{\partial \tilde{L}}{\partial r_t} = \frac{d}{dr_t}\left( 1.2000 \times (-10.0000) \right) = \mathbf{0.0000}$$
  **Catastrophic Failure:** The naive objective treats the disastrous policy update as a flat plateau! The optimizer receives zero gradient, meaning the algorithm **cannot detect or undo the catastrophic mistake**!
- **Under PPO-Clip $L^{\text{CLIP}}$:**
  Because $-20.0000 < -12.0000$, the $\min$ operator selects the unclipped branch $r_t \hat{A}_t$:
  $$\frac{\partial L^{\text{CLIP}}}{\partial r_t} = \frac{d}{dr_t}(r_t \hat{A}_t) = \hat{A}_t = \mathbf{-10.0000} \neq \mathbf{0.0000}$$
  **Restoration Success:** The gradient is active, large, and negative! It delivers an immediate, powerful corrective pull that forces the policy parameters back down.

---

#### Counterexample 2: Catastrophic Suppression of an Outstanding Action ($\hat{A}_t > 0, r_t < 1 - \epsilon$)
Suppose an action was exceptionally lucrative:
$$\hat{A}_t = +5.0000$$
Due to off-policy parameter drift, the policy inadvertently collapses the probability of this action:
$$r_t(\boldsymbol{\theta}) = 0.5000 \quad (\text{The action probability dropped by half!})$$

Let us evaluate the candidate objectives:
1. **Unclipped Surrogate ($T_{\text{unclip}}$):**
   $$T_{\text{unclip}} = r_t \hat{A}_t = 0.5000 \times (+5.0000) = \mathbf{+2.5000}$$
2. **Clipped Surrogate without $\min$ ($\tilde{L}$):**
   $$\tilde{r}_t = \operatorname{clip}(0.5000, 0.8000, 1.2000) = 0.8000$$
   $$\tilde{L}(\boldsymbol{\theta}) = \tilde{r}_t \hat{A}_t = 0.8000 \times (+5.0000) = \mathbf{+4.0000}$$
3. **PPO-Clip with $\min$ ($L^{\text{CLIP}}$):**
   $$L_t^{\text{CLIP}}(\boldsymbol{\theta}) = \min(T_{\text{unclip}}, \tilde{L}) = \min(+2.5000, +4.0000) = \mathbf{+2.5000}$$

Examining gradients:
- Under naive clipping $\tilde{L}$, $\tilde{r}_t = 0.8000$ is constant for all $r_t < 0.8000$, so $\frac{\partial \tilde{L}}{\partial r_t} = \mathbf{0.0000}$. The optimizer is completely blind to the fact that an outstanding action was suppressed.
- Under PPO-Clip, the $\min$ chooses $+2.5000$, preserving the active positive gradient $\frac{\partial L^{\text{CLIP}}}{\partial r_t} = \mathbf{+5.0000}$, which immediately restores the probability of the good action.

#### Summary Comparison:

| Scenario | $\hat{A}_t$ | $r_t$ | Naive $\tilde{L} = \operatorname{clip}(r)\hat{A}$ | Naive Gradient $\frac{\partial \tilde{L}}{\partial r}$ | PPO $L^{\text{CLIP}} = \min(r\hat{A}, \operatorname{clip}\hat{A})$ | PPO Gradient $\frac{\partial L^{\text{CLIP}}}{\partial r}$ | Safety Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exploded Bad Action** | $-10.0000$ | $2.0000$ | $-12.0000$ | $\mathbf{0.0000}$ (Frozen!) | $\mathbf{-20.0000}$ | $\mathbf{-10.0000}$ (Restoring) | Naive fails; PPO recovers $\checkmark$ |
| **Suppressed Good Action** | $+5.0000$ | $0.5000$ | $+4.0000$ | $\mathbf{0.0000}$ (Frozen!) | $\mathbf{+2.5000}$ | $\mathbf{+5.0000}$ (Restoring) | Naive fails; PPO recovers $\checkmark$ |

The outer $\min$ operator is therefore **mathematically mandatory** to prevent asymmetric policy collapse. $\blacksquare$

---

### Illustration 2: Step-by-Step Evaluation of $L^{\text{CLIP}}$ Across 4 Regime Cases: Positive vs. Negative Advantage, Clipped vs. Unclipped

**Problem Statement:**
Let the clipping hyperparameter be $\epsilon = 0.2000 \implies [1 - \epsilon, 1 + \epsilon] = [0.8000, 1.2000]$.
Perform complete step-by-step arithmetic to evaluate the PPO-Clip objective, clipping status, active branch, and subgradient for each of the four canonical operating regimes:
- **Sample 1:** $\hat{A}_1 = +2.0000, \quad r_1 = 1.1000$ (Positive advantage, inside trust region)
- **Sample 2:** $\hat{A}_2 = +2.0000, \quad r_2 = 1.3500$ (Positive advantage, outside trust region)
- **Sample 3:** $\hat{A}_3 = -1.5000, \quad r_3 = 0.9000$ (Negative advantage, inside trust region)
- **Sample 4:** $\hat{A}_4 = -1.5000, \quad r_4 = 0.6500$ (Negative advantage, outside trust region)

Compute the batch mean objective $\bar{L}^{\text{CLIP}}$ and verify all subgradients.

**Step-by-Step Arithmetic:**

#### Sample 1: Positive Advantage, Unclipped ($\hat{A}_1 = +2.0000, r_1 = 1.1000$)
1. **Unclipped surrogate term:**
   $$T_{\text{unclip}}^{(1)} = r_1 \hat{A}_1 = 1.1000 \times (+2.0000) = \mathbf{+2.2000}$$
2. **Clipped ratio:**
   $$\tilde{r}_1 = \operatorname{clip}(1.1000, 0.8000, 1.2000) = \mathbf{1.1000} \quad (\text{Since } 0.8000 \le 1.1000 \le 1.2000)$$
3. **Clipped surrogate term:**
   $$T_{\text{clip}}^{(1)} = \tilde{r}_1 \hat{A}_1 = 1.1000 \times (+2.0000) = \mathbf{+2.2000}$$
4. **PPO-Clip objective:**
   $$L_1^{\text{CLIP}} = \min\left( T_{\text{unclip}}^{(1)}, T_{\text{clip}}^{(1)} \right) = \min(+2.2000, +2.2000) = \mathbf{+2.2000}$$
5. **Subgradient evaluation:**
   Since $r_1 = 1.1000 < 1.2000$ and $\hat{A}_1 > 0$, by Derivation 11.19.2:
   $$\frac{\partial L_1^{\text{CLIP}}}{\partial r_1} = \hat{A}_1 = \mathbf{+2.0000} \quad (\textbf{Active Gradient Signal})$$

---

#### Sample 2: Positive Advantage, Clipped ($\hat{A}_2 = +2.0000, r_2 = 1.3500$)
1. **Unclipped surrogate term:**
   $$T_{\text{unclip}}^{(2)} = r_2 \hat{A}_2 = 1.3500 \times (+2.0000) = \mathbf{+2.7000}$$
2. **Clipped ratio:**
   $$\tilde{r}_2 = \operatorname{clip}(1.3500, 0.8000, 1.2000) = \mathbf{1.2000} \quad (\text{Capped at upper ceiling } 1 + \epsilon)$$
3. **Clipped surrogate term:**
   $$T_{\text{clip}}^{(2)} = \tilde{r}_2 \hat{A}_2 = 1.2000 \times (+2.0000) = \mathbf{+2.4000}$$
4. **PPO-Clip objective:**
   $$L_2^{\text{CLIP}} = \min\left( T_{\text{unclip}}^{(2)}, T_{\text{clip}}^{(2)} \right) = \min(+2.7000, +2.4000) = \mathbf{+2.4000}$$
5. **Subgradient evaluation:**
   Because $r_2 = 1.3500 > 1.2000$, the minimum selected $T_{\text{clip}}^{(2)}$ which is invariant to local perturbations in $r_2$:
   $$\frac{\partial L_2^{\text{CLIP}}}{\partial r_2} = \frac{d}{dr_2}(+2.4000) = \mathbf{0.0000} \quad (\textbf{Zero-Gradient Safety Plateau})$$

---

#### Sample 3: Negative Advantage, Unclipped ($\hat{A}_3 = -1.5000, r_3 = 0.9000$)
1. **Unclipped surrogate term:**
   $$T_{\text{unclip}}^{(3)} = r_3 \hat{A}_3 = 0.9000 \times (-1.5000) = \mathbf{-1.3500}$$
2. **Clipped ratio:**
   $$\tilde{r}_3 = \operatorname{clip}(0.9000, 0.8000, 1.2000) = \mathbf{0.9000} \quad (\text{Since } 0.8000 \le 0.9000 \le 1.2000)$$
3. **Clipped surrogate term:**
   $$T_{\text{clip}}^{(3)} = \tilde{r}_3 \hat{A}_3 = 0.9000 \times (-1.5000) = \mathbf{-1.3500}$$
4. **PPO-Clip objective:**
   $$L_3^{\text{CLIP}} = \min\left( T_{\text{unclip}}^{(3)}, T_{\text{clip}}^{(3)} \right) = \min(-1.3500, -1.3500) = \mathbf{-1.3500}$$
5. **Subgradient evaluation:**
   Since $r_3 = 0.9000 \ge 0.8000$ and $\hat{A}_3 < 0$, by Derivation 11.19.2:
   $$\frac{\partial L_3^{\text{CLIP}}}{\partial r_3} = \hat{A}_3 = \mathbf{-1.5000} \quad (\textbf{Active Gradient Signal})$$

---

#### Sample 4: Negative Advantage, Clipped ($\hat{A}_4 = -1.5000, r_4 = 0.6500$)
1. **Unclipped surrogate term:**
   $$T_{\text{unclip}}^{(4)} = r_4 \hat{A}_4 = 0.6500 \times (-1.5000) = \mathbf{-0.9750}$$
2. **Clipped ratio:**
   $$\tilde{r}_4 = \operatorname{clip}(0.6500, 0.8000, 1.2000) = \mathbf{0.8000} \quad (\text{Floored at lower boundary } 1 - \epsilon)$$
3. **Clipped surrogate term:**
   $$T_{\text{clip}}^{(4)} = \tilde{r}_4 \hat{A}_4 = 0.8000 \times (-1.5000) = \mathbf{-1.2000}$$
4. **PPO-Clip objective:**
   $$L_4^{\text{CLIP}} = \min\left( T_{\text{unclip}}^{(4)}, T_{\text{clip}}^{(4)} \right) = \min(-0.9750, -1.2000) = \mathbf{-1.2000}$$
   *Crucial Detail:* Because $-1.2000 < -0.9750$, the outer $\min$ correctly selects the lower (more pessimistic) value $-1.2000$!
5. **Subgradient evaluation:**
   Because $r_4 = 0.6500 < 0.8000$ and $\hat{A}_4 < 0$, the objective is pinned to $(1 - \epsilon)\hat{A}_4 = -1.2000$:
   $$\frac{\partial L_4^{\text{CLIP}}}{\partial r_4} = \frac{d}{dr_4}(-1.2000) = \mathbf{0.0000} \quad (\textbf{Zero-Gradient Safety Plateau})$$

---

#### Batch Summary & Mean Objective:
$$\bar{L}^{\text{CLIP}} = \frac{1}{4} \sum_{i=1}^4 L_i^{\text{CLIP}} = \frac{+2.2000 + 2.4000 - 1.3500 - 1.2000}{4} = \frac{+2.0500}{4} = \mathbf{+0.5125}$$

| Sample $i$ | $\hat{A}_i$ | $r_i$ | $T_{\text{unclip}} = r \hat{A}$ | $\tilde{r} = \operatorname{clip}(r)$ | $T_{\text{clip}} = \tilde{r}\hat{A}$ | Selected by $\min$ | $L_i^{\text{CLIP}}$ | Subgradient $\frac{\partial L}{\partial r_i}$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | $+2.0000$ | $1.1000$ | $+2.2000$ | $1.1000$ | $+2.2000$ | Both | $\mathbf{+2.2000}$ | $\mathbf{+2.0000}$ | Active $\checkmark$ |
| **2** | $+2.0000$ | $1.3500$ | $+2.7000$ | $1.2000$ | $+2.4000$ | $T_{\text{clip}}$ | $\mathbf{+2.4000}$ | $\mathbf{0.0000}$ | Clipped Plateau $\checkmark$ |
| **3** | $-1.5000$ | $0.9000$ | $-1.3500$ | $0.9000$ | $-1.3500$ | Both | $\mathbf{-1.3500}$ | $\mathbf{-1.5000}$ | Active $\checkmark$ |
| **4** | $-1.5000$ | $0.6500$ | $-0.9750$ | $0.8000$ | $-1.2000$ | $T_{\text{clip}}$ | $\mathbf{-1.2000}$ | $\mathbf{0.0000}$ | Clipped Plateau $\checkmark$ |

Every single intermediate value matches exact analytical arithmetic and PyTorch autograd tensors. $\blacksquare$

---

### Illustration 3: Complete PPO-Clip Mini-batch Backward Pass with Generalized Advantage Estimation on 4 Transitions

**Problem Statement:**
Consider a mini-batch of 4 consecutive episodic transitions $t \in \{0, 1, 2, 3\}$ terminating at step $T = 4$ ($V(S_4) \triangleq 0$).
Let the discount factor be $\gamma = 1.0000$, GAE parameter $\lambda = 0.5000$ ($\gamma \lambda = 0.5000$), clipping threshold $\epsilon = 0.2000$, critic loss coefficient $c_1 = 0.5000$, and entropy bonus coefficient $c_2 = 0.0100$.
The recorded rollout data from reference policy $\pi_{\boldsymbol{\theta}_{\text{old}}}$ is:
- **Rewards:** $R_1 = 0.0000, \; R_2 = +4.0000, \; R_3 = -3.0000, \; R_4 = +1.0000$
- **Critic Baseline Values:** $V(S_0) = 1.0000, \; V(S_1) = 2.0000, \; V(S_2) = 1.0000, \; V(S_3) = 3.0000, \; V(S_4) = 0.0000$
- **Old Action Probabilities:** $\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid S_t) = [0.4000, 0.5000, 0.6000, 0.5000]$
- **Updated Action Probabilities:** $\pi_{\boldsymbol{\theta}}(a_t \mid S_t) = [0.4400, 0.6500, 0.5400, 0.3500]$
- **Current Critic Predictions:** $V_{\boldsymbol{\phi}}(S_t) = [1.2000, 2.3000, 0.8000, 2.8000]$

Execute the complete end-to-end PPO forward and backward pass:
1. Compute 1-step TD errors $\delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$.
2. Compute unnormalized GAE advantages $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ via backward recursion.
3. Standardize advantages across the mini-batch: $\hat{A}_t = \frac{\hat{A}_t^{\text{GAE}} - \mu_A}{\sigma_A}$.
4. Compute critic target values: $V_t^{\text{targ}} = \hat{A}_t^{\text{GAE}} + V(S_t)$.
5. Compute probability ratios $r_t(\boldsymbol{\theta})$ and binary clipping masks $M_t \in \{0, 1\}$.
6. Compute clipped surrogate policy loss $L_t^{\text{CLIP}}$ and batch mean $\bar{L}^{\text{CLIP}}$.
7. Compute value function loss $L_t^{\text{VF}} = (V_{\boldsymbol{\phi}}(S_t) - V_t^{\text{targ}})^2$ and batch mean $\bar{L}^{\text{VF}}$.
8. Compute policy entropy bonus $\mathcal{H}_t = - \sum_a \pi(a \mid S_t) \ln \pi(a \mid S_t)$ for binary actions and batch mean $\bar{\mathcal{H}}$.
9. Compute total multi-task loss $\mathcal{L}^{\text{PPO}} = \bar{L}^{\text{CLIP}} - c_1 \bar{L}^{\text{VF}} + c_2 \bar{\mathcal{H}}$.
10. Compute exact backpropagation gradients $\frac{\partial \mathcal{L}^{\text{PPO}}}{\partial r_t}$ and $\frac{\partial \mathcal{L}^{\text{PPO}}}{\partial V(S_t)}$.

---

**Step-by-Step Numerical Solution:**

#### Step 1: 1-Step TD Errors ($\delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ with $\gamma = 1.0$)
- $t = 0$: $\delta_0 = 0.0000 + 1.0 \times 2.0000 - 1.0000 = \mathbf{+1.0000}$
- $t = 1$: $\delta_1 = 4.0000 + 1.0 \times 1.0000 - 2.0000 = \mathbf{+3.0000}$
- $t = 2$: $\delta_2 = -3.0000 + 1.0 \times 3.0000 - 1.0000 = \mathbf{-1.0000}$
- $t = 3$: $\delta_3 = 1.0000 + 1.0 \times 0.0000 - 3.0000 = \mathbf{-2.0000}$

---

#### Step 2: Backward GAE Recursion ($\hat{A}_t^{\text{GAE}} = \delta_t + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}$ with $\gamma \lambda = 0.5000$)
Unrolling backward from terminal step $t = 3$:
- $t = 3$: $\hat{A}_3^{\text{GAE}} = \delta_3 = \mathbf{-2.0000}$
- $t = 2$: $\hat{A}_2^{\text{GAE}} = \delta_2 + 0.5000 \times \hat{A}_3^{\text{GAE}} = -1.0000 + 0.5000(-2.0000) = -1.0000 - 1.0000 = \mathbf{-2.0000}$
- $t = 1$: $\hat{A}_1^{\text{GAE}} = \delta_1 + 0.5000 \times \hat{A}_2^{\text{GAE}} = +3.0000 + 0.5000(-2.0000) = +3.0000 - 1.0000 = \mathbf{+2.0000}$
- $t = 0$: $\hat{A}_0^{\text{GAE}} = \delta_0 + 0.5000 \times \hat{A}_1^{\text{GAE}} = +1.0000 + 0.5000(+2.0000) = +1.0000 + 1.0000 = \mathbf{+2.0000}$

---

#### Step 3: Advantage Normalization
- Mean:
  $$\mu_A = \frac{+2.0000 + 2.0000 - 2.0000 - 2.0000}{4} = \frac{0.0000}{4} = \mathbf{0.0000}$$
- Variance:
  $$\sigma_A^2 = \frac{(+2 - 0)^2 + (+2 - 0)^2 + (-2 - 0)^2 + (-2 - 0)^2}{4} = \frac{4 + 4 + 4 + 4}{4} = \frac{16}{4} = \mathbf{4.0000}$$
- Standard Deviation:
  $$\sigma_A = \sqrt{4.0000} = \mathbf{2.0000}$$
- Normalized Advantages $\hat{A}_t = \frac{\hat{A}_t^{\text{GAE}} - \mu_A}{\sigma_A}$:
  $$\hat{A}_0 = \frac{+2.0000}{2.0000} = \mathbf{+1.0000}, \quad \hat{A}_1 = \frac{+2.0000}{2.0000} = \mathbf{+1.0000}$$
  $$\hat{A}_2 = \frac{-2.0000}{2.0000} = \mathbf{-1.0000}, \quad \hat{A}_3 = \frac{-2.0000}{2.0000} = \mathbf{-1.0000}$$

---

#### Step 4: Critic Value Targets ($V_t^{\text{targ}} = \hat{A}_t^{\text{GAE}} + V(S_t)$)
- $t = 0$: $V_0^{\text{targ}} = +2.0000 + 1.0000 = \mathbf{+3.0000}$
- $t = 1$: $V_1^{\text{targ}} = +2.0000 + 2.0000 = \mathbf{+4.0000}$
- $t = 2$: $V_2^{\text{targ}} = -2.0000 + 1.0000 = \mathbf{-1.0000}$
- $t = 3$: $V_3^{\text{targ}} = -2.0000 + 3.0000 = \mathbf{+1.0000}$

---

#### Step 5: Probability Ratios and Clipping Masks ($[1 - \epsilon, 1 + \epsilon] = [0.8000, 1.2000]$)
- $t = 0$: $r_0 = \frac{0.4400}{0.4000} = \mathbf{1.1000} \in [0.8, 1.2] \implies \tilde{r}_0 = 1.1000, \; M_0 = 1$ (Unclipped)
- $t = 1$: $r_1 = \frac{0.6500}{0.5000} = \mathbf{1.3000} > 1.2000 \implies \tilde{r}_1 = 1.2000, \; M_1 = 0$ (Clipped!)
- $t = 2$: $r_2 = \frac{0.5400}{0.6000} = \mathbf{0.9000} \in [0.8, 1.2] \implies \tilde{r}_2 = 0.9000, \; M_2 = 1$ (Unclipped)
- $t = 3$: $r_3 = \frac{0.3500}{0.5000} = \mathbf{0.7000} < 0.8000 \implies \tilde{r}_3 = 0.8000, \; M_3 = 0$ (Clipped!)

---

#### Step 6: Clipped Policy Loss ($L_t^{\text{CLIP}} = \min(r_t \hat{A}_t, \tilde{r}_t \hat{A}_t)$)
- $t = 0$: $\min(1.1000 \times 1.0, 1.1000 \times 1.0) = \min(+1.1000, +1.1000) = \mathbf{+1.1000}$
- $t = 1$: $\min(1.3000 \times 1.0, 1.2000 \times 1.0) = \min(+1.3000, +1.2000) = \mathbf{+1.2000}$
- $t = 2$: $\min(0.9000 \times (-1.0), 0.9000 \times (-1.0)) = \min(-0.9000, -0.9000) = \mathbf{-0.9000}$
- $t = 3$: $\min(0.7000 \times (-1.0), 0.8000 \times (-1.0)) = \min(-0.7000, -0.8000) = \mathbf{-0.8000}$
- **Batch Mean Policy Loss:**
  $$\bar{L}^{\text{CLIP}} = \frac{+1.1000 + 1.2000 - 0.9000 - 0.8000}{4} = \frac{+0.6000}{4} = \mathbf{+0.1500}$$

---

#### Step 7: Value Function Loss ($L_t^{\text{VF}} = (V_{\boldsymbol{\phi}}(S_t) - V_t^{\text{targ}})^2$)
Given current predictions $V_{\boldsymbol{\phi}} = [1.2000, 2.3000, 0.8000, 2.8000]$ and targets $V^{\text{targ}} = [3.0, 4.0, -1.0, 1.0]$:
- $t = 0$: $(1.2000 - 3.0000)^2 = (-1.8000)^2 = \mathbf{3.2400}$
- $t = 1$: $(2.3000 - 4.0000)^2 = (-1.7000)^2 = \mathbf{2.8900}$
- $t = 2$: $(0.8000 - (-1.0000))^2 = (+1.8000)^2 = \mathbf{3.2400}$
- $t = 3$: $(2.8000 - 1.0000)^2 = (+1.8000)^2 = \mathbf{3.2400}$
- **Batch Mean Value Loss:**
  $$\bar{L}^{\text{VF}} = \frac{3.2400 + 2.8900 + 3.2400 + 3.2400}{4} = \frac{12.6100}{4} = \mathbf{3.1525}$$

---

#### Step 8: Policy Entropy Bonus ($\mathcal{H}(p) = - [p \ln p + (1 - p) \ln (1 - p)]$)
For binary action probabilities $p_t = [0.4400, 0.6500, 0.5400, 0.3500]$:
- $t = 0$: $- [0.44 \ln 0.44 + 0.56 \ln 0.56] = - [0.44(-0.82098) + 0.56(-0.57982)] = 0.36123 + 0.32470 = \mathbf{0.6859}$
- $t = 1$: $- [0.65 \ln 0.65 + 0.35 \ln 0.35] = - [0.65(-0.43078) + 0.35(-1.04982)] = 0.28001 + 0.36744 = \mathbf{0.6474}$
- $t = 2$: $- [0.54 \ln 0.54 + 0.46 \ln 0.46] = - [0.54(-0.61619) + 0.46(-0.77653)] = 0.33274 + 0.35720 = \mathbf{0.6899}$
- $t = 3$: $- [0.35 \ln 0.35 + 0.65 \ln 0.65] = \mathbf{0.6474}$
- **Batch Mean Entropy:**
  $$\bar{\mathcal{H}} = \frac{0.6859 + 0.6474 + 0.6899 + 0.6474}{4} = \frac{2.6706}{4} = \mathbf{0.6677}$$

---

#### Step 9: Combined Multi-Task PPO Objective
$$\mathcal{L}^{\text{PPO}} = \bar{L}^{\text{CLIP}} - c_1 \bar{L}^{\text{VF}} + c_2 \bar{\mathcal{H}}$$
Substituting $\bar{L}^{\text{CLIP}} = +0.1500$, $c_1 = 0.5000$, $\bar{L}^{\text{VF}} = 3.1525$, $c_2 = 0.0100$, and $\bar{\mathcal{H}} = 0.6677$:
$$\mathcal{L}^{\text{PPO}} = 0.1500 - (0.5000 \times 3.1525) + (0.0100 \times 0.6677)$$
$$\mathcal{L}^{\text{PPO}} = 0.1500 - 1.57625 + 0.006677 = \mathbf{-1.4196}$$

---

#### Step 10: Backpropagation Gradients
1. **Policy Ratio Gradients $\frac{\partial \bar{L}^{\text{CLIP}}}{\partial r_t} = \frac{1}{4} \frac{\partial L_t^{\text{CLIP}}}{\partial r_t}$:**
   - $t = 0$: $M_0 = 1 \implies \frac{\partial L_0^{\text{CLIP}}}{\partial r_0} = \hat{A}_0 = +1.0000 \implies \frac{\partial \bar{L}^{\text{CLIP}}}{\partial r_0} = \frac{+1.0000}{4} = \mathbf{+0.2500}$
   - $t = 1$: $M_1 = 0$ (Clipped!) $\implies \frac{\partial L_1^{\text{CLIP}}}{\partial r_1} = 0.0000 \implies \frac{\partial \bar{L}^{\text{CLIP}}}{\partial r_1} = \mathbf{0.0000}$
   - $t = 2$: $M_2 = 1 \implies \frac{\partial L_2^{\text{CLIP}}}{\partial r_2} = \hat{A}_2 = -1.0000 \implies \frac{\partial \bar{L}^{\text{CLIP}}}{\partial r_2} = \frac{-1.0000}{4} = \mathbf{-0.2500}$
   - $t = 3$: $M_3 = 0$ (Clipped!) $\implies \frac{\partial L_3^{\text{CLIP}}}{\partial r_3} = 0.0000 \implies \frac{\partial \bar{L}^{\text{CLIP}}}{\partial r_3} = \mathbf{0.0000}$

2. **Critic Prediction Gradients $\frac{\partial (- c_1 \bar{L}^{\text{VF}})}{\partial V(S_t)}$:**
   Since $\bar{L}^{\text{VF}} = \frac{1}{4} \sum_t (V(S_t) - V_t^{\text{targ}})^2$, we have $\frac{\partial \bar{L}^{\text{VF}}}{\partial V(S_t)} = \frac{2}{4} (V(S_t) - V_t^{\text{targ}}) = \frac{1}{2}(V(S_t) - V_t^{\text{targ}})$:
   - $t = 0$: $\frac{\partial \bar{L}^{\text{VF}}}{\partial V_0} = 0.5(1.2 - 3.0) = -0.9000 \implies \nabla_{V_0}(- c_1 \bar{L}^{\text{VF}}) = -0.5(-0.9000) = \mathbf{+0.4500}$
   - $t = 1$: $\frac{\partial \bar{L}^{\text{VF}}}{\partial V_1} = 0.5(2.3 - 4.0) = -0.8500 \implies \nabla_{V_1}(- c_1 \bar{L}^{\text{VF}}) = -0.5(-0.8500) = \mathbf{+0.4250}$
   - $t = 2$: $\frac{\partial \bar{L}^{\text{VF}}}{\partial V_2} = 0.5(0.8 - (-1.0)) = +0.9000 \implies \nabla_{V_2}(- c_1 \bar{L}^{\text{VF}}) = -0.5(+0.9000) = \mathbf{-0.4500}$
   - $t = 3$: $\frac{\partial \bar{L}^{\text{VF}}}{\partial V_3} = 0.5(2.8 - 1.0) = +0.9000 \implies \nabla_{V_3}(- c_1 \bar{L}^{\text{VF}}) = -0.5(+0.9000) = \mathbf{-0.4500}$

This gives the exact backward pass computed by PyTorch autograd! $\blacksquare$

---

### Illustration 4: Adaptive KL Penalty Parameter Update Trace over 3 Consecutive Policy Epochs

**Problem Statement:**
In the Adaptive KL Penalty variant of PPO (Derivation 11.19.3), the penalty parameter $\beta$ is dynamically adjusted between policy rollout epochs based on the target KL divergence $d_{\text{targ}} = 0.0100$ and hysteresis factor $\kappa = 1.5000$.
Let the initial penalty multiplier be $\beta_0 = 1.0000$.
Across three consecutive training epochs, the empirical average KL divergence measurements between the old policy and the updated policy are:
- **Epoch 0:** $d_0 = \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}_0}) = 0.0040$
- **Epoch 1:** $d_1 = \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}_1}) = 0.0250$
- **Epoch 2:** $d_2 = \bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \parallel \pi_{\boldsymbol{\theta}_2}) = 0.0090$

Calculate the decision boundaries, state condition checks, and exact updated multiplier values $\beta_1, \beta_2, \beta_3$. Provide the engineering rationale for each adaptation.

---

**Step-by-Step Numerical Solution:**

#### Step 1: Establish Decision Thresholds
Given target divergence $d_{\text{targ}} = 0.0100$:
1. **Lower Trigger Boundary (Undershooting Threshold):**
   $$d_{\text{lower}} \triangleq \frac{d_{\text{targ}}}{1.5} = \frac{0.0100}{1.5} = \frac{1}{150} \approx \mathbf{0.006667}$$
2. **Upper Trigger Boundary (Overshooting Threshold):**
   $$d_{\text{upper}} \triangleq 1.5 \times d_{\text{targ}} = 1.5 \times 0.0100 = \mathbf{0.015000}$$
3. **Hysteresis Deadband Window:**
   $$\mathcal{I}_{\text{deadband}} = [0.006667, \; 0.015000]$$

---

#### Step 2: Epoch 0 Adaptation ($d_0 = 0.0040$)
- **Condition Check:**
  $$d_0 = 0.0040 < 0.006667 = d_{\text{lower}}$$
- **Rule Triggered:** $d_0 < d_{\text{targ}} / 1.5$ (Constraint excessively tight).
- **Multiplier Update:**
  $$\beta_1 = \frac{\beta_0}{2} = \frac{1.0000}{2} = \mathbf{0.5000}$$
- **Physical Rationale:**
  The policy divergence $d_0 = 0.0040$ was less than half the target $0.0100$. The penalty coefficient $\beta_0 = 1.0$ was too rigid, suffocating policy updates and causing under-exploration. Halving the penalty to $\beta_1 = 0.5000$ reduces the tether tension by $50\%$, encouraging the optimizer to take bolder improvement steps during the next epoch.

---

#### Step 3: Epoch 1 Adaptation ($d_1 = 0.0250$)
- **Condition Check:**
  $$d_1 = 0.0250 > 0.015000 = d_{\text{upper}}$$
- **Rule Triggered:** $d_1 > 1.5 \times d_{\text{targ}}$ (Trust region violated!).
- **Multiplier Update:**
  $$\beta_2 = \beta_1 \times 2 = 0.5000 \times 2 = \mathbf{1.0000}$$
- **Physical Rationale:**
  Following the relaxation to $\beta_1 = 0.5$, the optimizer took an excessively large step, producing $d_1 = 0.0250$, which is $2.5\times$ the target $d_{\text{targ}}$ and exceeds the safety limit $0.0150$. The policy drifted into an uncalibrated distribution. Doubling the penalty to $\beta_2 = 1.0000$ immediately stiffens the tether, pulling subsequent updates back inside the trust region.

---

#### Step 4: Epoch 2 Adaptation ($d_2 = 0.0090$)
- **Condition Check:**
  $$0.006667 \le d_2 = 0.0090 \le 0.015000$$
- **Rule Triggered:** $d_2 \in [d_{\text{targ}}/1.5, 1.5 d_{\text{targ}}]$ (Divergence within target zone).
- **Multiplier Update:**
  $$\beta_3 = \beta_2 = \mathbf{1.0000} \quad (\text{Unchanged})$$
- **Physical Rationale:**
  The observed divergence $0.0090$ lies comfortably inside the hysteresis deadband, differing from $d_{\text{targ}} = 0.0100$ by only $10\%$. The deadband dampens updates, preventing high-frequency bang-bang oscillations. The penalty $\beta = 1.0000$ is preserved.

---

#### Comprehensive Epoch Adaptation Trace:

| Epoch $k$ | Starting Multiplier $\beta_k$ | Measured $d_k$ | Lower Bound $\frac{d_{\text{targ}}}{1.5}$ | Upper Bound $1.5 d_{\text{targ}}$ | Condition Status | Update Applied | Resulting Multiplier $\beta_{k+1}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $1.0000$ | $0.0040$ | $0.006667$ | $0.015000$ | $d_0 < d_{\text{lower}}$ | $\beta / 2$ | $\mathbf{0.5000}$ |
| **1** | $0.5000$ | $0.0250$ | $0.006667$ | $0.015000$ | $d_1 > d_{\text{upper}}$ | $\beta \times 2$ | $\mathbf{1.0000}$ |
| **2** | $1.0000$ | $0.0090$ | $0.006667$ | $0.015000$ | In Deadband | None | $\mathbf{1.0000}$ |

All numbers verified to exact decimal precision. $\blacksquare$

---

### Illustration 5: Multi-Epoch Gradient Reuse and Policy Collapse Prevention under Destructive Off-Policy Drift

**Problem Statement:**
In deep RL, sample efficiency is maximized by running multiple gradient descent epochs (e.g. $K = 5$) on the same mini-batch of rollout transitions.
Demonstrate mathematically why unconstrained Conservative Policy Iteration (CPI) suffers from catastrophic policy collapse under multi-epoch reuse, and how PPO-Clip's zero-gradient plateau physically prevents policy drift.

Consider a 1-parameter logistic policy:
$$\pi_\theta(a = 1) = \sigma(\theta) = \frac{1}{1 + e^{-\theta}}$$
Initially, $\theta_{\text{old}} = 0.0000$, so $\pi_{\theta_{\text{old}}}(a = 1) = \sigma(0) = 0.5000$.
Rollout experience recorded action $a = 1$ with positive advantage $\hat{A} = +2.0000$.
Let learning rate be $\eta = 0.4000$ and clipping parameter $\epsilon = 0.2000 \implies 1 + \epsilon = 1.2000$.
Trace 5 consecutive mini-batch gradient ascent steps on this transition for:
1. **Algorithm A (Unconstrained CPI):** Maximizing $L^{\text{CPI}}(\theta) = \frac{\pi_\theta(a=1)}{\pi_{\text{old}}(a=1)} \hat{A} = 2.0 \hat{A} \sigma(\theta)$.
2. **Algorithm B (PPO-Clip):** Maximizing $L^{\text{CLIP}}(\theta) = \min(r(\theta)\hat{A}, \operatorname{clip}(r(\theta), 0.8, 1.2)\hat{A})$.

---

**Mathematical Formulations:**
For logistic sigmoid $\sigma(\theta)$, the derivative is $\frac{d\sigma}{d\theta} = \sigma(\theta)(1 - \sigma(\theta))$.
- The probability ratio is:
  $$r(\theta) = \frac{\sigma(\theta)}{0.5000} = 2.0 \sigma(\theta)$$
- The unconstrained CPI gradient is:
  $$\nabla_\theta L^{\text{CPI}} = \frac{1}{0.5000} \hat{A} \frac{d\sigma}{d\theta} = 2.0 \times 2.0 \times \sigma(\theta)(1 - \sigma(\theta)) = 4.0 \sigma(\theta)(1 - \sigma(\theta))$$
- The PPO-Clip gradient is:
  $$\nabla_\theta L^{\text{CLIP}} = \begin{cases} 4.0 \sigma(\theta)(1 - \sigma(\theta)) & \text{if } r(\theta) \le 1.2000 \\ 0.0000 & \text{if } r(\theta) > 1.2000 \end{cases}$$
- Parameter update rule: $\theta_{m+1} = \theta_m + \eta \nabla_\theta L$.

---

**Step-by-Step Multi-Epoch Trace:**

#### Step 0 (Initial State):
- $\theta_0 = 0.0000$
- $\sigma(\theta_0) = \frac{1}{1 + e^0} = 0.5000$
- $r_0 = 2.0 \times 0.5000 = 1.0000 \le 1.2000$ (Both algorithms active!)
- $\nabla_\theta L = 4.0 \times 0.5000 \times 0.5000 = \mathbf{1.0000}$
- Update for both: $\theta_1 = 0.0000 + 0.4000 \times 1.0000 = \mathbf{0.4000}$

#### Step 1:
- $\theta_1 = 0.4000$
- $\sigma(\theta_1) = \frac{1}{1 + e^{-0.4000}} = \frac{1}{1 + 0.67032} = \mathbf{0.5987}$
- $r_1 = 2.0 \times 0.5987 = \mathbf{1.1974} \le 1.2000$ (Both algorithms still active!)
- $\nabla_\theta L = 4.0 \times 0.5987 \times (1 - 0.5987) = 4.0 \times 0.5987 \times 0.4013 = \mathbf{0.9610}$
- Update for both: $\theta_2 = 0.4000 + 0.4000 \times 0.9610 = 0.4000 + 0.3844 = \mathbf{0.7844}$

#### Step 2:
- $\theta_2 = 0.7844$
- $\sigma(\theta_2) = \frac{1}{1 + e^{-0.7844}} = \frac{1}{1 + 0.45639} = \mathbf{0.6866}$
- $r_2 = 2.0 \times 0.6866 = \mathbf{1.3733} > 1.2000$!
- **Branching divergence:**
  - **Algorithm A (CPI):** $r_2$ is ignored.
    $$\nabla_\theta L^{\text{CPI}} = 4.0 \times 0.6866 \times (1 - 0.6866) = 4.0 \times 0.6866 \times 0.3134 = \mathbf{0.8607}$$
    $$\theta_3^{\text{CPI}} = 0.7844 + 0.4000 \times 0.8607 = 0.7844 + 0.3443 = \mathbf{1.1287}$$
  - **Algorithm B (PPO-Clip):** Because $r_2 = 1.3733 > 1.2000$, clipping activates!
    $$\nabla_\theta L^{\text{CLIP}} = \mathbf{0.0000} \quad (\textbf{Zero-Gradient Safety Plateau reached})$$
    $$\theta_3^{\text{PPO}} = 0.7844 + 0.4000 \times 0.0000 = \mathbf{0.7844} \quad (\textbf{Parameter frozen!})$$

#### Steps 3, 4, 5:
- **Algorithm A (CPI):** Continues pushing parameters without constraint:
  - Step 3: $\theta_3 = 1.1287 \implies \sigma(\theta_3) = 0.7556, r_3 = 1.5112, \nabla_\theta = 0.7387 \implies \theta_4 = \mathbf{1.4242}$
  - Step 4: $\theta_4 = 1.4242 \implies \sigma(\theta_4) = 0.8060, r_4 = 1.6120, \nabla_\theta = 0.6255 \implies \theta_5 = \mathbf{1.6744}$
  - Final CPI probability: $\pi_{\theta_5}(a = 1) = \mathbf{0.8422}$. Ratio exploded to $r = \mathbf{1.6843}$ ($+68.4\%$ drift).
- **Algorithm B (PPO-Clip):**
  - Because $r(\theta_2) = 1.3733 > 1.2000$, the gradient remains identically $\mathbf{0.0000}$ for all subsequent steps!
  - $\theta_3 = \theta_4 = \theta_5 = \mathbf{0.7844}$.
  - Final PPO probability: $\pi(a = 1) = \mathbf{0.6866}$. Ratio safely capped at $r = \mathbf{1.3733}$.

---

#### Comparative Performance Trajectory:

| Step $m$ | CPI $\theta_m$ | CPI $\pi(a=1)$ | CPI Ratio $r_m$ | CPI $\nabla_\theta$ | PPO $\theta_m$ | PPO $\pi(a=1)$ | PPO Ratio $r_m$ | PPO $\nabla_\theta$ | PPO Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | $0.0000$ | $0.5000$ | $1.0000$ | $1.0000$ | $0.0000$ | $0.5000$ | $1.0000$ | $1.0000$ | Active |
| **1** | $0.4000$ | $0.5987$ | $1.1974$ | $0.9610$ | $0.4000$ | $0.5987$ | $1.1974$ | $0.9610$ | Active |
| **2** | $0.7844$ | $0.6866$ | $1.3733$ | $0.8607$ | $\mathbf{0.7844}$ | $\mathbf{0.6866}$ | $\mathbf{1.3733}$ | $\mathbf{0.0000}$ | Clipped Plateau! $\checkmark$ |
| **3** | $1.1287$ | $0.7556$ | $1.5112$ | $0.7387$ | $\mathbf{0.7844}$ | $\mathbf{0.6866}$ | $\mathbf{1.3733}$ | $\mathbf{0.0000}$ | Clipped Plateau! $\checkmark$ |
| **4** | $1.4242$ | $0.8060$ | $1.6120$ | $0.6255$ | $\mathbf{0.7844}$ | $\mathbf{0.6866}$ | $\mathbf{1.3733}$ | $\mathbf{0.0000}$ | Clipped Plateau! $\checkmark$ |
| **5** | $\mathbf{1.6744}$ | $\mathbf{0.8422}$ | $\mathbf{1.6843}$ | $0.5317$ | $\mathbf{0.7844}$ | $\mathbf{0.6866}$ | $\mathbf{1.3733}$ | $\mathbf{0.0000}$ | Clipped Plateau! $\checkmark$ |

**Conclusion:**
Unconstrained CPI drives the policy dangerously toward a deterministic point mass ($\pi \to 0.8422$), extinguishing policy entropy and causing severe off-policy sampling mismatch.
PPO-Clip detects that the ratio crossed the safety threshold at Step 2 and freezes the gradient completely, preserving exploration and guaranteeing stable multi-epoch mini-batch training. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. The Core of ChatGPT / RLHF (InstructGPT - Ouyang et al., NeurIPS 2022)
In Reinforcement Learning from Human Feedback (RLHF), an LLM generates a response $y$ to prompt $x$. The reward model outputs scalar score $R(x, y)$.
The PPO objective aligns the LLM while adding a token-level KL penalty from the base SFT model:
$$\mathcal{L}_{\text{RLHF}}(\boldsymbol{\theta}) = \mathbb{E}_{(x, y)} \left[ r_t(\boldsymbol{\theta}) \hat{A}_t - \beta D_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}}) \right]$$
PPO-Clip ensures the LLM's token distribution does not drift into hallucination or gibberish.

### 2. DeepSeek-R1 Predecessor: PPO vs GRPO
Before inventing Group Relative Policy Optimization (GRPO - Chapter 11.28), DeepSeek initially aligned its reasoning models using PPO with GAE. GRPO maintains the exact same PPO-Clip surrogate loss:
$$\min\left( \frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}, \operatorname{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) \hat{A} \right)$$
proving that the PPO-Clip operator remains foundational even in frontier reasoning models!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of the Part 5 hand calculations:
   - Evaluates all 4 cases of PPO-Clip
   - Verifies clipped values $L_1 = 2.20, L_2 = 2.40, L_3 = -1.35, L_4 = -1.20$
   - Computes batch mean $+0.5125$ and verifies autograd gradients matching machine precision to $< 10^{-14}$.
2. Complete, production-grade PPO Actor-Critic agent with GAE, multi-epoch mini-batch SGD, and entropy regularization tested on Inverted Pendulum.

See implementation in:
[`11_reinforcement_learning/code/19_proximal_policy_optimization.py`](./code/19_proximal_policy_optimization.py)
