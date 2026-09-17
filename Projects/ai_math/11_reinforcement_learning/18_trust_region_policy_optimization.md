# Module 11.18: Trust Region Policy Optimization (TRPO)

---

## 1. Intuition & 101 Motivation

In Chapter 11.17, we saw that **Natural Policy Gradients (NPG)** solve the fundamental coordinate-dependency problem of standard policy gradients by taking steps bounded in distribution space:
$$\bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}) \le \delta$$
yielding the optimal step direction $\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g}$.

However, applying NPG to modern deep neural networks with millions of parameters ($d \ge 10^6$) presents a catastrophic computational barrier:
- Forming the Fisher Information Matrix $\mathbf{F} \in \mathbb{R}^{d \times d}$ requires storing $10^6 \times 10^6 = 10^{12}$ floating-point numbers ($\approx 4 \text{ Terabytes}$ of GPU RAM!).
- Inverting $\mathbf{F}$ via Gaussian elimination requires $\mathcal{O}(d^3) \approx 10^{18}$ floating-point operations, taking weeks for a single gradient step.

In 2015, John Schulman et al. introduced **Trust Region Policy Optimization (TRPO)**, which made natural policy gradients scalable, practical, and robust for deep networks:
1. **The Surrogate Advantage Objective:** Uses importance sampling ratios to formulate an objective that can be evaluated on off-policy rollouts.
2. **Conjugate Gradient (CG) with Hessian-Vector Products:** Computes the step $\mathbf{x} \approx \mathbf{F}^{-1} \mathbf{g}$ in $k \le 10$ iterations without ever forming, storing, or inverting the Fisher matrix $\mathbf{F}$!
3. **Monotonic Improvement Guarantee:** Guarantees that every policy update increases expected return $J(\pi)$.
4. **Backtracking Line Search:** Verifies that the quadratic Taylor approximation didn't violate the non-linear trust region boundary.

```
+-----------------------------------------------------------------------------------------+
|                                    TRPO PIPELINE                                        |
|                                                                                         |
|   Rollout Trajectories ----> Compute Surrogate Gradient g = grad L(theta)               |
|                                     |                                                   |
|                                     v                                                   |
|                 [ Conjugate Gradient Solver: F x = g ]                                  |
|                 (Uses Hessian-Vector Products: O(d) memory!)                            |
|                                     |                                                   |
|                                     v                                                   |
|                 Search Direction: s = sqrt(2 * delta / x^T F x) * x                     |
|                                     |                                                   |
|                                     v                                                   |
|                 [ Backtracking Line Search: theta + alpha^j * s ]                       |
|                 (Accept step if: L(theta_new) >= 0 AND KL <= delta)                     |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Performance Difference Lemma (Kakade & Langford, 2002)

Let $\pi$ and $\tilde{\pi}$ be any two arbitrary policies. The difference in their expected returns satisfies:

$$J(\tilde{\pi}) - J(\pi) = \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t A^\pi(S_t, A_t) \right] = \sum_{s \in \mathcal{S}} d^{\tilde{\pi}}(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a)$$

*Interpretation:* The expected return of a new policy $\tilde{\pi}$ equals the return of the old policy $\pi$ plus the expected advantage of $\tilde{\pi}$'s actions evaluated under $\tilde{\pi}$'s state visitation distribution $d^{\tilde{\pi}}$.

---

### 2.2 The Surrogate Objective & The Monotonic Improvement Bound

The performance difference formula depends on $d^{\tilde{\pi}}(s)$, which is unknown because we have not yet run $\tilde{\pi}$ in the environment.
Schulman et al. define the **Surrogate Objective** $L_\pi(\tilde{\pi})$ by substituting the known state distribution $d^\pi(s)$ of the old policy:

$$L_\pi(\tilde{\pi}) \triangleq J(\pi) + \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a)$$

Using importance sampling, this can be estimated empirically from rollouts collected under policy $\theta_{\text{old}}$:
$$L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) = \mathbb{E}_{s \sim d^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a \mid s)} \hat{A}^{\pi_{\text{old}}}(s, a) \right]$$

Notice that at $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}$:
$$L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{old}}) = 0, \quad \left. \nabla_{\boldsymbol{\theta}} L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) \right|_{\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}} = \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_{\text{old}})$$
The surrogate objective matches the true return $J(\boldsymbol{\theta})$ to first order!

#### Theorem: Monotonic Improvement Bound (Schulman et al., 2015)
Let $D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) \triangleq \max_{s \in \mathcal{S}} D_{\text{KL}}(\pi(\cdot \mid s) \parallel \tilde{\pi}(\cdot \mid s))$ and $\epsilon \triangleq \max_{s, a} |A^\pi(s, a)|$.
Then the true return is lower-bounded by:
$$J(\tilde{\pi}) \ge L_\pi(\tilde{\pi}) - C \cdot D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) \quad \text{where} \quad C \triangleq \frac{4 \epsilon \gamma}{(1 - \gamma)^2}$$

This guarantees that maximizing the right-hand side is guaranteed to generate **monotonic policy improvement**: $J(\pi_{k+1}) \ge J(\pi_k)$!

---

### 2.3 The TRPO Constrained Optimization Problem

In practice, the penalty coefficient $C$ is too large, resulting in overly conservative steps. TRPO converts the penalty into a **hard trust region constraint** on the average KL divergence:

$$\max_{\boldsymbol{\theta}} \mathbb{E}_{s \sim d^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a \mid s)} \hat{A}_t \right] \quad \text{subject to} \quad \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \le \delta$$
where $\bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \triangleq \mathbb{E}_{s \sim d^{\pi_{\text{old}}}} [D_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}}(\cdot \mid s) \parallel \pi_{\boldsymbol{\theta}}(\cdot \mid s))]$.

#### Quadratic Approximation:
Taking a first-order Taylor expansion of the objective and a second-order expansion of the constraint:
$$\max_{\Delta \boldsymbol{\theta}} \mathbf{g}^\top \Delta \boldsymbol{\theta} \quad \text{subject to} \quad \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \le \delta$$
where:
$$\mathbf{g} \triangleq \left. \nabla_{\boldsymbol{\theta}} L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) \right|_{\boldsymbol{\theta}_{\text{old}}}, \quad \mathbf{F} \triangleq \left. \nabla_{\boldsymbol{\theta}}^2 \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \right|_{\boldsymbol{\theta}_{\text{old}}}$$

As derived in Chapter 11.17, the analytical solution is:
$$\Delta \boldsymbol{\theta} = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{F} \mathbf{x}}} \mathbf{x} \quad \text{where} \quad \mathbf{F} \mathbf{x} = \mathbf{g}$$

---

### 2.4 The Hessian-Vector Product Trick (Pearlmutter, 1994)

To solve $\mathbf{F} \mathbf{x} = \mathbf{g}$, we need to multiply the matrix $\mathbf{F}$ by arbitrary search vectors $\mathbf{v} \in \mathbb{R}^d$ without computing or storing $\mathbf{F}$.
Notice that:
$$\mathbf{F} \mathbf{v} = \nabla_{\boldsymbol{\theta}}^2 \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \mathbf{v} = \nabla_{\boldsymbol{\theta}} \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta})^\top \mathbf{v} \right)$$

#### In PyTorch / Autograd:
1. Compute the scalar quantity $y = \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}} \right)^\top \mathbf{v}$ (an inner product of vectors).
2. Take the gradient of $y$ with respect to $\boldsymbol{\theta}$:
   $$\mathbf{F} \mathbf{v} = \operatorname{autograd.grad}(y, \boldsymbol{\theta})$$
This requires only **one extra backward pass**, using $\mathcal{O}(d)$ memory!

---

### 2.5 Conjugate Gradient (CG) Algorithm

To solve $\mathbf{F} \mathbf{x} = \mathbf{g}$ iteratively:
```
Algorithm: Conjugate Gradient for F x = g
Input: Matrix-vector function F(v), vector g, tolerance tol, max_iter K=10

x_0 = 0,  r_0 = g,  p_0 = r_0
For i = 0, 1, ..., K - 1:
    v = F(p_i)                          # One Hessian-vector product!
    alpha_i = (r_i^T r_i) / (p_i^T v)
    x_{i+1} = x_i + alpha_i * p_i
    r_{i+1} = r_i - alpha_i * v
    if ||r_{i+1}|| < tol: break
    beta_i = (r_{i+1}^T r_{i+1}) / (r_i^T r_i)
    p_{i+1} = r_{i+1} + beta_i * p_i

Return x_{final}
```

---

### 2.6 Backtracking Line Search

Because the linear-quadratic approximation is only locally accurate, taking the full step $\Delta \boldsymbol{\theta}$ might violate the non-linear KL constraint or decrease the surrogate objective.
TRPO applies a **backtracking line search**:
$$\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} + \alpha^j \Delta \boldsymbol{\theta} \quad \text{for } j \in \{0, 1, 2, \dots, M\}$$
where $\alpha \in (0, 1)$ (typically $\alpha = 0.5$, max steps $M = 10$).

The candidate update is accepted at the smallest integer $j$ that satisfies both:
1. **Surrogate Improvement:** $L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{new}}) \ge 0$
2. **Trust Region Enforcement:** $\bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}_{\text{new}}) \le \delta$

If no step satisfies both criteria after $M$ reductions, the update is aborted ($\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}}$), ensuring total stability!

---

### 2.7 First-Principles Mathematical Derivations

#### Derivation 11.18.1: Kakade & Langford Monotonic Improvement Guarantee

```
====================================================================================================
DERIVATION 11.18.1: Kakade & Langford Monotonic Improvement Guarantee
====================================================================================================
Problem Statement:
Let an infinite-horizon Markov Decision Process (MDP) be defined by (𝒮, 𝒜, 𝒫, ℛ, γ, μ), with state space 𝒮,
action space 𝒜, transition probability kernel 𝒫(s' | s, a), reward function r(s, a) bounded by |r(s, a)| ≤ R_max,
discount factor γ ∈ [0, 1), and initial state distribution μ. For any two stationary policies π and π̃,
the expected discounted return is J(π) ≜ 𝔼_{τ ~ π}[∑_{t=0}^∞ γ^t r(s_t, a_t)].
Prove from first principles:
1. The Performance Difference Lemma:
       J(π̃) - J(π) = 𝔼_{τ ~ π̃} [ ∑_{t=0}^∞ γ^t A^π(s_t, a_t) ] = ∑_{s ∈ 𝒮} d^{π̃}(s) ∑_{a ∈ 𝒜} π̃(a | s) A^π(s, a)
   where d^{π̃}(s) ≜ ∑_{t=0}^∞ γ^t ℙ(s_t = s | s_0 ~ μ; π̃) is the unnormalized discounted state visitation distribution.
2. The Monotonic Improvement Bound:
   With surrogate objective L_π(π̃) ≜ J(π) + ∑_{s ∈ 𝒮} d^π(s) ∑_{a ∈ 𝒜} π̃(a | s) A^π(s, a),
       J(π̃) ≥ L_π(π̃) - C · D_KL^max(π, π̃)
   where C ≜ (4 ε γ) / (1 - γ)^2, ε ≜ max_{s, a} |A^π(s, a)|, and D_KL^max(π, π̃) ≜ max_{s ∈ 𝒮} D_KL(π(· | s) ∥ π̃(· | s)).
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In reinforcement learning policy optimization, we seek to update policy parameters from $\boldsymbol{\theta}_{\text{old}}$ (representing policy $\pi$) to $\boldsymbol{\theta}$ (representing policy $\tilde{\pi}$) to maximize the expected discounted return $J(\tilde{\pi})$. However, evaluating $J(\tilde{\pi})$ requires generating trajectories under $\tilde{\pi}$, which is unknown prior to executing the update.
Our mathematical goal is two-fold:
1. Express the exact performance difference $J(\tilde{\pi}) - J(\pi)$ as an expectation of the old advantage function $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ over trajectories generated by the new policy $\tilde{\pi}$.
2. Derive a rigorous lower bound on $J(\tilde{\pi})$ in terms of the surrogate objective $L_\pi(\tilde{\pi})$ (which evaluates state visitation under the old policy $\pi$) and the maximum KL divergence $D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$, establishing that maximizing this bound guarantees monotonic policy improvement: $J(\pi_{k+1}) \ge J(\pi_k)$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Markovian Dynamics & Stationarity:** The transition probability kernel $\mathcal{P}(s' \mid s, a)$ and reward function $r(s, a)$ are stationary, Markovian, and time-invariant.
2. **Bounded Rewards:** The scalar reward function is uniformly bounded: $\exists R_{\max} \in (0, \infty)$ such that $|r(s, a)| \le R_{\max}$ for all $(s, a) \in \mathcal{S} \times \mathcal{A}$. Consequently, the value functions are bounded: $|V^\pi(s)| \le \frac{R_{\max}}{1 - \gamma}$ and $|Q^\pi(s, a)| \le \frac{R_{\max}}{1 - \gamma}$, and the advantage magnitude satisfies $\epsilon \triangleq \max_{s, a} |A^\pi(s, a)| \le \frac{2 R_{\max}}{1 - \gamma} < \infty$.
3. **Strict Geometric Discounting:** The discount factor satisfies $\gamma \in [0, 1)$, which guarantees absolute convergence of all infinite-horizon sums and bounded state visitation measures $\sum_{s} d^\pi(s) = \frac{1}{1 - \gamma} < \infty$.
4. **Absolute Continuity & Finite Divergence:** The new policy $\tilde{\pi}$ dominates the old policy $\pi$ in support, i.e., $\tilde{\pi}(a \mid s) > 0$ whenever $\pi(a \mid s) > 0$, ensuring that the Kullback-Leibler divergence $D_{\text{KL}}(\pi(\cdot \mid s) \parallel \tilde{\pi}(\cdot \mid s)) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \log \frac{\pi(a \mid s)}{\tilde{\pi}(a \mid s)}$ is well-defined and finite for every state $s \in \mathcal{S}$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
The central dilemma in policy optimization is that changing the policy alters two coupled quantities:
(i) the *action selection probabilities* $\tilde{\pi}(a \mid s)$, and
(ii) the *state visitation distribution* $d^{\tilde{\pi}}(s)$.
While the policy designer directly controls (i), the change in state visitation (ii) is an indirect, compounding dynamical consequence of all past actions propagating through the environment.
The surrogate objective $L_\pi(\tilde{\pi})$ ignores the state distribution shift by freezing the state distribution at $d^\pi$. Geometrically, $L_\pi(\tilde{\pi})$ is the first-order Taylor tangent plane to the true objective surface $J(\tilde{\pi})$ at $\tilde{\pi} = \pi$.
Because the state visitation discrepancy $\|d^{\tilde{\pi}} - d^\pi\|_1$ compounds over an infinite horizon, trajectories diverge with a probability bounded by the total variation distance between policy action distributions. By bounding the probability of trajectory divergence via coupling and applying Pinsker's inequality, we show that the discrepancy between $J(\tilde{\pi})$ and $L_\pi(\tilde{\pi})$ is quadratic in policy divergence $\mathcal{O}(D_{\text{TV}}^2) \le \mathcal{O}(D_{\text{KL}})$.
Subtracting the penalty $C D_{\text{KL}}^{\max}$ bends the tangent plane into a concave supporting paraboloid that sits strictly *below* the true performance curve everywhere, touching it tangentially at $\tilde{\pi} = \pi$. Any step that increases this lower bound is guaranteed to strictly increase true performance $J(\tilde{\pi})$!

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Express the Advantage Function in Bellman Expectation Form.*
By definition of the advantage function $A^\pi(s, a)$ under policy $\pi$:
$$A^\pi(s, a) \triangleq Q^\pi(s, a) - V^\pi(s)$$
From the Bellman expectation equation for $Q^\pi(s, a)$:
$$Q^\pi(s, a) = \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ r(s, a) + \gamma V^\pi(s') \right]$$
Substituting this into the definition of $A^\pi$:
$$A^\pi(s, a) = \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ r(s, a) + \gamma V^\pi(s') - V^\pi(s) \right] \quad \text{(Equation 1)}$$

*Step 2: Accumulate Discounted Advantages along a Trajectory $\tau \sim \tilde{\pi}$.*
Let $\tau = (s_0, a_0, s_1, a_1, \dots)$ denote an arbitrary trajectory generated by the candidate policy $\tilde{\pi}$, with initial state $s_0 \sim \mu$.
Consider the infinite discounted sum of advantage values evaluated under the old policy $\pi$ along this new trajectory $\tau$:
$$\sum_{t=0}^\infty \gamma^t A^\pi(s_t, a_t)$$
Taking the conditional expectation of each term given $s_t, a_t$ and using Equation 1:
$$\mathbb{E}_{s_{t+1}} \left[ \gamma^t A^\pi(s_t, a_t) \mid s_t, a_t \right] = \mathbb{E}_{s_{t+1}} \left[ \gamma^t r(s_t, a_t) + \gamma^{t+1} V^\pi(s_{t+1}) - \gamma^t V^\pi(s_t) \right]$$
Summing over all time steps $t = 0, 1, 2, \dots, T-1$:
$$\sum_{t=0}^{T-1} \gamma^t \left( r(s_t, a_t) + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) \right) = \sum_{t=0}^{T-1} \gamma^t r(s_t, a_t) + \sum_{t=0}^{T-1} \left( \gamma^{t+1} V^\pi(s_{t+1}) - \gamma^t V^\pi(s_t) \right)$$

*Step 3: Telescoping the Value Function Differences.*
Examine the summation of value differences:
$$\sum_{t=0}^{T-1} \left( \gamma^{t+1} V^\pi(s_{t+1}) - \gamma^t V^\pi(s_t) \right)$$
Writing out the first few terms explicitly:
$$= \left( \gamma V^\pi(s_1) - V^\pi(s_0) \right) + \left( \gamma^2 V^\pi(s_2) - \gamma V^\pi(s_1) \right) + \dots + \left( \gamma^T V^\pi(s_T) - \gamma^{T-1} V^\pi(s_{T-1}) \right)$$
All intermediate terms cancel identically, leaving solely:
$$= - V^\pi(s_0) + \gamma^T V^\pi(s_T)$$
Taking the infinite-horizon limit $T \to \infty$:
Since rewards are bounded ($|r| \le R_{\max}$), $|V^\pi(s_T)| \le \frac{R_{\max}}{1 - \gamma}$.
Because $\gamma \in [0, 1)$, we have:
$$\lim_{T \to \infty} \left| \gamma^T V^\pi(s_T) \right| \le \lim_{T \to \infty} \gamma^T \frac{R_{\max}}{1 - \gamma} = 0$$
Hence, the infinite series telescopes deterministically to:
$$\sum_{t=0}^\infty \left( \gamma^{t+1} V^\pi(s_{t+1}) - \gamma^t V^\pi(s_t) \right) = - V^\pi(s_0)$$

*Step 4: Take Expectations over Trajectories Generated by $\tilde{\pi}$.*
Taking the full expectation over all trajectories $\tau = (s_0, a_0, s_1, a_1, \dots) \sim \tilde{\pi}$:
$$\mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t A^\pi(S_t, A_t) \right] = \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t r(S_t, A_t) - V^\pi(S_0) \right]$$
By linearity of expectation:
$$= \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t r(S_t, A_t) \right] - \mathbb{E}_{S_0 \sim \mu} \left[ V^\pi(S_0) \right]$$
Recognizing the definitions of expected return:
$$\mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t r(S_t, A_t) \right] = J(\tilde{\pi}), \quad \mathbb{E}_{S_0 \sim \mu} \left[ V^\pi(S_0) \right] = J(\pi)$$
Therefore, we have established the exact Performance Difference Lemma:
$$J(\tilde{\pi}) - J(\pi) = \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t A^\pi(S_t, A_t) \right] \quad \text{(Equation 2)}$$

*Step 5: Convert Trajectory Expectation to Discounted State Visitation Measure.*
By Tonelli-Fubini theorem for non-negative/absolutely convergent series:
$$\mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t A^\pi(S_t, A_t) \right] = \sum_{t=0}^\infty \gamma^t \mathbb{E}_{S_t \sim \tilde{\pi}, A_t \sim \tilde{\pi}} \left[ A^\pi(S_t, A_t) \right]$$
$$= \sum_{t=0}^\infty \gamma^t \sum_{s \in \mathcal{S}} \mathbb{P}(S_t = s \mid S_0 \sim \mu; \tilde{\pi}) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a)$$
Interchanging the summation over time and states:
$$= \sum_{s \in \mathcal{S}} \left( \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = s \mid S_0 \sim \mu; \tilde{\pi}) \right) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a)$$
Define the unnormalized discounted state visitation distribution under $\tilde{\pi}$:
$$d^{\tilde{\pi}}(s) \triangleq \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = s \mid S_0 \sim \mu; \tilde{\pi})$$
Thus:
$$J(\tilde{\pi}) - J(\pi) = \sum_{s \in \mathcal{S}} d^{\tilde{\pi}}(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a) \quad \text{(Equation 3)}$$

*Step 6: Decompose the Difference between True Return and Surrogate Objective.*
The surrogate objective is defined by evaluating the expectation under $d^\pi(s)$ rather than $d^{\tilde{\pi}}(s)$:
$$L_\pi(\tilde{\pi}) \triangleq J(\pi) + \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a) \quad \text{(Equation 4)}$$
Subtracting Equation 4 from Equation 3:
$$J(\tilde{\pi}) - L_\pi(\tilde{\pi}) = \sum_{s \in \mathcal{S}} \left( d^{\tilde{\pi}}(s) - d^\pi(s) \right) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a) \quad \text{(Equation 5)}$$
Define the expected advantage under policy $\tilde{\pi}$ at state $s$:
$$\bar{A}^\pi(s) \triangleq \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a)$$
Notice that under the baseline policy $\pi$, the expected advantage is identically zero for all states:
$$\sum_{a \in \mathcal{A}} \pi(a \mid s) A^\pi(s, a) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( Q^\pi(s, a) - V^\pi(s) \right) = V^\pi(s) - V^\pi(s) = 0$$
Therefore, we can subtract zero to write:
$$\bar{A}^\pi(s) = \sum_{a \in \mathcal{A}} \left( \tilde{\pi}(a \mid s) - \pi(a \mid s) \right) A^\pi(s, a)$$
Applying the triangle inequality:
$$|\bar{A}^\pi(s)| \le \sum_{a \in \mathcal{A}} |\tilde{\pi}(a \mid s) - \pi(a \mid s)| \cdot |A^\pi(s, a)| \le \left( \max_{a} |A^\pi(s, a)| \right) \sum_{a \in \mathcal{A}} |\tilde{\pi}(a \mid s) - \pi(a \mid s)|$$
Recall the definition of Total Variation (TV) distance between discrete distributions:
$$D_{\text{TV}}(P, Q) \triangleq \frac{1}{2} \sum_{x} |P(x) - Q(x)| \implies \sum_{a} |\tilde{\pi}(a \mid s) - \pi(a \mid s)| = 2 D_{\text{TV}}(\pi(\cdot \mid s), \tilde{\pi}(\cdot \mid s))$$
Letting $\epsilon \triangleq \max_{s, a} |A^\pi(s, a)|$ and $\alpha \triangleq \max_{s \in \mathcal{S}} D_{\text{TV}}(\pi(\cdot \mid s), \tilde{\pi}(\cdot \mid s))$:
$$|\bar{A}^\pi(s)| \le 2 \epsilon D_{\text{TV}}(\pi(\cdot \mid s), \tilde{\pi}(\cdot \mid s)) \le 2 \epsilon \alpha \quad \text{(Equation 6)}$$

*Step 7: Bound the State Distribution Shift via Coupling.*
Now express the state visitation difference in time-step marginal distributions:
$$d^{\tilde{\pi}}(s) - d^\pi(s) = \sum_{t=0}^\infty \gamma^t \left( \mathbb{P}(S_t = s \mid \tilde{\pi}) - \mathbb{P}(S_t = s \mid \pi) \right)$$
Substituting this into Equation 5:
$$J(\tilde{\pi}) - L_\pi(\tilde{\pi}) = \sum_{t=0}^\infty \gamma^t \left( \mathbb{E}_{S_t \sim \tilde{\pi}} [\bar{A}^\pi(S_t)] - \mathbb{E}_{S_t \sim \pi} [\bar{A}^\pi(S_t)] \right)$$
At time $t = 0$, both processes share the exact same initial state distribution $S_0 \sim \mu$, so:
$$\mathbb{E}_{S_0 \sim \tilde{\pi}} [\bar{A}^\pi(S_0)] - \mathbb{E}_{S_0 \sim \pi} [\bar{A}^\pi(S_0)] = \sum_s \mu(s) \bar{A}^\pi(s) - \sum_s \mu(s) \bar{A}^\pi(s) = 0$$
For time steps $t \ge 1$, we couple the action generation of $\pi$ and $\tilde{\pi}$ at each time step.
At each step $k$, given identical states, the maximal coupling allows $\pi$ and $\tilde{\pi}$ to select identical actions with probability $1 - D_{\text{TV}}(\pi(\cdot \mid S_k), \tilde{\pi}(\cdot \mid S_k)) \ge 1 - \alpha$.
The trajectories remain identical until the first time an action differs.
The probability that policies $\pi$ and $\tilde{\pi}$ have selected differing actions at or before step $t-1$ satisfies:
$$\mathbb{P}(\text{divergence before } t) = 1 - \mathbb{P}(\text{all actions agree for } k = 0, \dots, t-1) \le 1 - (1 - \alpha)^t$$
By Bernoulli's inequality or binomial expansion for $\alpha \in [0, 1]$:
$$1 - (1 - \alpha)^t \le t \alpha$$
When the trajectories have not diverged, their state distributions are identical. Therefore, the total variation distance between the state distributions at step $t$ is bounded by the probability of divergence:
$$D_{\text{TV}}\left( \mathbb{P}(S_t = \cdot \mid \tilde{\pi}), \mathbb{P}(S_t = \cdot \mid \pi) \right) \le 1 - (1 - \alpha)^t \le t \alpha$$
For any bounded function $f$ with $|f(s)| \le M$, $|\mathbb{E}_P[f] - \mathbb{E}_Q[f]| \le 2 M D_{\text{TV}}(P, Q)$. Here, $|\bar{A}^\pi(s)| \le 2 \epsilon \alpha$ (from Equation 6). Thus:
$$\left| \mathbb{E}_{S_t \sim \tilde{\pi}} [\bar{A}^\pi(S_t)] - \mathbb{E}_{S_t \sim \pi} [\bar{A}^\pi(S_t)] \right| \le 2 \cdot (2 \epsilon \alpha) \cdot (t \alpha) = 4 \epsilon \alpha^2 t$$

*Step 8: Sum the Infinite Geometric Series.*
Substitute this bound back into the summation over $t \ge 1$:
$$|J(\tilde{\pi}) - L_\pi(\tilde{\pi})| \le \sum_{t=1}^\infty \gamma^t \left( 4 \epsilon \alpha^2 t \right) = 4 \epsilon \alpha^2 \sum_{t=1}^\infty t \gamma^t$$
Evaluate the summation $\sum_{t=1}^\infty t \gamma^t$ analytically:
$$\sum_{t=1}^\infty t \gamma^t = \gamma \sum_{t=1}^\infty t \gamma^{t-1} = \gamma \frac{d}{d\gamma} \left( \sum_{t=0}^\infty \gamma^t \right) = \gamma \frac{d}{d\gamma} \left( \frac{1}{1 - \gamma} \right) = \gamma \cdot \frac{1}{(1 - \gamma)^2} = \frac{\gamma}{(1 - \gamma)^2}$$
Substituting this exact sum:
$$|J(\tilde{\pi}) - L_\pi(\tilde{\pi})| \le \frac{4 \epsilon \gamma}{(1 - \gamma)^2} \alpha^2 \quad \text{(Equation 7)}$$

*Step 9: Connect Total Variation Distance to KL Divergence via Pinsker's Inequality.*
Pinsker's inequality establishes a fundamental upper bound on Total Variation distance by Kullback-Leibler divergence:
$$D_{\text{TV}}(P, Q) \le \sqrt{\frac{1}{2} D_{\text{KL}}(P \parallel Q)} \implies \left( D_{\text{TV}}(P, Q) \right)^2 \le \frac{1}{2} D_{\text{KL}}(P \parallel Q) \le D_{\text{KL}}(P \parallel Q)$$
Taking the supremum over all states $s \in \mathcal{S}$:
$$\alpha^2 = \left( \max_{s \in \mathcal{S}} D_{\text{TV}}(\pi(\cdot \mid s), \tilde{\pi}(\cdot \mid s)) \right)^2 \le \max_{s \in \mathcal{S}} D_{\text{KL}}(\pi(\cdot \mid s) \parallel \tilde{\pi}(\cdot \mid s)) \triangleq D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$$
Define the constant:
$$C \triangleq \frac{4 \epsilon \gamma}{(1 - \gamma)^2}$$
Then Equation 7 yields:
$$|J(\tilde{\pi}) - L_\pi(\tilde{\pi})| \le C \cdot D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$$
Removing the absolute value gives the lower bound:
$$J(\tilde{\pi}) \ge L_\pi(\tilde{\pi}) - C \cdot D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) \quad \blacksquare$$

---

#### Derivation 11.18.2: Quadratic Approximation of Trust Region Constraint and Closed-Form Step Direction

```
====================================================================================================
DERIVATION 11.18.2: Quadratic Approximation of Trust Region Constraint and Closed-Form Step Direction
====================================================================================================
Problem Statement:
Consider the local TRPO optimization subproblem in parameter space ℝ^d:
    max_{Δθ}  g^T Δθ
    subject to (1/2) Δθ^T F Δθ ≤ δ
where g ≜ ∇_θ L_{θ_old}(θ)|_{θ_old} ∈ ℝ^d is the surrogate objective gradient,
F ≜ ∇_θ^2 D̄_KL(θ_old ∥ θ)|_{θ_old} ∈ ℝ^{d × d} is the symmetric positive definite Fisher Information Matrix,
and δ > 0 is the maximum allowable average KL divergence.
Using Karush-Kuhn-Tucker (KKT) conditions, derive from first principles:
1. The proof that the trust region constraint is strictly active ((1/2) Δθ^T F Δθ = δ) whenever g ≠ 0.
2. The closed-form analytical optimal dual multiplier λ* and primal step Δθ*:
       Δθ* = sqrt( (2 δ) / (g^T F^{-1} g) ) F^{-1} g
3. The proof of second-order sufficiency establishing that Δθ* is the unique global maximum.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In Trust Region Policy Optimization, taking an unconstrained step along the policy gradient $\mathbf{g}$ can cause catastrophic policy collapse because the Euclidean gradient does not account for the non-linear curvature of the probability distribution manifold.
To guarantee stable policy improvement within a local neighborhood, TRPO forms a local quadratic surrogate problem:
$$\max_{\Delta \boldsymbol{\theta} \in \mathbb{R}^d} \mathbf{g}^\top \Delta \boldsymbol{\theta} \quad \text{subject to} \quad \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \le \delta$$
Our mathematical goal is to solve this constrained quadratic program analytically using Karush-Kuhn-Tucker (KKT) duality theory, derive the exact closed-form optimal step direction and step magnitude $\Delta \boldsymbol{\theta}^*$, and prove that the solution is the unique global maximizer.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Non-Zero Gradient:** $\mathbf{g} \ne \mathbf{0}$. If $\mathbf{g} = \mathbf{0}$, the current parameter $\boldsymbol{\theta}_{\text{old}}$ is already a stationary point of the surrogate objective, yielding the trivial optimal step $\Delta \boldsymbol{\theta}^* = \mathbf{0}$.
2. **Symmetric Positive Definiteness:** The Fisher Information Matrix $\mathbf{F} \in \mathbb{R}^{d \times d}$ is symmetric ($\mathbf{F} = \mathbf{F}^\top$) and strictly positive definite ($\mathbf{F} \succ 0$), meaning all eigenvalues satisfy $\lambda_{\min}(\mathbf{F}) > 0$. This ensures that $\mathbf{F}^{-1}$ exists and is also strictly positive definite, and that the quadratic form $\Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} > 0$ for all $\Delta \boldsymbol{\theta} \ne \mathbf{0}$.
3. **Strictly Positive Trust Region Radius:** $\delta > 0$.
4. **Slater's Constraint Qualification:** The optimization problem is convex (minimizing a linear function over a convex quadratic set). The point $\Delta \boldsymbol{\theta} = \mathbf{0}$ satisfies $\frac{1}{2} \mathbf{0}^\top \mathbf{F} \mathbf{0} = 0 < \delta$. Hence, an interior point exists, satisfying Slater's condition, which guarantees strong duality and that the KKT conditions are both necessary and sufficient for global optimality.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
Geometrically, the objective function $\mathbf{g}^\top \Delta \boldsymbol{\theta}$ describes a family of parallel hyperplanes in $\mathbb{R}^d$ whose normal vector points along $\mathbf{g}$. We wish to move as far as possible in the direction of increasing $\mathbf{g}^\top \Delta \boldsymbol{\theta}$.
The constraint $\frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \le \delta$ defines a solid $d$-dimensional hyper-ellipsoid centered at the origin $\Delta \boldsymbol{\theta} = \mathbf{0}$. The principal axes of this ellipsoid align with the eigenvectors of $\mathbf{F}$, and the semi-axis lengths are inversely proportional to the square roots of the eigenvalues: $a_i = \sqrt{2 \delta / \lambda_i}$.
- Along directions of high curvature (large $\lambda_i$, where the policy distribution changes violently with tiny parameter perturbations), the ellipsoid is compressed, strictly limiting the step size.
- Along directions of low curvature (small $\lambda_i$, where the policy distribution is insensitive to parameter variations), the ellipsoid is elongated, permitting larger exploratory steps.
The optimal step $\Delta \boldsymbol{\theta}^*$ is the unique boundary point of this ellipsoid where the outward surface normal of the ellipsoid, given by $\nabla_{\Delta \boldsymbol{\theta}} \left( \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \right) = \mathbf{F} \Delta \boldsymbol{\theta}$, is parallel to the objective gradient $\mathbf{g}$.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Standard Form Formulation.*
We formulate the problem as a standard convex minimization problem:
$$\min_{\Delta \boldsymbol{\theta} \in \mathbb{R}^d} f_0(\Delta \boldsymbol{\theta}) \quad \text{subject to} \quad f_1(\Delta \boldsymbol{\theta}) \le 0$$
where:
$$f_0(\Delta \boldsymbol{\theta}) \triangleq -\mathbf{g}^\top \Delta \boldsymbol{\theta}, \quad f_1(\Delta \boldsymbol{\theta}) \triangleq \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} - \delta$$

*Step 2: Construct the Lagrangian Function.*
Introduce the Lagrange multiplier (dual variable) $\lambda \ge 0$ associated with the inequality constraint:
$$\mathcal{L}(\Delta \boldsymbol{\theta}, \lambda) \triangleq f_0(\Delta \boldsymbol{\theta}) + \lambda f_1(\Delta \boldsymbol{\theta}) = -\mathbf{g}^\top \Delta \boldsymbol{\theta} + \lambda \left( \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} - \delta \right)$$

*Step 3: State the Karush-Kuhn-Tucker (KKT) Conditions.*
Since Slater's condition holds, $\Delta \boldsymbol{\theta}^*$ is a global optimum if and only if there exists $\lambda^* \in \mathbb{R}$ satisfying:
1. **Stationarity:**
   $$\nabla_{\Delta \boldsymbol{\theta}} \mathcal{L}(\Delta \boldsymbol{\theta}^*, \lambda^*) = -\mathbf{g} + \lambda^* \mathbf{F} \Delta \boldsymbol{\theta}^* = \mathbf{0} \quad \text{(KKT 1)}$$
2. **Primal Feasibility:**
   $$\frac{1}{2} (\Delta \boldsymbol{\theta}^*)^\top \mathbf{F} \Delta \boldsymbol{\theta}^* - \delta \le 0 \quad \text{(KKT 2)}$$
3. **Dual Feasibility:**
   $$\lambda^* \ge 0 \quad \text{(KKT 3)}$$
4. **Complementary Slackness:**
   $$\lambda^* \left( \frac{1}{2} (\Delta \boldsymbol{\theta}^*)^\top \mathbf{F} \Delta \boldsymbol{\theta}^* - \delta \right) = 0 \quad \text{(KKT 4)}$$

*Step 4: Prove the Constraint is Strictly Active ($\lambda^* > 0$).*
We prove by contradiction that $\lambda^* \ne 0$.
Suppose $\lambda^* = 0$. Then by Stationarity (KKT 1):
$$-\mathbf{g} + (0) \mathbf{F} \Delta \boldsymbol{\theta}^* = \mathbf{0} \implies -\mathbf{g} = \mathbf{0} \implies \mathbf{g} = \mathbf{0}$$
However, this contradicts Assumption 1 ($\mathbf{g} \ne \mathbf{0}$).
Therefore, $\lambda^* \ne 0$.
Combining with Dual Feasibility $\lambda^* \ge 0$ (KKT 3), we establish:
$$\lambda^* > 0$$
From Complementary Slackness (KKT 4), since $\lambda^* \ne 0$, the constraint term must vanish identically:
$$\frac{1}{2} (\Delta \boldsymbol{\theta}^*)^\top \mathbf{F} \Delta \boldsymbol{\theta}^* - \delta = 0 \implies \frac{1}{2} (\Delta \boldsymbol{\theta}^*)^\top \mathbf{F} \Delta \boldsymbol{\theta}^* = \delta \quad \text{(Equation 1)}$$
The optimal solution lies strictly on the boundary of the trust region ellipsoid.

*Step 5: Solve for the Primal Step in Terms of Dual Variable $\lambda^*$.*
Rearranging the Stationarity condition (KKT 1):
$$\lambda^* \mathbf{F} \Delta \boldsymbol{\theta}^* = \mathbf{g}$$
Since $\mathbf{F}$ is positive definite, its inverse $\mathbf{F}^{-1}$ exists. Dividing by the non-zero scalar $\lambda^*$ and pre-multiplying by $\mathbf{F}^{-1}$:
$$\Delta \boldsymbol{\theta}^* = \frac{1}{\lambda^*} \mathbf{F}^{-1} \mathbf{g} \quad \text{(Equation 2)}$$

*Step 6: Substitute Primal Step into Boundary Constraint to Solve for $\lambda^*$.*
Substitute Equation 2 into the active constraint Equation 1:
$$\frac{1}{2} \left( \frac{1}{\lambda^*} \mathbf{F}^{-1} \mathbf{g} \right)^\top \mathbf{F} \left( \frac{1}{\lambda^*} \mathbf{F}^{-1} \mathbf{g} \right) = \delta$$
Using the transpose property $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top$ and symmetry of $\mathbf{F}^{-1} = (\mathbf{F}^{-1})^\top$:
$$\left( \frac{1}{\lambda^*} \mathbf{F}^{-1} \mathbf{g} \right)^\top = \frac{1}{\lambda^*} \mathbf{g}^\top \mathbf{F}^{-1}$$
Substituting this in:
$$\frac{1}{2 (\lambda^*)^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{F} \mathbf{F}^{-1} \mathbf{g} = \delta$$
Because $\mathbf{F}^{-1} \mathbf{F} = \mathbf{I}$:
$$\frac{1}{2 (\lambda^*)^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} = \delta$$
Multiply both sides by $2 (\lambda^*)^2$:
$$\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} = 2 \delta (\lambda^*)^2$$
Since $\mathbf{F} \succ 0$ and $\mathbf{g} \ne \mathbf{0}$, the quadratic form $\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} > 0$. Dividing by $2 \delta > 0$:
$$(\lambda^*)^2 = \frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \delta}$$
Taking the positive square root (since $\lambda^* > 0$):
$$\lambda^* = \sqrt{\frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \delta}} \quad \text{(Equation 3)}$$

*Step 7: Derive the Analytical Optimal Step $\Delta \boldsymbol{\theta}^*$.*
Substitute Equation 3 back into Equation 2:
$$\Delta \boldsymbol{\theta}^* = \frac{1}{\sqrt{\frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \delta}}} \mathbf{F}^{-1} \mathbf{g} = \sqrt{\frac{2 \delta}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g} \quad \text{(Equation 4)}$$

*Step 8: Express in Terms of the Linear System Solution $\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$.*
Let $\mathbf{x} \in \mathbb{R}^d$ be the solution to the linear system $\mathbf{F} \mathbf{x} = \mathbf{g}$, so that $\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$.
Then:
$$\mathbf{x}^\top \mathbf{F} \mathbf{x} = (\mathbf{F}^{-1} \mathbf{g})^\top \mathbf{F} (\mathbf{F}^{-1} \mathbf{g}) = \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{F} \mathbf{F}^{-1} \mathbf{g} = \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} = \mathbf{x}^\top \mathbf{g}$$
Defining the scaling factor $\beta \triangleq \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{F} \mathbf{x}}} = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{g}}}$, the optimal step is concisely:
$$\Delta \boldsymbol{\theta}^* = \beta \mathbf{x} \quad \text{where} \quad \mathbf{F} \mathbf{x} = \mathbf{g}, \quad \beta = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{g}}}$$

*Step 9: Second-Order Optimality Verification.*
To verify that $\Delta \boldsymbol{\theta}^*$ is a strict global minimum of $f_0$ (hence strict global maximum of $\mathbf{g}^\top \Delta \boldsymbol{\theta}$), examine the Hessian of the Lagrangian with respect to $\Delta \boldsymbol{\theta}$:
$$\nabla_{\Delta \boldsymbol{\theta}}^2 \mathcal{L}(\Delta \boldsymbol{\theta}, \lambda^*) = \lambda^* \mathbf{F}$$
Since $\lambda^* > 0$ and $\mathbf{F} \succ 0$:
$$\mathbf{z}^\top \left( \nabla_{\Delta \boldsymbol{\theta}}^2 \mathcal{L} \right) \mathbf{z} = \lambda^* \mathbf{z}^\top \mathbf{F} \mathbf{z} > 0 \quad \forall \mathbf{z} \ne \mathbf{0}$$
The Hessian of the Lagrangian is strictly positive definite over the entire space $\mathbb{R}^d$.
By the second-order sufficient conditions for constrained optimization, $\Delta \boldsymbol{\theta}^*$ is the unique, strict global maximizer of the surrogate objective subject to the quadratic trust region constraint. $\blacksquare$

---

#### Derivation 11.18.3: Conjugate Gradient Algorithm and Fisher-Vector Product (FVP) without Explicit Matrix Construction

```
====================================================================================================
DERIVATION 11.18.3: Conjugate Gradient Algorithm and Fisher-Vector Product (FVP)
====================================================================================================
Problem Statement:
In TRPO, computing the natural gradient search direction requires solving the d-dimensional linear system
    F x = g
where d ≈ 10^6 is the number of neural network parameters, g ∈ ℝ^d is the policy surrogate gradient,
and F ≜ 𝔼_{s ~ d^π}[ ∇_θ^2 D_KL(π_{θ_old}(· | s) ∥ π_θ(· | s)) |_{θ = θ_old} ] ∈ ℝ^{d × d} is the Fisher matrix.
Prove from first principles:
1. The Pearlmutter Fisher-Vector Product (FVP) Identity:
   For any arbitrary vector v ∈ ℝ^d, the matrix-vector product F v can be computed in 𝒪(d) time and memory
   without computing, storing, or approximating F, via the exact double-differentiation identity:
       F v = ∇_θ [ ( ∇_θ D̄_KL(θ_old ∥ θ) )^T v ] |_{θ = θ_old}
2. The Conjugate Gradient Algorithm:
   By minimizing the quadratic energy function ϕ(x) = (1/2) x^T F x - g^T x along F-conjugate search directions
   {p_0, p_1, ..., p_{k-1}}, derive the step size α_k = (r_k^T r_k) / (p_k^T F p_k), the residual update
   r_{k+1} = r_k - α_k F p_k, and the Gram-Schmidt conjugacy update β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k).
3. Prove that CG converges to the exact analytical solution in at most d iterations using only matrix-vector products.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
For a neural network with $d = 10^6$ parameters, storing the Fisher Information Matrix $\mathbf{F} \in \mathbb{R}^{d \times d}$ requires $\approx 4 \text{ Terabytes}$ of memory, and solving $\mathbf{F} \mathbf{x} = \mathbf{g}$ via direct matrix inversion ($\mathbf{F}^{-1} \mathbf{g}$) requires $\mathcal{O}(d^3) \approx 10^{18}$ floating-point operations.
Our mathematical goal is two-fold:
1. Prove analytically that the action of the Fisher matrix on any search vector $\mathbf{v}$ ($\mathbf{F} \mathbf{v}$) can be computed exactly using automatic differentiation in $\mathcal{O}(d)$ memory via Pearlmutter's double backward technique.
2. Derive the Conjugate Gradient (CG) algorithm from first principles as an exact iterative solver for $\mathbf{F} \mathbf{x} = \mathbf{g}$ that operates strictly through Fisher-vector products, guaranteeing monotone residual reduction and termination in at most $d$ iterations without ever forming matrix $\mathbf{F}$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Twice Continuous Differentiability ($C^2$):** The parameterized log-likelihood $\log \pi_{\boldsymbol{\theta}}(a \mid s)$ is twice continuously differentiable with respect to parameter vector $\boldsymbol{\theta}$. By Schwarz's theorem (Clairaut's theorem on mixed partials), mixed second derivatives commute: $\frac{\partial^2}{\partial \theta_i \partial \theta_j} = \frac{\partial^2}{\partial \theta_j \partial \theta_i}$.
2. **Symmetric Positive Definiteness:** $\mathbf{F} \in \mathbb{R}^{d \times d}$ is symmetric and positive definite ($\mathbf{F} \succ 0$). In empirical settings where $\mathbf{F}$ is rank-deficient due to finite batch sampling, a small damping factor $\delta_{\text{damp}} > 0$ is added ($\mathbf{F}_{\text{damp}} = \mathbf{F} + \delta_{\text{damp}} \mathbf{I}$) to guarantee strict positive definiteness.
3. **Constant Search Vector:** The search vector $\mathbf{v} \in \mathbb{R}^d$ is treated as an independent, constant vector during differentiation with respect to $\boldsymbol{\theta}$: $\frac{\partial v_j}{\partial \theta_i} = 0$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
- *Pearlmutter's FVP Trick:* The Fisher matrix $\mathbf{F}$ is the Jacobian of the gradient vector field $\nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}}$. Multiplying a Jacobian by a vector $\mathbf{v}$ represents the *directional derivative* of the gradient field along direction $\mathbf{v}$. By the chain rule, this directional derivative can be computed by first taking the inner product of the gradient field with $\mathbf{v}$ (which collapses the $d$-dimensional vector into a single scalar $y$) and then taking the gradient of that single scalar! This requires only one additional backward autodiff pass, consuming $\mathcal{O}(d)$ memory.
- *Conjugate Gradient Geometry:* Standard gradient descent takes steps orthogonal in Euclidean space ($\mathbf{r}_{k+1}^\top \mathbf{r}_k = 0$), which causes inefficient "zig-zagging" down narrow parabolic valleys. Conjugate Gradient eliminates this inefficiency by choosing search directions that are *mutually conjugate with respect to the metric tensor $\mathbf{F}$*: $\mathbf{p}_i^\top \mathbf{F} \mathbf{p}_j = 0$ for all $i \ne j$. Each step performs an exact line minimization along $\mathbf{p}_k$, guaranteeing that the error component along $\mathbf{p}_k$ is reduced to zero permanently, never to be undone by future iterations.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: First-Principles Derivation of Pearlmutter's Fisher-Vector Product.*
Let $\bar{D}_{\text{KL}}(\boldsymbol{\theta}) \triangleq \mathbb{E}_{s \sim d^\pi} [D_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}}(\cdot \mid s) \parallel \pi_{\boldsymbol{\theta}}(\cdot \mid s))]$.
The Fisher Information Matrix is the Hessian of average KL divergence evaluated at $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}$:
$$\mathbf{F} \triangleq \left. \nabla_{\boldsymbol{\theta}}^2 \bar{D}_{\text{KL}}(\boldsymbol{\theta}) \right|_{\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}}$$
In component notation, the $(i, j)$-th entry of $\mathbf{F}$ is:
$$F_{ij} = \left. \frac{\partial^2 \bar{D}_{\text{KL}}}{\partial \theta_i \partial \theta_j} \right|_{\boldsymbol{\theta}_{\text{old}}}$$
Now, consider the $i$-th component of the matrix-vector product $\mathbf{w} = \mathbf{F} \mathbf{v}$ for an arbitrary constant vector $\mathbf{v} = [v_1, \dots, v_d]^\top$:
$$w_i = (\mathbf{F} \mathbf{v})_i = \sum_{j=1}^d F_{ij} v_j = \sum_{j=1}^d \left( \left. \frac{\partial^2 \bar{D}_{\text{KL}}}{\partial \theta_i \partial \theta_j} \right|_{\boldsymbol{\theta}_{\text{old}}} \right) v_j$$
Since $\bar{D}_{\text{KL}} \in C^2$, by Clairaut's theorem mixed partial derivatives commute: $\frac{\partial^2 \bar{D}_{\text{KL}}}{\partial \theta_i \partial \theta_j} = \frac{\partial}{\partial \theta_i} \left( \frac{\partial \bar{D}_{\text{KL}}}{\partial \theta_j} \right)$.
Furthermore, since $v_j$ is independent of $\theta_i$, we can pull the scalar $v_j$ inside the derivative:
$$w_i = \sum_{j=1}^d \frac{\partial}{\partial \theta_i} \left( \frac{\partial \bar{D}_{\text{KL}}}{\partial \theta_j} v_j \right)$$
By linearity of differentiation, exchange the summation and differentiation operator:
$$w_i = \frac{\partial}{\partial \theta_i} \left( \sum_{j=1}^d \frac{\partial \bar{D}_{\text{KL}}}{\partial \theta_j} v_j \right)$$
Recognize the term in parentheses as the standard Euclidean inner product:
$$\sum_{j=1}^d \frac{\partial \bar{D}_{\text{KL}}}{\partial \theta_j} v_j = \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}} \right)^\top \mathbf{v}$$
Therefore:
$$w_i = \frac{\partial}{\partial \theta_i} \left( \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}} \right)^\top \mathbf{v} \right)$$
Stacking all $d$ components into vector form proves the Pearlmutter Fisher-Vector Product Identity:
$$\mathbf{F} \mathbf{v} = \left. \nabla_{\boldsymbol{\theta}} \left( \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}}(\boldsymbol{\theta}) \right)^\top \mathbf{v} \right) \right|_{\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}} \quad \blacksquare$$

*Step 2: Equivalent Expectation Form for Policy Gradients.*
Alternatively, by the Fisher Information identity, $\mathbf{F} = \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)^\top \right]$.
Multiplying by $\mathbf{v}$:
$$\mathbf{F} \mathbf{v} = \mathbb{E}_{s, a} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \left( \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)^\top \mathbf{v} \right) \right]$$
Notice that $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)^\top \mathbf{v}$ is a simple scalar. Thus, $\mathbf{F}\mathbf{v}$ is simply the expectation of the score vector scaled by this scalar product, completely avoiding matrix operations!

*Step 3: Quadratic Optimization Equivalence of $\mathbf{F} \mathbf{x} = \mathbf{g}$.*
Solving the linear system $\mathbf{F} \mathbf{x} = \mathbf{g}$ with $\mathbf{F} \succ 0$ is mathematically equivalent to finding the unconstrained global minimizer of the strictly convex quadratic energy function:
$$\phi(\mathbf{x}) \triangleq \frac{1}{2} \mathbf{x}^\top \mathbf{F} \mathbf{x} - \mathbf{g}^\top \mathbf{x}$$
The gradient of $\phi(\mathbf{x})$ is:
$$\nabla \phi(\mathbf{x}) = \mathbf{F} \mathbf{x} - \mathbf{g} = -\mathbf{r}(\mathbf{x})$$
where $\mathbf{r}(\mathbf{x}) \triangleq \mathbf{g} - \mathbf{F} \mathbf{x}$ is the linear residual. The stationary point $\nabla \phi(\mathbf{x}) = \mathbf{0}$ directly yields $\mathbf{F} \mathbf{x} = \mathbf{g}$.

*Step 4: Conjugate Directions and Exact 1D Line Minimization.*
Let $\{\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_{k-1}\}$ be a set of mutually $\mathbf{F}$-conjugate directions:
$$\mathbf{p}_i^\top \mathbf{F} \mathbf{p}_j = 0 \quad \forall i \ne j, \quad \text{and} \quad \mathbf{p}_i^\top \mathbf{F} \mathbf{p}_i > 0$$
At iteration $k$, given current iterate $\mathbf{x}_k$ and search direction $\mathbf{p}_k$, the next iterate is:
$$\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k$$
We select $\alpha_k$ to minimize $\phi(\mathbf{x}_k + \alpha_k \mathbf{p}_k)$ along the line:
$$\frac{d}{d \alpha_k} \phi(\mathbf{x}_k + \alpha_k \mathbf{p}_k) = \mathbf{p}_k^\top \nabla \phi(\mathbf{x}_k + \alpha_k \mathbf{p}_k) = \mathbf{p}_k^\top \left( \mathbf{F}(\mathbf{x}_k + \alpha_k \mathbf{p}_k) - \mathbf{g} \right) = 0$$
Using the residual definition $\mathbf{r}_k = \mathbf{g} - \mathbf{F} \mathbf{x}_k$:
$$\mathbf{p}_k^\top \left( \alpha_k \mathbf{F} \mathbf{p}_k - \mathbf{r}_k \right) = 0 \implies \alpha_k \mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k = \mathbf{p}_k^\top \mathbf{r}_k$$
Solving for step size $\alpha_k$:
$$\alpha_k = \frac{\mathbf{p}_k^\top \mathbf{r}_k}{\mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k} \quad \text{(Equation 1)}$$

*Step 5: Residual Recurrence Relation.*
Update the residual vector:
$$\mathbf{r}_{k+1} = \mathbf{g} - \mathbf{F} \mathbf{x}_{k+1} = \mathbf{g} - \mathbf{F}(\mathbf{x}_k + \alpha_k \mathbf{p}_k) = (\mathbf{g} - \mathbf{F} \mathbf{x}_k) - \alpha_k \mathbf{F} \mathbf{p}_k$$
$$\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{F} \mathbf{p}_k \quad \text{(Equation 2)}$$
Notice that computing $\mathbf{r}_{k+1}$ requires only the single matrix-vector product $\mathbf{F} \mathbf{p}_k$ already computed for $\alpha_k$.

*Step 6: Gram-Schmidt Conjugacy Update for New Direction $\mathbf{p}_{k+1}$.*
We generate the new search direction $\mathbf{p}_{k+1}$ by $\mathbf{F}$-orthogonalizing the new residual $\mathbf{r}_{k+1}$ against previous search directions.
In the Conjugate Gradient method, $\mathbf{p}_{k+1}$ is constructed as a linear combination of $\mathbf{r}_{k+1}$ and the previous direction $\mathbf{p}_k$:
$$\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k \quad \text{(Equation 3)}$$
Imposing the $\mathbf{F}$-conjugacy condition $\mathbf{p}_{k+1}^\top \mathbf{F} \mathbf{p}_k = 0$:
$$(\mathbf{r}_{k+1} + \beta_k \mathbf{p}_k)^\top \mathbf{F} \mathbf{p}_k = 0 \implies \mathbf{r}_{k+1}^\top \mathbf{F} \mathbf{p}_k + \beta_k \mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k = 0$$
Solving for $\beta_k$:
$$\beta_k = -\frac{\mathbf{r}_{k+1}^\top \mathbf{F} \mathbf{p}_k}{\mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k} \quad \text{(Equation 4)}$$

*Step 7: Simplification using Residual Orthogonality.*
We prove by induction that the residuals satisfy mutual Euclidean orthogonality:
$$\mathbf{r}_i^\top \mathbf{r}_j = 0 \quad \forall i \ne j, \quad \text{and} \quad \mathbf{p}_k^\top \mathbf{r}_k = \mathbf{r}_k^\top \mathbf{r}_k$$
From Equation 3, $\mathbf{p}_k = \mathbf{r}_k + \beta_{k-1} \mathbf{p}_{k-1}$.
Taking the inner product with $\mathbf{r}_k$:
$$\mathbf{p}_k^\top \mathbf{r}_k = \mathbf{r}_k^\top \mathbf{r}_k + \beta_{k-1} \mathbf{p}_{k-1}^\top \mathbf{r}_k$$
By induction, $\mathbf{r}_k$ is orthogonal to the subspace spanned by $\{\mathbf{p}_0, \dots, \mathbf{p}_{k-1}\}$, so $\mathbf{p}_{k-1}^\top \mathbf{r}_k = 0$.
Therefore:
$$\mathbf{p}_k^\top \mathbf{r}_k = \mathbf{r}_k^\top \mathbf{r}_k = \|\mathbf{r}_k\|^2 \quad \text{(Equation 5)}$$
Substituting Equation 5 into Equation 1 simplifies the step size to:
$$\alpha_k = \frac{\mathbf{r}_k^\top \mathbf{r}_k}{\mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k}$$
Now rearrange Equation 2 to express $\mathbf{F} \mathbf{p}_k$:
$$\mathbf{F} \mathbf{p}_k = \frac{1}{\alpha_k} (\mathbf{r}_k - \mathbf{r}_{k+1})$$
Substitute this into the numerator of Equation 4:
$$\mathbf{r}_{k+1}^\top \mathbf{F} \mathbf{p}_k = \frac{1}{\alpha_k} \mathbf{r}_{k+1}^\top (\mathbf{r}_k - \mathbf{r}_{k+1}) = \frac{1}{\alpha_k} \left( \mathbf{r}_{k+1}^\top \mathbf{r}_k - \mathbf{r}_{k+1}^\top \mathbf{r}_{k+1} \right)$$
By orthogonality $\mathbf{r}_{k+1}^\top \mathbf{r}_k = 0$, so:
$$\mathbf{r}_{k+1}^\top \mathbf{F} \mathbf{p}_k = -\frac{1}{\alpha_k} \|\mathbf{r}_{k+1}\|^2$$
For the denominator of Equation 4, using $\alpha_k = \frac{\|\mathbf{r}_k\|^2}{\mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k}$:
$$\mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k = \frac{1}{\alpha_k} \|\mathbf{r}_k\|^2$$
Dividing numerator by denominator in Equation 4:
$$\beta_k = -\frac{-\frac{1}{\alpha_k} \|\mathbf{r}_{k+1}\|^2}{\frac{1}{\alpha_k} \|\mathbf{r}_k\|^2} = \frac{\mathbf{r}_{k+1}^\top \mathbf{r}_{k+1}}{\mathbf{r}_k^\top \mathbf{r}_k}$$
This yields the celebrated Fletcher-Reeves conjugacy formula:
$$\beta_k = \frac{\|\mathbf{r}_{k+1}\|^2}{\|\mathbf{r}_k\|^2}$$

*Step 8: Finite-Dimensional Termination.*
Since each non-zero residual $\mathbf{r}_k$ is mutually orthogonal to all previous residuals $\{\mathbf{r}_0, \dots, \mathbf{r}_{k-1}\}$, the vectors $\{\mathbf{r}_0, \dots, \mathbf{r}_{k-1}\}$ form an orthogonal basis of the Krylov subspace $\mathcal{K}_k(\mathbf{F}, \mathbf{g}) = \operatorname{span}\{\mathbf{g}, \mathbf{F}\mathbf{g}, \dots, \mathbf{F}^{k-1}\mathbf{g}\}$.
In an exact arithmetic vector space $\mathbb{R}^d$, there can be at most $d$ mutually orthogonal non-zero vectors.
Hence, the residual must vanish identically ($\mathbf{r}_k = \mathbf{0}$) for some $k \le d$, terminating at the exact analytical solution $\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$.
Crucially, every single iteration requires only one matrix-vector product $\mathbf{v} = \mathbf{F} \mathbf{p}_k$ evaluated via Pearlmutter's FVP trick. The full matrix $\mathbf{F}$ is never formed or stored. $\blacksquare$

---

## 3. Geometric & Physical Interpretation

### The Trust Region Ellipsoid
In parameter space $\mathbb{R}^d$:
- The constraint $\frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \le \delta$ defines a high-dimensional **hyper-ellipsoid**.
- The semi-axes of this ellipsoid are proportional to $1/\sqrt{\lambda_i}$, where $\lambda_i$ are the eigenvalues of $\mathbf{F}$. In directions where the policy is sensitive (large $\lambda_i$), the ellipsoid is thin and narrow; in insensitive directions, the ellipsoid is elongated.
- The Conjugate Gradient algorithm finds the point on this ellipsoid that extends furthest in the direction of objective gradient $\mathbf{g}$.
- The Backtracking Line Search pulls the step backward along the chord if the curvature deviates from quadratic flatness.

```
                         Trust Region Ellipsoid
                        (1/2) dtheta^T F dtheta <= delta
                                   _____
                                .-'     `-.
                              .'     g     `.   * Step proposed by CG
                             /        \      \ /
                            |          *======* (Normalized boundary)
                            |         /       |
                             \       *       /
                              `.   theta_old.'
                                `-._____.-'
```

---

## 4. Real-World Analogy: The Rock Climber's Safety Harness

Imagine a rock climber navigating a sheer vertical cliff:
- **Vanilla Policy Gradient:** Takes a blind 3-foot leap upward toward the next handhold. If the hold breaks, the climber plummets to the ground.
- **Natural Policy Gradient:** Computes the exact tension vector on the rope, but requires solving complex multi-body physics equations on a supercomputer before each move.
- **TRPO:** The climber is attached to an adjustable **safety harness with dynamic friction braking**:
  1. The harness strictly limits any single upward move to within a guaranteed safe radius of 1 foot ($\bar{D}_{\text{KL}} \le \delta$).
  2. The climber tests the foothold with increasing weight: first $100\%$ force, then $50\%$, then $25\%$ (Backtracking Line Search).
  3. If the rock face crumbles at any point, the harness locks, and the climber resets to their last secure stance. A fall is mathematically impossible!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 2-Parameter Quadratic System
Let us execute the exact **Conjugate Gradient (CG)** solver and **Trust Region Scaling** step-by-step by hand!

**Objective Gradient Vector:**
$$\mathbf{g} = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix}$$

**Fisher Information Matrix:**
$$\mathbf{F} = \begin{bmatrix} 2.0000 & 0.5000 \\ 0.5000 & 1.0000 \end{bmatrix}$$

**Trust Region Boundary:**
$$\delta = 0.0500$$

We will compute:
1. Conjugate Gradient Iteration 0: Matrix-vector product $\mathbf{F} \mathbf{p}_0$, curvature, step size $\alpha_0$, updated position $\mathbf{x}_1$, residual $\mathbf{r}_1$.
2. Conjugate Gradient Iteration 1: Beta factor $\beta_0$, conjugate search direction $\mathbf{p}_1$, matrix-vector product $\mathbf{F} \mathbf{p}_1$, step size $\alpha_1$, final solution $\mathbf{x}_2$.
3. Exact analytical solution verification via $\mathbf{F}^{-1} \mathbf{g}$.
4. Quadratic form $\mathbf{x}^\top \mathbf{F} \mathbf{x}$.
5. Final trust region step scaling factor and update vector $\Delta \boldsymbol{\theta}^*$.

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Value / Matrix |
| :--- | :--- | :--- |
| $\mathbf{g}$ | Policy Surrogate Gradient | $[2.0000, 1.0000]^\top$ |
| $\mathbf{F}$ | Fisher Information Matrix | $\begin{bmatrix} 2.0 & 0.5 \\ 0.5 & 1.0 \end{bmatrix}$ |
| $\delta$ | Maximum Allowable KL Divergence | $0.0500$ |
| $\mathbf{x}_k$ | CG Approximate Solution to $\mathbf{F}\mathbf{x} = \mathbf{g}$ | Vectors in $\mathbb{R}^2$ |
| $\mathbf{r}_k$ | CG Residual Vector | $\mathbf{g} - \mathbf{F} \mathbf{x}_k$ |
| $\mathbf{p}_k$ | CG Conjugate Search Direction | Conjugate to all previous $\mathbf{p}$ |
| $\Delta \boldsymbol{\theta}^*$ | Final Scaled TRPO Step | $\sqrt{\frac{2\delta}{\mathbf{x}^\top \mathbf{F} \mathbf{x}}} \mathbf{x}$ |

---

### 5.3 Step 1: Conjugate Gradient — Iteration 0

**Initialization:**
- Initial guess: $\mathbf{x}_0 = \begin{bmatrix} 0.0000 \\ 0.0000 \end{bmatrix}$
- Initial residual: $\mathbf{r}_0 = \mathbf{g} - \mathbf{F} \mathbf{x}_0 = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix}$
- Initial search direction: $\mathbf{p}_0 = \mathbf{r}_0 = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix}$

**Compute Matrix-Vector Product $\mathbf{v}_0 = \mathbf{F} \mathbf{p}_0$:**
$$\mathbf{v}_0 = \begin{bmatrix} 2.0000 & 0.5000 \\ 0.5000 & 1.0000 \end{bmatrix} \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix} = \begin{bmatrix} (2.0 \times 2.0) + (0.5 \times 1.0) \\ (0.5 \times 2.0) + (1.0 \times 1.0) \end{bmatrix} = \begin{bmatrix} 4.0 + 0.5 \\ 1.0 + 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 4.5000 \\ 2.0000 \end{bmatrix}}$$

**Compute Curvature along $\mathbf{p}_0$:**
$$\mathbf{p}_0^\top \mathbf{v}_0 = \mathbf{p}_0^\top \mathbf{F} \mathbf{p}_0 = (2.0000 \times 4.5000) + (1.0000 \times 2.0000) = 9.0000 + 2.0000 = \mathbf{11.0000}$$

**Compute Residual Norm Squared:**
$$\mathbf{r}_0^\top \mathbf{r}_0 = (2.0000)^2 + (1.0000)^2 = 4.0000 + 1.0000 = \mathbf{5.0000}$$

**Compute Step Size $\alpha_0$:**
$$\alpha_0 = \frac{\mathbf{r}_0^\top \mathbf{r}_0}{\mathbf{p}_0^\top \mathbf{F} \mathbf{p}_0} = \frac{5.0000}{11.0000} \approx \mathbf{0.45455}$$

**Update Solution $\mathbf{x}_1$:**
$$\mathbf{x}_1 = \mathbf{x}_0 + \alpha_0 \mathbf{p}_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} + 0.45455 \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.90909 \\ 0.45455 \end{bmatrix}}$$

**Update Residual $\mathbf{r}_1$:**
$$\mathbf{r}_1 = \mathbf{r}_0 - \alpha_0 \mathbf{v}_0 = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix} - 0.45455 \begin{bmatrix} 4.5000 \\ 2.0000 \end{bmatrix}$$
$$\mathbf{r}_1 = \begin{bmatrix} 2.0000 - 2.04546 \\ 1.0000 - 0.90909 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.04545 \\ +0.09091 \end{bmatrix}}$$

---

### 5.4 Step 2: Conjugate Gradient — Iteration 1

**Compute New Residual Norm Squared:**
$$\mathbf{r}_1^\top \mathbf{r}_1 = (-0.04545)^2 + (0.09091)^2 \approx 0.002066 + 0.008264 = \mathbf{0.01033}$$

**Compute Gram-Schmidt Beta $\beta_0$:**
$$\beta_0 = \frac{\mathbf{r}_1^\top \mathbf{r}_1}{\mathbf{r}_0^\top \mathbf{r}_0} = \frac{0.01033}{5.0000} \approx \mathbf{0.002066}$$

**Compute Conjugate Direction $\mathbf{p}_1$:**
$$\mathbf{p}_1 = \mathbf{r}_1 + \beta_0 \mathbf{p}_0 = \begin{bmatrix} -0.04545 \\ +0.09091 \end{bmatrix} + 0.002066 \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} \approx \begin{bmatrix} -0.04545 + 0.00413 \\ 0.09091 + 0.00207 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.04132 \\ +0.09298 \end{bmatrix}}$$

**Compute Matrix-Vector Product $\mathbf{v}_1 = \mathbf{F} \mathbf{p}_1$:**
$$\mathbf{v}_1 = \begin{bmatrix} 2.0 & 0.5 \\ 0.5 & 1.0 \end{bmatrix} \begin{bmatrix} -0.04132 \\ 0.09298 \end{bmatrix} = \begin{bmatrix} -0.08264 + 0.04649 \\ -0.02066 + 0.09298 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.03615 \\ +0.07232 \end{bmatrix}}$$

**Compute Curvature along $\mathbf{p}_1$:**
$$\mathbf{p}_1^\top \mathbf{v}_1 = (-0.04132 \times -0.03615) + (0.09298 \times 0.07232) = 0.001494 + 0.006724 = \mathbf{0.008218}$$

**Compute Step Size $\alpha_1$:**
$$\alpha_1 = \frac{\mathbf{r}_1^\top \mathbf{r}_1}{\mathbf{p}_1^\top \mathbf{F} \mathbf{p}_1} = \frac{0.01033}{0.008218} \approx \mathbf{1.25700}$$

**Update Solution $\mathbf{x}_2$:**
$$\mathbf{x}_2 = \mathbf{x}_1 + \alpha_1 \mathbf{p}_1 = \begin{bmatrix} 0.90909 \\ 0.45455 \end{bmatrix} + 1.25700 \begin{bmatrix} -0.04132 \\ 0.09298 \end{bmatrix}$$
$$\mathbf{x}_2 = \begin{bmatrix} 0.90909 - 0.05194 \\ 0.45455 + 0.11688 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.85714 \\ 0.57143 \end{bmatrix}}$$

---

### 5.5 Step 3: Analytical Verification ($\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$)

Let us verify whether $\mathbf{x}_2$ is the exact analytical inverse solution!
Determinant of $\mathbf{F}$:
$$\det(\mathbf{F}) = (2.0 \times 1.0) - (0.5 \times 0.5) = 2.0 - 0.25 = \mathbf{1.7500} = \frac{7}{4}$$

Matrix Inverse $\mathbf{F}^{-1}$:
$$\mathbf{F}^{-1} = \frac{1}{1.7500} \begin{bmatrix} 1.0 & -0.5 \\ -0.5 & 2.0 \end{bmatrix} = \frac{4}{7} \begin{bmatrix} 1.0 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$$

Analytical Solution:
$$\mathbf{x}^* = \mathbf{F}^{-1} \mathbf{g} = \frac{4}{7} \begin{bmatrix} 1.0 & -0.5 \\ -0.5 & 2.0 \end{bmatrix} \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \frac{4}{7} \begin{bmatrix} 2.0 - 0.5 \\ -1.0 + 2.0 \end{bmatrix} = \frac{4}{7} \begin{bmatrix} 1.5 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 6/7 \\ 4/7 \end{bmatrix} \approx \mathbf{\begin{bmatrix} 0.85714 \\ 0.57143 \end{bmatrix}}$$

**The Conjugate Gradient solver converged to the exact analytical inverse in precisely 2 iterations!**

---

### 5.6 Step 4: Trust Region Step Normalization

Compute the quadratic curvature of the search direction:
$$\mathbf{x}^\top \mathbf{F} \mathbf{x} = \mathbf{x}^\top \mathbf{g} = (0.85714 \times 2.0000) + (0.57143 \times 1.0000) = 1.71429 + 0.57143 = \mathbf{2.28571} = \frac{16}{7}$$

Trust region constraint is $\delta = 0.0500$.
Compute scaling multiplier $\beta$:
$$\beta = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{F} \mathbf{x}}} = \sqrt{\frac{2 \times 0.0500}{2.28571}} = \sqrt{\frac{0.1000}{2.28571}} = \sqrt{0.04375} \approx \mathbf{0.20917}$$

Final TRPO Step $\Delta \boldsymbol{\theta}^*$:
$$\Delta \boldsymbol{\theta}^* = \beta \mathbf{x} = 0.20917 \begin{bmatrix} 0.85714 \\ 0.57143 \end{bmatrix} \approx \mathbf{\begin{bmatrix} 0.17929 \\ 0.11952 \end{bmatrix}}$$

Verify Constraint:
$$\frac{1}{2} (\Delta \boldsymbol{\theta}^*)^\top \mathbf{F} (\Delta \boldsymbol{\theta}^*) = \frac{1}{2} \beta^2 (\mathbf{x}^\top \mathbf{F} \mathbf{x}) = \frac{1}{2} (0.04375)(2.28571) = \mathbf{0.0500} = \delta \quad \checkmark$$

---

### 5.7 Visual Summary Grid: CG Walkthrough Matrix

| Iteration | Search Direction $\mathbf{p}_k$ | Matrix-Vector Product $\mathbf{F} \mathbf{p}_k$ | Curvature $\mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k$ | Step $\alpha_k$ | Solution Estimate $\mathbf{x}_k$ | Residual $\mathbf{r}_k$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 0$** | $[2.0000, 1.0000]^\top$ | $[4.5000, 2.0000]^\top$ | $11.0000$ | $0.45455$ | $[0.90909, 0.45455]^\top$ | $[-0.0455, +0.0909]^\top$ |
| **$k = 1$** | $[-0.0413, +0.0930]^\top$ | $[-0.0362, +0.0723]^\top$ | $0.00822$ | $1.25700$ | $\mathbf{[0.85714, 0.57143]^\top}$ | $\approx [0.0, 0.0]^\top$ |
| **Exact** | - | - | - | - | $\mathbf{[6/7, 4/7]^\top}$ | $\mathbf{0.0000}$ |

---

## 6. Solved Illustrations

### Illustration 1: Implementation of Hessian-Vector Product in PyTorch
**Problem:**
Write the exact 4-line PyTorch function that computes $\mathbf{F} \mathbf{v}$ without forming $\mathbf{F}$.
**Solution:**
```python
def hessian_vector_product(kl_div, policy_net, v):
    # First backward pass: gradient of KL divergence w.r.t parameters
    grads = torch.autograd.grad(kl_div, policy_net.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([g.view(-1) for g in grads])
    
    # Inner product with arbitrary search vector v
    grad_vector_product = torch.dot(flat_grad_kl, v)
    
    # Second backward pass: gradient of inner product w.r.t parameters
    hvp = torch.autograd.grad(grad_vector_product, policy_net.parameters(), retain_graph=True)
    return torch.cat([h.contiguous().view(-1) for h in hvp])
```
This fundamental snippet powers TRPO and natural gradient libraries worldwide. $\blacksquare$

---

### Illustration 2: 2-Iteration Conjugate Gradient Solve for $\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$ on a $2 \times 2$ Positive Definite Fisher Matrix

**Problem:**
Given the symmetric positive definite Fisher Information Matrix $\mathbf{F}$ and surrogate policy gradient $\mathbf{g}$:
$$\mathbf{F} = \begin{bmatrix} 2.0 & 1.0 \\ 1.0 & 2.0 \end{bmatrix}, \quad \mathbf{g} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$
1. Execute precisely 2 iterations of the Conjugate Gradient (CG) algorithm by hand starting from $\mathbf{x}_0 = [0.0, 0.0]^\top$.
2. Show step-by-step arithmetic for $\mathbf{v}_k = \mathbf{F} \mathbf{p}_k$, step length $\alpha_k$, iterate $\mathbf{x}_{k+1}$, residual $\mathbf{r}_{k+1}$, and conjugacy coefficient $\beta_k$.
3. Verify that the final iterate $\mathbf{x}_2$ identically matches the analytical matrix inverse solution $\mathbf{F}^{-1} \mathbf{g}$.

**Solution:**

#### Iteration 0 ($k = 0$):
- **Initial guess:** $\mathbf{x}_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$
- **Initial residual:** $\mathbf{r}_0 = \mathbf{g} - \mathbf{F} \mathbf{x}_0 = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} - \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$
- **Initial search direction:** $\mathbf{p}_0 = \mathbf{r}_0 = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$

**Step 1: Compute Matrix-Vector Product $\mathbf{v}_0 = \mathbf{F} \mathbf{p}_0$:**
$$\mathbf{v}_0 = \begin{bmatrix} 2.0 & 1.0 \\ 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} (2.0 \times 1.0) + (1.0 \times 2.0) \\ (1.0 \times 1.0) + (2.0 \times 2.0) \end{bmatrix} = \begin{bmatrix} 2.0 + 2.0 \\ 1.0 + 4.0 \end{bmatrix} = \begin{bmatrix} 4.0 \\ 5.0 \end{bmatrix}$$

**Step 2: Compute Residual Norm Squared and Curvature:**
$$\|\mathbf{r}_0\|^2 = \mathbf{r}_0^\top \mathbf{r}_0 = (1.0)^2 + (2.0)^2 = 1.0 + 4.0 = 5.0$$
$$\mathbf{p}_0^\top \mathbf{v}_0 = \mathbf{p}_0^\top \mathbf{F} \mathbf{p}_0 = (1.0 \times 4.0) + (2.0 \times 5.0) = 4.0 + 10.0 = 14.0$$

**Step 3: Compute Step Length $\alpha_0$:**
$$\alpha_0 = \frac{\mathbf{r}_0^\top \mathbf{r}_0}{\mathbf{p}_0^\top \mathbf{F} \mathbf{p}_0} = \frac{5.0}{14.0} = \frac{5}{14} \approx 0.357143$$

**Step 4: Update Solution $\mathbf{x}_1$:**
$$\mathbf{x}_1 = \mathbf{x}_0 + \alpha_0 \mathbf{p}_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} + \frac{5}{14} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 5/14 \\ 10/14 \end{bmatrix} = \begin{bmatrix} 5/14 \\ 5/7 \end{bmatrix} \approx \begin{bmatrix} 0.357143 \\ 0.714286 \end{bmatrix}$$

**Step 5: Update Residual $\mathbf{r}_1$:**
$$\mathbf{r}_1 = \mathbf{r}_0 - \alpha_0 \mathbf{v}_0 = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} - \frac{5}{14} \begin{bmatrix} 4.0 \\ 5.0 \end{bmatrix} = \begin{bmatrix} 14/14 - 20/14 \\ 28/14 - 25/14 \end{bmatrix} = \begin{bmatrix} -6/14 \\ 3/14 \end{bmatrix} = \begin{bmatrix} -3/7 \\ 3/14 \end{bmatrix} \approx \begin{bmatrix} -0.428571 \\ 0.214286 \end{bmatrix}$$

---

#### Iteration 1 ($k = 1$):
**Step 1: Compute New Residual Norm Squared and Beta Coefficient:**
$$\|\mathbf{r}_1\|^2 = \left(-\frac{6}{14}\right)^2 + \left(\frac{3}{14}\right)^2 = \frac{36}{196} + \frac{9}{196} = \frac{45}{196} \approx 0.229592$$
$$\beta_0 = \frac{\|\mathbf{r}_1\|^2}{\|\mathbf{r}_0\|^2} = \frac{45/196}{5.0} = \frac{45}{980} = \frac{9}{196} \approx 0.045918$$

**Step 2: Compute Conjugate Direction $\mathbf{p}_1$:**
$$\mathbf{p}_1 = \mathbf{r}_1 + \beta_0 \mathbf{p}_0 = \begin{bmatrix} -6/14 \\ 3/14 \end{bmatrix} + \frac{9}{196} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} -84/196 + 9/196 \\ 42/196 + 18/196 \end{bmatrix} = \begin{bmatrix} -75/196 \\ 60/196 \end{bmatrix} = \frac{15}{196} \begin{bmatrix} -5.0 \\ 4.0 \end{bmatrix} \approx \begin{bmatrix} -0.382653 \\ 0.306122 \end{bmatrix}$$

**Step 3: Compute Matrix-Vector Product $\mathbf{v}_1 = \mathbf{F} \mathbf{p}_1$:**
$$\mathbf{v}_1 = \begin{bmatrix} 2.0 & 1.0 \\ 1.0 & 2.0 \end{bmatrix} \left( \frac{15}{196} \begin{bmatrix} -5.0 \\ 4.0 \end{bmatrix} \right) = \frac{15}{196} \begin{bmatrix} (2 \times -5) + (1 \times 4) \\ (1 \times -5) + (2 \times 4) \end{bmatrix} = \frac{15}{196} \begin{bmatrix} -6.0 \\ 3.0 \end{bmatrix} = \frac{45}{196} \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix} \approx \begin{bmatrix} -0.459184 \\ 0.229592 \end{bmatrix}$$

**Step 4: Compute Curvature along $\mathbf{p}_1$:**
$$\mathbf{p}_1^\top \mathbf{v}_1 = \left( \frac{15}{196} \begin{bmatrix} -5.0 & 4.0 \end{bmatrix} \right) \left( \frac{45}{196} \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix} \right) = \frac{675}{196^2} \left( (-5 \times -2) + (4 \times 1) \right) = \frac{675 \times 14}{196^2} = \frac{675}{14 \times 196} = \frac{675}{2744} \approx 0.245991$$

**Step 5: Compute Step Length $\alpha_1$:**
$$\alpha_1 = \frac{\|\mathbf{r}_1\|^2}{\mathbf{p}_1^\top \mathbf{F} \mathbf{p}_1} = \frac{45/196}{675 / (14 \times 196)} = \frac{45 \times 14}{675} = \frac{14}{15} \approx 0.933333$$

**Step 6: Update Solution $\mathbf{x}_2$:**
$$\mathbf{x}_2 = \mathbf{x}_1 + \alpha_1 \mathbf{p}_1 = \begin{bmatrix} 5/14 \\ 10/14 \end{bmatrix} + \frac{14}{15} \left( \frac{15}{196} \begin{bmatrix} -5.0 \\ 4.0 \end{bmatrix} \right) = \begin{bmatrix} 5/14 \\ 10/14 \end{bmatrix} + \frac{14}{196} \begin{bmatrix} -5.0 \\ 4.0 \end{bmatrix} = \begin{bmatrix} 5/14 \\ 10/14 \end{bmatrix} + \frac{1}{14} \begin{bmatrix} -5.0 \\ 4.0 \end{bmatrix}$$
$$\mathbf{x}_2 = \begin{bmatrix} (5 - 5)/14 \\ (10 + 4)/14 \end{bmatrix} = \begin{bmatrix} 0/14 \\ 14/14 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$

**Step 7: Update Final Residual $\mathbf{r}_2$:**
$$\mathbf{r}_2 = \mathbf{r}_1 - \alpha_1 \mathbf{v}_1 = \begin{bmatrix} -6/14 \\ 3/14 \end{bmatrix} - \frac{14}{15} \left( \frac{45}{196} \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix} \right) = \begin{bmatrix} -6/14 \\ 3/14 \end{bmatrix} - \frac{3}{14} \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} -6/14 + 6/14 \\ 3/14 - 3/14 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

---

#### Analytical Verification:
Determinant: $\det(\mathbf{F}) = (2.0 \times 2.0) - (1.0 \times 1.0) = 4.0 - 1.0 = 3.0$.
Inverse matrix:
$$\mathbf{F}^{-1} = \frac{1}{3.0} \begin{bmatrix} 2.0 & -1.0 \\ -1.0 & 2.0 \end{bmatrix}$$
Direct solve:
$$\mathbf{x}^* = \mathbf{F}^{-1} \mathbf{g} = \frac{1}{3.0} \begin{bmatrix} 2.0 & -1.0 \\ -1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \frac{1}{3.0} \begin{bmatrix} (2 \times 1) - (1 \times 2) \\ (-1 \times 1) + (2 \times 2) \end{bmatrix} = \frac{1}{3.0} \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$
The Conjugate Gradient algorithm converged to the exact analytical inverse in exactly $d = 2$ iterations with residual $\|\mathbf{r}_2\| = 0.0000$. $\blacksquare$

---

### Illustration 3: Exact TRPO Step Length $\beta = \sqrt{\frac{2\delta}{\mathbf{x}^\top \mathbf{g}}}$ and Backtracking Line Search with KL Constraint Verification ($\delta = 0.01$)

**Problem:**
Using the CG solution $\mathbf{x} = [0.0, 1.0]^\top$ from Illustration 2 for $\mathbf{F} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ and $\mathbf{g} = [1.0, 2.0]^\top$:
1. Compute the exact trust region step length multiplier $\beta = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{g}}}$ for trust region radius $\delta = 0.0100$.
2. Form the proposed full update step $\Delta \boldsymbol{\theta}_0 = \beta \mathbf{x}$ and verify the second-order KL divergence approximation.
3. Suppose the true environment policy exhibits higher-order cubic non-linear distortion such that:
   $$\bar{D}_{\text{KL}}(\Delta \boldsymbol{\theta}) = \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} + 3.0 (\Delta \theta_2)^3$$
   and the surrogate return exhibits quadratic degradation:
   $$L(\Delta \boldsymbol{\theta}) = \mathbf{g}^\top \Delta \boldsymbol{\theta} - 15.0 (\Delta \theta_2)^2$$
4. Perform the backtracking line search with reduction rate $\alpha = 0.50$ ($j = 0, 1, 2, \dots$) to find the smallest backtracking iteration $j$ that simultaneously satisfies:
   - Trust region enforcement: $\bar{D}_{\text{KL}}(\alpha^j \Delta \boldsymbol{\theta}_0) \le \delta$
   - Surrogate improvement: $L(\alpha^j \Delta \boldsymbol{\theta}_0) > 0$.

**Solution:**

#### Step 1: Compute Quadratic Curvature and Step Scaling Factor $\beta$
Quadratic curvature inner product:
$$\mathbf{x}^\top \mathbf{g} = \begin{bmatrix} 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = (0.0 \times 1.0) + (1.0 \times 2.0) = 2.0000$$
(Equivalently, $\mathbf{x}^\top \mathbf{F} \mathbf{x} = [0, 1] \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = [0, 1] \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 2.0000$).
With trust region $\delta = 0.0100$:
$$\beta = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{g}}} = \sqrt{\frac{2 \times 0.0100}{2.0000}} = \sqrt{\frac{0.0200}{2.0000}} = \sqrt{0.0100} = 0.1000$$

#### Step 2: Propose Full Step $\Delta \boldsymbol{\theta}_0$ and Check Second-Order Approximation
$$\Delta \boldsymbol{\theta}_0 = \beta \mathbf{x} = 0.1000 \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.0000 \\ 0.1000 \end{bmatrix}$$
Second-order Taylor expansion check:
$$\frac{1}{2} \Delta \boldsymbol{\theta}_0^\top \mathbf{F} \Delta \boldsymbol{\theta}_0 = \frac{1}{2} \begin{bmatrix} 0.0 & 0.1 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix} = \frac{1}{2} \begin{bmatrix} 0.0 & 0.1 \end{bmatrix} \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} = \frac{1}{2} (0.0200) = 0.0100 = \delta \quad \checkmark$$

---

#### Step 3: Backtracking Line Search Evaluation

**Backtracking Iteration $j = 0$ (Full Step: decay factor $\alpha^0 = 1.00$):**
- Candidate step: $\Delta \boldsymbol{\theta}^{(0)} = 1.0 \times [0.0, 0.1]^\top = [0.0000, 0.1000]^\top$.
- True non-linear KL divergence:
  $$\bar{D}_{\text{KL}}(\Delta \boldsymbol{\theta}^{(0)}) = \frac{1}{2} (0.0200) + 3.0 \times (0.1000)^3 = 0.0100 + 3.0 \times (0.0010) = 0.0100 + 0.0030 = 0.0130$$
- Surrogate objective value:
  $$L(\Delta \boldsymbol{\theta}^{(0)}) = \mathbf{g}^\top \Delta \boldsymbol{\theta}^{(0)} - 15.0 \times (0.1000)^2 = (2.0 \times 0.1000) - 15.0 \times (0.0100) = 0.2000 - 0.1500 = +0.0500$$
- Check criteria:
  - Surrogate improvement: $L = +0.0500 > 0$ (Satisfied)
  - Trust region constraint: $\bar{D}_{\text{KL}} = 0.0130 > \delta = 0.0100$ (**VIOLATED!**)
- Outcome: **Reject step at $j = 0$ due to trust region boundary breach.**

**Backtracking Iteration $j = 1$ (Decay factor $\alpha^1 = 0.50$):**
- Candidate step: $\Delta \boldsymbol{\theta}^{(1)} = 0.50 \times [0.0, 0.1]^\top = [0.0000, 0.0500]^\top$.
- Quadratic Fisher form:
  $$\frac{1}{2} (\Delta \boldsymbol{\theta}^{(1)})^\top \mathbf{F} (\Delta \boldsymbol{\theta}^{(1)}) = (0.5)^2 \times 0.0100 = 0.25 \times 0.0100 = 0.002500$$
- Cubic distortion term:
  $$3.0 \times (0.0500)^3 = 3.0 \times (0.000125) = 0.000375$$
- True non-linear KL divergence:
  $$\bar{D}_{\text{KL}}(\Delta \boldsymbol{\theta}^{(1)}) = 0.002500 + 0.000375 = 0.002875$$
- Surrogate objective value:
  $$L(\Delta \boldsymbol{\theta}^{(1)}) = \mathbf{g}^\top \Delta \boldsymbol{\theta}^{(1)} - 15.0 \times (0.0500)^2 = (2.0 \times 0.0500) - 15.0 \times (0.0025) = 0.1000 - 0.0375 = +0.0625$$
- Check criteria:
  - Surrogate improvement: $L = +0.0625 > 0$ (Satisfied: objective improved even more than $j=0$ due to reduced quadratic penalty!)
  - Trust region constraint: $\bar{D}_{\text{KL}} = 0.002875 \le 0.0100$ (Satisfied: well within the safety boundary!)
- Outcome: **Accept update at backtracking iteration $j = 1$!**
- Final accepted TRPO update:
  $$\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} + \begin{bmatrix} 0.0000 \\ 0.0500 \end{bmatrix} \quad \blacksquare$$

---

### Illustration 4: Monotonic Improvement Bound Evaluation on a 3-State MDP

**Problem:**
Consider a 3-state MDP $\mathcal{S} = \{s_1, s_2, s_3\}$ with two discrete actions $\mathcal{A} = \{a_1, a_2\}$, discount factor $\gamma = 0.50$, and deterministic initial state $s_0 = s_1$ ($\mu = [1.0, 0.0, 0.0]^\top$).
The state transitions and rewards are:
- In $s_1$: action $a_1 \to s_2$ ($r = 1.0$); action $a_2 \to s_1$ ($r = 0.0$).
- In $s_2$: action $a_1 \to s_3$ ($r = 2.0$); action $a_2 \to s_2$ ($r = 1.0$).
- In $s_3$: action $a_1 \to s_3$ ($r = 3.0$); action $a_2 \to s_3$ ($r = 0.0$).

Let the baseline policy $\pi$ be uniform random: $\pi(a_1 \mid s) = 0.50, \pi(a_2 \mid s) = 0.50$ for all $s$.
Let the proposed candidate policy $\tilde{\pi}$ take action $a_1$ with higher probability: $\tilde{\pi}(a_1 \mid s) = 0.60, \tilde{\pi}(a_2 \mid s) = 0.40$ for all $s$.
1. Compute the exact value function $V^\pi$, action-values $Q^\pi(s, a)$, advantages $A^\pi(s, a)$, and baseline expected return $J(\pi)$.
2. Compute the unnormalized discounted state visitation distribution $d^\pi(s)$ and evaluate the TRPO surrogate objective $L_\pi(\tilde{\pi})$.
3. Compute the true candidate return $J(\tilde{\pi})$ and verify Kakade & Langford's Performance Difference Lemma: $J(\tilde{\pi}) - J(\pi) = \sum_s d^{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a \mid s) A^\pi(s, a)$.
4. Compute the maximum advantage magnitude $\epsilon = \max_{s, a} |A^\pi(s, a)|$, penalty constant $C = \frac{4 \epsilon \gamma}{(1 - \gamma)^2}$, maximum KL divergence $D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$, and evaluate the monotonic improvement lower bound $L_\pi(\tilde{\pi}) - C D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$. Show that $J(\tilde{\pi}) \ge L_\pi(\tilde{\pi}) - C D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$.

**Solution:**

#### Step 1: Compute $V^\pi, Q^\pi, A^\pi$, and $J(\pi)$
Under policy $\pi$, transition probabilities are the average of actions $a_1$ and $a_2$:
$$\mathbf{P}^\pi = 0.5 \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 1 \end{bmatrix} + 0.5 \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 & 0.0 \\ 0.0 & 0.5 & 0.5 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}$$
Expected immediate rewards:
$$\mathbf{r}^\pi = 0.5 \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \end{bmatrix} + 0.5 \begin{bmatrix} 0.0 \\ 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 1.5 \\ 1.5 \end{bmatrix}$$
Solve the Bellman equation $\mathbf{V}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi$ with $\gamma = 0.5$:
$$(\mathbf{I} - 0.5 \mathbf{P}^\pi) \mathbf{V}^\pi = \begin{bmatrix} 1 - 0.25 & -0.25 & 0.0 \\ 0.0 & 1 - 0.25 & -0.25 \\ 0.0 & 0.0 & 1 - 0.50 \end{bmatrix} \begin{bmatrix} V^\pi(s_1) \\ V^\pi(s_2) \\ V^\pi(s_3) \end{bmatrix} = \begin{bmatrix} 0.75 & -0.25 & 0.0 \\ 0.0 & 0.75 & -0.25 \\ 0.0 & 0.0 & 0.50 \end{bmatrix} \begin{bmatrix} V^\pi(s_1) \\ V^\pi(s_2) \\ V^\pi(s_3) \end{bmatrix} = \begin{bmatrix} 0.5 \\ 1.5 \\ 1.5 \end{bmatrix}$$
Back-substituting from bottom to top:
- $0.50 V^\pi(s_3) = 1.5 \implies V^\pi(s_3) = \frac{1.5}{0.50} = 3.0000$
- $0.75 V^\pi(s_2) - 0.25(3.0) = 1.5 \implies 0.75 V^\pi(s_2) = 1.5 + 0.75 = 2.25 \implies V^\pi(s_2) = \frac{2.25}{0.75} = 3.0000$
- $0.75 V^\pi(s_1) - 0.25(3.0) = 0.5 \implies 0.75 V^\pi(s_1) = 0.5 + 0.75 = 1.25 \implies V^\pi(s_1) = \frac{1.25}{0.75} = \frac{5}{3} \approx 1.666667$
Baseline return:
$$J(\pi) = V^\pi(s_1) = \frac{5}{3} \approx 1.666667$$

Compute $Q^\pi(s, a) = r(s, a) + \gamma V^\pi(s')$:
- $Q^\pi(s_1, a_1) = 1.0 + 0.5(3.0) = 2.5000 = \frac{5}{2}$
- $Q^\pi(s_1, a_2) = 0.0 + 0.5(5/3) = \frac{5}{6} \approx 0.833333$
- $Q^\pi(s_2, a_1) = 2.0 + 0.5(3.0) = 3.5000 = \frac{7}{2}$
- $Q^\pi(s_2, a_2) = 1.0 + 0.5(3.0) = 2.5000 = \frac{5}{2}$
- $Q^\pi(s_3, a_1) = 3.0 + 0.5(3.0) = 4.5000 = \frac{9}{2}$
- $Q^\pi(s_3, a_2) = 0.0 + 0.5(3.0) = 1.5000 = \frac{3}{2}$

Compute Advantages $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$:
- $A^\pi(s_1, a_1) = \frac{5}{2} - \frac{5}{3} = \mathbf{+\frac{5}{6}}, \quad A^\pi(s_1, a_2) = \frac{5}{6} - \frac{5}{3} = \mathbf{-\frac{5}{6}}$
- $A^\pi(s_2, a_1) = \frac{7}{2} - 3 = \mathbf{+\frac{1}{2}}, \quad A^\pi(s_2, a_2) = \frac{5}{2} - 3 = \mathbf{-\frac{1}{2}}$
- $A^\pi(s_3, a_1) = \frac{9}{2} - 3 = \mathbf{+\frac{3}{2}}, \quad A^\pi(s_3, a_2) = \frac{3}{2} - 3 = \mathbf{-\frac{3}{2}}$
Maximum advantage magnitude:
$$\epsilon \triangleq \max_{s, a} |A^\pi(s, a)| = \max\left( \frac{5}{6}, \frac{1}{2}, \frac{3}{2} \right) = \frac{3}{2} = 1.5000$$

#### Step 2: Compute $d^\pi$ and Surrogate Objective $L_\pi(\tilde{\pi})$
Unnormalized discounted state visitation $d^\pi = \boldsymbol{\mu}^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$:
$$\begin{bmatrix} d^\pi(s_1) & d^\pi(s_2) & d^\pi(s_3) \end{bmatrix} \begin{bmatrix} 0.75 & -0.25 & 0.0 \\ 0.0 & 0.75 & -0.25 \\ 0.0 & 0.0 & 0.50 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 & 0.0 \end{bmatrix}$$
- $0.75 d^\pi(s_1) = 1.0 \implies d^\pi(s_1) = \frac{4}{3} \approx 1.333333$
- $-0.25 d^\pi(s_1) + 0.75 d^\pi(s_2) = 0 \implies d^\pi(s_2) = \frac{1}{3} d^\pi(s_1) = \frac{4}{9} \approx 0.444444$
- $-0.25 d^\pi(s_2) + 0.50 d^\pi(s_3) = 0 \implies d^\pi(s_3) = \frac{1}{2} d^\pi(s_2) = \frac{2}{9} \approx 0.222222$
Expected advantage under candidate policy $\tilde{\pi}$ ($\tilde{\pi}(a_1)=0.6, \tilde{\pi}(a_2)=0.4$):
$$\sum_{a} \tilde{\pi}(a \mid s) A^\pi(s, a) = 0.6 A^\pi(s, a_1) + 0.4 (-A^\pi(s, a_1)) = 0.2 A^\pi(s, a_1)$$
- State $s_1$: $0.2 \times \frac{5}{6} = \frac{1}{6}$
- State $s_2$: $0.2 \times \frac{1}{2} = \frac{1}{10}$
- State $s_3$: $0.2 \times \frac{3}{2} = \frac{3}{10}$

Evaluate the surrogate objective:
$$L_\pi(\tilde{\pi}) = J(\pi) + \sum_{s} d^\pi(s) \sum_a \tilde{\pi}(a \mid s) A^\pi(s, a) = \frac{5}{3} + \left( \frac{4}{3} \times \frac{1}{6} + \frac{4}{9} \times \frac{1}{10} + \frac{2}{9} \times \frac{3}{10} \right)$$
$$= \frac{5}{3} + \left( \frac{4}{18} + \frac{4}{90} + \frac{6}{90} \right) = \frac{5}{3} + \left( \frac{20}{90} + \frac{4}{90} + \frac{6}{90} \right) = \frac{5}{3} + \frac{30}{90} = \frac{5}{3} + \frac{1}{3} = \frac{6}{3} = \mathbf{2.0000}$$

#### Step 3: Compute True Candidate Return $J(\tilde{\pi})$
Under policy $\tilde{\pi}$:
$$\mathbf{P}^{\tilde{\pi}} = \begin{bmatrix} 0.4 & 0.6 & 0.0 \\ 0.0 & 0.4 & 0.6 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}, \quad \mathbf{r}^{\tilde{\pi}} = 0.6 \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \end{bmatrix} + 0.4 \begin{bmatrix} 0.0 \\ 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 1.6 \\ 1.8 \end{bmatrix}$$
Solve $(\mathbf{I} - 0.5 \mathbf{P}^{\tilde{\pi}}) \mathbf{V}^{\tilde{\pi}} = \mathbf{r}^{\tilde{\pi}}$:
$$\begin{bmatrix} 0.8 & -0.3 & 0.0 \\ 0.0 & 0.8 & -0.3 \\ 0.0 & 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} V^{\tilde{\pi}}(s_1) \\ V^{\tilde{\pi}}(s_2) \\ V^{\tilde{\pi}}(s_3) \end{bmatrix} = \begin{bmatrix} 0.6 \\ 1.6 \\ 1.8 \end{bmatrix}$$
- $V^{\tilde{\pi}}(s_3) = \frac{1.8}{0.5} = 3.6000$
- $0.8 V^{\tilde{\pi}}(s_2) - 0.3(3.6) = 1.6 \implies 0.8 V^{\tilde{\pi}}(s_2) = 1.6 + 1.08 = 2.68 \implies V^{\tilde{\pi}}(s_2) = \frac{2.68}{0.8} = 3.3500$
- $0.8 V^{\tilde{\pi}}(s_1) - 0.3(3.35) = 0.6 \implies 0.8 V^{\tilde{\pi}}(s_1) = 0.6 + 1.005 = 1.605 \implies V^{\tilde{\pi}}(s_1) = \frac{1.605}{0.8} = \mathbf{2.00625} = \frac{321}{160}$
True performance:
$$J(\tilde{\pi}) = V^{\tilde{\pi}}(s_1) = \mathbf{2.00625}$$
True performance improvement:
$$J(\tilde{\pi}) - J(\pi) = 2.00625 - 1.666667 = \mathbf{+0.339583}$$

#### Step 4: Evaluate Monotonic Improvement Lower Bound
Penalty coefficient:
$$C = \frac{4 \epsilon \gamma}{(1 - \gamma)^2} = \frac{4 \times 1.50 \times 0.50}{(1 - 0.50)^2} = \frac{3.00}{0.25} = \mathbf{12.0000}$$
Kullback-Leibler divergence between $\pi$ and $\tilde{\pi}$ at each state:
$$D_{\text{KL}}(\pi \parallel \tilde{\pi}) = \sum_a \pi(a) \log \frac{\pi(a)}{\tilde{\pi}(a)} = 0.5 \ln\left(\frac{0.5}{0.6}\right) + 0.5 \ln\left(\frac{0.5}{0.4}\right) = 0.5 \ln\left( \frac{0.25}{0.24} \right) = 0.5 \ln\left( \frac{25}{24} \right)$$
$$D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) = 0.5 \ln(1.041667) \approx 0.5 \times 0.040822 = \mathbf{0.020411}$$
Penalty term:
$$C \cdot D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) = 12.0000 \times 0.020411 = \mathbf{0.244932}$$
Monotonic improvement lower bound:
$$L_\pi(\tilde{\pi}) - C \cdot D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) = 2.000000 - 0.244932 = \mathbf{1.755068}$$
Comparison:
$$J(\tilde{\pi}) = 2.006250 \ge 1.755068 \quad \checkmark$$
The true return $J(\tilde{\pi}) = 2.00625$ exceeds both the old return $J(\pi) = 1.66667$ and the theoretical monotonic lower bound ($1.75507$). Policy improvement is rigorously guaranteed! $\blacksquare$

---

### Illustration 5: Fisher-Vector Product Numerical Verification via PyTorch-style Pearlmutter Trick

**Problem:**
Consider a continuous 1D Gaussian policy $\pi_{\boldsymbol{\theta}}(a) = \mathcal{N}(\mu, \sigma^2)$ parameterized by $\boldsymbol{\theta} = [\mu, \rho]^\top \in \mathbb{R}^2$ where $\rho \triangleq \log \sigma$ (so $\sigma = e^\rho$).
Current parameter point:
$$\boldsymbol{\theta}_0 = [\mu_0, \rho_0]^\top = [1.0000, 0.5000]^\top$$
Search direction vector:
$$\mathbf{v} = [v_1, v_2]^\top = [0.6000, -0.8000]^\top$$
1. Compute the analytical Fisher Information Matrix $\mathbf{F}(\boldsymbol{\theta}_0)$ and the exact direct matrix-vector product $\mathbf{F} \mathbf{v}$.
2. Using the analytical KL divergence between two univariate Gaussians:
   $$D_{\text{KL}}(\boldsymbol{\theta}_0 \parallel \boldsymbol{\theta}) = (\rho - \rho_0) + \frac{e^{2\rho_0} + (\mu_0 - \mu)^2}{2 e^{2\rho}} - \frac{1}{2}$$
   execute the Pearlmutter automatic differentiation procedure step-by-step:
   - Compute the first gradient $\mathbf{g}_{\text{KL}}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} D_{\text{KL}}(\boldsymbol{\theta}_0 \parallel \boldsymbol{\theta})$.
   - Form the scalar inner product $y(\boldsymbol{\theta}) = \mathbf{g}_{\text{KL}}(\boldsymbol{\theta})^\top \mathbf{v}$.
   - Compute the second gradient $\nabla_{\boldsymbol{\theta}} y(\boldsymbol{\theta})$ evaluated at $\boldsymbol{\theta} = \boldsymbol{\theta}_0$.
3. Prove that the Pearlmutter autograd output $\nabla_{\boldsymbol{\theta}} y|_{\boldsymbol{\theta}_0}$ matches the direct product $\mathbf{F} \mathbf{v}$ to machine precision.

**Solution:**

#### Step 1: Analytical Fisher Matrix and Direct Product $\mathbf{F} \mathbf{v}$
For a Gaussian distribution with parameterization $[\mu, \rho]^\top$ where $\rho = \log \sigma$:
- $F_{\mu \mu} = \frac{1}{\sigma^2} = e^{-2\rho}$
- $F_{\mu \rho} = F_{\rho \mu} = 0$
- $F_{\rho \rho} = 2$ (since $\frac{\partial \sigma}{\partial \rho} = \sigma$, $F_{\sigma \sigma} = \frac{2}{\sigma^2} \implies F_{\rho \rho} = \frac{2}{\sigma^2} \left(\frac{\partial \sigma}{\partial \rho}\right)^2 = 2$)
Therefore, the exact Fisher matrix is diagonal:
$$\mathbf{F}(\boldsymbol{\theta}) = \begin{bmatrix} e^{-2\rho} & 0 \\ 0 & 2 \end{bmatrix}$$
At $\boldsymbol{\theta}_0 = [1.0000, 0.5000]^\top$ ($\rho_0 = 0.5000$):
$$\mathbf{F}(\boldsymbol{\theta}_0) = \begin{bmatrix} e^{-2(0.5)} & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} e^{-1.0} & 0 \\ 0 & 2 \end{bmatrix} \approx \begin{bmatrix} 0.367879 & 0.000000 \\ 0.000000 & 2.000000 \end{bmatrix}$$
Direct matrix-vector multiplication with $\mathbf{v} = [0.6000, -0.8000]^\top$:
$$\mathbf{F} \mathbf{v} = \begin{bmatrix} 0.367879 & 0.000000 \\ 0.000000 & 2.000000 \end{bmatrix} \begin{bmatrix} 0.6000 \\ -0.8000 \end{bmatrix} = \begin{bmatrix} 0.367879 \times 0.6000 \\ 2.000000 \times (-0.8000) \end{bmatrix} = \mathbf{\begin{bmatrix} +0.220728 \\ -1.600000 \end{bmatrix}}$$

---

#### Step 2: Pearlmutter Double-Backward Procedure by Hand

**Step 2a: First Backward Pass (Gradient of KL Divergence):**
Given $D_{\text{KL}}(\mu, \rho) = (\rho - \rho_0) + \frac{e^{2\rho_0} + (\mu_0 - \mu)^2}{2 e^{2\rho}} - \frac{1}{2}$.
Compute partial derivatives with respect to candidate parameters $\mu$ and $\rho$:
$$\frac{\partial D_{\text{KL}}}{\partial \mu} = \frac{1}{2 e^{2\rho}} \frac{\partial}{\partial \mu} \left( (\mu_0 - \mu)^2 \right) = \frac{-2(\mu_0 - \mu)}{2 e^{2\rho}} = \frac{\mu - \mu_0}{e^{2\rho}}$$
$$\frac{\partial D_{\text{KL}}}{\partial \rho} = 1 + \left( e^{2\rho_0} + (\mu_0 - \mu)^2 \right) \frac{\partial}{\partial \rho} \left( \frac{1}{2} e^{-2\rho} \right) = 1 - \frac{e^{2\rho_0} + (\mu_0 - \mu)^2}{e^{2\rho}}$$
Notice that evaluating at $\boldsymbol{\theta} = \boldsymbol{\theta}_0$ ($\mu = \mu_0, \rho = \rho_0$):
$$\left. \frac{\partial D_{\text{KL}}}{\partial \mu} \right|_{\boldsymbol{\theta}_0} = \frac{0}{e^{2\rho_0}} = 0, \quad \left. \frac{\partial D_{\text{KL}}}{\partial \rho} \right|_{\boldsymbol{\theta}_0} = 1 - \frac{e^{2\rho_0}}{e^{2\rho_0}} = 1 - 1 = 0$$
The first gradient is $\mathbf{0}$ at the reference point, as expected.

**Step 2b: Inner Product with Vector $\mathbf{v}$:**
Form the scalar function $y(\mu, \rho) \triangleq (\nabla_{\boldsymbol{\theta}} D_{\text{KL}})^\top \mathbf{v}$:
$$y(\mu, \rho) = v_1 \left( \frac{\mu - \mu_0}{e^{2\rho}} \right) + v_2 \left( 1 - \frac{e^{2\rho_0} + (\mu_0 - \mu)^2}{e^{2\rho}} \right)$$

**Step 2c: Second Backward Pass (Differentiating $y$ with respect to $\boldsymbol{\theta}$):**
Differentiate $y(\mu, \rho)$ with respect to $\mu$:
$$\frac{\partial y}{\partial \mu} = v_1 \frac{\partial}{\partial \mu} \left( \frac{\mu - \mu_0}{e^{2\rho}} \right) + v_2 \frac{\partial}{\partial \mu} \left( -\frac{(\mu_0 - \mu)^2}{e^{2\rho}} \right) = v_1 \left( \frac{1}{e^{2\rho}} \right) + v_2 \left( \frac{2(\mu_0 - \mu)}{e^{2\rho}} \right)$$
Evaluate at $\boldsymbol{\theta} = \boldsymbol{\theta}_0$ ($\mu = \mu_0, \rho = \rho_0$):
$$\left. \frac{\partial y}{\partial \mu} \right|_{\boldsymbol{\theta}_0} = v_1 e^{-2\rho_0} + v_2 (0) = v_1 e^{-2\rho_0} = 0.6000 \times e^{-1.0} \approx \mathbf{+0.220728}$$

Differentiate $y(\mu, \rho)$ with respect to $\rho$:
$$\frac{\partial y}{\partial \rho} = v_1 (\mu - \mu_0) \frac{\partial}{\partial \rho} (e^{-2\rho}) + v_2 \left( e^{2\rho_0} + (\mu_0 - \mu)^2 \right) \frac{\partial}{\partial \rho} (-e^{-2\rho})$$
$$= v_1 (\mu - \mu_0) (-2 e^{-2\rho}) + v_2 \left( e^{2\rho_0} + (\mu_0 - \mu)^2 \right) (2 e^{-2\rho})$$
Evaluate at $\boldsymbol{\theta} = \boldsymbol{\theta}_0$ ($\mu = \mu_0, \rho = \rho_0$):
$$\left. \frac{\partial y}{\partial \rho} \right|_{\boldsymbol{\theta}_0} = v_1 (0) (-2 e^{-2\rho_0}) + v_2 (e^{2\rho_0} + 0)(2 e^{-2\rho_0}) = 0 + v_2 (2 \cdot e^{2\rho_0} e^{-2\rho_0}) = 2 v_2$$
$$= 2 \times (-0.8000) = \mathbf{-1.600000}$$

#### Step 3: Comparison and Verification
Stacking the second gradient results:
$$\nabla_{\boldsymbol{\theta}} y(\boldsymbol{\theta}_0) = \begin{bmatrix} \left. \frac{\partial y}{\partial \mu} \right|_{\boldsymbol{\theta}_0} \\ \left. \frac{\partial y}{\partial \rho} \right|_{\boldsymbol{\theta}_0} \end{bmatrix} = \mathbf{\begin{bmatrix} +0.220728 \\ -1.600000 \end{bmatrix}} \equiv \mathbf{F} \mathbf{v}$$
The Pearlmutter double-backward autodiff output reproduces the direct matrix-vector product with $0.0000\%$ error, verifying the foundation of TRPO's matrix-free optimization! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. The PPO Succession: Simplifying the Trust Region (Schulman et al., 2017)
PPO's design is directly motivated by TRPO's theoretical guarantees, simplified for practical scale:
- **From Constrained to Clipped:** TRPO enforces $D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta) \leq \delta$ via a Lagrangian. PPO replaces this with a clipped objective $\mathbb{E}_t[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$, providing an implicit soft trust region at $O(1)$ overhead versus TRPO's conjugate gradient solve.
- **Empirical Parity:** On MuJoCo and Atari benchmarks, PPO matches TRPO's asymptotic performance while running 10–15× faster, making TRPO primarily a theoretical baseline and PPO the production standard.
- **When TRPO Wins:** In robotic contact-rich manipulation (dexterous hand manipulation, in-hand object reorientation), TRPO's exact KL constraint prevents catastrophic policy collapses that PPO's soft clip occasionally allows.

### 2. Safety-Critical Robotics: Hard KL Guarantees
TRPO's hard trust region constraint remains essential when policy collapses carry physical consequences:
- **Legged Locomotion (ETH Zurich ANYmal, 2022):** TRPO-derived safe policy optimization ensures that each update to the ANYmal quadruped's gait controller cannot deviate too far from the current stable walking policy. A PPO collapse mid-training can destabilize the robot; a TRPO constraint provably bounds the behavioral shift.
- **Constrained MDP Variants (CPO - Achiam et al., 2017):** Extends TRPO's conjugate gradient trust region to simultaneously enforce safety cost constraints (e.g., joint torque limits, obstacle proximity), enabling certified-safe policy learning for medical robotics and autonomous vehicles.

### 3. RLHF Alignment: The KL-Constrained Optimal Policy
TRPO's mathematical framework provides the closed-form solution underlying all KL-regularized LLM alignment:
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right)$$
This is exactly the TRPO-optimal policy under an infinite-sample KL constraint. The partition function $Z(x)$ is intractable in general, but **DPO** (Rafailov et al., 2023) cancels it algebraically, obtaining a closed-form alignment loss directly from preference data — inheriting TRPO's theoretical guarantees without ever running the trust-region optimizer.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of Part 5 Conjugate Gradient hand calculations:
   - Iteration 0: $\alpha_0 = 0.45455, \mathbf{x}_1 = [0.9091, 0.4545]^\top$
   - Iteration 1: $\mathbf{x}_2 = [0.85714, 0.57143]^\top$ matching analytical inverse $[6/7, 4/7]^\top$ to $< 10^{-14}$.
   - Trust region scaling $\beta = 0.20917$ and final step $\Delta \boldsymbol{\theta}^* = [0.17929, 0.11952]^\top$.
2. Standalone PyTorch implementation of Hessian-Vector Products and Conjugate Gradient.
3. Complete verification of Section 6 Solved Illustrations:
   - Illustration 2: 2-iteration CG solve on $2 \times 2$ Fisher matrix converging to $[0, 1]^\top$.
   - Illustration 3: Exact TRPO step $\beta = 0.1000$ and backtracking line search acceptance at $j=1$.
   - Illustration 4: Monotonic improvement bound $J(\tilde{\pi}) = 2.00625 \ge 1.75507$ on 3-state MDP.
   - Illustration 5: Pearlmutter FVP trick matching analytical matrix-vector product $[0.220728, -1.600000]^\top$.
4. Complete TRPO agent tested on continuous control inverted pendulum dynamics.

See implementation in:
[`11_reinforcement_learning/code/18_trust_region_policy_optimization.py`](./code/18_trust_region_policy_optimization.py)
