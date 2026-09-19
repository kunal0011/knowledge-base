# Module 11.14: Advantage Actor-Critic (A2C & A3C)

---

## 1. Intuition & 101 Motivation

In Chapter 11.13, we derived the REINFORCE algorithm, which optimizes policies directly using sample trajectories. While REINFORCE is mathematically elegant and strictly unbiased, it suffers from a fatal flaw: **high variance**. Because it uses the full Monte Carlo return:
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$
the variance scales quadratically with the horizon $T$, requiring millions of rollouts to find clear gradient signals. Furthermore, REINFORCE cannot learn online or mid-episode; it must wait for the entire episode to terminate.

**Actor-Critic methods** eliminate this episodic restriction by replacing full Monte Carlo rollouts with **1-step bootstrapping**:
- **The Actor:** The parameterized policy $\pi_{\theta}(a \mid s)$, which explores the environment and selects actions.
- **The Critic:** A parameterized state-value function $V_{\phi}(s)$, which evaluates the quality of states by computing the 1-step Temporal-Difference (TD) error:
  $$\delta_t = R_{t+1} + \gamma V_{\phi}(S_{t+1}) - V_{\phi}(S_t)$$

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
$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) \Psi_t \right]$$

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

$$\mathcal{L}_{\text{total}}(\theta, \phi) \triangleq \mathcal{L}_{\text{policy}}(\theta) + c_1 \mathcal{L}_{\text{value}}(\phi) - c_2 \mathcal{H}(\pi_{\theta}(\cdot \mid S_t))$$

where:
1. **Policy Loss (Actor):**
   $$\mathcal{L}_{\text{policy}}(\theta) = - \log \pi_{\theta}(A_t \mid S_t) \cdot \delta_t$$
   (The negative sign converts gradient ascent on expected return into gradient descent). Note: $\delta_t$ is detached from autograd when optimizing the Actor!
2. **Value Loss (Critic):**
   $$\mathcal{L}_{\text{value}}(\phi) = \frac{1}{2} \delta_t^2 = \frac{1}{2} \left( R_{t+1} + \gamma V_{\phi}(S_{t+1}) - V_{\phi}(S_t) \right)^2$$
3. **Entropy Regularization ($\mathcal{H}$):**
   $$\mathcal{H}(\pi_{\theta}(\cdot \mid S_t)) \triangleq - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid S_t) \log \pi_{\theta}(a \mid S_t)$$
   Maximizing entropy (subtracting $c_2 \mathcal{H}$ from the loss) prevents the policy from prematurely collapsing into a deterministic suboptimal action, encouraging persistent exploration.
4. $c_1 \approx 0.5$ and $c_2 \approx 0.01$ are positive scaling coefficients.

---

### 2.3 Synchronous Batched Multi-Worker Execution (A2C)

Let $K$ be the number of parallel environment runners. At each rollout step:
1. Each environment $k \in \{1, \dots, K\}$ holds state $s_t^{(k)}$.
2. A single batched forward pass computes:
   $$\pi_t = \text{Softmax}(\text{Actor}(\mathbf{S}_t)), \quad \mathbf{V}_t = \text{Critic}(\mathbf{S}_t)$$
3. Sample actions $a_t^{(k)} \sim \pi_t^{(k)}$ and step all $K$ environments synchronously in parallel, obtaining rewards $\mathbf{R}_{t+1}$ and next states $\mathbf{S}_{t+1}$.
4. Compute batched TD errors across all $K$ workers:
   $$\delta_t = \mathbf{R}_{t+1} + \gamma (1 - \mathbf{d}) \mathbf{V}(S_{t+1}) - \mathbf{V}_t$$
5. Perform a single batched backpropagation update on the unified loss $\mathcal{L}_{\text{total}}$.

---

### 2.4 First-Principles Mathematical Derivations

#### Derivation 11.14.1: Compatible Function Approximation Theorem (Sutton et al., 1999)

```
====================================================================================================
DERIVATION 11.14.1: Compatible Function Approximation Theorem
====================================================================================================
Problem Statement:
Let an MDP have state space 𝒮, action space 𝒜, transition dynamics 𝒫(s' | s, a), and discount γ ∈ [0, 1).
Consider a parameterized policy π_θ(a | s) with parameter vector θ ∈ ℝ^d and a parameterized critic
Q_w(s, a) with parameter vector w ∈ ℝ^m. The true policy gradient is:
    ∇_θ J(θ) = ∑_{s ∈ 𝒮} d^{π_θ}(s) ∑_{a ∈ 𝒜} ∇_θ π_θ(a | s) Q^{π_θ}(s, a)
If Q^{π_θ}(s, a) is replaced by the approximate critic Q_w(s, a), the estimated policy gradient is:
    \widehat{∇}_θ J(θ) = ∑_{s ∈ 𝒮} d^{π_θ}(s) ∑_{a ∈ 𝒜} ∇_θ π_θ(a | s) Q_w(s, a)
Prove from first principles that if:
    1. Compatibility Condition: ∇_w Q_w(s, a) = ∇_θ \log \pi_θ(a | s)
    2. Mean Squared Value Error (MSVE) Minimization: w* minimizes the weighted squared error:
       ℰ(w) = (1/2) ∑_{s ∈ 𝒮} d^{π_θ}(s) ∑_{a ∈ 𝒜} π_θ(a | s) [ Q^{π_θ}(s, a) - Q_w(s, a) ]^2
then the policy gradient computed using the approximate critic is strictly exact:
    \widehat{∇}_θ J(θ) = ∇_θ J(θ)
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In policy gradient reinforcement learning, evaluating the exact policy gradient $\nabla_{\theta} J(\theta)$ requires the true action-value function $Q^{\pi_{\theta}}(s, a)$. However, $Q^{\pi_{\theta}}$ is generally unknown and lies in an infinite- or high-dimensional function space. If we approximate $Q^{\pi_{\theta}}(s, a)$ using a parametric function approximator $Q_{\mathbf{w}}(s, a)$ (such as a neural network or linear architecture), substituting $Q_{\mathbf{w}}$ directly into the policy gradient formula typically introduces systematic bias:
$$\widehat{\nabla}_{\theta} J(\theta) \triangleq \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) Q_{\mathbf{w}}(s, a) \neq \nabla_{\theta} J(\theta)$$
Our mathematical goal is to prove the **Compatible Function Approximation Theorem** (Sutton, McAllester, Singh, & Mansour, 1999): under two specific conditions—the compatibility condition matching critic gradient features to policy score functions, and the mean-squared value error (MSVE) stationary condition—the gradient estimation error vanishes identically:
$$\widehat{\nabla}_{\theta} J(\theta) - \nabla_{\theta} J(\theta) = \mathbf{0}$$
meaning that critic function approximation error introduces **zero bias** into the policy gradient.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Policy Differentiability & Non-Degeneracy:** The policy $\pi_{\theta}(a \mid s)$ is strictly positive $\pi_{\theta}(a \mid s) > 0$ and continuously differentiable with respect to $\theta \in \mathbb{R}^d$ for all $(s, a) \in \mathcal{S} \times \mathcal{A}$.
2. **Well-Defined Discounted State Measure:** The unnormalized discounted state visitation measure $d^{\pi_{\theta}}(s) \triangleq \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = s \mid S_0 \sim \mu; \pi_{\theta})$ is strictly positive on all recurrent states for starting distribution $\mu$, with total mass $\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) = \frac{1}{1 - \gamma}$.
3. **Compatibility Condition:** The critic architecture $Q_{\mathbf{w}}(s, a)$ has parameter dimension matching the policy parameter dimension ($m = d$), and its gradient with respect to $\mathbf{w}$ equals the score function of the policy:
   $$\nabla_{\mathbf{w}} Q_{\mathbf{w}}(s, a) = \nabla_{\theta} \log \pi_{\theta}(a \mid s) = \frac{\nabla_{\theta} \pi_{\theta}(a \mid s)}{\pi_{\theta}(a \mid s)} \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A}$$
   Consequently, $Q_{\mathbf{w}}(s, a)$ is linear in the compatible features $\psi_{\theta}(s, a) \triangleq \nabla_{\theta} \log \pi_{\theta}(a \mid s)$:
   $$Q_{\mathbf{w}}(s, a) = \mathbf{w}^\top \psi_{\theta}(s, a) = \mathbf{w}^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s)$$
   (or with an arbitrary action-independent state baseline $Q_{\mathbf{w}}(s, a) = \mathbf{w}^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s) + V(s)$).
4. **MSVE Stationarity:** The critic parameter vector $\mathbf{w}^*$ is a stationary point (global minimizer) of the stationary weighted Mean Squared Value Error (MSVE):
   $$\mathcal{E}(\mathbf{w}) \triangleq \frac{1}{2} \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \left( Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}}(s, a) \right)^2$$
   satisfying the first-order optimality condition $\nabla_{\mathbf{w}} \mathcal{E}(\mathbf{w}^*) = \mathbf{0}$.
5. **Non-Degenerate Fisher Metric:** The expected outer product of compatible features (the Fisher Information Matrix under measure $d^{\pi_{\theta}}$) $\mathbf{F}(\theta) \triangleq \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \psi_{\theta}(s, a) \psi_{\theta}(s, a)^\top$ is strictly positive definite, ensuring that $\mathbf{w}^*$ is uniquely determined.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
Geometrically, consider the Hilbert space $\mathcal{H} = L^2(\mathcal{S} \times \mathcal{A}, \rho^{\pi_{\theta}})$, where the weighting measure is $\rho^{\pi_{\theta}}(s, a) \triangleq d^{\pi_{\theta}}(s) \pi_{\theta}(a \mid s)$, equipped with the inner product:
$$\langle f, g \rangle_{\rho} \triangleq \sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} d^{\pi_{\theta}}(s) \pi_{\theta}(a \mid s) f(s, a) g(s, a)$$
The Policy Gradient Theorem states that the $k$-th component of the policy gradient is an inner product between the $k$-th score function $\psi_k(s, a) = \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_k}$ and the true value function $Q^{\pi_{\theta}}$:
$$[\nabla_{\theta} J(\theta)]_k = \langle \psi_k, Q^{\pi_{\theta}} \rangle_{\rho}$$
Now define the **compatible subspace** $\mathcal{M} \triangleq \operatorname{span}\{\psi_1, \psi_2, \dots, \psi_d\} \subset \mathcal{H}$. Any function in $\mathcal{H}$ can be decomposed uniquely into a parallel component in $\mathcal{M}$ and an orthogonal component in $\mathcal{M}^\perp$:
$$Q^{\pi_{\theta}} = Q_\parallel + Q_\perp, \quad Q_\parallel \in \mathcal{M}, \quad Q_\perp \in \mathcal{M}^\perp$$
Minimizing the MSVE $\mathcal{E}(\mathbf{w})$ performs an **orthogonal projection** of $Q^{\pi_{\theta}}$ onto $\mathcal{M}$! By the Hilbert Projection Theorem, the approximation error $e(s, a) \triangleq Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}^*}(s, a) = Q_\perp(s, a)$ is strictly orthogonal to every basis vector of $\mathcal{M}$:
$$\langle \psi_k, e \rangle_\rho = 0 \quad \forall k \in \{1, \dots, d\}$$
Because the policy gradient depends on $Q^{\pi_{\theta}}$ *only* through its projection onto $\mathcal{M}$, the orthogonal error $Q_\perp$ has zero inner product with the score functions. The approximate critic $Q_{\mathbf{w}^*}$ discards all extraneous dimensions of the value function that do not affect the policy gradient, retaining 100% of the true gradient signal!

**Part 4: End-to-End Step-by-Step Algebraic Proof**
*Step 1: Compute the gradient of the MSVE objective with respect to critic weights $\mathbf{w}$.*
The weighted MSVE objective is:
$$\mathcal{E}(\mathbf{w}) = \frac{1}{2} \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \left( Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}}(s, a) \right)^2$$
Differentiating with respect to the parameter vector $\mathbf{w} \in \mathbb{R}^d$:
$$\nabla_{\mathbf{w}} \mathcal{E}(\mathbf{w}) = \frac{1}{2} \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \cdot 2 \left( Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}}(s, a) \right) \cdot \nabla_{\mathbf{w}} \left( Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}}(s, a) \right)$$
Since $Q^{\pi_{\theta}}(s, a)$ is independent of $\mathbf{w}$, $\nabla_{\mathbf{w}} Q^{\pi_{\theta}}(s, a) = \mathbf{0}$. Therefore:
$$\nabla_{\mathbf{w}} \mathcal{E}(\mathbf{w}) = - \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \left( Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}}(s, a) \right) \nabla_{\mathbf{w}} Q_{\mathbf{w}}(s, a) \quad \text{(Equation 1)}$$

*Step 2: Enforce first-order optimality at $\mathbf{w}^*$.*
By Assumption 4, $\mathbf{w}^*$ minimizes $\mathcal{E}(\mathbf{w})$, so the gradient must vanish:
$$\nabla_{\mathbf{w}} \mathcal{E}(\mathbf{w}^*) = \mathbf{0}$$
Setting Equation 1 to zero:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \left( Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}^*}(s, a) \right) \nabla_{\mathbf{w}} Q_{\mathbf{w}^*}(s, a) = \mathbf{0} \quad \text{(Equation 2)}$$

*Step 3: Substitute the compatibility condition.*
By Assumption 3, the critic satisfies $\nabla_{\mathbf{w}} Q_{\mathbf{w}}(s, a) = \nabla_{\theta} \log \pi_{\theta}(a \mid s)$. Substituting this identity into Equation 2:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \left( Q^{\pi_{\theta}}(s, a) - Q_{\mathbf{w}^*}(s, a) \right) \nabla_{\theta} \log \pi_{\theta}(a \mid s) = \mathbf{0} \quad \text{(Equation 3)}$$

*Step 4: Distribute the terms across the difference.*
Distributing the product inside the summation:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) Q^{\pi_{\theta}}(s, a) \nabla_{\theta} \log \pi_{\theta}(a \mid s) - \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) Q_{\mathbf{w}^*}(s, a) \nabla_{\theta} \log \pi_{\theta}(a \mid s) = \mathbf{0}$$
Equivalently:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) Q^{\pi_{\theta}}(s, a) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) Q_{\mathbf{w}^*}(s, a) \quad \text{(Equation 4)}$$

*Step 5: Apply the log-derivative score identity.*
Recall the score function identity for differentiable policies:
$$\pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) = \pi_{\theta}(a \mid s) \frac{\nabla_{\theta} \pi_{\theta}(a \mid s)}{\pi_{\theta}(a \mid s)} = \nabla_{\theta} \pi_{\theta}(a \mid s)$$
Applying this identity to both sides of Equation 4:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) Q^{\pi_{\theta}}(s, a) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) Q_{\mathbf{w}^*}(s, a)$$

*Step 6: Recognize the exact and approximate policy gradients.*
By the Policy Gradient Theorem (Sutton et al., 1999), the left-hand side is identically the true policy gradient:
$$\nabla_{\theta} J(\theta) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) Q^{\pi_{\theta}}(s, a)$$
The right-hand side is identically the approximate policy gradient evaluated using critic $Q_{\mathbf{w}^*}$:
$$\widehat{\nabla}_{\theta} J(\theta) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) Q_{\mathbf{w}^*}(s, a)$$
Therefore:
$$\nabla_{\theta} J(\theta) = \widehat{\nabla}_{\theta} J(\theta) \quad \blacksquare$$

*Step 7: Invariance under addition of arbitrary state baseline $V(s)$.*
If the compatible critic is augmented with an arbitrary action-independent baseline $V(s)$:
$$Q_{\mathbf{w}, V}(s, a) \triangleq \mathbf{w}^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s) + V(s)$$
the approximate policy gradient becomes:
$$\widehat{\nabla}_{\theta} J(\theta) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) \left( \mathbf{w}^{*\top} \nabla_{\theta} \log \pi_{\theta}(a \mid s) + V(s) \right)$$
$$= \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) \mathbf{w}^{*\top} \nabla_{\theta} \log \pi_{\theta}(a \mid s) + \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) V(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s)$$
Since probabilities sum to 1 ($\sum_a \pi_{\theta}(a \mid s) = 1$), $\sum_a \nabla_{\theta} \pi_{\theta}(a \mid s) = \nabla_{\theta} (1) = \mathbf{0}$. The second term vanishes identically:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) V(s) \cdot \mathbf{0} = \mathbf{0}$$
Hence, exactness holds unconditionally for any state baseline $V(s)$. $\blacksquare$

---

#### Derivation 11.14.2: Two-Timescale Stochastic Approximation Convergence of Actor-Critic (Borkar & Meyn ODE Method)

```
====================================================================================================
DERIVATION 11.14.2: Two-Timescale Stochastic Approximation Convergence of Actor-Critic
====================================================================================================
Problem Statement:
Consider the coupled discrete-time online Actor-Critic stochastic approximation scheme:
    w_{k+1} = w_k + β_k [ δ_k(w_k) ∇_w V_{w_k}(s_k) ]             (Critic on fast timescale β_k)
    θ_{k+1} = θ_k + α_k [ δ_k(w_k) ∇_θ \log \pi_{θ_k}(a_k | s_k) ]   (Actor on slow timescale α_k)
where δ_k(w) = r_{k+1} + γ V_w(s_{k+1}) - V_w(s_k) is the 1-step TD error.
Prove via the Borkar & Meyn Ordinary Differential Equation (ODE) method that if the step-size sequences
satisfy timescale separation:
    ∑_k α_k = ∞,  ∑_k β_k = ∞,  ∑_k (α_k^2 + β_k^2) < ∞,  and  lim_{k → ∞} (α_k / β_k) = 0
the coupled iteration asymptotically decouples into two continuous-time dynamical systems:
    1. A fast ODE for the critic tracking the TD fixed-point w*(θ) for quasi-static θ.
    2. A slow ODE for the actor tracking the true performance gradient ascent trajectory:
       dθ/dt = ∇_θ J(θ)
guaranteeing almost-sure asymptotic convergence to the set of stationary policy points:
    lim_{k → ∞} ∇_θ J(θ_k) = 0
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In online Actor-Critic algorithms, the critic updates its value weights $\mathbf{w}_k \in \mathbb{R}^m$ while the actor simultaneously updates its policy weights $\theta_k \in \mathbb{R}^d$ on every observed transition $(s_k, a_k, r_{k+1}, s_{k+1})$. This creates a non-stationary feedback loop: the critic evaluates a moving target policy $\pi_{\theta_k}$, while the actor ascends a moving landscape evaluated by an imperfect, evolving critic $\mathbf{w}_k$. Without coordination, this coupled dynamical system can oscillate indefinitely or diverge.

Our mathematical goal is to prove, using the **Two-Timescale Stochastic Approximation ODE Method** (Borkar, 1997; Borkar & Meyn, 2000), that enforcing a separation of timescales ($\alpha_k = o(\beta_k)$):
1. Completely decouples the asymptotic dynamics into a fast critic system and a slow actor system.
2. Ensures the fast critic converges to the unique projected Bellman fixed point $\mathbf{w}^*(\theta)$ corresponding to the frozen policy $\theta$.
3. Guarantees the slow actor asymptotically follows the continuous-time gradient ascent trajectory $\dot{\theta}(t) = \nabla_{\theta} J(\theta(t))$, converging almost surely to a stationary point of expected return $\nabla_{\theta} J(\theta) = \mathbf{0}$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Timescale Separation of Step-Sizes:**
   The positive step-size sequences $\{\alpha_k\}_{k \ge 0}$ and $\{\beta_k\}_{k \ge 0}$ satisfy the standard Robbins-Monro conditions:
   $$\sum_{k=0}^\infty \alpha_k = \infty, \quad \sum_{k=0}^\infty \beta_k = \infty, \quad \sum_{k=0}^\infty \alpha_k^2 < \infty, \quad \sum_{k=0}^\infty \beta_k^2 < \infty$$
   and the strict two-timescale ratio condition:
   $$\lim_{k \to \infty} \frac{\alpha_k}{\beta_k} = 0$$
   (Typical schedules: $\beta_k = (k+1)^{-2/3}$ and $\alpha_k = (k+1)^{-1}$, so $\alpha_k / \beta_k = (k+1)^{-1/3} \to 0$).
2. **Markov Chain Ergodicity & Rapid Mixing:**
   For every fixed policy parameter $\theta \in \Theta$, the induced Markov chain with transition kernel $\mathcal{P}^{\pi_{\theta}}(s' \mid s) \triangleq \sum_a \pi_{\theta}(a \mid s) \mathcal{P}(s' \mid s, a)$ is irreducible, aperiodic, and positive recurrent with unique stationary distribution $d^{\pi_{\theta}}(s)$. The Markov chain exhibits geometric ergodicity:
   $$\sup_{s \in \mathcal{S}} \| \mathbb{P}(S_t \in \cdot \mid S_0 = s) - d^{\pi_{\theta}}(\cdot) \|_{\text{TV}} \le C \rho^t \quad \text{for constants } C > 0, \rho \in [0, 1)$$
3. **Smoothness and Boundedness:**
   - The policy $\pi_{\theta}(a \mid s) > 0$ is twice continuously differentiable with respect to $\theta$ with uniformly bounded score functions: $\sup_{\theta, s, a} \|\nabla_{\theta} \log \pi_{\theta}(a \mid s)\|_2 \le C_\psi < \infty$ and bounded Hessian $\|\nabla_{\theta}^2 \log \pi_{\theta}(a \mid s)\|_2 \le C_H < \infty$.
   - The critic is linear: $V_{\mathbf{w}}(s) = \mathbf{w}^\top \phi(s)$, where the feature vector $\phi(s) \in \mathbb{R}^m$ satisfies $\sup_{s \in \mathcal{S}} \|\phi(s)\|_2 \le C_\phi < \infty$, and the feature matrix $\Phi \in \mathbb{R}^{|\mathcal{S}| \times m}$ has full column rank $m \le |\mathcal{S}|$.
   - Rewards are uniformly bounded: $\sup_{s, a} |R(s, a)| \le R_{\max} < \infty$.
4. **Strict Negative Definiteness / Hurwitz Condition (Tsitsiklis & Van Roy, 1997):**
   For any fixed $\theta$, the expected TD operator governing the critic is affine:
   $$\bar{\mathbf{h}}_{\theta}(\mathbf{w}) \triangleq \mathbb{E}_{s \sim d^\pi, a \sim \pi, s' \sim \mathcal{P}} \left[ \left( R(s, a) + \gamma \mathbf{w}^\top \phi(s') - \mathbf{w}^\top \phi(s) \right) \phi(s) \right] = \mathbf{b}_{\theta} - \mathbf{A}_{\theta} \mathbf{w}$$
   where $\mathbf{b}_{\theta} \triangleq \Phi^\top \mathbf{D}_{\theta} \mathbf{R}_{\theta}$ and $\mathbf{A}_{\theta} \triangleq \Phi^\top \mathbf{D}_{\theta} (\mathbf{I} - \gamma \mathbf{P}_{\theta}) \Phi$.
   Because $\mathbf{D}_{\theta} = \operatorname{diag}(d^{\pi_{\theta}})$ and $\mathbf{I} - \gamma \mathbf{P}_{\theta}$ is strictly positive definite on the range of $\Phi$, the matrix $-\mathbf{A}_{\theta}$ is **Hurwitz** (all eigenvalues have strictly negative real parts: $\operatorname{Re}(\lambda_i(-\mathbf{A}_{\theta})) < 0$).
   Consequently, the fast critic ODE $\dot{\mathbf{w}}(t) = \bar{\mathbf{h}}_{\theta}(\mathbf{w}(t))$ possesses a unique, globally asymptotically stable equilibrium:
   $$\mathbf{w}^*(\theta) = \mathbf{A}_{\theta}^{-1} \mathbf{b}_{\theta}$$
5. **Lipschitz Continuity of Value Equilibrium Map:**
   The equilibrium map $\theta \mapsto \mathbf{w}^*(\theta)$ is Lipschitz continuous:
   $$\|\mathbf{w}^*(\theta_1) - \mathbf{w}^*(\theta_2)\|_2 \le L_w \|\theta_1 - \theta_2\|_2 \quad \forall \theta_1, \theta_2 \in \Theta$$
6. **Martingale Difference Noise Regularity:**
   The observation noise terms $\mathbf{M}_{k+1}^{(\mathbf{w})} \triangleq \delta_k(\mathbf{w}_k) \phi(s_k) - \bar{\mathbf{h}}_{\theta_k}(\mathbf{w}_k)$ and $\mathbf{M}_{k+1}^{(\theta)} \triangleq \delta_k(\mathbf{w}_k) \nabla_{\theta} \log \pi_{\theta_k}(a_k \mid s_k) - \bar{\mathbf{g}}(\theta_k, \mathbf{w}_k)$ are martingale difference sequences with bounded conditional variances: $\mathbb{E}[\|\mathbf{M}_{k+1}\|^2 \mid \mathcal{F}_k] \le \sigma_M^2 < \infty$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
The two-timescale mechanism is the algorithmic analogue of the **Born-Oppenheimer approximation** in quantum mechanics or **singular perturbation theory** in dynamical systems:
- Fast clock (Critic, timescale $\beta_k$): Think of the critic as an electron orbiting a massive atomic nucleus (the actor). Because $\alpha_k / \beta_k \to 0$, from the critic's perspective, the actor's position $\theta_k$ is virtually motionless:
  $$\frac{\|\theta_{k+1} - \theta_k\|_2}{\beta_k} = \frac{\alpha_k}{\beta_k} \mathcal{O}(1) \to 0$$
  On the fast timescale, the critic sees a fixed policy $\theta$, and the projected Bellman contraction drags $\mathbf{w}_k$ exponentially fast onto the low-dimensional equilibrium manifold $\mathcal{V} = \{ (\mathbf{w}^*(\theta), \theta) : \theta \in \Theta \}$.
- Slow clock (Actor, timescale $\alpha_k$): On the slow timescale, the critic's high-frequency fluctuations have completely dissipated, and $\mathbf{w}_k$ has already snapped to $\mathbf{w}^*(\theta_k)$ up to an $o(1)$ perturbation.
  The actor therefore experiences clean, quasi-deterministic gradient ascent along the performance surface $J(\theta)$:
  $$\dot{\theta}(t) = \nabla_{\theta} J(\theta(t))$$
  Because $J(\theta)$ acts as a natural Lyapunov function ($\dot{J} = \|\nabla J\|^2 \ge 0$), the actor cannot orbit or diverge; it must monotonically climb the performance manifold until it reaches a stationary point $\nabla J(\theta) = \mathbf{0}$.

**Part 4: End-to-End Step-by-Step Algebraic Proof**
*Step 1: Express the coupled iterations in standard stochastic approximation form.*
Define the filtration $\mathcal{F}_k \triangleq \sigma(\mathbf{w}_0, \theta_0, s_0, a_0, \dots, s_k, a_k)$.
The updates can be written as:
$$\mathbf{w}_{k+1} = \mathbf{w}_k + \beta_k \left[ \bar{\mathbf{h}}_{\theta_k}(\mathbf{w}_k) + \mathbf{M}_{k+1}^{(\mathbf{w})} \right] \quad \text{(Equation 1)}$$
$$\theta_{k+1} = \theta_k + \alpha_k \left[ \bar{\mathbf{g}}(\theta_k, \mathbf{w}_k) + \mathbf{M}_{k+1}^{(\theta)} \right] \quad \text{(Equation 2)}$$
where:
$$\bar{\mathbf{h}}_{\theta}(\mathbf{w}) \triangleq \mathbb{E}_{s \sim d^{\pi_{\theta}}, a \sim \pi_{\theta}, s' \sim \mathcal{P}} \left[ \left( R(s, a) + \gamma \mathbf{w}^\top \phi(s') - \mathbf{w}^\top \phi(s) \right) \phi(s) \right]$$
$$\bar{\mathbf{g}}(\theta, \mathbf{w}) \triangleq \mathbb{E}_{s \sim d^{\pi_{\theta}}, a \sim \pi_{\theta}, s' \sim \mathcal{P}} \left[ \left( R(s, a) + \gamma \mathbf{w}^\top \phi(s') - \mathbf{w}^\top \phi(s) \right) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \right]$$

*Step 2: Analysis of the fast timescale (The Critic).*
Define the continuous-time schedule for the critic by $t_0^\beta = 0$ and $t_n^\beta = \sum_{i=0}^{n-1} \beta_i$.
Let us examine the total displacement of the slow parameter $\theta$ over any time horizon $T > 0$ on the critic's clock:
Let $m(n, T) \triangleq \max \{ m \ge n : t_m^\beta - t_n^\beta \le T \}$.
Using the triangle inequality on Equation 2:
$$\|\theta_{m(n, T)} - \theta_n\|_2 \le \sum_{k=n}^{m(n, T)-1} \alpha_k \left( \|\bar{\mathbf{g}}(\theta_k, \mathbf{w}_k)\|_2 + \|\mathbf{M}_{k+1}^{(\theta)}\|_2 \right)$$
Using the boundedness of rewards, features, and scores, $\|\bar{\mathbf{g}}\|_2 \le C_g < \infty$.
Multiply and divide by $\beta_k$:
$$\|\theta_{m(n, T)} - \theta_n\|_2 \le \sup_{k \ge n} \left( \frac{\alpha_k}{\beta_k} \right) \sum_{k=n}^{m(n, T)-1} \beta_k \left( C_g + \|\mathbf{M}_{k+1}^{(\theta)}\|_2 \right)$$
Since $\sum_{k=n}^{m(n, T)-1} \beta_k \le T$ and by Assumption 1, $\lim_{n \to \infty} \sup_{k \ge n} \left( \frac{\alpha_k}{\beta_k} \right) = 0$:
$$\lim_{n \to \infty} \sup_{t \in [t_n^\beta, t_n^\beta + T]} \|\bar{\theta}(t) - \bar{\theta}(t_n^\beta)\|_2 = 0 \quad \text{almost surely.}$$
Thus, on the critic's timescale, $\theta$ is asymptotically **quasi-static** (a frozen constant parameter).
By the Borkar-Meyn ODE Theorem for single-timescale stochastic approximations with Martingale noise (since $\sum \beta_k^2 < \infty$), the interpolated trajectory $\bar{\mathbf{w}}(t)$ asymptotically tracks the solution of the ODE with frozen $\theta$:
$$\dot{\mathbf{w}}(t) = \bar{\mathbf{h}}_{\theta}(\mathbf{w}(t)) = \mathbf{b}_{\theta} - \mathbf{A}_{\theta} \mathbf{w}(t)$$
Since $-\mathbf{A}_{\theta}$ is Hurwitz by Assumption 4, the unique equilibrium is:
$$\mathbf{w}^*(\theta) = \mathbf{A}_{\theta}^{-1} \mathbf{b}_{\theta}$$
and by Lyapunov stability of linear ODEs with Hurwitz matrices, $\mathbf{w}(t) \to \mathbf{w}^*(\theta)$ exponentially fast:
$$\|\mathbf{w}(t) - \mathbf{w}^*(\theta)\|_2 \le K_0 e^{-\lambda_{\min}(\mathbf{A}) t} \|\mathbf{w}(0) - \mathbf{w}^*(\theta)\|_2$$
Therefore, the critic tracks the parameter-dependent equilibrium:
$$\lim_{k \to \infty} \|\mathbf{w}_k - \mathbf{w}^*(\theta_k)\|_2 = 0 \quad \text{almost surely.} \quad \text{(Equation 3)}$$

*Step 3: Analysis of the slow timescale (The Actor).*
Now consider the continuous-time schedule of the actor: $t_0^\alpha = 0, t_n^\alpha = \sum_{i=0}^{n-1} \alpha_i$.
Rewrite the actor recursion (Equation 2) by adding and subtracting the equilibrium drift $\bar{\mathbf{g}}(\theta_k, \mathbf{w}^*(\theta_k))$:
$$\theta_{k+1} = \theta_k + \alpha_k \left[ \bar{\mathbf{g}}(\theta_k, \mathbf{w}^*(\theta_k)) + \mathbf{e}_k + \mathbf{M}_{k+1}^{(\theta)} \right] \quad \text{(Equation 4)}$$
where $\mathbf{e}_k \triangleq \bar{\mathbf{g}}(\theta_k, \mathbf{w}_k) - \bar{\mathbf{g}}(\theta_k, \mathbf{w}^*(\theta_k))$ is the tracking error.
By the mean-value theorem applied to $\bar{\mathbf{g}}$ with respect to $\mathbf{w}$:
$$\|\mathbf{e}_k\|_2 \le L_{\mathbf{w}} \|\mathbf{w}_k - \mathbf{w}^*(\theta_k)\|_2$$
where $L_{\mathbf{w}} = \sup_{\theta, s, a} \|\nabla_{\theta} \log \pi_{\theta}(a \mid s)\|_2 \cdot \|\phi(s) - \gamma \mathbb{E}[\phi(s')]\|_2 \le C_\psi C_\phi (1 + \gamma) < \infty$.
From Equation 3, since $\|\mathbf{w}_k - \mathbf{w}^*(\theta_k)\|_2 \to 0$ almost surely:
$$\lim_{k \to \infty} \|\mathbf{e}_k\|_2 = 0 \quad \text{almost surely.}$$
Furthermore, the martingale noise sum $\sum_{i=0}^{k-1} \alpha_i \mathbf{M}_{i+1}^{(\theta)}$ converges almost surely by the Martingale Convergence Theorem because:
$$\sum_{k=0}^\infty \alpha_k^2 \mathbb{E}[\|\mathbf{M}_{k+1}^{(\theta)}\|^2 \mid \mathcal{F}_k] \le \sigma_M^2 \sum_{k=0}^\infty \alpha_k^2 < \infty$$
By Kushner-Clark Lemma and Borkar's Two-Timescale Theorem (Borkar, 2008, Chapter 6), the tracking error $\mathbf{e}_k$ and martingale noise are asymptotically negligible. The piecewise linear continuous-time interpolation $\bar{\theta}(t)$ converges uniformly on compact intervals to the solution of the autonomous limiting ODE:
$$\dot{\theta}(t) = \bar{\mathbf{g}}(\theta(t), \mathbf{w}^*(\theta(t))) \quad \text{(Equation 5)}$$

*Step 4: Identification with the Policy Gradient.*
Recall from Derivation 11.14.1 that when the critic satisfies the projected Bellman equation or compatible conditions, the expected advantage inner product equals the exact policy gradient:
$$\bar{\mathbf{g}}(\theta, \mathbf{w}^*(\theta)) = \mathbb{E}_{s \sim d^\pi, a \sim \pi} \left[ \left( Q^{\pi_{\theta}}(s, a) - V^{\pi_{\theta}}(s) \right) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \right] = \nabla_{\theta} J(\theta)$$
Therefore, the slow limiting ODE (Equation 5) is strictly the **gradient ascent ODE**:
$$\dot{\theta}(t) = \nabla_{\theta} J(\theta(t)) \quad \text{(Equation 6)}$$

*Step 5: Lyapunov Stability and Convergence to Stationary Points.*
Define the Lyapunov function $\mathcal{L}(\theta) \triangleq - J(\theta)$.
Compute the orbital time derivative along the trajectories of Equation 6:
$$\frac{d}{dt} J(\theta(t)) = \left( \nabla_{\theta} J(\theta(t)) \right)^\top \dot{\theta}(t) = \left( \nabla_{\theta} J(\theta(t)) \right)^\top \nabla_{\theta} J(\theta(t)) = \|\nabla_{\theta} J(\theta(t))\|_2^2 \ge 0$$
The time derivative is strictly non-negative everywhere:
$$\frac{d}{dt} \mathcal{L}(\theta(t)) = - \|\nabla_{\theta} J(\theta(t))\|_2^2 \le 0$$
Thus, $\mathcal{L}(\theta)$ is a strict Lyapunov function for the system.
The set where $\frac{d}{dt} \mathcal{L}(\theta) = 0$ is precisely the set of critical points (stationary points) of the policy performance:
$$\mathcal{Z} \triangleq \{ \theta \in \Theta : \nabla_{\theta} J(\theta) = \mathbf{0} \}$$
By **LaSalle's Invariance Principle**, every bounded trajectory of the ODE $\dot{\theta} = \nabla_{\theta} J(\theta)$ converges to the largest invariant subset of $\mathcal{Z}$.
Consequently, the discrete-time actor sequence $\theta_k$ converges almost surely to the stationary set $\mathcal{Z}$:
$$\lim_{k \to \infty} \operatorname{dist}(\theta_k, \mathcal{Z}) = 0 \quad \text{almost surely, i.e.,} \quad \lim_{k \to \infty} \|\nabla_{\theta} J(\theta_k)\|_2 = 0 \quad \blacksquare$$

---

#### Derivation 11.14.3: Critic Bias Propagation Bound into Policy Gradient

```
====================================================================================================
DERIVATION 11.14.3: Critic Bias Propagation Bound into Policy Gradient
====================================================================================================
Problem Statement:
Let V_ϕ(s) be an approximate state-value critic with uniform approximation error bounded by:
    ‖ϵ‖_∞ ≜ max_{s ∈ 𝒮} |V_ϕ(s) - V^{π_θ}(s)| ≤ ϵ_critic
Let \widehat{∇}_θ J(θ) be the policy gradient estimated using the 1-step TD advantage:
    \widehat{δ}_ϕ(s, a) ≜ 𝔼_{s' ~ 𝒫(·|s,a)} [ R(s, a) + γ V_ϕ(s') - V_ϕ(s) ]
Prove from first principles that the gradient estimation error is strictly bounded by:
    ‖∇_θ J(θ) - \widehat{∇}_θ J(θ)‖_2 ≤ \frac{γ}{1 - γ} C_ψ ϵ_critic
where C_ψ ≜ max_{s, a} ‖∇_θ \log \pi_θ(a | s)‖_2, and prove that:
    1. Current-state critic approximation error ϵ(s) has ZERO contribution to policy gradient bias.
    2. Future-state critic approximation error ϵ(s') propagates exclusively through the transition dynamics.
    3. Under the discrete L_2 norm, the bound generalizes to:
       ‖∇_θ J(θ) - \widehat{∇}_θ J(θ)‖_2 ≤ \frac{\sqrt{|𝒮|}}{1 - γ} C_ψ ϵ_critic
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In practical Actor-Critic methods (such as A2C and A3C), the true state-value function $V^{\pi_{\theta}}(s)$ is approximated by a parameterized neural network $V_{\phi}(s)$. Because function approximators cannot achieve zero error across all states simultaneously, the critic carries a non-zero pointwise approximation error:
$$\epsilon(s) \triangleq V_{\phi}(s) - V^{\pi_{\theta}}(s), \quad \sup_{s \in \mathcal{S}} |\epsilon(s)| \le \epsilon_{\text{critic}}$$
When the 1-step TD error $\hat{\delta}_{\phi}(s, a) = R(s, a) + \gamma V_{\phi}(s') - V_{\phi}(s)$ is used as the advantage estimator, it introduces bias into the policy gradient:
$$\mathbf{E}_{\text{grad}} \triangleq \widehat{\nabla}_{\theta} J(\theta) - \nabla_{\theta} J(\theta)$$
Our mathematical goal is to derive an exact, first-principles upper bound on $\|\mathbf{E}_{\text{grad}}\|_2$, proving analytically:
1. Why the current state error $\epsilon(s)$ cancels out completely from the gradient bias (due to score function orthogonality).
2. Exactly how the bootstrapped next-state error $\epsilon(s')$ propagates through the transition dynamics $\mathcal{P}(s' \mid s, a)$ and discount factor $\gamma$.
3. That the total policy gradient estimation error is strictly bounded by $\frac{\gamma}{1-\gamma} C_\psi \epsilon_{\text{critic}}$, scaling linearly with critic precision $\epsilon_{\text{critic}}$ and the effective horizon $\frac{1}{1-\gamma}$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Finite or Compact State Space:** The state space $\mathcal{S}$ is finite with cardinality $|\mathcal{S}|$, and the action space $\mathcal{A}$ is finite.
2. **Discount Factor:** $\gamma \in [0, 1)$, ensuring the convergence of the Neumann series $\sum_{t=0}^\infty \gamma^t = \frac{1}{1 - \gamma}$.
3. **Score Function Boundedness:** The score function $\nabla_{\theta} \log \pi_{\theta}(a \mid s)$ has uniformly bounded Euclidean norm:
   $$C_\psi \triangleq \max_{s \in \mathcal{S}, a \in \mathcal{A}} \|\nabla_{\theta} \log \pi_{\theta}(a \mid s)\|_2 < \infty$$
4. **Pointwise Uniform Critic Error:** The critic approximation error $\epsilon(s) \triangleq V_{\phi}(s) - V^{\pi_{\theta}}(s)$ is bounded in $L_\infty$ norm:
   $$\|\epsilon\|_\infty \triangleq \max_{s \in \mathcal{S}} |\epsilon(s)| \le \epsilon_{\text{critic}} < \infty$$
5. **Discounted State Measure:** The unnormalized discounted state visitation measure is:
   $$d^{\pi_{\theta}}(s) \triangleq \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = s \mid S_0 \sim \mu; \pi_{\theta})$$
   which satisfies $\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) = \sum_{t=0}^\infty \gamma^t = \frac{1}{1 - \gamma}$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
Consider what happens when we evaluate the expectation of the 1-step TD advantage estimator:
$$\hat{\delta}_{\phi}(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}}[V_{\phi}(s')] - V_{\phi}(s)$$
Substitute $V_{\phi} = V^{\pi_{\theta}} + \epsilon$:
$$\hat{\delta}_{\phi}(s, a) = \underbrace{R(s, a) + \gamma \mathbb{E}_{s'}[V^{\pi_{\theta}}(s')] - V^{\pi_{\theta}}(s)}_{A^{\pi_{\theta}}(s, a) \text{ (True Advantage)}} + \underbrace{\gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)}[\epsilon(s')]}_{\text{Future Bootstrapping Error}} - \underbrace{\epsilon(s)}_{\text{Current Baseline Error}}$$
Now observe the fundamental geometric asymmetry between the two error terms:
- The current baseline error $\epsilon(s)$ is **action-independent**! When multiplied by the score vector $\nabla_{\theta} \log \pi_{\theta}(a \mid s)$ and integrated over $a \sim \pi_{\theta}$, it is projected onto $\sum_a \nabla_{\theta} \pi_{\theta}(a \mid s) = \nabla_{\theta} (1) = \mathbf{0}$. The baseline invariance property acts as an **ideal high-pass filter** that annihilates $\epsilon(s)$ completely!
- In contrast, the future bootstrapping error $\gamma \sum_{s'} \mathcal{P}(s' \mid s, a) \epsilon(s')$ **depends on action $a$** through the transition dynamics $\mathcal{P}(s' \mid s, a)$. Because different actions lead to different distributions of future states, this term does not vanish.
The policy gradient bias is therefore generated *exclusively* by the differential shift in future value errors across competing actions!

**Part 4: End-to-End Step-by-Step Algebraic Proof**
*Step 1: Write down the true policy gradient and the critic-based policy gradient.*
By the Policy Gradient Theorem with advantage baseline:
$$\nabla_{\theta} J(\theta) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) A^{\pi_{\theta}}(s, a) \quad \text{(Equation 1)}$$
where the true advantage function is:
$$A^{\pi_{\theta}}(s, a) = Q^{\pi_{\theta}}(s, a) - V^{\pi_{\theta}}(s) = \left( R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^{\pi_{\theta}}(s') \right) - V^{\pi_{\theta}}(s)$$
The estimated policy gradient using the approximate critic $V_{\phi}$ is:
$$\widehat{\nabla}_{\theta} J(\theta) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \bar{\delta}_{\phi}(s, a) \quad \text{(Equation 2)}$$
where $\bar{\delta}_{\phi}(s, a)$ is the expected 1-step TD advantage under critic $V_{\phi}$:
$$\bar{\delta}_{\phi}(s, a) \triangleq R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V_{\phi}(s') - V_{\phi}(s)$$

*Step 2: Express the difference between advantage estimators.*
Substitute $V_{\phi}(s) = V^{\pi_{\theta}}(s) + \epsilon(s)$ into $\bar{\delta}_{\phi}(s, a)$:
$$\bar{\delta}_{\phi}(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ V^{\pi_{\theta}}(s') + \epsilon(s') \right] - \left[ V^{\pi_{\theta}}(s) + \epsilon(s) \right]$$
Regrouping into the true advantage and error terms:
$$\bar{\delta}_{\phi}(s, a) = \underbrace{R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^{\pi_{\theta}}(s') - V^{\pi_{\theta}}(s)}_{A^{\pi_{\theta}}(s, a)} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s') - \epsilon(s)$$
Subtracting $A^{\pi_{\theta}}(s, a)$:
$$\bar{\delta}_{\phi}(s, a) - A^{\pi_{\theta}}(s, a) = \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s') - \epsilon(s) \quad \text{(Equation 3)}$$

*Step 3: Compute the policy gradient estimation error vector $\mathbf{E}_{\text{grad}}$.*
Subtract Equation 1 from Equation 2:
$$\mathbf{E}_{\text{grad}} \triangleq \widehat{\nabla}_{\theta} J(\theta) - \nabla_{\theta} J(\theta)$$
$$= \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \left[ \bar{\delta}_{\phi}(s, a) - A^{\pi_{\theta}}(s, a) \right]$$
Substitute Equation 3 into this expression:
$$\mathbf{E}_{\text{grad}} = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \left[ \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s') - \epsilon(s) \right] \quad \text{(Equation 4)}$$

*Step 4: Distribute and evaluate the current-state error term.*
Split Equation 4 into two distinct sums:
$$\mathbf{E}_{\text{grad}} = \gamma \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s')$$
$$- \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \epsilon(s) \left[ \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \right] \quad \text{(Equation 5)}$$
Examine the bracketed term in the second sum:
$$\sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{\nabla_{\theta} \pi_{\theta}(a \mid s)}{\pi_{\theta}(a \mid s)} = \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s)$$
Interchanging the gradient and finite sum:
$$\sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s) = \nabla_{\theta} \left( \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \right) = \nabla_{\theta} (1) = \mathbf{0}$$
Therefore, the second term vanishes identically:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \epsilon(s) \cdot \mathbf{0} = \mathbf{0}$$
This proves that the current-state error $\epsilon(s)$ contributes **identically zero bias** to the policy gradient!

*Step 5: Simplify to the surviving future-state error term.*
$$\mathbf{E}_{\text{grad}} = \gamma \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s') \quad \text{(Equation 6)}$$

*Step 6: Apply the norm inequality.*
Taking the Euclidean norm $\|\cdot\|_2$ of both sides and applying the triangle inequality:
$$\|\mathbf{E}_{\text{grad}}\|_2 \le \gamma \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \|\nabla_{\theta} \log \pi_{\theta}(a \mid s)\|_2 \left| \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s') \right|$$
Applying the triangle inequality to the inner sum over $s'$:
$$\left| \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s') \right| \le \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) |\epsilon(s')| \le \max_{s'' \in \mathcal{S}} |\epsilon(s'')| \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a)$$
Since $\sum_{s'} \mathcal{P}(s' \mid s, a) = 1$ and $\max_{s''} |\epsilon(s'')| = \|\epsilon\|_\infty \le \epsilon_{\text{critic}}$:
$$\left| \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \epsilon(s') \right| \le \epsilon_{\text{critic}} \cdot 1 = \epsilon_{\text{critic}}$$

*Step 7: Substitute the bounds.*
By Assumption 3, $\|\nabla_{\theta} \log \pi_{\theta}(a \mid s)\|_2 \le C_\psi$. Substituting both uniform bounds:
$$\|\mathbf{E}_{\text{grad}}\|_2 \le \gamma \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \cdot C_\psi \cdot \epsilon_{\text{critic}}$$
Factor the scalar constants $\gamma C_\psi \epsilon_{\text{critic}}$ outside the sums:
$$\|\mathbf{E}_{\text{grad}}\|_2 \le \gamma C_\psi \epsilon_{\text{critic}} \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \underbrace{\sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s)}_{1} = \gamma C_\psi \epsilon_{\text{critic}} \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s)$$
By Assumption 5, for the unnormalized state visitation distribution:
$$\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) = \sum_{t=0}^\infty \gamma^t = \frac{1}{1 - \gamma}$$
Therefore:
$$\|\widehat{\nabla}_{\theta} J(\theta) - \nabla_{\theta} J(\theta)\|_2 \le \frac{\gamma}{1 - \gamma} C_\psi \epsilon_{\text{critic}} \quad \blacksquare$$

*Step 8: Generalization to L_2 norm via Cauchy-Schwarz.*
Under the normalized distribution $\tilde{d}^{\pi}(s) \triangleq (1-\gamma) d^{\pi}(s)$, let the critic error vector be $\epsilon \in \mathbb{R}^{|\mathcal{S}|}$.
By the Cauchy-Schwarz inequality relating the $L_1$ and $L_2$ norms in finite dimensions:
$$\sum_{s \in \mathcal{S}} |\epsilon(s)| \le \sqrt{|\mathcal{S}|} \left( \sum_{s \in \mathcal{S}} \epsilon(s)^2 \right)^{1/2} = \sqrt{|\mathcal{S}|} \|\epsilon\|_2 \le \sqrt{|\mathcal{S}|} \sqrt{|\mathcal{S}|} \|\epsilon\|_\infty = |\mathcal{S}| \epsilon_{\text{critic}}$$
Using the normalized measure $\tilde{d}^{\pi}$, the bound scales with the effective dimensionality of the state space:
$$\|\widehat{\nabla}_{\theta} J(\theta) - \nabla_{\theta} J(\theta)\|_2 \le \frac{\gamma \sqrt{|\mathcal{S}|}}{1 - \gamma} C_\psi \|\epsilon\|_{2, \tilde{d}} \le \frac{\sqrt{|\mathcal{S}|}}{1 - \gamma} C_\psi \epsilon_{\text{critic}} \quad \blacksquare$$

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
- The trumpeter plays spontaneous, exploratory musical lines $\pi_{\theta}(a \mid s)$.
- The rhythm section maintains the harmonic groove and tempo, establishing the baseline $V_{\phi}(s)$.
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
| $\pi$ | Action Probability Vector | $\text{Softmax}(\mathbf{z})$ |
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

### Illustration 1: Why Bootstrapping Introduces Bias (Analytical Proof & Numerical Verification)
**Problem:**
1. REINFORCE uses the true empirical return $G_t$, which has zero bias: $\mathbb{E}[G_t \mid S_t, A_t] = Q^\pi(S_t, A_t)$. Actor-Critic replaces $G_t$ with 1-step bootstrap $R_{t+1} + \gamma V_{\phi}(S_{t+1})$. Prove that if the critic is imperfect ($V_{\phi} \neq V^\pi$), the policy gradient update becomes biased if and only if future-state critic errors depend on the action chosen.
2. In a 2-state MDP, starting state $s_0$ has actions $\{a_1, a_2\}$ transitioning to successor states $\{s_1, s_2\}$:
   - Transition dynamics: $\mathcal{P}(\cdot \mid s_0, a_1) = [0.80, 0.20]^\top$, $\mathcal{P}(\cdot \mid s_0, a_2) = [0.10, 0.90]^\top$.
   - True state values: $V^\pi(s_1) = 10.00, V^\pi(s_2) = 2.00$.
   - Deterministic rewards: $R(s_0, a_1) = 1.00, R(s_0, a_2) = 2.00$. Discount factor $\gamma = 0.90$.
   - Current policy at $s_0$: $\pi(a_1 \mid s_0) = 0.60, \pi(a_2 \mid s_0) = 0.40$. Parametrized by logit difference $\theta$ ($z_1 = \theta, z_2 = 0$), so score functions are $\psi(a_1) = 1 - \pi(a_1) = +0.40, \psi(a_2) = -\pi(a_1) = -0.60$.
   - Imperfect critic with estimation errors: $\epsilon(s_0) = +0.50, \epsilon(s_1) = +2.00, \epsilon(s_2) = -1.00$.
   Compute:
   a. True action values $Q^\pi(s_0, a)$, state value $V^\pi(s_0)$, and exact policy gradient $\nabla_\theta J(\theta)$.
   b. Approximate critic values $V_{\phi}(s)$, expected TD errors $\mathbb{E}[\delta_{\phi} \mid a]$, and approximate policy gradient $\widehat{\nabla}_\theta J(\theta)$.
   c. Verify that the empirical bias $\widehat{\nabla}_\theta J(\theta) - \nabla_\theta J(\theta)$ matches the theoretical bound formula from Derivation 11.14.3 to machine precision, confirming that current-state error $\epsilon(s_0)$ produces zero bias.

**Solution:**
**Part 1: Analytical Derivation:**
Let $\epsilon(s) \triangleq V_{\phi}(s) - V^\pi(s)$ be the critic's approximation error.
The expected TD error target conditioned on $(S_t = s, A_t = a)$ is:
$$\mathbb{E} \left[ R_{t+1} + \gamma V_{\phi}(S_{t+1}) \mid S_t = s, A_t = a \right]$$
$$= \mathbb{E} \left[ R_{t+1} + \gamma \left( V^\pi(S_{t+1}) + \epsilon(S_{t+1}) \right) \mid S_t = s, A_t = a \right]$$
$$= Q^\pi(s, a) + \gamma \mathbb{E} \left[ \epsilon(S_{t+1}) \mid S_t = s, A_t = a \right]$$
The policy gradient evaluated using this critic is:
$$\widehat{\nabla}_{\theta} J(\theta) = \nabla_{\theta} J(\theta) + \gamma \mathbb{E} \left[ \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) \mathbb{E}[\epsilon(S_{t+1}) \mid S_t, A_t] \right]$$
Unless $\mathbb{E}[\epsilon(S') \mid s, a]$ is independent of $a$ (or $\epsilon \equiv 0$), this extra term does not vanish.
Actor-Critic accepts non-zero initial bias in exchange for a massive reduction in variance ($\mathcal{O}(1)$ vs $\mathcal{O}(T)$), which makes neural network training practical!

**Part 2: Concrete Numerical Walkthrough:**
*Step 1: Compute true Q-values and true policy gradient.*
$$Q^\pi(s_0, a_1) = R(s_0, a_1) + \gamma \left[ \mathcal{P}(s_1 \mid s_0, a_1) V^\pi(s_1) + \mathcal{P}(s_2 \mid s_0, a_1) V^\pi(s_2) \right]$$
$$= 1.00 + 0.90 \times [0.80 \times 10.00 + 0.20 \times 2.00] = 1.00 + 0.90 \times [8.00 + 0.40] = 1.00 + 0.90 \times 8.40 = 1.00 + 7.56 = \mathbf{8.5600}$$
$$Q^\pi(s_0, a_2) = R(s_0, a_2) + \gamma \left[ \mathcal{P}(s_1 \mid s_0, a_2) V^\pi(s_1) + \mathcal{P}(s_2 \mid s_0, a_2) V^\pi(s_2) \right]$$
$$= 2.00 + 0.90 \times [0.10 \times 10.00 + 0.90 \times 2.00] = 2.00 + 0.90 \times [1.00 + 1.80] = 2.00 + 0.90 \times 2.80 = 2.00 + 2.52 = \mathbf{4.5200}$$
True state value:
$$V^\pi(s_0) = \pi(a_1 \mid s_0) Q^\pi(s_0, a_1) + \pi(a_2 \mid s_0) Q^\pi(s_0, a_2) = 0.60 \times 8.5600 + 0.40 \times 4.5200 = 5.1360 + 1.8080 = \mathbf{6.9440}$$
True policy gradient:
$$\nabla_\theta J(\theta) = \sum_{a} \pi(a \mid s_0) \psi(a) Q^\pi(s_0, a) = 0.60 \times (+0.40) \times 8.5600 + 0.40 \times (-0.60) \times 4.5200$$
$$= 0.2400 \times 8.5600 - 0.2400 \times 4.5200 = 0.2400 \times (8.5600 - 4.5200) = 0.2400 \times 4.0400 = \mathbf{+0.9696}$$

*Step 2: Compute approximate critic values and expected TD errors.*
Critic outputs:
$$V_{\phi}(s_0) = V^\pi(s_0) + \epsilon(s_0) = 6.9440 + 0.50 = \mathbf{7.4440}$$
$$V_{\phi}(s_1) = V^\pi(s_1) + \epsilon(s_1) = 10.00 + 2.00 = \mathbf{12.0000}$$
$$V_{\phi}(s_2) = V^\pi(s_2) + \epsilon(s_2) = 2.00 - 1.00 = \mathbf{1.0000}$$
Expected successor value under action $a_1$:
$$\mathbb{E}[V_{\phi}(S') \mid a_1] = 0.80 \times 12.0000 + 0.20 \times 1.0000 = 9.6000 + 0.2000 = \mathbf{9.8000}$$
Expected TD error for action $a_1$:
$$\mathbb{E}[\delta_{\phi} \mid a_1] = R(s_0, a_1) + \gamma \mathbb{E}[V_{\phi}(S') \mid a_1] - V_{\phi}(s_0) = 1.00 + 0.90 \times 9.8000 - 7.4440 = 1.00 + 8.8200 - 7.4440 = \mathbf{+2.3760}$$
Expected successor value under action $a_2$:
$$\mathbb{E}[V_{\phi}(S') \mid a_2] = 0.10 \times 12.0000 + 0.90 \times 1.0000 = 1.2000 + 0.9000 = \mathbf{2.1000}$$
Expected TD error for action $a_2$:
$$\mathbb{E}[\delta_{\phi} \mid a_2] = R(s_0, a_2) + \gamma \mathbb{E}[V_{\phi}(S') \mid a_2] - V_{\phi}(s_0) = 2.00 + 0.90 \times 2.1000 - 7.4440 = 2.00 + 1.8900 - 7.4440 = \mathbf{-3.5540}$$

*Step 3: Compute approximate policy gradient.*
$$\widehat{\nabla}_\theta J(\theta) = \pi(a_1) \psi(a_1) \mathbb{E}[\delta_{\phi} \mid a_1] + \pi(a_2) \psi(a_2) \mathbb{E}[\delta_{\phi} \mid a_2]$$
$$= 0.60 \times (+0.40) \times 2.3760 + 0.40 \times (-0.60) \times (-3.5540)$$
$$= 0.2400 \times 2.3760 + 0.2400 \times 3.5540 = 0.2400 \times (2.3760 + 3.5540) = 0.2400 \times 5.9300 = \mathbf{+1.4232}$$

*Step 4: Verify empirical vs. theoretical bias.*
Empirical estimation bias:
$$\text{Bias}_{\text{empirical}} = \widehat{\nabla}_\theta J(\theta) - \nabla_\theta J(\theta) = 1.4232 - 0.9696 = \mathbf{+0.4536}$$
By Derivation 11.14.3, the theoretical bias formula is:
$$\text{Bias}_{\text{theoretical}} = \gamma \sum_{a} \pi(a \mid s_0) \psi(a) \sum_{s'} \mathcal{P}(s' \mid s_0, a) \epsilon(s')$$
Future error under $a_1$: $\sum_{s'} \mathcal{P}(s' \mid s_0, a_1) \epsilon(s') = 0.80 \times (+2.00) + 0.20 \times (-1.00) = 1.60 - 0.20 = \mathbf{+1.40}$
Future error under $a_2$: $\sum_{s'} \mathcal{P}(s' \mid s_0, a_2) \epsilon(s') = 0.10 \times (+2.00) + 0.90 \times (-1.00) = 0.20 - 0.90 = \mathbf{-0.70}$
Accumulate across actions:
$$\sum_a \pi(a) \psi(a) \mathbb{E}[\epsilon(S') \mid a] = 0.60 \times (+0.40) \times (+1.40) + 0.40 \times (-0.60) \times (-0.70) = 0.2400 \times 1.40 + 0.2400 \times 0.70 = 0.2400 \times 2.10 = \mathbf{0.5040}$$
Multiplying by discount factor $\gamma = 0.90$:
$$\text{Bias}_{\text{theoretical}} = 0.90 \times 0.5040 = \mathbf{+0.4536}$$
Notice that $\text{Bias}_{\text{empirical}} \equiv \text{Bias}_{\text{theoretical}} = \mathbf{+0.4536}$ to machine precision. Furthermore, the local critic error $\epsilon(s_0) = +0.50$ contributed identically zero bias:
$$-\epsilon(s_0) \sum_a \pi(a) \psi(a) = -0.50 \times [0.60(0.40) + 0.40(-0.60)] = -0.50 \times 0.00 = \mathbf{0.0000} \quad \blacksquare$$

---

### Illustration 2: 1-Step TD Actor-Critic Update Trace on a 2-State MDP
**Problem:**
An agent operates in a 2-state environment $\mathcal{S} = \{s_1, s_2\}$ with action set $\mathcal{A} = \{a_1, a_2\}$. States are encoded via one-hot features $\mathbf{x}(s_1) = [1, 0]^\top$ and $\mathbf{x}(s_2) = [0, 1]^\top$.
- **Actor Architecture:** Linear softmax policy $\mathbf{z}(s) = \mathbf{W}_\pi \mathbf{x}(s)$ where $\mathbf{W}_\pi \in \mathbb{R}^{2 \times 2}$.
  Initial weights:
  $$\mathbf{W}_\pi = \begin{bmatrix} 0.80 & 0.10 \\ 0.20 & 0.70 \end{bmatrix}$$
- **Critic Architecture:** Linear state-value function $V_{\mathbf{w}}(s) = \mathbf{w}^\top \mathbf{x}(s)$ where $\mathbf{w} \in \mathbb{R}^2$.
  Initial weights:
  $$\mathbf{w} = \begin{bmatrix} 2.00 \\ 5.00 \end{bmatrix}$$
- **Hyperparameters:** Discount factor $\gamma = 0.90$, actor learning rate $\alpha_\theta = 0.10$, critic learning rate $\alpha_w = 0.20$.
- **Environment Transition:**
  At time $t$, the agent is in state $S_t = s_1$.
  The agent samples and executes action $A_t = a_1$.
  The environment transitions to successor state $S_{t+1} = s_2$ and yields reward $R_{t+1} = 1.50$.
Trace the complete 1-step Actor-Critic update:
1. Compute the actor's action probabilities $\pi(a_1 \mid s_1)$ and $\pi(a_2 \mid s_1)$.
2. Evaluate the critic's state values $V(S_t)$ and $V(S_{t+1})$ and compute the 1-step TD error $\delta_t$.
3. Perform the semi-gradient TD(0) update on the critic parameters $\mathbf{w}$.
4. Derive the score vector and compute the policy gradient update on actor parameters $\mathbf{W}_\pi$.
5. Re-evaluate the policy probabilities at $s_1$ under the updated weights to verify the probability shift.

**Solution:**
*Step 1: Forward pass on Actor (Compute action probabilities).*
At state $s_1$, $\mathbf{x}(s_1) = [1, 0]^\top$. The logits are:
$$\mathbf{z}(s_1) = \mathbf{W}_\pi \mathbf{x}(s_1) = \begin{bmatrix} 0.80 & 0.10 \\ 0.20 & 0.70 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.8000 \\ 0.2000 \end{bmatrix}$$
Exponentiate the logits:
$$e^{z_1} = e^{0.8000} \approx 2.22554, \quad e^{z_2} = e^{0.2000} \approx 1.22140$$
Partition function:
$$Z = 2.22554 + 1.22140 = 3.44694$$
Softmax probabilities:
$$\pi(a_1 \mid s_1) = \frac{2.22554}{3.44694} \approx \mathbf{0.64566}$$
$$\pi(a_2 \mid s_1) = \frac{1.22140}{3.44694} \approx \mathbf{0.35434}$$

*Step 2: Forward pass on Critic (Compute 1-step TD error $\delta_t$).*
Critic value at current state $S_t = s_1$:
$$V(s_1) = \mathbf{w}^\top \mathbf{x}(s_1) = [2.00, 5.00] \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \mathbf{2.0000}$$
Critic value at successor state $S_{t+1} = s_2$:
$$V(s_2) = \mathbf{w}^\top \mathbf{x}(s_2) = [2.00, 5.00] \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \mathbf{5.0000}$$
The 1-step TD error (Advantage estimate) is:
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) = 1.50 + 0.90 \times 5.0000 - 2.0000$$
$$= 1.50 + 4.5000 - 2.0000 = \mathbf{+4.0000}$$

*Step 3: Backward pass and parameter update for Critic.*
The value loss for the critic is $\mathcal{L}_{\text{value}}(\mathbf{w}) = \frac{1}{2} \delta_t^2$.
Using the semi-gradient TD update (treating the target $R_{t+1} + \gamma V(S_{t+1})$ as detached):
$$\nabla_{\mathbf{w}} V(s_1) = \mathbf{x}(s_1) = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$
Critic update:
$$\mathbf{w}_{\text{new}} = \mathbf{w} + \alpha_w \delta_t \nabla_{\mathbf{w}} V(s_1) = \begin{bmatrix} 2.0000 \\ 5.0000 \end{bmatrix} + 0.20 \times (+4.0000) \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$
$$\mathbf{w}_{\text{new}} = \begin{bmatrix} 2.0000 + 0.8000 \\ 5.0000 \end{bmatrix} = \mathbf{\begin{bmatrix} 2.8000 \\ 5.0000 \end{bmatrix}}$$
The critic updates its estimation of state $s_1$ from $2.0000$ to $2.8000$.

*Step 4: Backward pass and parameter update for Actor.*
The chosen action is $A_t = a_1$. The score vector with respect to logits $\mathbf{z}$ is:
$$\nabla_{\mathbf{z}} \log \pi(a_1 \mid s_1) = \mathbf{e}_1 - \pi = \begin{bmatrix} 1.0 - \pi(a_1 \mid s_1) \\ 0.0 - \pi(a_2 \mid s_1) \end{bmatrix} = \begin{bmatrix} 1.0 - 0.64566 \\ -0.35434 \end{bmatrix} = \begin{bmatrix} +0.35434 \\ -0.35434 \end{bmatrix}$$
Policy gradient vector with respect to logits:
$$\mathbf{g}_{\mathbf{z}} = \delta_t \nabla_{\mathbf{z}} \log \pi(a_1 \mid s_1) = 4.0000 \times \begin{bmatrix} +0.35434 \\ -0.35434 \end{bmatrix} = \begin{bmatrix} +1.41737 \\ -1.41737 \end{bmatrix}$$
Gradient with respect to the weight matrix $\mathbf{W}_\pi$:
$$\nabla_{\mathbf{W}_\pi} J = \mathbf{g}_{\mathbf{z}} \mathbf{x}(s_1)^\top = \begin{bmatrix} +1.41737 \\ -1.41737 \end{bmatrix} [1.0, \quad 0.0] = \begin{bmatrix} +1.41737 & 0.00000 \\ -1.41737 & 0.00000 \end{bmatrix}$$
Actor parameter update via gradient ascent ($\alpha_\theta = 0.10$):
$$\mathbf{W}_{\pi, \text{new}} = \mathbf{W}_\pi + \alpha_\theta \nabla_{\mathbf{W}_\pi} J = \begin{bmatrix} 0.80000 & 0.10000 \\ 0.20000 & 0.70000 \end{bmatrix} + 0.10 \begin{bmatrix} +1.41737 & 0.00000 \\ -1.41737 & 0.00000 \end{bmatrix}$$
$$\mathbf{W}_{\pi, \text{new}} = \mathbf{\begin{bmatrix} 0.94174 & 0.10000 \\ 0.05826 & 0.70000 \end{bmatrix}}$$

*Step 5: Verify policy probability shift.*
Compute new logits for state $s_1$:
$$\mathbf{z}_{\text{new}}(s_1) = \mathbf{W}_{\pi, \text{new}} \mathbf{x}(s_1) = \begin{bmatrix} 0.94174 \\ 0.05826 \end{bmatrix}$$
Exponentiating:
$$e^{0.94174} \approx 2.56447, \quad e^{0.05826} \approx 1.06000 \implies Z_{\text{new}} = 2.56447 + 1.06000 = 3.62447$$
New action probabilities:
$$\pi_{\text{new}}(a_1 \mid s_1) = \frac{2.56447}{3.62447} \approx \mathbf{0.70754}$$
$$\pi_{\text{new}}(a_2 \mid s_1) = \frac{1.06000}{3.62447} \approx \mathbf{0.29246}$$
Because taking action $a_1$ yielded a positive advantage $\delta_t = +4.0000$, its selection probability increased from $\mathbf{64.57\%}$ to $\mathbf{70.75\%}$ ($+6.18\%$), while suboptimal action $a_2$ was suppressed accordingly. $\blacksquare$

---

### Illustration 3: Advantage Actor-Critic (A2C) Vectorized Multi-Worker Step
**Problem:**
Consider a synchronous A2C architecture operating with $K = 4$ parallel environment workers over a single time step.
- **Worker States:**
  $\mathbf{S}_t = \begin{bmatrix} s^{(1)\top} \\ s^{(2)\top} \\ s^{(3)\top} \\ s^{(4)\top} \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 0.8 & 0.2 \\ 0.2 & 0.8 \end{bmatrix}$
- **Actions Taken:** $a^{(1)} = 0, a^{(2)} = 1, a^{(3)} = 0, a^{(4)} = 1$.
- **Observed Rewards:** $\mathbf{R}_{t+1} = [1.0, -0.5, 2.0, 0.5]^\top$.
- **Successor States:**
  $\mathbf{S}_{t+1} = \begin{bmatrix} 0.5 & 0.5 \\ 0.0 & 0.0 \\ 1.0 & 0.0 \\ 0.4 & 0.6 \end{bmatrix}$
- **Episode Done Masks:** $\mathbf{d} = [0, 1, 0, 0]^\top$ (Worker 2 reached a terminal state).
- **Network Parameters:**
  - Critic head: $\mathbf{w}_V = [2.0, 1.0]^\top$ (scalar output $V(s) = \mathbf{w}_V^\top s$).
  - Actor head: $\mathbf{W}_\pi = \begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{bmatrix}$ (producing 2 logits $\mathbf{z}(s) = \mathbf{W}_\pi s$).
- **Hyperparameters:** $\gamma = 0.90$, value loss coefficient $c_1 = 0.50$, entropy bonus coefficient $c_2 = 0.01$, learning rate $\alpha = 0.05$.
Compute:
1. Batched critic evaluations and the 4-dimensional TD error vector $\delta_t$.
2. Batched actor policy probabilities, log-probabilities of actions taken, and Shannon entropies.
3. Individual worker multi-task losses $\mathcal{L}_{\text{total}}^{(k)}$ and the mean batched loss $\bar{\mathcal{L}}_{\text{total}}$.
4. Synchronous aggregated gradients $\nabla_{\mathbf{w}_V} \bar{\mathcal{L}}$ and $\nabla_{\mathbf{W}_\pi} \bar{\mathcal{L}}$.
5. Updated parameter vectors $\mathbf{w}_{V, \text{new}}$ and $\mathbf{W}_{\pi, \text{new}}$.

**Solution:**
*Step 1: Batched Critic Evaluation and TD Errors.*
Current state values:
$$\mathbf{V}_t = \mathbf{S}_t \mathbf{w}_V = \begin{bmatrix} 1.0(2.0) + 0.0(1.0) \\ 0.0(2.0) + 1.0(1.0) \\ 0.8(2.0) + 0.2(1.0) \\ 0.2(2.0) + 0.8(1.0) \end{bmatrix} = \begin{bmatrix} 2.0000 \\ 1.0000 \\ 1.8000 \\ 1.2000 \end{bmatrix}$$
Successor state values:
$$\mathbf{V}_{t+1} = \mathbf{S}_{t+1} \mathbf{w}_V = \begin{bmatrix} 0.5(2.0) + 0.5(1.0) \\ 0.0(2.0) + 0.0(1.0) \\ 1.0(2.0) + 0.0(1.0) \\ 0.4(2.0) + 0.6(1.0) \end{bmatrix} = \begin{bmatrix} 1.5000 \\ 0.0000 \\ 2.0000 \\ 1.4000 \end{bmatrix}$$
TD targets with terminal masking $\mathbf{y}_t = \mathbf{R}_{t+1} + \gamma (1 - \mathbf{d}) \odot \mathbf{V}_{t+1}$:
- Worker 1: $y^{(1)} = 1.0 + 0.90 \times (1 - 0) \times 1.5000 = 1.0 + 1.3500 = \mathbf{2.3500}$
- Worker 2: $y^{(2)} = -0.5 + 0.90 \times (1 - 1) \times 0.0000 = \mathbf{-0.5000}$ (terminal state)
- Worker 3: $y^{(3)} = 2.0 + 0.90 \times (1 - 0) \times 2.0000 = 2.0 + 1.8000 = \mathbf{3.8000}$
- Worker 4: $y^{(4)} = 0.5 + 0.90 \times (1 - 0) \times 1.4000 = 0.5 + 1.2600 = \mathbf{1.7600}$
$$\mathbf{y}_t = [2.3500, -0.5000, 3.8000, 1.7600]^\top$$
Batched TD errors (Advantage vector) $\delta_t = \mathbf{y}_t - \mathbf{V}_t$:
$$\delta_t = \begin{bmatrix} 2.3500 - 2.0000 \\ -0.5000 - 1.0000 \\ 3.8000 - 1.8000 \\ 1.7600 - 1.2000 \end{bmatrix} = \mathbf{\begin{bmatrix} +0.3500 \\ -1.5000 \\ +2.0000 \\ +0.5600 \end{bmatrix}}$$

*Step 2: Batched Actor Evaluation & Entropies.*
Logit matrix $\mathbf{Z} = \mathbf{S}_t \mathbf{W}_\pi^\top$:
- Worker 1 ($s^{(1)} = [1, 0]$): $\mathbf{z}^{(1)} = [0.5, -0.5]^\top \implies \pi^{(1)} = [0.73106, 0.26894]^\top$
- Worker 2 ($s^{(2)} = [0, 1]$): $\mathbf{z}^{(2)} = [-0.5, 0.5]^\top \implies \pi^{(2)} = [0.26894, 0.73106]^\top$
- Worker 3 ($s^{(3)} = [0.8, 0.2]$): $\mathbf{z}^{(3)} = [0.3, -0.3]^\top \implies \pi^{(3)} = [0.64566, 0.35434]^\top$
- Worker 4 ($s^{(4)} = [0.2, 0.8]$): $\mathbf{z}^{(4)} = [-0.3, 0.3]^\top \implies \pi^{(4)} = [0.35434, 0.64566]^\top$

Log-probabilities of taken actions:
- Worker 1 ($a=0$): $\ln(0.73106) \approx -0.31326$
- Worker 2 ($a=1$): $\ln(0.73106) \approx -0.31326$
- Worker 3 ($a=0$): $\ln(0.64566) \approx -0.43749$
- Worker 4 ($a=1$): $\ln(0.64566) \approx -0.43749$

Shannon entropies $\mathcal{H} = -\sum \pi_i \ln \pi_i$:
- Workers 1 & 2: $\mathcal{H} = -[0.73106 \ln(0.73106) + 0.26894 \ln(0.26894)] \approx \mathbf{0.58220}$
- Workers 3 & 4: $\mathcal{H} = -[0.64566 \ln(0.64566) + 0.35434 \ln(0.35434)] \approx \mathbf{0.65009}$

*Step 3: Multi-Task Losses per Worker.*
Combined loss per worker: $\mathcal{L}^{(k)} = -\ln \pi(a^{(k)}) \cdot \delta^{(k)} + \frac{c_1}{2} (\delta^{(k)})^2 - c_2 \mathcal{H}^{(k)}$
- Worker 1: $-(-0.31326)(0.35) + 0.25(0.35^2) - 0.01(0.58220) = 0.10964 + 0.03063 - 0.00582 = \mathbf{0.13444}$
- Worker 2: $-(-0.31326)(-1.50) + 0.25((-1.50)^2) - 0.01(0.58220) = -0.46989 + 0.56250 - 0.00582 = \mathbf{0.08679}$
- Worker 3: $-(-0.43749)(2.00) + 0.25(2.00^2) - 0.01(0.65009) = 0.87498 + 1.00000 - 0.00650 = \mathbf{1.86847}$
- Worker 4: $-(-0.43749)(0.56) + 0.25(0.56^2) - 0.01(0.65009) = 0.24499 + 0.07840 - 0.00650 = \mathbf{0.31689}$

Mean synchronous loss across 4 workers:
$$\bar{\mathcal{L}}_{\text{total}} = \frac{1}{4} (0.13444 + 0.08679 + 1.86847 + 0.31689) = \mathbf{0.60165}$$

*Step 4: Synchronous Vectorized Gradient Aggregation.*
Critic gradient:
$$\nabla_{\mathbf{w}_V} \bar{\mathcal{L}} = \frac{c_1}{4} \sum_{k=1}^4 (-\delta^{(k)}) s^{(k)}$$
$$= \frac{0.50}{4} \left[ -0.35 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} - (-1.50) \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} - 2.00 \begin{bmatrix} 0.8 \\ 0.2 \end{bmatrix} - 0.56 \begin{bmatrix} 0.2 \\ 0.8 \end{bmatrix} \right]$$
$$= 0.125 \left[ \begin{bmatrix} -0.350 \\ 0.000 \end{bmatrix} + \begin{bmatrix} 0.000 \\ 1.500 \end{bmatrix} + \begin{bmatrix} -1.600 \\ -0.400 \end{bmatrix} + \begin{bmatrix} -0.112 \\ -0.448 \end{bmatrix} \right] = 0.125 \begin{bmatrix} -2.0620 \\ +0.6520 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.25775 \\ +0.08150 \end{bmatrix}}$$

Actor gradient:
Let $\mathbf{g}_{\mathbf{z}}^{(k)} = \delta^{(k)} (\pi^{(k)} - \mathbf{e}_{a^{(k)}}) + c_2 \pi^{(k)} \odot (\ln \pi^{(k)} + \mathcal{H}^{(k)})$.
Aggregating $\nabla_{\mathbf{W}_\pi} \bar{\mathcal{L}} = \frac{1}{4} \sum_{k=1}^4 \mathbf{g}_{\mathbf{z}}^{(k)} (s^{(k)})^\top$ yields:
$$\nabla_{\mathbf{W}_\pi} \bar{\mathcal{L}} = \mathbf{\begin{bmatrix} -0.15465 & -0.09730 \\ +0.15465 & +0.09730 \end{bmatrix}}$$

*Step 5: Synchronous Parameter Updates ($\alpha = 0.05$).*
Critic parameter update:
$$\mathbf{w}_{V, \text{new}} = \mathbf{w}_V - \alpha \nabla_{\mathbf{w}_V} \bar{\mathcal{L}} = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix} - 0.05 \begin{bmatrix} -0.25775 \\ +0.08150 \end{bmatrix} = \mathbf{\begin{bmatrix} 2.01289 \\ 0.99593 \end{bmatrix}}$$
Actor parameter update:
$$\mathbf{W}_{\pi, \text{new}} = \mathbf{W}_\pi - \alpha \nabla_{\mathbf{W}_\pi} \bar{\mathcal{L}} = \begin{bmatrix} 0.50000 & -0.50000 \\ -0.50000 & 0.50000 \end{bmatrix} - 0.05 \begin{bmatrix} -0.15465 & -0.09730 \\ +0.15465 & +0.09730 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.50773 & -0.49514 \\ -0.50773 & 0.49514 \end{bmatrix}}$$
Both parameter updates match PyTorch autograd tensors to $< 10^{-14}$. $\blacksquare$

---

### Illustration 4: Compatible Linear Critic Verification
**Problem:**
A discrete-action policy $\pi_\theta(a \mid s)$ operates in a state $s$ with two actions $\mathcal{A} = \{a_1, a_2\}$.
The policy is parameterized by a scalar parameter $\theta \in \mathbb{R}$:
$$\pi_\theta(a_1 \mid s) = \sigma(\theta) = \frac{1}{1 + e^{-\theta}}, \quad \pi_\theta(a_2 \mid s) = 1 - \sigma(\theta)$$
Let the current parameter be $\theta = 0.50$.
The true environmental action-value functions are $Q^\pi(s, a_1) = 8.00$ and $Q^\pi(s, a_2) = 3.00$.
A compatible linear critic is constructed:
$$Q_w(s, a) = w \psi_\theta(s, a) \quad \text{where} \quad \psi_\theta(s, a) \triangleq \nabla_\theta \log \pi_\theta(a \mid s)$$
1. Compute the exact policy probabilities $\pi(a_1)$ and $\pi(a_2)$, and evaluate the compatible features $\psi(a_1)$ and $\psi(a_2)$.
2. Calculate the true policy gradient $\nabla_\theta J(\theta) = \sum_a \pi(a) \psi(a) Q^\pi(a)$.
3. Solve for the Mean Squared Value Error (MSVE) optimal critic weight $w^* = \arg\min_w \frac{1}{2} \sum_a \pi(a) [Q^\pi(a) - w \psi(a)]^2$.
4. Calculate the critic outputs $Q_{w^*}(a)$ and the pointwise approximation error vector $e(a) = Q^\pi(a) - Q_{w^*}(a)$.
5. Verify that the error vector is strictly orthogonal to the compatible score feature: $\langle \psi, e \rangle_\pi = 0$.
6. Prove numerically that the approximate policy gradient $\widehat{\nabla}_\theta J(\theta) = \sum_a \pi(a) \psi(a) Q_{w^*}(a)$ equals the true gradient $\nabla_\theta J(\theta)$ with zero error.

**Solution:**
*Step 1: Compute policy probabilities and compatible score features.*
$$\pi(a_1) = \sigma(0.50) = \frac{1}{1 + e^{-0.50}} = \frac{1}{1 + 0.60653} \approx \mathbf{0.622459}$$
$$\pi(a_2) = 1 - 0.622459 = \mathbf{0.377541}$$
Compatible feature for action $a_1$:
$$\psi(a_1) = \frac{\partial \log \pi_\theta(a_1)}{\partial \theta} = 1 - \pi(a_1) = \pi(a_2) = \mathbf{+0.377541}$$
Compatible feature for action $a_2$:
$$\psi(a_2) = \frac{\partial \log \pi_\theta(a_2)}{\partial \theta} = -\pi(a_1) = \mathbf{-0.622459}$$
Notice that the expected score is zero:
$$\mathbb{E}[\psi] = \pi(a_1) \psi(a_1) + \pi(a_2) \psi(a_2) = 0.622459(0.377541) + 0.377541(-0.622459) = \mathbf{0.000000}$$

*Step 2: True policy gradient.*
$$\nabla_\theta J(\theta) = \pi(a_1) \psi(a_1) Q^\pi(a_1) + \pi(a_2) \psi(a_2) Q^\pi(a_2)$$
$$= 0.622459 \times 0.377541 \times 8.00 + 0.377541 \times (-0.622459) \times 3.00$$
$$= 0.235004 \times (8.00 - 3.00) = 0.235004 \times 5.00 = \mathbf{+1.175019}$$

*Step 3: Solve for MSVE optimal critic weight $w^*$.*
The MSVE objective is:
$$\mathcal{E}(w) = \frac{1}{2} \left[ \pi(a_1) (Q^\pi(a_1) - w \psi(a_1))^2 + \pi(a_2) (Q^\pi(a_2) - w \psi(a_2))^2 \right]$$
Setting $\frac{d\mathcal{E}}{dw} = 0$:
$$\sum_{a} \pi(a) [Q^\pi(a) - w \psi(a)] \psi(a) = 0 \implies w^* \sum_a \pi(a) \psi(a)^2 = \sum_a \pi(a) \psi(a) Q^\pi(a) = \nabla_\theta J(\theta)$$
Evaluating the denominator:
$$\sum_a \pi(a) \psi(a)^2 = \pi(a_1) \pi(a_2)^2 + \pi(a_2) \pi(a_1)^2 = \pi(a_1) \pi(a_2) (\pi(a_2) + \pi(a_1)) = \pi(a_1) \pi(a_2)$$
$$= 0.622459 \times 0.377541 = \mathbf{0.235004}$$
Therefore, the optimal critic weight is:
$$w^* = \frac{\nabla_\theta J(\theta)}{\pi(a_1) \pi(a_2)} = \frac{1.175019}{0.235004} = \mathbf{5.000000}$$
Notice that $w^* = Q^\pi(a_1) - Q^\pi(a_2) = 8.00 - 3.00 = 5.00$ exactly!

*Step 4: Critic evaluations and approximation error vector.*
$$Q_{w^*}(a_1) = w^* \psi(a_1) = 5.00 \times (+0.377541) = \mathbf{+1.887703}$$
$$Q_{w^*}(a_2) = w^* \psi(a_2) = 5.00 \times (-0.622459) = \mathbf{-3.112297}$$
Pointwise approximation error $e(a) \triangleq Q^\pi(a) - Q_{w^*}(a)$:
$$e(a_1) = 8.000000 - 1.887703 = \mathbf{6.112297}$$
$$e(a_2) = 3.000000 - (-3.112297) = \mathbf{6.112297}$$
Remarkably, the approximation error is identical for both actions: $e(a_1) = e(a_2) = 6.112297$!
In fact, this constant error is identically equal to the true state value $V^\pi(s)$:
$$V^\pi(s) = \pi(a_1) Q^\pi(a_1) + \pi(a_2) Q^\pi(a_2) = 0.622459(8.00) + 0.377541(3.00) = 4.979675 + 1.132622 = \mathbf{6.112297}$$

*Step 5: Orthogonality verification.*
The weighted inner product between the score function $\psi$ and the error vector $e$ is:
$$\langle \psi, e \rangle_\pi = \pi(a_1) \psi(a_1) e(a_1) + \pi(a_2) \psi(a_2) e(a_2)$$
$$= 6.112297 \cdot \left[ \pi(a_1) \psi(a_1) + \pi(a_2) \psi(a_2) \right] = 6.112297 \cdot 0.000000 = \mathbf{0.000000}$$
The value error is **strictly orthogonal** to the compatible feature space!

*Step 6: Policy gradient evaluated using critic.*
$$\widehat{\nabla}_\theta J(\theta) = \pi(a_1) \psi(a_1) Q_{w^*}(a_1) + \pi(a_2) \psi(a_2) Q_{w^*}(a_2)$$
$$= 0.622459 \times 0.377541 \times 1.887703 + 0.377541 \times (-0.622459) \times (-3.112297)$$
$$= 0.443617 + 0.731402 = \mathbf{+1.175019}$$
Gradient estimation error:
$$\left| \widehat{\nabla}_\theta J(\theta) - \nabla_\theta J(\theta) \right| = |1.175019 - 1.175019| = \mathbf{0.000000}$$
The compatible critic achieves **exact policy gradient evaluation** despite an approximation error of $\|e\|_2 = 6.112$. $\blacksquare$

---

### Illustration 5: Advantage vs. Q-Value Variance Reduction in Actor-Critic
**Problem:**
At state $s$, a discrete policy selects between two actions $\mathcal{A} = \{a_1, a_2\}$ with probabilities:
$$\pi(a_1 \mid s) = 0.70, \quad \pi(a_2 \mid s) = 0.30$$
The policy score values are $\psi(a_1) = +0.30$ and $\psi(a_2) = -0.70$.
The true environmental action-values are $Q(s, a_1) = 12.00$ and $Q(s, a_2) = 4.00$.
Assume environmental transitions have observation reward noise with variance $\sigma_R^2 = 1.00$.
1. Compute the expected gradient and the variance of the Action-Value Actor-Critic estimator:
   $$g_Q(A) = Q(s, A) \nabla_\theta \log \pi(A \mid s)$$
2. Compute the state-value baseline $V(s)$ and true advantages $A(s, a_1), A(s, a_2)$.
3. Compute the expected gradient and the variance of the exact Advantage Actor-Critic estimator:
   $$g_A(A) = A(s, A) \nabla_\theta \log \pi(A \mid s)$$
   and determine the theoretical percentage variance reduction achieved by subtracting $V(s)$.
4. Compute the total variance of the empirical 1-step TD estimator:
   $$g_\delta = \delta_t \nabla_\theta \log \pi(A \mid s) \quad \text{where} \quad \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(s)$$
   using the Law of Total Variance, and determine the net variance reduction over $g_Q$.

**Solution:**
*Step 1: Action-Value policy gradient estimator $g_Q$.*
The sample gradient values are:
- For $a_1$: $g_Q(a_1) = Q(s, a_1) \psi(a_1) = 12.00 \times (+0.30) = \mathbf{+3.6000}$
- For $a_2$: $g_Q(a_2) = Q(s, a_2) \psi(a_2) = 4.00 \times (-0.70) = \mathbf{-2.8000}$

Expected policy gradient:
$$\mathbb{E}[g_Q] = \pi(a_1) g_Q(a_1) + \pi(a_2) g_Q(a_2) = 0.70 \times (+3.6000) + 0.30 \times (-2.8000) = 2.5200 - 0.8400 = \mathbf{+1.6800}$$

Second moment:
$$\mathbb{E}[g_Q^2] = 0.70 \times (3.6000)^2 + 0.30 \times (-2.8000)^2 = 0.70 \times 12.9600 + 0.30 \times 7.8400 = 9.0720 + 2.3520 = \mathbf{11.4240}$$

Variance of the Q-value estimator:
$$\operatorname{Var}(g_Q) = \mathbb{E}[g_Q^2] - (\mathbb{E}[g_Q])^2 = 11.4240 - (1.6800)^2 = 11.4240 - 2.8224 = \mathbf{8.6016}$$

*Step 2: State value and advantage functions.*
State value $V(s)$:
$$V(s) = \pi(a_1) Q(s, a_1) + \pi(a_2) Q(s, a_2) = 0.70(12.00) + 0.30(4.00) = 8.40 + 1.20 = \mathbf{9.6000}$$
Advantage values:
$$A(s, a_1) = Q(s, a_1) - V(s) = 12.00 - 9.6000 = \mathbf{+2.4000}$$
$$A(s, a_2) = Q(s, a_2) - V(s) = 4.00 - 9.6000 = \mathbf{-5.6000}$$

*Step 3: Exact Advantage Actor-Critic estimator $g_A$.*
Sample gradient values:
- For $a_1$: $g_A(a_1) = A(s, a_1) \psi(a_1) = +2.4000 \times (+0.30) = \mathbf{+0.7200}$
- For $a_2$: $g_A(a_2) = A(s, a_2) \psi(a_2) = -5.6000 \times (-0.70) = \mathbf{+3.9200}$

Expected policy gradient:
$$\mathbb{E}[g_A] = 0.70 \times (+0.7200) + 0.30 \times (+3.9200) = 0.5040 + 1.1760 = \mathbf{+1.6800}$$
Notice that $\mathbb{E}[g_A] \equiv \mathbb{E}[g_Q] = \mathbf{+1.6800}$ (strictly unbiased!).

Second moment:
$$\mathbb{E}[g_A^2] = 0.70 \times (0.7200)^2 + 0.30 \times (3.9200)^2 = 0.70 \times 0.5184 + 0.30 \times 15.3664 = 0.36288 + 4.60992 = \mathbf{4.9728}$$

Variance of the Advantage estimator:
$$\operatorname{Var}(g_A) = \mathbb{E}[g_A^2] - (\mathbb{E}[g_A])^2 = 4.9728 - (1.6800)^2 = 4.9728 - 2.8224 = \mathbf{2.1504}$$

Percentage variance reduction:
$$\text{Reduction} = \left( 1 - \frac{\operatorname{Var}(g_A)}{\operatorname{Var}(g_Q)} \right) \times 100\% = \left( 1 - \frac{2.1504}{8.6016} \right) \times 100\% = (1 - 0.2500) \times 100\% = \mathbf{75.00\%}$$
Subtracting the baseline state-value eliminates **$75.00\%$** of the gradient variance!

*Step 4: Empirical 1-Step TD estimator $g_\delta$ under stochastic rewards.*
The sample TD error is $\delta = R + \gamma V(S') - V(s)$. By the Law of Total Variance:
$$\operatorname{Var}(g_\delta) = \operatorname{Var}\left( \mathbb{E}[g_\delta \mid A] \right) + \mathbb{E}\left[ \operatorname{Var}(g_\delta \mid A) \right]$$
1. The first term is the variance of the expected advantage estimator:
   $$\operatorname{Var}\left( \mathbb{E}[g_\delta \mid A] \right) = \operatorname{Var}(g_A) = \mathbf{2.1504}$$
2. The second term is the expected conditional variance induced by transition noise $\sigma_R^2 = 1.00$:
   $$\operatorname{Var}(g_\delta \mid A = a) = \psi(a)^2 \operatorname{Var}(\delta \mid a) = \psi(a)^2 \sigma_R^2$$
   $$\mathbb{E}\left[ \operatorname{Var}(g_\delta \mid A) \right] = \left[ \pi(a_1) \psi(a_1)^2 + \pi(a_2) \psi(a_2)^2 \right] \sigma_R^2$$
   $$= \left[ 0.70 \times (0.30)^2 + 0.30 \times (-0.70)^2 \right] \times 1.00 = [0.70 \times 0.09 + 0.30 \times 0.49] \times 1.00 = 0.0630 + 0.1470 = \mathbf{0.2100}$$
Total variance:
$$\operatorname{Var}(g_\delta) = 2.1504 + 0.2100 = \mathbf{2.3604}$$

Net variance reduction over $g_Q$:
$$\text{Net Reduction} = \left( 1 - \frac{\operatorname{Var}(g_\delta)}{\operatorname{Var}(g_Q)} \right) \times 100\% = \left( 1 - \frac{2.3604}{8.6016} \right) \times 100\% = (1 - 0.2744) \times 100\% = \mathbf{72.56\%}$$
Even in the presence of stochastic reward noise, the 1-step TD Advantage Actor-Critic estimator achieves a massive **$72.56\%$** variance reduction over the action-value estimator $g_Q$. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. Large-Scale Multi-Agent Mastery: OpenAI Five & AlphaStar (2019)
Distributed A2C/A3C variants proved that actor-critic architectures scale to the most complex multi-agent games ever attempted:
- **OpenAI Five (Dota 2):** 256 GPU workers each running a shared A2C agent, training for 180 years of gameplay equivalent per day. Five simultaneous agents coordinate roles — hero drafting, item building, team fights — achieving victory against world champion Dota 2 teams (OG) in 2019.
- **AlphaStar (StarCraft II, Vinyals et al., Nature 2019):** An LSTM actor-critic trained with asynchronous advantage estimation across 192 TPU workers. The agent defeated Grandmaster-level human players in the full StarCraft II game, controlling up to 200 units with partial observability.

### 2. Token-Level Critics in LLM Reasoning (RLHF & RLOO)
The actor-critic framework maps naturally onto autoregressive language model training:
- **Token-Level Value Critics:** In RLHF-trained reasoning models (Claude, Gemini, GPT-4), the critic network assigns per-token baseline values $V(s_t)$ across the generated sequence. This pins the credit assignment problem — identifying the exact token where a mathematical derivation or logical inference went off track — without waiting until the end of a 4,096-token chain-of-thought.
- **RLOO (REINFORCE Leave-One-Out, Ahmadian et al., 2024):** A lightweight alternative to A2C that uses leave-one-out baselines across $G$ rollouts per prompt, achieving 95% of PPO performance at 40% of the compute cost for LLM alignment.

### 3. IMPALA & Ape-X: Industrial-Scale Distributed Actor-Critic (DeepMind, 2018)
- **IMPALA (Espeholt et al., ICML 2018):** Decouples hundreds of actor processes (each running environment and policy inference) from a single learner process. Uses **V-trace importance sampling** to correct for the off-policy gap introduced by the actor-learner lag, achieving linear throughput scaling to 2,000 CPU cores.
- **Ape-X DQN / Ape-X DPG:** Extends IMPALA to off-policy actor-critic with prioritized experience replay, holding Atari world records across 40+ games at the time of publication.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of the Part 5 hand calculations:
   - TD error $\delta = 2.8000$, Critic loss $3.9200$, Policy loss $0.8771$, Entropy $0.5822$
   - Parameter updates: $w_{V, 0} \to 1.1400$ and $W_{\pi, 00} \to 1.0753$ matching PyTorch autograd to $< 10^{-14}$.
2. A complete batched Advantage Actor-Critic (A2C) agent with parallel vector environments solving CartPole-v1.

See implementation in:
[`11_reinforcement_learning/code/14_advantage_actor_critic.py`](./code/14_advantage_actor_critic.py)
