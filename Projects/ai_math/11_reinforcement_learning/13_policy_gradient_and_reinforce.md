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

### 2.7 First-Principles Mathematical Derivations

#### Derivation 11.13.1: Complete First-Principles Proof of the Policy Gradient Theorem for General MDPs

```
====================================================================================================
DERIVATION 11.13.1: The Policy Gradient Theorem for General MDPs
====================================================================================================
Problem Statement:
Let an MDP have state space 𝒮, action space 𝒜, transition dynamics 𝒫(s' | s, a), and discount γ ∈ [0, 1).
For policy π_θ, define the starting-state objective J(θ) ≜ 𝔼_{S_0 ~ μ}[V^{π_θ}(S_0)].
Prove from first principles that:
    ∇_θ J(θ) = ∑_{s ∈ 𝒮} d^{π_θ}(s) ∑_{a ∈ 𝒜} ∇_θ π_θ(a | s) Q^{π_θ}(s, a)
where d^{π_θ}(s) ≜ ∑_{t=0}^∞ γ^t ℙ(S_t = s | S_0 ~ μ) is the unnormalized discounted state visitation distribution,
WITHOUT assuming access to or knowledge of the transition dynamics 𝒫(s' | s, a).
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Differentiability:** The policy $\pi_\theta(a \mid s)$ is continuously differentiable with respect to parameter vector $\theta$ for all $s \in \mathcal{S}, a \in \mathcal{A}$.
2. **Discounting:** $0 \le \gamma < 1$, ensuring convergence of all infinite-horizon series.
3. **Stationary Dynamics:** Transition probabilities $\mathcal{P}(s' \mid s, a)$ and reward distributions $\mathcal{R}(s, a)$ are stationary and completely independent of policy parameters $\theta$: $\nabla_\theta \mathcal{P} = \mathbf{0}, \nabla_\theta \mathcal{R} = \mathbf{0}$.

**2. Underlying Intuition:**
Changing the policy parameters $\theta$ has a dual effect: it alters the action choices at the current state, and it alters the future distribution of states the agent will visit. Crucially, the derivative of the state visitation distribution does NOT need to be computed! By expanding the Bellman expectation equation recursively, the gradient of future state distributions telescopes into the unnormalized discounted visitation measure $d^\pi(s)$, leaving only the local policy derivative $\nabla_\theta \pi_\theta(a \mid s)$ to be evaluated.

**3. End-to-End Algebraic Derivation:**

*Step 1: Differentiate the state value function $V^{\pi_\theta}(s)$.*
Recall the Bellman expectation equation for $V^{\pi_\theta}(s)$:
$$V^{\pi_\theta}(s) = \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a)$$
Differentiating both sides with respect to $\theta$ using the product rule:
$$\nabla_\theta V^{\pi_\theta}(s) = \sum_{a \in \mathcal{A}} \left[ \nabla_\theta \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) + \pi_\theta(a \mid s) \nabla_\theta Q^{\pi_\theta}(s, a) \right] \quad \text{(Equation 1)}$$

*Step 2: Differentiate the action-value function $Q^{\pi_\theta}(s, a)$.*
Recall the Bellman equation for $Q^{\pi_\theta}(s, a)$:
$$Q^{\pi_\theta}(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^{\pi_\theta}(s')$$
Taking the gradient with respect to $\theta$:
Since $R(s, a)$ and $\mathcal{P}(s' \mid s, a)$ do not depend on $\theta$:
$$\nabla_\theta Q^{\pi_\theta}(s, a) = \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \nabla_\theta V^{\pi_\theta}(s') \quad \text{(Equation 2)}$$

*Step 3: Substitute Equation 2 into Equation 1 to form a linear recurrence.*
$$\nabla_\theta V^{\pi_\theta}(s) = \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) + \gamma \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \nabla_\theta V^{\pi_\theta}(s')$$
Define the transition probability from $s$ to $s'$ under policy $\pi_\theta$:
$$\mathcal{P}^{\pi_\theta}(s \to s') \triangleq \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \mathcal{P}(s' \mid s, a)$$
and define the local policy gradient source term:
$$\phi(s) \triangleq \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a)$$
Then Equation 1 simplifies to:
$$\nabla_\theta V^{\pi_\theta}(s) = \phi(s) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^{\pi_\theta}(s \to s') \nabla_\theta V^{\pi_\theta}(s')$$

*Step 4: Unrolling the recurrence forward in time.*
Unrolling this equation repeatedly for steps $t = 0, 1, 2, \dots$:
$$\nabla_\theta V^{\pi_\theta}(s) = \phi(s) + \gamma \sum_{s'} \mathcal{P}^{\pi_\theta}(s \to s') \left[ \phi(s') + \gamma \sum_{s''} \mathcal{P}^{\pi_\theta}(s' \to s'') \nabla_\theta V^{\pi_\theta}(s'') \right]$$
$$= \phi(s) + \gamma \sum_{s'} \mathcal{P}^{\pi_\theta}(s \to s') \phi(s') + \gamma^2 \sum_{s''} \mathcal{P}^{\pi_\theta}(s \to s'', 2) \phi(s'') + \dots$$
$$= \sum_{x \in \mathcal{S}} \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = x \mid S_0 = s; \pi_\theta) \phi(x)$$

*Step 5: Averaging over the initial state distribution $\mu(s)$.*
The objective is $J(\theta) = \sum_{s \in \mathcal{S}} \mu(s) V^{\pi_\theta}(s)$.
Taking the gradient:
$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \mu(s) \nabla_\theta V^{\pi_\theta}(s) = \sum_{s \in \mathcal{S}} \mu(s) \sum_{x \in \mathcal{S}} \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = x \mid S_0 = s; \pi_\theta) \phi(x)$$
Interchanging the orders of summation:
$$\nabla_\theta J(\theta) = \sum_{x \in \mathcal{S}} \left( \sum_{s \in \mathcal{S}} \mu(s) \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = x \mid S_0 = s; \pi_\theta) \right) \phi(x)$$
Define the unnormalized discounted state visitation distribution:
$$d^{\pi_\theta}(x) \triangleq \sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = x \mid S_0 \sim \mu; \pi_\theta)$$
Substituting $d^{\pi_\theta}(x)$ and the definition of $\phi(x)$:
$$\nabla_\theta J(\theta) = \sum_{x \in \mathcal{S}} d^{\pi_\theta}(x) \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid x) Q^{\pi_\theta}(x, a) \quad \blacksquare$$

---

#### Derivation 11.13.2: Exact Variance-Minimizing Baseline Formula

```
====================================================================================================
DERIVATION 11.13.2: The Optimal Variance-Minimizing Policy Gradient Baseline
====================================================================================================
Problem Statement:
Consider the policy gradient estimator with state-dependent baseline b(s):
    g(s, a) ≜ ∇_θ \log \pi_θ(a | s) (Q(s, a) - b(s))
Prove that the scalar baseline b*(s) that strictly minimizes the trace of the covariance matrix
    b*(s) = \arg\min_{b(s)} \operatorname{Tr}(\operatorname{Var}_{a \sim \pi}(g(s, a)))
is given in closed form by:
    b*(s) = \frac{ 𝔼_{a \sim \pi} [ ‖∇_θ \log \pi_θ(a | s)‖^2 Q(s, a) ] }{ 𝔼_{a \sim \pi} [ ‖∇_θ \log \pi_θ(a | s)‖^2 ] }
====================================================================================================
```

**1. Explicit Assumptions:**
1. **State-Conditioned Expectation:** The state $s$ is fixed; expectations are taken over $a \sim \pi_\theta(\cdot \mid s)$.
2. **Score Function Vector:** $\mathbf{\psi}(s, a) \triangleq \nabla_\theta \log \pi_\theta(a \mid s) \in \mathbb{R}^d$, with $\mathbb{E}_{a \sim \pi}[\mathbf{\psi}(s, a)] = \mathbf{0}$.
3. **Non-Degenerate Fisher Information:** $\mathbb{E}_{a \sim \pi}[\|\mathbf{\psi}(s, a)\|^2] > 0$.

**2. Underlying Intuition:**
Because subtracting any state baseline $b(s)$ preserves the expectation $\mathbb{E}[g] = \nabla_\theta J$ identically, minimizing the covariance trace $\operatorname{Tr}(\operatorname{Var}(g)) = \mathbb{E}[\|g\|^2] - \|\mathbb{E}[g]\|^2$ is equivalent to minimizing the expected squared norm $\mathbb{E}[\|\mathbf{\psi}(s, a)\|^2 (Q(s, a) - b(s))^2]$. This is a 1D quadratic polynomial in $b(s)$, whose vertex yields the optimal variance-minimizing baseline.

**3. End-to-End Algebraic Derivation:**

*Step 1: Express the variance trace.*
$$\operatorname{Tr}(\operatorname{Var}(g(s, a))) = \mathbb{E}_{a \sim \pi} \left[ \|g(s, a) - \mathbb{E}[g]\|^2 \right] = \mathbb{E}_{a \sim \pi} \left[ \|g(s, a)\|^2 \right] - \|\mathbb{E}[g(s, a)]\|^2$$
By baseline invariance (Section 2.5), $\mathbb{E}[g(s, a)] = \mathbb{E}[\mathbf{\psi}(s, a) Q(s, a)]$, which is completely independent of $b(s)$.
Therefore, minimizing variance with respect to $b(s)$ is equivalent to minimizing:
$$\mathcal{J}(b) \triangleq \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a) (Q(s, a) - b(s))\|^2 \right] = \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a)\|^2 (Q(s, a) - b(s))^2 \right]$$

*Step 2: Expand the objective as a quadratic in $b(s)$.*
$$\mathcal{J}(b) = \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a)\|^2 \left( Q(s, a)^2 - 2 b(s) Q(s, a) + b(s)^2 \right) \right]$$
$$= b(s)^2 \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a)\|^2 \right] - 2 b(s) \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a)\|^2 Q(s, a) \right] + \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a)\|^2 Q(s, a)^2 \right]$$

*Step 3: Differentiate with respect to scalar $b(s)$ and set to zero.*
$$\frac{d}{db(s)} \mathcal{J}(b) = 2 b(s) \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a)\|^2 \right] - 2 \mathbb{E}_{a \sim \pi} \left[ \|\mathbf{\psi}(s, a)\|^2 Q(s, a) \right] = 0$$
Dividing by 2 and solving for $b^*(s)$:
$$b^*(s) = \frac{\mathbb{E}_{a \sim \pi} \left[ \|\nabla_\theta \log \pi_\theta(a \mid s)\|^2 Q(s, a) \right]}{\mathbb{E}_{a \sim \pi} \left[ \|\nabla_\theta \log \pi_\theta(a \mid s)\|^2 \right]} \quad \blacksquare$$

*Step 4: Connection to the value function baseline $V^\pi(s)$.*
If the score function norm $\|\mathbf{\psi}(s, a)\|^2$ is approximately constant across actions (or uncorrelated with $Q(s, a)$):
$$\mathbb{E}[\|\mathbf{\psi}\|^2 Q] \approx \mathbb{E}[\|\mathbf{\psi}\|^2] \mathbb{E}[Q] = \mathbb{E}[\|\mathbf{\psi}\|^2] V^\pi(s)$$
Then $b^*(s) \approx \frac{\mathbb{E}[\|\mathbf{\psi}\|^2] V^\pi(s)}{\mathbb{E}[\|\mathbf{\psi}\|^2]} = V^\pi(s)$.
This proves why the state-value function $V^\pi(s)$ is universally adopted as the near-optimal baseline in Actor-Critic architectures! $\blacksquare$

---

#### Derivation 11.13.3: Analytical Policy Gradient for Multi-Variate Gaussian Policies

```
====================================================================================================
DERIVATION 11.13.3: Score Function for Multi-Variate Diagonal Gaussian Policies
====================================================================================================
Problem Statement:
Let the action space be continuous 𝒜 = ℝ^D.
The policy is parameterized as a multivariate Gaussian with diagonal covariance:
    π_θ(a | s) = 𝒩(a; 𝝁_θ(s), \operatorname{diag}(\exp(2 𝐬_θ(s))))
where 𝝁_θ(s) ∈ ℝ^D is the mean vector and 𝐬_θ(s) = \log 𝝈_θ(s) ∈ ℝ^D is the log-standard deviation vector.
Derive the complete analytical score function gradients with respect to 𝝁 and 𝐬:
    ∇_{𝝁} \log π_θ(a | s)   and   ∇_{𝐬} \log π_θ(a | s)
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Diagonal Covariance:** $\mathbf{\Sigma} = \operatorname{diag}(\sigma_1^2, \dots, \sigma_D^2)$, where $\sigma_d = \exp(s_d) > 0$.
2. **Log-Likelihood:** The probability density is:
   $$\pi_\theta(\mathbf{a} \mid s) = \prod_{d=1}^D \frac{1}{\sqrt{2\pi} \sigma_d(s)} \exp\left( -\frac{(a_d - \mu_d(s))^2}{2 \sigma_d(s)^2} \right)$$

**2. Underlying Intuition:**
Parameterizing the log-standard deviation $s = \log \sigma$ rather than $\sigma$ directly enforces the positivity constraint $\sigma = \exp(s) > 0$ without constrained optimization. In the log domain, taking derivatives with respect to $s$ yields clean quadratic terms $(a - \mu)^2 / \sigma^2 - 1$ that naturally push variance wider when errors are large and contract variance when predictions are accurate.

**3. End-to-End Algebraic Derivation:**

*Step 1: Compute the log-likelihood function.*
$$\log \pi_\theta(\mathbf{a} \mid s) = \sum_{d=1}^D \log \left( \frac{1}{\sqrt{2\pi} e^{s_d(s)}} \exp\left( -\frac{(a_d - \mu_d(s))^2}{2 e^{2 s_d(s)}} \right) \right)$$
$$= \sum_{d=1}^D \left[ -\frac{1}{2} \log(2\pi) - s_d(s) - \frac{1}{2} e^{-2 s_d(s)} (a_d - \mu_d(s))^2 \right]$$

*Step 2: Gradient with respect to mean $\mu_d$.*
Differentiating with respect to $\mu_d(s)$:
$$\frac{\partial \log \pi_\theta(\mathbf{a} \mid s)}{\partial \mu_d} = -\frac{1}{2} e^{-2 s_d(s)} \cdot 2(a_d - \mu_d(s)) \cdot (-1) = e^{-2 s_d(s)} (a_d - \mu_d(s)) = \frac{a_d - \mu_d(s)}{\sigma_d(s)^2}$$
In vector notation:
$$\nabla_{\boldsymbol{\mu}} \log \pi_\theta(\mathbf{a} \mid s) = \mathbf{\Sigma}^{-1} (\mathbf{a} - \boldsymbol{\mu}(s)) \quad \blacksquare$$

*Step 3: Gradient with respect to log-standard deviation $s_d$.*
Differentiating with respect to $s_d(s)$:
Notice $\frac{d}{ds_d} [-s_d] = -1$, and $\frac{d}{ds_d} [e^{-2 s_d}] = -2 e^{-2 s_d}$:
$$\frac{\partial \log \pi_\theta(\mathbf{a} \mid s)}{\partial s_d} = -1 - \frac{1}{2} \left( -2 e^{-2 s_d(s)} \right) (a_d - \mu_d(s))^2 = -1 + e^{-2 s_d(s)} (a_d - \mu_d(s))^2$$
$$= \frac{(a_d - \mu_d(s))^2}{\sigma_d(s)^2} - 1$$
In vector notation:
$$\nabla_{\mathbf{s}} \log \pi_\theta(\mathbf{a} \mid s) = \left( \frac{\mathbf{a} - \boldsymbol{\mu}(s)}{\boldsymbol{\sigma}(s)} \right)^{\odot 2} - \mathbf{1} \quad \blacksquare$$

*Step 4: Physical Interpretation of the variance gradient.*
Notice the normalized standardized residual $\epsilon_d \triangleq \frac{a_d - \mu_d}{\sigma_d} \sim \mathcal{N}(0, 1)$:
$$\frac{\partial \log \pi}{\partial s_d} = \epsilon_d^2 - 1$$
When an action delivers a positive advantage ($A > 0$):
- If $|\epsilon_d| > 1$ (the successful action was far out in the exploratory tail): $\epsilon_d^2 - 1 > 0$, so the gradient **increases $\sigma_d$** to encourage more exploration in that wider region!
- If $|\epsilon_d| < 1$ (the successful action was near the current mean): $\epsilon_d^2 - 1 < 0$, so the gradient **decreases $\sigma_d$**, tightening the policy around the mean to exploit! $\blacksquare$

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

### Illustration 2: Quantitative Variance Reduction — REINFORCE with Baseline vs. Vanilla REINFORCE
**Problem:**
Consider a single-state decision problem with two actions $\mathcal{A} = \{a_0, a_1\}$. The current policy parameter vector is $\boldsymbol{\theta} = [\theta_0, \theta_1]^\top$, giving action probabilities $\pi(a_0) = 0.40$ and $\pi(a_1) = 0.60$. The deterministic returns are $R(a_0) = 10.0$ and $R(a_1) = 2.0$.
1. Compute the expected policy gradient $\mathbb{E}[g_0]$ and the exact variance $\operatorname{Var}(g_0)$ of the first parameter coordinate for **Vanilla REINFORCE** (baseline $b = 0$).
2. Compute the exact variance $\operatorname{Var}(g_{0, b})$ when using the expected return baseline $b = V = \mathbb{E}[R]$.
3. Calculate the percentage variance reduction achieved by subtracting the baseline.

**Solution:**
**Step 1: Compute score function vectors.**
For softmax parametrization $\pi(a_i) = \frac{e^{\theta_i}}{e^{\theta_0} + e^{\theta_1}}$, the score function with respect to $\theta_0$ is:
$$\nabla_{\theta_0} \log \pi(a_i) = \mathbb{I}(i = 0) - \pi(a_0)$$
- If action $a_0$ is sampled: $\nabla_{\theta_0} \log \pi(a_0) = 1 - 0.40 = +0.60$.
- If action $a_1$ is sampled: $\nabla_{\theta_0} \log \pi(a_1) = 0 - 0.40 = -0.40$.

**Step 2: Vanilla REINFORCE ($b = 0$).**
The stochastic gradient sample is $g_0(a) = R(a) \nabla_{\theta_0} \log \pi(a)$:
- For $a_0$: $\nabla_{\theta_0} \log \pi(a_0) = +0.60 \implies g_0(a_0) = 10.0 \times (+0.60) = +6.00$.
- For $a_1$: $\nabla_{\theta_0} \log \pi(a_1) = -0.40 \implies g_0(a_1) = 2.0 \times (-0.40) = -0.80$.

Expected gradient:
$$\mathbb{E}[g_0] = \pi(a_0) g_0(a_0) + \pi(a_1) g_0(a_1) = 0.40(6.00) + 0.60(-0.80) = 2.40 - 0.48 = \mathbf{+1.9200}$$

Second moment:
$$\mathbb{E}[g_0^2] = 0.40(6.00^2) + 0.60((-0.80)^2) = 0.40(36.00) + 0.60(0.64) = 14.40 + 0.384 = \mathbf{14.7840}$$

Variance of Vanilla REINFORCE:
$$\operatorname{Var}(g_0) = \mathbb{E}[g_0^2] - (\mathbb{E}[g_0])^2 = 14.7840 - (1.9200)^2 = 14.7840 - 3.6864 = \mathbf{11.0976}$$

**Step 3: REINFORCE with Expected Return Baseline ($b = V$).**
Baseline value:
$$b = \mathbb{E}[R] = 0.40(10.0) + 0.60(2.0) = 4.00 + 1.20 = \mathbf{5.2000}$$

Centered advantage values:
- $A(a_0) = R(a_0) - b = 10.0 - 5.20 = \mathbf{+4.80}$
- $A(a_1) = R(a_1) - b = 2.0 - 5.20 = \mathbf{-3.20}$

Stochastic gradient sample $g_{0, b}(a) = A(a) \nabla_{\theta_0} \log \pi(a)$:
- $g_{0, b}(a_0) = 4.80 \times (+0.60) = \mathbf{+2.8800}$
- $g_{0, b}(a_1) = -3.20 \times (-0.40) = \mathbf{+1.2800}$

Expected gradient with baseline:
$$\mathbb{E}[g_{0, b}] = 0.40(2.8800) + 0.60(1.2800) = 1.1520 + 0.7680 = \mathbf{+1.9200}$$
*(Completely unbiased: identical to $\mathbb{E}[g_0]$!)*

Second moment with baseline:
$$\mathbb{E}[g_{0, b}^2] = 0.40(2.8800^2) + 0.60(1.2800^2) = 0.40(8.2944) + 0.60(1.6384) = 3.31776 + 0.98304 = \mathbf{4.3008}$$

Variance with baseline:
$$\operatorname{Var}(g_{0, b}) = \mathbb{E}[g_{0, b}^2] - (\mathbb{E}[g_{0, b}])^2 = 4.3008 - 3.6864 = \mathbf{0.6144}$$

**Step 4: Quantify Variance Reduction.**
$$\text{Reduction Ratio} = \frac{\operatorname{Var}(g_{0, b})}{\operatorname{Var}(g_0)} = \frac{0.6144}{11.0976} \approx \mathbf{0.05536} \quad (5.54\% \text{ of original variance})$$
$$\text{Percentage Variance Reduction} = \left(1 - \frac{0.6144}{11.0976}\right) \times 100\% = \mathbf{94.46\%}$$
Subtracting the baseline eliminates **$94.46\%$** of the gradient estimation variance in a single step without introducing any bias whatsoever. $\blacksquare$

---

### Illustration 3: Complete REINFORCE Trajectory Update with Discounted Returns and Learned Critic Baseline
**Problem:**
An agent generates an episode of length $T = 3$ with discount factor $\gamma = 0.90$:
- Transition $t=0$: state $s_0$, action $a_0$, reward $r_1 = 1.0$, next state $s_1$. Critic estimate $V_\phi(s_0) = 8.00$. Score $\nabla_\theta \log \pi_\theta(a_0 \mid s_0) = [+0.50, -0.50]^\top$.
- Transition $t=1$: state $s_1$, action $a_1$, reward $r_2 = 2.0$, next state $s_2$. Critic estimate $V_\phi(s_1) = 9.00$. Score $\nabla_\theta \log \pi_\theta(a_1 \mid s_1) = [-0.30, +0.30]^\top$.
- Transition $t=2$: state $s_2$, action $a_0$, reward $r_3 = 10.0$, terminal state $s_3$. Critic estimate $V_\phi(s_2) = 7.00$. Score $\nabla_\theta \log \pi_\theta(a_0 \mid s_2) = [+0.70, -0.70]^\top$.

Compute:
1. The exact discounted returns $G_2, G_1, G_0$ via backward recursion.
2. The advantage estimates $\hat{A}_0, \hat{A}_1, \hat{A}_2$.
3. The total policy gradient vector $\mathbf{g}_\theta = \sum_{t=0}^2 \gamma^t \hat{A}_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$.
4. The mean squared error loss gradient for the critic parameters if $V_\phi(s) = \boldsymbol{\phi}^\top \mathbf{x}(s)$ with $\mathbf{x}(s_0) = [1, 0]^\top, \mathbf{x}(s_1) = [0, 1]^\top, \mathbf{x}(s_2) = [1, 1]^\top$.

**Solution:**
**Step 1: Backward recursion for returns.**
$$G_2 = r_3 = \mathbf{10.0000}$$
$$G_1 = r_2 + \gamma G_2 = 2.0 + 0.90 \times 10.0 = 2.0 + 9.0 = \mathbf{11.0000}$$
$$G_0 = r_1 + \gamma G_1 = 1.0 + 0.90 \times 11.0 = 1.0 + 9.90 = \mathbf{10.9000}$$

**Step 2: Compute advantage estimates $\hat{A}_t = G_t - V_\phi(s_t)$.**
$$\hat{A}_0 = G_0 - V_\phi(s_0) = 10.9000 - 8.0000 = \mathbf{+2.9000}$$
$$\hat{A}_1 = G_1 - V_\phi(s_1) = 11.0000 - 9.0000 = \mathbf{+2.0000}$$
$$\hat{A}_2 = G_2 - V_\phi(s_2) = 10.0000 - 7.0000 = \mathbf{+3.0000}$$

**Step 3: Accumulate policy gradient vector.**
Using the discounted formulation $\mathbf{g}_\theta = \sum_{t=0}^2 \gamma^t \hat{A}_t \nabla_\theta \log \pi(a_t \mid s_t)$:
- For $t=0$ ($\gamma^0 = 1.0$):
  $$\mathbf{g}_0 = 1.0 \times 2.9000 \times \begin{bmatrix} +0.50 \\ -0.50 \end{bmatrix} = \begin{bmatrix} +1.4500 \\ -1.4500 \end{bmatrix}$$
- For $t=1$ ($\gamma^1 = 0.90$):
  $$\mathbf{g}_1 = 0.90 \times 2.0000 \times \begin{bmatrix} -0.30 \\ +0.30 \end{bmatrix} = 1.8000 \times \begin{bmatrix} -0.30 \\ +0.30 \end{bmatrix} = \begin{bmatrix} -0.5400 \\ +0.5400 \end{bmatrix}$$
- For $t=2$ ($\gamma^2 = 0.81$):
  $$\mathbf{g}_2 = 0.81 \times 3.0000 \times \begin{bmatrix} +0.70 \\ -0.70 \end{bmatrix} = 2.4300 \times \begin{bmatrix} +0.70 \\ -0.70 \end{bmatrix} = \begin{bmatrix} +1.7010 \\ -1.7010 \end{bmatrix}$$

Summing across all steps:
$$\mathbf{g}_\theta = \mathbf{g}_0 + \mathbf{g}_1 + \mathbf{g}_2 = \begin{bmatrix} 1.4500 - 0.5400 + 1.7010 \\ -1.4500 + 0.5400 - 1.7010 \end{bmatrix} = \mathbf{\begin{bmatrix} +2.6110 \\ -2.6110 \end{bmatrix}}$$

**Step 4: Critic parameter gradients.**
The critic objective per transition is $\mathcal{L}(\boldsymbol{\phi}) = \frac{1}{2} (G_t - V_\phi(s_t))^2$.
Gradient with respect to $\boldsymbol{\phi}$:
$$\nabla_{\boldsymbol{\phi}} \mathcal{L}_t = -(G_t - V_\phi(s_t)) \nabla_{\boldsymbol{\phi}} V_\phi(s_t) = -\hat{A}_t \mathbf{x}(s_t)$$
- $t=0$: $-\hat{A}_0 \mathbf{x}(s_0) = -2.9000 [1, 0]^\top = [-2.9000, 0.0000]^\top$
- $t=1$: $-\hat{A}_1 \mathbf{x}(s_1) = -2.0000 [0, 1]^\top = [0.0000, -2.0000]^\top$
- $t=2$: $-\hat{A}_2 \mathbf{x}(s_2) = -3.0000 [1, 1]^\top = [-3.0000, -3.0000]^\top$

Total critic gradient:
$$\nabla_{\boldsymbol{\phi}} \mathcal{L} = \sum_{t=0}^2 \nabla_{\boldsymbol{\phi}} \mathcal{L}_t = \begin{bmatrix} -2.9000 + 0 - 3.0000 \\ 0 - 2.0000 - 3.0000 \end{bmatrix} = \mathbf{\begin{bmatrix} -5.9000 \\ -5.0000 \end{bmatrix}} \quad \blacksquare$$

---

### Illustration 4: Analytical Variance-Minimizing Baseline vs. State-Value Baseline
**Problem:**
In a contextual bandit with three discrete actions $\mathcal{A} = \{a_0, a_1, a_2\}$, the current policy probabilities are $\boldsymbol{\pi} = [0.20, 0.50, 0.30]^\top$, and the deterministic rewards are $\mathbf{R} = [10.0, 4.0, 2.0]^\top$.
1. Compute the classical state-value baseline $V = \sum_a \pi(a) R(a)$.
2. Compute the exact variance-minimizing baseline $b^* = \frac{\mathbb{E}[\|\nabla_{\boldsymbol{\theta}} \log \pi(A)\|^2 R(A)]}{\mathbb{E}[\|\nabla_{\boldsymbol{\theta}} \log \pi(A)\|^2]}$ derived in Section 2.7.
3. Contrast $b^*$ and $V$, explaining why they differ.

**Solution:**
**Step 1: Compute state-value baseline $V$.**
$$V = \pi(a_0) R(a_0) + \pi(a_1) R(a_1) + \pi(a_2) R(a_2) = 0.20(10.0) + 0.50(4.0) + 0.30(2.0)$$
$$V = 2.00 + 2.00 + 0.60 = \mathbf{4.6000}$$

**Step 2: Compute score function squared $L_2$ norms.**
For a 3-class softmax policy parametrized by $\boldsymbol{\theta} \in \mathbb{R}^3$, the score vector for action $a$ is $\nabla_{\boldsymbol{\theta}} \log \pi(a) = \mathbf{e}_a - \boldsymbol{\pi}$.
The squared Euclidean norm is:
$$\|\nabla_{\boldsymbol{\theta}} \log \pi(a)\|^2 = (1 - \pi_a)^2 + \sum_{k \ne a} \pi_k^2$$
- For $a_0$ ($\pi_0 = 0.20$):
  $$\|\mathbf{g}_0\|^2 = (1 - 0.20)^2 + 0.50^2 + 0.30^2 = 0.64 + 0.25 + 0.09 = \mathbf{0.9800}$$
- For $a_1$ ($\pi_1 = 0.50$):
  $$\|\mathbf{g}_1\|^2 = 0.20^2 + (1 - 0.50)^2 + 0.30^2 = 0.04 + 0.25 + 0.09 = \mathbf{0.3800}$$
- For $a_2$ ($\pi_2 = 0.30$):
  $$\|\mathbf{g}_2\|^2 = 0.20^2 + 0.50^2 + (1 - 0.30)^2 = 0.04 + 0.25 + 0.49 = \mathbf{0.7800}$$

**Step 3: Compute numerator and denominator for $b^*$.**
Numerator:
$$\mathbb{E}[\|\nabla_{\boldsymbol{\theta}} \log \pi(A)\|^2 R(A)] = \sum_{a} \pi(a) \|\mathbf{g}_a\|^2 R(a)$$
$$= 0.20(0.9800)(10.0) + 0.50(0.3800)(4.0) + 0.30(0.7800)(2.0)$$
$$= 0.20 \times 9.80 + 0.50 \times 1.52 + 0.30 \times 1.56 = 1.9600 + 0.7600 + 0.4680 = \mathbf{3.1880}$$

Denominator:
$$\mathbb{E}[\|\nabla_{\boldsymbol{\theta}} \log \pi(A)\|^2] = \sum_{a} \pi(a) \|\mathbf{g}_a\|^2$$
$$= 0.20(0.9800) + 0.50(0.3800) + 0.30(0.7800)$$
$$= 0.1960 + 0.1900 + 0.2340 = \mathbf{0.6200}$$

Optimal baseline:
$$b^* = \frac{3.1880}{0.6200} \approx \mathbf{5.1419}$$

**Step 4: Comparison and insight.**
$$b^* = 5.1419 > V = 4.6000$$
The optimal baseline $b^*$ places heavier weight on actions that produce larger score vector norms $\|\nabla_{\boldsymbol{\theta}} \log \pi(a)\|^2$. Action $a_0$ has probability $0.20$ (low), giving it a large score norm ($0.98$), and it yields the highest reward ($10.0$). Consequently, the variance-minimizing baseline shifts upward toward $10.0$ ($5.1419 > 4.6000$) to suppress the high-norm gradient variance of action $a_0$. $\blacksquare$

---

### Illustration 5: Group Relative Policy Optimization (GRPO / DeepSeek-R1) Multi-Sample Baseline
**Problem:**
In the DeepSeek-R1 style GRPO alignment framework, the LLM policy generates a group of $G = 4$ independent solution completions $\{o_1, o_2, o_3, o_4\}$ for a given mathematical reasoning prompt $x$. A deterministic rule-based verification script evaluates each completion, returning binary correctness rewards:
$$\mathbf{r} = [r_1, r_2, r_3, r_4]^\top = [1.0, 0.0, 1.0, 0.0]^\top$$
1. Calculate the group empirical mean reward $\bar{r}$ and standard deviation $\sigma_r$.
2. Compute the normalized group relative advantage $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$ (with numerical stabilizer $\epsilon \to 0$).
3. Suppose the log-probability gradient of the correct completion $o_1$ with respect to a critical reasoning token is $\nabla_\theta \log \pi_\theta(w_{\text{step}} \mid x) = +0.85$, while for incorrect completion $o_2$ it is $\nabla_\theta \log \pi_\theta(w_{\text{err}} \mid x) = +0.60$. Compute the resulting policy gradient updates for both tokens under GRPO.

**Solution:**
**Step 1: Compute group mean and standard deviation.**
$$\bar{r} = \frac{1}{G} \sum_{i=1}^4 r_i = \frac{1.0 + 0.0 + 1.0 + 0.0}{4} = \frac{2.0}{4} = \mathbf{0.5000}$$

Variance across the group:
$$\sigma_r^2 = \frac{1}{G} \sum_{i=1}^4 (r_i - \bar{r})^2 = \frac{(1.0-0.5)^2 + (0.0-0.5)^2 + (1.0-0.5)^2 + (0.0-0.5)^2}{4}$$
$$\sigma_r^2 = \frac{0.25 + 0.25 + 0.25 + 0.25}{4} = 0.2500 \implies \sigma_r = \sqrt{0.2500} = \mathbf{0.5000}$$

**Step 2: Compute normalized group relative advantage $\hat{A}_i$.**
$$\hat{A}_1 = \frac{1.0 - 0.50}{0.50} = \mathbf{+1.0000}$$
$$\hat{A}_2 = \frac{0.0 - 0.50}{0.50} = \mathbf{-1.0000}$$
$$\hat{A}_3 = \frac{1.0 - 0.50}{0.50} = \mathbf{+1.0000}$$
$$\hat{A}_4 = \frac{0.0 - 0.50}{0.50} = \mathbf{-1.0000}$$

Notice:
$$\sum_{i=1}^4 \hat{A}_i = +1.0 - 1.0 + 1.0 - 1.0 = 0$$
The group-relative baseline naturally enforces zero-sum relative feedback without training a separate critic network!

**Step 3: Compute token-level policy gradients.**
For the correct token in completion $o_1$:
$$\mathbf{g}_1 = \hat{A}_1 \nabla_\theta \log \pi_\theta(w_{\text{step}} \mid x) = (+1.0000) \times (+0.85) = \mathbf{+0.8500}$$
*(Reinforces this reasoning token!)*

For the faulty token in completion $o_2$:
$$\mathbf{g}_2 = \hat{A}_2 \nabla_\theta \log \pi_\theta(w_{\text{err}} \mid x) = (-1.0000) \times (+0.60) = \mathbf{-0.6000}$$
*(Actively penalizes and suppresses this hallucinated/erroneous token!)* $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. The Mathematical Foundation of RLHF (OpenAI InstructGPT, 2022)
REINFORCE is the direct ancestor of modern LLM alignment. The RLHF policy gradient update is:
$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \cdot \left( R(x, y) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right) \right]$$
This is exactly REINFORCE with a KL-penalized reward signal. **InstructGPT** (Ouyang et al., NeurIPS 2022) showed that a 1.3B RLHF-aligned model is preferred over a raw 175B GPT-3 model in human evaluations — a 100× parameter advantage erased by alignment.

### 2. GRPO: Critic-Free REINFORCE for Frontier Reasoning (DeepSeek-AI, 2025)
**Group Relative Policy Optimization (GRPO)** eliminates the expensive critic network in PPO by using a group-normalized REINFORCE baseline:
$$A_i = \frac{r_i - \operatorname{mean}(\{r_1, \dots, r_G\})}{\operatorname{std}(\{r_1, \dots, r_G\})}$$
Sampling a group of $G = 8$ to $G = 64$ rollouts per prompt and computing group-relative advantages. DeepSeek-R1 trained on GRPO achieves 97.3% on MATH-500 and 79.8% pass@1 on AIME 2024 — matching OpenAI o1 without a value network.

### 3. Classic Production: Robotics & Game Playing
- **OpenAI Dactyl (2019):** Used distributed REINFORCE variants to train a five-fingered Shadow Hand robot to solve Rubik's Cube with one hand, collecting 13,000 years of simulated experience via domain randomization.
- **AlphaGo (Silver et al., Nature 2016):** Policy gradient fine-tuning with REINFORCE over Go rollouts was the critical step that lifted AlphaGo beyond its supervised initialization, creating the first Go AI to defeat a 9-dan professional player.
- **Multi-Agent Cooperative AI:** REINFORCE forms the policy update backbone in MAPPO (Multi-Agent PPO), used to coordinate teams of up to 100 agents in cooperative tasks (warehouse logistics, multi-drone swarms).

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
