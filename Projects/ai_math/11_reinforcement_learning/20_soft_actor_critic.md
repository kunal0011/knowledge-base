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

### 2.7 First-Principles Mathematical Derivations

#### Derivation 11.20.1: Soft Policy Iteration Convergence Theorem

```
====================================================================================================
DERIVATION 11.20.1: Convergence of Soft Policy Iteration (Contraction and Monotonic Improvement)
====================================================================================================
Problem Statement:
Prove that in an entropy-augmented Markov Decision Process with temperature alpha > 0 and discount
gamma in [0, 1):
1. Soft Policy Evaluation: The modified soft Bellman operator T^pi is a strict gamma-contraction in
   the L_infinity norm on the Banach space B(S x A), guaranteeing geometric convergence to a unique
   fixed-point soft action-value function Q^pi.
2. Soft Policy Improvement: The updated policy pi_new defined by:
       pi_new(· | s) = argmin_{pi' in Pi} D_KL( pi'(· | s) || exp( Q^{pi_old}(s, ·) / alpha ) / Z^{pi_old}(s) )
   strictly satisfies Q^{pi_new}(s, a) >= Q^{pi_old}(s, a) for all (s, a) in S x A, with equality
   holding if and only if pi_old is an optimal soft policy pi*.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
Consider an infinite-horizon entropy-augmented Markov Decision Process defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma, \alpha)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ is the Markovian transition probability kernel, $R: \mathcal{S} \times \mathcal{A} \to [R_{\min}, R_{\max}]$ is the bounded reward function, $\gamma \in [0, 1)$ is the discount factor, and $\alpha > 0$ is the constant entropy temperature.

The Maximum Entropy reinforcement learning objective seeks a policy $\pi \in \Pi$ maximizing expected discounted reward plus discounted policy entropy:
$$J(\pi) \triangleq \sum_{t=0}^\infty \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ \gamma^t \left( R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right) \right]$$
where $\mathcal{H}(\pi(\cdot \mid s)) \triangleq \mathbb{E}_{a \sim \pi(\cdot \mid s)} [-\log \pi(a \mid s)]$.

The soft state-value function $V^\pi: \mathcal{S} \to \mathbb{R}$ and soft action-value function $Q^\pi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ are defined recursively by:
$$V^\pi(s) \triangleq \mathbb{E}_{a \sim \pi(\cdot \mid s)} \left[ Q^\pi(s, a) - \alpha \log \pi(a \mid s) \right]$$
$$Q^\pi(s, a) \triangleq R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ V^\pi(s') \right]$$

The Soft Policy Evaluation operator $\mathcal{T}^\pi: \mathcal{B}(\mathcal{S} \times \mathcal{A}) \to \mathcal{B}(\mathcal{S} \times \mathcal{A})$ maps any bounded function $Q$ to:
$$\mathcal{T}^\pi Q(s, a) \triangleq R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ \mathbb{E}_{a' \sim \pi(\cdot \mid s')} \left[ Q(s', a') - \alpha \log \pi(a' \mid s') \right] \right]$$

The Soft Policy Improvement operator maps an existing policy $\pi_{\text{old}}$ to $\pi_{\text{new}}$ via:
$$\pi_{\text{new}}(\cdot \mid s) \triangleq \arg\min_{\pi' \in \Pi} D_{\text{KL}} \left( \pi'(\cdot \mid s) \;\Big\|\; \frac{\exp\left( \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, \cdot) \right)}{Z^{\pi_{\text{old}}}(s)} \right)$$
where $Z^{\pi_{\text{old}}}(s) \triangleq \int_{\mathcal{A}} \exp\left( \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, a) \right) da$ is the partition function.

Our mathematical goals are:
1. Prove that $\mathcal{T}^\pi$ is a strict $\gamma$-contraction in the supremum norm $\|\cdot\|_\infty$, establishing the existence and uniqueness of the fixed point $Q^\pi = \mathcal{T}^\pi Q^\pi$ and the linear convergence of iterates $(\mathcal{T}^\pi)^k Q_0 \to Q^\pi$.
2. Prove that the updated policy $\pi_{\text{new}}$ monotonically improves the soft action-value function: $Q^{\pi_{\text{new}}}(s, a) \ge Q^{\pi_{\text{old}}}(s, a)$ for all $(s, a) \in \mathcal{S} \times \mathcal{A}$.

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Bounded Rewards:** The reward function is uniformly bounded: $\sup_{(s, a) \in \mathcal{S} \times \mathcal{A}} |R(s, a)| \le M_R < \infty$.
2. **Bounded Entropy / Compact Action Domain:** The action space $\mathcal{A}$ is compact, or policies $\pi \in \Pi$ have differential entropy lower-bounded: $\mathcal{H}(\pi(\cdot \mid s)) \ge -M_{\mathcal{H}} > -\infty$ for all $s \in \mathcal{S}$. Furthermore, the partition function $Z^\pi(s) = \int_{\mathcal{A}} \exp\left(\frac{1}{\alpha} Q^\pi(s, a)\right) da$ satisfies $0 < Z^\pi(s) < \infty$ for all $s \in \mathcal{S}$.
3. **Strict Discounting:** The discount factor satisfies $\gamma \in [0, 1)$.
4. **Positive Temperature:** The temperature parameter satisfies $\alpha \in (0, \infty)$.
5. **Banach Space Completeness:** $\mathcal{B}(\mathcal{S} \times \mathcal{A})$ denotes the Banach space of bounded measurable real-valued functions on $\mathcal{S} \times \mathcal{A}$ equipped with the supremum norm $\|Q\|_\infty \triangleq \sup_{(s, a) \in \mathcal{S} \times \mathcal{A}} |Q(s, a)|$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
- **Contraction Mapping Intuition:** Notice that for a fixed policy $\pi$, the entropy term $h^\pi(s') \triangleq -\alpha \mathbb{E}_{a' \sim \pi} [\log \pi(a' \mid s')]$ depends exclusively on the state $s'$ and the stationary policy $\pi$; it is completely independent of the candidate value function $Q$. Consequently, the soft Bellman operator $\mathcal{T}^\pi$ is simply the classical Bellman expectation operator associated with a modified, state-augmented reward function $\tilde{R}^\pi(s, a, s') \triangleq R(s, a) + \gamma h^\pi(s')$. Because an additive constant offset does not alter distance under subtraction, the contractive modulus remains identical to the discount factor $\gamma < 1$.
- **Policy Improvement Intuition:** In classical RL, greedy improvement maximizes expected action-value by placing all probability mass on $\arg\max_a Q(s, a)$. In maximum-entropy RL, the agent balances reward against entropy. The target energy-based distribution $p^*(a \mid s) \propto \exp(Q^{\pi_{\text{old}}}(s, a)/\alpha)$ represents the exact thermodynamic Boltzmann equilibrium. Minimizing the Kullback-Leibler divergence $D_{\text{KL}}(\pi_{\text{new}} \| p^*)$ is equivalent to maximizing the expected free energy $\mathbb{E}_{\pi_{\text{new}}}[Q^{\pi_{\text{old}}} - \alpha \log \pi_{\text{new}}]$. Because $D_{\text{KL}} \ge 0$ with equality attained when $\pi_{\text{new}} = p^*$, the new policy is mathematically guaranteed to achieve an expected soft value at least as high as the old policy: $V^{\pi_{\text{new}}, 1}(s) \ge V^{\pi_{\text{old}}}(s)$. Telescoping this inequality forward across infinite future time steps compounds this advantage monotonically at every state-action pair.

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Phase 1: Proof that $\mathcal{T}^\pi$ is a strict $\gamma$-contraction in $L_\infty$.*

Let $Q_1, Q_2 \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$ be any two arbitrary bounded action-value functions.
Fix an arbitrary state-action pair $(s, a) \in \mathcal{S} \times \mathcal{A}$. Apply the operator $\mathcal{T}^\pi$ to both functions and evaluate their algebraic difference:
$$\mathcal{T}^\pi Q_1(s, a) - \mathcal{T}^\pi Q_2(s, a) = \left( R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}, a' \sim \pi} \left[ Q_1(s', a') - \alpha \log \pi(a' \mid s') \right] \right) - \left( R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}, a' \sim \pi} \left[ Q_2(s', a') - \alpha \log \pi(a' \mid s') \right] \right)$$

Distributing the minus sign across terms:
$$= R(s, a) - R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}, a' \sim \pi} \left[ Q_1(s', a') - Q_2(s', a') - \alpha \log \pi(a' \mid s') + \alpha \log \pi(a' \mid s') \right]$$

Observe that the immediate reward $R(s, a)$ cancels out identically ($R - R = 0$), and the policy entropy term $-\alpha \log \pi(a' \mid s')$ cancels out identically:
$$= \gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ \mathbb{E}_{a' \sim \pi(\cdot \mid s')} \left[ Q_1(s', a') - Q_2(s', a') \right] \right]$$

Taking the absolute value of both sides:
$$\left| \mathcal{T}^\pi Q_1(s, a) - \mathcal{T}^\pi Q_2(s, a) \right| = \gamma \left| \int_{\mathcal{S}} \mathcal{P}(s' \mid s, a) \int_{\mathcal{A}} \pi(a' \mid s') \left( Q_1(s', a') - Q_2(s', a') \right) da' \, ds' \right|$$

By Jensen's inequality (or the integral triangle inequality $|\int f| \le \int |f|$):
$$\left| \mathcal{T}^\pi Q_1(s, a) - \mathcal{T}^\pi Q_2(s, a) \right| \le \gamma \int_{\mathcal{S}} \mathcal{P}(s' \mid s, a) \int_{\mathcal{A}} \pi(a' \mid s') \left| Q_1(s', a') - Q_2(s', a') \right| da' \, ds'$$

By definition of the supremum norm, for all $(s', a') \in \mathcal{S} \times \mathcal{A}$:
$$\left| Q_1(s', a') - Q_2(s', a') \right| \le \sup_{(\tilde{s}, \tilde{a}) \in \mathcal{S} \times \mathcal{A}} \left| Q_1(\tilde{s}, \tilde{a}) - Q_2(\tilde{s}, \tilde{a}) \right| = \|Q_1 - Q_2\|_\infty$$

Substituting this uniform upper bound into the integral:
$$\le \gamma \int_{\mathcal{S}} \mathcal{P}(s' \mid s, a) \int_{\mathcal{A}} \pi(a' \mid s') \|Q_1 - Q_2\|_\infty \, da' \, ds'$$

Pulling the constant $\|Q_1 - Q_2\|_\infty$ outside the integrals:
$$= \gamma \|Q_1 - Q_2\|_\infty \int_{\mathcal{S}} \mathcal{P}(s' \mid s, a) \left( \int_{\mathcal{A}} \pi(a' \mid s') da' \right) ds'$$

Because $\pi(\cdot \mid s')$ is a valid probability density on $\mathcal{A}$, $\int_{\mathcal{A}} \pi(a' \mid s') da' = 1$.
Because $\mathcal{P}(\cdot \mid s, a)$ is a valid transition probability density on $\mathcal{S}$, $\int_{\mathcal{S}} \mathcal{P}(s' \mid s, a) ds' = 1$.
Therefore:
$$\left| \mathcal{T}^\pi Q_1(s, a) - \mathcal{T}^\pi Q_2(s, a) \right| \le \gamma \|Q_1 - Q_2\|_\infty \cdot 1 \cdot 1 = \gamma \|Q_1 - Q_2\|_\infty$$

Since this inequality holds uniformly for every choice of $(s, a) \in \mathcal{S} \times \mathcal{A}$, taking the supremum over all $(s, a)$ yields:
$$\|\mathcal{T}^\pi Q_1 - \mathcal{T}^\pi Q_2\|_\infty = \sup_{(s, a) \in \mathcal{S} \times \mathcal{A}} \left| \mathcal{T}^\pi Q_1(s, a) - \mathcal{T}^\pi Q_2(s, a) \right| \le \gamma \|Q_1 - Q_2\|_\infty$$

Because $\gamma \in [0, 1)$, $\mathcal{T}^\pi$ is a strict contraction mapping on the complete metric space $(\mathcal{B}(\mathcal{S} \times \mathcal{A}), \|\cdot\|_\infty)$.
By the **Banach Fixed-Point Theorem**:
1. There exists a unique fixed point $Q^\pi \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$ satisfying $\mathcal{T}^\pi Q^\pi = Q^\pi$.
2. For any arbitrary initial $Q_0 \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$, the sequence of iterates $Q_{k+1} = \mathcal{T}^\pi Q_k$ converges linearly to $Q^\pi$:
   $$\|Q_k - Q^\pi\|_\infty \le \frac{\gamma^k}{1 - \gamma} \|\mathcal{T}^\pi Q_0 - Q_0\|_\infty \xrightarrow{k \to \infty} 0$$

*Phase 2: Proof of the Soft Policy Improvement Theorem.*

Let $\pi_{\text{old}} \in \Pi$ be the current policy with fixed-point soft value functions $V^{\pi_{\text{old}}}$ and $Q^{\pi_{\text{old}}}$.
Define the energy-based Boltzmann target distribution:
$$p^*(a \mid s) \triangleq \frac{\exp\left( \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, a) \right)}{Z^{\pi_{\text{old}}}(s)}, \quad \text{where} \quad Z^{\pi_{\text{old}}}(s) = \int_{\mathcal{A}} \exp\left( \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, \tilde{a}) \right) d\tilde{a}$$

Let $\pi_{\text{new}}$ be the updated policy obtained by minimizing the Kullback-Leibler divergence:
$$\pi_{\text{new}}(\cdot \mid s) = \arg\min_{\pi' \in \Pi} D_{\text{KL}} \left( \pi'(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right)$$

Let us expand $D_{\text{KL}}(\pi'(\cdot \mid s) \| p^*(\cdot \mid s))$ for an arbitrary policy $\pi'$:
$$D_{\text{KL}}\left( \pi'(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right) \triangleq \mathbb{E}_{a \sim \pi'(\cdot \mid s)} \left[ \log \left( \frac{\pi'(a \mid s)}{p^*(a \mid s)} \right) \right]$$
$$= \mathbb{E}_{a \sim \pi'(\cdot \mid s)} \left[ \log \pi'(a \mid s) - \log p^*(a \mid s) \right]$$

Substitute the definition of $p^*(a \mid s)$:
$$\log p^*(a \mid s) = \log\left( \frac{\exp\left( \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, a) \right)}{Z^{\pi_{\text{old}}}(s)} \right) = \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, a) - \log Z^{\pi_{\text{old}}}(s)$$

Substituting this into the expectation:
$$D_{\text{KL}}\left( \pi'(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right) = \mathbb{E}_{a \sim \pi'(\cdot \mid s)} \left[ \log \pi'(a \mid s) - \frac{1}{\alpha} Q^{\pi_{\text{old}}}(s, a) + \log Z^{\pi_{\text{old}}}(s) \right]$$

Multiplying both sides by the positive scalar $\alpha > 0$:
$$\alpha D_{\text{KL}}\left( \pi'(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right) = \mathbb{E}_{a \sim \pi'(\cdot \mid s)} \left[ \alpha \log \pi'(a \mid s) - Q^{\pi_{\text{old}}}(s, a) \right] + \alpha \log Z^{\pi_{\text{old}}}(s) \int_{\mathcal{A}} \pi'(a \mid s) da$$
$$= - \mathbb{E}_{a \sim \pi'(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi'(a \mid s) \right] + \alpha \log Z^{\pi_{\text{old}}}(s)$$

Notice that the partition term $\alpha \log Z^{\pi_{\text{old}}}(s)$ depends solely on $Q^{\pi_{\text{old}}}$ and $s$; it does NOT depend on $\pi'$.
Because $\pi_{\text{new}}$ is defined as the minimizer of $D_{\text{KL}}(\pi'(\cdot \mid s) \| p^*(\cdot \mid s))$ over all $\pi' \in \Pi$, it must achieve a KL divergence less than or equal to that of the old policy $\pi_{\text{old}}$:
$$D_{\text{KL}}\left( \pi_{\text{new}}(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right) \le D_{\text{KL}}\left( \pi_{\text{old}}(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right)$$

Multiplying by $\alpha > 0$:
$$\alpha D_{\text{KL}}\left( \pi_{\text{new}}(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right) \le \alpha D_{\text{KL}}\left( \pi_{\text{old}}(\cdot \mid s) \;\Big\|\; p^*(\cdot \mid s) \right)$$

Substituting our expanded identity for both sides:
$$- \mathbb{E}_{a \sim \pi_{\text{new}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{new}}(a \mid s) \right] + \alpha \log Z^{\pi_{\text{old}}}(s) \le - \mathbb{E}_{a \sim \pi_{\text{old}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{old}}(a \mid s) \right] + \alpha \log Z^{\pi_{\text{old}}}(s)$$

Subtracting $\alpha \log Z^{\pi_{\text{old}}}(s)$ from both sides:
$$- \mathbb{E}_{a \sim \pi_{\text{new}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{new}}(a \mid s) \right] \le - \mathbb{E}_{a \sim \pi_{\text{old}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{old}}(a \mid s) \right]$$

Multiplying by $-1$ (which strictly reverses the direction of the inequality):
$$\mathbb{E}_{a \sim \pi_{\text{new}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{new}}(a \mid s) \right] \ge \mathbb{E}_{a \sim \pi_{\text{old}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{old}}(a \mid s) \right]$$

By definition of the soft state-value function $V^{\pi_{\text{old}}}(s)$, the right-hand side is identically $V^{\pi_{\text{old}}}(s)$:
$$\mathbb{E}_{a \sim \pi_{\text{new}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{new}}(a \mid s) \right] \ge V^{\pi_{\text{old}}}(s) \quad \forall s \in \mathcal{S}$$

Now, define $V^{\pi_{\text{new}}, 1}(s) \triangleq \mathbb{E}_{a \sim \pi_{\text{new}}(\cdot \mid s)} \left[ Q^{\pi_{\text{old}}}(s, a) - \alpha \log \pi_{\text{new}}(a \mid s) \right]$. We have established:
$$V^{\pi_{\text{new}}, 1}(s) \ge V^{\pi_{\text{old}}}(s) \quad \forall s \in \mathcal{S}$$

*Phase 3: Telescoping Monotonic Propagation.*

Now consider the action-value function under one application of the Bellman operator for policy $\pi_{\text{new}}$ applied to $Q^{\pi_{\text{old}}}$:
$$\mathcal{T}^{\pi_{\text{new}}} Q^{\pi_{\text{old}}}(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ \mathbb{E}_{a' \sim \pi_{\text{new}}(\cdot \mid s')} \left[ Q^{\pi_{\text{old}}}(s', a') - \alpha \log \pi_{\text{new}}(a' \mid s') \right] \right]$$
$$= R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ V^{\pi_{\text{new}}, 1}(s') \right]$$

Applying our inequality $V^{\pi_{\text{new}}, 1}(s') \ge V^{\pi_{\text{old}}}(s')$:
$$\ge R(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ V^{\pi_{\text{old}}}(s') \right]$$

By the Bellman equation for the old policy, $R(s, a) + \gamma \mathbb{E}_{s'} [V^{\pi_{\text{old}}}(s')] = Q^{\pi_{\text{old}}}(s, a)$. Therefore:
$$\mathcal{T}^{\pi_{\text{new}}} Q^{\pi_{\text{old}}}(s, a) \ge Q^{\pi_{\text{old}}}(s, a) \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A}$$

Next, we establish monotonicity of $\mathcal{T}^{\pi_{\text{new}}}$. Suppose $Q_A(s, a) \ge Q_B(s, a)$ for all $(s, a)$. Then:
$$\mathcal{T}^{\pi_{\text{new}}} Q_A(s, a) - \mathcal{T}^{\pi_{\text{new}}} Q_B(s, a) = \gamma \mathbb{E}_{s' \sim \mathcal{P}, a' \sim \pi_{\text{new}}} \left[ Q_A(s', a') - Q_B(s', a') \right] \ge 0$$
Thus, $\mathcal{T}^{\pi_{\text{new}}} Q_A \ge \mathcal{T}^{\pi_{\text{new}}} Q_B$.

Applying $\mathcal{T}^{\pi_{\text{new}}}$ repeatedly $k$ times to the inequality $\mathcal{T}^{\pi_{\text{new}}} Q^{\pi_{\text{old}}} \ge Q^{\pi_{\text{old}}}$:
$$(\mathcal{T}^{\pi_{\text{new}}})^k Q^{\pi_{\text{old}}}(s, a) \ge (\mathcal{T}^{\pi_{\text{new}}})^{k-1} Q^{\pi_{\text{old}}}(s, a) \ge \dots \ge \mathcal{T}^{\pi_{\text{new}}} Q^{\pi_{\text{old}}}(s, a) \ge Q^{\pi_{\text{old}}}(s, a)$$

Taking the limit as $k \to \infty$ on both sides:
By Phase 1, $\mathcal{T}^{\pi_{\text{new}}}$ is a $\gamma$-contraction whose repeated application converges unconditionally to its unique fixed point:
$$\lim_{k \to \infty} (\mathcal{T}^{\pi_{\text{new}}})^k Q^{\pi_{\text{old}}}(s, a) = Q^{\pi_{\text{new}}}(s, a)$$

Therefore:
$$Q^{\pi_{\text{new}}}(s, a) \ge Q^{\pi_{\text{old}}}(s, a) \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A} \quad \blacksquare$$

---

#### Derivation 11.20.2: Reparameterization Trick Gradient for Squashed Gaussian Policy

```
====================================================================================================
DERIVATION 11.20.2: Exact Change-of-Variables Density and Pathwise Reparameterized Policy Gradient
====================================================================================================
Problem Statement:
For a continuous action policy parameterized by theta where latent pre-squashed actions
u = mu_theta(s) + sigma_theta(s) \odot eps with eps ~ N(0, I_D) are squashed by the elementwise
hyperbolic tangent a = tanh(u) in (-1, 1)^D:
1. Prove the exact change-of-variables log-density formula:
       log pi_theta(a | s) = log mu_theta(u | s) - sum_{i=1}^D log(1 - tanh^2(u_i))
   via the multidimensional Jacobian determinant.
2. Derive the analytical pathwise policy gradient nabla_theta J(pi_theta) for the soft objective
   J(pi_theta) = E_{s, eps} [ Q(s, a_theta(s, eps)) - alpha log pi_theta(a_theta(s, eps) | s) ].
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
In continuous robotic control, physical actuators operate within bounded intervals, typically normalized to $a \in (-1, 1)^D$. To sample valid actions without boundary clipping artifacts, Soft Actor-Critic introduces a two-stage generative process:
1. An unconstrained Gaussian latent variable $u \in \mathbb{R}^D$ is generated via a neural network that outputs state-dependent mean $\boldsymbol{\mu}_{\boldsymbol{\theta}}(s) \in \mathbb{R}^D$ and standard deviation $\boldsymbol{\sigma}_{\boldsymbol{\theta}}(s) \in \mathbb{R}_{>0}^D$:
   $$u = g_{\boldsymbol{\theta}}(s, \epsilon) \triangleq \boldsymbol{\mu}_{\boldsymbol{\theta}}(s) + \boldsymbol{\sigma}_{\boldsymbol{\theta}}(s) \odot \epsilon, \quad \epsilon \sim p(\epsilon) = \mathcal{N}(\mathbf{0}, \mathbf{I}_D)$$
   where $\odot$ denotes the elementwise Hadamard product, and $\epsilon$ is parameter-independent standard Gaussian noise.
2. The latent variable $u$ is transformed via the coordinate-wise invertible squashing mapping $f: \mathbb{R}^D \to (-1, 1)^D$:
   $$a = f(u) \triangleq \left( \tanh(u_1), \tanh(u_2), \dots, \tanh(u_D) \right)^T$$

Our mathematical goals are:
1. Compute the Jacobian matrix $\mathbf{J}_f(u) \triangleq \frac{\partial a}{\partial u}$, evaluate its determinant $|\det(\mathbf{J}_f(u))|$, and derive the exact closed-form expression for the squashed probability density $\log \pi_{\boldsymbol{\theta}}(a \mid s)$ under the transformation of random variables theorem.
2. Using the Law of the Unconscious Statistician (LOTUS), pass the gradient operator $\nabla_{\boldsymbol{\theta}}$ through the expectation over base noise $\epsilon$, and derive the analytical pathwise policy gradient $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ for the soft actor objective:
   $$J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ Q(s, a_{\boldsymbol{\theta}}(s, \epsilon)) - \alpha \log \pi_{\boldsymbol{\theta}}(a_{\boldsymbol{\theta}}(s, \epsilon) \mid s) \right]$$

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **$C^1$-Diffeomorphism:** The squashing function $f: \mathbb{R}^D \to (-1, 1)^D$ given by $f_i(u) = \tanh(u_i)$ is a continuously differentiable, bijective diffeomorphism with non-zero Jacobian determinant everywhere on $\mathbb{R}^D$.
2. **Diagonal Latent Covariance:** The latent Gaussian distribution has a diagonal covariance matrix $\boldsymbol{\Sigma}_{\boldsymbol{\theta}}(s) = \operatorname{diag}\left(\sigma_{\boldsymbol{\theta}, 1}^2(s), \dots, \sigma_{\boldsymbol{\theta}, D}^2(s)\right)$, implying components $u_1, \dots, u_D$ are conditionally independent given $s$.
3. **Smoothness of Policy and Critic:** The network heads $\boldsymbol{\mu}_{\boldsymbol{\theta}}(s)$ and $\boldsymbol{\sigma}_{\boldsymbol{\theta}}(s)$ are continuously differentiable with respect to $\boldsymbol{\theta}$. The critic $Q(s, a)$ is continuously differentiable with respect to $a$.
4. **Dominated Convergence:** The gradient $\nabla_{\boldsymbol{\theta}} \left[ Q(s, a_{\boldsymbol{\theta}}(s, \epsilon)) - \alpha \log \pi_{\boldsymbol{\theta}}(a_{\boldsymbol{\theta}}(s, \epsilon) \mid s) \right]$ is bounded by an integrable function $M(s, \epsilon)$ such that $\mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}}[M(s, \epsilon)] < \infty$, permitting the interchange of derivative and expectation by the Leibniz integral rule.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
- **Geometric Volume Distortion:** When continuous probability mass is transported through a non-linear mapping $a = \tanh(u)$, probability mass must be conserved within corresponding differential volumes: $\pi(a \mid s) |da| = \mu(u \mid s) |du|$. Because $\tanh(u)$ has horizontal asymptotes at $\pm 1$, its slope $\frac{da}{du} = 1 - a^2$ approaches $0$ as $u \to \pm \infty$. This implies that a wide interval in latent space $u$ is compressed into an infinitesimally narrow interval near the action limits $a = \pm 1$. To conserve probability, the density $\pi(a \mid s)$ must blow up by the factor $\left|\frac{da}{du}\right|^{-1} = (1 - a^2)^{-1}$. In log-space, this appears as an additive penalty $-\sum_i \log(1 - a_i^2)$ that counteracts the compression.
- **Pathwise Gradient vs. REINFORCE:** In the classic policy gradient theorem (Williams, 1992), gradients are computed via the score function $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) Q(s, a)$, which treats the environment and critic as a black box and relies on high-variance trial-and-error reward weighting. In contrast, the reparameterization trick decouples the stochasticity (isolated in $\epsilon$) from the network parameters $\boldsymbol{\theta}$. The action $a = \tanh(\boldsymbol{\mu}_{\boldsymbol{\theta}} + \boldsymbol{\sigma}_{\boldsymbol{\theta}} \odot \epsilon)$ becomes a deterministic differentiable node in the computational graph. This allows the backpropagation algorithm to query the directional gradient of the critic $\nabla_a Q(s, a)$ directly, telling the actor precisely which direction in continuous action space produces higher Q-values.

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Phase 1: Multidimensional Jacobian Matrix & Determinant.*

Let $u = (u_1, u_2, \dots, u_D)^T \in \mathbb{R}^D$ and $a = (a_1, a_2, \dots, a_D)^T \in (-1, 1)^D$, where $a_i = \tanh(u_i)$ for $i = 1, \dots, D$.
The Jacobian matrix $\mathbf{J}_f(u) \in \mathbb{R}^{D \times D}$ of the vector-valued function $f: \mathbb{R}^D \to (-1, 1)^D$ is defined by:
$$[\mathbf{J}_f(u)]_{ij} \triangleq \frac{\partial a_i}{\partial u_j}, \quad i, j \in \{1, 2, \dots, D\}$$

Because each action component $a_i = \tanh(u_i)$ depends strictly on the corresponding scalar latent component $u_i$:
$$\frac{\partial a_i}{\partial u_j} = 0 \quad \text{for all } i \neq j$$
Thus, $\mathbf{J}_f(u)$ is a diagonal matrix.

Now compute the scalar diagonal entries. Recall the definition of the hyperbolic tangent:
$$\tanh(u_i) = \frac{\sinh(u_i)}{\cosh(u_i)} = \frac{e^{u_i} - e^{-u_i}}{e^{u_i} + e^{-u_i}}$$

Differentiating using the quotient rule:
$$\frac{d \tanh(u_i)}{du_i} = \frac{\frac{d \sinh(u_i)}{du_i} \cosh(u_i) - \sinh(u_i) \frac{d \cosh(u_i)}{du_i}}{\cosh^2(u_i)}$$
Using $\frac{d}{du} \sinh(u) = \cosh(u)$ and $\frac{d}{du} \cosh(u) = \sinh(u)$:
$$= \frac{\cosh(u_i)\cosh(u_i) - \sinh(u_i)\sinh(u_i)}{\cosh^2(u_i)} = \frac{\cosh^2(u_i) - \sinh^2(u_i)}{\cosh^2(u_i)}$$
Using the fundamental hyperbolic identity $\cosh^2(x) - \sinh^2(x) = 1$:
$$= \frac{1}{\cosh^2(u_i)} = 1 - \frac{\sinh^2(u_i)}{\cosh^2(u_i)} = 1 - \tanh^2(u_i)$$

Since $a_i = \tanh(u_i)$, we obtain:
$$[\mathbf{J}_f(u)]_{ii} = 1 - \tanh^2(u_i) = 1 - a_i^2$$

Because $\mathbf{J}_f(u)$ is diagonal, its determinant is the product of its diagonal elements:
$$\det\left( \mathbf{J}_f(u) \right) = \prod_{i=1}^D [\mathbf{J}_f(u)]_{ii} = \prod_{i=1}^D \left( 1 - \tanh^2(u_i) \right) = \prod_{i=1}^D \left( 1 - a_i^2 \right)$$

For any finite $u_i \in \mathbb{R}$, $-1 < \tanh(u_i) < 1$, which implies $\tanh^2(u_i) < 1$ and therefore $1 - \tanh^2(u_i) > 0$.
Because each diagonal factor is strictly positive:
$$\left| \det\left( \mathbf{J}_f(u) \right) \right| = \det\left( \mathbf{J}_f(u) \right) = \prod_{i=1}^D \left( 1 - \tanh^2(u_i) \right) > 0$$

*Phase 2: Change-of-Variables Density Formula.*

By the probability conservation theorem for invertible multivariate coordinate transformations $a = f(u)$:
$$\pi_{\boldsymbol{\theta}}(a \mid s) = \mu_{\boldsymbol{\theta}}(u \mid s) \cdot \left| \det\left( \mathbf{J}_f(u) \right) \right|^{-1}$$

Substituting the determinant expression:
$$\pi_{\boldsymbol{\theta}}(a \mid s) = \mu_{\boldsymbol{\theta}}(u \mid s) \cdot \left( \prod_{i=1}^D \left( 1 - \tanh^2(u_i) \right) \right)^{-1}$$

Taking the natural logarithm of both sides:
$$\log \pi_{\boldsymbol{\theta}}(a \mid s) = \log\left( \mu_{\boldsymbol{\theta}}(u \mid s) \cdot \prod_{i=1}^D \left( 1 - \tanh^2(u_i) \right)^{-1} \right)$$
$$= \log \mu_{\boldsymbol{\theta}}(u \mid s) + \sum_{i=1}^D \log\left( \left( 1 - \tanh^2(u_i) \right)^{-1} \right)$$
Using the logarithm power rule $\log(z^{-1}) = -\log(z)$:
$$\log \pi_{\boldsymbol{\theta}}(a \mid s) = \log \mu_{\boldsymbol{\theta}}(u \mid s) - \sum_{i=1}^D \log\left( 1 - \tanh^2(u_i) \right)$$

Because $a_i = \tanh(u_i)$, this is equivalently:
$$\log \pi_{\boldsymbol{\theta}}(a \mid s) = \log \mu_{\boldsymbol{\theta}}(u \mid s) - \sum_{i=1}^D \log\left( 1 - a_i^2 \right)$$

The latent distribution is a factorized multi-variate normal $\mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}(s), \operatorname{diag}(\boldsymbol{\sigma}_{\boldsymbol{\theta}}^2(s)))$:
$$\mu_{\boldsymbol{\theta}}(u \mid s) = \prod_{i=1}^D \frac{1}{\sqrt{2\pi}\sigma_{\boldsymbol{\theta}, i}(s)} \exp\left( - \frac{(u_i - \mu_{\boldsymbol{\theta}, i}(s))^2}{2\sigma_{\boldsymbol{\theta}, i}^2(s)} \right)$$
$$\log \mu_{\boldsymbol{\theta}}(u \mid s) = - \frac{1}{2} \sum_{i=1}^D \left[ \log(2\pi) + 2 \log \sigma_{\boldsymbol{\theta}, i}(s) + \left( \frac{u_i - \mu_{\boldsymbol{\theta}, i}(s)}{\sigma_{\boldsymbol{\theta}, i}(s)} \right)^2 \right]$$

*Phase 3: Analytical Pathwise Reparameterized Policy Gradient.*

The soft actor objective to be maximized is:
$$J(\boldsymbol{\theta}) \triangleq \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}(\cdot \mid s)} \left[ Q(s, a) - \alpha \log \pi_{\boldsymbol{\theta}}(a \mid s) \right] \right]$$

Substitute the reparameterized action mapping $a = \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \triangleq \tanh(g_{\boldsymbol{\theta}}(s, \epsilon)) = \tanh(\boldsymbol{\mu}_{\boldsymbol{\theta}}(s) + \boldsymbol{\sigma}_{\boldsymbol{\theta}}(s) \odot \epsilon)$, where $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D)$.
Under LOTUS:
$$J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ Q(s, \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon)) - \alpha \log \pi_{\boldsymbol{\theta}}(\tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \mid s) \right]$$

Because the noise distribution $p(\epsilon) = \mathcal{N}(\mathbf{0}, \mathbf{I}_D)$ does not depend on $\boldsymbol{\theta}$, we differentiate under the expectation:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \nabla_{\boldsymbol{\theta}} \left[ Q(s, \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon)) - \alpha \log \pi_{\boldsymbol{\theta}}(\tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \mid s) \right] \right]$$

Let us differentiate each term using the multivariate chain rule:
1. Critic term:
   $$\nabla_{\boldsymbol{\theta}} Q(s, \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon)) = \left( \nabla_{\boldsymbol{\theta}} \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \right)^T \nabla_a Q(s, a)\Big|_{a = \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon)}$$
2. Entropy term:
   Note that $\log \pi_{\boldsymbol{\theta}}(\tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \mid s)$ has two pathways of dependence on $\boldsymbol{\theta}$:
   - Direct parametric dependence in the distribution $\pi_{\boldsymbol{\theta}}(\cdot \mid s)$.
   - Indirect dependence through the evaluated action $\tilde{a}_{\boldsymbol{\theta}}(s, \epsilon)$.
   By the total derivative chain rule:
   $$\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(\tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \mid s) = \left( \nabla_{\boldsymbol{\theta}} \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \right)^T \nabla_a \log \pi_{\boldsymbol{\theta}}(a \mid s)\Big|_{a = \tilde{a}} + \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)\Big|_{a = \tilde{a}}$$

Combining these terms:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \nabla_{\boldsymbol{\theta}} \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon) \left( \nabla_a Q(s, a) - \alpha \nabla_a \log \pi_{\boldsymbol{\theta}}(a \mid s) \right)\Big|_{a = \tilde{a}} - \alpha \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)\Big|_{a = \tilde{a}} \right]$$

Now compute the Jacobian $\nabla_{\boldsymbol{\theta}} \tilde{a}_{\boldsymbol{\theta}}(s, \epsilon)$:
For each action dimension $i \in \{1, \dots, D\}$:
$$\tilde{a}_i = \tanh(u_i), \quad \text{where} \quad u_i = \mu_{\boldsymbol{\theta}, i}(s) + \sigma_{\boldsymbol{\theta}, i}(s) \epsilon_i$$
By the 1D chain rule:
$$\frac{\partial \tilde{a}_i}{\partial \boldsymbol{\theta}} = \frac{d \tanh(u_i)}{du_i} \frac{\partial u_i}{\partial \boldsymbol{\theta}} = \left( 1 - \tilde{a}_i^2 \right) \left[ \frac{\partial \mu_{\boldsymbol{\theta}, i}(s)}{\partial \boldsymbol{\theta}} + \epsilon_i \frac{\partial \sigma_{\boldsymbol{\theta}, i}(s)}{\partial \boldsymbol{\theta}} \right]$$

*Phase 4: Exact Decomposition of the Base Density Gradient.*

Consider the base Gaussian log-density:
$$\log \mu_{\boldsymbol{\theta}}(u \mid s) = - \frac{1}{2} \log(2\pi) - \log \sigma_{\boldsymbol{\theta}}(s) - \frac{(u - \mu_{\boldsymbol{\theta}}(s))^2}{2\sigma_{\boldsymbol{\theta}}^2(s)}$$
When evaluated at the reparameterized sample $u = \mu_{\boldsymbol{\theta}}(s) + \sigma_{\boldsymbol{\theta}}(s) \epsilon$, observe that:
$$\frac{u - \mu_{\boldsymbol{\theta}}(s)}{\sigma_{\boldsymbol{\theta}}(s)} = \frac{(\mu_{\boldsymbol{\theta}}(s) + \sigma_{\boldsymbol{\theta}}(s) \epsilon) - \mu_{\boldsymbol{\theta}}(s)}{\sigma_{\boldsymbol{\theta}}(s)} = \epsilon$$
Thus:
$$\log \mu_{\boldsymbol{\theta}}(u_{\boldsymbol{\theta}} \mid s) = - \frac{1}{2} \log(2\pi) - \log \sigma_{\boldsymbol{\theta}}(s) - \frac{1}{2} \epsilon^2$$

Differentiating this total expression with respect to $\mu_{\boldsymbol{\theta}}$:
$$\frac{d}{d \mu_{\boldsymbol{\theta}}} \log \mu_{\boldsymbol{\theta}}(u_{\boldsymbol{\theta}} \mid s) = 0$$
Indeed, expanding via the multivariate chain rule verifies this:
$$\frac{\partial \log \mu}{\partial \mu_{\boldsymbol{\theta}}} + \frac{\partial \log \mu}{\partial u} \frac{\partial u}{\partial \mu_{\boldsymbol{\theta}}} = \frac{u - \mu_{\boldsymbol{\theta}}}{\sigma_{\boldsymbol{\theta}}^2} + \left( - \frac{u - \mu_{\boldsymbol{\theta}}}{\sigma_{\boldsymbol{\theta}}^2} \right)(1) = \frac{\epsilon}{\sigma_{\boldsymbol{\theta}}} - \frac{\epsilon}{\sigma_{\boldsymbol{\theta}}} = 0$$
Differentiating with respect to $\log \sigma_{\boldsymbol{\theta}}$:
$$\frac{d}{d \log \sigma_{\boldsymbol{\theta}}} \log \mu_{\boldsymbol{\theta}}(u_{\boldsymbol{\theta}} \mid s) = \frac{d}{d \log \sigma_{\boldsymbol{\theta}}} \left[ - \frac{1}{2}\log(2\pi) - \log \sigma_{\boldsymbol{\theta}} - \frac{1}{2}\epsilon^2 \right] = -1$$
Expanding via the chain rule:
$$\frac{\partial \log \mu}{\partial \log \sigma_{\boldsymbol{\theta}}} + \frac{\partial \log \mu}{\partial u} \frac{\partial u}{\partial \log \sigma_{\boldsymbol{\theta}}} = \left( -1 + \frac{(u - \mu)^2}{\sigma^2} \right) + \left( - \frac{u - \mu}{\sigma^2} \right) (\sigma \epsilon) = (-1 + \epsilon^2) - \epsilon^2 = -1$$
This complete cancellation confirms that the reparameterized gradient of the base density is strictly constant w.r.t. the mean and w.r.t. the scale, providing exceptional gradient stability! $\blacksquare$

---

#### Derivation 11.20.3: Dual Gradient Ascent for Automatic Entropy Temperature Tuning

```
====================================================================================================
DERIVATION 11.20.3: Constrained Maximum Entropy Optimization and Convex Dual Temperature Dynamics
====================================================================================================
Problem Statement:
Given the constrained RL objective:
    max_pi E_{tau ~ pi} [ sum_{t=0}^infty gamma^t R(s_t, a_t) ]
    subject to: E_{(s_t, a_t) ~ rho_pi} [ -log pi(a_t | s_t) ] >= \bar{H}  forall t
1. Construct the Lagrangian and prove that the temperature alpha emerges as the Lagrange multiplier.
2. Define the dual objective function g(alpha) and prove that g(alpha) is convex in alpha >= 0.
3. Derive the dual subgradient and the log-temperature update rule beta = log alpha:
       beta <- beta - eta alpha ( -log pi(a | s) - \bar{H} )
   guaranteeing alpha > 0 strictly for all iterations.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
In Soft Actor-Critic, the temperature parameter $\alpha > 0$ controls the stochasticity of the policy. In tasks where rewards vary by several orders of magnitude, a fixed $\alpha$ leads to either premature entropy collapse (if $\alpha$ is too small) or complete disregard of rewards (if $\alpha$ is too large).

To eliminate manual tuning, Haarnoja et al. (2018) formulated the maximum entropy learning problem as a constrained optimization problem:
$$\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right] \quad \text{subject to} \quad \mathcal{H}(\pi(\cdot \mid s_t)) \ge \bar{\mathcal{H}} \quad \forall t$$
where $\mathcal{H}(\pi(\cdot \mid s_t)) \triangleq \mathbb{E}_{a_t \sim \pi(\cdot \mid s_t)} [-\log \pi(a_t \mid s_t)]$ and $\bar{\mathcal{H}} \in \mathbb{R}$ is a predefined minimum target entropy (heuristically set to $-\dim(\mathcal{A})$ for continuous control).

Our mathematical goals are:
1. Construct the Lagrangian of the per-timestep constrained problem, proving that maximizing the Lagrangian over policies $\pi$ corresponds precisely to the Soft Actor-Critic objective with temperature $\alpha$ acting as the dual variable (Lagrange multiplier).
2. Define the dual function $g(\alpha) \triangleq \max_\pi \mathcal{L}(\pi, \alpha)$ and prove from first principles that $g(\alpha)$ is convex on $\alpha \ge 0$.
3. Apply Danskin's theorem to derive the dual subgradient $\nabla_\alpha g(\alpha)$, prove convergence of dual gradient descent on the temperature loss $\mathcal{L}(\alpha) = \mathbb{E}[-\alpha \log \pi(a \mid s) - \alpha \bar{\mathcal{H}}]$, and derive the log-temperature parameterization $\beta = \log \alpha$ that enforces strict non-negativity $\alpha > 0$.

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Slater's Condition (Strict Feasibility):** There exists at least one feasible stochastic policy $\pi_0 \in \Pi$ such that $\mathbb{E}_{a \sim \pi_0(\cdot \mid s)} [-\log \pi_0(a \mid s)] > \bar{\mathcal{H}}$ for all $s \in \mathcal{S}$. This ensures strong duality holds by Slater's theorem.
2. **Dual Variable Constraint:** The temperature multiplier satisfies the non-negativity constraint $\alpha \ge 0$.
3. **Compact Policy Space & Bounded Expectations:** The policy space $\Pi$ is compact in the weak topology, and expected rewards $\mathbb{E}[R(s, a)]$ and entropies $\mathbb{E}[-\log \pi(a \mid s)]$ are bounded, ensuring the supremum in the dual function is achieved.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
- **Economic Shadow Price of Entropy:** In constrained optimization, the Lagrange multiplier $\alpha$ represents the "shadow price" of the constraint. If the current policy has insufficient entropy ($-\log \pi < \bar{\mathcal{H}}$), the constraint is violated; the shadow price $\alpha$ increases, making entropy much more valuable in the objective, which compels the actor to broaden its exploration. If the policy has excess entropy ($-\log \pi > \bar{\mathcal{H}}$), the constraint is slack; by complementary slackness, the shadow price $\alpha$ falls toward zero, allowing the actor to focus almost exclusively on exploiting high rewards.
- **Dual Convexity:** Even if the primal reinforcement learning objective is non-convex with respect to neural network weights, the dual function $g(\alpha)$ is guaranteed to be convex in the dual variable $\alpha$. This occurs because $g(\alpha)$ is formed by taking the pointwise supremum over a family of affine (linear) functions in $\alpha$. The upper envelope of linear lines is always convex. Hence, optimizing $\alpha$ via dual gradient descent has a unique global optimum without spurious local minima!

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Phase 1: Lagrangian Formulation.*

Consider the per-timestep policy optimization problem at state $s_t$:
$$\max_{\pi_t} \mathbb{E}_{a_t \sim \pi_t}[Q(s_t, a_t)] \quad \text{subject to} \quad \mathbb{E}_{a_t \sim \pi_t}[-\log \pi_t(a_t \mid s_t)] \ge \bar{\mathcal{H}}$$

Express the inequality constraint in standard non-positive form $c(\pi_t) \le 0$:
$$\bar{\mathcal{H}} - \mathbb{E}_{a_t \sim \pi_t}[-\log \pi_t(a_t \mid s_t)] \le 0 \iff \mathbb{E}_{a_t \sim \pi_t}[\log \pi_t(a_t \mid s_t)] + \bar{\mathcal{H}} \le 0$$

Introduce the Lagrange multiplier $\alpha_t \ge 0$. The Lagrangian $\mathcal{L}(\pi_t, \alpha_t)$ is defined as the objective minus the multiplier times the constraint:
$$\mathcal{L}(\pi_t, \alpha_t) \triangleq \mathbb{E}_{a_t \sim \pi_t}[Q(s_t, a_t)] - \alpha_t \left( \mathbb{E}_{a_t \sim \pi_t}[\log \pi_t(a_t \mid s_t)] + \bar{\mathcal{H}} \right)$$

Combining expectations over $a_t \sim \pi_t$:
$$= \mathbb{E}_{a_t \sim \pi_t} \left[ Q(s_t, a_t) - \alpha_t \log \pi_t(a_t \mid s_t) \right] - \alpha_t \bar{\mathcal{H}}$$

Notice the extraordinary result: for any fixed multiplier $\alpha_t$, the first term $\mathbb{E}_{a_t}[Q(s_t, a_t) - \alpha_t \log \pi_t(a_t \mid s_t)]$ is **IDENTICAL** to the Soft Actor-Critic maximum-entropy objective with temperature $\alpha_t$, while the second term $-\alpha_t \bar{\mathcal{H}}$ is an additive constant that does not depend on policy $\pi_t$.
Thus, maximizing the Lagrangian over $\pi_t$ yields the exact soft policy improvement step!

*Phase 2: Dual Objective Function and Rigorous Convexity Proof.*

The dual objective function $g(\alpha)$ is obtained by maximizing the Lagrangian over all policies $\pi \in \Pi$:
$$g(\alpha) \triangleq \max_{\pi \in \Pi} \mathcal{L}(\pi, \alpha) = \max_{\pi \in \Pi} \left\{ \mathbb{E}_{(s, a) \sim \pi} [Q(s, a)] - \alpha \left( \mathbb{E}_{(s, a) \sim \pi} [\log \pi(a \mid s)] + \bar{\mathcal{H}} \right) \right\}$$

For any fixed policy $\pi \in \Pi$, define the function $\phi_\pi: \mathbb{R}_{\ge 0} \to \mathbb{R}$:
$$\phi_\pi(\alpha) \triangleq A_\pi - \alpha B_\pi$$
where:
$$A_\pi \triangleq \mathbb{E}_{(s, a) \sim \pi}[Q(s, a)], \quad B_\pi \triangleq \mathbb{E}_{(s, a) \sim \pi}[\log \pi(a \mid s)] + \bar{\mathcal{H}}$$
Notice that $A_\pi and B_\pi$ are constants that depend only on $\pi$, not on $\alpha$.
Therefore, $\phi_\pi(\alpha)$ is an affine (linear plus constant) function of $\alpha$.

Now express the dual function as the pointwise supremum:
$$g(\alpha) = \sup_{\pi \in \Pi} \phi_\pi(\alpha)$$

We now prove that $g(\alpha)$ is convex.
Let $\alpha_1, \alpha_2 \ge 0$ and let $\lambda \in [0, 1]$.
Evaluate $g$ at the convex combination $\lambda \alpha_1 + (1 - \lambda)\alpha_2$:
$$g\left( \lambda \alpha_1 + (1 - \lambda)\alpha_2 \right) = \sup_{\pi \in \Pi} \phi_\pi\left( \lambda \alpha_1 + (1 - \lambda)\alpha_2 \right)$$

Because each $\phi_\pi$ is an affine function, it satisfies linearity:
$$\phi_\pi\left( \lambda \alpha_1 + (1 - \lambda)\alpha_2 \right) = A_\pi - \left(\lambda \alpha_1 + (1 - \lambda)\alpha_2\right) B_\pi$$
$$= \lambda (A_\pi - \alpha_1 B_\pi) + (1 - \lambda)(A_\pi - \alpha_2 B_\pi) = \lambda \phi_\pi(\alpha_1) + (1 - \lambda)\phi_\pi(\alpha_2)$$

Substituting this identity into the supremum:
$$g\left( \lambda \alpha_1 + (1 - \lambda)\alpha_2 \right) = \sup_{\pi \in \Pi} \left[ \lambda \phi_\pi(\alpha_1) + (1 - \lambda)\phi_\pi(\alpha_2) \right]$$

By the sub-additivity property of the supremum operator ($\sup_x (f(x) + h(x)) \le \sup_x f(x) + \sup_x h(x)$) and non-negativity of $\lambda, 1 - \lambda$:
$$\le \sup_{\pi \in \Pi} \left[ \lambda \phi_\pi(\alpha_1) \right] + \sup_{\pi' \in \Pi} \left[ (1 - \lambda)\phi_{\pi'}(\alpha_2) \right]$$
$$= \lambda \sup_{\pi \in \Pi} \phi_\pi(\alpha_1) + (1 - \lambda) \sup_{\pi' \in \Pi} \phi_{\pi'}(\alpha_2)$$
$$= \lambda g(\alpha_1) + (1 - \lambda) g(\alpha_2)$$

This proves conclusively that $g(\alpha)$ is convex on $\alpha \ge 0$.

*Phase 3: Dual Subgradient via Danskin's Theorem.*

The dual problem consists of finding the optimal multiplier that minimizes the dual upper bound:
$$\min_{\alpha \ge 0} g(\alpha)$$

Let $\pi^*_\alpha \triangleq \arg\max_{\pi \in \Pi} \mathcal{L}(\pi, \alpha)$ denote the optimal soft policy for a given $\alpha$.
By **Danskin's Theorem** for the subdifferential of a maximum function, the subgradient of $g(\alpha)$ at $\alpha$ is given by the partial derivative of the Lagrangian evaluated at the maximizer $\pi^*_\alpha$:
$$\nabla_\alpha g(\alpha) = \left. \frac{\partial \mathcal{L}(\pi, \alpha)}{\partial \alpha} \right|_{\pi = \pi^*_\alpha}$$

Computing the partial derivative:
$$\frac{\partial}{\partial \alpha} \left[ \mathbb{E}_{\pi} [Q(s, a)] - \alpha \left( \mathbb{E}_{\pi}[\log \pi(a \mid s)] + \bar{\mathcal{H}} \right) \right] = - \left( \mathbb{E}_{(s, a) \sim \pi^*_\alpha} [\log \pi^*_\alpha(a \mid s)] + \bar{\mathcal{H}} \right)$$

Therefore:
$$\nabla_\alpha g(\alpha) = - \mathbb{E}_{(s, a) \sim \pi^*_\alpha} \left[ \log \pi^*_\alpha(a \mid s) + \bar{\mathcal{H}} \right]$$

To minimize $g(\alpha)$ via gradient descent with learning rate $\eta_\alpha > 0$:
$$\alpha \leftarrow \alpha - \eta_\alpha \nabla_\alpha g(\alpha) = \alpha + \eta_\alpha \mathbb{E}_{(s, a) \sim \pi^*_\alpha} \left[ \log \pi^*_\alpha(a \mid s) + \bar{\mathcal{H}} \right]$$

Equivalently, defining the sample-based temperature loss function:
$$\mathcal{L}(\alpha) \triangleq \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ - \alpha \log \pi(a \mid s) - \alpha \bar{\mathcal{H}} \right]$$
Taking the derivative with respect to $\alpha$:
$$\nabla_\alpha \mathcal{L}(\alpha) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ - \log \pi(a \mid s) - \bar{\mathcal{H}} \right]$$

Examining the dynamics:
- **Case 1 (Entropy too low):** $-\mathbb{E}[\log \pi] < \bar{\mathcal{H}} \implies -\log \pi - \bar{\mathcal{H}} < 0 \implies \nabla_\alpha \mathcal{L}(\alpha) < 0$.
  The gradient descent update $\alpha \leftarrow \alpha - \eta_\alpha \nabla_\alpha \mathcal{L}$ results in $\Delta \alpha > 0$. The temperature **increases**, boosting the entropy reward and expanding policy exploration.
- **Case 2 (Entropy too high):** $-\mathbb{E}[\log \pi] > \bar{\mathcal{H}} \implies -\log \pi - \bar{\mathcal{H}} > 0 \implies \nabla_\alpha \mathcal{L}(\alpha) > 0$.
  The gradient descent update results in $\Delta \alpha < 0$. The temperature **decreases**, reducing exploration and encouraging greedy reward exploitation.

*Phase 4: Reparameterization via Log-Temperature $\beta = \log \alpha$.*

In discrete optimization steps, a large gradient step on $\alpha$ could cause $\alpha < 0$, violating the non-negativity constraint and causing the soft Bellman operator to become an anti-entropy expansion.
To enforce $\alpha > 0$ strictly, define:
$$\alpha \triangleq \exp(\beta), \quad \text{where} \quad \beta \in \mathbb{R} \quad (\beta = \log \alpha)$$

Differentiating $\mathcal{L}$ with respect to the unconstrained parameter $\beta$:
By the chain rule:
$$\nabla_\beta \mathcal{L}(\beta) = \frac{\partial \mathcal{L}}{\partial \alpha} \cdot \frac{\partial \alpha}{\partial \beta}$$
Since $\frac{\partial \alpha}{\partial \beta} = \frac{d}{d\beta} \exp(\beta) = \exp(\beta) = \alpha$:
$$\nabla_\beta \mathcal{L}(\beta) = \nabla_\alpha \mathcal{L}(\alpha) \cdot \alpha = \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ - \log \pi(a \mid s) - \bar{\mathcal{H}} \right]$$

The unconstrained gradient descent update on $\beta$ is:
$$\beta \leftarrow \beta - \eta_\alpha \nabla_\beta \mathcal{L}(\beta) = \beta - \eta_\alpha \alpha \left( - \log \pi(a \mid s) - \bar{\mathcal{H}} \right)$$

Because $\alpha = \exp(\beta)$, $\alpha > 0$ strictly for all $\beta \in (-\infty, \infty)$.
Moreover, when $\alpha$ is small, the effective gradient $\alpha \nabla_\alpha \mathcal{L}$ naturally scales down proportionally to $\alpha$, preventing abrupt oscillations near zero. This guarantees smooth, stable, and numerically safe temperature adaptation throughout training. $\blacksquare$

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
| $\left\vert \frac{da}{du} \right\vert$ | Jacobian Determinant | $1 - \tanh^2(u) = 1 - a'^2$ |
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

### Illustration 1: Derivation and Verification of the $\tanh$ Jacobian Correction
**Problem:**
Let $u \sim p(u)$ be a 1D continuous random variable representing an unconstrained latent action. Let $a = \tanh(u) \in (-1, 1)$ be the squashed physical action.
1. Prove rigorously from the continuous probability conservation law that:
   $$\log p(a) = \log p(u) - \log(1 - a^2)$$
2. For $u = 0.5000$, evaluate $a$, $\frac{da}{du}$, and verify that $p(a) > p(u)$.

**Solution:**

*Step 1: Conservation of probability.*
For any monotonically increasing continuous transformation $a = f(u)$, the cumulative distribution satisfies $F_A(a) = \mathbb{P}(A \le a) = \mathbb{P}(U \le f^{-1}(a)) = F_U(f^{-1}(a))$.
Differentiating both sides with respect to $a$ using the chain rule:
$$p_A(a) = \frac{d F_A(a)}{da} = \frac{d F_U(u)}{du} \frac{du}{da} = p_U(u) \left( \frac{da}{du} \right)^{-1} = p_U(u) \left| \frac{da}{du} \right|^{-1}$$

*Step 2: Derivative of $\tanh$.*
Using $a = \tanh(u) = \frac{\sinh(u)}{\cosh(u)}$:
$$\frac{da}{du} = \frac{\cosh^2(u) - \sinh^2(u)}{\cosh^2(u)} = 1 - \tanh^2(u) = 1 - a^2$$
Because $a \in (-1, 1)$, $a^2 < 1$, so $1 - a^2 > 0$ strictly, and $|1 - a^2| = 1 - a^2$.
Therefore:
$$p_A(a) = \frac{p_U(u)}{1 - a^2}$$

*Step 3: Log-density transformation.*
Taking the natural logarithm of both sides:
$$\log p_A(a) = \log\left( \frac{p_U(u)}{1 - a^2} \right) = \log p_U(u) - \log(1 - a^2) \quad \blacksquare$$

*Step 4: Concrete numerical evaluation at $u = 0.5000$.*
1. Squashed action:
   $$a = \tanh(0.5000) = \frac{e^{0.5} - e^{-0.5}}{e^{0.5} + e^{-0.5}} = \frac{1.648721 - 0.606531}{1.648721 + 0.606531} = \frac{1.042191}{2.255252} \approx \mathbf{0.462117}$$
2. Jacobian factor:
   $$\frac{da}{du} = 1 - a^2 = 1 - (0.462117)^2 = 1 - 0.213552 = \mathbf{0.786448}$$
3. Density ratio:
   $$p_A(a) = \frac{p_U(u)}{0.786448} \approx 1.27154 \cdot p_U(u)$$
   Because $0 < 1 - a^2 < 1$, the volume is compressed, so the squashed action density $p_A(a)$ is approximately $27.15\%$ higher than the latent density $p_U(u)$! $\blacksquare$

---

### Illustration 2: Squashed Gaussian Action Forward Pass and Exact Log-Determinant Probability Calculation
**Problem:**
At state $s$, a SAC stochastic Gaussian policy outputs latent mean $\mu(s) = 0.2000$ and standard deviation $\sigma(s) = 1.2000$.
A random noise variable $\epsilon = +0.5000$ is sampled from the standard normal $\mathcal{N}(0, 1)$.
1. Compute the unsquashed latent variable $u = \mu + \sigma \epsilon$.
2. Compute the squashed action $a = \tanh(u)$.
3. Compute the base Gaussian log-density $\log \mu(u \mid s)$.
4. Compute the Jacobian determinant log-correction $\log(1 - a^2)$.
5. Compute the final squashed policy log-probability $\log \pi(a \mid s)$.

**Solution:**

*Step 1: Latent variable $u$.*
Using the reparameterization formula:
$$u = \mu(s) + \sigma(s) \cdot \epsilon = 0.2000 + (1.2000 \times 0.5000) = 0.2000 + 0.6000 = \mathbf{0.8000}$$

*Step 2: Squashed action $a$.*
Apply the hyperbolic tangent squashing:
$$a = \tanh(u) = \tanh(0.8000) = \frac{e^{0.8} - e^{-0.8}}{e^{0.8} + e^{-0.8}}$$
Evaluate exponential terms:
$$e^{0.8000} \approx 2.225541, \quad e^{-0.8000} \approx 0.449329$$
$$a = \frac{2.225541 - 0.449329}{2.225541 + 0.449329} = \frac{1.776212}{2.674870} \approx \mathbf{0.664037}$$

*Step 3: Base Gaussian log-likelihood $\log \mu(u \mid s)$.*
For a 1D normal distribution $\mathcal{N}(\mu, \sigma^2)$:
$$\log \mu(u \mid s) = -\frac{1}{2} \ln(2\pi) - \ln(\sigma) - \frac{(u - \mu)^2}{2\sigma^2}$$
Evaluating each constituent term:
- Constant factor: $-\frac{1}{2}\ln(2\pi) = -0.5 \times 1.837877 = \mathbf{-0.918939}$
- Log standard deviation: $\ln(\sigma) = \ln(1.2000) = \mathbf{+0.182322}$
- Standardized quadratic deviation:
  $$\frac{(u - \mu)^2}{2\sigma^2} = \frac{(0.8000 - 0.2000)^2}{2(1.2000)^2} = \frac{(0.6000)^2}{2 \times 1.4400} = \frac{0.3600}{2.8800} = \mathbf{0.125000}$$
Summing terms:
$$\log \mu(u \mid s) = -0.918939 - 0.182322 - 0.125000 = \mathbf{-1.226260}$$

*Step 4: Jacobian correction term.*
Compute the squashing contraction factor:
$$1 - a^2 = 1 - (0.664037)^2 = 1 - 0.440945 = \mathbf{0.559055}$$
Taking the natural logarithm:
$$\ln(1 - a^2) = \ln(0.559055) \approx \mathbf{-0.581507}$$

*Step 5: Final squashed policy log-probability.*
Applying the change-of-variables theorem (Derivation 11.20.2):
$$\log \pi(a \mid s) = \log \mu(u \mid s) - \ln(1 - a^2)$$
$$= -1.226260 - (-0.581507) = -1.226260 + 0.581507 = \mathbf{-0.644753} \quad \blacksquare$$

---

### Illustration 3: Soft Bellman Target Calculation with Twin Critics and Temperature $\alpha = 0.20$
**Problem:**
A transition tuple $(s, a, r, s', d)$ is sampled from the replay buffer with:
- Reward: $r = 3.5000$
- Discount: $\gamma = 0.9500$
- Done flag: $d = 0$ (non-terminal)
- Next state: $s' = 1.5000$
- Current temperature: $\alpha = 0.2000$

The actor samples next-state action $a' = 0.664037$ with log-probability $\log \pi(a' \mid s') = -0.644753$ (from Illustration 2).
The twin target critic networks output:
$$Q_{\boldsymbol{\phi}_1^-}(s', a') = 3.0000 \cdot s' + 2.0000 \cdot a'$$
$$Q_{\boldsymbol{\phi}_2^-}(s', a') = 2.5000 \cdot s' + 3.0000 \cdot a'$$
1. Evaluate $Q_{\boldsymbol{\phi}_1^-}(s', a')$ and $Q_{\boldsymbol{\phi}_2^-}(s', a')$.
2. Compute the Clipped Double-Q minimum $\min(Q_1^-, Q_2^-)$.
3. Compute the entropy-augmented soft state value $V(s')$.
4. Compute the final Soft Bellman regression target $y$.

**Solution:**

*Step 1: Evaluate twin target critics.*
- Target Critic 1:
  $$Q_{\boldsymbol{\phi}_1^-}(s', a') = (3.0000 \times 1.5000) + (2.0000 \times 0.664037) = 4.500000 + 1.328074 = \mathbf{5.828074}$$
- Target Critic 2:
  $$Q_{\boldsymbol{\phi}_2^-}(s', a') = (2.5000 \times 1.5000) + (3.0000 \times 0.664037) = 3.750000 + 1.992111 = \mathbf{5.742111}$$

*Step 2: Clipped Double-Q minimum.*
To prevent overestimation bias:
$$\min\left( Q_{\boldsymbol{\phi}_1^-}(s', a'), Q_{\boldsymbol{\phi}_2^-}(s', a') \right) = \min(5.828074, 5.742111) = \mathbf{5.742111}$$

*Step 3: Entropy-augmented soft value $V(s')$.*
Subtract the scaled policy log-probability:
$$V(s') = \min\left( Q_1^-, Q_2^- \right) - \alpha \log \pi(a' \mid s')$$
$$\alpha \log \pi(a' \mid s') = 0.2000 \times (-0.644753) = -0.128951$$
$$V(s') = 5.742111 - (-0.128951) = 5.742111 + 0.128951 = \mathbf{5.871062}$$

*Step 4: Soft Bellman target $y$.*
$$y = r + (1 - d)\gamma V(s') = 3.500000 + (1 - 0)(0.9500 \times 5.871062)$$
$$0.9500 \times 5.871062 = 5.577509$$
$$y = 3.500000 + 5.577509 = \mathbf{9.077509} \quad \blacksquare$$

---

### Illustration 4: Dual Temperature Update Step
**Problem:**
Let the current entropy temperature be $\alpha = 0.2500$ with learning rate $\eta_\alpha = 0.0500$.
The target entropy for a 1D continuous action space is $\bar{\mathcal{H}} = -\dim(\mathcal{A}) = -1.0000$.
An action is sampled with log-probability $\log \pi(a \mid s) = -0.644753$.
1. Compute the analytical gradient $\nabla_\alpha \mathcal{L}(\alpha)$ of the temperature loss $\mathcal{L}(\alpha) = -\alpha (\log \pi(a \mid s) + \bar{\mathcal{H}})$.
2. Perform a direct gradient descent step $\alpha_{\text{new}} = \alpha - \eta_\alpha \nabla_\alpha \mathcal{L}(\alpha)$.
3. Perform the numerically stable log-temperature gradient step on $\beta = \log \alpha$, and compute the updated temperature $\alpha_{\text{new}}^{(\beta)} = \exp(\beta_{\text{new}})$.
4. Explain the qualitative behavior of the temperature change.

**Solution:**

*Step 1: Analytical gradient w.r.t. $\alpha$.*
The temperature loss is:
$$\mathcal{L}(\alpha) = - \alpha \left( \log \pi(a \mid s) + \bar{\mathcal{H}} \right)$$
Differentiating with respect to $\alpha$:
$$\nabla_\alpha \mathcal{L}(\alpha) = - \left( \log \pi(a \mid s) + \bar{\mathcal{H}} \right)$$
Substitute $\log \pi = -0.644753$ and $\bar{\mathcal{H}} = -1.000000$:
$$\log \pi(a \mid s) + \bar{\mathcal{H}} = -0.644753 + (-1.000000) = -1.644753$$
$$\nabla_\alpha \mathcal{L}(\alpha) = - (-1.644753) = \mathbf{+1.644753}$$

*Step 2: Direct gradient descent update on $\alpha$.*
$$\alpha_{\text{new}} = \alpha - \eta_\alpha \nabla_\alpha \mathcal{L}(\alpha) = 0.250000 - 0.0500 \times (+1.644753)$$
$$= 0.250000 - 0.082238 = \mathbf{0.167762}$$

*Step 3: Log-temperature update on $\beta = \log \alpha$.*
Initialize $\beta$:
$$\beta = \ln(\alpha) = \ln(0.250000) \approx \mathbf{-1.386294}$$
The gradient with respect to $\beta$ is (by Derivation 11.20.3):
$$\nabla_\beta \mathcal{L}(\beta) = \alpha \nabla_\alpha \mathcal{L}(\alpha) = 0.250000 \times (+1.644753) = \mathbf{+0.411188}$$
Gradient descent update on $\beta$:
$$\beta_{\text{new}} = \beta - \eta_\alpha \nabla_\beta \mathcal{L}(\beta) = -1.386294 - (0.0500 \times 0.411188)$$
$$= -1.386294 - 0.020559 = \mathbf{-1.406853}$$
Recover updated temperature:
$$\alpha_{\text{new}}^{(\beta)} = \exp(\beta_{\text{new}}) = \exp(-1.406853) \approx \mathbf{0.244913}$$

*Step 4: Intuition of Temperature Adaptation.*
The current policy entropy is $\mathcal{H}(\pi) = -\log \pi(a \mid s) = -(-0.644753) = +0.644753$.
The target entropy is $\bar{\mathcal{H}} = -1.000000$.
Because the policy entropy $+0.644753$ is significantly *greater* than the required minimum target $-1.000000$, the policy is already exploring sufficiently. The positive gradient pushes temperature downward ($\alpha \downarrow$), cooling the system so the policy can exploit higher rewards without being penalized for concentrating! $\blacksquare$

---

### Illustration 5: Reparameterized Actor Gradient Step with Automatic Differentiation Trace
**Problem:**
Consider a parameterized 1D Gaussian policy with single state feature $s = 1.0000$:
$$\mu_{\boldsymbol{\theta}}(s) = \theta_\mu = 0.5000, \quad \log \sigma_{\boldsymbol{\theta}}(s) = \theta_{\log\sigma} = 0.2000$$
The standard deviation is $\sigma = \exp(\theta_{\log\sigma}) = \exp(0.2000) \approx 1.221403$.
Let sampled noise $\epsilon = +0.4000$, temperature $\alpha = 0.2000$, and critic $Q(s, a) = 2.0 \cdot s + 3.0 \cdot a$.
1. Compute the forward pass: latent $u$, squashed action $a$, base log-prob $\log \mu(u)$, Jacobian correction $\ln(1 - a^2)$, policy log-prob $\log \pi(a \mid s)$, critic value $Q(s, a)$, and actor loss $\mathcal{L}(\boldsymbol{\theta}) = \alpha \log \pi(a \mid s) - Q(s, a)$.
2. Trace the exact backward pass through the computational graph: compute $\frac{\partial \mathcal{L}}{\partial a}$, $\frac{\partial a}{\partial u}$, $\frac{\partial \mathcal{L}}{\partial u}$, $\frac{\partial \mathcal{L}}{\partial \theta_\mu}$, and $\frac{\partial \mathcal{L}}{\partial \theta_{\log\sigma}}$.
3. Compare the manual chain rule gradients against PyTorch automatic differentiation to verify exact match.

**Solution:**

*Step 1: Forward Pass.*
1. Latent action $u$:
   $$u = \theta_\mu + \sigma \epsilon = 0.5000 + (1.221403 \times 0.4000) = 0.5000 + 0.488561 = \mathbf{0.988561}$$
2. Squashed action $a$:
   $$a = \tanh(u) = \tanh(0.988561) = \frac{e^{0.988561} - e^{-0.988561}}{e^{0.988561} + e^{-0.988561}} = \frac{2.687352 - 0.372113}{2.687352 + 0.372113} = \frac{2.315239}{3.059465} \approx \mathbf{0.756748}$$
3. Critic evaluation:
   $$Q(s, a) = (2.0000 \times 1.0000) + (3.0000 \times 0.756748) = 2.000000 + 2.270244 = \mathbf{4.270244}$$
4. Base Gaussian log-density:
   $$\log \mu(u) = -\frac{1}{2}\ln(2\pi) - \theta_{\log\sigma} - \frac{1}{2}\epsilon^2 = -0.918939 - 0.200000 - \frac{1}{2}(0.4000)^2$$
   $$= -1.118939 - 0.080000 = \mathbf{-1.198939}$$
5. Jacobian correction:
   $$1 - a^2 = 1 - (0.756748)^2 = 1 - 0.572668 = \mathbf{0.427332}$$
   $$\ln(1 - a^2) = \ln(0.427332) \approx \mathbf{-0.850193}$$
6. Squashed policy log-prob:
   $$\log \pi(a \mid s) = \log \mu(u) - \ln(1 - a^2) = -1.198939 - (-0.850193) = \mathbf{-0.348746}$$
7. Actor Loss:
   $$\mathcal{L}(\boldsymbol{\theta}) = \alpha \log \pi(a \mid s) - Q(s, a) = (0.2000 \times -0.348746) - 4.270244$$
   $$= -0.069749 - 4.270244 = \mathbf{-4.339993}$$

*Step 2: Backward Pass through Computational Graph.*
The actor objective is $\mathcal{L} = \alpha \log \mu - \alpha \ln(1 - a^2) - Q(s, a)$.
1. Derivative w.r.t. action $a$:
   - Critic component: $\frac{\partial (-Q)}{\partial a} = -3.000000$
   - Entropy component: $\frac{\partial}{\partial a}\left[ -\alpha \ln(1 - a^2) \right] = -\alpha \frac{-2a}{1 - a^2} = \frac{2\alpha a}{1 - a^2}$
     $$\frac{2 \times 0.2000 \times 0.756748}{0.427332} = \frac{0.302699}{0.427332} \approx \mathbf{+0.708346}$$
   - Total action derivative:
     $$\frac{\partial \mathcal{L}}{\partial a} = -3.000000 + 0.708346 = \mathbf{-2.291654}$$
2. Squashing derivative:
   $$\frac{\partial a}{\partial u} = 1 - a^2 = \mathbf{0.427332}$$
3. Derivative w.r.t. latent action $u$:
   $$\frac{\partial \mathcal{L}}{\partial u} = \frac{\partial \mathcal{L}}{\partial a} \frac{\partial a}{\partial u} = \left( -3.000000 + \frac{2\alpha a}{1 - a^2} \right) (1 - a^2) = -3.000000(1 - a^2) + 2\alpha a$$
   $$= -3.000000(0.427332) + 0.302699 = -1.281996 + 0.302699 = \mathbf{-0.979298}$$
4. Parameter gradient $\frac{\partial \mathcal{L}}{\partial \theta_\mu}$:
   Since $u = \theta_\mu + \sigma \epsilon$, $\frac{\partial u}{\partial \theta_\mu} = 1$.
   From Derivation 11.20.2, the total derivative of the base density w.r.t. $\theta_\mu$ is 0.
   Therefore:
   $$\frac{\partial \mathcal{L}}{\partial \theta_\mu} = \frac{\partial \mathcal{L}}{\partial u} \frac{\partial u}{\partial \theta_\mu} = -0.979298 \times 1.000000 = \mathbf{-0.979298}$$
5. Parameter gradient $\frac{\partial \mathcal{L}}{\partial \theta_{\log\sigma}}$:
   Since $\frac{\partial u}{\partial \theta_{\log\sigma}} = \frac{\partial (\theta_\mu + \exp(\theta_{\log\sigma})\epsilon)}{\partial \theta_{\log\sigma}} = \sigma \epsilon$:
   $$\sigma \epsilon = 1.221403 \times 0.4000 = \mathbf{0.488561}$$
   From Derivation 11.20.2, the total derivative of $\alpha \log \mu$ w.r.t. $\theta_{\log\sigma}$ is $-\alpha = -0.200000$.
   Therefore:
   $$\frac{\partial \mathcal{L}}{\partial \theta_{\log\sigma}} = \frac{\partial \mathcal{L}}{\partial u} \frac{\partial u}{\partial \theta_{\log\sigma}} - \alpha = (-0.979298 \times 0.488561) - 0.200000$$
   $$= -0.478447 - 0.200000 = \mathbf{-0.678447}$$

*Step 3: Verification with PyTorch Automatic Differentiation.*
Running PyTorch autograd on this exact computational graph:
- `loss.item() = -4.33999395` (matches hand calculation $-4.339993$ to $10^{-6}$)
- `theta_mu.grad = -0.97929767` (matches hand calculation $-0.979298$ to $10^{-6}$)
- `theta_logstd.grad = -0.67844675` (matches hand calculation $-0.678447$ to $10^{-6}$)

Both parameter gradients are negative, meaning gradient descent ($\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$) will *increase* both $\theta_\mu$ and $\theta_{\log\sigma}$, steering the actor toward higher actions (which yield higher Q-values, since $\nabla_a Q = +3.0 > 0$) while simultaneously increasing exploration entropy! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. The King of Model-Free Continuous Control: Real-Robot SAC (2019–2024)
SAC is the dominant off-policy continuous RL algorithm in both research and industry:
- **Quadruped Locomotion (Unitree Go1/Go2, ANYmal):** SAC trains walking, trotting, and bounding gaits in under 2 hours of real-world robot data. The maximum entropy objective naturally generates diverse exploratory gaits during early training, escaping local minima that deterministic TD3 policies get stuck in.
- **Dexterous Manipulation (Berkeley DEXTEROUS Lab, 2023):** SAC with automatic entropy tuning trained a five-fingered robot hand to perform in-hand object reorientation — rotating arbitrary objects to target orientations — achieving human-level dexterity from raw pixel observations.
- **Google Robotics (RT-X, 2023):** SAC fine-tuning on top of imitation-learning-initialized policies enables robots to learn new manipulation tasks from fewer than 10 real-world demonstrations, with the entropy term preventing collapse back to demonstration mode.

### 2. Diffusion Policies & Maximum Entropy Robotics Foundations
Modern robotics foundation models build directly on SAC's maximum-entropy philosophy:
- **Diffusion Policy (Chi et al., RSS 2023):** Models the action distribution as a diffusion denoising process $p_\theta(a_t | o_t)$, inheriting SAC's principle that behavioral diversity is critical for robust generalization. Outperforms deterministic behavior cloning on 11/12 manipulation tasks.
- **Octo (Ghosh et al., RSS 2024):** A 93M-parameter generalist robot policy trained on 800K demonstrations, using maximum entropy action head design inspired by SAC to handle multi-modal action distributions across diverse robot embodiments.
- **Energy-Based Models (EBMs):** The SAC soft Q-function $Q_\text{soft}(s, a) = r + \gamma (\mathbb{E}[Q_\text{soft}'] - \alpha \log \pi)$ is mathematically equivalent to an energy function on actions, connecting SAC to the statistical mechanics foundations underlying modern score-based generative models.

### 3. Automatic Entropy Tuning: The $\alpha$ Self-Calibration Mechanism
SAC's automatic entropy tuning (Haarnoja et al., 2019 follow-up) is now universal in production RL:
$$\alpha_{t+1} = \alpha_t - \eta_\alpha \cdot \mathbb{E}_{a_t \sim \pi}[-\log \pi_t(a_t | s_t) - \bar{\mathcal{H}}]$$
where $\bar{\mathcal{H}} = -|\mathcal{A}|$ is the target entropy (typically $-\dim(\mathcal{A})$). This single equation eliminates the need to manually tune $\alpha$ across tasks — the algorithm automatically decreases entropy (becoming more decisive) when the policy is uncertain about which actions are best, and increases it (exploring more) when the policy becomes overconfident.

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
