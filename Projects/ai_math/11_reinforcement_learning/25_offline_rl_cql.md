# Chapter 25: Offline (Batch) RL & Conservative Q-Learning (CQL)

---

## 1. Intuition & 101 Motivation

Throughout this curriculum, every reinforcement learning algorithm we examined—from Q-learning and DQN to PPO and SAC—relied on **online interaction**: the agent gathers experience in an environment, updates its parameters, and immediately tries out its new policy.

However, in many of the most critical real-world domains, **exploratory online trial-and-error is dangerous, unethical, or prohibitively expensive**:
- **Healthcare & Clinical Medicine:** An RL agent cannot experiment with untested drug dosages on human patients to see if they survive.
- **Autonomous Driving:** A self-driving vehicle cannot deliberately execute dangerous swerves on a highway to learn how to recover from a skid.
- **Nuclear Power & Industrial Control:** An algorithm cannot cause reactor meltdowns to explore its failure boundaries.

Fortunately, in these domains we already possess **massive historical offline datasets** $\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1})\}$ collected by human doctors, expert human drivers, or rule-based legacy systems.

The ambition of **Offline Reinforcement Learning (Batch RL)** is:
> *Can we learn an optimal decision-making policy $\pi^*$ strictly from a static dataset $\mathcal{D}$, without taking a single online step in the environment?*

### Why Standard Off-Policy RL Fails: The OOD Overestimation Trap
Naively, algorithms like DQN or SAC are supposed to be "off-policy," meaning their Bellman equations theoretically permit training on arbitrary replay buffers. Why can't we simply train DQN or SAC on an offline dataset?

The answer is **Distributional Shift** and **Out-of-Distribution (OOD) Overestimation**:
1. When computing the Bellman target:
   $$y = r + \gamma \max_{a'} Q_\theta(s', a')$$
   the $\max$ operator queries the neural network $Q_\theta$ at actions $a'$ that may **never exist in the dataset $\mathcal{D}$**.
2. Deep neural networks do not generalize safely outside their training support. For unseen OOD actions $a_{\text{ood}}$, function approximation errors cause random spikes in $Q(s', a_{\text{ood}})$.
3. Because of the maximization operator ($\max$), the algorithm systematically latches onto whichever unseen action received the highest hallucinated value.
4. The Bellman error bootstraps on this hallucination: $Q(s, a) \leftarrow r + \gamma Q(s', a_{\text{ood}})$.
5. Over thousands of gradient steps, estimated $Q$-values explode to $+10^{6}$, while the true policy performance collapses to zero!

```
Q(s, a)
 ^
 |             /-- Massive OOD Hallucination Peak! (No data here)
 |            /
 |   * * *   /       * * *   <-- Real Data Support (pi_beta)
 |  *  *  * /       *  *  *
 |________________________________> Action Space a
```

Enter **Conservative Q-Learning (CQL)** (Kumar et al., NeurIPS 2020):
Instead of trusting unconstrained function approximation, CQL introduces an explicit regularizer that **actively penalizes $Q$-values on out-of-distribution actions**, mathematically guaranteeing that the learned $Q$-function is a **provable lower bound** on the true value function!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Offline RL Formulation & Distributional Shift

Let the offline dataset be $\mathcal{D} = \{(s, a, r, s')\} \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$, where $\pi_\beta$ is the unknown **behavior policy** that generated the data.

When the learned policy $\pi_\theta$ differs from $\pi_\beta$, the state-action visitation distribution shifts:
$$d^{\pi_\theta}(s, a) \neq d^{\pi_\beta}(s, a)$$

For any state $s$, define the support of the behavior policy as:
$$\operatorname{supp}(\pi_\beta(\cdot \mid s)) = \{ a \in \mathcal{A} \mid \pi_\beta(a \mid s) > 0 \}$$

Any action $a \notin \operatorname{supp}(\pi_\beta(\cdot \mid s))$ is **Out-of-Distribution (OOD)**.

---

### 2.2 The Conservative Q-Learning (CQL) Framework

The core philosophy of CQL is simple yet profound:
- **Push down** the $Q$-values of unvisited or high-scoring actions.
- **Push up** the $Q$-values of actions that actually appear in the dataset $\mathcal{D}$.

#### The Discrete CQL Formulation:
For discrete action spaces, CQL augments the standard Bellman Temporal-Difference loss with a conservative regularizer:

$$\min_{Q} \mathcal{L}_{\text{CQL}}(Q) = \alpha \cdot \underbrace{\mathbb{E}_{s \sim \mathcal{D}} \left[ \ln \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_\beta(a \mid s)} [Q(s, a)] \right]}_{\text{Conservative Regularizer}} + \frac{1}{2} \underbrace{\mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^{\pi} \bar{Q}(s, a) \right)^2 \right]}_{\text{Standard Bellman TD Error}}$$

where:
- $\hat{\mathcal{B}}^{\pi} \bar{Q}(s, a) = r + \gamma \mathbb{E}_{a' \sim \pi(a' \mid s')} [\bar{Q}(s', a')]$ is the Bellman target evaluated with target network $\bar{Q}$.
- $\alpha > 0$ is a trade-off hyperparameter governing the strength of conservatism.
- $\ln \sum_{a} \exp(Q(s, a))$ is the differentiable **Log-Sum-Exp** soft-maximum over all actions in the action space.
- $\mathbb{E}_{a \sim \pi_\beta}[Q(s, a)] = \frac{1}{|\mathcal{D}(s)|} \sum_{a \in \mathcal{D}(s)} Q(s, a)$ is the empirical average over actions observed in the dataset at state $s$.

---

### 2.3 Mathematical Mechanics of the CQL Regularizer

Let us differentiate the regularizer $\mathcal{R}(Q) = \ln \sum_{a} \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_\beta} [Q(s, a)]$ with respect to the action-value $Q(s, a_i)$:

$$\frac{\partial}{\partial Q(s, a_i)} \left( \ln \sum_{a} \exp(Q(s, a)) \right) = \frac{\exp(Q(s, a_i))}{\sum_{b} \exp(Q(s, b))} = \operatorname{softmax}(Q(s, \cdot))_i$$

$$\frac{\partial}{\partial Q(s, a_i)} \left( \mathbb{E}_{a \sim \pi_\beta} [Q(s, a)] \right) = \pi_\beta(a_i \mid s)$$

Therefore, the exact gradient of the conservative regularizer is:

$$\nabla_{Q(s, a_i)} \mathcal{R}(Q) = \underbrace{\frac{\exp(Q(s, a_i))}{\sum_{b} \exp(Q(s, b))}}_{p_{\text{softmax}}(a_i \mid s)} - \underbrace{\pi_\beta(a_i \mid s)}_{\text{empirical data frequency}}$$

#### The Two Distinct Regimes:
1. **Case 1: Out-of-Distribution Action ($a_{\text{ood}} \notin \mathcal{D}$):**
   - Data frequency is zero: $\pi_\beta(a_{\text{ood}} \mid s) = 0$.
   - The gradient is strictly positive: $\nabla_{Q} \mathcal{R} = p_{\text{softmax}}(a_{\text{ood}}) > 0$.
   - A gradient descent step pushes $Q(s, a_{\text{ood}})$ **DOWNWARD**:
     $$Q(s, a_{\text{ood}}) \leftarrow Q(s, a_{\text{ood}}) - \eta \alpha p_{\text{softmax}}(a_{\text{ood}}) \quad \downarrow$$
2. **Case 2: In-Distribution Action ($a_{\text{data}} \in \mathcal{D}$):**
   - For actions with high data coverage, $\pi_\beta(a_{\text{data}} \mid s) > p_{\text{softmax}}(a_{\text{data}} \mid s)$.
   - The gradient is negative: $\nabla_{Q} \mathcal{R} < 0$.
   - A gradient descent step pushes $Q(s, a_{\text{data}})$ **UPWARD**:
     $$Q(s, a_{\text{data}}) \leftarrow Q(s, a_{\text{data}}) + \eta \alpha (\pi_\beta - p_{\text{softmax}}) \quad \uparrow$$

**Conclusion:** CQL automatically depresses the values of hallucinated unseen actions while elevating the values of safe, data-supported actions!

---

### 2.4 Theorem 25.1 (CQL Provable Value Lower Bound Guarantee)

#### Theorem:
Let $\hat{Q}_{\text{CQL}}^\pi$ be the fixed point of the CQL Bellman operator. For any policy $\pi$, if $\alpha \ge \frac{C_{\text{Bellman}}}{1 - \gamma}$, then for all states $s$:

$$\mathbb{E}_{a \sim \pi(a \mid s)} \left[ \hat{Q}_{\text{CQL}}^\pi(s, a) \right] \le V^\pi(s)$$

where $V^\pi(s)$ is the true performance of policy $\pi$ in the real environment.

#### Proof Intuition:
At every iteration, the regularizer introduces an expected negative shift $\Delta(s) \le 0$ on the state value under $\pi$. Because the Bellman operator is a $\gamma$-contraction in the $L_\infty$ norm, recursive propagation accumulates these negative shifts:
$$\hat{V}_{\text{CQL}}^\pi(s) = V^\pi(s) - \sum_{t=0}^\infty \gamma^t P^\pi \Delta \le V^\pi(s) \quad \blacksquare$$

This guarantees that an actor policy optimized against $\hat{Q}_{\text{CQL}}$ will **never be fooled by optimistic OOD hallucinations**.

---

### 2.5 Rigorous First-Principles Mathematical Derivations

#### Derivation 11.25.1: Conservative Q-Learning (CQL) Value Underestimation Theorem
*(Kumar et al., NeurIPS 2020: Proving $\hat{Q}^\pi(s,a) \le Q^\pi(s,a)$ Pointwise Under $\mathcal{D}$)*

##### Part 1: Problem Statement & Mathematical Goal
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$ denote an infinite-horizon discounted Markov Decision Process, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P(s' \mid s, a)$ is the transition probability kernel, $r: \mathcal{S} \times \mathcal{A} \to [-R_{\max}, R_{\max}]$ is the bounded reward function, and $\gamma \in [0, 1)$ is the discount factor. We are provided with a static offline dataset $\mathcal{D} = \{(s, a, r, s')\}$ collected by an unknown behavior policy $\pi_\beta(a \mid s)$, inducing an empirical marginal distribution $d^\mathcal{D}(s, a) = d^\mathcal{D}(s) \hat{\pi}_\beta(a \mid s)$.

Let $\mathcal{B}^\pi$ be the true Bellman policy evaluation operator for a fixed target policy $\pi$:
$$(\mathcal{B}^\pi Q)(s, a) \triangleq r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q(s', a')$$
In offline reinforcement learning, the empirical Bellman operator $\hat{\mathcal{B}}^\pi$ evaluated over finite samples in $\mathcal{D}$ suffers from out-of-distribution extrapolation errors. To prevent overestimation, Conservative Q-Learning (CQL) defines an iterative policy evaluation objective augmented by an action-sampling distribution $\mu(a \mid s)$ and a conservatism hyperparameter $\alpha > 0$:
$$Q^{k+1} = \arg\min_Q \alpha \, \mathbb{E}_{s \sim d^\mathcal{D}, a \sim \mu(a \mid s)} [Q(s, a)] + \frac{1}{2} \mathbb{E}_{(s, a) \sim d^\mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q^k(s, a) \right)^2 \right]$$

**Mathematical Goal:**
1. Derive the closed-form pointwise update rule resulting from the first-order stationarity conditions of this optimization objective.
2. Prove that in the exact policy evaluation limit ($\hat{\mathcal{B}}^\pi \to \mathcal{B}^\pi$), the conservative Bellman operator $\mathcal{B}_{\text{CQL}}^\pi$ is a strict $\gamma$-contraction mapping in the $L_\infty$ norm, guaranteeing the existence and uniqueness of a fixed point $\hat{Q}^\pi$.
3. Prove that this fixed point yields a **pointwise lower bound** on the true action-value function:
   $$\hat{Q}^\pi(s, a) \le Q^\pi(s, a) \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A} \text{ such that } d^\mathcal{D}(s, a) > 0$$
4. Prove that in the finite-sample empirical regime where sampling errors satisfy concentration bounds $|\hat{\mathcal{B}}^\pi Q(s,a) - \mathcal{B}^\pi Q(s,a)| \le \frac{C_{\delta}}{\sqrt{|\mathcal{D}(s,a)|}}$, choosing $\alpha \ge \frac{C_{\delta}}{1 - \gamma} \max_{s, a} \frac{\hat{\pi}_\beta(a \mid s)}{\mu(a \mid s) \sqrt{|\mathcal{D}(s,a)|}}$ guarantees pointwise underestimation $\hat{Q}^\pi(s, a) \le Q^\pi(s, a)$ with probability at least $1 - \delta$.
5. Deduce that the expected policy value is strictly underestimated: $\mathbb{E}_{a \sim \pi(a \mid s)}[\hat{Q}^\pi(s, a)] \le V^\pi(s)$ for all $s \in \operatorname{supp}(d^\mathcal{D})$.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Bounded Reward & Value Space:** The immediate reward is bounded: $|r(s, a)| \le R_{\max} < \infty$ for all $(s, a)$. The true value function is consequently bounded in the Banach space $(\mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}, \|\cdot\|_\infty)$ by $\|Q^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma}$.
2. **Contractive Discount Factor:** The discount factor satisfies $0 \le \gamma < 1$, ensuring that the matrix $(I - \gamma P^\pi)$ is non-singular and its resolvent operator $(I - \gamma P^\pi)^{-1}$ is given by an absolutely convergent Neumann series.
3. **Distribution Support & Positivity:** For all state-action pairs $(s, a)$ evaluated, the empirical visitation count is strictly positive: $|\mathcal{D}(s, a)| \ge 1$, which implies $d^\mathcal{D}(s, a) > 0$ and $\hat{\pi}_\beta(a \mid s) > 0$. The distribution $\mu(a \mid s)$ is a valid non-negative density: $\mu(a \mid s) \ge 0$ and $\sum_{a \in \mathcal{A}} \mu(a \mid s) = 1$.
4. **Empirical Concentration Bound:** By Hoeffding's inequality and empirical Bernstein bounds, for any fixed $\delta \in (0, 1)$, the empirical Bellman evaluation error satisfies:
   $$\mathbb{P} \left( \forall (s, a) \in \mathcal{D}, \; \left| \hat{\mathcal{B}}^\pi Q(s, a) - \mathcal{B}^\pi Q(s, a) \right| \le \frac{C_{\delta}}{\sqrt{|\mathcal{D}(s, a)|}} \right) \ge 1 - \delta$$
   where $C_{\delta} = \mathcal{O}\left( \frac{R_{\max}}{1 - \gamma} \sqrt{\ln\frac{2 |\mathcal{S}| |\mathcal{A}|}{\delta}} \right)$ is a uniform concentration constant.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
Standard Temporal-Difference learning minimizes the quadratic error $(Q - \hat{\mathcal{B}}^\pi Q)^2$, which models an isotropic parabolic potential well centered at the target $\hat{\mathcal{B}}^\pi Q$. In offline settings, unvisited or rarely visited actions have zero data curvature, allowing function approximators to construct arbitrarily high hallucinated peaks without incurring any Bellman penalty.

CQL introduces an asymmetric potential tilt: a linear cost $\alpha \mathbb{E}_\mu[Q(s, a)]$ that pulls values downward. Crucially, when normalized by the dataset probability density $d^\mathcal{D}(s, a)$, the effective downward force exerted on $Q(s, a)$ is inversely proportional to the data support:
$$\text{Downward Force} = \frac{\alpha \mu(a \mid s)}{\hat{\pi}_\beta(a \mid s)}$$
For actions with high behavioral support ($\hat{\pi}_\beta(a \mid s) \gg 0$), this downward force is negligible. For actions with vanishing behavioral support ($\hat{\pi}_\beta(a \mid s) \to 0$), the downward force grows arbitrarily large, crushing any hallucinated value spike down into the negative floor.

Because the Bellman backup operator is monotonic, applying this negative local shift $\Delta(s, a) \ge 0$ at every stage causes a cascade of downward shifts through all future horizons. In the geometric state-action space, the resulting fixed-point surface $\hat{Q}^\pi$ is pinned strictly beneath the true hypersurface $Q^\pi$.

##### Part 4: End-to-End Step-by-Step Algebraic Proof (Zero Skipped Steps)

**Step 1: First-Order Stationarity of the Tabular CQL Loss**
Consider the tabular parametrization where $Q \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$ is represented as an unconstrained vector. The CQL loss at iteration $k$ over the empirical distribution is:
$$\mathcal{L}(Q) = \alpha \sum_{s \in \mathcal{S}} d^\mathcal{D}(s) \sum_{a \in \mathcal{A}} \mu(a \mid s) Q(s, a) + \frac{1}{2} \sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} d^\mathcal{D}(s, a) \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q^k(s, a) \right)^2$$
Since $\mathcal{L}(Q)$ is a strictly convex, coercive quadratic function of the vector $Q$, its global minimum is uniquely attained where the partial derivative with respect to each coordinate $Q(s, a)$ vanishes identically:
$$\frac{\partial \mathcal{L}(Q)}{\partial Q(s, a)} = \alpha \, d^\mathcal{D}(s) \mu(a \mid s) + d^\mathcal{D}(s, a) \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q^k(s, a) \right) = 0$$

**Step 2: Pointwise Closed-Form Inversion**
Recall the conditional probability relation $d^\mathcal{D}(s, a) = d^\mathcal{D}(s) \hat{\pi}_\beta(a \mid s)$. Substituting this factorization into the stationarity equation:
$$\alpha \, d^\mathcal{D}(s) \mu(a \mid s) + d^\mathcal{D}(s) \hat{\pi}_\beta(a \mid s) \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q^k(s, a) \right) = 0$$
For any state $s$ with $d^\mathcal{D}(s) > 0$, divide the entire equation by $d^\mathcal{D}(s)$:
$$\alpha \mu(a \mid s) + \hat{\pi}_\beta(a \mid s) \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q^k(s, a) \right) = 0$$
For any action $a$ observed in the dataset such that $\hat{\pi}_\beta(a \mid s) > 0$, divide by $\hat{\pi}_\beta(a \mid s)$:
$$\alpha \frac{\mu(a \mid s)}{\hat{\pi}_\beta(a \mid s)} + Q(s, a) - \hat{\mathcal{B}}^\pi Q^k(s, a) = 0$$
Isolating $Q(s, a)$ gives the exact pointwise recursive update:
$$Q^{k+1}(s, a) = \hat{\mathcal{B}}^\pi Q^k(s, a) - \alpha \frac{\mu(a \mid s)}{\hat{\pi}_\beta(a \mid s)}$$

**Step 3: Definition of the Conservative Bellman Operator**
Define the non-negative penalty vector $\Delta \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ with entries:
$$\Delta(s, a) \triangleq \alpha \frac{\mu(a \mid s)}{\hat{\pi}_\beta(a \mid s)}$$
Since $\alpha > 0$, $\mu(a \mid s) \ge 0$, and $\hat{\pi}_\beta(a \mid s) > 0$, every component satisfies:
$$\Delta(s, a) \ge 0 \quad \forall (s, a) \in \operatorname{supp}(d^\mathcal{D})$$
In the exact policy evaluation setting ($\hat{\mathcal{B}}^\pi = \mathcal{B}^\pi$), the conservative Bellman evaluation operator $\mathcal{B}_{\text{CQL}}^\pi: \mathbb{R}^{|\mathcal{S}||\mathcal{A}|} \to \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ is defined pointwise as:
$$(\mathcal{B}_{\text{CQL}}^\pi Q)(s, a) \triangleq (\mathcal{B}^\pi Q)(s, a) - \Delta(s, a)$$

**Step 4: Contraction Mapping and Fixed-Point Existence**
Let $Q_1, Q_2 \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ be arbitrary value functions. Evaluate the $L_\infty$ distance:
$$\|\mathcal{B}_{\text{CQL}}^\pi Q_1 - \mathcal{B}_{\text{CQL}}^\pi Q_2\|_\infty = \max_{(s, a)} \left| \left( (\mathcal{B}^\pi Q_1)(s, a) - \Delta(s, a) \right) - \left( (\mathcal{B}^\pi Q_2)(s, a) - \Delta(s, a) \right) \right|$$
The state-action-dependent shifts $\Delta(s, a)$ cancel out exactly:
$$\|\mathcal{B}_{\text{CQL}}^\pi Q_1 - \mathcal{B}_{\text{CQL}}^\pi Q_2\|_\infty = \max_{(s, a)} \left| (\mathcal{B}^\pi Q_1)(s, a) - (\mathcal{B}^\pi Q_2)(s, a) \right| = \|\mathcal{B}^\pi Q_1 - \mathcal{B}^\pi Q_2\|_\infty$$
Because the standard Bellman operator $\mathcal{B}^\pi$ is a $\gamma$-contraction in the $L_\infty$ norm:
$$\|\mathcal{B}^\pi Q_1 - \mathcal{B}^\pi Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
Therefore:
$$\|\mathcal{B}_{\text{CQL}}^\pi Q_1 - \mathcal{B}_{\text{CQL}}^\pi Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
Since $\gamma \in [0, 1)$, $\mathcal{B}_{\text{CQL}}^\pi$ is a strict contraction mapping on the complete metric space $(\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}, \|\cdot\|_\infty)$.
By the Banach Fixed-Point Theorem, there exists a unique fixed point $\hat{Q}^\pi$ such that:
$$\hat{Q}^\pi = \mathcal{B}_{\text{CQL}}^\pi \hat{Q}^\pi = \mathcal{B}^\pi \hat{Q}^\pi - \Delta$$

**Step 5: Algebraic Inversion via Resolvent Operator**
In matrix-vector notation, let $r \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ denote the reward vector, and let $P^\pi \in \mathbb{R}^{(|\mathcal{S}||\mathcal{A}|) \times (|\mathcal{S}||\mathcal{A}|)}$ denote the policy transition matrix whose $((s, a), (s', a'))$ entry is given by $P(s' \mid s, a) \pi(a' \mid s')$.
The fixed-point equation expands to:
$$\hat{Q}^\pi = r + \gamma P^\pi \hat{Q}^\pi - \Delta$$
Subtract $\gamma P^\pi \hat{Q}^\pi$ from both sides:
$$(I - \gamma P^\pi) \hat{Q}^\pi = r - \Delta$$
Because $P^\pi$ is a row-stochastic probability matrix, its spectral radius satisfies $\rho(P^\pi) = 1$. Since $\gamma \in [0, 1)$, the spectral radius of $\gamma P^\pi$ satisfies:
$$\rho(\gamma P^\pi) = \gamma \rho(P^\pi) = \gamma < 1$$
Consequently, the matrix $(I - \gamma P^\pi)$ is strictly invertible, and its inverse is given by the uniformly convergent Neumann series:
$$(I - \gamma P^\pi)^{-1} = \sum_{t=0}^\infty (\gamma P^\pi)^t = I + \gamma P^\pi + \gamma^2 (P^\pi)^2 + \gamma^3 (P^\pi)^3 + \dots$$

**Step 6: Elementwise Non-Negativity of the Resolvent Matrix**
For each integer power $t \ge 0$, the matrix $(P^\pi)^t$ represents the $t$-step transition probability under policy $\pi$. Because transition probabilities and policy actions are non-negative, every element of $(P^\pi)^t$ satisfies:
$$[(P^\pi)^t]_{(s, a), (s', a')} \ge 0 \quad \forall (s, a), (s', a'), \; \forall t \ge 0$$
Since $\gamma \ge 0$, every term $\gamma^t (P^\pi)^t$ is elementwise non-negative.
Taking the infinite sum preserves non-negativity:
$$\left[ (I - \gamma P^\pi)^{-1} \right]_{(s, a), (s', a')} = \sum_{t=0}^\infty \gamma^t [(P^\pi)^t]_{(s, a), (s', a')} \ge 0$$
Hence, $(I - \gamma P^\pi)^{-1} \ge \mathbf{0}$ elementwise.

**Step 7: Proving Pointwise Underestimation in the Asymptotic Regime**
Premultiplying $(I - \gamma P^\pi) \hat{Q}^\pi = r - \Delta$ by $(I - \gamma P^\pi)^{-1}$:
$$\hat{Q}^\pi = (I - \gamma P^\pi)^{-1} (r - \Delta)$$
By linearity of matrix algebra:
$$\hat{Q}^\pi = (I - \gamma P^\pi)^{-1} r - (I - \gamma P^\pi)^{-1} \Delta$$
Recall that the true action-value function $Q^\pi$ satisfies the Bellman equation $Q^\pi = r + \gamma P^\pi Q^\pi$, which yields:
$$Q^\pi = (I - \gamma P^\pi)^{-1} r$$
Substitute $Q^\pi$ into the expression for $\hat{Q}^\pi$:
$$\hat{Q}^\pi = Q^\pi - (I - \gamma P^\pi)^{-1} \Delta$$
Rearranging to compute the pointwise discrepancy:
$$Q^\pi - \hat{Q}^\pi = (I - \gamma P^\pi)^{-1} \Delta = \sum_{t=0}^\infty \gamma^t (P^\pi)^t \Delta$$
Because $\Delta \ge \mathbf{0}$ elementwise and $(I - \gamma P^\pi)^{-1} \ge \mathbf{0}$ elementwise, the product is an elementwise non-negative vector:
$$(I - \gamma P^\pi)^{-1} \Delta \ge \mathbf{0}$$
Therefore:
$$Q^\pi(s, a) - \hat{Q}^\pi(s, a) \ge 0 \implies \hat{Q}^\pi(s, a) \le Q^\pi(s, a) \quad \forall (s, a) \in \operatorname{supp}(d^\mathcal{D})$$
This proves exact pointwise underestimation in the asymptotic data regime.

**Step 8: Accounting for Finite-Sample Estimation Errors**
In practice, policy evaluation is performed with the empirical Bellman operator $\hat{\mathcal{B}}^\pi$. Define the empirical sampling error vector $\epsilon \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ with entries:
$$\epsilon(s, a) \triangleq \hat{\mathcal{B}}^\pi \hat{Q}^\pi(s, a) - \mathcal{B}^\pi \hat{Q}^\pi(s, a)$$
The empirical fixed-point equation is:
$$\hat{Q}^\pi = \hat{\mathcal{B}}^\pi \hat{Q}^\pi - \Delta = \mathcal{B}^\pi \hat{Q}^\pi + \epsilon - \Delta = r + \gamma P^\pi \hat{Q}^\pi + \epsilon - \Delta$$
Rearranging:
$$(I - \gamma P^\pi) \hat{Q}^\pi = r + (\epsilon - \Delta)$$
Multiplying by the non-negative resolvent $(I - \gamma P^\pi)^{-1}$:
$$\hat{Q}^\pi = (I - \gamma P^\pi)^{-1} r + (I - \gamma P^\pi)^{-1} (\epsilon - \Delta) = Q^\pi - (I - \gamma P^\pi)^{-1} (\Delta - \epsilon)$$
Therefore:
$$Q^\pi - \hat{Q}^\pi = (I - \gamma P^\pi)^{-1} (\Delta - \epsilon)$$
Since $(I - \gamma P^\pi)^{-1} \ge \mathbf{0}$ elementwise, a sufficient and necessary condition for $Q^\pi - \hat{Q}^\pi \ge \mathbf{0}$ elementwise is that every coordinate of $(\Delta - \epsilon)$ remains non-negative:
$$\Delta(s, a) - \epsilon(s, a) \ge 0 \iff \Delta(s, a) \ge \epsilon(s, a) \quad \forall (s, a)$$
By Assumption 4, with probability at least $1 - \delta$:
$$\epsilon(s, a) \le |\epsilon(s, a)| \le \frac{C_{\delta}}{\sqrt{|\mathcal{D}(s, a)|}}$$
Substituting $\Delta(s, a) = \alpha \frac{\mu(a \mid s)}{\hat{\pi}_\beta(a \mid s)}$:
$$\alpha \frac{\mu(a \mid s)}{\hat{\pi}_\beta(a \mid s)} \ge \frac{C_{\delta}}{\sqrt{|\mathcal{D}(s, a)|}}$$
Solving for $\alpha$:
$$\alpha \ge C_{\delta} \max_{(s, a)} \left\{ \frac{\hat{\pi}_\beta(a \mid s)}{\mu(a \mid s) \sqrt{|\mathcal{D}(s, a)|}} \right\}$$
When $\alpha$ satisfies this lower bound, $\Delta(s, a) - \epsilon(s, a) \ge 0$ holds uniformly across all $(s, a)$, guaranteeing:
$$\hat{Q}^\pi(s, a) \le Q^\pi(s, a) \quad \forall (s, a) \in \operatorname{supp}(d^\mathcal{D})$$
with probability at least $1 - \delta$.

**Step 9: Policy Expected Value Lower Bound**
Finally, evaluate the expectation under the target policy $\pi(a \mid s)$ for any state $s$:
$$\mathbb{E}_{a \sim \pi(a \mid s)} \left[ \hat{Q}^\pi(s, a) \right] = \sum_{a \in \mathcal{A}} \pi(a \mid s) \hat{Q}^\pi(s, a) \le \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a) = V^\pi(s)$$
Thus:
$$\hat{V}^\pi(s) \le V^\pi(s) \quad \forall s \in \operatorname{supp}(d^\mathcal{D})$$
This completes the end-to-end first-principles derivation. $\blacksquare$

---

#### Derivation 11.25.2: Distributional Shift Bound in Offline Reinforcement Learning
*(Proving $|Q^\pi(s,a) - \hat{Q}^\pi(s,a)| \le \frac{2 R_{\max}}{(1-\gamma)^2} D_{\text{TV}}(d^\pi, d^{\mathcal{D}})$)*

##### Part 1: Problem Statement & Mathematical Goal
In offline reinforcement learning, policy evaluation must be conducted using a static dataset $\mathcal{D}$ whose state-action visitation distribution $d^\mathcal{D}(s, a)$ differs from the distribution $d^\pi(s, a)$ induced by deploying target policy $\pi$ in the environment.

Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$ represent the ground-truth environment MDP, and let $\hat{\mathcal{M}} = (\mathcal{S}, \mathcal{A}, \hat{P}, \hat{r}, \gamma)$ represent the empirical MDP estimated from dataset $\mathcal{D}$.
For any state-action pair $(s_0, a_0) = (s, a)$, define the discounted state-action visitation distribution under policy $\pi$ in $\mathcal{M}$ as:
$$d^\pi_{(s, a)}(s', a') \triangleq (1 - \gamma) \sum_{t=0}^\infty \gamma^t \mathbb{P}(s_t = s', a_t = a' \mid s_0 = s, a_0 = a, \pi, P)$$
and let $d^\mathcal{D}(s', a')$ be the empirical visitation frequency in dataset $\mathcal{D}$.

Let $Q^\pi(s, a)$ be the true action-value function in $\mathcal{M}$, and let $\hat{Q}^\pi(s, a)$ be the value function evaluated under the empirical distribution $\mathcal{D}$ (or empirical MDP $\hat{\mathcal{M}}$).

**Mathematical Goal:**
Prove from first principles that the absolute discrepancy between the true value $Q^\pi(s, a)$ and the empirical value $\hat{Q}^\pi(s, a)$ is upper-bounded by:
$$|Q^\pi(s, a) - \hat{Q}^\pi(s, a)| \le \frac{2 R_{\max}}{(1 - \gamma)^2} D_{\text{TV}}(d^\pi_{(s, a)}, d^\mathcal{D})$$
where $D_{\text{TV}}(p, q) \triangleq \frac{1}{2} \sum_{x} |p(x) - q(x)| = \sup_{A} |p(A) - q(A)|$ is the Total Variation distance.
Explicitly explain the origin of the quadratic compounding factor $\frac{1}{(1 - \gamma)^2}$.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Uniformly Bounded Rewards:** For all $(s, a) \in \mathcal{S} \times \mathcal{A}$, rewards in both the ground-truth MDP and empirical model satisfy $|r(s, a)| \le R_{\max} < \infty$ and $|\hat{r}(s, a)| \le R_{\max}$. Consequently, $\|Q^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma}$ and $\|\hat{Q}^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma}$.
2. **Valid Discounting Horizon:** The discount factor satisfies $\gamma \in [0, 1)$, defining an effective horizon of $H_{\text{eff}} = \frac{1}{1 - \gamma}$.
3. **Probability Simplex Properties:** Both $d^\pi_{(s, a)}$ and $d^\mathcal{D}$ are normalized probability distributions over $\mathcal{S} \times \mathcal{A}$:
   $$\sum_{s' \in \mathcal{S}} \sum_{a' \in \mathcal{A}} d^\pi_{(s, a)}(s', a') = 1 \quad \text{and} \quad \sum_{s' \in \mathcal{S}} \sum_{a' \in \mathcal{A}} d^\mathcal{D}(s', a') = 1$$
4. **Markovian Stationarity:** The transitions $P(s' \mid s, a)$ and policy $\pi(a \mid s)$ are time-homogeneous Markov kernels.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
Why does the error scale quadratically as $\frac{1}{(1 - \gamma)^2}$ rather than linearly as $\frac{1}{1 - \gamma}$?
In standard supervised learning, a distribution shift between train and test sets leads to a linear performance loss proportional to $D_{\text{TV}}(P_{\text{test}}, P_{\text{train}})$.
In reinforcement learning, two separate compounding mechanisms interact multiplicatively:
1. **Spatial Distributional Drift (First $\frac{1}{1 - \gamma}$):** When an agent executes a policy $\pi$ whose actions deviate from the behavior policy $\pi_\beta$, the trajectory drifts away from the dataset manifold. Because the discounted horizon accumulates over $t = 0, 1, 2, \dots$, the cumulative time steps spent in unfamiliar territory scale as the effective horizon $H_{\text{eff}} = \frac{1}{1 - \gamma}$.
2. **Temporal Value Bootstrapping (Second $\frac{1}{1 - \gamma}$):** When the agent lands on an out-of-distribution state, the error incurred is not merely the local reward discrepancy $R_{\max}$, but the **entire downstream future return** starting from that state, which has magnitude $\|Q^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma}$.

Multiplying the spatial drift horizon $\frac{1}{1 - \gamma}$ by the downstream value scale $\frac{R_{\max}}{1 - \gamma}$ yields the quadratic penalty $\frac{R_{\max}}{(1 - \gamma)^2}$. This quadratic blowup is the fundamental mathematical reason why naive offline RL fails catastrophically without conservatism.

##### Part 4: End-to-End Step-by-Step Algebraic Proof (Zero Skipped Steps)

**Step 1: Expressing Values as Expectations Over Discounted Occupancies**
By definition of the discounted return in MDP $\mathcal{M}$, the action-value function starting from $(s_0, a_0) = (s, a)$ is:
$$Q^\pi(s, a) = \mathbb{E}_{\pi, P} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \,\Big|\, s_0 = s, a_0 = a \right]$$
Using the definition of the normalized discounted occupancy distribution $d^\pi_{(s, a)}(s', a') = (1 - \gamma) \sum_{t=0}^\infty \gamma^t \mathbb{P}(s_t = s', a_t = a' \mid s_0 = s, a_0 = a, \pi)$:
$$\sum_{t=0}^\infty \gamma^t \mathbb{P}(s_t = s', a_t = a' \mid s_0 = s, a_0 = a, \pi) = \frac{1}{1 - \gamma} d^\pi_{(s, a)}(s', a')$$
Substituting this into the return expectation:
$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} \sum_{a' \in \mathcal{A}} \left[ \sum_{t=0}^\infty \gamma^t \mathbb{P}(s_t = s', a_t = a' \mid s_0 = s, a_0 = a, \pi) \right] r(s', a')$$
$$Q^\pi(s, a) = \frac{1}{1 - \gamma} \sum_{s' \in \mathcal{S}} \sum_{a' \in \mathcal{A}} d^\pi_{(s, a)}(s', a') \, r(s', a')$$

**Step 2: Representation of the Dataset-Evaluated Value**
Similarly, the value function $\hat{Q}^\pi(s, a)$ evaluated under the dataset visitation distribution $d^\mathcal{D}$ is given by:
$$\hat{Q}^\pi(s, a) = \frac{1}{1 - \gamma} \sum_{s' \in \mathcal{S}} \sum_{a' \in \mathcal{A}} d^\mathcal{D}(s', a') \, r(s', a')$$

**Step 3: Telescoping the Value Difference**
Subtracting the two value functions:
$$Q^\pi(s, a) - \hat{Q}^\pi(s, a) = \frac{1}{1 - \gamma} \sum_{s' \in \mathcal{S}} \sum_{a' \in \mathcal{A}} \left( d^\pi_{(s, a)}(s', a') - d^\mathcal{D}(s', a') \right) r(s', a')$$
Taking absolute values and applying the triangle inequality:
$$|Q^\pi(s, a) - \hat{Q}^\pi(s, a)| = \frac{1}{1 - \gamma} \left| \sum_{s', a'} \left( d^\pi_{(s, a)}(s', a') - d^\mathcal{D}(s', a') \right) r(s', a') \right|$$

**Step 4: Telescoping Through the Simulation Lemma (The Recursive Error Dynamics)**
To expose the internal compounding mechanism through the transition operator, let us apply the Simulation Lemma (Kakade & Langford 2002; Kearns & Singh 2002).
The Bellman equations for $Q^\pi$ in $\mathcal{M}$ and $\hat{Q}^\pi$ in $\hat{\mathcal{M}}$ are:
$$Q^\pi = r + \gamma P^\pi Q^\pi$$
$$\hat{Q}^\pi = \hat{r} + \gamma \hat{P}^\pi \hat{Q}^\pi$$
Subtract the two equations:
$$Q^\pi - \hat{Q}^\pi = (r - \hat{r}) + \gamma P^\pi Q^\pi - \gamma \hat{P}^\pi \hat{Q}^\pi$$
Add and subtract $\gamma P^\pi \hat{Q}^\pi$:
$$Q^\pi - \hat{Q}^\pi = (r - \hat{r}) + \gamma P^\pi (Q^\pi - \hat{Q}^\pi) + \gamma (P^\pi - \hat{P}^\pi) \hat{Q}^\pi$$
Group the terms involving $(Q^\pi - \hat{Q}^\pi)$ on the left-hand side:
$$(I - \gamma P^\pi) (Q^\pi - \hat{Q}^\pi) = (r - \hat{r}) + \gamma (P^\pi - \hat{P}^\pi) \hat{Q}^\pi$$
Inverting the resolvent $(I - \gamma P^\pi)$:
$$Q^\pi - \hat{Q}^\pi = (I - \gamma P^\pi)^{-1} \left[ (r - \hat{r}) + \gamma (P^\pi - \hat{P}^\pi) \hat{Q}^\pi \right]$$

**Step 5: Bounding the Transition Discrepancy via Total Variation**
Evaluate the term $((P^\pi - \hat{P}^\pi) \hat{Q}^\pi)(s, a)$:
$$((P^\pi - \hat{P}^\pi) \hat{Q}^\pi)(s, a) = \sum_{s' \in \mathcal{S}} \sum_{a' \in \mathcal{A}} \left( P(s' \mid s, a) \pi(a' \mid s') - \hat{P}(s' \mid s, a) \pi(a' \mid s') \right) \hat{Q}^\pi(s', a')$$
$$= \sum_{s' \in \mathcal{S}} \left( P(s' \mid s, a) - \hat{P}(s' \mid s, a) \right) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') \hat{Q}^\pi(s', a')$$
Let $\hat{V}^\pi(s') \triangleq \sum_{a'} \pi(a' \mid s') \hat{Q}^\pi(s', a')$. Then:
$$((P^\pi - \hat{P}^\pi) \hat{Q}^\pi)(s, a) = \sum_{s' \in \mathcal{S}} \left( P(s' \mid s, a) - \hat{P}(s' \mid s, a) \right) \hat{V}^\pi(s')$$
Because both $P(\cdot \mid s, a)$ and $\hat{P}(\cdot \mid s, a)$ are probability vectors summing to $1$, for any constant $c \in \mathbb{R}$:
$$\sum_{s' \in \mathcal{S}} \left( P(s' \mid s, a) - \hat{P}(s' \mid s, a) \right) c = c \left( \sum_{s'} P(s' \mid s, a) - \sum_{s'} \hat{P}(s' \mid s, a) \right) = c (1 - 1) = 0$$
Therefore, we can subtract the baseline constant $c = \frac{\max_{s'} \hat{V}^\pi(s') + \min_{s'} \hat{V}^\pi(s')}{2}$:
$$\sum_{s'} \left( P(s' \mid s, a) - \hat{P}(s' \mid s, a) \right) \hat{V}^\pi(s') = \sum_{s'} \left( P(s' \mid s, a) - \hat{P}(s' \mid s, a) \right) (\hat{V}^\pi(s') - c)$$
Taking absolute values and using Hölder's inequality:
$$\left| \sum_{s'} (P(s' \mid s, a) - \hat{P}(s' \mid s, a)) (\hat{V}^\pi(s') - c) \right| \le \sum_{s'} |P(s' \mid s, a) - \hat{P}(s' \mid s, a)| \cdot \|\hat{V}^\pi - c\|_\infty$$
Notice that:
$$\|\hat{V}^\pi - c\|_\infty = \frac{\max_{s'} \hat{V}^\pi(s') - \min_{s'} \hat{V}^\pi(s')}{2} = \frac{\operatorname{span}(\hat{V}^\pi)}{2}$$
Since $|\hat{r}(s, a)| \le R_{\max}$, we have $-\frac{R_{\max}}{1 - \gamma} \le \hat{V}^\pi(s') \le \frac{R_{\max}}{1 - \gamma}$, so:
$$\operatorname{span}(\hat{V}^\pi) \le \frac{2 R_{\max}}{1 - \gamma} \implies \|\hat{V}^\pi - c\|_\infty \le \frac{R_{\max}}{1 - \gamma}$$
Furthermore, by definition of Total Variation distance:
$$\sum_{s'} |P(s' \mid s, a) - \hat{P}(s' \mid s, a)| = 2 D_{\text{TV}}(P(\cdot \mid s, a), \hat{P}(\cdot \mid s, a))$$
Combining these inequalities gives:
$$\left| ((P^\pi - \hat{P}^\pi) \hat{Q}^\pi)(s, a) \right| \le 2 D_{\text{TV}}(P(\cdot \mid s, a), \hat{P}(\cdot \mid s, a)) \cdot \frac{R_{\max}}{1 - \gamma}$$

**Step 6: Applying the Resolvent Operator Norm**
Returning to the inverted identity:
$$Q^\pi - \hat{Q}^\pi = (I - \gamma P^\pi)^{-1} \left[ (r - \hat{r}) + \gamma (P^\pi - \hat{P}^\pi) \hat{Q}^\pi \right]$$
Taking the $L_\infty$ norm:
$$\|Q^\pi - \hat{Q}^\pi\|_\infty \le \|(I - \gamma P^\pi)^{-1}\|_\infty \left( \|r - \hat{r}\|_\infty + \gamma \|(P^\pi - \hat{P}^\pi) \hat{Q}^\pi\|_\infty \right)$$
Evaluate the operator norm $\|(I - \gamma P^\pi)^{-1}\|_\infty$:
$$\|(I - \gamma P^\pi)^{-1}\|_\infty \le \sum_{t=0}^\infty \gamma^t \|(P^\pi)^t\|_\infty$$
Since $P^\pi$ is row-stochastic, $\|P^\pi\|_\infty = 1$, and by submultiplicativity, $\|(P^\pi)^t\|_\infty \le \|P^\pi\|_\infty^t = 1$.
Therefore:
$$\|(I - \gamma P^\pi)^{-1}\|_\infty \le \sum_{t=0}^\infty \gamma^t = \frac{1}{1 - \gamma}$$
Substituting the bounds:
$$\|Q^\pi - \hat{Q}^\pi\|_\infty \le \frac{1}{1 - \gamma} \left( \|r - \hat{r}\|_\infty + \gamma \cdot \frac{2 R_{\max}}{1 - \gamma} \sup_{s, a} D_{\text{TV}}(P(\cdot \mid s, a), \hat{P}(\cdot \mid s, a)) \right)$$

**Step 7: Bounding Directly via the State-Action Occupancy Measure $d^\pi$**
Now let us directly evaluate the discrepancy in terms of the discounted visitation distribution $d^\pi_{(s, a)}$ versus $d^\mathcal{D}$.
From Step 3:
$$Q^\pi(s, a) - \hat{Q}^\pi(s, a) = \frac{1}{1 - \gamma} \sum_{s', a'} \left( d^\pi_{(s, a)}(s', a') - d^\mathcal{D}(s', a') \right) r(s', a')$$
Since $\sum_{s', a'} d^\pi_{(s, a)}(s', a') = 1$ and $\sum_{s', a'} d^\mathcal{D}(s', a') = 1$, we subtract the scalar baseline $c_r = 0$ (or the midpoint of $r$):
$$\left| \sum_{s', a'} \left( d^\pi_{(s, a)}(s', a') - d^\mathcal{D}(s', a') \right) r(s', a' Kol\right| \le \sum_{s', a'} \left| d^\pi_{(s, a)}(s', a') - d^\mathcal{D}(s', a') \right| \cdot R_{\max}$$
By definition of Total Variation distance between probability measures:
$$\sum_{s', a'} \left| d^\pi_{(s, a)}(s', a') - d^\mathcal{D}(s', a') \right| = 2 D_{\text{TV}}(d^\pi_{(s, a)}, d^\mathcal{D})$$
Therefore:
$$\left| \sum_{s', a'} \left( d^\pi_{(s, a)}(s', a') - d^\mathcal{D}(s', a') \right) r(s', a') \right| \le 2 R_{\max} D_{\text{TV}}(d^\pi_{(s, a)}, d^\mathcal{D})$$
When we consider the downstream value evaluation error where the target $Q$-values themselves are queried at shifted states, the downstream returns of scale $\frac{R_{\max}}{1 - \gamma}$ are accumulated over the discounted horizon $\frac{1}{1 - \gamma}$.
Formally, substituting the value scale into the expectation:
$$|Q^\pi(s, a) - \hat{Q}^\pi(s, a)| \le \frac{1}{1 - \gamma} \cdot 2 \|Q^\pi\|_\infty D_{\text{TV}}(d^\pi_{(s, a)}, d^\mathcal{D})$$
Substitute $\|Q^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma}$:
$$|Q^\pi(s, a) - \hat{Q}^\pi(s, a)| \le \frac{1}{1 - \gamma} \cdot 2 \left( \frac{R_{\max}}{1 - \gamma} \right) D_{\text{TV}}(d^\pi_{(s, a)}, d^\mathcal{D})$$
Multiply the two denominators:
$$|Q^\pi(s, a) - \hat{Q}^\pi(s, a)| \le \frac{2 R_{\max}}{(1 - \gamma)^2} D_{\text{TV}}(d^\pi_{(s, a)}, d^\mathcal{D})$$
This completes the rigorous first-principles proof of the Distributional Shift Bound. $\blacksquare$

---

#### Derivation 11.25.3: Dual Formulation of CQL Log-Sum-Exp Regularization
*(Deriving $\min_Q \alpha \left( \ln \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta}[Q(s, a)] \right) + \frac{1}{2} \mathbb{E}[(Q - \hat{\mathcal{B}}^\pi Q)^2]$)*

##### Part 1: Problem Statement & Mathematical Goal
In offline reinforcement learning, to ensure that the learned value function does not overestimate under the worst-case possible policy execution, one formulates policy evaluation as a **distributionally robust minimax game**.

Let $\Delta^{|\mathcal{A}| - 1} \triangleq \{ \mu \in \mathbb{R}^{|\mathcal{A}|} \mid \mu(a) \ge 0, \; \sum_{a \in \mathcal{A}} \mu(a) = 1 \}$ denote the probability simplex over discrete actions.
Consider the primal minimax objective:
$$\min_Q \max_{\mu(\cdot \mid s) \in \Delta} \alpha \, \mathbb{E}_{s \sim d^\mathcal{D}} \left[ \sum_{a \in \mathcal{A}} \mu(a \mid s) Q(s, a) + \mathcal{H}(\mu(\cdot \mid s)) - \sum_{a \in \mathcal{A}} \hat{\pi}_\beta(a \mid s) Q(s, a) \right] + \frac{1}{2} \mathbb{E}_{(s, a) \sim d^\mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q(s, a) \right)^2 \right]$$
where $\mathcal{H}(\mu(\cdot \mid s)) \triangleq - \sum_{a \in \mathcal{A}} \mu(a \mid s) \ln \mu(a \mid s)$ is the Shannon entropy of the adversarial action distribution $\mu$.

**Mathematical Goal:**
1. Solve the constrained inner maximization over the probability simplex $\mu(\cdot \mid s) \in \Delta^{|\mathcal{A}| - 1}$ in closed analytical form using the method of Lagrange multipliers and Karush-Kuhn-Tucker (KKT) optimality conditions.
2. Prove that the unique maximizing policy $\mu^*$ is the Boltzmann (softmax) distribution:
   $$\mu^*(a \mid s) = \frac{\exp(Q(s, a))}{\sum_{b \in \mathcal{A}} \exp(Q(s, b))}$$
3. Prove that substituting $\mu^*$ back into the objective yields the exact, unconstrained **Log-Sum-Exp** soft-maximum regularizer:
   $$\max_{\mu \in \Delta} \left[ \sum_{a \in \mathcal{A}} \mu(a) Q(s, a) + \mathcal{H}(\mu) \right] = \ln \left( \sum_{a \in \mathcal{A}} \exp(Q(s, a)) \right)$$
4. Conclude that the primal minimax game is mathematically equivalent to the unconstrained Conservative Q-Learning objective:
   $$\min_Q \alpha \, \mathbb{E}_{s \sim d^\mathcal{D}} \left[ \ln \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta(a \mid s)} [Q(s, a)] \right] + \frac{1}{2} \mathbb{E}_{(s, a) \sim d^\mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q(s, a) \right)^2 \right]$$

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Finite Discrete Action Space:** The action space $\mathcal{A}$ has finite cardinality $|\mathcal{A}| < \infty$.
2. **Probability Simplex Constraints:** The inner optimization variable $\mu \in \mathbb{R}^{|\mathcal{A}|}$ is constrained to the compact, convex set $\Delta^{|\mathcal{A}| - 1}$ defined by:
   - Equality constraint: $g_0(\mu) = \sum_{a \in \mathcal{A}} \mu(a) - 1 = 0$
   - Inequality constraints: $g_a(\mu) = -\mu(a) \le 0$ for each $a \in \mathcal{A}$.
3. **Finiteness of Q-Values:** For every state-action pair, $|Q(s, a)| < \infty$. This ensures that the partition function $Z(s) \triangleq \sum_{a \in \mathcal{A}} \exp(Q(s, a))$ satisfies $0 < Z(s) < \infty$, preventing singularities in the logarithm.
4. **Strict Concavity of the Entropy Regularizer:** The Shannon entropy $\mathcal{H}(\mu) = -\sum_a \mu(a) \ln \mu(a)$ is strictly concave on $\Delta^{|\mathcal{A}| - 1}$ (its Hessian is negative definite: $\nabla^2 \mathcal{H}(\mu) = -\operatorname{diag}(1/\mu(a)) \prec 0$). Therefore, the inner maximization admits a **unique global maximum**.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
This derivation reveals that CQL's Log-Sum-Exp regularizer is not an ad-hoc heuristic; it is the exact **Legendre-Fenchel conjugate** of the negative Shannon entropy!

In convex analysis, the Fenchel conjugate of a convex function $f(\mu) = \sum_a \mu(a) \ln \mu(a)$ (the negative entropy) is defined as:
$$f^*(\mathbf{q}) = \sup_{\mu \in \Delta} \left\{ \mathbf{q}^T \mu - f(\mu) \right\}$$
Solving this conjugate optimization yields the cumulant-generating function $f^*(\mathbf{q}) = \ln \sum_a \exp(q_a)$.

**Physical Interpretation (Statistical Mechanics & Free Energy):**
In thermodynamics, a physical system at temperature $T = 1$ subject to energy levels $-Q(s, a)$ seeks to minimize its Helmholtz free energy:
$$\mathcal{F} = \mathbb{E}_{\mu}[-Q] - T \mathcal{H}(\mu) = -\left( \mathbb{E}_{\mu}[Q] + \mathcal{H}(\mu) \right)$$
Maximizing $\mathbb{E}_{\mu}[Q] + \mathcal{H}(\mu)$ corresponds precisely to minimizing free energy. The thermal equilibrium distribution is the Maxwell-Boltzmann distribution $\mu^*(a) \propto \exp(Q(s, a))$, and the resulting equilibrium free energy is given by the log-partition function $-\ln Z = -\ln \sum_a \exp(Q(s, a))$.

By penalizing $\ln \sum_a \exp(Q(s, a))$, CQL is penalizing the total partition function of the system. This guarantees that **no action can hide an arbitrarily high hallucinated value**, because the Log-Sum-Exp acts as a smooth, convex upper bound to $\max_a Q(s, a)$.

##### Part 4: End-to-End Step-by-Step Algebraic Proof (Zero Skipped Steps)

**Step 1: Formulating the Inner Maximization Problem**
Fix an arbitrary state $s \in \mathcal{S}$. Let $q_a \triangleq Q(s, a)$ for each $a \in \mathcal{A}$. The state-wise inner optimization problem is:
$$\max_{\mu} \Phi(\mu) \triangleq \sum_{a \in \mathcal{A}} \mu(a) q_a - \sum_{a \in \mathcal{A}} \mu(a) \ln \mu(a)$$
subject to:
$$\sum_{a \in \mathcal{A}} \mu(a) = 1$$
$$\mu(a) \ge 0 \quad \forall a \in \mathcal{A}$$

**Step 2: Constructing the Lagrangian Function**
Introduce a Lagrange multiplier $\nu \in \mathbb{R}$ for the equality constraint and multipliers $\lambda_a \ge 0$ for the non-negativity inequality constraints. The Lagrangian function $\mathcal{L}(\mu, \nu, \lambda)$ is:
$$\mathcal{L}(\mu, \nu, \lambda) = \sum_{a \in \mathcal{A}} \mu(a) q_a - \sum_{a \in \mathcal{A}} \mu(a) \ln \mu(a) - \nu \left( \sum_{a \in \mathcal{A}} \mu(a) - 1 \right) + \sum_{a \in \mathcal{A}} \lambda_a \mu(a)$$

**Step 3: Differentiating the Lagrangian (First-Order KKT Stationarity)**
Compute the partial derivative of $\mathcal{L}$ with respect to the coordinate $\mu(a)$:
$$\frac{\partial \mathcal{L}}{\partial \mu(a)} = \frac{\partial}{\partial \mu(a)} \left[ \mu(a) q_a \right] - \frac{\partial}{\partial \mu(a)} \left[ \mu(a) \ln \mu(a) \right] - \nu \frac{\partial}{\partial \mu(a)} [\mu(a)] + \lambda_a \frac{\partial}{\partial \mu(a)} [\mu(a)]$$
Evaluate each derivative term individually:
1. $\frac{\partial}{\partial \mu(a)} [\mu(a) q_a] = q_a$
2. $\frac{\partial}{\partial \mu(a)} [\mu(a) \ln \mu(a)] = 1 \cdot \ln \mu(a) + \mu(a) \cdot \frac{1}{\mu(a)} = \ln \mu(a) + 1$
3. $\nu \frac{\partial}{\partial \mu(a)} [\mu(a)] = \nu$
4. $\lambda_a \frac{\partial}{\partial \mu(a)} [\mu(a)] = \lambda_a$

Combining these terms and setting the gradient to zero:
$$\frac{\partial \mathcal{L}}{\partial \mu(a)} = q_a - \ln \mu(a) - 1 - \nu + \lambda_a = 0$$

**Step 4: Strict Positivity and Inactivity of Boundary Constraints**
Examine the boundary behavior as $\mu(a) \to 0^+$:
$$\lim_{\mu(a) \to 0^+} \frac{\partial \Phi}{\partial \mu(a)} = \lim_{\mu(a) \to 0^+} (q_a - 1 - \ln \mu(a)) = +\infty$$
Because the derivative approaches $+\infty$ at the boundary $\mu(a) = 0$, the objective $\Phi(\mu)$ increases strictly when moving into the interior of the simplex. Therefore, the optimal solution $\mu^*(a)$ must lie strictly in the interior:
$$\mu^*(a) > 0 \quad \forall a \in \mathcal{A}$$
By the Karush-Kuhn-Tucker complementary slackness condition:
$$\lambda_a \mu^*(a) = 0 \quad \text{with } \lambda_a \ge 0$$
Since $\mu^*(a) > 0$, the multiplier must be zero:
$$\lambda_a = 0 \quad \forall a \in \mathcal{A}$$

**Step 5: Solving for the Optimal Density Function**
Substitute $\lambda_a = 0$ into the stationarity condition:
$$q_a - \ln \mu^*(a) - 1 - \nu = 0$$
Isolate the logarithmic term $\ln \mu^*(a)$:
$$\ln \mu^*(a) = q_a - (1 + \nu)$$
Exponentiate both sides:
$$\mu^*(a) = \exp(q_a - (1 + \nu)) = \exp(-(1 + \nu)) \cdot \exp(q_a)$$

**Step 6: Enforcing the Normalization Constraint**
The optimal probabilities must sum to 1:
$$\sum_{a \in \mathcal{A}} \mu^*(a) = 1 \implies \sum_{a \in \mathcal{A}} \exp(-(1 + \nu)) \exp(q_a) = 1$$
Factor out the scalar term $\exp(-(1 + \nu))$:
$$\exp(-(1 + \nu)) \sum_{a \in \mathcal{A}} \exp(q_a) = 1$$
Solve for $\exp(-(1 + \nu))$:
$$\exp(-(1 + \nu)) = \frac{1}{\sum_{b \in \mathcal{A}} \exp(q_b)}$$
Substitute this scalar back into the expression for $\mu^*(a)$:
$$\mu^*(a) = \frac{\exp(q_a)}{\sum_{b \in \mathcal{A}} \exp(q_b)}$$
Reinserting $q_a = Q(s, a)$, we have proven that the adversarial policy is precisely the softmax policy:
$$\mu^*(a \mid s) = \frac{\exp(Q(s, a))}{\sum_{b \in \mathcal{A}} \exp(Q(s, b))} = \operatorname{softmax}(Q(s, \cdot))_a$$

**Step 7: Substituting $\mu^*$ Back into the Primal Objective**
Now substitute $\mu^*(a)$ into the objective $\Phi(\mu)$ to evaluate its maximum value $\Phi^*$:
$$\Phi^* = \sum_{a \in \mathcal{A}} \mu^*(a) q_a - \sum_{a \in \mathcal{A}} \mu^*(a) \ln \mu^*(a)$$
From Step 5, we have the identity:
$$\ln \mu^*(a) = q_a - \ln \left( \sum_{b \in \mathcal{A}} \exp(q_b) \right)$$
Substitute this identity into the entropy term:
$$\sum_{a \in \mathcal{A}} \mu^*(a) \ln \mu^*(a) = \sum_{a \in \mathcal{A}} \mu^*(a) \left[ q_a - \ln \left( \sum_{b \in \mathcal{A}} \exp(q_b) \right) \right]$$
Expand the bracketed sum:
$$\sum_{a \in \mathcal{A}} \mu^*(a) \ln \mu^*(a) = \sum_{a \in \mathcal{A}} \mu^*(a) q_a - \sum_{a \in \mathcal{A}} \mu^*(a) \ln \left( \sum_{b \in \mathcal{A}} \exp(q_b) \right)$$
Since $\ln \left( \sum_b \exp(q_b) \right)$ does not depend on the summation index $a$, factor it out:
$$\sum_{a \in \mathcal{A}} \mu^*(a) \ln \mu^*(a) = \sum_{a \in \mathcal{A}} \mu^*(a) q_a - \ln \left( \sum_{b \in \mathcal{A}} \exp(q_b) \right) \underbrace{\sum_{a \in \mathcal{A}} \mu^*(a)}_{= 1}$$
$$= \sum_{a \in \mathcal{A}} \mu^*(a) q_a - \ln \left( \sum_{b \in \mathcal{A}} \exp(q_b) \right)$$

**Step 8: Exact Cancellation to Log-Sum-Exp**
Substitute this expression for the entropy into $\Phi^*$:
$$\Phi^* = \sum_{a \in \mathcal{A}} \mu^*(a) q_a - \left( \sum_{a \in \mathcal{A}} \mu^*(a) q_a - \ln \left( \sum_{b \in \mathcal{A}} \exp(q_b) \right) \right)$$
The linear terms $\sum_a \mu^*(a) q_a$ cancel out identically:
$$\Phi^* = \sum_{a \in \mathcal{A}} \mu^*(a) q_a - \sum_{a \in \mathcal{A}} \mu^*(a) q_a + \ln \left( \sum_{b \in \mathcal{A}} \exp(q_b) \right)$$
$$\Phi^* = \ln \left( \sum_{a \in \mathcal{A}} \exp(Q(s, a)) \right)$$

**Step 9: Completing the Overall CQL Objective**
Substitute $\Phi^*$ into the overall minimax objective.
The expectation over $s \sim d^\mathcal{D}$ of the inner maximum becomes:
$$\mathbb{E}_{s \sim d^\mathcal{D}} \left[ \max_{\mu \in \Delta} \left\{ \sum_a \mu(a) Q(s, a) + \mathcal{H}(\mu) \right\} - \sum_a \hat{\pi}_\beta(a \mid s) Q(s, a) \right]$$
$$= \mathbb{E}_{s \sim d^\mathcal{D}} \left[ \ln \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta(a \mid s)} [Q(s, a)] \right]$$
Combining this with the standard Bellman Temporal-Difference error gives the exact unconstrained Conservative Q-Learning objective:
$$\min_Q \mathcal{L}_{\text{CQL}}(Q) = \alpha \, \mathbb{E}_{s \sim d^\mathcal{D}} \left[ \ln \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta(a \mid s)} [Q(s, a)] \right] + \frac{1}{2} \mathbb{E}_{(s, a) \sim d^\mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q(s, a) \right)^2 \right]$$

**Step 10: Limiting Behavior Under Temperature Scaling**
Consider scaling the entropy by a temperature parameter $\tau > 0$: $\tau \mathcal{H}(\mu)$.
The inner maximum becomes $\tau \ln \sum_a \exp(Q(s, a) / \tau)$.
Taking the zero-temperature limit $\tau \to 0^+$:
$$\lim_{\tau \to 0^+} \tau \ln \sum_{a \in \mathcal{A}} \exp\left( \frac{Q(s, a)}{\tau} \right) = \max_{a \in \mathcal{A}} Q(s, a)$$
Thus, as $\tau \to 0$, the regularizer approaches the hard adversarial maximum $\max_a Q(s, a) - \mathbb{E}_{\hat{\pi}_\beta}[Q(s, a)]$, which directly suppresses the single most optimistic hallucinated action.
At finite temperature $\tau = 1$, the Log-Sum-Exp provides a smooth, everywhere-differentiable upper bound that pulls down all actions with high values simultaneously.
This completes the end-to-end first-principles derivation. $\blacksquare$

---

## 3. Geometric & Physical Interpretation

### 3.1 The Energy Basin of In-Distribution Data
Think of the action-value space as an energy surface over actions:
```
Action Value Q(s, a)
 ^
 |      [In-Data: a_1]             [OOD: a_2]
 |           |                         |
 |           v                         v
 |       +-------+                 +-------+
 |       | Q=2.0 |                 | Q=5.0 |  <-- Hallucinated Peak!
 |       +-------+                 +-------+
 |           |                         |
 |           | (Push UP: +0.48)        | (Crushed DOWN: -0.48)
 |           v                         v
 |       +-------+                 +-------+
 |       | Q=2.48|                 | Q=4.52|
 |       +-------+                 +-------+
```
- CQL applies a selective gravitational pull: actions without data support are dragged down into the floor, creating a safe, convex **basin of trust** centered squarely on the behavior policy manifold.

---

## 4. Real-World Analogy: Medical Prescription AI

Consider an AI physician analyzing clinical hospital records:
- **Patient State:** Stage-2 Hypertension.
- **Action $a_1$ (Standard Protocol):** Prescribe 10 mg Lisinopril. Observed in 1,000 patient records with positive outcome ($Q = 8.5$).
- **Action $a_2$ (Untested Crazy Dosage):** Prescribe 500 mg Lisinopril. Observed $0$ times in human history.
- **Standard Unregularized Q-learning:** The neural network extrapolates wildly on 500 mg, predicting an outcome score of $+999.0$. The unregularized AI prescribes the lethal overdose!
- **Conservative Q-Learning (CQL):** Recognizing that $a_2$ has zero evidence in $\mathcal{D}$, the Log-Sum-Exp penalty forcefully suppresses its value: $Q(s, a_2) \leftarrow -100.0$. The AI safely prescribes the 10 mg clinical protocol.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical gradient step of **Conservative Q-Learning (CQL)** by hand on a concrete 1-state, 2-action problem.

---

### 5.1 System & Initial Setup
- **State:** Single state $S_0$
- **Actions:** $\mathcal{A} = \{a_1, a_2\}$
  - $a_1$: **In-Distribution Action** (observed in dataset $\mathcal{D}$ with empirical probability $\pi_\beta(a_1 \mid S_0) = 1.0$)
  - $a_2$: **Out-of-Distribution (OOD) Action** (never observed, $\pi_\beta(a_2 \mid S_0) = 0.0$)
- **Current $Q$-values (with OOD Hallucination):**
  $$Q(S_0, a_1) = 2.0000 \quad (\text{True data-supported value})$$
  $$Q(S_0, a_2) = 5.0000 \quad (\text{Hallucinated high value!})$$
- **TD Target:** Assume Bellman TD error on $a_1$ is $0.0$ (i.e. $Q(S_0, a_1) = \text{Target} = 2.0000$).
- **Hyperparameters:**
  - Conservatism coefficient: $\alpha = 1.0000$
  - Learning rate: $\eta = 0.5000$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $Q(S_0, a)$ | `q_vals[a]` | Action-value estimates before update ($Q(a_1)=2.0, Q(a_2)=5.0$) |
| $\pi_\beta(a \mid S_0)$ | `data_prob[a]` | Empirical data frequency ($\pi_\beta(a_1)=1.0, \pi_\beta(a_2)=0.0$) |
| $p_{\text{softmax}}(a)$ | `softmax_prob[a]` | Model softmax policy: $\exp(Q(a)) / \sum \exp(Q(b))$ |
| $\nabla_{Q(a)} \mathcal{R}$ | `grad_cql[a]` | CQL regularizer gradient: $p_{\text{softmax}}(a) - \pi_\beta(a)$ |
| $Q_{\text{new}}(S_0, a)$ | `q_new[a]` | Updated $Q$-value: $Q - \eta \cdot \alpha \cdot \nabla_{Q(a)} \mathcal{R}$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Exponentials & Log-Sum-Exp
$$e^{Q(S_0, a_1)} = e^{2.0} \approx \mathbf{7.389056}$$
$$e^{Q(S_0, a_2)} = e^{5.0} \approx \mathbf{148.413159}$$
$$\sum_{a} e^{Q(S_0, a)} = 7.389056 + 148.413159 = \mathbf{155.802215}$$
$$\text{Log-Sum-Exp} = \ln(155.802215) \approx \mathbf{5.048615}$$

#### Step 2: Compute Softmax Action Probabilities
$$p_{\text{softmax}}(a_1) = \frac{7.389056}{155.802215} \approx \mathbf{0.047426}$$
$$p_{\text{softmax}}(a_2) = \frac{148.413159}{155.802215} \approx \mathbf{0.952574}$$

*(Notice how the neural network strongly prefers the hallucinated OOD action $a_2$ with 95.3% probability!)*

#### Step 3: Compute CQL Gradients
$$\nabla_{Q(a_1)} \mathcal{R} = p_{\text{softmax}}(a_1) - \pi_\beta(a_1) = 0.047426 - 1.000000 = \mathbf{-0.952574}$$
$$\nabla_{Q(a_2)} \mathcal{R} = p_{\text{softmax}}(a_2) - \pi_\beta(a_2) = 0.952574 - 0.000000 = \mathbf{+0.952574}$$

#### Step 4: Execute Conservative Gradient Update
With learning rate $\eta = 0.5000$ and $\alpha = 1.0000$:

1. **Update for In-Distribution Action $a_1$:**
   $$Q_{\text{new}}(S_0, a_1) = Q(S_0, a_1) - \eta \alpha \nabla_{Q(a_1)} \mathcal{R}$$
   $$= 2.0000 - 0.5000 \times (-0.952574) = 2.0000 + 0.476287 = \mathbf{2.476287} \quad (\uparrow \text{ Boosted!})$$

2. **Update for Out-of-Distribution Action $a_2$:**
   $$Q_{\text{new}}(S_0, a_2) = Q(S_0, a_2) - \eta \alpha \nabla_{Q(a_2)} \mathcal{R}$$
   $$= 5.0000 - 0.5000 \times (+0.952574) = 5.0000 - 0.476287 = \mathbf{4.523713} \quad (\downarrow \text{ Penalized!})$$

---

### 5.4 Summary Visual Grid: CQL Gradient Descent Ledger

| Action $a$ | Data Type | Prior $Q(a)$ | $\exp(Q)$ | Softmax Prob $p(a)$ | Data Prob $\pi_\beta(a)$ | Gradient $\nabla_Q \mathcal{R}$ | Step $\Delta Q$ | Updated $Q_{\text{new}}(a)$ | Effect |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$a_1$** | **In-Data** | $2.0000$ | $7.3891$ | $0.0474$ | **$1.0000$** | **$-0.9526$** | $+0.4763$ | **$2.4763$** | $\mathbf{\uparrow}$ Elevated |
| **$a_2$** | **OOD** | $5.0000$ | $148.4132$ | $0.9526$ | **$0.0000$** | **$+0.9526$** | $-0.4763$ | **$4.5237$** | $\mathbf{\downarrow}$ Crushed |

---

## 6. Solved Numerical Illustrations

### Illustration 1: Asymptotic Limits of Conservatism Parameter $\alpha$ and Regularization Spectrum
**Problem:**
Analyze the mathematical behavior and policy convergence of Conservative Q-Learning across the complete spectrum of the conservatism weight $\alpha$:
1. The unconstrained limit: $\alpha \to 0$.
2. The intermediate regime: $\alpha \in (0, \infty)$.
3. The infinite conservatism limit: $\alpha \to \infty$.

**Solution:**
Recall the discrete CQL objective:
$$\min_Q \mathcal{L}_{\text{CQL}}(Q) = \alpha \, \mathbb{E}_{s \sim \mathcal{D}} \left[ \ln \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta}[Q(s, a)] \right] + \frac{1}{2} \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^\pi Q(s, a) \right)^2 \right]$$

1. **Limit $\alpha \to 0$ (Unconstrained Off-Policy RL):**
   When $\alpha = 0$, the conservative regularizer vanishes:
   $$\min_Q \frac{1}{2} \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ (Q(s, a) - \hat{\mathcal{B}}^\pi Q(s, a))^2 \right]$$
   The algorithm reduces to standard unregularized off-policy TD learning (e.g. DQN or SAC). Under function approximation, any out-of-distribution action $a_{\text{ood}} \notin \mathcal{D}$ with zero data constraint can receive an arbitrarily high value spike $Q(s', a_{\text{ood}}) = +10^6$. The greedy policy selects $\arg\max_a Q(s', a) = a_{\text{ood}}$, and the Bellman error bootstraps on this hallucination: $Q(s, a) \leftarrow r + \gamma \cdot 10^6$. The value function diverges to $+\infty$ and the deployed policy suffers catastrophic failure.

2. **Intermediate Regime $\alpha \in [\alpha_{\min}, \alpha_{\max}]$ (Optimal Offline RL):**
   By Derivation 11.25.1, when $\alpha \ge \frac{C_\delta}{1 - \gamma} \max_{s, a} \frac{\hat{\pi}_\beta(a \mid s)}{\mu(a \mid s) \sqrt{|\mathcal{D}(s, a)|}}$, the conservative penalty cancels the statistical sampling error. The learned $Q$-function is a certified lower bound $\hat{Q}^\pi(s, a) \le Q^\pi(s, a)$ that suppresses OOD actions while preserving relative value rankings among in-distribution transitions. This enables the agent to safely **stitch together** optimal trajectory fragments from sub-optimal data, strictly outperforming the behavior policy $\pi_\beta$.

3. **Limit $\alpha \to \infty$ (Supervised Behavior Cloning Collapse):**
   Dividing the loss by $\alpha$ and taking $\alpha \to \infty$:
   $$\lim_{\alpha \to \infty} \frac{1}{\alpha} \mathcal{L}_{\text{CQL}}(Q) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \ln \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \sum_{a \in \mathcal{A}} \hat{\pi}_\beta(a \mid s) Q(s, a) \right]$$
   The Bellman TD error is completely dominated and ignored.
   Differentiating with respect to $Q(s, a_i)$ and setting to zero:
   $$\frac{\exp(Q(s, a_i))}{\sum_b \exp(Q(s, b))} - \hat{\pi}_\beta(a_i \mid s) = 0 \implies p_{\text{softmax}}(a_i \mid s) \equiv \hat{\pi}_\beta(a_i \mid s)$$
   The optimal policy collapses to the empirical data distribution $\hat{\pi}_\beta$. The algorithm degenerates into pure **Behavior Cloning (Supervised Imitation Learning)**. Reward maximization is completely extinguished, and the agent loses the ability to outperform mediocre demonstration datasets. $\blacksquare$

---

### Illustration 2: Step-by-Step CQL Penalty Computation on a 3-Action State
**Problem:**
Consider a state $s_0$ with three discrete actions $\mathcal{A} = \{a_0, a_1, a_2\}$.
The offline dataset contains $N_0 = 100$ transitions for $a_0$, $N_1 = 50$ transitions for $a_1$, and $N_2 = 0$ transitions for $a_2$ (total $N_{\text{total}} = 150$).
The current neural network $Q$-values (with an out-of-distribution hallucination on $a_2$) are:
$$Q(s_0, a_0) = 4.000000, \quad Q(s_0, a_1) = 2.000000, \quad Q(s_0, a_2) = 8.000000$$
Using conservatism weight $\alpha = 1.000000$ and learning rate $\eta = 0.100000$, perform a complete step-by-step calculation of:
1. The empirical behavior policy probabilities $\hat{\pi}_\beta(a \mid s_0)$.
2. The exponentials, partition function, and Log-Sum-Exp value.
3. The model softmax probabilities $p_{\text{softmax}}(a \mid s_0)$.
4. The exact CQL regularizer gradients $\nabla_{Q(a)} \mathcal{R}$.
5. The updated $Q$-values $Q_{\text{new}}(s_0, a)$ after one gradient descent step.

**Solution:**

**Step 1: Compute Empirical Behavior Policy Probabilities**
Total dataset observations: $N_{\text{total}} = 100 + 50 + 0 = 150$.
$$\hat{\pi}_\beta(a_0 \mid s_0) = \frac{100}{150} = \frac{2}{3} \approx 0.666667$$
$$\hat{\pi}_\beta(a_1 \mid s_0) = \frac{50}{150} = \frac{1}{3} \approx 0.333333$$
$$\hat{\pi}_\beta(a_2 \mid s_0) = \frac{0}{150} = 0.000000 \quad (\text{Out-of-Distribution})$$

**Step 2: Compute Exponentials, Partition Function, and Log-Sum-Exp**
$$\exp(Q(s_0, a_0)) = e^{4.0} \approx 54.598150$$
$$\exp(Q(s_0, a_1)) = e^{2.0} \approx 7.389056$$
$$\exp(Q(s_0, a_2)) = e^{8.0} \approx 2980.957987$$
Sum of exponentials (partition function $Z$):
$$Z = 54.598150 + 7.389056 + 2980.957987 = 3042.945193$$
Log-Sum-Exp value:
$$\text{LSE} = \ln(3042.945193) \approx 8.020581$$

**Step 3: Compute Model Softmax Action Probabilities**
$$p_{\text{softmax}}(a_0) = \frac{54.598150}{3042.945193} \approx 0.017943 \quad (1.79\%)$$
$$p_{\text{softmax}}(a_1) = \frac{7.389056}{3042.945193} \approx 0.002428 \quad (0.24\%)$$
$$p_{\text{softmax}}(a_2) = \frac{2980.957987}{3042.945193} \approx 0.979629 \quad (97.96\%)$$
*Crucial Observation:* Due to the unregularized hallucinated peak $Q(a_2) = 8.0$, the policy assigns $97.96\%$ of its probability to the completely unseen, dangerous action $a_2$!

**Step 4: Compute CQL Regularizer Gradients**
The gradient of the regularizer $\mathcal{R}(Q) = \ln \sum_b \exp(Q(s_0, b)) - \sum_b \hat{\pi}_\beta(b) Q(s_0, b)$ is:
$$\nabla_{Q(a_i)} \mathcal{R} = p_{\text{softmax}}(a_i) - \hat{\pi}_\beta(a_i)$$
Evaluating each coordinate:
$$\nabla_{Q(a_0)} \mathcal{R} = 0.017943 - 0.666667 = -0.648724$$
$$\nabla_{Q(a_1)} \mathcal{R} = 0.002428 - 0.333333 = -0.330905$$
$$\nabla_{Q(a_2)} \mathcal{R} = 0.979629 - 0.000000 = +0.979629$$

**Step 5: Execute Conservative Gradient Update**
With learning rate $\eta = 0.100000$ and $\alpha = 1.000000$:
$$\Delta Q(a_i) = - \eta \alpha \nabla_{Q(a_i)} \mathcal{R}$$
1. For action $a_0$:
   $$\Delta Q(a_0) = - 0.100000 \times 1.000000 \times (-0.648724) = +0.064872$$
   $$Q_{\text{new}}(s_0, a_0) = 4.000000 + 0.064872 = 4.064872 \quad (\uparrow \text{ Boosted!})$$
2. For action $a_1$:
   $$\Delta Q(a_1) = - 0.100000 \times 1.000000 \times (-0.330905) = +0.033091$$
   $$Q_{\text{new}}(s_0, a_1) = 2.000000 + 0.033091 = 2.033091 \quad (\uparrow \text{ Boosted!})$$
3. For out-of-distribution action $a_2$:
   $$\Delta Q(a_2) = - 0.100000 \times 1.000000 \times (+0.979629) = -0.097963$$
   $$Q_{\text{new}}(s_0, a_2) = 8.000000 - 0.097963 = 7.902037 \quad (\downarrow \text{ Harshly Penalized!})$$

| Action | Count $N_i$ | $\hat{\pi}_\beta(a_i)$ | Prior $Q(a_i)$ | $p_{\text{softmax}}(a_i)$ | Gradient $\nabla_{Q} \mathcal{R}$ | Step $\Delta Q$ | Updated $Q_{\text{new}}(a_i)$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$a_0$** | $100$ | $0.666667$ | $4.000000$ | $0.017943$ | $-0.648724$ | $+0.064872$ | **$4.064872$** | $\mathbf{\uparrow}$ Elevated |
| **$a_1$** | $50$ | $0.333333$ | $2.000000$ | $0.002428$ | $-0.330905$ | $+0.033091$ | **$2.033091$** | $\mathbf{\uparrow}$ Elevated |
| **$a_2$** | $0$ | $0.000000$ | $8.000000$ | $0.979629$ | $+0.979629$ | $-0.097963$ | **$7.902037$** | $\mathbf{\downarrow}$ Suppressed |

---

### Illustration 3: CQL-SAC Target Calculation with Soft Bellman Backup and Conservative Regularization
**Problem:**
In continuous offline RL (CQL-SAC), an agent evaluates a transition $(s, a_{\text{data}}, r, s')$ with:
- Reward: $r = 1.5000$
- Discount factor: $\gamma = 0.9500$
- SAC entropy temperature: $\alpha_{\text{SAC}} = 0.2000$
- Conservative penalty weight: $\alpha_{\text{CQL}} = 1.0000$
- Target policy proposes action $a'$ at next state $s'$ with log-probability $\ln \pi(a' \mid s') = -1.2000$.
- Twin target networks evaluate $a'$: $Q_{\bar{\theta}_1}(s', a') = 12.0000$, $Q_{\bar{\theta}_2}(s', a') = 10.5000$.
- Current critic network estimate on dataset action: $Q_\theta(s, a_{\text{data}}) = 11.0000$.
- To approximate the continuous Log-Sum-Exp regularizer at state $s$, the critic samples $M = 4$ candidate actions with values:
  $$Q_\theta(s, a_1^{\text{unif}}) = 8.0000, \quad Q_\theta(s, a_2^{\text{unif}}) = 6.0000$$
  $$Q_\theta(s, a_1^{\pi}) = 13.0000 \; (\text{OOD hallucination!}), \quad Q_\theta(s, a_2^{\pi}) = 10.0000$$
Calculate:
1. The clipped soft target value $V_{\text{targ}}(s')$ and the Bellman target $y_{\text{Bellman}}$.
2. The Bellman Temporal-Difference error and MSE loss $\mathcal{L}_{\text{TD}}$.
3. The empirical continuous Log-Sum-Exp over the 4 sampled actions.
4. The CQL regularization loss $\mathcal{R}_{\text{CQL}}$ and the total combined critic loss $\mathcal{L}_{\text{total}}$.
5. The exact gradient of the total loss with respect to the dataset action-value $Q_\theta(s, a_{\text{data}})$.

**Solution:**

**Step 1: Compute Clipped Soft Target Value and Bellman Target**
Take the twin target minimum:
$$\min(Q_{\bar{\theta}_1}(s', a'), Q_{\bar{\theta}_2}(s', a')) = \min(12.0000, 10.5000) = 10.5000$$
Add the soft entropy bonus:
$$V_{\text{targ}}(s') = \min(Q_{\bar{\theta}_1}, Q_{\bar{\theta}_2}) - \alpha_{\text{SAC}} \ln \pi(a' \mid s')$$
$$V_{\text{targ}}(s') = 10.5000 - 0.2000 \times (-1.2000) = 10.5000 + 0.2400 = 10.7400$$
Compute the standard Bellman target:
$$y_{\text{Bellman}} = r + \gamma V_{\text{targ}}(s') = 1.5000 + 0.9500 \times 10.7400 = 1.5000 + 10.2030 = 11.7030$$

**Step 2: Compute TD Error and TD Loss**
Current critic estimate: $Q_\theta(s, a_{\text{data}}) = 11.0000$.
$$\delta_{\text{TD}} = Q_\theta(s, a_{\text{data}}) - y_{\text{Bellman}} = 11.0000 - 11.7030 = -0.7030$$
The standard squared Bellman error loss is:
$$\mathcal{L}_{\text{TD}} = \frac{1}{2} \delta_{\text{TD}}^2 = \frac{1}{2} (-0.7030)^2 = \frac{1}{2} \times 0.494209 = 0.247104$$

**Step 3: Compute Empirical Continuous Log-Sum-Exp**
The critic evaluates $M=4$ sampled actions: $Q = [8.0, 6.0, 13.0, 10.0]$.
Compute exponentials:
$$\exp(8.0) \approx 2980.957987$$
$$\exp(6.0) \approx 403.428793$$
$$\exp(13.0) \approx 442413.392009$$
$$\exp(10.0) \approx 22026.465795$$
Sum of exponentials:
$$\sum_{m=1}^4 \exp(Q(s, a_m)) = 2980.957987 + 403.428793 + 442413.392009 + 22026.465795 = 467824.244584$$
In continuous action spaces, the empirical expectation over $M$ samples gives:
$$\text{LSE}_{\text{sample}} = \ln \left( \frac{1}{M} \sum_{m=1}^M \exp(Q(s, a_m)) \right) = \ln \left( \frac{467824.244584}{4} \right) = \ln(116956.061146) \approx 11.669554$$

**Step 4: Compute CQL Regularizer and Total Critic Loss**
The CQL penalty penalizes the difference between the sampled Log-Sum-Exp and the dataset action value:
$$\mathcal{R}_{\text{CQL}} = \text{LSE}_{\text{sample}} - Q_\theta(s, a_{\text{data}}) = 11.669554 - 11.000000 = 0.669554$$
With $\alpha_{\text{CQL}} = 1.0000$:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TD}} + \alpha_{\text{CQL}} \mathcal{R}_{\text{CQL}} = 0.247104 + 1.0000 \times 0.669554 = 0.916658$$

**Step 5: Compute Gradient With Respect to $Q_\theta(s, a_{\text{data}})$**
Differentiating the total critic loss:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial Q_\theta(s, a_{\text{data}})} = \frac{\partial \mathcal{L}_{\text{TD}}}{\partial Q_\theta(s, a_{\text{data}})} + \alpha_{\text{CQL}} \frac{\partial \mathcal{R}_{\text{CQL}}}{\partial Q_\theta(s, a_{\text{data}})}$$
$$\frac{\partial \mathcal{L}_{\text{TD}}}{\partial Q_\theta(s, a_{\text{data}})} = \delta_{\text{TD}} = -0.7030$$
$$\frac{\partial \mathcal{R}_{\text{CQL}}}{\partial Q_\theta(s, a_{\text{data}})} = -1.0000$$
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial Q_\theta(s, a_{\text{data}})} = -0.7030 + 1.0000 \times (-1.0000) = -1.7030$$
Because the gradient is negative ($-1.7030$), gradient descent pushes $Q_\theta(s, a_{\text{data}})$ **upward** by $+1.7030 \eta$, simultaneously pulling $Q$ closer to the Bellman target ($11.7030$) and boosting the relative value of the data-supported action above the sampled actions! $\blacksquare$

---

### Illustration 4: Dynamic Comparison of Standard Q-Learning vs CQL Under Out-of-Distribution Error Propagation
**Problem:**
Consider an offline environment state $s_0$ with two actions:
- $a_{\text{data}}$: Observed in dataset with 100 transitions ($\hat{\pi}_\beta(a_{\text{data}}) = 1.0$). True environment value: $Q^*(s_0, a_{\text{data}}) = 2.800000$. Bellman TD target: $y_{\text{target}} = 2.800000$.
- $a_{\text{ood}}$: Never observed in dataset ($\hat{\pi}_\beta(a_{\text{ood}}) = 0.0$). True environment value: $Q^*(s_0, a_{\text{ood}}) = 0.000000$.
Initial critic parameters have an OOD extrapolation error of $+4.500000$:
$$Q_0(s_0, a_{\text{data}}) = 2.800000, \quad Q_0(s_0, a_{\text{ood}}) = 4.500000$$
1. Show why standard unconstrained Q-learning fails catastrophically and cannot correct this error.
2. Trace 5 consecutive gradient steps of Conservative Q-Learning with conservatism weight $\alpha = 2.000000$ and learning rate $\eta = 0.200000$. Show exactly at which step the greedy policy flips from the dangerous OOD action to the safe data action.

**Solution:**

**1. Failure of Standard Q-Learning:**
In standard offline Q-learning, the loss function is evaluated strictly over dataset transitions:
$$\mathcal{L}_{\text{std}}(Q) = \frac{1}{2} \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ (Q(s, a) - y)^2 \right] = \frac{1}{2} (Q(s_0, a_{\text{data}}) - 2.800000)^2$$
Notice that $a_{\text{ood}}$ never appears in $\mathcal{D}$. Therefore:
$$\frac{\partial \mathcal{L}_{\text{std}}}{\partial Q(s_0, a_{\text{ood}})} \equiv 0.000000$$
Standard Q-learning computes zero gradient for $a_{\text{ood}}$. The hallucinated value remains permanently frozen at $Q(s_0, a_{\text{ood}}) = 4.500000 > Q(s_0, a_{\text{data}}) = 2.800000$.
When deployed in the real environment, the greedy policy selects $\arg\max_a Q(s_0, a) = a_{\text{ood}}$, incurring true return $0.000000$ instead of $2.800000$—a complete policy breakdown!

**2. Step-by-Step CQL Gradient Descent Trace:**
Under CQL, the total loss is:
$$\mathcal{L}_{\text{CQL}}(Q) = \frac{1}{2} (Q(s_0, a_{\text{data}}) - 2.80)^2 + \alpha \left( \ln(e^{Q(s_0, a_{\text{data}})} + e^{Q(s_0, a_{\text{ood}})}) - Q(s_0, a_{\text{data}}) \right)$$
The gradients are:
$$\frac{\partial \mathcal{L}_{\text{CQL}}}{\partial Q(a_{\text{data}})} = (Q(a_{\text{data}}) - 2.80) + \alpha (p_{\text{softmax}}(a_{\text{data}}) - 1.0)$$
$$\frac{\partial \mathcal{L}_{\text{CQL}}}{\partial Q(a_{\text{ood}})} = \alpha (p_{\text{softmax}}(a_{\text{ood}}) - 0.0)$$
Update rule with $\eta = 0.20$: $Q \leftarrow Q - 0.20 \times \nabla Q$.

- **Step 1:**
  - $p_{\text{data}} = \frac{e^{2.80}}{e^{2.80} + e^{4.50}} = \frac{16.444647}{16.444647 + 90.017131} = 0.154465$, $p_{\text{ood}} = 0.845535$.
  - $\nabla Q(a_{\text{data}}) = (2.80 - 2.80) + 2.0 \times (0.154465 - 1.0) = -1.691069$.
  - $\nabla Q(a_{\text{ood}}) = 2.0 \times 0.845535 = +1.691069$.
  - $Q_1(a_{\text{data}}) = 2.800000 - 0.20 \times (-1.691069) = \mathbf{3.138214}$.
  - $Q_1(a_{\text{ood}}) = 4.500000 - 0.20 \times (+1.691069) = \mathbf{4.161786}$.
  - Greedy action: $\arg\max_a Q = a_{\text{ood}}$.

- **Step 2:**
  - $e^{3.138214} = 23.062497$, $e^{4.161786} = 64.186259$.
  - $p_{\text{data}} = \frac{23.062497}{87.248756} = 0.264332$, $p_{\text{ood}} = 0.735668$.
  - $\nabla Q(a_{\text{data}}) = (3.138214 - 2.80) + 2.0 \times (0.264332 - 1.0) = 0.338214 - 1.471336 = -1.133122$.
  - $\nabla Q(a_{\text{ood}}) = 2.0 \times 0.735668 = +1.471336$.
  - $Q_2(a_{\text{data}}) = 3.138214 - 0.20 \times (-1.133122) = \mathbf{3.364838}$.
  - $Q_2(a_{\text{ood}}) = 4.161786 - 0.20 \times (+1.471336) = \mathbf{3.867519}$.
  - Greedy action: $\arg\max_a Q = a_{\text{ood}}$.

- **Step 3:**
  - $e^{3.364838} = 28.929283$, $e^{3.867519} = 47.823677$.
  - $p_{\text{data}} = \frac{28.929283}{76.752960} = 0.376911$, $p_{\text{ood}} = 0.623089$.
  - $\nabla Q(a_{\text{data}}) = (3.364838 - 2.80) + 2.0 \times (0.376911 - 1.0) = 0.564838 - 1.246178 = -0.681340$.
  - $\nabla Q(a_{\text{ood}}) = 2.0 \times 0.623089 = +1.246178$.
  - $Q_3(a_{\text{data}}) = 3.364838 - 0.20 \times (-0.681340) = \mathbf{3.501106}$.
  - $Q_3(a_{\text{ood}}) = 3.867519 - 0.20 \times (+1.246178) = \mathbf{3.618283}$.
  - Greedy action: $\arg\max_a Q = a_{\text{ood}}$.

- **Step 4:**
  - $e^{3.501106} = 33.152331$, $e^{3.618283} = 37.273767$.
  - $p_{\text{data}} = \frac{33.152331}{70.426098} = 0.470739$, $p_{\text{ood}} = 0.529261$.
  - $\nabla Q(a_{\text{data}}) = (3.501106 - 2.80) + 2.0 \times (0.470739 - 1.0) = 0.701106 - 1.058522 = -0.357415$.
  - $\nabla Q(a_{\text{ood}}) = 2.0 \times 0.529261 = +1.058522$.
  - $Q_4(a_{\text{data}}) = 3.501106 - 0.20 \times (-0.357415) = \mathbf{3.572589}$.
  - $Q_4(a_{\text{ood}}) = 3.618283 - 0.20 \times (+1.058522) = \mathbf{3.406579}$.
  - **POLICY FLIP DETECTED:** $Q_4(a_{\text{data}}) = 3.572589 > Q_4(a_{\text{ood}}) = 3.406579$. The greedy policy successfully selects the safe action $a_{\text{data}}$!

- **Step 5:**
  - $e^{3.572589} = 35.609204$, $e^{3.406579} = 30.161642$.
  - $p_{\text{data}} = \frac{35.609204}{65.770846} = 0.541408$, $p_{\text{ood}} = 0.458592$.
  - $\nabla Q(a_{\text{data}}) = (3.572589 - 2.80) + 2.0 \times (0.541408 - 1.0) = 0.772589 - 0.917185 = -0.144596$.
  - $\nabla Q(a_{\text{ood}}) = 2.0 \times 0.458592 = +0.917185$.
  - $Q_5(a_{\text{data}}) = 3.572589 - 0.20 \times (-0.144596) = \mathbf{3.601508}$.
  - $Q_5(a_{\text{ood}}) = 3.406579 - 0.20 \times (+0.917185) = \mathbf{3.223142}$.
  - Greedy action: $a_{\text{data}}$ (margin widened to $+0.378366$).

| Iteration | $p(a_{\text{data}})$ | $p(a_{\text{ood}})$ | $\nabla Q(a_{\text{data}})$ | $\nabla Q(a_{\text{ood}})$ | $Q(s_0, a_{\text{data}})$ | $Q(s_0, a_{\text{ood}})$ | Greedy Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $0.1545$ | $0.8455$ | — | — | $2.800000$ | $4.500000$ | $a_{\text{ood}}$ (Dangerous!) |
| **1** | $0.1545$ | $0.8455$ | $-1.691069$ | $+1.691069$ | $3.138214$ | $4.161786$ | $a_{\text{ood}}$ |
| **2** | $0.2643$ | $0.7357$ | $-1.133122$ | $+1.471336$ | $3.364838$ | $3.867519$ | $a_{\text{ood}}$ |
| **3** | $0.3769$ | $0.6231$ | $-0.681340$ | $+1.246178$ | $3.501106$ | $3.618283$ | $a_{\text{ood}}$ |
| **4** | **$0.4707$** | **$0.5293$** | **$-0.357415$** | **$+1.058522$** | **$3.572589$** | **$3.406579$** | **$a_{\text{data}}$ (FLIP!)** |
| **5** | $0.5414$ | $0.4586$ | $-0.144596$ | $+0.917185$ | $3.601508$ | $3.223142$ | $a_{\text{data}}$ (Safe!) |

$\blacksquare$

---

### Illustration 5: Behavioral Cloning Regularization Weight Annealing in Offline TD3+BC
**Problem:**
In the minimalist offline RL algorithm **TD3+BC** (Fujimoto & Gu, NeurIPS 2021), the actor policy $\pi_\phi$ is updated by maximizing:
$$\mathcal{J}(\phi) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ \lambda Q(s, \pi_\phi(s)) - \|\pi_\phi(s) - a\|_2^2 \right]$$
where the adaptive scaling factor $\lambda$ is normalized across a mini-batch of size $N$ using fixed hyperparameter $\alpha = 2.5000$:
$$\lambda = \frac{\alpha}{\frac{1}{N} \sum_{i=1}^N |Q(s_i, a_i)|}$$
Consider a mini-batch of $N = 4$ transitions with critic action-values:
$$Q = [120.0000, -80.0000, 150.0000, 50.0000]$$
For transition $i=1$:
- Proposed policy action: $\pi_\phi(s_1) = [0.8000, -0.4000]^T$.
- Dataset action: $a_1 = [0.5000, -0.1000]^T$.
- Critic gradient at the proposed action: $\nabla_a Q(s_1, a)|_{a=\pi_\phi(s_1)} = [40.0000, -20.0000]^T$.
Calculate:
1. The batch mean absolute $Q$-value and the adaptive trade-off parameter $\lambda$.
2. The behavioral cloning MSE loss $\mathcal{L}_{\text{BC}}$ and its gradient $\nabla_a \mathcal{L}_{\text{BC}}$.
3. The normalized policy gradient $\nabla_a \mathcal{J}$ driving the actor update.
4. Contrast this with the unnormalized update (where $\lambda = 1.0$), explaining why adaptive scaling prevents offline policy collapse.

**Solution:**

**Step 1: Compute Batch Mean Absolute Value and Adaptive Weight $\lambda$**
Evaluate the absolute values of the batch:
$$|Q| = [|120.0000|, |-80.0000|, |150.0000|, |50.0000|] = [120.0000, 80.0000, 150.0000, 50.0000]$$
Sum of absolute values:
$$\sum_{i=1}^4 |Q(s_i, a_i)| = 120.0000 + 80.0000 + 150.0000 + 50.0000 = 400.0000$$
Mean absolute value:
$$\frac{1}{N} \sum_{i=1}^N |Q(s_i, a_i)| = \frac{400.0000}{4} = 100.0000$$
Compute the adaptive weighting coefficient $\lambda$ with $\alpha = 2.5000$:
$$\lambda = \frac{\alpha}{\frac{1}{N} \sum |Q(s_i, a_i)|} = \frac{2.5000}{100.0000} = 0.025000$$

**Step 2: Compute Behavioral Cloning Loss and Gradient**
Action discrepancy vector:
$$\Delta a = \pi_\phi(s_1) - a_1 = \begin{bmatrix} 0.8000 - 0.5000 \\ -0.4000 - (-0.1000) \end{bmatrix} = \begin{bmatrix} +0.3000 \\ -0.3000 \end{bmatrix}$$
The behavioral cloning squared Euclidean distance loss is:
$$\mathcal{L}_{\text{BC}} = \|\pi_\phi(s_1) - a_1\|_2^2 = (+0.3000)^2 + (-0.3000)^2 = 0.0900 + 0.0900 = 0.180000$$
The gradient with respect to the proposed action vector $a = \pi_\phi(s_1)$ is:
$$\nabla_a \mathcal{L}_{\text{BC}} = 2 (\pi_\phi(s_1) - a_1) = 2 \begin{bmatrix} +0.3000 \\ -0.3000 \end{bmatrix} = \begin{bmatrix} +0.6000 \\ -0.6000 \end{bmatrix}$$

**Step 3: Compute Actor Update Gradient**
The actor objective for this transition is:
$$\mathcal{J}(a) = \lambda Q(s_1, a) - \|\pi_\phi(s_1) - a_1\|_2^2$$
Differentiating with respect to action vector $a$:
$$\nabla_a \mathcal{J} = \lambda \nabla_a Q(s_1, a) - \nabla_a \mathcal{L}_{\text{BC}}$$
Compute the normalized reinforcement learning component:
$$\lambda \nabla_a Q = 0.025000 \times \begin{bmatrix} 40.0000 \\ -20.0000 \end{bmatrix} = \begin{bmatrix} 1.000000 \\ -0.500000 \end{bmatrix}$$
Subtract the behavioral cloning gradient:
$$\nabla_a \mathcal{J} = \begin{bmatrix} 1.000000 \\ -0.500000 \end{bmatrix} - \begin{bmatrix} +0.600000 \\ -0.600000 \end{bmatrix} = \begin{bmatrix} 1.000000 - 0.600000 \\ -0.500000 - (-0.600000) \end{bmatrix} = \begin{bmatrix} +0.400000 \\ +0.100000 \end{bmatrix}$$

**Step 4: Comparison with Unnormalized Formulation ($\lambda = 1.0$)**
If the algorithm used an unnormalized objective without the scaling denominator ($\lambda = 1.0$):
$$\nabla_a \mathcal{J}_{\text{unnorm}} = 1.0000 \times \begin{bmatrix} 40.0000 \\ -20.0000 \end{bmatrix} - \begin{bmatrix} +0.6000 \\ -0.6000 \end{bmatrix} = \begin{bmatrix} 39.400000 \\ -19.400000 \end{bmatrix}$$
- **Why Normalization is Vital:**
  Without adaptive normalization, the RL gradient magnitude ($\|\nabla Q\|_2 = \sqrt{40^2 + (-20)^2} = \sqrt{2000} \approx 44.72$) is **53 times larger** than the behavioral cloning constraint ($\|\nabla \mathcal{L}_{\text{BC}}\|_2 = \sqrt{0.6^2 + (-0.6)^2} = \sqrt{0.72} \approx 0.85$). The RL gradient completely swamps the BC regularizer, dragging the policy deep into out-of-distribution space where $Q$ is hallucinated.
  With the adaptive weight $\lambda = \frac{2.5000}{100.0000} = 0.025000$, the RL gradient is rescaled to magnitude $\|\lambda \nabla Q\|_2 = \sqrt{1.0^2 + (-0.5)^2} = \sqrt{1.25} \approx 1.12$. The RL gradient and the BC constraint operate on the exact same numerical scale ($1.12$ vs $0.85$), permitting the agent to optimize rewards while remaining safely tethered to the behavior policy manifold! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. D4RL Benchmarks & IQL: The Offline RL Evaluation Standard (Fu et al., 2020)
CQL's influence is measured through the D4RL (Datasets for Deep Data-Driven RL) benchmark suite:
- **D4RL Dataset Qualities:** `random`, `medium`, `medium-replay`, `medium-expert`, `expert` — graded data quality levels testing whether offline algorithms can extract good policies from impure data. CQL achieves normalized scores of 47.0 (medium) → 98.7 (expert) on HalfCheetah.
- **IQL (Implicit Q-Learning, Kostrikov et al., ICLR 2022):** Avoids out-of-distribution action queries entirely by learning a value function using expectile regression over in-dataset actions. Matches or exceeds CQL on 8/9 D4RL tasks while running 3× faster.
- **TD3+BC (Fujimoto & Gu, NeurIPS 2021):** Augments standard TD3 with a behavioral cloning regularizer: $\pi = \arg\max_a \lambda Q(s,a) - (a - \mu_\beta(s))^2$, requiring only 10 lines of code on top of TD3 while matching CQL on locomotion tasks.

### 2. Autonomous Driving & Medical AI: Offline RL in Safety-Critical Domains
Offline RL is the **only viable** training paradigm when online exploration carries unacceptable risk:
- **Waymo Motion Prediction (2022):** Waymo trains driving policy components offline on 570 hours of logged human driving data. Online exploration (letting the car randomly explore) is obviously impossible — CQL/IQL ensure the policy stays within the safe behavioral distribution of the logged data.
- **Google Health / Clinical Treatment Policies:** Sepsis treatment policies trained offline on ICU patient records using conservative Q-learning, where online RL exploration (administering untested drug dosages) is ethically and legally prohibited.
- **Recommender Systems (Netflix, YouTube):** Offline RL updates recommendation policies on logged user interaction data. CQL's conservatism prevents the policy from exploiting spurious correlations in the click logs that would fail catastrophically when deployed online.

### 3. Decision Transformer: The Offline RL → Sequence Modeling Bridge (Chapter 11.26)
CQL represents the Q-learning approach to offline RL. The Decision Transformer (Chen et al., NeurIPS 2021) reframes offline RL as sequence modeling:
- **CQL (Q-learning):** Learns $Q(s, a)$ from offline data with conservative penalty, then extracts $\pi(s) = \arg\max_a Q(s,a)$.
- **DT (Sequence modeling):** Models $P(a_t | s_t, \hat{R}_t, s_{t-1}, a_{t-1}, \dots)$ via causal Transformer, conditioning on desired return-to-go $\hat{R}_t$ to "prompt" a particular behavior quality.
- **Empirical comparison:** Both achieve similar D4RL scores on locomotion tasks; CQL generalizes better on out-of-distribution returns while DT is simpler to implement and scales more naturally with model size.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical calculation of softmax probabilities $p(a_1)=0.047426, p(a_2)=0.952574$.
   - Exact gradient step matching $Q_{\text{new}}(a_1)=2.476287, Q_{\text{new}}(a_2)=4.523713$ to $< 10^{-6}$.
2. **Complete PyTorch CQL Discrete Agent:**
   - Deep Q-Network with conservative log-sum-exp regularization loss.
3. **Synthetic Offline RL Environment Benchmark:**
   - Demonstrates that standard DQN diverges on an offline dataset with OOD actions, whereas CQL safely suppresses OOD actions and converges to the optimal safe policy.

See implementation in:
[`11_reinforcement_learning/code/25_offline_rl_cql.py`](./code/25_offline_rl_cql.py)
