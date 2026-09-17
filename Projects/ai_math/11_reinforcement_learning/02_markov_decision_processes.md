# Markov Decision Processes (MDPs) & Formal Definitions

---

## 1. Intuition & 101 Motivation

### From Bandits to Sequential Decision Making
In the Multi-Armed Bandit (Chapter 11.1), decisions were **non-associative**: an action yielded an immediate reward, but had **zero impact** on subsequent situations. The world reset identically after every pull.

In the real world, actions have profound delayed consequences:
1. An action taken now alters the **state** of the universe.
2. A move in chess does not yield immediate points, but positions pieces to checkmate twenty turns later.
3. Steering a car changes its physical velocity and position, altering future risk.

```
                  THE SEQUENTIAL REINFORCEMENT LEARNING LOOP
                         Action A_t
                   ┌────────────────────┐
                   ▼                    │
         ┌───────────────────┐    ┌───────────┐
         │    Environment    │    │   Agent   │
         └───────────────────┘    └───────────┘
              │         │               ▲
     State S_{t+1}   Reward R_{t+1}     │
              └─────────┴───────────────┘
```

The **Markov Decision Process (MDP)** provides the formal, universal mathematical framework for modeling all sequential decision problems under uncertainty, where actions influence both immediate rewards and future state transitions.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Formal MDP 5-Tuple
A discrete-time Markov Decision Process is formally defined by the 5-tuple:
$$\mathcal{M} = \left( \mathcal{S}, \, \mathcal{A}, \, \mathcal{P}, \, \mathcal{R}, \, \gamma \right)$$

1. **State Space $\mathcal{S}$:** The set of all valid environment states. Can be discrete ($|\mathcal{S}| < \infty$) or a continuous manifold $\mathcal{S} \subseteq \mathbb{R}^D$.
2. **Action Space $\mathcal{A}$:** The set of all valid decisions available to the agent (or state-conditioned action sets $\mathcal{A}(s)$).
3. **Transition Probability Kernel $\mathcal{P}$:**
   The conditional probability distribution governing environment dynamics:
   $$\mathcal{P}(s' \mid s, a) \equiv \mathbb{P}\left( S_{t+1} = s' \mid S_t = s, A_t = a \right)$$
   For discrete states, $\mathcal{P}$ is a set of row-stochastic matrices satisfying:
   $$\sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) = 1, \quad \forall s \in \mathcal{S}, \, a \in \mathcal{A}$$
4. **Reward Function $\mathcal{R}$:**
   The expected immediate scalar feedback emitted by the environment:
   $$\mathcal{R}(s, a) \equiv \mathbb{E}\left[ R_{t+1} \mid S_t = s, A_t = a \right]$$
   or in its three-argument form $\mathcal{R}(s, a, s') \equiv \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$.
5. **Discount Factor $\gamma \in [0, 1)$:**
   A real number quantifying the present value of future rewards.

---

### 2.2 The Markov Property

**Definition (The Markov Property):**
A state $S_t$ possesses the Markov Property if and only if the future state and reward depend **strictly and solely** on the present state and action, being conditionally independent of the entire preceding history:
$$\mathbf{\mathbb{P}\left( S_{t+1} = s', R_{t+1} = r \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0 \right) = \mathbb{P}\left( S_{t+1} = s', R_{t+1} = r \mid S_t, A_t \right)}$$

$$\text{"The future is independent of the past, given the present."}$$

If an environment state contains all historical information necessary to predict the transition distribution, the state is a **sufficient statistic** of the history:
$$S_t \equiv f(\mathcal{H}_t), \quad \text{where } \mathcal{H}_t = (S_0, A_0, R_1, \dots, S_t)$$

---

### 2.3 Trajectories & The Discounted Return

An agent interacting with an MDP generates an alternating sequence of states, actions, and rewards termed a **Trajectory** (or Episode):
$$\tau = \left( S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots, S_T \right)$$

#### The Return $G_t$
The objective of the agent is to maximize the expected cumulative **Discounted Return** $G_t$ from time step $t$ onward:
$$G_t \equiv R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

#### Recursive Formulation of the Return
$$G_t = R_{t+1} + \gamma \left( R_{t+2} + \gamma R_{t+3} + \dots \right) = \mathbf{R_{t+1} + \gamma G_{t+1}}$$
This recursive decomposition is the foundational algebraic building block of all dynamic programming and temporal-difference learning!

#### Mathematical Rationale for the Discount Factor $\gamma$
1. **Convergence of Infinite Series:**
   If rewards are bounded $|R_t| \le R_{\max}$, then for $\gamma < 1$, the return is guaranteed to be finite:
   $$|G_t| \le \sum_{k=0}^\infty \gamma^k |R_{t+k+1}| \le R_{\max} \sum_{k=0}^\infty \gamma^k = \frac{R_{\max}}{1 - \gamma} < \infty$$
2. **Modeling Stochastic Mortality:**
   A discount factor $\gamma$ is mathematically equivalent to assuming an undiscounted process with a constant termination probability of $(1 - \gamma)$ at every time step.
3. **Animal & Economic Behavioral Alignment:**
   Immediate rewards offer higher utility and certainty than delayed, distant rewards.

---

### 2.4 Policies & The Induced Markov Reward Process (MRP)

A **Policy** $\pi$ specifies the agent's decision rule:
- **Deterministic Policy:** A mapping from state space to action space: $a = \pi(s)$.
- **Stochastic Policy:** A conditional probability distribution over actions given the current state:
  $$\pi(a \mid s) \equiv \mathbb{P}\left( A_t = a \mid S_t = s \right), \quad \text{where } \sum_{a \in \mathcal{A}} \pi(a \mid s) = 1$$

#### The Induced Markov Chain & MRP
When an agent fixes its policy $\pi$, the control problem collapses into a passive stochastic process called a **Markov Reward Process (MRP)**:
$$\mathcal{M}^\pi = \left( \mathcal{S}, \, \mathcal{P}^\pi, \, \mathcal{R}^\pi, \, \gamma \right)$$
where the induced state-to-state transition probability is:
$$\mathbf{\mathcal{P}^\pi(s' \mid s) \equiv \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{P}(s' \mid s, a)}$$
and the induced expected reward vector is:
$$\mathbf{\mathcal{R}^\pi(s) \equiv \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{R}(s, a)}$$

In matrix notation for $|\mathcal{S}| = n$ discrete states:
$$\mathbf{P}^\pi = \begin{bmatrix} \mathcal{P}^\pi(s_1 \mid s_1) & \cdots & \mathcal{P}^\pi(s_n \mid s_1) \\ \vdots & \ddots & \vdots \\ \mathcal{P}^\pi(s_1 \mid s_n) & \cdots & \mathcal{P}^\pi(s_n \mid s_n) \end{bmatrix}, \quad \mathbf{r}^\pi = \begin{bmatrix} \mathcal{R}^\pi(s_1) \\ \vdots \\ \mathcal{R}^\pi(s_n) \end{bmatrix}$$
$\mathbf{P}^\pi$ is a valid row-stochastic matrix ($\mathbf{P}^\pi \mathbf{1} = \mathbf{1}$).

---

### 2.5 Stationary Distributions & State Visitation Frequencies

#### Stationary Distribution (Ergodic MDP)
If the Markov chain $\mathbf{P}^\pi$ is irreducible and aperiodic, there exists a unique **Stationary Distribution** $\mathbf{d}^\pi \in \Delta^{|\mathcal{S}|-1}$ satisfying:
$$\mathbf{d}^\pi \mathbf{P}^\pi = \mathbf{d}^\pi, \quad \sum_{s \in \mathcal{S}} d^\pi(s) = 1$$
$\mathbf{d}^\pi$ is the normalized left-eigenvector of $\mathbf{P}^\pi$ associated with eigenvalue $\lambda = 1$.

#### Discounted State Visitation Distribution
In discounted continuing environments starting from initial distribution $S_0 \sim \rho_0(s)$, the discounted visitation frequency is:
$$d^\pi(s) \equiv (1 - \gamma) \sum_{t=0}^\infty \gamma^t \mathbb{P}\left( S_t = s \mid S_0 \sim \rho_0, \pi \right)$$
In vector notation:
$$\mathbf{d}^\pi = (1 - \gamma) \boldsymbol{\rho}_0^\top \sum_{t=0}^\infty \left( \gamma \mathbf{P}^\pi \right)^t = (1 - \gamma) \boldsymbol{\rho}_0^\top \left( \mathbf{I} - \gamma \mathbf{P}^\pi \right)^{-1}$$

---

### 2.6 Partially Observable Markov Decision Processes (POMDPs)

In many real-world applications (e.g., poker, robotics with noisy LiDAR, medical diagnosis), the agent cannot perceive the true environment state $S_t$ directly; it only receives noisy, partial **Observations** $O_t$.

A **POMDP** is defined by the 7-tuple:
$$\mathcal{M}_{\text{POMDP}} = \left( \mathcal{S}, \, \mathcal{A}, \, \mathcal{P}, \, \mathcal{R}, \, \Omega, \, \mathcal{O}, \, \gamma \right)$$
where:
- $\Omega$ is the Observation Space.
- $\mathcal{O}(o \mid s', a) \equiv \mathbb{P}(O_{t+1} = o \mid S_{t+1} = s', A_t = a)$ is the Observation Emission Function.

```
                           THE POMDP ARCHITECTURE
                                    Action A_t
                             ┌──────────────────────┐
                             │                      ▼
                      ┌───────────────┐     ┌───────────────┐
                      │  State S_t    │────►│ State S_{t+1} │ (Hidden Reality)
                      └───────────────┘     └───────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │  Obs O_{t+1}  │ (Noisy Sensor)
                                            └───────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │ Agent Belief  │
                                            │    b'(s')     │
                                            └───────────────┘
```

#### The Belief State $b(s)$
Because the raw observations are non-Markovian, the agent maintains a **Belief State** $b \in \Delta^{|\mathcal{S}|-1}$, which is a probability distribution over the hidden states:
$$b(s) \equiv \mathbb{P}\left( S_t = s \mid \mathcal{H}_t \right), \quad \text{where } \mathcal{H}_t = (a_0, o_1, a_1, o_2, \dots, a_{t-1}, o_t)$$

#### Exact Belief Update Rule (Bayes' Rule)
Upon taking action $a$ and observing $o$, the updated belief state $b'(s')$ is:
$$\mathbf{b'(s') = \frac{\mathcal{O}(o \mid s', a) \sum_{s \in \mathcal{S}} \mathcal{P}(s' \mid s, a) b(s)}{\sum_{s'' \in \mathcal{S}} \mathcal{O}(o \mid s'', a) \sum_{s \in \mathcal{S}} \mathcal{P}(s'' \mid s, a) b(s)}}$$

**Fundamental Theorem of POMDPs:**
Any POMDP can be transformed into a fully observable, continuous-state **Belief MDP** where the state space is the simplex of probability distributions $\mathcal{B} = \Delta^{|\mathcal{S}|-1}$!

---

### 2.7 First-Principles Mathematical Derivations

#### Derivation 11.2.1: MRP Value Operator Invertibility and the Neumann Series Expansion

##### Problem Statement & Goal
In any finite Markov Reward Process (MRP) $\mathcal{M} = (\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma)$ with state space $|\mathcal{S}| = n$, row-stochastic transition matrix $\mathbf{P} \in \mathbb{R}^{n \times n}$, and discount factor $\gamma \in [0, 1)$, the state-value function satisfies the linear Bellman expectation equation:
$$\mathbf{V} = \mathbf{r} + \gamma \mathbf{P} \mathbf{V}$$
We prove from first principles that:
1. The matrix $(\mathbf{I} - \gamma \mathbf{P})$ is strictly non-singular (invertible).
2. The matrix inverse admits the convergent Neumann power series:
   $$(\mathbf{I} - \gamma \mathbf{P})^{-1} = \sum_{k=0}^\infty \gamma^k \mathbf{P}^k$$
3. The unique analytical state-value vector is given by $\mathbf{V} = (\mathbf{I} - \gamma \mathbf{P})^{-1} \mathbf{r}$ and satisfies the uniform supremum bound $\|\mathbf{V}\|_\infty \le \frac{\|\mathbf{r}\|_\infty}{1 - \gamma}$.

##### Explicit Assumptions
1. **Finite State Space:** The state space $\mathcal{S} = \{s_1, \dots, s_n\}$ has cardinality $|\mathcal{S}| = n < \infty$.
2. **Row-Stochasticity of $\mathbf{P}$:** For all $i, j \in \{1, \dots, n\}$, $P_{ij} \ge 0$ and $\sum_{j=1}^n P_{ij} = 1$, which implies $\mathbf{P} \mathbf{1} = \mathbf{1}$.
3. **Contraction Discount Factor:** The discount factor satisfies $0 \le \gamma < 1$.
4. **Bounded Immediate Rewards:** The reward vector $\mathbf{r} \in \mathbb{R}^n$ satisfies $\|\mathbf{r}\|_\infty = \max_{i} |r_i| \le R_{\max} < \infty$.

##### Underlying Intuition
A row-stochastic matrix models probability conservation: no probability mass can expand or escape. Consequently, its eigenvalues cannot exceed 1 in absolute value. Multiplying by $\gamma < 1$ shrinks the operator, pulling all eigenvalues strictly inside the open unit disk $\{z \in \mathbb{C} : |z| < 1\}$. Just as the scalar geometric series $\sum_{k=0}^\infty x^k = \frac{1}{1-x}$ converges whenever $|x| < 1$, the matrix power series $\sum_{k=0}^\infty (\gamma \mathbf{P})^k$ converges absolutely to the operator inverse $(\mathbf{I} - \gamma \mathbf{P})^{-1}$.

##### End-to-End Mathematical Derivation

**Step 1: Spectral Radius Bound of Row-Stochastic $\mathbf{P}$**
Let $\|\cdot\|_\infty$ denote the induced matrix $\infty$-norm:
$$\|\mathbf{P}\|_\infty = \max_{1 \le i \le n} \sum_{j=1}^n |P_{ij}|$$
Because $\mathbf{P}$ is row-stochastic, $P_{ij} \ge 0$ and $\sum_{j=1}^n P_{ij} = 1$ for all $i$. Therefore:
$$\|\mathbf{P}\|_\infty = \max_{1 \le i \le n} 1 = 1$$
By the spectral radius theorem, the spectral radius $\rho(\mathbf{P}) = \max_{\lambda \in \sigma(\mathbf{P})} |\lambda|$ is bounded above by any induced matrix norm:
$$\rho(\mathbf{P}) \le \|\mathbf{P}\|_\infty = 1$$
Thus, for every eigenvalue $\lambda \in \sigma(\mathbf{P})$, $|\lambda| \le 1$.

**Step 2: Spectral Radius of Scaled Operator $\gamma \mathbf{P}$**
Let $\mathbf{v} \ne \mathbf{0}$ be an eigenvector of $\mathbf{P}$ corresponding to eigenvalue $\lambda$. Then:
$$(\gamma \mathbf{P}) \mathbf{v} = \gamma (\mathbf{P} \mathbf{v}) = \gamma (\lambda \mathbf{v}) = (\gamma \lambda) \mathbf{v}$$
Hence, the spectrum of $\gamma \mathbf{P}$ is $\sigma(\gamma \mathbf{P}) = \{ \gamma \lambda \mid \lambda \in \sigma(\mathbf{P}) \}$.
Because $0 \le \gamma < 1$ and $|\lambda| \le 1$:
$$\rho(\gamma \mathbf{P}) = \max_{\lambda \in \sigma(\mathbf{P})} |\gamma \lambda| = \gamma \rho(\mathbf{P}) \le \gamma < 1$$

**Step 3: Strict Non-Singularity of $(\mathbf{I} - \gamma \mathbf{P})$**
By the spectral mapping theorem, the eigenvalues of $(\mathbf{I} - \gamma \mathbf{P})$ are:
$$\sigma(\mathbf{I} - \gamma \mathbf{P}) = \{ 1 - \mu \mid \mu \in \sigma(\gamma \mathbf{P}) \} = \{ 1 - \gamma \lambda \mid \lambda \in \sigma(\mathbf{P}) \}$$
Assume for contradiction that $(\mathbf{I} - \gamma \mathbf{P})$ is singular. Then at least one eigenvalue must equal zero:
$$1 - \gamma \lambda = 0 \implies \gamma \lambda = 1 \implies |\gamma \lambda| = 1$$
However, from Step 2, $|\gamma \lambda| \le \gamma < 1$. This contradiction proves that $1 - \gamma \lambda \ne 0$ for all $\lambda \in \sigma(\mathbf{P})$.
Therefore:
$$\det(\mathbf{I} - \gamma \mathbf{P}) = \prod_{\lambda \in \sigma(\mathbf{P})} (1 - \gamma \lambda) \ne 0$$
which establishes that $(\mathbf{I} - \gamma \mathbf{P})$ is strictly non-singular and uniquely invertible.

**Step 4: Convergence of the Neumann Series**
Define the sequence of partial sum matrices:
$$\mathbf{S}_K \equiv \sum_{k=0}^K (\gamma \mathbf{P})^k = \mathbf{I} + \gamma \mathbf{P} + (\gamma \mathbf{P})^2 + \dots + (\gamma \mathbf{P})^K$$
Multiply $\mathbf{S}_K$ by $(\mathbf{I} - \gamma \mathbf{P})$:
$$(\mathbf{I} - \gamma \mathbf{P}) \mathbf{S}_K = (\mathbf{I} - \gamma \mathbf{P}) \sum_{k=0}^K (\gamma \mathbf{P})^k = \sum_{k=0}^K (\gamma \mathbf{P})^k - \sum_{k=0}^K (\gamma \mathbf{P})^{k+1} = \mathbf{I} - (\gamma \mathbf{P})^{K+1}$$
Because $\rho(\gamma \mathbf{P}) < 1$, by Gelfand's formula:
$$\lim_{K \to \infty} (\gamma \mathbf{P})^{K+1} = \mathbf{0}$$
Taking limits on both sides:
$$(\mathbf{I} - \gamma \mathbf{P}) \lim_{K \to \infty} \mathbf{S}_K = \mathbf{I} - \mathbf{0} = \mathbf{I}$$
Pre-multiplying by $(\mathbf{I} - \gamma \mathbf{P})^{-1}$:
$$\lim_{K \to \infty} \mathbf{S}_K = (\mathbf{I} - \gamma \mathbf{P})^{-1} = \sum_{k=0}^\infty \gamma^k \mathbf{P}^k$$

**Step 5: Analytical Solution and Supremum Bound**
Starting from Bellman expectation equation:
$$\mathbf{V} = \mathbf{r} + \gamma \mathbf{P} \mathbf{V} \iff (\mathbf{I} - \gamma \mathbf{P}) \mathbf{V} = \mathbf{r}$$
Since $(\mathbf{I} - \gamma \mathbf{P})$ is invertible:
$$\mathbf{V} = (\mathbf{I} - \gamma \mathbf{P})^{-1} \mathbf{r} = \sum_{k=0}^\infty \gamma^k \mathbf{P}^k \mathbf{r}$$
Taking the induced infinity-norm:
$$\|\mathbf{V}\|_\infty = \left\| \sum_{k=0}^\infty \gamma^k \mathbf{P}^k \mathbf{r} \right\|_\infty \le \sum_{k=0}^\infty \gamma^k \|\mathbf{P}^k\|_\infty \|\mathbf{r}\|_\infty$$
Since $\mathbf{P}^k$ is also row-stochastic for all $k \ge 0$, $\|\mathbf{P}^k\|_\infty = 1$. Hence:
$$\|\mathbf{V}\|_\infty \le \|\mathbf{r}\|_\infty \sum_{k=0}^\infty \gamma^k = \frac{\|\mathbf{r}\|_\infty}{1 - \gamma} \le \frac{R_{\max}}{1 - \gamma}$$
The solution is uniquely determined, computationally exact via matrix inversion $\mathcal{O}(n^3)$, and strictly bounded. $\blacksquare$

---

#### Derivation 11.2.2: The Discounted State Visitation Distribution & The Invariant Balance Equation

##### Problem Statement & Goal
For an MDP under fixed policy $\pi$ with initial state distribution $\boldsymbol{\rho}_0 \in \Delta^{n-1}$, the normalized discounted state visitation distribution is defined by:
$$d^\pi(s) \equiv (1 - \gamma) \sum_{t=0}^\infty \gamma^t \mathbb{P}\left( S_t = s \mid S_0 \sim \boldsymbol{\rho}_0, \, \pi \right)$$
We prove that:
1. $d^\pi(s)$ is a mathematically rigorous probability distribution: $d^\pi(s) \ge 0$ for all $s$ and $\sum_{s \in \mathcal{S}} d^\pi(s) = 1$.
2. In compact matrix notation, $\mathbf{d}^\pi = (1 - \gamma) \boldsymbol{\rho}_0^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$.
3. $d^\pi$ satisfies the fundamental discounted balance equation:
   $$d^\pi(s') = (1 - \gamma) \rho_0(s') + \gamma \sum_{s \in \mathcal{S}} d^\pi(s) \mathcal{P}^\pi(s' \mid s)$$

##### Explicit Assumptions
1. Discrete state space $|\mathcal{S}| = n < \infty$.
2. Initial state distribution $\boldsymbol{\rho}_0 = [\rho_0(s_1), \dots, \rho_0(s_n)]^\top$ satisfies $\rho_0(s) \ge 0$ and $\boldsymbol{\rho}_0^\top \mathbf{1} = \sum_{s} \rho_0(s) = 1$.
3. $\mathbf{P}^\pi$ is the induced row-stochastic transition matrix ($\mathbf{P}^\pi \ge \mathbf{0}$, $\mathbf{P}^\pi \mathbf{1} = \mathbf{1}$).
4. Discount factor satisfies $\gamma \in [0, 1)$.

##### Underlying Intuition
In ergodic un-discounted Markov chains, the stationary distribution satisfies $\mathbf{d} = \mathbf{d} \mathbf{P}$. In discounted reinforcement learning, discounting at rate $\gamma$ is mathematically isomorphic to an undiscounted process where at each time step, with probability $(1 - \gamma)$, the agent "dies" and restarts afresh from distribution $\boldsymbol{\rho}_0$. Thus, $d^\pi$ represents the exact equilibrium balance between exogenous initial restarts (weighted by $1 - \gamma$) and endogenous one-step transitions (weighted by $\gamma$).

##### End-to-End Mathematical Derivation

**Step 1: Expressing State Marginals in Vector Form**
Let $\mathbf{p}_t^\top \in \mathbb{R}^{1 \times n}$ denote the row vector of marginal state probabilities at time step $t$, where the $i$-th entry is $\mathbb{P}(S_t = s_i \mid S_0 \sim \boldsymbol{\rho}_0, \pi)$.
At $t = 0$:
$$\mathbf{p}_0^\top = \boldsymbol{\rho}_0^\top$$
By the Chapman-Kolmogorov equations and policy induction:
$$\mathbf{p}_1^\top = \mathbf{p}_0^\top \mathbf{P}^\pi = \boldsymbol{\rho}_0^\top \mathbf{P}^\pi$$
By mathematical induction, for any $t \ge 0$:
$$\mathbf{p}_t^\top = \boldsymbol{\rho}_0^\top (\mathbf{P}^\pi)^t$$

**Step 2: Vector Representation of $d^\pi$**
Assembling the discounted sum across all horizons $t \ge 0$:
$$\mathbf{d}^\pi = (1 - \gamma) \sum_{t=0}^\infty \gamma^t \mathbf{p}_t^\top = (1 - \gamma) \boldsymbol{\rho}_0^\top \sum_{t=0}^\infty (\gamma \mathbf{P}^\pi)^t$$
From Derivation 11.2.1, the Neumann series $\sum_{t=0}^\infty (\gamma \mathbf{P}^\pi)^t$ converges unconditionally to $(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$.
Therefore:
$$\mathbf{d}^\pi = (1 - \gamma) \boldsymbol{\rho}_0^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$$

**Step 3: Proof of Non-Negativity**
Because $\boldsymbol{\rho}_0 \ge \mathbf{0}$ and $\mathbf{P}^\pi \ge \mathbf{0}$, all powers $(\mathbf{P}^\pi)^t$ have purely non-negative elements. Since $\gamma \in [0, 1)$, the scalar factor $(1 - \gamma) \gamma^t > 0$ for all $t \ge 0$.
The sum of non-negative elements is non-negative:
$$d^\pi(s) \ge 0, \quad \forall s \in \mathcal{S}$$

**Step 4: Proof of Total Probability Conservation ($\sum_s d^\pi(s) = 1$)**
We evaluate the total probability mass by post-multiplying $\mathbf{d}^\pi$ by the all-ones vector $\mathbf{1}$:
$$\mathbf{d}^\pi \mathbf{1} = (1 - \gamma) \boldsymbol{\rho}_0^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{1}$$
Notice that because $\mathbf{P}^\pi$ is row-stochastic:
$$\mathbf{P}^\pi \mathbf{1} = \mathbf{1}$$
Consequently:
$$(\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{1} = \mathbf{I} \mathbf{1} - \gamma \mathbf{P}^\pi \mathbf{1} = \mathbf{1} - \gamma \mathbf{1} = (1 - \gamma) \mathbf{1}$$
Multiply both sides on the left by $(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$:
$$(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} (\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{1} = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} (1 - \gamma) \mathbf{1}$$
$$\mathbf{1} = (1 - \gamma) (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{1}$$
Dividing both sides by the non-zero scalar $(1 - \gamma)$:
$$(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{1} = \frac{1}{1 - \gamma} \mathbf{1}$$
Substitute this exact identity back into the probability sum:
$$\mathbf{d}^\pi \mathbf{1} = (1 - \gamma) \boldsymbol{\rho}_0^\top \left[ \frac{1}{1 - \gamma} \mathbf{1} \right] = \boldsymbol{\rho}_0^\top \mathbf{1} = 1$$
since $\boldsymbol{\rho}_0$ is a normalized probability distribution ($\boldsymbol{\rho}_0^\top \mathbf{1} = 1$).
Thus, $\sum_{s \in \mathcal{S}} d^\pi(s) = 1$ strictly holds.

**Step 5: Derivation of the Discounted Invariant Balance Equation**
Post-multiply the vector equation $\mathbf{d}^\pi = (1 - \gamma) \boldsymbol{\rho}_0^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$ by $(\mathbf{I} - \gamma \mathbf{P}^\pi)$:
$$\mathbf{d}^\pi (\mathbf{I} - \gamma \mathbf{P}^\pi) = (1 - \gamma) \boldsymbol{\rho}_0^\top$$
Expand the matrix multiplication:
$$\mathbf{d}^\pi - \gamma \mathbf{d}^\pi \mathbf{P}^\pi = (1 - \gamma) \boldsymbol{\rho}_0^\top \implies \mathbf{d}^\pi = (1 - \gamma) \boldsymbol{\rho}_0^\top + \gamma \mathbf{d}^\pi \mathbf{P}^\pi$$
Examining the $j$-th component corresponding to state $s' \in \mathcal{S}$:
$$d^\pi(s') = (1 - \gamma) \rho_0(s') + \gamma \sum_{s \in \mathcal{S}} d^\pi(s) \mathcal{P}^\pi(s' \mid s)$$
This proves the fundamental balance equation governing discounted state visitation in all policy evaluation and policy gradient theorems. $\blacksquare$

---

#### Derivation 11.2.3: Necessary and Sufficient Conditions for Policy Invariance under Potential-Based Reward Shaping

##### Problem Statement & Goal
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ be an MDP, and let $\mathcal{M}' = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}', \gamma)$ be a transformed MDP whose reward function is augmented by an auxiliary shaping term:
$$\mathcal{R}'(s, a, s') = \mathcal{R}(s, a, s') + F(s, a, s')$$
We prove Ng, Harada, and Russell's (1999) Theorem:
1. If the shaping function is chosen as the discounted potential difference:
   $$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
   for any bounded potential function $\Phi: \mathcal{S} \to \mathbb{R}$, then for all policies $\pi$ and all state-action pairs $(s, a)$:
   $$Q_{\mathcal{M}'}^\pi(s, a) = Q_{\mathcal{M}}^\pi(s, a) - \Phi(s)$$
   $$V_{\mathcal{M}'}^\pi(s) = V_{\mathcal{M}}^\pi(s) - \Phi(s)$$
2. The optimal policy set is completely invariant: $\Pi^*_{\mathcal{M}'} = \Pi^*_{\mathcal{M}}$.
3. Potential-based shaping is necessary: if $F(s, a, s')$ preserves optimal policies across all transition probability functions $\mathcal{P}$, then $F$ must take the potential-difference form.

##### Explicit Assumptions
1. Discrete or continuous state-action spaces $(\mathcal{S}, \mathcal{A})$.
2. Discount factor $\gamma \in [0, 1)$.
3. Potential function $\Phi: \mathcal{S} \to \mathbb{R}$ is uniformly bounded: $\sup_{s \in \mathcal{S}} |\Phi(s)| \le M < \infty$.
4. Immediate rewards are bounded: $\sup_{s, a, s'} |\mathcal{R}(s, a, s')| \le R_{\max} < \infty$.

##### Underlying Intuition
Along any trajectory of states $(S_0, S_1, S_2, \dots)$, the discounted sum of potential differences forms a telescoping series:
$$[\gamma \Phi(S_1) - \Phi(S_0)] + \gamma [\gamma \Phi(S_2) - \Phi(S_1)] + \gamma^2 [\gamma \Phi(S_3) - \Phi(S_2)] + \dots$$
Every intermediate term $\gamma^k \Phi(S_k)$ appears once with a positive sign and once with a negative sign. Consequently, all intermediate transitions cancel out completely, leaving solely $-\Phi(S_0)$! Because this extra return depends only on the starting state and is completely independent of the actions chosen by the agent along the path, every action's Q-value is shifted by the identical scalar offset $-\Phi(S_0)$. The relative ranking of actions remains unchanged.

##### End-to-End Mathematical Derivation

**Step 1: Telescoping Expansion of the Discounted Shaping Return**
Let $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \dots)$ be an infinite-horizon trajectory generated under policy $\pi$ starting from $(S_0, A_0) = (s, a)$.
The cumulative discounted shaped return is:
$$G_0' = \sum_{t=0}^\infty \gamma^t R'_{t+1} = \sum_{t=0}^\infty \gamma^t \left[ R_{t+1} + F(S_t, A_t, S_{t+1}) \right]$$
Substitute $F(S_t, A_t, S_{t+1}) = \gamma \Phi(S_{t+1}) - \Phi(S_t)$:
$$G_0' = \sum_{t=0}^\infty \gamma^t R_{t+1} + \sum_{t=0}^\infty \gamma^t \left[ \gamma \Phi(S_{t+1}) - \Phi(S_t) \right] = G_0 + \sum_{t=0}^\infty \left[ \gamma^{t+1} \Phi(S_{t+1}) - \gamma^t \Phi(S_t) \right]$$

**Step 2: Partial Sum Evaluation and Limiting Behavior**
Consider the $K$-step truncated partial sum of the shaping terms:
$$\mathcal{S}_K \equiv \sum_{t=0}^K \left[ \gamma^{t+1} \Phi(S_{t+1}) - \gamma^t \Phi(S_t) \right]$$
Expanding term-by-term:
$$\begin{aligned}
\mathcal{S}_K &= \left[ \gamma \Phi(S_1) - \Phi(S_0) \right] + \left[ \gamma^2 \Phi(S_2) - \gamma \Phi(S_1) \right] + \left[ \gamma^3 \Phi(S_3) - \gamma^2 \Phi(S_2) \right] + \dots + \left[ \gamma^{K+1} \Phi(S_{K+1}) - \gamma^K \Phi(S_K) \right] \\
&= -\Phi(S_0) + \gamma^{K+1} \Phi(S_{K+1})
\end{aligned}$$
Because $\Phi$ is uniformly bounded ($|\Phi(S_{K+1})| \le M$) and $\gamma \in [0, 1)$:
$$\lim_{K \to \infty} |\gamma^{K+1} \Phi(S_{K+1})| \le M \lim_{K \to \infty} \gamma^{K+1} = 0$$
Therefore, the infinite sum converges strictly:
$$\sum_{t=0}^\infty \left[ \gamma^{t+1} \Phi(S_{t+1}) - \gamma^t \Phi(S_t) \right] = -\Phi(S_0)$$
Substituting this back into the shaped return yields the trajectory-wise equality:
$$G_0' = G_0 - \Phi(S_0)$$

**Step 3: Action-Value and State-Value Transformations**
Take the conditional expectation under policy $\pi$ given initial state $S_0 = s$ and action $A_0 = a$:
$$\begin{aligned}
Q_{\mathcal{M}'}^\pi(s, a) &\equiv \mathbb{E}_\pi \left[ G_0' \mid S_0 = s, A_0 = a \right] \\
&= \mathbb{E}_\pi \left[ G_0 - \Phi(S_0) \mid S_0 = s, A_0 = a \right] \\
&= \mathbb{E}_\pi \left[ G_0 \mid S_0 = s, A_0 = a \right] - \mathbb{E}_\pi \left[ \Phi(s) \mid S_0 = s, A_0 = a \right] \\
&= Q_{\mathcal{M}}^\pi(s, a) - \Phi(s)
\end{aligned}$$
Similarly, for the state-value function $V_{\mathcal{M}'}^\pi(s) = \sum_{a} \pi(a \mid s) Q_{\mathcal{M}'}^\pi(s, a)$:
$$V_{\mathcal{M}'}^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ Q_{\mathcal{M}}^\pi(s, a) - \Phi(s) \right] = \left( \sum_{a \in \mathcal{A}} \pi(a \mid s) Q_{\mathcal{M}}^\pi(s, a) \right) - \Phi(s) \sum_{a \in \mathcal{A}} \pi(a \mid s)$$
Since $\sum_{a} \pi(a \mid s) = 1$:
$$V_{\mathcal{M}'}^\pi(s) = V_{\mathcal{M}}^\pi(s) - \Phi(s)$$

**Step 4: Invariance of Optimal Policies**
The optimal policy $\pi^*_{\mathcal{M}'}$ in the shaped MDP selects actions maximizing $Q_{\mathcal{M}'}^*(s, a)$:
$$\pi^*_{\mathcal{M}'}(s) \in \arg\max_{a \in \mathcal{A}} Q_{\mathcal{M}'}^*(s, a) = \arg\max_{a \in \mathcal{A}} \left[ Q_{\mathcal{M}}^*(s, a) - \Phi(s) \right]$$
Notice that $\Phi(s)$ does not depend on $a$. Subtracting a constant that depends purely on $s$ does not alter the argmax across actions:
$$\arg\max_{a \in \mathcal{A}} \left[ Q_{\mathcal{M}}^*(s, a) - \Phi(s) \right] = \arg\max_{a \in \mathcal{A}} Q_{\mathcal{M}}^*(s, a) = \pi^*_{\mathcal{M}}(s)$$
Furthermore:
$$V^*_{\mathcal{M}'}(s) = \max_{a \in \mathcal{A}} Q^*_{\mathcal{M}'}(s, a) = \max_{a \in \mathcal{A}} \left[ Q^*_{\mathcal{M}}(s, a) - \Phi(s) \right] = V^*_{\mathcal{M}}(s) - \Phi(s)$$
Hence, $\Pi^*_{\mathcal{M}'} = \Pi^*_{\mathcal{M}}$ for any arbitrary transition dynamics $\mathcal{P}$.

**Step 5: Necessity (Sketch)**
If $F(s, a, s')$ preserves optimal policies for all arbitrary transition models $\mathcal{P}$, consider single-step deterministic transitions to target states. Any cycle of states $s_0 \to s_1 \to \dots \to s_K \to s_0$ must satisfy $\sum_{t=0}^K \gamma^t F(s_t, a_t, s_{t+1}) = 0$ to prevent cyclical policy loops (reward hacking). By Kolmogorov's criterion for conservative potential fields, a cycle integral vanishes if and only if the vector field is the gradient/difference of a scalar potential $\Phi(s)$. Thus $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ is both necessary and sufficient. $\blacksquare$

---

## 3. Geometric & Physical Interpretation

### 1. The Markov Chain as a Strongly Connected Directed Graph
An MDP can be visualized as a directed bipartite graph:
- **State Nodes (Circles $\mathcal{S}$):** Points in space where the agent chooses an action.
- **Action Nodes (Dots $\mathcal{A}$):** Points where nature takes over, stochastically branching into child states according to probability distribution $\mathcal{P}(\cdot \mid s, a)$.

### 2. The Probability Simplex $\Delta^{|\mathcal{S}|-1}$
For an MDP with $n$ states, the belief state $b$ or stationary distribution $\mathbf{d}^\pi$ is constrained to live on the $(n-1)$-dimensional standard simplex:
$$\Delta^{n-1} = \left\{ \mathbf{p} \in \mathbb{R}^n \;\middle|\; \sum_{i=1}^n p_i = 1, \; p_i \ge 0 \right\}$$
Transitions under $\mathbf{P}^\pi$ act as geometric contractions that shrink any arbitrary distribution on the simplex toward the unique fixed-point attractor $\mathbf{d}^\pi$.

---

## 4. Real-World Analogy

### The Master Chess Grandmaster vs. The ICU Doctor
- **The MDP (Chess):** A grandmaster looks at the board. The position of all 32 pieces constitutes the complete state $S_t$. You do not need to know whether the opponent took 10 seconds or 2 hours to play their bishop, or what sequence of moves led to this board. The board state is **Markovian**: everything needed to plan the winning combination is visible right now.
- **The POMDP (Emergency Room Physician):** A patient arrives unconscious with abdominal pain. The doctor cannot see the internal organs (hidden states: appendicitis vs. kidney stone vs. food poisoning). The doctor measures vital signs (noisy observation $O_t$: blood pressure 130/80, temperature 38.5°C). The doctor maintains a **belief state** (60% probability appendicitis, 30% kidney stone, 10% other) and orders an ultrasound (action $A_t$) to update their belief state via Bayes' rule!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us compute an induced Markov Reward Process, trajectory return, and POMDP belief state update with exact concrete toy numbers.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in MDP / POMDP |
| :--- | :--- | :--- | :--- |
| $\mathcal{S}$ | State Set | 3 States | States $\{S_1, S_2, S_3\}$ representing environment conditions |
| $\mathcal{A}$ | Action Set | 2 Actions | Choices $\{a_1, a_2\}$ available to the agent |
| $\mathbf{P}_{a_1}, \mathbf{P}_{a_2}$ | Action Transition Matrices | $(3, 3)$ | State-to-state transition probabilities under specific actions |
| $\mathbf{r}_{a_1}, \mathbf{r}_{a_2}$ | Action Reward Vectors | $(3,)$ | Expected immediate reward emitted in each state under each action |
| $\boldsymbol{\pi}$ | Stochastic Policy | $(3, 2)$ | Action probabilities $\pi(a \mid s)$ for all state-action pairs |
| $\mathbf{P}^\pi$ | Induced Transition Matrix | $(3, 3)$ | Policy-weighted transition matrix $\sum_a \pi(a \mid s) \mathbf{P}_a$ |
| $\mathbf{r}^\pi$ | Induced Reward Vector | $(3,)$ | Policy-weighted reward vector $\sum_a \pi(a \mid s) \mathbf{r}_a$ |
| $\gamma$ | Discount Factor | Scalar | Value decay rate ($\gamma = 0.90$) |
| $\mathbf{b}$ | Prior Belief State | $(3,)$ | Probability distribution over hidden states $[b(S_1), b(S_2), b(S_3)]$ |
| $\mathbf{O}_{o_1}$ | Observation Emission Vector | $(3,)$ | Probabilities $\mathbb{P}(o_1 \mid S_i)$ of emitting sensor signal $o_1$ |
| $\mathbf{b}'$ | Posterior Belief State | $(3,)$ | Updated belief distribution after Bayesian conditioning |

---

### 5.2 Concrete Toy Setup

Let $\mathcal{S} = \{S_1, S_2, S_3\}$ and $\mathcal{A} = \{a_1, a_2\}$.

#### Transition Matrices:
$$\mathbf{P}_{a_1} = \begin{bmatrix} 0.6 & 0.4 & 0.0 \\ 0.0 & 0.7 & 0.3 \\ 0.2 & 0.0 & 0.8 \end{bmatrix}, \quad \mathbf{P}_{a_2} = \begin{bmatrix} 0.1 & 0.8 & 0.1 \\ 0.4 & 0.4 & 0.2 \\ 0.0 & 0.5 & 0.5 \end{bmatrix}$$

#### Reward Vectors:
$$\mathbf{r}_{a_1} = \begin{bmatrix} 5.0 \\ 2.0 \\ -1.0 \end{bmatrix}, \quad \mathbf{r}_{a_2} = \begin{bmatrix} 1.0 \\ 0.0 \\ 4.0 \end{bmatrix}$$

#### Agent Stochastic Policy $\pi(a \mid s)$:
- In $S_1$: $\pi(a_1 \mid S_1) = 0.80, \quad \pi(a_2 \mid S_1) = 0.20$
- In $S_2$: $\pi(a_1 \mid S_2) = 0.50, \quad \pi(a_2 \mid S_2) = 0.50$
- In $S_3$: $\pi(a_1 \mid S_3) = 0.10, \quad \pi(a_2 \mid S_3) = 0.90$

Discount factor: $\gamma = 0.90$.

---

### 5.3 Step 1: Induced Transition Matrix $\mathbf{P}^\pi$ Arithmetic

Compute row by row: $\mathbf{P}^\pi(s) = \pi(a_1 \mid s) \mathbf{P}_{a_1}(s) + \pi(a_2 \mid s) \mathbf{P}_{a_2}(s)$.

#### Row 1 ($S_1$):
$$\begin{aligned}
\mathbf{P}^\pi(S_1) &= 0.80 \begin{bmatrix} 0.6 & 0.4 & 0.0 \end{bmatrix} + 0.20 \begin{bmatrix} 0.1 & 0.8 & 0.1 \end{bmatrix} \\
&= \begin{bmatrix} 0.48 & 0.32 & 0.00 \end{bmatrix} + \begin{bmatrix} 0.02 & 0.16 & 0.02 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.50 & 0.48 & 0.02 \end{bmatrix}}
\end{aligned}$$
Verification: $0.50 + 0.48 + 0.02 = 1.00$.

#### Row 2 ($S_2$):
$$\begin{aligned}
\mathbf{P}^\pi(S_2) &= 0.50 \begin{bmatrix} 0.0 & 0.7 & 0.3 \end{bmatrix} + 0.50 \begin{bmatrix} 0.4 & 0.4 & 0.2 \end{bmatrix} \\
&= \begin{bmatrix} 0.00 & 0.35 & 0.15 \end{bmatrix} + \begin{bmatrix} 0.20 & 0.20 & 0.10 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.20 & 0.55 & 0.25 \end{bmatrix}}
\end{aligned}$$
Verification: $0.20 + 0.55 + 0.25 = 1.00$.

#### Row 3 ($S_3$):
$$\begin{aligned}
\mathbf{P}^\pi(S_3) &= 0.10 \begin{bmatrix} 0.2 & 0.0 & 0.8 \end{bmatrix} + 0.90 \begin{bmatrix} 0.0 & 0.5 & 0.5 \end{bmatrix} \\
&= \begin{bmatrix} 0.02 & 0.00 & 0.08 \end{bmatrix} + \begin{bmatrix} 0.00 & 0.45 & 0.45 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.02 & 0.45 & 0.53 \end{bmatrix}}
\end{aligned}$$
Verification: $0.02 + 0.45 + 0.53 = 1.00$.

$$\mathbf{P}^\pi = \begin{bmatrix} 0.50 & 0.48 & 0.02 \\ 0.20 & 0.55 & 0.25 \\ 0.02 & 0.45 & 0.53 \end{bmatrix}$$

---

### 5.4 Step 2: Induced Reward Vector $\mathbf{r}^\pi$ Arithmetic

$$r^\pi(s) = \pi(a_1 \mid s) r_{a_1}(s) + \pi(a_2 \mid s) r_{a_2}(s)$$
$$r^\pi(S_1) = 0.80(5.0) + 0.20(1.0) = 4.00 + 0.20 = \mathbf{4.2000}$$
$$r^\pi(S_2) = 0.50(2.0) + 0.50(0.0) = 1.00 + 0.00 = \mathbf{1.0000}$$
$$r^\pi(S_3) = 0.10(-1.0) + 0.90(4.0) = -0.10 + 3.60 = \mathbf{3.5000}$$

$$\mathbf{r}^\pi = \begin{bmatrix} 4.2000 \\ 1.0000 \\ 3.5000 \end{bmatrix}$$

---

### 5.5 Step 3: 4-Step Trajectory Return Calculation

Suppose an agent generates trajectory:
$$\tau = \left( S_1, \, R_1 = 4.0, \quad S_2, \, R_2 = 1.0, \quad S_3, \, R_3 = 3.0, \quad S_1, \, R_4 = 5.0 \right)$$
Discount factor: $\gamma = 0.90$.

Compute discounted returns from each time step backward:
1. **At $t = 3$ (after $S_3$):**
   $$G_3 = R_4 = \mathbf{5.0000}$$
2. **At $t = 2$ (after $S_2$):**
   $$G_2 = R_3 + \gamma G_3 = 3.0 + 0.90(5.0) = 3.0 + 4.50 = \mathbf{7.5000}$$
3. **At $t = 1$ (after $S_1$):**
   $$G_1 = R_2 + \gamma G_2 = 1.0 + 0.90(7.50) = 1.0 + 6.75 = \mathbf{7.7500}$$
4. **At $t = 0$ (Initial step):**
   $$G_0 = R_1 + \gamma G_1 = 4.0 + 0.90(7.75) = 4.0 + 6.975 = \mathbf{10.9750}$$

Direct expansion verification:
$$G_0 = 4.0 + 0.9(1.0) + 0.9^2(3.0) + 0.9^3(5.0) = 4.0 + 0.90 + 0.81(3.0) + 0.729(5.0) = 4.90 + 2.43 + 3.645 = \mathbf{10.9750}$$
Exact match!

---

### 5.6 Step 4: POMDP Bayesian Belief Update Arithmetic

Suppose state is partially observed.
Current prior belief state:
$$\mathbf{b} = \begin{bmatrix} b(S_1) \\ b(S_2) \\ b(S_3) \end{bmatrix} = \begin{bmatrix} 0.5000 \\ 0.3000 \\ 0.2000 \end{bmatrix}$$

1. The agent executes action $a_1$.
   The predicted state distribution before observation is:
   $$\mathbf{b}_{\text{pred}}^\top = \mathbf{b}^\top \mathbf{P}_{a_1} = \begin{bmatrix} 0.5 & 0.3 & 0.2 \end{bmatrix} \begin{bmatrix} 0.6 & 0.4 & 0.0 \\ 0.0 & 0.7 & 0.3 \\ 0.2 & 0.0 & 0.8 \end{bmatrix}$$
   $$b_{\text{pred}}(S_1) = 0.5(0.6) + 0.3(0.0) + 0.2(0.2) = 0.30 + 0.00 + 0.04 = \mathbf{0.3400}$$
   $$b_{\text{pred}}(S_2) = 0.5(0.4) + 0.3(0.7) + 0.2(0.0) = 0.20 + 0.21 + 0.00 = \mathbf{0.4100}$$
   $$b_{\text{pred}}(S_3) = 0.5(0.0) + 0.3(0.3) + 0.2(0.8) = 0.00 + 0.09 + 0.16 = \mathbf{0.2500}$$
   Verification: $0.34 + 0.41 + 0.25 = 1.00$.

2. The sensor emits observation $o_1$.
   The emission probabilities are:
   $$\mathbb{P}(o_1 \mid S_1) = 0.80, \quad \mathbb{P}(o_1 \mid S_2) = 0.20, \quad \mathbb{P}(o_1 \mid S_3) = 0.10$$

3. Compute unnormalized posterior joint probability:
   $$\tilde{b}(S_1) = \mathbb{P}(o_1 \mid S_1) b_{\text{pred}}(S_1) = 0.80 \times 0.34 = \mathbf{0.2720}$$
   $$\tilde{b}(S_2) = \mathbb{P}(o_1 \mid S_2) b_{\text{pred}}(S_2) = 0.20 \times 0.41 = \mathbf{0.0820}$$
   $$\tilde{b}(S_3) = \mathbb{P}(o_1 \mid S_3) b_{\text{pred}}(S_3) = 0.10 \times 0.25 = \mathbf{0.0250}$$

4. Normalizing Constant (Evidence):
   $$P(o_1) = \sum_s \tilde{b}(s) = 0.2720 + 0.0820 + 0.0250 = \mathbf{0.3790}$$

5. Posterior Belief State $\mathbf{b}'$:
   $$b'(S_1) = \frac{0.2720}{0.3790} \approx \mathbf{0.7177}$$
   $$b'(S_2) = \frac{0.0820}{0.3790} \approx \mathbf{0.2164}$$
   $$b'(S_3) = \frac{0.0250}{0.3790} \approx \mathbf{0.0660}$$
   Verification: $0.7177 + 0.2164 + 0.0660 = 1.0001 \approx 1.0000$.

Because observation $o_1$ is strongly indicative of state $S_1$ (80% probability), our belief that the system is in $S_1$ jumped from $34\% \to \mathbf{71.77\%}$!

---

## 6. Solved Illustrations

### Illustration 1: Converting Non-Markovian Observations to Markovian (Frame Stacking)
**Problem:** A video game agent observes a single grayscale image $I_t \in \mathbb{R}^{84 \times 84}$ of a Pong ball at position $(x, y)$. Why is this single frame non-Markovian, and how does **Frame Stacking** restore the Markov Property?
**Solution:**
From a single still frame $I_t$, the position $(x_t, y_t)$ is known, but the velocity vector $(v_x, v_y) = \frac{d\mathbf{x}}{dt}$ and direction of motion are completely undetermined.
The probability of next ball position $S_{t+1}$ depends on past frame $I_{t-1}$:
$$\mathbb{P}(S_{t+1} \mid I_t, I_{t-1}) \ne \mathbb{P}(S_{t+1} \mid I_t)$$
The state is non-Markovian.
**Solution (Mnih et al., 2015):** Augment the state space by stacking the $m = 4$ most recent frames:
$$\tilde{S}_t \equiv \left[ I_t, \, I_{t-1}, \, I_{t-2}, \, I_{t-3} \right] \in \mathbb{R}^{4 \times 84 \times 84}$$
Finite difference approximations:
$$\text{Velocity: } \mathbf{v}_t \approx I_t - I_{t-1}$$
$$\text{Acceleration: } \mathbf{a}_t \approx (I_t - I_{t-1}) - (I_{t-1} - I_{t-2}) = I_t - 2I_{t-1} + I_{t-2}$$
Now, $\tilde{S}_t$ contains position, velocity, and acceleration. All physical dynamics are completely captured, strictly restoring the **Markov Property**!

---

### Illustration 2: Analytical Solution of Stationary Distribution $\mathbf{d}^\pi$
**Problem:** Let a 2-state Markov chain have transition matrix:
$$\mathbf{P} = \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}$$
Find the stationary distribution $\mathbf{d} = [d_1, d_2]$.
**Solution:**
The stationary distribution satisfies $\mathbf{d} \mathbf{P} = \mathbf{d}$ and $d_1 + d_2 = 1$.
$$\begin{bmatrix} d_1 & d_2 \end{bmatrix} \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix} = \begin{bmatrix} d_1 & d_2 \end{bmatrix}$$
Row 1 equation:
$$0.7 d_1 + 0.4 d_2 = d_1 \implies 0.4 d_2 = 0.3 d_1 \implies d_1 = \frac{4}{3} d_2$$
Substitute into normalization constraint $d_1 + d_2 = 1$:
$$\frac{4}{3} d_2 + d_2 = 1 \implies \frac{7}{3} d_2 = 1 \implies d_2 = \mathbf{\frac{3}{7}} \approx 0.4286$$
$$d_1 = \frac{4}{3} \left( \frac{3}{7} \right) = \mathbf{\frac{4}{7}} \approx 0.5714$$

Stationary distribution: $\mathbf{d} = \begin{bmatrix} \frac{4}{7} & \frac{3}{7} \end{bmatrix} \approx \begin{bmatrix} 0.5714 & 0.4286 \end{bmatrix}$.

---

### Illustration 3: Exact Matrix Inversion and MRP State-Value Function Calculation

**Problem:**
Consider a 3-state Markov Reward Process $\mathcal{S} = \{S_1, S_2, S_3\}$ with discount factor $\gamma = 0.50$, cyclic-transition matrix:
$$\mathbf{P} = \begin{bmatrix} 0.5 & 0.5 & 0.0 \\ 0.0 & 0.5 & 0.5 \\ 0.5 & 0.0 & 0.5 \end{bmatrix}$$
and immediate reward vector $\mathbf{r} = [10.0, 0.0, -2.0]^\top$.
Find the exact matrix inverse $(\mathbf{I} - \gamma \mathbf{P})^{-1}$ by hand using Cramer's rule / adjugate method, and calculate the exact state-value vector $\mathbf{V}$.

**Solution:**

**Step 1: Construct the Operator Matrix $\mathbf{A} = \mathbf{I} - \gamma \mathbf{P}$**
With $\gamma = \frac{1}{2}$:
$$\gamma \mathbf{P} = \frac{1}{2} \begin{bmatrix} \frac{1}{2} & \frac{1}{2} & 0 \\ 0 & \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & 0 & \frac{1}{2} \end{bmatrix} = \begin{bmatrix} \frac{1}{4} & \frac{1}{4} & 0 \\ 0 & \frac{1}{4} & \frac{1}{4} \\ \frac{1}{4} & 0 & \frac{1}{4} \end{bmatrix}$$
$$\mathbf{A} = \mathbf{I} - \gamma \mathbf{P} = \begin{bmatrix} 1 - \frac{1}{4} & -\frac{1}{4} & 0 \\ 0 & 1 - \frac{1}{4} & -\frac{1}{4} \\ -\frac{1}{4} & 0 & 1 - \frac{1}{4} \end{bmatrix} = \begin{bmatrix} \frac{3}{4} & -\frac{1}{4} & 0 \\ 0 & \frac{3}{4} & -\frac{1}{4} \\ -\frac{1}{4} & 0 & \frac{3}{4} \end{bmatrix}$$

**Step 2: Compute Determinant $\det(\mathbf{A})$**
Expand along the first row:
$$\begin{aligned}
\det(\mathbf{A}) &= \frac{3}{4} \det \begin{bmatrix} \frac{3}{4} & -\frac{1}{4} \\ 0 & \frac{3}{4} \end{bmatrix} - \left(-\frac{1}{4}\right) \det \begin{bmatrix} 0 & -\frac{1}{4} \\ -\frac{1}{4} & \frac{3}{4} \end{bmatrix} + 0 \\
&= \frac{3}{4} \left( \frac{9}{16} - 0 \right) + \frac{1}{4} \left( 0 - \frac{1}{16} \right) \\
&= \frac{27}{64} - \frac{1}{64} = \mathbf{\frac{26}{64} = \frac{13}{32} = 0.40625}
\end{aligned}$$
Since $\det(\mathbf{A}) = \frac{13}{32} \ne 0$, $\mathbf{A}$ is strictly invertible.

**Step 3: Compute the Adjugate Matrix $\text{adj}(\mathbf{A})$**
The matrix of cofactors $\mathbf{C}$ has elements $C_{ij} = (-1)^{i+j} M_{ij}$:
- $C_{11} = +\left( \frac{9}{16} - 0 \right) = \frac{9}{16}$
- $C_{12} = -\left( 0 - \frac{1}{16} \right) = \frac{1}{16}$
- $C_{13} = +\left( 0 - \left(-\frac{3}{16}\right) \right) = \frac{3}{16}$
- $C_{21} = -\left( -\frac{3}{16} - 0 \right) = \frac{3}{16}$
- $C_{22} = +\left( \frac{9}{16} - 0 \right) = \frac{9}{16}$
- $C_{23} = -\left( 0 - \frac{1}{16} \right) = \frac{1}{16}$
- $C_{31} = +\left( \frac{1}{16} - 0 \right) = \frac{1}{16}$
- $C_{32} = -\left( -\frac{3}{16} - 0 \right) = \frac{3}{16}$
- $C_{33} = +\left( \frac{9}{16} - 0 \right) = \frac{9}{16}$

The adjugate is the transpose of the cofactor matrix $\text{adj}(\mathbf{A}) = \mathbf{C}^\top$:
$$\text{adj}(\mathbf{A}) = \begin{bmatrix} \frac{9}{16} & \frac{3}{16} & \frac{1}{16} \\ \frac{1}{16} & \frac{9}{16} & \frac{3}{16} \\ \frac{3}{16} & \frac{1}{16} & \frac{9}{16} \end{bmatrix} = \frac{1}{16} \begin{bmatrix} 9 & 3 & 1 \\ 1 & 9 & 3 \\ 3 & 1 & 9 \end{bmatrix}$$

**Step 4: Form the Inverse $(\mathbf{I} - \gamma \mathbf{P})^{-1}$**
$$(\mathbf{I} - \gamma \mathbf{P})^{-1} = \frac{1}{\det(\mathbf{A})} \text{adj}(\mathbf{A}) = \frac{32}{13} \cdot \frac{1}{16} \begin{bmatrix} 9 & 3 & 1 \\ 1 & 9 & 3 \\ 3 & 1 & 9 \end{bmatrix} = \mathbf{\frac{1}{13} \begin{bmatrix} 18 & 6 & 2 \\ 2 & 18 & 6 \\ 6 & 2 & 18 \end{bmatrix}}$$

**Step 5: Compute State-Value Vector $\mathbf{V} = (\mathbf{I} - \gamma \mathbf{P})^{-1} \mathbf{r}$**
$$\mathbf{V} = \frac{1}{13} \begin{bmatrix} 18 & 6 & 2 \\ 2 & 18 & 6 \\ 6 & 2 & 18 \end{bmatrix} \begin{bmatrix} 10 \\ 0 \\ -2 \end{bmatrix} = \frac{1}{13} \begin{bmatrix} 18(10) + 6(0) + 2(-2) \\ 2(10) + 18(0) + 6(-2) \\ 6(10) + 2(0) + 18(-2) \end{bmatrix} = \frac{1}{13} \begin{bmatrix} 180 - 4 \\ 20 - 12 \\ 60 - 36 \end{bmatrix} = \mathbf{\begin{bmatrix} \frac{176}{13} \\ \frac{8}{13} \\ \frac{24}{13} \end{bmatrix} \approx \begin{bmatrix} 13.5385 \\ 0.6154 \\ 1.8462 \end{bmatrix}}$$

**Verification via Bellman Equation $\mathbf{V} = \mathbf{r} + \gamma \mathbf{P} \mathbf{V}$:**
- $V(S_1) = 10 + 0.5 \left( 0.5 \cdot \frac{176}{13} + 0.5 \cdot \frac{8}{13} \right) = 10 + 0.5 \left( \frac{92}{13} \right) = 10 + \frac{46}{13} = \frac{176}{13}$ (Exact match!)
- $V(S_2) = 0 + 0.5 \left( 0.5 \cdot \frac{8}{13} + 0.5 \cdot \frac{24}{13} \right) = 0.5 \left( \frac{16}{13} \right) = \frac{8}{13}$ (Exact match!)
- $V(S_3) = -2 + 0.5 \left( 0.5 \cdot \frac{176}{13} + 0.5 \cdot \frac{24}{13} \right) = -2 + 0.5 \left( \frac{100}{13} \right) = -\frac{26}{13} + \frac{50}{13} = \frac{24}{13}$ (Exact match!)

---

### Illustration 4: Step-by-Step Discounted State Visitation Frequency Vector $\mathbf{d}^\pi$ Calculation

**Problem:**
For the 3-state MRP of Illustration 3 ($\gamma = 0.50$, transition matrix $\mathbf{P}$), assume the agent always begins in state $S_1$, so the initial distribution is:
$$\boldsymbol{\rho}_0 = \begin{bmatrix} 1.0 \\ 0.0 \\ 0.0 \end{bmatrix}$$
1. Calculate the exact normalized discounted state visitation distribution $\mathbf{d}^\pi = [d(S_1), d(S_2), d(S_3)]$.
2. Verify that the entries sum to exactly $1.0000$.
3. Verify that $\mathbf{d}^\pi$ satisfies the discounted balance equation $d^\pi(s') = (1 - \gamma)\rho_0(s') + \gamma \sum_s d^\pi(s) \mathcal{P}(s' \mid s)$ for all three states.

**Solution:**

**Step 1: Compute $\mathbf{d}^\pi = (1 - \gamma) \boldsymbol{\rho}_0^\top (\mathbf{I} - \gamma \mathbf{P})^{-1}$**
From Illustration 3, the inverse operator is:
$$(\mathbf{I} - \gamma \mathbf{P})^{-1} = \frac{1}{13} \begin{bmatrix} 18 & 6 & 2 \\ 2 & 18 & 6 \\ 6 & 2 & 18 \end{bmatrix}$$
Since $\boldsymbol{\rho}_0^\top = [1, 0, 0]$ and $1 - \gamma = 1 - 0.5 = \frac{1}{2}$:
$$\mathbf{d}^\pi = \frac{1}{2} \begin{bmatrix} 1 & 0 & 0 \end{bmatrix} \left( \frac{1}{13} \begin{bmatrix} 18 & 6 & 2 \\ 2 & 18 & 6 \\ 6 & 2 & 18 \end{bmatrix} \right) = \frac{1}{26} \begin{bmatrix} 18 & 6 & 2 \end{bmatrix} = \mathbf{\begin{bmatrix} \frac{9}{13} & \frac{3}{13} & \frac{1}{13} \end{bmatrix}}$$
In decimal approximations:
$$\mathbf{d}^\pi \approx \begin{bmatrix} 0.6923 & 0.2308 & 0.0769 \end{bmatrix}$$

**Step 2: Verify Total Probability Conservation**
$$\sum_{s \in \mathcal{S}} d^\pi(s) = \frac{9}{13} + \frac{3}{13} + \frac{1}{13} = \frac{13}{13} = \mathbf{1.0000}$$
Total probability is strictly conserved.

**Step 3: Verification of the Discounted Invariant Balance Equation**
The balance equation states:
$$d^\pi(s') = (1 - \gamma) \rho_0(s') + \gamma \sum_{s \in \mathcal{S}} d^\pi(s) P(s' \mid s)$$

- **For State $S_1$:**
  $$d^\pi(S_1) = \frac{1}{2}(1) + \frac{1}{2} \left[ d^\pi(S_1) P(S_1 \mid S_1) + d^\pi(S_2) P(S_1 \mid S_2) + d^\pi(S_3) P(S_1 \mid S_3) \right]$$
  $$= \frac{1}{2} + \frac{1}{2} \left[ \frac{9}{13}(0.5) + \frac{3}{13}(0.0) + \frac{1}{13}(0.5) \right] = \frac{1}{2} + \frac{1}{2} \left[ \frac{4.5 + 0.5}{13} \right] = \frac{1}{2} + \frac{1}{2} \left( \frac{5}{13} \right) = \frac{1}{2} + \frac{5}{26} = \frac{18}{26} = \mathbf{\frac{9}{13}}$$
  Matches $d^\pi(S_1) = \frac{9}{13}$!

- **For State $S_2$:**
  $$d^\pi(S_2) = \frac{1}{2}(0) + \frac{1}{2} \left[ d^\pi(S_1) P(S_2 \mid S_1) + d^\pi(S_2) P(S_2 \mid S_2) + d^\pi(S_3) P(S_2 \mid S_3) \right]$$
  $$= 0 + \frac{1}{2} \left[ \frac{9}{13}(0.5) + \frac{3}{13}(0.5) + \frac{1}{13}(0.0) \right] = \frac{1}{2} \left[ \frac{4.5 + 1.5}{13} \right] = \frac{1}{2} \left( \frac{6}{13} \right) = \mathbf{\frac{3}{13}}$$
  Matches $d^\pi(S_2) = \frac{3}{13}$!

- **For State $S_3$:**
  $$d^\pi(S_3) = \frac{1}{2}(0) + \frac{1}{2} \left[ d^\pi(S_1) P(S_3 \mid S_1) + d^\pi(S_2) P(S_3 \mid S_2) + d^\pi(S_3) P(S_3 \mid S_3) \right]$$
  $$= 0 + \frac{1}{2} \left[ \frac{9}{13}(0.0) + \frac{3}{13}(0.5) + \frac{1}{13}(0.5) \right] = \frac{1}{2} \left[ \frac{1.5 + 0.5}{13} \right] = \frac{1}{2} \left( \frac{2}{13} \right) = \mathbf{\frac{1}{13}}$$
  Matches $d^\pi(S_3) = \frac{1}{13}$!
All three balance equations are satisfied identically.

---

### Illustration 5: Numerical Verification of Policy Invariance under Potential-Based Reward Shaping

**Problem:**
Consider a 2-state MDP with $\mathcal{S} = \{S_1, S_2\}$, $\mathcal{A} = \{a_1, a_2\}$, and discount factor $\gamma = 0.80$.
The transitions and immediate rewards are deterministic:
- In $S_1$:
  - Taking $a_1$: transitions to $S_2$, yielding reward $R(S_1, a_1, S_2) = 5.0$.
  - Taking $a_2$: transitions to $S_1$, yielding reward $R(S_1, a_2, S_1) = 2.0$.
- In $S_2$:
  - Taking $a_1$: transitions to $S_1$, yielding reward $R(S_2, a_1, S_1) = -1.0$.
  - Taking $a_2$: transitions to $S_2$, yielding reward $R(S_2, a_2, S_2) = 0.0$.

Let the potential function be:
$$\Phi(S_1) = 10.0, \quad \Phi(S_2) = 2.0$$

1. Solve for the unshaped optimal values $V^*(s)$, $Q^*(s, a)$ and optimal policy $\pi^*(s)$.
2. Calculate the potential-shaped rewards $R'(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$.
3. Solve directly for the shaped optimal values $V'^*(s)$ and $Q'^*(s, a)$.
4. Numerically verify that $Q'^*(s, a) = Q^*(s, a) - \Phi(s)$ and confirm that the optimal policy $\pi'^*(s) = \arg\max_a Q'^*(s, a)$ is identical to $\pi^*(s)$.

**Solution:**

**Step 1: Unshaped Bellman Optimality Solution**
Under optimal policy hypothesis $\pi^*(S_1) = a_1$ and $\pi^*(S_2) = a_1$:
$$V^*(S_1) = 5.0 + 0.80 V^*(S_2)$$
$$V^*(S_2) = -1.0 + 0.80 V^*(S_1)$$
Substitute $V^*(S_2)$ into $V^*(S_1)$:
$$V^*(S_1) = 5.0 + 0.80 \left( -1.0 + 0.80 V^*(S_1) \right) = 5.0 - 0.80 + 0.64 V^*(S_1) = 4.20 + 0.64 V^*(S_1)$$
$$(1 - 0.64) V^*(S_1) = 4.20 \implies 0.36 V^*(S_1) = 4.20 \implies V^*(S_1) = \frac{4.20}{0.36} = \mathbf{\frac{35}{3} \approx 11.6667}$$
$$V^*(S_2) = -1.0 + 0.80 \left( \frac{35}{3} \right) = -1.0 + \frac{28}{3} = \mathbf{\frac{25}{3} \approx 8.3333}$$

Verify optimal actions via Q-values:
- State $S_1$:
  - $Q^*(S_1, a_1) = 5.0 + 0.80 V^*(S_2) = 5.0 + \frac{20}{3} = \mathbf{\frac{35}{3} \approx 11.6667}$
  - $Q^*(S_1, a_2) = 2.0 + 0.80 V^*(S_1) = 2.0 + 0.80 \left( \frac{35}{3} \right) = 2.0 + \frac{28}{3} = \mathbf{\frac{34}{3} \approx 11.3333}$
  - $\arg\max_{a} Q^*(S_1, a) = a_1$ (since $\frac{35}{3} > \frac{34}{3}$).
- State $S_2$:
  - $Q^*(S_2, a_1) = -1.0 + 0.80 V^*(S_1) = -1.0 + \frac{28}{3} = \mathbf{\frac{25}{3} \approx 8.3333}$
  - $Q^*(S_2, a_2) = 0.0 + 0.80 V^*(S_2) = 0.80 \left( \frac{25}{3} \right) = \mathbf{\frac{20}{3} \approx 6.6667}$
  - $\arg\max_{a} Q^*(S_2, a) = a_1$ (since $\frac{25}{3} > \frac{20}{3}$).

Thus, $\pi^*(S_1) = a_1$ and $\pi^*(S_2) = a_1$.

**Step 2: Compute Shaped Rewards $R'(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$**
- Transition $(S_1, a_1 \to S_2)$:
  $$R'(S_1, a_1, S_2) = 5.0 + 0.80(2.0) - 10.0 = 5.0 + 1.60 - 10.0 = \mathbf{-3.4000}$$
- Transition $(S_1, a_2 \to S_1)$:
  $$R'(S_1, a_2, S_1) = 2.0 + 0.80(10.0) - 10.0 = 2.0 + 8.0 - 10.0 = \mathbf{0.0000}$$
- Transition $(S_2, a_1 \to S_1)$:
  $$R'(S_2, a_1, S_1) = -1.0 + 0.80(10.0) - 2.0 = -1.0 + 8.0 - 2.0 = \mathbf{+5.0000}$$
- Transition $(S_2, a_2 \to S_2)$:
  $$R'(S_2, a_2, S_2) = 0.0 + 0.80(2.0) - 2.0 = 0.0 + 1.60 - 2.0 = \mathbf{-0.4000}$$

**Step 3: Direct Solution of the Shaped MDP**
Under the identical candidate policy $\pi(S_1) = a_1, \pi(S_2) = a_1$:
$$V'^*(S_1) = -3.40 + 0.80 V'^*(S_2)$$
$$V'^*(S_2) = +5.00 + 0.80 V'^*(S_1)$$
Substituting:
$$V'^*(S_1) = -3.40 + 0.80 \left( 5.00 + 0.80 V'^*(S_1) \right) = -3.40 + 4.00 + 0.64 V'^*(S_1) = 0.60 + 0.64 V'^*(S_1)$$
$$0.36 V'^*(S_1) = 0.60 \implies V'^*(S_1) = \frac{0.60}{0.36} = \mathbf{\frac{5}{3} \approx 1.6667}$$
$$V'^*(S_2) = 5.00 + 0.80 \left( \frac{5}{3} \right) = 5.00 + \frac{4}{3} = \mathbf{\frac{19}{3} \approx 6.3333}$$

Compute shaped Q-values:
- State $S_1$:
  - $Q'^*(S_1, a_1) = -3.40 + 0.80 V'^*(S_2) = -3.40 + 0.80 \left( \frac{19}{3} \right) = -\frac{17}{5} + \frac{76}{15} = \frac{-51 + 76}{15} = \mathbf{\frac{25}{15} = \frac{5}{3} \approx 1.6667}$
  - $Q'^*(S_1, a_2) = 0.00 + 0.80 V'^*(S_1) = 0.80 \left( \frac{5}{3} \right) = \mathbf{\frac{4}{3} \approx 1.3333}$
  - $\arg\max_a Q'^*(S_1, a) = a_1$ (since $\frac{5}{3} > \frac{4}{3}$).
- State $S_2$:
  - $Q'^*(S_2, a_1) = +5.00 + 0.80 V'^*(S_1) = 5.00 + 0.80 \left( \frac{5}{3} \right) = 5.00 + \frac{4}{3} = \mathbf{\frac{19}{3} \approx 6.3333}$
  - $Q'^*(S_2, a_2) = -0.40 + 0.80 V'^*(S_2) = -0.40 + 0.80 \left( \frac{19}{3} \right) = -\frac{2}{5} + \frac{76}{15} = \frac{-6 + 76}{15} = \mathbf{\frac{70}{15} = \frac{14}{3} \approx 4.6667}$
  - $\arg\max_a Q'^*(S_2, a) = a_1$ (since $\frac{19}{3} > \frac{14}{3}$).

**Step 4: Comparison and Verification of the Theoretical Theorem**
Check the relationship $Q'^*(s, a) = Q^*(s, a) - \Phi(s)$:
- $Q'^*(S_1, a_1) = \frac{35}{3} - 10 = \frac{5}{3}$ (Exact match!)
- $Q'^*(S_1, a_2) = \frac{34}{3} - 10 = \frac{4}{3}$ (Exact match!)
- $Q'^*(S_2, a_1) = \frac{25}{3} - 2 = \frac{19}{3}$ (Exact match!)
- $Q'^*(S_2, a_2) = \frac{20}{3} - 2 = \frac{14}{3}$ (Exact match!)
- Value difference: $V'^*(S_1) = \frac{35}{3} - 10 = \frac{5}{3}$ and $V'^*(S_2) = \frac{25}{3} - 2 = \frac{19}{3}$.

The optimal policy $\pi'^*(S_1) = a_1, \pi'^*(S_2) = a_1$ is **strictly identical** to $\pi^*(s)$. Potential-based shaping alters intermediate rewards to guide learning speed without biasing the optimal solution!

---

## 7. Deep Learning Connection & Application

### 1. Recurrent Policies for POMDPs (DRQN: Hausknecht & Stone, 2015)
When frame stacking is insufficient (e.g., long-term memory requirements in 3D Minecraft or Doom), deep RL replaces feedforward networks with **Deep Recurrent Q-Networks (DRQN)** or Transformer memory backbones:
$$\mathbf{h}_t = \text{LSTM}(\mathbf{h}_{t-1}, \, \text{CNN}(O_t))$$
The hidden state $\mathbf{h}_t$ serves as a learned, continuous approximation of the Bayesian belief state $b_t$!

### 2. Reward Shaping & Potential-Based Invariance (Ng, Harada, & Russell, 1999)
In complex sparse-reward MDPs, researchers often add auxiliary reward signals:
$$R'(s, a, s') = R(s, a, s') + F(s, a, s')$$
**Theorem (Policy Invariance under Potential Shaping):**
The optimal policy $\pi^*$ is invariant if and only if the shaping function is the difference of a potential function $\Phi(s)$:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
This guarantees the agent solves the original task without reward hacking!

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Exact numerical verification of Part 5 hand calculations:
   - Induced transition matrix $\mathbf{P}^\pi$ and reward vector $\mathbf{r}^\pi$.
   - 4-step trajectory return $G_0 = 10.9750$ via backward dynamic programming.
   - POMDP Bayesian belief update $b'(S_1) \approx 0.7177$.
2. Analytical stationary distribution solver using null space and eigenvector decomposition.
3. Full Discrete MDP Environment class supporting trajectory rollouts, transition verification, and memorylessness validation.
4. POMDP Belief State Tracker simulating noisy sensor observations and verifying Bayesian filter convergence.

See implementation in:
[`11_reinforcement_learning/code/02_markov_decision_processes.py`](./code/02_markov_decision_processes.py)
