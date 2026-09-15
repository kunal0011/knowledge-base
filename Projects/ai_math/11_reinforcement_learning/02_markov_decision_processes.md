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
