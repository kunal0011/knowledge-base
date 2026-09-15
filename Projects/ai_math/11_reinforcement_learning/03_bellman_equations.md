# The Bellman Equations: State-Value & Action-Value Functions

---

## 1. Intuition & 101 Motivation

### Richard Bellman's Principle of Optimality (1957)
In sequential decision-making, computing the value of a state by explicitly unrolling every possible future trajectory to infinity is mathematically and computationally impossible. A branching tree of depth $T$ with $|\mathcal{A}|$ actions explodes exponentially as $\mathcal{O}(|\mathcal{A}|^T)$.

In 1957, mathematician Richard Bellman formulated the **Principle of Optimality**:
> *"An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."*

This insight converts a global, infinite-horizon optimization problem into a **local, recursive equation**:
$$\text{Value of Current State} = \text{Immediate Reward} + \text{Discounted Value of Next State}$$

```
                       THE RECURSIVE BELLMAN BACKUP
                             State S_t (Value V(S_t))
                                      ●
                                     / \  Action A_t ~ pi(a | s)
                                    /   \
                                   ○     ○
                                  / \   / \  Transition P(s' | s, a)
                                 ●   ● ●   ● Next State S_{t+1} (Value V(S_{t+1}))
                                 
                     V(s) = E [ R_{t+1} + gamma * V(S_{t+1}) ]
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Formal Definitions of Value Functions

Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ be a Markov Decision Process, and let $\pi(a \mid s)$ be a policy.

1. **State-Value Function $V^\pi(s)$:**
   The expected discounted return starting from state $s$ and following policy $\pi$ thereafter:
   $$V^\pi(s) \equiv \mathbb{E}_\pi \left[ G_t \;\middle|\; S_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right]$$

2. **Action-Value Function $Q^\pi(s, a)$:**
   The expected discounted return starting from state $s$, taking arbitrary action $a$, and following policy $\pi$ thereafter:
   $$Q^\pi(s, a) \equiv \mathbb{E}_\pi \left[ G_t \;\middle|\; S_t = s, A_t = a \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \;\middle|\; S_t = s, A_t = a \right]$$

#### Fundamental Connection Between $V^\pi$ and $Q^\pi$
By the law of total expectation, marginalizing over all possible actions under policy $\pi$:
$$\mathbf{V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)}$$

---

### 2.2 First-Principles Derivation of the Bellman Expectation Equations

#### 1. Bellman Expectation Equation for $V^\pi(s)$
Decompose the return $G_t = R_{t+1} + \gamma G_{t+1}$:
$$\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \;\middle|\; S_t = s \right] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \;\middle|\; S_t = s, A_t = a \right] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( \mathbb{E}\left[ R_{t+1} \;\middle|\; S_t = s, A_t = a \right] + \gamma \mathbb{E}_\pi \left[ G_{t+1} \;\middle|\; S_t = s, A_t = a \right] \right) \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \mathbb{E}_\pi \left[ G_{t+1} \;\middle|\; S_{t+1} = s' \right] \right)
\end{aligned}$$

Recognizing $V^\pi(s') = \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$, we obtain:
$$\mathbf{V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]}$$

#### 2. Bellman Expectation Equation for $Q^\pi(s, a)$
Starting from the definition of action-value:
$$\begin{aligned}
Q^\pi(s, a) &= \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \;\middle|\; S_t = s, A_t = a \right] \\
&= \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \mathbb{E}_\pi \left[ G_{t+1} \;\middle|\; S_{t+1} = s' \right] \\
&= \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s')
\end{aligned}$$
Substitute $V^\pi(s') = \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')$:
$$\mathbf{Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')}$$

---

### 2.3 Linear Algebraic Matrix Solution of Bellman Expectation

For discrete MDPs with $|\mathcal{S}| = n$ states, the Bellman expectation equation is a **linear system of $n$ equations in $n$ unknowns**:
$$\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}^\pi$$
where:
$$\mathbf{v}^\pi = \begin{bmatrix} V^\pi(s_1) \\ \vdots \\ V^\pi(s_n) \end{bmatrix}, \quad \mathbf{r}^\pi = \begin{bmatrix} \mathcal{R}^\pi(s_1) \\ \vdots \\ \mathcal{R}^\pi(s_n) \end{bmatrix}, \quad \mathbf{P}^\pi \in \mathbb{R}^{n \times n}$$

Rearranging:
$$\left( \mathbf{I} - \gamma \mathbf{P}^\pi \right) \mathbf{v}^\pi = \mathbf{r}^\pi$$

#### Proof of Invertibility (The Neumann Series Theorem)
**Theorem:**
For any discount factor $\gamma \in [0, 1)$ and any row-stochastic matrix $\mathbf{P}^\pi$, the matrix $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ is non-singular and strictly invertible.

**Proof:**
Since $\mathbf{P}^\pi$ is row-stochastic ($\sum_j P_{ij}^\pi = 1$), its maximum absolute row sum is $\|\mathbf{P}^\pi\|_\infty = 1$.
By the spectral radius theorem, the spectral radius $\rho(\mathbf{P}^\pi) \le \|\mathbf{P}^\pi\|_\infty = 1$.
Therefore:
$$\rho\left( \gamma \mathbf{P}^\pi \right) = \gamma \rho(\mathbf{P}^\pi) \le \gamma < 1$$
Since the spectral radius of $\gamma \mathbf{P}^\pi$ is strictly less than 1, all eigenvalues $\lambda_i$ of $\gamma \mathbf{P}^\pi$ satisfy $|\lambda_i| < 1$.
Consequently, $(1 - \lambda_i) \ne 0$, meaning no eigenvalue of $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ is zero. The matrix is invertible, and its inverse is given by the absolutely convergent **Neumann Series**:
$$\left( \mathbf{I} - \gamma \mathbf{P}^\pi \right)^{-1} = \sum_{k=0}^\infty \left( \gamma \mathbf{P}^\pi \right)^k \quad \blacksquare$$

#### Exact Analytical Value Solution
$$\mathbf{\mathbf{v}^\pi = \left( \mathbf{I} - \gamma \mathbf{P}^\pi \right)^{-1} \mathbf{r}^\pi}$$
Computational complexity: Matrix inversion scales as $\mathcal{O}(|\mathcal{S}|^3)$. While practical for small state spaces ($|\mathcal{S}| \le 2000$), large or continuous state spaces require iterative methods.

---

### 2.4 The Bellman Optimality Equations

Define partial ordering over policies:
$$\pi \ge \pi' \iff V^\pi(s) \ge V^{\pi'}(s), \quad \forall s \in \mathcal{S}$$
There exists a unique **Optimal State-Value Function** $V^*(s)$ and **Optimal Action-Value Function** $Q^*(s, a)$:
$$V^*(s) \equiv \max_\pi V^\pi(s), \quad Q^*(s, a) \equiv \max_\pi Q^\pi(s, a)$$

Under an optimal policy, the value of a state must equal the expected return for the **best possible action**:
$$V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)$$

Substituting into the recursive expansion:
$$\mathbf{V^*(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^*(s') \right]}$$

Similarly for $Q^*(s, a)$:
$$\mathbf{Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')}$$

#### Why Bellman Optimality CANNOT Be Solved via Matrix Inversion
Notice the critical mathematical difference:
- Bellman **Expectation** equations are **linear** (weighted average $\sum_a \pi(a \mid s)$).
- Bellman **Optimality** equations are **non-linear** due to the $\max_{a \in \mathcal{A}}$ operator!
There is no linear system $(\mathbf{I} - \gamma \mathbf{P})^{-1}$ for $V^*$. Instead, $V^*$ must be solved using **fixed-point contraction iteration** (Value Iteration) or linear programming.

#### Extracting the Optimal Policy $\pi^*$
Once $Q^*(s, a)$ is known, finding the optimal policy requires **zero planning**: simply act greedily with respect to $Q^*(s, a)$:
$$\pi^*(a \mid s) = \begin{cases} 1 & \text{if } a = \operatorname{argmax}_{a' \in \mathcal{A}} Q^*(s, a') \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Geometric & Algebraic Interpretation

### Hyperplanes & The Upper Convex Envelope in Value Space
Consider the space of all value vectors $\mathbf{v} \in \mathbb{R}^{|\mathcal{S}|}$:
1. Each deterministic stationary policy $\pi$ induces a fixed transition matrix $\mathbf{P}^\pi$ and reward $\mathbf{r}^\pi$, defining an affine hyperplane:
   $$\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$$
2. The optimal value function $V^*$ is the **pointwise supremum** (upper envelope) over all valid policies:
   $$V^*(s) = \max_{\pi} V^\pi(s)$$
3. In geometric terms, the set of all achievable value vectors forms a bounded polytope, and $V^*$ sits precisely at its supreme vertex!

```
                    GEOMETRY OF THE VALUE SPACE
          V(s_2)
            ▲                Upper Envelope V^*
            │                    / \
            │           Policy B/   \Policy A
            │           -------/     \-------
            │                 /       \
            │                /         \
            │               /           \
            │              /             \
            │             /               \
            └────────────/─────────────────\────────► V(s_1)
```

---

## 4. Real-World Analogy

### The Compound Interest Bank Account vs. The Stock Portfolio
- **Bellman Expectation (Fixed Deposit / Fixed Policy):**
  You lock your money into a conservative savings plan that pays $5\%$ interest annually and reinvests automatically. The value of your wealth today is your cash income this year plus $0.95 \times$ your projected wealth next year. Because the plan is fixed, calculating your future wealth is a simple linear algebraic equation.
- **Bellman Optimality (Active Wall Street Hedge Fund):**
  At every single market open, you can choose whether to buy Real Estate, Gold, Tech Stocks, or Crypto. The value of your portfolio today is the maximum return among all available investments plus the discounted future wealth assuming you make the **optimal financial decision at every future morning**. Because of the "choose the best" operation, you cannot solve your wealth with a single linear formula; you must work backward from the future using dynamic programming!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us compute exact Bellman expectation matrix inversion, value vector recovery, action-value verification, and a Bellman optimality step by hand with concrete numbers.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Bellman Equations |
| :--- | :--- | :--- | :--- |
| $\mathcal{S}$ | State Set | 2 States | States $\{S_1, S_2\}$ representing agent locations |
| $\mathcal{A}$ | Action Set | 2 Actions | Choices $\{a_1, a_2\}$ available at each state |
| $\gamma$ | Discount Factor | Scalar ($0.50$) | Temporal discount parameter |
| $\mathbf{P}^\pi$ | Induced Transition Matrix | $(2, 2)$ | State-to-state transitions under policy $\pi$ |
| $\mathbf{r}^\pi$ | Induced Reward Vector | $(2,)$ | Immediate expected rewards under policy $\pi$ |
| $\mathbf{I} - \gamma \mathbf{P}^\pi$ | Bellman Characteristic Matrix | $(2, 2)$ | Linear operator defining the expectation equation |
| $(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$ | Resolvent Matrix | $(2, 2)$ | Exact Neumann inverse solving the linear system |
| $\mathbf{v}^\pi = [v_1, v_2]^\top$ | State-Value Vector | $(2,)$ | Exact true expected returns under policy $\pi$ |
| $Q^\pi(s, a)$ | Action-Value Scores | 4 Scalars | Expected return for taking action $a$ in state $s$ |
| $\mathcal{T}^* V$ | Bellman Optimality Step | $(2,)$ | $\max_a [\mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s')]$ |

---

### 5.2 Concrete Toy Setup

Let $|\mathcal{S}| = 2$ ($S_1, S_2$) and $|\mathcal{A}| = 2$ ($a_1, a_2$).
Discount factor: $\gamma = 0.50$.

#### Transition Dynamics $\mathcal{P}(s' \mid s, a)$:
- From $S_1$:
  - Under $a_1$: $\mathcal{P}(S_1 \mid S_1, a_1) = 0.8, \quad \mathcal{P}(S_2 \mid S_1, a_1) = 0.2, \quad \mathcal{R}(S_1, a_1) = 1.0$
  - Under $a_2$: $\mathcal{P}(S_1 \mid S_1, a_2) = 0.2, \quad \mathcal{P}(S_2 \mid S_1, a_2) = 0.8, \quad \mathcal{R}(S_1, a_2) = 2.0$
- From $S_2$:
  - Under $a_1$: $\mathcal{P}(S_1 \mid S_2, a_1) = 0.4, \quad \mathcal{P}(S_2 \mid S_2, a_1) = 0.6, \quad \mathcal{R}(S_2, a_1) = -1.0$
  - Under $a_2$: $\mathcal{P}(S_1 \mid S_2, a_2) = 0.0, \quad \mathcal{P}(S_2 \mid S_2, a_2) = 1.0, \quad \mathcal{R}(S_2, a_2) = 0.5$

#### Fixed Policy $\pi$:
- In $S_1$: $\pi(a_1 \mid S_1) = 0.50, \quad \pi(a_2 \mid S_1) = 0.50$ (50-50 mix)
- In $S_2$: $\pi(a_1 \mid S_2) = 0.00, \quad \pi(a_2 \mid S_2) = 1.00$ (Always choose $a_2$)

---

### 5.3 Step 1: Compute Induced Matrix $\mathbf{P}^\pi$ and Vector $\mathbf{r}^\pi$

#### Row 1 ($S_1$):
$$\mathbf{P}^\pi(S_1) = 0.5 \begin{bmatrix} 0.8 & 0.2 \end{bmatrix} + 0.5 \begin{bmatrix} 0.2 & 0.8 \end{bmatrix} = \begin{bmatrix} 0.4 & 0.1 \end{bmatrix} + \begin{bmatrix} 0.1 & 0.4 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.50 & 0.50 \end{bmatrix}}$$
$$r^\pi(S_1) = 0.5(1.0) + 0.5(2.0) = 0.5 + 1.0 = \mathbf{1.5000}$$

#### Row 2 ($S_2$):
$$\mathbf{P}^\pi(S_2) = 0.0 \begin{bmatrix} 0.4 & 0.6 \end{bmatrix} + 1.0 \begin{bmatrix} 0.0 & 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.00 & 1.00 \end{bmatrix}}$$
$$r^\pi(S_2) = 0.0(-1.0) + 1.0(0.5) = \mathbf{0.5000}$$

$$\mathbf{P}^\pi = \begin{bmatrix} 0.50 & 0.50 \\ 0.00 & 1.00 \end{bmatrix}, \quad \mathbf{r}^\pi = \begin{bmatrix} 1.50 \\ 0.50 \end{bmatrix}$$

---

### 5.4 Step 2: Exact Matrix Inversion of $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ by Hand

With $\gamma = 0.50$:
$$\gamma \mathbf{P}^\pi = 0.5 \begin{bmatrix} 0.50 & 0.50 \\ 0.00 & 1.00 \end{bmatrix} = \begin{bmatrix} 0.25 & 0.25 \\ 0.00 & 0.50 \end{bmatrix}$$

Characteristic matrix:
$$\mathbf{A} = \mathbf{I} - \gamma \mathbf{P}^\pi = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} - \begin{bmatrix} 0.25 & 0.25 \\ 0.00 & 0.50 \end{bmatrix} = \begin{bmatrix} 0.75 & -0.25 \\ 0.00 & 0.50 \end{bmatrix}$$

#### Analytical $2 \times 2$ Inverse Formula:
For $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:
$$\det(\mathbf{A}) = ad - bc = (0.75)(0.50) - (-0.25)(0.00) = 0.3750 = \mathbf{\frac{3}{8}}$$

$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} = \frac{1}{3/8} \begin{bmatrix} 0.50 & 0.25 \\ 0.00 & 0.75 \end{bmatrix} = \frac{8}{3} \begin{bmatrix} 1/2 & 1/4 \\ 0 & 3/4 \end{bmatrix}$$
$$\mathbf{A}^{-1} = \begin{bmatrix} \frac{8}{3} \times \frac{1}{2} & \frac{8}{3} \times \frac{1}{4} \\ 0 & \frac{8}{3} \times \frac{3}{4} \end{bmatrix} = \mathbf{\begin{bmatrix} 4/3 & 2/3 \\ 0 & 2 \end{bmatrix}} \approx \begin{bmatrix} 1.3333 & 0.6667 \\ 0.0000 & 2.0000 \end{bmatrix}$$

---

### 5.5 Step 3: Exact Value Vector $\mathbf{v}^\pi$ Recovery

$$\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi = \begin{bmatrix} 4/3 & 2/3 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1.5 \\ 0.5 \end{bmatrix} = \begin{bmatrix} 4/3 & 2/3 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 3/2 \\ 1/2 \end{bmatrix}$$
$$v^\pi(S_1) = \left(\frac{4}{3}\right)\left(\frac{3}{2}\right) + \left(\frac{2}{3}\right)\left(\frac{1}{2}\right) = 2.0 + \frac{1}{3} = \frac{7}{3} \approx \mathbf{2.3333}$$
$$v^\pi(S_2) = (0)\left(\frac{3}{2}\right) + (2)\left(\frac{1}{2}\right) = 0.0 + 1.0 = \mathbf{1.0000}$$

$$\mathbf{v}^\pi = \begin{bmatrix} 7/3 \\ 1 \end{bmatrix} \approx \begin{bmatrix} 2.3333 \\ 1.0000 \end{bmatrix}$$

---

### 5.6 Step 4: Verification of Action-Values $Q^\pi(s, a)$

Evaluate $Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^\pi(s')$:

1. **State $S_1$, Action $a_1$:**
   $$Q^\pi(S_1, a_1) = 1.0 + 0.5 \left[ 0.8 V^\pi(S_1) + 0.2 V^\pi(S_2) \right]$$
   $$= 1.0 + 0.5 \left[ 0.8\left(\frac{7}{3}\right) + 0.2(1.0) \right] = 1.0 + 0.5 \left[ \frac{5.6}{3} + \frac{0.6}{3} \right] = 1.0 + 0.5 \left[ \frac{6.2}{3} \right] = 1.0 + \frac{3.1}{3} = \mathbf{\frac{6.1}{3} \approx 2.0333}$$

2. **State $S_1$, Action $a_2$:**
   $$Q^\pi(S_1, a_2) = 2.0 + 0.5 \left[ 0.2 V^\pi(S_1) + 0.8 V^\pi(S_2) \right]$$
   $$= 2.0 + 0.5 \left[ 0.2\left(\frac{7}{3}\right) + 0.8(1.0) \right] = 2.0 + 0.5 \left[ \frac{1.4}{3} + \frac{2.4}{3} \right] = 2.0 + 0.5 \left[ \frac{3.8}{3} \right] = 2.0 + \frac{1.9}{3} = \mathbf{\frac{7.9}{3} \approx 2.6333}$$

3. **Check Consistency with $V^\pi(S_1)$:**
   $$V^\pi(S_1) = \pi(a_1 \mid S_1) Q^\pi(S_1, a_1) + \pi(a_2 \mid S_1) Q^\pi(S_1, a_2)$$
   $$= 0.5(2.0333) + 0.5(2.6333) = \frac{2.0333 + 2.6333}{2} = \frac{4.6667}{2} = \mathbf{2.3333} \equiv \frac{7}{3}$$
   Perfect mathematical consistency!

4. **State $S_2$, Action $a_1$:**
   $$Q^\pi(S_2, a_1) = -1.0 + 0.5 \left[ 0.4 V^\pi(S_1) + 0.6 V^\pi(S_2) \right] = -1.0 + 0.5 \left[ 0.4\left(\frac{7}{3}\right) + 0.6(1.0) \right] = -1.0 + 0.5 \left[ \frac{2.8 + 1.8}{3} \right] = -1.0 + \frac{2.3}{3} = \mathbf{-\frac{0.7}{3} \approx -0.2333}$$

5. **State $S_2$, Action $a_2$:**
   $$Q^\pi(S_2, a_2) = 0.5 + 0.5 \left[ 0.0 V^\pi(S_1) + 1.0 V^\pi(S_2) \right] = 0.5 + 0.5(1.0) = \mathbf{1.0000}$$

---

### 5.7 Step 5: The Bellman Optimality Improvement Operator Step

Evaluate $\mathcal{T}^* V^\pi(s) = \max_a Q^\pi(s, a)$:
- In $S_1$: $\max\left( Q^\pi(S_1, a_1), Q^\pi(S_1, a_2) \right) = \max(2.0333, \mathbf{2.6333}) = \mathbf{2.6333}$ (Action $a_2$ is strictly better than current value $2.3333$).
- In $S_2$: $\max\left( Q^\pi(S_2, a_1), Q^\pi(S_2, a_2) \right) = \max(-0.2333, \mathbf{1.0000}) = \mathbf{1.0000}$ (Action $a_2$ is optimal).

A single policy improvement step yields the strictly superior policy:
$$\pi'(S_1) = a_2, \quad \pi'(S_2) = a_2$$
Every single cell and calculation is exact to unlimited precision!

---

## 6. Solved Illustrations

### Illustration 1: Neumann Series Power Expansion
**Problem:** For the $2 \times 2$ matrix from Part 5 ($\gamma = 0.5, \mathbf{P}^\pi = \begin{bmatrix} 0.5 & 0.5 \\ 0 & 1 \end{bmatrix}$), approximate the inverse $(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$ using the first 3 terms of the Neumann series $\sum_{k=0}^2 (\gamma \mathbf{P}^\pi)^k$ and measure error to the analytical inverse $\begin{bmatrix} 4/3 & 2/3 \\ 0 & 2 \end{bmatrix}$.
**Solution:**
$$\mathbf{M} = \gamma \mathbf{P}^\pi = \begin{bmatrix} 0.25 & 0.25 \\ 0 & 0.5 \end{bmatrix}$$
1. $k = 0$: $\mathbf{M}^0 = \mathbf{I} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$
2. $k = 1$: $\mathbf{M}^1 = \begin{bmatrix} 0.25 & 0.25 \\ 0.00 & 0.50 \end{bmatrix}$
3. $k = 2$: $\mathbf{M}^2 = \begin{bmatrix} 0.25 & 0.25 \\ 0 & 0.5 \end{bmatrix} \begin{bmatrix} 0.25 & 0.25 \\ 0 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.0625 & 0.0625 + 0.1250 \\ 0 & 0.25 \end{bmatrix} = \begin{bmatrix} 0.0625 & 0.1875 \\ 0.0000 & 0.2500 \end{bmatrix}$

Sum:
$$\mathbf{S}_2 = \mathbf{I} + \mathbf{M} + \mathbf{M}^2 = \begin{bmatrix} 1.3125 & 0.4375 \\ 0.0000 & 1.7500 \end{bmatrix}$$
Analytical exact:
$$\mathbf{A}^{-1} = \begin{bmatrix} 1.3333 & 0.6667 \\ 0.0000 & 2.0000 \end{bmatrix}$$
Approximation captures over $88\%$ of the infinite sum in just 3 terms! Because $\gamma = 0.5$, the remainder decays exponentially as $\mathcal{O}(\gamma^{k+1}) = \mathcal{O}(0.5^3) = \mathcal{O}(0.125)$.

---

### Illustration 2: Bellman Optimality on Deterministic Gridworld
**Problem:** In a $1 \times 3$ gridworld ($S_1 \leftrightarrow S_2 \leftrightarrow S_3$ [Goal]).
Actions: Left, Right. Rewards are $0$ everywhere except entering $S_3$, which gives $+10$ and terminates. $\gamma = 0.90$.
Write down the analytical optimal value $V^*(S_1)$ and $V^*(S_2)$.
**Solution:**
From $S_2$, taking action Right reaches $S_3$:
$$V^*(S_2) = 10 + 0.90(0) = \mathbf{10.0000}$$
From $S_1$, taking action Right reaches $S_2$:
$$V^*(S_1) = 0 + 0.90 V^*(S_2) = 0.90(10.0000) = \mathbf{9.0000}$$
Optimal values propagate backward through space, discounted geometrically by distance from the goal!

---

## 7. Deep Learning Connection & Application

### 1. The Bellman Error Loss in Deep Q-Networks (DQN)
In deep reinforcement learning, we cannot store a table of $Q(s, a)$. We approximate it with a deep neural network $Q(s, a; \theta)$.
The training loss is the squared **Mean Squared Bellman Error (MSBE)**:
$$\mathcal{L}_{\text{Bellman}}(\theta) = \mathbb{E}_{(s, a, r, s')} \left[ \left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta^-)}_{\text{Bellman Optimality Target}} - \underbrace{Q(s, a; \theta)}_{\text{Current Prediction}} \right)^2 \right]$$

### 2. Actor-Critic Advantage Formulation
In policy optimization (A2C, PPO), the Actor uses the Advantage function derived from the Bellman expectation relation:
$$A(s, a) = Q(s, a) - V(s) \equiv r + \gamma V(s') - V(s)$$
The quantity $\delta = r + \gamma V(s') - V(s)$ is the local **Bellman residual (TD error)**!

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Exact numerical verification of Part 5 hand calculations:
   - Induced $\mathbf{P}^\pi$ and $\mathbf{r}^\pi$ evaluation.
   - Analytical matrix inversion $(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$ matching $[4/3, 2/3; 0, 2]$.
   - Value vector $\mathbf{v}^\pi = [7/3, 1]^\top$.
   - All 4 action-values $Q^\pi(s, a)$ and weighted consistency $V^\pi(s) = \sum_a \pi(a \mid s) Q(s, a)$ to $< 10^{-14}$.
2. Verification of the Neumann series power convergence: $\|\sum_{k=0}^N (\gamma \mathbf{P})^k - (\mathbf{I} - \gamma \mathbf{P})^{-1}\|_\infty \le \frac{\gamma^{N+1}}{1 - \gamma}$.
3. General iterative Bellman Expectation solver for arbitrary discrete MDPs.
4. General Bellman Optimality operator ($\mathcal{T}^*$) fixed point iteration.

See implementation in:
[`11_reinforcement_learning/code/03_bellman_equations.py`](./code/03_bellman_equations.py)
