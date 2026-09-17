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

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.3.1: Monotonicity and Contraction of the Bellman Expectation Operator $\mathcal{T}^\pi$ in the $L_\infty$ Norm

##### Problem Statement & Goal
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ be an MDP with $|\mathcal{S}| = n$, and let $\pi$ be an arbitrary fixed policy. Define the Bellman Expectation Operator $\mathcal{T}^\pi: \mathbb{R}^n \to \mathbb{R}^n$ acting on value functions $V \in \mathbb{R}^n$ by:
$$(\mathcal{T}^\pi V)(s) \equiv \mathcal{R}^\pi(s) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) V(s'), \quad \forall s \in \mathcal{S}$$
or in matrix-vector notation: $\mathcal{T}^\pi \mathbf{v} = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}$.
We prove from first principles that:
1. **Monotonicity:** If $\mathbf{u} \le \mathbf{v}$ componentwise ($u(s) \le v(s)$ for all $s$), then $\mathcal{T}^\pi \mathbf{u} \le \mathcal{T}^\pi \mathbf{v}$ componentwise.
2. **$\gamma$-Contraction in $L_\infty$ Norm:** For any two value functions $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
   $$\|\mathcal{T}^\pi \mathbf{u} - \mathcal{T}^\pi \mathbf{v}\|_\infty \le \gamma \|\mathbf{u} - \mathbf{v}\|_\infty$$
3. **Banach Fixed-Point Convergence:** There exists a unique fixed point $\mathbf{v}^\pi$ satisfying $\mathcal{T}^\pi \mathbf{v}^\pi = \mathbf{v}^\pi$, and the iterative sequence $\mathbf{v}_{k+1} = \mathcal{T}^\pi \mathbf{v}_k$ converges geometrically at linear rate $\gamma$:
   $$\|\mathbf{v}_k - \mathbf{v}^\pi\|_\infty \le \frac{\gamma^k}{1 - \gamma} \|\mathbf{v}_1 - \mathbf{v}_0\|_\infty$$

##### Explicit Assumptions
1. Finite state space $|\mathcal{S}| = n < \infty$.
2. $\mathbf{P}^\pi$ is row-stochastic: $P^\pi(s' \mid s) \ge 0$ and $\sum_{s'} P^\pi(s' \mid s) = 1$ for all $s \in \mathcal{S}$.
3. Discount factor satisfies $0 \le \gamma < 1$.
4. Value space $\mathbb{R}^n$ is equipped with the supremum norm $\|\mathbf{v}\|_\infty = \max_{s \in \mathcal{S}} |v(s)|$, forming a complete Banach space $(\mathbb{R}^n, \|\cdot\|_\infty)$.

##### Underlying Intuition
The Bellman expectation operator is an affine transformation consisting of adding an immediate reward vector $\mathbf{r}^\pi$ and taking a discounted convex combination of next-state values via row-stochastic matrix $\mathbf{P}^\pi$. Because convex combinations can never expand differences between vectors, and the scalar $\gamma < 1$ shrinks all differences strictly by at least a factor of $\gamma$, the operator acts as an accordion squeezing any two value functions closer together at every iteration.

##### End-to-End Mathematical Derivation

**Step 1: Proof of Monotonicity**
Let $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ such that $u(s) \le v(s)$ for all $s \in \mathcal{S}$.
Consider the difference for any state $s$:
$$(\mathcal{T}^\pi \mathbf{v})(s) - (\mathcal{T}^\pi \mathbf{u})(s) = \left[ \mathcal{R}^\pi(s) + \gamma \sum_{s'} \mathcal{P}^\pi(s' \mid s) v(s') \right] - \left[ \mathcal{R}^\pi(s) + \gamma \sum_{s'} \mathcal{P}^\pi(s' \mid s) u(s') \right]$$
The immediate reward terms $\mathcal{R}^\pi(s)$ cancel out identically:
$$(\mathcal{T}^\pi \mathbf{v})(s) - (\mathcal{T}^\pi \mathbf{u})(s) = \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) \left[ v(s') - u(s') \right]$$
Because $v(s') - u(s') \ge 0$ by hypothesis, $\mathcal{P}^\pi(s' \mid s) \ge 0$ by definition of probabilities, and $\gamma \ge 0$:
$$\gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) \left[ v(s') - u(s') \right] \ge 0$$
Therefore, $(\mathcal{T}^\pi \mathbf{u})(s) \le (\mathcal{T}^\pi \mathbf{v})(s)$ for all $s \in \mathcal{S}$.

**Step 2: Proof of $\gamma$-Contraction Mapping**
For any $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, examine the absolute difference at any state $s \in \mathcal{S}$:
$$|(\mathcal{T}^\pi \mathbf{u})(s) - (\mathcal{T}^\pi \mathbf{v})(s)| = \left| \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) \left( u(s') - v(s') \right) \right|$$
Applying the triangle inequality:
$$\left| \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) \left( u(s') - v(s') \right) \right| \le \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) \left| u(s') - v(s') \right|$$
Because $|u(s') - v(s')| \le \max_{s'' \in \mathcal{S}} |u(s'') - v(s'')| = \|\mathbf{u} - \mathbf{v}\|_\infty$ for all $s'$:
$$\gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) \left| u(s') - v(s') \right| \le \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) \|\mathbf{u} - \mathbf{v}\|_\infty = \gamma \|\mathbf{u} - \mathbf{v}\|_\infty \sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s)$$
Since $\mathbf{P}^\pi$ is row-stochastic, $\sum_{s' \in \mathcal{S}} \mathcal{P}^\pi(s' \mid s) = 1$. Hence:
$$|(\mathcal{T}^\pi \mathbf{u})(s) - (\mathcal{T}^\pi \mathbf{v})(s)| \le \gamma \|\mathbf{u} - \mathbf{v}\|_\infty, \quad \forall s \in \mathcal{S}$$
Taking the supremum over all $s \in \mathcal{S}$ on the left-hand side:
$$\|\mathcal{T}^\pi \mathbf{u} - \mathcal{T}^\pi \mathbf{v}\|_\infty \equiv \max_{s \in \mathcal{S}} |(\mathcal{T}^\pi \mathbf{u})(s) - (\mathcal{T}^\pi \mathbf{v})(s)| \le \gamma \|\mathbf{u} - \mathbf{v}\|_\infty$$
Since $\gamma \in [0, 1)$, $\mathcal{T}^\pi$ is a strict $\gamma$-contraction mapping on the Banach space $(\mathbb{R}^n, \|\cdot\|_\infty)$.

**Step 3: Unique Fixed Point and Banach Convergence Rate**
By the Banach Fixed-Point Theorem:
1. There exists a unique fixed point $\mathbf{v}^\pi \in \mathbb{R}^n$ such that $\mathcal{T}^\pi \mathbf{v}^\pi = \mathbf{v}^\pi$.
2. For any initial value function $\mathbf{v}_0 \in \mathbb{R}^n$, the sequence $\mathbf{v}_{k+1} = \mathcal{T}^\pi \mathbf{v}_k$ satisfies:
   $$\|\mathbf{v}_{k+1} - \mathbf{v}_k\|_\infty = \|\mathcal{T}^\pi \mathbf{v}_k - \mathcal{T}^\pi \mathbf{v}_{k-1}\|_\infty \le \gamma \|\mathbf{v}_k - \mathbf{v}_{k-1}\|_\infty \le \gamma^k \|\mathbf{v}_1 - \mathbf{v}_0\|_\infty$$
3. For any $m > k$, telescoping the difference:
   $$\|\mathbf{v}_m - \mathbf{v}_k\|_\infty \le \sum_{j=k}^{m-1} \|\mathbf{v}_{j+1} - \mathbf{v}_j\|_\infty \le \sum_{j=k}^{m-1} \gamma^j \|\mathbf{v}_1 - \mathbf{v}_0\|_\infty = \gamma^k \|\mathbf{v}_1 - \mathbf{v}_0\|_\infty \sum_{l=0}^{m-k-1} \gamma^l$$
   Taking the limit $m \to \infty$ (where $\mathbf{v}_m \to \mathbf{v}^\pi$):
   $$\|\mathbf{v}^\pi - \mathbf{v}_k\|_\infty \le \gamma^k \|\mathbf{v}_1 - \mathbf{v}_0\|_\infty \sum_{l=0}^\infty \gamma^l = \frac{\gamma^k}{1 - \gamma} \|\mathbf{v}_1 - \mathbf{v}_0\|_\infty \quad \blacksquare$$

---

#### Derivation 11.3.2: Contraction of the Non-Linear Bellman Optimality Operator $\mathcal{T}^*$ and Sub-Optimality Error Bound

##### Problem Statement & Goal
The Bellman Optimality Operator $\mathcal{T}^*: \mathbb{R}^n \to \mathbb{R}^n$ is defined by:
$$(\mathcal{T}^* V)(s) \equiv \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V(s') \right], \quad \forall s \in \mathcal{S}$$
We prove:
1. The algebraic identity $|\max_x f(x) - \max_x g(x)| \le \max_x |f(x) - g(x)|$.
2. $\mathcal{T}^*$ is a strict $\gamma$-contraction mapping in $L_\infty$ norm: $\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty$.
3. **Value Iteration Stopping Criterion & Sub-Optimality Guarantee:** If at step $k$, $\|V_{k+1} - V_k\|_\infty < \epsilon$, then:
   $$\|V_k - V^*\|_\infty \le \frac{\gamma \epsilon}{1 - \gamma}$$
   and the greedy policy $\pi_k$ extracted with respect to $V_k$ achieves an expected return bounded by:
   $$\|V^{\pi_k} - V^*\|_\infty \le \frac{2 \gamma \epsilon}{1 - \gamma}$$

##### Explicit Assumptions
1. Finite or compact action space $\mathcal{A}$ ensuring the maximum is attained.
2. Finite state space $|\mathcal{S}| = n < \infty$.
3. Transition distributions $\mathcal{P}(\cdot \mid s, a)$ are valid probability distributions for all $(s, a)$.
4. Discount factor satisfies $\gamma \in [0, 1)$.

##### Underlying Intuition
The non-linearity in $\mathcal{T}^*$ arises solely from the $\max_{a}$ operator. The maximum operator is 1-Lipschitz (non-expansive): picking the best action cannot magnify differences between functions. Since the inner expectation is a $\gamma$-contraction, composing a 1-Lipschitz maximum with a $\gamma$-contraction yields a strict $\gamma$-contraction overall.

##### End-to-End Mathematical Derivation

**Step 1: Non-Expansion of the Maximum Operator**
Let $f, g: \mathcal{A} \to \mathbb{R}$ be two bounded real-valued functions.
Let $a_f^* \in \arg\max_{a \in \mathcal{A}} f(a)$. Then:
$$\max_{a \in \mathcal{A}} f(a) - \max_{a \in \mathcal{A}} g(a) = f(a_f^*) - \max_{a \in \mathcal{A}} g(a) \le f(a_f^*) - g(a_f^*) \le \max_{a \in \mathcal{A}} \left( f(a) - g(a) \right) \le \max_{a \in \mathcal{A}} |f(a) - g(a)|$$
By symmetry, letting $a_g^* \in \arg\max_{a \in \mathcal{A}} g(a)$:
$$\max_{a \in \mathcal{A}} g(a) - \max_{a \in \mathcal{A}} f(a) \le g(a_g^*) - f(a_g^*) \le \max_{a \in \mathcal{A}} |g(a) - f(a)| = \max_{a \in \mathcal{A}} |f(a) - g(a)|$$
Combining both inequalities establishes:
$$\left| \max_{a \in \mathcal{A}} f(a) - \max_{a \in \mathcal{A}} g(a) \right| \le \max_{a \in \mathcal{A}} |f(a) - g(a)|$$

**Step 2: Proof of Contraction for $\mathcal{T}^*$**
Let $U, V \in \mathbb{R}^n$. For any fixed state $s \in \mathcal{S}$, define:
$$f_s(a) \equiv \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) U(s')$$
$$g_s(a) \equiv \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V(s')$$
Then $(\mathcal{T}^* U)(s) = \max_{a} f_s(a)$ and $(\mathcal{T}^* V)(s) = \max_a g_s(a)$.
Applying the lemma from Step 1:
$$\left| (\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s) \right| \le \max_{a \in \mathcal{A}} \left| f_s(a) - g_s(a) \right|$$
Compute the difference $f_s(a) - g_s(a)$:
$$f_s(a) - g_s(a) = \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left( U(s') - V(s') \right)$$
Taking the absolute value:
$$|f_s(a) - g_s(a)| \le \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) |U(s') - V(s')| \le \gamma \|U - V\|_\infty \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) = \gamma \|U - V\|_\infty$$
Because this upper bound holds uniformly for all $a \in \mathcal{A}$:
$$\max_{a \in \mathcal{A}} |f_s(a) - g_s(a)| \le \gamma \|U - V\|_\infty$$
Consequently, for every state $s \in \mathcal{S}$:
$$|(\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s)| \le \gamma \|U - V\|_\infty$$
Taking the maximum over all $s \in \mathcal{S}$:
$$\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty$$
Thus, $\mathcal{T}^*$ is a strict $\gamma$-contraction mapping.

**Step 3: Derivation of the Sub-Optimality Error Bound**
Let $V_k$ be the value estimate at iteration $k$, and $V_{k+1} = \mathcal{T}^* V_k$.
By the triangle inequality for any $m \ge 1$:
$$\|V_{k+m} - V_k\|_\infty \le \sum_{j=0}^{m-1} \|V_{k+j+1} - V_{k+j}\|_\infty \le \sum_{j=0}^{m-1} \gamma^j \|V_{k+1} - V_k\|_\infty$$
Taking the limit $m \to \infty$ where $V_{k+m} \to V^*$:
$$\|V^* - V_k\|_\infty \le \frac{1}{1 - \gamma} \|V_{k+1} - V_k\|_\infty$$
Furthermore, evaluating the error from $V_{k+1}$:
$$\|V^* - V_{k+1}\|_\infty = \|\mathcal{T}^* V^* - \mathcal{T}^* V_k\|_\infty \le \gamma \|V^* - V_k\|_\infty \le \frac{\gamma}{1 - \gamma} \|V_{k+1} - V_k\|_\infty$$
If the algorithm terminates when $\|V_{k+1} - V_k\|_\infty < \epsilon$, then:
$$\|V_{k+1} - V^*\|_\infty \le \frac{\gamma \epsilon}{1 - \gamma}$$

**Step 4: Sub-Optimality of the Greedy Policy**
Let $\pi$ be greedy with respect to $V_k$, so $\mathcal{T}^\pi V_k = \mathcal{T}^* V_k = V_{k+1}$.
Then:
$$\begin{aligned}
\|V^* - V^\pi\|_\infty &= \|V^* - V_{k+1} + V_{k+1} - V^\pi\|_\infty \\
&\le \|V^* - V_{k+1}\|_\infty + \|\mathcal{T}^\pi V_k - \mathcal{T}^\pi V^\pi\|_\infty \\
&\le \|V^* - V_{k+1}\|_\infty + \gamma \|V_k - V^\pi\|_\infty \\
&\le \|V^* - V_{k+1}\|_\infty + \gamma \left( \|V_k - V^*\|_\infty + \|V^* - V^\pi\|_\infty \right)
\end{aligned}$$
Rearranging terms involving $\|V^* - V^\pi\|_\infty$:
$$(1 - \gamma) \|V^* - V^\pi\|_\infty \le \|V^* - V_{k+1}\|_\infty + \gamma \|V_k - V^*\|_\infty$$
Using $\|V^* - V_{k+1}\|_\infty \le \frac{\gamma \epsilon}{1 - \gamma}$ and $\|V_k - V^*\|_\infty \le \frac{\epsilon}{1 - \gamma}$:
$$(1 - \gamma) \|V^* - V^\pi\|_\infty \le \frac{\gamma \epsilon}{1 - \gamma} + \frac{\gamma \epsilon}{1 - \gamma} = \frac{2 \gamma \epsilon}{1 - \gamma}$$
Dividing by $(1 - \gamma)$:
$$\|V^* - V^\pi\|_\infty \le \frac{2 \gamma \epsilon}{(1 - \gamma)^2} \quad \text{or for } V_{k+1}: \quad \|V^* - V^{\pi_{k+1}}\|_\infty \le \frac{2 \gamma \epsilon}{1 - \gamma} \quad \blacksquare$$

---

#### Derivation 11.3.3: The Policy Improvement Theorem via Monotonic Telescoping Backups

##### Problem Statement & Goal
Let $\pi$ and $\pi'$ be any pair of deterministic stationary policies such that for all states $s \in \mathcal{S}$:
$$Q^\pi(s, \pi'(s)) \ge V^\pi(s)$$
We prove that:
1. The policy $\pi'$ achieves weak dominance over $\pi$ in every state:
   $$V^{\pi'}(s) \ge V^\pi(s), \quad \forall s \in \mathcal{S}$$
2. If there exists at least one state $s_0 \in \mathcal{S}$ where $Q^\pi(s_0, \pi'(s_0)) > V^\pi(s_0)$, and $s_0$ is reachable under $\pi'$, then $V^{\pi'}$ strictly dominates $V^\pi$:
   $$V^{\pi'}(s) \ge V^\pi(s) \; \forall s, \quad \text{and} \quad V^{\pi'}(s_0) > V^\pi(s_0)$$
3. If $Q^\pi(s, \pi'(s)) = V^\pi(s)$ for all $s \in \mathcal{S}$, then $V^\pi(s) = V^*(s)$ and $\pi$ is an optimal policy.

##### Explicit Assumptions
1. Discrete-time MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ with discount factor $\gamma \in [0, 1)$.
2. Rewards are uniformly bounded: $|R_t| \le R_{\max} < \infty$.
3. $\pi'(s) \in \arg\max_{a \in \mathcal{A}} Q^\pi(s, a)$.

##### Underlying Intuition
Suppose an agent follows policy $\pi'$ for exactly 1 time step, and then follows policy $\pi$ forever after. The expected return is precisely $Q^\pi(s, \pi'(s))$, which by hypothesis is $\ge V^\pi(s)$. Now suppose the agent follows $\pi'$ for 2 time steps before reverting to $\pi$; by monotonicity of expectations, this return is even higher. Telescoping this substitution all the way to infinity replaces policy $\pi$ entirely with policy $\pi'$, proving that each additional step of improvement compounds monotonically.

##### End-to-End Mathematical Derivation

**Step 1: The One-Step Expansion**
By the hypothesis of greedy choice:
$$V^\pi(s) \le Q^\pi(s, \pi'(s))$$
Recall the definition of $Q^\pi$:
$$Q^\pi(s, \pi'(s)) = \mathbb{E}\left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \;\middle|\; S_t = s, A_t = \pi'(s) \right]$$
Therefore:
$$V^\pi(s) \le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \;\middle|\; S_t = s \right]$$

**Step 2: Recursive Monotonic Substitution**
Because the inequality $V^\pi(s') \le \mathbb{E}_{\pi'}[ R_{t+2} + \gamma V^\pi(S_{t+2}) \mid S_{t+1} = s']$ holds for all $s' \in \mathcal{S}$, we substitute it into the right-hand side:
$$\begin{aligned}
V^\pi(s) &\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \;\middle|\; S_t = s \right] \\
&\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma \mathbb{E}_{\pi'} \left[ R_{t+2} + \gamma V^\pi(S_{t+2}) \;\middle|\; S_{t+1} \right] \;\middle|\; S_t = s \right] \\
&= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) \;\middle|\; S_t = s \right]
\end{aligned}$$
Applying the substitution repeatedly $K$ times by induction:
$$V^\pi(s) \le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{K-1} R_{t+K} + \gamma^K V^\pi(S_{t+K}) \;\middle|\; S_t = s \right]$$
$$V^\pi(s) \le \mathbb{E}_{\pi'} \left[ \sum_{k=0}^{K-1} \gamma^k R_{t+k+1} + \gamma^K V^\pi(S_{t+K}) \;\middle|\; S_t = s \right]$$

**Step 3: Taking the Infinite Horizon Limit ($K \to \infty$)**
Because rewards are bounded ($\|V^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma} < \infty$) and $\gamma \in [0, 1)$:
$$\lim_{K \to \infty} \mathbb{E}_{\pi'} \left[ \gamma^K V^\pi(S_{t+K}) \;\middle|\; S_t = s \right] \le \lim_{K \to \infty} \gamma^K \frac{R_{\max}}{1 - \gamma} = 0$$
By the dominated convergence theorem, taking the limit as $K \to \infty$:
$$V^\pi(s) \le \mathbb{E}_{\pi'} \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right] = V^{\pi'}(s)$$
Thus, $V^{\pi'}(s) \ge V^\pi(s)$ for all $s \in \mathcal{S}$.

**Step 4: Strict Improvement and Optimality Condition**
If there exists a state $s_0$ where $Q^\pi(s_0, \pi'(s_0)) = V^\pi(s_0) + \delta$ with $\delta > 0$, then:
$$V^{\pi'}(s_0) \ge V^\pi(s_0) + \delta > V^\pi(s_0)$$
Finally, if no improvement can be made in any state, then for all $s$:
$$V^\pi(s) = \max_{a \in \mathcal{A}} Q^\pi(s, a) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]$$
This matches the unique Bellman Optimality Equation $V = \mathcal{T}^* V$. By uniqueness of the fixed point of $\mathcal{T}^*$, $V^\pi = V^*$ and $\pi$ is globally optimal. $\blacksquare$

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

### Illustration 3: 3-State Bellman Expectation System Solved Analytically and Verified via Fixed-Point Contraction

**Problem:**
Consider a 3-state MRP with $\mathcal{S} = \{S_1, S_2, S_3\}$, discount factor $\gamma = 0.60$, transition matrix:
$$\mathbf{P} = \begin{bmatrix} 0.5 & 0.5 & 0.0 \\ 0.0 & 0.5 & 0.5 \\ 0.5 & 0.0 & 0.5 \end{bmatrix}$$
and immediate reward vector $\mathbf{r} = [4.0, 2.0, -1.0]^\top$.
1. Solve for the analytical value vector $\mathbf{V} = (\mathbf{I} - \gamma \mathbf{P})^{-1} \mathbf{r}$.
2. Starting from initial estimate $\mathbf{V}_0 = [0.0, 0.0, 0.0]^\top$, perform 2 iterations of fixed-point contraction $\mathbf{V}_{k+1} = \mathbf{r} + \gamma \mathbf{P} \mathbf{V}_k$.
3. Numerically verify that the contraction condition $\|\mathbf{V}_2 - \mathbf{V}_1\|_\infty \le \gamma \|\mathbf{V}_1 - \mathbf{V}_0\|_\infty$ is strictly satisfied.

**Solution:**

**Step 1: Exact Analytical Solution via Matrix Inversion**
With $\gamma = 0.60 = \frac{3}{5}$:
$$\gamma \mathbf{P} = \begin{bmatrix} 0.30 & 0.30 & 0.00 \\ 0.00 & 0.30 & 0.30 \\ 0.30 & 0.00 & 0.30 \end{bmatrix} \implies \mathbf{A} = \mathbf{I} - \gamma \mathbf{P} = \begin{bmatrix} 0.70 & -0.30 & 0.00 \\ 0.00 & 0.70 & -0.30 \\ -0.30 & 0.00 & 0.70 \end{bmatrix}$$
Determinant:
$$\det(\mathbf{A}) = (0.7)^3 - (0.3)^3 = 0.343 - 0.027 = \mathbf{0.3160 = \frac{79}{250}}$$
Inverting $\mathbf{A}$ via adjugate matrix:
$$(\mathbf{I} - \gamma \mathbf{P})^{-1} = \frac{1}{0.316} \begin{bmatrix} 0.49 & 0.21 & 0.09 \\ 0.09 & 0.49 & 0.21 \\ 0.21 & 0.09 & 0.49 \end{bmatrix} \approx \begin{bmatrix} 1.5506 & 0.6646 & 0.2848 \\ 0.2848 & 1.5506 & 0.6646 \\ 0.6646 & 0.2848 & 1.5506 \end{bmatrix}$$
Compute true value vector:
$$\mathbf{V}^* = \frac{1}{0.316} \begin{bmatrix} 0.49(4) + 0.21(2) + 0.09(-1) \\ 0.09(4) + 0.49(2) + 0.21(-1) \\ 0.21(4) + 0.09(2) + 0.49(-1) \end{bmatrix} = \frac{1}{0.316} \begin{bmatrix} 1.96 + 0.42 - 0.09 \\ 0.36 + 0.98 - 0.21 \\ 0.84 + 0.18 - 0.49 \end{bmatrix} = \frac{1}{0.316} \begin{bmatrix} 2.29 \\ 1.13 \\ 0.53 \end{bmatrix} \approx \mathbf{\begin{bmatrix} 7.2468 \\ 3.5759 \\ 1.6772 \end{bmatrix}}$$

**Step 2: Iteration 1 of Fixed-Point Contraction ($k = 0 \to 1$)**
With $\mathbf{V}_0 = [0.0, 0.0, 0.0]^\top$:
$$\mathbf{V}_1 = \mathbf{r} + \gamma \mathbf{P} \mathbf{V}_0 = \mathbf{r} + \mathbf{0} = \mathbf{\begin{bmatrix} 4.0000 \\ 2.0000 \\ -1.0000 \end{bmatrix}}$$
The step-change norm is:
$$\|\mathbf{V}_1 - \mathbf{V}_0\|_\infty = \max(|4.0 - 0|, |2.0 - 0|, |-1.0 - 0|) = \mathbf{4.0000}$$

**Step 3: Iteration 2 of Fixed-Point Contraction ($k = 1 \to 2$)**
Compute $\mathbf{P} \mathbf{V}_1$:
$$\mathbf{P} \mathbf{V}_1 = \begin{bmatrix} 0.5(4.0) + 0.5(2.0) \\ 0.5(2.0) + 0.5(-1.0) \\ 0.5(4.0) + 0.5(-1.0) \end{bmatrix} = \begin{bmatrix} 2.0 + 1.0 \\ 1.0 - 0.5 \\ 2.0 - 0.5 \end{bmatrix} = \begin{bmatrix} 3.0000 \\ 0.5000 \\ 1.5000 \end{bmatrix}$$
Add discounted reward $\mathbf{V}_2 = \mathbf{r} + 0.60 \mathbf{P} \mathbf{V}_1$:
$$\mathbf{V}_2 = \begin{bmatrix} 4.0 \\ 2.0 \\ -1.0 \end{bmatrix} + 0.60 \begin{bmatrix} 3.0 \\ 0.5 \\ 1.5 \end{bmatrix} = \begin{bmatrix} 4.0 + 1.80 \\ 2.0 + 0.30 \\ -1.0 + 0.90 \end{bmatrix} = \mathbf{\begin{bmatrix} 5.8000 \\ 2.3000 \\ -0.1000 \end{bmatrix}}$$

**Step 4: Verify Contraction Property**
Compute the step difference vector $\mathbf{V}_2 - \mathbf{V}_1$:
$$\mathbf{V}_2 - \mathbf{V}_1 = \begin{bmatrix} 5.80 - 4.00 \\ 2.30 - 2.00 \\ -0.10 - (-1.00) \end{bmatrix} = \begin{bmatrix} 1.8000 \\ 0.3000 \\ 0.9000 \end{bmatrix}$$
Infinity norm:
$$\|\mathbf{V}_2 - \mathbf{V}_1\|_\infty = \max(1.80, 0.30, 0.90) = \mathbf{1.8000}$$
Check contraction inequality:
$$\|\mathbf{V}_2 - \mathbf{V}_1\|_\infty = 1.8000 \le \gamma \|\mathbf{V}_1 - \mathbf{V}_0\|_\infty = 0.60 \times 4.0000 = \mathbf{2.4000}$$
Because $1.8000 \le 2.4000$, the $\gamma$-contraction bound is strictly satisfied!

---

### Illustration 4: Exact 2-Iteration Trace of the Non-Linear Bellman Optimality Operator $\mathcal{T}^*$

**Problem:**
Consider an MDP with $\mathcal{S} = \{S_1, S_2\}$, $\mathcal{A} = \{a_1, a_2\}$, and $\gamma = 0.90$.
The transitions and expected immediate rewards are:
- In $S_1$:
  - Action $a_1$: $\mathcal{P}(S_1 \mid S_1, a_1) = 0.70, \; \mathcal{P}(S_2 \mid S_1, a_1) = 0.30, \quad \mathcal{R}(S_1, a_1) = 5.0$
  - Action $a_2$: $\mathcal{P}(S_1 \mid S_1, a_2) = 0.20, \; \mathcal{P}(S_2 \mid S_1, a_2) = 0.80, \quad \mathcal{R}(S_1, a_2) = 10.0$
- In $S_2$:
  - Action $a_1$: $\mathcal{P}(S_1 \mid S_2, a_1) = 0.90, \; \mathcal{P}(S_2 \mid S_2, a_1) = 0.10, \quad \mathcal{R}(S_2, a_1) = 0.0$
  - Action $a_2$: $\mathcal{P}(S_1 \mid S_2, a_2) = 0.10, \; \mathcal{P}(S_2 \mid S_2, a_2) = 0.90, \quad \mathcal{R}(S_2, a_2) = -2.0$

Starting from initial value function $\mathbf{V}_0 = [0.0, 0.0]^\top$:
1. Compute all four action-values $Q_1(s, a)$ and determine $\mathbf{V}_1 = \mathcal{T}^* \mathbf{V}_0$ along with the greedy policy $\pi_1$.
2. Compute all four action-values $Q_2(s, a)$ and determine $\mathbf{V}_2 = \mathcal{T}^* \mathbf{V}_1$ along with the greedy policy $\pi_2$.
3. Verify the non-linear contraction property $\|\mathbf{V}_2 - \mathbf{V}_1\|_\infty \le \gamma \|\mathbf{V}_1 - \mathbf{V}_0\|_\infty$.

**Solution:**

**Step 1: Iteration 1 ($k = 0 \to 1$)**
With $\mathbf{V}_0 = [0, 0]^\top$, future discounted values are identically zero:
- State $S_1$:
  - $Q_1(S_1, a_1) = 5.0 + 0.90 \left[ 0.7(0) + 0.3(0) \right] = \mathbf{5.0000}$
  - $Q_1(S_1, a_2) = 10.0 + 0.90 \left[ 0.2(0) + 0.8(0) \right] = \mathbf{10.0000}$
  - $V_1(S_1) = \max(5.0, 10.0) = \mathbf{10.0000}, \quad \pi_1(S_1) = a_2$
- State $S_2$:
  - $Q_1(S_2, a_1) = 0.0 + 0.90 \left[ 0.9(0) + 0.1(0) \right] = \mathbf{0.0000}$
  - $Q_1(S_2, a_2) = -2.0 + 0.90 \left[ 0.1(0) + 0.9(0) \right] = \mathbf{-2.0000}$
  - $V_1(S_2) = \max(0.0, -2.0) = \mathbf{0.0000}, \quad \pi_1(S_2) = a_1$

Result after step 1:
$$\mathbf{V}_1 = \begin{bmatrix} 10.0000 \\ 0.0000 \end{bmatrix}, \quad \pi_1 = \{S_1 \to a_2, \; S_2 \to a_1\}$$
$$\|\mathbf{V}_1 - \mathbf{V}_0\|_\infty = \max(|10 - 0|, |0 - 0|) = \mathbf{10.0000}$$

**Step 2: Iteration 2 ($k = 1 \to 2$)**
Evaluate $Q_2(s, a) = \mathcal{R}(s, a) + 0.90 \sum_{s'} \mathcal{P}(s' \mid s, a) V_1(s')$ with $\mathbf{V}_1 = [10.0, 0.0]^\top$:
- State $S_1$:
  - $Q_2(S_1, a_1) = 5.0 + 0.90 \left[ 0.70(10.0) + 0.30(0.0) \right] = 5.0 + 0.90(7.0) = 5.0 + 6.30 = \mathbf{11.3000}$
  - $Q_2(S_1, a_2) = 10.0 + 0.90 \left[ 0.20(10.0) + 0.80(0.0) \right] = 10.0 + 0.90(2.0) = 10.0 + 1.80 = \mathbf{11.8000}$
  - $V_2(S_1) = \max(11.30, 11.80) = \mathbf{11.8000}, \quad \pi_2(S_1) = a_2$
- State $S_2$:
  - $Q_2(S_2, a_1) = 0.0 + 0.90 \left[ 0.90(10.0) + 0.10(0.0) \right] = 0.0 + 0.90(9.0) = \mathbf{8.1000}$
  - $Q_2(S_2, a_2) = -2.0 + 0.90 \left[ 0.10(10.0) + 0.90(0.0) \right] = -2.0 + 0.90(1.0) = -2.0 + 0.90 = \mathbf{-1.1000}$
  - $V_2(S_2) = \max(8.10, -1.10) = \mathbf{8.1000}, \quad \pi_2(S_2) = a_1$

Result after step 2:
$$\mathbf{V}_2 = \begin{bmatrix} 11.8000 \\ 8.1000 \end{bmatrix}, \quad \pi_2 = \{S_1 \to a_2, \; S_2 \to a_1\}$$

**Step 3: Verification of Non-Linear Operator Contraction**
The difference between iterates is:
$$\mathbf{V}_2 - \mathbf{V}_1 = \begin{bmatrix} 11.80 - 10.00 \\ 8.10 - 0.00 \end{bmatrix} = \begin{bmatrix} 1.8000 \\ 8.1000 \end{bmatrix}$$
$$\|\mathbf{V}_2 - \mathbf{V}_1\|_\infty = \max(1.80, 8.10) = \mathbf{8.1000}$$
Check against the theoretical contraction bound:
$$\|\mathbf{V}_2 - \mathbf{V}_1\|_\infty = 8.1000 \le \gamma \|\mathbf{V}_1 - \mathbf{V}_0\|_\infty = 0.90 \times 10.0000 = \mathbf{9.0000}$$
Strict contraction holds with margin: $8.1000 < 9.0000$.

---

### Illustration 5: Exact Value Iteration Stopping Criterion and Error Bound Verification

**Problem:**
For the MDP of Illustration 4 ($\gamma = 0.90$):
1. Compute the analytical fixed-point $V^*$ by solving the linear system under optimal stationary policy $\pi^*(S_1) = a_2, \pi^*(S_2) = a_1$.
2. Suppose Value Iteration terminates after iteration 2 because step change was $\|V_2 - V_1\|_\infty = 8.1000$. Calculate the theoretical upper bound on error $\|V_2 - V^*\|_\infty$ guaranteed by the Banach contraction theorem and compare it against the actual true error.
3. Calculate the threshold $\epsilon_{\text{stop}}$ such that terminating with $\|V_{k+1} - V_k\|_\infty < \epsilon_{\text{stop}}$ mathematically guarantees that the extracted policy is within $\delta = 0.05$ of the optimal return: $\|V^{\pi_k} - V^*\|_\infty \le 0.05$.

**Solution:**

**Step 1: Analytical Closed-Form $V^*$**
Under optimal policy $\pi^*$:
$$V^*(S_1) = 10.0 + 0.90 \left[ 0.20 V^*(S_1) + 0.80 V^*(S_2) \right] = 10.0 + 0.18 V^*(S_1) + 0.72 V^*(S_2)$$
$$0.82 V^*(S_1) - 0.72 V^*(S_2) = 10.0$$
$$V^*(S_2) = 0.0 + 0.90 \left[ 0.90 V^*(S_1) + 0.10 V^*(S_2) \right] = 0.81 V^*(S_1) + 0.09 V^*(S_2)$$
$$0.91 V^*(S_2) - 0.81 V^*(S_1) = 0.0 \implies V^*(S_2) = \frac{0.81}{0.91} V^*(S_1) = \frac{81}{91} V^*(S_1)$$
Substitute into the first equation:
$$0.82 V^*(S_1) - 0.72 \left( \frac{81}{91} V^*(S_1) \right) = 10.0$$
$$\left( \frac{0.82 \times 91 - 0.72 \times 81}{91} \right) V^*(S_1) = 10.0 \implies \left( \frac{74.62 - 58.32}{91} \right) V^*(S_1) = 10.0$$
$$\frac{16.30}{91} V^*(S_1) = 10.0 \implies V^*(S_1) = \frac{910}{16.30} = \mathbf{\frac{9100}{163} \approx 55.8282}$$
$$V^*(S_2) = \frac{81}{91} \left( \frac{9100}{163} \right) = \mathbf{\frac{8100}{163} \approx 49.6933}$$

**Step 2: Comparison of Theoretical Error Bound vs. Actual Error**
From Derivation 11.3.2, the distance to the fixed point from iterate $V_k$ is bounded by:
$$\|V_k - V^*\|_\infty \le \frac{\gamma}{1 - \gamma} \|V_k - V_{k-1}\|_\infty$$
For $k = 2$, with $\|V_2 - V_1\|_\infty = 8.1000$ and $\gamma = 0.90$:
$$\text{Theoretical Upper Bound} = \frac{0.90}{1 - 0.90} \times 8.1000 = \frac{0.90}{0.10} \times 8.1000 = 9 \times 8.1000 = \mathbf{72.9000}$$
Now compute the actual true error:
$$\|V_2 - V^*\|_\infty = \max\left( |11.8000 - 55.8282|, \; |8.1000 - 49.6933| \right) = \max(44.0282, \; 41.5933) = \mathbf{44.0282}$$
The true error ($44.0282$) is strictly bounded by the theoretical contraction bound ($72.9000$), confirming the mathematical validity of the stopping guarantee!

**Step 3: Calculating Value Iteration Stopping Threshold for Policy Guarantee**
By the Policy Sub-Optimality Bound (Derivation 11.3.2):
$$\|V^{\pi_k} - V^*\|_\infty \le \frac{2 \gamma \epsilon_{\text{stop}}}{1 - \gamma}$$
To guarantee that $\|V^{\pi_k} - V^*\|_\infty \le \delta = 0.05$:
$$\frac{2 \gamma \epsilon_{\text{stop}}}{1 - \gamma} \le \delta \implies \epsilon_{\text{stop}} \le \frac{(1 - \gamma) \delta}{2 \gamma}$$
Substituting $\gamma = 0.90$ and $\delta = 0.05$:
$$\epsilon_{\text{stop}} \le \frac{(1 - 0.90) \times 0.05}{2 \times 0.90} = \frac{0.10 \times 0.05}{1.80} = \frac{0.0050}{1.80} = \mathbf{\frac{1}{360} \approx 0.002778}$$
When Value Iteration is terminated once the maximum value change between iterations falls below $\epsilon_{\text{stop}} = 0.002778$, the resulting greedy policy is mathematically guaranteed to achieve at least $99.9\%$ of the optimal return in every state!

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
