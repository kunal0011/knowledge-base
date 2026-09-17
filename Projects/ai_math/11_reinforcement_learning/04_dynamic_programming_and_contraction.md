# Dynamic Programming & The Banach Contraction Mapping Theorem

---

## 1. Intuition & 101 Motivation

### The Grand Guarantee of Convergence
In Chapter 11.3, we formulated the recursive Bellman equations for state and action values. But a critical theoretical question remains:
*If we start with a vector of arbitrary, random numbers $V_0 = [1000, -500, \dots]$ and repeatedly apply the Bellman update rule, are we mathematically guaranteed to converge to the true, unique optimal value function $V^*$?*

The answer is **yes**, and the mathematical foundation is the **Banach Contraction Mapping Theorem**.
In this chapter, we explore **Dynamic Programming (DP)** in reinforcement learning:
1. **Model-Based Setting:** We assume complete knowledge of the environment's transition dynamics $\mathcal{P}(s' \mid s, a)$ and reward function $\mathcal{R}(s, a)$.
2. **Contraction Mapping:** Every time we apply the Bellman operator, the "distance" between any two value function estimates shrinks by at least factor $\gamma < 1$.
3. **Algorithms:** We dissect the two classical algorithms of dynamic programming:
   - **Policy Iteration:** Alternates complete Policy Evaluation ($V^\pi$) with greedy Policy Improvement ($\pi' = \operatorname{greedy}(V^\pi)$).
   - **Value Iteration:** Merges evaluation and improvement into a single, fast Bellman optimality contraction sweep per step.

```
                  THE POLICY ITERATION DANCE
          Value V                                       Policy pi
             ▲                                             ▲
             │                                             │
             │           Evaluation: V = V^pi              │
             │     ●─────────────────────────────────►     │
             │     ▲                                 │     │
             │     │                                 ▼     │
             │     ◄─────────────────────────────────●     │
             │           Improvement: pi = greedy(V)       │
             └─────────────────────────────────────────────┘
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Bellman Operators as Functional Mappings

Let $\mathcal{V} = \mathbb{R}^{|\mathcal{S}|}$ be the vector space of bounded real-valued functions over state space $\mathcal{S}$, equipped with the **Infinity Norm** (Maximum Norm):
$$\|v\|_\infty \equiv \max_{s \in \mathcal{S}} |v(s)|$$
Because $\mathbb{R}^{|\mathcal{S}|}$ with $\|\cdot\|_\infty$ is a complete normed vector space, it is a **Banach Space**.

#### 1. The Bellman Expectation Operator $\mathcal{T}^\pi: \mathcal{V} \to \mathcal{V}$
For a fixed policy $\pi$:
$$\mathbf{(\mathcal{T}^\pi v)(s) \equiv \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) v(s') \right]}$$
In vector-matrix notation:
$$\mathcal{T}^\pi \mathbf{v} = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}$$

#### 2. The Bellman Optimality Operator $\mathcal{T}^*: \mathcal{V} \to \mathcal{V}$
$$\mathbf{(\mathcal{T}^* v)(s) \equiv \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) v(s') \right]}$$

A value function $V^\pi$ is a **fixed point** of $\mathcal{T}^\pi$ ($V^\pi = \mathcal{T}^\pi V^\pi$), and the optimal value function $V^*$ is a **fixed point** of $\mathcal{T}^*$ ($V^* = \mathcal{T}^* V^*$).

---

### 2.2 First-Principles Proof of Contraction for $\mathcal{T}^*$

**Definition (Contraction Mapping):**
An operator $\mathcal{T}: \mathcal{V} \to \mathcal{V}$ on a metric space $(\mathcal{V}, d)$ is a $\gamma$-contraction if there exists $\gamma \in [0, 1)$ such that:
$$d(\mathcal{T} u, \, \mathcal{T} v) \le \gamma \cdot d(u, v), \quad \forall u, v \in \mathcal{V}$$

**Theorem (Bellman Optimality Contraction):**
For any discount factor $\gamma \in [0, 1)$, the Bellman optimality operator $\mathcal{T}^*$ is a $\gamma$-contraction mapping under the infinity norm:
$$\mathbf{\|\mathcal{T}^* u - \mathcal{T}^* v\|_\infty \le \gamma \|u - v\|_\infty, \quad \forall u, v \in \mathbb{R}^{|\mathcal{S}|}}$$

**Proof:**
Let $u, v \in \mathbb{R}^{|\mathcal{S}|}$. Consider an arbitrary state $s \in \mathcal{S}$:
$$|(\mathcal{T}^* u)(s) - (\mathcal{T}^* v)(s)| = \left| \max_{a \in \mathcal{A}} \left( \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) u(s') \right) - \max_{a' \in \mathcal{A}} \left( \mathcal{R}(s, a') + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a') v(s') \right) \right|$$

We utilize the algebraic lemma for any real-valued functions $f, g$:
$$\left| \max_a f(a) - \max_a g(a) \right| \le \max_a |f(a) - g(a)|$$
Applying this lemma:
$$\begin{aligned}
|(\mathcal{T}^* u)(s) - (\mathcal{T}^* v)(s)| &\le \max_{a \in \mathcal{A}} \left| \left( \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) u(s') \right) - \left( \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) v(s') \right) \right| \\
&= \max_{a \in \mathcal{A}} \left| \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) \left( u(s') - v(s') \right) \right|
\end{aligned}$$

Apply the triangle inequality:
$$\le \gamma \max_{a \in \mathcal{A}} \sum_{s'} \mathcal{P}(s' \mid s, a) |u(s') - v(s')|$$
By definition of the infinity norm, $|u(s') - v(s')| \le \|u - v\|_\infty$ for all $s'$. Thus:
$$\le \gamma \max_{a \in \mathcal{A}} \sum_{s'} \mathcal{P}(s' \mid s, a) \|u - v\|_\infty$$
Because transition probabilities sum to 1 ($\sum_{s'} \mathcal{P}(s' \mid s, a) = 1$):
$$= \gamma \|u - v\|_\infty \cdot \underbrace{\max_{a \in \mathcal{A}} \sum_{s'} \mathcal{P}(s' \mid s, a)}_{= 1} = \gamma \|u - v\|_\infty$$

Since this inequality holds for every state $s \in \mathcal{S}$, it holds for the maximum over all states:
$$\|\mathcal{T}^* u - \mathcal{T}^* v\|_\infty = \max_{s \in \mathcal{S}} |(\mathcal{T}^* u)(s) - (\mathcal{T}^* v)(s)| \le \gamma \|u - v\|_\infty \quad \blacksquare$$

*(The proof that $\mathcal{T}^\pi$ is a $\gamma$-contraction follows identically, omitting the $\max$ operator).*

---

### 2.3 The Banach Fixed-Point Theorem & Convergence Guarantees

**The Banach Fixed-Point Theorem (Contraction Principle):**
Let $(\mathcal{V}, \|\cdot\|)$ be a complete metric space (Banach space), and let $\mathcal{T}: \mathcal{V} \to \mathcal{V}$ be a $\gamma$-contraction with $\gamma \in [0, 1)$. Then:
1. **Existence & Uniqueness:** There exists a **unique** fixed point $V^* \in \mathcal{V}$ such that:
   $$\mathcal{T}^* V^* = V^*$$
2. **Global Convergence:** For **any arbitrary initial vector** $V_0 \in \mathcal{V}$, the sequence $V_{k+1} = \mathcal{T}^* V_k$ converges to $V^*$ in norm:
   $$\lim_{k \to \infty} V_k = V^*$$
3. **Exponential Convergence Rate:**
   $$\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty \le \frac{\gamma^k}{1 - \gamma} \|V_1 - V_0\|_\infty$$
4. **Stopping Criterion:**
   If at step $k$, $\|V_{k+1} - V_k\|_\infty < \epsilon \frac{1 - \gamma}{2\gamma}$, then the true error to optimal value is bounded by:
   $$\|V_{k+1} - V^*\|_\infty < \frac{\epsilon}{2}$$

---

### 2.4 The Policy Improvement Theorem

Before detailing Policy Iteration, we establish why greedy action selection strictly improves a policy.

**Theorem (Policy Improvement Theorem):**
Let $\pi$ and $\pi'$ be two deterministic policies such that for all states $s \in \mathcal{S}$:
$$Q^\pi(s, \pi'(s)) \ge V^\pi(s)$$
Then the policy $\pi'$ must be globally as good as or strictly better than $\pi$ in every state:
$$\mathbf{V^{\pi'}(s) \ge V^\pi(s), \quad \forall s \in \mathcal{S}}$$
Moreover, if there is a strict inequality $Q^\pi(s, \pi'(s)) > V^\pi(s)$ in at least one state, then $V^{\pi'}(s) > V^\pi(s)$ strictly in that state.

**Proof by Induction (Repeated Unrolling):**
$$V^\pi(s) \le Q^\pi(s, \pi'(s)) = \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \;\middle|\; S_t = s \right]$$
Now substitute $V^\pi(S_{t+1}) \le Q^\pi(S_{t+1}, \pi'(S_{t+1}))$:
$$\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma Q^\pi(S_{t+1}, \pi'(S_{t+1})) \;\middle|\; S_t = s \right]$$
$$= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) \;\middle|\; S_t = s \right]$$
Continuing this expansion to infinity:
$$\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \;\middle|\; S_t = s \right] = V^{\pi'}(s) \quad \blacksquare$$

---

### 2.5 Classical Dynamic Programming Algorithms

#### 1. Policy Iteration
Policy Iteration alternates between two explicit phases until the policy stops changing:

1. **Policy Evaluation:** Given current policy $\pi_k$, compute $V^{\pi_k}$ by solving $\mathbf{v} = (\mathbf{I} - \gamma \mathbf{P}^{\pi_k})^{-1} \mathbf{r}^{\pi_k}$ or iterating $V_{m+1} = \mathcal{T}^{\pi_k} V_m$ until $\|V_{m+1} - V_m\|_\infty < \epsilon$.
2. **Policy Improvement:** Extract a strictly superior policy $\pi_{k+1}$ by acting greedily with respect to $V^{\pi_k}$:
   $$\pi_{k+1}(s) = \operatorname{argmax}_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^{\pi_k}(s') \right]$$
3. **Termination:** If $\pi_{k+1}(s) = \pi_k(s)$ for all $s$, stop! The policy is optimal ($\pi^* = \pi_k$).
   Since the number of deterministic policies is finite ($|\mathcal{A}|^{|\mathcal{S}|}$) and each improvement strictly increases value, **Policy Iteration converges in a finite number of iterations**!

---

#### 2. Value Iteration
Policy Iteration spends immense computational effort running policy evaluation to convergence at every step.
**Value Iteration** truncates evaluation to **a single sweep**:
$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V_k(s') \right]$$
By the Banach Fixed-Point Theorem, $V_k \to V^*$ at geometric rate $\mathcal{O}(\gamma^k)$. Once $\|V_{k+1} - V_k\|_\infty < \theta$, extract optimal policy $\pi^*(s) = \operatorname{argmax}_a Q^*(s, a)$ in a single final step.

---

### 2.6 Summary Comparison Matrix

| Property | Policy Iteration | Value Iteration |
| :--- | :--- | :--- |
| **Inner Loop** | Exact or multi-step evaluation ($\mathcal{T}^\pi$) | Single optimality sweep ($\mathcal{T}^*$) |
| **Outer Iterations** | Very few ($3-10$ iterations typically) | More iterations ($\sim \frac{\log \epsilon}{\log \gamma}$) |
| **Cost per Step** | High ($\mathcal{O}(\vert \mathcal{S} \vert^3)$ or $\mathcal{O}(m \vert \mathcal{S} \vert^2 \vert \mathcal{A} \vert)$) | Fast ($\mathcal{O}(\vert \mathcal{S} \vert^2 \vert \mathcal{A} \vert)$) |
| **Monotonicity** | Strictly monotonic $V^{\pi_{k+1}} \ge V^{\pi_k}$ | Monotonic contraction toward $V^*$ |
| **Guarantees** | Finite termination to exact $\pi^*$ | Asymptotic geometric convergence to $V^*$ |

---

### 2.7 First-Principles Mathematical Derivations

#### Derivation 11.4.1: Finite-Time Termination of Policy Iteration and Newton-Raphson Equivalence

##### Problem Statement & Goal
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ be a finite MDP with $|\mathcal{S}| = n$ and $|\mathcal{A}| = m$. Consider the Policy Iteration algorithm generating a sequence of stationary deterministic policies $\{\pi_0, \pi_1, \pi_2, \dots\}$.
We prove:
1. If $\pi_{k+1} \ne \pi_k$, then $V^{\pi_{k+1}}(s) \ge V^{\pi_k}(s)$ for all $s \in \mathcal{S}$ and there exists at least one state $s_0$ where $V^{\pi_{k+1}}(s_0) > V^{\pi_k}(s_0)$.
2. Policy Iteration terminates in at most $m^n$ iterations, and in practice exhibits superlinear/quadratic convergence because it is mathematically isomorphic to Newton-Raphson method for solving the non-smooth Bellman operator equation $F(V) \equiv V - \mathcal{T}^* V = \mathbf{0}$.

##### Explicit Assumptions
1. Finite state space $|\mathcal{S}| = n < \infty$ and finite action space $|\mathcal{A}| = m < \infty$.
2. The total number of deterministic policies is $|\Pi| = m^n < \infty$.
3. Discount factor satisfies $\gamma \in [0, 1)$.
4. In case of ties during greedy action selection, ties are broken systematically according to a fixed deterministic ordering, ensuring no policy cycles occur between equally valued actions.

##### Underlying Intuition
Because the value function strictly improves at every iteration where the policy changes, the algorithm can never revisit a previously evaluated policy. Since the universe of candidate policies is finite ($m^n$), the algorithm must terminate in a finite number of steps. Furthermore, when viewed in value space, evaluating a policy linearizes the Bellman optimality operator around that policy, exactly like a Newton step on a piecewise-linear operator.

##### End-to-End Mathematical Derivation

**Step 1: Strict Monotonic Value Progression**
Let $\pi_k$ be the policy at iteration $k$, and let $\pi_{k+1}(s) \in \arg\max_{a} Q^{\pi_k}(s, a)$.
By definition of greedy improvement:
$$Q^{\pi_k}(s, \pi_{k+1}(s)) = \max_{a \in \mathcal{A}} Q^{\pi_k}(s, a) \ge Q^{\pi_k}(s, \pi_k(s)) = V^{\pi_k}(s), \quad \forall s \in \mathcal{S}$$
If $\pi_{k+1}$ is not identical to $\pi_k$ (and assuming tie-breaking selects a new action only when strictly better or under a strict order):
$$\exists s_0 \in \mathcal{S} \quad \text{such that} \quad Q^{\pi_k}(s_0, \pi_{k+1}(s_0)) > V^{\pi_k}(s_0)$$
From the Policy Improvement Theorem (Derivation 11.3.3), unrolling the expectation under $\pi_{k+1}$:
$$V^{\pi_k}(s) \le \mathbb{E}_{\pi_{k+1}} \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \;\middle|\; S_0 = s \right] = V^{\pi_{k+1}}(s), \quad \forall s \in \mathcal{S}$$
and for state $s_0$:
$$V^{\pi_{k+1}}(s_0) > V^{\pi_k}(s_0)$$
Thus, the value vector strictly increases: $\mathbf{v}^{\pi_{k+1}} > \mathbf{v}^{\pi_k}$.

**Step 2: Finite Termination Guarantee**
Define the set of all deterministic stationary policies $\Pi = \{ \pi: \mathcal{S} \to \mathcal{A} \}$.
The cardinality of this set is:
$$|\Pi| = |\mathcal{A}|^{|\mathcal{S}|} = m^n < \infty$$
Because $\mathbf{v}^{\pi_{k+1}} > \mathbf{v}^{\pi_k}$, each policy encountered in the sequence $\{\pi_0, \pi_1, \dots\}$ has a strictly higher value function than all preceding policies:
$$\mathbf{v}^{\pi_j} > \mathbf{v}^{\pi_i}, \quad \forall j > i$$
Hence, no policy can appear more than once in the sequence.
By the Pigeonhole Principle, the sequence of distinct policies cannot exceed the total number of policies:
$$\text{Iterations to Termination} \le m^n < \infty$$
When $\pi_{k+1} = \pi_k$, no action improves value, meaning $V^{\pi_k} = \mathcal{T}^* V^{\pi_k} = V^*$. The algorithm terminates at the exact globally optimal policy.

**Step 3: Equivalence to the Newton-Raphson Method**
We seek the root of the Bellman residual equation:
$$F(\mathbf{v}) \equiv \mathbf{v} - \mathcal{T}^* \mathbf{v} = \mathbf{0}$$
At value estimate $\mathbf{v}_k = \mathbf{v}^{\pi_k}$, the active sub-gradient / Jacobian of $\mathcal{T}^*$ with respect to $\mathbf{v}$ corresponds to the greedy policy matrix $\gamma \mathbf{P}^{\pi_{k+1}}$:
$$D F(\mathbf{v}_k) = \mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}}$$
The classical Newton-Raphson update step is:
$$\mathbf{v}_{k+1} = \mathbf{v}_k - \left[ D F(\mathbf{v}_k) \right]^{-1} F(\mathbf{v}_k)$$
Substitute $F(\mathbf{v}_k) = \mathbf{v}_k - \mathcal{T}^{\pi_{k+1}} \mathbf{v}_k = \mathbf{v}_k - (\mathbf{r}^{\pi_{k+1}} + \gamma \mathbf{P}^{\pi_{k+1}} \mathbf{v}_k) = (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}}) \mathbf{v}_k - \mathbf{r}^{\pi_{k+1}}$:
$$\begin{aligned}
\mathbf{v}_{k+1} &= \mathbf{v}_k - (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}})^{-1} \left[ (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}}) \mathbf{v}_k - \mathbf{r}^{\pi_{k+1}} \right] \\
&= \mathbf{v}_k - \mathbf{v}_k + (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}})^{-1} \mathbf{r}^{\pi_{k+1}} \\
&= (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}})^{-1} \mathbf{r}^{\pi_{k+1}} = \mathbf{v}^{\pi_{k+1}}
\end{aligned}$$
The Newton-Raphson root update is mathematically identical to exact Policy Evaluation of the greedy policy! This explains why Policy Iteration exhibits rapid, superlinear convergence in practice. $\blacksquare$

---

#### Derivation 11.4.2: Asynchronous Dynamic Programming & Gauss-Seidel Contraction Mapping

##### Problem Statement & Goal
In synchronous (Jacobi) Value Iteration, all $n$ state values are updated in lockstep using old values $V_k$. In Asynchronous (Gauss-Seidel) Value Iteration, states are updated in-place sequentially:
$$V(s_i) \leftarrow \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s_i, a) + \gamma \sum_{j < i} \mathcal{P}(s_j \mid s_i, a) V(s_j) + \gamma \sum_{j \ge i} \mathcal{P}(s_j \mid s_i, a) V_{\text{old}}(s_j) \right]$$
We prove:
1. The Gauss-Seidel operator $\mathcal{T}_{GS}^*$ is a strict $\gamma$-contraction mapping in the $L_\infty$ norm:
   $$\|\mathcal{T}_{GS}^* U - \mathcal{T}_{GS}^* V\|_\infty \le \gamma \|U - V\|_\infty$$
2. Asynchronous dynamic programming converges to the unique optimal value function $V^*$ from any initial $V_0$, provided every state is selected for update infinitely often.

##### Explicit Assumptions
1. Finite state space $\mathcal{S} = \{s_1, \dots, s_n\}$ with fixed index ordering $1, \dots, n$.
2. Bounded reward function and discount factor $\gamma \in [0, 1)$.
3. Fairness condition: For any state $s \in \mathcal{S}$, the number of updates to $s$ diverges to infinity as total steps $t \to \infty$.

##### Underlying Intuition
Gauss-Seidel updates use fresh, recently computed values immediately within the same pass. If state $s_1$ updates, its error has already shrunk by $\gamma$. When state $s_2$ transitions to $s_1$, it uses the already-contracted value rather than the stale value, accelerating information propagation across the state graph.

##### End-to-End Mathematical Derivation

**Step 1: Inductive Bound across Ordered States**
Let $U, V \in \mathbb{R}^n$, and let $\tilde{U} = \mathcal{T}_{GS}^* U$ and $\tilde{V} = \mathcal{T}_{GS}^* V$ denote the vectors after one Gauss-Seidel sweep.
We prove by induction on state index $i \in \{1, \dots, n\}$ that:
$$|\tilde{U}(s_i) - \tilde{V}(s_i)| \le \gamma \|U - V\|_\infty, \quad \forall i \in \{1, \dots, n\}$$

**Base Case ($i = 1$):**
For the first state $s_1$, no preceding states have been updated yet. The update uses purely the old values:
$$\tilde{U}(s_1) = \max_a \left[ \mathcal{R}(s_1, a) + \gamma \sum_{j=1}^n \mathcal{P}(s_j \mid s_1, a) U(s_j) \right]$$
$$\tilde{V}(s_1) = \max_a \left[ \mathcal{R}(s_1, a) + \gamma \sum_{j=1}^n \mathcal{P}(s_j \mid s_1, a) V(s_j) \right]$$
By the non-expansion of the maximum operator and row-stochasticity (from Derivation 11.3.2):
$$|\tilde{U}(s_1) - \tilde{V}(s_1)| \le \gamma \max_a \sum_{j=1}^n \mathcal{P}(s_j \mid s_1, a) |U(s_j) - V(s_j)| \le \gamma \|U - V\|_\infty$$
The base case holds.

**Inductive Step:**
Assume the induction hypothesis holds for all preceding states $j \in \{1, \dots, i-1\}$:
$$|\tilde{U}(s_j) - \tilde{V}(s_j)| \le \gamma \|U - V\|_\infty \le \|U - V\|_\infty$$
Now consider state $s_i$. Its update formula is:
$$\tilde{U}(s_i) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s_i, a) + \gamma \sum_{j < i} \mathcal{P}(s_j \mid s_i, a) \tilde{U}(s_j) + \gamma \sum_{j \ge i} \mathcal{P}(s_j \mid s_i, a) U(s_j) \right]$$
Applying the maximum non-expansion lemma:
$$|\tilde{U}(s_i) - \tilde{V}(s_i)| \le \gamma \max_{a \in \mathcal{A}} \left[ \sum_{j < i} \mathcal{P}(s_j \mid s_i, a) |\tilde{U}(s_j) - \tilde{V}(s_j)| + \sum_{j \ge i} \mathcal{P}(s_j \mid s_i, a) |U(s_j) - V(s_j)| \right]$$
From the induction hypothesis, for $j < i$, $|\tilde{U}(s_j) - \tilde{V}(s_j)| \le \gamma \|U - V\|_\infty$.
For $j \ge i$, $|U(s_j) - V(s_j)| \le \|U - V\|_\infty$.
Therefore:
$$\begin{aligned}
|\tilde{U}(s_i) - \tilde{V}(s_i)| &\le \gamma \max_{a \in \mathcal{A}} \left[ \sum_{j < i} \mathcal{P}(s_j \mid s_i, a) \left( \gamma \|U - V\|_\infty \right) + \sum_{j \ge i} \mathcal{P}(s_j \mid s_i, a) \|U - V\|_\infty \right] \\
&\le \gamma \|U - V\|_\infty \max_{a \in \mathcal{A}} \left[ \gamma \sum_{j < i} \mathcal{P}(s_j \mid s_i, a) + \sum_{j \ge i} \mathcal{P}(s_j \mid s_i, a) \right]
\end{aligned}$$
Since $\gamma < 1$ and $\sum_{j=1}^n \mathcal{P}(s_j \mid s_i, a) = 1$:
$$\gamma \sum_{j < i} \mathcal{P}(s_j \mid s_i, a) + \sum_{j \ge i} \mathcal{P}(s_j \mid s_i, a) \le \sum_{j=1}^n \mathcal{P}(s_j \mid s_i, a) = 1$$
Hence:
$$|\tilde{U}(s_i) - \tilde{V}(s_i)| \le \gamma \|U - V\|_\infty \times 1 = \gamma \|U - V\|_\infty$$
By induction, the bound holds for all $i \in \{1, \dots, n\}$.

**Step 2: Conclusion of Contraction and Global Asynchronous Convergence**
Taking the maximum over all $i$:
$$\|\mathcal{T}_{GS}^* U - \mathcal{T}_{GS}^* V\|_\infty = \max_{1 \le i \le n} |\tilde{U}(s_i) - \tilde{V}(s_i)| \le \gamma \|U - V\|_\infty$$
Because $\mathcal{T}_{GS}^*$ is a strict $\gamma$-contraction mapping with the identical unique fixed point $V^*$, any asynchronous scheme updating each state infinitely often converges globally to $V^*$. $\blacksquare$

---

#### Derivation 11.4.3: Error Propagation Bound in Approximate Policy Iteration (API)

##### Problem Statement & Goal
In reinforcement learning with function approximation or sampling, exact policy evaluation and exact policy improvement are impossible.
Let Approximate Policy Iteration generate a sequence of policies $\{\pi_k\}$ and value approximations $\{V_k\}$ satisfying:
1. **Evaluation Error:** $\|V_k - V^{\pi_k}\|_\infty \le \epsilon$
2. **Improvement Error:** $\|\mathcal{T}^{\pi_{k+1}} V_k - \mathcal{T}^* V_k\|_\infty \le \delta$
We prove Bertsekas and Tsitsiklis's Asymptotic Error Bound:
$$\limsup_{k \to \infty} \|V^{\pi_k} - V^*\|_\infty \le \frac{2 \gamma \epsilon + \delta}{(1 - \gamma)^2}$$

##### Explicit Assumptions
1. Discrete or continuous MDP with discount factor $\gamma \in [0, 1)$.
2. Value estimates $V_k$ are uniformly bounded.
3. Errors $\epsilon, \delta \ge 0$ bound the maximum infinity-norm deviations across all iterations.

##### Underlying Intuition
Approximation errors commit a double compounding penalty: an error $\epsilon$ in evaluation distorts the Q-values, which causes the greedy choice to be sub-optimal by up to $2\gamma \epsilon + \delta$. Because sub-optimal policies compound discounted losses across an infinite horizon, dividing by $(1 - \gamma)$ converts one-step errors into return errors, and dividing by $(1 - \gamma)$ again accounts for error accumulation across policy improvement steps, yielding the quadratic multiplier $\frac{1}{(1 - \gamma)^2}$.

##### End-to-End Mathematical Derivation

**Step 1: One-Step Policy Value Difference Identity**
For any two policies $\pi$ and $\pi'$, the value difference is given by the Performance Difference Lemma:
$$V^* - V^{\pi_{k+1}} = (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}})^{-1} \left( \mathcal{T}^* V^* - \mathcal{T}^{\pi_{k+1}} V^* \right)$$
Now evaluate the difference between the optimal operator $\mathcal{T}^*$ and the policy operator $\mathcal{T}^{\pi_{k+1}}$ applied to $V_k$:
$$\mathcal{T}^* V_k - \mathcal{T}^{\pi_{k+1}} V_k \le \delta \mathbf{1}$$
by the assumption of $\delta$-greedy improvement.

**Step 2: Decomposition of Value Gap**
Examine $V^* - V_{k+1}$:
$$\begin{aligned}
V^* - V^{\pi_{k+1}} &= (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}})^{-1} \left[ \mathcal{T}^* V^* - \mathcal{T}^{\pi_{k+1}} V^{\pi_{k+1}} \right] \\
&\le (\mathbf{I} - \gamma \mathbf{P}^{\pi_{k+1}})^{-1} \left[ \mathcal{T}^* V^* - \mathcal{T}^* V_k + \mathcal{T}^* V_k - \mathcal{T}^{\pi_{k+1}} V_k + \mathcal{T}^{\pi_{k+1}} V_k - \mathcal{T}^{\pi_{k+1}} V^{\pi_{k+1}} \right]
\end{aligned}$$
Taking norms and using $\|\mathcal{T}^* V^* - \mathcal{T}^* V_k\|_\infty \le \gamma \|V^* - V_k\|_\infty$:
$$\|V^* - V^{\pi_{k+1}}\|_\infty \le \frac{1}{1 - \gamma} \left[ \gamma \|V^* - V_k\|_\infty + \delta + \gamma \|V_k - V^{\pi_{k+1}}\|_\infty \right]$$

**Step 3: Bounding Intermediate Distance $\|V^* - V_k\|_\infty$**
By the triangle inequality:
$$\|V^* - V_k\|_\infty \le \|V^* - V^{\pi_k}\|_\infty + \|V^{\pi_k} - V_k\|_\infty \le \|V^* - V^{\pi_k}\|_\infty + \epsilon$$
Similarly:
$$\|V_k - V^{\pi_k}\|_\infty \le \epsilon$$
Substituting these bounds into the recursive policy inequality:
$$\|V^* - V^{\pi_{k+1}}\|_\infty \le \gamma \|V^* - V^{\pi_k}\|_\infty + 2 \gamma \epsilon + \delta$$
or under the full resolvent operator:
$$(1 - \gamma \mathbf{P}^{\pi_{k+1}}) (V^* - V^{\pi_{k+1}}) \le \gamma \mathbf{P}^* (V^* - V^{\pi_k}) + (2 \gamma \epsilon + \delta) \mathbf{1}$$

**Step 4: Asymptotic Limit**
Applying the recurrence relation across iterations $k$:
$$\|V^* - V^{\pi_k}\|_\infty \le \gamma^k \|V^* - V^{\pi_0}\|_\infty + \sum_{j=0}^{k-1} \gamma^j \frac{2 \gamma \epsilon + \delta}{1 - \gamma}$$
Taking the limit superior as $k \to \infty$, the initial transient term $\gamma^k \|V^* - V^{\pi_0}\|_\infty \to 0$:
$$\limsup_{k \to \infty} \|V^{\pi_k} - V^*\|_\infty \le \frac{2 \gamma \epsilon + \delta}{1 - \gamma} \sum_{j=0}^\infty \gamma^j = \frac{2 \gamma \epsilon + \delta}{1 - \gamma} \left( \frac{1}{1 - \gamma} \right) = \frac{2 \gamma \epsilon + \delta}{(1 - \gamma)^2} \quad \blacksquare$$

---

## 3. Geometric & Physical Interpretation

### The Funnel Geometry of Contractions
In the vector space $\mathbb{R}^{|\mathcal{S}|}$:
1. Imagine two arbitrary points $u$ and $v$. The distance between them is $\|u - v\|_\infty$.
2. Applying $\mathcal{T}^*$ maps them to $\mathcal{T}^* u$ and $\mathcal{T}^* v$.
3. The new distance is bounded by $\gamma \|u - v\|_\infty$. Since $\gamma < 1$, the transformation **squeezes** the entire space inward like a high-dimensional funnel.
4. No matter where you drop a ball in this space, every iteration compresses it closer to the unique bottom point of the funnel: $V^*$.

```
                    THE CONTRACTION FUNNEL IN VALUE SPACE
                 u ─────────────────────────── v
                 │                             │
                 ▼                             ▼
              T*(u) ───────────────────────── T*(v)   (Distance shrunk by gamma)
                 │                             │
                 ▼                             ▼
             T*^2(u) ─────────────────────── T*^2(v)
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                               V* (Unique Fixed Point)
```

---

## 4. Real-World Analogy

### The Photocopier Scaling Loop
Imagine taking a sheet of paper with a drawing and putting it through a photocopier set to **$50\%$ reduction** ($\gamma = 0.50$), then taking the smaller copy, centering it, and reducing it by $50\%$ again:
- With every pass, every single line on the paper shrinks by half toward the exact center point.
- It does not matter what drawing you originally put on the glass (a portrait, a blank page, or random graffiti). After 20 photocopies, the image collapses to a microscopic, unique dot at the center of the page.
- That unique dot is the **Banach Fixed Point** ($V^*$), and the $50\%$ scaling factor is the **discount factor** ($\gamma$)!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace 3 full iterations of Value Iteration on a concrete 3-state MDP by hand, verifying the exact contraction ratio $\|V_{k+1} - V_k\|_\infty \le \gamma \|V_k - V_{k-1}\|_\infty$.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Dynamic Programming |
| :--- | :--- | :--- | :--- |
| $\mathcal{S}$ | State Set | 3 States | States $\{S_1, S_2, S_3\}$ |
| $\mathcal{A}$ | Action Set | 2 Actions | Choices $\{a_1, a_2\}$ available at each state |
| $\gamma$ | Discount Factor | Scalar ($0.50$) | Value contraction rate |
| $\mathbf{V}_k$ | State-Value Vector at Step $k$ | $(3,)$ | Estimated value vector at iteration $k$ |
| $Q_k(s, a)$ | Candidate Action-Values | $(3, 2)$ | $\mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V_k(s')$ |
| $\mathbf{V}_{k+1}$ | Updated Value Vector | $(3,)$ | New value vector $\max_a Q_k(s, a)$ |
| $\|\Delta_k\|_\infty$ | Step Infinity Norm Difference | Scalar | $\|V_{k+1} - V_k\|_\infty = \max_s \vert V_{k+1}(s) - V_k(s) \vert$ |
| $\pi^*(s)$ | Greedy Extracted Policy | $(3,)$ | $\operatorname{argmax}_a Q(s, a)$ |

---

### 5.2 Concrete Toy Setup

Let $|\mathcal{S}| = 3$, $|\mathcal{A}| = 2$, and $\gamma = 0.50$.
Initialization: $V_0 = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix}$.

#### Rewards $\mathcal{R}(s, a)$:
- $S_1$: $\mathcal{R}(S_1, a_1) = 4.0, \quad \mathcal{R}(S_1, a_2) = 1.0$
- $S_2$: $\mathcal{R}(S_2, a_1) = -1.0, \quad \mathcal{R}(S_2, a_2) = 2.0$
- $S_3$: $\mathcal{R}(S_3, a_1) = 0.0, \quad \mathcal{R}(S_3, a_2) = 6.0$

#### Transition Probabilities $\mathcal{P}(s' \mid s, a)$:
- From $S_1$:
  - $a_1$: transitions to $S_1$ (0.7), $S_2$ (0.3), $S_3$ (0.0)
  - $a_2$: transitions to $S_1$ (0.0), $S_2$ (0.5), $S_3$ (0.5)
- From $S_2$:
  - $a_1$: transitions to $S_1$ (0.5), $S_2$ (0.5), $S_3$ (0.0)
  - $a_2$: transitions to $S_1$ (0.0), $S_2$ (0.0), $S_3$ (1.0)
- From $S_3$:
  - $a_1$: transitions to $S_1$ (0.0), $S_2$ (0.0), $S_3$ (1.0)
  - $a_2$: transitions to $S_1$ (0.4), $S_2$ (0.4), $S_3$ (0.2)

---

### 5.3 Step 1: Iteration $k = 1$ ($V_0 \to V_1$)

Since $V_0 = [0, 0, 0]^\top$, the discounted future $\gamma \sum_{s'} \mathcal{P} V_0 = 0$.
$Q_0(s, a) = \mathcal{R}(s, a)$:
- $S_1$: $Q_0(S_1, a_1) = 4.0, \quad Q_0(S_1, a_2) = 1.0 \implies V_1(S_1) = \max(4.0, 1.0) = \mathbf{4.0000}$
- $S_2$: $Q_0(S_2, a_1) = -1.0, \quad Q_0(S_2, a_2) = 2.0 \implies V_1(S_2) = \max(-1.0, 2.0) = \mathbf{2.0000}$
- $S_3$: $Q_0(S_3, a_1) = 0.0, \quad Q_0(S_3, a_2) = 6.0 \implies V_1(S_3) = \max(0.0, 6.0) = \mathbf{6.0000}$

$$V_1 = \begin{bmatrix} 4.0000 \\ 2.0000 \\ 6.0000 \end{bmatrix}$$
Norm difference:
$$\|\Delta_0\|_\infty = \|V_1 - V_0\|_\infty = \max(|4-0|, |2-0|, |6-0|) = \mathbf{6.0000}$$

---

### 5.4 Step 2: Iteration $k = 2$ ($V_1 \to V_2$)

Now evaluate $Q_1(s, a) = \mathcal{R}(s, a) + 0.5 \sum_{s'} \mathcal{P}(s' \mid s, a) V_1(s')$ with $V_1 = [4.0, 2.0, 6.0]^\top$:

#### State $S_1$:
- Under $a_1$:
  $$Q_1(S_1, a_1) = 4.0 + 0.5 \left[ 0.7(4.0) + 0.3(2.0) + 0.0(6.0) \right] = 4.0 + 0.5(2.8 + 0.6) = 4.0 + 0.5(3.4) = 4.0 + 1.7 = \mathbf{5.7000}$$
- Under $a_2$:
  $$Q_1(S_1, a_2) = 1.0 + 0.5 \left[ 0.0(4.0) + 0.5(2.0) + 0.5(6.0) \right] = 1.0 + 0.5(1.0 + 3.0) = 1.0 + 0.5(4.0) = 1.0 + 2.0 = \mathbf{3.0000}$$
$$V_2(S_1) = \max(5.70, 3.00) = \mathbf{5.7000} \quad (\text{Choice: } a_1)$$

#### State $S_2$:
- Under $a_1$:
  $$Q_1(S_2, a_1) = -1.0 + 0.5 \left[ 0.5(4.0) + 0.5(2.0) + 0.0(6.0) \right] = -1.0 + 0.5(2.0 + 1.0) = -1.0 + 1.5 = \mathbf{0.5000}$$
- Under $a_2$:
  $$Q_1(S_2, a_2) = 2.0 + 0.5 \left[ 0.0(4.0) + 0.0(2.0) + 1.0(6.0) \right] = 2.0 + 0.5(6.0) = 2.0 + 3.0 = \mathbf{5.0000}$$
$$V_2(S_2) = \max(0.50, 5.00) = \mathbf{5.0000} \quad (\text{Choice: } a_2)$$

#### State $S_3$:
- Under $a_1$:
  $$Q_1(S_3, a_1) = 0.0 + 0.5 \left[ 1.0(6.0) \right] = \mathbf{3.0000}$$
- Under $a_2$:
  $$Q_1(S_3, a_2) = 6.0 + 0.5 \left[ 0.4(4.0) + 0.4(2.0) + 0.2(6.0) \right] = 6.0 + 0.5(1.6 + 0.8 + 1.2) = 6.0 + 0.5(3.6) = 6.0 + 1.8 = \mathbf{7.8000}$$
$$V_2(S_3) = \max(3.00, 7.80) = \mathbf{7.8000} \quad (\text{Choice: } a_2)$$

$$V_2 = \begin{bmatrix} 5.7000 \\ 5.0000 \\ 7.8000 \end{bmatrix}$$
Norm difference:
$$\|\Delta_1\|_\infty = \|V_2 - V_1\|_\infty = \max(|5.7-4.0|, |5.0-2.0|, |7.8-6.0|) = \max(1.7, 3.0, 1.8) = \mathbf{3.0000}$$

Notice the exact contraction ratio:
$$\frac{\|\Delta_1\|_\infty}{\|\Delta_0\|_\infty} = \frac{3.0000}{6.0000} = \mathbf{0.5000} \equiv \gamma$$
The distance shrunk by **exactly $\gamma = 0.5$**!

---

### 5.5 Step 3: Iteration $k = 3$ ($V_2 \to V_3$)

Now evaluate with $V_2 = [5.7, 5.0, 7.8]^\top$:

#### State $S_1$:
- Under $a_1$:
  $$Q_2(S_1, a_1) = 4.0 + 0.5 \left[ 0.7(5.7) + 0.3(5.0) \right] = 4.0 + 0.5(3.99 + 1.50) = 4.0 + 0.5(5.49) = 4.0 + 2.745 = \mathbf{6.7450}$$
- Under $a_2$:
  $$Q_2(S_1, a_2) = 1.0 + 0.5 \left[ 0.5(5.0) + 0.5(7.8) \right] = 1.0 + 0.5(2.5 + 3.9) = 1.0 + 0.5(6.4) = 1.0 + 3.2 = \mathbf{4.2000}$$
$$V_3(S_1) = \max(6.745, 4.20) = \mathbf{6.7450} \quad (\text{Choice: } a_1)$$

#### State $S_2$:
- Under $a_1$:
  $$Q_2(S_2, a_1) = -1.0 + 0.5 \left[ 0.5(5.7) + 0.5(5.0) \right] = -1.0 + 0.5(2.85 + 2.50) = -1.0 + 0.5(5.35) = -1.0 + 2.675 = \mathbf{1.6750}$$
- Under $a_2$:
  $$Q_2(S_2, a_2) = 2.0 + 0.5 \left[ 1.0(7.8) \right] = 2.0 + 3.90 = \mathbf{5.9000}$$
$$V_3(S_2) = \max(1.675, 5.90) = \mathbf{5.9000} \quad (\text{Choice: } a_2)$$

#### State $S_3$:
- Under $a_1$:
  $$Q_2(S_3, a_1) = 0.0 + 0.5(7.8) = \mathbf{3.9000}$$
- Under $a_2$:
  $$Q_2(S_3, a_2) = 6.0 + 0.5 \left[ 0.4(5.7) + 0.4(5.0) + 0.2(7.8) \right] = 6.0 + 0.5(2.28 + 2.00 + 1.56) = 6.0 + 0.5(5.84) = 6.0 + 2.92 = \mathbf{8.9200}$$
$$V_3(S_3) = \max(3.90, 8.92) = \mathbf{8.9200} \quad (\text{Choice: } a_2)$$

$$V_3 = \begin{bmatrix} 6.7450 \\ 5.9000 \\ 8.9200 \end{bmatrix}$$
Norm difference:
$$\|\Delta_2\|_\infty = \|V_3 - V_2\|_\infty = \max(|6.745 - 5.7|, |5.9 - 5.0|, |8.92 - 7.8|) = \max(1.045, 0.900, 1.120) = \mathbf{1.1200}$$

Contraction check:
$$\|\Delta_2\|_\infty = 1.1200 \le \gamma \|\Delta_1\|_\infty = 0.5(3.0000) = \mathbf{1.5000}$$
$1.1200 \le 1.5000$ strictly holds!

---

### 5.6 Summary Arithmetic State Grid

| Iteration $k$ | $V_k(S_1)$ | $V_k(S_2)$ | $V_k(S_3)$ | $\|\Delta_{k-1}\|_\infty$ | Upper Bound $\gamma \|\Delta_{k-2}\|_\infty$ | Optimal Action $\pi^*(s)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 0$** | $0.0000$ | $0.0000$ | $0.0000$ | — | — | — |
| **$k = 1$** | $4.0000$ | $2.0000$ | $6.0000$ | $6.0000$ | — | $[a_1, a_2, a_2]$ |
| **$k = 2$** | $5.7000$ | $5.0000$ | $7.8000$ | $3.0000$ | $0.5 \times 6.0 = 3.0000$ | $[a_1, a_2, a_2]$ |
| **$k = 3$** | $6.7450$ | $5.9000$ | $8.9200$ | $1.1200$ | $0.5 \times 3.0 = 1.5000$ | $[a_1, a_2, a_2]$ |

Notice that the optimal policy $\pi^* = [a_1, a_2, a_2]$ **stabilized by iteration 1**, long before the numerical values $V_k$ converged to their infinite decimal limits! This explains why Policy Iteration often finishes in fewer iterations than Value Iteration.

---

## 6. Solved Illustrations

### Illustration 1: Analytical Proof of the Contraction Bound on Error
**Problem:** In Value Iteration with discount factor $\gamma = 0.90$, we observe $\|V_6 - V_5\|_\infty = 0.02$.
1. Bound the true distance to the optimal value function $\|V_6 - V^*\|_\infty$.
2. How many additional iterations are guaranteed to achieve $\|V_{6+m} - V^*\|_\infty < 0.001$?
**Solution:**
1. Using the Banach bound:
   $$\|V_k - V^*\|_\infty \le \frac{\gamma}{1 - \gamma} \|V_k - V_{k-1}\|_\infty$$
   For $k = 6$:
   $$\|V_6 - V^*\|_\infty \le \frac{0.90}{1 - 0.90} (0.02) = \frac{0.90}{0.10} (0.02) = 9 \times 0.02 = \mathbf{0.1800}$$
2. After $m$ additional steps:
   $$\|V_{6+m} - V^*\|_\infty \le \gamma^m \|V_6 - V^*\|_\infty \le 0.90^m (0.1800) < 0.001$$
   $$0.90^m < \frac{0.001}{0.1800} \approx 0.005556$$
   Taking logarithms:
   $$m \ln(0.90) < \ln(0.005556) \implies m (-0.10536) < -5.19296 \implies m > \frac{5.19296}{0.10536} \approx \mathbf{49.28}$$
   In at most $50$ additional iterations, the value error is guaranteed to be less than $0.001$.

---

### Illustration 2: Complete End-to-End Policy Iteration Hand Trace on a 2-State MDP

**Problem:**
Consider a 2-state MDP with $\mathcal{S} = \{S_1, S_2\}$, $\mathcal{A} = \{a_1, a_2\}$, and discount factor $\gamma = 0.50$.
Transitions and immediate rewards:
- In $S_1$:
  - Action $a_1$: $\mathcal{P}(S_1 \mid S_1, a_1) = 0.60, \; \mathcal{P}(S_2 \mid S_1, a_1) = 0.40, \quad \mathcal{R}(S_1, a_1) = 2.0$
  - Action $a_2$: $\mathcal{P}(S_1 \mid S_1, a_2) = 0.20, \; \mathcal{P}(S_2 \mid S_1, a_2) = 0.80, \quad \mathcal{R}(S_1, a_2) = 4.0$
- In $S_2$:
  - Action $a_1$: $\mathcal{P}(S_1 \mid S_2, a_1) = 0.50, \; \mathcal{P}(S_2 \mid S_2, a_1) = 0.50, \quad \mathcal{R}(S_2, a_1) = 1.0$
  - Action $a_2$: $\mathcal{P}(S_1 \mid S_2, a_2) = 0.10, \; \mathcal{P}(S_2 \mid S_2, a_2) = 0.90, \quad \mathcal{R}(S_2, a_2) = -1.0$

Starting with initial policy $\pi_0 = \{S_1 \to a_1, \; S_2 \to a_1\}$:
1. Perform exact Policy Evaluation to compute $\mathbf{V}^{\pi_0}$.
2. Perform Policy Improvement to compute $Q^{\pi_0}(s, a)$ and extract greedy policy $\pi_1$.
3. Perform exact Policy Evaluation to compute $\mathbf{V}^{\pi_1}$ and verify strict monotonic improvement.
4. Perform Policy Improvement on $\mathbf{V}^{\pi_1}$ and verify convergence.

**Solution:**

**Step 1: Policy Evaluation for $\pi_0$**
Under $\pi_0 = [a_1, a_1]^\top$:
$$\mathbf{P}^{\pi_0} = \begin{bmatrix} 0.60 & 0.40 \\ 0.50 & 0.50 \end{bmatrix}, \quad \mathbf{r}^{\pi_0} = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}$$
$$\mathbf{A}_0 = \mathbf{I} - 0.50 \mathbf{P}^{\pi_0} = \begin{bmatrix} 1 - 0.30 & -0.20 \\ -0.25 & 1 - 0.25 \end{bmatrix} = \begin{bmatrix} 0.70 & -0.20 \\ -0.25 & 0.75 \end{bmatrix}$$
Determinant:
$$\det(\mathbf{A}_0) = (0.70)(0.75) - (-0.20)(-0.25) = 0.525 - 0.050 = \mathbf{0.475 = \frac{19}{40}}$$
Inverting $\mathbf{A}_0$:
$$\mathbf{A}_0^{-1} = \frac{1}{0.475} \begin{bmatrix} 0.75 & 0.20 \\ 0.25 & 0.70 \end{bmatrix} = \frac{40}{19} \begin{bmatrix} 3/4 & 1/5 \\ 1/4 & 7/10 \end{bmatrix} = \frac{1}{19} \begin{bmatrix} 30 & 8 \\ 10 & 28 \end{bmatrix}$$
Solve for $\mathbf{V}^{\pi_0} = \mathbf{A}_0^{-1} \mathbf{r}^{\pi_0}$:
$$\mathbf{V}^{\pi_0} = \frac{1}{19} \begin{bmatrix} 30(2) + 8(1) \\ 10(2) + 28(1) \end{bmatrix} = \frac{1}{19} \begin{bmatrix} 68 \\ 48 \end{bmatrix} = \mathbf{\begin{bmatrix} \frac{68}{19} \\ \frac{48}{19} \end{bmatrix} \approx \begin{bmatrix} 3.5789 \\ 2.5263 \end{bmatrix}}$$

**Step 2: Policy Improvement Step 1 ($\pi_0 \to \pi_1$)**
Compute action-values $Q^{\pi_0}(s, a) = \mathcal{R}(s, a) + 0.50 \sum_{s'} \mathcal{P}(s' \mid s, a) V^{\pi_0}(s')$:
- State $S_1$:
  - $Q^{\pi_0}(S_1, a_1) = 2.0 + 0.50 \left[ 0.60\left(\frac{68}{19}\right) + 0.40\left(\frac{48}{19}\right) \right] = 2.0 + \frac{0.50(40.8 + 19.2)}{19} = 2.0 + \frac{30}{19} = \mathbf{\frac{68}{19} \approx 3.5789}$
  - $Q^{\pi_0}(S_1, a_2) = 4.0 + 0.50 \left[ 0.20\left(\frac{68}{19}\right) + 0.80\left(\frac{48}{19}\right) \right] = 4.0 + \frac{0.50(13.6 + 38.4)}{19} = 4.0 + \frac{26}{19} = \mathbf{\frac{102}{19} \approx 5.3684}$
  - Greedy choice: $\pi_1(S_1) = \arg\max(3.5789, 5.3684) = \mathbf{a_2}$
- State $S_2$:
  - $Q^{\pi_0}(S_2, a_1) = 1.0 + 0.50 \left[ 0.50\left(\frac{68}{19}\right) + 0.50\left(\frac{48}{19}\right) \right] = 1.0 + \frac{0.50(58)}{19} = 1.0 + \frac{29}{19} = \mathbf{\frac{48}{19} \approx 2.5263}$
  - $Q^{\pi_0}(S_2, a_2) = -1.0 + 0.50 \left[ 0.10\left(\frac{68}{19}\right) + 0.90\left(\frac{48}{19}\right) \right] = -1.0 + \frac{0.50(6.8 + 43.2)}{19} = -1.0 + \frac{25}{19} = \mathbf{\frac{6}{19} \approx 0.3158}$
  - Greedy choice: $\pi_1(S_2) = \arg\max(2.5263, 0.3158) = \mathbf{a_1}$

Updated Policy: $\pi_1 = \{S_1 \to a_2, \; S_2 \to a_1\}$.

**Step 3: Policy Evaluation for $\pi_1$**
Under $\pi_1$:
$$\mathbf{P}^{\pi_1} = \begin{bmatrix} 0.20 & 0.80 \\ 0.50 & 0.50 \end{bmatrix}, \quad \mathbf{r}^{\pi_1} = \begin{bmatrix} 4.0 \\ 1.0 \end{bmatrix}$$
$$\mathbf{A}_1 = \mathbf{I} - 0.50 \mathbf{P}^{\pi_1} = \begin{bmatrix} 1 - 0.10 & -0.40 \\ -0.25 & 1 - 0.25 \end{bmatrix} = \begin{bmatrix} 0.90 & -0.40 \\ -0.25 & 0.75 \end{bmatrix}$$
Determinant:
$$\det(\mathbf{A}_1) = (0.90)(0.75) - (-0.40)(-0.25) = 0.675 - 0.100 = \mathbf{0.575 = \frac{23}{40}}$$
Inverting $\mathbf{A}_1$:
$$\mathbf{A}_1^{-1} = \frac{1}{0.575} \begin{bmatrix} 0.75 & 0.40 \\ 0.25 & 0.90 \end{bmatrix} = \frac{1}{23} \begin{bmatrix} 30 & 16 \\ 10 & 36 \end{bmatrix}$$
Solve for $\mathbf{V}^{\pi_1} = \mathbf{A}_1^{-1} \mathbf{r}^{\pi_1}$:
$$\mathbf{V}^{\pi_1} = \frac{1}{23} \begin{bmatrix} 30(4) + 16(1) \\ 10(4) + 36(1) \end{bmatrix} = \frac{1}{23} \begin{bmatrix} 136 \\ 76 \end{bmatrix} = \mathbf{\begin{bmatrix} \frac{136}{23} \\ \frac{76}{23} \end{bmatrix} \approx \begin{bmatrix} 5.9130 \\ 3.3043 \end{bmatrix}}$$
Strict improvement check:
$$V^{\pi_1}(S_1) = 5.9130 > V^{\pi_0}(S_1) = 3.5789 \quad (\Delta = +2.3341)$$
$$V^{\pi_1}(S_2) = 3.3043 > V^{\pi_0}(S_2) = 2.5263 \quad (\Delta = +0.7780)$$
Both state values increased strictly!

**Step 4: Policy Improvement Step 2 ($\pi_1 \to \pi_2$)**
Evaluate $Q^{\pi_1}(s, a)$:
- State $S_1$:
  - $Q^{\pi_1}(S_1, a_1) = 2.0 + 0.50 \left[ 0.60\left(\frac{136}{23}\right) + 0.40\left(\frac{76}{23}\right) \right] = 2.0 + \frac{0.50(81.6 + 30.4)}{23} = 2.0 + \frac{56}{23} = \mathbf{\frac{102}{23} \approx 4.4348}$
  - $Q^{\pi_1}(S_1, a_2) = 4.0 + 0.50 \left[ 0.20\left(\frac{136}{23}\right) + 0.80\left(\frac{76}{23}\right) \right] = 4.0 + \frac{0.50(27.2 + 60.8)}{23} = 4.0 + \frac{44}{23} = \mathbf{\frac{136}{23} \approx 5.9130}$
  - Greedy choice: $\arg\max(4.4348, 5.9130) = \mathbf{a_2}$
- State $S_2$:
  - $Q^{\pi_1}(S_2, a_1) = 1.0 + 0.50 \left[ 0.50\left(\frac{136}{23}\right) + 0.50\left(\frac{76}{23}\right) \right] = 1.0 + \frac{0.50(106)}{23} = 1.0 + \frac{53}{23} = \mathbf{\frac{76}{23} \approx 3.3043}$
  - $Q^{\pi_1}(S_2, a_2) = -1.0 + 0.50 \left[ 0.10\left(\frac{136}{23}\right) + 0.90\left(\frac{76}{23}\right) \right] = -1.0 + \frac{0.50(13.6 + 68.4)}{23} = -1.0 + \frac{41}{23} = \mathbf{\frac{18}{23} \approx 0.7826}$
  - Greedy choice: $\arg\max(3.3043, 0.7826) = \mathbf{a_1}$

Updated Policy: $\pi_2 = \{S_1 \to a_2, \; S_2 \to a_1\}$.
Because $\pi_2 = \pi_1$, the policy is completely invariant! Policy Iteration terminates at the global optimum in **exactly 2 iterations**.

---

### Illustration 3: Gauss-Seidel In-Place Value Iteration vs. Synchronous (Jacobi) Value Iteration

**Problem:**
For the 2-state MDP of Illustration 2 ($\gamma = 0.50$), starting with initial value estimate $\mathbf{V}_0 = [0.0, 0.0]^\top$:
1. Perform 2 sweeps of Synchronous (Jacobi) Value Iteration.
2. Perform 2 sweeps of Asynchronous Gauss-Seidel Value Iteration where states are updated in the order $(S_1, S_2)$, using the updated $V(S_1)$ immediately when evaluating $S_2$.
3. Compare the estimation error to the true optimal value $V^* = [5.9130, 3.3043]^\top$ after each sweep.

**Solution:**

**Sweep 1: Synchronous (Jacobi)**
Both states evaluate using $\mathbf{V}_0 = [0, 0]^\top$:
- $V_1^{\text{Jac}}(S_1) = \max\left( 2.0 + 0, \; 4.0 + 0 \right) = \mathbf{4.0000}$
- $V_1^{\text{Jac}}(S_2) = \max\left( 1.0 + 0, \; -1.0 + 0 \right) = \mathbf{1.0000}$
Result: $\mathbf{V}_1^{\text{Jac}} = [4.0000, 1.0000]^\top$.

**Sweep 1: Asynchronous (Gauss-Seidel)**
- Update $S_1$ first using old $V_0$:
  $$V_1^{\text{GS}}(S_1) = \max(2.0, 4.0) = \mathbf{4.0000} \quad (\text{stored immediately in-place!})$$
- Update $S_2$ using freshly updated $V(S_1) = 4.0$ and old $V_0(S_2) = 0$:
  - Under $a_1$: $1.0 + 0.50 \left[ 0.50(4.0) + 0.50(0) \right] = 1.0 + 0.50(2.0) = \mathbf{2.0000}$
  - Under $a_2$: $-1.0 + 0.50 \left[ 0.10(4.0) + 0.90(0) \right] = -1.0 + 0.20 = \mathbf{-0.8000}$
  - $V_1^{\text{GS}}(S_2) = \max(2.0, -0.8) = \mathbf{2.0000}$
Result: $\mathbf{V}_1^{\text{GS}} = [4.0000, 2.0000]^\top$.

**Sweep 2: Synchronous (Jacobi)**
Uses $\mathbf{V}_1^{\text{Jac}} = [4.0, 1.0]^\top$:
- $S_1$:
  - $a_1$: $2.0 + 0.50 [0.6(4.0) + 0.4(1.0)] = 2.0 + 0.50(2.8) = \mathbf{3.4000}$
  - $a_2$: $4.0 + 0.50 [0.2(4.0) + 0.8(1.0)] = 4.0 + 0.50(1.6) = \mathbf{4.8000}$
  - $V_2^{\text{Jac}}(S_1) = \mathbf{4.8000}$
- $S_2$:
  - $a_1$: $1.0 + 0.50 [0.5(4.0) + 0.5(1.0)] = 1.0 + 0.50(2.5) = \mathbf{2.2500}$
  - $a_2$: $-1.0 + 0.50 [0.1(4.0) + 0.9(1.0)] = -1.0 + 0.50(1.3) = \mathbf{-0.3500}$
  - $V_2^{\text{Jac}}(S_2) = \mathbf{2.2500}$
Result: $\mathbf{V}_2^{\text{Jac}} = [4.8000, 2.2500]^\top$.

**Sweep 2: Asynchronous (Gauss-Seidel)**
- $S_1$ uses current state values $\mathbf{V}_1^{\text{GS}} = [4.0, 2.0]^\top$:
  - $a_1$: $2.0 + 0.50 [0.6(4.0) + 0.4(2.0)] = 2.0 + 0.50(3.2) = \mathbf{3.6000}$
  - $a_2$: $4.0 + 0.50 [0.2(4.0) + 0.8(2.0)] = 4.0 + 0.50(2.4) = \mathbf{5.2000}$
  - $V_2^{\text{GS}}(S_1) = \mathbf{5.2000} \quad (\text{stored immediately!})$
- $S_2$ uses fresh $V(S_1) = 5.20$ and $V(S_2) = 2.0$:
  - $a_1$: $1.0 + 0.50 [0.5(5.20) + 0.5(2.0)] = 1.0 + 0.50(2.6 + 1.0) = 1.0 + 1.80 = \mathbf{2.8000}$
  - $a_2$: $-1.0 + 0.50 [0.1(5.20) + 0.9(2.0)] = -1.0 + 0.50(0.52 + 1.80) = -1.0 + 1.16 = \mathbf{0.1600}$
  - $V_2^{\text{GS}}(S_2) = \max(2.80, 0.16) = \mathbf{2.8000}$
Result: $\mathbf{V}_2^{\text{GS}} = [5.2000, 2.8000]^\top$.

**Error Comparison against True $V^* = [5.9130, 3.3043]^\top$:**
- **After Sweep 1:**
  - Jacobi Error: $\|V_1^{\text{Jac}} - V^*\|_\infty = \max(|4.0 - 5.9130|, |1.0 - 3.3043|) = \max(1.9130, 2.3043) = \mathbf{2.3043}$
  - Gauss-Seidel Error: $\|V_1^{\text{GS}} - V^*\|_\infty = \max(|4.0 - 5.9130|, |2.0 - 3.3043|) = \max(1.9130, 1.3043) = \mathbf{1.9130}$
- **After Sweep 2:**
  - Jacobi Error: $\|V_2^{\text{Jac}} - V^*\|_\infty = \max(|4.8 - 5.9130|, |2.25 - 3.3043|) = \max(1.1130, 1.0543) = \mathbf{1.1130}$
  - Gauss-Seidel Error: $\|V_2^{\text{GS}} - V^*\|_\infty = \max(|5.2 - 5.9130|, |2.8 - 3.3043|) = \max(0.7130, 0.5043) = \mathbf{0.7130}$

Gauss-Seidel reduced the error after 2 sweeps from $1.1130$ down to $0.7130$ (a $36\%$ faster error reduction), while requiring half the memory buffer size because it updates values directly in place!

---

### Illustration 4: Approximate Dynamic Programming Numerical Error Propagation Trace

**Problem:**
An RL engineer trains an actor-critic algorithm in a continuous control environment modeled with discount factor $\gamma = 0.90$.
Due to neural network approximation limits, the policy evaluation step has a uniform infinity-norm error of $\epsilon = 0.05$, and the policy improvement step has an optimization error of $\delta = 0.02$.
1. Compute the theoretical asymptotic upper bound on policy sub-optimality $\limsup_{k \to \infty} \|V^{\pi_k} - V^*\|_\infty$ using the Bertsekas-Tsitsiklis bound.
2. If the engineer re-tunes the discount factor down to $\gamma = 0.60$ with the same errors $(\epsilon = 0.05, \delta = 0.02)$, calculate the new sub-optimality bound.
3. Compute the error amplification factor $\frac{1}{(1 - \gamma)^2}$ for $\gamma \in \{0.50, 0.90, 0.99\}$ and explain why deep RL training becomes notoriously unstable at $\gamma = 0.99$.

**Solution:**

**Step 1: Asymptotic Sub-Optimality at $\gamma = 0.90$**
Using the Bertsekas & Tsitsiklis theorem (Derivation 11.4.3):
$$\text{Bound} = \frac{2 \gamma \epsilon + \delta}{(1 - \gamma)^2}$$
Substitute $\gamma = 0.90$, $\epsilon = 0.05$, and $\delta = 0.02$:
$$\text{Numerator} = 2(0.90)(0.05) + 0.02 = 0.090 + 0.020 = 0.1100$$
$$\text{Denominator} = (1 - 0.90)^2 = (0.10)^2 = 0.0100$$
$$\text{Bound}(\gamma = 0.90) = \frac{0.1100}{0.0100} = \mathbf{11.0000}$$
Even though the per-step neural network error was only $0.05$, the final policy can be sub-optimal by up to $11.00$ in cumulative return!

**Step 2: Asymptotic Sub-Optimality at $\gamma = 0.60$**
Substitute $\gamma = 0.60$, $\epsilon = 0.05$, and $\delta = 0.02$:
$$\text{Numerator} = 2(0.60)(0.05) + 0.02 = 0.060 + 0.020 = 0.0800$$
$$\text{Denominator} = (1 - 0.60)^2 = (0.40)^2 = 0.1600$$
$$\text{Bound}(\gamma = 0.60) = \frac{0.0800}{0.1600} = \mathbf{0.5000}$$
Lowering $\gamma$ from $0.90$ to $0.60$ slashed the worst-case policy sub-optimality by a factor of $22 \times$ (from $11.00 \to 0.50$)!

**Step 3: Quadratic Error Amplification Analysis**
Evaluate the multiplier $\kappa(\gamma) = \frac{1}{(1 - \gamma)^2}$:
- For $\gamma = 0.50$: $\kappa(0.50) = \frac{1}{(0.50)^2} = \frac{1}{0.25} = \mathbf{4}$
- For $\gamma = 0.90$: $\kappa(0.90) = \frac{1}{(0.10)^2} = \frac{1}{0.01} = \mathbf{100}$
- For $\gamma = 0.99$: $\kappa(0.99) = \frac{1}{(0.01)^2} = \frac{1}{0.0001} = \mathbf{10{,}000}$

At $\gamma = 0.99$, any slight neural network fitting error $\epsilon = 0.01$ is magnified by up to **$10{,}000 \times$**! This mathematical reality explains why deep RL algorithms (e.g. DQN, DDPG, SAC) require target networks, gradient clipping, replay buffers, and entropy regularization to stabilize training when high discount factors are required.

---

## 7. Deep Learning Connection & Application

### 1. Value Iteration Networks (VIN - Tamar et al., NIPS 2016 Best Paper)
Can a neural network learn to plan?
In a standard ConvNet, layers perform reactive pattern matching. A **Value Iteration Network (VIN)** implements the Bellman optimality operator as a recurrent convolutional layer:
- The transition probabilities $\mathcal{P}$ are parameterized as convolutional kernels.
- The $\max_{a}$ operator is implemented as a max-pooling channel operation:
  $$\bar{V}_{k+1} = \max_{a} \left[ \bar{R}_a + \gamma \mathbf{W}_a * \bar{V}_k \right]$$
Because the operation is differentiable, the entire MDP planner can be backpropagated through end-to-end!

### 2. MuZero & World-Model Planning (Schrittwieser et al., Nature 2020)
MuZero does not use human-designed MDP rules. It learns an internal representation $s = h(o)$, dynamics $s', r = g(s, a)$, and prediction $p, v = f(s)$. The core training target is aligned directly with the Bellman fixed point using multi-step TD targets.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Exact numerical verification of Part 5 hand calculations:
   - $V_1 = [4.0, 2.0, 6.0]^\top$
   - $V_2 = [5.7, 5.0, 7.8]^\top$
   - $V_3 = [6.745, 5.9, 8.92]^\top$
   - Contraction ratios $\|\Delta_k\|_\infty \le \gamma \|\Delta_{k-1}\|_\infty$ to $< 10^{-14}$.
2. Full Policy Iteration algorithm implementation with Policy Improvement step.
3. Full Value Iteration algorithm implementation with adaptive stopping condition.
4. Parity verification: proving Policy Iteration and Value Iteration converge to the identical optimal value vector $V^*$ and policy $\pi^*$.

See implementation in:
[`11_reinforcement_learning/code/04_dynamic_programming_and_contraction.py`](./code/04_dynamic_programming_and_contraction.py)
