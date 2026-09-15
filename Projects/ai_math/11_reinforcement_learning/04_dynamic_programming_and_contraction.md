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
| **Cost per Step** | High ($\mathcal{O}(|\mathcal{S}|^3)$ or $\mathcal{O}(m |\mathcal{S}|^2 |\mathcal{A}|)$) | Fast ($\mathcal{O}(|\mathcal{S}|^2 |\mathcal{A}|)$) |
| **Monotonicity** | Strictly monotonic $V^{\pi_{k+1}} \ge V^{\pi_k}$ | Monotonic contraction toward $V^*$ |
| **Guarantees** | Finite termination to exact $\pi^*$ | Asymptotic geometric convergence to $V^*$ |

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
| $\|\Delta_k\|_\infty$ | Step Infinity Norm Difference | Scalar | $\|V_{k+1} - V_k\|_\infty = \max_s |V_{k+1}(s) - V_k(s)|$ |
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
