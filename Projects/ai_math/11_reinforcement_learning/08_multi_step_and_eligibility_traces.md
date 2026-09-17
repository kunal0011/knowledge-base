# Module 11.8: Multi-Step Bootstrapping & Eligibility Traces: TD(λ)

---

## 1. Intuition & 101 Motivation

In the previous chapters, we encountered two polar extremes of reinforcement learning prediction:
- **TD(0) (1-step bootstrapping):** Updates values based solely on the very next reward and immediate successor state estimate: $R_{t+1} + \gamma V(S_{t+1})$. Extremely low variance, but propagates reward information backward only one step per episode.
- **Monte Carlo ($\infty$-step lookahead):** Waits for the complete episode to terminate, updating values using the full empirical return $G_t$. Completely unbiased, but suffers from high variance.

**Multi-step bootstrapping** bridges this continuum by looking ahead $n$ steps before bootstrapping off the estimated value of state $S_{t+n}$. 

Furthermore, **TD($\lambda$)** unifies all $n$-step returns simultaneously by taking an exponentially decaying geometric average of every $n$-step return. It reconciles two seemingly disparate perspectives:
1. **The Forward View (Acausal & Theoretical):** Look into the future of an episode, compute all $n$-step returns, and weight them by $(1 - \lambda)\lambda^{n-1}$.
2. **The Backward View (Causal & Mechanistic):** Look into the past using **eligibility traces** $e_t(s)$, updating all previously visited states at every single time step using the local 1-step TD error $\delta_t$.

```
   1-step TD         2-step TD         3-step TD            n-step TD            Monte Carlo
     TD(0)                                                                          TD(1)
      (s)               (s)               (s)                  (s)                   (s)
       |                 |                 |                    |                     |
      (s')              (s')              (s')                 (s')                  (s')
       |                 |                 |                    |                     |
      [V]               (s'')             (s'')                (s'')                 (s'')
                         |                 |                    |                     |
                        [V]               (s''')               ...                   ...
                                           |                    |                     |
                                          [V]                 (s^(n))             (Terminal)
                                                                |
                                                               [V]
   <----------------------------- THE TD(lambda) SPECTRUM ----------------------------->
   lambda = 0                                                                     lambda = 1
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The $n$-Step Return

Let $\tau = (S_0, R_1, S_1, R_2, \dots, S_T)$ be an episodic trajectory.
The $1$-step return is the TD(0) target:
$$G_{t:t+1} \triangleq R_{t+1} + \gamma V(S_{t+1})$$

The $2$-step return looks two steps into the future before bootstrapping:
$$G_{t:t+2} \triangleq R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$$

In general, the **$n$-step return** for any integer $n \ge 1$ is:
$$G_{t:t+n} \triangleq \sum_{k=1}^n \gamma^{k-1} R_{t+k} + \gamma^n V_{t+n-1}(S_{t+n})$$
If $t + n \ge T$ (the horizon extends past episode termination), all missing terms are substituted by zero, and the return truncates at the terminal state:
$$G_{t:t+n} = G_t \quad \text{if } t + n \ge T$$

#### The $n$-Step Error Reduction Property
Let $V$ be any arbitrary value function approximation. The expected error of the $n$-step return satisfies:
$$\left\| \mathbb{E}_\pi [G_{t:t+n} \mid S_t = \cdot] - V^\pi \right\|_\infty \le \gamma^n \left\| V - V^\pi \right\|_\infty$$
Because $\gamma < 1$, as $n \to \infty$, the bootstrapping bias vanishes at an exponential rate of $\gamma^n$.

---

### 2.2 The Forward View of TD($\lambda$): The $\lambda$-Return

Rather than arbitrarily choosing a single integer $n$, the **$\lambda$-return** $G_t^\lambda$ combines all $n$-step returns for $n \in \{1, 2, \dots, \infty\}$ into a single composite target weighted geometrically by powers of $\lambda \in [0, 1]$:

$$G_t^\lambda \triangleq (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n}$$

Notice that the sum of weights forms a convergent geometric series summing to exactly $1$:
$$(1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} = (1 - \lambda) \frac{1}{1 - \lambda} = 1$$

For a finite terminating episode of length $T$, all returns for $n \ge T - t$ equal the full Monte Carlo return $G_t$. The finite-horizon $\lambda$-return is therefore:
$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} G_{t:t+n} + \lambda^{T - t - 1} G_t$$

- When $\lambda = 0$: $G_t^0 = G_{t:t+1} = R_{t+1} + \gamma V(S_{t+1})$ (**TD(0)**).
- When $\lambda = 1$: $G_t^1 = G_t$ (**Monte Carlo**).

The **Forward-View TD($\lambda$) Update** is:
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t^\lambda - V(S_t) \right]$$
*Limitation:* The forward view is **acausal**; to compute $G_t^\lambda$, an agent must wait until time $T$ to observe all future transitions.

---

### 2.3 The Backward View of TD($\lambda$): Eligibility Traces

The backward view provides a fully **causal, incremental, step-by-step mechanism** that achieves the exact same result without looking into the future.

#### The Eligibility Trace Vector
For every state $s \in \mathcal{S}$, the agent maintains a continuous memory variable $e_t(s) \in \mathbb{R}_{\ge 0}$ called the **eligibility trace**.

1. **Accumulating Trace:**
   $$e_t(s) \triangleq \begin{cases} \gamma \lambda e_{t-1}(s) + 1 & \text{if } s = S_t \\ \gamma \lambda e_{t-1}(s) & \text{if } s \neq S_t \end{cases}$$
   In vector notation:
   $$\mathbf{e}_t = \gamma \lambda \mathbf{e}_{t-1} + \mathbf{x}_t$$
   where $\mathbf{x}_t(s) = \mathbb{I}(s = S_t)$ is the one-hot state indicator.

2. **Replacing Trace (Alternative Variant):**
   $$e_t(s) \triangleq \begin{cases} 1 & \text{if } s = S_t \\ \gamma \lambda e_{t-1}(s) & \text{if } s \neq S_t \end{cases}$$

#### The Online TD($\lambda$) Backward Algorithm:
At each time step $t$, upon observing $(S_t, A_t, R_{t+1}, S_{t+1})$:
1. Compute the local 1-step TD error:
   $$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
2. Update the eligibility trace for the current state:
   $$e_t(S_t) \leftarrow e_t(S_t) + 1$$
3. Update the value function for **all states in the state space simultaneously**:
   $$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \forall s \in \mathcal{S}$$
4. Decay the eligibility trace vector by factor $\gamma \lambda$:
   $$e_t(s) \leftarrow \gamma \lambda e_t(s) \quad \forall s \in \mathcal{S}$$

---

### 2.4 The Fundamental Equivalence Theorem

#### Theorem: Offline Equivalence of Forward and Backward TD($\lambda$) (Sutton, 1988)
Suppose the value function estimates $V(s)$ are held constant during the episode and updated in an offline batch at the end of the episode.
Then, for any trajectory $\tau = (S_0, R_1, S_1, \dots, S_T)$ and any $\lambda \in [0, 1]$, the total accumulated weight update to state $s$ under the Backward View is **identically equal** to the total weight update under the Forward View:

$$\sum_{t=0}^{T-1} \alpha \delta_t e_t(s) = \sum_{t=0}^{T-1} \alpha \left[ G_t^\lambda - V(S_t) \right] \mathbb{I}(S_t = s) \quad \forall s \in \mathcal{S}$$

#### Algebraic Proof
Let us expand the forward $\lambda$-error $G_t^\lambda - V(S_t)$ as a telescoping sum of 1-step TD errors $\delta_k$.
First, observe the algebraic identity relating $n$-step returns and 1-step errors:
$$G_{t:t+n} - V(S_t) = \sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k$$
Substituting this into the definition of $G_t^\lambda - V(S_t)$:
$$G_t^\lambda - V(S_t) = (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} \left( \sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k \right) + \lambda^{T-t-1} \left( \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k \right)$$
Interchanging the orders of summation:
$$G_t^\lambda - V(S_t) = \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k$$
Now sum over all time steps where $S_t = s$:
$$\sum_{t=0}^{T-1} \mathbb{I}(S_t = s) \left[ G_t^\lambda - V(S_t) \right] = \sum_{t=0}^{T-1} \mathbb{I}(S_t = s) \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k$$
Interchanging the time indices $t$ and $k$:
$$= \sum_{k=0}^{T-1} \delta_k \left[ \sum_{t=0}^k (\gamma \lambda)^{k-t} \mathbb{I}(S_t = s) \right]$$
Recognizing that the inner bracket is precisely the recursive definition of the accumulating eligibility trace $e_k(s)$:
$$e_k(s) = \sum_{t=0}^k (\gamma \lambda)^{k-t} \mathbb{I}(S_t = s)$$
Therefore:
$$\sum_{t=0}^{T-1} \left[ G_t^\lambda - V(S_t) \right] \mathbb{I}(S_t = s) = \sum_{k=0}^{T-1} \delta_k e_k(s) \quad \blacksquare$$

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.8.1: Complete First-Principles Proof of Forward-Backward TD($\lambda$) Equivalence

```
====================================================================================================
DERIVATION 11.8.1: Equivalence of Forward-View and Backward-View TD(λ)
====================================================================================================
Problem Statement:
Prove that for any finite episodic Markov decision process trajectory τ = (S_0, R_1, S_1, ..., S_T)
with constant value function V held fixed across the trajectory, the total sum of online backward
eligibility trace updates equals the total sum of forward λ-return updates:
    ∑_{t=0}^{T-1} δ_t e_t(s) = ∑_{t=0}^{T-1} [G_t^λ - V(S_t)] 𝕀(S_t = s)   ∀ s ∈ 𝒮
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Stationary Baseline Values:** The value estimates $V(s)$ are fixed during the trajectory evaluation; updates are applied in batch at the end of the episode (offline TD($\lambda$)).
2. **Episodic Horizon:** The episode terminates at step $T < \infty$, with $V(S_T) \triangleq 0$.
3. **Trace Initialization:** Eligibility traces are zero before the start of the trajectory: $e_{-1}(s) = 0$ for all $s \in \mathcal{S}$.
4. **Decay Parameters:** Discount factor $\gamma \in [0, 1)$ and trace decay parameter $\lambda \in [0, 1]$.

**2. Underlying Intuition:**
The forward view looks into the future from time $t$, forming geometric combinations of all future $n$-step returns. The backward view looks into the past from transition $k$, using an eligibility trace $e_k(s)$ to record how recently and frequently state $s$ occurred. Because both systems are linear combinations of the exact same elementary 1-step TD errors $\delta_k$, rearranging the double summation over time indices $(t, k)$ algebraically transforms the acausal forward perspective into the causal backward perspective.

**3. End-to-End Algebraic Derivation:**

*Step 1: Express the $n$-step error as a telescoping sum of 1-step TD errors.*
Recall the definitions:
$$\delta_k \triangleq R_{k+1} + \gamma V(S_{k+1}) - V(S_k)$$
$$G_{t:t+n} \triangleq \sum_{i=1}^n \gamma^{i-1} R_{t+i} + \gamma^n V(S_{t+n})$$
Subtract $V(S_t)$ from $G_{t:t+n}$:
$$G_{t:t+n} - V(S_t) = \sum_{i=1}^n \gamma^{i-1} R_{t+i} + \gamma^n V(S_{t+n}) - V(S_t)$$
Now consider the discounted sum of 1-step TD errors from step $t$ to $t+n-1$:
$$\sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k = \sum_{k=t}^{t+n-1} \gamma^{k-t} \left[ R_{k+1} + \gamma V(S_{k+1}) - V(S_k) \right]$$
Expanding the terms:
$$= \sum_{k=t}^{t+n-1} \gamma^{k-t} R_{k+1} + \sum_{k=t}^{t+n-1} \gamma^{k-t+1} V(S_{k+1}) - \sum_{k=t}^{t+n-1} \gamma^{k-t} V(S_k)$$
Notice the telescoping cancellation between the second and third sums:
$$\sum_{k=t}^{t+n-1} \gamma^{k-t+1} V(S_{k+1}) - \sum_{k=t}^{t+n-1} \gamma^{k-t} V(S_k) = \gamma^n V(S_{t+n}) - V(S_t)$$
Therefore, the $n$-step error is identically equal to the sum of discounted 1-step errors:
$$G_{t:t+n} - V(S_t) = \sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k \quad \text{(Identity 1)}$$

*Step 2: Substitute Identity 1 into the definition of the forward $\lambda$-return error.*
For a finite episode of length $T$, the $\lambda$-return error at time $t$ is:
$$G_t^\lambda - V(S_t) = (1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} \left( G_{t:t+n} - V(S_t) \right) + \lambda^{T - t - 1} \left( G_t - V(S_t) \right)$$
Substituting Identity 1 into each term:
$$G_t^\lambda - V(S_t) = (1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} \sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k + \lambda^{T - t - 1} \sum_{k=t}^{T - 1} \gamma^{k-t} \delta_k$$

*Step 3: Interchanging summations over $n$ and $k$.*
Examine the coefficient of each 1-step TD error $\delta_k$ for a fixed $k \in \{t, t+1, \dots, T-1\}$ in the first double sum:
$$\sum_{n=1}^{T - t - 1} \lambda^{n-1} \sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k = \sum_{k=t}^{T-2} \gamma^{k-t} \delta_k \left( \sum_{n=k-t+1}^{T-t-1} \lambda^{n-1} \right)$$
Evaluate the inner geometric sum for a given index $m = k - t + 1$:
$$\sum_{n=m}^{T-t-1} \lambda^{n-1} = \lambda^{m-1} \sum_{j=0}^{T-t-1-m} \lambda^j = \lambda^{k-t} \frac{1 - \lambda^{T-t-1-(k-t)}}{1 - \lambda} = \frac{\lambda^{k-t} - \lambda^{T-t-1}}{1 - \lambda}$$
Multiplying by $(1 - \lambda)$, the factor $(1 - \lambda)$ cancels exactly:
$$(1 - \lambda) \sum_{n=k-t+1}^{T-t-1} \lambda^{n-1} = \lambda^{k-t} - \lambda^{T-t-1}$$
Now, adding the second term $\lambda^{T-t-1} \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k$:
For every $k \in \{t, \dots, T-2\}$, the $-\lambda^{T-t-1}$ cancels with $+\lambda^{T-t-1}$.
For $k = T-1$, the first sum has no term, and the second term has weight $\lambda^{T-t-1} = \lambda^{(T-1)-t}$.
Hence, every single $\delta_k$ carries the combined factor:
$$(\lambda)^{k-t} \gamma^{k-t} = (\gamma \lambda)^{k-t}$$
We conclude:
$$G_t^\lambda - V(S_t) = \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k \quad \text{(Identity 2)}$$

*Step 4: Compute the total trajectory update for an arbitrary state $s$.*
Summing the forward updates over all time steps $t \in \{0, \dots, T-1\}$ where $S_t = s$:
$$\sum_{t=0}^{T-1} \mathbb{I}(S_t = s) \left[ G_t^\lambda - V(S_t) \right] = \sum_{t=0}^{T-1} \mathbb{I}(S_t = s) \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k$$
Because $k \ge t$, we can rewrite the double summation domain as $0 \le t \le k \le T-1$:
$$= \sum_{k=0}^{T-1} \sum_{t=0}^k \mathbb{I}(S_t = s) (\gamma \lambda)^{k-t} \delta_k = \sum_{k=0}^{T-1} \delta_k \left[ \sum_{t=0}^k (\gamma \lambda)^{k-t} \mathbb{I}(S_t = s) \right]$$

*Step 5: Identify the accumulating eligibility trace.*
The accumulating eligibility trace is defined recursively by $e_{-1}(s) = 0$ and:
$$e_k(s) = \gamma \lambda e_{k-1}(s) + \mathbb{I}(S_k = s)$$
Unrolling this linear recurrence from $k$ down to $0$:
$$e_k(s) = \mathbb{I}(S_k = s) + \gamma \lambda \mathbb{I}(S_{k-1} = s) + (\gamma \lambda)^2 \mathbb{I}(S_{k-2} = s) + \dots + (\gamma \lambda)^k \mathbb{I}(S_0 = s) = \sum_{t=0}^k (\gamma \lambda)^{k-t} \mathbb{I}(S_t = s)$$
Substituting $e_k(s)$ directly into the bracketed term in Step 4:
$$\sum_{t=0}^{T-1} \mathbb{I}(S_t = s) \left[ G_t^\lambda - V(S_t) \right] = \sum_{k=0}^{T-1} \delta_k e_k(s) \quad \blacksquare$$

---

#### Derivation 11.8.2: The $\lambda$-Return Bellman Operator as a Strict Contraction in $L_\infty$ Norm

```
====================================================================================================
DERIVATION 11.8.2: Contraction Mapping of the λ-Return Operator
====================================================================================================
Problem Statement:
Define the continuous-horizon Bellman λ-operator for policy π on value functions V ∈ ℝ^{|𝒮|} by:
    𝒯^λ V ≜ (1 - λ) ∑_{n=1}^∞ λ^{n-1} 𝒯^n V
where 𝒯 is the standard Bellman expectation operator 𝒯 V = R^π + γ P^π V.
Prove that 𝒯^λ is a strict contraction mapping in the L_∞ norm with contraction modulus:
    γ_λ = γ (1 - λ) / (1 - γ λ) < 1
and that its unique fixed point is the true policy value function V^π.
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Discounting:** $0 \le \gamma < 1$.
2. **Trace Parameter:** $0 \le \lambda < 1$. (For $\lambda = 1$, $\mathcal{T}^1 V = V^\pi$ identically for all $V$).
3. **Metric Space:** The state space $\mathcal{S}$ is finite, and $\mathbb{R}^{|\mathcal{S}|}$ is equipped with the $L_\infty$ norm $\|V\|_\infty \triangleq \max_{s \in \mathcal{S}} |V(s)|$.

**2. Underlying Intuition:**
The standard Bellman expectation operator $\mathcal{T}$ contracts by $\gamma$. Iterating it $n$ times yields an operator $\mathcal{T}^n$ that contracts by $\gamma^n$. The $\lambda$-operator is a convex combination of all these $n$-step operators. Since every constituent operator contracts at rate $\gamma^n \le \gamma$, their weighted average must also contract, and the modulus is obtained by evaluating the geometric sum.

**3. End-to-End Algebraic Derivation:**

*Step 1: Contraction of the $n$-step Bellman operator.*
For any two value functions $U, V \in \mathbb{R}^{|\mathcal{S}|}$ and any $n \ge 1$:
$$\mathcal{T}^n U - \mathcal{T}^n V = (\gamma P^\pi)^n (U - V)$$
Taking the $L_\infty$ norm:
$$\|\mathcal{T}^n U - \mathcal{T}^n V\|_\infty = \|(\gamma P^\pi)^n (U - V)\|_\infty \le \gamma^n \|(P^\pi)^n\|_\infty \|U - V\|_\infty$$
Since $P^\pi$ is a row-stochastic transition matrix, $\|P^\pi\|_\infty = 1$, so $\|(P^\pi)^n\|_\infty \le 1$. Thus:
$$\|\mathcal{T}^n U - \mathcal{T}^n V\|_\infty \le \gamma^n \|U - V\|_\infty \quad \forall n \ge 1$$

*Step 2: Apply the triangle inequality to the $\lambda$-operator difference.*
Subtracting $\mathcal{T}^\lambda V$ from $\mathcal{T}^\lambda U$:
$$\mathcal{T}^\lambda U - \mathcal{T}^\lambda V = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \left( \mathcal{T}^n U - \mathcal{T}^n V \right)$$
Taking the $L_\infty$ norm and applying the triangle inequality:
$$\|\mathcal{T}^\lambda U - \mathcal{T}^\lambda V\|_\infty \le (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \|\mathcal{T}^n U - \mathcal{T}^n V\|_\infty$$
Substituting the $n$-step contraction bound:
$$\|\mathcal{T}^\lambda U - \mathcal{T}^\lambda V\|_\infty \le (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \gamma^n \|U - V\|_\infty = \left[ (1 - \lambda) \gamma \sum_{n=1}^\infty (\gamma \lambda)^{n-1} \right] \|U - V\|_\infty$$

*Step 3: Evaluate the infinite geometric series.*
Since $\gamma \in [0, 1)$ and $\lambda \in [0, 1)$, we have $0 \le \gamma \lambda < 1$. The geometric series converges:
$$\sum_{n=1}^\infty (\gamma \lambda)^{n-1} = \sum_{k=0}^\infty (\gamma \lambda)^k = \frac{1}{1 - \gamma \lambda}$$
Therefore, the contraction modulus $\gamma_\lambda$ is:
$$\gamma_\lambda \triangleq \frac{\gamma (1 - \lambda)}{1 - \gamma \lambda}$$

*Step 4: Prove that $\gamma_\lambda < 1$ for all $\gamma \in [0, 1)$ and $\lambda \in [0, 1)$.*
Consider the difference $1 - \gamma_\lambda$:
$$1 - \gamma_\lambda = 1 - \frac{\gamma - \gamma \lambda}{1 - \gamma \lambda} = \frac{(1 - \gamma \lambda) - (\gamma - \gamma \lambda)}{1 - \gamma \lambda} = \frac{1 - \gamma}{1 - \gamma \lambda}$$
Because $\gamma < 1$, the numerator $1 - \gamma > 0$. Because $\gamma \lambda < 1$, the denominator $1 - \gamma \lambda > 0$.
Hence $1 - \gamma_\lambda > 0 \implies \gamma_\lambda < 1$.
Moreover, notice that:
$$\gamma_\lambda = \gamma \left( \frac{1 - \lambda}{1 - \gamma \lambda} \right) \le \gamma \quad (\text{since } 1 - \lambda \le 1 - \gamma \lambda \text{ because } \gamma < 1)$$
Thus, $\mathcal{T}^\lambda$ contracts strictly faster than $\mathcal{T}$ for any $\lambda > 0$!

*Step 5: Characterize the unique fixed point.*
Evaluate $\mathcal{T}^\lambda V^\pi$:
$$\mathcal{T}^\lambda V^\pi = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \mathcal{T}^n V^\pi$$
Since $V^\pi$ is the fixed point of $\mathcal{T}$, $\mathcal{T} V^\pi = V^\pi \implies \mathcal{T}^n V^\pi = V^\pi$ for all $n \ge 1$.
$$\mathcal{T}^\lambda V^\pi = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} V^\pi = \left[ (1 - \lambda) \frac{1}{1 - \lambda} \right] V^\pi = V^\pi$$
By the Banach Fixed-Point Theorem, since $(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)$ is a complete Banach space and $\mathcal{T}^\lambda$ is a strict contraction with modulus $\gamma_\lambda < 1$, $V^\pi$ is the **unique fixed point** of $\mathcal{T}^\lambda$. $\blacksquare$

---

#### Derivation 11.8.3: Mathematical Derivation of Dutch Traces and Exact Online Equivalence via True Online TD($\lambda$)

```
====================================================================================================
DERIVATION 11.8.3: Dutch Traces and True Online TD(λ)
====================================================================================================
Problem Statement:
In standard online TD(λ), weights w_t change at each step, causing the causal backward update
to drift away from the acausal forward λ-return target. Derive the Dutch eligibility trace vector:
    z_t = γ λ z_{t-1} + (1 - α γ λ z_{t-1}^⊤ x_t) x_t
and the True Online TD(λ) weight update rule that achieves exact step-by-step equivalence with
the online forward view.
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Linear Function Approximation:** Value function is parameterized as $\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$, where $\mathbf{x}(s) \in \mathbb{R}^d$ is the feature vector (or one-hot vector in tabular RL).
2. **Online Forward Target:** At time step $t$, the forward target utilizes the weights $\mathbf{w}_{t-1}$ at the start of the transition:
   $$\hat{G}_t^{\lambda} \triangleq (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \hat{G}_{t:t+n}$$
3. **Step Size:** Learning rate $\alpha \in (0, 1]$.

**2. Underlying Intuition:**
In classical online TD($\lambda$), when state $s$ is updated at step $t$, its value immediately increases to $V_t(s)$. When step $t+1$ evaluates $\delta_{t+1}$, the error uses this updated value, effectively "double counting" the effect of recent trace updates. To eliminate this distortion, the eligibility trace must discount the portion of the current feature $\mathbf{x}_t$ that has already been accounted for by prior trace-mediated weight changes. This correction term is precisely $-\alpha \gamma \lambda (\mathbf{z}_{t-1}^\top \mathbf{x}_t)\mathbf{x}_t$, yielding the **Dutch trace**.

**3. End-to-End Algebraic Derivation:**

*Step 1: The Online Forward-View Objective.*
In the online setting, we want the sequence of weights $\mathbf{w}_t$ to satisfy:
$$\mathbf{w}_t = \mathbf{w}_0 + \sum_{i=0}^t \alpha \left[ \hat{G}_i^{\lambda \mid t} - \mathbf{w}_{i-1}^\top \mathbf{x}_i \right] \mathbf{x}_i$$
where $\hat{G}_i^{\lambda \mid t}$ is the truncated forward $\lambda$-return constructed from data available up to horizon $t$:
$$\hat{G}_i^{\lambda \mid t} \triangleq (1 - \lambda) \sum_{n=1}^{t - i} \lambda^{n-1} G_{i:i+n} + \lambda^{t-i} G_{i:t}$$

*Step 2: Incremental difference in the forward weight sequence.*
Define the total forward weight at horizon $t$ as $\mathbf{w}_t^{\text{true}}$. The difference $\mathbf{w}_t^{\text{true}} - \mathbf{w}_{t-1}^{\text{true}}$ is caused by the arrival of the new transition $(S_t, R_{t+1}, S_{t+1})$:
$$\Delta \mathbf{w}_t^{\text{true}} = \mathbf{w}_t^{\text{true}} - \mathbf{w}_{t-1}^{\text{true}} = \alpha \sum_{i=0}^t \left[ \hat{G}_i^{\lambda \mid t+1} - \hat{G}_i^{\lambda \mid t} \right] \mathbf{x}_i$$
Observe that for any $i \le t$, the difference in truncated returns simplifies to:
$$\hat{G}_i^{\lambda \mid t+1} - \hat{G}_i^{\lambda \mid t} = (\gamma \lambda)^{t-i} \left[ R_{t+1} + \gamma \mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t \right] + (\gamma \lambda)^{t-i} (\mathbf{w}_t^\top \mathbf{x}_t - \mathbf{w}_{t-1}^\top \mathbf{x}_t)$$
Define the TD error:
$$\delta_t \triangleq R_{t+1} + \gamma \mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_{t-1}^\top \mathbf{x}_t$$
Then:
$$\hat{G}_i^{\lambda \mid t+1} - \hat{G}_i^{\lambda \mid t} = (\gamma \lambda)^{t-i} \left( \delta_t + \mathbf{w}_{t-1}^\top \mathbf{x}_t - \mathbf{w}_t^\top \mathbf{x}_t \right)$$

*Step 3: Deriving the Dutch Trace Recurrence.*
Factoring out the shared terms from the summation over $i$:
$$\Delta \mathbf{w}_t^{\text{true}} = \alpha \left( \delta_t + \mathbf{w}_{t-1}^\top \mathbf{x}_t - V_{\text{old}} \right) \mathbf{z}_t - \alpha (\mathbf{w}_{t-1}^\top \mathbf{x}_t - V_{\text{old}}) \mathbf{x}_t$$
where $V_{\text{old}} \triangleq \mathbf{w}_{t-2}^\top \mathbf{x}_{t-1}$ stores the previous value prediction, and the trace vector $\mathbf{z}_t$ accumulates features with decay $\gamma \lambda$:
$$\mathbf{z}_t = \sum_{i=0}^t (\gamma \lambda)^{t-i} \mathbf{x}_i - \alpha \gamma \lambda \sum_{i=0}^{t-1} (\gamma \lambda)^{t-1-i} (\mathbf{z}_{i}^\top \mathbf{x}_{i+1}) \mathbf{x}_{i+1}$$
This simplifies into the closed-form single-step **Dutch trace update**:
$$\mathbf{z}_t = \gamma \lambda \mathbf{z}_{t-1} + \left( 1 - \alpha \gamma \lambda \mathbf{z}_{t-1}^\top \mathbf{x}_t \right) \mathbf{x}_t$$

*Step 4: The Complete True Online TD($\lambda$) Algorithm.*
At each step $t$:
1. Compute TD error: $\delta_t = R_{t+1} + \gamma \mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_{t}^\top \mathbf{x}_t$
2. Update Dutch trace: $\mathbf{z}_t \leftarrow \gamma \lambda \mathbf{z}_{t-1} + (1 - \alpha \gamma \lambda \mathbf{z}_{t-1}^\top \mathbf{x}_t) \mathbf{x}_t$
3. Update weights:
   $$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t + \alpha \left( \delta_t + \mathbf{w}_t^\top \mathbf{x}_t - V_{\text{old}} \right) \mathbf{z}_t - \alpha \left( \mathbf{w}_t^\top \mathbf{x}_t - V_{\text{old}} \right) \mathbf{x}_t$$
4. Store $V_{\text{old}} \leftarrow \mathbf{w}_{t+1}^\top \mathbf{x}_{t+1}$.
This guarantees that $\mathbf{w}_T = \mathbf{w}_T^{\text{true}}$ exactly at every single step, uniting the online backward view with the true online forward view without requiring episode termination. $\blacksquare$

---

## 3. Geometric & Physical Interpretation: The Decaying Capacitor

Think of the eligibility trace $e(s)$ as the **electrostatic charge stored in a leaky capacitor**:
- Every time state $s$ is visited, a voltage pulse adds $+1$ charge to capacitor $e(s)$.
- At each subsequent time step, charge leaks away exponentially through a resistor at decay rate $\gamma \lambda$.
- When a TD error lightning strike $\delta_t$ occurs (e.g., encountering a large unexpected reward), the current distributed to update state $s$ is proportional to its remaining stored charge: $\Delta V(s) = \alpha \delta_t e_t(s)$.

```
   Charge e(s)
     ^
  2.0|              * (Second visit at t=2)
     |             / \
  1.0|   *        /   \
     |  / \      /     \_______
  0.0+---+---+---+---+---+---+---> Time t
         0   1   2   3   4   5
```

---

## 4. Real-World Analogy: The Restaurant Sickness Problem

Imagine you spend an evening visiting three establishments:
1. At 6:00 PM, you have an appetizer at a Seafood Bar ($S_1$).
2. At 7:30 PM, you eat dinner at a Steakhouse ($S_2$).
3. At 9:00 PM, you return to the Seafood Bar for dessert ($S_1$).
4. At 11:00 PM, you arrive home and suddenly become violently ill ($\delta_t = -100$).

How should you assign blame (credit assignment)?
- **TD(0):** Only blames the action immediately preceding the illness (drinking water at home). The Seafood Bar gets zero blame!
- **Monte Carlo:** Blames every establishment equally, regardless of timing or frequency.
- **TD($\lambda$) with Eligibility Traces:**
  - The Steakhouse ($S_2$) was visited once ($e(S_2) = \gamma \lambda$).
  - The Seafood Bar ($S_1$) was visited twice ($e(S_1) = (\gamma \lambda)^2 + 1$).
  The illness penalty is apportioned proportionally according to both **frequency** and **recency**!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 2-Step Episodic Walk
Let us trace a 2-step episode through a 2-state MDP:
- Non-terminal states: $S_1$ and $S_2$.
- Terminal state: $\text{End}$.
- Discount factor: $\gamma = 0.9000$
- Trace-decay parameter: $\lambda = 0.5000 \implies \gamma \lambda = 0.9000 \times 0.5000 = \mathbf{0.4500}$
- Learning rate: $\alpha = 0.5000$
- Initial values: $V_0(S_1) = 1.0000, \quad V_0(S_2) = 2.0000$

**Trajectory $\tau$:**
$$\text{Time } 0: S_0 = S_1 \xrightarrow{R_1 = 2.0} \text{Time } 1: S_1 = S_2 \xrightarrow{R_2 = 10.0} \text{Time } 2: S_2 = \text{End}$$

We will compute both the **Forward View $\lambda$-returns** and the **Backward View Eligibility Traces** to prove their exact numerical equivalence!

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Value / Formula |
| :--- | :--- | :--- |
| $\gamma$ | Discount Factor | $0.9000$ |
| $\lambda$ | Trace-Decay Parameter | $0.5000$ |
| $\gamma \lambda$ | Effective Decay Rate | $0.9000 \times 0.5000 = 0.4500$ |
| $\delta_t$ | 1-Step TD Error | $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ |
| $e_t(s)$ | Eligibility Trace for state $s$ | $e_t = \gamma \lambda e_{t-1} + \mathbb{I}(S_t = s)$ |
| $G_{t:t+n}$ | $n$-step Return | Truncated return bootstrapped at step $t+n$ |
| $G_t^\lambda$ | Forward $\lambda$-Return | $(1 - \lambda)\sum \lambda^{n-1} G_{t:t+n}$ |

---

### 5.3 Forward View Hand Calculation

#### At $t = 0$ (State $S_1$):
We have two possible $n$-step returns before termination ($T = 2$):
1. **$1$-step return ($n = 1$):**
   $$G_{0:1} = R_1 + \gamma V_0(S_2) = 2.0000 + 0.9000 \times 2.0000 = 2.0000 + 1.8000 = \mathbf{3.8000}$$
2. **$2$-step full return ($n = 2$):**
   $$G_{0:2} = R_1 + \gamma R_2 + \gamma^2 V_0(\text{End}) = 2.0000 + 0.9000 \times 10.0000 + 0 = 2.0000 + 9.0000 = \mathbf{11.0000}$$

Now compute the composite $\lambda$-return $G_0^\lambda$:
$$G_0^\lambda = (1 - \lambda) G_{0:1} + \lambda G_{0:2} = (1 - 0.5000) \times 3.8000 + 0.5000 \times 11.0000$$
$$= 0.5000 \times 3.8000 + 0.5000 \times 11.0000 = 1.9000 + 5.5000 = \mathbf{7.4000}$$

Forward update for $S_1$:
$$\Delta V_{\text{forward}}(S_1) = \alpha \left[ G_0^\lambda - V_0(S_1) \right] = 0.5000 \times (7.4000 - 1.0000) = 0.5000 \times 6.4000 = \mathbf{3.2000}$$

#### At $t = 1$ (State $S_2$):
Only 1 step remains until termination:
$$G_1^\lambda = G_{1:2} = R_2 + \gamma V_0(\text{End}) = 10.0000 + 0.0000 = \mathbf{10.0000}$$

Forward update for $S_2$:
$$\Delta V_{\text{forward}}(S_2) = \alpha \left[ G_1^\lambda - V_0(S_2) \right] = 0.5000 \times (10.0000 - 2.0000) = 0.5000 \times 8.0000 = \mathbf{4.0000}$$

---

### 5.4 Backward View Hand Calculation (Offline Batch)

Initial traces: $e_{-1}(S_1) = 0.0000, \quad e_{-1}(S_2) = 0.0000$

#### Step $t = 0$: Transition $S_1 \to S_2$ with $R_1 = 2.0$
1. Update Traces:
   $$e_0(S_1) = \gamma \lambda e_{-1}(S_1) + 1 = 0 + 1 = \mathbf{1.0000}$$
   $$e_0(S_2) = \gamma \lambda e_{-1}(S_2) + 0 = \mathbf{0.0000}$$
2. Compute 1-step TD error $\delta_0$:
   $$\delta_0 = R_1 + \gamma V_0(S_2) - V_0(S_1) = 2.0000 + 0.9000 \times 2.0000 - 1.0000$$
   $$= 2.0000 + 1.8000 - 1.0000 = \mathbf{2.8000}$$
3. Trace-weighted error contribution at $t = 0$:
   - For $S_1$: $\delta_0 e_0(S_1) = 2.8000 \times 1.0000 = \mathbf{2.8000}$
   - For $S_2$: $\delta_0 e_0(S_2) = 2.8000 \times 0.0000 = \mathbf{0.0000}$

#### Step $t = 1$: Transition $S_2 \to \text{End}$ with $R_2 = 10.0$
1. Decay and Update Traces:
   $$e_1(S_1) = \gamma \lambda e_0(S_1) + 0 = 0.4500 \times 1.0000 = \mathbf{0.4500}$$
   $$e_1(S_2) = \gamma \lambda e_0(S_2) + 1 = 0.4500 \times 0.0000 + 1 = \mathbf{1.0000}$$
2. Compute 1-step TD error $\delta_1$:
   $$\delta_1 = R_2 + \gamma V_0(\text{End}) - V_0(S_2) = 10.0000 + 0.0000 - 2.0000 = \mathbf{8.0000}$$
3. Trace-weighted error contribution at $t = 1$:
   - For $S_1$: $\delta_1 e_1(S_1) = 8.0000 \times 0.4500 = \mathbf{3.6000}$
   - For $S_2$: $\delta_1 e_1(S_2) = 8.0000 \times 1.0000 = \mathbf{8.0000}$

---

### 5.5 Verification of Total Accumulated Updates

Sum of backward updates across the full episode:
- **Total update to State $S_1$:**
  $$\Delta V_{\text{backward}}(S_1) = \alpha \left[ \delta_0 e_0(S_1) + \delta_1 e_1(S_1) \right] = 0.5000 \times \left[ 2.8000 + 3.6000 \right] = 0.5000 \times 6.4000 = \mathbf{3.2000}$$
- **Total update to State $S_2$:**
  $$\Delta V_{\text{backward}}(S_2) = \alpha \left[ \delta_0 e_0(S_2) + \delta_1 e_1(S_2) \right] = 0.5000 \times \left[ 0.0000 + 8.0000 \right] = 0.5000 \times 8.0000 = \mathbf{4.0000}$$

### 5.6 Synthesis Visual Grid: Forward vs. Backward Identity

| State | Forward Target $G^\lambda$ | Forward Error $G^\lambda - V$ | Forward $\Delta V$ | Backward $\sum \delta_t e_t$ | Backward $\Delta V$ | Match? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S_1$** | $7.4000$ | $6.4000$ | $\mathbf{3.2000}$ | $2.8000 + 3.6000 = 6.4000$ | $\mathbf{3.2000}$ | **EXACT IDENTICAL** $\checkmark$ |
| **$S_2$** | $10.0000$ | $8.0000$ | $\mathbf{4.0000}$ | $0.0000 + 8.0000 = 8.0000$ | $\mathbf{4.0000}$ | **EXACT IDENTICAL** $\checkmark$ |

The Forward theoretical view and Backward causal eligibility trace view produce the **exact same decimal result to machine precision**!

---

## 6. Solved Illustrations

### Illustration 1: Replacing Traces vs. Accumulating Traces on Looping States
**Problem:**
Suppose an agent visits state $S$ five times in rapid succession ($t = 0, 1, 2, 3, 4$).
With $\gamma \lambda = 0.8$, compare the trace value $e_4(S)$ under:
1. Accumulating trace: $e_t = \gamma \lambda e_{t-1} + 1$
2. Replacing trace: $e_t = 1$ (upon visit)

**Solution:**
- **Accumulating Trace:**
  - $e_0 = 1.0$
  - $e_1 = 0.8(1.0) + 1 = 1.8$
  - $e_2 = 0.8(1.8) + 1 = 2.44$
  - $e_3 = 0.8(2.44) + 1 = 2.952$
  - $e_4 = 0.8(2.952) + 1 = \mathbf{3.3616}$
  In long looping trajectories, accumulating traces can explode, leading to destabilizing large step updates.
- **Replacing Trace:**
  At every visited step, the trace resets to $1.0$:
  $$e_0 = 1.0, \quad e_1 = 1.0, \quad e_2 = 1.0, \quad e_3 = 1.0, \quad e_4 = \mathbf{1.0}$$
  Replacing traces clip the trace at $1.0$, preventing runaway gradient explosion in cyclic environments. $\blacksquare$

---

### Illustration 2: Multi-Step Return $G_{t:t+n}$ and Exponential Bias Contraction across $n \in \{1, 2, 3, 4\}$ on a 4-Step Chain

**Problem:**
Consider an episodic chain environment:
$$S_0 \xrightarrow{R_1 = 1.0} S_1 \xrightarrow{R_2 = 2.0} S_2 \xrightarrow{R_3 = 0.0} S_3 \xrightarrow{R_4 = 10.0} S_T$$
where $S_T$ is a terminal absorbing state with $V(S_T) \equiv 0$.
The discount factor is $\gamma = 0.90$.
The initial value function estimates are:
$$V(S_0) = 0.0000, \quad V(S_1) = 1.0000, \quad V(S_2) = 2.0000, \quad V(S_3) = 3.0000$$
1. Compute the $n$-step returns $G_{0:0+n}$ for $n \in \{1, 2, 3, 4\}$ starting from state $S_0$.
2. Calculate the exact true Monte Carlo return $G_0$.
3. Verify the theoretical error contraction bound $|G_{0:0+n} - G_0| = \gamma^n |V(S_n) - V^*(S_n)|$.

**Solution:**

*Step 1: Compute the $n$-step returns from $S_0$.*
- **1-step Return ($n = 1$):**
  $$G_{0:1} = R_1 + \gamma V(S_1) = 1.0000 + 0.9000 \times 1.0000 = 1.0000 + 0.9000 = \mathbf{1.9000}$$
- **2-step Return ($n = 2$):**
  $$G_{0:2} = R_1 + \gamma R_2 + \gamma^2 V(S_2) = 1.0000 + 0.9000 \times 2.0000 + (0.9000)^2 \times 2.0000$$
  $$= 1.0000 + 1.8000 + 0.8100 \times 2.0000 = 2.8000 + 1.6200 = \mathbf{4.4200}$$
- **3-step Return ($n = 3$):**
  $$G_{0:3} = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 V(S_3) = 1.0000 + 0.9000 \times 2.0000 + 0.8100 \times 0.0000 + (0.9000)^3 \times 3.0000$$
  $$= 2.8000 + 0.0000 + 0.7290 \times 3.0000 = 2.8000 + 2.1870 = \mathbf{4.9870}$$
- **4-step Return ($n = 4$ - full episode):**
  $$G_{0:4} = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \gamma^4 V(S_T) = 2.8000 + 0.7290 \times 10.0000 + 0 = 2.8000 + 7.2900 = \mathbf{10.0900}$$

*Step 2: True Return and Value Functions.*
The true return along this deterministic path is the 4-step return:
$$G_0 = G_{0:4} = \mathbf{10.0900}$$
The true values under this deterministic trajectory are:
- $V^*(S_3) = R_4 = 10.0000$
- $V^*(S_2) = R_3 + \gamma V^*(S_3) = 0 + 0.9000 \times 10.0000 = 9.0000$
- $V^*(S_1) = R_2 + \gamma V^*(S_2) = 2.0000 + 0.9000 \times 9.0000 = 10.1000$
- $V^*(S_0) = R_1 + \gamma V^*(S_1) = 1.0000 + 0.9000 \times 10.1000 = 10.0900$

*Step 3: Verification of Error Contraction.*
Notice the discrepancy between $G_{0:0+n}$ and $G_0$:
- For $n = 1$: $|G_{0:1} - G_0| = |1.9000 - 10.0900| = 8.1900$.
  Theoretical bound: $\gamma^1 |V(S_1) - V^*(S_1)| = 0.9000 \times |1.0000 - 10.1000| = 0.9000 \times 9.1000 = \mathbf{8.1900} \checkmark$
- For $n = 2$: $|G_{0:2} - G_0| = |4.4200 - 10.0900| = 5.6700$.
  Theoretical bound: $\gamma^2 |V(S_2) - V^*(S_2)| = 0.8100 \times |2.0000 - 9.0000| = 0.8100 \times 7.0000 = \mathbf{5.6700} \checkmark$
- For $n = 3$: $|G_{0:3} - G_0| = |4.9870 - 10.0900| = 5.1030$.
  Theoretical bound: $\gamma^3 |V(S_3) - V^*(S_3)| = 0.7290 \times |3.0000 - 10.0000| = 0.7290 \times 7.0000 = \mathbf{5.1030} \checkmark$
- For $n = 4$: $|G_{0:4} - G_0| = |10.0900 - 10.0900| = 0.0000$.
  Theoretical bound: $\gamma^4 |V(S_T) - V^*(S_T)| = 0.6561 \times |0 - 0| = \mathbf{0.0000} \checkmark$ $\blacksquare$

---

### Illustration 3: Complete Hand Trace of Forward $\lambda$-Return vs. Backward TD($\lambda$) on a Cyclic State Visit

**Problem:**
An agent executes a 3-step episode with a cycle where state $S_1$ is visited twice:
$$\tau = (S_0 = S_1 \xrightarrow{R_1 = 1.0} S_1 = S_2 \xrightarrow{R_2 = 2.0} S_2 = S_1 \xrightarrow{R_3 = 3.0} S_3 = S_T)$$
Let $\gamma = 0.90$, $\lambda = 0.60$ (so $\gamma \lambda = 0.5400$), and $\alpha = 0.50$.
Initial estimates: $V_0(S_1) = 1.0000, V_0(S_2) = 2.0000, V_0(S_T) = 0.0000$.
Under offline batch updating:
1. Compute the forward $\lambda$-returns $G_0^\lambda, G_1^\lambda, G_2^\lambda$ and the forward weight updates $\Delta V_{\text{fwd}}(S_1), \Delta V_{\text{fwd}}(S_2)$.
2. Compute the backward 1-step errors $\delta_0, \delta_1, \delta_2$ and trace vector sequence $\mathbf{e}_0, \mathbf{e}_1, \mathbf{e}_2$.
3. Compute the backward updates $\Delta V_{\text{bwd}}(S_1), \Delta V_{\text{bwd}}(S_2)$ and prove exact equivalence.

**Solution:**

*Step 1: Forward $\lambda$-Return Computation.*
Episode length $T = 3$.
- **At $t = 0$ (State $S_1$):**
  - $G_{0:1} = R_1 + \gamma V_0(S_2) = 1.0 + 0.90(2.0) = \mathbf{2.8000}$
  - $G_{0:2} = R_1 + \gamma R_2 + \gamma^2 V_0(S_1) = 1.0 + 0.90(2.0) + 0.81(1.0) = 1.0 + 1.8 + 0.81 = \mathbf{3.6100}$
  - $G_{0:3} = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 V_0(S_T) = 2.8 + 0.81(3.0) + 0 = 2.8 + 2.43 = \mathbf{5.2300}$
  Composite $\lambda$-return ($T - t - 1 = 3 - 0 - 1 = 2$):
  $$G_0^\lambda = (1 - \lambda) G_{0:1} + (1 - \lambda)\lambda G_{0:2} + \lambda^2 G_{0:3}$$
  $$= (1 - 0.60)(2.8000) + (1 - 0.60)(0.60)(3.6100) + (0.60)^2(5.2300)$$
  $$= 0.40 \times 2.8000 + 0.24 \times 3.6100 + 0.36 \times 5.2300$$
  $$= 1.1200 + 0.8664 + 1.8828 = \mathbf{3.8692}$$

- **At $t = 1$ (State $S_2$):**
  - $G_{1:2} = R_2 + \gamma V_0(S_1) = 2.0 + 0.90(1.0) = \mathbf{2.9000}$
  - $G_{1:3} = R_2 + \gamma R_3 + \gamma^2 V_0(S_T) = 2.0 + 0.90(3.0) + 0 = \mathbf{4.7000}$
  Composite $\lambda$-return ($T - t - 1 = 1$):
  $$G_1^\lambda = (1 - \lambda) G_{1:2} + \lambda G_{1:3} = 0.40(2.9000) + 0.60(4.7000) = 1.1600 + 2.8200 = \mathbf{3.9800}$$

- **At $t = 2$ (State $S_1$):**
  Only 1 step to termination:
  $$G_2^\lambda = G_{2:3} = R_3 + \gamma V_0(S_T) = 3.0000 + 0 = \mathbf{3.0000}$$

- **Total Forward Updates:**
  - For $S_1$ (visited at $t = 0$ and $t = 2$):
    $$\Delta V_{\text{fwd}}(S_1) = \alpha \left[ (G_0^\lambda - V_0(S_1)) + (G_2^\lambda - V_0(S_1)) \right] = 0.50 \times \left[ (3.8692 - 1.0) + (3.0000 - 1.0) \right]$$
    $$= 0.50 \times [2.8692 + 2.0000] = 0.50 \times 4.8692 = \mathbf{2.4346}$$
  - For $S_2$ (visited at $t = 1$):
    $$\Delta V_{\text{fwd}}(S_2) = \alpha \left[ G_1^\lambda - V_0(S_2) \right] = 0.50 \times [3.9800 - 2.0000] = 0.50 \times 1.9800 = \mathbf{0.9900}$$

*Step 2: Backward TD($\lambda$) Computation.*
- **1-Step TD Errors:**
  - $\delta_0 = R_1 + \gamma V_0(S_2) - V_0(S_1) = 1.0 + 0.90(2.0) - 1.0 = 1.0 + 1.8 - 1.0 = \mathbf{1.8000}$
  - $\delta_1 = R_2 + \gamma V_0(S_1) - V_0(S_2) = 2.0 + 0.90(1.0) - 2.0 = 2.0 + 0.9 - 2.0 = \mathbf{0.9000}$
  - $\delta_2 = R_3 + \gamma V_0(S_T) - V_0(S_1) = 3.0 + 0.0 - 1.0 = \mathbf{2.0000}$

- **Eligibility Trace Evolution ($\gamma \lambda = 0.5400$):**
  - At $t = 0$ (visiting $S_1$):
    $$e_0(S_1) = 0 + 1.0 = \mathbf{1.0000}, \quad e_0(S_2) = \mathbf{0.0000}$$
  - At $t = 1$ (visiting $S_2$):
    $$e_1(S_1) = 0.5400 \times 1.0000 = \mathbf{0.5400}, \quad e_1(S_2) = 0.5400(0) + 1.0 = \mathbf{1.0000}$$
  - At $t = 2$ (visiting $S_1$ again):
    $$e_2(S_1) = 0.5400 \times 0.5400 + 1.0 = 0.2916 + 1.0 = \mathbf{1.2916}, \quad e_2(S_2) = 0.5400 \times 1.0 = \mathbf{0.5400}$$

*Step 3: Total Backward Updates & Equivalence Verification.*
- For State $S_1$:
  $$\Delta V_{\text{bwd}}(S_1) = \alpha \sum_{t=0}^2 \delta_t e_t(S_1) = 0.50 \times \left[ 1.8000(1.0000) + 0.9000(0.5400) + 2.0000(1.2916) \right]$$
  $$= 0.50 \times [1.8000 + 0.4860 + 2.5832] = 0.50 \times 4.8692 = \mathbf{2.4346} \quad (\equiv \Delta V_{\text{fwd}}(S_1)) \checkmark$$
- For State $S_2$:
  $$\Delta V_{\text{bwd}}(S_2) = \alpha \sum_{t=0}^2 \delta_t e_t(S_2) = 0.50 \times \left[ 1.8000(0.0000) + 0.9000(1.0000) + 2.0000(0.5400) \right]$$
  $$= 0.50 \times [0.0000 + 0.9000 + 1.0800] = 0.50 \times 1.9800 = \mathbf{0.9900} \quad (\equiv \Delta V_{\text{fwd}}(S_2)) \checkmark$$
Exact equivalence confirmed on cyclic paths! $\blacksquare$

---

### Illustration 4: True Online TD($\lambda$) Step-by-Step State and Trace Vector Dynamics

**Problem:**
Take the 2-step episodic trajectory from Section 5:
$$S_0 = S_1 \xrightarrow{R_1 = 2.0} S_1 = S_2 \xrightarrow{R_2 = 10.0} S_2 = S_T$$
with $\gamma = 0.90, \lambda = 0.50 \implies \gamma \lambda = 0.4500$, learning rate $\alpha = 0.50$, and feature representations $\mathbf{x}(S_1) = [1, 0]^\top$, $\mathbf{x}(S_2) = [0, 1]^\top$, $\mathbf{x}(S_T) = [0, 0]^\top$.
Initial weights: $\mathbf{w}_0 = [1.0000, 2.0000]^\top$, Dutch trace $\mathbf{z}_{-1} = [0, 0]^\top$, $V_{\text{old}} = 0.0$.
Execute the True Online TD($\lambda$) algorithm step-by-step and demonstrate that it resolves the online drift.

**Solution:**

*Step 1: Transition $t = 0$ ($S_1 \to S_2$, $R_1 = 2.0$)*
- Current feature $\mathbf{x}_0 = [1, 0]^\top$, next feature $\mathbf{x}_1 = [0, 1]^\top$.
- Value estimates:
  $$V = \mathbf{w}_0^\top \mathbf{x}_0 = 1.0000, \quad V' = \mathbf{w}_0^\top \mathbf{x}_1 = 2.0000$$
- TD error:
  $$\delta_0 = R_1 + \gamma V' - V = 2.0000 + 0.9000(2.0000) - 1.0000 = 2.0000 + 1.8000 - 1.0000 = \mathbf{2.8000}$$
- Dutch trace update:
  $$\mathbf{z}_0 = \gamma \lambda \mathbf{z}_{-1} + \left( 1 - \alpha \gamma \lambda \mathbf{z}_{-1}^\top \mathbf{x}_0 \right) \mathbf{x}_0 = \mathbf{0} + (1 - 0) [1, 0]^\top = \begin{bmatrix} 1.0000 \\ 0.0000 \end{bmatrix}$$
- Weight update:
  $$\mathbf{w}_1 = \mathbf{w}_0 + \alpha (\delta_0 + V - V_{\text{old}}) \mathbf{z}_0 - \alpha (V - V_{\text{old}}) \mathbf{x}_0$$
  Since $V_{\text{old}} = 0.0$, we have $\delta_0 + V - V_{\text{old}} = 2.8000 + 1.0000 - 0.0 = 3.8000$, and $V - V_{\text{old}} = 1.0000$:
  $$\mathbf{w}_1 = \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix} + 0.50(3.8000) \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} - 0.50(1.0000) \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$
  $$\mathbf{w}_1 = \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix} + \begin{bmatrix} 1.9000 \\ 0.0000 \end{bmatrix} - \begin{bmatrix} 0.5000 \\ 0.0000 \end{bmatrix} = \begin{bmatrix} 2.4000 \\ 2.0000 \end{bmatrix}$$
- Cache $V_{\text{old}} \leftarrow V' = 2.0000$.

*Step 2: Transition $t = 1$ ($S_2 \to S_T$, $R_2 = 10.0$)*
- Current feature $\mathbf{x}_1 = [0, 1]^\top$, next feature $\mathbf{x}_2 = [0, 0]^\top$.
- Value estimates using current weights $\mathbf{w}_1$:
  $$V = \mathbf{w}_1^\top \mathbf{x}_1 = 2.0000, \quad V' = \mathbf{w}_1^\top \mathbf{x}_2 = 0.0000$$
- TD error:
  $$\delta_1 = R_2 + \gamma V' - V = 10.0000 + 0.0000 - 2.0000 = \mathbf{8.0000}$$
- Dutch trace update:
  $$\mathbf{z}_1 = \gamma \lambda \mathbf{z}_0 + \left( 1 - \alpha \gamma \lambda \mathbf{z}_0^\top \mathbf{x}_1 \right) \mathbf{x}_1$$
  Here $\mathbf{z}_0^\top \mathbf{x}_1 = [1, 0] [0, 1]^\top = 0$:
  $$\mathbf{z}_1 = 0.4500 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} + (1 - 0) \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.4500 \\ 1.0000 \end{bmatrix}$$
- Weight update:
  Notice $V - V_{\text{old}} = 2.0000 - 2.0000 = 0.0000$. The correction term completely vanishes!
  $$\mathbf{w}_2 = \mathbf{w}_1 + \alpha (\delta_1 + 0) \mathbf{z}_1 - 0 = \begin{bmatrix} 2.4000 \\ 2.0000 \end{bmatrix} + 0.50(8.0000) \begin{bmatrix} 0.4500 \\ 1.0000 \end{bmatrix}$$
  $$\mathbf{w}_2 = \begin{bmatrix} 2.4000 \\ 2.0000 \end{bmatrix} + \begin{bmatrix} 1.8000 \\ 4.0000 \end{bmatrix} = \begin{bmatrix} \mathbf{4.2000} \\ \mathbf{6.0000} \end{bmatrix}$$

*Step 3: Comparison with Offline Updates.*
Net parameter changes from $\mathbf{w}_0 = [1.0, 2.0]^\top$:
- $\Delta \mathbf{w}(S_1) = 4.2000 - 1.0000 = \mathbf{3.2000}$
- $\Delta \mathbf{w}(S_2) = 6.0000 - 2.0000 = \mathbf{4.0000}$
These match the forward $\lambda$-return updates from Section 5.3 ($\mathbf{3.2000}$ and $\mathbf{4.0000}$) to exact machine precision, executed fully online at each time step! $\blacksquare$

---

### Illustration 5: Generalized Advantage Estimation (GAE) Step-by-Step Backpropagation of Advantages

**Problem:**
In a reinforcement learning trajectory of length $T = 3$, an agent observes states $S_0, S_1, S_2, S_3 = S_T$ with rewards $R_1 = 1.0, R_2 = 1.0, R_3 = 4.0$.
Discount factor $\gamma = 0.90$, GAE parameter $\lambda = 0.80 \implies \gamma \lambda = 0.7200$.
A critic network predicts baseline state values:
$$V(S_0) = 1.0000, \quad V(S_1) = 1.5000, \quad V(S_2) = 2.0000, \quad V(S_3) = 0.0000$$
1. Calculate the 1-step TD residuals $\delta_0^V, \delta_1^V, \delta_2^V$.
2. Recursively backpropagate the GAE advantages $\hat{A}_2^{\text{GAE}}, \hat{A}_1^{\text{GAE}}, \hat{A}_0^{\text{GAE}}$.
3. Compare $\hat{A}_0^{\text{GAE}}$ against the 1-step advantage $\hat{A}_0^{(1)} \triangleq \delta_0^V$ and the Monte Carlo advantage $\hat{A}_0^{(\infty)} \triangleq G_0 - V(S_0)$.

**Solution:**

*Step 1: Compute 1-Step TD Residuals.*
- $\delta_0^V = R_1 + \gamma V(S_1) - V(S_0) = 1.0000 + 0.9000(1.5000) - 1.0000 = 1.0000 + 1.3500 - 1.0000 = \mathbf{1.3500}$
- $\delta_1^V = R_2 + \gamma V(S_2) - V(S_1) = 1.0000 + 0.9000(2.0000) - 1.5000 = 1.0000 + 1.8000 - 1.5000 = \mathbf{1.3000}$
- $\delta_2^V = R_3 + \gamma V(S_3) - V(S_2) = 4.0000 + 0.0000 - 2.0000 = \mathbf{2.0000}$

*Step 2: Backward Recursion for GAE Advantages.*
Using the recurrence $\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda \hat{A}_{t+1}^{\text{GAE}}$:
- **At step $t = 2$:**
  $$\hat{A}_2^{\text{GAE}} = \delta_2^V = \mathbf{2.0000}$$
- **At step $t = 1$:**
  $$\hat{A}_1^{\text{GAE}} = \delta_1^V + \gamma \lambda \hat{A}_2^{\text{GAE}} = 1.3000 + 0.7200 \times 2.0000 = 1.3000 + 1.4400 = \mathbf{2.7400}$$
- **At step $t = 0$:**
  $$\hat{A}_0^{\text{GAE}} = \delta_0^V + \gamma \lambda \hat{A}_1^{\text{GAE}} = 1.3500 + 0.7200 \times 2.7400 = 1.3500 + 1.9728 = \mathbf{3.3228}$$

*Step 3: Comparative Analysis.*
- **1-step advantage ($n = 1$, $\lambda = 0$):**
  $$\hat{A}_0^{(1)} = \delta_0^V = \mathbf{1.3500}$$
- **Monte Carlo advantage ($n = \infty$, $\lambda = 1$):**
  $$G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 = 1.0000 + 0.9000(1.0000) + 0.8100(4.0000) = 1.0 + 0.9 + 3.24 = 5.1400$$
  $$\hat{A}_0^{(\infty)} = G_0 - V(S_0) = 5.1400 - 1.0000 = \mathbf{4.1400}$$
- **Observation:**
  $$\hat{A}_0^{(1)} = 1.3500 < \hat{A}_0^{\text{GAE}(0.9, 0.8)} = 3.3228 < \hat{A}_0^{(\infty)} = 4.1400$$
  GAE smoothly interpolates between low-variance 1-step bootstrapping ($\hat{A}_0 = 1.3500$) and unbiased full Monte Carlo ($\hat{A}_0 = 4.1400$), balancing variance against critic bias. $\blacksquare$

---

## 7. Deep Learning Connection & Application

### 1. Generalized Advantage Estimation (GAE - Schulman et al., ICLR 2016)
In modern policy gradient algorithms (PPO, TRPO), the advantage estimator $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ is nothing other than **TD($\lambda$) applied to advantage residuals**:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + \gamma \lambda \hat{A}_{t+1}^{\text{GAE}}$$
where $\delta_t^V = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$. Setting $\lambda \in [0.95, 0.98]$ provides the optimal empirical bias-variance trade-off in continuous control robotics!

### 2. Multi-Step Returns in Deep Q-Networks (Rainbow DQN)
Hessel et al. (AAAI 2018) showed that replacing the 1-step target with an $n$-step target ($n = 3$):
$$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_{a'} Q(S_{t+n}, a'; \theta^-) - Q(S_t, A_t; \theta) \right)^2 \right]$$
was one of the single most effective components of Rainbow, dramatically speeding up reward propagation across long sparse-reward horizons.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of the Part 5 hand calculations:
   - $\Delta V(S_1) = 3.2000$ and $\Delta V(S_2) = 4.0000$ for both Forward $\lambda$-return and Backward eligibility trace updates.
2. Comparison of $n$-step TD across $n \in \{1, 2, 4, 8, \infty\}$ on Sutton's Random Walk, reproducing the classic U-shaped error curve.
3. Implementation of accumulating vs. replacing traces, verifying numerical stability.

See implementation in:
[`11_reinforcement_learning/code/08_multi_step_and_eligibility_traces.py`](./code/08_multi_step_and_eligibility_traces.py)
