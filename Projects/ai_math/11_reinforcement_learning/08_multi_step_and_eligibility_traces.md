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
