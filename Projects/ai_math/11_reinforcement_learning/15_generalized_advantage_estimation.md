# Module 11.15: Generalized Advantage Estimation (GAE)

---

## 1. Intuition & 101 Motivation

In policy gradient and actor-critic algorithms, the parameter update is driven by the policy gradient theorem:
$$g = \mathbb{E} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \hat{A}_t \right]$$

The performance, sample efficiency, and stability of the entire learning process hinge directly on how we estimate the **Advantage function** $\hat{A}_t \approx Q^\pi(s_t, a_t) - V^\pi(s_t)$.

We face a classic, fundamental tension:
1. **The 1-Step TD Advantage ($\hat{A}_t = \delta_t$):**
   $$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
   - *Advantage:* Minimal variance (only 1 step of environmental reward noise).
   - *Drawback:* High bias if the neural network critic $V_{\boldsymbol{\phi}}$ is inaccurate or improperly trained.
2. **The Empirical Monte Carlo Advantage ($\hat{A}_t = G_t - V(S_t)$):**
   $$G_t - V(S_t) = \sum_{k=0}^\infty \gamma^k R_{t+k+1} - V(S_t)$$
   - *Advantage:* Completely unbiased (the expectation is identically the true advantage).
   - *Drawback:* Extremely high variance compounding noise over hundreds or thousands of future steps.

In 2016, John Schulman et al. introduced **Generalized Advantage Estimation (GAE)**, which solves this dilemma by constructing an exponentially weighted average of all multi-step advantage estimators, controlled by a tuning parameter $\lambda \in [0, 1]$.

```
   1-Step Advantage                  Generalized Advantage                     Monte Carlo
     delta_t                         Estimation GAE(gamma, lambda)            G_t - V(S_t)
   [Actor-Critic]                    [Optimal Bias-Variance Blend]            [REINFORCE]
   <------------------------------------------------------------------------------------>
   lambda = 0                               lambda in [0.95, 0.99]                 lambda = 1
   Low Variance                              SWEET SPOT                            Zero Bias
   High Bias                                                                       Huge Variance
```

GAE is arguably the single most important advantage estimation technique in modern reinforcement learning, serving as the default algorithmic engine in **PPO**, **TRPO**, and **RLHF for Large Language Models**.

---

## 2. Rigorous Mathematical Formulation

### 2.1 Multi-Step Advantage Estimators

Let $\delta_t^V \triangleq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ be the 1-step Temporal-Difference residual.
We can define a sequence of $k$-step advantage estimators:

$$\hat{A}_t^{(1)} \triangleq \delta_t^V = -V(S_t) + R_{t+1} + \gamma V(S_{t+1})$$
$$\hat{A}_t^{(2)} \triangleq \delta_t^V + \gamma \delta_{t+1}^V = -V(S_t) + R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$$
$$\hat{A}_t^{(3)} \triangleq \delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V = -V(S_t) + R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 V(S_{t+3})$$

In general, the **$k$-step advantage estimator** is the telescoping sum of $k$ consecutive TD residuals:
$$\hat{A}_t^{(k)} \triangleq \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(S_t) + \sum_{l=0}^{k-1} \gamma^l R_{t+l+1} + \gamma^k V(S_{t+k})$$

Notice that as $k \to \infty$, the bootstrapped term $\gamma^k V(S_{t+k}) \to 0$, and $\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l R_{t+l+1} - V(S_t) = G_t - V(S_t)$.

---

### 2.2 The Definition of GAE($\gamma, \lambda$)

The Generalized Advantage Estimator $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ is defined as the exponentially decaying geometric average of all $k$-step advantage estimators, weighted by powers of $\lambda \in [0, 1]$:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)}$$

#### Algebraic Reduction to TD Residuals
Substitute $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$ into the definition:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \left( \hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \dots \right)$$
$$= (1 - \lambda) \left( \delta_t^V + \lambda (\delta_t^V + \gamma \delta_{t+1}^V) + \lambda^2 (\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V) + \dots \right)$$

Collecting coefficients for each $\delta_{t+l}^V$:
- For $\delta_t^V$: coefficient is $(1 - \lambda)(1 + \lambda + \lambda^2 + \dots) = (1 - \lambda) \frac{1}{1 - \lambda} = 1 = (\gamma \lambda)^0$.
- For $\delta_{t+1}^V$: coefficient is $(1 - \lambda)(\lambda \gamma + \lambda^2 \gamma + \dots) = (1 - \lambda) \gamma \lambda (1 + \lambda + \dots) = \gamma \lambda = (\gamma \lambda)^1$.
- For $\delta_{t+l}^V$: coefficient is $(\gamma \lambda)^l$.

Thus, GAE reduces to the remarkably compact infinite sum:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

---

### 2.3 The Backward Recursive Formula

In practice, we evaluate GAE over finite trajectories of length $T$.
The sum truncates at the terminal horizon:
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V$$

This can be written as an extremely efficient **$O(T)$ backward recursion** starting from the end of the episode $t = T - 1$ down to $t = 0$:

$$\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma \lambda) (1 - d_{t+1}) \hat{A}_{t+1}^{\text{GAE}}$$
where $d_{t+1} \in \{0, 1\}$ is the terminal flag ($d = 1$ resets the advantage to zero at episode boundaries).

---

### 2.4 The Two Special Boundary Cases

1. **When $\lambda = 0$ (Pure TD / Actor-Critic):**
   $$\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
   Only the immediate TD residual is used. Maximum bias, minimum variance.
2. **When $\lambda = 1$ (Pure Monte Carlo / REINFORCE):**
   $$\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \sum_{l=0}^\infty \gamma^l R_{t+l+1} - V(S_t) = G_t - V(S_t)$$
   All future TD residuals are accumulated without decay. Zero bias, maximum variance.

For intermediate values $\lambda \in [0.95, 0.99]$, GAE provides an exponential decay that suppresses variance from the distant future while retaining enough multi-step lookahead to neutralize critic approximation errors!

---

### 2.5 Rigorous First-Principles Derivations

This section establishes the three foundational mathematical theorems governing Generalized Advantage Estimation, derived from first principles with zero skipped algebraic steps.

---

#### Derivation 11.15.1: Telescoping Sum and Exponential Weighting of GAE

##### Part 1: Problem Statement & Mathematical Goal
Let an agent interact with a Markov Decision Process, generating a trajectory $(s_t, a_t, r_{t+1}, s_{t+1}, \dots)$. For an approximate value function $V: \mathcal{S} \to \mathbb{R}$, define the one-step Temporal-Difference (TD) residual at time step $t+l$ as:
$$\delta_{t+l}^V \triangleq r_{t+l+1} + \gamma V(s_{t+l+1}) - V(s_{t+l})$$
Define the family of $k$-step advantage estimators for $k \in \{1, 2, 3, \dots\}$ as the discounted sum of $k$ consecutive TD residuals:
$$\hat{A}_t^{(k)} \triangleq \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$$
The Generalized Advantage Estimator $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ is defined as the geometric mixture of all $k$-step advantage estimators weighted by powers of $\lambda \in [0, 1)$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)}$$
Our mathematical goals are:
1. Prove the exact algebraic telescoping expansion of $\hat{A}_t^{(k)}$:
   $$\hat{A}_t^{(k)} = -V(s_t) + \sum_{l=0}^{k-1} \gamma^l r_{t+l+1} + \gamma^k V(s_{t+k})$$
2. Prove that the geometric mixture collapses identically into the exponential weighting formula:
   $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$
3. Derive the exact $O(T)$ backward recursive formula for finite trajectories:
   $$\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda (1 - d_{t+1}) \hat{A}_{t+1}^{\text{GAE}}$$
   where $d_{t+1} \in \{0, 1\}$ is the episode termination indicator.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Parameter Domains:** The discount factor satisfies $\gamma \in [0, 1)$ and the mixture parameter satisfies $\lambda \in [0, 1)$. For $\lambda = 1$, the estimator is defined as the left limit $\lim_{\lambda \to 1^-} \hat{A}_t^{\text{GAE}(\gamma, \lambda)}$.
2. **Uniform Boundedness:** The rewards are uniformly bounded: $\sup_{t} |r_{t+1}| \le R_{\max} < \infty$. The value function is uniformly bounded: $\|V\|_\infty = \sup_{s \in \mathcal{S}} |V(s)| \le V_{\max} < \infty$.
3. **Absolute Convergence & Fubini-Tonelli:** The double series over $(k, l)$ is absolutely convergent:
   $$\sum_{k=1}^\infty \lambda^{k-1} \sum_{l=0}^{k-1} \gamma^l |\delta_{t+l}^V| \le (R_{\max} + (1 + \gamma) V_{\max}) \sum_{k=1}^\infty \lambda^{k-1} \frac{1 - \gamma^k}{1 - \gamma} \le \frac{R_{\max} + (1 + \gamma) V_{\max}}{(1 - \gamma)(1 - \lambda)} < \infty$$
   This guarantees that the summation order can be freely interchanged under the Fubini-Tonelli theorem.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
Consider a 2D integer grid of summands indexed by horizon $k \ge 1$ along the horizontal axis and transition lag $l \ge 0$ along the vertical axis:
- In the original definition, we sum vertically over $l \in \{0, \dots, k-1\}$ for each column $k$, and then aggregate all columns with weights $(1 - \lambda)\lambda^{k-1}$.
- When we interchange the summation order, we sum horizontally along each row $l$. The TD residual $\delta_{t+l}^V$ appears in every $k$-step estimator having horizon $k \ge l + 1$.
- The total weight assigned to the residual $\delta_{t+l}^V$ is the tail of the geometric probability distribution from horizon $l + 1$ to $\infty$:
  $$(1 - \lambda) \sum_{k=l+1}^\infty \lambda^{k-1} = \lambda^l$$
- Combining this tail probability $\lambda^l$ with the dynamic discount factor $\gamma^l$ yields a combined discount factor of $(\gamma \lambda)^l$.
- Physically, GAE acts as an exponential decay filter (an attenuated impulse response) applied to the stream of temporal-difference shocks $\delta_{t+l}^V$.

##### Part 4: End-to-End Step-by-Step Algebraic Proof

**Step 1: Telescoping the $k$-step Advantage Estimator $\hat{A}_t^{(k)}$**
Substitute the definition of $\delta_{t+l}^V$ into the $k$-step sum:
$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l \left( r_{t+l+1} + \gamma V(s_{t+l+1}) - V(s_{t+l}) \right)$$
Split the summation into the reward component and the value bootstrap component:
$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l+1} + \sum_{l=0}^{k-1} \left( \gamma^{l+1} V(s_{t+l+1}) - \gamma^l V(s_{t+l}) \right)$$
Write out the value bootstrap summation explicitly term-by-term:
$$\sum_{l=0}^{k-1} \left( \gamma^{l+1} V(s_{t+l+1}) - \gamma^l V(s_{t+l}) \right) = \begin{aligned}[t]
& \left[ \gamma V(s_{t+1}) - V(s_t) \right] \\
& + \left[ \gamma^2 V(s_{t+2}) - \gamma V(s_{t+1}) \right] \\
& + \left[ \gamma^3 V(s_{t+3}) - \gamma^2 V(s_{t+2}) \right] \\
& + \dots \\
& + \left[ \gamma^k V(s_{t+k}) - \gamma^{k-1} V(s_{t+k-1}) \right]
\end{aligned}$$
Every positive intermediate term $\gamma^m V(s_{t+m})$ cancels with the subsequent negative term $-\gamma^m V(s_{t+m})$ for all $m \in \{1, 2, \dots, k-1\}$. The only terms that survive the telescoping cancellation are the initial term at $l=0$ and the terminal term at $l=k-1$:
$$\sum_{l=0}^{k-1} \left( \gamma^{l+1} V(s_{t+l+1}) - \gamma^l V(s_{t+l}) \right) = \gamma^k V(s_{t+k}) - V(s_t)$$
Adding the accumulated rewards yields:
$$\hat{A}_t^{(k)} = -V(s_t) + \sum_{l=0}^{k-1} \gamma^l r_{t+l+1} + \gamma^k V(s_{t+k})$$

**Step 2: Double Summation Formulation of GAE**
Substitute the unrolled definition of $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$ into the GAE mixture formula:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \left( \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V \right)$$
The domain of summation in the $(k, l)$ index plane is:
$$\mathcal{D} = \left\{ (k, l) \in \mathbb{N} \times \mathbb{N}_0 \;\middle|\; 1 \le k < \infty, \; 0 \le l \le k - 1 \right\}$$

**Step 3: Interchanging the Order of Summation**
An element $(k, l) \in \mathcal{D}$ satisfies $0 \le l \le k - 1$, which is algebraically equivalent to $l \ge 0$ and $k \ge l + 1$. Thus, the summation region can be identically expressed as:
$$\mathcal{D} = \left\{ (l, k) \in \mathbb{N}_0 \times \mathbb{N} \;\middle|\; 0 \le l < \infty, \; l + 1 \le k < \infty \right\}$$
By Fubini-Tonelli, we interchange the two sums:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{l=0}^\infty \sum_{k=l+1}^\infty \lambda^{k-1} \gamma^l \delta_{t+l}^V = (1 - \lambda) \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V \left( \sum_{k=l+1}^\infty \lambda^{k-1} \right)$$

**Step 4: Evaluating the Geometric Tail Sum**
Perform the change of summation index $j = k - (l + 1) \iff k - 1 = j + l$.
When $k = l + 1$, $j = 0$; as $k \to \infty$, $j \to \infty$. Thus:
$$\sum_{k=l+1}^\infty \lambda^{k-1} = \sum_{j=0}^\infty \lambda^{j+l} = \lambda^l \sum_{j=0}^\infty \lambda^j$$
Since $|\lambda| < 1$, the infinite geometric series converges to $\sum_{j=0}^\infty \lambda^j = \frac{1}{1 - \lambda}$. Therefore:
$$\sum_{k=l+1}^\infty \lambda^{k-1} = \lambda^l \left( \frac{1}{1 - \lambda} \right) = \frac{\lambda^l}{1 - \lambda}$$

**Step 5: Multiplying by the Normalization Constant**
Multiply the inner tail sum by the outer normalization prefactor $(1 - \lambda)$:
$$(1 - \lambda) \left( \sum_{k=l+1}^\infty \lambda^{k-1} \right) = (1 - \lambda) \cdot \frac{\lambda^l}{1 - \lambda} = \lambda^l$$

**Step 6: Assembling the Closed-Form GAE Sum**
Substitute this result back into the outer summation over $l$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V \lambda^l = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$
This completes the proof of the closed-form exponential weighting representation.

**Step 7: Backward Recursive Formula**
For an episodic or truncated trajectory with horizon $T$, let the finite-horizon GAE be:
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V$$
Isolate the first term ($l = 0$):
$$\hat{A}_t^{\text{GAE}} = (\gamma \lambda)^0 \delta_t^V + \sum_{l=1}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + \sum_{l=1}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V$$
Perform the change of variable $m = l - 1 \iff l = m + 1$. When $l = 1$, $m = 0$; when $l = T - t - 1$, $m = T - t - 2 = T - (t + 1) - 1$:
$$\sum_{l=1}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V = \sum_{m=0}^{T - (t+1) - 1} (\gamma \lambda)^{m+1} \delta_{(t+1)+m}^V = (\gamma \lambda) \sum_{m=0}^{T - (t+1) - 1} (\gamma \lambda)^m \delta_{(t+1)+m}^V$$
Recognize that the summation over $m$ is precisely $\hat{A}_{t+1}^{\text{GAE}}$:
$$\sum_{m=0}^{T - (t+1) - 1} (\gamma \lambda)^m \delta_{(t+1)+m}^V = \hat{A}_{t+1}^{\text{GAE}}$$
Thus, we obtain the exact backward recursion:
$$\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda \hat{A}_{t+1}^{\text{GAE}}$$
To account for episode termination, if transition $t+1$ enters a terminal state, the future advantage must not propagate across episode boundaries. Introducing the binary termination flag $d_{t+1} \in \{0, 1\}$ (where $d_{t+1} = 1$ if $s_{t+1}$ is terminal and $0$ otherwise) yields:
$$\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda (1 - d_{t+1}) \hat{A}_{t+1}^{\text{GAE}} \quad \blacksquare$$

---

#### Derivation 11.15.2: Bias-Variance Tradeoff Bound for GAE as a Function of $\lambda \in [0, 1]$

##### Part 1: Problem Statement & Mathematical Goal
Let $\pi_{\boldsymbol{\theta}}$ denote a parameterized policy and let $V^\pi(s)$ denote the true state-value function under $\pi$:
$$V^\pi(s) \triangleq \mathbb{E}_{\tau \sim \pi} \left[ \sum_{l=0}^\infty \gamma^l R_{t+l+1} \;\middle|\; S_t = s \right]$$
The true advantage function is defined as $A^\pi(s, a) \triangleq Q^\pi(s, a) - V^\pi(s)$.
In practical actor-critic architectures, an approximate value function $V(s) \approx V^\pi(s)$ is employed. Define the pointwise critic error as:
$$e(s) \triangleq V(s) - V^\pi(s)$$
with uniform approximation bound $\epsilon_V \triangleq \sup_{s \in \mathcal{S}} |e(s)| < \infty$.
Our mathematical goals are:
1. Derive an exact closed-form expression for the estimation bias of $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ conditioned on $(s_t, a_t)$:
   $$\operatorname{Bias}\left( \hat{A}_t^{\text{GAE}} \;\middle|\; s_t, a_t \right) \triangleq \mathbb{E}_{\pi} \left[ \hat{A}_t^{\text{GAE}(\gamma, \lambda)} \;\middle|\; s_t, a_t \right] - A^\pi(s_t, a_t)$$
2. Prove that the local error term $-e(s_t)$ induces strictly zero bias in the policy gradient update, and that the net policy gradient bias satisfies the explicit upper bound:
   $$\left\| \operatorname{Bias}_{\text{PG}}(s_t) \right\| \le C_{\text{score}} \frac{\gamma (1 - \lambda)}{1 - \gamma \lambda} \epsilon_V$$
   proving that bias is bounded by $\mathcal{O}(\epsilon_V)$ at $\lambda = 0$ and vanishes identically to $0$ at $\lambda = 1$.
3. Prove that under independent per-step residual variance $\sigma^2$, the variance of the estimator scales as:
   $$\operatorname{Var}\left( \hat{A}_t^{\text{GAE}(\gamma, \lambda)} \;\middle|\; s_t, a_t \right) = \frac{\sigma^2}{1 - (\gamma \lambda)^2}$$
   growing monotonically from $\sigma^2$ at $\lambda = 0$ to $\frac{\sigma^2}{1 - \gamma^2}$ at $\lambda = 1$.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Stationary Markov Dynamics:** State transitions $S_{t+1} \sim \mathcal{P}(\cdot \mid S_t, A_t)$ and rewards $R_{t+1} \sim \mathcal{R}(\cdot \mid S_t, A_t)$ satisfy the Markov property.
2. **Bellman Expectation Equation:** The true value function satisfies the Bellman equation:
   $$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s), s' \sim \mathcal{P}(\cdot \mid s, a)} \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$
3. **Score Function Regularity:** The policy $\pi_{\boldsymbol{\theta}}(a \mid s)$ is differentiable with respect to $\boldsymbol{\theta}$, and the score function satisfies the zero-mean baseline condition:
   $$\mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}(\cdot \mid s)} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \right] = \int_{\mathcal{A}} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s) \, da = \nabla_{\boldsymbol{\theta}} (1) = \mathbf{0}$$
   Furthermore, define $C_{\text{score}} \triangleq \sup_{s} \mathbb{E}_{a \sim \pi} \left[ \|\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)\| \right] < \infty$.
4. **Martingale Difference Residuals:** The centered TD residuals $\xi_{t+l} \triangleq \delta_{t+l}^V - \mathbb{E}[\delta_{t+l}^V \mid \mathcal{F}_{t+l}]$ satisfy $\mathbb{E}[\xi_{t+l} \mid \mathcal{F}_{t+l}] = 0$ and have conditionally bounded variance $\mathbb{E}[\xi_{t+l}^2 \mid \mathcal{F}_{t+l}] \le \sigma^2 < \infty$.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- **Why $\lambda = 0$ has Maximum Bias:** At $\lambda = 0$, $\hat{A}_t = \delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$. Taking the expectation, $V(S_{t+1})$ introduces the critic error $\gamma e(S_{t+1})$. Because the critic estimate is not verified by future real rewards, this error directly contaminates the policy update.
- **Why $\lambda = 1$ has Zero Bias:** At $\lambda = 1$, all future TD residuals are summed without decay. Telescoping cancels all intermediate critic evaluations:
  $$\sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \sum_{l=0}^\infty \gamma^l R_{t+l+1} - V(S_t) = G_t - V(S_t)$$
  The critic $V$ only appears as the state baseline $V(S_t)$. By the policy gradient theorem, subtracting any state-dependent baseline induces zero bias in the gradient! Hence, all critic modeling errors are completely erased from the policy gradient.
- **The Variance Penalty:** In exchange for eliminating critic bias, $\lambda = 1$ forces the agent to accumulate variance from every stochastic transition along the entire trajectory. Setting $\lambda \in [0.95, 0.99]$ delivers the sweet spot: the critic error is attenuated by the factor $\frac{\gamma (1 - \lambda)}{1 - \gamma \lambda} \ll 1$, while variance remains strictly finite and damped by $\frac{1}{1 - (\gamma \lambda)^2}$.

##### Part 4: End-to-End Step-by-Step Algebraic Proof

**Step 1: Express TD Residuals in Terms of True Values and Error Function**
Substitute $V(s) = V^\pi(s) + e(s)$ into the definition of $\delta_{t+l}^V$:
$$\delta_{t+l}^V = R_{t+l+1} + \gamma V(S_{t+l+1}) - V(S_{t+l})$$
$$= R_{t+l+1} + \gamma \left[ V^\pi(S_{t+l+1}) + e(S_{t+l+1}) \right] - \left[ V^\pi(S_{t+l}) + e(S_{t+l}) \right]$$
Rearrange into the true TD residual and the approximation error residual:
$$\delta_{t+l}^V = \underbrace{\left[ R_{t+l+1} + \gamma V^\pi(S_{t+l+1}) - V^\pi(S_{t+l}) \right]}_{\delta_{t+l}^{V^\pi}} + \left[ \gamma e(S_{t+l+1}) - e(S_{t+l}) \right]$$

**Step 2: Evaluate the Conditional Expectation of the True Residual $\delta_{t+l}^{V^\pi}$**
Condition on the state-action pair $(S_t = s_t, A_t = a_t)$ at the origin step $t$:
- **Case $l = 0$:**
  $$\mathbb{E}_{\pi} \left[ \delta_t^{V^\pi} \;\middle|\; s_t, a_t \right] = \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \;\middle|\; s_t, a_t \right] - V^\pi(s_t)$$
  By definition of the state-action value function $Q^\pi(s_t, a_t) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid s_t, a_t]$:
  $$\mathbb{E}_{\pi} \left[ \delta_t^{V^\pi} \;\middle|\; s_t, a_t \right] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$
- **Case $l \ge 1$:**
  Apply the tower property of conditional expectation by conditioning on $S_{t+l}$:
  $$\mathbb{E}_{\pi} \left[ \delta_{t+l}^{V^\pi} \;\middle|\; s_t, a_t \right] = \mathbb{E}_{\pi} \left[ \mathbb{E}_{\pi} \left[ R_{t+l+1} + \gamma V^\pi(S_{t+l+1}) - V^\pi(S_{t+l}) \;\middle|\; S_{t+l} \right] \;\middle|\; s_t, a_t \right]$$
  Under policy $\pi$, $A_{t+l} \sim \pi(\cdot \mid S_{t+l})$. The inner expectation is:
  $$\mathbb{E}_{a \sim \pi, s' \sim \mathcal{P}} \left[ R(S_{t+l}, a, s') + \gamma V^\pi(s') \right] - V^\pi(S_{t+l})$$
  By the Bellman expectation equation, this inner expectation is identically zero:
  $$V^\pi(S_{t+l}) - V^\pi(S_{t+l}) = 0$$
  Therefore, for all $l \ge 1$:
  $$\mathbb{E}_{\pi} \left[ \delta_{t+l}^{V^\pi} \;\middle|\; s_t, a_t \right] = 0$$

**Step 3: Evaluate the Conditional Expectation of $\hat{A}_t^{\text{GAE}}$**
Substitute the decomposition of $\delta_{t+l}^V$ into the GAE infinite sum:
$$\mathbb{E}_{\pi} \left[ \hat{A}_t^{\text{GAE}} \;\middle|\; s_t, a_t \right] = \sum_{l=0}^\infty (\gamma \lambda)^l \mathbb{E}_{\pi} \left[ \delta_{t+l}^{V^\pi} \;\middle|\; s_t, a_t \right] + \sum_{l=0}^\infty (\gamma \lambda)^l \mathbb{E}_{\pi} \left[ \gamma e(S_{t+l+1}) - e(S_{t+l}) \;\middle|\; s_t, a_t \right]$$
Since $\mathbb{E}[\delta_{t+l}^{V^\pi} \mid s_t, a_t] = 0$ for all $l \ge 1$, the first summation collapses to the $l=0$ term:
$$\sum_{l=0}^\infty (\gamma \lambda)^l \mathbb{E}_{\pi} \left[ \delta_{t+l}^{V^\pi} \;\middle|\; s_t, a_t \right] = (\gamma \lambda)^0 A^\pi(s_t, a_t) = A^\pi(s_t, a_t)$$
Thus:
$$\mathbb{E}_{\pi} \left[ \hat{A}_t^{\text{GAE}} \;\middle|\; s_t, a_t \right] - A^\pi(s_t, a_t) = \sum_{l=0}^\infty (\gamma \lambda)^l \mathbb{E}_{\pi} \left[ \gamma e(S_{t+l+1}) - e(S_{t+l}) \;\middle|\; s_t, a_t \right]$$

**Step 4: Algebraic Reduction of the Error Summation**
Examine the partial sum of the error terms up to horizon $K$:
$$E_K \triangleq \sum_{l=0}^K (\gamma \lambda)^l \left[ \gamma e(S_{t+l+1}) - e(S_{t+l}) \right]$$
Split into two distinct series:
$$E_K = \sum_{l=0}^K (\gamma \lambda)^l \gamma e(S_{t+l+1}) - \sum_{l=0}^K (\gamma \lambda)^l e(S_{t+l})$$
Extract the $l=0$ term from the second sum:
$$\sum_{l=0}^K (\gamma \lambda)^l e(S_{t+l}) = e(s_t) + \sum_{l=1}^K (\gamma \lambda)^l e(S_{t+l})$$
In the first sum, change the index to $m = l + 1 \iff l = m - 1$:
$$\sum_{l=0}^K (\gamma \lambda)^l \gamma e(S_{t+l+1}) = \sum_{m=1}^{K+1} (\gamma \lambda)^{m-1} \gamma e(S_{t+m}) = \sum_{m=1}^K (\gamma \lambda)^{m-1} \gamma e(S_{t+m}) + (\gamma \lambda)^K \gamma e(S_{t+K+1})$$
In the second sum, rename index $l$ to $m$:
$$\sum_{l=1}^K (\gamma \lambda)^l e(S_{t+l}) = \sum_{m=1}^K (\gamma \lambda)^m e(S_{t+m})$$
Combine the sums over $m \in \{1, \dots, K\}$:
$$E_K = -e(s_t) + \sum_{m=1}^K \left[ (\gamma \lambda)^{m-1} \gamma - (\gamma \lambda)^m \right] e(S_{t+m}) + (\gamma \lambda)^K \gamma e(S_{t+K+1})$$
Factor out $(\gamma \lambda)^{m-1} \gamma$ from the bracketed difference:
$$(\gamma \lambda)^{m-1} \gamma - (\gamma \lambda)^m = (\gamma \lambda)^{m-1} \gamma \left( 1 - \frac{(\gamma \lambda)^m}{(\gamma \lambda)^{m-1} \gamma} \right) = (\gamma \lambda)^{m-1} \gamma (1 - \lambda)$$
Since $\gamma \lambda < 1$ and $\|e\|_\infty \le \epsilon_V$, the boundary term vanishes as $K \to \infty$:
$$\lim_{K \to \infty} \left| (\gamma \lambda)^K \gamma e(S_{t+K+1}) \right| \le \gamma \epsilon_V \lim_{K \to \infty} (\gamma \lambda)^K = 0$$
Taking the limit $K \to \infty$ yields the exact error representation:
$$\sum_{l=0}^\infty (\gamma \lambda)^l \left[ \gamma e(S_{t+l+1}) - e(S_{t+l}) \right] = -e(s_t) + \gamma (1 - \lambda) \sum_{m=1}^\infty (\gamma \lambda)^{m-1} e(S_{t+m})$$
Taking expectations conditioned on $(s_t, a_t)$:
$$\operatorname{Bias}\left( \hat{A}_t^{\text{GAE}} \;\middle|\; s_t, a_t \right) = -e(s_t) + \gamma (1 - \lambda) \sum_{m=1}^\infty (\gamma \lambda)^{m-1} \mathbb{E}_{\pi} \left[ e(S_{t+m}) \;\middle|\; s_t, a_t \right]$$

**Step 5: Impact on Policy Gradient Update (Zero Baseline Bias)**
The policy gradient update direction at state $s_t$ is:
$$g(s_t) \triangleq \mathbb{E}_{a_t \sim \pi} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \hat{A}_t^{\text{GAE}} \right]$$
Substitute the expected advantage $\mathbb{E}[\hat{A}_t^{\text{GAE}} \mid s_t, a_t] = A^\pi(s_t, a_t) + \operatorname{Bias}(\hat{A}_t^{\text{GAE}} \mid s_t, a_t)$:
$$g(s_t) = \underbrace{\mathbb{E}_{a_t \sim \pi} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) A^\pi(s_t, a_t) \right]}_{\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \text{ (True Policy Gradient)}} + \operatorname{Bias}_{\text{PG}}(s_t)$$
where the policy gradient bias vector is:
$$\operatorname{Bias}_{\text{PG}}(s_t) = \mathbb{E}_{a_t \sim \pi} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( -e(s_t) + \gamma (1 - \lambda) \sum_{m=1}^\infty (\gamma \lambda)^{m-1} \mathbb{E}_{\pi} \left[ e(S_{t+m}) \;\middle|\; s_t, a_t \right] \right) \right]$$
Notice the first term involving $-e(s_t)$:
$$\mathbb{E}_{a_t \sim \pi} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) (-e(s_t)) \right] = -e(s_t) \int_{\mathcal{A}} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \, da_t = -e(s_t) \nabla_{\boldsymbol{\theta}} (1) = \mathbf{0}$$
The local critic error $-e(s_t)$ produces identically zero bias in the policy gradient!
The policy gradient bias is driven entirely by the future states $S_{t+m}$ for $m \ge 1$:
$$\operatorname{Bias}_{\text{PG}}(s_t) = \gamma (1 - \lambda) \sum_{m=1}^\infty (\gamma \lambda)^{m-1} \mathbb{E}_{a_t \sim \pi} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \mathbb{E}_{\pi} \left[ e(S_{t+m}) \;\middle|\; s_t, a_t \right] \right]$$
Take the Euclidean norm on both sides and apply the triangle inequality:
$$\left\| \operatorname{Bias}_{\text{PG}}(s_t) \right\| \le \gamma (1 - \lambda) \sum_{m=1}^\infty (\gamma \lambda)^{m-1} \mathbb{E}_{a_t \sim \pi} \left[ \|\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)\| \cdot \left| \mathbb{E}_{\pi}[e(S_{t+m}) \mid s_t, a_t] \right| \right]$$
Since $\left| \mathbb{E}[e(S_{t+m}) \mid s_t, a_t] \right| \le \epsilon_V$:
$$\left\| \operatorname{Bias}_{\text{PG}}(s_t) \right\| \le \gamma (1 - \lambda) \epsilon_V \mathbb{E}_{a_t \sim \pi} \left[ \|\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)\| \right] \sum_{m=1}^\infty (\gamma \lambda)^{m-1}$$
Using $\sum_{m=1}^\infty (\gamma \lambda)^{m-1} = \sum_{j=0}^\infty (\gamma \lambda)^j = \frac{1}{1 - \gamma \lambda}$ and $C_{\text{score}} = \sup_s \mathbb{E}[\|\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}\|]$:
$$\left\| \operatorname{Bias}_{\text{PG}}(s_t) \right\| \le C_{\text{score}} \frac{\gamma (1 - \lambda)}{1 - \gamma \lambda} \epsilon_V$$
- **At $\lambda = 1$:** The factor $(1 - \lambda) = (1 - 1) = 0$, giving $\left\| \operatorname{Bias}_{\text{PG}} \right\| = 0$ (exact zero bias).
- **At $\lambda = 0$:** The factor becomes $\frac{\gamma (1 - 0)}{1 - 0} = \gamma$, giving $\left\| \operatorname{Bias}_{\text{PG}} \right\| \le C_{\text{score}} \gamma \epsilon_V = \mathcal{O}(\epsilon_V)$ (maximum bias).

**Step 6: Estimator Variance as a Function of $\lambda$**
Using the martingale difference property of centered residuals $\xi_{t+l} = \delta_{t+l}^V - \mathbb{E}[\delta_{t+l}^V \mid \mathcal{F}_{t+l}]$ with $\mathbb{E}[\xi_{t+l} \xi_{t+j}] = 0$ for $l \ne j$:
$$\operatorname{Var}\left( \hat{A}_t^{\text{GAE}} \;\middle|\; s_t, a_t \right) = \sum_{l=0}^\infty (\gamma \lambda)^{2l} \operatorname{Var}\left( \delta_{t+l}^V \;\middle|\; \mathcal{F}_{t+l} \right) = \sigma^2 \sum_{l=0}^\infty \left( (\gamma \lambda)^2 \right)^l$$
Since $(\gamma \lambda)^2 < 1$, the geometric series evaluates to:
$$\operatorname{Var}\left( \hat{A}_t^{\text{GAE}} \;\middle|\; s_t, a_t \right) = \frac{\sigma^2}{1 - (\gamma \lambda)^2}$$
For finite horizon $T$:
$$\operatorname{Var}\left( \hat{A}_t^{\text{GAE}; T} \right) = \sigma^2 \sum_{l=0}^{T - t - 1} (\gamma \lambda)^{2l} = \sigma^2 \frac{1 - (\gamma \lambda)^{2(T - t)}}{1 - (\gamma \lambda)^2}$$
- **At $\lambda = 0$:** $\operatorname{Var} = \sigma^2 \frac{1 - 0}{1 - 0} = \sigma^2$ (independent of horizon $T$).
- **At $\lambda = 1$:** As $\gamma \to 1^-$, $\lim_{\gamma \to 1} \frac{1 - \gamma^{2(T - t)}}{1 - \gamma^2} = T - t = \mathcal{O}(T)$ (variance scales linearly with trajectory horizon $T$). $\blacksquare$

---

#### Derivation 11.15.3: Equivalence of GAE to $\lambda$-Return Minus Value Baseline

##### Part 1: Problem Statement & Mathematical Goal
In classical temporal-difference learning (Sutton & Barto, 1988), the $n$-step return $G_t^{(n)}$ is defined as:
$$G_t^{(n)} \triangleq \sum_{l=0}^{n-1} \gamma^l r_{t+l+1} + \gamma^n V(s_{t+n})$$
The TD($\lambda$) algorithm constructs targets using the $\lambda$-return $G_t^\lambda$, defined as the exponentially weighted average of all $n$-step returns:
$$G_t^\lambda \triangleq (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$
In modern policy gradient algorithms, the Generalized Advantage Estimator is defined as:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \hat{A}_t^{(n)}$$
where $\hat{A}_t^{(n)} \triangleq \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}^V$.
Our mathematical goals are:
1. Prove from first principles that the Generalized Advantage Estimator is identically equal to the TD($\lambda$) $\lambda$-return minus the critic baseline:
   $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = G_t^\lambda - V(s_t)$$
2. Prove the equivalence for the practical truncated finite-horizon case over an episode of length $T$:
   $$\hat{A}_{t:T}^{\text{GAE}(\gamma, \lambda)} = G_{t:T}^\lambda - V(s_t)$$
   where the truncated $\lambda$-return is defined as:
   $$G_{t:T}^\lambda \triangleq (1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} G_{t:t+n} + \lambda^{T - t - 1} G_{t:T}$$
   with terminal return $G_{t:T} \triangleq \sum_{l=0}^{T - t - 1} \gamma^l r_{t+l+1} + \gamma^{T - t} V(s_T)$.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Convergence Domain:** The discount factor satisfies $\gamma \in (0, 1)$ and the mixing parameter satisfies $\lambda \in [0, 1)$.
2. **Bounded Quantities:** Rewards $|r_{t+1}| \le R_{\max} < \infty$ and value predictions $|V(s)| \le V_{\max} < \infty$ guarantee absolute convergence of both $G_t^\lambda$ and $\hat{A}_t^{\text{GAE}}$.
3. **Deterministic Evaluation:** The value baseline $V(s_t)$ evaluated at time $t$ is a deterministic function of state $s_t$.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- In value estimation, TD($\lambda$) provides a smooth bridge between dynamic programming (bootstrapping at $n=1$) and Monte Carlo sampling (full rollout as $n \to \infty$).
- In policy gradient methods, policy improvement requires the Advantage function $A(s, a) = Q(s, a) - V(s)$.
- Derivation 11.15.3 reveals that **GAE is mathematically identical to using the TD($\lambda$) target return $G_t^\lambda$ as the surrogate $Q$-value estimator**, and subtracting the baseline state value $V(s_t)$:
  $$\hat{A}_t^{\text{GAE}} \equiv G_t^\lambda - V(s_t)$$
- This provides deep conceptual unification: GAE is simply TD($\lambda$) centered around the value function baseline.

##### Part 4: End-to-End Step-by-Step Algebraic Proof

**Step 1: Express $n$-Step Advantage in Terms of $n$-Step Return**
From Derivation 11.15.1, the $n$-step advantage estimator satisfies:
$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{n-1} \gamma^l r_{t+l+1} + \gamma^n V(s_{t+n}) - V(s_t)$$
By definition of the $n$-step bootstrapped return:
$$G_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l+1} + \gamma^n V(s_{t+n})$$
Subtracting $V(s_t)$ gives:
$$\hat{A}_t^{(n)} = G_t^{(n)} - V(s_t)$$

**Step 2: Substitute into the GAE Series Definition**
Substitute $\hat{A}_t^{(n)} = G_t^{(n)} - V(s_t)$ into the definition of $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \hat{A}_t^{(n)} = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \left( G_t^{(n)} - V(s_t) \right)$$

**Step 3: Linearity and Separation of Sums**
Since both series converge absolutely, distribute the multiplication and separate the summation:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)} - (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} V(s_t)$$

**Step 4: Factor Out the Baseline $V(s_t)$**
The baseline term $V(s_t)$ is evaluated at step $t$ and has no dependence on the summation index $n$. Therefore, it can be factored outside the summation:
$$(1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} V(s_t) = V(s_t) \cdot (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1}$$
Let $j = n - 1$. As $n$ runs from $1$ to $\infty$, $j$ runs from $0$ to $\infty$:
$$\sum_{n=1}^\infty \lambda^{n-1} = \sum_{j=0}^\infty \lambda^j = \frac{1}{1 - \lambda}$$
Substitute this back:
$$(1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} V(s_t) = V(s_t) \cdot (1 - \lambda) \cdot \frac{1}{1 - \lambda} = V(s_t) \cdot 1 = V(s_t)$$

**Step 5: Identify the Infinite-Horizon $\lambda$-Return**
By definition:
$$G_t^\lambda \triangleq (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$
Substituting $G_t^\lambda$ and $V(s_t)$ back yields:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = G_t^\lambda - V(s_t)$$
This completes the infinite-horizon proof.

**Step 6: Proof for Truncated Finite Horizon $T$**
Over a finite trajectory of horizon $T$, the truncated $\lambda$-return $G_{t:T}^\lambda$ weights the intermediate returns $G_{t:t+n}$ for $n \in \{1, 2, \dots, T - t - 1\}$ by $(1 - \lambda)\lambda^{n-1}$, and assigns the remaining cumulative probability weight $\lambda^{T - t - 1}$ to the terminal return $G_{t:T}$:
$$G_{t:T}^\lambda \triangleq (1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} G_{t:t+n} + \lambda^{T - t - 1} G_{t:T}$$
First, verify that the weights sum to unity:
$$(1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} + \lambda^{T - t - 1} = (1 - \lambda) \frac{1 - \lambda^{T - t - 1}}{1 - \lambda} + \lambda^{T - t - 1} = \left( 1 - \lambda^{T - t - 1} \right) + \lambda^{T - t - 1} = 1$$
Now subtract $V(s_t) = 1 \cdot V(s_t)$:
$$G_{t:T}^\lambda - V(s_t) = (1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} \left( G_{t:t+n} - V(s_t) \right) + \lambda^{T - t - 1} \left( G_{t:T} - V(s_t) \right)$$
Using $G_{t:t+n} - V(s_t) = \hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}^V$ and $G_{t:T} - V(s_t) = \hat{A}_t^{(T - t)} = \sum_{l=0}^{T - t - 1} \gamma^l \delta_{t+l}^V$:
$$G_{t:T}^\lambda - V(s_t) = (1 - \lambda) \sum_{n=1}^{T - t - 1} \lambda^{n-1} \left( \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}^V \right) + \lambda^{T - t - 1} \sum_{l=0}^{T - t - 1} \gamma^l \delta_{t+l}^V$$
Interchange the order of summation for the double sum over $n$ and $l$. The condition $1 \le n \le T - t - 1$ and $0 \le l \le n - 1$ is equivalent to $0 \le l \le T - t - 2$ and $l + 1 \le n \le T - t - 1$:
$$(1 - \lambda) \sum_{l=0}^{T - t - 2} \gamma^l \delta_{t+l}^V \left( \sum_{n=l+1}^{T - t - 1} \lambda^{n-1} \right) + \lambda^{T - t - 1} \sum_{l=0}^{T - t - 1} \gamma^l \delta_{t+l}^V$$
Evaluate the inner sum for $l \le T - t - 2$:
$$\sum_{n=l+1}^{T - t - 1} \lambda^{n-1} = \lambda^l \sum_{j=0}^{T - t - 2 - l} \lambda^j = \lambda^l \frac{1 - \lambda^{T - t - 1 - l}}{1 - \lambda}$$
Multiply by $(1 - \lambda)$:
$$(1 - \lambda) \sum_{n=l+1}^{T - t - 1} \lambda^{n-1} = \lambda^l \left( 1 - \lambda^{T - t - 1 - l} \right) = \lambda^l - \lambda^{T - t - 1}$$
Combine the two parts:
- For each $l \in \{0, 1, \dots, T - t - 2\}$, the total coefficient on $\gamma^l \delta_{t+l}^V$ is:
  $$\left( \lambda^l - \lambda^{T - t - 1} \right) + \lambda^{T - t - 1} = \lambda^l$$
- For the final index $l = T - t - 1$, it does not appear in the first sum, and its coefficient in the second term is precisely $\lambda^{T - t - 1}$.
Thus, for every lag $l \in \{0, 1, \dots, T - t - 1\}$, the net coefficient on $\gamma^l \delta_{t+l}^V$ is exactly $\lambda^l$!
Therefore:
$$G_{t:T}^\lambda - V(s_t) = \sum_{l=0}^{T - t - 1} \gamma^l \lambda^l \delta_{t+l}^V = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V \equiv \hat{A}_{t:T}^{\text{GAE}(\gamma, \lambda)} \quad \blacksquare$$

---

## 3. Geometric & Physical Interpretation: Attenuated Shockwave

Think of a reinforcement learning episode as an **elastic string or transmission wire**:
- At each step $t$, an unexpected reward or state transition strikes the wire, generating a local physical vibration $\delta_t^V$ (a TD shock).
- Under $\lambda = 0$, the shock remains strictly local: the actor at time $t$ only feels the impulse $\delta_t^V$ occurring at step $t$.
- Under $\lambda = 1$, the wire has zero damping: an explosion occurring 500 steps into the future travels backward down the entire wire without attenuation, violently shaking earlier actions with noisy vibrations.
- Under GAE with $\lambda \in (0, 1)$, the wire has **viscous damping**: the shockwave decays exponentially with half-life $\tau_{1/2} = \frac{\ln(2)}{-\ln(\gamma \lambda)}$ steps, allowing recent future events to guide current actions while distant chaotic noise dies out.

```
   Shock Magnitude
        ^
    1.0 |    * delta_t (Local shock)
        |     \
    0.7 |      * (gamma * lambda) delta_{t+1}
        |       \
    0.5 |        * (gamma * lambda)^2 delta_{t+2}
        |         \___________________
    0.0 +----+----+----+----+----+----+---> Future Steps l
             0    1    2    3    4    5
```

---

## 4. Real-World Analogy: The Flight Simulator Debrief

Imagine a flight student performing a simulated cross-country flight:
- At 2:00 PM ($t = 0$), the student adjusts the rudder trim.
- At 2:05 PM ($t = 1$), the plane encounters turbulent wind shear.
- At 2:40 PM ($t = 10$), the plane lands smoothly in crosswinds.

How should the flight instructor evaluate the rudder trim adjustment at 2:00 PM?
- **$\lambda = 0$ (Myopic):** The instructor only looks at the aircraft's attitude 1 second after trimming ($t = 0$). If turbulence hit 30 seconds later, the instructor ignores it.
- **$\lambda = 1.0$ (Full Episode):** The instructor judges the 2:00 PM trim adjustment based on whether the final landing at 2:40 PM was smooth, crediting or blaming the rudder trim for random weather changes 40 minutes later!
- **GAE ($\lambda = 0.95$):** The instructor evaluates the plane's flight stability over the subsequent 3 to 5 minutes—long enough to confirm the aerodynamic trim settled properly, but short enough to ignore unrelated storms 40 minutes down the flight path.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 3-Step Trajectory
Let us compute GAE step-by-step by hand on a concrete 3-step episode terminating at step $T = 3$:
- **States:** $S_0, S_1, S_2 \to S_3$ ($\text{Terminal}$)
- **Discount factor:** $\gamma = 0.9000$
- **GAE parameter:** $\lambda = 0.8000$
- **Compound decay factor:**
  $$\gamma \lambda = 0.9000 \times 0.8000 = \mathbf{0.7200}$$

**Transitions & Rewards:**
- Step 0: $S_0 \to S_1$ with reward $R_1 = 1.0000$
- Step 1: $S_1 \to S_2$ with reward $R_2 = 2.0000$
- Step 2: $S_2 \to S_3$ with reward $R_3 = 5.0000$ ($S_3$ is terminal!)

**Critic State-Value Estimates:**
$$V(S_0) = 3.0000, \quad V(S_1) = 4.0000, \quad V(S_2) = 5.0000, \quad V(S_3) = 0.0000 \text{ (Terminal)}$$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Walkthrough Value |
| :--- | :--- | :--- |
| $\gamma$ | Discount Factor | $0.9000$ |
| $\lambda$ | GAE Mixing Parameter | $0.8000$ |
| $\gamma \lambda$ | Backward Decay Rate | $0.9000 \times 0.8000 = 0.7200$ |
| $V(S_t)$ | Critic Estimate at time $t$ | $V(S_0)=3, V(S_1)=4, V(S_2)=5, V(S_3)=0$ |
| $R_{t+1}$ | Observed Immediate Reward | $R_1=1.0, R_2=2.0, R_3=5.0$ |
| $\delta_t^V$ | 1-Step TD Residual | $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ |
| $\hat{A}_t^{\text{GAE}}$ | Generalized Advantage Estimate | $\delta_t^V + \gamma \lambda \hat{A}_{t+1}^{\text{GAE}}$ |

---

### 5.3 Step 1: Forward Pass (Compute 1-Step TD Errors $\delta_t^V$)

#### At $t = 0$:
$$\delta_0^V = R_1 + \gamma V(S_1) - V(S_0) = 1.0000 + 0.9000 \times 4.0000 - 3.0000$$
$$= 1.0000 + 3.6000 - 3.0000 = \mathbf{1.6000}$$

#### At $t = 1$:
$$\delta_1^V = R_2 + \gamma V(S_2) - V(S_1) = 2.0000 + 0.9000 \times 5.0000 - 4.0000$$
$$= 2.0000 + 4.5000 - 4.0000 = \mathbf{2.5000}$$

#### At $t = 2$:
Since $S_3$ is terminal, $V(S_3) \equiv 0.0000$:
$$\delta_2^V = R_3 + \gamma V(S_3) - V(S_2) = 5.0000 + 0.9000 \times 0.0000 - 5.0000$$
$$= 5.0000 - 5.0000 = \mathbf{0.0000}$$

---

### 5.4 Step 2: Backward Recursion (Compute GAE $\hat{A}_t^{\text{GAE}}$)

Boundary Condition at terminal state: $\hat{A}_3^{\text{GAE}} = 0.0000$.

#### Backward Step $t = 2$:
$$\hat{A}_2^{\text{GAE}} = \delta_2^V + (\gamma \lambda) \hat{A}_3^{\text{GAE}} = 0.0000 + 0.7200 \times 0.0000 = \mathbf{0.0000}$$

#### Backward Step $t = 1$:
$$\hat{A}_1^{\text{GAE}} = \delta_1^V + (\gamma \lambda) \hat{A}_2^{\text{GAE}} = 2.5000 + 0.7200 \times 0.0000 = \mathbf{2.5000}$$

#### Backward Step $t = 0$:
$$\hat{A}_0^{\text{GAE}} = \delta_0^V + (\gamma \lambda) \hat{A}_1^{\text{GAE}} = 1.6000 + 0.7200 \times 2.5000$$
$$= 1.6000 + 1.8000 = \mathbf{3.4000}$$

---

### 5.5 Step 3: Comparison with Boundary Extremes ($\lambda = 0$ vs $\lambda = 1$)

Let us verify what happens at $\lambda = 0$ and $\lambda = 1$:

#### 1. Pure 1-Step Advantage ($\lambda = 0$):
$$\hat{A}_0^{(1)} = \delta_0^V = \mathbf{1.6000}$$
$$\hat{A}_1^{(1)} = \delta_1^V = \mathbf{2.5000}$$
$$\hat{A}_2^{(1)} = \delta_2^V = \mathbf{0.0000}$$

#### 2. Pure Monte Carlo Advantage ($\lambda = 1.0$):
First, compute full discounted returns $G_t$:
- $G_2 = R_3 = \mathbf{5.0000} \implies \hat{A}_2^{(\infty)} = G_2 - V(S_2) = 5.0 - 5.0 = \mathbf{0.0000}$.
- $G_1 = R_2 + \gamma G_2 = 2.0 + 0.9(5.0) = 2.0 + 4.5 = \mathbf{6.5000} \implies \hat{A}_1^{(\infty)} = 6.5 - V(S_1) = 6.5 - 4.0 = \mathbf{2.5000}$.
- $G_0 = R_1 + \gamma G_1 = 1.0 + 0.9(6.5) = 1.0 + 5.85 = \mathbf{6.8500} \implies \hat{A}_0^{(\infty)} = 6.85 - V(S_0) = 6.85 - 3.0 = \mathbf{3.8500}$.

Under GAE ($\lambda = 0.8$):
$$\hat{A}_0^{\text{GAE}} = \mathbf{3.4000}$$
Notice that $\hat{A}_0^{\text{GAE}} = 3.4000$ lies elegantly between the 1-step estimate ($1.6000$) and the full Monte Carlo estimate ($3.8500$)!

---

### 5.6 Summary Visual Grid: GAE Computation Matrix

| Step $t$ | $R_{t+1}$ | $V(S_t)$ | $V(S_{t+1})$ | TD Error $\delta_t^V$ | Backward Term $(\gamma \lambda) \hat{A}_{t+1}$ | $\hat{A}_t^{\text{GAE}(0.8)}$ | $\hat{A}_t^{(\lambda=0)}$ | $\hat{A}_t^{(\lambda=1)}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 0$** | $1.0$ | $3.0$ | $4.0$ | $\mathbf{1.6000}$ | $0.72 \times 2.5 = 1.8000$ | $\mathbf{3.4000}$ | $1.6000$ | $3.8500$ |
| **$t = 1$** | $2.0$ | $4.0$ | $5.0$ | $\mathbf{2.5000}$ | $0.72 \times 0.0 = 0.0000$ | $\mathbf{2.5000}$ | $2.5000$ | $2.5000$ |
| **$t = 2$** | $5.0$ | $5.0$ | $0.0$ | $\mathbf{0.0000}$ | $0.72 \times 0.0 = 0.0000$ | $\mathbf{0.0000}$ | $0.0000$ | $0.0000$ |
| **$t = 3$** | - | $0.0$ | - | - | - | $\mathbf{0.0000}$ | - | - |

Every single arithmetic step is exact to machine precision!

---

## 6. Solved Illustrations

This section presents 5 complete, solved numerical illustrations covering the mathematical mechanics, backward recursions, parameter sweeps, batch normalization, and boundary truncation dynamics of Generalized Advantage Estimation.

---

### Illustration 1: Telescoping Sum Proof of $\lambda = 1$ Identity & Concrete Numerical Confirmation

#### Problem Statement
1. Prove analytically that the infinite sum of discounted TD residuals $\sum_{l=0}^\infty \gamma^l \delta_{t+l}^V$ collapses telescopically into the empirical Monte Carlo advantage $G_t - V(S_t)$.
2. Verify this identity numerically step-by-step on a concrete 3-step episodic trajectory with discount factor $\gamma = 0.9000$, rewards $R_1 = 1.0, R_2 = 2.0, R_3 = 5.0$, and critic state-value estimates $V(S_0) = 3.0, V(S_1) = 4.0, V(S_2) = 5.0, V(S_3) = 0.0$ (terminal state).

#### Analytical Derivation
Expand the sum of discounted TD errors over a horizon of $k$ steps:
$$\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l \left( R_{t+l+1} + \gamma V(S_{t+l+1}) - V(S_{t+l}) \right)$$
$$= \sum_{l=0}^{k-1} \gamma^l R_{t+l+1} + \sum_{l=0}^{k-1} \left( \gamma^{l+1} V(S_{t+l+1}) - \gamma^l V(S_{t+l}) \right)$$

Examine the second summation explicitly:
$$\sum_{l=0}^{k-1} \left( \gamma^{l+1} V(S_{t+l+1}) - \gamma^l V(S_{t+l}) \right) = \begin{aligned}[t]
& \left[ \gamma V(S_{t+1}) - V(S_t) \right] \\
& + \left[ \gamma^2 V(S_{t+2}) - \gamma V(S_{t+1}) \right] \\
& + \dots \\
& + \left[ \gamma^k V(S_{t+k}) - \gamma^{k-1} V(S_{t+k-1}) \right]
\end{aligned}$$
Every intermediate value term $\gamma^m V(S_{t+m})$ cancels telescopically:
$$\sum_{l=0}^{k-1} \left( \gamma^{l+1} V(S_{t+l+1}) - \gamma^l V(S_{t+l}) \right) = \gamma^k V(S_{t+k}) - V(S_t)$$

Combining with the accumulated rewards:
$$\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l R_{t+l+1} + \gamma^k V(S_{t+k}) - V(S_t)$$

Taking the limit as $k \to \infty$, since $\gamma \in [0, 1)$ and $\|V\|_\infty \le V_{\max} < \infty$, the bootstrap term vanishes: $\lim_{k \to \infty} \gamma^k V(S_{t+k}) = 0$. Hence:
$$\sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \underbrace{\sum_{l=0}^\infty \gamma^l R_{t+l+1}}_{G_t} - V(S_t) = G_t - V(S_t) \quad \blacksquare$$

#### Step-by-Step Numerical Walkthrough

**1. Calculate Individual 1-Step TD Residuals $\delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$:**
- For $t = 0$:
  $$\delta_0^V = 1.0000 + 0.9000 \times 4.0000 - 3.0000 = 1.0000 + 3.6000 - 3.0000 = \mathbf{1.6000}$$
- For $t = 1$:
  $$\delta_1^V = 2.0000 + 0.9000 \times 5.0000 - 4.0000 = 2.0000 + 4.5000 - 4.0000 = \mathbf{2.5000}$$
- For $t = 2$ (terminal step with $V(S_3) = 0.0$):
  $$\delta_2^V = 5.0000 + 0.9000 \times 0.0000 - 5.0000 = 5.0000 + 0.0000 - 5.0000 = \mathbf{0.0000}$$

**2. Evaluate Discounted TD Residual Sum at $t = 0$:**
$$\sum_{l=0}^2 \gamma^l \delta_l^V = \delta_0^V + \gamma \delta_1^V + \gamma^2 \delta_2^V$$
$$= 1.6000 + 0.9000 \times 2.5000 + (0.9000)^2 \times 0.0000$$
$$= 1.6000 + 2.2500 + 0.0000 = \mathbf{3.8500}$$

**3. Evaluate Monte Carlo Return $G_0$ and Advantage $G_0 - V(S_0)$:**
$$G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 = 1.0000 + 0.9000 \times 2.0000 + (0.9000)^2 \times 5.0000$$
$$= 1.0000 + 1.8000 + 0.8100 \times 5.0000 = 1.0000 + 1.8000 + 4.0500 = \mathbf{6.8500}$$
Subtracting the baseline $V(S_0) = 3.0000$:
$$G_0 - V(S_0) = 6.8500 - 3.0000 = \mathbf{3.8500}$$

**4. Explicit Numerical Demonstration of Telescoping Cancellations:**
$$\begin{aligned}
\sum_{l=0}^2 \gamma^l \delta_l^V &= \left[ R_1 + \gamma V(S_1) - V(S_0) \right] + \gamma \left[ R_2 + \gamma V(S_2) - V(S_1) \right] + \gamma^2 \left[ R_3 + \gamma V(S_3) - V(S_2) \right] \\
&= \left[ R_1 + \gamma R_2 + \gamma^2 R_3 \right] + \left[ \gamma V(S_1) - V(S_0) + \gamma^2 V(S_2) - \gamma V(S_1) + \gamma^3 V(S_3) - \gamma^2 V(S_2) \right] \\
&= 6.8500 + \left[ \mathbf{+3.6000} - 3.0000 + \mathbf{+4.0500} \mathbf{- 3.6000} + 0.0000 \mathbf{- 4.0500} \right] \\
&= 6.8500 - 3.0000 = \mathbf{3.8500}
\end{aligned}$$
The intermediate values $+3.6000$ and $-3.6000$, and $+4.0500$ and $-4.0500$, cancel to exact machine precision!

---

### Illustration 2: Step-by-Step Backward Recursion of $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ over a 4-Step Trajectory

#### Problem Statement
Compute the Generalized Advantage Estimates across a 4-step episodic trajectory ($T = 4$) terminating at step $S_4$ ($d_4 = 1$, $V(S_4) = 0.0000$), using standard PPO hyperparameters:
- Discount factor: $\gamma = 0.9900$
- GAE parameter: $\lambda = 0.9500$
- Compound backward decay rate:
  $$\gamma \lambda = 0.9900 \times 0.9500 = \mathbf{0.9405}$$

Trajectory inputs:
- Observed rewards: $\mathbf{R} = [R_1, R_2, R_3, R_4] = [1.5000, \; -0.5000, \; 2.0000, \; 3.0000]$
- Critic values: $[V(S_0), V(S_1), V(S_2), V(S_3), V(S_4)] = [2.0000, \; 2.5000, \; 1.8000, \; 3.2000, \; 0.0000]$

#### Step-by-Step Solution

**Step 1: Forward Pass (Compute 1-Step TD Residuals $\delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$)**
- At $t = 0$:
  $$\delta_0^V = 1.5000 + 0.9900 \times 2.5000 - 2.0000 = 1.5000 + 2.4750 - 2.0000 = \mathbf{1.9750}$$
- At $t = 1$:
  $$\delta_1^V = -0.5000 + 0.9900 \times 1.8000 - 2.5000 = -0.5000 + 1.7820 - 2.5000 = \mathbf{-1.2180}$$
- At $t = 2$:
  $$\delta_2^V = 2.0000 + 0.9900 \times 3.2000 - 1.8000 = 2.0000 + 3.1680 - 1.8000 = \mathbf{3.3680}$$
- At $t = 3$ (terminal transition into $S_4$ where $V(S_4) = 0.0$):
  $$\delta_3^V = 3.0000 + 0.9900 \times 0.0000 - 3.2000 = 3.0000 + 0.0000 - 3.2000 = \mathbf{-0.2000}$$

**Step 2: Backward Recursion Pass ($\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda \hat{A}_{t+1}^{\text{GAE}}$)**
Boundary condition at termination: $\hat{A}_4^{\text{GAE}} = 0.0000$.

- **At $t = 3$:**
  $$\hat{A}_3^{\text{GAE}} = \delta_3^V + (\gamma \lambda) \hat{A}_4^{\text{GAE}} = -0.2000 + 0.9405 \times 0.0000 = \mathbf{-0.2000}$$
- **At $t = 2$:**
  $$\hat{A}_2^{\text{GAE}} = \delta_2^V + (\gamma \lambda) \hat{A}_3^{\text{GAE}} = 3.3680 + 0.9405 \times (-0.2000)$$
  $$= 3.3680 - 0.188100 = \mathbf{3.1799}$$
- **At $t = 1$:**
  $$\hat{A}_1^{\text{GAE}} = \delta_1^V + (\gamma \lambda) \hat{A}_2^{\text{GAE}} = -1.2180 + 0.9405 \times 3.179900$$
  $$= -1.2180 + 2.99069595 = \mathbf{1.7727}$$
- **At $t = 0$:**
  $$\hat{A}_0^{\text{GAE}} = \delta_0^V + (\gamma \lambda) \hat{A}_1^{\text{GAE}} = 1.9750 + 0.9405 \times 1.77269595$$
  $$= 1.9750 + 1.66722054 = \mathbf{3.6422}$$

#### Summary Table

| Step $t$ | $R_{t+1}$ | $V(S_t)$ | $V(S_{t+1})$ | TD Error $\delta_t^V$ | Backward Product $(\gamma \lambda)\hat{A}_{t+1}$ | GAE Advantage $\hat{A}_t^{\text{GAE}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 0$** | $1.5000$ | $2.0000$ | $2.5000$ | $\mathbf{+1.9750}$ | $0.9405 \times 1.7727 = +1.6672$ | $\mathbf{+3.6422}$ |
| **$t = 1$** | $-0.5000$ | $2.5000$ | $1.8000$ | $\mathbf{-1.2180}$ | $0.9405 \times 3.1799 = +2.9907$ | $\mathbf{+1.7727}$ |
| **$t = 2$** | $2.0000$ | $1.8000$ | $3.2000$ | $\mathbf{+3.3680}$ | $0.9405 \times (-0.2000) = -0.1881$ | $\mathbf{+3.1799}$ |
| **$t = 3$** | $3.0000$ | $3.2000$ | $0.0000$ | $\mathbf{-0.2000}$ | $0.9405 \times 0.0000 = 0.0000$ | $\mathbf{-0.2000}$ |

---

### Illustration 3: Comparison of $\hat{A}_t^{\text{GAE}}$ Values across $\lambda \in \{0.0, 0.5, 0.95, 1.0\}$

#### Problem Statement
Using the 4-step trajectory from Illustration 2 ($\gamma = 0.9900$), compute the advantage estimates across four distinct values of $\lambda \in \{0.00, 0.50, 0.95, 1.00\}$.
Demonstrate explicitly how varying $\lambda$ smoothly transitions the estimator from the pure 1-step TD residual ($\lambda = 0$) to the unbiased Monte Carlo advantage $G_t - V(S_t)$ ($\lambda = 1$).

#### Step-by-Step Solution

**1. Case $\lambda = 0.00$ ($\gamma \lambda = 0.0000$, 1-Step TD):**
$$\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V$$
- $\hat{A}_3 = \delta_3^V = \mathbf{-0.2000}$
- $\hat{A}_2 = \delta_2^V = \mathbf{+3.3680}$
- $\hat{A}_1 = \delta_1^V = \mathbf{-1.2180}$
- $\hat{A}_0 = \delta_0^V = \mathbf{+1.9750}$

**2. Case $\lambda = 0.50$ ($\gamma \lambda = 0.9900 \times 0.5000 = 0.4950$):**
- $\hat{A}_3 = \delta_3^V = \mathbf{-0.2000}$
- $\hat{A}_2 = 3.3680 + 0.4950 \times (-0.2000) = 3.3680 - 0.099000 = \mathbf{+3.2690}$
- $\hat{A}_1 = -1.2180 + 0.4950 \times 3.269000 = -1.2180 + 1.618155 = \mathbf{+0.4002}$
- $\hat{A}_0 = 1.9750 + 0.4950 \times 0.400155 = 1.9750 + 0.198077 = \mathbf{+2.1731}$

**3. Case $\lambda = 0.95$ ($\gamma \lambda = 0.9405$, Standard PPO):**
- As computed in Illustration 2:
  $$\hat{\mathbf{A}}^{\text{GAE}} = [\mathbf{+3.6422}, \; \mathbf{+1.7727}, \; \mathbf{+3.1799}, \; \mathbf{-0.2000}]$$

**4. Case $\lambda = 1.00$ ($\gamma \lambda = 0.9900 \times 1.0000 = 0.9900$, Pure Monte Carlo):**
- $\hat{A}_3 = -0.2000 + 0.9900 \times 0.0000 = \mathbf{-0.2000}$
- $\hat{A}_2 = 3.3680 + 0.9900 \times (-0.2000) = 3.3680 - 0.198000 = \mathbf{+3.1700}$
- $\hat{A}_1 = -1.2180 + 0.9900 \times 3.170000 = -1.2180 + 3.138300 = \mathbf{+1.9203}$
- $\hat{A}_0 = 1.9750 + 0.9900 \times 1.920300 = 1.9750 + 1.901097 = \mathbf{+3.8761}$

**5. Verification via Monte Carlo Return Minus Baseline $G_t - V(S_t)$:**
- At $t = 3$: $G_3 = R_4 = 3.0000 \implies G_3 - V(S_3) = 3.0000 - 3.2000 = \mathbf{-0.2000}$
- At $t = 2$: $G_2 = R_3 + \gamma G_3 = 2.0 + 0.99(3.0) = 4.9700 \implies G_2 - V(S_2) = 4.9700 - 1.8000 = \mathbf{+3.1700}$
- At $t = 1$: $G_1 = R_2 + \gamma G_2 = -0.5 + 0.99(4.97) = 4.4203 \implies G_1 - V(S_1) = 4.4203 - 2.5000 = \mathbf{+1.9203}$
- At $t = 0$: $G_0 = R_1 + \gamma G_1 = 1.5 + 0.99(4.4203) = 5.876097 \implies G_0 - V(S_0) = 5.876097 - 2.0000 = \mathbf{+3.8761}$

The $\lambda = 1.00$ GAE column matches the analytical Monte Carlo advantage $G_t - V(S_t)$ identically to full precision.

#### Comparative Spectrum Table

| Time Step $t$ | $\lambda = 0.00$ (1-Step TD) | $\lambda = 0.50$ (Moderate Lookahead) | $\lambda = 0.95$ (GAE Sweet Spot) | $\lambda = 1.00$ (Monte Carlo) | Analytical $G_t - V(S_t)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 0$** | $+1.9750$ | $+2.1731$ | $+3.6422$ | $+3.8761$ | $\mathbf{+3.8761}$ |
| **$t = 1$** | $-1.2180$ | $+0.4002$ | $+1.7727$ | $+1.9203$ | $\mathbf{+1.9203}$ |
| **$t = 2$** | $+3.3680$ | $+3.2690$ | $+3.1799$ | $+3.1700$ | $\mathbf{+3.1700}$ |
| **$t = 3$** | $-0.2000$ | $-0.2000$ | $-0.2000$ | $-0.2000$ | $\mathbf{-0.2000}$ |

Observe that at step $t=0$, as $\lambda$ increases from $0.00$ to $1.00$, $\hat{A}_0^{\text{GAE}}$ smoothly interpolates through:
$$1.9750 \;\longrightarrow\; 2.1731 \;\longrightarrow\; 3.6422 \;\longrightarrow\; 3.8761$$
illustrating the exact continuous bridge from local bootstrapping to full Monte Carlo rollout.

---

### Illustration 4: GAE-Based PPO Advantage Normalization across a Mini-Batch of 6 Transitions

#### Problem Statement
In Proximal Policy Optimization (PPO), raw GAE advantages $\hat{A}_i$ are standardized across each training mini-batch before computing policy clipped surrogate losses:
$$\hat{A}_i^{\text{norm}} = \frac{\hat{A}_i - \mu_{\hat{A}}}{\sigma_{\hat{A}} + \epsilon}$$
where $\epsilon = 10^{-8}$ prevents division by zero.

Given a mini-batch of $N = 6$ transition advantages:
$$\hat{\mathbf{A}} = [3.6422, \; 1.7727, \; 3.1799, \; -0.2000, \; -1.4500, \; 0.8500]$$
Compute step-by-step:
1. The mini-batch sample mean $\mu_{\hat{A}}$
2. The deviations $(A_i - \mu)$, squared deviations $(A_i - \mu)^2$, sample variance $s^2$ (with Bessel's correction $N - 1 = 5$), and sample standard deviation $s$ (standard PyTorch default)
3. The normalized advantages $\hat{A}_i^{\text{norm}}$
4. Verify that the standardized vector has zero mean and unit variance.

#### Step-by-Step Solution

**Step 1: Compute the Mini-Batch Mean $\mu_{\hat{A}}$**
$$\mu_{\hat{A}} = \frac{1}{6} \sum_{i=1}^6 \hat{A}_i = \frac{3.6422 + 1.7727 + 3.1799 - 0.2000 - 1.4500 + 0.8500}{6}$$
$$\mu_{\hat{A}} = \frac{7.7948}{6} = \mathbf{1.299133}$$

**Step 2: Compute Deviations and Squared Deviations**

| Transition $i$ | Raw Advantage $\hat{A}_i$ | Deviation $(A_i - \mu)$ | Squared Deviation $(A_i - \mu)^2$ |
| :---: | :---: | :---: | :---: |
| **$i = 0$** | $+3.6422$ | $3.6422 - 1.299133 = \mathbf{+2.343067}$ | $(+2.343067)^2 = \mathbf{5.489963}$ |
| **$i = 1$** | $+1.7727$ | $1.7727 - 1.299133 = \mathbf{+0.473567}$ | $(+0.473567)^2 = \mathbf{0.224266}$ |
| **$i = 2$** | $+3.1799$ | $3.1799 - 1.299133 = \mathbf{+1.880767}$ | $(+1.880767)^2 = \mathbf{3.537285}$ |
| **$i = 3$** | $-0.2000$ | $-0.2000 - 1.299133 = \mathbf{-1.499133}$ | $(-1.499133)^2 = \mathbf{2.247400}$ |
| **$i = 4$** | $-1.4500$ | $-1.4500 - 1.299133 = \mathbf{-2.749133}$ | $(-2.749133)^2 = \mathbf{7.557732}$ |
| **$i = 5$** | $+0.8500$ | $0.8500 - 1.299133 = \mathbf{-0.449133}$ | $(-0.449133)^2 = \mathbf{0.201720}$ |
| **Sum** | $\mathbf{7.7948}$ | $\mathbf{0.000000}$ | $\mathbf{19.258366}$ |

**Step 3: Compute Sample Variance and Standard Deviation**
Using the standard unbiased sample estimator ($N - 1 = 5$ degrees of freedom, as implemented in PyTorch `torch.std(unbiased=True)`):
$$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (A_i - \mu)^2 = \frac{19.258366}{5} = \mathbf{3.851673}$$
$$s = \sqrt{3.851673} = \mathbf{1.962568}$$

*(Note: The population standard deviation without Bessel's correction is $\sigma = \sqrt{19.258366 / 6} = \mathbf{1.791571}$)*.

**Step 4: Standardize Each Transition Advantage**
With $\epsilon = 10^{-8}$, denominator is $s + \epsilon = 1.962568 + 10^{-8} \approx 1.962568$:
- For $i = 0$: $\hat{A}_0^{\text{norm}} = \frac{+2.343067}{1.962568} = \mathbf{+1.1939}$
- For $i = 1$: $\hat{A}_1^{\text{norm}} = \frac{+0.473567}{1.962568} = \mathbf{+0.2413}$
- For $i = 2$: $\hat{A}_2^{\text{norm}} = \frac{+1.880767}{1.962568} = \mathbf{+0.9583}$
- For $i = 3$: $\hat{A}_3^{\text{norm}} = \frac{-1.499133}{1.962568} = \mathbf{-0.7639}$
- For $i = 4$: $\hat{A}_4^{\text{norm}} = \frac{-2.749133}{1.962568} = \mathbf{-1.4008}$
- For $i = 5$: $\hat{A}_5^{\text{norm}} = \frac{-0.449133}{1.962568} = \mathbf{-0.2288}$

**Step 5: Verification of Statistical Moments**
- Normalized Mean:
  $$\mu_{\text{norm}} = \frac{1.1939 + 0.2413 + 0.9583 - 0.7639 - 1.4008 - 0.2288}{6} = \frac{0.0000}{6} = \mathbf{0.0000}$$
- Normalized Sample Variance:
  $$s_{\text{norm}}^2 = \frac{(1.1939)^2 + (0.2413)^2 + (0.9583)^2 + (-0.7639)^2 + (-1.4008)^2 + (-0.2288)^2}{5}$$
  $$= \frac{1.425397 + 0.058226 + 0.918339 + 0.583543 + 1.962241 + 0.052349}{5} = \frac{5.000095}{5} = \mathbf{1.0000}$$
- Normalized Sample Standard Deviation:
  $$s_{\text{norm}} = \sqrt{1.0000} = \mathbf{1.0000}$$

---

### Illustration 5: Generalized Advantage Estimation with Generalized Truncation at Episode Termination ($T=5$, Terminal Mask $d_T = 1$ vs Non-Terminal Truncation)

#### Problem Statement
In deep reinforcement learning implementations (e.g. Gym/Gymnasium and CleanRL), an episode may end for two completely distinct physical reasons:
1. **Natural Termination (`terminated = True`, $d = 1$):** The agent achieved a goal or suffered catastrophic failure (e.g. falling over). The subsequent state is truly terminal; future expected rewards are $0$, so $V(S_{\text{terminal}}) \equiv 0$.
2. **Artificial Truncation (`truncated = True`, $d = 0$):** The environment hit an artificial maximum time limit (e.g. `TimeLimit.max_episode_steps = 1000`). The episode was cut short artificially, but the agent was still functioning; therefore, future rewards are NOT zero and must be bootstrapped via the critic prediction $V(S_{\text{cutoff}})$.

Consider a 5-step rollout ($T = 5$, steps $t = 0, 1, 2, 3, 4$) with:
- Discount factor: $\gamma = 0.9500$
- GAE parameter: $\lambda = 0.9000$
- Compound factor:
  $$\gamma \lambda = 0.9500 \times 0.9000 = \mathbf{0.8550}$$

Trajectory parameters:
- Rewards: $\mathbf{R} = [R_1, R_2, R_3, R_4, R_5] = [1.0, \; 0.0, \; 2.0, \; -1.0, \; 3.0]$
- Critic values: $[V(S_0), V(S_1), V(S_2), V(S_3), V(S_4), V(S_5)] = [1.5, \; 2.0, \; 1.0, \; 2.5, \; 3.0, \; 2.0]$

Contrast the following two cases:
- **Case A (Natural Termination at step $t=4$):** Terminal flag $d_5 = 1 \implies$ future value is masked out: $V_{\text{boot}}(S_5) = 0.0000$.
- **Case B (Time-Limit Truncation at step $t=4$):** Non-terminal flag $d_5 = 0 \implies$ critic bootstraps future value: $V_{\text{boot}}(S_5) = V(S_5) = 2.0000$.

Compute step-by-step:
1. TD residuals $\delta_t^V$ for both cases
2. GAE advantages $\hat{A}_t^{\text{GAE}}$ for both cases
3. The difference vector $\Delta \hat{\mathbf{A}} = \hat{\mathbf{A}}^{\text{Case B}} - \hat{\mathbf{A}}^{\text{Case A}}$
4. Prove that the difference propagates backward through time according to the exact analytical relationship:
   $$\Delta \hat{A}_t = \gamma V(S_5) \cdot (\gamma \lambda)^{T - 1 - t}$$

#### Step-by-Step Solution

**1. Forward Pass: TD Residuals $\delta_t^V = R_{t+1} + \gamma V_{\text{next}} - V(S_t)$**
For $t = 0, 1, 2, 3$, both cases share identical transitions and critic values:
- $t = 0$: $\delta_0^V = 1.0 + 0.95(2.0) - 1.5 = 1.0 + 1.9 - 1.5 = \mathbf{+1.4000}$
- $t = 1$: $\delta_1^V = 0.0 + 0.95(1.0) - 2.0 = 0.95 - 2.0 = \mathbf{-1.0500}$
- $t = 2$: $\delta_2^V = 2.0 + 0.95(2.5) - 1.0 = 2.0 + 2.375 - 1.0 = \mathbf{+3.3750}$
- $t = 3$: $\delta_3^V = -1.0 + 0.95(3.0) - 2.5 = -1.0 + 2.85 - 2.5 = \mathbf{-0.6500}$

Now examine boundary step $t = 4$:
- **Case A (Termination, $V(S_5) = 0$):**
  $$\delta_{4, A}^V = R_5 + \gamma \times 0.0 - V(S_4) = 3.0 + 0.0 - 3.0 = \mathbf{0.0000}$$
- **Case B (Truncation, $V(S_5) = 2.0$):**
  $$\delta_{4, B}^V = R_5 + \gamma \times V(S_5) - V(S_4) = 3.0 + 0.95(2.0) - 3.0 = 3.0 + 1.9000 - 3.0 = \mathbf{+1.9000}$$
Notice:
$$\delta_{4, B}^V - \delta_{4, A}^V = 1.9000 - 0.0000 = \gamma V(S_5) = 0.95 \times 2.0 = \mathbf{1.9000}$$

**2. Backward Pass: Case A (Natural Termination, $d_5 = 1$)**
- $\hat{A}_{4, A}^{\text{GAE}} = \delta_{4, A}^V = \mathbf{0.0000}$
- $\hat{A}_{3, A}^{\text{GAE}} = -0.6500 + 0.8550 \times (0.0000) = \mathbf{-0.6500}$
- $\hat{A}_{2, A}^{\text{GAE}} = 3.3750 + 0.8550 \times (-0.6500) = 3.3750 - 0.555750 = \mathbf{+2.819250} \approx \mathbf{+2.8193}$
- $\hat{A}_{1, A}^{\text{GAE}} = -1.0500 + 0.8550 \times (2.819250) = -1.0500 + 2.410459 = \mathbf{+1.360459} \approx \mathbf{+1.3605}$
- $\hat{A}_{0, A}^{\text{GAE}} = 1.4000 + 0.8550 \times (1.360459) = 1.4000 + 1.163192 = \mathbf{+2.563192} \approx \mathbf{+2.5632}$

**3. Backward Pass: Case B (Truncation with Critic Bootstrap, $d_5 = 0$)**
- $\hat{A}_{4, B}^{\text{GAE}} = \delta_{4, B}^V = \mathbf{+1.9000}$
- $\hat{A}_{3, B}^{\text{GAE}} = -0.6500 + 0.8550 \times (1.9000) = -0.6500 + 1.624500 = \mathbf{+0.974500} \approx \mathbf{+0.9745}$
- $\hat{A}_{2, B}^{\text{GAE}} = 3.3750 + 0.8550 \times (0.974500) = 3.3750 + 0.833198 = \mathbf{+4.208198} \approx \mathbf{+4.2082}$
- $\hat{A}_{1, B}^{\text{GAE}} = -1.0500 + 0.8550 \times (4.208198) = -1.0500 + 3.598009 = \mathbf{+2.548009} \approx \mathbf{+2.5480}$
- $\hat{A}_{0, B}^{\text{GAE}} = 1.4000 + 0.8550 \times (2.548009) = 1.4000 + 2.178548 = \mathbf{+3.578548} \approx \mathbf{+3.5785}$

**4. Comparative Difference and Theoretical Shockwave Propagation**
Because the backward GAE recurrence is linear:
$$\Delta \hat{A}_t \triangleq \hat{A}_{t, B}^{\text{GAE}} - \hat{A}_{t, A}^{\text{GAE}} = (\delta_{t, B}^V - \delta_{t, A}^V) + \gamma \lambda \Delta \hat{A}_{t+1}$$
Since $\delta_{t, B}^V - \delta_{t, A}^V = 0$ for all $t < 4$, the single boundary perturbation $\Delta \delta_4 = \gamma V(S_5) = 1.9000$ propagates backward attenuated by $(\gamma \lambda)^{4 - t}$:
$$\Delta \hat{A}_t = 1.9000 \times (0.8550)^{4 - t}$$

| Step $t$ | Case A GAE $\hat{A}_{t, A}$ | Case B GAE $\hat{A}_{t, B}$ | Empirical Difference $\Delta \hat{A}_t$ | Theoretical Prediction $1.9 \times (0.8550)^{4-t}$ | Exact Match? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 4$** | $0.0000$ | $+1.9000$ | $\mathbf{+1.9000}$ | $1.9000 \times 1.0000 = \mathbf{+1.9000}$ | $\checkmark$ Exact |
| **$t = 3$** | $-0.6500$ | $+0.9745$ | $\mathbf{+1.6245}$ | $1.9000 \times 0.8550^1 = \mathbf{+1.6245}$ | $\checkmark$ Exact |
| **$t = 2$** | $+2.8193$ | $+4.2082$ | $\mathbf{+1.3889}$ | $1.9000 \times 0.8550^2 = \mathbf{+1.3889}$ | $\checkmark$ Exact |
| **$t = 1$** | $+1.3605$ | $+2.5480$ | $\mathbf{+1.1876}$ | $1.9000 \times 0.8550^3 = \mathbf{+1.1876}$ | $\checkmark$ Exact |
| **$t = 0$** | $+2.5632$ | $+3.5785$ | $\mathbf{+1.0154}$ | $1.9000 \times 0.8550^4 = \mathbf{+1.0154}$ | $\checkmark$ Exact |

#### Algorithmic Takeaway for Modern PPO
Failing to distinguish between termination and truncation causes severe learning distortion:
- If a timeout is treated as a terminal state ($d=1$), the algorithm erroneously penalizes actions taken prior to timeout by assuming the agent collapsed to value $0$.
- If a terminal failure is treated as a truncation ($d=0$), the algorithm erroneously credits actions prior to death by bootstrapping the critic's estimate of a phantom post-mortem future!
- Standard implementations (e.g. CleanRL) strictly maintain:
  $$\delta_t^V = R_{t+1} + \gamma V(S_{t+1}) (1 - d_{\text{terminated}}) - V(S_t)$$
  $$\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda (1 - d_{\text{terminated}}) \hat{A}_{t+1}^{\text{GAE}}$$
  ensuring true terminations reset bootstrapping, while timeouts bootstrap $V(S_{t+1})$ without propagating advantages across episode resets.

---

## 7. Deep Learning Connection & Modern Applications

### 1. The Universal Advantage Estimator: PPO Standard Hyperparameters
GAE with $\gamma = 0.99, \lambda = 0.95$ is the near-universal standard across all modern policy gradient systems:
- **PPO (Schulman et al., 2017):** Normalizes GAE advantages per mini-batch: $\hat{A}_t^{\text{norm}} = (\hat{A}_t^{\text{GAE}} - \mu_{\hat{A}}) / (\sigma_{\hat{A}} + \epsilon)$, a critical stability trick preventing gradient magnitude explosions on early training.
- **OpenAI Baselines / Stable-Baselines3:** Every production PPO implementation (coaching robots, playing Atari, fine-tuning LLMs) uses GAE with the identical $(\gamma, \lambda) = (0.99, 0.95)$ defaults because they hit the empirical sweet spot between TD bias and Monte Carlo variance.
- **Why $\lambda = 0.95$ and not 1.0?** Setting $\lambda = 1$ (pure Monte Carlo) produces gradients with $\sim 10\times$ higher variance on continuous control tasks like Ant-v4 ($T = 1000$ steps). Setting $\lambda = 0.95$ decays the contribution of transitions more than 60 steps into the future by $0.95^{60} \approx 0.046$ — empirically erasing noise while retaining 95%+ of the bias reduction from bootstrapping.

### 2. Token-Level GAE in RLHF (InstructGPT, TRL, Tulu)
GAE maps directly onto long autoregressive language model sequences:
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
Applied token-by-token with $T \in [512, 4096]$ tokens. Setting $\lambda = 0.95$ prevents the large terminal reward bonus (correct final answer: $r_T = +1$) from corrupting gradient updates on tokens 1–100, which are early reasoning setup tokens and should not receive large credit for a correct final mathematical step.

### 3. GRPO: Eliminating the Critic via Group Baselines (DeepSeek, 2025)
**Group Relative Policy Optimization** replaces the learned critic $V_\phi$ entirely by using sample-based baselines:
$$\hat{A}_i = \frac{r_i - \operatorname{mean}_{j=1}^G r_j}{\operatorname{std}_{j=1}^G r_j}$$
This eliminates GAE's learned critic dependency (saving $\sim$40% of compute), but requires $G \geq 8$ rollouts per prompt to produce stable baselines. Used in DeepSeek-R1-Zero to achieve math reasoning capability purely from RL, without any supervised fine-tuning warm start.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of Part 5 GAE hand calculations:
   - TD residuals $\boldsymbol{\delta}^V = [1.6000, 2.5000, 0.0000]$
   - GAE advantages $\hat{\mathbf{A}}^{\text{GAE}} = [3.4000, 2.5000, 0.0000]$ matching PyTorch/NumPy to $< 10^{-14}$.
2. Benchmark of empirical variance and policy gradient accuracy across $\lambda \in \{0.0, 0.5, 0.8, 0.95, 1.0\}$.
3. Production-grade, vectorized backward GAE implementation used in PPO.
4. Complete test harness validating all 5 solved numerical illustrations from Section 6 (telescoping sum, backward recursion, lambda parameter sweep, PPO batch normalization, and episode termination vs truncation shockwaves).

See implementation in:
[`11_reinforcement_learning/code/15_generalized_advantage_estimation.py`](./code/15_generalized_advantage_estimation.py)
