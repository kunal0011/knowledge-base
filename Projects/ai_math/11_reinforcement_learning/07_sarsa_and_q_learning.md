# Module 11.7: On-Policy vs. Off-Policy Control: SARSA, Q-Learning & Expected SARSA

---

## 1. Intuition & 101 Motivation

In Chapter 11.6, we studied Temporal-Difference learning for **policy evaluation** (estimating $V^\pi(s)$ for a fixed policy $\pi$). To solve the full reinforcement learning problem—finding the **optimal policy** $\pi^*$ without a transition model—we must extend TD methods to estimate **action-values** $Q(s, a)$.

When moving from prediction to control, a fundamental fork in algorithm design emerges:
1. **On-Policy Control (SARSA):** The agent learns the value of the policy it is currently executing, *including its exploratory mistakes*. If the agent takes $\epsilon$-greedy random exploratory actions, SARSA evaluates the true return of that exploratory policy.
2. **Off-Policy Control (Q-Learning):** The agent learns the optimal policy $Q^*(s, a)$ directly, *regardless of how exploratory or suboptimal its behavior policy is*.
3. **Expected SARSA:** Bridges both paradigms by computing the exact expectation over all next actions under policy $\pi$, eliminating target sample variance.

```
       SARSA (On-Policy)              Q-Learning (Off-Policy)           Expected SARSA
      [Next action A' sampled]         [Max over all actions]       [Expectation under pi]
               (S, A)                          (S, A)                       (S, A)
                 |                               |                            |
                 R                               R                            R
                 |                               |                            |
                (S')                            (S')                         (S')
                 |                             / | \                        / | \
                (A')                         (a)(a)(a)                    (a)(a)(a)
                                            \___max___/                  \__weighted_/
```

---

## 2. Rigorous Mathematical Formulation

Let $\mathcal{S}$ be the finite state space, $\mathcal{A}$ the finite action space, $\gamma \in [0, 1)$ the discount factor, and $\alpha \in (0, 1]$ the step-size parameter.

### 2.1 SARSA: On-Policy TD Control

SARSA derives its name from the tuple of experience transitions:
$$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$$

The agent is in state $S_t$, takes action $A_t$, observes reward $R_{t+1}$ and next state $S_{t+1}$, and then samples its **actual next action** $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$ from its behavior policy (e.g., $\epsilon$-greedy).

#### Update Equation:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

The TD error is:
$$\delta_t^{\text{SARSA}} \triangleq R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$

#### Convergence Condition (GLIE: Greedy in the Limit with Infinite Exploration):
If all state-action pairs $(s, a)$ are visited infinitely often, and the exploration parameter satisfies $\epsilon_t \to 0$ as $t \to \infty$, SARSA converges almost surely to the optimal action-value function $Q^*$ and optimal policy $\pi^*$.

---

### 2.2 Q-Learning: Off-Policy TD Control (Watkins, 1989)

Q-Learning decouples the **behavior policy** $b(a \mid s)$ (which generates transitions) from the **target policy** $\pi(a \mid s) = \arg \max_a Q(s, a)$ (which is evaluated and optimized).

#### Update Equation:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

The TD error is:
$$\delta_t^{\text{Q}} \triangleq R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)$$

#### Theorem: Convergence of Q-Learning (Watkins & Dayan, 1992)
Let the MDP have finite state and action spaces and bounded rewards. Suppose all state-action pairs $(s, a)$ are visited infinitely often, and the step sizes $\alpha_t(s, a)$ satisfy the Robbins-Monro conditions:
$$\sum_{t=1}^\infty \alpha_t(s, a) = \infty \quad \text{and} \quad \sum_{t=1}^\infty \alpha_t^2(s, a) < \infty \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A}$$
Then, regardless of the behavior policy $b$ used to generate actions (provided $b(a \mid s) > 0$ for all $(s, a)$), $Q_t(s, a)$ converges almost surely to the unique optimal action-value function $Q^*(s, a)$:
$$\mathbb{P} \left( \lim_{t \to \infty} Q_t(s, a) = Q^*(s, a) \right) = 1 \quad \forall (s, a)$$

#### Proof Outline (Bellman Optimality Contraction)
The expected update operator is the Bellman optimality operator $\mathcal{T}^*$:
$$(\mathcal{T}^* Q)(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \max_{a' \in \mathcal{A}} Q(s', a')$$
In Chapter 11.4, we proved that $\mathcal{T}^*$ is a $\gamma$-contraction in the supremum norm:
$$\|\mathcal{T}^* Q_1 - \mathcal{T}^* Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
By the stochastic approximation theorem of Jaakkola, Jordan, and Singh (1994), any asynchronous iterative process governed by a contraction mapping with Robbins-Monro step sizes converges almost surely to its unique fixed point $Q^*$. $\blacksquare$

---

### 2.3 Expected SARSA

Instead of sampling a single next action $A_{t+1}$ (like SARSA) or taking the hard maximum (like Q-learning), **Expected SARSA** computes the expectation over all possible next actions under the target policy $\pi$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

- If $\pi$ is greedy ($\pi(a \mid s) = 1$ for $a = \arg\max Q$, and $0$ otherwise), Expected SARSA becomes **identical to Q-Learning**.
- If $\pi$ is $\epsilon$-greedy, Expected SARSA accounts for exploration probability without introducing the sampling variance of choosing a single $A_{t+1}$.

---

### 2.4 Maximization Bias and Double Q-Learning (van Hasselt, 2010)

Both Q-Learning and Expected SARSA (with greedy targets) employ the maximum operator:
$$\text{Target} = R + \gamma \max_{a'} Q(S', a')$$

#### The Maximization Bias Problem
Let $X_1, X_2, \dots, X_n$ be independent random variables with true means $\mu_i = \mathbb{E}[X_i]$. If each estimate $\hat{X}_i$ is an unbiased estimator of $\mu_i$ ($\mathbb{E}[\hat{X}_i] = \mu_i$), then:
$$\mathbb{E} \left[ \max_{i} \hat{X}_i \right] \ge \max_i \mathbb{E}[\hat{X}_i] = \max_i \mu_i$$
with strict inequality whenever the distributions have overlapping support.

Because $\max$ is a convex function, **Jensen's inequality** dictates that taking the maximum over noisy estimates produces a systematic positive bias (**maximization bias**), leading Q-learning to severely overestimate values!

#### Double Q-Learning Solution
To eliminate this bias, van Hasselt proposed decoupling **action selection** from **action evaluation** using two independent value tables, $Q_A$ and $Q_B$.

With probability $0.5$, update $Q_A$:
$$A^* = \arg \max_a Q_A(S_{t+1}, a) \quad \text{(Selection)}$$
$$Q_A(S_t, A_t) \leftarrow Q_A(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_B(S_{t+1}, A^*) - Q_A(S_t, A_t) \right] \quad \text{(Evaluation)}$$

With probability $0.5$, update $Q_B$ symmetrically:
$$B^* = \arg \max_a Q_B(S_{t+1}, a)$$
$$Q_B(S_t, A_t) \leftarrow Q_B(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_A(S_{t+1}, B^*) - Q_B(S_t, A_t) \right]$$

Because $Q_B(S_{t+1}, A^*)$ is an unbiased estimate of the value of $A^*$, the expected update has **zero maximization bias**:
$$\mathbb{E} \left[ Q_B(S_{t+1}, A^*) \mid A^* \right] = Q^*(S_{t+1}, A^*)$$

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.7.1: Watkins & Dayan Theorem: Almost Sure Convergence of Q-Learning via Asynchronous Contractions

##### Problem Statement & Goal
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ be a finite MDP with bounded rewards $|R_t| \le R_{\max} < \infty$ and discount factor $\gamma \in [0, 1)$.
The tabular Q-Learning algorithm updates action-values according to:
$$Q_{t+1}(S_t, A_t) = Q_t(S_t, A_t) + \alpha_t(S_t, A_t) \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q_t(S_{t+1}, a') - Q_t(S_t, A_t) \right]$$
while all other entries $(s, a) \ne (S_t, A_t)$ remain unchanged.
We prove that under the Robbins-Monro conditions on learning rates:
$$\sum_{t=1}^\infty \alpha_t(s, a) = \infty \quad \text{and} \quad \sum_{t=1}^\infty \alpha_t^2(s, a) < \infty \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A}$$
the action-value table $Q_t$ converges almost surely to the unique optimal action-value function $Q^*$:
$$\mathbb{P}\left( \lim_{t \to \infty} Q_t(s, a) = Q^*(s, a) \right) = 1, \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A}$$

##### Explicit Assumptions
1. Finite state and action spaces $|\mathcal{S}| < \infty, |\mathcal{A}| < \infty$.
2. All state-action pairs $(s, a)$ are visited infinitely often.
3. The learning rate schedule satisfies the Robbins-Monro conditions for all $(s, a)$.
4. Discount factor satisfies $\gamma \in [0, 1)$.

##### Underlying Intuition
Q-learning is an asynchronous stochastic approximation of the Bellman Optimality Operator $\mathcal{T}^*$. In Chapter 11.4, we proved that $\mathcal{T}^*$ is a strict $\gamma$-contraction in the $L_\infty$ norm with unique fixed point $Q^*$. The stochastic update replaces the expected Bellman backup with a single sample. By Robbins-Monro conditions, the cumulative step sizes are large enough to overcome any initial condition, while the sum of squared step sizes is small enough to damp out the sample variance to zero, guaranteeing that the noise disappears and the contraction pulls the estimates to $Q^*$.

##### End-to-End Mathematical Derivation

**Step 1: The Error Dynamic Equation**
Define the error vector $\Delta_t \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$ with entries:
$$\Delta_t(s, a) \equiv Q_t(s, a) - Q^*(s, a)$$
Recall the Bellman optimality equation for $Q^*$:
$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a') = (\mathcal{T}^* Q^*)(s, a)$$
Subtract $Q^*(S_t, A_t)$ from both sides of the Q-learning update:
$$\Delta_{t+1}(S_t, A_t) = (1 - \alpha_t) \Delta_t(S_t, A_t) + \alpha_t \left[ R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - Q^*(S_t, A_t) \right]$$
Rewrite the bracketed term by adding and subtracting $\mathcal{T}^* Q_t(S_t, A_t)$:
$$\begin{aligned}
&R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - Q^*(S_t, A_t) \\
&= \left[ (\mathcal{T}^* Q_t)(S_t, A_t) - Q^*(S_t, A_t) \right] + \left[ R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - (\mathcal{T}^* Q_t)(S_t, A_t) \right] \\
&= \left[ (\mathcal{T}^* Q_t)(S_t, A_t) - (\mathcal{T}^* Q^*)(S_t, A_t) \right] + w_t(S_t, A_t)
\end{aligned}$$
where $w_t(S_t, A_t) \equiv R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - (\mathcal{T}^* Q_t)(S_t, A_t)$ is the zero-mean stochastic noise term.

**Step 2: Properties of the Noise Term $w_t$**
Condition on the filtration $\mathcal{F}_t$ (the history up to step $t$ including $S_t, A_t$):
$$\mathbb{E}\left[ w_t(S_t, A_t) \mid \mathcal{F}_t \right] = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') \mid S_t, A_t \right] - (\mathcal{T}^* Q_t)(S_t, A_t) = 0$$
Thus, $\{w_t\}$ is a Martingale Difference Sequence.
Furthermore, because rewards are bounded ($|R| \le R_{\max}$) and $\gamma < 1$, the variance of $w_t$ conditional on $\mathcal{F}_t$ is uniformly bounded:
$$\mathbb{E}\left[ w_t^2(S_t, A_t) \mid \mathcal{F}_t \right] \le C (1 + \|\Delta_t\|_\infty^2)$$
for some constant $C < \infty$.

**Step 3: Contraction Property of the Deterministic Part**
From Derivation 11.3.2, the Bellman optimality operator $\mathcal{T}^*$ is a strict $\gamma$-contraction in $L_\infty$ norm:
$$|(\mathcal{T}^* Q_t)(S_t, A_t) - (\mathcal{T}^* Q^*)(S_t, A_t)| \le \|\mathcal{T}^* Q_t - \mathcal{T}^* Q^*\|_\infty \le \gamma \|Q_t - Q^*\|_\infty = \gamma \|\Delta_t\|_\infty$$

**Step 4: Application of the Jaakkola-Jordan-Singh (1994) Theorem**
Consider the general stochastic iterative process on $\mathbb{R}^d$:
$$\Delta_{t+1}(i) = (1 - \alpha_t(i)) \Delta_t(i) + \alpha_t(i) F_t(i)$$
The Jaakkola, Jordan, and Singh theorem establishes that $\Delta_t \xrightarrow{a.s.} \mathbf{0}$ if:
1. $\sum_t \alpha_t(i) = \infty$ and $\sum_t \alpha_t^2(i) < \infty$ almost surely.
2. $\|\mathbb{E}[F_t \mid \mathcal{F}_t]\|_\infty \le \gamma \|\Delta_t\|_\infty$ with $\gamma < 1$.
3. $\operatorname{Var}(F_t(i) \mid \mathcal{F}_t) \le C (1 + \|\Delta_t\|_\infty^2)$.

All three conditions are strictly met:
1. Robbins-Monro conditions are assumed.
2. $\|\mathbb{E}[(\mathcal{T}^* Q_t - \mathcal{T}^* Q^*) + w_t \mid \mathcal{F}_t]\|_\infty \le \gamma \|\Delta_t\|_\infty + 0 = \gamma \|\Delta_t\|_\infty$.
3. Variance of $w_t$ is bounded by $C(1 + \|\Delta_t\|_\infty^2)$.

Therefore:
$$\lim_{t \to \infty} \|\Delta_t\|_\infty = 0 \quad \text{almost surely}$$
which implies $Q_t(s, a) \xrightarrow{a.s.} Q^*(s, a)$ for all $(s, a) \in \mathcal{S} \times \mathcal{A}$. $\blacksquare$

---

#### Derivation 11.7.2: Mathematical Proof of Maximization Bias via Jensen's Inequality and Order Statistics

##### Problem Statement & Goal
Let $X_1, X_2, \dots, X_m$ be $m \ge 2$ independent random variables with true means $\mu_i \equiv \mathbb{E}[X_i]$.
Let $\hat{X}_i$ be independent, unbiased estimators of $\mu_i$: $\mathbb{E}[\hat{X}_i] = \mu_i$, each having non-zero variance $\sigma_i^2 > 0$.
We prove:
1. By Jensen's inequality and convexity of the maximum function:
   $$\mathbb{E}\left[ \max_{1 \le i \le m} \hat{X}_i \right] \ge \max_{1 \le i \le m} \mathbb{E}[\hat{X}_i] = \max_{1 \le i \le m} \mu_i$$
2. The inequality is strict whenever there is a non-zero probability that the argmax over estimates differs from the argmax over true means.
3. For $m$ independent standard Gaussian estimators $\hat{X}_i \sim \mathcal{N}(0, \sigma^2)$, the expected overestimation bias scales asymptotically as:
   $$\mathbb{E}\left[ \max_{1 \le i \le m} \hat{X}_i \right] \sim \sigma \sqrt{2 \ln m}$$

##### Explicit Assumptions
1. Estimators $\hat{X}_i$ have finite second moments: $\mathbb{E}[\hat{X}_i^2] < \infty$.
2. The maximum operator acts on $m \ge 2$ distinct actions.

##### Underlying Intuition
The function $g(\mathbf{x}) = \max(x_1, \dots, x_m)$ is convex because it is the upper envelope of linear functions. Whenever random noise is added to the inputs of a convex function, Jensen's inequality guarantees that the expected output is greater than or equal to the function evaluated at the expected inputs. If any single action receives a lucky positive noise spike, the $\max$ operator latches onto that lucky outlier, systematically pulling the average upward.

##### End-to-End Mathematical Derivation

**Step 1: Convexity of the Maximum Operator**
Let $g: \mathbb{R}^m \to \mathbb{R}$ be defined by $g(\mathbf{x}) = \max_{1 \le i \le m} x_i$.
Let $\mathbf{x}, \mathbf{y} \in \mathbb{R}^m$ and $\lambda \in [0, 1]$.
For any specific coordinate $k \in \{1, \dots, m\}$:
$$\lambda x_k + (1 - \lambda) y_k \le \lambda \max_i x_i + (1 - \lambda) \max_j y_j = \lambda g(\mathbf{x}) + (1 - \lambda) g(\mathbf{y})$$
Since this holds for every coordinate $k$, it holds for the maximum over $k$:
$$g(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) = \max_{1 \le k \le m} \left[ \lambda x_k + (1 - \lambda) y_k \right] \le \lambda g(\mathbf{x}) + (1 - \lambda) g(\mathbf{y})$$
Thus, $g$ is a convex function on $\mathbb{R}^m$.

**Step 2: Proof of Jensen's Inequality for $g(\mathbf{X})$**
By Jensen's Inequality, for any convex function $g$ and random vector $\hat{\mathbf{X}} = (\hat{X}_1, \dots, \hat{X}_m)^\top$:
$$\mathbb{E}\left[ g(\hat{\mathbf{X}}) \right] \ge g\left( \mathbb{E}[\hat{\mathbf{X}}] \right)$$
Substitute $g(\mathbf{x}) = \max_i x_i$:
$$\mathbb{E}\left[ \max_{1 \le i \le m} \hat{X}_i \right] \ge \max_{1 \le i \le m} \mathbb{E}[\hat{X}_i] = \max_{1 \le i \le m} \mu_i$$

**Step 3: Strictness of the Maximization Bias**
Consider the two-action case $m = 2$ with $\mu_1 = \mu_2 = \mu$.
Let $\hat{X}_1 = \mu + \epsilon_1$ and $\hat{X}_2 = \mu + \epsilon_2$, where $\epsilon_1, \epsilon_2$ are independent zero-mean random variables with variance $\sigma^2 > 0$.
Recall the identity $\max(a, b) = \frac{a + b + |a - b|}{2}$:
$$\max(\hat{X}_1, \hat{X}_2) = \frac{\hat{X}_1 + \hat{X}_2 + |\hat{X}_1 - \hat{X}_2|}{2} = \mu + \frac{\epsilon_1 + \epsilon_2}{2} + \frac{|\epsilon_1 - \epsilon_2|}{2}$$
Taking expectations:
$$\mathbb{E}\left[ \max(\hat{X}_1, \hat{X}_2) \right] = \mu + 0 + \frac{1}{2} \mathbb{E}\left[ |\epsilon_1 - \epsilon_2| \right]$$
Because $\epsilon_1, \epsilon_2$ are independent with non-zero variance, the random variable $D = \epsilon_1 - \epsilon_2$ is non-degenerate with variance $2\sigma^2 > 0$.
The absolute value of a non-zero-variance random variable has strictly positive expectation:
$$\mathbb{E}\left[ |D| \right] > 0$$
Therefore:
$$\mathbb{E}\left[ \max(\hat{X}_1, \hat{X}_2) \right] = \mu + \frac{1}{2} \mathbb{E}[|D|] > \mu = \max(\mu_1, \mu_2)$$
The bias is strictly positive!

**Step 4: Asymptotic Scaling with Action Count $m$**
For $m$ independent standard Gaussian estimators $\hat{X}_i \sim \mathcal{N}(0, \sigma^2)$, let $Z_i = \hat{X}_i / \sigma \sim \mathcal{N}(0, 1)$.
Using the sub-Gaussian moment generating function $\mathbb{E}[e^{s Z_i}] = e^{s^2 / 2}$:
By Chernoff bounding:
$$\mathbb{E}\left[ \max_{1 \le i \le m} Z_i \right] \le \frac{\ln m}{s} + \frac{s}{2}$$
Setting optimal $s = \sqrt{2 \ln m}$:
$$\mathbb{E}\left[ \max_{1 \le i \le m} Z_i \right] \le \sqrt{2 \ln m}$$
From extreme value theory, the asymptotic limit satisfies:
$$\lim_{m \to \infty} \frac{\mathbb{E}[\max_{1 \le i \le m} \hat{X}_i]}{\sigma \sqrt{2 \ln m}} = 1$$
Thus, as the action space size $m$ grows, standard Q-learning's overestimation bias grows as $\mathcal{O}(\sigma \sqrt{\ln m})$! $\blacksquare$

---

#### Derivation 11.7.3: Target Variance Reduction in Expected SARSA vs. SARSA

##### Problem Statement & Goal
Let $(S_t, A_t, R_{t+1}, S_{t+1})$ be an observed transition.
Define the target variables:
1. **SARSA Target:** $Y_{\text{SARSA}} \equiv R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$, where $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$
2. **Expected SARSA Target:** $Y_{\text{Exp}} \equiv R_{t+1} + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a')$
We prove:
1. Unbiasedness equivalence: $\mathbb{E}[Y_{\text{SARSA}} \mid S_t, A_t, R_{t+1}, S_{t+1}] = Y_{\text{Exp}}$.
2. Conditional target variance:
   $$\operatorname{Var}(Y_{\text{SARSA}} \mid S_t, A_t, R_{t+1}, S_{t+1}) = \gamma^2 \sum_{a' \in \mathcal{A}} \pi(a' \mid S_{t+1}) \left( Q(S_{t+1}, a') - \bar{Q}(S_{t+1}) \right)^2 \ge 0$$
   while $\operatorname{Var}(Y_{\text{Exp}} \mid S_t, A_t, R_{t+1}, S_{t+1}) \equiv 0$, proving that Expected SARSA completely removes the action-selection sampling variance.

##### Explicit Assumptions
1. Discrete action space $|\mathcal{A}| < \infty$.
2. Target policy $\pi(\cdot \mid S_{t+1})$ is known and evaluable.

##### Underlying Intuition
In SARSA, after transitioning to $S_{t+1}$, the agent rolls a die to pick next action $A_{t+1}$ from $\pi$. That random roll adds pure sampling variance to the target. In Expected SARSA, since the policy probabilities $\pi(a' \mid S_{t+1})$ and the table values $Q(S_{t+1}, a')$ are already fully stored in memory, there is no need to roll a die! We can compute the exact weighted average analytically with zero added noise.

##### End-to-End Mathematical Derivation

**Step 1: Conditional Expectation Equivalence**
Let $\mathcal{H}_{t+1} = (S_t, A_t, R_{t+1}, S_{t+1})$ denote the transition history.
Evaluate the conditional expectation of the SARSA target with respect to the sampling of $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$:
$$\begin{aligned}
\mathbb{E}\left[ Y_{\text{SARSA}} \;\middle|\; \mathcal{H}_{t+1} \right] &= \mathbb{E}\left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) \;\middle|\; \mathcal{H}_{t+1} \right] \\
&= R_{t+1} + \gamma \mathbb{E}_{A_{t+1} \sim \pi}\left[ Q(S_{t+1}, A_{t+1}) \;\middle|\; S_{t+1} \right] \\
&= R_{t+1} + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') \\
&= Y_{\text{Exp}}
\end{aligned}$$
This proves that $Y_{\text{Exp}}$ has identically the same conditional mean as $Y_{\text{SARSA}}$.

**Step 2: Conditional Variance of the SARSA Target**
Evaluate the conditional variance of $Y_{\text{SARSA}}$:
$$\operatorname{Var}\left( Y_{\text{SARSA}} \;\middle|\; \mathcal{H}_{t+1} \right) = \operatorname{Var}\left( R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) \;\middle|\; \mathcal{H}_{t+1} \right)$$
Because $R_{t+1}$ and $S_{t+1}$ are fixed constants within $\mathcal{H}_{t+1}$:
$$\operatorname{Var}\left( Y_{\text{SARSA}} \;\middle|\; \mathcal{H}_{t+1} \right) = \gamma^2 \operatorname{Var}_{A_{t+1} \sim \pi}\left( Q(S_{t+1}, A_{t+1}) \;\middle|\; S_{t+1} \right)$$
Let $\bar{Q}(S_{t+1}) \equiv \sum_{a'} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a')$. Expanding the variance:
$$\operatorname{Var}_{A'}\left( Q(S_{t+1}, A') \right) = \sum_{a' \in \mathcal{A}} \pi(a' \mid S_{t+1}) \left[ Q(S_{t+1}, a') - \bar{Q}(S_{t+1}) \right]^2$$
Therefore:
$$\operatorname{Var}\left( Y_{\text{SARSA}} \;\middle|\; \mathcal{H}_{t+1} \right) = \gamma^2 \sum_{a' \in \mathcal{A}} \pi(a' \mid S_{t+1}) \left[ Q(S_{t+1}, a') - \bar{Q}(S_{t+1}) \right]^2 \ge 0$$

**Step 3: Variance of the Expected SARSA Target**
Notice that $Y_{\text{Exp}} = R_{t+1} + \gamma \bar{Q}(S_{t+1})$ is completely deterministic given $\mathcal{H}_{t+1}$:
$$\operatorname{Var}\left( Y_{\text{Exp}} \;\middle|\; \mathcal{H}_{t+1} \right) = 0$$

**Step 4: Total Variance Comparison via Eve's Law**
By Eve's Law of Total Variance over the joint state-action transition distribution:
$$\operatorname{Var}(Y_{\text{SARSA}}) = \operatorname{Var}(Y_{\text{Exp}}) + \mathbb{E}\left[ \operatorname{Var}(Y_{\text{SARSA}} \mid \mathcal{H}_{t+1}) \right]$$
Since $\mathbb{E}[\operatorname{Var}(Y_{\text{SARSA}} \mid \mathcal{H}_{t+1})] \ge 0$:
$$\operatorname{Var}(Y_{\text{SARSA}}) \ge \operatorname{Var}(Y_{\text{Exp}})$$
with strict inequality whenever the action-values at successor states are non-identical ($Q(s', a_1) \ne Q(s', a_2)$) and policy $\pi$ is stochastic ($\epsilon > 0$). Expected SARSA purges all action-selection noise without incurring any bias penalty. $\blacksquare$

---

## 3. Geometric & Physical Interpretation: The Cliff Walking Dilemma

Consider the canonical **Cliff Walking** gridworld:

```
+---+---+---+---+---+---+---+---+---+---+---+---+
| S |   |   |   |   |   |   |   |   |   |   | G |
+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+
| S | C | L | I | F | F |   | C | L | I | F | G |
+---+---+---+---+---+---+---+---+---+---+---+---+
```
- Step reward: $-1.0$ for normal transitions.
- Falling into the Cliff: $-100.0$ and resets agent to Start $S$.

### Geometric Separation of Policies:
- **Q-Learning** learns the true optimal path $Q^*$: the path that walks right along the cliff edge. It assumes greedy execution. However, during training with $\epsilon$-greedy exploration ($\epsilon = 0.1$), the agent frequently slips off the cliff due to random exploration, suffering poor online performance (average reward $\approx -45$).
- **SARSA** is on-policy: it incorporates the agent's actual $\epsilon$-greedy execution into its value estimates. It learns that being near the cliff is hazardous due to the $10\%$ probability of accidental suicide. Geometrically, SARSA's trajectory bends away from the cliff edge, discovering the **safe path** through the upper perimeter (average reward $\approx -17$).

```
^ Distance from Cliff
|
|   SARSA Safe Path (On-Policy: anticipates random exploration)
|   ------------------------------------------------------------> Goal
|
|   Q-Learning Optimal Path (Off-Policy: ignores exploration risk)
|   ............................................................> Goal
|   [========================= CLIFF =========================]
+------------------------------------------------------------------>
```

---

## 4. Real-World Analogy: Formula 1 Racing vs. Autonomous Safety

- **Q-Learning is the Formula 1 Telemetry Analysis:**
  It calculates the absolute fastest possible lap time assuming the driver hits the racing line with mathematical perfection. It ignores human error, fatigue, or wet patches off-line.
- **SARSA is the Pragmatic Race Strategy:**
  It observes how the human driver actually drives under fatigue and weather. If the driver frequently locks the brakes into Turn 3 when trying to take the aggressive curb, SARSA advises: "Brake 5 meters earlier; your execution isn't perfect, and going off-track costs 30 seconds."

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 2-State, 2-Action MDP
Let us compare **SARSA**, **Q-Learning**, and **Expected SARSA** on the exact same transition!

**States:** $S_1$ and $S_2$
**Actions:** $a_1$ (Action 1) and $a_2$ (Action 2)
**Hyperparameters:**
- Discount factor: $\gamma = 0.8000$
- Learning rate: $\alpha = 0.5000$
- Exploration rate: $\epsilon = 0.2000$

**Initial Action-Value Table:**
$$Q(S_1, a_1) = 2.0000, \quad Q(S_1, a_2) = 1.0000$$
$$Q(S_2, a_1) = 6.0000, \quad Q(S_2, a_2) = 10.0000$$

**Transition Observed:**
At time $t$, agent is in $S_1$ and takes action $A_t = a_1$.
- Immediate reward observed: $R_{t+1} = 4.0000$
- Next state reached: $S_{t+1} = S_2$
- Next action sampled from behavior policy: $A_{t+1} = a_1$ (an exploratory move, since $Q(S_2, a_1) < Q(S_2, a_2)$!).

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Walkthrough Value |
| :--- | :--- | :--- |
| $S_t, A_t$ | Current State-Action Pair | $S_1, a_1$ |
| $R_{t+1}$ | Observed Reward | $4.0000$ |
| $S_{t+1}$ | Successor State | $S_2$ |
| $A_{t+1}$ | Sampled Next Action | $a_1$ (suboptimal choice) |
| $Q(S_2, a_1)$ | Action-Value for $a_1$ at $S_2$ | $6.0000$ |
| $Q(S_2, a_2)$ | Action-Value for $a_2$ at $S_2$ | $10.0000$ (greedy choice) |
| $\pi(a \mid S_2)$ | $\epsilon$-greedy target policy probabilities | $\pi(a_2) = 1 - \frac{0.2}{2} = 0.90$, $\pi(a_1) = \frac{0.2}{2} = 0.10$ |

---

### 5.3 Method 1: SARSA Update Hand Arithmetic

SARSA targets the **sampled** action $A_{t+1} = a_1$:
$$\text{Target}_{\text{SARSA}} = R_{t+1} + \gamma Q(S_2, A_{t+1}) = 4.0000 + 0.8000 \times Q(S_2, a_1)$$
$$= 4.0000 + 0.8000 \times 6.0000 = 4.0000 + 4.8000 = \mathbf{8.8000}$$

TD Error:
$$\delta_{\text{SARSA}} = \text{Target}_{\text{SARSA}} - Q(S_1, a_1) = 8.8000 - 2.0000 = \mathbf{6.8000}$$

Updated Value:
$$Q_{\text{new}}^{\text{SARSA}}(S_1, a_1) = Q(S_1, a_1) + \alpha \delta_{\text{SARSA}} = 2.0000 + 0.5000 \times 6.8000 = 2.0000 + 3.4000 = \mathbf{5.4000}$$

---

### 5.4 Method 2: Q-Learning Update Hand Arithmetic

Q-Learning targets the **maximum** over all actions at $S_2$:
$$\max_{a'} Q(S_2, a') = \max(Q(S_2, a_1), Q(S_2, a_2)) = \max(6.0000, 10.0000) = \mathbf{10.0000}$$

$$\text{Target}_{\text{Q}} = R_{t+1} + \gamma \max_{a'} Q(S_2, a') = 4.0000 + 0.8000 \times 10.0000 = 4.0000 + 8.0000 = \mathbf{12.0000}$$

TD Error:
$$\delta_{\text{Q}} = \text{Target}_{\text{Q}} - Q(S_1, a_1) = 12.0000 - 2.0000 = \mathbf{10.0000}$$

Updated Value:
$$Q_{\text{new}}^{\text{Q}}(S_1, a_1) = Q(S_1, a_1) + \alpha \delta_{\text{Q}} = 2.0000 + 0.5000 \times 10.0000 = 2.0000 + 5.0000 = \mathbf{7.0000}$$

---

### 5.5 Method 3: Expected SARSA Hand Arithmetic

Under $\epsilon$-greedy exploration with $\epsilon = 0.20$ and $|\mathcal{A}| = 2$:
- Greedy action is $a_2$ (since $Q(S_2, a_2) = 10 > 6$):
  $$\pi(a_2 \mid S_2) = 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} = 1 - 0.20 + 0.10 = \mathbf{0.9000}$$
- Non-greedy action is $a_1$:
  $$\pi(a_1 \mid S_2) = \frac{\epsilon}{|\mathcal{A}|} = \frac{0.20}{2} = \mathbf{0.1000}$$

Expected Next-State Value:
$$\mathbb{E}_{\pi}[Q(S_2, \cdot)] = \pi(a_1 \mid S_2) Q(S_2, a_1) + \pi(a_2 \mid S_2) Q(S_2, a_2)$$
$$= (0.1000 \times 6.0000) + (0.9000 \times 10.0000) = 0.6000 + 9.0000 = \mathbf{9.6000}$$

TD Target:
$$\text{Target}_{\text{Exp}} = R_{t+1} + \gamma \mathbb{E}_{\pi}[Q(S_2, \cdot)] = 4.0000 + 0.8000 \times 9.6000 = 4.0000 + 7.6800 = \mathbf{11.6800}$$

TD Error:
$$\delta_{\text{Exp}} = \text{Target}_{\text{Exp}} - Q(S_1, a_1) = 11.6800 - 2.0000 = \mathbf{9.6800}$$

Updated Value:
$$Q_{\text{new}}^{\text{Exp}}(S_1, a_1) = Q(S_1, a_1) + \alpha \delta_{\text{Exp}} = 2.0000 + 0.5000 \times 9.6800 = 2.0000 + 4.8400 = \mathbf{6.8400}$$

---

### 5.6 Summary Comparison Visual Grid

| Method | Next-Action Philosophy | Target Formula | Target Value | TD Error $\delta$ | Updated $Q(S_1, a_1)$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SARSA** | Sampled $A_{t+1} = a_1$ (On-policy) | $4.0 + 0.8 \times 6.0$ | $\mathbf{8.8000}$ | $\mathbf{6.8000}$ | $\mathbf{5.4000}$ |
| **Q-Learning** | Maximum $\max_a Q$ (Off-policy) | $4.0 + 0.8 \times 10.0$ | $\mathbf{12.0000}$ | $\mathbf{10.0000}$ | $\mathbf{7.0000}$ |
| **Expected SARSA** | Expectation $\sum \pi(a) Q$ | $4.0 + 0.8 \times 9.6$ | $\mathbf{11.6800}$ | $\mathbf{9.6800}$ | $\mathbf{6.8400}$ |

Every single arithmetic operation is crystal clear, rigorous, and exact.

---

## 6. Solved Illustrations

### Illustration 1: Maximization Bias (Sutton Example 6.7)
**Problem:**
Consider a 2-state MDP: State $A$ and State $B$.
- From $A$, action `right` terminates with reward $0$.
- From $A$, action `left` transitions to $B$ with reward $0$.
- From $B$, there are $10$ available actions, each of which terminates with a stochastic reward drawn independently from a Gaussian distribution:
  $$R \sim \mathcal{N}(-0.1, 1.0)$$
Thus, the true expected value of entering $B$ is $\mathbb{E}[R] = -0.1 < 0$. The true optimal policy from $A$ is to choose `right` (return $0 > -0.1$).

Show why Q-Learning initially prefers `left`, and prove that Double Q-Learning correctly identifies `right`.

**Solution:**
1. **Under Standard Q-Learning:**
   At state $B$, the target for state $A$ is:
   $$\text{Target}(A, \text{left}) = 0 + \gamma \max_{a \in \{1,\dots,10\}} Q(B, a)$$
   Even though each $Q(B, a)$ has true mean $-0.1$, the maximum of 10 noisy independent estimates $\max_{i=1}^{10} X_i$ has expected value:
   $$\mathbb{E} \left[ \max_{1 \le i \le 10} \mathcal{N}(-0.1, 1.0) \right] \approx -0.1 + \frac{\sqrt{2 \ln(10)}}{\sqrt{\pi}} \approx +1.43 > 0!$$
   Consequently, Q-learning creates the illusion that moving to $B$ has high positive value, causing the agent to choose `left` up to $70\text{--}80\%$ of the time!

2. **Under Double Q-Learning:**
   Action selection and evaluation are decoupled across $Q_1$ and $Q_2$:
   $$a^* = \arg \max_a Q_1(B, a)$$
   $$\text{Target} = Q_2(B, a^*)$$
   Since $Q_2$ is independent of the noise that caused $Q_1(B, a^*)$ to be high:
   $$\mathbb{E} \left[ Q_2(B, a^*) \mid a^* \right] = \mathbb{E}[R] = -0.1 < 0$$
   Double Q-Learning produces an expected target of $-0.1$, so $Q(A, \text{left}) < Q(A, \text{right}) = 0$. The agent immediately learns the correct optimal policy! $\blacksquare$

---

### Illustration 2: The Cliff Walking Environment: Analytical Q-Values and Trajectory Risk Comparison

**Problem:**
Consider the $4 \times 12$ Cliff Walking gridworld with Start at $(3, 0)$ and Goal at $(3, 11)$. The Cliff occupies cells $(3, 1)$ through $(3, 10)$.
Transition rules:
- Normal transitions yield reward $-1.00$.
- Stepping into the cliff yields reward $-100.00$ and teleports the agent back to Start.
- Discount factor is $\gamma = 1.00$.
- Actions: $\{\text{Up, Down, Left, Right}\}$.

1. Calculate the return of the deterministic **The Optimal Path (Cliff Edge)**: Up to $(2, 0)$, Right $\times 11$ along row $2$, Down to Goal.
2. Suppose the agent executes an $\epsilon$-greedy policy with $\epsilon = 0.10$. In row $2$ (one cell above the cliff), each step has probability $\frac{\epsilon}{4} = \frac{0.10}{4} = 0.025$ of taking action Down into the cliff. Calculate the expected return of attempting the Cliff Edge path under $\epsilon = 0.10$.
3. Compute the expected return of the **Safe Path** (Up to row 0, Right $\times 11$ along the top row, Down to Goal) under $\epsilon = 0.10$.
4. Explain why SARSA chooses the safe path while Q-Learning chooses the risky edge path.

**Solution:**

**Step 1: Deterministic Optimal Path (Zero Exploration)**
The shortest path takes:
- $1$ step Up to $(2, 0)$
- $10$ steps Right to $(2, 11)$
- $1$ step Down to $(3, 11)$ (Goal)
Total steps: $12$ steps.
Return:
$$G_{\text{optimal}} = 12 \times (-1.00) = \mathbf{-12.0000}$$

**Step 2: Expected Return of Edge Path under $\epsilon = 0.10$ Exploration**
Along the 11 steps traversing row 2 directly above the cliff, the probability of taking random action Down at any step is $p_{\text{fall}} = \frac{\epsilon}{4} = 0.025$.
The probability of completing all 11 steps without falling once is:
$$P(\text{survive}) = (1 - 0.025)^{11} = (0.975)^{11} \approx \mathbf{0.7578}$$
Probability of falling at least once:
$$P(\text{fall}) = 1 - 0.7578 = \mathbf{0.2422}$$
If the agent falls into the cliff, it incurs a $-100$ penalty and restarts from the beginning.
The expected cost per completed episode can be modeled as:
$$\mathbb{E}[G_{\text{edge}}] \approx \frac{-12.00 - P(\text{fall}) \times 100}{P(\text{survive})} \approx \frac{-12.00 - 24.22}{0.7578} = \frac{-36.22}{0.7578} \approx \mathbf{-47.80}$$
Under an exploratory behavior policy, walking next to the cliff results in an average episodic return of approximately **$-48$**!

**Step 3: Expected Return of the Safe Path under $\epsilon = 0.10$ Exploration**
The safe path travels along the top perimeter (row 0):
- $3$ steps Up from $(3, 0)$ to $(0, 0)$
- $11$ steps Right from $(0, 0)$ to $(0, 11)$
- $3$ steps Down from $(0, 11)$ to $(3, 11)$
Total nominal length: $3 + 11 + 3 = 17$ steps.
Because row 0 is 3 cells away from the cliff, an accidental exploratory action cannot reach the cliff. The only penalty incurred from exploration is a temporary 1-step detour against grid boundaries.
Expected return:
$$\mathbb{E}[G_{\text{safe}}] \approx 17 \times (-1.00) + \text{minor detour delay} \approx \mathbf{-17.50}$$

**Step 4: Algorithm Policy Separation**
- **Q-Learning** updates off-policy using $\max_a Q(s', a)$. It assumes that once it is in row 2, it will execute the greedy action ($\text{Right}$) with probability $1.0$. Thus, it evaluates the edge path at its theoretical optimum $-12.0$, blinding it to the actual risk of exploration.
- **SARSA** updates on-policy using the sampled action $A_{t+1}$. It observes the actual $-100$ penalties suffered when random exploration triggers Down into the cliff. SARSA learns $Q^{\pi_{\epsilon}}(\text{row 2}) \approx -48$, while $Q^{\pi_{\epsilon}}(\text{safe}) \approx -17.5$. Consequently, SARSA's greedy choice shifts to the **Safe Path**!

---

### Illustration 3: Double Q-Learning vs. Standard Q-Learning: Exact Step-by-Step Update Trace on Overestimation State

**Problem:**
An agent is in state $S_0$, takes action $A_0$, observes reward $R = 0.00$, and transitions to state $S'$.
Discount factor: $\gamma = 0.90$. Learning rate: $\alpha = 0.20$.
In successor state $S'$, there are 3 available actions $\{a_1, a_2, a_3\}$ whose true values are all identically zero:
$$Q^*(S', a_1) = 0.0000, \quad Q^*(S', a_2) = 0.0000, \quad Q^*(S', a_3) = 0.0000$$
The current estimates in the tables have noisy approximation errors:
- In Standard Q-learning:
  $$Q(S', a_1) = +1.5000, \quad Q(S', a_2) = -0.5000, \quad Q(S', a_3) = +0.8000$$
- In Double Q-learning (two independent tables $Q_A$ and $Q_B$):
  $$Q_A(S') = \begin{bmatrix} +1.5000 \\ -0.5000 \\ +0.8000 \end{bmatrix}, \quad Q_B(S') = \begin{bmatrix} -0.4000 \\ +0.3000 \\ -0.9000 \end{bmatrix}$$
Let the prior estimate be $Q(S_0, A_0) = 0.0000$ (and $Q_A(S_0, A_0) = 0.0000$).

1. Compute the Standard Q-learning target and updated value $Q_{\text{new}}(S_0, A_0)$.
2. Compute the Double Q-learning target and updated value $Q_{A, \text{new}}(S_0, A_0)$ when updating table $Q_A$.
3. Compare the estimation error of both updates relative to the true target value $0.0000$.

**Solution:**

**Step 1: Standard Q-Learning Update**
- Target uses the maximum of the single noisy table:
  $$\max_{a'} Q(S', a') = \max(+1.5000, -0.5000, +0.8000) = \mathbf{+1.5000} \quad (\text{Action } a_1)$$
- TD Target:
  $$\text{Target}_{\text{Q}} = R + \gamma \max_{a'} Q(S', a') = 0.0000 + 0.90 \times (+1.5000) = \mathbf{+1.3500}$$
- Value Update:
  $$Q_{\text{new}}(S_0, A_0) = Q(S_0, A_0) + \alpha \left[ \text{Target}_{\text{Q}} - Q(S_0, A_0) \right] = 0.0000 + 0.20(1.3500 - 0.0000) = \mathbf{+0.2700}$$
Standard Q-learning created an artificial positive value $+0.2700$ out of pure noise!

**Step 2: Double Q-Learning Update**
- **Action Selection** uses $Q_A$:
  $$A^* = \arg\max_{a'} Q_A(S', a') = \arg\max(+1.5000, -0.5000, +0.8000) = \mathbf{a_1}$$
- **Action Evaluation** uses the independent table $Q_B$ evaluated at $A^* = a_1$:
  $$\text{Target}_{\text{DoubleQ}} = R + \gamma Q_B(S', A^*) = 0.0000 + 0.90 \times Q_B(S', a_1) = 0.90 \times (-0.4000) = \mathbf{-0.3600}$$
- Value Update for $Q_A$:
  $$Q_{A, \text{new}}(S_0, A_0) = Q_A(S_0, A_0) + \alpha \left[ \text{Target}_{\text{DoubleQ}} - Q_A(S_0, A_0) \right] = 0.0000 + 0.20(-0.3600 - 0.0000) = \mathbf{-0.0720}$$

**Step 3: Comparison of Estimation Error**
True optimal target is $R + \gamma \max_a Q^*(S', a) = 0 + 0.9(0) = \mathbf{0.0000}$.
- Standard Q-learning error: $|+0.2700 - 0.0000| = \mathbf{+0.2700}$ (Systematic Overestimation Bias).
- Double Q-learning error: $|-0.0720 - 0.0000| = \mathbf{0.0720}$ ($73.3\%$ lower absolute error, with zero systematic positive bias).

---

### Illustration 4: Expected SARSA vs. SARSA Target Variance Quantification Across Policy Stochasticity $\epsilon$

**Problem:**
Consider a transition $(S, A, R = 1.00, S')$ with discount factor $\gamma = 0.90$.
Successor state $S'$ has two available actions with stored values:
$$Q(S', a_1) = 2.0000, \quad Q(S', a_2) = 10.0000$$
The target policy $\pi(\cdot \mid S')$ is an $\epsilon$-greedy policy where $a_2$ is the greedy action.
For exploration parameters $\epsilon \in \{0.00, 0.10, 0.50, 1.00\}$:
1. Determine action probabilities $\pi(a_1 \mid S')$ and $\pi(a_2 \mid S')$.
2. Calculate the Expected SARSA target $Y_{\text{Exp}}$.
3. Calculate the conditional target variance $\operatorname{Var}(Y_{\text{SARSA}} \mid S, A, S')$ of standard SARSA.

**Solution:**

**General Formulas:**
For $|\mathcal{A}| = 2$:
$$\pi(a_2 \mid S') = 1 - \epsilon + \frac{\epsilon}{2} = 1 - 0.5\epsilon, \quad \pi(a_1 \mid S') = \frac{\epsilon}{2} = 0.5\epsilon$$
Expected next-state value:
$$\bar{Q}(S') = \pi(a_1 \mid S') (2.0) + \pi(a_2 \mid S') (10.0)$$
Expected SARSA Target:
$$Y_{\text{Exp}} = R + \gamma \bar{Q}(S') = 1.00 + 0.90 \bar{Q}(S')$$
SARSA Target Variance (Derivation 11.7.3):
$$\operatorname{Var}(Y_{\text{SARSA}} \mid S, A, S') = \gamma^2 \left[ \pi(a_1 \mid S')(2.0 - \bar{Q}(S'))^2 + \pi(a_2 \mid S')(10.0 - \bar{Q}(S'))^2 \right]$$
Since $\operatorname{Var}_{\text{Bernoulli}}(X) = p(1-p)(x_2 - x_1)^2$:
$$\operatorname{Var}(Q(S', A')) = \pi(a_1 \mid S') \pi(a_2 \mid S') (10.0 - 2.0)^2 = 64 \pi(a_1 \mid S') \pi(a_2 \mid S')$$
$$\operatorname{Var}(Y_{\text{SARSA}} \mid S, A, S') = (0.90)^2 \times 64 \pi(a_1 \mid S') \pi(a_2 \mid S') = 51.84 \pi(a_1 \mid S') \pi(a_2 \mid S')$$

**Step-by-Step Evaluation Across $\epsilon$:**

1. **Case $\epsilon = 0.00$ (Pure Greedy / Q-Learning Equivalence):**
   - Probabilities: $\pi(a_1) = 0.00, \quad \pi(a_2) = 1.00$
   - $\bar{Q}(S') = 1.00(10.0) = 10.0000$
   - $Y_{\text{Exp}} = 1.0 + 0.9(10.0) = \mathbf{10.0000}$
   - $\operatorname{Var}(Y_{\text{SARSA}}) = 51.84 \times (0.00)(1.00) = \mathbf{0.0000}$

2. **Case $\epsilon = 0.10$ (Standard Exploration):**
   - Probabilities: $\pi(a_1) = 0.05, \quad \pi(a_2) = 0.95$
   - $\bar{Q}(S') = 0.05(2.0) + 0.95(10.0) = 0.10 + 9.50 = 9.6000$
   - $Y_{\text{Exp}} = 1.0 + 0.9(9.60) = 1.0 + 8.64 = \mathbf{9.6400}$
   - $\operatorname{Var}(Y_{\text{SARSA}}) = 51.84 \times (0.05)(0.95) = 51.84 \times 0.0475 = \mathbf{2.4624}$
   - Expected SARSA eliminates variance of $2.4624$ entirely!

3. **Case $\epsilon = 0.50$ (Heavy Exploration):**
   - Probabilities: $\pi(a_1) = 0.25, \quad \pi(a_2) = 0.75$
   - $\bar{Q}(S') = 0.25(2.0) + 0.75(10.0) = 0.50 + 7.50 = 8.0000$
   - $Y_{\text{Exp}} = 1.0 + 0.9(8.00) = 1.0 + 7.20 = \mathbf{8.2000}$
   - $\operatorname{Var}(Y_{\text{SARSA}}) = 51.84 \times (0.25)(0.75) = 51.84 \times 0.1875 = \mathbf{9.7200}$

4. **Case $\epsilon = 1.00$ (Pure Uniform Random Exploration):**
   - Probabilities: $\pi(a_1) = 0.50, \quad \pi(a_2) = 0.50$
   - $\bar{Q}(S') = 0.50(2.0) + 0.50(10.0) = 1.00 + 5.00 = 6.0000$
   - $Y_{\text{Exp}} = 1.0 + 0.9(6.00) = 1.0 + 5.40 = \mathbf{6.4000}$
   - $\operatorname{Var}(Y_{\text{SARSA}}) = 51.84 \times (0.50)(0.50) = 51.84 \times 0.2500 = \mathbf{12.9600}$

**Summary Comparison Table:**

| Exploration $\epsilon$ | $\pi(a_1 \mid S')$ | $\pi(a_2 \mid S')$ | $Y_{\text{Exp}}$ Target | $\operatorname{Var}(Y_{\text{SARSA}})$ | $\operatorname{Var}(Y_{\text{Exp}})$ | Variance Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathbf{0.00}$ | $0.00$ | $1.00$ | $10.0000$ | $0.0000$ | $0.0000$ | $0\%$ |
| $\mathbf{0.10}$ | $0.05$ | $0.95$ | $9.6400$ | $2.4624$ | $0.0000$ | $\mathbf{100\%}$ |
| $\mathbf{0.50}$ | $0.25$ | $0.75$ | $8.2000$ | $9.7200$ | $0.0000$ | $\mathbf{100\%}$ |
| $\mathbf{1.00}$ | $0.50$ | $0.50$ | $6.4000$ | $12.9600$ | $0.0000$ | $\mathbf{100\%}$ |

Expected SARSA achieves complete $100\%$ elimination of target action-sampling variance regardless of policy exploration rate $\epsilon$!

---

## 7. Deep Learning Connection & Application

### 1. Deep Q-Networks (DQN - Mnih et al., Nature 2015)
DQN is the deep function approximation form of Watkins Q-learning:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
where experience replay decorrelates samples and target network $\theta^-$ stabilizes the moving TD target.

### 2. Double DQN (DDQN - van Hasselt et al., AAAI 2016)
To eliminate maximization bias in deep networks, Double DQN uses the online network $\theta$ to **select** the greedy action, but uses the target network $\theta^-$ to **evaluate** its value:
$$a^* = \arg \max_{a'} Q(s', a'; \theta)$$
$$Y_t^{\text{DoubleQ}} = r + \gamma Q(s', a^*; \theta^-)$$
This simple one-line modification eliminated massive overestimation spikes in Atari 2600 games and drastically improved sample efficiency.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of Part 5 hand calculations for SARSA ($5.4000$), Q-Learning ($7.0000$), and Expected SARSA ($6.8400$) to $< 10^{-14}$ machine precision.
2. The Cliff Walking benchmark: Running SARSA vs. Q-Learning vs. Expected SARSA, verifying that SARSA discovers the safe path while Q-learning takes the cliff edge.
3. Sutton's Maximization Bias experiment: Demonstrating Q-learning's severe overestimation on the roulette state vs. Double Q-learning's unbiased estimation.

See implementation in:
[`11_reinforcement_learning/code/07_sarsa_and_q_learning.py`](./code/07_sarsa_and_q_learning.py)
