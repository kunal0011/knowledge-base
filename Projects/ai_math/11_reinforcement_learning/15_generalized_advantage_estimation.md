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

### Illustration 1: Telescoping Sum Proof of $\lambda = 1$ Identity
**Problem:**
Prove that the sum of discounted TD residuals $\sum_{l=0}^\infty \gamma^l \delta_{t+l}^V$ is algebraically identical to the Monte Carlo advantage $G_t - V(S_t)$.

**Solution:**
Expand the sum for horizon $k$:
$$\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l \left( R_{t+l+1} + \gamma V(S_{t+l+1}) - V(S_{t+l}) \right)$$
$$= \sum_{l=0}^{k-1} \gamma^l R_{t+l+1} + \sum_{l=0}^{k-1} \left( \gamma^{l+1} V(S_{t+l+1}) - \gamma^l V(S_{t+l}) \right)$$

Observe the second summation:
$$\sum_{l=0}^{k-1} \left( \gamma^{l+1} V(S_{t+l+1}) - \gamma^l V(S_{t+l}) \right) = \left( \gamma V(S_{t+1}) - V(S_t) \right) + \left( \gamma^2 V(S_{t+2}) - \gamma V(S_{t+1}) \right) + \dots + \left( \gamma^k V(S_{t+k}) - \gamma^{k-1} V(S_{t+k-1}) \right)$$
All intermediate terms cancel out telescopically!
$$= \gamma^k V(S_{t+k}) - V(S_t)$$

Combining with the first summation:
$$\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l R_{t+l+1} + \gamma^k V(S_{t+k}) - V(S_t)$$
Taking $k \to \infty$, since $\gamma < 1$ and $V$ is bounded, $\gamma^k V(S_{t+k}) \to 0$:
$$\sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \underbrace{\sum_{l=0}^\infty \gamma^l R_{t+l+1}}_{G_t} - V(S_t) = G_t - V(S_t) \quad \blacksquare$$

---

## 7. Deep Learning Connection & Modern Applications

- **Proximal Policy Optimization (PPO):** PPO standardizes advantages using GAE:
  $$\hat{A}_t^{\text{norm}} = \frac{\hat{A}_t^{\text{GAE}} - \operatorname{mean}(\hat{A}^{\text{GAE}})}{\operatorname{std}(\hat{A}^{\text{GAE}}) + \epsilon}$$
  Standard hyperparameters are $\gamma = 0.99$ and $\lambda = 0.95$.
- **LLM Reinforcement Learning (RLHF):** In training reasoning models (e.g. GPT-4, DeepSeek-V3), GAE is applied token-by-token across long generated sequences ($T = 4096$ tokens). Setting $\lambda = 0.95$ prevents noisy final answer bonuses from corrupting the gradient updates on early reasoning tokens!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of Part 5 GAE hand calculations:
   - TD residuals $\boldsymbol{\delta}^V = [1.6000, 2.5000, 0.0000]$
   - GAE advantages $\hat{\mathbf{A}}^{\text{GAE}} = [3.4000, 2.5000, 0.0000]$ matching PyTorch/NumPy to $< 10^{-14}$.
2. Benchmark of empirical variance and policy gradient accuracy across $\lambda \in \{0.0, 0.5, 0.8, 0.95, 1.0\}$.
3. Production-grade, vectorized backward GAE implementation used in PPO.

See implementation in:
[`11_reinforcement_learning/code/15_generalized_advantage_estimation.py`](./code/15_generalized_advantage_estimation.py)
