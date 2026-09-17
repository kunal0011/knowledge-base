# Module 11.10: Deep Q-Networks (DQN)

---

## 1. Intuition & 101 Motivation

In 2015, Google DeepMind published a landmark breakthrough in *Nature* (Mnih et al.): an algorithm called **Deep Q-Networks (DQN)** that learned to master 49 different Atari 2600 games directly from raw pixel observations ($84 \times 84 \times 4$ luminance images), outperforming professional human testers on the majority of games without any game-specific handcrafting.

Prior to DQN, training non-linear deep neural networks to represent action-value functions $Q(s, a; \theta)$ in reinforcement learning was notoriously unstable and frequently diverged to garbage values. Three fatal obstacles plagued naive Deep Q-Learning:
1. **Strongly Correlated Data:** Consecutive video frames $(s_t, s_{t+1}, s_{t+2})$ in an RL trajectory violate the core assumption of independent and identically distributed (i.i.d.) training data required by stochastic gradient descent.
2. **The "Moving Goalpost" Instability:** When updating the network weights $\theta$ to make $Q(s_t, a_t; \theta)$ approach target $Y_t = r + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$, the target itself changes! A step to correct the prediction at $s_t$ inadvertently alters the predictions at $s_{t+1}$, inducing positive feedback loops and policy oscillation.
3. **Policy Distribution Shift:** Small changes in $Q(s, a; \theta)$ create massive, discontinuous changes in the policy $\pi(s) = \arg\max_a Q(s, a)$, dramatically shifting the distribution of states the agent explores.

DQN solved these fundamental problems with two breakthrough mechanisms:
- **Experience Replay ($\mathcal{D}$):** Stores past transitions and samples mini-batches uniformly at random, breaking temporal autocorrelation and restoring i.i.d. conditions.
- **Target Network ($\theta^-$):** Decouples the current online network from the TD target calculation, freezing the target for $C$ steps to stabilize regression.

```
+-----------------------------------------------------------------------------------------+
|                                    DQN ARCHITECTURE                                     |
|                                                                                         |
|   Environment ----(s, a, r, s', d)----> [ Experience Replay Buffer D ]                  |
|        ^                                            |                                   |
|        |                                    Sample Mini-batch                           |
|   Act  | eps-greedy                                 v                                   |
|        |                          +-----------------------------------+                 |
|        |                          |  Online Network Q(s, a; theta)    |                 |
|        |                          +-----------------------------------+                 |
|        |                                            |                                   |
|        |                                       Huber Loss                               |
|        |                                            ^                                   |
|        |                                            | TD Target                         |
|        |                          +-----------------------------------+                 |
|        +--------------------------|  Target Network Q(s', a'; theta-) |                 |
|                                   +-----------------------------------+                 |
|                                            ^                                            |
|                                            | Every C steps: theta- <-- theta            |
|                                            +--------------------------------            |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The DQN Loss Function

At each training iteration $i$, a mini-batch of transitions $B = \{(s, a, r, s', d)\}$ is sampled uniformly from the replay buffer $\mathcal{D}$:
$$(s, a, r, s', d) \sim U(\mathcal{D})$$
where $d \in \{0, 1\}$ is the terminal state indicator flag ($d = 1$ if $s'$ is terminal, $d = 0$ otherwise).

The target value $Y_i$ is computed using the **target network** parameterized by $\theta^-_i$:
$$Y_i \triangleq r + (1 - d) \gamma \max_{a' \in \mathcal{A}} Q(s', a'; \theta^-_i)$$

The objective is to minimize the empirical risk under the **Huber Loss** $\ell_\delta$:
$$\mathcal{L}(\theta_i) \triangleq \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \ell_\delta \left( Y_i - Q(s, a; \theta_i) \right) \right]$$

---

### 2.2 The Huber Loss (Smooth L1 Loss)

In deep reinforcement learning, immediate rewards and bootstrapped errors can occasionally be enormous (e.g., scoring $10,000$ points in an Atari game). Under standard Mean Squared Error (MSE), large TD errors $u = Y - Q$ produce quadratic penalties $\frac{1}{2} u^2$ and linear gradients $u$, causing gradient explosion.

The **Huber Loss** with threshold $\delta = 1.0$ acts quadratically for small errors (smooth convergence) and linearly for large errors (robustness):

$$\ell_\delta(u) \triangleq \begin{cases} \frac{1}{2} u^2 & \text{if } |u| \le \delta \\ \delta \left( |u| - \frac{1}{2} \delta \right) & \text{otherwise} \end{cases}$$

Its first derivative (gradient signal) is strictly clipped within $[-\delta, +\delta]$:
$$\nabla_u \ell_\delta(u) = \begin{cases} u & \text{if } |u| \le \delta \\ \delta \operatorname{sgn}(u) & \text{otherwise} \end{cases}$$

```
    Loss Value l(u)                             Gradient dl/du
        ^                                            ^
        |          / MSE (u^2)                    +1 |           /---------- (+1)
        |         /                                  |          /
        |        /   / Huber                         |         /
        |       /   /                                0 +------+------+-----> u
        |      /---/                                 |       /
        |     /   /                                  |      /
        +----+---+---+---> u                      -1 | ----/                 (-1)
            -1   0  +1                                  -1   0  +1
```

---

### 2.3 Gradient Computation

Differentiating the loss with respect to the online parameters $\theta$ (treating the target $Y$ as a constant scalar):

$$\nabla_{\theta} \mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ - \nabla_u \ell_\delta \left( Y - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

For errors within the quadratic zone ($|Y - Q| \le 1$):
$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E} \left[ - \left( Y - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

For errors in the linear zone ($|Y - Q| > 1$):
$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E} \left[ - \operatorname{sgn}\left( Y - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

---

### 2.4 Target Network Synchronization

Two synchronization schemes exist:
1. **Periodic Hard Update (Original Mnih et al., 2015):**
   Every $C$ environment steps (typically $C = 10,000$ in Atari), copy the parameters completely:
   $$\theta^- \leftarrow \theta$$
2. **Polyak Averaging / Soft Update (Lillicrap et al., 2016):**
   At every single training step, update the target network via an exponential moving average with hyperparameter $\tau \ll 1$ (e.g., $\tau = 0.005$):
   $$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.10.1: Rigorous Derivation of the Semi-Gradient DQN Update and Target Network Contraction

```
====================================================================================================
DERIVATION 11.10.1: Semi-Gradient DQN Objective and Fitted Q-Iteration Contraction
====================================================================================================
Problem Statement:
Consider the parameterized Deep Q-Network loss with decoupled target parameters θ^-:
    ℒ(θ; θ^-) ≜ 1/2 𝔼_{(s, a, r, s', d) ~ 𝒟} [ ( r + (1 - d) γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ) )^2 ]
1. Prove why taking the gradient solely with respect to θ (semi-gradient assumption)
   corresponds to minimizing the L_2 distance to the empirical Bellman target operator 𝒯* Q_{θ^-}.
2. Prove that under hard target synchronization every C steps (θ^- ← θ), the sequence of
   target networks executes an approximate Fitted Q-Iteration algorithm whose operator
   contracts toward Q* at geometric rate γ.
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Stationary Target Parameters:** During the inner optimization phase of length $C$, target weights $\theta^-$ are held fixed: $\nabla_\theta \theta^- = \mathbf{0}$.
2. **Replay Distribution:** Transitions $(s, a, r, s', d)$ are drawn from an empirical replay distribution $\mathcal{D}$ with support over state-action space $\mathcal{S} \times \mathcal{A}$.
3. **Discounting:** $0 \le \gamma < 1$.
4. **Universal Function Approximator:** The network class $\mathcal{F} = \{Q(\cdot, \cdot; \theta) \mid \theta \in \Theta\}$ is sufficiently expressive to represent the projected Bellman updates.

**2. Underlying Intuition:**
If one differentiates the full Bellman error where the target also depends on $\theta$, the resulting "residual gradient" algorithm generates unwanted cross-product terms that bias the parameters away from the true $Q^*$ fixed point. By freezing $\theta^-$, the target $Y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ becomes a deterministic supervisory signal. Minimizing $\mathcal{L}(\theta; \theta^-)$ with standard supervised regression is equivalent to projecting the Bellman optimality operator $\mathcal{T}^* Q_{\theta^-}$ onto the function manifold $\mathcal{F}$.

**3. End-to-End Algebraic Derivation:**

*Step 1: Differentiating the decoupled loss function.*
Let $Y(r, s', d; \theta^-) \triangleq r + (1 - d) \gamma \max_{a'} Q(s', a'; \theta^-)$.
The loss is:
$$\mathcal{L}(\theta; \theta^-) = \frac{1}{2} \mathbb{E}_{\mathcal{D}} \left[ \left( Y(r, s', d; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
Taking the partial derivative with respect to parameter vector $\theta$:
$$\nabla_\theta \mathcal{L}(\theta; \theta^-) = \nabla_\theta \left( \frac{1}{2} \mathbb{E}_{\mathcal{D}} \left[ \left( Y(r, s', d; \theta^-) - Q(s, a; \theta) \right)^2 \right] \right)$$
By the Leibniz integral rule (interchanging gradient and expectation):
$$= \mathbb{E}_{\mathcal{D}} \left[ \left( Y(r, s', d; \theta^-) - Q(s, a; \theta) \right) \cdot \nabla_\theta \left( Y(r, s', d; \theta^-) - Q(s, a; \theta) \right) \right]$$
Since $\theta^-$ is held constant with respect to $\theta$, $\nabla_\theta Y(r, s', d; \theta^-) = \mathbf{0}$.
Therefore:
$$\nabla_\theta \left( Y(r, s', d; \theta^-) - Q(s, a; \theta) \right) = - \nabla_\theta Q(s, a; \theta)$$
Substituting this back yields the celebrated DQN semi-gradient update rule:
$$\nabla_\theta \mathcal{L}(\theta; \theta^-) = - \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( r + (1 - d) \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

*Step 2: Relation to the Bellman Optimality Operator $\mathcal{T}^*$.*
Recall the continuous Bellman optimality operator:
$$(\mathcal{T}^* Q)(s, a) \triangleq \mathbb{E}_{r, s'} \left[ r + \gamma \max_{a'} Q(s', a') \;\middle|\; s, a \right]$$
The expected square error can be decomposed as:
$$\mathbb{E}_{\mathcal{D}} \left[ (Y - Q(s, a; \theta))^2 \right] = \mathbb{E}_{\mathcal{D}} \left[ \left( (\mathcal{T}^* Q_{\theta^-})(s, a) - Q(s, a; \theta) \right)^2 \right] + \operatorname{Var}_{\mathcal{D}}(Y \mid s, a)$$
Because the conditional variance term $\operatorname{Var}_{\mathcal{D}}(Y \mid s, a)$ does not depend on $\theta$, minimizing $\mathcal{L}(\theta; \theta^-)$ is identically equivalent to finding the best least-squares projection of the Bellman target function $\mathcal{T}^* Q_{\theta^-}$:
$$\theta_{k+1} = \arg\min_\theta \left\| Q_\theta - \mathcal{T}^* Q_{\theta_k^-} \right\|_{\mathcal{D}}^2$$

*Step 3: Contraction analysis of target network iterations.*
Suppose the inner optimization runs for $C$ steps and finds the exact projection $Q_{\theta_{k+1}} = \Pi_{\mathcal{D}} \mathcal{T}^* Q_{\theta_k^-}$.
At the synchronization checkpoint, we set $\theta_{k+1}^- \leftarrow \theta_{k+1}$.
In the supremum norm $\|Q\|_\infty = \max_{s, a} |Q(s, a)|$:
$$\|\mathcal{T}^* Q_1 - \mathcal{T}^* Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
Assuming a bounded projection error $\|\Pi_{\mathcal{D}} \mathcal{T}^* Q - \mathcal{T}^* Q\|_\infty \le \epsilon$:
$$\|Q_{\theta_{k+1}^-} - Q^*\|_\infty = \|\Pi_{\mathcal{D}} \mathcal{T}^* Q_{\theta_k^-} - \mathcal{T}^* Q^*\|_\infty \le \|\Pi_{\mathcal{D}} \mathcal{T}^* Q_{\theta_k^-} - \mathcal{T}^* Q_{\theta_k^-}\|_\infty + \|\mathcal{T}^* Q_{\theta_k^-} - \mathcal{T}^* Q^*\|_\infty$$
$$\le \epsilon + \gamma \|Q_{\theta_k^-} - Q^*\|_\infty$$
Unrolling this linear recurrence across $K$ target synchronization periods:
$$\|Q_{\theta_K^-} - Q^*\|_\infty \le \gamma^K \|Q_{\theta_0^-} - Q^*\|_\infty + \frac{\epsilon}{1 - \gamma}$$
Because $\gamma < 1$, as $K \to \infty$, the initial parameter error vanishes at geometric rate $\gamma^K$, bounding the asymptotic error to $\frac{\epsilon}{1 - \gamma}$. Freezing $\theta^-$ transforms a divergent non-linear system into a stable sequence of contracting fitted iterations. $\blacksquare$

---

#### Derivation 11.10.2: Mathematical Analysis of Experience Replay as Decorrelating Markovian Autocorrelation

```
====================================================================================================
DERIVATION 11.10.2: Autocorrelation Reduction via Uniform Experience Replay
====================================================================================================
Problem Statement:
Let {S_t}_{t ≥ 0} be an ergodic Markov chain generated by the agent's interaction with the environment,
exhibiting exponential decay of autocorrelation:
    Cov(f(S_t), f(S_{t+k})) = σ^2 ρ^k   with mixing rate 0 < ρ < 1
Prove that:
1. Online consecutive batch sampling of size B has variance Var(1/B ∑_{t=1}^B f(S_t)) scaling as
   O(1 / (B(1 - ρ))).
2. Sampling B transitions uniformly with replacement from a replay buffer of capacity N ≫ B
   reduces the expected covariance between any two sampled batch items to O(1 / N),
   recovering the classical i.i.d. variance scaling O(σ^2 / B).
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Geometric Ergodicity:** The environment Markov chain is irreducible and aperiodic with unique stationary distribution $d(s)$ and spectral gap $1 - \rho > 0$, implying geometric mixing: $|\operatorname{Cov}(f(S_t), f(S_{t+k}))| \le \sigma^2 \rho^k$ for $\rho \in (0, 1)$.
2. **Stationary Variance:** $\operatorname{Var}(f(S_t)) = \sigma^2 < \infty$.
3. **Replay Buffer Capacity:** The buffer stores $N$ sequential transitions $\mathcal{D} = \{e_1, e_2, \dots, e_N\}$ with $N \gg B$.

**2. Underlying Intuition:**
In online learning, consecutive transitions are tightly coupled ($\rho \approx 1$ in 60 FPS video games), effectively reducing the "effective sample size" of a batch of 32 frames to fewer than 2 independent frames. By drawing randomly from a buffer of size $N = 1,000,000$, the average time separation between two sampled elements is $O(N)$, where the autocorrelation $\rho^N \approx 0$ is exponentially suppressed.

**3. End-to-End Algebraic Derivation:**

*Step 1: Variance of an online consecutive batch.*
Consider an online batch of $B$ contiguous transitions: $\bar{f}_{\text{online}} = \frac{1}{B} \sum_{i=1}^B f(S_i)$.
The variance of the sample mean is:
$$\operatorname{Var}(\bar{f}_{\text{online}}) = \frac{1}{B^2} \sum_{i=1}^B \sum_{j=1}^B \operatorname{Cov}(f(S_i), f(S_j)) = \frac{1}{B^2} \left[ \sum_{i=1}^B \operatorname{Var}(f(S_i)) + 2 \sum_{i=1}^B \sum_{j=i+1}^B \operatorname{Cov}(f(S_i), f(S_j)) \right]$$
Using the exponential autocorrelation bound $\operatorname{Cov}(f(S_i), f(S_j)) = \sigma^2 \rho^{j - i}$:
$$\operatorname{Var}(\bar{f}_{\text{online}}) = \frac{\sigma^2}{B} + \frac{2 \sigma^2}{B^2} \sum_{k=1}^{B-1} (B - k) \rho^k$$
Notice that:
$$\sum_{k=1}^{B-1} (B - k) \rho^k \le B \sum_{k=1}^\infty \rho^k = B \frac{\rho}{1 - \rho}$$
Therefore:
$$\operatorname{Var}(\bar{f}_{\text{online}}) \le \frac{\sigma^2}{B} \left( 1 + \frac{2 \rho}{1 - \rho} \right) = \frac{\sigma^2}{B} \left( \frac{1 + \rho}{1 - \rho} \right)$$
When frames are sampled at 60 Hz, $\rho \approx 0.99$, so $\frac{1 + \rho}{1 - \rho} \approx \frac{1.99}{0.01} = 199$.
The variance of the gradient estimator is **inflated by a factor of $\approx 200\times$** compared to independent data!

*Step 2: Expected covariance between randomly sampled transitions from replay.*
Now let transitions $i, j$ be selected uniformly and independently at random from $\{1, 2, \dots, N\}$.
The time lag $\Delta = |i - j|$ is a random variable.
The probability of selecting two specific indices with separation $k = |i - j|$ for $k \ge 1$ is:
$$\mathbb{P}(|i - j| = k) = \frac{2(N - k)}{N^2}$$
The expected covariance between two distinct sampled items in the mini-batch is:
$$\mathbb{E}_{i, j \sim U(N)} [\operatorname{Cov}(f(S_i), f(S_j))] = \sum_{k=1}^{N-1} \mathbb{P}(|i - j| = k) \sigma^2 \rho^k = \frac{2 \sigma^2}{N^2} \sum_{k=1}^{N-1} (N - k) \rho^k$$
Bounding the summation:
$$\sum_{k=1}^{N-1} (N - k) \rho^k \le N \sum_{k=1}^\infty \rho^k = N \frac{\rho}{1 - \rho}$$
Therefore:
$$\mathbb{E}_{i, j \sim U(N)} [\operatorname{Cov}(f(S_i), f(S_j))] \le \frac{2 \sigma^2}{N^2} \left( N \frac{\rho}{1 - \rho} \right) = \frac{2 \sigma^2}{N} \left( \frac{\rho}{1 - \rho} \right) = O\left( \frac{1}{N} \right)$$

*Step 3: Variance of the replay mini-batch estimator.*
For a mini-batch of size $B$ sampled with replacement from buffer $\mathcal{D}$:
$$\operatorname{Var}(\bar{f}_{\text{replay}}) = \frac{\sigma^2}{B} + \frac{B - 1}{B} \mathbb{E}_{i \neq j} [\operatorname{Cov}(f(S_i), f(S_j))] \le \frac{\sigma^2}{B} + \frac{2 \sigma^2}{N} \left( \frac{\rho}{1 - \rho} \right)$$
For standard DQN hyperparameters ($B = 32, N = 1,000,000, \rho = 0.99$):
$$\frac{2}{N} \left( \frac{\rho}{1 - \rho} \right) = \frac{2}{10^6} \times 99 = 0.000198 \ll \frac{1}{B} = 0.03125$$
The correlation term is completely negligible, reducing the gradient variance back to the ideal i.i.d. rate $\frac{\sigma^2}{B}$! $\blacksquare$

---

#### Derivation 11.10.3: Convex Optimization Properties and Gradient Bound of the Huber Loss

```
====================================================================================================
DERIVATION 11.10.3: Mathematical Properties of the Huber Loss
====================================================================================================
Problem Statement:
Let the Huber loss with threshold δ > 0 be defined on ℝ by:
    ℓ_δ(u) ≜ { 1/2 u^2                  if |u| ≤ δ
             { δ (|u| - 1/2 δ)          if |u| > δ
Prove that:
1. ℓ_δ(u) is continuously differentiable (C^1) everywhere on ℝ.
2. The gradient ∇_u ℓ_δ(u) is globally Lipschitz continuous with Lipschitz constant L = 1:
       |∇_u ℓ_δ(u_1) - ∇_u ℓ_δ(u_2)| ≤ |u_1 - u_2|   ∀ u_1, u_2 ∈ ℝ
3. The gradient magnitude is globally bounded by δ:
       |∇_u ℓ_δ(u)| ≤ δ   ∀ u ∈ ℝ
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Threshold Parameter:** $\delta > 0$ (typically $\delta = 1.0$).
2. **Domain:** The error residue $u \in \mathbb{R}$, where $u \triangleq Y - Q(s, a; \theta)$.

**2. Underlying Intuition:**
Mean Squared Error $\frac{1}{2} u^2$ has unbounded gradient $u$, which causes exploding gradients in deep neural networks when an unexpected sparse reward creates a massive TD error. Pure $L_1$ loss $|u|$ has bounded gradient $\pm 1$, but its derivative is discontinuous at $u = 0$, causing optimization chatter and failure to settle at the minimum. The Huber loss smoothly splices a parabola onto two linear rays at $|u| = \delta$, inheriting the smooth convergence of $L_2$ near zero and the bounded, explosion-proof gradients of $L_1$ for outliers.

**3. End-to-End Algebraic Derivation:**

*Step 1: Continuous differentiability ($C^1$).*
Examine the piecewise definition of $\ell_\delta(u)$:
$$\ell_\delta(u) = \begin{cases} 
-\delta u - \frac{1}{2}\delta^2 & \text{if } u < -\delta \\
\frac{1}{2} u^2 & \text{if } -\delta \le u \le \delta \\
\delta u - \frac{1}{2}\delta^2 & \text{if } u > \delta 
\end{cases}$$
The function is obviously infinitely differentiable on the open intervals $(-\infty, -\delta)$, $(-\delta, \delta)$, and $(\delta, \infty)$.
Check continuity of the derivative at boundary $u = \delta$:
- Left derivative: $\lim_{u \to \delta^-} \frac{d}{du} \left( \frac{1}{2} u^2 \right) = \lim_{u \to \delta^-} u = \delta$.
- Right derivative: $\lim_{u \to \delta^+} \frac{d}{du} \left( \delta u - \frac{1}{2} \delta^2 \right) = \delta$.
Because the left and right derivatives match ($\delta = \delta$), the derivative exists at $u = \delta$ with value $\delta$.
Check continuity of the derivative at boundary $u = -\delta$:
- Left derivative: $\lim_{u \to -\delta^-} (-\delta) = -\delta$.
- Right derivative: $\lim_{u \to -\delta^+} (u) = -\delta$.
Both match ($-\delta = -\delta$).
Hence, the first derivative exists and is continuous everywhere on $\mathbb{R}$:
$$\nabla_u \ell_\delta(u) = \begin{cases} -\delta & \text{if } u < -\delta \\ u & \text{if } |u| \le \delta \\ +\delta & \text{if } u > \delta \end{cases} = \operatorname{clip}(u, -\delta, +\delta)$$

*Step 2: Lipschitz continuity of the gradient.*
Consider the derivative $g(u) \triangleq \nabla_u \ell_\delta(u) = \operatorname{clip}(u, -\delta, +\delta)$.
The weak derivative (almost everywhere on $\mathbb{R}$) is:
$$g'(u) = \begin{cases} 0 & \text{if } |u| > \delta \\ 1 & \text{if } |u| < \delta \end{cases}$$
Notice that $0 \le g'(u) \le 1$ for all $u \in \mathbb{R} \setminus \{-\delta, \delta\}$.
By the Mean Value Theorem, for any $u_1, u_2 \in \mathbb{R}$:
$$|g(u_1) - g(u_2)| \le \left( \sup_{u \in \mathbb{R}} |g'(u)| \right) |u_1 - u_2| = 1 \cdot |u_1 - u_2|$$
Thus, $\nabla_u \ell_\delta(u)$ is globally Lipschitz continuous with Lipschitz constant $L = 1$.
By the Descent Lemma, gradient descent on $\ell_\delta$ with step size $\alpha \le 1/L = 1$ is guaranteed to never diverge.

*Step 3: Global gradient bound.*
Directly from the clipping representation:
$$|\nabla_u \ell_\delta(u)| = |\operatorname{clip}(u, -\delta, +\delta)| \le \delta \quad \forall u \in \mathbb{R}$$
For standard DQN ($\delta = 1.0$), the gradient with respect to the prediction is bounded by $1.0$:
$$\left| \frac{\partial \ell_\delta}{\partial Q} \right| \le 1.0$$
Consequently, the gradient passed to the neural network parameters satisfies:
$$\left\| \nabla_\theta \ell_\delta(Y - Q(s, a; \theta)) \right\| \le 1.0 \times \left\| \nabla_\theta Q(s, a; \theta) \right\|$$
This mathematically guarantees that outlier transitions with arbitrarily large reward spikes cannot cause exploding gradients! $\blacksquare$

---

## 3. Geometric Interpretation: Freezing the Target Manifold


In parameter space $\mathbb{R}^P$:
- When training without a target network ($\theta^- = \theta$), the loss surface $\mathcal{L}(\theta) = \frac{1}{2} (r + \gamma \max Q(s', a'; \theta) - Q(s, a; \theta))^2$ changes dynamically with every gradient step. The minimum shifts as the weights move, turning the optimization landscape into a rolling, non-stationary wave where the gradient $\nabla_\theta \mathcal{L}$ can point in chaotic, destabilizing directions.
- By fixing $\theta^-$ as a constant vector, the target $Y = r + \gamma \max Q(s', a'; \theta^-)$ becomes a stationary fixed landmark in $\mathbb{R}$. The loss landscape freezes into a stationary quadratic bowl for $C$ steps, allowing standard stochastic gradient descent to descend monotonically.

```
       WITHOUT TARGET NETWORK                         WITH TARGET NETWORK
         (Moving Goalpost)                         (Stationary Goalpost)

           Loss                                        Loss
          Landscape                                   Landscape
             ~ ~                                          \
            ~ * ~  <- Minimum moves away                   \     * Minimum frozen at Y!
           ~ ~ ~     at every step!                         \   /
          ~ ~ ~ ~                                            \_/
```

---

## 4. Real-World Analogy: The Exam Preparation Strategy

Imagine a student studying for the SATs:
- **No Replay Buffer (Correlated Data):** The student studies 500 geometry questions in a row on Monday, then 500 vocabulary questions on Tuesday. By Friday, the student has completely forgotten geometry due to catastrophic forgetting.
  - **With Replay Buffer:** The student writes every question on an index card and tosses it into a giant box. Each study session draws a random shuffled handful of 32 cards across algebra, vocabulary, and geometry.
- **No Target Network (Moving Answer Key):** Every time the student gets a question wrong, the teacher changes the textbook answers to match the student's partial guess!
  - **With Target Network:** The textbook answer key is locked for the entire month. The student tests against fixed answers. Only on the 1st of every month does the teacher release an updated, verified answer key.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 1-Layer Linear Neural Network
To see the exact arithmetic down to machine precision, consider a single linear layer with 2 inputs and 2 discrete actions:
$$Q(s, a; \mathbf{W}) = \mathbf{W} s = \begin{bmatrix} W_{00} & W_{01} \\ W_{10} & W_{11} \end{bmatrix} \begin{bmatrix} s_0 \\ s_1 \end{bmatrix}$$
where row $0$ outputs $Q(s, a_0)$ and row $1$ outputs $Q(s, a_1)$.

**Hyperparameters:**
- Discount factor: $\gamma = 0.9000$
- Learning rate: $\alpha = 0.1000$
- Huber loss threshold: $\delta = 1.0000$

**Initial Online Network Weights $\mathbf{W}$:**
$$\mathbf{W} = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix}$$

**Target Network Weights $\mathbf{W}^-$ (Frozen):**
$$\mathbf{W}^- = \begin{bmatrix} 0.4000 & 0.1000 \\ -0.1000 & 0.6000 \end{bmatrix}$$

**Sampled Mini-Batch of 2 Transitions:**
- **Sample 1:** $s^{(1)} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$, taken action $a = 0$ ($a_0$), reward $r = 1.0$, next state $s'^{(1)} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$, done $d = 0$.
- **Sample 2:** $s^{(2)} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$, taken action $a = 1$ ($a_1$), reward $r = 2.0$, next state $s'^{(2)} = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$, done $d = 1$ (terminal!).

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Matrix / Vector in Hand Walkthrough |
| :--- | :--- | :--- |
| $\mathbf{W}$ | Online Network Parameter Matrix | $2 \times 2$ weights generating current predictions $Q(s, a; \mathbf{W})$ |
| $\mathbf{W}^-$ | Target Network Parameter Matrix | $2 \times 2$ frozen weights generating next-state targets $Q(s', a'; \mathbf{W}^-)$ |
| $s$ | Current State Input Vector | $2 \times 1$ column vector |
| $s'$ | Successor State Input Vector | $2 \times 1$ column vector |
| $Y$ | Bootstrapped Target | $r + (1 - d) \gamma \max_{a'} Q(s', a'; \mathbf{W}^-)$ |
| $u$ | TD Error Residue | $Y - Q(s, a; \mathbf{W})$ |
| $\ell_\delta(u)$ | Huber Loss | Quadratic if $|u| \le 1$, Linear if $|u| > 1$ |
| $g_u$ | Huber Gradient $\nabla_u \ell_\delta$ | $u$ if $|u| \le 1$, else $\operatorname{sgn}(u)$ |
| $\nabla_\mathbf{W} \mathcal{L}$ | Weight Gradient Tensor | Matrix of partial derivatives $\frac{\partial \mathcal{L}}{\partial W_{ij}}$ |

---

### 5.3 Step 1: Forward Pass on Target Network (Compute $Y$)

#### For Sample 1 ($s'^{(1)} = [0.0, 1.0]^\top, r = 1.0, d = 0$):
Compute next-state Q-values using frozen target network $\mathbf{W}^-$:
$$Q(s'^{(1)}; \mathbf{W}^-) = \begin{bmatrix} 0.4000 & 0.1000 \\ -0.1000 & 0.6000 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.1000 \\ 0.6000 \end{bmatrix}$$
Maximum target Q-value:
$$\max_{a'} Q(s'^{(1)}, a'; \mathbf{W}^-) = \max(0.1000, 0.6000) = \mathbf{0.6000}$$

Compute Target $Y^{(1)}$:
$$Y^{(1)} = r + (1 - d) \gamma \max_{a'} Q = 1.0000 + (1 - 0) \times 0.9000 \times 0.6000 = 1.0000 + 0.5400 = \mathbf{1.5400}$$

#### For Sample 2 ($s'^{(2)} = [1.0, 1.0]^\top, r = 2.0, d = 1$ [Terminal]):
Because $d = 1$, the successor state value is zeroed out:
$$Y^{(2)} = r + (1 - 1) \gamma \max_{a'} Q = 2.0000 + 0.0000 = \mathbf{2.0000}$$

---

### 5.4 Step 2: Forward Pass on Online Network (Compute Current Predictions)

#### For Sample 1 ($s^{(1)} = [1.0, 0.0]^\top$, taken action $a = 0$):
$$Q(s^{(1)}; \mathbf{W}) = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5000} \\ -0.3000 \end{bmatrix}$$
The prediction for chosen action $a = 0$ is:
$$Q(s^{(1)}, a_0; \mathbf{W}) = \mathbf{0.5000}$$

TD Error residue $u^{(1)}$:
$$u^{(1)} = Y^{(1)} - Q(s^{(1)}, a_0; \mathbf{W}) = 1.5400 - 0.5000 = \mathbf{1.0400}$$

#### For Sample 2 ($s^{(2)} = [0.0, 1.0]^\top$, taken action $a = 1$):
$$Q(s^{(2)}; \mathbf{W}) = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.2000 \\ \mathbf{0.8000} \end{bmatrix}$$
The prediction for chosen action $a = 1$ is:
$$Q(s^{(2)}, a_1; \mathbf{W}) = \mathbf{0.8000}$$

TD Error residue $u^{(2)}$:
$$u^{(2)} = Y^{(2)} - Q(s^{(2)}, a_1; \mathbf{W}) = 2.0000 - 0.8000 = \mathbf{1.2000}$$

---

### 5.5 Step 3: Huber Loss & Gradient Evaluation

#### For Sample 1:
- Error $u^{(1)} = 1.0400 > 1.0000$ (Falls into **Linear Zone**!).
- Huber Loss:
  $$\ell_\delta(u^{(1)}) = 1.0 \times \left( |1.0400| - 0.5 \times 1.0 \right) = 1.0400 - 0.5000 = \mathbf{0.5400}$$
- Huber Gradient:
  $$g_u^{(1)} = \operatorname{sgn}(1.0400) = \mathbf{+1.0000}$$

#### For Sample 2:
- Error $u^{(2)} = 1.2000 > 1.0000$ (Falls into **Linear Zone**!).
- Huber Loss:
  $$\ell_\delta(u^{(2)}) = 1.2000 - 0.5000 = \mathbf{0.7000}$$
- Huber Gradient:
  $$g_u^{(2)} = \operatorname{sgn}(1.2000) = \mathbf{+1.0000}$$

**Mean Mini-Batch Loss:**
$$\mathcal{L} = \frac{0.5400 + 0.7000}{2} = \frac{1.2400}{2} = \mathbf{0.6200}$$

---

### 5.6 Step 4: Backward Pass & Online Weight Update

The loss with respect to prediction is $\frac{\partial \mathcal{L}}{\partial Q} = - \frac{1}{|B|} g_u$.
Since $Q(s, a) = \sum_j W_{a, j} s_j$, the weight gradient for row $a$ is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{a, :}} = - \frac{1}{|B|} g_u s^\top$$

#### Gradient from Sample 1 ($a = 0, s = [1.0, 0.0]^\top, g_u = 1.0$):
$$\nabla_\mathbf{W}^{(1)} = \begin{bmatrix} -1.0 \times 1.0 & -1.0 \times 0.0 \\ 0.0 & 0.0 \end{bmatrix} = \begin{bmatrix} -1.0000 & 0.0000 \\ 0.0000 & 0.0000 \end{bmatrix}$$

#### Gradient from Sample 2 ($a = 1, s = [0.0, 1.0]^\top, g_u = 1.0$):
$$\nabla_\mathbf{W}^{(2)} = \begin{bmatrix} 0.0 & 0.0 \\ -1.0 \times 0.0 & -1.0 \times 1.0 \end{bmatrix} = \begin{bmatrix} 0.0000 & 0.0000 \\ 0.0000 & -1.0000 \end{bmatrix}$$

#### Average Batch Gradient $\nabla_\mathbf{W} \mathcal{L}$:
$$\nabla_\mathbf{W} \mathcal{L} = \frac{1}{2} \left( \nabla_\mathbf{W}^{(1)} + \nabla_\mathbf{W}^{(2)} \right) = \begin{bmatrix} -0.5000 & 0.0000 \\ 0.0000 & -0.5000 \end{bmatrix}$$

#### Gradient Descent Step ($\alpha = 0.1000$):
$$\mathbf{W}_{\text{new}} = \mathbf{W} - \alpha \nabla_\mathbf{W} \mathcal{L} = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix} - 0.1000 \begin{bmatrix} -0.5000 & 0.0000 \\ 0.0000 & -0.5000 \end{bmatrix}$$
$$= \begin{bmatrix} 0.5000 + 0.0500 & 0.2000 \\ -0.3000 & 0.8000 + 0.0500 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.5500 & 0.2000 \\ -0.3000 & 0.8500 \end{bmatrix}}$$

The target weights $\mathbf{W}^-$ remain **completely unchanged**:
$$\mathbf{W}^- = \begin{bmatrix} 0.4000 & 0.1000 \\ -0.1000 & 0.6000 \end{bmatrix}$$

---

### 5.7 Visual Summary Tensor Grid

| Sample | Input $s$ | Target $Y$ | Predict $Q(s, a)$ | Residue $u$ | Huber Loss | Huber Grad $g_u$ | Weight Update $\Delta W$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $[1, 0]^\top, a=0$ | $1.5400$ | $0.5000$ | $+1.0400$ | $0.5400$ | $+1.0000$ | $\Delta W_{00} = +0.0500$ |
| **2** | $[0, 1]^\top, a=1$ | $2.0000$ | $0.8000$ | $+1.2000$ | $0.7000$ | $+1.0000$ | $\Delta W_{11} = +0.0500$ |

Every single arithmetic step is completely crystal clear and verified to exact decimals.

---

## 6. Solved Illustrations

### Illustration 1: The Atari Frame-Skipping & Frame-Stacking Pipeline
**Problem:**
Why can a Deep Q-Network not take a single $84 \times 84$ Atari video frame as state input $s_t$?
**Solution:**
A single video frame contains only position information, but **zero velocity or acceleration information**.
In *Pong* or *Breakout*, seeing the ball at coordinates $(x, y)$ in a single snapshot does not tell the network whether the ball is traveling left or right, up or down.
Formally, a single screen observation is non-Markovian:
$$\mathbb{P}(S_{t+1} \mid S_t) \neq \mathbb{P}(S_{t+1} \mid S_t, S_{t-1})$$
The system is a Partially Observable Markov Decision Process (POMDP).
By **stacking the 4 most recent frames** into an $84 \times 84 \times 4$ tensor, the network can compute finite differences:
$$v \approx \frac{x_t - x_{t-1}}{\Delta t}, \quad a \approx \frac{v_t - v_{t-1}}{\Delta t}$$
restoring the Markov property $\mathbb{P}(S_{t+1} \mid S_t, \dots, S_0) = \mathbb{P}(S_{t+1} \mid S_t)$. $\blacksquare$

---

### Illustration 2: Complete End-to-End DQN Forward and Backward Pass on a Non-Linear 2-Layer MLP

**Problem:**
Consider a DQN agent whose Q-network is a 2-layer MLP with ReLU activation:
$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{s} + \mathbf{b}_1, \quad \mathbf{h}_1 = \operatorname{ReLU}(\mathbf{z}_1), \quad \mathbf{q} = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2$$
where input dimension is $2$, hidden dimension is $2$, and output dimension is $2$ (actions $a_0, a_1$).
Initial weights and biases:
$$\mathbf{W}_1 = \begin{bmatrix} 0.50 & -0.20 \\ 0.10 & 0.40 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0.10 \\ -0.10 \end{bmatrix}, \quad \mathbf{W}_2 = \begin{bmatrix} 0.30 & 0.50 \\ -0.40 & 0.20 \end{bmatrix}, \quad \mathbf{b}_2 = \begin{bmatrix} 0.00 \\ 0.10 \end{bmatrix}$$
A transition $(s, a, r, s', d)$ is sampled from replay:
$$s = \begin{bmatrix} 1.00 \\ 0.50 \end{bmatrix}, \quad a = 0 \ (a_0), \quad r = 1.00, \quad d = 0$$
The target network predicts $\max_{a'} Q(s', a'; \theta^-) = 0.5000$.
Hyperparameters: $\gamma = 0.90$, learning rate $\alpha = 0.10$, Huber loss threshold $\delta = 1.00$.
1. Compute the target value $Y$ and the forward pass predictions $\mathbf{q}$.
2. Compute the Huber loss and its gradient with respect to $\mathbf{q}$.
3. Backpropagate the gradients through all layers and compute the updated weights $\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2$.

**Solution:**

*Step 1: Compute target $Y$ and online forward pass.*
The bootstrapped target is:
$$Y = r + (1 - d) \gamma \max_{a'} Q(s', a'; \theta^-) = 1.0000 + (1 - 0) \times 0.9000 \times 0.5000 = 1.0000 + 0.4500 = \mathbf{1.4500}$$

Now compute the online forward pass for state $s = [1.0, 0.5]^\top$:
$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{s} + \mathbf{b}_1 = \begin{bmatrix} 0.50 & -0.20 \\ 0.10 & 0.40 \end{bmatrix} \begin{bmatrix} 1.00 \\ 0.50 \end{bmatrix} + \begin{bmatrix} 0.10 \\ -0.10 \end{bmatrix} = \begin{bmatrix} 0.50(1.0) - 0.20(0.5) + 0.10 \\ 0.10(1.0) + 0.40(0.5) - 0.10 \end{bmatrix} = \begin{bmatrix} 0.50 - 0.10 + 0.10 \\ 0.10 + 0.20 - 0.10 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5000} \\ \mathbf{0.2000} \end{bmatrix}$$
Apply ReLU activation:
$$\mathbf{h}_1 = \operatorname{ReLU}(\mathbf{z}_1) = \begin{bmatrix} \max(0, 0.50) \\ \max(0, 0.20) \end{bmatrix} = \begin{bmatrix} \mathbf{0.5000} \\ \mathbf{0.2000} \end{bmatrix}$$
Output layer:
$$\mathbf{q} = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2 = \begin{bmatrix} 0.30 & 0.50 \\ -0.40 & 0.20 \end{bmatrix} \begin{bmatrix} 0.50 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0.00 \\ 0.10 \end{bmatrix} = \begin{bmatrix} 0.30(0.50) + 0.50(0.20) + 0.0 \\ -0.40(0.50) + 0.20(0.20) + 0.10 \end{bmatrix} = \begin{bmatrix} 0.15 + 0.10 \\ -0.20 + 0.04 + 0.10 \end{bmatrix} = \begin{bmatrix} \mathbf{0.2500} \\ \mathbf{-0.0600} \end{bmatrix}$$
The prediction for chosen action $a = 0$ is $Q(s, a_0) = \mathbf{0.2500}$.

*Step 2: TD residue, Huber loss, and output gradient.*
$$u = Y - Q(s, a_0) = 1.4500 - 0.2500 = \mathbf{1.2000}$$
Because $|u| = 1.2000 > \delta = 1.0000$, the error falls into the **linear regime**:
$$\ell_\delta(u) = \delta (|u| - 0.5 \delta) = 1.0000 (1.2000 - 0.5000) = \mathbf{0.7000}$$
The gradient with respect to error residue is $g_u = \operatorname{sgn}(1.2000) = +1.0000$.
The gradient with respect to the Q-values $\mathbf{q} = [Q(s, a_0), Q(s, a_1)]^\top$ is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{q}} = \begin{bmatrix} -g_u \\ 0 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.0000} \\ \mathbf{0.0000} \end{bmatrix}$$

*Step 3: Backpropagation through layer 2 and layer 1.*
**Layer 2 Gradients:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{q}} \right) \mathbf{h}_1^\top = \begin{bmatrix} -1.0 \\ 0.0 \end{bmatrix} \begin{bmatrix} 0.50 & 0.20 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.5000} & \mathbf{-0.2000} \\ \mathbf{0.0000} & \mathbf{0.0000} \end{bmatrix}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \frac{\partial \mathcal{L}}{\partial \mathbf{q}} = \begin{bmatrix} \mathbf{-1.0000} \\ \mathbf{0.0000} \end{bmatrix}$$

**Backprop through Hidden Layer:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \mathbf{W}_2^\top \frac{\partial \mathcal{L}}{\partial \mathbf{q}} = \begin{bmatrix} 0.30 & -0.40 \\ 0.50 & 0.20 \end{bmatrix} \begin{bmatrix} -1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.3000} \\ \mathbf{-0.5000} \end{bmatrix}$$
Since $\mathbf{z}_1 = [0.50, 0.20]^\top > \mathbf{0}$, the ReLU indicator is $\mathbb{I}(\mathbf{z}_1 > 0) = [1.0, 1.0]^\top$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} \odot \mathbb{I}(\mathbf{z}_1 > 0) = \begin{bmatrix} \mathbf{-0.3000} \\ \mathbf{-0.5000} \end{bmatrix}$$

**Layer 1 Gradients:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} \right) \mathbf{s}^\top = \begin{bmatrix} -0.30 \\ -0.50 \end{bmatrix} \begin{bmatrix} 1.00 & 0.50 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.3000} & \mathbf{-0.1500} \\ \mathbf{-0.5000} & \mathbf{-0.2500} \end{bmatrix}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \begin{bmatrix} \mathbf{-0.3000} \\ \mathbf{-0.5000} \end{bmatrix}$$

*Step 4: Parameter Updates ($\alpha = 0.10$).*
$$\mathbf{W}_{2, \text{new}} = \mathbf{W}_2 - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \begin{bmatrix} 0.30 & 0.50 \\ -0.40 & 0.20 \end{bmatrix} - 0.10 \begin{bmatrix} -0.50 & -0.20 \\ 0.00 & 0.00 \end{bmatrix} = \begin{bmatrix} \mathbf{0.3500} & \mathbf{0.5200} \\ \mathbf{-0.4000} & \mathbf{0.2000} \end{bmatrix}$$
$$\mathbf{b}_{2, \text{new}} = \mathbf{b}_2 - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \begin{bmatrix} 0.00 \\ 0.10 \end{bmatrix} - 0.10 \begin{bmatrix} -1.00 \\ 0.00 \end{bmatrix} = \begin{bmatrix} \mathbf{0.1000} \\ \mathbf{0.1000} \end{bmatrix}$$
$$\mathbf{W}_{1, \text{new}} = \mathbf{W}_1 - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \begin{bmatrix} 0.50 & -0.20 \\ 0.10 & 0.40 \end{bmatrix} - 0.10 \begin{bmatrix} -0.30 & -0.15 \\ -0.50 & -0.25 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5300} & \mathbf{-0.1850} \\ \mathbf{0.1500} & \mathbf{0.4250} \end{bmatrix}$$
$$\mathbf{b}_{1, \text{new}} = \mathbf{b}_1 - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \begin{bmatrix} 0.10 \\ -0.10 \end{bmatrix} - 0.10 \begin{bmatrix} -0.30 \\ -0.50 \end{bmatrix} = \begin{bmatrix} \mathbf{0.1300} \\ \mathbf{-0.0500} \end{bmatrix} \quad \blacksquare$$

---

### Illustration 3: Huber Loss vs. Mean Squared Error (MSE) Gradient Dynamics under Outliers

**Problem:**
Consider two transitions sampled from replay:
- Transition A (typical TD error): $u_A = Y_A - Q_A = 0.5000$.
- Transition B (massive outlier TD error): $u_B = Y_B - Q_B = 5.0000$.
Let the feature be $x = 2.0$, and learning rate $\alpha = 0.05$.
Compare the resulting gradient $\nabla_w \mathcal{L}$ and weight update $\Delta w$ under:
1. Mean Squared Error: $\ell_{\text{MSE}}(u) = \frac{1}{2} u^2$.
2. Huber Loss ($\delta = 1.0$): $\ell_{\text{Huber}}(u)$.

**Solution:**
Recall that for a linear model $Q = w x$, $\frac{\partial Q}{\partial w} = x = 2.0$.
The weight gradient is $\nabla_w \mathcal{L} = -\left( \nabla_u \ell \right) \cdot x = -2.0 \nabla_u \ell$.
The parameter update is $\Delta w = -\alpha \nabla_w \mathcal{L} = 0.05 \times 2.0 \times \nabla_u \ell = 0.10 \nabla_u \ell$.

- **For Transition A ($u_A = 0.5000$):**
  - Under MSE: $\nabla_u \ell = u_A = \mathbf{0.5000} \implies \nabla_w \mathcal{L} = -1.0000 \implies \Delta w_{\text{MSE}} = \mathbf{+0.0500}$.
  - Under Huber ($|0.5| \le 1.0$): $\nabla_u \ell = u_A = \mathbf{0.5000} \implies \nabla_w \mathcal{L} = -1.0000 \implies \Delta w_{\text{Huber}} = \mathbf{+0.0500}$.
  In the quadratic zone, Huber loss behaves **identically to MSE**!

- **For Transition B ($u_B = 5.0000$):**
  - Under MSE: $\nabla_u \ell = u_B = \mathbf{5.0000} \implies \nabla_w \mathcal{L} = -10.0000 \implies \Delta w_{\text{MSE}} = \mathbf{+0.5000}$ ($10\times$ larger!).
  - Under Huber ($|5.0| > 1.0$): $\nabla_u \ell = \delta \operatorname{sgn}(u_B) = \mathbf{+1.0000} \implies \nabla_w \mathcal{L} = -2.0000 \implies \Delta w_{\text{Huber}} = \mathbf{+0.1000}$.
  The Huber update is clipped at $0.1000$ (only $2\times$ Transition A, rather than $10\times$), completely eliminating catastrophic gradient shock! $\blacksquare$

---

### Illustration 4: Polyak Soft Target Network Tracking Dynamics ($\tau$-Averaging)

**Problem:**
In continuous control algorithms (DDPG, SAC, TD3) and modern DQN variants, the target network parameters $\theta^-$ track the online parameters $\theta$ via Polyak averaging:
$$\theta^-_{t+1} = \tau \theta_t + (1 - \tau) \theta^-_t$$
Let smoothing rate $\tau = 0.2000$.
Suppose the online network has converged to a new value $\theta_t \equiv 2.0000$ for all $t \ge 0$, and initial target weights are $\theta^-_0 = 1.0000$.
1. Compute $\theta^-_1, \theta^-_2, \theta^-_3, \theta^-_4, \theta^-_5$ step-by-step.
2. Derive the closed-form equation for $\theta^-_t$ and verify exact numerical agreement.

**Solution:**

*Step 1: Step-by-step hand calculation.*
With $\tau = 0.2000$ and $1 - \tau = 0.8000$, the recurrence is $\theta^-_{t+1} = 0.20(2.0000) + 0.80 \theta^-_t = 0.4000 + 0.80 \theta^-_t$:
- **Step 1:** $\theta^-_1 = 0.4000 + 0.80(1.0000) = 0.4000 + 0.8000 = \mathbf{1.2000}$
- **Step 2:** $\theta^-_2 = 0.4000 + 0.80(1.2000) = 0.4000 + 0.9600 = \mathbf{1.3600}$
- **Step 3:** $\theta^-_3 = 0.4000 + 0.80(1.3600) = 0.4000 + 1.0880 = \mathbf{1.4880}$
- **Step 4:** $\theta^-_4 = 0.4000 + 0.80(1.4880) = 0.4000 + 1.1904 = \mathbf{1.5904}$
- **Step 5:** $\theta^-_5 = 0.4000 + 0.80(1.5904) = 0.4000 + 1.27232 = \mathbf{1.67232}$

*Step 2: Analytical closed-form derivation.*
Subtract $\theta$ from both sides of the update equation:
$$\theta^-_{t+1} - \theta = (1 - \tau) (\theta^-_t - \theta)$$
This is a standard geometric progression:
$$\theta^-_t - \theta = (1 - \tau)^t (\theta^-_0 - \theta) \implies \theta^-_t = \theta - (1 - \tau)^t (\theta - \theta^-_0)$$
Substituting $\theta = 2.0000, \theta^-_0 = 1.0000, 1 - \tau = 0.8000$:
$$\theta^-_t = 2.0000 - (0.8000)^t (2.0000 - 1.0000) = 2.0000 - (0.8000)^t$$
Evaluating at $t = 5$:
$$(0.8000)^5 = 0.32768 \implies \theta^-_5 = 2.0000 - 0.32768 = \mathbf{1.67232} \checkmark$$
Polyak averaging provides continuous, smooth tracking without the periodic shockwaves induced by hard updates. $\blacksquare$

---

### Illustration 5: Prioritized Experience Replay (PER) Sampling Probabilities & IS Weights

**Problem:**
A replay buffer holds $N = 4$ transitions with absolute TD errors $|\delta_i| = [0.10, 0.40, 0.20, 0.90]$.
Let stability constant $\epsilon = 0.01$, prioritization exponent $\alpha = 1.00$ (proportional prioritization), and importance sampling exponent $\beta = 0.50$.
1. Compute the sampling priorities $p_i$ and categorical probabilities $P(i)$ for each transition.
2. Compute the raw Importance Sampling (IS) weights $w_i = (N \cdot P(i))^{-\beta}$.
3. Normalize the weights such that $\max_j w_j = 1.0000$ to maintain gradient scaling.

**Solution:**

*Step 1: Compute priorities and probabilities.*
Priority definition: $p_i = |\delta_i| + \epsilon$.
- $p_1 = 0.10 + 0.01 = \mathbf{0.1100}$
- $p_2 = 0.40 + 0.01 = \mathbf{0.4100}$
- $p_3 = 0.20 + 0.01 = \mathbf{0.2100}$
- $p_4 = 0.90 + 0.01 = \mathbf{0.9100}$
Sum of priorities: $\sum_{j=1}^4 p_j = 0.11 + 0.41 + 0.21 + 0.91 = \mathbf{1.6400}$.

Sampling probabilities $P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha} = \frac{p_i}{1.6400}$:
- $P(1) = \frac{0.11}{1.64} = \frac{11}{164} \approx \mathbf{0.0671}$
- $P(2) = \frac{0.41}{1.64} = \frac{41}{164} = \mathbf{0.2500}$
- $P(3) = \frac{0.21}{1.64} = \frac{21}{164} \approx \mathbf{0.1280}$
- $P(4) = \frac{0.91}{1.64} = \frac{91}{164} \approx \mathbf{0.5549}$

*Step 2: Compute raw Importance Sampling weights ($N = 4, \beta = 0.50$).*
Formula: $w_i = (N \cdot P(i))^{-\beta} = \frac{1}{\sqrt{4 P(i)}}$:
- $w_1 = \frac{1}{\sqrt{4 \times 0.067073}} = \frac{1}{\sqrt{0.26829}} = \frac{1}{0.51797} \approx \mathbf{1.9306}$
- $w_2 = \frac{1}{\sqrt{4 \times 0.2500}} = \frac{1}{\sqrt{1.0000}} = \mathbf{1.0000}$
- $w_3 = \frac{1}{\sqrt{4 \times 0.128049}} = \frac{1}{\sqrt{0.512195}} = \frac{1}{0.71568} \approx \mathbf{1.3973}$
- $w_4 = \frac{1}{\sqrt{4 \times 0.554878}} = \frac{1}{\sqrt{2.219512}} = \frac{1}{1.48980} \approx \mathbf{0.6712}$

*Step 3: Normalize weights by $\max_j w_j = w_1 = 1.9306$.*
$$w_i^{\text{norm}} = \frac{w_i}{\max_j w_j}$$
- $w_1^{\text{norm}} = \frac{1.9306}{1.9306} = \mathbf{1.0000}$
- $w_2^{\text{norm}} = \frac{1.0000}{1.9306} \approx \mathbf{0.5180}$
- $w_3^{\text{norm}} = \frac{1.3973}{1.9306} \approx \mathbf{0.7237}$
- $w_4^{\text{norm}} = \frac{0.6712}{1.9306} \approx \mathbf{0.3477}$

Notice that transition 4, which has the highest probability ($55.5\%$) of being sampled due to its large TD error, is scaled down by weight $0.3477$ to prevent over-representing frequent updates and ensure unbiased expectation. $\blacksquare$

---

## 7. Deep Learning Connection & Application

### 1. Convolutional Nature Architecture (Mnih et al., 2015)
The classic Nature DQN architecture processes $84 \times 84 \times 4$ frames through:
1. Conv2D: 32 filters of size $8 \times 8$, stride 4, ReLU.
2. Conv2D: 64 filters of size $4 \times 4$, stride 2, ReLU.
3. Conv2D: 64 filters of size $3 \times 3$, stride 1, ReLU.
4. Fully Connected: 512 units, ReLU.
5. Linear Output: $|\mathcal{A}|$ Q-values (one for each joystick button combination).

### 2. Reward Clipping
To ensure stable gradients across vastly different Atari games without tuning game-specific hyperparameters, all positive rewards were clipped to $+1$, all negative rewards to $-1$, and neutral to $0$.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of the Part 5 hand calculations:
   - Target values $Y^{(1)} = 1.5400, Y^{(2)} = 2.0000$
   - Huber losses $0.5400, 0.7000$ and mean loss $0.6200$
   - Updated weights $\mathbf{W}_{\text{new}} = \begin{bmatrix} 0.5500 & 0.2000 \\ -0.3000 & 0.8500 \end{bmatrix}$ matching PyTorch / NumPy to $< 10^{-14}$.
2. Complete standalone Deep Q-Network agent (with Replay Buffer, Target Network, and $\epsilon$-greedy exploration) solving the CartPole balancing environment.

See implementation in:
[`11_reinforcement_learning/code/10_deep_q_networks.py`](./code/10_deep_q_networks.py)
