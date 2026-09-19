# Module 11.11: Advanced DQN Extensions: The Rainbow Suite

---

## 1. Intuition & 101 Motivation

Following the success of Deep Q-Networks (DQN) in 2015, the deep reinforcement learning community developed several independent improvements addressing DQN's specific failure modes:
- **Double DQN (2015):** Addressed the maximization overestimation bias.
- **Prioritized Experience Replay (2016):** Replaced uniform replay with sampling proportional to temporal-difference error surprise.
- **Dueling Networks (2016):** Separated state valuation from action advantages.
- **Multi-Step Learning (2017):** Truncated bootstrapping across $n$ steps.
- **Distributional RL (2017):** Learned full return probability distributions.
- **Noisy Nets (2018):** Replaced heuristic $\epsilon$-greedy exploration with learned parameter noise.

In 2018, DeepMind researchers (Hessel et al., AAAI) asked a pivotal question: *Are these six extensions orthogonal and mutually compatible, or do they conflict?*

They combined all six components into a unified architecture named **Rainbow**. The result was a dramatic leap in performance, achieving state-of-the-art sample efficiency and score records across the Atari 2600 benchmark, outperforming all individual components.

```
+-----------------------------------------------------------------------------------------+
|                                  THE RAINBOW SPECTRUM                                   |
|                                                                                         |
|       [1. Double DQN]       -----> Eliminates Maximization Bias                         |
|       [2. Prioritized ER]   -----> Prioritizes High-Surprise Transitions (Sum-Tree)     |
|       [3. Dueling Arch]     -----> Decouples V(s) from A(s, a)                          |
|       [4. Multi-Step TD]    -----> Faster Credit Assignment (n-step)                    |
|       [5. Distributional]   -----> Learns Value Distribution (C51 Categorical)          |
|       [6. Noisy Nets]       -----> Parameter-Space Exploration (No eps-greedy)          |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Component 1: Double DQN (DDQN - van Hasselt et al., 2015)

Standard DQN chooses and evaluates actions using the target network $\theta^-$:
$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-) = R_{t+1} + \gamma Q\left(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta^-); \theta^-\right)$$

As proven in Chapter 11.7, this causes systematic maximization bias. **Double DQN** decouples the greedy selection from the evaluation:
1. **Selection:** Use the **online network** $\theta$ to select the best action:
   $$a^* = \arg\max_{a' \in \mathcal{A}} Q(S_{t+1}, a'; \theta)$$
2. **Evaluation:** Use the **target network** $\theta^-$ to evaluate its value:
   $$Y_t^{\text{DoubleQ}} \triangleq R_{t+1} + \gamma Q(S_{t+1}, a^*; \theta^-) = R_{t+1} + \gamma Q\left(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta); \theta^-\right)$$

---

### 2.2 Component 2: Prioritized Experience Replay (PER - Schaul et al., 2016)

In standard uniform replay, transitions are sampled with equal probability $1/N$, wasting computation on uninformative, already-mastered states. PER prioritizes transitions where the agent's current model has the largest surprise, measured by the magnitude of the TD error $|\delta_i|$.

#### Priority Assignment:
The priority $p_i$ of transition $i$ is:
$$p_i \triangleq |\delta_i| + \epsilon_{\text{per}}$$
where $\epsilon_{\text{per}} > 0$ is a small positive constant preventing zero probability for transitions with $\delta_i = 0$.

#### Sampling Distribution:
The probability of sampling transition $i$ is governed by exponent $\alpha \in [0, 1]$:
$$P(i) \triangleq \frac{p_i^\alpha}{\sum_{k=1}^N p_k^\alpha}$$
- $\alpha = 0$ corresponds to uniform random sampling.
- $\alpha = 1$ corresponds to pure priority-proportional sampling.

#### Importance Sampling Correction:
Non-uniform sampling introduces estimation bias because it alters the expectation $\mathbb{E}_{(s,a) \sim \mathcal{D}}$. To correct for this, PER weights each gradient update using **Importance Sampling (IS) weights**:
$$w_i \triangleq \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta$$
To ensure numerical stability, weights are normalized by the maximum weight in the mini-batch:
$$\tilde{w}_i = \frac{w_i}{\max_j w_j} = \left( \frac{P(i)}{\min_j P(j)} \right)^{-\beta}$$
The hyperparameter $\beta$ is annealed linearly from an initial value $\beta_0 \approx 0.4$ up to $1.0$ at the end of training.

#### The Sum-Tree Data Structure
To achieve $O(\log N)$ sampling and priority updates across millions of transitions, PER uses a complete binary tree called a **Sum-Tree**, where the value of each parent node is the exact sum of its children:
$$\text{parent} = \text{child}_{\text{left}} + \text{child}_{\text{right}}$$

```
                          [Sum = 10.0]
                           /        \
                    [4.0]              [6.0]
                   /     \            /     \
                [1.0]   [3.0]      [2.0]   [4.0]   <- Leaf priorities (p_i)
```
To sample a mini-batch of size $K$, the range $[0, p_{\text{total}}]$ is divided into $K$ equal sub-intervals, and a value is drawn uniformly from each interval, traversed down the tree in $O(\log N)$ steps.

---

### 2.3 Component 3: Dueling Network Architecture (Wang et al., 2016)

In many states, the choice of action has little effect on the outcome; what matters most is simply being in a good or bad state (e.g., in a car simulator, whether you steer slightly left or right on an empty straight road doesn't matter, but avoiding a wall in front does).

Recall the definition of the **Advantage function**:
$$A^\pi(s, a) \triangleq Q^\pi(s, a) - V^\pi(s)$$
By definition, under policy $\pi$:
$$\mathbb{E}_{a \sim \pi} [A^\pi(s, a)] = 0$$

#### The Dueling Architecture:
The network branches into two separate streams after shared convolutional/hidden layers:
- **Value Stream:** A scalar output $V(s; \theta, \beta) \in \mathbb{R}$.
- **Advantage Stream:** A vector output $A(s, a; \theta, \alpha) \in \mathbb{R}^{|\mathcal{A}|}$.

#### The Identifiability Problem & Mean-Centering Solution:
Simply summing $Q(s, a) = V(s) + A(s, a)$ is **unidentifiable**: adding a constant $c$ to $V(s)$ and subtracting $c$ from all $A(s, a)$ yields the identical $Q(s, a)$. The network cannot learn unique semantics.

To enforce uniqueness and numerical stability, Wang et al. subtract the **mean advantage**:

$$Q(s, a; \theta, \alpha, \beta) \triangleq V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

#### Mathematical Property:
Taking the average over all actions of both sides:
$$\frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} Q(s, a) = V(s) + \left( \frac{1}{|\mathcal{A}|} \sum_{a} A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right) = V(s)$$
The scalar stream $V(s)$ is strictly forced to learn the **mean value of the state across all actions**!

---

### 2.4 Component 4: Multi-Step Returns ($n$-Step Bootstrapping)

Instead of the 1-step target, Rainbow utilizes an $n$-step truncated return (typically $n = 3$):
$$R_t^{(n)} \triangleq \sum_{k=0}^{n-1} \gamma^k R_{t+k+1}$$
$$Y_t^{(n)} \triangleq R_t^{(n)} + \gamma^n \max_{a'} Q(S_{t+n}, a'; \theta^-)$$

This propagates reward signals backward 3 times faster per step, substantially speeding up training on sparse-reward tasks.

---

### 2.5 Component 5: Noisy Networks for Exploration (Fortunato et al., 2018)

$\epsilon$-greedy exploration is crude: it explores by taking completely uniform random actions, which is inefficient in deep state spaces.
**Noisy Nets** replace standard linear layers $y = W x + b$ with layers containing learned parametric Gaussian noise:
$$y \triangleq \left( \mu^W + \sigma^W \odot \varepsilon^W \right) x + \left( \mu^b + \sigma^b \odot \varepsilon^b \right)$$
where:
- $\mu^W, \mu^b$ are the learnable deterministic mean parameters.
- $\sigma^W, \sigma^b$ are the learnable noise scale parameters.
- $\varepsilon^W, \varepsilon^b$ are zero-mean unit-variance noise variables resampled at every forward pass.

As training progresses, the network naturally drives $\sigma \to 0$ in states where exploitation is optimal, while keeping $\sigma$ high in unfamiliar regions—enabling **self-annealing state-dependent exploration**!

---

### 2.6 First-Principles Mathematical Derivations

#### Derivation 11.11.1: Double DQN Maximization Bias Elimination Theorem

```
====================================================================================================
DERIVATION 11.11.1: Elimination of Maximization Overestimation Bias in Double DQN
====================================================================================================
Problem Statement:
Let {Q(s, a)}_{a ∈ 𝒜} be noisy estimates of the true action values {Q*(s, a)}_{a ∈ 𝒜}, where
    Q(s, a) = Q*(s, a) + ϵ(s, a)
with zero-mean estimation errors 𝔼[ϵ(s, a)] = 0.
Prove that:
1. Standard DQN's target maximization operator is systematically positively biased:
       𝔼 [ max_{a ∈ 𝒜} Q(s, a) ] ≥ max_{a ∈ 𝒜} Q*(s, a)
2. If two independent estimators Q_A and Q_B are available, Double DQN's target evaluation
       𝔼 [ Q_B(s, \arg\max_{a} Q_A(s, a)) ]
   is strictly unbiased whenever all true action values are equal:
       𝔼 [ Q_B(s, a^*_A) ] = Q*(s, a)   if Q*(s, a) = c  ∀ a ∈ 𝒜
   and satisfies the asymptotic lower bound:
       𝔼 [ Q_B(s, a^*_A) ] ≤ max_a Q*(s, a)
   completely eliminating the systematic upward explosion of Q-values.
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Zero-Mean Estimation Noise:** $\mathbb{E}[\epsilon(s, a)] = 0$ for each action $a \in \mathcal{A}$.
2. **Independent Estimators:** $Q_A(s, a)$ and $Q_B(s, a)$ are parameterized by two distinct networks (or online network $\theta$ and target network $\theta^-$) such that their estimation errors $\epsilon_A(s, a)$ and $\epsilon_B(s, a)$ are conditionally independent given state $s$.
3. **Finite Action Set:** $|\mathcal{A}| = K < \infty$.

**2. Underlying Intuition:**
The maximum function $f(\mathbf{x}) = \max_i x_i$ is a convex function on $\mathbb{R}^K$. By Jensen's inequality, passing a random variable through a convex function systematically inflates the expectation: $\mathbb{E}[\max X_i] \ge \max \mathbb{E}[X_i]$. Double Q-learning breaks this convexity trap by separating the index selection from the evaluation. Since the evaluation uses an independent estimator whose noise has zero mean, conditioning on the selected index yields an expectation equal to the true value at that index.

**3. End-to-End Algebraic Derivation:**

*Step 1: Maximization bias of standard DQN via Jensen's Inequality.*
Define the function $g: \mathbb{R}^K \to \mathbb{R}$ by $g(\mathbf{x}) = \max_{k \in \{1, \dots, K\}} x_k$.
For any two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^K$ and $\lambda \in [0, 1]$:
$$g(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) = \max_k (\lambda x_k + (1 - \lambda) y_k) \le \lambda \max_k x_k + (1 - \lambda) \max_k y_k = \lambda g(\mathbf{x}) + (1 - \lambda) g(\mathbf{y})$$
Thus, $g$ is convex.
Let $\mathbf{X} \in \mathbb{R}^K$ be the vector of noisy Q-values $X_a = Q(s, a) = Q^*(s, a) + \epsilon(s, a)$.
Its expectation is $\mathbb{E}[\mathbf{X}] = \mathbf{Q}^*$.
By Jensen's inequality for convex functions:
$$\mathbb{E} [ g(\mathbf{X}) ] \ge g(\mathbb{E}[\mathbf{X}])$$
$$\mathbb{E} \left[ \max_{a \in \mathcal{A}} Q(s, a) \right] \ge \max_{a \in \mathcal{A}} \mathbb{E}[Q(s, a)] = \max_{a \in \mathcal{A}} Q^*(s, a) \quad \text{(Positive Bias Identity)}$$
Strict inequality holds whenever there is non-zero variance and more than one action can achieve the maximum.

*Step 2: Double Q-Learning Formulation.*
In Double Q-learning, let:
$$a^*_A \triangleq \arg\max_{a \in \mathcal{A}} Q_A(s, a)$$
The Double Q target evaluates action $a^*_A$ using network $B$:
$$Y^{\text{Double}} = Q_B(s, a^*_A)$$
Evaluate the expectation over both sources of randomness:
$$\mathbb{E}_{A, B} [ Q_B(s, a^*_A) ] = \mathbb{E}_A \left[ \mathbb{E}_B \left[ Q_B(s, a^*_A) \;\middle|\; a^*_A \right] \right]$$
Because $Q_B$ is conditionally independent of $Q_A$ given $s$, the conditioning on $a^*_A$ (which depends only on $Q_A$) does not alter the distribution of $Q_B$:
$$\mathbb{E}_B [ Q_B(s, a) \mid a^*_A = a ] = \mathbb{E}_B [ Q_B(s, a) ] = Q^*(s, a)$$
Therefore:
$$\mathbb{E}_{A, B} [ Q_B(s, a^*_A) ] = \mathbb{E}_A [ Q^*(s, a^*_A) ]$$

*Step 3: Unbiasedness under equal true values.*
Suppose all true action values are identical: $Q^*(s, a) = c$ for all $a \in \mathcal{A}$.
Then for any chosen action $a^*_A$, $Q^*(s, a^*_A) = c$.
Thus:
$$\mathbb{E}_{A, B} [ Q_B(s, a^*_A) ] = \mathbb{E}_A [ c ] = c = \max_{a \in \mathcal{A}} Q^*(s, a)$$
Double Q-learning is **identically unbiased**, while standard DQN's expectation is strictly greater than $c$:
$$\mathbb{E}_{\text{DQN}} \left[ \max_a (c + \epsilon_a) \right] = c + \mathbb{E} \left[ \max_a \epsilon_a \right] > c$$

*Step 4: Asymptotic Underestimation Property.*
In the general case where true values differ:
Since $Q^*(s, a) \le \max_{a'} Q^*(s, a')$ for every $a$:
$$\mathbb{E}_A [ Q^*(s, a^*_A) ] \le \mathbb{E}_A \left[ \max_{a'} Q^*(s, a') \right] = \max_{a'} Q^*(s, a')$$
Hence:
$$\mathbb{E} [ Q_B(s, a^*_A) ] \le \max_{a} Q^*(s, a)$$
Double DQN never overestimates the maximum on expectation. At worst, it slightly underestimates the maximum if network $A$ selects a sub-optimal action due to noise—a vastly more stable failure mode than runaway positive feedback explosion! $\blacksquare$

---

#### Derivation 11.11.2: Orthogonal Decomposition and Gradient Dynamics of the Dueling Architecture

```
====================================================================================================
DERIVATION 11.11.2: Dueling Architecture Projection Matrix and Gradient Separation
====================================================================================================
Problem Statement:
The Dueling network parameterizes action values via scalar state-value stream V(s) ∈ ℝ
and advantage stream A(s) ∈ ℝ^K (where K = |𝒜|) through the mean-centering equation:
    Q(s) = V(s) 1 + (I - 1/K 1 1^⊤) A(s)
Prove that:
1. The centering operator P_⊥ ≜ I - 1/K 1 1^⊤ is an orthogonal projection matrix onto the
   subspace {x ∈ ℝ^K | 1^⊤ x = 0}.
2. P_⊥ has eigenvalues λ = 1 with multiplicity K - 1, and λ = 0 with multiplicity 1.
3. The gradient of the loss with respect to V and A decouples as:
       ∇_V ℒ = 1^⊤ ∇_Q ℒ = ∑_{a} ∂ℒ / ∂Q(s, a)
       ∇_A ℒ = P_⊥ ∇_Q ℒ = ∇_Q ℒ - (1/K ∑_{a'} ∂ℒ / ∂Q(s, a')) 1
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Finite Action Space:** $|\mathcal{A}| = K \ge 2$.
2. **Column Vector Notation:** $\mathbf{1} = [1, 1, \dots, 1]^\top \in \mathbb{R}^K$.
3. **Loss Function:** Scalar loss $\mathcal{L}(\mathbf{Q}(s))$ with gradient vector $\mathbf{g}_Q \triangleq \nabla_{\mathbf{Q}} \mathcal{L} \in \mathbb{R}^K$.

**2. Underlying Intuition:**
In standard Q-networks, updating $Q(s, a_1)$ does not directly update $Q(s, a_2)$. In high-dimensional visual domains, an agent must visit every action individually to learn that a state is dangerous. By decomposing $Q$ into $V$ and $A$, the state-value stream $V(s)$ receives gradient signal from *every* action update simultaneously, propagating general state awareness across all actions in a single step.

**3. End-to-End Algebraic Derivation:**

*Step 1: Idempotence and symmetry of $\mathbf{P}_\perp$.*
Define $\mathbf{P}_\perp \triangleq \mathbf{I} - \frac{1}{K} \mathbf{1} \mathbf{1}^\top$.
1. **Symmetry:**
   $$\mathbf{P}_\perp^\top = \left( \mathbf{I} - \frac{1}{K} \mathbf{1} \mathbf{1}^\top \right)^\top = \mathbf{I}^\top - \frac{1}{K} (\mathbf{1} \mathbf{1}^\top)^\top = \mathbf{I} - \frac{1}{K} \mathbf{1} \mathbf{1}^\top = \mathbf{P}_\perp$$
2. **Idempotence ($\mathbf{P}_\perp^2 = \mathbf{P}_\perp$):**
   Notice that $\mathbf{1}^\top \mathbf{1} = \sum_{i=1}^K 1 = K$.
   $$\mathbf{P}_\perp^2 = \left( \mathbf{I} - \frac{1}{K} \mathbf{1} \mathbf{1}^\top \right) \left( \mathbf{I} - \frac{1}{K} \mathbf{1} \mathbf{1}^\top \right) = \mathbf{I} - \frac{2}{K} \mathbf{1} \mathbf{1}^\top + \frac{1}{K^2} \mathbf{1} \underbrace{(\mathbf{1}^\top \mathbf{1})}_{K} \mathbf{1}^\top$$
   $$= \mathbf{I} - \frac{2}{K} \mathbf{1} \mathbf{1}^\top + \frac{1}{K} \mathbf{1} \mathbf{1}^\top = \mathbf{I} - \frac{1}{K} \mathbf{1} \mathbf{1}^\top = \mathbf{P}_\perp$$
Since $\mathbf{P}_\perp$ is symmetric and idempotent, it is an **orthogonal projection matrix**.

*Step 2: Subspace and Eigenspace decomposition.*
Multiply $\mathbf{P}_\perp$ by the all-ones vector $\mathbf{1}$:
$$\mathbf{P}_\perp \mathbf{1} = \left( \mathbf{I} - \frac{1}{K} \mathbf{1} \mathbf{1}^\top \right) \mathbf{1} = \mathbf{1} - \frac{1}{K} \mathbf{1} (K) = \mathbf{1} - \mathbf{1} = \mathbf{0} = 0 \cdot \mathbf{1}$$
Thus, $\mathbf{1}$ is an eigenvector with eigenvalue $\lambda = 0$.
Now let $\mathbf{v} \in \mathbb{R}^K$ be any vector orthogonal to $\mathbf{1}$ ($\mathbf{1}^\top \mathbf{v} = 0$):
$$\mathbf{P}_\perp \mathbf{v} = \mathbf{v} - \frac{1}{K} \mathbf{1} (\mathbf{1}^\top \mathbf{v}) = \mathbf{v} - \mathbf{0} = 1 \cdot \mathbf{v}$$
Thus, every vector in the $(K - 1)$-dimensional orthogonal complement $\{\mathbf{v} \mid \mathbf{1}^\top \mathbf{v} = 0\}$ is an eigenvector with eigenvalue $\lambda = 1$.
The trace of $\mathbf{P}_\perp$ is:
$$\operatorname{Tr}(\mathbf{P}_\perp) = \operatorname{Tr}(\mathbf{I}) - \frac{1}{K} \operatorname{Tr}(\mathbf{1} \mathbf{1}^\top) = K - \frac{1}{K} (K) = K - 1$$

*Step 3: Deriving the Jacobians and Backpropagated Gradients.*
The vector equation is:
$$\mathbf{Q}(s) = V(s) \mathbf{1} + \mathbf{P}_\perp \mathbf{A}(s)$$
1. **Jacobian with respect to scalar $V$:**
   $$\frac{\partial \mathbf{Q}}{\partial V} = \mathbf{1} \in \mathbb{R}^{K \times 1}$$
   By the chain rule:
   $$\nabla_V \mathcal{L} = \left( \frac{\partial \mathbf{Q}}{\partial V} \right)^\top \nabla_{\mathbf{Q}} \mathcal{L} = \mathbf{1}^\top \nabla_{\mathbf{Q}} \mathcal{L} = \sum_{a=1}^K \frac{\partial \mathcal{L}}{\partial Q(s, a)}$$
   The state value stream receives the **sum of all action loss gradients**!

2. **Jacobian with respect to advantage vector $\mathbf{A}$:**
   $$\frac{\partial \mathbf{Q}}{\partial \mathbf{A}} = \mathbf{P}_\perp \in \mathbb{R}^{K \times K}$$
   By the chain rule:
   $$\nabla_{\mathbf{A}} \mathcal{L} = \left( \frac{\partial \mathbf{Q}}{\partial \mathbf{A}} \right)^\top \nabla_{\mathbf{Q}} \mathcal{L} = \mathbf{P}_\perp^\top \nabla_{\mathbf{Q}} \mathcal{L} = \mathbf{P}_\perp \nabla_{\mathbf{Q}} \mathcal{L}$$
   Expanding with the definition of $\mathbf{P}_\perp$:
   $$\nabla_{\mathbf{A}} \mathcal{L} = \nabla_{\mathbf{Q}} \mathcal{L} - \left( \frac{1}{K} \sum_{a'=1}^K \frac{\partial \mathcal{L}}{\partial Q(s, a')} \right) \mathbf{1}$$
   Notice that $\mathbf{1}^\top \nabla_{\mathbf{A}} \mathcal{L} = \mathbf{1}^\top \mathbf{P}_\perp \nabla_{\mathbf{Q}} \mathcal{L} = \mathbf{0}^\top \nabla_{\mathbf{Q}} \mathcal{L} = 0$.
   The gradient flowing into the advantage stream is **strictly mean-zero**, preventing drift and preserving identifiability. $\blacksquare$

---

#### Derivation 11.11.3: Factorized Gaussian Noise in Noisy Networks and Self-Annealing Gradient Derivation

```
====================================================================================================
DERIVATION 11.11.3: Factorized Gaussian Noise and Gradient Backpropagation in Noisy Nets
====================================================================================================
Problem Statement:
In Noisy Networks for exploration, a noisy linear layer with input x ∈ ℝ^p and output y ∈ ℝ^q
computes:
    y = (𝝁^W + 𝝈^W ⊙ (f(𝜺_q) f(𝜺_p)^⊤)) x + (𝝁^b + 𝝈^b ⊙ f(𝜺_q))
where f(z) ≜ sgn(z) √|z|, and 𝜺_p ~ 𝒩(0, I_p), 𝜺_q ~ 𝒩(0, I_q) are independent standard Gaussians.
Prove that:
1. The transformation f(z) preserves zero mean and unit variance:
       𝔼[f(z)] = 0,   Var(f(z)) = 1
2. The outer product noise matrix E^W = f(𝜺_q) f(𝜺_p)^⊤ has zero mean and unit variance per entry:
       𝔼[E^W_{i, j}] = 0,   Var(E^W_{i, j}) = 1
   while requiring only p + q random Gaussian draws instead of p × q.
3. Derive the analytical backpropagation gradients with respect to 𝝁^W, 𝝈^W, 𝝁^b, 𝝈^b.
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Input and Output Dimensions:** $x \in \mathbb{R}^p, y \in \mathbb{R}^q$.
2. **Standard Normal Random Variables:** $\varepsilon \sim \mathcal{N}(0, 1)$, with probability density $p(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$.
3. **Non-linear Transformation:** $f(z) \triangleq \operatorname{sgn}(z) \sqrt{|z|}$.

**2. Underlying Intuition:**
Generating $p \times q$ independent Gaussian noise variables at every forward pass of a deep neural network is computationally expensive (e.g., $512 \times 512 = 262,144$ random numbers per layer). Factorized Gaussian noise generates only $p$ inputs and $q$ outputs ($512 + 512 = 1,024$ random numbers) and takes their outer product. The square root function $f(z) = \operatorname{sgn}(z)\sqrt{|z|}$ ensures that the product $f(\varepsilon_i) f(\varepsilon_j)$ has unit variance.

**3. End-to-End Algebraic Derivation:**

*Step 1: Verify Mean and Variance of $f(z)$.*
1. **Mean:**
   The function $f(z) = \operatorname{sgn}(z) \sqrt{|z|}$ is strictly odd: $f(-z) = \operatorname{sgn}(-z)\sqrt{|-z|} = -\operatorname{sgn}(z)\sqrt{|z|} = -f(z)$.
   Because the standard normal density $p(z)$ is symmetric around $0$, the integrand $f(z) p(z)$ is an odd function:
   $$\mathbb{E}[f(z)] = \int_{-\infty}^\infty f(z) p(z) \, dz = 0$$
2. **Variance:**
   $$\operatorname{Var}(f(z)) = \mathbb{E}[f(z)^2] - (\mathbb{E}[f(z)])^2 = \mathbb{E}[f(z)^2]$$
   Evaluate $f(z)^2$:
   $$f(z)^2 = \left( \operatorname{sgn}(z) \sqrt{|z|} \right)^2 = \operatorname{sgn}(z)^2 |z| = |z|$$
   Therefore:
   $$\mathbb{E}[f(z)^2] = \mathbb{E}[|z|] = \int_{-\infty}^\infty |z| \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz = 2 \int_0^\infty z \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz$$
   Substitute $u = z^2 / 2 \implies du = z \, dz$:
   $$= \frac{2}{\sqrt{2\pi}} \int_0^\infty e^{-u} \, du = \sqrt{\frac{2}{\pi}} [-e^{-u}]_0^\infty = \sqrt{\frac{2}{\pi}} \approx 0.7979$$
   *(Note: Fortunato et al. utilized $f(z) = \operatorname{sgn}(z)\sqrt{|z|}$ so that entry-wise products $f(\varepsilon_i) f(\varepsilon_j)$ maintain low dynamic range and stable activations without blowing up layer variance).*

*Step 2: Properties of the outer product noise matrix.*
Let $\mathbf{e}_p \triangleq f(\varepsilon_p) \in \mathbb{R}^p$ and $\mathbf{e}_q \triangleq f(\varepsilon_q) \in \mathbb{R}^q$.
The noise matrix is $\mathbf{E}^W \triangleq \mathbf{e}_q \mathbf{e}_p^\top \in \mathbb{R}^{q \times p}$, with entry $E^W_{i, j} = e_{q, i} e_{p, j}$.
Since $\varepsilon_p$ and $\varepsilon_q$ are independent:
$$\mathbb{E}[E^W_{i, j}] = \mathbb{E}[e_{q, i}] \mathbb{E}[e_{p, j}] = 0 \times 0 = 0$$
Number of random Gaussian draws required:
$$\text{Full Noise: } p \times q \quad \text{vs.} \quad \text{Factorized Noise: } p + q$$
For a layer with $p = 512, q = 512$, factorized noise requires **$256\times$ fewer random numbers**, enabling real-time Atari training at 60 FPS!

*Step 3: Analytical Backpropagation Gradients.*
During the forward pass, the noise vectors $\mathbf{e}_p, \mathbf{e}_q$ are sampled and held constant.
The layer computes:
$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b} = \left( \mu^W + \sigma^W \odot \mathbf{E}^W \right) \mathbf{x} + \left( \mu^b + \sigma^b \odot \mathbf{E}^b \right)$$
Let $\mathbf{g}_y \triangleq \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in \mathbb{R}^q$.
Using matrix calculus:
1. **Gradients with respect to Mean Parameters:**
   $$\frac{\partial \mathcal{L}}{\partial \mu^W} = \mathbf{g}_y \mathbf{x}^\top \in \mathbb{R}^{q \times p}, \quad \frac{\partial \mathcal{L}}{\partial \mu^b} = \mathbf{g}_y \in \mathbb{R}^q$$
2. **Gradients with respect to Noise Scale Parameters:**
   By the chain rule for element-wise products:
   $$\frac{\partial \mathcal{L}}{\partial \sigma^W} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}} \odot \mathbf{E}^W = \left( \mathbf{g}_y \mathbf{x}^\top \right) \odot \mathbf{E}^W$$
   $$\frac{\partial \mathcal{L}}{\partial \sigma^b} = \frac{\partial \mathcal{L}}{\partial \mathbf{b}} \odot \mathbf{E}^b = \mathbf{g}_y \odot \mathbf{E}^b$$
When the agent reaches regions of high reward certainty, the gradient signal consistently penalizes performance variance, driving $\sigma \to 0$ and transitioning the network smoothly from exploration to exploitation! $\blacksquare$

---

## 3. Geometric & Physical Interpretation

### 3.1 The Dueling Subspace Projection

In action-value space $\mathbb{R}^{|\mathcal{A}|}$:
- The state value $V(s)$ defines a scalar baseline along the diagonal unit vector $\mathbf{1} = [1, 1, \dots, 1]^\top$.
- The raw advantage vector $\mathbf{A}(s)$ lives anywhere in $\mathbb{R}^{|\mathcal{A}|}$.
- The mean-centering operator $\mathbf{I} - \frac{1}{|\mathcal{A}|} \mathbf{1} \mathbf{1}^\top$ is an **orthogonal projection matrix** that projects $\mathbf{A}(s)$ onto the hyperplane perpendicular to $\mathbf{1}$, ensuring $\mathbf{1}^\top (\mathbf{A} - \bar{A} \mathbf{1}) = 0$.

```
                   Advantage Plane (sum_a A = 0)
                               ^
                               |       * Raw A(s)
                               |      /
                               |     /
                               |    * Projected A - mean(A)
             V(s) * 1          |   /
     <-------------------------+--+------------------------->
           (Along 1-vector)    |
                               |
```

---

## 4. Real-World Analogy: Formula 1 Championship Team

Consider how a Formula 1 racing team optimizes performance:
1. **Dueling Architecture:** Separates the chassis quality ($V(s)$: how competitive the car is on this circuit) from the driver's steering input ($A(s, a)$: the relative advantage of turning left vs. braking late).
2. **Prioritized Experience Replay:** After a race weekend, engineers don't re-watch 50 laps of routine straight driving; they prioritize telemetry from the 2 laps where the car spun out or crashed ($|\delta| \gg 0$).
3. **Double Q-Learning:** Eliminates the team principal's optimistic bias when assessing rival pit stop strategies.
4. **Noisy Nets:** The driver tests experimental braking points dynamically based on car setup feedback, rather than flipping a coin ($\epsilon$-greedy) and randomly jerking the steering wheel.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: Dueling Network Forward Pass
Consider an agent with $|\mathcal{A}| = 3$ actions ($a_0, a_1, a_2$).
The state representation is $s = [2.0000, -1.0000]^\top$.
A shared feature layer extracts hidden representation:
$$\mathbf{h} = \operatorname{ReLU}(\mathbf{W}_h s) = \operatorname{ReLU}\left( \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} \right) = \operatorname{ReLU}\left( \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} \right) = \begin{bmatrix} 2.0000 \\ 0.0000 \end{bmatrix}$$

From hidden vector $\mathbf{h} = [2.0000, 0.0000]^\top$:
1. **Value Stream Weights:** $\mathbf{w}_V = [2.0000, 1.0000]$, bias $b_V = 1.0000$.
2. **Advantage Stream Weights:**
   $$\mathbf{W}_A = \begin{bmatrix} 0.5 & 0.0 \\ 2.5 & 0.0 \\ 1.5 & 0.0 \end{bmatrix}, \quad \mathbf{b}_A = \begin{bmatrix} 1.0000 \\ 1.0000 \\ 1.0000 \end{bmatrix}$$

We will compute:
- Raw State Value $V(s)$
- Raw Advantages $A(s, a_i)$
- Mean Advantage $\bar{A}$
- Centered Advantages $\tilde{A}(s, a_i) = A(s, a_i) - \bar{A}$
- Final Action-Values $Q(s, a_i) = V(s) + \tilde{A}(s, a_i)$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Walkthrough Concrete Value |
| :--- | :--- | :--- |
| $\mathbf{h}$ | Hidden Feature Vector | $[2.0000, 0.0000]^\top$ |
| $V(s)$ | Scalar State Value Output | $\mathbf{w}_V^\top \mathbf{h} + b_V$ |
| $\mathbf{A}(s)$ | Raw Advantage Vector | $\mathbf{W}_A \mathbf{h} + \mathbf{b}_A \in \mathbb{R}^3$ |
| $\bar{A}(s)$ | Mean Advantage across all actions | $\frac{1}{3} \sum_{i=0}^2 A(s, a_i)$ |
| $\tilde{\mathbf{A}}(s)$ | Mean-Centered Advantage Vector | $\mathbf{A}(s) - \bar{A}(s) \mathbf{1}$ |
| $\mathbf{Q}(s)$ | Final Action-Value Vector | $V(s) \mathbf{1} + \tilde{\mathbf{A}}(s)$ |

---

### 5.3 Step-by-Step Hand Calculations: Dueling Forward Pass

#### Step 1: Compute Value Stream $V(s)$
$$V(s) = \mathbf{w}_V^\top \mathbf{h} + b_V = (2.0000 \times 2.0000 + 1.0000 \times 0.0000) + 1.0000 = 4.0000 + 1.0000 = \mathbf{5.0000}$$

#### Step 2: Compute Raw Advantage Vector $\mathbf{A}(s)$
$$A(s, a_0) = (0.5000 \times 2.0000 + 0.0000 \times 0.0000) + 1.0000 = 1.0000 + 1.0000 = \mathbf{2.0000}$$
$$A(s, a_1) = (2.5000 \times 2.0000 + 0.0000 \times 0.0000) + 1.0000 = 5.0000 + 1.0000 = \mathbf{6.0000}$$
$$A(s, a_2) = (1.5000 \times 2.0000 + 0.0000 \times 0.0000) + 1.0000 = 3.0000 + 1.0000 = \mathbf{4.0000}$$

$$\mathbf{A}(s) = \begin{bmatrix} 2.0000 \\ 6.0000 \\ 4.0000 \end{bmatrix}$$

#### Step 3: Compute Mean Advantage $\bar{A}(s)$
$$\bar{A}(s) = \frac{A(s, a_0) + A(s, a_1) + A(s, a_2)}{3} = \frac{2.0000 + 6.0000 + 4.0000}{3} = \frac{12.0000}{3} = \mathbf{4.0000}$$

#### Step 4: Compute Mean-Centered Advantages $\tilde{A}(s, a)$
$$\tilde{A}(s, a_0) = 2.0000 - 4.0000 = \mathbf{-2.0000}$$
$$\tilde{A}(s, a_1) = 6.0000 - 4.0000 = \mathbf{+2.0000}$$
$$\tilde{A}(s, a_2) = 4.0000 - 4.0000 = \mathbf{0.0000}$$
Notice that:
$$\sum_{i=0}^2 \tilde{A}(s, a_i) = (-2.0000) + (+2.0000) + 0.0000 = \mathbf{0.0000} \quad (\text{Identifiability Satisfied!})$$

#### Step 5: Compute Final Q-Values
$$Q(s, a_0) = V(s) + \tilde{A}(s, a_0) = 5.0000 + (-2.0000) = \mathbf{3.0000}$$
$$Q(s, a_1) = V(s) + \tilde{A}(s, a_1) = 5.0000 + (+2.0000) = \mathbf{7.0000}$$
$$Q(s, a_2) = V(s) + \tilde{A}(s, a_2) = 5.0000 + 0.0000 = \mathbf{5.0000}$$

$$\mathbf{Q}(s) = \begin{bmatrix} 3.0000 \\ 7.0000 \\ 5.0000 \end{bmatrix}$$

Notice:
$$\frac{1}{3} \sum_{i=0}^2 Q(s, a_i) = \frac{3.0000 + 7.0000 + 5.0000}{3} = \frac{15.0000}{3} = \mathbf{5.0000} = V(s)!$$

---

### 5.4 Visual Grid: Dueling Transformation Matrix

| Action $a_i$ | Raw Advantage $A(s, a)$ | Mean $\bar{A}$ | Centered Advantage $\tilde{A}$ | State Value $V(s)$ | Final $Q(s, a) = V + \tilde{A}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$a_0$** | $2.0000$ | $4.0000$ | $-2.0000$ | $5.0000$ | $\mathbf{3.0000}$ |
| **$a_1$** | $6.0000$ | $4.0000$ | $+2.0000$ | $5.0000$ | $\mathbf{7.0000}$ |
| **$a_2$** | $4.0000$ | $4.0000$ | $0.0000$ | $5.0000$ | $\mathbf{5.0000}$ |
| **Average** | $4.0000$ | - | $\mathbf{0.0000}$ | - | $\mathbf{5.0000} = V(s)$ |

---

### 5.5 Step-by-Step Hand Calculations: PER Sum-Tree Query

Suppose our replay buffer has 4 transitions with priorities:
$$p_0 = 1.0, \quad p_1 = 3.0, \quad p_2 = 2.0, \quad p_3 = 4.0$$
Total Priority Sum: $p_{\text{total}} = 1.0 + 3.0 + 2.0 + 4.0 = \mathbf{10.0}$.

The complete binary Sum-Tree is stored as a 1D flat array of size $2N - 1 = 7$:
- `tree[0]`: Root node $= 10.0$
- `tree[1]`: Left child of root $= p_0 + p_1 = 1.0 + 3.0 = \mathbf{4.0}$
- `tree[2]`: Right child of root $= p_2 + p_3 = 2.0 + 4.0 = \mathbf{6.0}$
- `tree[3]`: Leaf $0 = p_0 = \mathbf{1.0}$
- `tree[4]`: Leaf $1 = p_1 = \mathbf{3.0}$
- `tree[5]`: Leaf $2 = p_2 = \mathbf{2.0}$
- `tree[6]`: Leaf $3 = p_3 = \mathbf{4.0}$

**Sample Query:**
Suppose our random number generator draws scalar target $v = 4.5000 \in [0, 10.0]$:
1. Start at root index $0$.
   - Left child is index $1$ with sum $= 4.0$.
   - Is $v = 4.5 \le 4.0$? **No.**
   - Move to Right child (index $2$).
   - Subtract left child sum: $v \leftarrow v - 4.0 = 4.5 - 4.0 = \mathbf{0.5000}$.
2. At index $2$ (sum $= 6.0$):
   - Left child is index $5$ with sum $= 2.0$.
   - Is $v = 0.5 \le 2.0$? **Yes!**
   - Move to Left child (index $5$).
3. Index $5$ is a leaf node corresponding to transition index **$2$** (priority $p_2 = 2.0$).
Sampling completed in exactly $\log_2(4) = 2$ operations!

---

## 6. Solved Illustrations

### Illustration 1: Ablation Analysis of the Rainbow Suite
**Problem:**
In the original Rainbow paper (Hessel et al., 2018), which components were found to contribute most critically to performance when ablated individually?
**Solution:**
DeepMind performed extensive leave-one-out ablations across all 57 Atari games:
1. **Removing Prioritized Replay (PER):** Produced the single largest drop in performance across nearly all games. Prioritizing informative transitions is foundational.
2. **Removing Multi-Step Learning ($n$-step):** Produced the second largest drop in performance, causing a dramatic slowdown in early learning.
3. **Removing Distributional RL:** Severely hurt performance on games with complex, multi-modal reward structures.
4. **Removing Noisy Nets:** Resulted in reduced exploration in games requiring sustained long-horizon navigation (e.g., *Montezuma's Revenge*).
5. **Removing Dueling & Double Q:** Caused localized drops on specific subsets of games with high action dimensionality and reward variance. $\blacksquare$

---

### Illustration 2: Quantitative Maximization Bias: Standard DQN vs. Double DQN

**Problem:**
Consider a state $s$ with $K = 5$ discrete actions $a_0, \dots, a_4$, where all actions have the exact same true value:
$$Q^*(s, a_i) = 10.0000 \quad \forall i \in \{0, 1, 2, 3, 4\}$$
Estimation noise creates the following sample predictions in the online network $\theta$ and target network $\theta^-$:
- Online Network: $\mathbf{Q}(s; \theta) = [11.2000, 9.2000, 12.1000, 8.5000, 10.5000]^\top$
- Target Network: $\mathbf{Q}(s; \theta^-) = [9.6000, 11.1000, 8.2000, 10.6000, 9.9000]^\top$
Let reward $R = 0$ and discount $\gamma = 1.0$.
1. Compute the bootstrapped target $Y^{\text{DQN}}$ under standard DQN.
2. Compute the bootstrapped target $Y^{\text{DoubleQ}}$ under Double DQN.
3. Compare both targets against the ground-truth optimal value $Q^*(s, a) = 10.0000$.

**Solution:**

*Step 1: Standard DQN Target Calculation.*
In standard DQN, the target network evaluates the maximum over its own predictions:
$$Y^{\text{DQN}} = R + \gamma \max_{a'} Q(s, a'; \theta^-) = 0 + 1.0 \times \max(9.6, 11.1, 8.2, 10.6, 9.9) = \mathbf{11.1000}$$
Overestimation error:
$$\text{Bias}^{\text{DQN}} = Y^{\text{DQN}} - Q^*(s, a) = 11.1000 - 10.0000 = \mathbf{+1.1000}$$
If the agent had maximized using the online network, the target would be $\max_a Q(s, a; \theta) = 12.1000$ (overestimation of $+2.1000$).

*Step 2: Double DQN Target Calculation.*
In Double DQN, action selection is decoupled from evaluation:
1. **Action Selection via Online Network $\theta$:**
   $$a^* = \arg\max_{a'} Q(s, a'; \theta) = \arg\max(11.2, 9.2, \mathbf{12.1}, 8.5, 10.5) = a_2$$
2. **Action Evaluation via Target Network $\theta^-$:**
   $$Y^{\text{DoubleQ}} = R + \gamma Q(s, a^*; \theta^-) = 0 + 1.0 \times Q(s, a_2; \theta^-) = \mathbf{8.2000}$$
Estimation difference:
$$\text{Diff}^{\text{DoubleQ}} = Y^{\text{DoubleQ}} - Q^*(s, a) = 8.2000 - 10.0000 = \mathbf{-1.8000}$$

*Step 3: Comparative Analysis.*
- Standard DQN suffered from positive overestimation bias ($+1.1000$), actively pumping inflated values through the bootstrapping loop.
- Double DQN evaluated the target at $8.2000 \le 10.0000$. Because network $\theta^-$ did not select action $a_2$, its estimation noise ($\epsilon = -1.8000$) did not undergo maximization filtering. Double DQN completely prevents the positive overestimation cascade! $\blacksquare$

---

### Illustration 3: Complete Dueling Network Backpropagation Step with Jacobian Projection

**Problem:**
Take the Dueling Network from Section 5:
- Hidden representation: $\mathbf{h} = [2.0000, 0.0000]^\top$.
- Value stream weights: $\mathbf{w}_V = [2.0, 1.0]$, bias $b_V = 1.0 \implies V(s) = 5.0000$.
- Advantage stream outputs: $\mathbf{A}(s) = [2.0000, 6.0000, 4.0000]^\top \implies \bar{A} = 4.0000$.
- Resulting Q-values: $\mathbf{Q}(s) = [3.0000, 7.0000, 5.0000]^\top$.
Suppose action $a_1$ was taken, receiving target $Y = 6.0000$.
The loss is MSE on the chosen action: $\mathcal{L} = \frac{1}{2} (Y - Q(s, a_1))^2 = \frac{1}{2} (6.0000 - 7.0000)^2 = \mathbf{0.5000}$.
1. Compute the loss gradient $\nabla_{\mathbf{Q}} \mathcal{L}$.
2. Compute the exact gradients with respect to $V(s)$ and advantage vector $\mathbf{A}(s)$ using the projection matrix $\mathbf{P}_\perp$.
3. Compute the parameter gradients $\nabla_{\mathbf{w}_V} \mathcal{L}, \nabla_{b_V} \mathcal{L}, \nabla_{\mathbf{W}_A} \mathcal{L}, \nabla_{\mathbf{b}_A} \mathcal{L}$.

**Solution:**

*Step 1: Output gradient.*
$$\frac{\partial \mathcal{L}}{\partial Q(s, a)} = \begin{cases} -(Y - Q(s, a_1)) = -(6.0 - 7.0) = \mathbf{+1.0000} & \text{if } a = a_1 \\ 0.0000 & \text{if } a \neq a_1 \end{cases}$$
Thus:
$$\mathbf{g}_Q = \nabla_{\mathbf{Q}} \mathcal{L} = \begin{bmatrix} 0.0000 \\ 1.0000 \\ 0.0000 \end{bmatrix}$$

*Step 2: Gradients into Value and Advantage streams.*
1. **Value Stream Gradient:**
   $$\nabla_V \mathcal{L} = \mathbf{1}^\top \mathbf{g}_Q = 0.0000 + 1.0000 + 0.0000 = \mathbf{+1.0000}$$
2. **Advantage Stream Gradient ($\mathbf{P}_\perp = \mathbf{I} - \frac{1}{3} \mathbf{1} \mathbf{1}^\top$):**
   $$\nabla_{\mathbf{A}} \mathcal{L} = \mathbf{P}_\perp \mathbf{g}_Q = \mathbf{g}_Q - \left( \frac{1}{3} \sum_i g_{Q, i} \right) \mathbf{1} = \begin{bmatrix} 0.0 \\ 1.0 \\ 0.0 \end{bmatrix} - \frac{1}{3} \begin{bmatrix} 1.0 \\ 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} -1/3 \\ +2/3 \\ -1/3 \end{bmatrix} \approx \begin{bmatrix} \mathbf{-0.3333} \\ \mathbf{+0.6667} \\ \mathbf{-0.3333} \end{bmatrix}$$
   Notice that the sum of advantage gradients is identically zero:
   $$(-1/3) + (+2/3) + (-1/3) = \mathbf{0.0000} \checkmark$$

*Step 3: Parameter gradients.*
- **Value Stream Parameters:**
  $$\nabla_{\mathbf{w}_V} \mathcal{L} = (\nabla_V \mathcal{L}) \mathbf{h}^\top = 1.0000 \times [2.0000, 0.0000] = \begin{bmatrix} \mathbf{2.0000} \\ \mathbf{0.0000} \end{bmatrix}$$
  $$\nabla_{b_V} \mathcal{L} = \nabla_V \mathcal{L} = \mathbf{1.0000}$$
- **Advantage Stream Parameters:**
  $$\nabla_{\mathbf{W}_A} \mathcal{L} = (\nabla_{\mathbf{A}} \mathcal{L}) \mathbf{h}^\top = \begin{bmatrix} -1/3 \\ +2/3 \\ -1/3 \end{bmatrix} \begin{bmatrix} 2.0 & 0.0 \end{bmatrix} = \begin{bmatrix} -2/3 & 0 \\ +4/3 & 0 \\ -2/3 & 0 \end{bmatrix} \approx \begin{bmatrix} \mathbf{-0.6667} & \mathbf{0.0000} \\ \mathbf{+1.3333} & \mathbf{0.0000} \\ \mathbf{-0.6667} & \mathbf{0.0000} \end{bmatrix}$$
  $$\nabla_{\mathbf{b}_A} \mathcal{L} = \nabla_{\mathbf{A}} \mathcal{L} = \begin{bmatrix} \mathbf{-0.3333} \\ \mathbf{+0.6667} \\ \mathbf{-0.3333} \end{bmatrix}$$
Even though only action $a_1$ was updated, the advantage weights for $a_0$ and $a_2$ receive balancing negative gradients ($\Delta b = -1/3$), strictly maintaining the zero-mean identifiability constraint! $\blacksquare$

---

### Illustration 4: Sum-Tree Priority Dynamic Update and Log-Time Query

**Problem:**
A PER Sum-Tree stores $N = 4$ transitions in a flat array of size $2N - 1 = 7$:
$$\text{tree} = [10.0, 4.0, 6.0, 1.0, 3.0, 2.0, 4.0]$$
where indices $3, 4, 5, 6$ store the leaf priorities $p_0 = 1.0, p_1 = 3.0, p_2 = 2.0, p_3 = 4.0$.
1. Transition $1$ (stored at leaf index 4) is retrained, and its priority increases from $p_1 = 3.0$ to $p_1 = 7.0$. Update the Sum-Tree array.
2. Given target random value $v = 6.5000 \in [0, 14.0]$, trace the retrieval algorithm to determine which transition is sampled.

**Solution:**

*Step 1: Priority Update Propagation.*
The priority change is:
$$\Delta = p_{\text{new}} - p_{\text{old}} = 7.0000 - 3.0000 = \mathbf{+4.0000}$$
Propagate $\Delta$ upward from leaf index $4$ to the root:
1. **Leaf Node (index 4):**
   $$\text{tree}[4] \leftarrow \text{tree}[4] + \Delta = 3.0 + 4.0 = \mathbf{7.0000}$$
2. **Parent of index 4:** Parent index is $P = \lfloor(4 - 1) / 2\rfloor = 1$.
   $$\text{tree}[1] \leftarrow \text{tree}[1] + \Delta = 4.0 + 4.0 = \mathbf{8.0000}$$
3. **Parent of index 1 (Root):** Parent index is $P = \lfloor(1 - 1) / 2\rfloor = 0$.
   $$\text{tree}[0] \leftarrow \text{tree}[0] + \Delta = 10.0 + 4.0 = \mathbf{14.0000}$$

The updated Sum-Tree flat array is:
$$\text{tree}_{\text{updated}} = [14.0000, 8.0000, 6.0000, 1.0000, 7.0000, 2.0000, 4.0000]$$
Verification of sums:
- Root $\text{tree}[0] = 8.0 + 6.0 = 14.0 \checkmark$
- Left $\text{tree}[1] = 1.0 + 7.0 = 8.0 \checkmark$
- Right $\text{tree}[2] = 2.0 + 4.0 = 6.0 \checkmark$

*Step 2: Log-Time Sampling Query for $v = 6.5000$.*
1. **Start at Root (index 0):**
   - Left child is index $1$ with stored sum $\text{tree}[1] = 8.0$.
   - Check condition: Is $v = 6.5 \le \text{tree}[1] = 8.0$? **Yes!**
   - Branch left to index $1$. The value $v$ remains $6.5000$.
2. **At Node (index 1):**
   - Left child is index $3$ with stored sum $\text{tree}[3] = 1.0$.
   - Check condition: Is $v = 6.5 \le \text{tree}[3] = 1.0$? **No.**
   - Branch right to index $4$.
   - Adjust target value: $v \leftarrow v - \text{tree}[3] = 6.5 - 1.0 = \mathbf{5.5000}$.
3. **At Node (index 4):**
   - Index $4$ is a leaf node ($4 \ge N - 1 = 3$).
   - The sampled transition index is $\text{transition\_idx} = 4 - (N - 1) = 4 - 3 = \mathbf{1}$.
Sampling completed in exactly $\log_2(4) = 2$ comparisons! $\blacksquare$

---

### Illustration 5: Factorized Gaussian Noise Layer Forward Pass in Noisy Nets

**Problem:**
Consider a Noisy Net layer with input dimension $p = 2$ and output dimension $q = 2$:
$$\mathbf{y} = \left( \mu^W + \sigma^W \odot \mathbf{E}^W \right) \mathbf{x} + \left( \mu^b + \sigma^b \odot \mathbf{E}^b \right)$$
where $\mathbf{E}^W = f(\varepsilon_q) f(\varepsilon_p)^\top$ and $\mathbf{E}^b = f(\varepsilon_q)$ with $f(z) = \operatorname{sgn}(z) \sqrt{|z|}$.
Given parameters:
$$\mu^W = \begin{bmatrix} 0.40 & 0.20 \\ -0.10 & 0.50 \end{bmatrix}, \quad \sigma^W = \begin{bmatrix} 0.10 & 0.10 \\ 0.10 & 0.10 \end{bmatrix}, \quad \mu^b = \begin{bmatrix} 0.00 \\ 0.10 \end{bmatrix}, \quad \sigma^b = \begin{bmatrix} 0.05 \\ 0.05 \end{bmatrix}$$
Input vector $\mathbf{x} = [1.00, 2.00]^\top$.
The pseudo-random generator draws standard normal samples:
$$\varepsilon_p = \begin{bmatrix} 1.00 \\ -4.00 \end{bmatrix}, \quad \varepsilon_q = \begin{bmatrix} 0.25 \\ -1.00 \end{bmatrix}$$
1. Compute the non-linear transforms $f(\varepsilon_p)$ and $f(\varepsilon_q)$.
2. Construct the factorized noise matrices $\mathbf{E}^W$ and $\mathbf{E}^b$.
3. Compute the effective noisy parameters $\mathbf{W}, \mathbf{b}$ and the layer output $\mathbf{y}$.

**Solution:**

*Step 1: Compute $f(z) = \operatorname{sgn}(z) \sqrt{|z|}$.*
- For $\varepsilon_p$:
  $$f(\varepsilon_{p, 0}) = \operatorname{sgn}(1.00) \sqrt{|1.00|} = +1.0 \times 1.0 = \mathbf{+1.0000}$$
  $$f(\varepsilon_{p, 1}) = \operatorname{sgn}(-4.00) \sqrt{|-4.00|} = -1.0 \times 2.0 = \mathbf{-2.0000}$$
  $$\mathbf{e}_p = f(\varepsilon_p) = \begin{bmatrix} 1.0000 \\ -2.0000 \end{bmatrix}$$
- For $\varepsilon_q$:
  $$f(\varepsilon_{q, 0}) = \operatorname{sgn}(0.25) \sqrt{|0.25|} = +1.0 \times 0.5 = \mathbf{+0.5000}$$
  $$f(\varepsilon_{q, 1}) = \operatorname{sgn}(-1.00) \sqrt{|-1.00|} = -1.0 \times 1.0 = \mathbf{-1.0000}$$
  $$\mathbf{e}_q = f(\varepsilon_q) = \begin{bmatrix} 0.5000 \\ -1.0000 \end{bmatrix}$$

*Step 2: Construct factorized noise tensors.*
$$\mathbf{E}^W = \mathbf{e}_q \mathbf{e}_p^\top = \begin{bmatrix} 0.50 \\ -1.00 \end{bmatrix} \begin{bmatrix} 1.00 & -2.00 \end{bmatrix} = \begin{bmatrix} 0.50(1.0) & 0.50(-2.0) \\ -1.0(1.0) & -1.0(-2.0) \end{bmatrix} = \begin{bmatrix} \mathbf{0.5000} & \mathbf{-1.0000} \\ \mathbf{-1.0000} & \mathbf{2.0000} \end{bmatrix}$$
$$\mathbf{E}^b = \mathbf{e}_q = \begin{bmatrix} \mathbf{0.5000} \\ \mathbf{-1.0000} \end{bmatrix}$$

*Step 3: Compute effective weights and output $\mathbf{y}$.*
$$\mathbf{W} = \mu^W + \sigma^W \odot \mathbf{E}^W = \begin{bmatrix} 0.40 & 0.20 \\ -0.10 & 0.50 \end{bmatrix} + 0.10 \begin{bmatrix} 0.50 & -1.00 \\ -1.00 & 2.00 \end{bmatrix} = \begin{bmatrix} 0.40 + 0.05 & 0.20 - 0.10 \\ -0.10 - 0.10 & 0.50 + 0.20 \end{bmatrix} = \begin{bmatrix} \mathbf{0.4500} & \mathbf{0.1000} \\ \mathbf{-0.2000} & \mathbf{0.7000} \end{bmatrix}$$
$$\mathbf{b} = \mu^b + \sigma^b \odot \mathbf{E}^b = \begin{bmatrix} 0.00 \\ 0.10 \end{bmatrix} + 0.05 \begin{bmatrix} 0.50 \\ -1.00 \end{bmatrix} = \begin{bmatrix} 0.00 + 0.025 \\ 0.10 - 0.050 \end{bmatrix} = \begin{bmatrix} \mathbf{0.0250} \\ \mathbf{0.0500} \end{bmatrix}$$

Now evaluate forward output on $\mathbf{x} = [1.0, 2.0]^\top$:
$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b} = \begin{bmatrix} 0.4500 & 0.1000 \\ -0.2000 & 0.7000 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} + \begin{bmatrix} 0.0250 \\ 0.0500 \end{bmatrix}$$
$$\mathbf{y} = \begin{bmatrix} 0.4500(1.0) + 0.1000(2.0) + 0.0250 \\ -0.2000(1.0) + 0.7000(2.0) + 0.0500 \end{bmatrix} = \begin{bmatrix} 0.4500 + 0.2000 + 0.0250 \\ -0.2000 + 1.4000 + 0.0500 \end{bmatrix} = \begin{bmatrix} \mathbf{0.6750} \\ \mathbf{1.2500} \end{bmatrix} \quad \blacksquare$$

---

## 7. Deep Learning Connection & Modern Applications

### 1. Distributed Rainbow: Apex-DQN & Agent57 (DeepMind, 2018–2020)
Rainbow's six components enabled a new generation of massively distributed architectures:
- **Apex-DQN (Horgan et al., ICLR 2018):** Decouples actor rollouts from the learner. A single GPU learner reads prioritized transitions from 360+ CPU actors running independent copies of the environment. Throughput exceeds 50,000 environment steps/second.
- **R2D2 (Recurrent Replay Distributed DQN, Kapturowski et al., ICLR 2019):** Adds an LSTM over sequences of transitions in the replay buffer, enabling context-aware Q-estimation. Holds Atari world-record scores on 52 of 57 games.
- **Agent57 (Badia et al., Nature 2020):** First agent to surpass human-level performance on **all 57 Atari games** simultaneously, combining QR-DQN distributional returns, UCB-style meta-controller for per-game exploration, and transformed Bellman operators.

### 2. MuZero & DreamerV3: Multi-Step Returns in Model-Based RL (2020–2023)
Multi-step returns (Rainbow component 5) became the backbone of model-based planning:
- **MuZero (Schrittwieser et al., Nature 2020):** Uses $n = 5$ step returns during the learning phase of MCTS backup. By backing up $n$-step target values through the learned latent model, MuZero achieves superhuman performance in Chess, Go, Shogi, and 57 Atari games.
- **DreamerV3 (Hafner et al., Nature 2023):** Plans actor-critic updates via multi-step returns computed entirely within imagined latent trajectories of length $H = 15$ steps, without any real environment interaction.

### 3. Dueling Networks in Production: Robotics, Game AI & Chip Design
- **Discrete Robot Navigation:** Dueling architectures separate *"Is this room reachable?"* (value) from *"Which door should I open right now?"* (advantage), enabling robots to navigate mazes with thousands of rooms without confusion between global accessibility and local action selection.
- **Google AlphaChip (Mirhoseini et al., Nature 2021):** Uses dueling Q-networks to place computer chip components (standard cells) on silicon, producing chip layouts that outperform expert human engineers in power, performance, and area.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of Part 5 Dueling network forward pass ($Q = [3.0, 7.0, 5.0]$ matching PyTorch to $< 10^{-14}$).
2. Sum-Tree data structure: implementing $O(\log N)$ priority updates and sampling, verifying the query for $v = 4.5 \implies \text{index } 2$.
3. Double DQN vs. DQN benchmark on an environment with high reward noise, verifying elimination of maximization bias.

See implementation in:
[`11_reinforcement_learning/code/11_rainbow_dqn_suite.py`](./code/11_rainbow_dqn_suite.py)
