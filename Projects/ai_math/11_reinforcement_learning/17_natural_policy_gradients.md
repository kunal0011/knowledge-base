# Module 11.17: Natural Policy Gradients & Information Geometry

---

## 1. Intuition & 101 Motivation

In standard policy gradient methods (REINFORCE, A2C), parameters are updated along the direction of steepest ascent in flat Euclidean parameter space:
$$\theta_{t+1} = \theta_t + \alpha \nabla_{\theta} J(\theta)$$

This update is the solution to a constrained optimization problem bounded by Euclidean distance:
$$\max_{\Delta \theta} \nabla_{\theta} J^\top \Delta \theta \quad \text{subject to} \quad \|\Delta \theta\|_2^2 \le \epsilon$$

However, measuring step size in Euclidean parameter space $\|\Delta \theta\|_2$ is fundamentally flawed:
1. **Parameter Coordinates are Arbitrary:** Changing the units of a neural network's weights or re-parameterizing a layer (e.g., using log-variance instead of standard deviation) changes the Euclidean norm without altering the underlying policy at all!
2. **The "Cliff of Catastrophic Forgetting":** In probability distribution space, a tiny parameter change $\|\Delta \theta\|_2 = 0.01$ might alter action probabilities by a minuscule $0.1\%$ in flat regions, but could alter action probabilities by $50\%$ in saturated regions, causing policy collapse.

**Natural Policy Gradient (Kakade, 2002)** replaces the arbitrary Euclidean metric with **Information Geometry** (Amari, 1998). Instead of constraining parameter distance $\|\Delta \theta\|_2^2$, it constrains the **Kullback-Leibler (KL) divergence** between the old and new policies:
$$D_{\text{KL}}(\pi_{\theta} \parallel \pi_{\theta + \Delta \theta}) \le \epsilon$$

The natural gradient moves along the **Riemannian manifold of probability distributions**, where the metric tensor is the **Fisher Information Matrix (FIM)**.

```
       FLAT EUCLIDEAN PARAMETER SPACE             CURVED PROBABILITY MANIFOLD (FIM)
               Delta theta                                KL(pi_old || pi_new)
                   ^                                                ^
                   |                                                |
            *------+------*                                  *~~~~~~+~~~~~~*
            |             |                                 (               )
            |      *      |                                 (       *       )
            |             |                                 (               )
            *------+------*                                  *~~~~~~+~~~~~~*
                   |                                                |
          Arbitrary coordinate box                         True distance in behavior!
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Second-Order Taylor Expansion of KL Divergence

Let $\pi_{\theta}(a \mid s)$ be a parameterized policy.
The expected KL divergence between $\pi_{\theta}$ and $\pi_{\theta + \mathbf{d}\theta}$ across state visitation distribution $d^\pi(s)$ is:
$$\bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta) \triangleq \mathbb{E}_{s \sim d^\pi} \left[ D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta + \mathbf{d}\theta}(\cdot \mid s)) \right]$$

Perform a Taylor expansion of $\bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta)$ about $\mathbf{d}\theta = \mathbf{0}$:
1. **Zeroth-order term:**
   $$\bar{D}_{\text{KL}}(\theta \parallel \theta) = 0$$
2. **First-order term:**
   Because $\bar{D}_{\text{KL}} \ge 0$ for all distributions, $\mathbf{d}\theta = \mathbf{0}$ is a global minimum. Thus, the first derivative vanishes:
   $$\left. \nabla_{\mathbf{d}\theta} \bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta) \right|_{\mathbf{d}\theta = \mathbf{0}} = \mathbf{0}$$
3. **Second-order term (Hessian):**
   The Hessian of the KL divergence at $\mathbf{d}\theta = \mathbf{0}$ is the **Fisher Information Matrix** $\mathbf{F}(\theta)$:

$$\bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta) = \frac{1}{2} \mathbf{d}\theta^\top \mathbf{F}(\theta) \mathbf{d}\theta + \mathcal{O}(\|\mathbf{d}\theta\|_2^3)$$

---

### 2.2 The Fisher Information Matrix (FIM)

#### Definition:
$$\mathbf{F}(\theta) \triangleq \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s)^\top \right]$$

Equivalently, by the information equality (under mild regularity conditions):
$$\mathbf{F}(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}} \left[ - \nabla_{\theta}^2 \log \pi_{\theta}(a \mid s) \right]$$

#### Mathematical Properties of $\mathbf{F}$:
1. $\mathbf{F}(\theta) \in \mathbb{R}^{d \times d}$ is **symmetric** and **positive semi-definite** ($\mathbf{x}^\top \mathbf{F} \mathbf{x} \ge 0$).
2. It acts as the **Riemannian Metric Tensor** on the statistical manifold of policies.
3. It defines the local distance metric in distribution space:
   $$ds^2 = \mathbf{d}\theta^\top \mathbf{F}(\theta) \mathbf{d}\theta$$

---

### 2.3 The Constrained Optimization Problem

The Natural Policy Gradient seeks the step $\Delta \theta$ that maximizes expected return subject to a strict bound on policy divergence:

$$\max_{\Delta \theta} \nabla_{\theta} J(\theta)^\top \Delta \theta \quad \text{subject to} \quad \frac{1}{2} \Delta \theta^\top \mathbf{F}(\theta) \Delta \theta \le \epsilon$$
where $\mathbf{g} \triangleq \nabla_{\theta} J(\theta)$ is the standard vanilla policy gradient.

#### Derivation via Lagrangian Duality:
Form the Lagrangian with dual multiplier $\lambda \ge 0$:
$$\mathcal{L}(\Delta \theta, \lambda) = \mathbf{g}^\top \Delta \theta - \lambda \left( \frac{1}{2} \Delta \theta^\top \mathbf{F} \Delta \theta - \epsilon \right)$$

Set the derivative with respect to $\Delta \theta$ to zero:
$$\nabla_{\Delta \theta} \mathcal{L} = \mathbf{g} - \lambda \mathbf{F} \Delta \theta = \mathbf{0}$$
$$\mathbf{F} \Delta \theta = \frac{1}{\lambda} \mathbf{g} \implies \Delta \theta^* = \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g}$$

Substitute $\Delta \theta^*$ into the boundary constraint $\frac{1}{2} \Delta \theta^\top \mathbf{F} \Delta \theta = \epsilon$:
$$\frac{1}{2} \left( \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g} \right)^\top \mathbf{F} \left( \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g} \right) = \epsilon$$
$$\frac{1}{2 \lambda^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{F} \mathbf{F}^{-1} \mathbf{g} = \epsilon \implies \frac{1}{2 \lambda^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} = \epsilon$$
$$\lambda^* = \sqrt{\frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \epsilon}}$$

Substitute $\lambda^*$ back into the optimal step:
$$\Delta \theta^* = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}$$

#### The Natural Gradient Direction:
$$\tilde{\mathbf{g}} \triangleq \mathbf{F}(\theta)^{-1} \nabla_{\theta} J(\theta)$$
The Natural Policy Gradient update step scales $\tilde{\mathbf{g}}$ to exactly satisfy the trust region $\epsilon$:
$$\theta_{t+1} = \theta_t + \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}$$

---

### 2.4 Theorem: Invariance to Re-Parameterization (Kakade, 2002)

Let $\psi = f(\theta)$ be an arbitrary smooth, invertible change of coordinates.
Then the natural policy gradient update in $\psi$-space produces the **exact same change in probability distributions** as the natural policy gradient update in $\theta$-space:
$$\pi_{\theta + \Delta \theta}(a \mid s) = \pi_{\psi + \Delta \psi}(a \mid s) + \mathcal{O}(\epsilon^2)$$
The natural gradient is a **geometric invariant** of the manifold, completely independent of how the neural network weights are parameterized!

---

### 2.5 First-Principles Mathematical Derivations

```
====================================================================================================
DERIVATION 11.17.1: Fisher Information Matrix as the Hessian of KL Divergence
====================================================================================================
Problem Statement:
Let π_θ(a | s) be a parameterized policy family and d^π(s) be the stationary state distribution.
Define the expected KL divergence:
    D_bar_KL(θ || θ') ≜ 𝔼_{s ~ d^π}[ D_KL(π_θ(· | s) || π_θ'(· | s)) ]
Prove from first principles that:
    1. ∇_{θ'} D_bar_KL(θ || θ') |_{θ' = θ} = 0
    2. ∇_{θ'}^2 D_bar_KL(θ || θ') |_{θ' = θ} = 𝔼_{s, a}[ ∇_θ log π_θ(a | s) ∇_θ log π_θ(a | s)^T ] ≜ F(θ)
    3. F(θ) = 𝔼_{s, a}[ -∇_θ^2 log π_θ(a | s) ] (Information Equality)
    4. D_bar_KL(θ || θ + dθ) = (1/2) dθ^T F(θ) dθ + O(||dθ||_2^3)
====================================================================================================
```

#### Derivation 11.17.1: Fisher Information Matrix as the Hessian of KL Divergence

##### Part 1: Problem Statement & Mathematical Goal
Let $\mathcal{S}$ be the state space, $\mathcal{A}$ be the action space, and let $\pi_{\theta}(a \mid s)$ be a policy parameterized by $\theta \in \mathbb{R}^d$.
For a fixed state $s \in \mathcal{S}$ and candidate parameter vectors $\theta, \theta' \in \mathbb{R}^d$, the Kullback-Leibler (KL) divergence from $\pi_{\theta}(\cdot \mid s)$ to $\pi_{\theta'}(\cdot \mid s)$ is defined as:
$$D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \triangleq \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \left( \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta'}(a \mid s)} \right)$$
For continuous action spaces, the summation is replaced by an integral $\int_{\mathcal{A}} \pi_{\theta}(a \mid s) \log \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta'}(a \mid s)} \, da$.
Let $d^\pi(s)$ denote the stationary state visitation distribution under policy $\pi_{\theta}$. The expected (state-averaged) KL divergence is:
$$\bar{D}_{\text{KL}}(\theta \parallel \theta') \triangleq \mathbb{E}_{s \sim d^\pi}\left[ D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \right]$$
The Fisher Information Matrix (FIM) $\mathbf{F}(\theta) \in \mathbb{R}^{d \times d}$ is defined as the expected outer product of score functions:
$$\mathbf{F}(\theta) \triangleq \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s)^\top \right]$$

**Mathematical Goals:**
1. Prove that the zeroth-order divergence vanishes: $\bar{D}_{\text{KL}}(\theta \parallel \theta) = 0$.
2. Prove that the first-order gradient of $\bar{D}_{\text{KL}}(\theta \parallel \theta')$ with respect to $\theta'$, evaluated at $\theta' = \theta$, is identically zero:
   $$\left. \nabla_{\theta'} \bar{D}_{\text{KL}}(\theta \parallel \theta') \right|_{\theta' = \theta} = \mathbf{0}$$
3. Prove that the second-order derivative (Hessian matrix) with respect to $\theta'$, evaluated at $\theta' = \theta$, is exactly the Fisher Information Matrix:
   $$\left. \nabla_{\theta'}^2 \bar{D}_{\text{KL}}(\theta \parallel \theta') \right|_{\theta' = \theta} = \mathbf{F}(\theta)$$
4. Prove the Information Equality connecting the outer product of score functions to the negative expected Hessian of the log-likelihood:
   $$\mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s)^\top \right] = \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}} \left[ -\nabla_{\theta}^2 \log \pi_{\theta}(a \mid s) \right]$$
5. Prove that the second-order Taylor series expansion of $\bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta)$ about $\mathbf{d}\theta = \mathbf{0}$ is:
   $$\bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta) = \frac{1}{2} \mathbf{d}\theta^\top \mathbf{F}(\theta) \mathbf{d}\theta + \mathcal{O}(\|\mathbf{d}\theta\|_2^3)$$
6. Prove that the reverse KL divergence $\bar{D}_{\text{KL}}(\theta + \mathbf{d}\theta \parallel \theta)$ possesses the exact same Hessian and second-order Taylor expansion at $\mathbf{d}\theta = \mathbf{0}$.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Common Support:** For all parameter values $\theta, \theta' \in \Theta$, the support of the policy $\operatorname{supp}(\pi_{\theta}(\cdot \mid s)) = \{a \in \mathcal{A} : \pi_{\theta}(a \mid s) > 0\}$ is identical and strictly independent of $\theta$. This ensures $\log \pi_{\theta'}(a \mid s)$ is finite and continuously defined almost everywhere.
2. **Smoothness and Differentiability:** For every $s \in \mathcal{S}$ and $a \in \mathcal{A}$, the policy density/mass $\pi_{\theta}(a \mid s)$ is twice continuously differentiable ($C^2$) with respect to $\theta \in \mathbb{R}^d$.
3. **Leibniz Integral Rule / Dominated Convergence:** The order of differentiation with respect to $\theta$ and integration (or summation) over the action space $\mathcal{A}$ can be legitimately interchanged:
   $$\nabla_{\theta} \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) = \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a \mid s), \quad \nabla_{\theta}^2 \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) = \sum_{a \in \mathcal{A}} \nabla_{\theta}^2 \pi_{\theta}(a \mid s)$$
   This holds under dominated convergence when derivatives are uniformly bounded by an integrable envelope.
4. **Finite Moments:** The components of the score vector $\nabla_{\theta} \log \pi_{\theta}(a \mid s)$ have finite second moments under $\pi_{\theta}$, ensuring $|F_{ij}(\theta)| < \infty$ for all coordinates $i, j \in \{1, \dots, d\}$.
5. **Stationary State Measure:** The stationary state visitation distribution $d^\pi(s)$ exists and satisfies $\sum_{s \in \mathcal{S}} d^\pi(s) = 1$ (or $\int_{\mathcal{S}} d^\pi(s) \, ds = 1$).

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
The Kullback-Leibler divergence measures information loss between probability distributions. While KL divergence is not a true metric (it is asymmetric, $D_{\text{KL}}(P \parallel Q) \ne D_{\text{KL}}(Q \parallel P)$, and fails the triangle inequality), it is non-negative and achieves a unique global minimum of $0$ when $P = Q$.
Because $\theta' = \theta$ is a global minimum of the smooth function $f(\theta') = D_{\text{KL}}(\pi_{\theta} \parallel \pi_{\theta'})$, the tangent plane at $\theta' = \theta$ must be completely flat: the first derivative vanishes identically ($\nabla f = \mathbf{0}$).
Consequently, the local geometry of probability space around $\theta$ is governed entirely by the quadratic curvature (the Hessian $\nabla^2 f$). This Hessian is precisely the Fisher Information Matrix $\mathbf{F}(\theta)$.
Thus, for infinitesimal steps $\mathbf{d}\theta$, the asymmetry of the KL divergence vanishes, and probability space behaves as a Riemannian manifold equipped with metric tensor $\mathbf{g}_{ij}(\theta) = F_{ij}(\theta)$, where squared infinitesimal distance is:
$$ds^2 = 2 D_{\text{KL}}(\pi_{\theta} \parallel \pi_{\theta + \mathbf{d}\theta}) \approx \mathbf{d}\theta^\top \mathbf{F}(\theta) \mathbf{d}\theta$$

##### Part 4: End-to-End Step-by-Step Algebraic Proof
We prove the result for an arbitrary fixed state $s \in \mathcal{S}$. (The state-averaged result follows immediately by taking the expectation over $s \sim d^\pi(s)$ using linearity of expectation and dominated convergence).

**Step 1: Decompose the KL Divergence into Entropy and Cross-Entropy.**
By definition:
$$D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \left( \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta'}(a \mid s)} \right)$$
Using the logarithm identity $\log(u / v) = \log u - \log v$:
$$D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \pi_{\theta}(a \mid s) - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \pi_{\theta'}(a \mid s)$$
Notice that the first term is the negative Shannon entropy $-H(\pi_{\theta}(\cdot \mid s))$, which is constant with respect to $\theta'$.

**Step 2: Evaluate the Zeroth-Order Term.**
Setting $\theta' = \theta$:
$$D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta}(\cdot \mid s)) = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \left( \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta}(a \mid s)} \right) = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log(1) = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \cdot 0 = 0$$

**Step 3: Differentiate with respect to $\theta'$ (First-Order Gradient).**
Taking the partial derivative of $D_{\text{KL}}$ with respect to parameter component $\theta'_i$ for $i \in \{1, \dots, d\}$:
$$\frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = \frac{\partial}{\partial \theta'_i} \left[ \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \pi_{\theta}(a \mid s) - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \pi_{\theta'}(a \mid s) \right]$$
The first sum does not depend on $\theta'$, so its derivative is zero:
$$\frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = - \frac{\partial}{\partial \theta'_i} \left[ \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \log \pi_{\theta'}(a \mid s) \right]$$
Applying the Leibniz differentiation rule (Assumption 3):
$$\frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{\partial}{\partial \theta'_i} \log \pi_{\theta'}(a \mid s)$$
Using the derivative of the logarithm $\frac{\partial}{\partial x} \log u(x) = \frac{1}{u(x)} \frac{\partial u(x)}{\partial x}$:
$$\frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial \pi_{\theta'}(a \mid s)}{\partial \theta'_i}$$

**Step 4: Evaluate the Gradient at $\theta' = \theta$.**
Evaluating this derivative at $\theta' = \theta$:
$$\left. \frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \right|_{\theta' = \theta} = - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{1}{\pi_{\theta}(a \mid s)} \left. \frac{\partial \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \right|_{\theta' = \theta}$$
Because $\pi_{\theta}(a \mid s) > 0$ on its support, the terms cancel:
$$\left. \frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \right|_{\theta' = \theta} = - \sum_{a \in \mathcal{A}} \frac{\partial \pi_{\theta}(a \mid s)}{\partial \theta_i}$$
Interchanging summation and differentiation (Assumption 3):
$$\left. \frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \right|_{\theta' = \theta} = - \frac{\partial}{\partial \theta_i} \left[ \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \right]$$
Since probabilities must sum to 1 for any parameter setting, $\sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) = 1$:
$$\left. \frac{\partial}{\partial \theta'_i} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \right|_{\theta' = \theta} = - \frac{\partial}{\partial \theta_i} [1] = 0$$
In vector notation across all coordinates $i \in \{1, \dots, d\}$:
$$\left. \nabla_{\theta'} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \right|_{\theta' = \theta} = \mathbf{0}$$

**Step 5: Compute the Second-Order Partial Derivatives (Hessian).**
Differentiating the first derivative with respect to $\theta'_j$ for $j \in \{1, \dots, d\}$:
$$\frac{\partial^2}{\partial \theta'_i \partial \theta'_j} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = \frac{\partial}{\partial \theta'_j} \left[ - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{\partial \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \right]$$
Interchanging differentiation and summation:
$$\frac{\partial^2}{\partial \theta'_i \partial \theta'_j} D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) = - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{\partial^2 \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j}$$
Now evaluate the second derivative of the log-likelihood:
$$\frac{\partial^2 \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} = \frac{\partial}{\partial \theta'_j} \left( \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \right)$$
Applying the quotient rule $\frac{\partial}{\partial x} \left(\frac{u}{v}\right) = \frac{u' v - u v'}{v^2}$:
$$\frac{\partial^2 \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} = \frac{\frac{\partial^2 \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} \pi_{\theta'}(a \mid s) - \frac{\partial \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \frac{\partial \pi_{\theta'}(a \mid s)}{\partial \theta'_j}}{\left( \pi_{\theta'}(a \mid s) \right)^2}$$
Splitting into two terms:
$$\frac{\partial^2 \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} = \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial^2 \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} - \left( \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \right) \left( \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial \pi_{\theta'}(a \mid s)}{\partial \theta'_j} \right)$$
Using $\frac{\partial \log \pi_{\theta'}}{\partial \theta'_k} = \frac{1}{\pi_{\theta'}} \frac{\partial \pi_{\theta'}}{\partial \theta'_k}$:
$$\frac{\partial^2 \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} = \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial^2 \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} - \frac{\partial \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \frac{\partial \log \pi_{\theta'}(a \mid s)}{\partial \theta'_j}$$

**Step 6: Substitute into Hessian Expression.**
Negating this expression yields:
$$- \frac{\partial^2 \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} = \frac{\partial \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \frac{\partial \log \pi_{\theta'}(a \mid s)}{\partial \theta'_j} - \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial^2 \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j}$$
Substitute this directly into the Hessian formula:
$$\frac{\partial^2 D_{\text{KL}}}{\partial \theta'_i \partial \theta'_j} = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \left[ \frac{\partial \log \pi_{\theta'}(a \mid s)}{\partial \theta'_i} \frac{\partial \log \pi_{\theta'}(a \mid s)}{\partial \theta'_j} - \frac{1}{\pi_{\theta'}(a \mid s)} \frac{\partial^2 \pi_{\theta'}(a \mid s)}{\partial \theta'_i \partial \theta'_j} \right]$$

**Step 7: Evaluate the Hessian at $\theta' = \theta$.**
Now set $\theta' = \theta$:
$$\left. \frac{\partial^2 D_{\text{KL}}}{\partial \theta'_i \partial \theta'_j} \right|_{\theta' = \theta} = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_i} \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_j} - \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{1}{\pi_{\theta}(a \mid s)} \frac{\partial^2 \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j}$$
In the second summation, the probabilities cancel:
$$\sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{1}{\pi_{\theta}(a \mid s)} \frac{\partial^2 \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j} = \sum_{a \in \mathcal{A}} \frac{\partial^2 \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j}$$
Interchanging summation and derivatives (Assumption 3):
$$\sum_{a \in \mathcal{A}} \frac{\partial^2 \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j} = \frac{\partial^2}{\partial \theta_i \partial \theta_j} \left[ \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \right] = \frac{\partial^2}{\partial \theta_i \partial \theta_j} [1] = 0$$
Thus, the second term vanishes identically!
We obtain:
$$\left. \frac{\partial^2 D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s))}{\partial \theta'_i \partial \theta'_j} \right|_{\theta' = \theta} = \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_i} \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_j}$$
In full matrix form for state $s$:
$$\left. \nabla_{\theta'}^2 D_{\text{KL}}(\pi_{\theta}(\cdot \mid s) \parallel \pi_{\theta'}(\cdot \mid s)) \right|_{\theta' = \theta} = \mathbb{E}_{a \sim \pi_{\theta}(\cdot \mid s)} \left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s)^\top \right] \triangleq \mathbf{F}_s(\theta)$$

**Step 8: State-Averaged Fisher Information Matrix.**
Averaging over the stationary state visitation distribution $s \sim d^\pi(s)$:
$$\left. \nabla_{\theta'}^2 \bar{D}_{\text{KL}}(\theta \parallel \theta') \right|_{\theta' = \theta} = \mathbb{E}_{s \sim d^\pi} [\mathbf{F}_s(\theta)] = \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \nabla_{\theta} \log \pi_{\theta}(a \mid s)^\top \right] = \mathbf{F}(\theta)$$

**Step 9: Proof of the Information Equality.**
From Step 5, we found:
$$\frac{\partial^2 \log \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j} = \frac{1}{\pi_{\theta}(a \mid s)} \frac{\partial^2 \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j} - \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_i} \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_j}$$
Taking the expectation $\mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}}$ of both sides:
$$\mathbb{E}_{s, a} \left[ \frac{\partial^2 \log \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j} \right] = \mathbb{E}_{s \sim d^\pi} \left[ \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \frac{1}{\pi_{\theta}(a \mid s)} \frac{\partial^2 \pi_{\theta}(a \mid s)}{\partial \theta_i \partial \theta_j} \right] - \mathbb{E}_{s, a} \left[ \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_i} \frac{\partial \log \pi_{\theta}(a \mid s)}{\partial \theta_j} \right]$$
$$= \mathbb{E}_{s \sim d^\pi} \left[ \frac{\partial^2}{\partial \theta_i \partial \theta_j} \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) \right] - F_{ij}(\theta)$$
$$= \mathbb{E}_{s \sim d^\pi} \left[ \frac{\partial^2}{\partial \theta_i \partial \theta_j} (1) \right] - F_{ij}(\theta) = 0 - F_{ij}(\theta) = -F_{ij}(\theta)$$
Negating both sides:
$$\mathbf{F}(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\theta}} \left[ - \nabla_{\theta}^2 \log \pi_{\theta}(a \mid s) \right]$$

**Step 10: Taylor Series Expansion for Infinitesimal Perturbations.**
Let $\mathbf{d}\theta = \theta' - \theta$ be an infinitesimal displacement vector.
Define the multivariable function $g(\mathbf{d}\theta) \triangleq \bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta)$.
Computing the second-order Taylor expansion of $g(\mathbf{d}\theta)$ about $\mathbf{d}\theta = \mathbf{0}$:
$$g(\mathbf{d}\theta) = g(\mathbf{0}) + \nabla_{\mathbf{d}\theta} g(\mathbf{0})^\top \mathbf{d}\theta + \frac{1}{2} \mathbf{d}\theta^\top \nabla_{\mathbf{d}\theta}^2 g(\mathbf{0}) \mathbf{d}\theta + \mathcal{O}(\|\mathbf{d}\theta\|_2^3)$$
From Step 2, $g(\mathbf{0}) = 0$.
From Step 4, $\nabla_{\mathbf{d}\theta} g(\mathbf{0}) = \mathbf{0}$.
From Step 8, $\nabla_{\mathbf{d}\theta}^2 g(\mathbf{0}) = \mathbf{F}(\theta)$.
Substituting these values:
$$\bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}\theta) = 0 + \mathbf{0}^\top \mathbf{d}\theta + \frac{1}{2} \mathbf{d}\theta^\top \mathbf{F}(\theta) \mathbf{d}\theta + \mathcal{O}(\|\mathbf{d}\theta\|_2^3)$$
$$= \frac{1}{2} \mathbf{d}\theta^\top \mathbf{F}(\theta) \mathbf{d}\theta + \mathcal{O}(\|\mathbf{d}\theta\|_2^3)$$
By an identical symmetric argument on $h(\mathbf{d}\theta) \triangleq \bar{D}_{\text{KL}}(\theta + \mathbf{d}\theta \parallel \theta)$, we have $h(\mathbf{0}) = 0$, $\nabla h(\mathbf{0}) = \mathbf{0}$, and $\nabla^2 h(\mathbf{0}) = \mathbf{F}(\theta)$, so:
$$\bar{D}_{\text{KL}}(\theta + \mathbf{d}\theta \parallel \theta) = \frac{1}{2} \mathbf{d}\theta^\top \mathbf{F}(\theta) \mathbf{d}\theta + \mathcal{O}(\|\mathbf{d}\theta\|_2^3)$$
proving that both forward and reverse KL divergences possess the identical quadratic Riemannian metric $\mathbf{F}(\theta)$. $\blacksquare$

---

```
====================================================================================================
DERIVATION 11.17.2: Natural Policy Gradient as Steepest Ascent on Riemannian Manifold
====================================================================================================
Problem Statement:
Let J(θ) be the policy performance objective and g ≜ ∇_θ J(θ) be the vanilla policy gradient.
Consider the Riemannian manifold of policies equipped with Fisher metric tensor G(θ) = F(θ).
Solve the trust-region steepest ascent problem:
    d* = argmax_{d ∈ ℝ^d} g^T d  subject to  (1/2) d^T F(θ) d ≤ ϵ
Prove from first principles that:
    1. Optimal step is d* = sqrt(2ϵ / (g^T F(θ)^-1 g)) F(θ)^-1 g
    2. Steepest ascent direction on the manifold is the Natural Gradient g_tilde = F(θ)^-1 g
====================================================================================================
```

#### Derivation 11.17.2: Natural Policy Gradient as Steepest Ascent on Riemannian Manifold

##### Part 1: Problem Statement & Mathematical Goal
Let $J(\theta)$ be the expected discounted policy performance objective:
$$J(\theta) \triangleq \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^\infty \gamma^t R(S_t, A_t) \right]$$
Let $\mathbf{g} \triangleq \nabla_{\theta} J(\theta)$ denote the standard vanilla policy gradient vector in $\mathbb{R}^d$.
Consider the Riemannian manifold $\mathcal{M} = \{\pi_{\theta} : \theta \in \Theta\}$ equipped with the Fisher Information Matrix $\mathbf{F}(\theta) \in \mathbb{R}^{d \times d}$ as its metric tensor $\mathbf{G}(\theta) = \mathbf{F}(\theta)$.
The squared Riemannian distance of an infinitesimal parameter displacement vector $\mathbf{d} \in \mathbb{R}^d$ is:
$$\|\mathbf{d}\|_{\mathbf{F}}^2 \triangleq \mathbf{d}^\top \mathbf{F}(\theta) \mathbf{d}$$
By Derivation 11.17.1, this corresponds to twice the local KL divergence: $\bar{D}_{\text{KL}}(\theta \parallel \theta + \mathbf{d}) \approx \frac{1}{2} \mathbf{d}^\top \mathbf{F}(\theta) \mathbf{d}$.
We formulate the trust-region optimization problem: find the step direction $\mathbf{d}^* \in \mathbb{R}^d$ that maximizes the first-order improvement in return $J(\theta + \mathbf{d}) - J(\theta) \approx \mathbf{g}^\top \mathbf{d}$ subject to a strict bound $\epsilon > 0$ on the Riemannian distance:
$$\max_{\mathbf{d} \in \mathbb{R}^d} \mathbf{g}^\top \mathbf{d} \quad \text{subject to} \quad \frac{1}{2} \mathbf{d}^\top \mathbf{F}(\theta) \mathbf{d} \le \epsilon$$

**Mathematical Goals:**
1. Formulate the primal optimization problem and its Lagrangian function.
2. Apply the Karush-Kuhn-Tucker (KKT) first-order optimality conditions to solve for the analytical displacement vector $\mathbf{d}^*$.
3. Prove that the optimal Lagrange multiplier is:
   $$\lambda^* = \sqrt{\frac{\mathbf{g}^\top \mathbf{F}(\theta)^{-1} \mathbf{g}}{2 \epsilon}}$$
4. Prove that the normalized optimal step vector is:
   $$\mathbf{d}^* = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}(\theta)^{-1} \mathbf{g}}} \mathbf{F}(\theta)^{-1} \mathbf{g}$$
5. Prove that the unnormalized steepest ascent direction on the statistical manifold is the Natural Policy Gradient:
   $$\tilde{\mathbf{g}} \triangleq \mathbf{F}(\theta)^{-1} \mathbf{g} = \mathbf{F}(\theta)^{-1} \nabla_{\theta} J(\theta)$$
6. Provide a geometric proof using the Cauchy-Schwarz inequality on the Riemannian Hilbert space $(\mathbb{R}^d, \langle \cdot, \cdot \rangle_{\mathbf{F}})$.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Positive Definiteness:** The Fisher Information Matrix $\mathbf{F}(\theta)$ is symmetric and strictly positive definite ($\mathbf{F}(\theta) \succ \mathbf{0}$):
   $$\mathbf{d}^\top \mathbf{F}(\theta) \mathbf{d} > 0 \quad \forall \mathbf{d} \in \mathbb{R}^d \setminus \{\mathbf{0}\}$$
   This guarantees that $\mathbf{F}(\theta)$ is non-singular and its inverse $\mathbf{F}(\theta)^{-1}$ exists and is also strictly positive definite. (If rank-deficient, a positive damping term $\delta_{\text{damp}} \mathbf{I}$ is added).
2. **Non-Zero Gradient:** The vanilla policy gradient vector is non-vanishing: $\mathbf{g} = \nabla_{\theta} J(\theta) \ne \mathbf{0}$. (If $\mathbf{g} = \mathbf{0}$, the policy is at a local stationary point and $\mathbf{d}^* = \mathbf{0}$).
3. **Smooth Objective:** The expected return $J(\theta)$ is continuously differentiable ($C^1$) in a neighborhood around $\theta$, ensuring $J(\theta + \mathbf{d}) = J(\theta) + \mathbf{g}^\top \mathbf{d} + \mathcal{O}(\|\mathbf{d}\|_2^2)$.
4. **Positive Trust Region Radius:** $\epsilon > 0$ is a strictly positive, finite scalar.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
In Euclidean optimization, "steepest ascent" is defined by maximizing the directional derivative $\nabla J^\top \mathbf{u}$ over all unit vectors $\|\mathbf{u}\|_2 = 1$. By the Cauchy-Schwarz inequality, this yields $\mathbf{u} = \mathbf{g} / \|\mathbf{g}\|_2$. The standard gradient points orthogonal to the level sets in parameter coordinates.
However, parameter coordinates are an arbitrary convention. On a curved Riemannian manifold $(\mathcal{M}, \mathbf{G})$, the concept of a "unit step" is defined by the Riemannian metric tensor: $\mathbf{d}^\top \mathbf{G} \mathbf{d} = 1$. This defines an ellipsoid in parameter space.
Along axes where the policy distribution changes rapidly (large eigenvalues of $\mathbf{F}$), the ellipsoid is narrow, restricting the allowed coordinate displacement. Along axes where the policy distribution changes slowly (small eigenvalues of $\mathbf{F}$), the ellipsoid is elongated, allowing large coordinate displacements.
Geometrically, the gradient $\mathbf{g} = \nabla_{\theta} J$ is a covector (a linear functional in cotangent space $T_{\theta}^* \mathcal{M}$). To turn a covector into an actual motion vector in the tangent space $T_{\theta} \mathcal{M}$, one must apply the inverse of the metric tensor: $\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g}$ (the musical isomorphism $\sharp$).
This rotates and rescales the gradient vector so that it is steepest with respect to the intrinsic probability geometry, rather than Euclidean artifacts.

##### Part 4: End-to-End Step-by-Step Algebraic Proof

**Step 1: Set up the Primal Optimization Problem.**
We wish to solve:
$$\max_{\mathbf{d} \in \mathbb{R}^d} f_0(\mathbf{d}) \quad \text{subject to} \quad f_1(\mathbf{d}) \le 0$$
where:
$$f_0(\mathbf{d}) \triangleq \mathbf{g}^\top \mathbf{d}$$
$$f_1(\mathbf{d}) \triangleq \frac{1}{2} \mathbf{d}^\top \mathbf{F} \mathbf{d} - \epsilon$$

**Step 2: Form the Lagrangian Function.**
Let $\lambda \ge 0$ be the Lagrange multiplier associated with the inequality constraint. The Lagrangian $\mathcal{L} : \mathbb{R}^d \times \mathbb{R}_{\ge 0} \to \mathbb{R}$ is:
$$\mathcal{L}(\mathbf{d}, \lambda) = f_0(\mathbf{d}) - \lambda f_1(\mathbf{d}) = \mathbf{g}^\top \mathbf{d} - \lambda \left( \frac{1}{2} \mathbf{d}^\top \mathbf{F} \mathbf{d} - \epsilon \right)$$

**Step 3: Stationarity Condition (First-Order Derivative).**
Compute the gradient of $\mathcal{L}$ with respect to $\mathbf{d}$:
$$\nabla_{\mathbf{d}} \mathcal{L}(\mathbf{d}, \lambda) = \nabla_{\mathbf{d}} (\mathbf{g}^\top \mathbf{d}) - \frac{\lambda}{2} \nabla_{\mathbf{d}} (\mathbf{d}^\top \mathbf{F} \mathbf{d})$$
Using matrix calculus identities $\nabla_{\mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$ and $\nabla_{\mathbf{x}} (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$:
$$\nabla_{\mathbf{d}} \mathcal{L}(\mathbf{d}, \lambda) = \mathbf{g} - \frac{\lambda}{2} (\mathbf{F} + \mathbf{F}^\top) \mathbf{d}$$
Because $\mathbf{F}$ is symmetric ($\mathbf{F}^\top = \mathbf{F}$), $\mathbf{F} + \mathbf{F}^\top = 2\mathbf{F}$:
$$\nabla_{\mathbf{d}} \mathcal{L}(\mathbf{d}, \lambda) = \mathbf{g} - \lambda \mathbf{F} \mathbf{d}$$
Setting $\nabla_{\mathbf{d}} \mathcal{L} = \mathbf{0}$ for stationarity:
$$\mathbf{g} - \lambda \mathbf{F} \mathbf{d} = \mathbf{0} \implies \lambda \mathbf{F} \mathbf{d} = \mathbf{g}$$

**Step 4: Analyze Complementary Slackness and Invertibility.**
The KKT optimality conditions require:
1. **Stationarity:** $\lambda \mathbf{F} \mathbf{d} = \mathbf{g}$
2. **Primal Feasibility:** $\frac{1}{2} \mathbf{d}^\top \mathbf{F} \mathbf{d} \le \epsilon$
3. **Dual Feasibility:** $\lambda \ge 0$
4. **Complementary Slackness:** $\lambda \left( \frac{1}{2} \mathbf{d}^\top \mathbf{F} \mathbf{d} - \epsilon \right) = 0$

Assume for contradiction that $\lambda = 0$.
Then the stationarity condition reduces to $0 \cdot \mathbf{F} \mathbf{d} = \mathbf{g} \implies \mathbf{g} = \mathbf{0}$.
However, Assumption 2 states $\mathbf{g} \ne \mathbf{0}$. Contradiction!
Therefore, we must have $\lambda > 0$.
By complementary slackness, since $\lambda \ne 0$, the constraint must be **strictly active** at optimality:
$$\frac{1}{2} \mathbf{d}^\top \mathbf{F} \mathbf{d} = \epsilon$$

**Step 5: Solve for the Step Vector $\mathbf{d}$ in terms of $\lambda$.**
Since $\lambda > 0$ and $\mathbf{F}$ is strictly positive definite (hence invertible):
$$\mathbf{F} \mathbf{d} = \frac{1}{\lambda} \mathbf{g} \implies \mathbf{d}^* = \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g}$$

**Step 6: Solve for the Optimal Dual Multiplier $\lambda^*$.**
Substitute the expression for $\mathbf{d}^*$ into the active constraint $\frac{1}{2} \mathbf{d}^\top \mathbf{F} \mathbf{d} = \epsilon$:
$$\frac{1}{2} \left( \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g} \right)^\top \mathbf{F} \left( \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g} \right) = \epsilon$$
Using the transpose property of a matrix-vector product $(A \mathbf{x})^\top = \mathbf{x}^\top A^\top$:
$$\frac{1}{2 \lambda^2} \mathbf{g}^\top (\mathbf{F}^{-1})^\top \mathbf{F} \mathbf{F}^{-1} \mathbf{g} = \epsilon$$
Because $\mathbf{F}$ is symmetric, its inverse is also symmetric: $(\mathbf{F}^{-1})^\top = \mathbf{F}^{-1}$.
Furthermore, $\mathbf{F}^{-1} \mathbf{F} = \mathbf{I}$:
$$\frac{1}{2 \lambda^2} \mathbf{g}^\top \mathbf{F}^{-1} (\mathbf{F} \mathbf{F}^{-1}) \mathbf{g} = \epsilon$$
$$\frac{1}{2 \lambda^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{I} \mathbf{g} = \epsilon \implies \frac{1}{2 \lambda^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} = \epsilon$$
Multiply both sides by $\lambda^2$ and divide by $\epsilon$:
$$\lambda^2 = \frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \epsilon}$$
Because $\mathbf{F}$ is positive definite, $\mathbf{F}^{-1}$ is also positive definite. Since $\mathbf{g} \ne \mathbf{0}$, the quadratic form $\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} > 0$.
Since $\epsilon > 0$ and $\lambda > 0$, taking the positive square root yields:
$$\lambda^* = \sqrt{\frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \epsilon}}$$

**Step 7: Compute the Final Optimal Step Vector $\mathbf{d}^*$.**
Substitute $\lambda^*$ back into the expression for $\mathbf{d}^*$:
$$\mathbf{d}^* = \frac{1}{\lambda^*} \mathbf{F}^{-1} \mathbf{g} = \frac{1}{\sqrt{\frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \epsilon}}} \mathbf{F}^{-1} \mathbf{g} = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}$$

**Step 8: Verify Second-Order Sufficiency.**
The Hessian of the Lagrangian with respect to $\mathbf{d}$ is:
$$\nabla_{\mathbf{d}}^2 \mathcal{L}(\mathbf{d}, \lambda^*) = -\lambda^* \mathbf{F}$$
Since $\lambda^* > 0$ and $\mathbf{F} \succ \mathbf{0}$, the Hessian is strictly negative definite:
$$-\lambda^* \mathbf{F} \prec \mathbf{0}$$
Thus, $\mathcal{L}(\mathbf{d}, \lambda^*)$ is strictly concave in $\mathbf{d}$, and $\mathbf{d}^*$ is the unique, strict global maximizer.

**Step 9: Geometric Proof via Cauchy-Schwarz on Riemannian Space.**
To provide deeper geometric insight, consider the vector space $\mathbb{R}^d$ equipped with the Riemannian inner product defined by $\mathbf{F}$:
$$\langle \mathbf{u}, \mathbf{v} \rangle_{\mathbf{F}} \triangleq \mathbf{u}^\top \mathbf{F} \mathbf{v}$$
with induced norm $\|\mathbf{u}\|_{\mathbf{F}} = \sqrt{\langle \mathbf{u}, \mathbf{u} \rangle_{\mathbf{F}}} = \sqrt{\mathbf{u}^\top \mathbf{F} \mathbf{u}}$.
Notice that the objective function $\mathbf{g}^\top \mathbf{d}$ can be rewritten as:
$$\mathbf{g}^\top \mathbf{d} = (\mathbf{F} \mathbf{F}^{-1} \mathbf{g})^\top \mathbf{d} = (\mathbf{F}^{-1} \mathbf{g})^\top \mathbf{F} \mathbf{d} = \langle \mathbf{F}^{-1} \mathbf{g}, \mathbf{d} \rangle_{\mathbf{F}}$$
The constraint is $\|\mathbf{d}\|_{\mathbf{F}}^2 \le 2\epsilon \iff \|\mathbf{d}\|_{\mathbf{F}} \le \sqrt{2\epsilon}$.
By the Cauchy-Schwarz inequality for the inner product $\langle \cdot, \cdot \rangle_{\mathbf{F}}$:
$$\langle \mathbf{F}^{-1} \mathbf{g}, \mathbf{d} \rangle_{\mathbf{F}} \le \|\mathbf{F}^{-1} \mathbf{g}\|_{\mathbf{F}} \|\mathbf{d}\|_{\mathbf{F}} \le \|\mathbf{F}^{-1} \mathbf{g}\|_{\mathbf{F}} \sqrt{2\epsilon}$$
Equality holds if and only if $\mathbf{d}$ is a non-negative scalar multiple of $\mathbf{F}^{-1} \mathbf{g}$:
$$\mathbf{d}^* = c \mathbf{F}^{-1} \mathbf{g} \quad \text{for } c > 0$$
Setting the norm equal to the boundary $\sqrt{2\epsilon}$:
$$\|c \mathbf{F}^{-1} \mathbf{g}\|_{\mathbf{F}} = c \sqrt{(\mathbf{F}^{-1} \mathbf{g})^\top \mathbf{F} (\mathbf{F}^{-1} \mathbf{g})} = c \sqrt{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}} = \sqrt{2\epsilon}$$
$$c = \frac{\sqrt{2\epsilon}}{\sqrt{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} = \sqrt{\frac{2\epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}}$$
Substituting $c$ gives:
$$\mathbf{d}^* = \sqrt{\frac{2\epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}$$
This confirms the steepest ascent direction on the Riemannian manifold is unnormalized vector $\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g}$. $\blacksquare$

---

```
====================================================================================================
DERIVATION 11.17.3: Invariance of Natural Policy Gradient to Linear Reparameterization
====================================================================================================
Problem Statement:
Let θ ∈ ℝ^d parameterize a policy π_θ and consider linear transformation θ_tilde = M θ for invertible M.
Let J_tilde(θ_tilde) = J(M^-1 θ_tilde) and π_tilde(θ_tilde) = π(M^-1 θ_tilde).
Prove from first principles that:
    1. Vanilla gradient transforms as g_tilde = (M^-1)^T g (NOT invariant)
    2. Fisher matrix transforms as F_tilde = (M^-1)^T F M^-1
    3. Inverse Fisher transforms as F_tilde^-1 = M F^-1 M^T
    4. Natural gradient direction transforms contravariantly: g_tilde_nat = M g_nat
    5. Curvature is invariant: g_tilde^T F_tilde^-1 g_tilde = g^T F^-1 g
    6. Optimal step satisfies: Δθ_tilde* = M Δθ*
    7. Resulting policy distribution is strictly identical: π_tilde_{θ_tilde_new} = π_{θ_new}
====================================================================================================
```

#### Derivation 11.17.3: Invariance of Natural Policy Gradient to Linear Reparameterization

##### Part 1: Problem Statement & Mathematical Goal
Let $\theta \in \mathbb{R}^d$ parameterize a policy $\pi_{\theta}(a \mid s)$, and let $J(\theta)$ be the expected performance objective.
Consider an arbitrary linear, invertible change of coordinates (reparameterization):
$$\tilde{\theta} \triangleq \mathbf{M} \theta$$
where $\mathbf{M} \in \mathbb{R}^{d \times d}$ is an invertible constant matrix ($\det(\mathbf{M}) \ne 0$), with inverse $\mathbf{M}^{-1}$ such that $\theta = \mathbf{M}^{-1} \tilde{\theta}$.
In the new coordinate system, the policy is defined as:
$$\tilde{\pi}_{\tilde{\theta}}(a \mid s) \triangleq \pi_{\mathbf{M}^{-1} \tilde{\theta}}(a \mid s)$$
and the objective function is:
$$\tilde{J}(\tilde{\theta}) \triangleq J(\mathbf{M}^{-1} \tilde{\theta})$$
Clearly, at corresponding points $\tilde{\theta} = \mathbf{M} \theta$, both parameterizations describe the exact same physical probability distribution: $\tilde{\pi}_{\tilde{\theta}}(a \mid s) = \pi_{\theta}(a \mid s)$ for all $s, a$, and achieve identical return: $\tilde{J}(\tilde{\theta}) = J(\theta)$.

**Mathematical Goals:**
1. Derive the coordinate transformation rule for the vanilla policy gradient $\nabla_{\tilde{\theta}} \tilde{J}(\tilde{\theta})$.
2. Derive the transformation rule for the score function $\nabla_{\tilde{\theta}} \log \tilde{\pi}_{\tilde{\theta}}(a \mid s)$ and the Fisher Information Matrix $\tilde{\mathbf{F}}(\tilde{\theta})$.
3. Derive the transformation rule for the inverse Fisher Information Matrix $\tilde{\mathbf{F}}(\tilde{\theta})^{-1}$.
4. Prove that the trust-region quadratic curvature is an absolute scalar invariant:
   $$\nabla_{\tilde{\theta}} \tilde{J}(\tilde{\theta})^\top \tilde{\mathbf{F}}(\tilde{\theta})^{-1} \nabla_{\tilde{\theta}} \tilde{J}(\tilde{\theta}) = \nabla_{\theta} J(\theta)^\top \mathbf{F}(\theta)^{-1} \nabla_{\theta} J(\theta)$$
5. Prove that the optimal parameter update vector transforms contravariantly:
   $$\Delta \tilde{\theta}^* = \mathbf{M} \Delta \theta^*$$
6. Prove that the updated parameters satisfy $\tilde{\theta}_{\text{new}} = \mathbf{M} \theta_{\text{new}}$, and hence the resulting probability distributions after the update are strictly identical:
   $$\tilde{\pi}_{\tilde{\theta}_{\text{new}}}(a \mid s) = \pi_{\theta_{\text{new}}}(a \mid s) \quad \forall s \in \mathcal{S}, a \in \mathcal{A}$$
7. Prove that the standard vanilla policy gradient fails this invariance test unless $\mathbf{M}$ is an orthogonal matrix ($\mathbf{M}^\top \mathbf{M} = \mathbf{I}$).

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Invertibility:** $\mathbf{M} \in \mathbb{R}^{d \times d}$ is non-singular and constant, so $\det(\mathbf{M}) \ne 0$ and $\mathbf{M}^{-1}$ exists.
2. **Differentiability:** $J(\theta)$ and $\pi_{\theta}(a \mid s)$ are continuously differentiable with respect to $\theta$.
3. **Non-Degenerate Metric:** $\mathbf{F}(\theta) \succ \mathbf{0}$ is strictly positive definite.
4. **Equal Trust Region:** The KL trust region bound $\epsilon > 0$ is fixed to the same numerical value in both coordinate systems.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
Parameters are human-invented coordinate labels for probability distributions. If one researcher parameterizes a robotic joint controller in radians ($\theta$) and another in degrees ($\tilde{\theta} = \frac{180}{\pi} \theta$), or if one researcher scales a neural network layer's weights by a factor of 10 and divides the subsequent layer by 10, the physical agent and its actions are completely unchanged.
However, the vanilla policy gradient is a covector (type $(0, 1)$ tensor). When coordinates are scaled by $\mathbf{M}$, the covector transforms with $(\mathbf{M}^{-1})^\top$. A standard gradient descent update treats this covector as a tangent displacement vector, scaling coordinates by the inverse factor!
As a result, in vanilla policy gradients, scaling a parameter up by 10 makes its update step 100 times too small in physical effect!
In contrast, the Fisher Information Matrix is a type $(0, 2)$ metric tensor. It transforms with $(\mathbf{M}^{-1})^\top \mathbf{F} \mathbf{M}^{-1}$. Its inverse $\mathbf{F}^{-1}$ is a type $(2, 0)$ tensor.
When $\mathbf{F}^{-1}$ acts on the gradient $\mathbf{g}$, the inverse metric absorbs the coordinate distortion, transforming the covector into a true contravariant tangent vector that transforms with $\mathbf{M}$.
Natural policy gradient is coordinate-free: it operates directly on the intrinsic Riemannian manifold of probability distributions!

##### Part 4: End-to-End Step-by-Step Algebraic Proof

**Step 1: Transformation of the Vanilla Policy Gradient.**
Let $\theta = \mathbf{M}^{-1} \tilde{\theta}$. In index notation, for each component $k \in \{1, \dots, d\}$:
$$\theta_k = \sum_{l=1}^d (\mathbf{M}^{-1})_{kl} \tilde{\theta}_l$$
Taking the partial derivative of $\theta_k$ with respect to $\tilde{\theta}_j$:
$$\frac{\partial \theta_k}{\partial \tilde{\theta}_j} = (\mathbf{M}^{-1})_{kj}$$
Now apply the multivariable chain rule to differentiate $\tilde{J}(\tilde{\theta}) = J(\theta)$ with respect to $\tilde{\theta}_j$:
$$\frac{\partial \tilde{J}(\tilde{\theta})}{\partial \tilde{\theta}_j} = \sum_{k=1}^d \frac{\partial J(\theta)}{\partial \theta_k} \frac{\partial \theta_k}{\partial \tilde{\theta}_j} = \sum_{k=1}^d \frac{\partial J(\theta)}{\partial \theta_k} (\mathbf{M}^{-1})_{kj}$$
Using the transpose property of matrix elements $(\mathbf{M}^{-1})_{kj} = [(\mathbf{M}^{-1})^\top]_{jk}$:
$$\frac{\partial \tilde{J}(\tilde{\theta})}{\partial \tilde{\theta}_j} = \sum_{k=1}^d [(\mathbf{M}^{-1})^\top]_{jk} \frac{\partial J(\theta)}{\partial \theta_k}$$
Writing this in matrix-vector notation across all $j \in \{1, \dots, d\}$:
$$\nabla_{\tilde{\theta}} \tilde{J}(\tilde{\theta}) = (\mathbf{M}^{-1})^\top \nabla_{\theta} J(\theta)$$
Let $\mathbf{g}_{\theta} \triangleq \nabla_{\theta} J(\theta)$ and $\mathbf{g}_{\tilde{\theta}} \triangleq \nabla_{\tilde{\theta}} \tilde{J}(\tilde{\theta})$.
Then:
$$\mathbf{g}_{\tilde{\theta}} = (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta}$$

**Step 2: Transformation of the Score Function.**
By an identical application of the chain rule to $\tilde{\pi}_{\tilde{\theta}}(a \mid s) = \pi_{\mathbf{M}^{-1} \tilde{\theta}}(a \mid s)$:
$$\nabla_{\tilde{\theta}} \log \tilde{\pi}_{\tilde{\theta}}(a \mid s) = (\mathbf{M}^{-1})^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s)$$

**Step 3: Transformation of the Fisher Information Matrix.**
By definition, the Fisher Information Matrix in the $\tilde{\theta}$ parameterization is:
$$\tilde{\mathbf{F}}(\tilde{\theta}) = \mathbb{E}_{s \sim d^\pi, a \sim \tilde{\pi}_{\tilde{\theta}}} \left[ \nabla_{\tilde{\theta}} \log \tilde{\pi}_{\tilde{\theta}}(a \mid s) \left( \nabla_{\tilde{\theta}} \log \tilde{\pi}_{\tilde{\theta}}(a \mid s) \right)^\top \right]$$
Substitute the score function transformation from Step 2:
$$\tilde{\mathbf{F}}(\tilde{\theta}) = \mathbb{E}_{s, a} \left[ \left( (\mathbf{M}^{-1})^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s) \right) \left( (\mathbf{M}^{-1})^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s) \right)^\top \right]$$
Using the transpose rule for matrix-vector products $(A \mathbf{v})^\top = \mathbf{v}^\top A^\top$:
$$\left( (\mathbf{M}^{-1})^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s) \right)^\top = (\nabla_{\theta} \log \pi_{\theta}(a \mid s))^\top ((\mathbf{M}^{-1})^\top)^\top = (\nabla_{\theta} \log \pi_{\theta}(a \mid s))^\top \mathbf{M}^{-1}$$
Substitute this back into the expectation:
$$\tilde{\mathbf{F}}(\tilde{\theta}) = \mathbb{E}_{s, a} \left[ (\mathbf{M}^{-1})^\top \nabla_{\theta} \log \pi_{\theta}(a \mid s) (\nabla_{\theta} \log \pi_{\theta}(a \mid s))^\top \mathbf{M}^{-1} \right]$$
Because $\mathbf{M}$ is a constant matrix independent of the random variables $s$ and $a$, by linearity of expectation we pull $(\mathbf{M}^{-1})^\top$ to the front and $\mathbf{M}^{-1}$ to the rear:
$$\tilde{\mathbf{F}}(\tilde{\theta}) = (\mathbf{M}^{-1})^\top \left( \mathbb{E}_{s, a} \left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) (\nabla_{\theta} \log \pi_{\theta}(a \mid s))^\top \right] \right) \mathbf{M}^{-1}$$
Recognizing the inner expectation as $\mathbf{F}(\theta)$:
$$\tilde{\mathbf{F}}(\tilde{\theta}) = (\mathbf{M}^{-1})^\top \mathbf{F}(\theta) \mathbf{M}^{-1}$$

**Step 4: Transformation of the Inverse Fisher Matrix.**
To find $\tilde{\mathbf{F}}(\tilde{\theta})^{-1}$, we invert both sides of the relation:
$$\tilde{\mathbf{F}}(\tilde{\theta})^{-1} = \left( (\mathbf{M}^{-1})^\top \mathbf{F}(\theta) \mathbf{M}^{-1} \right)^{-1}$$
Using the matrix product inversion rule $(A B C)^{-1} = C^{-1} B^{-1} A^{-1}$:
$$\tilde{\mathbf{F}}(\tilde{\theta})^{-1} = (\mathbf{M}^{-1})^{-1} \mathbf{F}(\theta)^{-1} \left( (\mathbf{M}^{-1})^\top \right)^{-1}$$
Simplifying the matrix inverses:
1. $(\mathbf{M}^{-1})^{-1} = \mathbf{M}$
2. $\left( (\mathbf{M}^{-1})^\top \right)^{-1} = \left( (\mathbf{M}^{-1})^{-1} \right)^\top = \mathbf{M}^\top$
Substituting these:
$$\tilde{\mathbf{F}}(\tilde{\theta})^{-1} = \mathbf{M} \mathbf{F}(\theta)^{-1} \mathbf{M}^\top$$

**Step 5: Transformation of the Natural Gradient Direction.**
The unnormalized natural gradient direction in $\tilde{\theta}$ coordinates is:
$$\tilde{\mathbf{g}}_{\tilde{\theta}} \triangleq \tilde{\mathbf{F}}(\tilde{\theta})^{-1} \mathbf{g}_{\tilde{\theta}}$$
Substitute $\tilde{\mathbf{F}}(\tilde{\theta})^{-1} = \mathbf{M} \mathbf{F}(\theta)^{-1} \mathbf{M}^\top$ and $\mathbf{g}_{\tilde{\theta}} = (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta}$:
$$\tilde{\mathbf{g}}_{\tilde{\theta}} = \left( \mathbf{M} \mathbf{F}(\theta)^{-1} \mathbf{M}^\top \right) \left( (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta} \right)$$
Using matrix associativity:
$$\tilde{\mathbf{g}}_{\tilde{\theta}} = \mathbf{M} \mathbf{F}(\theta)^{-1} \left( \mathbf{M}^\top (\mathbf{M}^{-1})^\top \right) \mathbf{g}_{\theta}$$
Using the property $A^\top B^\top = (B A)^\top$:
$$\mathbf{M}^\top (\mathbf{M}^{-1})^\top = (\mathbf{M}^{-1} \mathbf{M})^\top = \mathbf{I}^\top = \mathbf{I}$$
Thus:
$$\tilde{\mathbf{g}}_{\tilde{\theta}} = \mathbf{M} \mathbf{F}(\theta)^{-1} \mathbf{I} \mathbf{g}_{\theta} = \mathbf{M} \left( \mathbf{F}(\theta)^{-1} \mathbf{g}_{\theta} \right) = \mathbf{M} \tilde{\mathbf{g}}_{\theta}$$
The unnormalized natural gradient vector in $\tilde{\theta}$ coordinates equals the transformation matrix $\mathbf{M}$ times the natural gradient vector in $\theta$ coordinates!

**Step 6: Invariance of Quadratic Curvature and Step Scaling Factor.**
Now evaluate the quadratic form curvature in $\tilde{\theta}$ coordinates:
$$\mathbf{g}_{\tilde{\theta}}^\top \tilde{\mathbf{F}}(\tilde{\theta})^{-1} \mathbf{g}_{\tilde{\theta}} = \left( (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta} \right)^\top \left( \mathbf{M} \mathbf{F}(\theta)^{-1} \mathbf{M}^\top \right) \left( (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta} \right)$$
Expand the transpose of the first term:
$$\left( (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta} \right)^\top = \mathbf{g}_{\theta}^\top ((\mathbf{M}^{-1})^\top)^\top = \mathbf{g}_{\theta}^\top \mathbf{M}^{-1}$$
Substitute this into the expression:
$$\mathbf{g}_{\tilde{\theta}}^\top \tilde{\mathbf{F}}(\tilde{\theta})^{-1} \mathbf{g}_{\tilde{\theta}} = \mathbf{g}_{\theta}^\top (\mathbf{M}^{-1} \mathbf{M}) \mathbf{F}(\theta)^{-1} (\mathbf{M}^\top (\mathbf{M}^{-1})^\top) \mathbf{g}_{\theta}$$
Because $\mathbf{M}^{-1} \mathbf{M} = \mathbf{I}$ and $\mathbf{M}^\top (\mathbf{M}^{-1})^\top = \mathbf{I}$:
$$\mathbf{g}_{\tilde{\theta}}^\top \tilde{\mathbf{F}}(\tilde{\theta})^{-1} \mathbf{g}_{\tilde{\theta}} = \mathbf{g}_{\theta}^\top \mathbf{I} \mathbf{F}(\theta)^{-1} \mathbf{I} \mathbf{g}_{\theta} = \mathbf{g}_{\theta}^\top \mathbf{F}(\theta)^{-1} \mathbf{g}_{\theta}$$
The quadratic curvature is **strictly invariant**:
$$\mathbf{g}_{\tilde{\theta}}^\top \tilde{\mathbf{F}}(\tilde{\theta})^{-1} \mathbf{g}_{\tilde{\theta}} = \mathbf{g}_{\theta}^\top \mathbf{F}(\theta)^{-1} \mathbf{g}_{\theta}$$
Consequently, the trust-region step size coefficient $\beta$ in $\tilde{\theta}$ space satisfies:
$$\tilde{\beta} = \sqrt{\frac{2\epsilon}{\mathbf{g}_{\tilde{\theta}}^\top \tilde{\mathbf{F}}(\tilde{\theta})^{-1} \mathbf{g}_{\tilde{\theta}}}} = \sqrt{\frac{2\epsilon}{\mathbf{g}_{\theta}^\top \mathbf{F}(\theta)^{-1} \mathbf{g}_{\theta}}} = \beta$$

**Step 7: Transformation of the Step Vector $\Delta \tilde{\theta}^*$.**
The optimal step in $\tilde{\theta}$ coordinates is:
$$\Delta \tilde{\theta}^* = \tilde{\beta} \tilde{\mathbf{g}}_{\tilde{\theta}} = \beta (\mathbf{M} \tilde{\mathbf{g}}_{\theta}) = \mathbf{M} (\beta \tilde{\mathbf{g}}_{\theta}) = \mathbf{M} \Delta \theta^*$$
Thus, the step vector transforms identically to the coordinate system itself:
$$\Delta \tilde{\theta}^* = \mathbf{M} \Delta \theta^*$$

**Step 8: Exact Equivalence of the Updated Policy Distributions.**
The updated parameter vector in $\tilde{\theta}$ space is:
$$\tilde{\theta}_{\text{new}} = \tilde{\theta} + \Delta \tilde{\theta}^* = \mathbf{M} \theta + \mathbf{M} \Delta \theta^* = \mathbf{M} (\theta + \Delta \theta^*) = \mathbf{M} \theta_{\text{new}}$$
Now evaluate the updated policy distribution under the new coordinates at any state $s$ and action $a$:
$$\tilde{\pi}_{\tilde{\theta}_{\text{new}}}(a \mid s) = \pi_{\mathbf{M}^{-1} \tilde{\theta}_{\text{new}}}(a \mid s) = \pi_{\mathbf{M}^{-1} (\mathbf{M} \theta_{\text{new}})}(a \mid s) = \pi_{\theta_{\text{new}}}(a \mid s)$$
This proves that the physical distribution step taken by the Natural Policy Gradient is **100% identical**, down to machine precision, regardless of the choice of invertible linear coordinates $\mathbf{M}$.

**Step 9: Contrast with Vanilla Policy Gradient Failure.**
Now consider what happens under standard vanilla policy gradient updates with learning rate $\alpha > 0$:
In $\theta$ coordinates:
$$\Delta \theta_{\text{vanilla}} = \alpha \mathbf{g}_{\theta}$$
In $\tilde{\theta}$ coordinates:
$$\Delta \tilde{\theta}_{\text{vanilla}} = \alpha \mathbf{g}_{\tilde{\theta}} = \alpha (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta}$$
To see what physical change this produces in the original parameter space, multiply $\Delta \tilde{\theta}_{\text{vanilla}}$ by $\mathbf{M}^{-1}$:
$$\Delta \theta_{\text{effective}} = \mathbf{M}^{-1} \Delta \tilde{\theta}_{\text{vanilla}} = \alpha \mathbf{M}^{-1} (\mathbf{M}^{-1})^\top \mathbf{g}_{\theta} = \alpha (\mathbf{M}^\top \mathbf{M})^{-1} \mathbf{g}_{\theta}$$
Comparing $\Delta \theta_{\text{effective}}$ to $\Delta \theta_{\text{vanilla}} = \alpha \mathbf{g}_{\theta}$:
$$\Delta \theta_{\text{effective}} = \Delta \theta_{\text{vanilla}} \iff (\mathbf{M}^\top \mathbf{M})^{-1} = \mathbf{I} \iff \mathbf{M}^\top \mathbf{M} = \mathbf{I}$$
This condition holds **if and only if $\mathbf{M}$ is an orthogonal matrix** (a pure rotation or reflection).
If $\mathbf{M}$ includes ANY coordinate scaling (e.g., $\mathbf{M} = \operatorname{diag}(10, 1)$), then $(\mathbf{M}^\top \mathbf{M})^{-1} = \operatorname{diag}(0.01, 1) \ne \mathbf{I}$.
The vanilla gradient step is distorted by a factor of $10^2 = 100$ along the scaled coordinate!
Only the Natural Policy Gradient correctly undoes this distortion and achieves true coordinate invariance. $\blacksquare$

---

## 3. Geometric Interpretation: Mercator Distortion vs. Geodesic Distance

- In flat Euclidean gradient descent, the gradient $\nabla_{\theta} J$ is orthogonal to the contour lines on a paper map. But on a curved surface (like a Mercator projection of Earth), walking 1 inch on the map near the equator covers 2,000 miles, whereas walking 1 inch near the poles covers 200 miles.
- The Fisher Information Matrix $\mathbf{F}$ is the **metric tensor of the globe**: it converts paper map coordinates into true geodesic physical distances on Earth.
- Multiplying by $\mathbf{F}^{-1}$ rotates and stretches the gradient vector so that the agent takes uniform, isotropic steps across the actual probability manifold!

```
                    EUCLIDEAN vs NATURAL GRADIENT FLOW
                                                                    
         Contour of J(theta)                                        
               \   /                                                
                \ /     Vanilla Gradient g (Orthogonal in parameter map)
                 *----------------------->                           
                  \     \                                           
                   \     \ Natural Gradient F^-1 g (Steepest on manifold!)
                    \     v                                         
                     \                                              
                      +--------------------------> theta            
```

---

## 4. Real-World Analogy: Steering a High-Performance Aircraft

Imagine flying a supersonic jet:
- The pilot's controls (the parameters $\theta$) have non-linear mechanical linkages: at low speeds ($100$ knots), deflecting the stick by $5^\circ$ barely turns the jet. At supersonic speeds (Mach 2), deflecting the stick by $5^\circ$ tears the wings off due to extreme aerodynamic loads.
- **Euclidean Policy Gradient:** Dictates: "Always move the stick by $5^\circ$." At high speeds, the jet disintegrates.
- **Natural Policy Gradient:** Senses the aerodynamic pressure manifold (Fisher Information Matrix) and dictates: "Apply whatever control deflection changes the aircraft's physical flight path by exactly $1.0^\circ$ of angular pitch." At low speeds, the stick moves $10^\circ$; at Mach 2, it moves $0.1^\circ$, maintaining safety across all regimes!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 2-Action Softmax Policy
Consider an agent with two discrete actions $\mathcal{A} = \{a_0, a_1\}$.
The policy parameters are $\theta = [\theta_0, \theta_1]^\top$.

**Current Parameter State:**
$$\pi(a_0) = 0.8000, \quad \pi(a_1) = 0.2000$$

**Observed Standard (Vanilla) Policy Gradient:**
$$\mathbf{g} = \nabla_{\theta} J = \begin{bmatrix} +1.0000 \\ -1.0000 \end{bmatrix}$$

**Hyperparameters:**
- KL divergence trust region: $\epsilon = 0.0100$
- Tikhonov regularization damping factor: $\delta_{\text{damp}} = 0.0400$ (ensures invertibility of $\mathbf{F}$)

We will compute:
1. Analytical score function vectors $\nabla_{\theta} \log \pi(a_0)$ and $\nabla_{\theta} \log \pi(a_1)$
2. The Fisher Information Matrix $\mathbf{F}$
3. The regularized inverse $\mathbf{F}_{\text{reg}}^{-1}$
4. The unnormalized natural gradient direction $\tilde{\mathbf{g}} = \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$
5. The quadratic form curvature $\mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$
6. The step size $\beta$ and the final normalized parameter update $\Delta \theta^*$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Walkthrough Value |
| :--- | :--- | :--- |
| $\pi$ | Action Probabilities | $[\pi_0 = 0.8000, \pi_1 = 0.2000]^\top$ |
| $\mathbf{s}_0, \mathbf{s}_1$ | Score Vectors $\nabla \log \pi(a_i)$ | $\mathbf{s}_0 = [1 - \pi_0, -\pi_1]^\top, \mathbf{s}_1 = [-\pi_0, 1 - \pi_1]^\top$ |
| $\mathbf{F}$ | Fisher Information Matrix | $\pi_0 \mathbf{s}_0 \mathbf{s}_0^\top + \pi_1 \mathbf{s}_1 \mathbf{s}_1^\top$ |
| $\mathbf{F}_{\text{reg}}$ | Damped Fisher Matrix | $\mathbf{F} + \delta_{\text{damp}} \mathbf{I}$ |
| $\mathbf{F}_{\text{reg}}^{-1}$ | Inverse Metric Tensor | Analytical $2 \times 2$ inverse |
| $\mathbf{g}$ | Vanilla Policy Gradient | $[+1.0000, -1.0000]^\top$ |
| $\tilde{\mathbf{g}}$ | Natural Gradient Direction | $\mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$ |
| $\beta$ | Trust Region Step Scaling Factor | $\sqrt{\frac{2\epsilon}{\mathbf{g}^\top \tilde{\mathbf{g}}}}$ |
| $\Delta \theta^*$ | Final Natural Policy Gradient Step | $\beta \tilde{\mathbf{g}}$ |

---

### 5.3 Step 1: Compute Score Functions

For action $a_0$:
$$\mathbf{s}_0 = \nabla_{\theta} \log \pi(a_0) = \begin{bmatrix} 1 - \pi_0 \\ -\pi_1 \end{bmatrix} = \begin{bmatrix} 1 - 0.8000 \\ -0.2000 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.2000} \\ \mathbf{-0.2000} \end{bmatrix}$$

For action $a_1$:
$$\mathbf{s}_1 = \nabla_{\theta} \log \pi(a_1) = \begin{bmatrix} -\pi_0 \\ 1 - \pi_1 \end{bmatrix} = \begin{bmatrix} -0.8000 \\ 1 - 0.2000 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.8000} \\ \mathbf{+0.8000} \end{bmatrix}$$

---

### 5.4 Step 2: Compute the Fisher Information Matrix $\mathbf{F}$

The outer products are:
$$\mathbf{s}_0 \mathbf{s}_0^\top = \begin{bmatrix} +0.2 \\ -0.2 \end{bmatrix} \begin{bmatrix} +0.2 & -0.2 \end{bmatrix} = \begin{bmatrix} 0.0400 & -0.0400 \\ -0.0400 & 0.0400 \end{bmatrix}$$
$$\mathbf{s}_1 \mathbf{s}_1^\top = \begin{bmatrix} -0.8 \\ +0.8 \end{bmatrix} \begin{bmatrix} -0.8 & +0.8 \end{bmatrix} = \begin{bmatrix} 0.6400 & -0.6400 \\ -0.6400 & 0.6400 \end{bmatrix}$$

Weight by probabilities:
$$\mathbf{F} = \pi_0 (\mathbf{s}_0 \mathbf{s}_0^\top) + \pi_1 (\mathbf{s}_1 \mathbf{s}_1^\top)$$
$$= 0.8000 \begin{bmatrix} 0.0400 & -0.0400 \\ -0.0400 & 0.0400 \end{bmatrix} + 0.2000 \begin{bmatrix} 0.6400 & -0.6400 \\ -0.6400 & 0.6400 \end{bmatrix}$$
$$= \begin{bmatrix} 0.0320 & -0.0320 \\ -0.0320 & 0.0320 \end{bmatrix} + \begin{bmatrix} 0.1280 & -0.1280 \\ -0.1280 & 0.1280 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.1600 & -0.1600 \\ -0.1600 & 0.1600 \end{bmatrix}}$$

---

### 5.5 Step 3: Regularize and Invert $\mathbf{F}$

Notice that $\det(\mathbf{F}) = (0.16)(0.16) - (-0.16)(-0.16) = 0$ (rank-deficient along the sum constraint!).
Add damping factor $\delta_{\text{damp}} = 0.0400 \mathbf{I}$:

$$\mathbf{F}_{\text{reg}} = \begin{bmatrix} 0.1600 & -0.1600 \\ -0.1600 & 0.1600 \end{bmatrix} + \begin{bmatrix} 0.0400 & 0.0000 \\ 0.0000 & 0.0400 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.2000 & -0.1600 \\ -0.1600 & 0.2000 \end{bmatrix}}$$

Compute Determinant:
$$\det(\mathbf{F}_{\text{reg}}) = (0.2000 \times 0.2000) - (-0.1600 \times -0.1600) = 0.0400 - 0.0256 = \mathbf{0.0144}$$

Compute Matrix Inverse:
$$\mathbf{F}_{\text{reg}}^{-1} = \frac{1}{0.0144} \begin{bmatrix} 0.2000 & 0.1600 \\ 0.1600 & 0.2000 \end{bmatrix} \approx \mathbf{\begin{bmatrix} 13.88889 & 11.11111 \\ 11.11111 & 13.88889 \end{bmatrix}}$$

---

### 5.6 Step 4: Compute Natural Gradient Direction $\tilde{\mathbf{g}}$

$$\tilde{\mathbf{g}} = \mathbf{F}_{\text{reg}}^{-1} \mathbf{g} = \begin{bmatrix} 13.88889 & 11.11111 \\ 11.11111 & 13.88889 \end{bmatrix} \begin{bmatrix} +1.0000 \\ -1.0000 \end{bmatrix}$$
$$\tilde{\mathbf{g}} = \begin{bmatrix} 13.88889 - 11.11111 \\ 11.11111 - 13.88889 \end{bmatrix} = \mathbf{\begin{bmatrix} +2.77778 \\ -2.77778 \end{bmatrix}}$$

---

### 5.7 Step 5: Trust Region Normalization ($\epsilon = 0.0100$)

Compute quadratic form curvature:
$$\mathbf{g}^\top \tilde{\mathbf{g}} = \mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g} = (+1.0000 \times 2.77778) + (-1.0000 \times -2.77778) = 2.77778 + 2.77778 = \mathbf{5.55556}$$

Compute scaling coefficient $\beta$:
$$\beta = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}}} = \sqrt{\frac{2 \times 0.0100}{5.55556}} = \sqrt{\frac{0.0200}{5.55556}} = \sqrt{0.003600} = \mathbf{0.06000}$$

Final Natural Policy Gradient Step:
$$\Delta \theta^* = \beta \tilde{\mathbf{g}} = 0.06000 \times \begin{bmatrix} +2.77778 \\ -2.77778 \end{bmatrix} = \mathbf{\begin{bmatrix} +0.16667 \\ -0.16667 \end{bmatrix}}$$

Check KL divergence constraint satisfaction:
$$\frac{1}{2} (\Delta \theta^*)^\top \mathbf{F}_{\text{reg}} (\Delta \theta^*) = \frac{1}{2} (0.0600)^2 (5.55556) = \frac{1}{2} (0.0036)(5.55556) = \mathbf{0.0100} = \epsilon \quad \checkmark$$

---

### 5.8 Visual Summary Matrix Grid

| Matrix / Vector | Formula / Operation | Value | Significance |
| :--- | :--- | :---: | :--- |
| **Score $\mathbf{s}_0$** | $[1 - \pi_0, -\pi_1]^\top$ | $[+0.2000, -0.2000]^\top$ | Sensitivity to action $a_0$ |
| **Score $\mathbf{s}_1$** | $[-\pi_0, 1 - \pi_1]^\top$ | $[-0.8000, +0.8000]^\top$ | Sensitivity to action $a_1$ |
| **Fisher $\mathbf{F}$** | $\sum \pi_i \mathbf{s}_i \mathbf{s}_i^\top$ | $\begin{bmatrix} 0.16 & -0.16 \\ -0.16 & 0.16 \end{bmatrix}$ | Metric tensor of policy manifold |
| **Damped $\mathbf{F}_{\text{reg}}$** | $\mathbf{F} + 0.04 \mathbf{I}$ | $\begin{bmatrix} 0.20 & -0.16 \\ -0.16 & 0.20 \end{bmatrix}$ | Invertible Riemannian metric |
| **Curvature $\mathbf{g}^\top \tilde{\mathbf{g}}$** | $\mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$ | $\mathbf{5.55556}$ | Metric curvature along gradient |
| **Step Multiplier $\beta$** | $\sqrt{2\epsilon / 5.55556}$ | $\mathbf{0.06000}$ | Normalizes step to KL $\le 0.01$ |
| **Final Step $\Delta \theta^*$** | $\beta \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$ | $\mathbf{[+0.16667, -0.16667]^\top}$ | Exact Riemannian update! |

---

## 6. Solved Illustrations

### Illustration 1: Analytical Fisher Matrix of a 1D Gaussian Policy (Analytical Derivation & Solved Numerical Walkthrough)
**Problem:**
Let $\pi_{\theta}(a) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(a - \mu)^2}{2\sigma^2} \right)$ be a 1D Gaussian policy with parameter vector $\theta = [\mu, \sigma]^\top$.
1. Derive the analytical Fisher Information Matrix $\mathbf{F}(\mu, \sigma)$ from first principles.
2. Given parameters $\mu = 1.5000, \sigma = 0.5000$, vanilla policy gradient $\mathbf{g} = \nabla_{\theta} J = [0.8000, -0.4000]^\top$, and trust region bound $\epsilon = 0.0200$, calculate step-by-step:
   - The numerical Fisher Information Matrix $\mathbf{F}$ and its inverse $\mathbf{F}^{-1}$.
   - The unnormalized natural gradient direction $\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g}$.
   - The trust-region quadratic curvature $\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}$ and step size $\beta$.
   - The final natural policy gradient update $\Delta \theta^*$.
   - Verify that $\frac{1}{2} (\Delta \theta^*)^\top \mathbf{F} (\Delta \theta^*) = \epsilon$.

**Analytical Derivation:**
Log-likelihood function:
$$\log \pi(a) = -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(a - \mu)^2}{2\sigma^2}$$

Compute first partial derivatives (score functions):
$$\frac{\partial \log \pi}{\partial \mu} = \frac{a - \mu}{\sigma^2}, \quad \frac{\partial \log \pi}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(a - \mu)^2}{\sigma^3}$$

Compute expectations of outer products using central Gaussian moments:
$\mathbb{E}[(a - \mu)] = 0$, $\mathbb{E}[(a - \mu)^2] = \sigma^2$, $\mathbb{E}[(a - \mu)^3] = 0$, and $\mathbb{E}[(a - \mu)^4] = 3\sigma^4$.
1. Diagonal mean entry $F_{\mu\mu}$:
   $$F_{\mu\mu} = \mathbb{E}\left[ \left( \frac{a - \mu}{\sigma^2} \right)^2 \right] = \frac{\mathbb{E}[(a - \mu)^2]}{\sigma^4} = \frac{\sigma^2}{\sigma^4} = \mathbf{\frac{1}{\sigma^2}}$$
2. Off-diagonal cross entry $F_{\mu\sigma}$:
   $$F_{\mu\sigma} = \mathbb{E}\left[ \left( \frac{a - \mu}{\sigma^2} \right) \left( -\frac{1}{\sigma} + \frac{(a - \mu)^2}{\sigma^3} \right) \right] = -\frac{\mathbb{E}[a - \mu]}{\sigma^3} + \frac{\mathbb{E}[(a - \mu)^3]}{\sigma^5} = 0 + 0 = \mathbf{0}$$
3. Diagonal standard deviation entry $F_{\sigma\sigma}$:
   $$F_{\sigma\sigma} = \mathbb{E}\left[ \left( -\frac{1}{\sigma} + \frac{(a - \mu)^2}{\sigma^3} \right)^2 \right] = \mathbb{E}\left[ \frac{1}{\sigma^2} - \frac{2(a - \mu)^2}{\sigma^4} + \frac{(a - \mu)^4}{\sigma^6} \right]$$
   $$= \frac{1}{\sigma^2} - \frac{2\sigma^2}{\sigma^4} + \frac{3\sigma^4}{\sigma^6} = \frac{1}{\sigma^2} - \frac{2}{\sigma^2} + \frac{3}{\sigma^2} = \mathbf{\frac{2}{\sigma^2}}$$

Analytical Matrix:
$$\mathbf{F}(\mu, \sigma) = \begin{bmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{2}{\sigma^2} \end{bmatrix}$$

**Numerical Walkthrough:**
*Step 1: Evaluate $\mathbf{F}$ at $\mu = 1.5000, \sigma = 0.5000$:*
$$F_{\mu\mu} = \frac{1}{0.5000^2} = \frac{1}{0.2500} = 4.0000$$
$$F_{\sigma\sigma} = \frac{2}{0.5000^2} = \frac{2}{0.2500} = 8.0000$$
$$\mathbf{F} = \begin{bmatrix} 4.0000 & 0.0000 \\ 0.0000 & 8.0000 \end{bmatrix}$$

*Step 2: Invert $\mathbf{F}$:*
Because $\mathbf{F}$ is diagonal, inversion is trivial:
$$\mathbf{F}^{-1} = \begin{bmatrix} \frac{1}{4.0000} & 0.0000 \\ 0.0000 & \frac{1}{8.0000} \end{bmatrix} = \begin{bmatrix} 0.2500 & 0.0000 \\ 0.0000 & 0.1250 \end{bmatrix}$$

*Step 3: Compute Unnormalized Natural Gradient $\tilde{\mathbf{g}}$:*
$$\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g} = \begin{bmatrix} 0.2500 & 0.0000 \\ 0.0000 & 0.1250 \end{bmatrix} \begin{bmatrix} 0.8000 \\ -0.4000 \end{bmatrix} = \begin{bmatrix} 0.2500 \times 0.8000 \\ 0.1250 \times (-0.4000) \end{bmatrix} = \begin{bmatrix} \mathbf{+0.2000} \\ \mathbf{-0.0500} \end{bmatrix}$$

*Step 4: Compute Curvature and Trust-Region Scaling Factor $\beta$:*
$$\mathbf{g}^\top \tilde{\mathbf{g}} = (0.8000)(0.2000) + (-0.4000)(-0.0500) = 0.1600 + 0.0200 = \mathbf{0.1800}$$
Given $\epsilon = 0.0200$:
$$\beta = \sqrt{\frac{2\epsilon}{\mathbf{g}^\top \tilde{\mathbf{g}}}} = \sqrt{\frac{2 \times 0.0200}{0.1800}} = \sqrt{\frac{0.0400}{0.1800}} = \sqrt{\frac{2}{9}} = \frac{\sqrt{2}}{3} \approx \mathbf{0.471405}$$

*Step 5: Compute Final Parameter Update $\Delta \theta^*$:*
$$\Delta \theta^* = \beta \tilde{\mathbf{g}} = 0.471405 \begin{bmatrix} 0.2000 \\ -0.0500 \end{bmatrix} = \begin{bmatrix} 0.471405 \times 0.2000 \\ 0.471405 \times (-0.0500) \end{bmatrix} = \begin{bmatrix} \mathbf{+0.094281} \\ \mathbf{-0.023570} \end{bmatrix}$$

*Step 6: Verify Trust-Region Constraint:*
$$\frac{1}{2} (\Delta \theta^*)^\top \mathbf{F} (\Delta \theta^*) = \frac{1}{2} \left[ 4.0000 \times (0.094281)^2 + 8.0000 \times (-0.023570)^2 \right]$$
$$= \frac{1}{2} \left[ 4.0000 \times 0.0088889 + 8.0000 \times 0.00055556 \right]$$
$$= \frac{1}{2} \left[ 0.0355556 + 0.0044444 \right] = \frac{1}{2} (0.040000) = \mathbf{0.0200} = \epsilon \quad \checkmark$$

---

### Illustration 2: Analytical Fisher Information Matrix of a 2-Action Softmax Policy
**Problem:**
Consider a policy with discrete actions $\mathcal{A} = \{a_0, a_1\}$ parameterized by logits $\theta = [\theta_0, \theta_1]^\top \in \mathbb{R}^2$:
$$\pi_0 = \frac{e^{\theta_0}}{e^{\theta_0} + e^{\theta_1}}, \quad \pi_1 = \frac{e^{\theta_1}}{e^{\theta_0} + e^{\theta_1}}$$
1. Prove analytically that for any discrete softmax policy, the Fisher Information Matrix satisfies $\mathbf{F}(\theta) = \operatorname{diag}(\pi) - \pi \pi^\top$.
2. For logits $\theta_0 = \ln(3) \approx 1.098612$ and $\theta_1 = 0.000000$, compute $\pi$ and evaluate $\mathbf{F}(\theta)$ via:
   - The analytical formula $\operatorname{diag}(\pi) - \pi \pi^\top$.
   - The expectation of outer products $\sum_{i} \pi_i \mathbf{s}_i \mathbf{s}_i^\top$.
   Confirm step-by-step arithmetic matches identically.

**Analytical Proof:**
For any softmax policy $\pi_i = \frac{e^{\theta_i}}{\sum_k e^{\theta_k}}$, the partial derivative is:
$$\frac{\partial \pi_i}{\partial \theta_j} = \begin{cases} \pi_i (1 - \pi_i) & \text{if } i = j \\ -\pi_i \pi_j & \text{if } i \ne j \end{cases} = \pi_i (\delta_{ij} - \pi_j)$$
The score function with respect to parameter $\theta_j$ is:
$$s_{i, j} = \frac{\partial \log \pi_i}{\partial \theta_j} = \frac{1}{\pi_i} \frac{\partial \pi_i}{\partial \theta_j} = \delta_{ij} - \pi_j$$
In vector notation, the score vector for action $a_i$ is $\mathbf{s}_i = \mathbf{e}_i - \pi$, where $\mathbf{e}_i$ is the $i$-th standard basis vector.
The Fisher Information Matrix is the expectation of the outer product:
$$\mathbf{F}(\theta) = \sum_{i} \pi_i \mathbf{s}_i \mathbf{s}_i^\top = \sum_i \pi_i (\mathbf{e}_i - \pi)(\mathbf{e}_i - \pi)^\top$$
Expanding the outer product:
$$(\mathbf{e}_i - \pi)(\mathbf{e}_i - \pi)^\top = \mathbf{e}_i \mathbf{e}_i^\top - \mathbf{e}_i \pi^\top - \pi \mathbf{e}_i^\top + \pi \pi^\top$$
Multiplying by $\pi_i$ and summing over all actions $i$:
1. $\sum_i \pi_i \mathbf{e}_i \mathbf{e}_i^\top = \operatorname{diag}(\pi)$
2. $\sum_i \pi_i \mathbf{e}_i \pi^\top = \left( \sum_i \pi_i \mathbf{e}_i \right) \pi^\top = \pi \pi^\top$
3. $\sum_i \pi_i \pi \mathbf{e}_i^\top = \pi \left( \sum_i \pi_i \mathbf{e}_i \right)^\top = \pi \pi^\top$
4. $\sum_i \pi_i \pi \pi^\top = \left( \sum_i \pi_i \right) \pi \pi^\top = 1 \cdot \pi \pi^\top = \pi \pi^\top$

Summing these four terms:
$$\mathbf{F}(\theta) = \operatorname{diag}(\pi) - \pi\pi^\top - \pi\pi^\top + \pi\pi^\top = \mathbf{\operatorname{diag}(\pi) - \pi\pi^\top} \quad \blacksquare$$

**Numerical Computation:**
*Step 1: Compute Action Probabilities:*
$$e^{\theta_0} = e^{\ln(3)} = 3.000000, \quad e^{\theta_1} = e^0 = 1.000000$$
$$\sum_k e^{\theta_k} = 3.000000 + 1.000000 = 4.000000$$
$$\pi_0 = \frac{3.000000}{4.000000} = \mathbf{0.750000}, \quad \pi_1 = \frac{1.000000}{4.000000} = \mathbf{0.250000}$$

*Step 2: Method A (Analytical Formula $\operatorname{diag}(\pi) - \pi\pi^\top$):*
$$\operatorname{diag}(\pi) = \begin{bmatrix} 0.750000 & 0.000000 \\ 0.000000 & 0.250000 \end{bmatrix}$$
$$\pi \pi^\top = \begin{bmatrix} 0.750000 \\ 0.250000 \end{bmatrix} \begin{bmatrix} 0.750000 & 0.250000 \end{bmatrix} = \begin{bmatrix} 0.7500^2 & 0.7500 \times 0.2500 \\ 0.2500 \times 0.7500 & 0.2500^2 \end{bmatrix} = \begin{bmatrix} 0.562500 & 0.187500 \\ 0.187500 & 0.062500 \end{bmatrix}$$
Subtracting the matrices:
$$\mathbf{F}(\theta) = \begin{bmatrix} 0.750000 - 0.562500 & 0.000000 - 0.187500 \\ 0.000000 - 0.187500 & 0.250000 - 0.062500 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.187500 & -0.187500 \\ -0.187500 & 0.187500 \end{bmatrix}}$$

*Step 3: Method B (Score Vectors & Outer Products):*
For action $a_0$:
$$\mathbf{s}_0 = \mathbf{e}_0 - \pi = \begin{bmatrix} 1 - 0.7500 \\ 0 - 0.2500 \end{bmatrix} = \begin{bmatrix} +0.2500 \\ -0.2500 \end{bmatrix}$$
$$\mathbf{s}_0 \mathbf{s}_0^\top = \begin{bmatrix} +0.2500 \\ -0.2500 \end{bmatrix} \begin{bmatrix} +0.2500 & -0.2500 \end{bmatrix} = \begin{bmatrix} 0.062500 & -0.062500 \\ -0.062500 & 0.062500 \end{bmatrix}$$
For action $a_1$:
$$\mathbf{s}_1 = \mathbf{e}_1 - \pi = \begin{bmatrix} 0 - 0.7500 \\ 1 - 0.2500 \end{bmatrix} = \begin{bmatrix} -0.7500 \\ +0.7500 \end{bmatrix}$$
$$\mathbf{s}_1 \mathbf{s}_1^\top = \begin{bmatrix} -0.7500 \\ +0.7500 \end{bmatrix} \begin{bmatrix} -0.7500 & +0.7500 \end{bmatrix} = \begin{bmatrix} 0.562500 & -0.562500 \\ -0.562500 & 0.562500 \end{bmatrix}$$
Weighting by action probabilities:
$$\mathbf{F}(\theta) = 0.7500 \begin{bmatrix} 0.062500 & -0.062500 \\ -0.062500 & 0.062500 \end{bmatrix} + 0.2500 \begin{bmatrix} 0.562500 & -0.562500 \\ -0.562500 & 0.562500 \end{bmatrix}$$
$$= \begin{bmatrix} 0.046875 & -0.046875 \\ -0.046875 & 0.046875 \end{bmatrix} + \begin{bmatrix} 0.140625 & -0.140625 \\ -0.140625 & 0.140625 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.187500 & -0.187500 \\ -0.187500 & 0.187500 \end{bmatrix}}$$
Both methods match to machine precision. $\blacksquare$

---

### Illustration 3: Natural Gradient vs. Vanilla Gradient on an Ill-Conditioned Plateau
**Problem:**
Suppose an agent operates in an ill-conditioned policy landscape where parameter $\theta_0$ causes massive distribution change, while parameter $\theta_1$ controls a flat, insensitive plateau:
$$\mathbf{F} = \begin{bmatrix} 25.0000 & 0.0000 \\ 0.0000 & 0.0400 \end{bmatrix}$$
The condition number of the metric tensor is $\kappa = 25.0000 / 0.0400 = 625$.
The observed policy gradient is $\mathbf{g} = [1.0000, 0.2000]^\top$, and the allowable KL trust region is $\epsilon = 0.0100$.
1. Compute the Vanilla policy gradient step $\Delta \theta_{\text{vanilla}}$ normalized to satisfy the trust-region budget $\frac{1}{2} \Delta \theta^\top \mathbf{F} \Delta \theta = \epsilon$.
2. Compute the Natural policy gradient step $\Delta \theta_{\text{natural}}$ under the same trust-region budget.
3. Compare the objective improvements $\Delta J \approx \mathbf{g}^\top \Delta \theta$ and explain the geometric speedup.

**Step-by-Step Solution:**

*Step 1: Normalized Vanilla Gradient Step:*
The unnormalized vanilla update is $\mathbf{d}_{\text{van}} = \mathbf{g} = [1.0000, 0.2000]^\top$.
To satisfy the trust-region constraint $\frac{1}{2} (\alpha \mathbf{g})^\top \mathbf{F} (\alpha \mathbf{g}) = \epsilon$, compute the metric norm of $\mathbf{g}$:
$$\mathbf{g}^\top \mathbf{F} \mathbf{g} = (1.0000)^2 \times 25.0000 + (0.2000)^2 \times 0.0400 = 25.0000 + 0.0400 \times 0.0400 = 25.0000 + 0.0016 = \mathbf{25.0016}$$
The maximum step size $\alpha$ that satisfies $\frac{1}{2} \alpha^2 (\mathbf{g}^\top \mathbf{F} \mathbf{g}) = \epsilon$ is:
$$\alpha = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F} \mathbf{g}}} = \sqrt{\frac{2 \times 0.0100}{25.0016}} = \sqrt{\frac{0.0200}{25.0016}} \approx \mathbf{0.028283}$$
The resulting vanilla parameter update vector is:
$$\Delta \theta_{\text{vanilla}} = 0.028283 \begin{bmatrix} 1.0000 \\ 0.2000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.028283 \\ 0.005657 \end{bmatrix}}$$
The first-order expected return improvement under vanilla gradient:
$$\Delta J_{\text{vanilla}} = \mathbf{g}^\top \Delta \theta_{\text{vanilla}} = 1.0000 \times 0.028283 + 0.2000 \times 0.005657 = 0.028283 + 0.001131 = \mathbf{0.029415}$$

*Step 2: Natural Policy Gradient Step:*
Compute the inverse metric tensor:
$$\mathbf{F}^{-1} = \begin{bmatrix} \frac{1}{25.0000} & 0.0000 \\ 0.0000 & \frac{1}{0.0400} \end{bmatrix} = \begin{bmatrix} 0.0400 & 0.0000 \\ 0.0000 & 25.0000 \end{bmatrix}$$
Compute the unnormalized natural gradient direction:
$$\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g} = \begin{bmatrix} 0.0400 & 0.0000 \\ 0.0000 & 25.0000 \end{bmatrix} \begin{bmatrix} 1.0000 \\ 0.2000 \end{bmatrix} = \begin{bmatrix} 0.0400 \times 1.0000 \\ 25.0000 \times 0.2000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.0400 \\ 5.0000 \end{bmatrix}}$$
Compute the Riemannian curvature:
$$\mathbf{g}^\top \tilde{\mathbf{g}} = \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} = 1.0000 \times 0.0400 + 0.2000 \times 5.0000 = 0.0400 + 1.0000 = \mathbf{1.0400}$$
Compute the natural step size $\beta$:
$$\beta = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} = \sqrt{\frac{2 \times 0.0100}{1.0400}} = \sqrt{\frac{0.0200}{1.0400}} = \sqrt{\frac{1}{52}} \approx \mathbf{0.138675}$$
The final natural policy gradient parameter update is:
$$\Delta \theta_{\text{natural}} = \beta \tilde{\mathbf{g}} = 0.138675 \begin{bmatrix} 0.0400 \\ 5.0000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.005547 \\ 0.693375 \end{bmatrix}}$$
The first-order expected return improvement under natural gradient:
$$\Delta J_{\text{natural}} = \mathbf{g}^\top \Delta \theta_{\text{natural}} = 1.0000 \times 0.005547 + 0.2000 \times 0.693375 = 0.005547 + 0.138675 = \mathbf{0.144222}$$

*Step 3: Comparison & Geometric Interpretation:*
Ratio of return improvements for the identical KL divergence constraint ($\epsilon = 0.0100$):
$$\frac{\Delta J_{\text{natural}}}{\Delta J_{\text{vanilla}}} = \frac{0.144222}{0.029415} \approx \mathbf{4.9031\times \text{ speedup!}}$$

**Key Insight:** 
- In vanilla gradient ascent, the agent pushes heavily along $\theta_0$ ($0.028283$) because $g_0 = 1.0$ is large. But because $\theta_0$ has huge curvature ($F_{00} = 25$), this tiny step exhausts the entire KL divergence budget ($0.5 \times 25 \times 0.028283^2 = 0.009999$), preventing the policy from advancing along $\theta_1$ ($0.005657$), leaving it stranded on the plateau.
- Natural policy gradient senses that moving along $\theta_0$ is dangerous (steep manifold curvature) and moving along $\theta_1$ is safe (flat manifold curvature). It preconditions the step, drastically damping the sensitive parameter ($0.005547$) while surging along the plateau ($0.693375$), yielding almost $5\times$ faster learning without violating safety bounds! $\blacksquare$

---

### Illustration 4: Empirical Fisher Matrix Estimation and Damped Inversion ($N = 4$)
**Problem:**
In deep reinforcement learning, the expectation over $(s, a)$ is approximated using a batch of $N$ transition samples. Suppose an empirical batch of $N = 4$ transitions produces the following score vectors $\mathbf{g}_i = \nabla_{\theta} \log \pi(a_i \mid s_i) \in \mathbb{R}^2$:
$$\mathbf{g}_1 = \begin{bmatrix} 1.0000 \\ 0.5000 \end{bmatrix}, \quad \mathbf{g}_2 = \begin{bmatrix} -0.5000 \\ 1.0000 \end{bmatrix}, \quad \mathbf{g}_3 = \begin{bmatrix} 0.8000 \\ -0.4000 \end{bmatrix}, \quad \mathbf{g}_4 = \begin{bmatrix} -0.6000 \\ -0.8000 \end{bmatrix}$$
1. Compute the empirical Fisher Information Matrix $\mathbf{F}_{\text{emp}} = \frac{1}{N} \sum_{i=1}^N \mathbf{g}_i \mathbf{g}_i^\top$.
2. Apply Tikhonov damping with regularization parameter $\lambda = 10^{-3} = 0.0010$:
   $$\mathbf{F}_{\text{damp}} = \mathbf{F}_{\text{emp}} + \lambda \mathbf{I}$$
3. Invert $\mathbf{F}_{\text{damp}}$ using the exact analytical $2 \times 2$ matrix inverse formula:
   $$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$
4. For an observed sample policy gradient $\hat{\mathbf{g}} = [0.5000, -0.2000]^\top$, compute the natural policy gradient direction $\tilde{\mathbf{g}} = \mathbf{F}_{\text{damp}}^{-1} \hat{\mathbf{g}}$.

**Step-by-Step Solution:**

*Step 1: Compute Outer Products $\mathbf{G}_i = \mathbf{g}_i \mathbf{g}_i^\top$:*
$$\mathbf{G}_1 = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix} \begin{bmatrix} 1.0 & 0.5 \end{bmatrix} = \begin{bmatrix} 1.0000 & 0.5000 \\ 0.5000 & 0.2500 \end{bmatrix}$$
$$\mathbf{G}_2 = \begin{bmatrix} -0.5 \\ 1.0 \end{bmatrix} \begin{bmatrix} -0.5 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.2500 & -0.5000 \\ -0.5000 & 1.0000 \end{bmatrix}$$
$$\mathbf{G}_3 = \begin{bmatrix} 0.8 \\ -0.4 \end{bmatrix} \begin{bmatrix} 0.8 & -0.4 \end{bmatrix} = \begin{bmatrix} 0.6400 & -0.3200 \\ -0.3200 & 0.1600 \end{bmatrix}$$
$$\mathbf{G}_4 = \begin{bmatrix} -0.6 \\ -0.8 \end{bmatrix} \begin{bmatrix} -0.6 & -0.8 \end{bmatrix} = \begin{bmatrix} 0.3600 & 0.4800 \\ 0.4800 & 0.6400 \end{bmatrix}$$

*Step 2: Sum Outer Products and Divide by $N = 4$:*
Summing component-wise:
$$\sum_{i=1}^4 \mathbf{G}_i = \begin{bmatrix} 1.0000 + 0.2500 + 0.6400 + 0.3600 & 0.5000 - 0.5000 - 0.3200 + 0.4800 \\ 0.5000 - 0.5000 - 0.3200 + 0.4800 & 0.2500 + 1.0000 + 0.1600 + 0.6400 \end{bmatrix} = \begin{bmatrix} 2.2500 & 0.1600 \\ 0.1600 & 2.0500 \end{bmatrix}$$
Dividing by $N = 4$:
$$\mathbf{F}_{\text{emp}} = \frac{1}{4} \begin{bmatrix} 2.2500 & 0.1600 \\ 0.1600 & 2.0500 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.562500 & 0.040000 \\ 0.040000 & 0.512500 \end{bmatrix}}$$

*Step 3: Add Damping Factor $\lambda = 0.0010$:*
$$\mathbf{F}_{\text{damp}} = \begin{bmatrix} 0.562500 & 0.040000 \\ 0.040000 & 0.512500 \end{bmatrix} + \begin{bmatrix} 0.001000 & 0.000000 \\ 0.000000 & 0.001000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.563500 & 0.040000 \\ 0.040000 & 0.513500 \end{bmatrix}}$$

*Step 4: Compute Determinant:*
$$\det(\mathbf{F}_{\text{damp}}) = a d - b c = (0.563500)(0.513500) - (0.040000)(0.040000)$$
$$(0.563500)(0.513500) = 0.28935725$$
$$(0.040000)(0.040000) = 0.00160000$$
$$\det(\mathbf{F}_{\text{damp}}) = 0.28935725 - 0.00160000 = \mathbf{0.28775725}$$

*Step 5: Compute Inverse Metric Tensor via $2 \times 2$ Formula:*
$$\mathbf{F}_{\text{damp}}^{-1} = \frac{1}{0.28775725} \begin{bmatrix} 0.513500 & -0.040000 \\ -0.040000 & 0.563500 \end{bmatrix}$$
Evaluating each entry:
$$\frac{0.513500}{0.28775725} \approx 1.78449023$$
$$\frac{-0.040000}{0.28775725} \approx -0.13900605$$
$$\frac{0.563500}{0.28775725} \approx 1.95824779$$
$$\mathbf{F}_{\text{damp}}^{-1} = \mathbf{\begin{bmatrix} 1.78449023 & -0.13900605 \\ -0.13900605 & 1.95824779 \end{bmatrix}}$$

*Step 6: Compute Natural Policy Gradient Step Direction:*
$$\tilde{\mathbf{g}} = \mathbf{F}_{\text{damp}}^{-1} \hat{\mathbf{g}} = \begin{bmatrix} 1.78449023 & -0.13900605 \\ -0.13900605 & 1.95824779 \end{bmatrix} \begin{bmatrix} 0.500000 \\ -0.200000 \end{bmatrix}$$
$$\tilde{g}_0 = (1.78449023)(0.500000) + (-0.13900605)(-0.200000) = 0.89224512 + 0.02780121 = \mathbf{+0.92004633}$$
$$\tilde{g}_1 = (-0.13900605)(0.500000) + (1.95824779)(-0.200000) = -0.06950302 - 0.39164956 = \mathbf{-0.46115258}$$
$$\tilde{\mathbf{g}} = \mathbf{\begin{bmatrix} +0.92004633 \\ -0.46115258 \end{bmatrix}} \quad \blacksquare$$

---

### Illustration 5: Continuous Action Gaussian Policy Natural Gradient & Coordinate Invariance
**Problem:**
Consider a 2D continuous action policy parameterized by mean vector $\mu \in \mathbb{R}^2$ with fixed covariance matrix $\Sigma$:
$$\pi_{\mu}(\mathbf{a}) = \frac{1}{(2\pi) |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{a} - \mu)^\top \Sigma^{-1} (\mathbf{a} - \mu) \right)$$
where:
$$\Sigma = \begin{bmatrix} 4.0000 & 1.0000 \\ 1.0000 & 2.0000 \end{bmatrix}$$
The observed policy gradient with respect to mean $\mu$ is $\mathbf{g} = [1.5000, -0.5000]^\top$, and the trust-region bound is $\epsilon = 0.0500$.
1. Prove that the Fisher Information Matrix with respect to the mean vector $\mu$ is exactly the precision matrix $\mathbf{F}_{\mu} = \Sigma^{-1}$, and therefore the natural gradient direction is $\tilde{\mathbf{g}} = \Sigma \mathbf{g}$.
2. Compute the natural policy gradient update $\Delta \mu^*$ in the original action coordinates.
3. Consider a coordinate transformation that scales the first action dimension by a factor of 10: $\tilde{\mathbf{a}} = \mathbf{C} \mathbf{a}$ where $\mathbf{C} = \operatorname{diag}(10.0, 1.0)$.
   Compute the scaled covariance $\tilde{\Sigma}$, scaled gradient $\mathbf{g}_{\text{scaled}}$, and scaled natural gradient update $\Delta \tilde{\mu}^*$.
   Show that mapping $\Delta \tilde{\mu}^*$ back to the original space via $\mathbf{C}^{-1} \Delta \tilde{\mu}^*$ reproduces $\Delta \mu^*$ exactly.
4. Contrast this with the vanilla policy gradient update to demonstrate vanilla failure under scaling.

**Step-by-Step Solution:**

*Step 1: Analytical Fisher Matrix for Gaussian Mean:*
Log-likelihood function:
$$\log \pi_{\mu}(\mathbf{a}) = -\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2} (\mathbf{a} - \mu)^\top \Sigma^{-1} (\mathbf{a} - \mu)$$
Compute the gradient with respect to $\mu$:
$$\nabla_{\mu} \log \pi_{\mu}(\mathbf{a}) = \Sigma^{-1} (\mathbf{a} - \mu)$$
Compute the Fisher Information Matrix:
$$\mathbf{F}_{\mu} = \mathbb{E}_{\mathbf{a} \sim \pi} \left[ (\nabla_{\mu} \log \pi)(\nabla_{\mu} \log \pi)^\top \right] = \mathbb{E}\left[ \Sigma^{-1} (\mathbf{a} - \mu) (\mathbf{a} - \mu)^\top \Sigma^{-1} \right]$$
Since $\Sigma^{-1}$ is constant, pull it outside the expectation:
$$\mathbf{F}_{\mu} = \Sigma^{-1} \left( \mathbb{E}\left[ (\mathbf{a} - \mu)(\mathbf{a} - \mu)^\top \right] \right) \Sigma^{-1}$$
Recognizing the inner expectation as the covariance matrix $\Sigma$:
$$\mathbf{F}_{\mu} = \Sigma^{-1} \Sigma \Sigma^{-1} = \mathbf{I} \Sigma^{-1} = \mathbf{\Sigma^{-1}}$$
Therefore, the inverse Fisher matrix is:
$$\mathbf{F}_{\mu}^{-1} = (\Sigma^{-1})^{-1} = \mathbf{\Sigma}$$
The unnormalized natural gradient direction is:
$$\tilde{\mathbf{g}}_{\mu} = \mathbf{F}_{\mu}^{-1} \mathbf{g} = \mathbf{\Sigma \mathbf{g}}$$
The natural gradient is simply the vanilla gradient smoothed and rotated by the action covariance matrix!

*Step 2: Natural Gradient Update in Original Coordinates:*
Compute the unnormalized natural gradient:
$$\tilde{\mathbf{g}} = \Sigma \mathbf{g} = \begin{bmatrix} 4.0000 & 1.0000 \\ 1.0000 & 2.0000 \end{bmatrix} \begin{bmatrix} 1.5000 \\ -0.5000 \end{bmatrix} = \begin{bmatrix} 4.0 \times 1.5 + 1.0 \times (-0.5) \\ 1.0 \times 1.5 + 2.0 \times (-0.5) \end{bmatrix} = \begin{bmatrix} 6.0000 - 0.5000 \\ 1.5000 - 1.0000 \end{bmatrix} = \mathbf{\begin{bmatrix} 5.5000 \\ 0.5000 \end{bmatrix}}$$
Compute curvature:
$$\mathbf{g}^\top \tilde{\mathbf{g}} = \mathbf{g}^\top \Sigma \mathbf{g} = 1.5000 \times 5.5000 + (-0.5000) \times 0.5000 = 8.2500 - 0.2500 = \mathbf{8.0000}$$
Compute step size $\beta$ for $\epsilon = 0.0500$:
$$\beta = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \Sigma \mathbf{g}}} = \sqrt{\frac{2 \times 0.0500}{8.0000}} = \sqrt{\frac{0.1000}{8.0000}} = \sqrt{0.012500} \approx \mathbf{0.1118034}$$
Compute the parameter update:
$$\Delta \mu^* = \beta \tilde{\mathbf{g}} = 0.1118034 \begin{bmatrix} 5.5000 \\ 0.5000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.6149187 \\ 0.0559017 \end{bmatrix}}$$

*Step 3: Natural Gradient Update in Transformed Coordinates ($\mathbf{C} = \operatorname{diag}(10.0, 1.0)$):*
In the scaled action space $\tilde{\mathbf{a}} = \mathbf{C} \mathbf{a}$, the mean is $\tilde{\mu} = \mathbf{C} \mu$, and the covariance matrix transforms as:
$$\tilde{\Sigma} = \mathbf{C} \Sigma \mathbf{C}^\top = \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 4.0 & 1.0 \\ 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 40.0 & 10.0 \\ 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 400.0000 & 10.0000 \\ 10.0000 & 2.0000 \end{bmatrix}}$$
The policy gradient transforms covariantly with $(\mathbf{C}^{-1})^\top = \mathbf{C}^{-1} = \operatorname{diag}(0.10, 1.00)$:
$$\mathbf{g}_{\text{scaled}} = \mathbf{C}^{-1} \mathbf{g} = \begin{bmatrix} 0.1000 & 0.0000 \\ 0.0000 & 1.0000 \end{bmatrix} \begin{bmatrix} 1.5000 \\ -0.5000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.1500 \\ -0.5000 \end{bmatrix}}$$
Compute the natural gradient direction in scaled coordinates:
$$\tilde{\mathbf{g}}_{\text{scaled}} = \tilde{\Sigma} \mathbf{g}_{\text{scaled}} = \begin{bmatrix} 400.0 & 10.0 \\ 10.0 & 2.0 \end{bmatrix} \begin{bmatrix} 0.1500 \\ -0.5000 \end{bmatrix} = \begin{bmatrix} 400.0 \times 0.15 + 10.0 \times (-0.5) \\ 10.0 \times 0.15 + 2.0 \times (-0.5) \end{bmatrix} = \begin{bmatrix} 60.0 - 5.0 \\ 1.5 - 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 55.0000 \\ 0.5000 \end{bmatrix}}$$
Notice that $\tilde{\mathbf{g}}_{\text{scaled}} = \mathbf{C} \tilde{\mathbf{g}}$:
$$\begin{bmatrix} 55.0000 \\ 0.5000 \end{bmatrix} = \begin{bmatrix} 10.0 \times 5.5000 \\ 1.0 \times 0.5000 \end{bmatrix} = \mathbf{C} \tilde{\mathbf{g}} \quad \checkmark$$
Compute the Riemannian curvature in scaled coordinates:
$$\mathbf{g}_{\text{scaled}}^\top \tilde{\mathbf{g}}_{\text{scaled}} = 0.1500 \times 55.0000 + (-0.5000) \times 0.5000 = 8.2500 - 0.2500 = \mathbf{8.0000}$$
The curvature is identical!
Compute step size:
$$\beta_{\text{scaled}} = \sqrt{\frac{2 \times 0.0500}{8.0000}} = \mathbf{0.1118034} = \beta$$
Compute update in scaled coordinates:
$$\Delta \tilde{\mu}^* = 0.1118034 \begin{bmatrix} 55.0000 \\ 0.5000 \end{bmatrix} = \mathbf{\begin{bmatrix} 6.1491869 \\ 0.0559017 \end{bmatrix}}$$
Now map $\Delta \tilde{\mu}^*$ back to the original physical coordinate system via $\mathbf{C}^{-1}$:
$$\mathbf{C}^{-1} \Delta \tilde{\mu}^* = \begin{bmatrix} 0.1000 & 0.0000 \\ 0.0000 & 1.0000 \end{bmatrix} \begin{bmatrix} 6.1491869 \\ 0.0559017 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.6149187 \\ 0.0559017 \end{bmatrix}} = \Delta \mu^* \quad \blacksquare$$

*Step 4: Contrast with Vanilla Policy Gradient Failure:*
Suppose a practitioner uses standard vanilla policy gradient with learning rate $\alpha = 0.1000$.
In the original coordinates:
$$\Delta \mu_{\text{vanilla}} = \alpha \mathbf{g} = 0.1000 \begin{bmatrix} 1.5000 \\ -0.5000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.1500 \\ -0.0500 \end{bmatrix}}$$
In the scaled coordinates, the vanilla gradient is $\mathbf{g}_{\text{scaled}} = [0.1500, -0.5000]^\top$.
The vanilla step in scaled space is:
$$\Delta \tilde{\mu}_{\text{vanilla}} = \alpha \mathbf{g}_{\text{scaled}} = 0.1000 \begin{bmatrix} 0.1500 \\ -0.5000 \end{bmatrix} = \begin{bmatrix} 0.0150 \\ -0.0500 \end{bmatrix}$$
Mapping this back to original physical action units:
$$\mathbf{C}^{-1} \Delta \tilde{\mu}_{\text{vanilla}} = \begin{bmatrix} 0.1000 & 0.0000 \\ 0.0000 & 1.0000 \end{bmatrix} \begin{bmatrix} 0.0150 \\ -0.0500 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.0015 \\ -0.0500 \end{bmatrix}} \ne \mathbf{\begin{bmatrix} 0.1500 \\ -0.0500 \end{bmatrix}}$$
The physical update along the first action dimension was suppressed by a factor of $100$ ($10^2$)!
Vanilla policy gradient is severely distorted by coordinate scaling, whereas Natural Policy Gradient produces the exact same physical probability distribution update. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. TRPO, K-FAC, and the Natural Gradient Family Tree
Natural Policy Gradients spawned a direct lineage of production-grade optimizers:
- **TRPO (Schulman et al., ICML 2015):** Scales NPG to deep neural networks by solving the constrained optimization using the Conjugate Gradient (CG) algorithm to compute $F^{-1}g$ in $O(d)$ time via Fisher-vector products, avoiding explicit Fisher matrix construction ($O(d^2)$ memory for $d \sim 10^7$ parameters).
- **ACKTR (Wu et al., NeurIPS 2017):** Approximates $F^{-1}$ using **Kronecker-Factored Approximate Curvature (K-FAC)**, factoring the Fisher matrix as $F \approx A \otimes G$ where $A$ is an input covariance and $G$ an output gradient covariance. Achieves 10–20× better sample efficiency than Adam on Atari with 3× faster wall-clock time.
- **PPO vs. NPG:** PPO's clipped surrogate objective is a first-order approximation to the natural gradient update: the clip $\epsilon \in [0.1, 0.3]$ provides an implicit trust region without the expensive CG solve, trading mathematical exactness for 10× computational speed.

### 2. Fisher-Rao Geometry in LLM Fine-Tuning
The Fisher information matrix $F$ governs the **geometry of the parameter space** for language models:
- **Natural Gradient RLHF:** During PPO-based RLHF, the natural gradient ensures parameter updates travel equal Riemannian distances in the KL-ball around $\pi_{\text{ref}}$, regardless of the local curvature of the log-likelihood surface. This directly prevents the "catastrophic forgetting" phenomenon where RLHF destroys the model's pre-trained capabilities.
- **Shampoo Optimizer (Gupta et al., Google, 2018):** A practical Fisher-preconditioned optimizer that maintains Kronecker-structured curvature matrices per layer, used to train Google's production language models with 2–3× better convergence rate than Adam.
- **FLAN, Tulu, Dolly fine-tuning:** All production instruction-following pipelines use Adam (an approximate natural gradient via adaptive diagonal Fisher) as the workhorse, with the NPG theory providing the mathematical justification for why adaptive learning rates are critical.

### 3. Riemannian Policy Optimization in Robotics & Control
- **Continuous locomotion (ANYmal, MIT Cheetah):** Natural gradient updates guarantee that each policy iteration moves the same "behavioral distance" in the space of trajectories, preventing aggressive updates from destabilizing leg coordination gaits that took millions of steps to converge.
- **Covariant Policy Search (Peters & Schaal, 2008):** Showed that NPG with compatible function approximation achieves monotonic policy improvement in continuous motor control — a theoretical guarantee that vanilla gradient ascent cannot provide.

---

## 8. Code Implementation & Verification

The accompanying Python script implements an exhaustive suite of 8 automated mathematical verification tests:
1. **Part 5 Hand Calculations Verification:** Confirms scores $\mathbf{s}_0, \mathbf{s}_1$, Fisher Matrix $\mathbf{F}$, damped inverse $\mathbf{F}_{\text{reg}}^{-1}$, natural gradient $\tilde{\mathbf{g}} = [+2.7778, -2.7778]^\top$, step size $\beta = 0.06000$, update $\Delta \theta^* = [+0.16667, -0.16667]^\top$, and exact KL divergence $\frac{1}{2}\Delta\theta^\top \mathbf{F}_{\text{reg}} \Delta\theta = 0.0100$ to machine precision.
2. **Kakade Invariance Theorem Test:** Verifies that under random non-singular coordinate transformations $\tilde{\theta} = \mathbf{M} \theta$, the natural gradient step satisfies $\mathbf{M}^{-1} \Delta \tilde{\theta}^* = \Delta \theta^*$ with zero coordinate distortion.
3. **Analytical vs. Monte Carlo Gaussian Fisher Matrix:** Validates analytical $\mathbf{F} = \operatorname{diag}(1/\sigma^2, 2/\sigma^2)$ against a $100,000$-sample empirical expectation.
4. **Illustration 1 Verification:** Verifies the 1D Gaussian policy natural gradient step ($\Delta \theta^* = [0.094281, -0.023570]^\top$) and exact trust-region satisfaction.
5. **Illustration 2 Verification:** Verifies the analytical 2-action softmax Fisher matrix $\mathbf{F} = \operatorname{diag}(\pi) - \pi\pi^\top = \begin{bmatrix} 0.1875 & -0.1875 \\ -0.1875 & 0.1875 \end{bmatrix}$ matches the expectation of outer product score vectors identically.
6. **Illustration 3 Verification:** Verifies the $4.9031\times$ speedup of Natural Policy Gradient over Vanilla Policy Gradient on an ill-conditioned plateau ($\kappa = 625$) under identical KL divergence $\epsilon = 0.0100$.
7. **Illustration 4 Verification:** Verifies batch sample outer products, empirical Fisher estimation ($N = 4$), Tikhonov damping ($\lambda = 10^{-3}$), analytical $2 \times 2$ determinant, and matrix inversion.
8. **Illustration 5 Verification:** Verifies the continuous Gaussian covariance-scaled natural direction ($\tilde{\mathbf{g}} = \Sigma\mathbf{g}$) and absolute coordinate invariance under diagonal scaling ($\mathbf{C} = \operatorname{diag}(10, 1)$), contrasting with the $100\times$ distortion suffered by vanilla gradients.

See implementation in:
[`11_reinforcement_learning/code/17_natural_policy_gradients.py`](./code/17_natural_policy_gradients.py)
