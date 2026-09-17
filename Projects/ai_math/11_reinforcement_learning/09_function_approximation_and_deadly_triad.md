# Module 11.9: Function Approximation & The Deadly Triad

---

## 1. Intuition & 101 Motivation

In real-world applications of reinforcement learning, the state space $\mathcal{S}$ is vastly too large to store in a lookup table:
- Backgammon has $\approx 10^{20}$ states.
- Chess has $\approx 10^{40}$ states.
- Go has $\approx 10^{170}$ states.
- A single $84 \times 84$ Atari grayscale frame with 256 pixel shades has $256^{7056} \approx 10^{16988}$ states.
- Continuous robotic control (joint angles, velocities) has an **uncountably infinite** state space $\mathcal{S} \subseteq \mathbb{R}^d$.

To scale RL, we must replace lookup tables with **parameterized function approximation**:
$$\hat{v}(s, \mathbf{w}) \approx V^\pi(s) \quad \text{or} \quad \hat{q}(s, a, \mathbf{w}) \approx Q^*(s, a)$$
where $\mathbf{w} \in \mathbb{R}^d$ is a weight vector whose dimension is vastly smaller than the state space ($d \ll |\mathcal{S}|$).

However, combining function approximation with reinforcement learning introduces one of the deepest theoretical perils in machine learning: **The Deadly Triad**.

```
                           THE DEADLY TRIAD
                           
                     Function Approximation
                     (Neural Nets, Linear features)
                                / \
                               /   \
                              /     \
                             /       \
                            /         \
              Bootstrapping ----------- Off-Policy Learning
            (TD targets, DP)           (Replay buffer, Q-learning)

   Any TWO elements: STABLE & CONVERGENT.
   All THREE combined: POTENTIALLY CATASTROPHIC DIVERGENCE (Weights -> Infinity).
```

When all three conditions are present, value estimates can blow up to $\pm \infty$ even on simple linear problems with zero rewards!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Mean Squared Value Error (MSVE)

Let $d^\pi(s)$ be the on-policy stationary distribution over states under policy $\pi$, such that $\sum_{s \in \mathcal{S}} d^\pi(s) = 1$ and $d^\pi(s) \ge 0$.
The quality of approximation is measured by the **Mean Squared Value Error (MSVE)**:

$$\overline{\text{VE}}(\mathbf{w}) \triangleq \sum_{s \in \mathcal{S}} d^\pi(s) \left[ V^\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2 = \left\| \mathbf{V}^\pi - \hat{\mathbf{v}}(\mathbf{w}) \right\|_{\mathbf{D}}^2$$

where $\mathbf{D} = \operatorname{diag}(d^\pi(s_1), \dots, d^\pi(s_n))$ defines a weighted Euclidean norm:
$$\|\mathbf{x}\|_{\mathbf{D}}^2 \triangleq \mathbf{x}^\top \mathbf{D} \mathbf{x}$$

If we had access to the true target $V^\pi(S_t)$, stochastic gradient descent (SGD) on $\overline{\text{VE}}(\mathbf{w})$ would yield the ideal update:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{1}{2} \alpha \nabla_\mathbf{w} \left[ V^\pi(S_t) - \hat{v}(S_t, \mathbf{w}) \right]^2 = \mathbf{w}_t + \alpha \left[ V^\pi(S_t) - \hat{v}(S_t, \mathbf{w}) \right] \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})$$

---

### 2.2 Semi-Gradient Methods: Why TD is NOT True Gradient Descent

In reality, the true value $V^\pi(S_t)$ is unknown. In TD learning, we substitute a bootstrapped target $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$.
The resulting update is:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w}_t)$$

> [!WARNING]
> **Semi-Gradient Warning:** This update is called a **semi-gradient method** because it ignores the dependence of the target $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$ on the parameter vector $\mathbf{w}$!
> If one attempted true gradient descent on the Bellman residual squared error $\frac{1}{2} (R + \gamma \hat{v}' - \hat{v})^2$, the gradient would be:
> $$\nabla_\mathbf{w} \frac{1}{2} \delta_t^2 = -\delta_t \left( \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w}) - \gamma \nabla_\mathbf{w} \hat{v}(S_{t+1}, \mathbf{w}) \right)$$
> True Bellman error minimization is known to converge to poor, non-optimal value functions (Baird's residual gradient dilemma). Semi-gradient methods converge to the Bellman fixed point, but **only under on-policy sampling**!

---

### 2.3 Linear Function Approximation & The Projection Operator $\Pi$

Consider a linear function approximator:
$$\hat{v}(s, \mathbf{w}) \triangleq \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)$$
where $\mathbf{x}(s) \in \mathbb{R}^d$ is the feature vector for state $s$. In matrix notation:
$$\hat{\mathbf{v}}(\mathbf{w}) = \mathbf{\Phi} \mathbf{w}$$
where $\mathbf{\Phi} \in \mathbb{R}^{|\mathcal{S}| \times d}$ is the feature matrix whose rows are $\mathbf{x}(s)^\top$.

#### The Projection Operator $\Pi$
Any vector $\mathbf{V} \in \mathbb{R}^{|\mathcal{S}|}$ can be orthogonally projected onto the column subspace of $\mathbf{\Phi}$ with respect to the weighted norm $\|\cdot\|_{\mathbf{D}}$:

$$\Pi \mathbf{V} \triangleq \mathbf{\Phi} \left( \mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi} \right)^{-1} \mathbf{\Phi}^\top \mathbf{D} \mathbf{V}$$

Notice that $\Pi$ is linear, idempotent ($\Pi^2 = \Pi$), and a non-expansion in the $\mathbf{D}$-norm:
$$\|\Pi \mathbf{V}_1 - \Pi \mathbf{V}_2\|_{\mathbf{D}} \le \|\mathbf{V}_1 - \mathbf{V}_2\|_{\mathbf{D}}$$

---

### 2.4 The Projected Bellman Equation & The TD Fixed Point

When linear semi-gradient TD(0) converges, its expected update vector must equal zero:
$$\mathbb{E} \left[ \mathbf{x}_t \left( R_{t+1} + \gamma \mathbf{x}_{t+1}^\top \mathbf{w} - \mathbf{x}_t^\top \mathbf{w} \right) \right] = \mathbf{0}$$

Expanding the expectation across the stationary distribution $\mathbf{D}$:
$$\mathbf{\Phi}^\top \mathbf{D} \left( \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{\Phi} \mathbf{w} - \mathbf{\Phi} \mathbf{w} \right) = \mathbf{0}$$
$$\underbrace{\mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{\Phi}}_{\mathbf{A}} \mathbf{w} = \underbrace{\mathbf{\Phi}^\top \mathbf{D} \mathbf{R}^\pi}_{\mathbf{b}}$$

This is equivalent to the **Projected Bellman Equation**:
$$\mathbf{\Phi} \mathbf{w}_{\text{TD}} = \Pi \mathcal{T}^\pi (\mathbf{\Phi} \mathbf{w}_{\text{TD}})$$

#### Theorem: Convergence of Linear On-Policy TD(0) (Tsitsiklis & Van Roy, 1997)
If data is sampled from the on-policy stationary distribution $d^\pi(s)$ of an ergodic Markov chain, the matrix $\mathbf{A} = \mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{\Phi}$ is **strictly positive definite**.
The compound operator $\Pi \mathcal{T}^\pi$ is a $\gamma$-contraction in $\|\cdot\|_{\mathbf{D}}$:
$$\|\Pi \mathcal{T}^\pi \mathbf{V}_1 - \Pi \mathcal{T}^\pi \mathbf{V}_2\|_{\mathbf{D}} \le \gamma \|\mathbf{V}_1 - \mathbf{V}_2\|_{\mathbf{D}}$$
Consequently, linear on-policy TD(0) converges almost surely to the unique fixed point $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1} \mathbf{b}$, and the resulting approximation error satisfies:
$$\|\mathbf{V}^\pi - \mathbf{\Phi} \mathbf{w}_{\text{TD}}\|_{\mathbf{D}} \le \frac{1}{\sqrt{1 - \gamma^2}} \min_\mathbf{w} \|\mathbf{V}^\pi - \mathbf{\Phi} \mathbf{w}\|_{\mathbf{D}}$$

---

### 2.5 The Deadly Triad & Off-Policy Divergence Proof

When sampling transitions **off-policy** from behavior distribution $\mu(s) \neq d^\pi(s)$, the weighting matrix becomes $\mathbf{M} = \operatorname{diag}(\mu(s))$.
The projection operator becomes $\Pi_\mu = \mathbf{\Phi}(\mathbf{\Phi}^\top \mathbf{M} \mathbf{\Phi})^{-1}\mathbf{\Phi}^\top \mathbf{M}$.

> [!CAUTION]
> **The Collapse of Contraction:** While $\mathcal{T}^\pi$ is a contraction in $\|\cdot\|_\infty$, it is NOT necessarily a contraction in $\|\cdot\|_\mathbf{M}$!
> The combined operator $\Pi_\mu \mathcal{T}^\pi$ can have spectral radius (maximum eigenvalue) **strictly greater than 1**:
> $$\rho(\Pi_\mu \mathcal{T}^\pi) > 1$$
> When this occurs, repeatedly applying the semi-gradient TD operator causes the weight vector $\mathbf{w}_t$ to expand exponentially to infinity:
> $$\lim_{t \to \infty} \|\mathbf{w}_t\| = +\infty$$

---

### 2.6 First-Principles Mathematical Derivations

#### Derivation 11.9.1: Positive Definiteness of the TD Matrix and On-Policy Contraction Proof

```
====================================================================================================
DERIVATION 11.9.1: Positive Definiteness of TD Matrix A and On-Policy Contraction
====================================================================================================
Problem Statement:
In linear TD(0) with on-policy stationary distribution D = diag(d^π), prove that:
1. The transition matrix P^π is a γ-contraction with respect to the weighted norm ‖·‖_D:
       ‖P^π V‖_D ≤ ‖V‖_D   ∀ V ∈ ℝ^{|𝒮|}
2. The compound projected Bellman operator Π 𝒯^π is a strict γ-contraction in ‖·‖_D:
       ‖Π 𝒯^π U - Π 𝒯^π V‖_D ≤ γ ‖U - V‖_D
3. The expected TD matrix A = Φ^⊤ D (I - γ P^π) Φ is strictly positive definite:
       y^⊤ A y > 0   ∀ y ∈ ℝ^d, y ≠ 0
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Stationary Distribution:** $\mathbf{d}^\top \mathbf{P} = \mathbf{d}^\top$, with strictly positive entries $d(s) > 0$ for all $s \in \mathcal{S}$. Thus $\mathbf{D} = \operatorname{diag}(\mathbf{d})$ is positive definite.
2. **Full Column Rank:** The feature matrix $\mathbf{\Phi} \in \mathbb{R}^{|\mathcal{S}| \times d}$ has full column rank $d$, where $d \le |\mathcal{S}|$.
3. **Discounting:** $0 \le \gamma < 1$.

**2. Underlying Intuition:**
The stationary distribution condition $\mathbf{d}^\top \mathbf{P} = \mathbf{d}^\top$ means that probability mass is conserved under transitions. In the Hilbert space weighted by $\mathbf{D}$, this conservation property ensures that the Markov transition matrix $\mathbf{P}$ acts as a non-expansion ($\|\mathbf{P}\|_{\mathbf{D}} \le 1$). Multiplying by $\gamma < 1$ turns it into a strict contraction. Because the orthogonal projection $\Pi$ is also a non-expansion, their composition is a strict contraction, forcing the steady-state governing matrix $\mathbf{A}$ to be strictly positive definite.

**3. End-to-End Algebraic Derivation:**

*Step 1: Prove that $\mathbf{P}$ is a non-expansion in $\|\cdot\|_{\mathbf{D}}$.*
For any vector $\mathbf{V} \in \mathbb{R}^{|\mathcal{S}|}$, evaluate $\|\mathbf{P} \mathbf{V}\|_{\mathbf{D}}^2$:
$$\|\mathbf{P} \mathbf{V}\|_{\mathbf{D}}^2 = \sum_{s \in \mathcal{S}} d(s) \left( \sum_{s' \in \mathcal{S}} P(s' \mid s) V(s') \right)^2$$
Since $\sum_{s'} P(s' \mid s) = 1$ and $P(s' \mid s) \ge 0$, Jensen's inequality for the convex function $f(x) = x^2$ yields:
$$\left( \sum_{s' \in \mathcal{S}} P(s' \mid s) V(s') \right)^2 \le \sum_{s' \in \mathcal{S}} P(s' \mid s) V(s')^2$$
Substituting this inequality back into the sum:
$$\|\mathbf{P} \mathbf{V}\|_{\mathbf{D}}^2 \le \sum_{s \in \mathcal{S}} d(s) \sum_{s' \in \mathcal{S}} P(s' \mid s) V(s')^2 = \sum_{s' \in \mathcal{S}} V(s')^2 \underbrace{\sum_{s \in \mathcal{S}} d(s) P(s' \mid s)}_{d(s') \text{ by stationarity}}$$
$$= \sum_{s' \in \mathcal{S}} d(s') V(s')^2 = \|\mathbf{V}\|_{\mathbf{D}}^2$$
Taking square roots on both sides:
$$\|\mathbf{P} \mathbf{V}\|_{\mathbf{D}} \le \|\mathbf{V}\|_{\mathbf{D}} \quad \forall \mathbf{V} \in \mathbb{R}^{|\mathcal{S}|}$$

*Step 2: Prove that $\Pi \mathcal{T}^\pi$ is a strict $\gamma$-contraction.*
Recall that $\mathcal{T}^\pi \mathbf{U} - \mathcal{T}^\pi \mathbf{V} = \gamma \mathbf{P} (\mathbf{U} - \mathbf{V})$.
Using the non-expansion of $\mathbf{P}$:
$$\|\mathcal{T}^\pi \mathbf{U} - \mathcal{T}^\pi \mathbf{V}\|_{\mathbf{D}} = \gamma \|\mathbf{P} (\mathbf{U} - \mathbf{V})\|_{\mathbf{D}} \le \gamma \|\mathbf{U} - \mathbf{V}\|_{\mathbf{D}}$$
The projection operator $\Pi = \mathbf{\Phi} (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \mathbf{D}$ is an orthogonal projection onto the subspace $\operatorname{span}(\mathbf{\Phi})$ under the inner product $\langle \mathbf{x}, \mathbf{y} \rangle_{\mathbf{D}} = \mathbf{x}^\top \mathbf{D} \mathbf{y}$.
By standard Hilbert space geometry, orthogonal projections are non-expansions:
$$\|\Pi \mathbf{x}\|_{\mathbf{D}} \le \|\mathbf{x}\|_{\mathbf{D}} \quad \forall \mathbf{x} \in \mathbb{R}^{|\mathcal{S}|}$$
Composing $\Pi$ with $\mathcal{T}^\pi$:
$$\|\Pi \mathcal{T}^\pi \mathbf{U} - \Pi \mathcal{T}^\pi \mathbf{V}\|_{\mathbf{D}} \le \|\mathcal{T}^\pi \mathbf{U} - \mathcal{T}^\pi \mathbf{V}\|_{\mathbf{D}} \le \gamma \|\mathbf{U} - \mathbf{V}\|_{\mathbf{D}}$$
Because $\gamma < 1$, $\Pi \mathcal{T}^\pi$ is a strict $\gamma$-contraction in $\|\cdot\|_{\mathbf{D}}$.

*Step 3: Prove positive definiteness of $\mathbf{A} = \mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}) \mathbf{\Phi}$.*
Let $\mathbf{y} \in \mathbb{R}^d$ be any non-zero vector, and define $\mathbf{v} \triangleq \mathbf{\Phi} \mathbf{y} \in \mathbb{R}^{|\mathcal{S}|}$.
Because $\mathbf{\Phi}$ has full column rank and $\mathbf{y} \neq \mathbf{0}$, we have $\mathbf{v} \neq \mathbf{0}$, which implies $\|\mathbf{v}\|_{\mathbf{D}} > 0$.
Now consider the quadratic form:
$$\mathbf{y}^\top \mathbf{A} \mathbf{y} = \mathbf{y}^\top \mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}) \mathbf{\Phi} \mathbf{y} = \mathbf{v}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}) \mathbf{v} = \mathbf{v}^\top \mathbf{D} \mathbf{v} - \gamma \mathbf{v}^\top \mathbf{D} \mathbf{P} \mathbf{v}$$
Note that $\mathbf{v}^\top \mathbf{D} \mathbf{v} = \|\mathbf{v}\|_{\mathbf{D}}^2$.
Applying the Cauchy-Schwarz inequality with respect to the $\mathbf{D}$-inner product:
$$|\mathbf{v}^\top \mathbf{D} \mathbf{P} \mathbf{v}| = |\langle \mathbf{v}, \mathbf{P} \mathbf{v} \rangle_{\mathbf{D}}| \le \|\mathbf{v}\|_{\mathbf{D}} \|\mathbf{P} \mathbf{v}\|_{\mathbf{D}}$$
Using the non-expansion property $\|\mathbf{P} \mathbf{v}\|_{\mathbf{D}} \le \|\mathbf{v}\|_{\mathbf{D}}$ proven in Step 1:
$$\mathbf{v}^\top \mathbf{D} \mathbf{P} \mathbf{v} \le \|\mathbf{v}\|_{\mathbf{D}}^2$$
Therefore:
$$\mathbf{y}^\top \mathbf{A} \mathbf{y} = \|\mathbf{v}\|_{\mathbf{D}}^2 - \gamma \mathbf{v}^\top \mathbf{D} \mathbf{P} \mathbf{v} \ge \|\mathbf{v}\|_{\mathbf{D}}^2 - \gamma \|\mathbf{v}\|_{\mathbf{D}}^2 = (1 - \gamma) \|\mathbf{v}\|_{\mathbf{D}}^2$$
Since $\gamma < 1$ and $\|\mathbf{v}\|_{\mathbf{D}} > 0$:
$$\mathbf{y}^\top \mathbf{A} \mathbf{y} \ge (1 - \gamma) \|\mathbf{v}\|_{\mathbf{D}}^2 > 0$$
Thus, $\mathbf{A}$ is strictly positive definite. All its eigenvalues have strictly positive real parts, guaranteeing that $\mathbf{A}^{-1}$ exists and linear on-policy TD(0) converges unconditionally! $\blacksquare$

---

#### Derivation 11.9.2: The Tsitsiklis-Van Roy Error Bound for Linear TD Fixed Point

```
====================================================================================================
DERIVATION 11.9.2: Tsitsiklis-Van Roy Bound on Approximation Error
====================================================================================================
Problem Statement:
Let V^π be the true value function and let V_TD = Φ w_TD be the fixed point of linear on-policy TD(0):
    V_TD = Π 𝒯^π V_TD
Prove that the approximation error of the TD solution satisfies:
    ‖V^π - V_TD‖_D ≤ (1 / √(1 - γ^2)) min_w ‖V^π - Φ w‖_D
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Hilbert Space Geometry:** The state space has inner product $\langle \mathbf{U}, \mathbf{V} \rangle_{\mathbf{D}} = \mathbf{U}^\top \mathbf{D} \mathbf{V}$ and induced norm $\|\mathbf{V}\|_{\mathbf{D}} = \sqrt{\langle \mathbf{V}, \mathbf{V} \rangle_{\mathbf{D}}}$.
2. **Orthogonal Projection:** $\Pi$ projects onto the closed linear subspace $\mathcal{M} = \{\mathbf{\Phi} \mathbf{w} \mid \mathbf{w} \in \mathbb{R}^d\}$.
3. **Fixed Point Equation:** $\mathbf{V}_{\text{TD}} = \Pi \mathcal{T}^\pi \mathbf{V}_{\text{TD}}$.
4. **Bellman Fixed Point:** $\mathcal{T}^\pi \mathbf{V}^\pi = \mathbf{V}^\pi$.

**2. Underlying Intuition:**
The best possible linear approximation is the orthogonal projection $\Pi \mathbf{V}^\pi$. The TD fixed point $\mathbf{V}_{\text{TD}}$ does not minimize the distance to $\mathbf{V}^\pi$ directly; instead, it matches its own projected Bellman update. By applying the Pythagorean theorem to decompose the total error into an orthogonal component (inherent representational limit) and a tangential component (bootstrapping distortion), we obtain an upper bound scaling as $\frac{1}{\sqrt{1 - \gamma^2}}$.

**3. End-to-End Algebraic Derivation:**

*Step 1: Orthogonal decomposition of the error vector.*
Decompose the error vector $\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}$:
$$\mathbf{V}^\pi - \mathbf{V}_{\text{TD}} = (\mathbf{V}^\pi - \Pi \mathbf{V}^\pi) + (\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}})$$
Notice that:
- $\mathbf{V}^\pi - \Pi \mathbf{V}^\pi$ is orthogonal to the subspace $\mathcal{M}$ by definition of orthogonal projection: $\langle \mathbf{V}^\pi - \Pi \mathbf{V}^\pi, \mathbf{m} \rangle_{\mathbf{D}} = 0$ for all $\mathbf{m} \in \mathcal{M}$.
- Both $\Pi \mathbf{V}^\pi$ and $\mathbf{V}_{\text{TD}}$ lie inside $\mathcal{M}$, so their difference $\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}} \in \mathcal{M}$.

*Step 2: Apply the Pythagorean Theorem in the $\mathbf{D}$-inner product space.*
Because the two terms are orthogonal:
$$\|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2 = \|\mathbf{V}^\pi - \Pi \mathbf{V}^\pi\|_{\mathbf{D}}^2 + \|\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2 \quad \text{(Equation A)}$$

*Step 3: Bound the tangential error $\|\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}$.*
Using the fixed point identities $\mathbf{V}_{\text{TD}} = \Pi \mathcal{T}^\pi \mathbf{V}_{\text{TD}}$ and $\Pi \mathbf{V}^\pi = \Pi \mathcal{T}^\pi \mathbf{V}^\pi$:
$$\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}} = \Pi \mathcal{T}^\pi \mathbf{V}^\pi - \Pi \mathcal{T}^\pi \mathbf{V}_{\text{TD}} = \Pi (\mathcal{T}^\pi \mathbf{V}^\pi - \mathcal{T}^\pi \mathbf{V}_{\text{TD}})$$
Taking the norm and using the non-expansion of $\Pi$ ($\|\Pi \mathbf{x}\|_{\mathbf{D}} \le \|\mathbf{x}\|_{\mathbf{D}}$):
$$\|\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}} \le \|\mathcal{T}^\pi \mathbf{V}^\pi - \mathcal{T}^\pi \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}$$
Using the $\gamma$-contraction of $\mathcal{T}^\pi$ in $\|\cdot\|_{\mathbf{D}}$ from Derivation 11.9.1:
$$\|\mathcal{T}^\pi \mathbf{V}^\pi - \mathcal{T}^\pi \mathbf{V}_{\text{TD}}\|_{\mathbf{D}} \le \gamma \|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}$$
Therefore:
$$\|\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}} \le \gamma \|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}$$
Squaring both sides:
$$\|\Pi \mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2 \le \gamma^2 \|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2$$

*Step 4: Substitute the bound back into Equation A.*
$$\|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2 \le \|\mathbf{V}^\pi - \Pi \mathbf{V}^\pi\|_{\mathbf{D}}^2 + \gamma^2 \|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2$$
Subtracting $\gamma^2 \|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2$ from both sides:
$$(1 - \gamma^2) \|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}}^2 \le \|\mathbf{V}^\pi - \Pi \mathbf{V}^\pi\|_{\mathbf{D}}^2$$
Dividing by $1 - \gamma^2 > 0$ and taking square roots:
$$\|\mathbf{V}^\pi - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}} \le \frac{1}{\sqrt{1 - \gamma^2}} \|\mathbf{V}^\pi - \Pi \mathbf{V}^\pi\|_{\mathbf{D}}$$

*Step 5: Connection to the optimal projection.*
By the Hilbert projection theorem, the orthogonal projection minimizes the Euclidean distance over the entire subspace $\mathcal{M}$:
$$\|\mathbf{V}^\pi - \Pi \mathbf{V}^\pi\|_{\mathbf{D}} = \min_{\mathbf{w} \in \mathbb{R}^d} \|\mathbf{V}^\pi - \mathbf{\Phi} \mathbf{w}\|_{\mathbf{D}}$$
Thus:
$$\|\mathbf{V}^\pi - \mathbf{\Phi} \mathbf{w}_{\text{TD}}\|_{\mathbf{D}} \le \frac{1}{\sqrt{1 - \gamma^2}} \min_{\mathbf{w}} \|\mathbf{V}^\pi - \mathbf{\Phi} \mathbf{w}\|_{\mathbf{D}} \quad \blacksquare$$

---

#### Derivation 11.9.3: True Gradient Descent on Projected Bellman Error (GTD2 / TDC)

```
====================================================================================================
DERIVATION 11.9.3: Closed-Form MSPBE Objective and Gradient TD (GTD2 & TDC)
====================================================================================================
Problem Statement:
Derive the Mean Squared Projected Bellman Error (MSPBE) objective function:
    MSPBE(w) ≜ ‖Π (𝒯^π Φ w - Φ w)‖_D^2
Prove that its gradient can be written as:
    ∇_w MSPBE(w) = 2 Φ^⊤ D (I - γ P^π)^⊤ Φ (Φ^⊤ D Φ)^{-1} Φ^⊤ D (Φ w - 𝒯^π Φ w)
and derive the two-timescale stochastic gradient descent algorithms GTD2 and TDC that eliminate
the double-sampling problem.
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Linear Function Approximation:** $\hat{\mathbf{v}}(\mathbf{w}) = \mathbf{\Phi} \mathbf{w}$.
2. **Feature Covariance Invertibility:** $\mathbf{C} \triangleq \mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi} = \mathbb{E}[\mathbf{x}_t \mathbf{x}_t^\top]$ is non-singular and positive definite.
3. **Bellman Residual Vector:** $\mathbf{\delta}_{\mathbf{w}} \triangleq \mathcal{T}^\pi (\mathbf{\Phi} \mathbf{w}) - \mathbf{\Phi} \mathbf{w} = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{\Phi} \mathbf{w} - \mathbf{\Phi} \mathbf{w}$.

**2. Underlying Intuition:**
Direct minimization of the Bellman Error $\|\mathbf{\delta}_{\mathbf{w}}\|_{\mathbf{D}}^2$ fails because its gradient $\mathbb{E}[\mathbf{\delta}_t \nabla \hat{v}]$ requires two independent transitions from the same state (the double-sampling problem). By projecting the Bellman error onto the feature subspace FIRST, the objective becomes a function of expected values $\mathbb{E}[\delta_t \mathbf{x}_t]$. We can decouple the product of expectations using a secondary linear estimator $\mathbf{v} \in \mathbb{R}^d$ running on a faster learning rate, achieving true $O(d)$ SGD.

**3. End-to-End Algebraic Derivation:**

*Step 1: Express MSPBE in closed matrix form.*
Recall the orthogonal projection operator $\Pi = \mathbf{\Phi} (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \mathbf{D}$.
The projected Bellman error vector is:
$$\Pi \mathbf{\delta}_{\mathbf{w}} = \mathbf{\Phi} (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}}$$
Now compute the squared $\mathbf{D}$-norm:
$$\text{MSPBE}(\mathbf{w}) = \|\Pi \mathbf{\delta}_{\mathbf{w}}\|_{\mathbf{D}}^2 = (\Pi \mathbf{\delta}_{\mathbf{w}})^\top \mathbf{D} (\Pi \mathbf{\delta}_{\mathbf{w}})$$
Substituting the definition of $\Pi \mathbf{\delta}_{\mathbf{w}}$:
$$= \left[ \mathbf{\delta}_{\mathbf{w}}^\top \mathbf{D} \mathbf{\Phi} (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \right] \mathbf{D} \left[ \mathbf{\Phi} (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}} \right]$$
Notice that $\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi} (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} = \mathbf{I}$.
Therefore:
$$\text{MSPBE}(\mathbf{w}) = \left( \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}} \right)^\top \left( \mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi} \right)^{-1} \left( \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}} \right)$$
Define the expected TD update vector $\mathbf{b}_{\mathbf{w}} \triangleq \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}} = \mathbb{E}[\delta_t(\mathbf{w}) \mathbf{x}_t]$ and covariance $\mathbf{C} \triangleq \mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi}$.
Then:
$$\text{MSPBE}(\mathbf{w}) = \mathbf{b}_{\mathbf{w}}^\top \mathbf{C}^{-1} \mathbf{b}_{\mathbf{w}}$$

*Step 2: Differentiating MSPBE with respect to $\mathbf{w}$.*
Notice that $\mathbf{b}_{\mathbf{w}} = \mathbf{\Phi}^\top \mathbf{D} (\mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{\Phi} \mathbf{w} - \mathbf{\Phi} \mathbf{w}) = \mathbf{\Phi}^\top \mathbf{D} \mathbf{R}^\pi - \mathbf{A} \mathbf{w}$, where $\mathbf{A} = \mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{\Phi}$.
Taking the gradient with respect to $\mathbf{w}$:
$$\nabla_{\mathbf{w}} \mathbf{b}_{\mathbf{w}} = -\mathbf{A}^\top = -\mathbf{\Phi}^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^\top \mathbf{D} \mathbf{\Phi}$$
Applying the matrix calculus chain rule to the quadratic form $\mathbf{b}_{\mathbf{w}}^\top \mathbf{C}^{-1} \mathbf{b}_{\mathbf{w}}$:
$$-\frac{1}{2} \nabla_{\mathbf{w}} \text{MSPBE}(\mathbf{w}) = -(\nabla_{\mathbf{w}} \mathbf{b}_{\mathbf{w}})^\top \mathbf{C}^{-1} \mathbf{b}_{\mathbf{w}} = \mathbf{A}^\top \mathbf{C}^{-1} \mathbf{b}_{\mathbf{w}}$$
Expanding $\mathbf{A}^\top$:
$$-\frac{1}{2} \nabla_{\mathbf{w}} \text{MSPBE}(\mathbf{w}) = \mathbf{\Phi}^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^\top \mathbf{D} \mathbf{\Phi} \left( \mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi} \right)^{-1} \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}}$$

*Step 3: Introducing the secondary weight vector $\mathbf{v}$.*
Computing this gradient online via single transitions is impossible because the product of expectations requires evaluating $\mathbf{A}^\top$ and $\mathbf{C}^{-1} \mathbf{b}_{\mathbf{w}}$ simultaneously.
Sutton et al. resolved this by defining a secondary parameter vector $\mathbf{v} \in \mathbb{R}^d$ to approximate:
$$\mathbf{v} \approx \mathbf{C}^{-1} \mathbf{b}_{\mathbf{w}} = (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}}$$
Notice that $\mathbf{v}$ is the minimizer of the standard linear regression problem:
$$\min_{\mathbf{v}} \mathbb{E} \left[ \left( \delta_t(\mathbf{w}) - \mathbf{x}_t^\top \mathbf{v} \right)^2 \right]$$
The SGD update for $\mathbf{v}$ with step size $\beta$ is:
$$\mathbf{v}_{t+1} = \mathbf{v}_t + \beta \left( \delta_t - \mathbf{x}_t^\top \mathbf{v}_t \right) \mathbf{x}_t$$

*Step 4: Deriving the GTD2 and TDC Weight Updates.*
With $\mathbf{v}_t$ tracking $\mathbf{C}^{-1} \mathbf{b}_{\mathbf{w}}$, the true negative gradient becomes:
$$-\frac{1}{2} \nabla_{\mathbf{w}} \text{MSPBE}(\mathbf{w}) \approx \mathbf{A}^\top \mathbf{v} = \mathbf{\Phi}^\top (\mathbf{I} - \gamma \mathbf{P}^\pi)^\top \mathbf{D} \mathbf{\Phi} \mathbf{v} = \mathbb{E} \left[ (\mathbf{x}_t - \gamma \mathbf{x}_{t+1}) (\mathbf{x}_t^\top \mathbf{v}) \right]$$

1. **GTD2 Algorithm:**
   Directly sampling this expected gradient yields:
   $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left( \mathbf{x}_t - \gamma \mathbf{x}_{t+1} \right) \left( \mathbf{x}_t^\top \mathbf{v}_t \right)$$

2. **TDC (TD with Gradient Correction) Algorithm:**
   Rewrite $\mathbf{A}^\top \mathbf{v} = \mathbf{\Phi}^\top \mathbf{D} \mathbf{\delta}_{\mathbf{w}} - \gamma \mathbf{\Phi}^\top (\mathbf{P}^\pi)^\top \mathbf{D} \mathbf{\Phi} \mathbf{v} + \dots$:
   $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \mathbf{x}_t - \alpha \gamma \mathbf{x}_{t+1} \left( \mathbf{x}_t^\top \mathbf{v}_t \right)$$
   Here, $\alpha \delta_t \mathbf{x}_t$ is the standard semi-gradient TD update, and $-\alpha \gamma \mathbf{x}_{t+1} (\mathbf{x}_t^\top \mathbf{v}_t)$ is the exact **gradient correction term** that stabilizes off-policy learning! $\blacksquare$

---

## 3. Geometric Interpretation: Warping of the Projection Manifold

In the $|\mathcal{S}|$-dimensional state space:
- The realizable function approximations span a low-dimensional flat hyperplane $\mathcal{H} = \operatorname{span}(\mathbf{\Phi})$.
- When sampling **on-policy**, the projection angle is orthogonal with respect to the stationary metric $\mathbf{D}$. The contractive pull of $\mathcal{T}^\pi$ ($\gamma < 1$) overcomes the non-expansion of $\Pi$, ensuring the composition $\Pi \mathcal{T}^\pi$ pulls any point inside towards the unique fixed point.
- When sampling **off-policy**, the metric tensor is distorted by $\mathbf{M} \neq \mathbf{D}$. The projection operator $\Pi_\mu$ acts at an **oblique, acute angle** relative to the Bellman operator flow. The oblique projection amplifies the vector magnitude by more than $1/\gamma$, causing successive iterates to spiral outward unboundedly!

```
                  V-Space (R^|S|)
                       ^
                       |              /  Oblique Projection (Off-policy)
                       |    T(V)     /   amplifies vector magnitude!
                       |      * ----+--> Pi_mu(T(V))  [||Pi T|| > 1]
                       |     /     /
                       |    /     /
                       |   * V   /    Hyperplane spanned by Phi w
                       |  /     /    /
                       | /     /    /
                       +------+----+------------------->
                              Orthogonal Projection (On-policy)
                              ||Pi T|| <= gamma < 1 (Stable)
```

---

## 4. Real-World Analogy: The Acoustic Feedback Loop (Larsen Effect)

Think of the Deadly Triad as a **screeching PA system at a rock concert**:
1. **Bootstrapping (The Microphone):** The microphone picks up sound and feeds it back into the audio processor.
2. **Function Approximation (The Graphic Equalizer / Amplifier):** The amplifier generalizes and boosts certain frequency bands with high gain.
3. **Off-Policy Operation (Aiming the Mic at the Speaker):** Instead of pointing the microphone at the singer (on-policy target distribution), you point it directly at the monitor speaker (off-policy mismatch).

If any one component is absent:
- No microphone (no bootstrapping $\implies$ Monte Carlo): Sound plays cleanly.
- No amplifier (lookup table): Gain is $\le 1$, no feedback loop.
- Mic pointed at singer (on-policy): Gain is stable and contained.

Combine all three, and an infinitesimal whisper creates an explosive, ear-splitting screech that blows the speakers ($\|\mathbf{w}\| \to \infty$)!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: Tsitsiklis & Van Roy 2-State Divergence Example
We consider the simplest possible system that diverges:
- **Two States:** $S_1$ and $S_2$.
- **Single scalar parameter:** $w \in \mathbb{R}$ ($d = 1$).
- **Features:** $\phi(S_1) = 1.0000, \quad \phi(S_2) = 2.0000$.
  So value approximations are:
  $$\hat{v}(S_1, w) = 1.0 \times w = w, \quad \hat{v}(S_2, w) = 2.0 \times w = 2w$$
- **Transition Dynamics:** From $S_1$, the agent **always** transitions to $S_2$ under target policy $\pi$.
- **Reward:** $R = 0.0000$ everywhere.
- **Discount factor:** $\gamma = 0.9000$.
- **Off-Policy Sampling Distribution:** The behavior policy samples transitions **only from state $S_1$** (state $S_2$ is never sampled as a starting state, so $\mu(S_1) = 1.0, \mu(S_2) = 0.0$).
- **Learning rate:** $\alpha = 0.5000$.
- **Initial weight:** $w_0 = 1.0000$.

Let us compute the semi-gradient update by hand step-by-step!

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Value / Formula |
| :--- | :--- | :--- |
| $S_t$ | Sampled Starting State | Always $S_1$ ($\mu(S_1) = 1.0$) |
| $S_{t+1}$ | Successor State under $\pi$ | Always $S_2$ |
| $R_{t+1}$ | Reward | $0.0000$ |
| $\gamma$ | Discount Factor | $0.9000$ |
| $\phi(S_t)$ | Feature at Current State | $\phi(S_1) = 1.0000$ |
| $\phi(S_{t+1})$ | Feature at Next State | $\phi(S_2) = 2.0000$ |
| $\hat{v}(S_t, w)$ | Current Estimated Value | $w \times 1.0000 = w$ |
| $\hat{v}(S_{t+1}, w)$ | Bootstrapped Successor Value | $w \times 2.0000 = 2w$ |
| Target | Bootstrapped Target | $R + \gamma \hat{v}(S_2, w) = 0 + 0.9(2w) = \mathbf{1.8w}$ |
| TD Error $\delta$ | Target minus Estimate | $1.8w - 1.0w = \mathbf{0.8w}$ |
| $\nabla_w \hat{v}$ | Gradient with respect to $w$ | $\phi(S_1) = 1.0000$ |
| $w_{\text{new}}$ | Updated Weight | $w + \alpha \delta \nabla_w \hat{v} = w + 0.5(0.8w)(1.0) = \mathbf{1.4w}$ |

---

### 5.3 Step-by-Step Hand Calculations: Iteration 1 to 5

At every step $t$, the update equation is:
$$w_{t+1} = w_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_2, w_t) - \hat{v}(S_1, w_t) \right] \phi(S_1)$$
$$= w_t + 0.5000 \left[ 0.0 + 0.9000(2.0000 w_t) - 1.0000 w_t \right] (1.0000)$$
$$= w_t + 0.5000 \left[ 1.8000 w_t - 1.0000 w_t \right]$$
$$= w_t + 0.5000 \left[ 0.8000 w_t \right]$$
$$= w_t + 0.4000 w_t = \mathbf{1.4000 w_t}$$

Notice: At every single update, the weight is multiplied by **$1.4000 > 1$**!
Let us trace the numbers starting from $w_0 = 1.0000$:

#### Iteration 1:
- $\text{Target} = 0 + 0.9(2 \times 1.0000) = \mathbf{1.8000}$
- $\delta_0 = 1.8000 - 1.0000 = \mathbf{0.8000}$
- $w_1 = 1.0000 + 0.5000 \times 0.8000 \times 1.0 = 1.0000 + 0.4000 = \mathbf{1.4000}$

#### Iteration 2:
- Current $w_1 = 1.4000$
- $\text{Target} = 0.9(2 \times 1.4000) = 0.9 \times 2.8000 = \mathbf{2.5200}$
- $\delta_1 = 2.5200 - (1 \times 1.4000) = 2.5200 - 1.4000 = \mathbf{1.1200}$
- $w_2 = 1.4000 + 0.5000 \times 1.1200 \times 1.0 = 1.4000 + 0.5600 = \mathbf{1.9600}$ ($= 1.4^2$)

#### Iteration 3:
- Current $w_2 = 1.9600$
- $\text{Target} = 0.9(2 \times 1.9600) = \mathbf{3.5280}$
- $\delta_2 = 3.5280 - 1.9600 = \mathbf{1.5680}$
- $w_3 = 1.9600 + 0.5000 \times 1.5680 = 1.9600 + 0.7840 = \mathbf{2.7440}$ ($= 1.4^3$)

#### Iteration 4:
- Current $w_3 = 2.7440$
- $\text{Target} = 0.9(2 \times 2.7440) = \mathbf{4.9392}$
- $\delta_3 = 4.9392 - 2.7440 = \mathbf{2.1952}$
- $w_4 = 2.7440 + 0.5000 \times 2.1952 = 2.7440 + 1.0976 = \mathbf{3.8416}$ ($= 1.4^4$)

#### Iteration 5:
- Current $w_4 = 3.8416$
- $\text{Target} = 0.9(2 \times 3.8416) = \mathbf{6.91488}$
- $\delta_4 = 6.91488 - 3.8416 = \mathbf{3.07328}$
- $w_5 = 3.8416 + 0.5000 \times 3.07328 = 3.8416 + 1.53664 = \mathbf{5.37824}$ ($= 1.4^5$)

---

### 5.4 Summary Arithmetic Progression Table

| Step $t$ | Current Weight $w_t$ | Target $0.9(2w_t)$ | TD Error $\delta_t$ | Update $\alpha \delta_t$ | New Weight $w_{t+1}$ | Exact Analytical $(1.4)^t$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 0$** | $1.0000$ | $1.8000$ | $0.8000$ | $+0.4000$ | $\mathbf{1.4000}$ | $1.4000$ |
| **$t = 1$** | $1.4000$ | $2.5200$ | $1.1200$ | $+0.5600$ | $\mathbf{1.9600}$ | $1.9600$ |
| **$t = 2$** | $1.9600$ | $3.5280$ | $1.5680$ | $+0.7840$ | $\mathbf{2.7440}$ | $2.7440$ |
| **$t = 3$** | $2.7440$ | $4.9392$ | $2.1952$ | $+1.0976$ | $\mathbf{3.8416}$ | $3.8416$ |
| **$t = 4$** | $3.8416$ | $6.91488$ | $3.07328$ | $+1.53664$ | $\mathbf{5.37824}$ | $5.37824$ |
| **$t = 10$** | $28.9255$ | - | - | - | $\mathbf{40.4957}$ | $40.4957$ |
| **$t = 50$** | - | - | - | - | - | $\mathbf{2.22 \times 10^7}$ |
| **$t \to \infty$** | - | - | - | - | - | $\mathbf{+\infty}$ (Explosion!) |

The system exhibits exponential divergence with Lyapunov exponent $\ln(1.4) \approx +0.3365$!

---

### Illustration 1: Baird's 7-State Counterexample and Explicit Eigenvalue Instability

**Problem:**
Consider Baird's 7-state MDP where states $\{s_1, \dots, s_6\}$ are upper states and $s_7$ is the lower state.
Under the solid action (target policy $\pi$), all 7 states transition deterministically to state $s_7$:
$$P^\pi(s_7 \mid s) = 1.0 \quad \forall s \in \{s_1, \dots, s_7\}$$
Rewards are zero everywhere: $\mathbf{R}^\pi = \mathbf{0}$. The discount factor is $\gamma = 0.99$.
The state is approximated linearly with $d = 7$ features $\mathbf{\Phi} \in \mathbb{R}^{7 \times 7}$:
$$\mathbf{x}(s_i) = 2 \mathbf{e}_i + \mathbf{e}_7 \quad (i = 1, \dots, 6), \quad \mathbf{x}(s_7) = 2 \mathbf{e}_7$$
Under uniform off-policy behavior sampling, the empirical distribution is $\mathbf{D}_\mu = \frac{1}{7} \mathbf{I}$.
1. Construct the expected semi-gradient update matrix $\mathbf{A}_{\text{baird}} = \mathbf{\Phi}^\top \mathbf{D}_\mu (\gamma \mathbf{P}^\pi \mathbf{\Phi} - \mathbf{\Phi})$.
2. Compute the eigenvalues of $\mathbf{A}_{\text{baird}}$ and prove why the discrete-time linear system $\mathbf{w}_{t+1} = (\mathbf{I} + \alpha \mathbf{A}_{\text{baird}}) \mathbf{w}_t$ is unconditionally unstable for any learning rate $\alpha > 0$.

**Solution:**

*Step 1: Construct the constituent matrices.*
The feature matrix $\mathbf{\Phi}$ has rows $\mathbf{x}(s_i)^\top$:
$$\mathbf{\Phi} = \begin{bmatrix}
2 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 2 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 2 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 2 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 2 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 2 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 2
\end{bmatrix}$$
Under the solid action, all states transition to $s_7$, so row $s$ of $\mathbf{P}^\pi \mathbf{\Phi}$ is identically $\mathbf{x}(s_7)^\top = [0, 0, 0, 0, 0, 0, 2]$.
With uniform behavior sampling $\mathbf{D}_\mu = \frac{1}{7} \mathbf{I}$, the expected update matrix is:
$$\mathbf{A}_{\text{baird}} = \mathbf{\Phi}^\top \mathbf{D}_\mu \left( \gamma \mathbf{P}^\pi \mathbf{\Phi} - \mathbf{\Phi} \right) = \frac{1}{7} \mathbf{\Phi}^\top \left( 0.99 \mathbf{P}^\pi \mathbf{\Phi} - \mathbf{\Phi} \right)$$

*Step 2: Eigenvalue analysis.*
Evaluating the characteristic polynomial $\det(\lambda \mathbf{I} - \mathbf{A}_{\text{baird}}) = 0$ yields 7 eigenvalues:
- Five identical negative eigenvalues corresponding to the decoupled coordinates: $\lambda_{1..5} = -\frac{4}{7} \approx \mathbf{-0.5714}$
- A small positive eigenvalue: $\lambda_6 \approx \mathbf{+0.0131}$
- A dominant positive eigenvalue: $\lambda_7 \approx \mathbf{+0.2498}$

*Step 3: Unconditional divergence proof.*
The expected semi-gradient iteration is $\mathbf{w}_{t+1} = \mathbf{M} \mathbf{w}_t$, where $\mathbf{M} = \mathbf{I} + \alpha \mathbf{A}_{\text{baird}}$.
By the spectral mapping theorem, the eigenvalues of $\mathbf{M}$ are:
$$\mu_i = 1 + \alpha \lambda_i$$
For the dominant eigenvalue $\lambda_7 \approx +0.2498 > 0$:
$$\mu_7 = 1 + 0.2498 \alpha > 1 \quad \forall \alpha > 0$$
The spectral radius satisfies:
$$\rho(\mathbf{M}) \triangleq \max_i |\mu_i| \ge 1 + 0.2498 \alpha > 1$$
Because $\rho(\mathbf{M}) > 1$, the linear dynamical system is unstable: for almost any non-zero initial weight vector $\mathbf{w}_0$, the component along the dominant eigenvector grows geometrically as $(1 + 0.2498 \alpha)^t$, leading to catastrophic divergence $\|\mathbf{w}_t\| \to \infty$. $\blacksquare$

---

### Illustration 2: Closed-Form Matrix Inversion for On-Policy Linear TD(0) Fixed Point ($\mathbf{w}^* = \mathbf{A}^{-1}\mathbf{b}$)

**Problem:**
Consider an ergodic 3-state Markov Reward Process with state space $\mathcal{S} = \{S_1, S_2, S_3\}$:
$$\mathbf{P} = \begin{bmatrix} 0.0 & 1.0 & 0.0 \\ 0.5 & 0.0 & 0.5 \\ 0.0 & 1.0 & 0.0 \end{bmatrix}, \quad \mathbf{R} = \begin{bmatrix} 1.0 \\ 0.0 \\ 2.0 \end{bmatrix}, \quad \gamma = 0.80$$
The stationary distribution is $\mathbf{d} = [0.25, 0.50, 0.25]^\top$.
Let the value function be approximated by 2 linear features $\mathbf{\Phi} \in \mathbb{R}^{3 \times 2}$:
$$\mathbf{\Phi} = \begin{bmatrix} 1.0 & 0.0 \\ 1.0 & 1.0 \\ 0.0 & 2.0 \end{bmatrix}$$
1. Compute the $\mathbf{A}$ matrix and $\mathbf{b}$ vector for linear TD(0).
2. Verify that $\mathbf{A}$ is strictly positive definite.
3. Solve for the analytical TD fixed point $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1} \mathbf{b}$ and corresponding value predictions $\hat{\mathbf{V}}_{\text{TD}}$.

**Solution:**

*Step 1: Compute $\mathbf{A}$ and $\mathbf{b}$.*
Stationary diagonal matrix $\mathbf{D} = \operatorname{diag}(0.25, 0.50, 0.25)$.
Evaluate $\mathbf{I} - \gamma \mathbf{P}$:
$$\mathbf{I} - 0.80 \mathbf{P} = \begin{bmatrix} 1.00 & -0.80 & 0.00 \\ -0.40 & 1.00 & -0.40 \\ 0.00 & -0.80 & 1.00 \end{bmatrix}$$
Now multiply by $\mathbf{D}$:
$$\mathbf{D} (\mathbf{I} - \gamma \mathbf{P}) = \begin{bmatrix} 0.25 & 0 & 0 \\ 0 & 0.50 & 0 \\ 0 & 0 & 0.25 \end{bmatrix} \begin{bmatrix} 1.00 & -0.80 & 0.00 \\ -0.40 & 1.00 & -0.40 \\ 0.00 & -0.80 & 1.00 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.20 & 0.00 \\ -0.20 & 0.50 & -0.20 \\ 0.00 & -0.20 & 0.25 \end{bmatrix}$$
Now project with feature matrix $\mathbf{\Phi}$:
$$\mathbf{A} = \mathbf{\Phi}^\top \left[ \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}) \right] \mathbf{\Phi}$$
First compute $\left[ \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}) \right] \mathbf{\Phi}$:
$$\begin{bmatrix} 0.25 & -0.20 & 0.00 \\ -0.20 & 0.50 & -0.20 \\ 0.00 & -0.20 & 0.25 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 1.0 & 1.0 \\ 0.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 0.05 & -0.20 \\ 0.30 & 0.10 \\ -0.20 & 0.30 \end{bmatrix}$$
Premultiply by $\mathbf{\Phi}^\top = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 2 \end{bmatrix}$:
$$\mathbf{A} = \begin{bmatrix} 0.05 + 0.30 & -0.20 + 0.10 \\ 0.30 + 2(-0.20) & 0.10 + 2(0.30) \end{bmatrix} = \begin{bmatrix} \mathbf{0.35} & \mathbf{-0.10} \\ \mathbf{-0.10} & \mathbf{0.70} \end{bmatrix}$$
Now compute vector $\mathbf{b} = \mathbf{\Phi}^\top \mathbf{D} \mathbf{R}$:
$$\mathbf{D} \mathbf{R} = \begin{bmatrix} 0.25 \times 1.0 \\ 0.50 \times 0.0 \\ 0.25 \times 2.0 \end{bmatrix} = \begin{bmatrix} 0.25 \\ 0.00 \\ 0.50 \end{bmatrix} \implies \mathbf{b} = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 0.25 \\ 0.00 \\ 0.50 \end{bmatrix} = \begin{bmatrix} \mathbf{0.25} \\ \mathbf{1.00} \end{bmatrix}$$

*Step 2: Verify positive definiteness.*
Matrix $\mathbf{A} = \begin{bmatrix} 0.35 & -0.10 \\ -0.10 & 0.70 \end{bmatrix}$ is symmetric.
Leading principal minors:
1. $M_1 = 0.35 > 0$
2. $\det(\mathbf{A}) = (0.35)(0.70) - (-0.10)^2 = 0.2450 - 0.0100 = \mathbf{0.2350} > 0$
Both minors are strictly positive, so $\mathbf{A} \succ 0$ (strictly positive definite).
Eigenvalues: $\lambda_{1, 2} = \frac{1.05 \pm \sqrt{(1.05)^2 - 4(0.235)}}{2} = \frac{1.05 \pm \sqrt{1.1025 - 0.9400}}{2} = \frac{1.05 \pm 0.4031}{2} \implies \lambda_1 \approx 0.3234, \lambda_2 \approx 0.7266 > 0$.

*Step 3: Solve for $\mathbf{w}_{\text{TD}}$ and $\hat{\mathbf{V}}_{\text{TD}}$.*
$$\mathbf{A}^{-1} = \frac{1}{0.2350} \begin{bmatrix} 0.70 & 0.10 \\ 0.10 & 0.35 \end{bmatrix}$$
$$\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1} \mathbf{b} = \frac{1}{0.2350} \begin{bmatrix} 0.70(0.25) + 0.10(1.00) \\ 0.10(0.25) + 0.35(1.00) \end{bmatrix} = \frac{1}{0.2350} \begin{bmatrix} 0.1750 + 0.1000 \\ 0.0250 + 0.3500 \end{bmatrix} = \frac{1}{0.2350} \begin{bmatrix} 0.2750 \\ 0.3750 \end{bmatrix} = \begin{bmatrix} \frac{55}{47} \\ \frac{75}{47} \end{bmatrix} \approx \begin{bmatrix} \mathbf{1.1702} \\ \mathbf{1.5957} \end{bmatrix}$$
The resulting value approximations across the 3 states are:
$$\hat{\mathbf{V}}_{\text{TD}} = \mathbf{\Phi} \mathbf{w}_{\text{TD}} = \begin{bmatrix} 1.0 & 0.0 \\ 1.0 & 1.0 \\ 0.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.1702 \\ 1.5957 \end{bmatrix} = \begin{bmatrix} \mathbf{1.1702} \\ \mathbf{2.7660} \\ \mathbf{3.1915} \end{bmatrix} \quad \blacksquare$$

---

## 6. Solved Illustrations

### Illustration 3: Numerical Verification of the Tsitsiklis-Van Roy Error Bound

**Problem:**
Using the 3-state MRP from Illustration 2:
1. Compute the exact analytical value function $\mathbf{V}^* = (\mathbf{I} - \gamma \mathbf{P})^{-1} \mathbf{R}$.
2. Compute the optimal weighted least-squares projection $\mathbf{w}^*_{\text{OLS}} = (\mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \mathbf{D} \mathbf{V}^*$ and its approximation error $\|\mathbf{V}^* - \mathbf{\Phi} \mathbf{w}^*_{\text{OLS}}\|_{\mathbf{D}}$.
3. Compute the TD approximation error $\|\mathbf{V}^* - \mathbf{\Phi} \mathbf{w}_{\text{TD}}\|_{\mathbf{D}}$.
4. Verify that the Tsitsiklis-Van Roy inequality holds: $\|\mathbf{V}^* - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}} \le \frac{1}{\sqrt{1 - \gamma^2}} \|\mathbf{V}^* - \mathbf{V}_{\text{OLS}}\|_{\mathbf{D}}$.

**Solution:**

*Step 1: Compute true values $\mathbf{V}^*$.*
Solve $(\mathbf{I} - 0.80 \mathbf{P}) \mathbf{V}^* = \mathbf{R}$:
$$\begin{bmatrix} 1.00 & -0.80 & 0.00 \\ -0.40 & 1.00 & -0.40 \\ 0.00 & -0.80 & 1.00 \end{bmatrix} \begin{bmatrix} V^*(S_1) \\ V^*(S_2) \\ V^*(S_3) \end{bmatrix} = \begin{bmatrix} 1.0 \\ 0.0 \\ 2.0 \end{bmatrix}$$
From row 1: $V^*(S_1) = 1.0 + 0.8 V^*(S_2)$.
From row 3: $V^*(S_3) = 2.0 + 0.8 V^*(S_2)$.
Substitute into row 2:
$$-0.4(1.0 + 0.8 V^*(S_2)) + V^*(S_2) - 0.4(2.0 + 0.8 V^*(S_2)) = 0$$
$$-0.4 - 0.32 V^*(S_2) + V^*(S_2) - 0.8 - 0.32 V^*(S_2) = 0$$
$$0.36 V^*(S_2) = 1.2 \implies V^*(S_2) = \frac{1.2}{0.36} = \frac{10}{3} \approx \mathbf{3.3333}$$
Then:
$$V^*(S_1) = 1.0 + 0.8\left(\frac{10}{3}\right) = 1.0 + \frac{8}{3} = \frac{11}{3} \approx \mathbf{3.6667}$$
$$V^*(S_3) = 2.0 + 0.8\left(\frac{10}{3}\right) = 2.0 + \frac{8}{3} = \frac{14}{3} \approx \mathbf{4.6667}$$
Thus:
$$\mathbf{V}^* = \begin{bmatrix} 11/3 \\ 10/3 \\ 14/3 \end{bmatrix} \approx \begin{bmatrix} 3.6667 \\ 3.3333 \\ 4.6667 \end{bmatrix}$$

*Step 2: Optimal OLS projection.*
Compute feature covariance $\mathbf{C} = \mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi}$:
$$\mathbf{C} = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 0.25 & 0 & 0 \\ 0 & 0.50 & 0 \\ 0 & 0 & 0.25 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 0.75 & 0.50 \\ 0.50 & 1.50 \end{bmatrix}$$
$\det(\mathbf{C}) = (0.75)(1.50) - (0.50)^2 = 1.1250 - 0.2500 = 0.8750 = \frac{7}{8}$.
$$\mathbf{\Phi}^\top \mathbf{D} \mathbf{V}^* = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 0.25(11/3) \\ 0.50(10/3) \\ 0.25(14/3) \end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 11/12 \\ 20/12 \\ 14/12 \end{bmatrix} = \begin{bmatrix} 31/12 \\ 48/12 \end{bmatrix} = \begin{bmatrix} 2.5833 \\ 4.0000 \end{bmatrix}$$
$$\mathbf{w}^*_{\text{OLS}} = \frac{8}{7} \begin{bmatrix} 1.50 & -0.50 \\ -0.50 & 0.75 \end{bmatrix} \begin{bmatrix} 31/12 \\ 4 \end{bmatrix} = \begin{bmatrix} 45/21 \\ 41/21 \end{bmatrix} \approx \begin{bmatrix} \mathbf{2.1429} \\ \mathbf{1.9524} \end{bmatrix}$$
Value predictions: $\mathbf{V}_{\text{OLS}} = \mathbf{\Phi} \mathbf{w}^*_{\text{OLS}} = [2.1429, 4.0952, 3.9048]^\top$.
The minimum projection error is:
$$\|\mathbf{V}^* - \mathbf{V}_{\text{OLS}}\|_{\mathbf{D}} = \sqrt{0.25(3.6667 - 2.1429)^2 + 0.50(3.3333 - 4.0952)^2 + 0.25(4.6667 - 3.9048)^2} \approx \mathbf{1.0079}$$

*Step 3: TD error.*
Using $\mathbf{V}_{\text{TD}} = [1.1702, 2.7660, 3.1915]^\top$ from Illustration 2:
$$\|\mathbf{V}^* - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}} = \sqrt{0.25(3.6667 - 1.1702)^2 + 0.50(3.3333 - 2.7660)^2 + 0.25(4.6667 - 3.1915)^2} \approx \mathbf{1.5043}$$

*Step 4: Verification of the theoretical bound.*
With $\gamma = 0.80$:
$$\frac{1}{\sqrt{1 - \gamma^2}} = \frac{1}{\sqrt{1 - (0.80)^2}} = \frac{1}{\sqrt{0.36}} = \frac{1}{0.60} = \frac{5}{3} \approx 1.6667$$
Theoretical upper bound:
$$\text{Bound} = \frac{1}{\sqrt{1 - \gamma^2}} \|\mathbf{V}^* - \mathbf{V}_{\text{OLS}}\|_{\mathbf{D}} = 1.6667 \times 1.0079 = \mathbf{1.6798}$$
Comparing the actual TD error against the bound:
$$\|\mathbf{V}^* - \mathbf{V}_{\text{TD}}\|_{\mathbf{D}} = \mathbf{1.5043} \le \mathbf{1.6798} \quad \checkmark$$
The Tsitsiklis-Van Roy error bound is confirmed with exact numerical rigor! $\blacksquare$

---

### Illustration 4: Gradient TD (GTD2) Step-by-Step Weight Dynamics Taming the 2-State Divergence

**Problem:**
Take the Tsitsiklis & Van Roy 2-state divergence example from Section 5 ($S_1 \to S_2$, $R = 0$, $\gamma = 0.90$, $\phi(S_1) = 1.0, \phi(S_2) = 2.0$, sampled always from $S_1$).
Recall that semi-gradient TD(0) exploded exponentially ($w_0 = 1 \to w_1 = 1.4 \to w_2 = 1.96 \to w_3 = 2.744 \to w_4 = 3.8416$).
Apply the **GTD2 algorithm** with primary learning rate $\alpha = 0.10$, secondary learning rate $\beta = 0.50$, and initial weights $w_0 = 1.0000, v_0 = 0.0000$.
Trace the weight updates for steps $t = 0, 1, 2, 3$ and show that GTD2 drives the weights toward the true value $w^* = 0$.

**Solution:**

The GTD2 update equations for scalar feature $x_t = 1.0, x_{t+1} = 2.0, R_{t+1} = 0$:
$$\delta_t = R_{t+1} + \gamma (w_t x_{t+1}) - w_t x_t = 0 + 0.90(2.0 w_t) - 1.0 w_t = 0.80 w_t$$
$$v_{t+1} = v_t + \beta \left( \delta_t - x_t v_t \right) x_t = v_t + 0.50 (\delta_t - v_t)$$
$$w_{t+1} = w_t + \alpha \left( x_t - \gamma x_{t+1} \right) (x_t v_t) = w_t + 0.10 (1.0 - 0.90 \times 2.0) v_t = w_t - 0.08 v_t$$

- **Step $t = 0$:**
  - $w_0 = 1.0000, \quad v_0 = 0.0000$
  - $\delta_0 = 0.80(1.0000) = \mathbf{0.8000}$
  - $v_1 = 0.0000 + 0.50(0.8000 - 0.0000) = \mathbf{0.4000}$
  - $w_1 = 1.0000 - 0.08(0.0000) = \mathbf{1.0000}$

- **Step $t = 1$:**
  - $w_1 = 1.0000, \quad v_1 = 0.4000$
  - $\delta_1 = 0.80(1.0000) = \mathbf{0.8000}$
  - $v_2 = 0.4000 + 0.50(0.8000 - 0.4000) = 0.4000 + 0.2000 = \mathbf{0.6000}$
  - $w_2 = 1.0000 - 0.08(0.4000) = 1.0000 - 0.0320 = \mathbf{0.9680}$

- **Step $t = 2$:**
  - $w_2 = 0.9680, \quad v_2 = 0.6000$
  - $\delta_2 = 0.80(0.9680) = \mathbf{0.7744}$
  - $v_3 = 0.6000 + 0.50(0.7744 - 0.6000) = 0.6000 + 0.0872 = \mathbf{0.6872}$
  - $w_3 = 0.9680 - 0.08(0.6000) = 0.9680 - 0.0480 = \mathbf{0.9200}$

- **Step $t = 3$:**
  - $w_3 = 0.9200, \quad v_3 = 0.6872$
  - $\delta_3 = 0.80(0.9200) = \mathbf{0.7360}$
  - $v_4 = 0.6872 + 0.50(0.7360 - 0.6872) = 0.6872 + 0.0244 = \mathbf{0.7116}$
  - $w_4 = 0.9200 - 0.08(0.6872) = 0.9200 - 0.0550 = \mathbf{0.8650}$

*Comparison Summary:*
| Step $t$ | Semi-Gradient TD(0) $w_t$ | GTD2 Weight $w_t$ | GTD2 Auxiliary $v_t$ | Behavior |
| :---: | :---: | :---: | :---: | :---: |
| **$0$** | $1.0000$ | $1.0000$ | $0.0000$ | Initial state |
| **$1$** | $1.4000$ | $1.0000$ | $0.4000$ | $v$ accumulates gradient |
| **$2$** | $1.9600$ | $0.9680$ | $0.6000$ | $w$ begins contracting |
| **$3$** | $2.7440$ | $0.9200$ | $0.6872$ | Continuous decrease |
| **$4$** | $3.8416$ | $0.8650$ | $0.7116$ | Contracting toward $w^* = 0$ |
| **$t \to \infty$** | **$+\infty$ (DIVERGES)** | **$0.0000$ (CONVERGES)** | $0.0000$ | GTD2 completely tames divergence! |

$\blacksquare$

---

### Illustration 5: The Double-Sampling Pitfall: Residual Gradient vs. Semi-Gradient TD on a Stochastic Fork

**Problem:**
Consider a state $S$ that branches stochastically to one of two terminal states under policy $\pi$:
- With probability $p = 0.5$, transitions to $S_A$ with reward $R = 0.0$.
- With probability $p = 0.5$, transitions to $S_B$ with reward $R = 2.0$.
Terminal values are known: $V(S_A) = V(S_B) = 0.0$, discount $\gamma = 1.0$.
The agent parameterizes $V(S) \triangleq w$.
1. Compute the true value $V^*(S)$ and the Bellman Error fixed point.
2. Show why Baird's single-sample Residual Gradient algorithm converges to the wrong value when transitions are sampled stochastic step-by-step without double sampling.

**Solution:**

*Step 1: True Bellman Error minimization.*
The expected Bellman error for state $S$ is:
$$\overline{\delta}(w) \triangleq \mathbb{E}[R + \gamma V(S') - V(S)] = [0.5(0) + 0.5(2.0)] - w = 1.0 - w$$
The squared expected Bellman error objective is:
$$J_{\text{Bellman}}(w) = \frac{1}{2} (\overline{\delta}(w))^2 = \frac{1}{2} (1.0 - w)^2$$
Setting the derivative to zero:
$$\frac{d}{dw} J_{\text{Bellman}}(w) = -(1.0 - w) = 0 \implies w^* = \mathbf{1.0000}$$
This is the true value $V^*(S) = 1.0000$.

*Step 2: Single-Sample Residual Gradient minimization.*
If an agent instead minimizes the expectation of the *squared* sample TD error:
$$J_{\text{Residual}}(w) \triangleq \frac{1}{2} \mathbb{E} \left[ (R + \gamma V(S') - V(S))^2 \right]$$
Expanding the expectation:
$$J_{\text{Residual}}(w) = \frac{1}{2} \left[ 0.5(0.0 - w)^2 + 0.5(2.0 - w)^2 \right] = \frac{1}{4} \left[ w^2 + (4.0 - 4.0w + w^2) \right] = \frac{1}{2} w^2 - w + 1.0$$
Notice the bias-variance decomposition:
$$J_{\text{Residual}}(w) = \frac{1}{2} (\mathbb{E}[\delta])^2 + \frac{1}{2} \operatorname{Var}(R + \gamma V(S')) = \frac{1}{2}(1 - w)^2 + \frac{1}{2}(1.0)$$
Taking the derivative:
$$\frac{d}{dw} J_{\text{Residual}}(w) = w - 1.0 = 0 \implies w = \mathbf{1.0000}$$
Here, because $V(S') = 0$, both minima happen to coincide at $1.0$.

Now suppose $S_A$ has successor estimate $\hat{V}(S_A) = 4.0$ (due to function approximation bias), while $\hat{V}(S_B) = 0.0$.
Then the sample targets are:
- Path A: $Y_A = 0.0 + 4.0 = 4.0$ (prob 0.5)
- Path B: $Y_B = 2.0 + 0.0 = 2.0$ (prob 0.5)
Expected target: $\mathbb{E}[Y] = 0.5(4.0) + 0.5(2.0) = \mathbf{3.0000}$.
Semi-gradient TD converges to:
$$\mathbb{E}[\delta] = 0 \implies 3.0 - w = 0 \implies w_{\text{TD}} = \mathbf{3.0000}$$
Now evaluate the Residual Gradient with single-sample updates:
$$\Delta w = \alpha \delta_t \left( -\nabla_w \delta_t \right) = \alpha (Y_t - w) (1) = \alpha (Y_t - w)$$
Its expectation is $\mathbb{E}[\Delta w] = \alpha (3.0 - w)$.
However, if $V(S)$ and $V(S')$ share weights $\mathbf{w}$ (e.g., $V(S) = w, V(S_A) = 2w$):
$$\nabla_w \delta_t = \gamma \nabla_w V(S') - \nabla_w V(S) = 2 - 1 = +1$$
The sample gradient becomes:
$$\mathbb{E} \left[ \delta_t \nabla_w \delta_t \right] \neq \mathbb{E}[\delta_t] \mathbb{E}[\nabla_w \delta_t]$$
Because $\mathbb{E}[X Y] = \mathbb{E}[X]\mathbb{E}[Y] + \operatorname{Cov}(X, Y)$, single-sample residual gradient incorporates an inescapable covariance distortion $\operatorname{Cov}(\delta_t, \nabla_w \delta_t)$ that biases the parameter solution away from the true Bellman fixed point, unless two independent successor transitions are drawn from state $S$ at every step! $\blacksquare$

---

## 7. Deep Learning Connection & Application

### 1. How Deep Q-Networks (DQN) Tamed the Deadly Triad (Mnih et al., 2015)
DQN successfully combined all three elements of the Deadly Triad (Neural Networks + Bootstrapping + Off-Policy Replay) through two breakthrough engineering innovations:
- **Target Network $\theta^-$ (Quenching the Bootstrapping Loop):**
  Instead of using the rapidly changing online parameters $\theta$ in the target, DQN computes targets with frozen parameters $\theta^-$ updated only every $C = 10,000$ steps:
  $$Y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$
  This breaks the instantaneous self-amplifying feedback loop $\theta \leftarrow \theta$, turning the loss temporarily into standard supervised regression!
- **Experience Replay Buffer $\mathcal{D}$ (Bridging Off-Policy Drift):**
  Sampling uniformly from a rolling buffer of 1,000,000 transitions decorrelates consecutive training examples and stabilizes the empirical state visitation distribution $\mu(s)$.

### 2. Gradient TD Methods: Provably Convergent Off-Policy Updates (Sutton et al., 2009)
GTD2 and TDC provide true $O(d)$ stochastic gradient descent on the Projected Bellman Error:
$$\text{PBE}(\mathbf{w}) = \|\Pi(T^\pi V_\mathbf{w}) - V_\mathbf{w}\|^2_\mathbf{D}$$
Unlike semi-gradient TD which follows a biased gradient direction, GTD2's two-timescale update guarantees convergence to the TD fixed point under **any** off-policy data distribution — the first truly off-policy stable linear function approximation algorithm.

### 3. Modern Deep RL Solutions: Stable Off-Policy Architectures
- **Double DQN (van Hasselt et al., 2016):** Decouples action selection ($\theta$) from value evaluation ($\theta^-$): $Y_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta); \theta^-)$, dramatically reducing overestimation bias.
- **Retrace($\lambda$) (Munos et al., NeurIPS 2016):** A safe, off-policy return operator using importance sampling ratio clipping $c_s = \min(1, \pi(a_s|s_s)/\mu(a_s|s_s))$, provably convergent for any behavior policy with full coverage support.
- **Emphatic TD (Sutton et al., 2016):** Reweights updates by an emphasis scalar $M_t = \lambda M_{t-1} + i(S_t)$ that emphasizes states with high interest $i(s)$, achieving stable convergence without requiring on-policy data.

---

## 8. Code Implementation & Verification

The accompanying Python script verifies:
1. Exact numerical reproduction of Part 5 hand calculations ($w_1 = 1.4000, w_2 = 1.9600, w_3 = 2.7440, w_4 = 3.8416, w_5 = 5.37824$) matching machine precision to $< 10^{-14}$.
2. Linear Semi-Gradient TD(0) on an on-policy Random Walk: verifying stable convergence to the analytical projected fixed point $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1} \mathbf{b}$.
3. Baird's Counterexample: Simulating off-policy semi-gradient Q-learning and observing the weight vector norm $\|\mathbf{w}_t\|$ diverging exponentially toward infinity.

See implementation in:
[`11_reinforcement_learning/code/09_function_approximation_and_deadly_triad.py`](./code/09_function_approximation_and_deadly_triad.py)
