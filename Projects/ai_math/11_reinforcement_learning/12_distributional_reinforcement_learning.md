# Module 11.12: Distributional Reinforcement Learning (C51 & QR-DQN)

---

## 1. Intuition & 101 Motivation

Throughout the classical history of reinforcement learning—from Bellman (1957) to DQN (2015)—the central object of study has been the **expectation of returns**:
$$Q(s, a) \triangleq \mathbb{E} [G_t \mid S_t = s, A_t = a]$$

While taking expectations simplifies math, it discards critical information about the **random variable of returns** $Z(s, a)$:
$$Z(s, a) \triangleq \sum_{t=0}^\infty \gamma^t R_{t+1}$$
whose expectation is the standard Q-value: $\mathbb{E}[Z(s, a)] = Q(s, a)$.

**Distributional Reinforcement Learning** (Bellemare et al., 2017) fundamentally changes this paradigm: instead of estimating the scalar mean $Q(s, a)$, the agent estimates the **complete probability distribution** of $Z(s, a)$.

### Why Distributional RL Outperforms Expectation RL:
1. **Preserving Multi-Modality:** In stochastic environments, a state might have a $50\%$ chance of $+100$ and $50\%$ chance of $0$. A scalar Q-value collapses this to $50$, indistinguishable from a deterministic outcome of $50$. Distributional RL preserves both modes.
2. **Mitigating Non-Stationarity & Policy Jumps:** Because distributions contain rich structural details about future state transitions, gradient updates adjust the full distribution rather than pushing a scalar average up or down, stabilizing deep representations.
3. **Risk-Sensitive Decision Making:** An agent can optimize for metrics beyond expected value, such as Value at Risk (VaR), Conditional Value at Risk (CVaR), or worst-case tail protection.

```
       CLASSICAL RL (Scalar Mean)                    DISTRIBUTIONAL RL (Full Spectrum)
                                                                 Probability
                  Q(s, a)                                           ^
                     |                                              |      /\
                     v                                              |     /  \    /\
                 +-------+                                          |    /    \  /  \
                 | 50.00 |                                          +---+------+----+---> Return Z
                 +-------+                                              0      50   100
             (One single number)                                  (Multi-modal distribution!)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Distributional Bellman Equation

Let $\stackrel{D}{=}$ denote equality in distribution. The **Distributional Bellman Equation** for policy $\pi$ is:
$$Z^\pi(s, a) \stackrel{D}{=} R(s, a) + \gamma Z^\pi(S', A')$$
where $S' \sim \mathcal{P}(\cdot \mid s, a)$ and $A' \sim \pi(\cdot \mid S')$.

For the optimal policy (control):
$$Z^*(s, a) \stackrel{D}{=} R(s, a) + \gamma Z^*(S', a^*)$$
where $a^* = \arg\max_{a'} \mathbb{E}[Z^*(S', a')]$. Note that the greedy choice is still made with respect to the **expectation** of the return distribution!

---

### 2.2 The Wasserstein Metric & Contraction Mapping Theorem

To establish convergence, we must measure the distance between two probability distributions $P$ and $Q$.
The **$p$-Wasserstein metric** (Earth Mover's Distance) between distributions $F_1, F_2 \in \mathcal{P}(\mathbb{R})$ is defined as:
$$\mathcal{W}_p(F_1, F_2) \triangleq \left( \int_0^1 \left| F_1^{-1}(u) - F_2^{-1}(u) \right|^p du \right)^{1/p}$$
where $F^{-1}(u) \triangleq \inf \{ x \in \mathbb{R} : F(x) \ge u \}$ is the inverse cumulative distribution function (quantile function).

#### Theorem: Contraction of the Distributional Bellman Operator (Bellemare et al., 2017)
Let $\mathcal{T}^\pi$ be the distributional Bellman operator defined by:
$$\mathcal{T}^\pi Z(s, a) \stackrel{D}{=} R(s, a) + \gamma Z(S', A')$$
Then $\mathcal{T}^\pi$ is a **$\gamma$-contraction** in the maximal $p$-Wasserstein metric $\bar{\mathcal{W}}_p$:
$$\bar{\mathcal{W}}_p(\mathcal{T}^\pi Z_1, \mathcal{T}^\pi Z_2) \le \gamma \bar{\mathcal{W}}_p(Z_1, Z_2)$$
where $\bar{\mathcal{W}}_p(Z_1, Z_2) \triangleq \sup_{(s,a)} \mathcal{W}_p(Z_1(s, a), Z_2(s, a))$.

Consequently, under policy evaluation, repeated application of $\mathcal{T}^\pi$ converges to the unique fixed-point return distribution $Z^\pi$!

---

### 2.3 The Categorical Algorithm: C51 (Bellemare et al., ICML 2017)

In practice, a continuous distribution cannot be represented exactly by a neural network. C51 represents the return distribution using a discrete **categorical distribution** over $N = 51$ fixed, equidistant atoms:
$$\operatorname{supp}(Z) = \{z_0, z_1, \dots, z_{N-1}\}$$
$$z_i \triangleq V_{\min} + i \cdot \Delta z, \quad \Delta z \triangleq \frac{V_{\max} - V_{\min}}{N - 1}$$
where $[V_{\min}, V_{\max}]$ is the predefined return support range (e.g., $[-10, +10]$).

The network outputs a categorical probability vector $\mathbf{p}(s, a; \theta) \in \Delta^N$:
$$p_i(s, a; \theta) = \frac{\exp(\psi_i(s, a; \theta))}{\sum_{j=0}^{N-1} \exp(\psi_j(s, a; \theta))}$$
where $p_i(s, a) = \mathbb{P}(Z(s, a) = z_i)$.

The scalar expected value is computed trivially via dot product:
$$Q(s, a) = \sum_{i=0}^{N-1} z_i p_i(s, a) = \mathbf{z}^\top \mathbf{p}(s, a)$$

#### The Categorical Projection Step $\Phi$
Applying the Bellman update shifts and scales each atom:
$$\hat{T} z_j = r + \gamma z_j$$
Crucially, the shifted location $\hat{T} z_j$ will generally **not coincide** with any of the fixed support atoms $\{z_i\}$.
We must project the shifted probability mass $p_j(s', a^*)$ onto the adjacent atoms $\{z_l, z_{l+1}\}$ via the **Cramer Projection** $\Phi$:

For each shifted atom $j \in \{0, \dots, N-1\}$:
1. Clip to support: $\hat{z}_j = \operatorname{clip}(\hat{T} z_j, V_{\min}, V_{\max})$.
2. Compute fractional index: $b_j = \frac{\hat{z}_j - V_{\min}}{\Delta z}$.
3. Lower and upper integer atom indices: $l = \lfloor b_j \rfloor, \quad u = \lceil b_j \rceil$.
4. Distribute probability mass $p_j(s', a^*)$ proportionally:
   $$m_l \leftarrow m_l + p_j(s', a^*) \cdot (u - b_j)$$
   $$m_u \leftarrow m_u + p_j(s', a^*) \cdot (b_j - l)$$

#### The Loss Function: Cross-Entropy
The network parameters $\theta$ are trained by minimizing the **Kullback-Leibler (KL) divergence** (equivalently, cross-entropy) between the projected target distribution $\mathbf{m}$ and the predicted distribution $\mathbf{p}(s, a; \theta)$:

$$\mathcal{L}(\theta) = D_{\text{KL}}(\mathbf{m} \parallel \mathbf{p}(s, a; \theta)) = - \sum_{i=0}^{N-1} m_i \log p_i(s, a; \theta)$$

---

### 2.4 Quantile Regression DQN: QR-DQN (Dabney et al., AAAI 2018)

C51 fixes the support $\{z_i\}$ and learns variable probabilities $\{p_i\}$. This creates a theoretical mismatch because the projection step is a heuristic non-expansion.

**QR-DQN** reverses this geometry:
- It **fixes uniform probabilities** $\tau_i = \frac{2i - 1}{2N}$ for $N$ quantiles.
- It **learns the variable support locations** $\theta_i(s, a) \in \mathbb{R}$.

The network is trained using the **Quantile Huber Loss**:
$$\rho_\tau^\kappa(u) = |\tau - \mathbb{I}(u < 0)| \cdot \mathcal{L}_\kappa(u)$$
QR-DQN provides rigorous theoretical guarantees minimizing the 1-Wasserstein distance directly via stochastic gradient descent!

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.12.1: $\gamma$-Contraction of the Distributional Bellman Operator in the Wasserstein Metric

```
====================================================================================================
DERIVATION 11.12.1: Wasserstein Contraction of the Distributional Bellman Operator
====================================================================================================
Problem Statement:
Let 𝒯^π be the distributional Bellman evaluation operator acting on return distributions
Z ∈ 𝒫(ℝ)^{|𝒮| × |𝒜|}:
    𝒯^π Z(s, a) \stackrel{D}{=} R(s, a) + γ Z(S', A')
where S' ~ P(· | s, a) and A' ~ π(· | S').
Prove that for any p ≥ 1, 𝒯^π is a strict γ-contraction in the maximal p-Wasserstein metric:
    𝒲_p(𝒯^π Z_1(s, a), 𝒯^π Z_2(s, a)) ≤ γ 𝔼_{S', A'} [ 𝒲_p(Z_1(S', A'), Z_2(S', A')) ]
and consequently:
    \bar{𝒲}_p(𝒯^π Z_1, 𝒯^π Z_2) ≤ γ \bar{𝒲}_p(Z_1, Z_2)
where \bar{𝒲}_p(U, V) ≜ sup_{s, a} 𝒲_p(U(s, a), V(s, a)).
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Discount Factor:** $0 \le \gamma < 1$.
2. **Probability Space:** $Z(s, a)$ has finite $p$-th moment for some $p \ge 1$: $\mathbb{E}[|Z(s, a)|^p] < \infty$.
3. **Wasserstein Metric Definition:**
   $$\mathcal{W}_p(X, Y) \triangleq \left( \inf_{J \in \Pi(X, Y)} \mathbb{E}_{(U, V) \sim J} \left[ |U - V|^p \right] \right)^{1/p} = \left( \int_0^1 |F_X^{-1}(u) - F_Y^{-1}(u)|^p du \right)^{1/p}$$
   where $\Pi(X, Y)$ is the set of all joint couplings with marginals $X$ and $Y$.

**2. Underlying Intuition:**
Adding a deterministic scalar reward $R$ simply translates both random variables by the exact same distance, leaving the difference $U - V$ unchanged (translation invariance of Earth Mover's distance). Multiplying by discount factor $\gamma < 1$ compresses the horizontal axis of the cumulative distribution function by factor $\gamma$. Because the horizontal distance between quantile curves is shrunk by $\gamma$, the Wasserstein distance shrinks by exactly $\gamma$.

**3. End-to-End Algebraic Derivation:**

*Step 1: Quantile function of an affine transformation.*
Let $X$ be a real-valued random variable with cumulative distribution function $F_X(x) = \mathbb{P}(X \le x)$ and inverse CDF $F_X^{-1}(u) = \inf \{x \mid F_X(x) \ge u\}$ for $u \in (0, 1)$.
Consider the affine transformation $Y = a + c X$ with $c > 0$:
$$F_Y(y) = \mathbb{P}(a + c X \le y) = \mathbb{P}\left( X \le \frac{y - a}{c} \right) = F_X\left( \frac{y - a}{c} \right)$$
Evaluate the quantile function $F_Y^{-1}(u)$:
$$F_Y^{-1}(u) = \inf \left\{ y \;\middle|\; F_X\left( \frac{y - a}{c} \right) \ge u \right\} = a + c \cdot \inf \{ x \mid F_X(x) \ge u \} = a + c F_X^{-1}(u)$$

*Step 2: Scaling and translation invariance of the Wasserstein metric.*
Let $X_1, X_2$ be two random variables. For any constant $a \in \mathbb{R}$ and scale factor $c > 0$:
$$\mathcal{W}_p(a + c X_1, a + c X_2) = \left( \int_0^1 \left| F_{a + c X_1}^{-1}(u) - F_{a + c X_2}^{-1}(u) \right|^p du \right)^{1/p}$$
Using the affine quantile property from Step 1:
$$= \left( \int_0^1 \left| (a + c F_{X_1}^{-1}(u)) - (a + c F_{X_2}^{-1}(u)) \right|^p du \right)^{1/p}$$
Notice the constant translation $a$ cancels out completely:
$$= \left( \int_0^1 \left| c \left( F_{X_1}^{-1}(u) - F_{X_2}^{-1}(u) \right) \right|^p du \right)^{1/p} = c \left( \int_0^1 \left| F_{X_1}^{-1}(u) - F_{X_2}^{-1}(u) \right|^p du \right)^{1/p} = c \, \mathcal{W}_p(X_1, X_2)$$

*Step 3: Evaluating the Distributional Bellman Operator difference.*
For a fixed state-action pair $(s, a)$, let $R \sim \mathcal{R}(\cdot \mid s, a)$, $S' \sim \mathcal{P}(\cdot \mid s, a)$, and $A' \sim \pi(\cdot \mid S')$.
By definition of the optimal coupling over conditioned next-state transitions:
$$\mathcal{W}_p^p(\mathcal{T}^\pi Z_1(s, a), \mathcal{T}^\pi Z_2(s, a)) \le \mathbb{E}_{R, S', A'} \left[ \mathcal{W}_p^p(R + \gamma Z_1(S', A'), R + \gamma Z_2(S', A')) \right]$$
Applying the affine scaling identity from Step 2 with $a = R$ and $c = \gamma$:
$$= \mathbb{E}_{S', A'} \left[ \gamma^p \mathcal{W}_p^p(Z_1(S', A'), Z_2(S', A')) \right] = \gamma^p \mathbb{E}_{S', A'} \left[ \mathcal{W}_p^p(Z_1(S', A'), Z_2(S', A')) \right]$$
Taking the $p$-th root on both sides:
$$\mathcal{W}_p(\mathcal{T}^\pi Z_1(s, a), \mathcal{T}^\pi Z_2(s, a)) \le \gamma \left( \mathbb{E}_{S', A'} \left[ \mathcal{W}_p^p(Z_1(S', A'), Z_2(S', A')) \right] \right)^{1/p}$$

*Step 4: Bounding by the maximal supremum norm.*
Since $\mathcal{W}_p(Z_1(s', a'), Z_2(s', a')) \le \sup_{(s'', a'')} \mathcal{W}_p(Z_1(s'', a''), Z_2(s'', a'')) = \bar{\mathcal{W}}_p(Z_1, Z_2)$ for all $s', a'$:
$$\left( \mathbb{E}_{S', A'} \left[ \mathcal{W}_p^p(Z_1(S', A'), Z_2(S', A')) \right] \right)^{1/p} \le \bar{\mathcal{W}}_p(Z_1, Z_2)$$
Taking the supremum over all $(s, a) \in \mathcal{S} \times \mathcal{A}$ on the left-hand side:
$$\bar{\mathcal{W}}_p(\mathcal{T}^\pi Z_1, \mathcal{T}^\pi Z_2) \le \gamma \bar{\mathcal{W}}_p(Z_1, Z_2)$$
Because $\gamma < 1$, $\mathcal{T}^\pi$ is a strict $\gamma$-contraction in $\bar{\mathcal{W}}_p$.
By the Banach Fixed-Point Theorem, there exists a **unique fixed-point distribution** $Z^\pi$ such that $\mathcal{T}^\pi Z^\pi = Z^\pi$. $\blacksquare$

---

#### Derivation 11.12.2: First-Principles Derivation of the C51 Categorical Cramer Projection

```
====================================================================================================
DERIVATION 11.12.2: The C51 Categorical Cramer Projection Operator
====================================================================================================
Problem Statement:
Let a point mass of probability p_j be located at continuous coordinate y = r + γ z_j, where
y lies between two consecutive support atoms z_l and z_u (with z_u - z_l = Δz):
    z_l ≤ y ≤ z_u
Prove that the Cramer projection of this point mass onto the categorical support {z_l, z_u}:
    m_l = p_j · (z_u - y) / Δz
    m_u = p_j · (y - z_l) / Δz
is the unique projection that simultaneously:
1. Conserves total probability mass: m_l + m_u = p_j
2. Exactly preserves the expected return: z_l m_l + z_u m_u = y p_j
3. Minimizes the Cramer distance ℓ_2(F_y, F_m) = ∫ (F_y(x) - F_m(x))^2 dx
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Discretized Grid:** Atoms are ordered and equidistant: $z_i = V_{\min} + i \Delta z$.
2. **Local Support:** The point mass coordinate $y \in [z_l, z_u]$, where $u = l + 1$ and $\Delta z = z_u - z_l > 0$.
3. **Probability Conservation:** $m_l, m_u \ge 0$ with $m_l + m_u = p_j$.

**2. Underlying Intuition:**
When projecting a probability distribution onto a fixed grid, one cannot simply round to the nearest atom because doing so shifts the mean of the distribution, corrupting value estimates. A valid projection must preserve both the zeroth moment (total mass $= 1$) and the first moment (expected value). These two linear constraints form a system of 2 equations in 2 unknowns, whose unique solution is precisely the linear interpolation weights.

**3. End-to-End Algebraic Derivation:**

*Step 1: Setting up the moment preservation linear system.*
We wish to find non-negative weights $m_l, m_u$ assigned to atoms $z_l$ and $z_u$ such that:
1. **Zeroth Moment (Total Probability):**
   $$m_l + m_u = p_j \quad \text{(Equation 1)}$$
2. **First Moment (Mean Preservation):**
   $$z_l m_l + z_u m_u = p_j y \quad \text{(Equation 2)}$$

*Step 2: Solving the linear system algebraically.*
From Equation 1, express $m_l$ in terms of $m_u$:
$$m_l = p_j - m_u$$
Substitute $m_l$ into Equation 2:
$$z_l (p_j - m_u) + z_u m_u = p_j y$$
$$z_l p_j - z_l m_u + z_u m_u = p_j y$$
$$(z_u - z_l) m_u = p_j (y - z_l)$$
Recall that $\Delta z = z_u - z_l$. Dividing by $\Delta z$:
$$m_u = p_j \frac{y - z_l}{\Delta z}$$
Now substitute $m_u$ back to find $m_l$:
$$m_l = p_j - p_j \frac{y - z_l}{\Delta z} = p_j \left( 1 - \frac{y - z_l}{\Delta z} \right) = p_j \left( \frac{\Delta z - (y - z_l)}{\Delta z} \right)$$
Since $\Delta z = z_u - z_l$, the numerator is $(z_u - z_l) - (y - z_l) = z_u - y$:
$$m_l = p_j \frac{z_u - y}{\Delta z}$$

*Step 3: Rewriting in normalized index coordinate notation.*
Define the continuous index coordinate $b \triangleq \frac{y - V_{\min}}{\Delta z}$.
Since $z_l = V_{\min} + l \Delta z$ and $z_u = V_{\min} + u \Delta z$:
$$\frac{y - z_l}{\Delta z} = \frac{y - (V_{\min} + l \Delta z)}{\Delta z} = b - l$$
$$\frac{z_u - y}{\Delta z} = \frac{(V_{\min} + u \Delta z) - y}{\Delta z} = u - b$$
Thus:
$$m_l = p_j (u - b), \quad m_u = p_j (b - l) \quad \blacksquare$$

*Step 4: Minimization of the Cramer Distance.*
The Cramer distance between cumulative distributions $F_y(x) = p_j \mathbb{I}(x \ge y)$ and $F_m(x) = m_l \mathbb{I}(x \ge z_l) + (m_l + m_u) \mathbb{I}(x \ge z_u)$ is:
$$\ell_2^2(F_y, F_m) = \int_{-\infty}^\infty (F_y(x) - F_m(x))^2 dx$$
On $(-\infty, z_l)$ and $[z_u, \infty)$, $F_y(x) \equiv F_m(x)$.
On $[z_l, y)$: $F_y(x) = 0$ and $F_m(x) = m_l$.
On $[y, z_u)$: $F_y(x) = p_j$ and $F_m(x) = m_l$.
$$\ell_2^2 = \int_{z_l}^y (0 - m_l)^2 dx + \int_y^{z_u} (p_j - m_l)^2 dx = m_l^2 (y - z_l) + (p_j - m_l)^2 (z_u - y)$$
Differentiating with respect to $m_l$ and setting to zero:
$$\frac{d}{dm_l} \ell_2^2 = 2 m_l (y - z_l) - 2(p_j - m_l)(z_u - y) = 0$$
$$m_l (y - z_l) + m_l (z_u - y) = p_j (z_u - y)$$
$$m_l (z_u - z_l) = p_j (z_u - y) \implies m_l = p_j \frac{z_u - y}{\Delta z}$$
The moment-preserving projection is **identically the optimal minimizer of the $L_2$ Cramer metric**! $\blacksquare$

---

#### Derivation 11.12.3: Quantile Regression and the Asymmetric Pinball Loss for QR-DQN

```
====================================================================================================
DERIVATION 11.12.3: Derivation of the Quantile Pinball Loss Minimizer
====================================================================================================
Problem Statement:
Let Y be a continuous random variable with cumulative distribution function F_Y(y) and density f_Y(y) > 0.
For any target quantile probability τ ∈ (0, 1), define the asymmetric pinball loss:
    ρ_τ(u) ≜ u (τ - 𝕀(u < 0)) = { τ u          if u ≥ 0
                                 { (τ - 1) u      if u < 0
Prove that the parameter θ* that minimizes the expected pinball loss:
    θ* = \arg\min_θ 𝔼_Y [ ρ_τ(Y - θ) ]
is identically the true τ-quantile of Y:
    θ* = F_Y^{-1}(τ)
====================================================================================================
```

**1. Explicit Assumptions:**
1. **Continuous Random Variable:** $Y$ admits a strictly positive probability density function $f_Y(y) > 0$ on its support, so $F_Y(y)$ is strictly monotonically increasing and invertible.
2. **Quantile Level:** $\tau \in (0, 1)$.
3. **Objective:** $\mathcal{J}(\theta) \triangleq \mathbb{E}_Y [ \rho_\tau(Y - \theta) ]$.

**2. Underlying Intuition:**
Mean squared error $\mathbb{E}[(Y - \theta)^2]$ penalizes positive and negative deviations symmetrically, causing the optimal estimate to settle at the mean $\mathbb{E}[Y]$. The pinball loss weights positive errors $(Y > \theta)$ by $\tau$ and negative errors $(Y < \theta)$ by $1 - \tau$. Balancing these opposing asymmetric penalties forces the probability mass below $\theta$ to equal exactly $\tau$.

**3. End-to-End Algebraic Derivation:**

*Step 1: Expanding the expected loss integral.*
Let $u = Y - \theta$. The piecewise pinball loss is:
$$\rho_\tau(Y - \theta) = \begin{cases} \tau (Y - \theta) & \text{if } Y \ge \theta \\ (1 - \tau) (\theta - Y) & \text{if } Y < \theta \end{cases}$$
The expected loss is:
$$\mathcal{J}(\theta) = \int_{-\infty}^\infty \rho_\tau(y - \theta) f_Y(y) \, dy = (1 - \tau) \int_{-\infty}^\theta (\theta - y) f_Y(y) \, dy + \tau \int_\theta^\infty (y - \theta) f_Y(y) \, dy$$

*Step 2: Differentiating $\mathcal{J}(\theta)$ using the Leibniz Integral Rule.*
Recall the Leibniz rule: $\frac{d}{d\theta} \int_{a(\theta)}^{b(\theta)} g(y, \theta) dy = g(b(\theta), \theta) b'(\theta) - g(a(\theta), \theta) a'(\theta) + \int_{a(\theta)}^{b(\theta)} \frac{\partial g}{\partial \theta} dy$.
Differentiating the first integral:
$$\frac{d}{d\theta} \left[ (1 - \tau) \int_{-\infty}^\theta (\theta - y) f_Y(y) \, dy \right] = (1 - \tau) \left[ (\theta - \theta) f_Y(\theta) \cdot (1) + \int_{-\infty}^\theta (1) f_Y(y) \, dy \right]$$
$$= (1 - \tau) \int_{-\infty}^\theta f_Y(y) \, dy = (1 - \tau) F_Y(\theta)$$
Differentiating the second integral:
$$\frac{d}{d\theta} \left[ \tau \int_\theta^\infty (y - \theta) f_Y(y) \, dy \right] = \tau \left[ - (\theta - \theta) f_Y(\theta) \cdot (1) + \int_\theta^\infty (-1) f_Y(y) \, dy \right]$$
$$= -\tau \int_\theta^\infty f_Y(y) \, dy = -\tau (1 - F_Y(\theta))$$

*Step 3: Combining terms and finding the first-order critical point.*
$$\frac{d}{d\theta} \mathcal{J}(\theta) = (1 - \tau) F_Y(\theta) - \tau (1 - F_Y(\theta))$$
Expanding:
$$\frac{d}{d\theta} \mathcal{J}(\theta) = F_Y(\theta) - \tau F_Y(\theta) - \tau + \tau F_Y(\theta) = F_Y(\theta) - \tau$$
Setting the derivative to zero:
$$F_Y(\theta^*) - \tau = 0 \implies F_Y(\theta^*) = \tau$$
Since $F_Y$ is strictly increasing, invert $F_Y$:
$$\theta^* = F_Y^{-1}(\tau) \quad \blacksquare$$

*Step 4: Verification of strict convexity (Second derivative).*
$$\frac{d^2}{d\theta^2} \mathcal{J}(\theta) = \frac{d}{d\theta} \left[ F_Y(\theta) - \tau \right] = f_Y(\theta)$$
Since $f_Y(\theta) > 0$ everywhere, $\frac{d^2}{d\theta^2} \mathcal{J}(\theta) > 0$.
The objective is strictly convex, and $\theta^* = F_Y^{-1}(\tau)$ is the **unique global minimum**. In QR-DQN, setting $\tau_i = \frac{2i - 1}{2N}$ mathematically guarantees that SGD converges to the true quantiles of the return distribution! $\blacksquare$

---

## 3. Geometric & Physical Interpretation


### The "Bucket Pouring" Projection Geometry
Imagine the support atoms as $N$ buckets placed along a line at positions $z_0, z_1, \dots, z_{N-1}$.
1. The transition $(r, \gamma)$ slides the contents of bucket $j$ to a new coordinate $\hat{T} z_j = r + \gamma z_j$.
2. This new point lands between bucket $l$ and bucket $u$.
3. The Cramer projection behaves like **lever mechanics** or **bilinear interpolation**: the mass $p_j$ is split between the two enclosing buckets inversely proportional to their Euclidean distance from $\hat{T} z_j$.

```
           z_l                                z_u
          Bucket                             Bucket
            |                                  |
            +------------*---------------------+
                       T(z_j)
            <--- u - b ---> <----- b - l ----->
            (goes to z_l)     (goes to z_u)
```

---

## 4. Real-World Analogy: Investment Portfolio Assessment

Imagine evaluating two investment strategies:
- **Strategy A:** Puts all money into government treasury bonds.
  Returns: Guaranteed $+5\%$ with probability $1.0$. Mean $= +5\%$.
- **Strategy B:** Puts money into tech venture capital.
  Returns: $+50\%$ with probability $0.2$, $0\%$ with probability $0.5$, and $-50\%$ with probability $0.3$. Mean $= (0.2)(50) + (0.5)(0) + (0.3)(-50) = 10 - 15 = -5\%$.

Standard scalar RL only sees $-5\%$ vs $+5\%$ and discards Strategy B. But what if the investor is a hedge fund seeking asymmetric upside exposure? Distributional RL maintains the complete return histogram, allowing the decision maker to price upside convexity and downside ruin probabilities!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 3-Atom Categorical Projection ($N = 3$)
To observe the exact projection arithmetic down to every single decimal digit, consider a minimal 3-atom support:
- Support Range: $V_{\min} = 0.0000, \quad V_{\max} = 10.0000$
- Number of atoms: $N = 3$
- Atom Step Size:
  $$\Delta z = \frac{V_{\max} - V_{\min}}{N - 1} = \frac{10.0000 - 0.0000}{3 - 1} = \frac{10.0000}{2} = \mathbf{5.0000}$$
- Atoms:
  $$z_0 = 0.0000, \quad z_1 = 5.0000, \quad z_2 = 10.0000$$

**Observed Transition:**
- Immediate reward: $r = 1.0000$
- Discount factor: $\gamma = 0.8000$
- Target distribution at next state $S'$ for greedy action $a^*$:
  $$\mathbf{p}' = [p_0' = 0.2000, \quad p_1' = 0.5000, \quad p_2' = 0.3000]$$

We will compute:
1. Shifted Bellman atom locations $\hat{T} z_j$
2. Linear projection fractions onto $\{z_0, z_1, z_2\}$
3. Final projected probability mass vector $\mathbf{m} = [m_0, m_1, m_2]$
4. Cross-entropy loss against a current uniform prediction $\mathbf{p} = [1/3, 1/3, 1/3]$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Walkthrough Concrete Value |
| :--- | :--- | :--- |
| $z_j$ | Fixed Support Atoms | $\{0.0000, 5.0000, 10.0000\}$ |
| $\Delta z$ | Support Grid Spacing | $5.0000$ |
| $p_j'$ | Next-State Target Probabilities | $p_0'=0.20, p_1'=0.50, p_2'=0.30$ |
| $\hat{T} z_j$ | Shifted Atom Locations | $r + \gamma z_j = 1.0 + 0.8 z_j$ |
| $b_j$ | Continuous Support Coordinate | $\frac{\hat{T} z_j - V_{\min}}{\Delta z}$ |
| $l, u$ | Lower and Upper Atom Indices | $l = \lfloor b_j \rfloor, u = \lceil b_j \rceil$ |
| $m_i$ | Projected Target Probability | Accumulator for atom $z_i$ |

---

### 5.3 Step-by-Step Hand Calculations: C51 Categorical Projection

#### Atom 0 ($z_0 = 0.0000, p_0' = 0.2000$):
1. Compute shifted location:
   $$\hat{T} z_0 = r + \gamma z_0 = 1.0000 + 0.8000 \times 0.0000 = \mathbf{1.0000}$$
2. Coordinate:
   $$b_0 = \frac{1.0000 - 0.0000}{5.0000} = \mathbf{0.2000}$$
   Lower index $l = \lfloor 0.2000 \rfloor = 0$, Upper index $u = \lceil 0.2000 \rceil = 1$.
3. Distribute probability mass $p_0' = 0.2000$:
   - Weight to atom $l = 0$: $u - b_0 = 1.0000 - 0.2000 = \mathbf{0.8000}$
     $$\Delta m_0 = p_0' \times (u - b_0) = 0.2000 \times 0.8000 = \mathbf{0.1600}$$
   - Weight to atom $u = 1$: $b_0 - l = 0.2000 - 0.0000 = \mathbf{0.2000}$
     $$\Delta m_1 = p_0' \times (b_0 - l) = 0.2000 \times 0.2000 = \mathbf{0.0400}$$

#### Atom 1 ($z_1 = 5.0000, p_1' = 0.5000$):
1. Compute shifted location:
   $$\hat{T} z_1 = r + \gamma z_1 = 1.0000 + 0.8000 \times 5.0000 = 1.0000 + 4.0000 = \mathbf{5.0000}$$
2. Coordinate:
   $$b_1 = \frac{5.0000 - 0.0000}{5.0000} = \mathbf{1.0000}$$
   $l = 1, u = 1$ (exact integer match!).
3. Distribute probability mass $p_1' = 0.5000$:
   - All mass lands directly on atom $1$:
     $$\Delta m_1 = p_1' = \mathbf{0.5000}$$

#### Atom 2 ($z_2 = 10.0000, p_2' = 0.3000$):
1. Compute shifted location:
   $$\hat{T} z_2 = r + \gamma z_2 = 1.0000 + 0.8000 \times 10.0000 = 1.0000 + 8.0000 = \mathbf{9.0000}$$
2. Coordinate:
   $$b_2 = \frac{9.0000 - 0.0000}{5.0000} = \mathbf{1.8000}$$
   Lower index $l = \lfloor 1.8000 \rfloor = 1$, Upper index $u = \lceil 1.8000 \rceil = 2$.
3. Distribute probability mass $p_2' = 0.3000$:
   - Weight to atom $l = 1$: $u - b_2 = 2.0000 - 1.8000 = \mathbf{0.2000}$
     $$\Delta m_1 = p_2' \times (u - b_2) = 0.3000 \times 0.2000 = \mathbf{0.0600}$$
   - Weight to atom $u = 2$: $b_2 - l = 1.8000 - 1.0000 = \mathbf{0.8000}$
     $$\Delta m_2 = p_2' \times (b_2 - l) = 0.3000 \times 0.8000 = \mathbf{0.2400}$$

---

### 5.4 Summing Total Projected Distribution $\mathbf{m}$

Accumulate total mass into each bucket:
- **Atom $z_0 = 0.0000$:**
  $$m_0 = \Delta m_0 (\text{from atom 0}) = \mathbf{0.1600}$$
- **Atom $z_1 = 5.0000$:**
  $$m_1 = \Delta m_1 (\text{from atom 0}) + \Delta m_1 (\text{from atom 1}) + \Delta m_1 (\text{from atom 2})$$
  $$m_1 = 0.0400 + 0.5000 + 0.0600 = \mathbf{0.6000}$$
- **Atom $z_2 = 10.0000$:**
  $$m_2 = \Delta m_2 (\text{from atom 2}) = \mathbf{0.2400}$$

**Final Projected Target Distribution Vector:**
$$\mathbf{m} = \begin{bmatrix} 0.1600 \\ 0.6000 \\ 0.2400 \end{bmatrix}$$

**Probability Mass Conservation Check:**
$$\sum_{i=0}^2 m_i = 0.1600 + 0.6000 + 0.2400 = \mathbf{1.0000} \quad \checkmark$$

**Expected Return of Projected Distribution:**
$$\mathbb{E}[\mathbf{m}] = \sum_{i=0}^2 z_i m_i = (0.0 \times 0.16) + (5.0 \times 0.60) + (10.0 \times 0.24) = 0.0 + 3.00 + 2.40 = \mathbf{5.4000}$$
Notice that this matches the scalar Bellman expectation:
$$r + \gamma \mathbb{E}[\mathbf{p}'] = 1.0 + 0.8 \times (0.0 \times 0.2 + 5.0 \times 0.5 + 10.0 \times 0.3) = 1.0 + 0.8 \times (0.0 + 2.5 + 3.0) = 1.0 + 0.8(5.5) = 1.0 + 4.4 = \mathbf{5.4000}!$$
The Cramer projection preserves the **exact mean**!

---

### 5.5 Visual Grid: Projection Matrix Walkthrough

| Atom $j$ | $z_j$ | $p'_j$ | Shifted $\hat{T} z_j$ | Coord $b_j$ | Enclosing Atoms | Mass to $z_0$ | Mass to $z_1$ | Mass to $z_2$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $0.0$ | $0.2000$ | $1.0000$ | $0.2000$ | $z_0, z_1$ | $\mathbf{0.1600}$ | $\mathbf{0.0400}$ | $0.0000$ |
| **1** | $5.0$ | $0.5000$ | $5.0000$ | $1.0000$ | $z_1$ | $0.0000$ | $\mathbf{0.5000}$ | $0.0000$ |
| **2** | $10.0$ | $0.3000$ | $9.0000$ | $1.8000$ | $z_1, z_2$ | $0.0000$ | $\mathbf{0.0600}$ | $\mathbf{0.2400}$ |
| **Total** | - | $1.0000$ | - | - | - | **$m_0 = 0.1600$** | **$m_1 = 0.6000$** | **$m_2 = 0.2400$** |

---

### 5.6 Step 5: Cross-Entropy Loss Calculation

Suppose the current online network predicts a uniform distribution:
$$\mathbf{p} = \begin{bmatrix} 1/3 \\ 1/3 \\ 1/3 \end{bmatrix} \approx \begin{bmatrix} 0.3333 \\ 0.3333 \\ 0.3333 \end{bmatrix}$$

The Cross-Entropy Loss is:
$$\mathcal{L} = - \sum_{i=0}^2 m_i \ln(p_i) = - \left[ 0.1600 \ln(1/3) + 0.6000 \ln(1/3) + 0.2400 \ln(1/3) \right]$$
$$= - \ln(1/3) \sum_{i=0}^2 m_i = \ln(3) \times 1.0000 \approx \mathbf{1.0986}$$

---

## 6. Solved Illustrations

### Illustration 1: Preserving Bimodality vs. Mean Collapse
**Problem:**
Consider a lottery transition: with probability $0.5$, reward is $+10$; with probability $0.5$, reward is $-10$. Assume $\gamma = 0$.
Compare the target representation learned by:
1. Standard DQN
2. C51 Distributional DQN
**Solution:**
- **Standard DQN:**
  Computes the scalar mean:
  $$Y = 0.5(+10) + 0.5(-10) = \mathbf{0.00}$$
  The network learns that the state has value $0$, giving the illusion that the state is neutral and risk-free.
- **C51 Distributional DQN:**
  Learns a bimodal distribution placing probability mass $0.5$ at atom $+10$ and $0.5$ at atom $-10$. The variance $\sigma^2 = 100$ is completely preserved, allowing risk-sensitive downstream agents to avoid this state if variance-averse! $\blacksquare$

---

### Illustration 2: 5-Atom C51 Projection on Out-of-Bounds Transitions (Boundary Clipping)

**Problem:**
Consider a C51 agent configured with a 5-atom support:
$$V_{\min} = 0.0000, \quad V_{\max} = 10.0000, \quad N = 5 \implies \Delta z = \frac{10.0 - 0.0}{5 - 1} = \mathbf{2.5000}$$
Fixed atoms: $\mathbf{z} = [0.0000, 2.5000, 5.0000, 7.5000, 10.0000]^\top$.
The agent encounters a high-reward transition:
$$r = 4.0000, \quad \gamma = 0.9000$$
The next-state target distribution is concentrated in upper atoms:
$$\mathbf{p}' = [0.0000, 0.0000, 0.0000, 0.4000, 0.6000]^\top$$
1. Compute the shifted locations $\hat{T} z_j$ for atoms $3$ and $4$.
2. Demonstrate how support boundary clipping operates.
3. Compute the final projected target distribution $\mathbf{m} \in \mathbb{R}^5$ and its expected return $\mathbb{E}[\mathbf{m}]$.

**Solution:**

*Step 1: Compute shifted atom coordinates.*
- **Atom 3 ($z_3 = 7.5000, p_3' = 0.4000$):**
  $$\hat{T} z_3 = r + \gamma z_3 = 4.0000 + 0.9000 \times 7.5000 = 4.0000 + 6.7500 = \mathbf{10.7500}$$
- **Atom 4 ($z_4 = 10.0000, p_4' = 0.6000$):**
  $$\hat{T} z_4 = r + \gamma z_4 = 4.0000 + 0.9000 \times 10.0000 = 4.0000 + 9.0000 = \mathbf{13.0000}$$

*Step 2: Apply support clipping to $[V_{\min}, V_{\max}] = [0.0, 10.0]$.*
Because both shifted coordinates exceed $V_{\max} = 10.0000$:
$$\hat{z}_3 = \operatorname{clip}(10.7500, 0.0, 10.0) = \mathbf{10.0000}$$
$$\hat{z}_4 = \operatorname{clip}(13.0000, 0.0, 10.0) = \mathbf{10.0000}$$

Fractional indices:
$$b_3 = \frac{\hat{z}_3 - V_{\min}}{\Delta z} = \frac{10.0000 - 0.0}{2.5000} = \mathbf{4.0000} \implies l_3 = 4, u_3 = 4$$
$$b_4 = \frac{\hat{z}_4 - V_{\min}}{\Delta z} = \frac{10.0000 - 0.0}{2.5000} = \mathbf{4.0000} \implies l_4 = 4, u_4 = 4$$

*Step 3: Distribute probability mass into target vector $\mathbf{m}$.*
Since $l_3 = u_3 = 4$ and $l_4 = u_4 = 4$, all probability mass from both atoms collapses into the final boundary atom $z_4 = 10.0000$:
- From atom 3: $\Delta m_4 = 0.4000$
- From atom 4: $\Delta m_4 = 0.6000$
Total target vector:
$$\mathbf{m} = \begin{bmatrix} 0.0000 \\ 0.0000 \\ 0.0000 \\ 0.0000 \\ 0.4000 + 0.6000 \end{bmatrix} = \begin{bmatrix} 0.0000 \\ 0.0000 \\ 0.0000 \\ 0.0000 \\ \mathbf{1.0000} \end{bmatrix}$$

*Step 4: Expected value comparison.*
- Unclipped true target return:
  $$\mathbb{E}[\text{unclipped}] = r + \gamma \mathbb{E}[\mathbf{p}'] = 4.0 + 0.90(7.5 \times 0.4 + 10.0 \times 0.6) = 4.0 + 0.90(3.0 + 6.0) = 4.0 + 8.10 = \mathbf{12.1000}$$
- Clipped projected return:
  $$\mathbb{E}[\mathbf{m}] = 10.0000 \times 1.0000 = \mathbf{10.0000}$$
When returns exceed $[V_{\min}, V_{\max}]$, C51 truncates the tail at $V_{\max}$. This highlights why setting an appropriate support range $[V_{\min}, V_{\max}]$ is crucial in categorical algorithms. $\blacksquare$

---

### Illustration 3: QR-DQN 4-Quantile Forward and Backward Pass with Quantile Huber Loss

**Problem:**
A QR-DQN agent uses $N = 4$ quantiles to model returns.
Target cumulative probabilities are:
$$\tau_i = \frac{2i - 1}{2N} \implies \tau = [0.1250, \ 0.3750, \ 0.6250, \ 0.8750]^\top$$
The current online network estimates quantile locations:
$$\theta = [\theta_1 = 1.0000, \quad \theta_2 = 3.0000, \quad \theta_3 = 5.0000, \quad \theta_4 = 7.0000]^\top$$
A transition yields target return sample $Y = 4.0000$.
Let the Huber threshold be $\kappa = 1.0000$, and learning rate $\alpha = 0.5000$.
1. Compute the TD error residues $u_i = Y - \theta_i$.
2. Compute the Quantile Huber loss $\rho_{\tau_i}^\kappa(u_i)$ for each quantile.
3. Compute the parameter gradients $\nabla_{\theta_i} \mathcal{L}$ and the updated quantile positions $\theta_{\text{new}}$.

**Solution:**

*Step 1: Compute error residues.*
$$u_1 = Y - \theta_1 = 4.0000 - 1.0000 = \mathbf{+3.0000}$$
$$u_2 = Y - \theta_2 = 4.0000 - 3.0000 = \mathbf{+1.0000}$$
$$u_3 = Y - \theta_3 = 4.0000 - 5.0000 = \mathbf{-1.0000}$$
$$u_4 = Y - \theta_4 = 4.0000 - 7.0000 = \mathbf{-3.0000}$$

*Step 2: Compute Huber Loss $\mathcal{L}_\kappa(u)$.*
With $\kappa = 1.0000$:
- For $u_1 = 3.0$ ($|u| > 1$): $\mathcal{L}_\kappa(u_1) = |3.0| - 0.5 = \mathbf{2.5000}$
- For $u_2 = 1.0$ ($|u| \le 1$): $\mathcal{L}_\kappa(u_2) = 0.5(1.0)^2 = \mathbf{0.5000}$
- For $u_3 = -1.0$ ($|u| \le 1$): $\mathcal{L}_\kappa(u_3) = 0.5(-1.0)^2 = \mathbf{0.5000}$
- For $u_4 = -3.0$ ($|u| > 1$): $\mathcal{L}_\kappa(u_4) = |-3.0| - 0.5 = \mathbf{2.5000}$

*Step 3: Asymmetric quantile weights $w_i = |\tau_i - \mathbb{I}(u_i < 0)|$.*
- For $i = 1$ ($u_1 = +3.0 \ge 0$): $w_1 = |\tau_1 - 0| = |0.1250 - 0| = \mathbf{0.1250}$
- For $i = 2$ ($u_2 = +1.0 \ge 0$): $w_2 = |\tau_2 - 0| = |0.3750 - 0| = \mathbf{0.3750}$
- For $i = 3$ ($u_3 = -1.0 < 0$): $w_3 = |\tau_3 - 1| = |0.6250 - 1| = \mathbf{0.3750}$
- For $i = 4$ ($u_4 = -3.0 < 0$): $w_4 = |\tau_4 - 1| = |0.8750 - 1| = \mathbf{0.1250}$

*Step 4: Compute Quantile Huber Loss $\rho_{\tau_i}^\kappa = w_i \mathcal{L}_\kappa(u_i)$.*
- $\rho_1 = 0.1250 \times 2.5000 = \mathbf{0.3125}$
- $\rho_2 = 0.3750 \times 0.5000 = \mathbf{0.1875}$
- $\rho_3 = 0.3750 \times 0.5000 = \mathbf{0.1875}$
- $\rho_4 = 0.1250 \times 2.5000 = \mathbf{0.3125}$
Mean loss:
$$\mathcal{L} = \frac{0.3125 + 0.1875 + 0.1875 + 0.3125}{4} = \frac{1.0000}{4} = \mathbf{0.2500}$$

*Step 5: Gradient calculation and parameter update.*
Derivative with respect to parameter $\theta_i$:
$$\frac{\partial \rho_{\tau_i}^\kappa}{\partial \theta_i} = - w_i \nabla_u \mathcal{L}_\kappa(u_i) = - w_i \operatorname{clip}(u_i, -1, 1)$$
- For $\theta_1$: $\frac{\partial \rho}{\partial \theta_1} = -0.1250 \times \operatorname{clip}(3.0, -1, 1) = -0.1250(1.0) = \mathbf{-0.1250}$
- For $\theta_2$: $\frac{\partial \rho}{\partial \theta_2} = -0.3750 \times \operatorname{clip}(1.0, -1, 1) = -0.3750(1.0) = \mathbf{-0.3750}$
- For $\theta_3$: $\frac{\partial \rho}{\partial \theta_3} = -0.3750 \times \operatorname{clip}(-1.0, -1, 1) = -0.3750(-1.0) = \mathbf{+0.3750}$
- For $\theta_4$: $\frac{\partial \rho}{\partial \theta_4} = -0.1250 \times \operatorname{clip}(-3.0, -1, 1) = -0.1250(-1.0) = \mathbf{+0.1250}$

Parameter updates ($\theta \leftarrow \theta - \alpha \nabla_\theta \rho$ with $\alpha = 0.50$):
$$\theta_{1, \text{new}} = 1.0000 - 0.50(-0.1250) = 1.0000 + 0.0625 = \mathbf{1.0625}$$
$$\theta_{2, \text{new}} = 3.0000 - 0.50(-0.3750) = 3.0000 + 0.1875 = \mathbf{3.1875}$$
$$\theta_{3, \text{new}} = 5.0000 - 0.50(+0.3750) = 5.0000 - 0.1875 = \mathbf{4.8125}$$
$$\theta_{4, \text{new}} = 7.0000 - 0.50(+0.1250) = 7.0000 - 0.0625 = \mathbf{6.9375}$$
Notice that quantiles below $Y = 4.0$ ($\theta_1, \theta_2$) are pulled upward, while quantiles above $Y = 4.0$ ($\theta_3, \theta_4$) are pulled downward, contractively squeezing around the target return! $\blacksquare$

---

### Illustration 4: 1-Wasserstein Distance (Earth Mover's Distance) Computation

**Problem:**
Let return distributions $P$ and $Q$ be supported on discrete atoms $\{0.0, 2.0, 4.0, 6.0\}$:
$$\mathbf{p} = [0.4000, 0.1000, 0.3000, 0.2000]^\top$$
$$\mathbf{q} = [0.1000, 0.4000, 0.2000, 0.3000]^\top$$
1. Construct the Cumulative Distribution Functions $F_P(x)$ and $F_Q(x)$.
2. Calculate the 1-Wasserstein distance $\mathcal{W}_1(P, Q) = \int_0^6 |F_P(x) - F_Q(x)| \, dx$ by hand.

**Solution:**

*Step 1: Construct CDFs.*
The CDFs are piecewise constant step functions:
- On $[0.0, 2.0)$:
  $$F_P(x) = p_0 = \mathbf{0.4000}, \quad F_Q(x) = q_0 = \mathbf{0.1000}$$
- On $[2.0, 4.0)$:
  $$F_P(x) = p_0 + p_1 = 0.40 + 0.10 = \mathbf{0.5000}, \quad F_Q(x) = q_0 + q_1 = 0.10 + 0.40 = \mathbf{0.5000}$$
- On $[4.0, 6.0)$:
  $$F_P(x) = 0.50 + 0.30 = \mathbf{0.8000}, \quad F_Q(x) = 0.50 + 0.20 = \mathbf{0.7000}$$
- For $x \ge 6.0$:
  $$F_P(x) = 1.0000, \quad F_Q(x) = 1.0000$$

*Step 2: Compute the integral over each sub-interval.*
$$\mathcal{W}_1(P, Q) = \int_0^2 |F_P(x) - F_Q(x)| \, dx + \int_2^4 |F_P(x) - F_Q(x)| \, dx + \int_4^6 |F_P(x) - F_Q(x)| \, dx$$
- **Interval 1 ($[0, 2]$, width $\Delta x = 2.0$):**
  $$|F_P - F_Q| = |0.4000 - 0.1000| = 0.3000 \implies I_1 = 0.3000 \times 2.0 = \mathbf{0.6000}$$
- **Interval 2 ($[2, 4]$, width $\Delta x = 2.0$):**
  $$|F_P - F_Q| = |0.5000 - 0.5000| = 0.0000 \implies I_2 = 0.0000 \times 2.0 = \mathbf{0.0000}$$
- **Interval 3 ($[4, 6]$, width $\Delta x = 2.0$):**
  $$|F_P - F_Q| = |0.8000 - 0.7000| = 0.1000 \implies I_3 = 0.1000 \times 2.0 = \mathbf{0.2000}$$

*Step 3: Total Wasserstein Distance.*
$$\mathcal{W}_1(P, Q) = I_1 + I_2 + I_3 = 0.6000 + 0.0000 + 0.2000 = \mathbf{0.8000}$$
The minimum work required to transport probability mass from distribution $P$ to distribution $Q$ is exactly $0.8000$. $\blacksquare$

---

### Illustration 5: Risk-Sensitive Decision Making via Conditional Value-at-Risk (CVaR)

**Problem:**
An autonomous vehicle faces an intersection and evaluates two candidate actions:
- **Action A (Wait safely for clear gap):**
  Yields returns $\{4.0, 5.0, 6.0\}$ with probabilities $\mathbf{p}_A = [0.20, 0.60, 0.20]^\top$.
- **Action B (Aggressive unprotected turn):**
  Yields returns $\{-10.0, 6.0, 10.0\}$ with probabilities $\mathbf{p}_B = [0.10, 0.50, 0.40]^\top$, where $-10.0$ represents a catastrophic near-collision event.
1. Compute the expected value $\mathbb{E}[Z]$ for both actions. Which action is chosen by a standard risk-neutral DQN agent?
2. Compute the Value at Risk $\operatorname{VaR}_{0.10}$ and Conditional Value at Risk $\operatorname{CVaR}_{0.10}$ (expected value of the worst $10\%$ outcomes) for both actions.
3. Determine which action is selected by a risk-averse safety agent maximizing $\operatorname{CVaR}_{0.10}$.

**Solution:**

*Step 1: Risk-Neutral Expected Value.*
- **Action A:**
  $$\mathbb{E}[Z_A] = 0.20(4.0) + 0.60(5.0) + 0.20(6.0) = 0.80 + 3.00 + 1.20 = \mathbf{5.0000}$$
- **Action B:**
  $$\mathbb{E}[Z_B] = 0.10(-10.0) + 0.50(6.0) + 0.40(10.0) = -1.00 + 3.00 + 4.00 = \mathbf{6.0000}$$
**Risk-Neutral Decision:** Standard DQN only optimizes expected return:
$$\mathbb{E}[Z_B] = 6.0000 > \mathbb{E}[Z_A] = 5.0000 \implies \text{Chooses Action B (Dangerous!)}$$

*Step 2: Risk-Averse Evaluation via $\operatorname{CVaR}_\alpha$.*
The Conditional Value at Risk at level $\alpha \in (0, 1]$ is:
$$\operatorname{CVaR}_\alpha(Z) \triangleq \frac{1}{\alpha} \int_0^\alpha F_Z^{-1}(u) \, du$$
For $\alpha = 0.10$:
- **For Action A:**
  The lowest atom $z = 4.0000$ carries probability $p_0 = 0.20 \ge 0.10$.
  For all $u \in [0, 0.10]$, $F_{Z_A}^{-1}(u) = 4.0000$.
  $$\operatorname{CVaR}_{0.10}(Z_A) = \frac{1}{0.10} \int_0^{0.10} 4.0000 \, du = \mathbf{+4.0000}$$
- **For Action B:**
  The lowest atom $z = -10.0000$ carries probability $p_0 = 0.10$, exactly occupying the lowest $10\%$ quantile interval $[0, 0.10]$.
  For all $u \in [0, 0.10]$, $F_{Z_B}^{-1}(u) = -10.0000$.
  $$\operatorname{CVaR}_{0.10}(Z_B) = \frac{1}{0.10} \int_0^{0.10} (-10.0000) \, du = \mathbf{-10.0000}$$

*Step 3: Risk-Averse Decision.*
$$\operatorname{CVaR}_{0.10}(Z_A) = \mathbf{+4.0000} \gg \operatorname{CVaR}_{0.10}(Z_B) = \mathbf{-10.0000}$$
The risk-averse distributional agent decisively rejects Action B and chooses **Action A**, eliminating the risk of catastrophic collision! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. Agent57 & Atari Super-Human Benchmarks (Badia et al., Nature 2020)
QR-DQN distributional returns were a critical component in achieving the first super-human agent across all 57 Atari games:
- **Transformed Bellman Operators:** Agent57 applies the elementwise transform $h(z) = \text{sign}(z)(\sqrt{|z| + 1} - 1) + \epsilon z$ to compress the unbounded return distribution, enabling stable value learning across games with vastly different reward scales (from Pong's $\pm 1$ to Montezuma's Revenge's thousands).
- **Per-Game Exploration:** A bandit meta-controller selects per-game $(\epsilon, \beta)$ exploration parameters, directing QR-DQN to learn game-specific value distributions without manual tuning.

### 2. Autonomous Driving & Robotics: Risk-Sensitive Control
Distributional RL uniquely enables **risk-sensitive decision-making**, going beyond expected value:
- **Waymo & Autonomous Drones:** Planning algorithms evaluate the **Conditional Value at Risk (CVaR$_\alpha$)**—the expected return in the worst $\alpha$-fraction of outcomes. Setting $\alpha = 0.05$ ensures the controller explicitly minimizes rare catastrophic collision events that would be averaged away under standard $\mathbb{E}[Q]$ optimization.
- **Medical Robotics (Surgical Robots):** Distributional critics evaluate the full distribution of tissue damage risk, enforcing that both the median and 95th-percentile force estimates remain below safe thresholds during autonomous suturing.

### 3. LLM Reasoning: Distributional Value Estimation for Chain-of-Thought
- **DeepSeek-Math & Process Reward Models:** Token-level critics in mathematical reasoning models maintain distributional value estimates across chain-of-thought token sequences. Instead of scalar $V(s)$, they learn $Z(s)$—the full distribution of final-answer correctness given partial solutions—allowing the rollout controller to distinguish between a reliably correct partial proof and a high-variance lucky guess.
- **OpenAI o1 Inference Scaling:** Distributional estimates of reasoning path quality guide best-of-N sampling strategies, weighting candidate solutions by their variance-adjusted expected correctness rather than raw predicted reward.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of the Part 5 C51 Cramer projection:
   - Target distribution $\mathbf{m} = [0.1600, 0.6000, 0.2400]$
   - Conservation of mass: $\sum m_i = 1.0000$
   - Exact mean conservation: $\mathbb{E}[\mathbf{m}] = 5.4000$
   - Cross-entropy loss matching machine precision to $< 10^{-14}$.
2. Quantile Regression loss function (Pinball Huber loss) verification for QR-DQN.
3. Bimodal vs. unimodal value distribution benchmark.

See implementation in:
[`11_reinforcement_learning/code/12_distributional_reinforcement_learning.py`](./code/12_distributional_reinforcement_learning.py)
