# Chapter 5.1: Convex Sets, Convex Functions & Jensen's Inequality

---

## Part 1: Intuition & 101 Motivation

In 1993, the mathematician R. Tyrrell Rockafellar made one of the most famous observations in the history of applied mathematics:
> *"The great watershed in optimization isn't between linearity and non-linearity, but between **convexity** and **non-convexity**."*

Why is convexity so celebrated?
In generic non-convex optimization, an algorithm can easily become trapped in a disastrous local minimum, a flat saddle point, or a plateau with zero gradient. You can never be certain if your model has reached the best possible solution.

In **Convex Optimization**, the mathematical landscape is uniquely well-behaved:
1. **Every local minimum is guaranteed to be a global minimum.**
2. If the function is strictly convex, the global minimizer is **unique**.
3. Stationarity ($\nabla f(x) = \mathbf{0}$) is both a **necessary and sufficient** condition for global optimality.

```
       Non-Convex Landscape                      Convex Landscape
          (The Egg Crate)                         (The Soup Bowl)
    f(x) ▲                                  f(x) ▲
         │    /\      /\                         │  \             /
         │   /  \    /  \   Local Minima         │   \           /   Global Minimum
         │  /    \  /    \      ▼                │    \    ●    /          ▼
         │ /      \/      \____●____             │     \       /       ┌───┴───┐
         └──────────────────────────► x          └──────┴─────┴────────►   ●   │
                                                                       └───────┘
```

While deep neural networks have non-convex loss surfaces, modern deep learning is thoroughly grounded in convexity:
- The building blocks of classical ML (Logistic Regression, Linear Regression, Support Vector Machines) are strictly convex.
- Training layers in deep networks (e.g. cross-entropy loss with respect to final logits) are locally convex.
- **Jensen's Inequality** is the foundational theorem used to derive the **Evidence Lower Bound (ELBO)** in Variational Autoencoders (VAEs), prove the non-negativity of KL Divergence, and derive the EM algorithm.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Convex Sets
A set $\mathcal{C} \subseteq \mathbb{R}^n$ is **convex** if the line segment connecting any two points in $\mathcal{C}$ lies entirely within $\mathcal{C}$:
$$\forall x, y \in \mathcal{C}, \quad \forall \theta \in [0, 1]: \quad \theta x + (1 - \theta) y \in \mathcal{C}$$

#### Canonical Examples of Convex Sets:
1. **Hyperplanes and Halfspaces:** $\mathcal{H} = \{x \in \mathbb{R}^n \mid a^T x = b\}$ and $\mathcal{H}_- = \{x \mid a^T x \le b\}$.
2. **Euclidean Norm Balls:** $B_r(x_0) = \{x \in \mathbb{R}^n \mid \|x - x_0\|_2 \le r\}$.
3. **The Probability Simplex $\Delta^K$:**
   $$\Delta^K = \left\{ p \in \mathbb{R}^K \,\,\middle|\,\, \sum_{k=1}^K p_k = 1, \,\, p_k \ge 0 \,\, \forall k \right\}$$
4. **The Cone of Positive Semi-Definite Matrices $\mathbb{S}_+^n$:**
   $$\mathbb{S}_+^n = \{X \in \mathbb{R}^{n \times n} \mid X = X^T, \,\, z^T X z \ge 0 \,\, \forall z \in \mathbb{R}^n\}$$

#### Convexity-Preserving Operations:
- The **intersection** of any arbitrary (even infinite) collection of convex sets is convex: $\mathcal{C} = \bigcap_{i \in \mathcal{I}} \mathcal{C}_i$.
- Affine transformations $f(x) = A x + b$ preserve convexity of sets (both forward image and pre-image).

---

### 2. Convex Functions
Let $\mathcal{C} \subseteq \mathbb{R}^n$ be a convex set. A function $f: \mathcal{C} \to \mathbb{R}$ is **convex** if for all $x, y \in \mathcal{C}$ and all $\theta \in [0, 1]$:
$$\mathbf{f(\theta x + (1 - \theta) y) \le \theta f(x) + (1 - \theta) f(y)}$$

- **Strictly Convex:** If the inequality is strict for all $x \ne y$ and $\theta \in (0, 1)$. (Guarantees at most one unique global minimum).
- **Concave Function:** $f$ is concave if $-f$ is convex:
  $$f(\theta x + (1 - \theta) y) \ge \theta f(x) + (1 - \theta) f(y)$$

---

### 3. First-Order Characterization of Convexity
Let $f: \mathcal{C} \to \mathbb{R}$ be continuously differentiable ($\mathcal{C}^1$) on an open convex set $\mathcal{C}$.
**Theorem:** $f$ is convex if and only if:
$$\mathbf{f(y) \ge f(x) + \nabla f(x)^T (y - x) \quad \forall x, y \in \mathcal{C}}$$

*Geometric Meaning:*
The first-order Taylor expansion $T_1(y) = f(x) + \nabla f(x)^T (y - x)$ forms a **global lower supporting affine hyperplane** to the graph of $f$ everywhere!
A local linear approximation never overestimates a convex function.

#### Corollary (Stationarity $\implies$ Global Minimum):
If $\nabla f(x^*) = \mathbf{0}$, then for all $y \in \mathcal{C}$:
$$f(y) \ge f(x^*) + \mathbf{0}^T (y - x^*) = f(x^*)$$
Hence, **any stationary point of a differentiable convex function is an absolute global minimum!**

---

### 4. Second-Order Characterization of Convexity
Let $f: \mathcal{C} \to \mathbb{R}$ be twice continuously differentiable ($\mathcal{C}^2$).
**Theorem:** $f$ is convex if and only if its Hessian matrix is **Positive Semi-Definite (PSD)** everywhere:
$$\mathbf{\nabla^2 f(x) \succeq 0 \quad \forall x \in \mathcal{C}}$$
that is, $z^T \nabla^2 f(x) z \ge 0$ for all $z \in \mathbb{R}^n$.
- If $\nabla^2 f(x) \succ 0$ (strictly positive definite, eigenvalues $\lambda_i > 0$) everywhere, then $f$ is **strictly convex**.

---

### 5. Strong Convexity and Lipschitz Smoothness
In optimization theory, we often bound the curvature of $f(x)$ from both below and above:

1. **$\mu$-Strongly Convex:**
   Curvature is bounded below by a positive quadratic:
   $$f(y) \ge f(x) + \nabla f(x)^T (y - x) + \frac{\mu}{2} \|y - x\|_2^2 \iff \nabla^2 f(x) \succeq \mu I$$
2. **$L$-Lipschitz Smooth:**
   The gradient $\nabla f$ does not change too rapidly (curvature is bounded above):
   $$\|\nabla f(y) - \nabla f(x)\|_2 \le L \|y - x\|_2 \iff \nabla^2 f(x) \preceq L I$$
   $$f(y) \le f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|_2^2$$

#### The Condition Number $\kappa$:
$$\kappa = \frac{L}{\mu} \ge 1$$
The condition number $\kappa$ is the ratio of maximum to minimum curvature.
- If $\kappa \approx 1$, the loss surface contours are circular spheres, and gradient descent converges at lightning speed.
- If $\kappa \gg 1$, the loss surface forms an elongated, steep ravine (a "canyon"), causing gradient descent to oscillate wildly back and forth!

---

### 6. Jensen's Inequality
Jensen's Inequality is arguably the most powerful inequality in probability and machine learning.

#### Theorem (Jensen's Inequality):
Let $X$ be a random variable taking values in convex domain $\mathcal{C} \subseteq \mathbb{R}^n$.
If $g: \mathcal{C} \to \mathbb{R}$ is a **convex function**, then:
$$\mathbf{g(\mathbb{E}[X]) \le \mathbb{E}[g(X)]}$$
provided expectations exist.
If $g$ is **strictly convex**, equality holds if and only if $X$ is a degenerate constant almost surely ($P(X = c) = 1$).

#### Concave Form of Jensen's Inequality:
If $h: \mathcal{C} \to \mathbb{R}$ is a **concave function** (such as $\ln(x)$ or $\sqrt{x}$):
$$\mathbf{h(\mathbb{E}[X]) \ge \mathbb{E}[h(X)]}$$

#### Finite Form (Weighted Convex Combinations):
For points $x_1, \dots, x_N \in \mathcal{C}$ and weights $\lambda_i \ge 0$ with $\sum_{i=1}^N \lambda_i = 1$:
$$g\left( \sum_{i=1}^N \lambda_i x_i \right) \le \sum_{i=1}^N \lambda_i g(x_i)$$

---

### Deep Derivation 5.1.1: Complete Proof of the First-Order Characterization of Convexity

We prove both directions of the theorem: A continuously differentiable function $f: \mathcal{C} \to \mathbb{R}$ on an open convex set $\mathcal{C}$ is convex if and only if $f(y) \ge f(x) + \nabla f(x)^T (y - x)$ for all $x, y \in \mathcal{C}$.

#### Direction 1: $f$ is convex $\implies f(y) \ge f(x) + \nabla f(x)^T (y - x)$
1. Let $x, y \in \mathcal{C}$. Since $\mathcal{C}$ is convex, the point $x + \theta(y - x) = (1 - \theta)x + \theta y \in \mathcal{C}$ for all $\theta \in (0, 1]$.
2. By the definition of convexity:
   $$f(x + \theta(y - x)) \le (1 - \theta) f(x) + \theta f(y) = f(x) + \theta(f(y) - f(x))$$
3. Subtract $f(x)$ from both sides and divide by $\theta > 0$:
   $$\frac{f(x + \theta(y - x)) - f(x)}{\theta} \le f(y) - f(x)$$
4. Take the limit as $\theta \to 0^+$. By definition of the directional derivative of a differentiable function:
   $$\lim_{\theta \to 0^+} \frac{f(x + \theta(y - x)) - f(x)}{\theta} = \nabla f(x)^T (y - x)$$
5. Therefore:
   $$\nabla f(x)^T (y - x) \le f(y) - f(x) \implies \mathbf{f(y) \ge f(x) + \nabla f(x)^T (y - x)}$$

#### Direction 2: $f(y) \ge f(x) + \nabla f(x)^T (y - x) \,\, \forall x, y \in \mathcal{C} \implies f$ is convex
1. Choose arbitrary $x, y \in \mathcal{C}$ and $\theta \in [0, 1]$.
2. Define the convex combination $z = \theta x + (1 - \theta) y \in \mathcal{C}$.
3. Apply the first-order condition at base point $z$ evaluated at $x$:
   $$f(x) \ge f(z) + \nabla f(z)^T (x - z)$$
4. Apply the first-order condition at base point $z$ evaluated at $y$:
   $$f(y) \ge f(z) + \nabla f(z)^T (y - z)$$
5. Multiply the first inequality by $\theta \ge 0$ and the second by $(1 - \theta) \ge 0$, and sum them:
   $$\theta f(x) + (1 - \theta) f(y) \ge \theta [f(z) + \nabla f(z)^T (x - z)] + (1 - \theta) [f(z) + \nabla f(z)^T (y - z)]$$
   $$= f(z) + \nabla f(z)^T \left[ \theta(x - z) + (1 - \theta)(y - z) \right]$$
6. Examine the bracketed vector term:
   $$\theta(x - z) + (1 - \theta)(y - z) = \theta x + (1 - \theta) y - (\theta + 1 - \theta) z = z - z = \mathbf{0}$$
7. The gradient term vanishes completely!
   $$\mathbf{\theta f(x) + (1 - \theta) f(y) \ge f(z) = f(\theta x + (1 - \theta) y)} \quad \blacksquare$$

---

### Deep Derivation 5.1.2: General Proof of Jensen's Inequality via Supporting Hyperplanes

We provide the modern variational proof of Jensen's Inequality, valid for any arbitrary random variable $X$ (discrete, continuous, or mixed) taking values in convex set $\mathcal{C} \subseteq \mathbb{R}^n$.

#### 1. Setup
Let $X \in \mathcal{C}$ with finite expectation $\mu = \mathbb{E}[X] \in \mathcal{C}$.
Let $g: \mathcal{C} \to \mathbb{R}$ be a convex function.

#### 2. Existence of Supporting Hyperplane (Subgradient)
Because $g$ is convex on $\mathcal{C}$, at any point $\mu \in \mathcal{C}$ there exists at least one supporting affine hyperplane (characterized by subgradient vector $\mathbf{s} \in \partial g(\mu)$):
$$g(x) \ge g(\mu) + \mathbf{s}^T (x - \mu) \quad \forall x \in \mathcal{C}$$
(If $g$ is differentiable, $\mathbf{s} = \nabla g(\mu)$ is the unique gradient).

#### 3. Evaluation on the Random Variable
Since $X$ takes values in $\mathcal{C}$ with probability 1, this inequality holds pointwise for the random variable $X$:
$$g(X) \ge g(\mu) + \mathbf{s}^T (X - \mu)$$

#### 4. Taking Expectations
Take the mathematical expectation $\mathbb{E}[\cdot]$ of both sides:
$$\mathbb{E}[g(X)] \ge \mathbb{E}\left[ g(\mu) + \mathbf{s}^T (X - \mu) \right]$$
By linearity of expectation, and noting that $g(\mu)$, $\mathbf{s}$, and $\mu$ are deterministic constants:
$$\mathbb{E}[g(X)] \ge g(\mu) + \mathbf{s}^T (\mathbb{E}[X] - \mu)$$
Since $\mu = \mathbb{E}[X]$, the deviation vector is identically zero:
$$\mathbb{E}[X] - \mu = \mu - \mu = \mathbf{0}$$
Therefore:
$$\mathbb{E}[g(X)] \ge g(\mu) + \mathbf{s}^T \mathbf{0} = g(\mu) = g(\mathbb{E}[X])$$
$$\mathbf{g(\mathbb{E}[X]) \le \mathbb{E}[g(X)]} \quad \blacksquare$$

*Significance:* This geometric proof avoids induction on discrete mixtures and holds unconditionally for high-dimensional continuous probability measures!

---

### Deep Derivation 5.1.3: Linear Convergence Contraction of Gradient Descent

We derive the exact contraction factor $\rho = \frac{\kappa - 1}{\kappa + 1}$ for Gradient Descent on $\mu$-strongly convex and $L$-smooth objectives.

#### 1. Setup
Let $f: \mathbb{R}^n \to \mathbb{R}$ be $\mu$-strongly convex and $L$-smooth ($0 < \mu \le L$).
Let $x^*$ be the unique global minimizer satisfying $\nabla f(x^*) = \mathbf{0}$.
Gradient descent performs the update:
$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

#### 2. Distance to Optimum Expansion
Evaluate the squared Euclidean distance to $x^*$:
$$\|x_{t+1} - x^*\|_2^2 = \|x_t - x^* - \eta \nabla f(x_t)\|_2^2 = \|x_t - x^*\|_2^2 - 2\eta \nabla f(x_t)^T (x_t - x^*) + \eta^2 \|\nabla f(x_t)\|_2^2$$

#### 3. Co-coercivity / Interpolation Inequality
For a function that is simultaneously $\mu$-strongly convex and $L$-smooth, Nesterov's theorem establishes the co-coercivity bound:
$$\nabla f(x_t)^T (x_t - x^*) \ge \frac{\mu L}{\mu + L} \|x_t - x^*\|_2^2 + \frac{1}{\mu + L} \|\nabla f(x_t)\|_2^2$$
*(Proof sketch: The shifted function $\phi(x) = f(x) - \frac{\mu}{2} \|x - x^*\|_2^2$ is $(L - \mu)$-smooth and convex; applying the standard cocoercivity theorem to $\nabla \phi$ yields this inequality).*

#### 4. Substituting into Distance Bound
Substitute this bound into the squared distance equation:
$$\|x_{t+1} - x^*\|_2^2 \le \|x_t - x^*\|_2^2 - 2\eta \left[ \frac{\mu L}{\mu + L} \|x_t - x^*\|_2^2 + \frac{1}{\mu + L} \|\nabla f(x_t)\|_2^2 \right] + \eta^2 \|\nabla f(x_t)\|_2^2$$
$$= \left( 1 - \frac{2\eta \mu L}{\mu + L} \right) \|x_t - x^*\|_2^2 + \eta \left( \eta - \frac{2}{\mu + L} \right) \|\nabla f(x_t)\|_2^2$$

#### 5. Optimal Learning Rate Choice
To eliminate the gradient norm $\|\nabla f(x_t)\|_2^2$ (which is non-negative and could otherwise hurt the bound), choose the optimal step size:
$$\mathbf{\eta^* = \frac{2}{\mu + L}}$$
At this step size, the gradient term $\eta(\eta - \frac{2}{\mu + L}) = 0$, leaving:
$$\|x_{t+1} - x^*\|_2^2 \le \left( 1 - \frac{2 \cdot \frac{2}{\mu + L} \mu L}{\mu + L} \right) \|x_t - x^*\|_2^2 = \left( 1 - \frac{4\mu L}{(\mu + L)^2} \right) \|x_t - x^*\|_2^2$$
Simplify the algebraic contraction factor:
$$1 - \frac{4\mu L}{(\mu + L)^2} = \frac{(\mu + L)^2 - 4\mu L}{(\mu + L)^2} = \frac{\mu^2 + 2\mu L + L^2 - 4\mu L}{(\mu + L)^2} = \frac{(L - \mu)^2}{(L + \mu)^2}$$

Taking the square root:
$$\mathbf{\|x_{t+1} - x^*\|_2 \le \left( \frac{L - \mu}{L + \mu} \right) \|x_t - x^*\|_2 = \left( \frac{\kappa - 1}{\kappa + 1} \right) \|x_t - x^*\|_2} \quad \blacksquare$$

---

## Part 3: Geometric & Algebraic Interpretation

### 1. The Epigraph Characterization
The **Epigraph** of a function $f: \mathcal{C} \to \mathbb{R}$ is the set of points lying on or above its graph:
$$\text{epi}(f) = \left\{ (x, t) \in \mathbb{R}^{n+1} \,\,\middle|\,\, x \in \mathcal{C}, \,\, t \ge f(x) \right\}$$

**Fundamental Equivalence:**
A function $f$ is convex if and only if its epigraph $\text{epi}(f)$ is a **convex set** in $\mathbb{R}^{n+1}$.
This allows any convex optimization problem $\min f(x)$ to be recast geometrically as finding the lowest point of a convex volume!

```
                  t ▲
                    │          epi(f)
                    │       (Convex Set)
                    │       \         /
                    │        \       /
                    │         \_____/ ◄── f(x) Graph Boundary
                    └───────────┴────────► x
```

---

### 2. The Chord vs. Curve Geometry
For any convex function $f$, the straight secant line (chord) connecting any two points $(x, f(x))$ and $(y, f(y))$ lies **entirely above or on top of the curve** for all intermediate points:
$$\text{Secant Line } \theta f(x) + (1 - \theta) f(y) \ge f(\theta x + (1 - \theta) y) \quad \forall \theta \in [0, 1]$$

---

## Part 4: Real-World Analogy

### The Ceramic Bowl vs. The Mountain Range
- **Convex Optimization (The Ceramic Soup Bowl):**
  Imagine placing a frictionless steel ball on the lip of a smooth, perfectly curved ceramic soup bowl.
  No matter which point on the rim you drop the ball from, gravity forces it along a clean path directly to the exact bottom of the bowl. There are no side ledges, no traps, no confusion.
- **Non-Convex Optimization (The Rocky Mountain Range):**
  Now imagine rolling the ball down a jagged mountain range in dense fog. It might settle in a shallow mud puddle 5,000 feet above the valley floor. To an algorithm that only checks if "every local step goes uphill," that shallow puddle looks identical to the true global bottom of the valley!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us verify **Jensen's Inequality** and the **First-Order Convexity Condition** by hand with concrete numbers.

### 1. Problem Setup & Toy Distribution
- Convex function: $f(x) = x^2$ (First derivative $f'(x) = 2x$, second derivative $f''(x) = 2 > 0 \implies$ strictly convex everywhere).
- Discrete Random Variable $X$ taking $N = 3$ values:
  $$x_1 = 1.0, \quad x_2 = 3.0, \quad x_3 = 6.0$$
- Probability weights $\lambda = [\lambda_1, \lambda_2, \lambda_3]$:
  $$\lambda_1 = 0.50, \quad \lambda_2 = 0.30, \quad \lambda_3 = 0.20 \quad \left( \sum \lambda_i = 1.0 \right)$$

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $x_i$ | Discrete state realization | Sample latent embedding / loss | Float scalar | $1.0, 3.0, 6.0$ |
| $\lambda_i$ | Probability mass $P(X = x_i)$ | Softmax attention weight $\alpha_i$ | Float scalar | $0.50, 0.30, 0.20$ |
| $\mathbb{E}[X]$ | Expected value of $X$ | Attention context vector $\sum \alpha_i v_i$ | Float scalar | $2.6000$ |
| $f(\mathbb{E}[X])$ | Convex function of expectation | Loss evaluated at average context | Float scalar | $2.6^2 = 6.7600$ |
| $f(x_i)$ | Convex function at states | Individual loss per token / sample | Float scalar | $1.0, 9.0, 36.0$ |
| $\mathbb{E}[f(X)]$ | Expectation of convex function | Expected individual loss | Float scalar | $10.4000$ |
| $\Delta_{\text{Jensen}}$ | Jensen's Gap: $\mathbb{E}[f(X)] - f(\mathbb{E}[X])$ | Variational lower bound slack | Float scalar | $3.6400 \ge 0$ |
| $\text{Var}(X)$ | Variance of $X$ | Intrinsic token / state variance | Float scalar | $3.6400$ (Identical!) |

---

### 3. Step-by-Step Manual Arithmetic

#### Step 1: Compute the Expectation $\mathbb{E}[X]$
$$\mathbb{E}[X] = \sum_{i=1}^3 \lambda_i x_i = (0.50)(1.0) + (0.30)(3.0) + (0.20)(6.0) = 0.50 + 0.90 + 1.20 = \mathbf{2.6000}$$

#### Step 2: Compute $f(\mathbb{E}[X])$ (Function of the Expectation)
$$f(\mathbb{E}[X]) = (\mathbb{E}[X])^2 = 2.6000^2 = \mathbf{6.7600}$$

#### Step 3: Compute $\mathbb{E}[f(X)]$ (Expectation of the Function)
- State 1: $f(x_1) = 1.0^2 = 1.0 \implies \lambda_1 f(x_1) = 0.50 \times 1.0 = \mathbf{0.5000}$
- State 2: $f(x_2) = 3.0^2 = 9.0 \implies \lambda_2 f(x_2) = 0.30 \times 9.0 = \mathbf{2.7000}$
- State 3: $f(x_3) = 6.0^2 = 36.0 \implies \lambda_3 f(x_3) = 0.20 \times 36.0 = \mathbf{7.2000}$
$$\mathbb{E}[f(X)] = \sum_{i=1}^3 \lambda_i f(x_i) = 0.5000 + 2.7000 + 7.2000 = \mathbf{10.4000}$$

#### Step 4: Verify Jensen's Inequality
$$\mathbf{f(\mathbb{E}[X]) = 6.7600 \le \mathbb{E}[f(X)] = 10.4000} \quad \checkmark$$
The Jensen Gap:
$$\Delta_{\text{Jensen}} = \mathbb{E}[f(X)] - f(\mathbb{E}[X]) = 10.4000 - 6.7600 = \mathbf{3.6400} > 0$$

*Deep Mathematical Insight:*
For $f(x) = x^2$, the Jensen gap is **identically equal to the variance of $X$**:
$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = 10.4000 - 6.7600 = \mathbf{3.6400}!$$
Jensen's gap measures the exact dispersion (variance) of the probability distribution!

---

#### Step 5: Verify First-Order Tangent Under-estimator at $x = 2.0$
Let base point be $x_0 = 2.0$.
Function value: $f(x_0) = 2.0^2 = 4.0$.
Gradient: $f'(x_0) = 2(2.0) = 4.0$.
Tangent line supporting hyper-plane:
$$T(y) = f(x_0) + f'(x_0)(y - x_0) = 4.0 + 4.0(y - 2.0) = 4y - 4.0$$

Test at arbitrary points $y \in \{0.0, 1.0, 2.0, 3.0, 4.0\}$:
- At $y = 0.0$: $f(0) = 0.0 \ge T(0) = 4(0) - 4 = \mathbf{-4.0}$ ($\Delta = +4.0$)
- At $y = 1.0$: $f(1) = 1.0 \ge T(1) = 4(1) - 4 = \mathbf{0.0}$ ($\Delta = +1.0$)
- At $y = 2.0$: $f(2) = 4.0 = T(2) = 4(2) - 4 = \mathbf{4.0}$ ($\Delta = 0.0$ — Tangent touchpoint!)
- At $y = 3.0$: $f(3) = 9.0 \ge T(3) = 4(3) - 4 = \mathbf{8.0}$ ($\Delta = +1.0$)
- At $y = 4.0$: $f(4) = 16.0 \ge T(4) = 4(4) - 4 = \mathbf{12.0}$ ($\Delta = +4.0$)
Notice: $f(y) - T(y) = y^2 - 4y + 4 = (y - 2)^2 \ge 0$ everywhere!

---

### 4. Visual Summary Grid

```
┌──────┬──────────┬──────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ i    │ State x_i│ Prob λ_i │ Product λ·x │ f(x_i) = x² │ Product λ·f │ Tangent T(x)│
├──────┼──────────┼──────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ 1    │   1.00   │   0.50   │    0.5000   │    1.0000   │    0.5000   │    0.0000   │
│ 2    │   3.00   │   0.30   │    0.9000   │    9.0000   │    2.7000   │    8.0000   │
│ 3    │   6.00   │   0.20   │    1.2000   │   36.0000   │    7.2000   │   20.0000   │
├──────┼──────────┼──────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ SUM  │    —     │ Σ = 1.00 │ E[X] = 2.60 │     —       │ E[f(X)]=10.4│     —       │
├──────┴──────────┴──────────┴─────────────┴─────────────┴─────────────┴─────────────┤
│ f(E[X]) = (2.6000)²                         │   6.7600                             │
│ E[f(X)] = 0.5(1) + 0.3(9) + 0.2(36)         │  10.4000                             │
│ Jensen's Inequality: f(E[X]) <= E[f(X)]     │   6.7600 <= 10.4000 (Holds strictly!)│
│ Jensen Gap: E[f(X)] - f(E[X]) = Var(X)      │   3.6400                             │
└─────────────────────────────────────────────┴──────────────────────────────────────┘
```

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Rigorous Derivation of the ELBO in VAEs
In a Variational Autoencoder, we wish to maximize the marginal log-likelihood of data $\ln p_\theta(x)$:
$$\ln p_\theta(x) = \ln \int p_\theta(x, z) \, dz = \ln \int q_\phi(z \mid x) \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \, dz = \ln \mathbb{E}_{z \sim q_\phi(z \mid x)}\left[ \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right]$$
The logarithm function $h(u) = \ln(u)$ is **strictly concave** ($h''(u) = -1/u^2 < 0$).
Applying the **Concave Form of Jensen's Inequality** ($\ln(\mathbb{E}[U]) \ge \mathbb{E}[\ln(U)]$):
$$\ln \mathbb{E}_{z \sim q_\phi(z \mid x)}\left[ \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] \ge \mathbb{E}_{z \sim q_\phi(z \mid x)}\left[ \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] = \mathbf{\mathcal{L}_{\text{ELBO}}(\theta, \phi; x)} \quad \blacksquare$$
Jensen's inequality guarantees that $\mathcal{L}_{\text{ELBO}} \le \ln p(x)$ everywhere, providing a rigorous variational lower bound!

---

### Illustration 2 (Boundary): Arithmetic Mean - Geometric Mean (AM-GM) Proof
For any positive real numbers $a_1, a_2, \dots, a_N > 0$:
$$\frac{a_1 + a_2 + \dots + a_N}{N} \ge \sqrt[N]{a_1 a_2 \dots a_N}$$
**Proof via Jensen:**
Consider the convex function $g(u) = -\ln(u)$. Let $X$ take values $a_i$ with uniform weights $\lambda_i = 1/N$.
$$g\left( \frac{1}{N}\sum_{i=1}^N a_i \right) \le \frac{1}{N}\sum_{i=1}^N g(a_i) \implies -\ln\left( \frac{1}{N}\sum_{i=1}^N a_i \right) \le -\frac{1}{N}\sum_{i=1}^N \ln(a_i) = -\ln\left( \prod_{i=1}^N a_i^{1/N} \right)$$
Multiply by $-1$ (flipping inequality):
$$\ln\left( \frac{1}{N}\sum_{i=1}^N a_i \right) \ge \ln\left( \left(\prod_{i=1}^N a_i\right)^{1/N} \right) \implies \frac{1}{N}\sum_{i=1}^N a_i \ge \left(\prod_{i=1}^N a_i\right)^{1/N} \quad \blacksquare$$

---

### Illustration 3 (Edge Case): Non-Convexity in Neural Networks via Permutation Symmetries
Why can't a neural network loss function ever be strictly convex?
Consider a two-layer neural network with $H$ hidden units:
$$f(x; W_1, W_2) = W_2 \sigma(W_1 x)$$
Suppose we permute the hidden units by a permutation matrix $\Pi \in \mathbb{R}^{H \times H}$:
$$\tilde{W}_1 = \Pi W_1, \quad \tilde{W}_2 = W_2 \Pi^T$$
The network output is mathematically identical:
$$\tilde{W}_2 \sigma(\tilde{W}_1 x) = W_2 \Pi^T \sigma(\Pi W_1 x) = W_2 \Pi^T \Pi \sigma(W_1 x) = W_2 \sigma(W_1 x)$$
Hence, the loss is identical: $\mathcal{L}(\theta) = \mathcal{L}(\tilde{\theta})$.
For $H$ hidden neurons, there are $H!$ distinct parameter vectors with the exact same loss!
If the loss were strictly convex, it could have at most one global minimum.
Because there are at least $H!$ distinct global minima, **neural network parameter spaces are fundamentally non-convex**!

---

### Illustration 4 (Numerical): Hessian Eigenvalues, Strong Convexity $\mu$, Lipschitz Constant $L$, and GD Contraction

Let us calculate the exact conditioning and convergence rate of Gradient Descent on a 2D quadratic convex objective.

#### 1. Problem Formulation
$$f(x_1, x_2) = 3 x_1^2 + x_2^2 + 2 x_1 x_2 - 4 x_1 - 2 x_2$$

#### 2. Gradient and Global Minimizer
$$\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 6 x_1 + 2 x_2 - 4 \\ 2 x_1 + 2 x_2 - 2 \end{bmatrix}$$
Setting $\nabla f(x^*) = \mathbf{0}$:
$$\begin{cases} 3 x_1^* + x_2^* = 2 \\ x_1^* + x_2^* = 1 \end{cases} \implies 2 x_1^* = 1 \implies x_1^* = 0.50, \quad x_2^* = 0.50$$
$$\mathbf{x^* = \begin{bmatrix} 0.50 \\ 0.50 \end{bmatrix}}, \quad f(x^*) = 3(0.25) + 0.25 + 2(0.25) - 4(0.5) - 2(0.5) = \mathbf{-1.5000}$$

#### 3. Hessian Matrix and Curvature Spectrum
$$H = \nabla^2 f(x) = \begin{bmatrix} 6 & 2 \\ 2 & 2 \end{bmatrix}$$
Compute the eigenvalues via the characteristic equation $\det(H - \lambda I) = 0$:
$$\det \begin{bmatrix} 6 - \lambda & 2 \\ 2 & 2 - \lambda \end{bmatrix} = (6 - \lambda)(2 - \lambda) - 4 = \lambda^2 - 8\lambda + 8 = 0$$
$$\lambda = \frac{8 \pm \sqrt{64 - 32}}{2} = \frac{8 \pm 4\sqrt{2}}{2} = 4 \pm 2\sqrt{2}$$
- Maximum eigenvalue (Lipschitz constant): $\mathbf{L = 4 + 2\sqrt{2} \approx 6.828427}$
- Minimum eigenvalue (Strong convexity parameter): $\mathbf{\mu = 4 - 2\sqrt{2} \approx 1.171573}$
Both eigenvalues are strictly positive $\implies H \succ 0$ (strictly convex everywhere).

#### 4. Condition Number and Theoretical Contraction Factor
$$\kappa = \frac{L}{\mu} = \frac{4 + 2\sqrt{2}}{4 - 2\sqrt{2}} = \frac{(4 + 2\sqrt{2})^2}{16 - 8} = \frac{24 + 16\sqrt{2}}{8} = \mathbf{3 + 2\sqrt{2} \approx 5.828427}$$
Optimal learning rate:
$$\mathbf{\eta^* = \frac{2}{\mu + L} = \frac{2}{8.0000} = 0.2500}$$
Theoretical per-step contraction factor:
$$\mathbf{\rho = \frac{L - \mu}{L + \mu} = \frac{4\sqrt{2}}{8} = \frac{\sqrt{2}}{2} \approx 0.707107}$$

#### 5. Step-by-Step Manual Gradient Descent Simulation
Start from the origin $\mathbf{x}_0 = [0.0, 0.0]^T$:
Initial distance: $\|\mathbf{x}_0 - \mathbf{x}^*\|_2 = \sqrt{0.5^2 + 0.5^2} = \sqrt{0.50} \approx \mathbf{0.707107}$.

- **Iteration 1:**
  $$\nabla f(\mathbf{x}_0) = \begin{bmatrix} -4.00 \\ -2.00 \end{bmatrix}$$
  $$\mathbf{x}_1 = \mathbf{x}_0 - \eta^* \nabla f(\mathbf{x}_0) = \begin{bmatrix} 0.00 \\ 0.00 \end{bmatrix} - 0.25 \begin{bmatrix} -4.00 \\ -2.00 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.00 \\ 0.50 \end{bmatrix}}$$
  Error vector: $\mathbf{x}_1 - \mathbf{x}^* = [1.00 - 0.50, \, 0.50 - 0.50]^T = [0.50, \, 0.00]^T$.
  Distance: $\|\mathbf{x}_1 - \mathbf{x}^*\|_2 = \mathbf{0.500000}$.
  Contraction ratio: $\frac{0.500000}{0.707107} \approx \mathbf{0.707107} = \rho \quad \checkmark$

- **Iteration 2:**
  $$\nabla f(\mathbf{x}_1) = \begin{bmatrix} 6(1.0) + 2(0.5) - 4 \\ 2(1.0) + 2(0.5) - 2 \end{bmatrix} = \begin{bmatrix} 3.00 \\ 1.00 \end{bmatrix}$$
  $$\mathbf{x}_2 = \mathbf{x}_1 - \eta^* \nabla f(\mathbf{x}_1) = \begin{bmatrix} 1.00 \\ 0.50 \end{bmatrix} - 0.25 \begin{bmatrix} 3.00 \\ 1.00 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.25 \\ 0.25 \end{bmatrix}}$$
  Error vector: $\mathbf{x}_2 - \mathbf{x}^* = [0.25 - 0.50, \, 0.25 - 0.50]^T = [-0.25, \, -0.25]^T$.
  Distance: $\|\mathbf{x}_2 - \mathbf{x}^*\|_2 = \sqrt{(-0.25)^2 + (-0.25)^2} = \sqrt{0.125} \approx \mathbf{0.353553}$.
  Contraction ratio: $\frac{0.353553}{0.500000} \approx \mathbf{0.707107} = \rho \quad \checkmark$

The observed error contracts by **exactly $\rho = 0.707107$ per step**, verifying the theoretical bound with zero slack!

---

### Illustration 5 (Numerical): Jensen's Inequality for Cross-Entropy Loss & Gibbs' Inequality

In classification, we train networks by minimizing the Cross-Entropy loss between ground-truth distribution $P$ and predicted distribution $Q$. Gibbs' inequality states that $D_{\text{KL}}(P \parallel Q) \ge 0$, with equality if and only if $P = Q$.

#### 1. First-Principles Derivation via Jensen's Inequality
Let $P$ and $Q$ be probability distributions over discrete alphabet $\mathcal{X}$.
$$-D_{\text{KL}}(P \parallel Q) = -\sum_{x \in \mathcal{X}} P(x) \ln \frac{P(x)}{Q(x)} = \sum_{x \in \mathcal{X}} P(x) \ln \frac{Q(x)}{P(x)} = \mathbb{E}_{X \sim P}\left[ \ln \frac{Q(X)}{P(X)} \right]$$

Since $h(u) = \ln(u)$ is strictly concave, apply the concave form of Jensen's inequality:
$$\mathbb{E}_{X \sim P}\left[ \ln \frac{Q(X)}{P(X)} \right] \le \ln\left( \mathbb{E}_{X \sim P}\left[ \frac{Q(X)}{P(X)} \right] \right) = \ln\left( \sum_{x \in \mathcal{X}} P(x) \frac{Q(x)}{P(x)} \right) = \ln\left( \sum_{x \in \mathcal{X}} Q(x) \right)$$
Since $Q$ is a valid probability distribution, $\sum_{x} Q(x) = 1$:
$$\ln(1) = 0 \implies -D_{\text{KL}}(P \parallel Q) \le 0 \implies \mathbf{D_{\text{KL}}(P \parallel Q) \ge 0} \quad \blacksquare$$

#### 2. Numerical Walkthrough with Concrete Probabilities
Let binary true distribution be $P = [0.70, 0.30]$ and model prediction be $Q = [0.50, 0.50]$ (uncertain uniform prediction):

- **Class 1 ($x = 1$):**
  $$P(1) \ln \frac{P(1)}{Q(1)} = 0.70 \ln\left( \frac{0.70}{0.50} \right) = 0.70 \ln(1.40) \approx 0.70(0.336472) = \mathbf{+0.235530}$$
- **Class 2 ($x = 2$):**
  $$P(2) \ln \frac{P(2)}{Q(2)} = 0.30 \ln\left( \frac{0.30}{0.50} \right) = 0.30 \ln(0.60) \approx 0.30(-0.510826) = \mathbf{-0.153248}$$

Summing both classes yields the KL divergence:
$$\mathbf{D_{\text{KL}}(P \parallel Q) = 0.235530 - 0.153248 = 0.082282 \ge 0} \quad \checkmark$$

#### 3. Impact on Cross-Entropy Loss
- **Shannon Entropy of Target:**
  $$H(P) = -0.70 \ln(0.70) - 0.30 \ln(0.30) = -0.70(-0.356675) - 0.30(-1.203973) = 0.249672 + 0.361192 = \mathbf{0.610864 \text{ nats}}$$
- **Cross-Entropy Loss:**
  $$H(P, Q) = -\sum_{x} P(x) \ln Q(x) = -0.70 \ln(0.50) - 0.30 \ln(0.50) = -\ln(0.50) = \mathbf{0.693147 \text{ nats}}$$
- **Decomposition Check:**
  $$H(P) + D_{\text{KL}}(P \parallel Q) = 0.610864 + 0.082282 = \mathbf{0.693146 \approx H(P, Q)}$$

By Jensen's inequality, Cross-Entropy loss is strictly lower-bounded by the data entropy $H(P)$; the extra $+0.0823$ nats is the exact Jensen penalty paid for the model's incorrect uniform hypothesis!

---

## Part 7: Deep Learning Connection & Application

### 1. Convergence Rate of Gradient Descent on Convex Objectives
For a $\mu$-strongly convex and $L$-smooth loss function $\mathcal{L}(\theta)$, Gradient Descent with step size $\eta = \frac{2}{\mu + L}$ converges **linearly (exponentially fast)**:
$$\|\theta_t - \theta^*\|_2 \le \left( \frac{L - \mu}{L + \mu} \right)^t \|\theta_0 - \theta^*\|_2 = \left( \frac{\kappa - 1}{\kappa + 1} \right)^t \|\theta_0 - \theta^*\|_2$$
- If $\kappa = L/\mu = 1$ (perfect spherical bowl), $\frac{\kappa - 1}{\kappa + 1} = 0$, converging in **a single step**!
- If $\kappa = 1000$ (ill-conditioned ravine), $\frac{\kappa - 1}{\kappa + 1} \approx 0.998$, requiring thousands of slow, zig-zagging steps.

### 2. Why Normalization Layers (BatchNorm, LayerNorm) Speed Up Training
Feature unnormalization creates pathological condition numbers ($\kappa \gg 10^4$).
By standardizing internal activations to zero mean and unit variance, **Batch Normalization and Layer Normalization precondition the local Hessian**, squashing the condition number $\kappa \approx 1$ and turning distorted ravines into isotropic convex-like bowls!

---

## Part 8: Code Implementation & Verification

The companion Python module [01_convexity_and_jensens_inequality.py](./code/01_convexity_and_jensens_inequality.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact calculations of $\mathbb{E}[X] = 2.60$, $f(\mathbb{E}[X]) = 6.76$, $\mathbb{E}[f(X)] = 10.40$, and verifies that the Jensen gap equals $\text{Var}(X) = 3.6400$.
2. **First-Order Convexity Lower Bound Test:** Verifies $f(y) \ge f(x) + \nabla f(x)^T (y - x)$ across 10,000 random point pairs for convex functions vs. violation for non-convex functions.
3. **AM-GM Inequality Monte Carlo Test:** Validates that Arithmetic Mean $\ge$ Geometric Mean across random positive vectors.
4. **VAE ELBO Variational Gap Verification:** Numerically computes the Jensen gap between true log-marginal $\ln p(x)$ and $\mathcal{L}_{\text{ELBO}}$, verifying $\mathcal{L}_{\text{ELBO}} \le \ln p(x)$ with slack equal to $D_{\text{KL}}(q(z \mid x) \parallel p(z \mid x))$.
5. **Permutation Symmetry Non-Convexity Proof in PyTorch:** Implements an MLP, permutes hidden layer weights, and verifies that the midpoint parameter vector has higher loss than the permuted endpoints!
