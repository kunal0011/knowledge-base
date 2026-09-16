# Chapter 1.8: Quadratic Forms & Positive Semi-Definite (PSD) Matrices

---

## Part 1: Intuition & 101 Motivation

In deep learning, we constantly optimize loss functions $\mathcal{L}(\theta)$.
Around any stationary point (where the gradient $\nabla \mathcal{L} = \mathbf{0}$), the local geometry of the loss landscape is dominated by its second-order Taylor expansion:
$$\mathcal{L}(\theta + \Delta \theta) \approx \mathcal{L}(\theta) + \frac{1}{2} \Delta \theta^T H \Delta \theta$$
where $H = \nabla^2 \mathcal{L}$ is the symmetric **Hessian matrix**, and the scalar expression $q(\Delta \theta) = \Delta \theta^T H \Delta \theta$ is a **Quadratic Form**.

The sign of this quadratic form decides the fate of our optimization:
- Does the surface curve upwards in all directions like a bowl, guaranteeing a **local minimum**?
- Does it curve downwards, forming a **local maximum**?
- Or does it curve upwards in some directions and downwards in others, creating a treacherous **saddle point**?

Beyond optimization, quadratic forms govern uncertainty:
- The probability density of a **Multivariate Gaussian Distribution** is driven by the quadratic form $(x - \mu)^T \Sigma^{-1} (x - \mu)$.
- For $\Sigma$ to represent a physically valid covariance matrix, its quadratic form must never be negative—it must be **Positive Semi-Definite (PSD)**.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Definition of a Quadratic Form
Let $A \in \mathbb{R}^{n \times n}$ be an $n \times n$ matrix, and let $x \in \mathbb{R}^n$ be a vector. The scalar function:
$$q(x) = x^T A x = \sum_{i=1}^n \sum_{j=1}^n A_{ij} x_i x_j$$
is called the **Quadratic Form** associated with matrix $A$.

#### The Symmetry Principle:
Any square matrix $A$ can be decomposed into symmetric and skew-symmetric parts:
$$A = \frac{A + A^T}{2} + \frac{A - A^T}{2} = A_{\text{sym}} + A_{\text{skew}}$$

#### First-Principles Derivation / Proof: $x^T A_{\text{skew}} x \equiv 0$
Let $K = A_{\text{skew}} \in \mathbb{R}^{n \times n}$ be any skew-symmetric matrix, satisfying $K^T = -K$.
Let $x \in \mathbb{R}^n$ be any vector.
The quadratic expression $s = x^T K x$ is a scalar ($1 \times 1$ matrix).
Taking the transpose of a scalar yields the identical scalar:
$$s = s^T = (x^T K x)^T = x^T K^T (x^T)^T = x^T K^T x$$
Substitute the defining skew-symmetry condition $K^T = -K$:
$$s = x^T (-K) x = - (x^T K x) = -s$$
$$s = -s \implies 2s = 0 \implies s = 0$$
Therefore, for any matrix $A$:
$$x^T A x = x^T (A_{\text{sym}} + A_{\text{skew}}) x = x^T A_{\text{sym}} x + x^T A_{\text{skew}} x = x^T A_{\text{sym}} x + 0 \equiv x^T A_{\text{sym}} x \quad \blacksquare$$

> [!NOTE]
> Throughout machine learning and optimization, **we can assume without loss of generality that the matrix in any quadratic form is real and symmetric ($A = A^T$)**.

---

#### The Rayleigh Quotient & Constrained Curvature Optimization:
For a real symmetric matrix $A \in \mathbb{R}^{n \times n}$, the **Rayleigh Quotient** is:
$$R(x) \triangleq \frac{x^T A x}{x^T x}, \quad \forall x \neq \mathbf{0}$$

##### First-Principles Derivation of Rayleigh Bounds:
$$\lambda_{\min}(A) \le \frac{x^T A x}{x^T x} \le \lambda_{\max}(A)$$

**Proof Steps:**
1. By the Spectral Theorem, $A = Q \Lambda Q^T$ where $Q = [q_1 \dots q_n]$ is orthogonal and eigenvalues are sorted: $\lambda_1 = \lambda_{\max} \ge \lambda_2 \ge \dots \ge \lambda_n = \lambda_{\min}$.
2. Change coordinates to the eigenvector basis: $y = Q^T x \iff x = Q y$.
   - Denominator: $x^T x = (Q y)^T (Q y) = y^T Q^T Q y = y^T y = \sum_{i=1}^n y_i^2$.
   - Numerator: $x^T A x = (Q y)^T (Q \Lambda Q^T) (Q y) = y^T \Lambda y = \sum_{i=1}^n \lambda_i y_i^2$.
3. Bound the numerator using $\lambda_{\max}$ and $\lambda_{\min}$:
   $$\lambda_{\min} \sum_{i=1}^n y_i^2 \le \sum_{i=1}^n \lambda_i y_i^2 \le \lambda_{\max} \sum_{i=1}^n y_i^2$$
4. Divide through by $\sum_{i=1}^n y_i^2 = \|x\|_2^2 > 0$:
   $$\lambda_{\min} \le \frac{x^T A x}{x^T x} \le \lambda_{\max}$$
5. Setting $x = q_1$ yields $R(q_1) = \frac{\lambda_1 (1)}{1} = \lambda_{\max}$; setting $x = q_n$ yields $R(q_n) = \lambda_{\min}$.
   Therefore:
   $$\max_{\|x\|_2 = 1} x^T A x = \lambda_{\max}(A), \quad \min_{\|x\|_2 = 1} x^T A x = \lambda_{\min}(A) \quad \blacksquare$$

---

### 2. Taxonomy of Matrix Definiteness
Let $A \in \mathbb{R}^{n \times n}$ be a real symmetric matrix ($A = A^T$).
For all non-zero vectors $x \in \mathbb{R}^n \setminus \{\mathbf{0}\}$:

| Definiteness Class | Mathematical Condition | Eigenvalue Condition | Geometric Landscape of $z = x^T A x$ | Optimization Meaning |
| :--- | :---: | :---: | :--- | :--- |
| **Positive Definite (PD)** | $x^T A x > 0$ | All $\lambda_i > 0$ | Upward-opening parabolic bowl | Strict local minimum |
| **Positive Semi-Definite (PSD)** | $x^T A x \ge 0$ | All $\lambda_i \ge 0$ | Upward trough / flat valley | Non-strict minimum (flat valley) |
| **Negative Definite (ND)** | $x^T A x < 0$ | All $\lambda_i < 0$ | Downward-opening dome | Strict local maximum |
| **Negative Semi-Definite (NSD)**| $x^T A x \le 0$ | All $\lambda_i \le 0$ | Downward ridge | Non-strict maximum |
| **Indefinite** | Takes both $+$ and $-$ | Some $\lambda_i > 0$, some $\lambda_j < 0$ | Horse saddle / Pringles chip | **Saddle point (unstable)** |

```
         Positive Definite (Bowl)                    Indefinite (Saddle Point)
                     z                                           z
                     ^                                           ^
                     |     /                                     |     /
                     |   .'                                      |   .'  (curves up)
                  \  |  /                                     \  |  /
                   \ | /                                       \ | /
                    \|/                                         \|/
          -----------+-----------> y                  -----------+-----------> y
                    /|\                                         /|\
                   / | \                                       / | \
                  /  |  \                                     /  |  \ (curves down)
                     |                                           |
```

---

### 3. The Five Equivalent Tests for Positive Definiteness (Gilbert Strang Test Battery)
For a real symmetric matrix $A = A^T \in \mathbb{R}^{n \times n}$, the following five tests are completely equivalent:

1. **The Energy Test**:
   $$x^T A x > 0 \quad \text{for all } x \neq \mathbf{0}$$
2. **The Eigenvalue Test**:
   All $n$ eigenvalues are strictly positive:
   $$\lambda_i > 0, \quad \forall i \in \{1, 2, \dots, n\}$$
3. **The Pivot Test**:
   During Gaussian elimination without row swaps, all $n$ pivots are strictly positive:
   $$d_i > 0, \quad \forall i \in \{1, 2, \dots, n\}$$
4. **Sylvester’s Criterion (Leading Principal Minors Test)**:
   The determinants of all upper-left $k \times k$ submatrices are strictly positive:
   $$\det(A_{1:1}) > 0, \quad \det(A_{1:2, 1:2}) > 0, \quad \dots, \quad \det(A) > 0$$
5. **The Factorization Test (Cholesky)**:
   There exists a unique lower-triangular matrix $L$ with positive diagonal entries such that:
   $$A = L L^T$$

---

### 4. Gram Matrices & The Guarantee of Positive Semi-Definiteness
**Theorem:** For *any* rectangular matrix $X \in \mathbb{R}^{m \times n}$ (regardless of its rank or values), the Gram matrix:
$$A = X^T X \in \mathbb{R}^{n \times n}$$
is **always symmetric and Positive Semi-Definite (PSD)**.

#### Proof:
1. **Symmetry:**
   $$A^T = (X^T X)^T = X^T (X^T)^T = X^T X = A \quad \checkmark$$
2. **Positive Semi-Definiteness:**
   For any arbitrary vector $x \in \mathbb{R}^n$:
   $$x^T A x = x^T (X^T X) x = (X x)^T (X x) = \|X x\|_2^2$$
   Since the Euclidean squared norm $\|v\|_2^2 \ge 0$ for all vectors:
   $$x^T A x \ge 0, \quad \forall x \in \mathbb{R}^n \quad \blacksquare$$

> [!IMPORTANT]
> **Why Covariance Matrices are Always Valid:**
> In data science, sample covariance is $\Sigma = \frac{1}{N} \tilde{X}^T \tilde{X}$.
> By this theorem, $\Sigma$ is guaranteed to be PSD. Variances along any projection direction $u^T \Sigma u = \text{Var}(u^T x)$ can **never be negative**!

---

### 5. The Cholesky Factorization ($A = L L^T$)
If $A \in \mathbb{R}^{n \times n}$ is symmetric and Positive Definite, it can be factored uniquely as:
$$A = L L^T$$
where $L$ is a **lower-triangular matrix** with strictly positive diagonal entries ($L_{ii} > 0$):
$$\begin{bmatrix} A_{11} & A_{12} & \dots & A_{1n} \\ A_{21} & A_{22} & \dots & A_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ A_{n1} & A_{n2} & \dots & A_{nn} \end{bmatrix} = \begin{bmatrix} L_{11} & 0 & \dots & 0 \\ L_{21} & L_{22} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ L_{n1} & L_{n2} & \dots & L_{nn} \end{bmatrix} \begin{bmatrix} L_{11} & L_{21} & \dots & L_{n1} \\ 0 & L_{22} & \dots & L_{n2} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & L_{nn} \end{bmatrix}$$

#### Computational Advantages of Cholesky:
- Computes in $\frac{1}{3} n^3$ FLOPs (twice as fast as standard Gaussian elimination / LU decomposition!).
- Completely stable without requiring numerical pivoting.
- Determinant is computed effortlessly: $\det(A) = \det(L L^T) = (\det(L))^2 = \left(\prod_{i=1}^n L_{ii}\right)^2$.

---

## Part 3: Geometric & Algebraic Interpretation

### Energy Ellipsoids
Consider the level set of a positive definite quadratic form:
$$x^T A x = 1$$
By the Spectral Theorem, $A = Q \Lambda Q^T = \sum \lambda_i q_i q_i^T$.
Let $y = Q^T x$ be the coordinates of $x$ along the orthonormal eigenvector axes:
$$x^T A x = y^T \Lambda y = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \dots + \lambda_n y_n^2 = 1$$
Rewriting into standard canonical ellipse form:
$$\frac{y_1^2}{(1/\sqrt{\lambda_1})^2} + \frac{y_2^2}{(1/\sqrt{\lambda_2})^2} + \dots + \frac{y_n^2}{(1/\sqrt{\lambda_n})^2} = 1$$

```
                     y2
                     ^                  q2 (Direction of sharp curvature, λ2 is LARGE)
                     |                 /   Semi-axis length = 1 / √λ2 (SHORT!)
                     |             .----+----.
                     |           .'     |     '.
                     |          /       |       \
                     |         |        +--------|---------> q1 (Direction of flat curvature, λ1 is SMALL)
                     +---------|-----------------|------> y1    Semi-axis length = 1 / √λ1 (LONG!)
                                \               /
                                 '.           .'
                                   '---------'
```

- **Large Eigenvalue $\lambda$**: The loss curves steeply; the ellipsoid is compressed; the semi-axis $\frac{1}{\sqrt{\lambda}}$ is **short**.
- **Small Eigenvalue $\lambda$**: The loss is nearly flat; the ellipsoid stretches far; the semi-axis $\frac{1}{\sqrt{\lambda}}$ is **long**.

---

## Part 4: Real-World Analogy

### The Rolling Marble on Terrain
- **Positive Definite Terrain (Cereal Bowl)**: A marble released from anywhere rolls directly and stably to the unique bottom at the center. Gradient descent converges easily.
- **Indefinite Terrain (Mountain Pass / Saddle Point)**: If the marble is placed at the pass, it is flat. But if nudged forward or backward, it climbs up; if nudged left or right, it rolls down the cliff. In deep networks, standard Newton's method can get tricked by the negative curvature and jump toward the cliff instead of the valley!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### Cholesky Factorization & Multivariate Gaussian Sampling by Hand

Let us perform the Cholesky factorization of a $2 \times 2$ covariance matrix $\Sigma$, verify its positive definiteness, and use it to sample a correlated 2D Gaussian point by hand!

$$\Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 10 \end{bmatrix}, \quad \text{Mean } \mu = \begin{bmatrix} 5 \\ 3 \end{bmatrix}$$

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Shape | Mathematical Role | Generative AI / VAE Meaning | Concrete Demo Value |
| :---: | :---: | :--- | :--- | :---: |
| $\Sigma$ | $[2 \times 2]$ | Symmetric PSD matrix | Covariance matrix of latent features | $\begin{bmatrix} 4 & 2 \\ 2 & 10 \end{bmatrix}$ |
| $L$ | $[2 \times 2]$ | Lower-triangular Cholesky factor | Latent reparameterization transformation | To be solved |
| $\mu$ | $[2 \times 1]$ | Vector | Prior mean vector | $\begin{bmatrix} 5 \\ 3 \end{bmatrix}$ |
| $\epsilon$ | $[2 \times 1]$ | Standard normal noise $\mathcal{N}(0, I)$ | Uncorrelated random seed | $\begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix}$ |
| $x$ | $[2 \times 1]$ | Sampled vector $\mu + L \epsilon$ | Correlated generated sample | To be computed |

---

#### 2. Step 1: Sylvester's Definiteness Test by Hand
- **Leading Principal Minor 1 ($1 \times 1$):**
  $$D_1 = \det([4]) = 4 > 0 \quad \checkmark$$
- **Leading Principal Minor 2 ($2 \times 2$):**
  $$D_2 = \det(\Sigma) = (4 \times 10) - (2 \times 2) = 40 - 4 = 36 > 0 \quad \checkmark$$
Since all leading principal minors are strictly positive, **$\Sigma$ is guaranteed to be strictly Positive Definite!**

---

#### 3. Step 2: Solve for Cholesky Factor $L = \begin{bmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{bmatrix}$

We expand $L L^T$:
$$\begin{bmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{bmatrix} \begin{bmatrix} L_{11} & L_{21} \\ 0 & L_{22} \end{bmatrix} = \begin{bmatrix} L_{11}^2 & L_{11} L_{21} \\ L_{21} L_{11} & L_{21}^2 + L_{22}^2 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 10 \end{bmatrix}$$

| Target Cell in $\Sigma$ | Symbolic Formula | Algebraic Equation | Step-by-Step Solution |
| :---: | :--- | :--- | :---: |
| **Row 1, Col 1** | $L_{11}^2 = 4$ | $L_{11} = \sqrt{4}$ | **$L_{11} = 2$** |
| **Row 2, Col 1** | $L_{21} L_{11} = 2$ | $L_{21} \times 2 = 2 \implies L_{21} = 2/2$ | **$L_{21} = 1$** |
| **Row 2, Col 2** | $L_{21}^2 + L_{22}^2 = 10$ | $1^2 + L_{22}^2 = 10 \implies L_{22}^2 = 10 - 1 = 9$ | **$L_{22} = \sqrt{9} = 3$** |

$$L = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}$$

---

#### 4. Step 3: Verification of $L L^T = \Sigma$ by Hand

$$L L^T = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} = \begin{bmatrix} (2)(2) + (0)(0) & (2)(1) + (0)(3) \\ (1)(2) + (3)(0) & (1)(1) + (3)(3) \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 1 + 9 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 10 \end{bmatrix} \equiv \Sigma \quad \checkmark$$

---

#### 5. Step 4: Generating a Correlated Sample $x = \mu + L \epsilon$ by Hand

Let standard uncorrelated Gaussian noise be $\epsilon = \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix}$:
1. **Transform Noise ($L \epsilon$):**
   $$\begin{bmatrix} \Delta x_1 \\ \Delta x_2 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 2(1.0) + 0(-0.5) \\ 1(1.0) + 3(-0.5) \end{bmatrix} = \begin{bmatrix} 2.0 \\ 1.0 - 1.5 \end{bmatrix} = \begin{bmatrix} 2.0 \\ -0.5 \end{bmatrix}$$
2. **Add Mean Shift ($x = \mu + L \epsilon$):**
   $$x = \begin{bmatrix} 5 \\ 3 \end{bmatrix} + \begin{bmatrix} 2.0 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 5 + 2.0 \\ 3 - 0.5 \end{bmatrix} = \begin{bmatrix} 7.0 \\ 2.5 \end{bmatrix}$$

> [!IMPORTANT]
> **The VAE Reparameterization Trick:**
> We have just drawn a valid sample from $\mathcal{N}(\mu, \Sigma)$ without ever sampling from a complex distribution!
> Because $x = \mu + L \epsilon$ is a purely deterministic linear function of $\mu$ and $L$, **gradients can flow backward through the sampling step directly to the encoder weights!**

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Testing Definiteness Across Four Matrices
Classify the following four symmetric matrices:
1. $A_1 = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$
2. $A_2 = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$
3. $A_3 = \begin{bmatrix} -3 & 1 \\ 1 & -2 \end{bmatrix}$
4. $A_4 = \begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix}$

#### Step-by-Step Evaluation:
- **Matrix $A_1$:**
  - $D_1 = 2 > 0$, $D_2 = (2)(2) - (-1)^2 = 4 - 1 = 3 > 0$.
  - Both leading minors are positive $\implies$ **Positive Definite (PD)**.
- **Matrix $A_2$:**
  - $D_1 = 1 > 0$, $D_2 = (1)(4) - (2)^2 = 4 - 4 = 0$.
  - Eigenvalues: $\text{Tr} = 5, \det = 0 \implies \lambda_1 = 5, \lambda_2 = 0$.
  - Eigenvalues are non-negative ($\ge 0$), with at least one zero $\implies$ **Positive Semi-Definite (PSD)**.
- **Matrix $A_3$:**
  - $D_1 = -3 < 0$, $D_2 = (-3)(-2) - 1 = 6 - 1 = 5 > 0$.
  - Eigenvalues: $\lambda^2 + 5\lambda + 5 = 0 \implies \lambda = \frac{-5 \pm \sqrt{5}}{2}$ (both are negative: $-1.38$ and $-3.62$).
  - All eigenvalues $< 0 \implies$ **Negative Definite (ND)**.
- **Matrix $A_4$:**
  - $D_1 = 1 > 0$, $D_2 = (1)(1) - (3)^2 = 1 - 9 = -8 < 0$.
  - Eigenvalues: $\lambda_1 = 4, \lambda_2 = -2$.
  - Eigenvalues have opposite signs $\implies$ **Indefinite (Saddle Point)**.

---

### Scenario B: The Mahalanobis Distance Metric
**Problem:** Given covariance matrix $\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}$, compute the distance between the origin $\mathbf{0}$ and two points:
$$p_1 = \begin{bmatrix} 2 \\ 0 \end{bmatrix}, \quad p_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix}$$
1. Standard Euclidean Distance:
   $$\|p_1\|_2 = \sqrt{2^2 + 0} = 2, \quad \|p_2\|_2 = \sqrt{0 + 2^2} = 2 \quad \text{(Equally distant)}$$
2. Mahalanobis Distance: $d_M(p) = \sqrt{p^T \Sigma^{-1} p}$:
   $$\Sigma^{-1} = \begin{bmatrix} 1/4 & 0 \\ 0 & 1 \end{bmatrix}$$
   $$d_M(p_1) = \sqrt{\begin{bmatrix} 2 & 0 \end{bmatrix} \begin{bmatrix} 1/4 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 0 \end{bmatrix}} = \sqrt{4/4 + 0} = \sqrt{1} = \mathbf{1.0}$$
   $$d_M(p_2) = \sqrt{\begin{bmatrix} 0 & 2 \end{bmatrix} \begin{bmatrix} 1/4 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 2 \end{bmatrix}} = \sqrt{0 + 4(1)} = \sqrt{4} = \mathbf{2.0}$$
*Insight:* Although $p_1$ and $p_2$ have identical Euclidean distance from the origin, $p_1$ lies along the high-variance axis ($\sigma_1^2 = 4$) and is only **1 standard deviation** away, while $p_2$ lies along the low-variance axis ($\sigma_2^2 = 1$) and is **2 standard deviations** away! Mahalanobis distance measures statistical distance in units of standard deviations.

---

### Scenario C: Complete $3 \times 3$ Cholesky Factorization by Hand

**Problem Formulation:**
Perform Cholesky factorization $A = L L^T$ for the symmetric matrix:
$$A = \begin{bmatrix} 4 & 2 & -2 \\ 2 & 10 & 2 \\ -2 & 2 & 6 \end{bmatrix}$$
1. Verify Sylvester's Criterion (Leading Principal Minors).
2. Compute each entry of lower-triangular matrix $L$ step-by-step.
3. Compute $\det(A)$ directly from the diagonal of $L$.

#### Step 1: Sylvester's Criterion Test
- $D_1 = \det([4]) = 4 > 0$
- $D_2 = \det \begin{bmatrix} 4 & 2 \\ 2 & 10 \end{bmatrix} = (4)(10) - (2)^2 = 40 - 4 = 36 > 0$
- $D_3 = \det(A) = 4(10 \times 6 - 2 \times 2) - 2(2 \times 6 - 2 \times (-2)) + (-2)(2 \times 2 - 10 \times (-2))$
  $$D_3 = 4(60 - 4) - 2(12 + 4) - 2(4 + 20) = 4(56) - 2(16) - 2(24) = 224 - 32 - 48 = 144 > 0$$
All leading principal minors are strictly positive; $A$ is guaranteed **Positive Definite**.

#### Step 2: Compute Entries of $L$
Set up $L L^T = A$:
$$\begin{bmatrix} L_{11} & 0 & 0 \\ L_{21} & L_{22} & 0 \\ L_{31} & L_{32} & L_{33} \end{bmatrix} \begin{bmatrix} L_{11} & L_{21} & L_{31} \\ 0 & L_{22} & L_{32} \\ 0 & 0 & L_{33} \end{bmatrix} = \begin{bmatrix} 4 & 2 & -2 \\ 2 & 10 & 2 \\ -2 & 2 & 6 \end{bmatrix}$$

1. **Column 1:**
   - $L_{11}^2 = 4 \implies L_{11} = \sqrt{4} = \mathbf{2}$
   - $L_{21} L_{11} = 2 \implies L_{21}(2) = 2 \implies L_{21} = \mathbf{1}$
   - $L_{31} L_{11} = -2 \implies L_{31}(2) = -2 \implies L_{31} = \mathbf{-1}$
2. **Column 2:**
   - $L_{21}^2 + L_{22}^2 = 10 \implies 1^2 + L_{22}^2 = 10 \implies L_{22}^2 = 9 \implies L_{22} = \mathbf{3}$
   - $L_{31} L_{21} + L_{32} L_{22} = 2 \implies (-1)(1) + L_{32}(3) = 2 \implies 3 L_{32} = 3 \implies L_{32} = \mathbf{1}$
3. **Column 3:**
   - $L_{31}^2 + L_{32}^2 + L_{33}^2 = 6 \implies (-1)^2 + 1^2 + L_{33}^2 = 6 \implies 1 + 1 + L_{33}^2 = 6 \implies L_{33}^2 = 4 \implies L_{33} = \mathbf{2}$

$$L = \begin{bmatrix} 2 & 0 & 0 \\ 1 & 3 & 0 \\ -1 & 1 & 2 \end{bmatrix}$$

#### Step 3: Fast Determinant Evaluation via Cholesky
$$\det(A) = \left( \prod_{i=1}^3 L_{ii} \right)^2 = (2 \times 3 \times 2)^2 = (12)^2 = \mathbf{144}$$
Matches the cofactor determinant from Step 1 bit-for-bit with zero polynomial expansion overhead. $\checkmark$

---

### Scenario D: Loss Landscape Saddle Point Analysis via Indefinite Hessian

**Problem Formulation:**
Consider a toy neural network loss objective with two parameters:
$$\mathcal{L}(w_1, w_2) = w_1^2 - 3w_2^2 + 4w_1 w_2$$
1. Find the stationary point by solving $\nabla \mathcal{L} = \mathbf{0}$.
2. Compute the Hessian matrix $H = \nabla^2 \mathcal{L}$.
3. Solve for the eigenvalues of $H$ to classify the stationary point.
4. Show why standard Newton's step diverges and how Levenberg-Marquardt damping stabilizes it.

#### Step 1: Compute Gradient & Stationary Point
$$\nabla \mathcal{L}(w) = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w_1} \\ \frac{\partial \mathcal{L}}{\partial w_2} \end{bmatrix} = \begin{bmatrix} 2w_1 + 4w_2 \\ -6w_2 + 4w_1 \end{bmatrix}$$
Setting $\nabla \mathcal{L} = \mathbf{0}$:
$$\begin{cases} 2w_1 + 4w_2 = 0 \implies w_1 = -2w_2 \\ 4(-2w_2) - 6w_2 = 0 \implies -14w_2 = 0 \implies w_2 = 0, w_1 = 0 \end{cases}$$
The unique critical point is the origin $w^* = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$.

#### Step 2: Compute Hessian Matrix $H$
$$H = \begin{bmatrix} \frac{\partial^2 \mathcal{L}}{\partial w_1^2} & \frac{\partial^2 \mathcal{L}}{\partial w_1 \partial w_2} \\ \frac{\partial^2 \mathcal{L}}{\partial w_2 \partial w_1} & \frac{\partial^2 \mathcal{L}}{\partial w_2^2} \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 4 & -6 \end{bmatrix}$$

#### Step 3: Eigenvalues & Definiteness
$$\det(H - \lambda I) = (2 - \lambda)(-6 - \lambda) - 16 = \lambda^2 + 4\lambda - 12 - 16 = \lambda^2 + 4\lambda - 28 = 0$$
$$\lambda = \frac{-4 \pm \sqrt{16 - 4(1)(-28)}}{2} = \frac{-4 \pm \sqrt{128}}{2} = -2 \pm 4\sqrt{2}$$
- $\lambda_1 = -2 + 4\sqrt{2} \approx -2 + 5.657 = \mathbf{+3.657} > 0$ (Upward curvature)
- $\lambda_2 = -2 - 4\sqrt{2} \approx -2 - 5.657 = \mathbf{-7.657} < 0$ (Downward curvature)
- **Conclusion:** $H$ is strictly **Indefinite**. The origin $w^*$ is a **Saddle Point**, not a local minimum!

#### Step 4: Damped Newton Stabilization
To turn the indefinite Hessian into a guaranteed descent direction:
$$H_{\text{damped}} = H + \lambda_{\text{damp}} I$$
By choosing $\lambda_{\text{damp}} = 8.0 > |\lambda_2| = 7.657$:
- $\lambda_1(H_{\text{damped}}) = 3.657 + 8.0 = 11.657 > 0$
- $\lambda_2(H_{\text{damped}}) = -7.657 + 8.0 = 0.343 > 0$
Now $H_{\text{damped}} \succ 0$ is strictly Positive Definite, eliminating negative curvature and allowing the optimizer to escape the saddle point safely.

---

## Part 7: Deep Learning Connection & Application

### 1. Newton's Method & Saddle-Point Avoidance
The classic second-order Newton step is:
$$\Delta w = -H^{-1} \nabla \mathcal{L}$$
- We test if $\Delta w$ points in a descent direction by taking its dot product with the negative gradient:
  $$(-\nabla \mathcal{L})^T \Delta w = \nabla \mathcal{L}^T H^{-1} \nabla \mathcal{L}$$
- If $H$ is **Positive Definite**, then $H^{-1}$ is also Positive Definite. Therefore:
  $$\nabla \mathcal{L}^T H^{-1} \nabla \mathcal{L} > 0 \implies \text{Guaranteed descent step!}$$
- But if $H$ is **Indefinite** (common in deep non-convex landscapes), $\Delta w$ can climb uphill toward the saddle point!
- **Levenberg-Marquardt & Damped Newton Solution:**
  Replace $H$ with:
  $$H_{\text{damped}} = H + \lambda I$$
  By choosing $\lambda > |\lambda_{\min}(H)|$, all eigenvalues are shifted to strictly positive values, guaranteeing a descent step!

### 2. Gaussian Process Kernels & Mercer's Theorem
In Kernel Ridge Regression, Support Vector Machines (SVMs), and Gaussian Processes, any valid kernel function $k(x, y)$ must produce a **Positive Semi-Definite Gram matrix**:
$$K_{ij} = k(x_i, x_j) \implies c^T K c \ge 0, \quad \forall c \in \mathbb{R}^N$$
If $K$ were not PSD, the model would predict **negative probabilities and negative variances**!

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/08_quadratic_forms_and_psd.py`](./code/08_quadratic_forms_and_psd.py) implements:
1. **Cholesky Factorization Engine from Scratch**: Computes lower-triangular $L$ and detects non-positive-definite matrices.
2. **Definiteness Classifier**: Evaluates eigenvalues, Sylvester's principal minors, and quadratic energy to categorize any symmetric matrix into PD, PSD, ND, NSD, or Indefinite.
3. **Multivariate Gaussian Sampler with Cholesky**: Implements the VAE reparameterization trick $x = \mu + L \epsilon$ and computes empirical covariance across 10,000 samples.
4. **Newton's Optimization Descent Direction Test**: Simulates parameter updates comparing a PD Hessian vs an Indefinite Hessian near a saddle point.

---

## Chapter 1.8 Summary & Review Checklist

- [x] What is the definition of a quadratic form $q(x) = x^T A x$?
- [x] Why can we always assume matrix $A$ is symmetric in a quadratic form?
- [x] What are the five equivalent tests for Positive Definiteness?
- [x] Why is every Gram matrix $X^T X$ guaranteed to be Positive Semi-Definite?
- [x] How does Cholesky decomposition enable the VAE reparameterization trick?
- [x] Why does an indefinite Hessian cause standard Newton's method to fail near saddle points?
