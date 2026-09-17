# Chapter 1.6: Eigendecomposition & Spectral Theory

---

## Part 1: Intuition & 101 Motivation

In Deep Learning, we repeatedly apply matrix transformations:
- In **Recurrent Neural Networks (RNNs)**, the hidden state dynamics across $T$ sequence steps involve repeated multiplication by the transition matrix:
  $$h_T \approx W_{hh}^T h_0$$
- In **Gradient Descent**, the parameter update step involves the curvature of the loss function described by the **Hessian matrix** $H = \nabla^2 \mathcal{L}$.
- In **Markov Chains and Attention Diffusion**, probability distributions evolve by repeated application of transition operators.

If you multiply a vector $x$ by a matrix $A$, its length and direction usually both change. But are there special, invariant directions in the space that **never rotate**?

Yes. These invariant axes are the **Eigenvectors**, and the factor by which they stretch or compress is their **Eigenvalue**:
$$A v = \lambda v$$

In the coordinate system formed by its eigenvectors, matrix multiplication ceases to be a complex, entangling operation. It decouples completely into **independent 1D scalar multiplications**:
- If $|\lambda| > 1$, signals along that eigenvector **explode exponentially**.
- If $|\lambda| < 1$, signals along that eigenvector **vanish exponentially**.
- If $|\lambda| = 1$, signals remain perfectly stable.

This single insight unlocks the dynamics of deep learning: the **exploding/vanishing gradient problem**, the **optimal learning rate** for Adam and SGD, and **Principal Component Analysis (PCA)**.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Definition of Eigenvalues & Eigenvectors
Let $A \in \mathbb{R}^{n \times n}$ be a square matrix.
A non-zero vector $v \in \mathbb{C}^n \setminus \{\mathbf{0}\}$ is called an **eigenvector** of $A$, and a scalar $\lambda \in \mathbb{C}$ is called its associated **eigenvalue**, if:
$$A v = \lambda v$$
Rearranging into a homogeneous linear system:
$$(A - \lambda I_n) v = \mathbf{0}$$

Since $v \neq \mathbf{0}$, the matrix $(A - \lambda I_n)$ must have a non-trivial nullspace, which means it must be **singular** ($\det = 0$):

$$\det(A - \lambda I_n) = 0 \quad \text{\bf [The Characteristic Equation]}$$

- The determinant expands into an $n$-th degree polynomial in $\lambda$:
  $$p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + c_{n-1} \lambda^{n-1} + \dots + c_1 \lambda + \det(A) = 0$$
- By the Fundamental Theorem of Algebra, $p(\lambda)$ has exactly $n$ roots (counting algebraic multiplicities), which may be real or complex.

---

### 2. Multiplicities & Defective Matrices

For any eigenvalue $\lambda_i$:
1. **Algebraic Multiplicity (AM)**: The multiplicity of $\lambda_i$ as a root of the characteristic polynomial $p(\lambda)$.
2. **Geometric Multiplicity (GM)**: The dimension of the eigenspace corresponding to $\lambda_i$:
   $$\text{GM}(\lambda_i) = \dim(N(A - \lambda_i I_n))$$
3. **Fundamental Theorem of Multiplicity**:
   $$1 \le \text{GM}(\lambda_i) \le \text{AM}(\lambda_i) \le n$$

> [!WARNING]
> **Defective Matrices:**
> If $\text{GM}(\lambda_i) < \text{AM}(\lambda_i)$ for any eigenvalue, the matrix lacks a complete set of $n$ linearly independent eigenvectors. Such a matrix is called **defective** and **cannot be diagonalized**!

---

### 3. Matrix Diagonalization ($A = S \Lambda S^{-1}$)
**Theorem (Diagonalizability):** An $n \times n$ matrix $A$ is diagonalizable if and only if it has $n$ linearly independent eigenvectors $\{v_1, v_2, \dots, v_n\}$.

Construct the **Eigenvector Matrix** $S$ and the **Eigenvalue Diagonal Matrix** $\Lambda$:
$$S = \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix} \in \mathbb{C}^{n \times n}, \quad \Lambda = \begin{bmatrix} \lambda_1 & 0 & \dots & 0 \\ 0 & \lambda_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \lambda_n \end{bmatrix}$$

Multiplying $A$ by $S$:
$$A S = A \begin{bmatrix} v_1 & \dots & v_n \end{bmatrix} = \begin{bmatrix} A v_1 & \dots & A v_n \end{bmatrix} = \begin{bmatrix} \lambda_1 v_1 & \dots & \lambda_n v_n \end{bmatrix} = \begin{bmatrix} v_1 & \dots & v_n \end{bmatrix} \Lambda = S \Lambda$$

Since the columns of $S$ are linearly independent, $S$ is invertible:
$$A = S \Lambda S^{-1}$$

#### Matrix Powers via Diagonalization:
$$A^2 = (S \Lambda S^{-1})(S \Lambda S^{-1}) = S \Lambda (S^{-1} S) \Lambda S^{-1} = S \Lambda^2 S^{-1}$$
For any positive integer $k$:
$$A^k = S \Lambda^k S^{-1} = S \begin{bmatrix} \lambda_1^k & 0 & \dots & 0 \\ 0 & \lambda_2^k & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \lambda_n^k \end{bmatrix} S^{-1}$$

---

### 4. The Spectral Theorem for Symmetric Matrices
In Deep Learning, the most important matrices are **real and symmetric** ($A = A^T$), such as:
- Covariance matrices $\Sigma = \frac{1}{N} X^T X$
- Hessian matrices $H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j}$ (Schwarz's theorem guarantees symmetry)
- Attention affinity matrices in Transformers

#### The Spectral Theorem:
Every real symmetric matrix $A \in \mathbb{R}^{n \times n}$ satisfies:
1. **All $n$ eigenvalues are strictly real numbers**: $\lambda_i \in \mathbb{R}$.
2. **Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal**:
   $$\lambda_i \neq \lambda_j \implies v_i \perp v_j \quad (v_i^T v_j = 0)$$
3. **There always exists an orthonormal basis of eigenvectors**, even for repeated eigenvalues ($\text{GM} = \text{AM}$ always holds; symmetric matrices are **never defective**).
4. **Orthogonal Diagonalization**:
   $$A = Q \Lambda Q^T$$
   where $Q$ is an **orthogonal matrix** ($Q^T Q = Q Q^T = I$).

#### Rigorous First-Principles Derivation / Proof of the Spectral Properties:

##### 1. Proof that All Eigenvalues of a Real Symmetric Matrix are Real:
Let $A \in \mathbb{R}^{n \times n}$ with $A = A^T$.
Suppose $\lambda \in \mathbb{C}$ is an eigenvalue with non-zero eigenvector $v \in \mathbb{C}^n \setminus \{\mathbf{0}\}$:
$$A v = \lambda v$$
Take the complex conjugate transpose (Hermitian conjugate) of both sides:
$$(A v)^H = (\lambda v)^H \implies v^H A^H = \bar{\lambda} v^H$$
Since $A$ is real and symmetric, $A^H = (A^*)^T = A^T = A$. Therefore:
$$v^H A = \bar{\lambda} v^H$$
Multiply on the right by $v$:
$$(v^H A) v = \bar{\lambda} v^H v$$
On the other hand, multiply $A v = \lambda v$ on the left by $v^H$:
$$v^H (A v) = \lambda v^H v$$
Since matrix multiplication is associative, $v^H (A v) = (v^H A) v$:
$$\lambda v^H v = \bar{\lambda} v^H v \implies (\lambda - \bar{\lambda}) v^H v = 0$$
Because $v \neq \mathbf{0}$, the inner product $v^H v = \sum_{i=1}^n |v_i|^2 > 0$ is strictly positive.
Therefore:
$$\lambda - \bar{\lambda} = 0 \implies \lambda = \bar{\lambda}$$
A complex number equal to its own complex conjugate is strictly real. Hence $\lambda \in \mathbb{R}$. $\blacksquare$

##### 2. Proof that Eigenvectors of Distinct Eigenvalues are Strictly Orthogonal:
Suppose $A v_1 = \lambda_1 v_1$ and $A v_2 = \lambda_2 v_2$, where $\lambda_1 \neq \lambda_2 \in \mathbb{R}$.
Consider the scalar quantity $v_1^T A v_2$:
- Evaluating $A$ applied to $v_2$:
  $$v_1^T (A v_2) = v_1^T (\lambda_2 v_2) = \lambda_2 (v_1^T v_2)$$
- Evaluating $A$ applied to $v_1$ using symmetry $A = A^T$:
  $$(v_1^T A) v_2 = (A^T v_1)^T v_2 = (A v_1)^T v_2 = (\lambda_1 v_1)^T v_2 = \lambda_1 (v_1^T v_2)$$
Equating both expressions:
$$\lambda_1 (v_1^T v_2) = \lambda_2 (v_1^T v_2) \implies (\lambda_1 - \lambda_2) (v_1^T v_2) = 0$$
Since $\lambda_1 \neq \lambda_2$, the difference $(\lambda_1 - \lambda_2) \neq 0$.
Therefore:
$$v_1^T v_2 = 0 \implies v_1 \perp v_2 \quad \blacksquare$$

---

#### The Cayley-Hamilton Theorem
**Theorem:** Every square matrix $A \in \mathbb{R}^{n \times n}$ satisfies its own characteristic equation:
$$p(A) = (-1)^n A^n + c_{n-1} A^{n-1} + \dots + c_1 A + \det(A) I_n = \mathbf{0}$$

##### First-Principles Derivation for Diagonalizable Matrices:
If $A = S \Lambda S^{-1}$, then for any polynomial $p(x)$:
$$p(A) = S p(\Lambda) S^{-1} = S \begin{bmatrix} p(\lambda_1) & & 0 \\ & \ddots & \\ 0 & & p(\lambda_n) \end{bmatrix} S^{-1}$$
Since each $\lambda_i$ is a root of the characteristic polynomial, $p(\lambda_i) = 0$ for all $i \in \{1, \dots, n\}$.
Therefore $p(\Lambda) = \mathbf{0}$, which implies:
$$p(A) = S \mathbf{0} S^{-1} = \mathbf{0} \quad \blacksquare$$
*Deep Learning Application:* Enables computing high matrix powers $A^k$ as linear combinations of lower powers $\{I, A, \dots, A^{n-1}\}$, accelerating graph neural networks and diffusion transitions.

#### The Spectral Decomposition (Outer Product Expansion):
Expanding $Q \Lambda Q^T$ using Gilbert Strang's outer product perspective:
$$A = \begin{bmatrix} q_1 & \dots & q_n \end{bmatrix} \begin{bmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{bmatrix} \begin{bmatrix} q_1^T \\ \vdots \\ q_n^T \end{bmatrix} = \sum_{i=1}^n \lambda_i q_i q_i^T$$

> [!IMPORTANT]
> **Every real symmetric matrix is a weighted sum of rank-1 orthogonal projection sheets $q_i q_i^T$, scaled by its eigenvalues $\lambda_i$!**

---

### 5. Spectral Invariants: Trace and Determinant
For any square matrix $A \in \mathbb{R}^{n \times n}$ with eigenvalues $\lambda_1, \dots, \lambda_n$:
1. **The Trace is the Sum of Eigenvalues**:
   $$\text{Tr}(A) = \sum_{i=1}^n A_{ii} = \sum_{i=1}^n \lambda_i$$
2. **The Determinant is the Product of Eigenvalues**:
   $$\det(A) = \prod_{i=1}^n \lambda_i$$
3. **Spectral Radius $\rho(A)$**:
   $$\rho(A) = \max_{1 \le i \le n} |\lambda_i|$$
   - **Stability Theorem:** $\lim_{k \to \infty} A^k = 0 \iff \rho(A) < 1$.

---

## Part 3: Geometric & Algebraic Interpretation

### The Eigenvector Coordinate Frame

```
                     y
                     ^                  q2 (Eigenvector 2, λ2 = 0.5)
                     |                 /
                     |                /   Ellipse of Ax (Unit circle transformed)
                     |               /   .-'      '-.
                     |              /  .'            '.
                     |             /  /                \
                     |            /  |                  |
                     +-----------/---|------------------|-------> x
                                /     \                /
                               /       '.            .'
                              /          '-.      .-'
                             /              \
                                             q1 (Eigenvector 1, λ1 = 2.5)
```

When a symmetric matrix $A$ transforms the unit circle $\|x\|_2 = 1$:
- The circle deforms into an **ellipse**.
- The **principal axes of the ellipse** point directly along the orthogonal eigenvectors $q_1$ and $q_2$.
- The **half-lengths of the axes** are exactly equal to the eigenvalues $|\lambda_1|$ and $|\lambda_2|$.
- In this rotated frame ($Q$), there is zero cross-coupling: coordinate 1 stretches purely by $\lambda_1$, and coordinate 2 stretches purely by $\lambda_2$.

---

## Part 4: Real-World Analogy

### Resonance & Musical Harmonics
Imagine striking a complex acoustic bell or plucking a guitar string:
- At the instant of impact ($t = 0$), the physical vibrations are a chaotic, messy mixture of thousands of spatial displacements ($x_0$).
- But as time evolves ($A^t$), almost all chaotic modes cancel each other out or decay away.
- Only the **fundamental resonant standing waves** survive—these are the **eigenvectors**!
- The pitch / frequency of each musical note corresponds to the **eigenvalue**.
- In an RNN or deep transformer, training aligns the network's recurrent matrices so that useful signals travel along the stable resonant eigenvectors ($\lambda \approx 1$), while noise decays along vanishing directions ($|\lambda| < 1$).

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### RNN Hidden State Dynamics & Spectral Radius by Hand

Let us trace a Recurrent Neural Network (RNN) linear transition layer over multiple time steps completely by hand:
$$h_t = W h_{t-1} \implies h_t = W^t h_0$$
where:
- Transition matrix $W = \begin{bmatrix} 1.2 & 0.4 \\ 0.4 & 0.6 \end{bmatrix}$ (symmetric matrix)
- Initial state $h_0 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Shape | Mathematical Role | RNN Semantic Role | Concrete Demo Value |
| :---: | :---: | :--- | :--- | :---: |
| $W$ | $[2 \times 2]$ | Real symmetric matrix | Recurrent transition weights $W_{hh}$ | $\begin{bmatrix} 1.2 & 0.4 \\ 0.4 & 0.6 \end{bmatrix}$ |
| $h_0$ | $[2 \times 1]$ | Initial vector | Initial hidden state at timestep $0$ | $\begin{bmatrix} 2 \\ 1 \end{bmatrix}$ |
| $\lambda_1, \lambda_2$ | Scalars | Eigenvalues of $W$ | Growth/decay rates per sequence step | To be solved |
| $q_1, q_2$ | $[2 \times 1]$ | Orthonormal eigenvectors | Independent temporal memory channels | To be solved |
| $\rho(W)$ | Scalar | Spectral radius $\max \vert \lambda_i \vert$ | Global recurrent stability indicator | To be computed |

---

#### 2. Visual Matrix Grid Setup

```
         RECURRENT WEIGHT MATRIX W (2 x 2)        INITIAL STATE h_0 (2 x 1)
                  Col 1      Col 2
               ┌──────────┬──────────┐                   ┌──────────┐
         Row 1 │  W_11=1.2│  W_12=0.4│             Row 1 │  h_01=2  │
               ├──────────┼──────────┤                   ├──────────┤
         Row 2 │  W_21=0.4│  W_22=0.6│             Row 2 │  h_02=1  │
               └──────────┴──────────┘                   └──────────┘
```

---

#### 3. Step 1: Characteristic Equation & Eigenvalues by Hand
$$\det(W - \lambda I) = \det \begin{bmatrix} 1.2 - \lambda & 0.4 \\ 0.4 & 0.6 - \lambda \end{bmatrix} = 0$$

$$\begin{aligned} p(\lambda) &= (1.2 - \lambda)(0.6 - \lambda) - (0.4 \times 0.4) \\ &= \lambda^2 - 1.8\lambda + 0.72 - 0.16 \\ &= \lambda^2 - 1.8\lambda + 0.56 = 0 \end{aligned}$$

Using the quadratic formula $\lambda = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$:
$$\lambda = \frac{1.8 \pm \sqrt{(-1.8)^2 - 4(1)(0.56)}}{2} = \frac{1.8 \pm \sqrt{3.24 - 2.24}}{2} = \frac{1.8 \pm \sqrt{1.0}}{2} = \frac{1.8 \pm 1.0}{2}$$

$$\lambda_1 = \frac{1.8 + 1.0}{2} = \mathbf{1.4}, \quad \lambda_2 = \frac{1.8 - 1.0}{2} = \mathbf{0.4}$$

*Invariants Check:*
- $\text{Tr}(W) = 1.2 + 0.6 = 1.8 = \lambda_1 + \lambda_2 \quad \checkmark$
- $\det(W) = (1.2 \times 0.6) - (0.4)^2 = 0.72 - 0.16 = 0.56 = \lambda_1 \times \lambda_2 \quad \checkmark$

---

#### 4. Step 2: Finding Eigenvectors by Hand

##### A. For $\lambda_1 = 1.4$:
$$(W - 1.4 I) v_1 = \begin{bmatrix} 1.2 - 1.4 & 0.4 \\ 0.4 & 0.6 - 1.4 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} -0.2 & 0.4 \\ 0.4 & -0.8 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
Equation: $-0.2x + 0.4y = 0 \implies x = 2y$.
Let $y = 1 \implies x = 2 \implies v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$.
Normalize to unit length $\|v_1\| = \sqrt{2^2 + 1^2} = \sqrt{5}$:
$$q_1 = \begin{bmatrix} 2/\sqrt{5} \\ 1/\sqrt{5} \end{bmatrix}$$

##### B. For $\lambda_2 = 0.4$:
$$(W - 0.4 I) v_2 = \begin{bmatrix} 1.2 - 0.4 & 0.4 \\ 0.4 & 0.6 - 0.4 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0.8 & 0.4 \\ 0.4 & 0.2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
Equation: $0.8x + 0.4y = 0 \implies y = -2x$.
Let $x = 1 \implies y = -2 \implies v_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$.
Normalize to unit length $\|v_2\| = \sqrt{1^2 + (-2)^2} = \sqrt{5}$:
$$q_2 = \begin{bmatrix} 1/\sqrt{5} \\ -2/\sqrt{5} \end{bmatrix}$$

*Orthogonality Check:*
$$q_1^T q_2 = \left( \frac{2}{\sqrt{5}} \times \frac{1}{\sqrt{5}} \right) + \left( \frac{1}{\sqrt{5}} \times \frac{-2}{\sqrt{5}} \right) = \frac{2 - 2}{5} = \mathbf{0} \quad \checkmark$$

---

#### 5. Step 3: Decomposing Initial State $h_0$ along Eigen-Coordinates
Notice that $h_0 = \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \sqrt{5} q_1 + 0 q_2$.
Let us verify the coordinate projections $c = Q^T h_0$:
$$c_1 = q_1^T h_0 = \frac{2(2) + 1(1)}{\sqrt{5}} = \frac{5}{\sqrt{5}} = \sqrt{5}$$
$$c_2 = q_2^T h_0 = \frac{1(2) + (-2)(1)}{\sqrt{5}} = \frac{0}{\sqrt{5}} = 0$$

---

#### 6. Step 4: Evolution Over Time $t$ by Hand

$$h_t = W^t h_0 = c_1 \lambda_1^t q_1 + c_2 \lambda_2^t q_2 = \sqrt{5} (1.4)^t \begin{bmatrix} 2/\sqrt{5} \\ 1/\sqrt{5} \end{bmatrix} + 0 = (1.4)^t \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

| Timestep $t$ | Growth Factor $(1.4)^t$ | Hidden State $h_t$ | Norm $\|h_t\|_2$ | Dynamic Behavior |
| :---: | :---: | :---: | :---: | :--- |
| **$t = 0$** | $(1.4)^0 = 1.000$ | $[2.000, 1.000]^T$ | $2.236$ | Baseline |
| **$t = 1$** | $(1.4)^1 = 1.400$ | $[2.800, 1.400]^T$ | $3.130$ | Growing ($+40\%$) |
| **$t = 5$** | $(1.4)^5 \approx 5.378$ | $[10.756, 5.378]^T$ | $12.027$ | Exploding ($5.38\times$) |
| **$t = 10$**| $(1.4)^{10} \approx 28.925$ | $[57.851, 28.925]^T$ | $64.679$ | **Exploded ($28.9\times$)!** |

> [!IMPORTANT]
> **The Mathematical Root of Exploding Gradients:**
> Because the spectral radius is $\rho(W) = 1.4 > 1.0$, the recurrent state expands by $1.4\times$ at every single step! Over 100 sequence tokens, $(1.4)^{100} \approx 3.9 \times 10^{14}$—the gradients completely overflow the floating-point exponent!
> Conversely, any component along $q_2$ shrinks by $(0.4)^{100} \approx 1.6 \times 10^{-40}$—**vanishing to exact zero**.

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Standard Case — Diagonalizing a Symmetric Matrix

**Problem:** Find the orthogonal diagonalization $A = Q \Lambda Q^T$ for:
$$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$$

1. **Eigenvalues:**
   $$\det(A - \lambda I) = (3 - \lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda - 4)(\lambda - 2) = 0$$
   $$\lambda_1 = 4, \quad \lambda_2 = 2$$
2. **Eigenvectors:**
   - For $\lambda_1 = 4$: $(A - 4I)v = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \mathbf{0} \implies x = y \implies q_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$.
   - For $\lambda_2 = 2$: $(A - 2I)v = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \mathbf{0} \implies y = -x \implies q_2 = \begin{bmatrix} -1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$.
3. **Assemble $Q$ and $\Lambda$:**
   $$Q = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}$$
4. **Spectral Decomposition Verification:**
   $$A = 4 q_1 q_1^T + 2 q_2 q_2^T = 4 \begin{bmatrix} 1/2 & 1/2 \\ 1/2 & 1/2 \end{bmatrix} + 2 \begin{bmatrix} 1/2 & -1/2 \\ -1/2 & 1/2 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} + \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \quad \checkmark$$

---

### Scenario B: Boundary / Pathological Case — A Defective Matrix

**Problem:** Show that the shear matrix $M = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$ cannot be diagonalized.

1. **Characteristic Equation:**
   $$\det(M - \lambda I) = (2 - \lambda)(2 - \lambda) - 0 = (2 - \lambda)^2 = 0 \implies \lambda = 2 \; (\text{AM} = 2)$$
2. **Finding Eigenvectors (Geometric Multiplicity):**
   $$(M - 2I) v = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies y = 0, \quad x \text{ is free}$$
   The nullspace $N(M - 2I)$ is spanned by only a single vector:
   $$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \implies \text{GM}(\lambda = 2) = 1$$
3. **Analysis:**
   Since $\text{GM} = 1 < \text{AM} = 2$, the matrix is **defective**. It is physically impossible to find 2 independent eigenvectors in $\mathbb{R}^2$ for $M$. Hence, $M$ cannot be diagonalized into $S \Lambda S^{-1}$.

---

### Scenario C: Boundary Case — Complex Eigenvalues from Pure Rotation

**Problem:** Find eigenvalues of the $90^\circ$ rotation matrix $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.

1. **Characteristic Equation:**
   $$\det(R - \lambda I) = (-\lambda)(-\lambda) - (-1)(1) = \lambda^2 + 1 = 0 \implies \lambda = \pm i$$
2. **Geometric Insight:**
   A rotation by $90^\circ$ turns *every* real vector in the 2D plane into an orthogonal direction. No real vector can ever satisfy $R v = \lambda v$ with real $\lambda$. The eigenvectors exist only in the complex plane $\mathbb{C}^2$!

---

### Scenario D: Power Iteration for Dominant Eigenvalue & Eigenvector

**Problem Formulation:**
In large-scale graph neural networks (PageRank) and spectral clustering, computing the full eigendecomposition of an $N \times N$ matrix ($N = 10^7$) is impossible.
Instead, **Power Iteration** estimates the dominant eigenvalue $\lambda_1$ and eigenvector $q_1$ via repeated matrix-vector multiplication:
$$y_{k+1} = A x_k, \quad x_{k+1} = \frac{y_{k+1}}{\|y_{k+1}\|_2}, \quad \mu_{k+1} = x_{k+1}^T A x_{k+1} \; \text{(Rayleigh Quotient)}$$

Trace 3 iterations of Power Iteration for symmetric matrix $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$ starting from $x_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

#### Step 1: Iteration 1 ($k = 0 \to 1$)
$$y_1 = A x_0 = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$$
$$\|y_1\|_2 = \sqrt{3^2 + 1^2} = \sqrt{10} \approx 3.162278$$
$$x_1 = \frac{1}{\sqrt{10}} \begin{bmatrix} 3 \\ 1 \end{bmatrix} \approx \begin{bmatrix} 0.948683 \\ 0.316228 \end{bmatrix}$$
**Rayleigh Quotient Estimate $\mu_1$:**
$$\mu_1 = x_1^T A x_1 = \frac{1}{10} \begin{bmatrix} 3 & 1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \frac{1}{10} \begin{bmatrix} 3 & 1 \end{bmatrix} \begin{bmatrix} 10 \\ 6 \end{bmatrix} = \frac{30 + 6}{10} = \mathbf{3.600000}$$

#### Step 2: Iteration 2 ($k = 1 \to 2$)
$$y_2 = A x_1 = \frac{1}{\sqrt{10}} \begin{bmatrix} 10 \\ 6 \end{bmatrix}$$
$$\|y_2\|_2 = \frac{1}{\sqrt{10}} \sqrt{10^2 + 6^2} = \frac{\sqrt{136}}{\sqrt{10}} \approx 3.687818$$
$$x_2 = \frac{1}{\sqrt{136}} \begin{bmatrix} 10 \\ 6 \end{bmatrix} = \frac{1}{2\sqrt{34}} \begin{bmatrix} 10 \\ 6 \end{bmatrix} = \begin{bmatrix} 5/\sqrt{34} \\ 3/\sqrt{34} \end{bmatrix} \approx \begin{bmatrix} 0.857493 \\ 0.514496 \end{bmatrix}$$
**Rayleigh Quotient Estimate $\mu_2$:**
$$\mu_2 = x_2^T A x_2 = \frac{1}{34} \begin{bmatrix} 5 & 3 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 5 \\ 3 \end{bmatrix} = \frac{1}{34} \begin{bmatrix} 5 & 3 \end{bmatrix} \begin{bmatrix} 18 \\ 14 \end{bmatrix} = \frac{90 + 42}{34} = \frac{132}{34} \approx \mathbf{3.882353}$$

#### Step 3: Iteration 3 ($k = 2 \to 3$)
$$y_3 = A x_2 = \frac{1}{\sqrt{34}} \begin{bmatrix} 18 \\ 14 \end{bmatrix} = \frac{2}{\sqrt{34}} \begin{bmatrix} 9 \\ 7 \end{bmatrix}$$
$$\|y_3\|_2 = \frac{2}{\sqrt{34}} \sqrt{9^2 + 7^2} = \frac{2\sqrt{130}}{\sqrt{34}}$$
$$x_3 = \frac{1}{\sqrt{130}} \begin{bmatrix} 9 \\ 7 \end{bmatrix} \approx \begin{bmatrix} 0.789352 \\ 0.613941 \end{bmatrix}$$
**Rayleigh Quotient Estimate $\mu_3$:**
$$\mu_3 = x_3^T A x_3 = \frac{1}{130} \begin{bmatrix} 9 & 7 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 9 \\ 7 \end{bmatrix} = \frac{1}{130} \begin{bmatrix} 9 & 7 \end{bmatrix} \begin{bmatrix} 34 \\ 30 \end{bmatrix} = \frac{306 + 210}{130} = \frac{516}{130} \approx \mathbf{3.969231}$$

*Convergence Summary:*
- $k=1$: $\mu_1 = 3.600000$ (Error: $10.0\%$)
- $k=2$: $\mu_2 = 3.882353$ (Error: $2.94\%$)
- $k=3$: $\mu_3 = 3.969231$ (Error: $0.77\%$)
- True eigenvalue $\lambda_1 = 4.0$. Convergence is governed by the spectral gap ratio $\left|\frac{\lambda_2}{\lambda_1}\right|^k = \left(\frac{2}{4}\right)^k = (0.5)^k$, halving error every iteration!

---

### Scenario E: High Matrix Powers via Spectral Decomposition

**Problem Formulation:**
In graph neural networks and diffusion processes, we must compute high powers of transition matrices $A^k$ where $k = 10$ and $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$.
1. Compute $A^{10}$ analytically via its spectral decomposition $A = Q \Lambda Q^T$.
2. Write down the exact integer matrix result.

#### Step 1: Diagonal Matrix Power $\Lambda^{10}$
From Scenario A:
$$\Lambda = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix} \implies \Lambda^{10} = \begin{bmatrix} 4^{10} & 0 \\ 0 & 2^{10} \end{bmatrix}$$
Notice that $4^{10} = (2^2)^{10} = 2^{20} = 1,048,576$ and $2^{10} = 1,024$.

#### Step 2: Outer Product Reconstruction $A^{10} = Q \Lambda^{10} Q^T$
With $Q = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$:
$$A^{10} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 4^{10} & 0 \\ 0 & 2^{10} \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}$$
$$A^{10} = \frac{1}{2} \begin{bmatrix} 4^{10} & -2^{10} \\ 4^{10} & 2^{10} \end{bmatrix} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}$$

Compute each entry:
- $(A^{10})_{11} = \frac{1}{2} \left( 4^{10}(1) + (-2^{10})(-1) \right) = \frac{4^{10} + 2^{10}}{2} = \frac{1,048,576 + 1,024}{2} = \frac{1,049,600}{2} = \mathbf{524,800}$
- $(A^{10})_{12} = \frac{1}{2} \left( 4^{10}(1) + (-2^{10})(1) \right) = \frac{4^{10} - 2^{10}}{2} = \frac{1,048,576 - 1,024}{2} = \frac{1,047,552}{2} = \mathbf{523,776}$
- $(A^{10})_{21} = \frac{1}{2} \left( 4^{10}(1) + (2^{10})(-1) \right) = \frac{4^{10} - 2^{10}}{2} = \mathbf{523,776}$
- $(A^{10})_{22} = \frac{1}{2} \left( 4^{10}(1) + (2^{10})(1) \right) = \frac{4^{10} + 2^{10}}{2} = \mathbf{524,800}$

$$A^{10} = \begin{bmatrix} 524800 & 523776 \\ 523776 & 524800 \end{bmatrix}$$
*Insight:* Eigendecomposition replaces 9 chained matrix multiplications with two scalar power operations and a constant-cost basis projection.

---

## Part 7: Deep Learning Connection & Application

### 1. Hessian Loss Curvature & Learning Rate Bounds
In deep neural network optimization, the local loss surface around a point $w$ is approximated by its second-order Taylor expansion:
$$\mathcal{L}(w + \Delta w) \approx \mathcal{L}(w) + \nabla \mathcal{L}^T \Delta w + \frac{1}{2} \Delta w^T H \Delta w$$
where $H = \nabla^2 \mathcal{L}$ is the symmetric Hessian matrix.
- Let $\lambda_{\max}(H)$ be the largest eigenvalue of the Hessian (the direction of sharpest curvature).
- **Fundamental Convergence Theorem:** Standard Gradient Descent converges if and only if the learning rate $\eta$ satisfies:
  $$\eta < \frac{2}{\lambda_{\max}(H)}$$
- If $\eta > \frac{2}{\lambda_{\max}}$, the optimizer oscillates with exponentially growing amplitude along the eigenvector of $\lambda_{\max}$, catastrophically diverging!

### 2. Principal Component Analysis (PCA)
Given zero-centered data $X \in \mathbb{R}^{N \times D}$, the sample covariance matrix is:
$$\Sigma = \frac{1}{N} X^T X \in \mathbb{R}^{D \times D}$$
- $\Sigma$ is symmetric and positive semi-definite.
- By the Spectral Theorem: $\Sigma = Q \Lambda Q^T$.
- The eigenvector $q_1$ corresponding to $\lambda_{\max}$ is the **first principal component**—the direction along which the data exhibits maximum variance ($\text{Var}(X q_1) = \lambda_1$).
- Projecting onto the top $k$ eigenvectors retains the maximum possible signal while compressing dimensionality.

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/06_eigendecomposition_and_spectral_theory.py`](./code/06_eigendecomposition_and_spectral_theory.py) implements:
1. **Symmetric Eigendecomposition Engine**: Computes eigenvalues and orthonormal eigenvectors, verifying $A = Q \Lambda Q^T$.
2. **Spectral Theorem Verifier**: Asserts reality of eigenvalues, pairwise orthogonality of eigenvectors, and trace/determinant invariant equalities.
3. **Defectiveness & Multiplicity Tester**: Identifies defective matrices where $\text{GM} < \text{AM}$.
4. **PyTorch RNN Recurrent Horizon Simulation**: Propagates hidden states over 50 steps across three regimes:
   - $\rho(W) > 1$ (Exploding signal)
   - $\rho(W) < 1$ (Vanishing signal)
   - $\rho(W) = 1$ (Stable unitary signal)

---

## Chapter 1.6 Summary & Review Checklist

- [x] What is the geometric definition of an eigenvector ($A v = \lambda v$)?
- [x] How do you compute eigenvalues using the characteristic polynomial $\det(A - \lambda I) = 0$?
- [x] What makes a matrix "defective", and why does it prevent diagonalization?
- [x] What four guarantees does the Spectral Theorem provide for real symmetric matrices?
- [x] How does the spectral radius $\rho(W_{hh})$ dictate vanishing vs exploding gradients in RNNs?
- [x] Why does the maximum eigenvalue of the Hessian $\lambda_{\max}(H)$ limit the maximum stable learning rate?
