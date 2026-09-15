# Chapter 1.5: Orthogonality, Projections & Least Squares

---

## Part 1: Intuition & 101 Motivation

In Deep Learning and machine learning, we almost always encounter **overdetermined systems**:
- We have $m = 1,000,000$ training examples, but only $n = 100$ features ($m \gg n$).
- We want to find weight vector $x \in \mathbb{R}^n$ such that $A x = b$, where $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$ is the target label vector.

Because the data is noisy, target vector $b$ almost **never** lies inside the column space $C(A)$.
Mathematically, $A x = b$ has **zero solutions**.

When an exact solution is impossible, what is the best possible approximation?
- We seek a vector $\hat{x}$ that makes the model's prediction $p = A \hat{x}$ as close as possible to the target $b$.
- That means minimizing the Euclidean error:
  $$\min_x \|A x - b\|_2^2$$

The geometric answer to this foundational problem is **Orthogonal Projection**:
- To get as close to $b$ as possible while staying on the subspace $C(A)$, we must drop a perpendicular from $b$ onto $C(A)$.
- The prediction $p = A \hat{x}$ is the **orthogonal projection** of $b$ onto $C(A)$.
- The residual error $e = b - p$ is strictly perpendicular to every single feature column in $A$.

This single geometric insight yields **Ordinary Least Squares (OLS)**, the **Normal Equations**, **QR Decomposition**, and the principles behind **Orthogonal Weight Initialization** in deep neural networks.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Orthogonal & Orthonormal Vectors
Let $V$ be an inner product space equipped with the Euclidean dot product.

- Two vectors $u, v \in V$ are **orthogonal** (written $u \perp v$) if their inner product is zero:
  $$\langle u, v \rangle = u^T v = 0$$
- A set of vectors $\{q_1, q_2, \dots, q_k\}$ is **orthonormal** if every vector has unit length ($\|q_i\|_2 = 1$) and all pairs are mutually orthogonal:
  $$q_i^T q_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

---

### 2. Orthogonal Matrices
A square matrix $Q \in \mathbb{R}^{n \times n}$ whose columns are orthonormal is called an **Orthogonal Matrix**:
$$Q^T Q = I_n \implies Q^{-1} = Q^T$$

#### Fundamental Properties of Orthogonal Matrices:
1. **Inverse is Free**: $Q^{-1} = Q^T$ (transposing inverts the matrix with zero computational cost).
2. **Norm Preservation (Isometry)**: For any vector $x \in \mathbb{R}^n$:
   $$\|Q x\|_2^2 = (Q x)^T (Q x) = x^T (Q^T Q) x = x^T I x = \|x\|_2^2 \implies \|Q x\|_2 = \|x\|_2$$
   *Multiplying by $Q$ preserves vector lengths perfectly!*
3. **Angle Preservation**:
   $$\langle Q x, Q y \rangle = (Q x)^T (Q y) = x^T Q^T Q y = x^T y = \langle x, y \rangle$$
   *Orthogonal transformations are pure rigid rotations and reflections—they never distort shapes.*
4. **Determinant**: $\det(Q) = \pm 1$.

---

### 3. Orthogonal Projection onto a 1D Line
Suppose we want to project vector $b \in \mathbb{R}^m$ onto the line spanned by a single non-zero vector $a \in \mathbb{R}^m$.

```
                     b
                     ^
                    /|
                   / |
                  /  | e = b - p  (e ⊥ a)
                 /   |
                /    v
               +-----+---------> a (Line direction)
              0      p = x̂ a
```

1. The projection vector $p$ must be a scalar multiple of $a$:
   $$p = \hat{x} a \quad (\hat{x} \in \mathbb{R})$$
2. The error vector $e = b - p = b - \hat{x} a$ must be perpendicular to $a$:
   $$a^T e = 0 \implies a^T (b - \hat{x} a) = 0$$
   $$a^T b - \hat{x} (a^T a) = 0 \implies \hat{x} = \frac{a^T b}{a^T a}$$
3. The projected vector is:
   $$p = \left( \frac{a^T b}{a^T a} \right) a = a \left( \frac{a^T b}{a^T a} \right) = \left( \frac{a a^T}{a^T a} \right) b$$
4. The **Projection Matrix** $P_a$ onto line $a$ is:
   $$P_a = \frac{a a^T}{a^T a} \in \mathbb{R}^{m \times m} \quad (\text{Rank } 1)$$

---

### 4. Orthogonal Projection onto an Arbitrary Subspace & The Normal Equations
Now consider a general matrix $A \in \mathbb{R}^{m \times n}$ with linearly independent columns ($m > n$).
We want to project $b \in \mathbb{R}^m$ onto the entire $n$-dimensional column space $C(A)$.

1. The projection $p$ lies in $C(A)$, so it is a linear combination of the columns of $A$:
   $$p = A \hat{x} \quad (\hat{x} \in \mathbb{R}^n)$$
2. The error vector $e = b - p = b - A \hat{x}$ must be orthogonal to **every column of $A$**:
   $$\text{col}_i(A)^T e = 0, \quad \forall i \in \{1, \dots, n\} \iff A^T e = \mathbf{0}$$
3. Substitute $e = b - A \hat{x}$:
   $$A^T (b - A \hat{x}) = \mathbf{0}$$
   $$A^T A \hat{x} = A^T b \quad \text{\bf [The Fundamental Normal Equations]}$$
4. Since the columns of $A$ are linearly independent, the symmetric square matrix $A^T A \in \mathbb{R}^{n \times n}$ is strictly positive-definite and **invertible**:
   $$\hat{x} = (A^T A)^{-1} A^T b$$
5. The projected vector $p$ is:
   $$p = A \hat{x} = A (A^T A)^{-1} A^T b$$
6. The **Orthogonal Projection Matrix** onto $C(A)$ is:
   $$P = A (A^T A)^{-1} A^T \in \mathbb{R}^{m \times m}$$

---

### 5. Essential Properties of Any Projection Matrix $P$
For any matrix $P$ representing an orthogonal projection:
1. **Idempotence ($P^2 = P$)**:
   Projecting a point that is already on the subspace changes nothing:
   $$P^2 = \left[ A (A^T A)^{-1} A^T \right] \left[ A (A^T A)^{-1} A^T \right] = A (A^T A)^{-1} \underbrace{(A^T A) (A^T A)^{-1}}_{I} A^T = A (A^T A)^{-1} A^T = P$$
2. **Symmetry ($P^T = P$)**:
   $$P^T = \left( A (A^T A)^{-1} A^T \right)^T = (A^T)^T \left( (A^T A)^{-1} \right)^T A^T = A (A^T A)^{-1} A^T = P$$
3. **The Complementary Projection ($I - P$)**:
   Projects onto the orthogonal complement $C(A)^\perp = N(A^T)$:
   $$e = b - p = b - P b = (I - P) b$$
   Notice that $(I - P)^2 = I - 2P + P^2 = I - 2P + P = I - P$.
4. **Trace equals Rank**:
   $$\text{Tr}(P) = \text{rank}(P) = \dim(C(A)) = n$$

---

### 6. The Gram-Schmidt Orthogonalization Process
Given a set of linearly independent vectors $\{a_1, a_2, \dots, a_n\}$, Gram-Schmidt constructs an orthonormal basis $\{q_1, q_2, \dots, q_n\}$ spanning the exact same subspace:

```
                  a2
                  ^
                 /|
                / |  u2 = a2 - (q1^T a2) q1
               /  |  (perpendicular to q1)
              /   v
             +----+----> q1 = a1 / ||a1||
             0
```

1. **First Vector**:
   $$u_1 = a_1, \quad q_1 = \frac{u_1}{\|u_1\|_2}$$
2. **Second Vector** (Subtract projection onto $q_1$):
   $$u_2 = a_2 - (q_1^T a_2) q_1, \quad q_2 = \frac{u_2}{\|u_2\|_2}$$
3. **General $k$-th Vector** (Subtract projections onto all prior $q_1, \dots, q_{k-1}$):
   $$u_k = a_k - \sum_{j=1}^{k-1} (q_j^T a_k) q_j, \quad q_k = \frac{u_k}{\|u_k\|_2}$$

---

### 7. The $QR$ Factorization
Every matrix $A \in \mathbb{R}^{m \times n}$ with linearly independent columns can be factored into:
$$A = Q R$$
where:
- $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns ($Q^T Q = I_n$).
- $R \in \mathbb{R}^{n \times n}$ is an **upper-triangular invertible matrix**:
  $$R = \begin{bmatrix} q_1^T a_1 & q_1^T a_2 & \dots & q_1^T a_n \\ 0 & q_2^T a_2 & \dots & q_2^T a_n \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & q_n^T a_n \end{bmatrix}$$

#### Why $QR$ is Numerically Superior for Least Squares:
In standard OLS, computing $\hat{x} = (A^T A)^{-1} A^T b$ requires forming $A^T A$. The condition number of $A^T A$ is **squared**: $\kappa(A^T A) = \kappa(A)^2$, leading to catastrophic floating-point precision loss.
Using $QR$ decomposition:
$$A^T A \hat{x} = A^T b \implies (R^T Q^T) (Q R) \hat{x} = R^T Q^T b$$
Since $Q^T Q = I$:
$$R^T R \hat{x} = R^T Q^T b$$
Multiplying by $(R^T)^{-1}$:
$$R \hat{x} = Q^T b$$
Because $R$ is upper-triangular, we solve for $\hat{x}$ instantly via **back-substitution** without ever inverting a matrix or squaring condition numbers!

---

## Part 3: Geometric & Algebraic Interpretation

### The Right Triangle of Least Squares

```
                           b (Observed target labels)
                          /|
                         / |
                        /  |
                       /   | e = b - p  (Residual error vector)
     ||b||            /    | ||e|| = ||b - Ax̂|| (MINIMAL ERROR!)
                     /     |
                    /      |
                   +-------+
                   0       p = Ax̂ (Orthogonal projection in C(A))
                           <------------------------------------->
                                    Column Space C(A)
```

1. By the **Pythagorean Theorem**:
   $$\|b\|_2^2 = \|p\|_2^2 + \|e\|_2^2 = \|A \hat{x}\|_2^2 + \|b - A \hat{x}\|_2^2$$
2. Since $\|b\|_2^2$ is fixed by our dataset, minimizing the error $\|e\|_2^2$ is mathematically identical to maximizing the length of the projected signal $\|p\|_2^2$. This is the geometric engine behind **Coefficient of Determination ($R^2$)**.

---

## Part 4: Real-World Analogy

### The Sunlight & Shadow Analogy
Imagine holding a javelin tilted in 3D air ($b$). At solar noon, sunlight shines straight down perpendicular to the flat horizontal ground ($C(A)$).
- The shadow cast on the ground is the **orthogonal projection $p$**.
- The altitude difference from each point of the javelin straight down to the shadow is the **error vector $e$**.
- If the javelin is already laying flat on the ground, its shadow is the javelin itself ($P b = b$, error $= \mathbf{0}$).
- If you hold the javelin pointing straight up toward the sun, its shadow collapses to a single dot ($P b = \mathbf{0}$, error $= b$).

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### Ordinary Least Squares (OLS) Regression by Hand via Projection

Let us fit a linear regression model $y = \theta_0 + \theta_1 x$ on 3 toy data points:
$$(1, 1), \quad (2, 2), \quad (3, 2)$$

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Shape | Mathematical Role | Machine Learning Meaning | Concrete Demo Value |
| :---: | :---: | :--- | :--- | :---: |
| $A$ | $[3 \times 2]$ | Feature matrix | Column 1: bias ones; Column 2: feature $x$ | $\begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}$ |
| $b$ | $[3 \times 1]$ | Target vector | Observed continuous labels $y$ | $\begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}$ |
| $\hat{x}$ | $[2 \times 1]$ | Parameter vector $[\theta_0, \theta_1]^T$ | Intercept and slope weights | To be solved |
| $p$ | $[3 \times 1]$ | Projection $A \hat{x}$ | Model predictions $\hat{y}$ | To be computed |
| $e$ | $[3 \times 1]$ | Residual error $b - p$ | Prediction errors | To be computed |

---

#### 2. Visual Matrix Setup

```
         FEATURE MATRIX A (3 x 2)             TARGET b (3 x 1)
           Bias (x0)  Feature (x1)
        ┌───────────┬───────────┐               ┌───────────┐
  Pt 1  │   A_11=1  │   A_12=1  │         Pt 1  │   b_1=1   │
        ├───────────┼───────────┤               ├───────────┤
  Pt 2  │   A_21=1  │   A_22=2  │         Pt 2  │   b_2=2   │
        ├───────────┼───────────┤               ├───────────┤
  Pt 3  │   A_31=1  │   A_32=3  │         Pt 3  │   b_3=2   │
        └───────────┴───────────┘               └───────────┘
```

---

#### 3. Step 1: Compute Normal Matrix $A^T A$ ($2 \times 2$)

$$A^T A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}$$

| Cell | Row of $A^T$ | $\cdot$ | Col of $A$ | Exact Arithmetic | Result |
| :---: | :---: | :---: | :---: | :--- | :---: |
| **$(A^T A)_{11}$** | $[1, 1, 1]$ | $\cdot$ | $[1, 1, 1]^T$ | $1(1) + 1(1) + 1(1) = 1 + 1 + 1$ | **$3$** |
| **$(A^T A)_{12}$** | $[1, 1, 1]$ | $\cdot$ | $[1, 2, 3]^T$ | $1(1) + 1(2) + 1(3) = 1 + 2 + 3$ | **$6$** |
| **$(A^T A)_{21}$** | $[1, 2, 3]$ | $\cdot$ | $[1, 1, 1]^T$ | $1(1) + 2(1) + 3(1) = 1 + 2 + 3$ | **$6$** |
| **$(A^T A)_{22}$** | $[1, 2, 3]$ | $\cdot$ | $[1, 2, 3]^T$ | $1(1) + 2(2) + 3(3) = 1 + 4 + 9$ | **$14$** |

$$A^T A = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$$

---

#### 4. Step 2: Compute Target Projection $A^T b$ ($2 \times 1$)

$$A^T b = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}$$

| Cell | Row of $A^T$ | $\cdot$ | Vector $b$ | Exact Arithmetic | Result |
| :---: | :---: | :---: | :---: | :--- | :---: |
| **$(A^T b)_1$** | $[1, 1, 1]$ | $\cdot$ | $[1, 2, 2]^T$ | $1(1) + 1(2) + 1(2) = 1 + 2 + 2$ | **$5$** |
| **$(A^T b)_2$** | $[1, 2, 3]$ | $\cdot$ | $[1, 2, 2]^T$ | $1(1) + 2(2) + 3(2) = 1 + 4 + 6$ | **$11$** |

$$A^T b = \begin{bmatrix} 5 \\ 11 \end{bmatrix}$$

---

#### 5. Step 3: Analytical Inversion of $A^T A$
$$\det(A^T A) = (3 \times 14) - (6 \times 6) = 42 - 36 = \mathbf{6}$$

$$(A^T A)^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} = \begin{bmatrix} 7/3 & -1 \\ -1 & 1/2 \end{bmatrix}$$

---

#### 6. Step 4: Solve for OLS Parameters $\hat{x} = (A^T A)^{-1} A^T b$

$$\hat{x} = \begin{bmatrix} 7/3 & -1 \\ -1 & 1/2 \end{bmatrix} \begin{bmatrix} 5 \\ 11 \end{bmatrix}$$

| Parameter | Row of $(A^T A)^{-1}$ | $\cdot$ | Vector $A^T b$ | Exact Arithmetic | Result |
| :---: | :---: | :---: | :---: | :--- | :---: |
| **$\theta_0$ (Intercept)**| $[7/3, -1]$ | $\cdot$ | $[5, 11]^T$ | $(7/3 \times 5) + (-1 \times 11) = 35/3 - 33/3$ | **$2/3 \approx 0.667$** |
| **$\theta_1$ (Slope)**    | $[-1, 1/2]$ | $\cdot$ | $[5, 11]^T$ | $(-1 \times 5) + (1/2 \times 11) = -5 + 5.5$  | **$1/2 = 0.500$** |

$$\hat{x} = \begin{bmatrix} 2/3 \\ 1/2 \end{bmatrix}$$
Fitted regression line: $y = \frac{2}{3} + \frac{1}{2} x$.

---

#### 7. Step 5: Model Predictions $p = A \hat{x}$ & Residual Errors $e = b - p$

$$p = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 2/3 \\ 1/2 \end{bmatrix} = \begin{bmatrix} 2/3 + 1/2 \\ 2/3 + 2(1/2) \\ 2/3 + 3(1/2) \end{bmatrix} = \begin{bmatrix} 7/6 \\ 10/6 \\ 13/6 \end{bmatrix} \approx \begin{bmatrix} 1.167 \\ 1.667 \\ 2.167 \end{bmatrix}$$

$$e = b - p = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix} - \begin{bmatrix} 7/6 \\ 10/6 \\ 13/6 \end{bmatrix} = \begin{bmatrix} 6/6 - 7/6 \\ 12/6 - 10/6 \\ 12/6 - 13/6 \end{bmatrix} = \begin{bmatrix} -1/6 \\ 2/6 \\ -1/6 \end{bmatrix}$$

---

#### 8. Step 6: Orthogonality Check ($A^T e = \mathbf{0}$)

$$A^T e = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} -1/6 \\ 2/6 \\ -1/6 \end{bmatrix}$$
- **Equation 1**: $1(-1/6) + 1(2/6) + 1(-1/6) = \frac{-1 + 2 - 1}{6} = \frac{0}{6} = \mathbf{0} \quad \checkmark$
- **Equation 2**: $1(-1/6) + 2(2/6) + 3(-1/6) = \frac{-1 + 4 - 3}{6} = \frac{0}{6} = \mathbf{0} \quad \checkmark$

> [!IMPORTANT]
> **The Miracle of Orthogonal Projection:**
> The error vector $e$ is **strictly orthogonal** to both the bias column ($[1, 1, 1]^T$) and the feature column ($[1, 2, 3]^T$).
> It is impossible to reduce the error any further using linear combinations of these features because the residual has zero projection along all available feature dimensions!

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Standard Case — Full Gram-Schmidt $QR$ Factorization

**Problem:** Given matrix $A \in \mathbb{R}^{3 \times 3}$:
$$a_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad a_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad a_3 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}$$
Compute its $Q R$ factorization via Gram-Schmidt.

#### Step 1: Normalize First Vector
$$u_1 = a_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$$
$$\|u_1\|_2 = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2}$$
$$q_1 = \frac{u_1}{\|u_1\|_2} = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{bmatrix}$$

---

#### Step 2: Orthogonalize Second Vector
Subtract projection of $a_2$ onto $q_1$:
$$q_1^T a_2 = \left( \frac{1}{\sqrt{2}} \times 1 \right) + \left( \frac{1}{\sqrt{2}} \times 0 \right) + (0 \times 1) = \frac{1}{\sqrt{2}}$$
$$u_2 = a_2 - (q_1^T a_2) q_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \frac{1}{\sqrt{2}} \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \begin{bmatrix} 1/2 \\ 1/2 \\ 0 \end{bmatrix} = \begin{bmatrix} 1/2 \\ -1/2 \\ 1 \end{bmatrix}$$
$$\|u_2\|_2 = \sqrt{(1/2)^2 + (-1/2)^2 + 1^2} = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2} = \frac{\sqrt{6}}{2}$$
$$q_2 = \frac{u_2}{\|u_2\|_2} = \frac{2}{\sqrt{6}} \begin{bmatrix} 1/2 \\ -1/2 \\ 1 \end{bmatrix} = \begin{bmatrix} 1/\sqrt{6} \\ -1/\sqrt{6} \\ 2/\sqrt{6} \end{bmatrix}$$

---

#### Step 3: Orthogonalize Third Vector
Subtract projections of $a_3$ onto $q_1$ and $q_2$:
$$q_1^T a_3 = \left( \frac{1}{\sqrt{2}} \times 0 \right) + \left( \frac{1}{\sqrt{2}} \times 1 \right) + (0 \times 1) = \frac{1}{\sqrt{2}}$$
$$q_2^T a_3 = \left( \frac{1}{\sqrt{6}} \times 0 \right) + \left( -\frac{1}{\sqrt{6}} \times 1 \right) + \left( \frac{2}{\sqrt{6}} \times 1 \right) = \frac{1}{\sqrt{6}}$$
$$u_3 = a_3 - (q_1^T a_3) q_1 - (q_2^T a_3) q_2$$
$$u_3 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix} - \frac{1}{\sqrt{2}} \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{bmatrix} - \frac{1}{\sqrt{6}} \begin{bmatrix} 1/\sqrt{6} \\ -1/\sqrt{6} \\ 2/\sqrt{6} \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} 1/2 \\ 1/2 \\ 0 \end{bmatrix} - \begin{bmatrix} 1/6 \\ -1/6 \\ 2/6 \end{bmatrix} = \begin{bmatrix} -2/3 \\ 2/3 \\ 2/3 \end{bmatrix}$$
$$\|u_3\|_2 = \sqrt{(-2/3)^2 + (2/3)^2 + (2/3)^2} = \sqrt{4/9 + 4/9 + 4/9} = \sqrt{12/9} = \frac{2\sqrt{3}}{3} = \frac{2}{\sqrt{3}}$$
$$q_3 = \frac{u_3}{\|u_3\|_2} = \frac{\sqrt{3}}{2} \begin{bmatrix} -2/3 \\ 2/3 \\ 2/3 \end{bmatrix} = \begin{bmatrix} -1/\sqrt{3} \\ 1/\sqrt{3} \\ 1/\sqrt{3} \end{bmatrix}$$

---

#### Step 4: Assemble $Q$ and $R$
$$Q = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{6} & -1/\sqrt{3} \\ 1/\sqrt{2} & -1/\sqrt{6} & 1/\sqrt{3} \\ 0 & 2/\sqrt{6} & 1/\sqrt{3} \end{bmatrix}$$

$$R = \begin{bmatrix} \|u_1\| & q_1^T a_2 & q_1^T a_3 \\ 0 & \|u_2\| & q_2^T a_3 \\ 0 & 0 & \|u_3\| \end{bmatrix} = \begin{bmatrix} \sqrt{2} & 1/\sqrt{2} & 1/\sqrt{2} \\ 0 & \sqrt{3/2} & 1/\sqrt{6} \\ 0 & 0 & 2/\sqrt{3} \end{bmatrix}$$

---

### Scenario B: Boundary Cases of Projection
1. **Target $b$ lies strictly inside $C(A)$:**
   Then $b = A x$ for some $x$.
   $$P b = A (A^T A)^{-1} A^T (A x) = A (A^T A)^{-1} (A^T A) x = A x = b$$
   Residual error: $e = b - P b = b - b = \mathbf{0}$.
2. **Target $b$ is strictly perpendicular to $C(A)$ ($b \in N(A^T)$):**
   Then $A^T b = \mathbf{0}$.
   $$P b = A (A^T A)^{-1} (A^T b) = A (A^T A)^{-1} \mathbf{0} = \mathbf{0}$$
   Residual error: $e = b - P b = b - \mathbf{0} = b$.

---

## Part 7: Deep Learning Connection & Application

### 1. Orthogonal Weight Initialization (`nn.init.orthogonal_`)
In deep networks (Transformers, deep RNNs), initializing weights with Gaussian noise leads to random correlations between weight vectors, causing eigenvalues of the layer Jacobian to spread out. Some directions vanish while others explode.
- **Solution**: Sample a random Gaussian matrix $M \sim \mathcal{N}(0, I)$ and perform $QR$ factorization:
  $$M = Q R$$
- Initialize weights with $W = Q$.
- Since $Q^T Q = I$, all singular values of $W$ are identically $1.0$ ($\sigma_i = 1.0$).
- Gradients and signals pass through $W$ with **exact unit norm preservation**, enabling stable training of 100+ layer architectures without batch normalization!

### 2. Low-Rank Subspace Projection in Linear Autoencoders
A linear autoencoder with encoder $W_e \in \mathbb{R}^{k \times d}$ and decoder $W_d \in \mathbb{R}^{d \times k}$ ($k \ll d$) trained with MSE loss:
$$\min_{W_e, W_d} \|x - W_d W_e x\|_2^2$$
learns to form an **orthogonal projection matrix**:
$$P = W_d W_e = U_k U_k^T$$
where $U_k$ contains the top $k$ principal eigenvectors of the data covariance matrix $\Sigma_x$. The autoencoder is identical to orthogonal projection onto the principal subspace!

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/05_orthogonality_projections_and_least_squares.py`](./code/05_orthogonality_projections_and_least_squares.py) implements:
1. **Modified Gram-Schmidt (MGS) QR Decomposition**: Numerically stable implementation producing orthonormal $Q$ and upper-triangular $R$.
2. **Orthogonal Projection Matrix Generator**: Computes $P = A (A^T A)^{-1} A^T$, verifying idempotence ($P^2 = P$) and symmetry ($P^T = P$).
3. **Ordinary Least Squares (OLS) Regression Engine**: Solves $A x = b$ via both Normal Equations and QR decomposition, verifying the residual orthogonality $A^T e = \mathbf{0}$.
4. **PyTorch Orthogonal Initialization Demo**: Demonstrates `torch.nn.init.orthogonal_` verifying norm conservation $\|W x\|_2 = \|x\|_2$.

---

## Chapter 1.5 Summary & Review Checklist

- [x] Why does $A x = b$ have no exact solution when $b \notin C(A)$?
- [x] How does requiring $e \perp C(A)$ directly produce the Normal Equations $A^T A \hat{x} = A^T b$?
- [x] What are the two defining mathematical properties of any projection matrix ($P^2 = P$ and $P^T = P$)?
- [x] Why is $QR$ decomposition numerically superior to forming $A^T A$ directly?
- [x] How does orthogonal initialization prevent vanishing/exploding gradients in deep networks?
