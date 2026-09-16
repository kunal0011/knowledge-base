# Chapter 1.7: Singular Value Decomposition (SVD)

---

## Part 1: Intuition & 101 Motivation

In Chapter 1.6, we learned that square symmetric matrices can be diagonalized into their eigenvalues and eigenvectors ($A = Q \Lambda Q^T$).
However, **real-world deep learning is dominated by rectangular matrices**:
- A dataset batch $X \in \mathbb{R}^{B \times D}$ (e.g., $B = 128$ tokens, $D = 4096$ embedding dimension).
- A linear layer projection $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$.
- A cross-attention key-value cache $K, V \in \mathbb{R}^{T \times d_k}$.

Rectangular matrices do not have eigenvalues or eigenvectors in the standard sense.
What is the universal factorization that applies to **every single matrix in existence**—rectangular or square, full-rank or singular, real or complex?

The answer is the crown jewel of linear algebra: **Singular Value Decomposition (SVD)**:
$$A = U \Sigma V^T$$

SVD reveals the deepest structural secrets of any linear mapping:
1. **Geometric Simplicity**: Every matrix transformation—no matter how high-dimensional or entangled—can be factored into three simple sequential actions: **Rotate $\to$ Scale $\to$ Rotate**.
2. **Optimal Compression (The Eckart-Young-Mirsky Theorem)**: SVD gives the mathematically perfect, optimal low-rank approximation of any matrix. This theorem directly powers:
   - **Low-Rank Adaptation (LoRA)**
   - **Weight matrix pruning and distillation**
   - **Principal Component Analysis (PCA)**
   - **Latent Semantic Analysis (LSA)**

---

## Part 2: Rigorous Mathematical Formulation

### 1. The Fundamental SVD Theorem
Let $A \in \mathbb{R}^{m \times n}$ be an arbitrary matrix with rank $r \le \min(m, n)$.
There exists a factorization:
$$A = U \Sigma V^T$$
where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix whose columns are the **left singular vectors** $\{u_1, u_2, \dots, u_m\}$ ($U^T U = I_m$).
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix whose columns are the **right singular vectors** $\{v_1, v_2, \dots, v_n\}$ ($V^T V = I_n$).
- $\Sigma \in \mathbb{R}^{m \times n}$ is a rectangular diagonal matrix with non-negative entries sorted in descending order:
  $$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > \sigma_{r+1} = \dots = \sigma_{\min(m, n)} = 0$$
  The values $\sigma_i$ are the **singular values** of $A$.

```
               FULL SVD DECOMPOSITION: A = U · Σ · V^T
       ┌───────────────┐     ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
       │               │     │               │   │ σ1            │   │               │
       │               │     │               │   │    σ2         │   │      V^T      │
       │       A       │  =  │       U       │ · │       \       │ · │    (n x n)    │
       │    (m x n)    │     │    (m x m)    │   │        0      │   │               │
       │               │     │               │   │               │   └───────────────┘
       └───────────────┘     └───────────────┘   └───────────────┘
                                                      (m x n)
```

---

### 2. Connection to the Symmetric Matrices $A^T A$ and $A A^T$
How do we find $U$, $\Sigma$, and $V$? We connect $A$ to the symmetric matrices $A^T A$ and $A A^T$:

#### A. The Right Singular Vectors $V$ and $A^T A$:
$$A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T \underbrace{U^T U}_{I_m} \Sigma V^T = V (\Sigma^T \Sigma) V^T$$
- Notice that $\Sigma^T \Sigma \in \mathbb{R}^{n \times n}$ is a diagonal matrix with entries $\sigma_1^2, \sigma_2^2, \dots, \sigma_r^2, 0, \dots, 0$.
- By the Spectral Theorem, $A^T A = Q \Lambda Q^T$.
- **Conclusion:**
  1. The right singular vectors $\{v_i\}$ are the **orthonormal eigenvectors of $A^T A$**.
  2. The singular values are the square roots of the eigenvalues of $A^T A$:
     $$\sigma_i = \sqrt{\lambda_i(A^T A)}$$

#### B. The Left Singular Vectors $U$ and $A A^T$:
$$A A^T = (U \Sigma V^T) (U \Sigma V^T)^T = U \Sigma \underbrace{V^T V}_{I_n} \Sigma^T U^T = U (\Sigma \Sigma^T) U^T$$
- **Conclusion:**
  1. The left singular vectors $\{u_i\}$ are the **orthonormal eigenvectors of $A A^T$**.
  2. For each non-zero singular value $\sigma_i > 0$, $u_i$ is directly coupled to $v_i$ by:
     $$u_i = \frac{1}{\sigma_i} A v_i \iff A v_i = \sigma_i u_i$$

---

#### Rigorous First-Principles Derivation / Proof of SVD Existence:
**Theorem:** For any matrix $A \in \mathbb{R}^{m \times n}$ with rank $r \le \min(m, n)$, there exist orthonormal sets $\{v_1, \dots, v_n\} \subset \mathbb{R}^n$ and $\{u_1, \dots, u_m\} \subset \mathbb{R}^m$, and positive scalars $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ such that:
$$A = \sum_{i=1}^r \sigma_i u_i v_i^T = U \Sigma V^T$$

**Proof Steps:**
1. Form the symmetric matrix $M = A^T A \in \mathbb{R}^{n \times n}$.
   By the Spectral Theorem, $M$ has an orthonormal basis of eigenvectors $\{v_1, \dots, v_n\}$ with real eigenvalues:
   $$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$$
2. Since $\|A x\|_2^2 = x^T A^T A x \ge 0$, all eigenvalues are non-negative: $\lambda_i \ge 0$.
   Because $\text{rank}(A^T A) = \text{rank}(A) = r$, exactly $r$ eigenvalues are strictly positive:
   $$\lambda_1 \ge \dots \ge \lambda_r > 0, \quad \lambda_{r+1} = \dots = \lambda_n = 0$$
   Define the singular values as $\sigma_i = \sqrt{\lambda_i} > 0$ for $i \in \{1, \dots, r\}$.
3. For $i \in \{1, \dots, r\}$, define:
   $$u_i \triangleq \frac{1}{\sigma_i} A v_i \in \mathbb{R}^m$$
4. **Prove $\{u_1, \dots, u_r\}$ is Orthonormal in $\mathbb{R}^m$:**
   $$\begin{aligned}
   u_i^T u_j &= \left(\frac{1}{\sigma_i} A v_i\right)^T \left(\frac{1}{\sigma_j} A v_j\right) = \frac{1}{\sigma_i \sigma_j} v_i^T (A^T A v_j) \\
   &= \frac{1}{\sigma_i \sigma_j} v_i^T (\lambda_j v_j) = \frac{\lambda_j}{\sigma_i \sigma_j} v_i^T v_j
   \end{aligned}$$
   - If $i \neq j$: $v_i^T v_j = 0 \implies u_i^T u_j = 0$ (Orthogonal).
   - If $i = j$: $\frac{\lambda_i}{\sigma_i^2} \|v_i\|_2^2 = \frac{\sigma_i^2}{\sigma_i^2} (1) = 1$ (Unit length).
5. Using Gram-Schmidt orthogonalization, extend $\{u_1, \dots, u_r\}$ to a full orthonormal basis $\{u_1, \dots, u_m\}$ for $\mathbb{R}^m$.
6. Form orthogonal matrices $U = [u_1 \dots u_m]$ and $V = [v_1 \dots v_n]$.
   For any basis vector $v_j$:
   - For $j \le r$: $A v_j = \sigma_j u_j$.
   - For $j > r$: $v_j \in N(A^T A) = N(A) \implies A v_j = \mathbf{0}$.
   Thus in matrix form:
   $$A V = U \Sigma \implies A = U \Sigma V^T \quad \blacksquare$$

---

### 3. Full SVD vs. Compact (Thin) SVD vs. Truncated SVD

| SVD Form | Dimensions | When to Use in Deep Learning |
| :--- | :--- | :--- |
| **Full SVD** | $U \in \mathbb{R}^{m \times m}, \Sigma \in \mathbb{R}^{m \times n}, V \in \mathbb{R}^{n \times n}$ | Theoretical proofs, complete subspace bases. |
| **Compact (Thin) SVD** | $U_r \in \mathbb{R}^{m \times r}, \Sigma_r \in \mathbb{R}^{r \times r}, V_r \in \mathbb{R}^{n \times r}$ | Exact reconstruction without storing zero blocks. |
| **Truncated SVD (Rank $k$)**| $U_k \in \mathbb{R}^{m \times k}, \Sigma_k \in \mathbb{R}^{k \times k}, V_k \in \mathbb{R}^{n \times k}$ | **LoRA, PCA, weight compression, model distillation**. |

---

### 4. The Eckart-Young-Mirsky Theorem (Optimal Low-Rank Approximation)
**Theorem:** Let $A \in \mathbb{R}^{m \times n}$ have singular value expansion $A = \sum_{i=1}^r \sigma_i u_i v_i^T$.
For any integer $k < r$, the truncated SVD matrix:
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T = U_k \Sigma_k V_k^T$$
is the **globally optimal rank-$k$ approximation** to $A$ under both the Spectral Norm ($\|\cdot\|_2$) and the Frobenius Norm ($\|\cdot\|_F$):

$$\min_{\text{rank}(B) \le k} \|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1}$$
$$\min_{\text{rank}(B) \le k} \|A - B\|_F^2 = \|A - A_k\|_F^2 = \sum_{j=k+1}^r \sigma_j^2$$

#### First-Principles Derivation / Proof for Spectral Norm:
1. **Upper Bound:** For $B = A_k$:
   $$A - A_k = \sum_{j=k+1}^r \sigma_j u_j v_j^T$$
   The largest singular value of this remainder matrix is $\sigma_{k+1}$.
   Therefore: $\|A - A_k\|_2 = \sigma_{k+1}$.
2. **Lower Bound for Any Arbitrary Matrix $B$ with $\text{rank}(B) \le k$:**
   - By the Rank-Nullity Theorem, the nullspace of $B$ has dimension:
     $$\dim(N(B)) = n - \text{rank}(B) \ge n - k$$
   - Define the subspace spanned by the first $k+1$ right singular vectors of $A$:
     $$W = \text{span}(v_1, v_2, \dots, v_{k+1}) \subseteq \mathbb{R}^n, \quad \dim(W) = k + 1$$
   - By the dimension formula for vector subspaces:
     $$\dim(W \cap N(B)) = \dim(W) + \dim(N(B)) - \dim(W + N(B)) \ge (k + 1) + (n - k) - n = 1$$
   - Therefore, there exists at least one unit vector $z \in W \cap N(B)$ with $\|z\|_2 = 1$.
   - Because $z \in N(B)$, $B z = \mathbf{0}$.
   - Thus:
     $$\|(A - B) z\|_2 = \|A z - \mathbf{0}\|_2 = \|A z\|_2$$
   - Since $z \in W$ and $\|z\|_2 = 1$, write $z = \sum_{j=1}^{k+1} c_j v_j$ with $\sum_{j=1}^{k+1} c_j^2 = 1$.
   - Compute $\|A z\|_2^2$:
     $$\|A z\|_2^2 = \left\| \sum_{j=1}^{k+1} c_j \sigma_j u_j \right\|_2^2 = \sum_{j=1}^{k+1} c_j^2 \sigma_j^2 \ge \sigma_{k+1}^2 \sum_{j=1}^{k+1} c_j^2 = \sigma_{k+1}^2$$
   - Hence $\|A - B\|_2 = \sup_{\|x\|=1} \|(A - B) x\|_2 \ge \|(A - B) z\|_2 \ge \sigma_{k+1}$.
3. **Conclusion:**
   $$\|A - B\|_2 \ge \sigma_{k+1} \quad \text{for all } \text{rank}(B) \le k$$
   Since $A_k$ achieves this lower bound with equality, $A_k$ is the unique optimal rank-$k$ approximation. $\blacksquare$

> [!IMPORTANT]
> **Why this matters for AI:**
> No algorithm in existence—whether a neural network, genetic algorithm, or convex solver—can ever find a rank-$k$ matrix closer to $A$ than $A_k$. The error is bounded purely by the discarded singular values $\sigma_{k+1}, \dots, \sigma_r$.

---

### 5. The Moore-Penrose Pseudoinverse ($A^+$)
When $A \in \mathbb{R}^{m \times n}$ is rectangular or rank-deficient, the standard inverse $A^{-1}$ does not exist.
Using SVD, the **Moore-Penrose Pseudoinverse** is defined as:
$$A^+ = V \Sigma^+ U^T = \sum_{i=1}^r \frac{1}{\sigma_i} v_i u_i^T$$
where $\Sigma^+ \in \mathbb{R}^{n \times m}$ is obtained by transposing $\Sigma$ and inverting all non-zero singular values ($\sigma_i \to \frac{1}{\sigma_i}$).

#### The Four Penrose Conditions (Unique Axioms of $A^+$):
1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^T = A A^+$ ($A A^+$ is the orthogonal projection onto $C(A)$)
4. $(A^+ A)^T = A^+ A$ ($A^+ A$ is the orthogonal projection onto $C(A^T)$)

For any system $A x = b$, the vector:
$$x^+ = A^+ b$$
is the **minimum-norm least-squares solution**: it minimizes $\|A x - b\|_2$, and if multiple solutions exist, it selects the unique one with minimal length $\|x\|_2$.

---

## Part 3: Geometric & Algebraic Interpretation

### The Geometry of SVD: Rotate $\to$ Stretch $\to$ Rotate

```
      Input Space R^n (Unit Sphere)                    Output Space R^m (Hyper-Ellipsoid)
                     y                                               y
                     ^                                               ^
                   .-'-.                                             |      u2 (Axis 2, length σ2)
                 .'     '.        V^T               Σ            U   |     /
                /    x    \   ─────────>   ────────────────>  ─────> |    /
               |     +     |  (Rotate)     (Stretch by σi)    (Rotate|   +-------> u1 (Axis 1, length σ1)
                \         /                                          |    \
                 '.     .'                                           |     \
                   '-.-'                                             +-------------------> x
```

1. **Step 1 ($V^T x$):** The input vector $x$ is projected onto the orthonormal right singular vectors $v_1, \dots, v_n$. This is a pure rigid rotation of coordinate axes (no shape distortion).
2. **Step 2 ($\Sigma (V^T x)$):** Each coordinate is multiplied by its singular value $\sigma_i$. The unit hypersphere is stretched along orthogonal axes into a hyper-ellipsoid with principal semi-axes $\sigma_1, \sigma_2, \dots, \sigma_r$.
3. **Step 3 ($U (\Sigma V^T x)$):** The stretched ellipsoid is rigidly rotated into the output space $\mathbb{R}^m$, aligning its axes with the left singular vectors $u_1, \dots, u_m$.

---

## Part 4: Real-World Analogy

### The Master Sculptor & Marble Slab Analogy
Imagine sculpting a human statue from a block of marble:
- **Rank-1 Sheet $1$ ($\sigma_1 u_1 v_1^T$):** The sculptor cuts away the bulk stone, establishing the coarse silhouette and posture (accounts for $80\%$ of the visual variance).
- **Rank-1 Sheet $2$ ($\sigma_2 u_2 v_2^T$):** The sculptor carves the head, arms, and torso (accounts for $15\%$ of variance).
- **Rank-1 Sheets $3$ to $k$:** Facial features, muscle contours, and clothing folds.
- **Rank-1 Sheets $k+1$ to $r$:** Microscopic surface grain and microscopic scratches.
- By stopping at rank $k$, you retain $99.9\%$ of the statue's recognizable beauty while discarding the noise. This is exactly how SVD compresses billion-parameter neural networks!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### SVD and Rank-1 Truncation by Hand

Let us perform the complete Singular Value Decomposition on a toy rectangular feature matrix $A \in \mathbb{R}^{3 \times 2}$:
$$A = \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix}$$

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Shape | Mathematical Role | Machine Learning Meaning | Concrete Demo Value |
| :---: | :---: | :--- | :--- | :---: |
| $A$ | $[3 \times 2]$ | Input rectangular matrix | Weight matrix / Token activations ($m=3, n=2$) | Given above |
| $A^T A$ | $[2 \times 2]$ | Symmetric Gramian matrix | Feature covariance / correlation structure | To be computed |
| $\sigma_1, \sigma_2$ | Scalars | Singular values of $A$ | Energy / variance captured per rank component | To be computed |
| $v_1, v_2$ | $[2 \times 1]$ | Right singular vectors | Orthonormal input principal directions | To be computed |
| $u_1, u_2$ | $[3 \times 1]$ | Left singular vectors | Orthonormal output feature directions | To be computed |
| $A_1$ | $[3 \times 2]$ | Best rank-1 approximation | Truncated matrix ($\sigma_1 u_1 v_1^T$) | To be computed |

---

#### 2. Step 1: Compute $A^T A$ ($2 \times 2$) by Hand

$$A^T A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix} \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix}$$

| Output Cell | Operands Involved | Exact Row $\times$ Col Arithmetic | Result |
| :---: | :--- | :--- | :---: |
| **$(A^T A)_{11}$** | Row 1: $[3, 2, 2]$, Col 1: $[3, 2, 2]^T$ | $3(3) + 2(2) + 2(2) = 9 + 4 + 4$ | **$17$** |
| **$(A^T A)_{12}$** | Row 1: $[3, 2, 2]$, Col 2: $[2, 3, -2]^T$| $3(2) + 2(3) + 2(-2) = 6 + 6 - 4$ | **$8$** |
| **$(A^T A)_{21}$** | Row 2: $[2, 3, -2]$, Col 1: $[3, 2, 2]^T$| $2(3) + 3(2) + (-2)(2) = 6 + 6 - 4$ | **$8$** |
| **$(A^T A)_{22}$** | Row 2: $[2, 3, -2]$, Col 2: $[2, 3, -2]^T$| $2(2) + 3(3) + (-2)(-2) = 4 + 9 + 4$ | **$17$** |

$$A^T A = \begin{bmatrix} 17 & 8 \\ 8 & 17 \end{bmatrix}$$

---

#### 3. Step 2: Solve for Singular Values $\sigma_i = \sqrt{\lambda_i(A^T A)}$

$$\det(A^T A - \lambda I) = (17 - \lambda)^2 - 8^2 = (17 - \lambda)^2 - 64 = 0$$
$$(17 - \lambda)^2 = 64 \implies 17 - \lambda = \pm 8$$
$$\lambda_1 = 17 + 8 = \mathbf{25} \implies \sigma_1 = \sqrt{25} = \mathbf{5}$$
$$\lambda_2 = 17 - 8 = \mathbf{9} \implies \sigma_2 = \sqrt{9} = \mathbf{3}$$

---

#### 4. Step 3: Find Right Singular Vectors $v_1, v_2$ (Eigenvectors of $A^T A$)

- **For $\lambda_1 = 25$:**
  $$(A^T A - 25I) v_1 = \begin{bmatrix} 17 - 25 & 8 \\ 8 & 17 - 25 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} -8 & 8 \\ 8 & -8 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies x = y$$
  Normalizing:
  $$v_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$$

- **For $\lambda_2 = 9$:**
  $$(A^T A - 9I) v_2 = \begin{bmatrix} 17 - 9 & 8 \\ 8 & 17 - 9 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 & 8 \\ 8 & 8 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies x = -y$$
  Normalizing:
  $$v_2 = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix}$$

$$V = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix}$$

---

#### 5. Step 4: Find Left Singular Vectors via $u_i = \frac{1}{\sigma_i} A v_i$

- **Left Singular Vector $u_1$ ($\sigma_1 = 5$):**
  $$A v_1 = \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix} = \begin{bmatrix} 5/\sqrt{2} \\ 5/\sqrt{2} \\ 0 \end{bmatrix}$$
  $$u_1 = \frac{1}{5} (A v_1) = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{bmatrix}$$

- **Left Singular Vector $u_2$ ($\sigma_2 = 3$):**
  $$A v_2 = \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix} = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \\ 4/\sqrt{2} \end{bmatrix}$$
  $$u_2 = \frac{1}{3} (A v_2) = \begin{bmatrix} 1/(3\sqrt{2}) \\ -1/(3\sqrt{2}) \\ 4/(3\sqrt{2}) \end{bmatrix}$$

*Orthonormality Check:*
- $\|u_1\|_2^2 = (1/\sqrt{2})^2 + (1/\sqrt{2})^2 + 0 = 1/2 + 1/2 = 1 \quad \checkmark$
- $\|u_2\|_2^2 = \frac{1 + 1 + 16}{18} = \frac{18}{18} = 1 \quad \checkmark$
- $u_1^T u_2 = \frac{1}{6} - \frac{1}{6} + 0 = 0 \quad \checkmark$

---

#### 6. Step 5: SVD Outer Product Expansion & Exact Reconstruction

$$A = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T$$

- **Rank-1 Sheet 1 ($M_1 = 5 u_1 v_1^T$):**
  $$M_1 = 5 \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \end{bmatrix} = 5 \begin{bmatrix} 1/2 & 1/2 \\ 1/2 & 1/2 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 2.5 & 2.5 \\ 2.5 & 2.5 \\ 0 & 0 \end{bmatrix}$$

- **Rank-1 Sheet 2 ($M_2 = 3 u_2 v_2^T$):**
  $$M_2 = 3 \begin{bmatrix} 1/(3\sqrt{2}) \\ -1/(3\sqrt{2}) \\ 4/(3\sqrt{2}) \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix} = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \\ 4/\sqrt{2} \end{bmatrix} \begin{bmatrix} 1/(2) & -1/(2) \end{bmatrix} = \begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \\ 2.0 & -2.0 \end{bmatrix}$$

- **Summing the Sheets ($M_1 + M_2$):**
  $$M_1 + M_2 = \begin{bmatrix} 2.5 + 0.5 & 2.5 - 0.5 \\ 2.5 - 0.5 & 2.5 + 0.5 \\ 0 + 2.0 & 0 - 2.0 \end{bmatrix} = \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix} \equiv A \quad \checkmark$$

---

#### 7. Step 6: The Eckart-Young Rank-1 Approximation
If we compress matrix $A$ to rank $k=1$, we keep only $M_1$:
$$A_1 = \begin{bmatrix} 2.5 & 2.5 \\ 2.5 & 2.5 \\ 0 & 0 \end{bmatrix}$$
- **Approximation Error Matrix:**
  $$E = A - A_1 = M_2 = \begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \\ 2.0 & -2.0 \end{bmatrix}$$
- **Frobenius Error:**
  $$\|A - A_1\|_F = \sqrt{0.5^2 + (-0.5)^2 + (-0.5)^2 + 0.5^2 + 2^2 + (-2)^2} = \sqrt{0.25 \times 4 + 4 + 4} = \sqrt{1 + 8} = \sqrt{9} = \mathbf{3}$$
- Notice: $\|A - A_1\|_F = \sigma_2 = 3$! The Eckart-Young theorem is verified by hand!

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Computing the Moore-Penrose Pseudoinverse by Hand
**Problem:** Using SVD from Part 5, find $A^+$ for $A = \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix}$.

1. Recall formula: $A^+ = \sum_{i=1}^r \frac{1}{\sigma_i} v_i u_i^T$.
   $$A^+ = \frac{1}{5} v_1 u_1^T + \frac{1}{3} v_2 u_2^T$$
2. **First Component:**
   $$\frac{1}{5} \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} & 0 \end{bmatrix} = \frac{1}{5} \begin{bmatrix} 1/2 & 1/2 & 0 \\ 1/2 & 1/2 & 0 \end{bmatrix} = \begin{bmatrix} 0.1 & 0.1 & 0 \\ 0.1 & 0.1 & 0 \end{bmatrix}$$
3. **Second Component:**
   $$\frac{1}{3} \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix} \begin{bmatrix} 1/(3\sqrt{2}) & -1/(3\sqrt{2}) & 4/(3\sqrt{2}) \end{bmatrix} = \frac{1}{18} \begin{bmatrix} 1 & -1 & 4 \\ -1 & 1 & -4 \end{bmatrix}$$
4. **Summing ($A^+$):**
   $$A^+ = \begin{bmatrix} 1/10 + 1/18 & 1/10 - 1/18 & 4/18 \\ 1/10 - 1/18 & 1/10 + 1/18 & -4/18 \end{bmatrix} = \begin{bmatrix} 7/45 & 2/45 & 2/9 \\ 2/45 & 7/45 & -2/9 \end{bmatrix}$$
5. **Verify Penrose Condition ($A^+ A = I_2$):**
   Since $A$ has full column rank, $A^+ A$ must equal the identity matrix $I_2$:
   $$A^+ A = \begin{bmatrix} 7/45 & 2/45 & 2/9 \\ 2/45 & 7/45 & -2/9 \end{bmatrix} \begin{bmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{bmatrix} = \begin{bmatrix} 21/45 + 4/45 + 20/45 & 14/45 + 6/45 - 20/45 \\ 6/45 + 14/45 - 20/45 & 4/45 + 21/45 + 20/45 \end{bmatrix} = \begin{bmatrix} 45/45 & 0 \\ 0 & 45/45 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \quad \checkmark$$

---

### Scenario B: Boundary Case — Rank-1 Matrix Decomposition
**Problem:** Find the SVD of $M = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}$.

1. Notice that Row 2 is $0.5 \times \text{Row 1}$. The rank is strictly $r = 1$.
2. Write as outer product: $M = \begin{bmatrix} 2 \\ 1 \end{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix}$.
3. Let $u = \begin{bmatrix} 2 \\ 1 \end{bmatrix} \implies \|u\| = \sqrt{5}, \quad \hat{u} = \begin{bmatrix} 2/\sqrt{5} \\ 1/\sqrt{5} \end{bmatrix}$.
4. Let $v = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \implies \|v\| = \sqrt{5}, \quad \hat{v} = \begin{bmatrix} 1/\sqrt{5} \\ 2/\sqrt{5} \end{bmatrix}$.
5. Therefore:
   $$M = \underbrace{(\sqrt{5} \times \sqrt{5})}_{\sigma_1 = 5} \hat{u} \hat{v}^T = 5 \hat{u} \hat{v}^T$$
   The singular values are $\sigma_1 = 5, \sigma_2 = 0$.

---

### Scenario C: Complete SVD of a Rectangular Underdetermined Matrix ($2 \times 3$)

**Problem Formulation:**
Find the complete SVD $A = U \Sigma V^T$ for the rectangular matrix:
$$A = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

#### Step 1: Compute $A A^T$ ($2 \times 2$) to find Left Singular Vectors $U$
$$A A^T = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$
Characteristic equation of $A A^T$:
$$\det(A A^T - \lambda I) = (2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 3)(\lambda - 1) = 0$$
$$\lambda_1 = 3 \implies \sigma_1 = \sqrt{3}$$
$$\lambda_2 = 1 \implies \sigma_2 = 1$$
Singular value matrix $\Sigma \in \mathbb{R}^{2 \times 3}$:
$$\Sigma = \begin{bmatrix} \sqrt{3} & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$

#### Step 2: Compute Orthonormal Eigenvectors of $A A^T$ ($U$)
- **For $\lambda_1 = 3$:**
  $$(A A^T - 3I) u_1 = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \mathbf{0} \implies x = y \implies u_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$$
- **For $\lambda_2 = 1$:**
  $$(A A^T - 1I) u_2 = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \mathbf{0} \implies x = -y \implies u_2 = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix}$$

$$U = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix}$$

#### Step 3: Compute Right Singular Vectors via $v_i = \frac{1}{\sigma_i} A^T u_i$
- **For $\sigma_1 = \sqrt{3}$:**
  $$v_1 = \frac{1}{\sqrt{3}} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix} = \frac{1}{\sqrt{3}} \begin{bmatrix} 1/\sqrt{2} \\ 2/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix} = \begin{bmatrix} 1/\sqrt{6} \\ 2/\sqrt{6} \\ 1/\sqrt{6} \end{bmatrix}$$
- **For $\sigma_2 = 1$:**
  $$v_2 = \frac{1}{1} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix} = \begin{bmatrix} 1/\sqrt{2} \\ 0 \\ -1/\sqrt{2} \end{bmatrix}$$
- **For nullspace direction $v_3 \in N(A)$ ($\sigma_3 = 0$):**
  Solve $A v_3 = \mathbf{0}$:
  $$\begin{cases} x_1 + x_2 = 0 \implies x_1 = -x_2 \\ x_2 + x_3 = 0 \implies x_3 = -x_2 \end{cases} \implies v_3 = \begin{bmatrix} 1/\sqrt{3} \\ -1/\sqrt{3} \\ 1/\sqrt{3} \end{bmatrix}$$

$$V = \begin{bmatrix} 1/\sqrt{6} & 1/\sqrt{2} & 1/\sqrt{3} \\ 2/\sqrt{6} & 0 & -1/\sqrt{3} \\ 1/\sqrt{6} & -1/\sqrt{2} & 1/\sqrt{3} \end{bmatrix}$$

#### Step 4: Verification of Reconstruction ($A = U \Sigma V^T$)
$$\sigma_1 u_1 v_1^T = \sqrt{3} \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix} \begin{bmatrix} 1/\sqrt{6} & 2/\sqrt{6} & 1/\sqrt{6} \end{bmatrix} = \frac{\sqrt{3}}{\sqrt{12}} \begin{bmatrix} 1 & 2 & 1 \\ 1 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 0.5 & 1.0 & 0.5 \\ 0.5 & 1.0 & 0.5 \end{bmatrix}$$
$$\sigma_2 u_2 v_2^T = 1 \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} & 0 & -1/\sqrt{2} \end{bmatrix} = \frac{1}{2} \begin{bmatrix} 1 & 0 & -1 \\ -1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.0 & -0.5 \\ -0.5 & 0.0 & 0.5 \end{bmatrix}$$
Summing:
$$A = \begin{bmatrix} 0.5+0.5 & 1.0+0.0 & 0.5-0.5 \\ 0.5-0.5 & 1.0+0.0 & 0.5+0.5 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix} \quad \checkmark$$

---

### Scenario D: Minimum-Norm Solution via Pseudoinverse ($x^+ = A^+ b$)

**Problem Formulation:**
Consider the underdetermined system $A x = b$ where $A = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}$ and $b = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$.
Because there are 3 variables and 2 equations, there are infinitely many valid solutions.
1. Compute the Moore-Penrose Pseudoinverse $A^+$.
2. Compute the minimum-norm solution $x^+ = A^+ b$.
3. Verify that $A x^+ = b$.
4. Prove that $x^+$ is strictly orthogonal to the nullspace $N(A)$, confirming it has minimal Euclidean length among all possible solutions.

#### Step 1: Compute $A^+$
Since $A$ has full row rank ($m = 2$), $A^+ = A^T (A A^T)^{-1}$:
$$(A A^T)^{-1} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}^{-1} = \frac{1}{3} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$$
$$A^+ = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \left( \frac{1}{3} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} \right) = \frac{1}{3} \begin{bmatrix} 2 & -1 \\ 1 & 1 \\ -1 & 2 \end{bmatrix}$$

#### Step 2: Compute $x^+ = A^+ b$
$$x^+ = \frac{1}{3} \begin{bmatrix} 2 & -1 \\ 1 & 1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \frac{1}{3} \begin{bmatrix} 2(2) - 1(2) \\ 1(2) + 1(2) \\ -1(2) + 2(2) \end{bmatrix} = \frac{1}{3} \begin{bmatrix} 2 \\ 4 \\ 2 \end{bmatrix} = \begin{bmatrix} 2/3 \\ 4/3 \\ 2/3 \end{bmatrix}$$

#### Step 3: Verification of Consistency ($A x^+ = b$)
$$A x^+ = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 2/3 \\ 4/3 \\ 2/3 \end{bmatrix} = \begin{bmatrix} 2/3 + 4/3 + 0 \\ 0 + 4/3 + 2/3 \end{bmatrix} = \begin{bmatrix} 6/3 \\ 6/3 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix} = b \quad \checkmark$$

#### Step 4: Verification of Minimum Norm Property
From Scenario C, the nullspace $N(A)$ is spanned by $v_3 = [1, -1, 1]^T$.
Compute the inner product $\langle x^+, v_3 \rangle$:
$$(x^+)^T v_3 = \frac{2}{3}(1) + \frac{4}{3}(-1) + \frac{2}{3}(1) = \frac{2 - 4 + 2}{3} = 0$$
Since $x^+ \perp N(A)$, any other solution has the form $x = x^+ + c v_3$ ($c \neq 0$).
By the Pythagorean theorem:
$$\|x\|_2^2 = \|x^+ + c v_3\|_2^2 = \|x^+\|_2^2 + c^2 \|v_3\|_2^2 = \left( \frac{4 + 16 + 4}{9} \right) + 3c^2 = \frac{8}{3} + 3c^2 > \frac{8}{3}$$
Thus $\|x\|_2 > \|x^+\|_2$ for any $c \neq 0$. The pseudoinverse solution $x^+$ strictly minimizes parameter norm!

---

## Part 7: Deep Learning Connection & Application

### 1. Low-Rank Adaptation (LoRA) & SVD Weight Pruning
A pre-trained weight matrix in an LLM (e.g., LLaMA-3 70B attention projection) has size $W_0 \in \mathbb{R}^{4096 \times 4096}$ (16.7 million parameters).
- **Weight Compression**: Computing truncated SVD $W_0 \approx U_k \Sigma_k V_k^T$ with rank $k=64$ replaces the single large layer with two small matrices:
  $$B = U_k \sqrt{\Sigma_k} \in \mathbb{R}^{4096 \times 64}, \quad A = \sqrt{\Sigma_k} V_k^T \in \mathbb{R}^{64 \times 4096}$$
- **Parameter Reduction**:
  $$\text{Original parameters} = 4096 \times 4096 = 16,777,216$$
  $$\text{Decomposed parameters} = 2 \times (4096 \times 64) = 524,288 \quad \text{\bf (96.9\% reduction!)}$$

### 2. Condition Number & Gradient Flow Stability
The **Condition Number** of a weight matrix $W$ is:
$$\kappa(W) = \frac{\sigma_{\max}(W)}{\sigma_{\min}(W)}$$
- If $\kappa(W) \approx 1$, the matrix is well-conditioned (near orthogonal), and gradients flow uniformly across all directions.
- If $\kappa(W) \gg 10^4$, the matrix is **ill-conditioned**. Gradients explode along $u_1$ and vanish along $u_{\min}$, causing optimizers like SGD to oscillate violently.

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/07_singular_value_decomposition.py`](./code/07_singular_value_decomposition.py) implements:
1. **Scratch SVD Engine**: Computes $U, \Sigma, V^T$ via eigendecomposition of $A^T A$ and $A A^T$, asserting $A = U \Sigma V^T$.
2. **Eckart-Young Rank-$k$ Approximator**: Computes truncated SVD, verifying that Frobenius error matches $\sqrt{\sum_{j=k+1} \sigma_j^2}$.
3. **Moore-Penrose Pseudoinverse Engine**: Computes $A^+$ and verifies all 4 Penrose conditions.
4. **PyTorch Linear Layer SVD Factorization**: Compresses a large $512 \times 512$ linear layer to rank $r=32$, measuring parameter savings ($93.75\%$) and relative reconstruction fidelity.

---

## Chapter 1.7 Summary & Review Checklist

- [x] Why does SVD apply to rectangular matrices while eigendecomposition requires square matrices?
- [x] How are the right singular vectors $V$ related to $A^T A$?
- [x] What does the Eckart-Young-Mirsky Theorem prove about truncated SVD?
- [x] What are the 4 defining Penrose conditions for the pseudoinverse $A^+$?
- [x] How is SVD used to compress a 16M parameter transformer layer into a 500K parameter bottleneck with $<3\%$ loss?
