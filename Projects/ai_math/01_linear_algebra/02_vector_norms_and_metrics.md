# Chapter 1.2: Vector Norms, Metrics & Inner Products

---

## Part 1: Intuition & 101 Motivation

In Deep Learning, vectors represent features, weights, token embeddings, and gradient updates. But representations are meaningless unless we can answer three fundamental geometric questions:
1. **Magnitude**: How "large" is this vector or parameter update? (e.g., Is the gradient exploding or vanishing?)
2. **Distance**: How far apart are two representations? (e.g., Did the autoencoder faithfully reconstruct the input image?)
3. **Direction & Similarity**: Do two vectors point in the same direction? (e.g., In modern Retrieval-Augmented Generation (RAG) or CLIP, is this user query semantically aligned with a stored document?)

A raw vector space alone cannot answer these questions because it lacks a concept of length or angle. To measure these, we equip our vector space with an **Inner Product** and a **Norm**.

Choosing the right norm is not merely aesthetic—it fundamentally dictates how a neural network learns:
- **$L_2$ Norm** penalizes large outlier errors heavily and shrinks weights smoothly (Weight Decay).
- **$L_1$ Norm** penalizes all errors linearly and forces weights to become exactly zero, driving **sparsity** (vital for Sparse Autoencoders in mechanistic interpretability and model compression).
- **$L_\infty$ Norm** caps the worst-case individual component (vital for adversarial robustness attacks like FGSM).

---

## Part 2: Rigorous Mathematical Formulation

### 1. Inner Product Spaces
Let $V$ be a vector space over the field $\mathbb{R}$. An **inner product** is a function $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfying four axioms for all $u, v, w \in V$ and $\alpha, \beta \in \mathbb{R}$:

| Axiom | Name | Mathematical Statement |
| :--- | :--- | :--- |
| **IP1** | **Linearity in the First Argument** | $\langle \alpha u + \beta v, w \rangle = \alpha \langle u, w \rangle + \beta \langle v, w \rangle$ |
| **IP2** | **Symmetry** | $\langle u, v \rangle = \langle v, u \rangle$ |
| **IP3** | **Positive-Definiteness** | $\langle v, v \rangle \ge 0, \quad \text{and} \quad \langle v, v \rangle = 0 \iff v = \mathbf{0}$ |

In standard Euclidean space $\mathbb{R}^n$, the canonical inner product is the **dot product**:
$$\langle u, v \rangle = u^T v = \sum_{i=1}^n u_i v_i$$

---

### 2. Definition of a Norm
A **norm** on a vector space $V$ is a function $\|\cdot\| : V \to \mathbb{R}_{\ge 0}$ that assigns a non-negative length to each vector, satisfying three strict axioms for all $u, v \in V$ and scalars $\alpha \in \mathbb{R}$:

| Axiom | Name | Mathematical Statement | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **N1** | **Positive Definiteness** | $\|v\| \ge 0$, and $\|v\| = 0 \iff v = \mathbf{0}$ | Only the null vector has zero length. |
| **N2** | **Absolute Homogeneity** | $\|\alpha v\| = |\alpha| \|v\|$ | Scaling a vector by $\alpha$ scales its length by $|\alpha|$. |
| **N3** | **Subadditivity (Triangle Inequality)**| $\|u + v\| \le \|u\| + \|v\|$ | The shortest path between two points is a straight line. |

Every inner product naturally induces a valid norm:
$$\|v\| = \sqrt{\langle v, v \rangle}$$

---

### 3. The Family of $L_p$ Vector Norms (Minkowski Norms)
For any real number $p \ge 1$, the **$L_p$ norm** of a vector $x \in \mathbb{R}^n$ is defined as:
$$\|x\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}$$

```
                p = 0.5 (Non-convex)       p = 1 (L1 Diamond)        p = 2 (L2 Circle)       p = ∞ (L∞ Square)
                     y                          y                         y                        y
                     ^                          ^                         ^                        ^
                     |                          |                         |                        |
                 \   |   /                      |   /\                    |   .--.                 +-------+
                  \  |  /                       |  /  \                   |  /    \                |       |
               ----+---+-----> x             ---+--+----+-> x          ---+-(------+-> x        ---+---+---+-> x
                  /  |  \                       |  \  /                   |  \    /                |       |
                 /   |   \                      |   \/                    |   '--'                 +-------+
                     |                          |                         |                        |
```

#### Special Cases in Deep Learning:

1. **$L_1$ Norm (Manhattan / Taxicab Norm)** ($p = 1$):
   $$\|x\|_1 = \sum_{i=1}^n |x_i|$$
   Measures distance traversed along grid-aligned axes. Induces **sparse representations**.

2. **$L_2$ Norm (Euclidean Norm)** ($p = 2$):
   $$\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^T x}$$
   Standard straight-line distance. Invariant to coordinate rotations. Basis of **Weight Decay**.

3. **$L_\infty$ Norm (Chebyshev / Maximum Norm)** ($p \to \infty$):
   $$\|x\|_\infty = \lim_{p \to \infty} \left( \sum_{i=1}^n |x_i|^p \right)^{1/p} = \max_{1 \le i \le n} |x_i|$$
   Measures the maximum individual perturbation. Foundational to **Adversarial Robustness (Fast Gradient Sign Method / FGSM)**.

4. **The "$L_0$ Pseudo-Norm"** ($p = 0$):
   $$\|x\|_0 = \sum_{i=1}^n \mathbb{I}(x_i \neq 0)$$
   Counts the number of non-zero entries.
   > [!WARNING]
   > $L_0$ is **NOT a true mathematical norm** because it violates Axiom N2 (Absolute Homogeneity): $\|\alpha x\|_0 = \|x\|_0 \neq |\alpha| \|x\|_0$. Minimizing $L_0$ directly is NP-hard; $L_1$ is the optimal convex relaxation of $L_0$.

5. **Why $p \ge 1$ is Mandatory for a Valid Norm:**
   For $0 < p < 1$, the function $\|x\|_p = (\sum |x_i|^p)^{1/p}$ violates the **Triangle Inequality (Axiom N3)**. Its unit ball is non-convex (inward-dented), making optimization non-convex.

---

### 4. Matrix Norms

#### A. The Frobenius Norm (Element-wise Matrix Norm)
Treats an $m \times n$ matrix $A$ as a flattened vector of size $mn$:
$$\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2} = \sqrt{\text{Tr}(A^T A)} = \sqrt{\sum_{k=1}^{\min(m,n)} \sigma_k^2}$$
where $\sigma_k$ are the singular values of $A$. In deep learning, Frobenius norm regularization is the direct matrix equivalent of $L_2$ weight decay applied across all weight tensors.

#### B. Induced (Operator) Norms
Measures the maximum factor by which matrix $A$ can stretch any vector:
$$\|A\|_p = \sup_{x \neq \mathbf{0}} \frac{\|A x\|_p}{\|x\|_p} = \max_{\|x\|_p = 1} \|A x\|_p$$
- **Spectral Norm ($\|A\|_2$)**: When $p = 2$, the induced norm is equal to the **largest singular value** of $A$:
  $$\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$$
  Spectral normalization ensures $1$-Lipschitz continuity, essential for stabilizing Wasserstein GANs (WGAN) and Transformer residual branches.

---

### 5. Angles & The Cauchy-Schwarz Inequality

#### The Cauchy-Schwarz Inequality
For any two vectors $u, v$ in an inner product space:
$$|\langle u, v \rangle| \le \|u\|_2 \|v\|_2$$
$$\left| \sum_{i=1}^n u_i v_i \right| \le \left( \sum_{i=1}^n u_i^2 \right)^{1/2} \left( \sum_{i=1}^n v_i^2 \right)^{1/2}$$
*Equality Condition:* Equality holds if and only if $u$ and $v$ are **linearly dependent** ($u = c v$ for scalar $c$).

#### Proof (Gilbert Strang Style via Quadratic Optimization):
Consider the vector $w = u + t v$ for an arbitrary scalar parameter $t \in \mathbb{R}$.
By Positive-Definiteness (Axiom IP3), the inner product of $w$ with itself must be non-negative:
$$\langle u + t v, u + t v \rangle \ge 0, \quad \forall t \in \mathbb{R}$$
Expanding by linearity:
$$\langle u, u \rangle + 2t \langle u, v \rangle + t^2 \langle v, v \rangle \ge 0$$
$$\|v\|_2^2 \, t^2 + 2\langle u, v \rangle \, t + \|u\|_2^2 \ge 0$$
This is a quadratic polynomial in $t$: $a t^2 + b t + c \ge 0$, where:
- $a = \|v\|_2^2$
- $b = 2\langle u, v \rangle$
- $c = \|u\|_2^2$

For a quadratic polynomial to remain $\ge 0$ for all real $t$, its parabola cannot dip below the axis, meaning it can have at most one real root. Therefore, its **discriminant** must be non-positive ($\Delta \le 0$):
$$\Delta = b^2 - 4ac \le 0$$
$$(2\langle u, v \rangle)^2 - 4 \|v\|_2^2 \|u\|_2^2 \le 0$$
$$4 \langle u, v \rangle^2 \le 4 \|u\|_2^2 \|v\|_2^2$$
Taking the square root of both sides:
$$|\langle u, v \rangle| \le \|u\|_2 \|v\|_2 \quad \blacksquare$$

#### Definition of Angle & Cosine Similarity
Because $|\langle u, v \rangle| \le \|u\|_2 \|v\|_2$, the ratio always lies in $[-1, 1]$:
$$-1 \le \frac{u^T v}{\|u\|_2 \|v\|_2} \le 1$$
We define the **angle $\theta$** between vectors $u$ and $v$ as:
$$\cos \theta = \frac{u^T v}{\|u\|_2 \|v\|_2} = \text{CosineSimilarity}(u, v)$$

```
  Vectors Collinear (θ = 0°)       Vectors Orthogonal (θ = 90°)      Vectors Opposite (θ = 180°)
  cos θ = +1.0                     cos θ = 0.0                       cos θ = -1.0
          u                                  v                                u
  ===================>                       ^                        <===============+===============>
          v                                  |                                        v
  ===================>                       +--------> u
```

---

## Part 3: Geometric & Algebraic Interpretation

### 1. Why $L_1$ Induces Sparsity: The Tangency Geometry
Consider solving a regularized regression problem (e.g., Lasso vs. Ridge):
$$\min_w \mathcal{L}(w) \quad \text{subject to} \quad \|w\|_p \le C$$

```
           L2 Regularization (Ridge)                       L1 Regularization (Lasso)
                     w2                                              w2
                     ^                                               ^
                     |    Contour lines of L(w)                      |    Contour lines of L(w)
                  .-'|'-.       (.. )                             /\ |   (.. )
                .'   |   '.   (       )                          /  \| (       )
               /     |     \ (  * min  )                        /    * ( * min )
              |------+------|---------> w1                     +-----+---------> w1
               \     |     /                                    \    /
                '.   |   .'                                      \  /
                  '-.|.-'                                         \/
              Tangency occurs at smooth                   Tangency occurs at the SHARP CORNER!
              point: w1 ≠ 0, w2 ≠ 0                       Here on the axis: w1 = 0 (EXACT SPARSITY!)
```

- **$L_2$ Unit Ball is Smooth**: The gradient of $\|w\|_2$ points radially everywhere. The elliptical loss contour touches the sphere at an arbitrary curved point where both $w_1$ and $w_2$ are non-zero.
- **$L_1$ Unit Ball Has Sharp Corners**: The corners of the diamond lie precisely on the coordinate axes (where one or more variables are zero). Because the corners stick out furthest in coordinate directions, expanding loss contours are mathematically far more likely to hit a sharp vertex, forcing the corresponding weights to **exact zero**.

---

## Part 4: Real-World Analogy

### 1. Navigating Manhattan vs. Flying "As the Crow Flies"
- **$L_1$ (Manhattan Distance):** You are a taxi driver in New York City. You cannot drive straight through skyscrapers; you must travel along rectangular street grids ($|\Delta x| + |\Delta y|$).
- **$L_2$ (Euclidean Distance):** You are a drone flying directly above the city straight to the target ($\sqrt{\Delta x^2 + \Delta y^2}$).
- **$L_\infty$ (Chebyshev Distance):** You are the King in a game of chess. Moving diagonally 1 square takes 1 turn, just like moving horizontally 1 square. The number of moves needed is $\max(|\Delta x|, |\Delta y|)$.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### Cosine Similarity & Attention Query-Key Matching by Hand

Let us trace by hand the exact dot products, vector norms, and cosine similarities between a Transformer **Query** vector and two candidate **Key** vectors.

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Mathematical Domain | Concrete Shape in Demo | Mathematical Meaning | Deep Learning Semantic Role | Concrete Demo Value |
| :---: | :---: | :---: | :--- | :--- | :---: |
| $q$ | $\mathbb{R}^{d_k}$ | Vector $[3 \times 1]$ | Query representation | Search intent token embedding | $\begin{bmatrix} 2 \\ 1 \\ -2 \end{bmatrix}$ |
| $k_1$ | $\mathbb{R}^{d_k}$ | Vector $[3 \times 1]$ | Key vector 1 | Candidate memory token A (semantically aligned) | $\begin{bmatrix} 4 \\ 2 \\ -4 \end{bmatrix}$ |
| $k_2$ | $\mathbb{R}^{d_k}$ | Vector $[3 \times 1]$ | Key vector 2 | Candidate memory token B (unrelated/orthogonal) | $\begin{bmatrix} 1 \\ -2 \\ 0 \end{bmatrix}$ |
| $\langle q, k \rangle$ | $\mathbb{R}$ | Scalar | Raw unnormalized dot product | Unscaled attention logit | Computed below |
| $\|q\|_2$ | $\mathbb{R}_{\ge 0}$ | Scalar | Euclidean norm of query | Query magnitude | $\sqrt{2^2 + 1^2 + (-2)^2} = 3$ |
| $\|k_1\|_2$ | $\mathbb{R}_{\ge 0}$ | Scalar | Euclidean norm of Key 1 | Key 1 magnitude | $\sqrt{4^2 + 2^2 + (-4)^2} = 6$ |
| $\|k_2\|_2$ | $\mathbb{R}_{\ge 0}$ | Scalar | Euclidean norm of Key 2 | Key 2 magnitude | $\sqrt{1^2 + (-2)^2 + 0^2} = \sqrt{5}$ |
| $S(q, k)$ | $[-1, 1]$ | Scalar | Cosine similarity | Directional alignment score | Normalized match score |

---

#### 2. Visual Vector Grids

```
         QUERY q                KEY 1 (k_1)              KEY 2 (k_2)
       ┌───────────┐            ┌───────────┐            ┌───────────┐
 Row 1 │ q_1 =  2  │      Row 1 │ k_11 =  4 │      Row 1 │ k_21 =  1 │
       ├───────────┤            ├───────────┤            ├───────────┤
 Row 2 │ q_2 =  1  │      Row 2 │ k_12 =  2 │      Row 2 │ k_22 = -2 │
       ├───────────┤            ├───────────┤            ├───────────┤
 Row 3 │ q_3 = -2  │      Row 3 │ k_13 = -4 │      Row 3 │ k_23 =  0 │
       └───────────┘            └───────────┘            └───────────┘
```

---

#### 3. Step-by-Step Dot Product Calculation ($q^T k$)

| Vector Pair | Exact Element-wise Multiplication | Intermediate Products | Raw Dot Product |
| :---: | :--- | :--- | :---: |
| **$q^T k_1$** | $(2 \times 4) + (1 \times 2) + (-2 \times -4)$ | $8 + 2 + 8$ | **$18$** |
| **$q^T k_2$** | $(2 \times 1) + (1 \times -2) + (-2 \times 0)$ | $2 - 2 + 0$ | **$0$** |

---

#### 4. Step-by-Step Euclidean Norm Calculation ($\|v\|_2 = \sqrt{\sum v_i^2}$)

| Vector | Coordinate Squares | Sum of Squares | Square Root | $L_2$ Norm Result |
| :---: | :--- | :--- | :---: | :---: |
| **$\|q\|_2$** | $2^2 + 1^2 + (-2)^2 = 4 + 1 + 4$ | $9$ | $\sqrt{9}$ | **$3$** |
| **$\|k_1\|_2$** | $4^2 + 2^2 + (-4)^2 = 16 + 4 + 16$ | $36$ | $\sqrt{36}$ | **$6$** |
| **$\|k_2\|_2$** | $1^2 + (-2)^2 + 0^2 = 1 + 4 + 0$ | $5$ | $\sqrt{5}$ | **$\approx 2.236$** |

---

#### 5. Step-by-Step Cosine Similarity Calculation ($S = \frac{q^T k}{\|q\|_2 \|k\|_2}$)

| Pair | Numerator ($q^T k$) | Denominator ($\|q\|_2 \times \|k\|_2$) | Ratio Arithmetic | Cosine Similarity Score | Geometric Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$(q, k_1)$** | $18$ | $3 \times 6 = 18$ | $\frac{18}{18}$ | **$+1.000$** | **Perfect Collinear Alignment ($\theta = 0^\circ$)** |
| **$(q, k_2)$** | $0$ | $3 \times \sqrt{5} \approx 6.708$ | $\frac{0}{6.708}$ | **$0.000$** | **Strict Orthogonality ($\theta = 90^\circ$)** |

> [!IMPORTANT]
> **Key Takeaway for Attention & Embedding Retrieval:**
> Notice that $k_1 = 2 \cdot q$. The raw dot product was $18$ solely because both vectors had larger magnitudes.
> Cosine similarity **eliminates scale bias**, isolating pure semantic angle ($\cos \theta = 1.0$).
> In standard Attention, scaled dot product $\frac{q^T k}{\sqrt{d_k}}$ preserves magnitude so the model can express confidence, whereas Vector Databases (FAISS, Milvus, Chroma) often rely purely on Cosine Similarity for RAG retrieval.

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Standard Case — Computing All $L_p$ Norms
**Problem:** Given vector $x = \begin{bmatrix} 3 \\ -4 \\ 0 \\ 12 \end{bmatrix} \in \mathbb{R}^4$:
Compute:
1. $\|x\|_0$ ("$L_0$ count")
2. $\|x\|_1$ (Manhattan norm)
3. $\|x\|_2$ (Euclidean norm)
4. $\|x\|_\infty$ (Chebyshev norm)
5. Show that the norm ordering holds: $\|x\|_\infty \le \|x\|_2 \le \|x\|_1$.

#### Step-by-Step Manual Arithmetic:
1. **$L_0$ Pseudo-norm:**
   Count non-zero entries: $x_1 = 3 \neq 0$, $x_2 = -4 \neq 0$, $x_3 = 0$, $x_4 = 12 \neq 0$.
   $$\|x\|_0 = 3$$
2. **$L_1$ Norm:**
   Sum of absolute values:
   $$\|x\|_1 = |3| + |-4| + |0| + |12| = 3 + 4 + 0 + 12 = 19$$
3. **$L_2$ Norm:**
   Square root of sum of squares:
   $$\|x\|_2 = \sqrt{3^2 + (-4)^2 + 0^2 + 12^2} = \sqrt{9 + 16 + 0 + 144} = \sqrt{169} = 13$$
4. **$L_\infty$ Norm:**
   Maximum absolute value:
   $$\|x\|_\infty = \max(|3|, |-4|, |0|, |12|) = \max(3, 4, 0, 12) = 12$$
5. **Verification of Universal Norm Hierarchy:**
   $$12 \le 13 \le 19 \iff \|x\|_\infty \le \|x\|_2 \le \|x\|_1 \quad \checkmark$$

---

### Scenario B: Boundary Case — Cauchy-Schwarz Equality Condition
**Problem:** Let $u = \begin{bmatrix} 2 \\ -1 \end{bmatrix}$. Find all vectors $v \in \mathbb{R}^2$ such that $|u^T v| = \|u\|_2 \|v\|_2$.

#### Step-by-Step Solution:
- We know by Cauchy-Schwarz that $|u^T v| \le \|u\|_2 \|v\|_2$, with equality if and only if $v$ is linearly dependent on $u$:
  $$v = c u = c \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \begin{bmatrix} 2c \\ -c \end{bmatrix}, \quad \forall c \in \mathbb{R}$$
- Let us test $c = -3 \implies v = \begin{bmatrix} -6 \\ 3 \end{bmatrix}$:
  - Left Hand Side (Dot product absolute value):
    $$|u^T v| = |(2 \times -6) + (-1 \times 3)| = |-12 - 3| = |-15| = 15$$
  - Right Hand Side (Product of $L_2$ norms):
    $$\|u\|_2 = \sqrt{2^2 + (-1)^2} = \sqrt{4 + 1} = \sqrt{5}$$
    $$\|v\|_2 = \sqrt{(-6)^2 + 3^2} = \sqrt{36 + 9} = \sqrt{45} = \sqrt{9 \times 5} = 3\sqrt{5}$$
    $$\|u\|_2 \|v\|_2 = \sqrt{5} \cdot 3\sqrt{5} = 3 \times 5 = 15$$
  - Result: $15 = 15$. Equality confirmed!
  - Angle: $\cos \theta = \frac{u^T v}{\|u\|_2 \|v\|_2} = \frac{-15}{15} = -1.0 \implies \theta = 180^\circ$ (opposite direction).

---

### Scenario C: Pathological Edge Case — Failure of Triangle Inequality for $p < 1$
**Problem:** Prove by explicit counterexample that $f(x) = (|x_1|^{0.5} + |x_2|^{0.5})^2$ fails the Triangle Inequality in $\mathbb{R}^2$.

#### Step-by-Step Counterexample:
Let $u = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $v = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.
1. Compute "norm" of $u$:
   $$\|u\|_{0.5} = (|1|^{0.5} + |0|^{0.5})^2 = (1 + 0)^2 = 1$$
2. Compute "norm" of $v$:
   $$\|v\|_{0.5} = (|0|^{0.5} + |1|^{0.5})^2 = (0 + 1)^2 = 1$$
3. Form sum $u + v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$:
   $$\|u + v\|_{0.5} = (|1|^{0.5} + |1|^{0.5})^2 = (1 + 1)^2 = 2^2 = 4$$
4. Check Triangle Inequality ($\|u + v\| \le \|u\| + \|v\|$):
   $$\|u + v\|_{0.5} = 4$$
   $$\|u\|_{0.5} + \|v\|_{0.5} = 1 + 1 = 2$$
   $$4 \not\le 2 \quad \text{\bf (VIOLATION!)}$$
- **Conclusion:** For $p = 0.5$, taking a direct path between $u$ and $v$ costs length $4$, whereas going through the origin costs $1 + 1 = 2$. It violates the triangle inequality; hence $L_{0.5}$ cannot be a norm.

---

## Part 7: Deep Learning Connection & Application

### 1. Weight Decay in Neural Networks ($L_2$ Regularization)
When training deep networks, minimizing training loss alone causes weights to grow excessively large, leading to overfitting and brittle decision boundaries.
- **Objective function with $L_2$ penalty**:
  $$\mathcal{J}_{\text{reg}}(W) = \mathcal{L}(W) + \frac{\lambda}{2} \|W\|_F^2$$
- **Gradient step**:
  $$\nabla_W \mathcal{J}_{\text{reg}} = \nabla_W \mathcal{L} + \lambda W$$
  $$W_{t+1} = W_t - \eta \left( \nabla \mathcal{L}(W_t) + \lambda W_t \right) = (1 - \eta \lambda) W_t - \eta \nabla \mathcal{L}(W_t)$$
  *Insight:* In every optimization step, the weight is multiplied by a shrinkage factor $(1 - \eta \lambda) < 1$. This is why $L_2$ regularization is called **Weight Decay**.

### 2. Gradient Clipping by Norm (Preventing Exploding Gradients)
In Deep RNNs and Transformers, gradients traversing many layers can explode ($\|g\|_2 \to \infty$).
- **Norm Clipping Rule**:
  $$\tilde{g} = g \cdot \min\left(1, \frac{C}{\|g\|_2}\right)$$
  where $C$ is the maximum permitted gradient norm threshold (e.g., $C = 1.0$).
  - If $\|g\|_2 \le C$, $\tilde{g} = g$ (untouched).
  - If $\|g\|_2 > C$, $\tilde{g} = C \frac{g}{\|g\|_2}$ (direction is **strictly preserved**, but magnitude is clipped to exactly $C$).

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/02_vector_norms_and_metrics.py`](./code/02_vector_norms_and_metrics.py) implements:
1. **Universal $L_p$ and Matrix Norm Engine**: Vectorized NumPy functions computing $L_0, L_1, L_2, L_\infty$, and Frobenius norms.
2. **Cosine Similarity & Softmax Matcher**: PyTorch module matching queries and keys with temperature scaling.
3. **Triangle Inequality Stress Tester**: Monte Carlo validation demonstrating that $p \ge 1$ always satisfies triangle inequality while $p < 1$ consistently fails.
4. **Gradient Clipping by Norm**: Vectorized PyTorch simulation comparing unclipped exploding updates with norm-clipped updates.

---

## Chapter 1.2 Summary & Review Checklist

- [x] What are the 3 axioms that any function must satisfy to be called a "norm"?
- [x] Why does $L_1$ regularization produce exact zero weights (sparsity) while $L_2$ does not? (Explain via tangency geometry).
- [x] Under what condition does the Cauchy-Schwarz inequality $|\langle u, v \rangle| \le \|u\| \|v\|$ hold with strict equality?
- [x] Why is $L_{0.5}$ not a valid norm?
- [x] How does gradient clipping by norm modify the gradient vector without altering its direction?
