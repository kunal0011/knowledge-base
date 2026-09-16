# Chapter 1.3: Matrix Algebra, Column Spaces & Systems of Linear Equations

---

## Part 1: Intuition & 101 Motivation

In Deep Learning, matrices play a dual role:
1. **As Data Tensors**: A batch of $B$ samples with feature dimension $D$ is stored as a matrix $X \in \mathbb{R}^{B \times D}$.
2. **As Linear Operators**: A neural network weight layer $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$ transforms an input activation vector $x$ into an output representation $y = W x$.

When we train and deploy deep networks, we are constantly solving or analyzing systems of equations:
- **Forward Mapping ($y = W x$)**: Can the layer generate *any* target feature vector $y$, or is its expressive capability restricted to a lower-dimensional subspace?
- **Invertibility & Reconstruction**: In an autoencoder or normalizing flow, can we uniquely reconstruct $x$ from $y$?
- **Overparameterization & Nullspaces**: Why can a 70-billion-parameter LLM be severely pruned or compressed with LoRA without losing performance? (Answer: Because billions of parameter combinations lie entirely inside the **Nullspace** of the loss surface).

To understand these phenomena with university-level rigor, we turn to **Gilbert Strang's Big Picture of Linear Algebra**: The Four Fundamental Subspaces and the four fundamental views of matrix multiplication.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Matrix Operations & Properties
Let $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times p}$.
- **Transpose Properties**:
  $$(A + B)^T = A^T + B^T, \quad (A B)^T = B^T A^T, \quad (A^T)^{-1} = (A^{-1})^T$$
  *(Note the reversal of multiplication order in transpose of a product).*

---

### 2. The Four Perspectives of Matrix Multiplication $C = A B$
Most high school and introductory courses teach only one way to multiply matrices: row dot product. But in advanced machine learning and high-performance computing (like CUDA kernels, FlashAttention, and SVD), **there are four distinct, mathematically equivalent perspectives**:

```
                       PERSPECTIVE 1: DOT PRODUCT VIEW
                   (Entry by entry: Row of A · Col of B)
                   ┌──────────┐   ┌───┬───┬───┐   ┌───┬───┬───┐
                   │ -------- │ · │   │ | │   │ = │   │ * │   │
                   └──────────┘   └───┴───┴───┘   └───┴───┴───┘
                       Row i          Col j          Entry C_ij

                       PERSPECTIVE 2: COLUMN VIEW
                   (Each column of C is a linear combo of columns of A)
                   ┌───┬───┬───┐   ┌───┐   ┌───┐
                   │   │   │   │ · │ b │ = │ c │  (Col j of C = A · col_j(B))
                   └───┴───┴───┘   └───┘   └───┘

                       PERSPECTIVE 3: ROW VIEW
                   (Each row of C is a linear combo of rows of B)
                   ┌──────────┐   ┌──────────┐   ┌──────────┐
                   │  a_row   │ · │ -------- │ = │  c_row   │  (Row i of C = row_i(A) · B)
                   └──────────┘   └──────────┘   └──────────┘

                       PERSPECTIVE 4: OUTER PRODUCT VIEW
                   (Sum of k Rank-1 Matrices: col_r(A) ⊗ row_r(B))
                   ┌───┐            ┌──────────┐   ┌────────────────────────┐
                   │   │ (col r)  ⊗ │  (row r) │ = │ Rank-1 Matrix (m x p)  │
                   └───┘            └──────────┘   └────────────────────────┘
                   C = ∑ (col_r(A) · row_r(B))   <-- Backbone of SVD & LoRA!
```

#### Mathematical Formulations of the 4 Perspectives:

| # | Perspective Name | Mathematical Formula | Algorithmic / Deep Learning Role |
| :-: | :--- | :--- | :--- |
| **1** | **Dot Product View** | $C_{ij} = \sum_{r=1}^k A_{ir} B_{rj} = \text{row}_i(A) \cdot \text{col}_j(B)$ | Computing a single scalar neuron activation. |
| **2** | **Column View** | $\text{col}_j(C) = A \cdot \text{col}_j(B) = \sum_{r=1}^k B_{rj} \text{col}_r(A)$ | Forward pass: output is in the column space of $A$. |
| **3** | **Row View** | $\text{row}_i(C) = \text{row}_i(A) \cdot B = \sum_{r=1}^k A_{ir} \text{row}_r(B)$ | Backpropagation: gradient backpropagated across rows. |
| **4** | **Outer Product View** | $C = \sum_{r=1}^k \text{col}_r(A) \cdot \text{row}_r(B)$ | LoRA rank decomposition & covariance matrix updates. |

---

### 3. Gilbert Strang's Four Fundamental Subspaces
For any matrix $A \in \mathbb{R}^{m \times n}$ with rank $r$:

```
        R^n (Input Space: n dimensions)               R^m (Output Space: m dimensions)
     ┌────────────────────────────────────┐        ┌────────────────────────────────────┐
     │                                    │        │                                    │
     │      Row Space: C(A^T)             │   A    │      Column Space: C(A)            │
     │      Dimension = r                 │───────>│      Dimension = r                 │
     │   (All vectors that produce        │ 1-to-1 │   (All reachable outputs Ax = b)   │
     │      non-zero outputs)             │ Invert.│                                    │
     │                                    │        │                                    │
     ├────────────────────────────────────┤        ├────────────────────────────────────┤
     │                                    │        │                                    │
     │      Nullspace: N(A)               │   A    │      Left Nullspace: N(A^T)        │
     │      Dimension = n - r             │───────>│      Dimension = m - r             │
     │   (All vectors collapsed to 0:     │  ALL   │   (Unreachable output directions:  │
     │      Ax = 0)                       │   TO   │      A^T y = 0)                    │
     │                                    │   0    │                                    │
     └────────────────────────────────────┘        └────────────────────────────────────┘
         Orthogonal: C(A^T) ⊥ N(A)                      Orthogonal: C(A) ⊥ N(A^T)
```

1. **The Column Space $C(A)$**:
   $$C(A) = \{ A x \mid x \in \mathbb{R}^n \} \subseteq \mathbb{R}^m, \quad \dim(C(A)) = r$$
   The subspace of all reachable outputs in $\mathbb{R}^m$.
2. **The Nullspace $N(A)$**:
   $$N(A) = \{ x \in \mathbb{R}^n \mid A x = \mathbf{0} \} \subseteq \mathbb{R}^n, \quad \dim(N(A)) = n - r$$
   The subspace of all input vectors that matrix $A$ maps to zero.
3. **The Row Space $C(A^T)$**:
   $$C(A^T) = \{ A^T y \mid y \in \mathbb{R}^m \} \subseteq \mathbb{R}^n, \quad \dim(C(A^T)) = r$$
   The column space of $A^T$. It contains all inputs that are orthogonal to the nullspace.
4. **The Left Nullspace $N(A^T)$**:
   $$N(A^T) = \{ y \in \mathbb{R}^m \mid A^T y = \mathbf{0} \} \subseteq \mathbb{R}^m, \quad \dim(N(A^T)) = m - r$$
   The nullspace of $A^T$. It contains all vectors in $\mathbb{R}^m$ perpendicular to $C(A)$.

---

### 4. The Fundamental Theorem of Linear Algebra
Gilbert Strang formalized two monumental geometric truths:

#### Part 1: Equality of Dimensions
$$\text{dim}(\text{Row Space}) = \text{dim}(\text{Column Space}) = r = \text{rank}(A)$$
The number of linearly independent rows in any matrix is **always identical** to the number of linearly independent columns, regardless of matrix dimensions ($m \times n$).

#### Part 2: Orthogonality of the Subspaces
- In the input domain $\mathbb{R}^n$: The Row Space is the **orthogonal complement** of the Nullspace:
  $$C(A^T) \perp N(A) \quad \text{and} \quad C(A^T) \oplus N(A) = \mathbb{R}^n$$
  Every vector $x \in \mathbb{R}^n$ can be split uniquely as $x = x_r + x_n$, where $x_r \in C(A^T)$ and $x_n \in N(A)$.
- In the output domain $\mathbb{R}^m$: The Column Space is the **orthogonal complement** of the Left Nullspace:
  $$C(A) \perp N(A^T) \quad \text{and} \quad C(A) \oplus N(A^T) = \mathbb{R}^m$$

#### Rigorous First-Principles Derivation / Proof: $C(A^T) \perp N(A)$
**Assumptions:**
1. $A \in \mathbb{R}^{m \times n}$ has rows $r_1^T, r_2^T, \dots, r_m^T \in \mathbb{R}^{1 \times n}$.
2. $N(A) = \{x \in \mathbb{R}^n \mid A x = \mathbf{0}\}$.
3. $C(A^T) = \text{span}(r_1, r_2, \dots, r_m) = \{A^T y \mid y \in \mathbb{R}^m\}$.

**Proof Steps:**
1. Let $x \in N(A)$. By definition of the nullspace:
   $$A x = \begin{bmatrix} r_1^T \\ r_2^T \\ \vdots \\ r_m^T \end{bmatrix} x = \begin{bmatrix} r_1^T x \\ r_2^T x \\ \vdots \\ r_m^T x \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$
   Therefore, for every row vector $r_i$ of matrix $A$:
   $$r_i^T x = \langle r_i, x \rangle = 0, \quad \forall i \in \{1, 2, \dots, m\}$$
   Every vector in the nullspace is perpendicular to every row of $A$.
2. Now choose an arbitrary vector $v$ in the row space $C(A^T)$. By definition, $v$ is a linear combination of the rows of $A$:
   $$v = A^T y = \sum_{i=1}^m y_i r_i \quad \text{for some scalar vector } y \in \mathbb{R}^m$$
3. Compute the inner product $\langle v, x \rangle$:
   $$\langle v, x \rangle = v^T x = (A^T y)^T x = y^T (A x)$$
   Since $x \in N(A)$, $A x = \mathbf{0}$:
   $$\langle v, x \rangle = y^T \mathbf{0} = 0$$
   Since this holds for every $v \in C(A^T)$ and every $x \in N(A)$, we conclude:
   $$C(A^T) \perp N(A) \quad \blacksquare$$

By applying this exact logic to the transposed matrix $A^T \in \mathbb{R}^{n \times m}$, where $N(A^T) = \{y \in \mathbb{R}^m \mid A^T y = \mathbf{0}\}$ and $C(A) = C((A^T)^T)$, we immediately obtain:
$$C(A) \perp N(A^T) \quad \blacksquare$$

---

#### Part 3: The Rank-Nullity Theorem: First-Principles Derivation
$$\text{rank}(A) + \dim(N(A)) = n$$

**Assumptions & Setup:**
1. Let $A \in \mathbb{R}^{m \times n}$.
2. Perform Gaussian elimination using elementary row operations to reduce $A$ to its unique Reduced Row Echelon Form $R = \text{RREF}(A)$.
3. Elementary row operations are invertible (represented by an invertible matrix $E \in \mathbb{R}^{m \times m}$ such that $R = E A$).

**Proof Steps:**
1. **Preservation of Nullspace:**
   $$A x = \mathbf{0} \iff E A x = E \mathbf{0} \iff R x = \mathbf{0}$$
   Thus, $N(A) = N(R)$.
2. **Pivot vs. Free Columns in RREF:**
   In $R$, every column corresponds to an unknown $x_j$ ($j = 1, \dots, n$):
   - Let $r$ be the number of pivot columns (leading 1s). By definition, $r = \text{rank}(A)$.
   - The remaining columns contain no pivots and correspond to **free variables**. The number of free variables is exactly $n - r$.
3. **Constructing the Basis for $N(R)$:**
   For each free variable $x_{f_k}$ ($k \in \{1, \dots, n - r\}$), set $x_{f_k} = 1$ and all other free variables to $0$.
   Because each pivot variable appears in exactly one equation containing that pivot's leading 1, each pivot variable is uniquely determined in terms of the free variables:
   $$x_{\text{pivot}, i} = - \sum_{k=1}^{n-r} R_{i, f_k} x_{f_k}$$
   This yields exactly $n - r$ special nullspace solution vectors $\{s_1, s_2, \dots, s_{n-r}\} \subset \mathbb{R}^n$.
4. **Independence and Spanning:**
   - **Linearly Independent:** In each vector $s_k$, the coordinate corresponding to free variable $x_{f_k}$ is $1$, while in all other vectors $s_j$ ($j \neq k$), that coordinate is $0$. Therefore, no non-trivial combination can sum to zero.
   - **Spanning:** Any solution to $R x = \mathbf{0}$ with free variable values $(c_1, \dots, c_{n-r})$ is identical to $\sum_{k=1}^{n-r} c_k s_k$, because both have identical free variables and RREF guarantees pivot variables are uniquely determined by free variables.
5. **Conclusion:**
   $\{s_1, \dots, s_{n-r}\}$ forms an exact basis for $N(A)$.
   $$\dim(N(A)) = n - r = n - \text{rank}(A)$$
   $$\text{rank}(A) + \dim(N(A)) = n \quad \blacksquare$$

---

### 5. Solvability of Linear Systems $A x = b$
Consider $A \in \mathbb{R}^{m \times n}$ and target vector $b \in \mathbb{R}^m$.

#### Consistency Criterion (The Rouché-Capelli Theorem):
A linear system $A x = b$ has at least one solution **if and only if** $b$ lies in the column space of $A$:
$$b \in C(A) \iff \text{rank}(A) = \text{rank}([A \mid b])$$

#### First-Principles Derivation of Rouché-Capelli Criterion:
1. By the column picture of matrix multiplication:
   $$A x = \sum_{j=1}^n x_j \text{col}_j(A)$$
   $A x = b$ possesses a solution $x \in \mathbb{R}^n$ if and only if $b$ can be expressed as a linear combination of the columns of $A$, meaning $b \in \text{span}(\text{col}_1(A), \dots, \text{col}_n(A)) = C(A)$.
2. Now consider the augmented matrix $[A \mid b] \in \mathbb{R}^{m \times (n+1)}$:
   - If $b \in C(A)$, $b$ is linearly dependent on the columns of $A$. Appending $b$ introduces no new pivot column during row reduction. Thus, $\text{rank}([A \mid b]) = \text{rank}(A)$.
   - If $b \notin C(A)$, $b$ is linearly independent of the columns of $A$. Appending $b$ creates an additional pivot column in the final augmented column during row reduction ($[0 \,\, 0 \,\, \dots \,\, 0 \mid d]$ with $d \neq 0$). Thus, $\text{rank}([A \mid b]) = \text{rank}(A) + 1$.
3. **Fredholm Alternative Formulation:**
   Since $C(A) \perp N(A^T)$, $b \in C(A) \iff \langle y, b \rangle = 0$ for all $y \in N(A^T)$.
   If there exists any $y \in N(A^T)$ such that $y^T b \neq 0$, multiplying $A x = b$ by $y^T$ yields:
   $$y^T A x = y^T b \implies (A^T y)^T x = y^T b \implies \mathbf{0}^T x = y^T b \implies 0 = y^T b \neq 0$$
   A direct contradiction! Hence no solution exists. $\blacksquare$

#### Complete Structure of the General Solution:
If $A x = b$ is consistent, its general solution is:
$$x = x_p + x_n$$
where:
- $x_p$ is a **particular solution** satisfying $A x_p = b$.
- $x_n$ is any vector in the **nullspace** $N(A)$ satisfying $A x_n = \mathbf{0}$.
- *Verification:* $A x = A(x_p + x_n) = A x_p + A x_n = b + \mathbf{0} = b$.

#### Uniqueness Conditions:
| Case | Rank Condition | Number of Solutions | Geometric Meaning | Deep Learning Analogy |
| :--- | :--- | :--- | :--- | :--- |
| **Full Column Rank** | $r = n < m$ | $0$ or $1$ | $N(A) = \{\mathbf{0}\}$. Columns are independent. | Overdetermined system (Linear Regression). |
| **Full Row Rank** | $r = m < n$ | $\infty$ | $C(A) = \mathbb{R}^m$. Every $b$ is reachable. | Underdetermined system (Overparameterized NN). |
| **Square Invertible** | $r = m = n$ | Exactly $1$ | $A$ is bijective (1-to-1 and onto). | Reversible Neural Network (Normalizing Flow). |
| **Rank Deficient** | $r < \min(m, n)$ | $0$ or $\infty$ | Both $N(A)$ and $N(A^T)$ are non-trivial. | Low-Rank bottleneck layer / collapsed layer. |

---

## Part 3: Geometric & Algebraic Interpretation

### The Invertibility of $A$ between $C(A^T)$ and $C(A)$
Look closely at Strang's diagram above. Even when matrix $A$ is not square and not invertible over the whole space:
- Every vector in the nullspace $x_n \in N(A)$ is wiped out to $\mathbf{0}$.
- But if we restrict $A$ to the row space $C(A^T)$, **$A$ is a perfect, 1-to-1, bijective transformation between $C(A^T)$ and $C(A)$!**
- The pseudo-inverse (Moore-Penrose) $A^+$ inverts this exact 1-to-1 mapping in reverse, mapping $C(A)$ back to $C(A^T)$.

---

## Part 4: Real-World Analogy

### The 3D Movie Projector
Imagine a 3D physical object in your room ($n = 3$), and a light projector casting its shadow onto a 2D cinema screen ($m = 2$):
- **Matrix $A$ ($2 \times 3$)** is the projection mapping.
- **Column Space $C(A)$**: The 2D cinema screen. Any point on the screen can be illuminated ($\dim = 2$).
- **Nullspace $N(A)$**: The line of sight from the projector lamp through an object. If you move an object along this 1-dimensional ray ($\dim = 3 - 2 = 1$), its shadow on the screen does not move at all!
- In Deep Learning, adding an adversarial perturbation in the direction of the model's nullspace changes the weights without changing the activations!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### The 4 Ways to Multiply Matrices on a Toy Neural Layer by Hand

Let us trace a linear layer forward pass:
$$Y = X W$$
where:
- **Batch Input**: $X \in \mathbb{R}^{2 \times 2}$ ($B = 2$ samples, $d_{\text{in}} = 2$ features)
- **Weight Matrix**: $W \in \mathbb{R}^{2 \times 3}$ ($d_{\text{in}} = 2$ inputs, $d_{\text{out}} = 3$ output neurons)
- **Output Activations**: $Y \in \mathbb{R}^{2 \times 3}$

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Shape | Mathematical Role | Deep Learning Meaning | Concrete Toy Values |
| :---: | :---: | :--- | :--- | :---: |
| $X$ | $[2 \times 2]$ | Input Matrix | Batch of 2 input tokens ($x_1, x_2$) | $\begin{bmatrix} 1 & 2 \\ 3 & 0 \end{bmatrix}$ |
| $W$ | $[2 \times 3]$ | Weight Matrix | Linear layer connection weights | $\begin{bmatrix} 2 & 1 & 4 \\ 0 & 3 & -1 \end{bmatrix}$ |
| $Y$ | $[2 \times 3]$ | Product Matrix | Layer output activations | To be computed via all 4 views |

---

#### 2. Visual Input Grids

```
         INPUT BATCH X (2 x 2)                      WEIGHT MATRIX W (2 x 3)
           Col 1      Col 2                    Col 1      Col 2      Col 3
        ┌──────────┬──────────┐             ┌──────────┬──────────┬──────────┐
  Row 1 │  X_11=1  │  X_12=2  │       Row 1 │  W_11=2  │  W_12=1  │  W_13=4  │
        ├──────────┼──────────┤             ├──────────┼──────────┼──────────┤
  Row 2 │  X_21=3  │  X_22=0  │       Row 2 │  W_21=0  │  W_22=3  │  W_23=-1 │
        └──────────┴──────────┘             └──────────┴──────────┴──────────┘
```

---

#### Perspective 1: Dot Product View (Cell by Cell)
Each cell $Y_{ij} = \text{row}_i(X) \cdot \text{col}_j(W)$:

$$\begin{bmatrix} Y_{11} & Y_{12} & Y_{13} \\ Y_{21} & Y_{22} & Y_{23} \end{bmatrix}$$

| Output Cell | Row of $X$ | $\cdot$ | Col of $W$ | Exact Arithmetic | Result |
| :---: | :---: | :---: | :---: | :--- | :---: |
| **$Y_{11}$** | $[1, 2]$ | $\cdot$ | $[2, 0]^T$ | $(1 \times 2) + (2 \times 0) = 2 + 0$ | **$2$** |
| **$Y_{12}$** | $[1, 2]$ | $\cdot$ | $[1, 3]^T$ | $(1 \times 1) + (2 \times 3) = 1 + 6$ | **$7$** |
| **$Y_{13}$** | $[1, 2]$ | $\cdot$ | $[4, -1]^T$| $(1 \times 4) + (2 \times -1) = 4 - 2$ | **$2$** |
| **$Y_{21}$** | $[3, 0]$ | $\cdot$ | $[2, 0]^T$ | $(3 \times 2) + (0 \times 0) = 6 + 0$ | **$6$** |
| **$Y_{22}$** | $[3, 0]$ | $\cdot$ | $[1, 3]^T$ | $(3 \times 1) + (0 \times 3) = 3 + 0$ | **$3$** |
| **$Y_{23}$** | $[3, 0]$ | $\cdot$ | $[4, -1]^T$| $(3 \times 4) + (0 \times -1) = 12 + 0$ | **$12$** |

$$Y = \begin{bmatrix} 2 & 7 & 2 \\ 6 & 3 & 12 \end{bmatrix}$$

---

#### Perspective 2: Column View (Linear Combinations of Columns of $X$)
$$\text{col}_j(Y) = W_{1j} \cdot \text{col}_1(X) + W_{2j} \cdot \text{col}_2(X)$$

- **Column 1 of $Y$**:
  $$\text{col}_1(Y) = 2 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + 0 \begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 6 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 6 \end{bmatrix} \quad \checkmark$$
- **Column 2 of $Y$**:
  $$\text{col}_2(Y) = 1 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + 3 \begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \end{bmatrix} + \begin{bmatrix} 6 \\ 0 \end{bmatrix} = \begin{bmatrix} 7 \\ 3 \end{bmatrix} \quad \checkmark$$
- **Column 3 of $Y$**:
  $$\text{col}_3(Y) = 4 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + (-1) \begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 4 \\ 12 \end{bmatrix} + \begin{bmatrix} -2 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 12 \end{bmatrix} \quad \checkmark$$

*Deep Learning Meaning:* Every output feature channel is a linear blend of the input feature channels!

---

#### Perspective 3: Row View (Linear Combinations of Rows of $W$)
$$\text{row}_i(Y) = X_{i1} \cdot \text{row}_1(W) + X_{i2} \cdot \text{row}_2(W)$$

- **Row 1 of $Y$ (Sample 1 activations)**:
  $$\text{row}_1(Y) = 1 \cdot [2, 1, 4] + 2 \cdot [0, 3, -1] = [2, 1, 4] + [0, 6, -2] = [2, 7, 2] \quad \checkmark$$
- **Row 2 of $Y$ (Sample 2 activations)**:
  $$\text{row}_2(Y) = 3 \cdot [2, 1, 4] + 0 \cdot [0, 3, -1] = [6, 3, 12] + [0, 0, 0] = [6, 3, 12] \quad \checkmark$$

*Deep Learning Meaning:* Each individual sample in a batch gathers a weighted mixture of the network's neuron filters!

---

#### Perspective 4: Outer Product View (Sum of Rank-1 Matrices)
$$Y = \text{col}_1(X) \cdot \text{row}_1(W) + \text{col}_2(X) \cdot \text{row}_2(W)$$

- **Rank-1 Outer Product 1**:
  $$M_1 = \begin{bmatrix} 1 \\ 3 \end{bmatrix} \begin{bmatrix} 2 & 1 & 4 \end{bmatrix} = \begin{bmatrix} 1 \times 2 & 1 \times 1 & 1 \times 4 \\ 3 \times 2 & 3 \times 1 & 3 \times 4 \end{bmatrix} = \begin{bmatrix} 2 & 1 & 4 \\ 6 & 3 & 12 \end{bmatrix}$$
- **Rank-1 Outer Product 2**:
  $$M_2 = \begin{bmatrix} 2 \\ 0 \end{bmatrix} \begin{bmatrix} 0 & 3 & -1 \end{bmatrix} = \begin{bmatrix} 2 \times 0 & 2 \times 3 & 2 \times -1 \\ 0 \times 0 & 0 \times 3 & 0 \times -1 \end{bmatrix} = \begin{bmatrix} 0 & 6 & -2 \\ 0 & 0 & 0 \end{bmatrix}$$
- **Summing the Outer Products**:
  $$Y = M_1 + M_2 = \begin{bmatrix} 2+0 & 1+6 & 4-2 \\ 6+0 & 3+0 & 12+0 \end{bmatrix} = \begin{bmatrix} 2 & 7 & 2 \\ 6 & 3 & 12 \end{bmatrix} \quad \checkmark$$

> [!IMPORTANT]
> **Why the Outer Product View is Crucial for Modern AI:**
> In Low-Rank Adaptation (LoRA), $\Delta W = B A$ expresses the weight update as a sum of $r$ rank-1 outer products $\sum_{i=1}^r b_i a_i^T$. This perspective proves that large matrix transformations are built by accumulating simple 1D rank-1 directional sheets!

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Standard Case — Full 4-Subspace Analysis of a $3 \times 4$ Matrix

**Problem:** Given matrix $A \in \mathbb{R}^{3 \times 4}$:
$$A = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 1 & 3 & 1 & 1 \end{bmatrix}$$
1. Find bases and dimensions for all **Four Fundamental Subspaces**: $C(A), N(A), C(A^T), N(A^T)$.
2. Verify the Rank-Nullity Theorem: $\text{rank}(A) + \dim(N(A)) = 4$.
3. Prove orthogonality: Check that every basis vector of $N(A)$ is orthogonal to every basis vector of $C(A^T)$.

#### "What Refers to What" — Variable Dictionary
| Symbol | Ambient Space | Meaning |
| :---: | :---: | :--- |
| $m = 3, n = 4$ | $\mathbb{N}$ | 3 equations (rows), 4 unknowns (columns) |
| $r$ | $\mathbb{N}$ | Matrix rank (number of pivots) |
| $C(A)$ | $\mathbb{R}^3$ | Column space (subspace in output space) |
| $N(A)$ | $\mathbb{R}^4$ | Nullspace (subspace in input space) |
| $C(A^T)$ | $\mathbb{R}^4$ | Row space (subspace in input space) |
| $N(A^T)$ | $\mathbb{R}^3$ | Left nullspace (subspace in output space) |

---

#### Step 1: Reduce $A$ to Reduced Row Echelon Form (RREF)
$$\begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 1 & 3 & 1 & 1 \end{bmatrix}$$

- **Row Operation 1 ($R_3 \leftarrow R_3 - R_1$):**
  $$\begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \end{bmatrix}$$
- **Row Operation 2 ($R_3 \leftarrow R_3 - R_2$):**
  $$\begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$
- **Row Operation 3 ($R_1 \leftarrow R_1 - 2R_2$):**
  $$\text{RREF}(A) = \begin{bmatrix} 1 & 0 & -2 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

---

#### Step 2: Extracting the Subspaces

1. **Rank & Pivots:**
   - Pivots are in **Column 1** and **Column 2** $\implies \text{rank}(A) = r = 2$.
   - Free variables are $x_3$ and $x_4$ ($n - r = 4 - 2 = 2$).

2. **Column Space $C(A) \subseteq \mathbb{R}^3$:**
   Take the pivot columns from the **original matrix $A$**:
   $$\mathcal{B}_{C(A)} = \left\{ \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix} \right\}, \quad \dim(C(A)) = 2$$

3. **Row Space $C(A^T) \subseteq \mathbb{R}^4$:**
   Take the non-zero rows of the **RREF matrix**:
   $$\mathcal{B}_{C(A^T)} = \left\{ \begin{bmatrix} 1 \\ 0 \\ -2 \\ 1 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} \right\}, \quad \dim(C(A^T)) = 2$$

4. **Nullspace $N(A) \subseteq \mathbb{R}^4$:**
   Solve $\text{RREF}(A) x = \mathbf{0}$:
   $$\begin{cases} x_1 - 2x_3 + x_4 = 0 \implies x_1 = 2x_3 - x_4 \\ x_2 + x_3 = 0 \implies x_2 = -x_3 \end{cases}$$
   Express general nullspace vector:
   $$x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} = \begin{bmatrix} 2x_3 - x_4 \\ -x_3 \\ x_3 \\ x_4 \end{bmatrix} = x_3 \begin{bmatrix} 2 \\ -1 \\ 1 \\ 0 \end{bmatrix} + x_4 \begin{bmatrix} -1 \\ 0 \\ 0 \\ 1 \end{bmatrix}$$
   $$\mathcal{B}_{N(A)} = \left\{ \begin{bmatrix} 2 \\ -1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -1 \\ 0 \\ 0 \\ 1 \end{bmatrix} \right\}, \quad \dim(N(A)) = 2$$

5. **Left Nullspace $N(A^T) \subseteq \mathbb{R}^3$:**
   Solve $A^T y = \mathbf{0}$. We row reduce $A^T$:
   $$A^T = \begin{bmatrix} 1 & 0 & 1 \\ 2 & 1 & 3 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix} \to \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
   Equations: $y_1 + y_3 = 0 \implies y_1 = -y_3$, and $y_2 + y_3 = 0 \implies y_2 = -y_3$.
   $$\mathcal{B}_{N(A^T)} = \left\{ \begin{bmatrix} -1 \\ -1 \\ 1 \end{bmatrix} \right\}, \quad \dim(N(A^T)) = m - r = 3 - 2 = 1$$

---

#### Step 3: Verification of Fundamental Theorems

1. **Rank-Nullity Theorem Check:**
   $$\text{rank}(A) + \dim(N(A)) = 2 + 2 = 4 = n \quad \checkmark$$
   $$\text{rank}(A) + \dim(N(A^T)) = 2 + 1 = 3 = m \quad \checkmark$$

2. **Orthogonality Check ($C(A^T) \perp N(A)$):**
   Take row space basis vectors $r_1, r_2$ and nullspace basis vectors $n_1, n_2$:
   $$r_1 = [1, 0, -2, 1]^T, \quad r_2 = [0, 1, 1, 0]^T$$
   $$n_1 = [2, -1, 1, 0]^T, \quad n_2 = [-1, 0, 0, 1]^T$$
   - $r_1 \cdot n_1 = (1 \times 2) + (0 \times -1) + (-2 \times 1) + (1 \times 0) = 2 + 0 - 2 + 0 = \mathbf{0} \quad \checkmark$
   - $r_1 \cdot n_2 = (1 \times -1) + (0 \times 0) + (-2 \times 0) + (1 \times 1) = -1 + 0 + 0 + 1 = \mathbf{0} \quad \checkmark$
   - $r_2 \cdot n_1 = (0 \times 2) + (1 \times -1) + (1 \times 1) + (0 \times 0) = 0 - 1 + 1 + 0 = \mathbf{0} \quad \checkmark$
   - $r_2 \cdot n_2 = (0 \times -1) + (1 \times 0) + (1 \times 0) + (0 \times 1) = 0 + 0 + 0 + 0 = \mathbf{0} \quad \checkmark$
   Every vector in the row space is strictly perpendicular to every vector in the nullspace!

---

### Scenario B: Solving $A x = b$ (Particular + Homogeneous Decomposition)

Using matrix $A$ from Scenario A:
Find the complete solution to $A x = b$ where $b = \begin{bmatrix} 3 \\ 1 \\ 4 \end{bmatrix}$.

1. **Check Consistency:**
   Set up augmented matrix $[A \mid b]$ and apply the exact same row operations:
   $$\begin{bmatrix} 1 & 2 & 0 & 1 & | & 3 \\ 0 & 1 & 1 & 0 & | & 1 \\ 1 & 3 & 1 & 1 & | & 4 \end{bmatrix} \xrightarrow{R_3 \leftarrow R_3 - R_1 - R_2} \begin{bmatrix} 1 & 2 & 0 & 1 & | & 3 \\ 0 & 1 & 1 & 0 & | & 1 \\ 0 & 0 & 0 & 0 & | & 0 \end{bmatrix} \xrightarrow{R_1 \leftarrow R_1 - 2R_2} \begin{bmatrix} 1 & 0 & -2 & 1 & | & 1 \\ 0 & 1 & 1 & 0 & | & 1 \\ 0 & 0 & 0 & 0 & | & 0 \end{bmatrix}$$
   Row 3 is $0 = 0 \implies$ The system is **consistent**! ($b \in C(A)$).

2. **Find a Particular Solution $x_p$:**
   Set all free variables to zero ($x_3 = 0, x_4 = 0$):
   - $x_1 = 1$
   - $x_2 = 1$
   $$x_p = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 0 \end{bmatrix}$$
   *Check:* $A x_p = \begin{bmatrix} 1(1) + 2(1) \\ 0(1) + 1(1) \\ 1(1) + 3(1) \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \\ 4 \end{bmatrix} = b \quad \checkmark$

3. **Complete General Solution:**
   $$x = x_p + x_n = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 0 \end{bmatrix} + c_1 \begin{bmatrix} 2 \\ -1 \\ 1 \\ 0 \end{bmatrix} + c_2 \begin{bmatrix} -1 \\ 0 \\ 0 \\ 1 \end{bmatrix}, \quad \forall c_1, c_2 \in \mathbb{R}$$
   There are infinitely many distinct inputs that map to the exact same target activation $b$!

---

### Scenario C: Inconsistent Linear System ($0 = k \neq 0$ and Left Nullspace Contradiction)

**Problem Formulation:**
Consider the overdetermined linear system $A x = b$:
$$A = \begin{bmatrix} 1 & 1 \\ 2 & 2 \\ 1 & 3 \end{bmatrix}, \quad b = \begin{bmatrix} 2 \\ 5 \\ 4 \end{bmatrix}$$
1. Set up the augmented matrix $[A \mid b]$ and apply Gaussian elimination to show that no exact solution exists.
2. Identify the pivot structure and compare $\text{rank}(A)$ vs $\text{rank}([A \mid b])$.
3. Find a non-zero left nullspace vector $y \in N(A^T)$ and verify that $y^T b \neq 0$, demonstrating Fredholm's obstruction directly.

#### Step 1: Row Operations on Augmented Matrix
$$\begin{bmatrix} 1 & 1 & | & 2 \\ 2 & 2 & | & 5 \\ 1 & 3 & | & 4 \end{bmatrix}$$
- **Row Operation 1 ($R_2 \leftarrow R_2 - 2R_1$):**
  - Col 1: $2 - 2(1) = 0$
  - Col 2: $2 - 2(1) = 0$
  - Target: $5 - 2(2) = 1$
  $$\begin{bmatrix} 1 & 1 & | & 2 \\ 0 & 0 & | & 1 \\ 1 & 3 & | & 4 \end{bmatrix}$$
- **Row Operation 2 ($R_3 \leftarrow R_3 - R_1$):**
  - Col 1: $1 - 1 = 0$
  - Col 2: $3 - 1 = 2$
  - Target: $4 - 2 = 2$
  $$\begin{bmatrix} 1 & 1 & | & 2 \\ 0 & 0 & | & 1 \\ 0 & 2 & | & 2 \end{bmatrix}$$
- **Row Operation 3 (Swap $R_2 \leftrightarrow R_3$):**
  $$\begin{bmatrix} 1 & 1 & | & 2 \\ 0 & 2 & | & 2 \\ 0 & 0 & | & 1 \end{bmatrix}$$

Look at the third row:
$$0 x_1 + 0 x_2 = 1 \implies 0 = 1 \quad \text{\bf (CONTRADICTION!)}$$

#### Step 2: Rank Evaluation
- Coefficient matrix $A$ has pivots in columns 1 and 2: $\text{rank}(A) = 2$.
- Augmented matrix $[A \mid b]$ has pivots in column 1, column 2, and the augmented column: $\text{rank}([A \mid b]) = 3$.
- Since $\text{rank}(A) = 2 \neq 3 = \text{rank}([A \mid b])$, by the Rouché-Capelli theorem the system is **inconsistent** (zero solutions).

#### Step 3: Fredholm Left Nullspace Obstruction
Solve $A^T y = \mathbf{0}$ for $y = [y_1, y_2, y_3]^T$:
$$A^T = \begin{bmatrix} 1 & 2 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
Subtracting equation 1 from equation 2 gives $2y_3 = 0 \implies y_3 = 0$.
Then $y_1 + 2y_2 = 0 \implies y_1 = -2y_2$.
Pick $y_2 = 1 \implies y = \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} \in N(A^T)$.
Now compute the inner product with $b$:
$$y^T b = (-2)(2) + (1)(5) + (0)(4) = -4 + 5 + 0 = 1 \neq 0$$
Since $b$ has a non-zero projection onto the left nullspace $N(A^T)$, target vector $b$ lies strictly outside $C(A)$.

---

### Scenario D: Rank-Deficient Bottleneck Layer (Information Erasure in Autoencoders)

**Problem Formulation:**
In an autoencoder or low-rank adapter, consider an input $x \in \mathbb{R}^3$, an encoder down-projection $A \in \mathbb{R}^{1 \times 3}$, and a decoder up-projection $B \in \mathbb{R}^{3 \times 1}$:
$$A = \begin{bmatrix} 1 & -1 & 2 \end{bmatrix}, \quad B = \begin{bmatrix} 2 \\ 0 \\ 1 \end{bmatrix}$$
The composite end-to-end transformation is $W = B A \in \mathbb{R}^{3 \times 3}$.
1. Compute the explicit matrix $W = B A$.
2. Determine $\text{rank}(W)$ and the dimension of the nullspace $\dim(N(W))$.
3. Find a basis for the nullspace $N(W) \subset \mathbb{R}^3$.
4. Test two distinct inputs $x_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$ and $x_2 = \begin{bmatrix} 3 \\ 5 \\ 1 \end{bmatrix}$, showing that their difference lies in $N(W)$ and that $W x_1 = W x_2$.

#### Step 1: Compute Outer Product $W = B A$
$$W = \begin{bmatrix} 2 \\ 0 \\ 1 \end{bmatrix} \begin{bmatrix} 1 & -1 & 2 \end{bmatrix} = \begin{bmatrix} 2(1) & 2(-1) & 2(2) \\ 0(1) & 0(-1) & 0(2) \\ 1(1) & 1(-1) & 1(2) \end{bmatrix} = \begin{bmatrix} 2 & -2 & 4 \\ 0 & 0 & 0 \\ 1 & -1 & 2 \end{bmatrix}$$

#### Step 2: Rank & Nullspace Dimension
- Row 1 is $2 \times$ Row 3, and Row 2 is all zeros.
- Therefore, there is only $1$ linearly independent row $\implies \text{rank}(W) = 1$.
- By the Rank-Nullity Theorem:
  $$\dim(N(W)) = n - \text{rank}(W) = 3 - 1 = 2$$

#### Step 3: Find Basis for $N(W)$
Solve $W x = \mathbf{0}$, which reduces to the single equation:
$$x_1 - x_2 + 2x_3 = 0 \implies x_1 = x_2 - 2x_3$$
Setting free variables $(x_2 = 1, x_3 = 0)$ and $(x_2 = 0, x_3 = 1)$:
$$\mathcal{B}_{N(W)} = \left\{ \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -2 \\ 0 \\ 1 \end{bmatrix} \right\}$$

#### Step 4: Verification of Nullspace Information Collapse
Let inputs be $x_1 = [1, 1, 0]^T$ and $x_2 = [3, 5, 1]^T$.
Compute the difference vector $\Delta x = x_2 - x_1$:
$$\Delta x = \begin{bmatrix} 3 - 1 \\ 5 - 1 \\ 1 - 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \\ 1 \end{bmatrix}$$
Check membership in $N(W)$:
$$W \Delta x = \begin{bmatrix} 2 & -2 & 4 \\ 0 & 0 & 0 \\ 1 & -1 & 2 \end{bmatrix} \begin{bmatrix} 2 \\ 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 2(2) - 2(4) + 4(1) \\ 0 \\ 1(2) - 1(4) + 2(1) \end{bmatrix} = \begin{bmatrix} 4 - 8 + 4 \\ 0 \\ 2 - 4 + 2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$
Indeed, $\Delta x \in N(W)$.
Now check forward passes:
$$W x_1 = \begin{bmatrix} 2 & -2 & 4 \\ 0 & 0 & 0 \\ 1 & -1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 - 2 + 0 \\ 0 \\ 1 - 1 + 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$
$$W x_2 = \begin{bmatrix} 2 & -2 & 4 \\ 0 & 0 & 0 \\ 1 & -1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ 5 \\ 1 \end{bmatrix} = \begin{bmatrix} 6 - 10 + 4 \\ 0 \\ 3 - 5 + 2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$
*Insight:* Even though $x_1$ and $x_2$ are distinct input states, the 2-dimensional nullspace wipes out their differences completely. This demonstrates how bottleneck dimensionality directly determines information loss in deep representations.

---

## Part 7: Deep Learning Connection & Application

### 1. Overparameterization & Pruning in Deep Networks
In large models (e.g., an MLP with $d_{\text{in}} = 10000$ and $d_{\text{out}} = 1000$):
- $\text{rank}(W) \le 1000$.
- The dimension of the nullspace is at least:
  $$\dim(N(W)) = 10000 - 1000 = 9000$$
- This means $90\%$ of the input space is completely zeroed out by this layer!
- When pruning or distilling a neural network, removing components that lie in $N(W)$ incurs **zero change** in model outputs.

### 2. SVD and Low-Rank Weight Approximations
By the outer product perspective, any weight matrix $W \in \mathbb{R}^{m \times n}$ can be written via its Singular Value Decomposition (SVD):
$$W = U \Sigma V^T = \sum_{i=1}^r \sigma_i u_i v_i^T$$
- Truncating this sum to the top $k$ terms ($k \ll r$) yields the optimal rank-$k$ approximation (Eckart-Young Theorem).
- This is the exact mathematical foundation behind weight quantization, parameter pruning, and LoRA adaptation.

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/03_matrix_algebra_and_systems.py`](./code/03_matrix_algebra_and_systems.py) implements:
1. **Four-Way Matrix Multiplication Engine**: Runs all 4 multiplication perspectives and asserts they match bit-for-bit.
2. **Four Fundamental Subspaces Extractor**: Computes exact bases for $C(A), N(A), C(A^T), N(A^T)$ and verifies the Rank-Nullity and Orthogonality theorems.
3. **Linear System Solver ($A x = b$)**: Decomposes solutions into $x = x_p + x_n$.
4. **PyTorch Layer Nullspace Invariance Demo**: Proves that adding a perturbation $\Delta x \in N(W)$ produces zero output change ($\|W(x + \Delta x) - W x\| = 0.0$).

---

## Chapter 1.3 Summary & Review Checklist

- [x] Can you calculate a matrix multiplication using all 4 perspectives (Dot product, Column view, Row view, Outer product view)?
- [x] What are the Four Fundamental Subspaces of a matrix, and what are their dimensions?
- [x] Why is the Row Space orthogonal to the Nullspace ($C(A^T) \perp N(A)$)?
- [x] Why does an overparameterized linear layer ($n > m$) guarantee the existence of a non-trivial nullspace?
- [x] How does the general solution $x = x_p + x_n$ explain why diverse inputs can yield identical hidden activations?
