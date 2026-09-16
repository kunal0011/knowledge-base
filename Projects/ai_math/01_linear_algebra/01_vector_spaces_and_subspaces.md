# Chapter 1.1: Vector Spaces, Subspaces & Basis

---

## Part 1: Intuition & 101 Motivation

In modern Artificial Intelligence and Deep Learning, **every entity is converted into a vector**:
- A single token in a Large Language Model (LLM) is an embedding vector $x \in \mathbb{R}^{4096}$.
- An image feature map in a Convolutional Network is a flattened vector in $\mathbb{R}^{2048}$.
- A layer's hidden activations represent a point in a high-dimensional continuous space.

To manipulate, combine, compress, and transform these representations without breaking them, we need an unambiguous set of mathematical rules. For instance:
- If we blend two latent representations (e.g., in image generation or word arithmetic $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}}$), does the result still belong to the same space?
- Can a neural network layer capture all necessary features with a small number of neurons, or will it lose information?
- What is the minimal set of "fundamental features" needed to reconstruct any input?

**Linear Algebra** gives us the rigorous language to answer these questions. At its core is the **Vector Space**: an algebraic arena where vectors can be scaled, added, and combined reliably without leaving the space.

---

## Part 2: Rigorous Mathematical Formulation

### 1. The Underlying Field $\mathbb{F}$
A **field** $(\mathbb{F}, +, \cdot)$ is an algebraic structure with two operations (addition and multiplication) satisfying commutativity, associativity, distributivity, existence of additive/multiplicative identities ($0$ and $1$), and inverses. In deep learning, our field is almost universally the field of real numbers $\mathbb{R}$ (or occasionally complex numbers $\mathbb{C}$ in Fourier neural operators or signal processing).

### 2. Definition of a Vector Space
Let $\mathbb{F}$ be a field. A **vector space** over $\mathbb{F}$ is a set $V$ equipped with two binary operations:
1. **Vector Addition**: $+ : V \times V \to V$, denoted $(u, v) \mapsto u + v$
2. **Scalar Multiplication**: $\cdot : \mathbb{F} \times V \to V$, denoted $(c, v) \mapsto c v$

Such that for all $u, v, w \in V$ and all $a, b \in \mathbb{F}$, the following **8 Axioms** hold:

| Axiom Category | Name | Formal Statement |
| :--- | :--- | :--- |
| **Addition** | **A1. Commutativity** | $u + v = v + u$ |
| | **A2. Associativity** | $(u + v) + w = u + (v + w)$ |
| | **A3. Additive Identity** | $\exists\, \mathbf{0} \in V \text{ such that } v + \mathbf{0} = v, \quad \forall v \in V$ |
| | **A4. Additive Inverse** | $\forall v \in V, \, \exists\, (-v) \in V \text{ such that } v + (-v) = \mathbf{0}$ |
| **Scalar Multiplication** | **M1. Associativity / Compatibility** | $a(bv) = (ab)v$ |
| | **M2. Multiplicative Identity** | $1 \cdot v = v \quad \text{where } 1 \text{ is the identity in } \mathbb{F}$ |
| **Distributivity** | **D1. Over Vector Addition** | $a(u + v) = au + av$ |
| | **D2. Over Field Addition** | $(a + b)v = av + bv$ |

> [!NOTE]
> The elements of $V$ are called **vectors**, and the elements of $\mathbb{F}$ are called **scalars**.

---

### 3. Vector Subspaces
A subset $W \subseteq V$ of a vector space $V$ is called a **subspace** of $V$ (written $W \le V$) if $W$ is itself a vector space over $\mathbb{F}$ under the operations inherited from $V$.

#### The Two-Step Subspace Test
Rather than checking all 8 axioms, a subset $W \subseteq V$ is a subspace if and only if:
1. **Non-emptiness / Zero Vector**: The zero vector of $V$ is in $W$, i.e., $\mathbf{0} \in W$.
2. **Closure under Linear Combinations**: For all $u, v \in W$ and all $a, b \in \mathbb{F}$:
   $$a u + b v \in W$$

---

### 4. Linear Combinations & Linear Span
Let $S = \{v_1, v_2, \dots, v_k\}$ be a non-empty subset of vectors in $V$.
- A vector $w \in V$ is a **linear combination** of $S$ if there exist scalars $c_1, c_2, \dots, c_k \in \mathbb{F}$ such that:
  $$w = \sum_{i=1}^k c_i v_i = c_1 v_1 + c_2 v_2 + \dots + c_k v_k$$
- The **Linear Span** (or simply **Span**) of $S$, denoted $\text{span}(S)$, is the set of all possible linear combinations of $S$:
  $$\text{span}(S) = \left\{ \sum_{i=1}^k c_i v_i \;\middle|\; c_i \in \mathbb{F}, \, k \in \mathbb{N} \right\}$$

> [!IMPORTANT]
> **Theorem (Span is a Subspace):** For any subset $S \subseteq V$, $\text{span}(S)$ is guaranteed to be a subspace of $V$. Moreover, it is the smallest subspace containing $S$.

#### Rigorous First-Principles Derivation / Proof:
**Assumptions:**
1. $V$ is a vector space over field $\mathbb{F}$ with additive identity $\mathbf{0}$.
2. $S = \{v_1, v_2, \dots, v_k\} \subseteq V$ is an arbitrary non-empty subset of vectors.
3. $\text{span}(S) = \left\{ \sum_{i=1}^k c_i v_i \;\middle|\; c_i \in \mathbb{F} \right\}$.

**Proof Steps:**
We verify the two-step subspace test for $W = \text{span}(S)$:
1. **Contains Zero Vector $\mathbf{0}$:**
   Choose scalar coefficients $c_1 = c_2 = \dots = c_k = 0 \in \mathbb{F}$.
   Then:
   $$\sum_{i=1}^k 0 \cdot v_i = \mathbf{0} + \mathbf{0} + \dots + \mathbf{0} = \mathbf{0} \in \text{span}(S)$$
   Thus, the additive identity is guaranteed to lie in $\text{span}(S)$.

2. **Closure under Addition and Scalar Multiplication (Linear Combinations):**
   Let $u, w \in \text{span}(S)$ and let $\alpha, \beta \in \mathbb{F}$.
   Since $u, w \in \text{span}(S)$, there exist scalar sets $\{a_1, \dots, a_k\} \subset \mathbb{F}$ and $\{b_1, \dots, b_k\} \subset \mathbb{F}$ such that:
   $$u = \sum_{i=1}^k a_i v_i, \quad w = \sum_{i=1}^k b_i v_i$$
   Now consider the linear combination $\alpha u + \beta w$:
   $$\begin{aligned}
   \alpha u + \beta w &= \alpha \left(\sum_{i=1}^k a_i v_i\right) + \beta \left(\sum_{i=1}^k b_i v_i\right) \\
   &= \sum_{i=1}^k (\alpha a_i) v_i + \sum_{i=1}^k (\beta b_i) v_i \quad \text{(by scalar multiplication associativity M1 and distributivity D1)} \\
   &= \sum_{i=1}^k (\alpha a_i + \beta b_i) v_i \quad \text{(by vector addition commutativity A1 and distributivity D2)}
   \end{aligned}$$
   Since $\mathbb{F}$ is a field closed under addition and multiplication, $\gamma_i \triangleq \alpha a_i + \beta b_i \in \mathbb{F}$ for each $i \in \{1, \dots, k\}$.
   Therefore, $\alpha u + \beta w = \sum_{i=1}^k \gamma_i v_i$ is explicitly a linear combination of elements in $S$, which means $\alpha u + \beta w \in \text{span}(S)$.
   
   Hence, $\text{span}(S)$ is a subspace of $V$. $\blacksquare$

---

### 5. Linear Independence and Dependence
Let $\{v_1, v_2, \dots, v_k\} \subset V$.
Consider the **vector equation**:
$$c_1 v_1 + c_2 v_2 + \dots + c_k v_k = \mathbf{0}$$

- The set is **Linearly Independent** if the *only* solution to this equation is the trivial solution:
  $$c_1 = c_2 = \dots = c_k = 0$$
- The set is **Linearly Dependent** if there exists at least one non-trivial solution, i.e., scalars $c_1, \dots, c_k$ not all zero such that $\sum_{i=1}^k c_i v_i = \mathbf{0}$.
  - Equivalently, at least one vector in the set can be written as a linear combination of the others:
    $$v_j = \sum_{i \neq j} \alpha_i v_i$$

---

### 6. Basis and Dimension
A set $\mathcal{B} = \{v_1, v_2, \dots, v_n\} \subset V$ is a **Basis** of $V$ if:
1. $\mathcal{B}$ is **Linearly Independent**.
2. $\mathcal{B}$ **Spans** $V$, i.e., $\text{span}(\mathcal{B}) = V$.

#### Key Theorems:
- **Unique Representation Theorem:** If $\mathcal{B} = \{v_1, \dots, v_n\}$ is a basis for $V$, then every vector $x \in V$ can be written in **exactly one way** as a linear combination:
  $$x = c_1 v_1 + c_2 v_2 + \dots + c_n v_n$$
  The unique tuple $[c_1, c_2, \dots, c_n]^T \in \mathbb{F}^n$ is called the **coordinate vector** of $x$ relative to $\mathcal{B}$, denoted $[x]_\mathcal{B}$.

#### Rigorous First-Principles Derivation / Proof of Unique Representation:
**Assumptions:**
1. $\mathcal{B} = \{v_1, \dots, v_n\}$ spans $V$, ensuring that for any $x \in V$, at least one linear combination exists.
2. $\mathcal{B}$ is linearly independent: $\sum_{i=1}^n \alpha_i v_i = \mathbf{0} \implies \alpha_1 = \alpha_2 = \dots = \alpha_n = 0$.

**Proof Steps:**
1. **Existence:** Because $\text{span}(\mathcal{B}) = V$, every $x \in V$ can be expressed as:
   $$x = \sum_{i=1}^n c_i v_i \quad \text{for some } c_i \in \mathbb{F}$$
2. **Uniqueness (by subtraction):**
   Suppose there exists a second representation of $x$ with scalar coefficients $d_1, \dots, d_n \in \mathbb{F}$:
   $$x = \sum_{i=1}^n d_i v_i$$
   Subtracting the two equations:
   $$x - x = \sum_{i=1}^n c_i v_i - \sum_{i=1}^n d_i v_i$$
   $$\mathbf{0} = \sum_{i=1}^n (c_i - d_i) v_i$$
   Because $\{v_1, \dots, v_n\}$ is linearly independent, the only combination that produces $\mathbf{0}$ has all zero coefficients:
   $$c_i - d_i = 0 \iff c_i = d_i, \quad \forall i \in \{1, 2, \dots, n\}$$
   Thus, both coefficient sets are identical, proving that coordinates are strictly unique. $\blacksquare$

- **Invariance of Dimension:** Every basis of a finite-dimensional vector space $V$ has the exact same number of vectors.
- **Dimension ($\dim(V)$):** The number of vectors in any basis of $V$ is called the **dimension** of $V$.

---

### 7. Change of Basis Matrix & Coordinate Transformation: Full Derivation
Let $\mathcal{B}_1 = \{u_1, \dots, u_n\}$ and $\mathcal{B}_2 = \{v_1, \dots, v_n\}$ be two distinct ordered bases of $V$.
Any vector $x \in V$ can be represented in both bases:
$$x = \sum_{j=1}^n c_j u_j \iff [x]_{\mathcal{B}_1} = \begin{bmatrix} c_1 \\ \vdots \\ c_n \end{bmatrix}, \qquad x = \sum_{i=1}^n d_i v_i \iff [x]_{\mathcal{B}_2} = \begin{bmatrix} d_1 \\ \vdots \\ d_n \end{bmatrix}$$

#### First-Principles Derivation of Transition Matrix $P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1}$:
Since $\mathcal{B}_2$ is a basis for $V$, every basis vector $u_j \in \mathcal{B}_1$ can be expressed uniquely as a linear combination of $\mathcal{B}_2$:
$$u_j = \sum_{i=1}^n p_{ij} v_i \iff [u_j]_{\mathcal{B}_2} = \begin{bmatrix} p_{1j} \\ p_{2j} \\ \vdots \\ p_{nj} \end{bmatrix}$$
Now substitute this expansion into the expression for $x$:
$$x = \sum_{j=1}^n c_j u_j = \sum_{j=1}^n c_j \left( \sum_{i=1}^n p_{ij} v_i \right) = \sum_{i=1}^n \left( \sum_{j=1}^n p_{ij} c_j \right) v_i$$
By the Unique Representation Theorem, the coordinate of $x$ along $v_i$ is uniquely given by $d_i$:
$$d_i = \sum_{j=1}^n p_{ij} c_j = (P)_{i, :} [x]_{\mathcal{B}_1}$$
In matrix form:
$$[x]_{\mathcal{B}_2} = P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1} [x]_{\mathcal{B}_1}$$
where the columns of the transition matrix are precisely the coordinates of the old basis vectors relative to the new basis:
$$P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1} = \begin{bmatrix} [u_1]_{\mathcal{B}_2} & [u_2]_{\mathcal{B}_2} & \cdots & [u_n]_{\mathcal{B}_2} \end{bmatrix}$$
Furthermore, if $\mathcal{E}$ is the standard canonical basis, then:
$$P_{\mathcal{E} \leftarrow \mathcal{B}_1} = \begin{bmatrix} u_1 & \cdots & u_n \end{bmatrix}, \quad P_{\mathcal{E} \leftarrow \mathcal{B}_2} = \begin{bmatrix} v_1 & \cdots & v_n \end{bmatrix}$$
Since $[x]_\mathcal{E} = P_{\mathcal{E} \leftarrow \mathcal{B}_1} [x]_{\mathcal{B}_1} = P_{\mathcal{E} \leftarrow \mathcal{B}_2} [x]_{\mathcal{B}_2}$, multiplying by $(P_{\mathcal{E} \leftarrow \mathcal{B}_2})^{-1}$ gives:
$$P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1} = \left( P_{\mathcal{E} \leftarrow \mathcal{B}_2} \right)^{-1} P_{\mathcal{E} \leftarrow \mathcal{B}_1}$$

## Part 3: Geometric & Algebraic Interpretation

### Geometric Visualization in $\mathbb{R}^2$ and $\mathbb{R}^3$

```
             y-axis
               ^
               |       Line L (Subspace: y = 2x)
               |      /
               |     /   Vector v = [1, 2]^T
               |    *
               |   /
               |  /
---------------+-----------------> x-axis
             O | (Origin [0, 0]^T MUST be included)
               |
               |       Line M (NOT a Subspace: y = 2x + 1)
               |      /   (Fails to pass through origin!)
```

1. **Subspaces in $\mathbb{R}^2$**:
   - The trivial $0$-dimensional subspace: $\{\mathbf{0}\}$.
   - Any $1$-dimensional line **passing directly through the origin** $(0,0)$.
   - The entire space $\mathbb{R}^2$.
2. **Subspaces in $\mathbb{R}^3$**:
   - The origin $\{\mathbf{0}\}$ ($\dim = 0$).
   - Any line passing through the origin ($\dim = 1$).
   - Any plane passing through the origin ($\dim = 2$).
   - The entire space $\mathbb{R}^3$ ($\dim = 3$).

### Why must a subspace pass through the origin?
If a set $W$ does not contain the origin $\mathbf{0}$, it violates Axiom A3 (additive identity).
Geometrically, if you pick a vector $v$ on the line $y = 2x + 1$ and scale it by scalar $c = 0$, the result is $0 \cdot v = \mathbf{0} = [0, 0]^T$, which lies outside the line! Thus, it fails closure under scalar multiplication.

### The Span as "Sweeping Out Dimensions"
- $1$ non-zero vector spans a **line** (1D).
- $2$ linearly independent vectors span a **plane** (2D).
- $3$ linearly independent vectors span a **volume** (3D).
- If the 3rd vector is a linear combination of the first two, it is coplanar; adding it **sweeps out zero new dimensions**, and the span remains 2D.

---

## Part 4: Real-World Analogy

### Analogy 1: The RGB Color Gamut
Think of standard computer display colors:
- Red $(R)$, Green $(G)$, and Blue $(B)$ are the primary color channels.
- Any displayable color is a combination: $C = c_1 \mathbf{e}_R + c_2 \mathbf{e}_G + c_3 \mathbf{e}_B$.
- $\{R, G, B\}$ forms a **basis** for the color space. They are linearly independent: you cannot create pure Red by mixing Green and Blue.
- If an artist adds "Yellow" as a fourth base tube where $\text{Yellow} = \frac{1}{2}\text{Red} + \frac{1}{2}\text{Green}$, the set $\{\text{Red}, \text{Green}, \text{Blue}, \text{Yellow}\}$ is **linearly dependent**; Yellow adds zero new color dimensions.

### Analogy 2: Word Embeddings & Semantic Concepts
In embedding models (e.g., Word2Vec, GloVe, transformer token embeddings):
- Semantic dimensions like "tense", "gender", or "royalty" act like basis directions.
- $\vec{v}_{\text{queen}} \approx \vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}}$.
- If a model's embedding matrix has collapsed onto a lower-dimensional subspace, distinct tokens become linearly dependent linear combinations, causing **semantic representation collapse**.

---

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### LoRA (Low-Rank Adaptation) Subspace Restriction by Hand

Let us trace a concrete Low-Rank Adaptation (LoRA) layer completely by hand with visual matrix grids, cell by cell, without skipping any arithmetic.

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Mathematical Domain | Concrete Shape in Demo | Mathematical Meaning | Deep Learning Semantic Role | Concrete Demo Value |
| :---: | :---: | :---: | :--- | :--- | :---: |
| $x$ | $x \in \mathbb{R}^{d_{\text{in}}}$ | Column vector $[2 \times 1]$ | Input vector in ambient input space | Input token embedding / layer activations | $\begin{bmatrix} 2 \\ 1 \end{bmatrix}$ |
| $W_0$ | $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ | Matrix $[3 \times 2]$ | Linear operator from $\mathbb{R}^2 \to \mathbb{R}^3$ | Pre-trained frozen layer weights | $\begin{bmatrix} 1 & 2 \\ 0 & 1 \\ 3 & -1 \end{bmatrix}$ |
| $r$ | $r \in \mathbb{N}^+$ | Scalar | Dimension of the adapter subspace | Bottleneck rank ($r \ll \min(d_{\text{in}}, d_{\text{out}})$) | $1$ |
| $A$ | $A \in \mathbb{R}^{r \times d_{\text{in}}}$ | Row vector $[1 \times 2]$ | Down-projection onto $r$-dimensional space | Trainable down-projector (encoder adapter) | $\begin{bmatrix} 1 & -1 \end{bmatrix}$ |
| $h$ | $h \in \mathbb{R}^r$ | Scalar $[1 \times 1]$ | Coordinate along the $r$ adapter axes | Bottleneck latent activation | $[1]$ |
| $B$ | $B \in \mathbb{R}^{d_{\text{out}} \times r}$ | Column vector $[3 \times 1]$ | Basis vector spanning adapter subspace | Trainable up-projector (decoder adapter) | $\begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}$ |
| $\Delta y$ | $\Delta y \in \mathbb{R}^{d_{\text{out}}}$ | Column vector $[3 \times 1]$ | Vector restricted to $\text{span}(B)$ | Output update contributed by LoRA adapter | $\begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}$ |
| $y_{\text{base}}$ | $y_{\text{base}} \in \mathbb{R}^{d_{\text{out}}}$ | Column vector $[3 \times 1]$ | Unmodified linear transform $W_0 x$ | Pre-trained model unadapted output | $\begin{bmatrix} 4 \\ 1 \\ 5 \end{bmatrix}$ |
| $y$ | $y \in \mathbb{R}^{d_{\text{out}}}$ | Column vector $[3 \times 1]$ | Total sum vector $y_{\text{base}} + \Delta y$ | Final adapted layer output | $\begin{bmatrix} 6 \\ 2 \\ 9 \end{bmatrix}$ |
| $\frac{\alpha}{r}$ | $\mathbb{R}$ | Scalar | Homothety / scaling scalar | LoRA hyperparameter scaling factor | $1.0$ |

---

#### 2. Visual Architecture Dataflow Grid

```
                   INPUT x (2 x 1)
                   ┌───────────────┐
                   │  x_1 = 2      │  <-- Feature 1 (e.g. token frequency)
                   ├───────────────┤
                   │  x_2 = 1      │  <-- Feature 2 (e.g. sentiment score)
                   └───────────────┘
                     /           \
                    /             \
                   v               v
        BASE WEIGHT W_0           LORA DOWN A
         (3 x 2)                   (1 x 2)
        ┌──────────┬──────────┐   ┌──────────┬──────────┐
  Row 1 │ W_11 = 1 │ W_12 = 2 │   │ A_11 = 1 │ A_12 =-1 │
        ├──────────┼──────────┤   └──────────┴──────────┘
  Row 2 │ W_21 = 0 │ W_22 = 1 │              │
        ├──────────┼──────────┤              │ [Matrix-Vector Multiply]
  Row 3 │ W_31 = 3 │ W_32 =-1 │              v
        └──────────┴──────────┘     INTERMEDIATE h
              │                         (1 x 1)
              │                     ┌──────────┐
              v                     │  h = 1   │  <-- 2D collapsed to 1D scalar
         BASE OUTPUT                └──────────┘
        y_base (3 x 1)                       │
        ┌──────────────┐                     v
        │ y_base_1 = 4 │            LORA UP B (3 x 1)
        ├──────────────┤            ┌──────────┐
        │ y_base_2 = 1 │      Row 1 │ B_11 = 2 │
        ├──────────────┤            ├──────────┤
        │ y_base_3 = 5 │      Row 2 │ B_21 = 1 │
        └──────────────┘            ├──────────┤
                                    │ B_31 = 4 │
                                    └──────────┘
                                             │
                                             │ [Scalar-Vector Multiply]
                                             v
                                        LORA DELTA
                                        Δy (3 x 1)
                                    ┌──────────┐
                                    │  Δy_1 = 2│
                                    ├──────────┤
                                    │  Δy_2 = 1│
                                    ├──────────┤
                                    │  Δy_3 = 4│
                                    └──────────┘
              \                              /
               \                            /
                v                          v
                 TOTAL OUTPUT y = y_base + Δy (3 x 1)
                 ┌──────────────────────────────────┐
                 │ y_1 = 4 + 2 = 6                  │
                 ├──────────────────────────────────┤
                 │ y_2 = 1 + 1 = 2                  │
                 ├──────────────────────────────────┤
                 │ y_3 = 5 + 4 = 9                  │
                 └──────────────────────────────────┘
```

---

#### 3. Base Forward Pass Calculation ($y_{\text{base}} = W_0 x$)

We compute $W_0 x$ using **both** classic textbook perspectives:

##### A. Gilbert Strang's Column Picture (Linear Combination of Columns of $W_0$)
$$y_{\text{base}} = x_1 \cdot (\text{Column 1 of } W_0) + x_2 \cdot (\text{Column 2 of } W_0)$$
$$y_{\text{base}} = 2 \cdot \begin{bmatrix} 1 \\ 0 \\ 3 \end{bmatrix} + 1 \cdot \begin{bmatrix} 2 \\ 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \times 1 \\ 2 \times 0 \\ 2 \times 3 \end{bmatrix} + \begin{bmatrix} 1 \times 2 \\ 1 \times 1 \\ 1 \times (-1) \end{bmatrix} = \begin{bmatrix} 2 + 2 \\ 0 + 1 \\ 6 - 1 \end{bmatrix} = \begin{bmatrix} 4 \\ 1 \\ 5 \end{bmatrix}$$

##### B. Row Picture (Dot Product of Each Row with $x$)
$$\begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix}_{\text{base}} = \begin{bmatrix} 1 & 2 \\ 0 & 1 \\ 3 & -1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

| Output Cell | Operands Involved | Exact Row $\times$ Column Arithmetic | Cell Result |
| :---: | :--- | :--- | :---: |
| **Row 1 ($y_1$)** | Row 1: $[1, 2]$, Vector: $[2, 1]^T$ | $(1 \times 2) + (2 \times 1) = 2 + 2$ | **$4$** |
| **Row 2 ($y_2$)** | Row 2: $[0, 1]$, Vector: $[2, 1]^T$ | $(0 \times 2) + (1 \times 1) = 0 + 1$ | **$1$** |
| **Row 3 ($y_3$)** | Row 3: $[3, -1]$, Vector: $[2, 1]^T$ | $(3 \times 2) + (-1 \times 1) = 6 - 1$ | **$5$** |

$$y_{\text{base}} = \begin{bmatrix} 4 \\ 1 \\ 5 \end{bmatrix}$$

---

#### 4. LoRA Down-Projection ($h = A x$)
The input vector $x \in \mathbb{R}^2$ is compressed down into the $r=1$ dimensional subspace:

$$h = \begin{bmatrix} 1 & -1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

| Intermediate Cell | Operands Involved | Exact Arithmetic | Result |
| :---: | :--- | :--- | :---: |
| **Bottleneck $h$ ($1 \times 1$)** | Row: $[1, -1]$, Col: $[2, 1]^T$ | $(1 \times 2) + (-1 \times 1) = 2 - 1$ | **$1$** |

*Pedagogical Insight:* Matrix $A$ has compressed a 2-dimensional continuous input down to a single coordinate scalar $h = 1$ along the projection axis $[1, -1]$.

---

#### 5. LoRA Up-Projection ($\Delta y = B h$)
The scalar bottleneck coordinate $h=1$ scales the basis vector $B \in \mathbb{R}^{3 \times 1}$:

$$\Delta y = \begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix} \cdot (1) = \begin{bmatrix} 2 \times 1 \\ 1 \times 1 \\ 4 \times 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}$$

| Dimension | Operands Involved | Exact Scaling Arithmetic | LoRA Delta Output |
| :---: | :--- | :--- | :---: |
| **Row 1 ($\Delta y_1$)** | Basis coordinate $B_{11} = 2$, scalar $h = 1$ | $2 \times 1$ | **$2$** |
| **Row 2 ($\Delta y_2$)** | Basis coordinate $B_{21} = 1$, scalar $h = 1$ | $1 \times 1$ | **$1$** |
| **Row 3 ($\Delta y_3$)** | Basis coordinate $B_{31} = 4$, scalar $h = 1$ | $4 \times 1$ | **$4$** |

---

#### 6. Total Forward Output ($y = y_{\text{base}} + \Delta y$)

| Dimension | Pre-trained Base Output | + | LoRA Residual Delta | = | Final Adapted Output |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Row 1 ($y_1$)** | $4$ | $+$ | $2$ | $=$ | **$6$** |
| **Row 2 ($y_2$)** | $1$ | $+$ | $1$ | $=$ | **$2$** |
| **Row 3 ($y_3$)** | $5$ | $+$ | $4$ | $=$ | **$9$** |

$$y = \begin{bmatrix} 6 \\ 2 \\ 9 \end{bmatrix}$$

---

#### 7. The Gilbert Strang Subspace Proof: Why $\Delta y$ Can Never Leave a 1D Line
Let us test an entirely different input vector $x' = \begin{bmatrix} 5 \\ 2 \end{bmatrix}$ through this exact layer:
1. **Down-projection**:
   $$h' = A x' = \begin{bmatrix} 1 & -1 \end{bmatrix} \begin{bmatrix} 5 \\ 2 \end{bmatrix} = (1 \times 5) + (-1 \times 2) = 5 - 2 = 3$$
2. **Up-projection**:
   $$\Delta y' = B h' = \begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix} \cdot (3) = \begin{bmatrix} 6 \\ 3 \\ 12 \end{bmatrix} = 3 \cdot \begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}$$

> [!IMPORTANT]
> **Look at the relationship between $\Delta y$ and $\Delta y'$**:
> $$\Delta y = 1 \cdot \begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}, \quad \Delta y' = 3 \cdot \begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}, \quad \Delta y_{\text{arbitrary}} = (x_1 - x_2) \cdot \begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}$$
> In Gilbert Strang's terminology:
> The column space of the LoRA update matrix is:
> $$C(\Delta W) = C(B A) = C(B) = \text{span}\left(\begin{bmatrix} 2 \\ 1 \\ 4 \end{bmatrix}\right) \subset \mathbb{R}^3$$
> **Conclusion:** No matter what input token or activation vector $x \in \mathbb{R}^2$ passes through the network, the LoRA adaptation $\Delta y$ is **permanently imprisoned on the 1-dimensional line (subspace)** defined by vector $B$. It is mathematically incapable of perturbing activations outside this 1D subspace!

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Standard Case — Checking Linear Independence & Finding a Basis

**Problem:** Given three vectors in $\mathbb{R}^3$:
$$v_1 = \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 1 \\ 1 \\ -2 \end{bmatrix}$$
1. Determine whether $\{v_1, v_2, v_3\}$ is linearly independent.
2. Do they form a basis for $\mathbb{R}^3$?
3. If they form a basis, find the coordinates $[b]_\mathcal{B}$ of the target vector $b = \begin{bmatrix} 5 \\ 3 \\ 0 \end{bmatrix}$.

#### "What Refers to What" — Variable & Symbol Dictionary
| Symbol | Mathematical Domain | Description | Value / Meaning |
| :---: | :---: | :--- | :---: |
| $v_1, v_2, v_3$ | $\mathbb{R}^3$ | Given feature vectors | Columns of matrix $A$ |
| $c_1, c_2, c_3$ | $\mathbb{R}$ | Linear combination coefficients (scalars) | Unknown coordinates to be solved |
| $A$ | $\mathbb{R}^{3 \times 3}$ | Matrix formed by setting vectors as columns | $A = [v_1 \mid v_2 \mid v_3]$ |
| $\mathbf{0}$ | $\mathbb{R}^3$ | Zero vector in ambient space $\mathbb{R}^3$ | Target vector for independence test $[0, 0, 0]^T$ |
| $R_1, R_2, R_3$ | Rows of matrix | Echelon row vectors during elimination | Elementary row operation targets |

---

#### Step 1: Set up the Vector & Matrix Equation
By definition, linear independence asks whether any non-trivial linear combination can reach $\mathbf{0}$:
$$c_1 v_1 + c_2 v_2 + c_3 v_3 = \mathbf{0}$$
$$c_1 \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix} + c_2 \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix} + c_3 \begin{bmatrix} 1 \\ 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

Using Gilbert Strang's column picture, this vector equation is identical to the matrix system $A c = \mathbf{0}$:
$$A = \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 1 & 1 \\ 2 & 0 & -2 \end{bmatrix}$$

---

#### Step 2: Gaussian Elimination to Row Echelon Form (With Explicit Explanations)
We form the augmented matrix $[A \mid \mathbf{0}]$:
$$\begin{bmatrix} 1 & 2 & 1 & | & 0 \\ 0 & 1 & 1 & | & 0 \\ 2 & 0 & -2 & | & 0 \end{bmatrix}$$

- **Row Operation 1 ($R_3 \leftarrow R_3 - 2R_1$):**
  - *Goal:* Create a zero in column 1, row 3 underneath the pivot $A_{11} = 1$.
  - *Arithmetic:*
    - Col 1: $2 - 2(1) = 0$
    - Col 2: $0 - 2(2) = -4$
    - Col 3: $-2 - 2(1) = -4$
    - Target: $0 - 2(0) = 0$
  $$\begin{bmatrix} 1 & 2 & 1 & | & 0 \\ 0 & 1 & 1 & | & 0 \\ 0 & -4 & -4 & | & 0 \end{bmatrix}$$

- **Row Operation 2 ($R_3 \leftarrow R_3 + 4R_2$):**
  - *Goal:* Create a zero in column 2, row 3 underneath the second pivot $A_{22} = 1$.
  - *Arithmetic:*
    - Col 1: $0 + 4(0) = 0$
    - Col 2: $-4 + 4(1) = 0$
    - Col 3: $-4 + 4(1) = 0$
    - Target: $0 + 4(0) = 0$
  $$\begin{bmatrix} 1 & 2 & 1 & | & 0 \\ 0 & 1 & 1 & | & 0 \\ 0 & 0 & 0 & | & 0 \end{bmatrix}$$

- **Row Operation 3 (Gauss-Jordan to RREF: $R_1 \leftarrow R_1 - 2R_2$):**
  - *Goal:* Clear entries above the pivot in column 2 to get Reduced Row Echelon Form (RREF).
  - *Arithmetic:*
    - Col 1: $1 - 2(0) = 1$
    - Col 2: $2 - 2(1) = 0$
    - Col 3: $1 - 2(1) = -1$
  $$\text{RREF}(A) = \begin{bmatrix} 1 & 0 & -1 & | & 0 \\ 0 & 1 & 1 & | & 0 \\ 0 & 0 & 0 & | & 0 \end{bmatrix}$$

---

#### Step 3: Gilbert Strang 4-Subspace & Nullspace Analysis
1. **Pivots & Rank:**
   - Leading 1s appear in **Column 1** and **Column 2**.
   - **Column 3 contains NO pivot** $\implies c_3$ is a **free variable**.
   - The matrix rank is $r = 2$.
2. **Finding the Nullspace $N(A)$:**
   Set the free variable $c_3 = t$ (where $t \in \mathbb{R}$ is any non-zero parameter, say $t = 1$):
   - From Row 2 of RREF: $c_2 + c_3 = 0 \implies c_2 = -c_3 = -t = -1$.
   - From Row 1 of RREF: $c_1 - c_3 = 0 \implies c_1 = c_3 = t = 1$.
   - The nullspace vector is:
     $$c = \begin{bmatrix} c_1 \\ c_2 \\ c_3 \end{bmatrix} = t \begin{bmatrix} 1 \\ -1 \\ 1 \end{bmatrix}$$
3. **Physical Meaning of the Nullspace Vector:**
   Since $c = [1, -1, 1]^T \in N(A)$, multiplying by $A$ yields zero:
   $$1 \cdot v_1 - 1 \cdot v_2 + 1 \cdot v_3 = \mathbf{0} \implies v_3 = -v_1 + v_2$$
   Let's check the arithmetic directly:
   $$- \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix} + \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -1 + 2 \\ 0 + 1 \\ -2 + 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ -2 \end{bmatrix} = v_3 \quad \checkmark$$
   *Conclusion:* Vector $v_3$ is an exact linear combination of $v_1$ and $v_2$. It adds **zero new directional information**.
4. **Rank-Nullity Theorem Verification:**
   $$\dim(C(A)) + \dim(N(A)) = n \implies 2 + 1 = 3$$
5. **Final Answers to Questions:**
   - Are $\{v_1, v_2, v_3\}$ linearly independent? **No, they are Linearly Dependent.**
   - Do they form a basis for $\mathbb{R}^3$? **No**, because any basis of $\mathbb{R}^3$ must contain 3 linearly independent vectors ($\dim(\mathbb{R}^3) = 3$).
   - Can we find coordinates $[b]_\mathcal{B}$ for $b = [5, 3, 0]^T$? **Not as a basis for $\mathbb{R}^3$**, because $b$ does not lie on the plane $\text{span}(v_1, v_2)$.

---

### Scenario B: Boundary Case — Extracting a Maximal Linearly Independent Basis

**Problem:** What is the basis for the subspace $W = \text{span}(v_1, v_2, v_3)$, and does $b = [5, 3, 0]^T$ lie inside $W$?

1. **Extracting the Basis:**
   According to the Basis Theorem, the **original columns corresponding to the pivot columns in RREF** form a basis for the column space $C(A) = W$.
   - Pivot columns were Column 1 and Column 2.
   - Therefore, a basis for $W$ is:
     $$\mathcal{B}_W = \left\{ \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}, \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix} \right\}$$
   - $\dim(W) = 2$ (a flat 2-dimensional plane passing through $[0, 0, 0]^T$ embedded in 3D space).
2. **Testing Target Vector Membership ($b \in W$):**
   Can we find scalars $\alpha_1, \alpha_2$ such that $\alpha_1 v_1 + \alpha_2 v_2 = b$?
   $$\alpha_1 \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix} + \alpha_2 \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 5 \\ 3 \\ 0 \end{bmatrix}$$
   System of linear equations:
   $$\begin{cases} 1\alpha_1 + 2\alpha_2 = 5 & \text{(Equation 1)} \\ 0\alpha_1 + 1\alpha_2 = 3 & \text{(Equation 2)} \\ 2\alpha_1 + 0\alpha_2 = 0 & \text{(Equation 3)} \end{cases}$$
   - From Equation 2: $\alpha_2 = 3$.
   - From Equation 3: $2\alpha_1 = 0 \implies \alpha_1 = 0$.
   - Now substitute $\alpha_1 = 0, \alpha_2 = 3$ into Equation 1:
     $$1(0) + 2(3) = 6 \neq 5$$
   - Contradiction! The system has **no solution**.
   - **Conclusion:** $b = [5, 3, 0]^T \notin W$. The target point $b$ floats outside the 2D plane spanned by our vectors.

---

### Scenario C: Pathological Edge Cases

| Scenario | Mathematical Example | Why it is Dependent / Edge Case | Deep Learning Implication |
| :--- | :--- | :--- | :--- |
| **Set containing $\mathbf{0}$** | $S = \{\mathbf{0}, v_2\}$ | $1 \cdot \mathbf{0} + 0 \cdot v_2 = \mathbf{0}$ has non-trivial coefficient $c_1 = 1 \neq 0$. | A "dead" feature column of all zeros is strictly redundant. |
| **More vectors than dimension ($k > n$)** | $4$ vectors in $\mathbb{R}^3$ | Matrix has shape $3 \times 4$. Max rank is 3. Free variables $\ge 4 - 3 = 1$. | Overcomplete dictionary / representations are guaranteed linearly dependent. |
| **Collinear vectors** | $v_1 = [1, 2]^T, v_2 = [3, 6]^T$ | $v_2 = 3 v_1 \implies 3v_1 - v_2 = \mathbf{0}$. | Two perfectly correlated feature channels / duplicate neurons. |

---

### Scenario D: Subspace Verification (Rigorous Algebraic Proof)

**Test:** Compare two sets in $\mathbb{R}^3$:
- $W_1 = \left\{ x \in \mathbb{R}^3 \mid 3x_1 - 2x_2 + x_3 = 0 \right\}$
- $W_2 = \left\{ x \in \mathbb{R}^3 \mid 3x_1 - 2x_2 + x_3 = 4 \right\}$

#### Proof for $W_1$:
1. **Zero Vector:** Check if $\mathbf{0} = [0, 0, 0]^T \in W_1$:
   $$3(0) - 2(0) + (0) = 0 - 0 + 0 = 0 \implies \mathbf{0} \in W_1 \quad \checkmark$$
2. **Closure under Linear Combinations:**
   Let $u, v \in W_1$. That means:
   $$3u_1 - 2u_2 + u_3 = 0 \quad \text{and} \quad 3v_1 - 2v_2 + v_3 = 0$$
   Let $a, b \in \mathbb{R}$. Form the linear combination $w = a u + b v = \begin{bmatrix} a u_1 + b v_1 \\ a u_2 + b v_2 \\ a u_3 + b v_3 \end{bmatrix}$.
   Evaluate the defining linear equation on $w$:
   $$\begin{aligned} 3w_1 - 2w_2 + w_3 &= 3(a u_1 + b v_1) - 2(a u_2 + b v_2) + (a u_3 + b v_3) \\ &= a(3u_1 - 2u_2 + u_3) + b(3v_1 - 2v_2 + v_3) \quad \text{(by regrouping terms)} \\ &= a(0) + b(0) \quad \text{(substituting membership conditions of } u \text{ and } v) \\ &= 0 \end{aligned}$$
   Since the equation equals $0$, $w \in W_1$.
   **Conclusion:** $W_1$ is a genuine $2$-dimensional subspace of $\mathbb{R}^3$.

#### Proof of Failure for $W_2$:
1. **Zero Vector Check:**
   $$3(0) - 2(0) + (0) = 0 \neq 4 \implies \mathbf{0} \notin W_2$$
   Since the additive identity $\mathbf{0}$ is absent, $W_2$ immediately fails the first axiom.
2. **Scalar Multiplication Failure:**
   Even if we pick a vector on $W_2$, say $u = [1, 0, 1]^T$ ($3(1) - 2(0) + 1 = 4 \implies u \in W_2$), scaling by scalar $c = 2$ gives $2u = [2, 0, 2]^T$:
   $$3(2) - 2(0) + 2 = 6 - 0 + 2 = 8 \neq 4 \implies 2u \notin W_2$$
   **Conclusion:** $W_2$ is an **affine plane**, NOT a vector subspace. In deep learning, unconstrained linear layers without bias are subspaces; adding a bias vector $b \neq \mathbf{0}$ shifts the subspace into an affine space.

---

### Scenario E: Change of Basis Coordinate Transformation (Full Numerical Problem with Inversion)

**Problem Formulation:**
Consider the 2D vector space $\mathbb{R}^2$ with standard canonical basis $\mathcal{E} = \{e_1, e_2\} = \left\{ \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix} \right\}$.
Suppose we are given two custom feature bases:
- **Basis 1 ($\mathcal{B}_1$):** $u_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad u_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$ (e.g., diagonal and anti-diagonal feature axes)
- **Basis 2 ($\mathcal{B}_2$):** $v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ (e.g., non-orthogonal latent representations)

A token activation vector has coordinates relative to $\mathcal{B}_1$:
$$[x]_{\mathcal{B}_1} = \begin{bmatrix} 3 \\ -2 \end{bmatrix}$$

**Tasks:**
1. Determine the canonical coordinate vector $x \in \mathbb{R}^2$.
2. Derive and compute the transition matrix $P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1}$.
3. Compute the transformed coordinates $[x]_{\mathcal{B}_2}$.
4. Verify by expanding in $\mathcal{B}_2$ that the exact same canonical vector is recovered.

#### Step 1: Compute Canonical Representation $x = [x]_\mathcal{E}$
By definition of coordinate representation with respect to $\mathcal{B}_1$:
$$x = 3 u_1 + (-2) u_2 = 3 \begin{bmatrix} 1 \\ 1 \end{bmatrix} - 2 \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$
$$x = \begin{bmatrix} 3(1) - 2(1) \\ 3(1) - 2(-1) \end{bmatrix} = \begin{bmatrix} 3 - 2 \\ 3 + 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 5 \end{bmatrix}$$

#### Step 2: Form Canonical Transition Matrices and Invert $P_{\mathcal{E} \leftarrow \mathcal{B}_2}$
The matrix converting $\mathcal{B}_1$ coordinates to standard coordinates is:
$$P_{\mathcal{E} \leftarrow \mathcal{B}_1} = \begin{bmatrix} u_1 & u_2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$
The matrix converting $\mathcal{B}_2$ coordinates to standard coordinates is:
$$P_{\mathcal{E} \leftarrow \mathcal{B}_2} = \begin{bmatrix} v_1 & v_2 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}$$

To convert from standard coordinates back to $\mathcal{B}_2$, we need $(P_{\mathcal{E} \leftarrow \mathcal{B}_2})^{-1}$.
For a $2 \times 2$ matrix $M = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:
$$M^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$
Here:
- $\det(P_{\mathcal{E} \leftarrow \mathcal{B}_2}) = (2)(1) - (1)(1) = 2 - 1 = 1 \neq 0$ (confirming $\mathcal{B}_2$ is a valid basis).
- The inverse matrix is:
  $$(P_{\mathcal{E} \leftarrow \mathcal{B}_2})^{-1} = \frac{1}{1} \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}$$

#### Step 3: Compute Transition Matrix $P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1}$
Using our derived formula:
$$P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1} = (P_{\mathcal{E} \leftarrow \mathcal{B}_2})^{-1} P_{\mathcal{E} \leftarrow \mathcal{B}_1} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

Let us compute each entry explicitly:
- Row 1, Col 1: $(1)(1) + (-1)(1) = 1 - 1 = 0$
- Row 1, Col 2: $(1)(1) + (-1)(-1) = 1 + 1 = 2$
- Row 2, Col 1: $(-1)(1) + (2)(1) = -1 + 2 = 1$
- Row 2, Col 2: $(-1)(1) + (2)(-1) = -1 - 2 = -3$

$$P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1} = \begin{bmatrix} 0 & 2 \\ 1 & -3 \end{bmatrix}$$

#### Step 4: Calculate Coordinates $[x]_{\mathcal{B}_2}$
Multiply the transition matrix by $[x]_{\mathcal{B}_1}$:
$$[x]_{\mathcal{B}_2} = P_{\mathcal{B}_2 \leftarrow \mathcal{B}_1} [x]_{\mathcal{B}_1} = \begin{bmatrix} 0 & 2 \\ 1 & -3 \end{bmatrix} \begin{bmatrix} 3 \\ -2 \end{bmatrix}$$
- First coordinate: $(0)(3) + (2)(-2) = 0 - 4 = -4$
- Second coordinate: $(1)(3) + (-3)(-2) = 3 + 6 = 9$

$$[x]_{\mathcal{B}_2} = \begin{bmatrix} -4 \\ 9 \end{bmatrix}$$

#### Step 5: Verification of Geometric Equivalence
Let us reconstruct the vector in standard coordinates using the newly found coordinates $[-4, 9]^T$ along basis $\mathcal{B}_2$:
$$x = (-4) v_1 + (9) v_2 = -4 \begin{bmatrix} 2 \\ 1 \end{bmatrix} + 9 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -8 + 9 \\ -4 + 9 \end{bmatrix} = \begin{bmatrix} 1 \\ 5 \end{bmatrix}$$
This matches the original canonical vector $[1, 5]^T$ from Step 1 with mathematical precision. $\checkmark$

---

## Part 7: Deep Learning Connection & Application

### 1. Latent Representations & Embedding Manifolds
In Transformer models (such as GPT-4 or Claude), the hidden states across layers live in vector spaces of dimension $d_{\text{model}}$ (e.g., $d = 4096$).
- Although the ambient vector space is $\mathbb{R}^{4096}$, the **intrinsic dimension** of valid semantic representations is much lower.
- Real-world data concentrates on low-dimensional curved subspaces (the **Manifold Hypothesis**).

### 2. Low-Rank Adaptation (LoRA)
When fine-tuning billion-parameter LLMs, updating the full weight matrix $W_0 \in \mathbb{R}^{d \times k}$ is computationally prohibitive.
- LoRA (Hu et al., 2021) hypothesizes that weight updates $\Delta W$ have a very low **intrinsic rank** $r \ll \min(d, k)$:
  $$\Delta W = B \cdot A, \quad \text{where } B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}$$
- The column space of $\Delta W$ is constrained to lie strictly within the **subspace spanned by the columns of $B$**:
  $$\text{col}(\Delta W) \subseteq \text{col}(B)$$
- Instead of learning in a $d \times k$ dimensional parameter space, training is projected onto an $r$-dimensional subspace (e.g., $r = 8$ or $16$), reducing trainable parameters by $99.9\%$.

```
Full Parameter Space R^(d x k)
     ┌──────────────────────────────────────────────┐
     │                                              │
     │     Subspace spanned by LoRA (Rank r)        │
     │         ┌──────────────────────┐             │
     │         │ ΔW = B · A           │             │
     │         │ Dim = r * (d + k)    │             │
     │         └──────────────────────┘             │
     │                                              │
     └──────────────────────────────────────────────┘
```

### 3. Bottlenecks in Autoencoders
An Autoencoder compresses input $x \in \mathbb{R}^D$ to bottleneck $z \in \mathbb{R}^d$ ($d \ll D$).
- If the encoder and decoder are purely linear layers, the autoencoder learns to project $x$ onto the $d$-dimensional subspace spanned by the top $d$ principal components (identical to PCA).
- Non-linear activations allow bending this subspace into non-linear manifolds.

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/01_vector_spaces_and_subspaces.py`](./code/01_vector_spaces_and_subspaces.py) implements:
1. **Subspace Closure Checker**: Monte Carlo empirical validation of additive and scalar closure.
2. **Linear Independence & Rank Solver**: Computing pivot columns and basis vectors using Reduced Row Echelon Form (RREF) and Singular Value Decomposition (SVD).
3. **Coordinate Solver**: Solving for coordinates relative to arbitrary basis sets.
4. **LoRA Subspace Projection Demo**: Showing how low-rank factorized matrices constrain forward passes to a low-dimensional subspace.

---

## Chapter 1.1 Summary & Review Checklist

- [x] Can you state the 8 vector space axioms from memory?
- [x] Why does any hyperplane $a_1 x_1 + \dots + a_n x_n = c$ fail to be a subspace if $c \neq 0$?
- [x] If a set of $k$ vectors in $\mathbb{R}^n$ has $k > n$, why is it guaranteed to be linearly dependent?
- [x] How does LoRA use subspace theory to fine-tune 70B parameter models efficiently?
