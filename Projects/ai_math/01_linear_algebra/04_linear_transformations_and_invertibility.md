# Chapter 1.4: Linear Transformations & Invertibility

---

## Part 1: Intuition & 101 Motivation

In deep learning, every feedforward layer without activation is a **linear transformation**:
$$y = T(x) = W x$$
When an activation function $\sigma(\cdot)$ is applied, the network becomes a composition of alternating linear and non-linear transformations:
$$f(x) = \sigma(W_L \dots \sigma(W_2 \sigma(W_1 x + b_1) + b_2) \dots + b_L)$$

Understanding the linear transformation $T(x) = W x$ answers the deepest structural questions in representation learning:
1. **Geometric Deformation**: Does the layer rotate the feature space, stretch it along specific directions, or flatten it into a pancake?
2. **Information Loss vs. Invertibility**: Can the original input $x$ be reconstructed from the hidden representation $y$? If a layer squashes a dimension to zero, information is permanently destroyed—no decoder can ever recover it.
3. **Volume Scaling & Probability Density**: In **Generative Normalizing Flows** (such as RealNVP, Glow, and continuous normalizing flows), neural networks generate images and audio by transforming simple Gaussian noise into complex data distributions. To compute exact likelihoods, we must calculate how the transformation stretches or compresses space. The exact measure of this volume expansion is the **Determinant**.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Definition of a Linear Transformation
Let $V$ and $W$ be vector spaces over the field $\mathbb{R}$. A mapping $T: V \to W$ is called a **Linear Transformation** (or linear operator) if it satisfies two fundamental axioms for all $u, v \in V$ and all scalars $\alpha \in \mathbb{R}$:

| Axiom | Name | Mathematical Statement |
| :--- | :--- | :--- |
| **LT1** | **Additivity** | $T(u + v) = T(u) + T(v)$ |
| **LT2** | **Homogeneity (Scalar Scaling)** | $T(\alpha u) = \alpha T(u)$ |

Combined into a single statement:
$$T(\alpha u + \beta v) = \alpha T(u) + \beta T(v), \quad \forall u, v \in V, \; \alpha, \beta \in \mathbb{R}$$

> [!NOTE]
> **Zero Preservation:** Every linear transformation must map the zero vector to the zero vector:
> $$T(\mathbf{0}_V) = T(0 \cdot v) = 0 \cdot T(v) = \mathbf{0}_W$$
> If a layer has a non-zero bias $b \neq \mathbf{0}$, $T(x) = W x + b$ is an **Affine Transformation**, not a purely linear one.

---

### 2. The Matrix Representation Theorem
**Theorem:** Let $V = \mathbb{R}^n$ and $W = \mathbb{R}^m$, and let $\{e_1, e_2, \dots, e_n\}$ be the standard basis of $\mathbb{R}^n$.
Any linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ is completely determined by what it does to the standard basis vectors:
$$A = \begin{bmatrix} T(e_1) & T(e_2) & \dots & T(e_n) \end{bmatrix} \in \mathbb{R}^{m \times n}$$
For every $x \in \mathbb{R}^n$:
$$T(x) = A x$$

---

### 3. Kernel and Image
For a linear transformation $T: V \to W$ with matrix $A$:
1. **The Kernel (Nullspace)**:
   $$\ker(T) = \{ x \in V \mid T(x) = \mathbf{0}_W \} = N(A) \subseteq V$$
2. **The Image (Column Space)**:
   $$\text{im}(T) = \{ T(x) \mid x \in V \} = C(A) \subseteq W$$
3. **The Dimension Theorem (Rank-Nullity)**:
   $$\dim(V) = \dim(\ker(T)) + \dim(\text{im}(T))$$

---

### 4. Invertibility & Isomorphisms
A linear transformation $T: V \to W$ is **invertible** if there exists a unique mapping $T^{-1}: W \to V$ such that:
$$T^{-1}(T(v)) = v, \quad \forall v \in V \quad \text{and} \quad T(T^{-1}(w)) = w, \quad \forall w \in W$$

#### Invertibility Conditions for Square Matrix $A \in \mathbb{R}^{n \times n}$:
The following statements are mathematically equivalent:
1. $A$ is invertible ($A^{-1}$ exists such that $A A^{-1} = A^{-1} A = I_n$).
2. $\ker(T) = \{ \mathbf{0} \}$ ($\dim(N(A)) = 0$, trivial nullspace).
3. $\text{rank}(A) = n$ (Full rank).
4. The columns of $A$ form a basis for $\mathbb{R}^n$.
5. The determinant is non-zero: $\det(A) \neq 0$.
6. None of the eigenvalues of $A$ are zero ($\lambda_i \neq 0, \forall i$).

#### Algebraic Properties of Inverses:
$$(A B)^{-1} = B^{-1} A^{-1}, \quad (A^T)^{-1} = (A^{-1})^T, \quad (c A)^{-1} = \frac{1}{c} A^{-1} \; (c \neq 0)$$

---

### 5. Change of Basis & Matrix Similarity
Suppose we have a vector $x$ and a linear transformation $T$. The coordinate representation of $x$ depends on our choice of coordinate frame (basis).

- Let $\mathcal{E}$ be the standard basis, and let $\mathcal{B} = \{v_1, v_2, \dots, v_n\}$ be an alternative basis.
- The **Change of Basis Matrix** (Transition Matrix) $P$ from $\mathcal{B}$ to $\mathcal{E}$ is constructed by placing the basis vectors of $\mathcal{B}$ as columns:
  $$P = \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix}$$
- For any vector $x$:
  $$x = P [x]_\mathcal{B} \iff [x]_\mathcal{B} = P^{-1} x$$
- If a linear operator is represented by matrix $A$ in standard coordinates, its representation $B$ in basis $\mathcal{B}$ is:
  $$B = P^{-1} A P$$

> [!IMPORTANT]
> **Matrix Similarity:** Two matrices $A$ and $B$ are **similar** if there exists an invertible matrix $P$ such that $B = P^{-1} A P$. Similar matrices represent the **exact same linear transformation** viewed under different coordinate systems! They share identical eigenvalues, trace, and determinant.

#### Rigorous First-Principles Derivation: Similarity Invariance of Spectrum, Trace, and Determinant
**Theorem:** If $B = P^{-1} A P$, then:
1. $B$ and $A$ have the exact same characteristic polynomial: $p_B(\lambda) = p_A(\lambda)$.
2. $B$ and $A$ share identical eigenvalues with identical algebraic and geometric multiplicities.
3. $\det(B) = \det(A)$ and $\text{Tr}(B) = \text{Tr}(A)$.

**Proof Steps:**
1. **Characteristic Polynomial:**
   By definition, the characteristic polynomial of $B$ is $p_B(\lambda) = \det(B - \lambda I)$.
   Substitute $B = P^{-1} A P$ and express the identity matrix as $I = P^{-1} I P$:
   $$B - \lambda I = P^{-1} A P - \lambda P^{-1} I P = P^{-1} (A - \lambda I) P$$
   Taking the determinant and applying the multiplicative rule $\det(X Y) = \det(X) \det(Y)$:
   $$\begin{aligned}
   p_B(\lambda) &= \det\left( P^{-1} (A - \lambda I) P \right) \\
   &= \det(P^{-1}) \cdot \det(A - \lambda I) \cdot \det(P) \\
   &= \frac{1}{\det(P)} \cdot \det(A - \lambda I) \cdot \det(P) \quad \text{(since scalars commute)} \\
   &= \det(A - \lambda I) = p_A(\lambda)
   \end{aligned}$$
2. **Eigenvalues:**
   Because $p_B(\lambda) \equiv p_A(\lambda)$ are identical polynomials in $\lambda$, their roots (the eigenvalues) are strictly identical.
3. **Trace and Determinant Invariance:**
   - The determinant is the constant term of the characteristic polynomial (and the product of all eigenvalues $\prod_{i=1}^n \lambda_i$):
     $$\det(B) = \det(P^{-1} A P) = \det(P^{-1})\det(A)\det(P) = \frac{1}{\det(P)}\det(A)\det(P) = \det(A) \quad \blacksquare$$
   - The trace is the coefficient of $(-\lambda)^{n-1}$ in the characteristic polynomial (and the sum of all eigenvalues $\sum_{i=1}^n \lambda_i$).
   - Direct algebraic proof via the cyclic permutation property of trace:
     $$\text{Tr}(B) = \text{Tr}(P^{-1} A P) = \text{Tr}\left( (P^{-1} A) P \right) = \text{Tr}\left( P (P^{-1} A) \right) = \text{Tr}\left( (P P^{-1}) A \right) = \text{Tr}(I A) = \text{Tr}(A) \quad \blacksquare$$

---

### 6. The Determinant: The Signed Volume Scaling Factor
The **determinant** $\det(A)$ (or $|A|$) of a square matrix $A \in \mathbb{R}^{n \times n}$ is the unique alternating multi-linear function of its columns that satisfies $\det(I) = 1$.

```
           2D Volume (Area) Scaling               3D Volume Scaling
                     y                                      z
                     ^                                      ^
                     |    transformed                       |     transformed
                     |    parallelogram                     |     parallelepiped
                     |       +-----+                        |        /----+
                     |      /     /                         |       /    /|
                     |     /     /                          |      +----+ |
                     |    +-----+                           |      |    | +
             +---+   |                                +-+   |      |    |/
             | 1 |   |                                | |   |      +----+
             +---+---+--------> x                     +-+---+---------> x
            Unit square                              Unit cube
            Area = 1.0                              Volume = 1.0
            Transformed Area = |det(A)|             Transformed Volume = |det(A)|
```

#### Fundamental Geometric Theorems of Determinants:
1. **Volume Scaling**: For any region $S \subset \mathbb{R}^n$, the volume of its image under $T(x) = A x$ is:
   $$\text{Volume}(T(S)) = |\det(A)| \cdot \text{Volume}(S)$$
2. **Singularity**: $\det(A) = 0 \iff A$ collapses space into at least one lower dimension (e.g., 3D volume squashed onto a 2D plane or 1D line). The transformation is **irreversible**.
3. **Orientation**:
   - $\det(A) > 0$: Preserves spatial orientation (right-handed remains right-handed).
   - $\det(A) < 0$: Inverts orientation (mirror reflection).
4. **Multiplication Rule & Inverse Rule: First-Principles Derivation**:
   - **Multiplication Rule:** An elementary row matrix $E$ scales volume by $\det(E)$. If $A$ is invertible, $A = E_k \dots E_1 I$, so:
     $$\det(A B) = \det(E_k \dots E_1 B) = \det(E_k) \dots \det(E_1) \det(B) = \det(A) \det(B)$$
     If $A$ is singular, $\text{rank}(A) < n \implies \text{rank}(AB) \le \text{rank}(A) < n$, so $\det(AB) = 0 = 0 \cdot \det(B) = \det(A)\det(B)$.
   - **Inverse Rule:** Since $A A^{-1} = I$, taking the determinant on both sides yields:
     $$\det(A A^{-1}) = \det(I) \implies \det(A) \det(A^{-1}) = 1 \implies \det(A^{-1}) = \frac{1}{\det(A)}$$
5. **Triangular / Diagonal Matrices**:
   If $A$ is upper-triangular, lower-triangular, or diagonal:
   $$\det(A) = \prod_{i=1}^n A_{ii}$$

---

### 7. The Trace: Sum of Diagonal Elements
The **trace** of a square matrix $A \in \mathbb{R}^{n \times n}$ is the sum of its diagonal entries:
$$\text{Tr}(A) = \sum_{i=1}^n A_{ii}$$

#### Essential Properties in Deep Learning:
1. **Linearity**: $\text{Tr}(\alpha A + \beta B) = \alpha \text{Tr}(A) + \beta \text{Tr}(B)$.
2. **Cyclic Property**: For matrices of compatible dimensions:
   $$\text{Tr}(A B C) = \text{Tr}(C A B) = \text{Tr}(B C A)$$
   *(Note: $\text{Tr}(A B C) \neq \text{Tr}(B A C)$ in general; only cyclic permutations preserve the trace).*
3. **Similarity Invariance**:
   $$\text{Tr}(P^{-1} A P) = \text{Tr}(P P^{-1} A) = \text{Tr}(A)$$
4. **The Trace Trick for Inner Products & Norms**:
   $$\|A\|_F^2 = \sum_{i,j} A_{ij}^2 = \text{Tr}(A^T A) = \text{Tr}(A A^T)$$
   For vectors: $x^T x = \text{Tr}(x^T x) = \text{Tr}(x x^T)$.

---

## Part 3: Geometric & Algebraic Interpretation

### Fundamental 2D Transformations & Their Determinants

| Transformation | Matrix $A$ | Geometric Action | Determinant |
| :--- | :---: | :--- | :---: |
| **Uniform Scaling** | $\begin{bmatrix} s & 0 \\ 0 & s \end{bmatrix}$ | Expands/shrinks all directions by $s$ | $s^2$ |
| **Pure Rotation** | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | Rotates coordinate axes by angle $\theta$ | $\cos^2\theta + \sin^2\theta = \mathbf{1.0}$ |
| **Shear (Skews space)** | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ | Shifts $x$ proportionally to $y$ (preserves area) | $1 \times 1 - 0 = \mathbf{1.0}$ |
| **Reflection** | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ | Flips sign of $y$-axis | $\mathbf{-1.0}$ |
| **Projection (Collapse)**| $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ | Squashes 2D plane onto the 1D $x$-axis | $\mathbf{0.0}$ (Irreversible!) |

---

## Part 4: Real-World Analogy

### The Currency Exchange & Inflation Analogy
- **Change of Basis**: Converting a price tag from USD to EUR to JPY. The underlying economic value of the car or laptop is unchanged; only the numbers representing it change based on the currency basis you choose.
- **Similar Matrices**: If you describe a market's economic growth using US dollars ($A$) versus Euros ($B$), $B = P^{-1} A P$. The market's intrinsic growth rate (eigenvalues) and total wealth flow (trace) remain identical regardless of currency!
- **Determinant as Volume Inflation**: If a central bank devalues currency such that prices quadruple in all dimensions, $\det(A) = 4^D$. A determinant of zero means the currency collapsed to zero worth—everything is wiped out.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### Invertible Layer & Change of Variables in Normalizing Flows by Hand

In **Generative Normalizing Flows** (e.g., RealNVP, Glow), an input $x \in \mathbb{R}^2$ is transformed invertibly into latent Gaussian noise $z = f(x) = A x$.
To compute the exact likelihood of the input $p_X(x)$, the model uses the **Multivariate Change of Variables Theorem**:
$$p_X(x) = p_Z(z) \cdot |\det(A)| \implies \log p_X(x) = \log p_Z(z) + \log |\det(A)|$$

Let us trace this transformation, its exact determinant, its inverse, and the density calculation completely by hand!

#### 1. "What Refers to What" — Explicit Symbol & Dimension Dictionary

| Symbol | Shape | Mathematical Meaning | Deep Learning Semantic Role | Concrete Demo Value |
| :---: | :---: | :--- | :--- | :---: |
| $x$ | $[2 \times 1]$ | Input vector | Data sample (e.g. image feature) | $\begin{bmatrix} 3 \\ 2 \end{bmatrix}$ |
| $A$ | $[2 \times 2]$ | Invertible upper-triangular matrix | Coupling layer linear weight | $\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$ |
| $z$ | $[2 \times 1]$ | Transformed vector | Latent code $z = A x \sim \mathcal{N}(0, I)$ | To be computed |
| $\det(A)$ | Scalar | Determinant of $A$ | Volume expansion factor | To be computed |
| $A^{-1}$ | $[2 \times 2]$ | Matrix inverse | Generative decoding weight ($x = A^{-1} z$) | To be computed |
| $p_Z(z)$ | Scalar | Standard Normal PDF $\frac{1}{2\pi} e^{-\frac{1}{2}\|z\|_2^2}$ | Prior latent probability density | To be computed |

---

#### 2. Visual Layer Grid Setup

```
             INPUT x (2 x 1)               INVERTIBLE LAYER A (2 x 2)
              ┌──────────┐                     Col 1      Col 2
        Row 1 │  x_1 = 3 │               Row 1 ┌──────────┬──────────┐
              ├──────────┤                     │  A_11=2  │  A_12=1  │
        Row 2 │  x_2 = 2 │               Row 2 ├──────────┼──────────┤
              └──────────┘                     │  A_21=0  │  A_22=3  │
                                               └──────────┴──────────┘
```

---

#### 3. Forward Pass Calculation ($z = A x$)

$$\begin{bmatrix} z_1 \\ z_2 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix}$$

| Latent Cell | Operands Involved | Exact Row $\times$ Column Arithmetic | Result |
| :---: | :--- | :--- | :---: |
| **Row 1 ($z_1$)** | Row 1: $[2, 1]$, Vector: $[3, 2]^T$ | $(2 \times 3) + (1 \times 2) = 6 + 2$ | **$8$** |
| **Row 2 ($z_2$)** | Row 2: $[0, 3]$, Vector: $[3, 2]^T$ | $(0 \times 3) + (3 \times 2) = 0 + 6$ | **$6$** |

$$z = \begin{bmatrix} 8 \\ 6 \end{bmatrix}$$

---

#### 4. By-Hand Determinant Calculation ($\det(A)$)
Since $A$ is upper-triangular:
$$\det(A) = A_{11} \cdot A_{22} - A_{12} \cdot A_{21} = (2 \times 3) - (1 \times 0) = 6 - 0 = \mathbf{6}$$

*Geometric Interpretation:* The layer stretches the 2D space by an area factor of **$6.0\times$**!
Log-determinant contribution:
$$\log |\det(A)| = \ln(6) \approx 1.79176$$

---

#### 5. Analytical Matrix Inversion By Hand ($A^{-1}$)
For a $2 \times 2$ matrix $M = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:
$$M^{-1} = \frac{1}{\det(M)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

Substituting $a=2, b=1, c=0, d=3$ and $\det(A) = 6$:
$$A^{-1} = \frac{1}{6} \begin{bmatrix} 3 & -1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 3/6 & -1/6 \\ 0/6 & 2/6 \end{bmatrix} = \begin{bmatrix} 1/2 & -1/6 \\ 0 & 1/3 \end{bmatrix}$$

---

#### 6. Generative Backward Inversion ($x = A^{-1} z$)
Let us generate the original input $x$ back from latent $z = \begin{bmatrix} 8 \\ 6 \end{bmatrix}$:

$$\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}_{\text{recon}} = \begin{bmatrix} 1/2 & -1/6 \\ 0 & 1/3 \end{bmatrix} \begin{bmatrix} 8 \\ 6 \end{bmatrix}$$

| Recovered Cell | Operands Involved | Exact Row $\times$ Column Arithmetic | Cell Result |
| :---: | :--- | :--- | :---: |
| **Row 1 ($x_1$)** | Row 1: $[1/2, -1/6]$, Vector: $[8, 6]^T$ | $(1/2 \times 8) + (-1/6 \times 6) = 4 - 1$ | **$3$** |
| **Row 2 ($x_2$)** | Row 2: $[0, 1/3]$, Vector: $[8, 6]^T$ | $(0 \times 8) + (1/3 \times 6) = 0 + 2$ | **$2$** |

$$x_{\text{recon}} = \begin{bmatrix} 3 \\ 2 \end{bmatrix} \equiv x_{\text{orig}} \quad \checkmark$$
Exact, bit-for-bit lossless mathematical inversion!

---

#### 7. Change of Variables Density Evaluation
Assume latent prior is standard 2D Gaussian:
$$p_Z(z) = \frac{1}{2\pi} \exp\left( -\frac{1}{2} \|z\|_2^2 \right) = \frac{1}{2\pi} \exp\left( -\frac{8^2 + 6^2}{2} \right) = \frac{1}{2\pi} \exp(-50)$$
$$\ln p_Z(z) = -\ln(2\pi) - 50 \approx -1.83788 - 50 = -51.83788$$
Now, applying the Change of Variables formula:
$$\ln p_X(x) = \ln p_Z(z) + \ln |\det(A)| = -51.83788 + 1.79176 = \mathbf{-50.04612}$$
Without the determinant term $\ln |\det(A)|$, the probability density would violate conservation of probability measure!

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Standard Case — Complete $3 \times 3$ Matrix Inversion via Gauss-Jordan

**Problem:** Given matrix $A \in \mathbb{R}^{3 \times 3}$:
$$A = \begin{bmatrix} 1 & 0 & 2 \\ 2 & -1 & 3 \\ 4 & 1 & 8 \end{bmatrix}$$
1. Compute $\det(A)$ using cofactor expansion.
2. Is $A$ invertible?
3. Find $A^{-1}$ using the augmented Gauss-Jordan elimination $[A \mid I] \to [I \mid A^{-1}]$.

#### Step 1: Compute $\det(A)$ via First Row Cofactors
$$\det(A) = 1 \cdot \begin{vmatrix} -1 & 3 \\ 1 & 8 \end{vmatrix} - 0 \cdot \begin{vmatrix} 2 & 3 \\ 4 & 8 \end{vmatrix} + 2 \cdot \begin{vmatrix} 2 & -1 \\ 4 & 1 \end{vmatrix}$$
- Sub-determinant 1: $(-1 \times 8) - (3 \times 1) = -8 - 3 = -11$.
- Sub-determinant 2: $(2 \times 1) - (-1 \times 4) = 2 + 4 = 6$.
$$\det(A) = 1(-11) - 0 + 2(6) = -11 + 12 = \mathbf{1}$$
Since $\det(A) = 1 \neq 0$, the matrix is **invertible**, and its transformation **preserves exact volumes**!

---

#### Step 2: Gauss-Jordan Inversion $[A \mid I]$
We construct $[A \mid I]$:
$$\left[\begin{array}{ccc|ccc} 1 & 0 & 2 & 1 & 0 & 0 \\ 2 & -1 & 3 & 0 & 1 & 0 \\ 4 & 1 & 8 & 0 & 0 & 1 \end{array}\right]$$

- **Row Operations to create zeros in Column 1:**
  - $R_2 \leftarrow R_2 - 2R_1$:
    $$\text{Left: } [0, -1, 3 - 4] = [0, -1, -1], \quad \text{Right: } [0 - 2, 1, 0] = [-2, 1, 0]$$
  - $R_3 \leftarrow R_3 - 4R_1$:
    $$\text{Left: } [0, 1, 8 - 8] = [0, 1, 0], \quad \text{Right: } [0 - 4, 0, 1] = [-4, 0, 1]$$
  $$\left[\begin{array}{ccc|ccc} 1 & 0 & 2 & 1 & 0 & 0 \\ 0 & -1 & -1 & -2 & 1 & 0 \\ 0 & 1 & 0 & -4 & 0 & 1 \end{array}\right]$$

- **Row Operations on Column 2:**
  - Scale $R_2 \leftarrow -R_2$:
    $$\left[\begin{array}{ccc|ccc} 1 & 0 & 2 & 1 & 0 & 0 \\ 0 & 1 & 1 & 2 & -1 & 0 \\ 0 & 1 & 0 & -4 & 0 & 1 \end{array}\right]$$
  - $R_3 \leftarrow R_3 - R_2$:
    $$\text{Left: } [0, 0, 0 - 1] = [0, 0, -1], \quad \text{Right: } [-4 - 2, 0 - (-1), 1] = [-6, 1, 1]$$
  $$\left[\begin{array}{ccc|ccc} 1 & 0 & 2 & 1 & 0 & 0 \\ 0 & 1 & 1 & 2 & -1 & 0 \\ 0 & 0 & -1 & -6 & 1 & 1 \end{array}\right]$$

- **Row Operations on Column 3:**
  - Scale $R_3 \leftarrow -R_3$:
    $$\left[\begin{array}{ccc|ccc} 1 & 0 & 2 & 1 & 0 & 0 \\ 0 & 1 & 1 & 2 & -1 & 0 \\ 0 & 0 & 1 & 6 & -1 & -1 \end{array}\right]$$
  - $R_2 \leftarrow R_2 - R_3$:
    $$\text{Right: } [2 - 6, -1 - (-1), 0 - (-1)] = [-4, 0, 1]$$
  - $R_1 \leftarrow R_1 - 2R_3$:
    $$\text{Right: } [1 - 2(6), 0 - 2(-1), 0 - 2(-1)] = [-11, 2, 2]$$
  $$\left[\begin{array}{ccc|ccc} 1 & 0 & 0 & -11 & 2 & 2 \\ 0 & 1 & 0 & -4 & 0 & 1 \\ 0 & 0 & 1 & 6 & -1 & -1 \end{array}\right]$$

The left side is now $I_3$. The right side is our inverse:
$$A^{-1} = \begin{bmatrix} -11 & 2 & 2 \\ -4 & 0 & 1 \\ 6 & -1 & -1 \end{bmatrix}$$

---

#### Step 3: Verification ($A A^{-1} = I$)
$$A A^{-1} = \begin{bmatrix} 1 & 0 & 2 \\ 2 & -1 & 3 \\ 4 & 1 & 8 \end{bmatrix} \begin{bmatrix} -11 & 2 & 2 \\ -4 & 0 & 1 \\ 6 & -1 & -1 \end{bmatrix}$$
- Row 1 $\times$ Col 1: $1(-11) + 0(-4) + 2(6) = -11 + 12 = 1 \quad \checkmark$
- Row 1 $\times$ Col 2: $1(2) + 0(0) + 2(-1) = 2 - 2 = 0 \quad \checkmark$
- Row 2 $\times$ Col 1: $2(-11) + (-1)(-4) + 3(6) = -22 + 4 + 18 = 0 \quad \checkmark$
- Row 2 $\times$ Col 2: $2(2) + (-1)(0) + 3(-1) = 4 + 0 - 3 = 1 \quad \checkmark$
- Row 3 $\times$ Col 3: $4(2) + 1(1) + 8(-1) = 8 + 1 - 8 = 1 \quad \checkmark$
Inversion is verified.

---

### Scenario B: Change of Basis on a Linear Operator
**Problem:** A linear operator in standard basis $\mathcal{E}$ is $A = \begin{bmatrix} 2 & 0 \\ 0 & 5 \end{bmatrix}$.
Find its matrix representation $[T]_\mathcal{B}$ in the rotated basis $\mathcal{B} = \left\{ \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \begin{bmatrix} -1 \\ 1 \end{bmatrix} \right\}$.

1. Construct change of basis matrix $P$:
   $$P = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \quad \det(P) = (1 \times 1) - (-1 \times 1) = 2$$
2. Compute $P^{-1}$:
   $$P^{-1} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \\ -0.5 & 0.5 \end{bmatrix}$$
3. Compute $[T]_\mathcal{B} = P^{-1} A P$:
   $$A P = \begin{bmatrix} 2 & 0 \\ 0 & 5 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & -2 \\ 5 & 5 \end{bmatrix}$$
   $$P^{-1} (A P) = \begin{bmatrix} 0.5 & 0.5 \\ -0.5 & 0.5 \end{bmatrix} \begin{bmatrix} 2 & -2 \\ 5 & 5 \end{bmatrix} = \begin{bmatrix} 0.5(2)+0.5(5) & 0.5(-2)+0.5(5) \\ -0.5(2)+0.5(5) & -0.5(-2)+0.5(5) \end{bmatrix} = \begin{bmatrix} 3.5 & 1.5 \\ 1.5 & 3.5 \end{bmatrix}$$
4. **Invariant Check**:
   - $\text{Tr}(A) = 2 + 5 = 7$.
   - $\text{Tr}([T]_\mathcal{B}) = 3.5 + 3.5 = \mathbf{7} \quad \checkmark$
   - $\det(A) = 2 \times 5 = 10$.
   - $\det([T]_\mathcal{B}) = (3.5 \times 3.5) - (1.5 \times 1.5) = 12.25 - 2.25 = \mathbf{10} \quad \checkmark$
   *Trace and determinant are strictly invariant under change of basis!*

---

### Scenario C: Pathological Boundary Case — Singular Matrices ($\det = 0$)
**Problem:** What happens during Gauss-Jordan inversion of $M = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$?

$$\left[\begin{array}{cc|cc} 1 & 2 & 1 & 0 \\ 2 & 4 & 0 & 1 \end{array}\right] \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \left[\begin{array}{cc|cc} 1 & 2 & 1 & 0 \\ 0 & 0 & -2 & 1 \end{array}\right]$$
- Row 2 on the left side has become **all zeros** ($[0, 0]$).
- It is impossible to produce a leading pivot in column 2.
- The rank is $r = 1 < 2$.
- $\det(M) = (1 \times 4) - (2 \times 2) = 0$.
- **Conclusion:** Space has collapsed onto the 1D line $y = 2x$. The matrix cannot be inverted.

---

### Scenario D: 2D Affine Transformation in Homogeneous Coordinates (Forward & Inverse Mapping)

**Problem Formulation:**
In computer vision and spatial transformer networks (STNs), spatial transformations compose linear operations (rotations, scales) with non-linear translations:
$$y = R x + t$$
Because translation $t \neq \mathbf{0}$ is not linear, it cannot be represented as a $2 \times 2$ matrix multiplication. We lift 2D coordinates into 3D **Homogeneous Coordinates**:
$$\tilde{x} = \begin{bmatrix} x_1 \\ x_2 \\ 1 \end{bmatrix} \in \mathbb{R}^3, \quad T_{\text{affine}} = \begin{bmatrix} R & t \\ \mathbf{0}^T & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

Consider a counter-clockwise rotation by $\theta = 90^\circ$ followed by translation $t = \begin{bmatrix} 2 \\ -3 \end{bmatrix}$ applied to point $p = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$:
1. Construct the 2D rotation matrix $R$ and the $3 \times 3$ homogeneous affine transformation matrix $T_{\text{affine}}$.
2. Compute the forward transformed point $\tilde{p}' = T_{\text{affine}} \tilde{p}$.
3. Derive the analytical formula for $T_{\text{affine}}^{-1}$ in block form.
4. Calculate $T_{\text{affine}}^{-1}$ and verify that it maps $p'$ back to $p$.

#### Step 1: Construct Homogeneous Transformation Matrix
For $\theta = 90^\circ$:
$$R = \begin{bmatrix} \cos(90^\circ) & -\sin(90^\circ) \\ \sin(90^\circ) & \cos(90^\circ) \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
With translation $t = \begin{bmatrix} 2 \\ -3 \end{bmatrix}$:
$$T_{\text{affine}} = \begin{bmatrix} 0 & -1 & 2 \\ 1 & 0 & -3 \\ 0 & 0 & 1 \end{bmatrix}$$

#### Step 2: Forward Mapping
Point $p = [1, 2]^T$ becomes $\tilde{p} = [1, 2, 1]^T$:
$$\tilde{p}' = T_{\text{affine}} \tilde{p} = \begin{bmatrix} 0 & -1 & 2 \\ 1 & 0 & -3 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}$$
- Row 1: $(0 \times 1) + (-1 \times 2) + (2 \times 1) = 0 - 2 + 2 = 0$
- Row 2: $(1 \times 1) + (0 \times 2) + (-3 \times 1) = 1 + 0 - 3 = -2$
- Row 3: $(0 \times 1) + (0 \times 2) + (1 \times 1) = 1$

$$\tilde{p}' = \begin{bmatrix} 0 \\ -2 \\ 1 \end{bmatrix} \implies p' = \begin{bmatrix} 0 \\ -2 \end{bmatrix}$$

#### Step 3: Analytical Inverse in Block Form
For any affine matrix $T = \begin{bmatrix} R & t \\ \mathbf{0}^T & 1 \end{bmatrix}$ where $R$ is orthogonal ($R^{-1} = R^T$):
We require $T^{-1} T = I$:
$$\begin{bmatrix} R^T & -R^T t \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} R & t \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} R^T R & R^T t - R^T t \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} I & \mathbf{0} \\ \mathbf{0}^T & 1 \end{bmatrix} = I_3$$
Therefore:
$$T_{\text{affine}}^{-1} = \begin{bmatrix} R^T & -R^T t \\ \mathbf{0}^T & 1 \end{bmatrix}$$

Compute the blocks:
- $R^T = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$
- $-R^T t = - \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} 2 \\ -3 \end{bmatrix} = - \begin{bmatrix} (0)(2) + (1)(-3) \\ (-1)(2) + (0)(-3) \end{bmatrix} = - \begin{bmatrix} -3 \\ -2 \end{bmatrix} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$

$$T_{\text{affine}}^{-1} = \begin{bmatrix} 0 & 1 & 3 \\ -1 & 0 & 2 \\ 0 & 0 & 1 \end{bmatrix}$$

#### Step 4: Verification of Inverse Mapping
Apply $T_{\text{affine}}^{-1}$ to $\tilde{p}' = [0, -2, 1]^T$:
$$\tilde{p}_{\text{recovered}} = \begin{bmatrix} 0 & 1 & 3 \\ -1 & 0 & 2 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ -2 \\ 1 \end{bmatrix}$$
- Row 1: $(0 \times 0) + (1 \times -2) + (3 \times 1) = 0 - 2 + 3 = 1$
- Row 2: $(-1 \times 0) + (0 \times -2) + (2 \times 1) = 0 + 0 + 2 = 2$
- Row 3: $(0 \times 0) + (0 \times -2) + (1 \times 1) = 1$

$$\tilde{p}_{\text{recovered}} = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \implies p = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \quad \checkmark$$
The original point is exactly recovered!

---

## Part 7: Deep Learning Connection & Application

### 1. Normalizing Flows (RealNVP, Glow)
Standard matrix determinant computation requires Gaussian elimination, which scales cubically as $\mathcal{O}(D^3)$. For an image of dimension $D = 1024 \times 1024 \times 3 \approx 3 \times 10^6$, computing $\det(W)$ is completely impossible.
- **Architectural Solution (Coupling Layers)**: Normalizing flows construct triangular Jacobian transformations:
  $$J = \begin{bmatrix} I & 0 \\ \frac{\partial f_2}{\partial x_1} & \text{diag}(s(x_1)) \end{bmatrix}$$
- Because $J$ is lower triangular, its determinant is simply the product of diagonal elements:
  $$\log |\det(J)| = \sum_{i} \log |s_i(x_1)|$$
  This drops computational complexity from $\mathcal{O}(D^3)$ to **$\mathcal{O}(D)$ linear time**!

### 2. Orthogonal Initialization & Unitary Transformations
If a neural network has $L = 100$ layers:
$$y_L = W_L W_{L-1} \dots W_1 x$$
- If $|\det(W_l)| > 1$, activations explode exponentially ($\|y_L\| \to \infty$).
- If $|\det(W_l)| < 1$, activations vanish exponentially ($\|y_L\| \to 0$).
- By choosing **Orthogonal matrices** ($W^T W = I \implies \det(W) = \pm 1$), the transformation preserves lengths and angles, guaranteeing stable forward signals and stable gradients across 100+ layers without normalization!

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/04_linear_transformations_and_invertibility.py`](./code/04_linear_transformations_and_invertibility.py) implements:
1. **Gauss-Jordan Matrix Inverter from Scratch**: Inverts any non-singular matrix and detects singular matrices gracefully.
2. **Determinant & Volume Scale Engine**: Computes exact determinants and traces, verifying trace/determinant invariance under similarity transformations ($P^{-1} A P$).
3. **Change of Basis Solver**: Converts coordinate vectors between arbitrary basis systems.
4. **PyTorch Normalizing Flow Coupling Layer**: Implements an invertible affine coupling block, verifying exact forward-backward reconstruction ($x = f^{-1}(f(x))$) and analytical log-determinant volume tracking.

---

## Chapter 1.4 Summary & Review Checklist

- [x] What are the two linearity axioms ($T(u+v) = T(u)+T(v)$ and $T(cu) = cT(u)$)?
- [x] Why does adding a non-zero bias $b$ turn a linear transformation into an affine transformation?
- [x] What is the geometric interpretation of the determinant as a signed volume scaling factor?
- [x] Why do similar matrices ($B = P^{-1} A P$) have the exact same trace and determinant?
- [x] How do Normalizing Flows leverage triangular matrices to compute log-determinants in $\mathcal{O}(D)$ time?
