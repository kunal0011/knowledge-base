# Chapter 1.9: Tensors, Contractions & Einstein Summation Notation

---

## Part 1: Intuition & 101 Motivation

In modern Deep Learning, data is rarely confined to 1D vectors or 2D matrices:
- **Computer Vision (CNNs)**: A mini-batch of images is a **4D Tensor** with shape $[B, C, H, W]$ (Batch size, Channels, Height, Width).
- **Large Language Models (Transformers)**: Multi-Head Attention activations live in a **4D Tensor** with shape $[B, H, T, d_k]$ (Batch size, Number of Attention Heads, Sequence Length, Head Dimension).
- **Video & Volumetric Models**: Video batches are **5D Tensors** with shape $[B, C, T, H, W]$.

When writing high-performance neural networks, manipulating these multidimensional tensors using standard matrix operations becomes a nightmare of cryptic tensor reshaping:
```python
# The Traditional Cryptic Way:
scores = torch.bmm(Q.view(-1, T, d), K.view(-1, T, d).transpose(1, 2)).view(B, H, T, T)
```
This is prone to shape bugs, index transposition errors, and memory layout issues.

In 1916, Albert Einstein solved an identical problem in General Relativity by introducing **Einstein Summation Notation (`einsum`)**.
With `einsum`, every tensor contraction, batched multiplication, transposition, trace, and attention pooling is expressed in a clean, unambiguous string of indices:
```python
# The Elegant Einsum Way:
scores = torch.einsum('b h i d, b h j d -> b h i j', Q, K)
```

In this final chapter of Module 1, we master the mathematics of multidimensional tensors, tensor contractions, broadcasting rules, and Einstein summation.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Tensors: Definitions and Terminology
A **Tensor** of order $k$ (or rank $k$) is a multidimensional array of real numbers indexed by $k$ coordinates:
$$T \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_k}$$

| Tensor Order (Rank) | Mathematical Object | Deep Learning Example | Indexing Notation |
| :---: | :--- | :--- | :---: |
| **Rank 0** | Scalar | Loss value $\mathcal{L} \in \mathbb{R}$ | $s$ |
| **Rank 1** | Vector | Word embedding / bias $b \in \mathbb{R}^D$ | $v_i$ |
| **Rank 2** | Matrix | Linear layer weights $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$ | $M_{ij}$ |
| **Rank 3** | 3D Tensor | Batched token sequence $X \in \mathbb{R}^{B \times T \times D}$ | $X_{b, t, d}$ |
| **Rank 4** | 4D Tensor | Multi-Head Attention $Q \in \mathbb{R}^{B \times H \times T \times d_k}$ | $Q_{b, h, t, d}$ |
| **Rank 5** | 5D Tensor | Batched video clips $V \in \mathbb{R}^{B \times C \times T \times H \times W}$ | $V_{b, c, t, h, w}$ |

---

### 2. Fundamental Tensor Operations

#### A. Tensor Contraction (Summation over Axes)
A **contraction** sums over one or more pairs of matching indices, reducing the overall tensor rank:
$$C_{ik} = \sum_{j=1}^m A_{ij} B_{jk}$$
Here, index $j$ is contracted (summed out). Two rank-2 matrices are contracted into another rank-2 matrix.

#### B. Outer Product / Tensor Product ($\otimes$)
Takes two tensors and combines them without contraction, increasing total rank:
$$T = u \otimes v \implies T_{ij} = u_i v_j \quad (\text{Rank } 1 + \text{Rank } 1 = \text{Rank } 2)$$

#### C. Hadamard (Element-wise) Product ($\odot$)
Multiplies corresponding elements of two tensors of identical shape:
$$C_{ijk} = A_{ijk} B_{ijk}$$

#### Rigorous First-Principles Derivation: Tensor Contraction as Canonical Dual Pairing
**Mathematical Setup:**
1. Let $V_1, V_2, \dots, V_p$ be finite-dimensional vector spaces over $\mathbb{R}$, and let $V_1^*, \dots, V_p^*$ be their dual spaces (spaces of linear functionals).
2. A tensor $T$ of type $(r, s)$ is formally defined as a multilinear map:
   $$T: \underbrace{V_1^* \times \dots \times V_r^*}_{r \text{ dual arguments}} \times \underbrace{V_1 \times \dots \times V_s}_{s \text{ primal arguments}} \to \mathbb{R}$$
3. Choose ordered bases $\{e_i\}$ for $V$ and dual bases $\{e^j\}$ for $V^*$ satisfying $e^j(e_i) = \delta^j_i$.
   The components of $T$ are given by evaluating on basis vectors:
   $$T^{i_1 \dots i_r}_{j_1 \dots j_s} = T(e^{i_1}, \dots, e^{i_r}, e_{j_1}, \dots, e_{j_s})$$
4. **The Contraction Operator:**
   Contracting the $a$-th upper index with the $b$-th lower index is the linear map:
   $$\mathcal{C}^a_b: T^{(r, s)} \to T^{(r-1, s-1)}$$
   defined by evaluating the natural dual pairing $\langle e^k, e_k \rangle = 1$ over the complete basis:
   $$(\mathcal{C}^a_b(T))^{i_1 \dots \hat{i}_a \dots i_r}_{j_1 \dots \hat{j}_b \dots j_s} = \sum_{k=1}^{\dim(V)} T^{i_1 \dots k \dots i_r}_{j_1 \dots k \dots j_s}$$
   This proves that tensor contraction is coordinate-invariant: it does not depend on the choice of basis! $\blacksquare$

---

### 3. The Einstein Summation Convention & FLOP Complexity

#### The Einsum FLOP Counting Theorem:
For any general `einsum` contraction equation:
$$\text{'[input indices] -> [output indices]'}$$
1. Identify the set of **Free Indices** $\mathcal{F} = \{f_1, f_2, \dots, f_m\}$ that appear on the right side of `->`.
2. Identify the set of **Contracted Indices** $\mathcal{C} = \{c_1, c_2, \dots, c_p\}$ that appear on the left but NOT on the right.
3. Let $d(i)$ be the dimension size of index $i$.
4. **Total Floating Point Operations (FLOPs):**
   Each output element requires a multiply-accumulate operation (1 multiply + 1 add = 2 FLOPs) across all contracted combinations:
   $$\text{Total FLOPs} = 2 \cdot \left( \prod_{f \in \mathcal{F}} d(f) \right) \cdot \left( \prod_{c \in \mathcal{C}} d(c) \right)$$

##### Deep Learning Examples:
- **Matrix Multiply** `'i k, k j -> i j'`:
  $\mathcal{F} = \{i, j\}$ ($M \times N$), $\mathcal{C} = \{k\}$ ($K$).
  $$\text{FLOPs} = 2 M N K$$
- **Transformer Self-Attention Logits** `'b h i d, b h j d -> b h i j'`:
  $\mathcal{F} = \{b, h, i, j\}$ ($B \times H \times T \times T$), $\mathcal{C} = \{d\}$ ($d_k$).
  $$\text{FLOPs} = 2 B H T^2 d_k \quad (\text{Explains the } \mathcal{O}(T^2) \text{ quadratic context scaling!})$$

---

### 4. The Three Universal Rules of Einsum
Einstein's summation convention establishes three universal rules:

1. **Rule 1 (Repeated Index Summation):**
   Any index that appears in the input terms on the left of `->` but does **not** appear in the output term on the right is **summed over (contracted)**:
   $$A_{ik} B_{kj} \longrightarrow C_{ij} = \sum_k A_{ik} B_{kj}$$
2. **Rule 2 (Free Indices):**
   Indices that appear on both sides of `->` are **free indices**; they define the dimensions and order of the output tensor:
   $$\text{'i j -> j i'} \iff \text{Matrix Transposition } (A^T)_{ji} = A_{ij}$$
3. **Rule 3 (The Ellipsis `...`):**
   An ellipsis `...` represents arbitrary leading batch dimensions that are passed through without modification.

---

### 4. Master Rosetta Stone: Deep Learning Primitives in `einsum`

| Operation | Mathematical Formula | `einsum` Equation String | Equivalent PyTorch Primitive |
| :--- | :--- | :--- | :--- |
| **Vector Dot Product** | $\sum_i u_i v_i$ | `'i, i ->'` | `torch.dot(u, v)` |
| **Vector Outer Product** | $u_i v_j$ | `'i, j -> i j'` | `torch.outer(u, v)` |
| **Matrix-Vector Multiply** | $\sum_j W_{ij} x_j$ | `'i j, j -> i'` | `torch.mv(W, x)` |
| **Matrix Multiplication** | $\sum_k A_{ik} B_{kj}$ | `'i k, k j -> i j'` | `torch.mm(A, B)` |
| **Batch Matrix Multiply** | $\sum_k A_{b,i,k} B_{b,k,j}$ | `'b i k, b k j -> b i j'` | `torch.bmm(A, B)` |
| **Matrix Trace** | $\sum_i A_{ii}$ | `'i i ->'` | `torch.trace(A)` |
| **Diagonal Extraction** | $A_{ii}$ | `'i i -> i'` | `torch.diagonal(A)` |
| **Matrix Transpose** | $A_{ji}$ | `'i j -> j i'` | `A.t()` |
| **Batch Transpose** | $A_{b, j, i}$ | `'b i j -> b j i'` | `A.transpose(1, 2)` |
| **Global Avg Pooling (2D)**| $\frac{1}{HW}\sum_{h,w} X_{b,c,h,w}$ | `'b c h w -> b c'` | `nn.AdaptiveAvgPool2d(1)` |
| **Attention Scores** | $\sum_d Q_{b,h,i,d} K_{b,h,j,d}$ | `'b h i d, b h j d -> b h i j'` | Custom `bmm` chain |
| **Attention Context Pool** | $\sum_j A_{b,h,i,j} V_{b,h,j,d}$ | `'b h i j, b h j d -> b h i d'` | Custom `bmm` chain |

---

## Part 3: Geometric & Algebraic Interpretation

### Visualizing Tensor Contractions as Wire Diagrams

```
         Matrix Multiplication: 'i k, k j -> i j'
             ┌───────────┐         ┌───────────┐
      i ---->│           │  k   k  │           │----> j
             │     A     │========>│     B     │
             └───────────┘         └───────────┘
               Free Axis             Free Axis
                                          │
                                          v
                                    ┌───────────┐
                             i ---->│     C     │----> j
                                    └───────────┘
                                       Output

         Multi-Head Attention: 'b h i d, b h j d -> b h i j'
             ┌───────────┐         ┌───────────┐
      b ---->│           │         │           │<---- b  (Batch aligned)
      h ---->│     Q     │  d   d  │     K     │<---- h  (Head aligned)
      i ---->│           │========>│           │<---- j  (Key Token)
             └───────────┘         └───────────┘
               Query Token          Contraction
                                         │
                                         v
                                   ┌───────────┐
                            b ---->│           │<---- b
                            h ---->│  Scores   │<---- h
                            i ---->│     S     │<---- j
                                   └───────────┘
                                   Attention Grid
```

- Each index is a "wire" or "pin".
- When an index appears in two inputs ($k$ in $A_{ik}$ and $B_{kj}$), the wires are plugged into each other and **contracted** (summed).
- The remaining unconnected wires ($i$ and $j$) form the output tensor's shape.

---

## Part 4: Real-World Analogy

### The Multi-Warehouse Inventory System
Imagine managing a global retail inventory database:
- A 4D inventory tensor: `Inventory[City, Warehouse, Floor, Product]`
  - Shape: $[C, W, F, P]$
- **Operation 1: Total products across all cities and floors:**
  - `einsum('c w f p -> p', Inventory)`
  - Contracts (sums) over $C, W, F$, leaving only the product counts $P$.
- **Operation 2: Average inventory per city and warehouse:**
  - `einsum('c w f p -> c w', Inventory)`
  - Contracts over $F$ and $P$.
- Rather than running 4 nested `for` loops in Python, `einsum` describes the exact aggregation slice in a single declarative string!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

### Multi-Head Attention Score Calculation by Hand via `einsum`

Let us compute the Attention Logit Matrix completely by hand using the exact `einsum` contract:
$$\text{'b h i d, b h j d -> b h i j'}$$

#### 1. Setup & "What Refers to What" Dictionary

| Index | Name | Range in Demo | Semantic Role in Transformers |
| :---: | :--- | :---: | :--- |
| `b` | Batch index | $b = 0$ ($B = 1$) | Single prompt sequence |
| `h` | Head index | $h = 0$ ($H = 1$) | Single attention head |
| `i` | Query token index | $i \in \{0, 1\}$ ($T = 2$) | Token 1: "The", Token 2: "Cat" |
| `j` | Key token index | $j \in \{0, 1\}$ ($T = 2$) | Candidate memory tokens |
| `d` | Head dimension | $d \in \{0, 1\}$ ($d_k = 2$) | Latent feature channels |

- **Query Tensor $Q$** (Shape: $[1, 1, 2, 2]$):
  $$Q_{0, 0} = \begin{bmatrix} Q_{00} & Q_{01} \\ Q_{10} & Q_{11} \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 3 & 0 \end{bmatrix}$$
  - Token 0 query: $q_0 = [1, 2]$
  - Token 1 query: $q_1 = [3, 0]$

- **Key Tensor $K$** (Shape: $[1, 1, 2, 2]$):
  $$K_{0, 0} = \begin{bmatrix} K_{00} & K_{01} \\ K_{10} & K_{11} \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 4 & -1 \end{bmatrix}$$
  - Token 0 key: $k_0 = [2, 1]$
  - Token 1 key: $k_1 = [4, -1]$

---

#### 2. Visual Tensor Grid Representation

```
      QUERY Q (Tokens x Dim)                     KEY K (Tokens x Dim)
           Dim 0      Dim 1                           Dim 0      Dim 1
        ┌──────────┬──────────┐                    ┌──────────┬──────────┐
  Tok 0 │ Q_00 = 1 │ Q_01 = 2 │              Tok 0 │ K_00 = 2 │ K_01 = 1 │
        ├──────────┼──────────┤                    ├──────────┼──────────┤
  Tok 1 │ Q_10 = 3 │ Q_11 = 0 │              Tok 1 │ K_10 = 4 │ K_11 =-1 │
        └──────────┴──────────┘                    └──────────┴──────────┘
```

---

#### 3. Step-by-Step Cell Contraction ($S_{ij} = \sum_d Q_{id} K_{jd}$)

We contract along index $d \in \{0, 1\}$:

| Output Cell $(i, j)$ | Query Vector $q_i$ | Key Vector $k_j$ | Exact Contraction Arithmetic $\sum_d Q_{id} K_{jd}$ | Raw Logit $S_{ij}$ |
| :---: | :---: | :---: | :--- | :---: |
| **$S_{00}$ (Tok 0 $\to$ Tok 0)** | $[1, 2]$ | $[2, 1]$ | $(1 \times 2) + (2 \times 1) = 2 + 2$ | **$4$** |
| **$S_{01}$ (Tok 0 $\to$ Tok 1)** | $[1, 2]$ | $[4, -1]$| $(1 \times 4) + (2 \times -1) = 4 - 2$ | **$2$** |
| **$S_{10}$ (Tok 1 $\to$ Tok 0)** | $[3, 0]$ | $[2, 1]$ | $(3 \times 2) + (0 \times 1) = 6 + 0$ | **$6$** |
| **$S_{11}$ (Tok 1 $\to$ Tok 1)** | $[3, 0]$ | $[4, -1]$| $(3 \times 4) + (0 \times -1) = 12 + 0$ | **$12$** |

$$S = \begin{bmatrix} 4 & 2 \\ 6 & 12 \end{bmatrix} \in \mathbb{R}^{1 \times 1 \times 2 \times 2}$$

---

#### 4. Scaled Dot-Product Temperature Normalization ($\frac{S}{\sqrt{d_k}}$)
Here $d_k = 2 \implies \sqrt{d_k} = \sqrt{2} \approx 1.4142$:

| Cell | Raw Logit $S_{ij}$ | $\div \sqrt{2}$ | Scaled Logit $\tilde{S}_{ij}$ |
| :---: | :---: | :---: | :---: |
| **$\tilde{S}_{00}$** | $4$ | $4 / 1.4142$ | **$2.8284$** |
| **$\tilde{S}_{01}$** | $2$ | $2 / 1.4142$ | **$1.4142$** |
| **$\tilde{S}_{10}$** | $6$ | $6 / 1.4142$ | **$4.2426$** |
| **$\tilde{S}_{11}$** | $12$ | $12 / 1.4142$ | **$8.4853$** |

---

#### 5. Row-Wise Softmax Normalization by Hand

$$\text{AttnWeights}_{i, :} = \text{softmax}(\tilde{S}_{i, :})$$

##### A. Row 0 (Token 0 Attention Distribution):
- $e^{2.8284} \approx 16.918$
- $e^{1.4142} \approx 4.113$
- $\sum = 16.918 + 4.113 = 21.031$
$$A_{00} = \frac{16.918}{21.031} \approx \mathbf{0.804}, \quad A_{01} = \frac{4.113}{21.031} \approx \mathbf{0.196}$$
Token 0 pays $80.4\%$ attention to itself, and $19.6\%$ to Token 1.

##### B. Row 1 (Token 1 Attention Distribution):
- $e^{4.2426} \approx 69.589$
- $e^{8.4853} \approx 4843.08$
- $\sum = 69.589 + 4843.08 = 4912.67$
$$A_{10} = \frac{69.589}{4912.67} \approx \mathbf{0.014}, \quad A_{11} = \frac{4843.08}{4912.67} \approx \mathbf{0.986}$$
Token 1 pays $98.6\%$ attention to itself.

$$A = \begin{bmatrix} 0.804 & 0.196 \\ 0.014 & 0.986 \end{bmatrix}$$

---

#### 6. Context Value Pooling via `einsum('b h i j, b h j d -> b h i d')`
Let Value tensor $V$ be:
$$V = \begin{bmatrix} 10 & 0 \\ 0 & 20 \end{bmatrix}$$
- **Output for Token 0 ($i = 0$):**
  $$\text{Out}_0 = A_{00} v_0 + A_{01} v_1 = 0.804 \begin{bmatrix} 10 \\ 0 \end{bmatrix} + 0.196 \begin{bmatrix} 0 \\ 20 \end{bmatrix} = \begin{bmatrix} 8.04 \\ 3.92 \end{bmatrix}$$
- **Output for Token 1 ($i = 1$):**
  $$\text{Out}_1 = A_{10} v_0 + A_{11} v_1 = 0.014 \begin{bmatrix} 10 \\ 0 \end{bmatrix} + 0.986 \begin{bmatrix} 0 \\ 20 \end{bmatrix} = \begin{bmatrix} 0.14 \\ 19.72 \end{bmatrix}$$

Both attention scoring and value aggregation are executed by two single-line `einsum` calls!

---

## Part 6: Step-by-Step Solved Illustrations

### Scenario A: Translating Common Tensor Expressions to `einsum`

1. **Bilinear Form ($x^T A y$):**
   - Math: $\sum_{i} \sum_{j} x_i A_{ij} y_j$
   - `einsum`: `'i, i j, j ->'`
2. **Batched Covariance Trace ($\text{Tr}(\Sigma_b)$):**
   - Math: $T_b = \sum_{i} \Sigma_{b, i, i}$
   - `einsum`: `'b i i -> b'`
3. **Graph Neural Network (GNN) Message Passing:**
   - Adjacency matrix $A \in \mathbb{R}^{B \times N \times N}$, Node features $H \in \mathbb{R}^{B \times N \times D}$.
   - Math: $H^{\text{new}}_{b, i, d} = \sum_j A_{b, i, j} H_{b, j, d}$
   - `einsum`: `'b i j, b j d -> b i d'`
4. **Grouped-Query Attention (GQA) Key Replication:**
   - Query heads $H_q = 8$, Key heads $H_{kv} = 2$.
   - Each Key head is shared by 4 Query heads.
   - `einsum` automatically broadcasts along singleton dimensions:
     `'b (g h) t d, b h s d -> b (g h) t s'`

---

### Scenario B: Matrix Diagonal & Trace Mechanics
Given matrix $M = \begin{bmatrix} 2 & 7 \\ 3 & 5 \end{bmatrix}$:
1. **Extract Diagonal:**
   - Signature: `'i i -> i'`
   - Takes entries where row index equals column index: $[M_{00}, M_{11}] = [2, 5]$.
2. **Compute Trace:**
   - Signature: `'i i ->'`
   - Sums entries where row equals column: $M_{00} + M_{11} = 2 + 5 = \mathbf{7}$.

---

### Scenario C: Multimodal Bilinear Tensor Contraction by Hand

**Problem Formulation:**
In vision-language alignment models (e.g., VQA, CLIP projection), an image feature vector $v \in \mathbb{R}^2$ and a text feature vector $t \in \mathbb{R}^2$ are fused via a learned 3D weight tensor $W \in \mathbb{R}^{2 \times 2 \times 2}$ into output representation $y \in \mathbb{R}^2$:
$$y_o = \sum_{v=0}^1 \sum_{t=0}^1 v_v W_{o, v, t} t_t \iff \text{'v, o v t, t -> o'}$$

Given:
- Visual embedding: $v = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$
- Text embedding: $t = \begin{bmatrix} 3 \\ -1 \end{bmatrix}$
- Weight tensor slices:
  $$W_{o=0} = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}, \quad W_{o=1} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$

Calculate the 2D fused output vector $y = \begin{bmatrix} y_0 \\ y_1 \end{bmatrix}$ completely by hand.

#### Step 1: Compute Output Channel $0$ ($y_0 = v^T W_{o=0} t$)
$$W_{o=0} t = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ -1 \end{bmatrix} = \begin{bmatrix} 1(3) + 0(-1) \\ 0(3) + 2(-1) \end{bmatrix} = \begin{bmatrix} 3 \\ -2 \end{bmatrix}$$
$$y_0 = v^T (W_{o=0} t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ -2 \end{bmatrix} = (1)(3) + (2)(-2) = 3 - 4 = \mathbf{-1}$$

#### Step 2: Compute Output Channel $1$ ($y_1 = v^T W_{o=1} t$)
$$W_{o=1} t = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} 3 \\ -1 \end{bmatrix} = \begin{bmatrix} 0(3) + 1(-1) \\ -1(3) + 0(-1) \end{bmatrix} = \begin{bmatrix} -1 \\ -3 \end{bmatrix}$$
$$y_1 = v^T (W_{o=1} t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} -1 \\ -3 \end{bmatrix} = (1)(-1) + (2)(-3) = -1 - 6 = \mathbf{-7}$$

#### Step 3: Final Output Tensor
$$y = \begin{bmatrix} -1 \\ -7 \end{bmatrix}$$
*Insight:* `einsum('v, o v t, t -> o', v, W, t)` performs this multi-linear contraction in a single hardware kernel without ever allocating intermediate $2 \times 2 \times 2$ outer product memory tensors.

---

### Scenario D: Batched Matrix Multiplication & Index Permutation Mechanics

**Problem Formulation:**
Consider a mini-batch of 2 samples, each containing a $2 \times 2$ transformation matrix:
$$A \in \mathbb{R}^{2 \times 2 \times 2}, \quad B \in \mathbb{R}^{2 \times 2 \times 2}$$
where:
$$A_{b=0} = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}, \quad A_{b=1} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$
$$B_{b=0} = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}, \quad B_{b=1} = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}$$

Compute the batched product $C$ via:
$$\text{'b i k, b k j -> b i j'}$$

#### Step 1: Batch Item $b = 0$
$$C_{b=0} = A_{b=0} B_{b=0} = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$$
- $(C_{0})_{11} = (1)(3) + (0)(0) = 3$
- $(C_{0})_{12} = (1)(1) + (0)(2) = 1$
- $(C_{0})_{21} = (2)(3) + (1)(0) = 6$
- $(C_{0})_{22} = (2)(1) + (1)(2) = 2 + 2 = 4$
$$C_{b=0} = \begin{bmatrix} 3 & 1 \\ 6 & 4 \end{bmatrix}$$

#### Step 2: Batch Item $b = 1$
$$C_{b=1} = A_{b=1} B_{b=1} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}$$
- $(C_{1})_{11} = (0)(2) + (1)(1) = 1$
- $(C_{1})_{12} = (0)(0) + (1)(3) = 3$
- $(C_{1})_{21} = (1)(2) + (0)(1) = 2$
- $(C_{1})_{22} = (1)(0) + (0)(3) = 0$
$$C_{b=1} = \begin{bmatrix} 1 & 3 \\ 2 & 0 \end{bmatrix}$$

#### Step 3: Transposed Output Permutation `'b i k, b k j -> b j i'`
If the einsum string is modified to transpose each output matrix:
$$C^T_{b=0} = \begin{bmatrix} 3 & 6 \\ 1 & 4 \end{bmatrix}, \quad C^T_{b=1} = \begin{bmatrix} 1 & 2 \\ 3 & 0 \end{bmatrix}$$
The indices on the right hand side dictate the exact layout and stride ordering of the result!

---

## Part 7: Deep Learning Connection & Application

### 1. The Multi-Head Attention Engine in PyTorch
With `einsum`, a complete multi-head self-attention forward pass is implemented cleanly:
```python
def multi_head_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # 1. Compute raw attention scores: [B, H, T_q, T_k]
    scores = torch.einsum('b h i d, b h j d -> b h i j', Q, K) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 2. Softmax along key token axis
    weights = torch.softmax(scores, dim=-1)
    
    # 3. Aggregate values: [B, H, T_q, d_v]
    output = torch.einsum('b h i j, b h j d -> b h i d', weights, V)
    return output
```

### 2. Bilinear Pooling in Multimodal Learning (Vision-Language)
In models combining image features $v \in \mathbb{R}^{B \times D_v}$ and text features $t \in \mathbb{R}^{B \times D_t}$ via a learned 3D tensor $W \in \mathbb{R}^{D_{\text{out}} \times D_v \times D_t}$:
$$y_b = v_b^T W t_b \iff \text{'b v, o v t, b t -> b o'}$$
A single `einsum` executes this 3-way contraction without creating massive intermediate outer-product tensors!

---

## Part 8: Code Implementation & Verification

The accompanying Python script [`code/09_tensors_and_einsum.py`](./code/09_tensors_and_einsum.py) implements:
1. **The Einsum Rosetta Stone Suite**: Verifies all 12 canonical operations against standard NumPy/PyTorch primitives (`dot`, `matmul`, `outer`, `bmm`, `trace`, `transpose`).
2. **Pure `einsum` Multi-Head Attention Module**: Complete PyTorch implementation with causal masking.
3. **Equivalence & Sanity Check**: Verifies that our scratch `einsum` attention output matches PyTorch's native `nn.MultiheadAttention` bit-for-bit.
4. **Part 5 Numerical Reproduction**: Asserts the exact attention scores and softmax probabilities derived by hand.

---

## Chapter 1.9 Summary & Review Checklist

- [x] What is the difference between a free index and a contracted (summed) index in `einsum`?
- [x] How do you write matrix multiplication, trace, and transpose in `einsum` notation?
- [x] Why is `torch.einsum('b h i d, b h j d -> b h i j', Q, K)` superior to nested loops or `.permute().view().bmm()` chains?
- [x] How does tensor contraction generalize the dot product to multidimensional arrays?
