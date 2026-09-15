# Chapter 9.1: Scaled Dot-Product & Multi-Head Attention (MHA)

---

## 1. Intuition & 101 Motivation

For decades, the dominant paradigm for sequential deep learning was recurrence (RNNs, LSTMs, GRUs).
However, recurrent models suffered from an inescapable architectural bottleneck: **the temporal sequential dependency**.
To compute the hidden state at step $t$, the hardware must wait for step $t-1$ to finish. On modern massively parallel GPUs equipped with tens of thousands of compute cores, sequential execution leaves hardware drastically underutilized, capping training throughput and scaling potential.

In 2017, Ashish Vaswani and the Google Brain / Google Research team published:
> *"Attention Is All You Need"*

Their radical breakthrough was to discard recurrence and convolution entirely. Instead of passing information sequentially through time, **Self-Attention** allows every token in a sequence to connect directly to every other token in a single, parallel matrix multiplication.

```
       RECURRENT O(T) SEQUENTIAL CHAIN vs. TRANSFORMER O(1) PARALLEL ATTENTION
       
       Recurrent:    x_1 ──► [ h_1 ] ──► [ h_2 ] ──► ... ──► [ h_T ]   (O(T) Sequential Steps)
                               │           │                   │
                               
       Transformer:  [ x_1, x_2, ... , x_T ]
                           │
                           ▼  (All tokens attend to all tokens simultaneously)
                     ┌────────────────────────────────────────────────────────┐
                     │ Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V │  (O(1) Parallel Depth)
                     └────────────────────────────────────────────────────────┘
```

### The Database Metaphor: Queries, Keys, and Values
Self-attention models information retrieval like a soft, differentiable database lookup:
- **Query ($\mathbf{Q}$):** What a token is searching for (*"I am a pronoun; where is my antecedent noun?"*).
- **Key ($\mathbf{K}$):** What a token advertises (*"I am a singular masculine noun"*).
- **Value ($\mathbf{V}$):** The actual semantic content transferred if a Query and Key match (*"The underlying concept of 'John'"*).

### Why Multi-Head Attention?
A single attention distribution $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^T / \sqrt{d_k})$ can only compute a single weighted average over values. But natural language requires tracking multiple syntactic and semantic relationships simultaneously:
- Head 1 tracks grammatical subject-verb agreement.
- Head 2 tracks pronoun coreference resolution.
- Head 3 tracks local modifier-noun phrases.
- Head 4 tracks global discourse structure.

**Multi-Head Attention (MHA)** projects the input into $h$ distinct representation subspaces, applies scaled dot-product attention in each subspace independently, and concatenates the results.

---

## 2. Rigorous Mathematical Formulation

```
                     SCALED DOT-PRODUCT ATTENTION
                          Queries (Q)    Keys (K)
                               │            │
                               ▼            ▼
                            [ MatMul: Q * K^T ]
                                    │
                                    ▼
                         [ Scale: / sqrt(d_k) ]
                                    │
                                    ▼
                         [ Optional Mask (Add) ]
                                    │
                                    ▼
                             [ Softmax (dim=-1) ] ──► Attention Matrix A
                                    │
                                    ▼
                          [ MatMul: A * V ] ◄── Values (V)
                                    │
                                    ▼
                              Output Context
```

---

### 2.1 Scaled Dot-Product Attention

Let the input packed tensors be:
- $\mathbf{Q} \in \mathbb{R}^{T_q \times d_k}$ (Queries)
- $\mathbf{K} \in \mathbb{R}^{T_k \times d_k}$ (Keys)
- $\mathbf{V} \in \mathbb{R}^{T_k \times d_v}$ (Values)

The scaled dot-product attention mapping is:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}$$

where:
- $\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{T_q \times T_k}$ is the unnormalized attention score matrix.
- $\mathbf{M} \in \mathbb{R}^{T_q \times T_k}$ is an optional attention mask (e.g., causal autoregressive mask or padding mask).
- $\mathbf{A} = \text{softmax}(\mathbf{S} + \mathbf{M}) \in \mathbb{R}^{T_q \times T_k}$ is the row-stochastic attention weight matrix ($\sum_{j=1}^{T_k} A_{i, j} = 1$).
- The output tensor has shape $\mathbb{R}^{T_q \times d_v}$.

---

### 2.2 Mathematical Proof: Why Divide by $\sqrt{d_k}$? (The Variance Inflation Theorem)

Why does the dot product require scaling by $\frac{1}{\sqrt{d_k}}$?

#### Lemma: Variance of the Dot Product of Independent Vectors
Let $\mathbf{q} = [q_1, \dots, q_{d_k}]^T$ and $\mathbf{k} = [k_1, \dots, k_{d_k}]^T$ be two independent random vectors whose components are independent and identically distributed (I.I.D.) with zero mean and unit variance:
$$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1, \quad \forall i \in \{1, \dots, d_k\}$$

Consider their inner product $S = \mathbf{q}^T \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$.

1. **Expected Value:**
   $$\mathbb{E}[S] = \mathbb{E}\left[ \sum_{i=1}^{d_k} q_i k_i \right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = \sum_{i=1}^{d_k} (0)(0) = 0$$

2. **Variance:**
   Since the terms $q_i k_i$ are mutually independent:
   $$\operatorname{Var}(S) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i)$$
   Using the identity for the variance of products of independent zero-mean variables:
   $$\operatorname{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0 = (1)(1) = 1$$
   Therefore:
   $$\operatorname{Var}(S) = \sum_{i=1}^{d_k} 1 = d_k$$
   $$\sigma_S = \sqrt{\operatorname{Var}(S)} = \sqrt{d_k}$$

#### The Softmax Saturation Phenomenon:
As the head dimension $d_k$ grows large (e.g., $d_k = 64$ in BERT, $d_k = 128$ in GPT-3):
- The raw dot products $S$ have standard deviation $\sigma_S = \sqrt{64} = 8$ or $\sqrt{128} \approx 11.3$.
- Typical dot product values range widely across $[-30, +30]$.
- When fed into the softmax function $\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$, large input differences push the softmax into regions with **extremely small gradients**:
  $$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{i, j} - \text{softmax}(z)_j)$$
  If one logit is significantly larger than the others, $\text{softmax}(z)_i \approx 1.0$ and all others $\approx 0.0$.
  The gradient evaluates to:
  $$1.0 \times (1.0 - 1.0) = 0.0$$
- The attention mechanism saturates into a hard, non-differentiable argmax, **killing backpropagation gradient flow** through the Query and Key projection weights!

#### The Scaled Renormalization:
By dividing the dot product by $\sqrt{d_k}$:
$$\operatorname{Var}\left( \frac{S}{\sqrt{d_k}} \right) = \frac{1}{(\sqrt{d_k})^2} \operatorname{Var}(S) = \frac{1}{d_k} \cdot d_k = \mathbf{1.0}$$
The variance of attention scores is stabilized to $1.0$ regardless of how large $d_k$ becomes, keeping the softmax in its sensitive, active gradient regime.

---

### 2.3 Causal Masking (Autoregressive Decoder Attention)

In autoregressive language modeling (e.g., GPT), generation proceeds strictly left-to-right: token $i$ is forbidden from attending to future tokens $j > i$.

This constraint is enforced algebraically by adding an upper-triangular causal mask $\mathbf{M} \in \mathbb{R}^{T \times T}$:
$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

For a sequence of length $T = 3$:
$$\mathbf{M} = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}$$

When added before softmax:
$$\exp(S_{i, j} + (-\infty)) = \exp(-\infty) = 0$$
$$\mathbf{A} = \text{softmax}(\mathbf{S} + \mathbf{M}) = \begin{bmatrix}
1.0 & 0.0 & 0.0 \\
A_{2, 1} & A_{2, 2} & 0.0 \\
A_{3, 1} & A_{3, 2} & A_{3, 3}
\end{bmatrix}$$
The attention matrix becomes strictly lower-triangular, mathematically preserving causality while allowing the entire sequence to be computed in parallel during training.

---

### 2.4 Multi-Head Attention (MHA) Formulation

Let the input hidden sequence be $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$.
Given $h$ attention heads, we define head dimensions:
$$d_k = d_v = \frac{d_{\text{model}}}{h}$$

For each head $i \in \{1, \dots, h\}$, learnable linear projections map $\mathbf{X}$ into distinct subspaces:
$$\mathbf{Q}_i = \mathbf{X} \mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{X} \mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{X} \mathbf{W}_i^V$$
where $\mathbf{W}_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

Each head executes scaled dot-product attention:
$$\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) \in \mathbb{R}^{T \times d_v}$$

The $h$ heads are concatenated horizontally and projected back to $d_{\text{model}}$ using an output projection matrix $\mathbf{W}^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:
$$\text{MHA}(\mathbf{X}) = \left[ \text{head}_1; \, \text{head}_2; \, \dots; \, \text{head}_h \right] \mathbf{W}^O$$

```
                           MULTI-HEAD ATTENTION
                                   Input X
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                      Head 1        Head 2      Head h
                   ┌──────────┐  ┌──────────┐  ┌──────────┐
                   │ Q1,K1,V1 │  │ Q2,K2,V2 │  │ Qh,Kh,Vh │
                   └────┬─────┘  └────┬─────┘  └────┬─────┘
                        ▼             ▼             ▼
                   [Attn Head 1] [Attn Head 2] [Attn Head h]
                        │             │             │
                        └─────────────┼─────────────┘
                                      ▼
                        Concat: [ h_1, h_2, ..., h_h ]
                                      ▼
                               Linear (W^O)
                                      ▼
                               Output Tensor
```

---

### 2.5 Complete Analytical Gradient Derivation of Self-Attention

Given upstream loss gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{O}} \in \mathbb{R}^{T \times d_v}$ where $\mathbf{O} = \mathbf{A} \mathbf{V}$:

1. **Gradient w.r.t. Values ($\mathbf{V}$):**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{A}^T \frac{\partial \mathcal{L}}{\partial \mathbf{O}} \in \mathbb{R}^{T \times d_v}$$

2. **Gradient w.r.t. Attention Matrix ($\mathbf{A}$):**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \frac{\partial \mathcal{L}}{\partial \mathbf{O}} \mathbf{V}^T \in \mathbb{R}^{T \times T}$$

3. **Gradient w.r.t. Unnormalized Scaled Scores ($\mathbf{S}$):**
   Using the Jacobian of the softmax operator:
   $$\frac{\partial \mathcal{L}}{\partial S_{i, j}} = A_{i, j} \left( \frac{\partial \mathcal{L}}{\partial A_{i, j}} - \sum_{k=1}^T \frac{\partial \mathcal{L}}{\partial A_{i, k}} A_{i, k} \right)$$
   In vectorized matrix notation:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{S}} = \mathbf{A} \odot \left( \frac{\partial \mathcal{L}}{\partial \mathbf{A}} - \left[ \left(\frac{\partial \mathcal{L}}{\partial \mathbf{A}} \odot \mathbf{A}\right) \mathbf{1}_T \right] \mathbf{1}_T^T \right)$$

4. **Gradient w.r.t. Queries ($\mathbf{Q}$) and Keys ($\mathbf{K}$):**
   Since $\mathbf{S} = \frac{1}{\sqrt{d_k}} \mathbf{Q} \mathbf{K}^T$:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial \mathcal{L}}{\partial \mathbf{S}}\right) \mathbf{K} \in \mathbb{R}^{T \times d_k}$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial \mathcal{L}}{\partial \mathbf{S}}\right)^T \mathbf{Q} \in \mathbb{R}^{T \times d_k}$$

---

## 3. Geometric & Algebraic Interpretation

### The Low-Rank Bottleneck of Attention (Bhojanapalli et al., 2020)
Notice that the unnormalized score matrix is formed by:
$$\mathbf{S} = \frac{1}{\sqrt{d_k}} \mathbf{Q} \mathbf{K}^T$$
where $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{T \times d_k}$.
By the rank inequality of matrix products:
$$\operatorname{rank}(\mathbf{S}) \le \min(T, d_k)$$
If sequence length $T = 4,096$ and head dimension $d_k = 64$, the $4,096 \times 4,096$ attention matrix before softmax has **rank at most $64$**!
While the element-wise exponential in softmax strictly increases the rank, the effective representation space remains constrained. Multi-Head Attention solves this by summing $h$ distinct low-rank projections, yielding an effective rank of up to $h \cdot d_k = d_{\text{model}}$.

---

## 4. Real-World Analogy

### The Research Symposium / Panel Discussion
- **Recurrent Network (Single Microphone Pass):**
  A microphone is passed down a row of 100 scholars. Scholar 1 whispers to Scholar 2, who whispers to Scholar 3. By the time it reaches Scholar 100, the original question is distorted beyond recognition.
- **Single-Head Attention (The Loudspeaker):**
  The room has a central loudspeaker. One scholar speaks, and everyone in the room hears them simultaneously. However, because only one voice speaks at a time, they can only discuss one topic.
- **Multi-Head Attention (The Round-Table Breakout Rooms):**
  The 100 scholars split into 8 specialized breakout tables simultaneously:
  - Table 1 (Syntax) analyzes grammar and parsing.
  - Table 2 (Coreference) resolves pronoun references.
  - Table 3 (Entity Extraction) links names to dates.
  At the end of the round, each table presents its findings, which are merged onto the master whiteboard ($\mathbf{W}^O$).

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace forward and backward propagation through a Multi-Head Attention layer with concrete hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Multi-Head Attention |
| :--- | :--- | :--- | :--- |
| $T$ | Sequence Length | Scalar ($2$) | Number of tokens in sequence |
| $d_{\text{model}}$ | Model Dimension | Scalar ($4$) | Hidden embedding size |
| $h$ | Number of Heads | Scalar ($2$) | Parallel attention heads |
| $d_k, d_v$ | Head Dimensions | Scalar ($2$) | Dimension per head ($d_k = d_{\text{model}} / h = 2$) |
| $\mathbf{X}$ | Input Sequence Matrix | $(2, 4)$ | 2 tokens, each of dimension 4 |
| $\mathbf{W}_1^Q, \mathbf{W}_1^K, \mathbf{W}_1^V$ | Head 1 Projections | $(4, 2)$ each | Weight matrices for Head 1 |
| $\mathbf{Q}_1, \mathbf{K}_1, \mathbf{V}_1$ | Head 1 Projected Tensors | $(2, 2)$ each | Query, Key, Value representations for Head 1 |
| $\mathbf{S}_1$ | Scaled Dot-Product Scores | $(2, 2)$ | $\mathbf{Q}_1 \mathbf{K}_1^T / \sqrt{2}$ |
| $\mathbf{A}_1$ | Attention Matrix | $(2, 2)$ | Softmax normalized attention distribution |
| $\mathbf{O}_1$ | Head 1 Output | $(2, 2)$ | $\mathbf{A}_1 \mathbf{V}_1$ |

---

### 5.2 Concrete Toy Numbers

#### Input Matrix $\mathbf{X} \in \mathbb{R}^{2 \times 4}$:
$$\mathbf{X} = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 1.0
\end{bmatrix} \quad \begin{array}{l} \text{(Token 1)} \\ \text{(Token 2)} \end{array}$$

#### Head 1 Projection Matrices ($\mathbb{R}^{4 \times 2}$):
$$\mathbf{W}_1^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{W}_1^K = \begin{bmatrix} 1 & 1 \\ 0 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad \mathbf{W}_1^V = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

---

### 5.3 Step 1: Linear Projections for Head 1

1. **Queries $\mathbf{Q}_1 = \mathbf{X} \mathbf{W}_1^Q$:**
   $$\mathbf{Q}_1 = \begin{bmatrix}
   (1)(1) + (0)(0) + (1)(1) + (0)(0) & (1)(0) + (0)(1) + (1)(0) + (0)(1) \\
   (0)(1) + (1)(0) + (0)(1) + (1)(0) & (0)(0) + (1)(1) + (0)(0) + (1)(1)
   \end{bmatrix} = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$

2. **Keys $\mathbf{K}_1 = \mathbf{X} \mathbf{W}_1^K$:**
   $$\mathbf{K}_1 = \begin{bmatrix}
   (1)(1) + (0)(0) + (1)(0) + (0)(1) & (1)(1) + (0)(0) + (1)(1) + (0)(0) \\
   (0)(1) + (1)(0) + (0)(0) + (1)(1) & (0)(1) + (1)(0) + (0)(1) + (1)(0)
   \end{bmatrix} = \begin{bmatrix} 1.0 & 2.0 \\ 1.0 & 0.0 \end{bmatrix}$$

3. **Values $\mathbf{V}_1 = \mathbf{X} \mathbf{W}_1^V$:**
   $$\mathbf{V}_1 = \begin{bmatrix}
   (1)(2) + (0)(0) + (1)(1) + (0)(0) & (1)(0) + (0)(2) + (1)(0) + (0)(1) \\
   (0)(2) + (1)(0) + (0)(1) + (1)(0) & (0)(0) + (1)(2) + (0)(0) + (1)(1)
   \end{bmatrix} = \begin{bmatrix} 3.0 & 0.0 \\ 0.0 & 3.0 \end{bmatrix}$$

---

### 5.4 Step 2: Raw Dot Products and Scaling ($\sqrt{d_k} = \sqrt{2} \approx 1.414214$)

1. **Unscaled MatMul $\mathbf{Q}_1 \mathbf{K}_1^T$:**
   $$\mathbf{K}_1^T = \begin{bmatrix} 1.0 & 1.0 \\ 2.0 & 0.0 \end{bmatrix}$$
   $$\mathbf{Q}_1 \mathbf{K}_1^T = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.0 & 1.0 \\ 2.0 & 0.0 \end{bmatrix} = \begin{bmatrix} 2.0 & 2.0 \\ 4.0 & 0.0 \end{bmatrix}$$

2. **Scaling by $\frac{1}{\sqrt{2}} \approx 0.707107$:**
   $$\mathbf{S}_1 = \frac{\mathbf{Q}_1 \mathbf{K}_1^T}{\sqrt{2}} = \begin{bmatrix}
   2.0 \times 0.707107 & 2.0 \times 0.707107 \\
   4.0 \times 0.707107 & 0.0 \times 0.707107
   \end{bmatrix} = \mathbf{\begin{bmatrix} 1.414214 & 1.414214 \\ 2.828427 & 0.000000 \end{bmatrix}}$$

---

### 5.5 Step 3: Softmax Attention Weights ($\mathbf{A}_1$)

- **Row 1 ($S_{1, 1} = 1.414214, S_{1, 2} = 1.414214$):**
  Because logits are identical:
  $$A_{1, 1} = 0.5, \quad A_{1, 2} = 0.5$$

- **Row 2 ($S_{2, 1} = 2.828427, S_{2, 2} = 0.0$):**
  $$\exp(2.828427) \approx 16.918738, \quad \exp(0.0) = 1.0$$
  $$\text{Sum} = 16.918738 + 1.0 = 17.918738$$
  $$A_{2, 1} = \frac{16.918738}{17.918738} \approx \mathbf{0.944192}$$
  $$A_{2, 2} = \frac{1.0}{17.918738} \approx \mathbf{0.055808}$$

$$\mathbf{A}_1 = \begin{bmatrix}
0.500000 & 0.500000 \\
0.944192 & 0.055808
\end{bmatrix}$$

---

### 5.6 Step 4: Value Aggregation ($\mathbf{O}_1 = \mathbf{A}_1 \mathbf{V}_1$)

$$\mathbf{O}_1 = \begin{bmatrix}
0.500000 & 0.500000 \\
0.944192 & 0.055808
\end{bmatrix} \begin{bmatrix} 3.0 & 0.0 \\ 0.0 & 3.0 \end{bmatrix}$$

- **Row 1:**
  $$O_{1, 1} = 0.5(3.0) + 0.5(0.0) = \mathbf{1.5}$$
  $$O_{1, 2} = 0.5(0.0) + 0.5(3.0) = \mathbf{1.5}$$

- **Row 2:**
  $$O_{2, 1} = (0.944192)(3.0) + (0.055808)(0.0) = \mathbf{2.832576}$$
  $$O_{2, 2} = (0.944192)(0.0) + (0.055808)(3.0) = \mathbf{0.167424}$$

$$\mathbf{O}_1 = \begin{bmatrix} 1.500000 & 1.500000 \\ 2.832576 & 0.167424 \end{bmatrix}$$

Head 1 is complete! A parallel Head 2 computes $\mathbf{O}_2 \in \mathbb{R}^{2 \times 2}$. The concatenated output $[\mathbf{O}_1; \mathbf{O}_2] \in \mathbb{R}^{2 \times 4}$ is multiplied by $\mathbf{W}^O \in \mathbb{R}^{4 \times 4}$.

---

## 6. Solved Illustrations

### Illustration 1: Numerical Proof of Softmax Vanishing Gradient Without Scaling

**Problem:**
Let two keys have dot products with query $\mathbf{q}$: $S_1 = 30.0, S_2 = 10.0$.
Compare the softmax derivative $\frac{\partial A_1}{\partial S_1}$ for:
1. Unscaled scores ($S_1 = 30, S_2 = 10$).
2. Scaled scores with $d_k = 100 \implies \sqrt{d_k} = 10$ ($S_1' = 3.0, S_2' = 1.0$).

**Solution:**
1. **Unscaled Case:**
   $$\Delta S = 30 - 10 = 20 \implies \exp(20) \approx 4.85 \times 10^8$$
   $$A_1 = \frac{e^{30}}{e^{30} + e^{10}} = \frac{1}{1 + e^{-20}} \approx 1 - 2.06 \times 10^{-9}$$
   Softmax gradient:
   $$\frac{\partial A_1}{\partial S_1} = A_1 (1 - A_1) \approx (1.0)(2.06 \times 10^{-9}) \approx \mathbf{2.06 \times 10^{-9}}$$
   The gradient has virtually vanished! The weights receive practically zero update.

2. **Scaled Case ($\sqrt{d_k} = 10$):**
   $$S_1' = 3.0, \quad S_2' = 1.0 \implies \Delta S' = 2.0$$
   $$A_1' = \frac{e^3}{e^3 + e^1} = \frac{20.0855}{20.0855 + 2.7183} = \frac{20.0855}{22.8038} \approx 0.8808$$
   Softmax gradient:
   $$\frac{\partial A_1'}{\partial S_1'} = A_1' (1 - A_1') = (0.8808)(1 - 0.8808) = (0.8808)(0.1192) \approx \mathbf{0.1050}$$
   The gradient is **$50,000,000\times$ larger**! Scaling prevents gradient death.

---

### Illustration 2: Computational Complexity and FLOPs of Multi-Head Attention

**Problem:**
Calculate the exact training FLOPs for a single Multi-Head Attention layer with batch size $B$, sequence length $T$, model dimension $d_{\text{model}} = 4,096$, and $h = 32$ heads ($d_k = 128$).

**Solution:**
A standard matrix multiplication $(M \times K) \times (K \times N)$ requires $2 M K N$ floating-point operations.

1. **Q, K, V Projections:**
   Three projections $\mathbf{X} \mathbf{W}^Q, \mathbf{X} \mathbf{W}^K, \mathbf{X} \mathbf{W}^V$:
   $$\text{FLOPs}_{\text{proj}} = 3 \times 2 B T d_{\text{model}}^2 = 6 B T d_{\text{model}}^2$$
2. **Attention Scores ($\mathbf{Q} \mathbf{K}^T$):**
   Per head: $(T \times d_k) \times (d_k \times T) \to 2 T^2 d_k$.
   Across $h$ heads: $2 h T^2 d_k = 2 T^2 (h d_k) = 2 T^2 d_{\text{model}}$.
   $$\text{FLOPs}_{\text{score}} = 2 B T^2 d_{\text{model}}$$
3. **Value Aggregation ($\mathbf{A} \mathbf{V}$):**
   Per head: $(T \times T) \times (T \times d_v) \to 2 T^2 d_v$.
   Across $h$ heads:
   $$\text{FLOPs}_{\text{value}} = 2 B T^2 d_{\text{model}}$$
4. **Output Projection ($\mathbf{O} \mathbf{W}^O$):**
   $$\text{FLOPs}_{\text{out}} = 2 B T d_{\text{model}}^2$$

**Total Forward FLOPs:**
$$\text{FLOPs}_{\text{MHA}} = 8 B T d_{\text{model}}^2 + 4 B T^2 d_{\text{model}}$$

Notice the two terms:
- The first term $8 B T d_{\text{model}}^2$ scales **linearly with sequence length $T$**.
- The second term $4 B T^2 d_{\text{model}}$ scales **quadratically with sequence length $T^2$**.
When $T \ll d_{\text{model}}$ (e.g., $T = 512, d = 4,096$), linear projection dominates ($88\%$).
When $T \gg d_{\text{model}}$ (e.g., $T = 32,768$), the quadratic attention term dominates ($80\%$).

---

## 7. Deep Learning Connection & Application

### Standard MHA Configurations in Modern LLMs

| Model | $d_{\text{model}}$ | Heads ($h$) | $d_k$ | Sequence Length ($T$) | Total Attention Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT-Base** | $768$ | $12$ | $64$ | $512$ | $2.36 \text{ M}$ |
| **GPT-2 (1.5B)** | $1,600$ | $25$ | $64$ | $1,024$ | $10.24 \text{ M}$ |
| **GPT-3 (175B)** | $12,288$ | $96$ | $128$ | $2,048$ | $603.98 \text{ M}$ |
| **LLaMA-3 (70B)** | $8,192$ | $64$ | $128$ | $8,192$ | $268.44 \text{ M}$ |

In standard inference, MHA requires storing past Keys and Values in the **KV-Cache** ($2 \times B \times T \times h \times d_k$), which consumes over $1.5 \text{ GB}$ of GPU memory per user session at 8k context length, prompting modern architectures to adopt Grouped-Query Attention (GQA) and FlashAttention (covered in Chapter 9.4).

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `ScaledDotProductAttention`: Pure PyTorch module supporting both bidirectional and causal masking.
2. `MultiHeadAttentionScratch`: Clean, vectorized multi-head attention module.
3. Verification of the Part 5 Visual Grid hand calculations to machine precision.
4. Empirical proof of the Variance Inflation Theorem ($\operatorname{Var}(\mathbf{q}^T \mathbf{k}) \approx d_k$).
5. End-to-end parity validation against PyTorch's `torch.nn.MultiheadAttention` with discrepancy $< 10^{-12}$.

See implementation in:
[`09_transformers_and_llms/code/01_scaled_dot_product_and_mha.py`](./code/01_scaled_dot_product_and_mha.py)
