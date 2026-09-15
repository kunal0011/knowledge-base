# Chapter 8.4: Sequence-to-Sequence Models & Classical Attention (Bahdanau, Luong)

---

## 1. Intuition & 101 Motivation

In 2014, Ilya Sutskever, Oriol Vinyals, and Quoc Le published the foundational **Sequence-to-Sequence (Seq2Seq)** framework, establishing that recurrent neural networks could map sequences of arbitrary length to output sequences of completely different lengths—transforming Machine Translation, Speech Recognition, and Text Summarization.

However, the classic Seq2Seq architecture suffered from a catastrophic mathematical flaw: **The Fixed-Vector Information Bottleneck**.

```
             THE CLASSIC SEQ2SEQ INFORMATION BOTTLENECK
  Source Tokens: [ x_1 ] ──► [ x_2 ] ──► ... ──► [ x_T ]
                                                    │
                                                    ▼
                           Single Fixed Vector ──► [ c = h_T ]  <-- Catastrophic Bottleneck!
                                                    │
                                                    ▼
  Target Tokens: [ y_1 ] ◄── [ y_2 ] ◄── ... ◄── [ y_T' ]
```

### The Information Bottleneck Crisis
In classic Seq2Seq:
1. The **Encoder** consumes the source sentence $(\mathbf{x}_1, \dots, \mathbf{x}_{T_x})$ and discards all intermediate hidden states $(\mathbf{h}_1, \dots, \mathbf{h}_{T_x-1})$.
2. The entire semantic meaning of a 50-word sentence must be compressed into the single final state vector $\mathbf{h}_{T_x} \in \mathbb{R}^{d_h}$.
3. The **Decoder** must generate the entire translation relying solely on this single compressed vector.

As Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio observed, translation quality collapses precipitously for sentences longer than $15-20$ words. An engineer cannot expect a 512-dimensional floating-point vector to preserve every nuance, entity, clause, and syntactic relationship of an entire paragraph.

### The Breakthrough: Classical Attention (Bahdanau 2014, Luong 2015)
Instead of forcing the encoder to compress everything into a single fixed vector:
- The encoder **retains all intermediate representations** $\mathbf{H} = (\mathbf{h}_1, \dots, \mathbf{h}_{T_x})$.
- At every decoding step $i$, the decoder dynamically generates a **Query** based on its current state.
- It compares this query against all encoder hidden states to compute an **alignment distribution** (attention weights $\alpha_{i, 1}, \dots, \alpha_{i, T_x}$).
- It computes a customized, dynamic **Context Vector** $\mathbf{c}_i = \sum_j \alpha_{i, j} \mathbf{h}_j$.
- This dynamic context vector gives the decoder a direct, high-bandwidth window into whichever source words are relevant to the word currently being translated.

This single innovation directly spawned the Transformer revolution and modern Generative AI.

---

## 2. Rigorous Mathematical Formulation

```
                 ATTENTION-AUGMENTED SEQ2SEQ DECODER
                          y_i (Target Output)
                               ▲
                               │
                       [ Attentional State s~_i ]
                               ▲
                     ┌─────────┴─────────┐
                     │                   │
               Decoder State s_i    Context Vector c_i = Sum(alpha_ij * h_j)
                     ▲                   ▲
                     │                   │ [Softmax Weights alpha_ij]
                Decoder Step             │
                 (s_{i-1}, y_{i-1})      ├─── [Score e_i1] ◄── h_1 (Encoder Token 1)
                                         ├─── [Score e_i2] ◄── h_2 (Encoder Token 2)
                                         └─── [Score e_iT] ◄── h_T (Encoder Token T)
```

---

### 2.1 Classic Sequence-to-Sequence (Without Attention)

1. **Encoder:**
   $$\mathbf{h}_t = \text{RNN}_{\text{enc}}(\mathbf{x}_t, \mathbf{h}_{t-1}), \quad t = 1, \dots, T_x$$
   The context vector is static: $\mathbf{c} = \mathbf{h}_{T_x}$.

2. **Decoder:**
   $$\mathbf{s}_i = \text{RNN}_{\text{dec}}([\mathbf{y}_{i-1}; \mathbf{c}], \, \mathbf{s}_{i-1}), \quad i = 1, \dots, T_y$$
   $$\hat{\mathbf{y}}_i = \text{softmax}(\mathbf{W}_y \mathbf{s}_i + \mathbf{b}_y)$$

---

### 2.2 Bahdanau Additive Attention (Bahdanau et al., 2014)

In Bahdanau's formulation, the encoder is typically a **Bidirectional RNN** (BiRNN), yielding forward and backward annotations:
$$\mathbf{h}_j = \begin{bmatrix} \overrightarrow{\mathbf{h}}_j \\ \overleftarrow{\mathbf{h}}_j \end{bmatrix} \in \mathbb{R}^{2d_h}$$

At decoding step $i$, the decoder is at state $\mathbf{s}_{i-1}$.

#### Step 1: Alignment Energy Score
The compatibility between previous decoder state $\mathbf{s}_{i-1}$ and source annotation $\mathbf{h}_j$ is computed via a single-hidden-layer feedforward network:
$$e_{i, j} = \mathbf{v}_a^T \tanh\left( \mathbf{W}_a \mathbf{s}_{i-1} + \mathbf{U}_a \mathbf{h}_j \right) \in \mathbb{R}$$
where:
- $\mathbf{W}_a \in \mathbb{R}^{d_a \times d_{\text{dec}}}$,
- $\mathbf{U}_a \in \mathbb{R}^{d_a \times 2d_h}$,
- $\mathbf{v}_a \in \mathbb{R}^{d_a}$ is a learnable attention projection vector.

#### Step 2: Attention Softmax Normalization
The scalar energies are normalized across all $T_x$ source positions using the softmax operator:
$$\alpha_{i, j} = \frac{\exp(e_{i, j})}{\sum_{k=1}^{T_x} \exp(e_{i, k})}, \quad \sum_{j=1}^{T_x} \alpha_{i, j} = 1$$
Here, $\alpha_{i, j} \in (0, 1)$ represents the probability that the target token $y_i$ aligns with or translates from the source token $x_j$.

#### Step 3: Dynamic Context Vector
The context vector $\mathbf{c}_i$ is the expected source annotation under distribution $\boldsymbol{\alpha}_i$:
$$\mathbf{c}_i = \sum_{j=1}^{T_x} \alpha_{i, j} \mathbf{h}_j \in \mathbb{R}^{2d_h}$$

#### Step 4: Decoder State Update
$$\mathbf{s}_i = \text{RNN}_{\text{dec}}\left( [\mathbf{y}_{i-1}; \mathbf{c}_i], \, \mathbf{s}_{i-1} \right)$$

---

### 2.3 Luong Multiplicative Attention (Luong et al., 2015)

Minh-Thang Luong et al. simplified and expanded the attention framework:
1. They compute attention **after** the current decoder hidden state $\mathbf{s}_t$ is computed, rather than using $\mathbf{s}_{t-1}$.
2. They proposed three alternative alignment scoring functions:

$$\text{score}(\mathbf{s}_t, \mathbf{h}_i) = \begin{cases} 
\mathbf{s}_t^T \mathbf{h}_i & \text{\textbf{Dot Product}} \quad (d_{\text{dec}} = d_{\text{enc}}) \\
\mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_i & \text{\textbf{General (Bilinear)}} \quad (\mathbf{W}_a \in \mathbb{R}^{d_{\text{dec}} \times d_{\text{enc}}}) \\
\mathbf{v}_a^T \tanh(\mathbf{W}_a [\mathbf{s}_t; \mathbf{h}_i]) & \text{\textbf{Concat (Additive)}}
\end{cases}$$

#### The Attentional Hidden State ($\tilde{\mathbf{s}}_t$):
Once the context vector $\mathbf{c}_t = \sum_i \alpha_{t, i} \mathbf{h}_i$ is obtained, Luong combines the context and the decoder state through an attentional layer:
$$\tilde{\mathbf{s}}_t = \tanh\left( \mathbf{W}_c [\mathbf{c}_t; \mathbf{s}_t] + \mathbf{b}_c \right)$$
The final target token probability is emitted from $\tilde{\mathbf{s}}_t$:
$$\hat{\mathbf{y}}_t = \text{softmax}\left( \mathbf{W}_s \tilde{\mathbf{s}}_t + \mathbf{b}_s \right)$$

---

### 2.4 Comparison: Bahdanau vs. Luong

| Feature | Bahdanau (Additive) Attention (2014) | Luong (Multiplicative) Attention (2015) |
| :--- | :--- | :--- |
| **Scoring Function** | $e_{i, j} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{i-1} + \mathbf{U}_a \mathbf{h}_j)$ | Dot: $\mathbf{s}_t^T \mathbf{h}_i$, General: $\mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_i$ |
| **Decoder State Timing** | Uses previous decoder state $\mathbf{s}_{i-1}$ | Uses current decoder state $\mathbf{s}_t$ |
| **Context Integration** | Fed directly as input into decoder RNN | Combined with $\mathbf{s}_t$ to produce attentional state $\tilde{\mathbf{s}}_t$ |
| **Computational Efficiency** | Slower (requires addition + $\tanh$ + dot product) | **Faster** (pure matrix multiplication GEMM) |
| **Impact on Modern AI** | Proved the concept of soft alignment | **Direct precursor to Transformer Dot-Product Attention** |

---

## 3. Geometric & Algebraic Interpretation

### The Convex Hull Interpretation of Context
Because the attention distribution satisfies:
$$\alpha_{i, j} \ge 0 \quad \text{and} \quad \sum_{j=1}^{T_x} \alpha_{i, j} = 1$$
The mathematical context vector $\mathbf{c}_i$ is a **convex combination** of the set of encoder vectors $\{\mathbf{h}_1, \dots, \mathbf{h}_{T_x}\}$.

Geometrically:
$$\mathbf{c}_i \in \operatorname{conv}\left( \{\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_{T_x}\} \right)$$
The context vector $\mathbf{c}_i$ is strictly constrained to lie inside the **polyhedral convex hull** whose vertices are the encoder token embeddings.
- If $\alpha_{i, 1} = 1$ and all others $0$, $\mathbf{c}_i$ rests on vertex $\mathbf{h}_1$ (hard deterministic retrieval).
- If $\alpha_{i, j} = 1/T_x$, $\mathbf{c}_i$ rests at the center of mass (uniform global pooling).
- The attention mechanism acts as a **continuous, differentiable soft pointer** that glides along the surface and interior of this convex polytope.

```
                  CONVEX HULL IN EMBEDDING SPACE
                              h_2
                               ▲
                              / \
                             /   \
                            /  *  \  <-- Context Vector c_i = Sum(alpha_j * h_j)
                           /       \
                          /_________\
                        h_1          h_3
```

---

## 4. Real-World Analogy

### The Closed-Book vs. The Open-Book Examination
- **Classic Seq2Seq (Closed-Book Exam):**
  A student reads a 400-page historical treatise in German. They close the book, put it away, and are asked to write a 10-page analysis in English entirely from memory. They will inevitably forget specific dates, names, and secondary arguments because human short-term memory has a strict capacity limit.
- **Attention Seq2Seq (Open-Book Exam):**
  The student keeps the original German book wide open on their desk.
  - When writing a sentence about the Treaty of Versailles (Query $\mathbf{s}_t$), they glance down at the index, scan page 142 (Attention weights $\boldsymbol{\alpha}$), extract the exact clause (Context $\mathbf{c}_t$), and write the translated passage.
  - Because they can look back at any page at any moment, the length of the book no longer causes memory decay.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete Luong dot-product attention calculation with exact hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Attention Mechanism |
| :--- | :--- | :--- | :--- |
| $T_x$ | Source Sequence Length | Scalar ($3$) | Number of encoder tokens ("The", "black", "cat") |
| $d$ | Hidden Dimension | Scalar ($2$) | Vector dimension of encoder and decoder states |
| $\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3$ | Encoder States | $(2,)$ each | Source token representations |
| $\mathbf{s}_t$ | Decoder Query State | $(2,)$ | Current decoder hidden state seeking information |
| $e_1, e_2, e_3$ | Raw Alignment Scores | Scalar each | Unnormalized dot-product similarities $\mathbf{s}_t^T \mathbf{h}_j$ |
| $\alpha_1, \alpha_2, \alpha_3$ | Attention Weights | Scalar each $\in (0, 1)$ | Softmax-normalized alignment distribution |
| $\mathbf{c}_t$ | Context Vector | $(2,)$ | Dynamic weighted sum $\sum_j \alpha_j \mathbf{h}_j$ |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t}$ | Upstream Context Grad | $(2,)$ | Sensitivity of objective loss w.r.t. context vector |

---

### 5.2 Concrete Toy Numbers

#### Encoder Annotations ($T_x = 3, d = 2$):
$$\mathbf{h}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{h}_2 = \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix}, \quad \mathbf{h}_3 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$$

#### Decoder Query State:
$$\mathbf{s}_t = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$$

---

### 5.3 Step 1: Alignment Energy Scores (Luong Dot-Product)

$$e_j = \mathbf{s}_t^T \mathbf{h}_j$$

1. **Token 1:**
   $$e_1 = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = (1.0)(1.0) + (1.0)(0.0) = \mathbf{1.0}$$
2. **Token 2:**
   $$e_2 = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = (1.0)(0.0) + (1.0)(2.0) = \mathbf{2.0}$$
3. **Token 3:**
   $$e_3 = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = (1.0)(1.0) + (1.0)(1.0) = \mathbf{2.0}$$

$$\mathbf{e} = \begin{bmatrix} 1.0 \\ 2.0 \\ 2.0 \end{bmatrix}$$

---

### 5.4 Step 2: Softmax Attention Distribution ($\boldsymbol{\alpha}$)

Exponentials ($e^1 \approx 2.718282$, $e^2 \approx 7.389056$):
$$\exp(e_1) = 2.718282$$
$$\exp(e_2) = 7.389056$$
$$\exp(e_3) = 7.389056$$

Sum of exponentials:
$$\sum_{j=1}^3 \exp(e_j) = 2.718282 + 7.389056 + 7.389056 = 17.496394$$

Normalized attention probabilities:
$$\alpha_1 = \frac{2.718282}{17.496394} \approx \mathbf{0.155362}$$
$$\alpha_2 = \frac{7.389056}{17.496394} \approx \mathbf{0.422319}$$
$$\alpha_3 = \frac{7.389056}{17.496394} \approx \mathbf{0.422319}$$

$$\boldsymbol{\alpha} = \begin{bmatrix} 0.155362 \\ 0.422319 \\ 0.422319 \end{bmatrix}, \quad \sum \alpha_j = 1.000000$$

---

### 5.5 Step 3: Context Vector Computation ($\mathbf{c}_t = \sum \alpha_j \mathbf{h}_j$)

$$\mathbf{c}_t = \alpha_1 \mathbf{h}_1 + \alpha_2 \mathbf{h}_2 + \alpha_3 \mathbf{h}_3$$

Component 1:
$$c_{t, 1} = (0.155362)(1.0) + (0.422319)(0.0) + (0.422319)(1.0) = 0.155362 + 0.422319 = \mathbf{0.577681}$$

Component 2:
$$c_{t, 2} = (0.155362)(0.0) + (0.422319)(2.0) + (0.422319)(1.0) = 0.844638 + 0.422319 = \mathbf{1.266957}$$

$$\mathbf{c}_t = \begin{bmatrix} 0.577681 \\ 1.266957 \end{bmatrix}$$

The context vector is a concrete synthesis of the entire source sentence, heavily weighting Tokens 2 and 3 while giving Token 1 a modest $15.5\%$ contribution.

---

### 5.6 Step 4: Backward Gradient Flow Through Attention

Suppose the upstream loss gradient w.r.t. the context vector is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$

1. **Gradient w.r.t. Attention Weights $\alpha_j$:**
   $$\frac{\partial \mathcal{L}}{\partial \alpha_j} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t}\right)^T \mathbf{h}_j$$
   $$\frac{\partial \mathcal{L}}{\partial \alpha_1} = [1.0, 0.0] \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = 1.0$$
   $$\frac{\partial \mathcal{L}}{\partial \alpha_2} = [1.0, 0.0] \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = 0.0$$
   $$\frac{\partial \mathcal{L}}{\partial \alpha_3} = [1.0, 0.0] \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = 1.0$$
   $$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}} = \begin{bmatrix} 1.0 \\ 0.0 \\ 1.0 \end{bmatrix}$$

2. **Gradient w.r.t. Encoder Hidden States $\mathbf{h}_j$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_j} = \alpha_j \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} + \frac{\partial \mathcal{L}}{\partial e_j} \frac{\partial e_j}{\partial \mathbf{h}_j}$$
   The direct term is $\alpha_j \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t}$:
   $$\alpha_1 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.155362 \\ 0.0 \end{bmatrix}, \quad \alpha_2 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.422319 \\ 0.0 \end{bmatrix}, \quad \alpha_3 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.422319 \\ 0.0 \end{bmatrix}$$

Every encoder state $\mathbf{h}_j$ receives gradient scaled directly by its attention weight $\alpha_j$, providing an unimpeded error pathway back into the source encoder.

---

## 6. Solved Illustrations

### Illustration 1: Teacher Forcing vs. Beam Search Decoding

**Problem:**
Why is greedy argmax decoding ($\hat{y}_t = \operatorname{argmax}_w p(w \mid y_{<t}, \mathbf{x})$) suboptimal during inference, and how does **Beam Search** resolve this?

**Solution:**
1. **Greedy Decoding:**
   Greedy search selects the single most probable token at each step $t$. However, an early highly probable token might lead to a catastrophic low-probability dead-end, while an early lower-probability token might unlock a vastly higher global sequence probability:
   $$P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{T_y} P(y_t \mid y_{<t}, \mathbf{x})$$
2. **Beam Search Algorithm:**
   Instead of keeping only 1 candidate, Beam Search maintains the **top-$K$ partial hypotheses** (where $K$ is the beam width, typically $K \in [4, 10]$):
   - At step 1, select the top-$K$ initial words.
   - At step $t$, expand all $K$ hypotheses into $K \times |V|$ candidates.
   - Retain only the top-$K$ sequences with highest cumulative log-probabilities:
     $$\text{Score}(\mathbf{y}_{1:t}) = \sum_{k=1}^t \log P(y_k \mid y_{<k}, \mathbf{x})$$
   - To prevent penalizing longer translations, apply length normalization:
     $$\text{Score}_{\text{norm}}(\mathbf{y}) = \frac{1}{T^\alpha} \sum_{k=1}^T \log P(y_k \mid y_{<k}, \mathbf{x}), \quad \alpha \in [0.6, 0.8]$$

---

### Illustration 2: Soft vs. Hard Attention

**Problem:**
Distinguish mathematically between **Soft Attention** and **Hard Attention** (Xu et al., 2015). Why is Soft Attention universally favored in deep learning?

**Solution:**
1. **Soft Attention (Differentiable):**
   The context vector is the expected value under the multinomial distribution:
   $$\mathbf{c} = \mathbb{E}_{j \sim \boldsymbol{\alpha}}[\mathbf{h}_j] = \sum_{j=1}^{T_x} \alpha_j \mathbf{h}_j$$
   Because $\alpha_j = \text{softmax}(e_j)$ is smooth and continuously differentiable everywhere, the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_j}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{s}_t}$ can be computed in closed form via standard backpropagation.
2. **Hard Attention (Stochastic / Non-differentiable):**
   Instead of a weighted sum, Hard Attention treats the choice of position as a discrete latent variable $z \sim \operatorname{Categorical}(\boldsymbol{\alpha})$:
   $$\mathbf{c} = \mathbf{h}_z, \quad z \in \{1, \dots, T_x\}$$
   Because selecting a discrete index is non-differentiable, backpropagation cannot compute $\frac{\partial \mathbf{c}}{\partial \boldsymbol{\alpha}}$. One must resort to Reinforcement Learning policy gradients (e.g., REINFORCE) or Gumbel-Softmax relaxations, which suffer from high variance and slow training convergence.

---

## 7. Deep Learning Connection & Application

### The Direct Lineage to Transformer Scaled Dot-Product Attention
Every single concept of the modern Transformer's self-attention mechanism is directly inherited from Luong attention:

| Seq2Seq Classical Attention (Luong 2015) | Modern Transformer Attention (Vaswani 2017) |
| :--- | :--- |
| Decoder Hidden State $\mathbf{s}_t$ | **Query ($\mathbf{Q}$):** $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$ |
| Encoder Hidden States $\mathbf{h}_i$ (for scoring) | **Key ($\mathbf{K}$):** $\mathbf{K} = \mathbf{X} \mathbf{W}_K$ |
| Encoder Hidden States $\mathbf{h}_i$ (for context) | **Value ($\mathbf{V}$):** $\mathbf{V} = \mathbf{X} \mathbf{W}_V$ |
| Alignment Score: $e_{i, j} = \mathbf{s}_i^T \mathbf{h}_j$ | **Scaled Score:** $\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}$ |
| Softmax Weights: $\alpha_{i, j} = \text{softmax}(e)$ | **Attention Matrix:** $\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right)$ |
| Dynamic Context: $\mathbf{c} = \sum \alpha_j \mathbf{h}_j$ | **Output Representation:** $\mathbf{O} = \mathbf{A} \mathbf{V}$ |

The Transformer's revolutionary step was merely recognizing that recurrent networks were unnecessary: one could simply use attention to attend to the sequence *itself* (Self-Attention)!

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `BahdanauAttention`: Pure PyTorch module computing additive attention alignment.
2. `LuongAttention`: Pure PyTorch module supporting dot, general, and concat scoring functions.
3. Verification of the Part 5 Visual Grid hand arithmetic to machine precision.
4. Complete Seq2Seq Translation Model with Attention (Bidirectional GRU Encoder + Attentional GRU Decoder).
5. Attention weight matrix extraction and alignment verification.

See implementation in:
[`08_recurrent_networks/code/04_seq2seq_and_attention.py`](./code/04_seq2seq_and_attention.py)
