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

### 2.5 Deep Derivation 8.4.1: Full Analytical Backpropagation Through Classical Attention and the Softmax Jacobian

In this derivation, we derive the exact vector-matrix calculus of backpropagation through the attention mechanism, rigorously evaluating the interaction between the context vector, the Softmax Jacobian, and the query/key projections.

#### Step 1: Forward Attention Computation Graph
At decoding step $t$, let the decoder query vector be $\mathbf{s}_t \in \mathbb{R}^{d_{\text{dec}}}$, and let the collection of $N$ encoder key/value vectors be $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_N] \in \mathbb{R}^{d_{\text{enc}} \times N}$.
For Luong General Attention with learned alignment matrix $\mathbf{W}_a \in \mathbb{R}^{d_{\text{dec}} \times d_{\text{enc}}}$:
1. **Raw Energy Scores:**
   $$e_j = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_j, \quad \forall j \in \{1, \dots, N\} \implies \mathbf{e} = \mathbf{H}^T \mathbf{W}_a^T \mathbf{s}_t \in \mathbb{R}^N$$
2. **Attention Weights (Softmax):**
   $$\boldsymbol{\alpha} = \operatorname{softmax}(\mathbf{e}) \in \mathbb{R}^N, \quad \alpha_j = \frac{\exp(e_j)}{\sum_{k=1}^N \exp(e_k)}$$
3. **Context Vector:**
   $$\mathbf{c}_t = \sum_{j=1}^N \alpha_j \mathbf{h}_j = \mathbf{H} \boldsymbol{\alpha} \in \mathbb{R}^{d_{\text{enc}}}$$

#### Step 2: Backward Pass from Upstream Context Gradient
Let $\boldsymbol{\delta}_t^c \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \in \mathbb{R}^{d_{\text{enc}}}$ be the gradient of the objective loss with respect to the context vector $\mathbf{c}_t$.
By the multivariable chain rule, the gradient w.r.t. the attention distribution vector $\boldsymbol{\alpha}$ is:
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}} = \left( \frac{\partial \mathbf{c}_t}{\partial \boldsymbol{\alpha}} \right)^T \boldsymbol{\delta}_t^c = \mathbf{H}^T \boldsymbol{\delta}_t^c \in \mathbb{R}^N$$
Coordinate-wise: $\frac{\partial \mathcal{L}}{\partial \alpha_j} = \langle \boldsymbol{\delta}_t^c, \mathbf{h}_j \rangle = (\boldsymbol{\delta}_t^c)^T \mathbf{h}_j$.

#### Step 3: Exact Evaluation through the Softmax Jacobian Matrix
The mapping $\mathbf{e} \mapsto \boldsymbol{\alpha} = \operatorname{softmax}(\mathbf{e})$ has Jacobian matrix $\mathbf{J}_{\text{softmax}} \in \mathbb{R}^{N \times N}$:
$$\frac{\partial \alpha_j}{\partial e_k} = \begin{cases} \alpha_j (1 - \alpha_j) & \text{if } j = k \\ -\alpha_j \alpha_k & \text{if } j \neq k \end{cases} = \alpha_j \left( \delta_{j, k} - \alpha_k \right)$$
In matrix notation:
$$\mathbf{J}_{\text{softmax}} = \operatorname{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha} \boldsymbol{\alpha}^T$$

Applying the chain rule to obtain the gradient w.r.t. the pre-softmax scores $\mathbf{e}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{e}} = \mathbf{J}_{\text{softmax}}^T \frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}} = \left( \operatorname{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha} \boldsymbol{\alpha}^T \right) \left( \mathbf{H}^T \boldsymbol{\delta}_t^c \right)$$

Coordinate-wise:
$$\frac{\partial \mathcal{L}}{\partial e_k} = \sum_{j=1}^N \frac{\partial \mathcal{L}}{\partial \alpha_j} \frac{\partial \alpha_j}{\partial e_k} = \sum_{j=1}^N \left( (\boldsymbol{\delta}_t^c)^T \mathbf{h}_j \right) \alpha_j (\delta_{j, k} - \alpha_k) = \alpha_k (\boldsymbol{\delta}_t^c)^T \mathbf{h}_k - \alpha_k \sum_{j=1}^N \alpha_j (\boldsymbol{\delta}_t^c)^T \mathbf{h}_j$$
Recognizing that $\sum_{j=1}^N \alpha_j \mathbf{h}_j = \mathbf{c}_t$:
$$\frac{\partial \mathcal{L}}{\partial e_k} = \alpha_k \left[ (\boldsymbol{\delta}_t^c)^T \mathbf{h}_k - (\boldsymbol{\delta}_t^c)^T \mathbf{c}_t \right] = \alpha_k \, (\boldsymbol{\delta}_t^c)^T (\mathbf{h}_k - \mathbf{c}_t)$$

**Geometric Meaning:**
The gradient w.r.t. the score $e_k$ is proportional to how much encoder state $\mathbf{h}_k$ deviates from the current mean context vector $\mathbf{c}_t$, projected along the error direction $\boldsymbol{\delta}_t^c$!

#### Step 4: Gradients w.r.t. Parameters, Query, and Keys
1. **Gradient w.r.t. Bilinear Weight Matrix $\mathbf{W}_a$:**
   Since $e_k = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_k$:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_a} = \sum_{k=1}^N \frac{\partial \mathcal{L}}{\partial e_k} \mathbf{s}_t \mathbf{h}_k^T = \mathbf{s}_t \left( \sum_{k=1}^N \frac{\partial \mathcal{L}}{\partial e_k} \mathbf{h}_k^T \right) \in \mathbb{R}^{d_{\text{dec}} \times d_{\text{enc}}}$$
2. **Gradient w.r.t. Decoder Query State $\mathbf{s}_t$:**
   $$\frac{\partial \mathcal{L}_{\text{attn}}}{\partial \mathbf{s}_t} = \mathbf{W}_a \left( \sum_{k=1}^N \frac{\partial \mathcal{L}}{\partial e_k} \mathbf{h}_k \right) \in \mathbb{R}^{d_{\text{dec}}}$$
3. **Gradient w.r.t. Encoder Annotations $\mathbf{h}_k$:**
   Each $\mathbf{h}_k$ influences the loss through two simultaneous pathways—the context sum and the score computation:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_k} = \underbrace{\alpha_k \boldsymbol{\delta}_t^c}_{\text{Direct Context Contribution}} + \underbrace{\frac{\partial \mathcal{L}}{\partial e_k} \mathbf{W}_a^T \mathbf{s}_t}_{\text{Indirect Alignment Scoring Contribution}} \in \mathbb{R}^{d_{\text{enc}}}$$
This establishes the complete, exact closed-form backpropagation pass for attention layers. $\blacksquare$

---

### 2.6 Deep Derivation 8.4.2: Information-Theoretic Capacity Limits of the Fixed-Vector Bottleneck

In this derivation, we prove using Shannon's Rate-Distortion Theory why fixed-vector representations fail catastrophically for long sequences.

#### Step 1: Information Transmission Across a Hidden Bottleneck
Let a source sequence of $T$ tokens $\mathbf{X} = (X_1, X_2, \dots, X_T)$ be modeled as a discrete stochastic process with alphabet $\mathcal{V}$ and entropy rate:
$$\mathcal{H}(\mathcal{X}) = \lim_{T \to \infty} \frac{1}{T} H(X_1, \dots, X_T) > 0 \text{ bits/token}$$
The total information content of the sequence is $H(\mathbf{X}) \approx T \cdot \mathcal{H}(\mathcal{X})$ bits.

In classic Seq2Seq, the encoder compresses $\mathbf{X}$ into a single real-valued continuous vector $\mathbf{h}_T \in \mathbb{R}^d$.
In any physical digital computer or noisy neural channel, continuous activations are subject to finite precision (e.g., IEEE 754 float32) or bounded signal-to-noise ratio (SNR) due to noise perturbations $\epsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$.

By Shannon's Channel Capacity Theorem for Gaussian channels, the maximum mutual information $I(\mathbf{X}; \mathbf{h}_T)$ that can be stored in a $d$-dimensional vector with power constraint $P = \mathbb{E}[\|\mathbf{h}_T\|_2^2] \le d \cdot \sigma_h^2$ is bounded by:
$$C_{\text{vector}} = \frac{d}{2} \log_2\left(1 + \frac{P}{d \cdot \sigma_\epsilon^2}\right) = \frac{d}{2} \log_2(1 + \text{SNR}) \text{ bits}$$

#### Step 2: The Inevitable Information Deficit
Notice the critical dimensional asymmetry:
- Information content of source sentence: grows **linearly** with length: $H(\mathbf{X}) \sim \mathcal{O}(T)$.
- Information capacity of fixed vector $\mathbf{h}_T$: is **strictly constant**: $C_{\text{vector}} \sim \mathcal{O}(d)$.

For any sequence length $T$ exceeding the critical threshold:
$$T > T_{\text{crit}} \equiv \frac{d \log_2(1 + \text{SNR})}{2 \, \mathcal{H}(\mathcal{X})}$$
the source entropy strictly exceeds the channel capacity:
$$H(\mathbf{X}) > C_{\text{vector}}$$

#### Step 3: Rate-Distortion Lower Bound on Reconstruction Error
By the Data Processing Inequality and Shannon's Rate-Distortion Theorem, for any distortion measure $\mathcal{D}(\mathbf{X}, \hat{\mathbf{X}}) = \mathbb{E}[d(\mathbf{X}, \hat{\mathbf{X}})]$, the minimum distortion $D^*$ achievable by the decoder satisfies:
$$R(D^*) \le I(\mathbf{X}; \mathbf{h}_T) \le C_{\text{vector}}$$
where $R(D)$ is the rate-distortion function of the source.

For a memoryless source with variance $\sigma_X^2$ under mean-squared error distortion:
$$R(D) = \frac{T}{2} \log_2\left(\frac{\sigma_X^2}{D}\right) \le C_{\text{vector}}$$
Exponentiating both sides:
$$D^* \ge \sigma_X^2 \cdot 2^{-\frac{2 C_{\text{vector}}}{T}} = \sigma_X^2 \cdot \exp\left( - \frac{d \ln(1 + \text{SNR})}{T} \right)$$
As sequence length $T \to \infty$:
$$\lim_{T \to \infty} D^* = \sigma_X^2 \cdot e^0 = \sigma_X^2$$
The reconstruction error approaches $100\%$ of the signal variance. The decoder is mathematically unable to reconstruct the sequence.

**Why Attention Solves the Dilemma:**
By retaining the full matrix of intermediate states $\mathbf{H} \in \mathbb{R}^{d \times T}$, the channel capacity becomes:
$$C_{\text{attention}} = T \cdot \frac{d}{2} \log_2(1 + \text{SNR}) \sim \mathcal{O}(T \cdot d)$$
The memory channel scales linearly with sequence length $T$, permanently eliminating the rate-distortion bottleneck. $\blacksquare$

---

### 2.7 Deep Derivation 8.4.3: Coverage Vectors and Attention Over-Allocation Dynamics

Standard classical attention decoders frequently suffer from two pathological failure modes in long-text generation:
1. **Repetition:** The decoder repeatedly translates the same source phrase multiple times.
2. **Under-translation (Omission):** The decoder skips critical source clauses completely.

Here we derive the mathematical formulation of the **Coverage Mechanism** (Tu et al., 2016; See et al., 2017) that guarantees source token consumption.

#### Step 1: Definition of the Cumulative Coverage Vector
Let $\alpha_{t, j}$ be the attention weight assigned to source token $j \in \{1, \dots, T_x\}$ at decoding step $t \in \{1, \dots, T_y\}$.
Define the **Coverage Vector** $\mathbf{c}^t \in \mathbb{R}^{T_x}$ at step $t$ as the cumulative sum of attention distributions over all past decoding steps:
$$c_j^t \equiv \sum_{\tau=1}^{t-1} \alpha_{\tau, j}$$
$c_j^t$ represents the total amount of attention weight source token $j$ has received up to the current step.

#### Step 2: Coverage-Aware Energy Scoring
In standard Bahdanau attention, the alignment energy is $e_{t, j} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j)$.
In a coverage-augmented model, the historical coverage $c_j^t$ is added directly into the alignment scoring function via a learnable weight vector $\mathbf{w}_c \in \mathbb{R}^{d_a}$:
$$e_{t, j} = \mathbf{v}_a^T \tanh\left( \mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j + \mathbf{w}_c \, c_j^t \right)$$

If token $j$ has already been heavily attended to in past steps ($c_j^t \ge 1.0$), the trained weight vector forces $e_{t, j}$ downward, naturally discouraging the model from re-attending to the same token.

#### Step 3: The Auxiliary Coverage Loss
To enforce that every source token is attended to approximately once, an auxiliary **Coverage Loss** $\mathcal{L}_{\text{cov}}$ is penalized during training:
$$\mathcal{L}_{\text{cov}}^t = \sum_{j=1}^{T_x} \min\left( \alpha_{t, j}, \, c_j^t \right)$$
The total objective function becomes:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} + \lambda_{\text{cov}} \sum_{t=1}^{T_y} \mathcal{L}_{\text{cov}}^t$$

**Mathematical Proof of Repetition Suppression:**
1. If token $j$ is being attended to for the first time ($c_j^t \approx 0$), then $\min(\alpha_{t, j}, c_j^t) \approx 0$, incurring zero penalty.
2. If token $j$ was already translated in previous steps ($c_j^t \ge 1.0$), and the model attends to it again ($\alpha_{t, j} > 0$), then:
   $$\min(\alpha_{t, j}, c_j^t) = \alpha_{t, j} > 0$$
   The model incurs an immediate, linear gradient penalty proportional to $\alpha_{t, j}$:
   $$\frac{\partial \mathcal{L}_{\text{cov}}^t}{\partial \alpha_{t, j}} = 1.0$$
This forces $\alpha_{t, j} \to 0$ for all previously translated words, mathematically guaranteeing uniform coverage across the source sentence. $\blacksquare$

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

### Illustration 3: Complete Hand Trace of Bahdanau Additive Attention with Backward Gradients

**Problem:**
A Bahdanau additive attention layer aligns decoder hidden state $\mathbf{s}_{t-1} \in \mathbb{R}^2$ with two encoder hidden states $\mathbf{h}_1, \mathbf{h}_2 \in \mathbb{R}^2$.
Given:
$$\mathbf{s}_{t-1} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}, \quad \mathbf{h}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{h}_2 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$
$$\mathbf{W}_a = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad \mathbf{U}_a = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}, \quad \mathbf{v}_a = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$$
1. Calculate the pre-activations $\mathbf{z}_j = \mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j$, non-linearities $\mathbf{a}_j = \tanh(\mathbf{z}_j)$, scalar alignment scores $e_j = \mathbf{v}_a^T \mathbf{a}_j$, softmax weights $\boldsymbol{\alpha}$, and dynamic context vector $\mathbf{c}_t$.
2. Given upstream loss gradient $\boldsymbol{\delta}_t^c = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$, backpropagate step-by-step through the softmax Jacobian to evaluate $\frac{\partial \mathcal{L}}{\partial \mathbf{e}}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_j}$, parameter gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_a}$, and decoder gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{s}_{t-1}}$.

**Solution:**

#### Step 1: Forward Pass Calculations

1. **Decoder Projection:**
   $$\mathbf{W}_a \mathbf{s}_{t-1} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

2. **Encoder Projections and Pre-activations:**
   - For position $j=1$:
     $$\mathbf{U}_a \mathbf{h}_1 = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix}$$
     $$\mathbf{z}_1 = \mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_1 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} + \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix}$$
     $$\mathbf{a}_1 = \tanh(\mathbf{z}_1) = \begin{bmatrix} \tanh(1.0) \\ \tanh(-0.5) \end{bmatrix} = \begin{bmatrix} 0.761594 \\ -0.462117 \end{bmatrix}$$
     $$e_1 = \mathbf{v}_a^T \mathbf{a}_1 = [1.0, 1.0] \begin{bmatrix} 0.761594 \\ -0.462117 \end{bmatrix} = 0.761594 - 0.462117 = \mathbf{0.299477}$$

   - For position $j=2$:
     $$\mathbf{U}_a \mathbf{h}_2 = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.5 \end{bmatrix}$$
     $$\mathbf{z}_2 = \mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_2 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix}$$
     $$\mathbf{a}_2 = \tanh(\mathbf{z}_2) = \begin{bmatrix} \tanh(0.5) \\ \tanh(0.0) \end{bmatrix} = \begin{bmatrix} 0.462117 \\ 0.000000 \end{bmatrix}$$
     $$e_2 = \mathbf{v}_a^T \mathbf{a}_2 = [1.0, 1.0] \begin{bmatrix} 0.462117 \\ 0.000000 \end{bmatrix} = \mathbf{0.462117}$$

3. **Softmax Probabilities:**
   $$\exp(e_1) = \exp(0.299477) = 1.349152$$
   $$\exp(e_2) = \exp(0.462117) = 1.587431$$
   $$\sum_{k=1}^2 \exp(e_k) = 1.349152 + 1.587431 = 2.936583$$
   $$\alpha_1 = \frac{1.349152}{2.936583} = \mathbf{0.459429}, \quad \alpha_2 = \frac{1.587431}{2.936583} = \mathbf{0.540571}$$
   *(Verification: $0.459429 + 0.540571 = 1.000000$)*

4. **Dynamic Context Vector $\mathbf{c}_t$:**
   $$\mathbf{c}_t = \alpha_1 \mathbf{h}_1 + \alpha_2 \mathbf{h}_2 = 0.459429 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} + 0.540571 \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.459429} \\ \mathbf{0.540571} \end{bmatrix}$$

---

#### Step 2: Backward Sensitivity Pass

1. **Gradient w.r.t. Attention Probabilities $\alpha_j$:**
   $$\frac{\partial \mathcal{L}}{\partial \alpha_j} = (\boldsymbol{\delta}_t^c)^T \mathbf{h}_j$$
   $$\frac{\partial \mathcal{L}}{\partial \alpha_1} = [1.0, 2.0] \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = 1.000000$$
   $$\frac{\partial \mathcal{L}}{\partial \alpha_2} = [1.0, 2.0] \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = 2.000000$$

2. **Backpropagation through Softmax Jacobian to Alignment Scores $e_j$:**
   The projected inner product is:
   $$\boldsymbol{\alpha}^T \frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}} = (0.459429)(1.0) + (0.540571)(2.0) = 0.459429 + 1.081142 = 1.540571$$
   Applying $\frac{\partial \mathcal{L}}{\partial e_j} = \alpha_j \left( \frac{\partial \mathcal{L}}{\partial \alpha_j} - \boldsymbol{\alpha}^T \frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}} \right)$:
   $$\frac{\partial \mathcal{L}}{\partial e_1} = 0.459429 (1.000000 - 1.540571) = 0.459429(-0.540571) = \mathbf{-0.248354}$$
   $$\frac{\partial \mathcal{L}}{\partial e_2} = 0.540571 (2.000000 - 1.540571) = 0.540571(0.459429) = \mathbf{+0.248354}$$
   *(Verification: $\sum_j \frac{\partial \mathcal{L}}{\partial e_j} = -0.248354 + 0.248354 = 0$, adhering exactly to the probability simplex tangent space).*

3. **Gradient w.r.t. Pre-activations $\mathbf{z}_j$:**
   Using $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_j} = \frac{\partial \mathcal{L}}{\partial e_j} \operatorname{diag}(1 - \mathbf{a}_j^2) \mathbf{v}_a$:
   - For $j=1$:
     $$1 - \mathbf{a}_1^2 = \begin{bmatrix} 1 - (0.761594)^2 \\ 1 - (-0.462117)^2 \end{bmatrix} = \begin{bmatrix} 1 - 0.580025 \\ 1 - 0.213552 \end{bmatrix} = \begin{bmatrix} 0.419975 \\ 0.786448 \end{bmatrix}$$
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = -0.248354 \begin{bmatrix} 0.419975 \\ 0.786448 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.104302} \\ \mathbf{-0.195318} \end{bmatrix}$$
   - For $j=2$:
     $$1 - \mathbf{a}_2^2 = \begin{bmatrix} 1 - (0.462117)^2 \\ 1 - (0.000000)^2 \end{bmatrix} = \begin{bmatrix} 0.786448 \\ 1.000000 \end{bmatrix}$$
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} = +0.248354 \begin{bmatrix} 0.786448 \\ 1.000000 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.195318} \\ \mathbf{+0.248354} \end{bmatrix}$$

4. **Gradient w.r.t. Weight Matrix $\mathbf{W}_a$ and Decoder State $\mathbf{s}_{t-1}$:**
   Summing pre-activation errors:
   $$\boldsymbol{\delta}_z = \sum_{j=1}^2 \frac{\partial \mathcal{L}}{\partial \mathbf{z}_j} = \begin{bmatrix} -0.104302 + 0.195318 \\ -0.195318 + 0.248354 \end{bmatrix} = \begin{bmatrix} 0.091016 \\ 0.053036 \end{bmatrix}$$
   Then the outer product gives:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_a} = \boldsymbol{\delta}_z \mathbf{s}_{t-1}^T = \begin{bmatrix} 0.091016 \\ 0.053036 \end{bmatrix} \begin{bmatrix} 0.5 & -0.5 \end{bmatrix} = \begin{bmatrix} \mathbf{0.045508} & \mathbf{-0.045508} \\ \mathbf{0.026518} & \mathbf{-0.026518} \end{bmatrix}$$
   The decoder adjoint is:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{s}_{t-1}} = \mathbf{W}_a^T \boldsymbol{\delta}_z = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.091016 \\ 0.053036 \end{bmatrix} = \begin{bmatrix} \mathbf{0.091016} \\ \mathbf{0.053036} \end{bmatrix}$$

---

### Illustration 4: Luong General (Bilinear) Attention Step-by-Step with Learned Parameter Matrix $\mathbf{W}_a$

**Problem:**
A Luong attention layer uses general bilinear scoring $e_j = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_j$ between decoder query $\mathbf{s}_t = [1.0, 2.0]^T$ and encoder memory keys $\mathbf{h}_1 = [2.0, -1.0]^T, \mathbf{h}_2 = [0.0, 1.0]^T$.
The bilinear weight parameter matrix is:
$$\mathbf{W}_a = \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.5 \end{bmatrix}$$
1. Compute the bilinear transformation $\mathbf{s}_t^T \mathbf{W}_a$, alignment scores $e_1, e_2$, attention weights $\boldsymbol{\alpha}$, and context vector $\mathbf{c}_t$.
2. Given upstream loss gradient $\boldsymbol{\delta}_t^c = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}$, compute the exact analytical gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_a}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{s}_t}$.

**Solution:**

#### Step 1: Forward Attention Pass

1. **Bilinear Query Projection:**
   $$\mathbf{m}^T = \mathbf{s}_t^T \mathbf{W}_a = [1.0, 2.0] \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.5 \end{bmatrix}$$
   $$m_1 = (1.0)(0.5) + (2.0)(-0.5) = 0.5 - 1.0 = -0.5$$
   $$m_2 = (1.0)(1.0) + (2.0)(0.5) = 1.0 + 1.0 = 2.0$$
   $$\mathbf{m}^T = [-0.5, 2.0]$$

2. **Alignment Scores:**
   $$e_1 = \mathbf{m}^T \mathbf{h}_1 = [-0.5, 2.0] \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} = (-0.5)(2.0) + (2.0)(-1.0) = -1.0 - 2.0 = \mathbf{-3.0}$$
   $$e_2 = \mathbf{m}^T \mathbf{h}_2 = [-0.5, 2.0] \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = (-0.5)(0.0) + (2.0)(1.0) = 0.0 + 2.0 = \mathbf{+2.0}$$

3. **Softmax Normalization:**
   $$\exp(e_1) = \exp(-3.0) = 0.049787$$
   $$\exp(e_2) = \exp(2.0) = 7.389056$$
   $$\sum_{k=1}^2 \exp(e_k) = 0.049787 + 7.389056 = 7.438843$$
   $$\alpha_1 = \frac{0.049787}{7.438843} = \mathbf{0.006693}, \quad \alpha_2 = \frac{7.389056}{7.438843} = \mathbf{0.993307}$$

4. **Context Vector $\mathbf{c}_t$:**
   $$\mathbf{c}_t = \alpha_1 \mathbf{h}_1 + \alpha_2 \mathbf{h}_2 = 0.006693 \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} + 0.993307 \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.013386 \\ -0.006693 + 0.993307 \end{bmatrix} = \begin{bmatrix} \mathbf{0.013386} \\ \mathbf{0.986614} \end{bmatrix}$$

---

#### Step 2: Backward Sensitivity Pass

1. **Gradient w.r.t. Attention Probabilities:**
   $$\frac{\partial \mathcal{L}}{\partial \alpha_1} = (\boldsymbol{\delta}_t^c)^T \mathbf{h}_1 = [2.0, 1.0] \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} = 4.0 - 1.0 = 3.000000$$
   $$\frac{\partial \mathcal{L}}{\partial \alpha_2} = (\boldsymbol{\delta}_t^c)^T \mathbf{h}_2 = [2.0, 1.0] \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = 1.000000$$

2. **Gradient w.r.t. Alignment Scores $e_j$:**
   Weighted average gradient:
   $$\boldsymbol{\alpha}^T \frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}} = (0.006693)(3.0) + (0.993307)(1.0) = 0.020079 + 0.993307 = 1.013386$$
   Applying the Softmax Jacobian:
   $$\frac{\partial \mathcal{L}}{\partial e_1} = \alpha_1 \left( 3.0 - 1.013386 \right) = 0.006693(1.986614) = \mathbf{+0.013296}$$
   $$\frac{\partial \mathcal{L}}{\partial e_2} = \alpha_2 \left( 1.0 - 1.013386 \right) = 0.993307(-0.013386) = \mathbf{-0.013296}$$

3. **Gradient w.r.t. Bilinear Weight Matrix $\mathbf{W}_a$:**
   Since $e_j = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_j = \operatorname{tr}(\mathbf{W}_a \mathbf{h}_j \mathbf{s}_t^T)$, the derivative is:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_a} = \sum_{j=1}^2 \frac{\partial \mathcal{L}}{\partial e_j} \mathbf{s}_t \mathbf{h}_j^T = \mathbf{s}_t \left( \sum_{j=1}^2 \frac{\partial \mathcal{L}}{\partial e_j} \mathbf{h}_j \right)^T$$
   Evaluating the weighted encoder state sum $\mathbf{u}$:
   $$\mathbf{u} = \frac{\partial \mathcal{L}}{\partial e_1} \mathbf{h}_1 + \frac{\partial \mathcal{L}}{\partial e_2} \mathbf{h}_2 = (+0.013296) \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} + (-0.013296) \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.026592 \\ -0.026592 \end{bmatrix}$$
   Forming the outer product with $\mathbf{s}_t = [1.0, 2.0]^T$:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_a} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} [0.026592, -0.026592] = \begin{bmatrix} \mathbf{0.026592} & \mathbf{-0.026592} \\ \mathbf{0.053184} & \mathbf{-0.053184} \end{bmatrix}$$

4. **Gradient w.r.t. Decoder State $\mathbf{s}_t$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{s}_t} = \mathbf{W}_a \mathbf{u} = \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.5 \end{bmatrix} \begin{bmatrix} 0.026592 \\ -0.026592 \end{bmatrix} = \begin{bmatrix} (0.5)(0.026592) + (1.0)(-0.026592) \\ (-0.5)(0.026592) + (0.5)(-0.026592) \end{bmatrix} = \begin{bmatrix} \mathbf{-0.013296} \\ \mathbf{-0.026592} \end{bmatrix}$$

---

### Illustration 5: Beam Search Decoding Arithmetic on a 3-Step Vocabulary

**Problem:**
A trained attentional sequence model decodes over a vocabulary $V = \{A, B, \text{EOS}\}$ using beam search with width $K = 2$ and length penalty exponent $\alpha = 0.7$.
The scoring objective is:
$$S_{\text{norm}}(\mathbf{y}_{1:T}) = \frac{1}{T^\alpha} \sum_{t=1}^T \log P(y_t \mid y_{<t}, \mathbf{x})$$
Given conditional step probabilities:
- Step 1 ($t=1$ from $\text{BOS}$): $P(A) = 0.60, P(B) = 0.35, P(\text{EOS}) = 0.05$.
- Step 2 ($t=2$):
  - From $A$: $P(A \mid A) = 0.20, P(B \mid A) = 0.70, P(\text{EOS} \mid A) = 0.10$.
  - From $B$: $P(A \mid B) = 0.50, P(B \mid B) = 0.10, P(\text{EOS} \mid B) = 0.40$.
- Step 3 ($t=3$):
  - From $[A, B]$: $P(A \mid A, B) = 0.10, P(B \mid A, B) = 0.10, P(\text{EOS} \mid A, B) = 0.80$.
  - From $[B, A]$: $P(A \mid B, A) = 0.15, P(B \mid B, A) = 0.25, P(\text{EOS} \mid B, A) = 0.60$.

Perform the complete beam search trace: expand candidates, prune to top-$K$, apply length normalization upon completion, and identify the final winning sequence.

**Solution:**

#### Step 1: Initialization ($t = 1$)

Convert probabilities to log-probabilities:
$$\log P(A) = \ln(0.60) = -0.5108$$
$$\log P(B) = \ln(0.35) = -1.0498$$
$$\log P(\text{EOS}) = \ln(0.05) = -2.9957$$

Rank hypotheses:
1. Hypothesis 1: $[A]$ with cumulative log-prob $-0.5108$
2. Hypothesis 2: $[B]$ with cumulative log-prob $-1.0498$
*(Prune $[ \text{EOS} ]$ as $K = 2$)*.

Beam active set $\mathcal{B}_1 = \{ [A], [B] \}$.

---

#### Step 2: Expansion at $t = 2$

Expand each active beam candidate across all 3 vocabulary tokens ($K \times |V| = 2 \times 3 = 6$ paths):

1. **Expanding $[A]$ (prior score $-0.5108$):**
   - $[A, A]$: $\text{score} = -0.5108 + \ln(0.20) = -0.5108 - 1.6094 = \mathbf{-2.1202}$
   - $[A, B]$: $\text{score} = -0.5108 + \ln(0.70) = -0.5108 - 0.3567 = \mathbf{-0.8675}$
   - $[A, \text{EOS}]$: $\text{score} = -0.5108 + \ln(0.10) = -0.5108 - 2.3026 = \mathbf{-2.8134}$

2. **Expanding $[B]$ (prior score $-1.0498$):**
   - $[B, A]$: $\text{score} = -1.0498 + \ln(0.50) = -1.0498 - 0.6931 = \mathbf{-1.7429}$
   - $[B, B]$: $\text{score} = -1.0498 + \ln(0.10) = -1.0498 - 2.3026 = \mathbf{-3.3524}$
   - $[B, \text{EOS}]$: $\text{score} = -1.0498 + \ln(0.40) = -1.0498 - 0.9163 = \mathbf{-1.9661}$

**Global Candidate Ranking across all 6:**
1. $[A, B]$: score $\mathbf{-0.8675}$ (Active)
2. $[B, A]$: score $\mathbf{-1.7429}$ (Active)
3. $[B, \text{EOS}]$: score $\mathbf{-1.9661}$ (Completed $\to$ move to completed list)
4. $[A, A]$: score $-2.1202$ (Pruned)
5. $[A, \text{EOS}]$: score $-2.8134$ (Pruned)
6. $[B, B]$: score $-3.3524$ (Pruned)

The active beam for $t=3$ is $\mathcal{B}_2 = \{ [A, B], [B, A] \}$.
Completed pool: $\mathcal{C} = \{ [B, \text{EOS}] \text{ (len } 2\text{, score } -1.9661) \}$.

---

#### Step 3: Expansion at $t = 3$

Expand active candidates:

1. **Expanding $[A, B]$ (prior score $-0.8675$):**
   - $[A, B, A]$: $\text{score} = -0.8675 + \ln(0.10) = -0.8675 - 2.3026 = -3.1701$
   - $[A, B, B]$: $\text{score} = -0.8675 + \ln(0.10) = -0.8675 - 2.3026 = -3.1701$
   - $[A, B, \text{EOS}]$: $\text{score} = -0.8675 + \ln(0.80) = -0.8675 - 0.2231 = \mathbf{-1.0906}$

2. **Expanding $[B, A]$ (prior score $-1.7429$):**
   - $[B, A, A]$: $\text{score} = -1.7429 + \ln(0.15) = -1.7429 - 1.8971 = -3.6400$
   - $[B, A, B]$: $\text{score} = -1.7429 + \ln(0.25) = -1.7429 - 1.3863 = -3.1292$
   - $[B, A, \text{EOS}]$: $\text{score} = -1.7429 + \ln(0.60) = -1.7429 - 0.5108 = \mathbf{-2.2537}$

All promising branches have produced $\text{EOS}$. The completed hypotheses are:
- Hypothesis 1: $\mathbf{y}^{(1)} = [B, \text{EOS}]$, length $T = 2$, cumulative log-prob $= -1.9661$
- Hypothesis 2: $\mathbf{y}^{(2)} = [A, B, \text{EOS}]$, length $T = 3$, cumulative log-prob $= -1.0906$
- Hypothesis 3: $\mathbf{y}^{(3)} = [B, A, \text{EOS}]$, length $T = 3$, cumulative log-prob $= -2.2537$

---

#### Step 4: Length Normalization and Final Selection

Evaluate $S_{\text{norm}}(\mathbf{y}) = \frac{\sum \log P}{T^{0.7}}$:
- Normalization factors:
  $$2^{0.7} = 1.6245, \quad 3^{0.7} = 2.1577$$

- Normalized scores:
  $$S_{\text{norm}}(\mathbf{y}^{(1)}) = \frac{-1.9661}{1.6245} = \mathbf{-1.2103}$$
  $$S_{\text{norm}}(\mathbf{y}^{(2)}) = \frac{-1.0906}{2.1577} = \mathbf{-0.5054}$$
  $$S_{\text{norm}}(\mathbf{y}^{(3)}) = \frac{-2.2537}{2.1577} = \mathbf{-1.0445}$$

**Conclusion:**
Hypothesis $\mathbf{y}^{(2)} = [A, B, \text{EOS}]$ wins convincingly with normalized score $\mathbf{-0.5054}$.
Notice that although $[B, \text{EOS}]$ completed early at step 2, beam search avoided premature greedy commitment and preserved the longer, vastly higher-probability path $[A, B, \text{EOS}]$ while penalizing sequence length proportionally.

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
