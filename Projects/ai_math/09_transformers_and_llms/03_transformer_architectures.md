# Chapter 9.3: Transformer Architectures (BERT, GPT, T5)

---

## 1. Intuition & 101 Motivation

The original 2017 Transformer was designed specifically for sequence-to-sequence neural machine translation, consisting of an **Encoder** coupled to a **Decoder** via cross-attention.

However, in the late 2010s, researchers discovered that the two halves of the Transformer could be cleaved apart and specialized for distinct learning objectives. This led to **The Great Architectural Divergence**:

```
                       THE THREE TRANSFORMER ARCHETYPES
┌───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│     ENCODER-ONLY (BERT)       │     DECODER-ONLY (GPT)        │    ENCODER-DECODER (T5)       │
├───────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│  • Full Bidirectional         │  • Causal (Autoregressive)    │  • Bidirectional Encoder      │
│  • Masked Language Model      │  • Next-Token Prediction      │  • Autoregressive Decoder     │
│  • Every token attends to all │  • Tokens attend only to past │  • Coupled via Cross-Attn     │
│  • Best for: Classification,  │  • Best for: Open generation, │  • Best for: Translation,     │
│    Embeddings, NER, Retrieval │    In-context reasoning, LLMs │    Summarization, Denoising   │
└───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

### The Tectonic Shift Toward Decoder-Only Models
Why did the AI industry (OpenAI, Anthropic, Meta, Google) overwhelmingly converge on **Decoder-Only architectures** (GPT-4, Claude 3, LLaMA 3, Gemma) for modern frontier foundation models?
1. **Unification of Task Representation:** In an autoregressive decoder, every task—translation, coding, mathematics, reasoning, conversation—is framed identically as predicting the next token $p(x_t \mid x_{<t})$. There is no separate encoder representation to adapt.
2. **In-Context Learning (Prompting):** Pre-training on trillions of tokens teaches the autoregressive decoder to simulate arbitrary algorithms conditioned on prompts, enabling zero-shot and few-shot generalization without task-specific architectural heads.
3. **KV-Cache Inference Efficiency:** In autoregressive decoding, past Keys and Values are cached in GPU memory. Generating token $t+1$ requires only a single vector-matrix multiply ($\mathbf{q}_{t+1} \mathbf{K}^T$), avoiding redundant forward passes.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Architectural Taxonomy

```
      BERT (Encoder-Only)              GPT (Decoder-Only)            T5 (Encoder-Decoder)
       [ y1, y2, y3, y4 ]              [ y1, y2, y3, y4 ]             [ y1, y2 ] (Target)
               ▲                               ▲                              ▲
               │                               │                              │
         ┌───────────┐                   ┌───────────┐                  ┌───────────┐
         │Bidirection│                   │  Causal   │                  │  Decoder  │
         │ Attention │                   │ Attention │                  │(Cross-Att)│
         └───────────┘                   └───────────┘                  └─────▲─────┘
               ▲                               ▲                              │
               │                               │                        ┌─────┴─────┐
       [ x1, x2, x3, x4 ]              [ x1, x2, x3, x4 ]               │  Encoder  │
                                                                        └─────▲─────┘
                                                                              │
                                                                      [ x1, x2, x3, x4 ]
```

---

### 2.2 Encoder-Only Architecture (BERT & RoBERTa)

Let the input sequence be $\mathbf{x} = (x_1, \dots, x_T)$.

#### 1. Attention Mask:
The attention mask is unconstrained (all zeros):
$$\mathbf{M}_{\text{BERT}} = \mathbf{0}_{T \times T} \implies A_{i, j} > 0, \quad \forall i, j \in \{1, \dots, T\}$$
Every token $x_i$ attends bidirectionally to every token $x_j$ in the sequence.

#### 2. Masked Language Modeling (MLM) Objective:
A random subset of tokens $\mathcal{M} \subset \{1, \dots, T\}$ (typically $15\%$) is selected:
- $80\%$ of the time: replaced with a special `[MASK]` token.
- $10\%$ of the time: replaced with a random vocabulary token.
- $10\%$ of the time: left unchanged.

The model is trained to reconstruct the original tokens by minimizing cross-entropy over the masked indices:
$$\mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log p\left( x_i^* \mid \mathbf{h}_i^L \right) = - \sum_{i \in \mathcal{M}} \log \left( \frac{\exp(\mathbf{w}_{x_i^*}^T \mathbf{h}_i^L)}{\sum_{v \in \mathcal{V}} \exp(\mathbf{w}_v^T \mathbf{h}_i^L)} \right)$$

---

### 2.3 Decoder-Only Architecture (GPT & LLaMA)

#### 1. Attention Mask:
The attention scores are masked with a strictly lower-triangular causal mask:
$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
$$\mathbf{A}_{\text{GPT}} = \text{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right)$$
Token $i$ can only gather information from past and present tokens $x_{\le i}$.

#### 2. Autoregressive Next-Token Prediction Objective:
The network models the exact joint probability distribution of the text corpus via the autoregressive chain rule:
$$p(\mathbf{x}) = \prod_{t=1}^T p(x_t \mid x_1, \dots, x_{t-1})$$
$$\mathcal{L}_{\text{AR}} = - \sum_{t=1}^T \log p(x_t \mid x_{<t}) = - \sum_{t=1}^T \log \text{softmax}\left( \mathbf{W}_{\text{head}} \mathbf{h}_t^L \right)_{x_t}$$
Every token in the sequence acts simultaneously as a target label for the preceding context. A training sequence of length $T$ yields **$T$ loss gradients simultaneously** in a single forward/backward pass.

---

### 2.4 Encoder-Decoder Architecture (T5)

In T5 (Raffel et al., 2020), every NLP task is framed as a text-to-text transformation:
- Classification: `"sentiment: I loved the film" \to "positive"`
- Translation: `"translate German to English: Das Haus ist groß" \to "The house is big"`

#### The Span Corruption Denoising Objective:
Consecutive spans of tokens in the encoder input are replaced with unique sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, etc.):
- **Source:** `"Thank you <extra_id_0> my party <extra_id_1> week"`
- **Target:** `"<extra_id_0> for inviting me to <extra_id_1> last <extra_id_2>"`

#### The Cross-Attention Mechanism:
The decoder contains two attention sub-layers in each block:
1. **Masked Causal Self-Attention:** Decoder tokens attend causally to previous decoder tokens:
   $$\mathbf{Q}_{\text{dec}} = \mathbf{s}_t \mathbf{W}_Q, \quad \mathbf{K}_{\text{dec}} = \mathbf{s}_{\le t} \mathbf{W}_K, \quad \mathbf{V}_{\text{dec}} = \mathbf{s}_{\le t} \mathbf{W}_V$$
2. **Cross-Attention:** Queries originate from the decoder, while Keys and Values are projected from the final encoder representations $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{T_{\text{enc}} \times d_{\text{model}}}$:
   $$\mathbf{Q} = \mathbf{s}_t \mathbf{W}_Q^{\text{cross}}, \quad \mathbf{K} = \mathbf{H}_{\text{enc}} \mathbf{W}_K^{\text{cross}}, \quad \mathbf{V} = \mathbf{H}_{\text{enc}} \mathbf{W}_V^{\text{cross}}$$
   $$\text{CrossAttn} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V} \in \mathbb{R}^{T_{\text{dec}} \times d_v}$$

---

### 2.5 Pre-LN vs. Post-LN: The Gradient Highway Derivation

```
         POST-LN (Vaswani 2017)                       PRE-LN (Modern Standard)
                 x_l                                            x_l
                  │                                              ├──┐ (Identity Highway)
                  ▼                                              │  ▼
            ┌───────────┐                                        │ [LayerNorm]
            │ SubLayer  │                                        │  ▼
            └─────┬─────┘                                        │ [SubLayer]
                  │                                              │  ▼
                  ▼                                              ▼  │
                [ + ] ◄── x_l (Skip)                            [ + ] ◄┘
                  │                                              │
                  ▼                                              ▼
             [LayerNorm]                                       x_{l+1}
                  │
                  ▼
               x_{l+1}
```

#### 1. Post-LN Formulation:
$$\mathbf{x}_{l+1} = \text{LN}\left( \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l) \right)$$
Because LayerNorm encapsulates the addition, the backward gradient is scaled by the normalization Jacobian:
$$\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} = \mathbf{J}_{\text{LN}} \left( \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l} \right)$$
As shown by Xiong et al. (2020), across $L$ layers:
$$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_1} \right\| \sim \mathcal{O}\left( \frac{1}{\sqrt{L}} \right) \quad \text{or vanishes exponentially}$$
Without careful learning rate warmup (forcing small updates until statistics stabilize), deep Post-LN Transformers diverge immediately.

#### 2. Pre-LN Formulation:
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\text{LN}(\mathbf{x}_l))$$
Expanding recursively from layer $l$ to layer $L$:
$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\text{LN}(\mathbf{x}_i))$$
Applying the chain rule:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \sum_{i=l}^{L-1} \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l} \right)$$
The identity matrix $\mathbf{I}$ establishes an **unimpeded gradient highway** spanning all $L$ layers. The expected gradient norm is $\mathcal{O}(1)$ independent of network depth, allowing Transformers with hundreds of layers (e.g., 120 layers in GPT-3) to train robustly without learning rate warmup gymnastics.

---

### 2.6 Deep Derivation 9.3.1: Cross-Attention Backward Sensitivity Equations

#### Context and Setup
In Encoder-Decoder models (such as T5 or original Transformer), decoder states query encoder representations.
Let:
- $\mathbf{S} \in \mathbb{R}^{T_{\text{dec}} \times d}$ denote decoder hidden states (Queries).
- $\mathbf{H} \in \mathbb{R}^{T_{\text{enc}} \times d}$ denote encoder representations (Keys & Values).
- Projections: $\mathbf{Q} = \mathbf{S} \mathbf{W}_Q \in \mathbb{R}^{T_{\text{dec}} \times d_k}$, $\mathbf{K} = \mathbf{H} \mathbf{W}_K \in \mathbb{R}^{T_{\text{enc}} \times d_k}$, $\mathbf{V} = \mathbf{H} \mathbf{W}_V \in \mathbb{R}^{T_{\text{enc}} \times d_v}$.
- Scaled Cross-Attention Scores: $\mathbf{P} = \frac{1}{\sqrt{d_k}} \mathbf{Q} \mathbf{K}^T \in \mathbb{R}^{T_{\text{dec}} \times T_{\text{enc}}}$.
- Attention matrix: $\mathbf{A} = \operatorname{softmax}(\mathbf{P}) \in \mathbb{R}^{T_{\text{dec}} \times T_{\text{enc}}}$.
- Cross context: $\mathbf{C} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{T_{\text{dec}} \times d_v}$.

#### Theorem: Analytical Adjoints of Cross-Attention
Given upstream loss gradient $\boldsymbol{\Delta}^C = \frac{\partial \mathcal{L}}{\partial \mathbf{C}} \in \mathbb{R}^{T_{\text{dec}} \times d_v}$:

1. **Decoder Value and Score Sensitivities:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{A}^T \boldsymbol{\Delta}^C \in \mathbb{R}^{T_{\text{enc}} \times d_v}$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \boldsymbol{\Delta}^C \mathbf{V}^T \in \mathbb{R}^{T_{\text{dec}} \times T_{\text{enc}}}$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{P}} = \mathbf{A} \odot \left( \frac{\partial \mathcal{L}}{\partial \mathbf{A}} - \left[ \left(\frac{\partial \mathcal{L}}{\partial \mathbf{A}} \odot \mathbf{A}\right) \mathbf{1}_{T_{\text{enc}}} \right] \mathbf{1}_{T_{\text{enc}}}^T \right) \in \mathbb{R}^{T_{\text{dec}} \times T_{\text{enc}}}$$

2. **Decoder Query and Encoder Key Sensitivities:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{P}} \right) \mathbf{K} \in \mathbb{R}^{T_{\text{dec}} \times d_k}$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{P}} \right)^T \mathbf{Q} \in \mathbb{R}^{T_{\text{enc}} \times d_k}$$

3. **Total Encoder Adjoint Injection ($\frac{\partial \mathcal{L}}{\partial \mathbf{H}}$):**
   The encoder representations $\mathbf{H}$ supply both Keys and Values:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{H}} = \frac{\partial \mathcal{L}}{\partial \mathbf{K}} \mathbf{W}_K^T + \frac{\partial \mathcal{L}}{\partial \mathbf{V}} \mathbf{W}_V^T \in \mathbb{R}^{T_{\text{enc}} \times d}$$
   This vector bridges the gradient path directly from the generated decoder target text all the way back into the encoder's bidirectional backbone. $\blacksquare$

---

### 2.7 Deep Derivation 9.3.2: The Layer Normalization Jacobian and Variance Dissipation Theorem

#### LayerNorm Definition
For input $\mathbf{x} \in \mathbb{R}^d$:
$$\mu = \frac{1}{d} \sum_{i=1}^d x_i = \frac{1}{d} \mathbf{1}^T \mathbf{x}, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 = \frac{1}{d} \|\mathbf{x} - \mu \mathbf{1}\|_2^2$$
$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu \mathbf{1}}{\sqrt{\sigma^2 + \epsilon}}, \quad \mathbf{y} = \boldsymbol{\gamma} \odot \hat{\mathbf{x}} + \boldsymbol{\beta}$$

#### Theorem: The Exact Matrix Jacobian of LayerNorm
Let $\sigma = \sqrt{\sigma^2 + \epsilon}$. The Jacobian matrix $\mathbf{J}_{\text{LN}} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{d \times d}$ is:
$$\mathbf{J}_{\text{LN}} = \frac{1}{\sigma} \operatorname{diag}(\boldsymbol{\gamma}) \left( \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} \hat{\mathbf{x}} \hat{\mathbf{x}}^T \right)$$
where $\mathbf{P}_{\mathbf{1}}^\perp = \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T$ is the centering projection matrix, and $\mathbf{P}_{\hat{\mathbf{x}}}^\perp = \mathbf{I} - \frac{1}{d} \hat{\mathbf{x}} \hat{\mathbf{x}}^T$ removes the radial scaling component.

#### Proof:
1. **Total Differential:**
   $$d\mathbf{y} = \boldsymbol{\gamma} \odot d\hat{\mathbf{x}} = \operatorname{diag}(\boldsymbol{\gamma}) d\hat{\mathbf{x}}$$
   Differentiating the standardized vector $\hat{\mathbf{x}} = \sigma^{-1} (\mathbf{x} - \mu \mathbf{1})$:
   $$d\hat{\mathbf{x}} = \sigma^{-1} (d\mathbf{x} - d\mu \mathbf{1}) - \sigma^{-2} d\sigma (\mathbf{x} - \mu \mathbf{1}) = \frac{1}{\sigma} (d\mathbf{x} - d\mu \mathbf{1}) - \frac{d\sigma}{\sigma} \hat{\mathbf{x}}$$

2. **Differentials of Mean and Variance:**
   $$d\mu = \frac{1}{d} \mathbf{1}^T d\mathbf{x}$$
   $$\sigma^2 = \frac{1}{d} (\mathbf{x} - \mu \mathbf{1})^T (\mathbf{x} - \mu \mathbf{1}) \implies 2\sigma d\sigma = \frac{2}{d} (\mathbf{x} - \mu \mathbf{1})^T d(\mathbf{x} - \mu \mathbf{1})$$
   Since $(\mathbf{x} - \mu \mathbf{1})^T \mathbf{1} = 0$:
   $$\frac{d\sigma}{\sigma} = \frac{1}{d \sigma^2} (\mathbf{x} - \mu \mathbf{1})^T d\mathbf{x} = \frac{1}{d} \hat{\mathbf{x}}^T d\mathbf{x}$$

3. **Substitution into $d\hat{\mathbf{x}}$:**
   $$d\hat{\mathbf{x}} = \frac{1}{\sigma} \left( d\mathbf{x} - \frac{1}{d} \mathbf{1}\mathbf{1}^T d\mathbf{x} \right) - \left( \frac{1}{d} \hat{\mathbf{x}}^T d\mathbf{x} \right) \hat{\mathbf{x}} = \frac{1}{\sigma} \left( \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} \hat{\mathbf{x}} \hat{\mathbf{x}}^T \right) d\mathbf{x}$$
   Multiplying by $\operatorname{diag}(\boldsymbol{\gamma})$ yields the stated Jacobian.

#### Corollary: Why Post-LN Vanishes as $\mathcal{O}(L^{-1/2})$
In Post-LN, $\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + \mathcal{F}(\mathbf{x}_l))$.
Assuming $\mathbf{x}_l$ and $\mathcal{F}(\mathbf{x}_l)$ have comparable variance $\nu$, the unnormalized variance grows linearly with depth: $\operatorname{Var}(\mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)) \approx (l + 1) \nu$.
The normalizing factor is $\sigma_l \approx \sqrt{l \nu}$.
Because the Jacobian has the prefactor $\frac{1}{\sigma_l}$:
$$\|\mathbf{J}_{\text{LN}}^{(l)}\|_2 \sim \frac{1}{\sqrt{l}}$$
As gradients propagate backward through $L$ layers, early layers receive gradients scaled by $\frac{1}{\sqrt{L}}$, causing severe gradient starvation in the bottom layers during initial training epochs. $\blacksquare$

---

### 2.8 Deep Derivation 9.3.3: Information-Theoretic Comparison of MLM vs. Autoregressive Training

#### 1. Autoregressive Entropy and The Likelihood Upper Bound
In a causal decoder (GPT), the objective minimizes cross-entropy across all $T$ positions:
$$\mathcal{L}_{\text{AR}}(\theta) = -\frac{1}{T} \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})$$
By the asymptotic equipartition property (Shannon-McMillan-Breiman theorem) for stationary ergodic language sources:
$$\lim_{T \to \infty} \mathbb{E}[\mathcal{L}_{\text{AR}}(\theta)] = H(\mathcal{X}) + D_{\text{KL}}(P_{\text{true}} \parallel P_\theta)$$
where $H(\mathcal{X})$ is the true entropy rate of natural language.
Every single token provides an unbiased learning gradient:
$$\text{Tokens Trained per Forward-Backward Pass} = T$$

#### 2. Masked Language Model Sample Efficiency
In BERT, only a subset $\mathcal{M}$ of tokens is predicted:
$$\mathcal{L}_{\text{MLM}}(\theta) = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid \tilde{\mathbf{x}})$$
where $|\mathcal{M}| \approx 0.15 T$.
- **Training Signal Waste:** A BERT forward pass processes $T$ tokens, but computes cross-entropy gradients for only $15\%$ of them. $85\%$ of the compute contributes zero prediction loss signal.
- **The $15\%$ Dilemma:**
  - If masking rate $p < 0.15$, too few learning signals are generated per batch, making pre-training astronomically expensive.
  - If masking rate $p > 0.15$, the sentence context becomes excessively corrupted, destroying syntactic dependencies needed to reconstruct words.
- This $6.67\times$ lower training signal density per FLOP was a primary empirical driver for the industry-wide transition to autoregressive foundation models. $\blacksquare$

---

## 3. Geometric & Algebraic Interpretation

### Graph Topologies of the Three Architectures

| Model | Graph Type | Adjacency Matrix $\mathbf{A}$ | Information Flow |
| :--- | :--- | :--- | :--- |
| **BERT** | Complete Graph $K_T$ | Dense, fully symmetric | Omnidirectional, unconstrained |
| **GPT** | Directed Acyclic Graph (DAG) | Strictly lower-triangular | Causal, non-anticipative |
| **T5** | Bipartite Cross-Graph | $T_{\text{dec}} \times T_{\text{enc}}$ rectangular | Asymmetric information routing |

---

## 4. Real-World Analogy

### The Detective, The Novelist, and The Diplomatic Translator
- **BERT (The Detective):** Arrives at a crime scene where 15% of the evidence is covered with black tape. The detective inspects the entire room simultaneously—looking back and forth between the broken window on the left and the footprint on the right—to deduce what the black tape conceals.
- **GPT (The Novelist):** Sits in front of a typewriter. They can re-read all pages written so far, but cannot jump ahead to chapter 10 because chapter 10 does not yet exist. They produce the next sentence word-by-word.
- **T5 (The Diplomatic Translator):** Receives a treaty written in French. They read and analyze the entire document thoroughly (Encoder), then produce the official English treaty sequentially line-by-line (Decoder), glancing back at specific French paragraphs via Cross-Attention as needed.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace forward and backward propagation through a Cross-Attention step with concrete hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Cross-Attention |
| :--- | :--- | :--- | :--- |
| $T_{\text{enc}}$ | Encoder Tokens | Scalar ($2$) | Length of source sentence representations |
| $T_{\text{dec}}$ | Decoder Step | Scalar ($1$) | Single decoding step query |
| $d$ | Hidden Dimension | Scalar ($2$) | Dimensionality of tokens |
| $\mathbf{h}_1, \mathbf{h}_2$ | Encoder Keys/Values | $(2,)$ each | Source representations from Encoder |
| $\mathbf{s}_1$ | Decoder Query | $(2,)$ | State of Decoder seeking information |
| $\mathbf{K}, \mathbf{V}$ | Encoder Key/Value Matrix | $(2, 2)$ each | Stacked representations $[\mathbf{h}_1; \mathbf{h}_2]$ |
| $\mathbf{Q}$ | Decoder Query Matrix | $(1, 2)$ | Row vector $[\mathbf{s}_1]$ |
| $\mathbf{S}_{\text{cross}}$ | Cross-Attention Scores | $(1, 2)$ | $\mathbf{Q} \mathbf{K}^T / \sqrt{2}$ |
| $\mathbf{A}_{\text{cross}}$ | Cross-Attention Weights | $(1, 2)$ | Softmax distribution over encoder tokens |
| $\mathbf{C}$ | Context Vector | $(1, 2)$ | $\mathbf{A}_{\text{cross}} \mathbf{V}$ |

---

### 5.2 Concrete Toy Numbers

#### Encoder Representations ($T_{\text{enc}} = 2, d = 2$):
$$\mathbf{h}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{h}_2 = \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} \implies \mathbf{K} = \mathbf{V} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$

#### Decoder Query ($T_{\text{dec}} = 1, d = 2$):
$$\mathbf{s}_1 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} \implies \mathbf{Q} = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix}$$

---

### 5.3 Step 1: Raw Dot Products and Scaling ($\sqrt{d_k} = \sqrt{2} \approx 1.414214$)

$$\mathbf{Q} \mathbf{K}^T = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}$$

Scaling by $\frac{1}{\sqrt{2}} \approx 0.707107$:
$$\mathbf{S}_{\text{cross}} = \begin{bmatrix} \frac{1.0}{\sqrt{2}} & \frac{2.0}{\sqrt{2}} \end{bmatrix} = \mathbf{\begin{bmatrix} 0.707107 & 1.414214 \end{bmatrix}}$$

---

### 5.4 Step 2: Softmax Alignment Weights

$$\exp(0.707107) \approx 2.028115, \quad \exp(1.414214) \approx 4.113250$$
$$\text{Sum} = 2.028115 + 4.113250 = 6.141365$$

$$A_1 = \frac{2.028115}{6.141365} \approx \mathbf{0.330239}$$
$$A_2 = \frac{4.113250}{6.141365} \approx \mathbf{0.669761}$$

$$\mathbf{A}_{\text{cross}} = \begin{bmatrix} 0.330239 & 0.669761 \end{bmatrix}$$

---

### 5.5 Step 3: Value Aggregation ($\mathbf{C} = \mathbf{A}_{\text{cross}} \mathbf{V}$)

$$\mathbf{C} = \begin{bmatrix} 0.330239 & 0.669761 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$

$$C_1 = (0.330239)(1.0) + (0.669761)(0.0) = \mathbf{0.330239}$$
$$C_2 = (0.330239)(0.0) + (0.669761)(2.0) = \mathbf{1.339522}$$

$$\mathbf{C} = \begin{bmatrix} 0.330239 & 1.339522 \end{bmatrix}$$

The decoder state $\mathbf{s}_1 = [1, 1]$ has dynamically extracted a context vector $\mathbf{C}$ that draws twice as much energy from encoder token 2 as from token 1!

---

## 6. Solved Illustrations

### Illustration 1: Why Masked Language Models (BERT) Cannot Generate Efficiently

**Problem:**
Explain why BERT cannot be used for fast autoregressive generation, and why generating text with BERT requires Gibbs sampling costing $\mathcal{O}(T^2)$ passes.

**Solution:**
In an autoregressive decoder (GPT), tokens are conditionally independent given past tokens:
$$p(x_1, \dots, x_T) = \prod_{t=1}^T p(x_t \mid x_{<t})$$
Generating $T$ tokens requires exactly $T$ sequential forward steps ($1$ step per token).
In BERT, the model parameterizes conditional distributions where every token depends on *both* past and future tokens:
$$p(x_i \mid x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_T)$$
This defines a **Markov Random Field (MRF)**, not an autoregressive sequence. The joint distribution cannot be factorized into simple sequential probabilities.
To generate $T$ tokens, one must initialize a canvas of $T$ `[MASK]` tokens and run **iterative Gibbs sampling**:
- Sample token 1 given masks.
- Sample token 2 given updated token 1.
- Repeat for $K$ iterations over all $T$ positions until convergence.
Generating a 100-token paragraph requires $K \times 100 \approx 2,000 - 5,000$ forward passes, rendering generation orders of magnitude slower than a causal decoder!

---

### Illustration 2: Pre-LN vs. Post-LN Numerical Gradient Preservation Across 30 Layers

**Problem:**
Calculate the backpropagated gradient norm at Layer 1 for a 30-layer Post-LN Transformer vs. a 30-layer Pre-LN Transformer under standard initialization.

**Solution:**
1. **Post-LN Transformer:**
   At each layer, the output is normalized: $\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + \mathcal{F}(\mathbf{x}_l))$.
   Because variance of $\mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$ grows with layer depth ($\approx l$), the normalizing divisor $\sigma_l \sim \sqrt{l}$ scales down the backward gradient by factor $\frac{1}{\sqrt{l}}$ at every layer:
   $$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_1} \right\| \approx \prod_{l=1}^{30} \frac{1}{\sqrt{1 + 1/l}} \sim \frac{1}{\sqrt{30}} \approx \mathbf{0.1825}$$
   In deep networks ($L = 100$), gradients vanish below $0.10$, stalling early layer updates.
2. **Pre-LN Transformer:**
   The identity highway $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\text{LN}(\mathbf{x}_l))$ transfers gradient directly:
   $$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_1} \right\| = \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left(\mathbf{I} + \sum \dots \right) \right\| \approx \mathbf{1.0000}$$
   Pre-LN completely isolates backpropagation from network depth $L$.

---

### Illustration 3: Complete Hand Trace of Backward Pass through Cross-Attention Layer

**Problem:**
Consider the Cross-Attention step from Section 5 where a single decoder token query $\mathbf{Q} \in \mathbb{R}^{1 \times 2}$ attends to two encoder representations $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{2 \times 2}$ with $d_k = 2$ ($\frac{1}{\sqrt{d_k}} \approx 0.707107$):
$$\mathbf{Q} = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix}, \quad \mathbf{K} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}, \quad \mathbf{V} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$
$$\mathbf{S}_{\text{cross}} = \begin{bmatrix} 0.707107 & 1.414214 \end{bmatrix}, \quad \mathbf{A}_{\text{cross}} = \begin{bmatrix} 0.330239 & 0.669761 \end{bmatrix}, \quad \mathbf{C} = \begin{bmatrix} 0.330239 & 1.339522 \end{bmatrix}$$
Given upstream loss sensitivity w.r.t. the extracted cross context:
$$\boldsymbol{\delta}_C = \frac{\partial \mathcal{L}}{\partial \mathbf{C}} = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix}$$
1. Calculate the gradient w.r.t. encoder values $\frac{\partial \mathcal{L}}{\partial \mathbf{V}}$ and attention probabilities $\frac{\partial \mathcal{L}}{\partial \mathbf{A}_{\text{cross}}}$.
2. Propagate through the Softmax Jacobian to compute $\frac{\partial \mathcal{L}}{\partial \mathbf{S}_{\text{cross}}}$.
3. Evaluate the decoder query gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Q}}$ and encoder key gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$.

**Solution:**

#### Step 1: Sensitivities for Values and Attention Weights

1. **Gradient w.r.t. Encoder Values $\mathbf{V}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{A}_{\text{cross}}^T \boldsymbol{\delta}_C = \begin{bmatrix} 0.330239 \\ 0.669761 \end{bmatrix} \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.330239} & \mathbf{0.330239} \\ \mathbf{0.669761} & \mathbf{0.669761} \end{bmatrix}$$

2. **Gradient w.r.t. Attention Weights $\mathbf{A}_{\text{cross}}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{A}_{\text{cross}}} = \boldsymbol{\delta}_C \mathbf{V}^T = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{1.000000} & \mathbf{2.000000} \end{bmatrix}$$

---

#### Step 2: Backpropagation through Softmax Jacobian to Scores $\mathbf{S}_{\text{cross}}$

Inner product of weights and sensitivities:
$$\sum_{k=1}^2 A_k \frac{\partial \mathcal{L}}{\partial A_k} = (0.330239)(1.0) + (0.669761)(2.0) = 0.330239 + 1.339522 = 1.669761$$
Applying $\frac{\partial \mathcal{L}}{\partial S_j} = A_j \left( \frac{\partial \mathcal{L}}{\partial A_j} - \sum_k A_k \frac{\partial \mathcal{L}}{\partial A_k} \right)$:
$$\frac{\partial \mathcal{L}}{\partial S_1} = 0.330239 \times (1.000000 - 1.669761) = 0.330239 \times (-0.669761) = \mathbf{-0.221181}$$
$$\frac{\partial \mathcal{L}}{\partial S_2} = 0.669761 \times (2.000000 - 1.669761) = 0.669761 \times (+0.330239) = \mathbf{+0.221181}$$
*(Check: $-0.221181 + 0.221181 = 0$)*.
$$\frac{\partial \mathcal{L}}{\partial \mathbf{S}_{\text{cross}}} = \begin{bmatrix} -0.221181 & 0.221181 \end{bmatrix}$$

---

#### Step 3: Decoder Query and Encoder Key Gradients

1. **Decoder Query Gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Q}}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \frac{1}{\sqrt{2}} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{S}_{\text{cross}}} \right) \mathbf{K} = \frac{1}{\sqrt{2}} \begin{bmatrix} -0.221181 & 0.221181 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$
   $$= 0.707107 \times \begin{bmatrix} -0.221181 & 0.442362 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.156399} & \mathbf{0.312798} \end{bmatrix}$$

2. **Encoder Key Gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \frac{1}{\sqrt{2}} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{S}_{\text{cross}}} \right)^T \mathbf{Q} = 0.707107 \times \begin{bmatrix} -0.221181 \\ 0.221181 \end{bmatrix} \begin{bmatrix} 1.0 & 1.0 \end{bmatrix}$$
   $$= \begin{bmatrix} \mathbf{-0.156399} & \mathbf{-0.156399} \\ \mathbf{0.156399} & \mathbf{0.156399} \end{bmatrix}$$

The gradient backpropagates simultaneously into both the decoding step query $\mathbf{Q}$ and the encoder memory $\mathbf{K}, \mathbf{V}$.

---

### Illustration 4: Hand Calculation of Layer Normalization Forward Pass and Backward Jacobian

**Problem:**
A LayerNorm module operates on a 4-dimensional hidden activation vector $\mathbf{x} = [2.0, 4.0, 6.0, 8.0]^T \in \mathbb{R}^4$.
Assume learnable affine scale $\boldsymbol{\gamma} = [1, 1, 1, 1]^T$ and bias $\boldsymbol{\beta} = [0, 0, 0, 0]^T$ (with stability $\epsilon = 0$).
1. Compute the mean $\mu$, sample variance $\sigma^2$, standard deviation $\sigma$, and normalized output vector $\hat{\mathbf{x}} = \mathbf{y}$.
2. Given upstream loss gradient $\boldsymbol{\delta}_y = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = [1.0, 0.0, 0.0, -1.0]^T$, calculate the exact input gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ step-by-step using the analytical LayerNorm Jacobian formula.

**Solution:**

#### Step 1: Forward Pass

1. **Mean:**
   $$\mu = \frac{1}{4} (2.0 + 4.0 + 6.0 + 8.0) = \frac{20.0}{4} = \mathbf{5.0}$$

2. **Variance and Standard Deviation:**
   $$\mathbf{x} - \mu \mathbf{1} = \begin{bmatrix} 2.0 - 5.0 \\ 4.0 - 5.0 \\ 6.0 - 5.0 \\ 8.0 - 5.0 \end{bmatrix} = \begin{bmatrix} -3.0 \\ -1.0 \\ 1.0 \\ 3.0 \end{bmatrix}$$
   $$\sigma^2 = \frac{1}{4} \left( (-3.0)^2 + (-1.0)^2 + (1.0)^2 + (3.0)^2 \right) = \frac{9.0 + 1.0 + 1.0 + 9.0}{4} = \frac{20.0}{4} = \mathbf{5.0}$$
   $$\sigma = \sqrt{5.0} \approx \mathbf{2.236068}$$

3. **Standardized Representation $\hat{\mathbf{x}}$:**
   $$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu \mathbf{1}}{\sigma} = \begin{bmatrix} -3.0 / 2.236068 \\ -1.0 / 2.236068 \\ 1.0 / 2.236068 \\ 3.0 / 2.236068 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.341641} \\ \mathbf{-0.447214} \\ \mathbf{0.447214} \\ \mathbf{1.341641} \end{bmatrix}$$
   *(Verification: $\sum \hat{x}_i = 0$, and $\sum \hat{x}_i^2 = 1.8 + 0.2 + 0.2 + 1.8 = 4.0 = d$)*.
   Since $\boldsymbol{\gamma} = \mathbf{1}$ and $\boldsymbol{\beta} = \mathbf{0}$, $\mathbf{y} = \hat{\mathbf{x}}$.

---

#### Step 2: Backward Sensitivity via Analytical LayerNorm Jacobian

The closed-form derivative is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{1}{\sigma} \left[ \boldsymbol{\delta}_y - \bar{\delta}_y \mathbf{1} - \hat{\mathbf{x}} \left( \frac{1}{d} \sum_{i=1}^d \delta_{y, i} \hat{x}_i \right) \right]$$

1. **Mean of Upstream Gradient:**
   $$\bar{\delta}_y = \frac{1}{4} (1.0 + 0.0 + 0.0 - 1.0) = \mathbf{0.0}$$
   $$\boldsymbol{\delta}_y - \bar{\delta}_y \mathbf{1} = \begin{bmatrix} 1.0 \\ 0.0 \\ 0.0 \\ -1.0 \end{bmatrix}$$

2. **Inner Product with $\hat{\mathbf{x}}$:**
   $$\boldsymbol{\delta}_y^T \hat{\mathbf{x}} = (1.0)(-1.341641) + (0.0)(-0.447214) + (0.0)(0.447214) + (-1.0)(1.341641) = -2.683282$$
   Dividing by $d = 4$:
   $$\frac{1}{4} \boldsymbol{\delta}_y^T \hat{\mathbf{x}} = \frac{-2.683282}{4} = \mathbf{-0.6708205}$$

3. **Projection Subtraction:**
   $$\hat{\mathbf{x}} \left( \frac{1}{d} \boldsymbol{\delta}_y^T \hat{\mathbf{x}} \right) = -0.6708205 \times \begin{bmatrix} -1.341641 \\ -0.447214 \\ 0.447214 \\ 1.341641 \end{bmatrix} = \begin{bmatrix} +0.900000 \\ +0.300000 \\ -0.300000 \\ -0.900000 \end{bmatrix}$$
   Subtracting from centered gradient:
   $$\boldsymbol{\Delta} = \begin{bmatrix} 1.0 - 0.900000 \\ 0.0 - 0.300000 \\ 0.0 - (-0.300000) \\ -1.0 - (-0.900000) \end{bmatrix} = \begin{bmatrix} +0.100000 \\ -0.300000 \\ +0.300000 \\ -0.100000 \end{bmatrix}$$
   *(Verification: sum is $0.1 - 0.3 + 0.3 - 0.1 = 0$)*.

4. **Scaling by $\frac{1}{\sigma}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{1}{2.236068} \begin{bmatrix} +0.100000 \\ -0.300000 \\ +0.300000 \\ -0.100000 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.044721} \\ \mathbf{-0.134164} \\ \mathbf{+0.134164} \\ \mathbf{-0.044721} \end{bmatrix}$$

Notice how LayerNorm projects the upstream gradient onto the tangent space orthogonal to both $\mathbf{1}$ and $\hat{\mathbf{x}}$, maintaining scale and mean invariance!

---

### Illustration 5: Autoregressive KV-Cache Memory Footprint and FLOP Arithmetic for GQA vs. MHA

**Problem:**
A 70-billion parameter foundation model (similar to LLaMA-3-70B) has the following architectural specifications:
- Number of layers: $L = 80$
- Hidden dimension: $d_{\text{model}} = 8,192$
- Query attention heads: $h_q = 64$
- Head dimension: $d_k = 128$
- Precision: 16-bit floating point (FP16 $\to 2$ bytes per element).

Compare two configurations for Key-Value caching during inference:
1. **Multi-Head Attention (MHA):** $h_{kv} = 64$ heads.
2. **Grouped-Query Attention (GQA):** $h_{kv} = 8$ heads (each KV head shared across 8 query heads).

Calculate:
1. The KV-cache memory required per token per sequence (in KB).
2. The total KV-cache RAM consumed by a concurrent batch of $B = 4$ users at context lengths $T = 2,048$, $T = 8,192$, and $T = 32,768$ tokens.

**Solution:**

#### Step 1: Memory Footprint Formula per Token

For each token generated at each layer, the model appends:
- Key tensor: $h_{kv} \times d_k$ elements
- Value tensor: $h_{kv} \times d_k$ elements
Total elements per layer $= 2 \times h_{kv} \times d_k$.
Across all $L = 80$ layers with 2 bytes per scalar:
$$\text{Bytes per Token} = 2 \times L \times h_{kv} \times d_k \times 2 = 4 \times L \times h_{kv} \times d_k$$

1. **Standard MHA ($h_{kv} = 64$):**
   $$\text{Bytes} = 4 \times 80 \times 64 \times 128 = 2,621,440 \text{ bytes} = \frac{2,621,440}{1,024} = \mathbf{2,560 \text{ KB}} = \mathbf{2.50 \text{ MB per token}}$$

2. **Grouped-Query Attention ($h_{kv} = 8$):**
   $$\text{Bytes} = 4 \times 80 \times 8 \times 128 = 327,680 \text{ bytes} = \frac{327,680}{1,024} = \mathbf{320 \text{ KB}} = \mathbf{0.3125 \text{ MB per token}}$$

GQA achieves an **$8\times$ reduction** in KV-cache storage per token!

---

#### Step 2: Batch Memory Requirements ($B = 4$)

Total Cache Memory:
$$\text{Memory}(B, T) = B \times T \times \text{Bytes per Token}$$

| Context Length ($T$) | Standard MHA Cache ($B = 4$) | Grouped-Query Attention (GQA) Cache ($B = 4$) | Memory Savings Factor |
| :--- | :--- | :--- | :--- |
| **$2,048$ tokens** | $4 \times 2048 \times 2.50 \text{ MB} = \mathbf{20.00 \text{ GB}}$ | $4 \times 2048 \times 0.3125 \text{ MB} = \mathbf{2.50 \text{ GB}}$ | $8\times$ (Fits easily in 24GB VRAM) |
| **$8,192$ tokens** | $4 \times 8192 \times 2.50 \text{ MB} = \mathbf{80.00 \text{ GB}}$ | $4 \times 8192 \times 0.3125 \text{ MB} = \mathbf{10.00 \text{ GB}}$ | $8\times$ (MHA exhausts full 80GB A100 GPU!) |
| **$32,768$ tokens** | $4 \times 32768 \times 2.50 \text{ MB} = \mathbf{320.00 \text{ GB}}$ | $4 \times 32768 \times 0.3125 \text{ MB} = \mathbf{40.00 \text{ GB}}$ | $8\times$ (MHA requires 4 node cluster; GQA fits on 1 GPU) |

**Conclusion:**
Under MHA, the KV-cache for just 4 users at 8k context length consumes 80 GB of VRAM—completely crowding out the 140 GB model weights. By switching to GQA, the cache drops to 10 GB, enabling modern multi-tenant serving at massive context lengths.

---

## 7. Deep Learning Connection & Application

### Architecture Selection Guidelines for Real-World AI Systems

| Use-Case / Task | Recommended Archetype | Leading Production Models | Why This Choice? |
| :--- | :--- | :--- | :--- |
| **Semantic Search & Vector RAG** | **Encoder-Only** | BGE-M3, E5, ColBERT | Dense bidirectional representations maximize retrieval accuracy |
| **Document Classification & NER** | **Encoder-Only** | DeBERTa-v3, RoBERTa | Bidirectional context inspects whole document simultaneously |
| **Code Generation & Chatbot LLMs** | **Decoder-Only** | LLaMA 3, Mistral, GPT-4 | Native next-token generation, KV-cache speed, in-context learning |
| **Document Summarization / Translation** | **Encoder-Decoder** | T5, NLLB-200, Whisper | Separates variable-length input ingestion from target output drafting |

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `PreLNTransformerBlock`: Vectorized Pre-LN Transformer block with self-attention and MLP.
2. `PostLNTransformerBlock`: Classic Post-LN Transformer block.
3. `CrossAttentionBlock`: Decoder block incorporating both causal self-attention and cross-attention.
4. Exact numerical verification of the Part 5 Visual Grid Cross-Attention hand calculations.
5. The **Pre-LN vs. Post-LN Gradient Flow Experiment**:
   - Compares backpropagated gradient norms across 30 layers of Pre-LN vs. Post-LN Transformers.
   - Proves that Pre-LN preserves stable $\mathcal{O}(1)$ gradient norms, while Post-LN exhibits gradient decay.

See implementation in:
[`09_transformers_and_llms/code/03_transformer_architectures.py`](./code/03_transformer_architectures.py)
