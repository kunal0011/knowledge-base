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
