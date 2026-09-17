# 12.12 Multimodal Vision-Language Architectures (Vision Encoders, Cross-Attention & Linear Projections)

---

## 1. Intuition & 101 Motivation

Human intelligence does not operate on pure text in isolation; it integrates sensory perception (vision, audio) with cognitive reasoning and language. The earliest attempts at multimodal AI attempted to train unified multimodal networks from scratch, requiring exorbitant datasets and often degrading text reasoning performance.

The breakthrough of modern **Multimodal Large Language Models (MLLMs)**—such as **LLaVA**, **Flamingo**, **Qwen2-VL**, **Gemini**, and **GPT-4o**—relies on a modular paradigm:
> *Connect an already powerful, frozen pre-trained Vision Encoder (e.g. CLIP ViT, SigLIP) to an already powerful, pre-trained Large Language Model (e.g. LLaMA, Mistral, Qwen) using a lightweight cross-modal adapter.*

By treating visual representations as simply **another foreign dialect** that can be translated into token embeddings, the LLM's vast pre-trained world knowledge, reasoning circuits, and conversational fluency are preserved intact.

Three dominant cross-modal bridging architectures define the state of the art:
1. **Linear & MLP Projectors (LLaVA style):** Visual tokens are projected into the text embedding space via a 2-layer MLP and prefixed directly to the text token sequence: $[v_1, \dots, v_k, t_1, \dots, t_m]$.
2. **Perceiver Resampler / Q-Former (Flamingo, BLIP-2):** A fixed set of learnable query tokens cross-attends to the visual patch features, compressing hundreds of spatial tokens into a compact, fixed-size semantic bottleneck (e.g. 64 tokens).
3. **Gated Cross-Attention (Flamingo, IDEFICS):** Interleaves dedicated cross-attention layers directly inside the LLM Transformer blocks, allowing text tokens to attend to visual features via tanh-gated residual connections.

```
                    MULTIMODAL BRIDGING PARADIGMS
                    
    LLaVA (MLP Projector)                Flamingo (Perceiver Resampler)
   Image I                              Image I
      ▼                                    ▼
   [Vision Encoder (ViT)]               [Vision Encoder (ViT)]
      ▼ (576 patch tokens)                 ▼ (Variable tokens N_v)
   [2-Layer MLP Projector]              [Perceiver Resampler (Cross-Attn)]
      ▼                                    ▼
   [v_1, v_2, ..., v_576] (Tokens)      [q_1, q_2, ..., q_64] (Fixed 64 Tokens)
      ▼                                    ▼
   Concat with Text: [V || Text]        Prefix or Gated Cross-Attention
      ▼                                    ▼
   Standard Autoregressive LLM          Standard Autoregressive LLM
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Vision Feature Extraction (ViT / CLIP)

Given an input image $I \in \mathbb{R}^{3 \times H \times W}$ (e.g., $336 \times 336 \times 3$):
1. The image is split into non-overlapping spatial patches of size $p \times p$ (e.g. $p = 14$).
2. The number of visual tokens is:
   $$N_v = \left(\frac{H}{p}\right) \times \left(\frac{W}{p}\right) = \left(\frac{336}{14}\right) \times \left(\frac{336}{14}\right) = 24 \times 24 = 576 \text{ tokens}$$
3. The pre-trained Vision Transformer maps patches to feature vectors:
   $$Z_v = \operatorname{ViT}(I) \in \mathbb{R}^{N_v \times d_v}$$
   where $d_v$ is the vision encoder hidden dimension (e.g., $d_v = 1024$ for CLIP ViT-L/14).

---

### 2.2 Multimodal Projection: Linear vs. Two-Layer MLP

The LLM operates on a text embedding space of dimension $d_{\text{text}}$ (e.g., $d_{\text{text}} = 4096$ for LLaMA-3 8B), which is incompatible with $d_v = 1024$.

#### 1. Linear Projector (LLaVA-1.0):
$$H_v = Z_v W_p \in \mathbb{R}^{N_v \times d_{\text{text}}}, \quad W_p \in \mathbb{R}^{d_v \times d_{\text{text}}}$$

#### 2. Two-Layer MLP Projector with GELU (LLaVA-1.5):
A simple linear projection forces an affine alignment between vision and language. Adding non-linear capacity significantly improves complex spatial and OCR reasoning:
$$H_v = \operatorname{GELU}(Z_v W_1 + b_1) W_2 + b_2$$
where $W_1 \in \mathbb{R}^{d_v \times d_{\text{text}}}$, $b_1 \in \mathbb{R}^{d_{\text{text}}}$, $W_2 \in \mathbb{R}^{d_{\text{text}} \times d_{\text{text}}}$, $b_2 \in \mathbb{R}^{d_{\text{text}}}$.

---

### 2.3 Prefix Concatenation & Autoregressive Decoder

Given a text prompt $T_{\text{prompt}} = (t_1, t_2, \dots, t_{N_t})$ embedded via the LLM word embedding lookup:
$$H_t = \operatorname{Embed}(T_{\text{prompt}}) \in \mathbb{R}^{N_t \times d_{\text{text}}}$$

The multimodal sequence is constructed by **concatenating visual tokens and text tokens** along the sequence dimension:

$$H_{\text{seq}} = \left[ H_v \,\|\, H_t \right] \in \mathbb{R}^{(N_v + N_t) \times d_{\text{text}}}$$

The unified sequence is passed through the standard causal autoregressive Transformer. Causal masking ensures:
- Visual tokens can attend to preceding visual tokens (or bidirectionally to all visual tokens).
- Text prompt tokens attend to all visual tokens and preceding prompt tokens.
- Generated response tokens attend to visual tokens, prompt tokens, and previously generated response tokens.

$$\mathcal{L}_{\text{MLLM}}(\theta) = - \sum_{i=1}^{N_{\text{resp}}} \log P_\theta(y_i \mid H_v, T_{\text{prompt}}, y_{<i})$$

---

### 2.4 Perceiver Resampler Bottleneck (Flamingo / BLIP-2)

When high-resolution images or multi-frame video inputs are used, $N_v$ can exceed thousands of tokens, overflowing the LLM's context window.
The **Perceiver Resampler** compresses arbitrary $N_v$ tokens into a fixed budget of $K$ latent tokens ($K \ll N_v$, typically $K = 64$):

1. Initialize $K$ learnable query tokens $Q_{\text{learn}} \in \mathbb{R}^{K \times d}$.
2. Project visual features $Z_v \in \mathbb{R}^{N_v \times d_v}$ into keys and values:
   $$K_v = Z_v W_k, \quad V_v = Z_v W_v$$
3. Perform cross-attention:
   $$H_{\text{bottleneck}} = \operatorname{softmax}\left( \frac{(Q_{\text{learn}} W_q)(K_v)^T}{\sqrt{d_k}} \right) V_v \in \mathbb{R}^{K \times d}$$
Regardless of image size ($512\times 512$ or $4K$ resolution), the LLM receives **strictly $K = 64$ visual tokens**!

---

### 2.5 Two-Stage Training Protocol (LLaVA Recipe)

Training an MLLM proceeds in two distinct stages:

```
                  LLaVA TWO-STAGE ALIGNMENT
                  
  Stage 1: Pre-Training Feature Alignment
  [Vision ViT] (FROZEN) ──► [MLP Projector] (TRAINABLE) ──► [LLM] (FROZEN)
  Dataset: 600K Image-Caption Pairs (CC3M / ShareGPT4V)
  Goal: Align visual feature distribution with LLM word embeddings.
  
  Stage 2: Visual Instruction Tuning
  [Vision ViT] (FROZEN) ──► [MLP Projector] (TRAINABLE) ──► [LLM] (TRAINABLE)
  Dataset: 150K Multi-Turn Visual Conversations (Complex Reasoning, Detail Description)
  Goal: Endow model with conversational visual reasoning and question-answering.
```

---

## 3. Geometric & Physical Interpretation

### 3.1 Manifold Alignment
The visual representation space $\mathcal{M}_v \subset \mathbb{R}^{d_v}$ and text representation space $\mathcal{M}_t \subset \mathbb{R}^{d_{\text{text}}}$ are disjoint manifolds with completely different geometric metrics.
The MLP projector acts as an **affine homeomorphic bridge**: it translates clustered visual topological structures (e.g. animal shapes, textures) directly into the semantic attractor basins of the text decoder, so that the vector for an image patch of a cat lands adjacent to the token embedding for the word `"cat"`.

```
   Vision Space M_v (CLIP)                 Text Space M_t (LLaMA)
      [Cat Patch Feature]                     [Token 'cat']
             \                                      ▲
              \          MLP Projector             /
               \──────────────────────────────────/
                       Affine Translation
```

---

## 4. Real-World Analogy: The Scientist and the Diplomatic Translator

- **Vision Encoder (ViT):** An astrophysicist who speaks only in raw mathematical formulas and spectral diagrams ($Z_v$).
- **LLM:** A world leader who only speaks English ($H_t$).
- **MLP Projector:** The professional diplomatic translator. When the astrophysicist holds up a spectrum of a distant galaxy, the translator quickly summarizes the observation into concise English sentences ($H_v$) and places them on the leader's desk. The leader reads the briefing and decides on foreign policy without ever needing to study astrophysics directly!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete, step-by-step numerical example of:
1. **Visual Patch Feature Projection through a 2-Layer MLP**
2. **Multimodal Sequence Concatenation ($[H_v \,\|\, H_t]$)**
3. **Multimodal Causal Attention Layout**

---

### 5.1 Concrete Input Values

- **Visual Patches:** $N_v = 2$ patches, visual feature dimension $d_v = 2$.
  $$Z_v = \begin{bmatrix} 1.0 & 2.0 \\ 0.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$
  - Patch 1: $z_1 = [1.0, 2.0]$
  - Patch 2: $z_2 = [0.0, 1.0]$

- **Text Prompt:** $N_t = 1$ token, text embedding dimension $d_{\text{text}} = 2$.
  $$H_t = \begin{bmatrix} 3.0 & -1.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}$$

- **MLP Projector Parameters:**
  - Layer 1 Weight $W_1 \in \mathbb{R}^{2 \times 2}$, Bias $b_1 \in \mathbb{R}^2$:
    $$W_1 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} = I_2, \quad b_1 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$
  - Activation function: $\operatorname{ReLU}(z) = \max(0, z)$ (for clean integer arithmetic).
  - Layer 2 Weight $W_2 \in \mathbb{R}^{2 \times 2}$, Bias $b_2 \in \mathbb{R}^2$:
    $$W_2 = \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.0 \end{bmatrix}, \quad b_2 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $Z_v$ | `vision_features` | Raw extracted visual patch representations ($2 \times 2$) |
| $U$ | `mlp_hidden` | Activated intermediate representations $\operatorname{ReLU}(Z_v W_1)$ |
| $H_v$ | `projected_vision` | Aligned visual token embeddings $U W_2 \in \mathbb{R}^{2 \times 2}$ |
| $H_t$ | `text_embeddings` | Pre-trained text prompt token embeddings ($1 \times 2$) |
| $H_{\text{seq}}$ | `multimodal_input` | Concatenated sequence $[H_v \,\|\, H_t] \in \mathbb{R}^{3 \times 2}$ |
| $M_{\text{causal}}$ | `causal_mask` | $3 \times 3$ lower-triangular causal attention mask |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Layer 1 Forward Pass & Non-Linear Activation
$$U = \operatorname{ReLU}(Z_v W_1 + b_1) = \operatorname{ReLU}(Z_v I_2) = \operatorname{ReLU}(Z_v)$$
Since all values in $Z_v$ are non-negative ($1.0, 2.0, 0.0, 1.0 \ge 0$):
$$U = \begin{bmatrix} 1.0 & 2.0 \\ 0.0 & 1.0 \end{bmatrix}$$

---

#### Step 2: Layer 2 Projection to Text Space $H_v = U W_2 + b_2$
$$W_2 = \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.0 \end{bmatrix}$$

- **For Patch 1 ($u_1 = [1.0, 2.0]$):**
  $$H_{v, 1, 1} = (1.0 \times 0.5) + (2.0 \times -0.5) = 0.5 - 1.0 = \mathbf{-0.500000}$$
  $$H_{v, 1, 2} = (1.0 \times 1.0) + (2.0 \times 0.0) = 1.0 + 0.0 = \mathbf{1.000000}$$
  $$H_{v, 1} = [-0.500000, 1.000000]$$

- **For Patch 2 ($u_2 = [0.0, 1.0]$):**
  $$H_{v, 2, 1} = (0.0 \times 0.5) + (1.0 \times -0.5) = 0.0 - 0.5 = \mathbf{-0.500000}$$
  $$H_{v, 2, 2} = (0.0 \times 1.0) + (1.0 \times 0.0) = 0.0 + 0.0 = \mathbf{0.000000}$$
  $$H_{v, 2} = [-0.500000, 0.000000]$$

Projected Vision Token Matrix:
$$H_v = \begin{bmatrix} \mathbf{-0.500000} & \mathbf{1.000000} \\ \mathbf{-0.500000} & \mathbf{0.000000} \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

---

#### Step 3: Multimodal Sequence Concatenation
Concatenate $H_v$ ($2 \times 2$) and $H_t$ ($1 \times 2$):
$$H_t = \begin{bmatrix} 3.0 & -1.0 \end{bmatrix}$$

$$H_{\text{seq}} = \begin{bmatrix} H_v \\ H_t \end{bmatrix} = \begin{bmatrix}
-0.500000 & 1.000000 \\
-0.500000 & 0.000000 \\
3.000000 & -1.000000
\end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

- Sequence Position 0: Visual Patch 1 token
- Sequence Position 1: Visual Patch 2 token
- Sequence Position 2: Text Prompt token

---

#### Step 4: Causal Attention Mask Structure ($3 \times 3$)
$$M = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}$$

- At Position 2 (Text Token): It attends to Visual Patch 1 (col 0), Visual Patch 2 (col 1), and itself (col 2). The LLM can now condition its next-token prediction on the image! $\blacksquare$

---

### 5.4 Summary Visual Grid: Multimodal Projection Ledger

| Token Index | Modality | Raw Vector | Intermediate $U$ | Projected $H_{\text{seq}}$ | Causal Visibility |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Pos 0** | Vision (Patch 1) | $[1.0, 2.0]$ | $[1.0, 2.0]$ | $\mathbf{[-0.50, 1.00]}$ | Attends to [Pos 0] |
| **Pos 1** | Vision (Patch 2) | $[0.0, 1.0]$ | $[0.0, 1.0]$ | $\mathbf{[-0.50, 0.00]}$ | Attends to [Pos 0, 1] |
| **Pos 2** | Text (Prompt) | $[3.0, -1.0]$ | — | $\mathbf{[3.00, -1.00]}$ | Attends to [Pos 0, 1, 2] |

---

## 6. Solved Illustrations

### Illustration 1: Context Window Consumption: Fixed vs. Dynamic Patches
**Problem:**
An input image has resolution $672 \times 672$ pixels.
1. If processed by a standard ViT with patch size $p = 14$, how many visual tokens are generated?
2. If using a Perceiver Resampler with $K = 64$, what is the token compression ratio?
3. In a 4,096-token LLM context window, what percentage of the context remains for text conversation under both approaches?

**Solution:**
1. **Standard ViT Patch Count:**
   $$N_v = \left(\frac{672}{14}\right) \times \left(\frac{672}{14}\right) = 48 \times 48 = \mathbf{2,304 \text{ tokens}}$$
2. **Compression with Perceiver Resampler ($K = 64$):**
   $$\text{Compression Ratio} = \frac{2304}{64} = \mathbf{36\times \text{ compression}}$$
3. **Context Window Impact ($L_{\text{ctx}} = 4096$):**
   - **Direct ViT Prefix (LLaVA):**
     $$\text{Remaining for text} = 4096 - 2304 = 1,792 \text{ tokens} \quad (\mathbf{43.75\% \text{ remaining}})$$
   - **Perceiver Resampler (Flamingo):**
     $$\text{Remaining for text} = 4096 - 64 = 4,032 \text{ tokens} \quad (\mathbf{98.44\% \text{ remaining}})! \quad \blacksquare$$

---

### Illustration 2: Patch Embedding for a $4 \times 4$ Image
**Problem:**
Consider a small RGB image $I \in \mathbb{R}^{4 \times 4 \times 3}$. Using a patch size $p=2$, the image yields 4 patches, each containing $2 \times 2 \times 3 = 12$ features.
For the top-left patch (`patch_00`), the flattened 12-dimensional vector is:
$$\text{patch}_{00} = [0.2, 0.8, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.3, 0.7, 0.2]$$
We project this patch into a dimension $d=8$ using a linear embedding matrix $W_e \in \mathbb{R}^{12 \times 8}$. For simplicity, let $W_e$ be the first 8 columns of the $12 \times 12$ identity matrix ($I_{12 \times 8}$). Compute the resulting visual token $t_{00}$.

**Solution:**
1. **Understand the Projection:**
   Multiplying a $1 \times 12$ vector by a $12 \times 8$ matrix that consists of the first 8 columns of an identity matrix simply truncates the vector to its first 8 elements.
   
2. **Compute $t_{00}$:**
   $$t_{00} = \text{patch}_{00} \cdot W_e$$
   $$t_{00} = [0.2, 0.8, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.3, 0.7, 0.2] \cdot I_{12 \times 8}$$
   $$t_{00} = \mathbf{[0.2, 0.8, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4]}$$
   
This token vector represents the spatial patch and is now ready for the Vision Transformer layers. $\blacksquare$

---

### Illustration 3: Cross-Attention Between Vision and Language Tokens
**Problem:**
A Perceiver Resampler uses cross-attention to compress vision tokens. 
Let the vision tokens act as Keys and Values:
$$K = V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ -1 & 0 \end{bmatrix} \quad \text{(4 vision tokens, } d_k=2\text{)}$$
Let the language queries be:
$$Q = \begin{bmatrix} 0.5 & 0.5 \\ 1.0 & -0.5 \end{bmatrix} \quad \text{(2 query tokens, } d_k=2\text{)}$$
Compute the scaled dot-product attention output.

**Solution:**
1. **Compute Scaled Scores ($Q K^T / \sqrt{d_k}$):**
   $$Q K^T = \begin{bmatrix} 0.5 & 0.5 \\ 1.0 & -0.5 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 & -1 \\ 0 & 1 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 & 1.0 & -0.5 \\ 1.0 & -0.5 & 0.5 & -1.0 \end{bmatrix}$$
   Divide by $\sqrt{2} \approx 1.4142$:
   $$\text{Scores} = \begin{bmatrix} 0.3536 & 0.3536 & 0.7071 & -0.3536 \\ 0.7071 & -0.3536 & 0.3536 & -0.7071 \end{bmatrix}$$

2. **Compute Softmax Weights (per row):**
   - **Row 1 Exponentials:** $e^{0.3536}=1.4242, e^{0.7071}=2.0281, e^{-0.3536}=0.7022$. Sum $= 5.5787$.
     $$\text{Softmax}_1 = [0.2553, 0.2553, 0.3635, 0.1259]$$
   - **Row 2 Exponentials:** Sum $= 2.0281 + 0.7022 + 1.4242 + 0.4931 = 4.6476$.
     $$\text{Softmax}_2 = [0.4364, 0.1511, 0.3064, 0.1061]$$

3. **Compute Final Output ($\text{Softmax} \times V$):**
   - **Query 1 Output:**
     $$0.2553[1,0] + 0.2553[0,1] + 0.3635[1,1] + 0.1259[-1,0] = \mathbf{[0.4929, 0.6188]}$$
   - **Query 2 Output:**
     $$0.4364[1,0] + 0.1511[0,1] + 0.3064[1,1] + 0.1061[-1,0] = \mathbf{[0.6367, 0.4575]}$$

The 4 vision tokens have been successfully compressed into 2 token representations! $\blacksquare$

---

### Illustration 4: LLaVA-style MLP Projector Dimension Mapping
**Problem:**
Calculate the number of parameters in the visual MLP projector for a LLaVA-1.5 model.
- Vision Encoder: CLIP ViT-L/14 (outputs $d_v = 1024$).
- Language Model: Mistral-7B (input dimension $d_{\text{text}} = 4096$).
- Projector: 2-layer MLP with GELU activation ($W_1, b_1, W_2, b_2$).
What percentage of the full 7B model parameters does the projector represent?

**Solution:**
1. **Layer 1 ($W_1, b_1$):**
   - $W_1 \in \mathbb{R}^{1024 \times 4096}$, $b_1 \in \mathbb{R}^{4096}$.
   - Params = $(1024 \times 4096) + 4096 = 4,194,304 + 4096 = \mathbf{4,198,400}$.

2. **Layer 2 ($W_2, b_2$):**
   - $W_2 \in \mathbb{R}^{4096 \times 4096}$, $b_2 \in \mathbb{R}^{4096}$.
   - Params = $(4096 \times 4096) + 4096 = 16,777,216 + 4096 = \mathbf{16,781,312}$.

3. **Total Projector Parameters:**
   - Total = $4,198,400 + 16,781,312 = \mathbf{20,979,712 \text{ params}}$ ($\approx 21\text{M}$).

4. **Ratio to 7B Model:**
   - Ratio = $\frac{20,979,712}{7,000,000,000} \approx 0.0030 = \mathbf{0.3\%}$.

**Conclusion:** The cross-modal adapter is incredibly lightweight, requiring only 0.3% of the total parameter budget to endow a blind language model with full vision capabilities! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **LLaVA-1.5 & LLaVA-NeXT:** Open-source standard demonstrating that a frozen CLIP ViT-L/14 coupled with a 2-layer MLP projector to Vicuna/LLaMA achieves state-of-the-art visual instruction following with under \$1,000 in compute training cost!
- **Qwen2-VL & NaViT:** Implements dynamic aspect ratio native resolution patchification (AnyRes), eliminating the need to resize and distort images into square grids.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - 2-layer MLP projection matching $[-0.50, 1.00]$ and $[-0.50, 0.00]$.
   - Multimodal sequence concatenation and causal masking.
2. **Production-Ready Multimodal Vision-Language Modules:**
   - `MLPProjector` module with GELU non-linearity.
   - `PerceiverResampler` module with learnable queries and cross-attention bottleneck.
   - Forward pass concatenating vision tokens and text embeddings, verified with end-to-end backpropagation.

See implementation in:
[`12_modern_llm_architectures/code/12_multimodal_vision_language.py`](./code/12_multimodal_vision_language.py)
