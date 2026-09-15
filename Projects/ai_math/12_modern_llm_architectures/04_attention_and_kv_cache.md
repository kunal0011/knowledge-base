# Chapter 04: Attention Evolution: MHA, MQA, Grouped-Query Attention (GQA) & The KV Cache

---

## 1. Intuition & 101 Motivation

When training a Large Language Model, all tokens in an input sequence are processed simultaneously in a single massive parallel forward pass. In training, high-throughput matrix multiplications ($Q K^T$) fully saturate the compute capabilities of modern GPUs (**compute-bound**).

However, during **inference (text generation)**, language models operate autoregressively:
$$\text{Generate token } y_1 \longrightarrow \text{Generate token } y_2 \longrightarrow \dots \longrightarrow \text{Generate token } y_T$$

If we naively re-ran the full attention matrix across all previous tokens at every single generation step $t$, generation speed would grind to a halt with $O(T^2)$ computational complexity.

### The KV Cache & The Memory Bandwidth Bottleneck
To make inference fast, LLM inference engines employ a **Key-Value Cache (KV Cache)**:
- At each generation step $t$, we compute the Key ($k_t$) and Value ($v_t$) vectors for *only the newly generated token*.
- We append these new vectors to a running cache stored in GPU High-Bandwidth Memory (HBM):
  $$\mathcal{K}_{\le t} = [\mathcal{K}_{<t}, k_t], \quad \mathcal{V}_{\le t} = [\mathcal{V}_{<t}, v_t]$$
- We compute attention between the single new query vector $q_t$ and the cached keys and values.

While this slashes computational complexity from $O(T^2)$ to $O(T)$, it introduces a catastrophic new problem: **The KV Cache Memory Explosion**!
For a 70-billion parameter model serving a batch of 32 requests with an 8,192-token context window in standard Multi-Head Attention (MHA), the KV cache alone demands **over 65 GB of VRAM**! The GPU runs out of memory before even loading a single request.

```
Multi-Head Attention (MHA)            Grouped-Query Attention (GQA)           Multi-Query Attention (MQA)
   [ Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 ]           [ Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 ]           [ Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 ]
    |  |  |  |  |  |  |  |                \  /  \  /  \  /  \  /                \  |  |  |  |  |  |  /
   [ K1 K2 K3 K4 K5 K6 K7 K8 ]           [   K1    K2    K3    K4   ]           [            K1            ]
   [ V1 V2 V3 V4 V5 V6 V7 V8 ]           [   V1    V2    V3    V4   ]           [            V1            ]
   Ratio 1:1 (Max Cache Size)            Ratio 2:1 / 4:1 / 8:1 (LLaMA 3)        Ratio 8:1 (Minimal Cache Size)
```

Enter **Grouped-Query Attention (GQA)** (Ainslie et al., 2023):
- Instead of maintaining dedicated Key and Value heads for every single Query head (MHA), multiple Query heads share a single Key and Value head!
- GQA provides an ideal sweet spot: it slashes KV cache memory consumption by **$4\times$ to $8\times$** while matching the perplexity and downstream accuracy of full MHA.
- Today, GQA is the universal attention standard adopted by **LLaMA 2/3, Mistral, Mixtral, Gemma 2, and Qwen 2.5**.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Attention Architectural Spectrum

Let the hidden dimension be $d$, number of Query heads be $H_q$, and head dimension be $d_{\text{head}} = d / H_q$.
Let $H_{kv}$ denote the number of Key and Value heads.

The relationship between $H_q$ and $H_{kv}$ defines the three core paradigms:

#### 1. Multi-Head Attention (MHA, Vaswani et al., 2017):
$$H_{kv} = H_q$$
Every query head has its own private key and value head ($1:1$ ratio).
- **Pros:** Highest modeling capacity.
- **Cons:** Maximum KV cache memory footprint.

#### 2. Multi-Query Attention (MQA, Shazeer, 2019):
$$H_{kv} = 1$$
All $H_q$ query heads share a single key head and a single value head ($H_q : 1$ ratio).
- **Pros:** Minimal KV cache memory consumption ($H_q \times$ smaller than MHA).
- **Cons:** Measurable drop in model perplexity and reasoning capacity on large-scale tasks.

#### 3. Grouped-Query Attention (GQA, Ainslie et al., 2023):
$$1 < H_{kv} < H_q$$
The $H_q$ query heads are partitioned into $H_{kv}$ distinct groups. Each group contains $R = H_q / H_{kv}$ query heads that share a single key head and value head.
- **Example (LLaMA 3-8B):** $H_q = 32$, $H_{kv} = 8 \implies R = 4$ query heads per KV head ($4\times$ smaller KV cache).
- **Example (LLaMA 3-70B):** $H_q = 64$, $H_{kv} = 8 \implies R = 8$ query heads per KV head ($8\times$ smaller KV cache).

---

### 2.2 Tensor Dimensions & The Repeat-Interleave Operation

Let:
- Batch size: $B$
- Sequence length: $L$
- $Q \in \mathbb{R}^{B \times L \times H_q \times d_{\text{head}}}$
- $K \in \mathbb{R}^{B \times L \times H_{kv} \times d_{\text{head}}}$
- $V \in \mathbb{R}^{B \times L \times H_{kv} \times d_{\text{head}}}$

To execute multi-head attention, the Key and Value heads are expanded by repeating each head $R = H_q / H_{kv}$ times along the head dimension:

$$K_{\text{expanded}} = \operatorname{repeat\_interleave}(K, \text{dim}=2, \text{repeats}=R) \in \mathbb{R}^{B \times L \times H_q \times d_{\text{head}}}$$
$$V_{\text{expanded}} = \operatorname{repeat\_interleave}(V, \text{dim}=2, \text{repeats}=R) \in \mathbb{R}^{B \times L \times H_q \times d_{\text{head}}}$$

The attention output is computed via standard scaled dot-product attention:

$$\operatorname{Attention}(Q, K_{\text{expanded}}, V_{\text{expanded}}) = \operatorname{softmax}\left( \frac{Q K_{\text{expanded}}^T}{\sqrt{d_{\text{head}}}} + \mathbf{M} \right) V_{\text{expanded}}$$

where $\mathbf{M}$ is the causal lower-triangular attention mask.

---

### 2.3 The KV Cache Step-by-Step State Update

During autoregressive generation at step $t$, the input to the Transformer is strictly the single new token $x_t \in \mathbb{R}^{B \times 1 \times d}$:

1. **Linear Projections:**
   $$q_t = x_t W_Q \in \mathbb{R}^{B \times 1 \times H_q \times d_{\text{head}}}$$
   $$k_t = x_t W_K \in \mathbb{R}^{B \times 1 \times H_{kv} \times d_{\text{head}}}$$
   $$v_t = x_t W_V \in \mathbb{R}^{B \times 1 \times H_{kv} \times d_{\text{head}}}$$

2. **Rotary Position Embedding:**
   Rotate $q_t$ at position $t$ and $k_t$ at position $t$:
   $$q_t \leftarrow \operatorname{RoPE}(q_t, t), \quad k_t \leftarrow \operatorname{RoPE}(k_t, t)$$

3. **Cache Update (Concatenation):**
   Append the new key and value vectors to the persistent cache tensors:
   $$K_{\text{cache}} \leftarrow \big[ K_{\text{cache}}, \; k_t \big] \in \mathbb{R}^{B \times (t+1) \times H_{kv} \times d_{\text{head}}}$$
   $$V_{\text{cache}} \leftarrow \big[ V_{\text{cache}}, \; v_t \big] \in \mathbb{R}^{B \times (t+1) \times H_{kv} \times d_{\text{head}}}$$

4. **Grouped Attention Computation:**
   Broadcast $K_{\text{cache}}$ and $V_{\text{cache}}$ across the $R$ query heads in each group and compute the single-row attention:
   $$\operatorname{Score}_t = \frac{q_t \cdot \operatorname{repeat}(K_{\text{cache}})^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{B \times H_q \times 1 \times (t+1)}$$
   $$\operatorname{Attn}_t = \operatorname{softmax}(\operatorname{Score}_t) \cdot \operatorname{repeat}(V_{\text{cache}}) \in \mathbb{R}^{B \times 1 \times H_q \times d_{\text{head}}}$$

---

### 2.4 KV Cache Memory Footprint Derivation

The total VRAM consumed by the KV cache across a network of $N_{\text{layers}}$ layers is:

$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times N_{\text{layers}} \times H_{kv} \times d_{\text{head}} \times P_{\text{bytes}}$$

where:
- Factor $2$: Stores both Keys and Values ($K$ and $V$).
- $B$: Batch size (concurrent user requests).
- $L$: Context window length in tokens.
- $N_{\text{layers}}$: Total transformer layers.
- $H_{kv}$: Number of Key/Value heads.
- $d_{\text{head}}$: Dimension per attention head.
- $P_{\text{bytes}}$: Bytes per element (2 bytes for FP16/BF16, 1 byte for FP8).

---

## 3. Geometric & Physical Interpretation

### 3.1 The Roofline Model: Arithmetic Intensity
In computer architecture, the **Arithmetic Intensity** is the ratio of compute operations to memory bytes transferred:
$$I = \frac{\text{FLOPs}}{\text{Bytes Transferred from HBM}}$$

```
Performance (TFLOPs)
 ^
 |             /------------------------- Compute Bound Roofline (Matrix Multiplies)
 |            /
 |           /  <--- GQA / FlashDecoding (High Intensity: 15-30 FLOPs/Byte)
 |          /
 |         /    <--- MHA Inference (Memory-Bandwidth Bound: ~1 FLOP/Byte)
 |        /
 |_______/________________________________> Arithmetic Intensity (FLOPs/Byte)
```
- In standard MHA inference, the GPU loads hundreds of megabytes of KV tensors from high-bandwidth memory (HBM) into on-chip cache (SRAM) just to compute a single vector dot product. The Tensor Cores spend $90\%$ of their clock cycles stalled waiting for memory.
- GQA reduces memory traffic by **$4\times$ to $8\times$**, moving the operational point substantially closer to the compute roofline!

---

## 4. Real-World Analogy: The Corporate Legal Department

Consider a major corporation with 32 senior attorneys (Query Heads):
- **Multi-Head Attention (MHA):** Every single attorney hires a personal research paralegal (32 Key/Value heads). The law firm must maintain 32 duplicate filing cabinets (huge KV cache storage).
- **Multi-Query Attention (MQA):** All 32 attorneys share 1 single paralegal. When attorneys have specialized, conflicting requests, the single paralegal drops crucial details (perplexity degradation).
- **Grouped-Query Attention (GQA):** Attorneys are organized into 8 specialized practice groups of 4 attorneys each (Tax, IP, Litigation, Corporate, etc.). Each 4-attorney group shares 1 dedicated senior paralegal (8 paralegals total). Memory storage is reduced by $4\times$, while retaining deep specialization!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace the complete numerical mechanics of a **Grouped-Query Attention (GQA)** computation by hand on a concrete 4-Query, 2-KV system.

---

### 5.1 System & Parameter Setup
- Batch size: $B = 1$
- Sequence length: $L = 1$ (single query token at step $t$)
- Head dimension: $d_{\text{head}} = 2 \implies \sqrt{d_{\text{head}}} = \sqrt{2} \approx 1.414214$
- Number of Query heads: $H_q = 4$
- Number of Key/Value heads: $H_{kv} = 2$
- Group ratio: $R = \frac{H_q}{H_{kv}} = \frac{4}{2} = 2$
  - **Group 0 (Heads 0 and 1):** Shares Key head $K_0$ and Value head $V_0$.
  - **Group 1 (Heads 2 and 3):** Shares Key head $K_1$ and Value head $V_1$.

#### Input Vectors:
- Query heads:
  $$Q_0 = [1.0, 2.0], \quad Q_1 = [2.0, 0.0], \quad Q_2 = [-1.0, 1.0], \quad Q_3 = [0.0, -2.0]$$
- Key heads (only 2):
  $$K_0 = [1.0, 1.0], \quad K_1 = [2.0, -1.0]$$
- Value heads (only 2):
  $$V_0 = [0.5, 0.5], \quad V_1 = [1.0, -1.0]$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $Q_i$ | `query_heads[i]` | Query vector for attention head $i \in \{0, 1, 2, 3\}$ |
| $K_g, V_g$ | `key_heads[g], val_heads[g]` | Key and Value vectors for group $g \in \{0, 1\}$ |
| $K_{\text{exp}}$ | `key_expanded` | Expanded key tensor repeating each head $R = 2$ times |
| $\text{Dot}_i$ | `dot_product[i]` | Inner product: $Q_i \cdot K_{\text{exp}, i}$ |
| $z_i$ | `scaled_score[i]` | Scaled attention logit: $\text{Dot}_i / \sqrt{2}$ |
| $\text{Out}_i$ | `attn_output[i]` | Final attention head output vector |

---

### 5.3 Step-by-Step Hand Calculations: GQA Forward Pass

#### Step 1: Expand Key and Value Heads
Since $R = 2$, each KV head is repeated twice:
$$K_{\text{exp}} = \begin{bmatrix} K_0 \\ K_0 \\ K_1 \\ K_1 \end{bmatrix} = \begin{bmatrix} [1.0, 1.0] \\ [1.0, 1.0] \\ [2.0, -1.0] \\ [2.0, -1.0] \end{bmatrix}, \quad V_{\text{exp}} = \begin{bmatrix} V_0 \\ V_0 \\ V_1 \\ V_1 \end{bmatrix} = \begin{bmatrix} [0.5, 0.5] \\ [0.5, 0.5] \\ [1.0, -1.0] \\ [1.0, -1.0] \end{bmatrix}$$

---

#### Step 2: Compute Dot Products & Scaled Attention Logits
Recall $\sqrt{d_{\text{head}}} = \sqrt{2} \approx \mathbf{1.414214}$.

- **Head 0 (Group 0, with $K_0$):**
  $$\text{Dot}_0 = Q_0 \cdot K_0 = 1.0 \times 1.0 + 2.0 \times 1.0 = 1.0 + 2.0 = \mathbf{3.000000}$$
  $$z_0 = \frac{3.000000}{1.414214} \approx \mathbf{2.121320}$$

- **Head 1 (Group 0, with $K_0$):**
  $$\text{Dot}_1 = Q_1 \cdot K_0 = 2.0 \times 1.0 + 0.0 \times 1.0 = 2.0 + 0.0 = \mathbf{2.000000}$$
  $$z_1 = \frac{2.000000}{1.414214} \approx \mathbf{1.414214}$$

- **Head 2 (Group 1, with $K_1$):**
  $$\text{Dot}_2 = Q_2 \cdot K_1 = (-1.0) \times 2.0 + 1.0 \times (-1.0) = -2.0 - 1.0 = \mathbf{-3.000000}$$
  $$z_2 = \frac{-3.000000}{1.414214} \approx \mathbf{-2.121320}$$

- **Head 3 (Group 1, with $K_1$):**
  $$\text{Dot}_3 = Q_3 \cdot K_1 = 0.0 \times 2.0 + (-2.0) \times (-1.0) = 0.0 + 2.0 = \mathbf{2.000000}$$
  $$z_3 = \frac{2.000000}{1.414214} \approx \mathbf{1.414214}$$

---

#### Step 3: Softmax & Output Projection
Since there is only $L = 1$ token in this step, $\operatorname{softmax}([z_i]) = [1.000000]$.
The output of each attention head is directly its associated expanded Value vector:
- **Head 0 Output:**
  $$\text{Out}_0 = 1.0 \times V_0 = \mathbf{[0.500000, 0.500000]}$$
- **Head 1 Output:**
  $$\text{Out}_1 = 1.0 \times V_0 = \mathbf{[0.500000, 0.500000]}$$
- **Head 2 Output:**
  $$\text{Out}_2 = 1.0 \times V_1 = \mathbf{[1.000000, -1.000000]}$$
- **Head 3 Output:**
  $$\text{Out}_3 = 1.0 \times V_1 = \mathbf{[1.000000, -1.000000]}$$

#### Final Concatenated Multi-Head Output:
$$\text{Out}_{\text{concat}} = [\text{Out}_0, \text{Out}_1, \text{Out}_2, \text{Out}_3] = \mathbf{[0.5, 0.5, 0.5, 0.5, 1.0, -1.0, 1.0, -1.0]} \in \mathbb{R}^8$$

---

### 5.4 Summary Visual Grid: GQA Architecture & Memory Ledger

| Query Head | Assigned Group | Shared Key Head | Shared Value Head | Dot Product $Q_i \cdot K_g$ | Scaled Logit $z_i$ | Head Output Vector |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$Q_0 = [1, 2]$** | **Group 0** | $K_0 = [1, 1]$ | $V_0 = [0.5, 0.5]$ | $1(1) + 2(1) = 3.0$ | **$+2.1213$** | **$[0.5, 0.5]$** |
| **$Q_1 = [2, 0]$** | **Group 0** | $K_0 = [1, 1]$ | $V_0 = [0.5, 0.5]$ | $2(1) + 0(1) = 2.0$ | **$+1.4142$** | **$[0.5, 0.5]$** |
| **$Q_2 = [-1, 1]$**| **Group 1** | $K_1 = [2, -1]$ | $V_1 = [1.0, -1.0]$ | $-1(2) + 1(-1) = -3.0$| **$-2.1213$** | **$[1.0, -1.0]$** |
| **$Q_3 = [0, -2]$**| **Group 1** | $K_1 = [2, -1]$ | $V_1 = [1.0, -1.0]$ | $0(2) - 2(-1) = 2.0$ | **$+1.4142$** | **$[1.0, -1.0]$** |

---

## 6. Solved Illustrations

### Illustration 1: Memory Footprint of 70B Model KV Cache
**Problem:**
Calculate the exact KV cache memory requirement for a 70B parameter model serving a batch of $B = 32$ requests with context length $L = 8,192$ under:
1. Standard MHA ($H_q = 64, H_{kv} = 64, d_{\text{head}} = 128, N_{\text{layers}} = 80$) in FP16 (2 bytes).
2. Grouped-Query Attention ($H_q = 64, H_{kv} = 8, d_{\text{head}} = 128, N_{\text{layers}} = 80$) in FP16.

**Solution:**
1. **Multi-Head Attention (MHA):**
   $$\text{Bytes} = 2 \times B \times L \times N_{\text{layers}} \times H_{kv} \times d_{\text{head}} \times 2$$
   $$\text{Bytes} = 2 \times 32 \times 8192 \times 80 \times 64 \times 128 \times 2 = 687,194,767,360 \text{ bytes} = \mathbf{640.0 \text{ GiB}}$$
   *(This would require eight 80 GB A100 GPUs solely to store the KV cache for 32 users!)*

2. **Grouped-Query Attention (GQA, $H_{kv} = 8$):**
   $$\text{Bytes} = 2 \times 32 \times 8192 \times 80 \times 8 \times 128 \times 2 = 85,899,345,920 \text{ bytes} = \mathbf{80.0 \text{ GiB}}$$
   **Result:** GQA delivers an **$8\times$ reduction in VRAM consumption** ($640 \text{ GiB} \to 80 \text{ GiB}$), allowing high-throughput batched serving on a standard node! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **LLaMA 3 (8B and 70B):** Both models standardize on $H_{kv} = 8$ heads, allowing LLaMA 3 to handle 128k context lengths during serving without out-of-memory crashes.
- **DeepSeek-V2 / V3 Multi-Head Latent Attention (MLA):** DeepSeek took GQA to its theoretical frontier: instead of grouping heads, MLA compresses the entire Key and Value vectors into a low-dimensional latent vector $c_{kv} \in \mathbb{R}^{512}$ via low-rank matrix projection, slashing the KV cache by an astonishing **$93\%$** compared to standard MHA!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Group broadcasting of $K$ and $V$ vectors.
   - Scaled attention logits matching $z = [2.121320, 1.414214, -2.121320, 1.414214]$ to $< 10^{-6}$.
   - Exact concatenation of multi-head output $[0.5, 0.5, 0.5, 0.5, 1.0, -1.0, 1.0, -1.0]$.
2. **Production-Ready PyTorch GQA Layer with KV Cache:**
   - Vectorized `GroupedQueryAttention` class supporting arbitrary $H_q, H_{kv}$.
   - Autoregressive generation simulation with incremental KV caching.
   - Exact numerical output match between prefill and cached single-token generation.

See implementation in:
[`12_modern_llm_architectures/code/04_attention_and_kv_cache.py`](./code/04_attention_and_kv_cache.py)
