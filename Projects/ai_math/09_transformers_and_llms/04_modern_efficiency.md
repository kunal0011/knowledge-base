# Chapter 9.4: Modern Efficiency (MQA, GQA, FlashAttention, MoE, KV-Cache)

---

## 1. Intuition & 101 Motivation

As Large Language Models scaled from billions to hundreds of billions of parameters, artificial intelligence hit a fundamental physical bottleneck: **The Memory Bandwidth Wall**.

During training and prompt prefill, operations are **compute-bound**: GPUs multiply large matrices where every byte loaded from memory is reused across thousands of arithmetic operations.

However, during autoregressive token generation (the decode phase), the model produces tokens **one by one**:
- To generate a single new token, the GPU must stream tens of billions of weights and gigabytes of past Key-Value states from High Bandwidth Memory (HBM) into on-chip cache.
- Modern GPUs like the NVIDIA H100 can perform $2,000 \text{ TFLOPs}$ of computation per second, but can only stream memory at $3.35 \text{ TB/s}$.
- The arithmetic intensity plummets to $\sim 1 \text{ FLOP/byte}$. The GPU cores sit idle over $95\%$ of the time, waiting for memory buses to deliver data!

To conquer this memory wall, modern LLM architecture underwent four seismic engineering innovations:
1. **KV-Caching:** Caching previous Keys and Values to avoid quadratic $\mathcal{O}(T^2)$ recomputation at each step.
2. **Multi-Query Attention (MQA) & Grouped-Query Attention (GQA):** Compressing KV-cache memory by $8\times$ to $32\times$ by sharing Key/Value heads across Query heads.
3. **FlashAttention:** Reorganizing attention algorithms to execute entirely within fast on-chip SRAM using tiling and online softmax, completely avoiding writing the $\mathcal{O}(T^2)$ attention matrix to DRAM.
4. **Mixture of Experts (MoE):** Decoupling total parameter capacity from FLOP compute by dynamically routing each token to a sparse subset of expert feedforward networks (e.g., Mixtral 8x7B, DeepSeek-V3).

---

## 2. Rigorous Mathematical Formulation

```
                     MHA vs. GQA vs. MQA ARCHITECTURAL COMPARISON
                     
        MHA (Multi-Head)              GQA (Grouped-Query)            MQA (Multi-Query)
     Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8   Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8   Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8
     │   │   │   │   │   │   │   │    └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘   └───────────┬───────────┘
     ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼       ▼       ▼       ▼       ▼                 ▼
     K1  K2  K3  K4  K5  K6  K7  K8      K1      K2      K3      K4                K1
     V1  V2  V3  V4  V5  V6  V7  V8      V1      V2      V3      V4                V1
     (8 Key/Value Heads)              (4 Key/Value Heads - 2x)         (1 Key/Value Head - 8x)
```

---

### 2.1 The KV-Cache Mechanics

In naive autoregressive decoding, predicting token $t+1$ given sequence $(x_1, \dots, x_t)$ recomputes Keys and Values for all previous $t$ tokens from scratch:
$$\text{Cost for } T \text{ tokens} = \sum_{t=1}^T \mathcal{O}(t) = \mathcal{O}(T^2) \text{ operations}$$

Because previous tokens are fixed, their projected Key and Value vectors never change:
$$\mathbf{k}_{\tau} = \mathbf{x}_{\tau} \mathbf{W}^K, \quad \mathbf{v}_{\tau} = \mathbf{x}_{\tau} \mathbf{W}^V \quad (\forall \tau \le t)$$

#### The KV-Cache Invariant:
At step $t+1$, compute $\mathbf{q}_{t+1}, \mathbf{k}_{t+1}, \mathbf{v}_{t+1}$ only for the single newly generated token.
Append them to the cached history:
$$\mathbf{K}_{\text{cache}} \leftarrow \left[ \mathbf{K}_{\text{cache}}; \, \mathbf{k}_{t+1} \right] \in \mathbb{R}^{(t+1) \times d_k}$$
$$\mathbf{V}_{\text{cache}} \leftarrow \left[ \mathbf{V}_{\text{cache}}; \, \mathbf{v}_{t+1} \right] \in \mathbb{R}^{(t+1) \times d_v}$$
The Query $\mathbf{q}_{t+1} \in \mathbb{R}^{1 \times d_k}$ attends against the cached history:
$$\mathbf{o}_{t+1} = \text{softmax}\left( \frac{\mathbf{q}_{t+1} \mathbf{K}_{\text{cache}}^T}{\sqrt{d_k}} \right) \mathbf{V}_{\text{cache}} \in \mathbb{R}^{1 \times d_v}$$
This reduces per-step compute from $\mathcal{O}(t)$ full forward passes to a single vector-matrix multiply ($\mathcal{O}(1)$ projection $+ \mathcal{O}(t)$ dot product).

#### Exact KV-Cache Memory Formula:
For a model with $L$ layers, batch size $B$, sequence length $T$, and $H_{KV}$ key-value heads with dimension $d_k$:
$$\text{Memory}_{\text{KV}} = 2 \times B \times T \times L \times H_{KV} \times d_k \times \text{bytes\_per\_elem}$$
*(The factor of $2$ accounts for both Keys and Values).*

---

### 2.2 MHA vs. MQA vs. GQA (Ainslie et al., 2023)

Let $H_Q$ be the number of query heads, and $H_{KV}$ be the number of key/value heads:

1. **Multi-Head Attention (MHA):**
   $$H_{KV} = H_Q$$
   Every query head has its own dedicated key and value head. Maximum representational capacity, but massive KV-cache footprint.

2. **Multi-Query Attention (MQA, Shazeer 2019):**
   $$H_{KV} = 1$$
   All $H_Q$ query heads share a single key head and a single value head.
   $$\text{KV Cache Reduction Factor} = H_Q \quad (\text{e.g., } 32\times \text{ to } 64\times \text{ memory savings!})$$
   *Drawback:* Can suffer slight quality degradation and training instability on complex multi-task benchmarks.

3. **Grouped-Query Attention (GQA, Ainslie et al., 2023):**
   $$1 < H_{KV} < H_Q$$
   $H_Q$ query heads are partitioned into $G = H_{KV}$ groups. Each group contains $H_Q / H_{KV}$ query heads that share a single key and value head.
   - For example, in **LLaMA-3 70B**: $H_Q = 64$ and $H_{KV} = 8$.
   - Cache reduction factor: $\frac{64}{8} = \mathbf{8\times \text{ smaller KV-cache}}$ with **zero loss in accuracy** compared to full MHA.

---

### 2.3 FlashAttention & The Online Softmax Algorithm (Dao et al., 2022, 2023)

Standard attention implementations are memory-bandwidth bound because they materialize intermediate $T \times T$ matrices in GPU High-Bandwidth Memory (HBM):
$$\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \in \text{HBM} \xrightarrow{\text{Read/Write}} \mathbf{A} = \text{softmax}(\mathbf{S}) \in \text{HBM} \xrightarrow{\text{Read}} \mathbf{O} = \mathbf{A} \mathbf{V} \in \text{HBM}$$
Writing and re-reading $\mathbf{S}$ and $\mathbf{A}$ requires $\mathcal{O}(T^2)$ memory transactions.

Tri Dao et al. developed **FlashAttention**, which loads blocks of $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ directly into ultra-fast on-chip SRAM ($\approx 19 \text{ TB/s}$), computes attention tiles locally, and writes back only the final output $\mathbf{O}$ ($\mathcal{O}(T)$ memory access).

```
                      GPU MEMORY HIERARCHY
      ┌────────────────────────────────────────────────────────┐
      │  HBM (High Bandwidth Memory): 40-80 GB | ~2-3 TB/s    │
      └───────────────────────────▲────────────────────────────┘
                                  │ (FlashAttention minimizes this traffic)
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  On-Chip SRAM: ~200 KB per SM (Total ~20-50 MB)       │
      │  Speed: ~19 TB/s (Over 6x faster than HBM!)            │
      └────────────────────────────────────────────────────────┘
```

#### The Online Softmax Mathematics (Milakov & Gimelshein, 2018):
Standard softmax requires seeing all $T$ elements to compute the normalization constant:
$$m = \max_j x_j, \quad l = \sum_j e^{x_j - m}, \quad \text{softmax}(x_i) = \frac{e^{x_i - m}}{l}$$
How can we compute attention in blocks without knowing the global maximum?

Let Block 1 have local maximum $m^{(1)}$ and local sum $l^{(1)}$ with output $\mathbf{O}^{(1)}$.
When Block 2 arrives with local maximum $m^{(2)}$ and local sum $l^{(2)}$:
1. **Update Global Maximum:**
   $$m^{\text{new}} = \max(m^{(1)}, m^{(2)})$$
2. **Rescale Previous and New Sums:**
   $$l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} l^{(1)} + e^{m^{(2)} - m^{\text{new}}} l^{(2)}$$
3. **Rescale and Accumulate Output Vector:**
   $$\mathbf{O}^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} l^{(1)}}{l^{\text{new}}} \mathbf{O}^{(1)} + \frac{e^{m^{(2)} - m^{\text{new}}}}{l^{\text{new}}} \left( \mathbf{A}^{(2)} \mathbf{V}^{(2)} \right)$$

This exact recurrence enables tiling attention across blocks of size $B_r \times B_c$ entirely within SRAM, eliminating the $T \times T$ intermediate matrix completely!

---

### 2.4 Sparse Mixture of Experts (MoE) (Shazeer et al., 2017)

In a standard dense Transformer, every token passes through the exact same Feedforward Network (FFN/MLP). In a **Mixture of Experts (MoE)** model:
- The dense MLP layer is replaced by $E$ independent expert MLPs $\{\text{Expert}_1, \dots, \text{Expert}_E\}$.
- A lightweight **Gating Router Network** computes routing logits:
  $$\mathbf{H}(\mathbf{x}) = \mathbf{x} \mathbf{W}_g \in \mathbb{R}^E$$
- A **Top-$k$ Softmax operator** selects the top $k$ experts (typically $k=2$ out of $E=8$ or $E=64$):
  $$\mathcal{T} = \operatorname{TopK}(\mathbf{H}(\mathbf{x}), k)$$
  $$G(\mathbf{x})_i = \begin{cases} \frac{\exp(H(\mathbf{x})_i)}{\sum_{j \in \mathcal{T}} \exp(H(\mathbf{x})_j)} & \text{if } i \in \mathcal{T} \\ 0 & \text{otherwise} \end{cases}$$
- The output is the dynamically weighted sum of only the $k$ activated experts:
  $$\mathbf{y} = \sum_{i \in \mathcal{T}} G(\mathbf{x})_i \cdot \text{Expert}_i(\mathbf{x})$$

#### Decoupling Parameters from FLOPs:
- **Total Parameters:** Proportional to $E$ (e.g., Mixtral 8x7B has $47\text{B}$ parameters).
- **Compute per Token:** Proportional to only $k$ active experts (e.g., Mixtral uses only $13\text{B}$ active parameters per token).
- MoE models achieve the capacity and knowledge retention of a massive model with the inference speed and FLOP cost of a small model.

#### The Auxiliary Load-Balancing Loss:
Without regularization, the router suffers from **winner-take-all collapse**: it routes all tokens to 1 or 2 favorite experts, leaving the remaining $E-2$ experts completely untrained.
To enforce uniform expert utilization, an auxiliary loss is added to the objective:
$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i P_i$$
where:
- $f_i = \frac{1}{T} \sum_{t=1}^T \mathbf{1}(i \in \mathcal{T}_t)$ is the actual fraction of tokens assigned to expert $i$.
- $P_i = \frac{1}{T} \sum_{t=1}^T \text{softmax}(\mathbf{H}(\mathbf{x}_t))_i$ is the average routing probability for expert $i$.
The product $f_i P_i$ is minimized when distributions are uniform ($f_i = 1/E, P_i = 1/E$).

---

### 2.5 Deep Derivation 9.4.1: Mathematical Proof of FlashAttention Tiled Online Softmax Invariance

#### Context and Setup
Standard scaled dot-product attention computes:
$$\mathbf{O} = \operatorname{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} \right) \mathbf{V} \in \mathbb{R}^{T \times d}$$
Let an arbitrary row of attention logits be $\mathbf{s} \in \mathbb{R}^N$ and corresponding values $\mathbf{V} \in \mathbb{R}^{N \times d}$.
Partition the sequence into $B$ consecutive blocks:
$$\mathbf{s} = [\mathbf{s}^{(1)}, \mathbf{s}^{(2)}, \dots, \mathbf{s}^{(B)}], \quad \mathbf{V} = [\mathbf{V}^{(1)}; \mathbf{V}^{(2)}; \dots; \mathbf{V}^{(B)}]$$
where each block has length $K = N / B$.

#### Theorem: Exact Tiled Online Softmax Accumulation
For any block $b \in \{1, \dots, B\}$, define the local statistics:
$$m^{(b)} = \max_{j} s_j^{(b)}, \quad l^{(b)} = \sum_{j=1}^K \exp\left(s_j^{(b)} - m^{(b)}\right), \quad \mathbf{O}^{(b)} = \frac{1}{l^{(b)}} \sum_{j=1}^K \exp\left(s_j^{(b)} - m^{(b)}\right) \mathbf{v}_j^{(b)}$$
Let $(m_k, l_k, \mathbf{O}_k)$ be the running state after processing the first $k$ blocks ($k \ge 1$), initialized at $k=1$ with $(m^{(1)}, l^{(1)}, \mathbf{O}^{(1)})$.
Then updating with block $k+1$ via the recurrence:
$$m_{k+1} = \max\left(m_k, \, m^{(k+1)}\right)$$
$$l_{k+1} = e^{m_k - m_{k+1}} l_k + e^{m^{(k+1)} - m_{k+1}} l^{(k+1)}$$
$$\mathbf{O}_{k+1} = \frac{e^{m_k - m_{k+1}} l_k}{l_{k+1}} \mathbf{O}_k + \frac{e^{m^{(k+1)} - m_{k+1}} l^{(k+1)}}{l_{k+1}} \mathbf{O}^{(k+1)}$$
yields at the final block $B$ the exact global softmax output:
$$\mathbf{O}_B \equiv \frac{\sum_{j=1}^N e^{s_j - m_B} \mathbf{v}_j}{\sum_{j=1}^N e^{s_j - m_B}} = \operatorname{softmax}(\mathbf{s}) \mathbf{V}$$

#### Proof:
1. **Base Case ($k=1$):**
   By definition, $\mathbf{O}_1 = \frac{\sum_{j=1}^K e^{s_j - m_1} \mathbf{v}_j}{\sum_{j=1}^K e^{s_j - m_1}}$, which is exact for the first block.

2. **Inductive Step:**
   Assume that for step $k$, the running state represents the exact normalized attention over all tokens in blocks $1$ through $k$:
   $$l_k = \sum_{j=1}^{k K} e^{s_j - m_k}, \quad \mathbf{O}_k = \frac{1}{l_k} \sum_{j=1}^{k K} e^{s_j - m_k} \mathbf{v}_j$$
   Consider step $k+1$:
   $$m_{k+1} = \max(m_k, m^{(k+1)}) = \max_{j \in \{1, \dots, (k+1)K\}} s_j$$
   Substitute the induction hypothesis into the running sum $l_{k+1}$:
   $$l_{k+1} = e^{m_k - m_{k+1}} \left( \sum_{j=1}^{k K} e^{s_j - m_k} \right) + e^{m^{(k+1)} - m_{k+1}} \left( \sum_{j=k K + 1}^{(k+1) K} e^{s_j - m^{(k+1)}} \right)$$
   Distributing the exponential scale factors:
   $$l_{k+1} = \sum_{j=1}^{k K} e^{s_j - m_{k+1}} + \sum_{j=k K + 1}^{(k+1) K} e^{s_j - m_{k+1}} = \sum_{j=1}^{(k+1) K} e^{s_j - m_{k+1}}$$
   which matches the exact definition of the global softmax denominator over $(k+1)$ blocks.

3. **Inductive Step for Output Vector $\mathbf{O}_{k+1}$:**
   Substitute $\mathbf{O}_k$ and $\mathbf{O}^{(k+1)}$:
   $$\mathbf{O}_{k+1} = \frac{1}{l_{k+1}} \left[ e^{m_k - m_{k+1}} l_k \left( \frac{1}{l_k} \sum_{j=1}^{k K} e^{s_j - m_k} \mathbf{v}_j \right) + e^{m^{(k+1)} - m_{k+1}} l^{(k+1)} \left( \frac{1}{l^{(k+1)}} \sum_{j=k K + 1}^{(k+1) K} e^{s_j - m^{(k+1)}} \mathbf{v}_j \right) \right]$$
   Canceling $l_k$ and $l^{(k+1)}$:
   $$\mathbf{O}_{k+1} = \frac{1}{l_{k+1}} \left[ \sum_{j=1}^{k K} e^{s_j - m_{k+1}} \mathbf{v}_j + \sum_{j=k K + 1}^{(k+1) K} e^{s_j - m_{k+1}} \mathbf{v}_j \right] = \frac{\sum_{j=1}^{(k+1) K} e^{s_j - m_{k+1}} \mathbf{v}_j}{\sum_{j=1}^{(k+1) K} e^{s_j - m_{k+1}}}$$
   By mathematical induction, at block $B$, $\mathbf{O}_B$ is algebraically identical to standard global softmax attention. FlashAttention performs zero numerical approximation. $\blacksquare$

---

### 2.6 Deep Derivation 9.4.2: FlashAttention Backward Pass with SRAM Recomputation

#### The Memory Bottleneck of Standard Attention Backprop
Standard backpropagation through attention requires the forward attention matrix $\mathbf{A} \in \mathbb{R}^{T \times T}$ to compute:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{A}^T \frac{\partial \mathcal{L}}{\partial \mathbf{O}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{S}} = \mathbf{A} \odot \left( \frac{\partial \mathcal{L}}{\partial \mathbf{A}} - \left[(\dots)\right] \right)$$
Storing $\mathbf{A}$ across $L$ layers and $H$ heads consumes $\mathcal{O}(L \cdot H \cdot T^2)$ bytes in HBM, causing out-of-memory crashes for $T > 4,096$.

#### Theorem: Recomputation from Logsumexp in SRAM
FlashAttention avoids storing $\mathbf{A}$ by saving only the **softmax log-partition vector** $\mathbf{L} \in \mathbb{R}^T$:
$$L_i = m_i + \log l_i = \log \left( \sum_{j=1}^T \exp(S_{i, j}) \right)$$
which requires only $\mathcal{O}(T)$ memory in HBM!

#### Proof of Backward Recomputation:
1. **On-the-Fly Recomputation of Attention Weights:**
   During the backward pass, for each block of queries $\mathbf{Q}_i$ and keys $\mathbf{K}_j$, the scores are recomputed in on-chip SRAM:
   $$\mathbf{S}_{i, j} = \frac{\mathbf{Q}_i \mathbf{K}_j^T}{\sqrt{d}}$$
   The attention weights are recovered instantly using the saved vector $\mathbf{L}_i$:
   $$A_{i, j} = \exp\left( S_{i, j} - L_i \right)$$

2. **Backward Adjoint Derivation:**
   Let $\mathbf{D}_i = \sum_{k=1}^d \left( \frac{\partial \mathcal{L}}{\partial O_{i, k}} \right) O_{i, k} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{o}_i}\right)^T \mathbf{o}_i \in \mathbb{R}$.
   Then the score derivative simplifies to:
   $$\frac{\partial \mathcal{L}}{\partial S_{i, j}} = A_{i, j} \left( \left(\frac{\partial \mathcal{L}}{\partial \mathbf{o}_i}\right)^T \mathbf{v}_j - D_i \right)$$
   This form computes the backward gradient in SRAM without reading or writing any $T \times T$ intermediate matrix to HBM.
   The total HBM memory reads/writes drop from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$, achieving a $2\times - 4\times$ end-to-end wall-clock speedup. $\blacksquare$

---

### 2.7 Deep Derivation 9.4.3: DeepSeek Multi-Head Latent Attention (MLA) Mathematics

#### The Core Objective
Standard MHA caches Key and Value heads of dimension $H \cdot d_h$ per token.
DeepSeek-V2 / V3 introduced **Multi-Head Latent Attention (MLA)**, which compresses Keys and Values into a single shared low-rank latent vector:
$$\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}, \quad \text{where } d_c \ll H \cdot d_h$$

#### Mathematical Formulation:
1. **Low-Rank KV Down-Projection:**
   Given token representation $\mathbf{h}_t \in \mathbb{R}^d$:
   $$\mathbf{c}_t^{KV} = \mathbf{h}_t \mathbf{W}_{DKV} \in \mathbb{R}^{d_c} \quad (\mathbf{W}_{DKV} \in \mathbb{R}^{d \times d_c})$$
   Only $\mathbf{c}_t^{KV}$ is saved in the KV-cache!

2. **Decoupled RoPE Key:**
   To apply RoPE without breaking low-rank matrix absorption, a separate low-dimensional positional key is projected:
   $$\mathbf{k}_t^R = \operatorname{RoPE}\left( \mathbf{h}_t \mathbf{W}_{KR} \right) \in \mathbb{R}^{d_R}$$
   The cached state per token is just $[\mathbf{c}_t^{KV}, \, \mathbf{k}_t^R]$.

3. **Key and Value Up-Projections:**
   When needed, the $H$ heads are uncompressed:
   $$\mathbf{K}_{t, i}^C = \mathbf{c}_t^{KV} \mathbf{W}_{UK, i} \in \mathbb{R}^{d_h}, \quad \mathbf{V}_{t, i} = \mathbf{c}_t^{KV} \mathbf{W}_{UV, i} \in \mathbb{R}^{d_h}$$
   $$\mathbf{K}_{t, i} = \left[ \mathbf{K}_{t, i}^C; \, \mathbf{k}_t^R \right] \in \mathbb{R}^{d_h + d_R}$$

4. **The Matrix Absorption Theorem (Zero-Decompression Inference):**
   In generation, we do not need to materialize $\mathbf{K}_{t, i}^C$!
   $$\mathbf{q}_i^T \mathbf{K}_{t, i}^C = \mathbf{q}_i^T \left( \mathbf{c}_t^{KV} \mathbf{W}_{UK, i} \right) = \left( \mathbf{q}_i^T \mathbf{W}_{UK, i}^T \right) \mathbf{c}_t^{KV} = \tilde{\mathbf{q}}_i^T \mathbf{c}_t^{KV}$$
   By absorbing $\mathbf{W}_{UK, i}$ directly into the Query projection $\tilde{\mathbf{q}}_i = \mathbf{W}_{UK, i} \mathbf{q}_i$, the Query attends directly to the compressed latent vector $\mathbf{c}_t^{KV}$.
   The KV-cache footprint drops from $2 \times H \times d_h \times 2$ bytes to $(d_c + d_R) \times 2$ bytes, achieving a **$93\%$ KV-cache memory reduction** compared to MHA while retaining the full representational capacity of 128 attention heads! $\blacksquare$

---

## 3. Geometric & Algebraic Interpretation

### The Roofline Model for LLM Inference

The **Roofline Model** relates performance (TFLOPs/sec) to the **Operational Intensity** $I$ (FLOPs per byte of DRAM traffic):
$$\text{Attainable Performance} = \min\left( \text{Peak Compute Bandwidth}, \, I \times \text{Peak Memory Bandwidth} \right)$$

```
     Attainable FLOPs/s
            ▲
 Peak Compute │───────────────────────────── (Compute-Bound: Prompt Prefill)
              │                            /
              │                           /
              │                          /  <-- Memory-Bound: Token Generation
              │                         /       (GQA & KV-Cache shift operational
              │                        /         intensity to the right!)
              └───────────────────────┴──────────────────────►
                                   Operational Intensity I (FLOPs/Byte)
```

In the token-by-token decode phase, reading $W$ weights and $K$ cache bytes to perform $2W$ FLOPs yields $I \approx 1 \text{ FLOP/Byte}$. GQA and FlashAttention drastically compress memory traffic, shifting operational intensity to the right toward peak compute hardware saturation.

---

## 4. Real-World Analogy

### The Consulting Firm with Specialist Partners (MoE)
- **Dense Transformer:** A single generalist consultant answers every single question—from tax law to brain surgery to poetry. Because one brain must hold all human knowledge, they are expensive to query.
- **Sparse MoE (Mixtral):** A prestigious consulting firm with 64 world-class specialist partners.
  - A receptionist (the router) sits at the front desk.
  - A client enters asking about intellectual property law in Japan.
  - The receptionist does not convene all 64 partners; that would be an exorbitant waste of time and money.
  - Instead, the receptionist routes the client to Partner 4 (IP Law) and Partner 17 (Japanese Jurisprudence).
  - Only those 2 partners do the work. The firm possesses 64 experts of institutional knowledge, but bills the client for only 2 billable hours.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete Grouped-Query Attention (GQA) calculation with exact hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in GQA |
| :--- | :--- | :--- | :--- |
| $H_Q$ | Number of Query Heads | Scalar ($4$) | Four parallel query heads ($Q_1, Q_2, Q_3, Q_4$) |
| $H_{KV}$ | Number of KV Groups | Scalar ($2$) | Two shared key/value groups ($G_1, G_2$) |
| $d_k$ | Head Dimension | Scalar ($2$) | Vector dimension per head |
| $\mathbf{q}_1, \mathbf{q}_2$ | Query Vectors for Group 1 | $(2,)$ each | Share Key $\mathbf{k}_1$ and Value $\mathbf{v}_1$ |
| $\mathbf{q}_3, \mathbf{q}_4$ | Query Vectors for Group 2 | $(2,)$ each | Share Key $\mathbf{k}_2$ and Value $\mathbf{v}_2$ |
| $\mathbf{k}_1, \mathbf{k}_2$ | Group Keys | $(2,)$ each | Shared Key vectors for Group 1 and Group 2 |
| $\mathbf{v}_1, \mathbf{v}_2$ | Group Values | $(2,)$ each | Shared Value vectors for Group 1 and Group 2 |
| $\mathbf{o}_1, \mathbf{o}_2, \mathbf{o}_3, \mathbf{o}_4$ | GQA Head Outputs | $(2,)$ each | Final attention outputs across all 4 query heads |

---

### 5.2 Concrete Toy Numbers

#### Query Vectors ($H_Q = 4, d_k = 2$):
- **Group 1 Queries:**
  $$\mathbf{q}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{q}_2 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$
- **Group 2 Queries:**
  $$\mathbf{q}_3 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}, \quad \mathbf{q}_4 = \begin{bmatrix} 2.0 \\ 0.0 \end{bmatrix}$$

#### Shared Key and Value Vectors ($H_{KV} = 2, d_k = 2$):
- **Group 1 (assigned to $\mathbf{q}_1, \mathbf{q}_2$):**
  $$\mathbf{k}_1 = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}, \quad \mathbf{v}_1 = \begin{bmatrix} 3.0 \\ 0.0 \end{bmatrix}$$
- **Group 2 (assigned to $\mathbf{q}_3, \mathbf{q}_4$):**
  $$\mathbf{k}_2 = \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0.0 \\ 4.0 \end{bmatrix}$$

---

### 5.3 Step 1: Compute Dot Products and Scores ($\sqrt{d_k} = \sqrt{2} \approx 1.414214$)

1. **Head 1 (Query 1 vs Group 1 Key):**
   $$\mathbf{q}_1^T \mathbf{k}_1 = (1.0)(1.0) + (0.0)(2.0) = 1.0 \implies S_1 = \frac{1.0}{\sqrt{2}} \approx \mathbf{0.707107}$$

2. **Head 2 (Query 2 vs Group 1 Key):**
   $$\mathbf{q}_2^T \mathbf{k}_1 = (0.0)(1.0) + (1.0)(2.0) = 2.0 \implies S_2 = \frac{2.0}{\sqrt{2}} \approx \mathbf{1.414214}$$

3. **Head 3 (Query 3 vs Group 2 Key):**
   $$\mathbf{q}_3^T \mathbf{k}_2 = (1.0)(0.0) + (1.0)(2.0) = 2.0 \implies S_3 = \frac{2.0}{\sqrt{2}} \approx \mathbf{1.414214}$$

4. **Head 4 (Query 4 vs Group 2 Key):**
   $$\mathbf{q}_4^T \mathbf{k}_2 = (2.0)(0.0) + (0.0)(2.0) = 0.0 \implies S_4 = \frac{0.0}{\sqrt{2}} = \mathbf{0.000000}$$

---

### 5.4 Step 2: Attention Weight Normalization and Value Aggregation

For single-key attention (per step in decode), softmax normalization over a single element evaluates to $1.0$:
$$\text{softmax}([S_i]) = 1.0, \quad \forall i$$

Thus each head's output is directly proportional to its shared group value:
- **Head 1:** $\mathbf{o}_1 = 1.0 \times \mathbf{v}_1 = [3.0, 0.0]^T$
- **Head 2:** $\mathbf{o}_2 = 1.0 \times \mathbf{v}_1 = [3.0, 0.0]^T$
- **Head 3:** $\mathbf{o}_3 = 1.0 \times \mathbf{v}_2 = [0.0, 4.0]^T$
- **Head 4:** $\mathbf{o}_4 = 1.0 \times \mathbf{v}_2 = [0.0, 4.0]^T$

Concatenated Output across all 4 Query Heads:
$$\mathbf{O}_{\text{GQA}} = \begin{bmatrix} \mathbf{o}_1 \\ \mathbf{o}_2 \\ \mathbf{o}_3 \\ \mathbf{o}_4 \end{bmatrix} = \begin{bmatrix} 3.0 \\ 0.0 \\ 3.0 \\ 0.0 \\ 0.0 \\ 4.0 \\ 0.0 \\ 4.0 \end{bmatrix} \in \mathbb{R}^8$$

Notice the massive memory savings: We produced 4 distinct query representations while storing only **2 Key-Value pairs** in GPU memory, cutting cache bandwidth by exactly **$50\%$**!

---

## 6. Solved Illustrations

### Illustration 1: KV-Cache Memory Calculation (LLaMA-3 70B: MHA vs. GQA)

**Problem:**
Calculate the exact GPU memory footprint of the KV-Cache for a batch size $B = 4$, context length $T = 8,192$, under FP16 ($2$ bytes per element) for **LLaMA-3 70B** ($L = 80$ layers, $d_k = 128$):
1. Under standard Multi-Head Attention ($H_Q = 64, H_{KV} = 64$).
2. Under Grouped-Query Attention ($H_Q = 64, H_{KV} = 8$).

**Solution:**
Recall the formula:
$$\text{Memory}_{\text{KV}} = 2 \times B \times T \times L \times H_{KV} \times d_k \times 2 \text{ bytes}$$

1. **Standard MHA ($H_{KV} = 64$):**
   $$\text{Bytes} = 2 \times 4 \times 8,192 \times 80 \times 64 \times 128 \times 2$$
   $$= 8 \times 8,192 \times 80 \times 8,192 \times 2 = 85,899,345,920 \text{ bytes} \approx \mathbf{80.0 \text{ GB}}$$
   The KV-cache alone completely fills an entire $80\text{GB}$ A100 GPU before loading a single model parameter!

2. **Grouped-Query Attention ($H_{KV} = 8$):**
   $$\text{Bytes} = 2 \times 4 \times 8,192 \times 80 \times 8 \times 128 \times 2 = 10,737,418,240 \text{ bytes} \approx \mathbf{10.0 \text{ GB}}$$
   GQA cuts memory consumption from **$80\text{ GB} \to 10\text{ GB}$**, an **$8\times$ reduction**, allowing 8 times larger batch sizes and dramatically higher server throughput!

---

### Illustration 2: MoE Router Top-2 Gating Arithmetic

**Problem:**
A token representation $\mathbf{x}$ produces routing logits across $E = 4$ experts:
$$\mathbf{H}(\mathbf{x}) = [2.0, \, 5.0, \, 1.0, \, 5.0]$$
Compute the Top-2 expert indices and their normalized gating probabilities $G(\mathbf{x})$.

**Solution:**
1. **Identify Top-2 Experts:**
   The highest values are Expert 1 ($5.0$) and Expert 3 ($5.0$).
   $$\mathcal{T} = \{1, 3\}$$
2. **Compute Softmax over Active Experts Only:**
   $$\exp(H_1) = e^5 \approx 148.413, \quad \exp(H_3) = e^5 \approx 148.413$$
   $$\text{Sum} = 148.413 + 148.413 = 296.826$$
   $$G_1 = \frac{148.413}{296.826} = \mathbf{0.5}, \quad G_3 = \frac{148.413}{296.826} = \mathbf{0.5}$$
   $$G_0 = 0.0, \quad G_2 = 0.0$$
3. **Combined Output:**
   $$\mathbf{y} = 0.5 \cdot \text{Expert}_1(\mathbf{x}) + 0.5 \cdot \text{Expert}_3(\mathbf{x})$$
   Experts 0 and 2 are never evaluated, saving $50\%$ of FFN FLOPs.

---

### Illustration 3: Hand Trace of 2-Block Online Softmax and Value Accumulation (FlashAttention Primitive)

**Problem:**
A query token $\mathbf{q} = [1.0, 1.0]^T \in \mathbb{R}^2$ attends to $N = 4$ keys and values with head dimension $d_k = 2$ ($\frac{1}{\sqrt{2}} \approx 0.707107$).
The memory is partitioned into two on-chip blocks ($B_c = 2$):
- **Block 1 ($j \in \{1, 2\}$):**
  $$\mathbf{k}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \, \mathbf{k}_2 = \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix}, \quad \mathbf{v}_1 = \begin{bmatrix} 2.0 \\ 0.0 \end{bmatrix}, \, \mathbf{v}_2 = \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix}$$
- **Block 2 ($j \in \{3, 4\}$):**
  $$\mathbf{k}_3 = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}, \, \mathbf{k}_4 = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}, \, \mathbf{v}_4 = \begin{bmatrix} 3.0 \\ 0.0 \end{bmatrix}$$
1. Calculate local Block 1 statistics: max $m^{(1)}$, sum $l^{(1)}$, and output vector $\mathbf{o}^{(1)}$.
2. Calculate local Block 2 statistics: max $m^{(2)}$, sum $l^{(2)}$, and output vector $\mathbf{o}^{(2)}$.
3. Execute the FlashAttention online recurrence to merge Block 1 and Block 2 into running global state $(m^{\text{new}}, l^{\text{new}}, \mathbf{o}^{\text{new}})$.
4. Verify that $\mathbf{o}^{\text{new}}$ exactly matches standard full-sequence attention without approximation.

**Solution:**

#### Step 1: Block 1 Execution (Local SRAM)

1. **Dot-Product Scores:**
   $$S_1 = \frac{\mathbf{q}^T \mathbf{k}_1}{\sqrt{2}} = \frac{(1)(1) + (1)(0)}{\sqrt{2}} = \frac{1.0}{\sqrt{2}} \approx \mathbf{0.707107}$$
   $$S_2 = \frac{\mathbf{q}^T \mathbf{k}_2}{\sqrt{2}} = \frac{(1)(0) + (1)(2)}{\sqrt{2}} = \frac{2.0}{\sqrt{2}} \approx \mathbf{1.414214}$$

2. **Block 1 Statistics:**
   $$m^{(1)} = \max(0.707107, 1.414214) = \mathbf{1.414214}$$
   $$P_1^{(1)} = \exp(0.707107 - 1.414214) = \exp(-0.707107) \approx 0.493069$$
   $$P_2^{(1)} = \exp(1.414214 - 1.414214) = \exp(0.0) = 1.000000$$
   $$l^{(1)} = 0.493069 + 1.000000 = \mathbf{1.493069}$$

3. **Block 1 Partial Output:**
   $$\mathbf{o}^{(1)} = \frac{0.493069 \begin{bmatrix} 2.0 \\ 0.0 \end{bmatrix} + 1.000000 \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix}}{1.493069} = \frac{\begin{bmatrix} 0.986138 \\ 2.000000 \end{bmatrix}}{1.493069} = \begin{bmatrix} \mathbf{0.660477} \\ \mathbf{1.339523} \end{bmatrix}$$

---

#### Step 2: Block 2 Execution (Local SRAM)

1. **Dot-Product Scores:**
   $$S_3 = \frac{\mathbf{q}^T \mathbf{k}_3}{\sqrt{2}} = \frac{(1)(2) + (1)(1)}{\sqrt{2}} = \frac{3.0}{\sqrt{2}} \approx \mathbf{2.121320}$$
   $$S_4 = \frac{\mathbf{q}^T \mathbf{k}_4}{\sqrt{2}} = \frac{(1)(1) + (1)(2)}{\sqrt{2}} = \frac{3.0}{\sqrt{2}} \approx \mathbf{2.121320}$$

2. **Block 2 Statistics:**
   $$m^{(2)} = \max(2.121320, 2.121320) = \mathbf{2.121320}$$
   $$P_3^{(2)} = \exp(2.121320 - 2.121320) = 1.000000$$
   $$P_4^{(2)} = \exp(2.121320 - 2.121320) = 1.000000$$
   $$l^{(2)} = 1.000000 + 1.000000 = \mathbf{2.000000}$$

3. **Block 2 Partial Output:**
   $$\mathbf{o}^{(2)} = \frac{1.0 \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} + 1.0 \begin{bmatrix} 3.0 \\ 0.0 \end{bmatrix}}{2.000000} = \begin{bmatrix} \mathbf{2.000000} \\ \mathbf{0.500000} \end{bmatrix}$$

---

#### Step 3: FlashAttention Online Merge Recurrence

1. **Updated Global Maximum:**
   $$m^{\text{new}} = \max(m^{(1)}, m^{(2)}) = \max(1.414214, 2.121320) = \mathbf{2.121320}$$

2. **Rescaling Factors:**
   $$\alpha_1 = \exp(m^{(1)} - m^{\text{new}}) = \exp(1.414214 - 2.121320) = \exp(-0.707106) \approx \mathbf{0.493069}$$
   $$\alpha_2 = \exp(m^{(2)} - m^{\text{new}}) = \exp(0.0) = \mathbf{1.000000}$$

3. **Accumulated Global Denominator:**
   $$l^{\text{new}} = \alpha_1 l^{(1)} + \alpha_2 l^{(2)} = (0.493069)(1.493069) + (1.0)(2.0) = 0.736187 + 2.0 = \mathbf{2.736187}$$

4. **Rescaled and Merged Output Vector:**
   $$\mathbf{o}^{\text{new}} = \frac{\alpha_1 l^{(1)}}{l^{\text{new}}} \mathbf{o}^{(1)} + \frac{\alpha_2 l^{(2)}}{l^{\text{new}}} \mathbf{o}^{(2)} = \frac{0.736187}{2.736187} \begin{bmatrix} 0.660477 \\ 1.339523 \end{bmatrix} + \frac{2.000000}{2.736187} \begin{bmatrix} 2.000000 \\ 0.500000 \end{bmatrix}$$
   $$= (0.269056) \begin{bmatrix} 0.660477 \\ 1.339523 \end{bmatrix} + (0.730944) \begin{bmatrix} 2.000000 \\ 0.500000 \end{bmatrix}$$
   $$= \begin{bmatrix} 0.177705 \\ 0.360407 \end{bmatrix} + \begin{bmatrix} 1.461888 \\ 0.365472 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.639593 \\ 0.725879 \end{bmatrix}}$$

---

#### Step 4: Verification Against Standard Full Softmax

Evaluating all 4 tokens simultaneously with global max $m = 2.121320$:
$$\exp(\mathbf{S} - m) = [\exp(-1.414213), \, \exp(-0.707106), \, \exp(0), \, \exp(0)] = [0.243117, \, 0.493069, \, 1.0, \, 1.0]$$
$$\text{Sum} = 0.243117 + 0.493069 + 1.0 + 1.0 = \mathbf{2.736186}$$
Attention weights:
$$A_1 = \frac{0.243117}{2.736186} \approx 0.088852, \quad A_2 = \frac{0.493069}{2.736186} \approx 0.180203$$
$$A_3 = \frac{1.0}{2.736186} \approx 0.365472, \quad A_4 = \frac{1.0}{2.736186} \approx 0.365472$$
Standard attention output:
$$\mathbf{o} = 0.088852 \begin{bmatrix} 2 \\ 0 \end{bmatrix} + 0.180203 \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.365472 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + 0.365472 \begin{bmatrix} 3 \\ 0 \end{bmatrix}$$
$$o_1 = 0.177704 + 0.0 + 0.365472 + 1.096416 = \mathbf{1.639592}$$
$$o_2 = 0.0 + 0.360406 + 0.365472 + 0.0 = \mathbf{0.725878}$$
The tiled online result matches the global full-matrix result to machine precision ($< 10^{-6}$), proving that FlashAttention introduces **zero approximation error**.

---

### Illustration 4: Sparse MoE Backward Pass and Router Gradient Step-by-Step

**Problem:**
A token representation $\mathbf{x} = [1.0, 2.0]^T \in \mathbb{R}^2$ is routed to $E = 3$ experts using router weight matrix:
$$\mathbf{W}_g = \begin{bmatrix} 1.0 & 0.0 & -1.0 \\ 0.0 & 1.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$
1. Evaluate raw routing logits $\mathbf{h} = \mathbf{x}^T \mathbf{W}_g$ and identify the Top-2 active experts.
2. Compute normalized gating weights $G_i$ over the Top-2 set.
3. The selected experts produce outputs $\mathbf{e}_0 = [1.0, 0.0]^T, \mathbf{e}_1 = [0.0, 2.0]^T$. The layer output is $\mathbf{y} = G_0 \mathbf{e}_0 + G_1 \mathbf{e}_1$.
4. Given upstream loss sensitivity $\boldsymbol{\delta}_y = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = [1.0, 1.0]^T$, backpropagate through the gating network to compute parameter gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_g} \in \mathbb{R}^{2 \times 3}$.

**Solution:**

#### Step 1: Forward Routing Pass

1. **Routing Logits $\mathbf{h} = \mathbf{x}^T \mathbf{W}_g$:**
   $$\mathbf{h}^T = [1.0, 2.0] \begin{bmatrix} 1.0 & 0.0 & -1.0 \\ 0.0 & 1.0 & 1.0 \end{bmatrix}$$
   $$h_0 = (1.0)(1.0) + (2.0)(0.0) = \mathbf{1.0}$$
   $$h_1 = (1.0)(0.0) + (2.0)(1.0) = \mathbf{2.0}$$
   $$h_2 = (1.0)(-1.0) + (2.0)(1.0) = -1.0 + 2.0 = \mathbf{1.0}$$
   $$\mathbf{h} = [1.0, \, 2.0, \, 1.0]$$

2. **Top-2 Selection:**
   The highest logit is Expert 1 ($2.0$). Between Expert 0 ($1.0$) and Expert 2 ($1.0$), the tie-breaker selects Expert 0.
   Active set $\mathcal{T} = \{0, 1\}$. Expert 2 is masked to $0$.

3. **Gating Softmax Probabilities:**
   $$\exp(h_0) = \exp(1.0) \approx 2.718282, \quad \exp(h_1) = \exp(2.0) \approx 7.389056$$
   $$\text{Sum} = 2.718282 + 7.389056 = 10.107338$$
   $$G_0 = \frac{2.718282}{10.107338} = \mathbf{0.268941}$$
   $$G_1 = \frac{7.389056}{10.107338} = \mathbf{0.731059}$$
   $$G_2 = \mathbf{0.000000}$$

4. **Layer Output:**
   $$\mathbf{y} = 0.268941 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} + 0.731059 \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.268941} \\ \mathbf{1.462118} \end{bmatrix}$$

---

#### Step 2: Backward Pass Through Gating Mechanism

1. **Gradient w.r.t. Gating Weights $G_i$:**
   $$\frac{\partial \mathcal{L}}{\partial G_0} = \boldsymbol{\delta}_y^T \mathbf{e}_0 = [1.0, 1.0] \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \mathbf{1.000000}$$
   $$\frac{\partial \mathcal{L}}{\partial G_1} = \boldsymbol{\delta}_y^T \mathbf{e}_1 = [1.0, 1.0] \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = \mathbf{2.000000}$$

2. **Backpropagation Through Softmax to Active Logits $h_i$:**
   Weighted mean:
   $$\bar{g} = G_0 \frac{\partial \mathcal{L}}{\partial G_0} + G_1 \frac{\partial \mathcal{L}}{\partial G_1} = (0.268941)(1.0) + (0.731059)(2.0) = 0.268941 + 1.462118 = \mathbf{1.731059}$$
   Applying $\frac{\partial \mathcal{L}}{\partial h_i} = G_i \left( \frac{\partial \mathcal{L}}{\partial G_i} - \bar{g} \right)$:
   $$\frac{\partial \mathcal{L}}{\partial h_0} = 0.268941 \times (1.000000 - 1.731059) = 0.268941 \times (-0.731059) = \mathbf{-0.196612}$$
   $$\frac{\partial \mathcal{L}}{\partial h_1} = 0.731059 \times (2.000000 - 1.731059) = 0.731059 \times (+0.268941) = \mathbf{+0.196612}$$
   For the inactive expert:
   $$\frac{\partial \mathcal{L}}{\partial h_2} = \mathbf{0.000000}$$
   *(Check: $-0.196612 + 0.196612 + 0 = 0$)*.

3. **Gradient w.r.t. Router Weight Matrix $\mathbf{W}_g$:**
   Since $\mathbf{h} = \mathbf{W}_g^T \mathbf{x}$:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_g} = \mathbf{x} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \right)^T = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} \begin{bmatrix} -0.196612 & 0.196612 & 0.000000 \end{bmatrix}$$
   $$= \begin{bmatrix}
   \mathbf{-0.196612} & \mathbf{+0.196612} & \mathbf{0.000000} \\
   \mathbf{-0.393224} & \mathbf{+0.393224} & \mathbf{0.000000}
   \end{bmatrix}$$

Notice that unselected Expert 2 receives exactly zero gradient, demonstrating how MoE routing preserves sparsity throughout the entire backward pass.

---

### Illustration 5: DeepSeek Multi-Head Latent Attention (MLA) vs. GQA Memory Compression

**Problem:**
An LLM architecture has $L = 60$ layers, model dimension $d_{\text{model}} = 4,096$, $H = 32$ attention heads, and head dimension $d_h = 128$.
Compare three attention variants under FP16 ($2$ bytes/scalar):
1. **Multi-Head Attention (MHA):** $H_{KV} = 32$.
2. **Grouped-Query Attention (GQA):** $H_{KV} = 4$ ($8\times$ grouped).
3. **DeepSeek Multi-Head Latent Attention (MLA):** Compresses Keys and Values to latent dimension $d_c = 512$, plus decoupled RoPE key $d_R = 64$.

Calculate:
1. The KV-cache bytes stored per token per layer, and across the full 60-layer model.
2. Total KV-cache RAM consumed by a concurrent batch of $B = 8$ users at context length $T = 32,768$.

**Solution:**

#### Step 1: KV-Cache Bytes per Token per Layer

1. **Multi-Head Attention (MHA):**
   $$\text{Bytes}_{\text{layer}} = 2 \times H_{KV} \times d_h \times 2 = 2 \times 32 \times 128 \times 2 = \mathbf{16,384 \text{ bytes (16 KB)}}$$
   Full model ($60$ layers):
   $$\text{Bytes}_{\text{total}} = 60 \times 16,384 = \mathbf{983,040 \text{ bytes (960 KB per token)}}$$

2. **Grouped-Query Attention (GQA, $H_{KV} = 4$):**
   $$\text{Bytes}_{\text{layer}} = 2 \times 4 \times 128 \times 2 = \mathbf{2,048 \text{ bytes (2 KB)}}$$
   Full model ($60$ layers):
   $$\text{Bytes}_{\text{total}} = 60 \times 2,048 = \mathbf{122,880 \text{ bytes (120 KB per token)}}$$
   *(Savings: $8\times$ smaller than MHA)*.

3. **DeepSeek MLA:**
   Stores only the low-rank latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{512}$ and the decoupled RoPE key $\mathbf{k}_t^R \in \mathbb{R}^{64}$:
   $$\text{Scalars per Layer} = d_c + d_R = 512 + 64 = 576 \text{ elements}$$
   $$\text{Bytes}_{\text{layer}} = 576 \times 2 = \mathbf{1,152 \text{ bytes (1.125 KB)}}$$
   Full model ($60$ layers):
   $$\text{Bytes}_{\text{total}} = 60 \times 1,152 = \mathbf{69,120 \text{ bytes (67.5 KB per token)}}$$
   *(Savings: $\mathbf{14.22\times \text{ smaller than MHA}}$, and $\mathbf{1.78\times \text{ smaller than GQA}}$!)*.

---

#### Step 2: Total Memory for Batch $B = 8$ at $T = 32,768$

Total tokens in cache: $N_{\text{tokens}} = B \times T = 8 \times 32,768 = 262,144 \text{ tokens}$.

| Architecture | Bytes / Token (60 Layers) | Total Cache Footprint ($B=8, T=32k$) | Memory Reduction Factor |
| :--- | :--- | :--- | :--- |
| **Standard MHA** | $960 \text{ KB}$ | $262,144 \times 983,040 \text{ B} \approx \mathbf{257.70 \text{ GB}}$ | $1.0\times$ (Requires 4x 80GB GPUs just for cache!) |
| **GQA ($H_{KV}=4$)** | $120 \text{ KB}$ | $262,144 \times 122,880 \text{ B} \approx \mathbf{32.21 \text{ GB}}$ | $8.0\times$ (Fits in a single 48GB/80GB GPU) |
| **DeepSeek MLA** | **$67.5 \text{ KB}$** | $262,144 \times 69,120 \text{ B} \approx \mathbf{18.12 \text{ GB}}$ | **$14.22\times$** (Leaves over 60 GB free for weights & batching!) |

**Conclusion:**
DeepSeek's MLA drastically outperforms GQA in memory efficiency by compressing across head channels via low-rank projection, while using the matrix absorption identity ($\tilde{\mathbf{q}} = \mathbf{W}_{UK} \mathbf{q}$) to avoid ever decompressing Keys during inference.

---

## 7. Deep Learning Connection & Application

### State-of-the-Art Architecture Implementations

| Frontier Architecture | Attention Mechanism | FFN / Expert Topology | Context Window | Key Innovation |
| :--- | :--- | :--- | :--- | :--- |
| **LLaMA-3 (8B / 70B)** | GQA ($8$ KV heads) | Dense SwiGLU | $8\text{k} - 128\text{k}$ | Fast inference, open-weight standard |
| **Mixtral 8x7B** | GQA ($8$ KV heads) | Sparse MoE ($8$ experts, top-2) | $32\text{k}$ | $47\text{B}$ capacity at $13\text{B}$ active cost |
| **DeepSeek-V2 / V3** | **MLA (Multi-Head Latent Attention)** | DeepSeekMoE ($64$ experts, top-8) | $128\text{k}$ | Low-rank joint KV compression into $512$-dim latent vector |

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `GroupedQueryAttention`: Complete, vectorized PyTorch implementation supporting MHA, GQA, and MQA.
2. `OnlineSoftmaxAttention`: Pure Python/NumPy implementation of the FlashAttention tiling and online softmax algorithm.
3. `SparseMoELayer`: Top-$k$ gating routing network with auxiliary load-balancing loss.
4. Exact numerical verification of the Part 5 Visual Grid GQA hand calculations.
5. Online Softmax parity verification matching standard full-matrix attention to $< 10^{-12}$.

See implementation in:
[`09_transformers_and_llms/code/04_modern_efficiency.py`](./code/04_modern_efficiency.py)
