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
