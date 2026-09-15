# Chapter 05: Memory & Hardware Efficiency: FlashAttention-1/2/3 & Online Softmax

---

## 1. Intuition & 101 Motivation

In standard self-attention, given query $Q \in \mathbb{R}^{L \times d}$, key $K \in \mathbb{R}^{L \times d}$, and value $V \in \mathbb{R}^{L \times d}$, the attention operation is expressed as:
$$\mathbf{S} = Q K^T \in \mathbb{R}^{L \times L}, \quad \mathbf{P} = \operatorname{softmax}\left( \frac{\mathbf{S}}{\sqrt{d}} \right) \in \mathbb{R}^{L \times L}, \quad \mathbf{O} = \mathbf{P} V \in \mathbb{R}^{L \times d}$$

For a sequence length of $L = 32,768$ (common in modern LLMs), the intermediate attention matrix $\mathbf{S}$ contains:
$$32,768 \times 32,768 \approx 1.07 \times 10^9 \text{ elements} \approx \mathbf{4.3 \text{ GB of VRAM per attention head!}}$$
For a 32-head, 32-layer model, storing these intermediate matrices would require over **4,400 GB of VRAM**, which is physically impossible on even the largest multi-GPU clusters.

### The GPU Memory Hierarchy: SRAM vs. HBM
Modern GPUs (NVIDIA A100, H100) are not single monolithic processors; they have a multi-tiered memory architecture:
1. **High-Bandwidth Memory (HBM / GPU RAM):** Large ($40–80\text{ GB}$), but relatively slow ($1.5–3.3\text{ TB/s}$).
2. **On-Chip Static RAM (SRAM / Shared Memory):** Ultra-fast ($19\text{ TB/s}$, $\approx 10\times$ faster than HBM), but tiny ($\approx 100–228\text{ KB}$ per Streaming Multiprocessor).

In standard attention, the GPU is severely **memory-bandwidth bound**: it repeatedly writes the $L \times L$ matrix $\mathbf{S}$ to slow HBM, reads it back to compute softmax $\mathbf{P}$, writes $\mathbf{P}$ to HBM, and reads it again to multiply by $V$.

```
Standard Attention (HBM Memory Bottleneck):
   GPU SRAM (Fast) -----> Write S (L x L) to HBM (Slow) -----> Read S -----> Write P to HBM -----> Read P -----> Compute O

FlashAttention (Zero HBM Intermediate Writes):
   Load Block Q_i, K_j to SRAM -----> Compute Local Attention in SRAM -----> Stream into Output O in SRAM
```

Enter **FlashAttention** (Tri Dao et al., NeurIPS 2022, 2023, 2024):
- **Never materialize the $L \times L$ attention matrix in HBM!**
- Compute exact attention by tiling $Q, K, V$ into small blocks that fit entirely within ultra-fast **on-chip SRAM**.
- The mathematical foundation enabling this is **Online Softmax**: an algorithm that dynamically updates running softmax statistics as key-value blocks arrive sequentially, producing the exact mathematical result with zero approximation!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Standard 3-Pass Softmax vs. Online Softmax

To compute $\operatorname{softmax}(x)$ for a vector $x \in \mathbb{R}^N$ safely without numerical overflow:
1. **Pass 1 (Max):** $m = \max_{j=1}^N x_j$
2. **Pass 2 (Sum of Exponentials):** $d = \sum_{j=1}^N e^{x_j - m}$
3. **Pass 3 (Normalization):** $p_i = \frac{e^{x_i - m}}{d}$

This standard formulation requires holding the entire vector $x$ in memory across multiple passes.

---

### 2.2 Theorem 12.3 (Online Softmax Mechanics)

Let vector $x$ be partitioned into two arbitrary sub-vectors:
$$x = \big[ x^{(1)}, \; x^{(2)} \big], \quad \text{where } x^{(1)} \in \mathbb{R}^{N_1}, \; x^{(2)} \in \mathbb{R}^{N_2}$$

Let the local statistics for each chunk be:
$$m_1 = \max(x^{(1)}), \quad d_1 = \sum_{j=1}^{N_1} e^{x_j^{(1)} - m_1}$$
$$m_2 = \max(x^{(2)}), \quad d_2 = \sum_{j=1}^{N_2} e^{x_j^{(2)} - m_2}$$

#### 1. Global Maximum Update:
The combined maximum across both chunks is simply:
$$m_{\text{new}} = \max(m_1, m_2)$$

#### 2. Denominator Rescaling & Update:
The combined denominator $d_{\text{new}}$ is computed **without ever re-reading $x^{(1)}$**:

$$d_{\text{new}} = d_1 \cdot e^{m_1 - m_{\text{new}}} + d_2 \cdot e^{m_2 - m_{\text{new}}}$$

#### 3. Online Attention Output Accumulation:
Let $O^{(1)} \in \mathbb{R}^d$ be the partial attention output vector computed on chunk 1 using local normalizer $d_1$:
$$O^{(1)} = \sum_{j=1}^{N_1} \frac{e^{x_j^{(1)} - m_1}}{d_1} v_j^{(1)}$$
Let $O^{(2)} \in \mathbb{R}^d$ be the partial output on chunk 2:
$$O^{(2)} = \sum_{j=1}^{N_2} \frac{e^{x_j^{(2)} - m_2}}{d_2} v_j^{(2)}$$

The globally correct combined output $O_{\text{new}}$ is obtained by **rescaling and convex blending**:

$$O_{\text{new}} = \left( \frac{d_1 \cdot e^{m_1 - m_{\text{new}}}}{d_{\text{new}}} \right) O^{(1)} + \left( \frac{d_2 \cdot e^{m_2 - m_{\text{new}}}}{d_{\text{new}}} \right) O^{(2)}$$

#### Proof of Exact Equivalence:
Expanding the right-hand side:
$$O_{\text{new}} = \frac{1}{d_{\text{new}}} \left( e^{m_1 - m_{\text{new}}} \sum_{j=1}^{N_1} e^{x_j^{(1)} - m_1} v_j^{(1)} + e^{m_2 - m_{\text{new}}} \sum_{j=1}^{N_2} e^{x_j^{(2)} - m_2} v_j^{(2)} \right)$$
$$= \frac{1}{d_{\text{new}}} \left( \sum_{j=1}^{N_1} e^{x_j^{(1)} - m_{\text{new}}} v_j^{(1)} + \sum_{j=1}^{N_2} e^{x_j^{(2)} - m_{\text{new}}} v_j^{(2)} \right)$$
$$= \sum_{j=1}^{N_1 + N_2} \frac{e^{x_j - m_{\text{new}}}}{d_{\text{new}}} v_j \equiv \operatorname{softmax}(x) V \quad \blacksquare$$

---

### 2.3 The FlashAttention Tiling Algorithm

Let on-chip SRAM capacity per Streaming Multiprocessor be $M$ bytes.
1. Choose block sizes:
   $$B_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B_r = \min\left( \left\lceil \frac{M}{4d} \right\rceil, d \right)$$
2. Partition $Q$ into $T_r = \lceil L / B_r \rceil$ blocks: $Q_1, Q_2, \dots, Q_{T_r} \in \mathbb{R}^{B_r \times d}$.
3. Partition $K, V$ into $T_c = \lceil L / B_c \rceil$ blocks: $K_1, \dots, K_{T_c}$ and $V_1, \dots, V_{T_c}$.
4. Initialize running statistics in HBM: $O = \mathbf{0} \in \mathbb{R}^{L \times d}$, $m = -\boldsymbol{\infty} \in \mathbb{R}^L$, $d = \mathbf{0} \in \mathbb{R}^L$.
5. **Outer Loop (over KV blocks $j = 1, \dots, T_c$):**
   - Load $K_j, V_j$ from slow HBM to fast SRAM.
   - **Inner Loop (over Query blocks $i = 1, \dots, T_r$):**
     - Load $Q_i, O_i, m_i, d_i$ to SRAM.
     - Compute local block dot products: $S_{ij} = \frac{Q_i K_j^T}{\sqrt{d}} \in \mathbb{R}^{B_r \times B_c}$.
     - Update local statistics via Online Softmax:
       $$\tilde{m}_{ij} = \operatorname{rowmax}(S_{ij}), \quad m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij})$$
       $$P_{ij} = \exp(S_{ij} - m_i^{\text{new}})$$
       $$d_i^{\text{new}} = d_i \cdot e^{m_i - m_i^{\text{new}}} + \operatorname{rowsum}(P_{ij})$$
       $$O_i \leftarrow \operatorname{diag}\left( \frac{d_i e^{m_i - m_i^{\text{new}}}}{d_i^{\text{new}}} \right) O_i + \operatorname{diag}\left( \frac{1}{d_i^{\text{new}}} \right) P_{ij} V_j$$
       $$m_i \leftarrow m_i^{\text{new}}, \quad d_i \leftarrow d_i^{\text{new}}$$
     - Write updated $O_i$ back to HBM.

#### Hardware Memory Complexity:
- **Standard Attention HBM accesses:** $O(L^2 + L d)$ (Reads and writes full $L \times L$ attention matrix).
- **FlashAttention HBM accesses:** $O\left( \frac{L^2 d^2}{M} \right)$ (Reduces HBM memory traffic by **$5\times$ to $20\times$**).
- **VRAM Memory Usage:** Drops from $O(L^2)$ to $O(L)$!

---

## 3. Geometric & Physical Interpretation

### 3.1 Streaming Center of Mass (Convex Combination)
The attention output $O$ is a convex combination of value vectors: $O = \sum p_i v_i$.
Think of $O$ as the **center of mass** of physical particles $v_i$ with masses $w_i = e^{s_i - m}$.
```
Chunk 1 Center of Mass (Mass d_1):               Chunk 2 Center of Mass (Mass d_2):
             O^(1)                                            O^(2)
               *                                                *
                \                                              /
                 \                                            /
                  \                                          /
                   \                                        /
                    \                                      /
                     +--------------*---------------------+
                                  O_new
                       (New Combined Center of Mass)
```
When chunk 2 arrives with a higher maximum $m_2 > m_1$, the old mass $d_1$ is compressed by factor $e^{m_1 - m_2}$. The updated center of mass $O_{\text{new}}$ is simply the re-weighted centroid between $O^{(1)}$ and $O^{(2)}$!

---

## 4. Real-World Analogy: Running Average in Your Head

Imagine an accountant calculating the average transaction value of 10,000 receipts:
- **Standard Attention:** Photocopies all 10,000 receipts, lays them out across an entire football field (HBM memory explosion), sums them up, and divides by 10,000.
- **FlashAttention (Online Softmax):** Keeps two numbers on an index card in their pocket:
  1. The running total so far ($\sum x$)
  2. The number of receipts processed ($N$)
- For every new receipt, they update the card: $\text{Total}_{\text{new}} = \text{Total}_{\text{old}} + x_{\text{new}}$, $N_{\text{new}} = N_{\text{old}} + 1$.
- Zero football fields required!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace **Online Softmax across 2 distinct blocks** by hand on a concrete numerical sequence and prove that it matches the global 4-token softmax identically.

---

### 5.1 System & Parameter Setup
- Query vector: $q = [1.0, 1.0]^T$ ($d = 2 \implies \text{no scaling for clarity}$)
- 4 Key-Value pairs split into two chunks of size 2:

#### Chunk 1 (Tokens 1 and 2):
- $k_1 = [1.0, 0.0]^T, \quad v_1 = [2.0, 0.0]^T$
- $k_2 = [0.0, 1.0]^T, \quad v_2 = [0.0, 4.0]^T$

#### Chunk 2 (Tokens 3 and 4):
- $k_3 = [2.0, 0.0]^T, \quad v_3 = [1.0, 1.0]^T$
- $k_4 = [1.0, 2.0]^T, \quad v_4 = [3.0, -1.0]^T$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $s_i$ | `dot_scores[i]` | Raw attention logit: $q \cdot k_i$ |
| $m_1, m_2$ | `max_chunk1, max_chunk2` | Local maximum of scores in chunk 1 and chunk 2 |
| $d_1, d_2$ | `sum_exp1, sum_exp2` | Local sum of exponentials for chunk 1 and chunk 2 |
| $O^{(1)}, O^{(2)}$ | `out_chunk1, out_chunk2` | Local normalized attention output vectors |
| $m_{\text{new}}$ | `global_max` | Updated global maximum: $\max(m_1, m_2)$ |
| $d_{\text{new}}$ | `global_denom` | Rescaled global denominator: $d_1 e^{m_1 - m_{\text{new}}} + d_2 e^{m_2 - m_{\text{new}}}$ |
| $O_{\text{new}}$ | `final_out` | Rescaled and combined global attention output |

---

### 5.3 Step-by-Step Hand Calculations: Chunk 1

#### Dot products for Chunk 1:
$$s_1 = q \cdot k_1 = 1.0 \times 1.0 + 1.0 \times 0.0 = \mathbf{1.000000}$$
$$s_2 = q \cdot k_2 = 1.0 \times 0.0 + 1.0 \times 1.0 = \mathbf{1.000000}$$

#### Local Statistics:
$$m_1 = \max(s_1, s_2) = \max(1.0, 1.0) = \mathbf{1.000000}$$
$$d_1 = e^{s_1 - m_1} + e^{s_2 - m_1} = e^{1.0 - 1.0} + e^{1.0 - 1.0} = 1.0 + 1.0 = \mathbf{2.000000}$$

#### Local Output Vector $O^{(1)}$:
$$p_1^{(1)} = \frac{e^0}{d_1} = \frac{1.0}{2.0} = 0.500000, \quad p_2^{(1)} = \frac{e^0}{d_1} = 0.500000$$
$$O^{(1)} = 0.5 \cdot v_1 + 0.5 \cdot v_2 = 0.5 \begin{bmatrix} 2.0 \\ 0.0 \end{bmatrix} + 0.5 \begin{bmatrix} 0.0 \\ 4.0 \end{bmatrix} = \begin{bmatrix} \mathbf{1.000000} \\ \mathbf{2.000000} \end{bmatrix}$$

---

### 5.4 Step-by-Step Hand Calculations: Chunk 2

#### Dot products for Chunk 2:
$$s_3 = q \cdot k_3 = 1.0 \times 2.0 + 1.0 \times 0.0 = \mathbf{2.000000}$$
$$s_4 = q \cdot k_4 = 1.0 \times 1.0 + 1.0 \times 2.0 = 1.0 + 2.0 = \mathbf{3.000000}$$

#### Local Statistics:
$$m_2 = \max(s_3, s_4) = \max(2.0, 3.0) = \mathbf{3.000000}$$
$$d_2 = e^{s_3 - m_2} + e^{s_4 - m_2} = e^{2.0 - 3.0} + e^{3.0 - 3.0} = e^{-1} + 1.0 \approx 0.367879 + 1.0 = \mathbf{1.367879}$$

#### Local Output Vector $O^{(2)}$:
$$p_3^{(2)} = \frac{0.367879}{1.367879} \approx 0.268941, \quad p_4^{(2)} = \frac{1.0}{1.367879} \approx 0.731059$$
$$O^{(2)} = 0.268941 \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} + 0.731059 \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix} = \begin{bmatrix} 0.268941 + 2.193177 \\ 0.268941 - 0.731059 \end{bmatrix} = \begin{bmatrix} \mathbf{2.462118} \\ \mathbf{-0.462118} \end{bmatrix}$$

---

### 5.5 Step-by-Step Hand Calculations: Online Merge Step

Now merge Chunk 1 into Chunk 2 without re-reading Tokens 1 and 2!

1. **Update Global Maximum:**
   $$m_{\text{new}} = \max(m_1, m_2) = \max(1.0, 3.0) = \mathbf{3.000000}$$

2. **Compute Rescaling Factors:**
   - For Chunk 1: $e^{m_1 - m_{\text{new}}} = e^{1.0 - 3.0} = e^{-2.0} \approx \mathbf{0.135335}$
   - For Chunk 2: $e^{m_2 - m_{\text{new}}} = e^{3.0 - 3.0} = e^0 = \mathbf{1.000000}$

3. **Update Global Denominator $d_{\text{new}}$:**
   $$d_{\text{new}} = d_1 \cdot e^{-2.0} + d_2 \cdot e^0$$
   $$= (2.000000 \times 0.135335) + (1.367879 \times 1.000000) = 0.270670 + 1.367879 = \mathbf{1.638549}$$

4. **Compute Combined Output Vector $O_{\text{new}}$:**
   $$O_{\text{new}} = \left( \frac{d_1 e^{-2.0}}{d_{\text{new}}} \right) O^{(1)} + \left( \frac{d_2 e^0}{d_{\text{new}}} \right) O^{(2)}$$
   Weight for Chunk 1:
   $$w_1 = \frac{0.270670}{1.638549} \approx \mathbf{0.165189}$$
   Weight for Chunk 2:
   $$w_2 = \frac{1.367879}{1.638549} \approx \mathbf{0.834811}$$

   $$O_{\text{new}} = 0.165189 \begin{bmatrix} 1.000000 \\ 2.000000 \end{bmatrix} + 0.834811 \begin{bmatrix} 2.462118 \\ -0.462118 \end{bmatrix}$$
   $$= \begin{bmatrix} 0.165189 \\ 0.330378 \end{bmatrix} + \begin{bmatrix} 2.055398 \\ -0.385781 \end{bmatrix} = \begin{bmatrix} \mathbf{2.220587} \\ \mathbf{-0.055403} \end{bmatrix}$$

---

### 5.6 Global Exact Verification (Standard Multi-Pass Softmax)

Let us verify by computing standard global attention over all 4 tokens at once:
- Scores: $s = [1.0, 1.0, 2.0, 3.0]$
- Global max: $m = 3.0$
- Exponentials:
  $$e^{1-3} = e^{-2} \approx 0.135335$$
  $$e^{1-3} = e^{-2} \approx 0.135335$$
  $$e^{2-3} = e^{-1} \approx 0.367879$$
  $$e^{3-3} = e^0 = 1.000000$$
- Total Sum:
  $$d_{\text{global}} = 0.135335 + 0.135335 + 0.367879 + 1.000000 = \mathbf{1.638549} \equiv d_{\text{new}} \quad \checkmark$$
- Normalized Softmax Weights:
  $$p = [0.082594, \; 0.082594, \; 0.224515, \; 0.610296]$$
- Output:
  $$O_{\text{global}} = p_1 v_1 + p_2 v_2 + p_3 v_3 + p_4 v_4$$
  $$= 0.082594 \begin{bmatrix} 2.0 \\ 0.0 \end{bmatrix} + 0.082594 \begin{bmatrix} 0.0 \\ 4.0 \end{bmatrix} + 0.224515 \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} + 0.610296 \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix}$$
  $$= \begin{bmatrix} 0.165188 + 0.0 + 0.224515 + 1.830888 \\ 0.0 + 0.330376 + 0.224515 - 0.610296 \end{bmatrix} = \begin{bmatrix} \mathbf{2.220591} \\ \mathbf{-0.055405} \end{bmatrix}$$

$$\mathbf{O_{\text{new}} \equiv O_{\text{global}} = [2.22059, -0.05540]} \quad \text{(Exact Match to 5 decimal places!)}$$

---

### 5.7 Summary Visual Grid: Online Softmax Block Merge Ledger

| Stage | Processed Chunks | Max $m$ | Denominator $d$ | Rescaling Factor | Weighted Output Vector $O$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Chunk 1** | Tokens 1 & 2 | $m_1 = 1.0$ | $d_1 = 2.0000$ | $e^0 = 1.0$ | $O^{(1)} = [1.0000, 2.0000]$ |
| **Chunk 2** | Tokens 3 & 4 | $m_2 = 3.0$ | $d_2 = 1.3679$ | $e^0 = 1.0$ | $O^{(2)} = [2.4621, -0.4621]$ |
| **Online Merge**| Tokens 1 to 4 | $\mathbf{m_{\text{new}} = 3.0}$ | $\mathbf{d_{\text{new}} = 1.6385}$ | $e^{1-3} = 0.1353$ | $\mathbf{O_{\text{new}} = [2.22059, -0.05540]}$ |
| **Standard Multi-Pass**| All 4 at once | $m_{\text{global}} = 3.0$ | $d_{\text{global}} = 1.6385$ | — | $\mathbf{O_{\text{global}} = [2.22059, -0.05540]}$ |

---

## 6. Solved Illustrations

### Illustration 1: FlashAttention-2 vs. FlashAttention-3
**Problem:**
What were the critical engineering innovations that allowed FlashAttention-2 and FlashAttention-3 to achieve $2\times$ and $4\times$ speedups over FlashAttention-1?

**Solution:**
1. **FlashAttention-2 (Dao, 2023):**
   - In FlashAttention-1, the outer loop was over $KV$ blocks and inner loop over $Q$ blocks. FlashAttention-2 swapped the loops: outer loop over $Q$, parallelizing across the sequence length of queries. This maximized thread occupancy and reduced shared memory synchronization barriers.
   - Reduced non-matmul FLOPs by deferring division by $d_{\text{new}}$ to the very end of the loop.
2. **FlashAttention-3 (Dao et al., 2024 - NVIDIA Hopper H100):**
   - Exploits **TMA (Tensor Memory Accelerator)** for asynchronous memory transfers directly between HBM and SRAM, bypassing the register file entirely.
   - Uses **WGMMA (Warpgroup Matrix Multiply-Accumulate)** instructions to interleave FP8 tensor core math with asynchronous memory copies in a hardware ping-pong buffer, achieving over **$75\%$ of theoretical peak H100 FLOPs**! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **PyTorch Native Integration:** In PyTorch 2.0+, `torch.nn.functional.scaled_dot_product_attention` automatically detects GPU capabilities and routes to the FlashAttention CUDA kernel if installed.
- **Enabling Long Contexts (128k - 1M):** Without FlashAttention, models like LLaMA 3.1 (128k context) and Gemini 1.5 (1M+ context) would run out of GPU memory immediately during pre-training.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Chunk 1 and Chunk 2 local statistics calculation.
   - Online Softmax merge step matching $O_{\text{new}} = [2.220587, -0.055403]$ against standard global attention to $< 10^{-5}$.
2. **Pure Python/NumPy Tiled FlashAttention Algorithm:**
   - Full block tiling implementation with configurable block sizes $B_r, B_c$.
   - Proves zero $O(L^2)$ HBM memory materialization.
3. **Equivalence Benchmark against PyTorch Native Attention:**
   - Verified on random $L = 64, d = 32$ multi-head tensors.

See implementation in:
[`12_modern_llm_architectures/code/05_flash_attention_online_softmax.py`](./code/05_flash_attention_online_softmax.py)
