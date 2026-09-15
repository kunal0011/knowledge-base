# 12.8 Parameter-Efficient Fine-Tuning: LoRA, QLoRA (NF4 Quantization) & DoRA

---

## 1. Intuition & 101 Motivation

Full parameter fine-tuning of Large Language Models has become intractable for all but the largest industrial research labs. Updating all weights of a 70-billion parameter model in 16-bit precision requires over **1.1 TB of GPU VRAM** to hold optimizer states, gradients, and activations across expensive GPU clusters. Furthermore, deploying distinct full fine-tuned checkpoints for dozens of custom client tasks requires storing hundreds of gigabytes per client.

**Parameter-Efficient Fine-Tuning (PEFT)** solves this crisis based on the **Intrinsic Rank Hypothesis** (Aghajanyan et al., 2020):
> *The parameter updates $\Delta W$ necessary for an LLM to adapt to specific downstream tasks reside in a manifold of extremely low intrinsic dimension (rank $r \ll d$).*

Instead of updating the full dense matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, we freeze $W_0$ entirely and decompose the weight update into two low-rank matrices:
$$\Delta W = \frac{\alpha}{r} B A$$

This unlocks three major paradigms:
1. **LoRA (Low-Rank Adaptation):** Freezes base weights, reducing trainable parameters by **$99\%+$** and drastically slashing optimizer memory.
2. **QLoRA (Quantized LoRA):** Compresses the frozen base weights to 4-bit NormalFloat (NF4), fine-tuning a 70B model on a **single 48 GB GPU** with zero loss in benchmark performance.
3. **DoRA (Weight-Decomposed Low-Rank Adaptation):** Decomposes weight matrices into magnitude vectors and directional matrices, closing the remaining performance gap between LoRA and full fine-tuning.

```
                  LORA vs. FULL FINE-TUNING
                  
     Full Fine-Tuning                        LoRA (Frozen Base + Low-Rank)
  Trainable Dense Weight W                Frozen W0           Trainable BA
   ┌───────────────────┐               ┌───────────┐         ┌─┐
   │                   │               │           │       B │ │  r 
   │     W_0 + ΔW      │       =       │    W_0    │   +     │ │ ┌─────┐
   │                   │               │ (Frozen)  │         └─┘ └─────┘
   └───────────────────┘               └───────────┘              A (r x d)
   Trainable: d_in × d_out             Trainable: 0        Trainable: r(d_in + d_out)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The LoRA Architecture & Forward Formulation

Let $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the pre-trained, frozen weight matrix of a linear projection (typically query $W_q$, key $W_k$, value $W_v$, or MLP up-projections).

For an input token representation $x \in \mathbb{R}^{d_{\text{in}} \times 1}$ (or batch matrix $X \in \mathbb{R}^{B \times d_{\text{in}}}$), the forward pass is:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

where:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$: Down-projection matrix, initialized via random Gaussian $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$: Up-projection matrix, initialized strictly to **zero** ($B = 0$).
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$: Intrinsic low rank (typically $r \in \{8, 16, 32, 64\}$).
- $\alpha \in \mathbb{R}^+$: Constant scaling hyperparameter (typically $\alpha = 2r$ or $\alpha = 16$).
- Scaling factor $\frac{\alpha}{r}$: A stabilizing multiplier ensuring that when experimenting with different ranks $r$, the learning rate does not need to be re-tuned.

#### Property: Zero-Initial-Drift
Because $B = 0$ at initialization:
$$\Delta W_{\text{init}} = \frac{\alpha}{r} B_{\text{init}} A_{\text{init}} = \frac{\alpha}{r} (0) A = 0 \implies h_{\text{init}} = W_0 x$$
The model begins training with the **exact pre-trained behavior**, preventing catastrophic early training instability.

#### Parameter Compression Ratio:
$$\text{Parameters}_{\text{Full}} = d_{\text{in}} \times d_{\text{out}}$$
$$\text{Parameters}_{\text{LoRA}} = r (d_{\text{in}} + d_{\text{out}})$$
$$\text{Compression Factor} = \frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}} = r \left( \frac{1}{d_{\text{out}}} + \frac{1}{d_{\text{in}}} \right)$$

*Example:* For LLaMA-3 70B attention head ($d_{\text{in}} = d_{\text{out}} = 8192$), with rank $r = 16$:
- Full weights: $8192 \times 8192 = 67,108,864$ parameters ($134.2\text{ MB}$ in FP16).
- LoRA adapters: $16 \times (8192 + 8192) = 262,144$ parameters ($0.52\text{ MB}$ in FP16).
- **Compression: $256\times$ fewer parameters (99.61% parameter reduction)!**

---

### 2.2 Zero Inference Latency via Weight Merging

During training, $W_0 x$ and $\frac{\alpha}{r} B A x$ are computed separately to avoid materializing $\Delta W$ in memory.
However, for deployment in production inference, the adapter can be **fused directly** into the base weight:

$$W_{\text{deploy}} = W_0 + \frac{\alpha}{r} B A \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

Because matrix multiplication is distributive:
$$W_{\text{deploy}} x = (W_0 + \Delta W) x = W_0 x + \Delta W x$$
This means LoRA introduces **0% additional inference latency and 0 bytes of runtime memory overhead** when deployed as a dedicated single model!

---

### 2.3 QLoRA & 4-bit NormalFloat (NF4)

QLoRA (Dettmers et al., 2023) dramatically reduces the memory footprint of $W_0$ by quantizing the frozen weights to 4 bits while keeping the low-rank adapters in 16-bit BF16.

#### 1. Information-Theoretic 4-Bit NormalFloat (NF4):
Standard uniform quantization (INT4) divides the dynamic range $[-x_{\text{max}}, x_{\text{max}}]$ into 16 evenly spaced intervals. However, pre-trained neural network weights follow a **zero-mean normal distribution**:
$$W_0 \sim \mathcal{N}(0, \sigma^2)$$

Uniform quantization wastes precious bits in the low-probability tails while poorly resolving high-density values near 0.
The **NF4 (NormalFloat4)** data type builds 16 discrete quantile bins $q_i$ such that every bin has an **equal probability of occurrence** under standard normal distribution $\mathcal{N}(0, 1)$:

$$q_i = \frac{1}{2}\left( Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right) \right), \quad i \in \{0, \dots, 15\}, \quad k = 4$$

The values are normalized so that the dynamic range spans $[-1.0, 1.0]$, with an exact representation for zero ($q = 0.0$ is a discrete bin).

#### 2. Double Quantization (DQ):
Weights are quantized in blocks of size $B_1 = 64$. Each block has an FP32 quantization scaling constant $c_1$.
Normally, storing $c_1$ requires $32 \text{ bits} / 64 \text{ weights} = 0.5 \text{ bits/weight}$.
Double Quantization quantizes the first-level scaling constants $c_1$ into 8-bit FP8 constants with a second block size $B_2 = 256$:
$$\text{Memory Overhead of } c_1 = \frac{8 \text{ bits}}{64} + \frac{32 \text{ bits}}{64 \times 256} \approx 0.125 + 0.002 = \mathbf{0.127 \text{ bits/weight}}$$
Saves **$0.373$ bits per parameter**, which across a 70B model saves over **$3.2\text{ GB}$ of VRAM**!

---

### 2.4 DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA (Liu et al., 2024) decomposes any weight matrix into its **magnitude** $\|W\|_c$ (column-wise norm) and its **direction** $V / \|V\|_c$:

$$W = m \frac{V}{\|V\|_c}$$

where:
- $m \in \mathbb{R}^{1 \times d_{\text{in}}}$: Magnitude vector, initialized to $m = \|W_0\|_c$.
- $V = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$: Directional matrix adjusted by LoRA.
- $\|\cdot\|_c$: Vector norm across each column.

By decoupling directional learning from magnitude scaling, DoRA mimics the optimization trajectory of full fine-tuning, consistently outperforming standard LoRA on mathematical reasoning and coding benchmarks.

---

## 3. Geometric & Physical Interpretation

### 3.1 Low-Rank Subspace Projection
In high-dimensional weight space $\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, full fine-tuning allows gradients to travel along all millions of dimensions.
LoRA restricts weight updates to an $r$-dimensional subspace spanned by the column space of $B$. The down-projection matrix $A$ acts as a **feature compressor**, while $B$ acts as a **feature expander**.

```
   Dimension 2
        ▲
        │         Full Fine-Tuning: unconstrained traversal
        │           \  /\  /
        │            \/  \/  * (Final Optimum)
        │
        │         LoRA Traversal: restricted to r-dimensional hyperplane
        │          ──────*────────► (Span of Low-Rank B)
        │         /
        └────────────────────────► Dimension 1
```

---

## 4. Real-World Analogy: The Textbook and the Sticky Notes

Imagine an expensive, 1,000-page medical encyclopedia ($W_0$):
- **Full Fine-Tuning:** You hire a printing press to re-print all 1,000 pages of the encyclopedia from scratch, altering thousands of sentences ($16N$ memory footprint, high cost).
- **LoRA:** You keep the encyclopedia pristine and frozen. Whenever you encounter a new medical subfield (e.g., pediatric cardiology), you carry a small pack of sticky notes ($BA$). You write short notes on them and stick them to the relevant pages. When reading the page, you combine the printed text with your sticky note.
- **Weight Merging:** Once your annotations are finalized, you photocopy the pages with sticky notes attached to create a single clean monograph ($W_{\text{deploy}}$) with zero reading delay.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete, cell-by-cell numerical example of:
1. **LoRA Forward Pass on Input $x$**
2. **LoRA Adapter Computation ($B A x$) and Scaling**
3. **Weight Merge Equivalency Verification**

---

### 5.1 Concrete Input Values

- **Input vector:** $x = [2.0, 1.0]^T \in \mathbb{R}^2$ ($d_{\text{in}} = 2$)
- **Base weight matrix ($d_{\text{out}} = 2, d_{\text{in}} = 2$):**
  $$W_0 = \begin{bmatrix} 1.0 & 2.0 \\ 0.0 & -1.0 \end{bmatrix}$$
- **Low-Rank configuration:** $r = 1$, $\alpha = 2.0 \implies \text{Scaling Factor } \frac{\alpha}{r} = \frac{2.0}{1} = \mathbf{2.0}$
- **Down-projection matrix $A \in \mathbb{R}^{1 \times 2}$:**
  $$A = \begin{bmatrix} 1.0 & 0.5 \end{bmatrix}$$
- **Up-projection matrix $B \in \mathbb{R}^{2 \times 1}$:**
  $$B = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $x$ | `input_x` | Input token activation vector $[2.0, 1.0]^T$ |
| $y_0$ | `base_output` | Unadapted base output $W_0 x$ |
| $u$ | `adapter_mid` | Down-projected scalar activation $A x$ |
| $v$ | `adapter_proj` | Up-projected vector activation $B u$ |
| $\Delta y$ | `scaled_delta` | Scaled adapter contribution $\frac{\alpha}{r} v$ |
| $y$ | `final_output` | Total adapted forward activation $y_0 + \Delta y$ |
| $W_{\text{merged}}$ | `merged_weights` | Fused deployable weight matrix $W_0 + \frac{\alpha}{r} B A$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Base Model Output $y_0 = W_0 x$
$$y_{0, 1} = (1.0 \times 2.0) + (2.0 \times 1.0) = 2.0 + 2.0 = \mathbf{4.000000}$$
$$y_{0, 2} = (0.0 \times 2.0) + (-1.0 \times 1.0) = 0.0 - 1.0 = \mathbf{-1.000000}$$
$$y_0 = \begin{bmatrix} 4.0 \\ -1.0 \end{bmatrix}$$

---

#### Step 2: Compute Adapter Down-Projection $u = A x$
$$u = \begin{bmatrix} 1.0 & 0.5 \end{bmatrix} \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = (1.0 \times 2.0) + (0.5 \times 1.0) = 2.0 + 0.5 = \mathbf{2.500000}$$
*(Input compressed from dimension 2 to scalar dimension $r = 1$)*.

---

#### Step 3: Compute Adapter Up-Projection $v = B u$
$$v = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} (2.5) = \begin{bmatrix} 0.5 \times 2.5 \\ -0.5 \times 2.5 \end{bmatrix} = \begin{bmatrix} \mathbf{1.250000} \\ \mathbf{-1.250000} \end{bmatrix}$$

---

#### Step 4: Scale by $\frac{\alpha}{r} = 2.0$
$$\Delta y = 2.0 \times v = 2.0 \times \begin{bmatrix} 1.25 \\ -1.25 \end{bmatrix} = \begin{bmatrix} \mathbf{2.500000} \\ \mathbf{-2.500000} \end{bmatrix}$$

---

#### Step 5: Compute Final LoRA Output $y = y_0 + \Delta y$
$$y = \begin{bmatrix} 4.0 \\ -1.0 \end{bmatrix} + \begin{bmatrix} 2.5 \\ -2.5 \end{bmatrix} = \begin{bmatrix} 4.0 + 2.5 \\ -1.0 - 2.5 \end{bmatrix} = \begin{bmatrix} \mathbf{6.500000} \\ \mathbf{-3.500000} \end{bmatrix}$$

---

#### Step 6: Verify Weight Merging Invariance $W_{\text{merged}} x \equiv y$
First compute $\Delta W = \frac{\alpha}{r} B A$:
$$B A = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} 1.0 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.5 \times 1.0 & 0.5 \times 0.5 \\ -0.5 \times 1.0 & -0.5 \times 0.5 \end{bmatrix} = \begin{bmatrix} 0.50 & 0.25 \\ -0.50 & -0.25 \end{bmatrix}$$

Multiply by scaling factor $\frac{\alpha}{r} = 2.0$:
$$\Delta W = 2.0 \times \begin{bmatrix} 0.50 & 0.25 \\ -0.50 & -0.25 \end{bmatrix} = \begin{bmatrix} \mathbf{1.00} & \mathbf{0.50} \\ \mathbf{-1.00} & \mathbf{-0.50} \end{bmatrix}$$

Merge into base weight $W_0$:
$$W_{\text{merged}} = W_0 + \Delta W = \begin{bmatrix} 1.0 & 2.0 \\ 0.0 & -1.0 \end{bmatrix} + \begin{bmatrix} 1.0 & 0.5 \\ -1.0 & -0.5 \end{bmatrix} = \begin{bmatrix} \mathbf{2.00} & \mathbf{2.50} \\ \mathbf{-1.00} & \mathbf{-1.50} \end{bmatrix}$$

Compute product $W_{\text{merged}} x$:
$$W_{\text{merged}} x = \begin{bmatrix} 2.00 & 2.50 \\ -1.00 & -1.50 \end{bmatrix} \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} (2.0 \times 2.0) + (2.5 \times 1.0) \\ (-1.0 \times 2.0) + (-1.5 \times 1.0) \end{bmatrix}$$
$$= \begin{bmatrix} 4.0 + 2.5 \\ -2.0 - 1.5 \end{bmatrix} = \begin{bmatrix} \mathbf{6.500000} \\ \mathbf{-3.500000} \end{bmatrix}$$

**Exact Match to Step 5! Zero error ($\Delta = 0.000000$).** $\blacksquare$

---

### 5.4 Summary Visual Grid: LoRA Execution Ledger

| Operation | Matrix Expression | Numerical Dimensions | Values | Role |
| :--- | :---: | :---: | :---: | :--- |
| **Input** | $x$ | $2 \times 1$ | $[2.0, 1.0]^T$ | Token input |
| **Base Pass** | $W_0 x$ | $2 \times 1$ | $[4.0, -1.0]^T$ | Frozen base activation |
| **Down-Project** | $A x$ | $1 \times 1$ | $[2.5]$ | Low-rank compression ($r=1$) |
| **Up-Project** | $B(A x)$ | $2 \times 1$ | $[1.25, -1.25]^T$ | Subspace expansion |
| **LoRA Delta** | $\frac{\alpha}{r} B A x$ | $2 \times 1$ | $[2.5, -2.5]^T$ | Scaled update |
| **Total Out** | $y_0 + \Delta y$ | $2 \times 1$ | $\mathbf{[6.5, -3.5]^T}$ | Output activation |
| **Merged Check** | $W_{\text{merged}} x$ | $2 \times 1$ | $\mathbf{[6.5, -3.5]^T}$ | Verified identical |

---

## 6. Solved Illustrations

### Illustration 1: Memory Footprint: Full Fine-Tuning vs. LoRA vs. QLoRA
**Problem:**
Calculate the GPU memory required to fine-tune a 70-billion parameter model ($N = 70 \times 10^9$) with batch size $B=1$ across three paradigms:
1. Full Fine-Tuning in FP16/BF16 (with AdamW)
2. LoRA in FP16 ($r = 16$, targeting all linear layers accounting for 80% of parameters)
3. QLoRA with 4-bit NF4 base weights + LoRA in BF16

**Solution:**
1. **Full Fine-Tuning:**
   - Model weights (FP16): $2 \times 70 = 140\text{ GB}$
   - Gradients (FP16): $2 \times 70 = 140\text{ GB}$
   - AdamW states (FP32 master + moments): $12 \times 70 = 840\text{ GB}$
   - Total static VRAM: $140 + 140 + 840 = \mathbf{1,120\text{ GB}}$ *(Requires at least 16x 80GB H100 GPUs!)*

2. **LoRA in FP16 ($r = 16$):**
   - Base weights (frozen, FP16): $2 \times 70 = 140\text{ GB}$
   - Trainable adapter parameters: $\approx 0.5\% \times 70\text{B} = 0.35\text{B}$ params.
   - Adapter weights (FP16): $2 \times 0.35 = 0.7\text{ GB}$
   - Adapter gradients (FP16): $0.7\text{ GB}$
   - Adapter AdamW states: $12 \times 0.35 = 4.2\text{ GB}$
   - Total static VRAM: $140 + 0.7 + 0.7 + 4.2 = \mathbf{145.6\text{ GB}}$ *(Fits on 2x 80GB GPUs!)*

3. **QLoRA (4-bit NF4):**
   - Base weights (NF4 + Double Quantization): $0.55 \text{ bytes/param} \times 70 = 38.5\text{ GB}$
   - Base gradients & AdamW: $0\text{ GB}$ (Frozen)
   - Adapter parameters (BF16): $0.7 + 0.7 + 4.2 = 5.6\text{ GB}$
   - Total static VRAM: $38.5 + 5.6 = \mathbf{44.1\text{ GB}}$ *(Fits on a **single 48GB A6000 / A100** GPU!).* $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **Hugging Face PEFT & Unsloth:** The de facto open-source infrastructure for low-rank fine-tuning. Unsloth writes custom OpenAI Triton kernels for QLoRA that bypass Python overhead, achieving $2\times$ faster training and $70\%$ less VRAM.
- **Multi-Tenant Serving in vLLM:** Rather than loading 50 separate 70B models for 50 clients, vLLM loads a **single 4-bit base model** and dynamically swaps tiny 50MB LoRA adapter weights in GPU SRAM on a per-request basis, enabling 100x higher serving density!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Base forward pass, down-projection, up-projection, and scaled output matching $[6.5, -3.5]^T$.
   - Matrix merge verification confirming $W_{\text{merged}} x \equiv y$.
2. **Production-Grade `LoRALinear` Layer:**
   - Parameter freezing for $W_0$.
   - Trainable low-rank $A$ and $B$ decomposition.
   - `merge()` and `unmerge()` runtime methods.
   - Full gradient backpropagation unit test.

See implementation in:
[`12_modern_llm_architectures/code/08_peft_lora_and_qlora.py`](./code/08_peft_lora_and_qlora.py)
