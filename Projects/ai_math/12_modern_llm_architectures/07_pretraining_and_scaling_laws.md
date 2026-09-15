# 12.7 Pre-Training Engineering: Chinchilla Scaling Laws, WSD Schedules & Distributed Training (FSDP/ZeRO)

---

## 1. Intuition & 101 Motivation

Pre-training accounts for over **99% of the computational budget, energy, and wall-clock time** in the lifecycle of any Large Language Model. A single training run for a frontier foundation model (e.g., LLaMA-3 405B, DeepSeek-V3) requires tens of thousands of GPUs running continuously for months, consuming millions of megawatt-hours and tens of millions of dollars.

At this industrial scale, guessing hyperparameters or model sizes is catastrophic. Pre-training engineering is governed by three foundational pillars:
1. **Compute Optimal Scaling Laws (Chinchilla):** For a given compute budget in floating-point operations (FLOPs), what is the optimal trade-off between the number of parameters $N$ and the number of training tokens $D$?
2. **Learning Rate Dynamics (Warmup-Stable-Decay - WSD):** Traditional cosine decay ties the learning rate schedule to a fixed token horizon $T$. Modern foundation models adopt **WSD (Warmup-Stable-Decay)**, enabling perpetual pre-training, dynamic branching, and continual domain adaptation without restarting from scratch.
3. **Distributed Memory Engineering (ZeRO & FSDP):** A 70-billion parameter model cannot be trained on an 80 GB GPU. Training requires storing parameters, gradients, and optimizer states, totaling ~16–20 bytes per parameter ($\approx 1.12$–$1.4$ TB for 70B). Distributed memory partitioning (**ZeRO-1/2/3** and PyTorch **FSDP**) breaks this memory wall.

```
       PRE-TRAINING TRIFECTA
       
   Compute Allocation          Optimization Dynamics         Hardware Memory
  [Chinchilla Scaling]     [Warmup-Stable-Decay (WSD)]    [ZeRO-1/2/3 & FSDP]
  C = 6·N·D FLOPs           Perpetual Training Sched.     Sharded States (16N/P)
  N_opt ∝ √C, D_opt ∝ √C    Branching & Continual Learn   Breaks 80GB Memory Wall
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Transformer FLOP Budget Equation

For an autoregressive decoder-only Transformer with $N$ non-embedding parameters processing $D$ tokens:

$$\text{FLOPs per forward pass per token} \approx 2N$$
$$\text{FLOPs per backward pass per token} \approx 4N$$
$$\text{Total Compute Budget } C \approx 6ND$$

#### Mathematical Proof / Derivation:
1. In a linear projection $y = x W$ where $x \in \mathbb{R}^{1 \times d_{\text{in}}}$ and $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:
   - There are $d_{\text{in}} \times d_{\text{out}}$ multiplications and $d_{\text{in}} \times d_{\text{out}}$ additions, totaling $2 \times (d_{\text{in}} d_{\text{out}}) = 2 N_{\text{layer}}$ floating point operations.
2. In the forward pass, this applies to all projection weights in Attention ($W_q, W_k, W_v, W_o$) and MLP ($W_1, W_2, W_3$), yielding $2N$ FLOPs per token.
3. In the backward pass:
   - Gradient with respect to input $\nabla_x \mathcal{L} = (\nabla_y \mathcal{L}) W^T$: requires $2N$ FLOPs.
   - Gradient with respect to weights $\nabla_W \mathcal{L} = x^T (\nabla_y \mathcal{L})$: requires $2N$ FLOPs.
   - Total backward FLOPs = $4N$ per token.
4. Total training compute per token: $2N + 4N = 6N$ FLOPs.
5. Over $D$ tokens: $C = 6ND$ FLOPs.

---

### 2.2 Chinchilla Scaling Law Derivation (Hoffmann et al., 2022)

The pre-training cross-entropy loss $L(N, D)$ can be modeled as a power-law decomposition:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where:
- $E$: Irreducible loss of the natural language data distribution.
- $A / N^\alpha$: Parameter-constrained excess risk (under-parameterization penalty).
- $B / D^\beta$: Data-constrained excess risk (under-training penalty).
- $A, B, \alpha, \beta$: Empirically fitted positive constants ($\alpha \approx 0.34, \beta \approx 0.28$).

#### Optimization under Fixed Compute Budget $C$:
We seek to minimize $L(N, D)$ subject to $C = 6ND \implies D = \frac{C}{6N}$.

Substitute $D$ into $L(N, D)$:
$$\min_N \left( \frac{A}{N^\alpha} + \frac{B}{\left(\frac{C}{6N}\right)^\beta} \right) = \min_N \left( A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^\beta \right)$$

Differentiating with respect to $N$ and setting to 0:
$$\frac{d}{dN} \left[ A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^\beta \right] = -\alpha A N^{-\alpha - 1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1} = 0$$

$$\alpha A N^{-\alpha - 1} = \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1}$$

Multiply both sides by $N^{\alpha + 1}$:
$$\alpha A = \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\alpha + \beta}$$

Solving for $N_{\text{opt}}$:
$$N_{\text{opt}}^{\alpha + \beta} = \frac{\alpha A}{\beta B} \left(\frac{C}{6}\right)^\beta \implies N_{\text{opt}} = G \cdot C^{\frac{\beta}{\alpha + \beta}} = G \cdot C^a$$

Since $D = \frac{C}{6N}$:
$$D_{\text{opt}} \propto \frac{C}{C^{\frac{\beta}{\alpha + \beta}}} = C^{1 - \frac{\beta}{\alpha + \beta}} = C^{\frac{\alpha}{\alpha + \beta}} = C^b$$

Because $\alpha \approx 0.34$ and $\beta \approx 0.28$:
$$a = \frac{0.28}{0.34 + 0.28} = \frac{0.28}{0.62} \approx 0.45, \quad b = \frac{0.34}{0.34 + 0.28} = \frac{0.34}{0.62} \approx 0.55$$

Since $a \approx 0.5$ and $b \approx 0.5$:
$$\mathbf{N_{\text{opt}} \propto \sqrt{C}, \quad D_{\text{opt}} \propto \sqrt{C}}$$

**The Chinchilla Ratio:** Model parameters and data tokens should scale in **equal proportion**! Specifically, for compute-optimal training:
$$\frac{D_{\text{opt}}}{N_{\text{opt}}} \approx 20 \text{ tokens per parameter}$$

#### Modern Inference-Optimal Regime (Overtraining):
Chinchilla optimality minimizes **training** compute. However, once trained, a model is queried billions of times during **inference**. The cost of serving an $N$-parameter model scales linearly with $N$. Hence, modern architectures (LLaMA-3, DeepSeek) train smaller models on far more data ($D/N \ge 100\text{--}200$ tokens per parameter) to minimize downstream inference latency and memory.

---

### 2.3 Learning Rate Schedules: Cosine vs. Warmup-Stable-Decay (WSD)

#### 1. Standard Cosine Annealing Schedule:
$$\eta_t = \begin{cases} \eta_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}} & t < T_{\text{warmup}} \\ \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\pi\right)\right) & t \ge T_{\text{warmup}} \end{cases}$$

**The Cosine Flaw:** The schedule is intrinsically hardcoded to $T_{\text{total}}$. If training is interrupted early at $t = 0.5 T_{\text{total}}$, the model has not cooled down and performs suboptimally. Conversely, extending training beyond $T_{\text{total}}$ requires restarting or manual learning rate surgery.

#### 2. Warmup-Stable-Decay (WSD) Schedule (MiniCPM, DeepSeek-V3):
$$\eta_t = \begin{cases} \eta_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}} & 0 \le t < T_{\text{warmup}} \quad \text{(Warmup)} \\ \eta_{\text{max}} & T_{\text{warmup}} \le t < T_{\text{decay\_start}} \quad \text{(Stable)} \\ \eta_{\text{min}} + (\eta_{\text{max}} - \eta_{\text{min}}) \cdot f_{\text{decay}}(t) & t \ge T_{\text{decay\_start}} \quad \text{(Decay)} \end{cases}$$

where decay is typically 10% of the remaining steps using cosine or exponential annealing:
$$f_{\text{decay}}(t) = \frac{1}{2}\left(1 + \cos\left(\frac{t - T_{\text{decay\_start}}}{T_{\text{total}} - T_{\text{decay\_start}}}\pi\right)\right)$$

**WSD Advantages:**
- **Perpetual Training:** Keep training in the **Stable** phase indefinitely as new data arrives.
- **Branching Checkpoints:** Branch at any checkpoint in the Stable phase to train specialized models (code, math, multilingual) and decay only the final 10% of tokens.

---

### 2.4 Distributed Memory Breakdown: The 16N Memory Wall

Training an $N$-parameter model using 16-bit mixed precision (FP16 or BF16) with the standard AdamW optimizer requires storing four core tensor categories:

1. **Model Parameters ($W$):** In 16-bit precision:
   $$\text{Memory}_{\text{weights}} = 2 \text{ bytes/parameter} \implies \mathbf{2N \text{ bytes}}$$
2. **Gradients ($g$):** In 16-bit precision:
   $$\text{Memory}_{\text{gradients}} = 2 \text{ bytes/parameter} \implies \mathbf{2N \text{ bytes}}$$
3. **AdamW Optimizer States:**
   - FP32 Master Weights (for numerical stability under small gradient updates): $4N \text{ bytes}$
   - FP32 First Moment $m_t$ (momentum): $4N \text{ bytes}$
   - FP32 Second Moment $v_t$ (variance): $4N \text{ bytes}$
   $$\text{Memory}_{\text{optimizer}} = 4N + 4N + 4N = \mathbf{12N \text{ bytes}}$$
4. **Total Static Memory (Weights + Gradients + Optimizer):**
   $$\text{Memory}_{\text{static}} = 2N + 2N + 12N = \mathbf{16N \text{ bytes}}$$

*Example:* For a 70B parameter model ($N = 70 \times 10^9$):
$$\text{Static Memory} = 16 \times 70 \times 10^9 = 1,120 \times 10^9 \text{ bytes} \approx \mathbf{1,043 \text{ GiB}}$$
An 80 GB NVIDIA H100 GPU can only hold **~5B parameters** in standard Data Parallelism (DDP)!

```
STATIC MEMORY BREAKDOWN PER PARAMETER (16 Bytes Total)
+---------------+---------------+-----------------------------------------------+
| Weights (2B)  | Gradients (2B)|             AdamW States (12B)                |
| FP16/BF16     | FP16/BF16     | Master W (4B) + 1st Mom m (4B) + 2nd Mom v(4B)|
+---------------+---------------+-----------------------------------------------+
```

---

### 2.5 ZeRO (Zero Redundancy Optimizer) & FSDP

To eliminate memory redundancy across $P$ data-parallel GPUs:

| Memory Partition Level | Sharded Components | Replicated Components | Memory per GPU |
| :--- | :--- | :--- | :--- |
| **Standard DDP** | None | Weights, Gradients, Optimizer | $16N$ |
| **ZeRO Stage 1** | Optimizer States ($12N$) | Weights ($2N$), Gradients ($2N$) | $4N + \frac{12N}{P}$ |
| **ZeRO Stage 2** | Optimizer States ($12N$), Gradients ($2N$) | Weights ($2N$) | $2N + \frac{14N}{P}$ |
| **ZeRO Stage 3 / FSDP** | Optimizer ($12N$), Gradients ($2N$), Weights ($2N$) | None (all sharded) | $\frac{16N}{P}$ |

#### Communication Overhead in ZeRO-3 / FSDP:
In standard DDP, GPUs perform an `All-Reduce` of gradients ($2 \times \text{size}$ communication volume).
In ZeRO-3 / FSDP:
1. **Forward Pass:** `All-Gather` weights layer-by-layer ($N$ bytes). Layer computes activations, then discards non-local weights.
2. **Backward Pass:** `All-Gather` weights again layer-by-layer ($N$ bytes) to compute gradients.
3. **Gradient Reduction:** `Reduce-Scatter` gradients ($N$ bytes) directly into the sharded optimizer states.
Total communication volume: $3N$ bytes per step vs. $2N$ bytes in DDP (only **50% extra communication** for an infinite memory scaling ceiling!).

---

## 3. Geometric & Physical Interpretation

### 3.1 Loss Hyperbolas in Iso-FLOP Space
Plotting $\log N$ on the x-axis and $\log D$ on the y-axis, the compute constraint $C = 6ND \implies \log C = \log 6 + \log N + \log D$ forms straight downward-sloping lines ($45^\circ$ angle: $\log D = \text{const} - \log N$).
The loss function $L(N, D)$ forms concentric elliptical contours. The point where the constant-compute constraint line is tangent to the lowest loss contour defines the Chinchilla optimal point $(N^*, D^*)$.

```
   Log Data (D)
        ▲
        │       Iso-Loss Contours
        │      (    (      (
        │     (    (  * (Chinchilla Optimum)
        │    (    (    \   (
        │   (    (      \   (
        │  ───────────────────► Iso-FLOP Line (C = 6ND)
        └────────────────────────► Log Parameters (N)
```

### 3.2 Annealing in WSD
In physics, **annealing** is the process of heating a metal and then slowly cooling it so atoms settle into minimum-energy crystalline lattices.
In WSD:
- The **Stable** phase ($0.9 T_{\text{total}}$) operates at high temperature ($\eta_{\text{max}}$), aggressively exploring wide basins in the non-convex loss landscape.
- The **Decay** phase ($0.1 T_{\text{total}}$) cools down rapidly, allowing the weights to collapse into the deepest local minimum of the basin.

---

## 4. Real-World Analogy: The Industrial Bakery

Imagine opening an industrial bakery:
- **Parameters $N$:** The number of specialized pastry chefs.
- **Data Tokens $D$:** The kilograms of raw flour, sugar, and butter.
- **Compute Budget $C$:** Your financial capital for the season.
- **Kaplan (2020) Error:** Hired 100 chefs with only 10 kg of flour. Chefs stood idle waiting for ingredients (under-trained model).
- **Chinchilla (2022) Discovery:** If your budget increases $4\times$, you should hire $2\times$ as many chefs ($N \propto \sqrt{C}$) and buy $2\times$ as much flour ($D \propto \sqrt{C}$).
- **ZeRO Stage 1–3:**
  - *Standard DDP:* Every chef buys their own 200 kg mixer, oven, and cookbook ($16N$ memory footprint per chef).
  - *ZeRO-3 (FSDP):* Chefs share a single kitchen warehouse. Chef A holds the pastry dough recipe, Chef B holds the croissants, Chef C holds the glazes. When a croissant needs baking, Chef B calls Chef A on the radio (`All-Gather`), finishes the step, and returns the station to clean storage.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us work through a complete, concrete numerical trace of:
1. **Chinchilla Compute & Token Allocation**
2. **Static GPU Memory Breakdown across DDP, ZeRO-1, ZeRO-2, and ZeRO-3**
3. **Warmup-Stable-Decay (WSD) vs. Cosine Learning Rate Schedule**

---

### 5.1 Problem Setup & Input Concrete Values

- **Target Compute Budget:** $C = 6.0 \times 10^{20}$ FLOPs
- **Candidate Model Size:** $N = 5 \times 10^9$ parameters ($5\text{B}$)
- **Target Distributed Cluster:** $P = 8$ GPUs (each with $80\text{ GiB}$ VRAM)
- **Schedule Parameters:**
  - Total steps $T = 1,000$ steps
  - Warmup steps $T_{\text{warmup}} = 100$ steps
  - Decay start step $T_{\text{decay}} = 800$ steps
  - $\eta_{\text{max}} = 1.0 \times 10^{-3}$, $\eta_{\text{min}} = 1.0 \times 10^{-5}$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $C$ | `total_flops` | Total available floating point operations ($6.0 \times 10^{20}$) |
| $N$ | `param_count` | Number of non-embedding model parameters ($5 \times 10^9$) |
| $D$ | `token_budget` | Data token count: $D = \frac{C}{6N}$ |
| $M_{\text{static}}$ | `static_mem_gb` | Combined memory of Weights, Gradients, and AdamW states |
| $P$ | `world_size` | Number of participating GPUs in data parallelism ($P = 8$) |
| $\eta_t$ | `learning_rate` | Current learning rate at step $t$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Part A: Compute-Optimal Token Budget
$$D = \frac{C}{6N} = \frac{6.0 \times 10^{20}}{6 \times (5.0 \times 10^9)} = \frac{6.0 \times 10^{20}}{30.0 \times 10^9} = \mathbf{20.0 \times 10^9 \text{ tokens (20 Billion tokens)}}$$
Ratio: $\frac{D}{N} = \frac{20 \times 10^9}{5 \times 10^9} = \mathbf{4.0 \text{ tokens/parameter}}$.
*(Note: Chinchilla optimal recommends $\approx 20\text{ tokens/param}$; this model is in the compute-efficient prototyping tier).*

---

#### Part B: Memory Consumption Breakdown for $N = 5\text{B}$ Model ($5 \times 10^9$ params)
Convert bytes directly to Gigabytes ($1 \text{ GB} = 10^9 \text{ bytes}$):
1. **Weights (16-bit):** $2 \times 5 \times 10^9 \text{ bytes} = \mathbf{10.0 \text{ GB}}$
2. **Gradients (16-bit):** $2 \times 5 \times 10^9 \text{ bytes} = \mathbf{10.0 \text{ GB}}$
3. **AdamW Optimizer States (32-bit):**
   - FP32 Master Weights: $4 \times 5 \times 10^9 = 20.0 \text{ GB}$
   - FP32 Momentum ($m$): $4 \times 5 \times 10^9 = 20.0 \text{ GB}$
   - FP32 Variance ($v$): $4 \times 5 \times 10^9 = 20.0 \text{ GB}$
   - Subtotal Optimizer: $12 \times 5 \times 10^9 = \mathbf{60.0 \text{ GB}}$
4. **Total Static Memory:**
   $$M_{\text{static}} = 10.0 + 10.0 + 60.0 = \mathbf{80.0 \text{ GB}}$$

Now calculate memory per GPU across $P = 8$ GPUs:
- **Standard DDP:** Replicates everything on all GPUs:
  $$M_{\text{DDP}} = 80.0 \text{ GB}$$
  *(Exceeds safe threshold when activations are added; instant OOM on an 80GB GPU!)*
- **ZeRO Stage 1 (Optimizer Sharded):**
  $$M_{\text{ZeRO-1}} = 10.0 + 10.0 + \frac{60.0}{8} = 20.0 + 7.50 = \mathbf{27.50 \text{ GB}}$$
- **ZeRO Stage 2 (Optimizer + Gradient Sharded):**
  $$M_{\text{ZeRO-2}} = 10.0 + \frac{10.0 + 60.0}{8} = 10.0 + \frac{70.0}{8} = 10.0 + 8.75 = \mathbf{18.75 \text{ GB}}$$
- **ZeRO Stage 3 / FSDP (All Sharded):**
  $$M_{\text{ZeRO-3}} = \frac{80.0}{8} = \mathbf{10.00 \text{ GB}}$$
  *(Reduces memory from 80 GB down to 10 GB — an **$8\times$ reduction**, leaving 70 GB for batch activations!).*

---

#### Part C: WSD vs. Cosine Learning Rate Hand Trace
Trace learning rate at step $t = 50, 500, 900$:

1. **Step $t = 50$ ($t < T_{\text{warmup}} = 100$):**
   $$\eta_{50}^{\text{WSD}} = \eta_{50}^{\text{Cosine}} = 1.0 \times 10^{-3} \times \left(\frac{50}{100}\right) = \mathbf{0.000500}$$
2. **Step $t = 500$ (Middle of training):**
   - **WSD (Stable Phase):**
     $$\eta_{500}^{\text{WSD}} = \eta_{\text{max}} = \mathbf{0.001000}$$
   - **Cosine Schedule ($T_{\text{total}} = 1000$):**
     $$\theta = \frac{500 - 100}{1000 - 100}\pi = \frac{400}{900}\pi = \frac{4}{9}\pi \approx 1.39626 \text{ rad}$$
     $$\cos(1.39626) \approx 0.173648$$
     $$\eta_{500}^{\text{Cosine}} = 10^{-5} + \frac{1}{2}(10^{-3} - 10^{-5})(1 + 0.173648) \approx 10^{-5} + 0.000495 \times 1.173648 = \mathbf{0.000591}$$
3. **Step $t = 900$ (Decay Phase):**
   - **WSD (Decay Phase from 800 to 1000):**
     $$\theta = \frac{900 - 800}{1000 - 800}\pi = \frac{100}{200}\pi = 0.5\pi \implies \cos(0.5\pi) = 0.0$$
     $$\eta_{900}^{\text{WSD}} = 10^{-5} + (10^{-3} - 10^{-5}) \cdot \frac{1}{2}(1 + 0.0) \approx 10^{-5} + 0.000495 = \mathbf{0.000505}$$
   - **Cosine Schedule ($t = 900$):**
     $$\theta = \frac{900 - 100}{900}\pi = \frac{8}{9}\pi \approx 2.79253 \text{ rad}$$
     $$\cos(2.79253) \approx -0.939693$$
     $$\eta_{900}^{\text{Cosine}} = 10^{-5} + 0.000495 \times (1 - 0.939693) = 10^{-5} + 0.000495 \times 0.060307 = \mathbf{0.0000398}$$

---

### 5.4 Summary Visual Grid: Distributed Memory Partition Ledger

| Tensor Component | Raw Size ($5\text{B}$) | Standard DDP | ZeRO-1 ($P=8$) | ZeRO-2 ($P=8$) | ZeRO-3 / FSDP ($P=8$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Weights (FP16)** | $10.0 \text{ GB}$ | $10.0 \text{ GB}$ | $10.0 \text{ GB}$ | $10.0 \text{ GB}$ | **$1.25 \text{ GB}$** |
| **Gradients (FP16)** | $10.0 \text{ GB}$ | $10.0 \text{ GB}$ | $10.0 \text{ GB}$ | **$1.25 \text{ GB}$** | **$1.25 \text{ GB}$** |
| **AdamW Master W (FP32)** | $20.0 \text{ GB}$ | $20.0 \text{ GB}$ | **$2.50 \text{ GB}$** | **$2.50 \text{ GB}$** | **$2.50 \text{ GB}$** |
| **AdamW Momentum (FP32)** | $20.0 \text{ GB}$ | $20.0 \text{ GB}$ | **$2.50 \text{ GB}$** | **$2.50 \text{ GB}$** | **$2.50 \text{ GB}$** |
| **AdamW Variance (FP32)** | $20.0 \text{ GB}$ | $20.0 \text{ GB}$ | **$2.50 \text{ GB}$** | **$2.50 \text{ GB}$** | **$2.50 \text{ GB}$** |
| **Total Static Memory** | **$80.0 \text{ GB}$** | **$80.0 \text{ GB}$** | **$27.5 \text{ GB}$** | **$18.75 \text{ GB}$** | **$\mathbf{10.0 \text{ GB}}$** |

---

## 6. Solved Illustrations

### Illustration 1: Chinchilla Optimal Allocation for a 1-Billion Parameter Model
**Problem:**
A laboratory has a budget of $C = 1.2 \times 10^{20}$ FLOPs.
1. What is the compute-optimal number of tokens $D_{\text{opt}}$ to train an $N = 1.0 \times 10^9$ (1B) parameter model?
2. What is the Chinchilla ratio $D/N$?
3. How long would this run take on 64 NVIDIA A100 GPUs operating at 150 TFLOP/s (effective MFU)?

**Solution:**
1. **Token Budget:**
   $$D = \frac{C}{6N} = \frac{1.2 \times 10^{20}}{6 \times (1.0 \times 10^9)} = \frac{1.2 \times 10^{20}}{6.0 \times 10^9} = \mathbf{2.0 \times 10^{10} \text{ tokens (20 Billion tokens)}}$$
2. **Chinchilla Ratio:**
   $$\frac{D}{N} = \frac{20 \times 10^9}{1 \times 10^9} = \mathbf{20.0 \text{ tokens/parameter}} \quad (\text{Exact Chinchilla optimal ratio!})$$
3. **Wall-Clock Time:**
   - Total cluster throughput: $64 \text{ GPUs} \times 150 \times 10^{12} \text{ FLOP/s} = 9.6 \times 10^{15} \text{ FLOP/s} = 9.6 \text{ PFLOP/s}$.
   - Time in seconds:
     $$t_{\text{seconds}} = \frac{1.2 \times 10^{20}}{9.6 \times 10^{15}} = 12,500 \text{ seconds}$$
   - In hours:
     $$t_{\text{hours}} = \frac{12,500}{3600} \approx \mathbf{3.47 \text{ hours}} \quad \blacksquare$$

---

## 7. Deep Learning Connection & Modern Applications

- **DeepSeek-V3 & LLaMA-3 Pre-Training:**
  - LLaMA-3 8B was trained on **15 trillion tokens** ($D/N \approx 1,875$). This massive overtraining yielded a frontier-grade 8B model that rivals older 70B models, optimizing post-deployment serving economics.
- **MiniCPM & WSD Adoption:**
  - MiniCPM pioneered the systematic use of WSD schedules in open-weights models, demonstrating that the stable phase can be continuously fine-tuned and branched into specialized coding and vision models without performance degradation.
- **PyTorch FSDP (Fully Sharded Data Parallel):**
  - PyTorch's native implementation of ZeRO-3 is the de facto industry standard for distributed training, powering models like Megatron-LLaMA and Hugging Face Accelerate with zero code change to model forward architectures.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Chinchilla Scaling & Compute Calculator:**
   - Exact FLOP calculation ($C = 6ND$).
   - Token budget and duration estimators.
2. **Warmup-Stable-Decay (WSD) vs. Cosine Learning Rate Engine:**
   - Step-by-step verification matching the Part 5 hand trace ($t=50, 500, 900$).
3. **ZeRO-1, ZeRO-2, and ZeRO-3 / FSDP Memory Accounting:**
   - Mathematical validation of static memory per GPU across cluster size $P$.

See implementation in:
[`12_modern_llm_architectures/code/07_pretraining_and_scaling_laws.py`](./code/07_pretraining_and_scaling_laws.py)
