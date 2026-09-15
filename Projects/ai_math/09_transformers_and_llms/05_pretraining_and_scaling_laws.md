# Chapter 9.5: Pre-training Objectives & Compute-Optimal Scaling Laws (Chinchilla)

---

## 1. Intuition & 101 Motivation

Prior to 2020, training deep neural networks was widely viewed as an empirical art—dominated by trial-and-error hyperparameter tuning, mysterious loss spikes, and architectural tinkering.

In 2020 and 2022, two landmark papers transformed deep learning from an empirical craft into a **predictive quantitative science**:
1. **Kaplan et al. (OpenAI, 2020):** *"Scaling Laws for Neural Language Models"* established that the test loss of autoregressive Transformers follows smooth, monotonic **power laws** across many orders of magnitude of compute, data, and parameters.
2. **Hoffmann et al. (DeepMind, 2022):** *"Training Compute-Optimal Large Language Models"* (The **Chinchilla** paper) corrected Kaplan's scaling exponents, proving that contemporary models (including GPT-3 and Gopher) were severely **undertrained** on data relative to their size.

```
                  THE COMPUTE-OPTIMAL ALLOCATION PARADIGM
      ┌────────────────────────────────────────────────────────┐
      │  Total Training Compute Budget: C ≈ 6 * N * D FLOPs    │
      └───────────────────────────┬────────────────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [ Kaplan et al. (2020) ]          [ Chinchilla (2022) ]
      • N ∝ C^0.73                      • N ∝ C^0.50
      • D ∝ C^0.27                      • D ∝ C^0.50
      • Scale parameters 3x             • Scale parameters and tokens
        faster than tokens                EQUALLY in 1:1 proportion!
      • Led to: GPT-3 (175B / 300B)     • Led to: Chinchilla (70B / 1.4T)
        (Severely Undertrained)           (Crushed 280B Gopher at 4x lower cost!)
```

### The Central Insight:
Given a fixed budget of compute FLOPs, doubling model size requires doubling the training tokens. A smaller model trained on vastly more high-quality data achieves superior perplexity and costs significantly less to serve in production.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Pre-training Objectives

#### 1. Autoregressive Causal Language Modeling (CLM)
The dominant objective for modern foundation models (GPT, LLaMA, Mistral). Given text tokens $\mathbf{x} = (x_1, \dots, x_T)$:
$$\mathcal{L}_{\text{CLM}}(\boldsymbol{\theta}) = - \frac{1}{T} \sum_{t=1}^T \log P_{\boldsymbol{\theta}}(x_t \mid x_1, \dots, x_{t-1})$$

#### 2. Fill-in-the-Middle (FIM) Infilling (Bavarian et al., 2022)
Standard autoregressive models can only generate text forward. For code completion and document editing, the model must condition on both prefix and suffix to synthesize the middle.

**FIM Transformation Algorithm:**
1. A document $D$ is split into three contiguous segments: $\text{Prefix } P$, $\text{Middle } M$, and $\text{Suffix } S$.
2. With probability $p_{\text{FIM}} \approx 0.5$, the document is rearranged using special sentinel tokens into:
   $$D_{\text{FIM}} = \langle \text{PRE} \rangle \circ P \circ \langle \text{SUF} \rangle \circ S \circ \langle \text{MID} \rangle \circ M \circ \langle \text{EOT} \rangle$$
3. The model is trained autoregressively on $D_{\text{FIM}}$ with standard next-token prediction! Because $M$ appears at the end, predicting $M$ is conditioned jointly on $P$ and $S$.

---

### 2.2 Derivation of the $6ND$ Training FLOPs Rule

Why does pre-training an $N$-parameter Transformer on $D$ tokens require approximately $6ND$ floating-point operations?

Let $N$ denote the non-embedding parameter count.
A standard matrix multiplication $\mathbf{y} = \mathbf{W}\mathbf{x}$ where $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ has $P = d_{\text{out}} \cdot d_{\text{in}}$ parameters:
- Computing $\mathbf{y}$ takes $d_{\text{out}} \cdot d_{\text{in}}$ multiplications and $d_{\text{out}} \cdot d_{\text{in}}$ additions:
  $$\text{FLOPs}_{\text{forward}} = 2 P \text{ operations per token}$$

During the backward pass:
1. Computing gradient w.r.t. input: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{W}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \implies 2 P \text{ FLOPs}$.
2. Computing gradient w.r.t. weights: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T \implies 2 P \text{ FLOPs}$.
   $$\text{FLOPs}_{\text{backward}} = 2P + 2P = 4 P \text{ operations per token}$$

Summing forward and backward passes:
$$\text{FLOPs}_{\text{total}} = \text{FLOPs}_{\text{forward}} + \text{FLOPs}_{\text{backward}} = 2P + 4P = 6 P \text{ operations per token}$$

Accumulating across all $N$ parameters and $D$ dataset tokens:
$$C \approx 6 N D \text{ FLOPs}$$

*(Note: If activation checkpointing is used to trade compute for memory, the forward pass is recomputed once during backward, yielding $C \approx 8ND$).*

---

### 2.3 The Chinchilla Parametric Loss Formulation

Jordan Hoffmann et al. modeled the cross-entropy loss $L(N, D)$ as a joint power law with an irreducible loss baseline:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where:
- $E \approx 1.69$ is the **irreducible loss** (the inherent Shannon entropy of natural language).
- $\frac{A}{N^\alpha}$ represents the parameter capacity error ($A \approx 406.4, \alpha \approx 0.34$).
- $\frac{B}{D^\beta}$ represents the finite data estimation error ($B \approx 410.7, \beta \approx 0.28$).
- $N$ is the number of model parameters, and $D$ is the number of training tokens.

---

### 2.4 Derivation of the Optimal Compute Allocation (Lagrange Multipliers)

We formulate the compute-optimal training problem as constrained minimization:
$$\min_{N, D} L(N, D) = E + A N^{-\alpha} + B D^{-\beta} \quad \text{subject to } 6ND = C$$

Construct the Lagrangian:
$$\mathcal{L}(N, D, \lambda) = E + A N^{-\alpha} + B D^{-\beta} + \lambda (6ND - C)$$

Compute first-order necessary conditions:
$$\frac{\partial \mathcal{L}}{\partial N} = -\alpha A N^{-\alpha - 1} + 6 \lambda D = 0 \implies 6 \lambda N D = \alpha A N^{-\alpha}$$
$$\frac{\partial \mathcal{L}}{\partial D} = -\beta B D^{-\beta - 1} + 6 \lambda N = 0 \implies 6 \lambda N D = \beta B D^{-\beta}$$

Equating the two expressions:
$$\alpha A N^{-\alpha} = \beta B D^{-\beta}$$

Taking logarithms and solving for $N$ in terms of $D$:
$$N^{-\alpha} = \frac{\beta B}{\alpha A} D^{-\beta} \implies N = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha}} D^{\frac{\beta}{\alpha}}$$

Now, substitute $N$ into the compute constraint $C = 6ND$:
$$C = 6 \left[ \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha}} D^{\frac{\beta}{\alpha}} \right] D = 6 \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha}} D^{\frac{\alpha + \beta}{\alpha}}$$

Solving for optimal tokens $D^*(C)$ and optimal parameters $N^*(C)$:
$$D^*(C) = G^{-1} \left( \frac{C}{6} \right)^{\frac{\alpha}{\alpha + \beta}}$$
$$N^*(C) = G \left( \frac{C}{6} \right)^{\frac{\beta}{\alpha + \beta}}$$
where $G = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha + \beta}}$.

#### Evaluating the Exponents:
Substituting the empirical Chinchilla coefficients ($\alpha \approx 0.34, \beta \approx 0.28$):
$$a = \frac{\beta}{\alpha + \beta} = \frac{0.28}{0.34 + 0.28} = \frac{0.28}{0.62} \approx \mathbf{0.45}$$
$$b = \frac{\alpha}{\alpha + \beta} = \frac{0.34}{0.34 + 0.28} = \frac{0.34}{0.62} \approx \mathbf{0.55}$$

**The Fundamental Law:**
$$N^*(C) \propto C^{0.45} \approx C^{0.5}, \quad D^*(C) \propto C^{0.55} \approx C^{0.5}$$
Parameters and training tokens should scale in **equal proportions ($\approx 1:1$)** as compute budget grows.

Evaluating the proportionality constant:
$$D^* \approx 20 \times N^*$$
A compute-optimal model requires approximately **20 tokens per parameter**.

---

## 3. Geometric & Algebraic Interpretation

### Iso-FLOP Curves on the Loss Surface

```
      Log(Parameters N)
             ▲
             │      Iso-FLOP Curves (6ND = C)
             │        \         \
             │         \         \      * Optimal Allocation Path (Slope ~ 1.0)
             │          \    *    \    /
             │           \  /      \  /
             │            \/________\/
             │            /\        /\
             │           /  \      /  \
             └──────────/────\────/────\─────────► Log(Tokens D)
                       C_1        C_2
```

In the $(\log N, \log D)$ coordinate plane:
- The compute constraint $\log N + \log D = \log(C/6)$ forms straight lines of slope $-1$ (Iso-FLOP lines).
- The loss surface $L(N, D)$ forms concentric convex elliptical level contours.
- The compute-optimal point for any budget $C$ is the **tangency point** between the Iso-FLOP line and the lowest reachable loss contour.
- Connecting all tangency points across varying budgets traces the **Optimal Scaling Trajectory**, which forms a straight ray of slope $\approx 1.0$.

---

## 4. Real-World Analogy

### The Star Student vs. The Library of Books
- **Kaplan Paradigm (The Undertrained Prodigy):**
  A research university recruits a genius with a massive photographic brain ($N = 175\text{B}$ parameters), but only purchases 100 books for their library ($D = 300\text{B}$ tokens). The genius reads all 100 books once. Despite their brilliant capacity, their actual knowledge is severely limited because they simply ran out of reading material.
- **Chinchilla Paradigm (The Balanced Scholar):**
  Instead of an oversized brain, the university hires a scholar with a well-proportioned brain ($N = 70\text{B}$ parameters), and stocks the library with 1,000 diverse books ($D = 1.4\text{T}$ tokens).
  The balanced scholar reads 10 times more material, easily outperforms the undertrained genius on every exam, and costs $4\times$ less to maintain and consult!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us verify the Chinchilla loss equation and optimal allocation with concrete hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Scaling Laws |
| :--- | :--- | :--- | :--- |
| $C$ | Total Compute Budget | Scalar ($6 \times 10^{18}$ FLOPs) | Total floating-point operations allocated |
| $N \cdot D$ | Parameter-Token Product | Scalar ($10^{18}$) | $C / 6$ |
| $E$ | Irreducible Loss | Scalar ($1.50$) | Minimum theoretical entropy of language |
| $A, B$ | Scaling Coefficients | Scalar ($400.0, 400.0$) | Scale factors for parameter and data errors |
| $\alpha, \beta$ | Scaling Exponents | Scalar ($0.30, 0.30$) | Rate of power-law error reduction |
| $L(N, D)$ | Expected Test Loss | Scalar | $E + A N^{-\alpha} + B D^{-\beta}$ |

---

### 5.2 Concrete Toy Budget

Let:
$$E = 1.5, \quad A = 400.0, \quad B = 400.0, \quad \alpha = 0.3, \quad \beta = 0.3$$
$$C / 6 = N \cdot D = 10^{18}$$

Because $\alpha = \beta$ and $A = B$, by symmetry the global mathematical optimum is:
$$N^* = D^* = \sqrt{10^{18}} = 10^9 \quad (1\text{ Billion parameters, } 1\text{ Billion tokens})$$

Let us compare three candidate allocations under the exact same compute budget:

---

### 5.3 Candidate 1: Over-Parametric (Kaplan-Style: $N = 10^{10}, D = 10^8$)

1. **Parameter Term $\frac{A}{N^\alpha}$:**
   $$N^\alpha = (10^{10})^{0.3} = 10^{3.0} = 1,000$$
   $$\frac{A}{N^\alpha} = \frac{400.0}{1,000} = \mathbf{0.4000}$$

2. **Data Term $\frac{B}{D^\beta}$:**
   $$D^\beta = (10^8)^{0.3} = 10^{2.4} \approx 251.1886$$
   $$\frac{B}{D^\beta} = \frac{400.0}{251.1886} \approx \mathbf{1.5924}$$

3. **Total Loss $L_1$:**
   $$L_1 = 1.50 + 0.4000 + 1.5924 = \mathbf{3.4924}$$

Notice how severely data-starved this model is: the data error ($1.5924$) is almost $4\times$ larger than the parameter error ($0.4000$)!

---

### 5.4 Candidate 2: Under-Parametric (Data-Heavy: $N = 10^8, D = 10^{10}$)

1. **Parameter Term $\frac{A}{N^\alpha}$:**
   $$N^\alpha = (10^8)^{0.3} = 10^{2.4} \approx 251.1886 \implies \frac{400.0}{251.1886} \approx \mathbf{1.5924}$$

2. **Data Term $\frac{B}{D^\beta}$:**
   $$D^\beta = (10^{10})^{0.3} = 10^{3.0} = 1,000 \implies \frac{400.0}{1,000} = \mathbf{0.4000}$$

3. **Total Loss $L_2$:**
   $$L_2 = 1.50 + 1.5924 + 0.4000 = \mathbf{3.4924}$$

---

### 5.5 Candidate 3: Compute-Optimal (Chinchilla-Style: $N = 10^9, D = 10^9$)

1. **Parameter Term $\frac{A}{N^\alpha}$:**
   $$N^\alpha = (10^9)^{0.3} = 10^{2.7} \approx 501.1872$$
   $$\frac{A}{N^\alpha} = \frac{400.0}{501.1872} \approx \mathbf{0.7981}$$

2. **Data Term $\frac{B}{D^\beta}$:**
   $$D^\beta = (10^9)^{0.3} = 10^{2.7} \approx 501.1872$$
   $$\frac{B}{D^\beta} = \frac{400.0}{501.1872} \approx \mathbf{0.7981}$$

3. **Total Loss $L_3$:**
   $$L_3 = 1.50 + 0.7981 + 0.7981 = \mathbf{3.0962}$$

#### Comparison Table:
| Configuration | Parameters $N$ | Tokens $D$ | Compute $C$ | Loss $L$ | Performance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Over-Parametric** | $10\text{B}$ | $100\text{M}$ | $6 \times 10^{18}$ | $3.4924$ | Suboptimal (Data-Starved) |
| **2. Under-Parametric** | $100\text{M}$ | $10\text{B}$ | $6 \times 10^{18}$ | $3.4924$ | Suboptimal (Capacity-Bound) |
| **3. Chinchilla Optimal** | **$1\text{B}$** | **$1\text{B}$** | **$6 \times 10^{18}$** | **$3.0962$** | **Global Optimum ($\Delta L = -0.3962$)** |

Under the exact same compute budget, the balanced allocation achieves a massive loss reduction of nearly **$0.40$ nats**, proving the Chinchilla theorem with hand arithmetic!

---

## 6. Solved Illustrations

### Illustration 1: Compute-Optimal vs. Inference-Optimal Training (The LLaMA Philosophy)

**Problem:**
Chinchilla dictates that compute-optimal training requires $D \approx 20 N$.
Meta trained **LLaMA-1 7B** and **LLaMA-3 8B** on $1.0\text{T}$ to $15.0\text{T}$ tokens:
$$\frac{D}{N} = \frac{15 \times 10^{12}}{8 \times 10^9} \approx \mathbf{1,875 \text{ tokens per parameter}}$$
Why did Meta deliberately violate Chinchilla scaling by almost **$100\times$**?

**Solution:**
Chinchilla optimizes **training compute efficiency**: achieving the lowest pre-training loss for a given budget of training GPU hours.
However, in production deployments:
- A foundation model is trained **once**, but served **billions of times** across millions of users.
- The cost of serving an 8B model is **$9\times$ cheaper** in latency, memory, and energy than serving a 70B model.
- By "over-training" an 8B model on 15T tokens, Meta absorbed higher one-time pre-training costs to create a tiny model that matches the performance of a 70B model while remaining cheap and fast to serve indefinitely.
- This is termed **Inference-Optimal Scaling**.

---

### Illustration 2: Fill-in-the-Middle (FIM) Data Transformation

**Problem:**
Given document $D = [\text{"def add(a, b):"}, \text{"\n    return a + b"}, \text{"\n# End of function"}]$, trace the FIM transformation using sentinels `[PRE]`, `[SUF]`, `[MID]`.

**Solution:**
1. Split into segments:
   - $P = \text{"def add(a, b):"}$
   - $M = \text{"\n    return a + b"}$
   - $S = \text{"\n# End of function"}$
2. Reorder with sentinels:
   $$D_{\text{FIM}} = \text{"[PRE] def add(a, b): [SUF] \n# End of function [MID] \n    return a + b"}$$
3. When fine-tuned or pre-trained on this sequence, the model learns to fill in the code body $M$ conditioned simultaneously on the function header $P$ and trailing comments $S$.

---

## 7. Deep Learning Connection & Application

### Real-World Training Cluster Engineering
When training frontier models across clusters of 16,384 NVIDIA H100 GPUs:
- **Model FLOPs Utilization (MFU):** The ratio of observed training throughput to peak theoretical GPU FLOPs:
  $$\text{MFU} = \frac{6 \times N \times \text{Tokens/sec}}{\text{Num GPUs} \times \text{Peak Theoretical FLOPs}}$$
- Well-engineered distributed frameworks (Megatron-LM, DeepSpeed) achieve $\text{MFU} \approx 45\% - 55\%$.
- A 70B model trained on 15T tokens requires:
  $$C \approx 6 \times (70 \times 10^9) \times (15 \times 10^{12}) = 6.3 \times 10^{24} \text{ FLOPs} \approx 6.3 \times 10^6 \text{ YottaFLOPs}$$
  Running on 1,024 H100s at $50\%$ MFU takes approximately **$73$ continuous days**.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `ChinchillaScalingLaw`: Mathematical solver computing optimal $N^*(C)$ and $D^*(C)$ using Lagrange multipliers.
2. Verification of the Part 5 Visual Grid hand arithmetic.
3. Numerical Iso-FLOP curve simulation finding the minimum loss point.
4. `FillInTheMiddleTransform`: Production-grade FIM document preprocessing pipeline with unit tests.

See implementation in:
[`09_transformers_and_llms/code/05_scaling_laws_and_chinchilla.py`](./code/05_scaling_laws_and_chinchilla.py)
