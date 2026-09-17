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

### 2.5 Deep Derivation 9.5.1: Rigorous Transformer Parameter Counting Formulas

#### Architectural Parameter Breakdown
For an autoregressive Transformer decoder with $L$ layers, hidden dimension $d_{\text{model}}$, vocabulary size $V$, and feedforward hidden dimension $d_{ff}$:

1. **Embedding Matrix:**
   $$\mathbf{W}_{\text{embed}} \in \mathbb{R}^{V \times d_{\text{model}}} \implies N_{\text{embed}} = V \cdot d_{\text{model}}$$
   *(If the LM head shares weights with embeddings via weight tying, this matrix is counted only once).*

2. **Self-Attention Sub-Layer (Per Layer):**
   - For standard MHA ($H$ query, key, value heads of size $d_k$ where $H \cdot d_k = d_{\text{model}}$):
     $$\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$
     $$N_{\text{attn}} = 4 d_{\text{model}}^2$$
   - For Grouped-Query Attention (GQA) with $H_{KV}$ key-value heads:
     $$N_{\text{attn}}^{\text{GQA}} = d_{\text{model}}^2 (\text{for } Q) + 2 d_{\text{model}} (H_{KV} d_k) (\text{for } K, V) + d_{\text{model}}^2 (\text{for } O) = 2 d_{\text{model}}^2 + 2 d_{\text{model}} (H_{KV} d_k)$$

3. **Feed-Forward Network (FFN / MLP Sub-Layer):**
   - **Standard 2-Layer MLP (GPT-2/3, BERT):**
     $$\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}, \quad \mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}} \quad (d_{ff} = 4 d_{\text{model}})$$
     $$N_{\text{MLP}} = 2 \cdot d_{\text{model}} \cdot d_{ff} = 8 d_{\text{model}}^2$$
   - **Gated SwiGLU / GeGLU (LLaMA, Mistral, PaLM):**
     SwiGLU uses three projections: Gate $\mathbf{W}_{\text{gate}}$, Up $\mathbf{W}_{\text{up}}$, and Down $\mathbf{W}_{\text{down}}$ with $d_{ff} \approx \frac{8}{3} d_{\text{model}}$:
     $$N_{\text{SwiGLU}} = 3 \cdot d_{\text{model}} \cdot d_{ff} \approx 3 \cdot d_{\text{model}} \left(\frac{8}{3} d_{\text{model}}\right) = 8 d_{\text{model}}^2$$

4. **Layer Normalization / RMSNorm (Per Layer):**
   Two RMSNorm instances per layer (pre-attention and pre-FFN), each with $d_{\text{model}}$ gain parameters:
   $$N_{\text{norm}} = 2 d_{\text{model}}$$

#### Total Non-Embedding Parameter Formula:
Summing across all $L$ layers:
$$N_{\text{non-embed}} = L \cdot \left( N_{\text{attn}} + N_{\text{FFN}} + N_{\text{norm}} \right)$$
For standard architectures where $N_{\text{attn}} = 4 d_{\text{model}}^2$ and $N_{\text{FFN}} = 8 d_{\text{model}}^2$:
$$N_{\text{non-embed}} = L \cdot \left( 4 d_{\text{model}}^2 + 8 d_{\text{model}}^2 + 2 d_{\text{model}} \right) \approx \mathbf{12 L d_{\text{model}}^2}$$
For LLaMA-style models where $d_{ff} \approx 2.6875 d_{\text{model}}$:
$$N_{\text{non-embed}} = L \cdot \left( 4 d_{\text{model}}^2 + 3 d_{\text{model}} d_{ff} \right) \approx L \cdot \left( 4 d_{\text{model}}^2 + 8.06 d_{\text{model}}^2 \right) \approx \mathbf{12.06 L d_{\text{model}}^2}$$
This fundamental rule of thumb allows engineers to instantly calculate parameter counts and FLOP budgets given $L$ and $d_{\text{model}}$. $\blacksquare$

---

### 2.6 Deep Derivation 9.5.2: The Kaplan vs. Chinchilla Scaling Discrepancy

#### The Historical Conflict
- **Kaplan et al. (OpenAI, 2020):** Found $N \propto C^{0.73}$ and $D \propto C^{0.27}$. This implied that for every doubling of compute, one should scale model capacity by $73\%$ and training tokens by only $27\%$.
- **Hoffmann et al. (Chinchilla, 2022):** Found $N \propto C^{0.50}$ and $D \propto C^{0.50}$.

#### Mathematical Cause of the Discrepancy: Learning Rate Schedule Coupling
Why were Kaplan's empirical exponents skewed toward parameters?

1. **The Cosine Decay Schedule:**
   In modern pre-training, learning rate decays following a cosine schedule:
   $$\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \frac{\pi t}{T_{\text{max}}} \right) \right)$$
   A fundamental property of cosine decay is that the learning rate must approach its minimum near the end of training for the loss to reach its true optimum.

2. **Kaplan's Flaw: Fixed Cosine Schedule Length:**
   Kaplan et al. trained a large model family using a fixed cosine schedule horizon ($T_{\text{max}} = 300\text{B}$ tokens) and evaluated checkpoints at early intermediate steps (e.g., at $t = 10\text{B}, 50\text{B}$ tokens).
   - At $t = 10\text{B}$, the learning rate had barely begun to decay ($\eta_{10B} \approx \eta_{\max}$).
   - The evaluated checkpoints had not benefited from the critical annealing phase.
   - A larger model with high learning rate underperforms its true capacity on few tokens, artificially deflating the apparent utility of training tokens!

3. **Chinchilla's Correction: Retuned Iso-FLOP Schedules:**
   Hoffmann et al. trained over 400 separate models where **each individual run had its own customized cosine decay schedule terminating exactly at that run's token budget $D$**.
   Evaluating models only at the end of their dedicated schedule revealed that tokens contribute equally to loss reduction:
   $$\alpha \approx 0.34, \quad \beta \approx 0.28 \implies a = \frac{\beta}{\alpha+\beta} \approx 0.45, \quad b = \frac{\alpha}{\alpha+\beta} \approx 0.55$$
   The symmetric $1:1$ allocation was restored. $\blacksquare$

---

### 2.7 Deep Derivation 9.5.3: Smooth Pre-training Power Laws and "Emergent" Abilities

#### The Paradox of Emergent Abilities
Many high-profile benchmarks (such as BIG-bench, GSM8K, and arithmetic) show sudden, non-linear jumps in accuracy: a model scores $0\%$ accuracy up to $50\text{B}$ parameters, and then leaps abruptly to $70\%$ accuracy at $100\text{B}$ parameters.
Does scaling create discontinuous, emergent phase transitions in intelligence?

#### Theorem: Emergent Abilities as a Metric Artifact (Schaeffer et al., NeurIPS 2023)
Let pre-training cross-entropy per token $L(C)$ follow a perfectly smooth, continuous power law with compute $C$:
$$L(C) = E + A C^{-\gamma}$$
Let the average per-token prediction error probability be $\epsilon(C) = 1 - e^{-L(C)} \approx L(C) - E \propto C^{-\gamma}$.

#### Proof:
1. **Multi-Token Exact Match Metric:**
   Consider a reasoning or arithmetic task requiring $k$ consecutive correct tokens (e.g., generating a 5-digit number or a multi-line proof).
   Under the independence approximation across target tokens:
   $$\text{Accuracy}_{\text{EM}}(C) = (1 - \epsilon(C))^k = \left( 1 - A_0 C^{-\gamma} \right)^k$$

2. **Taylor Series and Threshold Behavior for Large $k$:**
   Using the exponential limit $(1 - x/k)^k \approx e^{-x}$:
   $$\text{Accuracy}_{\text{EM}}(C) \approx \exp\left( -k A_0 C^{-\gamma} \right)$$
   Taking the second derivative w.r.t. $\log C$:
   $$\frac{d^2 \text{Accuracy}}{d(\log C)^2} \text{ exhibits a sharp inflection at } C_{\text{threshold}} = (k A_0)^{1/\gamma}$$
   - When $C < C_{\text{threshold}}$, $(1 - \epsilon)^k \approx 0.00$ (e.g., $0.80^5 \approx 0.32$, $0.70^{10} \approx 0.02$).
   - When $C > C_{\text{threshold}}$, accuracy curves upward exponentially toward $1.00$.

3. **Continuous Alternative Metrics:**
   If the exact-match 0/1 step-metric is replaced by a continuous metric (such as Brier score, edit distance, or token perplexity), the "emergence" vanishes completely, revealing a strictly smooth, predictable power-law improvement across all model scales. Intelligence scales smoothly; non-linear evaluation metrics create the illusion of emergence. $\blacksquare$

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
   $$D_{\text{FIM}} = \text{"[PRE] def add(a, b): [SUF] \textbackslash n\# End of function [MID] \textbackslash n    return a + b"}$$
3. When fine-tuned or pre-trained on this sequence, the model learns to fill in the code body $M$ conditioned simultaneously on the function header $P$ and trailing comments $S$.

---

### Illustration 3: Exact Chinchilla Optimal Allocation for a $10^{23}$ FLOP Compute Budget

**Problem:**
A laboratory has an allocated pre-training compute budget of $C = 1.0 \times 10^{23}$ FLOPs.
Using the empirical Chinchilla loss equation coefficients:
$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$
with $E = 1.69, A = 406.4, B = 410.7, \alpha = 0.34, \beta = 0.28$:
1. Determine the compute-optimal parameter count $N^*(C)$ and training token count $D^*(C)$.
2. Calculate the optimal loss $L^*$.
3. Compare against a sub-optimal, over-parameterized model with $N = 2 \times N^*$ (and $D = D^* / 2$), calculating the resulting loss penalty.

**Solution:**

#### Step 1: Analytical Allocation Formula

From the Lagrangian derivation in Section 2.4:
$$\frac{\alpha A}{\beta B} = \frac{(0.34)(406.4)}{(0.28)(410.7)} = \frac{138.176}{114.996} \approx 1.201572$$
$$\alpha + \beta = 0.34 + 0.28 = 0.62$$
$$G = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha + \beta}} = (1.201572)^{\frac{1}{0.62}} = (1.201572)^{1.612903} \approx \mathbf{1.343516}$$

The available parameter-token product budget is:
$$\frac{C}{6} = \frac{1.0 \times 10^{23}}{6} \approx 1.666667 \times 10^{22}$$

1. **Optimal Parameters $N^*$:**
   $$a = \frac{\beta}{\alpha + \beta} = \frac{0.28}{0.62} \approx 0.451613$$
   $$N^*(C) = G \cdot \left( \frac{C}{6} \right)^a = 1.343516 \times (1.666667 \times 10^{22})^{0.451613}$$
   $$(1.666667 \times 10^{22})^{0.451613} = (1.666667)^{0.451613} \times 10^{22 \times 0.451613} \approx 1.25925 \times 10^{9.935486}$$
   $$10^{9.935486} = 8.61958 \times 10^9 \implies 1.25925 \times 8.61958 \times 10^9 \approx 1.08542 \times 10^{10}$$
   $$N^* = 1.343516 \times 1.08542 \times 10^{10} = \mathbf{1.458 \times 10^{10} \text{ parameters (14.58 Billion)}}$$

2. **Optimal Tokens $D^*$:**
   $$b = \frac{\alpha}{\alpha + \beta} = \frac{0.34}{0.62} \approx 0.548387$$
   $$D^*(C) = G^{-1} \cdot \left( \frac{C}{6} \right)^b = \frac{1}{1.343516} \times (1.666667 \times 10^{22})^{0.548387}$$
   $$(1.666667 \times 10^{22})^{0.548387} \approx 1.32354 \times 10^{12.064514} \approx 1.32354 \times (1.16016 \times 10^{12}) \approx 1.53552 \times 10^{12}$$
   $$D^* = 0.744315 \times 1.53552 \times 10^{12} = \mathbf{1.143 \times 10^{12} \text{ tokens (1.143 Trillion)}}$$

*(Sanity check: $6 N^* D^* = 6 \times (1.458 \times 10^{10}) \times (1.143 \times 10^{12}) \approx 1.000 \times 10^{23}$ FLOPs)*.
Ratio: $\frac{D^*}{N^*} = \frac{1.143 \times 10^{12}}{1.458 \times 10^{10}} \approx \mathbf{78.4 \text{ tokens/param}}$.

---

#### Step 2: Optimal Loss Calculation

1. **Parameter Error:**
   $$(N^*)^\alpha = (1.458 \times 10^{10})^{0.34} = (1.458)^{0.34} \times 10^{3.4} = 1.1370 \times 2,511.89 \approx 2,856.02$$
   $$\frac{A}{(N^*)^\alpha} = \frac{406.4}{2,856.02} \approx \mathbf{0.1423}$$

2. **Data Error:**
   $$(D^*)^\beta = (1.143 \times 10^{12})^{0.28} = (1.143)^{0.28} \times 10^{3.36} = 1.0379 \times 2,290.87 \approx 2,377.70$$
   $$\frac{B}{(D^*)^\beta} = \frac{410.7}{2,377.70} \approx \mathbf{0.1727}$$

3. **Total Optimal Loss:**
   $$L^* = 1.6900 + 0.1423 + 0.1727 = \mathbf{2.0050 \text{ nats}}$$

---

#### Step 3: Comparison with Over-Parameterized Model ($N = 2 N^*, D = D^* / 2$)

Let $N_{\text{sub}} = 2.916 \times 10^{10}$ ($29.16\text{B}$) and $D_{\text{sub}} = 5.715 \times 10^{11}$ ($571.5\text{B}$):
- New parameter term: $\frac{0.1423}{2^{0.34}} = \frac{0.1423}{1.2658} \approx 0.1124$
- New data term: $0.1727 \times 2^{0.28} = 0.1727 \times 1.2142 \approx 0.2097$
- Suboptimal loss:
  $$L_{\text{sub}} = 1.6900 + 0.1124 + 0.2097 = \mathbf{2.0121 \text{ nats}}$$
- Loss Penalty: $\Delta L = +0.0071$ nats.
While $0.0071$ nats appears small numerically, in perplexity ($\text{PPL} = e^L$):
$$\text{PPL}^* = e^{2.0050} \approx 7.426, \quad \text{PPL}_{\text{sub}} = e^{2.0121} \approx 7.479$$
The sub-optimal model degrades perplexity despite using an identical budget of GPU hours.

---

### Illustration 4: Detailed Parameter and FLOP Breakdown of a LLaMA-Style 7B Architecture

**Problem:**
A modern autoregressive LLM (similar to LLaMA-1/2 7B) specifies:
- Layers: $L = 32$
- Hidden dimension: $d_{\text{model}} = 4,096$
- Heads: $H = 32$ ($d_k = 128$)
- FFN type: SwiGLU with hidden size $d_{ff} = 11,008$
- Vocabulary: $V = 32,000$ (un-tied LM head)

1. Compute the exact parameter count for:
   - Word embeddings and LM head.
   - Attention projections ($Q, K, V, O$).
   - SwiGLU FFN projections ($\text{gate}, \text{up}, \text{down}$).
   - Total model parameters $N$.
2. Calculate the theoretical forward and backward FLOPs per token and verify the $6N$ rule.

**Solution:**

#### Step 1: Parameter Inventory

1. **Embedding & Head:**
   $$\text{Token Embeddings} = V \cdot d_{\text{model}} = 32,000 \times 4,096 = \mathbf{131,072,000}$$
   $$\text{Output LM Head} = V \cdot d_{\text{model}} = 32,000 \times 4,096 = \mathbf{131,072,000}$$

2. **Per-Layer Attention Sub-layer:**
   $$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O \in \mathbb{R}^{4096 \times 4096}$$
   $$\text{Params}_{\text{attn}} = 4 \times 4,096 \times 4,096 = 4 \times 16,777,216 = \mathbf{67,108,864}$$

3. **Per-Layer SwiGLU FFN Sub-layer:**
   Three matrices of shape $(4096 \times 11008)$:
   $$\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{4096 \times 11008}, \quad \mathbf{W}_{\text{down}} \in \mathbb{R}^{11008 \times 4096}$$
   $$\text{Params}_{\text{FFN}} = 3 \times 4,096 \times 11,008 = \mathbf{135,266,304}$$

4. **Per-Layer Normalization:**
   Two RMSNorm instances per layer: $2 \times 4,096 = \mathbf{8,192}$.
   Final layer norm: $\mathbf{4,096}$.

5. **Layer Total (across all 32 layers):**
   $$\text{Params}_{\text{per\_layer}} = 67,108,864 + 135,266,304 + 8,192 = 202,383,360$$
   $$\text{Total 32 Layers} = 32 \times 202,383,360 = \mathbf{6,476,267,520}$$

6. **Grand Total Parameters $N$:**
   $$N = 6,476,267,520 + 131,072,000 (\text{Embed}) + 131,072,000 (\text{Head}) + 4,096 (\text{Final Norm})$$
   $$N = \mathbf{6,738,415,616 \text{ parameters } (\approx 6.74 \text{ Billion})}$$

---

#### Step 2: FLOPs Calculation per Token

For non-embedding matrix multiplications ($P = 6.476 \times 10^9$ params):
- **Forward Pass:**
  $$\text{FLOPs}_{\text{fwd}} = 2 \times P = 2 \times 6.476 \times 10^9 = \mathbf{12.95 \times 10^9 \text{ FLOPs}}$$
- **Backward Pass:**
  $$\text{FLOPs}_{\text{bwd}} = 4 \times P = 4 \times 6.476 \times 10^9 = \mathbf{25.90 \times 10^9 \text{ FLOPs}}$$
- **Total per Token:**
  $$\text{FLOPs}_{\text{total}} = 6 \times P = 6 \times 6.476 \times 10^9 \approx \mathbf{38.86 \times 10^9 \text{ FLOPs}}$$
  This precisely confirms the theoretical rule: $C = 6ND$ operations per token during pre-training.

---

### Illustration 5: Cluster Sizing, Wall-Clock Time, and Energy Consumption for 70B Model Training

**Problem:**
An enterprise trains a 70B parameter model on $D = 15.0 \text{ Trillion tokens}$ on an AI supercomputer comprising $2,048$ NVIDIA H100 SXM GPUs.
Specifications:
- Each H100 provides $989 \text{ TFLOPs}$ dense BF16 tensor core throughput.
- Observed Model FLOPs Utilization: $\text{MFU} = 46\%$.
- GPU power draw: $700\text{ W}$ per GPU; Data center Power Usage Effectiveness: $\text{PUE} = 1.20$.

Calculate:
1. Total pre-training compute budget $C$ in FLOPs.
2. Effective cluster throughput in ExaFLOPs/sec.
3. Training duration in days.
4. Total electricity consumption in Megawatt-hours (MWh).

**Solution:**

#### Step 1: Total Compute Budget $C$
$$C = 6 N D = 6 \times (70 \times 10^9) \times (15.0 \times 10^{12}) = \mathbf{6.30 \times 10^{24} \text{ FLOPs (6.30 YottaFLOPs)}}$$

---

#### Step 2: Cluster Effective Compute Throughput
$$\text{Cluster Peak} = 2,048 \text{ GPUs} \times 989 \times 10^{12} \text{ FLOPs/s} = 2.02547 \times 10^{18} \text{ FLOPs/s} = \mathbf{2.025 \text{ ExaFLOPs/s}}$$
At $\text{MFU} = 46\%$:
$$\text{Throughput}_{\text{eff}} = 0.46 \times 2.02547 \times 10^{18} = \mathbf{9.317 \times 10^{17} \text{ FLOPs/s (931.7 PFLOPs/s)}}$$

---

#### Step 3: Wall-Clock Duration
$$\text{Time (seconds)} = \frac{C}{\text{Throughput}_{\text{eff}}} = \frac{6.30 \times 10^{24}}{9.317 \times 10^{17}} \approx 6,761,833 \text{ seconds}$$
$$\text{Time (days)} = \frac{6,761,833}{86,400} \approx \mathbf{78.26 \text{ days (approx 2.6 months)}}$$

---

#### Step 4: Electricity Consumption
1. **Total Facility Power Draw:**
   $$\text{Power}_{\text{GPUs}} = 2,048 \times 700\text{ W} = 1,433,600\text{ W} = 1.4336\text{ MW}$$
   Including host CPUs, storage, networking, and cooling ($\text{PUE} = 1.20$):
   $$\text{Power}_{\text{total}} = 1.4336 \times 1.20 = \mathbf{1.72032 \text{ MW}}$$

2. **Total Energy Consumed:**
   $$\text{Hours} = 78.26 \times 24 = 1,878.24 \text{ hours}$$
   $$\text{Energy} = 1.72032 \text{ MW} \times 1,878.24 \text{ hours} \approx \mathbf{3,231.18 \text{ MWh (3.23 GWh)}}$$

This provides a complete end-to-end mathematical model connecting algorithmic scaling theory to physical data center infrastructure.

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
