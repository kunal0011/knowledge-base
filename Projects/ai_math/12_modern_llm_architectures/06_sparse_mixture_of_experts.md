# Chapter 06: Sparse Mixture of Experts (MoE): Top-K Routing, Auxiliary Losses & DeepSeek-V3 MoE / MLA

---

## 1. Intuition & 101 Motivation

In traditional **dense** transformer architectures (like GPT-3, LLaMA 1/2, or Gemma), every single parameter in the network is activated for every single token:
$$\text{Compute Cost per Token} \propto N_{\text{total}}$$
As models grow to 70B, 405B, or 1T parameters, training and inference latency scale linearly with parameter count, requiring thousands of GPUs just to generate text.

Yet intuitively, not all knowledge is needed for every token:
- A token in a Python script does not need the parameters encoding 18th-century French literature.
- A token in a medical diagnosis does not need the parameters encoding quantum chromodynamics.

Enter **Sparse Mixture of Experts (MoE)** (Shazeer et al., 2017; Mixtral 8x7B, 2024; DeepSeek-V3, 2025):
- We replace the monolithic feed-forward network (FFN) with an ensemble of $E$ independent, parallel **Expert Networks** $\{E_1, E_2, \dots, E_E\}$.
- A lightweight neural **Router (Gating Network)** routes each token to only a sparse **Top-$K$** subset of experts (e.g., $K = 2$ out of $E = 8$ in Mixtral, or $K = 8$ out of $E = 256$ in DeepSeek-V3).
- **The Result:** Total parameter capacity expands massively (e.g. 671 billion parameters), but inference compute per token remains that of a compact model (e.g. 37 billion active parameters)!

In late 2024 and early 2025, **DeepSeek-V3** set a new global benchmark by introducing **DeepSeekMoE** with **Fine-Grained Expert Segmentation** and **Dedicated Shared Experts**, proving that MoE models can match the world's best dense models at a fraction of training and inference costs.

```
Token x -----------------------------------> Shared Expert E_shared (Always Active) ---\
   |                                                                                     \
   +-----> [ Router Network ] ---> Top-K Softmax Gating                                   +---> Output y
                |                                                                        /
                +---> Expert 1 (Weight g_1 = 0.27) -------------------------------------/
                +---> Expert 3 (Weight g_3 = 0.73) ------------------------------------/
                +---> Experts 2, 4..E (Inactive: Zero Compute!)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Classic Top-$K$ Sparse Gating Mechanism

Let the input token representation be $x \in \mathbb{R}^d$.
The MoE layer consists of:
1. $E$ expert feed-forward networks: $\operatorname{Expert}_i(x): \mathbb{R}^d \to \mathbb{R}^d$ for $i \in \{1, \dots, E\}$.
2. A gating router parameterized by matrix $W_g \in \mathbb{R}^{d \times E}$.

#### Step 1: Compute Raw Router Logits
$$H(x) = x W_g \in \mathbb{R}^E$$

#### Step 2: Keep Top-$K$ Elements
Select the $K$ experts with the highest logit values and mask all other experts to $-\infty$:
$$\operatorname{KeepTopK}(H(x), K)_i = \begin{cases}
H(x)_i, & \text{if } H(x)_i \text{ is in the top } K \text{ values of } H(x) \\
-\infty, & \text{otherwise}
\end{cases}$$

#### Step 3: Compute Gating Weights via Softmax
$$G(x) = \operatorname{softmax}\big( \operatorname{KeepTopK}(H(x), K) \big) \in \mathbb{R}^E$$
Notice that for all non-selected experts $j \notin \operatorname{TopK}$, $G(x)_j \equiv 0$.

#### Step 4: Weighted Combination
The output of the MoE layer is the dynamically weighted sum over the $K$ active experts:
$$y = \sum_{i \in \operatorname{TopK}} G(x)_i \cdot \operatorname{Expert}_i(x)$$

---

### 2.2 The Routing Collapse Problem & Auxiliary Load Balancing Loss

A notorious pathological failure mode in naive MoE training is **Routing Collapse (Winner-Takes-All Dynamic)**:
- Early in training, due to random initialization, 1 or 2 experts slightly outperform the others.
- The router preferentially directs more tokens to these favored experts.
- Because these favored experts receive all the tokens, they receive all the gradient updates, improving further.
- The remaining $E - 2$ experts receive zero tokens, zero gradients, and permanently starve!

#### The Switch Transformer Auxiliary Load Balancing Loss (Fedus et al., 2022):
To enforce balanced utilization across all $E$ experts, modern MoE models add an auxiliary loss to the pre-training objective.

For a batch of $T$ tokens:
1. **Fraction of tokens dispatched to Expert $i$ ($f_i$):**
   $$f_i = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\big( \text{expert } i \in \operatorname{TopK}(x_t) \big)$$
2. **Average router probability assigned to Expert $i$ ($P_i$):**
   $$P_i = \frac{1}{T} \sum_{t=1}^T \operatorname{softmax}(x_t W_g)_i$$
3. **The Auxiliary Loss:**
   $$\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{i=1}^E f_i P_i$$
   where $\alpha \approx 10^{-2}$ is a balancing hyperparameter.

#### Mathematical Property:
The Cauchy-Schwarz inequality guarantees that $\sum_{i=1}^E f_i P_i$ is minimized if and only if both distributions are uniform:
$$f_i = \frac{1}{E}, \quad P_i = \frac{1}{E} \quad \forall i \in \{1, \dots, E\}$$
Any deviation toward routing collapse creates a severe quadratic penalty!

---

### 2.3 The DeepSeekMoE Architecture (DeepSeek-V2 & DeepSeek-V3)

DeepSeek identified two critical flaws in traditional MoE architectures (like Mixtral):
1. **Coarse-Grained Experts:** Having only 8 large experts limits the number of possible expert combinations ($\binom{8}{2} = 28$ combinations).
2. **Knowledge Redundancy:** Multiple experts end up learning redundant basic English grammar, punctuation, and common facts.

DeepSeek introduced two foundational structural innovations:

#### 1. Fine-Grained Expert Segmentation:
Instead of 8 large experts, DeepSeek divides the FFN intermediate dimension into $E = 256$ smaller experts, routing to $K = 8$ active experts per token.
The number of possible expert combinations surges from $28$ to $\binom{256}{8} \approx 4.38 \times 10^{14}$, creating vastly richer combinatoric specialization!

#### 2. Dedicated Shared Experts:
DeepSeek isolates $N_s$ dedicated **Shared Experts** that are **always active for every single token** with fixed gating weight $1.0$:

$$y = \sum_{j=1}^{N_s} \operatorname{FFN}_{\text{shared}, j}(x) + \sum_{i \in \operatorname{TopK}_r} G(x)_i \cdot \operatorname{FFN}_{\text{routed}, i}(x)$$

- **Shared Experts:** Capture universal, foundational syntax, logic, and common knowledge.
- **Routed Experts:** Freed from basic syntax, routed experts specialize deeply into narrow, specialized domains (e.g. quantum computing, legal contracts, Python syntax).

---

## 3. Geometric & Physical Interpretation

### 3.1 Voronoi Cell Partitioning of Feature Space
The router weight matrix $W_g \in \mathbb{R}^{d \times E}$ can be viewed as $E$ anchor vectors in $d$-dimensional activation space:
```
Activation Space (R^d)
               Expert 1 Domain               Expert 2 Domain
              /                 \           /                \
             /                   \         /                  \
            |          *          |-------|         *          |
            |       Anchor 1      |       |      Anchor 2      |
             \                   /         \                  /
              \                 /           \                /
               -----------------             ----------------
                                       ^
                                       |
                           Boundary Hyperplane: w_1^T x = w_2^T x
```
The router partitions $\mathbb{R}^d$ into polyhedral **Voronoi cells**. When a token enters cell $i$, Expert $i$ fires. Fine-grained segmentation carves the activation space into hundreds of micro-cells, enabling surgical non-linear feature transformation.

---

## 4. Real-World Analogy: The Multi-Specialty Hospital

Consider a patient seeking healthcare:
- **Dense Model:** Every patient must see all 100 doctors in the hospital (Cardiologist, Neurosurgeon, Dermatologist, Pediatrician...). It is enormously expensive, slow, and wasteful.
- **Classic MoE (Mixtral):** A triage nurse directs the patient to 2 specialists ($K = 2$). But if the patient just has a minor headache, specialists waste time checking basic vital signs.
- **DeepSeekMoE:** Every patient first sees a **General Triage Physician (Shared Expert)** who takes blood pressure, temperature, and medical history. Then, and only then, does the triage physician page the 2 specialized surgeons (Routed Experts) for the specific diagnosis!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical forward pass of **Sparse MoE Top-$K$ Routing ($K = 2, E = 4$) with a Shared Expert** by hand.

### 5.1 Hand-Calculation Walkthrough 1: Top-2 Routing with Shared Expert

**System & Parameter Setup**
- Input token: $x = [2.0, 1.0]^T \in \mathbb{R}^2$ ($d = 2$)
- Total routed experts: $E = 4$
- Active routed experts per token: $K = 2$
- Router weight matrix $W_g \in \mathbb{R}^{2 \times 4}$:
  $$W_g = \begin{bmatrix} 1.0 & -1.0 & 2.0 & 0.0 \\ 0.0 & 2.0 & -1.0 & 1.0 \end{bmatrix}$$
- Expert Network outputs on input $x$:
  - $\operatorname{Expert}_1(x) = [1.0, 4.0]^T$
  - $\operatorname{Expert}_2(x) = [0.0, 1.0]^T$
  - $\operatorname{Expert}_3(x) = [3.0, -2.0]^T$
  - $\operatorname{Expert}_4(x) = [2.0, 2.0]^T$
- Shared Expert output on input $x$:
  - $\operatorname{Expert}_{\text{shared}}(x) = [0.5, 1.0]^T$

**Step 1: Compute Router Logits $H = x W_g$**
$$H_1 = (2.0 \times 1.0) + (1.0 \times 0.0) = \mathbf{2.000000}$$
$$H_2 = (2.0 \times -1.0) + (1.0 \times 2.0) = \mathbf{0.000000}$$
$$H_3 = (2.0 \times 2.0) + (1.0 \times -1.0) = \mathbf{3.000000}$$
$$H_4 = (2.0 \times 0.0) + (1.0 \times 1.0) = \mathbf{1.000000}$$
Router Logits: $H = [2.0, 0.0, 3.0, 1.0]$.

**Step 2: Select Top-$K$ Experts ($K = 2$)**
Ranking the logits:
1. Rank 1: $H_3 = 3.0$ (**Expert 3**)
2. Rank 2: $H_1 = 2.0$ (**Expert 1**)
Active Experts: **$\operatorname{TopK} = \{ \text{Expert 3}, \text{Expert 1} \}$**.

**Step 3: Compute Softmax Gating Weights over Top-2**
$$e^{H_1} = e^{2.0} \approx \mathbf{7.389056}, \quad e^{H_3} = e^{3.0} \approx \mathbf{20.085537}$$
$$\sum = 7.389056 + 20.085537 = \mathbf{27.474593}$$
$$g_1 = \frac{7.389056}{27.474593} \approx \mathbf{0.268941}, \quad g_3 = \frac{20.085537}{27.474593} \approx \mathbf{0.731059}$$

**Step 4: Compute Routed Mixture Output $y_{\text{routed}}$**
$$y_{\text{routed}} = g_1 E_1(x) + g_3 E_3(x)$$
$$= 0.268941 \begin{bmatrix} 1.0 \\ 4.0 \end{bmatrix} + 0.731059 \begin{bmatrix} 3.0 \\ -2.0 \end{bmatrix} = \begin{bmatrix} 0.268941 \\ 1.075764 \end{bmatrix} + \begin{bmatrix} 2.193177 \\ -1.462118 \end{bmatrix} = \begin{bmatrix} \mathbf{2.462118} \\ \mathbf{-0.386354} \end{bmatrix}$$

**Step 5: Add Dedicated Shared Expert Output**
$$y_{\text{total}} = \begin{bmatrix} 0.5 \\ 1.0 \end{bmatrix} + \begin{bmatrix} 2.462118 \\ -0.386354 \end{bmatrix} = \begin{bmatrix} \mathbf{2.962118} \\ \mathbf{0.613646} \end{bmatrix}$$

### 5.2 Hand-Calculation Walkthrough 2: Routing with Extreme Logits

**System & Parameter Setup**
- Token $z$: Router logits $H = [10.0, 10.1, -5.0, 2.0]$
- $K = 2$ experts, $E = 4$ total experts.
- Let $E_1(z) = 10, E_2(z) = 20, E_3(z) = 30, E_4(z) = 40$ (scalar outputs for simplicity).

**Step-by-Step Selection**
- Top-2 logits are $H_2 = 10.1$ and $H_1 = 10.0$.
- Softmax weights:
  $e^{10.1} \approx 24343.0, \quad e^{10.0} \approx 22026.5$
  Sum $= 46369.5$
  $g_2 = \frac{24343.0}{46369.5} = \mathbf{0.524979}$
  $g_1 = \frac{22026.5}{46369.5} = \mathbf{0.475021}$
- Output: $y = g_2 E_2(z) + g_1 E_1(z) = 0.524979 \times 20 + 0.475021 \times 10 = 10.49958 + 4.75021 = \mathbf{15.24979}$.

### 5.3 Summary Visual Grid: MoE Routing Ledger

| Component | Logit $H_i$ | Top-2 Rank | Gating Weight $g_i$ | Status | Expert Output $E_i(x)$ | Weighted Contribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Expert 1** | $2.0000$ | **Rank 2** | **$0.2689$** | Active | $[1.0, 4.0]^T$ | $[0.2689, 1.0758]^T$ |
| **Expert 2** | $0.0000$ | Rank 4 | $0.0000$ | Pruned | $[0.0, 1.0]^T$ | $[0.0000, 0.0000]^T$ |
| **Expert 3** | $3.0000$ | **Rank 1** | **$0.7311$** | Active | $[3.0, -2.0]^T$ | $[2.1932, -1.4621]^T$ |
| **Expert 4** | $1.0000$ | Rank 3 | $0.0000$ | Pruned | $[2.0, 2.0]^T$ | $[0.0000, 0.0000]^T$ |
| **Shared** | Always 1.0 | — | $1.0000$ | Active | $[0.5, 1.0]^T$ | $[0.5000, 1.0000]^T$ |
| **Total Out** | — | — | — | — | — | $\mathbf{[2.962118, 0.613646]^T}$ |

---

## 6. Solved Illustrations

### Illustration 1: Auxiliary Load Balancing Loss Calculation
**Problem:**
Consider a batch of $T = 2$ tokens routed across $E = 2$ experts.
- Token 1 router probabilities: $[P_{1, 1}, P_{1, 2}] = [0.8, 0.2] \implies \operatorname{Top-1} = \text{Expert 1}$.
- Token 2 router probabilities: $[P_{2, 1}, P_{2, 2}] = [0.6, 0.4] \implies \operatorname{Top-1} = \text{Expert 1}$.
Calculate the auxiliary load balancing loss with $\alpha = 0.01$.

**Solution:**
1. **Fraction of tokens dispatched to each expert ($f_i$):**
   Both tokens went to Expert 1: $f_1 = \frac{2}{2} = 1.0$, $f_2 = \frac{0}{2} = 0.0$.
2. **Mean router probabilities ($P_i$):**
   $P_1 = \frac{0.8 + 0.6}{2} = 0.70$
   $P_2 = \frac{0.2 + 0.4}{2} = 0.30$
3. **Balancing Loss:**
   $$\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{i=1}^E f_i P_i = 0.01 \times 2 \times (1.0 \times 0.70 + 0.0 \times 0.30) = 0.02 \times 0.70 = \mathbf{0.0140}$$
   *(If perfectly balanced $f_1=f_2=0.5, P_1=P_2=0.5$, loss would be $0.02 \times (0.25 + 0.25) = \mathbf{0.0100}$, so the collapse is penalized by $+40\%$).* $\blacksquare$

### Illustration 2: Top-2 Routing on 4-Expert Network
**Problem:**
A token's hidden state is $h = [0.5, -0.2, 0.8, 0.1]$. Router weight matrix $W_r \in \mathbb{R}^{4 \times 4}$ is the identity matrix, meaning router logits $g = h W_r^T = h$.
Find the Top-2 experts, compute their softmax weights, and state the final output expression.

**Step-by-Step Solution:**
1. **Compute Logits:**
   $g = [0.5, -0.2, 0.8, 0.1]$.
2. **Top-2 Selection:**
   - Highest logit: $g_2 = 0.8$ (Expert 2, 0-indexed).
   - The top two logits are $g_2 = 0.8$ and $g_0 = 0.5$.
3. **Compute Softmax Weights for Active Experts:**
   - $e^{g_2} = e^{0.8} = \mathbf{2.225541}$
   - $e^{g_0} = e^{0.5} = \mathbf{1.648721}$
   - Sum $= 2.225541 + 1.648721 = \mathbf{3.874262}$
   - Normalized Weights:
     $w_2 = \frac{2.225541}{3.874262} = \mathbf{0.574443}$
     $w_0 = \frac{1.648721}{3.874262} = \mathbf{0.425557}$
     (Check sum: $0.574443 + 0.425557 = 1.000000$).
4. **Final Output:**
   $y = w_0 E_0(h) + w_2 E_2(h) = \mathbf{0.425557} E_0(h) + \mathbf{0.574443} E_2(h)$. $\blacksquare$

### Illustration 3: Load Balancing Auxiliary Loss
**Problem:**
In a network with $E=4$ experts, a batch of 8 tokens routes to experts such that the token count per expert is: E0: 3, E1: 1, E2: 3, E3: 1.
The average router probability (after softmax) over the batch for each expert is $P = [0.35, 0.15, 0.35, 0.15]$.
Compute the auxiliary loss $L_{\text{aux}} = E \sum_{i=1}^E f_i P_i$. Compare it to the ideal balanced loss.

**Step-by-Step Solution:**
1. **Fraction of tokens $f_i$:**
   $f_0 = 3/8 = \mathbf{0.375}$
   $f_1 = 1/8 = \mathbf{0.125}$
   $f_2 = 3/8 = \mathbf{0.375}$
   $f_3 = 1/8 = \mathbf{0.125}$
2. **Given $P_i$:**
   $P_0 = 0.35, \quad P_1 = 0.15, \quad P_2 = 0.35, \quad P_3 = 0.15$.
3. **Compute Loss:**
   $L_{\text{aux}} = 4 \times (f_0 P_0 + f_1 P_1 + f_2 P_2 + f_3 P_3)$
   $f_0 P_0 = 0.375 \times 0.35 = \mathbf{0.13125}$
   $f_1 P_1 = 0.125 \times 0.15 = \mathbf{0.01875}$
   $f_2 P_2 = 0.375 \times 0.35 = \mathbf{0.13125}$
   $f_3 P_3 = 0.125 \times 0.15 = \mathbf{0.01875}$
   Sum $= 0.13125 + 0.01875 + 0.13125 + 0.01875 = \mathbf{0.30000}$
   $L_{\text{aux}} = 4 \times 0.30000 = \mathbf{1.20000}$
4. **Ideal Balanced Loss:**
   If perfectly balanced, $f_i = 1/4 = 0.25$ and $P_i = 0.25$.
   $L_{\text{ideal}} = 4 \times (4 \times (0.25 \times 0.25)) = 4 \times 4 \times 0.0625 = 4 \times 0.25 = \mathbf{1.00000}$.
   The imbalance results in a $20\%$ penalty. $\blacksquare$

### Illustration 4: MoE vs. Dense Parameter Calculation
**Problem:**
Compare active parameters per token for a dense vs. MoE model.
**Dense Model:** $d=4096$, intermediate FFN dim $d_{\text{ffn}}=11008$, $L=32$ layers.
**MoE Model:** Same total parameters per layer, but uses $E=8$ experts, routing to $K=2$ experts per token.
Calculate FFN active parameters per token for each model layer.

**Step-by-Step Solution:**
1. **Dense Model FFN Size:**
   - A standard FFN has three weight matrices (e.g. SwiGLU: up, gate, down).
   - Parameters per layer = $3 \times d \times d_{\text{ffn}} = 3 \times 4096 \times 11008 = \mathbf{135,266,304}$ ($\approx 135.3$M).
   - All $135.3$M parameters are active for every token.

2. **MoE Model Setup:**
   - To keep total parameters the same, each of the 8 experts must have an intermediate dimension of $11008 / 8 = \mathbf{1376}$.
   - Params per expert = $3 \times 4096 \times 1376 = \mathbf{16,908,288}$ ($\approx 16.9$M).
   - Total FFN params per layer = $8 \times 16,908,288 = \mathbf{135,266,304}$ (same as dense).

3. **Active Parameters in MoE:**
   - Since each token routes to only $K=2$ experts, active params per token per layer:
   - $2 \times 16,908,288 = \mathbf{33,816,576}$ ($\approx 33.8$M).

4. **Comparison:**
   - Ratio: $\frac{33,816,576}{135,266,304} = \mathbf{0.25}$.
   - The MoE model uses the same memory footprint but achieves a $4\times$ reduction in FLOPs per token (only $25\%$ active parameters)! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **Mixtral 8x7B & 8x22B (Mistral AI):** Popularized MoE in the open-weights ecosystem, proving that an 8x7B model (13B active params) could beat LLaMA-2 70B dense in reasoning and coding benchmarks.
- **DeepSeek-V3 Architecture:** Features 671 billion total parameters with 37 billion active parameters per token, utilizing 1 shared expert and 256 routed experts (Top-8 routed). Trained for just \$6M compute, it redefined the economics of frontier AI!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Router logits $H = [2.0, 0.0, 3.0, 1.0]$.
   - Top-2 selection (Experts 3 and 1) with softmax weights $g_1 = 0.268941, g_3 = 0.731059$.
   - Final combined output matching $[2.962118, 0.613646]$ to $< 10^{-6}$.
2. **Production-Ready PyTorch Sparse MoE Layer:**
   - Vectorized Top-$K$ gating with `torch.topk`.
   - Auxiliary load balancing loss calculation.
   - DeepSeekMoE architecture with dedicated shared experts.

See implementation in:
[`12_modern_llm_architectures/code/06_sparse_mixture_of_experts.py`](./code/06_sparse_mixture_of_experts.py)
