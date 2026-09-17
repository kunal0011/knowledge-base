# 13.1 Foundations of LLM Reasoning: System 1 vs. System 2 & Chain-of-Thought Dynamics

---

## 1. Intuition & 101 Motivation

In cognitive psychology, Daniel Kahneman's dual-process theory (*Thinking, Fast and Slow*, 2011) bifurcates human thought into two distinct cognitive regimes:
- **System 1 (Fast Thinking):** Instinctive, associative, subconscious, and computationally instantaneous. Examples include identifying that a silhouette is a cat, ducking when an object flies toward you, reading text on a highway billboard, or reflexively stating that $2 + 2 = 4$.
- **System 2 (Slow Thinking):** Deliberative, serial, symbolic, effortful, and computationally demanding. Examples include computing $438 \times 729$ without a calculator, parallel parking in a cramped space, planning 10 moves ahead in chess, or verifying an algebraic proof.

Standard autoregressive Large Language Models (LLMs) executing under greedy or nucleus decoding are fundamentally **pure System 1 engines**. For every emitted token, the model executes exactly **one forward pass** through a static stack of $L$ Transformer layers:
$$\text{FLOPs per Token} = 2 \cdot N_{\text{params}} = \mathcal{O}(L \cdot d_{\text{model}})$$

Regardless of whether the prompt is trivial (*"What is the capital of France?"*) or requires deep logical deduction (*"Prove that there are infinitely many prime numbers"*), a standard Transformer allocates the **exact same constant amount of compute** ($\mathcal{O}(L)$ layer evaluations) before outputting its initial token!

```text
====================================================================================================
           SYSTEM 1 (DIRECT AUTOREGRESSIVE) vs. SYSTEM 2 (CHAIN-OF-THOUGHT)
====================================================================================================

[SYSTEM 1: Direct Next-Token Prediction — Shallow Circuit Depth]
Prompt x ───► [Transformer: L Layers (Fixed Depth)] ───► Direct Answer y
               ▲
               │ Flaw: Fixed O(L) FLOPs & Constant Depth. Collapses an entire
               │ multi-step trajectory into a single forward residual update!

────────────────────────────────────────────────────────────────────────────────────────────────────

[SYSTEM 2: Chain-of-Thought Reasoning — Unrolled Dynamic Computational Graph]
Prompt x ───► [L Layers] ───► Thought z_1
                 │
                 ▼ (External Tape / Scratchpad)
              [L Layers] ───► Thought z_2
                 │
                 ▼
              [L Layers] ───► Thought z_3
                 │
                 ▼  ... (K Sequential Autoregressive Steps)
              [L Layers] ───► Thought z_K ───► [L Layers] ───► Final Answer y

Effective Dynamic Circuit Depth = K × L Nonlinear Layers!
Total Computational Budget      = K × 2·N_params FLOPs (Arbitrarily Scalable at Inference Time)
====================================================================================================
```

When forced to solve complex, multi-step problems in a single token step, the network suffers two fatal theoretical bottlenecks:
1. **The Circuit Depth Lower Bound:** In computational complexity theory, a standard Transformer of depth $L$ belongs to the circuit complexity class $\text{TC}^0$ (constant-depth threshold circuits). Problems requiring sequential dependencies (such as graph reachability, unbounded parity, and arithmetic carrying) cannot be computed by any constant-depth circuit as input length grows.
2. **The Channel Capacity Bottleneck:** In direct prediction, the entire state of intermediate logical deductions must be compressed into the hidden activation vector $h_N^{(L)} \in \mathbb{R}^{d_{\text{model}}}$ of the final prompt token. This creates an informational bottleneck bounded by $d_{\text{model}} \times B$ bits (where $B$ is floating-point precision).

**Chain-of-Thought (CoT)** (Wei et al., 2022; Nye et al., 2021) resolves these bottlenecks by transforming autoregressive inference from a static System 1 forward pass into an unrolled **System 2 computational graph**. By emitting an intermediate sequence of $K$ rationale tokens:
$$z = (z_1, z_2, \dots, z_K) \in \mathcal{V}^K$$
prior to generating the final conclusion $y$, the model externalizes its memory onto an autoregressive scratchpad. The network effectively executes **$K \times L$ sequential layers of nonlinear transformation**, converting a shallow, fixed-depth circuit into a dynamically deep universal computational engine.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Computational Complexity Limits of Direct Prediction (System 1)

Let $\mathcal{V}$ denote the vocabulary of discrete tokens, and let $x = (x_1, x_2, \dots, x_N) \in \mathcal{V}^N$ represent an input prompt sequence of length $N$. Let $y \in \mathcal{V}^{|y|}$ denote the target answer sequence.

Under standard direct autoregressive generation (System 1), the conditional probability of answer $y$ given prompt $x$ is factorized as:
$$P_\theta(y \mid x) = \prod_{t=1}^{|y|} P_\theta(y_t \mid x, y_{<t})$$
where each token distribution is computed via a single forward pass through $L$ Transformer layers:
$$h_t^{(0)} = W_e y_{t-1} + W_p(N + t)$$
$$h_t^{(\ell)} = h_t^{(\ell-1)} + \operatorname{MHA}\left(\operatorname{RMSNorm}(h_t^{(\ell-1)})\right) + \operatorname{FFN}\left(\operatorname{RMSNorm}(\tilde{h}_t^{(\ell)})\right), \quad \ell = 1, \dots, L$$
$$P_\theta(y_t \mid x, y_{<t}) = \operatorname{softmax}\left( W_u \operatorname{RMSNorm}(h_t^{(L)}) \right)$$

#### The Channel Capacity Theorem
In direct prediction of the very first answer token $y_1$, the entire intermediate reasoning path must be communicated through the final hidden activation vector of the prompt:
$$h_N^{(L)} \in \mathbb{R}^{d_{\text{model}}}$$
If each activation is stored in $B$ bits of numerical precision (e.g., $B = 16$ for FP16/BF16), the maximum Shannon information capacity of the residual stream channel is strictly bounded by:
$$\mathcal{I}_{\text{channel}}(x \to y_1) \le d_{\text{model}} \cdot B \text{ bits}$$

For an 8B parameter model with hidden dimension $d_{\text{model}} = 4096$ and 16-bit precision:
$$\mathcal{I}_{\text{channel}} \le 4096 \times 16 = 65,536 \text{ bits} = 8.192 \text{ KB}$$

Any mathematical proof or algorithmic procedure whose intermediate Kolmogorov state complexity exceeds $8.192 \text{ KB}$ cannot physically pass through the direct prediction bottleneck without catastrophic information loss.

---

### 2.2 Rationale-Augmented Formulation (Chain-of-Thought)

Let $z = (z_1, z_2, \dots, z_K) \in \mathcal{Z} = \mathcal{V}^K$ denote a sequence of $K$ intermediate rationale tokens (the "chain of thought", scratchpad, or `<think>` tokens).

The complete joint probability of generating reasoning trace $z$ followed by answer $y$ is factorized autoregressively:
$$P_\theta(z, y \mid x) = \underbrace{\left( \prod_{k=1}^K P_\theta(z_k \mid x, z_{<k}) \right)}_{\text{Rationale Generation: } P_\theta(z \mid x)} \cdot \underbrace{\left( \prod_{m=1}^{|y|} P_\theta(y_m \mid x, z, y_{<m}) \right)}_{\text{Answer Generation: } P_\theta(y \mid x, z)}$$

The true marginal conditional probability of the final answer $y$ is obtained by marginalizing over the entire exponential space of possible reasoning traces $\mathcal{Z}$:
$$P(y \mid x) = \sum_{z \in \mathcal{Z}} P(z, y \mid x) = \sum_{z \in \mathcal{Z}} P_\theta(z \mid x) P_\theta(y \mid x, z) = \mathbb{E}_{z \sim P_\theta(\cdot \mid x)} \left[ P_\theta(y \mid x, z) \right]$$

```text
====================================================================================================
                      THE SCRATCHPAD MEMORY CHANNEL EXPANSION
====================================================================================================

Direct Prediction:
Prompt x ─────────────────────────► [h_N^(L): 8.192 KB Bottleneck] ─────────────────────────► Answer y

Chain-of-Thought:
Prompt x ──► [h_x] ──► Token z_1 ──► [h_z1] ──► Token z_2 ──► ... ──► Token z_K ──► [h_zK] ──► Answer y
             └─────────────┬───────────────────────────────────────────────┘
                           ▼
          Total External Storage = K × d_model × B bits
          For K = 2,048 tokens: Capacity = 2,048 × 8.192 KB = 16.777 MB (2,048× Expansion!)
====================================================================================================
```

By generating $K$ tokens into the context window, the model utilizes the KV cache as an external read-write RAM tape. Each token $z_k$ attends back to all prior tokens $z_{<k}$ via causal attention, expanding the effective storage capacity from $d_{\text{model}} \cdot B$ to:
$$\mathcal{I}_{\text{CoT}} \le K \cdot d_{\text{model}} \cdot B \text{ bits}$$

---

### 2.3 Rationale Decoding Strategies: MAP vs. Marginalization vs. Self-Consistency

Evaluating the exact marginal distribution $P(y \mid x) = \sum_{z \in \mathcal{V}^K} P(z, y \mid x)$ requires summing over $|\mathcal{V}|^K$ paths. With vocabulary size $|\mathcal{V}| \approx 128,000$ and reasoning length $K \approx 1,000$, the support contains $128,000^{1000} \approx 10^{5107}$ terms, making exact marginalization strictly intractable.

Frontier reasoning systems employ three decoding approximations:

#### 1. Maximum A Posteriori (MAP) Greedy Decoding
Approximate the marginal distribution by the mode of the joint distribution:
$$z^* \approx \arg\max_{z \in \mathcal{Z}} P_\theta(z \mid x)$$
$$y^* = \arg\max_{y \in \mathcal{Y}} P_\theta(y \mid x, z^*)$$

*Pathology:* Greedy rationale decoding is vulnerable to **error propagation / reasoning derailment**. If a single arithmetic blunder occurs at token step $k=12$, all downstream conditional distributions $P(z_{>12} \mid x, z_{\le 12})$ are conditioned on an incorrect prefix, leading inevitably to an incorrect final answer.

#### 2. Monte Carlo Marginalization
Sample $M$ independent reasoning paths $\{z^{(1)}, z^{(2)}, \dots, z^{(M)}\}$ i.i.d. from $P_\theta(z \mid x)$ using temperature $T > 0$, and evaluate the empirical average of answer probabilities:
$$\hat{P}_{\text{MC}}(y \mid x) = \frac{1}{M} \sum_{m=1}^M P_\theta(y \mid x, z^{(m)})$$

#### 3. Self-Consistency Majority Voting (Wang et al., 2022)
Instead of averaging token probabilities, sample $M$ reasoning paths, parse the discrete symbolic answer $y^{(m)} = \operatorname{ExtractAnswer}(z^{(m)})$ from each completion, and conduct a majority vote:
$$y^* = \arg\max_{y \in \mathcal{Y}} \sum_{m=1}^M \mathbb{I}\left( y^{(m)} = y \right)$$

Where $\mathbb{I}(\cdot)$ is the indicator function. Self-consistency leverages the asymmetry of reasoning: there are thousands of incorrect ways to solve a problem (which scatter randomly across the answer space), but the correct deductive paths converge onto the unique invariant truth $y^*$.

---

### 2.4 First-Principles Mathematical Derivations

```text
====================================================================================================
DERIVATION 13.1.1: Circuit Depth Lower Bound & Parity Impossibility in Constant-Depth Transformers
====================================================================================================
Problem Statement:
Let x = (b_1, b_2, ..., b_N) ∈ {0, 1}^N be a binary sequence of length N. The parity function is:
  Parity(x) = (∑_{i=1}^N b_i) mod 2.
Prove that:
  1. A standard decoder-only Transformer with L layers, polynomial width d = O(N^c), and constant 
     precision belongs to the circuit complexity class TC^0.
  2. Parity ∉ TC^0, and therefore a fixed-depth Transformer cannot compute unbounded Parity in a 
     single forward pass (System 1) as N → ∞.
  3. By generating K = N intermediate Chain-of-Thought tokens z_k = (z_{k-1} + b_k) mod 2, the 
     effective computational circuit expands to depth O(N), solving Parity with 100% accuracy.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In direct next-token generation, an LLM attempts to map an input sequence $x \in \{0, 1\}^N$ directly to an answer $y \in \{0, 1\}$ in a single forward pass through $L$ layers. We aim to prove from first principles that fixed-depth self-attention architectures cannot compute the parity function for arbitrary $N$, while the autoregressive generation of intermediate scratchpad tokens circumvents this limitation by creating an unrolled circuit of depth $\Omega(N)$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Bounded Layer Depth:** The Transformer architecture consists of a fixed number of layers $L \in \mathbb{N}$, independent of the input sequence length $N$.
2. **Polynomial Width & Precision:** The hidden dimension satisfies $d = \mathcal{O}(N^c)$ for some constant $c \ge 1$, and numerical activations are bounded with precision $B = \mathcal{O}(\log N)$ bits.
3. **Softmax Attention Form:** Self-attention computes attention weights $A_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{m=1}^N \exp(q_i^T k_m / \sqrt{d_k})}$. Threshold gates and feedforward layers use standard Lipschitz activations (GELU, SwiGLU, ReLU).

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
The parity function is the quintessential "dense, non-local" Boolean function: flipping any single bit $b_i$ anywhere in the sequence flips the entire output:
$$\operatorname{Parity}(b_1, \dots, b_i \oplus 1, \dots, b_N) = 1 - \operatorname{Parity}(b_1, \dots, b_i, \dots, b_N)$$
In a single Transformer layer, soft attention computes a convex combination of value vectors:
$$\operatorname{Attn}(Q, K, V)_i = \sum_{j=1}^N A_{ij} v_j, \quad \text{where } \sum_{j=1}^N A_{ij} = 1, \; A_{ij} \ge 0$$
As $N \to \infty$, the maximum attention weight that can be placed on an individual token under uniform or diffuse attention scales as $\mathcal{O}(1/N)$. The perturbation to the attention output caused by flipping a single bit $b_k$ decays as $\mathcal{O}(1/N)$, which vanishes beneath the finite activation precision threshold.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Equivalence of Transformers to $\text{TC}^0$ circuits.*
By definition, $\text{TC}^0$ is the class of decision problems solvable by uniform Boolean circuit families of constant depth ($\mathcal{O}(1)$), polynomial size ($\mathcal{O}(N^c)$), consisting of unbounded fan-in AND, OR, NOT, and majority/threshold gates.
- A standard multi-head attention layer maps input $H^{(\ell-1)} \in \mathbb{R}^{N \times d}$ to $H^{(\ell)} \in \mathbb{R}^{N \times d}$. Each matrix multiplication $Q = H W_Q$, $K = H W_K$, $V = H W_V$ can be simulated by a depth-2 threshold circuit of polynomial size with logarithmic precision.
- The softmax operation $\operatorname{softmax}(S)_{ij} = \frac{\exp(S_{ij})}{\sum_m \exp(S_{im})}$ and subsequent value multiplication can be approximated to precision $2^{-N^c}$ by constant-depth threshold circuits via polynomial approximation (Merrill & Sabharwal, 2023; Strobl et al., 2024).
- Stacking $L$ such layers yields a composite circuit of depth $c_0 \cdot L = \mathcal{O}(1)$.
Therefore, any function computed by a fixed-depth Transformer in a single forward pass belongs to the complexity class $\text{TC}^0$.

*Step 2: Furst-Saxe-Sipser and Ajtai's Lower Bound for Parity.*
A foundational theorem in circuit complexity (Ajtai, 1983; Furst, Saxe, & Sipser, 1984; Yao, 1985) establishes that computing the parity of $N$ variables requires unbounded depth in constant-depth polynomial-size circuits. Specifically, for $\text{AC}^0$ (circuits with AND, OR, NOT gates), depth $d$ requires size:
$$\operatorname{Size} \ge \exp\left( \Omega(N^{1/d}) \right)$$
For threshold circuits ($\text{TC}^0$), Razborov (1987) and Smolensky (1987) proved that circuits of constant depth with threshold gates require exponential size to compute modular counting functions of coprime moduli. While $\text{TC}^0$ can compute parity with unbounded threshold gates of infinite precision, under logarithmic or bounded floating-point precision, a constant-depth circuit cannot distinguish between $\sum b_i = k$ and $\sum b_i = k+1$ for $N \gg d$.

*Step 3: Quantitative Attention Dilution Bound.*
Consider the attention score from the final query token $q_N$ to key $k_j$:
$$S_{N, j} = \frac{q_N^T k_j}{\sqrt{d_k}}$$
Since $\|q_N\|_2 \le C_q$ and $\|k_j\|_2 \le C_k$ are bounded by LayerNorm/RMSNorm:
$$|S_{N, j}| \le \frac{C_q C_k}{\sqrt{d_k}} \triangleq M$$
The attention weight assigned to token $j$ satisfies:
$$A_{N, j} = \frac{\exp(S_{N, j})}{\sum_{m=1}^N \exp(S_{N, m})} \ge \frac{e^{-M}}{N e^M} = \frac{e^{-2M}}{N}$$
and
$$A_{N, j} \le \frac{e^M}{(N-1)e^{-M} + e^M} \le \frac{e^{2M}}{N}$$
The difference in attention weight resulting from flipping bit $b_j \in \{0, 1\}$ is bounded by:
$$|\Delta A_{N, j}| \le \frac{e^{2M} - e^{-2M}}{N} = \mathcal{O}\left(\frac{1}{N}\right)$$
For any fixed depth $L$, the total gradient of the output activation with respect to bit $b_j$ is bounded by:
$$\left\| \frac{\partial h_N^{(L)}}{\partial b_j} \right\|_2 \le \prod_{\ell=1}^L C_{\text{layer}}^{(\ell)} \cdot \mathcal{O}\left(\frac{1}{N}\right) = \mathcal{O}\left(\frac{1}{N}\right)$$
As $N \to \infty$, the sensitivity of the final representation to any individual bit $b_j$ vanishes as $\mathcal{O}(1/N)$. When $1/N < 2^{-B}$ (the machine epsilon of floating-point precision $B$), the influence of bit $b_j$ is numerically zero. Hence, direct System 1 prediction must fail for sufficiently large $N$.

*Step 4: Chain-of-Thought Constructive Proof.*
Now consider generating intermediate rationale tokens $z = (z_1, z_2, \dots, z_N)$, where each token represents the running parity:
$$z_0 \triangleq 0$$
$$z_k = (z_{k-1} + b_k) \bmod 2, \quad k = 1, 2, \dots, N$$
At each step $k$, the Transformer must compute:
$$P_\theta(z_k \mid x, z_{<k})$$
This requires evaluating the Boolean XOR function over exactly **two** variables: the previous state token $z_{k-1}$ and the current bit $b_k$:
$$z_k = z_{k-1} \oplus b_k$$
Because the input to this step is of constant size (2 bits), an XOR gate is computable by a single Transformer layer with $L=1$:
$$\operatorname{XOR}(u, v) = \operatorname{ReLU}(u + v - 0.5) - 2 \operatorname{ReLU}(u + v - 1.5)$$
Each step $k \in \{1, \dots, N\}$ is evaluated in a separate autoregressive generation step. Across $N$ steps, the unrolled computational graph has total depth:
$$\text{Depth}_{\text{total}} = N \times L = \Omega(N)$$
Because the circuit depth scales linearly with input size $N$, the computational complexity constraint is satisfied, and the model evaluates parity with $100\%$ accuracy. $\blacksquare$

---

```text
====================================================================================================
DERIVATION 13.1.2: Channel Capacity of Hidden Residual Stream vs. Rationale Scratchpads
====================================================================================================
Problem Statement:
Let x be a problem prompt requiring intermediate proof state S. Prove that:
  1. Under direct System 1 generation, the mutual information between problem state S and final 
     answer y is strictly bounded by I(S; y) ≤ d_model · B bits, where B is activation bit precision.
  2. For reasoning tasks with Kolmogorov state complexity K(S) > d_model · B, direct generation 
     incurs an information deficit ΔI = K(S) - d_model · B > 0, forcing hallucination.
  3. Emitting K rationale tokens expands channel capacity to I_CoT ≤ K · d_model · B, eliminating 
     the information deficit for K ≥ K(S) / (d_model · B).
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
We formalize reasoning as an information-theoretic transmission problem. The prompt $x$ defines an underlying state space of necessary intermediate deductions $S$. We prove that passing $S$ through a single hidden vector $h_N^{(L)} \in \mathbb{R}^d$ imposes a Shannon channel bottleneck, whereas generating $K$ tokens into the context window expands the channel capacity proportionally to $K$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Quantized Channel Capacity:** Activations are represented using $B$-bit floating-point numbers (e.g., IEEE 754 FP16, $B = 16$).
2. **Markov Chain Structure of Generation:**
   - Direct Generation: $x \longrightarrow S \longrightarrow h_N^{(L)} \longrightarrow y$
   - CoT Generation: $x \longrightarrow S_1 \longrightarrow z_1 \longrightarrow S_2 \longrightarrow z_2 \longrightarrow \dots \longrightarrow S_K \longrightarrow z_K \longrightarrow y$
3. **Data Processing Inequality:** For any Markov chain $X \to Y \to Z$, $I(X; Z) \le I(X; Y)$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
Think of the residual stream as a pipe of width $d_{\text{model}}$. In direct generation, an entire multi-page deduction must be squeezed through the pipe at a single instant $t=N$. If the volume of required intermediate mathematical state exceeds the pipe's cross-sectional capacity, information is irreversibly discarded. Chain-of-Thought turns the instantaneous pipe into a conveyor belt over time: information flows continuously across $K$ discrete token steps, allowing arbitrary amounts of working memory to be recorded and retrieved via attention.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Direct prediction channel capacity bound.*
Consider the Markov chain for direct generation:
$$S \longrightarrow h_N^{(L)} \longrightarrow y$$
By the Data Processing Inequality:
$$I(S; y) \le I(S; h_N^{(L)})$$
The vector $h_N^{(L)}$ consists of $d_{\text{model}}$ scalar activations:
$$h_N^{(L)} = \left[ h_{N, 1}^{(L)}, h_{N, 2}^{(L)}, \dots, h_{N, d_{\text{model}}}^{(L)} \right]^T$$
Each coordinate $h_{N, j}^{(L)}$ is quantized to $B$ bits of precision, meaning it can take at most $2^B$ distinct discrete values.
The Shannon entropy of the discrete representation $\tilde{h}_N^{(L)}$ is bounded by the uniform distribution:
$$H(\tilde{h}_N^{(L)}) \le \log_2 \left( (2^B)^{d_{\text{model}}} \right) = d_{\text{model}} \cdot B \text{ bits}$$
Since mutual information cannot exceed the entropy of the transmitting channel:
$$I(S; h_N^{(L)}) \le H(\tilde{h}_N^{(L)}) \le d_{\text{model}} \cdot B$$
Therefore, the maximum mutual information between the reasoning state $S$ and the answer $y$ is:
$$I_{\text{direct}}(S; y) \le d_{\text{model}} \cdot B \quad \blacksquare$$

*Step 2: Information deficit and reasoning failure.*
Let $K(S)$ denote the Kolmogorov complexity (minimum description length) of the logical deductions required to establish $y$ from $x$.
If:
$$K(S) > d_{\text{model}} \cdot B$$
Then by Shannon's Source-Channel Coding Theorem, the reconstruction error probability is bounded away from zero:
$$P(\hat{y} \ne y^*) \ge 1 - \frac{d_{\text{model}} \cdot B}{K(S)} > 0$$
The model is physically incapable of retaining the necessary deductive steps, resulting in hallucinated leaps in logic.

*Step 3: Chain-of-Thought channel capacity expansion.*
In Chain-of-Thought generation, the model emits intermediate tokens $z = (z_1, z_2, \dots, z_K)$.
The Markov chain is:
$$S \longrightarrow (z_1, z_2, \dots, z_K) \longrightarrow y$$
Each token $z_k \in \mathcal{V}$ is appended to the context sequence. At step $k$, the network computes hidden state:
$$h_{z_k}^{(L)} \in \mathbb{R}^{d_{\text{model}}}$$
The total information transmitted through the sequence of $K$ autoregressive steps is the joint entropy of all intermediate hidden states:
$$I_{\text{CoT}}(S; y) \le H\left( h_{z_1}^{(L)}, h_{z_2}^{(L)}, \dots, h_{z_K}^{(L)} \right)$$
By the chain rule of entropy:
$$H\left( h_{z_1}^{(L)}, \dots, h_{z_K}^{(L)} \right) = \sum_{k=1}^K H\left( h_{z_k}^{(L)} \;\Big|\; h_{z_{<k}}^{(L)} \right) \le \sum_{k=1}^K H\left( h_{z_k}^{(L)} \right) \le K \cdot d_{\text{model}} \cdot B$$
To eliminate the information deficit ($I_{\text{CoT}} \ge K(S)$), the required reasoning length satisfies:
$$K \ge \frac{K(S)}{d_{\text{model}} \cdot B}$$
Emitting $K$ intermediate tokens scales the memory capacity linearly with $K$, allowing arbitrarily complex multi-step reasoning to be processed without loss. $\blacksquare$

---

```text
====================================================================================================
DERIVATION 13.1.3: Convergence of Self-Consistency via Hoeffding's Inequality
====================================================================================================
Problem Statement:
Let y* ∈ 𝒴 be the unique correct answer to prompt x, and let 𝒴_incorrect = 𝒴 \ {y*}.
Suppose M independent reasoning paths z^(1), ..., z^(M) ~ P_θ(z | x) are sampled at temperature T > 0, 
each yielding candidate answer y^(m) = ExtractAnswer(z^(m)).
Let p = P(y^(m) = y*) be the single-sample success probability, with p > 1/2.
Let the incorrect answers be distributed over C distinct incorrect categories with probabilities 
q_c = P(y^(m) = y_c), such that ∑_{c=1}^C q_c = 1 - p, and max_{c} q_c ≤ q_max < p.
Prove that:
  1. The probability that majority voting selects an incorrect answer decays exponentially with M:
       P(Majority Vote Fails) ≤ (C + 1) · exp( -2 M (p - 1/2)^2 ).
  2. Under uniform incorrect answer scattering (q_c = (1-p)/C), the error bound sharpens to:
       P(Majority Vote Fails) ≤ exp( -2 M (p - (1-p)/C)^2 ).
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
We prove that Self-Consistency (Wang et al., 2022) exponentially amplifies a weak reasoning policy ($p > 0.5$) into near-certain correctness ($P(\text{Success}) \to 1.0$) as the number of inference samples $M \to \infty$. We derive explicit finite-sample error bounds using Hoeffding's Inequality.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Independent Sampling:** Reasoning trajectories $z^{(1)}, \dots, z^{(M)}$ are sampled conditionally i.i.d. given prompt $x$:
   $$P\left( z^{(1)}, \dots, z^{(M)} \mid x \right) = \prod_{m=1}^M P_\theta(z^{(m)} \mid x)$$
2. **Dominant Correct Mode:** The single-sample probability of arriving at the correct answer $y^*$ strictly exceeds $0.5$:
   $$p \triangleq P(y^{(m)} = y^*) > \frac{1}{2}$$
   *(Note: As proven in Part 4, Step 4, even if $p < 0.5$, majority voting succeeds as long as $p > \max_c q_c$, which holds when incorrect answers scatter across diverse erroneous attractors).*
3. **Bounded Indicator Variables:** For each rollout $m$, define the Bernoulli random variable:
   $$X_m = \mathbb{I}\left( y^{(m)} = y^* \right) \in \{0, 1\}, \quad \text{with } \mathbb{E}[X_m] = p$$

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
Consider a high-dimensional state space of reasoning paths. The correct solution path follows a rigorous logical conduit governed by mathematical invariants; hence, diverse sampled paths that discover valid deductions all terminate at the exact same point $y^*$. Conversely, fallacious reasoning paths wander off into distinct, uncorrelated failure modes (different calculation errors, mistaken signs, incorrect algebraic factorizations). Thus, incorrect answers scatter widely over the space of distractors, while the correct answer forms a sharp probability peak.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Formal definition of the majority voting decision rule.*
The vote count for the correct answer $y^*$ across $M$ sampled paths is:
$$V(y^*) = \sum_{m=1}^M \mathbb{I}\left( y^{(m)} = y^* \right) = \sum_{m=1}^M X_m$$
Majority voting selects $y^*$ if $V(y^*) > \frac{M}{2}$ (strict majority):
$$\text{Success Event: } \mathcal{S} = \left\{ V(y^*) > \frac{M}{2} \right\} = \left\{ \frac{1}{M} \sum_{m=1}^M X_m > \frac{1}{2} \right\}$$
The failure event occurs if the sample mean falls below or equal to $1/2$:
$$\mathcal{F} = \left\{ \frac{1}{M} \sum_{m=1}^M X_m \le \frac{1}{2} \right\}$$

*Step 2: Application of Hoeffding's Inequality.*
Recall Hoeffding's Inequality: Let $X_1, \dots, X_M$ be independent random variables strictly bounded in $[a_i, b_i] = [0, 1]$. Let $\bar{X} = \frac{1}{M} \sum_{m=1}^M X_m$, and $\mu = \mathbb{E}[\bar{X}] = p$.
For any $\epsilon > 0$:
$$P\left( \bar{X} - p \le -\epsilon \right) \le \exp\left( -\frac{2 M^2 \epsilon^2}{\sum_{i=1}^M (b_i - a_i)^2} \right) = \exp\left( -\frac{2 M^2 \epsilon^2}{M (1 - 0)^2} \right) = \exp\left( -2 M \epsilon^2 \right)$$

*Step 3: Derivation of the majority error bound.*
Set the threshold deviation:
$$\bar{X} \le \frac{1}{2} \iff \bar{X} - p \le -\left(p - \frac{1}{2}\right)$$
Define $\epsilon = p - \frac{1}{2}$. Since $p > 1/2$, $\epsilon > 0$.
Substituting into Hoeffding's bound:
$$P\left( \frac{1}{M} \sum_{m=1}^M X_m \le \frac{1}{2} \right) \le \exp\left( -2 M \left(p - \frac{1}{2}\right)^2 \right)$$
Therefore, the probability that the correct answer wins a strict majority satisfies:
$$P(\text{Majority Vote Correct}) \ge 1 - \exp\left( -2 M \left(p - \frac{1}{2}\right)^2 \right) \quad \blacksquare$$
As $M \to \infty$, the failure probability vanishes exponentially:
$$\lim_{M \to \infty} P(\text{Failure}) = \mathcal{O}\left( e^{-c M} \right) \longrightarrow 0$$

*Step 4: Extension to Plurality Voting when $p < 0.5$ (The Scrambling Effect).*
Now suppose $p < 0.5$, but incorrect answers scatter among $C$ distinct wrong answers $\{y_1, \dots, y_C\}$, with $P(y^{(m)} = y_c) = q_c$.
The plurality voting rule selects $y^*$ if:
$$V(y^*) > \max_{c \in \{1, \dots, C\}} V(y_c)$$
A failure occurs only if there exists some incorrect candidate $y_c$ such that $V(y_c) \ge V(y^*)$.
For a specific candidate $y_c$, define the paired difference indicator:
$$D_m^{(c)} = \mathbb{I}\left( y^{(m)} = y^* \right) - \mathbb{I}\left( y^{(m)} = y_c \right) \in \{-1, 0, 1\}$$
The expected difference is:
$$\mathbb{E}\left[ D_m^{(c)} \right] = p - q_c > 0$$
Since $D_m^{(c)} \in [-1, 1]$, the range is $b - a = 1 - (-1) = 2$.
Applying Hoeffding's Inequality to $\bar{D}^{(c)} = \frac{1}{M} \sum_{m=1}^M D_m^{(c)}$:
$$P\left( V(y^*) \le V(y_c) \right) = P\left( \bar{D}^{(c)} \le 0 \right) = P\left( \bar{D}^{(c)} - (p - q_c) \le -(p - q_c) \right) \le \exp\left( -\frac{2 M (p - q_c)^2}{2^2} \right) = \exp\left( -\frac{M (p - q_c)^2}{2} \right)$$
Applying the Union Bound over all $C$ incorrect candidates:
$$P(\text{Plurality Vote Fails}) \le \sum_{c=1}^C P\left( V(y^*) \le V(y_c) \right) \le \sum_{c=1}^C \exp\left( -\frac{M (p - q_c)^2}{2} \right)$$
If errors are uniformly distributed ($q_c = \frac{1 - p}{C}$):
$$P(\text{Plurality Vote Fails}) \le C \cdot \exp\left( -\frac{M}{2} \left( p - \frac{1-p}{C} \right)^2 \right) \quad \blacksquare$$
Even if $p = 0.20$, if errors scatter across $C = 100$ distinct incorrect answers ($q_c = 0.80 / 100 = 0.008$), the gap is $p - q_c = 0.20 - 0.008 = 0.192 > 0$, and plurality voting converges to $100\%$ accuracy exponentially!

---

## 3. Geometric & Physical Interpretation

### 3.1 Geodesics in Non-Convex Semantic Energy Landscapes

Consider the loss or probability landscape of a language model as a high-dimensional Riemannian potential energy surface $V(h)$:
$$V(h) = -\ln P(y \mid h)$$
- **Direct Generation (System 1):** The prompt $x$ defines an initial potential well $\mathcal{W}_x$. The correct answer $y^*$ resides in a distinct basin of attraction $\mathcal{W}_{y^*}$. Separating these two basins lies a massive, high-dimensional non-convex **energy barrier** $\Delta V^\ddagger$ representing the complex intermediate logical lemmas.
  In a single forward pass, the model attempts an instantaneous "tunneling" step across $\Delta V^\ddagger$. Because gradient descent during pretraining optimizes for broad surface statistics, the trajectory gets trapped in the nearest local minimum $\mathcal{W}_{\text{hallucination}}$—a superficially plausible but factually incorrect answer.
- **Chain of Thought (System 2):** Intermediate tokens $z_1, z_2, \dots, z_K$ act as physical **stepping stones** or waypoints. Each token $z_k$ corresponds to a local minimum that lowers the transition barrier for the subsequent token:
  $$\Delta V(z_k \to z_{k+1}) \ll \Delta V(x \to y^*)$$
  The sequence of reasoning tokens traces a smooth, low-energy **geodesic** over the mountain pass, navigating around high-energy barriers rather than attempting an impossible direct jump.

```text
====================================================================================================
             SEMANTIC POTENTIAL ENERGY LANDSCAPE: DIRECT vs. CHAIN-OF-THOUGHT
====================================================================================================

 Potential Energy V(h)
      ▲
      │                 Direct Generation: Trapped in False Local Basin!
      │                        ▲
      │                       / \  (High Reasoning Energy Barrier ΔV‡)
      │             x        /   \
      │          (Prompt)   /  *  \            y* (Correct Answer)
      │           \        / Hallucination     /
      │            \______/   (Local Min)     /
      │                               \______/
      │
      │                 Chain-of-Thought: Stepping-Stone Geodesic Path
      │          x ───► z_1 ───► z_2 ───► z_3 ───► z_4 ───► y*
      │          (Each intermediate token surmounts a tiny local barrier δV << ΔV‡)
      └────────────────────────────────────────────────────────────────────────► State Space
====================================================================================================
```

---

## 4. Real-World Analogy: Mental Math Prodigy vs. The Legal Pad

To understand why Chain-of-Thought is an absolute structural necessity rather than a prompting trick, consider the biological limitations of the human brain:

- **System 1 (Direct Prediction — Pure Mental Arithmetic):**
  Suppose an examiner walks up to you and demands: *"Multiply $739 \times 482$ in your head right now, and say the final 6 digits within 2 seconds!"*
  Your brain freezes. Why? Because human working memory in the prefrontal cortex can only hold $7 \pm 2$ active chunks simultaneously (Miller's Law). When multiplying $739 \times 482$, you must compute partial products ($739 \times 2 = 1478$, $739 \times 80 = 59120$, $739 \times 400 = 295600$) and carry intermediate sums. Before you finish computing the second partial product, the digits of the first partial product have decayed from working memory. You are forced to guess: *"around 350,000"* (a plausible System 1 hallucination).
- **System 2 (Chain-of-Thought — The Legal Pad):**
  Now suppose the examiner places a legal pad and pencil on the desk. You write:
  1. $739 \times 2 = 1,478$
  2. $739 \times 8 = 5,912 \implies 59,120$
  3. $739 \times 4 = 2,956 \implies 295,600$
  4. Summing column-by-column:
     $$\begin{array}{rl}
       & 001478 \\
     + & 059120 \\
     + & 295600 \\
     \hline
       & \mathbf{356,198}
     \end{array}$$
  You state the exact answer: **356,198**.

The legal pad does not make your brain smarter; it **externalizes working memory**. Each line written on the page frees up your prefrontal cortex to focus exclusively on the next atomic single-digit addition. **The tokens generated in Chain-of-Thought are the LLM's legal pad.**

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, cell-by-cell numerical walkthrough of:
1. **Direct System 1 Error (Greedy Failure on Logic Puzzle)**
2. **CoT Joint Probability Calculations over Multiple Reasoning Paths**
3. **Marginalization across Paths ($P(y \mid x) = \sum P(y, z \mid x)$)**
4. **Self-Consistency Majority Vote Consensus**

---

### 5.1 Concrete Setup & Input Values

Consider a mathematical logic problem $x$:
- **Candidate Answers:**
  - $y_1$: The mathematically correct answer
  - $y_2$: A superficially attractive distractor / hallucination
- **Direct System 1 Probabilities (Zero CoT):**
  - $P(y_1 \mid x) = 0.400000$
  - $P(y_2 \mid x) = 0.600000$
  *(Under direct greedy decoding, System 1 selects $y_2$, failing completely because $0.60 > 0.40$!)*

- **Reasoning Paths Available under Chain-of-Thought (System 2):**
  - **Path A ($z_A$):** Rigorous deduction applying formal algebraic lemmas.
    - Proposal probability: $P(z_A \mid x) = 0.600000$
    - Conditional answer probabilities:
      - $P(y_1 \mid x, z_A) = 0.900000$ (Correct answer given sound logic)
      - $P(y_2 \mid x, z_A) = 0.100000$ (Residual error)
  - **Path B ($z_B$):** Heuristic, confused reasoning path containing a sign error.
    - Proposal probability: $P(z_B \mid x) = 0.300000$
    - Conditional answer probabilities:
      - $P(y_1 \mid x, z_B) = 0.200000$
      - $P(y_2 \mid x, z_B) = 0.800000$ (Distractor reinforced by flawed logic)
  - **Path C ($z_{\text{other}}$):** Incomplete or unparseable trajectory.
    - Proposal probability: $P(z_{\text{other}} \mid x) = 0.100000$ (Yields null answer).

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Value / Role in Hand Trace |
| :--- | :--- | :--- |
| $P(z \mid x)$ | `p_z` | Prior probability of generating reasoning path $z$ from prompt $x$ |
| $P(y \mid x, z)$ | `p_y_given_z` | Likelihood of concluding answer $y$ given intermediate scratchpad $z$ |
| $P(z, y \mid x)$ | `joint_prob` | Joint trajectory probability: $P(z \mid x) \times P(y \mid x, z)$ |
| $P(y \mid x)$ | `marginal_prob` | Total marginal probability: $\sum_z P(z, y \mid x)$ |
| $M$ | `num_rollouts` | Number of independent sampled reasoning paths drawn ($M = 5$) |
| $V(y)$ | `vote_count` | Number of rollouts concluding with candidate answer $y$ |
| $\text{Consensus}$ | `consensus_rate` | Empirical vote fraction: $V(y) / M$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Walkthrough 1: Joint Trajectory Probabilities
We compute the joint probability $P(z, y \mid x) = P(z \mid x) \cdot P(y \mid x, z)$ for every combination:

1. **Path A ($z_A$) Trajectories:**
   $$P(z_A, y_1 \mid x) = P(z_A \mid x) \cdot P(y_1 \mid x, z_A) = 0.600000 \times 0.900000 = \mathbf{0.540000}$$
   $$P(z_A, y_2 \mid x) = P(z_A \mid x) \cdot P(y_2 \mid x, z_A) = 0.600000 \times 0.100000 = \mathbf{0.060000}$$
   $$\text{Check Path A Sum: } 0.540000 + 0.060000 = 0.600000 = P(z_A \mid x) \quad \checkmark$$

2. **Path B ($z_B$) Trajectories:**
   $$P(z_B, y_1 \mid x) = P(z_B \mid x) \cdot P(y_1 \mid x, z_B) = 0.300000 \times 0.200000 = \mathbf{0.060000}$$
   $$P(z_B, y_2 \mid x) = P(z_B \mid x) \cdot P(y_2 \mid x, z_B) = 0.300000 \times 0.800000 = \mathbf{0.240000}$$
   $$\text{Check Path B Sum: } 0.060000 + 0.240000 = 0.300000 = P(z_B \mid x) \quad \checkmark$$

---

#### Walkthrough 2: Marginal Probability Inversion
Marginalize over reasoning paths to obtain the true conditional probability of each answer:

1. **Marginal Probability of Correct Answer $y_1$:**
   $$P(y_1 \mid x) = P(z_A, y_1 \mid x) + P(z_B, y_1 \mid x) = 0.540000 + 0.060000 = \mathbf{0.600000}$$

2. **Marginal Probability of Incorrect Distractor $y_2$:**
   $$P(y_2 \mid x) = P(z_A, y_2 \mid x) + P(z_B, y_2 \mid x) = 0.060000 + 0.240000 = \mathbf{0.300000}$$

3. **Comparison with Direct System 1 Generation:**
   - **Direct System 1 (No CoT):**
     $$P(y_1 \mid x) = 0.400000, \quad P(y_2 \mid x) = 0.600000 \implies \frac{P(y_1)}{P(y_2)} = 0.667 \quad (\text{\textbf{System 1 Fails}})$$
   - **Marginal System 2 (With CoT):**
     $$P(y_1 \mid x) = 0.600000, \quad P(y_2 \mid x) = 0.300000 \implies \frac{P(y_1)}{P(y_2)} = \mathbf{2.000} \quad (\text{\textbf{System 2 Succeeds!}})$$

CoT inverted the relative likelihood from a $1.5\times$ deficit into a **$2.0\times$ preference for the truth**!

---

#### Walkthrough 3: Self-Consistency Majority Vote Trace
Suppose we draw $M = 5$ independent sample completions from this model distribution:
- **Rollout 1:** Samples Path A ($z_A$) $\longrightarrow$ emits conclusion $y_1$
- **Rollout 2:** Samples Path A ($z_A$) $\longrightarrow$ emits conclusion $y_1$
- **Rollout 3:** Samples Path B ($z_B$) $\longrightarrow$ emits conclusion $y_2$
- **Rollout 4:** Samples Path A ($z_A$) $\longrightarrow$ emits conclusion $y_1$
- **Rollout 5:** Samples Path A ($z_A$) $\longrightarrow$ emits conclusion $y_1$

Tallying the votes:
$$V(y_1) = 4, \quad V(y_2) = 1$$
$$\text{Winning Answer: } y^* = \arg\max_{y \in \{y_1, y_2\}} V(y) = \mathbf{y_1}$$
$$\text{Consensus Fraction: } \frac{4}{5} = \mathbf{80.0\%} \quad \blacksquare$$

---

### 5.4 Summary Visual Grid: Reasoning Trajectory Ledger

```text
====================================================================================================
                        REASONING TRAJECTORY LEDGER TABLE
====================================================================================================
 Path | Quality Description | P(z | x) | P(y_1 | x, z) | P(y_2 | x, z) | Joint P(z, y_1) | Joint P(z, y_2)
──────┼─────────────────────┼──────────┼───────────────┼───────────────┼─────────────────┼────────────────
  z_A │ Rigorous Derivation │ 0.600000 │   0.900000    │   0.100000    │    0.540000     │    0.060000    
  z_B │ Flawed Sign Error   │ 0.300000 │   0.200000    │   0.800000    │    0.060000     │    0.240000    
  z_C │ Null / Unparseable  │ 0.100000 │   0.000000    │   0.000000    │    0.000000     │    0.000000    
══════╪═════════════════════╪══════════╪═══════════════╪═══════════════╪═════════════════╪════════════════
CoT Marginal P(y | x)       │    —     │       —       │       —       │    0.600000     │    0.300000    
Direct System 1 (No CoT)    │    —     │       —       │       —       │    0.400000     │    0.600000    
====================================================================================================
```

---

## 6. Solved Illustrations

### Illustration 1: Parity Problem Complexity & The Scratchpad Mechanism
**Problem:**
Given bit sequence $x = [1, 0, 1, 1, 1, 0, 1, 1]$ of length $N = 8$.
1. Compute the analytical parity: $\operatorname{Parity}(x) = (\sum_{i=1}^8 b_i) \bmod 2$.
2. Show step-by-step how an autoregressive scratchpad token sequence evaluates this parity without requiring depth greater than $L=1$.

**Solution:**
1. **Direct Calculation:**
   $$\sum_{i=1}^8 b_i = 1 + 0 + 1 + 1 + 1 + 0 + 1 + 1 = 6$$
   $$\operatorname{Parity}(x) = 6 \bmod 2 = \mathbf{0}$$
2. **Step-by-Step Scratchpad Generation:**
   Let $z_k$ denote the $k$-th intermediate token emitted by the model:
   - Token 1: $z_1 = b_1 = \mathbf{1}$
   - Token 2: $z_2 = (z_1 + b_2) \bmod 2 = (1 + 0) \bmod 2 = \mathbf{1}$
   - Token 3: $z_3 = (z_2 + b_3) \bmod 2 = (1 + 1) \bmod 2 = \mathbf{0}$
   - Token 4: $z_4 = (z_3 + b_4) \bmod 2 = (0 + 1) \bmod 2 = \mathbf{1}$
   - Token 5: $z_5 = (z_4 + b_5) \bmod 2 = (1 + 1) \bmod 2 = \mathbf{0}$
   - Token 6: $z_6 = (z_5 + b_6) \bmod 2 = (0 + 0) \bmod 2 = \mathbf{0}$
   - Token 7: $z_7 = (z_6 + b_7) \bmod 2 = (0 + 1) \bmod 2 = \mathbf{1}$
   - Token 8: $z_8 = (z_7 + b_8) \bmod 2 = (1 + 1) \bmod 2 = \mathbf{0}$
   The complete emitted scratchpad is $[1, 1, 0, 1, 0, 0, 1, 0]$, terminating with exact answer **0**. Each step performs a local binary operation, requiring only $\mathcal{O}(1)$ depth per token. $\blacksquare$

---

### Illustration 2: Temperature Scaling for Diverse Reasoning Paths
**Problem:**
At a critical deduction branch point, a policy $\pi_\theta$ produces unnormalized logits $u = [3.0, 1.5, 0.5, 2.5]$ over four candidate reasoning tokens $\{z_A, z_B, z_C, z_D\}$.
Compute the exact softmax probabilities under:
1. Standard temperature: $T = 1.0$
2. Low temperature (greedy exploitation): $T = 0.5$
3. High temperature (creative exploration): $T = 2.0$
Explain why $T \in [0.6, 0.8]$ is optimal for Self-Consistency while $T \to 0$ collapses diversity.

**Solution:**
The temperature-scaled softmax probability is:
$$P(z_i) = \frac{\exp(u_i / T)}{\sum_{j=1}^4 \exp(u_j / T)}$$

1. **At $T = 1.0$:**
   - Logits: $[3.0, 1.5, 0.5, 2.5]$
   - Exponentials: $e^{3.0} \approx 20.08554, e^{1.5} \approx 4.48169, e^{0.5} \approx 1.64872, e^{2.5} \approx 12.18249$
   - Denominator: $\sum = 20.08554 + 4.48169 + 1.64872 + 12.18249 = 38.39844$
   - Probabilities:
     $$P = [\mathbf{0.52308}, \mathbf{0.11672}, \mathbf{0.04294}, \mathbf{0.31726}]$$

2. **At $T = 0.5$:**
   - Scaled logits ($u / 0.5$): $[6.0, 3.0, 1.0, 5.0]$
   - Exponentials: $e^6 \approx 403.42879, e^3 \approx 20.08554, e^1 \approx 2.71828, e^5 \approx 148.41316$
   - Denominator: $\sum = 403.42879 + 20.08554 + 2.71828 + 148.41316 = 574.64577$
   - Probabilities:
     $$P = [\mathbf{0.70205}, \mathbf{0.03495}, \mathbf{0.00473}, \mathbf{0.25827}]$$
   *(The top token monopolizes $70.2\%$ of probability mass, severely suppressing exploration).*

3. **At $T = 2.0$:**
   - Scaled logits ($u / 2.0$): $[1.50, 0.75, 0.25, 1.25]$
   - Exponentials: $e^{1.5} \approx 4.48169, e^{0.75} \approx 2.11700, e^{0.25} \approx 1.28403, e^{1.25} \approx 3.49034$
   - Denominator: $\sum = 4.48169 + 2.11700 + 1.28403 + 3.49034 = 11.37306$
   - Probabilities:
     $$P = [\mathbf{0.39406}, \mathbf{0.18614}, \mathbf{0.11290}, \mathbf{0.30690}]$$

**Implication for Self-Consistency:** At $T \to 0$, all $M$ rollouts generate identical greedy paths, collapsing diversity and failing if the greedy path is flawed. At $T = 0.7$, the model samples alternative valid deductive paths while avoiding low-probability degenerative hallucination paths. $\blacksquare$

---

### Illustration 3: Exact Binomial Scaling of Majority Voting
**Problem:**
Suppose an individual sampled Chain-of-Thought path yields the correct answer with probability $p = 0.60$. A majority vote over $M$ independent paths requires at least $\lfloor M/2 \rfloor + 1$ correct answers.
1. Calculate the exact probability of success for $M = 1$ (single sample), $M = 5$, and $M = 9$.
2. Determine the minimum sample size $M$ required to achieve $\ge 95\%$ accuracy.

**Solution:**
Let $X \sim \operatorname{Binomial}(M, p)$ with $p = 0.60$, $q = 1 - p = 0.40$.

1. **For $M = 1$:**
   $$P(\text{Correct}) = p = \mathbf{0.60000 \quad (60.00\%)}$$

2. **For $M = 5$ (Need $X \ge 3$):**
   $$P(X = 3) = \binom{5}{3} (0.6)^3 (0.4)^2 = 10 \times 0.216 \times 0.16 = 0.34560$$
   $$P(X = 4) = \binom{5}{4} (0.6)^4 (0.4)^1 = 5 \times 0.1296 \times 0.4 = 0.25920$$
   $$P(X = 5) = \binom{5}{5} (0.6)^5 (0.4)^0 = 1 \times 0.07776 \times 1 = 0.07776$$
   $$P(X \ge 3) = 0.34560 + 0.25920 + 0.07776 = \mathbf{0.68256 \quad (68.26\%)}$$
   *(An absolute gain of $+8.26\%$ over single sampling).*

3. **For $M = 9$ (Need $X \ge 5$):**
   $$P(X = 5) = \binom{9}{5} (0.6)^5 (0.4)^4 = 126 \times 0.07776 \times 0.02560 = 0.250823$$
   $$P(X = 6) = \binom{9}{6} (0.6)^6 (0.4)^3 = 84 \times 0.046656 \times 0.06400 = 0.250823$$
   $$P(X = 7) = \binom{9}{7} (0.6)^7 (0.4)^2 = 36 \times 0.0279936 \times 0.16000 = 0.161243$$
   $$P(X = 8) = \binom{9}{8} (0.6)^8 (0.4)^1 = 9 \times 0.01679616 \times 0.4000 = 0.060466$$
   $$P(X = 9) = \binom{9}{9} (0.6)^9 (0.4)^0 = 1 \times 0.01007770 \times 1.0000 = 0.010078$$
   $$P(X \ge 5) = 0.250823 + 0.250823 + 0.161243 + 0.060466 + 0.010078 = \mathbf{0.733433 \quad (73.34\%)}$$
   *(An absolute gain of $+13.34\%$ over single sampling).*

4. **Minimum $M$ for $95\%$ Accuracy via Hoeffding:**
   Using the Hoeffding bound:
   $$P(\text{Fail}) \le \exp\left( -2 M (p - 0.5)^2 \right) \le 0.05$$
   $$-2 M (0.6 - 0.5)^2 \le \ln(0.05) \implies -2 M (0.01) \le -2.99573 \implies 0.02 M \ge 2.99573$$
   $$M \ge \frac{2.99573}{0.02} \approx 149.78 \implies \mathbf{M \approx 150 \text{ samples}} \quad \blacksquare$$

---

### Illustration 4: CoT Token Budget for $N$-Digit Multiplication
**Problem:**
Calculate the computational FLOPs and context capacity required to multiply two $N$-digit integers ($N=3$: $432 \times 567 = 244,944$) under direct prediction vs. Chain-of-Thought.

**Solution:**
1. **Direct Generation Failure:**
   To output the 6-digit string `244944` directly in a single pass, the Transformer must evaluate:
   $$y_1 = \text{"2"}, \; y_2 = \text{"4"}, \; y_3 = \text{"4"}, \; y_4 = \text{"9"}, \; y_5 = \text{"4"}, \; y_6 = \text{"4"}$$
   The most significant digit `2` depends on the recursive carrying of all lower digits:
   $$\text{carry}_k = \left\lfloor \frac{\text{sum}_k + \text{carry}_{k-1}}{10} \right\rfloor$$
   This creates an unrolled dependency chain of length $\Omega(N)$. In a single forward pass of depth $L < N$, the model cannot compute the carries, achieving empirically $< 5\%$ accuracy.

2. **CoT Decomposition:**
   The model externalizes compute across intermediate token steps:
   - Partial Product 1: $432 \times 7 = 3,024$ ($\approx 4$ tokens)
   - Partial Product 2: $432 \times 60 = 25,920$ ($\approx 4$ tokens)
   - Partial Product 3: $432 \times 500 = 216,000$ ($\approx 4$ tokens)
   - Column Addition: $3,024 + 25,920 + 216,000 = 244,944$ ($\approx 6$ tokens)
   Total CoT Tokens: $K \approx 18$ tokens.
   Each partial product computation requires $\mathcal{O}(1)$ depth. Total FLOPs scale from $2 \cdot N_{\text{params}}$ to $18 \times 2 \cdot N_{\text{params}}$, achieving $100\%$ accuracy. $\blacksquare$

---

### Illustration 5: Faithfulness vs. Unfaithfulness in Chain-of-Thought
**Problem:**
A controversial question in AI safety and interpretability is whether an LLM's Chain-of-Thought is **faithful** (reflecting the true causal reasons for its decision) or **post-hoc rationalization** (fabricating an explanation to justify an intuitive System 1 bias).
1. Define the counterfactual test for CoT faithfulness.
2. Formulate the mathematical condition for unfaithful reasoning (CoT Snooping).

**Solution:**
1. **Counterfactual Faithfulness Test (Turpin et al., 2023):**
   Let $x$ be a problem prompt. Introduce a biasing, task-irrelevant feature $b$ (e.g., adding *"My professor thinks the answer is (A)"*).
   - If the model changes its answer from $y^* = B$ to $\hat{y} = A$, a **faithful** rationale must explicitly cite the biasing feature:
     $$z_{\text{faithful}} = \text{"Based on the professor's opinion, I choose (A)."}$$
   - An **unfaithful** rationale conceals the true causal driver, inventing flawed or hallucinated math steps to post-hoc rationalize $(A)$:
     $$z_{\text{unfaithful}} = \text{"Deriving from equation 1, we clearly see that A is the only valid solution."}$$
2. **Mathematical Condition for Unfaithfulness:**
   Let $Y$ be the generated answer, $Z$ be the rationale, and $B$ be the latent bias variable.
   The rationale $Z$ is unfaithful if:
   $$I(Y; B \mid x) > 0 \quad \text{and} \quad I(Y; B \mid x, Z) > 0$$
   while the causal influence of $Z$ on $Y$ under interventional perturbation $\operatorname{do}(Z = \tilde{z})$ is zero:
   $$P\left( Y \mid x, \operatorname{do}(Z = \tilde{z}) \right) = P(Y \mid x)$$
   In unfaithful models, the final answer was pre-determined in early System 1 layers, and the CoT tokens were generated merely as decorative text. $\blacksquare$

---

### Illustration 6: The "Reasoning Tax" — FLOPs, Latency & KV-Cache Footprint
**Problem:**
An inference cluster serves an 8B parameter model ($d_{\text{model}} = 4096$, $L = 32$, $H_{\text{KV}} = 8$ for GQA, FP16 precision).
Compare serving costs between:
- Standard System 1 direct response: $T_{\text{gen}} = 30$ tokens
- System 2 Long Reasoning Trace (o1/R1 style): $T_{\text{gen}} = 4,000$ tokens
Compute:
1. Computational FLOPs per query.
2. Peak KV cache memory per active user.

**Solution:**
1. **Computational FLOPs:**
   $$\text{FLOPs} \approx 2 \cdot N_{\text{params}} \cdot T_{\text{gen}} = 2 \cdot (8 \times 10^9) \cdot T_{\text{gen}} = 1.6 \times 10^{10} \cdot T_{\text{gen}} \text{ FLOPs}$$
   - System 1: $1.6 \times 10^{10} \times 30 = \mathbf{4.8 \times 10^{11} \text{ FLOPs} \quad (0.48 \text{ TFLOPs})}$
   - System 2: $1.6 \times 10^{10} \times 4000 = \mathbf{6.4 \times 10^{13} \text{ FLOPs} \quad (64.0 \text{ TFLOPs})}$
   *The reasoning trace increases compute by an exact factor of $133.3\times$!*

2. **Peak KV Cache Memory per User:**
   With GQA ($H_{\text{KV}} = 8$, head dimension $d_k = 128$), KV cache size per token across $L=32$ layers in FP16 (2 bytes/param):
   $$\text{Bytes per Token} = 2 \times 2 \times L \times (H_{\text{KV}} \cdot d_k) = 4 \times 32 \times (8 \times 128) = 128 \times 1024 = 131,072 \text{ bytes} = 128 \text{ KB/token}$$
   - System 1 ($T = 30$): $30 \times 128 \text{ KB} = \mathbf{3.84 \text{ MB}}$
   - System 2 ($T = 4000$): $4000 \times 128 \text{ KB} = \mathbf{512.0 \text{ MB}}$
   *Each concurrent reasoning user consumes half a gigabyte of GPU VRAM solely for intermediate thought storage!* $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. OpenAI o1, o3, and the Third Scaling Law (2024–2025)
In September 2024, OpenAI announced **o1** (code-named "Strawberry"), establishing that **inference-time compute is a third orthogonal scaling axis** alongside pre-training compute and dataset size.
- Standard scaling laws (Kaplan et al., Chinchilla) dictate that increasing accuracy requires pre-training larger models on more tokens.
- OpenAI o1 proved that keeping model weights fixed while allowing the model to spend thousands of `<think>` tokens dynamically at test-time achieves superhuman performance on the International Mathematical Olympiad (IMO) qualifier, AIME, and Codeforces.

### 2. DeepSeek-R1 & DeepSeek-R1-Zero: Pure RL Emergence (January 2025)
DeepSeek stunned the AI research community by demonstrating that long Chain-of-Thought reasoning is not an artifact of human prompt engineering or supervised fine-tuning (SFT).
- In **DeepSeek-R1-Zero**, a base language model (DeepSeek-V3-Base) was trained via pure reinforcement learning (GRPO) using only rule-based correctness rewards on math and code.
- Without a single human demonstration, the model autonomously learned to allocate thousands of thought tokens, discovering **self-reflection, error checking, recalculation, and backtracking** (the celebrated *"Aha! Moment"*).

### 3. QwQ-32B & The Open-Source Reasoning Ecosystem
Alibaba's Qwen team released **QwQ-32B**, demonstrating that mid-sized open-weight models (32B parameters) can match or surpass closed 70B+ models on competitive math (MATH-500, AIME) by training models specifically to emit rich, structured reasoning tokens.

### 4. Claude 3.7 Sonnet: Hybrid Thinking Architecture (February 2025)
Anthropic introduced **Claude 3.7 Sonnet**, the first frontier hybrid model allowing users to smoothly dial between pure System 1 (zero reasoning tokens for instantaneous API responses) and full System 2 (up to 64,000 reasoning tokens with strict token budgeting).

---

## 8. Code Implementation & Verification

The accompanying Python script implements and formally verifies all mathematical concepts developed in this chapter:

1. **Part 5 Hand Trace Verification (`verify_part5_hand_trace`):**
   - Implements the exact joint probability computations matching $P(z_A, y_1) = 0.540000$, $P(z_A, y_2) = 0.060000$, $P(z_B, y_1) = 0.060000$, $P(z_B, y_2) = 0.240000$.
   - Verifies marginalization showing $P(y_1 \mid x) = 0.600000$ and $P(y_2 \mid x) = 0.300000$, confirming the $2.0\times$ probability inversion.
   - Executes self-consistency voting over $M=5$ samples, verifying $80.0\%$ consensus for answer $y_1$.
2. **Parity Reasoning Simulation (`simulate_parity_reasoning`):**
   - Compares single-step direct evaluation against an unrolled autoregressive scratchpad on bit sequence $[1, 0, 1, 1, 1, 0, 1, 1]$.
   - Verifies that the intermediate token trajectory exactly matches $[1, 1, 0, 1, 0, 0, 1, 0]$ with final parity $0$.
3. **Self-Consistency Temperature Sampler (`test_self_consistency`):**
   - Implements PyTorch multinomial sampling with temperature scaling $T = 0.5$ across 20 rollouts, verifying that majority voting converges to $y_1$.

### Verification Execution
Execute the verification suite from the repository root:
```bash
python3 Projects/ai_math/13_reasoning_and_test_time_compute/code/01_reasoning_foundations_cot.py
```
Expected output:
```
--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---
Joint P(zA, y1) = 0.540000, P(zA, y2) = 0.060000
Joint P(zB, y1) = 0.060000, P(zB, y2) = 0.240000 -> EXACT MATCH
Marginal P(y1) = 0.600000, Marginal P(y2) = 0.300000 (Ratio: 2.0x) -> EXACT MATCH
Self-consistency vote winner: y1 with 80.0% consensus -> EXACT MATCH

--- 2. Verifying Parity Reasoning: System 1 (Direct) vs System 2 (CoT) ---
Input bits: [1, 0, 1, 1, 1, 0, 1, 1] (Sum = 6)
CoT step-by-step trace: [1, 1, 0, 1, 0, 0, 1, 0]
Final Parity Answer: 0 -> Verified 100% Correct!

--- 3. Testing Self-Consistency Sampler ---
Self-Consistency over 20 rollouts: Winner=y1, Consensus=80.0%
All tests in Chapter 13.1 passed successfully!

🟢 Chapter 13.1 Verification 100% Complete.
```

Accompanying code file:
[`13_reasoning_and_test_time_compute/code/01_reasoning_foundations_cot.py`](./code/01_reasoning_foundations_cot.py)
