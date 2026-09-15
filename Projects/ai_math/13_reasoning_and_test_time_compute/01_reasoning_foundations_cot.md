# 13.1 Foundations of LLM Reasoning: System 1 vs. System 2 & Chain-of-Thought Dynamics

---

## 1. Intuition & 101 Motivation

In cognitive psychology, Daniel Kahneman's dual-process theory bifurcates human thought into two distinct cognitive modes:
- **System 1 (Fast Thinking):** Instinctive, emotional, automatic, and subconscious. Examples: recognizing a familiar face, reading a billboard, or answering $2 + 2 = 4$.
- **System 2 (Slow Thinking):** Deliberative, logical, computational, and effortful. Examples: parallel parking in a tight space, calculating $438 \times 729$, or formulating an algebraic proof.

Standard autoregressive Large Language Models operating under direct greedy decoding are fundamentally **pure System 1 engines**. For every generated token, the model executes exactly **one forward pass** through a static stack of $L$ Transformer layers. Regardless of whether the prompt asks for the capital of France or a proof of Fermat's Last Theorem, the model spends the exact same constant amount of compute ($O(L \cdot d_{\text{model}})$ FLOPs) before outputting the first token!

When forced to solve complex, multi-step problems in a single token generation step, the network suffers an immediate **computational and information bottleneck**: it attempts to collapse a problem requiring sequential state transitions into a single feed-forward residual update.

**Chain-of-Thought (CoT)** (Wei et al., 2022; Nye et al., 2021) unlocks **System 2 reasoning** in Transformers. By prompting or training the model to emit a sequence of intermediate thought tokens $z = (z_1, z_2, \dots, z_K)$ (a scratchpad) before producing the final answer $y$, we externalize computation:
$$\text{Total Compute for Answer } y = K \times \text{Compute per Token}$$

Generating $K$ intermediate tokens allows a model with $L$ layers to effectively execute **$K \times L$ sequential layers of nonlinear transformation**, transforming a shallow circuit into an arbitrarily deep dynamic computational graph.

```
                    SYSTEM 1 vs. SYSTEM 2 IN LLMs
                    
    System 1: Direct Next-Token Prediction (Shallow Circuit)
    Prompt x ──► [Transformer: L Layers] ──► Direct Answer y  (O(L) Compute)
    
    System 2: Chain-of-Thought Reasoning (Deep Dynamic Graph)
    Prompt x ──► [L Layers] ──► Thought z_1
                    │
                    ▼
                 [L Layers] ──► Thought z_2
                    │
                    ▼  ... (K Sequential Steps)
                 [L Layers] ──► Thought z_K ──► Final Answer y (O(K·L) Compute)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Computational Limit of Direct Prediction (System 1)

Let $x = (x_1, \dots, x_N)$ be an input problem prompt and $y$ be the correct answer.
In direct autoregressive decoding:

$$P(y \mid x) = \prod_{t=1}^{|y|} P_\theta(y_t \mid x, y_{<t})$$

#### Theorem 1: Circuit Depth Lower Bound (Merrill & Sabharwal, 2023)
A standard decoder-only Transformer with $L$ layers, constant precision, and polynomial hidden dimension $d$ belongs to the complexity class $\text{TC}^0$ (constant-depth threshold circuits).
Consequently, problems requiring $\Omega(L)$ sequential computational steps—such as graph connectivity, parity evaluation, evaluating boolean formulas, or $N$-step arithmetic—**cannot be computed in a single forward step** for general $N > L$.

#### Theorem 2: The Hidden State Channel Capacity Bottleneck
In direct answer generation, the entire intermediate proof state must be compressed into the hidden state vector of the final prompt token:
$$h_N^{(L)} \in \mathbb{R}^d$$
If floating-point activations are stored in $B$ bits of precision, the maximum information that can be passed from the question to the answer is strictly bounded by:
$$\mathcal{I}_{\text{channel}} \le d \times B \text{ bits}$$
For a 7B model ($d = 4096$, FP16 $B = 16$), the maximum channel capacity is $4096 \times 16 = 65,536 \text{ bits} = 8.192 \text{ KB}$. A complex mathematical proof requiring megabytes of intermediate deduction cannot physically pass through this bottleneck!

---

### 2.2 Rationale-Augmented Formulation (Chain-of-Thought)

Let $z = (z_1, z_2, \dots, z_K) \in \mathcal{Z}$ denote a sequence of $K$ intermediate rationale tokens (the chain of thought, or `<think>` tokens).
The true conditional probability of the final answer $y$ is the **marginal probability** over all possible rationales:

$$P(y \mid x) = \sum_{z \in \mathcal{Z}} P(y, z \mid x) = \sum_{z \in \mathcal{Z}} P(z \mid x) P(y \mid x, z)$$

Expanding via autoregressive factorization:

$$P(z, y \mid x) = \underbrace{\left( \prod_{k=1}^K P_\theta(z_k \mid x, z_{<k}) \right)}_{\text{Rationale Generation Probability } P(z \mid x)} \cdot \underbrace{\left( \prod_{m=1}^{|y|} P_\theta(y_m \mid x, z, y_{<m}) \right)}_{\text{Answer Probability Given Rationale } P(y \mid x, z)}$$

By generating $K$ tokens, the effective channel capacity expands from $d \cdot B$ to:
$$\mathcal{I}_{\text{CoT}} \le K \times d \times B \text{ bits}$$
A 1,000-token reasoning trace provides **$1,000\times$ more storage capacity** for intermediate scratchpad states!

---

### 2.3 Rationale Decoding: MAP vs. Self-Consistency

Evaluating the exact marginal $P(y \mid x) = \sum_{z} P(y, z \mid x)$ is computationally intractable because the rationale space $\mathcal{Z} = \mathcal{V}^K$ contains $|\mathcal{V}|^K$ possibilities (e.g. $100,000^{1000}$).

In practice, two primary decoding strategies are used:

#### 1. Maximum A Posteriori (MAP) Greedy Rationale:
Approximate the sum by the single mode of the distribution:
$$z^* \approx \arg\max_z P_\theta(z \mid x)$$
$$y^* = \arg\max_y P_\theta(y \mid x, z^*)$$
*Limitation:* If greedy search makes an early logical error at step $k=3$, all downstream reasoning $z_{4 \dots K}$ is corrupted.

#### 2. Self-Consistency Majority Voting (Wang et al., 2022):
Instead of relying on a single greedy trajectory, draw $M$ independent reasoning paths using temperature sampling:
$$z^{(1)}, z^{(2)}, \dots, z^{(M)} \sim P_\theta(z \mid x)$$
Extract the candidate final answer $y^{(m)}$ from each path:
$$y^{(m)} = \operatorname{ExtractAnswer}\left( z^{(m)} \right)$$
The optimal answer is determined via **majority vote**:
$$y^* = \arg\max_{y \in \mathcal{Y}} \sum_{m=1}^M \mathbb{I}\left( y^{(m)} = y \right)$$

By the Law of Large Numbers, if the probability of sampling a valid reasoning path is $p > 0.5$, the probability that the majority vote selects the correct answer approaches $1.0$ exponentially as $M \to \infty$:
$$P(\text{Majority Correct}) \ge 1 - \exp\left( -2 M \left(p - \frac{1}{2}\right)^2 \right)$$

---

## 3. Geometric & Physical Interpretation

### 3.1 Geodesics in Semantic Energy Landscapes
Consider the model's loss landscape as a high-dimensional potential energy surface $V(h)$.
- **Direct Answer (System 1):** The model is asked to tunnel directly from the question potential well $x$ to the answer well $y$. Between them lies a massive non-convex energy barrier (the reasoning gap). Tunnelling fails, and the trajectory gets trapped in an intuitive but incorrect local minimum (a common reasoning hallucination).
- **Chain of Thought (System 2):** Intermediate tokens $z_1, z_2, \dots, z_K$ act as physical stepping stones or base camps along a mountain pass. Each token lowers the local energy barrier for the next token, allowing the trajectory to trace a smooth, low-energy **geodesic** over the mountain.

```
   Potential Energy V(h)
        ▲
        │          Direct Prediction: Trapped in Local Minimum!
        │                 ▲
        │                / \  (High Energy Barrier)
        │       x       /   \
        │     (Well)   /  *  \        y (Answer)
        │      \      /  Hallucination  /
        │       \────/                  \────/
        │
        │          Chain of Thought: Stepping Stones (Geodesic)
        │       x ──► z_1 ──► z_2 ──► z_3 ──► y
        └────────────────────────────────────────► State Space
```

---

## 4. Real-World Analogy: Mental Math vs. The Legal Pad

- **System 1 (Direct Prediction):**
  Someone abruptly shouts at you: *"Multiply $739 \times 482$ in your head right now, and say the answer in 2 seconds!"*
  Your brain panics. You might guess `"around 350,000"`, but the exact digits are impossible to compute mentally because human working memory (the prefrontal residual stream) can only hold $7 \pm 2$ items simultaneously.
- **System 2 (Chain-of-Thought):**
  You are given a legal pad and pencil. You write down:
  1. $739 \times 2 = 1,478$
  2. $739 \times 80 = 59,120$
  3. $739 \times 400 = 295,600$
  4. Summing column by column: $1,478 + 59,120 + 295,600 = 356,198$.
  You write the final answer: **356,198**.
  The legal pad is the Chain-of-Thought tokens!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step numerical walkthrough of:
1. **Direct System 1 Error (Greedy Failure)**
2. **CoT Joint Probability Calculations over Multiple Reasoning Paths**
3. **Marginalization across Paths ($P(y \mid x) = \sum P(y, z \mid x)$)**
4. **Self-Consistency Majority Vote Consensus**

---

### 5.1 Concrete Setup & Input Values

- **Problem:** Logic puzzle with question $x$.
- **Possible Answers:**
  - $y_1$: Correct answer
  - $y_2$: Plausible hallucination / distractor
- **Direct System 1 Predictions:**
  - $P(y_1 \mid x) = 0.400000$
  - $P(y_2 \mid x) = 0.600000$
  *(Notice: Direct generation gets it WRONG! It selects $y_2$ because $0.60 > 0.40$).*

- **Sampled Reasoning Paths (CoT):**
  - **Path A ($z_A$):** Rigorous deduction path.
    - Probability of generating this rationale: $P(z_A \mid x) = 0.600000$
    - Conditional probability of answers:
      - $P(y_1 \mid x, z_A) = 0.900000$ (Correct answer given sound reasoning)
      - $P(y_2 \mid x, z_A) = 0.100000$
  - **Path B ($z_B$):** Flawed/confused reasoning path.
    - Probability of generating this rationale: $P(z_B \mid x) = 0.300000$
    - Conditional probability of answers:
      - $P(y_1 \mid x, z_B) = 0.200000$
      - $P(y_2 \mid x, z_B) = 0.800000$ (Incorrect answer reinforced by flawed logic)
  - *(Remaining $P(z_{\text{other}} \mid x) = 0.100000$ leads to unparseable output).*

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $P(z \mid x)$ | `path_prob` | Probability of the model generating reasoning path $z$ |
| $P(y \mid x, z)$ | `cond_ans_prob` | Probability of answer $y$ given reasoning path $z$ |
| $P(z, y \mid x)$ | `joint_prob` | Product $P(z \mid x) \times P(y \mid x, z)$ |
| $P(y \mid x)$ | `marginal_prob` | Sum of joint probabilities over all reasoning paths $\sum_z P(z, y \mid x)$ |
| $\text{Vote}(y)$ | `vote_count` | Number of sampled paths that conclude with answer $y$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Joint Probabilities $P(z, y \mid x) = P(z \mid x) \cdot P(y \mid x, z)$

- **For Path A ($z_A$):**
  $$P(z_A, y_1 \mid x) = P(z_A \mid x) \cdot P(y_1 \mid x, z_A) = 0.600000 \times 0.900000 = \mathbf{0.540000}$$
  $$P(z_A, y_2 \mid x) = P(z_A \mid x) \cdot P(y_2 \mid x, z_A) = 0.600000 \times 0.100000 = \mathbf{0.060000}$$

- **For Path B ($z_B$):**
  $$P(z_B, y_1 \mid x) = P(z_B \mid x) \cdot P(y_1 \mid x, z_B) = 0.300000 \times 0.200000 = \mathbf{0.060000}$$
  $$P(z_B, y_2 \mid x) = P(z_B \mid x) \cdot P(y_2 \mid x, z_B) = 0.300000 \times 0.800000 = \mathbf{0.240000}$$

---

#### Step 2: Marginalize across Reasoning Paths to Find $P(y \mid x)$

- **Total Marginal Probability for Correct Answer $y_1$:**
  $$P(y_1 \mid x) = P(z_A, y_1 \mid x) + P(z_B, y_1 \mid x) = 0.540000 + 0.060000 = \mathbf{0.600000}$$

- **Total Marginal Probability for Incorrect Answer $y_2$:**
  $$P(y_2 \mid x) = P(z_A, y_2 \mid x) + P(z_B, y_2 \mid x) = 0.060000 + 0.240000 = \mathbf{0.300000}$$

#### Result Analysis:
- Without CoT (System 1): $P(y_1 \mid x) = 0.40 < P(y_2 \mid x) = 0.60 \implies$ **Failure!**
- With CoT (System 2): $P(y_1 \mid x) = 0.60 > P(y_2 \mid x) = 0.30 \implies$ **Success!**
The probability of the correct answer **inverted from 40% to 60%**, achieving a $2\times$ preference ratio over the hallucination!

---

#### Step 3: Self-Consistency Majority Vote Hand Trace
Suppose we draw $M = 5$ independent sample rollouts from this distribution:
- Rollout 1: Samples Path A $\to$ concludes $y_1$
- Rollout 2: Samples Path A $\to$ concludes $y_1$
- Rollout 3: Samples Path B $\to$ concludes $y_2$
- Rollout 4: Samples Path A $\to$ concludes $y_1$
- Rollout 5: Samples Path A $\to$ concludes $y_1$

Tallying the votes:
$$\text{Votes}(y_1) = 4, \quad \text{Votes}(y_2) = 1$$
$$\text{Majority Consensus} = \frac{4}{5} = \mathbf{80.0\% \text{ confidence for } y_1} \quad \blacksquare$$

---

### 5.4 Summary Visual Grid: Reasoning Trajectory Ledger

| Path | Rationale Quality | $P(z \mid x)$ | $P(y_1 \mid x, z)$ | $P(y_2 \mid x, z)$ | Joint $P(z, y_1)$ | Joint $P(z, y_2)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Path A** | Rigorous Logic | $0.6000$ | $0.9000$ | $0.1000$ | $\mathbf{0.5400}$ | $0.0600$ |
| **Path B** | Flawed Logic | $0.3000$ | $0.2000$ | $0.8000$ | $0.0600$ | $\mathbf{0.2400}$ |
| **Marginal** | CoT Sum | — | — | — | $\mathbf{0.6000}$ | $\mathbf{0.3000}$ |
| **Direct** | System 1 (No CoT) | — | — | — | $0.4000$ | $\mathbf{0.6000}$ |

---

## 6. Solved Illustrations

### Illustration 1: Parity Problem Complexity & The Necessity of Token Scratchpads
**Problem:**
Let $x = (b_1, b_2, \dots, b_N) \in \{0, 1\}^N$ be a bit string. The task is to compute parity:
$$\operatorname{Parity}(x) = \left( \sum_{i=1}^N b_i \right) \pmod 2$$
1. Why does a standard Transformer with depth $L < N$ fail to compute this in a single forward pass as $N \to \infty$?
2. How does Chain-of-Thought solve it in $O(N)$ token steps?

**Solution:**
1. **Direct Failure:**
   Parity is a dense, non-local function: flipping any single bit $b_i$ flips the final output. In circuit complexity theory, computing the parity of $N$ inputs requires circuit depth that scales with $N$ if the gate fan-in is bounded. A fixed-depth Transformer ($L$ layers) cannot compute unbounded parity because attention averaging dilutes single-bit flips in large contexts.
2. **CoT Solution:**
   By emitting intermediate tokens representing the cumulative parity:
   $$z_1 = b_1, \quad z_2 = (z_1 + b_2) \pmod 2, \quad \dots, \quad z_N = (z_{N-1} + b_N) \pmod 2$$
   Each step $z_k$ only requires computing a local XOR between $z_{k-1}$ and $b_k$, which a single Transformer layer computes trivially ($O(1)$ depth per step). Over $N$ generated tokens, the model computes the exact parity with $100\%$ accuracy! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **OpenAI o1 & o3 ("Strawberry"):** The seminal frontier models proving that inference-time compute scaling is a third scaling axis (alongside pre-training compute and dataset size).
- **DeepSeek-R1 & DeepSeek-R1-Zero:** Demonstrated that long reasoning traces (`<think> ... </think>`) with dynamic backtracking, verification, and self-correction emerge naturally from pure reinforcement learning without supervised demonstration data!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Joint probability table calculation matching $[0.54, 0.06]$ and $[0.06, 0.24]$.
   - Marginalization matching $P(y_1)=0.60$ and $P(y_2)=0.30$.
   - Self-consistency majority voting algorithm.
2. **PyTorch Chain-of-Thought vs. Direct Reasoning Simulator:**
   - Evaluates multi-step parity on variable sequence lengths $N \in [2, 16]$.
   - Shows direct single-step Transformer accuracy degrading to random guess ($50\%$) while CoT retains $100\%$ accuracy.

See implementation in:
[`13_reasoning_and_test_time_compute/code/01_reasoning_foundations_cot.py`](./code/01_reasoning_foundations_cot.py)
