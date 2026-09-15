# 12.9 Post-Training I: Supervised Instruction Fine-Tuning (SFT) & Multi-Turn Trajectory Packing

---

## 1. Intuition & 101 Motivation

Pre-training yields a powerful autocomplete engine. However, raw base models do not follow instructions, often continuing a user's question with more questions or generating endless web-scraped boilerplate.

**Supervised Fine-Tuning (SFT)** is the first and most critical stage of post-training. It conditions the pre-trained distribution on human-like dialogue demonstrations, turning the autocomplete model into an aligned, coherent AI assistant.

At production scale, SFT engineering relies on two foundational techniques:
1. **Prompt-Response Loss Masking:** The model must be penalized **only** for predicting the assistant's answer, not the user's prompt or system instructions. Computing loss on the user's prompt causes the model to memorize questions, destroys zero-shot generalization, and distorts the calibrated prior learned during pre-training.
2. **Multi-Turn Trajectory Packing (Sample Packing):** Instruction datasets have highly variable conversation lengths (e.g., 50 to 4,000 tokens). Traditional batch padding fills shorter sequences with `<pad>` tokens, resulting in **60–80% wasted compute and memory**. Packing concatenates multiple independent dialogues into a single contiguous context window while enforcing **block-diagonal causal attention masks** to prevent information leakage between unrelated conversations.

```
                   CONVERSATION PACKING IN SFT
                   
  Traditional Padded Batch (65% Wasted Compute on PAD)
  Doc 1: [User: Hi | Asst: Hello! | PAD | PAD | PAD | PAD]
  Doc 2: [User: Code a binary search tree in Python...  ]
  
  Packed Sequence with Block-Diagonal Attention (100% Compute Efficiency)
  +-----------------------+-----------------------------+
  |  Doc 1 (Len 4)        |  Doc 2 (Len 6)              |
  |  [U1, A1, EOS]        |  [U2, A2, EOS]              |
  +-----------------------+-----------------------------+
  Pos: 0   1   2             0   1   2   3   4   5
  Mask: Attends only to Doc 1 | Attends only to Doc 2
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The SFT Loss with Target Masking

Consider a multi-turn conversation $C$ tokenized into a sequence of $T$ tokens $X = (x_1, x_2, \dots, x_T)$.
The sequence contains two alternating partitions:
- Prompt tokens $\mathcal{P}$ (System instructions, User queries, Context documents)
- Response tokens $\mathcal{R}$ (Assistant responses, Chain-of-thought rationales)

The Supervised Fine-Tuning objective minimizes the negative log-likelihood computed **strictly over response tokens**:

$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{\sum_{t=1}^T m_t} \sum_{t=1}^T m_t \log P_\theta(x_t \mid x_{<t})$$

where $m_t \in \{0, 1\}$ is a binary mask defined as:
$$m_t = \begin{cases} 1 & \text{if } x_t \in \mathcal{R} \quad (\text{Assistant Token}) \\ 0 & \text{if } x_t \in \mathcal{P} \quad (\text{Prompt Token or Padding}) \end{cases}$$

#### Implementation via PyTorch `ignore_index = -100`:
In standard cross-entropy loss implementations:
$$\text{target}_t = \begin{cases} x_t & \text{if } m_t = 1 \\ -100 & \text{if } m_t = 0 \end{cases}$$
When `F.cross_entropy(logits, targets, ignore_index=-100)` is invoked, PyTorch skips all positions where $\text{target}_t = -100$, completely eliminating them from both the loss calculation and the gradient backpropagation pass.

---

### 2.2 Sequence Packing & Block-Diagonal Attention

In sequence packing (also called Multipack), $K$ independent conversations $\{D_1, D_2, \dots, D_K\}$ with lengths $\{L_1, L_2, \dots, L_K\}$ are packed into a single tensor of length $L_{\text{packed}} = \sum_{k=1}^K L_k \le L_{\text{max}}$.

#### 1. Attention Leakage Catastrophe:
If standard causal masking is used on a packed sequence, tokens in Conversation $k$ will attend to tokens in Conversation $k-1$. For example, a code-generation prompt in Conversation 2 would attend to medical advice generated in Conversation 1, causing catastrophic cross-contamination!

#### 2. Block-Diagonal Causal Attention Mask:
To isolate documents, the attention mask $M \in \mathbb{R}^{L_{\text{packed}} \times L_{\text{packed}}}$ is defined as:

$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \text{ AND } \operatorname{DocID}(i) = \operatorname{DocID}(j) \\ -\infty & \text{otherwise} \end{cases}$$

This transforms the full triangular causal mask into a **block-diagonal staircase of mini causal triangles**:

$$M = \begin{bmatrix}
\begin{matrix} 0 & -\infty \\ 0 & 0 \end{matrix} & \mathbf{-\infty} \\
\mathbf{-\infty} & \begin{matrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{matrix}
\end{bmatrix}$$

#### 3. Position ID Reset:
RoPE positional frequencies encode token distance $|i - j|$. If position IDs were monotonically increasing ($0, 1, 2, 3, 4, \dots$), the first token of Conversation 2 would have $\text{pos} = L_1$, causing RoPE to apply large rotation angles that distort its initial token semantics.
Therefore, position IDs are reset to $0$ at the start of each document:
$$\operatorname{PosID}(t) = t - \text{start\_index}(\operatorname{DocID}(t))$$

---

## 3. Geometric & Physical Interpretation

### 3.1 Block-Diagonal Causal Topology
In standard attention, the connectivity graph is a single directed acyclic tournament.
Under block-diagonal causal masking, the attention adjacency matrix decomposes into the direct sum of disconnected subgraphs:
$$\mathcal{G}_{\text{packed}} = \mathcal{G}_1 \oplus \mathcal{G}_2 \oplus \dots \oplus \mathcal{G}_K$$
Information flows causally within each subgraph $\mathcal{G}_k$, but the mutual information between distinct subgraphs is strictly zero:
$$I(D_k; D_{k'}) = 0 \quad \forall k \neq k'$$

```
   Token Position (i)
        ▲
        │  [Doc 1]
        │   \
        │    \
        │     ▼
        │    [Doc 1]  (Inf Barrier)
        │            [Doc 2]
        │             \
        │              \
        │               ▼
        │              [Doc 2]
        └────────────────────────► Token Position (j)
```

---

## 4. Real-World Analogy: The Examination & The Postal Box

1. **Prompt-Response Masking (The Exam):**
   In a history examination, the question printed on the paper is: *"In what year did the French Revolution begin?"*
   The student writes: *"1789"*.
   The examiner grades the student solely on whether they wrote "1789". If the examiner deducted points because the student did not write out the printed question word-for-word, the student would waste all their energy memorizing questions instead of historical facts.
2. **Trajectory Packing (The Postal Box):**
   Shipping one small envelope in a giant $1\text{ m}^3$ wooden crate wastes massive shipping space ($95\%$ empty space / padding). Trajectory packing places 10 different envelopes into the same wooden crate, separated by cardboard dividers so the letters never touch or mix.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step numerical trace of:
1. **Packed Sequence Layout (2 Documents)**
2. **Target Masking Vector (`ignore_index = -100`)**
3. **Position ID Reset Vector**
4. **$5 \times 5$ Block-Diagonal Attention Mask Matrix**
5. **Loss Computation on Unmasked Positions**

---

### 5.1 Concrete Input Values

We pack two independent conversations into a single context of length $L = 5$:
- **Document 1 (Length $L_1 = 3$):**
  - Token sequence: `[Token 10, Token 20, Token 30]`
  - Roles: `Token 10` (User), `Token 20` (User), `Token 30` (Assistant)
- **Document 2 (Length $L_2 = 2$):**
  - Token sequence: `[Token 40, Token 50]`
  - Roles: `Token 40` (User), `Token 50` (Assistant)

- **Predicted Softmax Probabilities for the Assistant Tokens:**
  - At position index $t = 2$ (predicting `Token 30`): $P(\text{Token } 30) = 0.60$
  - At position index $t = 4$ (predicting `Token 50`): $P(\text{Token } 50) = 0.80$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $x_t$ | `input_ids[t]` | Input token ID at sequence index $t$ |
| $m_t$ | `loss_mask[t]` | Binary indicator: 1 for assistant response, 0 for prompt |
| $\text{target}_t$ | `labels[t]` | Label passed to loss: token ID if $m_t=1$, else $-100$ |
| $\operatorname{PosID}(t)$ | `position_ids[t]` | Reset position index within current document |
| $M_{i, j}$ | `attention_mask[i, j]` | Attention weight logit bias: $0$ (allow) or $-\infty$ (block) |
| $\mathcal{L}$ | `sft_loss` | Mean cross-entropy loss over unmasked response tokens |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Target Mask Vector (`labels`)
- Index 0 (`Token 10`, User): Masked $\implies \text{target}_0 = \mathbf{-100}$
- Index 1 (`Token 20`, User): Masked $\implies \text{target}_1 = \mathbf{-100}$
- Index 2 (`Token 30`, Assistant): Supervised $\implies \text{target}_2 = \mathbf{30}$
- Index 3 (`Token 40`, User): Masked $\implies \text{target}_3 = \mathbf{-100}$
- Index 4 (`Token 50`, Assistant): Supervised $\implies \text{target}_4 = \mathbf{50}$

$$\text{Targets} = [-100, -100, 30, -100, 50]$$

---

#### Step 2: Compute Reset Position IDs
- Document 1 occupies indices $\{0, 1, 2\}$:
  $$\operatorname{PosID}(0) = 0, \quad \operatorname{PosID}(1) = 1, \quad \operatorname{PosID}(2) = 2$$
- Document 2 occupies indices $\{3, 4\}$:
  $$\operatorname{PosID}(3) = 3 - 3 = \mathbf{0}, \quad \operatorname{PosID}(4) = 4 - 3 = \mathbf{1}$$

$$\text{Position IDs} = [0, 1, 2, 0, 1]$$

---

#### Step 3: Construct the $5 \times 5$ Block-Diagonal Attention Mask Matrix $M$
Let $0$ denote unmasked attention and $-\infty$ denote masked attention:

$$M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty & -\infty \\
0 & 0 & 0 & -\infty & -\infty \\
-\infty & -\infty & -\infty & 0 & -\infty \\
-\infty & -\infty & -\infty & 0 & 0
\end{bmatrix}$$

*Verification:*
- At row 3 (Token 40, Doc 2): Columns 0, 1, 2 are $-\infty$. Token 40 **cannot see** any part of Document 1!
- At row 4 (Token 50, Doc 2): Columns 0, 1, 2 are $-\infty$. Token 50 can only see Token 40 and itself.

---

#### Step 4: Compute SFT Cross-Entropy Loss
Only indices $t = 2$ and $t = 4$ have $\text{target} \neq -100$:

1. **Loss at Position 2 (Target 30):**
   $$\ell_2 = -\ln(P(\text{Token } 30)) = -\ln(0.60) = -(-0.5108256) \approx \mathbf{0.510826}$$
2. **Loss at Position 4 (Target 50):**
   $$\ell_4 = -\ln(P(\text{Token } 50)) = -\ln(0.80) = -(-0.2231436) \approx \mathbf{0.223144}$$
3. **Total Mean SFT Loss:**
   $$\mathcal{L}_{\text{SFT}} = \frac{\ell_2 + \ell_4}{2} = \frac{0.510826 + 0.223144}{2} = \frac{0.733970}{2} = \mathbf{0.366985}$$

---

### 5.4 Summary Visual Grid: SFT Trajectory Packing Ledger

| Token Index ($t$) | Token ID ($x_t$) | Role | Doc ID | Position ID | Label ($\text{target}_t$) | $P(x_t)$ | Loss $\ell_t$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 10 | User | Doc 1 | 0 | **-100** | — | Skipped |
| **1** | 20 | User | Doc 1 | 1 | **-100** | — | Skipped |
| **2** | 30 | Assistant | Doc 1 | 2 | **30** | $0.60$ | **$0.510826$** |
| **3** | 40 | User | Doc 2 | 0 | **-100** | — | Skipped |
| **4** | 50 | Assistant | Doc 2 | 1 | **50** | $0.80$ | **$0.223144$** |
| **Mean Loss** | — | — | — | — | — | — | $\mathbf{0.366985}$ |

---

## 6. Solved Illustrations

### Illustration 1: Compute Efficiency Gain of Sample Packing vs. Padding
**Problem:**
A batch consists of 4 conversations with token lengths: $[128, 256, 512, 1024]$.
1. What is the total number of tokens processed under standard batch padding to max length $1024$?
2. What percentage of compute is wasted on padding tokens?
3. If packed into a single sequence of length $1920$, how many FLOPs are saved?

**Solution:**
1. **Standard Padding:**
   Batch size $B = 4$, Max length $L = 1024$.
   $$\text{Total Tokens} = 4 \times 1024 = \mathbf{4,096 \text{ tokens}}$$
   Actual information tokens: $128 + 256 + 512 + 1024 = \mathbf{1,920 \text{ tokens}}$.
2. **Compute Waste:**
   $$\text{Padding Tokens} = 4096 - 1920 = 2,176 \text{ tokens}$$
   $$\text{Wasted Compute Percentage} = \frac{2176}{4096} \times 100\% = \mathbf{53.125\%}$$
3. **Packing Efficiency:**
   Sample packing processes exactly $1,920$ tokens, completely eliminating $2,176$ useless tokens.
   At $6N$ FLOPs/token, for an 8B parameter model ($N = 8 \times 10^9$):
   $$\text{FLOPs Saved} = 6 \times (8 \times 10^9) \times 2176 = \mathbf{1.044 \times 10^{14} \text{ FLOPs per step}} \quad \blacksquare$$

---

## 7. Deep Learning Connection & Modern Applications

- **Axolotl & LLaMA-Factory:** Open-source LLM post-training harnesses rely on Multipack algorithms that bin-pack sequences into fixed context lengths (e.g. 8192) using the first-fit decreasing (FFD) algorithm.
- **FlashAttention-2 Varlen Mode:**
  Rather than instantiating the $N \times N$ block-diagonal mask in VRAM, modern GPU implementations pass a 1D tensor of cumulative sequence lengths (`cu_seqlens = [0, 3, 5]`) directly to FlashAttention-2. The GPU kernel internally loops over documents, completely eliminating attention masking overhead.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact label masking vector with `ignore_index = -100`.
   - Block-diagonal attention mask generation.
   - Position ID resets.
   - Cross-entropy loss matching $0.366985$ to $< 10^{-6}$.
2. **Production-Ready Multi-Turn Trajectory Packer:**
   - ChatML template tokenizer and parser.
   - Dynamic prompt-response mask generation.
   - Block-diagonal attention mask builder.
   - End-to-end PyTorch SFT loss computation.

See implementation in:
[`12_modern_llm_architectures/code/09_instruction_tuning_sft.py`](./code/09_instruction_tuning_sft.py)
