# 13.6 Pure Reinforcement Learning for Reasoning: DeepSeek-R1-Zero & The "Aha" Moment Emergence

---

## 1. Intuition & 101 Motivation

Prior to the release of **DeepSeek-R1** in January 2025, the established dogma of artificial intelligence asserted that Large Language Models could not learn complex reasoning without extensive human-curated demonstration data. Industrial post-training pipelines universally followed a rigid sequence:
1. Collect tens of thousands of human chain-of-thought demonstrations.
2. Supervised Fine-Tuning (SFT) to teach the model how to reason.
3. Reinforcement Learning from Human Feedback (RLHF / PPO) to tune tone and safety.

DeepSeek researchers asked an audacious, foundational research question:
> *Can reasoning capabilities emerge purely from large-scale Reinforcement Learning applied directly to a pre-trained base model, with ZERO human demonstration data?*

To answer this, they created **DeepSeek-R1-Zero**: taking the raw pre-trained `DeepSeek-V3-Base` model and training it directly with **Group Relative Policy Optimization (GRPO)** using **rule-based deterministic verifiers** on mathematical and coding problems.

The result was a historic milestone in machine learning. Without a single human demonstration, the base model autonomously discovered:
1. **Long-Chain Reasoning:** The model learned to expand its thought traces from a few hundred tokens to over 20,000 tokens of internal monologue wrapped in `<think> ... </think>` tags.
2. **Self-Verification & Backtracking:** The model learned to pause midway through a derivation, re-evaluate its equations, check edge cases, and cross-examine its assumptions.
3. **The "Aha! Moment":** The model spontaneously exhibited self-awareness of its own mistakes, generating phrases like:
   > *"Wait, let me double check this equation... Ah, wait, if $x$ is negative, then the square root is undefined! Let me reconsider from the beginning..."*

```
                 THE EMERGENCE OF REASONING IN R1-ZERO
                 
     Pre-Trained Base Model (DeepSeek-V3-Base)
                     │
                     ▼  (Pure RL with GRPO - NO Human SFT Data!)
         Rule-Based Verifiers:
         1. Accuracy Reward (SymPy / Compiler Match)
         2. Format Reward (<think> ... </think> <answer> ... </answer>)
                     │
                     ▼  (After 1,000+ Training Steps)
         AUTONOMOUS EMERGENCE:
         - Extended Chain-of-Thought (20,000+ tokens)
         - Self-Reflection and Double-Checking
         - The "Aha!" Moment (Backtracking from Dead Ends)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Rule-Based Verification Reward Function

Unlike open-ended creative writing, formal reasoning domains (mathematics, algorithmic coding, logic) possess objective, binary ground truths.
DeepSeek-R1-Zero eliminates subjective reward models in favor of a deterministic rule-based scoring function:

$$R(q, o) = R_{\text{accuracy}}(q, o) + R_{\text{format}}(o)$$

#### 1. Accuracy Reward ($R_{\text{accuracy}}$):
Evaluates whether the candidate completion $o$ reached the mathematically sound final answer $y^*$:

$$R_{\text{accuracy}}(q, o) = \begin{cases} 1.0 & \text{if } \operatorname{Verify}(\operatorname{ExtractAnswer}(o), y^*) = \text{True} \\ 0.0 & \text{otherwise} \end{cases}$$

- **Mathematical Problems:** Evaluated using symbolic algebra engines (e.g. SymPy):
  $$\operatorname{Verify}(y, y^*) = \mathbb{I}\left( \operatorname{simplify}(y - y^*) == 0 \right)$$
- **Coding Problems:** Evaluated by executing generated code inside an isolated Docker sandbox against test suites:
  $$\operatorname{Verify}(C, \mathcal{T}) = \prod_{k=1}^{|\mathcal{T}|} \mathbb{I}(\text{Program}(C, \text{input}_k) == \text{expected}_k)$$

#### 2. Format Reward ($R_{\text{format}}$):
Enforces that the model strictly demarcates its intermediate thinking from its final response:

$$R_{\text{format}}(o) = \begin{cases} 0.1 & \text{if } o \text{ matches the regex } \texttt{"^<think>.*?</think><answer>.*?</answer>\$"} \\ 0.0 & \text{otherwise} \end{cases}$$

Total reward spans $R \in [0.0, 1.1]$.

---

### 2.2 Group Relative Policy Optimization (GRPO)

To avoid maintaining a separate Critic Value Model (which would require an additional 671B parameters in GPU VRAM), R1-Zero employs **GRPO** (Chapter 11.28).

For each input question $q$, the policy $\pi_{\theta_{\text{old}}}$ generates a group of $G$ candidate rollouts $\{o_1, o_2, \dots, o_G\}$.
Each rollout receives a scalar reward $R_i = R(q, o_i)$.

The **Group-Normalized Advantage** is computed via group mean and standard deviation:

$$\mu_R = \frac{1}{G} \sum_{j=1}^G R_j, \quad \sigma_R = \sqrt{\frac{1}{G} \sum_{j=1}^G (R_j - \mu_R)^2 + \epsilon}$$
$$A_i = \frac{R_i - \mu_R}{\sigma_R}$$

The GRPO surrogate objective minimizes:

$$\mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( \rho_{i, t} A_i, \, \operatorname{clip}(\rho_{i, t}, 1-\epsilon, 1+\epsilon) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \right]$$

where probability ratio $\rho_{i, t} = \frac{\pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})}$.

---

### 2.3 The Dynamics of Emergence: Length Surge & Language Mixing

During pure RL training, the model exhibits two distinct behavioral phenomena:

1. **The Reasoning Length Surge (Phase Transition):**
   - At step 0, average response length is $\approx 600$ tokens.
   - Around step 500–1000, response length explodes to $\approx 5,000\text{--}15,000$ tokens.
   - *Mathematical Driver:* Generating intermediate verification loops increases the probability of passing the test from $p \approx 0.15$ to $p \approx 0.70$. GRPO heavily rewards these longer, successful paths ($A_i > 0$), reinforcing self-reflection behaviors.
2. **Language Mixing (R1-Zero Limitation):**
   Because pure RL optimizes strictly for mathematical accuracy, the model often mixes languages (e.g. switching between English and Chinese within the same sentence) if it finds that certain concepts are more concisely expressed in another language. DeepSeek-R1 resolved this in later stages by adding language-consistency rewards during multi-stage SFT.

---

## 3. Geometric & Physical Interpretation

### 3.1 Thermodynamic Entropy Expansion and Collapse
The reasoning process in R1-Zero can be understood through non-equilibrium statistical mechanics:
- **Exploration Phase (Entropy Expansion):** During `<think>`, the model increases entropy, radiating candidate hypotheses and exploring multiple energetic microstates.
- **The "Aha" Transition:** When a contradiction is detected, entropy sharply decreases as the model collapses non-viable microstates.
- **Consensus Phase (Entropy Minimization):** The model settles into the ground state, outputting the verified answer in `<answer>`.

```
   State Entropy S(t)
        ▲
        │           Hypothesis Exploration (High Entropy)
        │              /\  /\  /\
        │             /  \/  \/  \
        │            /            \  "Aha! That's wrong!" (Sharp Entropy Collapse)
        │           /              \
        │  *───────/                \────────* (Ground State: <answer>)
        └────────────────────────────────────────► Thinking Token Index (t)
```

---

## 4. Real-World Analogy: The Locked Puzzle Box

Imagine a prisoner given a wooden puzzle box with 10 interlocking sliding tiles:
- **No Instructions (Zero SFT):** Nobody shows the prisoner how to open the box. There are no tutorial videos or expert demonstrations.
- **The Objective Verification:** The box simply clicks open when the correct physical combination is found ($R = 1.0$).
- **The Emergent Behavior:**
  - On Day 1, the prisoner pushes tiles randomly.
  - On Day 5, the prisoner starts using a pencil to sketch the internal gear mechanisms on the cell wall (`<think>`).
  - On Day 10, the prisoner slides tile 4, stops abruptly, mutters: *"Wait, if I slide tile 4 now, tile 7 will jam. Let me slide tile 2 first!"* (The Aha Moment).
  - The prisoner solves the puzzle purely through exploration and verification.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step numerical trace of:
1. **Rule-Based Reward Evaluation across a Group of $G = 4$ Rollouts**
2. **Group Mean and Standard Deviation Calculations**
3. **Normalized Advantage $A_i$ Derivation**
4. **Policy Gradient Direction Analysis**
5. **KL Regularization Penalty Update**

---

### 5.1 Concrete Setup & Input Values

- **Question $q$:** *"Solve for $x$: $2x + 7 = 37$."* (Ground Truth Answer: $x = \mathbf{15}$).
- **Reward Configuration:**
  - Accuracy Reward: $R_{\text{acc}} = 1.0$ if answer is 15, else $0.0$.
  - Format Reward: $R_{\text{fmt}} = 0.1$ if tags `<think>...</think><answer>...</answer>` are strictly valid, else $0.0$.
- **Group of $G = 4$ Rollouts:**
  - **Rollout 1 ($o_1$):**
    `<think>2x + 7 = 37 => 2x = 30 => x = 15</think><answer>15</answer>`
    - Correct answer ($R_{\text{acc}} = 1.0$), Proper format ($R_{\text{fmt}} = 0.1$).
  - **Rollout 2 ($o_2$):**
    `<think>2x + 7 = 37 => 2x = 24 => x = 12</think><answer>12</answer>`
    - Wrong answer ($R_{\text{acc}} = 0.0$), Proper format ($R_{\text{fmt}} = 0.1$).
  - **Rollout 3 ($o_3$):**
    `The answer is 15 because 30/2 = 15.`
    - Correct answer ($R_{\text{acc}} = 1.0$), Missing format tags ($R_{\text{fmt}} = 0.0$).
  - **Rollout 4 ($o_4$):**
    `I think it is 42 maybe?`
    - Wrong answer ($R_{\text{acc}} = 0.0$), Missing format tags ($R_{\text{fmt}} = 0.0$).

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $R_{\text{acc}}$ | `acc_reward` | Deterministic verification reward ($1.0$ or $0.0$) |
| $R_{\text{fmt}}$ | `format_reward` | Regex tag compliance reward ($0.1$ or $0.0$) |
| $R_i$ | `total_reward` | Combined reward $R_{\text{acc}} + R_{\text{fmt}}$ |
| $\mu_R$ | `group_mean` | Average reward across group of $G = 4$ candidates |
| $\sigma_R$ | `group_std` | Population standard deviation of group rewards |
| $A_i$ | `advantage` | Normalized advantage $\frac{R_i - \mu_R}{\sigma_R}$ |
| $\beta_{\text{KL}}$ | `kl_beta` | Penalty coefficient for diverging from reference policy |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Total Rewards $R_i = R_{\text{acc}} + R_{\text{fmt}}$
- **Rollout 1:** $R_1 = 1.0 + 0.1 = \mathbf{1.100000}$
- **Rollout 2:** $R_2 = 0.0 + 0.1 = \mathbf{0.100000}$
- **Rollout 3:** $R_3 = 1.0 + 0.0 = \mathbf{1.000000}$
- **Rollout 4:** $R_4 = 0.0 + 0.0 = \mathbf{0.000000}$

---

#### Step 2: Compute Group Statistics ($\mu_R, \sigma_R$)
1. **Group Mean $\mu_R$:**
   $$\mu_R = \frac{R_1 + R_2 + R_3 + R_4}{4} = \frac{1.10 + 0.10 + 1.00 + 0.00}{4} = \frac{2.20}{4} = \mathbf{0.550000}$$

2. **Group Deviations $(R_i - \mu_R)$:**
   - $R_1 - \mu = 1.10 - 0.55 = +0.55$
   - $R_2 - \mu = 0.10 - 0.55 = -0.45$
   - $R_3 - \mu = 1.00 - 0.55 = +0.45$
   - $R_4 - \mu = 0.00 - 0.55 = -0.55$

3. **Group Variance $\sigma_R^2$:**
   $$\sigma_R^2 = \frac{(+0.55)^2 + (-0.45)^2 + (+0.45)^2 + (-0.55)^2}{4}$$
   $$(+0.55)^2 = 0.3025, \quad (-0.45)^2 = 0.2025$$
   $$\sigma_R^2 = \frac{0.3025 + 0.2025 + 0.2025 + 0.3025}{4} = \frac{1.010000}{4} = \mathbf{0.252500}$$

4. **Group Standard Deviation $\sigma_R$:**
   $$\sigma_R = \sqrt{0.252500} \approx \mathbf{0.50249378}$$

---

#### Step 3: Compute Normalized Advantages $A_i = \frac{R_i - \mu_R}{\sigma_R}$

- **Rollout 1:**
  $$A_1 = \frac{+0.550000}{0.502494} \approx \mathbf{+1.094541}$$
- **Rollout 2:**
  $$A_2 = \frac{-0.450000}{0.502494} \approx \mathbf{-0.895534}$$
- **Rollout 3:**
  $$A_3 = \frac{+0.450000}{0.502494} \approx \mathbf{+0.895534}$$
- **Rollout 4:**
  $$A_4 = \frac{-0.550000}{0.502494} \approx \mathbf{-1.094541}$$

---

#### Step 4: Policy Gradient Direction Analysis
- **Rollout 1 ($A_1 = +1.0945$):** Receives the **strongest positive gradient**. It reinforces both the correct mathematical deduction and the explicit `<think>` tag format!
- **Rollout 3 ($A_3 = +0.8955$):** Rewarded for mathematical correctness, but receives less gradient than Rollout 1 ($+0.8955 < +1.0945$) because it failed format compliance.
- **Rollout 2 ($A_2 = -0.8955$):** Penalized despite having clean formatting, ensuring format compliance never supersedes arithmetic truth.
- **Rollout 4 ($A_4 = -1.0945$):** Receives the **strongest negative penalty**, extinguishing unstructured hallucinations.

---

#### Step 5: KL Regularization Penalty Hand Calculation
To ensure the policy does not collapse during GRPO, a KL penalty is computed per token. Let's compute this for a specific token generated in Rollout 1.
- Reference policy log-probability: $\log \pi_{\text{ref}}(a_t|o_{<t}) = -2.5000$
- Current policy log-probability: $\log \pi_\theta(a_t|o_{<t}) = -1.8000$
- KL Coefficient: $\beta_{\text{KL}} = \mathbf{0.04}$

1. **Compute Log Ratio (KL estimate):**
   $$\text{KL}_{\text{est}} = \log \pi_\theta - \log \pi_{\text{ref}} = -1.8000 - (-2.5000) = \mathbf{+0.7000}$$
2. **Compute KL Penalty:**
   $$\text{Penalty} = \beta_{\text{KL}} \times \text{KL}_{\text{est}} = 0.04 \times 0.7000 = \mathbf{0.0280}$$
3. **Compute Effective Reward:**
   For Rollout 1, base reward was $1.1000$. The effective regularized reward becomes:
   $$R_{\text{eff}} = 1.1000 - 0.0280 = \mathbf{1.0720}$$ $\blacksquare$

---

### 5.4 Summary Visual Grid: R1-Zero GRPO Advantage Ledger

| Rollout | Answer Extracted | Accuracy ($R_{\text{acc}}$) | Format Valid ($R_{\text{fmt}}$) | Total Reward ($R_i$) | Deviation ($R_i - \mu$) | Normalized Advantage ($A_i$) | Policy Update Effect |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$o_1$** | $15$ | $1.0$ | $0.1$ | **$1.1000$** | $+0.5500$ | **$+1.094541$** | **Maximum Boost (Math + Tags)** |
| **$o_2$** | $12$ | $0.0$ | $0.1$ | **$0.1000$** | $-0.4500$ | **$-0.895534$** | Suppressed (Math Error) |
| **$o_3$** | $15$ | $1.0$ | $0.0$ | **$1.0000$** | $+0.4500$ | **$+0.895534$** | Boosted (Math Correct) |
| **$o_4$** | $42$ | $0.0$ | $0.0$ | **$0.0000$** | $-0.5500$ | **$-1.094541$** | **Maximum Suppression** |

---

## 6. Solved Illustrations

### Illustration 1: Transcript Anatomy of the "Aha!" Moment
**Problem:**
Examine the following real excerpt from an early DeepSeek-R1-Zero training checkpoint solving: *"Find the number of real roots of $x^4 + 2x^2 + 1 = 0$."*
```
<think>
Let u = x^2. Then the equation becomes u^2 + 2u + 1 = 0.
Factoring this gives (u + 1)^2 = 0, so u = -1.
Since u = x^2, we have x^2 = -1.
Therefore, x = i and x = -i.
So there are 2 roots!
Wait, wait, wait! The question asks for REAL roots!
Let me re-read carefully: "number of REAL roots".
i and -i are complex numbers, not real!
For real x, x^2 cannot be negative.
Since x^2 >= 0 for all real x, x^2 = -1 has no real solutions!
Ah, so the number of real roots is 0.
Let me double check: x^4 + 2x^2 + 1 = (x^2 + 1)^2 >= 1 > 0 for all real x.
It never crosses the x-axis! Indeed, there are 0 real roots.
</think>
<answer>0</answer>
```
1. Identify the exact turning point where the model backtracks.
2. What reward mechanism encouraged the model to output the word `"Wait"` and reconsider?

**Solution:**
1. **Turning Point:**
   The phrase `"Wait, wait, wait! The question asks for REAL roots!"` marks the self-correction event. The model initially fell into the trap of answering $2$ (complex roots), but an attention head conditioned on the word `"real"` triggered an internal contradiction.
2. **Reward Mechanism Driver:**
   If the model stopped at line 5 and emitted `<answer>2</answer>`, $R_{\text{accuracy}} = 0.0$.
   By continuing to generate thoughts and finding `<answer>0</answer>`, $R_{\text{accuracy}} = 1.0$.
   GRPO reinforced trajectories containing self-checking phrases (`"Wait"`, `"Let me double check"`), transforming hesitation words into functional search operators! $\blacksquare$

---

### Illustration 2: GRPO Group Relative Advantage Computation

**Problem:**
Compute the group relative advantage and policy gradient clipping in a GRPO setup.
A model generates a group of $G=8$ rollouts for a math problem. The verified rewards (correct=1, wrong=0) are:
$\mathbf{r} = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]$

For a correct sample $i$, calculate the policy gradient term $L_i$ where:
$$L_i = \min\left(r_t A_i, \text{clip}(r_t, 0.8, 1.2) A_i\right)$$
Assume the policy ratio $r_t = \frac{\pi_\theta(a_t|o_t)}{\pi_{\text{old}}(a_t|o_t)}$ is $1.1$ for a small policy step and $1.3$ for a large policy step. Show the clip in action.

**Step-by-Step Solution:**

**1. Calculate Group Statistics:**
- Mean: $\mu = \frac{1 + 0 + 1 + 0 + 1 + 0 + 0 + 1}{8} = \frac{4}{8} = \mathbf{0.5000}$
- Variance: $\sigma^2 = \frac{4 \times (1 - 0.5)^2 + 4 \times (0 - 0.5)^2}{8} = \frac{4(0.25) + 4(0.25)}{8} = \frac{2.0}{8} = \mathbf{0.2500}$
- Std Dev: $\sigma = \sqrt{0.2500} = \mathbf{0.5000}$

**2. Calculate Normalized Advantages $A_i$:**
- For correct samples ($r_i = 1.0$): $A_i = \frac{1.0 - 0.5000}{0.5000} = \mathbf{+1.0000}$
- For wrong samples ($r_i = 0.0$): $A_i = \frac{0.0 - 0.5000}{0.5000} = \mathbf{-1.0000}$

**3. Policy Gradient Term with Clipping (Correct Sample, $A_i = 1.0$):**
- **Case A: Small step ($r_t = 1.1$)**
  $$L_i = \min(1.1 \times 1.0, \text{clip}(1.1, 0.8, 1.2) \times 1.0)$$
  $$L_i = \min(1.1, 1.1) = \mathbf{1.1000}$$
  (The step is unclipped, policy moves smoothly).

- **Case B: Large step ($r_t = 1.3$)**
  $$L_i = \min(1.3 \times 1.0, \text{clip}(1.3, 0.8, 1.2) \times 1.0)$$
  $$L_i = \min(1.3, 1.2) = \mathbf{1.2000}$$
  (The clip function activates at $1.2$, preventing the policy from stepping too far, stabilizing training). $\blacksquare$

---

### Illustration 3: Format Reward Function Assignment

**Problem:**
Evaluate three generated responses for format reward and correctness reward.
- `format_reward = +0.1` if response has strictly valid `<think>` and `<answer>` tags.
- `correctness_reward = +1.0` if the extracted answer evaluates to the expected ground truth `5`.

**Responses:**
1. `<think>Let me solve: x=5</think><answer>5</answer>`
2. `The answer is 5`
3. `<think>Let me think...</think>`

**Step-by-Step Solution:**

**Response 1:**
- Has `<think>`? YES.
- Has `</think>`? YES.
- Has `<answer>`? YES.
- Has `</answer>`? YES.
- Format valid? YES. **$R_{\text{format}} = +0.1$**
- Content between answer tags: `5`.
- Matches expected? `5 == 5`. YES. **$R_{\text{correct}} = +1.0$**
- Total Reward = $1.0 + 0.1 = \mathbf{1.1000}$

**Response 2:**
- Format valid? NO (Missing tags). **$R_{\text{format}} = 0.0$**
- Extract answer fallback: Evaluates to `5`. Matches? YES. **$R_{\text{correct}} = +1.0$**
- Total Reward = $1.0 + 0.0 = \mathbf{1.0000}$

**Response 3:**
- Format valid? NO (Missing `<answer>` tags). **$R_{\text{format}} = 0.0$**
- Extract answer: Fails. Matches expected? NO. **$R_{\text{correct}} = 0.0$**
- Total Reward = $0.0 + 0.0 = \mathbf{0.0000}$

The reward function enforces strict format compliance as a continuous soft constraint (worth 0.1) without overriding the primary correctness goal (worth 1.0). $\blacksquare$

---

### Illustration 4: KL Regularization in GRPO

**Problem:**
GRPO relies on KL-divergence regularization to prevent the policy from deviating too far from the reference model.
Given:
- Reference policy log-prob: $\log \pi_{\text{ref}}(a_t|o_t) = -2.5$
- Case A (Moved toward action): Current policy $\log \pi_\theta(a_t|o_t) = -1.8$
- Case B (Moved away from action): Current policy $\log \pi_\theta(a_t|o_t) = -3.5$
- Base correctness reward: $R = 1.0$
- KL coefficient: $\beta_{\text{KL}} = 0.04$

Compute the effective regularized reward for both cases using the per-token KL estimate: $\text{KL} = \log \pi_\theta - \log \pi_{\text{ref}}$.

**Step-by-Step Solution:**

**Case A (Policy moved toward action):**
1. KL estimate: $-1.8 - (-2.5) = -1.8 + 2.5 = \mathbf{0.7000}$ nats.
2. KL penalty: $\beta_{\text{KL}} \times \text{KL} = 0.04 \times 0.7000 = \mathbf{0.0280}$.
3. Effective reward: $r_{\text{eff}} = 1.0 - 0.0280 = \mathbf{0.9720}$.
*(The model is slightly penalized for increasing the probability of this action away from the reference).*

**Case B (Policy moved away from action):**
1. KL estimate: $-3.5 - (-2.5) = -3.5 + 2.5 = \mathbf{-1.0000}$ nats.
2. KL penalty: $\beta_{\text{KL}} \times \text{KL} = 0.04 \times (-1.0000) = \mathbf{-0.0400}$.
3. Effective reward: $r_{\text{eff}} = 1.0 - (-0.0400) = 1.0 + 0.0400 = \mathbf{1.0400}$.
*(The model receives a slight bonus for staying nearer or falling behind the reference, heavily stabilizing the training trajectory).* $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **DeepSeek-R1-Zero (671B MoE):** Achieved an AIME 2024 pass rate of **71.0%** (up from 15.6% in the base model), rivaling OpenAI o1-0912 without a single human demonstration!
- **Kimi k1.5 & QwQ-32B:** Adopted the R1-Zero paradigm of long-trajectory pure reinforcement learning on verifiable domains to build open-weights reasoning systems.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Group total rewards: $[1.1, 0.1, 1.0, 0.0]$.
   - Mean $\mu_R = 0.550000$, standard deviation $\sigma_R = 0.502494$.
   - Normalized advantages matching $+1.094541$, $-0.895534$, $+0.895534$, $-1.094541$ to $< 10^{-6}$.
2. **Rule-Based Reward Engine & GRPO Training Simulator:**
   - SymPy symbolic algebra verifier.
   - Regex format compliance verifier.
   - PyTorch GRPO gradient update step verifying positive policy updates on valid reasoning rollouts.

See implementation in:
[`13_reasoning_and_test_time_compute/code/06_pure_rl_reasoning_r1_zero.py`](./code/06_pure_rl_reasoning_r1_zero.py)
