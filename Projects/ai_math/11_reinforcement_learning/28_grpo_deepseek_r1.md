# Chapter 28: Group Relative Policy Optimization (GRPO) & Frontier Reasoning (DeepSeek-R1)

---

## 1. Intuition & 101 Motivation

In the previous chapter on **RLHF with PPO**, we saw how large language models can be aligned with human intent. However, applying classical PPO to complex mathematical reasoning, competitive programming, and multi-step scientific problem-solving reveals severe bottlenecks:

1. **The Four-Model Memory Wall:** PPO requires keeping four massive copies of the language model in GPU memory simultaneously:
   - The **Actor** $\pi_\theta$ (trainable)
   - The **Critic** $V_\phi$ (trainable)
   - The **Reference Model** $\pi_{\text{ref}}$ (frozen)
   - The **Reward Model** $r_\psi$ (frozen)
   For a 70B or 671B MoE model, the critic network alone consumes hundreds of gigabytes of VRAM for parameters, Adam optimizer states, and forward activations.
2. **The Brittleness of Learned Reward Models:** Neural reward models are vulnerable to reward hacking, sycophancy, and length bias. In mathematics and coding, we do not need a subjective neural reward model—we have **deterministic truth verifiers** (a Python compiler or symbolic math engine)!
3. **The Value Function Bottleneck:** Training a token-level value critic $V_\phi$ to accurately predict whether a 10,000-token mathematical proof will succeed is notoriously noisy and unstable.

Enter **Group Relative Policy Optimization (GRPO)** (Shao et al., DeepSeekMath 2024; DeepSeek-AI, DeepSeek-R1 2025):
- **Eliminate the Critic Network entirely!** GRPO requires no value network $V_\phi$, cutting GPU memory requirements by more than **$50\%$**.
- For each question/prompt $q$, the model samples a **group of $G$ candidate reasoning paths**:
  $$\mathcal{G} = \{o_1, o_2, \dots, o_G\}$$
- Each completion is evaluated by **rule-based deterministic verifiers** (mathematical correctness + XML tag structure).
- The baseline is computed dynamically **from the group itself** by normalizing rewards:
  $$\hat{A}_i = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}$$
- Completions that outperform their peers receive positive advantages; completions that fail receive negative advantages.

In January 2025, DeepSeek released **DeepSeek-R1-Zero** and **DeepSeek-R1**: trained entirely via GRPO without any human supervised warm-start data, the model autonomously developed **long Chain-of-Thought (CoT)**, self-reflection, and backtracking, rivaling OpenAI's proprietary o1 on competitive benchmarks (AIME, MATH-500, Codeforces)!

```
Prompt q: "Solve 2 + 3 * 4"
       |
       +---> Output 1: "... = 14" (Correct) ------> Reward r_1 = 1.0  \
       +---> Output 2: "... = 14" (Correct) ------> Reward r_2 = 1.0   |--> Group Mean mu = 0.525
       +---> Output 3: "... = 20" (Arithmetic Err) > Reward r_3 = 0.1   |--> Group Std  sigma = 0.476
       +---> Output 4: "broken xml..." -----------> Reward r_4 = 0.0  /
                                                         |
                                                         v
                                  Advantage A_1 = +1.00  (Reinforce!)
                                  Advantage A_2 = +1.00  (Reinforce!)
                                  Advantage A_3 = -0.89  (Suppress!)
                                  Advantage A_4 = -1.10  (Suppress!)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Group Sampling & Verifier Reward Functions

For a given training prompt $q \sim P(Q)$, the old policy $\pi_{\theta_{\text{old}}}$ generates a group of $G$ candidate completions:
$$\mathcal{G} = \{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$$

Each candidate completion $o_i$ is evaluated by two deterministic, rule-based reward functions:

$$r_i = r_{\text{acc}}(q, o_i) + r_{\text{format}}(o_i)$$

1. **Accuracy Verifier ($r_{\text{acc}}$):**
   - **Math:** Extracts the final answer inside `\boxed{...}` and compares it symbolically to ground truth:
     $$r_{\text{acc}}(q, o_i) = \begin{cases} 1.0, & \text{if answer is correct} \\ 0.0, & \text{if answer is incorrect} \end{cases}$$
   - **Code:** Executes generated code against unit test testbeds (compiler execution).
2. **Format Verifier ($r_{\text{format}}$):**
   - Enforces that the model places its reasoning process inside `<think> ... </think>` tags and its final conclusion inside `<answer> ... </answer>` tags:
     $$r_{\text{format}}(o_i) = \begin{cases} 0.1, & \text{if XML tags are properly formatted} \\ 0.0, & \text{otherwise} \end{cases}$$

---

### 2.2 Group Relative Advantage Estimation

Unlike PPO which requires a parametric value network $V_\phi(s)$, GRPO normalizes rewards across the group $\mathcal{G}$:

$$\mu_{\mathcal{G}} = \frac{1}{G} \sum_{j=1}^G r_j$$

$$\sigma_{\mathcal{G}} = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_j - \mu_{\mathcal{G}})^2 + \epsilon}$$

The advantage of completion $o_i$ is defined as the standardized score:

$$\hat{A}_i = \frac{r_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}}$$

#### Mathematical Properties of GRPO Advantages:
1. **Zero-Sum Group Property:**
   $$\sum_{i=1}^G \hat{A}_i = \frac{\sum_{i=1}^G (r_i - \mu_{\mathcal{G}})}{\sigma_{\mathcal{G}}} = \frac{G \mu_{\mathcal{G}} - G \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}} \equiv 0$$
   The group provides its own self-centering baseline!
2. **Dynamic Task Difficulty Adaptation:**
   - On an **easy question** where all $G$ outputs succeed ($r_i = 1.0$), $\sigma_{\mathcal{G}} \to 0$, and no spurious gradient updates are taken.
   - On an **impossible question** where all outputs fail ($r_i = 0.0$), the policy is not destabilized.
   - Updates occur primarily on **frontier questions** with high variance (some successes, some failures), focusing learning where the policy has the highest information gain!

---

### 2.3 The GRPO Objective Function

The policy parameters $\theta$ are trained to maximize the clipped surrogate objective with an analytical token-level KL divergence penalty:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( \rho_{i, t}(\theta) \hat{A}_i, \; \operatorname{clip}(\rho_{i, t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) \right) \right]$$

where the importance sampling probability ratio is:
$$\rho_{i, t}(\theta) = \frac{\pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})}$$

---

### 2.4 Analytical Unbiased KL Divergence Estimator

Instead of computing a sample-based log-ratio approximation $(\ln \pi_\theta - \ln \pi_{\text{ref}})$, DeepSeek-R1 employs the **Schulman unbiased non-negative KL estimator**:

$$D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} - \ln \left( \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} \right) - 1$$

Letting $u = \frac{\pi_{\text{ref}}}{\pi_\theta}$:
$$k(u) = u - \ln u - 1$$

#### Properties:
1. Since $u - 1 \ge \ln u$ for all $u > 0$, $k(u) \ge 0$ strictly.
2. $k(u) = 0$ if and only if $u = 1$ (i.e. $\pi_\theta \equiv \pi_{\text{ref}}$).
3. The estimator has **zero variance** when the policy matches reference, preventing gradient noise from destabilizing training.

---

## 3. Geometric & Physical Interpretation

### 3.1 The Self-Centering Point Cloud
In reward space, the group of completions $\{o_1, \dots, o_G\}$ forms a constellation of outcomes:
```
Reward Axis:
 0.0 ---------- 0.1 ----------------------- 0.525 ----------------------- 1.0
  |              |                            |                            |
[o_4]          [o_3]                        [Mean mu]                 [o_1, o_2]
Broken XML     Wrong Math                                             Correct Proofs
  |              |                                                         |
  v              v                                                         v
A_4 = -1.10    A_3 = -0.89                                               A_1,2 = +1.00
<-- Suppressed Tokens                                       Reinforced Tokens -->
```
By subtracting $\mu_{\mathcal{G}}$, the centroid of the constellation is shifted to zero. The gradient acts as an electrostatic dipole: it attracts the policy distribution toward the successful reasoning trajectory cluster ($+1.00$) while repelling it from the faulty dead ends ($-0.89, -1.10$).

---

## 4. Real-World Analogy: The Math Olympiad Study Group

Imagine a team of 4 students working together to solve an unsolved Olympiad geometry problem:
- **No Teacher / Critic:** There is no omniscient professor hovering over them predicting the probability of success at every step.
- **Trial and Verification:** All 4 students independently draft a proof. At the end, they check the final answer against an official answer key.
- **Relative Comparison:**
  - Alice and Bob find the correct proof.
  - Charlie makes an algebraic error on line 5.
  - David writes an illegible mess.
- The group discusses: *"Alice and Bob's lemma worked ($A > 0$); let us all study and adopt that technique! Charlie's false assumption ($A < 0$) should never be used again."*

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical calculation of **GRPO Group Statistics, Advantages, and Token PPO Gradients** by hand.

---

### 5.1 System & Group Parameters
- **Prompt $q$:** *"Evaluate $2 + 3 \times 4$."*
- **Ground Truth Answer:** $14$
- **Group Size:** $G = 4$ candidate completions
- **Hyperparameters:**
  - PPO clipping threshold: $\epsilon = 0.2000$
  - KL penalty coefficient: $\beta = 0.0500$
- **Candidate Outputs & Evaluated Rewards:**
  - $o_1$: Correct reasoning & answer ($14$), valid format $\implies r_1 = \mathbf{1.0000}$
  - $o_2$: Correct reasoning & answer ($14$), valid format $\implies r_2 = \mathbf{1.0000}$
  - $o_3$: Arithmetic error (answer $20$), valid format $\implies r_3 = \mathbf{0.1000}$ (format only)
  - $o_4$: Wrong answer, broken XML format $\implies r_4 = \mathbf{0.0000}$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $r_i$ | `rewards[i]` | Deterministic scalar reward assigned to candidate completion $o_i$ |
| $\mu_{\mathcal{G}}$ | `group_mean` | Empirical mean reward across the group of $G = 4$ completions |
| $\sigma_{\mathcal{G}}$ | `group_std` | Empirical standard deviation across group completions |
| $\hat{A}_i$ | `group_adv[i]` | Standardized advantage: $(r_i - \mu_{\mathcal{G}}) / \sigma_{\mathcal{G}}$ |
| $\rho_{i, t}$ | `prob_ratio` | Importance sampling token probability ratio: $\pi_\theta / \pi_{\text{old}}$ |
| $D_{\text{KL}}$ | `kl_term` | Schulman unbiased non-negative KL divergence: $u - \ln u - 1$ |

---

### 5.3 Step-by-Step Hand Calculations: Group Advantages

#### Step 1: Compute Group Mean $\mu_{\mathcal{G}}$
$$\mu_{\mathcal{G}} = \frac{r_1 + r_2 + r_3 + r_4}{4} = \frac{1.0000 + 1.0000 + 0.1000 + 0.0000}{4} = \frac{2.1000}{4} = \mathbf{0.525000}$$

#### Step 2: Compute Group Variance & Standard Deviation $\sigma_{\mathcal{G}}$
1. Squared deviations from mean:
   $$(r_1 - \mu)^2 = (1.0000 - 0.5250)^2 = (0.4750)^2 = \mathbf{0.225625}$$
   $$(r_2 - \mu)^2 = (1.0000 - 0.5250)^2 = (0.4750)^2 = \mathbf{0.225625}$$
   $$(r_3 - \mu)^2 = (0.1000 - 0.5250)^2 = (-0.4250)^2 = \mathbf{0.180625}$$
   $$(r_4 - \mu)^2 = (0.0000 - 0.5250)^2 = (-0.5250)^2 = \mathbf{0.275625}$$
2. Sum of squared deviations:
   $$\sum = 0.225625 + 0.225625 + 0.180625 + 0.275625 = \mathbf{0.907500}$$
3. Variance:
   $$\sigma^2 = \frac{0.907500}{4} = \mathbf{0.226875}$$
4. Standard Deviation:
   $$\sigma_{\mathcal{G}} = \sqrt{0.226875} \approx \mathbf{0.47631397}$$

#### Step 3: Compute Standardized Group Advantages $\hat{A}_i$
$$\hat{A}_1 = \frac{1.0000 - 0.5250}{0.47631397} = \frac{+0.4750}{0.47631397} \approx \mathbf{+0.997241}$$
$$\hat{A}_2 = \frac{1.0000 - 0.5250}{0.47631397} = \frac{+0.4750}{0.47631397} \approx \mathbf{+0.997241}$$
$$\hat{A}_3 = \frac{0.1000 - 0.5250}{0.47631397} = \frac{-0.4250}{0.47631397} \approx \mathbf{-0.892269}$$
$$\hat{A}_4 = \frac{0.0000 - 0.5250}{0.47631397} = \frac{-0.5250}{0.47631397} \approx \mathbf{-1.102214}$$

#### Verification of Zero-Sum Property:
$$\sum_{i=1}^4 \hat{A}_i = +0.997241 + 0.997241 - 0.892269 - 1.102214 = \mathbf{0.000000} \quad \checkmark$$

---

### 5.4 Step-by-Step Hand Calculations: Token PPO & KL on Completion $o_1$

Consider a specific token in completion $o_1$:
- Probability under new policy: $\pi_\theta = 0.5500$
- Probability under old policy: $\pi_{\theta_{\text{old}}} = 0.5000$
- Probability under reference policy: $\pi_{\text{ref}} = 0.5000$
- Group Advantage: $\hat{A}_1 = +0.997241$

#### 1. Probability Ratio:
$$\rho = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} = \frac{0.5500}{0.5000} = \mathbf{1.100000}$$

#### 2. PPO Clipped Surrogate:
With $\epsilon = 0.2000$, the clipping interval is $[1 - \epsilon, 1 + \epsilon] = [0.80, 1.20]$.
- Since $\rho = 1.10 \in [0.80, 1.20]$, the ratio is unclipped!
$$\text{Surrogate} = \rho \cdot \hat{A}_1 = 1.100000 \times 0.997241 = \mathbf{1.096965}$$

#### 3. Schulman Analytical KL Penalty:
$$u = \frac{\pi_{\text{ref}}}{\pi_\theta} = \frac{0.5000}{0.5500} = \frac{10}{11} \approx \mathbf{0.909091}$$
$$\ln u = \ln(0.909091) \approx \mathbf{-0.095310}$$
$$D_{\text{KL}} = u - \ln u - 1 = 0.909091 - (-0.095310) - 1.0 = \mathbf{0.004401}$$

#### 4. Net Token Objective:
$$\text{Token Objective} = \text{Surrogate} - \beta \cdot D_{\text{KL}}$$
$$= 1.096965 - 0.0500 \times 0.004401 = 1.096965 - 0.000220 = \mathbf{1.096745}$$

---

### 5.5 Summary Visual Grid: GRPO Optimization Ledger

| Completion $o_i$ | Verifier Acc | Format Tag | Total $r_i$ | Deviation $(r_i - \mu)$ | Advantage $\hat{A}_i$ | Direction | Policy Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$o_1$** | $1.0$ | $0.1$ | **$1.0000$** | $+0.4750$ | $\mathbf{+0.9972}$ | Positive | $\mathbf{\uparrow}$ Strongly Boost Logits |
| **$o_2$** | $1.0$ | $0.1$ | **$1.0000$** | $+0.4750$ | $\mathbf{+0.9972}$ | Positive | $\mathbf{\uparrow}$ Strongly Boost Logits |
| **$o_3$** | $0.0$ | $0.1$ | **$0.1000$** | $-0.4250$ | $\mathbf{-0.8923}$ | Negative | $\mathbf{\downarrow}$ Suppress Wrong Steps |
| **$o_4$** | $0.0$ | $0.0$ | **$0.0000$** | $-0.5250$ | $\mathbf{-1.1022}$ | Negative | $\mathbf{\downarrow}$ Suppress Broken Output |
| **Group Sum** | — | — | $\mu = 0.5250$ | $\sigma = 0.4763$ | $\mathbf{\sum A_i \equiv 0.0000}$ | Centered | **Zero Critic Memory Needed!** |

---

## 6. Solved Illustrations

### Illustration 1: Pure Verification Reward Functions
**Problem:**
Implement a complete Python accuracy verifier for a math question that extracts `\boxed{...}` and compares strings safely.

**Solution:**
```python
import re

def math_accuracy_verifier(completion: str, ground_truth: str) -> float:
    # Extract contents inside \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', completion)
    if not match:
        return 0.0
    extracted_answer = match.group(1).strip()
    return 1.0 if extracted_answer == ground_truth.strip() else 0.0

def format_verifier(completion: str) -> float:
    # Check for valid <think>...</think> and <answer>...</answer> tags
    has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
    return 0.1 if (has_think and has_answer) else 0.0
```
$\blacksquare$

### Illustration 2: The Emergence of the "Aha Moment"
In the DeepSeek-R1 paper, during pure RL exploration on mathematical proofs, the model spontaneously learned to output phrases like:
> *"Wait, let me double check this equation... Ah, wait, that was a mistake! If $x < 0$, the square root is undefined. Let me re-evaluate from line 3..."*

Because GRPO rewards only the final verified conclusion, the policy learned that spending extra tokens generating internal test cases and self-verification drastically increases its probability of landing inside the $+1.0$ reward group!

---

## 7. Deep RL Connection & Modern Applications

- **DeepSeek-R1 & DeepSeek-R1-Zero (DeepSeek-AI, 2025):** The world's premier open-weights reasoning model, achieving a 97.3% score on MATH-500 and 79.8% pass@1 on AIME 2024 through large-scale GRPO.
- **OpenAI o1 & o3-mini (2024–2025):** Built on similar reinforcement learning over reasoning tokens with process/outcome verifiers.
- **Inference-Time Compute Scaling Laws:** Proved that scaling generation compute at test time (longer reasoning traces) produces exponential improvements in mathematical accuracy, reshaping the AI frontier from pre-training scaling to inference-time reasoning RL.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical calculation of group mean $\mu = 0.525000$, standard deviation $\sigma = 0.476314$.
   - Group advantages $\hat{A} = [+0.997241, +0.997241, -0.892269, -1.102214]$ verifying zero-sum property $\sum \hat{A}_i = 0.000000$ to $< 10^{-6}$.
   - Schulman analytical non-negative KL divergence $D_{\text{KL}} = 0.004401$.
2. **Complete GRPO Engine & Rule-Based Verifiers:**
   - Regex-based accuracy and `<think>` format verifiers.
   - Vectorized group-relative advantage estimation.
   - Full GRPO surrogate loss computation with Schulman KL regularizer.

See implementation in:
[`11_reinforcement_learning/code/28_grpo_deepseek_r1.py`](./code/28_grpo_deepseek_r1.py)
