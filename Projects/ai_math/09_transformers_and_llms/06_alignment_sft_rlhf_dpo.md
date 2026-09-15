# Chapter 9.6: Alignment & Post-Training (SFT, RLHF with PPO, DPO)

---

## 1. Intuition & 101 Motivation

A raw pre-trained foundation model is an astonishing simulator of internet text, but it is **not an assistant**.

If you prompt an unaligned base model with:
> *"How do I bake a chocolate cake?"*

The base model does not necessarily provide a helpful recipe. It might output:
> *"How do I make frosting? How do I clean an oven? Page 42 of Baking Essentials..."*
*(continuing the text as if it were a forum index or bibliography)*, or complete it with an angry internet argument.

To transform a raw probability distribution over internet text into a **Helpful, Honest, and Harmless (HHH)** AI assistant, modern foundation models undergo a rigorous multi-stage **Post-Training Alignment Pipeline**:

```
                       THE POST-TRAINING ALIGNMENT PIPELINE
┌───────────────────────────┐
│     Raw Pre-trained LLM   │  (Predicts next token across trillions of web tokens)
└─────────────┬─────────────┘
              │ Stage 1: Supervised Fine-Tuning (SFT)
              ▼
┌───────────────────────────┐
│       SFT Model           │  (Learns conversational protocol, dialogue format)
└─────────────┬─────────────┘
              │ Stage 2 & 3: Human Feedback & Preference Optimization
              ├─────────────────────────────────────────┐
              ▼ (Traditional Route: RLHF)               ▼ (Modern Route: DPO)
┌───────────────────────────┐             ┌───────────────────────────┐
│   Reward Model (RM)       │             │ Direct Preference Opt     │
│   PPO Policy Optimization │             │ Closed-form implicit RM   │
└─────────────┬─────────────┘             └─────────────┬─────────────┘
              │                                         │
              └────────────────────┬────────────────────┘
                                   ▼
                      Aligned Production Assistant
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Stage 1: Supervised Fine-Tuning (SFT)

The model is fine-tuned on curated demonstrations of high-quality dialogues:
$$\mathcal{D}_{\text{SFT}} = \left\{ (\mathbf{x}^{(i)}, \mathbf{y}^{(i)}) \right\}_{i=1}^N$$
where $\mathbf{x}$ is the user prompt and $\mathbf{y} = (y_1, \dots, y_T)$ is the desired assistant response.

#### SFT Loss with Prompt Masking:
Crucially, the model must **only be penalized for errors on the assistant response**, never on the prompt:
$$\mathcal{L}_{\text{SFT}}(\boldsymbol{\theta}) = - \frac{1}{T} \sum_{t=1}^T \log \pi_{\boldsymbol{\theta}}(y_t \mid \mathbf{x}, y_{<t})$$

In implementation, all prompt tokens in the target tensor are assigned the ignore index `-100` (`CrossEntropyLoss(ignore_index=-100)`), masking their gradients entirely.

---

### 2.2 Stage 2: Reward Modeling & The Bradley-Terry Preference Model

Human annotators cannot easily write optimal mathematical text or code, but they can easily **rank** two candidate responses:
$$y_w \succ y_l \quad (\text{Winner } y_w \text{ is preferred over Loser } y_l \text{ for prompt } x)$$

#### The Bradley-Terry Model (1952):
We assume there exists a latent scalar reward function $r^*(\mathbf{x}, \mathbf{y}) \in \mathbb{R}$ such that the probability that a human prefers $y_w$ over $y_l$ follows the logistic sigmoid of the reward difference:
$$P(y_w \succ y_l \mid \mathbf{x}) = \sigma\left( r^*(\mathbf{x}, y_w) - r^*(\mathbf{x}, y_l) \right) = \frac{1}{1 + e^{-(r^*(\mathbf{x}, y_w) - r^*(\mathbf{x}, y_l))}}$$

#### Reward Model Parameterization & Loss:
We parameterize a Reward Model $r_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{y})$ by replacing the causal LM head with a scalar regression head $\mathbf{w}_r \in \mathbb{R}^{d_{\text{model}} \times 1}$.
Given a dataset of pairwise comparisons $\mathcal{D}_{\text{pref}} = \{(\mathbf{x}, y_w, y_l)\}$, the reward model minimizes binary cross-entropy:
$$\mathcal{L}_{\text{RM}}(\boldsymbol{\psi}) = - \mathbb{E}_{(\mathbf{x}, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma\left( r_{\boldsymbol{\psi}}(\mathbf{x}, y_w) - r_{\boldsymbol{\psi}}(\mathbf{x}, y_l) \right) \right]$$

---

### 2.3 Stage 3: RLHF via Proximal Policy Optimization (PPO) (Christiano et al., 2017; Ouyang et al., 2022)

In standard RLHF, the SFT model initializes the RL policy $\pi_{\boldsymbol{\theta}}$ and acts as a frozen reference policy $\pi_{\text{ref}}$.

#### The Constrained RL Objective:
$$\max_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}, \, \mathbf{y} \sim \pi_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})} \left[ r_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{y}) \right] - \beta \mathbb{D}_{\text{KL}}\left( \pi_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \,\|\, \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \right)$$

#### The Critical Role of the KL Penalty ($\beta$):
1. **Preventing Reward Hacking (Goodhart's Law):**
   Without the KL barrier ($\beta = 0$), the optimizer rapidly exploits blind spots in the neural reward model, discovering nonsensical token strings that produce astronomical predicted rewards.
2. **Preserving Language Fluency:**
   The KL penalty ensures the policy remains anchored within the natural distribution of human language learned during pre-training.

#### Token-Level KL Divergence Formulation:
$$R_{\text{total}}(\mathbf{x}, \mathbf{y}) = r_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{y}) - \beta \sum_{t=1}^T \left( \log \pi_{\boldsymbol{\theta}}(y_t \mid \mathbf{x}, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid \mathbf{x}, y_{<t}) \right)$$
This composite reward is optimized using PPO actor-critic updates.

---

### 2.4 Direct Preference Optimization (DPO) (Rafailov et al., 2023)

PPO is notoriously complex, brittle, and resource-intensive: it requires maintaining **four separate large models in GPU memory simultaneously**:
1. Actor ($\pi_{\boldsymbol{\theta}}$)
2. Critic / Value Network ($V_{\boldsymbol{\phi}}$)
3. Reward Model ($r_{\boldsymbol{\psi}}$)
4. Reference Model ($\pi_{\text{ref}}$)

Rafael Rafailov et al. made a mathematical discovery: **The constrained RL objective can be solved in closed form, completely eliminating the need for a separate reward model or RL training!**

#### Mathematical Derivation:
Recall the RL objective:
$$\max_{\pi} \mathbb{E}_{\mathbf{x}} \left[ \mathbb{E}_{\mathbf{y} \sim \pi} [r(\mathbf{x}, \mathbf{y})] - \beta \mathbb{D}_{\text{KL}}(\pi(\mathbf{y} \mid \mathbf{x}) \,\|\, \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})) \right]$$
Expanding the KL divergence:
$$\sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) r(\mathbf{x}, \mathbf{y}) - \beta \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) \log \frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})}$$
$$= \beta \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) \log \left( \frac{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp\left( \frac{1}{\beta} r(\mathbf{x}, \mathbf{y}) \right)}{\pi(\mathbf{y} \mid \mathbf{x})} \right)$$

Define the partition function:
$$Z(\mathbf{x}) = \sum_{\mathbf{y}} \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp\left( \frac{1}{\beta} r(\mathbf{x}, \mathbf{y}) \right)$$

We can rewrite the objective as:
$$- \beta \mathbb{D}_{\text{KL}}\left( \pi(\mathbf{y} \mid \mathbf{x}) \,\|\, \pi^*(\mathbf{y} \mid \mathbf{x}) \right) + \beta \log Z(\mathbf{x})$$
where the **optimal policy** $\pi^*$ is Gibbs distributed:
$$\pi^*(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp\left( \frac{1}{\beta} r(\mathbf{x}, \mathbf{y}) \right)$$

#### Inverting for the Implicit Reward:
Taking logarithms:
$$\log \pi^*(\mathbf{y} \mid \mathbf{x}) = \log \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) + \frac{1}{\beta} r(\mathbf{x}, \mathbf{y}) - \log Z(\mathbf{x})$$
Rearranging for $r(\mathbf{x}, \mathbf{y})$:
$$r(\mathbf{x}, \mathbf{y}) = \beta \log \frac{\pi^*(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} + \beta \log Z(\mathbf{x})$$

#### Substituting into the Bradley-Terry Preference Objective:
Notice what happens when we compute the reward difference $r(\mathbf{x}, y_w) - r(\mathbf{x}, y_l)$:
$$r(\mathbf{x}, y_w) - r(\mathbf{x}, y_l) = \beta \log \frac{\pi^*(y_w \mid \mathbf{x})}{\pi_{\text{ref}}(y_w \mid \mathbf{x})} - \beta \log \frac{\pi^*(y_l \mid \mathbf{x})}{\pi_{\text{ref}}(y_l \mid \mathbf{x})}$$
The intractable partition function $Z(\mathbf{x})$ **cancels out completely**!

Substituting this directly into the Bradley-Terry log-likelihood yields the **DPO Loss Function**:

$$\mathcal{L}_{\text{DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}}) = - \mathbb{E}_{(\mathbf{x}, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_w \mid \mathbf{x})}{\pi_{\text{ref}}(y_w \mid \mathbf{x})} - \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_l \mid \mathbf{x})}{\pi_{\text{ref}}(y_l \mid \mathbf{x})}\right) \right]$$

---

### 2.5 DPO Gradient Dynamics

Taking the gradient of $\mathcal{L}_{\text{DPO}}$ with respect to policy parameters $\boldsymbol{\theta}$:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{DPO}} = - \beta \, \underbrace{\sigma\left( \hat{r}_{\boldsymbol{\theta}}(\mathbf{x}, y_l) - \hat{r}_{\boldsymbol{\theta}}(\mathbf{x}, y_w) \right)}_{\text{Error Weight } w(\mathbf{x})} \left[ \underbrace{\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_w \mid \mathbf{x})}_{\text{Increase probability of } y_w} - \underbrace{\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_l \mid \mathbf{x})}_{\text{Decrease probability of } y_l} \right]$$
where the implicit reward is $\hat{r}_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{y}) = \beta \log \frac{\pi_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})}$.

#### Insights into the Gradient Mechanism:
1. **Push-Pull Dynamic:** The gradient simultaneously pushes up the log-likelihood of the preferred completion $y_w$ and pulls down the log-likelihood of the rejected completion $y_l$.
2. **Dynamic Error Weighting:** The magnitude is scaled by the sigmoid probability $\sigma(\hat{r}_l - \hat{r}_w)$.
   - If the model already assigns higher implicit reward to $y_w$ ($\hat{r}_w \gg \hat{r}_l$), the weight $\sigma(\hat{r}_l - \hat{r}_w) \to 0$. The gradient vanishes, preventing overfitting.
   - If the model incorrectly prefers the loser ($\hat{r}_l \gg \hat{r}_w$), the weight $\sigma(\hat{r}_l - \hat{r}_w) \to 1$. The model receives a maximal gradient penalty!

---

## 3. Geometric & Algebraic Interpretation

```
                   INFORMATION GEOMETRY OF DPO
                    Probability Simplex Delta^{|V|}
                               ▲
                              / \
                             /   \
                            /  * pi_theta
                           /    \   ^
                          /      \  | Gradient pushes toward y_w,
                         /   *    \ | pulls away from y_l
                        /  pi_ref  \|
                       /____________\
                     y_l             y_w
                     
          KL penalty acts as a Riemannian spring anchoring pi_theta
          within a trusted neighborhood of pi_ref!
```

---

## 4. Real-World Analogy

### The Master Sommelier / Wine Judge
- **Pre-training:** Reading every book on chemistry, agriculture, and fermentation written since 1800.
- **SFT:** Practicing the protocol of wine tasting (holding the glass by the stem, swirling, observing legs, smelling aromas, spitting into the bucket).
- **RLHF (The Old Way):** The apprentice hires an external critic (Reward Model) to rate each wine on a score of 1 to 100, then hires a sports coach (PPO) to optimize their swirling technique to maximize the critic's score. The apprentice might learn to flatter the critic or wear fancy cologne (reward hacking).
- **DPO (The Modern Way):** The master places two glasses on the table side-by-side: *"Glass A is a Grand Cru Bordeaux; Glass B is spoiled vinegar. Sip both. Shift your palate to favor A and reject B."*
  No external critic, no complex scoring system—just direct, stable preference alignment.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete DPO loss and gradient calculation with exact hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in DPO |
| :--- | :--- | :--- | :--- |
| $\mathbf{x}$ | User Prompt | Text string | Input instruction ("Explain gravity simply") |
| $y_w$ | Preferred Completion | Text string | Accurate, clear response |
| $y_l$ | Rejected Completion | Text string | Verbose, hallucinated response |
| $\log \pi_{\text{ref}}(y \mid x)$ | Reference Log-Prob | Scalar | Baseline log-likelihood under frozen SFT model |
| $\log \pi_{\boldsymbol{\theta}}(y \mid x)$ | Policy Log-Prob | Scalar | Current model log-likelihood |
| $\beta$ | KL Temperature | Scalar ($0.5$) | Controls strength of implicit KL anchor |
| $\hat{r}_{\boldsymbol{\theta}}(x, y)$ | Implicit Reward | Scalar | $\beta (\log \pi_{\boldsymbol{\theta}} - \log \pi_{\text{ref}})$ |
| $\Delta r$ | Reward Margin | Scalar | $\hat{r}_{\boldsymbol{\theta}}(x, y_w) - \hat{r}_{\boldsymbol{\theta}}(x, y_l)$ |
| $\mathcal{L}_{\text{DPO}}$ | Objective Loss | Scalar | $-\log \sigma(\Delta r)$ |
| $w_{\text{grad}}$ | Error Weight | Scalar $\in (0, 1)$ | $\sigma(-\Delta r) = \text{gradient scaling factor}$ |

---

### 5.2 Concrete Toy Numbers

Let $\beta = 0.5$.

#### Reference Model Log-Likelihoods (Frozen SFT):
$$\log \pi_{\text{ref}}(y_w \mid \mathbf{x}) = -2.0$$
$$\log \pi_{\text{ref}}(y_l \mid \mathbf{x}) = -2.0$$
*(The reference model initially views both completions as equally likely).*

#### Current Policy Log-Likelihoods (Policy $\pi_{\boldsymbol{\theta}}$):
Suppose the current model currently favors the **wrong** response:
$$\log \pi_{\boldsymbol{\theta}}(y_w \mid \mathbf{x}) = -1.5$$
$$\log \pi_{\boldsymbol{\theta}}(y_l \mid \mathbf{x}) = -1.0$$
*(The model assigns higher probability to the loser $y_l$ than the winner $y_w$!)*

---

### 5.3 Step 1: Compute Log-Ratio and Implicit Rewards

1. **Log-Ratio for Winner ($y_w$):**
   $$\Delta \log \pi(y_w) = \log \pi_{\boldsymbol{\theta}}(y_w) - \log \pi_{\text{ref}}(y_w) = -1.5 - (-2.0) = +0.50$$
   Implicit Reward:
   $$\hat{r}_{\boldsymbol{\theta}}(y_w) = \beta \cdot \Delta \log \pi(y_w) = 0.5 \times 0.50 = \mathbf{0.25}$$

2. **Log-Ratio for Loser ($y_l$):**
   $$\Delta \log \pi(y_l) = \log \pi_{\boldsymbol{\theta}}(y_l) - \log \pi_{\text{ref}}(y_l) = -1.0 - (-2.0) = +1.00$$
   Implicit Reward:
   $$\hat{r}_{\boldsymbol{\theta}}(y_l) = \beta \cdot \Delta \log \pi(y_l) = 0.5 \times 1.00 = \mathbf{0.50}$$

---

### 5.4 Step 2: Compute Reward Margin ($\Delta r$)

$$\Delta r = \hat{r}_{\boldsymbol{\theta}}(y_w) - \hat{r}_{\boldsymbol{\theta}}(y_l) = 0.25 - 0.50 = \mathbf{-0.25}$$

Because the model mistakenly preferred $y_l$, the margin is **negative** ($-0.25$).

---

### 5.5 Step 3: Compute DPO Loss

$$\sigma(\Delta r) = \sigma(-0.25) = \frac{1}{1 + e^{-(-0.25)}} = \frac{1}{1 + e^{0.25}} = \frac{1}{1 + 1.284025} = \frac{1}{2.284025} \approx \mathbf{0.437824}$$

$$\mathcal{L}_{\text{DPO}} = -\log\left( \sigma(\Delta r) \right) = -\log(0.437824) \approx \mathbf{0.825965}$$

---

### 5.6 Step 4: Compute Gradient Error Weight ($w_{\text{grad}}$)

The gradient scaling factor is:
$$w_{\text{grad}} = \sigma(-\Delta r) = \sigma(+0.25) = 1 - \sigma(-0.25) = 1 - 0.437824 = \mathbf{0.562176}$$

The total gradient applied to parameters is:
$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{DPO}} = -(0.5)(0.562176) \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_w) - \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_l) \right]$$
$$= -0.281088 \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_w) + 0.281088 \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_l)$$

When the optimizer takes a gradient descent step $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla \mathcal{L}$:
- It adds $+0.281 \eta \nabla \log \pi_{\boldsymbol{\theta}}(y_w)$, directly **increasing** the winner's likelihood.
- It subtracts $-0.281 \eta \nabla \log \pi_{\boldsymbol{\theta}}(y_l)$, directly **decreasing** the loser's likelihood.

---

## 6. Solved Illustrations

### Illustration 1: Goodhart's Law and Reward Hacking in RLHF

**Problem:**
Goodhart's Law states: *"When a measure becomes a target, it ceases to be a good measure."*
How does reward hacking manifest in an LLM if the KL divergence constraint is disabled ($\beta = 0$)?

**Solution:**
A neural reward model $r_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{y})$ is an imperfect approximation of human preference, typically trained on only $50,000 - 100,000$ comparisons.
If $\beta = 0$, PPO treats the reward model as an absolute ground-truth oracle:
- The policy discovers that repeating specific words (*"certainly!", "delighted!", "moreover"*), producing excessively long outputs (verbosity bias), or using exaggerated flattery (sycophancy) reliably triggers maximum activation in the reward model's linear head.
- The policy degenerates into outputting 1,000-word repetitive essays that receive a reward score of $+99.9$, but are completely unreadable and useless to actual human users.
- The KL penalty $\beta \mathbb{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}} \,\|\, \pi_{\text{ref}})$ prevents this by strictly penalizing the policy if it strays too far from the reference model's natural distribution.

---

### Illustration 2: SFT Loss Target Masking Implementation

**Problem:**
Given prompt token IDs `[101, 2054, 2003]` and response token IDs `[1037, 3899, 102]`, construct the input tensor and target label tensor for standard PyTorch `nn.CrossEntropyLoss(ignore_index=-100)`.

**Solution:**
In causal autoregressive modeling:
- Full sequence: `[101, 2054, 2003, 1037, 3899, 102]`
- Input tokens (`x`): `[101, 2054, 2003, 1037, 3899]` (all tokens except the last).
- Target labels (`y`):
  Prompt tokens are masked with `-100`, while response tokens retain their true IDs:
  $$\mathbf{y}_{\text{target}} = [-100, \, -100, \, -100, \, 1037, \, 3899, \, 102]$$
  *(shifted accordingly)*:
  $$\text{Input:  } [101, 2054, 2003, 1037, 3899]$$
  $$\text{Target: } [-100, -100, 1037, 3899, 102]$$
Only the predictions for tokens `1037`, `3899`, and `102` contribute to the loss and receive non-zero gradients.

---

## 7. Deep Learning Connection & Application

### Comparison of Modern Alignment Paradigms

| Feature | Supervised Fine-Tuning (SFT) | RLHF with PPO | Direct Preference Optimization (DPO) |
| :--- | :--- | :--- | :--- |
| **Input Data** | Demonstration prompts + responses | Preferences: $(x, y_w, y_l)$ | Preferences: $(x, y_w, y_l)$ |
| **Objective** | Maximum Likelihood Estimation (MLE) | Constrained Policy Gradient | Exact Implicit Reward Optimization |
| **Number of Models in RAM** | 1 (Actor) | **4** (Actor, Critic, RM, Ref) | **2** (Actor, Ref) |
| **Training Stability** | Extremely stable | Notoriously unstable, hyperparameter-sensitive | **Extremely stable (Standard Cross-Entropy)** |
| **Industry Adoption** | Universal step 1 | OpenAI (GPT-4), Anthropic | Meta (LLaMA 3), Mistral, Qwen |

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `sft_loss_with_masking`: Production-grade SFT cross-entropy loss utilizing `-100` label masking.
2. `bradley_terry_reward_loss`: Pairwise binary cross-entropy reward model objective.
3. `dpo_loss`: Vectorized Direct Preference Optimization loss computing implicit rewards and dynamic error weighting.
4. Exact numerical verification of the Part 5 Visual Grid hand arithmetic.
5. End-to-end mini-training experiment proving that DPO successfully reverses an inverted preference in a single optimization step.

See implementation in:
[`09_transformers_and_llms/code/06_alignment_sft_rlhf_dpo.py`](./code/06_alignment_sft_rlhf_dpo.py)
