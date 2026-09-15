# Chapter 27: Reinforcement Learning from Human Feedback (RLHF) with PPO

---

## 1. Intuition & 101 Motivation

When a large language model (like GPT-3 or LLaMA) is pre-trained on hundreds of billions of Internet tokens, its objective is strictly **self-supervised next-token prediction**:
$$\max_\theta \sum_{t} \ln P(w_t \mid w_{<t})$$

Because the Internet contains toxicity, falsehoods, hate speech, and unhelpful rants, a raw pre-trained model will happily generate harmful, deceptive, or offensive text if prompted.

While **Supervised Fine-Tuning (SFT)** on curated question-answer pairs helps, human demonstration data is expensive, scarce, and brittle:
- For complex tasks (writing code, summarizing research papers, explaining quantum physics), humans find it **much easier to compare and rank two responses** than to author the perfect response from scratch!

Enter **Reinforcement Learning from Human Feedback (RLHF)** (Christiano et al., NeurIPS 2017; Ouyang et al., InstructGPT 2022):
1. **Step 1 (Preference Data Collection):** The LLM generates two candidate answers $(y_w, y_l)$ for a prompt $x$. A human annotator selects the better answer ($y_w \succ y_l$).
2. **Step 2 (Reward Modeling):** Train a **Reward Model $r_\psi(x, y)$** that scores how helpful, honest, and harmless a response is using the Bradley-Terry preference framework.
3. **Step 3 (RL Fine-Tuning with PPO):** Treat the language model as an **RL Actor**:
   - The **Prompt $x$** is the initial state $s_0$.
   - Each **generated Token $y_t$** is an action $a_t$.
   - The **Environment** is the frozen Reward Model, which delivers a scalar reward when the sentence ends.
   - A **Kullback-Leibler (KL) divergence penalty** anchors the RL policy to the original SFT model, preventing linguistic collapse and reward hacking!

RLHF is the foundational breakthrough that transformed raw predictive models into reliable, conversational AI assistants (ChatGPT, Claude, Gemini).

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Bradley-Terry Preference Model

Suppose human evaluators are presented with prompt $x$ and two candidate responses $y_1, y_2$. The probability that human judges prefer $y_w$ over $y_l$ ($y_w \succ y_l$) is modeled using the **Bradley-Terry logistic model**:

$$P(y_w \succ y_l \mid x) = \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) = \frac{1}{1 + \exp\big( -(r_\psi(x, y_w) - r_\psi(x, y_l)) \big)}$$

where $r_\psi(x, y) \in \mathbb{R}$ is a scalar reward model parameterized by a Transformer network.

#### The Reward Model Loss:
Given a dataset of pairwise preferences $\mathcal{D}_{\text{pref}} = \{(x, y_w, y_l)\}$, the reward model is trained by minimizing the binary cross-entropy loss:

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \ln \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$

---

### 2.2 The RLHF Policy Optimization Objective

Once $r_\psi$ is trained, it is frozen. We wish to optimize the parameters $\theta$ of the language model policy $\pi_\theta(y \mid x)$ to maximize the expected reward while penalizing divergence from the frozen SFT reference model $\pi_{\text{ref}}$:

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\psi(x, y) - \beta D_{\text{KL}}\left( \pi_\theta(\cdot \mid x) \;\Vert\; \pi_{\text{ref}}(\cdot \mid x) \right) \right]$$

where $\beta > 0$ controls the strength of the KL divergence penalty.

---

### 2.3 Token-Level Reward Decomposition

In standard RL, rewards can occur at each step. In language modeling, the reward model $r_\psi(x, y)$ evaluates the *entire completed sequence* $y = (y_1, \dots, y_T)$.
To apply actor-critic temporal-difference learning, the sequence-level reward and KL divergence are decomposed into **per-token rewards $R_t$**:

$$R_t = \begin{cases}
-\beta \left( \ln \pi_\theta(y_t \mid x, y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right), & \text{for } t = 1, \dots, T-1 \\
r_\psi(x, y) - \beta \left( \ln \pi_\theta(y_T \mid x, y_{<T}) - \ln \pi_{\text{ref}}(y_T \mid x, y_{<T}) \right), & \text{for } t = T
\end{cases}$$

Notice:
- For all intermediate tokens $t < T$, the reward is purely the negative token-level KL divergence.
- At the final terminal token $t = T$, the agent receives the full reward model score $r_\psi(x, y)$ plus the final token's KL penalty.

The sum of token rewards reproduces the exact sequence objective:
$$\sum_{t=1}^T R_t = r_\psi(x, y) - \beta \sum_{t=1}^T \left( \ln \pi_\theta(y_t \mid x, y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right)$$

---

### 2.4 Token-Level GAE & PPO-Clipped Update

Four separate models are coordinated during RLHF training:
1. **Actor Policy $\pi_\theta$:** The language model being trained.
2. **Reference Policy $\pi_{\text{ref}}$:** Frozen copy of the SFT model (provides the KL baseline).
3. **Reward Model $r_\psi$:** Frozen preference scoring model.
4. **Critic / Value Network $V_\phi$:** Language model with scalar head predicting expected future token returns from prefix $(x, y_{\le t})$.

#### Token-Level TD Residual & GAE:
With discount $\gamma = 1.0$:
$$\delta_t = R_t + V_\phi(x, y_{\le t+1}) - V_\phi(x, y_{\le t})$$
$$\hat{A}_t = \sum_{l=0}^{T-t-1} \lambda^l \delta_{t+l}$$

#### PPO Actor Loss:
$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$$
$$\mathcal{L}_{\text{actor}}(\theta) = -\frac{1}{T} \sum_{t=1}^T \min \left( r_t(\theta) \hat{A}_t, \; \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right)$$

#### Critic Loss:
$$\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{2 T} \sum_{t=1}^T \left( V_\phi(x, y_{\le t}) - (V_{\text{old}}(x, y_{\le t}) + \hat{A}_t) \right)^2$$

---

## 3. Geometric & Physical Interpretation

### 3.1 The Restoring Spring of the KL Penalty
Think of the policy parameter space as a physical landscape:
```
Reward Surface r_psi(x, y)
 ^
 |             /-- Over-optimized Adversarial Peak (Gibberish / Sycophancy)
 |            /
 |           /     * <-- Optimal Trade-off Point (High Reward + Natural Language)
 |          /     /
 |   [ SFT Reference pi_ref ] <==== KL Spring (Hooke's Law: -beta * grad KL) ===
 |________________________________________________________> Policy Space
```
Without the KL penalty ($\beta = 0$), the policy optimizes ruthlessly against the reward model, finding adversarial blind spots (e.g., repeating the word *"helpful"* 50 times, or sycophantically agreeing with false user statements). The KL penalty acts as a **Hooke's Law elastic spring**, pulling the policy back toward fluent, human-like distribution support.

---

## 4. Real-World Analogy: Speechwriter & Public Opinion Poll

Imagine a politician preparing a major address:
- **The Speechwriter ($\pi_\theta$):** Writes sentences word by word.
- **The Opinion Poll ($r_\psi$):** Measures voter sentiment ($+10$ for inspiring visions, $-10$ for gaffes).
- **The Politician's Core Identity ($\pi_{\text{ref}}$):** The politician's authentic speaking style and principles.
- **The KL Penalty ($\beta$):** If the speechwriter writes pandering, robotic buzzwords just to spike focus-group meters, the politician vetoes it: *"That doesn't sound like me at all!"*

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical calculation of **Token-Level Rewards, TD Errors, and GAE Advantages** by hand for a 2-token generation sequence.

---

### 5.1 System & Sequence Parameters
- **Prompt:** $x$
- **Generated Response:** 2 tokens $y = (y_1, y_2)$
- **KL Coefficient:** $\beta = 0.1000$
- **Discount:** $\gamma = 1.0000$
- **GAE Parameter:** $\lambda = 0.9500$
- **Token Probabilities:**
  - Token 1:
    $$\pi_\theta(y_1 \mid x) = 0.4000, \quad \pi_{\text{ref}}(y_1 \mid x) = 0.5000$$
  - Token 2 (Terminal):
    $$\pi_\theta(y_2 \mid x, y_1) = 0.8000, \quad \pi_{\text{ref}}(y_2 \mid x, y_1) = 0.4000$$
- **Reward Model Output at EOS:**
  $$r_\psi(x, y) = 3.0000$$
- **Critic Value Predictions:**
  - $V(s_1) = 2.5000$ (after seeing prompt $x$)
  - $V(s_2) = 2.8000$ (after seeing $x, y_1$)
  - $V(s_3) = 0.0000$ (terminal state after $y_2$)

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\pi_\theta(y_t), \pi_{\text{ref}}(y_t)$ | `pi_curr[t], pi_ref[t]` | Current and reference policy token probabilities |
| $\Delta \ln \pi_t$ | `log_ratio[t]` | Per-token KL divergence component: $\ln \pi_\theta(y_t) - \ln \pi_{\text{ref}}(y_t)$ |
| $R_t$ | `reward_t[t]` | Decomposed token-level reward incorporating reward model & KL |
| $V(s_t)$ | `critic_v[t]` | Baseline value estimate of expected remaining reward |
| $\delta_t$ | `td_delta[t]` | Token TD error: $R_t + \gamma V(s_{t+1}) - V(s_t)$ |
| $\hat{A}_t$ | `gae_adv[t]` | Generalized Advantage Estimate for token $y_t$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Token Log-Probabilities & KL Penalties
- **Token 1 ($t = 1$):**
  $$\ln \pi_\theta(y_1) = \ln(0.4000) \approx \mathbf{-0.916291}$$
  $$\ln \pi_{\text{ref}}(y_1) = \ln(0.5000) \approx \mathbf{-0.693147}$$
  $$\Delta \ln \pi_1 = -0.916291 - (-0.693147) = \mathbf{-0.223144}$$
  *(Since $\pi_\theta < \pi_{\text{ref}}$, the policy is less confident than reference, so KL penalty is negative, meaning a positive bonus!)*
  $$R_1 = -\beta \cdot \Delta \ln \pi_1 = -0.1000 \times (-0.223144) = \mathbf{+0.022314}$$

- **Token 2 ($t = 2$, Terminal):**
  $$\ln \pi_\theta(y_2) = \ln(0.8000) \approx \mathbf{-0.223144}$$
  $$\ln \pi_{\text{ref}}(y_2) = \ln(0.4000) \approx \mathbf{-0.916291}$$
  $$\Delta \ln \pi_2 = -0.223144 - (-0.916291) = \mathbf{+0.693147}$$
  Terminal reward:
  $$R_2 = r_\psi(x, y) - \beta \cdot \Delta \ln \pi_2 = 3.0000 - 0.1000 \times 0.693147 = 3.0000 - 0.069315 = \mathbf{2.930685}$$

#### Step 2: Compute Token TD Residuals ($\gamma = 1.0$)
- **Residual for Token 1:**
  $$\delta_1 = R_1 + V(s_2) - V(s_1) = 0.022314 + 2.800000 - 2.500000 = 0.022314 + 0.300000 = \mathbf{0.322314}$$

- **Residual for Token 2:**
  $$\delta_2 = R_2 + V(s_3) - V(s_2) = 2.930685 + 0.000000 - 2.800000 = \mathbf{0.130685}$$

#### Step 3: Compute GAE Advantages ($\lambda = 0.95$)
We compute backward from terminal token $t = 2$:
$$\hat{A}_2 = \delta_2 = \mathbf{0.130685}$$
$$\hat{A}_1 = \delta_1 + (\gamma \lambda) \hat{A}_2 = 0.322314 + (1.0 \times 0.95) \times 0.130685$$
$$= 0.322314 + 0.124151 = \mathbf{0.446465}$$

---

### 5.4 Summary Visual Grid: RLHF Token PPO Ledger

| Token | $\pi_\theta$ | $\pi_{\text{ref}}$ | $\Delta \ln \pi$ | RM $r_\psi$ | Token Reward $R_t$ | Critic $V(s)$ | TD Error $\delta_t$ | GAE $\hat{A}_t$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$y_1$** | $0.40$ | $0.50$ | $-0.2231$ | — | **$+0.0223$** | $2.5000$ | **$+0.3223$** | $\mathbf{+0.4465}$ |
| **$y_2$ (EOS)** | $0.80$ | $0.40$ | $+0.6931$ | $3.0000$ | **$+2.9307$** | $2.8000$ | **$+0.1307$** | $\mathbf{+0.1307}$ |
| **$s_3$ (End)** | — | — | — | — | — | $0.0000$ | — | — |

---

## 6. Solved Illustrations

### Illustration 1: Bradley-Terry Pairwise Loss Gradient
**Problem:**
Suppose prompt $x$ produces two completions. The current reward model outputs:
$$r_\psi(x, y_w) = 1.2000, \quad r_\psi(x, y_l) = 0.2000$$
1. Calculate the predicted preference probability $P(y_w \succ y_l)$.
2. Calculate the scalar gradient of the loss with respect to $r_\psi(x, y_w)$ and $r_\psi(x, y_l)$.

**Solution:**
1. **Preference Probability:**
   $$\Delta r = 1.2000 - 0.2000 = 1.0000$$
   $$P(y_w \succ y_l) = \sigma(\Delta r) = \frac{1}{1 + e^{-1.0}} \approx \frac{1}{1 + 0.367879} = \frac{1}{1.367879} \approx \mathbf{0.731059}$$

2. **Loss Gradients:**
   $$\mathcal{L} = -\ln \sigma(\Delta r)$$
   $$\frac{\partial \mathcal{L}}{\partial \Delta r} = -(1 - \sigma(\Delta r)) = -(1.0 - 0.731059) = \mathbf{-0.268941}$$
   Therefore:
   $$\nabla_{r(y_w)} \mathcal{L} = \mathbf{-0.268941} \quad (\text{gradient descent pushes } r(y_w) \text{ UP!})$$
   $$\nabla_{r(y_l)} \mathcal{L} = \mathbf{+0.268941} \quad (\text{gradient descent pushes } r(y_l) \text{ DOWN!}) \quad \blacksquare$$

---

## 7. Deep RL Connection & Modern Applications

- **InstructGPT / ChatGPT (OpenAI, 2022):** Proved that a 1.3B parameter model aligned with RLHF was preferred by humans over a 175B unaligned raw GPT-3 model!
- **Direct Preference Optimization (DPO, Rafailov et al., NeurIPS 2023):**
  Showed mathematically that the optimal policy under the RLHF objective satisfies:
  $$r(x, y) = \beta \ln \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \ln Z(x)$$
  Substituting this directly into the Bradley-Terry loss bypasses training a separate reward model and PPO actor-critic, training directly on preference pairs via implicit language modeling loss!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Bradley-Terry loss & gradients matching $-0.268941, +0.268941$.
   - Token-level KL rewards: $R_1 = 0.022314, R_2 = 2.930685$.
   - TD residuals $\delta_1 = 0.322314, \delta_2 = 0.130685$ and GAE advantages $\hat{A}_1 = 0.446465, \hat{A}_2 = 0.130685$ matching to $< 10^{-6}$.
2. **Complete PyTorch Bradley-Terry Preference Reward Model Trainer.**
3. **Token-Level PPO RLHF Engine:**
   - Vectorized computation of token KL penalties, per-token GAE, and clipped surrogate loss.

See implementation in:
[`11_reinforcement_learning/code/27_rlhf_ppo.py`](./code/27_rlhf_ppo.py)
