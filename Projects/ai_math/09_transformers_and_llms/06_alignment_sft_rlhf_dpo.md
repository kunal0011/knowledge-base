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
$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{T} \sum_{t=1}^T \log \pi_{\theta}(y_t \mid \mathbf{x}, y_{<t})$$

In implementation, all prompt tokens in the target tensor are assigned the ignore index `-100` (`CrossEntropyLoss(ignore_index=-100)`), masking their gradients entirely.

---

### 2.2 Stage 2: Reward Modeling & The Bradley-Terry Preference Model

Human annotators cannot easily write optimal mathematical text or code, but they can easily **rank** two candidate responses:
$$y_w \succ y_l \quad (\text{Winner } y_w \text{ is preferred over Loser } y_l \text{ for prompt } x)$$

#### The Bradley-Terry Model (1952):
We assume there exists a latent scalar reward function $r^*(\mathbf{x}, \mathbf{y}) \in \mathbb{R}$ such that the probability that a human prefers $y_w$ over $y_l$ follows the logistic sigmoid of the reward difference:
$$P(y_w \succ y_l \mid \mathbf{x}) = \sigma\left( r^*(\mathbf{x}, y_w) - r^*(\mathbf{x}, y_l) \right) = \frac{1}{1 + e^{-(r^*(\mathbf{x}, y_w) - r^*(\mathbf{x}, y_l))}}$$

#### Reward Model Parameterization & Loss:
We parameterize a Reward Model $r_{\psi}(\mathbf{x}, \mathbf{y})$ by replacing the causal LM head with a scalar regression head $\mathbf{w}_r \in \mathbb{R}^{d_{\text{model}} \times 1}$.
Given a dataset of pairwise comparisons $\mathcal{D}_{\text{pref}} = \{(\mathbf{x}, y_w, y_l)\}$, the reward model minimizes binary cross-entropy:
$$\mathcal{L}_{\text{RM}}(\psi) = - \mathbb{E}_{(\mathbf{x}, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma\left( r_{\psi}(\mathbf{x}, y_w) - r_{\psi}(\mathbf{x}, y_l) \right) \right]$$

---

### 2.3 Stage 3: RLHF via Proximal Policy Optimization (PPO) (Christiano et al., 2017; Ouyang et al., 2022)

In standard RLHF, the SFT model initializes the RL policy $\pi_{\theta}$ and acts as a frozen reference policy $\pi_{\text{ref}}$.

#### The Constrained RL Objective:
$$\max_{\theta} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}, \, \mathbf{y} \sim \pi_{\theta}(\cdot \mid \mathbf{x})} \left[ r_{\psi}(\mathbf{x}, \mathbf{y}) \right] - \beta \mathbb{D}_{\text{KL}}\left( \pi_{\theta}(\mathbf{y} \mid \mathbf{x}) \,\|\, \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \right)$$

#### The Critical Role of the KL Penalty ($\beta$):
1. **Preventing Reward Hacking (Goodhart's Law):**
   Without the KL barrier ($\beta = 0$), the optimizer rapidly exploits blind spots in the neural reward model, discovering nonsensical token strings that produce astronomical predicted rewards.
2. **Preserving Language Fluency:**
   The KL penalty ensures the policy remains anchored within the natural distribution of human language learned during pre-training.

#### Token-Level KL Divergence Formulation:
$$R_{\text{total}}(\mathbf{x}, \mathbf{y}) = r_{\psi}(\mathbf{x}, \mathbf{y}) - \beta \sum_{t=1}^T \left( \log \pi_{\theta}(y_t \mid \mathbf{x}, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid \mathbf{x}, y_{<t}) \right)$$
This composite reward is optimized using PPO actor-critic updates.

---

### 2.4 Direct Preference Optimization (DPO) (Rafailov et al., 2023)

PPO is notoriously complex, brittle, and resource-intensive: it requires maintaining **four separate large models in GPU memory simultaneously**:
1. Actor ($\pi_{\theta}$)
2. Critic / Value Network ($V_{\phi}$)
3. Reward Model ($r_{\psi}$)
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

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(\mathbf{x}, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_w \mid \mathbf{x})}{\pi_{\text{ref}}(y_w \mid \mathbf{x})} - \beta \log \frac{\pi_{\theta}(y_l \mid \mathbf{x})}{\pi_{\text{ref}}(y_l \mid \mathbf{x})}\right) \right]$$

---

### 2.5 DPO Gradient Dynamics

Taking the gradient of $\mathcal{L}_{\text{DPO}}$ with respect to policy parameters $\theta$:

$$\nabla_{\theta} \mathcal{L}_{\text{DPO}} = - \beta \, \underbrace{\sigma\left( \hat{r}_{\theta}(\mathbf{x}, y_l) - \hat{r}_{\theta}(\mathbf{x}, y_w) \right)}_{\text{Error Weight } w(\mathbf{x})} \left[ \underbrace{\nabla_{\theta} \log \pi_{\theta}(y_w \mid \mathbf{x})}_{\text{Increase probability of } y_w} - \underbrace{\nabla_{\theta} \log \pi_{\theta}(y_l \mid \mathbf{x})}_{\text{Decrease probability of } y_l} \right]$$
where the implicit reward is $\hat{r}_{\theta}(\mathbf{x}, \mathbf{y}) = \beta \log \frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})}$.

#### Insights into the Gradient Mechanism:
1. **Push-Pull Dynamic:** The gradient simultaneously pushes up the log-likelihood of the preferred completion $y_w$ and pulls down the log-likelihood of the rejected completion $y_l$.
2. **Dynamic Error Weighting:** The magnitude is scaled by the sigmoid probability $\sigma(\hat{r}_l - \hat{r}_w)$.
   - If the model already assigns higher implicit reward to $y_w$ ($\hat{r}_w \gg \hat{r}_l$), the weight $\sigma(\hat{r}_l - \hat{r}_w) \to 0$. The gradient vanishes, preventing overfitting.
   - If the model incorrectly prefers the loser ($\hat{r}_l \gg \hat{r}_w$), the weight $\sigma(\hat{r}_l - \hat{r}_w) \to 1$. The model receives a maximal gradient penalty!

---

### 2.6 Deep Derivation 9.6.1: Full Analytical Derivation of PPO Clipped Surrogate and GAE for Language Models

#### Context and Setup
In RLHF, an LLM generates a trajectory of tokens $\mathbf{y} = (y_1, \dots, y_T)$ given prompt $\mathbf{x}$.
A scalar reward model assigns terminal score $r_{\psi}(\mathbf{x}, \mathbf{y})$ upon generating the final end-of-sequence token.

#### Token-Level MDP Formulation:
1. **State:** $s_t = (\mathbf{x}, y_{<t})$ (the prompt concatenated with all tokens generated so far).
2. **Action:** $a_t = y_t \in \mathcal{V}$ (the selected token from vocabulary $\mathcal{V}$).
3. **Transition:** Deterministic string concatenation $s_{t+1} = (s_t, a_t)$.
4. **Token Reward:**
   $$R_t = \begin{cases} -\beta \left( \log \pi_{\theta}(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t) \right) & \text{for } t < T \\ r_{\psi}(\mathbf{x}, \mathbf{y}) - \beta \left( \log \pi_{\theta}(y_T \mid s_T) - \log \pi_{\text{ref}}(y_T \mid s_T) \right) & \text{for } t = T \end{cases}$$

#### Generalized Advantage Estimation (GAE):
A learned Value Network (Critic) $V_{\phi}(s_t)$ predicts the expected cumulative discounted future reward.
The temporal difference (TD) residual at step $t$ is:
$$\delta_t^V = R_t + \gamma V_{\phi}(s_{t+1}) - V_{\phi}(s_t)$$
The $\text{GAE}(\gamma, \lambda)$ advantage function is defined as the exponentially weighted sum of future TD residuals:
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T - t} (\gamma \lambda)^l \delta_{t+l}^V$$
- In language modeling, typically $\gamma = 1.0$ and $\lambda \in [0.95, 1.0]$.
- Expanding recursively backwards from the final token $T$:
  $$\hat{A}_T = \delta_T^V = R_T - V_{\phi}(s_T)$$
  $$\hat{A}_t = \delta_t^V + \gamma \lambda \hat{A}_{t+1} \quad (\forall t = T-1, \dots, 1)$$

#### The PPO Clipped Surrogate Objective:
To prevent destructively large policy updates, PPO defines the probability ratio:
$$r_t(\theta) = \frac{\pi_{\theta}(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)}$$
and optimizes the pessimistic clipped surrogate loss:
$$\mathcal{L}_t^{\text{CLIP}}(\theta) = \min\left( r_t(\theta) \hat{A}_t, \, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)$$
- If $\hat{A}_t > 0$ (action was better than expected), $r_t$ is pushed upward, but clipped at $1 + \epsilon$ to prevent over-optimizing lucky samples.
- If $\hat{A}_t < 0$ (action was worse than expected), $r_t$ is pushed downward, but clipped at $1 - \epsilon$. $\blacksquare$

---

### 2.7 Deep Derivation 9.6.2: Variational Derivation of DPO via Convex Duality

#### The Primal Constrained Problem
Consider the general KL-regularized contextual bandit objective for prompt $\mathbf{x}$:
$$\max_{\pi} \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) r(\mathbf{x}, \mathbf{y}) - \beta \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) \log \frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} \quad \text{s.t.} \quad \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) = 1$$

#### Theorem: Global Optimum and Equivalence to Bradley-Terry
The unconstrained variational optimum is uniquely achieved by the Gibbs distribution:
$$\pi^*(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp\left( \frac{1}{\beta} r(\mathbf{x}, \mathbf{y}) \right)$$
where $Z(\mathbf{x}) = \sum_{\mathbf{y}} \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp\left( \frac{1}{\beta} r(\mathbf{x}, \mathbf{y}) \right)$.

#### Proof via Fenchel Duality:
1. **Lagrangian Formulation:**
   $$\mathcal{L}(\pi, \lambda) = \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) r(\mathbf{x}, \mathbf{y}) - \beta \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) \log \frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} + \lambda \left( 1 - \sum_{\mathbf{y}} \pi(\mathbf{y} \mid \mathbf{x}) \right)$$

2. **First-Order Optimality Condition:**
   Taking the functional derivative w.r.t. $\pi(\mathbf{y} \mid \mathbf{x})$:
   $$\frac{\partial \mathcal{L}}{\partial \pi(\mathbf{y} \mid \mathbf{x})} = r(\mathbf{x}, \mathbf{y}) - \beta \left( \log \frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} + 1 \right) - \lambda = 0$$
   Rearranging for the log-ratio:
   $$\log \frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} = \frac{r(\mathbf{x}, \mathbf{y}) - \lambda}{\beta} - 1$$
   Exponentiating both sides:
   $$\pi^*(\mathbf{y} \mid \mathbf{x}) = \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp\left( \frac{r(\mathbf{x}, \mathbf{y})}{\beta} \right) \cdot \exp\left( -\frac{\lambda}{\beta} - 1 \right)$$

3. **Enforcing the Probability Simplex Sum:**
   Summing over all $\mathbf{y} \in \mathcal{Y}$:
   $$\sum_{\mathbf{y}} \pi^*(\mathbf{y} \mid \mathbf{x}) = \exp\left( -\frac{\lambda}{\beta} - 1 \right) \sum_{\mathbf{y}} \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp\left( \frac{r(\mathbf{x}, \mathbf{y})}{\beta} \right) = 1$$
   Therefore:
   $$\exp\left( -\frac{\lambda}{\beta} - 1 \right) = \frac{1}{Z(\mathbf{x})}$$
   which proves the Gibbs form.

4. **Cancellation of $Z(\mathbf{x})$ in the Bradley-Terry Model:**
   The Bradley-Terry preference probability is:
   $$P(y_w \succ y_l \mid \mathbf{x}) = \sigma(r(\mathbf{x}, y_w) - r(\mathbf{x}, y_l))$$
   Substituting $r(\mathbf{x}, \mathbf{y}) = \beta \log \frac{\pi^*(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} + \beta \log Z(\mathbf{x})$:
   $$r(\mathbf{x}, y_w) - r(\mathbf{x}, y_l) = \left( \beta \log \frac{\pi^*(y_w \mid \mathbf{x})}{\pi_{\text{ref}}(y_w \mid \mathbf{x})} + \beta \log Z(\mathbf{x}) \right) - \left( \beta \log \frac{\pi^*(y_l \mid \mathbf{x})}{\pi_{\text{ref}}(y_l \mid \mathbf{x})} + \beta \log Z(\mathbf{x}) \right)$$
   $$= \beta \log \frac{\pi^*(y_w \mid \mathbf{x})}{\pi_{\text{ref}}(y_w \mid \mathbf{x})} - \beta \log \frac{\pi^*(y_l \mid \mathbf{x})}{\pi_{\text{ref}}(y_l \mid \mathbf{x})}$$
   The normalization constant $Z(\mathbf{x})$ cancels unconditionally. Parameterizing $\pi^*$ directly with policy $\pi_{\theta}$ bypasses RL entirely. $\blacksquare$

---

### 2.8 Deep Derivation 9.6.3: Beyond DPO: Identity Preference Optimization (IPO) and Kahneman-Tversky Optimization (KTO)

#### 1. The DPO Overfitting Dilemma & IPO (Azar et al., 2023)
In DPO, if the dataset contains deterministic pairs ($P(y_w \succ y_l) = 1$), the cross-entropy loss drives the log-ratio difference $\beta (\log \pi(y_w) - \log \pi(y_l)) \to +\infty$.
This causes the policy to collapse: $\pi_{\theta}(y_l) \to 0$ and likelihood of $y_w$ drifts into degenerate low-entropy modes.

**Identity Preference Optimization (IPO):**
Azar et al. bypass the Bradley-Terry non-linear link entirely, minimizing a regularized quadratic loss on the implicit reward margin:
$$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(\mathbf{x}, y_w, y_l)} \left[ \left( \log \frac{\pi_{\theta}(y_w \mid \mathbf{x})}{\pi_{\text{ref}}(y_w \mid \mathbf{x})} - \log \frac{\pi_{\theta}(y_l \mid \mathbf{x})}{\pi_{\text{ref}}(y_l \mid \mathbf{x})} - \frac{\tau}{2} \right)^2 \right]$$
where $\tau$ target controls the margin gap. IPO guarantees that the implicit reward gap never grows unbounded, preventing over-optimization.

#### 2. Kahneman-Tversky Optimization (KTO) (Ethayarajh et al., 2024)
DPO requires **paired preferences** $(x, y_w, y_l)$. In real-world products, user feedback is overwhelmingly **unpaired binary feedback** (e.g., thumbs-up or thumbs-down on single outputs).

KTO builds on Daniel Kahneman and Amos Tversky's **Prospect Theory**:
- Humans evaluate outcomes relative to a reference point with **loss aversion** (losses hurt more than equal gains feel good).
- Let $z(\mathbf{x}, \mathbf{y}) = \beta \log \frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})}$ be the implicit reward, and $z_{\text{ref}} = \mathbb{E}_{\mathbf{x}, \mathbf{y}}[z(\mathbf{x}, \mathbf{y})]$ be the reference anchor.
- The KTO loss is:
  $$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{y}} \left[ w(\mathbf{y}) \cdot \sigma\left( \lambda_{\mathbf{y}} \left( z(\mathbf{x}, \mathbf{y}) - z_{\text{ref}} \right) \right) \right]$$
  where $\lambda_{y} = 1$ for desirable outputs ($y \in \mathcal{Y}_{\text{desirable}}$) and $\lambda_y = -1$ with higher penalty weight $\lambda_{\text{loss}} > 1$ for rejected outputs.
KTO matches or exceeds DPO performance while learning directly from cheap, natural thumbs-up/down signals without requiring artificial pairwise contrast. $\blacksquare$

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
| $\log \pi_{\theta}(y \mid x)$ | Policy Log-Prob | Scalar | Current model log-likelihood |
| $\beta$ | KL Temperature | Scalar ($0.5$) | Controls strength of implicit KL anchor |
| $\hat{r}_{\theta}(x, y)$ | Implicit Reward | Scalar | $\beta (\log \pi_{\theta} - \log \pi_{\text{ref}})$ |
| $\Delta r$ | Reward Margin | Scalar | $\hat{r}_{\theta}(x, y_w) - \hat{r}_{\theta}(x, y_l)$ |
| $\mathcal{L}_{\text{DPO}}$ | Objective Loss | Scalar | $-\log \sigma(\Delta r)$ |
| $w_{\text{grad}}$ | Error Weight | Scalar $\in (0, 1)$ | $\sigma(-\Delta r) = \text{gradient scaling factor}$ |

---

### 5.2 Concrete Toy Numbers

Let $\beta = 0.5$.

#### Reference Model Log-Likelihoods (Frozen SFT):
$$\log \pi_{\text{ref}}(y_w \mid \mathbf{x}) = -2.0$$
$$\log \pi_{\text{ref}}(y_l \mid \mathbf{x}) = -2.0$$
*(The reference model initially views both completions as equally likely).*

#### Current Policy Log-Likelihoods (Policy $\pi_{\theta}$):
Suppose the current model currently favors the **wrong** response:
$$\log \pi_{\theta}(y_w \mid \mathbf{x}) = -1.5$$
$$\log \pi_{\theta}(y_l \mid \mathbf{x}) = -1.0$$
*(The model assigns higher probability to the loser $y_l$ than the winner $y_w$!)*

---

### 5.3 Step 1: Compute Log-Ratio and Implicit Rewards

1. **Log-Ratio for Winner ($y_w$):**
   $$\Delta \log \pi(y_w) = \log \pi_{\theta}(y_w) - \log \pi_{\text{ref}}(y_w) = -1.5 - (-2.0) = +0.50$$
   Implicit Reward:
   $$\hat{r}_{\theta}(y_w) = \beta \cdot \Delta \log \pi(y_w) = 0.5 \times 0.50 = \mathbf{0.25}$$

2. **Log-Ratio for Loser ($y_l$):**
   $$\Delta \log \pi(y_l) = \log \pi_{\theta}(y_l) - \log \pi_{\text{ref}}(y_l) = -1.0 - (-2.0) = +1.00$$
   Implicit Reward:
   $$\hat{r}_{\theta}(y_l) = \beta \cdot \Delta \log \pi(y_l) = 0.5 \times 1.00 = \mathbf{0.50}$$

---

### 5.4 Step 2: Compute Reward Margin ($\Delta r$)

$$\Delta r = \hat{r}_{\theta}(y_w) - \hat{r}_{\theta}(y_l) = 0.25 - 0.50 = \mathbf{-0.25}$$

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
$$\nabla_{\theta} \mathcal{L}_{\text{DPO}} = -(0.5)(0.562176) \left[ \nabla_{\theta} \log \pi_{\theta}(y_w) - \nabla_{\theta} \log \pi_{\theta}(y_l) \right]$$
$$= -0.281088 \nabla_{\theta} \log \pi_{\theta}(y_w) + 0.281088 \nabla_{\theta} \log \pi_{\theta}(y_l)$$

When the optimizer takes a gradient descent step $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$:
- It adds $+0.281 \eta \nabla \log \pi_{\theta}(y_w)$, directly **increasing** the winner's likelihood.
- It subtracts $-0.281 \eta \nabla \log \pi_{\theta}(y_l)$, directly **decreasing** the loser's likelihood.

---

## 6. Solved Illustrations

### Illustration 1: Goodhart's Law and Reward Hacking in RLHF

**Problem:**
Goodhart's Law states: *"When a measure becomes a target, it ceases to be a good measure."*
How does reward hacking manifest in an LLM if the KL divergence constraint is disabled ($\beta = 0$)?

**Solution:**
A neural reward model $r_{\psi}(\mathbf{x}, \mathbf{y})$ is an imperfect approximation of human preference, typically trained on only $50,000 - 100,000$ comparisons.
If $\beta = 0$, PPO treats the reward model as an absolute ground-truth oracle:
- The policy discovers that repeating specific words (*"certainly!", "delighted!", "moreover"*), producing excessively long outputs (verbosity bias), or using exaggerated flattery (sycophancy) reliably triggers maximum activation in the reward model's linear head.
- The policy degenerates into outputting 1,000-word repetitive essays that receive a reward score of $+99.9$, but are completely unreadable and useless to actual human users.
- The KL penalty $\beta \mathbb{D}_{\text{KL}}(\pi_{\theta} \,\|\, \pi_{\text{ref}})$ prevents this by strictly penalizing the policy if it strays too far from the reference model's natural distribution.

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

### Illustration 3: Complete Hand Trace of Token-Level PPO with GAE Advantage and KL Penalty

**Problem:**
An RLHF policy generates a 3-token trajectory $\mathbf{y} = (y_1, y_2, y_3)$ given prompt $\mathbf{x}$.
A trained reward model evaluates the complete completion, assigning scalar score $r_{\psi}(\mathbf{x}, \mathbf{y}) = 2.50$.
Let the per-token generation probabilities be:
- Step 1: $\log \pi_{\theta}(y_1) = -1.00, \quad \log \pi_{\text{ref}}(y_1) = -1.50$
- Step 2: $\log \pi_{\theta}(y_2) = -0.80, \quad \log \pi_{\text{ref}}(y_2) = -0.80$
- Step 3: $\log \pi_{\theta}(y_3) = -1.20, \quad \log \pi_{\text{ref}}(y_3) = -0.60$

Let KL coefficient $\beta = 0.20$, discount $\gamma = 1.00$, and GAE parameter $\lambda = 0.95$.
The Value network (Critic) estimates state values:
$$V(s_1) = 2.00, \quad V(s_2) = 2.20, \quad V(s_3) = 2.40, \quad V(s_4) = 0.00 \text{ (terminal)}$$
1. Calculate the token-level composite rewards $R_t$.
2. Calculate the temporal difference residuals $\delta_t^V$.
3. Compute the GAE advantage estimates $\hat{A}_t^{\text{GAE}}$ recursively for all 3 tokens.
4. For step 2, suppose an updated policy produces ratio $r_2 = 1.30$. Evaluate the PPO clipped surrogate objective $\mathcal{L}_2^{\text{CLIP}}$ under clipping parameter $\epsilon = 0.20$.

**Solution:**

#### Step 1: Token-Level Composite Rewards $R_t$
The composite reward penalizes policy drift at each token:
$$R_t = -\beta \left( \log \pi_{\theta}(y_t) - \log \pi_{\text{ref}}(y_t) \right) \quad (\text{for } t < 3)$$
$$R_3 = r_{\psi} - \beta \left( \log \pi_{\theta}(y_3) - \log \pi_{\text{ref}}(y_3) \right)$$

- **Token 1:**
  $$R_1 = -0.20 \times (-1.00 - (-1.50)) = -0.20 \times (+0.50) = \mathbf{-0.1000}$$
- **Token 2:**
  $$R_2 = -0.20 \times (-0.80 - (-0.80)) = -0.20 \times 0.00 = \mathbf{0.0000}$$
- **Token 3:**
  $$R_3 = 2.50 - 0.20 \times (-1.20 - (-0.60)) = 2.50 - 0.20 \times (-0.60) = 2.50 + 0.120 = \mathbf{2.6200}$$

---

#### Step 2: Temporal Difference Residuals $\delta_t^V$
$$\delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$$
- **Token 1:**
  $$\delta_1^V = -0.1000 + (1.00)(2.20) - 2.00 = -0.1000 + 0.20 = \mathbf{+0.1000}$$
- **Token 2:**
  $$\delta_2^V = 0.0000 + (1.00)(2.40) - 2.20 = \mathbf{+0.2000}$$
- **Token 3:**
  $$\delta_3^V = 2.6200 + (1.00)(0.00) - 2.40 = \mathbf{+0.2200}$$

---

#### Step 3: Backward Recursive Generalized Advantage Estimation ($\hat{A}_t$)
Using $\hat{A}_t = \delta_t^V + \gamma \lambda \hat{A}_{t+1}$ with $\gamma \lambda = (1.00)(0.95) = 0.95$:

- **Token 3 (Base Case):**
  $$\hat{A}_3 = \delta_3^V = \mathbf{+0.220000}$$
- **Token 2:**
  $$\hat{A}_2 = \delta_2^V + 0.95 \hat{A}_3 = 0.2000 + 0.95(0.220000) = 0.2000 + 0.209000 = \mathbf{+0.409000}$$
- **Token 1:**
  $$\hat{A}_1 = \delta_1^V + 0.95 \hat{A}_2 = 0.1000 + 0.95(0.409000) = 0.1000 + 0.388550 = \mathbf{+0.488550}$$

Each generated token receives a positive credit assignment indicating it performed better than the Critic expected.

---

#### Step 4: PPO Clipped Surrogate Evaluation at Step 2
Given ratio $r_2 = 1.30$, advantage $\hat{A}_2 = 0.4090$, and clipping parameter $\epsilon = 0.20$:
- Clipping bounds: $[1 - \epsilon, \, 1 + \epsilon] = [0.80, \, 1.20]$.
- Unclipped objective:
  $$r_2 \hat{A}_2 = 1.30 \times 0.4090 = \mathbf{0.531700}$$
- Clipped objective:
  $$\operatorname{clip}(r_2, 0.80, 1.20) \hat{A}_2 = 1.20 \times 0.4090 = \mathbf{0.490800}$$
- PPO Pessimistic Objective:
  $$\mathcal{L}_2^{\text{CLIP}} = \min(0.531700, \, 0.490800) = \mathbf{0.490800}$$
The clipping mechanism activates, successfully dampening the gradient and preventing an overly aggressive policy update.

---

### Illustration 4: Complete DPO Forward and Parameter Update on a 2-Step Vocabulary

**Problem:**
A simple decision policy evaluates two candidate tokens: winner $y_w$ and loser $y_l$.
The current model parameters are the unnormalized logits $\mathbf{z} = [z_w, z_l]^T = [0.0, 1.0]^T$.
The frozen reference model has uniform logits $\mathbf{z}_{\text{ref}} = [0.5, 0.5]^T$.
KL scale is $\beta = 1.0$, and learning rate is $\eta = 1.0$.
1. Compute the probabilities under policy $\pi_{\theta}$ and reference $\pi_{\text{ref}}$.
2. Compute the implicit rewards $\hat{r}(y_w), \hat{r}(y_l)$ and reward margin $\Delta r$.
3. Compute the DPO loss $\mathcal{L}_{\text{DPO}}$.
4. Evaluate the analytical parameter gradient $\nabla_{\mathbf{z}} \mathcal{L}_{\text{DPO}}$ and apply one gradient descent step $\mathbf{z} \leftarrow \mathbf{z} - \eta \nabla \mathcal{L}$. Verify that the preference inverts.

**Solution:**

#### Step 1: Probability Distributions
1. **Policy $\pi_{\theta}$ ($\mathbf{z} = [0.0, 1.0]$):**
   $$\exp(0.0) = 1.000000, \quad \exp(1.0) = 2.718282 \implies \text{Sum} = 3.718282$$
   $$\pi(y_w) = \frac{1.000000}{3.718282} \approx \mathbf{0.268941} \implies \log \pi(y_w) = \ln(0.268941) = \mathbf{-1.313262}$$
   $$\pi(y_l) = \frac{2.718282}{3.718282} \approx \mathbf{0.731059} \implies \log \pi(y_l) = \ln(0.731059) = \mathbf{-0.313262}$$

2. **Reference Model $\pi_{\text{ref}}$ ($\mathbf{z}_{\text{ref}} = [0.5, 0.5]$):**
   $$\pi_{\text{ref}}(y_w) = 0.500000 \implies \log \pi_{\text{ref}}(y_w) = \ln(0.5) = \mathbf{-0.693147}$$
   $$\pi_{\text{ref}}(y_l) = 0.500000 \implies \log \pi_{\text{ref}}(y_l) = \ln(0.5) = \mathbf{-0.693147}$$

---

#### Step 2: Implicit Rewards and Margin ($\beta = 1.0$)
$$\hat{r}(y) = \beta \left( \log \pi(y) - \log \pi_{\text{ref}}(y) \right)$$
$$\hat{r}(y_w) = 1.0 \times (-1.313262 - (-0.693147)) = \mathbf{-0.620115}$$
$$\hat{r}(y_l) = 1.0 \times (-0.313262 - (-0.693147)) = \mathbf{+0.379885}$$
$$\Delta r = \hat{r}(y_w) - \hat{r}(y_l) = -0.620115 - 0.379885 = \mathbf{-1.000000}$$

---

#### Step 3: DPO Loss
$$\sigma(\Delta r) = \sigma(-1.000000) = \frac{1}{1 + e^{1.0}} = \frac{1}{1 + 2.718282} = \frac{1}{3.718282} \approx \mathbf{0.268941}$$
$$\mathcal{L}_{\text{DPO}} = -\log(\sigma(\Delta r)) = -\ln(0.268941) = \mathbf{1.313262}$$

---

#### Step 4: Gradient and Parameter Update
The gradient scaling weight is:
$$w_{\text{grad}} = \sigma(-\Delta r) = \sigma(+1.000000) = 1 - 0.268941 = \mathbf{0.731059}$$

Evaluating softmax derivatives w.r.t. logits $\mathbf{z}$:
$$\nabla_{\mathbf{z}} \log \pi(y_w) = \mathbf{e}_w - \pi = \begin{bmatrix} 1.0 - 0.268941 \\ 0.0 - 0.731059 \end{bmatrix} = \begin{bmatrix} +0.731059 \\ -0.731059 \end{bmatrix}$$
$$\nabla_{\mathbf{z}} \log \pi(y_l) = \mathbf{e}_l - \pi = \begin{bmatrix} 0.0 - 0.268941 \\ 1.0 - 0.731059 \end{bmatrix} = \begin{bmatrix} -0.268941 \\ +0.268941 \end{bmatrix}$$
$$\nabla_{\mathbf{z}} \log \pi(y_w) - \nabla_{\mathbf{z}} \log \pi(y_l) = \begin{bmatrix} 0.731059 - (-0.268941) \\ -0.731059 - 0.268941 \end{bmatrix} = \begin{bmatrix} +1.000000 \\ -1.000000 \end{bmatrix}$$

Multiplying by $-w_{\text{grad}}$:
$$\nabla_{\mathbf{z}} \mathcal{L}_{\text{DPO}} = -0.731059 \begin{bmatrix} +1.000000 \\ -1.000000 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.731059} \\ \mathbf{+0.731059} \end{bmatrix}$$

Applying gradient descent step $\mathbf{z} \leftarrow \mathbf{z} - \eta \nabla_{\mathbf{z}} \mathcal{L}$:
$$z_w \leftarrow 0.0 - (1.0)(-0.731059) = \mathbf{+0.731059}$$
$$z_l \leftarrow 1.0 - (1.0)(+0.731059) = \mathbf{+0.268941}$$

**Conclusion:**
In a single step, the logits have flipped from $z_l > z_w$ ($1.0 > 0.0$) to $z_w > z_l$ ($0.731 > 0.269$). The winner probability surged from $26.9\% \to 61.4\%$, demonstrating the direct and stable corrective power of DPO.

---

### Illustration 5: Bradley-Terry Reward Model Ranking and Margin Hand Trace across 3 Completions

**Problem:**
A reward model evaluates three candidate completions $y_1, y_2, y_3$ for a user prompt, outputting scalar scores:
$$r_1 = 2.40, \quad r_2 = 1.80, \quad r_3 = 0.60$$
The human preference ordering is $y_1 \succ y_2 \succ y_3$.
1. Compute the pairwise Bradley-Terry preference probabilities $P(y_i \succ y_j) = \sigma(r_i - r_j)$ for all pairs $(1, 2), (2, 3), (1, 3)$.
2. Compute the individual cross-entropy losses and total pairwise ranking loss $\mathcal{L}_{\text{total}}$.
3. Compute the reward gradients $\frac{\partial \mathcal{L}}{\partial r_i}$ and verify the zero-sum invariance $\sum_{i=1}^3 \frac{\partial \mathcal{L}}{\partial r_i} = 0$.

**Solution:**

#### Step 1: Pairwise Bradley-Terry Probabilities
- **Pair $(1, 2)$:** $\Delta r_{12} = r_1 - r_2 = 2.40 - 1.80 = 0.60$.
  $$P(y_1 \succ y_2) = \sigma(0.60) = \frac{1}{1 + e^{-0.60}} = \frac{1}{1 + 0.548812} = \frac{1}{1.548812} \approx \mathbf{0.645656}$$
- **Pair $(2, 3)$:** $\Delta r_{23} = r_2 - r_3 = 1.80 - 0.60 = 1.20$.
  $$P(y_2 \succ y_3) = \sigma(1.20) = \frac{1}{1 + e^{-1.20}} = \frac{1}{1 + 0.301194} = \frac{1}{1.301194} \approx \mathbf{0.768525}$$
- **Pair $(1, 3)$:** $\Delta r_{13} = r_1 - r_3 = 2.40 - 0.60 = 1.80$.
  $$P(y_1 \succ y_3) = \sigma(1.80) = \frac{1}{1 + e^{-1.80}} = \frac{1}{1 + 0.165299} = \frac{1}{1.165299} \approx \mathbf{0.858149}$$

---

#### Step 2: Loss Computation
$$\mathcal{L}_{ij} = -\ln P(y_i \succ y_j)$$
$$\mathcal{L}_{12} = -\ln(0.645656) \approx \mathbf{0.437488}$$
$$\mathcal{L}_{23} = -\ln(0.768525) \approx \mathbf{0.263283}$$
$$\mathcal{L}_{13} = -\ln(0.858149) \approx \mathbf{0.152973}$$
$$\mathcal{L}_{\text{total}} = 0.437488 + 0.263283 + 0.152973 = \mathbf{0.853744}$$

---

#### Step 3: Reward Gradients and Zero-Sum Check
For each pair $(w, l)$:
$$\frac{\partial \mathcal{L}_{wl}}{\partial r_w} = -(1 - \sigma(r_w - r_l)), \quad \frac{\partial \mathcal{L}_{wl}}{\partial r_l} = +(1 - \sigma(r_w - r_l))$$

- **For $r_1$ (Winner in $(1, 2)$ and $(1, 3)$):**
  $$\frac{\partial \mathcal{L}}{\partial r_1} = -(1 - 0.645656) - (1 - 0.858149) = -0.354344 - 0.141851 = \mathbf{-0.496195}$$

- **For $r_2$ (Loser in $(1, 2)$, Winner in $(2, 3)$):**
  $$\frac{\partial \mathcal{L}}{\partial r_2} = +(1 - 0.645656) - (1 - 0.768525) = +0.354344 - 0.231475 = \mathbf{+0.122869}$$

- **For $r_3$ (Loser in $(2, 3)$ and $(1, 3)$):**
  $$\frac{\partial \mathcal{L}}{\partial r_3} = +(1 - 0.768525) + (1 - 0.858149) = +0.231475 + 0.141851 = \mathbf{+0.373326}$$

- **Zero-Sum Verification:**
  $$\sum_{i=1}^3 \frac{\partial \mathcal{L}}{\partial r_i} = -0.496195 + 0.122869 + 0.373326 = \mathbf{0.000000}$$
  The gradient sums to exactly zero, reflecting that the Bradley-Terry preference model depends exclusively on relative scalar differences, not absolute reward values.

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
