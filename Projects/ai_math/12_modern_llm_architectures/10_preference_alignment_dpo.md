# 12.10 Post-Training II: Direct Preference Alignment (DPO, ORPO & SimPO)

---

## 1. Intuition & 101 Motivation

Supervised Fine-Tuning (SFT) teaches an LLM conversational grammar, tool usage, and prompt formats. However, SFT alone is fundamentally inadequate for aligning models with human values, safety criteria, and nuanced reasoning quality. Because standard cross-entropy loss treats all tokens equally, it cannot penalize a subtle hallucination or reward a concise, elegant answer over a verbose, rambling one.

Traditionally, alignment was solved via **Reinforcement Learning from Human Feedback (RLHF)** using Proximal Policy Optimization (PPO). As seen in Chapter 11.27, RLHF requires maintaining **four separate foundation models** in GPU memory concurrently:
1. Active Actor Policy $\pi_\theta$
2. Critic Value Network $V_\phi$
3. Reward Model $r_\psi$
4. Reference Policy $\pi_{\text{ref}}$

This multi-model setup suffers from notorious training instability, hyperparameter sensitivity, and extreme GPU memory overhead.

**Direct Preference Optimization (DPO)** (Rafailov et al., 2023) revolutionized alignment by proving an exact mathematical duality: **the optimal policy itself implicitly defines the reward function**. By substituting this closed-form relation directly into the Bradley-Terry preference objective, DPO bypasses the reward model and PPO reinforcement loop entirely!

Subsequent breakthroughs further refined this paradigm:
- **ORPO (Odds Ratio Preference Optimization):** Fuses SFT and preference alignment into a single monolithic training stage without needing any reference model $\pi_{\text{ref}}$.
- **SimPO (Simple Preference Optimization):** Replaces the reference model with length-normalized sequence log-probabilities and an explicit target margin $\gamma$, eliminating the severe verbosity bias of standard DPO.

```
                   ALIGNMENT PIPELINE EVOLUTION
                   
   Traditional RLHF (PPO)                Direct Preference Optimization (DPO)
  4 Models in GPU VRAM                   2 Models (or 1 in SimPO / ORPO)
  ┌───────────────────────┐              ┌───────────────────────┐
  │ Actor Policy π_θ      │              │ Actor Policy π_θ      │
  │ Critic Network V_ϕ    │              │ Reference Policy π_ref│
  │ Reward Model r_ψ      │              └───────────────────────┘
  │ Reference Model π_ref │              Direct loss on (x, y_w, y_l)
  └───────────────────────┘              Zero RL loops, Stable Supervised Training
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Bradley-Terry Preference Model

Given a prompt $x$ and a pair of candidate responses $(y_w, y_l)$, where $y_w$ is the preferred (winning) completion and $y_l$ is the dispreferred (losing) completion, human preference probability is modeled via the Bradley-Terry formulation:

$$P(y_w \succ y_l \mid x) = \sigma\left( r^*(x, y_w) - r^*(x, y_l) \right) = \frac{1}{1 + \exp\left( -\left( r^*(x, y_w) - r^*(x, y_l) \right) \right)}$$

where $r^*(x, y)$ is the latent, unobserved ground-truth reward function.

---

### 2.2 The DPO Derivation & Mathematical Proof

In standard RLHF, the objective optimizes the policy $\pi$ to maximize expected reward while penalizing KL divergence from a frozen reference model $\pi_{\text{ref}}$:

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \, \mathbb{D}_{\text{KL}}\left( \pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x) \right)$$

where $\beta > 0$ controls the strength of the KL penalty constraint.

#### Theorem 1: Closed-Form Optimal Policy Solution
The exact analytical solution to this constrained optimization problem is given by:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

where the partition function $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ normalizes the distribution over all possible completions.

#### Proof of DPO Inversion:
Taking the natural logarithm of both sides:
$$\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

Rearranging for the reward $r(x, y)$:
$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Now, evaluate the reward difference between winner $y_w$ and loser $y_l$:
$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right)$$

The intractable partition function $\beta \log Z(x)$ **cancels out completely**:

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

#### The DPO Objective Function:
Substituting this reward difference directly into the negative log-likelihood of the Bradley-Terry preference model yields the **DPO Loss**:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] \quad \blacksquare$$

---

### 2.3 Gradient Dynamics of DPO

Differentiating $\mathcal{L}_{\text{DPO}}$ with respect to policy parameters $\theta$:

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = - \beta \underbrace{\sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right)}_{\text{Dynamic Weight } w(x, y_w, y_l)} \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$

where $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the implicit reward.

**Properties of the Gradient:**
1. **Direction:** Increases the probability of the winning response $\nabla_\theta \log \pi_\theta(y_w \mid x)$ while actively decreasing the probability of the losing response $-\nabla_\theta \log \pi_\theta(y_l \mid x)$.
2. **Magnitude:** The scale factor $\sigma(\hat{r}_l - \hat{r}_w)$ approaches $1.0$ when the model currently assigns higher implicit reward to the loser ($\hat{r}_l > \hat{r}_w$). When the model already assigns higher reward to the winner, the gradient naturally vanishes ($\to 0$).

---

### 2.4 Beyond DPO: SimPO (Simple Preference Optimization)

Standard DPO suffers from **verbosity bias**: because sequence log-probability is the sum of per-token log-probabilities ($\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid y_{<t})$), longer responses accumulate more negative values, but the ratio differences often favor long, repetitive responses.

**SimPO** (Meng et al., 2024) fixes this by introducing:
1. **Length-Normalized Implicit Reward:**
   $$\hat{r}_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$
2. **Explicit Target Margin $\gamma > 0$:**
   Enforces a strict reward buffer between winning and losing sequences.
3. **Reference-Free Formulation:**
   Eliminates $\pi_{\text{ref}}$, saving 50% GPU memory!

$$\mathcal{L}_{\text{SimPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$

---

## 3. Geometric & Physical Interpretation

### 3.1 Force Balance in Policy Space
In the geometry of probability simplexes, DPO acts as an **electrostatic dipole**:
- An attractive electrostatic force pulls $\pi_\theta(y_w \mid x)$ towards probability $1.0$.
- A repulsive electrostatic force pushes $\pi_\theta(y_l \mid x)$ towards $0.0$.
- The reference model $\pi_{\text{ref}}$ acts as an **anchor spring** with stiffness constant $\beta$. If the policy moves too far from $\pi_{\text{ref}}$, the restoring force increases, preventing mode collapse.

```
             DPO LOG-PROBABILITY DYNAMICS
             
       Log P(y_w)                   Log P(y_l)
          ▲                            ▲
          │    + Force                 │    - Force
     ────►│──────────             ────►│──────────
          │                            │
   Pushed UPWARDS               Pushed DOWNWARDS
   Bounded by β-Anchor          Bounded by β-Anchor
```

---

## 4. Real-World Analogy: The Restaurant Critic vs. The Diners

- **PPO / Traditional RLHF:**
  A restaurant hires a famous food critic (Reward Model). The chef cooks dishes, the critic writes an essay scoring the dish, and a consulting firm uses complex statistical regression (PPO Actor-Critic) to tell the cooks how to tweak the oven temperature. The process is expensive, chaotic, and the chef often exploits the critic's bias (e.g., drowning dishes in butter for high scores).
- **DPO:**
  The restaurant fires the food critic and the consulting firm. Instead, everyday diners are given two dishes ($y_w$ and $y_l$). The diners simply point to $y_w$. The chef immediately increases the production of recipe $y_w$ and decreases recipe $y_l$. Direct, stable, and zero overhead!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, cell-by-cell numerical trace of:
1. **Winning vs. Losing Log-Probability Ratios**
2. **Implicit Reward Difference $\Delta r$**
3. **DPO Sigmoid Probability and Loss Value**
4. **SimPO Length-Normalized Loss with Margin**

---

### 5.1 Concrete Input Values

- **Temperature hyperparameter:** $\beta = 0.5$
- **Prompt:** $x$ = *"Explain quantum superposition in one sentence."*
- **Candidate completions:**
  - $y_w$ (Winning response, Length $|y_w| = 2$ tokens): *"Particles exist in all states simultaneously."*
  - $y_l$ (Losing response, Length $|y_l| = 4$ tokens): *"It is basically when stuff is weird and confused."*

- **Sequence Log-Probabilities under Reference Model $\pi_{\text{ref}}$:**
  - $\log \pi_{\text{ref}}(y_w \mid x) = -2.000000$
  - $\log \pi_{\text{ref}}(y_l \mid x) = -3.000000$

- **Sequence Log-Probabilities under Current Policy $\pi_\theta$:**
  - $\log \pi_\theta(y_w \mid x) = -1.200000$
  - $\log \pi_\theta(y_l \mid x) = -2.600000$

- **SimPO Specific Parameters:**
  - Margin $\gamma = 0.50$, SimPO $\beta = 1.0$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\log \pi_\theta(y_w)$ | `policy_logp_w` | Log-probability of winning response under actor policy ($-1.2$) |
| $\log \pi_\theta(y_l)$ | `policy_logp_l` | Log-probability of losing response under actor policy ($-2.6$) |
| $\log \pi_{\text{ref}}(y_w)$ | `ref_logp_w` | Log-probability of winning response under reference policy ($-2.0$) |
| $\log \pi_{\text{ref}}(y_l)$ | `ref_logp_l` | Log-probability of losing response under reference policy ($-3.0$) |
| $\Delta r_{\text{DPO}}$ | `dpo_reward_diff` | Implicit reward difference $\beta \log(\pi_\theta(w)/\pi_{\text{ref}}(w)) - \beta \log(\pi_\theta(l)/\pi_{\text{ref}}(l))$ |
| $\mathcal{L}_{\text{DPO}}$ | `dpo_loss` | Negative log-sigmoid of reward difference |
| $\mathcal{L}_{\text{SimPO}}$ | `simpo_loss` | Length-normalized reward difference with margin |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Policy / Reference Log-Ratios
- For Winner ($y_w$):
  $$\Delta \log \pi(y_w) = \log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x) = -1.200000 - (-2.000000) = \mathbf{+0.800000}$$
- For Loser ($y_l$):
  $$\Delta \log \pi(y_l) = \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x) = -2.600000 - (-3.000000) = \mathbf{+0.400000}$$

---

#### Step 2: Compute Implicit Reward Difference $\Delta r_{\text{DPO}}$
With $\beta = 0.5$:
$$\hat{r}_\theta(x, y_w) = \beta \cdot (+0.800000) = 0.5 \times 0.80 = \mathbf{+0.400000}$$
$$\hat{r}_\theta(x, y_l) = \beta \cdot (+0.400000) = 0.5 \times 0.40 = \mathbf{+0.200000}$$

$$\Delta r_{\text{DPO}} = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) = 0.400000 - 0.200000 = \mathbf{+0.200000}$$

---

#### Step 3: Compute Sigmoid Preference Probability
$$\sigma(\Delta r_{\text{DPO}}) = \frac{1}{1 + e^{-0.200000}}$$
$$e^{-0.20} \approx 0.81873075$$
$$\sigma(0.20) = \frac{1}{1 + 0.81873075} = \frac{1}{1.81873075} \approx \mathbf{0.549834}$$

---

#### Step 4: Compute DPO Loss
$$\mathcal{L}_{\text{DPO}} = -\ln\left( \sigma(\Delta r_{\text{DPO}}) \right) = -\ln(0.549834) \approx -(-0.598139) = \mathbf{0.598139}$$

---

#### Step 5: Compute SimPO Loss on Same Sample
SimPO uses length normalization ($|y_w| = 2, |y_l| = 4$) and target margin $\gamma = 0.50$ with $\beta = 1.0$:

1. Length-normalized reward for winner:
   $$\hat{r}_{\text{SimPO}}(y_w) = \frac{1.0}{2} \log \pi_\theta(y_w) = \frac{-1.200000}{2} = \mathbf{-0.600000}$$
2. Length-normalized reward for loser:
   $$\hat{r}_{\text{SimPO}}(y_l) = \frac{1.0}{4} \log \pi_\theta(y_l) = \frac{-2.600000}{4} = \mathbf{-0.650000}$$
3. Margin-adjusted reward difference:
   $$\Delta r_{\text{SimPO}} = \hat{r}_{\text{SimPO}}(y_w) - \hat{r}_{\text{SimPO}}(y_l) - \gamma = (-0.600000) - (-0.650000) - 0.500000$$
   $$= +0.050000 - 0.500000 = \mathbf{-0.450000}$$
4. Sigmoid:
   $$e^{-(-0.45)} = e^{+0.45} \approx 1.568312$$
   $$\sigma(-0.45) = \frac{1}{1 + 1.568312} = \frac{1}{2.568312} \approx \mathbf{0.389365}$$
5. SimPO Loss:
   $$\mathcal{L}_{\text{SimPO}} = -\ln(0.389365) \approx \mathbf{0.943249}$$

---

### 5.4 Summary Visual Grid: Preference Alignment Ledger

| Quantity | Mathematical Term | Numerical Value | Intuition / Role |
| :--- | :---: | :---: | :--- |
| **Winner Policy LogP** | $\log \pi_\theta(y_w)$ | $-1.2000$ | Current model likelihood of winning reply |
| **Winner Ref LogP** | $\log \pi_{\text{ref}}(y_w)$ | $-2.0000$ | Baseline likelihood of winning reply |
| **Loser Policy LogP** | $\log \pi_\theta(y_l)$ | $-2.6000$ | Current model likelihood of losing reply |
| **Loser Ref LogP** | $\log \pi_{\text{ref}}(y_l)$ | $-3.0000$ | Baseline likelihood of losing reply |
| **Implicit Winner Reward** | $\hat{r}_w$ | $+0.4000$ | $\beta \log(\pi_\theta / \pi_{\text{ref}})$ |
| **Implicit Loser Reward** | $\hat{r}_l$ | $+0.2000$ | $\beta \log(\pi_\theta / \pi_{\text{ref}})$ |
| **DPO Reward Diff** | $\hat{r}_w - \hat{r}_l$ | $\mathbf{+0.2000}$ | Positive preference margin achieved |
| **DPO Loss** | $-\log \sigma(\Delta r)$ | $\mathbf{0.598139}$ | Cross-entropy preference error |
| **SimPO Reward Diff** | $\Delta r_{\text{len}} - \gamma$ | $\mathbf{-0.4500}$ | Fails target margin $\gamma=0.50$ |
| **SimPO Loss** | $-\log \sigma(\Delta r_{\text{SimPO}})$ | $\mathbf{0.943249}$ | Penalizes insufficient margin |

---

## 6. Solved Illustrations

### Illustration 1: Resolving the DPO Verbosity Exploit with Length Normalization
**Problem:**
A model generates two responses to the prompt $x$:
- Short response $y_{\text{short}}$ ($|y|=10$ tokens): high per-token quality ($\text{mean logp} = -0.50$). Total $\log \pi_\theta = -5.0$.
- Verbose response $y_{\text{verbose}}$ ($|y|=100$ tokens): mediocre per-token quality ($\text{mean logp} = -0.80$). Total $\log \pi_\theta = -80.0$.
Assume reference model per-token logp is $-1.00$ for both:
- $\log \pi_{\text{ref}}(y_{\text{short}}) = -10.0$
- $\log \pi_{\text{ref}}(y_{\text{verbose}}) = -100.0$
With $\beta = 0.5$, which response does DPO prefer? Does SimPO fix it?

**Solution:**
1. **DPO Implicit Rewards:**
   $$\hat{r}_{\text{DPO}}(y_{\text{short}}) = 0.5 \times (-5.0 - (-10.0)) = 0.5 \times (+5.0) = \mathbf{+2.50}$$
   $$\hat{r}_{\text{DPO}}(y_{\text{verbose}}) = 0.5 \times (-80.0 - (-100.0)) = 0.5 \times (+20.0) = \mathbf{+10.00}$$
   **Pathology:** DPO assigns a reward of $+10.00$ to the mediocre verbose response and only $+2.50$ to the crisp concise response! DPO heavily rewards verbosity because the log-ratio sums over length.
2. **SimPO Length-Normalized Rewards ($\beta = 1.0$):**
   $$\hat{r}_{\text{SimPO}}(y_{\text{short}}) = \frac{-5.0}{10} = \mathbf{-0.50}$$
   $$\hat{r}_{\text{SimPO}}(y_{\text{verbose}}) = \frac{-80.0}{100} = \mathbf{-0.80}$$
   $$\Delta r_{\text{SimPO}} = (-0.50) - (-0.80) = \mathbf{+0.30} > 0$$
   SimPO correctly prefers the high-quality short response by $+0.30$! $\blacksquare$

---

### Illustration 2: DPO Loss Computation
**Problem:**
Given a chosen response $y_w$ and rejected response $y_l$, compute the DPO loss.
The log probabilities are:
- $\log \pi_\theta(y_w \mid x) = -2.3$
- $\log \pi_\theta(y_l \mid x) = -3.8$
- $\log \pi_{\text{ref}}(y_w \mid x) = -2.5$
- $\log \pi_{\text{ref}}(y_l \mid x) = -3.2$
Assume $\beta = 0.1$.

**Solution:**
1. **Compute Log Ratios:**
   - $\log\text{ratio}_w = \log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w) = -2.3 - (-2.5) = \mathbf{0.2}$
   - $\log\text{ratio}_l = \log \pi_\theta(y_l) - \log \pi_{\text{ref}}(y_l) = -3.8 - (-3.2) = \mathbf{-0.6}$

2. **Compute DPO Logit:**
   - $\text{Logit} = \beta \times (\log\text{ratio}_w - \log\text{ratio}_l) = 0.1 \times (0.2 - (-0.6)) = 0.1 \times 0.8 = \mathbf{0.08}$

3. **Compute DPO Loss:**
   - $\sigma(0.08) = \frac{1}{1 + e^{-0.08}} = \frac{1}{1 + 0.9231} = \frac{1}{1.9231} = \mathbf{0.5200}$
   - $\mathcal{L}_{\text{DPO}} = -\ln(\sigma(0.08)) = -\ln(0.5200) = \mathbf{0.6539} \quad \blacksquare$

---

### Illustration 3: ORPO Loss Combining NLL and Odds Ratio
**Problem:**
For the same sequence log probabilities as Illustration 2 (under $\pi_\theta$), compute the ORPO loss. 
Assume the multiplier $\lambda = 1.0$.

**Solution:**
1. **Supervised NLL Loss on Chosen Response:**
   - $\mathcal{L}_{\text{NLL}} = -\log \pi_\theta(y_w \mid x) = \mathbf{2.3}$

2. **Compute Probabilities:**
   - $\pi_\theta(y_w) = e^{-2.3} = \mathbf{0.1003}$
   - $\pi_\theta(y_l) = e^{-3.8} = \mathbf{0.0224}$

3. **Compute Odds Ratio (OR):**
   - $\text{Odds}_w = \frac{\pi_\theta(y_w)}{1 - \pi_\theta(y_w)} = \frac{0.1003}{0.8997} = \mathbf{0.1115}$
   - $\text{Odds}_l = \frac{\pi_\theta(y_l)}{1 - \pi_\theta(y_l)} = \frac{0.0224}{0.9776} = \mathbf{0.0229}$
   - $OR = \frac{\text{Odds}_w}{\text{Odds}_l} = \frac{0.1115}{0.0229} = \mathbf{4.869}$
   - $\log OR = \ln(4.869) = \mathbf{1.583}$

4. **Compute ORPO Loss Component:**
   - $\sigma(\log OR) = \frac{1}{1 + e^{-1.583}} = \frac{1}{1 + 0.205} = \frac{1}{1.205} = \mathbf{0.830}$
   - $\mathcal{L}_{\text{OR}} = -\ln(0.830) = \mathbf{0.186}$

5. **Total ORPO Loss:**
   - $\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{NLL}} + \lambda \times \mathcal{L}_{\text{OR}} = 2.3 + 1.0 \times 0.186 = \mathbf{2.486} \quad \blacksquare$

---

### Illustration 4: Sensitivity Analysis for DPO $\beta$
**Problem:**
Using the same base log-ratio difference as Illustration 2 ($0.8$), evaluate the sensitivity of the DPO loss across different temperatures: $\beta \in \{0.01, 0.1, 0.5, 1.0\}$.

**Solution:**
We compute $\text{logit} = \beta \times 0.8$ and $\text{loss} = -\ln(\sigma(\text{logit}))$.

1. **For $\beta = 0.01$:**
   - $\text{Logit} = 0.01 \times 0.8 = 0.008$
   - $\sigma(0.008) = \frac{1}{1 + e^{-0.008}} = 0.502$
   - $\text{Loss} = -\ln(0.502) = \mathbf{0.689}$

2. **For $\beta = 0.1$:**
   - $\text{Logit} = 0.1 \times 0.8 = 0.08$
   - $\sigma(0.08) = 0.520$
   - $\text{Loss} = -\ln(0.520) = \mathbf{0.654}$

3. **For $\beta = 0.5$:**
   - $\text{Logit} = 0.5 \times 0.8 = 0.4$
   - $\sigma(0.4) = \frac{1}{1 + e^{-0.4}} = \frac{1}{1 + 0.670} = 0.599$
   - $\text{Loss} = -\ln(0.599) = \mathbf{0.512}$

4. **For $\beta = 1.0$:**
   - $\text{Logit} = 1.0 \times 0.8 = 0.8$
   - $\sigma(0.8) = \frac{1}{1 + e^{-0.8}} = \frac{1}{1 + 0.449} = 0.690$
   - $\text{Loss} = -\ln(0.690) = \mathbf{0.371}$

**Conclusion:** 
A higher $\beta$ creates a much sharper preference signal, yielding a lower loss when the correct ordering is maintained, but induces larger gradient steps that risk policy instability. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **TRL (Transformer Reinforcement Learning) `DPOTrainer`:** The standard implementation in the Hugging Face ecosystem, used to train Zephyr-7B, Starling-LM, and open frontier models.
- **LLaMA-3 Alignment Pipeline:** Meta's LLaMA-3 technical report documented an iterative post-training pipeline combining multiple rounds of DPO with rejection sampling and online preference updates, completely eliminating PPO actor-critic instability.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - DPO log-ratio calculation.
   - Implicit reward difference matching $+0.200000$.
   - DPO loss matching $0.598144$ to $< 10^{-6}$.
   - SimPO length-normalized loss matching $0.943235$.
2. **Production-Ready PyTorch Alignment Modules:**
   - `DPOLoss` module with reference model regularization and label masking.
   - `SimPOLoss` module with length normalization and target margin $\gamma$.
   - Full gradient backpropagation test.

See implementation in:
[`12_modern_llm_architectures/code/10_preference_alignment_dpo.py`](./code/10_preference_alignment_dpo.py)
