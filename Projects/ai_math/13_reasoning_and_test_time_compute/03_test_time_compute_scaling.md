# 13.3 Test-Time Compute & Inference Scaling Laws (Best-of-N, Beam Search & Budget Allocation)

---

## 1. Intuition & 101 Motivation

From 2020 to 2024, the primary path to increasing artificial intelligence capabilities was governed by **pre-training scaling laws** (Kaplan et al., Hoffmann et al.): train larger models with more parameters ($N$) on more tokens ($D$). However, this paradigm faces severe structural headwinds:
1. **Capital & Energy Walls:** Training a trillion-parameter foundation model requires tens of thousands of GPUs consuming hundreds of megawatts and costing hundreds of millions of dollars.
2. **Pre-Training Data Exhaustion:** The human-generated internet has effectively been ingested; high-quality text data is largely depleted.

In late 2024, the frontier shifted toward a revolutionary third axis: **Inference-Time (Test-Time) Compute Scaling** (Snell et al., UC Berkeley; Brown et al., OpenAI o1/o3).

> *Rather than making the model larger, keep the model fixed and allow it to spend $10\times$, $100\times$, or $1,000\times$ more computational FLOPs thinking at inference time before providing its answer.*

On challenging reasoning benchmarks (e.g. American Invitational Mathematics Examination - AIME, GPQA Diamond, Codeforces), scaling test-time compute allows an open **8B parameter model** to match or surpass a **405B parameter model** operating under single-sample greedy decoding!

```
                 THE THREE SCALING AXES OF AI
                 
                     Model Parameters (N)
                              ▲
                             /│\
                            / │ \
                           /  │  \
                          /   │   \
                         /    │    \
   Pre-Training Data (D) ◄────┼────► Inference-Time Compute (C_test)
                              │
                    [The Frontier Frontier]
         Spend 10,000 FLOPs thinking per token at test-time
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Test-Time Compute Formulation

Let $C_{\text{test}}$ denote the total floating-point operations expended during inference to answer a single prompt $x$.
For an autoregressive model with $N_{\text{params}}$ non-embedding parameters generating $M$ candidate trajectories, each consisting of $L_m$ tokens:

$$C_{\text{test}} \approx \sum_{m=1}^M 2 \cdot N_{\text{params}} \cdot L_m \text{ FLOPs}$$

Test-time compute can be scaled across three primary orthogonal dimensions:
1. **Parallel Sampling (Best-of-$N$ / Rejection Sampling):** Increase $M$ (generate many independent answers and pick the best using a verifier).
2. **Sequential Search (Beam Search / MCTS):** Prune and steer reasoning tokens step-by-step using a Process Reward Model.
3. **Internal Thinking Budget (Length Scaling):** Allow the model to generate longer reasoning traces ($L_m \to 10,000+$ `<think>` tokens).

---

### 2.2 Parallel Sampling: Best-of-$N$ & Pass@$k$

In Best-of-$N$, the model samples $N$ independent candidate reasoning paths $\{y_1, y_2, \dots, y_N\} \sim P_\theta(\cdot \mid x)$ using temperature $T > 0$.
A verifier or Reward Model $R(x, y) \in \mathbb{R}$ scores each candidate, selecting:

$$y^* = \arg\max_{y \in \{y_1, \dots, y_N\}} R(x, y)$$

#### 1. Theoretical Coverage Probability:
If the model has an independent single-sample pass rate $p \in (0, 1)$, the probability that **at least one** of the $N$ generated samples contains the correct answer (the coverage) is:

$$P_{\text{coverage}}(N) = 1 - (1 - p)^N$$

As $N \to \infty$, $P_{\text{coverage}} \to 1.0$. Even if $p = 0.05$ (only 5% chance of solving an Olympiad problem on a single attempt), sampling $N = 64$ attempts yields:
$$P_{\text{coverage}}(64) = 1 - (0.95)^{64} = 1 - 0.0375 = \mathbf{96.25\% \text{ coverage!}}$$

#### 2. The Unbiased Pass@$k$ Estimator (Chen et al., HumanEval):
When evaluating models across $N$ generated samples with $c$ correct solutions ($c \le N$), calculating the probability of getting at least one correct solution among $k$ samples ($k \le N$) without bias is given by:

$$\operatorname{Pass@}k = \mathbb{E} \left[ 1 - \frac{\binom{N - c}{k}}{\binom{N}{k}} \right]$$

This combinatorial estimator prevents high-variance naive sampling artifacts.

---

### 2.3 The Verifier Bottleneck & Goodhart's Law

In practice, Best-of-$N$ accuracy does **not** scale infinitely with $N$ because reward models are imperfect.

Let $R(x, y) = R^*(x, y) + \epsilon$, where $R^*$ is the true correctness and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is the verifier noise.
As $N$ increases, the probability of encountering an **adversarial outlier** (an incorrect response that receives an erroneously massive reward score) scales with extreme value statistics:

$$\max_{1 \le i \le N} \epsilon_i = \sigma \sqrt{2 \ln N} - \sigma \frac{\ln(\ln N) + \ln(4\pi)}{2 \sqrt{2 \ln N}}$$

#### Goodhart's Law for Verifiers:
When $N$ becomes excessively large ($N > 1000$), Best-of-$N$ **over-optimizes** the imperfect proxy reward, and accuracy begins to **degrade**:

$$\lim_{N \to \infty} P(\text{Best-of-}N \text{ Correct}) = 0$$

```
   Accuracy
        ▲
        │           Optimal N* (Peak Accuracy)
        │                *
        │              /   \   Verifier Over-Optimization (Goodhart Collapse)
        │             /     \
        │            /       ▼
        │           /
        │  ────────/
        └────────────────────────────────────────► Number of Samples N (Log Scale)
```

---

### 2.4 Compute-Optimal Allocation: Pre-Training vs. Test-Time

Snell et al. (2024) proved the fundamental trade-off:
Given a fixed total FLOP budget $C_{\text{total}} = C_{\text{train}} + C_{\text{test}}$:
- On **easy problems** where the base pass rate is already high ($p > 0.8$), spending test-time compute is wasteful. Allocate compute to pre-training.
- On **hard problems** near the model's frontier ($0.01 < p < 0.3$), spending $10\times$ more test-time compute via Best-of-$N$ or tree search improves accuracy far more efficiently than training a model $10\times$ larger!

---

## 3. Geometric & Physical Interpretation

### 3.1 Monte Carlo Volume Probing
In high-dimensional solution space, correct answers form a tiny attractor basin $\mathcal{A}_{\text{correct}} \subset \mathcal{Y}$.
- **Greedy Decoding (System 1):** Drops a single plumb line. If the plumb line lands in a neighboring false basin, the run fails permanently.
- **Best-of-$N$ Sampling:** Casts an ensemble of $N$ stochastic probes across the landscape. As long as at least one probe lands within the gravitational basin of $\mathcal{A}_{\text{correct}}$, the verifier pulls the solution into the target well.

---

## 4. Real-World Analogy: The Speed Chess Player vs. The Clock

Imagine a competitive chess match:
- **Zero Test-Time Compute (Bullet Chess - 1 minute total):** Even a grandmaster blunders frequently because they can only rely on immediate visual intuition (System 1).
- **High Test-Time Compute (Classical Match - 2 hours per player):** The grandmaster sits motionless for 30 minutes on move 18. In their head, they simulate 100 candidate move sequences ($N=100$), discard the variations that leave their king exposed, and play the single most devastating knight sacrifice. Giving the player more thinking time directly elevates the quality of their decisions.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, cell-by-cell numerical trace of:
1. **Coverage Probability $P_{\text{coverage}}(N) = 1 - (1 - p)^N$ across $N \in \{1, 2, 4, 8\}$**
2. **Unbiased Pass@$k$ Combinatorial Calculation**
3. **Best-of-$N$ Verifier Selection Ranking**

---

### 5.1 Concrete Input Values

- **Single-sample pass probability on a difficult math problem:** $p = 0.20$ ($1 - p = 0.80$)
- **Sample Budgets to Evaluate:** $N \in \{1, 2, 4, 8\}$
- **Pass@$k$ Evaluation:**
  - In a trial of $N = 5$ samples, exactly $c = 2$ samples are correct, and $N - c = 3$ samples are incorrect.
  - Calculate $\operatorname{Pass@1}$ and $\operatorname{Pass@2}$.
- **Best-of-$3$ Candidate Pool:**
  - Candidate 1 ($y_1$): Incorrect answer, Verifier Score $R = 0.35$
  - Candidate 2 ($y_2$): **Correct answer**, Verifier Score $R = 0.85$
  - Candidate 3 ($y_3$): Incorrect answer, Verifier Score $R = 0.60$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $p$ | `base_pass_rate` | Probability of generating correct solution on single attempt ($0.20$) |
| $N$ | `num_samples` | Number of parallel candidate rollouts generated |
| $P_{\text{cov}}$ | `coverage_prob` | Probability that at least one solution is correct: $1 - (1 - p)^N$ |
| $c$ | `num_correct` | Number of verified correct samples in budget $N$ ($c = 2$) |
| $\operatorname{Pass@}k$ | `pass_at_k` | Unbiased combinatorial estimator $1 - \frac{\binom{N-c}{k}}{\binom{N}{k}}$ |
| $R(x, y)$ | `verifier_score` | Reward model score assigned to candidate $y$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Coverage Probability as Budget $N$ Scales ($p = 0.20$)

1. **For $N = 1$:**
   $$P_{\text{cov}}(1) = 1 - (0.80)^1 = 1 - 0.80 = \mathbf{0.200000 \quad (20.00\%)}$$
2. **For $N = 2$:**
   $$(0.80)^2 = 0.640000$$
   $$P_{\text{cov}}(2) = 1 - 0.640000 = \mathbf{0.360000 \quad (36.00\%)}$$
3. **For $N = 4$:**
   $$(0.80)^4 = (0.64)^2 = 0.409600$$
   $$P_{\text{cov}}(4) = 1 - 0.409600 = \mathbf{0.590400 \quad (59.04\%)}$$
4. **For $N = 8$:**
   $$(0.80)^8 = (0.4096)^2 \approx 0.167772$$
   $$P_{\text{cov}}(8) = 1 - 0.167772 = \mathbf{0.832228 \quad (83.22\%)}$$

*Result:* By increasing inference compute by $8\times$, the probability that the correct answer is present jumps from **20.0% to 83.2%**!

---

#### Step 2: Compute Unbiased Pass@$k$ ($N = 5, c = 2$)

$$\operatorname{Pass@}k = 1 - \frac{\binom{N - c}{k}}{\binom{N}{k}} = 1 - \frac{\binom{3}{k}}{\binom{5}{k}}$$

1. **For $k = 1$:**
   $$\binom{3}{1} = 3, \quad \binom{5}{1} = 5$$
   $$\operatorname{Pass@1} = 1 - \frac{3}{5} = 1 - 0.60 = \mathbf{0.400000 \quad (40.00\%)}$$
2. **For $k = 2$:**
   $$\binom{3}{2} = \frac{3 \times 2}{2 \times 1} = 3, \quad \binom{5}{2} = \frac{5 \times 4}{2 \times 1} = 10$$
   $$\operatorname{Pass@2} = 1 - \frac{3}{10} = 1 - 0.30 = \mathbf{0.700000 \quad (70.00\%)}$$

---

#### Step 3: Best-of-3 Selection via Verifier Scoring
Rank candidates by $R(x, y)$:
- Rank 1: Candidate 2 ($y_2$) with $R = 0.85$ (**Selected**)
- Rank 2: Candidate 3 ($y_3$) with $R = 0.60$
- Rank 3: Candidate 1 ($y_1$) with $R = 0.35$

Since Candidate 2 is the correct answer, the Best-of-3 selection is **$100\%$ successful**! $\blacksquare$

---

### 5.4 Summary Visual Grid: Test-Time Scaling Ledger

| Sample Budget ($N$) | Failure Prob $(1-p)^N$ | Coverage $P_{\text{cov}}$ | Relative Compute | Marginal Gain |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | $0.800000$ | $\mathbf{0.200000}$ | $1\times$ | Base Rate |
| **$N = 2$** | $0.640000$ | $\mathbf{0.360000}$ | $2\times$ | $+16.00\%$ |
| **$N = 4$** | $0.409600$ | $\mathbf{0.590400}$ | $4\times$ | $+23.04\%$ |
| **$N = 8$** | $0.167772$ | $\mathbf{0.832228}$ | $8\times$ | $+24.18\%$ |
| **$N = 16$** | $0.028147$ | $\mathbf{0.971853}$ | $16\times$ | $+13.96\%$ |

---

## 6. Solved Illustrations

### Illustration 1: Diminishing Returns & Marginal Cost of Compute
**Problem:**
A laboratory has an 8B model with pass rate $p = 0.10$.
How many samples $N$ are required to achieve:
1. $50\%$ coverage probability?
2. $90\%$ coverage probability?
3. $99\%$ coverage probability?

**Solution:**
We solve $1 - (1 - p)^N \ge P_{\text{target}} \implies (1 - p)^N \le 1 - P_{\text{target}}$.
Taking natural logarithms:
$$N \ln(1 - p) \le \ln(1 - P_{\text{target}}) \implies N \ge \frac{\ln(1 - P_{\text{target}})}{\ln(0.90)}$$
Since $\ln(0.90) \approx -0.1053605$:

1. **For 50% Coverage:**
   $$N \ge \frac{\ln(0.50)}{-0.1053605} = \frac{-0.693147}{-0.1053605} \approx 6.58 \implies \mathbf{N = 7 \text{ samples}}$$
2. **For 90% Coverage:**
   $$N \ge \frac{\ln(0.10)}{-0.1053605} = \frac{-2.302585}{-0.1053605} \approx 21.85 \implies \mathbf{N = 22 \text{ samples}}$$
3. **For 99% Coverage:**
   $$N \ge \frac{\ln(0.01)}{-0.1053605} = \frac{-4.605170}{-0.1053605} \approx 43.71 \implies \mathbf{N = 44 \text{ samples}}$$
*(Notice: Going from 50% to 90% requires 15 additional samples. Going from 90% to 99% requires 22 additional samples, showing logarithmic diminishing returns).* $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **OpenAI o1 & o3 ("Strawberry"):** Users can select the reasoning effort: Low ($N \approx 1\text{--}4$), Medium ($N \approx 8\text{--}16$), High ($N \ge 32$). Compute is dynamically allocated based on problem difficulty.
- **DeepSeek-R1 Dynamic Thinking:** Rather than fixed sampling budgets, R1 internally decides whether to stop thinking at 500 tokens or continue exploring up to 32,768 tokens depending on confidence and internal backtrack verifications.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact coverage probabilities for $N \in \{1, 2, 4, 8\}$ matching $0.20, 0.36, 0.5904, 0.832228$.
   - Combinatorial Pass@$k$ estimator matching $0.40$ and $0.70$.
   - Best-of-$N$ verifier ranking selection.
2. **Production-Ready Inference Scaling Simulator:**
   - Unbiased HumanEval Pass@$k$ estimator implementation.
   - Simulation of verifier over-optimization (Goodhart's Law collapse).
   - Pareto efficiency curve comparing model size vs. test-time compute.

See implementation in:
[`13_reasoning_and_test_time_compute/code/03_test_time_compute_scaling.py`](./code/03_test_time_compute_scaling.py)
