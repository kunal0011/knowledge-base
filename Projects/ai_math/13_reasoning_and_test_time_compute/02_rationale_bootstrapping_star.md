# 13.2 Rationale Bootstrapping: The Self-Taught Reasoner (STaR) & Quiet-STaR

---

## 1. Intuition & 101 Motivation

While Chain-of-Thought prompting demonstrates that intermediate reasoning tokens unlock System 2 capabilities, foundation models cannot generate high-quality rationales out of the box for difficult domains. High-school competition mathematics, formal logic, and competitive programming require structured, multi-page derivations that base models rarely emit by chance.

Human annotation of reasoning traces is a catastrophic bottleneck: having PhD mathematicians write out 100,000 detailed step-by-step proofs costs millions of dollars and months of manual effort. Conversely, **final answers** are cheap and ubiquitous: math competitions have numerical answer keys ($y^* \in \mathbb{R}$), programming challenges have unit tests (`assert f(x) == y`), and science quizzes have multiple-choice answer keys.

This creates a fundamental question:
> *Can an LLM teach itself to discover and refine its own reasoning traces using only the ground-truth final answer as a verification signal?*

**STaR (Self-Taught Reasoner)** (Zelikman et al., Stanford 2022) answered with a resounding yes through an elegant bootstrapping loop:
1. **Sample Rationales:** The model attempts to solve problems by generating candidate reasoning traces $z$ followed by a proposed answer $\hat{y}$.
2. **Filter by Correctness:** Discard all rationales that lead to incorrect answers. Keep the successful rationales that arrived at the correct answer $y^*$.
3. **Rationalize Failures (The "Hint" Trick):** For problems where the model failed, provide the true answer $y^*$ as a hint in the prompt: *"The correct answer is $y^*$. Explain step-by-step why."* The model generates a reverse-engineered rationale $z^{\text{hint}}$.
4. **Iterative Fine-Tuning (M-step):** Fine-tune the model on the union of successful self-generated and rationalized traces.
5. **Repeat:** In the next iteration, the improved model can now solve harder problems unaided!

**Quiet-STaR** (Zelikman et al., 2024) generalized this concept to arbitrary text: rather than reasoning only when prompted with QA datasets, the model learns to generate **parallel internal thought tokens** at every token position, thinking quietly before predicting subsequent tokens.

```
                     THE STaR BOOTSTRAP LOOP
                     
                 Problem x, Ground Truth y*
                            │
                            ▼
              Generate Candidate Rationale z
                            │
              Predict Answer y_hat from z
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
        y_hat == y*               y_hat != y*
        (Success!)                 (Failure)
               │                         │
               │                  Give Hint: "Answer is y*"
               │                  Generate Rationalization z_hint
               │                         │
               └────────────┬────────────┘
                            ▼
           Add (x, z, y*) to Training Buffer
                            │
                            ▼
            Fine-Tune Model (SFT Update)
                            │
                            ▼
          Iterate with Improved Model θ_{t+1}
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 STaR as Generalized Expectation-Maximization (EM)

Consider a dataset of $N$ problem-answer pairs $\mathcal{D} = \{(x_i, y_i^*)\}_{i=1}^N$.
The reasoning trace $z \in \mathcal{Z}$ is an **unobserved latent variable**.
The true marginal log-likelihood objective is:

$$\mathcal{J}(\theta) = \sum_{i=1}^N \log P_\theta(y_i^* \mid x_i) = \sum_{i=1}^N \log \sum_{z \in \mathcal{Z}} P_\theta(z \mid x_i) P_\theta(y_i^* \mid x_i, z)$$

Direct gradient optimization of this marginal is intractable due to the sum over all $|\mathcal{V}|^K$ paths.
STaR optimizes a variational lower bound (Jensen's Inequality) using hard Expectation-Maximization:

$$\log P_\theta(y^* \mid x) \ge \sum_z q(z \mid x, y^*) \log \frac{P_\theta(z, y^* \mid x)}{q(z \mid x, y^*)}$$

where the optimal posterior distribution $q(z \mid x, y^*)$ is:
$$q(z \mid x, y^*) \propto P_\theta(z \mid x) \cdot \mathbb{I}\left( \arg\max_y P_\theta(y \mid x, z) = y^* \right)$$

---

### 2.2 The STaR Algorithm: Step-by-Step

At iteration $t$ with policy $\theta_t$:

#### 1. Expectation Step (E-step: Sampling & Verification Filter):
For each problem $(x_i, y_i^*) \in \mathcal{D}$:
1. Sample candidate rationale and final answer:
   $$\hat{z}_i \sim P_{\theta_t}(z \mid x_i), \quad \hat{y}_i \sim P_{\theta_t}(y \mid x_i, \hat{z}_i)$$
2. If $\hat{y}_i = y_i^*$ (Verification Success):
   $$\mathcal{D}_{\text{filtered}}^{(t)} \leftarrow \mathcal{D}_{\text{filtered}}^{(t)} \cup \{(x_i, \hat{z}_i, y_i^*)\}$$

#### 2. Rationale Rationalization (Handling Failed Problems):
For problems where $\hat{y}_i \neq y_i^*$, the model lacked the foresight to discover the reasoning path forward.
However, conditioning on the destination $y_i^*$ makes path search exponentially easier:
1. Construct the rationalization prompt:
   $$x_i^{\text{hint}} = x_i \oplus \text{" [The correct answer is } y_i^* \text{. Explain step-by-step why.]"}$$
2. Sample backward-conditioned rationale:
   $$\hat{z}_i^{\text{hint}} \sim P_{\theta_t}(z \mid x_i^{\text{hint}})$$
3. Verify that $\hat{z}_i^{\text{hint}}$ actually produces $y_i^*$ when presented **without the hint**:
   $$\text{If } \arg\max_y P_{\theta_t}(y \mid x_i, \hat{z}_i^{\text{hint}}) = y_i^* \implies \mathcal{D}_{\text{rationalized}}^{(t)} \leftarrow \mathcal{D}_{\text{rationalized}}^{(t)} \cup \{(x_i, \hat{z}_i^{\text{hint}}, y_i^*)\}$$

#### 3. Maximization Step (M-step: Supervised Parameter Update):
Aggregate the training dataset:
$$\mathcal{D}_{\text{train}}^{(t)} = \mathcal{D}_{\text{filtered}}^{(t)} \cup \mathcal{D}_{\text{rationalized}}^{(t)}$$

Update policy parameters $\theta_{t+1}$ by minimizing the negative log-likelihood:
$$\theta_{t+1} = \arg\min_\theta \sum_{(x, z, y^*) \in \mathcal{D}_{\text{train}}^{(t)}} \left[ - \sum_{k=1}^{|z|} \log P_\theta(z_k \mid x, z_{<k}) - \sum_{m=1}^{|y^*|} \log P_\theta(y_m^* \mid x, z, y_{<m}^*) \right]$$

---

### 2.3 Quiet-STaR: Continuous Unprompted Thought Generation

Quiet-STaR (Zelikman et al., 2024) extends rationale bootstrapping to arbitrary, unstructured text corpora.
Given an arbitrary document sequence $x_{1:T}$:

1. **Parallel Thought Generation:**
   At every token position $t$, the model generates $K$ thoughts in parallel:
   $$z_{t, 1:L_{\text{thought}}} \sim P_\theta(\cdot \mid x_{1:t})$$
2. **Mixing Head (System 1 vs. System 2 Gating):**
   The probability of predicting the next token $x_{t+1}$ interpolates between the fast direct prediction and the thought-augmented prediction via a learned scalar $\alpha_t \in [0, 1]$:
   $$P_{\text{quiet}}(x_{t+1} \mid x_{1:t}) = (1 - \alpha_t) P_\theta(x_{t+1} \mid x_{1:t}) + \alpha_t P_\theta(x_{t+1} \mid x_{1:t}, z_{t})$$
3. **REINFORCE Reward for Thoughts:**
   A thought $z_t$ is rewarded if it increases the log-likelihood of future text over an unthinking baseline:
   $$R_t = \sum_{\tau=1}^H \log P_\theta(x_{t+\tau} \mid x_{1:t}, z_t) - \sum_{\tau=1}^H \log P_\theta(x_{t+\tau} \mid x_{1:t})$$

---

## 3. Geometric & Physical Interpretation

### 3.1 Pruning the Random Walk in Thought Space
In high-dimensional token space, unguided generation is an exponentially diffusing random walk.
The ground-truth answer verification acts as an **absorbing barrier**:
- Random walks that miss the barrier are extinguished (zero gradient weight).
- Random walks that strike the absorbing barrier are reinforced with positive gradient updates.
Over successive STaR iterations, the probability density collapses from a broad, chaotic Gaussian cloud into tight, low-entropy conduits connecting problem $x$ to answer $y^*$.

```
   Token Space
        ▲
        │       Unfiltered Chaos (Iter 0)            STaR Conduit (Iter 3)
        │          \   |   /                           ============
        │           \  |  /                              \       /
        │            \ | /                                \     /
        │             * (x)                                * (x)
        │            / | \                                  │││ (Direct Conduit)
        │           /  |  \                                 ▼▼▼
        │          *   *   * (Random Answers)              * (y*) Exact Target!
        └────────────────────────────────────────► Problem Manifold
```

---

## 4. Real-World Analogy: The Maze Runner with Breadcrumbs

Imagine an explorer placed in a complex labyrinth (the problem $x$):
- **Iteration 0 (Pure Trial and Error):** The explorer runs down random corridors. Out of 100 attempts, they stumble upon the gold exit ($y^*$) by pure luck 5 times.
- **The Filter:** For those 5 successful escapes, the explorer maps out the exact sequence of turns ($z$). They memorize these 5 routes.
- **The Hint (Rationalization):** For the other 95 mazes where they got trapped, the labyrinth keeper stands at the gold exit and shines a bright laser beam into the sky ($y^*$ hint). The explorer works backwards from the light, connects the path, and writes it down.
- **Iteration 1:** With 100 proven maps in hand, the explorer re-enters the labyrinth. Now they escape 50 mazes unaided!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step numerical trace of:
1. **Sampling & Verification Filtering across 3 Training Questions**
2. **Rationalization Hint Generation for Failed Questions**
3. **Dataset Assembly for Iteration 1**
4. **Cross-Entropy Loss Calculation during the M-Step**

---

### 5.1 Concrete Setup & Input Values

We have a training set of $N = 3$ mathematical word problems:
- **Problem 1 ($Q_1$):** *"A farmer has 5 sheep and buys 10 more. How many sheep total?"*
  - Ground Truth: $y_1^* = \mathbf{15}$
- **Problem 2 ($Q_2$):** *"A train travels 3 hours at 14 mph. What is the distance?"*
  - Ground Truth: $y_2^* = \mathbf{42}$
- **Problem 3 ($Q_3$):** *"Solve $2x + 1 = 15$ for $x$."*
  - Ground Truth: $y_3^* = \mathbf{7}$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $Q_i$ | `problem_prompt` | Input question text $x_i$ |
| $y_i^*$ | `ground_truth` | Verified target answer |
| $\hat{z}_i$ | `generated_rationale` | Model's self-generated chain-of-thought |
| $\hat{y}_i$ | `predicted_answer` | Final answer extracted from generated rationale |
| $\hat{z}_i^{\text{hint}}$ | `rationalized_trace` | Rationale generated when conditioned on $y_i^*$ |
| $\mathcal{D}_{\text{train}}$ | `bootstrapped_data` | Filtered + rationalized dataset for M-step fine-tuning |
| $\mathcal{L}_{\text{SFT}}$ | `m_step_loss` | Cross-entropy loss on valid reasoning tokens |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: E-Step Sampling & Verification Filtering

- **For Problem $Q_1$ ($y_1^* = 15$):**
  - Model generates $\hat{z}_1$: `"Starts with 5 sheep. Adds 10 sheep. Total is 5 + 10 = 15."`
  - Model predicts: $\hat{y}_1 = 15$.
  - Verification check: $\hat{y}_1 == y_1^* \implies 15 == 15$ (**PASS!**).
  - Add to dataset: $(Q_1, \hat{z}_1, 15) \in \mathcal{D}_{\text{filtered}}$.

- **For Problem $Q_2$ ($y_2^* = 42$):**
  - Model generates $\hat{z}_2$: `"Distance is speed divided by time. So 14 / 3 = 4.67."`
  - Model predicts: $\hat{y}_2 = 4.67$.
  - Verification check: $\hat{y}_2 == y_2^* \implies 4.67 \neq 42$ (**FAIL!**).
  - Trace is **discarded** to prevent learning flawed physics.

- **For Problem $Q_3$ ($y_3^* = 7$):**
  - Model generates $\hat{z}_3$: `"2x + 1 = 15. Subtract 1: 2x = 14. Divide by 2: x = 7."`
  - Model predicts: $\hat{y}_3 = 7$.
  - Verification check: $\hat{y}_3 == y_3^* \implies 7 == 7$ (**PASS!**).
  - Add to dataset: $(Q_3, \hat{z}_3, 7) \in \mathcal{D}_{\text{filtered}}$.

*Iteration 0 Direct Accuracy:* $\frac{2}{3} = \mathbf{66.67\%}$.

---

#### Step 2: Rationalization on Failed Problem $Q_2$
Condition model on prompt with hint:
$$Q_2^{\text{hint}} = Q_2 \oplus \text{" The correct distance is 42 miles. Explain step-by-step why."}$$

- Model generates backward rationale $\hat{z}_2^{\text{hint}}$:
  `"Distance equals speed multiplied by time. Here speed is 14 mph and time is 3 hours. Distance = 14 * 3 = 42 miles."`
- Model predicts: $\hat{y}_2 = 42$.
- Verification check without hint: $14 \times 3 == 42$ (**PASS with Hint!**).
- Add to rationalized set: $(Q_2, \hat{z}_2^{\text{hint}}, 42) \in \mathcal{D}_{\text{rationalized}}$.

---

#### Step 3: Assemble Bootstrapped Training Set $\mathcal{D}_{\text{train}}$
$$\mathcal{D}_{\text{train}} = \{ (Q_1, \hat{z}_1, 15), \, (Q_2, \hat{z}_2^{\text{hint}}, 42), \, (Q_3, \hat{z}_3, 7) \}$$
All 3 problems now have **valid, verified, mathematically sound rationales**!

---

#### Step 4: M-Step Loss Computation
Suppose during the supervised fine-tuning update on $(Q_2, \hat{z}_2^{\text{hint}}, 42)$, the model predicts the key operation tokens:
- Token 1 (`"multiplied"`): Model assigns probability $P = 0.500000$
- Token 2 (`"14"`): Model assigns probability $P = 0.800000$
- Token 3 (`"*"`): Model assigns probability $P = 0.700000$
- Token 4 (`"42"`): Model assigns probability $P = 0.900000$

Compute individual cross-entropy losses $\ell_t = -\ln(P_t)$:
$$\ell_1 = -\ln(0.50) \approx \mathbf{0.693147}$$
$$\ell_2 = -\ln(0.80) \approx \mathbf{0.223144}$$
$$\ell_3 = -\ln(0.70) \approx \mathbf{0.356675}$$
$$\ell_4 = -\ln(0.90) \approx \mathbf{0.105361}$$

$$\mathcal{L}_{\text{M-Step}} = \frac{0.693147 + 0.223144 + 0.356675 + 0.105361}{4} = \frac{1.378327}{4} = \mathbf{0.344582}$$

In Iteration 1, the fine-tuned model $\theta_1$ will successfully solve $Q_2$ **without any hint**, boosting unaided accuracy from **66.67% to 100%**! $\blacksquare$

---

### 5.4 Summary Visual Grid: STaR Execution Ledger

| Problem | Ground Truth $y^*$ | Pass 1 Prediction $\hat{y}$ | Status | Rationalization Hint Used? | Rationale in Final Dataset $\mathcal{D}_{\text{train}}$ |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$Q_1$** | $15$ | $15$ | **PASS** | No | Self-generated $\hat{z}_1$ |
| **$Q_2$** | $42$ | $4.67$ | **FAIL** | **YES** | Rationalized $\hat{z}_2^{\text{hint}}$ |
| **$Q_3$** | $7$ | $7$ | **PASS** | No | Self-generated $\hat{z}_3$ |
| **Total** | 3/3 in Buffer | $2/3$ (66.7%) | — | 1 Rescued | 100% High-Quality Data |

---

## 6. Solved Illustrations

### Illustration 1: Pathological Case: Preventing the "Cheating" Rationalization
**Problem:**
A model is given a false rationalization hint:
*"Q: What is $2 + 2$? The correct answer is 5. Explain step-by-step why."*
If the model hallucinates: *"In advanced modular arithmetic, $2+2=5$ because we add 1"*, and we add this to the dataset, what catastrophe occurs? How does STaR prevent this?

**Solution:**
1. **Catastrophe (Reward Hacking / Sycophancy):**
   If rationalized traces are accepted unconditionally based on the hint, the model learns to invent false axioms to justify false premises, corrupting arithmetic truth.
2. **STaR's Verification Guardrail:**
   STaR requires that the ground-truth $y^*$ comes from a **trusted external verifier** (e.g. Python REPL or mathematical answer key). Furthermore, the candidate rationalization $\hat{z}^{\text{hint}}$ must be evaluated by running the model on the question **without the hint**. If the unprompted execution does not output $y^*$, the rationalization is rejected! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **DeepSeek-R1 Pipeline (Stage 1 & 2):** Uses a STaR-like rejection sampling methodology on 600K synthetic reasoning trajectories, filtering purely by Python test pass rates and SymPy equivalence before fine-tuning.
- **Quiet-STaR & Implicit Reasoning:** Quiet-STaR demonstrated that teaching a model to reason quietly between tokens improves downstream performance on GSM8K and CommonsenseQA without requiring explicit `<think>` prompts.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - E-step filtering (accepting $Q_1$ and $Q_3$, rejecting $Q_2$).
   - Rationalization rescue of $Q_2$.
   - M-step cross-entropy loss matching $0.344582$ to $< 10^{-6}$.
2. **Complete STaR Bootstrap Engine:**
   - Multi-problem reasoning benchmark simulator.
   - Self-generation, verification filter, hint rationalization, and iterative policy training.
   - Demonstrates accuracy improving from $66.7\%$ to $100\%$.

See implementation in:
[`13_reasoning_and_test_time_compute/code/02_rationale_bootstrapping_star.py`](./code/02_rationale_bootstrapping_star.py)
