# 13.4 Process Reward Models (PRMs) & Step-Level Value Estimation (PRM800K & Math-Shepherd)

---

## 1. Intuition & 101 Motivation

In traditional reinforcement learning and alignment (RLHF, DPO, Best-of-$N$), reward models are **Outcome-based Reward Models (ORMs)**. An ORM inspects only the final answer at the very end of a generation, assigning a single scalar reward ($R \in \{0, 1\}$ or $R \in \mathbb{R}$) to the entire trajectory.

Outcome supervision suffers from the **Credit Assignment Catastrophe**:
1. **False Positives (Lucky Hallucinations):** A 10-step mathematical solution makes a catastrophic algebraic blunder on step 2, but due to a second cancelling mistake on step 7, the final answer happens to match the ground truth. An ORM rewards the entire flawed trajectory with $+1$, teaching the model invalid logic!
2. **False Negatives (Unforgiving Penalties):** A model writes 9 steps of brilliant mathematical deduction, but makes a tiny arithmetic slip on the 10th step. An ORM penalizes the entire trajectory with $0$, discouraging creative, sophisticated exploration.

**Process Supervision** (Lightman et al., OpenAI 2023 - *"Let's Verify Step by Step"*) solves credit assignment by evaluating and scoring **each individual reasoning step**:
$$\text{Trajectory } z = (z_1, z_2, \dots, z_K) \implies \text{Step Scores } (r_1, r_2, \dots, r_K)$$

A **Process Reward Model (PRM)** acts as a fine-grained step-level value function $V(s_t)$. It pinpoints the exact token index where an error occurs, enabling search algorithms (Beam Search, MCTS) to prune bad reasoning branches the moment they diverge!

To overcome the human annotation bottleneck (OpenAI's PRM800K required 800,000 human step labels), **Math-Shepherd** (Wang et al., 2023) automates PRM training via **Monte Carlo rollouts**: by sampling multiple completions from an intermediate step, the empirical fraction of completions reaching the correct answer serves as the ground-truth step value.

```
                    OUTCOME vs. PROCESS SUPERVISION
                    
  Outcome Supervision (ORM)
  Step 1 (Good) ──► Step 2 (Fatal Error!) ──► Step 3 (Flawed) ──► y (Wrong)
  Reward: [                      ORM Score: 0.0                           ]
  (Zero localized feedback: Cannot tell where the mistake happened)
  
  Process Supervision (PRM)
  Step 1 (Good) ────────► Step 2 (Fatal Error!) ────────► Step 3 (Pruned)
  PRM Score: +0.95        PRM Score: -0.85 (FLAW DETECTED!)
  (Immediate step-level pruning: Prevents wasting compute on doomed branches)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Reasoning Process as a Markov Decision Process (MDP)

Let problem $x$ define the initial state $s_0$.
At each reasoning step $t \in \{1, 2, \dots, K\}$:
- **State $s_t$:** The problem prompt concatenated with all reasoning steps generated so far:
  $$s_t = (x, z_1, z_2, \dots, z_t)$$
- **Action $a_t = z_{t+1}$:** The next sentence, equation, or thought step.
- **Deterministic Transition:** $s_{t+1} = s_t \circ z_{t+1}$.

---

### 2.2 ORM vs. PRM Loss Formulations

#### 1. Outcome Reward Model (ORM) Loss:
Given a dataset of full trajectories with binary correctness labels $y^* \in \{0, 1\}$:
$$\mathcal{L}_{\text{ORM}}(\phi) = - \mathbb{E} \left[ y^* \log \sigma(R_\phi(s_K)) + (1 - y^*) \log (1 - \sigma(R_\phi(s_K))) \right]$$

#### 2. Process Reward Model (PRM) Loss:
Given a dataset where each step $t \in \{1, \dots, K\}$ has a step correctness label $y_t \in \{0, 1\}$:
$$\mathcal{L}_{\text{PRM}}(\psi) = - \frac{1}{K} \sum_{t=1}^K \left[ y_t \log \sigma(r_\psi(s_t)) + (1 - y_t) \log (1 - \sigma(r_\psi(s_t))) \right]$$
where $r_\psi(s_t) \in \mathbb{R}$ is the scalar logit predicted by the PRM at the final token delimiter (e.g. `\n\n`) of step $t$.

---

### 2.3 Automated Process Supervision: Math-Shepherd Monte Carlo Rollouts

Rather than paying human annotators to verify millions of steps, Math-Shepherd uses the policy model $\pi_\theta$ to estimate the **empirical state value**:

For an intermediate prefix state $s_t = (x, z_{1:t})$:
1. Sample $N_{\text{roll}}$ independent completions to the final answer:
   $$\hat{z}_{t+1:K}^{(j)} \sim \pi_\theta(\cdot \mid s_t), \quad j \in \{1, 2, \dots, N_{\text{roll}}\}$$
2. Evaluate each completion against the ground-truth final answer $y^*$:
   $$\mathbb{I}^{(j)} = \mathbb{I}\left( \operatorname{ExtractAnswer}(\hat{z}_{t+1:K}^{(j)}) = y^* \right)$$
3. The empirical Monte Carlo value of state $s_t$ is:
   $$V_{\text{MC}}(s_t) = \frac{1}{N_{\text{roll}}} \sum_{j=1}^{N_{\text{roll}}} \mathbb{I}^{(j)} \in [0, 1]$$

#### Step Correctness Label Derivation:
- If $V_{\text{MC}}(s_t) \ge \tau_{\text{pos}}$: Step $t$ is labeled **correct** ($y_t = 1$).
- If $V_{\text{MC}}(s_{t-1}) - V_{\text{MC}}(s_t) > \Delta_{\text{drop}}$: Step $t$ caused the derivation to fail and is labeled **incorrect** ($y_t = 0$).

---

### 2.4 Trajectory Aggregation Functions: Product vs. Min

When using a trained PRM to rank candidate solutions during Best-of-$N$ or Beam Search, how should individual step probabilities $p_t = \sigma(r_\psi(s_t))$ be aggregated into a full trajectory score?

#### 1. Product Aggregation (Joint Probability):
$$R_{\text{prod}}(z_{1:K}) = \prod_{t=1}^K p_t = \exp\left( \sum_{t=1}^K \log p_t \right)$$
*Pathology:* Severely penalizes long trajectories. A 20-step proof where every step is $95\%$ confident receives $0.95^{20} \approx \mathbf{0.358}$, scoring lower than a 2-step guess with $65\%$ confidence ($0.65^2 \approx \mathbf{0.4225}$)!

#### 2. Min Aggregation (The Weakest Link Principle):
$$R_{\text{min}}(z_{1:K}) = \min_{1 \le t \le K} p_t$$
*Mathematical Rationale:* In formal logic and mathematical proof, deductive truth is a **conjunctive chain**:
$$\text{Proof Valid} = \text{Step}_1 \land \text{Step}_2 \land \dots \land \text{Step}_K$$
If even a single step has $p_t < 0.5$, the proof is invalid regardless of how brilliant the other 19 steps are. Min aggregation is completely invariant to sequence length, eliminating length penalty bias!

---

## 3. Geometric & Physical Interpretation

### 3.1 Potential Flow and the Error Cliff
In the geometry of reasoning search trees, the empirical value $V(s_t)$ acts as a conserved potential function.
- Along a sound mathematical derivation, $V(s_t)$ remains high ($0.8 \to 0.85 \to 0.9 \to 1.0$).
- When an algebraic blunder occurs at step $k$, the trajectory drops off a **vertical potential cliff** ($V(s_{k-1}) = 0.80 \to V(s_k) = 0.00$).
The PRM detects this sharp negative spatial gradient $\nabla_t V(s_t) \ll 0$, acting as an early warning sensor.

```
   State Value V(s_t)
        ▲
   1.0  │  *─────* (Step 1: Sound Logic)
        │         \
        │          \  Fatal Error Cliff! (Step 2: Sign error)
        │           \
   0.0  │            *─────────* (Step 3: Pointless math on wrong state)
        └────────────────────────────────────────► Reasoning Step (t)
```

---

## 4. Real-World Analogy: The Flying Lesson

- **Outcome Supervision (ORM):**
  A student pilot flies a solo cross-country flight. The flight instructor waits at the destination runway. If the plane lands safely on the runway, the instructor gives an A+. (The student may have stalled the engine twice, violated restricted airspace, and flown upside down, but survived by pure luck).
- **Process Supervision (PRM):**
  The instructor sits in the cockpit with a checklist.
  - Step 1: Pre-flight checklist completed? Check (+1).
  - Step 2: Altimeter set correctly? Check (+1).
  - Step 3: Forget to extend landing gear? **FAIL (-1)**.
  The instructor immediately takes the controls, preventing a crash.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step numerical trace of:
1. **Monte Carlo Step Value Estimation across a 3-Step Derivation**
2. **Automated Step Label Assignment ($y_t \in \{0, 1\}$)**
3. **PRM Logit Evaluation & Binary Cross-Entropy Loss**
4. **Trajectory Aggregation: Product vs. Min Ranking**

---

### 5.1 Concrete Setup & Input Values

- **Problem:** Solve $3x - 5 = 10$. (True solution: $x = 5$).
- **Candidate 3-Step Trajectory:**
  - **Step 1 ($z_1$):** *"Add 5 to both sides: $3x = 15$."* (Mathematically correct).
  - **Step 2 ($z_2$):** *"Subtract 3: $x = 15 - 3 = 12$."* (**Fatal Blunder!** Subtracted 3 instead of dividing).
  - **Step 3 ($z_3$):** *"Therefore, the answer is 12."* (Consistent with Step 2, but wrong).

- **Monte Carlo Rollouts ($N_{\text{roll}} = 4$ independent rollouts per prefix):**
  - From Step 1 prefix: 3 rollouts reach correct answer $5$, 1 rollout fails.
  - From Step 2 prefix: 0 rollouts reach correct answer $5$, 4 rollouts fail.
  - From Step 3 prefix: 0 rollouts reach correct answer $5$, 4 rollouts fail.

- **PRM Predicted Logits for each step:**
  - Step 1 logit: $r_1 = +1.500000$
  - Step 2 logit: $r_2 = -1.000000$
  - Step 3 logit: $r_3 = -2.000000$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $V_{\text{MC}}(s_t)$ | `mc_value` | Fraction of rollouts reaching correct answer from step $t$ |
| $y_t$ | `step_label` | Derived ground-truth correctness label ($1$ = correct, $0$ = flawed) |
| $r_t$ | `prm_logit` | Raw scalar logit output from PRM at step delimiter |
| $p_t = \sigma(r_t)$ | `step_prob` | PRM confidence probability $\frac{1}{1 + e^{-r_t}}$ |
| $\ell_t$ | `step_loss` | Binary cross-entropy loss for step $t$ |
| $R_{\text{min}}, R_{\text{prod}}$ | `min_score`, `prod_score` | Aggregated trajectory scores across the 3 steps |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Monte Carlo State Values
- **Step 1 ($s_1$):**
  $$V_{\text{MC}}(s_1) = \frac{1 + 1 + 1 + 0}{4} = \frac{3}{4} = \mathbf{0.750000}$$
- **Step 2 ($s_2$):**
  $$V_{\text{MC}}(s_2) = \frac{0 + 0 + 0 + 0}{4} = \frac{0}{4} = \mathbf{0.000000}$$
- **Step 3 ($s_3$):**
  $$V_{\text{MC}}(s_3) = \frac{0 + 0 + 0 + 0}{4} = \frac{0}{4} = \mathbf{0.000000}$$

#### Step 2: Assign Step Correctness Labels
- Step 1: $V_{\text{MC}}(s_1) = 0.75 > 0.50 \implies \mathbf{y_1 = 1}$ (Correct step).
- Step 2: Value dropped from $0.75 \to 0.00$ ($\Delta V = -0.75$) $\implies \mathbf{y_2 = 0}$ (**Fatal Error Step!**).
- Step 3: $V_{\text{MC}}(s_3) = 0.00 \le 0.50 \implies \mathbf{y_3 = 0}$ (Incorrect continuation).

$$\text{Step Labels } y = [1, 0, 0]$$

---

#### Step 3: Compute PRM Probabilities and Cross-Entropy Loss

1. **Step 1 ($r_1 = +1.50, y_1 = 1$):**
   $$p_1 = \sigma(1.50) = \frac{1}{1 + e^{-1.50}} = \frac{1}{1 + 0.223130} \approx \mathbf{0.817574}$$
   $$\ell_1 = -\ln(p_1) = -\ln(0.817574) \approx \mathbf{0.201413}$$

2. **Step 2 ($r_2 = -1.00, y_2 = 0$):**
   $$p_2 = \sigma(-1.00) = \frac{1}{1 + e^{1.00}} = \frac{1}{1 + 2.718282} \approx \mathbf{0.268941}$$
   $$\ell_2 = -\ln(1 - p_2) = -\ln(1 - 0.268941) = -\ln(0.731059) \approx \mathbf{0.313262}$$

3. **Step 3 ($r_3 = -2.00, y_3 = 0$):**
   $$p_3 = \sigma(-2.00) = \frac{1}{1 + e^{2.00}} = \frac{1}{1 + 7.389056} \approx \mathbf{0.119203}$$
   $$\ell_3 = -\ln(1 - p_3) = -\ln(1 - 0.119203) = -\ln(0.880797) \approx \mathbf{0.126928}$$

4. **Mean PRM Loss:**
   $$\mathcal{L}_{\text{PRM}} = \frac{\ell_1 + \ell_2 + \ell_3}{3} = \frac{0.201413 + 0.313262 + 0.126928}{3} = \frac{0.641603}{3} = \mathbf{0.213868}$$

---

#### Step 4: Compute Trajectory Aggregations
- **Product Aggregation:**
  $$R_{\text{prod}} = p_1 \times p_2 \times p_3 = 0.817574 \times 0.268941 \times 0.119203 \approx \mathbf{0.026211}$$
- **Min Aggregation:**
  $$R_{\text{min}} = \min(0.817574, 0.268941, 0.119203) = \mathbf{0.119203}$$
Both metrics correctly identify the trajectory as completely rejected ($< 0.15$), but Min aggregation identifies Step 3 / Step 2 as the decisive bottleneck! $\blacksquare$

---

### 5.4 Summary Visual Grid: PRM Verification Ledger

| Step ($t$) | Reasoning Content | MC Rollouts | MC Value | True Label ($y_t$) | PRM Logit ($r_t$) | PRM Prob ($p_t$) | Step Loss ($\ell_t$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 1** | $3x = 15$ | $3/4$ Pass | $0.7500$ | **1 (Good)** | $+1.5000$ | $0.8176$ | **$0.201413$** |
| **Step 2** | $x = 15 - 3$ | $0/4$ Pass | $0.0000$ | **0 (Blunder)** | $-1.0000$ | $0.2689$ | **$0.313262$** |
| **Step 3** | $x = 12$ | $0/4$ Pass | $0.0000$ | **0 (Flawed)** | $-2.0000$ | $0.1192$ | **$0.126928$** |
| **Mean** | Full Chain | — | — | — | — | — | $\mathbf{0.213868}$ |

---

## 6. Solved Illustrations

### Illustration 1: Why Min Aggregation Outperforms Product Aggregation on Long Proofs
**Problem:**
Consider two proofs for an AIME geometry problem:
- Proof A (Direct and short): 3 steps. PRM probabilities: $[0.70, 0.70, 0.70]$.
- Proof B (Rigorous, formal, comprehensive): 20 steps. PRM probabilities: $0.90$ for all 20 steps.
Which proof is selected under Product aggregation vs. Min aggregation?

**Solution:**
1. **Product Aggregation:**
   $$R_{\text{prod}}(A) = (0.70)^3 = \mathbf{0.3430}$$
   $$R_{\text{prod}}(B) = (0.90)^{20} \approx \mathbf{0.1216}$$
   Product aggregation selects Proof A ($0.3430 > 0.1216$), heavily penalizing the rigorous 20-step proof simply because it has more steps!
2. **Min Aggregation:**
   $$R_{\text{min}}(A) = \min(0.70) = \mathbf{0.7000}$$
   $$R_{\text{min}}(B) = \min(0.90) = \mathbf{0.9000}$$
   Min aggregation correctly selects Proof B ($0.90 > 0.70$), recognizing that every single step in Proof B has a higher certainty threshold. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **OpenAI PRM800K:** Trained on 800,000 human step-level annotations on the MATH dataset, proving that process supervision beats outcome supervision by over 15% on competition problems.
- **Qwen2.5-Math-PRM & Math-Shepherd:** Open-weights process reward models that power Monte Carlo Tree Search and step-level Best-of-$N$ selection in state-of-the-art mathematical agents.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Monte Carlo rollout value estimation.
   - Step probability conversions.
   - Exact binary cross-entropy loss matching $0.213868$ to $< 10^{-6}$.
   - Product vs. Min trajectory aggregation.
2. **Production PyTorch PRM Module:**
   - Step-level reward model head predicting logits at delimiter tokens.
   - Beam search simulator guided by PRM step-level pruning.

See implementation in:
[`13_reasoning_and_test_time_compute/code/04_process_reward_models_prm.py`](./code/04_process_reward_models_prm.py)
