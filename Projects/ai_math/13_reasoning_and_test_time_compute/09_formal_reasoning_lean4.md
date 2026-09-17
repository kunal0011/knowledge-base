# 13.9 Formal Reasoning & Interactive Theorem Proving (Lean 4, AlphaProof & Autoformalization)

---

## 1. Intuition & 101 Motivation

Even with Chain-of-Thought, Process Reward Models, and test-time search, natural language mathematical reasoning remains vulnerable to subtle errors:
- **Hand-Wavy Leaps:** Phrases like *"It is obvious that..."* or *"By symmetry, it easily follows..."* often disguise logical fallacies.
- **Missing Edge Cases:** Forgetting that a variable could be zero, negative, or undefined.
- **Hallucinated Theorems:** Generating an algebraic identity that looks plausible but is mathematically false.

In human mathematics, the gold standard of absolute certainty is **Interactive Theorem Provers (ITPs)**, most notably **Lean 4** (alongside Isabelle, Coq, and Metamath).

In Lean 4, a proof is not informal prose; it is **formal, type-checked code**. The mathematical claims are compiled and verified by a tiny, deterministic, trusted mathematical kernel based on the **Curry-Howard Isomorphism** (Propositions as Types). If the Lean 4 compiler reports:
```lean
goals accomplished
```
the proof is **100% sound, rigorous, and verified beyond all doubt**. There is zero room for hallucination.

In July 2024, Google DeepMind achieved a historic milestone at the International Mathematical Olympiad (IMO): **AlphaProof** solved formal Olympiad problems in Lean 4 to achieve an IMO Silver Medal score. This was achieved via **Autoformalization** (translating natural language math into formal Lean statements) and **Tree Search** over formal proof tactics.

```
                   THE FORMAL REASONING PIPELINE
                   
  Natural Language Math Problem (IMO / Competition)
                       │
                       ▼  (Autoformalization via LLM)
  Formal Theorem Statement in Lean 4:
  `theorem imo_2024_p1 (n : ℕ) : ... := by`
                       │
                       ▼  (Test-Time Search / MCTS)
  LLM Proposes Candidate Tactics:
  `intro h`, `apply le_trans`, `omega`, `linarith`
                       │
                       ▼  (Lean 4 Kernel Step Transition)
  Deterministic Compiler Feedback:
  - Error: "Type mismatch" (Branch Pruned)
  - Success: New Sub-Goals Γ ⊢ G'
                       │
                       ▼  (When Goals Reached Zero)
              `goals accomplished`
  (Mathematically Certified IMO Solution: Zero Hallucination!)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Curry-Howard Isomorphism (Propositions as Types)

The theoretical foundation of Lean 4 is the **Calculus of Inductive Constructions (CIC)**:

| Logic Domain | Type Theory / Computer Science Domain |
| :--- | :--- |
| **Mathematical Proposition $P$** | **Type** ($P : \text{Type}$) |
| **Proof $p$ of Proposition $P$** | **Term / Program $p$** of type $P$ ($p : P$) |
| **Conjunction ($P \land Q$)** | **Product Type** ($P \times Q$) |
| **Disjunction ($P \lor Q$)** | **Sum Type** ($P + Q$) |
| **Implication ($P \implies Q$)** | **Function Type** ($P \to Q$) |
| **Universal Quantifier ($\forall x \in A, P(x)$)** | **Dependent Function Type ($\Pi$-type)** |
| **Existential Quantifier ($\exists x \in A, P(x)$)** | **Dependent Pair Type ($\Sigma$-type)** |

#### Theorem (Proof Verification as Type-Checking):
A theorem statement $P$ is true if and only if the type $P$ is **inhabited** by a valid program $p$.
Verifying a proof reduces to **deterministic type-checking**:
$$\operatorname{VerifyProof}(p, P) = \begin{cases} \text{True} & \text{if } \operatorname{InferType}(p) \equiv P \\ \text{False} & \text{otherwise} \end{cases}$$

---

### 2.2 Formal Proof Search as an MDP in Lean 4

Proof construction in Lean operates via **tactics**, forming a discrete Markov Decision Process:

- **State $s_t$ (The Tactic State):**
  A set of active goals. Each goal consists of a local context of hypotheses $\Gamma_t$ and a target proposition $G_t$:
  $$s_t = \{ (\Gamma_{t, 1} \vdash G_{t, 1}), \, (\Gamma_{t, 2} \vdash G_{t, 2}), \, \dots \}$$
- **Action $a_t$ (A Tactic):**
  A formal command (e.g. `intro h`, `cases h`, `exact h`, `ring`, `linarith`).
- **Transition Function:**
  The Lean 4 elaboration kernel executes tactic $a_t$ on goal $G_{t, 1}$:
  $$s_{t+1} = \operatorname{LeanKernel}(s_t, a_t)$$
  - If $a_t$ is mathematically invalid: returns compiler syntax/type error $e$ (invalid state transition, branch immediately killed).
  - If $a_t$ simplifies the goal: returns new sub-goals $s_{t+1}$.
  - If $a_t$ fully solves the goal: remaining goals decrease by 1.
- **Terminal Success State:**
  $$s_{\text{terminal}} = \emptyset \quad (\texttt{"no goals"})$$
  Reward $R = 1.0$ if $s = \emptyset$, else $0.0$.

---

### 2.3 Autoformalization

Most mathematical literature is written in informal English or LaTeX.
**Autoformalization** is the sequence-to-sequence translation of informal natural language statements into formal Lean 4 declarations:

$$S_{\text{natural}} \xrightarrow{\text{LLM Autoformalizer}} S_{\text{Lean}}$$

#### Example:
- **Informal English:** *"If $n$ is an even integer, then $n^2$ is divisible by 4."*
- **Formal Lean 4 Statement:**
  ```lean
  theorem even_square (n : ℤ) (h : Even n) : 4 ∣ n^2 := by
  ```

#### Elaboration Filter:
Autoformalized candidates are verified by feeding them to the Lean 4 compiler (`lean --run`).
If the declaration throws a type error (e.g., using undefined mathematical symbols or mismatched types), the sample is rejected. Only syntactically and semantically valid Lean definitions are admitted to the proof search pipeline.

---

## 3. Geometric & Physical Interpretation

### 3.1 The Tactic Hypergraph
In standard games (chess, Go), each move transitions state $s_t$ to exactly one next state $s_{t+1}$ (a standard graph).
In formal theorem proving, tactics like `induction n` or `constructor` split one goal into **multiple independent sub-goals** (e.g. base case and inductive step).
The state space forms a **Directed AND/OR Hypergraph**:
- **OR Nodes:** Multiple alternative candidate tactics suggested by the LLM (only one needs to succeed).
- **AND Nodes:** Sub-goals created by tactics (all must be proven to close the parent goal).

```
                            [Goal: P ∧ Q]  (OR Node)
                           /             \
            Tactic 1: `constructor`      Tactic 2: `exact ...`
                     │
                     ▼
          ┌─────────────────────┐
          │      AND Node       │
          └──────────┬──────────┘
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
      [Goal: P]           [Goal: Q]  (Both must be proven!)
```

---

## 4. Real-World Analogy: The Legal Contract and the Compiler

- **Natural Language Proof:** An oral agreement between two business partners over lunch. Everyone smiles and nods ("It is obvious we share the profits"). Months later, a lawsuit occurs because neither defined what happens if revenue is negative.
- **Formal Proof in Lean 4:** A computer-verified smart contract on a strict compiler. Every possible variable type, integer overflow, and edge case is rigidly checked. The contract will literally not compile unless every logical loophole is sealed.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step trace of:
1. **Curry-Howard Goal Formulation**
2. **Step-by-Step Tactic State Transitions ($\Gamma \vdash G$)**
3. **Closing Sub-Goals to Reach `no goals`**
4. **MCTS Action Probability Calculation in AlphaProof**

---

### 5.1 Concrete Setup & Target Theorem

- **Theorem Statement:** Prove that logical conjunction is commutative:
  $$P \land Q \implies Q \land P$$
- **Lean 4 Formal Definition:**
  ```lean
  theorem and_comm (P Q : Prop) : P ∧ Q → Q ∧ P := by
  ```

---

### 5.2 What Refers to What: Legend Protocol Table

| Step | Mathematical State | Lean 4 Hypotheses ($\Gamma$) | Target Goal ($G$) | Applied Tactic | Kernel Result |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **0** | Initial State | $\emptyset$ | `P ∧ Q → Q ∧ P` | `intro h` | Goal becomes implication body |
| **1** | Hypothesis Introduced | $h : P \land Q$ | `Q ∧ P` | `rcases h with ⟨hp, hq⟩` | Deconstruct conjunction |
| **2** | Primitive Hypotheses | $hp : P, \quad hq : Q$ | `Q ∧ P` | `exact ⟨hq, hp⟩` | Provide constructor pair |
| **3** | Terminal State | — | $\emptyset$ | — | **`goals accomplished`** |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 0: Initial State ($s_0$)
- Context $\Gamma_0 = \emptyset$
- Goal $G_0 = P \land Q \to Q \land P$
- Tactic applied: $\mathbf{\texttt{intro h}}$
  *(Moves the antecedent $P \land Q$ into the local context as hypothesis $h$).*

---

#### Step 1: State after `intro h` ($s_1$)
- Context $\Gamma_1 = \{ h : P \land Q \}$
- Goal $G_1 = Q \land P$
- Tactic applied: $\mathbf{\texttt{rcases h with ⟨hp, hq⟩}}$
  *(Deconstructs the product type $h$ into its two component proofs: $hp$ of type $P$, and $hq$ of type $Q$).*

---

#### Step 2: State after `rcases` ($s_2$)
- Context $\Gamma_2 = \{ hp : P, \quad hq : Q \}$
- Goal $G_2 = Q \land P$
- Notice: We need a proof of $Q \land P$. Under Curry-Howard, this is a pair $\langle \text{proof of } Q, \text{proof of } P \rangle$.
- We have $hq : Q$ and $hp : P$!
- Tactic applied: $\mathbf{\texttt{exact ⟨hq, hp⟩}}$

---

#### Step 3: Terminal Verification ($s_3$)
The Lean 4 kernel type-checks the term:
$$\operatorname{typeOf}(\langle hq, hp \rangle) = Q \land P \equiv G_2$$
The goal is discharged!
$$\text{Remaining Goals} = \emptyset \implies \mathbf{\texttt{goals accomplished}}$$

---

### 5.4 Summary Visual Grid: Formal Tactic Ledger

| State | Hypotheses ($\Gamma$) | Active Goal ($G$) | Tactic Executed | Goal Status |
| :---: | :--- | :--- | :--- | :---: |
| **$s_0$** | None | `P ∧ Q → Q ∧ P` | `intro h` | Transformed |
| **$s_1$** | $h : P \land Q$ | `Q ∧ P` | `rcases h with ⟨hp, hq⟩` | Decomposed |
| **$s_2$** | $hp : P, \; hq : Q$ | `Q ∧ P` | `exact ⟨hq, hp⟩` | Solved |
| **$s_3$** | — | $\emptyset$ | — | **VERIFIED CERTIFIED** |

---

### 5.5 MCTS Action Probability Calculation

In AlphaProof, tactic selection uses Monte Carlo Tree Search (MCTS) policy priors and Q-values.
Given three candidate tactics proposed by the LLM:
- **Tactic 1 (`linarith`):** Prior $P_1 = 0.50$, Action Value $Q_1 = 0.80$, Visit Count $N_1 = 10$.
- **Tactic 2 (`ring`):** Prior $P_2 = 0.30$, Action Value $Q_2 = 0.10$, Visit Count $N_2 = 5$.
- **Tactic 3 (`omega`):** Prior $P_3 = 0.20$, Action Value $Q_3 = 0.90$, Visit Count $N_3 = 2$.

Total parent visits $N = 10 + 5 + 2 = 17$. $C_{\text{PUCT}} = 1.5$.
1. **Compute UCB Score for Tactic 1:**
   $$U_1 = Q_1 + C_{\text{PUCT}} \times P_1 \times \frac{\sqrt{N}}{1 + N_1} = 0.80 + 1.5 \times 0.50 \times \frac{\sqrt{17}}{1 + 10} = 0.80 + 0.75 \times \frac{4.1231}{11} = 0.80 + 0.2811 = \mathbf{1.0811}$$
2. **Compute UCB Score for Tactic 2:**
   $$U_2 = 0.10 + 1.5 \times 0.30 \times \frac{\sqrt{17}}{1 + 5} = 0.10 + 0.45 \times \frac{4.1231}{6} = 0.10 + 0.3092 = \mathbf{0.4092}$$
3. **Compute UCB Score for Tactic 3:**
   $$U_3 = 0.90 + 1.5 \times 0.20 \times \frac{\sqrt{17}}{1 + 2} = 0.90 + 0.30 \times \frac{4.1231}{3} = 0.90 + 0.4123 = \mathbf{1.3123}$$
4. **Action Selection:**
   Tactic 3 has the highest UCB score ($1.3123 > 1.0811 > 0.4092$). MCTS prioritizes its exploration and selects `omega` for the next simulation step. $\blacksquare$

---

## 6. Solved Illustrations

### Illustration 1: Autoformalization Ambiguity & Grounding
**Problem:**
Examine this human natural language prompt:
*"Prove that if $x$ and $y$ are positive integers, then $x + y > 0$."*
1. What implicit assumption in human language must be formally stated in Lean?
2. Write the exact Lean 4 theorem statement.

**Solution:**
1. **Implicit Assumptions in Natural Language:**
   Humans say "positive integers" loosely. In formal logic, we must specify the exact mathematical set ($\mathbb{N}$ natural numbers or $\mathbb{Z}$ integers) and the strict inequality hypothesis ($x > 0$ and $y > 0$).
2. **Formal Lean 4 Statement:**
   ```lean
   theorem add_pos_of_pos {x y : ℤ} (hx : x > 0) (hy : y > 0) : x + y > 0 := by
     linarith
   ```
   *(The automated tactic `linarith` solves linear arithmetic inequalities instantly using Fourier-Motzkin elimination).* $\blacksquare$

---

### Illustration 2: Lean 4 Type Theory Proof Term Construction

**Problem:**
Construct Curry-Howard proof terms for the following two theorems:
1. Prove $P \to P$ (the identity function on propositions).
2. Prove $A \to B \to A$ (the K combinator).

**Step-by-Step Solution:**

**1. Prove $P \to P$:**
- **Lean 4 Tactic:** `theorem id_prop (P : Prop) : P → P := fun h => h`
- **Type Derivation:**
  (1) Introduce $P : \text{Prop}$.
  (2) Introduce assumption $h : P$.
  (3) Return $h : P$.
- **Lambda Term:** This corresponds exactly to the $\lambda$-term $\lambda P. \lambda h:P. h$.
- **Type Signature:** $\forall P : \text{Prop}, P \to P$.

**2. Prove $A \to B \to A$:**
- **Lean 4 Tactic:** `theorem k_prop (A B : Prop) : A → B → A := fun ha _ => ha`
- **Type Derivation:**
  (1) Introduce $A, B : \text{Prop}$.
  (2) Introduce $ha : A$.
  (3) Introduce \_ : $B$ (ignored).
  (4) Return $ha : A$.
- **Lambda Term:** $\lambda A. \lambda B. \lambda ha:A. \lambda \_:B. ha$.
- **Type Signature:** $\forall A B : \text{Prop}, A \to B \to A$.

The dependent type checker verifies these lambda expressions in $O(1)$ per term, proving the logic seamlessly! $\blacksquare$

---

### Illustration 3: AlphaProof Reward Signal Computation

**Problem:**
Demonstrate the severe sparsity of reward signals in formal theorem proving.
A tactic proof attempt operates on a simplified IMO problem. Evaluate the reward $R$ given by the Lean kernel after multiple distinct tactic steps.

**Step-by-Step Solution:**

**Scenario A:**
- **Proof state after 3 tactics:** Goal remaining: prove $n^2 + n + 1 > 0$ for all $n : \mathbb{Z}$.
- **LLM action:** Proposes tactic `positivity`.
- **Lean response:** Kernel applies tactic, succeeds. Remaining goals = $0$. State is `no goals` (QED).
- **Reward:** $R = \mathbf{+1.0}$ (proof complete).

**Scenario B:**
- **Proof state after 3 different tactics:** Goal remaining: $\sum_{i \in \text{Finset.range } n} (i+1) = n(n+1)/2$.
- **LLM action 1:** Proposes tactic `ring`.
- **Lean response:** `ring` fails (goal is not purely algebraic).
- **Reward:** $R = \mathbf{0.0}$ (not done).
- **LLM action 2:** Proposes tactic `induction n`.
- **Lean response:** Goal splits into 2 complex subgoals (base case and inductive step).
- **Reward:** $R = \mathbf{0.0}$ (still not done).

**Conclusion:** Only the final QED state yields $R=1.0$; everything else, even highly productive mathematical inductions, yields $R=0.0$. This extreme sparsity is exactly why pure gradient descent struggles, and why MCTS requires 800+ simulations per step to surface rare $R=1.0$ signals! $\blacksquare$

---

### Illustration 4: Autoformalization Accuracy Estimation

**Problem:**
Evaluate the autoformalization accuracy across statements of varying complexity.
1. Translate "For all real $x$, $x^2 \ge 0$" into Lean 4 and verify.
2. Translate "The sum of the first $n$ natural numbers is $n(n+1)/2$" into Lean 4 and manually check correctness for $n=4$.
3. Compare LLM success rates vs AlphaProof's round-trip verification system.

**Step-by-Step Solution:**

**1. Natural Language Math 1:**
- Statement: "For all real $x$, $x^2 \ge 0$"
- Formal Lean 4: `∀ x : ℝ, x^2 ≥ 0`
- Correctness: The `sq_nonneg x` tactic closes the goal immediately. Valid!

**2. Natural Language Math 2:**
- Statement: "The sum of the first $n$ natural numbers is $n(n+1)/2$"
- Formal Lean 4: `∑ i in Finset.range (n+1), i = n*(n+1)/2`
- Test for $n=4$:
  LHS: $\sum_{i=0}^4 i = 0 + 1 + 2 + 3 + 4 = \mathbf{10}$.
  RHS: $4 \times 5 / 2 = 20 / 2 = \mathbf{10}$.
  $10 = 10 \checkmark$. Proven in Lean via mathematical induction.

**3. Error Rate Comparison:**
- **Standard GPT-4 on AMC/AIME:** $\approx 35\%$ produce syntactically valid Lean code but structurally *semantically wrong* formalizations (e.g., swapping `∃` and `∀`).
- **AlphaProof System:** Leverages a 3-step loop: Formalize $\to$ Lean checker $\to$ Backtranslate to English $\to$ Match against original prompt. This drives the error rate down to **$<10\%$**, ensuring that the theorems being proven are actually the ones asked by the competition! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **Google DeepMind AlphaProof (IMO 2024):**
  Paired a fine-tuned Gemini model for autoformalization with AlphaZero search in Lean 4 to solve two algebra problems and one number theory problem, securing **28/42 points** (Silver Medal tier).
- **DeepSeek-Prover-V1.5:** Open-source foundation model trained on formal Lean 4 Mathlib repositories, introducing sub-goal tree search and proof completion scoring.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact simulation of the Lean 4 tactic state machine for commutativity of conjunction.
   - Verification of `intro`, `rcases`, and `exact` transitions.
   - Terminal goal closure checking.
2. **Automated Tactic Proof Search Engine:**
   - Formal environment state transition kernel.
   - Breadth-First Search (BFS) and MCTS tactic exploration engine.
   - Demonstrates complete automatic resolution of formal logic goals.

See implementation in:
[`13_reasoning_and_test_time_compute/code/09_formal_reasoning_lean4.py`](./code/09_formal_reasoning_lean4.py)
