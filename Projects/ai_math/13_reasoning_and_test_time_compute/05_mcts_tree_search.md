# 13.5 Tree Search & Monte Carlo Tree Search (MCTS) for Language Models (Tree of Thoughts & PUCT)

---

## 1. Intuition & 101 Motivation

Autoregressive token generation in standard LLMs is strictly linear: the model commits to every token sequentially without any mechanism to look ahead, explore parallel hypotheses, or backtrack when a dead end is reached. Once an erroneous token is emitted, it enters the model's own context window, dragging downstream reasoning into compounding hallucinations.

In 1997, Deep Blue defeated Garry Kasparov not through intuition alone, but by combining heuristic evaluation with **tree search** (evaluating millions of future board states). In 2016, AlphaGo mastered Go by coupling deep neural networks with **Monte Carlo Tree Search (MCTS)**.

Bringing tree search to Large Language Models bridges the gap between neural intuition and algorithmic planning:
1. **Tree of Thoughts (ToT)** (Yao et al., 2023): Generalizes Chain-of-Thought from a single linear thread into a tree structure where nodes are coherent "thoughts" (steps, lemmas, sub-goals) and branches represent alternative approaches.
2. **MCTS with PUCT (Predictor Upper Confidence Bound for Trees):** Mathematically balances **Exploitation** (pursuing reasoning paths that have yielded high reward) with **Exploration** (investigating unvisited thoughts that the prior policy considers promising).

By utilizing a pre-trained LLM as the **Policy Network** ($\pi_\theta$) and a Process Reward Model as the **Value Network** ($V_\psi$), MCTS allows an LLM to dynamically explore thousands of potential proof steps at test-time, discover creative solutions, and discard blunders with mathematical precision.

```
                     MCTS REASONING CYCLE
                     
            Selection (PUCT)              Expansion (Policy LLM)
            Find leaf with max Q + U      Generate candidate thought steps
                 (s_0)                         (s_0)
                /     \                       /     \
             (s_1)    (s_2)*               (s_1)    (s_2)
                                                   /  |  \
                                                 (a) (b) (c)
                                                 
            Evaluation (PRM)              Backpropagation
            Score leaf via PRM Value      Propagate score up the tree
                 (s_0)                         (s_0) ◄── N+=1, W+=v
                /     \                       /     \
             (s_1)    (s_2)                (s_1)    (s_2) ◄── N+=1, W+=v
                     /  |  \                       /  |  \
                   (a) (b) (c)*                  (a) (b) (c) ◄── v = 1.0
                        │
                        ▼
                     PRM Score: v = 1.0
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Language Reasoning Tree Structure

Let the problem prompt $x$ define the root node $s_0$.
Each node in the search tree represents a partial reasoning state:
$$s = (x, z_1, z_2, \dots, z_t)$$
An edge $(s, a)$ represents the generation of the next coherent reasoning thought $a = z_{t+1}$.

Each state-action edge $(s, a)$ maintains four critical statistics:
1. **Visit Count $N(s, a) \in \mathbb{N}$:** The number of times action $a$ was selected from state $s$.
2. **Total Action-Value $W(s, a) \in \mathbb{R}$:** The accumulated value scores backpropagated through edge $(s, a)$.
3. **Mean Action-Value $Q(s, a) \in [0, 1]$:**
   $$Q(s, a) = \frac{W(s, a)}{N(s, a)}$$
   (If $N(s, a) = 0$, $Q(s, a) = Q_{\text{init}}$, typically $0.0$ or parent value).
4. **Prior Policy Probability $P(s, a) \in [0, 1]$:**
   $$P(s, a) = P_\theta(a \mid s)$$
   The prior likelihood assigned to thought $a$ by the base generator policy model.

---

### 2.2 The Four Phases of MCTS

Each MCTS iteration executes four sequential operations:

#### Phase 1: Selection (PUCT Formula)
Starting at root $s_0$, recursively descend the tree by selecting the action that maximizes the **Predictor Upper Confidence Bound applied to Trees (PUCT)**:

$$a^* = \arg\max_{a \in \mathcal{A}(s)} \left[ Q(s, a) + U(s, a) \right]$$

where the exploration bonus $U(s, a)$ is:

$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

- $c_{\text{puct}} > 0$: A constant balancing exploration and exploitation.
- $P(s, a)$: Prior guidance (favors thoughts the LLM deems natural).
- $\frac{\sqrt{N(s)}}{1 + N(s, a)}$: Decays as edge $(s, a)$ is visited repeatedly, forcing the tree to explore less-visited branches.

#### Phase 2: Expansion
When selection reaches an unexpanded leaf state $s_L$:
1. The base LLM samples $K$ candidate next reasoning thoughts:
   $$\{a^{(1)}, a^{(2)}, \dots, a^{(K)}\} \sim P_\theta(\cdot \mid s_L)$$
2. Compute prior probabilities for each child: $P(s_L, a^{(k)})$.
3. Initialize child statistics: $N(s_L, a^{(k)}) = 0, W(s_L, a^{(k)}) = 0, Q(s_L, a^{(k)}) = 0$.

#### Phase 3: Evaluation
Estimate the state value $v(s_L) \in [0, 1]$ of the newly expanded leaf using one of two methods:
1. **PRM Value Evaluation:** Pass state $s_L$ to a Process Reward Model:
   $$v(s_L) = \sigma(r_\psi(s_L))$$
2. **Monte Carlo Rollout:** Complete the trajectory to the terminal answer using fast greedy sampling and verify against ground truth:
   $$v(s_L) = \mathbb{I}(\text{Answer} == y^*)$$

#### Phase 4: Backpropagation
Traverse the selected trajectory backwards from leaf $s_L$ to root $s_0$.
For every visited edge $(s, a)$ in the path:
$$N(s, a) \leftarrow N(s, a) + 1$$
$$W(s, a) \leftarrow W(s, a) + v(s_L)$$
$$Q(s, a) \leftarrow \frac{W(s, a)}{N(s, a)}$$

---

### 2.3 Terminal Action Selection at Root

After $M$ iterations of MCTS, the search concludes.
The next reasoning thought executed at the root is chosen proportional to its visit count:

$$\pi_{\text{search}}(a \mid s_0) = \frac{N(s_0, a)^{1/\tau}}{\sum_b N(s_0, b)^{1/\tau}}$$

Under greedy inference ($\tau \to 0$):
$$a^* = \arg\max_a N(s_0, a)$$
*Property:* MCTS selects based on **visit count $N$**, not raw value $Q$. Visit count is far more robust against outlier noise because a node can only achieve high $N$ if it was repeatedly verified across many independent rollouts!

---

## 3. Geometric & Physical Interpretation

### 3.1 Asymmetric Tree Condensation
In an unguided search, the tree explodes exponentially with branching factor $b$ and depth $d$ ($O(b^d)$ nodes).
MCTS with PUCT acts like a **directional laser**:
- Low-value branches are pruned after 1 or 2 visits ($U \to 0$ as parent visits climb).
- High-value branches form a narrow, deep condensation column (filament) penetrating hundreds of steps into the solution space.

```
   Uniform Breadth Search (Explosion)          MCTS PUCT Search (Laser Filament)
               (Root)                                     (Root)
              /  |   \                                      │
            *    *    *                                     ▼
           /|\  /|\  /|\                                  (Node)
          * * * * * * * * *                                 │
                                                            ▼
                                                          (Node)
                                                         /  │  \
                                                        *   *   * (Deep Leaf)
```

---

## 4. Real-World Analogy: The Detective's Whiteboard

Imagine a detective investigating a complex criminal case:
- **Linear Generation:** The detective writes down the first suspect that comes to mind, arrests them, and closes the case. (High blunder rate).
- **MCTS on the Whiteboard:**
  - The detective writes down 3 possible theories ($a_1, a_2, a_3$).
  - Theory 1 initially seemed likely ($P = 0.70$), but phone records showed an alibi ($Q = 0.60$).
  - Theory 2 seemed unlikely at first ($P = 0.30$), but forensic evidence matched ($Q = 0.90$).
  - PUCT tells the detective: *"Spend your morning investigating Theory 2!"*
  - After 10 interviews ($N=10$), Theory 2 leads directly to the stolen goods ($v=1.0$). Case solved!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step numerical trace of:
1. **PUCT Calculation across Two Candidate Thoughts at Root $s_0$**
2. **Exploitation ($Q$) vs. Exploration ($U$) Trade-off**
3. **Selection Decision**
4. **Leaf Evaluation & Upward Backpropagation Update**

---

### 5.1 Concrete Setup & Input Values

- **Parent Root Node $s_0$:**
  - Total visits: $\sum_b N(s_0, b) = 3 + 1 = \mathbf{4}$
  - Exploration constant: $c_{\text{puct}} = \mathbf{2.0}$
- **Action 1 ($a_1$):**
  - Prior policy probability: $P(s_0, a_1) = 0.70$ (LLM's intuitive favorite)
  - Visit count: $N(s_0, a_1) = 3$
  - Value accumulator: $W(s_0, a_1) = 1.80 \implies Q(s_0, a_1) = \frac{1.80}{3} = \mathbf{0.600000}$
- **Action 2 ($a_2$):**
  - Prior policy probability: $P(s_0, a_2) = 0.30$ (LLM's less intuitive alternative)
  - Visit count: $N(s_0, a_2) = 1$
  - Value accumulator: $W(s_0, a_2) = 0.90 \implies Q(s_0, a_2) = \frac{0.90}{1} = \mathbf{0.900000}$
- **Newly evaluated leaf value under $a_2$:** $v = \mathbf{1.000000}$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $N(s, a)$ | `visit_count` | Number of times action $a$ has been traversed |
| $Q(s, a)$ | `mean_value` | Average empirical reward $W / N$ (Exploitation score) |
| $P(s, a)$ | `prior_prob` | Base LLM policy prior probability |
| $U(s, a)$ | `puct_bonus` | Exploration bonus $c_{\text{puct}} P \frac{\sqrt{\sum N}}{1 + N}$ |
| $\operatorname{PUCT}(a)$ | `total_score` | Sum $Q(s, a) + U(s, a)$ used for selection |
| $v$ | `leaf_eval` | Value score returned by PRM or rollout ($1.0$) |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Exploration Scale Factor
Total parent visits:
$$N_{\text{parent}} = \sum_b N(s_0, b) = 3 + 1 = 4.0$$
$$\sqrt{N_{\text{parent}}} = \sqrt{4.0} = \mathbf{2.000000}$$

---

#### Step 2: Compute PUCT Score for Action $a_1$
1. **Exploration Bonus $U(s_0, a_1)$:**
   $$U(s_0, a_1) = c_{\text{puct}} \cdot P(s_0, a_1) \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N(s_0, a_1)}$$
   $$= 2.0 \times 0.70 \times \frac{2.0}{1 + 3} = 1.40 \times \frac{2.0}{4.0} = 1.40 \times 0.50 = \mathbf{0.700000}$$
2. **Total Selection Score $\operatorname{PUCT}(a_1)$:**
   $$\operatorname{PUCT}(a_1) = Q(s_0, a_1) + U(s_0, a_1) = 0.600000 + 0.700000 = \mathbf{1.300000}$$

---

#### Step 3: Compute PUCT Score for Action $a_2$
1. **Exploration Bonus $U(s_0, a_2)$:**
   $$U(s_0, a_2) = c_{\text{puct}} \cdot P(s_0, a_2) \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N(s_0, a_2)}$$
   $$= 2.0 \times 0.30 \times \frac{2.0}{1 + 1} = 0.60 \times \frac{2.0}{2.0} = 0.60 \times 1.00 = \mathbf{0.600000}$$
2. **Total Selection Score $\operatorname{PUCT}(a_2)$:**
   $$\operatorname{PUCT}(a_2) = Q(s_0, a_2) + U(s_0, a_2) = 0.900000 + 0.600000 = \mathbf{1.500000}$$

---

#### Step 4: Selection Decision
$$\operatorname{PUCT}(a_2) = 1.500000 > \operatorname{PUCT}(a_1) = 1.300000 \implies \mathbf{\text{Select Action } a_2!}$$

*Insight:* Although the LLM's intuition preferred $a_1$ ($P=0.70$ vs $0.30$), MCTS correctly chooses $a_2$ because $a_2$ has demonstrated high empirical value ($Q=0.90$) and is relatively underexplored ($N=1$ vs $3$).

---

#### Step 5: Backpropagation Update on Selected Action $a_2$ ($v = 1.0$)
1. **Increment Visit Count:**
   $$N_{\text{new}}(s_0, a_2) = N_{\text{old}}(s_0, a_2) + 1 = 1 + 1 = \mathbf{2}$$
2. **Update Value Accumulator:**
   $$W_{\text{new}}(s_0, a_2) = W_{\text{old}}(s_0, a_2) + v = 0.90 + 1.00 = \mathbf{1.900000}$$
3. **Update Mean Action-Value:**
   $$Q_{\text{new}}(s_0, a_2) = \frac{W_{\text{new}}}{N_{\text{new}}} = \frac{1.90}{2} = \mathbf{0.950000}$$
4. **Parent Root Total Visits:**
   $$N_{\text{new}}(s_0) = 4 + 1 = \mathbf{5} \quad \blacksquare$$

---

### 5.4 Summary Visual Grid: PUCT Selection Ledger

| Action | Prior $P(s, a)$ | Visits $N(s, a)$ | Value $Q(s, a)$ | Expl. Bonus $U(s, a)$ | $\operatorname{PUCT} = Q + U$ | Selected? | Post-Backprop $Q_{\text{new}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$a_1$** | $0.7000$ | $3$ | $0.6000$ | $0.7000$ | $1.3000$ | No | $0.6000$ (Unchanged) |
| **$a_2$** | $0.3000$ | $1$ | $0.9000$ | $0.6000$ | **$1.5000$** | **YES** | **$0.9500$** |

---

## 6. Solved Illustrations

### Illustration 1: How $c_{\text{puct}}$ Prevents Premature Collapse
**Problem:**
Suppose Action 1 gets lucky on visit 1: $v=1.0 \implies Q(a_1)=1.0, N(a_1)=1$.
Action 2 has not been visited yet: $N(a_2)=0, Q(a_2)=0$.
Priors are equal: $P(a_1) = P(a_2) = 0.50$. Total parent visits $N(s) = 1$.
What is the minimum value of $c_{\text{puct}}$ required to ensure Action 2 is explored on the next iteration?

**Solution:**
We require $\operatorname{PUCT}(a_2) > \operatorname{PUCT}(a_1)$.
- For $a_1$:
  $$\operatorname{PUCT}(a_1) = 1.0 + c_{\text{puct}} \cdot 0.50 \cdot \frac{\sqrt{1}}{1 + 1} = 1.0 + 0.25 \, c_{\text{puct}}$$
- For $a_2$:
  $$\operatorname{PUCT}(a_2) = 0.0 + c_{\text{puct}} \cdot 0.50 \cdot \frac{\sqrt{1}}{1 + 0} = 0.50 \, c_{\text{puct}}$$
We solve:
$$0.50 \, c_{\text{puct}} > 1.0 + 0.25 \, c_{\text{puct}} \implies 0.25 \, c_{\text{puct}} > 1.0 \implies \mathbf{c_{\text{puct}} > 4.0}$$
*(If $c_{\text{puct}} \le 4.0$, search would repeatedly exploit Action 1 without ever testing Action 2! This shows why tuning $c_{\text{puct}}$ is critical for balanced exploration).* $\blacksquare$

---

### Illustration 2: PUCT Score Computation for 4 Child Nodes

**Problem:**
Parent visit count $N=100$. Children stats:
- Node A: $Q=0.72$, $N_a=45$, $P=0.4$
- Node B: $Q=0.58$, $N_b=30$, $P=0.3$
- Node C: $Q=0.81$, $N_c=15$, $P=0.2$
- Node D: $Q=0.35$, $N_d=10$, $P=0.1$
Compute PUCT $U(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N}}{1 + N_a}$ with $c_{\text{puct}} = 1.5$. Which node is selected?

**Step-by-Step Solution:**
1. **Node A:**
   $$U_A = 0.72 + 1.5 \times 0.4 \times \frac{\sqrt{100}}{1 + 45} = 0.72 + 0.6 \times \frac{10}{46} = 0.72 + 0.1304 = \mathbf{0.8504}$$
2. **Node B:**
   $$U_B = 0.58 + 1.5 \times 0.3 \times \frac{10}{1 + 30} = 0.58 + 0.45 \times \frac{10}{31} = 0.58 + 0.1452 = \mathbf{0.7252}$$
3. **Node C:**
   $$U_C = 0.81 + 1.5 \times 0.2 \times \frac{10}{1 + 15} = 0.81 + 0.3 \times \frac{10}{16} = 0.81 + 0.1875 = \mathbf{0.9975}$$
4. **Node D:**
   $$U_D = 0.35 + 1.5 \times 0.1 \times \frac{10}{1 + 10} = 0.35 + 0.15 \times \frac{10}{11} = 0.35 + 0.1364 = \mathbf{0.4864}$$

**Result Table:**
| Node | $Q$ | $U_{\text{bonus}}$ | $\text{PUCT}$ |
| :--- | :--- | :--- | :--- |
| A | $0.72$ | $0.1304$ | $0.8504$ |
| B | $0.58$ | $0.1452$ | $0.7252$ |
| C | $0.81$ | $0.1875$ | **$0.9975$** |
| D | $0.35$ | $0.1364$ | $0.4864$ |

**Select max:** Node C wins (0.9975). $\blacksquare$

---

### Illustration 3: MCTS Backup Phase

**Problem:**
A simulation from Node C reaches terminal state with reward $R = 1.0$ (correct answer).
Backup path: C $\rightarrow$ parent P1 $\rightarrow$ root R.
Before update:
- $N_C=15, W_C=8.1$ (so $Q_C=8.1/15=0.54$)
- $N_{\text{P1}}=60, W_{\text{P1}}=33$ (so $Q_{\text{P1}}=33/60=0.55$)
- $N_{\text{root}}=100, W_{\text{root}}=55$ (so $Q_{\text{root}}=55/100=0.55$)
Compute the updated statistics for each node.

**Step-by-Step Solution:**
1. **Node C Update:**
   - $N_C \leftarrow 15 + 1 = \mathbf{16}$
   - $W_C \leftarrow 8.1 + 1.0 = \mathbf{9.1}$
   - $Q_C = 9.1 / 16 = \mathbf{0.56875}$

2. **Parent P1 Update:**
   - $N_{\text{P1}} \leftarrow 60 + 1 = \mathbf{61}$
   - $W_{\text{P1}} \leftarrow 33 + 1.0 = \mathbf{34}$
   - $Q_{\text{P1}} = 34 / 61 \approx \mathbf{0.55738}$

3. **Root Update:**
   - $N_{\text{root}} \leftarrow 100 + 1 = \mathbf{101}$
   - $W_{\text{root}} \leftarrow 55 + 1.0 = \mathbf{56}$
   - $Q_{\text{root}} = 56 / 101 \approx \mathbf{0.55446}$

**Conclusion:** The reward of $1.0$ is backpropagated up the tree, raising the average action-value ($Q$) for every node along the winning path. $\blacksquare$

---

### Illustration 4: Tree of Thoughts Branching Analysis

**Problem:**
A Tree of Thoughts has depth $d=3$ and branching factor $b=3$.
Compute the total number of nodes in the tree.
If each node costs $C=1$ LLM call, what is the ToT budget vs. a Best-of-1 budget?
Assuming each node has $P(\text{correct} \mid \text{partial}) = 0.4$, compute the pass@1 improvement of searching the tree versus standard sequential generation.

**Step-by-Step Solution:**
1. **Total Nodes:**
   Total nodes = $1 + b + b^2 + b^3 = 1 + 3 + 9 + 27 = \mathbf{40}$.
   ToT budget = **40C**, Best-of-1 budget = **1C**.

2. **Tree of Thoughts Pass@1 (Probability of at least 1 correct leaf):**
   - The tree explores $b^3 = 27$ independent leaf paths.
   - $P(\text{at least 1 correct leaf}) = 1 - (1 - 0.4)^{27} = 1 - 0.6^{27}$.
   - $0.6^{27} \approx 1.11 \times 10^{-6}$.
   - $P \approx 1 - 1.11 \times 10^{-6} \approx \mathbf{1.0}$ (near certainty).

3. **Sequential Best-of-1:**
   - Without tree search, the model commits to a single path.
   - $P = \mathbf{0.4}$.

**Conclusion:** At a compute multiplier of $40\times$, Tree of Thoughts transforms a $40\%$ unreliable process into a near-$100\%$ reliable reasoning engine by thoroughly exploring the solution space geometry. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **DeepSeek-Prover & AlphaProof:** Combines Monte Carlo Tree Search with formal Lean 4 interactive theorem provers, using policy LLMs to suggest formal tactics and state verification to update search trees.
- **Tree-Search Augmented Decoding:** Frontier models (e.g. OpenAI o3) utilize internal tree search over reasoning tokens to navigate mathematical branch cuts and backtrack out of dead ends.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - PUCT calculations matching $1.300000$ and $1.500000$.
   - Selection decision and backpropagation updates matching $Q_{\text{new}} = 0.950000$.
2. **Complete MCTS Reasoning Engine for LLMs:**
   - Object-oriented `MCTSNode` maintaining visit counts, $Q$-values, and priors.
   - Simulation of PUCT tree expansion, PRM leaf evaluation, and backpropagation over multiple iterations.

See implementation in:
[`13_reasoning_and_test_time_compute/code/05_mcts_tree_search.py`](./code/05_mcts_tree_search.py)
