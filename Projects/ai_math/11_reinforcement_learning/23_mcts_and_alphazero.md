# Chapter 23: Monte Carlo Tree Search (MCTS) & AlphaZero

---

## 1. Intuition & 101 Motivation

In the previous chapters on Model-Based RL, we learned how transition models enable planning through mental rollouts (Dyna-Q) or trajectory optimization over continuous actions (MPC). However, when dealing with discrete, combinatorial decision trees of astronomical scale—such as Chess ($10^{120}$ game states) or Go ($10^{170}$ game states)—classical search methods collapse:
- **Exhaustive Minimax:** Intractable due to exponential branching ($b^d$).
- **Alpha-Beta Pruning:** Relies on hand-crafted evaluation heuristics that fail in subtle, complex board configurations.
- **Pure Deep Reinforcement Learning (Model-Free):** Without search, a policy network acts on immediate intuition and frequently misses deep tactical traps.

Enter **Monte Carlo Tree Search (MCTS)** and **AlphaZero**:
- Rather than searching uniformly across all branches, MCTS builds an **asymmetric search tree**, focusing its computational budget on lines of play that are most promising while maintaining a mathematically grounded exploration bonus.
- In 2017, DeepMind's **AlphaZero** revolutionized AI by replacing hand-crafted heuristics and random rollout simulations with a single **dual-headed neural network** $(p, v) = f_\theta(s)$:
  1. **Policy Head $p_\theta(a \mid s)$:** Prior probabilities guiding *where to look* (slashing the effective branching factor).
  2. **Value Head $v_\theta(s)$:** Position evaluation predicting *who will win* (eliminating noisy random rollouts).

AlphaZero achieved superhuman mastery in Chess, Shogi, and Go starting from pure self-play with zero human knowledge beyond the basic rules!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The PUCT Selection Formula (Predictor + UCT)

At any internal node $s$ of the search tree, each child action $a \in \mathcal{A}(s)$ maintains four statistics:
1. $N(s, a)$: Visit count (number of times action $a$ was traversed).
2. $W(s, a)$: Total action-value accumulated across all visits.
3. $Q(s, a) = \frac{W(s, a)}{N(s, a)}$: Mean action-value (empirical win rate / expected return).
4. $P(s, a)$: Prior probability of taking action $a$ from policy network $p_\theta(s, a)$.

During the tree traversal phase, the agent selects the action $a^*$ maximizing the **PUCT** (Polynomial Upper Confidence trees) objective:

$$a^* = \arg\max_{a \in \mathcal{A}(s)} \left[ Q(s, a) + U(s, a) \right]$$

where the exploration bonus $U(s, a)$ is defined as:

$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_{b \in \mathcal{A}(s)} N(s, b)}}{1 + N(s, a)}$$

#### Mathematical Mechanics of PUCT:
1. **Initial Bias by Prior:** When an action has not been visited ($N(s, a) = 0$), $U(s, a) \propto P(s, a) \sqrt{\sum N}$. Actions favored by the neural network prior $P(s, a)$ are explored first.
2. **Diminishing Bonus:** As an action is visited repeatedly, the denominator $(1 + N(s, a))$ suppresses $U(s, a)$.
3. **Asymptotic Convergence to Value:** As total visits $\sum N \to \infty$, $U(s, a) \to 0$, and action selection becomes strictly governed by the empirical value $Q(s, a)$.

---

### 2.2 The Four Phases of AlphaZero MCTS

Each MCTS step consists of executing $M$ iterations (e.g., $M = 800$ or $1600$) of a four-phase cycle:

```
          [ 1. Selection ]
                 |
                 v (Traverse via PUCT until leaf)
          [ 2. Expansion ]
                 |
                 v (Create child nodes for legal actions)
          [ 3. Evaluation ]
                 |
                 v (Query Neural Net: (p, v) = f_theta(s_leaf))
          [ 4. Backup ]
                 |
                 v (Backpropagate v up visited path: N += 1, W += v)
```

#### Phase 1: Selection
Starting at the root node $s_0$, recursively select actions according to $a = \arg\max_a [Q(s, a) + U(s, a)]$ until reaching a leaf node $s_L$ (a node that has not yet been expanded).

#### Phase 2: Expansion
Create child edges for all legal actions $a \in \mathcal{A}(s_L)$.

#### Phase 3: Evaluation
Instead of playing a random rollout to the end of the game (as in classical MCTS), pass state $s_L$ through the dual-headed neural network:
$$(\mathbf{p}, v) = f_\theta(s_L)$$
Store prior probabilities $P(s_L, a) = p_a$ on the outgoing child edges, and initialize visit statistics:
$$N(s_L, a) = 0, \quad W(s_L, a) = 0, \quad Q(s_L, a) = 0$$

#### Phase 4: Backup (Backpropagation)
Traverse backward along the path of visited state-action pairs $(s, a)$ from $s_L$ back to the root $s_0$.
For two-player zero-sum games, the perspective of the player flips at each ply:
$$v_{\text{parent}} = -v_{\text{child}}$$
Update edge statistics:
$$N(s, a) \leftarrow N(s, a) + 1$$
$$W(s, a) \leftarrow W(s, a) + v$$
$$Q(s, a) \leftarrow \frac{W(s, a)}{N(s, a)}$$

---

### 2.3 Policy Improvement & Action Execution

After $M$ simulations from root state $s_0$, the search policy $\boldsymbol{\pi}$ is extracted directly from the **visit counts**, not the $Q$-values:

$$\pi_{\text{MCTS}}(a \mid s_0) = \frac{N(s_0, a)^{1/\tau}}{\sum_{b \in \mathcal{A}(s_0)} N(s_0, b)^{1/\tau}}$$

where $\tau$ is a temperature parameter:
- **During Training ($\tau = 1$):** $\pi(a \mid s_0) \propto N(s_0, a)$, ensuring exploratory diversity.
- **During Competitive Play ($\tau \to 0$):** Deterministically select the most visited move:
  $$a^* = \arg\max_{a} N(s_0, a)$$

> **Key Theoretical Result:** The visit count distribution $\boldsymbol{\pi}_{\text{MCTS}}$ is a **strictly improved policy** compared to the raw neural network prior $\mathbf{p}_\theta(s_0)$:
> $$\pi_{\text{MCTS}} \succ \mathbf{p}_\theta(s_0)$$
> MCTS acts as a powerful **policy improvement operator**!

---

### 2.4 Self-Play Training Loss

During self-play, games are played to completion. At game's end, the terminal outcome from the perspective of player at step $t$ is recorded as $z_t \in \{-1, 0, +1\}$ (loss, draw, win).

The neural network parameters $\theta$ are optimized end-to-end to minimize the joint loss:

$$\mathcal{L}(\theta) = \underbrace{(z - v_\theta(s))^2}_{\text{Value MSE Loss}} - \underbrace{\boldsymbol{\pi}_{\text{MCTS}}^T \log \mathbf{p}_\theta(s)}_{\text{Policy Cross-Entropy Loss}} + c \|\theta\|^2$$

---

## 3. Geometric & Physical Interpretation

### 3.1 Asymmetric vs. Symmetric Search Trees
In classical game theory, minimax explores a uniform tree of depth $d$, requiring $O(b^d)$ evaluations.

```
       Minimax (Symmetric, Shallow)              MCTS / AlphaZero (Asymmetric, Deep)
                   ( )                                           ( )
              /     |     \                                    /  |  \
            ( )    ( )    ( )                                ( ) ( ) ( )
           / | \  / | \  / | \                                |   |    \
          ....................                              ( )  ( )   ( )
                                                             |          |
                                                            ( )        ( )
                                                             |
                                                            ( )  <-- Deep tactical line!
```
- AlphaZero selectively drives 95% of its search depth down the critical, game-deciding tactical variations while visiting unpromising moves only once or twice to ensure they contain no immediate refutations.

---

## 4. Real-World Analogy: Chess Grandmaster's Intuition

Consider a Chess Grandmaster analyzing a complex board:
1. **The Policy Prior ($P$):** Out of 35 legal moves, intuition immediately highlights 2 or 3 candidate moves ($P = 0.70, 0.25, 0.05$). The rest are filtered out.
2. **Deep Calculation ($N$):** The grandmaster calculates 15 plies deep along the top candidate line ($N = 1200$ mental visits).
3. **Position Evaluation ($V$):** At the end of the calculation, they assess the resulting endgame: *"White has an outside passed pawn, value $v = +0.85$."*
4. **Backing Up ($Q$):** That evaluation flows back to the initial move.
5. **Execution:** The grandmaster plays the move that survived the deepest scrutiny with the highest confidence!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace **4 complete MCTS simulations by hand** on a 2-action root node ($a_1, a_2$).

---

### 5.1 System & Prior Parameters
- **Root State:** $s_0$
- **Actions:** $\mathcal{A} = \{a_1, a_2\}$
- **Policy Network Prior:**
  $$P(s_0, a_1) = 0.70, \quad P(s_0, a_2) = 0.30$$
- **Exploration Coefficient:** $c_{\text{puct}} = 1.0$
- **Initial Tree Statistics:**
  $$N(s_0, a_1) = 0, \quad W(s_0, a_1) = 0.0, \quad Q(s_0, a_1) = 0.0$$
  $$N(s_0, a_2) = 0, \quad W(s_0, a_2) = 0.0, \quad Q(s_0, a_2) = 0.0$$
  Total root visits: $N_{\text{total}} = \sum_b N(s_0, b) = 0$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $N(s_0, a)$ | `visit_count[a]` | Number of times action $a$ was selected from root |
| $W(s_0, a)$ | `value_sum[a]` | Accumulated neural network value backpropagated to action $a$ |
| $Q(s_0, a)$ | `mean_q[a]` | Empirical mean value: $Q = W / N$ (or $0.0$ if $N=0$) |
| $P(s_0, a)$ | `prior_p[a]` | Neural network prior probability from policy head |
| $U(s_0, a)$ | `u_bonus[a]` | PUCT exploration bonus: $c_{\text{puct}} \cdot P \cdot \frac{\sqrt{\sum N}}{1 + N}$ |
| $\text{Score}(a)$ | `puct_score[a]` | Selection metric: $Q(s_0, a) + U(s_0, a)$ |
| $v$ | `v_eval` | Value evaluated by neural network at leaf node |

---

### 5.3 Simulation-by-Simulation Hand Calculations

#### Simulation 1:
- **Selection:**
  At $N_{\text{total}} = 0$, both visit counts are $0$. With zero visits, $U(a) = 0$, so both actions tie at score $0.0$.
  Tie-breaking follows higher prior: $P(a_1) = 0.70 > 0.30 \implies$ **Select $a_1$**.
- **Expansion & Evaluation:**
  Leaf node $s_1$ is evaluated by the neural network:
  $$v_1 = \mathbf{+0.6000}$$
- **Backup:**
  $$N(s_0, a_1) \leftarrow 0 + 1 = \mathbf{1}$$
  $$W(s_0, a_1) \leftarrow 0.0 + 0.6000 = \mathbf{0.6000}$$
  $$Q(s_0, a_1) \leftarrow \frac{0.6000}{1} = \mathbf{0.6000}$$
  Total root visits: $N_{\text{total}} = 1$.

---

#### Simulation 2:
- **Selection:**
  Total root visits $N_{\text{total}} = 1 \implies \sqrt{N_{\text{total}}} = 1.0000$.
  - **Action $a_1$:**
    $$Q(s_0, a_1) = 0.6000$$
    $$U(s_0, a_1) = 1.0 \times 0.70 \times \frac{\sqrt{1}}{1 + 1} = 0.70 \times 0.5000 = \mathbf{0.3500}$$
    $$\text{Score}(a_1) = Q(a_1) + U(a_1) = 0.6000 + 0.3500 = \mathbf{0.9500}$$
  - **Action $a_2$:**
    $$Q(s_0, a_2) = 0.0000$$
    $$U(s_0, a_2) = 1.0 \times 0.30 \times \frac{\sqrt{1}}{1 + 0} = 0.30 \times 1.0000 = \mathbf{0.3000}$$
    $$\text{Score}(a_2) = Q(a_2) + U(a_2) = 0.0000 + 0.3000 = \mathbf{0.3000}$$
  - **Decision:** $\max(0.9500, 0.3000) \implies$ **Select $a_1$**.
- **Expansion & Evaluation:**
  Child of $a_1$ evaluated by neural network:
  $$v_2 = \mathbf{+0.4000}$$
- **Backup:**
  $$N(s_0, a_1) \leftarrow 1 + 1 = \mathbf{2}$$
  $$W(s_0, a_1) \leftarrow 0.6000 + 0.4000 = \mathbf{1.0000}$$
  $$Q(s_0, a_1) \leftarrow \frac{1.0000}{2} = \mathbf{0.5000}$$
  Total root visits: $N_{\text{total}} = 2$.

---

#### Simulation 3:
- **Selection:**
  Total root visits $N_{\text{total}} = 2 \implies \sqrt{N_{\text{total}}} = \sqrt{2} \approx 1.414214$.
  - **Action $a_1$:**
    $$Q(s_0, a_1) = 0.5000$$
    $$U(s_0, a_1) = 1.0 \times 0.70 \times \frac{1.414214}{1 + 2} = 0.70 \times 0.471405 \approx \mathbf{0.329983}$$
    $$\text{Score}(a_1) = 0.5000 + 0.329983 = \mathbf{0.829983}$$
  - **Action $a_2$:**
    $$Q(s_0, a_2) = 0.0000$$
    $$U(s_0, a_2) = 1.0 \times 0.30 \times \frac{1.414214}{1 + 0} = 0.30 \times 1.414214 \approx \mathbf{0.424264}$$
    $$\text{Score}(a_2) = 0.0000 + 0.424264 = \mathbf{0.424264}$$
  - **Decision:** $\max(0.829983, 0.424264) \implies$ **Select $a_1$**.
- **Expansion & Evaluation:**
  Child evaluated with poor outcome:
  $$v_3 = \mathbf{-0.2000}$$
- **Backup:**
  $$N(s_0, a_1) \leftarrow 2 + 1 = \mathbf{3}$$
  $$W(s_0, a_1) \leftarrow 1.0000 + (-0.2000) = \mathbf{0.8000}$$
  $$Q(s_0, a_1) \leftarrow \frac{0.8000}{3} \approx \mathbf{0.266667}$$
  Total root visits: $N_{\text{total}} = 3$.

---

#### Simulation 4:
- **Selection:**
  Total root visits $N_{\text{total}} = 3 \implies \sqrt{N_{\text{total}}} = \sqrt{3} \approx 1.732051$.
  - **Action $a_1$:**
    $$Q(s_0, a_1) = 0.266667$$
    $$U(s_0, a_1) = 1.0 \times 0.70 \times \frac{1.732051}{1 + 3} = 0.70 \times 0.433013 \approx \mathbf{0.303109}$$
    $$\text{Score}(a_1) = 0.266667 + 0.303109 = \mathbf{0.569776}$$
  - **Action $a_2$:**
    $$Q(s_0, a_2) = 0.0000$$
    $$U(s_0, a_2) = 1.0 \times 0.30 \times \frac{1.732051}{1 + 0} = 0.30 \times 1.732051 \approx \mathbf{0.519615}$$
    $$\text{Score}(a_2) = 0.0000 + 0.519615 = \mathbf{0.519615}$$
  - **Decision:** $\max(0.569776, 0.519615) \implies$ $a_1$ still edges out $a_2$ ($0.5698 > 0.5196$), but $a_2$'s exploration pressure is rapidly surging!

---

### 5.4 Summary Visual Grid: AlphaZero PUCT MCTS Ledger

| Sim # | Candidate Action | Prior $P$ | Visits $N$ | Mean $Q$ | $\sum N$ | Exploration $U$ | PUCT Score | Selected Action | Value $v$ | New $N$ | New $Q$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $a_1$ / $a_2$ | $0.7 / 0.3$ | $0 / 0$ | $0.0 / 0.0$ | $0$ | $0.0 / 0.0$ | $0.000 / 0.000$ | **$a_1$** (prior) | $+0.60$ | $N_1 = 1$ | $Q_1 = +0.6000$ |
| **2** | $a_1$ | $0.70$ | $1$ | $+0.6000$ | $1$ | $0.3500$ | **$0.9500$** | **$a_1$** | $+0.40$ | $N_1 = 2$ | $Q_1 = +0.5000$ |
|  | $a_2$ | $0.30$ | $0$ | $0.0000$ | $1$ | $0.3000$ | $0.3000$ | | | $N_2 = 0$ | $Q_2 = 0.0000$ |
| **3** | $a_1$ | $0.70$ | $2$ | $+0.5000$ | $2$ | $0.3300$ | **$0.8300$** | **$a_1$** | $-0.20$ | $N_1 = 3$ | $Q_1 = +0.2667$ |
|  | $a_2$ | $0.30$ | $0$ | $0.0000$ | $2$ | $0.4243$ | $0.4243$ | | | $N_2 = 0$ | $Q_2 = 0.0000$ |
| **4** | $a_1$ | $0.70$ | $3$ | $+0.2667$ | $3$ | $0.3031$ | **$0.5698$** | **$a_1$** | — | — | — |
|  | $a_2$ | $0.30$ | $0$ | $0.0000$ | $3$ | $0.5196$ | $0.5196$ | | | — | — |

---

## 6. Solved Illustrations

### Illustration 1: Zero-Sum Value Inversion
**Problem:**
In a 2-player zero-sum game (White vs. Black), why does the backpropagated value invert sign at each ply ($v \leftarrow -v$)?
**Solution:**
A state that is wonderful for White ($v = +1.0$) is catastrophic for Black ($v = -1.0$). If Black evaluates a position resulting in $+0.80$ from Black's perspective, this means Black is winning. When backed up to White's decision node, White must view this move as $-0.80$ to correctly avoid playing into Black's winning trap! $\blacksquare$

### Illustration 2: Dirichlet Noise at the Root Node
**Problem:**
Why does AlphaZero add Dirichlet noise $\boldsymbol{\eta} \sim \operatorname{Dir}(\alpha)$ to the root node prior during self-play:
$$P(s_0, a) \leftarrow (1 - \epsilon) P(s_0, a) + \epsilon \eta_a \quad (\epsilon = 0.25)$$
**Solution:**
If the policy network strongly assigns $P(s_0, a^*) \approx 0.999$, the PUCT formula will persistently select $a^*$ and never explore alternative openings. Injecting Dirichlet noise guarantees that all legal opening moves have non-zero probability of being explored, preventing self-play from collapsing into a narrow, repetitive opening book. $\blacksquare$

---

## 7. Deep RL Connection & Modern Applications

- **AlphaZero & MuZero:** MuZero (Schrittwieser et al., Nature 2020) extended AlphaZero to learn an internal latent dynamics model, beating human world records on Atari 2600 games without access to the game engine or rules!
- **Reasoning in Frontier LLMs (OpenAI o1, DeepSeek-R1, Q*):**
  Modern reasoning models conceptualize multi-step mathematical proofs and coding solutions as tree search over thoughts. MCTS is combined with Process Reward Models (PRMs) or Outcome Reward Models (ORMs) to guide the search over chain-of-thought tokens!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical calculation of all PUCT scores, $Q$-values, and visit counts for Simulations 1–4 matching to $< 10^{-6}$.
2. **Complete Generic MCTS Node & Search Engine:**
   - Full implementation of selection, expansion, neural-net evaluation, and backup.
3. **Tic-Tac-Toe Self-Play Engine:**
   - Self-play match verification where AlphaZero MCTS plays against a random agent, demonstrating $100\%$ win/draw rate.

See implementation in:
[`11_reinforcement_learning/code/23_mcts_and_alphazero.py`](./code/23_mcts_and_alphazero.py)
