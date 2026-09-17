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

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.23.1: Upper Confidence Bound for Trees (UCT) Logarithmic Regret Bound

```
====================================================================================================
DERIVATION 11.23.1: Finite-Time Logarithmic Regret Bound of UCT (Kocsis & Szepesvári Theorem)
====================================================================================================
Problem Statement:
Prove that for any tree search decision node with K candidate actions 𝒜(s) = {a_1, ..., a_K} and true
suboptimality gaps Δ_a = μ^* - μ_a > 0, the UCB1/UCT tree selection rule achieves a finite-time
expected visit count bound:
    𝔼[T_a(N)] ≤ (8 ln N) / (Δ_a^2) + 1 + π^2 / 3
and an optimal logarithmic cumulative pseudo-regret bound:
    R(N) = ∑_{a: Δ_a > 0} Δ_a 𝔼[T_a(N)] ≤ ∑_{a: Δ_a > 0} [ (8 ln N) / Δ_a + (1 + π^2/3) Δ_a ] = 𝒪((K ln N) / Δ)
Furthermore, prove that when extended recursively across a finite-depth tree of depth D, the probability
of selecting a suboptimal action at the root decays to zero at a polynomial rate 𝒪(N^{-c}), ensuring
almost-sure convergence to the true minimax game-theoretic value.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
Let $s$ be a decision node in an asymmetric search tree with legal action set $\mathcal{A}(s) = \{a_1, \dots, a_K\}$. When an action $a$ is selected at round $t$, the simulation returns a bounded leaf evaluation $X_{a, i} \in [0, 1]$ drawn from an underlying distribution with unknown true mean $\mu_a \triangleq \mathbb{E}[X_{a, i}]$.
Define the optimal action $a^*$ and optimal expected payoff $\mu^*$ by:
$$a^* \triangleq \arg\max_{a \in \mathcal{A}(s)} \mu_a, \quad \mu^* \triangleq \mu_{a^*} = \max_{a \in \mathcal{A}(s)} \mu_a$$
For any suboptimal action $a \ne a^*$, define the suboptimality gap:
$$\Delta_a \triangleq \mu^* - \mu_a > 0$$
Let $T_a(t) \triangleq \sum_{i=1}^t \mathbb{I}(I_i = a)$ denote the number of times action $a$ has been selected up to round $t$, where $I_i \in \mathcal{A}(s)$ denotes the action selected at round $i$. Let $\bar{X}_{a, s} \triangleq \frac{1}{s} \sum_{i=1}^s X_{a, i}$ be the empirical mean after $s$ selections of arm $a$.
At round $t$, the UCB1 selection policy chooses:
$$I_t = \arg\max_{a \in \mathcal{A}(s)} \left[ \bar{X}_{a, T_a(t-1)} + c_{t-1, T_a(t-1)} \right], \quad \text{where} \quad c_{t, s} \triangleq \sqrt{\frac{2 \ln t}{s}}$$
The cumulative pseudo-regret after $N$ total simulations through state $s$ is defined as:
$$R(N) \triangleq N \mu^* - \mathbb{E}\left[ \sum_{t=1}^N X_{I_t, T_{I_t}(t)} \right] = \sum_{a: \Delta_a > 0} \Delta_a \mathbb{E}[T_a(N)]$$
Our mathematical goal is to establish:
1. The finite-time bound on the expected number of suboptimal visits: $\mathbb{E}[T_a(N)] \le \frac{8 \ln N}{\Delta_a^2} + 1 + \frac{\pi^2}{3}$.
2. The logarithmic finite-time regret bound: $R(N) \le \sum_{a: \Delta_a > 0} \left( \frac{8 \ln N}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \Delta_a \right) = \mathcal{O}\left( \frac{K \ln N}{\Delta_{\min}} \right)$.
3. The Kocsis & Szepesvári tree search induction guaranteeing asymptotic convergence to the true minimax value $Q^*(s, a)$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Bounded Support:** All payoffs/leaf evaluations are bounded: $X_{a, i} \in [0, 1]$ almost surely.
2. **Sub-Gaussian Tail:** For any arm $a$ and fixed sample count $s \ge 1$, the deviation $(\bar{X}_{a, s} - \mu_a)$ satisfies Hoeffding's inequality with sub-Gaussian parameter $\sigma^2 = \frac{1}{4}$:
   $$\mathbb{P}\left( |\bar{X}_{a, s} - \mu_a| \ge \epsilon \right) \le 2 \exp(-2 s \epsilon^2) \quad \forall \epsilon > 0$$
3. **Exploration Constant Calibration:** The exploration constant in $c_{t, s} = \sqrt{\frac{2 \ln t}{s}}$ is set to $\sqrt{2}$, which guarantees an exponential tail decay rate of $t^{-4}$, ensuring the summability of error probabilities over all rounds $t \ge 1$.
4. **Initial Coverage:** Each action $a \in \mathcal{A}(s)$ is pulled once during initialization ($t = 1, \dots, K$), so $T_a(K) = 1$ for all $a$.
5. **Finite Tree Topology:** Tree depth is bounded by $D < \infty$, and branching factor $|\mathcal{A}(s)| = K < \infty$ for all nodes $s$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
The UCB1 algorithm implements the fundamental principle of **Optimism in the Face of Uncertainty (OFU)**. Each action maintains an empirical confidence interval $[\bar{X}_{a, s} - c_{t, s}, \, \bar{X}_{a, s} + c_{t, s}]$. The upper boundary $\text{UCB}_a = \bar{X} + c$ represents the *highest plausible payoff* compatible with the observed data.
Geometrically, selecting an arm corresponds to testing whether its plausible ceiling can beat the optimal arm. A suboptimal action $a$ can only be chosen if its upper confidence bound exceeds that of the optimal action $a^*$:
$$\bar{X}_{a, T_a(t-1)} + c_{t-1, T_a(t-1)} \ge \bar{X}_{a^*, T_{a^*}(t-1)} + c_{t-1, T_{a^*}(t-1)}$$
This event requires at least one of three anomalous conditions:
1. The optimal arm was severely underestimated ($\bar{X}_{a^*}$ fell below $\mu^* - c$).
2. The suboptimal arm was severely overestimated ($\bar{X}_a$ surged above $\mu_a + c$).
3. The suboptimal arm was visited too few times, so its confidence radius $c$ is still wider than the gap $\Delta_a$.
Hoeffding's inequality guarantees that conditions 1 and 2 occur with negligible, polynomially decaying probability ($\mathcal{O}(t^{-4})$). Meanwhile, condition 3 is permanently extinguished as soon as the visit count $T_a$ reaches the threshold $\frac{8 \ln N}{\Delta_a^2}$. Consequently, the algorithm automatically allocates $\mathcal{O}(\log N)$ exploratory trials to suboptimal actions and concentrates the remaining $N - \mathcal{O}(\log N)$ trials purely on the optimal action.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Indicator Decomposition of Visit Counts.*
For any arbitrary integer threshold $\ell \ge 1$, we decompose the total visit count $T_a(N)$ to suboptimal action $a$:
$$T_a(N) = \sum_{t=1}^N \mathbb{I}(I_t = a) = 1 + \sum_{t=K+1}^N \mathbb{I}(I_t = a)$$
Splitting the sum conditioned on whether $T_a(t-1)$ is strictly less than $\ell$ or at least $\ell$:
$$T_a(N) \le \ell + \sum_{t=K+1}^N \mathbb{I}(I_t = a, \, T_a(t-1) \ge \ell)$$
Taking mathematical expectations on both sides:
$$\mathbb{E}[T_a(N)] \le \ell + \sum_{t=K+1}^N \mathbb{P}(I_t = a, \, T_a(t-1) \ge \ell) \quad \text{(Eq. 1.1)}$$

*Step 2: Necessary Condition for Suboptimal Selection.*
Action $a$ is selected at step $t$ if and only if its UCB index equals or exceeds that of all other actions, including the optimal action $a^*$:
$$\{I_t = a\} \implies \bar{X}_{a, T_a(t-1)} + c_{t-1, T_a(t-1)} \ge \bar{X}_{a^*, T_{a^*}(t-1)} + c_{t-1, T_{a^*}(t-1)}$$
Since the true sample counts $T_a(t-1)$ and $T_{a^*}(t-1)$ are random variables taking integer values in $\{1, \dots, t-1\}$, we bound the event by taking the union over all possible realization pairs $(s, s^*)$:
$$\{I_t = a, \, T_a(t-1) \ge \ell\} \subseteq \bigcup_{s=\ell}^{t-1} \bigcup_{s^*=1}^{t-1} \left\{ \bar{X}_{a, s} + c_{t-1, s} \ge \bar{X}_{a^*, s^*} + c_{t-1, s^*} \right\} \quad \text{(Eq. 1.2)}$$

*Step 3: Three-Way Event Decomposition.*
We claim that the inequality $\bar{X}_{a, s} + c_{t-1, s} \ge \bar{X}_{a^*, s^*} + c_{t-1, s^*}$ implies that at least one of the following three elementary events must hold:
$$E_1(s^*): \quad \bar{X}_{a^*, s^*} \le \mu^* - c_{t-1, s^*}$$
$$E_2(s): \quad \bar{X}_{a, s} \ge \mu_a + c_{t-1, s}$$
$$E_3(s): \quad \mu^* < \mu_a + 2 c_{t-1, s}$$
*Proof by Contradiction:*
Suppose for contradiction that all three events are false. That is, assume:
$$\bar{X}_{a^*, s^*} > \mu^* - c_{t-1, s^*} \quad \text{(False } E_1\text{)}$$
$$\bar{X}_{a, s} < \mu_a + c_{t-1, s} \quad \text{(False } E_2\text{)}$$
$$\mu^* \ge \mu_a + 2 c_{t-1, s} \quad \text{(False } E_3\text{)}$$
Adding $c_{t-1, s^*}$ to both sides of the negated $E_1$:
$$\bar{X}_{a^*, s^*} + c_{t-1, s^*} > \mu^*$$
Substituting the negated $E_3$ ($\mu^* \ge \mu_a + 2 c_{t-1, s}$):
$$\bar{X}_{a^*, s^*} + c_{t-1, s^*} > \mu_a + 2 c_{t-1, s} = (\mu_a + c_{t-1, s}) + c_{t-1, s}$$
Substituting the negated $E_2$ ($\mu_a + c_{t-1, s} > \bar{X}_{a, s}$):
$$\bar{X}_{a^*, s^*} + c_{t-1, s^*} > \bar{X}_{a, s} + c_{t-1, s}$$
This directly contradicts the selection condition $\bar{X}_{a, s} + c_{t-1, s} \ge \bar{X}_{a^*, s^*} + c_{t-1, s^*}$.
Hence, by contraposition, at least one of $E_1(s^*), E_2(s), E_3(s)$ must be true.

*Step 4: Eliminating Event $E_3$ via Choice of Threshold $\ell$.*
Examine event $E_3(s)$:
$$\mu^* < \mu_a + 2 c_{t-1, s} \iff \mu^* - \mu_a < 2 \sqrt{\frac{2 \ln(t-1)}{s}} \iff \Delta_a < \sqrt{\frac{8 \ln(t-1)}{s}}$$
Squaring both sides (both are strictly positive):
$$\Delta_a^2 < \frac{8 \ln(t-1)}{s} \iff s < \frac{8 \ln(t-1)}{\Delta_a^2}$$
Since $t \le N$, we have $\ln(t-1) \le \ln N$. Therefore, if $s \ge \frac{8 \ln N}{\Delta_a^2}$, then:
$$s \ge \frac{8 \ln(t-1)}{\Delta_a^2} \implies \Delta_a \ge \sqrt{\frac{8 \ln(t-1)}{s}} = 2 c_{t-1, s} \implies \mu^* - \mu_a - 2 c_{t-1, s} \ge 0$$
which renders event $E_3(s)$ mathematically impossible.
We therefore define the threshold:
$$\ell \triangleq \left\lceil \frac{8 \ln N}{\Delta_a^2} \right\rceil$$
For all $s \ge \ell$, $\mathbb{P}(E_3(s)) = 0$.

*Step 5: Concentration Bounds via Hoeffding's Inequality.*
For events $E_1(s^*)$ and $E_2(s)$, the random variables $X_{a, i} \in [0, 1]$ are independent and bounded.
By Hoeffding's inequality for the empirical mean of $s^*$ independent bounded random variables:
$$\mathbb{P}\left( \bar{X}_{a^*, s^*} - \mu^* \le -\epsilon \right) \le \exp\left( -2 s^* \epsilon^2 \right)$$
Setting $\epsilon = c_{t-1, s^*} = \sqrt{\frac{2 \ln(t-1)}{s^*}}$:
$$\mathbb{P}(E_1(s^*)) = \mathbb{P}\left( \bar{X}_{a^*, s^*} \le \mu^* - \sqrt{\frac{2 \ln(t-1)}{s^*}} \right) \le \exp\left( -2 s^* \cdot \frac{2 \ln(t-1)}{s^*} \right)$$
$$-2 s^* \cdot \frac{2 \ln(t-1)}{s^*} = -4 \ln(t-1) = \ln\left( (t-1)^{-4} \right)$$
$$\implies \mathbb{P}(E_1(s^*)) \le \exp\left( \ln(t-1)^{-4} \right) = (t-1)^{-4}$$
Applying the identical step to event $E_2(s)$ with sample count $s$:
$$\mathbb{P}(E_2(s)) = \mathbb{P}\left( \bar{X}_{a, s} - \mu_a \ge \sqrt{\frac{2 \ln(t-1)}{s}} \right) \le \exp\left( -2 s \cdot \frac{2 \ln(t-1)}{s} \right) = (t-1)^{-4}$$

*Step 6: Summing Probability Over Rounds.*
Now substitute these probabilities back into the expectation bound from Eq. 1.1:
$$\mathbb{P}(I_t = a, \, T_a(t-1) \ge \ell) \le \mathbb{P}\left( \bigcup_{s^*=1}^{t-1} E_1(s^*) \cup \bigcup_{s=\ell}^{t-1} E_2(s) \right)$$
Using the union bound over the actual counters realized at step $t-1$:
$$\mathbb{P}(I_t = a, \, T_a(t-1) \ge \ell) \le \sum_{s^*=1}^{t-1} \mathbb{P}(E_1(s^*)) + \sum_{s=\ell}^{t-1} \mathbb{P}(E_2(s))$$
$$\le \sum_{s^*=1}^{t-1} (t-1)^{-4} + \sum_{s=\ell}^{t-1} (t-1)^{-4} \le (t-1) \cdot (t-1)^{-4} + (t-1) \cdot (t-1)^{-4} = 2 (t-1)^{-3}$$
Summing over all time steps $t$ from $K+1$ up to $N$:
$$\sum_{t=K+1}^N \mathbb{P}(I_t = a, \, T_a(t-1) \ge \ell) \le \sum_{t=K+1}^N 2 (t-1)^{-3} \le \sum_{k=1}^{N-1} 2 k^{-3} = 2 \sum_{k=1}^{N-1} \frac{1}{k^3}$$
Recall that for any $k \ge 1$, $\frac{1}{k^3} \le \frac{1}{k^2}$. Using the Basel problem identity $\sum_{k=1}^\infty \frac{1}{k^2} = \frac{\pi^2}{6}$:
$$2 \sum_{k=1}^\infty \frac{1}{k^3} < 2 \sum_{k=1}^\infty \frac{1}{k^2} = 2 \cdot \frac{\pi^2}{6} = \frac{\pi^2}{3} \approx 3.289868$$
Therefore:
$$\sum_{t=K+1}^N \mathbb{P}(I_t = a, \, T_a(t-1) \ge \ell) \le \frac{\pi^2}{3}$$

*Step 7: Final Visit Count and Pseudo-Regret Bounds.*
Substituting back into Eq. 1.1:
$$\mathbb{E}[T_a(N)] \le \ell + \frac{\pi^2}{3} = \left\lceil \frac{8 \ln N}{\Delta_a^2} \right\rceil + \frac{\pi^2}{3} \le \frac{8 \ln N}{\Delta_a^2} + 1 + \frac{\pi^2}{3}$$
Multiplying by $\Delta_a$ and summing over all suboptimal actions $a \in \mathcal{A}(s)$ with $\Delta_a > 0$:
$$R(N) = \sum_{a: \Delta_a > 0} \Delta_a \mathbb{E}[T_a(N)] \le \sum_{a: \Delta_a > 0} \left[ \frac{8 \ln N}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \Delta_a \right]$$
$$R(N) = \mathcal{O}\left( \sum_{a: \Delta_a > 0} \frac{\ln N}{\Delta_a} \right) = \mathcal{O}\left( \frac{K \ln N}{\Delta_{\min}} \right)$$

*Step 8: Kocsis & Szepesvári Tree Search Induction.*
In a tree of depth $D$, let node $s$ sit at depth $d \in \{0, \dots, D-1\}$. Unlike standard multi-armed bandits, the payoffs returned by child nodes are non-stationary because child nodes are themselves updating their empirical means via UCT.
Kocsis & Szepesvári (2006) proved by induction on tree depth $d$:
- **Base Case ($d = D-1$, immediate parents of leaves):** Leaf payoffs are static terminal evaluations; the empirical means are stationary, so the visit count bound holds with error probability $\mathcal{O}(t^{-4})$.
- **Inductive Step ($d < D-1$):** At depth $d$, the empirical mean $\bar{X}_{a, t}$ is a non-stationary average of child estimates. The bias of child estimates decays as $|\mathbb{E}[\bar{X}_{a, t}] - Q^*(s, a)| \le C_d t^{-1/2}$.
By applying Azuma-Hoeffding bounds for drifting martingales, the probability of selecting a suboptimal action at depth $d$ decays polynomially:
$$\mathbb{P}(I_t(d) = a) = \mathcal{O}\left( t^{-\rho_d} \right) \quad \text{for some } \rho_d > 0$$
Consequently, the root node selection converges almost surely to the true minimax value:
$$\lim_{N \to \infty} Q(s_0, a) = Q^*(s_0, a) \quad \text{and} \quad \lim_{N \to \infty} \mathbb{P}\left( I_N(s_0) \ne a^* \right) = 0 \quad \blacksquare$$

---

#### Derivation 11.23.2: PUCT Policy-Value Network Exploration Guidance Formula

```
====================================================================================================
DERIVATION 11.23.2: First-Principles Derivation of the PUCT Formula
====================================================================================================
Problem Statement:
Derive the AlphaZero PUCT (Polynomial Upper Confidence trees) action selection formula:
    a^* = argmax_{a ∈ 𝒜(s)} [ Q(s, a) + U(s, a) ]
where:
    U(s, a) = c_{puct} · P(s, a) · ( √(∑_{b ∈ 𝒜(s)} N(s, b)) ) / ( 1 + N(s, a) )
from first principles of Bayesian regret minimization under a non-uniform policy prior P(s, a),
prove that it satisfies asymptotic consistency (convergence to the true minimax action as N → ∞),
and show why the polynomial regularizer (1 + N(s, a)) eliminates the zero-visit explosion of UCB1.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
Let $s$ be an internal state in an MCTS tree with legal action set $\mathcal{A}(s)$. A deep neural network policy head outputs prior probabilities $\mathbf{P}(s) \triangleq [P(s, a_1), \dots, P(s, a_K)]^\top$ satisfying:
$$P(s, a) > 0 \quad \forall a \in \mathcal{A}(s) \quad \text{and} \quad \sum_{a \in \mathcal{A}(s)} P(s, a) = 1$$
Let $N(s, a)$ denote the empirical visit count of edge $(s, a)$, and let $N(s) \triangleq \sum_{b \in \mathcal{A}(s)} N(s, b)$ denote the total visit count to parent node $s$. Let $Q(s, a) \triangleq \frac{W(s, a)}{N(s, a)} \in [-1, 1]$ denote the mean action-value.
Our mathematical goal is to:
1. Derive the functional form $U(s, a) = c_{\text{puct}} P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)}$ from optimal information allocation under a Bayesian prior.
2. Prove that in equilibrium (equal $Q$-values), visit ratios asymptotically match prior ratios: $\frac{N(s, a)}{N(s, b)} \to \frac{P(s, a)}{P(s, b)}$.
3. Prove asymptotic consistency: as total simulations $N(s) \to \infty$, every legal action is explored infinitely often ($\lim_{N(s) \to \infty} N(s, a) = \infty$), and action selection converges strictly to the minimax optimal action $\arg\max_a Q^*(s, a)$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Valid Probability Simplex:** The prior distribution $\mathbf{P}(s)$ lies strictly in the interior of the probability simplex $\Delta^{K-1}$, meaning $P(s, a) \ge \epsilon_{\min} > 0$ for all legal actions $a$.
2. **Prior-Weighted Epistemic Uncertainty:** Under Bayesian belief modeling with prior distribution $P(s, a)$, the epistemic standard error $\sigma(s, a)$ of the action-value estimate scales proportionally to prior weight $P(s, a)$ and inversely with effective observations $(1 + N(s, a))$.
3. **Zero-Visit Regularity:** At $N(s, a) = 0$, the exploration bonus must remain finite and proportional to the prior $P(s, a)$, eliminating the division-by-zero singularity $\infty$ of standard UCB1.
4. **Scale Invariance of Simulation Horizon:** As the total simulation budget $N(s)$ scales by factor $\alpha$, the exploration pressure scales by $\sqrt{\alpha}$, maintaining dimensional balance between exploration variance and exploitation mean.
5. **AlphaZero Exploration Schedule:** The exploration coefficient is modulated by:
   $$c_{\text{puct}}(s) \triangleq \ln\left( \frac{1 + N(s) + c_{\text{base}}}{c_{\text{base}}} \right) + c_{\text{init}}, \quad c_{\text{init}} \approx 1.25, \quad c_{\text{base}} \approx 19652$$

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
In classic UCB1, the exploration bonus is $U_{\text{UCB}}(s, a) = c \sqrt{\frac{\ln N(s)}{N(s, a)}}$. When an action has never been visited ($N(s, a) = 0$), $U_{\text{UCB}} = \infty$.
In game trees with massive branching factors—such as Go ($K \le 361$) or Chess ($K \approx 35$)—this infinite bonus is catastrophic: it forces the search to explore *every single legal move* before deepening any single promising branch. If a position has 35 legal moves and 33 of them immediately hang the Queen, standard UCB1 expends 33 expensive neural evaluations on absurd blunders before performing depth-2 search on the 2 sensible candidate moves.
The PUCT formula resolves this by replacing the uniform, singular bonus with a **prior-gated polynomial bonus**:
$$U(s, a) = c_{\text{puct}} P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
- At $N(s, a) = 0$, $U(s, a) = c_{\text{puct}} P(s, a) \sqrt{N(s)} < \infty$.
- If the neural network predicts $P(s, a) = 0.0001$ (hanging a Queen), its exploration bonus is $0.0001 \times \sqrt{N(s)}$, which is negligibly small; the blunder is effectively pruned without wasting a single simulation.
- If $P(s, a) = 0.70$ (tactical masterstroke), its bonus is large, directing the search immediately down that line.
- The linear denominator $(1 + N(s, a))$ suppresses the bonus much faster than $\sqrt{N(s, a)}$, allowing empirical experience $Q(s, a)$ to quickly supersede prior intuition $P(s, a)$.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Constrained Information-Theoretic Budget Allocation.*
Let $N(s)$ be the total simulation budget available at node $s$, to be partitioned across actions: $\sum_{a=1}^K N(s, a) = N(s)$.
In Bayesian statistics, suppose we have prior belief $\mathbf{P}(s)$ representing the probability that action $a$ is optimal. The variance of our posterior value estimate for action $a$ after $N(s, a)$ independent trials is proportional to:
$$\operatorname{Var}(Q(s, a)) \approx \frac{\sigma_0^2}{1 + N(s, a)}$$
We seek a visit allocation vector $\mathbf{N} = (N(s, a_1), \dots, N(s, a_K))$ that minimizes the prior-weighted expected posterior variance:
$$\min_{N(s, a_1), \dots, N(s, a_K)} \sum_{a=1}^K P(s, a)^2 \cdot \frac{1}{1 + N(s, a)} \quad \text{subject to} \quad \sum_{a=1}^K N(s, a) = N(s)$$
We construct the Lagrangian with multiplier $\lambda > 0$:
$$\mathcal{L}(N_1, \dots, N_K, \lambda) = \sum_{a=1}^K \frac{P(s, a)^2}{1 + N(s, a)} + \lambda \left( \sum_{a=1}^K N(s, a) - N(s) \right)$$
Setting the partial derivative with respect to $N(s, a)$ to zero:
$$\frac{\partial \mathcal{L}}{\partial N(s, a)} = -\frac{P(s, a)^2}{(1 + N(s, a))^2} + \lambda = 0$$
$$\frac{P(s, a)^2}{(1 + N(s, a))^2} = \lambda \implies 1 + N(s, a) = \frac{P(s, a)}{\sqrt{\lambda}}$$
Summing over all actions $a \in \{1, \dots, K\}$:
$$\sum_{a=1}^K (1 + N(s, a)) = K + \sum_{a=1}^K N(s, a) = K + N(s) = \frac{1}{\sqrt{\lambda}} \sum_{a=1}^K P(s, a) = \frac{1}{\sqrt{\lambda}} \cdot 1$$
$$\implies \frac{1}{\sqrt{\lambda}} = N(s) + K$$
Substituting $\frac{1}{\sqrt{\lambda}}$ back into the expression for $1 + N(s, a)$:
$$1 + N(s, a)^* = P(s, a) (N(s) + K) \implies N(s, a)^* \approx P(s, a) N(s) \quad \text{for } N(s) \gg K$$
Thus, optimal allocation under prior $\mathbf{P}$ demands that visit counts grow proportionally to prior weights: $N(s, a) \propto P(s, a) N(s)$.

*Step 2: Exploration Bonus as Marginal Deficit Gradient.*
The marginal utility of allocating an additional visit to action $a$ is the negative gradient of the estimation variance with respect to $N(s, a)$:
$$-\frac{\partial}{\partial N(s, a)} \left[ \frac{P(s, a)^2}{1 + N(s, a)} \right] = \frac{P(s, a)^2}{(1 + N(s, a))^2}$$
Taking the square root to convert variance back to standard deviation (confidence radius) units:
$$\text{Radius}(s, a) \propto \frac{P(s, a)}{1 + N(s, a)}$$
To ensure that exploration pressure does not collapse to zero as total tree visits $N(s)$ grow, we scale the confidence radius by the global standard deviation of total root trials, which by the Central Limit Theorem scales as $\sqrt{N(s)}$:
$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_{b \in \mathcal{A}(s)} N(s, b)}}{1 + N(s, a)} \quad \text{(Eq. 2.1)}$$

*Step 3: Equilibrium Visit Ratio Analysis.*
Consider two legal actions $a_1$ and $a_2$ that yield identical empirical returns from search:
$$Q(s, a_1) = Q(s, a_2) = Q^*$$
The PUCT decision rule selects between them by comparing their scores:
$$\text{Score}(a_1) = Q^* + c_{\text{puct}} P(s, a_1) \frac{\sqrt{N(s)}}{1 + N(s, a_1)}$$
$$\text{Score}(a_2) = Q^* + c_{\text{puct}} P(s, a_2) \frac{\sqrt{N(s)}}{1 + N(s, a_2)}$$
In equilibrium, the algorithm alternates visits between $a_1$ and $a_2$ such that their PUCT scores remain equal:
$$\text{Score}(a_1) = \text{Score}(a_2) \iff c_{\text{puct}} P(s, a_1) \frac{\sqrt{N(s)}}{1 + N(s, a_1)} = c_{\text{puct}} P(s, a_2) \frac{\sqrt{N(s)}}{1 + N(s, a_2)}$$
Dividing both sides by $c_{\text{puct}} \sqrt{N(s)}$:
$$\frac{P(s, a_1)}{1 + N(s, a_1)} = \frac{P(s, a_2)}{1 + N(s, a_2)} \iff \frac{1 + N(s, a_1)}{1 + N(s, a_2)} = \frac{P(s, a_1)}{P(s, a_2)}$$
In the asymptotic regime where visit counts are large ($N(s, a) \gg 1$):
$$\lim_{N(s) \to \infty} \frac{N(s, a_1)}{N(s, a_2)} = \frac{P(s, a_1)}{P(s, a_2)}$$
The ratio of visits allocated by PUCT matches the exact ratio of prior probabilities, confirming that the prior correctly guides search effort under neutral value feedback.

*Step 4: Asymptotic Consistency Proof (Infinite Exploration of All Actions).*
We now prove that the PUCT rule guarantees **infinite exploration of every legal action** as total node visits $N(s) \to \infty$, ensuring that no action is permanently starved.
*Theorem:* If $N(s) = \sum_{b \in \mathcal{A}(s)} N(s, b) \to \infty$, then for every action $a \in \mathcal{A}(s)$ with prior $P(s, a) > 0$:
$$\lim_{N(s) \to \infty} N(s, a) = \infty$$
*Proof by Contradiction:*
Suppose for contradiction that there exists some action $a_{\text{starved}} \in \mathcal{A}(s)$ whose visit count is bounded by a finite integer $M_{\max} < \infty$:
$$N(s, a_{\text{starved}}) \le M_{\max} < \infty \quad \text{for all } N(s)$$
Now examine the exploration bonus of $a_{\text{starved}}$ as $N(s) \to \infty$:
$$U(s, a_{\text{starved}}) = c_{\text{puct}} P(s, a_{\text{starved}}) \frac{\sqrt{N(s)}}{1 + N(s, a_{\text{starved}})} \ge c_{\text{puct}} P(s, a_{\text{starved}}) \frac{\sqrt{N(s)}}{1 + M_{\max}}$$
Since $c_{\text{puct}} > 0, P(s, a_{\text{starved}}) > 0$, and $(1 + M_{\max})$ is a fixed finite constant:
$$\lim_{N(s) \to \infty} U(s, a_{\text{starved}}) \ge \frac{c_{\text{puct}} P(s, a_{\text{starved}})}{1 + M_{\max}} \lim_{N(s) \to \infty} \sqrt{N(s)} = +\infty$$
Because action values are bounded in $[-1, 1]$, the total selection score of the starved action satisfies:
$$\lim_{N(s) \to \infty} \left[ Q(s, a_{\text{starved}}) + U(s, a_{\text{starved}}) \right] \ge -1 + \infty = +\infty \quad \text{(Eq. 2.2)}$$
Now consider the set of actions that are visited infinitely often. Because the total visits $N(s) \to \infty$ and $|\mathcal{A}(s)| = K < \infty$, by the Pigeonhole Principle there exists at least one action $b^* \in \mathcal{A}(s)$ such that:
$$N(s, b^*) \ge \frac{N(s)}{K}$$
Evaluate the exploration bonus of this frequently visited action $b^*$:
$$U(s, b^*) = c_{\text{puct}} P(s, b^*) \frac{\sqrt{N(s)}}{1 + N(s, b^*)} \le c_{\text{puct}} (1) \frac{\sqrt{N(s)}}{N(s, b^*)} \le c_{\text{puct}} \frac{\sqrt{N(s)}}{\frac{N(s)}{K}} = \frac{c_{\text{puct}} K}{\sqrt{N(s)}}$$
Taking the limit as $N(s) \to \infty$:
$$\lim_{N(s) \to \infty} U(s, b^*) \le \lim_{N(s) \to \infty} \frac{c_{\text{puct}} K}{\sqrt{N(s)}} = 0$$
Since $Q(s, b^*) \le +1$:
$$\lim_{N(s) \to \infty} \left[ Q(s, b^*) + U(s, b^*) \right] \le 1 + 0 = 1 \quad \text{(Eq. 2.3)}$$
Comparing Eq. 2.2 and Eq. 2.3:
$$\text{Score}(s, a_{\text{starved}}) \to +\infty \quad \text{while} \quad \text{Score}(s, b^*) \le 1$$
Therefore, there exists a finite threshold $N^*$ such that for all $N(s) > N^*$:
$$Q(s, a_{\text{starved}}) + U(s, a_{\text{starved}}) > Q(s, b^*) + U(s, b^*)$$
At this point, the PUCT argmax rule is forced to select $a_{\text{starved}}$. This selection increments $N(s, a_{\text{starved}})$. This process repeats indefinitely whenever $N(s, a_{\text{starved}})$ falls behind, contradicting the hypothesis that $N(s, a_{\text{starved}}) \le M_{\max} < \infty$.
We conclude:
$$\lim_{N(s) \to \infty} N(s, a) = \infty \quad \forall a \in \mathcal{A}(s)$$

*Step 5: Asymptotic Optimality of Selection.*
Since every action is visited infinitely often ($N(s, a) \to \infty$):
1. By the Strong Law of Large Numbers, empirical values converge almost surely to their true minimax values:
   $$\lim_{N(s) \to \infty} Q(s, a) = Q^*(s, a) \quad \text{almost surely}$$
2. For any action $a$ with non-zero asymptotic fraction $\lim \frac{N(s, a)}{N(s)} > 0$, its bonus decays as:
   $$U(s, a) \propto \frac{\sqrt{N(s)}}{N(s, a)} = \frac{1}{\sqrt{N(s)}} \to 0$$
3. Thus, as $N(s) \to \infty$, the exploration bonus vanishes across all competitive branches:
   $$\lim_{N(s) \to \infty} \left[ Q(s, a) + U(s, a) \right] = Q^*(s, a)$$
4. The PUCT selection rule converges to the true game-theoretic minimax action:
   $$\lim_{N(s) \to \infty} \arg\max_{a \in \mathcal{A}(s)} \left[ Q(s, a) + U(s, a) \right] = \arg\max_{a \in \mathcal{A}(s)} Q^*(s, a) \quad \blacksquare$$

---

#### Derivation 11.23.3: AlphaZero Dual-Head Loss Function and Policy Improvement Theorem

```
====================================================================================================
DERIVATION 11.23.3: AlphaZero Dual-Head Loss Function and Policy Improvement Theorem
====================================================================================================
Problem Statement:
Derive the joint dual-head AlphaZero training loss function:
    ℒ(θ) = (z - v_θ(s))^2 - π_MCTS^⊤ log p_θ(s) + c ‖θ‖_2^2
from Maximum A Posteriori (MAP) estimation on self-play game trajectories, compute exact analytical
gradients with respect to neural network output heads, and prove the MCTS Policy Improvement Theorem
showing that the search policy π_MCTS is a strictly monotonic improvement over the network prior p_θ.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
Let self-play games be generated by AlphaZero. A completed game produces a sequence of transition tuples:
$$\mathcal{D} = \left\{ \left( s_i, \boldsymbol{\pi}_i, z_i \right) \right\}_{i=1}^B$$
where for transition $i$:
- $s_i$ is the board state.
- $\boldsymbol{\pi}_i \triangleq \boldsymbol{\pi}_{\text{MCTS}}(\cdot \mid s_i) \in \Delta^{K-1}$ is the search policy vector formed from MCTS visit counts:
  $$\pi_i(a) \triangleq \frac{N(s_i, a)^{1/\tau}}{\sum_{b=1}^K N(s_i, b)^{1/\tau}}$$
- $z_i \in \{-1, 0, +1\}$ is the terminal game outcome evaluated from the perspective of the player whose turn it was at state $s_i$.
The dual-headed neural network $f_\theta(s) = (\mathbf{p}_\theta(s), v_\theta(s))$ maps state $s$ to:
1. Policy head logits $\mathbf{z}_{\text{pol}} \in \mathbb{R}^K$, producing probabilities via softmax:
   $$p_\theta(a \mid s) = \frac{\exp(z_{\text{pol}, a})}{\sum_{b=1}^K \exp(z_{\text{pol}, b})}$$
2. Value head logit $z_{\text{val}} \in \mathbb{R}$, producing scalar evaluation via hyperbolic tangent:
   $$v_\theta(s) = \tanh(z_{\text{val}}) \in [-1, 1]$$
Our mathematical goal is to:
1. Derive the joint loss function $\mathcal{L}(\theta) = (z - v_\theta(s))^2 - \boldsymbol{\pi}_i^\top \log \mathbf{p}_\theta(s) + c \|\theta\|_2^2$ from Bayesian Maximum A Posteriori (MAP) principles.
2. Derive exact closed-form analytical gradients with respect to the network logits:
   $$\nabla_{\mathbf{z}_{\text{pol}}} \mathcal{L}_{\text{pol}} = \mathbf{p}_\theta(s) - \boldsymbol{\pi}_i \quad \text{and} \quad \frac{\partial \mathcal{L}_{\text{val}}}{\partial z_{\text{val}}} = -2 (z - v_\theta)(1 - v_\theta^2)$$
3. Prove the MCTS Policy Improvement Theorem: $\mathbb{E}_{a \sim \boldsymbol{\pi}}[Q(s, a)] \ge \mathbb{E}_{a \sim \mathbf{p}}[Q(s, a)]$, showing that MCTS acts as a strict Generalized Policy Iteration operator.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Conditional Independence:** Conditioned on the state $s_i$ and network weights $\theta$, the observed game outcome $z_i$ and the search policy target $\boldsymbol{\pi}_i$ are conditionally independent:
   $$p(z_i, \boldsymbol{\pi}_i \mid s_i, \theta) = p(z_i \mid s_i, \theta) \cdot p(\boldsymbol{\pi}_i \mid s_i, \theta)$$
2. **Gaussian Value Likelihood:** The terminal game outcome $z_i$ is modeled as a Gaussian random variable centered at $v_\theta(s_i)$ with observation variance $\sigma_v^2 = \frac{1}{2}$:
   $$z_i \mid s_i, \theta \sim \mathcal{N}\left( v_\theta(s_i), \, \sigma_v^2 \right)$$
3. **Multinomial Search Policy Likelihood:** The MCTS simulation count vector $\mathbf{N}_i = (N_{i, 1}, \dots, N_{i, K})$ containing $M = \sum_a N_{i, a}$ total visits follows a multinomial distribution parameterized by the neural prior $\mathbf{p}_\theta(s_i)$:
   $$\mathbf{N}_i \mid s_i, \theta \sim \operatorname{Multinomial}(M, \, \mathbf{p}_\theta(s_i))$$
4. **Isotropic Gaussian Weight Prior:** Network parameters $\theta \in \mathbb{R}^D$ follow an isotropic Gaussian prior centered at zero:
   $$\theta \sim \mathcal{N}\left( \mathbf{0}, \, \sigma_\theta^2 \mathbf{I} \right)$$
5. **Differentiability:** Activations $\operatorname{softmax}$ and $\tanh$ are infinitely differentiable ($\mathcal{C}^\infty$).

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
AlphaZero implements **Generalized Policy Iteration (GPI)** in a continuous, self-reinforcing self-play loop:
1. **Policy Improvement (MCTS):** Given a current neural network prior $\mathbf{p}_\theta$, MCTS conducts $M$ simulations of forward lookahead tree search. Because MCTS aggregates deep tactical rollouts and propagates minimax backed-up values, the resulting visit count distribution $\boldsymbol{\pi}_{\text{MCTS}}$ is a **strictly superior, tactically sounder policy** than the raw prior $\mathbf{p}_\theta$.
2. **Policy Evaluation & Distillation (Gradient Descent):** However, running 800 MCTS simulations at test time for every single move is computationally expensive. The dual-head loss acts as a **policy distillation and value regression engine**:
   - The cross-entropy loss $-\boldsymbol{\pi}^\top \log \mathbf{p}$ trains the fast neural intuition $\mathbf{p}_\theta$ to mimic the deep tactical search $\boldsymbol{\pi}_{\text{MCTS}}$. In information geometry, this minimizes the Kullback-Leibler divergence $D_{\text{KL}}(\boldsymbol{\pi} \parallel \mathbf{p})$, projecting the tree search distribution onto the neural network manifold.
   - The MSE loss $(z - v_\theta)^2$ regresses the scalar value head onto the true game outcome $z \in \{-1, 0, +1\}$, forcing $v_\theta(s)$ to converge to the true minimax game-theoretic winning probability $P(\text{Win} \mid s) - P(\text{Loss} \mid s)$.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Joint Posterior Formulation via Bayes' Theorem.*
Given dataset $\mathcal{D} = \{(s_i, \boldsymbol{\pi}_i, z_i)\}_{i=1}^B$, the posterior distribution of parameter vector $\theta$ is given by Bayes' Theorem:
$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) p(\theta)}{p(\mathcal{D})} \propto p(\theta) \prod_{i=1}^B p(z_i, \boldsymbol{\pi}_i \mid s_i, \theta)$$
Applying conditional independence of the value outcome and search distribution:
$$p(\theta \mid \mathcal{D}) \propto p(\theta) \prod_{i=1}^B \left[ p(z_i \mid s_i, \theta) \cdot p(\boldsymbol{\pi}_i \mid s_i, \theta) \right]$$
Taking the natural logarithm of both sides:
$$\ln p(\theta \mid \mathcal{D}) = \ln p(\theta) + \sum_{i=1}^B \ln p(z_i \mid s_i, \theta) + \sum_{i=1}^B \ln p(\boldsymbol{\pi}_i \mid s_i, \theta) + \text{const} \quad \text{(Eq. 3.1)}$$

*Step 2: Log-Likelihood of Value and Policy Heads.*
1. **Value Head Log-Likelihood:**
   Under the Gaussian model $z_i \sim \mathcal{N}(v_\theta(s_i), \sigma_v^2)$:
   $$p(z_i \mid s_i, \theta) = \frac{1}{\sqrt{2\pi \sigma_v^2}} \exp\left( -\frac{(z_i - v_\theta(s_i))^2}{2 \sigma_v^2} \right)$$
   $$\ln p(z_i \mid s_i, \theta) = -\frac{1}{2 \sigma_v^2} (z_i - v_\theta(s_i))^2 - \frac{1}{2} \ln(2\pi \sigma_v^2)$$
   Setting $\sigma_v^2 = \frac{1}{2}$:
   $$\ln p(z_i \mid s_i, \theta) = -(z_i - v_\theta(s_i))^2 + \text{const}$$

2. **Policy Head Log-Likelihood:**
   Under the Multinomial model with $M$ visits and normalized targets $\boldsymbol{\pi}_i = \frac{\mathbf{N}_i}{M}$:
   $$p(\mathbf{N}_i \mid s_i, \theta) = \frac{M!}{\prod_{a=1}^K N_{i, a}!} \prod_{a=1}^K p_\theta(a \mid s_i)^{N_{i, a}}$$
   Taking the natural logarithm and dividing by total visits $M$:
   $$\frac{1}{M} \ln p(\mathbf{N}_i \mid s_i, \theta) = \sum_{a=1}^K \frac{N_{i, a}}{M} \ln p_\theta(a \mid s_i) + \frac{1}{M} \ln \left( \frac{M!}{\prod_a N_{i, a}!} \right)$$
   Recognizing $\frac{N_{i, a}}{M} = \pi_i(a)$:
   $$\frac{1}{M} \ln p(\mathbf{N}_i \mid s_i, \theta) = \sum_{a=1}^K \pi_i(a) \ln p_\theta(a \mid s_i) + \text{const} = \boldsymbol{\pi}_i^\top \log \mathbf{p}_\theta(s_i) + \text{const}$$

3. **Gaussian Prior on Weights:**
   Under $\theta \sim \mathcal{N}(\mathbf{0}, \sigma_\theta^2 \mathbf{I})$ with $D$ parameters:
   $$\ln p(\theta) = -\frac{\|\theta\|_2^2}{2 \sigma_\theta^2} - \frac{D}{2} \ln(2\pi \sigma_\theta^2)$$

*Step 3: Deriving the AlphaZero Objective Functional.*
Substitute these three expressions into Eq. 3.1:
$$\ln p(\theta \mid \mathcal{D}) = \sum_{i=1}^B \left[ -(z_i - v_\theta(s_i))^2 + M \boldsymbol{\pi}_i^\top \log \mathbf{p}_\theta(s_i) \right] - \frac{\|\theta\|_2^2}{2 \sigma_\theta^2} + \text{const}$$
Maximum A Posteriori (MAP) estimation minimizes the negative log-posterior. Scaling by $\frac{1}{M}$ per sample and defining the $L_2$ weight penalty $c \triangleq \frac{1}{2 M \sigma_\theta^2}$, the per-sample training loss is:
$$\mathcal{L}(\theta) = \left( z - v_\theta(s) \right)^2 - \boldsymbol{\pi}_{\text{MCTS}}^\top \log \mathbf{p}_\theta(s) + c \|\theta\|_2^2 \quad \text{(Eq. 3.2)}$$

*Step 4: Analytical Policy Head Gradient with Softmax Output.*
Let the policy logits be $\mathbf{z} \triangleq \mathbf{z}_{\text{pol}} \in \mathbb{R}^K$, with probabilities $p_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$.
The policy loss is $\mathcal{L}_{\text{pol}} = -\sum_{k=1}^K \pi_k \ln p_k$.
First, compute the Jacobian of the softmax transformation $\frac{\partial p_k}{\partial z_i}$:
- **Case 1 ($k = i$):**
  $$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \sum_j \exp(z_j) - \exp(z_i) \exp(z_i)}{\left( \sum_j \exp(z_j) \right)^2} = \frac{\exp(z_i)}{\sum_j \exp(z_j)} - \left( \frac{\exp(z_i)}{\sum_j \exp(z_j)} \right)^2 = p_i (1 - p_i)$$
- **Case 2 ($k \ne i$):**
  $$\frac{\partial p_k}{\partial z_i} = \frac{0 - \exp(z_k) \exp(z_i)}{\left( \sum_j \exp(z_j) \right)^2} = -p_k p_i$$
Combining both cases using the Kronecker delta $\delta_{ik}$:
$$\frac{\partial p_k}{\partial z_i} = p_k (\delta_{ik} - p_i)$$
Now differentiate $\mathcal{L}_{\text{pol}}$ with respect to logit $z_i$ via the multivariate chain rule:
$$\frac{\partial \mathcal{L}_{\text{pol}}}{\partial z_i} = -\sum_{k=1}^K \frac{\pi_k}{p_k} \frac{\partial p_k}{\partial z_i} = -\sum_{k=1}^K \frac{\pi_k}{p_k} \left[ p_k (\delta_{ik} - p_i) \right]$$
Canceling $p_k$:
$$\frac{\partial \mathcal{L}_{\text{pol}}}{\partial z_i} = -\sum_{k=1}^K \pi_k (\delta_{ik} - p_i) = -\left( \pi_i (1 - p_i) + \sum_{k \ne i} \pi_k (-p_i) \right)$$
$$= -\pi_i + \pi_i p_i + p_i \sum_{k \ne i} \pi_k = -\pi_i + p_i \left( \pi_i + \sum_{k \ne i} \pi_k \right)$$
Since $\boldsymbol{\pi}$ is a valid probability distribution, $\sum_{k=1}^K \pi_k = \pi_i + \sum_{k \ne i} \pi_k = 1$:
$$\frac{\partial \mathcal{L}_{\text{pol}}}{\partial z_i} = -\pi_i + p_i (1) = p_i - \pi_i$$
In full vector form:
$$\nabla_{\mathbf{z}_{\text{pol}}} \mathcal{L}_{\text{pol}} = \mathbf{p}_\theta(s) - \boldsymbol{\pi}_{\text{MCTS}} \quad \text{(Eq. 3.3)}$$
This gradient is remarkably elegant, numerically stable, and naturally bounded: $\|\nabla_{\mathbf{z}_{\text{pol}}} \mathcal{L}_{\text{pol}}\|_\infty \le 1$.

*Step 5: Analytical Value Head Gradient with Tanh Output.*
Let the value logit be $z_{\text{val}} \in \mathbb{R}$, with output $v = \tanh(z_{\text{val}})$.
The value loss is $\mathcal{L}_{\text{val}} = (z - v)^2$.
Differentiating with respect to $v$:
$$\frac{\partial \mathcal{L}_{\text{val}}}{\partial v} = 2 (z - v) \cdot (-1) = -2 (z - v)$$
Differentiating $v = \tanh(z_{\text{val}})$ with respect to $z_{\text{val}}$:
$$\frac{dv}{dz_{\text{val}}} = \frac{d}{dz_{\text{val}}} \left( \frac{\sinh(z_{\text{val}})}{\cosh(z_{\text{val}})} \right) = \frac{\cosh^2(z_{\text{val}}) - \sinh^2(z_{\text{val}})}{\cosh^2(z_{\text{val}})} = 1 - \tanh^2(z_{\text{val}}) = 1 - v^2$$
Applying the scalar chain rule:
$$\frac{\partial \mathcal{L}_{\text{val}}}{\partial z_{\text{val}}} = \frac{\partial \mathcal{L}_{\text{val}}}{\partial v} \cdot \frac{dv}{dz_{\text{val}}} = -2 (z - v)(1 - v^2) \quad \text{(Eq. 3.4)}$$

*Step 6: Proof of the MCTS Policy Improvement Theorem.*
We now prove that the MCTS search policy $\boldsymbol{\pi}_{\text{MCTS}}$ achieves an expected action value strictly greater than or equal to that of the neural network prior $\mathbf{p}$:
$$\sum_{a=1}^K \pi_{\text{MCTS}}(a) Q(s, a) \ge \sum_{a=1}^K p(a) Q(s, a)$$
*Proof:*
Sort the $K$ legal actions in descending order of empirical action values:
$$Q(s, a_{(1)}) \ge Q(s, a_{(2)}) \ge \dots \ge Q(s, a_{(K)})$$
Under the PUCT selection mechanism, the selection score is $Q(s, a) + U(s, a)$. For actions with equal or higher prior, higher $Q$-value actions receive strictly more visits.
After $M$ simulations, the visit counts $N(s, a_{(k)})$ satisfy the Monotone Likelihood Ratio Property (MLRP) with respect to the prior:
$$\frac{\pi_{(1)}}{p_{(1)}} \ge \frac{\pi_{(2)}}{p_{(2)}} \ge \dots \ge \frac{\pi_{(K)}}{p_{(K)}}$$
Define the cumulative distribution functions $F_{\boldsymbol{\pi}}(m) \triangleq \sum_{k=1}^m \pi_{(k)}$ and $F_{\mathbf{p}}(m) \triangleq \sum_{k=1}^m p_{(k)}$.
Due to the MLRP condition and $\sum_{k=1}^K \pi_{(k)} = \sum_{k=1}^K p_{(k)} = 1$, the search policy **first-order stochastically dominates** the prior policy:
$$F_{\boldsymbol{\pi}}(m) \ge F_{\mathbf{p}}(m) \quad \forall m \in \{1, \dots, K\}$$
Applying Abel's summation by parts to the difference of expectations:
$$\sum_{k=1}^K \pi_{(k)} Q_{(k)} - \sum_{k=1}^K p_{(k)} Q_{(k)} = \sum_{k=1}^K (\pi_{(k)} - p_{(k)}) Q_{(k)}$$
$$= \sum_{m=1}^{K-1} \left( F_{\boldsymbol{\pi}}(m) - F_{\mathbf{p}}(m) \right) \left( Q_{(m)} - Q_{(m+1)} \right) + \left( F_{\boldsymbol{\pi}}(K) - F_{\mathbf{p}}(K) \right) Q_{(K)}$$
Since $F_{\boldsymbol{\pi}}(K) = F_{\mathbf{p}}(K) = 1$, the terminal term vanishes:
$$\sum_{k=1}^K (\pi_{(k)} - p_{(k)}) Q_{(k)} = \sum_{m=1}^{K-1} \underbrace{\left( F_{\boldsymbol{\pi}}(m) - F_{\mathbf{p}}(m) \right)}_{\ge 0} \underbrace{\left( Q_{(m)} - Q_{(m+1)} \right)}_{\ge 0} \ge 0$$
Equality holds if and only if all action values are identical ($Q_{(1)} = Q_{(K)}$) or $\boldsymbol{\pi} \equiv \mathbf{p}$.
Therefore, MCTS is a strictly monotonic policy improvement operator:
$$\boldsymbol{\pi}_{\text{MCTS}} \succ \mathbf{p}_\theta(s) \quad \blacksquare$$

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

### Illustration 1: Zero-Sum Value Inversion & Negamax Minimax Equivalence
**Problem:**
In a 2-player zero-sum game (Player 1 / White vs. Player 2 / Black), prove why negamax value negation ($v_{\text{parent}} \leftarrow -v_{\text{child}}$) is mathematically equivalent to minimax optimization. Demonstrate this equivalence with a concrete 2-ply game tree:
- Root state $s_0$ (White to move) has two legal actions $\{a_1, a_2\}$ leading to Black decision states $s_1$ and $s_2$.
- From $s_1$, Black has actions $\{b_{11}, b_{12}\}$ leading to terminal positions with White payoffs:
  $$v_{\text{White}}(s_{11}) = +0.70, \quad v_{\text{White}}(s_{12}) = +0.20$$
- From $s_2$, Black has actions $\{b_{21}, b_{22}\}$ leading to terminal positions with White payoffs:
  $$v_{\text{White}}(s_{21}) = -0.50, \quad v_{\text{White}}(s_{22}) = +0.90$$

**Solution:**
1. **Classical Minimax Formulation (White Maximizes, Black Minimizes White's Payoff):**
   - At node $s_1$, Black chooses the action minimizing White's payoff:
     $$v(s_1) = \min\left( v_{\text{White}}(s_{11}), \, v_{\text{White}}(s_{12}) \right) = \min(0.70, \, 0.20) = \mathbf{+0.20}$$
   - At node $s_2$, Black chooses the action minimizing White's payoff:
     $$v(s_2) = \min\left( v_{\text{White}}(s_{21}), \, v_{\text{White}}(s_{22}) \right) = \min(-0.50, \, 0.90) = \mathbf{-0.50}$$
   - At root $s_0$, White chooses the action maximizing White's payoff:
     $$v^*(s_0) = \max\left( v(s_1), \, v(s_2) \right) = \max(+0.20, \, -0.50) = \mathbf{+0.20} \quad (\text{Select } a_1)$$

2. **AlphaZero Negamax Formulation (Every Player Maximizes Their Own Utility):**
   By zero-sum duality, Black's utility is $v_{\text{Black}} = -v_{\text{White}}$.
   - Terminal states from Black's perspective:
     $$v_{\text{Black}}(s_{11}) = -0.70, \quad v_{\text{Black}}(s_{12}) = -0.20$$
     $$v_{\text{Black}}(s_{21}) = +0.50, \quad v_{\text{Black}}(s_{22}) = -0.90$$
   - At node $s_1$, Black maximizes Black's utility:
     $$u(s_1) = \max(-0.70, \, -0.20) = \mathbf{-0.20} \quad (\text{Select } b_{12})$$
   - At node $s_2$, Black maximizes Black's utility:
     $$u(s_2) = \max(+0.50, \, -0.90) = \mathbf{+0.50} \quad (\text{Select } b_{21})$$
   - When backing up to parent node $s_0$, White negates Black's utility ($v_{\text{parent}} = -u_{\text{child}}$) and maximizes:
     $$v^*(s_0) = \max\left( -u(s_1), \, -u(s_2) \right) = \max(-(-0.20), \, -(+0.50)) = \max(+0.20, \, -0.50) = \mathbf{+0.20}$$
   Using the algebraic identity $\max(-x, -y) \equiv -\min(x, y)$, both formulations yield identical decisions ($a_1$) and values ($+0.20$)! $\blacksquare$

---

### Illustration 2: Step-by-Step PUCT Selection Across 3 Actions
**Problem:**
At the root node $s_0$ of an AlphaZero search tree, there are 3 candidate actions $\mathcal{A}(s_0) = \{a_1, a_2, a_3\}$ with the following current statistics:
- Policy network priors: $\mathbf{P} = [0.60, 0.30, 0.10]$
- Visit counts: $\mathbf{N} = [10, 5, 1]$
- Empirical mean values: $\mathbf{Q} = [+0.40, +0.50, +0.20]$
- Exploration constant: $c_{\text{puct}} = 1.414$

Calculate the total parent visits $\sum_b N(s_0, b)$, the exploration bonus $U(s_0, a)$ for each action, the overall PUCT score $Q(s_0, a) + U(s_0, a)$, and determine which action is selected.

**Solution:**
1. **Compute Total Parent Visits and Square Root:**
   $$N_{\text{total}} = \sum_{b=1}^3 N(s_0, b) = 10 + 5 + 1 = \mathbf{16}$$
   $$\sqrt{N_{\text{total}}} = \sqrt{16} = \mathbf{4.000000}$$

2. **Evaluate Exploration Bonus $U(s_0, a) = c_{\text{puct}} \cdot P(s_0, a) \cdot \frac{\sqrt{N_{\text{total}}}}{1 + N(s_0, a)}$:**
   - **For Action $a_1$:**
     $$U(s_0, a_1) = 1.414 \times 0.60 \times \frac{4.000000}{1 + 10} = 0.8484 \times \frac{4}{11} = \frac{3.3936}{11} \approx \mathbf{0.308509}$$
     $$\text{Score}(a_1) = Q(s_0, a_1) + U(s_0, a_1) = 0.400000 + 0.308509 = \mathbf{0.708509}$$

   - **For Action $a_2$:**
     $$U(s_0, a_2) = 1.414 \times 0.30 \times \frac{4.000000}{1 + 5} = 0.4242 \times \frac{4}{6} = 0.4242 \times \frac{2}{3} = \frac{0.8484}{3} = \mathbf{0.282800}$$
     $$\text{Score}(a_2) = Q(s_0, a_2) + U(s_0, a_2) = 0.500000 + 0.282800 = \mathbf{0.782800}$$

   - **For Action $a_3$:**
     $$U(s_0, a_3) = 1.414 \times 0.10 \times \frac{4.000000}{1 + 1} = 0.1414 \times \frac{4}{2} = 0.1414 \times 2.0 = \mathbf{0.282800}$$
     $$\text{Score}(a_3) = Q(s_0, a_3) + U(s_0, a_3) = 0.200000 + 0.282800 = \mathbf{0.482800}$$

3. **Comparison and Selection:**
   $$\text{Score}(a_2) = \mathbf{0.782800} > \text{Score}(a_1) = \mathbf{0.708509} > \text{Score}(a_3) = \mathbf{0.482800}$$
   **Result:** **Action $a_2$ is selected** because its higher empirical value ($Q = 0.50$ vs $0.40$) outweighs $a_1$'s higher prior ($0.60$ vs $0.30$). $\blacksquare$

---

### Illustration 3: Multi-Ply Backpropagation Through a 3-Level Tree with Virtual Loss
**Problem:**
In a parallelized AlphaZero MCTS implementation, multiple worker threads search the tree concurrently. To prevent threads from duplicating search trajectories, a **virtual loss** of $n_{\text{vl}} = 1$ with penalty $v_{\text{loss}} = -1.0$ is applied to traversed edges during selection.
Consider a 3-level tree path from root $s_0$:
- Edge $(s_0, a_0) \to s_1$: Player 1 to move. Prior state: $N(s_0, a_0) = 4, \, W(s_0, a_0) = 1.60 \implies Q = +0.4000$.
- Edge $(s_1, a_1) \to s_2$: Player 2 to move. Prior state: $N(s_1, a_1) = 2, \, W(s_1, a_1) = -0.60 \implies Q = -0.3000$.
- State $s_2$: Leaf node (Player 1 to move).
Trace:
1. The temporary visit counts and $Q$-values during the selection phase when virtual loss is applied.
2. The neural network leaf evaluation of $s_2$ returning $v_{\text{eval}} = +0.75$ (from Player 1's perspective).
3. The exact backup phase that reverts the virtual loss and applies the true backpropagated evaluation up to the root.

**Solution:**
1. **Selection Phase (Applying Virtual Loss):**
   When Thread $T$ traverses the path $(s_0, a_0) \to (s_1, a_1)$:
   - At edge $(s_0, a_0)$:
     $$N_{\text{active}} = N + n_{\text{vl}} = 4 + 1 = \mathbf{5}$$
     $$W_{\text{active}} = W + n_{\text{vl}} \cdot v_{\text{loss}} = 1.60 + 1 \times (-1.0) = \mathbf{0.60}$$
     $$Q_{\text{active}}(s_0, a_0) = \frac{W_{\text{active}}}{N_{\text{active}}} = \frac{0.60}{5} = \mathbf{0.1200} \quad (\text{Dropped from } 0.4000 \to 0.1200)$$
   - At edge $(s_1, a_1)$:
     $$N_{\text{active}} = N + n_{\text{vl}} = 2 + 1 = \mathbf{3}$$
     $$W_{\text{active}} = W + n_{\text{vl}} \cdot v_{\text{loss}} = -0.60 + 1 \times (-1.0) = \mathbf{-1.60}$$
     $$Q_{\text{active}}(s_1, a_1) = \frac{-1.60}{3} \approx \mathbf{-0.5333} \quad (\text{Dropped from } -0.3000 \to -0.5333)$$
   Other threads traversing $s_0$ or $s_1$ now see artificially depressed scores, diverting them to explore alternative branches!

2. **Evaluation Phase:**
   Neural network evaluates leaf $s_2$:
   $$v(s_2) = \mathbf{+0.7500} \quad (\text{Perspective: Player 1})$$

3. **Backup Phase (Reverting Virtual Loss & Accumulating True Value):**
   - **At Edge $(s_1, a_1)$ (Player 2's turn):**
     Since Player 1 evaluates $s_2$ as $+0.75$, the value from Player 2's perspective is:
     $$v_2 = -v(s_2) = -0.7500$$
     Revert virtual loss and record true visit:
     $$N(s_1, a_1) \leftarrow N_{\text{prior}} + 1 = 2 + 1 = \mathbf{3}$$
     $$W(s_1, a_1) \leftarrow W_{\text{prior}} + v_2 = -0.60 + (-0.7500) = \mathbf{-1.3500}$$
     $$Q(s_1, a_1) \leftarrow \frac{-1.3500}{3} = \mathbf{-0.4500}$$

   - **At Edge $(s_0, a_0)$ (Player 1's turn):**
     Inverting Player 2's value back to Player 1's perspective:
     $$v_1 = -v_2 = -(-0.7500) = +0.7500$$
     Revert virtual loss and record true visit:
     $$N(s_0, a_0) \leftarrow N_{\text{prior}} + 1 = 4 + 1 = \mathbf{5}$$
     $$W(s_0, a_0) \leftarrow W_{\text{prior}} + v_1 = 1.60 + 0.7500 = \mathbf{2.3500}$$
     $$Q(s_0, a_0) \leftarrow \frac{2.3500}{5} = \mathbf{0.4700}$$

   **Summary Table:**
   | Level / Edge | Player | Prior $N$ | Prior $Q$ | Active $Q$ (VL) | Leaf Backed $v$ | Final $N$ | Final $W$ | Final $Q$ |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | $(s_1, a_1)$ | P2 | $2$ | $-0.3000$ | $-0.5333$ | $-0.7500$ | **$3$** | **$-1.3500$** | **$-0.4500$** |
   | $(s_0, a_0)$ | P1 | $4$ | $+0.4000$ | $+0.1200$ | $+0.7500$ | **$5$** | **$+2.3500$** | **$+0.4700$** |
   $\blacksquare$

---

### Illustration 4: AlphaZero Policy Target Under Temperature Annealing ($\tau = 1.0 \to 0.2$)
**Problem:**
After $M = 100$ simulations from root state $s$, the empirical visit counts across 4 legal actions are:
$$\mathbf{N} = [N(s, a_1), \, N(s, a_2), \, N(s, a_3), \, N(s, a_4)] = [60, \, 25, \, 10, \, 5]$$
The policy target vector $\boldsymbol{\pi}_\tau$ is defined by:
$$\pi_\tau(a) = \frac{N(s, a)^{1/\tau}}{\sum_{b=1}^4 N(s, b)^{1/\tau}}$$
Calculate the exact policy target probabilities under three temperature regimes:
1. Training exploration temperature: $\tau = 1.0$ ($1/\tau = 1.0$).
2. Intermediate temperature: $\tau = 0.5$ ($1/\tau = 2.0$).
3. Competitive play temperature: $\tau = 0.2$ ($1/\tau = 5.0$).
4. Asymptotic competitive limit: $\tau \to 0$.

**Solution:**
1. **Regime 1: Exploration ($\tau = 1.0$, $1/\tau = 1$):**
   $$N_a^1 = [60, \, 25, \, 10, \, 5]$$
   $$\sum_{b=1}^4 N_b^1 = 60 + 25 + 10 + 5 = 100$$
   $$\pi_{1.0}(a_1) = \frac{60}{100} = \mathbf{0.600000}, \quad \pi_{1.0}(a_2) = \frac{25}{100} = \mathbf{0.250000}$$
   $$\pi_{1.0}(a_3) = \frac{10}{100} = \mathbf{0.100000}, \quad \pi_{1.0}(a_4) = \frac{5}{100} = \mathbf{0.050000}$$

2. **Regime 2: Intermediate Annealing ($\tau = 0.5$, $1/\tau = 2$):**
   $$N_a^2 = [60^2, \, 25^2, \, 10^2, \, 5^2] = [3600, \, 625, \, 100, \, 25]$$
   $$\sum_{b=1}^4 N_b^2 = 3600 + 625 + 100 + 25 = \mathbf{4350}$$
   $$\pi_{0.5}(a_1) = \frac{3600}{4350} = \frac{24}{29} \approx \mathbf{0.827586}$$
   $$\pi_{0.5}(a_2) = \frac{625}{4350} = \frac{25}{174} \approx \mathbf{0.143678}$$
   $$\pi_{0.5}(a_3) = \frac{100}{4350} = \frac{2}{87} \approx \mathbf{0.022989}$$
   $$\pi_{0.5}(a_4) = \frac{25}{4350} = \frac{1}{174} \approx \mathbf{0.005747}$$

3. **Regime 3: Low Temperature ($\tau = 0.2$, $1/\tau = 5$):**
   $$N_a^5 = [60^5, \, 25^5, \, 10^5, \, 5^5]$$
   $$60^5 = 777,600,000$$
   $$25^5 = 9,765,625$$
   $$10^5 = 100,000$$
   $$5^5 = 3,125$$
   $$\sum_{b=1}^4 N_b^5 = 777,600,000 + 9,765,625 + 100,000 + 3,125 = \mathbf{787,468,750}$$
   $$\pi_{0.2}(a_1) = \frac{777,600,000}{787,468,750} \approx \mathbf{0.987468}$$
   $$\pi_{0.2}(a_2) = \frac{9,765,625}{787,468,750} \approx \mathbf{0.012401}$$
   $$\pi_{0.2}(a_3) = \frac{100,000}{787,468,750} \approx \mathbf{0.000127}$$
   $$\pi_{0.2}(a_4) = \frac{3,125}{787,468,750} \approx \mathbf{0.000004}$$

4. **Regime 4: Greedy Limit ($\tau \to 0$):**
   $$\lim_{\tau \to 0} \pi_\tau(a) = \begin{cases} 1.0 & \text{if } a = \arg\max_b N(s, b) = a_1 \\ 0.0 & \text{otherwise} \end{cases} \implies \boldsymbol{\pi}_0 = [\mathbf{1.0}, \, \mathbf{0.0}, \, \mathbf{0.0}, \, \mathbf{0.0}]$$
   Notice how annealing $\tau = 1.0 \to 0.2$ amplifies the dominant action probability from $60\%$ to $98.75\%$, suppressing exploratory noise for match play! $\blacksquare$

---

### Illustration 5: Dual-Head Gradient Backpropagation on a Single State
**Problem:**
A single state $s$ has 3 legal actions. The neural network outputs policy logits $\mathbf{z}_{\text{pol}} = [1.2, 0.4, -0.6]$ and value logit $z_{\text{val}} = 0.50$.
From self-play tree search, the target search policy is $\boldsymbol{\pi}_{\text{target}} = [0.70, 0.20, 0.10]$ and the terminal outcome is $z_{\text{target}} = +1.00$ (win).
Calculate:
1. The forward network predictions $\mathbf{p}_\theta(s) = \operatorname{softmax}(\mathbf{z}_{\text{pol}})$ and $v_\theta(s) = \tanh(z_{\text{val}})$.
2. The policy cross-entropy loss $\mathcal{L}_{\text{pol}}$ and value MSE loss $\mathcal{L}_{\text{val}}$.
3. The exact analytical gradient vector with respect to policy logits $\nabla_{\mathbf{z}_{\text{pol}}} \mathcal{L}_{\text{pol}}$ and value logit $\frac{\partial \mathcal{L}_{\text{val}}}{\partial z_{\text{val}}}$.

**Solution:**
1. **Forward Propagation:**
   - **Policy Head Softmax:**
     $$\exp(z_1) = e^{1.2} \approx 3.320117$$
     $$\exp(z_2) = e^{0.4} \approx 1.491825$$
     $$\exp(z_3) = e^{-0.6} \approx 0.548812$$
     $$\sum_{k=1}^3 \exp(z_k) = 3.320117 + 1.491825 + 0.548812 = 5.360754$$
     $$p_1 = \frac{3.320117}{5.360754} \approx \mathbf{0.619338}$$
     $$p_2 = \frac{1.491825}{5.360754} \approx \mathbf{0.278286}$$
     $$p_3 = \frac{0.548812}{5.360754} \approx \mathbf{0.102376}$$
   - **Value Head Tanh:**
     $$v = \tanh(0.50) = \frac{e^{0.50} - e^{-0.50}}{e^{0.50} + e^{-0.50}} = \frac{1.648721 - 0.606531}{1.648721 + 0.606531} = \frac{1.042190}{2.255252} \approx \mathbf{0.462117}$$

2. **Loss Computation:**
   - **Policy Cross-Entropy Loss:**
     $$\mathcal{L}_{\text{pol}} = -\sum_{k=1}^3 \pi_k \ln p_k = -\left[ 0.70 \ln(0.619338) + 0.20 \ln(0.278286) + 0.10 \ln(0.102376) \right]$$
     $$\ln(0.619338) \approx -0.479104 \implies 0.70 \times (-0.479104) = -0.335373$$
     $$\ln(0.278286) \approx -1.279104 \implies 0.20 \times (-1.279104) = -0.255821$$
     $$\ln(0.102376) \approx -2.279104 \implies 0.10 \times (-2.279104) = -0.227910$$
     $$\mathcal{L}_{\text{pol}} = -(-0.335373 - 0.255821 - 0.227910) = -(-0.819104) = \mathbf{0.819104}$$

   - **Value MSE Loss:**
     $$\mathcal{L}_{\text{val}} = (z_{\text{target}} - v)^2 = (1.000000 - 0.462117)^2 = (0.537883)^2 \approx \mathbf{0.289318}$$
   - **Total Loss:**
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{val}} + \mathcal{L}_{\text{pol}} = 0.289318 + 0.819104 = \mathbf{1.108422}$$

3. **Analytical Gradients w.r.t. Logits:**
   - **Policy Logit Gradient (Eq. 3.3):**
     $$\nabla_{\mathbf{z}_{\text{pol}}} \mathcal{L}_{\text{pol}} = \mathbf{p} - \boldsymbol{\pi}_{\text{target}}$$
     $$\frac{\partial \mathcal{L}}{\partial z_1} = 0.619338 - 0.700000 = \mathbf{-0.080662}$$
     $$\frac{\partial \mathcal{L}}{\partial z_2} = 0.278286 - 0.200000 = \mathbf{+0.078286}$$
     $$\frac{\partial \mathcal{L}}{\partial z_3} = 0.102376 - 0.100000 = \mathbf{+0.002376}$$
     *Interpretation:* The negative gradient on $z_1$ increases logit $z_1$ in gradient descent ($\Delta z_1 = -\eta \nabla z_1 > 0$), boosting $p_1$ toward the MCTS target $0.70$!

   - **Value Logit Gradient (Eq. 3.4):**
     $$\frac{\partial \mathcal{L}_{\text{val}}}{\partial z_{\text{val}}} = -2 (z_{\text{target}} - v)(1 - v^2)$$
     $$1 - v^2 = 1 - (0.462117)^2 = 1 - 0.213552 = 0.786448$$
     $$\frac{\partial \mathcal{L}_{\text{val}}}{\partial z_{\text{val}}} = -2 \times (0.537883) \times (0.786448) = -1.075766 \times 0.786448 \approx \mathbf{-0.846033}$$
     *Interpretation:* The negative gradient drives $z_{\text{val}}$ higher, pushing predicted value $v$ up from $0.462$ toward $1.000$. $\blacksquare$

---

### Illustration 6: Root Dirichlet Noise Injection & Opening Diversity
**Problem:**
During self-play, AlphaZero adds Dirichlet noise $\boldsymbol{\eta} \sim \operatorname{Dir}(\alpha)$ to the root node prior:
$$P'(s_0, a) = (1 - \epsilon) P(s_0, a) + \epsilon \eta_a, \quad \text{where } \epsilon = 0.25$$
Suppose for 3 opening moves, the neural network prior strongly prefers move 1:
$$\mathbf{P} = [0.90, \, 0.08, \, 0.02]$$
A sampled Dirichlet vector is $\boldsymbol{\eta} = [0.10, \, 0.60, \, 0.30]$.
1. Calculate the modified prior $\mathbf{P}'$.
2. Explain why this prevents opening book collapse during self-play.

**Solution:**
1. **Calculation of Blended Prior:**
   $$P'(s_0, a_1) = (1 - 0.25) \times 0.90 + 0.25 \times 0.10 = 0.75 \times 0.90 + 0.025 = 0.675 + 0.025 = \mathbf{0.7000}$$
   $$P'(s_0, a_2) = (1 - 0.25) \times 0.08 + 0.25 \times 0.60 = 0.75 \times 0.08 + 0.150 = 0.060 + 0.150 = \mathbf{0.2100}$$
   $$P'(s_0, a_3) = (1 - 0.25) \times 0.02 + 0.25 \times 0.30 = 0.75 \times 0.02 + 0.075 = 0.015 + 0.075 = \mathbf{0.0900}$$
   $$\sum_{a=1}^3 P'(s_0, a) = 0.7000 + 0.2100 + 0.0900 = 1.0000$$

2. **Preventing Opening Book Collapse:**
   Before noise injection, move $a_2$ had prior $0.08$, meaning it would almost never be selected in early simulations. After injection, its prior jumps to $0.2100$ ($> 2.6\times$ increase). This ensures alternative opening moves receive enough initial visits to be rigorously explored, guaranteeing the self-play agent does not overfit to a single repetitive line of play. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. MuZero: MCTS with Learned Latent Dynamics (Schrittwieser et al., Nature 2020)
AlphaZero required access to a perfect game simulator (rules of Go/Chess/Shogi). MuZero eliminates this requirement:
- **Learned Transition Model:** $h_{t+1} = g_\theta(h_t, a_t)$ predicts value, policy, and reward in latent space without knowing the game rules.
- **MCTS Over Latent States:** AlphaZero-style PUCT selection runs 800 simulations per move through the latent model, achieving superhuman performance on 57 Atari games (many without visual structure) plus all three perfect-information board games — with a **single unified architecture**.
- **EfficientZero (Ye et al., NeurIPS 2021):** Adds self-supervised consistency loss between the latent model's predicted states and true encoded observations, achieving human-level Atari performance with only 2 hours of real gameplay (100k steps).

### 2. LLM Reasoning as Tree Search: OpenAI o1, DeepSeek-R1, Q* (2024–2025)
MCTS is now a core component of frontier reasoning model inference:
- **Tree-of-Thoughts (Yao et al., 2023):** Frames LLM reasoning as BFS/DFS search over intermediate thought steps, with the LLM acting as both the transition model (generating next thoughts) and the value function (evaluating partial solution quality).
- **Process Reward Models (PRMs) as Value Functions:** OpenAI's o1 combines MCTS-style step-level search with a PRM trained on MATH datasets as the PUCT value oracle. Each reasoning step is a "move"; the PRM provides $Q(s, a)$ estimates for partial proof progress.
- **Monte Carlo Self-Consistency:** DeepSeek-R1's inference-time compute scaling samples $N \in [8, 64]$ independent reasoning paths and applies MCTS-inspired majority voting, achieving pass@1 performance that scales logarithmically with inference compute.

### 3. AlphaProof: MCTS for Formal Mathematical Theorem Proving (DeepMind, 2024)
AlphaProof combined AlphaZero-style MCTS with Lean 4 formal verification:
- **State Space:** Lean 4 proof states (partially proven theorems with open goals).
- **Action Space:** Available tactics (rewrite, apply, intro, ring\_nf, etc.).
- **Value Function:** A language model evaluating proof state quality — trained via RL on MCTS rollouts reaching QED or dead ends.
- **Result:** Solved 4 of 6 IMO 2024 problems, becoming the first AI system to reach silver-medal performance on the International Mathematical Olympiad.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical calculation of all PUCT scores, $Q$-values, and visit counts for Simulations 1–4 matching to $< 10^{-6}$.
2. **Complete Generic MCTS Node & Search Engine:**
   - Full implementation of selection, expansion, neural-net evaluation, and backup.
3. **Tic-Tac-Toe Self-Play Engine:**
   - Self-play match verification where AlphaZero MCTS plays against a random agent, demonstrating $100\%$ win/draw rate.
4. **Section 6 Solved Illustrations Verification:**
   - Numerical verification of Minimax-Negamax equivalence, 3-action PUCT selection, 3-level tree backpropagation with virtual loss, temperature annealing visit distributions ($\tau = 1.0, 0.5, 0.2$), dual-head loss gradients, and Dirichlet noise blending matching to $< 10^{-6}$.

See implementation in:
[`11_reinforcement_learning/code/23_mcts_and_alphazero.py`](./code/23_mcts_and_alphazero.py)
