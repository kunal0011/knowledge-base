# Module 11.7: On-Policy vs. Off-Policy Control: SARSA, Q-Learning & Expected SARSA

---

## 1. Intuition & 101 Motivation

In Chapter 11.6, we studied Temporal-Difference learning for **policy evaluation** (estimating $V^\pi(s)$ for a fixed policy $\pi$). To solve the full reinforcement learning problem—finding the **optimal policy** $\pi^*$ without a transition model—we must extend TD methods to estimate **action-values** $Q(s, a)$.

When moving from prediction to control, a fundamental fork in algorithm design emerges:
1. **On-Policy Control (SARSA):** The agent learns the value of the policy it is currently executing, *including its exploratory mistakes*. If the agent takes $\epsilon$-greedy random exploratory actions, SARSA evaluates the true return of that exploratory policy.
2. **Off-Policy Control (Q-Learning):** The agent learns the optimal policy $Q^*(s, a)$ directly, *regardless of how exploratory or suboptimal its behavior policy is*.
3. **Expected SARSA:** Bridges both paradigms by computing the exact expectation over all next actions under policy $\pi$, eliminating target sample variance.

```
       SARSA (On-Policy)              Q-Learning (Off-Policy)           Expected SARSA
      [Next action A' sampled]         [Max over all actions]       [Expectation under pi]
               (S, A)                          (S, A)                       (S, A)
                 |                               |                            |
                 R                               R                            R
                 |                               |                            |
                (S')                            (S')                         (S')
                 |                             / | \                        / | \
                (A')                         (a)(a)(a)                    (a)(a)(a)
                                            \___max___/                  \__weighted_/
```

---

## 2. Rigorous Mathematical Formulation

Let $\mathcal{S}$ be the finite state space, $\mathcal{A}$ the finite action space, $\gamma \in [0, 1)$ the discount factor, and $\alpha \in (0, 1]$ the step-size parameter.

### 2.1 SARSA: On-Policy TD Control

SARSA derives its name from the tuple of experience transitions:
$$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$$

The agent is in state $S_t$, takes action $A_t$, observes reward $R_{t+1}$ and next state $S_{t+1}$, and then samples its **actual next action** $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$ from its behavior policy (e.g., $\epsilon$-greedy).

#### Update Equation:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

The TD error is:
$$\delta_t^{\text{SARSA}} \triangleq R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$

#### Convergence Condition (GLIE: Greedy in the Limit with Infinite Exploration):
If all state-action pairs $(s, a)$ are visited infinitely often, and the exploration parameter satisfies $\epsilon_t \to 0$ as $t \to \infty$, SARSA converges almost surely to the optimal action-value function $Q^*$ and optimal policy $\pi^*$.

---

### 2.2 Q-Learning: Off-Policy TD Control (Watkins, 1989)

Q-Learning decouples the **behavior policy** $b(a \mid s)$ (which generates transitions) from the **target policy** $\pi(a \mid s) = \arg \max_a Q(s, a)$ (which is evaluated and optimized).

#### Update Equation:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

The TD error is:
$$\delta_t^{\text{Q}} \triangleq R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)$$

#### Theorem: Convergence of Q-Learning (Watkins & Dayan, 1992)
Let the MDP have finite state and action spaces and bounded rewards. Suppose all state-action pairs $(s, a)$ are visited infinitely often, and the step sizes $\alpha_t(s, a)$ satisfy the Robbins-Monro conditions:
$$\sum_{t=1}^\infty \alpha_t(s, a) = \infty \quad \text{and} \quad \sum_{t=1}^\infty \alpha_t^2(s, a) < \infty \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A}$$
Then, regardless of the behavior policy $b$ used to generate actions (provided $b(a \mid s) > 0$ for all $(s, a)$), $Q_t(s, a)$ converges almost surely to the unique optimal action-value function $Q^*(s, a)$:
$$\mathbb{P} \left( \lim_{t \to \infty} Q_t(s, a) = Q^*(s, a) \right) = 1 \quad \forall (s, a)$$

#### Proof Outline (Bellman Optimality Contraction)
The expected update operator is the Bellman optimality operator $\mathcal{T}^*$:
$$(\mathcal{T}^* Q)(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \max_{a' \in \mathcal{A}} Q(s', a')$$
In Chapter 11.4, we proved that $\mathcal{T}^*$ is a $\gamma$-contraction in the supremum norm:
$$\|\mathcal{T}^* Q_1 - \mathcal{T}^* Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
By the stochastic approximation theorem of Jaakkola, Jordan, and Singh (1994), any asynchronous iterative process governed by a contraction mapping with Robbins-Monro step sizes converges almost surely to its unique fixed point $Q^*$. $\blacksquare$

---

### 2.3 Expected SARSA

Instead of sampling a single next action $A_{t+1}$ (like SARSA) or taking the hard maximum (like Q-learning), **Expected SARSA** computes the expectation over all possible next actions under the target policy $\pi$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

- If $\pi$ is greedy ($\pi(a \mid s) = 1$ for $a = \arg\max Q$, and $0$ otherwise), Expected SARSA becomes **identical to Q-Learning**.
- If $\pi$ is $\epsilon$-greedy, Expected SARSA accounts for exploration probability without introducing the sampling variance of choosing a single $A_{t+1}$.

---

### 2.4 Maximization Bias and Double Q-Learning (van Hasselt, 2010)

Both Q-Learning and Expected SARSA (with greedy targets) employ the maximum operator:
$$\text{Target} = R + \gamma \max_{a'} Q(S', a')$$

#### The Maximization Bias Problem
Let $X_1, X_2, \dots, X_n$ be independent random variables with true means $\mu_i = \mathbb{E}[X_i]$. If each estimate $\hat{X}_i$ is an unbiased estimator of $\mu_i$ ($\mathbb{E}[\hat{X}_i] = \mu_i$), then:
$$\mathbb{E} \left[ \max_{i} \hat{X}_i \right] \ge \max_i \mathbb{E}[\hat{X}_i] = \max_i \mu_i$$
with strict inequality whenever the distributions have overlapping support.

Because $\max$ is a convex function, **Jensen's inequality** dictates that taking the maximum over noisy estimates produces a systematic positive bias (**maximization bias**), leading Q-learning to severely overestimate values!

#### Double Q-Learning Solution
To eliminate this bias, van Hasselt proposed decoupling **action selection** from **action evaluation** using two independent value tables, $Q_A$ and $Q_B$.

With probability $0.5$, update $Q_A$:
$$A^* = \arg \max_a Q_A(S_{t+1}, a) \quad \text{(Selection)}$$
$$Q_A(S_t, A_t) \leftarrow Q_A(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_B(S_{t+1}, A^*) - Q_A(S_t, A_t) \right] \quad \text{(Evaluation)}$$

With probability $0.5$, update $Q_B$ symmetrically:
$$B^* = \arg \max_a Q_B(S_{t+1}, a)$$
$$Q_B(S_t, A_t) \leftarrow Q_B(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_A(S_{t+1}, B^*) - Q_B(S_t, A_t) \right]$$

Because $Q_B(S_{t+1}, A^*)$ is an unbiased estimate of the value of $A^*$, the expected update has **zero maximization bias**:
$$\mathbb{E} \left[ Q_B(S_{t+1}, A^*) \mid A^* \right] = Q^*(S_{t+1}, A^*)$$

---

## 3. Geometric & Physical Interpretation: The Cliff Walking Dilemma

Consider the canonical **Cliff Walking** gridworld:

```
+---+---+---+---+---+---+---+---+---+---+---+---+
| S |   |   |   |   |   |   |   |   |   |   | G |
+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+
| S | C | L | I | F | F |   | C | L | I | F | G |
+---+---+---+---+---+---+---+---+---+---+---+---+
```
- Step reward: $-1.0$ for normal transitions.
- Falling into the Cliff: $-100.0$ and resets agent to Start $S$.

### Geometric Separation of Policies:
- **Q-Learning** learns the true optimal path $Q^*$: the path that walks right along the cliff edge. It assumes greedy execution. However, during training with $\epsilon$-greedy exploration ($\epsilon = 0.1$), the agent frequently slips off the cliff due to random exploration, suffering poor online performance (average reward $\approx -45$).
- **SARSA** is on-policy: it incorporates the agent's actual $\epsilon$-greedy execution into its value estimates. It learns that being near the cliff is hazardous due to the $10\%$ probability of accidental suicide. Geometrically, SARSA's trajectory bends away from the cliff edge, discovering the **safe path** through the upper perimeter (average reward $\approx -17$).

```
^ Distance from Cliff
|
|   SARSA Safe Path (On-Policy: anticipates random exploration)
|   ------------------------------------------------------------> Goal
|
|   Q-Learning Optimal Path (Off-Policy: ignores exploration risk)
|   ............................................................> Goal
|   [========================= CLIFF =========================]
+------------------------------------------------------------------>
```

---

## 4. Real-World Analogy: Formula 1 Racing vs. Autonomous Safety

- **Q-Learning is the Formula 1 Telemetry Analysis:**
  It calculates the absolute fastest possible lap time assuming the driver hits the racing line with mathematical perfection. It ignores human error, fatigue, or wet patches off-line.
- **SARSA is the Pragmatic Race Strategy:**
  It observes how the human driver actually drives under fatigue and weather. If the driver frequently locks the brakes into Turn 3 when trying to take the aggressive curb, SARSA advises: "Brake 5 meters earlier; your execution isn't perfect, and going off-track costs 30 seconds."

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 2-State, 2-Action MDP
Let us compare **SARSA**, **Q-Learning**, and **Expected SARSA** on the exact same transition!

**States:** $S_1$ and $S_2$
**Actions:** $a_1$ (Action 1) and $a_2$ (Action 2)
**Hyperparameters:**
- Discount factor: $\gamma = 0.8000$
- Learning rate: $\alpha = 0.5000$
- Exploration rate: $\epsilon = 0.2000$

**Initial Action-Value Table:**
$$Q(S_1, a_1) = 2.0000, \quad Q(S_1, a_2) = 1.0000$$
$$Q(S_2, a_1) = 6.0000, \quad Q(S_2, a_2) = 10.0000$$

**Transition Observed:**
At time $t$, agent is in $S_1$ and takes action $A_t = a_1$.
- Immediate reward observed: $R_{t+1} = 4.0000$
- Next state reached: $S_{t+1} = S_2$
- Next action sampled from behavior policy: $A_{t+1} = a_1$ (an exploratory move, since $Q(S_2, a_1) < Q(S_2, a_2)$!).

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Walkthrough Value |
| :--- | :--- | :--- |
| $S_t, A_t$ | Current State-Action Pair | $S_1, a_1$ |
| $R_{t+1}$ | Observed Reward | $4.0000$ |
| $S_{t+1}$ | Successor State | $S_2$ |
| $A_{t+1}$ | Sampled Next Action | $a_1$ (suboptimal choice) |
| $Q(S_2, a_1)$ | Action-Value for $a_1$ at $S_2$ | $6.0000$ |
| $Q(S_2, a_2)$ | Action-Value for $a_2$ at $S_2$ | $10.0000$ (greedy choice) |
| $\pi(a \mid S_2)$ | $\epsilon$-greedy target policy probabilities | $\pi(a_2) = 1 - \frac{0.2}{2} = 0.90$, $\pi(a_1) = \frac{0.2}{2} = 0.10$ |

---

### 5.3 Method 1: SARSA Update Hand Arithmetic

SARSA targets the **sampled** action $A_{t+1} = a_1$:
$$\text{Target}_{\text{SARSA}} = R_{t+1} + \gamma Q(S_2, A_{t+1}) = 4.0000 + 0.8000 \times Q(S_2, a_1)$$
$$= 4.0000 + 0.8000 \times 6.0000 = 4.0000 + 4.8000 = \mathbf{8.8000}$$

TD Error:
$$\delta_{\text{SARSA}} = \text{Target}_{\text{SARSA}} - Q(S_1, a_1) = 8.8000 - 2.0000 = \mathbf{6.8000}$$

Updated Value:
$$Q_{\text{new}}^{\text{SARSA}}(S_1, a_1) = Q(S_1, a_1) + \alpha \delta_{\text{SARSA}} = 2.0000 + 0.5000 \times 6.8000 = 2.0000 + 3.4000 = \mathbf{5.4000}$$

---

### 5.4 Method 2: Q-Learning Update Hand Arithmetic

Q-Learning targets the **maximum** over all actions at $S_2$:
$$\max_{a'} Q(S_2, a') = \max(Q(S_2, a_1), Q(S_2, a_2)) = \max(6.0000, 10.0000) = \mathbf{10.0000}$$

$$\text{Target}_{\text{Q}} = R_{t+1} + \gamma \max_{a'} Q(S_2, a') = 4.0000 + 0.8000 \times 10.0000 = 4.0000 + 8.0000 = \mathbf{12.0000}$$

TD Error:
$$\delta_{\text{Q}} = \text{Target}_{\text{Q}} - Q(S_1, a_1) = 12.0000 - 2.0000 = \mathbf{10.0000}$$

Updated Value:
$$Q_{\text{new}}^{\text{Q}}(S_1, a_1) = Q(S_1, a_1) + \alpha \delta_{\text{Q}} = 2.0000 + 0.5000 \times 10.0000 = 2.0000 + 5.0000 = \mathbf{7.0000}$$

---

### 5.5 Method 3: Expected SARSA Hand Arithmetic

Under $\epsilon$-greedy exploration with $\epsilon = 0.20$ and $|\mathcal{A}| = 2$:
- Greedy action is $a_2$ (since $Q(S_2, a_2) = 10 > 6$):
  $$\pi(a_2 \mid S_2) = 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} = 1 - 0.20 + 0.10 = \mathbf{0.9000}$$
- Non-greedy action is $a_1$:
  $$\pi(a_1 \mid S_2) = \frac{\epsilon}{|\mathcal{A}|} = \frac{0.20}{2} = \mathbf{0.1000}$$

Expected Next-State Value:
$$\mathbb{E}_{\pi}[Q(S_2, \cdot)] = \pi(a_1 \mid S_2) Q(S_2, a_1) + \pi(a_2 \mid S_2) Q(S_2, a_2)$$
$$= (0.1000 \times 6.0000) + (0.9000 \times 10.0000) = 0.6000 + 9.0000 = \mathbf{9.6000}$$

TD Target:
$$\text{Target}_{\text{Exp}} = R_{t+1} + \gamma \mathbb{E}_{\pi}[Q(S_2, \cdot)] = 4.0000 + 0.8000 \times 9.6000 = 4.0000 + 7.6800 = \mathbf{11.6800}$$

TD Error:
$$\delta_{\text{Exp}} = \text{Target}_{\text{Exp}} - Q(S_1, a_1) = 11.6800 - 2.0000 = \mathbf{9.6800}$$

Updated Value:
$$Q_{\text{new}}^{\text{Exp}}(S_1, a_1) = Q(S_1, a_1) + \alpha \delta_{\text{Exp}} = 2.0000 + 0.5000 \times 9.6800 = 2.0000 + 4.8400 = \mathbf{6.8400}$$

---

### 5.6 Summary Comparison Visual Grid

| Method | Next-Action Philosophy | Target Formula | Target Value | TD Error $\delta$ | Updated $Q(S_1, a_1)$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SARSA** | Sampled $A_{t+1} = a_1$ (On-policy) | $4.0 + 0.8 \times 6.0$ | $\mathbf{8.8000}$ | $\mathbf{6.8000}$ | $\mathbf{5.4000}$ |
| **Q-Learning** | Maximum $\max_a Q$ (Off-policy) | $4.0 + 0.8 \times 10.0$ | $\mathbf{12.0000}$ | $\mathbf{10.0000}$ | $\mathbf{7.0000}$ |
| **Expected SARSA** | Expectation $\sum \pi(a) Q$ | $4.0 + 0.8 \times 9.6$ | $\mathbf{11.6800}$ | $\mathbf{9.6800}$ | $\mathbf{6.8400}$ |

Every single arithmetic operation is crystal clear, rigorous, and exact.

---

## 6. Solved Illustrations

### Illustration 1: Maximization Bias (Sutton Example 6.7)
**Problem:**
Consider a 2-state MDP: State $A$ and State $B$.
- From $A$, action `right` terminates with reward $0$.
- From $A$, action `left` transitions to $B$ with reward $0$.
- From $B$, there are $10$ available actions, each of which terminates with a stochastic reward drawn independently from a Gaussian distribution:
  $$R \sim \mathcal{N}(-0.1, 1.0)$$
Thus, the true expected value of entering $B$ is $\mathbb{E}[R] = -0.1 < 0$. The true optimal policy from $A$ is to choose `right` (return $0 > -0.1$).

Show why Q-Learning initially prefers `left`, and prove that Double Q-Learning correctly identifies `right`.

**Solution:**
1. **Under Standard Q-Learning:**
   At state $B$, the target for state $A$ is:
   $$\text{Target}(A, \text{left}) = 0 + \gamma \max_{a \in \{1,\dots,10\}} Q(B, a)$$
   Even though each $Q(B, a)$ has true mean $-0.1$, the maximum of 10 noisy independent estimates $\max_{i=1}^{10} X_i$ has expected value:
   $$\mathbb{E} \left[ \max_{1 \le i \le 10} \mathcal{N}(-0.1, 1.0) \right] \approx -0.1 + \frac{\sqrt{2 \ln(10)}}{\sqrt{\pi}} \approx +1.43 > 0!$$
   Consequently, Q-learning creates the illusion that moving to $B$ has high positive value, causing the agent to choose `left` up to $70\text{--}80\%$ of the time!

2. **Under Double Q-Learning:**
   Action selection and evaluation are decoupled across $Q_1$ and $Q_2$:
   $$a^* = \arg \max_a Q_1(B, a)$$
   $$\text{Target} = Q_2(B, a^*)$$
   Since $Q_2$ is independent of the noise that caused $Q_1(B, a^*)$ to be high:
   $$\mathbb{E} \left[ Q_2(B, a^*) \mid a^* \right] = \mathbb{E}[R] = -0.1 < 0$$
   Double Q-Learning produces an expected target of $-0.1$, so $Q(A, \text{left}) < Q(A, \text{right}) = 0$. The agent immediately learns the correct optimal policy! $\blacksquare$

---

## 7. Deep Learning Connection & Application

### 1. Deep Q-Networks (DQN - Mnih et al., Nature 2015)
DQN is the deep function approximation form of Watkins Q-learning:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
where experience replay decorrelates samples and target network $\theta^-$ stabilizes the moving TD target.

### 2. Double DQN (DDQN - van Hasselt et al., AAAI 2016)
To eliminate maximization bias in deep networks, Double DQN uses the online network $\theta$ to **select** the greedy action, but uses the target network $\theta^-$ to **evaluate** its value:
$$a^* = \arg \max_{a'} Q(s', a'; \theta)$$
$$Y_t^{\text{DoubleQ}} = r + \gamma Q(s', a^*; \theta^-)$$
This simple one-line modification eliminated massive overestimation spikes in Atari 2600 games and drastically improved sample efficiency.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of Part 5 hand calculations for SARSA ($5.4000$), Q-Learning ($7.0000$), and Expected SARSA ($6.8400$) to $< 10^{-14}$ machine precision.
2. The Cliff Walking benchmark: Running SARSA vs. Q-Learning vs. Expected SARSA, verifying that SARSA discovers the safe path while Q-learning takes the cliff edge.
3. Sutton's Maximization Bias experiment: Demonstrating Q-learning's severe overestimation on the roulette state vs. Double Q-learning's unbiased estimation.

See implementation in:
[`11_reinforcement_learning/code/07_sarsa_and_q_learning.py`](./code/07_sarsa_and_q_learning.py)
