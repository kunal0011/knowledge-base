# Module 11.21: Model-Based Foundations & Dyna-Q

---

## 1. Intuition & 101 Motivation

Throughout Sub-Modules 2 through 5, we focused exclusively on **model-free reinforcement learning** (Q-learning, Actor-Critic, PPO, SAC). Model-free agents learn policy or value functions directly from raw environmental experience through pure trial and error. 

While model-free RL is simple and avoids assumptions about the environment, it is notoriously **sample-inefficient**:
- A model-free agent often requires tens of millions of frames in Atari (equivalent to weeks of continuous gameplay).
- In physical robotics, running 10 million real-world trial steps wears out electric motors, overheats gearboxes, and damages mechanical linkages.

**Model-Based Reinforcement Learning** introduces a fundamentally different philosophy:
1. The agent uses its real-world experience to learn an **internal predictive model of the environment**:
   $$\hat{\mathcal{P}}(s' \mid s, a) \approx \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a), \quad \hat{\mathcal{R}}(s, a) \approx \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$$
2. The agent then uses this internal simulator to **plan**—mentally simulating thousands of "hallucinated" trajectories in computer memory without touching the real physical world!

In 1990, Richard Sutton introduced the **Dyna architecture**, which seamlessly unifies model-free learning, model learning, and model-based planning into a single integrated loop.

```
+-----------------------------------------------------------------------------------------+
|                                  THE DYNA ARCHITECTURE                                  |
|                                                                                         |
|                               +-------------------------+                               |
|                               |       ENVIRONMENT       |                               |
|                               +-------------------------+                               |
|                                   ^ Action        | Real Experience                     |
|                                   |               v (S, A, R, S')                       |
|                             +-----+---------------+-----+                               |
|                             |                           |                               |
|                             v                           v                               |
|              +-----------------------------+     +-----------------------------+        |
|              |      DIRECT RL UPDATE       |     |        MODEL LEARNING       |        |
|              |  Q(S, A) <- Direct Sample   |     |  Model(S, A) <- (R, S')     |        |
|              +-----------------------------+     +-----------------------------+        |
|                             ^                                   |                       |
|                             | Simulated Experience              v                       |
|                             | (S_k, A_k, R_k, S'_k)      +-----------------------------+        |
|                             +----------------------------|       PLANNING ENGINE       |        |
|                                                          |  (N Hallucinated Steps!)    |        |
|                                                          +-----------------------------+        |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Environmental Model

An environment model consists of two components:
1. **Transition Model:** $\hat{\mathcal{P}}(s' \mid s, a) \approx \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$
2. **Reward Model:** $\hat{\mathcal{R}}(s, a) \approx \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$

In a deterministic environment, the model is simply a pair of functions:
$$\hat{s}' = f_{\boldsymbol{\theta}}(s, a), \quad \hat{r} = g_{\boldsymbol{\psi}}(s, a)$$
In a tabular setting, the model is a dictionary lookup table:
$$\text{Model}(s, a) \leftarrow (r, s')$$

Model learning is standard supervised regression/classification:
$$\min_{\boldsymbol{\theta}, \boldsymbol{\psi}} \sum_{i=1}^M \left[ \mathcal{L}_{\text{trans}}(s_{i+1}, f_{\boldsymbol{\theta}}(s_i, a_i)) + \mathcal{L}_{\text{rew}}(r_{i+1}, g_{\boldsymbol{\psi}}(s_i, a_i)) \right]$$

---

### 2.2 The Tabular Dyna-Q Algorithm

At each real-world interaction:
1. **Act:** In current state $S$, choose action $A \sim \epsilon\text{-greedy}(Q(S, \cdot))$.
2. **Execute:** Execute $A$ in the real environment; observe real reward $R$ and real next state $S'$.
3. **Direct RL:** Update $Q(S, A)$ using the 1-step Q-learning rule:
   $$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a'} Q(S', a') - Q(S, A) \right]$$
4. **Model Learning:** Update the internal model with the observed transition:
   $$\text{Model}(S, A) \leftarrow (R, S')$$
5. **Planning Phase (Mental Rehearsal):**
   Repeat $N$ times:
   - Sample a previously visited state $S_k$ uniformly from all states recorded in $\text{Model}$.
   - Sample an action $A_k$ previously taken in state $S_k$.
   - Query the model for predicted reward and next state:
     $$(R_k, S'_k) \leftarrow \text{Model}(S_k, A_k)$$
   - Perform a Q-learning update on the **hallucinated transition**:
     $$Q(S_k, A_k) \leftarrow Q(S_k, A_k) + \alpha \left[ R_k + \gamma \max_{a'} Q(S'_k, a') - Q(S_k, A_k) \right]$$

If $N = 0$, Dyna-Q reduces to standard model-free Q-learning.
If $N = 50$, the agent performs **50 internal planning updates** for every single real step taken in the physical world!

---

### 2.3 Non-Stationary Environments & Dyna-Q+

If the environment changes dynamically (e.g., a shortcut opens or a corridor is blocked), a standard Dyna model will continue hallucinating old, outdated transitions.

To encourage the agent to re-explore previously learned transitions that haven't been tried recently, **Dyna-Q+** adds an **exploration bonus** proportional to the square root of elapsed time:

$$R_{\text{sim}} \triangleq R + \kappa \sqrt{\tau(s, a)}$$
where:
- $\tau(s, a)$ is the number of real time steps that have passed since action $a$ was last executed in state $s$ in the physical world.
- $\kappa > 0$ is an exploration bonus scale (typically $\kappa = 0.001$).

During the planning phase, the agent updates using the bonus-augmented reward:
$$Q(S_k, A_k) \leftarrow Q(S_k, A_k) + \alpha \left[ R_k + \kappa \sqrt{\tau(S_k, A_k)} + \gamma \max_{a'} Q(S'_k, a') - Q(S_k, A_k) \right]$$
This guarantees that transitions neglected for a long time develop artificially high simulated value, prompting the agent to physically revisit them to verify whether the world has changed!

---

### 2.4 Compounding Errors & The Limits of Model-Based Rollouts

Why can't we set the planning horizon to infinity ($N \to \infty$)?
Let $\hat{\mathcal{P}}$ be an imperfect learned model with single-step error bounded by total variation distance $\epsilon_m$:
$$\max_{s, a} \|\mathcal{P}(\cdot \mid s, a) - \hat{\mathcal{P}}(\cdot \mid s, a)\|_{\text{TV}} \le \epsilon_m$$

#### Theorem: Compounding Model Error (Talvitie, 2014; Janner et al., 2019)
The total variation error between the true state distribution $d_H^\pi$ and the model-generated rollout distribution $\hat{d}_H^\pi$ at horizon $H$ is bounded by:

$$\|d_H^\pi - \hat{d}_H^\pi\|_{\text{TV}} \le \sum_{t=1}^H t \cdot \epsilon_m = \mathcal{O}(H^2 \epsilon_m)$$

When rolling out an autoregressive model step-by-step ($s_0 \to \hat{s}_1 \to \hat{s}_2 \to \dots \to \hat{s}_H$), errors in early steps feed into subsequent steps as out-of-distribution inputs. Small single-step errors compound **quadratically**, leading the agent to hallucinate impossible physics and learn pathological policies (**model exploitation**).

This explains why tabular Dyna-Q restricts its planning queries to **1-step lookahead from previously visited real states** ($H = 1$), avoiding the compounding error trap!

---

## 3. Geometric & Physical Interpretation: Bellman Flow Along the Empirical Graph

In state-action space $\mathcal{S} \times \mathcal{A}$:
- Real physical transitions carve out a directed multigraph $G = (V, E)$.
- In model-free Q-learning, reward information flows backward along this graph strictly **one edge per episode**. To propagate reward from a goal state across a 10-step corridor to the starting state requires taking at least 10 full physical episodes.
- In Dyna-Q, the model stores the entire graph $G$ in memory. During the planning loop, the Bellman operator acts as an **information diffusion process**: it rapidly propagates value backwards across all edges simultaneously while the agent stands completely still in the physical world!

```
   REAL ENVIRONMENT (1 step/time)                DYNA-Q PLANNING (Fast diffusion)
        S1 ---> S2 ---> S3                              S1 <==== S2 <==== S3 (Goal)
                                                        Values propagate backward
                                                        through all stored edges
                                                        during mental rehearsal!
```

---

## 4. Real-World Analogy: The Chess Grandmaster vs. The Novice

- **The Model-Free Novice:** The novice must physically sit at the board, make a move, play through 40 moves to checkmate, lose, and only then realize that move 3 was bad. They need to play 100,000 physical games to discover opening theory.
- **The Dyna Grandmaster:** The grandmaster plays one physical move on the board (real step). Between moves, while the opponent's clock is ticking, the grandmaster closes their eyes and mentally simulates 100 different branching move sequences in their head ($N = 100$ planning steps), updating their evaluation of the board state before touching another physical piece.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 3-State Maze
Consider a 3-state deterministic environment:
$$S_1 \xrightarrow{a_0} S_2 \xrightarrow{a_1} S_3 \text{ (Terminal)}$$

- States: $S_1, S_2, S_3$.
- Actions: $a_0, a_1$.
- Discount factor: $\gamma = 0.9000$.
- Learning rate: $\alpha = 0.5000$.
- Initial Q-table: all zeros ($Q(s, a) = 0.0000$ everywhere).
- Model starts empty.

We will trace two sequential real-world transitions with **$N = 1$ planning step** per real step.

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Value / Meaning |
| :--- | :--- | :--- |
| $S, A$ | Real Step State & Action | Transition experienced in the real world |
| $R, S'$ | Real Step Reward & Next State | Observed outcome from the environment |
| $\text{Model}(S, A)$ | Internal Memory Dictionary | Stores transition mapping $(R, S')$ |
| $S_k, A_k$ | Planning Sample State & Action | Transition drawn from memory for mental rehearsal |
| Target | Q-learning Target | $R + \gamma \max_{a'} Q(S', a')$ |
| $Q_{\text{new}}(S, A)$ | Updated Action-Value | $Q(S, A) + \alpha (\text{Target} - Q(S, A))$ |

---

### 5.3 Step-by-Step Hand Calculations: Transition 1

#### Real Transition 1:
The agent starts at $S_1$, takes action $a_0$, receives reward $R = 0.0000$, and arrives at $S_2$.
1. **Direct Q-Update:**
   $$\text{Target} = R + \gamma \max_a Q(S_2, a) = 0.0000 + 0.9 \times 0.0000 = 0.0000$$
   $$Q(S_1, a_0) = 0.0000 + 0.5000 \times (0.0000 - 0.0000) = \mathbf{0.0000}$$
2. **Model Learning:**
   The agent records this experience in its model:
   $$\text{Model}(S_1, a_0) \leftarrow (R = 0.0000, S' = S_2)$$
3. **Planning Step 1 ($N = 1$):**
   - Sample previously visited state-action pair: only $(S_1, a_0)$ exists in memory.
   - Query Model: returns $R_k = 0.0000, S'_k = S_2$.
   - Planning Q-Update:
     $$\text{Target} = 0.0000 + 0.9 \times 0.0000 = 0.0000 \implies Q(S_1, a_0) = \mathbf{0.0000}$$

At the end of Transition 1: $Q(S_1, a_0) = 0.0000$.

---

### 5.4 Step-by-Step Hand Calculations: Transition 2 (The Breakthrough!)

#### Real Transition 2:
The agent is now at $S_2$, takes action $a_1$, receives a massive reward $R = 10.0000$, and reaches terminal state $S_3$.
1. **Direct Q-Update for $(S_2, a_1)$:**
   Since $S_3$ is terminal, $\max_a Q(S_3, a) \equiv 0.0000$:
   $$\text{Target} = R + \gamma \times 0.0000 = 10.0000 + 0.0000 = \mathbf{10.0000}$$
   $$Q(S_2, a_1) = Q(S_2, a_1) + \alpha \left[ \text{Target} - Q(S_2, a_1) \right] = 0.0000 + 0.5000 \times (10.0000 - 0.0000) = \mathbf{5.0000}$$
2. **Model Learning:**
   The agent records:
   $$\text{Model}(S_2, a_1) \leftarrow (R = 10.0000, S' = S_3)$$
   Model memory now contains two transitions: $\{(S_1, a_0), (S_2, a_1)\}$.

3. **Planning Step 2 ($N = 1$):**
   Suppose the planning phase randomly draws $(S_k, A_k) = (S_1, a_0)$ from memory!
   - Query Model: returns $R_k = 0.0000, S'_k = S_2$.
   - Now look at the target for $(S_1, a_0)$:
     $$\text{Target} = R_k + \gamma \max_a Q(S_2, a) = 0.0000 + 0.9000 \times \max(Q(S_2, a_0), Q(S_2, a_1))$$
     Since $Q(S_2, a_1) = \mathbf{5.0000}$:
     $$\text{Target} = 0.0000 + 0.9000 \times 5.0000 = 0.0000 + 4.5000 = \mathbf{4.5000}$$
   - Update $Q(S_1, a_0)$ in memory:
     $$Q_{\text{new}}(S_1, a_0) = Q(S_1, a_0) + \alpha \left[ \text{Target} - Q(S_1, a_0) \right]$$
     $$= 0.0000 + 0.5000 \times (4.5000 - 0.0000) = \mathbf{2.2500}$$

---

### 5.5 Profound Insight from Hand Calculation

Look at what just happened:
- In standard **model-free Q-learning**, $Q(S_1, a_0)$ would still be **$0.0000$** at the end of Episode 1. The agent would have to start a brand new episode, physically walk to $S_1$, take $a_0$, and only then update $Q(S_1, a_0)$.
- In **Dyna-Q**, $Q(S_1, a_0)$ jumped to **$2.2500$ immediately during mental planning**! The reward at $S_3$ traveled backward to $S_1$ within Episode 1 without the agent taking a single physical step!

---

### 5.6 Summary Visual Grid: Dyna-Q Transition Ledger

| Event | Phase | State-Action $(S, A)$ | Reward $R$ | Next $S'$ | Target Formula | Target Value | Updated $Q(S, A)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 1** | Real | $(S_1, a_0)$ | $0.0$ | $S_2$ | $0 + 0.9 \times 0$ | $0.0000$ | $Q(S_1, a_0) = 0.0000$ |
| **Step 1** | Planning | $(S_1, a_0)$ | $0.0$ | $S_2$ | $0 + 0.9 \times 0$ | $0.0000$ | $Q(S_1, a_0) = 0.0000$ |
| **Step 2** | Real | $(S_2, a_1)$ | $10.0$ | $S_3$ (End) | $10.0 + 0$ | $\mathbf{10.0000}$ | $Q(S_2, a_1) = \mathbf{5.0000}$ |
| **Step 2** | Planning | $(S_1, a_0)$ | $0.0$ | $S_2$ | $0 + 0.9 \times 5.0$ | $\mathbf{4.5000}$ | $Q(S_1, a_0) = \mathbf{2.2500}$ $\uparrow$ |

---

## 6. Solved Illustrations

### Illustration 1: Dyna-Q vs. Dyna-Q+ in a Changing Environment
**Problem (Sutton's Shortcut Maze):**
Suppose an agent learns the optimal path in a maze. At step 1000, a short wall is removed, creating a massive shortcut that was previously blocked.
Compare the behavior of:
1. Standard Dyna-Q
2. Dyna-Q+ with exploration bonus $\kappa \sqrt{\tau}$

**Solution:**
- **Standard Dyna-Q:** The agent's model remembers that the shortcut wall produces reward $-1$ and blocks movement. Because the agent's greedy policy exploits the existing longer path, it never physically enters the shortcut wall. The model is never updated, and the agent **never discovers the shortcut**.
- **Dyna-Q+:** As thousands of steps pass, the elapsed time $\tau(s_{\text{wall}}, a)$ since trying the wall grows large ($\tau = 2000 \implies \sqrt{\tau} \approx 44.7$). The simulated planning reward becomes positive:
  $$R_{\text{sim}} = -1.0 + \kappa \sqrt{\tau} > 0$$
  During mental planning, the shortcut suddenly appears enticing! The agent is driven to physically re-try the wall, discovers the shortcut is open, updates its model, and permanently shifts to the faster path! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **World Models (Ha & Schmidhuber, 2018):** Trained a VAE + Recurrent Neural Network (MDN-RNN) to model CarRacing pixel dynamics, allowing a small controller to learn entirely inside "dreams" (hallucinated latent rollouts).
- **DreamerV1 - V3 (Hafner et al., 2020-2023):** Modern state-of-the-art model-based RL. Uses a Recurrent State-Space Model (RSSM) to learn Minecraft from scratch (obtaining diamonds without human data) solely through latent model planning!
- **MuZero (Silver et al., Nature 2020):** Combines model learning with Monte Carlo Tree Search (MCTS), learning an implicit latent transition model without reconstructing images.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of Part 5 Dyna-Q hand calculations:
   - Direct Q-update for $(S_2, a_1) = 5.0000$
   - Planning Q-update for $(S_1, a_0) = 2.2500$ matching machine precision to $< 10^{-14}$.
2. Complete Dyna-Q benchmark on a Gridworld:
   - Compares planning steps $N = 0$ (model-free Q-learning), $N = 5$, and $N = 50$, demonstrating massive sample efficiency improvements.
3. Dyna-Q+ exploration bonus simulation on a non-stationary blocking maze.

See implementation in:
[`11_reinforcement_learning/code/21_model_based_dyna_q.py`](./code/21_model_based_dyna_q.py)
