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
$$\hat{s}' = f_{\theta}(s, a), \quad \hat{r} = g_{\psi}(s, a)$$
In a tabular setting, the model is a dictionary lookup table:
$$\text{Model}(s, a) \leftarrow (r, s')$$

Model learning is standard supervised regression/classification:
$$\min_{\theta, \psi} \sum_{i=1}^M \left[ \mathcal{L}_{\text{trans}}(s_{i+1}, f_{\theta}(s_i, a_i)) + \mathcal{L}_{\text{rew}}(r_{i+1}, g_{\psi}(s_i, a_i)) \right]$$

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

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.21.1: Dyna Architecture Planning Update Convergence Theorem

```
====================================================================================================
DERIVATION 11.21.1: Dyna Planning Contraction & Almost Sure Empirical Convergence
====================================================================================================
Problem Statement:
Prove that for any finite Markov Decision Process ℳ = (𝒮, 𝒜, 𝒫, ℛ, γ), the empirical Bellman
optimality operator 𝒯̂_t^* induced by the empirical maximum likelihood transition model 𝒫̂_t and
reward model ℛ̂_t is a strict γ-contraction in the L_∞ norm. Furthermore, prove that under
standard asynchronous stochastic approximation conditions (Robbins-Monro step sizes and persistent
sampling of visited transitions), the planning sequence converges almost surely to the empirical
fixed point Q^*_{ℳ̂_t}, and establish the perturbation bound:
    ‖Q^*_{ℳ̂_t} - Q^*‖_∞ ≤ (1 / (1 - γ)) ‖ℛ̂_t - ℛ‖_∞ + (γ / (1 - γ)^2) ‖𝒫̂_t - 𝒫‖_∞ R_max
guaranteeing almost sure convergence to the true optimal value function Q^* as empirical visits N_t(s,a) → ∞.
====================================================================================================
```

**1. Problem Statement & Mathematical Goal:**
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ be a stationary finite Markov Decision Process, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{P}(s' \mid s, a)$ is the transition distribution, $\mathcal{R}(s, a) \in [-R_{\max}, R_{\max}]$ is the expected immediate reward, and $\gamma \in [0, 1)$ is the discount factor.

At physical time step $t$, the agent maintains an empirical maximum likelihood model $\hat{\mathcal{M}}_t = (\mathcal{S}, \mathcal{A}, \hat{\mathcal{P}}_t, \hat{\mathcal{R}}_t, \gamma)$, constructed from sample transition counts $N_t(s, a, s')$ and visitation counts $N_t(s, a) = \sum_{s'} N_t(s, a, s')$:
$$\hat{\mathcal{P}}_t(s' \mid s, a) = \frac{N_t(s, a, s')}{N_t(s, a)}, \quad \hat{\mathcal{R}}_t(s, a) = \frac{1}{N_t(s, a)} \sum_{i=1}^{N_t(s, a)} R_i(s, a)$$
The empirical Bellman optimality operator $\hat{\mathcal{T}}_t^*: \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|} \to \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$ is defined component-wise by:
$$(\hat{\mathcal{T}}_t^* Q)(s, a) \triangleq \hat{\mathcal{R}}_t(s, a) + \gamma \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}_t(s' \mid s, a) \max_{a' \in \mathcal{A}} Q(s', a')$$
During the Dyna planning phase, the agent executes $N$ asynchronous stochastic approximation updates using transitions sampled from $\hat{\mathcal{M}}_t$:
$$Q_{k+1}(s_k, a_k) = (1 - \alpha_k(s_k, a_k)) Q_k(s_k, a_k) + \alpha_k(s_k, a_k) \left[ \hat{\mathcal{R}}_t(s_k, a_k) + \gamma \max_{a'} Q_k(s'_k, a') \right]$$
where $s'_k \sim \hat{\mathcal{P}}_t(\cdot \mid s_k, a_k)$.

Our mathematical goal is four-fold:
1. Prove rigorously that $\hat{\mathcal{T}}_t^*$ is a strict contraction mapping in the supremum norm with Lipschitz modulus $\gamma < 1$.
2. Prove that the planning iterate sequence $\{Q_k\}_{k=0}^\infty$ converges almost surely to the unique fixed point $Q_{\hat{\mathcal{M}}_t}^*$ of the empirical model.
3. Derive the exact fixed-point perturbation bound relating $Q_{\hat{\mathcal{M}}_t}^*$ to the true optimal action-value function $Q^*$.
4. Establish that under persistent real-world exploration, $Q_{\hat{\mathcal{M}}_t}^* \xrightarrow{\text{a.s.}} Q^*$.

**2. Explicit Assumptions & Regularity Conditions:**
1. **Finite State-Action Cardinality:** $|\mathcal{S}| < \infty$ and $|\mathcal{A}| < \infty$.
2. **Discount Regularity:** $\gamma \in [0, 1)$, ensuring the convergence of geometric series $\sum_{k=0}^\infty \gamma^k = \frac{1}{1 - \gamma} < \infty$.
3. **Reward Boundedness:** $|R(s, a)| \le R_{\max} < \infty$ almost surely for all $(s, a)$, which ensures that any value function satisfies $\|V\|_\infty \le \frac{R_{\max}}{1 - \gamma}$.
4. **Empirical Stochasticity:** For every visited pair $(s, a)$ with $N_t(s, a) \ge 1$, $\hat{\mathcal{P}}_t(s' \mid s, a) \ge 0$ for all $s'$ and $\sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}_t(s' \mid s, a) = 1$.
5. **Robbins-Monro Planning Step Sizes:** For all $(s, a) \in \text{dom}(\hat{\mathcal{M}}_t)$, the simulated learning rate sequence $\alpha_k(s, a) \in (0, 1]$ satisfies:
   $$\sum_{k=0}^\infty \alpha_k(s, a) = \infty, \quad \sum_{k=0}^\infty \alpha_k^2(s, a) < \infty$$
6. **Persistent Planning Sampling:** Each visited state-action pair $(s, a)$ in the empirical model domain is sampled infinitely often during planning: $\lim_{K \to \infty} \sum_{k=1}^K \mathbb{I}((s_k, a_k) = (s, a)) = \infty$ almost surely.
7. **Empirical Model Consistency:** As real-world exploration time $t \to \infty$, every state-action pair $(s, a)$ is visited infinitely often ($N_t(s, a) \to \infty$), so that by the Strong Law of Large Numbers, $\hat{\mathcal{P}}_t(s' \mid s, a) \xrightarrow{\text{a.s.}} \mathcal{P}(s' \mid s, a)$ and $\hat{\mathcal{R}}_t(s, a) \xrightarrow{\text{a.s.}} \mathcal{R}(s, a)$.

**3. Underlying Intuition & Geometric / Physical Interpretation:**
- **The Planning Micro-Universe:** The empirical model $\hat{\mathcal{M}}_t$ defines a fully closed, Markovian "simulated universe". Because $\hat{\mathcal{P}}_t$ is a row-stochastic transition matrix, it preserves the exact algebraic contraction properties of Bellman systems. The planning loop is not merely arbitrary data augmentation; it is asynchronous stochastic value iteration running inside this empirical micro-universe.
- **Contraction in Banach Space:** In the complete metric space $(\mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}, \|\cdot\|_\infty)$, the empirical Bellman optimality operator $\hat{\mathcal{T}}_t^*$ shrinks the distance between any two value tables by at least $\gamma$ at every full pass. Any asynchronous sequence that visits all states infinitely often contracts the error tube exponentially.
- **Perturbation Geometry:** The empirical fixed point $Q_{\hat{\mathcal{M}}_t}^*$ sits at a location in $\mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$ displaced from the true fixed point $Q^*$. The distance $\|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty$ is geometrically bounded by the vertical operator divergence $\|\hat{\mathcal{T}}_t^* Q^* - \mathcal{T}^* Q^*\|_\infty$ magnified by the resolvent condition number $\frac{1}{1 - \gamma}$. As real transitions accumulate, the empirical operator surface coalesces with the true operator surface, driving the fixed-point displacement to zero.

**4. End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Rigorous proof of the non-expansion property of the maximum operator.*
Let $Q_1, Q_2 \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$ be arbitrary action-value functions. Fix any state $s' \in \mathcal{S}$.
Let $a_1^* \in \arg\max_{a'} Q_1(s', a')$ and $a_2^* \in \arg\max_{a'} Q_2(s', a')$.
By definition of the supremum / maximum:
$$\max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') = Q_1(s', a_1^*) - Q_2(s', a_2^*)$$
Since $a_2^*$ maximizes $Q_2(s', \cdot)$, we have $Q_2(s', a_2^*) \ge Q_2(s', a_1^*)$. Negating this inequality yields $-Q_2(s', a_2^*) \le -Q_2(s', a_1^*)$. Therefore:
$$\max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') \le Q_1(s', a_1^*) - Q_2(s', a_1^*)$$
Since $x \le |x|$ for any real number $x$:
$$Q_1(s', a_1^*) - Q_2(s', a_1^*) \le |Q_1(s', a_1^*) - Q_2(s', a_1^*)| \le \max_{a'} |Q_1(s', a') - Q_2(s', a')|$$
By symmetry, reversing the roles of $Q_1$ and $Q_2$ yields:
$$\max_{a'} Q_2(s', a') - \max_{a'} Q_1(s', a') \le \max_{a'} |Q_1(s', a') - Q_2(s', a')|$$
Multiplying by $-1$:
$$\max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') \ge -\max_{a'} |Q_1(s', a') - Q_2(s', a')|$$
Combining both inequalities:
$$\left| \max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') \right| \le \max_{a'} |Q_1(s', a') - Q_2(s', a')| \le \|Q_1 - Q_2\|_\infty \quad \text{(Lemma 1)}$$

*Step 2: Contraction mapping proof for the empirical Bellman operator $\hat{\mathcal{T}}_t^*$.*
For any state-action pair $(s, a) \in \mathcal{S} \times \mathcal{A}$:
$$|(\hat{\mathcal{T}}_t^* Q_1)(s, a) - (\hat{\mathcal{T}}_t^* Q_2)(s, a)| = \left| \left( \hat{\mathcal{R}}_t(s, a) + \gamma \sum_{s'} \hat{\mathcal{P}}_t(s' \mid s, a) \max_{a'} Q_1(s', a') \right) - \left( \hat{\mathcal{R}}_t(s, a) + \gamma \sum_{s'} \hat{\mathcal{P}}_t(s' \mid s, a) \max_{a'} Q_2(s', a') \right) \right|$$
The empirical reward terms $\hat{\mathcal{R}}_t(s, a)$ cancel out identically:
$$= \gamma \left| \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}_t(s' \mid s, a) \left[ \max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') \right] \right|$$
Applying the triangle inequality:
$$\le \gamma \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}_t(s' \mid s, a) \left| \max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') \right|$$
Applying Lemma 1:
$$\le \gamma \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}_t(s' \mid s, a) \|Q_1 - Q_2\|_\infty$$
Because $\hat{\mathcal{P}}_t$ is row-stochastic (Assumption 4), $\sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}_t(s' \mid s, a) = 1$:
$$= \gamma \|Q_1 - Q_2\|_\infty \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}_t(s' \mid s, a) = \gamma \|Q_1 - Q_2\|_\infty$$
Taking the supremum over all $(s, a) \in \mathcal{S} \times \mathcal{A}$:
$$\|\hat{\mathcal{T}}_t^* Q_1 - \hat{\mathcal{T}}_t^* Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
Since $\gamma \in [0, 1)$, $\hat{\mathcal{T}}_t^*$ is a strict contraction mapping on the complete Banach space $(\mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}, \|\cdot\|_\infty)$.
By the Banach Fixed-Point Theorem, there exists a unique fixed point $Q_{\hat{\mathcal{M}}_t}^*$ satisfying:
$$\hat{\mathcal{T}}_t^* Q_{\hat{\mathcal{M}}_t}^* = Q_{\hat{\mathcal{M}}_t}^*$$

*Step 3: Asynchronous stochastic approximation convergence of the planning loop.*
Let the planning update at simulated step $k$ for pair $(s_k, a_k)$ be written as:
$$Q_{k+1}(s, a) = Q_k(s, a) + \alpha_k(s, a) \mathbb{I}((s, a) = (s_k, a_k)) \left[ \hat{\mathcal{R}}_t(s, a) + \gamma \max_{a'} Q_k(s', a') - Q_k(s, a) \right]$$
where $s' \sim \hat{\mathcal{P}}_t(\cdot \mid s, a)$.
Rewrite this update in canonical stochastic approximation form:
$$Q_{k+1}(s, a) = Q_k(s, a) + \alpha_k(s, a) \mathbb{I}((s, a) = (s_k, a_k)) \left[ (\hat{\mathcal{T}}_t^* Q_k)(s, a) - Q_k(s, a) + \omega_{k+1}(s, a) \right]$$
where the noise term $\omega_{k+1}(s, a)$ is defined as:
$$\omega_{k+1}(s, a) \triangleq \gamma \max_{a'} Q_k(s', a') - \gamma \sum_{s''} \hat{\mathcal{P}}_t(s'' \mid s, a) \max_{a'} Q_k(s'', a')$$
Let $\mathcal{F}_k = \sigma(Q_0, (s_0, a_0, s'_0), \dots, (s_{k-1}, a_{k-1}, s'_{k-1}))$ be the filtration generated by the history up to step $k$.
Evaluate the conditional expectation of $\omega_{k+1}$:
$$\mathbb{E}[\omega_{k+1}(s, a) \mid \mathcal{F}_k] = \gamma \sum_{s'} \hat{\mathcal{P}}_t(s' \mid s, a) \max_{a'} Q_k(s', a') - \gamma \sum_{s''} \hat{\mathcal{P}}_t(s'' \mid s, a) \max_{a'} Q_k(s'', a') = 0$$
Hence, $\{\omega_{k+1}\}$ is a Martingale difference sequence.
Furthermore, the conditional variance is bounded:
$$\mathbb{E}\left[ |\omega_{k+1}(s, a)|^2 \mid \mathcal{F}_k \right] \le 4 \gamma^2 \|Q_k\|_\infty^2 \le C (1 + \|Q_k\|_\infty^2)$$
Under Assumptions 5 and 6 (Robbins-Monro conditions and persistent exploration of the empirical model domain), the conditions of the Tsitsiklis (1994) / Borkar-Meyn asynchronous contraction theorem are satisfied identically. Therefore:
$$\lim_{k \to \infty} Q_k = Q_{\hat{\mathcal{M}}_t}^* \quad \text{almost surely}$$

*Step 4: Derivation of the empirical fixed-point perturbation bound.*
Let $Q^*$ be the unique fixed point of the true Bellman optimality operator $\mathcal{T}^* Q^* = Q^*$, and let $Q_{\hat{\mathcal{M}}_t}^*$ be the unique fixed point of $\hat{\mathcal{T}}_t^* Q_{\hat{\mathcal{M}}_t}^* = Q_{\hat{\mathcal{M}}_t}^*$.
Subtract $Q^*$ from $Q_{\hat{\mathcal{M}}_t}^*$:
$$\|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty = \|\hat{\mathcal{T}}_t^* Q_{\hat{\mathcal{M}}_t}^* - \mathcal{T}^* Q^*\|_\infty$$
Add and subtract $\hat{\mathcal{T}}_t^* Q^*$ inside the norm:
$$= \|\hat{\mathcal{T}}_t^* Q_{\hat{\mathcal{M}}_t}^* - \hat{\mathcal{T}}_t^* Q^* + \hat{\mathcal{T}}_t^* Q^* - \mathcal{T}^* Q^*\|_\infty$$
Applying the triangle inequality:
$$\le \|\hat{\mathcal{T}}_t^* Q_{\hat{\mathcal{M}}_t}^* - \hat{\mathcal{T}}_t^* Q^*\|_\infty + \|\hat{\mathcal{T}}_t^* Q^* - \mathcal{T}^* Q^*\|_\infty$$
Using the contraction property of $\hat{\mathcal{T}}_t^*$ from Step 2:
$$\le \gamma \|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty + \|\hat{\mathcal{T}}_t^* Q^* - \mathcal{T}^* Q^*\|_\infty$$
Subtract $\gamma \|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty$ from both sides:
$$(1 - \gamma) \|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty \le \|\hat{\mathcal{T}}_t^* Q^* - \mathcal{T}^* Q^*\|_\infty$$
Dividing by $(1 - \gamma) > 0$:
$$\|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty \le \frac{1}{1 - \gamma} \|\hat{\mathcal{T}}_t^* Q^* - \mathcal{T}^* Q^*\|_\infty \quad \text{(Inequality 1)}$$

*Step 5: Operator distance evaluation at the true fixed point $Q^*$.*
Fix $(s, a)$. Evaluate the difference between operators applied to $Q^*$:
$$|(\hat{\mathcal{T}}_t^* Q^*)(s, a) - (\mathcal{T}^* Q^*)(s, a)| = \left| \left( \hat{\mathcal{R}}_t(s, a) + \gamma \sum_{s'} \hat{\mathcal{P}}_t(s' \mid s, a) V^*(s') \right) - \left( \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^*(s') \right) \right|$$
where $V^*(s') \triangleq \max_{a'} Q^*(s', a')$.
Regrouping:
$$= \left| (\hat{\mathcal{R}}_t(s, a) - \mathcal{R}(s, a)) + \gamma \sum_{s' \in \mathcal{S}} (\hat{\mathcal{P}}_t(s' \mid s, a) - \mathcal{P}(s' \mid s, a)) V^*(s') \right|$$
Applying the triangle inequality:
$$\le |\hat{\mathcal{R}}_t(s, a) - \mathcal{R}(s, a)| + \gamma \left| \sum_{s' \in \mathcal{S}} (\hat{\mathcal{P}}_t(s' \mid s, a) - \mathcal{P}(s' \mid s, a)) V^*(s') \right|$$
Applying Hölder's inequality on the second term:
$$\left| \sum_{s' \in \mathcal{S}} (\hat{\mathcal{P}}_t(s' \mid s, a) - \mathcal{P}(s' \mid s, a)) V^*(s') \right| \le \sum_{s' \in \mathcal{S}} |\hat{\mathcal{P}}_t(s' \mid s, a) - \mathcal{P}(s' \mid s, a)| \cdot \|V^*\|_\infty$$
Define the model transition matrix infinity norm:
$$\|\hat{\mathcal{P}}_t - \mathcal{P}\|_\infty \triangleq \max_{s, a} \sum_{s' \in \mathcal{S}} |\hat{\mathcal{P}}_t(s' \mid s, a) - \mathcal{P}(s' \mid s, a)|$$
Since $|R(s, a)| \le R_{\max}$, we have $\|V^*\|_\infty \le \frac{R_{\max}}{1 - \gamma}$.
Substituting this bound:
$$\|\hat{\mathcal{T}}_t^* Q^* - \mathcal{T}^* Q^*\|_\infty \le \|\hat{\mathcal{R}}_t - \mathcal{R}\|_\infty + \gamma \|\hat{\mathcal{P}}_t - \mathcal{P}\|_\infty \frac{R_{\max}}{1 - \gamma}$$
Substituting into Inequality 1:
$$\|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty \le \frac{1}{1 - \gamma} \|\hat{\mathcal{R}}_t - \mathcal{R}\|_\infty + \frac{\gamma}{(1 - \gamma)^2} \|\hat{\mathcal{P}}_t - \mathcal{P}\|_\infty R_{\max}$$

*Step 6: Empirical consistency and final convergence.*
By Assumption 7, each state-action pair is visited infinitely often in real experience ($N_t(s, a) \to \infty$).
By the Strong Law of Large Numbers:
$$\lim_{t \to \infty} \|\hat{\mathcal{R}}_t - \mathcal{R}\|_\infty = 0 \quad \text{a.s.}, \quad \lim_{t \to \infty} \|\hat{\mathcal{P}}_t - \mathcal{P}\|_\infty = 0 \quad \text{a.s.}$$
Taking the limit as $t \to \infty$:
$$\lim_{t \to \infty} \|Q_{\hat{\mathcal{M}}_t}^* - Q^*\|_\infty \le \frac{1}{1 - \gamma} (0) + \frac{\gamma}{(1 - \gamma)^2} (0) R_{\max} = 0$$
Hence, $Q_{\hat{\mathcal{M}}_t}^* \xrightarrow{\text{a.s.}} Q^*$. $\blacksquare$

---

#### Derivation 11.21.2: Model Error Compounding in Multi-Step Rollouts (The Simulation Lemma)

```
====================================================================================================
DERIVATION 11.21.2: The Simulation Lemma & Quadratic Horizon Error Compounding
====================================================================================================
Problem Statement:
Let ℳ = (𝒮, 𝒜, 𝒫, ℛ, γ) be the true MDP and ℳ̂ = (𝒮, 𝒜, 𝒫̂, ℛ̂, γ) be an approximate model with
transition total variation error max_{s,a} ‖𝒫(·|s,a) - 𝒫̂(·|s,a)‖_TV ≤ ϵ_m / 2 and reward error
max_{s,a} |ℛ(s,a) - ℛ̂(s,a)| ≤ ϵ_r. Prove rigorously from the Bellman resolvent equations that
for any stationary policy π, the performance gap between true and simulated value functions satisfies:
    |J(π) - Ĵ(π)| ≤ ‖V^π - V̂^π‖_∞ ≤ (ϵ_r / (1 - γ)) + (γ ϵ_m R_max / (1 - γ)^2)
Furthermore, prove that for an autoregressively unrolled multi-step rollout of horizon H, the
state distribution error compounds quadratically:
    ∑_{t=1}^H ‖d_t^π - d̂_t^π‖_TV ≤ (H(H + 1) / 2) ϵ_TV = 𝒪(H^2 ϵ_TV)
====================================================================================================
```

**1. Problem Statement & Mathematical Goal:**
In model-based reinforcement learning, planning involves generating simulated state trajectories using a learned transition model $\hat{\mathcal{P}}$ and reward model $\hat{\mathcal{R}}$.
Let $V^\pi \in \mathbb{R}^{|\mathcal{S}|}$ denote the true value function of policy $\pi$ under true dynamics $(\mathcal{P}, \mathcal{R})$, and let $\hat{V}^\pi \in \mathbb{R}^{|\mathcal{S}|}$ denote the value function under model dynamics $(\hat{\mathcal{P}}, \hat{\mathcal{R}})$.
Let the policy performance under initial state distribution $\mu \in \Delta(\mathcal{S})$ be defined by:
$$J(\pi) \triangleq \mu^\top \mathbf{V}^\pi, \quad \hat{J}(\pi) \triangleq \mu^\top \hat{\mathbf{V}}^\pi$$
Let the single-step model errors be bounded by:
$$\max_{s, a} |\mathcal{R}(s, a) - \hat{\mathcal{R}}(s, a)| \le \epsilon_r, \quad \max_{s, a} \|\mathcal{P}(\cdot \mid s, a) - \hat{\mathcal{P}}(\cdot \mid s, a)\|_1 \le \epsilon_m$$
where $\|\mathcal{P}(\cdot \mid s, a) - \hat{\mathcal{P}}(\cdot \mid s, a)\|_1 = 2 \|\mathcal{P}(\cdot \mid s, a) - \hat{\mathcal{P}}(\cdot \mid s, a)\|_{\text{TV}} = 2 \epsilon_{\text{TV}}$.

Our mathematical goal is two-fold:
1. Prove the fundamental **Simulation Lemma**:
   $$\|\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi\|_\infty \le \frac{\epsilon_r}{1 - \gamma} + \frac{\gamma \epsilon_m R_{\max}}{(1 - \gamma)^2}$$
   which bounds the policy evaluation error in the model.
2. Prove the **Compounding Distribution Drift Theorem**: In a finite-horizon autoregressive rollout of length $H$, the total variation error of the state distribution at step $t$ satisfies $\|d_t^\pi - \hat{d}_t^\pi\|_{\text{TV}} \le t \cdot \epsilon_{\text{TV}}$, and the cumulative trajectory error compounds as $\mathcal{O}(H^2 \epsilon_{\text{TV}})$.

**2. Explicit Assumptions & Regularity Conditions:**
1. **Stationary Policy:** Policy $\pi(a \mid s)$ is a valid conditional probability distribution ($\pi(a \mid s) \ge 0, \sum_a \pi(a \mid s) = 1$).
2. **Transition Probability Kernels:** Both $\mathcal{P}$ and $\hat{\mathcal{P}}$ are valid Markov transition matrices: $\mathcal{P}(s' \mid s, a) \ge 0, \hat{\mathcal{P}}(s' \mid s, a) \ge 0$ and $\sum_{s'} \mathcal{P}(s' \mid s, a) = \sum_{s'} \hat{\mathcal{P}}(s' \mid s, a) = 1$ for all $(s, a)$.
3. **Bounded Rewards:** $R(s, a) \in [0, R_{\max}]$ (or $|R(s, a)| \le R_{\max}$), ensuring that $\|\hat{\mathbf{V}}^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma}$.
4. **Discount Modulus:** $\gamma \in [0, 1)$, ensuring invertibility of the Bellman operators $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ and $(\mathbf{I} - \gamma \hat{\mathbf{P}}^\pi)$.
5. **Initial State Distribution Alignment:** Both the true system and the model rollout start from the identical initial state distribution $d_0 = \hat{d}_0 = \mu$.

**3. Underlying Intuition & Geometric / Physical Interpretation:**
- **The Resolvent Filter:** The true value function and model value function are solutions to linear algebraic systems $(\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{V}^\pi = \mathbf{R}^\pi$ and $(\mathbf{I} - \gamma \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi = \hat{\mathbf{R}}^\pi$. The difference vector $\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi$ can be decomposed using the matrix resolvent identity into:
  1) Immediate reward mismatch: $(\mathbf{R}^\pi - \hat{\mathbf{R}}^\pi)$.
  2) Transition prediction drift: $\gamma (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi$.
- **Why $(1 - \gamma)^2$ appears:** The transition error $(\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi)$ acts on the continuation value vector $\hat{\mathbf{V}}^\pi$. Since $\hat{\mathbf{V}}^\pi$ has magnitude $\mathcal{O}\left(\frac{R_{\max}}{1 - \gamma}\right)$, the one-step transition mismatch produces an effective perturbation of magnitude $\epsilon_m \frac{R_{\max}}{1 - \gamma}$. Then, passing this perturbation through the infinite-horizon Bellman resolvent $(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}$ amplifies it by another factor of $\frac{1}{1 - \gamma}$, producing the characteristic $(1 - \gamma)^{-2}$ denominator!
- **Quadratic Divergence of Hallucinations:** When an agent simulates $H$ steps autoregressively ($s_0 \to \hat{s}_1 \to \hat{s}_2 \dots$), an error made at step 1 distorts the conditioning distribution for step 2. Step 2 adds its own error and cascades into step 3. The error at step $t$ grows linearly with $t$, so integrating over an $H$-step planning horizon produces quadratic area growth $\sum_{t=1}^H t \approx \frac{H^2}{2}$.

**4. End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Formulate the vector-matrix Bellman expectation systems.*
Define the state transition matrices induced by policy $\pi$:
$$[\mathbf{P}^\pi]_{s, s'} \triangleq \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{P}(s' \mid s, a), \quad [\hat{\mathbf{P}}^\pi]_{s, s'} \triangleq \sum_{a \in \mathcal{A}} \pi(a \mid s) \hat{\mathcal{P}}(s' \mid s, a)$$
Define the expected immediate reward vectors:
$$\mathbf{R}^\pi(s) \triangleq \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{R}(s, a), \quad \hat{\mathbf{R}}^\pi(s) \triangleq \sum_{a \in \mathcal{A}} \pi(a \mid s) \hat{\mathcal{R}}(s, a)$$
The Bellman expectation equations in matrix form are:
$$\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi \iff (\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{V}^\pi = \mathbf{R}^\pi \quad \text{(Eq. 1)}$$
$$\hat{\mathbf{V}}^\pi = \hat{\mathbf{R}}^\pi + \gamma \hat{\mathbf{P}}^\pi \hat{\mathbf{V}}^\pi \iff (\mathbf{I} - \gamma \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi = \hat{\mathbf{R}}^\pi \quad \text{(Eq. 2)}$$

*Step 2: Derive the exact algebraic value difference identity.*
Subtract Eq. 2 from Eq. 1:
$$\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi = (\mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi) - \hat{\mathbf{V}}^\pi$$
Add and subtract the cross term $\gamma \mathbf{P}^\pi \hat{\mathbf{V}}^\pi$:
$$\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi = \mathbf{R}^\pi - \hat{\mathbf{V}}^\pi + \gamma \mathbf{P}^\pi \hat{\mathbf{V}}^\pi + \gamma \mathbf{P}^\pi (\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi)$$
Now substitute the Bellman expansion $\hat{\mathbf{V}}^\pi = \hat{\mathbf{R}}^\pi + \gamma \hat{\mathbf{P}}^\pi \hat{\mathbf{V}}^\pi$ into the first $\hat{\mathbf{V}}^\pi$:
$$\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi = \mathbf{R}^\pi - (\hat{\mathbf{R}}^\pi + \gamma \hat{\mathbf{P}}^\pi \hat{\mathbf{V}}^\pi) + \gamma \mathbf{P}^\pi \hat{\mathbf{V}}^\pi + \gamma \mathbf{P}^\pi (\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi)$$
$$= (\mathbf{R}^\pi - \hat{\mathbf{R}}^\pi) + \gamma (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi + \gamma \mathbf{P}^\pi (\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi)$$
Isolate the value difference terms on the left-hand side:
$$(\mathbf{I} - \gamma \mathbf{P}^\pi) (\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi) = (\mathbf{R}^\pi - \hat{\mathbf{R}}^\pi) + \gamma (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi$$
Since $\mathbf{P}^\pi$ is row-stochastic ($\|\mathbf{P}^\pi\|_\infty = 1$) and $\gamma < 1$, the spectral radius $\rho(\gamma \mathbf{P}^\pi) \le \gamma < 1$.
Thus, $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ is non-singular and its inverse is given by the convergent Neumann series:
$$(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} = \sum_{k=0}^\infty \gamma^k (\mathbf{P}^\pi)^k$$
Multiplying both sides by the inverse resolvent:
$$\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \left[ (\mathbf{R}^\pi - \hat{\mathbf{R}}^\pi) + \gamma (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi \right] \quad \text{(Identity 1)}$$

*Step 3: Bound the individual components in $L_\infty$ norm.*
Taking the $L_\infty$ norm of both sides of Identity 1 and applying sub-multiplicativity:
$$\|\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi\|_\infty \le \|(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}\|_\infty \left( \|\mathbf{R}^\pi - \hat{\mathbf{R}}^\pi\|_\infty + \gamma \|(\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi\|_\infty \right)$$

1. **Resolvent Norm Bound:**
   $$\|(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}\|_\infty = \left\| \sum_{k=0}^\infty \gamma^k (\mathbf{P}^\pi)^k \right\|_\infty \le \sum_{k=0}^\infty \gamma^k \|(\mathbf{P}^\pi)^k\|_\infty$$
   Since the product of stochastic matrices is stochastic, $\|(\mathbf{P}^\pi)^k\|_\infty = 1$. Therefore:
   $$\|(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}\|_\infty \le \sum_{k=0}^\infty \gamma^k = \frac{1}{1 - \gamma}$$

2. **Reward Discrepancy Bound:**
   For any state $s \in \mathcal{S}$:
   $$|(\mathbf{R}^\pi - \hat{\mathbf{R}}^\pi)(s)| = \left| \sum_{a \in \mathcal{A}} \pi(a \mid s) (\mathcal{R}(s, a) - \hat{\mathcal{R}}(s, a)) \right| \le \sum_{a \in \mathcal{A}} \pi(a \mid s) |\mathcal{R}(s, a) - \hat{\mathcal{R}}(s, a)| \le \sum_{a} \pi(a \mid s) \epsilon_r = \epsilon_r$$
   Taking the maximum over $s \implies \|\mathbf{R}^\pi - \hat{\mathbf{R}}^\pi\|_\infty \le \epsilon_r$.

3. **Transition Discrepancy Bound:**
   For any state $s \in \mathcal{S}$:
   $$|[(\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi](s)| = \left| \sum_{s' \in \mathcal{S}} \left( [\mathbf{P}^\pi]_{s, s'} - [\hat{\mathbf{P}}^\pi]_{s, s'} \right) \hat{\mathbf{V}}^\pi(s') \right|$$
   $$= \left| \sum_{s' \in \mathcal{S}} \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( \mathcal{P}(s' \mid s, a) - \hat{\mathcal{P}}(s' \mid s, a) \right) \hat{\mathbf{V}}^\pi(s') \right|$$
   Exchanging summations and applying the triangle inequality:
   $$\le \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \left| \mathcal{P}(s' \mid s, a) - \hat{\mathcal{P}}(s' \mid s, a) \right| |\hat{\mathbf{V}}^\pi(s')|$$
   $$\le \sum_{a \in \mathcal{A}} \pi(a \mid s) \|\hat{\mathbf{V}}^\pi\|_\infty \sum_{s' \in \mathcal{S}} \left| \mathcal{P}(s' \mid s, a) - \hat{\mathcal{P}}(s' \mid s, a) \right|$$
   Since $\sum_{s'} |\mathcal{P}(s' \mid s, a) - \hat{\mathcal{P}}(s' \mid s, a)| = \|\mathcal{P}(\cdot \mid s, a) - \hat{\mathcal{P}}(\cdot \mid s, a)\|_1 \le \epsilon_m$:
   $$\le \sum_{a \in \mathcal{A}} \pi(a \mid s) \|\hat{\mathbf{V}}^\pi\|_\infty \epsilon_m = \epsilon_m \|\hat{\mathbf{V}}^\pi\|_\infty$$
   By Assumption 3, the simulated values are bounded by $\|\hat{\mathbf{V}}^\pi\|_\infty \le \frac{R_{\max}}{1 - \gamma}$.
   Therefore:
   $$\|(\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \hat{\mathbf{V}}^\pi\|_\infty \le \frac{\epsilon_m R_{\max}}{1 - \gamma}$$

*Step 4: Combine to obtain the Simulation Lemma.*
Substitute the derived bounds into the $L_\infty$ inequality:
$$\|\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi\|_\infty \le \frac{1}{1 - \gamma} \left[ \epsilon_r + \gamma \frac{\epsilon_m R_{\max}}{1 - \gamma} \right] = \frac{\epsilon_r}{1 - \gamma} + \frac{\gamma \epsilon_m R_{\max}}{(1 - \gamma)^2}$$
For the scalar expected return $J(\pi) = \mu^\top \mathbf{V}^\pi$:
$$|J(\pi) - \hat{J}(\pi)| = |\mu^\top (\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi)| \le \|\mu\|_1 \|\mathbf{V}^\pi - \hat{\mathbf{V}}^\pi\|_\infty$$
Since $\mu$ is a probability distribution, $\|\mu\|_1 = \sum_s \mu(s) = 1$:
$$|J(\pi) - \hat{J}(\pi)| \le \frac{\epsilon_r}{1 - \gamma} + \frac{\gamma \epsilon_m R_{\max}}{(1 - \gamma)^2}$$
When the reward model is exact ($\epsilon_r = 0$), this simplifies directly to:
$$|J(\pi) - \hat{J}(\pi)| \le \frac{\gamma \epsilon_m R_{\max}}{(1 - \gamma)^2}$$

*Step 5: Multi-step distribution drift compounding proof.*
Let $d_t^\top \triangleq d_0^\top (\mathbf{P}^\pi)^t$ be the true state distribution vector at step $t$.
Let $\hat{d}_t^\top \triangleq d_0^\top (\hat{\mathbf{P}}^\pi)^t$ be the model rollout state distribution vector at step $t$.
Compute the difference between the two distributions:
$$d_t^\top - \hat{d}_t^\top = d_0^\top \left[ (\mathbf{P}^\pi)^t - (\hat{\mathbf{P}}^\pi)^t \right]$$
Using the telescoping matrix factorization identity:
$$(\mathbf{P}^\pi)^t - (\hat{\mathbf{P}}^\pi)^t = \sum_{k=0}^{t-1} (\mathbf{P}^\pi)^k (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) (\hat{\mathbf{P}}^\pi)^{t - 1 - k}$$
Multiplying by $d_0^\top$:
$$d_t^\top - \hat{d}_t^\top = \sum_{k=0}^{t-1} d_k^\top (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) (\hat{\mathbf{P}}^\pi)^{t - 1 - k}$$
where $d_k^\top = d_0^\top (\mathbf{P}^\pi)^k$ is the true state distribution at step $k$.
Take the total variation norm, where $\|p - q\|_{\text{TV}} = \frac{1}{2} \|p - q\|_1$:
$$\|d_t^\pi - \hat{d}_t^\pi\|_{\text{TV}} = \frac{1}{2} \|d_t^\top - \hat{d}_t^\top\|_1 \le \frac{1}{2} \sum_{k=0}^{t-1} \left\| d_k^\top (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) (\hat{\mathbf{P}}^\pi)^{t - 1 - k} \right\|_1$$
Because $(\hat{\mathbf{P}}^\pi)^{t - 1 - k}$ is a row-stochastic Markov matrix, it is a non-expansion in $\ell_1$ norm ($\|v^\top \mathbf{P}\|_1 \le \|v\|_1$):
$$\le \frac{1}{2} \sum_{k=0}^{t-1} \left\| d_k^\top (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \right\|_1$$
Expand the matrix product inside the $\ell_1$ norm:
$$\left\| d_k^\top (\mathbf{P}^\pi - \hat{\mathbf{P}}^\pi) \right\|_1 = \sum_{s' \in \mathcal{S}} \left| \sum_{s \in \mathcal{S}} d_k(s) \left( [\mathbf{P}^\pi]_{s, s'} - [\hat{\mathbf{P}}^\pi]_{s, s'} \right) \right|$$
Applying the triangle inequality:
$$\le \sum_{s \in \mathcal{S}} d_k(s) \sum_{s' \in \mathcal{S}} \left| [\mathbf{P}^\pi]_{s, s'} - [\hat{\mathbf{P}}^\pi]_{s, s'} \right| \le \sum_{s \in \mathcal{S}} d_k(s) (2 \epsilon_{\text{TV}}) = 2 \epsilon_{\text{TV}} \sum_s d_k(s) = 2 \epsilon_{\text{TV}}$$
Multiplying by $\frac{1}{2}$:
$$\|d_t^\pi - \hat{d}_t^\pi\|_{\text{TV}} \le \frac{1}{2} \sum_{k=0}^{t-1} (2 \epsilon_{\text{TV}}) = \sum_{k=0}^{t-1} \epsilon_{\text{TV}} = t \cdot \epsilon_{\text{TV}}$$
Summing the total variation drift over the entire rollout horizon $H$:
$$\sum_{t=1}^H \|d_t^\pi - \hat{d}_t^\pi\|_{\text{TV}} \le \sum_{t=1}^H t \cdot \epsilon_{\text{TV}} = \frac{H(H + 1)}{2} \epsilon_{\text{TV}} = \mathcal{O}(H^2 \epsilon_{\text{TV}}) \quad \blacksquare$$

---

#### Derivation 11.21.3: Dyna-Q+ Exploration Bonus Operator and Asymptotic Drift Dynamics

```
====================================================================================================
DERIVATION 11.21.3: Dyna-Q+ Exploration Bonus Operator & Overtaking Dynamics
====================================================================================================
Problem Statement:
In non-stationary MDPs, let τ_t(s, a) denote the elapsed real interaction steps since (s, a) was last
executed. Dyna-Q+ augments the planning reward with an exploration bonus r_bonus(s,a) = r̂(s,a) + κ √τ(s,a).
Prove that for any fixed counter snapshot τ, the bonus planning operator 𝒯̂_{κ,τ}^* remains a strict
γ-contraction. Prove that the bonus-augmented fixed point Q^*_{κ,τ} satisfies:
    Q^*_{κ,τ}(s, a) ≥ Q^*_0(s, a) + κ √τ(s, a)
and determine the exact critical threshold time τ^*(s) required for a neglected action with
suboptimality gap Δ(s) = max_a Q^*(s, a) - Q^*(s, a_neg) > 0 to overtake the currently active policy:
    τ^*(s) = ⌈(Δ(s) / κ)^2⌉
Finally, prove that Dyna-Q+ achieves bounded recurrence time: no state-action pair is starved of
exploration indefinitely.
====================================================================================================
```

**1. Problem Statement & Mathematical Goal:**
In non-stationary reinforcement learning, the transition or reward dynamics of the physical environment can change arbitrarily at unannounced times.
To prevent an agent from becoming permanently locked into an obsolete policy, Sutton's **Dyna-Q+** algorithm tracks an elapsed time counter $\tau_t(s, a) \in \mathbb{N}_{\ge 0}$ for every state-action pair $(s, a) \in \mathcal{S} \times \mathcal{A}$. The counter updates deterministically at each real physical interaction step according to:
$$\tau_{t+1}(s, a) = \begin{cases} 0 & \text{if } (S_t, A_t) = (s, a) \\ \tau_t(s, a) + 1 & \text{otherwise} \end{cases}$$
During the internal planning phase, Dyna-Q+ replaces the empirical reward $\hat{\mathcal{R}}(s, a)$ with the bonus-augmented reward:
$$R_{\text{bonus}}(s, a) \triangleq \hat{\mathcal{R}}(s, a) + \kappa \sqrt{\tau(s, a)}$$
where $\kappa > 0$ is a strictly positive exploration bonus scale parameter.

The corresponding Dyna-Q+ planning operator $\hat{\mathcal{T}}_{\kappa, \tau}^*: \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|} \to \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$ is defined by:
$$(\hat{\mathcal{T}}_{\kappa, \tau}^* Q)(s, a) \triangleq \hat{\mathcal{R}}(s, a) + \kappa \sqrt{\tau(s, a)} + \gamma \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q(s', a')$$

Our mathematical goals are:
1. Prove that for any static snapshot of elapsed counters $\tau$, $\hat{\mathcal{T}}_{\kappa, \tau}^*$ is a strict $\gamma$-contraction in $L_\infty$ norm.
2. Establish the lower bound drift dynamic $Q_{\kappa, \tau}^*(s, a) \ge Q_0^*(s, a) + \kappa \sqrt{\tau(s, a)}$ for any state-action pair neglected for $\tau$ steps.
3. Derive the exact closed-form overtaking threshold $\tau^*(s)$ at which an unvisited action's simulated value overtakes an active greedy action with suboptimality gap $\Delta(s)$.
4. Prove that the recurrence time between physical executions of any state-action pair is bounded, establishing guaranteed change detection.

**2. Explicit Assumptions & Regularity Conditions:**
1. **Positive Exploration Scale:** $\kappa \in (0, \infty)$ is a strictly positive real constant.
2. **Counter Dynamics:** Time counters $\tau_t(s, a)$ are non-negative integers incrementing by 1 at every real physical time step unless physically executed, at which point they reset to 0.
3. **Discount Modulus:** $\gamma \in [0, 1)$.
4. **Finite Cardinality:** $|\mathcal{S}| < \infty$ and $|\mathcal{A}| < \infty$.
5. **Greedy Action Selection:** Action selection in the real environment is $\epsilon$-greedy (or greedy) with respect to the planning action-values $Q$.
6. **Planning Convergence:** The planning engine executes sufficient iterations $N$ such that $Q$ closely tracks the fixed point $Q_{\kappa, \tau}^*$.

**3. Underlying Intuition & Geometric / Physical Interpretation:**
- **Epistemic Uncertainty Pressure:** As an agent operates in one corner of an environment, the elapsed time $\tau(s, a)$ for unvisited actions elsewhere steadily accumulates. The square-root bonus $\kappa \sqrt{\tau}$ acts like a thermodynamic pressure vessel accumulating internal tension over time.
- **Why $\sqrt{\tau}$ instead of $\tau$?** A linear bonus $\kappa \tau$ accumulates too aggressively, causing the agent to thrash wildly and abandon newly discovered optimal paths after merely a few steps. A logarithmic bonus $\kappa \log \tau$ grows too sluggishly, requiring astronomical time to detect changes. The square-root rate $\sqrt{\tau}$ exactly matches the standard deviation of diffusive Brownian motion $\sigma_t = \sqrt{t}$, representing the standard rate of variance expansion in an unknown stochastic environment!
- **Phase Transition in Action Selection:** Geometrically, the action values in state $s$ define a set of points $\{Q(s, a)\}_{a \in \mathcal{A}}$. As $\tau(s, a_{\text{neglected}})$ increases, its value coordinate rises along a parabolic trajectory. Eventually, it intersects and surpasses the highest active value $Q(s, a^*)$. The $\arg\max$ flips, forcing the agent to physically test the transition. If the world has changed (e.g., a shortcut opened), the transition model is permanently updated. If not, $\tau$ resets to zero, dropping the value back down and allowing the agent to return to exploitation.

**4. End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Strict $\gamma$-contraction of the Dyna-Q+ bonus operator.*
Let $\tau \in \mathbb{R}_{\ge 0}^{|\mathcal{S}| \times |\mathcal{A}|}$ be any fixed vector of counter values.
Let $Q_1, Q_2 \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$. For any $(s, a) \in \mathcal{S} \times \mathcal{A}$:
$$|(\hat{\mathcal{T}}_{\kappa, \tau}^* Q_1)(s, a) - (\hat{\mathcal{T}}_{\kappa, \tau}^* Q_2)(s, a)|$$
$$= \left| \left( \hat{\mathcal{R}}(s, a) + \kappa \sqrt{\tau(s, a)} + \gamma \sum_{s'} \hat{\mathcal{P}}(s' \mid s, a) \max_{a'} Q_1(s', a') \right) - \left( \hat{\mathcal{R}}(s, a) + \kappa \sqrt{\tau(s, a)} + \gamma \sum_{s'} \hat{\mathcal{P}}(s' \mid s, a) \max_{a'} Q_2(s', a') \right) \right|$$
Notice that both $\hat{\mathcal{R}}(s, a)$ and the bonus $\kappa \sqrt{\tau(s, a)}$ are identical in both terms:
$$\left[ \hat{\mathcal{R}}(s, a) + \kappa \sqrt{\tau(s, a)} \right] - \left[ \hat{\mathcal{R}}(s, a) + \kappa \sqrt{\tau(s, a)} \right] = 0$$
Thus, the difference reduces identically to:
$$= \gamma \left| \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}(s' \mid s, a) \left[ \max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a') \right] \right|$$
Applying the non-expansion property of the maximum operator (Lemma 1 from Derivation 11.21.1):
$$\le \gamma \sum_{s' \in \mathcal{S}} \hat{\mathcal{P}}(s' \mid s, a) \|Q_1 - Q_2\|_\infty = \gamma \|Q_1 - Q_2\|_\infty$$
Taking the supremum over all $(s, a)$:
$$\|\hat{\mathcal{T}}_{\kappa, \tau}^* Q_1 - \hat{\mathcal{T}}_{\kappa, \tau}^* Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
Because $\gamma < 1$, $\hat{\mathcal{T}}_{\kappa, \tau}^*$ is a strict $\gamma$-contraction in $L_\infty$ norm, possessing a unique fixed point $Q_{\kappa, \tau}^*$.

*Step 2: Decomposition and lower bound on the bonus fixed point.*
Let $Q_0^*$ denote the baseline fixed point with zero bonus ($\kappa = 0$):
$$Q_0^*(s, a) = \hat{\mathcal{R}}(s, a) + \gamma \sum_{s'} \hat{\mathcal{P}}(s' \mid s, a) \max_{a'} Q_0^*(s', a')$$
The bonus fixed point satisfies:
$$Q_{\kappa, \tau}^*(s, a) = \hat{\mathcal{R}}(s, a) + \kappa \sqrt{\tau(s, a)} + \gamma \sum_{s'} \hat{\mathcal{P}}(s' \mid s, a) \max_{a'} Q_{\kappa, \tau}^*(s', a')$$
Subtract $Q_0^*(s, a)$ from $Q_{\kappa, \tau}^*(s, a)$:
$$Q_{\kappa, \tau}^*(s, a) - Q_0^*(s, a) = \kappa \sqrt{\tau(s, a)} + \gamma \sum_{s'} \hat{\mathcal{P}}(s' \mid s, a) \left[ \max_{a'} Q_{\kappa, \tau}^*(s', a') - \max_{a'} Q_0^*(s', a') \right]$$
Since $\kappa \sqrt{\tau(s, a)} \ge 0$ for all $(s, a)$, the bonus reward is a non-negative perturbation.
By the monotonicity of the Bellman optimality operator:
$$\mathcal{T}_1 \ge \mathcal{T}_2 \implies Q_{\mathcal{T}_1}^* \ge Q_{\mathcal{T}_2}^*$$
Therefore, $Q_{\kappa, \tau}^*(s', a') \ge Q_0^*(s', a')$ for all $(s', a')$, which implies:
$$\max_{a'} Q_{\kappa, \tau}^*(s', a') - \max_{a'} Q_0^*(s', a') \ge 0$$
Because $\hat{\mathcal{P}}(s' \mid s, a) \ge 0$ and $\gamma \ge 0$, the continuation term is strictly non-negative:
$$\gamma \sum_{s'} \hat{\mathcal{P}}(s' \mid s, a) \left[ \max_{a'} Q_{\kappa, \tau}^*(s', a') - \max_{a'} Q_0^*(s', a') \right] \ge 0$$
We conclude the fundamental lower bound:
$$Q_{\kappa, \tau}^*(s, a) \ge Q_0^*(s, a) + \kappa \sqrt{\tau(s, a)} \quad \text{(Inequality 2)}$$

*Step 3: Analytical derivation of the critical overtaking threshold $\tau^*(s)$.*
Consider a decision state $s$ where:
1. Action $a^*$ is the current greedy choice, regularly executed along the agent's nominal path, so $\tau(s, a^*) \approx 0 \implies Q_{\kappa, \tau}^*(s, a^*) \approx Q_0^*(s, a^*)$.
2. Action $a_{\text{neg}}$ is a neglected, seemingly suboptimal action last executed $\tau$ real time steps ago, with baseline value $Q_0^*(s, a_{\text{neg}})$.

Define the suboptimality gap:
$$\Delta(s) \triangleq Q_0^*(s, a^*) - Q_0^*(s, a_{\text{neg}}) > 0$$
Under greedy action selection, the agent switches from executing $a^*$ to testing $a_{\text{neg}}$ when:
$$Q_{\kappa, \tau}^*(s, a_{\text{neg}}) > Q_{\kappa, \tau}^*(s, a^*)$$
Using the lower bound from Inequality 2:
$$Q_0^*(s, a_{\text{neg}}) + \kappa \sqrt{\tau(s, a_{\text{neg}})} > Q_0^*(s, a^*)$$
Subtract $Q_0^*(s, a_{\text{neg}})$ from both sides:
$$\kappa \sqrt{\tau(s, a_{\text{neg}})} > Q_0^*(s, a^*) - Q_0^*(s, a_{\text{neg}}) = \Delta(s)$$
Dividing by $\kappa > 0$:
$$\sqrt{\tau(s, a_{\text{neg}})} > \frac{\Delta(s)}{\kappa}$$
Squaring both sides (valid since both sides are positive):
$$\tau(s, a_{\text{neg}}) > \left( \frac{\Delta(s)}{\kappa} \right)^2$$
Since $\tau$ is integer-valued, the minimum elapsed time required to guarantee that $a_{\text{neg}}$ overtakes $a^*$ is:
$$\tau^*(s) = \left\lceil \left( \frac{\Delta(s)}{\kappa} \right)^2 \right\rceil$$

*Step 4: Bounded recurrence time and guaranteed change discovery.*
Since the state and action spaces are finite and rewards are bounded by $R_{\max}$, the suboptimality gap is uniformly bounded for all states:
$$\Delta(s) \le \max_{a} Q^*(s, a) - \min_{a} Q^*(s, a) \le \frac{R_{\max}}{1 - \gamma} - \left( -\frac{R_{\max}}{1 - \gamma} \right) = \frac{2 R_{\max}}{1 - \gamma}$$
Consequently, the maximum threshold for any action in any state is bounded by a finite constant:
$$\tau_{\max}^* \le \left\lceil \left( \frac{2 R_{\max}}{\kappa (1 - \gamma)} \right)^2 \right\rceil < \infty$$
Suppose for the sake of contradiction that a state-action pair $(s, a)$ is never visited again after time $t_0$.
Then $\tau_t(s, a) = t - t_0 \to \infty$ as $t \to \infty$.
By Inequality 2, $Q_{\kappa, \tau_t}^*(s, a) \ge Q_0^*(s, a) + \kappa \sqrt{t - t_0} \to \infty$.
However, the values of all frequently visited pairs $(s', a')$ are bounded by:
$$Q_{\kappa, \tau_t}^*(s', a') \le \frac{R_{\max} + \kappa \sqrt{\tau_{\max}^*}}{1 - \gamma} < \infty$$
Thus, there must exist a finite time $T \le t_0 + \tau_{\max}^*$ at which $Q(s, a) > \max_{a' \neq a} Q(s, a')$, forcing the greedy policy to select $a$.
This contradiction proves that no state-action pair can remain unvisited indefinitely. The recurrence time between visits to every state-action pair is finite, guaranteeing that any environmental modification (such as the opening of a shortcut) is discovered within finite time. $\blacksquare$

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

### Illustration 1: Dyna-Q vs. Dyna-Q+ in a Changing Environment (Shortcut Maze Quantitative Analysis)
**Problem (Sutton's Shortcut Maze):**
Consider an agent navigating a gridworld where it has learned an established detour path around a central barrier.
At the primary decision state $S_{\text{fork}}$:
- Action $a_{\text{long}}$ follows the known detour, yielding nominal discounted path value $Q(S_{\text{fork}}, a_{\text{long}}) = 0.1200$. Because this action is regularly executed by the policy, its elapsed counter is fresh: $\tau(S_{\text{fork}}, a_{\text{long}}) = 0$.
- Action $a_{\text{wall}}$ hits an interior wall, previously producing immediate reward $R = 0.0000$ and bouncing the agent back to $S_{\text{fork}}$. Its baseline model value is $Q_0(S_{\text{fork}}, a_{\text{wall}}) = 0.0000$.
- The suboptimality gap is $\Delta = Q(S_{\text{fork}}, a_{\text{long}}) - Q(S_{\text{fork}}, a_{\text{wall}}) = 0.1200 - 0.0000 = 0.1200$.

At real interaction step $t = 1000$, the experimenter removes the barrier, creating a shortcut to the goal through $a_{\text{wall}}$.
Let the Dyna-Q+ exploration bonus scale be $\kappa = 0.0010$.

1. Analyze why standard Dyna-Q ($R_{\text{sim}} = \hat{\mathcal{R}}$) permanently fails to discover the open shortcut.
2. For Dyna-Q+ ($R_{\text{sim}} = \hat{\mathcal{R}} + \kappa \sqrt{\tau}$), calculate the bonus-augmented planning reward for $a_{\text{wall}}$ at elapsed intervals $\tau \in \{900, 4900, 10000, 14400, 25600\}$.
3. Compute the exact critical real time step threshold $\tau^*$ at which the greedy policy flips, compelling the agent to physically test the shortcut.

**Solution:**

*Part 1: Failure of Standard Dyna-Q.*
In standard Dyna-Q, the internal model stores $\text{Model}(S_{\text{fork}}, a_{\text{wall}}) = (R = 0.0000, S' = S_{\text{fork}})$.
During planning updates, the simulated target is:
$$\text{Target} = 0.0000 + \gamma \max_a Q(S_{\text{fork}}, a) = 0.9000 \times 0.1200 = 0.1080$$
Applying the Q-update with $\alpha = 0.5000$:
$$Q(S_{\text{fork}}, a_{\text{wall}}) \leftarrow Q(S_{\text{fork}}, a_{\text{wall}}) + \alpha \left[ 0.0000 + \gamma Q(S_{\text{fork}}, a_{\text{long}}) - Q(S_{\text{fork}}, a_{\text{wall}}) \right]$$
Even after multiple planning updates, the continuation state $S_{\text{fork}}$ discount prevents $Q(S_{\text{fork}}, a_{\text{wall}})$ from exceeding $Q(S_{\text{fork}}, a_{\text{long}})$ because $\gamma Q(S_{\text{fork}}, a_{\text{long}}) = 0.9 \times 0.1200 = 0.1080 < 0.1200$.
Because the greedy action selection always chooses $\arg\max_a Q(S_{\text{fork}}, a) = a_{\text{long}}$, the agent never executes $a_{\text{wall}}$ in the real world. The physical removal of the wall at $t = 1000$ never enters the model, and standard Dyna-Q is permanently blinded to the discovery.

*Part 2: Dyna-Q+ Bonus Progression Table.*
In Dyna-Q+, the simulated reward during mental rehearsal incorporates the time counter:
$$R_{\text{sim}}(S_{\text{fork}}, a_{\text{wall}}) = 0.0000 + 0.0010 \times \sqrt{\tau}$$

Let us compute the exact numerical bonus across elapsed intervals:
- At $\tau = 900$:
  $$\sqrt{900} = 30.0000 \implies R_{\text{bonus}} = 0.0010 \times 30.0000 = \mathbf{0.0300} < 0.1200 \quad (\text{Exploit } a_{\text{long}})$$
- At $\tau = 4,900$:
  $$\sqrt{4900} = 70.0000 \implies R_{\text{bonus}} = 0.0010 \times 70.0000 = \mathbf{0.0700} < 0.1200 \quad (\text{Exploit } a_{\text{long}})$$
- At $\tau = 10,000$:
  $$\sqrt{10000} = 100.0000 \implies R_{\text{bonus}} = 0.0010 \times 100.0000 = \mathbf{0.1000} < 0.1200 \quad (\text{Exploit } a_{\text{long}})$$
- At $\tau = 14,400$:
  $$\sqrt{14400} = 120.0000 \implies R_{\text{bonus}} = 0.0010 \times 120.0000 = \mathbf{0.1200} = \Delta \quad (\text{Indifference Boundary!})$$
- At $\tau = 25,600$:
  $$\sqrt{25600} = 160.0000 \implies R_{\text{bonus}} = 0.0010 \times 160.0000 = \mathbf{0.1600} > 0.1200 \quad (\text{Forced Exploration!})$$

*Part 3: Critical Threshold Calculation.*
Applying the analytical overtaking formula from Derivation 11.21.3:
$$\tau^* = \left\lceil \left( \frac{\Delta}{\kappa} \right)^2 \right\rceil = \left\lceil \left( \frac{0.1200}{0.0010} \right)^2 \right\rceil = \left\lceil (120)^2 \right\rceil = \mathbf{14,400} \text{ steps}$$
At step $t = 1000 + 14400 = 15,400$, the simulated planning target for $a_{\text{wall}}$ reaches:
$$\text{Target} = R_{\text{bonus}} + \gamma \max_a Q(S_{\text{fork}}, a) = 0.1200 + 0.9000 \times 0.1200 = 0.1200 + 0.1080 = \mathbf{0.2280} > 0.1200$$
The planning updates drive $Q(S_{\text{fork}}, a_{\text{wall}}) > Q(S_{\text{fork}}, a_{\text{long}})$. The greedy policy selects $a_{\text{wall}}$ in the real world, walks through the newly opened wall, records the shortcut transition in $\text{Model}$, and permanently adopts the faster route! $\blacksquare$

---

### Illustration 2: Dyna-Q Step-by-Step Cycle on a Gridworld (1 Real Update + $N = 3$ Planning Updates)
**Problem:**
Consider a 4-state deterministic corridor:
$$S_0 \xrightarrow{a_0} S_1 \xrightarrow{a_0} S_2 \xrightarrow{a_0} S_3 \text{ (Goal)}$$
Parameters:
- Learning rate: $\alpha = 0.2000$
- Discount factor: $\gamma = 0.9000$
- Terminal state: $S_3$ is absorbing with value $V(S_3) \equiv 0.0000$. Transition into $S_3$ yields reward $R = 10.0000$. All other transitions yield reward $R = 0.0000$.
- Model memory prior to the step already contains transitions:
  $$\text{Model}(S_0, a_0) = (R = 0.0000, S' = S_1, \text{Done} = \text{False})$$
  $$\text{Model}(S_1, a_0) = (R = 0.0000, S' = S_2, \text{Done} = \text{False})$$
- Current Q-table: $Q(S_0, a_0) = 0.0000, Q(S_1, a_0) = 0.0000, Q(S_2, a_0) = 0.0000$.

The agent executes real transition $(S_2, a_0) \to S_3$ with reward $R = 10.0000$.
Trace the exact arithmetic of:
1. Direct real-world Q-learning update.
2. Model learning registration.
3. $N = 3$ successive planning updates where the simulated sequence samples:
   - Planning Step 1: $(S_1, a_0)$
   - Planning Step 2: $(S_0, a_0)$
   - Planning Step 3: $(S_1, a_0)$

**Solution:**

*Step 1: Real-World Direct Q-Learning Update.*
The real transition is $(S = S_2, A = a_0, R = 10.0000, S' = S_3, \text{Done} = \text{True})$.
Since $S_3$ is terminal, continuation target is $0.0000$:
$$\text{Target}_{\text{real}} = R + \gamma (0.0000) = 10.0000 + 0.0000 = \mathbf{10.0000}$$
Perform TD update:
$$Q(S_2, a_0) \leftarrow Q(S_2, a_0) + \alpha \left[ \text{Target}_{\text{real}} - Q(S_2, a_0) \right]$$
$$= 0.0000 + 0.2000 \times (10.0000 - 0.0000) = 0.0000 + 2.0000 = \mathbf{2.0000}$$

*Step 2: Model Learning Registration.*
The agent registers the observed transition into memory:
$$\text{Model}(S_2, a_0) \leftarrow (R = 10.0000, S' = S_3, \text{Done} = \text{True})$$
Model dictionary now contains: $\{(S_0, a_0), (S_1, a_0), (S_2, a_0)\}$.

*Step 3: Planning Loop ($N = 3$).*

- **Planning Step 1 ($k = 1$): Sample $(S_1, a_0)$ from model.**
  Model query returns $R_k = 0.0000, S'_k = S_2, \text{Done} = \text{False}$.
  Compute simulated target:
  $$\text{Target}_{p1} = R_k + \gamma \max_a Q(S_2, a) = 0.0000 + 0.9000 \times Q(S_2, a_0) = 0.0000 + 0.9000 \times 2.0000 = \mathbf{1.8000}$$
  Update $Q(S_1, a_0)$:
  $$Q(S_1, a_0) \leftarrow Q(S_1, a_0) + \alpha \left[ \text{Target}_{p1} - Q(S_1, a_0) \right]$$
  $$= 0.0000 + 0.2000 \times (1.8000 - 0.0000) = \mathbf{0.3600}$$

- **Planning Step 2 ($k = 2$): Sample $(S_0, a_0)$ from model.**
  Model query returns $R_k = 0.0000, S'_k = S_1, \text{Done} = \text{False}$.
  Compute simulated target:
  $$\text{Target}_{p2} = R_k + \gamma \max_a Q(S_1, a) = 0.0000 + 0.9000 \times Q(S_1, a_0) = 0.0000 + 0.9000 \times 0.3600 = \mathbf{0.3240}$$
  Update $Q(S_0, a_0)$:
  $$Q(S_0, a_0) \leftarrow Q(S_0, a_0) + \alpha \left[ \text{Target}_{p2} - Q(S_0, a_0) \right]$$
  $$= 0.0000 + 0.2000 \times (0.3240 - 0.0000) = \mathbf{0.0648}$$

- **Planning Step 3 ($k = 3$): Sample $(S_1, a_0)$ from model again.**
  Model query returns $R_k = 0.0000, S'_k = S_2, \text{Done} = \text{False}$.
  Compute simulated target:
  $$\text{Target}_{p3} = R_k + \gamma \max_a Q(S_2, a) = 0.0000 + 0.9000 \times 2.0000 = \mathbf{1.8000}$$
  Update $Q(S_1, a_0)$ with its prior value $0.3600$:
  $$Q(S_1, a_0) \leftarrow 0.3600 + 0.2000 \times (1.8000 - 0.3600) = 0.3600 + 0.2000 \times 1.4400 = 0.3600 + 0.2880 = \mathbf{0.6480}$$

*Summary Arithmetic Table:*

| Phase | Event | Transition Sampled | Model Target Formula | Numerical Target | Updated Value |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Real Step** | Direct Q-Learning | $(S_2, a_0) \to S_3$ | $10.0000 + 0.9 \times 0.0000$ | $10.0000$ | $Q(S_2, a_0) = \mathbf{2.0000}$ |
| **Planning 1** | Hallucination 1 | $(S_1, a_0) \to S_2$ | $0.0000 + 0.9 \times 2.0000$ | $1.8000$ | $Q(S_1, a_0) = \mathbf{0.3600}$ |
| **Planning 2** | Hallucination 2 | $(S_0, a_0) \to S_1$ | $0.0000 + 0.9 \times 0.3600$ | $0.3240$ | $Q(S_0, a_0) = \mathbf{0.0648}$ |
| **Planning 3** | Hallucination 3 | $(S_1, a_0) \to S_2$ | $0.0000 + 0.9 \times 2.0000$ | $1.8000$ | $Q(S_1, a_0) = \mathbf{0.6480}$ |

*Key Takeaway:* In standard model-free Q-learning, propagating reward from $S_3$ back to $S_0$ requires three physical episodes ($S_2 \to S_3$, then in episode 2 $S_1 \to S_2$, then in episode 3 $S_0 \to S_1$). In Dyna-Q, all three states received non-zero credit within **a single physical step** while the agent remained stationed at the goal! $\blacksquare$

---

### Illustration 3: Dyna-Q+ Exploration Bonus Calculation and Path Switching Under Shortcut Discovery
**Problem:**
An autonomous mobile robot operates in a warehouse with two route options from starting depot $S_{\text{start}}$:
- **Detour Route ($a_{\text{detour}}$):** Navigates a circuitous 5-step perimeter corridor. Each step yields reward $0.0000$, and reaching the dropoff yields reward $R_{\text{goal}} = 1.0000$. With discount $\gamma = 0.9000$, the discounted value is:
  $$Q(S_{\text{start}}, a_{\text{detour}}) = \gamma^5 \times 1.0000 = (0.9000)^5 = \mathbf{0.5905}$$
- **Shortcut Route ($a_{\text{shortcut}}$):** A direct 2-step route with potential value $\gamma^2 \times 1.0000 = (0.9000)^2 = \mathbf{0.8100}$. However, it was historically blocked by a security gate. The robot's stored model records a bounce-back transition with reward $0.0000$:
  $$\text{Model}(S_{\text{start}}, a_{\text{shortcut}}) = (R = 0.0000, S' = S_{\text{start}})$$
  yielding baseline un-augmented value $Q_0(S_{\text{start}}, a_{\text{shortcut}}) = 0.0000$.

The gate is opened permanently at time $t = 0$.
The robot runs Dyna-Q+ with exploration bonus coefficient $\kappa = 0.0010$.
1. Determine the suboptimality gap $\Delta$ between the detour and the shortcut in the robot's memory.
2. Compute the bonus reward $R_{\text{bonus}}$ and effective planning target across timestamps $\tau \in \{10000, 100000, 250000, 348691, 400000\}$.
3. Calculate the exact physical step threshold $\tau^*$ at which the robot switches policies and executes $a_{\text{shortcut}}$.

**Solution:**

*Part 1: Suboptimality Gap.*
$$\Delta(S_{\text{start}}) = Q(S_{\text{start}}, a_{\text{detour}}) - Q_0(S_{\text{start}}, a_{\text{shortcut}}) = 0.5905 - 0.0000 = \mathbf{0.5905}$$

*Part 2: Numerical Bonus and Planning Target Evaluation.*
In Dyna-Q+, the simulated reward during planning is:
$$R_{\text{bonus}}(S_{\text{start}}, a_{\text{shortcut}}) = \hat{\mathcal{R}} + \kappa \sqrt{\tau} = 0.0000 + 0.0010 \times \sqrt{\tau}$$
Evaluating across the given elapsed counters:
- $\tau = 10,000$:
  $$\sqrt{10000} = 100.0000 \implies R_{\text{bonus}} = 0.0010 \times 100.0000 = \mathbf{0.1000} < 0.5905$$
  Planning target: $\text{Target} = 0.1000 + 0.9000 \times 0.0000 = \mathbf{0.1000} < Q(S_{\text{start}}, a_{\text{detour}})$. Policy exploits detour.
- $\tau = 100,000$:
  $$\sqrt{100000} \approx 316.2278 \implies R_{\text{bonus}} = 0.0010 \times 316.2278 = \mathbf{0.3162} < 0.5905$$
  Policy continues exploiting detour.
- $\tau = 250,000$:
  $$\sqrt{250000} = 500.0000 \implies R_{\text{bonus}} = 0.0010 \times 500.0000 = \mathbf{0.5000} < 0.5905$$
  Policy continues exploiting detour.
- $\tau = 348,691$:
  $$\sqrt{348691} \approx 590.4998 \implies R_{\text{bonus}} = 0.0010 \times 590.4998 = \mathbf{0.5905} = \Delta$$
  Planning target matches detour value exactly.
- $\tau = 400,000$:
  $$\sqrt{400000} \approx 632.4555 \implies R_{\text{bonus}} = 0.0010 \times 632.4555 = \mathbf{0.6325} > 0.5905$$
  Planning target strictly exceeds detour value!

*Part 3: Exact Critical Overtaking Threshold.*
By Derivation 11.21.3:
$$\tau^* = \left\lceil \left( \frac{\Delta}{\kappa} \right)^2 \right\rceil = \left\lceil \left( \frac{0.5905}{0.0010} \right)^2 \right\rceil = \left\lceil (590.5000)^2 \right\rceil = \lceil 348690.25 \rceil = \mathbf{348,691} \text{ steps}$$

*Physical Mechanism:* At step 348,691, the simulated Q-value under planning for $a_{\text{shortcut}}$ overtakes $a_{\text{detour}}$. At the next depot visit, the greedy decision rule executes $a_{\text{shortcut}}$. The robot finds the security gate open, reaches the dropoff in only 2 steps, and updates its model with $\text{Model}(S_{\text{start}}, a_{\text{shortcut}}) = (0.0000, S_{\text{shortcut1}})$. The true value jumps to $0.8100$, permanently establishing the shortcut as the optimal policy! $\blacksquare$

---

### Illustration 4: Compounding Error Bound Numerical Evaluation ($H \in \{1, 5, 10, 20\}$)
**Problem:**
A reinforcement learning researcher trains an autoregressive neural network transition model $\hat{\mathcal{P}}_{\theta}(s_{t+1} \mid s_t, a_t)$ on an environment with maximum immediate reward $R_{\max} = 1.0000$ and discount factor $\gamma = 0.9500$.
Validation testing reveals that the single-step model transition error is bounded by:
$$\epsilon_m \triangleq \max_{s, a} \|\mathcal{P}(\cdot \mid s, a) - \hat{\mathcal{P}}(\cdot \mid s, a)\|_1 = 0.0500$$
(equivalent to total variation error $\epsilon_{\text{TV}} = \frac{\epsilon_m}{2} = 0.0250$).
The reward model is assumed exact ($\epsilon_r = 0.0000$).

1. Using the theorems proven in Derivation 11.21.2, evaluate:
   - The single-step state distribution drift at step $H$: $\|d_H^\pi - \hat{d}_H^\pi\|_{\text{TV}} \le H \epsilon_m$.
   - The cumulative trajectory total variation error: $\sum_{t=1}^H \|d_t^\pi - \hat{d}_t^\pi\|_{\text{TV}} \le \frac{H(H+1)}{2} \epsilon_m$.
   - The discounted value performance error bound:
     $$|J_H(\pi) - \hat{J}_H(\pi)| \le \sum_{t=1}^H \gamma^{t-1} t \epsilon_m R_{\max}$$
   for rollout horizons $H \in \{1, 5, 10, 20\}$.
2. Compute the theoretical infinite-horizon Simulation Lemma bound $\frac{\gamma \epsilon_m R_{\max}}{(1 - \gamma)^2}$.
3. Explain why model-based algorithms (such as Dyna-Q and MBPO) truncate rollouts to short horizons.

**Solution:**

*Part 1: Step-by-Step Arithmetic for Horizon Values.*

- **Horizon $H = 1$ (Standard Dyna-Q Lookahead):**
  - Single-step TV drift: $1 \times 0.0500 = \mathbf{0.0500}$
  - Cumulative TV drift: $\frac{1 \times 2}{2} \times 0.0500 = \mathbf{0.0500}$
  - Value error bound:
    $$\gamma^0 (1) (0.0500) (1.0000) = 1.0000 \times 0.0500 = \mathbf{0.0500}$$

- **Horizon $H = 5$:**
  - Single-step TV drift: $5 \times 0.0500 = \mathbf{0.2500}$
  - Cumulative TV drift: $\frac{5 \times 6}{2} \times 0.0500 = 15 \times 0.0500 = \mathbf{0.7500}$
  - Value error bound:
    $$\sum_{t=1}^5 (0.95)^{t-1} t (0.0500)(1.0000) = 0.0500 \left[ 1(1) + 0.95(2) + 0.9025(3) + 0.857375(4) + 0.814506(5) \right]$$
    $$= 0.0500 \left[ 1.0000 + 1.9000 + 2.7075 + 3.4295 + 4.0725 \right] = 0.0500 \times 13.1095 = \mathbf{0.6555}$$

- **Horizon $H = 10$:**
  - Single-step TV drift: $10 \times 0.0500 = \mathbf{0.5000}$
  - Cumulative TV drift: $\frac{10 \times 11}{2} \times 0.0500 = 55 \times 0.0500 = \mathbf{2.7500}$
  - Value error bound:
    $$\sum_{t=1}^{10} (0.95)^{t-1} t (0.0500)(1.0000) = 0.0500 \times 40.7572 = \mathbf{2.0379}$$

- **Horizon $H = 20$:**
  - Single-step TV drift: $20 \times 0.0500 = \mathbf{1.0000}$ (Distribution completely diverges!)
  - Cumulative TV drift: $\frac{20 \times 21}{2} \times 0.0500 = 210 \times 0.0500 = \mathbf{10.5000}$
  - Value error bound:
    $$\sum_{t=1}^{20} (0.95)^{t-1} t (0.0500)(1.0000) = 0.0500 \times 113.2121 = \mathbf{5.6606}$$

*Part 2: Infinite-Horizon Bound Evaluation.*
Using the Simulation Lemma resolvent bound:
$$\text{Bound}_\infty = \frac{\gamma \epsilon_m R_{\max}}{(1 - \gamma)^2} = \frac{0.9500 \times 0.0500 \times 1.0000}{(1 - 0.9500)^2} = \frac{0.047500}{(0.0500)^2} = \frac{0.047500}{0.002500} = \mathbf{19.0000}$$

*Summary Comparison Table:*

| Rollout Horizon $H$ | Step TV Drift $\|d_H - \hat{d}_H\|_{\text{TV}}$ | Cumulative TV Drift $\sum \|d_t - \hat{d}_t\|_{\text{TV}}$ | Discounted Value Error $\vert J_H - \hat{J}_H \vert$ | Practical Reliability |
| :---: | :---: | :---: | :---: | :--- |
| **$H = 1$ (Dyna-Q)** | $\mathbf{0.0500}$ | $\mathbf{0.0500}$ | $\mathbf{0.0500}$ | **Extremely high:** Negligible bias |
| **$H = 5$** | $\mathbf{0.2500}$ | $\mathbf{0.7500}$ | $\mathbf{0.6555}$ | **Moderate:** Useful for short plans |
| **$H = 10$** | $\mathbf{0.5000}$ | $\mathbf{2.7500}$ | $\mathbf{2.0379}$ | **Degraded:** Significant compounding |
| **$H = 20$** | $\mathbf{1.0000}$ | $\mathbf{10.5000}$ | $\mathbf{5.6606}$ | **Catastrophic:** Hallucination / exploitation |
| **$H \to \infty$** | N/A | $\infty$ | $\mathbf{19.0000}$ | **Vacuous:** Error exceeds total return |

*Part 3: Algorithmic Insight.*
This arithmetic demonstrates why tabular Dyna-Q is engineered around **1-step lookahead ($H = 1$) from real visited states**. At $H = 1$, the value error is merely $0.0500$. At $H = 20$, compounding distribution drift produces an error of $5.6606$—larger than the maximum possible return from any 5-step trajectory! Algorithms that roll out multi-step models (e.g. MBPO, Janner et al. 2019) explicitly dynamically anneal horizon $H$ between $1$ and $5$ to avoid falling into the compounding hallucination trap. $\blacksquare$

---

### Illustration 5: Linear Gaussian Transition Model Parameter Estimation via Recursive Least Squares (RLS)
**Problem:**
Consider an agent learning a continuous 1D transition model:
$$s_{t+1} = a^* s_t + b^* a_t + w_t$$
where the ground-truth physical parameters are $a^* = 0.8000$ and $b^* = 0.5000$, and $w_t \sim \mathcal{N}(0, \sigma^2)$ is Gaussian process noise.
Let the regressor vector be $\mathbf{x}_t \triangleq [s_t, a_t]^\top \in \mathbb{R}^2$ and parameter vector be $\theta \triangleq [a, b]^\top \in \mathbb{R}^2$, so that the model prediction is $\hat{s}_{t+1} = \mathbf{x}_t^\top \theta$.

The agent estimates $\theta$ online using **Recursive Least Squares (RLS)** with:
- Initial parameter prior: $\theta_0 = [0.0000, 0.0000]^\top$
- Initial inverse covariance matrix: $\mathbf{P}_0 = 10.0000 \cdot \mathbf{I}_2 = \begin{bmatrix} 10.0000 & 0.0000 \\ 0.0000 & 10.0000 \end{bmatrix}$

Trace two successive real interaction steps:
- **Observation 1:** Current state $s_0 = 1.0000$, action taken $a_0 = 2.0000$, observed next state $s_1 = 1.8000$.
- **Observation 2:** Current state $s_1 = 1.8000$, action taken $a_1 = -1.0000$, observed next state $s_2 = 0.9400$.

Compute step-by-step:
1. Denominator scalar $1 + \mathbf{x}_t^\top \mathbf{P}_{t-1} \mathbf{x}_t$
2. Kalman gain vector $\mathbf{k}_t = \frac{\mathbf{P}_{t-1} \mathbf{x}_t}{1 + \mathbf{x}_t^\top \mathbf{P}_{t-1} \mathbf{x}_t}$
3. Prediction error (innovation) $e_t = s_{t+1} - \mathbf{x}_t^\top \theta_{t-1}$
4. Updated parameter vector $\theta_t = \theta_{t-1} + \mathbf{k}_t e_t$
5. Updated covariance matrix $\mathbf{P}_t = (\mathbf{I} - \mathbf{k}_t \mathbf{x}_t^\top) \mathbf{P}_{t-1}$

**Solution:**

*Update Step 1 ($t = 1$):*
Regressor: $\mathbf{x}_1 = [1.0000, 2.0000]^\top$, observed target $y_1 = s_1 = 1.8000$.

1. **Matrix-vector product $\mathbf{P}_0 \mathbf{x}_1$:**
   $$\mathbf{P}_0 \mathbf{x}_1 = \begin{bmatrix} 10.0000 & 0.0000 \\ 0.0000 & 10.0000 \end{bmatrix} \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix} = \begin{bmatrix} 10.0000 \\ 20.0000 \end{bmatrix}$$

2. **Denominator scalar:**
   $$\mathbf{x}_1^\top \mathbf{P}_0 \mathbf{x}_1 = 1.0000 \times 10.0000 + 2.0000 \times 20.0000 = 10.0000 + 40.0000 = 50.0000$$
   $$\text{Denom}_1 = 1.0000 + 50.0000 = \mathbf{51.0000}$$

3. **Gain vector $\mathbf{k}_1$:**
   $$\mathbf{k}_1 = \frac{1}{51.0000} \begin{bmatrix} 10.0000 \\ 20.0000 \end{bmatrix} = \begin{bmatrix} 0.196078 \\ 0.392157 \end{bmatrix}$$

4. **Prediction error $e_1$:**
   $$\hat{y}_1 = \mathbf{x}_1^\top \theta_0 = [1.0, 2.0] \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} = 0.0000$$
   $$e_1 = y_1 - \hat{y}_1 = 1.8000 - 0.0000 = \mathbf{1.8000}$$

5. **Updated parameter vector $\theta_1$:**
   $$\theta_1 = \begin{bmatrix} 0.0000 \\ 0.0000 \end{bmatrix} + 1.8000 \begin{bmatrix} 0.196078 \\ 0.392157 \end{bmatrix} = \begin{bmatrix} 0.352941 \\ 0.705882 \end{bmatrix}$$

6. **Updated covariance matrix $\mathbf{P}_1$:**
   Outer product $\mathbf{k}_1 \mathbf{x}_1^\top$:
   $$\mathbf{k}_1 \mathbf{x}_1^\top = \begin{bmatrix} 0.196078 \\ 0.392157 \end{bmatrix} \begin{bmatrix} 1.0000 & 2.0000 \end{bmatrix} = \begin{bmatrix} 0.196078 & 0.392157 \\ 0.392157 & 0.784314 \end{bmatrix}$$
   $$\mathbf{I} - \mathbf{k}_1 \mathbf{x}_1^\top = \begin{bmatrix} 0.803922 & -0.392157 \\ -0.392157 & 0.215686 \end{bmatrix}$$
   Multiplying by $\mathbf{P}_0 = 10 \mathbf{I}_2$:
   $$\mathbf{P}_1 = \begin{bmatrix} 8.039216 & -3.921569 \\ -3.921569 & 2.156863 \end{bmatrix}$$

---

*Update Step 2 ($t = 2$):*
Regressor: $\mathbf{x}_2 = [1.8000, -1.0000]^\top$, observed target $y_2 = s_2 = 0.9400$.

1. **Matrix-vector product $\mathbf{P}_1 \mathbf{x}_2$:**
   $$\mathbf{P}_1 \mathbf{x}_2 = \begin{bmatrix} 8.039216 & -3.921569 \\ -3.921569 & 2.156863 \end{bmatrix} \begin{bmatrix} 1.8000 \\ -1.0000 \end{bmatrix}$$
   $$= \begin{bmatrix} 8.039216(1.8) - 3.921569(-1.0) \\ -3.921569(1.8) + 2.156863(-1.0) \end{bmatrix} = \begin{bmatrix} 14.470589 + 3.921569 \\ -7.058824 - 2.156863 \end{bmatrix} = \begin{bmatrix} 18.392157 \\ -9.215686 \end{bmatrix}$$

2. **Denominator scalar:**
   $$\mathbf{x}_2^\top \mathbf{P}_1 \mathbf{x}_2 = 1.8000 \times 18.392157 + (-1.0000) \times (-9.215686) = 33.105883 + 9.215686 = 42.321569$$
   $$\text{Denom}_2 = 1.0000 + 42.321569 = \mathbf{43.321569}$$

3. **Gain vector $\mathbf{k}_2$:**
   $$\mathbf{k}_2 = \frac{1}{43.321569} \begin{bmatrix} 18.392157 \\ -9.215686 \end{bmatrix} = \begin{bmatrix} 0.424550 \\ -0.212727 \end{bmatrix}$$

4. **Prediction error $e_2$:**
   $$\hat{y}_2 = \mathbf{x}_2^\top \theta_1 = 1.8000(0.352941) + (-1.0000)(0.705882) = 0.635294 - 0.705882 = \mathbf{-0.070588}$$
   $$e_2 = y_2 - \hat{y}_2 = 0.9400 - (-0.070588) = \mathbf{1.010588}$$

5. **Updated parameter vector $\theta_2$:**
   $$\theta_2 = \begin{bmatrix} 0.352941 \\ 0.705882 \end{bmatrix} + 1.010588 \begin{bmatrix} 0.424550 \\ -0.212727 \end{bmatrix} = \begin{bmatrix} 0.352941 + 0.429045 \\ 0.705882 - 0.214980 \end{bmatrix} = \begin{bmatrix} \mathbf{0.7820} \\ \mathbf{0.4909} \end{bmatrix}$$

6. **Updated covariance matrix $\mathbf{P}_2$:**
   $$\mathbf{P}_2 = (\mathbf{I} - \mathbf{k}_2 \mathbf{x}_2^\top) \mathbf{P}_1 = \begin{bmatrix} \mathbf{0.2308} & \mathbf{-0.0091} \\ \mathbf{-0.0091} & \mathbf{0.1964} \end{bmatrix}$$

*Summary Comparison Table:*

| Step | State $s_t$ | Action $a_t$ | Next $s_{t+1}$ | Prediction $\hat{s}_{t+1}$ | Innovation $e_t$ | Estimated $\hat{a}$ (True: 0.8) | Estimated $\hat{b}$ (True: 0.5) | $\text{Tr}(\mathbf{P}_t)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | — | — | — | — | — | $0.0000$ | $0.0000$ | $20.0000$ |
| **1** | $1.0000$ | $2.0000$ | $1.8000$ | $0.0000$ | $+1.8000$ | $0.3529$ | $0.7059$ | $10.1961$ |
| **2** | $1.8000$ | $-1.0000$ | $0.9400$ | $-0.0706$ | $+1.0106$ | $\mathbf{0.7820}$ | $\mathbf{0.4909}$ | $\mathbf{0.4273}$ |

*Key Takeaway:* In only two observations, the recursive least squares estimator reduced the trace of parameter uncertainty from $20.0$ down to $0.4273$ and identified the dynamical coefficients to within $2.3\%$ ($\hat{a} = 0.7820 \approx 0.8$) and $1.8\%$ ($\hat{b} = 0.4909 \approx 0.5$). This demonstrates how continuous linear-Gaussian Dyna agents rapidly estimate environment models with minimal real-world interactions! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. World Models: Ha & Schmidhuber → DreamerV1/V2/V3 (2018–2023)
Dyna's latent model planning lineage culminates in the DreamerV series:
- **World Models (Ha & Schmidhuber, NeurIPS 2018):** A VAE compresses pixel observations into compact latent codes $z_t$; an MDN-RNN predicts $z_{t+1}$ from $(z_t, a_t)$; a tiny controller $\pi(a_t | z_t, h_t)$ is optimized **entirely inside hallucinated latent dreams** — never touching the real environment during policy search.
- **DreamerV2 (Hafner et al., ICLR 2021):** Replaces the Gaussian VAE with a categorical discrete VAE (RSSM), matches Rainbow DQN performance on Atari 100k with 40× fewer real environment interactions.
- **DreamerV3 (Hafner et al., Nature 2023):** Trained on 7 diverse domains (Atari, DMControl, Minecraft, DMLab, Crafter, Memory Maze, BSuite) without changing a **single hyperparameter**. Achieved diamond collection in Minecraft from scratch — the first agent to do so without human demonstrations.

### 2. MuZero: Dyna Without a Reconstruction Decoder (Silver et al., Nature 2020)
MuZero eliminates Dyna's requirement for a reconstruction-capable world model:
- **Implicit Latent Dynamics:** Instead of learning $\hat{s}_{t+1} = f(s_t, a_t)$ in pixel space, MuZero learns a latent transition $h_{t+1} = g_\theta(h_t, a_t)$ optimized only to predict future rewards and values — no image reconstruction loss at all.
- **MCTS Planning in Latent Space:** At each real step, MuZero runs 800 MCTS simulations through the latent model to select an action — exactly Dyna's "simulate-then-act" loop, but inside a learned non-Euclidean state representation.
- **Results:** Achieves superhuman performance in Chess (Elo 3,600), Go (AlphaZero parity), Shogi, and 57 Atari games — the broadest domain coverage of any single RL system.

### 3. MBPO: Short-Horizon Dyna for Continuous Control (Janner et al., NeurIPS 2019)
**Model-Based Policy Optimization** operationalizes Dyna's key insight — use the model only for short rollouts to avoid error compounding:
- **Branched Rollout Strategy:** From real transitions in the replay buffer, MBPO generates short model rollouts of $k \in [1, 15]$ steps. The imagined transitions are blended with real data in SAC's off-policy replay buffer.
- **Sample Efficiency:** Achieves the same MuJoCo performance as model-free SAC with 20–40× fewer real environment steps, demonstrating that even imperfect learned models dramatically accelerate learning when used conservatively.

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
