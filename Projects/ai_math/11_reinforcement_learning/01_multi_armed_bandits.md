# Multi-Armed Bandits & The Exploration-Exploitation Dilemma

---

## 1. Intuition & 101 Motivation

### The Fundamental Dilemma of Decision-Making Under Uncertainty
At the core of all reinforcement learning lies a foundational conflict:
- **Exploitation:** Choose the action currently known to yield the highest reward to maximize immediate payout.
- **Exploration:** Choose an uncertain or under-explored action to acquire new information that might reveal a significantly superior reward source.

If an agent **only exploits**, it falls into the **Greedy Lock-In Trap**: based on early noisy observations, it permanently locks onto a mediocre action, never discovering that an alternative option is vastly superior.
If an agent **only explores**, it continually gathers knowledge but wastes vast amounts of time pulling known low-yield levers, sacrificing cumulative reward.

```
                  THE EXPLOITATION-EXPLORATION TRADE-OFF
       Pure Exploitation (Greedy)                 Pure Exploration (Random)
  ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
  │ Locks onto first decent option  │       │ Tries everything uniformly      │
  │ High risk of suboptimality      │       │ Gathers perfect knowledge       │
  │ Regret = O(T) [Linear / Bad]    │       │ Regret = O(T) [Linear / Bad]    │
  └─────────────────────────────────┘       └─────────────────────────────────┘
                                     ▲
                                     │ Optimal Balance (UCB / Thompson)
                                     ▼
                      ┌─────────────────────────────┐
                      │ Regret = O(log T) [OPTIMAL] │
                      └─────────────────────────────┘
```

The **Multi-Armed Bandit** is the simplest mathematical formulation of this problem: a single non-associative state where an agent repeatedly selects from $K$ discrete actions, receiving scalar rewards from unknown stationary distributions.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The $K$-Armed Bandit Problem Setup
Let there be $K$ discrete actions $\mathcal{A} = \{1, 2, \dots, K\}$.
At each discrete time step $t = 1, 2, \dots, T$:
1. The agent selects an action $A_t = a \in \mathcal{A}$.
2. The environment emits a scalar reward $R_t \sim P(r \mid a)$.

The **true value** of action $a$ is its expected reward:
$$q_*(a) \equiv \mathbb{E}\left[ R_t \mid A_t = a \right]$$
The optimal action and optimal value are:
$$a^* = \operatorname{argmax}_{a \in \mathcal{A}} q_*(a), \quad q_*^* = q_*(a^*) = \max_{a \in \mathcal{A}} q_*(a)$$

---

### 2.2 Sample-Average Estimation & Incremental Updates

Let $N_t(a) = \sum_{i=1}^{t-1} \mathbb{I}(A_i = a)$ be the number of times action $a$ was selected prior to step $t$.
The sample-average estimate of action value is:
$$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}(A_i = a)}{N_t(a)}$$

#### Incremental Update Derivation
Computing the sample-average directly requires storing all past rewards. Instead, we derive a constant-memory recursive update.
Let $R_1, R_2, \dots, R_n$ denote the sequence of rewards observed for a specific action. After $n-1$ observations:
$$Q_n = \frac{1}{n-1} \sum_{i=1}^{n-1} R_i$$
Upon receiving the $n$-th reward $R_n$:
$$\begin{aligned}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^n R_i = \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right) \\
&= \frac{1}{n} \left( R_n + (n - 1) Q_n \right) \\
&= \frac{1}{n} \left( R_n + n Q_n - Q_n \right) \\
&= Q_n + \frac{1}{n} \left( R_n - Q_n \right)
\end{aligned}$$

The universal update rule follows the canonical format:
$$\mathbf{\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \cdot (\text{Target} - \text{OldEstimate})}$$
where $(R_n - Q_n)$ is the **Estimation Error**, and $\alpha_n = \frac{1}{n}$ is the learning step size.

> **Non-Stationary Environments:** If reward distributions drift over time, we replace $\frac{1}{n}$ with a constant step size $\alpha \in (0, 1]$:
> $$Q_{n+1} = Q_n + \alpha(R_n - Q_n) = (1 - \alpha)^n Q_1 + \sum_{i=1}^n \alpha(1 - \alpha)^{n-i} R_i$$
> This yields an **Exponentially Decaying Recency-Weighted Average** where past rewards fade exponentially at rate $(1 - \alpha)$.

---

### 2.3 Regret Formulation & The Lai-Robbins Lower Bound

To measure the efficiency of an exploration algorithm, we evaluate its **Regret**—the cumulative loss incurred by failing to always select the optimal action $a^*$.

#### Definitions
1. **Suboptimality Gap:** For each action $a$, define:
   $$\Delta_a \equiv q_*^* - q_*(a)$$
   For the optimal action $a^*$, $\Delta_{a^*} = 0$. For suboptimal actions, $\Delta_a > 0$.
2. **Total Expected Regret after $T$ steps:**
   $$L_T \equiv \mathbb{E}\left[ \sum_{t=1}^T \left( q_*^* - q_*(A_t) \right) \right] = T q_*^* - \sum_{t=1}^T \mathbb{E}[q_*(A_t)] = \sum_{a=1}^K \Delta_a \mathbb{E}\left[ N_T(a) \right]$$

Minimizing cumulative regret is strictly equivalent to minimizing the expected number of pulls $\mathbb{E}[N_T(a)]$ of suboptimal actions!

#### The Lai & Robbins Theorem (1985)
**Theorem (Theoretical Lower Bound):**
For any consistent policy (a policy that finds the optimal action with probability 1 as $T \to \infty$) operating on stationary reward distributions parameterized by distributions $p_a$:
$$\mathbf{\liminf_{T \to \infty} \frac{L_T}{\log T} \ge \sum_{a: \Delta_a > 0} \frac{\Delta_a}{D_{\text{KL}}(p_a \,\|\, p_*)} = \Omega(\log T)}$$
where $D_{\text{KL}}(p_a \,\|\, p_*)$ is the Kullback-Leibler divergence between suboptimal reward distribution $p_a$ and optimal distribution $p_*$.

**Implications:**
1. **Linear Regret ($\mathcal{O}(T)$):** Catastrophic failure. The algorithm continuously pulls suboptimal arms at a constant non-vanishing rate (e.g., pure random, constant $\epsilon$-greedy).
2. **Logarithmic Regret ($\mathcal{O}(\log T)$):** Asymptotically optimal. The algorithm pulls suboptimal arms only as much as mathematically necessary to verify they are suboptimal.

---

### 2.4 Exploration Algorithms

#### 1. $\epsilon$-Greedy Selection
With probability $1 - \epsilon$, exploit the best known action; with probability $\epsilon$, explore uniformly at random:
$$A_t = \begin{cases} \operatorname{argmax}_a Q_t(a) & \text{with probability } 1 - \epsilon \\ \text{UniformRandom}(\mathcal{A}) & \text{with probability } \epsilon \end{cases}$$
- **Constant $\epsilon$:** Suffers **linear regret** $\mathcal{O}(\epsilon T)$ because it wastes $\frac{\epsilon(K-1)}{K}$ fraction of all time steps pulling suboptimal arms even when $T \to \infty$.
- **Decaying $\epsilon_t = \min\left(1, \frac{c}{t}\right)$:** Can achieve theoretical $\mathcal{O}(\log T)$ regret, but tuning constant $c$ requires knowledge of the unknown gaps $\Delta_a$.

---

#### 2. Upper Confidence Bound (UCB1 - Auer, Cesa-Bianchi, & Fischer, 2002)

Instead of exploring purely randomly, **UCB implements Optimism in the Face of Uncertainty**: select the action that maximizes the upper bound of a statistical confidence interval.

#### First-Principles Derivation via Hoeffding's Inequality
Let reward observations $R_{a, 1}, \dots, R_{a, s} \in [0, 1]$ be independent and identically distributed with mean $q_*(a)$.
By **Hoeffding's Inequality**:
$$\mathbb{P}\left( q_*(a) > Q_t(a) + U_t(a) \right) \le e^{-2 N_t(a) U_t(a)^2}$$
where $U_t(a)$ is our uncertainty padding.
We want the probability of true value exceeding our upper bound to be tiny, say $p = t^{-4}$ (decaying with time $t$):
$$e^{-2 N_t(a) U_t(a)^2} = t^{-4}$$
Taking the natural logarithm of both sides:
$$-2 N_t(a) U_t(a)^2 = -4 \ln t \implies U_t(a)^2 = \frac{4 \ln t}{2 N_t(a)} = \frac{2 \ln t}{N_t(a)}$$
Solving for $U_t(a)$:
$$U_t(a) = \sqrt{\frac{2 \ln t}{N_t(a)}}$$

The **UCB1 Decision Rule** is:
$$\mathbf{A_t = \operatorname{argmax}_{a \in \mathcal{A}} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]}$$
where $c > 0$ is an exploration parameter (theoretically $c = \sqrt{2}$).

```
                     THE UCB1 CONFIDENCE BOUND MECHANISM
            Action Value Q(a)
                  ▲
                  │             Uncertainty Interval U(a) = sqrt(2 ln(t) / N(a))
                  │             ┌─────────┴─────────┐
         Arm 1:   │             [====== ● ======]   <-- High mean, well-sampled (N=100)
                  │                     │
                  │                     Q_1
                  │
         Arm 2:   │       [============= ● ===============]  <-- HIGHEST UCB SCORE!
                  │                      │                     (Low sample count N=4)
                  │                     Q_2                    CHOSEN BY UCB1!
                  │
         Arm 3:   │   [==== ● ====]  <-- Low mean, well-sampled (N=80)
                  │         │
                  │        Q_3
                  └────────────────────────────────────────────────────────► Reward
```

**Key Properties:**
1. If an action has high empirical mean $Q_t(a)$, its score is high (Exploitation).
2. If an action has been pulled rarely ($N_t(a)$ is small), its uncertainty $U_t(a)$ is large, boosting its score (Exploration).
3. Every time an action is pulled, $N_t(a)$ increases in the denominator, shrinking $U_t(a)$ and preventing infinite exploration of suboptimal arms.
4. **Regret Guarantee:** UCB1 achieves provable logarithmic regret $L_T \le 8 \sum_{a: \Delta_a > 0} \frac{\ln T}{\Delta_a} + \mathcal{O}(1) = \mathcal{O}(\log T)$.

---

#### 3. Thompson Sampling (Posterior Sampling - Thompson, 1933)

Thompson Sampling adopts a **Bayesian perspective**: rather than computing a deterministic upper bound, the agent maintains a posterior probability distribution over the expected reward of each arm and samples actions according to the probability that they are optimal.

For binary rewards $r \in \{0, 1\}$ (Bernoulli bandit), we place conjugate **Beta priors** on each arm:
$$P(q_*(a)) = \operatorname{Beta}(\alpha_a, \beta_a)$$
where $\alpha_a$ represents prior pseudo-successes and $\beta_a$ represents prior pseudo-failures. Initially, $\alpha_a = 1, \beta_a = 1$ (Uniform prior $\mathcal{U}[0, 1]$).

#### Algorithm (Thompson Sampling for Bernoulli Bandits):
1. For each arm $a \in \{1, \dots, K\}$:
   Sample a candidate value $\theta_a$ from its posterior:
   $$\theta_a \sim \operatorname{Beta}(\alpha_a, \beta_a)$$
2. Pull the arm with the highest sampled value:
   $$A_t = \operatorname{argmax}_{a \in \mathcal{A}} \theta_a$$
3. Observe reward $R_t \in \{0, 1\}$.
4. Update the posterior via conjugate Bayesian update:
   $$\text{If } R_t = 1 \implies \alpha_{A_t} \leftarrow \alpha_{A_t} + 1$$
   $$\text{If } R_t = 0 \implies \beta_{A_t} \leftarrow \beta_{A_t} + 1$$

Thompson Sampling naturally adapts its exploration: arms with wide, uncertain posteriors have a non-trivial probability of generating high samples $\theta_a$. As pulls accumulate, posteriors narrow into sharp delta functions around true means $q_*(a)$, transitioning seamlessly into pure exploitation.
It provably achieves the asymptotic **Lai-Robbins lower bound**!

---

## 3. Geometric & Physical Interpretation

### Uncertainty Ellipsoids & The Shrinking Confidence Horizon
In parameter space, each action's true mean $q_*(a)$ is bounded within a 1D confidence interval:
$$\mathcal{I}_t(a) = \left[ Q_t(a) - c \sqrt{\frac{\ln t}{N_t(a)}}, \, Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
1. As $t$ increases, the numerator $\ln t$ grows logarithmically (very slowly), while for the chosen arm, the denominator $N_t(a)$ grows linearly.
2. The radius of the interval shrinks at rate $\mathcal{O}\left(\sqrt{\frac{\ln t}{t}}\right) \to 0$.
3. Geometrically, the agent only pulls suboptimal arm $a$ when its upper confidence tip overlaps with the lower confidence tip of optimal arm $a^*$. As soon as the two intervals separate disjointly, suboptimal arm $a$ is never pulled again!

---

## 4. Real-World Analogy

### The Clinical Drug Trial & The Restaurant Dilemma
- **The Restaurant Dilemma:** You move to a new city with 10 restaurants. On day 1, you eat at Restaurant A and have an amazing pasta (Reward = 9/10).
  - **Greedy Strategy:** Eat at Restaurant A for the next 365 days. You eat well, but you never discover that Restaurant B across the street has a Michelin-starred chef with Reward = 10/10.
  - **Random Exploration:** Roll a 10-sided die every day. You discover every restaurant, but 90% of the year you eat awful food.
  - **UCB1 / Thompson Strategy:** Eat at Restaurant A for a few days. Once its uncertainty interval shrinks, the tantalizing uncertainty of Restaurant B pulls you over. You test B. If B is mediocre, its score drops and you return to A. If B is extraordinary, you settle into B!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace an exact multi-armed bandit with $K = 3$ arms running UCB1 with exploration coefficient $c = \sqrt{2} \approx 1.4142$ through exact hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Multi-Armed Bandit |
| :--- | :--- | :--- | :--- |
| $a$ | Arm Index | Integer $\in \{1, 2, 3\}$ | Index of the discrete slot machine lever |
| $q_*(a)$ | Ground Truth Mean | Scalar | Hidden expected payout (Arm 1: 0.8, Arm 2: 0.6, Arm 3: 0.3) |
| $N_t(a)$ | Pull Count | Integer | Number of times arm $a$ has been selected prior to step $t$ |
| $Q_t(a)$ | Empirical Mean Reward | Scalar | Running sample-average payout estimate for arm $a$ |
| $t$ | Timestep Counter | Integer | Total pulls executed across all arms |
| $U_t(a)$ | Uncertainty Bonus | Scalar | $c \sqrt{\frac{\ln t}{N_t(a)}}$ Hoeffding exploration bonus |
| $\text{UCB}_t(a)$ | Total Priority Score | Scalar | $Q_t(a) + U_t(a)$ (selected arm has maximum score) |
| $R_t$ | Observed Reward | Scalar | Stochastic reward emitted by environment |
| $\Delta_a$ | Suboptimality Gap | Scalar | $q_*^* - q_*(a)$ loss per pull |

---

### 5.2 Concrete Toy Setup

Let there be $K = 3$ arms with true hidden means:
$$q_*(1) = 0.8000 \quad (\text{Optimal Arm } a^*), \quad q_*(2) = 0.6000, \quad q_*(3) = 0.3000$$
Exploration constant: $c = \sqrt{2} \approx 1.4142$.

#### Initialization ($t = 1, 2, 3$):
To avoid division by zero ($N(a) = 0$), we initialize by pulling each arm once:
- Pull Arm 1 $\to$ Observe $R_1 = 0.9000 \implies N_4(1) = 1, Q_4(1) = 0.9000$
- Pull Arm 2 $\to$ Observe $R_2 = 0.5000 \implies N_4(2) = 1, Q_4(2) = 0.5000$
- Pull Arm 3 $\to$ Observe $R_3 = 0.2000 \implies N_4(3) = 1, Q_4(3) = 0.2000$

We now evaluate decisions starting at $t = 4$.

---

### 5.3 Step 1: Decision at $t = 4$ ($N_{\text{total}} = 3$)

Evaluate $\ln(t) = \ln(4) = 1.3863$.
For each arm $a$, compute bonus $U_4(a) = \sqrt{\frac{2 \ln 4}{N_4(a)}}$:
Since $N_4(1) = N_4(2) = N_4(3) = 1$:
$$U_4(a) = \sqrt{\frac{2 \times 1.3863}{1}} = \sqrt{2.7726} = \mathbf{1.6651}$$

Now compute UCB scores:
$$\text{UCB}_4(1) = Q_4(1) + U_4(1) = 0.9000 + 1.6651 = \mathbf{2.5651}$$
$$\text{UCB}_4(2) = Q_4(2) + U_4(2) = 0.5000 + 1.6651 = \mathbf{2.1651}$$
$$\text{UCB}_4(3) = Q_4(3) + U_4(3) = 0.2000 + 1.6651 = \mathbf{1.8651}$$

**Action Selected:** $\operatorname{argmax}_a \text{UCB}_4(a) = \mathbf{\text{Arm 1}}$.
**Environment Interaction:** Arm 1 emits noisy reward $R_4 = 0.7000$.

**Incremental Update for Arm 1:**
$$N_5(1) = N_4(1) + 1 = 1 + 1 = 2$$
$$Q_5(1) = Q_4(1) + \frac{1}{2}(R_4 - Q_4(1)) = 0.9000 + \frac{1}{2}(0.7000 - 0.9000) = 0.9000 - 0.1000 = \mathbf{0.8000}$$
Arms 2 and 3 remain unchanged ($N_5(2) = 1, Q_5(2) = 0.5000$, $N_5(3) = 1, Q_5(3) = 0.2000$).

---

### 5.4 Step 2: Decision at $t = 5$ ($N_{\text{total}} = 4$)

Evaluate $\ln(t) = \ln(5) = 1.6094$.
Compute bonuses:
- For Arm 1 ($N_5(1) = 2$):
  $$U_5(1) = \sqrt{\frac{2 \times 1.6094}{2}} = \sqrt{1.6094} = \mathbf{1.2686}$$
  $$\text{UCB}_5(1) = Q_5(1) + U_5(1) = 0.8000 + 1.2686 = \mathbf{2.0686}$$
- For Arm 2 ($N_5(2) = 1$):
  $$U_5(2) = \sqrt{\frac{2 \times 1.6094}{1}} = \sqrt{3.2189} = \mathbf{1.7941}$$
  $$\text{UCB}_5(2) = Q_5(2) + U_5(2) = 0.5000 + 1.7941 = \mathbf{2.2941}$$
- For Arm 3 ($N_5(3) = 1$):
  $$U_5(3) = \sqrt{\frac{2 \times 1.6094}{1}} = 1.7941$$
  $$\text{UCB}_5(3) = Q_5(3) + U_5(3) = 0.2000 + 1.7941 = \mathbf{1.9941}$$

Notice what happened: Even though Arm 1 had the higher empirical mean ($0.80 > 0.50$), its uncertainty shrunk from $1.6651 \to 1.2686$, while Arm 2 remained under-explored!
**Action Selected:** $\operatorname{argmax}_a \text{UCB}_5(a) = \mathbf{\text{Arm 2}}$.
**Environment Interaction:** Arm 2 emits reward $R_5 = 0.6000$.

**Incremental Update for Arm 2:**
$$N_6(2) = 1 + 1 = 2$$
$$Q_6(2) = 0.5000 + \frac{1}{2}(0.6000 - 0.5000) = 0.5000 + 0.0500 = \mathbf{0.5500}$$

---

### 5.5 Step 3: Decision at $t = 6$ ($N_{\text{total}} = 5$)

Evaluate $\ln(t) = \ln(6) = 1.7918$.
Compute bonuses:
- For Arm 1 ($N_6(1) = 2$):
  $$U_6(1) = \sqrt{\frac{2 \times 1.7918}{2}} = \sqrt{1.7918} = \mathbf{1.3386}$$
  $$\text{UCB}_6(1) = 0.8000 + 1.3386 = \mathbf{2.1386}$$
- For Arm 2 ($N_6(2) = 2$):
  $$U_6(2) = \sqrt{\frac{2 \times 1.7918}{2}} = \mathbf{1.3386}$$
  $$\text{UCB}_6(2) = 0.5500 + 1.3386 = \mathbf{1.8886}$$
- For Arm 3 ($N_6(3) = 1$):
  $$U_6(3) = \sqrt{\frac{2 \times 1.7918}{1}} = \sqrt{3.5835} = \mathbf{1.8930}$$
  $$\text{UCB}_6(3) = 0.2000 + 1.8930 = \mathbf{2.0930}$$

Ranking:
$\text{UCB}_6(1) = 2.1386 > \text{UCB}_6(3) = 2.0930 > \text{UCB}_6(2) = 1.8886$.
Arm 1 is selected again!

---

### 5.6 Summary Arithmetic State Grid

| Step $t$ | Arm 1 $(N, Q, U, \text{UCB})$ | Arm 2 $(N, Q, U, \text{UCB})$ | Arm 3 $(N, Q, U, \text{UCB})$ | Action Chosen | Observed Reward |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 4$** | $(1, 0.900, 1.665, \mathbf{2.565})$ | $(1, 0.500, 1.665, 2.165)$ | $(1, 0.200, 1.665, 1.865)$ | **Arm 1** | $R_4 = 0.700$ |
| **$t = 5$** | $(2, 0.800, 1.269, 2.069)$ | $(1, 0.500, 1.794, \mathbf{2.294})$ | $(1, 0.200, 1.794, 1.994)$ | **Arm 2** | $R_5 = 0.600$ |
| **$t = 6$** | $(2, 0.800, 1.339, \mathbf{2.139})$ | $(2, 0.550, 1.339, 1.889)$ | $(1, 0.200, 1.893, 2.093)$ | **Arm 1** | $R_6 = 0.850$ |

Every single arithmetic step is completely transparent and verified by code!

---

## 6. Solved Illustrations

### Illustration 1: Mathematical Proof of Linear Regret for Constant $\epsilon$-Greedy
**Problem:** Prove that an $\epsilon$-greedy algorithm with constant $\epsilon > 0$ inevitably suffers linear cumulative regret $L_T = \Omega(T)$, violating the Lai-Robbins bound.
**Solution:**
At every step $t$, the agent selects a random action with probability $\epsilon$.
Among the $K$ actions, at least $K - 1$ actions are suboptimal.
The probability of selecting a suboptimal action at any time step $t$ is bounded below by:
$$\mathbb{P}(A_t \ne a^*) \ge \epsilon \left( \frac{K - 1}{K} \right)$$
Let $\Delta_{\min} = \min_{a \ne a^*} \Delta_a > 0$ be the minimum non-zero suboptimality gap.
The expected cumulative regret over $T$ steps is:
$$L_T = \sum_{t=1}^T \mathbb{E}[\Delta_{A_t}] \ge \sum_{t=1}^T \mathbb{P}(A_t \ne a^*) \Delta_{\min} \ge T \cdot \epsilon \left( \frac{K - 1}{K} \right) \Delta_{\min}$$
Since $\epsilon, K, \Delta_{\min}$ are all strictly positive constants independent of $T$:
$$L_T = \mathcal{O}(T) = \Omega(T)$$
Cumulative regret grows linearly with $T$. The average regret per step $\frac{L_T}{T} \ge \epsilon \frac{K-1}{K} \Delta_{\min} > 0$ never decays to zero! $\blacksquare$

---

### Illustration 2: Thompson Sampling Bayesian Conjugate Update
**Problem:** A medical trial tests a drug with unknown recovery rate $q_* \in [0, 1]$. We use prior $\text{Beta}(1, 1)$. Over 5 patients, the outcomes are $[1, 0, 1, 1, 1]$. Write down the analytical posterior distribution and compute its mean and variance.
**Solution:**
Prior: $\alpha_0 = 1, \beta_0 = 1$.
Successes $k = 4$, Failures $m = 1$.
The posterior is:
$$P(q \mid \text{data}) = \operatorname{Beta}(\alpha_0 + k, \, \beta_0 + m) = \operatorname{Beta}(1 + 4, \, 1 + 1) = \mathbf{\operatorname{Beta}(5, 2)}$$
Expected value (Bayesian point estimate):
$$\mathbb{E}[q] = \frac{\alpha}{\alpha + \beta} = \frac{5}{5 + 2} = \frac{5}{7} \approx \mathbf{0.7143}$$
Posterior variance:
$$\operatorname{Var}(q) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)} = \frac{5 \times 2}{(7)^2 (8)} = \frac{10}{49 \times 8} = \frac{10}{392} \approx \mathbf{0.0255}$$
The standard deviation shrunk from prior $\sqrt{1/12} \approx 0.2887$ down to $\sqrt{0.0255} \approx 0.1597$.

---

## 7. Deep Learning Connection & Application

### 1. Recommender Systems (TikTok, YouTube, Netflix)
Modern recommendation systems cannot afford pure A/B testing (which wastes millions of impressions on losing models). They use **Contextual Bandits** (e.g., LinUCB):
Given user embedding context vector $\mathbf{x} \in \mathbb{R}^d$, the expected click-through rate of video $a$ is modeled as $\mathbf{w}_a^\top \mathbf{x}$. UCB computes the ridge regression covariance confidence ellipsoid:
$$A_t = \operatorname{argmax}_a \left[ \hat{\mathbf{w}}_a^\top \mathbf{x} + \alpha \sqrt{\mathbf{x}^\top (\mathbf{A}_a)^{-1} \mathbf{x}} \right]$$
This guarantees that fresh content with high uncertainty is tested safely without tanking user engagement.

### 2. Monte Carlo Tree Search (AlphaZero & DeepSeek-R1)
In AlphaGo, AlphaZero, and modern reasoning model tree searches, the node selection phase uses the **Polynomial Upper Confidence Trees (PUCT)** formula:
$$a^* = \operatorname{argmax}_a \left[ Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$
This is mathematically identical to UCB1, modulated by prior policy probability $P(s, a)$!

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Exact numerical verification of the Part 5 hand arithmetic (UCB scores, decisions, rewards, and incremental $Q$-value updates matching hand calculations to $< 10^{-6}$).
2. 10-Armed Gaussian Bandit environment benchmark.
3. Full comparative experiment comparing:
   - Pure Greedy
   - $\epsilon$-Greedy ($\epsilon = 0.1$ and $\epsilon = 0.01$)
   - UCB1 ($c = \sqrt{2}$)
   - Thompson Sampling (Beta-Bernoulli)
4. Verification of the Lai-Robbins logarithmic regret bound: asserting that UCB1 and Thompson Sampling achieve sub-linear $\mathcal{O}(\log T)$ regret while constant $\epsilon$-greedy scales linearly $\mathcal{O}(T)$.

See implementation in:
[`11_reinforcement_learning/code/01_multi_armed_bandits.py`](./code/01_multi_armed_bandits.py)
