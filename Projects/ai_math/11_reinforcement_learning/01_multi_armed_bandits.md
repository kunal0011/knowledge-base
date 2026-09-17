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

### 2.5 Rigorous Mathematical Derivations

#### Derivation 11.1.1: Complete First-Principles Regret Bound of UCB1 via Hoeffding's Inequality

**1. Context and Assumptions:**
Let there be $K$ discrete actions $\mathcal{A} = \{1, \dots, K\}$.
Assume reward distributions are bounded within the unit interval: $R_{a, s} \in [0, 1]$ for all arms $a$ and trial pulls $s \ge 1$.
Let $q_*(a) \equiv \mathbb{E}[R_{a}]$ denote the true expected reward of arm $a$, with optimal value $q_*^* = \max_a q_*(a)$ achieved at arm $a^*$.
For every suboptimal arm $a$, the suboptimality gap is $\Delta_a \equiv q_*^* - q_*(a) > 0$.
The UCB1 index at step $t$ after $N_t(a)$ pulls of arm $a$ is:
$$U_t(a) = \sqrt{\frac{2 \ln t}{N_t(a)}}, \qquad \text{Score}_t(a) = Q_t(a) + U_t(a)$$

**2. Step 1: Decomposition of the Selection Condition:**
Suppose suboptimal arm $a \ne a^*$ is selected at step $t$ ($A_t = a$).
This event implies that arm $a$'s UCB score exceeded the UCB score of the optimal arm $a^*$:
$$Q_t(a) + U_t(a) \ge Q_t(a^*) + U_t(a^*)$$
We show that this inequality strictly requires at least one of the following three elementary events to occur:
1. $E_1$: The optimal arm's sample average severely underestimates its true value:
   $$Q_t(a^*) \le q_*^* - U_t(a^*)$$
2. $E_2$: The suboptimal arm's sample average severely overestimates its true value:
   $$Q_t(a) \ge q_*(a) + U_t(a)$$
3. $E_3$: The true suboptimality gap is smaller than twice the suboptimal arm's uncertainty:
   $$q_*^* < q_*(a) + 2 U_t(a)$$

*Proof of Exhaustion:*
Assume that both $E_1$ and $E_2$ are false, meaning:
$$Q_t(a^*) > q_*^* - U_t(a^*) \quad \text{and} \quad Q_t(a) < q_*(a) + U_t(a)$$
Then:
$$Q_t(a^*) + U_t(a^*) > q_*^*$$
$$Q_t(a) + U_t(a) < q_*(a) + 2 U_t(a)$$
Since $A_t = a$, we have $Q_t(a) + U_t(a) \ge Q_t(a^*) + U_t(a^*)$. Therefore:
$$q_*^* < Q_t(a^*) + U_t(a^*) \le Q_t(a) + U_t(a) < q_*(a) + 2 U_t(a)$$
which implies $q_*^* < q_*(a) + 2 U_t(a)$, exactly event $E_3$.
Thus, $\{A_t = a\} \subseteq E_1 \cup E_2 \cup E_3$.

**3. Step 2: Bounding the Deterministic Event $E_3$:**
Event $E_3$ states that:
$$q_*^* - q_*(a) < 2 U_t(a) \implies \Delta_a < 2 \sqrt{\frac{2 \ln t}{N_t(a)}}$$
Squaring both sides:
$$\Delta_a^2 < \frac{8 \ln t}{N_t(a)} \implies N_t(a) < \frac{8 \ln t}{\Delta_a^2} \le \frac{8 \ln T}{\Delta_a^2}$$
Define the integer threshold:
$$u = \left\lceil \frac{8 \ln T}{\Delta_a^2} \right\rceil$$
If $N_t(a) \ge u$, then event $E_3$ **cannot possibly occur**.

**4. Step 3: Bounding the Probabilities of $E_1$ and $E_2$ via Hoeffding's Inequality:**
By Hoeffding's Inequality for $s$ independent bounded random variables in $[0, 1]$ with sample average $\bar{X}_s$ and true mean $\mu$:
$$\mathbb{P}(\bar{X}_s - \mu \ge \epsilon) \le e^{-2 s \epsilon^2}, \qquad \mathbb{P}(\mu - \bar{X}_s \ge \epsilon) \le e^{-2 s \epsilon^2}$$
Substituting $\epsilon = \sqrt{\frac{2 \ln t}{s}}$:
$$\mathbb{P}\left( \bar{X}_s - \mu \ge \sqrt{\frac{2 \ln t}{s}} \right) \le \exp\left( -2 s \frac{2 \ln t}{s} \right) = e^{-4 \ln t} = t^{-4}$$
Because the number of pulls $N_t(a)$ is a random variable between $1$ and $t-1$, we apply the union bound over all possible sample counts $s \in \{1, \dots, t-1\}$:
$$\mathbb{P}(E_1) \le \sum_{s=1}^t \mathbb{P}\left( q_*^* - \bar{R}_{a^*, s} \ge \sqrt{\frac{2 \ln t}{s}} \right) \le \sum_{s=1}^t t^{-4} = t \cdot t^{-4} = t^{-3}$$
Similarly, for event $E_2$:
$$\mathbb{P}(E_2) \le \sum_{s=1}^t \mathbb{P}\left( \bar{R}_{a, s} - q_*(a) \ge \sqrt{\frac{2 \ln t}{s}} \right) \le t \cdot t^{-4} = t^{-3}$$

**5. Step 4: Summing Total Expected Suboptimal Pulls:**
The expected number of pulls of suboptimal arm $a$ up to horizon $T$ is:
$$\mathbb{E}[N_T(a)] = \sum_{t=1}^T \mathbb{I}(A_t = a) \le u + \sum_{t=u+1}^T \mathbb{I}(A_t = a, N_{t-1}(a) \ge u)$$
For $t > u$, event $E_3$ is false, so $\{A_t = a\} \subseteq E_1 \cup E_2$:
$$\mathbb{E}[N_T(a)] \le u + \sum_{t=u+1}^T \left( \mathbb{P}(E_1) + \mathbb{P}(E_2) \right) \le \left\lceil \frac{8 \ln T}{\Delta_a^2} \right\rceil + \sum_{t=1}^T 2 t^{-3} \le \frac{8 \ln T}{\Delta_a^2} + 1 + 2 \sum_{t=1}^\infty t^{-2}$$
Since $\sum_{t=1}^\infty t^{-2} = \frac{\pi^2}{6} \approx 1.6449$:
$$\mathbf{\mathbb{E}[N_T(a)] \le \frac{8 \ln T}{\Delta_a^2} + 1 + \frac{\pi^2}{3} \approx \frac{8 \ln T}{\Delta_a^2} + 4.2899}$$

**6. Step 5: Finite-Time Total Expected Regret Bound:**
Multiplying by gaps $\Delta_a$ across all suboptimal arms:
$$\mathbf{L_T = \sum_{a: \Delta_a > 0} \Delta_a \mathbb{E}[N_T(a)] \le \sum_{a: \Delta_a > 0} \left( \frac{8 \ln T}{\Delta_a} \right) + \left( 1 + \frac{\pi^2}{3} \right) \sum_{a=1}^K \Delta_a = \mathcal{O}(\log T)}$$
This completes the rigorous finite-time proof of UCB1's logarithmic regret bound.

---

#### Derivation 11.1.2: Bayesian Conjugate Beta-Bernoulli Posterior and Thompson Sampling Decision Theory

**1. Context and Assumptions:**
Consider an arm with binary rewards $R \in \{0, 1\}$ following a Bernoulli distribution with unknown parameter $\theta \in [0, 1]$:
$$P(R = r \mid \theta) = \theta^r (1 - \theta)^{1 - r}, \quad r \in \{0, 1\}$$
Assume a prior distribution over $\theta$ given by the Beta distribution:
$$p(\theta; \alpha, \beta) = \frac{1}{\mathrm{B}(\alpha, \beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}$$
where $\mathrm{B}(\alpha, \beta) \triangleq \int_0^1 u^{\alpha - 1} (1 - u)^{\beta - 1} du = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$ is the Beta function.

**2. Step 1: Conjugate Posterior Derivation:**
Suppose we observe a sequence of $n$ independent trials yielding $k$ successes ($R=1$) and $m = n - k$ failures ($R=0$).
The joint likelihood of the observations $\mathcal{D} = \{r_1, \dots, r_n\}$ is:
$$P(\mathcal{D} \mid \theta) = \prod_{i=1}^n \theta^{r_i} (1 - \theta)^{1 - r_i} = \theta^{\sum r_i} (1 - \theta)^{n - \sum r_i} = \theta^k (1 - \theta)^m$$
By Bayes' theorem, the posterior probability density is:
$$p(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) p(\theta)}{\int_0^1 P(\mathcal{D} \mid \theta') p(\theta') d\theta'} \propto \left[ \theta^k (1 - \theta)^m \right] \cdot \left[ \theta^{\alpha - 1} (1 - \theta)^{\beta - 1} \right] = \theta^{(\alpha + k) - 1} (1 - \theta)^{(\beta + m) - 1}$$
Recognizing the kernel of the Beta distribution:
$$\mathbf{p(\theta \mid \mathcal{D}) = \operatorname{Beta}(\alpha + k, \, \beta + m)}$$
The Beta family is strictly conjugate to the Bernoulli likelihood: each success increments $\alpha$ by 1, and each failure increments $\beta$ by 1.

**3. Step 2: Posterior Mean and Variance:**
To compute the $m$-th raw moment $\mathbb{E}[\theta^m]$:
$$\mathbb{E}[\theta^m] = \int_0^1 \theta^m \frac{\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}}{\mathrm{B}(\alpha, \beta)} d\theta = \frac{\mathrm{B}(\alpha + m, \beta)}{\mathrm{B}(\alpha, \beta)} = \frac{\Gamma(\alpha + m)\Gamma(\beta)}{\Gamma(\alpha + \beta + m)} \cdot \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} = \frac{\Gamma(\alpha + m)\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\alpha + \beta + m)}$$
For the first moment ($m = 1$), using $\Gamma(z + 1) = z \Gamma(z)$:
$$\mathbb{E}[\theta] = \frac{\alpha \Gamma(\alpha)\Gamma(\alpha + \beta)}{\Gamma(\alpha)(\alpha + \beta)\Gamma(\alpha + \beta)} = \mathbf{\frac{\alpha}{\alpha + \beta}}$$
For the second moment ($m = 2$):
$$\mathbb{E}[\theta^2] = \frac{(\alpha + 1)\alpha \Gamma(\alpha)\Gamma(\alpha + \beta)}{\Gamma(\alpha)(\alpha + \beta + 1)(\alpha + \beta)\Gamma(\alpha + \beta)} = \frac{\alpha(\alpha + 1)}{(\alpha + \beta)(\alpha + \beta + 1)}$$
The variance is:
$$\operatorname{Var}(\theta) = \mathbb{E}[\theta^2] - (\mathbb{E}[\theta])^2 = \frac{\alpha(\alpha + 1)}{(\alpha + \beta)(\alpha + \beta + 1)} - \frac{\alpha^2}{(\alpha + \beta)^2} = \frac{\alpha(\alpha + \beta)(\alpha + 1) - \alpha^2(\alpha + \beta + 1)}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$
Expanding the numerator:
$$\alpha(\alpha^2 + \alpha\beta + \alpha + \beta) - \alpha^3 - \alpha^2\beta - \alpha^2 = \alpha^3 + \alpha^2\beta + \alpha^2 + \alpha\beta - \alpha^3 - \alpha^2\beta - \alpha^2 = \alpha\beta$$
Therefore:
$$\mathbf{\operatorname{Var}(\theta) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}}$$
As total pulls $n = \alpha + \beta \to \infty$, variance decays at rate $\mathcal{O}(n^{-1}) \to 0$.

**4. Step 3: Closed-Form Probability of Action Selection in 2-Armed Bandits:**
Let Arm 1 have posterior $\theta_1 \sim \operatorname{Beta}(\alpha_1, \beta_1)$ and Arm 2 have posterior $\theta_2 \sim \operatorname{Beta}(\alpha_2, \beta_2)$.
Thompson Sampling selects Arm 1 with probability:
$$\mathbb{P}(A = 1) = \mathbb{P}(\theta_1 > \theta_2) = \int_0^1 p_1(\theta_1) \left[ \int_0^{\theta_1} p_2(\theta_2) d\theta_2 \right] d\theta_1 = \int_0^1 p_1(\theta_1) I_{\theta_1}(\alpha_2, \beta_2) d\theta_1$$
where $I_x(a, b) = \frac{1}{\mathrm{B}(a, b)} \int_0^x u^{a-1} (1-u)^{b-1} du$ is the regularized incomplete Beta function.
For integer values of $\beta_2$, applying integration by parts yields the exact finite sum:
$$\mathbb{P}(\theta_1 > \theta_2) = \sum_{j=0}^{\beta_2 - 1} \frac{\mathrm{B}(\alpha_1 + \alpha_2 + j, \, \beta_1 + \beta_2 - 1 - j)}{(\alpha_2 + j) \, \mathrm{B}(\alpha_1, \beta_1) \, \mathrm{B}(\alpha_2, \beta_2)}$$

---

#### Derivation 11.1.3: Gradient Bandit Algorithm and Policy Gradient Derivation via Softmax Action Preferences

**1. Context and Assumptions:**
Let there be $K$ actions $\mathcal{A} = \{1, \dots, K\}$ with numerical action preferences $H_t(a) \in \mathbb{R}$.
The probability of selecting action $a$ is defined by the Softmax distribution (Gibbs policy):
$$\pi_t(a) \triangleq \mathbb{P}(A_t = a) = \frac{e^{H_t(a)}}{\sum_{b=1}^K e^{H_t(b)}}$$
The performance metric to maximize is the expected reward:
$$\eta(\mathbf{H}) \triangleq \mathbb{E}[R_t] = \sum_{x=1}^K \pi_t(x) q_*(x)$$
where $q_*(x) \equiv \mathbb{E}[R_t \mid A_t = x]$ is the true action value.

**2. Step 1: Derivative of the Softmax Action Probabilities:**
Compute the partial derivative of $\pi_t(x)$ with respect to preference $H_t(a)$:
$$\frac{\partial \pi_t(x)}{\partial H_t(a)} = \frac{\partial}{\partial H_t(a)} \left[ \frac{e^{H_t(x)}}{\sum_{b=1}^K e^{H_t(b)}} \right]$$
Applying the quotient rule:
- Case 1 ($x = a$):
  $$\frac{\partial \pi_t(a)}{\partial H_t(a)} = \frac{e^{H_t(a)}\sum e^{H_t(b)} - e^{H_t(a)}e^{H_t(a)}}{\left( \sum e^{H_t(b)} \right)^2} = \frac{e^{H_t(a)}}{\sum e^{H_t(b)}} \left( 1 - \frac{e^{H_t(a)}}{\sum e^{H_t(b)}} \right) = \pi_t(a)(1 - \pi_t(a))$$
- Case 2 ($x \ne a$):
  $$\frac{\partial \pi_t(x)}{\partial H_t(a)} = \frac{0 - e^{H_t(x)} e^{H_t(a)}}{\left( \sum e^{H_t(b)} \right)^2} = -\frac{e^{H_t(x)}}{\sum e^{H_t(b)}} \frac{e^{H_t(a)}}{\sum e^{H_t(b)}} = -\pi_t(x)\pi_t(a)$$
Combining both cases using the Kronecker delta $\mathbb{I}(x = a)$:
$$\mathbf{\frac{\partial \pi_t(x)}{\partial H_t(a)} = \pi_t(x) \left( \mathbb{I}(x = a) - \pi_t(a) \right)}$$

**3. Step 2: Policy Gradient of Expected Reward:**
Differentiating the objective $\eta(\mathbf{H})$ with respect to $H_t(a)$:
$$\frac{\partial \eta(\mathbf{H})}{\partial H_t(a)} = \frac{\partial}{\partial H_t(a)} \left[ \sum_{x=1}^K \pi_t(x) q_*(x) \right] = \sum_{x=1}^K q_*(x) \frac{\partial \pi_t(x)}{\partial H_t(a)} = \sum_{x=1}^K q_*(x) \pi_t(x) \left( \mathbb{I}(x = a) - \pi_t(a) \right)$$

**4. Step 3: Baseline Invariance Lemma:**
Let $B_t$ be any arbitrary baseline that does **not** depend on the action index $x$.
Consider the term $\sum_{x=1}^K B_t \frac{\partial \pi_t(x)}{\partial H_t(a)}$:
$$\sum_{x=1}^K B_t \pi_t(x) \left( \mathbb{I}(x = a) - \pi_t(a) \right) = B_t \left( \sum_{x=1}^K \pi_t(x) \mathbb{I}(x = a) - \pi_t(a) \sum_{x=1}^K \pi_t(x) \right) = B_t \left( \pi_t(a) - \pi_t(a)(1) \right) = 0$$
Because the sum of probabilities is identically 1, the gradient of the baseline sum is identically zero!
Therefore, we can subtract $B_t$ freely inside the sum without altering the gradient:
$$\frac{\partial \eta(\mathbf{H})}{\partial H_t(a)} = \sum_{x=1}^K \left( q_*(x) - B_t \right) \pi_t(x) \left( \mathbb{I}(x = a) - \pi_t(a) \right)$$

**5. Step 4: Expectation Representation & Stochastic Update:**
By definition of mathematical expectation under policy $A_t \sim \pi_t$:
$$\frac{\partial \eta(\mathbf{H})}{\partial H_t(a)} = \mathbb{E}_{A_t \sim \pi_t} \left[ \left( q_*(A_t) - B_t \right) \left( \mathbb{I}(A_t = a) - \pi_t(a) \right) \right]$$
Since $\mathbb{E}[R_t \mid A_t] = q_*(A_t)$, by the Law of Total Expectation:
$$\mathbf{\frac{\partial \eta(\mathbf{H})}{\partial H_t(a)} = \mathbb{E}\left[ \left( R_t - \bar{R}_t \right) \left( \mathbb{I}(A_t = a) - \pi_t(a) \right) \right]}$$
where $\bar{R}_t = \frac{1}{t}\sum_{i=1}^t R_i$ is the running average reward baseline.
The stochastic gradient ascent update rule with step size $\alpha > 0$ is:
$$\mathbf{H_{t+1}(a) = H_t(a) + \alpha \left( R_t - \bar{R}_t \right) \left( \mathbb{I}(A_t = a) - \pi_t(a) \right)}$$
- If the chosen action $A_t = a$ yields reward $R_t > \bar{R}_t$ (above average), its preference $H(a)$ increases by $\alpha(R_t - \bar{R}_t)(1 - \pi_t(a))$, and all other arm preferences decrease.
- If $R_t < \bar{R}_t$ (below average), its preference decreases, boosting unchosen arms.
This is the mathematical origin of the **REINFORCE policy gradient algorithm** in general RL!

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

### Illustration 3: 2-Armed Bernoulli Thompson Sampling Hand Trace with Exact Win Probabilities

**Problem:**
Consider a 2-armed Bernoulli bandit with true hidden payout rates:
$$\theta_1^* = 0.7000 \quad (\text{Arm 1 - Superior}), \qquad \theta_2^* = 0.4000 \quad (\text{Arm 2 - Suboptimal})$$
Both arms start with uninformative uniform priors:
$$\theta_1 \sim \operatorname{Beta}(1, 1), \qquad \theta_2 \sim \operatorname{Beta}(1, 1)$$
Trace the first 3 trials under Thompson Sampling given the following Monte Carlo draws from the current posteriors:
- **Round 1:** Sampled $\hat{\theta}_1^{(1)} = 0.6200$, $\hat{\theta}_2^{(1)} = 0.7500$.
- **Round 2:** Sampled $\hat{\theta}_1^{(2)} = 0.5500$, $\hat{\theta}_2^{(2)} = 0.2800$.
- **Round 3:** Sampled $\hat{\theta}_1^{(3)} = 0.7800$, $\hat{\theta}_2^{(3)} = 0.3500$.

For each round:
1. Identify the selected arm $A_t = \operatorname{argmax}_a \hat{\theta}_a^{(t)}$.
2. Receive reward $R_t$ ($R_1 = 0$ for Arm 2, $R_2 = 1$ for Arm 1, $R_3 = 1$ for Arm 1).
3. Update the Beta posterior parameters $(\alpha_a, \beta_a)$, posterior mean $\mathbb{E}[\theta_a]$, and posterior variance $\operatorname{Var}(\theta_a)$.
4. Compute the exact analytical probability that Arm 1 is superior to Arm 2: $\mathbb{P}(\theta_1 > \theta_2)$.

**Step-by-Step Solution:**

**1. Initial State ($t = 1$):**
$$\alpha_1 = 1, \beta_1 = 1 \implies \mathbb{E}[\theta_1] = 0.5000, \, \operatorname{Var}(\theta_1) = \frac{1}{12} \approx 0.0833$$
$$\alpha_2 = 1, \beta_2 = 1 \implies \mathbb{E}[\theta_2] = 0.5000, \, \operatorname{Var}(\theta_2) = \frac{1}{12} \approx 0.0833$$
By symmetry: $\mathbb{P}(\theta_1 > \theta_2) = \mathbf{0.5000}$.

**2. Round 1 ($t = 1$):**
- Candidate draws: $\hat{\theta}_1 = 0.6200, \hat{\theta}_2 = 0.7500$.
- Action chosen: $A_1 = \operatorname{argmax}(0.6200, 0.7500) = \mathbf{\text{Arm 2}}$.
- Environment observation: $R_1 = 0$ (failure).
- Posterior update for Arm 2:
  $$\alpha_2 \leftarrow 1 + 0 = 1, \qquad \beta_2 \leftarrow 1 + 1 = 2 \implies \theta_2 \sim \operatorname{Beta}(1, 2)$$
  $$\mathbb{E}[\theta_2] = \frac{1}{1 + 2} = \mathbf{0.3333}$$
  $$\operatorname{Var}(\theta_2) = \frac{1 \times 2}{(3)^2 (4)} = \frac{2}{36} = \mathbf{0.0556}$$
  Arm 1 remains unchanged: $\theta_1 \sim \operatorname{Beta}(1, 1)$.
- Exact win probability $\mathbb{P}(\theta_1 > \theta_2)$:
  Density of $\theta_1$: $p_1(u) = 1$ on $[0, 1]$.
  CDF of $\theta_2 \sim \operatorname{Beta}(1, 2)$: $F_2(u) = 1 - (1 - u)^2 = 2u - u^2$.
  $$\mathbb{P}(\theta_1 > \theta_2) = \int_0^1 p_1(u) F_2(u) du = \int_0^1 (2u - u^2) du = \left[ u^2 - \frac{u^3}{3} \right]_0^1 = 1 - \frac{1}{3} = \mathbf{\frac{2}{3} \approx 0.6667}$$

**3. Round 2 ($t = 2$):**
- Candidate draws: $\hat{\theta}_1 = 0.5500, \hat{\theta}_2 = 0.2800$.
- Action chosen: $A_2 = \operatorname{argmax}(0.5500, 0.2800) = \mathbf{\text{Arm 1}}$.
- Environment observation: $R_2 = 1$ (success).
- Posterior update for Arm 1:
  $$\alpha_1 \leftarrow 1 + 1 = 2, \qquad \beta_1 \leftarrow 1 + 0 = 1 \implies \theta_1 \sim \operatorname{Beta}(2, 1)$$
  $$\mathbb{E}[\theta_1] = \frac{2}{2 + 1} = \mathbf{0.6667}$$
  $$\operatorname{Var}(\theta_1) = \frac{2 \times 1}{(3)^2 (4)} = \frac{2}{36} = \mathbf{0.0556}$$
  Arm 2 remains: $\theta_2 \sim \operatorname{Beta}(1, 2)$.
- Exact win probability $\mathbb{P}(\theta_1 > \theta_2)$:
  Density of $\theta_1 \sim \operatorname{Beta}(2, 1)$: $p_1(u) = \frac{\Gamma(3)}{\Gamma(2)\Gamma(1)} u^{2-1} = 2u$.
  $$\mathbb{P}(\theta_1 > \theta_2) = \int_0^1 2u (2u - u^2) du = \int_0^1 (4u^2 - 2u^3) du = \left[ \frac{4}{3} u^3 - \frac{2}{4} u^4 \right]_0^1 = \frac{4}{3} - \frac{1}{2} = \mathbf{\frac{5}{6} \approx 0.8333}$$

**4. Round 3 ($t = 3$):**
- Candidate draws: $\hat{\theta}_1 = 0.7800, \hat{\theta}_2 = 0.3500$.
- Action chosen: $A_3 = \operatorname{argmax}(0.7800, 0.3500) = \mathbf{\text{Arm 1}}$.
- Environment observation: $R_3 = 1$ (success).
- Posterior update for Arm 1:
  $$\alpha_1 \leftarrow 2 + 1 = 3, \qquad \beta_1 \leftarrow 1 + 0 = 1 \implies \theta_1 \sim \operatorname{Beta}(3, 1)$$
  $$\mathbb{E}[\theta_1] = \frac{3}{3 + 1} = \mathbf{0.7500}$$
  $$\operatorname{Var}(\theta_1) = \frac{3 \times 1}{(4)^2 (5)} = \frac{3}{80} = \mathbf{0.0375}$$
- Exact win probability $\mathbb{P}(\theta_1 > \theta_2)$:
  Density of $\theta_1 \sim \operatorname{Beta}(3, 1)$: $p_1(u) = 3u^2$.
  $$\mathbb{P}(\theta_1 > \theta_2) = \int_0^1 3u^2 (2u - u^2) du = \int_0^1 (6u^3 - 3u^4) du = \left[ \frac{6}{4} u^4 - \frac{3}{5} u^5 \right]_0^1 = \frac{3}{2} - \frac{3}{5} = \frac{9}{10} = \mathbf{0.9000}$$

**Summary Evolution Table:**

| Trial $t$ | Arm 1 Posterior | $\mathbb{E}[\theta_1]$ | Arm 2 Posterior | $\mathbb{E}[\theta_2]$ | Arm Chosen | Reward $R_t$ | $\mathbb{P}(\theta_1 > \theta_2)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Start ($t=0$)** | $\operatorname{Beta}(1, 1)$ | $0.5000$ | $\operatorname{Beta}(1, 1)$ | $0.5000$ | — | — | $50.00\%$ |
| **After Round 1** | $\operatorname{Beta}(1, 1)$ | $0.5000$ | $\operatorname{Beta}(1, 2)$ | $0.3333$ | Arm 2 | $0$ | $66.67\%$ |
| **After Round 2** | $\operatorname{Beta}(2, 1)$ | $0.6667$ | $\operatorname{Beta}(1, 2)$ | $0.3333$ | Arm 1 | $1$ | $83.33\%$ |
| **After Round 3** | $\operatorname{Beta}(3, 1)$ | $0.7500$ | $\operatorname{Beta}(1, 2)$ | $0.3333$ | Arm 1 | $1$ | **$90.00\%$** |

Thompson Sampling rapidly allocates $90\%$ of future sampling budget to the truly superior Arm 1, providing smooth convergence to optimal exploitation.

---

### Illustration 4: Gradient Bandit Algorithm Hand Trace with Softmax Policy and Running Baseline

**Problem:**
Consider a 3-armed bandit with action preferences initialized to zero: $\mathbf{H}_1 = [0.0, 0.0, 0.0]^\top$.
Learning rate: $\alpha = 0.20$.
Running reward baseline initialized to $\bar{R}_1 = 0.00$.
Trace 2 consecutive training steps:
- **Step 1:** Action $A_1 = 2$ is chosen, emitting reward $R_1 = 2.5000$.
- **Step 2:** Action $A_2 = 1$ is chosen, emitting reward $R_2 = 0.5000$.

Compute for each step:
1. The Softmax action probability distribution $\boldsymbol{\pi}_t$.
2. The running baseline update $\bar{R}_{t+1} = \bar{R}_t + \frac{1}{t}(R_t - \bar{R}_t)$.
3. The preference vector update $\mathbf{H}_{t+1}$.

**Step-by-Step Solution:**

**1. Step 1 ($t = 1$):**
- Preferences: $\mathbf{H}_1 = [0.0, 0.0, 0.0]^\top$.
  $$\sum_{b=1}^3 e^{H_1(b)} = e^0 + e^0 + e^0 = 1 + 1 + 1 = 3.0$$
  $$\pi_1(a) = \frac{1.0}{3.0} \approx \mathbf{0.333333} \quad \forall a \in \{1, 2, 3\}$$
- Action chosen: $A_1 = 2$, with reward $R_1 = 2.5000$.
- Baseline update:
  $$\bar{R}_2 = \bar{R}_1 + \frac{1}{1}(R_1 - \bar{R}_1) = 0.0 + (2.5000 - 0.0) = \mathbf{2.5000}$$
- Advantage term: $R_1 - \bar{R}_1 = 2.5000 - 0.0000 = +2.5000 > 0$ (above average!).
- Preference updates with $\alpha = 0.20$:
  For chosen arm $a = 2$:
  $$H_2(2) = H_1(2) + \alpha(R_1 - \bar{R}_1)(1 - \pi_1(2)) = 0.0 + 0.20(2.5000)(1 - 0.333333) = 0.5000(0.666667) = \mathbf{+0.333333}$$
  For unchosen arms $a \in \{1, 3\}$:
  $$H_2(1) = H_1(1) - \alpha(R_1 - \bar{R}_1)\pi_1(1) = 0.0 - 0.20(2.5000)(0.333333) = -0.5000(0.333333) = \mathbf{-0.166667}$$
  $$H_2(3) = \mathbf{-0.166667}$$
  Updated preference vector: $\mathbf{H}_2 = [-0.166667, +0.333333, -0.166667]^\top$.

**2. Step 2 ($t = 2$):**
- Softmax probabilities at $t = 2$:
  $$e^{H_2(1)} = e^{-0.166667} \approx 0.846482$$
  $$e^{H_2(2)} = e^{+0.333333} \approx 1.395612$$
  $$e^{H_2(3)} = e^{-0.166667} \approx 0.846482$$
  $$\sum_{b=1}^3 e^{H_2(b)} = 0.846482 + 1.395612 + 0.846482 = 3.088576$$
  $$\pi_2(1) = \frac{0.846482}{3.088576} = \mathbf{0.274069}$$
  $$\pi_2(2) = \frac{1.395612}{3.088576} = \mathbf{0.451863}$$
  $$\pi_2(3) = \frac{0.846482}{3.088576} = \mathbf{0.274069}$$
  Notice that Arm 2's selection probability jumped from $33.33\% \to 45.19\%$!
- Action chosen: $A_2 = 1$, with reward $R_2 = 0.5000$.
- Baseline update:
  $$\bar{R}_3 = \bar{R}_2 + \frac{1}{2}(R_2 - \bar{R}_2) = 2.5000 + 0.50(0.5000 - 2.5000) = 2.5000 - 1.0000 = \mathbf{1.5000}$$
- Advantage term: $R_2 - \bar{R}_2 = 0.5000 - 2.5000 = \mathbf{-2.0000} < 0$ (below average!).
- Preference updates with $\alpha = 0.20$ and advantage $\Delta = -2.0000$:
  For chosen arm $a = 1$:
  $$H_3(1) = H_2(1) + \alpha(R_2 - \bar{R}_2)(1 - \pi_2(1)) = -0.166667 + 0.20(-2.0000)(1 - 0.274069)$$
  $$= -0.166667 - 0.4000(0.725931) = -0.166667 - 0.290372 = \mathbf{-0.457039}$$
  For unchosen arm $a = 2$:
  $$H_3(2) = H_2(2) - \alpha(R_2 - \bar{R}_2)\pi_2(2) = +0.333333 - 0.20(-2.0000)(0.451863)$$
  $$= +0.333333 + 0.4000(0.451863) = +0.333333 + 0.180745 = \mathbf{+0.514078}$$
  For unchosen arm $a = 3$:
  $$H_3(3) = H_2(3) - \alpha(R_2 - \bar{R}_2)\pi_2(3) = -0.166667 + 0.4000(0.274069) = -0.166667 + 0.109628 = \mathbf{-0.057039}$$

**Resulting Distribution at $t = 3$:**
$$e^{-0.457039} \approx 0.633152, \quad e^{0.514078} \approx 1.672097, \quad e^{-0.057039} \approx 0.944555 \implies \sum = 3.249804$$
$$\pi_3(1) = 19.48\%, \qquad \pi_3(2) = \mathbf{51.45\%}, \qquad \pi_3(3) = 29.07\%$$
Because Arm 1 performed below the baseline ($0.50 < 2.50$), its preference was penalized and its probability dropped from $27.4\% \to 19.5\%$, while Arm 2's probability expanded to over $51\%$.

---

### Illustration 5: LinUCB Contextual Bandit Step-by-Step Matrix Inversion and Ridge Confidence Ellipsoid Calculation

**Problem:**
A personalized news recommendation engine selects between $K = 2$ articles ($a \in \{1, 2\}$) based on a $d = 2$ dimensional user context vector $\mathbf{x} = [x_1, x_2]^\top$.
For each arm $a$, the expected reward is modeled as $q(\mathbf{x}, a) = \mathbf{x}^\top \boldsymbol{\theta}_a^*$.
Ridge regularization parameter: $\lambda = 1.0 \implies \mathbf{A}_1 = \mathbf{A}_2 = \mathbf{I}_2$.
Response vectors initialized to zero: $\mathbf{b}_1 = \mathbf{b}_2 = [0.0, 0.0]^\top$.
Exploration parameter: $\alpha = 1.0$.

1. **User 1 arrives** with context vector:
   $$\mathbf{x}^{(1)} = \begin{bmatrix} 0.80 \\ 0.60 \end{bmatrix}$$
   Compute the parameter estimates $\hat{\boldsymbol{\theta}}_a = \mathbf{A}_a^{-1} \mathbf{b}_a$ and the LinUCB decision scores:
   $$\text{UCB}_a = \hat{\boldsymbol{\theta}}_a^\top \mathbf{x} + \alpha \sqrt{\mathbf{x}^\top \mathbf{A}_a^{-1} \mathbf{x}}$$
   Break ties in favor of Arm 1.
2. The user clicks on Article 1 ($R_1 = 1.0$).
   Update the covariance matrix $\mathbf{A}_1 \leftarrow \mathbf{A}_1 + \mathbf{x}^{(1)} (\mathbf{x}^{(1)})^\top$ and response vector $\mathbf{b}_1 \leftarrow \mathbf{b}_1 + R_1 \mathbf{x}^{(1)}$.
3. Compute the analytical inverse $\mathbf{A}_1^{-1}$ using direct $2 \times 2$ matrix inversion.
4. **User 2 arrives** with identical context vector $\mathbf{x}^{(2)} = [0.80, 0.60]^\top$.
   Compute the updated parameter vector $\hat{\boldsymbol{\theta}}_1$, the new uncertainty bonus, and the resulting LinUCB scores for both arms.

**Step-by-Step Solution:**

**1. Decision for User 1 ($t = 1$):**
- Parameter estimates:
  $$\hat{\boldsymbol{\theta}}_1 = \mathbf{A}_1^{-1} \mathbf{b}_1 = \mathbf{I}_2^{-1} \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}, \qquad \hat{\boldsymbol{\theta}}_2 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$
- Uncertainty bonus for both arms:
  $$\mathbf{x}^\top \mathbf{A}_a^{-1} \mathbf{x} = \mathbf{x}^\top \mathbf{I}_2 \mathbf{x} = \|\mathbf{x}\|_2^2 = (0.80)^2 + (0.60)^2 = 0.64 + 0.36 = 1.0000$$
  $$\text{Bonus: } \alpha \sqrt{1.0000} = 1.0(1.0) = \mathbf{1.0000}$$
- Total UCB scores:
  $$\text{UCB}_1 = 0.0 + 1.0000 = \mathbf{1.0000}, \qquad \text{UCB}_2 = 0.0 + 1.0000 = \mathbf{1.0000}$$
  Tie broken in favor of **Arm 1**.

**2. Covariance and Response Vector Update for Arm 1:**
- Outer product $\mathbf{x} \mathbf{x}^\top$:
  $$\mathbf{x} \mathbf{x}^\top = \begin{bmatrix} 0.80 \\ 0.60 \end{bmatrix} \begin{bmatrix} 0.80 & 0.60 \end{bmatrix} = \begin{bmatrix} 0.64 & 0.48 \\ 0.48 & 0.36 \end{bmatrix}$$
- Updated covariance $\mathbf{A}_1$:
  $$\mathbf{A}_1 = \begin{bmatrix} 1.00 & 0.00 \\ 0.00 & 1.00 \end{bmatrix} + \begin{bmatrix} 0.64 & 0.48 \\ 0.48 & 0.36 \end{bmatrix} = \begin{bmatrix} \mathbf{1.64} & \mathbf{0.48} \\ \mathbf{0.48} & \mathbf{1.36} \end{bmatrix}$$
- Updated response vector $\mathbf{b}_1$:
  $$\mathbf{b}_1 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} + 1.0 \begin{bmatrix} 0.80 \\ 0.60 \end{bmatrix} = \begin{bmatrix} \mathbf{0.80} \\ \mathbf{0.60} \end{bmatrix}$$

**3. Direct Analytical Matrix Inversion of $\mathbf{A}_1$:**
$$\det(\mathbf{A}_1) = (1.64)(1.36) - (0.48)^2 = 2.2304 - 0.2304 = \mathbf{2.0000}$$
Using the $2 \times 2$ inverse formula $\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$:
$$\mathbf{A}_1^{-1} = \frac{1}{2.0000} \begin{bmatrix} 1.36 & -0.48 \\ -0.48 & 1.64 \end{bmatrix} = \begin{bmatrix} \mathbf{0.6800} & \mathbf{-0.2400} \\ \mathbf{-0.2400} & \mathbf{0.8200} \end{bmatrix}$$

**4. Decision for User 2 ($t = 2$):**
- Updated Ridge regression weights $\hat{\boldsymbol{\theta}}_1$:
  $$\hat{\boldsymbol{\theta}}_1 = \mathbf{A}_1^{-1} \mathbf{b}_1 = \begin{bmatrix} 0.6800 & -0.2400 \\ -0.2400 & 0.8200 \end{bmatrix} \begin{bmatrix} 0.80 \\ 0.60 \end{bmatrix} = \begin{bmatrix} 0.6800(0.80) - 0.2400(0.60) \\ -0.2400(0.80) + 0.8200(0.60) \end{bmatrix} = \begin{bmatrix} 0.5440 - 0.1440 \\ -0.1920 + 0.4920 \end{bmatrix} = \begin{bmatrix} \mathbf{0.4000} \\ \mathbf{0.3000} \end{bmatrix}$$
- Expected reward estimate for Arm 1:
  $$\hat{\boldsymbol{\theta}}_1^\top \mathbf{x} = 0.4000(0.80) + 0.3000(0.60) = 0.3200 + 0.1800 = \mathbf{0.5000}$$
- Uncertainty quadric for Arm 1:
  $$\mathbf{x}^\top \mathbf{A}_1^{-1} \mathbf{x} = \begin{bmatrix} 0.80 & 0.60 \end{bmatrix} \begin{bmatrix} 0.4000 \\ 0.3000 \end{bmatrix} = 0.80(0.4000) + 0.60(0.3000) = 0.3200 + 0.1800 = \mathbf{0.5000}$$
  $$\text{Uncertainty Bonus: } 1.0 \sqrt{0.5000} \approx \mathbf{0.707107}$$
  Notice that Arm 1's uncertainty shrunk from $1.0000 \to 0.7071$!
- Arm 1 UCB Score:
  $$\text{UCB}_1 = 0.5000 + 0.707107 = \mathbf{1.207107}$$
- Arm 2 UCB Score:
  $$\text{UCB}_2 = 0.0000 + 1.0000 = \mathbf{1.000000}$$

**Conclusion:**
$\text{UCB}_1 = 1.2071 > \text{UCB}_2 = 1.0000$.
Arm 1 is selected again with high confidence: although its uncertainty shrunk from $1.0 \to 0.707$, its high empirical reward ($0.50$) more than compensated for the uncertainty reduction, naturally balancing contextual exploitation and exploration!

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
