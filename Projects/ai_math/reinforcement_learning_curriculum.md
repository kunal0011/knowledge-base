# Reinforcement Learning Master Curriculum: Foundations to Frontier Reasoning (DeepSeek-R1)

---

## 1. Pedagogical Architecture (The 8-Part Standard)

Every single chapter in this curriculum adheres strictly to the uncompromised university-grade standard established across Modules 1–10:
1. **Part 1: Intuition & 101 Motivation** — Why does this algorithm or theorem exist? What failure mode of previous methods does it solve?
2. **Part 2: Rigorous Mathematical Formulation** — Formal definitions, Bellman operators, Markov Decision Processes, theorems, and mathematical proofs (Gilbert Strang, Bishop & Murphy rigor).
3. **Part 3: Geometric & Physical Interpretation** — High-dimensional state-action manifolds, simplex probability spaces, value landscapes, and dynamical systems.
4. **Part 4: Real-World Analogy** — An intuitive, memorable real-world mental model.
5. **Part 5: Prof. Tom Yeh "AI by Hand" Visual Grids** — Cell-by-cell matrix/tensor arithmetic walkthroughs with concrete numbers, visual grids, zero black boxes, and an explicit **"What Refers to What" Legend Protocol** table (symbol, mathematical representation, shape, and deep RL role).
6. **Part 6: Solved Step-by-Step Illustrations** — Manual arithmetic covering standard cases, boundary conditions, and pathological failure modes (e.g., Baird's counterexample, cliff walking, policy collapse).
7. **Part 7: Deep RL Connection & Modern Application** — Modern production applications in Robotics, Autonomous Agents, Game AI (AlphaZero), and Large Language Model Reasoning (DeepSeek-R1, OpenAI o1/o3).
8. **Part 8: Code Implementation & Verification** — Complete, standalone, vectorized NumPy / PyTorch implementations with rigorous automated unit tests.

---

## 2. Complete Syllabus Breakdown (7 Sub-Modules, 28 Chapters)

```
                       REINFORCEMENT LEARNING ARCHITECTURE MAP
                       
   [ Tabular Foundations ] ──► [ Deep Q-Networks ] ──► [ Policy Optimization ]
      • MDPs & Bellman            • DQN & Rainbow         • Policy Gradient & GAE
      • Contraction Mappings      • Distributional RL     • TRPO, PPO & SAC
               │                                                │
               ▼                                                ▼
   [ Model-Based & Planning ] ──────────────────────► [ Modern Frontier & LLMs ]
      • Dyna-Q & MPC                                      • Offline RL (CQL)
      • MCTS & AlphaZero                                  • Decision Transformers
      • World Models & Dreamer                            • RLHF & GRPO (DeepSeek-R1)
```

---

### Module RL-1: Foundations of Reinforcement Learning & Markov Decision Processes (MDPs)

#### RL 1.1 Multi-Armed Bandits & Exploration-Exploitation Dilemma
- **Core Theory:** The $K$-armed bandit problem; action-value estimates $Q_t(a)$; sample-average updates; theoretical lower bound on regret (Lai & Robbins $\mathcal{O}(\log T)$ regret bound).
- **Exploration Strategies:** $\epsilon$-greedy (constant vs. decaying); Upper Confidence Bound (UCB1 derived via Hoeffding's Inequality); Thompson Sampling (Bayesian conjugate priors with Beta-Bernoulli distributions).
- **Tom Yeh Visual Grid:** 3-armed bandit payout tracking, empirical reward updates, UCB exploration bonus calculation, and Beta posterior parameter updates by hand.
- **Code & Test Suite:** Vectorized simulation of 10-armed bandit comparing $\epsilon$-greedy, UCB1, and Thompson Sampling on cumulative regret.

#### RL 1.2 Markov Decision Processes (MDPs) & Formal Definitions
- **Core Theory:** The formal 5-tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$; state transition probability kernel $\mathcal{P}(s' \mid s, a)$; reward function $\mathcal{R}(s, a, s')$; discount factor $\gamma \in [0, 1)$; episodic vs. continuing tasks; the Markov Property $\mathbb{P}(S_{t+1} \mid S_t, A_t, \dots, S_0, A_0) = \mathbb{P}(S_{t+1} \mid S_t, A_t)$.
- **Policies & Trajectories:** Deterministic $\pi(s)$ vs. stochastic policies $\pi(a \mid s)$; trajectory distribution $P(\tau; \pi)$; discounted return $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$; stationary state visitation distribution $d^\pi(s)$.
- **Extensions:** Partially Observable Markov Decision Processes (POMDPs): Observation function $\mathcal{O}(o \mid s)$, belief states $b(s) = \mathbb{P}(S_t = s \mid o_{1:t}, a_{1:t-1})$.
- **Tom Yeh Visual Grid:** 3-state gridworld transition matrix $\mathcal{P}$, trajectory discount calculation, and belief state update by hand.
- **Code & Test Suite:** Exact matrix representation of finite MDP and trajectory generator verifying Markovian memorylessness.

#### RL 1.3 The Bellman Equations: State-Value and Action-Value Functions
- **Core Theory:** Definition of State-Value $V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ and Action-Value $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$; relationship $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$.
- **First-Principles Derivation:** Bellman Expectation Equation for $V^\pi(s)$ and $Q^\pi(s, a)$ via law of total expectation; backup diagrams.
- **Bellman Optimality Equations:** Non-linear optimality equations for $V^*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)[r + \gamma V^*(s')]$ and $Q^*(s, a)$.
- **Matrix Formulation:** Linear system $\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}^\pi \implies \mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$; condition number and computational complexity $\mathcal{O}(|\mathcal{S}|^3)$.
- **Tom Yeh Visual Grid:** 2-state MDP matrix inversion $(\mathbf{I} - \gamma \mathbf{P})^{-1}$ and cell-by-cell Bellman expectation verification.
- **Code & Test Suite:** Analytical matrix inversion vs. iterative Bellman expectation evaluation matching to $< 10^{-12}$.

#### RL 1.4 Dynamic Programming & The Banach Contraction Mapping Theorem
- **Core Theory:** Bellman backup operator $\mathcal{T}^\pi: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ defined by $(\mathcal{T}^\pi v)(s) = r^\pi(s) + \gamma \sum_{s'} P^\pi(s, s') v(s')$; non-linear optimality operator $(\mathcal{T}^* v)(s) = \max_a [r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) v(s')]$.
- **Rigorous Mathematical Proof:** Proof that $\mathcal{T}^*$ is a strict $\gamma$-contraction in supremum norm: $\|\mathcal{T}^* u - \mathcal{T}^* v\|_\infty \le \gamma \|u - v\|_\infty$.
- **Banach Fixed-Point Theorem:** Existence, uniqueness of $V^*$, and geometric convergence rate $\|V_k - V^*\|_\infty \le \frac{\gamma^k}{1 - \gamma} \|V_1 - V_0\|_\infty$.
- **Algorithms:** Policy Evaluation (iterative contraction); Policy Iteration (Policy Improvement Theorem proof via induction: $\pi_{k+1} \ge \pi_k$); Value Iteration.
- **Tom Yeh Visual Grid:** Step-by-step Value Iteration on a 3-state chain showing monotonic contraction $\|V_{k+1} - V_k\|_\infty \le \gamma \|V_k - V_{k-1}\|_\infty$.
- **Code & Test Suite:** Policy Iteration and Value Iteration implementations on FrozenLake, verifying exact convergence to analytical optimal policy.

---

### Module RL-2: Tabular Model-Free Reinforcement Learning

#### RL 2.1 Monte Carlo Methods for Prediction and Control
- **Core Theory:** Learning directly from episodic experience without transition model $\mathcal{P}$; First-Visit vs. Every-Visit Monte Carlo; law of large numbers asymptotic convergence.
- **Monte Carlo Control:** Policy evaluation + $\epsilon$-greedy policy improvement; Monte Carlo with Exploring Starts (MCES); overcoming the exploring starts assumption via $\epsilon$-soft on-policy control.
- **Off-Policy Monte Carlo & Importance Sampling:** Behavior policy $b(a \mid s)$ vs. target policy $\pi(a \mid s)$; importance sampling ratio $\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$; Ordinary Importance Sampling (unbiased, infinite variance) vs. Weighted Importance Sampling (consistent, biased, bounded variance).
- **Tom Yeh Visual Grid:** Trajectory return calculation, importance sampling ratio derivation, and weighted value update by hand.
- **Code & Test Suite:** Blackjack environment simulation comparing Ordinary vs. Weighted Importance Sampling variance.

#### RL 2.2 Temporal-Difference Learning (TD(0))
- **Core Theory:** Combining Monte Carlo sampling with Dynamic Programming bootstrapping; TD(0) update rule $V(S_t) \leftarrow V(S_t) + \alpha \delta_t$; TD Error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$.
- **Theoretical Comparison:** Bias-variance trade-off across MC (zero bias, high variance) and TD(0) (low variance, non-zero bias); online updates without waiting for episode termination; convergence proof under Robbins-Monro conditions ($\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$).
- **Batch TD vs. Batch MC:** Why Batch TD converges to the certainty-equivalence estimate (maximum likelihood MDP model) while Batch MC minimizes mean squared error on training trajectories.
- **Tom Yeh Visual Grid:** Cell-by-cell numerical update of TD(0) along a 4-step sequence with explicit TD error calculation.
- **Code & Test Suite:** Empirical comparison of TD(0) vs. MC on a Random Walk Markov Reward Process, measuring RMSE to analytical ground truth.

#### RL 2.3 On-Policy vs. Off-Policy Control: SARSA, Q-Learning & Expected SARSA
- **Core Theory:**
  - **SARSA (On-Policy TD Control):** Updates based on actual action taken: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$.
  - **Q-Learning (Watkins, 1989 - Off-Policy TD Control):** Updates directly toward optimal greedy action regardless of behavior policy: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a'} Q(S', a') - Q(S, A)]$.
  - **Expected SARSA:** Variance reduction by taking expectation over policy: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \sum_{a'} \pi(a' \mid S') Q(S', a') - Q(S, A)]$.
- **The Cliff Walking Dilemma:** Mathematical explanation of why Q-learning learns the optimal dangerous path (high risk under $\epsilon$-exploration) while SARSA learns the safe detour.
- **Tom Yeh Visual Grid:** Parallel execution of Q-Learning vs. SARSA on a 2x2 grid showing Q-value divergence under $\epsilon$-greedy exploration.
- **Code & Test Suite:** Cliff Walking simulation comparing online episode rewards and optimal path recovery for SARSA, Q-Learning, and Expected SARSA.

#### RL 2.4 Multi-Step Bootstrapping & Eligibility Traces: TD($\lambda$)
- **Core Theory:** Bridging the spectrum from TD(0) to Monte Carlo; $n$-step returns $G_{t:t+n} = \sum_{k=1}^n \gamma^{k-1} R_{t+k} + \gamma^n V_{t+n-1}(S_{t+n})$.
- **The Forward View:** The $\lambda$-return $G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n}$; compounding geometric weighting.
- **The Backward View (Eligibility Traces):** Mechanistic online implementation via short-term memory vector $e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbb{I}(S_t = s)$; accumulating vs. replacing vs. Dutch traces.
- **Equivalence Theorem:** Proof that offline backward-view TD($\lambda$) is mathematically identical to forward-view TD($\lambda$).
- **Tom Yeh Visual Grid:** 3-step episode trace accumulation, exponential decay, and credit assignment calculation by hand.
- **Code & Test Suite:** Implementation of True Online TD($\lambda$) on a random walk verifying equivalence with the forward $\lambda$-return.

---

### Module RL-3: Value-Based Deep Reinforcement Learning

#### RL 3.1 Function Approximation & The Deadly Triad
- **Core Theory:** Scaling beyond tabular matrices to continuous states $\mathcal{S} = \mathbb{R}^D$; linear value function approximation $V_{\mathbf{w}}(s) = \mathbf{w}^\top \mathbf{x}(s)$; semi-gradient TD updates (why the target gradient is omitted).
- **The Deadly Triad:** The catastrophic interaction of:
  1. Function Approximation (e.g., neural networks).
  2. Bootstrapping (e.g., TD and Bellman targets).
  3. Off-Policy Learning (e.g., Q-learning or training on replay data).
- **Baird's Counterexample:** Mathematical proof of divergence where value function weights explode to infinity under linear off-policy TD.
- **Tom Yeh Visual Grid:** Single-step iteration of Baird's 7-state star system demonstrating numeric weight divergence.
- **Code & Test Suite:** Numerical simulation of Baird's counterexample reproducing weight explosion vs. convergence under on-policy updates.

#### RL 3.2 Deep Q-Networks (DQN: Mnih et al., 2015)
- **Core Theory:** Deep neural network parameterized action-value function $Q(s, a; \theta)$; loss function:
  $$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
- **Algorithmic Innovations:**
  1. **Experience Replay Buffer $\mathcal{D}$:** Uniform random sampling breaking temporal correlation and stabilizing non-i.i.d. data streams.
  2. **Target Network $\theta^-$:** Freezing target parameters and updating periodically ($\theta^- \leftarrow \theta$ every $C$ steps) to break non-stationary target chasing.
  3. **Huber Loss (Smooth L1):** Preventing exploding gradients from large temporal-difference errors:
     $$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le 1 \\ |y - \hat{y}| - \frac{1}{2} & \text{otherwise} \end{cases}$$
- **Tom Yeh Visual Grid:** Replay buffer sample retrieval, target network evaluation, Huber loss computation, and backprop gradient calculation by hand.
- **Code & Test Suite:** Complete PyTorch DQN implementation solving CartPole-v1 with automated convergence unit tests.

#### RL 3.3 Advanced DQN Extensions: The Rainbow Suite
- **Core Theory:**
  - **Double DQN (Van Hasselt et al., 2015):** Eliminating maximization bias ($\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$) by decoupling action selection from action evaluation:
    $$Y^{\text{DoubleQ}} = R_{t+1} + \gamma Q\left( S_{t+1}, \operatorname{argmax}_a Q(S_{t+1}, a; \theta_t); \theta_t^- \right)$$
  - **Dueling DQN (Wang et al., 2016):** Decoupling representation into state value $V(s)$ and action advantage $A(s, a)$; identifiability constraint via mean subtraction:
    $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$
  - **Prioritized Experience Replay (PER, Schaul et al., 2016):** Sampling transitions proportional to TD error $p_i = |\delta_i| + \epsilon$; Importance Sampling weights $w_i = (N \cdot P(i))^{-\beta} / \max_j w_j$ correcting distribution bias.
  - **Noisy Networks (Fortunato et al., 2018):** Replacing $\epsilon$-greedy exploration with learned parametric noise on linear layer weights $y = (\mu^w + \sigma^w \odot \epsilon^w)x + (\mu^b + \sigma^b \odot \epsilon^b)$.
- **Tom Yeh Visual Grid:** Dueling stream summation, Double Q target calculation, and PER importance sampling weight adjustments by hand.
- **Code & Test Suite:** Modular PyTorch implementation comparing Vanilla DQN, Double DQN, and Dueling DQN on LunarLander-v2.

#### RL 3.4 Distributional Reinforcement Learning (C51 & QR-DQN)
- **Core Theory:** Shifting from scalar expectation $\mathbb{E}[G]$ to the full probability distribution of returns $Z(s, a)$; the Distributional Bellman Equation:
  $$Z(s, a) \stackrel{D}{=} R(s, a) + \gamma Z(S', A^*)$$
- **Categorical DQN (C51, Bellemare et al., 2017):** 51 discrete fixed support atoms $z_i \in [V_{\min}, V_{\max}]$; Bellman projection step onto support bins via linear interpolation; cross-entropy loss minimization.
- **Quantile Regression DQN (QR-DQN, Dabney et al., 2018):** Transposing the problem: fixed uniform probabilities with learned variable quantile locations $\theta_i$; minimizing Quantile Huber Loss under 1-Wasserstein metric.
- **Tom Yeh Visual Grid:** 5-atom discrete return distribution projection onto fixed target bins with exact probability redistribution arithmetic.
- **Code & Test Suite:** PyTorch implementation of categorical Bellman projection and C51 loss calculation verified against analytical distributions.

---

### Module RL-4: Policy Gradient Methods & Actor-Critic Architectures

#### RL 4.1 The Policy Gradient Theorem & REINFORCE
- **Core Theory:** Parameterized stochastic policy $\pi_\theta(a \mid s)$; objective function $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int P(\tau; \theta) R(\tau) d\tau$.
- **The Log-Derivative (Likelihood Ratio) Trick:** $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$.
- **First-Principles Proof of Policy Gradient Theorem:**
  $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right]$$
  Step-by-step cancellation showing independence from unknown environment transition dynamics $\nabla_\theta \mathcal{P}(s' \mid s, a) = 0$.
- **REINFORCE (Williams, 1992):** Monte Carlo rollout implementation $\nabla_\theta J(\theta) \approx \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)$.
- **Baseline Subtraction:** Proof that subtracting state-dependent baseline $b(s)$ introduces **zero bias**:
  $$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s) b(s)] = b(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) = b(s) \nabla_\theta (1) = 0$$
  Analytical derivation of the optimal variance-minimizing baseline $b^*(s) = \frac{\mathbb{E}[\|\nabla_\theta \log \pi\|^2 Q]}{\mathbb{E}[\|\nabla_\theta \log \pi\|^2]}$.
- **Tom Yeh Visual Grid:** Softmax / Gaussian policy log-derivative calculation, trajectory return multiplication, baseline subtraction, and parameter update by hand.
- **Code & Test Suite:** PyTorch REINFORCE with and without baseline on CartPole-v1 verifying variance reduction.

#### RL 4.2 Advantage Actor-Critic (A2C & A3C)
- **Core Theory:** Replacing high-variance Monte Carlo returns $G_t$ with a learned Critic; Actor parameterizes policy $\pi_\theta(a \mid s)$; Critic parameterizes state-value function $V_\phi(s)$.
- **The Advantage Function:** $A(s, a) = Q(s, a) - V(s)$; single-step TD advantage estimator:
  $$\hat{A}(S_t, A_t) = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$$
- **A2C vs. A3C:** Synchronous deterministic batched updates (A2C) vs. Asynchronous lock-free multi-threaded updates (A3C, Mnih et al., 2016).
- **Multi-Task Objective Function:**
  $$L(\theta, \phi) = \underbrace{-\log \pi_\theta(a_t \mid s_t) \hat{A}_t}_{\text{Policy Loss}} + c_1 \underbrace{\left( R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right)^2}_{\text{Critic Value Loss}} - c_2 \underbrace{\mathcal{H}(\pi_\theta(\cdot \mid s_t))}_{\text{Entropy Bonus}}$$
- **Tom Yeh Visual Grid:** Actor-Critic forward pass, TD advantage calculation, policy entropy evaluation, and simultaneous gradient backpropagation by hand.
- **Code & Test Suite:** Vectorized PyTorch A2C implementation running 8 parallel environments on CartPole-v1.

#### RL 4.3 Generalized Advantage Estimation (GAE)
- **Core Theory:** Unifying $k$-step advantage estimators into an exponentially weighted average; GAE($\gamma, \lambda$, Schulman et al., 2015):
  $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V, \quad \text{where } \delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
- **Recursive Formulation:** $\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda \hat{A}_{t+1}^{\text{GAE}}$ (enables backward $\mathcal{O}(T)$ dynamic programming calculation).
- **Bias-Variance Continuum:**
  - $\lambda = 0 \implies \hat{A}_t = \delta_t^V$ (lowest variance, highest bias from value function error).
  - $\lambda = 1 \implies \hat{A}_t = \sum_{l=0}^\infty \gamma^l R_{t+l+1} - V(S_t) = G_t - V(S_t)$ (unbiased, highest variance from Monte Carlo returns).
- **Tom Yeh Visual Grid:** Step-by-step backward recursive GAE calculation across a 4-step rollout with concrete values.
- **Code & Test Suite:** NumPy and PyTorch vectorized GAE implementation verified against explicit matrix formulations.

#### RL 4.4 Continuous Action Spaces: DDPG & TD3
- **Core Theory:** Deterministic Policy Gradient (DPG) Theorem (Silver et al., 2014):
  $$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu} \left[ \left. \nabla_a Q^\mu(s, a) \right|_{a = \mu_\theta(s)} \nabla_\theta \mu_\theta(s) \right]$$
  Avoids integrating over continuous action space $\mathcal{A} \subset \mathbb{R}^d$.
- **Deep Deterministic Policy Gradient (DDPG, Lillicrap et al., 2015):** Continuous Actor $\mu_\theta(s)$, Critic $Q_\phi(s, a)$; Polyak averaging for target networks ($\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ with $\tau \approx 0.005$).
- **Twin Delayed DDPG (TD3, Fujimoto et al., 2018):**
  1. **Clipped Double Q-Learning:** Twin critics $Q_{\phi_1}, Q_{\phi_2}$; target uses minimum $y = r + \gamma \min_{i=1, 2} Q_{\phi_i^-}(s', \tilde{a})$.
  2. **Delayed Policy Updates:** Update Actor and targets once every $d=2$ Critic updates.
  3. **Target Policy Smoothing:** Adding clipped noise $\tilde{a} = \mu_{\theta^-}(s') + \operatorname{clip}(\epsilon, -c, c)$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$ to prevent exploitation of narrow Q-peaks.
- **Tom Yeh Visual Grid:** Target noise addition, twin critic evaluation, minimum selection, and deterministic actor gradient chain rule by hand.
- **Code & Test Suite:** PyTorch TD3 implementation tested on Pendulum-v1 with reward convergence assertion.

---

### Module RL-5: Trust Region & Proximal Policy Optimization

#### RL 5.1 Natural Policy Gradients & Information Geometry
- **Core Theory:** Why standard Euclidean gradient ascent $\theta \leftarrow \theta + \alpha \nabla J$ fails in policy space (a small step in parameter space $\Delta \theta$ can cause a catastrophic jump in output probability distribution $\pi_\theta$).
- **Riemannian Manifold Optimization:** Measuring distance on probability simplex via Kullback-Leibler divergence $D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\theta + \Delta \theta})$.
- **Fisher Information Matrix (FIM):** Second-order Taylor expansion of KL divergence:
  $$D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\theta + \Delta \theta}) \approx \frac{1}{2} \Delta \theta^\top F(\theta) \Delta \theta, \quad F(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)^\top \right]$$
- **Natural Policy Gradient Update:** $\Delta \theta = F(\theta)^{-1} \nabla_\theta J(\theta)$; coordinate invariance under reparameterization.
- **Tom Yeh Visual Grid:** $2 \times 2$ Fisher Information Matrix calculation, matrix inversion, and natural gradient step for a Gaussian policy by hand.
- **Code & Test Suite:** Natural Policy Gradient vs. Vanilla Policy Gradient trajectory comparison on a toy quadratic MDP.

#### RL 5.2 Trust Region Policy Optimization (TRPO: Schulman et al., 2015)
- **Core Theory:** The Relative Performance Identity (Kakade & Langford):
  $$J(\tilde{\pi}) - J(\pi) = \mathbb{E}_{s \sim d^{\tilde{\pi}}, a \sim \tilde{\pi}} \left[ A^\pi(s, a) \right]$$
- **Monotonic Improvement Guarantee:** Minorize-Maximization (MM algorithm); bounding difference in state visitation frequencies via maximum total variation / KL divergence.
- **The Constrained Optimization Problem:**
  $$\max_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \quad \text{s.t.} \quad \hat{\mathbb{E}}_t \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t)) \right] \le \delta$$
- **Taylor Approximations & Analytical Solution:** Linear objective $g^\top \Delta \theta$ + Quadratic constraint $\frac{1}{2} \Delta \theta^\top H \Delta \theta \le \delta$; optimal search direction:
  $$x = H^{-1} g, \quad \Delta \theta = \sqrt{\frac{2\delta}{x^\top H x}} x$$
- **Conjugate Gradient & Line Search:** Hessian-Free optimization: computing Fisher-Vector products $Hv$ via autograd without storing $H$; backtracking line search verifying improvement and constraint satisfaction.
- **Tom Yeh Visual Grid:** Conjugate Gradient step evaluation, step length scaling, and backtracking line search acceptance check by hand.
- **Code & Test Suite:** PyTorch Fisher-Vector product implementation and Conjugate Gradient solver verified against exact Hessian matrix.

#### RL 5.3 Proximal Policy Optimization (PPO: Schulman et al., 2017)
- **Core Theory:** Replacing complex second-order constrained optimization (TRPO) with a first-order clipped surrogate objective.
- **The Probability Ratio:** $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$.
- **The Clipped Objective:**
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
- **Rigorous Case Analysis:**
  - Case 1: $\hat{A}_t > 0$ (Action was better than average). If $r_t > 1 + \epsilon$, policy is already much more likely; gradient is clipped to $0$ to prevent over-eager steps.
  - Case 2: $\hat{A}_t < 0$ (Action was worse than average). If $r_t < 1 - \epsilon$, policy has already suppressed the action; gradient is clipped to $0$.
- **Complete PPO-Clip Training Pipeline:** Generalized Advantage Estimation (GAE), mini-batch SGD over multiple epochs, value loss clipping $L^{\text{VF}}$, and entropy exploration bonus.
- **Tom Yeh Visual Grid:** Cell-by-cell clipping evaluation across four scenarios ($A>0$ with $r>1+\epsilon$, $A>0$ with $r<1+\epsilon$, $A<0$ with $r<1-\epsilon$, $A<0$ with $r>1-\epsilon$) and resulting parameter gradients.
- **Code & Test Suite:** Full PyTorch PPO implementation tested on CartPole-v1 and InvertedPendulum-v4 with complete unit test suite.

#### RL 5.4 Soft Actor-Critic (SAC: Haarnoja et al., 2018)
- **Core Theory:** Maximum Entropy Reinforcement Learning:
  $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right]$$
  where $\mathcal{H}(\pi(\cdot \mid s)) = \mathbb{E}_{a \sim \pi}[-\log \pi(a \mid s)]$ is Shannon entropy. Encourages exploration and captures multi-modal optimal behaviors.
- **Soft Bellman Equations:** Soft state-value $V(s) = \mathbb{E}_{a}[Q(s, a) - \alpha \log \pi(a \mid s)]$; Soft policy iteration convergence proof.
- **The Reparameterization Trick for Continuous Action Policies:** $a = f_\theta(\epsilon; s) = \tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \epsilon)$ with $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
- **Enforcing Action Bounds Correction:** Accounting for the change of variables under the $\tanh$ squashing bijection:
  $$\log \pi(a \mid s) = \log \mu(u \mid s) - \sum_{i=1}^d \log(1 - \tanh^2(u_i))$$
- **Automatic Temperature Tuning:** Dual convex optimization adaptively scaling entropy weight $\alpha$:
  $$J(\alpha) = \mathbb{E}_{a_t \sim \pi_t} \left[ -\alpha \log \pi_t(a_t \mid s_t) - \alpha \bar{\mathcal{H}} \right]$$
  where $\bar{\mathcal{H}} = -\operatorname{dim}(\mathcal{A})$ is the target entropy.
- **Tom Yeh Visual Grid:** Gaussian sample transformation through $\tanh$, log-determinant Jacobian penalty correction, and soft Q-target evaluation by hand.
- **Code & Test Suite:** Vectorized PyTorch SAC implementation verifying automatic entropy tuning and continuous control on Pendulum-v1.

---

### Module RL-6: Model-Based Reinforcement Learning & Planning

#### RL 6.1 Model-Based Foundations & Dyna-Q
- **Core Theory:** Model-Free (reactive, high sample complexity) vs. Model-Based (sample efficient, vulnerable to model bias).
- **Model Learning:** Transition model $\hat{p}_\phi(s' \mid s, a)$ and reward model $\hat{r}_\psi(s, a)$ trained via supervised maximum likelihood.
- **The Dyna Architecture (Sutton):** Interleaving real environment experience, model learning, and simulated background planning:
  1. Act in real world $\to (s, a, r, s')$.
  2. Model-free update: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max Q(s', \cdot) - Q(s, a)]$.
  3. Update world model with $(s, a, r, s')$.
  4. Repeat $N$ times: Sample random previously seen $(s, a)$, simulate $s', r$ from model, update $Q(s, a)$.
- **Tom Yeh Visual Grid:** 1 real environment transition followed by $N=3$ hallucinated planning updates on a 3-state gridworld by hand.
- **Code & Test Suite:** Dyna-Q vs. Standard Q-learning benchmark on a Maze environment showing $10\times$ faster convergence.

#### RL 6.2 Trajectory Optimization & Model Predictive Control (MPC)
- **Core Theory:** Planning in continuous spaces; Shooting methods vs. Collocation; Open-loop vs. Closed-loop control.
- **The Cross-Entropy Method (CEM):** Population-based derivative-free optimization:
  1. Sample $N$ action sequences from candidate Gaussian distribution $\mathcal{N}(\mu, \Sigma)$.
  2. Evaluate predicted trajectory returns using the learned world model.
  3. Select top $K$ elite sequences.
  4. Refit Gaussian parameters $(\mu, \Sigma)$ to the elite set; iterate.
- **Model Predictive Control (MPC):** Receding horizon planning: optimize over horizon $H$, execute *only the first action* $a_0$, observe actual state $s_1$, re-plan from $s_1$ (robustness to model errors).
- **Probabilistic Ensembles with Trajectory Sampling (PETS, Chua et al., 2018):** Ensembles of probabilistic neural networks capturing aleatoric (inherent stochasticity) and epistemic (lack of data) uncertainty.
- **Tom Yeh Visual Grid:** 2-iteration Cross-Entropy Method optimization with elite selection and mean/variance refitting by hand.
- **Code & Test Suite:** PyTorch CEM planner driving a simulated inverted pendulum via an ensemble dynamics model.

#### RL 6.3 Monte Carlo Tree Search (MCTS) & AlphaZero
- **Core Theory:** Discrete planning under perfect or learned dynamics; The 4-Phase MCTS Cycle:
  1. **Selection:** Traverse tree from root using Upper Confidence bounds for Trees (UCT / PUCT).
  2. **Expansion:** Add novel leaf nodes to the search tree.
  3. **Evaluation:** Estimate leaf value (via random rollouts or neural network evaluation).
  4. **Backup:** Propagate return back up the path, incrementing visit counts $N(s, a)$ and updating mean values $Q(s, a)$.
- **The PUCT Selection Formula (AlphaZero - Silver et al., 2017):**
  $$a^* = \operatorname{argmax}_a \left[ Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$
  where $P(s, a)$ is the prior probability from policy network $p_\theta(a \mid s)$, and $Q(s, a)$ is the mean value from value network $v_\theta(s)$.
- **Self-Play Training Loop:** MCTS acts as a policy improvement operator: the visit count distribution $\pi_{\text{MCTS}}(a \mid s) \propto N(s, a)^{1/\tau}$ provides a superior target for training policy network $p_\theta$.
- **Tom Yeh Visual Grid:** Complete 3-level MCTS search tree traversal: PUCT score evaluation, leaf expansion, value backup, and visit count update by hand.
- **Code & Test Suite:** Complete MCTS engine in Python playing Tic-Tac-Toe / Connect-4 with perfect minimax convergence.

#### RL 6.4 World Models & Dreamer (Hafner et al., 2020–2024)
- **Core Theory:** Learning entirely inside a compact latent imagination space without raw pixel rendering.
- **Recurrent State-Space Model (RSSM):** Decomposing world dynamics into:
  - Deterministic recurrent state: $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$.
  - Stochastic latent state: $z_t \sim q_\phi(z_t \mid h_t, x_t)$ (posterior) or $p_\phi(z_t \mid h_t)$ (prior).
- **DreamerV1 to DreamerV3:**
  - Training Actor-Critic entirely inside the "dream": hallucinating rollouts using RSSM latent transitions.
  - Propagating analytical gradients through time via the reparameterization trick on imagined latent trajectories:
    $$\max_\theta \mathbb{E}\left[ \sum_{\tau=t}^{t+H} \gamma^{\tau-t} V_\psi(z_\tau) \right]$$
- **Tom Yeh Visual Grid:** RSSM latent step calculation, KL divergence between posterior and prior latents, and imagined actor gradient backpropagation by hand.
- **Code & Test Suite:** Minimal PyTorch RSSM latent dynamics model with imagined rollout verification.

---

### Module RL-7: Offline RL, Sequence Modeling & Frontier Alignment (LLMs)

#### RL 7.1 Offline (Batch) Reinforcement Learning & Conservative Q-Learning (CQL)
- **Core Theory:** The Offline RL paradigm: learning strictly from a fixed, static dataset $\mathcal{D} = \{(s, a, r, s')\}$ without environment interaction.
- **The Out-of-Distribution (OOD) Action Catastrophe:** Standard Q-learning queries $\max_{a'} Q(s', a')$; function approximators overestimate unseen actions, creating positive feedback loops that destroy policy performance.
- **Conservative Q-Learning (CQL, Kumar et al., 2020):** Adding a regularizer that pushes down Q-values of policy actions while pulling up Q-values of dataset actions:
  $$\min_Q \alpha \left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta}[Q(s, a)] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)] \right) + \frac{1}{2} \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q(s, a) - \mathcal{B}^* Q(s, a) \right)^2 \right]$$
- **Theoretical Lower Bound Proof:** Proof that under CQL, the expected value function is a guaranteed lower bound on the true policy value: $\hat{V}^{\text{CQL}}(s) \le V^\pi(s)$ everywhere.
- **Tom Yeh Visual Grid:** CQL loss calculation on in-distribution vs. out-of-distribution actions and parameter update arithmetic by hand.
- **Code & Test Suite:** PyTorch CQL implementation on a 1D continuous environment demonstrating resistance to OOD value overestimation.

#### RL 7.2 Decision Transformers & Trajectory Modeling
- **Core Theory:** Re-framing Reinforcement Learning as conditional autoregressive sequence modeling (Chen et al., 2021).
- **Trajectory Tokenization:** Representing an episode as a sequence of three-token tuples:
  $$\tau = \left( \hat{R}_1, s_1, a_1, \, \hat{R}_2, s_2, a_2, \, \dots, \, \hat{R}_T, s_T, a_T \right)$$
  where $\hat{R}_t = \sum_{k=t}^T R_k$ is the **Return-To-Go (RTG)**.
- **Causal Transformer Architecture:** Linear projection of tokens into common embedding space $d_{\text{model}}$; episodic timestep embeddings; causal self-attention masking preventing information leakage from future tokens.
- **Inference Time Conditioning:** Prompt the model with a high desired return $\hat{R}_1 = R_{\text{target}}$ and initial state $s_1$; the model autoregressively generates expert actions to fulfill the specified return!
- **Tom Yeh Visual Grid:** Sequence tokenization, linear embedding, causal self-attention matrix calculation, and RTG-conditioned action prediction by hand.
- **Code & Test Suite:** PyTorch Decision Transformer architecture trained on offline trajectories with conditioning test.

#### RL 7.3 Reinforcement Learning from Human Feedback (RLHF) with PPO
- **Core Theory:** The standard post-training pipeline for Large Language Models (LLMs):
  1. Supervised Fine-Tuning (SFT).
  2. Reward Model (RM) training via Bradley-Terry preference pairs:
     $$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$
  3. PPO Policy Optimization with per-token KL divergence penalty.
- **Per-Token KL Penalty Formulation:** Preventing the policy $\pi_\theta$ from diverging too far from the base model $\pi_{\text{ref}}$ (reward hacking / mode collapse):
  $$R_{\text{total}}(x, y) = r_\psi(x, y) - \beta D_{\text{KL}}(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x))$$
  $$R_{\text{token}}(x, y_t) = \begin{cases} -\beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} & \text{for } t < |y| \\ r_\psi(x, y) - \beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} & \text{for } t = |y| \end{cases}$$
- **Tom Yeh Visual Grid:** Prompt-response token-level log-prob calculation, per-token KL divergence penalty, advantage calculation, and PPO ratio update by hand.
- **Code & Test Suite:** PyTorch mini-RLHF pipeline computing token-level KL penalties, GAE advantages, and PPO actor updates.

#### RL 7.4 Group Relative Policy Optimization (GRPO) & Frontier Reasoning (DeepSeek-R1)
- **Core Theory:** The architectural breakthrough of DeepSeek-Math and DeepSeek-R1 (Shao et al., 2024).
- **The Memory & Compute Bottleneck of Traditional RLHF:** In standard PPO, training a $70\text{B}$ Actor requires loading an equally large $70\text{B}$ Critic into GPU memory, doubling VRAM requirements and communication overhead.
- **GRPO Architecture (Eliminating the Critic):**
  1. For each prompt $q$, sample a group of $G$ independent candidate responses: $\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(q)$.
  2. Evaluate each response using verifiable reward functions (e.g., mathematical proof verifier, unit test execution, or reward model): $\{r_1, r_2, \dots, r_G\}$.
  3. Compute normalized advantages **within the group** using empirical group mean and standard deviation:
     $$\hat{A}_i = \frac{r_i - \operatorname{mean}(\{r_1, \dots, r_G\})}{\operatorname{std}(\{r_1, \dots, r_G\}) + \epsilon}$$
  4. Optimize the policy using the clipped surrogate objective with an analytical reverse KL divergence penalty:
     $$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( \frac{\pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})} \hat{A}_i, \, \operatorname{clip}\left( \dots \right) \hat{A}_i \right) - \beta D_{\text{KL}}\left( \pi_\theta \,\|\, \pi_{\text{ref}} \right) \right]$$
- **Emergence of Complex Reasoning Behaviors:** How GRPO enables large-scale self-verification, backtracking, and long-chain thought formulation (e.g., DeepSeek-R1, OpenAI o1).
- **Tom Yeh Visual Grid:** Group of $G=4$ outputs, reward scoring, empirical group mean and std calculation, normalized advantage assignment, and token-level clipped gradient update by hand.
- **Code & Test Suite:** Complete, standalone PyTorch GRPO implementation verifying group normalization, clipping, and policy parameter convergence without a Critic network.

---

## 3. Directory Layout in Workspace (`AI_MATH`)

```
/Users/kunalkumar/Desktop/AI_MATH/
├── 00_index_and_tracker.md              <-- Master Course Tracker
├── reinforcement_learning_curriculum.md <-- Master RL Plan & Syllabus
└── 11_reinforcement_learning/           <-- Module 11 Directory
    ├── 01_multi_armed_bandits.md
    ├── 02_markov_decision_processes.md
    ├── 03_bellman_equations.md
    ├── 04_dynamic_programming_and_contraction.md
    ├── 05_monte_carlo_methods.md
    ├── 06_temporal_difference_learning.md
    ├── 07_sarsa_and_q_learning.md
    ├── 08_multi_step_and_eligibility_traces.md
    ├── 09_function_approximation_and_deadly_triad.md
    ├── 10_deep_q_networks.md
    ├── 11_rainbow_dqn_extensions.md
    ├── 12_distributional_rl.md
    ├── 13_policy_gradient_and_reinforce.md
    ├── 14_advantage_actor_critic.md
    ├── 15_generalized_advantage_estimation.md
    ├── 16_continuous_control_ddpg_td3.md
    ├── 17_natural_policy_gradients.md
    ├── 18_trust_region_policy_optimization.md
    ├── 19_proximal_policy_optimization.md
    ├── 20_soft_actor_critic.md
    ├── 21_model_based_rl_and_dyna.md
    ├── 22_trajectory_optimization_mpc.md
    ├── 23_mcts_and_alphazero.md
    ├── 24_world_models_and_dreamer.md
    ├── 25_offline_rl_and_cql.md
    ├── 26_decision_transformers.md
    ├── 27_rlhf_with_ppo.md
    ├── 28_grpo_and_frontier_reasoning.md
    └── code/                             <-- Vectorized PyTorch Test Suite
        ├── 01_multi_armed_bandits.py
        ├── ...
        └── 28_grpo_and_frontier_reasoning.py
```
