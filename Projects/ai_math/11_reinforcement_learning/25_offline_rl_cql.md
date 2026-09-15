# Chapter 25: Offline (Batch) RL & Conservative Q-Learning (CQL)

---

## 1. Intuition & 101 Motivation

Throughout this curriculum, every reinforcement learning algorithm we examined—from Q-learning and DQN to PPO and SAC—relied on **online interaction**: the agent gathers experience in an environment, updates its parameters, and immediately tries out its new policy.

However, in many of the most critical real-world domains, **exploratory online trial-and-error is dangerous, unethical, or prohibitively expensive**:
- **Healthcare & Clinical Medicine:** An RL agent cannot experiment with untested drug dosages on human patients to see if they survive.
- **Autonomous Driving:** A self-driving vehicle cannot deliberately execute dangerous swerves on a highway to learn how to recover from a skid.
- **Nuclear Power & Industrial Control:** An algorithm cannot cause reactor meltdowns to explore its failure boundaries.

Fortunately, in these domains we already possess **massive historical offline datasets** $\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1})\}$ collected by human doctors, expert human drivers, or rule-based legacy systems.

The ambition of **Offline Reinforcement Learning (Batch RL)** is:
> *Can we learn an optimal decision-making policy $\pi^*$ strictly from a static dataset $\mathcal{D}$, without taking a single online step in the environment?*

### Why Standard Off-Policy RL Fails: The OOD Overestimation Trap
Naively, algorithms like DQN or SAC are supposed to be "off-policy," meaning their Bellman equations theoretically permit training on arbitrary replay buffers. Why can't we simply train DQN or SAC on an offline dataset?

The answer is **Distributional Shift** and **Out-of-Distribution (OOD) Overestimation**:
1. When computing the Bellman target:
   $$y = r + \gamma \max_{a'} Q_\theta(s', a')$$
   the $\max$ operator queries the neural network $Q_\theta$ at actions $a'$ that may **never exist in the dataset $\mathcal{D}$**.
2. Deep neural networks do not generalize safely outside their training support. For unseen OOD actions $a_{\text{ood}}$, function approximation errors cause random spikes in $Q(s', a_{\text{ood}})$.
3. Because of the maximization operator ($\max$), the algorithm systematically latches onto whichever unseen action received the highest hallucinated value.
4. The Bellman error bootstraps on this hallucination: $Q(s, a) \leftarrow r + \gamma Q(s', a_{\text{ood}})$.
5. Over thousands of gradient steps, estimated $Q$-values explode to $+10^{6}$, while the true policy performance collapses to zero!

```
Q(s, a)
 ^
 |             /-- Massive OOD Hallucination Peak! (No data here)
 |            /
 |   * * *   /       * * *   <-- Real Data Support (pi_beta)
 |  *  *  * /       *  *  *
 |________________________________> Action Space a
```

Enter **Conservative Q-Learning (CQL)** (Kumar et al., NeurIPS 2020):
Instead of trusting unconstrained function approximation, CQL introduces an explicit regularizer that **actively penalizes $Q$-values on out-of-distribution actions**, mathematically guaranteeing that the learned $Q$-function is a **provable lower bound** on the true value function!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Offline RL Formulation & Distributional Shift

Let the offline dataset be $\mathcal{D} = \{(s, a, r, s')\} \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$, where $\pi_\beta$ is the unknown **behavior policy** that generated the data.

When the learned policy $\pi_\theta$ differs from $\pi_\beta$, the state-action visitation distribution shifts:
$$d^{\pi_\theta}(s, a) \neq d^{\pi_\beta}(s, a)$$

For any state $s$, define the support of the behavior policy as:
$$\operatorname{supp}(\pi_\beta(\cdot \mid s)) = \{ a \in \mathcal{A} \mid \pi_\beta(a \mid s) > 0 \}$$

Any action $a \notin \operatorname{supp}(\pi_\beta(\cdot \mid s))$ is **Out-of-Distribution (OOD)**.

---

### 2.2 The Conservative Q-Learning (CQL) Framework

The core philosophy of CQL is simple yet profound:
- **Push down** the $Q$-values of unvisited or high-scoring actions.
- **Push up** the $Q$-values of actions that actually appear in the dataset $\mathcal{D}$.

#### The Discrete CQL Formulation:
For discrete action spaces, CQL augments the standard Bellman Temporal-Difference loss with a conservative regularizer:

$$\min_{Q} \mathcal{L}_{\text{CQL}}(Q) = \alpha \cdot \underbrace{\mathbb{E}_{s \sim \mathcal{D}} \left[ \ln \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_\beta(a \mid s)} [Q(s, a)] \right]}_{\text{Conservative Regularizer}} + \frac{1}{2} \underbrace{\mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^{\pi} \bar{Q}(s, a) \right)^2 \right]}_{\text{Standard Bellman TD Error}}$$

where:
- $\hat{\mathcal{B}}^{\pi} \bar{Q}(s, a) = r + \gamma \mathbb{E}_{a' \sim \pi(a' \mid s')} [\bar{Q}(s', a')]$ is the Bellman target evaluated with target network $\bar{Q}$.
- $\alpha > 0$ is a trade-off hyperparameter governing the strength of conservatism.
- $\ln \sum_{a} \exp(Q(s, a))$ is the differentiable **Log-Sum-Exp** soft-maximum over all actions in the action space.
- $\mathbb{E}_{a \sim \pi_\beta}[Q(s, a)] = \frac{1}{|\mathcal{D}(s)|} \sum_{a \in \mathcal{D}(s)} Q(s, a)$ is the empirical average over actions observed in the dataset at state $s$.

---

### 2.3 Mathematical Mechanics of the CQL Regularizer

Let us differentiate the regularizer $\mathcal{R}(Q) = \ln \sum_{a} \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_\beta} [Q(s, a)]$ with respect to the action-value $Q(s, a_i)$:

$$\frac{\partial}{\partial Q(s, a_i)} \left( \ln \sum_{a} \exp(Q(s, a)) \right) = \frac{\exp(Q(s, a_i))}{\sum_{b} \exp(Q(s, b))} = \operatorname{softmax}(Q(s, \cdot))_i$$

$$\frac{\partial}{\partial Q(s, a_i)} \left( \mathbb{E}_{a \sim \pi_\beta} [Q(s, a)] \right) = \pi_\beta(a_i \mid s)$$

Therefore, the exact gradient of the conservative regularizer is:

$$\nabla_{Q(s, a_i)} \mathcal{R}(Q) = \underbrace{\frac{\exp(Q(s, a_i))}{\sum_{b} \exp(Q(s, b))}}_{p_{\text{softmax}}(a_i \mid s)} - \underbrace{\pi_\beta(a_i \mid s)}_{\text{empirical data frequency}}$$

#### The Two Distinct Regimes:
1. **Case 1: Out-of-Distribution Action ($a_{\text{ood}} \notin \mathcal{D}$):**
   - Data frequency is zero: $\pi_\beta(a_{\text{ood}} \mid s) = 0$.
   - The gradient is strictly positive: $\nabla_{Q} \mathcal{R} = p_{\text{softmax}}(a_{\text{ood}}) > 0$.
   - A gradient descent step pushes $Q(s, a_{\text{ood}})$ **DOWNWARD**:
     $$Q(s, a_{\text{ood}}) \leftarrow Q(s, a_{\text{ood}}) - \eta \alpha p_{\text{softmax}}(a_{\text{ood}}) \quad \downarrow$$
2. **Case 2: In-Distribution Action ($a_{\text{data}} \in \mathcal{D}$):**
   - For actions with high data coverage, $\pi_\beta(a_{\text{data}} \mid s) > p_{\text{softmax}}(a_{\text{data}} \mid s)$.
   - The gradient is negative: $\nabla_{Q} \mathcal{R} < 0$.
   - A gradient descent step pushes $Q(s, a_{\text{data}})$ **UPWARD**:
     $$Q(s, a_{\text{data}}) \leftarrow Q(s, a_{\text{data}}) + \eta \alpha (\pi_\beta - p_{\text{softmax}}) \quad \uparrow$$

**Conclusion:** CQL automatically depresses the values of hallucinated unseen actions while elevating the values of safe, data-supported actions!

---

### 2.4 Theorem 25.1 (CQL Provable Value Lower Bound Guarantee)

#### Theorem:
Let $\hat{Q}_{\text{CQL}}^\pi$ be the fixed point of the CQL Bellman operator. For any policy $\pi$, if $\alpha \ge \frac{C_{\text{Bellman}}}{1 - \gamma}$, then for all states $s$:

$$\mathbb{E}_{a \sim \pi(a \mid s)} \left[ \hat{Q}_{\text{CQL}}^\pi(s, a) \right] \le V^\pi(s)$$

where $V^\pi(s)$ is the true performance of policy $\pi$ in the real environment.

#### Proof Intuition:
At every iteration, the regularizer introduces an expected negative shift $\Delta(s) \le 0$ on the state value under $\pi$. Because the Bellman operator is a $\gamma$-contraction in the $L_\infty$ norm, recursive propagation accumulates these negative shifts:
$$\hat{V}_{\text{CQL}}^\pi(s) = V^\pi(s) - \sum_{t=0}^\infty \gamma^t P^\pi \Delta \le V^\pi(s) \quad \blacksquare$$

This guarantees that an actor policy optimized against $\hat{Q}_{\text{CQL}}$ will **never be fooled by optimistic OOD hallucinations**.

---

## 3. Geometric & Physical Interpretation

### 3.1 The Energy Basin of In-Distribution Data
Think of the action-value space as an energy surface over actions:
```
Action Value Q(s, a)
 ^
 |      [In-Data: a_1]             [OOD: a_2]
 |           |                         |
 |           v                         v
 |       +-------+                 +-------+
 |       | Q=2.0 |                 | Q=5.0 |  <-- Hallucinated Peak!
 |       +-------+                 +-------+
 |           |                         |
 |           | (Push UP: +0.48)        | (Crushed DOWN: -0.48)
 |           v                         v
 |       +-------+                 +-------+
 |       | Q=2.48|                 | Q=4.52|
 |       +-------+                 +-------+
```
- CQL applies a selective gravitational pull: actions without data support are dragged down into the floor, creating a safe, convex **basin of trust** centered squarely on the behavior policy manifold.

---

## 4. Real-World Analogy: Medical Prescription AI

Consider an AI physician analyzing clinical hospital records:
- **Patient State:** Stage-2 Hypertension.
- **Action $a_1$ (Standard Protocol):** Prescribe 10 mg Lisinopril. Observed in 1,000 patient records with positive outcome ($Q = 8.5$).
- **Action $a_2$ (Untested Crazy Dosage):** Prescribe 500 mg Lisinopril. Observed $0$ times in human history.
- **Standard Unregularized Q-learning:** The neural network extrapolates wildly on 500 mg, predicting an outcome score of $+999.0$. The unregularized AI prescribes the lethal overdose!
- **Conservative Q-Learning (CQL):** Recognizing that $a_2$ has zero evidence in $\mathcal{D}$, the Log-Sum-Exp penalty forcefully suppresses its value: $Q(s, a_2) \leftarrow -100.0$. The AI safely prescribes the 10 mg clinical protocol.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical gradient step of **Conservative Q-Learning (CQL)** by hand on a concrete 1-state, 2-action problem.

---

### 5.1 System & Initial Setup
- **State:** Single state $S_0$
- **Actions:** $\mathcal{A} = \{a_1, a_2\}$
  - $a_1$: **In-Distribution Action** (observed in dataset $\mathcal{D}$ with empirical probability $\pi_\beta(a_1 \mid S_0) = 1.0$)
  - $a_2$: **Out-of-Distribution (OOD) Action** (never observed, $\pi_\beta(a_2 \mid S_0) = 0.0$)
- **Current $Q$-values (with OOD Hallucination):**
  $$Q(S_0, a_1) = 2.0000 \quad (\text{True data-supported value})$$
  $$Q(S_0, a_2) = 5.0000 \quad (\text{Hallucinated high value!})$$
- **TD Target:** Assume Bellman TD error on $a_1$ is $0.0$ (i.e. $Q(S_0, a_1) = \text{Target} = 2.0000$).
- **Hyperparameters:**
  - Conservatism coefficient: $\alpha = 1.0000$
  - Learning rate: $\eta = 0.5000$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $Q(S_0, a)$ | `q_vals[a]` | Action-value estimates before update ($Q(a_1)=2.0, Q(a_2)=5.0$) |
| $\pi_\beta(a \mid S_0)$ | `data_prob[a]` | Empirical data frequency ($\pi_\beta(a_1)=1.0, \pi_\beta(a_2)=0.0$) |
| $p_{\text{softmax}}(a)$ | `softmax_prob[a]` | Model softmax policy: $\exp(Q(a)) / \sum \exp(Q(b))$ |
| $\nabla_{Q(a)} \mathcal{R}$ | `grad_cql[a]` | CQL regularizer gradient: $p_{\text{softmax}}(a) - \pi_\beta(a)$ |
| $Q_{\text{new}}(S_0, a)$ | `q_new[a]` | Updated $Q$-value: $Q - \eta \cdot \alpha \cdot \nabla_{Q(a)} \mathcal{R}$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Exponentials & Log-Sum-Exp
$$e^{Q(S_0, a_1)} = e^{2.0} \approx \mathbf{7.389056}$$
$$e^{Q(S_0, a_2)} = e^{5.0} \approx \mathbf{148.413159}$$
$$\sum_{a} e^{Q(S_0, a)} = 7.389056 + 148.413159 = \mathbf{155.802215}$$
$$\text{Log-Sum-Exp} = \ln(155.802215) \approx \mathbf{5.048615}$$

#### Step 2: Compute Softmax Action Probabilities
$$p_{\text{softmax}}(a_1) = \frac{7.389056}{155.802215} \approx \mathbf{0.047426}$$
$$p_{\text{softmax}}(a_2) = \frac{148.413159}{155.802215} \approx \mathbf{0.952574}$$

*(Notice how the neural network strongly prefers the hallucinated OOD action $a_2$ with 95.3% probability!)*

#### Step 3: Compute CQL Gradients
$$\nabla_{Q(a_1)} \mathcal{R} = p_{\text{softmax}}(a_1) - \pi_\beta(a_1) = 0.047426 - 1.000000 = \mathbf{-0.952574}$$
$$\nabla_{Q(a_2)} \mathcal{R} = p_{\text{softmax}}(a_2) - \pi_\beta(a_2) = 0.952574 - 0.000000 = \mathbf{+0.952574}$$

#### Step 4: Execute Conservative Gradient Update
With learning rate $\eta = 0.5000$ and $\alpha = 1.0000$:

1. **Update for In-Distribution Action $a_1$:**
   $$Q_{\text{new}}(S_0, a_1) = Q(S_0, a_1) - \eta \alpha \nabla_{Q(a_1)} \mathcal{R}$$
   $$= 2.0000 - 0.5000 \times (-0.952574) = 2.0000 + 0.476287 = \mathbf{2.476287} \quad (\uparrow \text{ Boosted!})$$

2. **Update for Out-of-Distribution Action $a_2$:**
   $$Q_{\text{new}}(S_0, a_2) = Q(S_0, a_2) - \eta \alpha \nabla_{Q(a_2)} \mathcal{R}$$
   $$= 5.0000 - 0.5000 \times (+0.952574) = 5.0000 - 0.476287 = \mathbf{4.523713} \quad (\downarrow \text{ Penalized!})$$

---

### 5.4 Summary Visual Grid: CQL Gradient Descent Ledger

| Action $a$ | Data Type | Prior $Q(a)$ | $\exp(Q)$ | Softmax Prob $p(a)$ | Data Prob $\pi_\beta(a)$ | Gradient $\nabla_Q \mathcal{R}$ | Step $\Delta Q$ | Updated $Q_{\text{new}}(a)$ | Effect |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$a_1$** | **In-Data** | $2.0000$ | $7.3891$ | $0.0474$ | **$1.0000$** | **$-0.9526$** | $+0.4763$ | **$2.4763$** | $\mathbf{\uparrow}$ Elevated |
| **$a_2$** | **OOD** | $5.0000$ | $148.4132$ | $0.9526$ | **$0.0000$** | **$+0.9526$** | $-0.4763$ | **$4.5237$** | $\mathbf{\downarrow}$ Crushed |

---

## 6. Solved Illustrations

### Illustration 1: The Role of Hyperparameter $\alpha$
**Problem:**
What happens at the theoretical limits of the conservatism weight $\alpha$?
1. $\alpha \to 0$
2. $\alpha \to \infty$

**Solution:**
1. **$\alpha \to 0$:** CQL regularizer vanishes. The algorithm collapses to standard unconstrained off-policy Q-learning, suffering immediate OOD divergence and policy collapse.
2. **$\alpha \to \infty$:** The regularizer completely dominates the TD error:
   $$\min_Q \mathbb{E}_s \left[ \ln \sum_a \exp(Q(s, a)) - \mathbb{E}_{\pi_\beta}[Q(s, a)] \right]$$
   The unique global minimum occurs when $p_{\text{softmax}}(a \mid s) \equiv \pi_\beta(a \mid s)$. The policy collapses to pure **Behavior Cloning (Supervised Imitation)**, losing the ability to stitch together and outperform the sub-optimal dataset. $\blacksquare$

---

## 7. Deep RL Connection & Modern Applications

- **D4RL (Datasets for Deep Data-Driven RL, Fu et al., 2020):**
  The standard benchmark for offline RL, containing millions of transitions across MuJoCo locomotion, AntMaze navigation, Adroit robotic dexterous hand manipulation, and Kitchen tasks. CQL established the foundational baseline by vastly outperforming online algorithms trained offline.
- **Autonomous Driving (Waymo, Cruise):**
  Offline RL models are trained on petabytes of real human driving logs to evaluate candidate vehicle trajectories without deploying risky exploratory policies on real streets.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical calculation of softmax probabilities $p(a_1)=0.047426, p(a_2)=0.952574$.
   - Exact gradient step matching $Q_{\text{new}}(a_1)=2.476287, Q_{\text{new}}(a_2)=4.523713$ to $< 10^{-6}$.
2. **Complete PyTorch CQL Discrete Agent:**
   - Deep Q-Network with conservative log-sum-exp regularization loss.
3. **Synthetic Offline RL Environment Benchmark:**
   - Demonstrates that standard DQN diverges on an offline dataset with OOD actions, whereas CQL safely suppresses OOD actions and converges to the optimal safe policy.

See implementation in:
[`11_reinforcement_learning/code/25_offline_rl_cql.py`](./code/25_offline_rl_cql.py)
