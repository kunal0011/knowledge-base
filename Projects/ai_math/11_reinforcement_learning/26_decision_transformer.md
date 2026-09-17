# Chapter 26: Decision Transformers & Sequence Modeling RL

---

## 1. Intuition & 101 Motivation

For decades, reinforcement learning was framed exclusively through the lens of **Dynamic Programming and the Bellman Equation**:
$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$

This formulation requires temporal-difference bootstrapping, actor-critic architectures, discount factors, target networks, and replay buffers. When scaled to large neural networks, this paradigm often suffers from numerical instabilities—most notoriously the **Deadly Triad** (bootstrapping + function approximation + off-policy distribution shift).

In 2021, a revolutionary paradigm emerged: **Can we throw away the Bellman equation entirely and solve Reinforcement Learning as generic autoregressive Sequence Modeling?**

Enter the **Decision Transformer (DT)** (Chen et al., NeurIPS 2021):
- An entire RL episode is transcribed into a sequence of tokens:
  $$\tau = \big( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \big)$$
  where $\hat{R}_t = \sum_{t'=t}^T r_{t'}$ is the **Return-to-Go** (the sum of future rewards remaining in the episode).
- A standard causal autoregressive Transformer (identical to GPT) is trained on offline trajectories using simple **supervised regression / cross-entropy loss**.
- At test time, to generate expert behavior, we simply condition the Transformer on a high desired return:
  $$\text{Prompt: } \text{"Generate actions that achieve } \hat{R}_1 = 100.0\text{"}$$
- The Transformer autoregressively predicts the exact sequence of actions required to satisfy that return prompt!

```
Target Return R_1 ---> State s_1 ---> [ Transformer (GPT) ] ---> Action a_1
                                                                     |
Environment Step <---------------------------------------------------+
       |
       v
Real Reward r_1 & Next State s_2
       |
       v
New Return R_2 = R_1 - r_1 ---> State s_2 ---> [ Transformer ] ---> Action a_2
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Trajectory Representation & Return-to-Go

Let a collected episode of length $T$ consist of states $s_t \in \mathbb{R}^{d_s}$, actions $a_t \in \mathbb{R}^{d_a}$, and scalar stage rewards $r_t \in \mathbb{R}$.

The **Return-to-Go (RTG)** at timestep $t$ is the exact empirical sum of future rewards:
$$\hat{R}_t \triangleq \sum_{t'=t}^T r_{t'}$$

It satisfies the deterministic backward recurrence:
$$\hat{R}_t = r_t + \hat{R}_{t+1} \iff \hat{R}_{t+1} = \hat{R}_t - r_t$$

The trajectory is ordered chronologically into a sequence of triplets:
$$\tau = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \right)$$

To enable finite-length transformer processing, the model attends over a sliding context window of the most recent $K$ timesteps, resulting in a sequence of $3K$ tokens:
$$\mathbf{x} = \left[ \hat{R}_{t-K+1}, s_{t-K+1}, a_{t-K+1}, \dots, \hat{R}_t, s_t \right]$$

---

### 2.2 Token Embedding & Modality Encodings

Because the three modalities (returns, states, actions) have different physical dimensions, each modality is passed through a distinct linear projection matrix:

$$e_t^R = W_R \hat{R}_t + e_t^{\text{pos}}$$
$$e_t^s = W_s s_t + e_t^{\text{pos}}$$
$$e_t^a = W_a a_t + e_t^{\text{pos}}$$

where:
- $W_R \in \mathbb{R}^{d \times 1}$, $W_s \in \mathbb{R}^{d \times d_s}$, $W_a \in \mathbb{R}^{d \times d_a}$ project each token into a shared hidden dimension $d$.
- $e_t^{\text{pos}} \in \mathbb{R}^d$ is a learnable or sinusoidal **timestep positional embedding** corresponding to episode step $t$. (Critically, all three tokens belonging to the same timestep $t$ share the exact same positional embedding $e_t^{\text{pos}}$).

The input token sequence fed into the Transformer is:
$$\mathbf{E} = \left[ e_1^R, e_1^s, e_1^a, e_2^R, e_2^s, e_2^a, \dots, e_K^R, e_K^s \right] \in \mathbb{R}^{N \times d}$$

---

### 2.3 Causal Masking & Supervised Training Objective

The Transformer blocks use standard **multi-head causal self-attention**:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

where $M$ is the upper-triangular causal mask ($M_{i, j} = -\infty$ for $j > i$).

Under this causal structure:
- Token $e_t^s$ can attend to all past tokens and $\hat{R}_t$, but **cannot see future tokens or its own action $a_t$**.
- Token $e_t^s$ is directly used to predict action $a_t$ via a linear action head $W_{\text{act}}$:
  $$\hat{a}_t = W_{\text{act}} h_t^s$$
  where $h_t^s \in \mathbb{R}^d$ is the Transformer output representation corresponding to token $e_t^s$.

#### The Training Objective:
The entire architecture is trained via standard supervised regression (MSE for continuous actions, Cross-Entropy for discrete actions):

$$\mathcal{L}_{\text{DT}}(\theta) = \frac{1}{T} \sum_{t=1}^T \left\| a_t - \hat{a}_t \left( \hat{R}_{1:t}, s_{1:t}, a_{1:t-1} \right) \right\|_2^2$$

Notice what is absent:
- **No Bellman equations.**
- **No value bootstrapping.**
- **No policy gradient estimators.**
- **No discount factor $\gamma$.**
Training is as stable and scalable as training a standard GPT language model!

---

### 2.4 Test-Time Autoregressive Rollout Protocol

To deploy a trained Decision Transformer in an environment:
1. **Prompt Initialization:** The user chooses a target return $\hat{R}_1$ (e.g. the maximum return in the training dataset, or an ambitious expert target).
2. **Observe Initial State:** Receive $s_1$ from the environment.
3. **Action Generation:** Pass $[\hat{R}_1, s_1]$ to the Transformer. Predict $\hat{a}_1$.
4. **Environment Execution:** Execute $\hat{a}_1$, observe environment reward $r_1$ and next state $s_2$.
5. **Return-to-Go Update:** Decrement the return-to-go:
   $$\hat{R}_2 \leftarrow \hat{R}_1 - r_1$$
6. **Context Extension:** Append $(\hat{a}_1, \hat{R}_2, s_2)$ to the context window and repeat until episode termination.

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.26.1: Autoregressive Sequence Modeling Formulation of RL as Conditional Imitation

```
====================================================================================================
DERIVATION 11.26.1: Autoregressive Sequence Modeling Formulation of RL as Conditional Imitation
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
Consider an offline reinforcement learning dataset $\mathcal{D} = \{\tau^{(i)}\}_{i=1}^M$ of trajectories generated by an unknown behavior policy $\pi_\beta$ in a Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \rho_0, \gamma)$. Each trajectory of horizon $T$ is transcribed as a chronological sequence of return-to-go, state, and action triplets:
$$\tau = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \right)$$
where the scalar Return-to-Go (RTG) at step $t$ is defined as the empirical sum of future observed rewards:
$$\hat{R}_t \triangleq \sum_{t'=t}^T r_{t'}$$
Our mathematical goal is to:
1. Formulate the joint trajectory probability density $P_\theta(\tau)$ under an autoregressive generative model parameterized by $\theta$.
2. Prove that under standard conditional independence properties of the MDP and deterministic return dynamics, maximizing the joint log-likelihood of the offline dataset over agent policy parameters $\theta$ strictly decouples from environment transition dynamics and reduces to the autoregressive action-sequence prediction objective:
   $$\mathcal{L}_{\text{DT}}(\theta) = -\sum_{t=1}^T \log P_\theta\left(a_t \;\middle|\; \hat{R}_1, s_1, a_1, \dots, \hat{R}_t, s_t\right)$$
3. Prove that under a homoscedastic conditional Gaussian distribution over continuous actions, this maximum likelihood objective is algebraically equivalent to the mean-squared error (MSE) sequence loss:
   $$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{T} \sum_{t=1}^T \left\| a_t - \hat{a}_\theta\left( \hat{R}_{1:t}, s_{1:t}, a_{1:t-1} \right) \right\|_2^2$$

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Offline Data Generation:** Trajectories $\tau^{(i)} \sim P^{\pi_\beta}(\tau)$ are drawn independently and identically distributed from the historical occupancy distribution of a stationary or non-stationary behavior policy $\pi_\beta$.
2. **Deterministic Return Recurrence:** The return-to-go satisfies the exact backward identity $\hat{R}_{t+1} = \hat{R}_t - r_t$. Thus, the conditional transition distribution of the return-to-go is a Dirac delta:
   $$P\left(\hat{R}_{t+1} \;\middle|\; \tau_{\le t}, r_t\right) = \delta\left(\hat{R}_{t+1} - (\hat{R}_t - r_t)\right)$$
3. **Environment Dynamics Invariance (Exogenous Transitions):** The environmental transition density $\mathcal{P}(s_{t+1} \mid s_t, a_t)$ is a fixed property of the physical world and does not depend on model parameters $\theta$:
   $$\nabla_\theta \mathcal{P}(s_{t+1} \mid s_t, a_t) = \mathbf{0}$$
4. **Causal Non-Anticipating Policy Head:** The policy density at timestep $t$ conditions strictly on the observed historical trajectory prefix up to state $s_t$ and target $\hat{R}_t$:
   $$\mathbf{h}_t \triangleq \left( \hat{R}_1, s_1, a_1, \dots, \hat{R}_{t-1}, s_{t-1}, a_{t-1}, \hat{R}_t, s_t \right) = \left( \hat{R}_{1:t}, s_{1:t}, a_{1:t-1} \right)$$
   and cannot attend to current action $a_t$ or any future quantities ($s_{t'>t}, a_{t'>t}, \hat{R}_{t'>t}$).
5. **Conditional Gaussian Density:** For continuous action space $\mathcal{A} \subseteq \mathbb{R}^{d_a}$, the conditional distribution is assumed to be an isotropic Gaussian:
   $$P_\theta(a_t \mid \mathbf{h}_t) = \mathcal{N}\left( a_t \;\middle|\; \hat{a}_\theta(\mathbf{h}_t), \sigma^2 \mathbf{I}_{d_a} \right) = \frac{1}{(2\pi \sigma^2)^{d_a/2}} \exp\left( -\frac{1}{2\sigma^2} \left\| a_t - \hat{a}_\theta(\mathbf{h}_t) \right\|_2^2 \right)$$
   with fixed variance $\sigma^2 > 0$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
In standard Reinforcement Learning, the policy is optimized to maximize expected value $J(\pi) = \mathbb{E}_\pi[\sum_t \gamma^t r_t]$ via temporal difference bootstrapping or policy gradients. In contrast, Decision Transformer reframes control as **conditional generative modeling**:
- **Behavior Cloning (BC) Failure:** Standard BC fits the marginal action policy $P(a_t \mid s_t) = \int P(a_t \mid s_t, R) P(R \mid s_t) dR$. When the dataset contains a mixture of expert runs ($R = 100$) and novice collisions ($R = 0$), BC averages over all returns, yielding a multi-modal or blurred policy that drives into obstacles.
- **Conditional Slicing of the Trajectory Manifold:** By conditioning on scalar return-to-go $\hat{R}_t$, trajectory space $\mathcal{T}$ is partitioned into level sets (iso-return manifolds) $\mathcal{M}_R = \{ \tau \in \mathcal{T} \mid R(\tau) = R \}$. Maximizing conditional likelihood $P(a_t \mid s_t, \hat{R}_t)$ trains the model to associate specific action patterns with specific cumulative outcomes. At inference time, setting $\hat{R}_1 = R_{\text{target}}$ selects the desired high-return manifold slice, forcing the model to reproduce expert actions without ever computing a Bellman backup.

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Autoregressive Factorization of the Trajectory Joint Distribution.*
Let the trajectory be written as a sequence of $3T$ random variables:
$$\tau = \left( X_1, X_2, \dots, X_{3T} \right) = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \right)$$
where for timestep $t \in \{1, \dots, T\}$:
$$X_{3t-2} = \hat{R}_t, \quad X_{3t-1} = s_t, \quad X_{3t} = a_t$$
By the exact chain rule of conditional probability:
$$P_\theta(\tau) = \prod_{k=1}^{3T} P_\theta\left( X_k \;\middle|\; X_1, X_2, \dots, X_{k-1} \right)$$
Grouping the terms by timestep $t \in \{1, \dots, T\}$:
$$P_\theta(\tau) = \prod_{t=1}^T \underbrace{P_\theta\left( \hat{R}_t \;\middle|\; \tau_{<t} \right)}_{(A)} \cdot \underbrace{P_\theta\left( s_t \;\middle|\; \tau_{<t}, \hat{R}_t \right)}_{(B)} \cdot \underbrace{P_\theta\left( a_t \;\middle|\; \tau_{<t}, \hat{R}_t, s_t \right)}_{(C)}$$
where $\tau_{<t} \triangleq (\hat{R}_1, s_1, a_1, \dots, \hat{R}_{t-1}, s_{t-1}, a_{t-1})$.

*Step 2: Analysis of Component Factors under the MDP Structure.*
We evaluate each factor individually:
1. **Factor (A) - Return-to-Go Dynamics:**
   For $t = 1$, $\hat{R}_1 \sim P(\hat{R}_1)$ is the marginal return distribution of the behavior policy.
   For $t \ge 2$, by Assumption 2, $\hat{R}_t = \hat{R}_{t-1} - r_{t-1}$. Since $r_{t-1} = \mathcal{R}(s_{t-1}, a_{t-1})$ is fixed given $(s_{t-1}, a_{t-1}) \in \tau_{<t}$, this transition is a deterministic physical identity independent of $\theta$:
   $$P_\theta\left(\hat{R}_t \;\middle|\; \tau_{<t}\right) = \delta\left(\hat{R}_t - (\hat{R}_{t-1} - r_{t-1})\right)$$
   Because this Dirac delta contains no trainable parameters:
   $$\nabla_\theta \log P_\theta\left(\hat{R}_t \;\middle|\; \tau_{<t}\right) = \mathbf{0}$$
2. **Factor (B) - Environmental State Transition Dynamics:**
   Under the Markov assumption of the environment, $s_t$ depends physically only on $(s_{t-1}, a_{t-1})$:
   $$P_\theta\left( s_t \;\middle|\; \tau_{<t}, \hat{R}_t \right) = \mathcal{P}\left( s_t \;\middle|\; s_{t-1}, a_{t-1} \right)$$
   By Assumption 3, the physical environment dynamics $\mathcal{P}$ are exogenous and independent of policy parameters $\theta$:
   $$\nabla_\theta \log P_\theta\left( s_t \;\middle|\; \tau_{<t}, \hat{R}_t \right) = \nabla_\theta \log \mathcal{P}\left( s_t \;\middle|\; s_{t-1}, a_{t-1} \right) = \mathbf{0}$$
3. **Factor (C) - Policy Action Distribution:**
   This factor represents the agent's conditional action generation given past experience and the desired target return:
   $$P_\theta\left( a_t \;\middle|\; \tau_{<t}, \hat{R}_t, s_t \right) = P_\theta\left( a_t \;\middle|\; \mathbf{h}_t \right)$$
   where $\mathbf{h}_t \triangleq (\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$ is the causal context history. This is the only factor parameterized by $\theta$.

*Step 3: Trajectory Log-Likelihood and Objective Isolation.*
Taking the natural logarithm of the joint probability:
$$\log P_\theta(\tau) = \sum_{t=1}^T \left[ \log P\left(\hat{R}_t \;\middle|\; \tau_{<t}\right) + \log \mathcal{P}\left(s_t \;\middle|\; s_{t-1}, a_{t-1}\right) + \log P_\theta\left(a_t \;\middle|\; \mathbf{h}_t\right) \right]$$
Taking the gradient with respect to $\theta$:
$$\nabla_\theta \log P_\theta(\tau) = \sum_{t=1}^T \Big( \underbrace{\nabla_\theta \log P\left(\hat{R}_t \;\middle|\; \tau_{<t}\right)}_{\mathbf{0}} + \underbrace{\nabla_\theta \log \mathcal{P}\left(s_t \;\middle|\; s_{t-1}, a_{t-1}\right)}_{\mathbf{0}} + \nabla_\theta \log P_\theta\left(a_t \;\middle|\; \mathbf{h}_t\right) \Big)$$
$$\nabla_\theta \log P_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log P_\theta\left(a_t \;\middle|\; \mathbf{h}_t\right)$$
Therefore, the maximum likelihood estimation problem over dataset $\mathcal{D}$:
$$\theta^* = \arg\max_\theta \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \log P_\theta(\tau) \right]$$
is identically equivalent to minimizing the negative log-likelihood of action prediction:
$$\mathcal{L}_{\text{DT}}(\theta) \triangleq -\frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=1}^T \log P_\theta\left(a_t \;\middle|\; \hat{R}_{1:t}, s_{1:t}, a_{1:t-1}\right)$$

*Step 4: Continuous Actions and Equivalence to Mean-Squared Error.*
Now substitute the conditional Gaussian density from Assumption 5:
$$P_\theta(a_t \mid \mathbf{h}_t) = \frac{1}{(2\pi \sigma^2)^{d_a/2}} \exp\left( -\frac{1}{2\sigma^2} \left\| a_t - \hat{a}_\theta(\mathbf{h}_t) \right\|_2^2 \right)$$
Taking the natural logarithm:
$$\log P_\theta(a_t \mid \mathbf{h}_t) = -\frac{d_a}{2} \ln(2\pi \sigma^2) - \frac{1}{2\sigma^2} \left\| a_t - \hat{a}_\theta(\mathbf{h}_t) \right\|_2^2$$
Substituting into the negative log-likelihood objective:
$$\mathcal{L}_{\text{DT}}(\theta) = -\sum_{t=1}^T \left[ -\frac{d_a}{2} \ln(2\pi \sigma^2) - \frac{1}{2\sigma^2} \left\| a_t - \hat{a}_\theta(\mathbf{h}_t) \right\|_2^2 \right]$$
$$\mathcal{L}_{\text{DT}}(\theta) = \frac{T d_a}{2} \ln(2\pi \sigma^2) + \frac{1}{2\sigma^2} \sum_{t=1}^T \left\| a_t - \hat{a}_\theta(\mathbf{h}_t) \right\|_2^2$$
Since $T$, $d_a$, and $\sigma^2$ are strictly positive constants with respect to $\theta$, minimizing $\mathcal{L}_{\text{DT}}(\theta)$ is strictly invariant under positive affine transformations:
$$\arg\min_\theta \mathcal{L}_{\text{DT}}(\theta) = \arg\min_\theta \frac{1}{T} \sum_{t=1}^T \left\| a_t - \hat{a}_\theta\left( \hat{R}_{1:t}, s_{1:t}, a_{1:t-1} \right) \right\|_2^2 \triangleq \mathcal{L}_{\text{MSE}}(\theta) \quad \blacksquare$$

---

#### Derivation 11.26.2: Return-to-Go Conditioning Equivalence to Dynamic Programming Policy Iteration in Deterministic MDPs

```
====================================================================================================
DERIVATION 11.26.2: Return-to-Go Conditioning Equivalence to Dynamic Programming in Deterministic MDPs
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
Consider a finite-horizon deterministic Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, f, r, T)$ with deterministic state transition dynamics $s_{t+1} = f(s_t, a_t)$ and deterministic reward function $r_t = r(s_t, a_t)$.
The optimal value function $V^*(s)$ is defined by the Bellman optimality principle:
$$V^*(s) \triangleq \max_{\pi} \sum_{k=t}^T r(s_k, \pi(s_k)) \quad \text{subject to } s_t = s, \; s_{k+1} = f(s_k, \pi(s_k))$$
Let a Decision Transformer model be trained on offline dataset $\mathcal{D}$ to convergence, parameterizing the conditional policy $\pi_{\text{DT}}(a_t \mid s_t, \hat{R}_t)$. At test time, the user prompts the model with target return $\hat{R}_1 = V^*(s_1)$ and applies the online return-to-go decrement update rule:
$$\hat{R}_{t+1} = \hat{R}_t - r(s_t, a_t)$$
Our mathematical goal is to:
1. Prove by backward induction that if the dataset $\mathcal{D}$ has optimal trajectory coverage, conditioning on $\hat{R}_t = V^*(s_t)$ deterministically forces the Decision Transformer to select an action belonging to the optimal Bellman action set:
   $$a_t \in \mathcal{A}^*(s_t) \triangleq \arg\max_{a \in \mathcal{A}} \big[ r(s_t, a) + V^*(f(s_t, a)) \big] \quad \forall t \in \{1, \dots, T\}$$
2. Prove that the online decrement rule preserves optimality at every step: $\hat{R}_{t+1} = V^*(s_{t+1})$, ensuring that the trajectory rolled out by the Decision Transformer attains the true optimal return $V^*(s_1)$.
3. Identify the exact failure boundary: prove why this equivalence breaks fundamentally in stochastic MDPs (the "gambler's fallacy" / tail conditioning breakdown).

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Deterministic Environment:** Both the state transition operator $s_{t+1} = f(s_t, a_t)$ and reward function $r(s_t, a_t)$ are deterministic functions of state and action.
2. **Finite Episode Horizon:** The horizon $T < \infty$ is finite, and the discount factor is undiscounted: $\gamma = 1$.
3. **Optimal Trajectory Dataset Coverage:** For every state $s$ visited along an optimal trajectory from $s_1$, the offline dataset $\mathcal{D}$ contains at least one trajectory starting at $s$ that achieves the optimal cumulative return $V^*(s)$.
4. **Exact Support Consistency of the Conditional Model:** If an action $a$ is never observed in $\mathcal{D}$ starting from $s$ with return $R$, then $P_{\text{DT}}(a \mid s, R) = 0$. That is, the model does not place probability mass outside the empirical support of $(s, R)$.
5. **Discrete or Compact Action Space:** $\mathcal{A}$ is compact, guaranteeing that the maximum in Bellman optimality is attained.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
In dynamic programming, Bellman's Principle of Optimality states:
> *An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.*

In a deterministic MDP, the total return of any trajectory $\tau = (s_1, a_1, \dots, s_T, a_T)$ is an exact deterministic function of the sequence of actions: $R(\tau) = \sum_{t=1}^T r(s_t, a_t) \le V^*(s_1)$.
Therefore, the level set of trajectories achieving maximum possible return:
$$\mathcal{T}^* \triangleq \{ \tau \mid R(\tau) = V^*(s_1) \}$$
consists **strictly and exclusively of optimal trajectories**. Slicing the trajectory distribution by conditioning on $\hat{R}_1 = V^*(s_1)$ eliminates all suboptimal branches. Because there is no environment randomness, achieving $V^*(s_1)$ mathematically necessitates taking an optimal action $a_1^*$, which leaves a remaining return budget of $V^*(s_1) - r_1 = V^*(s_2)$. Decrementing $\hat{R}$ maintains this exact boundary condition across the entire rollout!

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Bellman Optimality Equation in Deterministic MDPs.*
In a deterministic finite-horizon MDP with undiscounted rewards ($\gamma = 1$), the optimal value function satisfies the dynamic programming backward recurrence:
$$V^*(s_t) = \max_{a \in \mathcal{A}} \left[ r(s_t, a) + V^*(f(s_t, a)) \right], \quad \forall t \in \{1, \dots, T-1\}$$
with terminal boundary condition at timestep $T$:
$$V^*(s_T) = \max_{a \in \mathcal{A}} r(s_T, a)$$
Define the set of optimal actions at state $s_t$:
$$\mathcal{A}^*(s_t) \triangleq \left\{ a \in \mathcal{A} \;\middle|\; r(s_t, a) + V^*(f(s_t, a)) = V^*(s_t) \right\}$$
For any suboptimal action $a' \notin \mathcal{A}^*(s_t)$, by definition of the strict maximum:
$$r(s_t, a') + V^*(f(s_t, a')) < V^*(s_t)$$

*Step 2: Empirical Support of Trajectories Conditioned on Optimal Return.*
Let $\mathcal{D}(s_t, R)$ denote the subset of transitions in training dataset $\mathcal{D}$ where the agent was at state $s_t$ with empirical return-to-go $\hat{R}_t = R$:
$$\mathcal{D}(s_t, R) \triangleq \left\{ (s_t, a_t, r_t, s_{t+1}, \dots) \in \mathcal{D} \;\middle|\; \sum_{k=t}^T r(s_k, a_k) = R \right\}$$
Suppose we condition on $R = V^*(s_t)$.
For any trajectory prefix in $\mathcal{D}(s_t, V^*(s_t))$, its empirical return satisfies:
$$\hat{R}_t = r(s_t, a_t) + \hat{R}_{t+1} = V^*(s_t)$$
By definition of the optimal value function $V^*$, no trajectory from $s_{t+1} = f(s_t, a_t)$ can ever achieve a return exceeding $V^*(s_{t+1})$:
$$\hat{R}_{t+1} \le V^*(s_{t+1}) = V^*(f(s_t, a_t))$$
Substitute this upper bound into the return decomposition:
$$V^*(s_t) = r(s_t, a_t) + \hat{R}_{t+1} \le r(s_t, a_t) + V^*(f(s_t, a_t))$$
On the other hand, by the Bellman optimality equation:
$$r(s_t, a_t) + V^*(f(s_t, a_t)) \le \max_{a \in \mathcal{A}} \left[ r(s_t, a) + V^*(f(s_t, a)) \right] = V^*(s_t)$$
Combining these two inequalities:
$$V^*(s_t) \le r(s_t, a_t) + V^*(f(s_t, a_t)) \le V^*(s_t)$$
Because the lower bound equals the upper bound, the intermediate terms must hold with **strict equality**:
$$r(s_t, a_t) + V^*(f(s_t, a_t)) = V^*(s_t)$$
and
$$\hat{R}_{t+1} = V^*(f(s_t, a_t)) = V^*(s_{t+1})$$

*Step 3: Action Selection and Value Propagation.*
From equality $r(s_t, a_t) + V^*(f(s_t, a_t)) = V^*(s_t)$, it follows immediately that:
$$a_t \in \mathcal{A}^*(s_t)$$
Thus, every transition in $\mathcal{D}$ that exhibits return-to-go $\hat{R}_t = V^*(s_t)$ must execute an optimal action $a_t \in \mathcal{A}^*(s_t)$.
By Assumption 4 (exact model support), the Decision Transformer policy satisfies:
$$P_{\text{DT}}\left(a_t \;\middle|\; s_t, \hat{R}_t = V^*(s_t)\right) > 0 \implies a_t \in \mathcal{A}^*(s_t)$$
Now evaluate the online return-to-go update applied at test time:
$$\hat{R}_{t+1} = \hat{R}_t - r(s_t, a_t)$$
Since $\hat{R}_t = V^*(s_t)$ and $a_t \in \mathcal{A}^*(s_t)$, we have:
$$\hat{R}_{t+1} = V^*(s_t) - r(s_t, a_t) = \left[ r(s_t, a_t) + V^*(s_{t+1}) \right] - r(s_t, a_t) = V^*(s_{t+1})$$
Hence, the target return-to-go passed into timestep $t+1$ is **identically equal to the true optimal value of the next state** $V^*(s_{t+1})$.
By mathematical induction from $t = 1$ to $t = T$:
1. At $t = 1$, user prompts $\hat{R}_1 = V^*(s_1) \implies a_1 \in \mathcal{A}^*(s_1)$.
2. The environment transitions to $s_2 = f(s_1, a_1)$ and emits $r_1 = r(s_1, a_1)$.
3. Return decrements to $\hat{R}_2 = \hat{R}_1 - r_1 = V^*(s_2)$.
4. By induction, for all $t \in \{1, \dots, T\}$, $a_t \in \mathcal{A}^*(s_t)$ and $\hat{R}_t = V^*(s_t)$.
The total cumulative reward realized by the DT rollout is:
$$\sum_{t=1}^T r(s_t, a_t) = \sum_{t=1}^T \left( \hat{R}_t - \hat{R}_{t+1} \right) = \hat{R}_1 - \hat{R}_{T+1} = V^*(s_1) - 0 = V^*(s_1)$$
The Decision Transformer policy achieves the exact optimal Dynamic Programming performance. $\blacksquare$

*Step 4: Breakdown in Stochastic MDPs (The Gambler's Fallacy).*
Now consider a stochastic MDP where transition dynamics are governed by probability distribution $s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a_t)$.
The true optimal value function under Dynamic Programming maximizes **expected** return:
$$V^*(s_t) = \max_{a \in \mathcal{A}} \left[ r(s_t, a) + \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s_t, a)} [V^*(s')] \right]$$
In contrast, the conditional Decision Transformer policy evaluates the posterior:
$$P_{\text{DT}}(a_t \mid s_t, R) = \frac{P(R \mid s_t, a_t) \pi_\beta(a_t \mid s_t)}{P(R \mid s_t)}$$
Suppose at state $s_t$, there are two actions:
- Action $a_{\text{safe}}$: deterministic transition yielding high expected return $\mathbb{E}[R \mid s_t, a_{\text{safe}}] = 90$, with support $\operatorname{supp}(R \mid s_t, a_{\text{safe}}) = \{90\}$.
- Action $a_{\text{gamble}}$: stochastic transition yielding low expected return $\mathbb{E}[R \mid s_t, a_{\text{gamble}}] = 10$, but with a heavy-tailed lottery distribution:
  $$R = \begin{cases} 100 & \text{with probability } 0.10 \\ 0 & \text{with probability } 0.90 \end{cases}$$
The optimal DP policy chooses:
$$a^* = \arg\max \{ 90, 10 \} = a_{\text{safe}}$$
However, if the user prompts the Decision Transformer with the maximum observed return in the dataset, $R = 100$:
$$P(R = 100 \mid s_t, a_{\text{safe}}) = 0 \implies P_{\text{DT}}(a_{\text{safe}} \mid s_t, R = 100) = 0$$
$$P(R = 100 \mid s_t, a_{\text{gamble}}) = 0.10 > 0 \implies P_{\text{DT}}(a_{\text{gamble}} \mid s_t, R = 100) = 1.0$$
The Decision Transformer selects $a_{\text{gamble}}$ with probability $1.0$.
When deployed online, the agent faces the true environment dynamics: with probability $0.90$ the gamble fails, and the agent receives return $0$. The expected return of the DT agent is:
$$\mathbb{E}_{\tau \sim \pi_{\text{DT}}} [R] = 0.10 \times 100 + 0.90 \times 0 = 10 \ll 90 = V^*(s_t)$$
**Conclusion:** Conditioning on high return-to-go confounds agent competence with environment luck. In stochastic domains, Decision Transformers suffer from the "gambler's fallacy," selecting high-variance reckless actions that happened to get lucky in the offline training dataset.

---

#### Derivation 11.26.3: Causal Masking and Attention Complexity in Multi-Token Decision Transformers

```
====================================================================================================
DERIVATION 11.26.3: Causal Masking and Attention Complexity in Multi-Token Decision Transformers
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal:**
In a Decision Transformer operating over a context window of $K$ timesteps, the trajectory is tokenized into an interleaved multi-modal sequence of $N = 3K$ tokens:
$$\mathbf{U} = \left[ u_1, u_2, \dots, u_{3K} \right]^\top \in \mathbb{R}^{3K \times d}$$
where each timestep $t \in \{1, \dots, K\}$ contributes three distinct tokens corresponding to return-to-go, state, and action:
$$u_{3(t-1)+1} = e_t^R, \quad u_{3(t-1)+2} = e_t^s, \quad u_{3(t-1)+3} = e_t^a$$
Our mathematical goal is to:
1. Formulate the exact lower-triangular causal attention mask matrix $\mathbf{M} \in \{0, -\infty\}^{3K \times 3K}$.
2. Prove that under this causal mask, the output representation of the state token $h_t^s \triangleq \operatorname{Transformer}(\mathbf{U})_{3(t-1)+2}$ satisfies the strict causal non-leakage condition:
   $$\frac{\partial h_t^s}{\partial e_t^a} = \mathbf{0}, \quad \text{and} \quad \frac{\partial h_t^s}{\partial u_j} = \mathbf{0} \quad \forall j \ge 3(t-1) + 3$$
   guaranteeing that the action prediction $\hat{a}_t = W_{\text{act}} h_t^s$ cannot access its own target $a_t$ or any future information.
3. Derive the exact analytical sparsity ratio $\mathcal{S}_N$ of the $3K \times 3K$ attention connectivity matrix.
4. Derive the exact floating-point operation (FLOP) count for a forward and backward pass of a multi-head causal attention layer as an explicit function of context window $K$ and hidden dimension $d$.

**Part 2: Explicit Assumptions & Regularity Conditions:**
1. **Tri-Token Modality Interleaving:** Modality ordering within each timestep $t$ is strictly fixed as $\left(\hat{R}_t, s_t, a_t\right)$, defining the bijection between timestep-modality pair $(t, m)$ and 1-indexed token position:
   $$i(t, m) = 3(t-1) + m, \quad \text{where } t \in \{1, \dots, K\}, \; m \in \{1, 2, 3\}$$
   with $m = 1 \implies \hat{R}_t$, $m = 2 \implies s_t$, $m = 3 \implies a_t$.
2. **Standard Multi-Head Attention Architecture:** The model uses $H$ attention heads with head dimension $d_k = d / H$. Linear projection matrices are $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$.
3. **Causal Additive Attention Mask:** The attention weight matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ is computed via masked softmax:
   $$\mathbf{A} = \operatorname{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right)$$
   where the additive mask entries are:
   $$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
4. **Finite Numerical Precision Rule:** In floating-point arithmetic, $\exp(-\infty) \triangleq 0$, and for any finite logits $z_k$:
   $$\lim_{M_{i, j} \to -\infty} \frac{\exp(z_j + M_{i, j})}{\sum_{l \le i} \exp(z_l) + \sum_{l > i} \exp(z_l + M_{i, l})} = 0$$

**Part 3: Underlying Intuition & Geometric / Physical Interpretation:**
In standard natural language processing (e.g. GPT-3), tokens represent homogeneous text subwords. In reinforcement learning, however, the sequence is composed of three physically distinct modalities with a strict intra-timestep causal hierarchy:
$$\text{Desired Goal } (\hat{R}_t) \longrightarrow \text{Current Observation } (s_t) \longrightarrow \text{Decision / Action } (a_t)$$
- The agent must know what goal to achieve ($\hat{R}_t$) and where it currently is ($s_t$) before selecting action ($a_t$).
- Therefore, token $s_t$ must be allowed to attend to $\hat{R}_t$!
- However, if token $s_t$ could attend to $a_t$, the transformer would simply copy the ground-truth action from the input embedding, driving training loss to zero while achieving zero performance during deployment (where $a_t$ is not yet known!).
- The standard causal lower-triangular mask automatically enforces both **inter-timestep causality** ($t' \le t$) and **intra-timestep causality** ($\hat{R}_t \to s_t \to a_t$) simultaneously!

```
Token Index:    1(R_1)   2(s_1)   3(a_1)   4(R_2)   5(s_2)   6(a_2)
1 (R_1):      [   *        .        .        .        .        .   ]
2 (s_1):      [   *        *        .        .        .        .   ]  <-- s_1 attends to R_1, s_1 (Predicts a_1)
3 (a_1):      [   *        *        *        .        .        .   ]
4 (R_2):      [   *        *        *        *        .        .   ]
5 (s_2):      [   *        *        *        *        *        .   ]  <-- s_2 attends to past + R_2, s_2 (Predicts a_2)
6 (a_2):      [   *        *        *        *        *        *   ]
(* = allowed attention weight > 0; . = masked out to strictly 0.0)
```

**Part 4: End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Formal Definition of Causal Masking Operator.*
Let $N = 3K$ be the total sequence length. The token embedding matrix is $\mathbf{U} \in \mathbb{R}^{N \times d}$.
The query, key, and value matrices are linear projections:
$$\mathbf{Q} = \mathbf{U} W_Q, \quad \mathbf{K} = \mathbf{U} W_K, \quad \mathbf{V} = \mathbf{U} W_V$$
The pre-softmax scaled dot-product logit between query token $i$ and key token $j$ is:
$$S_{i, j} = \frac{q_i^\top k_j}{\sqrt{d_k}} = \frac{u_i^\top W_Q W_K^\top u_j}{\sqrt{d_k}}$$
Adding the causal mask matrix $\mathbf{M} \in \{0, -\infty\}^{N \times N}$:
$$\tilde{S}_{i, j} = S_{i, j} + M_{i, j} = \begin{cases} S_{i, j} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
The attention weight from token $i$ to token $j$ is given by the row-wise softmax:
$$A_{i, j} = \frac{\exp(\tilde{S}_{i, j})}{\sum_{l=1}^N \exp(\tilde{S}_{i, l})} = \frac{\exp(S_{i, j} + M_{i, j})}{\sum_{l=1}^i \exp(S_{i, l}) + \sum_{l=i+1}^N \exp(-\infty)}$$
Since $\exp(-\infty) = 0$:
$$A_{i, j} = \begin{cases} \frac{\exp(S_{i, j})}{\sum_{l=1}^i \exp(S_{i, l})} & \text{if } j \le i \\ 0 & \text{if } j > i \end{cases}$$
The resulting context representation for token $i$ is:
$$Y_i = \sum_{j=1}^N A_{i, j} v_j = \sum_{j=1}^i A_{i, j} v_j$$

*Step 2: Proof of Zero Information Leakage for Action Prediction.*
In the Decision Transformer architecture, the predicted action at timestep $t$ is computed from the output representation corresponding to the state token $s_t$:
$$\hat{a}_t = W_{\text{act}} Y_{i_s(t)}$$
where by Assumption 1, the index of the state token is $i_s(t) = 3(t-1) + 2$.
The context vector $Y_{i_s(t)}$ is:
$$Y_{i_s(t)} = \sum_{j=1}^{3(t-1)+2} A_{i_s(t), j} v_j$$
We test the sensitivity of $Y_{i_s(t)}$ to any token $u_k$ with index $k \ge 3(t-1) + 3$:
1. For token $u_{3(t-1)+3} = e_t^a$ (the action token at timestep $t$):
   Since $k = 3(t-1) + 3 > 3(t-1) + 2 = i_s(t)$, we have $M_{i_s(t), k} = -\infty$.
   Consequently, $A_{i_s(t), k} = 0$, and $v_k$ does not appear in the linear combination for $Y_{i_s(t)}$.
   Therefore:
   $$\frac{\partial Y_{i_s(t)}}{\partial e_t^a} = \mathbf{0} \implies \frac{\partial \hat{a}_t}{\partial e_t^a} = \mathbf{0}$$
2. For any future token $u_k$ belonging to timestep $t' > t$:
   The smallest index belonging to any future timestep is $i(t+1, 1) = 3t + 1$.
   Comparing indices:
   $$3t + 1 = 3(t-1) + 4 > 3(t-1) + 2 = i_s(t)$$
   Thus, for all $t' > t$ and all modalities $m' \in \{1, 2, 3\}$, $i(t', m') > i_s(t)$, which implies:
   $$A_{i_s(t), i(t', m')} = 0 \implies \frac{\partial \hat{a}_t}{\partial u_{i(t', m')}} = \mathbf{0}$$
**Conclusion:** Action prediction $\hat{a}_t$ is mathematically independent of both the ground-truth action $a_t$ and all future states, actions, and returns. Causal integrity is strictly preserved. $\blacksquare$

*Step 3: Derivation of the Exact Analytical Sparsity Ratio.*
The full attention matrix $\mathbf{A}$ has dimension $N \times N$, containing a total of $N^2$ potential connections.
Under the causal mask, row $i$ has non-zero weights only for columns $j \in \{1, \dots, i\}$.
The total number of active (non-zero) attention connections is the sum of integers from $1$ to $N$:
$$N_{\text{active}} = \sum_{i=1}^N i = \frac{N(N+1)}{2}$$
Substitute $N = 3K$ (where $K$ is the context length in timesteps):
$$N_{\text{active}}(K) = \frac{3K(3K+1)}{2} = \frac{9K^2 + 3K}{2}$$
The total number of entries in the attention matrix is:
$$N_{\text{total}}(K) = (3K)^2 = 9K^2$$
The number of masked (zeroed) entries is:
$$N_{\text{masked}}(K) = N_{\text{total}} - N_{\text{active}} = 9K^2 - \frac{9K^2 + 3K}{2} = \frac{9K^2 - 3K}{2}$$
The sparsity ratio $\mathcal{S}(K)$ is defined as the fraction of zero entries:
$$\mathcal{S}(K) \triangleq \frac{N_{\text{masked}}(K)}{N_{\text{total}}(K)} = \frac{\frac{9K^2 - 3K}{2}}{9K^2} = \frac{9K^2 - 3K}{18K^2} = \frac{3K - 1}{6K} = \frac{1}{2} - \frac{1}{6K}$$
In the asymptotic limit of large context length $K \to \infty$:
$$\lim_{K \to \infty} \mathcal{S}(K) = \frac{1}{2} = 50\% \text{ sparsity}$$

*Step 4: Computational FLOP Complexity per Attention Layer.*
Let us calculate the exact number of floating-point operations (FLOPs) required by a single Transformer layer processing $N = 3K$ tokens with hidden dimension $d$.
Standard matrix multiplication of $(M \times P)$ by $(P \times Q)$ requires $2 M P Q$ FLOPs ($M P Q$ multiplications and $M P Q$ additions).

1. **Projection to Q, K, V:**
   Three distinct linear projections from $\mathbb{R}^{N \times d}$ to $\mathbb{R}^{N \times d}$ using weights $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$:
   $$\text{FLOPs}_{QKV} = 3 \times (2 \cdot N \cdot d \cdot d) = 6 N d^2$$
2. **Attention Logit Computation ($\mathbf{Q} \mathbf{K}^\top$):**
   Multiplication of $(N \times d)$ by $(d \times N)$:
   $$\text{FLOPs}_{\text{logits}} = 2 \cdot N \cdot d \cdot N = 2 N^2 d$$
3. **Softmax Normalization:**
   Evaluating $\exp(\cdot)$ and sum normalization per row across $N$ rows: $\approx 3 N^2$ FLOPs (negligible compared to matrix multiplications).
4. **Context Aggregation ($\mathbf{A} \mathbf{V}$):**
   Multiplication of $(N \times N)$ by $(N \times d)$:
   $$\text{FLOPs}_{\text{context}} = 2 \cdot N \cdot N \cdot d = 2 N^2 d$$
5. **Output Projection ($W_O$):**
   Linear projection from $\mathbb{R}^{N \times d}$ to $\mathbb{R}^{N \times d}$ using $W_O \in \mathbb{R}^{d \times d}$:
   $$\text{FLOPs}_{\text{out}} = 2 \cdot N \cdot d \cdot d = 2 N d^2$$
6. **Feedforward Network (MLP):**
   A standard 2-layer MLP with hidden expansion factor $4d$:
   - Layer 1: $(N \times d) \times (d \times 4d) \implies 2 \cdot N \cdot d \cdot 4d = 8 N d^2$ FLOPs.
   - Layer 2: $(N \times 4d) \times (4d \times d) \implies 2 \cdot N \cdot 4d \cdot d = 8 N d^2$ FLOPs.
   $$\text{FLOPs}_{\text{MLP}} = 16 N d^2$$

Summing all dominant operations:
$$\text{FLOPs}_{\text{layer}}(N, d) = \underbrace{(6 N d^2 + 2 N d^2 + 16 N d^2)}_{\text{Linear in } N} + \underbrace{(2 N^2 d + 2 N^2 d)}_{\text{Quadratic in } N} = 24 N d^2 + 4 N^2 d$$
Substituting sequence length $N = 3K$:
$$\text{FLOPs}_{\text{layer}}(K, d) = 24(3K) d^2 + 4(3K)^2 d = 72 K d^2 + 36 K^2 d \quad \blacksquare$$

---

## 3. Geometric & Physical Interpretation

### 3.1 Partitioning the Trajectory Manifold by Return
Why does conditioning on Return-to-Go work better than naive Behavior Cloning?

```
Trajectory Space
 ^
 |    [ Low Return Manifold: R = 10 ]  ---> Sub-optimal random wander
 |
 |    [ Medium Return: R = 50 ]        ---> Mediocre heuristics
 |
 |    [ High Return Manifold: R = 100] ---> Direct, optimal shortest path!
 |___________________________________________________> Action Space
```
- **Behavior Cloning (BC):** Averages across all trajectories, learning a blurry, mediocre policy that collides with walls.
- **Decision Transformer (DT):** Learns the entire joint distribution $p(\tau)$. Conditioning on $\hat{R}_{\text{target}} = 100$ slices the manifold, conditioning the model *strictly* on the subspace of optimal expert transitions!

---

## 4. Real-World Analogy: GPS Navigation

Consider using a GPS navigation app on your phone:
- **Standard RL (Q-Learning):** At an intersection, the GPS evaluates: *"Turning left gives an expected commute time of 28 minutes; turning right gives 35 minutes."*
- **Decision Transformer:** You tell the GPS: *"I need to reach the airport in exactly 20 minutes ($\hat{R}_{\text{target}}$). I am currently at 5th Avenue ($s_1$). What turns are consistent with that outcome?"* The GPS outputs the high-speed highway route!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete forward pass of a **1-Layer, 1-Head Decision Transformer** by hand on a 2-token context $[\hat{R}_1, s_1]$ predicting continuous action $\hat{a}_1$.

---

### 5.1 System & Model Parameters
- **Hidden Embedding Dimension:** $d = 2$
- **Input Tokens ($T = 1$):**
  - Return-to-Go: $\hat{R}_1 = [10.0]$ (scalar)
  - State: $s_1 = [5.0]$ (scalar)
- **Linear Projection Weights:**
  $$W_R = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}, \quad W_s = \begin{bmatrix} 0.4 \\ -0.2 \end{bmatrix}$$
  *(Zero positional embedding for simplicity: $e_t^{\text{pos}} = [0, 0]^T$)*
- **Self-Attention Weights ($d = 2, d_k = 2$):**
  $$W_Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad W_K = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad W_V = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}$$
- **Linear Action Head:**
  $$W_{\text{act}} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\hat{R}_1, s_1$ | `rtg_1, state_1` | Raw input return-to-go and environment state scalars |
| $e_1^R, e_1^s$ | `embed_R, embed_s` | Projected token embedding vectors in $\mathbb{R}^2$ |
| $X$ | `input_seq` | Matrix of input tokens $\begin{bmatrix} (e_1^R)^T \\ (e_1^s)^T \end{bmatrix} \in \mathbb{R}^{2 \times 2}$ |
| $Q, K, V$ | `q_mat, k_mat, v_mat` | Query, Key, and Value matrices ($X W_Q, X W_K, X W_V$) |
| $z_{2, 1}, z_{2, 2}$ | `attn_logits` | Scaled dot-product attention logits for token 2 ($s_1$) |
| $w_{2, 1}, w_{2, 2}$ | `attn_weights` | Softmax normalized attention probabilities for token 2 |
| $Y_2$ | `attn_out_2` | Attention context output for state token $s_1$ |
| $\hat{a}_1$ | `pred_action` | Final continuous action prediction: $W_{\text{act}} Y_2$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Token Embeddings
$$e_1^R = W_R \hat{R}_1 = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} \times 10.0 = \begin{bmatrix} \mathbf{1.0} \\ \mathbf{2.0} \end{bmatrix}$$
$$e_1^s = W_s s_1 = \begin{bmatrix} 0.4 \\ -0.2 \end{bmatrix} \times 5.0 = \begin{bmatrix} \mathbf{2.0} \\ \mathbf{-1.0} \end{bmatrix}$$

Sequence matrix $X \in \mathbb{R}^{2 \times 2}$:
$$X = \begin{bmatrix} (e_1^R)^T \\ (e_1^s)^T \end{bmatrix} = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & -1.0 \end{bmatrix}$$

#### Step 2: Linear Projections for $Q, K, V$
Since $W_Q = I$ and $W_K = I$:
$$Q = X = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & -1.0 \end{bmatrix}, \quad K = X = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & -1.0 \end{bmatrix}$$
Since $W_V = 0.5 \cdot I$:
$$V = 0.5 \cdot X = \begin{bmatrix} 0.5 & 1.0 \\ 1.0 & -0.5 \end{bmatrix}$$

For predicting action $a_1$, we look at the row corresponding to state token $s_1$ (row index 2):
- Query vector: $q_2 = \begin{bmatrix} 2.0 & -1.0 \end{bmatrix}$
- Key 1 (Return token): $k_1 = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}$
- Key 2 (State token): $k_2 = \begin{bmatrix} 2.0 & -1.0 \end{bmatrix}$
- Value 1: $v_1 = \begin{bmatrix} 0.5 & 1.0 \end{bmatrix}$
- Value 2: $v_2 = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix}$

#### Step 3: Dot-Product Attention Logits for State Token $s_1$
1. **Dot product with Return Token $e_1^R$:**
   $$q_2 \cdot k_1 = (2.0 \times 1.0) + (-1.0 \times 2.0) = 2.0 - 2.0 = \mathbf{0.0000}$$
2. **Dot product with State Token $e_1^s$:**
   $$q_2 \cdot k_2 = (2.0 \times 2.0) + (-1.0 \times -1.0) = 4.0 + 1.0 = \mathbf{5.0000}$$

Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414214$:
$$z_{2, 1} = \frac{0.0000}{1.414214} = \mathbf{0.0000}$$
$$z_{2, 2} = \frac{5.0000}{1.414214} \approx \mathbf{3.535534}$$

#### Step 4: Softmax Attention Weights
$$e^{z_{2, 1}} = e^{0.0} = \mathbf{1.0000}$$
$$e^{z_{2, 2}} = e^{3.535534} \approx \mathbf{34.313410}$$
$$\sum = 1.0000 + 34.313410 = \mathbf{35.313410}$$

$$w_{2, 1} = \frac{1.0000}{35.313410} \approx \mathbf{0.028318}$$
$$w_{2, 2} = \frac{34.313410}{35.313410} \approx \mathbf{0.971682}$$

#### Step 5: Attention Output Context $Y_2$
$$Y_2 = w_{2, 1} v_1 + w_{2, 2} v_2$$
$$= 0.028318 \begin{bmatrix} 0.5 & 1.0 \end{bmatrix} + 0.971682 \begin{bmatrix} 1.0 & -0.5 \end{bmatrix}$$
$$= \begin{bmatrix} 0.014159 & 0.028318 \end{bmatrix} + \begin{bmatrix} 0.971682 & -0.485841 \end{bmatrix}$$
$$= \begin{bmatrix} \mathbf{0.985841} & \mathbf{-0.457523} \end{bmatrix}$$

#### Step 6: Action Prediction via Linear Head $W_{\text{act}}$
$$\hat{a}_1 = W_{\text{act}} Y_2^T = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 0.985841 \\ -0.457523 \end{bmatrix}$$
$$= 1.0 \times 0.985841 + 2.0 \times (-0.457523) = 0.985841 - 0.915046 = \mathbf{0.070795}$$

---

### 5.4 Summary Visual Grid: Decision Transformer Forward Ledger

| Step | Operation | Components | Arithmetic Formulation | Output Value |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $e_1^R$ Embedding | $W_R \cdot 10.0$ | $[0.1, 0.2]^T \times 10$ | $[1.0, 2.0]^T$ |
| **2** | $e_1^s$ Embedding | $W_s \cdot 5.0$ | $[0.4, -0.2]^T \times 5$ | $[2.0, -1.0]^T$ |
| **3** | Attention Logits | $q_2 \cdot K^T / \sqrt{2}$ | $[2.0, -1.0] \cdot K^T / 1.4142$ | $[0.0000, 3.5355]$ |
| **4** | Softmax Weights | $\operatorname{softmax}(\mathbf{z})$ | $\exp(\mathbf{z}) / \sum \exp(\mathbf{z})$ | $[0.0283, 0.9717]$ |
| **5** | Context Vector | $\sum w_i v_i$ | $0.0283 v_1 + 0.9717 v_2$ | $[0.9858, -0.4575]$ |
| **6** | Action Head | $W_{\text{act}} Y_2$ | $1.0(0.9858) + 2.0(-0.4575)$ | $\mathbf{\hat{a}_1 = 0.0708}$ |

---

## 6. Solved Illustrations

### Illustration 1: Prompting Beyond the Data Horizon (Return Extrapolation)

**Problem:**
Suppose an offline training dataset $\mathcal{D}$ contains trajectories whose returns-to-go lie strictly in the range $\hat{R} \in [10.0000, 100.0000]$, with dataset maximum $R_{\max} = 100.0000$.
At test time, an operator prompts the Decision Transformer with an extreme target return:
$$\hat{R}_{\text{target}} = 500.0000$$
Using the mathematical formulation of token embeddings and attention layers, analyze what happens algebraically and geometrically:
1. What is the effect on the return embedding vector $e^R = W_R \hat{R}$?
2. How does the magnitude of $e^R$ distort the dot-product attention logits $S_{s, R} = \frac{q_s^\top k_R}{\sqrt{d_k}}$?
3. How does the softmax attention distribution respond to this out-of-distribution input?
4. Contrast this outcome with standard Bellman value iteration.

**Solution:**

*Step 1: Distortion of the return token embedding.*
In the Decision Transformer embedding layer:
$$e^R = W_R \hat{R} + e_t^{\text{pos}}$$
For any training trajectory, the maximum Euclidean norm of the return embedding (assuming centered positional encoding) satisfies:
$$\|W_R \hat{R}\|_2 \le \|W_R\|_2 \times 100.0000$$
When prompted with $\hat{R}_{\text{target}} = 500.0000$:
$$\|W_R \hat{R}_{\text{target}}\|_2 = 500.0000 \times \|W_R\|_2 = 5 \times \max_{\tau \in \mathcal{D}} \|W_R \hat{R}\|_2$$
The input vector is pushed $5\times$ outside the compact convex hull of representations seen during gradient training.

*Step 2: Logit amplification in scaled dot-product attention.*
The query vector emitted by current state token $s_t$ is $q_s = W_Q e_t^s$.
The key vector corresponding to the return token is $k_R = W_K e_t^R \approx 500.0000 \cdot W_K W_R$.
The scaled attention logit between state $s_t$ and return $\hat{R}$ scales linearly with $\hat{R}$:
$$S_{s, R} = \frac{q_s^\top k_R}{\sqrt{d_k}} = 500.0000 \times \left( \frac{q_s^\top W_K W_R}{\sqrt{d_k}} \right)$$
Assuming the inner product $q_s^\top W_K W_R > 0$ (learned to reward high targets), $S_{s, R}$ becomes massive relative to all other competitive keys (such as past states $s_{t'}$ or actions $a_{t'}$):
$$S_{s, R} \gg S_{s, s} \quad \text{and} \quad S_{s, R} \gg S_{s, a}$$

*Step 3: Degenerate softmax saturation.*
The attention probability assigned to the return token is:
$$w_R = \frac{\exp(S_{s, R})}{\exp(S_{s, R}) + \sum_{j \neq R} \exp(S_{s, j})} = \frac{1}{1 + \sum_{j \neq R} \exp(S_{s, j} - S_{s, R})} \xrightarrow{S_{s, R} \to \infty} \mathbf{1.0000}$$
Consequently, all other historical context weights collapse:
$$w_j \to \mathbf{0.0000} \quad \forall j \neq R$$
The state token context vector collapses to:
$$Y_s = \sum_j w_j v_j \approx \mathbf{v}_R$$
This creates **attention starvation**: the state token cannot attend to its own current physical coordinates $s_t$, blinding the action prediction head from recognizing obstacle locations or boundary walls. In the subsequent MLP layers, $5\times$ enlarged activations push activations into saturated zero-gradient regimes.

*Step 4: Empirical behavior vs. Bellman dynamic programming.*
- In standard RL, value functions $Q(s, a)$ approximate the Bellman expectation. When an agent enters a state with high $Q$-value, the greedy policy $\arg\max_a Q(s, a)$ remains grounded in local action comparisons.
- In Decision Transformers, prompting with $500.0000$ does **not** magically unlock superhuman speed or $5\times$ higher rewards. Instead, as demonstrated empirically by Chen et al. (2021) across D4RL benchmarks:
  - Performance typically **saturates** at the level corresponding to the best policy seen during training ($R \approx 100.0000$).
  - For extreme extrapolations ($\hat{R} \gg R_{\max}$), performance **degrades catastrophically** toward random walk due to softmax saturation and out-of-distribution feature collapse. $\blacksquare$

---

### Illustration 2: Return-to-Go Trajectory Tokenization and Positional Embedding Construction for a 3-Step Episode

**Problem:**
Consider an offline episode of length $T = 3$ with observed scalar environment rewards:
$$r_1 = 2.0000, \quad r_2 = 5.0000, \quad r_3 = -1.0000$$
Environment states $s_t \in \mathbb{R}^2$ and actions $a_t \in \mathbb{R}^1$ are given by:
$$s_1 = \begin{bmatrix} 1.0000 \\ 0.0000 \end{bmatrix}, \quad s_2 = \begin{bmatrix} 0.5000 \\ 1.5000 \end{bmatrix}, \quad s_3 = \begin{bmatrix} -1.0000 \\ 2.0000 \end{bmatrix}$$
$$a_1 = [0.5000], \quad a_2 = [-0.8000], \quad a_3 = [1.2000]$$
The model maps all modalities into a hidden embedding dimension $d = 2$ using linear projections:
$$W_R = \begin{bmatrix} 0.5000 \\ -0.2000 \end{bmatrix} \in \mathbb{R}^{2 \times 1}, \quad W_s = \begin{bmatrix} 0.4000 & 0.1000 \\ -0.2000 & 0.3000 \end{bmatrix} \in \mathbb{R}^{2 \times 2}, \quad W_a = \begin{bmatrix} 0.6000 \\ 0.2000 \end{bmatrix} \in \mathbb{R}^{2 \times 1}$$
Timestep positional embeddings $e_t^{\text{pos}} \in \mathbb{R}^2$ for timesteps $t \in \{1, 2, 3\}$ are:
$$e_1^{\text{pos}} = \begin{bmatrix} 0.1000 \\ 0.1000 \end{bmatrix}, \quad e_2^{\text{pos}} = \begin{bmatrix} 0.2000 \\ 0.2000 \end{bmatrix}, \quad e_3^{\text{pos}} = \begin{bmatrix} 0.3000 \\ 0.3000 \end{bmatrix}$$
Perform the complete numerical walkthrough:
1. Compute the backward return-to-go sequence $\hat{R}_3, \hat{R}_2, \hat{R}_1$.
2. Compute the token embedding vectors $e_t^R, e_t^s, e_t^a$ for each timestep $t \in \{1, 2, 3\}$.
3. Assemble the complete $9 \times 2$ sequence matrix $\mathbf{U}$ fed into the transformer.

**Solution:**

*Step 1: Compute empirical Returns-to-Go backwards.*
Using the backward recurrence $\hat{R}_t = r_t + \hat{R}_{t+1}$ with $\hat{R}_{T+1} = 0$:
$$\hat{R}_3 = r_3 = \mathbf{-1.0000}$$
$$\hat{R}_2 = r_2 + \hat{R}_3 = 5.0000 + (-1.0000) = \mathbf{4.0000}$$
$$\hat{R}_1 = r_1 + \hat{R}_2 = 2.0000 + 4.0000 = \mathbf{6.0000}$$

*Step 2: Timestep $t = 1$ Token Embeddings.*
1. **Return Token $e_1^R$:**
   $$W_R \hat{R}_1 = \begin{bmatrix} 0.5000 \\ -0.2000 \end{bmatrix} \times 6.0000 = \begin{bmatrix} 3.0000 \\ -1.2000 \end{bmatrix}$$
   $$e_1^R = W_R \hat{R}_1 + e_1^{\text{pos}} = \begin{bmatrix} 3.0000 + 0.1000 \\ -1.2000 + 0.1000 \end{bmatrix} = \begin{bmatrix} \mathbf{3.1000} \\ \mathbf{-1.1000} \end{bmatrix}$$
2. **State Token $e_1^s$:**
   $$W_s s_1 = \begin{bmatrix} 0.4000(1.0000) + 0.1000(0.0000) \\ -0.2000(1.0000) + 0.3000(0.0000) \end{bmatrix} = \begin{bmatrix} 0.4000 \\ -0.2000 \end{bmatrix}$$
   $$e_1^s = W_s s_1 + e_1^{\text{pos}} = \begin{bmatrix} 0.4000 + 0.1000 \\ -0.2000 + 0.1000 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5000} \\ \mathbf{-0.1000} \end{bmatrix}$$
3. **Action Token $e_1^a$:**
   $$W_a a_1 = \begin{bmatrix} 0.6000(0.5000) \\ 0.2000(0.5000) \end{bmatrix} = \begin{bmatrix} 0.3000 \\ 0.1000 \end{bmatrix}$$
   $$e_1^a = W_a a_1 + e_1^{\text{pos}} = \begin{bmatrix} 0.3000 + 0.1000 \\ 0.1000 + 0.1000 \end{bmatrix} = \begin{bmatrix} \mathbf{0.4000} \\ \mathbf{0.2000} \end{bmatrix}$$

*Step 3: Timestep $t = 2$ Token Embeddings.*
1. **Return Token $e_2^R$:**
   $$W_R \hat{R}_2 = \begin{bmatrix} 0.5000 \\ -0.2000 \end{bmatrix} \times 4.0000 = \begin{bmatrix} 2.0000 \\ -0.8000 \end{bmatrix}$$
   $$e_2^R = W_R \hat{R}_2 + e_2^{\text{pos}} = \begin{bmatrix} 2.0000 + 0.2000 \\ -0.8000 + 0.2000 \end{bmatrix} = \begin{bmatrix} \mathbf{2.2000} \\ \mathbf{-0.6000} \end{bmatrix}$$
2. **State Token $e_2^s$:**
   $$W_s s_2 = \begin{bmatrix} 0.4000(0.5000) + 0.1000(1.5000) \\ -0.2000(0.5000) + 0.3000(1.5000) \end{bmatrix} = \begin{bmatrix} 0.2000 + 0.1500 \\ -0.1000 + 0.4500 \end{bmatrix} = \begin{bmatrix} 0.3500 \\ 0.3500 \end{bmatrix}$$
   $$e_2^s = W_s s_2 + e_2^{\text{pos}} = \begin{bmatrix} 0.3500 + 0.2000 \\ 0.3500 + 0.2000 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5500} \\ \mathbf{0.5500} \end{bmatrix}$$
3. **Action Token $e_2^a$:**
   $$W_a a_2 = \begin{bmatrix} 0.6000(-0.8000) \\ 0.2000(-0.8000) \end{bmatrix} = \begin{bmatrix} -0.4800 \\ -0.1600 \end{bmatrix}$$
   $$e_2^a = W_a a_2 + e_2^{\text{pos}} = \begin{bmatrix} -0.4800 + 0.2000 \\ -0.1600 + 0.2000 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.2800} \\ \mathbf{0.0400} \end{bmatrix}$$

*Step 4: Timestep $t = 3$ Token Embeddings.*
1. **Return Token $e_3^R$:**
   $$W_R \hat{R}_3 = \begin{bmatrix} 0.5000 \\ -0.2000 \end{bmatrix} \times (-1.0000) = \begin{bmatrix} -0.5000 \\ 0.2000 \end{bmatrix}$$
   $$e_3^R = W_R \hat{R}_3 + e_3^{\text{pos}} = \begin{bmatrix} -0.5000 + 0.3000 \\ 0.2000 + 0.3000 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.2000} \\ \mathbf{0.5000} \end{bmatrix}$$
2. **State Token $e_3^s$:**
   $$W_s s_3 = \begin{bmatrix} 0.4000(-1.0000) + 0.1000(2.0000) \\ -0.2000(-1.0000) + 0.3000(2.0000) \end{bmatrix} = \begin{bmatrix} -0.4000 + 0.2000 \\ 0.2000 + 0.6000 \end{bmatrix} = \begin{bmatrix} -0.2000 \\ 0.8000 \end{bmatrix}$$
   $$e_3^s = W_s s_3 + e_3^{\text{pos}} = \begin{bmatrix} -0.2000 + 0.3000 \\ 0.8000 + 0.3000 \end{bmatrix} = \begin{bmatrix} \mathbf{0.1000} \\ \mathbf{1.1000} \end{bmatrix}$$
3. **Action Token $e_3^a$:**
   $$W_a a_3 = \begin{bmatrix} 0.6000(1.2000) \\ 0.2000(1.2000) \end{bmatrix} = \begin{bmatrix} 0.7200 \\ 0.2400 \end{bmatrix}$$
   $$e_3^a = W_a a_3 + e_3^{\text{pos}} = \begin{bmatrix} 0.7200 + 0.3000 \\ 0.2400 + 0.3000 \end{bmatrix} = \begin{bmatrix} \mathbf{1.0200} \\ \mathbf{0.5400} \end{bmatrix}$$

*Step 5: Sequence Matrix Summary Ledger.*
The interleaved input sequence $\mathbf{U} \in \mathbb{R}^{9 \times 2}$ is:

| Token Index $i$ | Timestep $t$ | Modality $m$ | Physical Input | Positional $e_t^{\text{pos}}$ | Final Embedding $u_i^\top$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $t = 1$ | $\hat{R}_1$ | $\hat{R}_1 = 6.0000$ | $[0.1000, 0.1000]^\top$ | $[\mathbf{3.1000}, \mathbf{-1.1000}]$ |
| **2** | $t = 1$ | $s_1$ | $s_1 = [1.0, 0.0]^\top$ | $[0.1000, 0.1000]^\top$ | $[\mathbf{0.5000}, \mathbf{-0.1000}]$ |
| **3** | $t = 1$ | $a_1$ | $a_1 = [0.5000]$ | $[0.1000, 0.1000]^\top$ | $[\mathbf{0.4000}, \mathbf{0.2000}]$ |
| **4** | $t = 2$ | $\hat{R}_2$ | $\hat{R}_2 = 4.0000$ | $[0.2000, 0.2000]^\top$ | $[\mathbf{2.2000}, \mathbf{-0.6000}]$ |
| **5** | $t = 2$ | $s_2$ | $s_2 = [0.5, 1.5]^\top$ | $[0.2000, 0.2000]^\top$ | $[\mathbf{0.5500}, \mathbf{0.5500}]$ |
| **6** | $t = 2$ | $a_2$ | $a_2 = [-0.8000]$ | $[0.2000, 0.2000]^\top$ | $[\mathbf{-0.2800}, \mathbf{0.0400}]$ |
| **7** | $t = 3$ | $\hat{R}_3$ | $\hat{R}_3 = -1.0000$ | $[0.3000, 0.3000]^\top$ | $[\mathbf{-0.2000}, \mathbf{0.5000}]$ |
| **8** | $t = 3$ | $s_3$ | $s_3 = [-1.0, 2.0]^\top$ | $[0.3000, 0.3000]^\top$ | $[\mathbf{0.1000}, \mathbf{1.1000}]$ |
| **9** | $t = 3$ | $a_3$ | $a_3 = [1.2000]$ | $[0.3000, 0.3000]^\top$ | $[\mathbf{1.0200}, \mathbf{0.5400}]$ |

All 9 tokens are projected, position-encoded, and formatted in $O(T)$ without temporal overlap. $\blacksquare$

---

### Illustration 3: Causal Self-Attention Matrix Forward Pass with Tri-Token Ordering

**Problem:**
Consider a 4-token prefix representing $(\hat{R}_1, s_1, a_1, \hat{R}_2)$ with hidden dimension $d = 2$:
$$u_1 = e_1^R = \begin{bmatrix} 1.0000 \\ 0.0000 \end{bmatrix}, \quad u_2 = e_1^s = \begin{bmatrix} 0.0000 \\ 2.0000 \end{bmatrix}, \quad u_3 = e_1^a = \begin{bmatrix} 1.0000 \\ 1.0000 \end{bmatrix}, \quad u_4 = e_2^R = \begin{bmatrix} 2.0000 \\ -1.0000 \end{bmatrix}$$
The attention parameters are:
$$W_Q = \mathbf{I}_2, \quad W_K = \mathbf{I}_2, \quad W_V = \mathbf{I}_2, \quad d_k = 2, \quad \sqrt{d_k} = \sqrt{2} \approx 1.414214$$
The causal mask $M$ enforces $M_{i, j} = 0$ for $j \le i$ and $M_{i, j} = -\infty$ for $j > i$.
1. Compute the raw query-key dot-product matrix $S_{\text{raw}} = \mathbf{Q} \mathbf{K}^\top$.
2. Compute the scaled masked logits $\tilde{S} = S_{\text{raw}} / \sqrt{2} + \mathbf{M}$.
3. Compute the attention probability matrix $\mathbf{A} \in \mathbb{R}^{4 \times 4}$ via row-wise softmax.
4. Compute the output context matrix $\mathbf{Y} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{4 \times 2}$.
5. Verify that token 2 ($s_1$) receives exactly zero attention from current action $a_1$ and future return $\hat{R}_2$.

**Solution:**

*Step 1: Compute raw dot products $S_{\text{raw}} = \mathbf{U} \mathbf{U}^\top$.*
$$\mathbf{U} = \begin{bmatrix} 1.0000 & 0.0000 \\ 0.0000 & 2.0000 \\ 1.0000 & 1.0000 \\ 2.0000 & -1.0000 \end{bmatrix}$$
Evaluating inner products $u_i^\top u_j$:
- Row 1 ($u_1 = [1, 0]^\top$):
  $$u_1 \cdot u_1 = 1.0, \quad u_1 \cdot u_2 = 0.0, \quad u_1 \cdot u_3 = 1.0, \quad u_1 \cdot u_4 = 2.0$$
- Row 2 ($u_2 = [0, 2]^\top$):
  $$u_2 \cdot u_1 = 0.0, \quad u_2 \cdot u_2 = 4.0, \quad u_2 \cdot u_3 = 2.0, \quad u_2 \cdot u_4 = -2.0$$
- Row 3 ($u_3 = [1, 1]^\top$):
  $$u_3 \cdot u_1 = 1.0, \quad u_3 \cdot u_2 = 2.0, \quad u_3 \cdot u_3 = 2.0, \quad u_3 \cdot u_4 = 1.0$$
- Row 4 ($u_4 = [2, -1]^\top$):
  $$u_4 \cdot u_1 = 2.0, \quad u_4 \cdot u_2 = -2.0, \quad u_4 \cdot u_3 = 1.0, \quad u_4 \cdot u_4 = 5.0$$
Thus:
$$S_{\text{raw}} = \begin{bmatrix} 1.0000 & 0.0000 & 1.0000 & 2.0000 \\ 0.0000 & 4.0000 & 2.0000 & -2.0000 \\ 1.0000 & 2.0000 & 2.0000 & 1.0000 \\ 2.0000 & -2.0000 & 1.0000 & 5.0000 \end{bmatrix}$$

*Step 2: Scale by $\sqrt{2} \approx 1.414214$ and apply causal mask.*
Scaling each entry by $1.414214$:
$$S = \begin{bmatrix} 0.707107 & 0.000000 & 0.707107 & 1.414214 \\ 0.000000 & 2.828427 & 1.414214 & -1.414214 \\ 0.707107 & 1.414214 & 1.414214 & 0.707107 \\ 1.414214 & -1.414214 & 0.707107 & 3.535534 \end{bmatrix}$$
Applying mask $M_{i, j} = -\infty$ for $j > i$:
$$\tilde{S} = \begin{bmatrix} 0.707107 & -\infty & -\infty & -\infty \\ 0.000000 & 2.828427 & -\infty & -\infty \\ 0.707107 & 1.414214 & 1.414214 & -\infty \\ 1.414214 & -1.414214 & 0.707107 & 3.535534 \end{bmatrix}$$

*Step 3: Row-by-Row Softmax Normalization.*
- **Row 1 ($\hat{R}_1$):**
  Only position 1 is valid:
  $$A_{1, \cdot} = \begin{bmatrix} \mathbf{1.000000} & \mathbf{0.000000} & \mathbf{0.000000} & \mathbf{0.000000} \end{bmatrix}$$
- **Row 2 ($s_1$):**
  Valid positions are $j \in \{1, 2\}$:
  $$e^{0.000000} = 1.000000, \quad e^{2.828427} \approx 16.918829$$
  $$\sum = 1.000000 + 16.918829 = 17.918829$$
  $$A_{2, 1} = \frac{1.000000}{17.918829} \approx \mathbf{0.055807}, \quad A_{2, 2} = \frac{16.918829}{17.918829} \approx \mathbf{0.944193}, \quad A_{2, 3} = \mathbf{0.000000}, \quad A_{2, 4} = \mathbf{0.000000}$$
- **Row 3 ($a_1$):**
  Valid positions are $j \in \{1, 2, 3\}$:
  $$e^{0.707107} \approx 2.028115, \quad e^{1.414214} \approx 4.113250, \quad e^{1.414214} \approx 4.113250$$
  $$\sum = 2.028115 + 4.113250 + 4.113250 = 10.254615$$
  $$A_{3, 1} = \frac{2.028115}{10.254615} \approx \mathbf{0.197776}, \quad A_{3, 2} = \frac{4.113250}{10.254615} \approx \mathbf{0.401112}, \quad A_{3, 3} = \frac{4.113250}{10.254615} \approx \mathbf{0.401112}, \quad A_{3, 4} = \mathbf{0.000000}$$
- **Row 4 ($\hat{R}_2$):**
  All positions valid:
  $$e^{1.414214} \approx 4.113250, \quad e^{-1.414214} \approx 0.243117, \quad e^{0.707107} \approx 2.028115, \quad e^{3.535534} \approx 34.313410$$
  $$\sum = 4.113250 + 0.243117 + 2.028115 + 34.313410 = 40.697892$$
  $$A_{4, 1} = \frac{4.113250}{40.697892} \approx \mathbf{0.101068}, \quad A_{4, 2} = \frac{0.243117}{40.697892} \approx \mathbf{0.005974}$$
  $$A_{4, 3} = \frac{2.028115}{40.697892} \approx \mathbf{0.049834}, \quad A_{4, 4} = \frac{34.313410}{40.697892} \approx \mathbf{0.843125}$$

Full Attention Matrix $\mathbf{A}$:
$$\mathbf{A} = \begin{bmatrix} 1.000000 & 0.000000 & 0.000000 & 0.000000 \\ 0.055807 & 0.944193 & 0.000000 & 0.000000 \\ 0.197776 & 0.401112 & 0.401112 & 0.000000 \\ 0.101068 & 0.005974 & 0.049834 & 0.843125 \end{bmatrix}$$

*Step 4: Compute context output matrix $\mathbf{Y} = \mathbf{A} \mathbf{V}$ (since $W_V = I$, $V = U$).*
- $Y_1 = 1.000000 [1, 0] = \begin{bmatrix} \mathbf{1.000000} & \mathbf{0.000000} \end{bmatrix}$
- $Y_2 = 0.055807 [1, 0] + 0.944193 [0, 2] = \begin{bmatrix} \mathbf{0.055807} & \mathbf{1.888386} \end{bmatrix}$
- $Y_3 = 0.197776 [1, 0] + 0.401112 [0, 2] + 0.401112 [1, 1] = \begin{bmatrix} 0.197776 + 0.401112 \\ 0.802224 + 0.401112 \end{bmatrix} = \begin{bmatrix} \mathbf{0.598888} & \mathbf{1.203336} \end{bmatrix}$
- $Y_4 = 0.101068 [1, 0] + 0.005974 [0, 2] + 0.049834 [1, 1] + 0.843125 [2, -1]$
  $$Y_4[0] = 0.101068 + 0.049834 + 1.686250 = \mathbf{1.837151}$$
  $$Y_4[1] = 0.011948 + 0.049834 - 0.843125 = \mathbf{-0.781344}$$
  $$Y_4 = \begin{bmatrix} \mathbf{1.837151} & \mathbf{-0.781344} \end{bmatrix}$$

*Step 5: Verification of non-leakage.*
In row 2 (which represents state token $s_1$), we observe:
$$A_{2, 3} = 0.000000 \quad (a_1 \text{ weight}), \quad A_{2, 4} = 0.000000 \quad (\hat{R}_2 \text{ weight})$$
The context vector $Y_2 = [0.055807, 1.888386]$ depends solely on $\hat{R}_1$ and $s_1$. The action head applied to $Y_2$ predicts $a_1$ with strictly zero leakage. $\blacksquare$

---

### Illustration 4: Decision Transformer Inference Action Sampling with Immediate Reward Feedback and RTG Decrementing

**Problem:**
Walk through a complete 2-step online rollout of a deployed Decision Transformer:
- **Parameters:** $d = 2, d_k = 2, \sqrt{d_k} = \sqrt{2} \approx 1.414214$.
  $$W_R = \begin{bmatrix} 0.1000 \\ 0.1000 \end{bmatrix}, \quad W_s = \begin{bmatrix} 1.0000 & 0.0000 \\ 0.0000 & 1.0000 \end{bmatrix}, \quad W_a = \begin{bmatrix} 1.0000 \\ 0.0000 \end{bmatrix}$$
  $$W_Q = \mathbf{I}_2, \quad W_K = \mathbf{I}_2, \quad W_V = \mathbf{I}_2, \quad W_{\text{act}} = \begin{bmatrix} 1.0000 & 1.0000 \end{bmatrix}$$
  *(Positional embeddings $e_t^{\text{pos}} = [0, 0]^\top$ for simplicity).*
- **Step 1 ($t = 1$):**
  - Operator conditions policy on target return $\hat{R}_1 = 20.0000$.
  - Environment yields initial state $s_1 = [1.0000, 0.0000]^\top$.
  - Compute predicted continuous action $\hat{a}_1$.
- **Environment Interaction ($t = 1$):**
  - Executing $\hat{a}_1$ yields scalar reward $r_1 = 3.0000$ and transitions to state $s_2 = [0.0000, 2.0000]^\top$.
- **Step 2 ($t = 2$):**
  - Decrement return-to-go: $\hat{R}_2 = \hat{R}_1 - r_1$.
  - Assemble the 5-token context sequence $\mathbf{U} = [\hat{R}_1, s_1, \hat{a}_1, \hat{R}_2, s_2]$.
  - Compute attention weights from state query $s_2$ over all 5 tokens and predict action $\hat{a}_2$.

**Solution:**

*Step 1: Compute Token Embeddings and Action $\hat{a}_1$ at $t = 1$.*
1. **Embeddings:**
   $$e_1^R = W_R \hat{R}_1 = \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix} \times 20.0000 = \begin{bmatrix} \mathbf{2.0000} \\ \mathbf{2.0000} \end{bmatrix}$$
   $$e_1^s = W_s s_1 = \mathbf{I}_2 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{1.0000} \\ \mathbf{0.0000} \end{bmatrix}$$
2. **Attention from State Token $s_1$ (row 2):**
   Query vector: $q_2 = e_1^s = [1.0000, 0.0000]^\top$.
   Keys: $k_1 = e_1^R = [2.0000, 2.0000]^\top$, $k_2 = e_1^s = [1.0000, 0.0000]^\top$.
   Dot products:
   $$q_2 \cdot k_1 = (1.0 \times 2.0) + (0.0 \times 2.0) = \mathbf{2.0000}$$
   $$q_2 \cdot k_2 = (1.0 \times 1.0) + (0.0 \times 0.0) = \mathbf{1.0000}$$
   Scaled logits ($/ \sqrt{2} \approx 1.414214$):
   $$z_1 = \frac{2.0000}{1.414214} \approx \mathbf{1.414214}, \quad z_2 = \frac{1.0000}{1.414214} \approx \mathbf{0.707107}$$
   Softmax:
   $$e^{z_1} = e^{1.414214} \approx 4.113250, \quad e^{z_2} = e^{0.707107} \approx 2.028115$$
   $$\sum = 4.113250 + 2.028115 = 6.141365$$
   $$w_{2, 1} = \frac{4.113250}{6.141365} \approx \mathbf{0.669762}, \quad w_{2, 2} = \frac{2.028115}{6.141365} \approx \mathbf{0.330238}$$
3. **Context Vector $Y_2$:**
   $$Y_2 = 0.669762 \begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix} + 0.330238 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 1.339523 + 0.330238 \\ 1.339523 + 0.000000 \end{bmatrix} = \begin{bmatrix} \mathbf{1.669762} \\ \mathbf{1.339523} \end{bmatrix}$$
4. **Action Prediction $\hat{a}_1$:**
   $$\hat{a}_1 = W_{\text{act}} Y_2 = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.669762 \\ 1.339523 \end{bmatrix} = 1.669762 + 1.339523 = \mathbf{3.009285}$$

*Step 2: Environment Interaction & RTG Decrementing.*
The agent executes $\hat{a}_1 = 3.009285$.
The environment returns $r_1 = 3.000000$ and $s_2 = [0.0000, 2.0000]^\top$.
Decremented Return-to-Go for step 2:
$$\hat{R}_2 = \hat{R}_1 - r_1 = 20.000000 - 3.000000 = \mathbf{17.000000}$$

*Step 3: Embed Context Extension at $t = 2$.*
1. Executed Action Token:
   $$e_1^a = W_a \hat{a}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} \times 3.009285 = \begin{bmatrix} \mathbf{3.009285} \\ \mathbf{0.000000} \end{bmatrix}$$
2. Decremented Return Token:
   $$e_2^R = W_R \hat{R}_2 = \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix} \times 17.000000 = \begin{bmatrix} \mathbf{1.700000} \\ \mathbf{1.700000} \end{bmatrix}$$
3. Next State Token:
   $$e_2^s = W_s s_2 = \mathbf{I}_2 \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.000000} \\ \mathbf{2.000000} \end{bmatrix}$$

The complete 5-token context matrix $\mathbf{U} \in \mathbb{R}^{5 \times 2}$ is:
$$\mathbf{U} = \begin{bmatrix} u_1 (\hat{R}_1) \\ u_2 (s_1) \\ u_3 (a_1) \\ u_4 (\hat{R}_2) \\ u_5 (s_2) \end{bmatrix} = \begin{bmatrix} 2.000000 & 2.000000 \\ 1.000000 & 0.000000 \\ 3.009285 & 0.000000 \\ 1.700000 & 1.700000 \\ 0.000000 & 2.000000 \end{bmatrix}$$

*Step 4: Attention Forward Pass for State Token $s_2$ (row 5).*
Query vector: $q_5 = u_5 = [0.000000, 2.000000]^\top$.
Dot products $q_5 \cdot u_j$ for $j \in \{1, 2, 3, 4, 5\}$:
$$q_5 \cdot u_1 = (0.0 \times 2.0) + (2.0 \times 2.0) = \mathbf{4.000000}$$
$$q_5 \cdot u_2 = (0.0 \times 1.0) + (2.0 \times 0.0) = \mathbf{0.000000}$$
$$q_5 \cdot u_3 = (0.0 \times 3.009285) + (2.0 \times 0.0) = \mathbf{0.000000}$$
$$q_5 \cdot u_4 = (0.0 \times 1.7) + (2.0 \times 1.7) = \mathbf{3.400000}$$
$$q_5 \cdot u_5 = (0.0 \times 0.0) + (2.0 \times 2.0) = \mathbf{4.000000}$$

Scale by $\sqrt{2} \approx 1.414214$:
$$z_1 = \frac{4.0}{1.414214} \approx \mathbf{2.828427}$$
$$z_2 = \frac{0.0}{1.414214} = \mathbf{0.000000}$$
$$z_3 = \frac{0.0}{1.414214} = \mathbf{0.000000}$$
$$z_4 = \frac{3.4}{1.414214} \approx \mathbf{2.404163}$$
$$z_5 = \frac{4.0}{1.414214} \approx \mathbf{2.828427}$$

Compute exponentials:
$$e^{z_1} = e^{2.828427} \approx 16.918829$$
$$e^{z_2} = e^{0.0} = 1.000000$$
$$e^{z_3} = e^{0.0} = 1.000000$$
$$e^{z_4} = e^{2.404163} \approx 11.069162$$
$$e^{z_5} = e^{2.828427} \approx 16.918829$$
Sum:
$$\sum_{j=1}^5 e^{z_j} = 16.918829 + 1.000000 + 1.000000 + 11.069162 + 16.918829 = \mathbf{46.906819}$$

Normalized attention weights $w_{5, j}$:
$$w_{5, 1} = \frac{16.918829}{46.906819} \approx \mathbf{0.360690}$$
$$w_{5, 2} = \frac{1.000000}{46.906819} \approx \mathbf{0.021319}$$
$$w_{5, 3} = \frac{1.000000}{46.906819} \approx \mathbf{0.021319}$$
$$w_{5, 4} = \frac{11.069162}{46.906819} \approx \mathbf{0.235982}$$
$$w_{5, 5} = \frac{16.918829}{46.906819} \approx \mathbf{0.360690}$$

*Step 5: Output Context Vector $Y_5$ and Action Prediction $\hat{a}_2$.*
$$Y_5 = \sum_{j=1}^5 w_{5, j} u_j$$
Component 1:
$$Y_5[0] = 0.360690(2.0) + 0.021319(1.0) + 0.021319(3.009285) + 0.235982(1.7) + 0.360690(0.0)$$
$$= 0.721380 + 0.021319 + 0.064155 + 0.401169 + 0.000000 = \mathbf{1.208023}$$
Component 2:
$$Y_5[1] = 0.360690(2.0) + 0.021319(0.0) + 0.021319(0.0) + 0.235982(1.7) + 0.360690(2.0)$$
$$= 0.721380 + 0.000000 + 0.000000 + 0.401169 + 0.721380 = \mathbf{1.843930}$$
Thus:
$$Y_5 = \begin{bmatrix} \mathbf{1.208023} \\ \mathbf{1.843930} \end{bmatrix}$$

Action head projection:
$$\hat{a}_2 = W_{\text{act}} Y_5 = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.208023 \\ 1.843930 \end{bmatrix} = 1.208023 + 1.843930 = \mathbf{3.051953}$$

The online loop successfully incorporated real environment feedback ($r_1 = 3.0$), adapted its budget to $\hat{R}_2 = 17.0$, and computed the next action in closed form without value re-estimation. $\blacksquare$

---

### Illustration 5: Comparison of Decision Transformer vs Standard Q-Learning on Stochastic Environment Failure Modes and Sub-Trajectory Stitching

**Problem:**
A common misconception is that Decision Transformer universally subsumes Temporal-Difference Reinforcement Learning (Q-learning / SAC / CQL).
Evaluate both algorithms on two classical benchmark scenarios:
1. **Scenario A (The Sub-Trajectory Stitching Problem):**
   Can the algorithm combine non-overlapping sub-optimal trajectories to discover a novel optimal path?
2. **Scenario B (The Gambler's Fallacy in Stochastic Environments):**
   Can the algorithm distinguish between high expected reward and lucky heavy-tailed stochastic lottery transitions?

Provide complete numerical payoff matrices, exact Bellman backup calculations, and exact empirical conditional probabilities for both algorithms.

**Solution:**

#### Scenario A: Sub-Trajectory Stitching (AntMaze Setting)
Consider four states $\{S_0, S_1, S_{\text{goal}}, S_{\text{trap}}\}$ and discount factor $\gamma = 1.0$:
- At initial state $S_0$, taking action $a_1$ leads deterministically to $S_1$ with immediate reward $r(S_0, a_1) = 0.0000$.
- At intermediate state $S_1$:
  - Action $a^*$ leads to $S_{\text{goal}}$ with reward $r(S_1, a^*) = 50.0000$.
  - Action $a_{\text{sub}}$ leads to $S_{\text{trap}}$ with reward $r(S_1, a_{\text{sub}}) = 10.0000$.
- **Offline Training Dataset $\mathcal{D}$ contains only two types of episodes:**
  - **Trajectory 1 (Suboptimal full run):** $S_0 \xrightarrow{a_1} S_1 \xrightarrow{a_{\text{sub}}} S_{\text{trap}}$. Total return $R = 0.0 + 10.0 = \mathbf{10.0000}$.
  - **Trajectory 2 (Expert partial run):** $S_1 \xrightarrow{a^*} S_{\text{goal}}$. Total return $R = \mathbf{50.0000}$.
  - *Notice:* The dataset contains **zero** demonstrations that start at $S_0$ and reach $S_{\text{goal}}$.

```
Trajectory 1 (Observed in D, R = 10):
S_0 ------------(a_1, r=0)------------> S_1 ------------(a_sub, r=10)------------> S_trap

Trajectory 2 (Observed in D, R = 50):
                                        S_1 ------------(a*, r=50)----------------> S_goal

Desired Optimal Stitched Path (Unobserved in D, R = 50):
S_0 ------------(a_1, r=0)------------> S_1 ------------(a*, r=50)----------------> S_goal
```

1. **Standard Q-Learning (Dynamic Programming):**
   At state $S_1$, Q-learning evaluates both actions:
   $$Q(S_1, a^*) = 50.0000, \quad Q(S_1, a_{\text{sub}}) = 10.0000 \implies V(S_1) = \max_a Q(S_1, a) = 50.0000$$
   Now perform the 1-step Bellman backup from transition $(S_0, a_1, 0, S_1)$ in Trajectory 1:
   $$Q(S_0, a_1) = r(S_0, a_1) + \gamma V(S_1) = 0.0000 + 1.0 \times 50.0000 = \mathbf{50.0000}$$
   **Result:** Q-learning stitches the first segment of Trajectory 1 with the second segment of Trajectory 2. At test time, starting at $S_0$, it chooses $a_1$, reaches $S_1$, and chooses $a^*$, successfully achieving the optimal return of **$50.0000$**!
2. **Decision Transformer:**
   At test time, the user prompts the Decision Transformer at $S_0$ with target return $\hat{R}_{\text{target}} = 50.0000$.
   In dataset $\mathcal{D}$, the conditional empirical distribution at state $S_0$ is:
   $$\operatorname{supp}\left(\hat{R} \;\middle|\; S_0\right) = \{ 10.0000 \}$$
   The target pair $(S_0, \hat{R} = 50.0000)$ is **completely Out-of-Distribution (OOD)**. Because the Decision Transformer performs conditional sequence imitation rather than dynamic programming value maximization, it cannot propagate the future reward at $S_{\text{goal}}$ across the disconnected trajectory boundaries.
   **Result:** The DT fails to stitch the sub-trajectories, defaulting to the observed return of **$10.0000$** or producing erratic behavior.

---

#### Scenario B: The Gambler's Fallacy in Stochastic MDPs
Consider initial state $S_0$ with two available actions:
- **Action $a_{\text{safe}}$ (Consistent Safe Choice):**
  Deterministically transitions to terminal state, emitting reward $r = 18.0000$. (Return $R = 18.0000$).
- **Action $a_{\text{gamble}}$ (Volatile Lottery):**
  Stochastic transition:
  - With probability $p = 0.10$, transitions to jackpot state emitting reward $r = 100.0000$.
  - With probability $p = 0.90$, transitions to bust state emitting reward $r = 0.0000$.

True expected values:
$$\mathbb{E}[R \mid S_0, a_{\text{safe}}] = \mathbf{18.0000}$$
$$\mathbb{E}[R \mid S_0, a_{\text{gamble}}] = 0.10 \times 100.0000 + 0.90 \times 0.0000 = \mathbf{10.0000}$$
The safe action is strictly superior in expectation by $+8.0000$ points ($80\%$ higher expected utility).

Suppose the offline training dataset $\mathcal{D}$ contains $M = 1,000$ episodes sampled uniformly under behavior policy $\pi_\beta(a_{\text{safe}}) = 0.5, \pi_\beta(a_{\text{gamble}}) = 0.5$:
- $500$ episodes executed $a_{\text{safe}}$: all $500$ realized return $R = 18.0000$.
- $500$ episodes executed $a_{\text{gamble}}$:
  - $50$ episodes ($10\%$) got lucky, realizing return $R = 100.0000$.
  - $450$ episodes ($90\%$) got unlucky, realizing return $R = 0.0000$.

1. **Standard Q-Learning Evaluation:**
   $$Q(S_0, a_{\text{safe}}) = \frac{1}{500} \sum_{i=1}^{500} 18.0000 = \mathbf{18.0000}$$
   $$Q(S_0, a_{\text{gamble}}) = \frac{50 \times 100.0000 + 450 \times 0.0000}{500} = \frac{5000.0}{500} = \mathbf{10.0000}$$
   The optimal greedy policy selects:
   $$\pi^*(S_0) = \arg\max_a Q(S_0, a) = a_{\text{safe}}$$
   **Realized online expected return under Q-learning:** $\mathbf{18.0000}$.
2. **Decision Transformer Evaluation:**
   At test time, the operator seeks expert performance and prompts the DT with the maximum return observed in the dataset:
   $$\hat{R}_{\text{prompt}} = 100.0000$$
   We compute the empirical conditional probability $P(a \mid S_0, \hat{R} = 100.0000)$ in dataset $\mathcal{D}$:
   $$P(a_{\text{safe}} \mid S_0, \hat{R} = 100.0000) = \frac{N(S_0, a_{\text{safe}}, R=100)}{N(S_0, R=100)} = \frac{0}{50} = \mathbf{0.0000}$$
   $$P(a_{\text{gamble}} \mid S_0, \hat{R} = 100.0000) = \frac{N(S_0, a_{\text{gamble}}, R=100)}{N(S_0, R=100)} = \frac{50}{50} = \mathbf{1.0000}$$
   Conditioned on $\hat{R} = 100.0000$, the Decision Transformer executes $a_{\text{gamble}}$ with probability $1.0000$.
   However, the environment is stochastic and does not care about the agent's prompt! When $a_{\text{gamble}}$ is executed online, the true environment probability applies ($90\%$ failure rate):
   $$\mathbb{E}_{\tau \sim \pi_{\text{DT}}}[R] = 0.10 \times 100.0000 + 0.90 \times 0.0000 = \mathbf{10.0000}$$
   **Performance degradation vs. Q-learning:**
   $$\Delta R = 10.0000 - 18.0000 = \mathbf{-8.0000} \quad \left( \frac{-8.0}{18.0} \approx \mathbf{-44.4\% \text{ drop}} \right)$$

---

#### Algorithmic Comparison Summary Ledger

| Metric / Dimension | Standard Q-Learning / Bellman DP | Decision Transformer (DT) |
| :--- | :--- | :--- |
| **Foundational Paradigm** | Temporal Difference Bootstrapping & Dynamic Programming | Conditional Autoregressive Sequence Modeling (GPT) |
| **Training Objective** | Bellman Residual Minimization: $\min \|Q - \mathcal{B} Q\|^2$ | Supervised Regression: $\min \|a_t - \hat{a}_\theta(\tau_{<t}, \hat{R}_t, s_t)\|^2$ |
| **Sub-Trajectory Stitching** | **Provably Optimal:** Chains $V(s')$ across disjoint episodes | **Fails:** Cannot stitch across unobserved return combinations |
| **Stochastic Environment Handling** | **Expectation Maximization:** Evaluates true expected utility $\mathbb{E}[R]$ | **Gambler's Fallacy:** Confounds high return with lucky tail transitions |
| **Training Numerical Stability** | Vulnerable to Deadly Triad (divergence, overestimation) | Highly stable standard transformer training (AdamW, cosine lr) |
| **Deployment Horizon Scaling** | Evaluates 1-step Bellman greedy decisions ($O(1)$) | Attends over full context window ($O(K^2)$ attention complexity) |

Decision Transformers trade off dynamic programming stitching and stochastic risk-neutrality in exchange for scalable, ultra-stable sequence training. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. Robotic Foundation Models: RT-1, RT-2, OpenVLA (Google DeepMind, 2022–2024)
Decision Transformer's sequence modeling paradigm enables large-scale robot learning:
- **RT-1 (Brohan et al., 2022):** A 35M-parameter Transformer trained on 130,000 real-robot episodes across 700+ tasks. Conditions on language task description and image observation sequence $\tau = (o_1, a_1, \dots, o_t)$ to autoregressively predict tokenized end-effector actions. Generalizes to 97% of training tasks and 25% of unseen tasks.
- **RT-2 (Brohan et al., Science 2023):** Extends RT-1 by fine-tuning a 55B-parameter PaLM-E vision-language model on robot trajectories, enabling zero-shot reasoning like "Move the object that is the same color as the apple to the bowl" without task-specific demonstrations.
- **OpenVLA (Kim et al., 2024):** A 7.5B open-weights vision-language-action model achieving RT-2 parity on LIBERO manipulation benchmarks, fully open-sourced for community deployment.

### 2. Multi-Game Decision Transformers: One Policy, All Games (Lee et al., NeurIPS 2022)
The Decision Transformer's return-conditioning generalizes across radically different environments:
- **Architecture:** A single 200M-parameter Transformer trained on offline data from all 41 Atari games simultaneously, using one-hot game embeddings as task identifiers in the context.
- **Zero-Shot Transfer:** By conditioning on high return-to-go values ($\hat{R}_t = $ max possible game score), the policy implicitly extracts expert-level behavior from mixed-quality offline data across all games — without per-game fine-tuning.
- **Human-Norm Score:** Achieves 1.11× human-normalized performance averaged across 41 games in a single model, compared to 1.00× for game-specific DQN agents.

### 3. Gato: The Generalist Agent (Reed et al., 2022)
DeepMind's Gato extended the Decision Transformer paradigm to the broadest multi-task policy yet:
- **Unified Token Space:** Text tokens, image patch tokens, and action tokens (discretized into 1,024 bins per dimension) all share the same vocabulary. A single 1.2B-parameter Transformer processes mixed sequences of text, images, and actions autoregressively.
- **604 Tasks Simultaneously:** Atari games, robotic manipulation, image captioning, dialogue — all learned from a single policy checkpoint through return-conditioning and task-context prompting.
- **Implication:** Gato establishes that the Decision Transformer architecture (offline sequence modeling + return conditioning) is a viable path toward a **truly generalist AI agent**, connecting RL, NLP, and computer vision under one roof.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical reproduction of token embeddings, dot-product attention weights, context vector $Y_2$, and predicted action $\hat{a}_1 = 0.070795$ matching to $< 10^{-6}$.
2. **Complete PyTorch Decision Transformer Architecture:**
   - Modality embedding projections with positional encoding.
   - Causal multi-head self-attention blocks.
   - Autoregressive inference rollout loop with online return-to-go decrementing.
3. **Section 6 Solved Illustrations Verification Suite:**
   - Exact numerical verification of 3-step episode trajectory tokenization and positional embeddings (Illustration 2).
   - Exact causal masking and self-attention forward pass without future leakage (Illustration 3).
   - Multi-step online inference action sampling and RTG budget decrementing (Illustration 4).
   - Payoff calculations for sub-trajectory stitching and the stochastic gambler's fallacy pitfall (Illustration 5).

See implementation in:
[`11_reinforcement_learning/code/26_decision_transformer.py`](./code/26_decision_transformer.py)
