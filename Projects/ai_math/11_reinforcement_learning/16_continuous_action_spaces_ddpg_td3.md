# Module 11.16: Continuous Action Spaces: DDPG & TD3

---

## 1. Intuition & 101 Motivation

In the previous chapters, we focused primarily on **discrete action spaces** (e.g., choosing among 4 joystick directions or 2 CartPole moves). However, the vast majority of physical real-world control systems operate in **continuous action spaces**:
- Autonomous vehicles: Steering angle $\theta \in [-180^\circ, +180^\circ]$, throttle $T \in [0, 1]$, braking $B \in [0, 1]$.
- Robotic Manipulators: Continuous motor torques $\boldsymbol{\tau} \in \mathbb{R}^7$ for each joint.
- Drone Quadrotors: Individual rotor RPMs $\boldsymbol{\omega} \in \mathbb{R}^4$.

### Why Value-Based Methods (DQN) Fail in Continuous Spaces
Recall the standard Q-learning target:
$$Y = R + \gamma \max_{a' \in \mathcal{A}} Q(S', a')$$
In continuous action spaces ($\mathcal{A} \subseteq \mathbb{R}^m$), finding $\max_{a'} Q(S', a')$ requires solving a complex, non-linear numerical optimization problem at *every single time step for every single transition*, rendering training hopelessly slow and prone to local minima.

### The Breakthrough: The Deterministic Policy Gradient (DPG)
David Silver et al. (ICML 2014) showed that we do not need to search over all actions; we can train a **deterministic actor network** $\mu_{\boldsymbol{\theta}}(s) \in \mathbb{R}^m$ to output the optimal continuous action directly! The critic $Q_{\boldsymbol{\phi}}(s, a)$ then provides a directional gradient vector $\nabla_a Q(s, a)$ that steers the actor uphill via the chain rule.

- **DDPG (Deep Deterministic Policy Gradient - Lillicrap et al., 2016):** Extended DPG to deep neural networks using replay buffers and Polyak target networks.
- **TD3 (Twin Delayed Deep Deterministic Policy Gradient - Fujimoto et al., 2018):** Solved DDPG's severe value overestimation and instability by introducing three foundational algorithmic enhancements:
  1. **Clipped Double Q-Learning**
  2. **Delayed Policy Updates**
  3. **Target Policy Smoothing**

```
+-----------------------------------------------------------------------------------------+
|                                    TD3 ARCHITECTURE                                     |
|                                                                                         |
|       State s ------> [ Actor Network mu_theta(s) ] ------> Action a in R^m             |
|                          |                                    |                         |
|                          |           +------------------------+                         |
|                          |           |                                                  |
|                          v           v                                                  |
|                  [ Critic Network Q_phi1(s, a) ] ---> Q1 value                          |
|                  [ Critic Network Q_phi2(s, a) ] ---> Q2 value                          |
|                                     |                                                   |
|                        Target: min(Q1_targ, Q2_targ)                                    |
|                                     |                                                   |
|       Actor Gradient: grad_theta = grad_theta mu(s) * grad_a Q1(s, mu(s))               |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Deterministic Policy Gradient Theorem (Silver et al., 2014)

Let $\mu_{\boldsymbol{\theta}}: \mathcal{S} \to \mathcal{A}$ be a deterministic policy with continuous parameters $\boldsymbol{\theta} \in \mathbb{R}^d$.
Let $J(\boldsymbol{\theta}) \triangleq \mathbb{E}_{s \sim d^\mu} [r(s, \mu_{\boldsymbol{\theta}}(s))]$ be the expected return.

#### Theorem:
Under standard regularity conditions on the MDP, the gradient of the performance objective with respect to the deterministic policy parameters $\boldsymbol{\theta}$ is:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim d^\mu} \left[ \nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(s) \left. \nabla_a Q^\mu(s, a) \right|_{a = \mu_{\boldsymbol{\theta}}(s)} \right]$$

In tensor notation, $\nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(s) \in \mathbb{R}^{d \times m}$ is the Jacobian matrix of the actor, and $\nabla_a Q^\mu(s, a) \in \mathbb{R}^m$ is the gradient vector of the action-value function with respect to actions.
The actor update is a simple application of the multivariable **chain rule**:
$$\frac{\partial J}{\partial \boldsymbol{\theta}} = \frac{\partial Q}{\partial a} \frac{\partial a}{\partial \boldsymbol{\theta}}$$

---

### 2.2 Deep Deterministic Policy Gradient (DDPG)

DDPG maintains an actor $\mu_{\boldsymbol{\theta}}$, a critic $Q_{\boldsymbol{\phi}}$, and corresponding target networks $\mu_{\boldsymbol{\theta}^-}, Q_{\boldsymbol{\phi}^-}$.

#### 1. Critic Update:
Transitions $(s, a, r, s', d)$ are sampled from replay buffer $\mathcal{D}$.
Target value:
$$y = r + (1 - d) \gamma Q_{\boldsymbol{\phi}^-}(s', \mu_{\boldsymbol{\theta}^-}(s'))$$
Critic loss (MSE):
$$\mathcal{L}(\boldsymbol{\phi}) = \frac{1}{|B|} \sum_{i \in B} \left( y_i - Q_{\boldsymbol{\phi}}(s_i, a_i) \right)^2$$

#### 2. Actor Update:
$$\nabla_{\boldsymbol{\theta}} J \approx \frac{1}{|B|} \sum_{i \in B} \left. \nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(s_i) \nabla_a Q_{\boldsymbol{\phi}}(s_i, a) \right|_{a = \mu_{\boldsymbol{\theta}}(s_i)}$$

#### 3. Polyak Averaging (Soft Target Updates):
Instead of periodic hard copies, target parameters track online parameters via exponential moving averages:
$$\boldsymbol{\phi}^- \leftarrow \tau \boldsymbol{\phi} + (1 - \tau) \boldsymbol{\phi}^-$$
$$\boldsymbol{\theta}^- \leftarrow \tau \boldsymbol{\theta} + (1 - \tau) \boldsymbol{\theta}^-$$
where $\tau \ll 1$ (typically $\tau = 0.005$).

---

### 2.3 The Failure Mode of DDPG: Severe Overestimation

Just like discrete Q-learning, the DDPG target $y = r + \gamma Q_{\boldsymbol{\phi}^-}(s', \mu_{\boldsymbol{\theta}^-}(s'))$ suffers from **maximization bias**.
Because the actor is trained to maximize $Q$, it constantly exploits local overestimation errors in the critic. As training proceeds:
1. The critic overestimates values.
2. The actor updates toward actions with falsely inflated Q-values.
3. The target values become even higher, triggering runaway overestimation and policy degradation.

---

### 2.4 Twin Delayed DDPG (TD3 - Fujimoto et al., 2018)

TD3 fixes DDPG through three foundational algorithmic principles:

#### 1. Clipped Double Q-Learning:
TD3 maintains **two independent critic networks**, $Q_{\boldsymbol{\phi}_1}$ and $Q_{\boldsymbol{\phi}_2}$, and computes the target using the minimum between their target networks:

$$y = r + (1 - d) \gamma \min_{j = 1, 2} Q_{\boldsymbol{\phi}_j^-}(s', \tilde{a})$$

Taking the minimum provides an upper bound that prevents positive maximization bias from propagating through the Bellman equation!

#### 2. Target Policy Smoothing (Action Regularization):
In continuous control, deterministic targets $Q(s', \mu(s'))$ are vulnerable to sharp, narrow local peaks in the critic. If the critic has a spurious spike at action $a^*$, the actor collapses into it.
TD3 adds small, clipped Gaussian noise to the target action:

$$\tilde{a} \triangleq \operatorname{clip}\left( \mu_{\boldsymbol{\theta}^-}(s') + \operatorname{clip}(\epsilon, -c, c), \quad a_{\text{low}}, a_{\text{high}} \right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
where $\sigma \approx 0.2$ and noise clip $c \approx 0.5$.
This effectively fits the value function over a **smooth local neighborhood** of actions, enforcing that similar actions must have similar values.

#### 3. Delayed Policy Updates:
Critic networks must be well-trained before they can provide accurate policy gradients. TD3 updates the actor network $\mu_{\boldsymbol{\theta}}$ and target networks less frequently than the critics (typically updating the actor once every $d = 2$ critic updates):
- Every step: Update critics $Q_{\boldsymbol{\phi}_1}, Q_{\boldsymbol{\phi}_2}$.
- Every $d$ steps: Update actor $\mu_{\boldsymbol{\theta}}$, then update targets $\boldsymbol{\phi}_1^-, \boldsymbol{\phi}_2^-, \boldsymbol{\theta}^-$.

---

## 3. Geometric & Physical Interpretation

### 3.1 The Action Gradient Vector Field
In continuous action space $\mathbb{R}^m$:
- The critic $Q(s, a)$ forms a continuous scalar elevation landscape over the action coordinates.
- At any state $s$, the gradient $\nabla_a Q(s, a)$ is a **vector field of forces**: it points in the direction of steepest ascent on the value terrain.
- The actor's parameters $\boldsymbol{\theta}$ are updated so that action $\mu_{\boldsymbol{\theta}}(s)$ slides up the mountainside along the force lines of $\nabla_a Q$.

```
       Action Coordinate a
             ^
             |                 / Critic Contour Q(s, a)
             |                /
             |           *---/----> Force Vector: grad_a Q(s, a)
             |          /   /
             |         * Current Actor Output mu_theta(s)
             |        /
             +-------+------------------------------> State s
```

---

## 4. Real-World Analogy: The Robotic Surgery Precision Team

Consider a robotic surgical system performing delicate suturing:
- **DDPG (A single aggressive surgeon):** The robotic arm commands precise sub-millimeter motor angles. If the evaluator network hallucinates a tiny spike of reward at an unstable motor angle, the arm jerks violently into that spike, damaging surrounding tissue.
- **TD3 (The Two-Surgeon Consensus Protocol):**
  1. *Clipped Double Q:* Two independent neural evaluation models analyze the incision; only the more conservative estimate is trusted ($\min(Q_1, Q_2)$).
  2. *Smoothing Noise:* The system ensures the suture is safe not just at an infinitely precise coordinate, but also if there is a 0.1 mm mechanical vibration (smoothing noise).
  3. *Delayed Updates:* The arm does not change its operational protocol on every single microsecond sensor reading; it updates motors continuously but only revises overall strategy after verified sensor consensus.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 1D Continuous State & Action
Let us trace an exact, cell-by-cell numerical forward and backward pass of a TD3 step!

**Environment Transition:**
- State: $s = 2.0000$ (1D scalar)
- Action taken: $a = 1.0000$ (1D scalar)
- Immediate reward: $r = 3.0000$
- Successor state: $s' = 1.0000$
- Done: $d = 0$ (non-terminal)
- Hyperparameters: $\gamma = 0.9000$, learning rate $\alpha = 0.1000$, noise $\epsilon = +0.1000$, noise clip $c = 0.5000$, action limits $[-2.0, 2.0]$.

**Networks:**
- **Online Actor:** $\mu_\theta(s) = \theta \cdot s$, with parameter $\theta = 0.5000$.
- **Target Actor:** $\mu_{\theta^-}(s) = \theta^- \cdot s$, with parameter $\theta^- = 0.4000$.
- **Critic 1 (Online):** $Q_1(s, a) = \phi_{11} s + \phi_{12} a$, with weights $\boldsymbol{\phi}_1 = [1.0000, 2.0000]^\top$.
- **Critic 1 (Target):** $Q_1^-(s, a) = \boldsymbol{\phi}_1^- \cdot [s, a]^\top$, with weights $\boldsymbol{\phi}_1^- = [0.8000, 1.5000]^\top$.
- **Critic 2 (Online):** $Q_2(s, a) = \phi_{21} s + \phi_{22} a$, with weights $\boldsymbol{\phi}_2 = [2.0000, 1.0000]^\top$.
- **Critic 2 (Target):** $Q_2^-(s, a) = \boldsymbol{\phi}_2^- \cdot [s, a]^\top$, with weights $\boldsymbol{\phi}_2^- = [1.2000, 1.0000]^\top$.

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Value / Matrix |
| :--- | :--- | :--- |
| $s, s'$ | Current and Next State Scalars | $s = 2.0000, s' = 1.0000$ |
| $a$ | Taken Action | $1.0000$ |
| $r$ | Immediate Reward | $3.0000$ |
| $\mu_{\theta^-}(s')$ | Raw Target Actor Action | $\theta^- \cdot s' = 0.4000 \times 1.0 = 0.4000$ |
| $\tilde{a}'$ | Smoothed Clipped Target Action | $\mu_{\theta^-}(s') + \operatorname{clip}(\epsilon, -c, c)$ |
| $Q_1^-(s', \tilde{a}'), Q_2^-(s', \tilde{a}')$ | Target Critic Evaluations | $\boldsymbol{\phi}_1^- \cdot [s', \tilde{a}']^\top, \boldsymbol{\phi}_2^- \cdot [s', \tilde{a}']^\top$ |
| $y$ | Clipped Double Q Target | $r + \gamma \min(Q_1^-, Q_2^-)$ |
| $\mathcal{L}(\boldsymbol{\phi}_1)$ | Critic 1 Loss | $\frac{1}{2} (y - Q_1(s, a))^2$ |
| $\nabla_\theta J$ | Deterministic Policy Gradient | $\nabla_a Q_1(s, \mu_\theta(s)) \cdot \nabla_\theta \mu_\theta(s)$ |

---

### 5.3 Step 1: Target Action Generation with Policy Smoothing

1. Target Actor raw output:
   $$\mu_{\theta^-}(s') = \theta^- \cdot s' = 0.4000 \times 1.0000 = \mathbf{0.4000}$$
2. Add target smoothing noise $\epsilon = +0.1000$ (since $|0.1| \le 0.5$, clip has no effect):
   $$\tilde{a}' = \operatorname{clip}(0.4000 + 0.1000, -2.0, 2.0) = \mathbf{0.5000}$$

---

### 5.4 Step 2: Clipped Double Target Evaluation

Evaluate both target critics at next state $s' = 1.0000$ with smoothed action $\tilde{a}' = 0.5000$:

#### Target Critic 1:
$$Q_1^-(s', \tilde{a}') = \phi_{11}^- s' + \phi_{12}^- \tilde{a}' = (0.8000 \times 1.0000) + (1.5000 \times 0.5000)$$
$$= 0.8000 + 0.7500 = \mathbf{1.5500}$$

#### Target Critic 2:
$$Q_2^-(s', \tilde{a}') = \phi_{21}^- s' + \phi_{22}^- \tilde{a}' = (1.2000 \times 1.0000) + (1.0000 \times 0.5000)$$
$$= 1.2000 + 0.5000 = \mathbf{1.7000}$$

#### Clipped Double Q Minimum:
$$\min\left( Q_1^-(s', \tilde{a}'), Q_2^-(s', \tilde{a}') \right) = \min(1.5500, 1.7000) = \mathbf{1.5500}$$

Compute Target $y$:
$$y = r + \gamma \min(Q_1^-, Q_2^-) = 3.0000 + 0.9000 \times 1.5500 = 3.0000 + 1.3950 = \mathbf{4.3950}$$

---

### 5.5 Step 3: Online Critic Forward Passes & Loss

Evaluate online critics at current transition $(s = 2.0000, a = 1.0000)$:

#### Online Critic 1:
$$Q_1(s, a) = \phi_{11} s + \phi_{12} a = (1.0000 \times 2.0000) + (2.0000 \times 1.0000) = 2.0000 + 2.0000 = \mathbf{4.0000}$$
TD Error for Critic 1:
$$u_1 = y - Q_1(s, a) = 4.3950 - 4.0000 = \mathbf{+0.3950}$$
Loss for Critic 1:
$$\mathcal{L}(\boldsymbol{\phi}_1) = \frac{1}{2} u_1^2 = \frac{1}{2} (0.3950)^2 \approx \mathbf{0.07801}$$

#### Online Critic 2:
$$Q_2(s, a) = \phi_{21} s + \phi_{22} a = (2.0000 \times 2.0000) + (1.0000 \times 1.0000) = 4.0000 + 1.0000 = \mathbf{5.0000}$$
TD Error for Critic 2:
$$u_2 = y - Q_2(s, a) = 4.3950 - 5.0000 = \mathbf{-0.6050}$$
Loss for Critic 2:
$$\mathcal{L}(\boldsymbol{\phi}_2) = \frac{1}{2} u_2^2 = \frac{1}{2} (-0.6050)^2 \approx \mathbf{0.18301}$$

---

### 5.6 Step 4: Critic Gradient Updates ($\alpha = 0.1000$)

#### Update Critic 1:
$$\nabla_{\boldsymbol{\phi}_1} \mathcal{L} = - u_1 \begin{bmatrix} s \\ a \end{bmatrix} = - 0.3950 \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} -0.7900 \\ -0.3950 \end{bmatrix}$$
$$\boldsymbol{\phi}_{1, \text{new}} = \boldsymbol{\phi}_1 - \alpha \nabla_{\boldsymbol{\phi}_1} = \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix} - 0.1000 \begin{bmatrix} -0.7900 \\ -0.3950 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.0790 \\ 2.0395 \end{bmatrix}}$$

#### Update Critic 2:
$$\nabla_{\boldsymbol{\phi}_2} \mathcal{L} = - u_2 \begin{bmatrix} s \\ a \end{bmatrix} = - (-0.6050) \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} +1.2100 \\ +0.6050 \end{bmatrix}$$
$$\boldsymbol{\phi}_{2, \text{new}} = \boldsymbol{\phi}_2 - \alpha \nabla_{\boldsymbol{\phi}_2} = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix} - 0.1000 \begin{bmatrix} 1.2100 \\ 0.6050 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.8790 \\ 0.9395 \end{bmatrix}}$$

---

### 5.7 Step 5: Deterministic Policy Gradient Update

Now compute the gradient to maximize $Q_1(s, \mu_\theta(s))$:
1. Current actor action at $s = 2.0000$:
   $$a_{\text{actor}} = \mu_\theta(s) = \theta \cdot s = 0.5000 \times 2.0000 = \mathbf{1.0000}$$
2. Action gradient of Critic 1:
   $$\nabla_a Q_1(s, a) = \frac{\partial}{\partial a} (\phi_{11} s + \phi_{12} a) = \phi_{12} = \mathbf{2.0000}$$
3. Parameter gradient of Actor:
   $$\nabla_\theta \mu_\theta(s) = \frac{\partial}{\partial \theta} (\theta \cdot s) = s = \mathbf{2.0000}$$
4. Total Policy Gradient via Chain Rule:
   $$g_\theta = \nabla_\theta \mu_\theta(s) \cdot \nabla_a Q_1 = 2.0000 \times 2.0000 = \mathbf{+4.0000}$$

Actor Parameter Update ($\alpha = 0.1000$, gradient ascent):
$$\theta_{\text{new}} = \theta + \alpha g_\theta = 0.5000 + 0.1000 \times 4.0000 = 0.5000 + 0.4000 = \mathbf{0.9000}$$

---

### 5.8 Visual Summary Grid

| Component | Entity / Operation | Formula / Value | Gradient Component | Updated Parameter |
| :---: | :---: | :---: | :---: | :---: |
| **Smoothed Action** | Target Actor + Noise | $0.4000 + 0.1000 = \mathbf{0.5000}$ | - | - |
| **Clipped Double Q** | $\min(Q_1^-, Q_2^-)$ | $\min(1.5500, 1.7000) = \mathbf{1.5500}$ | - | - |
| **Target $y$** | $r + \gamma \min$ | $3.0 + 0.9(1.55) = \mathbf{4.3950}$ | - | - |
| **Critic 1 Loss** | $0.5 \times (4.395 - 4.0)^2$ | $0.07801$ | $\nabla = [-0.7900, -0.3950]^\top$ | $\boldsymbol{\phi}_1: \mathbf{[1.0790, 2.0395]^\top}$ |
| **Critic 2 Loss** | $0.5 \times (4.395 - 5.0)^2$ | $0.18301$ | $\nabla = [+1.2100, +0.6050]^\top$ | $\boldsymbol{\phi}_2: \mathbf{[1.8790, 0.9395]^\top}$ |
| **Actor Gradient** | $\nabla_a Q_1 \cdot \nabla_\theta \mu$ | $2.0000 \times 2.0000 = \mathbf{+4.0000}$ | Ascent step $+0.4$ | $\theta: 0.5000 \to \mathbf{0.9000}$ |

---

## 6. Solved Illustrations

### Illustration 1: Why Clipped Double Q Fixes Continuous Overestimation
**Problem:**
Let $Q_{\text{approx}}(s, a) = Q_{\text{true}}(s, a) + \epsilon(s, a)$, where $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, \sigma^2)$ are independent errors.
Prove that $\mathbb{E}[\min(Q_1, Q_2)] \le Q_{\text{true}}$.

**Solution:**
$$\mathbb{E}[\min(Q_1, Q_2)] = Q_{\text{true}} + \mathbb{E}[\min(\epsilon_1, \epsilon_2)]$$
For two independent zero-mean normal random variables with variance $\sigma^2$:
$$\mathbb{E}[\min(\epsilon_1, \epsilon_2)] = - \frac{\sigma}{\sqrt{\pi}} < 0$$
Therefore:
$$\mathbb{E}[\min(Q_1, Q_2)] = Q_{\text{true}} - \frac{\sigma}{\sqrt{\pi}} < Q_{\text{true}}$$
The minimum estimator creates a slight **underestimation bias**, which is fundamentally benign (it merely acts like an increased discount factor $\gamma$), whereas overestimation creates an explosive positive feedback loop that destabilizes control! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **Robotics & Continuous Manipulation:** TD3 is a standard workhorse baseline in Mujoco (HalfCheetah, Ant, Humanoid, Walker2d) and Isaac Gym (NVIDIA GPU-accelerated robotics).
- **Soft Actor-Critic (SAC - Chapter 11.20):** SAC adopted all three TD3 improvements (Clipped Double Q, target networks, entropy-smoothed actions) to create an even more sample-efficient maximum entropy continuous controller.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of the Part 5 hand calculations:
   - Target $y = 4.3950$
   - Critic losses $0.07801$ and $0.18301$
   - Updated critic weights $\boldsymbol{\phi}_1 = [1.0790, 2.0395]^\top, \boldsymbol{\phi}_2 = [1.8790, 0.9395]^\top$
   - Updated actor parameter $\theta_{\text{new}} = 0.9000$ matching PyTorch to $< 10^{-14}$.
2. A complete TD3 agent implementation (Replay Buffer, Twin Critics, Target Smoothing, Delayed Updates) tested on continuous Inverted Pendulum dynamics.

See implementation in:
[`11_reinforcement_learning/code/16_continuous_action_spaces_ddpg_td3.py`](./code/16_continuous_action_spaces_ddpg_td3.py)
