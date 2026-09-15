# Module 11.10: Deep Q-Networks (DQN)

---

## 1. Intuition & 101 Motivation

In 2015, Google DeepMind published a landmark breakthrough in *Nature* (Mnih et al.): an algorithm called **Deep Q-Networks (DQN)** that learned to master 49 different Atari 2600 games directly from raw pixel observations ($84 \times 84 \times 4$ luminance images), outperforming professional human testers on the majority of games without any game-specific handcrafting.

Prior to DQN, training non-linear deep neural networks to represent action-value functions $Q(s, a; \theta)$ in reinforcement learning was notoriously unstable and frequently diverged to garbage values. Three fatal obstacles plagued naive Deep Q-Learning:
1. **Strongly Correlated Data:** Consecutive video frames $(s_t, s_{t+1}, s_{t+2})$ in an RL trajectory violate the core assumption of independent and identically distributed (i.i.d.) training data required by stochastic gradient descent.
2. **The "Moving Goalpost" Instability:** When updating the network weights $\theta$ to make $Q(s_t, a_t; \theta)$ approach target $Y_t = r + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$, the target itself changes! A step to correct the prediction at $s_t$ inadvertently alters the predictions at $s_{t+1}$, inducing positive feedback loops and policy oscillation.
3. **Policy Distribution Shift:** Small changes in $Q(s, a; \theta)$ create massive, discontinuous changes in the policy $\pi(s) = \arg\max_a Q(s, a)$, dramatically shifting the distribution of states the agent explores.

DQN solved these fundamental problems with two breakthrough mechanisms:
- **Experience Replay ($\mathcal{D}$):** Stores past transitions and samples mini-batches uniformly at random, breaking temporal autocorrelation and restoring i.i.d. conditions.
- **Target Network ($\theta^-$):** Decouples the current online network from the TD target calculation, freezing the target for $C$ steps to stabilize regression.

```
+-----------------------------------------------------------------------------------------+
|                                    DQN ARCHITECTURE                                     |
|                                                                                         |
|   Environment ----(s, a, r, s', d)----> [ Experience Replay Buffer D ]                  |
|        ^                                            |                                   |
|        |                                    Sample Mini-batch                           |
|   Act  | eps-greedy                                 v                                   |
|        |                          +-----------------------------------+                 |
|        |                          |  Online Network Q(s, a; theta)    |                 |
|        |                          +-----------------------------------+                 |
|        |                                            |                                   |
|        |                                       Huber Loss                               |
|        |                                            ^                                   |
|        |                                            | TD Target                         |
|        |                          +-----------------------------------+                 |
|        +--------------------------|  Target Network Q(s', a'; theta-) |                 |
|                                   +-----------------------------------+                 |
|                                            ^                                            |
|                                            | Every C steps: theta- <-- theta            |
|                                            +--------------------------------            |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The DQN Loss Function

At each training iteration $i$, a mini-batch of transitions $B = \{(s, a, r, s', d)\}$ is sampled uniformly from the replay buffer $\mathcal{D}$:
$$(s, a, r, s', d) \sim U(\mathcal{D})$$
where $d \in \{0, 1\}$ is the terminal state indicator flag ($d = 1$ if $s'$ is terminal, $d = 0$ otherwise).

The target value $Y_i$ is computed using the **target network** parameterized by $\theta^-_i$:
$$Y_i \triangleq r + (1 - d) \gamma \max_{a' \in \mathcal{A}} Q(s', a'; \theta^-_i)$$

The objective is to minimize the empirical risk under the **Huber Loss** $\ell_\delta$:
$$\mathcal{L}(\theta_i) \triangleq \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \ell_\delta \left( Y_i - Q(s, a; \theta_i) \right) \right]$$

---

### 2.2 The Huber Loss (Smooth L1 Loss)

In deep reinforcement learning, immediate rewards and bootstrapped errors can occasionally be enormous (e.g., scoring $10,000$ points in an Atari game). Under standard Mean Squared Error (MSE), large TD errors $u = Y - Q$ produce quadratic penalties $\frac{1}{2} u^2$ and linear gradients $u$, causing gradient explosion.

The **Huber Loss** with threshold $\delta = 1.0$ acts quadratically for small errors (smooth convergence) and linearly for large errors (robustness):

$$\ell_\delta(u) \triangleq \begin{cases} \frac{1}{2} u^2 & \text{if } |u| \le \delta \\ \delta \left( |u| - \frac{1}{2} \delta \right) & \text{otherwise} \end{cases}$$

Its first derivative (gradient signal) is strictly clipped within $[-\delta, +\delta]$:
$$\nabla_u \ell_\delta(u) = \begin{cases} u & \text{if } |u| \le \delta \\ \delta \operatorname{sgn}(u) & \text{otherwise} \end{cases}$$

```
    Loss Value l(u)                             Gradient dl/du
        ^                                            ^
        |          / MSE (u^2)                    +1 |           /---------- (+1)
        |         /                                  |          /
        |        /   / Huber                         |         /
        |       /   /                                0 +------+------+-----> u
        |      /---/                                 |       /
        |     /   /                                  |      /
        +----+---+---+---> u                      -1 | ----/                 (-1)
            -1   0  +1                                  -1   0  +1
```

---

### 2.3 Gradient Computation

Differentiating the loss with respect to the online parameters $\theta$ (treating the target $Y$ as a constant scalar):

$$\nabla_{\theta} \mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ - \nabla_u \ell_\delta \left( Y - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

For errors within the quadratic zone ($|Y - Q| \le 1$):
$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E} \left[ - \left( Y - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

For errors in the linear zone ($|Y - Q| > 1$):
$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E} \left[ - \operatorname{sgn}\left( Y - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

---

### 2.4 Target Network Synchronization

Two synchronization schemes exist:
1. **Periodic Hard Update (Original Mnih et al., 2015):**
   Every $C$ environment steps (typically $C = 10,000$ in Atari), copy the parameters completely:
   $$\theta^- \leftarrow \theta$$
2. **Polyak Averaging / Soft Update (Lillicrap et al., 2016):**
   At every single training step, update the target network via an exponential moving average with hyperparameter $\tau \ll 1$ (e.g., $\tau = 0.005$):
   $$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

---

## 3. Geometric Interpretation: Freezing the Target Manifold

In parameter space $\mathbb{R}^P$:
- When training without a target network ($\theta^- = \theta$), the loss surface $\mathcal{L}(\theta) = \frac{1}{2} (r + \gamma \max Q(s', a'; \theta) - Q(s, a; \theta))^2$ changes dynamically with every gradient step. The minimum shifts as the weights move, turning the optimization landscape into a rolling, non-stationary wave where the gradient $\nabla_\theta \mathcal{L}$ can point in chaotic, destabilizing directions.
- By fixing $\theta^-$ as a constant vector, the target $Y = r + \gamma \max Q(s', a'; \theta^-)$ becomes a stationary fixed landmark in $\mathbb{R}$. The loss landscape freezes into a stationary quadratic bowl for $C$ steps, allowing standard stochastic gradient descent to descend monotonically.

```
       WITHOUT TARGET NETWORK                         WITH TARGET NETWORK
         (Moving Goalpost)                         (Stationary Goalpost)

           Loss                                        Loss
          Landscape                                   Landscape
             ~ ~                                          \
            ~ * ~  <- Minimum moves away                   \     * Minimum frozen at Y!
           ~ ~ ~     at every step!                         \   /
          ~ ~ ~ ~                                            \_/
```

---

## 4. Real-World Analogy: The Exam Preparation Strategy

Imagine a student studying for the SATs:
- **No Replay Buffer (Correlated Data):** The student studies 500 geometry questions in a row on Monday, then 500 vocabulary questions on Tuesday. By Friday, the student has completely forgotten geometry due to catastrophic forgetting.
  - **With Replay Buffer:** The student writes every question on an index card and tosses it into a giant box. Each study session draws a random shuffled handful of 32 cards across algebra, vocabulary, and geometry.
- **No Target Network (Moving Answer Key):** Every time the student gets a question wrong, the teacher changes the textbook answers to match the student's partial guess!
  - **With Target Network:** The textbook answer key is locked for the entire month. The student tests against fixed answers. Only on the 1st of every month does the teacher release an updated, verified answer key.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 1-Layer Linear Neural Network
To see the exact arithmetic down to machine precision, consider a single linear layer with 2 inputs and 2 discrete actions:
$$Q(s, a; \mathbf{W}) = \mathbf{W} s = \begin{bmatrix} W_{00} & W_{01} \\ W_{10} & W_{11} \end{bmatrix} \begin{bmatrix} s_0 \\ s_1 \end{bmatrix}$$
where row $0$ outputs $Q(s, a_0)$ and row $1$ outputs $Q(s, a_1)$.

**Hyperparameters:**
- Discount factor: $\gamma = 0.9000$
- Learning rate: $\alpha = 0.1000$
- Huber loss threshold: $\delta = 1.0000$

**Initial Online Network Weights $\mathbf{W}$:**
$$\mathbf{W} = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix}$$

**Target Network Weights $\mathbf{W}^-$ (Frozen):**
$$\mathbf{W}^- = \begin{bmatrix} 0.4000 & 0.1000 \\ -0.1000 & 0.6000 \end{bmatrix}$$

**Sampled Mini-Batch of 2 Transitions:**
- **Sample 1:** $s^{(1)} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$, taken action $a = 0$ ($a_0$), reward $r = 1.0$, next state $s'^{(1)} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$, done $d = 0$.
- **Sample 2:** $s^{(2)} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$, taken action $a = 1$ ($a_1$), reward $r = 2.0$, next state $s'^{(2)} = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$, done $d = 1$ (terminal!).

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Matrix / Vector in Hand Walkthrough |
| :--- | :--- | :--- |
| $\mathbf{W}$ | Online Network Parameter Matrix | $2 \times 2$ weights generating current predictions $Q(s, a; \mathbf{W})$ |
| $\mathbf{W}^-$ | Target Network Parameter Matrix | $2 \times 2$ frozen weights generating next-state targets $Q(s', a'; \mathbf{W}^-)$ |
| $s$ | Current State Input Vector | $2 \times 1$ column vector |
| $s'$ | Successor State Input Vector | $2 \times 1$ column vector |
| $Y$ | Bootstrapped Target | $r + (1 - d) \gamma \max_{a'} Q(s', a'; \mathbf{W}^-)$ |
| $u$ | TD Error Residue | $Y - Q(s, a; \mathbf{W})$ |
| $\ell_\delta(u)$ | Huber Loss | Quadratic if $|u| \le 1$, Linear if $|u| > 1$ |
| $g_u$ | Huber Gradient $\nabla_u \ell_\delta$ | $u$ if $|u| \le 1$, else $\operatorname{sgn}(u)$ |
| $\nabla_\mathbf{W} \mathcal{L}$ | Weight Gradient Tensor | Matrix of partial derivatives $\frac{\partial \mathcal{L}}{\partial W_{ij}}$ |

---

### 5.3 Step 1: Forward Pass on Target Network (Compute $Y$)

#### For Sample 1 ($s'^{(1)} = [0.0, 1.0]^\top, r = 1.0, d = 0$):
Compute next-state Q-values using frozen target network $\mathbf{W}^-$:
$$Q(s'^{(1)}; \mathbf{W}^-) = \begin{bmatrix} 0.4000 & 0.1000 \\ -0.1000 & 0.6000 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.1000 \\ 0.6000 \end{bmatrix}$$
Maximum target Q-value:
$$\max_{a'} Q(s'^{(1)}, a'; \mathbf{W}^-) = \max(0.1000, 0.6000) = \mathbf{0.6000}$$

Compute Target $Y^{(1)}$:
$$Y^{(1)} = r + (1 - d) \gamma \max_{a'} Q = 1.0000 + (1 - 0) \times 0.9000 \times 0.6000 = 1.0000 + 0.5400 = \mathbf{1.5400}$$

#### For Sample 2 ($s'^{(2)} = [1.0, 1.0]^\top, r = 2.0, d = 1$ [Terminal]):
Because $d = 1$, the successor state value is zeroed out:
$$Y^{(2)} = r + (1 - 1) \gamma \max_{a'} Q = 2.0000 + 0.0000 = \mathbf{2.0000}$$

---

### 5.4 Step 2: Forward Pass on Online Network (Compute Current Predictions)

#### For Sample 1 ($s^{(1)} = [1.0, 0.0]^\top$, taken action $a = 0$):
$$Q(s^{(1)}; \mathbf{W}) = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5000} \\ -0.3000 \end{bmatrix}$$
The prediction for chosen action $a = 0$ is:
$$Q(s^{(1)}, a_0; \mathbf{W}) = \mathbf{0.5000}$$

TD Error residue $u^{(1)}$:
$$u^{(1)} = Y^{(1)} - Q(s^{(1)}, a_0; \mathbf{W}) = 1.5400 - 0.5000 = \mathbf{1.0400}$$

#### For Sample 2 ($s^{(2)} = [0.0, 1.0]^\top$, taken action $a = 1$):
$$Q(s^{(2)}; \mathbf{W}) = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.2000 \\ \mathbf{0.8000} \end{bmatrix}$$
The prediction for chosen action $a = 1$ is:
$$Q(s^{(2)}, a_1; \mathbf{W}) = \mathbf{0.8000}$$

TD Error residue $u^{(2)}$:
$$u^{(2)} = Y^{(2)} - Q(s^{(2)}, a_1; \mathbf{W}) = 2.0000 - 0.8000 = \mathbf{1.2000}$$

---

### 5.5 Step 3: Huber Loss & Gradient Evaluation

#### For Sample 1:
- Error $u^{(1)} = 1.0400 > 1.0000$ (Falls into **Linear Zone**!).
- Huber Loss:
  $$\ell_\delta(u^{(1)}) = 1.0 \times \left( |1.0400| - 0.5 \times 1.0 \right) = 1.0400 - 0.5000 = \mathbf{0.5400}$$
- Huber Gradient:
  $$g_u^{(1)} = \operatorname{sgn}(1.0400) = \mathbf{+1.0000}$$

#### For Sample 2:
- Error $u^{(2)} = 1.2000 > 1.0000$ (Falls into **Linear Zone**!).
- Huber Loss:
  $$\ell_\delta(u^{(2)}) = 1.2000 - 0.5000 = \mathbf{0.7000}$$
- Huber Gradient:
  $$g_u^{(2)} = \operatorname{sgn}(1.2000) = \mathbf{+1.0000}$$

**Mean Mini-Batch Loss:**
$$\mathcal{L} = \frac{0.5400 + 0.7000}{2} = \frac{1.2400}{2} = \mathbf{0.6200}$$

---

### 5.6 Step 4: Backward Pass & Online Weight Update

The loss with respect to prediction is $\frac{\partial \mathcal{L}}{\partial Q} = - \frac{1}{|B|} g_u$.
Since $Q(s, a) = \sum_j W_{a, j} s_j$, the weight gradient for row $a$ is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{a, :}} = - \frac{1}{|B|} g_u s^\top$$

#### Gradient from Sample 1 ($a = 0, s = [1.0, 0.0]^\top, g_u = 1.0$):
$$\nabla_\mathbf{W}^{(1)} = \begin{bmatrix} -1.0 \times 1.0 & -1.0 \times 0.0 \\ 0.0 & 0.0 \end{bmatrix} = \begin{bmatrix} -1.0000 & 0.0000 \\ 0.0000 & 0.0000 \end{bmatrix}$$

#### Gradient from Sample 2 ($a = 1, s = [0.0, 1.0]^\top, g_u = 1.0$):
$$\nabla_\mathbf{W}^{(2)} = \begin{bmatrix} 0.0 & 0.0 \\ -1.0 \times 0.0 & -1.0 \times 1.0 \end{bmatrix} = \begin{bmatrix} 0.0000 & 0.0000 \\ 0.0000 & -1.0000 \end{bmatrix}$$

#### Average Batch Gradient $\nabla_\mathbf{W} \mathcal{L}$:
$$\nabla_\mathbf{W} \mathcal{L} = \frac{1}{2} \left( \nabla_\mathbf{W}^{(1)} + \nabla_\mathbf{W}^{(2)} \right) = \begin{bmatrix} -0.5000 & 0.0000 \\ 0.0000 & -0.5000 \end{bmatrix}$$

#### Gradient Descent Step ($\alpha = 0.1000$):
$$\mathbf{W}_{\text{new}} = \mathbf{W} - \alpha \nabla_\mathbf{W} \mathcal{L} = \begin{bmatrix} 0.5000 & 0.2000 \\ -0.3000 & 0.8000 \end{bmatrix} - 0.1000 \begin{bmatrix} -0.5000 & 0.0000 \\ 0.0000 & -0.5000 \end{bmatrix}$$
$$= \begin{bmatrix} 0.5000 + 0.0500 & 0.2000 \\ -0.3000 & 0.8000 + 0.0500 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.5500 & 0.2000 \\ -0.3000 & 0.8500 \end{bmatrix}}$$

The target weights $\mathbf{W}^-$ remain **completely unchanged**:
$$\mathbf{W}^- = \begin{bmatrix} 0.4000 & 0.1000 \\ -0.1000 & 0.6000 \end{bmatrix}$$

---

### 5.7 Visual Summary Tensor Grid

| Sample | Input $s$ | Target $Y$ | Predict $Q(s, a)$ | Residue $u$ | Huber Loss | Huber Grad $g_u$ | Weight Update $\Delta W$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $[1, 0]^\top, a=0$ | $1.5400$ | $0.5000$ | $+1.0400$ | $0.5400$ | $+1.0000$ | $\Delta W_{00} = +0.0500$ |
| **2** | $[0, 1]^\top, a=1$ | $2.0000$ | $0.8000$ | $+1.2000$ | $0.7000$ | $+1.0000$ | $\Delta W_{11} = +0.0500$ |

Every single arithmetic step is completely crystal clear and verified to exact decimals.

---

## 6. Solved Illustrations

### Illustration 1: The Atari Frame-Skipping & Frame-Stacking Pipeline
**Problem:**
Why can a Deep Q-Network not take a single $84 \times 84$ Atari video frame as state input $s_t$?
**Solution:**
A single video frame contains only position information, but **zero velocity or acceleration information**.
In *Pong* or *Breakout*, seeing the ball at coordinates $(x, y)$ in a single snapshot does not tell the network whether the ball is traveling left or right, up or down.
Formally, a single screen observation is non-Markovian:
$$\mathbb{P}(S_{t+1} \mid S_t) \neq \mathbb{P}(S_{t+1} \mid S_t, S_{t-1})$$
The system is a Partially Observable Markov Decision Process (POMDP).
By **stacking the 4 most recent frames** into an $84 \times 84 \times 4$ tensor, the network can compute finite differences:
$$v \approx \frac{x_t - x_{t-1}}{\Delta t}, \quad a \approx \frac{v_t - v_{t-1}}{\Delta t}$$
restoring the Markov property $\mathbb{P}(S_{t+1} \mid S_t, \dots, S_0) = \mathbb{P}(S_{t+1} \mid S_t)$. $\blacksquare$

---

## 7. Deep Learning Connection & Application

### 1. Convolutional Nature Architecture (Mnih et al., 2015)
The classic Nature DQN architecture processes $84 \times 84 \times 4$ frames through:
1. Conv2D: 32 filters of size $8 \times 8$, stride 4, ReLU.
2. Conv2D: 64 filters of size $4 \times 4$, stride 2, ReLU.
3. Conv2D: 64 filters of size $3 \times 3$, stride 1, ReLU.
4. Fully Connected: 512 units, ReLU.
5. Linear Output: $|\mathcal{A}|$ Q-values (one for each joystick button combination).

### 2. Reward Clipping
To ensure stable gradients across vastly different Atari games without tuning game-specific hyperparameters, all positive rewards were clipped to $+1$, all negative rewards to $-1$, and neutral to $0$.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of the Part 5 hand calculations:
   - Target values $Y^{(1)} = 1.5400, Y^{(2)} = 2.0000$
   - Huber losses $0.5400, 0.7000$ and mean loss $0.6200$
   - Updated weights $\mathbf{W}_{\text{new}} = \begin{bmatrix} 0.5500 & 0.2000 \\ -0.3000 & 0.8500 \end{bmatrix}$ matching PyTorch / NumPy to $< 10^{-14}$.
2. Complete standalone Deep Q-Network agent (with Replay Buffer, Target Network, and $\epsilon$-greedy exploration) solving the CartPole balancing environment.

See implementation in:
[`11_reinforcement_learning/code/10_deep_q_networks.py`](./code/10_deep_q_networks.py)
