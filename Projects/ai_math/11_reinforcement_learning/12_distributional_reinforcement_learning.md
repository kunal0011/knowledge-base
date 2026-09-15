# Module 11.12: Distributional Reinforcement Learning (C51 & QR-DQN)

---

## 1. Intuition & 101 Motivation

Throughout the classical history of reinforcement learning—from Bellman (1957) to DQN (2015)—the central object of study has been the **expectation of returns**:
$$Q(s, a) \triangleq \mathbb{E} [G_t \mid S_t = s, A_t = a]$$

While taking expectations simplifies math, it discards critical information about the **random variable of returns** $Z(s, a)$:
$$Z(s, a) \triangleq \sum_{t=0}^\infty \gamma^t R_{t+1}$$
whose expectation is the standard Q-value: $\mathbb{E}[Z(s, a)] = Q(s, a)$.

**Distributional Reinforcement Learning** (Bellemare et al., 2017) fundamentally changes this paradigm: instead of estimating the scalar mean $Q(s, a)$, the agent estimates the **complete probability distribution** of $Z(s, a)$.

### Why Distributional RL Outperforms Expectation RL:
1. **Preserving Multi-Modality:** In stochastic environments, a state might have a $50\%$ chance of $+100$ and $50\%$ chance of $0$. A scalar Q-value collapses this to $50$, indistinguishable from a deterministic outcome of $50$. Distributional RL preserves both modes.
2. **Mitigating Non-Stationarity & Policy Jumps:** Because distributions contain rich structural details about future state transitions, gradient updates adjust the full distribution rather than pushing a scalar average up or down, stabilizing deep representations.
3. **Risk-Sensitive Decision Making:** An agent can optimize for metrics beyond expected value, such as Value at Risk (VaR), Conditional Value at Risk (CVaR), or worst-case tail protection.

```
       CLASSICAL RL (Scalar Mean)                    DISTRIBUTIONAL RL (Full Spectrum)
                                                                 Probability
                  Q(s, a)                                           ^
                     |                                              |      /\
                     v                                              |     /  \    /\
                 +-------+                                          |    /    \  /  \
                 | 50.00 |                                          +---+------+----+---> Return Z
                 +-------+                                              0      50   100
             (One single number)                                  (Multi-modal distribution!)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Distributional Bellman Equation

Let $\stackrel{D}{=}$ denote equality in distribution. The **Distributional Bellman Equation** for policy $\pi$ is:
$$Z^\pi(s, a) \stackrel{D}{=} R(s, a) + \gamma Z^\pi(S', A')$$
where $S' \sim \mathcal{P}(\cdot \mid s, a)$ and $A' \sim \pi(\cdot \mid S')$.

For the optimal policy (control):
$$Z^*(s, a) \stackrel{D}{=} R(s, a) + \gamma Z^*(S', a^*)$$
where $a^* = \arg\max_{a'} \mathbb{E}[Z^*(S', a')]$. Note that the greedy choice is still made with respect to the **expectation** of the return distribution!

---

### 2.2 The Wasserstein Metric & Contraction Mapping Theorem

To establish convergence, we must measure the distance between two probability distributions $P$ and $Q$.
The **$p$-Wasserstein metric** (Earth Mover's Distance) between distributions $F_1, F_2 \in \mathcal{P}(\mathbb{R})$ is defined as:
$$\mathcal{W}_p(F_1, F_2) \triangleq \left( \int_0^1 \left| F_1^{-1}(u) - F_2^{-1}(u) \right|^p du \right)^{1/p}$$
where $F^{-1}(u) \triangleq \inf \{ x \in \mathbb{R} : F(x) \ge u \}$ is the inverse cumulative distribution function (quantile function).

#### Theorem: Contraction of the Distributional Bellman Operator (Bellemare et al., 2017)
Let $\mathcal{T}^\pi$ be the distributional Bellman operator defined by:
$$\mathcal{T}^\pi Z(s, a) \stackrel{D}{=} R(s, a) + \gamma Z(S', A')$$
Then $\mathcal{T}^\pi$ is a **$\gamma$-contraction** in the maximal $p$-Wasserstein metric $\bar{\mathcal{W}}_p$:
$$\bar{\mathcal{W}}_p(\mathcal{T}^\pi Z_1, \mathcal{T}^\pi Z_2) \le \gamma \bar{\mathcal{W}}_p(Z_1, Z_2)$$
where $\bar{\mathcal{W}}_p(Z_1, Z_2) \triangleq \sup_{(s,a)} \mathcal{W}_p(Z_1(s, a), Z_2(s, a))$.

Consequently, under policy evaluation, repeated application of $\mathcal{T}^\pi$ converges to the unique fixed-point return distribution $Z^\pi$!

---

### 2.3 The Categorical Algorithm: C51 (Bellemare et al., ICML 2017)

In practice, a continuous distribution cannot be represented exactly by a neural network. C51 represents the return distribution using a discrete **categorical distribution** over $N = 51$ fixed, equidistant atoms:
$$\operatorname{supp}(Z) = \{z_0, z_1, \dots, z_{N-1}\}$$
$$z_i \triangleq V_{\min} + i \cdot \Delta z, \quad \Delta z \triangleq \frac{V_{\max} - V_{\min}}{N - 1}$$
where $[V_{\min}, V_{\max}]$ is the predefined return support range (e.g., $[-10, +10]$).

The network outputs a categorical probability vector $\mathbf{p}(s, a; \theta) \in \Delta^N$:
$$p_i(s, a; \theta) = \frac{\exp(\psi_i(s, a; \theta))}{\sum_{j=0}^{N-1} \exp(\psi_j(s, a; \theta))}$$
where $p_i(s, a) = \mathbb{P}(Z(s, a) = z_i)$.

The scalar expected value is computed trivially via dot product:
$$Q(s, a) = \sum_{i=0}^{N-1} z_i p_i(s, a) = \mathbf{z}^\top \mathbf{p}(s, a)$$

#### The Categorical Projection Step $\Phi$
Applying the Bellman update shifts and scales each atom:
$$\hat{T} z_j = r + \gamma z_j$$
Crucially, the shifted location $\hat{T} z_j$ will generally **not coincide** with any of the fixed support atoms $\{z_i\}$.
We must project the shifted probability mass $p_j(s', a^*)$ onto the adjacent atoms $\{z_l, z_{l+1}\}$ via the **Cramer Projection** $\Phi$:

For each shifted atom $j \in \{0, \dots, N-1\}$:
1. Clip to support: $\hat{z}_j = \operatorname{clip}(\hat{T} z_j, V_{\min}, V_{\max})$.
2. Compute fractional index: $b_j = \frac{\hat{z}_j - V_{\min}}{\Delta z}$.
3. Lower and upper integer atom indices: $l = \lfloor b_j \rfloor, \quad u = \lceil b_j \rceil$.
4. Distribute probability mass $p_j(s', a^*)$ proportionally:
   $$m_l \leftarrow m_l + p_j(s', a^*) \cdot (u - b_j)$$
   $$m_u \leftarrow m_u + p_j(s', a^*) \cdot (b_j - l)$$

#### The Loss Function: Cross-Entropy
The network parameters $\theta$ are trained by minimizing the **Kullback-Leibler (KL) divergence** (equivalently, cross-entropy) between the projected target distribution $\mathbf{m}$ and the predicted distribution $\mathbf{p}(s, a; \theta)$:

$$\mathcal{L}(\theta) = D_{\text{KL}}(\mathbf{m} \parallel \mathbf{p}(s, a; \theta)) = - \sum_{i=0}^{N-1} m_i \log p_i(s, a; \theta)$$

---

### 2.4 Quantile Regression DQN: QR-DQN (Dabney et al., AAAI 2018)

C51 fixes the support $\{z_i\}$ and learns variable probabilities $\{p_i\}$. This creates a theoretical mismatch because the projection step is a heuristic non-expansion.

**QR-DQN** reverses this geometry:
- It **fixes uniform probabilities** $\tau_i = \frac{2i - 1}{2N}$ for $N$ quantiles.
- It **learns the variable support locations** $\theta_i(s, a) \in \mathbb{R}$.

The network is trained using the **Quantile Huber Loss**:
$$\rho_\tau^\kappa(u) = |\tau - \mathbb{I}(u < 0)| \cdot \mathcal{L}_\kappa(u)$$
QR-DQN provides rigorous theoretical guarantees minimizing the 1-Wasserstein distance directly via stochastic gradient descent!

---

## 3. Geometric & Physical Interpretation

### The "Bucket Pouring" Projection Geometry
Imagine the support atoms as $N$ buckets placed along a line at positions $z_0, z_1, \dots, z_{N-1}$.
1. The transition $(r, \gamma)$ slides the contents of bucket $j$ to a new coordinate $\hat{T} z_j = r + \gamma z_j$.
2. This new point lands between bucket $l$ and bucket $u$.
3. The Cramer projection behaves like **lever mechanics** or **bilinear interpolation**: the mass $p_j$ is split between the two enclosing buckets inversely proportional to their Euclidean distance from $\hat{T} z_j$.

```
           z_l                                z_u
          Bucket                             Bucket
            |                                  |
            +------------*---------------------+
                       T(z_j)
            <--- u - b ---> <----- b - l ----->
            (goes to z_l)     (goes to z_u)
```

---

## 4. Real-World Analogy: Investment Portfolio Assessment

Imagine evaluating two investment strategies:
- **Strategy A:** Puts all money into government treasury bonds.
  Returns: Guaranteed $+5\%$ with probability $1.0$. Mean $= +5\%$.
- **Strategy B:** Puts money into tech venture capital.
  Returns: $+50\%$ with probability $0.2$, $0\%$ with probability $0.5$, and $-50\%$ with probability $0.3$. Mean $= (0.2)(50) + (0.5)(0) + (0.3)(-50) = 10 - 15 = -5\%$.

Standard scalar RL only sees $-5\%$ vs $+5\%$ and discards Strategy B. But what if the investor is a hedge fund seeking asymmetric upside exposure? Distributional RL maintains the complete return histogram, allowing the decision maker to price upside convexity and downside ruin probabilities!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 3-Atom Categorical Projection ($N = 3$)
To observe the exact projection arithmetic down to every single decimal digit, consider a minimal 3-atom support:
- Support Range: $V_{\min} = 0.0000, \quad V_{\max} = 10.0000$
- Number of atoms: $N = 3$
- Atom Step Size:
  $$\Delta z = \frac{V_{\max} - V_{\min}}{N - 1} = \frac{10.0000 - 0.0000}{3 - 1} = \frac{10.0000}{2} = \mathbf{5.0000}$$
- Atoms:
  $$z_0 = 0.0000, \quad z_1 = 5.0000, \quad z_2 = 10.0000$$

**Observed Transition:**
- Immediate reward: $r = 1.0000$
- Discount factor: $\gamma = 0.8000$
- Target distribution at next state $S'$ for greedy action $a^*$:
  $$\mathbf{p}' = [p_0' = 0.2000, \quad p_1' = 0.5000, \quad p_2' = 0.3000]$$

We will compute:
1. Shifted Bellman atom locations $\hat{T} z_j$
2. Linear projection fractions onto $\{z_0, z_1, z_2\}$
3. Final projected probability mass vector $\mathbf{m} = [m_0, m_1, m_2]$
4. Cross-entropy loss against a current uniform prediction $\mathbf{p} = [1/3, 1/3, 1/3]$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Walkthrough Concrete Value |
| :--- | :--- | :--- |
| $z_j$ | Fixed Support Atoms | $\{0.0000, 5.0000, 10.0000\}$ |
| $\Delta z$ | Support Grid Spacing | $5.0000$ |
| $p_j'$ | Next-State Target Probabilities | $p_0'=0.20, p_1'=0.50, p_2'=0.30$ |
| $\hat{T} z_j$ | Shifted Atom Locations | $r + \gamma z_j = 1.0 + 0.8 z_j$ |
| $b_j$ | Continuous Support Coordinate | $\frac{\hat{T} z_j - V_{\min}}{\Delta z}$ |
| $l, u$ | Lower and Upper Atom Indices | $l = \lfloor b_j \rfloor, u = \lceil b_j \rceil$ |
| $m_i$ | Projected Target Probability | Accumulator for atom $z_i$ |

---

### 5.3 Step-by-Step Hand Calculations: C51 Categorical Projection

#### Atom 0 ($z_0 = 0.0000, p_0' = 0.2000$):
1. Compute shifted location:
   $$\hat{T} z_0 = r + \gamma z_0 = 1.0000 + 0.8000 \times 0.0000 = \mathbf{1.0000}$$
2. Coordinate:
   $$b_0 = \frac{1.0000 - 0.0000}{5.0000} = \mathbf{0.2000}$$
   Lower index $l = \lfloor 0.2000 \rfloor = 0$, Upper index $u = \lceil 0.2000 \rceil = 1$.
3. Distribute probability mass $p_0' = 0.2000$:
   - Weight to atom $l = 0$: $u - b_0 = 1.0000 - 0.2000 = \mathbf{0.8000}$
     $$\Delta m_0 = p_0' \times (u - b_0) = 0.2000 \times 0.8000 = \mathbf{0.1600}$$
   - Weight to atom $u = 1$: $b_0 - l = 0.2000 - 0.0000 = \mathbf{0.2000}$
     $$\Delta m_1 = p_0' \times (b_0 - l) = 0.2000 \times 0.2000 = \mathbf{0.0400}$$

#### Atom 1 ($z_1 = 5.0000, p_1' = 0.5000$):
1. Compute shifted location:
   $$\hat{T} z_1 = r + \gamma z_1 = 1.0000 + 0.8000 \times 5.0000 = 1.0000 + 4.0000 = \mathbf{5.0000}$$
2. Coordinate:
   $$b_1 = \frac{5.0000 - 0.0000}{5.0000} = \mathbf{1.0000}$$
   $l = 1, u = 1$ (exact integer match!).
3. Distribute probability mass $p_1' = 0.5000$:
   - All mass lands directly on atom $1$:
     $$\Delta m_1 = p_1' = \mathbf{0.5000}$$

#### Atom 2 ($z_2 = 10.0000, p_2' = 0.3000$):
1. Compute shifted location:
   $$\hat{T} z_2 = r + \gamma z_2 = 1.0000 + 0.8000 \times 10.0000 = 1.0000 + 8.0000 = \mathbf{9.0000}$$
2. Coordinate:
   $$b_2 = \frac{9.0000 - 0.0000}{5.0000} = \mathbf{1.8000}$$
   Lower index $l = \lfloor 1.8000 \rfloor = 1$, Upper index $u = \lceil 1.8000 \rceil = 2$.
3. Distribute probability mass $p_2' = 0.3000$:
   - Weight to atom $l = 1$: $u - b_2 = 2.0000 - 1.8000 = \mathbf{0.2000}$
     $$\Delta m_1 = p_2' \times (u - b_2) = 0.3000 \times 0.2000 = \mathbf{0.0600}$$
   - Weight to atom $u = 2$: $b_2 - l = 1.8000 - 1.0000 = \mathbf{0.8000}$
     $$\Delta m_2 = p_2' \times (b_2 - l) = 0.3000 \times 0.8000 = \mathbf{0.2400}$$

---

### 5.4 Summing Total Projected Distribution $\mathbf{m}$

Accumulate total mass into each bucket:
- **Atom $z_0 = 0.0000$:**
  $$m_0 = \Delta m_0 (\text{from atom 0}) = \mathbf{0.1600}$$
- **Atom $z_1 = 5.0000$:**
  $$m_1 = \Delta m_1 (\text{from atom 0}) + \Delta m_1 (\text{from atom 1}) + \Delta m_1 (\text{from atom 2})$$
  $$m_1 = 0.0400 + 0.5000 + 0.0600 = \mathbf{0.6000}$$
- **Atom $z_2 = 10.0000$:**
  $$m_2 = \Delta m_2 (\text{from atom 2}) = \mathbf{0.2400}$$

**Final Projected Target Distribution Vector:**
$$\mathbf{m} = \begin{bmatrix} 0.1600 \\ 0.6000 \\ 0.2400 \end{bmatrix}$$

**Probability Mass Conservation Check:**
$$\sum_{i=0}^2 m_i = 0.1600 + 0.6000 + 0.2400 = \mathbf{1.0000} \quad \checkmark$$

**Expected Return of Projected Distribution:**
$$\mathbb{E}[\mathbf{m}] = \sum_{i=0}^2 z_i m_i = (0.0 \times 0.16) + (5.0 \times 0.60) + (10.0 \times 0.24) = 0.0 + 3.00 + 2.40 = \mathbf{5.4000}$$
Notice that this matches the scalar Bellman expectation:
$$r + \gamma \mathbb{E}[\mathbf{p}'] = 1.0 + 0.8 \times (0.0 \times 0.2 + 5.0 \times 0.5 + 10.0 \times 0.3) = 1.0 + 0.8 \times (0.0 + 2.5 + 3.0) = 1.0 + 0.8(5.5) = 1.0 + 4.4 = \mathbf{5.4000}!$$
The Cramer projection preserves the **exact mean**!

---

### 5.5 Visual Grid: Projection Matrix Walkthrough

| Atom $j$ | $z_j$ | $p'_j$ | Shifted $\hat{T} z_j$ | Coord $b_j$ | Enclosing Atoms | Mass to $z_0$ | Mass to $z_1$ | Mass to $z_2$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $0.0$ | $0.2000$ | $1.0000$ | $0.2000$ | $z_0, z_1$ | $\mathbf{0.1600}$ | $\mathbf{0.0400}$ | $0.0000$ |
| **1** | $5.0$ | $0.5000$ | $5.0000$ | $1.0000$ | $z_1$ | $0.0000$ | $\mathbf{0.5000}$ | $0.0000$ |
| **2** | $10.0$ | $0.3000$ | $9.0000$ | $1.8000$ | $z_1, z_2$ | $0.0000$ | $\mathbf{0.0600}$ | $\mathbf{0.2400}$ |
| **Total** | - | $1.0000$ | - | - | - | **$m_0 = 0.1600$** | **$m_1 = 0.6000$** | **$m_2 = 0.2400$** |

---

### 5.6 Step 5: Cross-Entropy Loss Calculation

Suppose the current online network predicts a uniform distribution:
$$\mathbf{p} = \begin{bmatrix} 1/3 \\ 1/3 \\ 1/3 \end{bmatrix} \approx \begin{bmatrix} 0.3333 \\ 0.3333 \\ 0.3333 \end{bmatrix}$$

The Cross-Entropy Loss is:
$$\mathcal{L} = - \sum_{i=0}^2 m_i \ln(p_i) = - \left[ 0.1600 \ln(1/3) + 0.6000 \ln(1/3) + 0.2400 \ln(1/3) \right]$$
$$= - \ln(1/3) \sum_{i=0}^2 m_i = \ln(3) \times 1.0000 \approx \mathbf{1.0986}$$

---

## 6. Solved Illustrations

### Illustration 1: Preserving Bimodality vs. Mean Collapse
**Problem:**
Consider a lottery transition: with probability $0.5$, reward is $+10$; with probability $0.5$, reward is $-10$. Assume $\gamma = 0$.
Compare the target representation learned by:
1. Standard DQN
2. C51 Distributional DQN
**Solution:**
- **Standard DQN:**
  Computes the scalar mean:
  $$Y = 0.5(+10) + 0.5(-10) = \mathbf{0.00}$$
  The network learns that the state has value $0$, giving the illusion that the state is neutral and risk-free.
- **C51 Distributional DQN:**
  Learns a bimodal distribution placing probability mass $0.5$ at atom $+10$ and $0.5$ at atom $-10$. The variance $\sigma^2 = 100$ is completely preserved, allowing risk-sensitive downstream agents to avoid this state if variance-averse! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

Distributional RL is now a core pillar of production-grade RL systems:
1. **Atari 57 Super-Human Agents (Agent57 - Badia et al., Nature 2020):** Used transformed Bellman operators with QR-DQN.
2. **Autonomous Driving & Robotics:** Waymo and autonomous drone controllers use distributional RL to evaluate **Value at Risk (VaR)**, explicitly planning trajectories that keep the 99th percentile catastrophic collision probability below safety thresholds.
3. **DeepSeek-Math & Reasoning:** Value baselines in reasoning LLMs leverage distributional returns to distinguish consistent moderate chain-of-thought solutions from high-variance fluke tokens.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of the Part 5 C51 Cramer projection:
   - Target distribution $\mathbf{m} = [0.1600, 0.6000, 0.2400]$
   - Conservation of mass: $\sum m_i = 1.0000$
   - Exact mean conservation: $\mathbb{E}[\mathbf{m}] = 5.4000$
   - Cross-entropy loss matching machine precision to $< 10^{-14}$.
2. Quantile Regression loss function (Pinball Huber loss) verification for QR-DQN.
3. Bimodal vs. unimodal value distribution benchmark.

See implementation in:
[`11_reinforcement_learning/code/12_distributional_reinforcement_learning.py`](./code/12_distributional_reinforcement_learning.py)
