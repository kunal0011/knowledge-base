# Module 11.9: Function Approximation & The Deadly Triad

---

## 1. Intuition & 101 Motivation

In real-world applications of reinforcement learning, the state space $\mathcal{S}$ is vastly too large to store in a lookup table:
- Backgammon has $\approx 10^{20}$ states.
- Chess has $\approx 10^{40}$ states.
- Go has $\approx 10^{170}$ states.
- A single $84 \times 84$ Atari grayscale frame with 256 pixel shades has $256^{7056} \approx 10^{16988}$ states.
- Continuous robotic control (joint angles, velocities) has an **uncountably infinite** state space $\mathcal{S} \subseteq \mathbb{R}^d$.

To scale RL, we must replace lookup tables with **parameterized function approximation**:
$$\hat{v}(s, \mathbf{w}) \approx V^\pi(s) \quad \text{or} \quad \hat{q}(s, a, \mathbf{w}) \approx Q^*(s, a)$$
where $\mathbf{w} \in \mathbb{R}^d$ is a weight vector whose dimension is vastly smaller than the state space ($d \ll |\mathcal{S}|$).

However, combining function approximation with reinforcement learning introduces one of the deepest theoretical perils in machine learning: **The Deadly Triad**.

```
                           THE DEADLY TRIAD
                           
                     Function Approximation
                     (Neural Nets, Linear features)
                                / \
                               /   \
                              /     \
                             /       \
                            /         \
              Bootstrapping ----------- Off-Policy Learning
            (TD targets, DP)           (Replay buffer, Q-learning)

   Any TWO elements: STABLE & CONVERGENT.
   All THREE combined: POTENTIALLY CATASTROPHIC DIVERGENCE (Weights -> Infinity).
```

When all three conditions are present, value estimates can blow up to $\pm \infty$ even on simple linear problems with zero rewards!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Mean Squared Value Error (MSVE)

Let $d^\pi(s)$ be the on-policy stationary distribution over states under policy $\pi$, such that $\sum_{s \in \mathcal{S}} d^\pi(s) = 1$ and $d^\pi(s) \ge 0$.
The quality of approximation is measured by the **Mean Squared Value Error (MSVE)**:

$$\overline{\text{VE}}(\mathbf{w}) \triangleq \sum_{s \in \mathcal{S}} d^\pi(s) \left[ V^\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2 = \left\| \mathbf{V}^\pi - \hat{\mathbf{v}}(\mathbf{w}) \right\|_{\mathbf{D}}^2$$

where $\mathbf{D} = \operatorname{diag}(d^\pi(s_1), \dots, d^\pi(s_n))$ defines a weighted Euclidean norm:
$$\|\mathbf{x}\|_{\mathbf{D}}^2 \triangleq \mathbf{x}^\top \mathbf{D} \mathbf{x}$$

If we had access to the true target $V^\pi(S_t)$, stochastic gradient descent (SGD) on $\overline{\text{VE}}(\mathbf{w})$ would yield the ideal update:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{1}{2} \alpha \nabla_\mathbf{w} \left[ V^\pi(S_t) - \hat{v}(S_t, \mathbf{w}) \right]^2 = \mathbf{w}_t + \alpha \left[ V^\pi(S_t) - \hat{v}(S_t, \mathbf{w}) \right] \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})$$

---

### 2.2 Semi-Gradient Methods: Why TD is NOT True Gradient Descent

In reality, the true value $V^\pi(S_t)$ is unknown. In TD learning, we substitute a bootstrapped target $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$.
The resulting update is:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w}_t)$$

> [!WARNING]
> **Semi-Gradient Warning:** This update is called a **semi-gradient method** because it ignores the dependence of the target $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$ on the parameter vector $\mathbf{w}$!
> If one attempted true gradient descent on the Bellman residual squared error $\frac{1}{2} (R + \gamma \hat{v}' - \hat{v})^2$, the gradient would be:
> $$\nabla_\mathbf{w} \frac{1}{2} \delta_t^2 = -\delta_t \left( \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w}) - \gamma \nabla_\mathbf{w} \hat{v}(S_{t+1}, \mathbf{w}) \right)$$
> True Bellman error minimization is known to converge to poor, non-optimal value functions (Baird's residual gradient dilemma). Semi-gradient methods converge to the Bellman fixed point, but **only under on-policy sampling**!

---

### 2.3 Linear Function Approximation & The Projection Operator $\Pi$

Consider a linear function approximator:
$$\hat{v}(s, \mathbf{w}) \triangleq \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)$$
where $\mathbf{x}(s) \in \mathbb{R}^d$ is the feature vector for state $s$. In matrix notation:
$$\hat{\mathbf{v}}(\mathbf{w}) = \mathbf{\Phi} \mathbf{w}$$
where $\mathbf{\Phi} \in \mathbb{R}^{|\mathcal{S}| \times d}$ is the feature matrix whose rows are $\mathbf{x}(s)^\top$.

#### The Projection Operator $\Pi$
Any vector $\mathbf{V} \in \mathbb{R}^{|\mathcal{S}|}$ can be orthogonally projected onto the column subspace of $\mathbf{\Phi}$ with respect to the weighted norm $\|\cdot\|_{\mathbf{D}}$:

$$\Pi \mathbf{V} \triangleq \mathbf{\Phi} \left( \mathbf{\Phi}^\top \mathbf{D} \mathbf{\Phi} \right)^{-1} \mathbf{\Phi}^\top \mathbf{D} \mathbf{V}$$

Notice that $\Pi$ is linear, idempotent ($\Pi^2 = \Pi$), and a non-expansion in the $\mathbf{D}$-norm:
$$\|\Pi \mathbf{V}_1 - \Pi \mathbf{V}_2\|_{\mathbf{D}} \le \|\mathbf{V}_1 - \mathbf{V}_2\|_{\mathbf{D}}$$

---

### 2.4 The Projected Bellman Equation & The TD Fixed Point

When linear semi-gradient TD(0) converges, its expected update vector must equal zero:
$$\mathbb{E} \left[ \mathbf{x}_t \left( R_{t+1} + \gamma \mathbf{x}_{t+1}^\top \mathbf{w} - \mathbf{x}_t^\top \mathbf{w} \right) \right] = \mathbf{0}$$

Expanding the expectation across the stationary distribution $\mathbf{D}$:
$$\mathbf{\Phi}^\top \mathbf{D} \left( \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{\Phi} \mathbf{w} - \mathbf{\Phi} \mathbf{w} \right) = \mathbf{0}$$
$$\underbrace{\mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{\Phi}}_{\mathbf{A}} \mathbf{w} = \underbrace{\mathbf{\Phi}^\top \mathbf{D} \mathbf{R}^\pi}_{\mathbf{b}}$$

This is equivalent to the **Projected Bellman Equation**:
$$\mathbf{\Phi} \mathbf{w}_{\text{TD}} = \Pi \mathcal{T}^\pi (\mathbf{\Phi} \mathbf{w}_{\text{TD}})$$

#### Theorem: Convergence of Linear On-Policy TD(0) (Tsitsiklis & Van Roy, 1997)
If data is sampled from the on-policy stationary distribution $d^\pi(s)$ of an ergodic Markov chain, the matrix $\mathbf{A} = \mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{\Phi}$ is **strictly positive definite**.
The compound operator $\Pi \mathcal{T}^\pi$ is a $\gamma$-contraction in $\|\cdot\|_{\mathbf{D}}$:
$$\|\Pi \mathcal{T}^\pi \mathbf{V}_1 - \Pi \mathcal{T}^\pi \mathbf{V}_2\|_{\mathbf{D}} \le \gamma \|\mathbf{V}_1 - \mathbf{V}_2\|_{\mathbf{D}}$$
Consequently, linear on-policy TD(0) converges almost surely to the unique fixed point $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1} \mathbf{b}$, and the resulting approximation error satisfies:
$$\|\mathbf{V}^\pi - \mathbf{\Phi} \mathbf{w}_{\text{TD}}\|_{\mathbf{D}} \le \frac{1}{\sqrt{1 - \gamma^2}} \min_\mathbf{w} \|\mathbf{V}^\pi - \mathbf{\Phi} \mathbf{w}\|_{\mathbf{D}}$$

---

### 2.5 The Deadly Triad & Off-Policy Divergence Proof

When sampling transitions **off-policy** from behavior distribution $\mu(s) \neq d^\pi(s)$, the weighting matrix becomes $\mathbf{M} = \operatorname{diag}(\mu(s))$.
The projection operator becomes $\Pi_\mu = \mathbf{\Phi}(\mathbf{\Phi}^\top \mathbf{M} \mathbf{\Phi})^{-1}\mathbf{\Phi}^\top \mathbf{M}$.

> [!CAUTION]
> **The Collapse of Contraction:** While $\mathcal{T}^\pi$ is a contraction in $\|\cdot\|_\infty$, it is NOT necessarily a contraction in $\|\cdot\|_\mathbf{M}$!
> The combined operator $\Pi_\mu \mathcal{T}^\pi$ can have spectral radius (maximum eigenvalue) **strictly greater than 1**:
> $$\rho(\Pi_\mu \mathcal{T}^\pi) > 1$$
> When this occurs, repeatedly applying the semi-gradient TD operator causes the weight vector $\mathbf{w}_t$ to expand exponentially to infinity:
> $$\lim_{t \to \infty} \|\mathbf{w}_t\| = +\infty$$

---

## 3. Geometric Interpretation: Warping of the Projection Manifold

In the $|\mathcal{S}|$-dimensional state space:
- The realizable function approximations span a low-dimensional flat hyperplane $\mathcal{H} = \operatorname{span}(\mathbf{\Phi})$.
- When sampling **on-policy**, the projection angle is orthogonal with respect to the stationary metric $\mathbf{D}$. The contractive pull of $\mathcal{T}^\pi$ ($\gamma < 1$) overcomes the non-expansion of $\Pi$, ensuring the composition $\Pi \mathcal{T}^\pi$ pulls any point inside towards the unique fixed point.
- When sampling **off-policy**, the metric tensor is distorted by $\mathbf{M} \neq \mathbf{D}$. The projection operator $\Pi_\mu$ acts at an **oblique, acute angle** relative to the Bellman operator flow. The oblique projection amplifies the vector magnitude by more than $1/\gamma$, causing successive iterates to spiral outward unboundedly!

```
                  V-Space (R^|S|)
                       ^
                       |              /  Oblique Projection (Off-policy)
                       |    T(V)     /   amplifies vector magnitude!
                       |      * ----+--> Pi_mu(T(V))  [||Pi T|| > 1]
                       |     /     /
                       |    /     /
                       |   * V   /    Hyperplane spanned by Phi w
                       |  /     /    /
                       | /     /    /
                       +------+----+------------------->
                              Orthogonal Projection (On-policy)
                              ||Pi T|| <= gamma < 1 (Stable)
```

---

## 4. Real-World Analogy: The Acoustic Feedback Loop (Larsen Effect)

Think of the Deadly Triad as a **screeching PA system at a rock concert**:
1. **Bootstrapping (The Microphone):** The microphone picks up sound and feeds it back into the audio processor.
2. **Function Approximation (The Graphic Equalizer / Amplifier):** The amplifier generalizes and boosts certain frequency bands with high gain.
3. **Off-Policy Operation (Aiming the Mic at the Speaker):** Instead of pointing the microphone at the singer (on-policy target distribution), you point it directly at the monitor speaker (off-policy mismatch).

If any one component is absent:
- No microphone (no bootstrapping $\implies$ Monte Carlo): Sound plays cleanly.
- No amplifier (lookup table): Gain is $\le 1$, no feedback loop.
- Mic pointed at singer (on-policy): Gain is stable and contained.

Combine all three, and an infinitesimal whisper creates an explosive, ear-splitting screech that blows the speakers ($\|\mathbf{w}\| \to \infty$)!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: Tsitsiklis & Van Roy 2-State Divergence Example
We consider the simplest possible system that diverges:
- **Two States:** $S_1$ and $S_2$.
- **Single scalar parameter:** $w \in \mathbb{R}$ ($d = 1$).
- **Features:** $\phi(S_1) = 1.0000, \quad \phi(S_2) = 2.0000$.
  So value approximations are:
  $$\hat{v}(S_1, w) = 1.0 \times w = w, \quad \hat{v}(S_2, w) = 2.0 \times w = 2w$$
- **Transition Dynamics:** From $S_1$, the agent **always** transitions to $S_2$ under target policy $\pi$.
- **Reward:** $R = 0.0000$ everywhere.
- **Discount factor:** $\gamma = 0.9000$.
- **Off-Policy Sampling Distribution:** The behavior policy samples transitions **only from state $S_1$** (state $S_2$ is never sampled as a starting state, so $\mu(S_1) = 1.0, \mu(S_2) = 0.0$).
- **Learning rate:** $\alpha = 0.5000$.
- **Initial weight:** $w_0 = 1.0000$.

Let us compute the semi-gradient update by hand step-by-step!

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Value / Formula |
| :--- | :--- | :--- |
| $S_t$ | Sampled Starting State | Always $S_1$ ($\mu(S_1) = 1.0$) |
| $S_{t+1}$ | Successor State under $\pi$ | Always $S_2$ |
| $R_{t+1}$ | Reward | $0.0000$ |
| $\gamma$ | Discount Factor | $0.9000$ |
| $\phi(S_t)$ | Feature at Current State | $\phi(S_1) = 1.0000$ |
| $\phi(S_{t+1})$ | Feature at Next State | $\phi(S_2) = 2.0000$ |
| $\hat{v}(S_t, w)$ | Current Estimated Value | $w \times 1.0000 = w$ |
| $\hat{v}(S_{t+1}, w)$ | Bootstrapped Successor Value | $w \times 2.0000 = 2w$ |
| Target | Bootstrapped Target | $R + \gamma \hat{v}(S_2, w) = 0 + 0.9(2w) = \mathbf{1.8w}$ |
| TD Error $\delta$ | Target minus Estimate | $1.8w - 1.0w = \mathbf{0.8w}$ |
| $\nabla_w \hat{v}$ | Gradient with respect to $w$ | $\phi(S_1) = 1.0000$ |
| $w_{\text{new}}$ | Updated Weight | $w + \alpha \delta \nabla_w \hat{v} = w + 0.5(0.8w)(1.0) = \mathbf{1.4w}$ |

---

### 5.3 Step-by-Step Hand Calculations: Iteration 1 to 5

At every step $t$, the update equation is:
$$w_{t+1} = w_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_2, w_t) - \hat{v}(S_1, w_t) \right] \phi(S_1)$$
$$= w_t + 0.5000 \left[ 0.0 + 0.9000(2.0000 w_t) - 1.0000 w_t \right] (1.0000)$$
$$= w_t + 0.5000 \left[ 1.8000 w_t - 1.0000 w_t \right]$$
$$= w_t + 0.5000 \left[ 0.8000 w_t \right]$$
$$= w_t + 0.4000 w_t = \mathbf{1.4000 w_t}$$

Notice: At every single update, the weight is multiplied by **$1.4000 > 1$**!
Let us trace the numbers starting from $w_0 = 1.0000$:

#### Iteration 1:
- $\text{Target} = 0 + 0.9(2 \times 1.0000) = \mathbf{1.8000}$
- $\delta_0 = 1.8000 - 1.0000 = \mathbf{0.8000}$
- $w_1 = 1.0000 + 0.5000 \times 0.8000 \times 1.0 = 1.0000 + 0.4000 = \mathbf{1.4000}$

#### Iteration 2:
- Current $w_1 = 1.4000$
- $\text{Target} = 0.9(2 \times 1.4000) = 0.9 \times 2.8000 = \mathbf{2.5200}$
- $\delta_1 = 2.5200 - (1 \times 1.4000) = 2.5200 - 1.4000 = \mathbf{1.1200}$
- $w_2 = 1.4000 + 0.5000 \times 1.1200 \times 1.0 = 1.4000 + 0.5600 = \mathbf{1.9600}$ ($= 1.4^2$)

#### Iteration 3:
- Current $w_2 = 1.9600$
- $\text{Target} = 0.9(2 \times 1.9600) = \mathbf{3.5280}$
- $\delta_2 = 3.5280 - 1.9600 = \mathbf{1.5680}$
- $w_3 = 1.9600 + 0.5000 \times 1.5680 = 1.9600 + 0.7840 = \mathbf{2.7440}$ ($= 1.4^3$)

#### Iteration 4:
- Current $w_3 = 2.7440$
- $\text{Target} = 0.9(2 \times 2.7440) = \mathbf{4.9392}$
- $\delta_3 = 4.9392 - 2.7440 = \mathbf{2.1952}$
- $w_4 = 2.7440 + 0.5000 \times 2.1952 = 2.7440 + 1.0976 = \mathbf{3.8416}$ ($= 1.4^4$)

#### Iteration 5:
- Current $w_4 = 3.8416$
- $\text{Target} = 0.9(2 \times 3.8416) = \mathbf{6.91488}$
- $\delta_4 = 6.91488 - 3.8416 = \mathbf{3.07328}$
- $w_5 = 3.8416 + 0.5000 \times 3.07328 = 3.8416 + 1.53664 = \mathbf{5.37824}$ ($= 1.4^5$)

---

### 5.4 Summary Arithmetic Progression Table

| Step $t$ | Current Weight $w_t$ | Target $0.9(2w_t)$ | TD Error $\delta_t$ | Update $\alpha \delta_t$ | New Weight $w_{t+1}$ | Exact Analytical $(1.4)^t$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 0$** | $1.0000$ | $1.8000$ | $0.8000$ | $+0.4000$ | $\mathbf{1.4000}$ | $1.4000$ |
| **$t = 1$** | $1.4000$ | $2.5200$ | $1.1200$ | $+0.5600$ | $\mathbf{1.9600}$ | $1.9600$ |
| **$t = 2$** | $1.9600$ | $3.5280$ | $1.5680$ | $+0.7840$ | $\mathbf{2.7440}$ | $2.7440$ |
| **$t = 3$** | $2.7440$ | $4.9392$ | $2.1952$ | $+1.0976$ | $\mathbf{3.8416}$ | $3.8416$ |
| **$t = 4$** | $3.8416$ | $6.91488$ | $3.07328$ | $+1.53664$ | $\mathbf{5.37824}$ | $5.37824$ |
| **$t = 10$** | $28.9255$ | - | - | - | $\mathbf{40.4957}$ | $40.4957$ |
| **$t = 50$** | - | - | - | - | - | $\mathbf{2.22 \times 10^7}$ |
| **$t \to \infty$** | - | - | - | - | - | $\mathbf{+\infty}$ (Explosion!) |

The system exhibits exponential divergence with Lyapunov exponent $\ln(1.4) \approx +0.3365$!

---

## 6. Solved Illustrations

### Illustration 1: Baird's Counterexample (The Canonical Benchmark)
**Problem:**
Consider Baird's 7-state MDP with 2 actions (dashed and solid).
The parameter vector has dimension $d = 14$ (or $d = 7$).
All rewards are identically $0$. The true value function is everywhere $0$.
Prove why standard Q-learning with linear function approximation diverges to infinity when trained with uniform behavior policy.

**Solution:**
In Baird's system, states 1 through 6 transition to state 7 under the solid action.
The feature representation assigns:
$$\mathbf{x}(s) = [0, \dots, 2, \dots, 1]^\top$$
The projection operator under the uniform behavior distribution produces a matrix $\mathbf{A} = \mathbf{\Phi}^\top \mathbf{D} (\mathbf{I} - \gamma \mathbf{P}) \mathbf{\Phi}$ that has eigenvalues with **strictly negative real parts** (in the update equation $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \mathbf{A} \mathbf{w}_t$).
Because the real parts of eigenvalues are negative, the linear dynamical system is unstable:
$$\mathbf{w}_{t+1} = (\mathbf{I} - \alpha \mathbf{A}) \mathbf{w}_t \implies \|\mathbf{w}_t\| \to \infty \quad \blacksquare$$

---

## 7. Deep Learning Connection & Application

### How Deep Q-Networks (DQN) Tamed the Deadly Triad
Deep Q-Networks (Mnih et al., 2015) successfully combined all three elements of the Deadly Triad (Neural Networks + Bootstrapping + Off-Policy Replay) through two breakthrough engineering innovations:

1. **Target Network $\theta^-$ (Quenching the Bootstrapping Loop):**
   Instead of using the rapidly changing online parameters $\theta$ in the target, DQN computes targets with frozen parameters $\theta^-$ updated only every $C = 10,000$ steps:
   $$Y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$
   This breaks the instantaneous self-amplifying feedback loop $\theta \leftarrow \theta$, turning the loss temporarily into standard supervised regression!

2. **Experience Replay Buffer $\mathcal{D}$ (Bridging Off-Policy Drift):**
   Sampling uniformly from a rolling buffer of 1,000,000 transitions decorrelates consecutive training examples and stabilizes the empirical state visitation distribution $\mu(s)$.

3. **Gradient TD Methods (GTD2 & TDC - Sutton et al., 2009):**
   For linear models, Gradient TD methods provide true $O(d)$ stochastic gradient descent on the Projected Bellman Error objective $\text{PBE}(\mathbf{w})$, guaranteeing unconditional convergence even off-policy!

---

## 8. Code Implementation & Verification

The accompanying Python script verifies:
1. Exact numerical reproduction of Part 5 hand calculations ($w_1 = 1.4000, w_2 = 1.9600, w_3 = 2.7440, w_4 = 3.8416, w_5 = 5.37824$) matching machine precision to $< 10^{-14}$.
2. Linear Semi-Gradient TD(0) on an on-policy Random Walk: verifying stable convergence to the analytical projected fixed point $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1} \mathbf{b}$.
3. Baird's Counterexample: Simulating off-policy semi-gradient Q-learning and observing the weight vector norm $\|\mathbf{w}_t\|$ diverging exponentially toward infinity.

See implementation in:
[`11_reinforcement_learning/code/09_function_approximation_and_deadly_triad.py`](./code/09_function_approximation_and_deadly_triad.py)
