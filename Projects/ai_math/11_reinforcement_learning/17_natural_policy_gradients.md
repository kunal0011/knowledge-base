# Module 11.17: Natural Policy Gradients & Information Geometry

---

## 1. Intuition & 101 Motivation

In standard policy gradient methods (REINFORCE, A2C), parameters are updated along the direction of steepest ascent in flat Euclidean parameter space:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$

This update is the solution to a constrained optimization problem bounded by Euclidean distance:
$$\max_{\Delta \boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}} J^\top \Delta \boldsymbol{\theta} \quad \text{subject to} \quad \|\Delta \boldsymbol{\theta}\|_2^2 \le \epsilon$$

However, measuring step size in Euclidean parameter space $\|\Delta \boldsymbol{\theta}\|_2$ is fundamentally flawed:
1. **Parameter Coordinates are Arbitrary:** Changing the units of a neural network's weights or re-parameterizing a layer (e.g., using log-variance instead of standard deviation) changes the Euclidean norm without altering the underlying policy at all!
2. **The "Cliff of Catastrophic Forgetting":** In probability distribution space, a tiny parameter change $\|\Delta \boldsymbol{\theta}\|_2 = 0.01$ might alter action probabilities by a minuscule $0.1\%$ in flat regions, but could alter action probabilities by $50\%$ in saturated regions, causing policy collapse.

**Natural Policy Gradient (Kakade, 2002)** replaces the arbitrary Euclidean metric with **Information Geometry** (Amari, 1998). Instead of constraining parameter distance $\|\Delta \boldsymbol{\theta}\|_2^2$, it constrains the **Kullback-Leibler (KL) divergence** between the old and new policies:
$$D_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}) \le \epsilon$$

The natural gradient moves along the **Riemannian manifold of probability distributions**, where the metric tensor is the **Fisher Information Matrix (FIM)**.

```
       FLAT EUCLIDEAN PARAMETER SPACE             CURVED PROBABILITY MANIFOLD (FIM)
               Delta theta                                KL(pi_old || pi_new)
                   ^                                                ^
                   |                                                |
            *------+------*                                  *~~~~~~+~~~~~~*
            |             |                                 (               )
            |      *      |                                 (       *       )
            |             |                                 (               )
            *------+------*                                  *~~~~~~+~~~~~~*
                   |                                                |
          Arbitrary coordinate box                         True distance in behavior!
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Second-Order Taylor Expansion of KL Divergence

Let $\pi_{\boldsymbol{\theta}}(a \mid s)$ be a parameterized policy.
The expected KL divergence between $\pi_{\boldsymbol{\theta}}$ and $\pi_{\boldsymbol{\theta} + \mathbf{d}\boldsymbol{\theta}}$ across state visitation distribution $d^\pi(s)$ is:
$$\bar{D}_{\text{KL}}(\boldsymbol{\theta} \parallel \boldsymbol{\theta} + \mathbf{d}\boldsymbol{\theta}) \triangleq \mathbb{E}_{s \sim d^\pi} \left[ D_{\text{KL}}(\pi_{\boldsymbol{\theta}}(\cdot \mid s) \parallel \pi_{\boldsymbol{\theta} + \mathbf{d}\boldsymbol{\theta}}(\cdot \mid s)) \right]$$

Perform a Taylor expansion of $\bar{D}_{\text{KL}}(\boldsymbol{\theta} \parallel \boldsymbol{\theta} + \mathbf{d}\boldsymbol{\theta})$ about $\mathbf{d}\boldsymbol{\theta} = \mathbf{0}$:
1. **Zeroth-order term:**
   $$\bar{D}_{\text{KL}}(\boldsymbol{\theta} \parallel \boldsymbol{\theta}) = 0$$
2. **First-order term:**
   Because $\bar{D}_{\text{KL}} \ge 0$ for all distributions, $\mathbf{d}\boldsymbol{\theta} = \mathbf{0}$ is a global minimum. Thus, the first derivative vanishes:
   $$\left. \nabla_{\mathbf{d}\boldsymbol{\theta}} \bar{D}_{\text{KL}}(\boldsymbol{\theta} \parallel \boldsymbol{\theta} + \mathbf{d}\boldsymbol{\theta}) \right|_{\mathbf{d}\boldsymbol{\theta} = \mathbf{0}} = \mathbf{0}$$
3. **Second-order term (Hessian):**
   The Hessian of the KL divergence at $\mathbf{d}\boldsymbol{\theta} = \mathbf{0}$ is the **Fisher Information Matrix** $\mathbf{F}(\boldsymbol{\theta})$:

$$\bar{D}_{\text{KL}}(\boldsymbol{\theta} \parallel \boldsymbol{\theta} + \mathbf{d}\boldsymbol{\theta}) = \frac{1}{2} \mathbf{d}\boldsymbol{\theta}^\top \mathbf{F}(\boldsymbol{\theta}) \mathbf{d}\boldsymbol{\theta} + \mathcal{O}(\|\mathbf{d}\boldsymbol{\theta}\|_2^3)$$

---

### 2.2 The Fisher Information Matrix (FIM)

#### Definition:
$$\mathbf{F}(\boldsymbol{\theta}) \triangleq \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)^\top \right]$$

Equivalently, by the information equality (under mild regularity conditions):
$$\mathbf{F}(\boldsymbol{\theta}) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_{\boldsymbol{\theta}}} \left[ - \nabla_{\boldsymbol{\theta}}^2 \log \pi_{\boldsymbol{\theta}}(a \mid s) \right]$$

#### Mathematical Properties of $\mathbf{F}$:
1. $\mathbf{F}(\boldsymbol{\theta}) \in \mathbb{R}^{d \times d}$ is **symmetric** and **positive semi-definite** ($\mathbf{x}^\top \mathbf{F} \mathbf{x} \ge 0$).
2. It acts as the **Riemannian Metric Tensor** on the statistical manifold of policies.
3. It defines the local distance metric in distribution space:
   $$ds^2 = \mathbf{d}\boldsymbol{\theta}^\top \mathbf{F}(\boldsymbol{\theta}) \mathbf{d}\boldsymbol{\theta}$$

---

### 2.3 The Constrained Optimization Problem

The Natural Policy Gradient seeks the step $\Delta \boldsymbol{\theta}$ that maximizes expected return subject to a strict bound on policy divergence:

$$\max_{\Delta \boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} \quad \text{subject to} \quad \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta} \le \epsilon$$
where $\mathbf{g} \triangleq \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ is the standard vanilla policy gradient.

#### Derivation via Lagrangian Duality:
Form the Lagrangian with dual multiplier $\lambda \ge 0$:
$$\mathcal{L}(\Delta \boldsymbol{\theta}, \lambda) = \mathbf{g}^\top \Delta \boldsymbol{\theta} - \lambda \left( \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} - \epsilon \right)$$

Set the derivative with respect to $\Delta \boldsymbol{\theta}$ to zero:
$$\nabla_{\Delta \boldsymbol{\theta}} \mathcal{L} = \mathbf{g} - \lambda \mathbf{F} \Delta \boldsymbol{\theta} = \mathbf{0}$$
$$\mathbf{F} \Delta \boldsymbol{\theta} = \frac{1}{\lambda} \mathbf{g} \implies \Delta \boldsymbol{\theta}^* = \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g}$$

Substitute $\Delta \boldsymbol{\theta}^*$ into the boundary constraint $\frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} = \epsilon$:
$$\frac{1}{2} \left( \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g} \right)^\top \mathbf{F} \left( \frac{1}{\lambda} \mathbf{F}^{-1} \mathbf{g} \right) = \epsilon$$
$$\frac{1}{2 \lambda^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{F} \mathbf{F}^{-1} \mathbf{g} = \epsilon \implies \frac{1}{2 \lambda^2} \mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g} = \epsilon$$
$$\lambda^* = \sqrt{\frac{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}{2 \epsilon}}$$

Substitute $\lambda^*$ back into the optimal step:
$$\Delta \boldsymbol{\theta}^* = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}$$

#### The Natural Gradient Direction:
$$\tilde{\mathbf{g}} \triangleq \mathbf{F}(\boldsymbol{\theta})^{-1} \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$
The Natural Policy Gradient update step scales $\tilde{\mathbf{g}}$ to exactly satisfy the trust region $\epsilon$:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}$$

---

### 2.4 Theorem: Invariance to Re-Parameterization (Kakade, 2002)

Let $\boldsymbol{\psi} = f(\boldsymbol{\theta})$ be an arbitrary smooth, invertible change of coordinates.
Then the natural policy gradient update in $\boldsymbol{\psi}$-space produces the **exact same change in probability distributions** as the natural policy gradient update in $\boldsymbol{\theta}$-space:
$$\pi_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}(a \mid s) = \pi_{\boldsymbol{\psi} + \Delta \boldsymbol{\psi}}(a \mid s) + \mathcal{O}(\epsilon^2)$$
The natural gradient is a **geometric invariant** of the manifold, completely independent of how the neural network weights are parameterized!

---

## 3. Geometric Interpretation: Mercator Distortion vs. Geodesic Distance

- In flat Euclidean gradient descent, the gradient $\nabla_{\boldsymbol{\theta}} J$ is orthogonal to the contour lines on a paper map. But on a curved surface (like a Mercator projection of Earth), walking 1 inch on the map near the equator covers 2,000 miles, whereas walking 1 inch near the poles covers 200 miles.
- The Fisher Information Matrix $\mathbf{F}$ is the **metric tensor of the globe**: it converts paper map coordinates into true geodesic physical distances on Earth.
- Multiplying by $\mathbf{F}^{-1}$ rotates and stretches the gradient vector so that the agent takes uniform, isotropic steps across the actual probability manifold!

```
                    EUCLIDEAN vs NATURAL GRADIENT FLOW
                                                                    
         Contour of J(theta)                                        
               \   /                                                
                \ /     Vanilla Gradient g (Orthogonal in parameter map)
                 *----------------------->                           
                  \     \                                           
                   \     \ Natural Gradient F^-1 g (Steepest on manifold!)
                    \     v                                         
                     \                                              
                      +--------------------------> theta            
```

---

## 4. Real-World Analogy: Steering a High-Performance Aircraft

Imagine flying a supersonic jet:
- The pilot's controls (the parameters $\boldsymbol{\theta}$) have non-linear mechanical linkages: at low speeds ($100$ knots), deflecting the stick by $5^\circ$ barely turns the jet. At supersonic speeds (Mach 2), deflecting the stick by $5^\circ$ tears the wings off due to extreme aerodynamic loads.
- **Euclidean Policy Gradient:** Dictates: "Always move the stick by $5^\circ$." At high speeds, the jet disintegrates.
- **Natural Policy Gradient:** Senses the aerodynamic pressure manifold (Fisher Information Matrix) and dictates: "Apply whatever control deflection changes the aircraft's physical flight path by exactly $1.0^\circ$ of angular pitch." At low speeds, the stick moves $10^\circ$; at Mach 2, it moves $0.1^\circ$, maintaining safety across all regimes!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 2-Action Softmax Policy
Consider an agent with two discrete actions $\mathcal{A} = \{a_0, a_1\}$.
The policy parameters are $\boldsymbol{\theta} = [\theta_0, \theta_1]^\top$.

**Current Parameter State:**
$$\pi(a_0) = 0.8000, \quad \pi(a_1) = 0.2000$$

**Observed Standard (Vanilla) Policy Gradient:**
$$\mathbf{g} = \nabla_{\boldsymbol{\theta}} J = \begin{bmatrix} +1.0000 \\ -1.0000 \end{bmatrix}$$

**Hyperparameters:**
- KL divergence trust region: $\epsilon = 0.0100$
- Tikhonov regularization damping factor: $\delta_{\text{damp}} = 0.0400$ (ensures invertibility of $\mathbf{F}$)

We will compute:
1. Analytical score function vectors $\nabla_{\boldsymbol{\theta}} \log \pi(a_0)$ and $\nabla_{\boldsymbol{\theta}} \log \pi(a_1)$
2. The Fisher Information Matrix $\mathbf{F}$
3. The regularized inverse $\mathbf{F}_{\text{reg}}^{-1}$
4. The unnormalized natural gradient direction $\tilde{\mathbf{g}} = \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$
5. The quadratic form curvature $\mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$
6. The step size $\beta$ and the final normalized parameter update $\Delta \boldsymbol{\theta}^*$

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Walkthrough Value |
| :--- | :--- | :--- |
| $\boldsymbol{\pi}$ | Action Probabilities | $[\pi_0 = 0.8000, \pi_1 = 0.2000]^\top$ |
| $\mathbf{s}_0, \mathbf{s}_1$ | Score Vectors $\nabla \log \pi(a_i)$ | $\mathbf{s}_0 = [1 - \pi_0, -\pi_1]^\top, \mathbf{s}_1 = [-\pi_0, 1 - \pi_1]^\top$ |
| $\mathbf{F}$ | Fisher Information Matrix | $\pi_0 \mathbf{s}_0 \mathbf{s}_0^\top + \pi_1 \mathbf{s}_1 \mathbf{s}_1^\top$ |
| $\mathbf{F}_{\text{reg}}$ | Damped Fisher Matrix | $\mathbf{F} + \delta_{\text{damp}} \mathbf{I}$ |
| $\mathbf{F}_{\text{reg}}^{-1}$ | Inverse Metric Tensor | Analytical $2 \times 2$ inverse |
| $\mathbf{g}$ | Vanilla Policy Gradient | $[+1.0000, -1.0000]^\top$ |
| $\tilde{\mathbf{g}}$ | Natural Gradient Direction | $\mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$ |
| $\beta$ | Trust Region Step Scaling Factor | $\sqrt{\frac{2\epsilon}{\mathbf{g}^\top \tilde{\mathbf{g}}}}$ |
| $\Delta \boldsymbol{\theta}^*$ | Final Natural Policy Gradient Step | $\beta \tilde{\mathbf{g}}$ |

---

### 5.3 Step 1: Compute Score Functions

For action $a_0$:
$$\mathbf{s}_0 = \nabla_{\boldsymbol{\theta}} \log \pi(a_0) = \begin{bmatrix} 1 - \pi_0 \\ -\pi_1 \end{bmatrix} = \begin{bmatrix} 1 - 0.8000 \\ -0.2000 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.2000} \\ \mathbf{-0.2000} \end{bmatrix}$$

For action $a_1$:
$$\mathbf{s}_1 = \nabla_{\boldsymbol{\theta}} \log \pi(a_1) = \begin{bmatrix} -\pi_0 \\ 1 - \pi_1 \end{bmatrix} = \begin{bmatrix} -0.8000 \\ 1 - 0.2000 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.8000} \\ \mathbf{+0.8000} \end{bmatrix}$$

---

### 5.4 Step 2: Compute the Fisher Information Matrix $\mathbf{F}$

The outer products are:
$$\mathbf{s}_0 \mathbf{s}_0^\top = \begin{bmatrix} +0.2 \\ -0.2 \end{bmatrix} \begin{bmatrix} +0.2 & -0.2 \end{bmatrix} = \begin{bmatrix} 0.0400 & -0.0400 \\ -0.0400 & 0.0400 \end{bmatrix}$$
$$\mathbf{s}_1 \mathbf{s}_1^\top = \begin{bmatrix} -0.8 \\ +0.8 \end{bmatrix} \begin{bmatrix} -0.8 & +0.8 \end{bmatrix} = \begin{bmatrix} 0.6400 & -0.6400 \\ -0.6400 & 0.6400 \end{bmatrix}$$

Weight by probabilities:
$$\mathbf{F} = \pi_0 (\mathbf{s}_0 \mathbf{s}_0^\top) + \pi_1 (\mathbf{s}_1 \mathbf{s}_1^\top)$$
$$= 0.8000 \begin{bmatrix} 0.0400 & -0.0400 \\ -0.0400 & 0.0400 \end{bmatrix} + 0.2000 \begin{bmatrix} 0.6400 & -0.6400 \\ -0.6400 & 0.6400 \end{bmatrix}$$
$$= \begin{bmatrix} 0.0320 & -0.0320 \\ -0.0320 & 0.0320 \end{bmatrix} + \begin{bmatrix} 0.1280 & -0.1280 \\ -0.1280 & 0.1280 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.1600 & -0.1600 \\ -0.1600 & 0.1600 \end{bmatrix}}$$

---

### 5.5 Step 3: Regularize and Invert $\mathbf{F}$

Notice that $\det(\mathbf{F}) = (0.16)(0.16) - (-0.16)(-0.16) = 0$ (rank-deficient along the sum constraint!).
Add damping factor $\delta_{\text{damp}} = 0.0400 \mathbf{I}$:

$$\mathbf{F}_{\text{reg}} = \begin{bmatrix} 0.1600 & -0.1600 \\ -0.1600 & 0.1600 \end{bmatrix} + \begin{bmatrix} 0.0400 & 0.0000 \\ 0.0000 & 0.0400 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.2000 & -0.1600 \\ -0.1600 & 0.2000 \end{bmatrix}}$$

Compute Determinant:
$$\det(\mathbf{F}_{\text{reg}}) = (0.2000 \times 0.2000) - (-0.1600 \times -0.1600) = 0.0400 - 0.0256 = \mathbf{0.0144}$$

Compute Matrix Inverse:
$$\mathbf{F}_{\text{reg}}^{-1} = \frac{1}{0.0144} \begin{bmatrix} 0.2000 & 0.1600 \\ 0.1600 & 0.2000 \end{bmatrix} \approx \mathbf{\begin{bmatrix} 13.88889 & 11.11111 \\ 11.11111 & 13.88889 \end{bmatrix}}$$

---

### 5.6 Step 4: Compute Natural Gradient Direction $\tilde{\mathbf{g}}$

$$\tilde{\mathbf{g}} = \mathbf{F}_{\text{reg}}^{-1} \mathbf{g} = \begin{bmatrix} 13.88889 & 11.11111 \\ 11.11111 & 13.88889 \end{bmatrix} \begin{bmatrix} +1.0000 \\ -1.0000 \end{bmatrix}$$
$$\tilde{\mathbf{g}} = \begin{bmatrix} 13.88889 - 11.11111 \\ 11.11111 - 13.88889 \end{bmatrix} = \mathbf{\begin{bmatrix} +2.77778 \\ -2.77778 \end{bmatrix}}$$

---

### 5.7 Step 5: Trust Region Normalization ($\epsilon = 0.0100$)

Compute quadratic form curvature:
$$\mathbf{g}^\top \tilde{\mathbf{g}} = \mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g} = (+1.0000 \times 2.77778) + (-1.0000 \times -2.77778) = 2.77778 + 2.77778 = \mathbf{5.55556}$$

Compute scaling coefficient $\beta$:
$$\beta = \sqrt{\frac{2 \epsilon}{\mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}}} = \sqrt{\frac{2 \times 0.0100}{5.55556}} = \sqrt{\frac{0.0200}{5.55556}} = \sqrt{0.003600} = \mathbf{0.06000}$$

Final Natural Policy Gradient Step:
$$\Delta \boldsymbol{\theta}^* = \beta \tilde{\mathbf{g}} = 0.06000 \times \begin{bmatrix} +2.77778 \\ -2.77778 \end{bmatrix} = \mathbf{\begin{bmatrix} +0.16667 \\ -0.16667 \end{bmatrix}}$$

Check KL divergence constraint satisfaction:
$$\frac{1}{2} (\Delta \boldsymbol{\theta}^*)^\top \mathbf{F}_{\text{reg}} (\Delta \boldsymbol{\theta}^*) = \frac{1}{2} (0.0600)^2 (5.55556) = \frac{1}{2} (0.0036)(5.55556) = \mathbf{0.0100} = \epsilon \quad \checkmark$$

---

### 5.8 Visual Summary Matrix Grid

| Matrix / Vector | Formula / Operation | Value | Significance |
| :--- | :--- | :---: | :--- |
| **Score $\mathbf{s}_0$** | $[1 - \pi_0, -\pi_1]^\top$ | $[+0.2000, -0.2000]^\top$ | Sensitivity to action $a_0$ |
| **Score $\mathbf{s}_1$** | $[-\pi_0, 1 - \pi_1]^\top$ | $[-0.8000, +0.8000]^\top$ | Sensitivity to action $a_1$ |
| **Fisher $\mathbf{F}$** | $\sum \pi_i \mathbf{s}_i \mathbf{s}_i^\top$ | $\begin{bmatrix} 0.16 & -0.16 \\ -0.16 & 0.16 \end{bmatrix}$ | Metric tensor of policy manifold |
| **Damped $\mathbf{F}_{\text{reg}}$** | $\mathbf{F} + 0.04 \mathbf{I}$ | $\begin{bmatrix} 0.20 & -0.16 \\ -0.16 & 0.20 \end{bmatrix}$ | Invertible Riemannian metric |
| **Curvature $\mathbf{g}^\top \tilde{\mathbf{g}}$** | $\mathbf{g}^\top \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$ | $\mathbf{5.55556}$ | Metric curvature along gradient |
| **Step Multiplier $\beta$** | $\sqrt{2\epsilon / 5.55556}$ | $\mathbf{0.06000}$ | Normalizes step to KL $\le 0.01$ |
| **Final Step $\Delta \boldsymbol{\theta}^*$** | $\beta \mathbf{F}_{\text{reg}}^{-1} \mathbf{g}$ | $\mathbf{[+0.16667, -0.16667]^\top}$ | Exact Riemannian update! |

---

## 6. Solved Illustrations

### Illustration 1: Analytical Fisher Matrix of a 1D Gaussian Policy
**Problem:**
Let $\pi_\theta(a) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(a - \mu)^2}{2\sigma^2} \right)$ with parameter vector $\boldsymbol{\theta} = [\mu, \sigma]^\top$.
Derive the exact analytical Fisher Information Matrix $\mathbf{F}(\mu, \sigma)$.

**Solution:**
Log-likelihood:
$$\log \pi = -\log(\sqrt{2\pi}) - \log(\sigma) - \frac{(a - \mu)^2}{2\sigma^2}$$

Compute first derivatives:
$$\frac{\partial \log \pi}{\partial \mu} = \frac{a - \mu}{\sigma^2}, \quad \frac{\partial \log \pi}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(a - \mu)^2}{\sigma^3}$$

Compute expectations of outer products (using $\mathbb{E}[(a - \mu)^2] = \sigma^2, \mathbb{E}[(a - \mu)^3] = 0, \mathbb{E}[(a - \mu)^4] = 3\sigma^4$):
1. $F_{\mu\mu} = \mathbb{E}\left[ \frac{(a - \mu)^2}{\sigma^4} \right] = \frac{\sigma^2}{\sigma^4} = \mathbf{\frac{1}{\sigma^2}}$
2. $F_{\mu\sigma} = \mathbb{E}\left[ \frac{a - \mu}{\sigma^2} \left( -\frac{1}{\sigma} + \frac{(a - \mu)^2}{\sigma^3} \right) \right] = 0$ (odd moment)
3. $F_{\sigma\sigma} = \mathbb{E}\left[ \left( -\frac{1}{\sigma} + \frac{(a - \mu)^2}{\sigma^3} \right)^2 \right] = \frac{1}{\sigma^2} - \frac{2\sigma^2}{\sigma^4} + \frac{3\sigma^4}{\sigma^6} = \frac{1 - 2 + 3}{\sigma^2} = \mathbf{\frac{2}{\sigma^2}}$

**Analytical Fisher Matrix:**
$$\mathbf{F}(\mu, \sigma) = \begin{bmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{2}{\sigma^2} \end{bmatrix}$$

**Conclusion:** 
When variance $\sigma$ is small, $F$ blows up as $\mathcal{O}(1/\sigma^2)$, forcing natural gradient steps to become tiny and cautious, preventing catastrophic divergence! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **TRPO (Trust Region Policy Optimization - Chapter 11.18):** Directly scales Natural Policy Gradients to large neural networks using the **Conjugate Gradient algorithm** and **Fisher-vector products** ($O(d)$ compute without inverting $F$).
- **ACKTR (Wu et al., NeurIPS 2017):** Approximates the inverse Fisher matrix using Kronecker-Factored Approximate Curvature (K-FAC) for deep actor-critic networks.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical reproduction of the Part 5 hand calculations:
   - Score functions $\mathbf{s}_0, \mathbf{s}_1$
   - Fisher Matrix $\mathbf{F} = \begin{bmatrix} 0.16 & -0.16 \\ -0.16 & 0.16 \end{bmatrix}$
   - Damped Inverse $\mathbf{F}_{\text{reg}}^{-1}$
   - Natural gradient $\tilde{\mathbf{g}} = [+2.7778, -2.7778]^\top$
   - Step size $\beta = 0.06000$ and update $\Delta \boldsymbol{\theta} = [+0.16667, -0.16667]^\top$ matching NumPy to $< 10^{-14}$.
2. Coordinate-Invariance Test: Verifying that re-scaling policy parameters does not change the distribution update under natural gradients.

See implementation in:
[`11_reinforcement_learning/code/17_natural_policy_gradients.py`](./code/17_natural_policy_gradients.py)
