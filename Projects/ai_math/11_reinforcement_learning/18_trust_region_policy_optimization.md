# Module 11.18: Trust Region Policy Optimization (TRPO)

---

## 1. Intuition & 101 Motivation

In Chapter 11.17, we saw that **Natural Policy Gradients (NPG)** solve the fundamental coordinate-dependency problem of standard policy gradients by taking steps bounded in distribution space:
$$\bar{D}_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}) \le \delta$$
yielding the optimal step direction $\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g}$.

However, applying NPG to modern deep neural networks with millions of parameters ($d \ge 10^6$) presents a catastrophic computational barrier:
- Forming the Fisher Information Matrix $\mathbf{F} \in \mathbb{R}^{d \times d}$ requires storing $10^6 \times 10^6 = 10^{12}$ floating-point numbers ($\approx 4 \text{ Terabytes}$ of GPU RAM!).
- Inverting $\mathbf{F}$ via Gaussian elimination requires $\mathcal{O}(d^3) \approx 10^{18}$ floating-point operations, taking weeks for a single gradient step.

In 2015, John Schulman et al. introduced **Trust Region Policy Optimization (TRPO)**, which made natural policy gradients scalable, practical, and robust for deep networks:
1. **The Surrogate Advantage Objective:** Uses importance sampling ratios to formulate an objective that can be evaluated on off-policy rollouts.
2. **Conjugate Gradient (CG) with Hessian-Vector Products:** Computes the step $\mathbf{x} \approx \mathbf{F}^{-1} \mathbf{g}$ in $k \le 10$ iterations without ever forming, storing, or inverting the Fisher matrix $\mathbf{F}$!
3. **Monotonic Improvement Guarantee:** Guarantees that every policy update increases expected return $J(\pi)$.
4. **Backtracking Line Search:** Verifies that the quadratic Taylor approximation didn't violate the non-linear trust region boundary.

```
+-----------------------------------------------------------------------------------------+
|                                    TRPO PIPELINE                                        |
|                                                                                         |
|   Rollout Trajectories ----> Compute Surrogate Gradient g = grad L(theta)               |
|                                     |                                                   |
|                                     v                                                   |
|                 [ Conjugate Gradient Solver: F x = g ]                                  |
|                 (Uses Hessian-Vector Products: O(d) memory!)                            |
|                                     |                                                   |
|                                     v                                                   |
|                 Search Direction: s = sqrt(2 * delta / x^T F x) * x                     |
|                                     |                                                   |
|                                     v                                                   |
|                 [ Backtracking Line Search: theta + alpha^j * s ]                       |
|                 (Accept step if: L(theta_new) >= 0 AND KL <= delta)                     |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Performance Difference Lemma (Kakade & Langford, 2002)

Let $\pi$ and $\tilde{\pi}$ be any two arbitrary policies. The difference in their expected returns satisfies:

$$J(\tilde{\pi}) - J(\pi) = \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^\infty \gamma^t A^\pi(S_t, A_t) \right] = \sum_{s \in \mathcal{S}} d^{\tilde{\pi}}(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a)$$

*Interpretation:* The expected return of a new policy $\tilde{\pi}$ equals the return of the old policy $\pi$ plus the expected advantage of $\tilde{\pi}$'s actions evaluated under $\tilde{\pi}$'s state visitation distribution $d^{\tilde{\pi}}$.

---

### 2.2 The Surrogate Objective & The Monotonic Improvement Bound

The performance difference formula depends on $d^{\tilde{\pi}}(s)$, which is unknown because we have not yet run $\tilde{\pi}$ in the environment.
Schulman et al. define the **Surrogate Objective** $L_\pi(\tilde{\pi})$ by substituting the known state distribution $d^\pi(s)$ of the old policy:

$$L_\pi(\tilde{\pi}) \triangleq J(\pi) + \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) A^\pi(s, a)$$

Using importance sampling, this can be estimated empirically from rollouts collected under policy $\theta_{\text{old}}$:
$$L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) = \mathbb{E}_{s \sim d^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a \mid s)} \hat{A}^{\pi_{\text{old}}}(s, a) \right]$$

Notice that at $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}$:
$$L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{old}}) = 0, \quad \left. \nabla_{\boldsymbol{\theta}} L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) \right|_{\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}} = \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_{\text{old}})$$
The surrogate objective matches the true return $J(\boldsymbol{\theta})$ to first order!

#### Theorem: Monotonic Improvement Bound (Schulman et al., 2015)
Let $D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) \triangleq \max_{s \in \mathcal{S}} D_{\text{KL}}(\pi(\cdot \mid s) \parallel \tilde{\pi}(\cdot \mid s))$ and $\epsilon \triangleq \max_{s, a} |A^\pi(s, a)|$.
Then the true return is lower-bounded by:
$$J(\tilde{\pi}) \ge L_\pi(\tilde{\pi}) - C \cdot D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) \quad \text{where} \quad C \triangleq \frac{4 \epsilon \gamma}{(1 - \gamma)^2}$$

This guarantees that maximizing the right-hand side is guaranteed to generate **monotonic policy improvement**: $J(\pi_{k+1}) \ge J(\pi_k)$!

---

### 2.3 The TRPO Constrained Optimization Problem

In practice, the penalty coefficient $C$ is too large, resulting in overly conservative steps. TRPO converts the penalty into a **hard trust region constraint** on the average KL divergence:

$$\max_{\boldsymbol{\theta}} \mathbb{E}_{s \sim d^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a \mid s)} \hat{A}_t \right] \quad \text{subject to} \quad \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \le \delta$$
where $\bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \triangleq \mathbb{E}_{s \sim d^{\pi_{\text{old}}}} [D_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}}(\cdot \mid s) \parallel \pi_{\boldsymbol{\theta}}(\cdot \mid s))]$.

#### Quadratic Approximation:
Taking a first-order Taylor expansion of the objective and a second-order expansion of the constraint:
$$\max_{\Delta \boldsymbol{\theta}} \mathbf{g}^\top \Delta \boldsymbol{\theta} \quad \text{subject to} \quad \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \le \delta$$
where:
$$\mathbf{g} \triangleq \left. \nabla_{\boldsymbol{\theta}} L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) \right|_{\boldsymbol{\theta}_{\text{old}}}, \quad \mathbf{F} \triangleq \left. \nabla_{\boldsymbol{\theta}}^2 \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \right|_{\boldsymbol{\theta}_{\text{old}}}$$

As derived in Chapter 11.17, the analytical solution is:
$$\Delta \boldsymbol{\theta} = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{F} \mathbf{x}}} \mathbf{x} \quad \text{where} \quad \mathbf{F} \mathbf{x} = \mathbf{g}$$

---

### 2.4 The Hessian-Vector Product Trick (Pearlmutter, 1994)

To solve $\mathbf{F} \mathbf{x} = \mathbf{g}$, we need to multiply the matrix $\mathbf{F}$ by arbitrary search vectors $\mathbf{v} \in \mathbb{R}^d$ without computing or storing $\mathbf{F}$.
Notice that:
$$\mathbf{F} \mathbf{v} = \nabla_{\boldsymbol{\theta}}^2 \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \mathbf{v} = \nabla_{\boldsymbol{\theta}} \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta})^\top \mathbf{v} \right)$$

#### In PyTorch / Autograd:
1. Compute the scalar quantity $y = \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}} \right)^\top \mathbf{v}$ (an inner product of vectors).
2. Take the gradient of $y$ with respect to $\boldsymbol{\theta}$:
   $$\mathbf{F} \mathbf{v} = \operatorname{autograd.grad}(y, \boldsymbol{\theta})$$
This requires only **one extra backward pass**, using $\mathcal{O}(d)$ memory!

---

### 2.5 Conjugate Gradient (CG) Algorithm

To solve $\mathbf{F} \mathbf{x} = \mathbf{g}$ iteratively:
```
Algorithm: Conjugate Gradient for F x = g
Input: Matrix-vector function F(v), vector g, tolerance tol, max_iter K=10

x_0 = 0,  r_0 = g,  p_0 = r_0
For i = 0, 1, ..., K - 1:
    v = F(p_i)                          # One Hessian-vector product!
    alpha_i = (r_i^T r_i) / (p_i^T v)
    x_{i+1} = x_i + alpha_i * p_i
    r_{i+1} = r_i - alpha_i * v
    if ||r_{i+1}|| < tol: break
    beta_i = (r_{i+1}^T r_{i+1}) / (r_i^T r_i)
    p_{i+1} = r_{i+1} + beta_i * p_i

Return x_{final}
```

---

### 2.6 Backtracking Line Search

Because the linear-quadratic approximation is only locally accurate, taking the full step $\Delta \boldsymbol{\theta}$ might violate the non-linear KL constraint or decrease the surrogate objective.
TRPO applies a **backtracking line search**:
$$\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} + \alpha^j \Delta \boldsymbol{\theta} \quad \text{for } j \in \{0, 1, 2, \dots, M\}$$
where $\alpha \in (0, 1)$ (typically $\alpha = 0.5$, max steps $M = 10$).

The candidate update is accepted at the smallest integer $j$ that satisfies both:
1. **Surrogate Improvement:** $L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{new}}) \ge 0$
2. **Trust Region Enforcement:** $\bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}_{\text{new}}) \le \delta$

If no step satisfies both criteria after $M$ reductions, the update is aborted ($\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}}$), ensuring total stability!

---

## 3. Geometric & Physical Interpretation

### The Trust Region Ellipsoid
In parameter space $\mathbb{R}^d$:
- The constraint $\frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F} \Delta \boldsymbol{\theta} \le \delta$ defines a high-dimensional **hyper-ellipsoid**.
- The semi-axes of this ellipsoid are proportional to $1/\sqrt{\lambda_i}$, where $\lambda_i$ are the eigenvalues of $\mathbf{F}$. In directions where the policy is sensitive (large $\lambda_i$), the ellipsoid is thin and narrow; in insensitive directions, the ellipsoid is elongated.
- The Conjugate Gradient algorithm finds the point on this ellipsoid that extends furthest in the direction of objective gradient $\mathbf{g}$.
- The Backtracking Line Search pulls the step backward along the chord if the curvature deviates from quadratic flatness.

```
                         Trust Region Ellipsoid
                        (1/2) dtheta^T F dtheta <= delta
                                   _____
                                .-'     `-.
                              .'     g     `.   * Step proposed by CG
                             /        \      \ /
                            |          *======* (Normalized boundary)
                            |         /       |
                             \       *       /
                              `.   theta_old.'
                                `-._____.-'
```

---

## 4. Real-World Analogy: The Rock Climber's Safety Harness

Imagine a rock climber navigating a sheer vertical cliff:
- **Vanilla Policy Gradient:** Takes a blind 3-foot leap upward toward the next handhold. If the hold breaks, the climber plummets to the ground.
- **Natural Policy Gradient:** Computes the exact tension vector on the rope, but requires solving complex multi-body physics equations on a supercomputer before each move.
- **TRPO:** The climber is attached to an adjustable **safety harness with dynamic friction braking**:
  1. The harness strictly limits any single upward move to within a guaranteed safe radius of 1 foot ($\bar{D}_{\text{KL}} \le \delta$).
  2. The climber tests the foothold with increasing weight: first $100\%$ force, then $50\%$, then $25\%$ (Backtracking Line Search).
  3. If the rock face crumbles at any point, the harness locks, and the climber resets to their last secure stance. A fall is mathematically impossible!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: 2-Parameter Quadratic System
Let us execute the exact **Conjugate Gradient (CG)** solver and **Trust Region Scaling** step-by-step by hand!

**Objective Gradient Vector:**
$$\mathbf{g} = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix}$$

**Fisher Information Matrix:**
$$\mathbf{F} = \begin{bmatrix} 2.0000 & 0.5000 \\ 0.5000 & 1.0000 \end{bmatrix}$$

**Trust Region Boundary:**
$$\delta = 0.0500$$

We will compute:
1. Conjugate Gradient Iteration 0: Matrix-vector product $\mathbf{F} \mathbf{p}_0$, curvature, step size $\alpha_0$, updated position $\mathbf{x}_1$, residual $\mathbf{r}_1$.
2. Conjugate Gradient Iteration 1: Beta factor $\beta_0$, conjugate search direction $\mathbf{p}_1$, matrix-vector product $\mathbf{F} \mathbf{p}_1$, step size $\alpha_1$, final solution $\mathbf{x}_2$.
3. Exact analytical solution verification via $\mathbf{F}^{-1} \mathbf{g}$.
4. Quadratic form $\mathbf{x}^\top \mathbf{F} \mathbf{x}$.
5. Final trust region step scaling factor and update vector $\Delta \boldsymbol{\theta}^*$.

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Concrete Hand Value / Matrix |
| :--- | :--- | :--- |
| $\mathbf{g}$ | Policy Surrogate Gradient | $[2.0000, 1.0000]^\top$ |
| $\mathbf{F}$ | Fisher Information Matrix | $\begin{bmatrix} 2.0 & 0.5 \\ 0.5 & 1.0 \end{bmatrix}$ |
| $\delta$ | Maximum Allowable KL Divergence | $0.0500$ |
| $\mathbf{x}_k$ | CG Approximate Solution to $\mathbf{F}\mathbf{x} = \mathbf{g}$ | Vectors in $\mathbb{R}^2$ |
| $\mathbf{r}_k$ | CG Residual Vector | $\mathbf{g} - \mathbf{F} \mathbf{x}_k$ |
| $\mathbf{p}_k$ | CG Conjugate Search Direction | Conjugate to all previous $\mathbf{p}$ |
| $\Delta \boldsymbol{\theta}^*$ | Final Scaled TRPO Step | $\sqrt{\frac{2\delta}{\mathbf{x}^\top \mathbf{F} \mathbf{x}}} \mathbf{x}$ |

---

### 5.3 Step 1: Conjugate Gradient — Iteration 0

**Initialization:**
- Initial guess: $\mathbf{x}_0 = \begin{bmatrix} 0.0000 \\ 0.0000 \end{bmatrix}$
- Initial residual: $\mathbf{r}_0 = \mathbf{g} - \mathbf{F} \mathbf{x}_0 = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix}$
- Initial search direction: $\mathbf{p}_0 = \mathbf{r}_0 = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix}$

**Compute Matrix-Vector Product $\mathbf{v}_0 = \mathbf{F} \mathbf{p}_0$:**
$$\mathbf{v}_0 = \begin{bmatrix} 2.0000 & 0.5000 \\ 0.5000 & 1.0000 \end{bmatrix} \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix} = \begin{bmatrix} (2.0 \times 2.0) + (0.5 \times 1.0) \\ (0.5 \times 2.0) + (1.0 \times 1.0) \end{bmatrix} = \begin{bmatrix} 4.0 + 0.5 \\ 1.0 + 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 4.5000 \\ 2.0000 \end{bmatrix}}$$

**Compute Curvature along $\mathbf{p}_0$:**
$$\mathbf{p}_0^\top \mathbf{v}_0 = \mathbf{p}_0^\top \mathbf{F} \mathbf{p}_0 = (2.0000 \times 4.5000) + (1.0000 \times 2.0000) = 9.0000 + 2.0000 = \mathbf{11.0000}$$

**Compute Residual Norm Squared:**
$$\mathbf{r}_0^\top \mathbf{r}_0 = (2.0000)^2 + (1.0000)^2 = 4.0000 + 1.0000 = \mathbf{5.0000}$$

**Compute Step Size $\alpha_0$:**
$$\alpha_0 = \frac{\mathbf{r}_0^\top \mathbf{r}_0}{\mathbf{p}_0^\top \mathbf{F} \mathbf{p}_0} = \frac{5.0000}{11.0000} \approx \mathbf{0.45455}$$

**Update Solution $\mathbf{x}_1$:**
$$\mathbf{x}_1 = \mathbf{x}_0 + \alpha_0 \mathbf{p}_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} + 0.45455 \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.90909 \\ 0.45455 \end{bmatrix}}$$

**Update Residual $\mathbf{r}_1$:**
$$\mathbf{r}_1 = \mathbf{r}_0 - \alpha_0 \mathbf{v}_0 = \begin{bmatrix} 2.0000 \\ 1.0000 \end{bmatrix} - 0.45455 \begin{bmatrix} 4.5000 \\ 2.0000 \end{bmatrix}$$
$$\mathbf{r}_1 = \begin{bmatrix} 2.0000 - 2.04546 \\ 1.0000 - 0.90909 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.04545 \\ +0.09091 \end{bmatrix}}$$

---

### 5.4 Step 2: Conjugate Gradient — Iteration 1

**Compute New Residual Norm Squared:**
$$\mathbf{r}_1^\top \mathbf{r}_1 = (-0.04545)^2 + (0.09091)^2 \approx 0.002066 + 0.008264 = \mathbf{0.01033}$$

**Compute Gram-Schmidt Beta $\beta_0$:**
$$\beta_0 = \frac{\mathbf{r}_1^\top \mathbf{r}_1}{\mathbf{r}_0^\top \mathbf{r}_0} = \frac{0.01033}{5.0000} \approx \mathbf{0.002066}$$

**Compute Conjugate Direction $\mathbf{p}_1$:**
$$\mathbf{p}_1 = \mathbf{r}_1 + \beta_0 \mathbf{p}_0 = \begin{bmatrix} -0.04545 \\ +0.09091 \end{bmatrix} + 0.002066 \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} \approx \begin{bmatrix} -0.04545 + 0.00413 \\ 0.09091 + 0.00207 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.04132 \\ +0.09298 \end{bmatrix}}$$

**Compute Matrix-Vector Product $\mathbf{v}_1 = \mathbf{F} \mathbf{p}_1$:**
$$\mathbf{v}_1 = \begin{bmatrix} 2.0 & 0.5 \\ 0.5 & 1.0 \end{bmatrix} \begin{bmatrix} -0.04132 \\ 0.09298 \end{bmatrix} = \begin{bmatrix} -0.08264 + 0.04649 \\ -0.02066 + 0.09298 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.03615 \\ +0.07232 \end{bmatrix}}$$

**Compute Curvature along $\mathbf{p}_1$:**
$$\mathbf{p}_1^\top \mathbf{v}_1 = (-0.04132 \times -0.03615) + (0.09298 \times 0.07232) = 0.001494 + 0.006724 = \mathbf{0.008218}$$

**Compute Step Size $\alpha_1$:**
$$\alpha_1 = \frac{\mathbf{r}_1^\top \mathbf{r}_1}{\mathbf{p}_1^\top \mathbf{F} \mathbf{p}_1} = \frac{0.01033}{0.008218} \approx \mathbf{1.25700}$$

**Update Solution $\mathbf{x}_2$:**
$$\mathbf{x}_2 = \mathbf{x}_1 + \alpha_1 \mathbf{p}_1 = \begin{bmatrix} 0.90909 \\ 0.45455 \end{bmatrix} + 1.25700 \begin{bmatrix} -0.04132 \\ 0.09298 \end{bmatrix}$$
$$\mathbf{x}_2 = \begin{bmatrix} 0.90909 - 0.05194 \\ 0.45455 + 0.11688 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.85714 \\ 0.57143 \end{bmatrix}}$$

---

### 5.5 Step 3: Analytical Verification ($\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$)

Let us verify whether $\mathbf{x}_2$ is the exact analytical inverse solution!
Determinant of $\mathbf{F}$:
$$\det(\mathbf{F}) = (2.0 \times 1.0) - (0.5 \times 0.5) = 2.0 - 0.25 = \mathbf{1.7500} = \frac{7}{4}$$

Matrix Inverse $\mathbf{F}^{-1}$:
$$\mathbf{F}^{-1} = \frac{1}{1.7500} \begin{bmatrix} 1.0 & -0.5 \\ -0.5 & 2.0 \end{bmatrix} = \frac{4}{7} \begin{bmatrix} 1.0 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$$

Analytical Solution:
$$\mathbf{x}^* = \mathbf{F}^{-1} \mathbf{g} = \frac{4}{7} \begin{bmatrix} 1.0 & -0.5 \\ -0.5 & 2.0 \end{bmatrix} \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \frac{4}{7} \begin{bmatrix} 2.0 - 0.5 \\ -1.0 + 2.0 \end{bmatrix} = \frac{4}{7} \begin{bmatrix} 1.5 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 6/7 \\ 4/7 \end{bmatrix} \approx \mathbf{\begin{bmatrix} 0.85714 \\ 0.57143 \end{bmatrix}}$$

**The Conjugate Gradient solver converged to the exact analytical inverse in precisely 2 iterations!**

---

### 5.6 Step 4: Trust Region Step Normalization

Compute the quadratic curvature of the search direction:
$$\mathbf{x}^\top \mathbf{F} \mathbf{x} = \mathbf{x}^\top \mathbf{g} = (0.85714 \times 2.0000) + (0.57143 \times 1.0000) = 1.71429 + 0.57143 = \mathbf{2.28571} = \frac{16}{7}$$

Trust region constraint is $\delta = 0.0500$.
Compute scaling multiplier $\beta$:
$$\beta = \sqrt{\frac{2 \delta}{\mathbf{x}^\top \mathbf{F} \mathbf{x}}} = \sqrt{\frac{2 \times 0.0500}{2.28571}} = \sqrt{\frac{0.1000}{2.28571}} = \sqrt{0.04375} \approx \mathbf{0.20917}$$

Final TRPO Step $\Delta \boldsymbol{\theta}^*$:
$$\Delta \boldsymbol{\theta}^* = \beta \mathbf{x} = 0.20917 \begin{bmatrix} 0.85714 \\ 0.57143 \end{bmatrix} \approx \mathbf{\begin{bmatrix} 0.17929 \\ 0.11952 \end{bmatrix}}$$

Verify Constraint:
$$\frac{1}{2} (\Delta \boldsymbol{\theta}^*)^\top \mathbf{F} (\Delta \boldsymbol{\theta}^*) = \frac{1}{2} \beta^2 (\mathbf{x}^\top \mathbf{F} \mathbf{x}) = \frac{1}{2} (0.04375)(2.28571) = \mathbf{0.0500} = \delta \quad \checkmark$$

---

### 5.7 Visual Summary Grid: CG Walkthrough Matrix

| Iteration | Search Direction $\mathbf{p}_k$ | Matrix-Vector Product $\mathbf{F} \mathbf{p}_k$ | Curvature $\mathbf{p}_k^\top \mathbf{F} \mathbf{p}_k$ | Step $\alpha_k$ | Solution Estimate $\mathbf{x}_k$ | Residual $\mathbf{r}_k$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 0$** | $[2.0000, 1.0000]^\top$ | $[4.5000, 2.0000]^\top$ | $11.0000$ | $0.45455$ | $[0.90909, 0.45455]^\top$ | $[-0.0455, +0.0909]^\top$ |
| **$k = 1$** | $[-0.0413, +0.0930]^\top$ | $[-0.0362, +0.0723]^\top$ | $0.00822$ | $1.25700$ | $\mathbf{[0.85714, 0.57143]^\top}$ | $\approx [0.0, 0.0]^\top$ |
| **Exact** | - | - | - | - | $\mathbf{[6/7, 4/7]^\top}$ | $\mathbf{0.0000}$ |

---

## 6. Solved Illustrations

### Illustration 1: Implementation of Hessian-Vector Product in PyTorch
**Problem:**
Write the exact 4-line PyTorch function that computes $\mathbf{F} \mathbf{v}$ without forming $\mathbf{F}$.
**Solution:**
```python
def hessian_vector_product(kl_div, policy_net, v):
    # First backward pass: gradient of KL divergence w.r.t parameters
    grads = torch.autograd.grad(kl_div, policy_net.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([g.view(-1) for g in grads])
    
    # Inner product with arbitrary search vector v
    grad_vector_product = torch.dot(flat_grad_kl, v)
    
    # Second backward pass: gradient of inner product w.r.t parameters
    hvp = torch.autograd.grad(grad_vector_product, policy_net.parameters(), retain_graph=True)
    return torch.cat([h.contiguous().view(-1) for h in hvp])
```
This fundamental snippet powers TRPO and natural gradient libraries worldwide. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **Foundation for PPO:** TRPO's mathematical theory (monotonic improvement, surrogate objectives, trust region bounds) directly inspired Proximal Policy Optimization (PPO), which achieves similar performance using a simpler first-order clipped surrogate loss.
- **Safety-Critical Robotics:** In legged locomotion (ANYmal, Boston Dynamics Spot) and drone flight control, TRPO remains a gold standard when hard KL divergence safety guarantees are non-negotiable.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of Part 5 Conjugate Gradient hand calculations:
   - Iteration 0: $\alpha_0 = 0.45455, \mathbf{x}_1 = [0.9091, 0.4545]^\top$
   - Iteration 1: $\mathbf{x}_2 = [0.85714, 0.57143]^\top$ matching analytical inverse $[6/7, 4/7]^\top$ to $< 10^{-14}$.
   - Trust region scaling $\beta = 0.20917$ and final step $\Delta \boldsymbol{\theta}^* = [0.17929, 0.11952]^\top$.
2. Standalone PyTorch implementation of Hessian-Vector Products and Conjugate Gradient.
3. Complete TRPO agent tested on continuous control inverted pendulum dynamics.

See implementation in:
[`11_reinforcement_learning/code/18_trust_region_policy_optimization.py`](./code/18_trust_region_policy_optimization.py)
