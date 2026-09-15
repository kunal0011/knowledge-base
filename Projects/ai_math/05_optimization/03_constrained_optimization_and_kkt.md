# Chapter 5.3: Constrained Optimization, Lagrange Multipliers & KKT Conditions

---

## Part 1: Intuition & 101 Motivation

In Chapter 5.2, we assumed parameters could wander freely through all of Euclidean space $\mathbb{R}^n$.
In real-world machine learning and artificial intelligence, however, parameters are almost always subject to **hard physical or mathematical constraints**:

1. **Adversarial Robustness:** Generate the most deceptive adversarial attack image $\mathbf{x}_{\text{adv}} = \mathbf{x} + \delta$ **subject to** the perturbation being imperceptible: $\|\delta\|_\infty \le \epsilon$.
2. **Support Vector Machines (SVMs):** Maximize the separation margin between two classes **subject to** zero training classification errors: $y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1$.
3. **Generative Adversarial Networks (GANs):** Enforce Lipschitz continuity in Wasserstein GANs **subject to** the discriminator's spectral norm: $\sigma(W) \le 1$.
4. **Trust Region Policy Optimization (TRPO):** Take the largest possible policy update step in reinforcement learning **subject to** a maximum distribution shift: $D_{\text{KL}}(\pi_{\text{old}} \parallel \pi_\theta) \le \delta$.

How do we optimize an objective when trapped against a wall of constraints?

The answer is the theory of **Lagrange Multipliers** and the **Karush-Kuhn-Tucker (KKT) Conditions**.
By converting geometric constraint walls into mathematical "forces" (dual variables $\lambda_i$), the KKT conditions provide the exact language to solve constrained problems and unlock the powerful theory of **Primal-Dual Optimization**.

---

## Part 2: Rigorous Mathematical Formulation

### 1. General Non-Linear Constrained Optimization
The standard form of a constrained optimization problem is:
$$\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{subject to} \quad & g_i(x) \le 0, \quad i = 1, 2, \dots, m \quad (\text{Inequality Constraints}) \\
& h_j(x) = 0, \quad j = 1, 2, \dots, p \quad (\text{Equality Constraints})
\end{aligned}$$
where $f, g_i, h_j: \mathbb{R}^n \to \mathbb{R}$ are continuously differentiable functions.
The **Feasible Set** is $\mathcal{D} = \{x \in \mathbb{R}^n \mid g_i(x) \le 0 \,\, \forall i, \,\, h_j(x) = 0 \,\, \forall j\}$.

---

### 2. The Lagrangian Function & Dual Multipliers
To absorb the constraints into an unconstrained objective, we define the **Lagrangian** $\mathcal{L}: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$:
$$\mathbf{\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)}$$
where:
- $\lambda_i \ge 0$ is the **Lagrange multiplier** associated with inequality constraint $g_i(x) \le 0$.
- $\nu_j \in \mathbb{R}$ is the Lagrange multiplier associated with equality constraint $h_j(x) = 0$.

---

### 3. The Lagrange Dual Problem

#### The Lagrange Dual Function $g(\lambda, \nu)$:
The dual function is defined as the infimum of the Lagrangian over the primal variable $x$:
$$g(\lambda, \nu) = \inf_{x \in \mathbb{R}^n} \mathcal{L}(x, \lambda, \nu) = \inf_{x \in \mathbb{R}^n} \left[ f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x) \right]$$

> [!NOTE]
> **Concavity of the Dual Function:**
> Since $g(\lambda, \nu)$ is the pointwise infimum of a family of affine functions in $(\lambda, \nu)$, **$g(\lambda, \nu)$ is ALWAYS concave**, even if the original objective $f(x)$ and constraints $g_i(x)$ are horribly non-convex!

#### The Dual Optimization Problem:
$$\max_{\lambda \succeq \mathbf{0}, \,\, \nu} \quad g(\lambda, \nu)$$

#### Weak Duality Theorem:
For any feasible primal point $x$ and any feasible dual point $(\lambda \succeq \mathbf{0}, \nu)$:
$$g(\lambda, \nu) \le f(x)$$
Let $p^* = \min_x f(x)$ be the primal optimal value, and $d^* = \max_{\lambda \ge 0, \nu} g(\lambda, \nu)$ be the dual optimal value.
Weak Duality states:
$$\mathbf{d^* \le p^*}$$
The difference $p^* - d^* \ge 0$ is the **Duality Gap**.

#### Strong Duality & Slater's Condition:
**Strong Duality** holds when the duality gap is zero:
$$d^* = p^*$$
**Slater's Condition (Convex Problems):**
If the primal problem is convex ($f$ and $g_i$ are convex, $h_j$ are affine) and there exists at least one **strictly feasible point** $\tilde{x}$ such that:
$$g_i(\tilde{x}) < 0 \,\, \forall i = 1, \dots, m \quad \text{and} \quad h_j(\tilde{x}) = 0 \,\, \forall j = 1, \dots, p$$
then **Strong Duality holds ($p^* = d^*$)** and the dual optimum is attained!

---

### 4. The Karush-Kuhn-Tucker (KKT) Conditions
When strong duality holds, any primal-dual optimal pair $(x^*, \lambda^*, \nu^*)$ must satisfy the **four KKT conditions**:

#### 1. Stationarity:
The gradient of the Lagrangian with respect to $x$ must vanish:
$$\mathbf{\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = \nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = \mathbf{0}}$$

#### 2. Primal Feasibility:
The optimal point must satisfy all original problem constraints:
$$g_i(x^*) \le 0 \quad \forall i = 1, \dots, m$$
$$h_j(x^*) = 0 \quad \forall j = 1, \dots, p$$

#### 3. Dual Feasibility:
The inequality multipliers must be non-negative:
$$\mathbf{\lambda_i^* \ge 0 \quad \forall i = 1, \dots, m}$$

#### 4. Complementary Slackness:
The product of each multiplier and its inequality constraint must be exactly zero:
$$\mathbf{\lambda_i^* \cdot g_i(x^*) = 0 \quad \forall i = 1, \dots, m}$$

---

### 5. The Deep Significance of Complementary Slackness
The condition $\lambda_i^* g_i(x^*) = 0$ allows only **two mutually exclusive states** for each constraint:

1. **State 1 (Inactive / Non-Binding Constraint):**
   $$g_i(x^*) < 0 \implies \mathbf{\lambda_i^* = 0}$$
   The optimal point lies strictly inside the interior of the feasible set. Removing this constraint entirely would not change the optimal solution at all!
2. **State 2 (Active / Binding Constraint):**
   $$\mathbf{\lambda_i^* > 0} \implies g_i(x^*) = 0$$
   The optimal point is pressed right against the constraint boundary. The multiplier $\lambda_i^* > 0$ represents the **force / pressure** the constraint exerts to prevent the solution from descending further!

---

## Part 3: Geometric & Algebraic Interpretation

### The Force Balance Geometry of KKT Stationarity
At the boundary of an active constraint $g(x) = 0$:
$$\nabla f(x^*) + \lambda^* \nabla g(x^*) = \mathbf{0} \implies -\nabla f(x^*) = \lambda^* \nabla g(x^*)$$

```
                           Constraint Boundary g(x) = 0
                               Feasible Region (g <= 0)
                                      │
                                      │
                                      │   λ* ∇g(x*)  (Constraint Normal Force)
                                      ├───►
                            x* ●──────┤
                                      │
                                  ◄───┤
                            -∇f(x*)   │
                    (Gravitational    │   Infeasible Region (g > 0)
                     Descent Pull)    │
```

- $-\nabla f(x^*)$ is the direction in which the objective function wants to roll downhill.
- $\nabla g(x^*)$ is the outward normal vector pointing into the forbidden (infeasible) territory.
- Because $\lambda^* \ge 0$, $-\nabla f(x^*)$ and $\nabla g(x^*)$ are **parallel and point in the exact same direction**!
- The downward pull of the objective is perfectly balanced by the perpendicular normal reaction force of the boundary wall!

---

## Part 4: Real-World Analogy

### The Sled on the Mountain Slope
Imagine you place a sled on a snowy mountainside:
- **Objective $f(x)$:** Minimize gravitational potential energy (slide downhill).
- **Constraint $g(x) \le 0$:** A sturdy wooden boundary fence erected across the middle of the slope.

**Scenario A (Active Constraint / Binding Boundary):**
You release the sled above the fence. It slides down and slams into the fence ($g(x^*) = 0$), coming to a dead stop.
The fence pushes back on the sled with a positive normal reaction force ($\lambda^* > 0$). If you remove the fence, the sled slides further down.

**Scenario B (Inactive Constraint / Slack Boundary):**
The fence is built at the bottom of the valley, but the natural bowl of the hill stops the sled 50 meters before reaching the fence ($g(x^*) < 0$).
The fence exerts zero physical force on the sled ($\lambda^* = 0$). Removing the fence changes nothing!
**This is precisely Complementary Slackness: $\lambda^* \cdot g(x^*) = 0$!**

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us solve a constrained quadratic optimization problem by hand from first principles using the KKT conditions.

### 1. Problem Setup
$$\min_{x, y \in \mathbb{R}} \quad f(x, y) = x^2 + y^2$$
$$\text{subject to} \quad g(x, y) = 1.0 - x - y \le 0 \quad (\text{i.e. } x + y \ge 1.0)$$

Notice:
- Without constraints, the global unconstrained minimum is at the origin $(0, 0)$ where $f = 0$.
- But at $(0, 0)$, $x + y = 0 < 1.0$, which violates the constraint!
- The constraint $x + y \ge 1.0$ cuts off the origin. Where is the constrained optimum?

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $(x, y)$ | Primal variables | Network weights $(w_1, w_2)$ | Float pair | Evaluated at candidate points |
| $f(x, y)$ | Primal objective function | Training loss $\mathcal{L}(w)$ | Float scalar | $x^2 + y^2$ |
| $g(x, y)$ | Inequality constraint | Sparsity / Trust-Region constraint | Float scalar | $1.0 - x - y \le 0$ |
| $\lambda$ | Dual Lagrange multiplier | Regularization penalty strength | Float scalar $\ge 0$ | Solved to be $1.0000$ |
| $\mathcal{L}(x, y, \lambda)$| Lagrangian function | Augmented regularized objective | Float scalar | $x^2 + y^2 + \lambda(1 - x - y)$ |
| $(x^*, y^*)$ | Primal optimal solution | Constrained optimal weights | Float pair | $(0.5000, 0.5000)$ |

---

### 3. Step-by-Step Manual KKT Derivation

#### Step 1: Formulate the Lagrangian
$$\mathcal{L}(x, y, \lambda) = x^2 + y^2 + \lambda(1.0 - x - y)$$

#### Step 2: Write Down the Four KKT Conditions
1. **Stationarity:**
   $$\frac{\partial \mathcal{L}}{\partial x} = 2x - \lambda = 0 \implies x = \frac{\lambda}{2}$$
   $$\frac{\partial \mathcal{L}}{\partial y} = 2y - \lambda = 0 \implies y = \frac{\lambda}{2}$$
2. **Primal Feasibility:**
   $$1.0 - x - y \le 0 \implies x + y \ge 1.0$$
3. **Dual Feasibility:**
   $$\lambda \ge 0$$
4. **Complementary Slackness:**
   $$\lambda (1.0 - x - y) = 0$$

---

#### Step 3: Test Both Complementary Slackness Cases

**Case A: $\lambda = 0$ (Inactive Constraint Hypothesis)**
- By stationarity: $x = 0 / 2 = 0.0$ and $y = 0 / 2 = 0.0$.
- Test Primal Feasibility:
  $$g(0, 0) = 1.0 - 0.0 - 0.0 = +1.0 > 0 \quad (\mathbf{VIOLATION!})$$
- The unconstrained minimum is not feasible. Therefore, the constraint **must be active ($\lambda > 0$)**!

**Case B: $\lambda > 0$ (Active Constraint Hypothesis)**
- By complementary slackness, since $\lambda \ne 0$, we must have:
  $$1.0 - x - y = 0 \implies x + y = 1.0$$
- Substitute $x = \lambda / 2$ and $y = \lambda / 2$:
  $$\frac{\lambda}{2} + \frac{\lambda}{2} = 1.0 \implies \mathbf{\lambda = 1.0000}$$
- Check dual feasibility: $\lambda = 1.0 \ge 0 \quad \checkmark$
- Compute optimal primal coordinates:
  $$\mathbf{x^* = \frac{1.0}{2} = 0.5000}, \quad \mathbf{y^* = \frac{1.0}{2} = 0.5000}$$
- Optimal objective value:
  $$\mathbf{f(x^*, y^*) = 0.5000^2 + 0.5000^2 = 0.2500 + 0.2500 = 0.5000}$$

---

### 4. Step-by-Step Grid Evaluation of Candidate Points

Let us compare candidate points to verify that $(0.5, 0.5)$ is the true constrained minimum:

```
┌──────────────────┬──────────┬──────────┬─────────────┬─────────────┬───────────────────────────┐
│ Candidate Point  │ x        │ y        │ f(x,y)=x²+y²│ g(x,y)=1-x-y│ Feasibility Status        │
├──────────────────┼──────────┼──────────┼─────────────┼─────────────┼───────────────────────────┤
│ Origin (0, 0)    │  0.0000  │  0.0000  │   0.0000    │   +1.0000   │ INFEASIBLE (g > 0) ✕      │
│ Point (1.0, 0.0) │  1.0000  │  0.0000  │   1.0000    │    0.0000   │ Feasible (Boundary)       │
│ Point (0.2, 0.8) │  0.2000  │  0.8000  │   0.6800    │    0.0000   │ Feasible (Boundary)       │
│ Point (0.8, 0.8) │  0.8000  │  0.8000  │   1.2800    │   -0.6000   │ Feasible (Interior)       │
│ KKT Point (½, ½) │  0.5000  │  0.5000  │   0.5000    │    0.0000   │ OPTIMAL MINIMUM! 🎯       │
└──────────────────┴──────────┴──────────┴─────────────┴─────────────┴───────────────────────────┘
```

Notice: Every other feasible point has $f(x, y) \ge 0.5000$. The KKT point $(0.5, 0.5)$ achieves the strictly lowest loss among all feasible candidates!

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Support Vector Machines (Hard-Margin SVM)
Given linearly separable training data $\{(x_i, y_i)\}_{i=1}^N$ with $y_i \in \{-1, +1\}$:
$$\min_{w, b} \frac{1}{2}\|w\|_2^2 \quad \text{subject to } 1 - y_i(w^T x_i + b) \le 0 \,\, \forall i$$
The Lagrangian is:
$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|_2^2 + \sum_{i=1}^N \alpha_i [1 - y_i(w^T x_i + b)]$$
KKT Complementary Slackness:
$$\mathbf{\alpha_i^* \left[ 1 - y_i(w^{*T} x_i + b^*) \right] = 0 \quad \forall i}$$
- For samples strictly beyond the margin ($y_i(w^T x_i + b) > 1$):
  The constraint is slack $\implies \mathbf{\alpha_i^* = 0}$!
  These points have **zero influence** on the decision boundary.
- Only the difficult points lying directly on the margin ($y_i(w^T x_i + b) = 1$) have $\mathbf{\alpha_i^* > 0}$.
  These critical points are the **Support Vectors**! The entire boundary is defined by:
  $$w^* = \sum_{i \in \text{SV}} \alpha_i^* y_i x_i$$

---

### Illustration 2 (Boundary): Trust Region Policy Optimization (TRPO)
In policy gradient RL, updating policy parameters $\theta$ too far in one step can cause policy collapse.
TRPO solves the constrained optimization problem:
$$\max_\theta \quad g^T (\theta - \theta_{\text{old}}) \quad \text{subject to} \quad \frac{1}{2}(\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}}) \le \delta$$
where $F$ is the Fisher Information Matrix and $\delta$ is the trust-region radius.
By applying KKT stationarity:
$$g - \lambda F (\theta - \theta_{\text{old}}) = 0 \implies \theta - \theta_{\text{old}} = \frac{1}{\lambda} F^{-1} g$$
Substitute into the boundary constraint to solve for multiplier $\lambda^*$:
$$\lambda^* = \sqrt{\frac{g^T F^{-1} g}{2\delta}} \implies \mathbf{\Delta \theta^* = \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g}$$
This analytical KKT solution provides the exact TRPO step size!

---

### Illustration 3 (Edge Case): Projected Gradient Descent (PGD) on Convex Sets
What if the constraint set $\mathcal{C}$ is complex, and solving the KKT equations analytically is intractable?
We use **Projected Gradient Descent (PGD)**:
$$x_{t+1} = \Pi_{\mathcal{C}}\left( x_t - \eta \nabla f(x_t) \right)$$
where $\Pi_{\mathcal{C}}(v) = \arg\min_{z \in \mathcal{C}} \|z - v\|_2$ is the Euclidean projection onto the convex set $\mathcal{C}$.
In adversarial robustness, for an $L_\infty$ attack with budget $\epsilon$:
$$\Pi_{[-\epsilon, \epsilon]}(\delta) = \text{clip}(\delta, -\epsilon, +\epsilon)$$
The projection simply clamps each pixel perturbation between $-\epsilon$ and $+\epsilon$!

---

## Part 7: Deep Learning Connection & Application

### 1. Spectral Normalization in GANs (Miyato et al., 2018)
To stabilize Wasserstein GANs, the discriminator must be a 1-Lipschitz function.
This requires constraining the spectral norm (maximum singular value) of each weight matrix:
$$\|W\|_2 = \sigma_{\max}(W) \le 1$$
Instead of solving a complex constrained optimization during backpropagation, Spectral Normalization dynamically scales the weight matrix at every forward pass:
$$W_{\text{SN}} = \frac{W}{\sigma_{\max}(W)}$$
where $\sigma_{\max}(W)$ is estimated in 1 step using the **Power Iteration method**:
$$v \leftarrow \frac{W^T u}{\|W^T u\|_2}, \quad u \leftarrow \frac{W v}{\|W v\|_2}, \quad \sigma_{\max}(W) \approx u^T W v$$
This enforces the KKT boundary constraint implicitly and effortlessly!

---

## Part 8: Code Implementation & Verification

The companion Python module [03_constrained_optimization_and_kkt.py](./code/03_constrained_optimization_and_kkt.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact calculations of $(x^*, y^*) = (0.5000, 0.5000)$, $\lambda^* = 1.0000$, and $f = 0.5000$.
2. **SciPy SLSQP Constrained Solver Comparison:** Confirms that SciPy's Sequential Least Squares Programming solver converges to the exact analytical KKT solution within $10^{-9}$.
3. **Hard-Margin SVM Dual Quadratic Program:** Implements the dual SVM optimization problem using quadratic programming, confirming complementary slackness $\alpha_i [y_i(w^T x_i + b) - 1] = 0$ on toy 2D classification data.
4. **Projected Gradient Descent (PGD) on $L_2$ Ball:** Simulates PGD projecting steps back onto the ball $\|x\|_2 \le R$, verifying that the trajectory hugs the boundary where $\nabla f + \lambda x = 0$.
