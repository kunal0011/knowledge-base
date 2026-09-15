# Chapter 5.2: Unconstrained Optimization, Stationary Points & Saddle Points

---

## Part 1: Intuition & 101 Motivation

In deep learning, almost all training takes place as an **unconstrained optimization problem**:
$$\min_{\theta \in \mathbb{R}^P} \mathcal{L}(\theta)$$
where $P$ can range from thousands to hundreds of billions of parameters.

For decades, the machine learning community believed that the primary barrier to training deep neural networks was getting trapped in **bad local minima** (sub-optimal valleys with high loss).

In 2014, researchers made a revolutionary discovery that transformed our understanding of deep learning theory (Dauphin et al., 2014; Choromanska et al., 2015):
> **In high-dimensional spaces, local minima are not the problem. Saddle points are.**

Consider a critical point in $P = 1{,}000{,}000$ dimensions where $\nabla \mathcal{L}(\theta) = \mathbf{0}$.
For this critical point to be a local minimum, **all 1,000,000 eigenvalues of the Hessian matrix must be simultaneously positive** ($\lambda_i > 0$).
If we assume each eigenvalue has a rough 50% chance of being positive or negative:
$$P(\text{Local Minimum}) \approx \left(\frac{1}{2}\right)^{1{,}000{,}000} \approx 0$$
The probability of encountering a true local minimum at high loss is virtually zero!
Instead, the high-dimensional landscape is teeming with **Saddle Points**—critical points where the gradient vanishes, but some directions curve upward while other directions curve downward.

```
                  Curvature Along North-South (Valley, λ_1 > 0)
                                     ▲
                                     │
                        ╭────────────┴────────────╮
                      ╭─╯                         ╰─╮
                   .──┴─────────────────────────────┴──.
                  /                                     \
                 /          ● SADDLE POINT (0, 0)        \
                /           (∇f = 0, Indefinite H)        \
               '───────────────────────────────────────────'
                                     │
                                     ▼
                  Curvature Along East-West (Ridge, λ_2 < 0)
```

Understanding how optimizers navigate stationary points, escape saddles, and identify flat, generalizing minima is the essential prerequisite for modern optimization.

---

## Part 2: Rigorous Mathematical Formulation

Let $f: \mathbb{R}^n \to \mathbb{R}$ be twice continuously differentiable ($\mathcal{C}^2$). We seek to minimize $f(x)$ over unconstrained $\mathbb{R}^n$.

### 1. First-Order Necessary Condition (FONC)
**Theorem:** If $x^*$ is a local minimizer of $f$, then $x^*$ must be a **stationary point** (critical point):
$$\mathbf{\nabla f(x^*) = \mathbf{0}}$$

**Proof (by Contradiction):**
Suppose $\nabla f(x^*) \ne \mathbf{0}$. Choose search direction $d = -\nabla f(x^*)$.
By Taylor's theorem:
$$f(x^* + \alpha d) = f(x^*) + \alpha \nabla f(x^*)^T d + o(\alpha) = f(x^*) - \alpha \|\nabla f(x^*)\|_2^2 + o(\alpha)$$
For sufficiently small $\alpha > 0$, $-\alpha \|\nabla f(x^*)\|_2^2 + o(\alpha) < 0$, which implies $f(x^* + \alpha d) < f(x^*)$, contradicting that $x^*$ is a local minimum. Hence $\nabla f(x^*) = \mathbf{0}$. $\blacksquare$

---

### 2. Second-Order Necessary & Sufficient Conditions

#### Second-Order Necessary Condition (SONC):
If $x^*$ is a local minimizer of $f$, then the Hessian matrix must be **Positive Semi-Definite (PSD)**:
$$\mathbf{\nabla^2 f(x^*) \succeq 0}$$
All eigenvalues of the Hessian must satisfy $\lambda_i \ge 0$.

#### Second-Order Sufficient Condition (SOSC):
If $x^*$ satisfies:
1. $\nabla f(x^*) = \mathbf{0}$ (Stationary), and
2. $\nabla^2 f(x^*) \succ 0$ (Strictly Positive Definite, all $\lambda_i > 0$),
then $x^*$ is a **strict local minimizer**.

---

### 3. Rigorous Classification of Critical Points via the Hessian Eigenspectrum
Let $x^*$ be a critical point ($\nabla f(x^*) = \mathbf{0}$), and let $\mathcal{H} = \nabla^2 f(x^*)$ be the symmetric Hessian matrix with eigenvalues $\lambda_1 \le \lambda_2 \le \dots \le \lambda_n$:

| Condition on Eigenvalues | Definiteness of Hessian | Classification of $x^*$ |
| :--- | :--- | :--- |
| $\lambda_i > 0 \,\, \forall i$ | Positive Definite ($\mathcal{H} \succ 0$) | **Strict Local Minimum** |
| $\lambda_i < 0 \,\, \forall i$ | Negative Definite ($\mathcal{H} \prec 0$) | **Strict Local Maximum** |
| $\exists \lambda_i > 0$ and $\exists \lambda_j < 0$ | Indefinite ($\mathcal{H}$ indefinite) | **Strict Saddle Point** |
| At least one $\lambda_i = 0$, all others $\ge 0$ | Positive Semi-Definite ($\det(\mathcal{H}) = 0$) | **Degenerate Valley / Inconclusive** |
| All $\lambda_i = 0$ ($\mathcal{H} = \mathbf{0}$) | Zero Matrix | **Higher-Order Monkey Saddle** |

---

### 4. The Strict Saddle Property & The Center Stable Manifold Theorem

#### Definition (Strict Saddle Property):
A function $f(x)$ satisfies the **strict saddle property** if every critical point $x^*$ is either a local minimum or a strict saddle point with minimum eigenvalue bounded away from zero:
$$\lambda_{\min}(\nabla^2 f(x^*)) \le -\gamma < 0$$

#### Theorem (Lee et al., 2016 - Center Stable Manifold Theorem):
Let $f$ be twice continuously differentiable and satisfy the strict saddle property.
If Gradient Descent is initialized randomly:
$$x_0 \sim P_{\text{init}} \quad (\text{absolutely continuous w.r.t. Lebesgue measure})$$
then the set of initializations that converge to a strict saddle point has **measure zero**:
$$\mathbf{P\left( \lim_{t \to \infty} x_t = x_{\text{saddle}} \right) = 0}$$
**Gradient descent with random initialization almost surely escapes all strict saddle points!**

---

## Part 3: Geometric & Algebraic Interpretation

### The Quadratic Form Expansion Around a Saddle Point
Let $x^*$ be a saddle point ($\nabla f(x^*) = \mathbf{0}$). For any small displacement $\Delta x$:
$$f(x^* + \Delta x) \approx f(x^*) + \frac{1}{2} \Delta x^T \mathcal{H} \Delta x$$
By the Spectral Theorem, write $\mathcal{H} = \sum_{i=1}^n \lambda_i v_i v_i^T$ where $v_i$ are orthonormal eigenvectors:
$$f(x^* + \Delta x) \approx f(x^*) + \frac{1}{2} \sum_{i=1}^n \lambda_i (v_i^T \Delta x)^2$$

```
                   Hessian Eigenspace Decomposition
                                 v_1 (λ_1 > 0: Valley Direction)
                                  ▲
                                  │   Curvature Upward
                                  │   (f increases)
                                  │
                                  └───────────────► v_2 (λ_2 < 0: Escape Direction)
                                      Curvature Downward
                                      (f decreases! Highway out of saddle!)
```

1. **The Trapping Illusion:**
   At $x^*$, $\|\nabla f(x^*)\| = 0$. If you stand precisely at $x^*$, vanilla gradient descent takes zero step size: $\Delta x = -\eta (\mathbf{0}) = \mathbf{0}$.
2. **The Escape Direction:**
   Any infinitesimal perturbation $\epsilon v_2$ along the negative eigenvector direction ($v_2$ where $\lambda_2 < 0$) produces:
   $$f(x^* + \epsilon v_2) \approx f(x^*) + \frac{1}{2} \lambda_2 \epsilon^2 < f(x^*)$$
   The loss drops below the saddle point! The gradient along $v_2$ becomes non-zero:
   $$\nabla f(x^* + \epsilon v_2) \approx \lambda_2 \epsilon v_2$$
   Gradient descent accelerates exponentially away from the saddle point along $v_2$!

---

## Part 4: Real-World Analogy

### The Mountain Pass (Col)
Imagine you are hiking across the Alps:
- You reach a **Mountain Pass** (a col) between two majestic peaks.
- If you look North or South toward the peaks, the ground slopes steeply upward ($\lambda_1 > 0$). You are at the *lowest point* of the ridge connecting the two peaks.
- If you look East or West toward the green valleys, the ground slopes steeply downward ($\lambda_2 < 0$). You are at the *highest point* of the trail connecting the two valleys.
- Right at the pass marker, the ground under your boots is completely level ($\nabla f = \mathbf{0}$).
- If you close your eyes and stand completely still, you remain at the pass. But if a gentle gust of wind (SGD mini-batch noise) nudges you one inch to the East, gravity seizes you and rolls you rapidly down into the lush valley below!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us analyze all stationary points and trace gradient descent escaping a saddle point by hand for the 2D function:
$$f(x, y) = x^2 - y^2 + \frac{1}{4} y^4$$

### 1. Problem Setup & Analytic Critical Points
1. **Gradient Vector:**
   $$\nabla f(x, y) = \left[\begin{matrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{matrix}\right] = \left[\begin{matrix} 2x \\ -2y + y^3 \end{matrix}\right] = \left[\begin{matrix} 2x \\ y(y^2 - 2) \end{matrix}\right] = \left[\begin{matrix} 0 \\ 0 \end{matrix}\right]$$
   - $2x = 0 \implies x = 0$.
   - $y(y^2 - 2) = 0 \implies y = 0, \,\, y = +\sqrt{2}, \,\, y = -\sqrt{2}$.

There are **three distinct critical points**:
- **Point A:** $(0, 0)$
- **Point B:** $(0, +\sqrt{2}) \approx (0, 1.4142)$
- **Point C:** $(0, -\sqrt{2}) \approx (0, -1.4142)$

2. **Hessian Matrix:**
   $$\mathcal{H}(x, y) = \left[\begin{matrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{matrix}\right] = \left[\begin{matrix} 2 & 0 \\ 0 & -2 + 3y^2 \end{matrix}\right]$$

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $(x, y)$ | 2D parameter coordinates | Weights $(w_1, w_2)$ | Float pair | $(x_t, y_t)$ |
| $f(x, y)$ | Objective function value | Training loss $\mathcal{L}(\theta)$ | Float scalar | $x^2 - y^2 + 0.25 y^4$ |
| $\nabla f$ | Gradient vector $[\partial_x, \partial_y]^T$ | Backprop gradient | Float 2-vector | $[2x, \,\, -2y + y^3]^T$ |
| $\mathcal{H}$ | 2x2 Hessian matrix | Local loss curvature | Matrix $\mathbb{R}^{2 \times 2}$ | $\text{diag}(2, \,\, -2 + 3y^2)$ |
| $\lambda_1, \lambda_2$ | Eigenvalues of Hessian | Principal curvatures | Float scalars | Evaluated at points A, B, C |
| $\eta$ | Learning rate | Gradient descent step size | Float scalar | $0.10$ |

---

### 3. Step-by-Step Critical Point Classification

#### Critical Point A: $(0, 0)$
$$\mathcal{H}(0, 0) = \left[\begin{matrix} 2 & 0 \\ 0 & -2 \end{matrix}\right]$$
- Eigenvalues: $\lambda_1 = +2.0, \quad \lambda_2 = -2.0$.
- One positive, one negative $\implies$ **STRICT SADDLE POINT!**
- Function value: $f(0, 0) = 0.0000$.

#### Critical Point B: $(0, \sqrt{2})$
$$\mathcal{H}(0, \sqrt{2}) = \left[\begin{matrix} 2 & 0 \\ 0 & -2 + 3(2) \end{matrix}\right] = \left[\begin{matrix} 2 & 0 \\ 0 & 4 \end{matrix}\right]$$
- Eigenvalues: $\lambda_1 = +2.0, \quad \lambda_2 = +4.0$.
- Both strictly positive $\implies$ **STRICT LOCAL MINIMUM!**
- Function value: $f(0, \sqrt{2}) = 0 - 2 + \frac{1}{4}(4) = -2 + 1 = \mathbf{-1.0000}$.

#### Critical Point C: $(0, -\sqrt{2})$
$$\mathcal{H}(0, -\sqrt{2}) = \left[\begin{matrix} 2 & 0 \\ 0 & 4 \end{matrix}\right] \implies \lambda_1 = +2.0, \,\, \lambda_2 = +4.0 \implies \mathbf{\text{STRICT LOCAL MINIMUM!}}$$
- Function value: $f(0, -\sqrt{2}) = \mathbf{-1.0000}$.

---

### 4. Step-by-Step Gradient Descent Trajectory Escaping the Saddle Point
Let learning rate $\eta = 0.10$.
Initialize near the saddle point with a tiny perturbation: $(x_0, y_0) = (0.20, 0.10)$.
Notice: $x_0$ is perturbed along the valley direction ($\lambda_1 = +2$), and $y_0$ is perturbed along the escape direction ($\lambda_2 = -2$).

#### Step 1:
- Function value: $f(0.2, 0.1) = 0.2^2 - 0.1^2 + 0.25(0.1^4) = 0.04 - 0.01 + 0.000025 = \mathbf{0.030025}$
- Gradient:
  $$\nabla f_x = 2(0.20) = \mathbf{+0.4000}$$
  $$\nabla f_y = -2(0.10) + (0.10)^3 = -0.20 + 0.0010 = \mathbf{-0.1990}$$
- Parameter update:
  $$x_1 = x_0 - \eta \nabla f_x = 0.20 - 0.10(0.40) = 0.20 - 0.04 = \mathbf{0.1600}$$
  $$y_1 = y_0 - \eta \nabla f_y = 0.10 - 0.10(-0.1990) = 0.10 + 0.0199 = \mathbf{0.1199}$$

#### Step 2:
- Gradient at $(0.1600, 0.1199)$:
  $$\nabla f_x = 2(0.1600) = \mathbf{+0.3200}$$
  $$\nabla f_y = -2(0.1199) + (0.1199)^3 = -0.2398 + 0.0017 = \mathbf{-0.2381}$$
- Parameter update:
  $$x_2 = 0.1600 - 0.10(0.3200) = \mathbf{0.1280} \quad (\text{Contracting toward } 0!)$$
  $$y_2 = 0.1199 - 0.10(-0.2381) = 0.1199 + 0.0238 = \mathbf{0.1437} \quad (\text{Accelerating toward } \sqrt{2}!)$$

#### Step 3:
- Gradient at $(0.1280, 0.1437)$:
  $$\nabla f_x = 2(0.1280) = \mathbf{+0.2560}$$
  $$\nabla f_y = -2(0.1437) + (0.1437)^3 = -0.2874 + 0.0030 = \mathbf{-0.2844}$$
- Parameter update:
  $$x_3 = 0.1280 - 0.10(0.2560) = \mathbf{0.1024}$$
  $$y_3 = 0.1437 - 0.10(-0.2844) = \mathbf{0.1721}$$

---

### 5. Visual Summary Grid

```
┌──────┬──────────┬──────────┬─────────────┬─────────────┬──────────────┬───────────────────────────────┐
│ Iter │ Param x  │ Param y  │ Grad ∇f_x   │ Grad ∇f_y   │ Loss f(x, y) │ Dynamics Behavior             │
├──────┼──────────┼──────────┼─────────────┼─────────────┼──────────────┼───────────────────────────────┤
│ t=0  │  0.2000  │  0.1000  │   +0.4000   │   -0.1990   │   +0.0300    │ Initial perturbation          │
│ t=1  │  0.1600  │  0.1199  │   +0.3200   │   -0.2381   │   +0.0114    │ x contracts, y expands        │
│ t=2  │  0.1280  │  0.1437  │   +0.2560   │   -0.2844   │   -0.0040    │ Loss turns negative!          │
│ t=3  │  0.1024  │  0.1721  │   +0.2048   │   -0.3391   │   -0.0189    │ Rapid acceleration along y    │
├──────┴──────────┴──────────┴─────────────┴─────────────┴──────────────┴───────────────────────────────┤
│ As t -> inf: x -> 0.0000, y -> +sqrt(2) = 1.4142, Loss -> -1.0000 (Global Minimum reached!)          │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Notice the beautiful dynamics:
- Along the $x$-axis ($\lambda_1 = +2$), the gradient pushes $x \to 0$.
- Along the $y$-axis ($\lambda_2 = -2$), the negative curvature propels $y$ away from the saddle, rapidly escaping toward the global minimum at $y = \sqrt{2}$!

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): The Rosenbrock "Banana" Valley
The Rosenbrock function is the universal benchmark for optimization algorithms:
$$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$
- Unique global minimum at $(1, 1)$ where $f(1, 1) = 0$.
- Let us evaluate the Hessian at the global minimum $(1, 1)$:
  $$\mathcal{H}(1, 1) = \left[\begin{matrix} 802 & -400 \\ -400 & 200 \end{matrix}\right]$$
  Eigenvalues: $\lambda_1 \approx 0.4$, $\lambda_2 \approx 1001.6$.
  Condition number: $\kappa = \frac{1001.6}{0.4} \approx \mathbf{2500}$!
- Because the condition number is so extreme, the valley is shaped like a curved, razor-thin canyon. Standard gradient descent bounces back and forth between the steep walls thousands of times, while **Momentum** and **Adam** glide smoothly along the valley floor!

---

### Illustration 2 (Boundary): The Monkey Saddle (Degenerate Critical Point)
Consider the function:
$$f(x, y) = x^3 - 3 x y^2$$
1. **Gradient:**
   $$\nabla f(x, y) = \left[\begin{matrix} 3x^2 - 3y^2 \\ -6xy \end{matrix}\right] \implies \nabla f(0, 0) = \left[\begin{matrix} 0 \\ 0 \end{matrix}\right]$$
2. **Hessian Matrix:**
   $$\mathcal{H}(x, y) = \left[\begin{matrix} 6x & -6y \\ -6y & -6x \end{matrix}\right] \implies \mathcal{H}(0, 0) = \left[\begin{matrix} 0 & 0 \\ 0 & 0 \end{matrix}\right]$$
- All eigenvalues are zero: $\lambda_1 = 0, \lambda_2 = 0$!
- The Second-Order Sufficient Condition is **completely inconclusive**.
- In polar coordinates: $f(r, \theta) = r^3 \cos(3\theta)$.
- The surface has **three ascending lobes and three descending valleys** (two for the monkey's legs, and one for its tail!). Second-order Hessian methods fail completely on degenerate saddles.

---

### Illustration 3 (Edge Case): Flat Saddle Points in Deep Networks (Vanishing Gradients)
In deep networks with Sigmoid activations $\sigma(z) = \frac{1}{1 + e^{-z}}$, when weights become large, $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \to 0$.
Both the gradient and the Hessian vanish:
$$\|\nabla \mathcal{L}\| \to 0 \quad \text{and} \quad \|\nabla^2 \mathcal{L}\| \to 0$$
This creates a **flat plateau / higher-order saddle point**.
Standard Gradient Descent stalls completely because the step size $-\eta \nabla \mathcal{L} \approx \mathbf{0}$. This is why deep learning replaced Sigmoid with **ReLU / GELU** and introduced **Residual Connections (ResNets)**!

---

## Part 7: Deep Learning Connection & Application

### 1. Perturbed Stochastic Gradient Descent Escapes Saddle Points
Why doesn't SGD get stuck at saddle points in practice?
Ge et al. (2015) and Jin et al. (2017) proved that adding noise to the gradient guarantees polynomial-time escape from all strict saddle points:
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \sigma^2 I)$$
The thermal noise $\xi_t$ kicks the optimizer in all directions. As soon as a kick points slightly along the negative eigenvalue eigenvector $v_{\min}$ ($\lambda_{\min} < 0$), the negative curvature takes over and propels the parameters downhill!
In practice, **the stochasticity of mini-batch sampling naturally provides this exact noise!**

---

### 2. Flat Minima vs. Sharp Minima & SAM (Sharpness-Aware Minimization)
Not all local minima are created equal:
- **Sharp Minimum:** The loss has massive curvature ($\lambda_{\max}(\mathcal{H}) \gg 10^3$). A microscopic shift between training distribution and test distribution causes test error to skyrocket!
- **Flat Minimum:** The loss basin is broad and flat ($\lambda_{\max}(\mathcal{H})$ is small). Shifts in test data produce virtually zero change in test loss!

```
         Sharp Minimum (Overfitting)                  Flat Minimum (Generalizing)
     Loss ▲                                       Loss ▲
          │      \   /                                 │    \                     /
          │       \ /   ◄── High Test Error!           │     \                   /  ◄── Low Test Error!
          │        ●                                   │      ╰───────●─────────╯
          └────────────────────────► Weight            └────────────────────────► Weight
```

Foret et al. (2021) introduced **Sharpness-Aware Minimization (SAM)**, which explicitly seeks flat minima by optimizing:
$$\min_\theta \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}(\theta + \epsilon)$$
SAM is now a staple technique for state-of-the-art vision models and language model fine-tuning!

---

## Part 8: Code Implementation & Verification

The companion Python module [02_unconstrained_optimization_and_saddles.py](./code/02_unconstrained_optimization_and_saddles.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact calculations of critical points A, B, C, Hessian eigenvalues, and the first 3 gradient descent steps escaping the saddle point.
2. **Saddle Point Escape Simulation:** Compares vanilla GD starting exactly on a saddle vs. Perturbed SGD escaping along the negative eigenvalue direction.
3. **Rosenbrock Banana Optimization:** Analyzes Hessian eigenspectrum and condition number $\kappa \approx 2500$ at the minimum.
4. **Monkey Saddle Higher-Order Inconclusive Test:** Confirms $\mathcal{H}(0, 0) = \mathbf{0}$ and verifies 3-fold cyclic valley structure.
5. **Sharpness-Aware Minimization (SAM) PyTorch Step:** Implements the SAM adversarial perturbation step $\epsilon^* = \rho \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|_2}$ and verifies flatness penalization.
