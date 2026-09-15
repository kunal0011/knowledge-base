# 5.7 Second-Order Optimization & Natural Gradients (Newton, BFGS, Fisher Information)

---

## Part 1: Intuition & 101 Motivation

Throughout Chapters 5.4, 5.5, and 5.6, we focused on **first-order optimization methods**. First-order methods rely solely on the gradient $\nabla f(\theta)$, which represents the linear tangent hyperplane of the loss surface. Because a linear tangent has zero curvature, first-order algorithms are fundamentally blind to how rapidly the landscape curves, forcing them to use cautious, manually tuned step sizes or heuristic coordinate-wise dampening.

**Second-order optimization methods** eliminate this blindness by measuring both the slope (gradient $\nabla f$) and the local curvature (Hessian matrix $\mathcal{H} = \nabla^2 f$). Instead of approximating the loss with a flat tangent plane, second-order methods approximate the objective with a local **osculating paraboloid** (a multidimensional quadratic bowl):
$$f(\theta + \Delta \theta) \approx f(\theta) + \nabla f(\theta)^T \Delta \theta + \frac{1}{2} \Delta \theta^T \mathcal{H} \Delta \theta$$

By setting the derivative of this quadratic model to zero, **Newton's Method** jumps directly to the vertex of the bowl in a single step:
$$\Delta \theta = -\mathcal{H}^{-1} \nabla f(\theta)$$

```
First-Order (Linear Model) vs. Second-Order (Quadratic Model):

    Loss f(x)
       |          , - - - - - - .  <-- True Non-Convex Landscape
       |        /   \          /
       |       /     \        /
       |      /  Tangent Plane \   <-- First-Order GD: Flat slope,
       |     /   (Blind to curve)      blind to canyon depth.
       |    /                     \
       |   |      Osculating Bowl  | <-- Second-Order Newton:
       |    \    (Matches curvature)    Jumps directly to vertex!
       |     ` - . _ _ _ _ . - '
       +---------------------------------------------> Parameter Space
```

If Newton's method is mathematically so powerful, **why does modern deep learning almost exclusively run on first-order methods (SGD, AdamW)?**
1. **Memory Catastrophe**: The Hessian has size $d \times d$. For a modern 7-billion parameter language model, $\mathcal{H}$ contains $4.9 \times 10^{19}$ elements. Storing it in 32-bit floats requires **196 Exabytes of VRAM**—more than the combined memory of all GPUs on Earth.
2. **Computational Catastrophe**: Solving $\mathcal{H}^{-1} \nabla f$ requires matrix inversion or Cholesky decomposition, taking $\mathcal{O}(d^3)$ FLOPS.
3. **The Non-Convex Saddle Trap**: In high-dimensional deep learning landscapes, most stationary points are saddle points ($\lambda < 0$). Standard Newton's method actively seeks points where $\nabla f = 0$, jumping *straight into saddle points and local maxima*!

To make curvature information computationally viable, researchers developed:
- **Quasi-Newton Methods (BFGS, L-BFGS)**: Sequentially approximate $\mathcal{H}^{-1}$ using only gradient differences without ever inverting a matrix.
- **Hessian-Free Optimization (Pearlmutter's Trick)**: Computes the exact directional curvature $\mathcal{H} v$ in $\mathcal{O}(d)$ time and memory using automatic differentiation.
- **Natural Gradient Descent (Amari 1998)**: Replaces the arbitrary Euclidean parameter metric with the **Fisher Information Matrix**, defining distance along the Riemannian manifold of probability distributions.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Classical Newton's Method

Let $f: \mathbb{R}^d \to \mathbb{R}$ be twice continuously differentiable ($C^2$). Consider the second-order Taylor expansion around the current iterate $\theta_t$:
$$f(\theta_t + \Delta \theta) \approx q(\Delta \theta) = f(\theta_t) + \nabla f(\theta_t)^T \Delta \theta + \frac{1}{2} \Delta \theta^T \nabla^2 f(\theta_t) \Delta \theta$$
where $\mathcal{H}_t = \nabla^2 f(\theta_t) \in \mathbb{R}^{d \times d}$ is the symmetric Hessian matrix.

Differentiating the quadratic model $q(\Delta \theta)$ with respect to $\Delta \theta$ and setting it to zero:
$$\nabla_{\Delta \theta} q(\Delta \theta) = \nabla f(\theta_t) + \mathcal{H}_t \Delta \theta = 0$$
$$\mathcal{H}_t \Delta \theta = -\nabla f(\theta_t) \implies \Delta \theta = -\mathcal{H}_t^{-1} \nabla f(\theta_t)$$

The classical Newton-Raphson update is:
$$\theta_{t+1} = \theta_t - \left[ \nabla^2 f(\theta_t) \right]^{-1} \nabla f(\theta_t)$$

#### Local Quadratic Convergence Theorem
If $f$ is strongly convex ($\mathcal{H} \succeq \mu I > 0$) with Lipschitz continuous Hessian ($\|\nabla^2 f(x) - \nabla^2 f(y)\| \le L_2 \|x - y\|$), and $\theta_0$ is within a radius $r$ of the local minimum $\theta^*$, then Newton's method converges **quadratically**:
$$\|\theta_{t+1} - \theta^*\| \le \frac{L_2}{2 \mu} \|\theta_t - \theta^*\|^2$$

The number of correct decimal digits **doubles at every iteration**!

---

### 2. Quasi-Newton Methods & The BFGS Algorithm

Quasi-Newton methods avoid computing or inverting the Hessian. Instead, they maintain a running estimate of the inverse Hessian $H_t \approx \mathcal{H}_t^{-1}$ that satisfies the **Secant Equation**:

Define parameter displacement $s_t$ and gradient change $y_t$:
$$s_t = \theta_{t+1} - \theta_t, \quad y_t = \nabla f(\theta_{t+1}) - \nabla f(\theta_t)$$

By Taylor's theorem, $\nabla f(\theta_{t+1}) - \nabla f(\theta_t) \approx \mathcal{H}_{t+1} (\theta_{t+1} - \theta_t)$, yielding the Secant Equation:
$$B_{t+1} s_t = y_t \iff H_{t+1} y_t = s_t$$
where $B \approx \mathcal{H}$ and $H \approx \mathcal{H}^{-1}$.

#### The BFGS Update (Broyden-Fletcher-Goldfarb-Shanno)
BFGS is the most robust rank-2 quasi-Newton update. It updates the inverse Hessian estimate $H_t$ directly:
$$H_{t+1} = (I - \rho_t s_t y_t^T) H_t (I - \rho_t y_t s_t^T) + \rho_t s_t s_t^T$$
where scalar $\rho_t = \frac{1}{y_t^T s_t}$.
- **Guaranteed Positive Definiteness**: If the step satisfies the Wolfe curvature condition ($y_t^T s_t > 0$), and $H_t \succ 0$, then $H_{t+1} \succ 0$ is strictly guaranteed!

#### L-BFGS (Limited-Memory BFGS)
In high dimensions, storing $H_t \in \mathbb{R}^{d \times d}$ is still prohibitive. **L-BFGS** does not store $H_t$. Instead, it stores only the last $m$ displacement pairs:
$$\mathcal{S} = \{(s_{t-1}, y_{t-1}), (s_{t-2}, y_{t-2}), \dots, (s_{t-m}, y_{t-m})\}$$
typically with memory size $m \in [5, 20]$.

The direction $r = -H_t \nabla f(\theta_t)$ is computed in $\mathcal{O}(m d)$ operations via the celebrated **two-loop recursion**:
```
Algorithm: L-BFGS Two-Loop Recursion
Input: Gradient g = \nabla f(\theta_t), History {(s_i, y_i)}_{i=t-m}^{t-1}
1. q = g
2. For i = t - 1 down to t - m:
       alpha_i = rho_i * (s_i^T q)
       q = q - alpha_i * y_i
3. gamma_k = (s_{t-1}^T y_{t-1}) / (y_{t-1}^T y_{t-1})
4. r = gamma_k * q
5. For i = t - m up to t - 1:
       beta = rho_i * (y_i^T r)
       r = r + s_i * (alpha_i - beta)
Output: Search direction Delta \theta = -r
```

---

### 3. Hessian-Free Optimization & Hessian-Vector Products

Can we compute the exact action of the Hessian on an arbitrary vector $v \in \mathbb{R}^d$ without ever instantiating the matrix $\mathcal{H}$?

#### Pearlmutter's $\mathcal{R}$-Operator (Pearlmutter 1994)
Notice that the directional derivative of the gradient along $v$ is exactly the Hessian-vector product:
$$\mathcal{H} v = \lim_{\epsilon \to 0} \frac{\nabla f(\theta + \epsilon v) - \nabla f(\theta)}{\epsilon} = \left. \frac{d}{d \epsilon} \nabla f(\theta + \epsilon v) \right|_{\epsilon = 0}$$

Using chain rule on the scalar inner product:
$$\mathcal{H} v = \nabla_\theta \left( \nabla f(\theta)^T v \right)$$

This requires **exactly two backward passes**:
1. First autograd pass: Compute gradient $g(\theta) = \nabla f(\theta)$.
2. Compute scalar dot-product: $\phi(\theta) = g(\theta)^T v$.
3. Second autograd pass: Compute $\nabla_\theta \phi(\theta) = \mathcal{H} v$.
- **Cost**: $\mathcal{O}(d)$ memory, $\sim 2\times$ the computation of a standard gradient!
- Paired with the **Conjugate Gradient (CG)** algorithm, this solves $\mathcal{H} \Delta \theta = -\nabla f$ iteratively without ever storing the Hessian.

---

### 4. Natural Gradient Descent (Amari 1998)

In deep learning, neural networks parameterize conditional probability distributions $p(y | x, \theta)$ (e.g., softmax for classification, Gaussian mean for regression).

Standard gradient descent defines the steepest descent direction under the **Euclidean metric**:
$$\Delta \theta_{\text{Euclidean}} = \arg\min_{\|\Delta \theta\|_2^2 \le \epsilon^2} \nabla f(\theta)^T \Delta \theta$$
However, Euclidean distance $\|\Delta \theta\|_2$ depends entirely on arbitrary parameterization! If you rescale a layer's weights by 10 and biases by 0.1, the physical neural network output is identical, but Euclidean distance changes radically.

#### The Statistical Manifold and KL Divergence
Amari realized that the true distance between two models is not the distance between their weights, but the distance between their **predicted output distributions**, measured by Kullback-Leibler (KL) divergence:
$$D_{\text{KL}}(p_\theta \parallel p_{\theta + \Delta \theta}) \approx \frac{1}{2} \Delta \theta^T F(\theta) \Delta \theta$$
where $F(\theta)$ is the **Fisher Information Matrix (FIM)**:
$$F(\theta) = \mathbb{E}_{x \sim p_{\text{data}}, y \sim p_\theta(y|x)}\left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^T \right]$$

#### The Natural Gradient Update
The steepest descent direction on the Riemannian manifold constrained by distribution shift is:
$$\Delta \theta_{\text{Nat}} = \arg\min_{\Delta \theta^T F \Delta \theta \le \epsilon^2} \nabla f(\theta)^T \Delta \theta = -\alpha F(\theta)^{-1} \nabla f(\theta)$$

- **Key Equivalence**: For negative log-likelihood loss $\mathcal{L}(\theta) = -\log p_\theta(y|x)$, the Fisher Information Matrix is the **expected Hessian**:
  $$F(\theta) = \mathbb{E}_{y \sim p_\theta}\left[ \nabla^2_\theta (-\log p_\theta(y|x)) \right]$$
  Thus, Natural Gradient Descent is an expected-Hessian second-order method that is **strictly positive semi-definite** ($F \succeq 0$), completely immune to the saddle point attraction that breaks Newton's method!

#### Kronecker-Factored Approximate Curvature (K-FAC)
For a dense layer $y = W x$, the gradient with respect to weight matrix $W$ is an outer product: $\nabla_W \mathcal{L} = s x^T$, where $s = \nabla_y \mathcal{L}$ is the output sensitivity.
The Fisher block for layer $W$ has size $d_1 d_2 \times d_1 d_2$:
$$F_W = \mathbb{E}\left[ (x \otimes s) (x \otimes s)^T \right] = \mathbb{E}\left[ (x x^T) \otimes (s s^T) \right]$$

Martens & Grosse (2015) introduced the Kronecker factorization approximation:
$$F_W \approx \mathbb{E}[x x^T] \otimes \mathbb{E}[s s^T] = A \otimes S$$

Using the Kronecker product inversion property $(A \otimes S)^{-1} = A^{-1} \otimes S^{-1}$:
$$F_W^{-1} \text{vec}(\nabla_W \mathcal{L}) = \text{vec}\left( S^{-1} (\nabla_W \mathcal{L}) A^{-1} \right)$$
Inverting $A \in \mathbb{R}^{d_1 \times d_1}$ and $S \in \mathbb{R}^{d_2 \times d_2}$ requires only $\mathcal{O}(d_1^3 + d_2^3)$ operations instead of $\mathcal{O}((d_1 d_2)^3)$!

---

## Part 3: Geometric Interpretation

### 1. Curvature Compensation on Quadratic Bowls
Consider a 2D quadratic bowl with non-axis-aligned contours:
$$f(x_1, x_2) = 5 x_1^2 + 2 x_1 x_2 + 2 x_2^2$$

```
            x_2
             ^               _ - - _
             |           _ '         ' _
             |         /                 \
             |       /      Iso-contours   \
             |      |       of f(x)         |
             |       \                     /
             |         \ _             _ /
             |             ' - - - - '
             +-------------------------------------> x_1
             |  \
             |   \  Standard Gradient: Points perpendicular to contour
             |    \ (misses center!)
             |     v
             |      \
             |       * Newton Step (-H^{-1} g): Exactly rotates and scales
             |         directly to the center (0, 0)!
```

- **Gradient Descent**: The negative gradient $-\nabla f$ points perpendicular to the iso-contour ellipse. On elongated ellipses, this direction points almost perpendicular to the true optimum!
- **Newton's Step**: The inverse Hessian $\mathcal{H}^{-1}$ acts as a metric tensor that stretches the steep axis, compresses the flat axis, and rotates coordinates into an isotropic space where the gradient points directly toward the center.

### 2. Euclidean vs. Riemannian Manifolds
- In Euclidean space, step size is measured in parameter coordinates: $\|\Delta \theta\|_2^2 = \sum_i \Delta \theta_i^2$. Moving $\Delta \theta = 1.0$ in a dead neuron changes nothing, but moving $\Delta \theta = 1.0$ in an active classifier completely breaks predictions.
- On the statistical manifold, step size is measured in **information divergence**: how much the probability distribution moves. Natural Gradient ensures that every step changes the model's behavior by an equal amount, regardless of parameter scaling.

---

## Part 4: Real-World Analogy

### 1. The Blind Hiker vs. The Topographical Radar
- **First-Order (GD)**: You are a hiker trapped in a thick mountain fog with only a carpenter's level on your shoes. You feel the slope under your boots and take a step downhill. In a steep canyon, you bounce between the canyon walls.
- **Second-Order (Newton)**: You carry a 3D LiDAR topographical scanner that maps the entire valley bowl around you. In a single scan, the computer calculates the exact geometric basin of the valley and teleports you directly to the bottom.
- **Natural Gradient**: You are a pilot navigating an aircraft across the surface of the Earth. If you use a flat 2D Mercator map (Euclidean parameter space), straight lines on the map are distorted, leading to wasted fuel. If you use spherical trigonometry (Fisher information metric), you follow the true great-circle geodesic route across the globe.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us perform a complete, hand-calculated trace of a **Full Newton Step** on a concrete 2D non-diagonal quadratic function, and compare it cell-by-cell with **Standard Gradient Descent**.

### Problem Setup
Objective function:
$$f(x_1, x_2) = 5 x_1^2 + 2 x_1 x_2 + 2 x_2^2$$
- Gradient vector:
  $$\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 10 x_1 + 2 x_2 \\ 2 x_1 + 4 x_2 \end{bmatrix}$$
- Hessian matrix:
  $$\mathcal{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} \end{bmatrix} = \begin{bmatrix} 10 & 2 \\ 2 & 4 \end{bmatrix}$$

- Determinant of Hessian:
  $$\det(\mathcal{H}) = (10)(4) - (2)(2) = 40 - 4 = 36$$
- Analytical 2D matrix inverse ($\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$):
  $$\mathcal{H}^{-1} = \frac{1}{36} \begin{bmatrix} 4 & -2 \\ -2 & 10 \end{bmatrix}$$

- Initial starting point:
  $$x_0 = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix}$$

---

### Step-by-Step Manual Calculations

#### 1. Evaluate Gradient at $x_0 = [2.0, -1.0]^T$:
$$\nabla f(x_0) = \begin{bmatrix} 10(2.0) + 2(-1.0) \\ 2(2.0) + 4(-1.0) \end{bmatrix} = \begin{bmatrix} 20 - 2 \\ 4 - 4 \end{bmatrix} = \begin{bmatrix} 18.0 \\ 0.0 \end{bmatrix}$$
*(Notice: The gradient along $x_2$ is ZERO! A first-order method will not move along $x_2$ at all!)*

#### 2. Standard Gradient Descent Step (with learning rate $\alpha = 0.05$):
$$x_1^{\text{GD}} = x_0 - \alpha \nabla f(x_0) = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} - 0.05 \begin{bmatrix} 18.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 2.0 - 0.90 \\ -1.0 - 0.00 \end{bmatrix} = \begin{bmatrix} 1.1000 \\ -1.0000 \end{bmatrix}$$
Loss at $x_1^{\text{GD}}$:
$$f(1.1, -1.0) = 5(1.1)^2 + 2(1.1)(-1.0) + 2(-1.0)^2 = 5(1.21) - 2.2 + 2.0 = 6.05 - 2.2 + 2.0 = \mathbf{5.8500}$$
*(Still far from global optimum at $(0, 0)$ where $f(0,0)=0$.)*

#### 3. Newton's Method Step:
Compute Newton direction $\Delta x = -\mathcal{H}^{-1} \nabla f(x_0)$:
$$\Delta x = -\frac{1}{36} \begin{bmatrix} 4 & -2 \\ -2 & 10 \end{bmatrix} \begin{bmatrix} 18.0 \\ 0.0 \end{bmatrix} = -\frac{1}{36} \begin{bmatrix} (4)(18) + (-2)(0) \\ (-2)(18) + (10)(0) \end{bmatrix} = -\frac{1}{36} \begin{bmatrix} 72.0 \\ -36.0 \end{bmatrix} = \begin{bmatrix} -\frac{72}{36} \\ +\frac{36}{36} \end{bmatrix} = \begin{bmatrix} -2.0000 \\ +1.0000 \end{bmatrix}$$

Update parameter position:
$$x_1^{\text{Newton}} = x_0 + \Delta x = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} + \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.0000} \\ \mathbf{0.0000} \end{bmatrix}$$

Loss at $x_1^{\text{Newton}}$:
$$f(0, 0) = 5(0)^2 + 2(0)(0) + 2(0)^2 = \mathbf{0.0000}$$

**Exact global optimum reached in EXACTLY ONE STEP!**
Even though the gradient along $x_2$ was exactly 0.0, the off-diagonal cross-coupling term $\mathcal{H}_{12} = 2$ informed Newton's method that moving along $x_1$ shifts the optimal $x_2$, calculating the exact corrective step $+1.0$ along $x_2$!

---

### Comparative Visual Grid: Newton vs. Gradient Descent

```
+----------------------------------------------------------------------------------------------------+
|                                    STARTING POINT: x_0 = [2.0, -1.0]                               |
|                                    LOSS: f(x_0) = 5(4) + 2(-2) + 2(1) = 18.0000                    |
|                                    GRADIENT: [18.0000, 0.0000]                                     |
+-------------------+----------------------+------------------------+--------------------------------+
| Optimization      | Step Formula         | Parameter Result x_1   | Resulting Loss f(x_1)          |
| Algorithm         |                      |                        |                                |
+-------------------+----------------------+------------------------+--------------------------------+
| Standard GD       | x_0 - alpha * grad   | [1.1000, -1.0000]      | 5.8500 (32% reduction)         |
| (alpha = 0.05)    |                      |                        |                                |
+-------------------+----------------------+------------------------+--------------------------------+
| Newton's Method   | x_0 - H^{-1} * grad  | [0.0000, 0.0000]       | 0.0000 (100% convergence in    |
| (Second-Order)    |                      | (EXACT OPTIMUM)        | EXACTLY 1 STEP!)               |
+-------------------+----------------------+------------------------+--------------------------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning Analog |
| :--- | :--- | :--- | :--- |
| $\nabla f(\theta)$ | $\mathbb{R}^d$ vector | First derivative (gradient) vector | Backpropagated loss gradients (`param.grad`) |
| $\mathcal{H} = \nabla^2 f(\theta)$ | $\mathbb{R}^{d \times d}$ matrix | Hessian curvature matrix | Local curvature tensor of loss surface |
| $\mathcal{H}^{-1} \nabla f$ | $\mathbb{R}^d$ vector | Newton search direction | Curvature-corrected parameter step |
| $H_t$ | $\mathbb{R}^{d \times d}$ matrix | Quasi-Newton running inverse Hessian | BFGS / L-BFGS metric operator |
| $\mathcal{H} v$ | $\mathbb{R}^d$ vector | Directional derivative of gradient along $v$ | Pearlmutter vector product via autograd |
| $F(\theta)$ | $\mathbb{R}^{d \times d}$ matrix | Fisher Information Matrix | Expected outer product of log-likelihood gradients |
| $A \otimes S$ | Kronecker product | Matrix factorization of Fisher block | K-FAC activation and sensitivity covariance |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Newton's Catastrophic Failure on Non-Convex Functions
**Statement**: Consider the non-convex double-well potential:
$$f(x) = x^4 - 2x^2$$
1. Find all stationary points ($\nabla f = 0$) and classify them using the second derivative.
2. Suppose Newton's method is initialized at $x_0 = 0.5$. Compute the next iterate $x_1$.
3. What happens if initialized at $x_0 = 0.1$? Does Newton move toward the minima ($\pm 1$) or the local maximum ($0$)?

**Solution**:
1. **Derivatives**:
   $$f'(x) = 4x^3 - 4x = 4x(x^2 - 1) = 4x(x - 1)(x + 1)$$
   $$f''(x) = 12x^2 - 4$$
   Stationary points:
   - $x = 0$: $f''(0) = -4 < 0 \implies$ **Local Maximum** ($f(0) = 0$)
   - $x = +1$: $f''(1) = 12(1) - 4 = +8 > 0 \implies$ **Global Minimum** ($f(1) = -1$)
   - $x = -1$: $f''(-1) = 12(1) - 4 = +8 > 0 \implies$ **Global Minimum** ($f(-1) = -1$)

2. **Iteration from $x_0 = 0.5$**:
   $$f'(0.5) = 4(0.125) - 4(0.5) = 0.5 - 2.0 = -1.5$$
   $$f''(0.5) = 12(0.25) - 4 = 3 - 4 = -1.0$$
   Newton step:
   $$x_1 = x_0 - \frac{f'(x_0)}{f''(x_0)} = 0.5 - \frac{-1.5}{-1.0} = 0.5 - 1.5 = \mathbf{-1.0000}$$
   *Notice*: In one step, $x_1 = -1.0$, landing on the opposite global minimum!

3. **Iteration from $x_0 = 0.1$**:
   $$f'(0.1) = 4(0.001) - 4(0.1) = 0.004 - 0.4 = -0.396$$
   $$f''(0.1) = 12(0.01) - 4 = 0.12 - 4 = -3.88$$
   Newton step:
   $$x_1 = 0.1 - \frac{-0.396}{-3.88} = 0.1 - 0.010206 = \mathbf{-0.00206}$$
   *Disaster*: Because curvature $f'' < 0$, Newton's method leaped straight toward $x=0$—the **local maximum**! Standard GD with learning rate $\alpha = 0.1$ moves:
   $$x_1^{\text{GD}} = 0.1 - 0.1(-0.396) = 0.1396$$
   GD moves *away* from the local maximum toward the minimum at $+1$, whereas Newton moves *toward* the local maximum! This demonstrates why vanilla Newton is deadly in non-convex neural networks.

---

### Problem 2: BFGS Rank-2 Update Arithmetic
**Statement**: Suppose at step $t$, the current inverse Hessian estimate is the identity: $H_0 = I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$.
A step is taken with displacement $s = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and gradient change $y = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$.
1. Verify the curvature condition $y^T s > 0$.
2. Compute $\rho = \frac{1}{y^T s}$.
3. Compute the updated inverse Hessian matrix $H_1$ using the BFGS formula.

**Solution**:
1. **Curvature Condition**:
   $$y^T s = (2)(1) + (1)(2) = 2 + 2 = 4 > 0 \quad \text{(Positive curvature satisfied!)}$$
2. **Scalar $\rho$**:
   $$\rho = \frac{1}{4} = 0.25$$
3. **BFGS Update Matrix**:
   $$H_1 = (I - \rho s y^T) H_0 (I - \rho y s^T) + \rho s s^T$$
   Since $H_0 = I$:
   $$s y^T = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 2 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 4 & 2 \end{bmatrix}$$
   $$y s^T = (s y^T)^T = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}$$
   $$s s^T = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

   Now:
   $$I - \rho s y^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} - 0.25 \begin{bmatrix} 2 & 1 \\ 4 & 2 \end{bmatrix} = \begin{bmatrix} 0.5 & -0.25 \\ -1.0 & 0.5 \end{bmatrix}$$
   $$(I - \rho y s^T) = (I - \rho s y^T)^T = \begin{bmatrix} 0.5 & -1.0 \\ -0.25 & 0.5 \end{bmatrix}$$

   Multiply:
   $$M = (I - \rho s y^T)(I - \rho y s^T) = \begin{bmatrix} 0.5 & -0.25 \\ -1.0 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 & -1.0 \\ -0.25 & 0.5 \end{bmatrix}$$
   $$M_{11} = (0.5)(0.5) + (-0.25)(-0.25) = 0.25 + 0.0625 = 0.3125$$
   $$M_{12} = (0.5)(-1.0) + (-0.25)(0.5) = -0.5 - 0.125 = -0.625$$
   $$M_{21} = (-1.0)(0.5) + (0.5)(-0.25) = -0.5 - 0.125 = -0.625$$
   $$M_{22} = (-1.0)(-1.0) + (0.5)(0.5) = 1.0 + 0.25 = 1.25$$

   Add $\rho s s^T = 0.25 \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} = \begin{bmatrix} 0.25 & 0.5 \\ 0.5 & 1.0 \end{bmatrix}$:
   $$H_1 = \begin{bmatrix} 0.3125 & -0.625 \\ -0.625 & 1.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.5 \\ 0.5 & 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5625} & \mathbf{-0.1250} \\ \mathbf{-0.1250} & \mathbf{2.2500} \end{bmatrix}$$

   **Verification of Secant Equation**:
   $$H_1 y = \begin{bmatrix} 0.5625 & -0.1250 \\ -0.1250 & 2.2500 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 1.125 - 0.125 \\ -0.250 + 2.250 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = s$$
   The Secant Equation $H_1 y = s$ is verified to 100% exact precision!

---

### Problem 3: Natural Gradient Invariance to Parameterization
**Statement**: Consider a 1D Bernoulli model parameterized by probability $p \in (0, 1)$ with likelihood $P(Y=1) = p$.
Now reparameterize using logit $\theta$ where $p = \sigma(\theta) = \frac{1}{1 + e^{-\theta}}$.
Show that the natural gradient update step in probability space is identical regardless of whether optimization is conducted directly on $p$ or on logit $\theta$.

**Derivation**:
1. Log-likelihood for single observation $y \in \{0, 1\}$:
   $$\log p(y) = y \log p + (1-y) \log(1-p)$$
2. Score function with respect to $p$:
   $$\frac{\partial \log p(y)}{\partial p} = \frac{y}{p} - \frac{1-y}{1-p} = \frac{y - p}{p(1-p)}$$
   Fisher Information with respect to $p$:
   $$F_p = \mathbb{E}\left[ \left(\frac{y - p}{p(1-p)}\right)^2 \right] = \frac{\text{Var}(y)}{p^2(1-p)^2} = \frac{p(1-p)}{p^2(1-p)^2} = \frac{1}{p(1-p)}$$
   Natural gradient step on $p$:
   $$\widetilde{\nabla}_p \mathcal{L} = F_p^{-1} \nabla_p \mathcal{L} = p(1-p) \nabla_p \mathcal{L}$$

3. Score function with respect to logit $\theta$:
   By chain rule: $\nabla_\theta \mathcal{L} = \frac{\partial p}{\partial \theta} \nabla_p \mathcal{L} = p(1-p) \nabla_p \mathcal{L}$.
   Fisher Information with respect to $\theta$:
   $$F_\theta = \left(\frac{\partial p}{\partial \theta}\right)^2 F_p = [p(1-p)]^2 \frac{1}{p(1-p)} = p(1-p)$$
   Natural gradient step on $\theta$:
   $$\widetilde{\nabla}_\theta \mathcal{L} = F_\theta^{-1} \nabla_\theta \mathcal{L} = \frac{1}{p(1-p)} \cdot \left[ p(1-p) \nabla_p \mathcal{L} \right] = \nabla_p \mathcal{L}$$

4. Induced change in probability $p$:
   $$\Delta p \approx \frac{\partial p}{\partial \theta} \widetilde{\nabla}_\theta \mathcal{L} = p(1-p) \nabla_p \mathcal{L} = \widetilde{\nabla}_p \mathcal{L}$$
   The step taken in model prediction space is **strictly identical**. Natural gradient eliminates arbitrary parameterization distortion!

---

## Part 7: Deep Learning Connection & Application

### 1. Second-Order Ideas in Modern Deep Learning
While full Newton is never run on 100-billion parameter models, second-order mathematics powers essential deep learning systems:
1. **Reinforcement Learning (TRPO / PPO / ACKTR)**:
   - **Trust Region Policy Optimization (TRPO)** solves $\max_\theta \mathbb{E}[\frac{\pi_\theta}{\pi_{\text{old}}} A]$ subject to a KL constraint $D_{\text{KL}}(\pi_{\text{old}} \parallel \pi_\theta) \le \delta$.
   - TRPO uses **Conjugate Gradient on Hessian-vector products ($F v$)** to compute the natural policy gradient without ever storing the Fisher matrix!
2. **K-FAC in LLM Optimization & Science**:
   - DeepMind's **FermiNet** (solving the Schrödinger equation for quantum chemistry via deep neural networks) relies heavily on **K-FAC** because quantum ground-state optimization has extreme condition numbers ($\kappa > 10^8$) where Adam completely stalls.
3. **Sharpness-Aware Minimization (SAM)**:
   - SAM seeks parameters lying in flat minima by computing a worst-case perturbation: $\max_{\|\epsilon\| \le \rho} \mathcal{L}(\theta + \epsilon)$.
   - The Taylor expansion reveals SAM implicitly minimizes $\mathcal{L}(\theta) + \rho \|\nabla \mathcal{L}\| + \frac{\rho^2}{2} \lambda_{\max}(\mathcal{H})$, directly regularizing the maximum eigenvalue of the Hessian!

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `05_optimization/code/07_second_order_and_natural_gradients.py`. It implements:
1. Exact visual grid verification of Newton's method vs. Standard GD on the 2D non-diagonal quadratic.
2. Secant equation and positive definiteness verification of the BFGS rank-2 update.
3. Hessian-Free optimization using Pearlmutter's $\mathcal{R}$-operator (autograd Hessian-vector product) compared against explicit full Hessian matrix multiplication.
4. Natural Gradient Descent on a softmax classification head verifying coordinate invariance.

Save the code and run it directly in Python 3.
