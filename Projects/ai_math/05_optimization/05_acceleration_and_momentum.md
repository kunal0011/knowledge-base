# 5.5 Acceleration & Momentum (Polyak Momentum & Nesterov NAG)

---

## Part 1: Intuition & 101 Motivation

In Chapter 5.4, we explored first-order gradient descent. While gradient descent possesses strong theoretical guarantees for smooth convex objectives, its practical behavior on real-world loss landscapes is plagued by severe inefficiency.

The fundamental culprit is **anisotropic curvature** (ill-conditioning). In typical deep neural networks, loss surfaces are shaped like narrow ravines or steep valleys:
- Along transverse directions (across the valley walls), curvature is extraordinarily high ($\lambda_{\max} \gg 1$).
- Along longitudinal directions (down the base of the valley toward the minimum), curvature is extraordinarily flat ($\lambda_{\min} \ll 1$).

Standard gradient descent updates parameters strictly along the instantaneous negative gradient:
$$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$

Because $\nabla f(\theta_t)$ is dominated by the steep valley walls, gradient descent takes giant steps perpendicular to the valley, bouncing violently between walls while crawling at a snail's pace along the valley floor. If the step size $\alpha$ is increased to accelerate along the valley floor, the algorithm violently diverges across the steep walls ($\alpha > 2/\lambda_{\max}$).

```
Standard GD in an Ill-Conditioned Ravine (Violent Zig-Zagging):
       Wall 1
         |  \      /
         |   \    /
         |    \  /
         |     \/  <-- Bounces back and forth across high curvature
         |     /\
         |    /  \
         |   /    \
       Wall 2
       ----------------------------------------------------->
                        Flat valley floor (slow crawl)
```

To resolve this pathology, Boris Polyak (1964) introduced the **Heavy-Ball Method** (classical momentum), inspired by physical mechanics: endow the optimization point with mass and inertia. Instead of setting velocity purely from the instantaneous slope, the update accumulates past velocity, dampening high-frequency perpendicular oscillations (which have alternating signs and cancel out) while accelerating low-frequency forward progress (which point persistently in the same direction).

In 1983, Yurii Nesterov made a revolutionary leap with **Nesterov Accelerated Gradient (NAG)**: rather than calculating the gradient at the current position and adding momentum, NAG "looks ahead" along the momentum vector first, calculates the gradient at this projected position, and uses that future gradient to correct course. This lookahead acts as an anticipatory brake, dramatically reducing overshoot and improving the theoretical convergence rate on general convex smooth functions from $\mathcal{O}(1/t)$ to the provably optimal $\mathcal{O}(1/t^2)$.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Classical Momentum (Polyak Heavy-Ball)

Let $f: \mathbb{R}^d \to \mathbb{R}$ be continuously differentiable. The Polyak heavy-ball algorithm maintains a parameter vector $\theta_t \in \mathbb{R}^d$ and a velocity vector $v_t \in \mathbb{R}^d$.

At iteration $t \ge 0$, with step size $\alpha > 0$ and momentum coefficient $\beta \in [0, 1)$:
$$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

*(Note: Some literature defines $v_{t+1} = \beta v_t + \nabla f(\theta_t)$ and $\theta_{t+1} = \theta_t - \alpha v_{t+1}$. Both parameterizations are mathematically isomorphic under re-scaling of $v$.)*

Expanding recursively reveals momentum as an **Exponentially Weighted Moving Average (EWMA)** of historical gradients:
$$v_{t+1} = \sum_{\tau=0}^t \beta^{t-\tau} \alpha \nabla f(\theta_\tau)$$

The effective horizon of accumulated history is approximately:
$$\tau_{\text{eff}} = \frac{1}{1 - \beta}$$
For $\beta = 0.9$, the optimizer integrates gradients over approximately $\frac{1}{1-0.9} = 10$ steps. In a stationary direction where $\nabla f(\theta) \approx g$, terminal velocity reaches:
$$v_{\infty} = \frac{\alpha g}{1 - \beta} = 10 \alpha g$$
providing a $10\times$ speedup along flat plateaus.

---

### 2. Nesterov Accelerated Gradient (NAG)

Nesterov's acceleration replaces the gradient at $\theta_t$ with the gradient at the **lookahead position** $\theta_t - \beta v_t$:

$$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t - \beta v_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

#### Canonical Formulation (Extrapolation Sequence)
In convex optimization literature, NAG is frequently written with an auxiliary sequence $(y_k, x_k)$:
$$x_{k+1} = y_k - \alpha \nabla f(y_k)$$
$$y_{k+1} = x_{k+1} + \frac{k-1}{k+2} (x_{k+1} - x_k)$$
where the momentum coefficient $\beta_k = \frac{k-1}{k+2} \to 1$ increases with iteration index $k$.

---

### 3. Continuous-Time Dynamical Systems (ODEs)

The behavior of momentum algorithms is illuminated by taking the continuous-time limit ($\Delta t \to 0$).

#### Polyak Momentum as a Damped Harmonic Oscillator
Discretizing time with step size $h = \sqrt{\alpha}$, Polyak's update satisfies:
$$\frac{\theta_{t+1} - 2\theta_t + \theta_{t-1}}{h^2} + \frac{1 - \beta}{h} \frac{\theta_t - \theta_{t-1}}{h} + \nabla f(\theta_t) = 0$$

Taking $h \to 0$ with damping coefficient $\gamma = \frac{1 - \beta}{h}$ yields Newton's second law for a particle of unit mass in a potential field $f(\theta)$ with viscous friction:
$$\ddot{\theta}(t) + \gamma \dot{\theta}(t) + \nabla f(\theta(t)) = 0$$

#### Nesterov Acceleration as Asymptotically Vanishing Friction
In their seminal 2014 paper, Su, Boyd, and Candès proved that the continuous-time limit of Nesterov's accelerated gradient method is an ODE with time-dependent friction:
$$\ddot{\theta}(t) + \frac{3}{t} \dot{\theta}(t) + \nabla f(\theta(t)) = 0$$

- When $t \to 0$, friction $\frac{3}{t} \to \infty$, preventing early instability.
- When $t \to \infty$, friction $\frac{3}{t} \to 0$, allowing maximum kinetic acceleration along flat directions.
- This specific vanishing friction coefficient $\frac{3}{t}$ is the exact mathematical key to unlocking the optimal $\mathcal{O}(1/t^2)$ convergence rate.

---

### 4. Spectral Analysis on Quadratic Objectives

Consider a strictly convex quadratic objective:
$$f(\theta) = \frac{1}{2} \theta^T A \theta$$
where $A \in \mathbb{R}^{d \times d}$ is symmetric positive definite with eigenvalues $\mu = \lambda_{\min} \le \dots \le \lambda_{\max} = L$. The condition number is:
$$\kappa = \frac{L}{\mu} \ge 1$$

#### Standard Gradient Descent Contraction
Under optimal step size $\alpha^* = \frac{2}{L + \mu}$, the error $e_t = \theta_t - \theta^*$ contracts as:
$$\|e_t\| \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \|e_0\|$$
For $\kappa = 100$, the contraction factor is $\frac{99}{101} \approx 0.9802$. To reduce error by $e^{-1}$, standard GD requires $\sim \frac{\kappa}{2} = 50$ iterations.

#### Polyak Heavy-Ball Contraction
For quadratic $f(\theta)$, the heavy-ball recurrence can be decoupled across each eigenvector of $A$ with eigenvalue $\lambda \in [\mu, L]$:
$$\theta_{t+1} - \theta_t = \beta(\theta_t - \theta_{t-1}) - \alpha \lambda \theta_t$$
$$\theta_{t+1} - (1 + \beta - \alpha \lambda)\theta_t + \beta \theta_{t-1} = 0$$

In companion matrix form:
$$\begin{bmatrix} e_{t+1} \\ e_t \end{bmatrix} = \begin{bmatrix} (1 + \beta - \alpha \lambda) I & -\beta I \\ I & 0 \end{bmatrix} \begin{bmatrix} e_t \\ e_{t-1} \end{bmatrix} = T_\lambda \begin{bmatrix} e_t \\ e_{t-1} \end{bmatrix}$$

The characteristic equation of $T_\lambda$ is:
$$\det(z I - T_\lambda) = z^2 - (1 + \beta - \alpha \lambda)z + \beta = 0$$

To minimize the spectral radius $\rho(T_\lambda) = \max_i |\sigma_i(T_\lambda)|$ uniformly across all $\lambda \in [\mu, L]$, the discriminant must be non-positive for all $\lambda$, yielding complex conjugate roots with magnitude $\sqrt{\beta}$. The optimal hyperparameters are uniquely determined:
$$\beta^* = \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)^2 = \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^2$$
$$\alpha^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}$$

The optimal spectral radius becomes:
$$\rho(T^*) = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$$

For $\kappa = 100$:
$$\rho(T^*) = \frac{\sqrt{100} - 1}{\sqrt{100} + 1} = \frac{9}{11} \approx 0.8182$$
Compare $0.8182$ to standard GD's $0.9802$! The number of iterations to reduce error scales as $\mathcal{O}(\sqrt{\kappa})$ instead of $\mathcal{O}(\kappa)$. When $\kappa = 10{,}000$, Polyak requires $\sim 100$ steps where GD requires $\sim 10{,}000$ steps—a **$100\times$ acceleration**.

---

### Deep Derivation 5.5.1: First-Principles Proof of Optimal Polyak Hyperparameters $\alpha^*$ and $\beta^*$

We rigorously derive the optimal step size $\alpha^*$ and momentum coefficient $\beta^*$ that minimize the spectral radius of the heavy-ball companion matrix over all eigenvalues $\lambda \in [\mu, L]$.

#### 1. The Characteristic Polynomial
For an eigenvalue $\lambda$, the error dynamics matrix is:
$$T_\lambda = \begin{bmatrix} (1 + \beta - \alpha \lambda) & -\beta \\ 1 & 0 \end{bmatrix}$$
The characteristic polynomial is:
$$P(z) = \det(z I - T_\lambda) = z^2 - (1 + \beta - \alpha \lambda) z + \beta = 0$$
The discriminant of this quadratic is:
$$\Delta(\lambda) = (1 + \beta - \alpha \lambda)^2 - 4\beta$$

#### 2. Complex Conjugate Roots and Constant Modulus
When $\Delta(\lambda) \le 0$, the roots are complex conjugate pairs $z_{1, 2} = \frac{(1 + \beta - \alpha \lambda) \pm i \sqrt{4\beta - (1 + \beta - \alpha \lambda)^2}}{2}$.
The squared modulus of the roots is:
$$|z|^2 = \left( \frac{1 + \beta - \alpha \lambda}{2} \right)^2 + \frac{4\beta - (1 + \beta - \alpha \lambda)^2}{4} = \frac{4\beta}{4} = \beta$$
$$|z| = \sqrt{\beta}$$
Remarkably, whenever the roots are complex, **their magnitude is strictly constant and independent of the eigenvalue $\lambda$**:
$$\rho(T_\lambda) = \sqrt{\beta} \quad \forall \lambda \text{ such that } \Delta(\lambda) \le 0$$

#### 3. Enclosing the Entire Spectrum in the Complex Disc
To ensure convergence across the entire spectrum $[\mu, L]$ while keeping $\sqrt{\beta}$ as small as possible, the discriminant $\Delta(\lambda)$ must remain non-positive across the full interval $\lambda \in [\mu, L]$, with critical damping (discriminant zero) at both endpoints:
$$(1 + \beta - \alpha \mu)^2 = 4\beta \implies 1 + \beta - \alpha \mu = +2\sqrt{\beta}$$
$$(1 + \beta - \alpha L)^2 = 4\beta \implies 1 + \beta - \alpha L = -2\sqrt{\beta}$$

#### 4. Solving the Simultaneous Equations
Subtract the second equation from the first:
$$\alpha (L - \mu) = 4\sqrt{\beta} \implies \mathbf{\alpha = \frac{4\sqrt{\beta}}{L - \mu}}$$

Add the two equations:
$$2(1 + \beta) - \alpha(L + \mu) = 0 \implies 2(1 + \beta) = \alpha(L + \mu)$$
Substitute $\alpha = \frac{4\sqrt{\beta}}{L - \mu}$:
$$2(1 + \beta) = \frac{4\sqrt{\beta}(L + \mu)}{L - \mu} \implies 1 + \beta = 2\sqrt{\beta} \left( \frac{L + \mu}{L - \mu} \right)$$

Let $u = \sqrt{\beta}$. The equation becomes a quadratic in $u$:
$$u^2 - 2 \left( \frac{L + \mu}{L - \mu} \right) u + 1 = 0$$

Apply the quadratic formula:
$$u = \frac{2 \frac{L + \mu}{L - \mu} \pm \sqrt{4\left(\frac{L + \mu}{L - \mu}\right)^2 - 4}}{2} = \frac{L + \mu \pm \sqrt{(L + \mu)^2 - (L - \mu)^2}}{L - \mu} = \frac{L + \mu \pm \sqrt{4 L \mu}}{L - \mu}$$
Factoring numerator and denominator:
$$u = \frac{(\sqrt{L} \pm \sqrt{\mu})^2}{(\sqrt{L} - \sqrt{\mu})(\sqrt{L} + \sqrt{\mu})}$$
For stability, we require $u = \sqrt{\beta} < 1$, selecting the minus sign:
$$\mathbf{\sqrt{\beta^*} = \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}}$$
Squaring yields the optimal momentum coefficient:
$$\mathbf{\beta^* = \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^2}$$

#### 5. Recovering the Optimal Step Size $\alpha^*$
Substitute $\sqrt{\beta^*} = \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}$ into the expression for $\alpha$:
$$\alpha^* = \frac{4 \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)}{L - \mu} = \frac{4 (\sqrt{L} - \sqrt{\mu})}{(\sqrt{L} - \sqrt{\mu})(\sqrt{L} + \sqrt{\mu})^2} = \mathbf{\frac{4}{(\sqrt{L} + \sqrt{\mu})^2}} \quad \blacksquare$$

---

### Deep Derivation 5.5.2: Algebraic Equivalence of Sutskever's PyTorch NAG Formulation

In production frameworks, evaluating gradients at lookahead positions $\theta_t - \beta v_t$ is costly. Sutskever et al. (2013) proved that a simple change of variables eliminates the auxiliary forward-backward pass.

#### 1. The Standard NAG Equations
$$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t - \beta v_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

#### 2. The Coordinate Transformation
Define the lookahead variable as the primary state variable:
$$\phi_t = \theta_t - \beta v_t \implies \theta_t = \phi_t + \beta v_t$$
Substituting $\phi_t$ into the velocity update:
$$v_{t+1} = \beta v_t + \alpha \nabla f(\phi_t)$$

#### 3. Propagating the Transformed State
Now compute the transformed state at step $t+1$:
$$\phi_{t+1} = \theta_{t+1} - \beta v_{t+1}$$
Substitute $\theta_{t+1} = \theta_t - v_{t+1}$:
$$\phi_{t+1} = (\theta_t - v_{t+1}) - \beta v_{t+1} = (\phi_t + \beta v_t - v_{t+1}) - \beta v_{t+1}$$
Notice that from the velocity update: $\beta v_t - v_{t+1} = -\alpha \nabla f(\phi_t)$.
Substitute this relation:
$$\phi_{t+1} = \phi_t - \alpha \nabla f(\phi_t) - \beta v_{t+1}$$

#### 4. The Production Update Rule
Renaming the dummy symbol $\phi_t \to \theta_t$:
$$\mathbf{v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t)}$$
$$\mathbf{\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t) - \beta v_{t+1}}$$
*Result:* The gradient is computed **strictly at the current parameter $\theta_t$**, exactly like standard SGD, while preserving the exact accelerated trajectory of Nesterov's lookahead! $\blacksquare$

---

### 5. Summary of Convergence Rates

| Setting | Standard GD | Polyak Heavy-Ball | Nesterov (NAG) | Theoretical Lower Bound |
| :--- | :---: | :---: | :---: | :---: |
| **Convex, $L$-Smooth** | $\mathcal{O}(1/t)$ | $\mathcal{O}(1/t)$ | $\mathcal{O}(1/t^2)$ | $\Omega(1/t^2)$ (Nemirovski) |
| **Strongly Convex ($\mu > 0$)** | $\mathcal{O}\left( \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \right)$ | $\mathcal{O}\left( \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^t \right)$ | $\mathcal{O}\left( \left(1 - \frac{1}{\sqrt{\kappa}}\right)^t \right)$ | $\Omega\left( \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^t \right)$ |
| **Iteration Complexity ($\epsilon$-acc)** | $\mathcal{O}(\kappa \log(1/\epsilon))$ | $\mathcal{O}(\sqrt{\kappa} \log(1/\epsilon))$ | $\mathcal{O}(\sqrt{\kappa} \log(1/\epsilon))$ | $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ |

---

## Part 3: Geometric & Physical Interpretation

### 1. Phase-Space Trajectories and Ravine Navigation

Consider the 2D contour plot of an ill-conditioned quadratic valley:

```
        ^ x_2 (Transverse axis: high curvature lambda_1 = 20)
        |
        |      (.)  Start: (1, 1)
        |     /   \
        |    /  GD \  (High-frequency zig-zags)
        |   /       \
        |  /         \
        +-----------------------------------> x_1 (Longitudinal axis: low curvature lambda_2 = 2)
        |  \         /
        |   \       /
        |    \ NAG /   (Smooth parabolic glide with anticipatory damping)
        |     \   /
        |      \ /
        |       * Optimum: (0, 0)
```

1. **Gradient Vectors Point Perpendicular to Iso-contours**:
   Because iso-contours are tight ellipses, the normal to the ellipse points almost purely along the steep $x_2$ axis, rather than toward the global minimum at $(0,0)$.
2. **Cancellation of Transverse Oscillations**:
   The gradient component along $x_2$ oscillates between positive and negative values: $\nabla_{x_2} f \in \{+g_2, -g_2, +g_2, \dots\}$. The EWMA momentum accumulator performs a running sum:
   $$v_{t+1, x_2} = \beta v_{t, x_2} + \alpha g_{t, x_2} \approx 0$$
   The perpendicular velocity cancels out.
3. **Constructive Reinforcement of Longitudinal Flow**:
   The gradient component along $x_1$ persistently points in the negative direction: $\nabla_{x_1} f < 0$. The EWMA accumulates these coherent signals:
   $$v_{t+1, x_1} \approx \frac{\alpha g_1}{1 - \beta}$$
   The iterate glides smoothly down the ravine floor.

### 2. Polyak vs. Nesterov Lookahead Geometry

Why does NAG prevent the heavy overshoot that plagues Polyak momentum?

```
Polyak Heavy Ball:
  theta_t -------- Momentum vector (beta * v_t) --------> [Overshoots minimum!]
     |
     +--- Gradient at theta_t: points downhill, but evaluated TOO LATE to brake.

Nesterov Accelerated Gradient (NAG):
  theta_t -------- Momentum vector (beta * v_t) --------> Lookahead point theta_look
                                                                |
                                                                v
                                                     Gradient at theta_look:
                                                     Points UPHILL against momentum!
                                                     Acts as an AUTOMATIC BRAKE.
```

- **Polyak evaluates gradient before moving**: If momentum is already carrying the ball rapidly toward the valley floor, evaluating the gradient at the current position adds even more forward thrust, causing the ball to blast past the minimum and ride far up the opposite wall.
- **NAG evaluates gradient after preliminary jump**: NAG first makes the jump along the momentum vector to $\theta_{\text{look}} = \theta_t - \beta v_t$. If that jump carried the point across the minimum and up the opposite slope, $\nabla f(\theta_{\text{look}})$ points backward! Adding this backward gradient immediately applies a corrective brake to $v_{t+1}$.

---

## Part 4: Real-World Analogy

### 1. The Heavy Bobsled vs. The Lightweight Cart
Imagine pushing a vehicle down an icy bobsled run with high banked curves:
- **Standard GD is a massless toy cart**: Every micro-bump on the banked walls flings the cart sideways across the track. It expends all its energy bouncing from left wall to right wall, making agonizingly slow forward progress.
- **Polyak Momentum is a 500 kg bobsled**: Due to high physical inertia, rapid lateral forces from the walls cannot immediately deflect its path; they cancel out. The bobsled accelerates relentlessly along the main channel. However, at a sharp hairpin turn, its momentum causes it to skid up the wall before gravity slows it down.
- **Nesterov Momentum is an Olympic bobsled team with a spotter**: The team looks ahead down the track. As they approach the hairpin turn, they see the upcoming banking *before* they arrive, applying the brakes in advance to carve a sharp, stable trajectory through the corner without skidding out.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us calculate by hand the exact cell-by-cell numerical updates for **Standard GD**, **Polyak Heavy-Ball**, and **Nesterov Accelerated Gradient (NAG)** on a concrete 2D ill-conditioned quadratic function.

### Objective Function & Hyperparameters
$$f(x_1, x_2) = 10 x_1^2 + x_2^2$$
- Gradient: $\nabla f(x_1, x_2) = \begin{bmatrix} 20 x_1 \\ 2 x_2 \end{bmatrix}$
- Curvatures: $\lambda_1 = 20$ (steep), $\lambda_2 = 2$ (gentle). Condition number $\kappa = 10$.
- Starting point: $x_0 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$, initial velocity $v_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$.
- Learning rate: $\alpha = 0.08$.
- Momentum coefficient: $\beta = 0.5$.

---

### Step-by-Step Manual Calculations

#### Method 1: Standard Gradient Descent (No Momentum, $\beta = 0$)
- **Iteration 1**:
  $$\nabla f(x_0) = \begin{bmatrix} 20(1.0) \\ 2(1.0) \end{bmatrix} = \begin{bmatrix} 20.0 \\ 2.0 \end{bmatrix}$$
  $$x_1 = x_0 - \alpha \nabla f(x_0) = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} - 0.08 \begin{bmatrix} 20.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 - 1.60 \\ 1.0 - 0.16 \end{bmatrix} = \begin{bmatrix} -0.6000 \\ 0.8400 \end{bmatrix}$$
  *(Notice the violent sign flip in $x_1$: from $+1.0$ to $-0.60$!)*

- **Iteration 2**:
  $$\nabla f(x_1) = \begin{bmatrix} 20(-0.60) \\ 2(0.84) \end{bmatrix} = \begin{bmatrix} -12.0 \\ 1.68 \end{bmatrix}$$
  $$x_2 = x_1 - \alpha \nabla f(x_1) = \begin{bmatrix} -0.60 \\ 0.84 \end{bmatrix} - 0.08 \begin{bmatrix} -12.0 \\ 1.68 \end{bmatrix} = \begin{bmatrix} -0.60 + 0.96 \\ 0.84 - 0.1344 \end{bmatrix} = \begin{bmatrix} +0.3600 \\ 0.7056 \end{bmatrix}$$
  *(Notice: $x_1$ flipped back to $+0.3600$. Severe zig-zag oscillation!)*

---

#### Method 2: Polyak Heavy-Ball Momentum
- **Iteration 1**:
  $$\nabla f(x_0) = \begin{bmatrix} 20.0 \\ 2.0 \end{bmatrix}$$
  $$v_1 = \beta v_0 + \alpha \nabla f(x_0) = 0.5 \begin{bmatrix} 0 \\ 0 \end{bmatrix} + 0.08 \begin{bmatrix} 20.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 1.6000 \\ 0.1600 \end{bmatrix}$$
  $$x_1 = x_0 - v_1 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} - \begin{bmatrix} 1.6000 \\ 0.1600 \end{bmatrix} = \begin{bmatrix} -0.6000 \\ 0.8400 \end{bmatrix}$$

- **Iteration 2**:
  $$\nabla f(x_1) = \begin{bmatrix} 20(-0.60) \\ 2(0.84) \end{bmatrix} = \begin{bmatrix} -12.0 \\ 1.68 \end{bmatrix}$$
  $$v_2 = \beta v_1 + \alpha \nabla f(x_1) = 0.5 \begin{bmatrix} 1.6000 \\ 0.1600 \end{bmatrix} + 0.08 \begin{bmatrix} -12.0 \\ 1.68 \end{bmatrix} = \begin{bmatrix} 0.8000 - 0.9600 \\ 0.0800 + 0.1344 \end{bmatrix} = \begin{bmatrix} -0.1600 \\ 0.2144 \end{bmatrix}$$
  $$x_2 = x_1 - v_2 = \begin{bmatrix} -0.6000 \\ 0.8400 \end{bmatrix} - \begin{bmatrix} -0.1600 \\ 0.2144 \end{bmatrix} = \begin{bmatrix} -0.4400 \\ 0.6256 \end{bmatrix}$$
  *(Notice: $v_2[0] = -0.1600$ dampened the return bounce! $x_1$ moved from $-0.60$ to $-0.44$ instead of wildly overshooting back to $+0.36$!)*

---

#### Method 3: Nesterov Accelerated Gradient (NAG)
- **Iteration 1**:
  Lookahead point: $x_{\text{look}, 0} = x_0 - \beta v_0 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} - \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$
  $$\nabla f(x_{\text{look}, 0}) = \begin{bmatrix} 20.0 \\ 2.0 \end{bmatrix}$$
  $$v_1 = \beta v_0 + \alpha \nabla f(x_{\text{look}, 0}) = \begin{bmatrix} 1.6000 \\ 0.1600 \end{bmatrix}$$
  $$x_1 = x_0 - v_1 = \begin{bmatrix} -0.6000 \\ 0.8400 \end{bmatrix}$$

- **Iteration 2**:
  Lookahead point:
  $$x_{\text{look}, 1} = x_1 - \beta v_1 = \begin{bmatrix} -0.6000 \\ 0.8400 \end{bmatrix} - 0.5 \begin{bmatrix} 1.6000 \\ 0.1600 \end{bmatrix} = \begin{bmatrix} -0.6000 - 0.8000 \\ 0.8400 - 0.0800 \end{bmatrix} = \begin{bmatrix} -1.4000 \\ 0.7600 \end{bmatrix}$$
  *(The lookahead anticipates where momentum will carry the particle!)*

  Gradient evaluated at lookahead point:
  $$\nabla f(x_{\text{look}, 1}) = \begin{bmatrix} 20(-1.4000) \\ 2(0.7600) \end{bmatrix} = \begin{bmatrix} -28.0000 \\ 1.5200 \end{bmatrix}$$

  Update velocity using lookahead gradient:
  $$v_2 = \beta v_1 + \alpha \nabla f(x_{\text{look}, 1}) = 0.5 \begin{bmatrix} 1.6000 \\ 0.1600 \end{bmatrix} + 0.08 \begin{bmatrix} -28.0000 \\ 1.5200 \end{bmatrix} = \begin{bmatrix} 0.8000 - 2.2400 \\ 0.0800 + 0.1216 \end{bmatrix} = \begin{bmatrix} -1.4400 \\ 0.2016 \end{bmatrix}$$

  Update position:
  $$x_2 = x_1 - v_2 = \begin{bmatrix} -0.6000 \\ 0.8400 \end{bmatrix} - \begin{bmatrix} -1.4400 \\ 0.2016 \end{bmatrix} = \begin{bmatrix} +0.8400 \\ 0.6384 \end{bmatrix}$$

---

### Comparative Visual Grid: Iteration 1 vs. Iteration 2

```
+----------------------------------------------------------------------------------------------------+
|                                    ITERATION 1 (Starting at x_0 = [1.0, 1.0])                       |
+-------------------+----------------------+------------------------+--------------------------------+
| Method            | Gradient Evaluated   | Velocity Vector v_1    | Position x_1                   |
+-------------------+----------------------+------------------------+--------------------------------+
| Standard GD       | [20.0000, 2.0000]    | N/A                    | [-0.6000, 0.8400]              |
| Polyak Heavy-Ball | [20.0000, 2.0000]    | [1.6000, 0.1600]       | [-0.6000, 0.8400]              |
| Nesterov NAG      | [20.0000, 2.0000]    | [1.6000, 0.1600]       | [-0.6000, 0.8400]              |
+-------------------+----------------------+------------------------+--------------------------------+

+----------------------------------------------------------------------------------------------------+
|                                    ITERATION 2 (Step from x_1)                                     |
+-------------------+----------------------+------------------------+--------------------------------+
| Method            | Gradient Evaluated   | Velocity Vector v_2    | Position x_2                   |
+-------------------+----------------------+------------------------+--------------------------------+
| Standard GD       | [-12.0000, 1.6800]   | N/A                    | [+0.3600, 0.7056]              |
| Polyak Heavy-Ball | [-12.0000, 1.6800]   | [-0.1600, 0.2144]      | [-0.4400, 0.6256]              |
| Nesterov NAG      | [-28.0000, 1.5200]   | [-1.4400, 0.2016]      | [+0.8400, 0.6384]              |
+-------------------+----------------------+------------------------+--------------------------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning & Physical Analog |
| :--- | :--- | :--- | :--- |
| $x_t$ or $\theta_t$ | $\mathbb{R}^d$ vector | Current point in parameter domain | Trainable weights and biases at step $t$ |
| $v_t$ | $\mathbb{R}^d$ vector | Accumulated momentum / velocity vector | Exponential moving average of past gradient updates |
| $\alpha$ | $\mathbb{R}_{>0}$ scalar | Step size along gradient direction | Learning rate hyperparameter (`lr`) |
| $\beta$ | $[0, 1)$ scalar | Friction / inertia damping coefficient | Momentum factor (`momentum=0.9` in PyTorch) |
| $x_{\text{look}, t}$ | $\mathbb{R}^d$ vector | Extrapolated lookahead point $\theta_t - \beta v_t$ | Position where future gradient is probed before moving |
| $\nabla f(\theta_t)$ | $\mathbb{R}^d$ vector | Gradient of potential function at current position | Batch or mini-batch backpropagated loss gradient |
| $\kappa = \lambda_{\max}/\mu$ | $\mathbb{R}_{\ge 1}$ scalar | Condition number of Hessian matrix | Ratio of highest to lowest curvature in loss landscape |
| $\rho(T)$ | $[0, 1)$ scalar | Spectral radius of error transition operator | Asymptotic convergence factor per iteration ($e_t \sim \rho^t$) |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Optimal Momentum & Spectral Radius on an Ill-Conditioned Matrix
**Statement**: Suppose a neural network's local quadratic approximation has extreme Hessian eigenvalues $\lambda_{\max} = L = 81$ and $\lambda_{\min} = \mu = 1$.
1. Compute the condition number $\kappa$.
2. Compute the optimal contraction factor $\rho_{\text{GD}}$ for Standard Gradient Descent.
3. Compute the optimal momentum coefficient $\beta^*$ and contraction factor $\rho_{\text{Polyak}}$ for Polyak's Heavy-Ball Method.
4. Calculate how many iterations each method requires to reduce the initial error by a factor of $10^{-4}$.

**Solution**:
1. **Condition Number**:
   $$\kappa = \frac{L}{\mu} = \frac{81}{1} = 81$$

2. **Standard Gradient Descent**:
   $$\alpha^* = \frac{2}{L + \mu} = \frac{2}{81 + 1} = \frac{2}{82} = \frac{1}{41} \approx 0.02439$$
   $$\rho_{\text{GD}} = \frac{\kappa - 1}{\kappa + 1} = \frac{81 - 1}{81 + 1} = \frac{80}{82} = \frac{40}{41} \approx 0.97561$$

3. **Polyak Heavy-Ball Method**:
   $$\sqrt{\kappa} = \sqrt{81} = 9$$
   $$\beta^* = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2 = \left(\frac{9 - 1}{9 + 1}\right)^2 = \left(\frac{8}{10}\right)^2 = 0.8^2 = 0.6400$$
   $$\alpha^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2} = \frac{4}{(9 + 1)^2} = \frac{4}{100} = 0.0400$$
   $$\rho_{\text{Polyak}} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} = \frac{9 - 1}{9 + 1} = \frac{8}{10} = 0.8000$$

4. **Iteration Count to reach $\|e_t\| / \|e_0\| \le 10^{-4}$**:
   $$\rho^t \le 10^{-4} \implies t \ge \frac{-4 \ln(10)}{\ln(\rho)}$$
   - For GD:
     $$\ln(\rho_{\text{GD}}) = \ln(0.97561) \approx -0.02470$$
     $$t_{\text{GD}} = \frac{-9.21034}{-0.02470} \approx \mathbf{373\text{ iterations}}$$
   - For Polyak:
     $$\ln(\rho_{\text{Polyak}}) = \ln(0.8000) \approx -0.22314$$
     $$t_{\text{Polyak}} = \frac{-9.21034}{-0.22314} \approx \mathbf{41\text{ iterations}}$$

   *Result*: Polyak momentum delivers more than a **$9\times$ acceleration** ($\sqrt{81} = 9$), achieving in 41 steps what takes GD 373 steps!

---

### Problem 2: Boundary / Edge Case — The Instability Threshold ($\beta \ge 1$)
**Statement**: In physical damping, friction must be strictly positive ($\gamma > 0$). What happens mathematically if the momentum coefficient is set to $\beta \ge 1.0$?

**Mathematical Analysis**:
Consider the 1D harmonic oscillator with zero gradient ($f(x) \equiv 0$, flat plateau):
$$v_{t+1} = \beta v_t \implies v_t = \beta^t v_0$$
$$x_{t+1} = x_t - v_{t+1} = x_0 - \sum_{\tau=1}^{t+1} v_\tau$$

- **Case 1: $\beta = 1.0$ (Zero Friction / Undamped System)**:
  $$v_t = v_0 \quad \forall t$$
  $$x_t = x_0 - t \cdot v_0$$
  The particle never stops; it drifts infinitely at constant velocity. In a quadratic well $f(x) = \frac{1}{2} k x^2$, the system forms an undamped conservative oscillator that perpetually orbits the minimum with energy $E = \frac{1}{2} v^2 + \frac{1}{2} k x^2 = \text{const}$, never converging to stationary point $x^*$.
- **Case 2: $\beta > 1.0$ (Negative Friction / Explosive Resonance)**:
  $$v_t = \beta^t v_0 \to \infty \quad \text{exponentially!}$$
  The state-space transition matrix $T$ has spectral radius $\rho(T) = \sqrt{\beta} > 1.0$. Any infinitesimal perturbation or machine precision noise triggers unbounded exponential explosion: $\|x_t\| \to \infty$.

*Conclusion*: For numerical stability, the momentum parameter must strictly satisfy $0 \le \beta < 1$. Typical empirical practice fixes $\beta \in [0.85, 0.99]$.

---

### Problem 3: PyTorch Implementation Detail (Sutskever Formulation of NAG)
**Statement**: PyTorch provides `torch.optim.SGD(params, lr=alpha, momentum=beta, nesterov=True)`. However, deep learning frameworks do not want to evaluate forward-backward passes at an auxiliary lookahead point $\theta_t - \beta v_t$ because that would require an extra parameter write and cache invalidation before `loss.backward()`. How does Sutskever et al. (2013) reformulate NAG so the gradient is computed at the standard parameter $\theta_t$?

**Derivation**:
Define the transformed variable $\phi_t = \theta_t - \beta v_t$. Substituting this into Nesterov's equations allows updating the parameter directly using the gradient at $\theta_t$:
$$v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - \beta v_{t+1} - \alpha \nabla f(\theta_t)$$

Notice that this requires **zero auxiliary forward passes**:
1. Compute standard gradient $\nabla f(\theta_t)$.
2. Update velocity: $v_{t+1} = \beta v_t + \alpha \nabla f(\theta_t)$.
3. Update parameters using the linear combination of current gradient and newly accumulated velocity:
   $$\theta_{t+1} = \theta_t - (\alpha \nabla f(\theta_t) + \beta v_{t+1})$$
This exact algebraic equivalence is what runs inside PyTorch's C++ backend!

---

### Problem 4: Step-by-Step 3-Iteration Trace of NAG vs. Polyak on an Asymmetric Ravine

Let us trace how Polyak momentum overshoots along an ill-conditioned ravine while Nesterov Accelerated Gradient (NAG) applies anticipatory braking.

#### 1. Problem Setup
$$f(x, y) = 15 x^2 + y^2$$
- Gradient: $\nabla f(x, y) = \begin{bmatrix} 30 x \\ 2 y \end{bmatrix}$
- Curvatures: $\lambda_1 = 30$ (steep valley wall), $\lambda_2 = 2$ (gentle valley floor). Condition number $\kappa = 15$.
- Initial state: $\mathbf{x}_0 = \begin{bmatrix} 1.0000 \\ 1.0000 \end{bmatrix}, \quad \mathbf{v}_0 = \begin{bmatrix} 0.0000 \\ 0.0000 \end{bmatrix}$.
- Hyperparameters: learning rate $\alpha = 0.05$, momentum $\beta = 0.60$.

#### 2. Iteration 1 (Identical for both methods since $\mathbf{v}_0 = \mathbf{0}$)
$$\nabla f(\mathbf{x}_0) = \begin{bmatrix} 30(1.0) \\ 2(1.0) \end{bmatrix} = \begin{bmatrix} 30.0000 \\ 2.0000 \end{bmatrix}$$
$$\mathbf{v}_1 = \beta \mathbf{v}_0 + \alpha \nabla f(\mathbf{x}_0) = 0.05 \begin{bmatrix} 30.0000 \\ 2.0000 \end{bmatrix} = \begin{bmatrix} 1.5000 \\ 0.1000 \end{bmatrix}$$
$$\mathbf{x}_1 = \mathbf{x}_0 - \mathbf{v}_1 = \begin{bmatrix} 1.0000 - 1.5000 \\ 1.0000 - 0.1000 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.5000 \\ 0.9000 \end{bmatrix}}$$

#### 3. Iteration 2: Polyak Heavy-Ball vs. NAG
- **Polyak Momentum:**
  $$\nabla f(\mathbf{x}_1) = \begin{bmatrix} 30(-0.50) \\ 2(0.90) \end{bmatrix} = \begin{bmatrix} -15.0000 \\ 1.8000 \end{bmatrix}$$
  $$\mathbf{v}_2^{\text{Polyak}} = 0.60 \begin{bmatrix} 1.5000 \\ 0.1000 \end{bmatrix} + 0.05 \begin{bmatrix} -15.0000 \\ 1.8000 \end{bmatrix} = \begin{bmatrix} 0.9000 - 0.7500 \\ 0.0600 + 0.0900 \end{bmatrix} = \mathbf{\begin{bmatrix} +0.1500 \\ +0.1500 \end{bmatrix}}$$
  $$\mathbf{x}_2^{\text{Polyak}} = \mathbf{x}_1 - \mathbf{v}_2 = \begin{bmatrix} -0.5000 - 0.1500 \\ 0.9000 - 0.1500 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.6500 \\ 0.7500 \end{bmatrix}}$$
  *(Notice: The positive residual inertia $0.9000$ overwhelmed the negative gradient $-0.7500$, causing $x$ to overshoot further from $-0.50 \to -0.65$!)*

- **NAG Lookahead:**
  Lookahead point:
  $$\mathbf{x}_{\text{look}, 1} = \mathbf{x}_1 - \beta \mathbf{v}_1 = \begin{bmatrix} -0.5000 \\ 0.9000 \end{bmatrix} - 0.60 \begin{bmatrix} 1.5000 \\ 0.1000 \end{bmatrix} = \begin{bmatrix} -0.5000 - 0.9000 \\ 0.9000 - 0.0600 \end{bmatrix} = \mathbf{\begin{bmatrix} -1.4000 \\ 0.8400 \end{bmatrix}}$$
  Gradient at lookahead point:
  $$\nabla f(\mathbf{x}_{\text{look}, 1}) = \begin{bmatrix} 30(-1.40) \\ 2(0.84) \end{bmatrix} = \begin{bmatrix} -42.0000 \\ 1.6800 \end{bmatrix}$$
  Velocity update with lookahead gradient:
  $$\mathbf{v}_2^{\text{NAG}} = 0.60 \begin{bmatrix} 1.5000 \\ 0.1000 \end{bmatrix} + 0.05 \begin{bmatrix} -42.0000 \\ 1.6800 \end{bmatrix} = \begin{bmatrix} 0.9000 - 2.1000 \\ 0.0600 + 0.0840 \end{bmatrix} = \mathbf{\begin{bmatrix} -1.2000 \\ +0.1440 \end{bmatrix}}$$
  Position update:
  $$\mathbf{x}_2^{\text{NAG}} = \mathbf{x}_1 - \mathbf{v}_2^{\text{NAG}} = \begin{bmatrix} -0.5000 - (-1.2000) \\ 0.9000 - 0.1440 \end{bmatrix} = \mathbf{\begin{bmatrix} +0.7000 \\ 0.7560 \end{bmatrix}}$$

#### 4. Iteration 3
- **Polyak Momentum:**
  $$\nabla f(\mathbf{x}_2^{\text{Polyak}}) = \begin{bmatrix} 30(-0.65) \\ 2(0.75) \end{bmatrix} = \begin{bmatrix} -19.5000 \\ 1.5000 \end{bmatrix}$$
  $$\mathbf{v}_3^{\text{Polyak}} = 0.60 \begin{bmatrix} 0.1500 \\ 0.1500 \end{bmatrix} + 0.05 \begin{bmatrix} -19.5000 \\ 1.5000 \end{bmatrix} = \begin{bmatrix} 0.0900 - 0.9750 \\ 0.0900 + 0.0750 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.8850 \\ +0.1650 \end{bmatrix}}$$
  $$\mathbf{x}_3^{\text{Polyak}} = \begin{bmatrix} -0.6500 - (-0.8850) \\ 0.7500 - 0.1650 \end{bmatrix} = \mathbf{\begin{bmatrix} +0.2350 \\ 0.5850 \end{bmatrix}}$$

- **NAG Lookahead:**
  $$\mathbf{x}_{\text{look}, 2} = \mathbf{x}_2^{\text{NAG}} - \beta \mathbf{v}_2^{\text{NAG}} = \begin{bmatrix} 0.7000 \\ 0.7560 \end{bmatrix} - 0.60 \begin{bmatrix} -1.2000 \\ 0.1440 \end{bmatrix} = \begin{bmatrix} 0.7000 + 0.7200 \\ 0.7560 - 0.0864 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.4200 \\ 0.6696 \end{bmatrix}}$$
  $$\nabla f(\mathbf{x}_{\text{look}, 2}) = \begin{bmatrix} 30(1.42) \\ 2(0.6696) \end{bmatrix} = \begin{bmatrix} 42.6000 \\ 1.3392 \end{bmatrix}$$
  $$\mathbf{v}_3^{\text{NAG}} = 0.60 \begin{bmatrix} -1.2000 \\ 0.1440 \end{bmatrix} + 0.05 \begin{bmatrix} 42.6000 \\ 1.3392 \end{bmatrix} = \begin{bmatrix} -0.7200 + 2.1300 \\ 0.0864 + 0.0670 \end{bmatrix} = \mathbf{\begin{bmatrix} +1.4100 \\ +0.1534 \end{bmatrix}}$$
  $$\mathbf{x}_3^{\text{NAG}} = \begin{bmatrix} 0.7000 - 1.4100 \\ 0.7560 - 0.1534 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.7100 \\ 0.6026 \end{bmatrix}}$$

#### 5. Comparison Verdict
```
Iteration │ Metric          │ Standard GD       │ Polyak Heavy-Ball │ Nesterov NAG
──────────┼─────────────────┼───────────────────┼───────────────────┼───────────────────
t = 0     │ Position [x, y] │ [ 1.0000, 1.0000] │ [ 1.0000, 1.0000] │ [ 1.0000, 1.0000]
t = 1     │ Position [x, y] │ [-0.5000, 0.9000] │ [-0.5000, 0.9000] │ [-0.5000, 0.9000]
t = 2     │ Position [x, y] │ [+0.2500, 0.8100] │ [-0.6500, 0.7500] │ [+0.7000, 0.7560]
t = 3     │ Position [x, y] │ [-0.1250, 0.7290] │ [+0.2350, 0.5850] │ [-0.7100, 0.6026]
──────────┼─────────────────┼───────────────────┼───────────────────┼───────────────────
Progress  │ Valley Floor y  │ 1.0000 -> 0.7290  │ 1.0000 -> 0.5850  │ 1.0000 -> 0.6026
```
Both momentum methods advance along the flat valley floor $y$ nearly **$1.5\times$ faster** than standard GD ($0.5850$ vs $0.7290$).

---

### Problem 5: Concrete Euler Discretization of the Damped Harmonic Oscillator

Let us demonstrate the exact mathematical mapping between a physical damped harmonic oscillator and Polyak momentum.

#### 1. Continuous Physical Model
Consider a unit-mass particle in a 1D quadratic potential $f(x) = \frac{1}{2} k x^2$ with viscous friction $\gamma$:
$$\ddot{x}(t) + \gamma \dot{x}(t) + k x(t) = 0$$
Let physical parameters be: spring stiffness $k = 25.0$, damping coefficient $\gamma = 4.0$.
The analytical roots of the characteristic polynomial $r^2 + 4r + 25 = 0$ are:
$$r = \frac{-4 \pm \sqrt{16 - 100}}{2} = -2 \pm i\sqrt{21} \approx -2.0 \pm 4.5826 i$$
This is an **underdamped oscillator** with exponential decay envelope $e^{-2t}$ and oscillation period $T = \frac{2\pi}{\sqrt{21}} \approx 1.37$ seconds.

#### 2. Discrete Finite-Difference Scheme
Approximate derivatives with discrete time step $\Delta t = 0.10$ seconds:
$$\dot{x}(t) \approx \frac{x_t - x_{t-1}}{\Delta t}, \qquad \ddot{x}(t) \approx \frac{x_{t+1} - 2x_t + x_{t-1}}{\Delta t^2}$$
Substitute into the ODE:
$$\frac{x_{t+1} - 2x_t + x_{t-1}}{\Delta t^2} + \gamma \frac{x_t - x_{t-1}}{\Delta t} + k x_t = 0$$
Multiply through by $\Delta t^2$:
$$x_{t+1} - 2x_t + x_{t-1} + \gamma \Delta t (x_t - x_{t-1}) + k \Delta t^2 x_t = 0$$
Group terms for $x_{t+1}$:
$$x_{t+1} = (2 - \gamma \Delta t - k \Delta t^2) x_t - (1 - \gamma \Delta t) x_{t-1}$$

#### 3. Mapping to Polyak Momentum Recurrence
Recall the Polyak momentum recurrence on $f(x) = \frac{1}{2} k x^2$:
$$x_{t+1} - x_t = \beta(x_t - x_{t-1}) - \alpha k x_t \implies x_{t+1} = (1 + \beta - \alpha k) x_t - \beta x_{t-1}$$
Comparing coefficients:
1. **Momentum coefficient:** $\mathbf{\beta = 1 - \gamma \Delta t = 1 - 4.0(0.10) = 0.6000}$
2. **Learning rate:** $\alpha k = k \Delta t^2 \implies \mathbf{\alpha = \Delta t^2 = 0.10^2 = 0.0100}$
3. Coefficient of $x_t$: $1 + \beta - \alpha k = 1 + 0.60 - 0.25 = \mathbf{1.3500}$.
$$(2 - \gamma \Delta t - k \Delta t^2) = 2 - 0.40 - 0.25 = \mathbf{1.3500} \quad (\text{Exact Match!})$$

#### 4. Step-by-Step Trajectory Simulation
Initialize at $x_0 = 1.0000$ from rest ($x_{-1} = 1.0000$):
- **Step 1 ($t = 1$):**
  $$x_1 = 1.3500(x_0) - 0.6000(x_{-1}) = 1.3500(1.0) - 0.6000(1.0) = 1.3500 - 0.6000 = \mathbf{0.7500}$$
- **Step 2 ($t = 2$):**
  $$x_2 = 1.3500(x_1) - 0.6000(x_0) = 1.3500(0.7500) - 0.6000(1.0000) = 1.0125 - 0.6000 = \mathbf{0.4125}$$
- **Step 3 ($t = 3$):**
  $$x_3 = 1.3500(x_2) - 0.6000(x_1) = 1.3500(0.4125) - 0.6000(0.7500) = 0.556875 - 0.4500 = \mathbf{0.106875}$$
- **Step 4 ($t = 4$):**
  $$x_4 = 1.3500(x_3) - 0.6000(x_2) = 1.3500(0.106875) - 0.6000(0.4125) = 0.144281 - 0.2475 = \mathbf{-0.103219}$$

The particle smoothly glides from $+1.0 \to +0.75 \to +0.41 \to +0.11$, crosses zero at step 4 to $-0.10$, and dampens exponentially to the equilibrium $x^* = 0$, validating the mechanical oscillator equivalence with exact numerical fidelity!

---

## Part 7: Deep Learning Connection & Application

### 1. The Undisputed Workhorse of Computer Vision
While adaptive optimizers like Adam dominate NLP and Transformers, **SGD with Momentum ($\beta = 0.9$ or $0.95$) remains the gold standard for Convolutional Neural Networks** (ResNet, ConvNeXt, EfficientNet, Mask R-CNN).
- **Generalization Gap**: SGD with momentum systematically achieves lower test error and better out-of-distribution generalization than Adam on image classification benchmarks.
- **Flat vs. Sharp Minima**: SGD with momentum and a properly tuned learning rate schedule traverses saddle-point plateaus and settles into broad, flat minima whose eigenvalues satisfy $\lambda_{\max} \le 2/\alpha$.

### 2. Escaping Saddle Points via Kinetic Energy
In Chapter 5.2, we proved that deep neural networks are dominated by high-dimensional saddle points where $\nabla f(\theta) = 0$ with indefinite Hessian $H$.
- For standard GD, if initialized near a saddle point where the negative eigenvalue direction is small, the gradient is nearly zero, causing GD to stall for hundreds of iterations.
- For Momentum, the particle carries non-zero velocity $v$ accumulated while descending into the saddle region. This residual kinetic energy propels the optimizer straight across the flat saddle plateau and immediately catches the downward escape trajectory.

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `05_optimization/code/05_acceleration_and_momentum.py`. It implements:
1. Exact visual grid verification comparing GD, Polyak, and NAG across 2 iterations.
2. Contraction factor and convergence rate analysis on an ill-conditioned quadratic matrix ($\kappa = 81$).
3. Rosenbrock valley optimization benchmarking GD vs. Polyak vs. NAG.
4. PyTorch parity test verifying Sutskever's reformulated NAG matches PyTorch's native `torch.optim.SGD(..., nesterov=True)`.

Save the code and run it directly in Python 3.
