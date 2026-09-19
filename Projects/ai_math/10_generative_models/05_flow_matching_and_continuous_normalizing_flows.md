# Flow Matching Models & Continuous Normalizing Flows (CNFs)

> **Canonical Literature & Reference Foundations:**
> - *Deep Learning: Foundations and Concepts* (Christopher M. Bishop & Hugh Bishop, Springer 2024, Chapter 19 "Generative Models")
> - *Deep Generative Models: Stanford CS236 Monograph* (Stefano Ermon & Aditya Grover, 2024)
> - *Flow Matching for Generative Modeling* (Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, *ICLR 2023*)
> - *Building Normalizing Flows with Stochastic Interpolants* (Michael S. Albergo & Eric Vanden-Eijnden, *ICLR 2023*)
> - *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* (Xingchao Liu, Chengyue Gong, Qiang Liu, *ICLR 2023*)
> - *Neural Ordinary Differential Equations* (Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud, *NeurIPS 2018 Best Paper Award*)
> - *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)* (Patrick Esser, Sumith Kulal, Andreas Blattmann et al., *CVPR 2024*)
> - *Scalable Diffusion Models with Transformers (DiT)* (William Peebles & Saining Xie, *ICCV 2023*)

---

## 1. Intuition & 101 Motivation

### 1.1 The Fundamental Flaw of Diffusion: The Curvature Penalty
Denoising Diffusion Probabilistic Models (DDPM) construct generative paths using Brownian motion. In the forward diffusion process, isotropic Gaussian noise is injected at every infinitesimal step, which continuously scatters probability mass in all random directions. 

When the reverse process is integrated back from pure static $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ to clean data $\mathbf{x}_0 \sim p_{\text{data}}$, the trajectory taken by the probability flow Ordinary Differential Equation (ODE) or Stochastic Differential Equation (SDE) is **inherently curved and serpentine**.

```
        DIFFUSION SDE / ODE PATH              FLOW MATCHING (OT-CFM) PATH
             (Curved, High Curvature)                 (Straight, Zero Curvature)
        
        Noise x_1                                Noise x_0
           ●                                        ●
            \                                        \
             \                                        \
              )  Curved Brownian Path                  \  Straight Velocity Vector Field
             /   Requires 50 - 1000 steps               \  Requires 1 - 8 Euler steps!
            (                                            \
             \                                            \
              ▼                                            ▼
           Clean Data x_0                           Clean Data x_1
```

Because of this severe trajectory curvature:
1. First-order numerical solvers (like forward Euler) accumulate severe discretization truncation errors $\mathcal{O}(\Delta t)$.
2. Standard diffusion requires **50 to 1,000 sequential network evaluations** or highly complex multi-step adaptive numerical solvers (e.g., DPM-Solver++, DPMSolver-Multistep) to prevent trajectories from derailing off the manifold.
3. Training requires simulation of complex variance schedules ($\beta_t, \bar{\alpha}_t$) and score functions that blow up near $t \to 0$ or $t \to 1$.

---

### 1.2 The Velocity Vector Field Paradigm
**Flow Matching (Lipman et al., 2023)** and **Rectified Flow (Liu et al., 2023)** completely discard Brownian noise injection and score matching. Instead, they frame generative modeling as learning a **deterministic, continuous velocity vector field** $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ that pushes a simple prior distribution $p_0 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ directly onto the target data distribution $p_1 = p_{\text{data}}$ along **straight Euclidean trajectories**:

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t), \quad \mathbf{x}_0 \sim p_0, \quad \mathbf{x}_1 \sim p_{\text{data}}$$

Because the trajectories between noise and data are constructed to be straight lines, the velocity vector field is nearly constant along the path:
$$\mathbf{v}(\mathbf{x}_t, t) \approx \mathbf{x}_1 - \mathbf{x}_0$$
A straight line can be integrated accurately using simple Forward Euler in as few as **1 to 8 steps**, completely unlocking photorealistic real-time inference without complex distillation!

---

### 1.3 Continuous Normalizing Flows (CNFs) Without Architectural Bottlenecks
In classical discrete Normalizing Flows (such as RealNVP, Glow, and MAF), transforming a latent variable $\mathbf{z}$ into an image $\mathbf{x}$ requires an invertible neural network $\mathbf{x} = f_{\boldsymbol{\theta}}(\mathbf{z})$. To compute the exact log-likelihood via the change of variables theorem:
$$\log p(\mathbf{x}) = \log p(\mathbf{z}) - \log \left| \det \frac{\partial f_{\boldsymbol{\theta}}}{\partial \mathbf{z}} \right|$$
the network architecture was forced to have an easily computable Jacobian determinant (e.g., triangular matrices via coupling layers). This severely crippled neural network expressivity.

**Continuous Normalizing Flows (CNFs - Chen et al., 2018)** replace discrete layers with continuous time:
$$\mathbf{x}_1 = \mathbf{x}_0 + \int_0^1 \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \, dt$$
In CNFs, the neural network $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ can be **any arbitrary neural architecture**—a standard Convolutional U-Net, an MLP, or a state-of-the-art Diffusion Transformer (DiT)—with zero invertibility constraints or triangular Jacobian requirements!

Historically, CNFs were intractable because training them required backpropagating through numerical ODE solvers (the Adjoint State Method). **Flow Matching eliminates ODE simulation during training entirely**, providing a simulation-free, closed-form regression loss that is as simple and fast as training DDPM!

---

## 2. Rigorous Mathematical Formulation

### 2.1 Probability Density Paths & The Continuity Equation
Let $\mathbb{R}^d$ denote the data space. We consider a continuous time parameter $t \in [0, 1]$.
- At $t = 0$, we have a known, easily sampleable prior distribution $p_0(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \mathbf{0}, \mathbf{I})$.
- At $t = 1$, we have the empirical target data distribution $p_1(\mathbf{x}) = p_{\text{data}}(\mathbf{x})$.

A **probability density path** is a time-dependent probability density function:
$$p_t: [0, 1] \times \mathbb{R}^d \to \mathbb{R}_{\ge 0}, \quad \text{such that } \int_{\mathbb{R}^d} p_t(\mathbf{x}) \, d\mathbf{x} = 1 \quad \forall t \in [0, 1]$$

A time-dependent **velocity vector field** $\mathbf{v}_t: \mathbb{R}^d \to \mathbb{R}^d$ generates the probability density path $p_t$ if and only if they satisfy the fundamental **Continuity Equation** (from fluid dynamics and the zero-diffusion Fokker-Planck / Liouville equation):

$$\mathbf{\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla \cdot \left( p_t(\mathbf{x}) \mathbf{v}_t(\mathbf{x}) \right) = 0}$$

where $\nabla \cdot \mathbf{F}(\mathbf{x}) = \text{div}(\mathbf{F}(\mathbf{x})) = \sum_{i=1}^d \frac{\partial F_i}{\partial x_i}$ is the divergence operator.

```
                         THE CONTINUITY EQUATION
           Rate of change of density = - Net outgoing flux
                 dp_t(x)/dt = - div( p_t(x) * v_t(x) )
                 
                     ▲ Flux Out: p_t * v_t
                     │
               ┌─────┴─────┐
               │  Density  │
         ◄─────┤  p_t(x)   ├─────► Net Divergence
               │  at (x,t) │
               └─────┬─────┘
                     │
                     ▼
```

---

### 2.2 The Flow Map & Instantaneous Change of Variables
The vector field $\mathbf{v}_t(\mathbf{x})$ defines an autonomous Ordinary Differential Equation (ODE):
$$\frac{d \boldsymbol{\phi}_t(\mathbf{x})}{dt} = \mathbf{v}_t(\boldsymbol{\phi}_t(\mathbf{x})), \quad \boldsymbol{\phi}_0(\mathbf{x}) = \mathbf{x}$$

where $\boldsymbol{\phi}_t: \mathbb{R}^d \to \mathbb{R}^d$ is the **flow map** (diffeomorphism) transporting points from time $0$ to time $t$.
By definition, if $\mathbf{x}_0 \sim p_0(\mathbf{x})$, then the pushed-forward random variable $\mathbf{x}_t = \boldsymbol{\phi}_t(\mathbf{x}_0)$ is distributed according to $p_t$:
$$p_t = [\boldsymbol{\phi}_t]_* p_0$$

**Theorem (Instantaneous Change of Variables - Chen et al., 2018):**
The continuous evolution of the log-probability density along the trajectory $\mathbf{x}_t = \boldsymbol{\phi}_t(\mathbf{x}_0)$ satisfies:
$$\mathbf{\frac{d}{dt} \log p_t(\boldsymbol{\phi}_t(\mathbf{x})) = - \text{div}(\mathbf{v}_t(\boldsymbol{\phi}_t(\mathbf{x}))) = - \text{Tr}\left( \frac{\partial \mathbf{v}_t}{\partial \mathbf{x}} \Big|_{\boldsymbol{\phi}_t(\mathbf{x})} \right)}$$

Integrating both sides from $t = 0$ to $t = 1$:
$$\log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \text{Tr}\left( \frac{\partial \mathbf{v}_t}{\partial \mathbf{x}} \Big|_{\mathbf{x}_t} \right) dt$$

Unlike discrete normalizing flows where computing $\det(J)$ takes $\mathcal{O}(d^3)$ time, the trace $\text{Tr}(\frac{\partial \mathbf{v}}{\partial \mathbf{x}})$ can be computed unbiasedly in $\mathcal{O}(d)$ time using **Hutchinson's Trace Estimator**:
$$\text{Tr}\left( \frac{\partial \mathbf{v}_t}{\partial \mathbf{x}} \right) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \boldsymbol{\epsilon}^T \left( \frac{\partial \mathbf{v}_t}{\partial \mathbf{x}} \boldsymbol{\epsilon} \right) \right]$$
which requires only a single vector-Jacobian product (VJP) in PyTorch autograd!

---

### 2.3 The Flow Matching Objective
If we knew the ground-truth marginal vector field $\mathbf{u}_t(\mathbf{x})$ that generates the target probability path $p_t(\mathbf{x})$, we could train a neural network $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ by direct Mean Squared Error (MSE) regression:

$$\mathbf{\mathcal{L}_{\text{FM}}(\boldsymbol{\theta}) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], \, \mathbf{x} \sim p_t(\mathbf{x})} \left[ \left\| \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \mathbf{u}_t(\mathbf{x}) \right\|_2^2 \right]}$$

#### The Intractability Dilemma:
In real-world generative modeling, we only have discrete empirical samples from the data distribution $\mathbf{x}_1 \sim p_{\text{data}}(\mathbf{x})$. 
The marginal density $p_t(\mathbf{x})$ and the true marginal velocity field $\mathbf{u}_t(\mathbf{x})$ are **completely intractable** because they require integrating over the entire unknown data distribution:
$$p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1) \, d\mathbf{x}_1, \qquad \mathbf{u}_t(\mathbf{x}) = \int \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \frac{p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1)}{p_t(\mathbf{x})} \, d\mathbf{x}_1$$

This makes evaluating $\mathcal{L}_{\text{FM}}(\boldsymbol{\theta})$ directly impossible.

---

## 3. The Conditional Flow Matching (CFM) Theorem & Proof

To bypass this intractability, Lipman et al. (2023) and Albergo & Vanden-Eijnden (2023) introduced **Conditional Flow Matching (CFM)**.

Instead of matching the intractable marginal vector field $\mathbf{u}_t(\mathbf{x})$, we condition on a specific target data point $\mathbf{x}_1 \sim q(\mathbf{x}_1)$ (or a data pair $(\mathbf{x}_0, \mathbf{x}_1)$) and define a simple, tractable **conditional probability path** $p_t(\mathbf{x} \mid \mathbf{x}_1)$ and **conditional vector field** $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1)$.

The **Conditional Flow Matching (CFM) Objective** is defined as:
$$\mathbf{\mathcal{L}_{\text{CFM}}(\boldsymbol{\theta}) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], \, \mathbf{x}_1 \sim q(\mathbf{x}_1), \, \mathbf{x} \sim p_t(\mathbf{x} \mid \mathbf{x}_1)} \left[ \left\| \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \right\|_2^2 \right]}$$

---

### 3.1 The Marginal Vector Field Theorem
**Theorem 1 (Lipman et al., 2023):**
Let $p_t(\mathbf{x} \mid \mathbf{x}_1)$ be a conditional probability path generated by conditional vector field $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1)$, satisfying the conditional continuity equation:
$$\frac{\partial p_t(\mathbf{x} \mid \mathbf{x}_1)}{\partial t} + \nabla \cdot \left( p_t(\mathbf{x} \mid \mathbf{x}_1) \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \right) = 0$$

Define the marginal density $p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1) d\mathbf{x}_1$ and the marginal vector field:
$$\mathbf{u}_t(\mathbf{x}) = \int \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \frac{p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1)}{p_t(\mathbf{x})} d\mathbf{x}_1$$

Then:
1. $\mathbf{u}_t(\mathbf{x})$ generates the marginal probability path $p_t(\mathbf{x})$, satisfying $\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla \cdot (p_t(\mathbf{x}) \mathbf{u}_t(\mathbf{x})) = 0$.
2. The parameter gradients of the intractable objective $\mathcal{L}_{\text{FM}}(\boldsymbol{\theta})$ and the tractable objective $\mathcal{L}_{\text{CFM}}(\boldsymbol{\theta})$ are **strictly identical**:
   $$\mathbf{\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{FM}}(\boldsymbol{\theta}) \equiv \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{CFM}}(\boldsymbol{\theta})}$$

---

### 3.2 First-Principles Derivation and Proof

**Part 1: Proving that $\mathbf{u}_t(\mathbf{x})$ generates $p_t(\mathbf{x})$:**

Multiply the definition of $\mathbf{u}_t(\mathbf{x})$ by $p_t(\mathbf{x})$:
$$p_t(\mathbf{x}) \mathbf{u}_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{x}_1) \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1) d\mathbf{x}_1$$

Take the divergence with respect to $\mathbf{x}$ on both sides:
$$\nabla \cdot \left( p_t(\mathbf{x}) \mathbf{u}_t(\mathbf{x}) \right) = \int \nabla \cdot \left( p_t(\mathbf{x} \mid \mathbf{x}_1) \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \right) q(\mathbf{x}_1) d\mathbf{x}_1$$

Substitute the conditional continuity equation $\nabla \cdot \left( p_t(\mathbf{x} \mid \mathbf{x}_1) \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \right) = -\frac{\partial p_t(\mathbf{x} \mid \mathbf{x}_1)}{\partial t}$:
$$\begin{aligned}
\nabla \cdot \left( p_t(\mathbf{x}) \mathbf{u}_t(\mathbf{x}) \right) &= - \int \frac{\partial p_t(\mathbf{x} \mid \mathbf{x}_1)}{\partial t} q(\mathbf{x}_1) d\mathbf{x}_1 \\
&= - \frac{\partial}{\partial t} \int p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1) d\mathbf{x}_1 \\
&= - \frac{\partial p_t(\mathbf{x})}{\partial t}
\end{aligned}$$

Rearranging gives:
$$\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla \cdot \left( p_t(\mathbf{x}) \mathbf{u}_t(\mathbf{x}) \right) = 0 \quad \blacksquare$$

---

**Part 2: Proving Gradient Equivalence $\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{FM}} \equiv \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{CFM}}$:**

Expand the quadratic Euclidean norm in both loss functions:
$$\|\mathbf{v}_{\boldsymbol{\theta}} - \mathbf{u}\|^2 = \|\mathbf{v}_{\boldsymbol{\theta}}\|^2 - 2 \langle \mathbf{v}_{\boldsymbol{\theta}}, \mathbf{u} \rangle + \|\mathbf{u}\|^2$$

Since the term $\|\mathbf{u}\|^2$ does not depend on the network parameters $\boldsymbol{\theta}$, taking the gradient with respect to $\boldsymbol{\theta}$ eliminates it:
$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{FM}}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \int_0^1 \int_{\mathbb{R}^d} \left( \|\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2 - 2 \langle \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t), \mathbf{u}_t(\mathbf{x}) \rangle \right) p_t(\mathbf{x}) \, d\mathbf{x} \, dt$$

Now expand the inner product term:
$$\begin{aligned}
\int_{\mathbb{R}^d} \langle \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t), \mathbf{u}_t(\mathbf{x}) \rangle p_t(\mathbf{x}) d\mathbf{x} &= \int_{\mathbb{R}^d} \langle \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t), \int_{\mathbb{R}^d} \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \frac{p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1)}{p_t(\mathbf{x})} d\mathbf{x}_1 \rangle p_t(\mathbf{x}) d\mathbf{x} \\
&= \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} \langle \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t), \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \rangle p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1) d\mathbf{x}_1 d\mathbf{x}
\end{aligned}$$

Notice that the $p_t(\mathbf{x})$ denominator cancelled out entirely!
Also note that:
$$\int_{\mathbb{R}^d} \|\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2 p_t(\mathbf{x}) d\mathbf{x} = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} \|\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2 p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1) d\mathbf{x}_1 d\mathbf{x}$$

Therefore:
$$\begin{aligned}
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{FM}}(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} \int_0^1 \int \int \left( \|\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2 - 2 \langle \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t), \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \rangle \right) p_t(\mathbf{x} \mid \mathbf{x}_1) q(\mathbf{x}_1) d\mathbf{x}_1 d\mathbf{x} dt \\
&= \nabla_{\boldsymbol{\theta}} \mathbb{E}_{t, \mathbf{x}_1 \sim q(\mathbf{x}_1), \mathbf{x} \sim p_t(\mathbf{x} \mid \mathbf{x}_1)} \left[ \|\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1)\|^2 \right] \\
&= \mathbf{\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{CFM}}(\boldsymbol{\theta})} \quad \blacksquare
\end{aligned}$$

> **Significance:** Minimizing the tractable conditional loss $\mathcal{L}_{\text{CFM}}$ trains the neural network $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ to learn the exact true marginal vector field $\mathbf{u}_t(\mathbf{x})$ of the data!

---

## 4. Optimal Transport Conditional Flow Matching (OT-CFM)

### 4.1 Gaussian Probability Paths
To instantiate Conditional Flow Matching, we choose a family of Gaussian conditional probability paths:
$$p_t(\mathbf{x} \mid \mathbf{x}_1) = \mathcal{N}\left( \mathbf{x}; \boldsymbol{\mu}_t(\mathbf{x}_1), \sigma_t^2(\mathbf{x}_1) \mathbf{I} \right)$$
where the mean $\boldsymbol{\mu}_t$ and standard deviation $\sigma_t$ satisfy boundary conditions:
- At $t = 0$: $\boldsymbol{\mu}_0 = \mathbf{0}, \sigma_0 = 1$ (standard normal prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$).
- At $t = 1$: $\boldsymbol{\mu}_1 = \mathbf{x}_1, \sigma_1 = \sigma_{\min} \approx 0$ (concentrated sharply at data point $\mathbf{x}_1$).

Any such Gaussian path is generated by the analytical conditional vector field:
$$\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{\sigma'_t(\mathbf{x}_1)}{\sigma_t(\mathbf{x}_1)} \left( \mathbf{x} - \boldsymbol{\mu}_t(\mathbf{x}_1) \right) + \boldsymbol{\mu}'_t(\mathbf{x}_1)$$

---

### 4.2 The Optimal Transport Interpolant
In **Optimal Transport Conditional Flow Matching (OT-CFM)** (Lipman et al., 2023), we couple initial noise $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and data $\mathbf{x}_1 \sim q(\mathbf{x}_1)$. We choose the simplest possible path: a **straight Euclidean linear interpolation**:

$$\mathbf{\mathbf{x}_t = (1 - t) \mathbf{x}_0 + t \mathbf{x}_1, \quad t \in [0, 1]}$$

Differentiating directly with respect to time $t$:
$$\mathbf{u}_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = \mathbf{\mathbf{x}_1 - \mathbf{x}_0}$$

The target velocity is **completely constant along the entire trajectory** and equals the displacement vector $\mathbf{x}_1 - \mathbf{x}_0$!

```
                OPTIMAL TRANSPORT DISPLACEMENT INTERPOLANT
        
         x_0 ~ N(0, I)                     x_t                      x_1 ~ p_data
            ●───────────────────────────────●────────────────────────────►●
           t=0                             t=0.5                         t=1
           
                    Velocity vector: u_t(x_t | x_0, x_1) = x_1 - x_0
                           (Constant direction and speed!)
```

### 4.3 Why is this "Optimal Transport"?
The Monge-Kantorovich Optimal Transport problem seeks a transport map $T: \mathbb{R}^d \to \mathbb{R}^d$ minimizing the quadratic Wasserstein kinetic energy:
$$\mathcal{W}_2^2(p_0, p_1) = \inf_{\mathbf{v}_t} \int_0^1 \int_{\mathbb{R}^d} \|\mathbf{v}_t(\mathbf{x})\|_2^2 \, p_t(\mathbf{x}) \, d\mathbf{x} \, dt$$

The straight-line interpolant $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ with constant velocity $\mathbf{v}_t = \mathbf{x}_1 - \mathbf{x}_0$ is the **exact analytical geodesic** of the 2-Wasserstein metric space. 
It minimizes the kinetic energy action $\int_0^1 \|\mathbf{v}_t\|^2 dt = \|\mathbf{x}_1 - \mathbf{x}_0\|^2$, meaning probability mass is transported from noise to data along the shortest possible distance with minimum acceleration.

---

### 4.4 The OT-CFM Training Algorithm
The resulting training algorithm is remarkably clean and requires only standard PyTorch autograd:

```
Algorithm: Optimal Transport Conditional Flow Matching (OT-CFM)
Input: Clean dataset D, Neural network v_theta(x, t), batch size B.
1. Sample clean data batch:       x_1 ~ D
2. Sample pure Gaussian noise:     x_0 ~ N(0, I)
3. Sample uniform time steps:      t ~ U[0, 1]
4. Compute intermediate points:   x_t = (1 - t) * x_0 + t * x_1
5. Target velocity:               u_t = x_1 - x_0
6. Predict velocity:              v_pred = v_theta(x_t, t)
7. Compute MSE Loss:              L = || v_pred - u_t ||^2
8. Take gradient step:            theta <- theta - gamma * grad(L)
```

---

## 5. Rectified Flow & The Straight-and-Fast Paradigm (Liu et al., 2023)

While OT-CFM constructs straight lines between each pair $(\mathbf{x}_0, \mathbf{x}_1)$, when multiple independent trajectories from different noise vectors cross paths in the intermediate space, their velocity vectors conflict.

The marginal vector field $\mathbf{u}_t(\mathbf{x})$ is the **average** of all conflicting conditional velocities passing through $\mathbf{x}$ at time $t$:
$$\mathbf{u}_t(\mathbf{x}) = \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim q} \left[ \mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x} \right]$$

When trajectories cross, this conditional expectation causes the marginal velocity field to bend, introducing **curvature into the learned ODE trajectories**!

```
      TRAJECTORY CROSSING CAUSES CURVATURE IN 1-RECTIFIED FLOW
      
      x_0 (Noise)                                            x_1 (Data)
      (A) ●──────────────────────\     /───────────────────────►● Mode 1
                                  \   /
                                    X   <--- TRAJECTORY CROSSING POINT!
                                  /   \      (conflicting velocities)
      (B) ●──────────────────────/     \───────────────────────►● Mode 2
```

---

### 5.1 The 2-Rectified Flow (Re-Flow) Algorithm
Liu, Gong, & Liu (2023) introduced the revolutionary **Re-Flow** procedure to systematically eliminate trajectory crossings and straighten the flow map.

**The Re-Flow Algorithm:**
1. **Train 1-Rectified Flow:** Train initial network $\mathbf{v}_{\boldsymbol{\theta}_1}(\mathbf{x}, t)$ on empirical independent pairs $(\mathbf{x}_0, \mathbf{x}_1) \sim p_0 \times p_1$.
2. **Generate Straight Pairs via ODE Simulation:**
   Sample noise $\mathbf{x}_0 \sim p_0$. Integrate the learned ODE forward from $t = 0$ to $t = 1$ using an accurate multi-step solver (e.g., RK4):
   $$\hat{\mathbf{x}}_1 = \mathbf{x}_0 + \int_0^1 \mathbf{v}_{\boldsymbol{\theta}_1}(\mathbf{x}_t, t) \, dt$$
   This yields a new synthetic paired coupling $(\mathbf{x}_0, \hat{\mathbf{x}}_1) \sim \pi_{\text{reflow}}$.
3. **Train 2-Rectified Flow:**
   Train a new network $\mathbf{v}_{\boldsymbol{\theta}_2}$ (or continue fine-tuning) on the new straight pairs:
   $$\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t \hat{\mathbf{x}}_1, \qquad \mathbf{u}_t = \hat{\mathbf{x}}_1 - \mathbf{x}_0$$
   $$\mathcal{L}_{\text{2-Rectified}}(\boldsymbol{\theta}_2) = \mathbb{E}_{t, (\mathbf{x}_0, \hat{\mathbf{x}}_1)} \left[ \| \mathbf{v}_{\boldsymbol{\theta}_2}(\mathbf{x}_t, t) - (\hat{\mathbf{x}}_1 - \mathbf{x}_0) \|_2^2 \right]$$

```
                   THE RE-FLOW TRAJECTORY STRAIGHTENING
        
       1-Rectified Flow (Bends at crossings)       2-Rectified Flow (Completely Straight!)
             Noise                  Data                 Noise                  Data
             ● \                    / ●                  ● ───────────────────────► ●
                \                  /                        (No crossings!)
                 ●────────────────●                      ● ───────────────────────► ●
                /                  \
             ● /                    \ ●                  ● ───────────────────────► ●
```

### 5.2 Mathematical Monotonicity of Re-Flow
**Theorem (Straightening Effect of Re-Flow - Liu et al., 2023):**
Let $L(Z)$ denote the expected straightness / transport cost $L(Z) = \mathbb{E}[\|\mathbf{x}_0 - \hat{\mathbf{x}}_1\|^2]$.
Then every iteration of the Re-Flow procedure **monotonically decreases or preserves the transport cost**:
$$\mathbb{E}[\|\mathbf{x}_0 - \hat{\mathbf{x}}_1^{(k+1)}\|^2] \le \mathbb{E}[\|\mathbf{x}_0 - \hat{\mathbf{x}}_1^{(k)}\|^2]$$
with equality holding if and only if the trajectories are already perfectly straight non-crossing lines!

As trajectories become straight lines, **1-step forward Euler sampling**:
$$\mathbf{x}_1 \approx \mathbf{x}_0 + 1.0 \cdot \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_0, 0)$$
becomes an exact integration with zero truncation error, enabling **sub-millisecond single-step generative generation** (e.g., InstaFlow, SD3 Turbo).

---

## 6. Classifier-Free Guidance (CFG) in Flow Matching

In text-to-image or class-conditional generation (e.g., Stable Diffusion 3, FLUX.1), we condition the vector field on conditioning vector $\mathbf{c}$ (such as T5 or CLIP text embeddings):
$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \mathbf{c})$$

During training, condition $\mathbf{c}$ is dropped with probability $p_{\text{uncond}} \approx 0.10$ and replaced with the empty/null conditioning token $\emptyset$.

At inference time, the guided velocity vector field $\tilde{\mathbf{v}}_{\boldsymbol{\theta}}$ is computed by extrapolating along the conditioning vector:

$$\mathbf{\tilde{\mathbf{v}}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \mathbf{c}) = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \emptyset) + s \cdot \left( \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \emptyset) \right)}$$

where $s \ge 1.0$ is the CFG scale (typically $s \in [3.0, 7.5]$).

### Geometric Meaning in Velocity Space:
- $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \emptyset)$ points in the direction of the general natural image manifold.
- $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \emptyset)$ is the **semantic velocity component** steering the flow specifically toward the text prompt semantics.
- Scaling by $s > 1$ amplifies the prompt velocity, strongly suppressing off-prompt image modes and boosting visual sharpness.

---

## 7. Numerical ODE Solvers for Flow Matching Sampling

Given a trained neural velocity field $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$, we generate new samples by integrating the ODE from $t = 0$ to $t = 1$:
$$\mathbf{x}_1 = \mathbf{x}_0 + \int_0^1 \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \, dt, \quad \mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Let $0 = t_0 < t_1 < \dots < t_N = 1$ be a partition of $[0, 1]$ into $N$ steps with step size $h = \frac{1}{N}$.

### 7.1 Forward Euler Method (First-Order, $\mathcal{O}(h)$)
For $n = 0, 1, \dots, N-1$:
$$\mathbf{x}_{n+1} = \mathbf{x}_n + h \cdot \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_n, t_n)$$
- **Function Evaluations per Step:** 1
- **Best for:** Rectified Flow models after Reflow / Distillation ($N \in [2, 8]$).

---

### 7.2 Midpoint Method / Runge-Kutta 2 (Second-Order, $\mathcal{O}(h^2)$)
For $n = 0, 1, \dots, N-1$:
$$\begin{aligned}
\mathbf{k}_1 &= \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_n, t_n) \\
\mathbf{x}_{\text{mid}} &= \mathbf{x}_n + \frac{h}{2} \mathbf{k}_1 \\
\mathbf{k}_2 &= \mathbf{v}_{\boldsymbol{\theta}}\left(\mathbf{x}_{\text{mid}}, t_n + \frac{h}{2}\right) \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + h \cdot \mathbf{k}_2
\end{aligned}$$
- **Function Evaluations per Step:** 2
- **Best for:** Standard 1-Rectified Flow ($N \in [10, 25]$).

---

### 7.3 Classical Runge-Kutta 4 (RK4, Fourth-Order, $\mathcal{O}(h^4)$)
For $n = 0, 1, \dots, N-1$:
$$\begin{aligned}
\mathbf{k}_1 &= \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_n, t_n) \\
\mathbf{k}_2 &= \mathbf{v}_{\boldsymbol{\theta}}\left(\mathbf{x}_n + \frac{h}{2}\mathbf{k}_1, t_n + \frac{h}{2}\right) \\
\mathbf{k}_3 &= \mathbf{v}_{\boldsymbol{\theta}}\left(\mathbf{x}_n + \frac{h}{2}\mathbf{k}_2, t_n + \frac{h}{2}\right) \\
\mathbf{k}_4 &= \mathbf{v}_{\boldsymbol{\theta}}\left(\mathbf{x}_n + h\mathbf{k}_3, t_n + h\right) \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \frac{h}{6} \left( \mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4 \right)
\end{aligned}$$
- **Function Evaluations per Step:** 4
- **Best for:** Generating ground-truth training pairs during the Re-Flow step.

---

## 8. Prof. Tom Yeh "AI by Hand" Visual Grids & Solved Numerical Problems

### 8.1 Concrete Problem 1: Step-by-Step OT-CFM Training Arithmetic

**Scenario:**
Consider a 2D scalar problem where:
- Initial Gaussian noise sample at $t = 0$: $\mathbf{x}_0 = \begin{bmatrix} -1.0 \\ 2.0 \end{bmatrix}$
- Clean data point at $t = 1$: $\mathbf{x}_1 = \begin{bmatrix} 3.0 \\ -2.0 \end{bmatrix}$
- Randomly sampled training timestep: $t = 0.40$

**Questions:**
1. Compute the intermediate interpolated state $\mathbf{x}_t$.
2. Compute the exact analytical target velocity vector $\mathbf{u}_t$.
3. Assume our neural network currently outputs prediction:
   $$\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = \begin{bmatrix} 3.20 \\ -3.50 \end{bmatrix}$$
   Compute the Mean Squared Error (MSE) loss for this sample.
4. If the velocity output is parameterized by a single linear layer $\mathbf{v} = \mathbf{W} \mathbf{x}_t$ where $\mathbf{W} = \begin{bmatrix} 1.0 & 2.0 \\ -1.0 & -1.0 \end{bmatrix}$, calculate the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$.

---

**Step-by-Step Hand Calculations:**

**1. Intermediate State $\mathbf{x}_t$ at $t = 0.40$:**
$$\begin{aligned}
\mathbf{x}_{0.40} &= (1 - t) \mathbf{x}_0 + t \mathbf{x}_1 \\
&= (1 - 0.40) \begin{bmatrix} -1.0 \\ 2.0 \end{bmatrix} + 0.40 \begin{bmatrix} 3.0 \\ -2.0 \end{bmatrix} \\
&= 0.60 \begin{bmatrix} -1.0 \\ 2.0 \end{bmatrix} + 0.40 \begin{bmatrix} 3.0 \\ -2.0 \end{bmatrix} \\
&= \begin{bmatrix} -0.60 \\ 1.20 \end{bmatrix} + \begin{bmatrix} 1.20 \\ -0.80 \end{bmatrix} = \mathbf{\begin{bmatrix} +0.60 \\ +0.40 \end{bmatrix}}
\end{aligned}$$

**2. Target Velocity $\mathbf{u}_t$:**
$$\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0 = \begin{bmatrix} 3.0 - (-1.0) \\ -2.0 - 2.0 \end{bmatrix} = \mathbf{\begin{bmatrix} +4.00 \\ -4.00 \end{bmatrix}}$$

**3. Velocity Prediction Error & Loss:**
$$\mathbf{e} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) - \mathbf{u}_t = \begin{bmatrix} 3.20 - 4.00 \\ -3.50 - (-4.00) \end{bmatrix} = \begin{bmatrix} -0.80 \\ +0.50 \end{bmatrix}$$
The squared Euclidean loss is:
$$\mathcal{L} = \|\mathbf{e}\|_2^2 = (-0.80)^2 + (+0.50)^2 = 0.6400 + 0.2500 = \mathbf{0.8900}$$
Mean Squared Error per coordinate ($d = 2$):
$$\text{MSE} = \frac{0.8900}{2} = \mathbf{0.4450}$$

**4. Parameter Gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$:**
For linear prediction $\mathbf{v}_{\text{linear}} = \mathbf{W} \mathbf{x}_t$ with $\mathbf{W} = \begin{bmatrix} 1.0 & 2.0 \\ -1.0 & -1.0 \end{bmatrix}$:
$$\mathbf{v}_{\text{linear}} = \begin{bmatrix} 1.0(0.60) + 2.0(0.40) \\ -1.0(0.60) - 1.0(0.40) \end{bmatrix} = \begin{bmatrix} 0.60 + 0.80 \\ -0.60 - 0.40 \end{bmatrix} = \mathbf{\begin{bmatrix} +1.40 \\ -1.00 \end{bmatrix}}$$
The error vector for this linear layer is:
$$\mathbf{e}_{\text{linear}} = \mathbf{v}_{\text{linear}} - \mathbf{u}_t = \begin{bmatrix} 1.40 - 4.00 \\ -1.00 - (-4.00) \end{bmatrix} = \mathbf{\begin{bmatrix} -2.60 \\ +3.00 \end{bmatrix}}$$
The loss gradient with respect to predicted velocity:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}} = 2 \mathbf{e}_{\text{linear}} = 2 \begin{bmatrix} -2.60 \\ +3.00 \end{bmatrix} = \begin{bmatrix} -5.20 \\ +6.00 \end{bmatrix}$$
Using the outer product identity $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{v}}\right) \mathbf{x}_t^T$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \begin{bmatrix} -5.20 \\ +6.00 \end{bmatrix} \begin{bmatrix} 0.60 & 0.40 \end{bmatrix} = \mathbf{\begin{bmatrix} -3.120 & -2.080 \\ +3.600 & +2.400 \end{bmatrix}}$$

---

### 8.2 Concrete Problem 2: Hand Trace of ODE Sampling (Euler vs. Midpoint vs. RK4)

**Scenario:**
Suppose a trained 1D velocity field is given by:
$$v(x, t) = 2.0 \, x + 1.0$$
We wish to integrate from initial noise $x_0 = 1.00$ at $t = 0$ to $t = 1.0$ in **$N = 2$ steps** (step size $h = 0.50$).

#### Method A: Forward Euler ($N = 2$ steps, $h = 0.50$)
1. **Step 1 ($t = 0.0 \to 0.5$):**
   - $v(x_0, t_0) = 2.0(1.00) + 1.0 = 3.00$
   - $x_{0.5} = x_0 + h \cdot v = 1.00 + 0.50(3.00) = 1.00 + 1.50 = \mathbf{2.5000}$
2. **Step 2 ($t = 0.5 \to 1.0$):**
   - $v(x_{0.5}, t_{0.5}) = 2.0(2.50) + 1.0 = 5.0 + 1.0 = 6.00$
   - $x_{1.0} = 2.50 + 0.50(6.00) = 2.50 + 3.00 = \mathbf{5.5000}$

---

#### Method B: Midpoint / RK2 ($N = 2$ steps, $h = 0.50$)
1. **Step 1 ($t = 0.0 \to 0.5$):**
   - $k_1 = v(1.00, 0.0) = 3.00$
   - $x_{\text{mid}} = 1.00 + \frac{0.50}{2}(3.00) = 1.00 + 0.75 = 1.75$
   - $k_2 = v(1.75, 0.25) = 2.0(1.75) + 1.0 = 3.50 + 1.0 = 4.50$
   - $x_{0.5} = 1.00 + 0.50(4.50) = 1.00 + 2.25 = \mathbf{3.2500}$
2. **Step 2 ($t = 0.5 \to 1.0$):**
   - $k_1 = v(3.25, 0.5) = 2.0(3.25) + 1.0 = 6.50 + 1.0 = 7.50$
   - $x_{\text{mid}} = 3.25 + \frac{0.50}{2}(7.50) = 3.25 + 1.875 = 5.125$
   - $k_2 = v(5.125, 0.75) = 2.0(5.125) + 1.0 = 10.25 + 1.0 = 11.25$
   - $x_{1.0} = 3.25 + 0.50(11.25) = 3.25 + 5.625 = \mathbf{8.8750}$

---

#### Analytical Exact Solution:
The ODE $\frac{dx}{dt} = 2x + 1$ with $x(0) = 1$:
$$x(t) = \frac{3}{2} e^{2t} - \frac{1}{2}$$
At $t = 1.0$:
$$x_{\text{exact}}(1.0) = 1.5 e^{2.0} - 0.5 = 1.5(7.389056) - 0.5 = 11.08358 - 0.5 = \mathbf{10.5836}$$

#### Summary Numerical Comparison Table:
| Integration Method | Estimated $x(1.0)$ | Exact $x(1.0)$ | Absolute Error | Relative Error | Number of Network Passes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Forward Euler ($N=2$)** | $5.5000$ | $10.5836$ | $5.0836$ | $48.03\%$ | 2 passes |
| **Midpoint / RK2 ($N=2$)** | $8.8750$ | $10.5836$ | $1.7086$ | $16.14\%$ | 4 passes |
| **Exact Analytical ODE** | $\mathbf{10.5836}$ | $10.5836$ | $0.0000$ | $0.00\%$ | Closed-form |

> **Key Insight:** When the vector field is curved ($\frac{\partial v}{\partial x} \gg 0$), naive Euler accumulates large errors. This mathematically demonstrates why **Re-Flow is so vital**: Re-Flow forces $v(x, t)$ to become constant along trajectories ($v(x, t) \approx x_1 - x_0$), making Euler error drop to virtually zero!

---

## 9. Architectural Comparison Matrix: Diffusion vs. Flow Matching

| Architectural Dimension | Standard DDPM (Ho et al., 2020) | DDIM (Song et al., 2020) | Score SDE (Song et al., 2021) | Optimal Transport CFM (Lipman et al., 2023) | Rectified Flow (Liu et al., 2023) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fundamental Model** | Discrete Markov Chain | Non-Markovian Gaussian Chain | Itô Stochastic Differential Eq. | Continuous Normalizing Flow (ODE) | Continuous Normalizing Flow (ODE) |
| **Target Variable** | Noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | Noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | Stein Score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ | Velocity field $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$ | Velocity field $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$ |
| **Path Geometry** | Curved Brownian Motion | Curved Deterministic ODE | Curved SDE / Probability Flow | Straight Euclidean Geodesic | Straight Non-Crossing Lines (Reflow) |
| **Prior Distribution** | $p_T = \mathcal{N}(\mathbf{0}, \mathbf{I})$ | $p_T = \mathcal{N}(\mathbf{0}, \mathbf{I})$ | $p_T = \mathcal{N}(\mathbf{0}, \mathbf{I})$ | $p_0 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ | Arbitrary $p_0$ (Noise or Domain A) |
| **Sampling Steps** | $1,000$ steps | $20 - 50$ steps | $50 - 500$ steps (Euler-Maruyama) | $10 - 25$ steps (Euler / RK2) | **1 - 4 steps** (Euler after Reflow) |
| **Exact Likelihood** | Lower Bound (VLB) | Intractable | Exact via Probability Flow ODE | Exact via Continuous Change of Variables | Exact via Continuous Change of Variables |
| **Production Backbone** | U-Net | U-Net | U-Net / DiT | Diffusion Transformer (DiT / MMDiT) | Diffusion Transformer (DiT / MMDiT) |
| **Flagship Implementations** | DALL-E 2, Imagen | Stable Diffusion 1.5 / 2.1 | EDM (NVIDIA) | Meta Flow Matching, TorchCFM | **Stable Diffusion 3, FLUX.1, Midjourney v6** |

---

## 10. Production Gotchas, Do's and Don'ts

### The Timestep Convention Gotcha ($t=0$ vs $t=1$)
- **Gotcha:** In Diffusion literature (DDPM/DDIM), $t = 0$ is clean data and $t = 1$ (or $t = T$) is pure noise. In Flow Matching and Continuous Normalizing Flow literature, **the direction is inverted**: $t = 0$ is the Gaussian prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ and $t = 1$ is clean data!
- **Consequence:** Reversing the time index flips the sign of the velocity vector field: $\mathbf{v}_{\text{reverse}} = -\mathbf{v}_{\text{forward}}$. Always verify which boundary condition your codebase uses.

### The Boundary Singularity Avoidance ($t \in [\epsilon, 1 - \epsilon]$)
- **Problem:** In variance-exploding or variance-preserving formulations where $\sigma_t \to 0$ as $t \to 1$, division by $\sigma_t$ causes NaN overflow in FP16 / BF16 mixed-precision training.
- **Remedy:** Clamp the minimum noise standard deviation $\sigma_{\min} = 10^{-4}$ or sample $t \sim \mathcal{U}[\epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-5}$.

### Velocity Parameterization vs. $x$-prediction vs. Noise prediction
- Modern Rectified Flow models (such as Stable Diffusion 3) parameterize the neural network to output the velocity directly: $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$.
- Alternatively, one can predict the clean data $\hat{\mathbf{x}}_1$ and recover velocity via:
  $$\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = \frac{\hat{\mathbf{x}}_1 - \mathbf{x}_t}{1 - t}$$
  However, as $t \to 1$, $1 - t \to 0$, causing extreme numerical instability. Direct velocity prediction is strictly superior.

---

## 11. Code Implementation & Verification

The accompanying production-grade PyTorch script implements:
1. **The Exact OT-CFM Loss Module:** Clean linear interpolation $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ and MSE loss.
2. **Velocity Vector Field Backbone:** Multi-layer residual network with sinusoidal positional time embeddings.
3. **ODE Numerical Samplers:** Vectorized Forward Euler, Midpoint (RK2), and Classical RK4 solvers.
4. **The 2-Rectified Flow (Re-Flow) Procedure:** End-to-end synthetic pair generation and trajectory straightening.
5. **Exact Numerical Unit Tests:** Automated verification of Hand Calculation Problems 1 and 2 matching manual arithmetic to $< 10^{-6}$.

See implementation in:
[`10_generative_models/code/05_flow_matching_and_continuous_normalizing_flows.py`](./code/05_flow_matching_and_continuous_normalizing_flows.py)
