# Generative Adversarial Networks (GANs) & Wasserstein GAN

---

## 1. Intuition & 101 Motivation

### The Game-Theoretic Paradigm Shift
Prior to 2014, generative modeling was dominated by explicit density estimation: maximizing $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ directly or via variational lower bounds ($\text{ELBO}$).
In 2014, Ian Goodfellow introduced **Generative Adversarial Networks (GANs)**, abandoning explicit likelihoods in favor of a two-player zero-sum game:
1. **The Generator ($G_{\boldsymbol{\theta}}$):** Takes standard normal noise $\mathbf{z} \sim p_{\mathbf{z}}$ and maps it to the data space $\mathbf{x}_{\text{fake}} = G_{\boldsymbol{\theta}}(\mathbf{z})$. Its goal is to fool the discriminator.
2. **The Discriminator ($D_{\boldsymbol{\phi}}$):** Takes a sample $\mathbf{x}$ (either real from $p_{\text{data}}$ or fake from $p_g$) and outputs a scalar probability $D_{\boldsymbol{\phi}}(\mathbf{x}) \in [0, 1]$ indicating whether $\mathbf{x}$ is a genuine data point.

```
                    THE ADVERSARIAL TRAINING LOOP
        Real Data x ~ p_data ────► ┌────────────────┐
                                   │ Discriminator  │ ──► D(x) in [0, 1]
        Noise z ~ N(0, I)          │    D_phi       │
               │                   └────────────────┘
               ▼                           ▲
        ┌──────────────┐                   │
        │  Generator   │ ──► x_fake = G(z) ┘
        │   G_theta    │
        └──────────────┘
               ▲
               │ Backprop gradients from D: dLoss / dG
               └───────────────────────────────────────
```

### The Breakdown of Vanilla GANs
Despite producing razor-sharp images, vanilla GANs were notoriously unstable:
1. **Vanishing Gradients:** When the discriminator gets too good early in training, its output for generated samples saturates to $D(G(\mathbf{z})) \approx 0$, causing gradients flowing to the generator to completely vanish.
2. **Mode Collapse:** The generator discovers a single realistic output that fools the discriminator (e.g., generating only the number "8" on MNIST) and collapses entirely onto that mode, refusing to generate any other class.

### The Wasserstein Revolution (WGAN & WGAN-GP)
Martin Arjovsky et al. (2017) proved that the core defect of vanilla GANs stems from minimizing **Jensen-Shannon Divergence ($D_{\text{JS}}$)** across lower-dimensional manifolds in high-dimensional space.
By replacing $D_{\text{JS}}$ with the **Wasserstein-1 (Earth Mover's) Distance**, the discriminator becomes a **Critic** whose outputs are unbounded real numbers, providing a continuous, smooth, non-vanishing gradient signal everywhere in space!

---

## 2. Rigorous Mathematical Formulation

### 2.1 Vanilla GAN Minimax Game

The original GAN objective is formulated as a two-player minimax game:
$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log D(\mathbf{x}) \right] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log\left( 1 - D(G(\mathbf{z})) \right) \right]$$

Using the push-forward measure $p_g$ induced by $G(\mathbf{z})$:
$$V(D, G) = \int_{\mathcal{X}} \left( p_{\text{data}}(\mathbf{x}) \log D(\mathbf{x}) + p_g(\mathbf{x}) \log(1 - D(\mathbf{x})) \right) d\mathbf{x}$$

---

### 2.2 Proof of the Optimal Discriminator $D^*(\mathbf{x})$

**Theorem (Goodfellow et al., 2014):**
For any fixed generator $G$, the optimal discriminator $D^*$ that maximizes $V(D, G)$ is:
$$\mathbf{D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}}$$

**Proof:**
For any $\mathbf{x} \in \mathcal{X}$, the objective inside the integral is of the form:
$$f(y) = a \log y + b \log(1 - y)$$
where $y = D(\mathbf{x}) \in (0, 1)$, $a = p_{\text{data}}(\mathbf{x}) \ge 0$, and $b = p_g(\mathbf{x}) \ge 0$.
To find the critical point, take the derivative with respect to $y$:
$$f'(y) = \frac{a}{y} - \frac{b}{1 - y} = 0 \implies a(1 - y) = by \implies a = (a + b)y \implies y^* = \frac{a}{a + b}$$
Check the second derivative:
$$f''(y) = -\frac{a}{y^2} - \frac{b}{(1 - y)^2} < 0 \quad \forall y \in (0, 1)$$
Since $f''(y)$ is strictly negative everywhere, $y^*$ is the unique global maximum. Substituting $a = p_{\text{data}}(\mathbf{x})$ and $b = p_g(\mathbf{x})$ yields $D^*(\mathbf{x})$. $\blacksquare$

---

### 2.3 Equivalence to Jensen-Shannon Divergence

Substitute the optimal discriminator $D^*(\mathbf{x})$ back into the minimax objective:
$$\begin{aligned}
V(D^*, G) &= \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\left[ \log \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})} \right] + \mathbb{E}_{\mathbf{x} \sim p_g}\left[ \log \frac{p_g(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})} \right] \\
&= \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\left[ \log \left( \frac{1}{2} \frac{p_{\text{data}}(\mathbf{x})}{\frac{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}{2}} \right) \right] + \mathbb{E}_{\mathbf{x} \sim p_g}\left[ \log \left( \frac{1}{2} \frac{p_g(\mathbf{x})}{\frac{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}{2}} \right) \right] \\
&= -\log 2 - \log 2 + D_{\text{KL}}\left( p_{\text{data}} \,\|\, \frac{p_{\text{data}} + p_g}{2} \right) + D_{\text{KL}}\left( p_g \,\|\, \frac{p_{\text{data}} + p_g}{2} \right) \\
&= -\log 4 + 2 D_{\text{JS}}\left( p_{\text{data}} \,\|\, p_g \right)
\end{aligned}$$

where $D_{\text{JS}}$ is the Jensen-Shannon divergence:
$$D_{\text{JS}}(P \,\|\, Q) = \frac{1}{2} D_{\text{KL}}\left( P \,\|\, \frac{P+Q}{2} \right) + \frac{1}{2} D_{\text{KL}}\left( Q \,\|\, \frac{P+Q}{2} \right)$$

**Global Minimum:**
Because $D_{\text{JS}}(P \,\|\, Q) \ge 0$ with equality if and only if $P = Q$, the global minimum of the minimax game is achieved at:
$$p_g = p_{\text{data}}$$
At this point, $D^*(\mathbf{x}) = \frac{1}{2}$ everywhere, and the value of the game is $V(D^*, G^*) = -\log 4 \approx -1.3863$.

---

### 2.4 The Pathology of Jensen-Shannon on Manifolds: Why Vanilla GANs Fail

In deep learning, data $\mathbf{x} \in \mathbb{R}^D$ does not fill the entire high-dimensional space; it is concentrated on a low-dimensional manifold $\mathcal{M}$ of dimension $d \ll D$ (e.g., natural images lie on a thin manifold inside $\mathbb{R}^{3 \times 512 \times 512}$).
Similarly, the generated distribution $p_g$ has support of dimension at most $\operatorname{dim}(\mathbf{z})$.

**Arjovsky's Manifold Intersection Theorem:**
If the supports of $p_{\text{data}}$ and $p_g$ are submanifolds of $\mathbb{R}^D$ of lower dimension, then generically:
$$\operatorname{supp}(p_{\text{data}}) \cap \operatorname{supp}(p_g) = \emptyset \quad (\text{measure zero intersection})$$
When supports do not overlap:
$$D_{\text{JS}}\left( p_{\text{data}} \,\|\, p_g \right) = \log 2 \quad (\text{a constant!})$$
Because $D_{\text{JS}}$ is locally constant, its gradient with respect to generator parameters $\boldsymbol{\theta}$ is:
$$\nabla_{\boldsymbol{\theta}} D_{\text{JS}}(p_{\text{data}} \,\|\, p_{g_{\boldsymbol{\theta}}}) = \mathbf{0}$$

As soon as the discriminator learns to separate the two disjoint manifolds ($D(\mathbf{x}) = 1$ on real data, $D(G(\mathbf{z})) = 0$ on fake data), **the generator receives exactly zero gradient!**

```
               THE GEOMETRY OF DISJOINT MANIFOLDS
            Data Space X in R^2 (Manifold dimension = 1)
                     x_2
                      ▲
                      │       p_data: Line at x_1 = 0
                      │          │
                      │          │             p_g: Line at x_1 = theta
                      │          │                │
                      │          │                │
                      │          │                │
                      └──────────┴────────────────┴──────► x_1
                                 0              theta

           Distance between lines = |theta|
           JS Divergence D_JS(p_data || p_g) = log(2) = 0.6931 (CONSTANT!)
           Gradient d(D_JS)/d(theta) = 0  <-- NO LEARNING SIGNAL!
           Wasserstein W_1(p_data, p_g) = |theta| (LINEAR!)
           Gradient d(W_1)/d(theta) = sign(theta) <-- PERFECT CONSTANT GRADIENT!
```

---

### 2.5 The Non-Saturating Heuristic

To combat vanishing gradients early in training, Goodfellow proposed the **non-saturating generator loss**:
$$\min_G \mathcal{L}_G^{\text{NS}} = -\mathbb{E}_{\mathbf{z}} \left[ \log D(G(\mathbf{z})) \right]$$
While this provides non-zero gradients when $D(G(\mathbf{z})) \approx 0$, it optimizes a reverse KL divergence surrogate:
$$-\mathbb{E}_{\mathbf{z}} [\log D(G(\mathbf{z}))] \approx D_{\text{KL}}(p_g \,\|\, p_{\text{data}}) - 2 D_{\text{JS}}(p_g \,\|\, p_{\text{data}})$$
Recall from Chapter 10.1 that **Reverse KL is Zero-Forcing (Mode-Dropping)**: it heavily penalizes $G$ for generating samples in regions where $p_{\text{data}} \approx 0$, but assigns zero penalty if $G$ completely ignores major modes of the data! This is the fundamental mathematical cause of **mode collapse**.

---

### 2.6 Wasserstein Distance (Earth Mover's Distance)

Let $\Pi(p_r, p_g)$ be the set of all joint distributions (transport plans) $\gamma(\mathbf{x}, \mathbf{y})$ whose marginals are $p_r$ and $p_g$:
$$W_1(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma}\left[ \|\mathbf{x} - \mathbf{y}\| \right]$$

Intuitively, $W_1$ measures the minimum amount of "work" (mass $\times$ distance) required to morph distribution $p_g$ into distribution $p_r$.

**Key Property:**
Even if the supports of $p_r$ and $p_g$ do not overlap, $W_1$ is **continuous and differentiable almost everywhere**. For the two parallel lines at distance $\theta$:
$$W_1(p_{\text{data}}, p_g) = |\theta| \implies \frac{d W_1}{d\theta} = \operatorname{sgn}(\theta)$$
The gradient never vanishes, regardless of how far apart the distributions are!

---

### 2.7 Kantorovich-Rubinstein Duality

The infimum over all joint distributions $\Pi(p_r, p_g)$ is computationally intractable. By the **Kantorovich-Rubinstein Duality theorem**:
$$\mathbf{W_1(p_r, p_g) = \sup_{\|D\|_L \le 1} \mathbb{E}_{\mathbf{x} \sim p_r}[D(\mathbf{x})] - \mathbb{E}_{\tilde{\mathbf{x}} \sim p_g}[D(\tilde{\mathbf{x}})]}$$

where $\|D\|_L \le 1$ denotes the set of all **1-Lipschitz continuous functions**:
$$|D(\mathbf{x}_1) - D(\mathbf{x}_2)| \le \|\mathbf{x}_1 - \mathbf{x}_2\|_2 \quad \forall \mathbf{x}_1, \mathbf{x}_2$$

Under this formulation:
1. $D$ is called the **Critic** (not a discriminator, since it outputs unconstrained real scores, not probabilities).
2. The Critic is trained to assign high values to real data and low values to fake data, subject strictly to the 1-Lipschitz constraint.
3. The Generator objective becomes:
   $$\min_G \mathcal{L}_G = -\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}\left[ D(G(\mathbf{z})) \right]$$

---

### 2.8 Enforcing the 1-Lipschitz Constraint: WGAN-GP

#### 1. Weight Clipping (WGAN - Arjovsky et al., 2017)
The original WGAN enforced Lipschitz continuity by clamping the Critic's weights to a compact metric space after every gradient step:
$$w \leftarrow \operatorname{clip}(w, -c, c), \quad c \approx 0.01$$
**Fatal Flaws of Weight Clipping:**
- Critic capacity is severely underutilized: weights polarize to the boundaries $\pm c$, causing the critic to learn simple pathological step functions.
- Fragile sensitivity to threshold $c$: slightly too large leads to exploding gradients; slightly too small leads to vanishing gradients.

#### 2. Gradient Penalty (WGAN-GP - Gulrajani et al., 2017)
A differentiable function $D$ is 1-Lipschitz if and only if its gradient norm is bounded by 1 everywhere:
$$\|\nabla_{\mathbf{x}} D(\mathbf{x})\|_2 \le 1 \quad \forall \mathbf{x}$$
Moreover, along the optimal transport trajectory connecting real sample $\mathbf{x} \sim p_r$ and fake sample $\tilde{\mathbf{x}} \sim p_g$, the optimal critic satisfies:
$$\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 = 1$$
almost everywhere!

We sample points uniformly along straight lines between pairs of real and fake points:
$$\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) \tilde{\mathbf{x}}, \quad \epsilon \sim \mathcal{U}[0, 1]$$

The complete **WGAN-GP Critic Objective** is:
$$\mathbf{\mathcal{L}_{\text{critic}} = \underbrace{\mathbb{E}_{\tilde{\mathbf{x}} \sim p_g}[D(\tilde{\mathbf{x}})] - \mathbb{E}_{\mathbf{x} \sim p_r}[D(\mathbf{x})]}_{\text{Wasserstein-1 Distance Surrogate}} + \lambda \underbrace{\mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}} \left[ \left( \|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1 \right)^2 \right]}_{\text{Gradient Penalty (GP)}}}$$
Standard default penalty coefficient: $\lambda = 10$.

---

## 3. Geometric & Algebraic Interpretation

### Critic Vector Field as an Optimal Transport Flow
In WGAN-GP, the gradient of the critic $\nabla_{\mathbf{x}} D(\mathbf{x})$ defines a smooth vector field in data space:
1. Because $\|\nabla_{\mathbf{x}} D(\mathbf{x})\|_2 \approx 1$, the gradient vectors have unit length everywhere between the generated and data manifolds.
2. The vector field points directly along the geodesic from $\tilde{\mathbf{x}}$ toward the nearest data point $\mathbf{x}$.
3. When updating the generator:
   $$\nabla_{\boldsymbol{\theta}} \mathcal{L}_G = -\nabla_{\boldsymbol{\theta}} D(G(\mathbf{z})) = -\underbrace{\nabla_{\mathbf{x}} D(G(\mathbf{z}))}_{\text{Unit Direction Vector}} \cdot \nabla_{\boldsymbol{\theta}} G(\mathbf{z})$$
   The generator is gently guided along the shortest Euclidean path toward the real data manifold, with constant non-decaying force!

```
                  WGAN-GP GRADIENT VECTOR FIELD
              Data Space X (Unit vectors pointing from fake to real)
         x_2
          ▲
          │    Real Data Manifold (p_data)
          │    ┌───────────────────────────┐
          │    │  ●      ●       ●      ●  │  D(x) high (+5)
          │    └───────────────────────────┘
          │       ▲      ▲       ▲      ▲
          │       │      │       │      │  <-- ||grad D|| = 1
          │       │      │       │      │      Smooth vector field
          │       │      │       │      │
          │    ┌──┴──────┴───────┴──────┴──┐
          │    │  x      x       x      x  │  D(x) low (-5)
          │    └───────────────────────────┘
          │    Generated Manifold (p_g)
          └────────────────────────────────────────► x_1
```

---

## 4. Real-World Analogy

### The Pass/Fail Drill Sergeant vs. The Terrain Surveyor
- **Vanilla GAN (The Pass/Fail Drill Sergeant):**
  A student attempts high-jump hurdles. If the student touches the bar, the sergeant yells *"FAIL! ZERO MARKS!"* without telling them if they missed by 2 meters or by 1 millimeter. The student has zero gradient signal to improve.
- **WGAN-GP (The Topographical Surveyor):**
  The Critic does not give a binary grade. Instead, the Critic builds a smooth, gentle incline: a linear slope with a constant $45^\circ$ grade ($|\nabla D| = 1$) leading from the student's current position all the way to the top of the mountain. No matter where the student stands, they can feel the slope under their feet and know exactly which direction to climb!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us compute a complete WGAN-GP forward pass, gradient penalty computation, and backward gradient update with exact concrete toy numbers.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in WGAN-GP Architecture |
| :--- | :--- | :--- | :--- |
| $x_r$ | Real Data Point | Scalar | Ground truth real sample ($2.0$) |
| $z$ | Latent Noise Sample | Scalar | Uniform or standard normal noise input ($0.5$) |
| $G(z)$ | Generated Point | Scalar | Fake sample $w_g z + b_g$ produced by generator |
| $\epsilon$ | Interpolation Weight | Scalar $\in [0, 1]$ | Random straight-line mixing coefficient ($0.5$) |
| $\hat{x}$ | Interpolated Point | Scalar | Straight line sample $\epsilon x_r + (1 - \epsilon) x_f$ |
| $D(x)$ | Critic Output Score | Scalar | Unconstrained real score assigned by Critic |
| $\nabla_{\hat{x}} D(\hat{x})$ | Critic Gradient at $\hat{x}$ | Scalar | Spatial derivative of Critic at interpolated point |
| $\text{GP}$ | Gradient Penalty Term | Scalar | Penalty $(\|\nabla_{\hat{x}} D\| - 1)^2$ enforcing 1-Lipschitz |
| $\mathcal{L}_{\text{critic}}$ | Critic Objective Loss | Scalar | $D(x_f) - D(x_r) + \lambda \cdot \text{GP}$ |
| $\mathcal{L}_G$ | Generator Objective Loss | Scalar | $-D(x_f)$ |

---

### 5.2 Concrete Toy Setup

Let:
- Real data sample: $x_r = 2.0$
- Latent noise input: $z = 0.5$
- Generator parameters:
  $$w_g = 1.2000, \quad b_g = 0.2000$$
  $$x_f = G(z) = w_g z + b_g = 1.2000(0.5) + 0.2000 = 0.6000 + 0.2000 = \mathbf{0.8000}$$
- Interpolation weight: $\epsilon = 0.5$
  $$\hat{x} = \epsilon x_r + (1 - \epsilon) x_f = 0.5(2.0) + 0.5(0.8) = 1.0000 + 0.4000 = \mathbf{1.4000}$$
- Critic parameterized as a polynomial model:
  $$D(x) = w_c x^2 + v_c x$$
  with parameters:
  $$w_c = 0.5000, \quad v_c = 0.1000$$
- Gradient penalty coefficient: $\lambda = 10.0$

---

### 5.3 Step 1: Critic Forward Pass Evaluations

1. **Critic on Real Data ($x_r = 2.0$):**
   $$D(x_r) = 0.5(2.0)^2 + 0.1(2.0) = 0.5(4.0) + 0.2 = 2.0 + 0.2 = \mathbf{2.2000}$$

2. **Critic on Fake Data ($x_f = 0.8$):**
   $$D(x_f) = 0.5(0.8)^2 + 0.1(0.8) = 0.5(0.64) + 0.08 = 0.32 + 0.08 = \mathbf{0.4000}$$

3. **Wasserstein Distance Surrogate (Unpenalized Loss):**
   $$\mathcal{L}_{\text{wass}} = D(x_f) - D(x_r) = 0.4000 - 2.2000 = \mathbf{-1.8000}$$
   (The Critic successfully scores real data $2.2000 > 0.4000$ for fake data).

---

### 5.4 Step 2: Gradient Penalty Arithmetic at $\hat{x} = 1.4$

1. **Spatial derivative of Critic with respect to input $x$:**
   $$\nabla_x D(x) = \frac{d}{dx} \left( w_c x^2 + v_c x \right) = 2 w_c x + v_c$$

2. **Evaluate at interpolated point $\hat{x} = 1.4000$:**
   $$\nabla_{\hat{x}} D(\hat{x}) = 2(0.5000)(1.4000) + 0.1000 = 1.0000(1.4000) + 0.1000 = \mathbf{1.5000}$$

3. **Gradient Penalty Calculation:**
   $$\text{Gradient Norm} = |\nabla_{\hat{x}} D(\hat{x})| = 1.5000$$
   $$\text{Penalty} = (\|\nabla_{\hat{x}} D\| - 1)^2 = (1.5000 - 1.0000)^2 = 0.5000^2 = \mathbf{0.2500}$$

4. **Weighted Gradient Penalty ($\lambda = 10$):**
   $$\lambda \cdot \text{GP} = 10.0 \times 0.2500 = \mathbf{2.5000}$$

---

### 5.5 Step 3: Total Critic Loss & Gradients

$$\mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{wass}} + \lambda \cdot \text{GP} = -1.8000 + 2.5000 = \mathbf{0.7000}$$

Now, let us calculate the gradient with respect to Critic parameters $w_c$ and $v_c$:

#### Gradient of $\mathcal{L}_{\text{wass}}$:
$$\frac{\partial \mathcal{L}_{\text{wass}}}{\partial w_c} = x_f^2 - x_r^2 = 0.8^2 - 2.0^2 = 0.64 - 4.00 = \mathbf{-3.3600}$$
$$\frac{\partial \mathcal{L}_{\text{wass}}}{\partial v_c} = x_f - x_r = 0.8 - 2.0 = \mathbf{-1.2000}$$

#### Gradient of Penalty Term $\text{GP} = (\nabla_{\hat{x}} D - 1)^2$:
$$\frac{\partial \text{GP}}{\partial (\nabla_{\hat{x}} D)} = 2(\nabla_{\hat{x}} D - 1) = 2(1.5000 - 1.0000) = 2(0.5000) = 1.0000$$
Since $\nabla_{\hat{x}} D = 2 w_c \hat{x} + v_c$:
$$\frac{\partial (\nabla_{\hat{x}} D)}{\partial w_c} = 2 \hat{x} = 2(1.4000) = 2.8000$$
$$\frac{\partial (\nabla_{\hat{x}} D)}{\partial v_c} = 1.0000$$
Therefore:
$$\lambda \frac{\partial \text{GP}}{\partial w_c} = 10.0 \times (1.0000 \times 2.8000) = \mathbf{28.0000}$$
$$\lambda \frac{\partial \text{GP}}{\partial v_c} = 10.0 \times (1.0000 \times 1.0000) = \mathbf{10.0000}$$

#### Total Critic Parameter Gradients:
$$\frac{\partial \mathcal{L}_{\text{critic}}}{\partial w_c} = -3.3600 + 28.0000 = \mathbf{24.6400}$$
$$\frac{\partial \mathcal{L}_{\text{critic}}}{\partial v_c} = -1.2000 + 10.0000 = \mathbf{8.8000}$$

---

### 5.6 Step 4: Generator Loss & Update Arithmetic

The generator wants to maximize the critic score of fake data:
$$\mathcal{L}_G = -D(x_f) = -(w_c x_f^2 + v_c x_f) = -0.4000$$

Gradient with respect to generator weight $w_g$:
$$\frac{\partial \mathcal{L}_G}{\partial w_g} = \frac{\partial \mathcal{L}_G}{\partial x_f} \cdot \frac{\partial x_f}{\partial w_g}$$
1. $\frac{\partial \mathcal{L}_G}{\partial x_f} = -(2 w_c x_f + v_c) = -(2(0.5000)(0.8000) + 0.1000) = -(0.8000 + 0.1000) = \mathbf{-0.9000}$
2. $\frac{\partial x_f}{\partial w_g} = z = \mathbf{0.5000}$
$$\frac{\partial \mathcal{L}_G}{\partial w_g} = (-0.9000)(0.5000) = \mathbf{-0.4500}$$

Notice the sign: $\frac{\partial \mathcal{L}_G}{\partial w_g} = -0.4500 < 0$.
Under gradient descent ($w_g \leftarrow w_g - \eta \frac{\partial \mathcal{L}_G}{\partial w_g}$):
$w_g$ will **increase**, pushing $x_f = w_g z + b_g$ upward from $0.8$ toward real target $2.0$!
Every single number matches autograd to numerical precision.

---

## 6. Solved Illustrations

### Illustration 1: The Parallel Lines Problem
**Problem:** Let $p_r$ be uniform on line $x = 0, y \in [0, 1]$. Let $p_g$ be uniform on line $x = \theta, y \in [0, 1]$.
Compute $D_{\text{KL}}(p_r \| p_g)$, $D_{\text{JS}}(p_r \| p_g)$, and $W_1(p_r, p_g)$ for $\theta \ne 0$.
**Solution:**
1. **KL Divergence:**
   Since $p_g(0, y) = 0$ wherever $p_r(0, y) = 1$:
   $$D_{\text{KL}}(p_r \,\|\, p_g) = \int p_r \log \frac{p_r}{p_g} = \mathbf{+\infty}$$
2. **JS Divergence:**
   The mixture $M = \frac{p_r + p_g}{2}$ equals $\frac{1}{2}$ on both lines:
   $$D_{\text{JS}}(p_r \,\|\, p_g) = \frac{1}{2} \int_0^1 1 \log \frac{1}{0.5} dy + \frac{1}{2} \int_0^1 1 \log \frac{1}{0.5} dy = \log 2 \approx \mathbf{0.6931}$$
   For all $\theta \ne 0$, $D_{\text{JS}} = \log 2$. The derivative $\frac{d D_{\text{JS}}}{d\theta} = \mathbf{0}$.
3. **Wasserstein Distance:**
   Every point $(0, y)$ must be transported by horizontal distance $|\theta|$ to $(\theta, y)$:
   $$W_1(p_r, p_g) = \mathbf{|\theta|}$$
   Derivative $\frac{d W_1}{d\theta} = \operatorname{sgn}(\theta) = \pm 1 \ne 0$. The gradient is always informative and constant!

---

### Illustration 2: Why WGAN Critic Requires Multiple Updates Per Generator Step
**Question:** Why do we train the WGAN critic for $n_{\text{critic}} = 5$ steps for every 1 generator step?
**Answer:**
In WGAN, the Kantorovich-Rubinstein duality holds if and only if $D$ is the supremum over all 1-Lipschitz functions:
$$\mathcal{L}_G \approx -W_1(p_r, p_g) \iff D \approx D^*$$
If the Critic is undertrained, $-D(G(\mathbf{z}))$ does not approximate the true Wasserstein distance, and the generator optimizes an erroneous, distorted transport plan. Keeping the Critic near optimality ensures the gradient field $\nabla_x D(x)$ points reliably in the true optimal transport direction.

---

## 7. Deep Learning Connection & Application

### State-of-the-Art Production Architectures
1. **StyleGAN2 & StyleGAN3 (Karras et al., NVIDIA):**
   - Employs non-saturating loss with $R_1$ zero-centered gradient penalty:
     $$R_1 = \frac{\gamma}{2} \mathbb{E}_{\mathbf{x} \sim p_r} [\|\nabla_{\mathbf{x}} D(\mathbf{x})\|^2]$$
   - Uses style-based generator where mapping network $f$ maps $\mathbf{z} \to \mathbf{w}$, modulating convolutional weights via Weight Modulation/Demodulation.
2. **Pix2Pix & CycleGAN (Image-to-Image Translation):**
   - Paired translation (Pix2Pix) and unpaired translation (CycleGAN with cycle-consistency loss $\|\mathbf{x} - G(F(\mathbf{x}))\|$).
   - Uses **PatchGAN Discriminator**, penalizing structure at the scale of local $70 \times 70$ patches.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Exact numerical verification of Part 5 hand arithmetic (Critic forward, GP, total loss, and parameter gradients matching autograd to $< 10^{-6}$).
2. Analytical proof of Parallel Lines divergence comparison ($D_{\text{JS}}$ vanishing vs. $W_1$ constant slope).
3. Complete PyTorch implementation of WGAN-GP training loop with Critic gradient penalty computation.
4. Mode collapse demonstration comparing Vanilla GAN vs. WGAN-GP on an 8-Gaussian ring distribution.

See implementation in:
[`10_generative_models/code/03_generative_adversarial_networks_and_wgan.py`](./code/03_generative_adversarial_networks_and_wgan.py)
