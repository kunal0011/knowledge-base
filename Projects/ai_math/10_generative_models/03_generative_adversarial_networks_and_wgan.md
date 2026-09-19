# Generative Adversarial Networks (GANs) & Wasserstein GAN

---

## 1. Intuition & 101 Motivation

### The Game-Theoretic Paradigm Shift
Prior to 2014, generative modeling was dominated by explicit density estimation: maximizing $\log p_{\theta}(\mathbf{x})$ directly or via variational lower bounds ($\text{ELBO}$).
In 2014, Ian Goodfellow introduced **Generative Adversarial Networks (GANs)**, abandoning explicit likelihoods in favor of a two-player zero-sum game:
1. **The Generator ($G_{\theta}$):** Takes standard normal noise $\mathbf{z} \sim p_{\mathbf{z}}$ and maps it to the data space $\mathbf{x}_{\text{fake}} = G_{\theta}(\mathbf{z})$. Its goal is to fool the discriminator.
2. **The Discriminator ($D_{\phi}$):** Takes a sample $\mathbf{x}$ (either real from $p_{\text{data}}$ or fake from $p_g$) and outputs a scalar probability $D_{\phi}(\mathbf{x}) \in [0, 1]$ indicating whether $\mathbf{x}$ is a genuine data point.

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
Because $D_{\text{JS}}$ is locally constant, its gradient with respect to generator parameters $\theta$ is:
$$\nabla_{\theta} D_{\text{JS}}(p_{\text{data}} \,\|\, p_{g_{\theta}}) = \mathbf{0}$$

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

### 2.9 Rigorous Mathematical Derivations

#### Derivation 10.3.1: Kantorovich-Rubinstein Duality Theorem via Fenchel-Rockafellar Duality

**1. Context and Assumptions:**
Let $(\mathcal{X}, d)$ be a complete separable metric space (Polish space) equipped with metric $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$.
Let $\mathcal{P}_1(\mathcal{X})$ denote the set of probability distributions on $\mathcal{X}$ with finite first moment: $\int_{\mathcal{X}} \|\mathbf{x}\| dP(\mathbf{x}) < \infty$.
Let $P, Q \in \mathcal{P}_1(\mathcal{X})$.
Let $\Pi(P, Q)$ be the collection of all joint probability measures $\pi$ on $\mathcal{X} \times \mathcal{X}$ having marginal distributions $P$ and $Q$, satisfying:
$$\int_{\mathcal{X}} d\pi(\mathbf{x}, \mathbf{y}) = Q(\mathbf{y}), \qquad \int_{\mathcal{X}} d\pi(\mathbf{x}, \mathbf{y}) = P(\mathbf{x})$$

**2. Step 1: Primal Optimal Transport Formulation:**
The Kantorovich formulation of the optimal transport problem defines the Wasserstein-1 distance as:
$$W_1(P, Q) \triangleq \inf_{\pi \in \Pi(P, Q)} \int_{\mathcal{X} \times \mathcal{X}} \|\mathbf{x} - \mathbf{y}\|_2 \, d\pi(\mathbf{x}, \mathbf{y})$$
This is an infinite-dimensional linear programming problem: minimizing a linear functional of $\pi$ subject to linear marginal constraints and non-negativity $\pi \ge 0$.

**3. Step 2: Lagrangian Formulation with Multiplier Potentials:**
We relax the marginal constraints using Lagrange multiplier functions $\phi, \psi \in C_b(\mathcal{X})$ (bounded continuous real functions):
$$\mathcal{L}(\pi, \phi, \psi) = \int_{\mathcal{X} \times \mathcal{X}} \|\mathbf{x} - \mathbf{y}\|_2 \, d\pi(\mathbf{x}, \mathbf{y}) + \int_{\mathcal{X}} \phi(\mathbf{x}) \left( dP(\mathbf{x}) - \int_{\mathcal{X}} d\pi(\mathbf{x}, \mathbf{y}) \right) + \int_{\mathcal{X}} \psi(\mathbf{y}) \left( dQ(\mathbf{y}) - \int_{\mathcal{X}} d\pi(\mathbf{x}, \mathbf{y}) \right)$$
Rearranging terms by grouping measures:
$$\mathcal{L}(\pi, \phi, \psi) = \int_{\mathcal{X}} \phi(\mathbf{x}) dP(\mathbf{x}) + \int_{\mathcal{X}} \psi(\mathbf{y}) dQ(\mathbf{y}) + \int_{\mathcal{X} \times \mathcal{X}} \left( \|\mathbf{x} - \mathbf{y}\|_2 - \phi(\mathbf{x}) - \psi(\mathbf{y}) \right) d\pi(\mathbf{x}, \mathbf{y})$$
Taking the infimum over all positive measures $\pi \ge 0$:
$$\inf_{\pi \ge 0} \int_{\mathcal{X} \times \mathcal{X}} \left( \|\mathbf{x} - \mathbf{y}\|_2 - \phi(\mathbf{x}) - \psi(\mathbf{y}) \right) d\pi(\mathbf{x}, \mathbf{y}) = \begin{cases} 0 & \text{if } \phi(\mathbf{x}) + \psi(\mathbf{y}) \le \|\mathbf{x} - \mathbf{y}\|_2 \quad \forall \mathbf{x}, \mathbf{y} \\ -\infty & \text{otherwise} \end{cases}$$
By strong linear programming duality (Fenchel-Rockafellar theorem, since the feasible set has non-empty interior):
$$W_1(P, Q) = \sup_{\substack{\phi, \psi \in C_b(\mathcal{X}) \\ \phi(\mathbf{x}) + \psi(\mathbf{y}) \le \|\mathbf{x} - \mathbf{y}\|_2}} \left( \int_{\mathcal{X}} \phi(\mathbf{x}) dP(\mathbf{x}) + \int_{\mathcal{X}} \psi(\mathbf{y}) dQ(\mathbf{y}) \right)$$

**4. Step 3: $c$-Transform and Reduction to 1-Lipschitz Potentials:**
For any fixed $\phi(\mathbf{x})$, to maximize the objective with respect to $\psi(\mathbf{y})$ under the constraint $\psi(\mathbf{y}) \le \|\mathbf{x} - \mathbf{y}\|_2 - \phi(\mathbf{x})$, we must set $\psi(\mathbf{y})$ to its point-wise infimum over all $\mathbf{x}$:
$$\psi(\mathbf{y}) = \inf_{\mathbf{x} \in \mathcal{X}} \left( \|\mathbf{x} - \mathbf{y}\|_2 - \phi(\mathbf{x}) \right) \triangleq -\phi^c(\mathbf{y})$$
Now substitute $\mathbf{x} = \mathbf{y}$:
$$\psi(\mathbf{x}) \le \|\mathbf{x} - \mathbf{x}\|_2 - \phi(\mathbf{x}) = -\phi(\mathbf{x}) \implies \psi(\mathbf{x}) + \phi(\mathbf{x}) \le 0$$
Furthermore, if $\phi$ is optimal, setting $\psi(\mathbf{y}) = -\phi(\mathbf{y})$ yields:
$$\phi(\mathbf{x}) - \phi(\mathbf{y}) \le \|\mathbf{x} - \mathbf{y}\|_2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathcal{X}$$
Swapping $\mathbf{x}$ and $\mathbf{y}$:
$$\phi(\mathbf{y}) - \phi(\mathbf{x}) \le \|\mathbf{y} - \mathbf{x}\|_2 = \|\mathbf{x} - \mathbf{y}\|_2 \implies |\phi(\mathbf{x}) - \phi(\mathbf{y})| \le \|\mathbf{x} - \mathbf{y}\|_2$$
This is the exact mathematical definition of a **1-Lipschitz function** $\|\phi\|_L \le 1$!
Renaming $\phi \triangleq D$ (the Critic):
$$\mathbf{W_1(P, Q) = \sup_{\|D\|_L \le 1} \left( \mathbb{E}_{\mathbf{x} \sim P}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim Q}[D(\mathbf{y})] \right)}$$

---

#### Derivation 10.3.2: Unit Gradient Norm Property of the Optimal Critic along Optimal Transport Geodesics

**1. Context and Assumptions:**
Let $\pi^*$ be an optimal transport plan achieving $W_1(P, Q)$.
Let $(\mathbf{x}, \mathbf{y}) \in \operatorname{supp}(\pi^*)$, where $\mathbf{x} \sim P$ (real) and $\mathbf{y} \sim Q$ (generated).
Let $D^*$ be the optimal 1-Lipschitz Critic function achieving the supremum in the Kantorovich-Rubinstein duality.
Define the line segment (geodesic) connecting $\mathbf{y}$ to $\mathbf{x}$:
$$\hat{\mathbf{x}}(t) = t \mathbf{x} + (1 - t) \mathbf{y} = \mathbf{y} + t(\mathbf{x} - \mathbf{y}), \quad t \in [0, 1]$$

**2. Step 1: Exact Potential Difference on Support:**
From the complementary slackness conditions of optimal transport duality:
$$D^*(\mathbf{x}) - D^*(\mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$$
That is, the optimal critic achieves the maximal possible difference permitted by 1-Lipschitz continuity along any coupled pair $(\mathbf{x}, \mathbf{y}) \sim \pi^*$.

**3. Step 2: Bounding the Potential along the Interpolation Geodesic:**
For any intermediate point $\hat{\mathbf{x}}(t)$ with $t \in (0, 1)$:
1. By the 1-Lipschitz condition between $\hat{\mathbf{x}}(t)$ and $\mathbf{y}$:
   $$D^*(\hat{\mathbf{x}}(t)) - D^*(\mathbf{y}) \le \|\hat{\mathbf{x}}(t) - \mathbf{y}\|_2 = \|t(\mathbf{x} - \mathbf{y})\|_2 = t \|\mathbf{x} - \mathbf{y}\|_2$$
2. By the 1-Lipschitz condition between $\mathbf{x}$ and $\hat{\mathbf{x}}(t)$:
   $$D^*(\mathbf{x}) - D^*(\hat{\mathbf{x}}(t)) \le \|\mathbf{x} - \hat{\mathbf{x}}(t)\|_2 = \|(1 - t)(\mathbf{x} - \mathbf{y})\|_2 = (1 - t) \|\mathbf{x} - \mathbf{y}\|_2$$
Now sum the two inequalities:
$$\left( D^*(\hat{\mathbf{x}}(t)) - D^*(\mathbf{y}) \right) + \left( D^*(\mathbf{x}) - D^*(\hat{\mathbf{x}}(t)) \right) \le t \|\mathbf{x} - \mathbf{y}\|_2 + (1 - t) \|\mathbf{x} - \mathbf{y}\|_2 = \|\mathbf{x} - \mathbf{y}\|_2$$
Notice that the left-hand side telescopes to:
$$D^*(\mathbf{x}) - D^*(\mathbf{y}) \le \|\mathbf{x} - \mathbf{y}\|_2$$
However, from Step 1, $D^*(\mathbf{x}) - D^*(\mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$ exactly.
Because the sum of two upper bounds equals their exact sum, **both individual inequalities must hold with strict equality!**
$$D^*(\hat{\mathbf{x}}(t)) - D^*(\mathbf{y}) = t \|\mathbf{x} - \mathbf{y}\|_2$$
$$D^*(\hat{\mathbf{x}}(t)) = D^*(\mathbf{y}) + t \|\mathbf{x} - \mathbf{y}\|_2$$
The optimal critic $D^*$ is **strictly linear** along the straight line connecting $\mathbf{y}$ to $\mathbf{x}$.

**4. Step 3: Gradient Evaluation:**
Differentiating $D^*(\hat{\mathbf{x}}(t))$ with respect to $t$ using the multivariable chain rule:
$$\frac{d}{dt} D^*(\hat{\mathbf{x}}(t)) = \nabla D^*(\hat{\mathbf{x}}(t))^\top \frac{d \hat{\mathbf{x}}(t)}{dt} = \nabla D^*(\hat{\mathbf{x}}(t))^\top (\mathbf{x} - \mathbf{y})$$
From the explicit linear solution $D^*(\hat{\mathbf{x}}(t)) = D^*(\mathbf{y}) + t \|\mathbf{x} - \mathbf{y}\|_2$:
$$\frac{d}{dt} D^*(\hat{\mathbf{x}}(t)) = \|\mathbf{x} - \mathbf{y}\|_2$$
Equating the two expressions:
$$\nabla D^*(\hat{\mathbf{x}}(t))^\top \frac{\mathbf{x} - \mathbf{y}}{\|\mathbf{x} - \mathbf{y}\|_2} = 1$$
Let $\mathbf{v} = \frac{\mathbf{x} - \mathbf{y}}{\|\mathbf{x} - \mathbf{y}\|_2}$ be the unit vector pointing from fake to real.
By Cauchy-Schwarz:
$$1 = \nabla D^*(\hat{\mathbf{x}}(t))^\top \mathbf{v} \le \|\nabla D^*(\hat{\mathbf{x}}(t))\|_2 \|\mathbf{v}\|_2 = \|\nabla D^*(\hat{\mathbf{x}}(t))\|_2 (1) = \|\nabla D^*(\hat{\mathbf{x}}(t))\|_2$$
Thus:
$$\|\nabla D^*(\hat{\mathbf{x}}(t))\|_2 \ge 1$$
Combined with the global 1-Lipschitz condition $\|\nabla D^*(\hat{\mathbf{x}})\|_2 \le 1$:
$$\mathbf{\|\nabla D^*(\hat{\mathbf{x}}(t))\|_2 = 1, \qquad \text{and } \nabla D^*(\hat{\mathbf{x}}(t)) = \frac{\mathbf{x} - \mathbf{y}}{\|\mathbf{x} - \mathbf{y}\|_2}}$$
almost everywhere along the straight line.
This proves that the optimal gradient norm is **identically 1**, rigorously justifying Gulrajani et al.'s two-sided gradient penalty $(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2$.

---

#### Derivation 10.3.3: Spectral Normalization and Power Iteration for Exact Matrix Lipschitz Bounds

**1. Context and Assumptions:**
Consider a deep neural network critic $f: \mathbb{R}^D \to \mathbb{R}$ defined as a composition of $L$ layers:
$$f(\mathbf{x}) = \mathbf{W}_{L+1} \, a_L\left( \mathbf{W}_L \dots a_1\left( \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 \right) \dots + \mathbf{b}_L \right) + b_{L+1}$$
where $\mathbf{W}_l$ are weight matrices and $a_l(\cdot)$ are non-linear activation functions (e.g. ReLU, LeakyReLU with negative slope $|\alpha| \le 1$).

**2. Step 1: Layer-Wise Lipschitz Constant Composition:**
The Lipschitz constant of a composite function $f = g \circ h$ satisfies $\|f\|_L \le \|g\|_L \cdot \|h\|_L$.
Therefore:
$$\|f\|_L \le \|\mathbf{W}_{L+1}\|_2 \cdot \prod_{l=1}^L \left( \|a_l\|_L \cdot \|\mathbf{W}_l\|_2 \right)$$
For standard activations (ReLU, LeakyReLU, ELU, Sigmoid), the maximum derivative $|a'_l(u)| \le 1$, so $\|a_l\|_L = 1$.
The matrix operator norm $\|\mathbf{W}\|_2$ induced by the Euclidean norm is the **spectral norm** $\sigma_{\max}(\mathbf{W})$ (the largest singular value):
$$\|\mathbf{W}\|_2 \triangleq \sup_{\mathbf{h} \ne \mathbf{0}} \frac{\|\mathbf{W}\mathbf{h}\|_2}{\|\mathbf{h}\|_2} = \sigma_{\max}(\mathbf{W}) = \sqrt{\lambda_{\max}(\mathbf{W}^\top \mathbf{W})}$$
Thus:
$$\|f\|_L \le \prod_{l=1}^{L+1} \sigma_{\max}(\mathbf{W}_l)$$
To guarantee that the entire network is 1-Lipschitz ($\|f\|_L \le 1$), a sufficient condition is that every individual layer has spectral norm bounded by 1:
$$\sigma_{\max}(\mathbf{W}_l) \le 1 \quad \forall l \in \{1, \dots, L+1\}$$

**3. Step 2: The Spectral Normalization Operator:**
Miyato et al. (2018) normalize each weight matrix by its spectral norm:
$$\mathbf{W}_{\text{SN}} \triangleq \frac{\mathbf{W}}{\sigma_{\max}(\mathbf{W})}$$
Evaluating the spectral norm of $\mathbf{W}_{\text{SN}}$:
$$\sigma_{\max}(\mathbf{W}_{\text{SN}}) = \sigma_{\max}\left( \frac{\mathbf{W}}{\sigma_{\max}(\mathbf{W})} \right) = \frac{\sigma_{\max}(\mathbf{W})}{\sigma_{\max}(\mathbf{W})} = 1.0$$
This guarantees $\|f\|_L \le 1$ unconditionally without adding penalty terms to the loss function!

**4. Step 3: Power Iteration for Fast Singular Value Approximation:**
Computing a full SVD of $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ requires cubic $\mathcal{O}(d^3)$ operations.
Instead, Power Iteration maintains persistent unit vectors $\mathbf{u} \in \mathbb{R}^{d_{\text{out}}}$ and $\mathbf{v} \in \mathbb{R}^{d_{\text{in}}}$.
At each training step, perform one step of power iteration:
$$\mathbf{v} \leftarrow \frac{\mathbf{W}^\top \mathbf{u}}{\|\mathbf{W}^\top \mathbf{u}\|_2}, \qquad \mathbf{u} \leftarrow \frac{\mathbf{W}\mathbf{v}}{\|\mathbf{W}\mathbf{v}\|_2}$$
The largest singular value is estimated by:
$$\sigma(\mathbf{W}) \approx \mathbf{u}^\top \mathbf{W} \mathbf{v}$$
Because $\mathbf{W}$ changes slowly over gradient steps, a single iteration per training step converges extremely rapidly.

**5. Step 4: Analytical Gradient of Spectrally Normalized Weights:**
Let $\mathcal{L}$ be the objective loss. By matrix differential calculus:
$$d\mathbf{W}_{\text{SN}} = d\left( \frac{\mathbf{W}}{\sigma(\mathbf{W})} \right) = \frac{d\mathbf{W}}{\sigma(\mathbf{W})} - \frac{\mathbf{W}}{\sigma(\mathbf{W})^2} d\sigma(\mathbf{W})$$
From the variational characterization $\sigma(\mathbf{W}) = \mathbf{u}^\top \mathbf{W} \mathbf{v}$:
$$d\sigma(\mathbf{W}) = \mathbf{u}^\top (d\mathbf{W}) \mathbf{v} = \operatorname{tr}\left( \mathbf{v}\mathbf{u}^\top d\mathbf{W} \right)$$
Using $d\mathcal{L} = \operatorname{tr}\left( (\nabla_{\mathbf{W}_{\text{SN}}} \mathcal{L})^\top d\mathbf{W}_{\text{SN}} \right)$:
$$d\mathcal{L} = \frac{1}{\sigma(\mathbf{W})} \operatorname{tr}\left( (\nabla_{\mathbf{W}_{\text{SN}}} \mathcal{L})^\top d\mathbf{W} \right) - \frac{\operatorname{tr}\left( (\nabla_{\mathbf{W}_{\text{SN}}} \mathcal{L})^\top \mathbf{W} \right)}{\sigma(\mathbf{W})^2} \operatorname{tr}\left( \mathbf{v}\mathbf{u}^\top d\mathbf{W} \right)$$
Recognizing that $\frac{\mathbf{W}}{\sigma(\mathbf{W})} = \mathbf{W}_{\text{SN}}$:
$$\nabla_{\mathbf{W}} \mathcal{L} = \frac{1}{\sigma(\mathbf{W})} \left[ \nabla_{\mathbf{W}_{\text{SN}}} \mathcal{L} - \left( \nabla_{\mathbf{W}_{\text{SN}}} \mathcal{L} : \mathbf{W}_{\text{SN}} \right) \mathbf{u}\mathbf{v}^\top \right]$$
where $\mathbf{A} : \mathbf{B} \triangleq \operatorname{tr}(\mathbf{A}^\top \mathbf{B}) = \sum_{i, j} A_{i, j} B_{i, j}$.
Notice the geometric structure: the second term subtracts the projection of the gradient along $\mathbf{u}\mathbf{v}^\top$, preventing weight explosion while preserving directionality.

---

## 3. Geometric & Algebraic Interpretation

### Critic Vector Field as an Optimal Transport Flow
In WGAN-GP, the gradient of the critic $\nabla_{\mathbf{x}} D(\mathbf{x})$ defines a smooth vector field in data space:
1. Because $\|\nabla_{\mathbf{x}} D(\mathbf{x})\|_2 \approx 1$, the gradient vectors have unit length everywhere between the generated and data manifolds.
2. The vector field points directly along the geodesic from $\tilde{\mathbf{x}}$ toward the nearest data point $\mathbf{x}$.
3. When updating the generator:
   $$\nabla_{\theta} \mathcal{L}_G = -\nabla_{\theta} D(G(\mathbf{z})) = -\underbrace{\nabla_{\mathbf{x}} D(G(\mathbf{z}))}_{\text{Unit Direction Vector}} \cdot \nabla_{\theta} G(\mathbf{z})$$
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

### Illustration 3: Complete Step-by-Step Numerical WGAN-GP Critic and Generator Update in 2D Space

**Problem:**
Consider a WGAN-GP model operating on a 2D continuous space ($D=2$).
Observed real sample:
$$\mathbf{x}_r = \begin{bmatrix} 2.00 \\ 4.00 \end{bmatrix}$$
Fake sample produced by generator:
$$\mathbf{x}_f = \begin{bmatrix} 1.00 \\ 1.00 \end{bmatrix}$$
The Critic network is parameterized by a quadratic function:
$$D(\mathbf{x}) = \mathbf{w}_c^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \mathbf{A} \mathbf{x}$$
with parameter values:
$$\mathbf{w}_c = \begin{bmatrix} 0.50 \\ 0.20 \end{bmatrix}, \qquad \mathbf{A} = \begin{bmatrix} 0.10 & 0.00 \\ 0.00 & 0.10 \end{bmatrix}$$
Interpolation random draw: $\epsilon = 0.60$.
Gradient penalty coefficient: $\lambda = 10.0$.
1. Compute the interpolated point $\hat{\mathbf{x}} = \epsilon \mathbf{x}_r + (1 - \epsilon) \mathbf{x}_f$.
2. Compute the critic scores $D(\mathbf{x}_r)$ and $D(\mathbf{x}_f)$, and the Wasserstein surrogate loss $\mathcal{L}_{\text{EMD}} = D(\mathbf{x}_f) - D(\mathbf{x}_r)$.
3. Compute the spatial gradient of the critic $\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})$ at the interpolated point.
4. Compute the gradient norm $\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2$ and the exact Gradient Penalty $\text{GP} = \left( \|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1 \right)^2$.
5. Compute the total Critic objective loss $\mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{EMD}} + \lambda \cdot \text{GP}$.
6. Compute the analytical parameter gradient $\frac{\partial \mathcal{L}_{\text{critic}}}{\partial \mathbf{w}_c}$.

**Step-by-Step Solution:**

**1. Interpolated Point $\hat{\mathbf{x}}$:**
$$\hat{\mathbf{x}} = 0.60 \begin{bmatrix} 2.00 \\ 4.00 \end{bmatrix} + (1 - 0.60) \begin{bmatrix} 1.00 \\ 1.00 \end{bmatrix} = \begin{bmatrix} 1.20 \\ 2.40 \end{bmatrix} + \begin{bmatrix} 0.40 \\ 0.40 \end{bmatrix} = \begin{bmatrix} \mathbf{1.60} \\ \mathbf{2.80} \end{bmatrix}$$

**2. Critic Forward Scores & Wasserstein Loss:**
- **Real Score $D(\mathbf{x}_r)$:**
  $$\mathbf{w}_c^\top \mathbf{x}_r = 0.50(2.00) + 0.20(4.00) = 1.00 + 0.80 = 1.8000$$
  $$\frac{1}{2} \mathbf{x}_r^\top \mathbf{A} \mathbf{x}_r = \frac{1}{2} \left( 0.10(2.00)^2 + 0.10(4.00)^2 \right) = \frac{1}{2} (0.40 + 1.60) = \frac{1}{2}(2.00) = 1.0000$$
  $$D(\mathbf{x}_r) = 1.8000 + 1.0000 = \mathbf{2.8000}$$
- **Fake Score $D(\mathbf{x}_f)$:**
  $$\mathbf{w}_c^\top \mathbf{x}_f = 0.50(1.00) + 0.20(1.00) = 0.7000$$
  $$\frac{1}{2} \mathbf{x}_f^\top \mathbf{A} \mathbf{x}_f = \frac{1}{2} \left( 0.10(1.00)^2 + 0.10(1.00)^2 \right) = \frac{1}{2} (0.10 + 0.10) = 0.1000$$
  $$D(\mathbf{x}_f) = 0.7000 + 0.1000 = \mathbf{0.8000}$$
- **Wasserstein Surrogate Loss:**
  $$\mathcal{L}_{\text{EMD}} = D(\mathbf{x}_f) - D(\mathbf{x}_r) = 0.8000 - 2.8000 = \mathbf{-2.0000}$$

**3. Spatial Gradient of Critic $\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})$:**
Differentiating $D(\mathbf{x})$ with respect to spatial coordinate $\mathbf{x}$:
$$\nabla_{\mathbf{x}} D(\mathbf{x}) = \mathbf{w}_c + \mathbf{A} \mathbf{x}$$
At $\hat{\mathbf{x}} = [1.60, 2.80]^\top$:
$$\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}}) = \begin{bmatrix} 0.50 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0.10 & 0.00 \\ 0.00 & 0.10 \end{bmatrix} \begin{bmatrix} 1.60 \\ 2.80 \end{bmatrix} = \begin{bmatrix} 0.50 + 0.16 \\ 0.20 + 0.28 \end{bmatrix} = \begin{bmatrix} \mathbf{0.6600} \\ \mathbf{0.4800} \end{bmatrix}$$

**4. Gradient Norm and Gradient Penalty:**
$$\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2^2 = (0.6600)^2 + (0.4800)^2 = 0.4356 + 0.2304 = 0.666000$$
$$\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 = \sqrt{0.666000} \approx \mathbf{0.816088}$$
Gradient penalty:
$$\text{GP} = \left( \|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1 \right)^2 = (0.816088 - 1.000000)^2 = (-0.183912)^2 = \mathbf{0.033824}$$

**5. Total Critic Loss:**
$$\mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{EMD}} + \lambda \cdot \text{GP} = -2.000000 + 10.0 \times 0.033824 = -2.000000 + 0.338236 = \mathbf{-1.661764}$$

**6. Parameter Gradient with Respect to $\mathbf{w}_c$:**
$$\frac{\partial \mathcal{L}_{\text{critic}}}{\partial \mathbf{w}_c} = \frac{\partial \mathcal{L}_{\text{EMD}}}{\partial \mathbf{w}_c} + \lambda \frac{\partial \text{GP}}{\partial \mathbf{w}_c}$$
1. EMD derivative:
   $$\frac{\partial \mathcal{L}_{\text{EMD}}}{\partial \mathbf{w}_c} = \mathbf{x}_f - \mathbf{x}_r = \begin{bmatrix} 1.00 - 2.00 \\ 1.00 - 4.00 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.0000} \\ \mathbf{-3.0000} \end{bmatrix}$$
2. GP derivative via chain rule:
   Let $\mathbf{g} = \nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}}) \implies \frac{\partial \mathbf{g}}{\partial \mathbf{w}_c} = \mathbf{I}_2$.
   $$\frac{\partial \text{GP}}{\partial \mathbf{w}_c} = 2 \left( \|\mathbf{g}\|_2 - 1 \right) \frac{\partial \|\mathbf{g}\|_2}{\partial \mathbf{g}} \frac{\partial \mathbf{g}}{\partial \mathbf{w}_c} = 2 \left( \|\mathbf{g}\|_2 - 1 \right) \frac{\mathbf{g}}{\|\mathbf{g}\|_2} \mathbf{I}$$
   $$\frac{\partial \text{GP}}{\partial \mathbf{w}_c} = 2(0.816088 - 1.0) \frac{1}{0.816088} \begin{bmatrix} 0.6600 \\ 0.4800 \end{bmatrix} = 2(-0.183912)(1.225358) \begin{bmatrix} 0.6600 \\ 0.4800 \end{bmatrix}$$
   $$= 2(-0.225358) \begin{bmatrix} 0.6600 \\ 0.4800 \end{bmatrix} = -0.450716 \begin{bmatrix} 0.6600 \\ 0.4800 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.297473} \\ \mathbf{-0.216344} \end{bmatrix}$$
3. Combining terms with $\lambda = 10.0$:
   $$\frac{\partial \mathcal{L}_{\text{critic}}}{\partial \mathbf{w}_c} = \begin{bmatrix} -1.000000 \\ -3.000000 \end{bmatrix} + 10.0 \begin{bmatrix} -0.297473 \\ -0.216344 \end{bmatrix} = \begin{bmatrix} -1.000000 - 2.974730 \\ -3.000000 - 2.163440 \end{bmatrix} = \begin{bmatrix} \mathbf{-3.974730} \\ \mathbf{-5.163440} \end{bmatrix}$$
Under gradient descent, $\mathbf{w}_c$ increases, reinforcing the slope along both coordinates to penalize the gradient norm deficit toward 1!

---

### Illustration 4: Power Iteration for Spectral Normalization Step-by-Step

**Problem:**
Consider a critic linear layer weight matrix $\mathbf{W} \in \mathbb{R}^{2 \times 2}$:
$$\mathbf{W} = \begin{bmatrix} 3.00 & 1.00 \\ 2.00 & 4.00 \end{bmatrix}$$
Initial unit vector guess:
$$\mathbf{v}^{(0)} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.707107 \\ 0.707107 \end{bmatrix}$$
1. Execute 2 full steps of Power Iteration to compute left singular vector $\mathbf{u}^{(k)}$, right singular vector $\mathbf{v}^{(k)}$, and spectral norm estimate $\sigma_k$.
2. Compute the exact analytical eigenvalues of $\mathbf{W}^\top \mathbf{W}$ to determine the true spectral norm $\sigma_{\max}(\mathbf{W}) = \sqrt{\lambda_{\max}}$.
3. Compute the spectrally normalized weight matrix $\mathbf{W}_{\text{SN}} = \frac{\mathbf{W}}{\sigma_{\max}}$ and verify that its spectral norm is exactly $1.000$.

**Step-by-Step Solution:**

**1. Power Iteration Step 1:**
- Forward multiplication:
  $$\mathbf{y}^{(1)} = \mathbf{W} \mathbf{v}^{(0)} = \begin{bmatrix} 3.00 & 1.00 \\ 2.00 & 4.00 \end{bmatrix} \begin{bmatrix} 0.707107 \\ 0.707107 \end{bmatrix} = \begin{bmatrix} 3.0(0.707107) + 1.0(0.707107) \\ 2.0(0.707107) + 4.0(0.707107) \end{bmatrix} = \begin{bmatrix} 2.828427 \\ 4.242641 \end{bmatrix}$$
- Vector norm:
  $$\|\mathbf{y}^{(1)}\|_2 = \sqrt{(2.828427)^2 + (4.242641)^2} = \sqrt{8.000000 + 18.000000} = \sqrt{26.000000} \approx \mathbf{5.099020}$$
- Left singular vector estimate $\mathbf{u}^{(1)}$:
  $$\mathbf{u}^{(1)} = \frac{\mathbf{y}^{(1)}}{\|\mathbf{y}^{(1)}\|_2} = \begin{bmatrix} \frac{2.828427}{5.099020} \\ \frac{4.242641}{5.099020} \end{bmatrix} = \begin{bmatrix} \mathbf{0.554700} \\ \mathbf{0.832050} \end{bmatrix}$$
- Backward multiplication:
  $$\tilde{\mathbf{v}}^{(1)} = \mathbf{W}^\top \mathbf{u}^{(1)} = \begin{bmatrix} 3.00 & 2.00 \\ 1.00 & 4.00 \end{bmatrix} \begin{bmatrix} 0.554700 \\ 0.832050 \end{bmatrix} = \begin{bmatrix} 3(0.554700) + 2(0.832050) \\ 1(0.554700) + 4(0.832050) \end{bmatrix} = \begin{bmatrix} 1.664100 + 1.664100 \\ 0.554700 + 3.328200 \end{bmatrix} = \begin{bmatrix} 3.328200 \\ 3.882900 \end{bmatrix}$$
- Vector norm:
  $$\|\tilde{\mathbf{v}}^{(1)}\|_2 = \sqrt{(3.328200)^2 + (3.882900)^2} = \sqrt{11.076915 + 15.076912} = \sqrt{26.153827} \approx \mathbf{5.114081}$$
- Updated right singular vector estimate $\mathbf{v}^{(1)}$:
  $$\mathbf{v}^{(1)} = \frac{\tilde{\mathbf{v}}^{(1)}}{\|\tilde{\mathbf{v}}^{(1)}\|_2} = \begin{bmatrix} \frac{3.328200}{5.114081} \\ \frac{3.882900}{5.114081} \end{bmatrix} = \begin{bmatrix} \mathbf{0.650791} \\ \mathbf{0.759257} \end{bmatrix}$$
- First spectral norm estimate:
  $$\sigma_1 = (\mathbf{u}^{(1)})^\top \mathbf{W} \mathbf{v}^{(1)} = [0.554700, 0.832050] \begin{bmatrix} 3(0.650791) + 1(0.759257) \\ 2(0.650791) + 4(0.759257) \end{bmatrix} = [0.554700, 0.832050] \begin{bmatrix} 2.711630 \\ 4.338610 \end{bmatrix}$$
  $$\sigma_1 = 0.554700(2.711630) + 0.832050(4.338610) = 1.504141 + 3.609941 = \mathbf{5.114082}$$

**2. Power Iteration Step 2:**
- Forward multiplication:
  $$\mathbf{y}^{(2)} = \mathbf{W} \mathbf{v}^{(1)} = \begin{bmatrix} 2.711630 \\ 4.338610 \end{bmatrix}$$
  $$\|\mathbf{y}^{(2)}\|_2 = \sqrt{2.711630^2 + 4.338610^2} = \sqrt{7.352937 + 18.823537} = \sqrt{26.176474} \approx \mathbf{5.116295}$$
  $$\mathbf{u}^{(2)} = \begin{bmatrix} \frac{2.711630}{5.116295} \\ \frac{4.338610}{5.116295} \end{bmatrix} = \begin{bmatrix} \mathbf{0.530000} \\ \mathbf{0.847998} \end{bmatrix}$$
- Backward multiplication:
  $$\tilde{\mathbf{v}}^{(2)} = \mathbf{W}^\top \mathbf{u}^{(2)} = \begin{bmatrix} 3.00 & 2.00 \\ 1.00 & 4.00 \end{bmatrix} \begin{bmatrix} 0.530000 \\ 0.847998 \end{bmatrix} = \begin{bmatrix} 1.590000 + 1.695996 \\ 0.530000 + 3.391992 \end{bmatrix} = \begin{bmatrix} 3.285996 \\ 3.921992 \end{bmatrix}$$
  $$\|\tilde{\mathbf{v}}^{(2)}\|_2 = \sqrt{3.285996^2 + 3.921992^2} = \sqrt{10.797770 + 15.382021} = \sqrt{26.179791} \approx \mathbf{5.116619}$$
  $$\mathbf{v}^{(2)} = \begin{bmatrix} \frac{3.285996}{5.116619} \\ \frac{3.921992}{5.116619} \end{bmatrix} = \begin{bmatrix} \mathbf{0.642220} \\ \mathbf{0.766520} \end{bmatrix}$$
- Second spectral norm estimate:
  $$\sigma_2 \approx \mathbf{5.116619}$$

**3. Exact Analytical Eigenvalue Solution:**
Compute $\mathbf{W}^\top \mathbf{W}$:
$$\mathbf{W}^\top \mathbf{W} = \begin{bmatrix} 3.0 & 2.0 \\ 1.0 & 4.0 \end{bmatrix} \begin{bmatrix} 3.0 & 1.0 \\ 2.0 & 4.0 \end{bmatrix} = \begin{bmatrix} 9 + 4 & 3 + 8 \\ 3 + 8 & 1 + 16 \end{bmatrix} = \begin{bmatrix} 13.0 & 11.0 \\ 11.0 & 17.0 \end{bmatrix}$$
Characteristic equation $\det(\mathbf{W}^\top \mathbf{W} - \lambda \mathbf{I}) = 0$:
$$(13 - \lambda)(17 - \lambda) - 121 = \lambda^2 - 30 \lambda + 221 - 121 = \lambda^2 - 30 \lambda + 100 = 0$$
Roots via quadratic formula:
$$\lambda = \frac{30 \pm \sqrt{900 - 400}}{2} = \frac{30 \pm \sqrt{500}}{2} = \frac{30 \pm 22.360680}{2}$$
$$\lambda_{\max} = \frac{52.360680}{2} = \mathbf{26.180340}$$
$$\sigma_{\text{true}}(\mathbf{W}) = \sqrt{\lambda_{\max}} = \sqrt{26.180340} = \mathbf{5.116673}$$
Power iteration reached $5.116619$ in just 2 iterations, achieving relative accuracy of $0.001\%$.

**4. Spectrally Normalized Weight Matrix:**
$$\mathbf{W}_{\text{SN}} = \frac{1}{5.116673} \begin{bmatrix} 3.00 & 1.00 \\ 2.00 & 4.00 \end{bmatrix} = \begin{bmatrix} \mathbf{0.586318} & \mathbf{0.195439} \\ \mathbf{0.390879} & \mathbf{0.781758} \end{bmatrix}$$
Verifying the spectral norm:
$$\sigma_{\max}(\mathbf{W}_{\text{SN}}) = \frac{\sigma_{\max}(\mathbf{W})}{5.116673} = \frac{5.116673}{5.116673} = \mathbf{1.000000}$$
The layer's Lipschitz constant is exactly 1.0.

---

### Illustration 5: Mode Collapse Dynamics on a Bimodal Target: Vanilla GAN vs. WGAN-GP

**Problem:**
Consider a 1D toy distribution with two well-separated modes:
$$p_{\text{data}}(x) = 0.50 \, \delta(x + 2.0) + 0.50 \, \delta(x - 2.0)$$
Let the generator output a single deterministic point $\theta \in [-2, 2]$: $p_g(x) = \delta(x - \theta)$.
Assume the generator is initialized near Mode 1: $\theta = -1.90$.
1. Analyze the Vanilla GAN non-saturating generator loss $\mathcal{L}_G^{\text{NS}}(\theta) = -\log D^*(G(z))$. Show why the gradient $\frac{d \mathcal{L}_G^{\text{NS}}}{d\theta}$ completely ignores the right mode at $+2.0$ (Mode Collapse trap).
2. Compute the exact Wasserstein-1 distance $W_1(p_{\text{data}}, p_g)$ as a function of $\theta$.
3. Evaluate the WGAN generator loss and gradient at $\theta = -1.90$ and explain how WGAN prevents mode collapse.

**Step-by-Step Solution:**

**1. Vanilla GAN Landscape & Mode Collapse Trap:**
The optimal discriminator is:
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$
At the generator's position $x = \theta$:
$$p_g(\theta) = 1, \quad p_{\text{data}}(\theta) = \begin{cases} 0.50 & \text{if } \theta \in \{-2.0, +2.0\} \\ 0 & \text{if } \theta \notin \{-2.0, +2.0\} \end{cases}$$
When smoothed by infinitesimal Gaussian bandwidth $\sigma \to 0$:
$$D^*(\theta) \approx \frac{0.5 e^{-(\theta + 2)^2 / (2\sigma^2)} + 0.5 e^{-(\theta - 2)^2 / (2\sigma^2)}}{0.5 e^{-(\theta + 2)^2 / (2\sigma^2)} + 0.5 e^{-(\theta - 2)^2 / (2\sigma^2)} + 1.0}$$
At $\theta = -1.90$:
- The distance to Mode 1 is $|\theta - (-2.0)| = 0.10$.
- The distance to Mode 2 is $|\theta - (+2.0)| = 3.90$.
The likelihood ratio between Mode 2 and Mode 1 is:
$$\frac{e^{-(3.90)^2 / (2\sigma^2)}}{e^{-(0.10)^2 / (2\sigma^2)}} = e^{-(15.21 - 0.01)/(2\sigma^2)} = e^{-7.6 / \sigma^2} \approx 10^{-330}$$
The gradient of non-saturating loss is dominated entirely by Mode 1:
$$\frac{d \mathcal{L}_G^{\text{NS}}}{d\theta} \approx \frac{d}{d\theta} \left[ \frac{(\theta + 2.0)^2}{2\sigma^2} \right] = \frac{\theta + 2.0}{\sigma^2} = \frac{-1.90 + 2.0}{\sigma^2} = \frac{+0.10}{\sigma^2} > 0$$
Gradient descent pulls $\theta$ directly into Mode 1 ($\theta \to -2.0$).
Once $\theta = -2.0$, $D^*(-2.0) = \frac{0.5}{0.5 + 1.0} = \frac{1}{3}$.
The gradient with respect to Mode 2 ($x = +2.0$) is mathematically zero:
$$\left. \frac{\partial \mathcal{L}_G^{\text{NS}}}{\partial \theta} \right|_{\text{due to Mode 2}} \approx 0.0000000000000000$$
The generator is trapped permanently in Mode 1. It has zero knowledge that Mode 2 even exists!

**2. Wasserstein-1 Distance as a Function of $\theta$:**
By definition of optimal transport on $\mathbb{R}$:
$$W_1(p_{\text{data}}, \delta_\theta) = \mathbb{E}_{x \sim p_{\text{data}}} [|x - \theta|] = 0.50 |-2.0 - \theta| + 0.50 |+2.0 - \theta|$$
For any $\theta \in [-2.0, +2.0]$:
$$|-2.0 - \theta| = \theta + 2.0, \qquad |+2.0 - \theta| = 2.0 - \theta$$
$$W_1(p_{\text{data}}, \delta_\theta) = 0.50(\theta + 2.0) + 0.50(2.0 - \theta) = 0.50\theta + 1.00 + 1.00 - 0.50\theta = \mathbf{2.000000}$$
Notice that across the entire interval $\theta \in [-2.0, +2.0]$, $W_1$ is constant ($2.0$).
If the generator uses a latent noise distribution $\mathbf{z} \sim \mathcal{U}[-1, 1]$ with $G_{\theta}(z) = \mu + s z$:
- Mode collapse occurs if $s = 0$ (single point $\mu$).
- The Wasserstein distance for spread $s$ is:
  $$W_1(s) = \int_{-1}^1 \left( 0.5 |-2 - (\mu + sz)| + 0.5 |2 - (\mu + sz)| \right) \frac{1}{2} dz$$
As $s \to 2.0$ (spanning the entire distance between $-2$ and $+2$), $W_1(s)$ drops strictly from $2.0 \to 1.0$!
The WGAN objective provides a constant, non-saturating force pushing the generator to expand its scale parameter $s$ to encompass both modes, providing the algebraic foundation for mode recovery.

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
