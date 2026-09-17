# Chapter 10.1: Taxonomy of Generative Models

---

## 1. Intuition & 101 Motivation

Machine learning models fall into two foundational probabilistic paradigms:
1. **Discriminative Models:** Model the conditional probability distribution $P(Y \mid \mathbf{X})$. Given an image $\mathbf{X}$, they partition the input space into decision regions to predict a target label $Y \in \{0, 1\}$. They do not need to understand what makes an image realistic; they only care about differences between classes.
2. **Generative Models:** Model the underlying data distribution $P(\mathbf{X})$ directly (or the joint distribution $P(\mathbf{X}, Y)$). To generate a realistic sample, the model must understand the complete structural, textural, and semantic dependencies of the high-dimensional world.

```
       DISCRIMINATIVE vs. GENERATIVE PARADIGM
       
       Discriminative: P(Y | X)                 Generative: P(X)
       Finds decision boundary                Captures complete data distribution
       
            Class 0      Class 1                     High-Density Manifold
            o   o   |   x   x                             *   *   *
              o  o  |  x   x                             *  DATA  *
            o   o   |   x   x                             *   *   *
                    ▲                               (Can sample novel points!)
                    │ Decision Boundary
```

### The Curse of Dimensionality & The Manifold Hypothesis
Consider a modest image of $256 \times 256$ RGB pixels. It lives in a continuous space of dimension:
$$D = 256 \times 256 \times 3 = 196,608 \text{ dimensions}$$
If you randomly sample a vector from $[0, 255]^{196608}$, the result is pure, imperceptible Gaussian static with probability $1 - 10^{-1000}$.

**The Manifold Hypothesis:** Natural data (faces, natural landscapes, legal contracts, human speech) does not fill this vast $196,608$-dimensional void. Instead, it concentrates along an infinitesimally thin, highly non-linear, low-dimensional sub-manifold embedded within $\mathbb{R}^D$.
The goal of a generative model is to learn a smooth mapping (a push-forward measure) that transports a simple, tractable probability distribution (such as an isotropic Gaussian $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$) onto this intricate data manifold.

---

### The Generative Learning Trilemma (Xiao et al., 2021)
Every generative modeling architecture navigates a fundamental trilemma between three competing properties:
1. **High Sample Quality:** Samples are crisp, coherent, photorealistic, and indistinguishable from real data.
2. **Fast Sampling:** Generating a sample requires only one (or very few) forward passes through a neural network.
3. **Mode Coverage (Diversity):** The model captures all modes of the true distribution without collapsing or dropping rare categories.

```
                         THE GENERATIVE LEARNING TRILEMMA
                                High Sample Quality
                                       ▲
                                      / \
                                     /   \
                             GANs   /     \  Diffusion Models
                                   /       \
                                  /         \
                                 /           \
               Fast Sampling ◄──/─────────────\──► Mode Coverage
                                  VAEs & Flows
```

Historically, no single architecture maximized all three:
- **GANs:** High quality + Fast sampling, but suffer from **Mode Collapse** and unstable minimax training.
- **VAEs:** Fast sampling + Full mode coverage, but generate **blurry, lower-fidelity samples**.
- **Diffusion Models:** Unmatched quality + Full mode coverage, but historically suffered from **slow iterative sampling** (requiring 20 to 1,000 sequential denoising steps).

---

## 2. Rigorous Mathematical Formulation

```
                     THE MATHEMATICAL TAXONOMY OF GENERATIVE MODELS
                                  Generative Models
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
          Explicit Density                               Implicit Density
                   │                                             │
         ┌─────────┴─────────┐                                   ▼
         ▼                   ▼                         GANs (Goodfellow 2014)
   Tractable Density   Approximate Density            (Minimax Game, no density)
         │                   │
         ├─ Autoregressive   ├─ Variational Autoencoders (VAEs)
         │  (PixelCNN, GPT)  │  (ELBO optimization)
         │                   │
         └─ Normalizing      ├─ Diffusion Models (DDPM)
            Flows (RealNVP)  │  (Score matching, reverse SDE)
                             │
                             └─ Energy-Based Models (EBMs)
                                (Unnormalized e^-E(x)/Z)
```

---

### 2.1 The Maximum Likelihood Estimator & Forward vs. Reverse KL

Given true empirical data distribution $p_{\text{data}}(\mathbf{x})$ and parameterized model distribution $p_{\boldsymbol{\theta}}(\mathbf{x})$, maximum likelihood estimation minimizes the **Forward Kullback-Leibler (KL) Divergence**:

$$D_{\text{KL}}(p_{\text{data}} \,\|\, p_{\boldsymbol{\theta}}) = \int p_{\text{data}}(\mathbf{x}) \log \frac{p_{\text{data}}(\mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{x})} d\mathbf{x} = \underbrace{\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}(\mathbf{x})]}_{\text{Constant entropy } -H(p_{\text{data}})} - \mathbb{E}_{p_{\text{data}}}[\log p_{\boldsymbol{\theta}}(\mathbf{x})]$$

Minimizing $D_{\text{KL}}(p_{\text{data}} \,\|\, p_{\boldsymbol{\theta}})$ is strictly equivalent to maximizing the expected log-likelihood:
$$\arg\min_{\boldsymbol{\theta}} D_{\text{KL}}(p_{\text{data}} \,\|\, p_{\boldsymbol{\theta}}) \equiv \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log p_{\boldsymbol{\theta}}(\mathbf{x}) \right]$$

#### The Forward vs. Reverse KL Asymmetry:
1. **Forward KL ($D_{\text{KL}}(p_{\text{data}} \,\|\, p_{\boldsymbol{\theta}})$ - Mode-Covering / Zero-Avoiding):**
   If $p_{\text{data}}(\mathbf{x}) > 0$, the model must ensure $p_{\boldsymbol{\theta}}(\mathbf{x}) > 0$; otherwise $\log \frac{p_{\text{data}}}{p_{\boldsymbol{\theta}}} \to \infty$.
   To avoid infinite penalties, $p_{\boldsymbol{\theta}}$ broadens its support to cover **all modes** of the data, even if it places probability mass in empty valleys between modes (producing blurry VAE samples).
2. **Reverse KL ($D_{\text{KL}}(p_{\boldsymbol{\theta}} \,\|\, p_{\text{data}})$ - Mode-Dropping / Zero-Forcing):**
   If $p_{\text{data}}(\mathbf{x}) = 0$, the model must force $p_{\boldsymbol{\theta}}(\mathbf{x}) = 0$ to avoid infinite loss.
   The model prefers to concentrate its probability mass sharply on a **single mode** of the data, completely ignoring other modes (the mode collapse phenomenon in GANs).

```
         FORWARD KL (Mode-Covering)                  REVERSE KL (Mode-Dropping)
               p_data (Bimodal)                            p_data (Bimodal)
               ┌──┐        ┌──┐                            ┌──┐        ┌──┐
               │  │        │  │                            │  │        │  │
            ───┴──┴────────┴──┴───                      ───┴──┴────────┴──┴───
               p_theta (Spreads across both!)              p_theta (Collapses to one mode!)
             ┌────────────────────┐                        ┌──┐
            ─┴────────────────────┴─                    ───┴──┴───────────────
```

---

### 2.2 The Invertible Change of Variables Theorem (Normalizing Flows)

Let $\mathbf{z} \in \mathbb{R}^d$ be a latent vector sampled from a known tractable prior $p_{\mathbf{z}}(\mathbf{z})$ (e.g., standard normal $\mathcal{N}(\mathbf{0}, \mathbf{I})$).
Let $\mathbf{x} = f_{\boldsymbol{\theta}}(\mathbf{z})$ be a bijective (invertible) and differentiable transformation ($f_{\boldsymbol{\theta}}: \mathbb{R}^d \to \mathbb{R}^d$).

By conservation of probability mass across differential volumes $d\mathbf{x}$ and $d\mathbf{z}$:
$$p_{\mathbf{x}}(\mathbf{x}) |d\mathbf{x}| = p_{\mathbf{z}}(\mathbf{z}) |d\mathbf{z}|$$

Rearranging:
$$p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}\left( f_{\boldsymbol{\theta}}^{-1}(\mathbf{x}) \right) \cdot \left| \det \left( \frac{\partial f_{\boldsymbol{\theta}}^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right) \right|$$

Taking the logarithm:
$$\log p_{\mathbf{x}}(\mathbf{x}) = \log p_{\mathbf{z}}\left( f_{\boldsymbol{\theta}}^{-1}(\mathbf{x}) \right) + \log \left| \det \mathbf{J}_{f^{-1}}(\mathbf{x}) \right|$$
where $\mathbf{J}_{f^{-1}} \in \mathbb{R}^{d \times d}$ is the Jacobian matrix of the inverse mapping.

**Architectural Requirement:**
For Normalizing Flows (e.g., RealNVP, Glow), the transformation $f_{\boldsymbol{\theta}}$ must be designed with an **upper or lower triangular Jacobian**, so that the determinant is simply the product of diagonal elements:
$$\det \mathbf{J} = \prod_{i=1}^d J_{i, i} \implies \log |\det \mathbf{J}| = \sum_{i=1}^d \log |J_{i, i}|$$
evaluating in $\mathcal{O}(d)$ time instead of $\mathcal{O}(d^3)$.

---

### 2.3 Energy-Based Models (EBMs) & Intractable Partition Functions

An Energy-Based Model parameterizes the probability density using an unconstrained neural network scalar energy function $E_{\boldsymbol{\theta}}(\mathbf{x}) \in \mathbb{R}$:
$$p_{\boldsymbol{\theta}}(\mathbf{x}) = \frac{\exp\left( -E_{\boldsymbol{\theta}}(\mathbf{x}) \right)}{Z(\boldsymbol{\theta})}$$
where $Z(\boldsymbol{\theta})$ is the **partition function**:
$$Z(\boldsymbol{\theta}) = \int_{\mathbb{R}^D} \exp\left( -E_{\boldsymbol{\theta}}(\mathbf{x}) \right) d\mathbf{x}$$

#### The Intractability Crisis:
In $D = 196,608$ dimensions, computing the integral $Z(\boldsymbol{\theta})$ analytically or numerically via naive Monte Carlo is physically impossible.
Taking the log-likelihood gradient:
$$\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}) = -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) - \nabla_{\boldsymbol{\theta}} \log Z(\boldsymbol{\theta})$$
$$\nabla_{\boldsymbol{\theta}} \log Z(\boldsymbol{\theta}) = \frac{1}{Z(\boldsymbol{\theta})} \int -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}') e^{-E_{\boldsymbol{\theta}}(\mathbf{x}')} d\mathbf{x}' = - \mathbb{E}_{\mathbf{x}' \sim p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}') \right]$$

Thus:
$$\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}) = \underbrace{-\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x})}_{\text{Positive Phase (Pushes energy of data down)}} + \underbrace{\mathbb{E}_{\mathbf{x}' \sim p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}') \right]}_{\text{Negative Phase (Pushes energy of hallucinations up)}}$$

Sampling $\mathbf{x}' \sim p_{\boldsymbol{\theta}}$ requires expensive Markov Chain Monte Carlo (MCMC) Langevin dynamics at every gradient step.

---

### 2.4 Rigorous Mathematical Derivations

#### Derivation 10.1.1: Equivalence of Maximum Likelihood Estimation and Forward KL Minimization under Empirical Measure and Local Fisher Curvature

**1. Context and Assumptions:**
Let $\mathcal{X} \subseteq \mathbb{R}^D$ denote the sample data space.
Assume an unknown true data-generating distribution $p^*(\mathbf{x}) = p_{\text{data}}(\mathbf{x})$.
We observe an independently and identically distributed (i.i.d.) dataset $\mathcal{D} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(N)}\}$.
The empirical distribution is represented as a sum of Dirac delta measures:
$$p_{\text{data}}^{(N)}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N \delta(\mathbf{x} - \mathbf{x}^{(i)})$$
Let $\mathcal{P} = \{p_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \Theta \subseteq \mathbb{R}^P\}$ be a family of parametric probability densities with shared support satisfying $\operatorname{supp}(p^*) \subseteq \operatorname{supp}(p_{\boldsymbol{\theta}})$, where $p_{\boldsymbol{\theta}}(\mathbf{x}) > 0$ for all $\mathbf{x} \in \operatorname{supp}(p^*)$. Assume $p_{\boldsymbol{\theta}}(\mathbf{x})$ is twice continuously differentiable with respect to $\boldsymbol{\theta}$.

**2. Step 1: Definition of KL Divergence & Non-Negativity (Gibbs' Inequality):**
The Forward Kullback-Leibler divergence from $p_{\boldsymbol{\theta}}$ to $p^*$ is defined as:
$$D_{\mathrm{KL}}(p^* \,\|\, p_{\boldsymbol{\theta}}) \triangleq \int_{\mathcal{X}} p^*(\mathbf{x}) \log \left( \frac{p^*(\mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{x})} \right) d\mathbf{x} = -\int_{\mathcal{X}} p^*(\mathbf{x}) \log \left( \frac{p_{\boldsymbol{\theta}}(\mathbf{x})}{p^*(\mathbf{x})} \right) d\mathbf{x}$$
Because $\phi(u) = -\log(u)$ is strictly convex for $u > 0$ (since $\phi''(u) = 1/u^2 > 0$), by Jensen's inequality:
$$D_{\mathrm{KL}}(p^* \,\|\, p_{\boldsymbol{\theta}}) = \mathbb{E}_{\mathbf{x} \sim p^*} \left[ -\log \left( \frac{p_{\boldsymbol{\theta}}(\mathbf{x})}{p^*(\mathbf{x})} \right) \right] \ge -\log \left( \mathbb{E}_{\mathbf{x} \sim p^*} \left[ \frac{p_{\boldsymbol{\theta}}(\mathbf{x})}{p^*(\mathbf{x})} \right] \right)$$
Evaluating the inner expectation:
$$\mathbb{E}_{\mathbf{x} \sim p^*} \left[ \frac{p_{\boldsymbol{\theta}}(\mathbf{x})}{p^*(\mathbf{x})} \right] = \int_{\mathcal{X}} p^*(\mathbf{x}) \frac{p_{\boldsymbol{\theta}}(\mathbf{x})}{p^*(\mathbf{x})} d\mathbf{x} = \int_{\mathcal{X}} p_{\boldsymbol{\theta}}(\mathbf{x}) d\mathbf{x} = 1$$
Therefore:
$$D_{\mathrm{KL}}(p^* \,\|\, p_{\boldsymbol{\theta}}) \ge -\log(1) = 0$$
Strict convexity implies equality holds if and only if $\frac{p_{\boldsymbol{\theta}}(\mathbf{x})}{p^*(\mathbf{x})} = 1$ almost everywhere with respect to $p^*$.

**3. Step 2: Information Decomposition into Entropy and Cross-Entropy:**
Expanding the logarithmic ratio:
$$D_{\mathrm{KL}}(p^* \,\|\, p_{\boldsymbol{\theta}}) = \int_{\mathcal{X}} p^*(\mathbf{x}) \log p^*(\mathbf{x}) d\mathbf{x} - \int_{\mathcal{X}} p^*(\mathbf{x}) \log p_{\boldsymbol{\theta}}(\mathbf{x}) d\mathbf{x} = -H(p^*) + H(p^*, p_{\boldsymbol{\theta}})$$
Notice that the true data entropy $H(p^*) \triangleq -\mathbb{E}_{\mathbf{x} \sim p^*}[\log p^*(\mathbf{x})]$ is purely a property of nature and has zero partial derivative with respect to model parameters $\boldsymbol{\theta}$:
$$\nabla_{\boldsymbol{\theta}} D_{\mathrm{KL}}(p^* \,\|\, p_{\boldsymbol{\theta}}) = -\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x} \sim p^*} [\log p_{\boldsymbol{\theta}}(\mathbf{x})]$$

**4. Step 3: Empirical Approximation via Sample Measure:**
Substituting the empirical measure $p_{\text{data}}^{(N)}(\mathbf{x})$ in place of the population density $p^*(\mathbf{x})$:
$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}^{(N)}} [\log p_{\boldsymbol{\theta}}(\mathbf{x})] = \int_{\mathcal{X}} \left( \frac{1}{N} \sum_{i=1}^N \delta(\mathbf{x} - \mathbf{x}^{(i)}) \right) \log p_{\boldsymbol{\theta}}(\mathbf{x}) d\mathbf{x} = \frac{1}{N} \sum_{i=1}^N \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})$$
Hence, minimizing empirical forward KL divergence:
$$\arg\min_{\boldsymbol{\theta} \in \Theta} D_{\mathrm{KL}}\left( p_{\text{data}}^{(N)} \,\|\, p_{\boldsymbol{\theta}} \right) \equiv \arg\max_{\boldsymbol{\theta} \in \Theta} \frac{1}{N} \sum_{i=1}^N \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) = \arg\max_{\boldsymbol{\theta} \in \Theta} \mathcal{L}_{\mathrm{MLE}}(\boldsymbol{\theta})$$
By the Strong Law of Large Numbers, $\frac{1}{N}\sum_{i=1}^N \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) \xrightarrow{\text{a.s.}} \mathbb{E}_{p^*}[\log p_{\boldsymbol{\theta}}(\mathbf{x})]$ as $N \to \infty$.

**5. Step 4: Local Curvature and the Fisher Information Metric:**
Suppose the model is well-specified, meaning $\exists \boldsymbol{\theta}^* \in \Theta$ such that $p_{\boldsymbol{\theta}^*} = p^*$.
Consider a small parameter perturbation $\boldsymbol{\theta} = \boldsymbol{\theta}^* + \Delta \boldsymbol{\theta}$.
Taylor expand $D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}^* + \Delta \boldsymbol{\theta}})$ around $\Delta \boldsymbol{\theta} = \mathbf{0}$:
$$D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}^* + \Delta \boldsymbol{\theta}}) = D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}^*}) + \left. \nabla_{\boldsymbol{\theta}} D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}}) \right|_{\boldsymbol{\theta}^*}^\top \Delta \boldsymbol{\theta} + \frac{1}{2} \Delta \boldsymbol{\theta}^\top \left. \nabla_{\boldsymbol{\theta}}^2 D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}}) \right|_{\boldsymbol{\theta}^*} \Delta \boldsymbol{\theta} + \mathcal{O}(\|\Delta \boldsymbol{\theta}\|^3)$$
Since $D_{\mathrm{KL}} \ge 0$ achieves its global minimum at $\boldsymbol{\theta}^*$, the zeroth-order term is 0 and the first-order gradient vanishes:
$$\left. \nabla_{\boldsymbol{\theta}} D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}}) \right|_{\boldsymbol{\theta}^*} = -\mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}^*}} \left[ \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}) \right]_{\boldsymbol{\theta}^*} = -\int \nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}^*}(\mathbf{x}) d\mathbf{x} = -\nabla_{\boldsymbol{\theta}} (1) = \mathbf{0}$$
Now differentiate the expectation twice:
$$\nabla_{\boldsymbol{\theta}}^2 D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}}) = -\mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}^*}} \left[ \nabla_{\boldsymbol{\theta}}^2 \log p_{\boldsymbol{\theta}}(\mathbf{x}) \right]$$
Using the identity $\nabla^2 \log f = \frac{\nabla^2 f}{f} - \frac{\nabla f \nabla f^\top}{f^2}$:
$$\nabla_{\boldsymbol{\theta}}^2 \log p_{\boldsymbol{\theta}}(\mathbf{x}) = \frac{\nabla_{\boldsymbol{\theta}}^2 p_{\boldsymbol{\theta}}(\mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{x})} - \left( \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}) \right) \left( \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}) \right)^\top$$
Taking expectation under $p_{\boldsymbol{\theta}^*}$:
$$\mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}^*}} \left[ \frac{\nabla_{\boldsymbol{\theta}}^2 p_{\boldsymbol{\theta}^*}(\mathbf{x})}{p_{\boldsymbol{\theta}^*}(\mathbf{x})} \right] = \int_{\mathcal{X}} \nabla_{\boldsymbol{\theta}}^2 p_{\boldsymbol{\theta}^*}(\mathbf{x}) d\mathbf{x} = \nabla_{\boldsymbol{\theta}}^2 \int_{\mathcal{X}} p_{\boldsymbol{\theta}^*}(\mathbf{x}) d\mathbf{x} = \mathbf{0}$$
Therefore:
$$\left. \nabla_{\boldsymbol{\theta}}^2 D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}}) \right|_{\boldsymbol{\theta}=\boldsymbol{\theta}^*} = \mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}^*}} \left[ \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}^*}(\mathbf{x}) \, \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}^*}(\mathbf{x})^\top \right] \equiv \mathbf{I}(\boldsymbol{\theta}^*)$$
where $\mathbf{I}(\boldsymbol{\theta}^*)$ is the **Fisher Information Matrix (FIM)**.
Locally, the Forward KL divergence is a Riemannian metric with metric tensor given by the Fisher Information:
$$D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^*} \,\|\, p_{\boldsymbol{\theta}^* + \Delta \boldsymbol{\theta}}) = \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{I}(\boldsymbol{\theta}^*) \Delta \boldsymbol{\theta} + \mathcal{O}(\|\Delta \boldsymbol{\theta}\|^3)$$

---

#### Derivation 10.1.2: Multidimensional Change of Variables, Differential Volume Forms, and Affine Coupling Layer Determinant (RealNVP)

**1. Context and Assumptions:**
Let $\mathbf{z} \in \mathbb{R}^D$ be a random vector with known continuous probability density $p_{\mathbf{z}}(\mathbf{z})$.
Let $f: \mathbb{R}^D \to \mathbb{R}^D$ be a $C^1$-diffeomorphism (a smooth, bijective map with smooth inverse $f^{-1}: \mathbb{R}^D \to \mathbb{R}^D$).
Let $\mathbf{x} = f(\mathbf{z})$. We seek the push-forward density $p_{\mathbf{x}}(\mathbf{x})$.

**2. Step 1: Differential Volume Forms and the Exterior Product:**
In multivariable calculus, an infinitesimal parallelotope spanned by differential coordinate vectors $\{d\mathbf{z}_1, \dots, d\mathbf{z}_D\}$ transforms under $f$ into an infinitesimal parallelotope in $\mathcal{X}$ space.
Using the exterior wedge product:
$$dx_1 \wedge dx_2 \wedge \cdots \wedge dx_D = \det(\mathbf{J}_f(\mathbf{z})) \, dz_1 \wedge dz_2 \wedge \cdots \wedge dz_D$$
where the Jacobian matrix $\mathbf{J}_f(\mathbf{z}) \in \mathbb{R}^{D \times D}$ is defined element-wise by $[\mathbf{J}_f(\mathbf{z})]_{i, j} = \frac{\partial f_i(\mathbf{z})}{\partial z_j}$.
By conservation of probability mass over any measurable set $\Omega \subset \mathbb{R}^D$:
$$\mathbb{P}(\mathbf{x} \in f(\Omega)) = \mathbb{P}(\mathbf{z} \in \Omega) \implies \int_{f(\Omega)} p_{\mathbf{x}}(\mathbf{x}) d\mathbf{x} = \int_{\Omega} p_{\mathbf{z}}(\mathbf{z}) d\mathbf{z}$$
Applying the multivariable change of variables theorem to the left integral with substitution $\mathbf{x} = f(\mathbf{z}), d\mathbf{x} = |\det \mathbf{J}_f(\mathbf{z})| d\mathbf{z}$:
$$\int_{\Omega} p_{\mathbf{x}}(f(\mathbf{z})) |\det \mathbf{J}_f(\mathbf{z})| d\mathbf{z} = \int_{\Omega} p_{\mathbf{z}}(\mathbf{z}) d\mathbf{z}$$
Since this equality must hold for arbitrary open sets $\Omega$, the integrands must be identical almost everywhere:
$$p_{\mathbf{x}}(f(\mathbf{z})) |\det \mathbf{J}_f(\mathbf{z})| = p_{\mathbf{z}}(\mathbf{z}) \implies p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \cdot \frac{1}{|\det \mathbf{J}_f(f^{-1}(\mathbf{x}))|}$$
By the Inverse Function Theorem, $\mathbf{J}_{f^{-1}}(\mathbf{x}) = [\mathbf{J}_f(f^{-1}(\mathbf{x}))]^{-1}$, and $\det(\mathbf{A}^{-1}) = (\det \mathbf{A})^{-1}$:
$$p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \cdot \left| \det \mathbf{J}_{f^{-1}}(\mathbf{x}) \right|$$
In log-density space:
$$\log p_{\mathbf{x}}(\mathbf{x}) = \log p_{\mathbf{z}}(f^{-1}(\mathbf{x})) + \log \left| \det \mathbf{J}_{f^{-1}}(\mathbf{x}) \right|$$

**3. Step 2: Multi-Layer Composition of Flows:**
If $f = f_K \circ f_{K-1} \circ \cdots \circ f_1$, let $\mathbf{h}_0 = \mathbf{z}$ and $\mathbf{h}_k = f_k(\mathbf{h}_{k-1})$ with $\mathbf{h}_K = \mathbf{x}$.
By the multivariate chain rule:
$$\mathbf{J}_f(\mathbf{z}) = \mathbf{J}_{f_K}(\mathbf{h}_{K-1}) \cdot \mathbf{J}_{f_{K-1}}(\mathbf{h}_{K-2}) \cdots \mathbf{J}_{f_1}(\mathbf{h}_0)$$
Taking the determinant of the product:
$$\det \mathbf{J}_f(\mathbf{z}) = \prod_{k=1}^K \det \mathbf{J}_{f_k}(\mathbf{h}_{k-1})$$
Taking logarithms:
$$\log p_{\mathbf{x}}(\mathbf{x}) = \log p_{\mathbf{z}}(\mathbf{h}_0) - \sum_{k=1}^K \log \left| \det \mathbf{J}_{f_k}(\mathbf{h}_{k-1}) \right|$$

**4. Step 3: RealNVP Affine Coupling Layer Construction:**
To evaluate $\log |\det \mathbf{J}|$ in $\mathcal{O}(D)$ rather than cubic $\mathcal{O}(D^3)$ time, RealNVP partitions the vector into two parts: $\mathbf{z} = [\mathbf{z}_{1:d}^\top, \mathbf{z}_{d+1:D}^\top]^\top$, where $1 \le d < D$.
Define the forward transformation:
$$\begin{cases}
\mathbf{x}_{1:d} = \mathbf{z}_{1:d} \\
\mathbf{x}_{d+1:D} = \mathbf{z}_{d+1:D} \odot \exp\left( s(\mathbf{z}_{1:d}) \right) + t(\mathbf{z}_{1:d})
\end{cases}$$
where $s: \mathbb{R}^d \to \mathbb{R}^{D-d}$ is a scale neural network, $t: \mathbb{R}^d \to \mathbb{R}^{D-d}$ is a translation neural network, and $\odot$ denotes the Hadamard (element-wise) product.
Crucially, **$s$ and $t$ can be arbitrarily complex, non-invertible neural networks** (e.g. deep ResNets with ReLU/GELU activations).

**5. Step 4: Jacobian Matrix and Trivial Log-Determinant:**
Compute the partial derivatives of the output blocks with respect to the input blocks:
$$\frac{\partial \mathbf{x}_{1:d}}{\partial \mathbf{z}_{1:d}} = \mathbf{I}_d, \qquad \frac{\partial \mathbf{x}_{1:d}}{\partial \mathbf{z}_{d+1:D}} = \mathbf{0}_{d \times (D-d)}$$
$$\frac{\partial \mathbf{x}_{d+1:D}}{\partial \mathbf{z}_{1:d}} = \mathbf{M} \in \mathbb{R}^{(D-d) \times d} \quad \left( \text{complex Jacobian of } s \text{ and } t \right)$$
$$\frac{\partial \mathbf{x}_{d+1:D}}{\partial \mathbf{z}_{d+1:D}} = \operatorname{diag}\left( \exp(s(\mathbf{z}_{1:d})) \right) \in \mathbb{R}^{(D-d) \times (D-d)}$$
The full Jacobian matrix is block lower-triangular:
$$\mathbf{J} = \begin{bmatrix}
\mathbf{I}_d & \mathbf{0} \\
\mathbf{M} & \operatorname{diag}\left( \exp(s(\mathbf{z}_{1:d})) \right)
\end{bmatrix}$$
The determinant of a block triangular matrix is the product of the determinants of its diagonal blocks:
$$\det \mathbf{J} = \det(\mathbf{I}_d) \cdot \det\left( \operatorname{diag}\left( \exp(s(\mathbf{z}_{1:d})) \right) \right) = 1 \cdot \prod_{j=1}^{D-d} \exp(s_j(\mathbf{z}_{1:d})) = \exp \left( \sum_{j=1}^{D-d} s_j(\mathbf{z}_{1:d}) \right)$$
Taking the log absolute determinant:
$$\log |\det \mathbf{J}| = \sum_{j=1}^{D-d} s_j(\mathbf{z}_{1:d})$$
Notice that $\mathbf{M}$ (which contains all derivatives of $s$ and $t$) completely disappears from the determinant! Computing the Jacobian determinant requires **zero extra backpropagation steps**, evaluating in exact $\mathcal{O}(D-d)$ scalar additions.

**6. Step 5: Exact Inversion Without Inverting $s$ or $t$:**
Given $\mathbf{x} = [\mathbf{x}_{1:d}^\top, \mathbf{x}_{d+1:D}^\top]^\top$:
1. $\mathbf{z}_{1:d} = \mathbf{x}_{1:d}$ (identity copy).
2. Because $\mathbf{z}_{1:d}$ is known, evaluate $s(\mathbf{z}_{1:d}) = s(\mathbf{x}_{1:d})$ and $t(\mathbf{z}_{1:d}) = t(\mathbf{x}_{1:d})$ in a standard forward pass.
3. Invert the affine transformation element-wise:
   $$\mathbf{z}_{d+1:D} = \left( \mathbf{x}_{d+1:D} - t(\mathbf{x}_{1:d}) \right) \odot \exp\left( -s(\mathbf{x}_{1:d}) \right)$$
This yields exact algebraic reconstruction with zero numerical drift.

---

#### Derivation 10.1.3: Energy-Based Models: Log-Likelihood Gradient, the Covariance Identity, and Langevin Sampling Dynamics

**1. Context and Assumptions:**
Let $E_{\boldsymbol{\theta}}(\mathbf{x}) \in \mathbb{R}$ be an unnormalized scalar energy function parameterized by $\boldsymbol{\theta} \in \mathbb{R}^P$.
The probability density is the Gibbs-Boltzmann distribution:
$$p_{\boldsymbol{\theta}}(\mathbf{x}) = \frac{e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}}{Z(\boldsymbol{\theta})}, \quad \text{where } Z(\boldsymbol{\theta}) \triangleq \int_{\mathbb{R}^D} e^{-E_{\boldsymbol{\theta}}(\mathbf{x})} d\mathbf{x} < \infty$$
Assume $e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}$ and its parameter derivatives are integrable over $\mathbb{R}^D$, permitting differentiation under the integral sign via the Dominated Convergence Theorem.

**2. Step 1: Derivative of the Log-Partition Function:**
Compute the gradient of $\log Z(\boldsymbol{\theta})$ with respect to $\boldsymbol{\theta}$:
$$\nabla_{\boldsymbol{\theta}} \log Z(\boldsymbol{\theta}) = \frac{1}{Z(\boldsymbol{\theta})} \nabla_{\boldsymbol{\theta}} Z(\boldsymbol{\theta}) = \frac{1}{Z(\boldsymbol{\theta})} \nabla_{\boldsymbol{\theta}} \int_{\mathbb{R}^D} e^{-E_{\boldsymbol{\theta}}(\mathbf{x})} d\mathbf{x}$$
Passing the gradient operator through the integral:
$$\nabla_{\boldsymbol{\theta}} \log Z(\boldsymbol{\theta}) = \frac{1}{Z(\boldsymbol{\theta})} \int_{\mathbb{R}^D} \left( -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \right) e^{-E_{\boldsymbol{\theta}}(\mathbf{x})} d\mathbf{x} = \int_{\mathbb{R}^D} \left( -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \right) \underbrace{\frac{e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}}{Z(\boldsymbol{\theta})}}_{p_{\boldsymbol{\theta}}(\mathbf{x})} d\mathbf{x}$$
By definition of mathematical expectation:
$$\nabla_{\boldsymbol{\theta}} \log Z(\boldsymbol{\theta}) = -\mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \right]$$
Now compute the gradient of the log-likelihood for a data point $\mathbf{x}_{\text{data}}$:
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}_{\text{data}}) = -E_{\boldsymbol{\theta}}(\mathbf{x}_{\text{data}}) - \log Z(\boldsymbol{\theta})$$
$$\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}_{\text{data}}) = -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}_{\text{data}}) - \nabla_{\boldsymbol{\theta}} \log Z(\boldsymbol{\theta}) = -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}_{\text{data}}) + \mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \right]$$
Taking expectation over the true data distribution $p_{\text{data}}$:
$$\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log p_{\boldsymbol{\theta}}(\mathbf{x})] = \underbrace{-\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \right]}_{\text{Positive Phase}} + \underbrace{\mathbb{E}_{\mathbf{x}' \sim p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}') \right]}_{\text{Negative Phase}}$$

**3. Step 2: The Covariance Identity & Hessian of the Free Energy:**
Differentiate $\nabla_{\boldsymbol{\theta}} \log Z(\boldsymbol{\theta})$ a second time:
$$\nabla_{\boldsymbol{\theta}}^2 \log Z(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \left( \frac{\nabla_{\boldsymbol{\theta}} Z(\boldsymbol{\theta})}{Z(\boldsymbol{\theta})} \right) = \frac{\nabla_{\boldsymbol{\theta}}^2 Z(\boldsymbol{\theta})}{Z(\boldsymbol{\theta})} - \frac{\nabla_{\boldsymbol{\theta}} Z(\boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} Z(\boldsymbol{\theta})^\top}{Z(\boldsymbol{\theta})^2}$$
Compute $\nabla_{\boldsymbol{\theta}}^2 Z(\boldsymbol{\theta})$:
$$\nabla_{\boldsymbol{\theta}}^2 Z(\boldsymbol{\theta}) = \int_{\mathbb{R}^D} \nabla_{\boldsymbol{\theta}} \left( -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) e^{-E_{\boldsymbol{\theta}}(\mathbf{x})} \right) d\mathbf{x} = \int_{\mathbb{R}^D} \left( -\nabla_{\boldsymbol{\theta}}^2 E_{\boldsymbol{\theta}}(\mathbf{x}) + \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x})^\top \right) e^{-E_{\boldsymbol{\theta}}(\mathbf{x})} d\mathbf{x}$$
Dividing by $Z(\boldsymbol{\theta})$:
$$\frac{\nabla_{\boldsymbol{\theta}}^2 Z(\boldsymbol{\theta})}{Z(\boldsymbol{\theta})} = -\mathbb{E}_{p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}}^2 E_{\boldsymbol{\theta}}(\mathbf{x}) \right] + \mathbb{E}_{p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x})^\top \right]$$
Since $\frac{\nabla_{\boldsymbol{\theta}} Z(\boldsymbol{\theta})}{Z(\boldsymbol{\theta})} = -\mathbb{E}_{p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x})]$, we substitute:
$$\nabla_{\boldsymbol{\theta}}^2 \log Z(\boldsymbol{\theta}) = \operatorname{Cov}_{\mathbf{x} \sim p_{\boldsymbol{\theta}}} \left( \nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) \right) - \mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}}^2 E_{\boldsymbol{\theta}}(\mathbf{x}) \right]$$
For linear energy models (exponential families where $E_{\boldsymbol{\theta}}(\mathbf{x}) = -\boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x})$), the second derivative $\nabla^2_{\boldsymbol{\theta}} E = \mathbf{0}$, yielding:
$$\nabla_{\boldsymbol{\theta}}^2 \log Z(\boldsymbol{\theta}) = \operatorname{Cov}_{\mathbf{x} \sim p_{\boldsymbol{\theta}}}(\boldsymbol{\phi}(\mathbf{x})) \succeq 0$$
This proves that the log-partition function $\log Z(\boldsymbol{\theta})$ is strictly convex in $\boldsymbol{\theta}$ for exponential families, ensuring a unique global maximum likelihood solution.

**4. Step 3: Derivation of the Stationary Distribution of Langevin Diffusion:**
To sample $\mathbf{x}' \sim p_{\boldsymbol{\theta}}(\mathbf{x})$ for the negative phase, consider the Itô Stochastic Differential Equation (SDE):
$$d\mathbf{x}(t) = -\frac{1}{2} \nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x}(t)) dt + d\mathbf{w}(t)$$
where $\mathbf{w}(t) \in \mathbb{R}^D$ is standard Brownian motion with covariance $\mathbb{E}[d\mathbf{w}(t) d\mathbf{w}(t)^\top] = \mathbf{I} dt$.
Let $q(\mathbf{x}, t)$ be the probability density of the process at time $t$. By the Fokker-Planck (Kolmogorov Forward) Equation:
$$\frac{\partial q(\mathbf{x}, t)}{\partial t} = -\nabla_{\mathbf{x}} \cdot \left( \boldsymbol{\mu}(\mathbf{x}) q(\mathbf{x}, t) \right) + \frac{1}{2} \sum_{i=1}^D \frac{\partial^2}{\partial x_i^2} q(\mathbf{x}, t)$$
where the drift vector is $\boldsymbol{\mu}(\mathbf{x}) = -\frac{1}{2} \nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x})$, and diffusion coefficient is $\mathbf{D} = \mathbf{I}$:
$$\frac{\partial q(\mathbf{x}, t)}{\partial t} = \nabla_{\mathbf{x}} \cdot \left( \frac{1}{2} \nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x}) q(\mathbf{x}, t) + \frac{1}{2} \nabla_{\mathbf{x}} q(\mathbf{x}, t) \right) = \frac{1}{2} \nabla_{\mathbf{x}} \cdot \left[ q(\mathbf{x}, t) \nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x}) + \nabla_{\mathbf{x}} q(\mathbf{x}, t) \right]$$
At stationarity, $\frac{\partial q^*}{\partial t} = 0$. A sufficient condition is zero probability flux:
$$q^*(\mathbf{x}) \nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x}) + \nabla_{\mathbf{x}} q^*(\mathbf{x}) = \mathbf{0} \implies \frac{\nabla_{\mathbf{x}} q^*(\mathbf{x})}{q^*(\mathbf{x})} = -\nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x})$$
$$\nabla_{\mathbf{x}} \log q^*(\mathbf{x}) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x}) \implies \log q^*(\mathbf{x}) = -E_{\boldsymbol{\theta}}(\mathbf{x}) - \log Z \implies q^*(\mathbf{x}) = \frac{e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}}{Z(\boldsymbol{\theta})} = p_{\boldsymbol{\theta}}(\mathbf{x})$$
The stationary distribution of the SDE is exactly the Gibbs density $p_{\boldsymbol{\theta}}(\mathbf{x})$!

**5. Step 4: Discretization and Unadjusted Langevin Algorithm (ULA):**
Applying the Euler-Maruyama discretization with step size $\epsilon > 0$:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\epsilon}{2} \nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}(\mathbf{x}_k) + \sqrt{\epsilon} \boldsymbol{\xi}_k, \quad \boldsymbol{\xi}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
As $k \to \infty$ and $\epsilon \to 0$, the distribution of $\mathbf{x}_k$ converges in Wasserstein distance to $p_{\boldsymbol{\theta}}(\mathbf{x})$. In Contrastive Divergence (CD-$k$), running $k$ steps initialized from data $\mathbf{x}^{(0)} \sim p_{\text{data}}$ yields the negative phase sample $\mathbf{x}^- \approx \mathbf{x}_k$.

---

## 3. Geometric & Algebraic Interpretation

### Push-Forward Measure & Diffeomorphic Warping

Let $(\mathcal{Z}, \Sigma_z, \mu_z)$ be a probability space with Gaussian measure.
A generative network $\mathbf{x} = G_{\boldsymbol{\theta}}(\mathbf{z})$ defines a **push-forward measure** $G_\# \mu_z$:
$$(G_\# \mu_z)(A) = \mu_z\left( G^{-1}(A) \right) \quad \text{for any measurable set } A \subset \mathcal{X}$$

```
                PUSH-FORWARD MEASURE (GAUSSIAN TO MANIFOLD)
        Latent Space Z (Gaussian)              Data Space X (Complex Manifold)
                 z_2                                      x_2
                  ▲                                        ▲
              ┌───┼───┐                                    │       * (S-shaped
             ┌┘   │   └┐                                   │     *    manifold)
            ─┼────┼────┼─► z_1    ───► G_theta ───►        │   *
             └┐   │   ┌┘                                   │ *
              └───┼───┘                                    └──────────────► x_1
             p_z ~ N(0, I)                                  p_data
```

The generator acts as a non-linear geometric deformation that stretches, compresses, and folds the Euclidean Gaussian sphere into the intricate topological geometry of the data manifold.

---

## 4. Real-World Analogy

### The Five Art Studios
1. **Autoregressive (The Mosaic Tile Worker):** Lays one tiny colored tile at a time from top-left to bottom-right. Each tile depends on all previous tiles. The final mosaic is exquisite, but placing 1,000,000 tiles takes a very long time.
2. **Normalizing Flows (The Glassblower):** Starts with a uniform sphere of molten glass. Without tearing or puncturing the glass, they smoothly bend and stretch it into a delicate vase. Every point on the vase can be traced back invertibly to the original sphere.
3. **VAEs (The Forensic Sketch Artist):** Listens to a witness describe a crime suspect, compresses the description into a brief list of traits (latent code $\mathbf{z}$), and draws a reconstruction. Because the summary discards fine micro-details, the resulting drawing looks slightly soft and smoothed out.
4. **GANs (The Art Forger & The Detective):** A forger manufactures fake Rembrandts; a museum curator inspects them and catches the fakes. Through fierce competition, the forger becomes so brilliant that their paintings fool the curator completely, though the forger may choose to only forge one specific painting they are comfortable with (mode collapse).
5. **Diffusion Models (The Sculptor in Reverse):** Starts with a pure block of noisy television static. Using a gentle chisel, they shave away a microscopic grain of noise at step 1, another at step 2, repeating for 50 passes until a breathtaking sculpture emerges from the static.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete 1D Change of Variables and probability density transformation with exact hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Density Transformation |
| :--- | :--- | :--- | :--- |
| $z$ | Base Latent Variable | Scalar $\in [0, 2]$ | Uniform base variable $z \sim \mathcal{U}[0, 2]$ |
| $p_z(z)$ | Base Probability Density | Scalar ($0.5$) | $p_z(z) = \frac{1}{2 - 0} = 0.5$ |
| $x = f(z)$ | Forward Mapping | Scalar $\in [0, 4]$ | Non-linear quadratic expansion $f(z) = z^2$ |
| $z = f^{-1}(x)$ | Inverse Mapping | Scalar | $f^{-1}(x) = \sqrt{x}$ |
| $\frac{d f^{-1}(x)}{dx}$ | Inverse Jacobian Derivative | Scalar | Rate of volume change $\frac{1}{2\sqrt{x}}$ |
| $p_x(x)$ | Transformed Probability Density | Scalar | $p_z(f^{-1}(x)) \cdot \left\vert \frac{d f^{-1}}{dx} \right\vert = \frac{1}{4\sqrt{x}}$ |

---

### 5.2 Concrete Toy Transformation

Let $z \sim \mathcal{U}[0, 2] \implies p_z(z) = 0.5$ for $z \in [0, 2]$.
Let $x = f(z) = z^2$.
As $z$ ranges from $0 \to 2$, $x$ ranges from $0^2 \to 2^2 = [0, 4]$.

---

### 5.3 Step 1: Analytical Derivation of Transformed Density $p_x(x)$

1. **Inverse Function:**
   $$x = z^2 \implies z = \sqrt{x}$$
2. **Derivative of Inverse Function:**
   $$\frac{d f^{-1}(x)}{dx} = \frac{d}{dx} (\sqrt{x}) = \frac{1}{2\sqrt{x}}$$
3. **Change of Variables Formula:**
   $$p_x(x) = p_z(\sqrt{x}) \cdot \left| \frac{d f^{-1}(x)}{dx} \right| = 0.5 \cdot \frac{1}{2\sqrt{x}} = \mathbf{\frac{1}{4\sqrt{x}}}, \quad \forall x \in (0, 4]$$

---

### 5.4 Step 2: Cell-by-Cell Density Evaluation

Let us evaluate $p_x(x)$ at three concrete points:

1. **Point 1 ($x = 0.25$):**
   $$\sqrt{0.25} = 0.5$$
   $$p_x(0.25) = \frac{1}{4(0.5)} = \frac{1}{2.0} = \mathbf{0.5000}$$

2. **Point 2 ($x = 1.00$):**
   $$\sqrt{1.0} = 1.0$$
   $$p_x(1.0) = \frac{1}{4(1.0)} = \frac{1}{4.0} = \mathbf{0.2500}$$

3. **Point 3 ($x = 4.00$):**
   $$\sqrt{4.0} = 2.0$$
   $$p_x(4.0) = \frac{1}{4(2.0)} = \frac{1}{8.0} = \mathbf{0.1250}$$

Notice the density behavior: As $x$ increases, the non-linear transformation $z^2$ stretches space out, diluting the probability mass and causing density $p_x(x)$ to drop from $0.50 \to 0.25 \to 0.125$!

---

### 5.5 Step 3: Exact Proof of Probability Conservation

Verify that the total probability integrates to exactly $1.0$:
$$\int_0^4 p_x(x) dx = \int_0^4 \frac{1}{4} x^{-1/2} dx = \left[ \frac{1}{4} \cdot \frac{x^{1/2}}{1/2} \right]_0^4 = \left[ \frac{1}{2} \sqrt{x} \right]_0^4$$
$$= \frac{1}{2} \sqrt{4} - \frac{1}{2} \sqrt{0} = \frac{1}{2}(2) - 0 = \mathbf{1.0000}$$

The probability mass is perfectly conserved. This is the foundational mathematical mechanism underlying all Normalizing Flows!

---

## 6. Solved Illustrations

### Illustration 1: Mode-Covering vs. Mode-Collapsing on Bimodal Target

**Problem:**
Let the true data distribution be an equal mixture of two distant 1D Gaussians:
$$p_{\text{data}}(x) = 0.5 \mathcal{N}(-5, 1) + 0.5 \mathcal{N}(+5, 1)$$
Suppose we fit a single unimodal Gaussian model $q(x) = \mathcal{N}(\mu, \sigma^2)$.
Predict the optimal parameters $(\mu, \sigma^2)$ under:
1. Forward KL: $\min_q D_{\text{KL}}(p_{\text{data}} \,\|\, q)$ (Maximum Likelihood / VAE).
2. Reverse KL: $\min_q D_{\text{KL}}(q \,\|\, p_{\text{data}})$ (GAN objective).

**Solution:**
1. **Forward KL ($D_{\text{KL}}(p \,\|\, q)$ - Zero-Avoiding):**
   The model cannot allow $q(x) \approx 0$ anywhere that $p(x) > 0$.
   To cover both modes at $-5$ and $+5$:
   $$\mu^* = \mathbb{E}_{p_{\text{data}}}[x] = 0.5(-5) + 0.5(5) = \mathbf{0.0}$$
   $$\sigma^{2*} = \operatorname{Var}_{p_{\text{data}}}(x) = \mathbb{E}[x^2] - \mu^2 = \left( 0.5(25 + 1) + 0.5(25 + 1) \right) - 0 = \mathbf{26.0}$$
   The model expands its variance to $\sigma = \sqrt{26} \approx 5.1$, placing its mean at $0$ in the empty dead space between modes. It covers both modes, but generates poor samples in the middle!
2. **Reverse KL ($D_{\text{KL}}(q \,\|\, p)$ - Zero-Forcing):**
   If $q(x) > 0$ where $p(x) \approx 0$, the penalty explodes.
   The model collapses entirely onto **one** of the two modes:
   $$\mu^* = -5.0 \quad (\text{or } +5.0), \quad \sigma^{2*} = 1.0$$
   The model generates pristine samples for mode 1, but completely forgets that mode 2 exists (Mode Collapse).

---

### Illustration 2: Comprehensive Generative Architecture Comparison Matrix

| Property | Autoregressive (GPT) | Normalizing Flows (Glow) | VAEs (Kingma 2013) | GANs (Goodfellow 2014) | Diffusion (DDPM 2020) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Density Evaluation** | **Exact** $p(\mathbf{x})$ | **Exact** $p(\mathbf{x})$ | Approximate ($\text{ELBO}$) | None (Implicit) | Approximate ($\text{ELBO}$) |
| **Sampling Speed** | Slow ($\mathcal{O}(T)$ steps) | **Fast** ($1$ step) | **Fast** ($1$ step) | **Fast** ($1$ step) | Moderate ($10-50$ steps) |
| **Sample Quality** | State-of-the-Art | Moderate | Blurry | **Exceptional** | **State-of-the-Art** |
| **Mode Coverage** | Complete | Complete | Complete | **Poor (Mode Collapse)** | Complete |
| **Training Stability** | Extremely Stable | Stable | Stable | **Highly Unstable** | Stable |

---

### Illustration 3: Complete Step-by-Step Numerical RealNVP Forward and Inverse Density Calculation

**Problem:**
Consider a RealNVP affine coupling layer operating on a $D=4$ dimensional continuous space partitioned at $d=2$ into $\mathbf{z}_{1:2}$ and $\mathbf{z}_{3:4}$.
The base prior distribution is a standard 4D isotropic Gaussian:
$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_4) \implies p_{\mathbf{z}}(\mathbf{z}) = (2\pi)^{-2} \exp\left( -\frac{1}{2} \|\mathbf{z}\|^2 \right)$$
Given the input latent vector:
$$\mathbf{z} = \begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ z_4 \end{bmatrix} = \begin{bmatrix} 0.50 \\ -1.00 \\ 1.50 \\ -0.50 \end{bmatrix} \implies \mathbf{z}_{1:2} = \begin{bmatrix} 0.50 \\ -1.00 \end{bmatrix}, \quad \mathbf{z}_{3:4} = \begin{bmatrix} 1.50 \\ -0.50 \end{bmatrix}$$
The scaling and translation networks $s, t: \mathbb{R}^2 \to \mathbb{R}^2$ are defined as linear layers:
$$s(\mathbf{z}_{1:2}) = \mathbf{W}_s \mathbf{z}_{1:2} + \mathbf{b}_s, \quad t(\mathbf{z}_{1:2}) = \mathbf{W}_t \mathbf{z}_{1:2} + \mathbf{b}_t$$
with parameters:
$$\mathbf{W}_s = \begin{bmatrix} 0.20 & -0.40 \\ 0.10 & 0.30 \end{bmatrix}, \quad \mathbf{b}_s = \begin{bmatrix} 0.10 \\ -0.20 \end{bmatrix}, \qquad \mathbf{W}_t = \begin{bmatrix} 0.50 & 0.20 \\ -0.30 & 0.40 \end{bmatrix}, \quad \mathbf{b}_t = \begin{bmatrix} 0.00 \\ 0.50 \end{bmatrix}$$
1. Compute scale vector $s(\mathbf{z}_{1:2})$, scaling factors $\exp(s)$, and translation vector $t(\mathbf{z}_{1:2})$.
2. Compute transformed output vector $\mathbf{x} \in \mathbb{R}^4$.
3. Compute the full $4 \times 4$ Jacobian matrix and its log-determinant $\log |\det \mathbf{J}|$.
4. Compute base log-density $\log p_{\mathbf{z}}(\mathbf{z})$ and the exact push-forward log-density $\log p_{\mathbf{x}}(\mathbf{x})$.
5. Invert the transformation from $\mathbf{x}$ to algebraically recover $\mathbf{z}$ and verify exact reconstruction.

**Step-by-Step Solution:**

**1. Forward Evaluation of Scale and Translation:**
$$\mathbf{s} = \begin{bmatrix} s_1 \\ s_2 \end{bmatrix} = \begin{bmatrix} 0.20(0.50) + (-0.40)(-1.00) + 0.10 \\ 0.10(0.50) + 0.30(-1.00) + (-0.20) \end{bmatrix} = \begin{bmatrix} 0.10 + 0.40 + 0.10 \\ 0.05 - 0.30 - 0.20 \end{bmatrix} = \begin{bmatrix} \mathbf{0.60} \\ \mathbf{-0.45} \end{bmatrix}$$
$$\exp(\mathbf{s}) = \begin{bmatrix} e^{0.60} \\ e^{-0.45} \end{bmatrix} = \begin{bmatrix} \mathbf{1.822119} \\ \mathbf{0.637628} \end{bmatrix}$$
$$\mathbf{t} = \begin{bmatrix} t_1 \\ t_2 \end{bmatrix} = \begin{bmatrix} 0.50(0.50) + 0.20(-1.00) + 0.00 \\ -0.30(0.50) + 0.40(-1.00) + 0.50 \end{bmatrix} = \begin{bmatrix} 0.25 - 0.20 + 0.00 \\ -0.15 - 0.40 + 0.50 \end{bmatrix} = \begin{bmatrix} \mathbf{0.05} \\ \mathbf{-0.05} \end{bmatrix}$$

**2. Transformed Output Vector $\mathbf{x}$:**
$$\mathbf{x}_{1:2} = \mathbf{z}_{1:2} = \begin{bmatrix} \mathbf{0.50} \\ \mathbf{-1.00} \end{bmatrix}$$
$$\mathbf{x}_{3:4} = \mathbf{z}_{3:4} \odot \exp(\mathbf{s}) + \mathbf{t} = \begin{bmatrix} 1.50 \times 1.822119 + 0.05 \\ -0.50 \times 0.637628 + (-0.05) \end{bmatrix} = \begin{bmatrix} 2.733178 + 0.05 \\ -0.318814 - 0.05 \end{bmatrix} = \begin{bmatrix} \mathbf{2.783178} \\ \mathbf{-0.368814} \end{bmatrix}$$
Full transformed vector:
$$\mathbf{x} = \begin{bmatrix} 0.500000 \\ -1.000000 \\ 2.783178 \\ -0.368814 \end{bmatrix}$$

**3. Jacobian Matrix and Log-Determinant:**
The Jacobian matrix $\mathbf{J} = \frac{\partial \mathbf{x}}{\partial \mathbf{z}}$ has the block lower-triangular structure:
$$\mathbf{J} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
\frac{\partial x_3}{\partial z_1} & \frac{\partial x_3}{\partial z_2} & e^{s_1} & 0 \\
\frac{\partial x_4}{\partial z_1} & \frac{\partial x_4}{\partial z_2} & 0 & e^{s_2}
\end{bmatrix} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
M_{11} & M_{12} & 1.822119 & 0 \\
M_{21} & M_{22} & 0 & 0.637628
\end{bmatrix}$$
Because $\mathbf{J}$ is block triangular, its determinant depends only on the diagonal blocks:
$$\det \mathbf{J} = 1 \times 1 \times e^{s_1} \times e^{s_2} = e^{s_1 + s_2} = e^{0.60 - 0.45} = e^{0.15} \approx 1.161834$$
$$\log |\det \mathbf{J}| = s_1 + s_2 = 0.60 + (-0.45) = \mathbf{+0.150000}$$

**4. Push-Forward Exact Log-Likelihood:**
Calculate latent squared norm $\|\mathbf{z}\|^2$:
$$\|\mathbf{z}\|^2 = (0.50)^2 + (-1.00)^2 + (1.50)^2 + (-0.50)^2 = 0.25 + 1.00 + 2.25 + 0.25 = \mathbf{3.750000}$$
Latent prior log-density:
$$\log p_{\mathbf{z}}(\mathbf{z}) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\|\mathbf{z}\|^2 = -2 \log(2\pi) - \frac{1}{2}(3.75) = -2(1.837877) - 1.875 = -3.675754 - 1.875 = \mathbf{-5.550754}$$
Exact push-forward log-density:
$$\log p_{\mathbf{x}}(\mathbf{x}) = \log p_{\mathbf{z}}(\mathbf{z}) - \log |\det \mathbf{J}| = -5.550754 - 0.150000 = \mathbf{-5.700754}$$

**5. Verification of Analytical Inversion:**
Given $\mathbf{x} = [0.50, -1.00, 2.783178, -0.368814]^\top$:
1. $\mathbf{z}_{1:2} = \mathbf{x}_{1:2} = [0.50, -1.00]^\top$.
2. Compute $\mathbf{s} = s(\mathbf{z}_{1:2}) = [0.60, -0.45]^\top$ and $\mathbf{t} = t(\mathbf{z}_{1:2}) = [0.05, -0.05]^\top$.
3. Compute $\mathbf{z}_{3:4} = (\mathbf{x}_{3:4} - \mathbf{t}) \odot \exp(-\mathbf{s})$:
   $$z_3 = (2.783178 - 0.05) \times e^{-0.60} = 2.733178 \times 0.548812 = \mathbf{1.500000}$$
   $$z_4 = (-0.368814 - (-0.05)) \times e^{+0.45} = -0.318814 \times 1.568312 = \mathbf{-0.500000}$$
Exact recovery $\mathbf{z} = [0.50, -1.00, 1.50, -0.50]^\top$ verified to 6 decimal places.

---

### Illustration 4: Contrastive Divergence (CD-1) Parameter Update Trace for an Energy-Based Model

**Problem:**
Consider a 2D continuous Energy-Based Model with bilinear quadratic energy function:
$$E_{\mathbf{w}}(\mathbf{x}) = \frac{1}{2} w_1 x_1^2 + \frac{1}{2} w_2 x_2^2 + w_{12} x_1 x_2$$
parameterized by weight vector $\mathbf{w} = [w_1, w_2, w_{12}]^\top$.
At iteration step $t$, the current parameter values are:
$$\mathbf{w}^{(t)} = \begin{bmatrix} w_1 \\ w_2 \\ w_{12} \end{bmatrix} = \begin{bmatrix} 1.00 \\ 1.00 \\ 0.00 \end{bmatrix}$$
We observe a single empirical data point (positive sample):
$$\mathbf{x}^+ = \begin{bmatrix} 2.00 \\ 1.00 \end{bmatrix}$$
To approximate the intractable negative phase gradient, we execute one step of Unadjusted Langevin Algorithm (CD-1) initialized at $\mathbf{x}^{(0)} = \mathbf{x}^+$, with step size $\epsilon = 0.10$ and injected Gaussian noise $\boldsymbol{\xi} = [0.40, -0.60]^\top$.
1. Compute the data gradient $\nabla_{\mathbf{x}} E_{\mathbf{w}}(\mathbf{x}^{(0)})$.
2. Execute the Langevin transition step to obtain negative hallucinated sample $\mathbf{x}^- = \mathbf{x}^{(1)}$.
3. Compute the positive phase parameter gradient $\nabla_{\mathbf{w}} E_{\mathbf{w}}(\mathbf{x}^+)$ and negative phase parameter gradient $\nabla_{\mathbf{w}} E_{\mathbf{w}}(\mathbf{x}^-)$.
4. Compute the Contrastive Divergence gradient $\mathbf{g}_{\text{CD}} = \nabla_{\mathbf{w}} E_{\mathbf{w}}(\mathbf{x}^+) - \nabla_{\mathbf{w}} E_{\mathbf{w}}(\mathbf{x}^-)$.
5. Update $\mathbf{w}$ using gradient ascent on log-likelihood with learning rate $\eta = 0.10$ and interpret the parameter trajectory.

**Step-by-Step Solution:**

**1. Data Gradient with Respect to Coordinate Vector $\mathbf{x}$:**
$$\nabla_{\mathbf{x}} E_{\mathbf{w}}(\mathbf{x}) = \begin{bmatrix} \frac{\partial E}{\partial x_1} \\ \frac{\partial E}{\partial x_2} \end{bmatrix} = \begin{bmatrix} w_1 x_1 + w_{12} x_2 \\ w_2 x_2 + w_{12} x_1 \end{bmatrix}$$
At $\mathbf{x}^{(0)} = [2.00, 1.00]^\top$ with $\mathbf{w} = [1.00, 1.00, 0.00]^\top$:
$$\nabla_{\mathbf{x}} E_{\mathbf{w}}(\mathbf{x}^{(0)}) = \begin{bmatrix} 1.00(2.00) + 0.00(1.00) \\ 1.00(1.00) + 0.00(2.00) \end{bmatrix} = \begin{bmatrix} \mathbf{2.00} \\ \mathbf{1.00} \end{bmatrix}$$

**2. Langevin Diffusion Step (Negative Sample Generation):**
The discrete Langevin transition is:
$$\mathbf{x}^{(1)} = \mathbf{x}^{(0)} - \frac{\epsilon}{2} \nabla_{\mathbf{x}} E_{\mathbf{w}}(\mathbf{x}^{(0)}) + \sqrt{\epsilon} \boldsymbol{\xi}$$
With $\epsilon = 0.10 \implies \frac{\epsilon}{2} = 0.05$ and $\sqrt{\epsilon} = \sqrt{0.10} \approx 0.316228$:
$$\text{Drift term: } -\frac{\epsilon}{2} \nabla_{\mathbf{x}} E = -0.05 \begin{bmatrix} 2.00 \\ 1.00 \end{bmatrix} = \begin{bmatrix} -0.1000 \\ -0.0500 \end{bmatrix}$$
$$\text{Diffusion term: } \sqrt{\epsilon} \boldsymbol{\xi} = 0.316228 \begin{bmatrix} 0.40 \\ -0.60 \end{bmatrix} = \begin{bmatrix} +0.126491 \\ -0.189737 \end{bmatrix}$$
Summing the terms:
$$\mathbf{x}^- = \mathbf{x}^{(1)} = \begin{bmatrix} 2.00 - 0.1000 + 0.126491 \\ 1.00 - 0.0500 - 0.189737 \end{bmatrix} = \begin{bmatrix} \mathbf{2.026491} \\ \mathbf{0.760263} \end{bmatrix}$$

**3. Parameter Gradients of Energy Function:**
$$\nabla_{\mathbf{w}} E_{\mathbf{w}}(\mathbf{x}) = \begin{bmatrix} \frac{\partial E}{\partial w_1} \\ \frac{\partial E}{\partial w_2} \\ \frac{\partial E}{\partial w_{12}} \end{bmatrix} = \begin{bmatrix} \frac{1}{2} x_1^2 \\ \frac{1}{2} x_2^2 \\ x_1 x_2 \end{bmatrix}$$
- **Positive Phase Gradient ($\mathbf{x}^+ = [2.00, 1.00]^\top$):**
  $$\nabla_{\mathbf{w}} E(\mathbf{x}^+) = \begin{bmatrix} \frac{1}{2}(2.00)^2 \\ \frac{1}{2}(1.00)^2 \\ (2.00)(1.00) \end{bmatrix} = \begin{bmatrix} \frac{1}{2}(4.00) \\ \frac{1}{2}(1.00) \\ 2.00 \end{bmatrix} = \begin{bmatrix} \mathbf{2.0000} \\ \mathbf{0.5000} \\ \mathbf{2.0000} \end{bmatrix}$$
- **Negative Phase Gradient ($\mathbf{x}^- = [2.026491, 0.760263]^\top$):**
  $$\nabla_{\mathbf{w}} E(\mathbf{x}^-) = \begin{bmatrix} \frac{1}{2}(2.026491)^2 \\ \frac{1}{2}(0.760263)^2 \\ (2.026491)(0.760263) \end{bmatrix} = \begin{bmatrix} \frac{1}{2}(4.106666) \\ \frac{1}{2}(0.578000) \\ 1.540666 \end{bmatrix} = \begin{bmatrix} \mathbf{2.053333} \\ \mathbf{0.289000} \\ \mathbf{1.540666} \end{bmatrix}$$

**4. Contrastive Divergence Gradient:**
The gradient of negative log-likelihood under CD-1 is:
$$\mathbf{g}_{\text{CD}} = \nabla_{\mathbf{w}} E(\mathbf{x}^+) - \nabla_{\mathbf{w}} E(\mathbf{x}^-) = \begin{bmatrix} 2.0000 - 2.053333 \\ 0.5000 - 0.289000 \\ 2.0000 - 1.540666 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.053333} \\ \mathbf{+0.211000} \\ \mathbf{+0.459334} \end{bmatrix}$$

**5. Parameter Update & Intuition:**
To maximize log-likelihood (minimizing negative log-likelihood), perform gradient descent on CD loss:
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \, \mathbf{g}_{\text{CD}}$$
With learning rate $\eta = 0.10$:
$$w_1^{(t+1)} = 1.00 - 0.10(-0.053333) = 1.00 + 0.005333 = \mathbf{1.005333}$$
$$w_2^{(t+1)} = 1.00 - 0.10(+0.211000) = 1.00 - 0.021100 = \mathbf{0.978900}$$
$$w_{12}^{(t+1)} = 0.00 - 0.10(+0.459334) = 0.00 - 0.045933 = \mathbf{-0.045933}$$

**Physical Interpretation:**
Because $x_1 x_2 = 2.0$ at the real data point while the hallucinated sample had a lower product ($1.54$), the model lowers $w_{12}$ to $-0.0459$, making the bilinear term $w_{12} x_1 x_2$ more negative. This lowers the energy at correlated data points, effectively pulling down the energy well around real data!

---

### Illustration 5: Quantitative Comparison of Divergences on Disjoint Supports: Forward KL, Reverse KL, JSD, and Wasserstein-1

**Problem:**
Consider a 1D discrete domain with $K=3$ states $\{1, 2, 3\}$.
The true data distribution is bimodal with equal mass on the outer boundaries and lower mass in the center:
$$\mathbf{p} = \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix} = \begin{bmatrix} 0.40 \\ 0.20 \\ 0.40 \end{bmatrix}$$
Evaluate three candidate generative models:
1. **Model A (Mode-Covering / Uniform Spread):** $\mathbf{q}_A = [0.3333, 0.3333, 0.3334]^\top$
2. **Model B (Mode-Collapsed onto State 1):** $\mathbf{q}_B = [0.90, 0.05, 0.05]^\top$
3. **Model C (Mode-Dropped Center Valley):** $\mathbf{q}_C = [0.50, 0.00, 0.50]^\top$

Compute the following metrics for all three models:
1. Forward KL divergence: $D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{q})$.
2. Reverse KL divergence: $D_{\mathrm{KL}}(\mathbf{q} \,\|\, \mathbf{p})$.
3. Jensen-Shannon Divergence: $\mathrm{JSD}(\mathbf{p} \,\|\, \mathbf{q}) = \frac{1}{2} D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{m}) + \frac{1}{2} D_{\mathrm{KL}}(\mathbf{q} \,\|\, \mathbf{m})$, where $\mathbf{m} = \frac{1}{2}(\mathbf{p} + \mathbf{q})$.
4. 1D Wasserstein-1 Earth Mover's Distance: $W_1(\mathbf{p}, \mathbf{q}) = \sum_{k=1}^{K-1} |P_k - Q_k|$, where $P_k = \sum_{i=1}^k p_i$ and $Q_k = \sum_{i=1}^k q_i$.
Demonstrate why GAN training crashes under KL divergences when supports do not overlap, whereas Wasserstein distance remains well-behaved.

**Step-by-Step Solution:**

**1. Forward KL Divergence $D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{q}) = \sum_{i=1}^3 p_i \log \frac{p_i}{q_i}$:**
- **Model A ($\mathbf{q}_A$):**
  $$D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{q}_A) = 0.40 \log\left(\frac{0.40}{0.3333}\right) + 0.20 \log\left(\frac{0.20}{0.3333}\right) + 0.40 \log\left(\frac{0.40}{0.3334}\right)$$
  $$= 0.40 \log(1.2001) + 0.20 \log(0.6001) + 0.40 \log(1.1998)$$
  $$= 0.40(0.1824) + 0.20(-0.5107) + 0.40(0.1821) = 0.07296 - 0.10214 + 0.07284 = \mathbf{0.04366} \text{ nats}$$
- **Model B ($\mathbf{q}_B$):**
  $$D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{q}_B) = 0.40 \log\left(\frac{0.40}{0.90}\right) + 0.20 \log\left(\frac{0.20}{0.05}\right) + 0.40 \log\left(\frac{0.40}{0.05}\right)$$
  $$= 0.40 \log(0.4444) + 0.20 \log(4.000) + 0.40 \log(8.000)$$
  $$= 0.40(-0.8109) + 0.20(1.3863) + 0.40(2.0794) = -0.3244 + 0.2773 + 0.8318 = \mathbf{0.7847} \text{ nats}$$
- **Model C ($\mathbf{q}_C$):**
  At state $i=2$, $p_2 = 0.20 > 0$ but $q_{C, 2} = 0.00$.
  $$p_2 \log\left( \frac{p_2}{q_{C, 2}} \right) = 0.20 \log\left( \frac{0.20}{0.00} \right) = \mathbf{+\infty}$$
  Forward KL produces an infinite penalty because the model failed to cover state 2!

**2. Reverse KL Divergence $D_{\mathrm{KL}}(\mathbf{q} \,\|\, \mathbf{p}) = \sum_{i=1}^3 q_i \log \frac{q_i}{p_i}$:**
- **Model A ($\mathbf{q}_A$):**
  $$D_{\mathrm{KL}}(\mathbf{q}_A \,\|\, \mathbf{p}) = 0.3333 \log\left(\frac{0.3333}{0.40}\right) + 0.3333 \log\left(\frac{0.3333}{0.20}\right) + 0.3334 \log\left(\frac{0.3334}{0.40}\right)$$
  $$= 0.3333(-0.1824) + 0.3333(+0.5107) + 0.3334(-0.1821) = -0.0608 + 0.1702 - 0.0607 = \mathbf{0.0487} \text{ nats}$$
- **Model B ($\mathbf{q}_B$):**
  $$D_{\mathrm{KL}}(\mathbf{q}_B \,\|\, \mathbf{p}) = 0.90 \log\left(\frac{0.90}{0.40}\right) + 0.05 \log\left(\frac{0.05}{0.20}\right) + 0.05 \log\left(\frac{0.05}{0.40}\right)$$
  $$= 0.90(0.8109) + 0.05(-1.3863) + 0.05(-2.0794) = 0.7298 - 0.0693 - 0.1040 = \mathbf{0.5565} \text{ nats}$$
- **Model C ($\mathbf{q}_C$):**
  Notice that where $q_{C, 2} = 0$, $0 \log(0 / 0.20) = 0$ by continuity ($\lim_{u \to 0^+} u \log u = 0$):
  $$D_{\mathrm{KL}}(\mathbf{q}_C \,\|\, \mathbf{p}) = 0.50 \log\left(\frac{0.50}{0.40}\right) + 0.00 + 0.50 \log\left(\frac{0.50}{0.40}\right) = 2 \times 0.50 \log(1.25) = \log(1.25) = \mathbf{0.2231} \text{ nats}$$
  Unlike Forward KL which was $+\infty$, Reverse KL is completely finite ($0.2231$ nats)! The model happily drops mode 2 without severe penalty.

**3. Jensen-Shannon Divergence $\mathrm{JSD}(\mathbf{p} \,\|\, \mathbf{q})$:**
For Model C ($\mathbf{q}_C = [0.5, 0.0, 0.5]^\top$):
$$\mathbf{m} = \frac{1}{2}(\mathbf{p} + \mathbf{q}_C) = \begin{bmatrix} 0.45 \\ 0.10 \\ 0.45 \end{bmatrix}$$
$$D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{m}) = 0.40 \log\left(\frac{0.40}{0.45}\right) + 0.20 \log\left(\frac{0.20}{0.10}\right) + 0.40 \log\left(\frac{0.40}{0.45}\right)$$
$$= 2(0.40)(-0.1178) + 0.20(0.6931) = -0.09424 + 0.13862 = 0.04438$$
$$D_{\mathrm{KL}}(\mathbf{q}_C \,\|\, \mathbf{m}) = 0.50 \log\left(\frac{0.50}{0.45}\right) + 0 + 0.50 \log\left(\frac{0.50}{0.45}\right) = \log(1.1111) = 0.10536$$
$$\mathrm{JSD}(\mathbf{p} \,\|\, \mathbf{q}_C) = \frac{1}{2}(0.04438 + 0.10536) = \mathbf{0.07487} \text{ nats}$$
JSD is symmetric and bounded: $\mathrm{JSD} \le \log 2 \approx 0.6931$.

**4. 1D Wasserstein-1 Earth Mover's Distance:**
Cumulative Distribution Functions (CDFs):
$$P_1 = p_1 = 0.40, \quad P_2 = p_1 + p_2 = 0.60, \quad P_3 = 1.00$$
- **Model A ($\mathbf{q}_A = [1/3, 1/3, 1/3]^\top$):**
  $$Q_{A, 1} = 0.3333, \quad Q_{A, 2} = 0.6667, \quad Q_{A, 3} = 1.000$$
  $$W_1(\mathbf{p}, \mathbf{q}_A) = |0.40 - 0.3333| + |0.60 - 0.6667| = 0.0667 + 0.0667 = \mathbf{0.1334}$$
- **Model B ($\mathbf{q}_B = [0.90, 0.05, 0.05]^\top$):**
  $$Q_{B, 1} = 0.90, \quad Q_{B, 2} = 0.95, \quad Q_{B, 3} = 1.00$$
  $$W_1(\mathbf{p}, \mathbf{q}_B) = |0.40 - 0.90| + |0.60 - 0.95| = 0.50 + 0.35 = \mathbf{0.8500}$$
- **Model C ($\mathbf{q}_C = [0.50, 0.00, 0.50]^\top$):**
  $$Q_{C, 1} = 0.50, \quad Q_{C, 2} = 0.50, \quad Q_{C, 3} = 1.00$$
  $$W_1(\mathbf{p}, \mathbf{q}_C) = |0.40 - 0.50| + |0.60 - 0.50| = 0.10 + 0.10 = \mathbf{0.2000}$$

**Summary Comparison Matrix:**

| Candidate Model | Forward KL ($D_{\mathrm{KL}}(p \| q)$) | Reverse KL ($D_{\mathrm{KL}}(q \| p)$) | JSD | Wasserstein-1 ($W_1$) |
| :--- | :--- | :--- | :--- | :--- |
| **Model A (Uniform Spread)** | **0.0437** (Lowest) | **0.0487** (Lowest) | **0.0112** (Lowest) | **0.1334** (Lowest) |
| **Model B (Mode Collapsed)** | 0.7847 | 0.5565 | 0.2415 | 0.8500 (Highest) |
| **Model C (Mode Dropped)** | **$+\infty$ (EXPLODES)** | 0.2231 (Tolerated!) | 0.0749 | 0.2000 (Smooth) |

**Mathematical Insight:**
Notice that Model C drops state 2 completely. Under Forward KL, the loss is $+\infty$ (strictly forbidding zero support). Under Reverse KL, the penalty is small ($0.2231$), mathematically illustrating why GANs suffer mode collapse: the generator faces minimal penalty for omitting modes entirely.
Finally, Wasserstein distance ($W_1$) is continuous, linear with metric displacement, and never explodes or saturates to a constant gradient, providing the mathematical foundation for stable Wasserstein GAN (WGAN) optimization.

---

## 7. Deep Learning Connection & Application

### Where Each Architecture Dominates in Production AI (2024–2026)
1. **Diffusion Models:** Dominate **Visual & Continuous Media Generation**:
   - Image Synthesis: Stable Diffusion 3, Midjourney v6, FLUX.1.
   - Video Generation: OpenAI Sora, Google Veo, Runway Gen-3.
2. **Autoregressive Models:** Dominate **Discrete Symbolic Reasoning**:
   - Large Language Models: GPT-4, LLaMA 3, Claude 3.5 Sonnet.
   - Code & Mathematical Proof Generation.
3. **Variational Autoencoders (VAEs):** Used ubiquitously as **Perceptual Compression Backbones**:
   - Stable Diffusion does not run diffusion in pixel space ($512 \times 512 \times 3$); it first uses an SD-VAE to compress the image $8\times$ into a compact latent space ($64 \times 64 \times 4$), where diffusion executes with $64\times$ less compute!
4. **GANs:** Dominate **Real-Time Edge Super-Resolution & Style Transfer**:
   - Real-ESRGAN, video game texture upscaling, and low-latency facial animation.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Analytical and numerical verification of the Part 5 Change of Variables 1D transformation ($z \sim \mathcal{U}[0, 2] \to x = z^2 \implies p_x(x) = \frac{1}{4\sqrt{x}}$).
2. Numerical integration of $p_x(x)$ confirming probability conservation to $< 10^{-12}$.
3. Forward KL vs. Reverse KL optimization on a bimodal target distribution (proving mode-covering vs. mode-dropping).
4. Generation of empirical histogram samples matching the analytical density formula.

See implementation in:
[`10_generative_models/code/01_taxonomy_of_generative_models.py`](./code/01_taxonomy_of_generative_models.py)
