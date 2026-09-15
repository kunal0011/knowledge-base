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
| $p_x(x)$ | Transformed Probability Density | Scalar | $p_z(f^{-1}(x)) \cdot \left| \frac{d f^{-1}}{dx} \right| = \frac{1}{4\sqrt{x}}$ |

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
