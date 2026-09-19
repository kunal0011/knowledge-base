# Diffusion Models: DDPM, DDIM, Score-Based SDEs & Classifier-Free Guidance

> **Canonical Literature & Reference Foundations:**
> - *Deep Learning: Foundations and Concepts* (Christopher M. Bishop & Hugh Bishop, Springer 2024, Ch. 19 "Generative Models")
> - *Deep Generative Models* (Stefano Ermon & Aditya Grover, Stanford CS236 Monograph)
> - *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (Jascha Sohl-Dickstein et al., ICML 2015)
> - *Denoising Diffusion Probabilistic Models (DDPM)* (Jonathan Ho, Ajay Jain, Pieter Abbeel, NeurIPS 2020)
> - *Denoising Diffusion Implicit Models (DDIM)* (Jiaming Song, Chenlin Meng, Stefano Ermon, ICLR 2021)
> - *Score-Based Generative Modeling through Stochastic Differential Equations* (Yang Song et al., ICLR 2021)
> - *Classifier-Free Diffusion Guidance* (Jonathan Ho & Tim Salimans, NeurIPS Workshop 2021)
> - *Elucidating the Design Space of Diffusion-Based Generative Models (EDM)* (Tero Karras et al., NeurIPS 2022)

---

## 1. Intuition & 101 Motivation

### The Thermodynamic Analogy & The Denoising Paradigm
In GANs and VAEs, a neural network is asked to perform an impossible, discontinuous leap: transform a simple isotropic Gaussian noise vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ into a complex, high-dimensional image $\mathbf{x} \in \mathbb{R}^{3 \times 512 \times 512}$ in a **single forward step**.
This single step requires the generator to resolve global semantics, anatomical structures, and high-frequency textures simultaneously, leading to notorious training instabilities and mode collapse.

**Denoising Diffusion Probabilistic Models (DDPM)**—pioneered by Sohl-Dickstein et al. (2015) and perfected by Ho, Jain, & Abbeel (2020)—take inspiration from non-equilibrium thermodynamics:
1. **The Forward Process (Diffusion):** We take an image $\mathbf{x}_0$ and systematically destroy its structure by adding microscopic amounts of Gaussian noise over $T = 1000$ discrete timesteps. By timestep $T$, the image is indistinguishable from pure white noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
2. **The Reverse Process (Generation):** Because each individual forward step adds only a microscopic amount of noise, the true reverse distribution $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ is **also Gaussian**! We train a neural network to estimate and subtract this microscopic noise at every step.
3. **Sampling:** To generate a brand-new masterpiece from scratch, we draw pure Gaussian static $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and run the learned reverse process backward for $T$ steps:
   $$\mathbf{x}_T \xrightarrow{\text{denoise}} \mathbf{x}_{T-1} \xrightarrow{\text{denoise}} \cdots \xrightarrow{\text{denoise}} \mathbf{x}_1 \xrightarrow{\text{denoise}} \mathbf{x}_0$$

```
                   THE FORWARD AND REVERSE DIFFUSION PROCESS
    x_0 (Clean Image)     x_1               x_t                 x_T (Pure Noise)
       ┌─────────┐    ┌─────────┐       ┌─────────┐         ┌─────────┐
       │  (Dog)  │───►│ (Noisy) │───►...│ (Static)│───►...──►│ (Noise) │
       └─────────┘    └─────────┘       └─────────┘         └─────────┘
            ◄────────────── ◄─────────────── ◄───────────────────┘
               Reverse Denoising Step: p_theta(x_{t-1} | x_t)
                     (Neural Network predicts noise epsilon)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Forward Diffusion Process

Let $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ be the initial data distribution.
The forward process is a Markov chain that successively corrupts the data according to a pre-defined variance schedule $\beta_1, \beta_2, \dots, \beta_T$ with $\beta_t \in (0, 1)$:
$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$
where each transition is a Gaussian distribution:
$$\mathbf{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left( \mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I} \right)}$$

---

### 2.2 First-Principles Derivation of the Direct Closed-Form Marginal $q(\mathbf{x}_t \mid \mathbf{x}_0)$

A naive implementation would require sampling all intermediate states $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_{t-1}$ to obtain $\mathbf{x}_t$, which would make training prohibitively slow ($\mathcal{O}(T)$ operations per sample).
Remarkably, we can jump directly from clean image $\mathbf{x}_0$ to noisy image $\mathbf{x}_t$ in **$\mathcal{O}(1)$ closed form**.

Let:
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

**Theorem:**
$$\mathbf{q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I} \right)}$$
or equivalently via the reparameterization trick:
$$\mathbf{\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}$$

**Proof by Mathematical Induction:**

1. **Base Step ($t = 1$):**
   $$\mathbf{x}_1 = \sqrt{\alpha_1} \mathbf{x}_0 + \sqrt{1 - \alpha_1} \boldsymbol{\epsilon}_0, \quad \boldsymbol{\epsilon}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
   Since $\bar{\alpha}_1 = \alpha_1$, the base case holds.

2. **Induction Step:**
   Assume the theorem holds for $t - 1$:
   $$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_{t-2}$$
   From the definition of the forward transition:
   $$\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$$
   Substitute $\mathbf{x}_{t-1}$:
   $$\begin{aligned}
   \mathbf{x}_t &= \sqrt{\alpha_t} \left( \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \\
   &= \sqrt{\alpha_t \bar{\alpha}_{t-1}} \mathbf{x}_0 + \underbrace{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}}_{\text{Sum of two independent Gaussians}}
   \end{aligned}$$

   Recall that for independent Gaussians $\mathcal{N}(\mathbf{0}, \sigma_A^2 \mathbf{I})$ and $\mathcal{N}(\mathbf{0}, \sigma_B^2 \mathbf{I})$, their sum is Gaussian with variance $\sigma_A^2 + \sigma_B^2$:
   $$\begin{aligned}
   \sigma_{\text{total}}^2 &= \left( \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})} \right)^2 + \left( \sqrt{1 - \alpha_t} \right)^2 \\
   &= \alpha_t(1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t) \\
   &= \alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t \\
   &= 1 - \alpha_t \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t
   \end{aligned}$$

   Therefore:
   $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \blacksquare$$

---

### 2.3 The Reverse Process Posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$

The reverse step $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ is intractable because it depends on the entire data distribution $q(\mathbf{x}_0)$. However, if we **condition on the original clean image $\mathbf{x}_0$**, the posterior becomes tractable via Bayes' rule:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}$$

Expanding the Gaussian exponents:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \propto \exp\left( -\frac{1}{2} \left[ \frac{\|\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1}\|^2}{\beta_t} + \frac{\|\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0\|^2}{1 - \bar{\alpha}_{t-1}} - \frac{\|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2}{1 - \bar{\alpha}_t} \right] \right)$$

Completing the square with respect to $\mathbf{x}_{t-1}$:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I} \right)$$

where the posterior mean is:
$$\mathbf{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t}$$
and the posterior variance is:
$$\mathbf{\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}$$

---

### 2.4 The Denoising Objective: Noise Prediction Parameterization

From our direct marginal formula $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$, solve for $\mathbf{x}_0$:
$$\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

Substitute this expression for $\mathbf{x}_0$ into the posterior mean $\tilde{\boldsymbol{\mu}}_t$:
$$\begin{aligned}
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}} \right) + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t \\
&= \left[ \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \right] \mathbf{x}_t - \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t \sqrt{1 - \bar{\alpha}_t}}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \boldsymbol{\epsilon}
\end{aligned}$$

Using $\sqrt{\bar{\alpha}_t} = \sqrt{\alpha_t} \sqrt{\bar{\alpha}_{t-1}}$:
- The coefficient of $\mathbf{x}_t$ simplifies to:
  $$\frac{\beta_t + \alpha_t(1 - \bar{\alpha}_{t-1})}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1 - \alpha_t + \alpha_t - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \mathbf{\frac{1}{\sqrt{\alpha_t}}}$$
- The coefficient of $\boldsymbol{\epsilon}$ simplifies to:
  $$\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}$$

Thus, the exact posterior mean is:
$$\mathbf{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \boldsymbol{\epsilon}) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)}$$

Instead of predicting the mean $\tilde{\boldsymbol{\mu}}_t$ directly, we train a neural network $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ to predict the **exact noise vector** $\boldsymbol{\epsilon}$ that was added to $\mathbf{x}_0$!

#### The Simplified DDPM Loss Function (Ho et al., 2020)
$$\mathbf{\mathcal{L}_{\text{simple}}(\boldsymbol{\theta}) = \mathbb{E}_{t \sim \mathcal{U}\{1, T\}, \mathbf{x}_0 \sim q(\mathbf{x}_0), \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t \right) \right\|_2^2 \right]}$$

#### The Reverse Sampling Algorithm (Generation)
To sample a new image at inference time:
1. Sample $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
2. For $t = T, T-1, \dots, 1$:
   - Sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ if $t > 1$, else $\mathbf{z} = \mathbf{0}$.
   - Compute:
     $$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$
     where $\sigma_t = \sqrt{\tilde{\beta}_t}$ or $\sigma_t = \sqrt{\beta_t}$.

---

### 2.5 Denoising Diffusion Implicit Models (DDIM - Song et al., 2020)

Standard DDPM requires sampling every single step of the Markov chain ($T = 1000$), which is prohibitively slow for interactive inference (taking 10 to 30 seconds per image).

**Key Theoretical Innovation:** Song et al. (2020) showed that the exact same neural network trained with the DDPM objective can sample along **non-Markovian forward inference distributions** that share the exact same marginals $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$:

$$q_\sigma(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = q_\sigma(\mathbf{x}_T \mid \mathbf{x}_0) \prod_{t=2}^T q_\sigma(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$$

where:
$$q_\sigma(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left( \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I} \right)$$

We parameterize the stochasticity via $\eta \in [0, 1]$:
$$\sigma_t = \eta \cdot \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$

1. **When $\eta = 1$:** The generative process becomes identical to standard stochastic **DDPM**.
2. **When $\eta = 0$ (DDIM Deterministic Sampling):** The variance $\sigma_t = 0$. The reverse step becomes completely **deterministic**:
   $$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{Predicted Clean Image } \hat{\mathbf{x}}_0} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)}_{\text{Direction pointing to } \mathbf{x}_{t-1}}$$

**Benefits of DDIM:**
- **Sub-sequence Accelerated Sampling:** One can sample along an arbitrary sub-sequence $\tau_1 < \tau_2 < \dots < \tau_S \ll T$ (e.g., $S = 20$ or $50$ steps instead of $1000$).
- **Invertibility & Latent Manipulation:** When $\eta = 0$, the forward and reverse trajectories are deterministic ODE paths. An image $\mathbf{x}_0$ can be inverted back to noise $\mathbf{x}_T$ and accurately edited in latent space.

---

### 2.6 The Karras EDM Framework (Elucidating the Design Space - Karras et al., 2022)

In modern production systems (such as Stable Diffusion XL and FLUX), continuous noise variance $\sigma \in [\sigma_{\min}, \sigma_{\max}]$ is decoupled from discrete step indices.

Karras et al. formalize diffusion as continuous denoising with clean signal $\mathbf{y} \sim p_{\text{data}}$ corrupted by additive Gaussian noise $\mathbf{x} = \mathbf{y} + \mathbf{n}$ where $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$.

The network is parameterized through scale-invariant **preconditioning functions**:
$$D_{\boldsymbol{\theta}}(\mathbf{x}; \sigma) = c_{\text{skip}}(\sigma) \mathbf{x} + c_{\text{out}}(\sigma) F_{\boldsymbol{\theta}}(c_{\text{in}}(\sigma) \mathbf{x}; c_{\text{noise}}(\sigma))$$

where:
$$c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \quad c_{\text{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{noise}}(\sigma) = \frac{1}{4} \ln(\sigma)$$

with data standard deviation $\sigma_{\text{data}} \approx 0.5$.
This preconditioning guarantees that:
- When $\sigma \to 0$ (almost clean): $c_{\text{skip}} \to 1, c_{\text{out}} \to 0$, protecting the network from predicting huge values.
- When $\sigma \to \infty$ (pure noise): $c_{\text{skip}} \to 0$, forcing the network to predict the clean data mode.
- The training loss is uniformly weighted across all noise levels: $\lambda(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$.

---

### 2.7 Unification with Score-Based Generative Modeling & SDEs (Song et al., 2020)

What is the neural network $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ actually learning?

**Tweedie's Formula:**
For a Gaussian corrupted observation $\mathbf{x} \sim \mathcal{N}(\mathbf{x}_0, \sigma^2 \mathbf{I})$, the posterior expectation of the clean data is:
$$\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}] = \mathbf{x} + \sigma^2 \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

For our diffusion marginal $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$:
$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1 - \bar{\alpha}_t} = -\frac{\sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{1 - \bar{\alpha}_t} = \mathbf{-\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}}$$

The **Stein Score Function** of the marginal data density is:
$$\mathbf{\mathbf{s}(\mathbf{x}_t, t) \equiv \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}}$$

> **Key Fundamental Insight:** Predicting the noise $\boldsymbol{\epsilon}$ is mathematically identical to estimating the **Score Function** (the gradient of log-density $\nabla_{\mathbf{x}} \log p(\mathbf{x})$)! Denoising diffusion models are simply **Score-Based Generative Models** scaled to deep neural networks.

#### The Continuous-Time SDE Formulation
In the limit as $T \to \infty$ and $\Delta t \to 0$:
- **Forward SDE:**
  $$d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} dt + \sqrt{\beta(t)} d\mathbf{w}$$
- **Reverse SDE (Anderson, 1982):**
  $$d\mathbf{x} = \left[ -\frac{1}{2} \beta(t) \mathbf{x} - \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + \sqrt{\beta(t)} d\bar{\mathbf{w}}$$
- **Probability Flow ODE:**
  $$d\mathbf{x} = \left[ -\frac{1}{2} \beta(t) \mathbf{x} - \frac{1}{2} \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt$$
  This allows **exact deterministic sampling** and **exact likelihood computation** using standard ODE solvers (Runge-Kutta, Euler-Maruyama)!

---

### 2.8 Classifier-Free Guidance (CFG - Ho & Salimans, 2021)

In conditioned diffusion (e.g., text-to-image prompts $\mathbf{c}$), we want to sample from:
$$p(\mathbf{x} \mid \mathbf{c}) \propto p(\mathbf{x}) p(\mathbf{c} \mid \mathbf{x})$$

Taking the gradient of the log-probability:
$$\nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{c}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \nabla_{\mathbf{x}} \log p(\mathbf{c} \mid \mathbf{x})$$

To control how aggressively the model adheres to prompt $\mathbf{c}$, we introduce a guidance scale $s \ge 1$:
$$\nabla_{\mathbf{x}} \log \tilde{p}_s(\mathbf{x} \mid \mathbf{c}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + s \nabla_{\mathbf{x}} \log p(\mathbf{c} \mid \mathbf{x})$$

Substitute $\nabla_{\mathbf{x}} \log p(\mathbf{c} \mid \mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{c}) - \nabla_{\mathbf{x}} \log p(\mathbf{x})$:
$$\begin{aligned}
\nabla_{\mathbf{x}} \log \tilde{p}_s(\mathbf{x} \mid \mathbf{c}) &= \nabla_{\mathbf{x}} \log p(\mathbf{x}) + s \left( \nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{c}) - \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right) \\
&= (1 - s) \nabla_{\mathbf{x}} \log p(\mathbf{x}) + s \nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{c})
\end{aligned}$$

Translating to noise predictions using $\nabla_{\mathbf{x}} \log p \propto -\boldsymbol{\epsilon}$:
$$\mathbf{\tilde{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{x}_t, \mathbf{c}) = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, \emptyset) + s \cdot \left( \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, \emptyset) \right)}$$
where $\emptyset$ is the null conditioning token (unconditional generation).

```
                      CLASSIFIER-FREE GUIDANCE VECTOR
                          Noise Space (Direction of Denoising)
                 Unconditional
                   Prediction:
                   eps(x_t, null)
                       ●
                       │ \
                       │  \  Conditioning vector:
                       │   \ [eps(x_t, c) - eps(x_t, null)]
                       │    ▼
                       │    ● Conditional Prediction: eps(x_t, c)
                       │     \
                       │      \  Extrapolate by scale s > 1
                       │       ▼
                       └───────► ● GUIDED NOISE:
                                   eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
```

- **$s = 1$:** Standard conditional generation.
- **$s > 1$ (e.g., $s = 7.5$ in Stable Diffusion):** Dramatically amplifies prompt adherence and perceptual contrast, boosting visual aesthetics at the cost of diversity.

---

### 2.9 Rigorous Mathematical Derivations

#### Derivation 10.4.1: Variational Lower Bound (VLB) on Diffusion Log-Likelihood and Rao-Blackwellized Step-by-Step KL Decomposition

**1. Context and Assumptions:**
Let $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ be real data.
The forward process is a fixed Gaussian Markov chain:
$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1}), \quad q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$
The parameterized reverse process is a Markov chain initialized at standard normal noise $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:
$$p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t), \quad p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I}\right)$$

**2. Step 1: Variational Bound Formulation:**
By Jensen's inequality:
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}_0) = \log \int p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} = \log \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \frac{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right] \ge \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right]$$
We define the negative variational lower bound $\mathcal{L}_{\text{VLB}}$:
$$\mathcal{L}_{\text{VLB}} \triangleq \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T})} \right] \ge -\log p_{\boldsymbol{\theta}}(\mathbf{x}_0)$$

**3. Step 2: Factoring Products and Applying Bayes Rule:**
Expand the log-ratio:
$$\log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T})} = \log \left( \frac{\prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}{p(\mathbf{x}_T) \prod_{t=1}^T p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} \right)$$
Condition each forward transition on the clean image $\mathbf{x}_0$:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \cdot \frac{q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}$$
For $t = 1$, $q(\mathbf{x}_1 \mid \mathbf{x}_0)$ has no conditioning denominator. For $t \ge 2$, substitute this Bayes identity into the product:
$$\prod_{t=2}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \prod_{t=2}^T \left( q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \frac{q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)} \right) = \left( \prod_{t=2}^T q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \right) \cdot \frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)}$$
where the intermediate terms $q(\mathbf{x}_t \mid \mathbf{x}_0)$ telescope completely!
Multiplying by $q(\mathbf{x}_1 \mid \mathbf{x}_0)$ from the $t=1$ term:
$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = q(\mathbf{x}_1 \mid \mathbf{x}_0) \cdot \frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)} \cdot \prod_{t=2}^T q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = q(\mathbf{x}_T \mid \mathbf{x}_0) \prod_{t=2}^T q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$$

**4. Step 3: Grouping into KL Divergences:**
Substituting this into $\mathcal{L}_{\text{VLB}}$:
$$\begin{aligned}
\mathcal{L}_{\text{VLB}} &= \mathbb{E}_q \left[ \log \left( \frac{q(\mathbf{x}_T \mid \mathbf{x}_0) \prod_{t=2}^T q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p(\mathbf{x}_T) p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1) \prod_{t=2}^T p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} \right) \right] \\
&= \mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{p(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} - \log p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1) \right]
\end{aligned}$$
Taking expectations over the corresponding marginals (Rao-Blackwellization):
$$\mathbf{\mathcal{L}_{\text{VLB}} = \underbrace{D_{\mathrm{KL}}\left( q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T) \right)}_{L_T \text{ (Prior Matching)}} + \sum_{t=2}^T \underbrace{\mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)}\left[ D_{\mathrm{KL}}\left( q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \right) \right]}_{L_{t-1} \text{ (Denoising Transitions)}} - \underbrace{\mathbb{E}_{q(\mathbf{x}_1 \mid \mathbf{x}_0)}\left[ \log p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1) \right]}_{L_0 \text{ (Reconstruction)}}}$$

**5. Step 4: Analytical Gaussian Evaluation of $L_{t-1}$:**
Both $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t \mathbf{I})$ and $p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}, \sigma_t^2 \mathbf{I})$ are isotropic Gaussians.
For $\sigma_t^2 = \tilde{\beta}_t$:
$$D_{\mathrm{KL}}\left( \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t \mathbf{I}) \,\|\, \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}, \tilde{\beta}_t \mathbf{I}) \right) = \frac{1}{2\tilde{\beta}_t} \|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\|_2^2$$
Substituting $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}\right)$ and parameterizing $\boldsymbol{\mu}_{\boldsymbol{\theta}} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\right)$:
$$L_{t-1} = \frac{1}{2\tilde{\beta}_t} \left\| \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}} \left( \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \right) \right\|_2^2 = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t(1 - \bar{\alpha}_t)} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \right\|_2^2$$
Ho et al. (2020) demonstrated empirically that dropping the prefactor $\frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t(1 - \bar{\alpha}_t)}$ (which heavily weights small $t$) down-weights trivial reconstruction and emphasizes perceptual sample quality, yielding the unweighted $\mathcal{L}_{\text{simple}}$ objective.

---

#### Derivation 10.4.2: Denoising Score Matching, Fisher Divergence, and Equivalence to Noise Prediction

**1. Context and Assumptions:**
Let $\mathbf{x} \sim p_{\text{data}}(\mathbf{x})$ be an uncorrupted data vector.
Let $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}$ be an observation perturbed by Gaussian noise with variance $\sigma^2$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
The perturbation kernel is $q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I})$, and the perturbed marginal density is:
$$p_\sigma(\tilde{\mathbf{x}}) = \int p_{\text{data}}(\mathbf{x}) q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) d\mathbf{x}$$
We parameterize a score network $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})$ to estimate the Stein score function $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$.

**2. Step 1: Explicit Score Matching (Fisher Divergence):**
The ideal objective minimizes the expected squared Euclidean error to the true score:
$$J_{\text{ESM}}(\boldsymbol{\theta}) \triangleq \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma} \left[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}) \right\|_2^2 \right]$$
Expanding the norm:
$$J_{\text{ESM}}(\boldsymbol{\theta}) = \frac{1}{2} \int p_\sigma(\tilde{\mathbf{x}}) \|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})\|^2 d\tilde{\mathbf{x}} - \int p_\sigma(\tilde{\mathbf{x}}) \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}) d\tilde{\mathbf{x}} + C_{\text{ESM}}$$
where $C_{\text{ESM}} = \frac{1}{2} \mathbb{E}_{p_\sigma}[\|\nabla \log p_\sigma\|^2]$ is independent of $\boldsymbol{\theta}$.

**3. Step 2: Denoising Score Matching Formulation (Vincent, 2011):**
Since $p_\sigma(\tilde{\mathbf{x}})$ and its gradient are intractable, Vincent proposed replacing the marginal score with the **conditional score** $\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$:
$$J_{\text{DSM}}(\boldsymbol{\theta}) \triangleq \frac{1}{2} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})} \left[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \right\|_2^2 \right]$$
Expanding:
$$J_{\text{DSM}}(\boldsymbol{\theta}) = \frac{1}{2} \mathbb{E}_{p_\sigma}[\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})\|^2] - \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}} \left[ \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \right] + C_{\text{DSM}}$$
Now examine the cross term:
$$\begin{aligned}
\mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}} \left[ \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \right] &= \int_{\mathbf{x}} \int_{\tilde{\mathbf{x}}} p_{\text{data}}(\mathbf{x}) q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \frac{\nabla_{\tilde{\mathbf{x}}} q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})} d\tilde{\mathbf{x}} d\mathbf{x} \\
&= \int_{\tilde{\mathbf{x}}} \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \left( \int_{\mathbf{x}} p_{\text{data}}(\mathbf{x}) \nabla_{\tilde{\mathbf{x}}} q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) d\mathbf{x} \right) d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \nabla_{\tilde{\mathbf{x}}} \underbrace{\left( \int_{\mathbf{x}} p_{\text{data}}(\mathbf{x}) q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) d\mathbf{x} \right)}_{p_\sigma(\tilde{\mathbf{x}})} d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} p_\sigma(\tilde{\mathbf{x}}) \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \frac{\nabla_{\tilde{\mathbf{x}}} p_\sigma(\tilde{\mathbf{x}})}{p_\sigma(\tilde{\mathbf{x}})} d\tilde{\mathbf{x}} \\
&= \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma} \left[ \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})^\top \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}) \right]
\end{aligned}$$
The cross term of $J_{\text{DSM}}$ is **identically equal** to the cross term of $J_{\text{ESM}}$!
Therefore:
$$J_{\text{DSM}}(\boldsymbol{\theta}) = J_{\text{ESM}}(\boldsymbol{\theta}) + \text{const}$$
Minimizing Denoising Score Matching over samples $(\mathbf{x}, \tilde{\mathbf{x}})$ is mathematically identical to minimizing Explicit Score Matching over the intractable true data density!

**4. Step 3: Analytical Equivalence to Noise Prediction:**
For Gaussian perturbation $q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\mathbf{x}, \sigma^2 \mathbf{I})$:
$$\log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|^2}{2\sigma^2} \implies \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}$$
Parameterize the score network as $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})}{\sigma}$:
$$\begin{aligned}
J_{\text{DSM}}(\boldsymbol{\theta}) &= \frac{1}{2} \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \left[ \left\| -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})}{\sigma} - \left( -\frac{\boldsymbol{\epsilon}}{\sigma} \right) \right\|_2^2 \right] \\
&= \frac{1}{2\sigma^2} \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) \right\|_2^2 \right]
\end{aligned}$$
Score matching on continuous noisy data is mathematically isomorphic to predicting standard Gaussian noise $\boldsymbol{\epsilon}$.

---

#### Derivation 10.4.3: Continuous-Time Probability Flow ODE and Instantaneous Change of Variables

**1. Context and Assumptions:**
Let $\mathbf{x}_t \in \mathbb{R}^D$ evolve according to the general Itô Forward Stochastic Differential Equation (SDE):
$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}$$
where $\mathbf{f}(\mathbf{x}, t) \in \mathbb{R}^D$ is the drift coefficient, $g(t) \in \mathbb{R}$ is the scalar diffusion coefficient, and $\mathbf{w}$ is standard $D$-dimensional Brownian motion.
Let $p_t(\mathbf{x})$ denote the marginal probability density of $\mathbf{x}_t$.

**2. Step 1: Forward Fokker-Planck (Kolmogorov Forward) Equation:**
The time evolution of the marginal density $p_t(\mathbf{x})$ is governed by:
$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\sum_{i=1}^D \frac{\partial}{\partial x_i} \left[ f_i(\mathbf{x}, t) p_t(\mathbf{x}) \right] + \frac{1}{2} g(t)^2 \sum_{i=1}^D \frac{\partial^2}{\partial x_i^2} p_t(\mathbf{x})$$
In vector notation:
$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_{\mathbf{x}} \cdot \left( \mathbf{f}(\mathbf{x}, t) p_t(\mathbf{x}) \right) + \frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \cdot \left( \nabla_{\mathbf{x}} p_t(\mathbf{x}) \right)$$

**3. Step 2: Score Function Substitution:**
Using the identity $\nabla_{\mathbf{x}} p_t(\mathbf{x}) = p_t(\mathbf{x}) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$:
$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_{\mathbf{x}} \cdot \left( \mathbf{f}(\mathbf{x}, t) p_t(\mathbf{x}) \right) + \frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \cdot \left( p_t(\mathbf{x}) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right)$$
Factoring out the divergence operator $-\nabla_{\mathbf{x}} \cdot$:
$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_{\mathbf{x}} \cdot \left( \left[ \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] p_t(\mathbf{x}) \right)$$

**4. Step 3: Derivation of the Deterministic Probability Flow ODE:**
Recall the continuity equation of fluid dynamics: for any deterministic ODE $\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t)$, the probability density evolves according to:
$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_{\mathbf{x}} \cdot \left( \mathbf{v}(\mathbf{x}, t) p_t(\mathbf{x}) \right)$$
Comparing the two equations, the deterministic velocity field:
$$\mathbf{v}(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$
generates **identical marginal densities $p_t(\mathbf{x})$ at all times $t \in [0, T]$** as the stochastic diffusion process!
Thus, the **Probability Flow ODE** is:
$$\mathbf{\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t)}$$

**5. Step 4: Exact Log-Likelihood Computation via Instantaneous Change of Variables:**
Along any trajectory $\mathbf{x}(t)$ governed by $\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t)$, compute the total time derivative of $\log p_t(\mathbf{x}(t))$:
$$\frac{d}{dt} \log p_t(\mathbf{x}(t)) = \frac{\partial \log p_t}{\partial t} + \nabla_{\mathbf{x}} \log p_t(\mathbf{x})^\top \frac{d\mathbf{x}}{dt} = \frac{1}{p_t} \frac{\partial p_t}{\partial t} + \nabla_{\mathbf{x}} \log p_t^\top \mathbf{v}$$
Substitute $\frac{\partial p_t}{\partial t} = -\nabla_{\mathbf{x}} \cdot (\mathbf{v} p_t) = -p_t (\nabla_{\mathbf{x}} \cdot \mathbf{v}) - \mathbf{v}^\top \nabla_{\mathbf{x}} p_t$:
$$\frac{d}{dt} \log p_t(\mathbf{x}(t)) = \frac{-p_t (\nabla_{\mathbf{x}} \cdot \mathbf{v}) - \mathbf{v}^\top \nabla_{\mathbf{x}} p_t}{p_t} + \mathbf{v}^\top \nabla_{\mathbf{x}} \log p_t = -(\nabla_{\mathbf{x}} \cdot \mathbf{v}) - \mathbf{v}^\top \nabla \log p_t + \mathbf{v}^\top \nabla \log p_t = -\nabla_{\mathbf{x}} \cdot \mathbf{v}(\mathbf{x}, t)$$
Integrating from data $t = 0$ to prior noise $t = T$:
$$\log p_0(\mathbf{x}_0) = \log p_T(\mathbf{x}_T) + \int_0^T \nabla_{\mathbf{x}} \cdot \left[ \mathbf{f}(\mathbf{x}(t), t) - \frac{1}{2} g(t)^2 \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t) \right] dt$$
The divergence of the score network can be evaluated efficiently using the **Hutchinson trace estimator**:
$$\nabla_{\mathbf{x}} \cdot \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) = \operatorname{tr}\left( \nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}} \right) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \boldsymbol{\epsilon}^\top \nabla_{\mathbf{x}} \left( \mathbf{s}_{\boldsymbol{\theta}}^\top \boldsymbol{\epsilon} \right) \right]$$
requiring only a single vector-Jacobian product (VJP)! This enables exact likelihood evaluation of continuous diffusion models.

---

## 3. Geometric & Algebraic Interpretation

### The Denoising Vector Field Across Timesteps
The score function $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ defines a time-varying vector field:
1. **At Large $t \approx T$ (Coarse Macro-Structure):** The data distribution is heavily blurred by large Gaussian variance. The score field points globally toward the broad center of data mass, establishing composition and layout.
2. **At Intermediate $t \approx T/2$ (Semantic Formation):** The vector field bifurcates into distinct basins of attraction, steering samples toward specific semantic categories (e.g., separating cats from dogs).
3. **At Small $t \approx 0$ (High-Frequency Micro-Details):** The variance is microscopic. The vector field performs ultra-fine local adjustments, sharpening crisp edges, hair strands, and lighting highlights.

---

## 4. Real-World Analogy

### The Glass of Water & The Time-Reversal Movie
- **The Forward Process (Entropy at Work):**
  Drop a droplet of dark blue ink into a glass of crystal-clear water. At $t = 0$, you have a concentrated, structured droplet. At $t = 50$, the ink blooms into tendrils. By $t = 1000$, thermodynamic entropy has completely dispersed the ink molecules uniformly throughout the glass. You now have homogenous pale blue water (pure static).
- **The Reverse Process (Maxwell's Denoising Demon):**
  A trained neural network acts as a microscopic demon watching each water molecule. At step $t = 999$, it nudges ink molecules slightly backward toward their neighbors. By step $t = 500$, clear streams of water and rich ink tendrils reappear. By step $t = 0$, the demon has assembled a pristine, perfectly concentrated ink droplet out of total chaos!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us compute a complete DDPM forward noise injection, network prediction, simple loss, reverse step, and CFG extrapolation by hand with concrete toy numbers.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in DDPM Architecture |
| :--- | :--- | :--- | :--- |
| $\mathbf{x}_0$ | Clean Data Sample | $(2,)$ | Ground truth input vector $[1.0, 2.0]$ |
| $t$ | Discrete Timestep | Scalar | Current diffusion step ($t = 2$) |
| $\beta_t$ | Forward Variance Schedule | Scalar | Noise variance at step $t$ ($\beta_1 = 0.1, \beta_2 = 0.2$) |
| $\alpha_t = 1 - \beta_t$ | Single-Step Retained Variance | Scalar | Signal retained ($\alpha_1 = 0.9, \alpha_2 = 0.8$) |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | Cumulative Retained Signal | Scalar | Cumulative product ($\bar{\alpha}_1 = 0.9, \bar{\alpha}_2 = 0.72$) |
| $\boldsymbol{\epsilon}$ | Ground Truth Added Noise | $(2,)$ | Standard Gaussian noise sample $[0.5, -1.0]$ |
| $\mathbf{x}_t$ | Noisy State Vector | $(2,)$ | $\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$ |
| $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ | Predicted Noise Vector | $(2,)$ | Neural network estimate of noise |
| $\mathcal{L}_{\text{simple}}$ | Denoising Loss | Scalar | Squared error $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\|^2$ |
| $\tilde{\boldsymbol{\mu}}_t$ | Posterior Mean Vector | $(2,)$ | Denoised center for reverse transition |
| $\tilde{\beta}_t$ | Posterior Variance | Scalar | Variance of reverse transition |
| $\mathbf{x}_{t-1}$ | Denoised Sample | $(2,)$ | Sample from reverse posterior step |
| $s$ | CFG Scale | Scalar | Classifier-Free Guidance extrapolation scale ($s = 2.0$) |

---

### 5.2 Concrete Toy Setup

Let:
- Clean data vector: $\mathbf{x}_0 = \begin{bmatrix} 1.0000 \\ 2.0000 \end{bmatrix}$
- Diffusion schedule:
  - Step $1$: $\beta_1 = 0.1000 \implies \alpha_1 = 0.9000 \implies \bar{\alpha}_1 = 0.9000$
  - Step $2$: $\beta_2 = 0.2000 \implies \alpha_2 = 0.8000 \implies \bar{\alpha}_2 = \alpha_1 \alpha_2 = 0.9000 \times 0.8000 = \mathbf{0.7200}$
- At timestep $t = 2$:
  $$\sqrt{\bar{\alpha}_2} = \sqrt{0.7200} \approx \mathbf{0.8485}$$
  $$\sqrt{1 - \bar{\alpha}_2} = \sqrt{1 - 0.7200} = \sqrt{0.2800} \approx \mathbf{0.5292}$$
- Fixed Gaussian noise sample:
  $$\boldsymbol{\epsilon} = \begin{bmatrix} 0.5000 \\ -1.0000 \end{bmatrix}$$

---

### 5.3 Step 1: Forward Noise Injection Arithmetic ($\mathcal{O}(1)$ Closed-Form)

$$\mathbf{x}_2 = \sqrt{\bar{\alpha}_2} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_2} \boldsymbol{\epsilon}$$
$$x_{2, 1} = 0.8485(1.0000) + 0.5292(0.5000) = 0.8485 + 0.2646 = \mathbf{1.1131}$$
$$x_{2, 2} = 0.8485(2.0000) + 0.5292(-1.0000) = 1.6971 - 0.5292 = \mathbf{1.1679}$$

$$\mathbf{x}_2 = \begin{bmatrix} 1.1131 \\ 1.1679 \end{bmatrix}$$

---

### 5.4 Step 2: Denoising Loss Evaluation Arithmetic

Suppose our neural network predicts:
$$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, t=2) = \begin{bmatrix} 0.4000 \\ -0.8000 \end{bmatrix}$$
The error vector:
$$\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}} = \begin{bmatrix} 0.5000 - 0.4000 \\ -1.0000 - (-0.8000) \end{bmatrix} = \begin{bmatrix} +0.1000 \\ -0.2000 \end{bmatrix}$$

The simplified loss:
$$\mathcal{L}_{\text{simple}} = \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\|^2 = (0.1000)^2 + (-0.2000)^2 = 0.0100 + 0.0400 = \mathbf{0.0500}$$

---

### 5.5 Step 3: Single Reverse Denoising Step Arithmetic ($t = 2 \to t = 1$)

Using the network's predicted noise $\boldsymbol{\epsilon}_{\boldsymbol{\theta}} = [0.4000, -0.8000]^\top$:

1. **Posterior Mean Formula:**
   $$\tilde{\boldsymbol{\mu}}_2 = \frac{1}{\sqrt{\alpha_2}} \left( \mathbf{x}_2 - \frac{\beta_2}{\sqrt{1 - \bar{\alpha}_2}} \boldsymbol{\epsilon}_{\boldsymbol{\theta}} \right)$$
   Evaluate coefficients:
   $$\frac{1}{\sqrt{\alpha_2}} = \frac{1}{\sqrt{0.8000}} = \frac{1}{0.8944} \approx 1.1180$$
   $$\frac{\beta_2}{\sqrt{1 - \bar{\alpha}_2}} = \frac{0.2000}{\sqrt{0.2800}} = \frac{0.2000}{0.5292} \approx 0.3780$$

2. **Compute Inner Subtraction:**
   $$\mathbf{x}_2 - 0.3780 \boldsymbol{\epsilon}_{\boldsymbol{\theta}} = \begin{bmatrix} 1.1131 - 0.3780(0.4000) \\ 1.1679 - 0.3780(-0.8000) \end{bmatrix} = \begin{bmatrix} 1.1131 - 0.1512 \\ 1.1679 + 0.3024 \end{bmatrix} = \begin{bmatrix} 0.9619 \\ 1.4703 \end{bmatrix}$$

3. **Scale by $1 / \sqrt{\alpha_2}$:**
   $$\tilde{\boldsymbol{\mu}}_2 = 1.1180 \begin{bmatrix} 0.9619 \\ 1.4703 \end{bmatrix} = \begin{bmatrix} \mathbf{1.0754} \\ \mathbf{1.6438} \end{bmatrix}$$

4. **Posterior Variance:**
   $$\tilde{\beta}_2 = \frac{1 - \bar{\alpha}_1}{1 - \bar{\alpha}_2} \beta_2 = \frac{1 - 0.9000}{1 - 0.7200} (0.2000) = \frac{0.1000}{0.2800} (0.2000) = \frac{0.0200}{0.2800} = \mathbf{0.0714}$$
   $$\sigma_2 = \sqrt{\tilde{\beta}_2} = \sqrt{0.0714} \approx \mathbf{0.2673}$$

5. **Sample Denoised State $\mathbf{x}_1$:**
   Let auxiliary noise $\mathbf{z} = [0.1000, -0.1000]^\top$:
   $$\mathbf{x}_1 = \tilde{\boldsymbol{\mu}}_2 + \sigma_2 \mathbf{z} = \begin{bmatrix} 1.0754 + 0.2673(0.1000) \\ 1.6438 + 0.2673(-0.1000) \end{bmatrix} = \begin{bmatrix} 1.0754 + 0.0267 \\ 1.6438 - 0.0267 \end{bmatrix} = \begin{bmatrix} \mathbf{1.1021} \\ \mathbf{1.6171} \end{bmatrix}$$

Notice how $\mathbf{x}_2 = [1.1131, 1.1679] \to \mathbf{x}_1 = [1.1021, 1.6171]$, steadily moving back toward clean data $\mathbf{x}_0 = [1.0000, 2.0000]$!

---

### 5.6 Step 4: Classifier-Free Guidance (CFG) Extrapolation Arithmetic

Suppose under conditional prompt $\mathbf{c}$ and null prompt $\emptyset$:
$$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, \emptyset) = \begin{bmatrix} 0.4000 \\ -0.8000 \end{bmatrix}, \quad \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, \mathbf{c}) = \begin{bmatrix} 0.6000 \\ -1.1000 \end{bmatrix}$$
Let guidance scale $s = 2.0$:
$$\begin{aligned}
\tilde{\boldsymbol{\epsilon}} &= \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, \emptyset) + s \cdot \left( \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, \mathbf{c}) - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, \emptyset) \right) \\
&= \begin{bmatrix} 0.4000 \\ -0.8000 \end{bmatrix} + 2.0 \left( \begin{bmatrix} 0.6000 \\ -1.1000 \end{bmatrix} - \begin{bmatrix} 0.4000 \\ -0.8000 \end{bmatrix} \right) \\
&= \begin{bmatrix} 0.4000 \\ -0.8000 \end{bmatrix} + 2.0 \begin{bmatrix} 0.2000 \\ -0.3000 \end{bmatrix} \\
&= \begin{bmatrix} 0.4000 + 0.4000 \\ -0.8000 - 0.6000 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.8000 \\ -1.4000 \end{bmatrix}}
\end{aligned}$$

The conditioning direction $[0.2, -0.3]$ is doubled, aggressively forcing the denoising path toward the prompt condition!

---

## 6. Solved Illustrations

### Illustration 1: Analytical Derivation of the Posterior Variance $\tilde{\beta}_t$
**Problem:** Prove that $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ minimizes the KL divergence between reverse transition $p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ and true posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$.
**Solution:**
From Bayes' rule:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \propto \exp\left( -\frac{1}{2} \left[ \frac{\|\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1}\|^2}{\beta_t} + \frac{\|\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0\|^2}{1 - \bar{\alpha}_{t-1}} \right] \right)$$
Collect the quadratic terms in $\mathbf{x}_{t-1}$:
$$\mathbf{x}_{t-1}^\top \left[ \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right] \mathbf{x}_{t-1} = \mathbf{x}_{t-1}^\top \left[ \frac{1}{\tilde{\beta}_t} \right] \mathbf{x}_{t-1}$$
Therefore:
$$\frac{1}{\tilde{\beta}_t} = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})} = \frac{\alpha_t - \bar{\alpha}_t + 1 - \alpha_t}{\beta_t(1 - \bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{\beta_t(1 - \bar{\alpha}_{t-1})}$$
Inverting:
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \quad \blacksquare$$

---

### Illustration 2: Influence of Guidance Scale $s$ on Fréchet Inception Distance (FID)
**Question:** Why does increasing guidance scale $s$ from $1 \to 7$ improve prompt alignment, but increasing $s > 15$ causes image oversaturation and artifacts?
**Answer:**
Classifier-Free Guidance acts as an exponential tilt on the density: $p_s(\mathbf{x} \mid \mathbf{c}) \propto p(\mathbf{x}) p(\mathbf{c} \mid \mathbf{x})^s$.
1. **At $s \in [3, 8]$:** The distribution sharpens around high-confidence modes of $p(\mathbf{c} \mid \mathbf{x})$, trimming away improbable tails and boosting FID and CLIP score.
2. **At $s > 15$:** The score vector norm $\|\tilde{\boldsymbol{\epsilon}}\|$ explodes, overshooting the valid dynamic range $[-1, 1]$. Latent activations saturate at extreme values, causing harsh chromatic contrast, unnatural haloing, and cartoonish "burned" textures.

---

### Illustration 3: Complete Step-by-Step Numerical 3-Step DDPM Sampling Run from Pure Noise

**Problem:**
Consider a 2D toy DDPM with $T = 3$ timesteps and linear variance schedule:
$$\beta_1 = 0.10, \qquad \beta_2 = 0.20, \qquad \beta_3 = 0.30$$
The schedule parameters are:
$$\alpha_1 = 0.90, \quad \alpha_2 = 0.80, \quad \alpha_3 = 0.70$$
$$\bar{\alpha}_1 = 0.9000, \quad \bar{\alpha}_2 = 0.7200, \quad \bar{\alpha}_3 = 0.5040$$
Posterior variances $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$:
$$\tilde{\beta}_3 = \frac{1 - 0.7200}{1 - 0.5040}(0.30) = \frac{0.2800}{0.4960}(0.30) \approx 0.169355 \implies \sqrt{\tilde{\beta}_3} \approx 0.411527$$
$$\tilde{\beta}_2 = \frac{1 - 0.9000}{1 - 0.7200}(0.20) = \frac{0.1000}{0.2800}(0.20) \approx 0.071429 \implies \sqrt{\tilde{\beta}_2} \approx 0.267261$$
$$\tilde{\beta}_1 = \beta_1 = 0.1000 \implies \sqrt{\tilde{\beta}_1} \approx 0.316228$$
We start sampling at $t = 3$ from pure standard normal noise:
$$\mathbf{x}_3 = \begin{bmatrix} 1.2000 \\ -0.8000 \end{bmatrix}$$
The trained noise prediction network outputs:
$$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_3, 3) = \begin{bmatrix} 0.8000 \\ -0.5000 \end{bmatrix}, \qquad \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, 2) = \begin{bmatrix} 0.4000 \\ -0.3000 \end{bmatrix}, \qquad \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_1, 1) = \begin{bmatrix} 0.1000 \\ -0.1000 \end{bmatrix}$$
Stochastic noise injected during sampling:
$$\mathbf{z}_3 = \begin{bmatrix} 0.5000 \\ 0.2000 \end{bmatrix}, \qquad \mathbf{z}_2 = \begin{bmatrix} -0.3000 \\ 0.4000 \end{bmatrix}, \qquad \mathbf{z}_1 = \begin{bmatrix} 0.0000 \\ 0.0000 \end{bmatrix} \quad (\text{deterministic at final step})$$
Execute the complete reverse sampling pass from $\mathbf{x}_3 \to \mathbf{x}_2 \to \mathbf{x}_1 \to \mathbf{x}_0$ step-by-step.

**Step-by-Step Solution:**

**1. Reverse Step $t = 3 \to 2$:**
- Constants at $t = 3$:
  $$\sqrt{\alpha_3} = \sqrt{0.70} \approx 0.836660, \quad \sqrt{1 - \bar{\alpha}_3} = \sqrt{1 - 0.5040} = \sqrt{0.4960} \approx 0.704273$$
  $$\text{Denoising factor: } \frac{\beta_3}{\sqrt{1 - \bar{\alpha}_3}} = \frac{0.30}{0.704273} \approx 0.425971$$
- Predicted noise subtraction:
  $$\mathbf{x}_3 - \frac{\beta_3}{\sqrt{1 - \bar{\alpha}_3}} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_3, 3) = \begin{bmatrix} 1.2000 - 0.425971(0.8000) \\ -0.8000 - 0.425971(-0.5000) \end{bmatrix} = \begin{bmatrix} 1.2000 - 0.340777 \\ -0.8000 + 0.212986 \end{bmatrix} = \begin{bmatrix} 0.859223 \\ -0.587014 \end{bmatrix}$$
- Posterior mean $\tilde{\boldsymbol{\mu}}_3$:
  $$\tilde{\boldsymbol{\mu}}_3 = \frac{1}{0.836660} \begin{bmatrix} 0.859223 \\ -0.587014 \end{bmatrix} = \begin{bmatrix} \mathbf{1.026968} \\ \mathbf{-0.701616} \end{bmatrix}$$
- Adding reverse stochastic noise:
  $$\mathbf{x}_2 = \tilde{\boldsymbol{\mu}}_3 + \sqrt{\tilde{\beta}_3} \mathbf{z}_3 = \begin{bmatrix} 1.026968 \\ -0.701616 \end{bmatrix} + 0.411527 \begin{bmatrix} 0.5000 \\ 0.2000 \end{bmatrix} = \begin{bmatrix} 1.026968 + 0.205764 \\ -0.701616 + 0.082305 \end{bmatrix} = \begin{bmatrix} \mathbf{1.232732} \\ \mathbf{-0.619311} \end{bmatrix}$$

**2. Reverse Step $t = 2 \to 1$:**
- Constants at $t = 2$:
  $$\sqrt{\alpha_2} = \sqrt{0.80} \approx 0.894427, \quad \sqrt{1 - \bar{\alpha}_2} = \sqrt{1 - 0.7200} = \sqrt{0.2800} \approx 0.529150$$
  $$\text{Denoising factor: } \frac{\beta_2}{\sqrt{1 - \bar{\alpha}_2}} = \frac{0.20}{0.529150} \approx 0.377964$$
- Predicted noise subtraction:
  $$\mathbf{x}_2 - \frac{\beta_2}{\sqrt{1 - \bar{\alpha}_2}} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_2, 2) = \begin{bmatrix} 1.232732 - 0.377964(0.4000) \\ -0.619311 - 0.377964(-0.3000) \end{bmatrix} = \begin{bmatrix} 1.232732 - 0.151186 \\ -0.619311 + 0.113389 \end{bmatrix} = \begin{bmatrix} 1.081546 \\ -0.505922 \end{bmatrix}$$
- Posterior mean $\tilde{\boldsymbol{\mu}}_2$:
  $$\tilde{\boldsymbol{\mu}}_2 = \frac{1}{0.894427} \begin{bmatrix} 1.081546 \\ -0.505922 \end{bmatrix} = \begin{bmatrix} \mathbf{1.209206} \\ \mathbf{-0.565638} \end{bmatrix}$$
- Adding reverse stochastic noise:
  $$\mathbf{x}_1 = \tilde{\boldsymbol{\mu}}_2 + \sqrt{\tilde{\beta}_2} \mathbf{z}_2 = \begin{bmatrix} 1.209206 \\ -0.565638 \end{bmatrix} + 0.267261 \begin{bmatrix} -0.3000 \\ 0.4000 \end{bmatrix} = \begin{bmatrix} 1.209206 - 0.080178 \\ -0.565638 + 0.106904 \end{bmatrix} = \begin{bmatrix} \mathbf{1.129028} \\ \mathbf{-0.458734} \end{bmatrix}$$

**3. Final Reverse Step $t = 1 \to 0$:**
- Constants at $t = 1$:
  $$\sqrt{\alpha_1} = \sqrt{0.90} \approx 0.948683, \quad \sqrt{1 - \bar{\alpha}_1} = \sqrt{0.1000} \approx 0.316228$$
  $$\text{Denoising factor: } \frac{\beta_1}{\sqrt{1 - \bar{\alpha}_1}} = \frac{0.10}{0.316228} \approx 0.316228$$
- Predicted noise subtraction:
  $$\mathbf{x}_1 - \frac{\beta_1}{\sqrt{1 - \bar{\alpha}_1}} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_1, 1) = \begin{bmatrix} 1.129028 - 0.316228(0.1000) \\ -0.458734 - 0.316228(-0.1000) \end{bmatrix} = \begin{bmatrix} 1.129028 - 0.031623 \\ -0.458734 + 0.031623 \end{bmatrix} = \begin{bmatrix} 1.097405 \\ -0.427111 \end{bmatrix}$$
- Final deterministic clean sample ($\mathbf{z}_1 = \mathbf{0}$):
  $$\mathbf{x}_0 = \frac{1}{0.948683} \begin{bmatrix} 1.097405 \\ -0.427111 \end{bmatrix} = \begin{bmatrix} \mathbf{1.156767} \\ \mathbf{-0.450215} \end{bmatrix}$$
From pure Gaussian static $\mathbf{x}_3 = [1.20, -0.80]^\top$, the 3 reverse denoising steps synthesized a crisp data point $\mathbf{x}_0 = [1.1568, -0.4502]^\top$!

---

### Illustration 4: Score-Based Continuous Diffusion SDE vs. Probability Flow ODE Comparison

**Problem:**
Consider continuous-time diffusion at time $t = 0.50$.
Current 2D state vector:
$$\mathbf{x}(t) = \begin{bmatrix} 1.50 \\ -2.00 \end{bmatrix}$$
Linear Variance-Preserving (VP) SDE schedule:
$$\beta(t) = 0.10 + 0.90 t \implies \beta(0.50) = 0.10 + 0.90(0.50) = \mathbf{0.5500}$$
The score network evaluates to:
$$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, 0.50) = \begin{bmatrix} -1.2000 \\ 1.8000 \end{bmatrix}$$
1. Compute the forward SDE drift vector $\mathbf{f}_{\text{fwd}}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}$ and diffusion coefficient $g(t) = \sqrt{\beta(t)}$.
2. Compute the reverse SDE drift vector $\mathbf{f}_{\text{rev}}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x} - \beta(t)\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t)$.
3. Compute the Probability Flow ODE velocity vector $\mathbf{v}_{\text{ODE}}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x} - \frac{1}{2}\beta(t)\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t)$.
4. For step size $\Delta t = 0.05$ and injected standard normal noise $\mathbf{z} = [0.40, -0.60]^\top$, compute:
   - One reverse SDE Euler-Maruyama step: $\mathbf{x}_{\text{SDE}}(t - \Delta t) = \mathbf{x}(t) - \mathbf{f}_{\text{rev}} \Delta t + g(t) \sqrt{\Delta t} \mathbf{z}$.
   - One Probability Flow ODE Euler step: $\mathbf{x}_{\text{ODE}}(t - \Delta t) = \mathbf{x}(t) - \mathbf{v}_{\text{ODE}} \Delta t$.

**Step-by-Step Solution:**

**1. Forward SDE Terms:**
$$\mathbf{f}_{\text{fwd}}(\mathbf{x}, t) = -\frac{1}{2}(0.5500) \begin{bmatrix} 1.50 \\ -2.00 \end{bmatrix} = -0.2750 \begin{bmatrix} 1.50 \\ -2.00 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.4125} \\ \mathbf{+0.5500} \end{bmatrix}$$
$$g(t) = \sqrt{\beta(t)} = \sqrt{0.5500} \approx \mathbf{0.741620}$$

**2. Reverse SDE Drift:**
$$\mathbf{f}_{\text{rev}}(\mathbf{x}, t) = \mathbf{f}_{\text{fwd}}(\mathbf{x}, t) - \beta(t) \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) = \begin{bmatrix} -0.4125 \\ 0.5500 \end{bmatrix} - 0.5500 \begin{bmatrix} -1.2000 \\ 1.8000 \end{bmatrix}$$
$$= \begin{bmatrix} -0.4125 - (-0.6600) \\ 0.5500 - 0.9900 \end{bmatrix} = \begin{bmatrix} -0.4125 + 0.6600 \\ 0.5500 - 0.9900 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.2475} \\ \mathbf{-0.4400} \end{bmatrix}$$

**3. Probability Flow ODE Velocity:**
$$\mathbf{v}_{\text{ODE}}(\mathbf{x}, t) = \mathbf{f}_{\text{fwd}}(\mathbf{x}, t) - \frac{1}{2} \beta(t) \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) = \begin{bmatrix} -0.4125 \\ 0.5500 \end{bmatrix} - 0.2750 \begin{bmatrix} -1.2000 \\ 1.8000 \end{bmatrix}$$
$$= \begin{bmatrix} -0.4125 - (-0.3300) \\ 0.5500 - 0.4950 \end{bmatrix} = \begin{bmatrix} -0.4125 + 0.3300 \\ 0.5500 - 0.4950 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.0825} \\ \mathbf{+0.0550} \end{bmatrix}$$

**4. Update Step Arithmetic ($\Delta t = 0.05$):**
- **Stochastic Reverse SDE Step:**
  $$\text{Deterministic drift update: } \mathbf{x} - \mathbf{f}_{\text{rev}} \Delta t = \begin{bmatrix} 1.50 \\ -2.00 \end{bmatrix} - 0.05 \begin{bmatrix} 0.2475 \\ -0.4400 \end{bmatrix} = \begin{bmatrix} 1.50 - 0.012375 \\ -2.00 + 0.022000 \end{bmatrix} = \begin{bmatrix} 1.487625 \\ -1.978000 \end{bmatrix}$$
  $$\text{Diffusion noise term: } g(t) \sqrt{\Delta t} \mathbf{z} = 0.741620 \times \sqrt{0.05} \begin{bmatrix} 0.40 \\ -0.60 \end{bmatrix} = 0.741620(0.223607) \begin{bmatrix} 0.40 \\ -0.60 \end{bmatrix} = 0.165831 \begin{bmatrix} 0.40 \\ -0.60 \end{bmatrix} = \begin{bmatrix} +0.066332 \\ -0.099499 \end{bmatrix}$$
  Summing:
  $$\mathbf{x}_{\text{SDE}}(0.45) = \begin{bmatrix} 1.487625 + 0.066332 \\ -1.978000 - 0.099499 \end{bmatrix} = \begin{bmatrix} \mathbf{1.553957} \\ \mathbf{-2.077499} \end{bmatrix}$$
- **Deterministic Probability Flow ODE Step:**
  $$\mathbf{x}_{\text{ODE}}(0.45) = \mathbf{x}(t) - \mathbf{v}_{\text{ODE}} \Delta t = \begin{bmatrix} 1.50 \\ -2.00 \end{bmatrix} - 0.05 \begin{bmatrix} -0.0825 \\ 0.0550 \end{bmatrix} = \begin{bmatrix} 1.50 - (-0.004125) \\ -2.00 - 0.002750 \end{bmatrix} = \begin{bmatrix} \mathbf{1.504125} \\ \mathbf{-2.002750} \end{bmatrix}$$

**Mathematical Comparison:**
The Probability Flow ODE trajectory is completely deterministic: reversing the sign of $\Delta t$ in an ODE solver maps $\mathbf{x}_{\text{ODE}}$ back to the exact initial noise $\mathbf{x}(T)$, enabling exact image inversion, latent editing, and exact likelihood calculation. In contrast, the SDE path is stochastic, exploring varied diffusion modes across repeated runs.

---

### Illustration 5: Exact Numerical Trace of Classifier-Free Guidance across Multiple Guidance Scales

**Problem:**
Consider a scalar diffusion state $x_t = 1.00$ at timestep $t$, with cumulative variance coefficients:
$$\sqrt{\bar{\alpha}_t} = 0.8000, \qquad \sqrt{1 - \bar{\alpha}_t} = 0.6000$$
The model evaluates:
- Unconditional prediction: $\epsilon_{\boldsymbol{\theta}}(x_t, \emptyset) = 0.5000$
- Conditional prediction for target prompt $c$: $\epsilon_{\boldsymbol{\theta}}(x_t, c) = 0.9000$
1. Compute the conditioning noise gradient $\Delta \epsilon = \epsilon_{\boldsymbol{\theta}}(x_t, c) - \epsilon_{\boldsymbol{\theta}}(x_t, \emptyset)$.
2. Compute the guided noise prediction:
   $$\tilde{\epsilon}(s) = \epsilon_{\boldsymbol{\theta}}(x_t, \emptyset) + s \cdot \Delta \epsilon$$
   for guidance scales $s \in \{0.0, 1.0, 3.0, 7.5, 15.0\}$.
3. Compute the Tweedie single-step clean reconstruction:
   $$\hat{x}_0(s) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \tilde{\epsilon}(s)}{\sqrt{\bar{\alpha}_t}} = \frac{1.0000 - 0.6000 \, \tilde{\epsilon}(s)}{0.8000}$$
4. Tabulate all quantities and mathematically explain the phase transition from unconditional generation to mode collapse and color-burn saturation.

**Step-by-Step Solution:**

**1. Conditioning Noise Gradient:**
$$\Delta \epsilon = \epsilon_{\boldsymbol{\theta}}(x_t, c) - \epsilon_{\boldsymbol{\theta}}(x_t, \emptyset) = 0.9000 - 0.5000 = \mathbf{+0.4000}$$

**2. Calculation Across Scales $s$:**

- **Scale $s = 0.0$ (Pure Unconditional):**
  $$\tilde{\epsilon}(0.0) = 0.5000 + 0.0(0.4000) = \mathbf{0.5000}$$
  $$\hat{x}_0(0.0) = \frac{1.0000 - 0.6000(0.5000)}{0.8000} = \frac{1.0000 - 0.3000}{0.8000} = \frac{0.7000}{0.8000} = \mathbf{0.8750}$$

- **Scale $s = 1.0$ (Standard Conditional Generation, No Guidance):**
  $$\tilde{\epsilon}(1.0) = 0.5000 + 1.0(0.4000) = \mathbf{0.9000}$$
  $$\hat{x}_0(1.0) = \frac{1.0000 - 0.6000(0.9000)}{0.8000} = \frac{1.0000 - 0.5400}{0.8000} = \frac{0.4600}{0.8000} = \mathbf{0.5750}$$

- **Scale $s = 3.0$ (Mild Creative Guidance):**
  $$\tilde{\epsilon}(3.0) = 0.5000 + 3.0(0.4000) = 0.5000 + 1.2000 = \mathbf{1.7000}$$
  $$\hat{x}_0(3.0) = \frac{1.0000 - 0.6000(1.7000)}{0.8000} = \frac{1.0000 - 1.0200}{0.8000} = \frac{-0.0200}{0.8000} = \mathbf{-0.0250}$$

- **Scale $s = 7.5$ (Standard Production Setting in Stable Diffusion):**
  $$\tilde{\epsilon}(7.5) = 0.5000 + 7.5(0.4000) = 0.5000 + 3.0000 = \mathbf{3.5000}$$
  $$\hat{x}_0(7.5) = \frac{1.0000 - 0.6000(3.5000)}{0.8000} = \frac{1.0000 - 2.1000}{0.8000} = \frac{-1.1000}{0.8000} = \mathbf{-1.3750}$$

- **Scale $s = 15.0$ (Extreme Guidance - Artifact Regime):**
  $$\tilde{\epsilon}(15.0) = 0.5000 + 15.0(0.4000) = 0.5000 + 6.0000 = \mathbf{6.5000}$$
  $$\hat{x}_0(15.0) = \frac{1.0000 - 0.6000(6.5000)}{0.8000} = \frac{1.0000 - 3.9000}{0.8000} = \frac{-2.9000}{0.8000} = \mathbf{-3.6250}$$

**Summary Comparison Table:**

| Guidance Scale $s$ | Guided Noise $\tilde{\epsilon}(s)$ | Clean Estimate $\hat{x}_0(s)$ | Implied Score $\nabla \log p_s$ | Perceptual & Mathematical Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **$s = 0.0$** | $0.5000$ | $+0.8750$ | $-0.8333$ | Pure unconditional generation; ignores prompt $c$. |
| **$s = 1.0$** | $0.9000$ | $+0.5750$ | $-1.5000$ | Standard Bayes posterior; high sample diversity. |
| **$s = 3.0$** | $1.7000$ | $-0.0250$ | $-2.8333$ | Enhanced prompt adherence; sharpens major modes. |
| **$s = 7.5$** | $\mathbf{3.5000}$ | $\mathbf{-1.3750}$ | $\mathbf{-5.8333}$ | **Optimal visual quality**; maximum text alignment. |
| **$s = 15.0$** | $6.5000$ | $-3.6250$ | $-10.8333$ | **Oversaturated / Burned**; exceeds dynamic pixel range $[-1, 1]$. |

**Mathematical Insight:**
Natural normalized images have pixel values bounded in $[-1, 1]$.
At $s = 7.5$, the estimated clean pixel value reaches $-1.375$, which is lightly clamped by dynamic range thresholding.
However, at $s = 15.0$, $\hat{x}_0$ overshoots violently to $-3.625$. When passed through subsequent denoising steps, the network activations saturate at extreme values, causing severe chromatic aberration, loss of textural micro-gradients, and cartoonish "burned" contrast artifacts.

---

## 7. Deep Learning Connection & Application

### State-of-the-Art Production Architectures (2024–2026)
1. **Latent Diffusion Models (Stable Diffusion 1.5, SDXL, SD 3):**
   - Employs an AutoencoderKL backbone to compress pixel space $8\times$.
   - Uses Cross-Attention in the UNet / Diffusion Transformer (DiT) to condition on CLIP / T5 text embeddings.
2. **Diffusion Transformers (DiT - Peebles & Xie, 2023):**
   - Replaces convolutional UNets entirely with standard Vision Transformer backbones operating on flattened latent patches.
   - Powers OpenAI Sora, Google Veo, and Black Forest Labs FLUX.1.
3. **Flow Matching & Rectified Flow (Lipman et al., Liu et al.):**
   - Straightens the curved SDE diffusion trajectories into straight Euclidean lines:
     $$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = \mathbf{x}_1 - \mathbf{x}_0$$
   - Reduces sampling steps from 50 to as few as 4 to 8 steps!

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Exact numerical verification of Part 5 hand calculations:
   - Closed-form noisy state $\mathbf{x}_2$ derivation.
   - Simplified loss evaluation.
   - Posterior mean $\tilde{\boldsymbol{\mu}}_2$, variance $\tilde{\beta}_2$, and reverse sample $\mathbf{x}_1$.
   - CFG vector extrapolation matching manual arithmetic to $< 10^{-6}$.
2. Closed-form marginal equivalence verification: comparing step-by-step Markov simulation against $\mathcal{O}(1)$ closed-form sampling.
3. Tweedie's formula score function consistency test: verifying $\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = -\boldsymbol{\epsilon} / \sqrt{1 - \bar{\alpha}_t}$.
4. Full end-to-end DDPM training loop and reverse sampling generation on a 2D synthetic manifold.

See implementation in:
[`10_generative_models/code/04_diffusion_models_ddpm_and_score_sdes.py`](./code/04_diffusion_models_ddpm_and_score_sdes.py)
