# Diffusion Models: DDPM, Score-Based SDEs & Classifier-Free Guidance

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

### 2.5 Unification with Score-Based Generative Modeling & SDEs (Song et al., 2020)

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

### 2.6 Classifier-Free Guidance (CFG - Ho & Salimans, 2021)

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
