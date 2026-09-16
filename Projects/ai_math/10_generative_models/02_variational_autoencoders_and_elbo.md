# Variational Autoencoders (VAEs) & ELBO Derivation

---

## 1. Intuition & 101 Motivation

### The Fatal Flaw of Deterministic Autoencoders
In a classical Autoencoder (AE), an encoder network compresses an input $\mathbf{x} \in \mathbb{R}^D$ into a deterministic latent code $\mathbf{z} = f_{\boldsymbol{\phi}}(\mathbf{x}) \in \mathbb{R}^d$, and a decoder reconstructs $\hat{\mathbf{x}} = g_{\boldsymbol{\theta}}(\mathbf{z})$.
While standard autoencoders excel at dimensionality reduction and denoising, **they fail fundamentally as generative models**:
1. **Discontinuous Latent Space:** Because the network is trained solely on reconstruction loss $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$, the latent codes of training points collapse into isolated, arbitrary islands in $\mathbb{R}^d$.
2. **The "Empty Space" Problem:** If you draw a random sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and pass it to the decoder, the sample almost certainly falls into an unmapped "hole" between training manifolds, decoding into meaningless, blurry noise.

```
       DETERMINISTIC AUTOENCODER               VARIATIONAL AUTOENCODER (VAE)
     (Gaps, holes & non-generative)           (Continuous, dense & generative)
            Latent Space z                           Latent Space z
         ▲                                        ▲
         │   [Dog island]                         │      (Dog cluster)
         │       ●●●                              │        ╭─────╮
         │                                        │       │  ●●●  │  <-- Smooth
         │         ? <-- Draw z here?             │        ╰──┼──╯       overlap
         │               Decode = Garbage!        │         ╭─┴───╮
         │                                        │        │  ■■■  │
         │       ■■■                              │         ╰─────╯
         │   [Cat island]                         │      (Cat cluster)
         └───────────────────────►                └───────────────────────►
```

### The Variational Insight: Encoders as Distribution Producers
Rather than mapping $\mathbf{x}$ to a single static vector $\mathbf{z}$, a **Variational Autoencoder (Kingma & Welling, 2013)** maps $\mathbf{x}$ to a probability distribution:
$$q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\left(\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \operatorname{diag}\left(\boldsymbol{\sigma}_{\boldsymbol{\phi}}^2(\mathbf{x})\right)\right)$$
To generate an output, we sample $\mathbf{z} \sim q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ and reconstruct $\hat{\mathbf{x}} = g_{\boldsymbol{\theta}}(\mathbf{z})$.

To guarantee that the entire latent space is continuous, smooth, and easily sampleable at inference time, we force $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ to be close to a standard isotropic Gaussian prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ via a **Kullback-Leibler (KL) divergence penalty**.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Intractable Marginal Likelihood
Let $\mathbf{x} \in \mathcal{X}$ be observed data and $\mathbf{z} \in \mathcal{Z}$ be continuous latent variables.
The true generative process defines a joint distribution:
$$p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) = p(\mathbf{z}) p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$
where $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ is the prior, and $p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$ is the likelihood parameterized by decoder network $\boldsymbol{\theta}$.

We wish to maximize the marginal log-likelihood (evidence) across the dataset:
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log \int_{\mathcal{Z}} p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) d\mathbf{z} = \log \int_{\mathcal{Z}} p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

**Why is this integral intractable?**
In high dimensions (e.g., $d = 128$), the space of $\mathbf{z}$ is vast. For a given image $\mathbf{x}$, only an infinitesimal fraction of $\mathcal{Z}$ will generate $\mathbf{x}$. A simple Monte Carlo average $\frac{1}{S} \sum_s p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}^{(s)})$ with $\mathbf{z}^{(s)} \sim p(\mathbf{z})$ will almost always evaluate to $0$, yielding infinite variance.

Furthermore, the true posterior is intractable by Bayes' rule:
$$p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) = \frac{p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{x})} = \frac{p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{\int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}') d\mathbf{z}'}$$

---

### 2.2 Complete First-Principles Derivation of the Evidence Lower Bound (ELBO)

We introduce a tractable variational recognition model (encoder) $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ to approximate the true posterior $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$.

#### Method A: Derivation via KL Divergence of the Posterior
Consider the KL divergence between variational posterior $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ and true posterior $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$:
$$D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) \right) = \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \log \left( \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})} \right) d\mathbf{z}$$

Substitute Bayes' theorem $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) = \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{x})}$:
$$\begin{aligned}
D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) \right) &= \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \log \left( \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) p_{\boldsymbol{\theta}}(\mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})} \right) d\mathbf{z} \\
&= \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \left[ \log p_{\boldsymbol{\theta}}(\mathbf{x}) + \log \left( \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})} \right) \right] d\mathbf{z} \\
&= \log p_{\boldsymbol{\theta}}(\mathbf{x}) \underbrace{\int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) d\mathbf{z}}_{= 1} + \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \log \left( \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})} \right) d\mathbf{z} \\
&= \log p_{\boldsymbol{\theta}}(\mathbf{x}) - \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[ \log \left( \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right) \right]
\end{aligned}$$

Rearranging for the marginal log-likelihood $\log p_{\boldsymbol{\theta}}(\mathbf{x})$:
$$\mathbf{\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[ \log \left( \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right) \right]}_{\text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x})} + \underbrace{D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) \right)}_{\ge 0}}$$

Since the Kullback-Leibler divergence is non-negative ($D_{\text{KL}} \ge 0$ by Gibbs' inequality):
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) \ge \text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x})$$

#### Method B: Derivation via Jensen's Inequality
$$\begin{aligned}
\log p_{\boldsymbol{\theta}}(\mathbf{x}) &= \log \int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) d\mathbf{z} \\
&= \log \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} d\mathbf{z} \\
&= \log \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \left[ \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right] \\
&\ge \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right] \quad (\text{by concavity of } \log)
\end{aligned}$$

---

### 2.3 Decomposition of the ELBO

Expand the joint distribution $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) = p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})$:
$$\begin{aligned}
\text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}) &= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[ \log \left( \frac{p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right) \right] \\
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[ \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) \right] + \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[ \log \frac{p(\mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right] \\
&= \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[ \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) \right]}_{\text{Reconstruction Term}} - \underbrace{D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}) \right)}_{\text{Prior Regularization Term}}
\end{aligned}$$

Thus, maximizing the ELBO balances two opposing physical forces:
1. **Reconstruction Term:** Encourages the decoder to reconstruct $\mathbf{x}$ faithfully from latent samples $\mathbf{z}$.
2. **KL Regularization Term:** Compresses the latent representation by penalizing deviations of the posterior from standard normal prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$.

To minimize loss in deep learning frameworks, we define:
$$\mathcal{L}_{\text{VAE}}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}) = -\text{ELBO} = -\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[ \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) \right] + D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}) \right)$$

---

### 2.4 Analytical Derivation of Gaussian KL Divergence

Let the approximate posterior be a diagonal Gaussian:
$$q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\left(\boldsymbol{\mu}, \operatorname{diag}\left(\boldsymbol{\sigma}^2\right)\right)$$
and the prior be standard normal:
$$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$
where $\mathbf{z} \in \mathbb{R}^d$. Because components are independent, the total KL divergence is the sum over individual dimensions $j \in \{1, \dots, d\}$:
$$D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}) \right) = \sum_{j=1}^d D_{\text{KL}}\left( \mathcal{N}(\mu_j, \sigma_j^2) \,\|\, \mathcal{N}(0, 1) \right)$$

For 1D Gaussians $q(z) = \mathcal{N}(\mu, \sigma^2)$ and $p(z) = \mathcal{N}(0, 1)$:
$$\begin{aligned}
D_{\text{KL}}(q \,\|\, p) &= \int_{-\infty}^\infty q(z) \log \frac{q(z)}{p(z)} dz = \mathbb{E}_q\left[ \log q(z) - \log p(z) \right] \\
\log q(z) &= -\frac{1}{2} \log(2\pi) - \frac{1}{2} \log(\sigma^2) - \frac{(z - \mu)^2}{2\sigma^2} \\
\log p(z) &= -\frac{1}{2} \log(2\pi) - \frac{z^2}{2}
\end{aligned}$$

Subtracting:
$$\log q(z) - \log p(z) = -\frac{1}{2} \log(\sigma^2) - \frac{(z - \mu)^2}{2\sigma^2} + \frac{z^2}{2}$$

Taking the expectation $\mathbb{E}_{z \sim q}$:
1. $\mathbb{E}_q\left[ -\frac{1}{2} \log(\sigma^2) \right] = -\frac{1}{2} \log(\sigma^2)$
2. $\mathbb{E}_q\left[ \frac{(z - \mu)^2}{2\sigma^2} \right] = \frac{\operatorname{Var}(z)}{2\sigma^2} = \frac{\sigma^2}{2\sigma^2} = \frac{1}{2}$
3. $\mathbb{E}_q\left[ \frac{z^2}{2} \right] = \frac{1}{2} \left( \operatorname{Var}(z) + (\mathbb{E}[z])^2 \right) = \frac{1}{2} (\sigma^2 + \mu^2)$

Combining all three terms:
$$D_{\text{KL}}\left( \mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1) \right) = -\frac{1}{2} \log(\sigma^2) - \frac{1}{2} + \frac{1}{2} \sigma^2 + \frac{1}{2} \mu^2 = -\frac{1}{2} \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)$$

Summing over all $d$ dimensions:
$$\mathbf{D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}) \right) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)}$$

> **Numerical Stability Rule:** Encoders never output $\sigma_j$ directly (since $\sigma_j > 0$ strictly). Instead, the encoder outputs unconstrained log-variance $\gamma_j = \log(\sigma_j^2)$. The formula becomes:
> $$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \gamma_j - \mu_j^2 - e^{\gamma_j} \right)$$

---

### 2.5 The Reparameterization Trick

#### The Bottleneck of Stochastic Backpropagation
We must optimize encoder parameters $\boldsymbol{\phi}$ using gradient descent:
$$\nabla_{\boldsymbol{\phi}} \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \left[ f(\mathbf{z}) \right]$$
where $f(\mathbf{z}) = \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$.
Because the sampling distribution $q_{\boldsymbol{\phi}}$ depends on $\boldsymbol{\phi}$, we cannot simply push $\nabla_{\boldsymbol{\phi}}$ inside the expectation:
$$\nabla_{\boldsymbol{\phi}} \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) f(\mathbf{z}) d\mathbf{z} = \int \nabla_{\boldsymbol{\phi}} q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) f(\mathbf{z}) d\mathbf{z}$$
The standard score-function estimator (REINFORCE):
$$\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}}[f(\mathbf{z})] = \mathbb{E}_{q_{\boldsymbol{\phi}}}\left[ f(\mathbf{z}) \nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \right]$$
suffers from catastrophic, unusable variance in practice.

```
       DIRECT STOCHASTIC NODE                     REPARAMETERIZATION TRICK
    (Gradient cannot pass through)               (Gradient flows smoothly!)

            x                                                 x
            │                                                 │
      [ Encoder phi ]                                   [ Encoder phi ]
            │                                                 │
       mu, sigma                                          mu, sigma
            │                                                 │   eps ~ N(0, I)
            ▼                                                 │       │
       ( z ~ N )  <-- NO GRADIENT!                            ▼       ▼
            │                                           [ z = mu + sigma * eps ]
            ▼                                                 │
      [ Decoder theta ]                                       ▼
            │                                           [ Decoder theta ]
            ▼                                                 │
          x_hat                                               ▼
                                                            x_hat
```

#### The Pathwise Derivative Solution
We isolate the randomness into an independent auxiliary noise variable $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
We express $\mathbf{z}$ as a deterministic, differentiable transformation:
$$\mathbf{z} = g_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon}) = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x}) \odot \boldsymbol{\epsilon}$$
where $\odot$ denotes element-wise Hadamard multiplication.

By the Law of the Unconscious Statistician:
$$\mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \left[ f(\mathbf{z}) \right] = \mathbb{E}_{\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})} \left[ f\left( g_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon}) \right) \right]$$

Now, because the distribution $p(\boldsymbol{\epsilon}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ has **zero dependence** on network parameters $\boldsymbol{\phi}$, the gradient moves directly inside the expectation:
$$\mathbf{\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}}[f(\mathbf{z})] = \mathbb{E}_{p(\boldsymbol{\epsilon})} \left[ \nabla_{\mathbf{z}} f(\mathbf{z}) \cdot \nabla_{\boldsymbol{\phi}} g_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon}) \right]}$$

Using a single Monte Carlo sample $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ per data point:
$$\frac{\partial z_j}{\partial \mu_j} = 1, \quad \frac{\partial z_j}{\partial \sigma_j} = \epsilon_j, \quad \frac{\partial z_j}{\partial \gamma_j} = \frac{\partial (\mu_j + e^{\gamma_j/2} \epsilon_j)}{\partial \gamma_j} = \frac{1}{2} e^{\gamma_j/2} \epsilon_j = \frac{1}{2} \sigma_j \epsilon_j$$
Gradient propagation through $\mathbf{z}$ to the encoder is now completely exact, deterministic, and low-variance!

---

### 2.6 Decoder Likelihood Formulations

Depending on the nature of the data $\mathbf{x}$, the reconstruction loss corresponds to specific log-likelihood distributions:

1. **Gaussian Decoder with Fixed Identity Covariance $\sigma_{\text{dec}}^2 \mathbf{I}$:**
   $$p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}\left(\hat{\mathbf{x}}_{\boldsymbol{\theta}}(\mathbf{z}), \sigma_{\text{dec}}^2 \mathbf{I}\right)$$
   $$\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) = -\frac{D}{2} \log(2\pi \sigma_{\text{dec}}^2) - \frac{1}{2\sigma_{\text{dec}}^2} \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$
   For $\sigma_{\text{dec}}^2 = 1$, maximizing log-likelihood is identical to minimizing **Mean Squared Error (MSE)**:
   $$\mathcal{L}_{\text{recon}} = \frac{1}{2} \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

2. **Bernoulli Decoder (Binary or Normalized Grayscale Pixels $x_i \in [0, 1]$):**
   $$p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) = \prod_{i=1}^D \hat{x}_i^{x_i} (1 - \hat{x}_i)^{1 - x_i}, \quad \hat{x}_i = \sigma\left( \text{decoder}(\mathbf{z})_i \right)$$
   $$\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) = \sum_{i=1}^D \left[ x_i \log \hat{x}_i + (1 - x_i) \log (1 - \hat{x}_i) \right]$$
   Maximizing log-likelihood is identical to minimizing **Binary Cross-Entropy (BCE)**:
   $$\mathcal{L}_{\text{recon}} = -\sum_{i=1}^D \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]$$

---

### 2.7 Posterior Collapse & $\beta$-VAE

#### The Posterior Collapse Pathology
When the decoder network is excessively expressive (e.g., an autoregressive PixelCNN or deep Transformer decoder), it can generate plausible data $\mathbf{x}$ using only past context without needing the latent code $\mathbf{z}$ at all.
In this case, the optimizer finds a degenerate shortcut:
$$\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) \to \mathbf{0}, \quad \boldsymbol{\sigma}_{\boldsymbol{\phi}}^2(\mathbf{x}) \to \mathbf{1} \implies D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}) \right) \to 0$$
The encoder shuts down completely, $q(\mathbf{z} \mid \mathbf{x}) = p(\mathbf{z})$, and the latent variables are completely ignored.

**Remedies for Posterior Collapse:**
1. **KL Annealing (Warmup):** Scale the KL term by a weighting coefficient $\beta_t$ that ramps smoothly from $0 \to 1$ over early training epochs, allowing the autoencoder to first learn meaningful reconstructions before regularizing latent space.
2. **Cyclical Annealing (Fu et al., 2019):** Periodically reset $\beta_t$ to $0$ in multiple cycles to escape local minima.
3. **Free Bits / Minimum KL:** Impose a floor on each dimension's KL divergence: $\max(\tau, D_{\text{KL}}(q_j \,\|\, p_j))$.

#### $\beta$-VAE (Higgins et al., 2017)
$$\mathcal{L}_{\beta\text{-VAE}} = -\mathbb{E}_{q}[\log p(\mathbf{x} \mid \mathbf{z})] + \beta D_{\text{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}) \right)$$
- **$\beta > 1$ (Disentanglement Regime):** Forces latent dimensions to become statistically independent, discovering interpretable generative factors (e.g., azimuth, elevation, stroke width).
- **$\beta < 1$ (High Fidelity Regime):** Prioritizes pixel-perfect reconstruction at the expense of latent space smoothness.

---

### 2.8 Rigorous Mathematical Derivations

#### Derivation 10.2.1: Tightness of the ELBO, Monotonicity under Variational Families, and the Variational EM Equivalence

**1. Context and Assumptions:**
Let $\mathbf{x} \in \mathcal{X}$ be observed data and $\mathbf{z} \in \mathcal{Z}$ be latent variables.
The true joint density is $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) = p(\mathbf{z}) p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$, parameterized by decoder weights $\boldsymbol{\theta} \in \Theta$.
We choose an approximate posterior distribution $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ from a family of distributions $\mathcal{Q} = \{q_{\boldsymbol{\phi}} : \boldsymbol{\phi} \in \Phi\}$.
Assume $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) > 0$ whenever $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) > 0$.

**2. Step 1: Exact Non-Negative Gap Identity:**
Expand the log marginal likelihood $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ using the Radon-Nikodym derivative with respect to $q_{\boldsymbol{\phi}}$:
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \int_{\mathcal{Z}} q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \log p_{\boldsymbol{\theta}}(\mathbf{x}) d\mathbf{z}$$
Since $p_{\boldsymbol{\theta}}(\mathbf{x}) = \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}$:
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \int_{\mathcal{Z}} q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \log \left( \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})} \cdot \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right) d\mathbf{z}$$
Splitting the logarithm:
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \int_{\mathcal{Z}} q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \log \left( \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right) d\mathbf{z} + \int_{\mathcal{Z}} q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \log \left( \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})} \right) d\mathbf{z}$$
By definition of the ELBO and KL divergence:
$$\mathbf{\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}) + D_{\mathrm{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) \right)}$$

**3. Step 2: Bound Tightness Condition:**
Because $D_{\mathrm{KL}}(q \,\|\, p) \ge 0$ with equality if and only if $q = p$ almost everywhere:
$$\text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}) \le \log p_{\boldsymbol{\theta}}(\mathbf{x}), \quad \text{with equality } \iff q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$$
Hence, the gap between the true marginal evidence and the ELBO is **identically equal to the approximation error of the variational posterior**.
If the variational family $\mathcal{Q}$ is unconstrained, setting $q^*(\mathbf{z} \mid \mathbf{x}) = p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$ makes the lower bound strictly tight.

**4. Step 3: Variational Expectation-Maximization (VEM) and Monotonic Ascent:**
The joint optimization of ELBO over $(\boldsymbol{\theta}, \boldsymbol{\phi})$ corresponds to coordinate ascent:
- **Variational E-Step (Fix $\boldsymbol{\theta}^{(t)}$, update $\boldsymbol{\phi}$):**
  $$\boldsymbol{\phi}^{(t+1)} = \arg\max_{\boldsymbol{\phi} \in \Phi} \text{ELBO}(\boldsymbol{\theta}^{(t)}, \boldsymbol{\phi}; \mathbf{x}) \equiv \arg\min_{\boldsymbol{\phi} \in \Phi} D_{\mathrm{KL}}\left( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p_{\boldsymbol{\theta}^{(t)}}(\mathbf{z} \mid \mathbf{x}) \right)$$
- **Variational M-Step (Fix $\boldsymbol{\phi}^{(t+1)}$, update $\boldsymbol{\theta}$):**
  $$\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta} \in \Theta} \text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}^{(t+1)}; \mathbf{x}) \equiv \arg\max_{\boldsymbol{\theta} \in \Theta} \mathbb{E}_{q_{\boldsymbol{\phi}^{(t+1)}}(\mathbf{z} \mid \mathbf{x})} \left[ \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) \right]$$

**Monotonicity Proof:**
$$\begin{aligned}
\log p_{\boldsymbol{\theta}^{(t)}}(\mathbf{x}) &= \text{ELBO}(\boldsymbol{\theta}^{(t)}, \boldsymbol{\phi}^{(t+1)}; \mathbf{x}) + D_{\mathrm{KL}}\left( q_{\boldsymbol{\phi}^{(t+1)}} \,\|\, p_{\boldsymbol{\theta}^{(t)}} \right) \\
&\le \text{ELBO}(\boldsymbol{\theta}^{(t+1)}, \boldsymbol{\phi}^{(t+1)}; \mathbf{x}) + D_{\mathrm{KL}}\left( q_{\boldsymbol{\phi}^{(t+1)}} \,\|\, p_{\boldsymbol{\theta}^{(t)}} \right) \quad (\text{by M-step optimality}) \\
&\le \text{ELBO}(\boldsymbol{\theta}^{(t+1)}, \boldsymbol{\phi}^{(t+1)}; \mathbf{x}) + D_{\mathrm{KL}}\left( q_{\boldsymbol{\phi}^{(t+1)}} \,\|\, p_{\boldsymbol{\theta}^{(t+1)}} \right) \quad (\text{since } D_{\mathrm{KL}} \ge 0) \\
&= \log p_{\boldsymbol{\theta}^{(t+1)}}(\mathbf{x})
\end{aligned}$$
Thus, every iteration of Variational EM is guaranteed to monotonically non-decrease the true marginal data evidence $\log p_{\boldsymbol{\theta}}(\mathbf{x})$.

---

#### Derivation 10.2.2: Variance Analysis of Gradient Estimators: Pathwise Reparameterization vs. Score Function (REINFORCE)

**1. Context and Assumptions:**
Consider the expectation $\mathcal{J}(\mu) = \mathbb{E}_{z \sim q_\mu}[f(z)]$, where $q_\mu(z) = \mathcal{N}(\mu, \sigma^2)$ with fixed variance $\sigma^2$ and mean $\mu$.
We evaluate the gradient $\frac{d \mathcal{J}}{d\mu}$ under a general quadratic cost function:
$$f(z) = \frac{1}{2} a z^2 + b z + c \quad (a, b, c \in \mathbb{R})$$
The true analytical derivative is:
$$\mathbb{E}[f(z)] = \frac{1}{2} a (\mu^2 + \sigma^2) + b \mu + c \implies \frac{d}{d\mu} \mathbb{E}[f(z)] = a \mu + b$$

**2. Step 1: Score-Function (REINFORCE) Estimator:**
Using the identity $\nabla_\mu q_\mu(z) = q_\mu(z) \nabla_\mu \log q_\mu(z)$:
$$\hat{g}_{\text{score}} = f(z) \frac{\partial \log q_\mu(z)}{\partial \mu}$$
Since $\log q_\mu(z) = -\frac{1}{2}\log(2\pi \sigma^2) - \frac{(z - \mu)^2}{2\sigma^2}$, we have:
$$\frac{\partial \log q_\mu(z)}{\partial \mu} = \frac{z - \mu}{\sigma^2} = \frac{\sigma \epsilon}{\sigma^2} = \frac{\epsilon}{\sigma}, \quad \text{where } z = \mu + \sigma \epsilon, \, \epsilon \sim \mathcal{N}(0, 1)$$
Substituting $f(z)$:
$$\hat{g}_{\text{score}} = \left[ \frac{1}{2} a (\mu + \sigma \epsilon)^2 + b(\mu + \sigma \epsilon) + c \right] \frac{\epsilon}{\sigma}$$
Expanding powers of $\epsilon$:
$$\hat{g}_{\text{score}} = \frac{1}{\sigma} \left[ \left(\frac{1}{2}a\mu^2 + b\mu + c\right) \epsilon + (a\mu\sigma + b\sigma)\epsilon^2 + \frac{1}{2} a \sigma^2 \epsilon^3 \right]$$
Taking expectations using standard normal moments $\mathbb{E}[\epsilon] = 0, \mathbb{E}[\epsilon^2] = 1, \mathbb{E}[\epsilon^3] = 0, \mathbb{E}[\epsilon^4] = 3, \mathbb{E}[\epsilon^6] = 15$:
$$\mathbb{E}[\hat{g}_{\text{score}}] = \frac{1}{\sigma} (a\mu\sigma + b\sigma) (1) = a\mu + b \quad (\text{Unbiased})$$
Now compute the variance $\operatorname{Var}(\hat{g}_{\text{score}}) = \mathbb{E}[\hat{g}_{\text{score}}^2] - (\mathbb{E}[\hat{g}_{\text{score}}])^2$:
Notice the term proportional to $c$:
$$\hat{g}_{\text{score}} \supset \frac{c}{\sigma} \epsilon \implies \operatorname{Var}(\hat{g}_{\text{score}}) \ge \frac{c^2}{\sigma^2}$$
As $\sigma \to 0$, the variance of the score function estimator diverges to infinity: $\mathcal{O}(\sigma^{-2})$. Furthermore, adding an arbitrary constant baseline $c \to c + K$ scales variance quadratically $\mathcal{O}(K^2)$.

**3. Step 2: Pathwise Reparameterization Estimator:**
Transform $z = g(\mu, \epsilon) = \mu + \sigma \epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$:
$$\hat{g}_{\text{rep}} = \frac{\partial f(z)}{\partial z} \cdot \frac{\partial z}{\partial \mu} = f'(\mu + \sigma \epsilon) \cdot 1$$
Differentiating $f(z) = \frac{1}{2} a z^2 + b z + c$:
$$f'(z) = a z + b = a(\mu + \sigma \epsilon) + b = (a\mu + b) + a\sigma \epsilon$$
Taking expectation:
$$\mathbb{E}[\hat{g}_{\text{rep}}] = (a\mu + b) + a\sigma \mathbb{E}[\epsilon] = a\mu + b \quad (\text{Unbiased})$$
Computing the variance:
$$\operatorname{Var}(\hat{g}_{\text{rep}}) = \operatorname{Var}((a\mu + b) + a\sigma \epsilon) = a^2 \sigma^2 \operatorname{Var}(\epsilon) = \mathbf{a^2 \sigma^2}$$

**4. Step 3: Comparative Analysis & Theoretical Implication:**
$$\frac{\operatorname{Var}(\hat{g}_{\text{score}})}{\operatorname{Var}(\hat{g}_{\text{rep}})} \approx \frac{\frac{f(\mu)^2}{\sigma^2}}{a^2 \sigma^2} = \frac{f(\mu)^2}{a^2 \sigma^4} \xrightarrow{\sigma \to 0} \infty$$
1. **Variance Vanishing:** For the reparameterization estimator, as posterior uncertainty decreases ($\sigma \to 0$), the gradient variance vanishes to **zero** ($\operatorname{Var} = a^2 \sigma^2 \to 0$).
2. **Invariance to Baselines:** $\operatorname{Var}(\hat{g}_{\text{rep}})$ is completely independent of constant shifts $c$ and linear shifts $b$.
3. **First-Order Linearity:** If $f(z)$ is linear ($a = 0$), $\operatorname{Var}(\hat{g}_{\text{rep}}) \equiv 0$ (the single-sample Monte Carlo estimator is deterministic and exact).
This explains why standard backpropagation with the reparameterization trick trains stably across millions of parameters, whereas REINFORCE collapses without aggressive baseline subtraction.

---

#### Derivation 10.2.3: General Multivariate Gaussian KL Divergence with Full Non-Diagonal Covariances

**1. Context and Assumptions:**
Let $\mathbf{z} \in \mathbb{R}^d$. Consider two multivariate normal distributions:
$$q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q), \qquad p(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$$
where $\boldsymbol{\mu}_q, \boldsymbol{\mu}_p \in \mathbb{R}^d$ and $\boldsymbol{\Sigma}_q, \boldsymbol{\Sigma}_p \in \mathbb{S}_{++}^d$ are symmetric positive-definite covariance matrices.

**2. Step 1: Log-Likelihood Ratio Expansion:**
The log-densities are given by:
$$\log q(\mathbf{z}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det \boldsymbol{\Sigma}_q - \frac{1}{2}(\mathbf{z} - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_q^{-1} (\mathbf{z} - \boldsymbol{\mu}_q)$$
$$\log p(\mathbf{z}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det \boldsymbol{\Sigma}_p - \frac{1}{2}(\mathbf{z} - \boldsymbol{\mu}_p)^\top \boldsymbol{\Sigma}_p^{-1} (\mathbf{z} - \boldsymbol{\mu}_p)$$
Subtracting the two expressions:
$$\log \frac{q(\mathbf{z})}{p(\mathbf{z})} = \frac{1}{2} \log \left( \frac{\det \boldsymbol{\Sigma}_p}{\det \boldsymbol{\Sigma}_q} \right) - \frac{1}{2}(\mathbf{z} - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_q^{-1} (\mathbf{z} - \boldsymbol{\mu}_q) + \frac{1}{2}(\mathbf{z} - \boldsymbol{\mu}_p)^\top \boldsymbol{\Sigma}_p^{-1} (\mathbf{z} - \boldsymbol{\mu}_p)$$

**3. Step 2: Expectation of Quadratic Forms via Trace Trick:**
For any symmetric matrix $\mathbf{A}$ and random vector $\mathbf{y}$ with mean $\mathbf{m}$ and covariance $\mathbf{C}$:
$$\mathbb{E}[\mathbf{y}^\top \mathbf{A} \mathbf{y}] = \mathbb{E}\left[ \operatorname{tr}(\mathbf{y}^\top \mathbf{A} \mathbf{y}) \right] = \mathbb{E}\left[ \operatorname{tr}(\mathbf{A} \mathbf{y}\mathbf{y}^\top) \right] = \operatorname{tr}\left( \mathbf{A} \mathbb{E}[\mathbf{y}\mathbf{y}^\top] \right) = \operatorname{tr}(\mathbf{A} \mathbf{C}) + \mathbf{m}^\top \mathbf{A} \mathbf{m}$$
Applying this identity to each quadratic term under expectation $\mathbb{E}_{\mathbf{z} \sim q}$:

- **First Quadratic Term:** Let $\mathbf{y} = \mathbf{z} - \boldsymbol{\mu}_q \implies \mathbb{E}[\mathbf{y}] = \mathbf{0}, \operatorname{Cov}(\mathbf{y}) = \boldsymbol{\Sigma}_q$:
  $$\mathbb{E}_{\mathbf{z} \sim q}\left[ (\mathbf{z} - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_q^{-1} (\mathbf{z} - \boldsymbol{\mu}_q) \right] = \operatorname{tr}\left( \boldsymbol{\Sigma}_q^{-1} \boldsymbol{\Sigma}_q \right) + \mathbf{0} = \operatorname{tr}(\mathbf{I}_d) = d$$

- **Second Quadratic Term:** Decompose $\mathbf{z} - \boldsymbol{\mu}_p = (\mathbf{z} - \boldsymbol{\mu}_q) + (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p)$:
  $$\mathbb{E}_{\mathbf{z} \sim q}\left[ (\mathbf{z} - \boldsymbol{\mu}_p)^\top \boldsymbol{\Sigma}_p^{-1} (\mathbf{z} - \boldsymbol{\mu}_p) \right] = \operatorname{tr}\left( \boldsymbol{\Sigma}_p^{-1} \boldsymbol{\Sigma}_q \right) + (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p)^\top \boldsymbol{\Sigma}_p^{-1} (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p)$$

**4. Step 3: Analytical General Form:**
Combining the terms:
$$\mathbf{D_{\mathrm{KL}}(q \,\|\, p) = \frac{1}{2} \left[ \operatorname{tr}\left( \boldsymbol{\Sigma}_p^{-1} \boldsymbol{\Sigma}_q \right) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_p^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) - d + \log \left( \frac{\det \boldsymbol{\Sigma}_p}{\det \boldsymbol{\Sigma}_q} \right) \right]}$$

**5. Step 4: Verification of Standard Isotropic Prior Specialization:**
Let the prior be standard isotropic normal: $\boldsymbol{\mu}_p = \mathbf{0}$ and $\boldsymbol{\Sigma}_p = \mathbf{I}_d$.
Let the approximate posterior be diagonal: $\boldsymbol{\Sigma}_q = \operatorname{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_d^2)$ and $\boldsymbol{\mu}_q = \boldsymbol{\mu}$.
Then:
1. $\operatorname{tr}(\boldsymbol{\Sigma}_p^{-1} \boldsymbol{\Sigma}_q) = \operatorname{tr}(\mathbf{I}_d \boldsymbol{\Sigma}_q) = \sum_{j=1}^d \sigma_j^2$
2. $(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_p^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) = (-\boldsymbol{\mu})^\top \mathbf{I} (-\boldsymbol{\mu}) = \sum_{j=1}^d \mu_j^2$
3. $\det \boldsymbol{\Sigma}_p = \det \mathbf{I} = 1 \implies \log \det \boldsymbol{\Sigma}_p = 0$
4. $\det \boldsymbol{\Sigma}_q = \prod_{j=1}^d \sigma_j^2 \implies \log \det \boldsymbol{\Sigma}_q = \sum_{j=1}^d \log(\sigma_j^2)$
Substituting into the general formula:
$$D_{\mathrm{KL}} = \frac{1}{2} \left[ \sum_{j=1}^d \sigma_j^2 + \sum_{j=1}^d \mu_j^2 - d - \sum_{j=1}^d \log(\sigma_j^2) \right] = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
which identically matches Kingma & Welling's diagonal Gaussian formulation in Section 2.4!

---

## 3. Geometric & Algebraic Interpretation

### Latent Manifold & Spherical Interpolation (SLERP)

The latent space $\mathcal{Z} \cong \mathbb{R}^d$ endowed with prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ exhibits an intriguing geometric phenomenon in high dimensions ($d \ge 64$):
By Gaussian concentration of measure (the Gaussian Annulus Theorem), almost all probability mass does not reside near the origin $\mathbf{0}$; it concentrates in a thin spherical shell of radius:
$$r \approx \sqrt{d}$$

```
                GAUSSIAN CONCENTRATION OF MEASURE
                    (The "Gaussian Soap Bubble")
                               d = 2
                               ┌───┐
                              ┌┘   └┐
                             ┌┘  ●  └┐   High mass at origin
                              └┐   ┌┘
                               └───┘

                              d = 128
                              ┌─────┐
                            ┌─┘     └─┐
                           ┌┘  Empty  └┐
                          ┌┘   origin  └┐  All probability mass
                          │       ○     │  concentrated on thin
                          └┐           ┌┘  surface shell: r = sqrt(d)
                           └┐         ┌┘
                            └─┐     ┌─┘
                              └─────┘
```

#### Why Linear Interpolation (LERP) Causes Blurriness
If we linearly interpolate between two latent vectors $\mathbf{z}_A$ and $\mathbf{z}_B$:
$$\mathbf{z}_{\text{lerp}}(\alpha) = (1 - \alpha)\mathbf{z}_A + \alpha \mathbf{z}_B, \quad \alpha \in [0, 1]$$
At the midpoint $\alpha = 0.5$:
$$\|\mathbf{z}_{\text{lerp}}(0.5)\| = \frac{1}{2} \|\mathbf{z}_A + \mathbf{z}_B\| \approx \frac{\sqrt{2 d}}{2} = \sqrt{\frac{d}{2}} \ll \sqrt{d}$$
The trajectory plunges through the low-probability center of the Gaussian sphere, where the decoder was never trained, producing blurry, distorted images!

#### Spherical Linear Interpolation (SLERP)
To stay strictly within the high-probability manifold shell, we must interpolate along great-circle arcs on the hypersphere:
$$\mathbf{z}_{\text{slerp}}(\alpha) = \frac{\sin((1 - \alpha)\Omega)}{\sin \Omega} \mathbf{z}_A + \frac{\sin(\alpha \Omega)}{\sin \Omega} \mathbf{z}_B$$
where $\Omega = \arccos\left( \frac{\mathbf{z}_A^\top \mathbf{z}_B}{\|\mathbf{z}_A\| \|\mathbf{z}_B\|} \right)$.
SLERP preserves norm $\|\mathbf{z}\| \approx \sqrt{d}$ across the entire morphing path, generating smooth, photo-realistic transitions.

---

## 4. Real-World Analogy

### The Master Perfume Blender & The Recipe Dial
Imagine an artisanal perfume house cataloging 10,000 scents:
- **Standard Autoencoder (The Memorizer):** Labels each perfume bottle with an arbitrary shelf number (Bottle #4, #812, #9914). If you ask for bottle #400 (which was never cataloged), the assistant pulls an empty dust bottle.
- **Variational Autoencoder (The Flavor Chemist):** Deconstructs each perfume into a distribution over 4 continuous universal flavor dials:
  $$\mathbf{z} = [\text{Floral}, \text{Woody}, \text{Citrus}, \text{Musk}]$$
  Rather than giving an exact single number, the chemist says: *"Perfume A has Floral dial at $0.8 \pm 0.1$ and Woody dial at $-0.4 \pm 0.05$."*
  - The **Reconstruction Objective** ensures the recipe actually smells like Perfume A.
  - The **KL Penalty** acts as a gentle spring pulling all dials toward neutral $0.0 \pm 1.0$, preventing any dial from blowing up to $+1000$.
  - Now, if you spin the dials to any random setting or smoothly slide between Floral and Woody, you are 100% guaranteed to concoct a viable, delicious new fragrance!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us compute a complete forward pass, reparameterization, loss computation, and backpropagation gradient of a VAE by hand with concrete toy numbers.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in VAE Architecture |
| :--- | :--- | :--- | :--- |
| $\mathbf{x}$ | Input Vector | $(2,)$ | Ground truth normalized input data $[0.8, 0.2]$ |
| $\boldsymbol{\mu}$ | Latent Mean Vector | $(2,)$ | Encoder output center of posterior distribution |
| $\boldsymbol{\gamma} = \log \boldsymbol{\sigma}^2$ | Latent Log-Variance Vector | $(2,)$ | Encoder output unconstrained log-variance |
| $\boldsymbol{\sigma} = e^{\boldsymbol{\gamma} / 2}$ | Latent Standard Deviation | $(2,)$ | Standard deviation of posterior |
| $\boldsymbol{\epsilon}$ | Auxiliary Noise Vector | $(2,)$ | Standard normal stochastic noise $\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| $\mathbf{z}$ | Reparameterized Latent Code | $(2,)$ | Sampled latent vector $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ |
| $\hat{\mathbf{x}}$ | Reconstructed Vector | $(2,)$ | Decoder output probabilities $\sigma(\mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}})$ |
| $\mathcal{L}_{\text{BCE}}$ | Reconstruction Loss | Scalar | Binary Cross-Entropy loss between $\mathbf{x}$ and $\hat{\mathbf{x}}$ |
| $D_{\text{KL}}$ | Latent Regularization Loss | Scalar | Exact analytical Gaussian KL divergence |
| $\mathcal{L}_{\text{total}}$ | Total VAE Objective | Scalar | $\mathcal{L}_{\text{BCE}} + D_{\text{KL}}$ |

---

### 5.2 Concrete Toy Setup

Let:
- Input vector: $\mathbf{x} = \begin{bmatrix} 0.8 \\ 0.2 \end{bmatrix}$
- Latent dimension $d = 2$.
- Encoder predictions:
  $$\boldsymbol{\mu} = \begin{bmatrix} 0.5000 \\ -0.4000 \end{bmatrix}, \quad \boldsymbol{\gamma} = \log \boldsymbol{\sigma}^2 = \begin{bmatrix} 0.0000 \\ -0.6931 \end{bmatrix}$$
  $$\implies \boldsymbol{\sigma}^2 = \begin{bmatrix} e^0 \\ e^{-0.6931} \end{bmatrix} = \begin{bmatrix} 1.0000 \\ 0.5000 \end{bmatrix} \implies \boldsymbol{\sigma} = \begin{bmatrix} 1.0000 \\ \sqrt{0.5} \end{bmatrix} = \begin{bmatrix} 1.0000 \\ 0.7071 \end{bmatrix}$$
- Fixed auxiliary noise sample:
  $$\boldsymbol{\epsilon} = \begin{bmatrix} 0.2000 \\ -0.5000 \end{bmatrix}$$
- Decoder parameters:
  $$\mathbf{W}_{\text{dec}} = \begin{bmatrix} 1.0 & -0.5 \\ 0.5 & 1.0 \end{bmatrix}, \quad \mathbf{b}_{\text{dec}} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

---

### 5.3 Step 1: Reparameterization Trick Arithmetic

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$
$$z_1 = \mu_1 + \sigma_1 \epsilon_1 = 0.5000 + (1.0000)(0.2000) = \mathbf{0.7000}$$
$$z_2 = \mu_2 + \sigma_2 \epsilon_2 = -0.4000 + (0.7071)(-0.5000) = -0.4000 - 0.3536 = \mathbf{-0.7536}$$

$$\mathbf{z} = \begin{bmatrix} 0.7000 \\ -0.7536 \end{bmatrix}$$

---

### 5.4 Step 2: Decoder Forward Pass Arithmetic

Linear pre-activation:
$$\mathbf{a} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}} = \begin{bmatrix} 1.0 & -0.5 \\ 0.5 & 1.0 \end{bmatrix} \begin{bmatrix} 0.7000 \\ -0.7536 \end{bmatrix} = \begin{bmatrix} 1.0(0.7000) - 0.5(-0.7536) \\ 0.5(0.7000) + 1.0(-0.7536) \end{bmatrix} = \begin{bmatrix} 0.7000 + 0.3768 \\ 0.3500 - 0.7536 \end{bmatrix} = \begin{bmatrix} 1.0768 \\ -0.4036 \end{bmatrix}$$

Sigmoid activation $\hat{x}_i = \sigma(a_i) = \frac{1}{1 + e^{-a_i}}$:
$$\hat{x}_1 = \sigma(1.0768) = \frac{1}{1 + e^{-1.0768}} = \frac{1}{1 + 0.3407} = \mathbf{0.7459}$$
$$\hat{x}_2 = \sigma(-0.4036) = \frac{1}{1 + e^{0.4036}} = \frac{1}{1 + 1.4972} = \mathbf{0.4004}$$

$$\hat{\mathbf{x}} = \begin{bmatrix} 0.7459 \\ 0.4004 \end{bmatrix}$$

---

### 5.5 Step 3: Loss Evaluation Arithmetic

#### A. Reconstruction Loss ($\mathcal{L}_{\text{BCE}}$)
$$\mathcal{L}_{\text{BCE}} = -\sum_{i=1}^2 \left[ x_i \log \hat{x}_i + (1 - x_i) \log (1 - \hat{x}_i) \right]$$
For $i = 1$ ($x_1 = 0.8$, $\hat{x}_1 = 0.7459$):
$$-(0.8 \log(0.7459) + 0.2 \log(0.2541)) = -(0.8(-0.2932) + 0.2(-1.3701)) = -(-0.2346 - 0.2740) = \mathbf{0.5086}$$
For $i = 2$ ($x_2 = 0.2$, $\hat{x}_2 = 0.4004$):
$$-(0.2 \log(0.4004) + 0.8 \log(0.5996)) = -(0.2(-0.9153) + 0.8(-0.5115)) = -(-0.1831 - 0.4092) = \mathbf{0.5923}$$

$$\mathcal{L}_{\text{BCE}} = 0.5086 + 0.5923 = \mathbf{1.1009}$$

#### B. Analytical Gaussian KL Divergence ($D_{\text{KL}}$)
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^2 \left( 1 + \gamma_j - \mu_j^2 - e^{\gamma_j} \right)$$
For $j = 1$ ($\mu_1 = 0.5$, $\gamma_1 = 0.0$):
$$1 + 0.0 - 0.5^2 - e^0 = 1 + 0 - 0.25 - 1.0 = -0.25$$
$$\text{term}_1 = -\frac{1}{2}(-0.25) = \mathbf{0.1250}$$
For $j = 2$ ($\mu_2 = -0.4$, $\gamma_2 = -0.6931$):
$$1 + (-0.6931) - (-0.4)^2 - 0.5 = 1 - 0.6931 - 0.16 - 0.5 = -0.3531$$
$$\text{term}_2 = -\frac{1}{2}(-0.3531) = \mathbf{0.1766}$$

$$D_{\text{KL}} = 0.1250 + 0.1766 = \mathbf{0.3016}$$

#### C. Total VAE Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + D_{\text{KL}} = 1.1009 + 0.3016 = \mathbf{1.4025}$$

---

### 5.6 Step 4: Backpropagation Gradients by Hand

Let us trace gradients back to encoder parameters $\boldsymbol{\mu}$ and $\boldsymbol{\gamma}$.

#### 1. Gradient of BCE with respect to pre-activation $\mathbf{a}$:
For Sigmoid + BCE loss:
$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{a}} = \hat{\mathbf{x}} - \mathbf{x} = \begin{bmatrix} 0.7459 - 0.8000 \\ 0.4004 - 0.2000 \end{bmatrix} = \begin{bmatrix} -0.0541 \\ +0.2004 \end{bmatrix}$$

#### 2. Gradient with respect to latent code $\mathbf{z}$:
$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{z}} = \mathbf{W}_{\text{dec}}^\top \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{a}} = \begin{bmatrix} 1.0 & 0.5 \\ -0.5 & 1.0 \end{bmatrix} \begin{bmatrix} -0.0541 \\ 0.2004 \end{bmatrix} = \begin{bmatrix} -0.0541 + 0.1002 \\ 0.0271 + 0.2004 \end{bmatrix} = \begin{bmatrix} \mathbf{0.0461} \\ \mathbf{0.2275} \end{bmatrix}$$

#### 3. Gradient with respect to $\boldsymbol{\mu}$:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \boldsymbol{\mu}} = \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{z}} \odot \underbrace{\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}}}_{= 1} + \frac{\partial D_{\text{KL}}}{\partial \boldsymbol{\mu}}$$
From $D_{\text{KL}} = -\frac{1}{2} (1 + \gamma - \mu^2 - e^\gamma) \implies \frac{\partial D_{\text{KL}}}{\partial \boldsymbol{\mu}} = \boldsymbol{\mu}$.
Therefore:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mu_1} = 0.0461(1) + 0.5000 = \mathbf{0.5461}$$
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mu_2} = 0.2275(1) + (-0.4000) = \mathbf{-0.1725}$$

#### 4. Gradient with respect to log-variance $\boldsymbol{\gamma}$:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \boldsymbol{\gamma}} = \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{z}} \odot \frac{\partial \mathbf{z}}{\partial \boldsymbol{\gamma}} + \frac{\partial D_{\text{KL}}}{\partial \boldsymbol{\gamma}}$$
Recall $\frac{\partial z_j}{\partial \gamma_j} = \frac{1}{2} \sigma_j \epsilon_j$:
$$\frac{\partial D_{\text{KL}}}{\partial \gamma_j} = -\frac{1}{2}(1 - e^{\gamma_j}) = \frac{1}{2}(e^{\gamma_j} - 1) = \frac{1}{2}(\sigma_j^2 - 1)$$
Therefore:
For $j = 1$ ($\sigma_1 = 1.0, \epsilon_1 = 0.2, \sigma_1^2 = 1.0$):
$$\frac{\partial \mathcal{L}}{\partial \gamma_1} = (0.0461)\left(\frac{1}{2}(1.0)(0.2)\right) + \frac{1}{2}(1.0 - 1.0) = 0.0461(0.1) + 0.0 = \mathbf{0.00461}$$
For $j = 2$ ($\sigma_2 = 0.7071, \epsilon_2 = -0.5, \sigma_2^2 = 0.5$):
$$\frac{\partial \mathcal{L}}{\partial \gamma_2} = (0.2275)\left(\frac{1}{2}(0.7071)(-0.5)\right) + \frac{1}{2}(0.5 - 1.0) = 0.2275(-0.1768) - 0.2500 = -0.0402 - 0.2500 = \mathbf{-0.2902}$$

Every single floating point value above is exact and verified by autograd!

---

## 6. Solved Illustrations

### Illustration 1: Zero KL Divergence Benchmark Case
**Problem:** Prove that when the approximate posterior matches the prior exactly ($\boldsymbol{\mu} = \mathbf{0}, \boldsymbol{\sigma}^2 = \mathbf{I}$), $D_{\text{KL}}$ evaluates to exactly $0.0$.
**Solution:**
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(1) - 0^2 - 1 \right) = -\frac{1}{2} \sum_{j=1}^d (1 + 0 - 0 - 1) = -\frac{1}{2} \sum_{j=1}^d (0) = \mathbf{0.0}$$
The KL loss reaches its global minimum at $\boldsymbol{\mu} = \mathbf{0}$ and $\boldsymbol{\gamma} = \mathbf{0}$.

### Illustration 2: Cyclical KL Annealing Schedule Derivation
**Problem:** In training deep VAEs, posterior collapse occurs if the KL penalty dominates before the autoencoder learns to reconstruct. Formulate a cyclical annealing schedule $\beta(t)$ with cycle length $M$ and ratio $R$.
**Solution:**
Let $t$ be the current training step. Within each cycle $\tau = (t \bmod M) / M \in [0, 1)$:
$$\beta(t) = \begin{cases} \frac{\tau}{R} & \text{if } \tau \le R \\ 1.0 & \text{if } \tau > R \end{cases}$$
If $M = 1000$ steps and $R = 0.5$:
- For $t = 0 \to 500$: $\beta$ ramps linearly from $0.0 \to 1.0$.
- For $t = 500 \to 1000$: $\beta = 1.0$ (standard ELBO).
- At $t = 1000$: $\beta$ resets to $0.0$, reopening the latent bottleneck and preventing inactive latent units from remaining collapsed permanently.

---

### Illustration 3: Complete Step-by-Step Numerical VAE Forward and Backward Pass with Gaussian Decoder (MSE Loss)

**Problem:**
Consider a VAE operating on a 2D continuous space ($D=2$) with a 2D latent space ($d=2$).
Observed data vector:
$$\mathbf{x} = \begin{bmatrix} 3.00 \\ -1.00 \end{bmatrix}$$
The encoder produces mean $\boldsymbol{\mu}$ and log-variance $\boldsymbol{\gamma} = \log(\boldsymbol{\sigma}^2)$:
$$\boldsymbol{\mu} = \begin{bmatrix} 0.80 \\ -0.60 \end{bmatrix}, \qquad \boldsymbol{\gamma} = \begin{bmatrix} 0.00 \\ -0.693147 \end{bmatrix}$$
Standard normal noise draw: $\boldsymbol{\epsilon} = [0.50, -1.00]^\top$.
The decoder is a linear layer with weights $\mathbf{W}_{\text{dec}}$ and bias $\mathbf{b}_{\text{dec}}$ parameterizing a Gaussian likelihood with unit covariance $\sigma_{\text{dec}}^2 = 1.0$:
$$\mathbf{W}_{\text{dec}} = \begin{bmatrix} 1.50 & 0.50 \\ -0.50 & 1.00 \end{bmatrix}, \qquad \mathbf{b}_{\text{dec}} = \begin{bmatrix} 0.20 \\ -0.10 \end{bmatrix}, \qquad \hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$$
1. Compute standard deviation vector $\boldsymbol{\sigma}$ and the sampled latent vector $\mathbf{z}$.
2. Compute reconstructed output $\hat{\mathbf{x}}$ and the Gaussian reconstruction loss $\mathcal{L}_{\text{recon}} = \frac{1}{2}\|\mathbf{x} - \hat{\mathbf{x}}\|^2$.
3. Compute exact analytical KL divergence $D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}))$.
4. Compute total VAE loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + D_{\mathrm{KL}}$.
5. Compute analytical backpropagation gradients: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}}$, $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}}$, $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\gamma}}$, and $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{dec}}}$.

**Step-by-Step Solution:**

**1. Latent Vector Sampling via Reparameterization:**
$$\boldsymbol{\sigma}^2 = \begin{bmatrix} e^{0.00} \\ e^{-0.693147} \end{bmatrix} = \begin{bmatrix} 1.000000 \\ 0.500000 \end{bmatrix} \implies \boldsymbol{\sigma} = \begin{bmatrix} \sqrt{1.000000} \\ \sqrt{0.500000} \end{bmatrix} = \begin{bmatrix} 1.000000 \\ 0.707107 \end{bmatrix}$$
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} = \begin{bmatrix} 0.80 + 1.000000(0.50) \\ -0.60 + 0.707107(-1.00) \end{bmatrix} = \begin{bmatrix} 0.80 + 0.500000 \\ -0.60 - 0.707107 \end{bmatrix} = \begin{bmatrix} \mathbf{1.300000} \\ \mathbf{-1.307107} \end{bmatrix}$$

**2. Reconstruction and Reconstruction Loss:**
$$\hat{\mathbf{x}} = \begin{bmatrix} 1.50 & 0.50 \\ -0.50 & 1.00 \end{bmatrix} \begin{bmatrix} 1.300000 \\ -1.307107 \end{bmatrix} + \begin{bmatrix} 0.20 \\ -0.10 \end{bmatrix}$$
$$\hat{x}_1 = 1.50(1.300000) + 0.50(-1.307107) + 0.20 = 1.950000 - 0.653554 + 0.20 = \mathbf{1.496446}$$
$$\hat{x}_2 = -0.50(1.300000) + 1.00(-1.307107) - 0.10 = -0.650000 - 1.307107 - 0.10 = \mathbf{-2.057107}$$
Residual error $\mathbf{e} = \hat{\mathbf{x}} - \mathbf{x}$:
$$e_1 = 1.496446 - 3.00 = -1.503554$$
$$e_2 = -2.057107 - (-1.00) = -1.057107$$
Reconstruction loss:
$$\mathcal{L}_{\text{recon}} = \frac{1}{2} \left( (-1.503554)^2 + (-1.057107)^2 \right) = \frac{1}{2} (2.260675 + 1.117475) = \frac{1}{2} (3.378150) = \mathbf{1.689075}$$

**3. Analytical Gaussian KL Divergence:**
$$D_{\mathrm{KL}} = -\frac{1}{2} \sum_{j=1}^2 \left( 1 + \gamma_j - \mu_j^2 - e^{\gamma_j} \right)$$
- Dimension 1: $1 + 0.00 - (0.80)^2 - 1.00 = 1 + 0 - 0.64 - 1 = -0.640000$
- Dimension 2: $1 + (-0.693147) - (-0.60)^2 - 0.50 = 1 - 0.693147 - 0.36 - 0.50 = -0.553147$
Sum:
$$D_{\mathrm{KL}} = -\frac{1}{2} (-0.640000 - 0.553147) = -\frac{1}{2}(-1.193147) = \mathbf{0.596574}$$

**4. Total Loss:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + D_{\mathrm{KL}} = 1.689075 + 0.596574 = \mathbf{2.285649}$$

**5. Backpropagation Gradients:**
- **Decoder Output Gradient:**
  $$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \hat{\mathbf{x}}} = \hat{\mathbf{x}} - \mathbf{x} = \begin{bmatrix} -1.503554 \\ -1.057107 \end{bmatrix}$$
- **Latent Vector Gradient:**
  $$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \mathbf{z}} = \mathbf{W}_{\text{dec}}^\top \frac{\partial \mathcal{L}_{\text{recon}}}{\partial \hat{\mathbf{x}}} = \begin{bmatrix} 1.50 & -0.50 \\ 0.50 & 1.00 \end{bmatrix} \begin{bmatrix} -1.503554 \\ -1.057107 \end{bmatrix} = \begin{bmatrix} 1.50(-1.503554) - 0.50(-1.057107) \\ 0.50(-1.503554) + 1.00(-1.057107) \end{bmatrix} = \begin{bmatrix} \mathbf{-1.726777} \\ \mathbf{-1.808884} \end{bmatrix}$$
- **Encoder Mean Gradient ($\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mu_j} = \frac{\partial \mathcal{L}_{\text{recon}}}{\partial z_j} \cdot 1 + \mu_j$):**
  $$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mu_1} = -1.726777 + 0.800000 = \mathbf{-0.926777}$$
  $$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mu_2} = -1.808884 + (-0.600000) = \mathbf{-2.408884}$$
- **Encoder Log-Variance Gradient ($\frac{\partial \mathcal{L}_{\text{total}}}{\partial \gamma_j} = \frac{\partial \mathcal{L}_{\text{recon}}}{\partial z_j} \left(\frac{1}{2}\sigma_j \epsilon_j\right) + \frac{1}{2}(e^{\gamma_j} - 1)$):**
  $$\frac{\partial z_1}{\partial \gamma_1} = \frac{1}{2}(1.000000)(0.50) = 0.250000 \implies \frac{\partial \mathcal{L}}{\partial \gamma_1} = (-1.726777)(0.25) + \frac{1}{2}(1.00 - 1) = \mathbf{-0.431694}$$
  $$\frac{\partial z_2}{\partial \gamma_2} = \frac{1}{2}(0.707107)(-1.00) = -0.353553 \implies \frac{\partial \mathcal{L}}{\partial \gamma_2} = (-1.808884)(-0.353553) + \frac{1}{2}(0.50 - 1) = 0.639536 - 0.250000 = \mathbf{+0.389536}$$
- **Decoder Weights Gradient ($\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{dec}}} = (\hat{\mathbf{x}} - \mathbf{x}) \mathbf{z}^\top$):**
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{dec}}} = \begin{bmatrix} -1.503554 \\ -1.057107 \end{bmatrix} \begin{bmatrix} 1.300000 & -1.307107 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.954620} & \mathbf{+1.965306} \\ \mathbf{-1.374239} & \mathbf{+1.381752} \end{bmatrix}$$

---

### Illustration 4: Numerical Demonstration of the Variance Gap: Reparameterization vs. REINFORCE

**Problem:**
Let latent variable $z \sim \mathcal{N}(\mu, \sigma^2)$ with mean $\mu = 2.00$ and standard deviation $\sigma = 1.50$.
We seek to estimate the parameter gradient $\frac{d}{d\mu} \mathbb{E}_{z \sim q_\mu}[f(z)]$ for the scalar objective function:
$$f(z) = 2 z^2 + 4 z + 10$$
1. Compute the exact theoretical expectation $\mathbb{E}[f(z)]$ and true analytical derivative $\frac{d}{d\mu}\mathbb{E}[f(z)]$.
2. Given 3 standard normal noise samples $\epsilon^{(1)} = -1.20, \epsilon^{(2)} = 0.00, \epsilon^{(3)} = +1.20$, compute the corresponding latent samples $z^{(i)} = \mu + \sigma \epsilon^{(i)}$.
3. Compute the REINFORCE score-function gradient estimate $\hat{g}_{\text{score}}^{(i)} = f(z^{(i)}) \frac{\epsilon^{(i)}}{\sigma}$ for each sample.
4. Compute the pathwise reparameterization gradient estimate $\hat{g}_{\text{rep}}^{(i)} = f'(z^{(i)}) = 4 z^{(i)} + 4$ for each sample.
5. Compute the sample mean and empirical sample variance for both estimators and compare their numerical stability.

**Step-by-Step Solution:**

**1. Theoretical Target Values:**
$$\mathbb{E}[f(z)] = 2(\mu^2 + \sigma^2) + 4\mu + 10 = 2(2.0^2 + 1.5^2) + 4(2.0) + 10 = 2(4.0 + 2.25) + 8 + 10 = 2(6.25) + 18 = \mathbf{30.500}$$
$$\text{True Gradient: } \frac{d}{d\mu} \mathbb{E}[f(z)] = 4\mu + 4 = 4(2.0) + 4 = \mathbf{12.000}$$

**2. Sample Generation ($z^{(i)} = 2.0 + 1.5 \epsilon^{(i)}$):**
- Sample 1 ($\epsilon^{(1)} = -1.20$):
  $$z^{(1)} = 2.0 + 1.5(-1.20) = 2.0 - 1.80 = \mathbf{0.200}$$
  $$f(z^{(1)}) = 2(0.2)^2 + 4(0.2) + 10 = 2(0.04) + 0.8 + 10 = 0.08 + 10.8 = \mathbf{10.880}$$
- Sample 2 ($\epsilon^{(2)} = 0.00$):
  $$z^{(2)} = 2.0 + 1.5(0.0) = \mathbf{2.000}$$
  $$f(z^{(2)}) = 2(2.0)^2 + 4(2.0) + 10 = 2(4.0) + 8.0 + 10 = 8.0 + 18.0 = \mathbf{26.000}$$
- Sample 3 ($\epsilon^{(3)} = +1.20$):
  $$z^{(3)} = 2.0 + 1.5(+1.20) = 2.0 + 1.80 = \mathbf{3.800}$$
  $$f(z^{(3)}) = 2(3.8)^2 + 4(3.8) + 10 = 2(14.44) + 15.2 + 10 = 28.88 + 25.2 = \mathbf{54.080}$$

**3. REINFORCE Score Function Estimator ($\hat{g}_{\text{score}} = f(z) \frac{\epsilon}{1.5}$):**
- Sample 1: $\hat{g}_{\text{score}}^{(1)} = 10.880 \times \frac{-1.20}{1.50} = 10.880 \times (-0.80) = \mathbf{-8.704}$
- Sample 2: $\hat{g}_{\text{score}}^{(2)} = 26.000 \times \frac{0.00}{1.50} = \mathbf{0.000}$
- Sample 3: $\hat{g}_{\text{score}}^{(3)} = 54.080 \times \frac{+1.20}{1.50} = 54.080 \times (+0.80) = \mathbf{+43.264}$

Sample mean:
$$\bar{g}_{\text{score}} = \frac{-8.704 + 0.000 + 43.264}{3} = \frac{34.560}{3} = \mathbf{11.520} \quad (\approx 12.00)$$
Sample variance ($s^2 = \frac{1}{N-1}\sum (\hat{g}_i - \bar{g})^2$):
$$(-8.704 - 11.52)^2 = (-20.224)^2 = 409.010$$
$$(0.000 - 11.52)^2 = (-11.52)^2 = 132.710$$
$$(43.264 - 11.52)^2 = (31.744)^2 = 1007.682$$
$$s_{\text{score}}^2 = \frac{409.010 + 132.710 + 1007.682}{2} = \frac{1549.402}{2} = \mathbf{774.701}$$

**4. Pathwise Reparameterization Estimator ($\hat{g}_{\text{rep}} = 4 z + 4$):**
- Sample 1: $\hat{g}_{\text{rep}}^{(1)} = 4(0.200) + 4 = 0.80 + 4 = \mathbf{4.800}$
- Sample 2: $\hat{g}_{\text{rep}}^{(2)} = 4(2.000) + 4 = 8.00 + 4 = \mathbf{12.000}$
- Sample 3: $\hat{g}_{\text{rep}}^{(3)} = 4(3.800) + 4 = 15.20 + 4 = \mathbf{19.200}$

Sample mean:
$$\bar{g}_{\text{rep}} = \frac{4.800 + 12.000 + 19.200}{3} = \frac{36.000}{3} = \mathbf{12.000} \quad (\text{Exact target!})$$
Sample variance:
$$(4.800 - 12.00)^2 = (-7.2)^2 = 51.84$$
$$(12.000 - 12.00)^2 = 0^2 = 0.00$$
$$(19.200 - 12.00)^2 = (7.2)^2 = 51.84$$
$$s_{\text{rep}}^2 = \frac{51.84 + 0.00 + 51.84}{2} = \frac{103.68}{2} = \mathbf{51.840}$$

**Variance Comparison Summary:**
$$\frac{s_{\text{score}}^2}{s_{\text{rep}}^2} = \frac{774.701}{51.840} \approx \mathbf{14.94\times}$$
The score-function estimator fluctuates violently between $-8.70$ and $+43.26$, having a variance almost **$15\times$ higher** than the reparameterization estimator on just 3 samples. In high-dimensional latent spaces with complex deep decoders, this variance ratio explodes into thousands, rendering REINFORCE training practically impossible without baselines.

---

### Illustration 5: Importance Weighted Autoencoder (IWAE) Multi-Sample Bound Calculation

**Problem:**
Given an observed datum $x = 2.00 \in \mathbb{R}$.
Consider a 1D latent variable $z \in \mathbb{R}$ with:
- Prior: $p(z) = \mathcal{N}(0, 2^2) \implies p(z) = \frac{1}{\sqrt{8\pi}} \exp\left( -\frac{z^2}{8} \right)$
- Likelihood: $p(x \mid z) = \mathcal{N}(z, 1^2) \implies p(x \mid z) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(x - z)^2}{2} \right)$
- Variational Recognition Posterior: $q(z \mid x) = \mathcal{N}(1.5, 1^2) \implies q(z \mid x) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(z - 1.5)^2}{2} \right)$

We evaluate $K = 3$ latent Monte Carlo proposals drawn from $q(z \mid x)$:
$$z_1 = 0.50, \qquad z_2 = 1.50, \qquad z_3 = 2.50$$
1. Compute likelihood $p(x \mid z_k)$, prior $p(z_k)$, and proposal density $q(z_k \mid x)$ for each sample $k \in \{1, 2, 3\}$.
2. Compute unnormalized importance weights $w_k = \frac{p(x, z_k)}{q(z_k \mid x)} = \frac{p(x \mid z_k) p(z_k)}{q(z_k \mid x)}$.
3. Compute normalized importance weights $\tilde{w}_k = \frac{w_k}{\sum_{j=1}^3 w_j}$.
4. Compute the standard single-sample VAE ELBO estimate: $\hat{\mathcal{L}}_1 = \frac{1}{3} \sum_{k=1}^3 \log w_k$.
5. Compute the 3-sample IWAE bound: $\mathcal{L}_3 = \log\left( \frac{1}{3} \sum_{k=1}^3 w_k \right)$.
6. Verify Jensen's inequality numerically: $\mathcal{L}_3 \ge \hat{\mathcal{L}}_1$, and calculate the exact reduction in variational gap.

**Step-by-Step Solution:**

**1. Density Evaluations at $x = 2.00$:**
Constants: $\frac{1}{\sqrt{2\pi}} \approx 0.398942$, $\frac{1}{\sqrt{8\pi}} = \frac{0.398942}{2} \approx 0.199471$.

- **Sample 1 ($z_1 = 0.50$):**
  $$p(x \mid z_1) = 0.398942 \exp\left( -\frac{(2.0 - 0.5)^2}{2} \right) = 0.398942 e^{-1.125} = 0.398942(0.324652) = \mathbf{0.129517}$$
  $$p(z_1) = 0.199471 \exp\left( -\frac{0.5^2}{8} \right) = 0.199471 e^{-0.03125} = 0.199471(0.969233) = \mathbf{0.193334}$$
  $$q(z_1 \mid x) = 0.398942 \exp\left( -\frac{(0.5 - 1.5)^2}{2} \right) = 0.398942 e^{-0.5} = 0.398942(0.606531) = \mathbf{0.241971}$$

- **Sample 2 ($z_2 = 1.50$):**
  $$p(x \mid z_2) = 0.398942 \exp\left( -\frac{(2.0 - 1.5)^2}{2} \right) = 0.398942 e^{-0.125} = 0.398942(0.882497) = \mathbf{0.352065}$$
  $$p(z_2) = 0.199471 \exp\left( -\frac{1.5^2}{8} \right) = 0.199471 e^{-0.28125} = 0.199471(0.754840) = \mathbf{0.150569}$$
  $$q(z_2 \mid x) = 0.398942 \exp\left( -\frac{(1.5 - 1.5)^2}{2} \right) = 0.398942 e^0 = \mathbf{0.398942}$$

- **Sample 3 ($z_3 = 2.50$):**
  $$p(x \mid z_3) = 0.398942 \exp\left( -\frac{(2.0 - 2.5)^2}{2} \right) = 0.398942 e^{-0.125} = 0.398942(0.882497) = \mathbf{0.352065}$$
  $$p(z_3) = 0.199471 \exp\left( -\frac{2.5^2}{8} \right) = 0.199471 e^{-0.78125} = 0.199471(0.457833) = \mathbf{0.199471 \times 0.457833 = 0.091324}$$
  $$q(z_3 \mid x) = 0.398942 \exp\left( -\frac{(2.5 - 1.5)^2}{2} \right) = 0.398942 e^{-0.5} = 0.398942(0.606531) = \mathbf{0.241971}$$

**2. Unnormalized Importance Weights $w_k = \frac{p(x \mid z_k) p(z_k)}{q(z_k \mid x)}$:**
$$w_1 = \frac{0.129517 \times 0.193334}{0.241971} = \frac{0.025040}{0.241971} = \mathbf{0.103483}$$
$$w_2 = \frac{0.352065 \times 0.150569}{0.398942} = \frac{0.053010}{0.398942} = \mathbf{0.132877}$$
$$w_3 = \frac{0.352065 \times 0.091324}{0.241971} = \frac{0.032152}{0.241971} = \mathbf{0.132875}$$
Sum of weights:
$$\sum_{k=1}^3 w_k = 0.103483 + 0.132877 + 0.132875 = \mathbf{0.369235}$$

**3. Normalized Importance Weights:**
$$\tilde{w}_1 = \frac{0.103483}{0.369235} = \mathbf{0.280263}$$
$$\tilde{w}_2 = \frac{0.132877}{0.369235} = \mathbf{0.359871}$$
$$\tilde{w}_3 = \frac{0.132875}{0.369235} = \mathbf{0.359866}$$

**4. Standard VAE ELBO Estimate ($\hat{\mathcal{L}}_1$):**
$$\log w_1 = \log(0.103483) = -2.268383$$
$$\log w_2 = \log(0.132877) = -2.018330$$
$$\log w_3 = \log(0.132875) = -2.018345$$
$$\hat{\mathcal{L}}_1 = \frac{1}{3} (-2.268383 - 2.018330 - 2.018345) = \frac{-6.305058}{3} = \mathbf{-2.101686}$$

**5. 3-Sample IWAE Bound ($\mathcal{L}_3$):**
$$\mathcal{L}_3 = \log \left( \frac{1}{3} \sum_{k=1}^3 w_k \right) = \log\left( \frac{0.369235}{3} \right) = \log(0.123078) = \mathbf{-2.094943}$$

**6. Numerical Proof of Bound Tightening:**
$$\mathcal{L}_3 - \hat{\mathcal{L}}_1 = -2.094943 - (-2.101686) = \mathbf{+0.006743} > 0$$
Because the logarithm is strictly concave, Jensen's inequality guarantees:
$$\log\left( \frac{1}{K}\sum_{k=1}^K w_k \right) \ge \frac{1}{K}\sum_{k=1}^K \log w_k$$
The IWAE bound is strictly closer to the true marginal log-likelihood $\log p(x)$ by $+0.006743$ nats. As $K \to \infty$, Burda et al. (2015) prove that $\mathcal{L}_K \to \log p(x)$ asymptotically!

---

## 7. Deep Learning Connection & Application

### 1. Latent Diffusion Models (Stable Diffusion AutoencoderKL)
In Stable Diffusion (SD 1.5, SDXL, SD 3), diffusion is never executed in raw pixel space ($512 \times 512 \times 3 \approx 786\text{k}$ floats).
Instead:
1. An AutoencoderKL compresses an image $\mathbf{x} \in \mathbb{R}^{3 \times 512 \times 512}$ by factor $f = 8$ down to a compact latent tensor $\mathbf{z} \in \mathbb{R}^{4 \times 64 \times 64}$.
2. The VAE is regularized with a tiny KL weight ($\beta \approx 10^{-6}$), ensuring the latent distribution is Gaussian-like without compromising perceptual sharpness.
3. The diffusion UNet/DiT operates entirely within this $4 \times 64 \times 64$ latent space, accelerating training and inference speed by over **$64\times$**!

### 2. Unsupervised Anomaly Detection
Since a VAE trained on normal data learns the distribution $p(\mathbf{x})$, anomalous inputs (e.g., manufacturing defects, medical lesions) cannot be reconstructed accurately. The pixel-wise reconstruction error $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ acts as an exact heatmap of anomalous regions.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Exact numerical verification of the Part 5 hand arithmetic (reparameterization, BCE, analytical KL, total loss, and autograd gradient equivalence to $< 10^{-6}$).
2. Full PyTorch implementation of a Variational Autoencoder (`VAE`) with MLP encoder and decoder.
3. Analytical Gaussian KL divergence vs. Monte Carlo sampled KL divergence consistency check.
4. Reparameterization gradient flow test verifying non-zero gradients to both mean and log-variance heads.
5. Linear (LERP) vs Spherical (SLERP) latent space interpolation demonstration.

See implementation in:
[`10_generative_models/code/02_variational_autoencoders_and_elbo.py`](./code/02_variational_autoencoders_and_elbo.py)
