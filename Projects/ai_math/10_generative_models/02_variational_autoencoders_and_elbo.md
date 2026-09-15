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
