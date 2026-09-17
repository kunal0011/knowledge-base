# Chapter 24: World Models & Dreamer (Latent Dynamics & Imagination)

---

## 1. Intuition & 101 Motivation

In traditional model-based reinforcement learning (such as Dyna-Q and standard MPC), the transition model operates directly in the raw observation space:
$$\hat{x}_{t+1} \sim p(x_{t+1} \mid x_t, a_t)$$

When observations are high-dimensional pixels (e.g., $64 \times 64 \times 3 = 12,288$ numbers per frame), predicting raw pixels is catastrophic:
1. **Computational Waste:** The neural network squanders 99% of its capacity predicting visual noise—background clouds drifting, flickering stadium lights, or leaves rustling—which has zero relevance to optimal decision-making.
2. **Compounding Blur:** Pixel-level autoregressive models blur rapidly over multi-step rollouts, turning imagined futures into gray soup within 10 steps.

Enter **World Models** (Ha & Schmidhuber, 2018) and the **Dreamer** suite (Hafner et al., DreamerV1 2020, DreamerV2 2021, DreamerV3 2023):
- Instead of modeling pixels, the agent compresses raw frames into a compact, low-dimensional **latent vector** $z_t \in \mathbb{R}^d$ via a variational autoencoder.
- It learns **latent dynamics**: how actions transform latent states $z_{t+1} = f(z_t, a_t)$.
- Crucially, it trains an Actor-Critic policy **entirely inside imagination** (hallucinated latent rollouts) by backpropagating analytic gradients directly through the world model's differentiable transition dynamics!

In 2023, **DreamerV3** achieved a historic milestone: it was the first AI system to collect diamonds from scratch in *Minecraft*—a task requiring thousands of sequential actions and tool-crafting steps—purely through latent imagination without any human demonstrations or domain tuning!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Recurrent State-Space Model (RSSM)

The backbone of Dreamer is the **Recurrent State-Space Model (RSSM)**. It decomposes the latent state at time $t$ into two components:
- **Deterministic Recurrent State $h_t \in \mathbb{R}^{d_h}$:** Maintained by a Recurrent Neural Network (GRU), providing long-term memory across thousands of timesteps.
- **Stochastic Latent State $z_t \in \mathbb{R}^{d_z}$:** Captures unpredictable stochasticity and instantaneous uncertainty.

The combined model state is $s_t \triangleq (h_t, z_t)$.

```
   Observation x_t -------> Encoder ------> q(z_t | h_t, x_t) [Posterior]
                                                |
                                                v (sample z_t)
   Action a_{t-1} ------> Recurrent State h_t ---+---> Decoders (Image x_t, Reward r_t, Discount gamma_t)
                              |
                              v
                        p(z_hat_t | h_t) [Imagination Prior]
```

#### The Six Core Sub-Models of RSSM:
1. **Recurrent Model (Deterministic Dynamics):**
   $$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$$
2. **Representation Model (Posterior):**
   $$q_\theta(z_t \mid h_t, x_t) \quad \text{(Incorporates true real-world observation } x_t \text{)}$$
3. **Transition Predictor (Prior / Imagination Dynamics):**
   $$p_\theta(\hat{z}_t \mid h_t) \quad \text{(Predicts next latent without seeing } x_t \text{)}$$
4. **Observation Decoder (Image Reconstruction):**
   $$p_\theta(x_t \mid h_t, z_t)$$
5. **Reward Predictor:**
   $$p_\theta(r_t \mid h_t, z_t)$$
6. **Continue / Discount Predictor:**
   $$p_\theta(\gamma_t \mid h_t, z_t) \quad \text{where } \gamma_t \in [0, 1] \text{ predicts episode termination } (1 - d_t)$$

---

### 2.2 The Variational ELBO Objective

The world model parameters $\theta$ are trained end-to-end to maximize the Evidence Lower Bound (ELBO) over trajectories $(x_1, a_1, r_1, \dots, x_T)$:

$$\mathcal{L}_{\text{world}}(\theta) = \sum_{t=1}^T \mathbb{E}_{q_\theta(z_t \mid h_t, x_t)} \Big[ \underbrace{\ln p_\theta(x_t \mid h_t, z_t)}_{\text{Image Reconstruction}} + \underbrace{\ln p_\theta(r_t \mid h_t, z_t)}_{\text{Reward Prediction}} + \underbrace{\ln p_\theta(\gamma_t \mid h_t, z_t)}_{\text{Discount Prediction}} - \beta \underbrace{D_{\text{KL}}\left( q_\theta(z_t \mid h_t, x_t) \;\Vert\; p_\theta(\hat{z}_t \mid h_t) \right)}_{\text{KL Information Regularizer}} \Big]$$

#### Analytical Gaussian KL Divergence:
When both prior $p(z) = \mathcal{N}(\mu_p, \sigma_p^2)$ and posterior $q(z) = \mathcal{N}(\mu_q, \sigma_q^2)$ are diagonal Gaussians, the KL divergence is computable in closed form:

$$D_{\text{KL}}(q \;\Vert\; p) = \sum_{j=1}^{d_z} \left[ \ln\frac{\sigma_{p, j}}{\sigma_{q, j}} + \frac{\sigma_{q, j}^2 + (\mu_{q, j} - \mu_{p, j})^2}{2 \sigma_{p, j}^2} - \frac{1}{2} \right]$$

---

### 2.3 KL Balancing (DreamerV2 & DreamerV3)

If optimized naively, the KL term exhibits an asymmetry: the representation model $q$ can collapse toward the prior $p$ before the prior learns meaningful dynamics.
To fix this, Dreamer introduces **KL Balancing** with asymmetric gradient stops:

$$\mathcal{L}_{\text{KL}}(\theta) = \alpha D_{\text{KL}}\left( \operatorname{sg}[q_\theta] \;\Vert\; p_\theta \right) + (1 - \alpha) D_{\text{KL}}\left( q_\theta \;\Vert\; \operatorname{sg}[p_\theta] \right)$$

where $\operatorname{sg}[\cdot]$ is the `stop_gradient` operator, and $\alpha = 0.8$.
This forces the prior $p$ to chase the representation $q$ four times faster than $q$ is penalized for escaping $p$!

---

### 2.4 Learning Behaviors in Imagination

Once the world model is trained, the agent freezes the real world and trains an **Actor $\pi_\phi(a \mid s_t)$** and **Critic $v_\psi(s_t)$** purely within **dream trajectories** of horizon $H = 15$:

1. **Seed Initial State:** Sample $s_\tau = (h_\tau, z_\tau)$ from the replay buffer.
2. **Roll Out in Imagination:** For $t = \tau, \dots, \tau + H - 1$:
   $$a_t \sim \pi_\phi(a_t \mid s_t)$$
   $$h_{t+1} = f_\theta(h_t, z_t, a_t)$$
   $$z_{t+1} \sim p_\theta(z_{t+1} \mid h_{t+1}) \quad \text{(Imagined prior dynamics)}$$
   $$\hat{r}_t = \mathbb{E}[p_\theta(r \mid s_t)], \quad \hat{\gamma}_t = \mathbb{E}[p_\theta(\gamma \mid s_t)]$$
3. **Compute Generalized $\lambda$-Returns:**
   $$V_t^\lambda = \hat{r}_t + \hat{\gamma}_t \left( (1 - \lambda) v_\psi(s_{t+1}) + \lambda V_{t+1}^\lambda \right)$$
   with boundary condition $V_{\tau+H}^\lambda = v_\psi(s_{\tau+H})$.
4. **Update Critic:**
   $$\mathcal{L}(\psi) = \frac{1}{2} \sum_{t=\tau}^{\tau+H-1} \left( v_\psi(s_t) - \operatorname{sg}[V_t^\lambda] \right)^2$$
5. **Update Actor via Analytic Backpropagation:**
   Because the RSSM transition $h_{t+1} = f(h_t, z_t, a_t)$ is fully differentiable, gradients can flow directly from the value function $V_t^\lambda$ back through the world model dynamics into the policy parameters $\phi$:
   $$\max_\phi \sum_{t=\tau}^{\tau+H-1} V_t^\lambda + \eta \mathcal{H}(\pi_\phi(\cdot \mid s_t))$$

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.24.1: Recurrent State Space Model (RSSM) Evidence Lower Bound (ELBO) Derivation

```
====================================================================================================
DERIVATION 11.24.1: RSSM Variational Evidence Lower Bound (ELBO)
====================================================================================================
Problem Statement:
Derive from first principles the Variational Evidence Lower Bound (ELBO) for a sequence of 
observations x_{1:T} and rewards r_{1:T} conditioned on executed actions a_{1:T} under the 
Recurrent State-Space Model (RSSM) with deterministic state h_t and stochastic latent z_t:
  ln p(x_{1:T}, r_{1:T} | a_{1:T}) >= \mathcal{L}_{ELBO}(\theta, \phi)
    = \sum_{t=1}^T \mathbb{E}_{q_\phi(z_{1:t} | x_{1:t}, a_{1:t-1})} [ \ln p_\theta(x_t | h_t, z_t) 
      + \ln p_\theta(r_t | h_t, z_t) - D_{KL}( q_\phi(z_t | h_t, x_t) \parallel p_\theta(z_t | h_t) ) ]
====================================================================================================
```

**1. Problem Statement & Mathematical Goal:**
Let an agent interact with an environment over a trajectory of horizon $T$, generating an action sequence $a_{1:T} = (a_1, \dots, a_T)$, an observation sequence $x_{1:T} = (x_1, \dots, x_T)$, and scalar rewards $r_{1:T} = (r_1, \dots, r_T)$. The agent maintains a recurrent latent state $s_t \triangleq (h_t, z_t)$, where $h_t \in \mathbb{R}^{d_h}$ is a deterministic recurrent state updated via $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$, and $z_t \in \mathbb{R}^{d_z}$ is a stochastic latent variable. The generative model defines:
- The transition prior: $p_\theta(z_t \mid h_t)$
- The observation emission decoder: $p_\theta(x_t \mid h_t, z_t)$
- The reward emission predictor: $p_\theta(r_t \mid h_t, z_t)$

The true marginal log-likelihood $\ln p_\theta(x_{1:T}, r_{1:T} \mid a_{1:T}) = \ln \int p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T}) \, dz_{1:T}$ is computationally intractable due to the high-dimensional nonlinear neural networks parameterizing $f_\theta, p_\theta$. The mathematical goal is to derive the sequence-level Evidence Lower Bound $\mathcal{L}_{\text{ELBO}}(\theta, \phi)$ under a causally structured variational approximate posterior $q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T}) = \prod_{t=1}^T q_\phi(z_t \mid h_t, x_t)$, and prove that the approximation gap is exactly the non-negative Kullback-Leibler divergence between the approximate posterior and the true posterior.

**2. Explicit Assumptions & Regularity Conditions:**
1. **Causal Generative Factorization:** Conditioned on the executed action sequence $a_{1:T}$, the joint generative distribution over latent trajectories $z_{1:T}$ and emissions $(x_{1:T}, r_{1:T})$ decomposes causally as:
   $$p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T}) = \prod_{t=1}^T p_\theta(z_t \mid h_t) \, p_\theta(x_t \mid h_t, z_t) \, p_\theta(r_t \mid h_t, z_t)$$
   where each deterministic state $h_t$ is a deterministic function of $(h_{t-1}, z_{t-1}, a_{t-1})$ with initial condition $h_0, z_0$.
2. **Causal Variational Posterior Factorization:** The approximate posterior $q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})$ factorizes forward in time without future observation leakage:
   $$q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T}) = \prod_{t=1}^T q_\phi(z_t \mid h_t, x_t)$$
   where $h_t$ is computed using the previously sampled latents $z_{<t}$ and actions $a_{<t}$.
3. **Absolute Continuity & Support:** For all $t \in \{1, \dots, T\}$ and for any history $(h_t, x_t)$, the support of the approximate posterior $q_\phi(z_t \mid h_t, x_t)$ is a subset of the support of the prior $p_\theta(z_t \mid h_t)$, ensuring $D_{\text{KL}}(q_\phi \parallel p_\theta) < \infty$.
4. **Conditional Independence of Emissions:** Conditioned on the state tuple $s_t = (h_t, z_t)$, the observation $x_t$ and reward $r_t$ are conditionally independent of all past and future states, actions, and observations.
5. **Integrability:** All log-density functions $\ln p_\theta(x_t \mid s_t)$, $\ln p_\theta(r_t \mid s_t)$, $\ln p_\theta(z_t \mid h_t)$, and $\ln q_\phi(z_t \mid h_t, x_t)$ are Lebesgue-integrable with respect to the measure induced by $q_\phi$.

**3. Underlying Intuition & Geometric / Physical Interpretation:**
In statistical physics and information theory, the marginal log-likelihood represents the negative free energy of the observed sensory stream. Direct evaluation requires summing over all astronomical possible paths of latent thoughts that could have generated the observations. The variational principle introduces a guide distribution $q_\phi$ (the sensory encoder), transforming the intractable integral into an expectation.
The resulting objective balances two fundamental thermodynamic forces:
- **Energy (Likelihood / Accuracy):** $\mathbb{E}_q[\ln p_\theta(x_t \mid s_t) + \ln p_\theta(r_t \mid s_t)]$ penalizes states that fail to reconstruct visual reality or predict physical task rewards.
- **Entropy / Regularization (Occam's Razor):** $-D_{\text{KL}}(q_\phi(z_t \mid h_t, x_t) \parallel p_\theta(z_t \mid h_t))$ measures the informational divergence (or surprise) between the sensory-grounded belief $q_\phi$ and the purely mental imagination prior $p_\theta$. By minimizing this divergence, the world model is trained to anticipate sensory reality before observing it.

Geometrically, in the infinite-dimensional probability manifold of trajectory distributions, the ELBO represents the orthogonal projection of the true posterior $p_\theta(z_{1:T} \mid x_{1:T}, r_{1:T}, a_{1:T})$ onto the tractable subspace of causal autoregressive distributions $\mathcal{Q}$. The variational gap is precisely the informational distance between the truth and the projection.

**4. End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Express the marginal log-evidence as an integral over latent trajectories.*
By the law of total probability, marginalize out the latent variables $z_{1:T} \in \mathbb{R}^{T \times d_z}$:
$$\ln p_\theta(x_{1:T}, r_{1:T} \mid a_{1:T}) = \ln \int p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T}) \, dz_{1:T}$$

*Step 2: Introduce the variational posterior via importance weighting.*
Multiply and divide the integrand by the variational density $q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})$:
$$\ln p_\theta(x_{1:T}, r_{1:T} \mid a_{1:T}) = \ln \int q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T}) \left( \frac{p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} \right) dz_{1:T}$$
Expressing this integral as an expectation with respect to $q_\phi$:
$$\ln p_\theta(x_{1:T}, r_{1:T} \mid a_{1:T}) = \ln \mathbb{E}_{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} \left[ \frac{p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} \right]$$

*Step 3: Apply Jensen's Inequality.*
The natural logarithm function $g(u) = \ln(u)$ is strictly concave on $(0, \infty)$ since $g''(u) = -1/u^2 < 0$.
By Jensen's inequality, for any positive random variable $U$, $\ln \mathbb{E}[U] \ge \mathbb{E}[\ln U]$:
$$\ln \mathbb{E}_{q_\phi} \left[ \frac{p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} \right] \ge \mathbb{E}_{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} \left[ \ln \frac{p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} \right] \triangleq \mathcal{L}_{\text{ELBO}}(\theta, \phi)$$

*Step 4: Expand the numerator and denominator using the causal factorizations.*
Using Assumption 1 for the joint generative model:
$$p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T}) = \prod_{t=1}^T p_\theta(z_t \mid h_t) \, p_\theta(x_t \mid h_t, z_t) \, p_\theta(r_t \mid h_t, z_t)$$
Using Assumption 2 for the approximate posterior:
$$q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T}) = \prod_{t=1}^T q_\phi(z_t \mid h_t, x_t)$$
Substitute both product expansions into the log-ratio inside the expectation:
$$\ln \frac{p_\theta(x_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} = \ln \left( \frac{\prod_{t=1}^T p_\theta(z_t \mid h_t) \, p_\theta(x_t \mid h_t, z_t) \, p_\theta(r_t \mid h_t, z_t)}{\prod_{t=1}^T q_\phi(z_t \mid h_t, x_t)} \right)$$
Using the logarithm identity $\ln \prod_t y_t = \sum_t \ln y_t$:
$$= \sum_{t=1}^T \Big( \ln p_\theta(x_t \mid h_t, z_t) + \ln p_\theta(r_t \mid h_t, z_t) + \ln p_\theta(z_t \mid h_t) - \ln q_\phi(z_t \mid h_t, x_t) \Big)$$
$$= \sum_{t=1}^T \left[ \ln p_\theta(x_t \mid h_t, z_t) + \ln p_\theta(r_t \mid h_t, z_t) + \ln \frac{p_\theta(z_t \mid h_t)}{q_\phi(z_t \mid h_t, x_t)} \right]$$

*Step 5: Apply linearity of expectation and marginalize irrelevant future latents.*
Substitute this sum back into $\mathcal{L}_{\text{ELBO}}$:
$$\mathcal{L}_{\text{ELBO}}(\theta, \phi) = \sum_{t=1}^T \mathbb{E}_{q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T})} \left[ \ln p_\theta(x_t \mid h_t, z_t) + \ln p_\theta(r_t \mid h_t, z_t) + \ln \frac{p_\theta(z_t \mid h_t)}{q_\phi(z_t \mid h_t, x_t)} \right]$$
Examine the terms at time $t$. The deterministic state $h_t$ depends deterministically on $z_{1:t-1}$ and $a_{1:t-1}$. The distributions $p_\theta(\cdot \mid h_t, z_t)$ and $q_\phi(z_t \mid h_t, x_t)$ depend only on $(z_1, \dots, z_t)$ and are completely independent of future latents $z_{t+1:T}$.
Because $q_\phi$ is normalized, the future variables integrate to unity:
$$\int q_\phi(z_{t+1:T} \mid z_{1:t}, x_{1:T}, a_{1:T}) \, dz_{t+1:T} = 1$$
Therefore, by Fubini-Tonelli theorem, the full expectation collapses to:
$$\mathbb{E}_{q_\phi(z_{1:T})} [\cdot] = \mathbb{E}_{q_\phi(z_{1:t} \mid x_{1:t}, a_{1:t-1})} [\cdot]$$

*Step 6: Decompose expectation into past history and current step.*
By the law of total expectation:
$$\mathbb{E}_{q_\phi(z_{1:t})} [\cdot] = \mathbb{E}_{q_\phi(z_{1:t-1})} \left[ \mathbb{E}_{q_\phi(z_t \mid h_t, x_t)} [\cdot] \right]$$
Conditioned on fixed history $z_{1:t-1}$ (which fixes $h_t$), evaluate the expectation of the log-ratio over $z_t \sim q_\phi(z_t \mid h_t, x_t)$:
$$\mathbb{E}_{q_\phi(z_t \mid h_t, x_t)} \left[ \ln \frac{p_\theta(z_t \mid h_t)}{q_\phi(z_t \mid h_t, x_t)} \right] = \int q_\phi(z_t \mid h_t, x_t) \ln \left( \frac{p_\theta(z_t \mid h_t)}{q_\phi(z_t \mid h_t, x_t)} \right) dz_t$$
Using $\ln(a/b) = -\ln(b/a)$:
$$= - \int q_\phi(z_t \mid h_t, x_t) \ln \left( \frac{q_\phi(z_t \mid h_t, x_t)}{p_\theta(z_t \mid h_t)} \right) dz_t$$
By the formal definition of the Kullback-Leibler divergence $D_{\text{KL}}(q \parallel p) = \int q \ln(q/p)$:
$$= - D_{\text{KL}}\left( q_\phi(z_t \mid h_t, x_t) \;\Vert\; p_\theta(z_t \mid h_t) \right)$$

*Step 7: Assemble the sequence-level ELBO objective.*
Recombining the reconstruction expectations and the KL divergence term:
$$\mathcal{L}_{\text{ELBO}}(\theta, \phi) = \sum_{t=1}^T \mathbb{E}_{q_\phi(z_{1:t-1} \mid x_{1:t-1}, a_{1:t-2})} \left[ \mathbb{E}_{q_\phi(z_t \mid h_t, x_t)} \Big[ \ln p_\theta(x_t \mid h_t, z_t) + \ln p_\theta(r_t \mid h_t, z_t) \Big] - D_{\text{KL}}\left( q_\phi(z_t \mid h_t, x_t) \;\Vert\; p_\theta(z_t \mid h_t) \right) \right]$$
Writing $s_t = (h_t, z_t)$ and collecting the expectations:
$$\mathcal{L}_{\text{ELBO}}(\theta, \phi) = \sum_{t=1}^T \mathbb{E}_{q_\phi(z_{1:t} \mid x_{1:t}, a_{1:t-1})} \left[ \ln p_\theta(x_t \mid s_t) + \ln p_\theta(r_t \mid s_t) - D_{\text{KL}}\left( q_\phi(z_t \mid h_t, x_t) \;\Vert\; p_\theta(z_t \mid h_t) \right) \right]$$

*Step 8: Prove that the approximation gap is the exact posterior KL divergence.*
To verify that $\mathcal{L}_{\text{ELBO}}$ is a strict lower bound, evaluate the difference $\Delta \triangleq \ln p_\theta(x_{1:T}, r_{1:T} \mid a_{1:T}) - \mathcal{L}_{\text{ELBO}}(\theta, \phi)$:
$$\Delta = \ln p(X, R \mid A) - \int q(Z \mid X, A) \ln \left( \frac{p(X, R, Z \mid A)}{q(Z \mid X, A)} \right) dZ$$
Since $\int q(Z \mid X, A) dZ = 1$, we can pull $\ln p(X, R \mid A)$ inside the integral:
$$\Delta = \int q(Z \mid X, A) \left[ \ln p(X, R \mid A) - \ln \left( \frac{p(X, R, Z \mid A)}{q(Z \mid X, A)} \right) \right] dZ$$
$$= \int q(Z \mid X, A) \ln \left( \frac{q(Z \mid X, A) \, p(X, R \mid A)}{p(X, R, Z \mid A)} \right) dZ$$
By Bayes' theorem, the true posterior is $p(Z \mid X, R, A) = \frac{p(X, R, Z \mid A)}{p(X, R, \mid A)}$, which implies $\frac{p(X, R \mid A)}{p(X, R, Z \mid A)} = \frac{1}{p(Z \mid X, R, A)}$:
$$\Delta = \int q(Z \mid X, A) \ln \left( \frac{q(Z \mid X, A)}{p(Z \mid X, R, A)} \right) dZ = D_{\text{KL}}\left( q_\phi(z_{1:T} \mid x_{1:T}, a_{1:T}) \;\Vert\; p_\theta(z_{1:T} \mid x_{1:T}, r_{1:T}, a_{1:T}) \right)$$
By Gibbs' inequality, $D_{\text{KL}}(q \parallel p) \ge 0$ for all distributions $q, p$, with equality if and only if $q(Z) = p(Z)$ almost everywhere:
$$\Delta \ge 0 \implies \ln p_\theta(x_{1:T}, r_{1:T} \mid a_{1:T}) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi)$$
This completes the first-principles derivation. $\blacksquare$

---

#### Derivation 11.24.2: Dreamer Latent Value Estimation via $\lambda$-Return in Imagination

```
====================================================================================================
DERIVATION 11.24.2: Recursive Latent λ-Return Dynamic Programming in Imagination
====================================================================================================
Problem Statement:
Prove that the finite-horizon geometric mixture of multi-step bootstrapped returns:
  V_t^λ ≜ (1 - λ) \sum_{n=1}^{H - t - 1} λ^{n-1} R_t^{(n)} + λ^{H - t - 1} R_t^{(H - t)}
where the n-step return is R_t^{(n)} = \sum_{i=0}^{n-1} γ^i r_{t+i} + γ^n v(s_{t+n}),
satisfies the exact backward dynamic programming recursion:
  V_t^λ = r_t + γ [ (1 - λ) v(s_{t+1}) + λ V_{t+1}^λ ]
for all t \in \{\tau, ..., \tau + H - 1\}, with terminal condition V_{\tau+H}^λ = v(s_{\tau+H}).
====================================================================================================
```

**1. Problem Statement & Mathematical Goal:**
In Dreamer's latent imagination phase, an actor-critic policy is optimized inside a neural simulation without interacting with the physical environment. Starting from a latent state $s_\tau = (h_\tau, z_\tau)$ sampled from the replay buffer, the agent executes its policy $\pi(a \mid s)$ through the world model transition dynamics $s_{t+1} \sim p(s_{t+1} \mid s_t, a_t)$ over a finite planning horizon of length $H$, generating an imagined trajectory $(s_\tau, a_\tau, r_\tau, \dots, s_{\tau+H})$.
For any time step $t \in \{\tau, \dots, \tau + H - 1\}$, the $n$-step bootstrapped return over horizon $n \in \{1, \dots, H - t\}$ is defined by:
$$R_t^{(n)} \triangleq \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n v(s_{t+n})$$
where $v(s)$ is the critic value baseline.
The finite-horizon $\lambda$-return $V_t^\lambda$ is defined as the exponentially weighted convex mixture of all $n$-step returns up to the terminal step $H - t$:
$$V_t^\lambda \triangleq (1 - \lambda) \sum_{n=1}^{H - t - 1} \lambda^{n-1} R_t^{(n)} + \lambda^{H - t - 1} R_t^{(H - t)}$$
The mathematical goal is to prove algebraically that $V_t^\lambda$ satisfies the recursive backward dynamic programming equation:
$$V_t^\lambda = r_t + \gamma \left[ (1 - \lambda) v(s_{t+1}) + \lambda V_{t+1}^\lambda \right]$$
with boundary condition $V_{\tau+H}^\lambda = v(s_{\tau+H})$, enabling an $O(H)$ backward sweep instead of an $O(H^2)$ explicit summation.

**2. Explicit Assumptions & Regularity Conditions:**
1. **Finite Planning Horizon:** Trajectories terminate at a finite lookahead horizon $H \in \mathbb{N}_{\ge 1}$, with terminal index $T_{\text{term}} = \tau + H$.
2. **Convex Combination Parameter:** The bootstrapping weight satisfies $\lambda \in [0, 1]$, and the discount factor satisfies $\gamma \in (0, 1]$ (or state-dependent discount $\gamma_t \in [0, 1]$).
3. **Partition of Unity:** For any integer $K \ge 1$, the geometric mixture weights sum to identically $1$:
   $$(1 - \lambda) \sum_{n=1}^{K} \lambda^{n-1} + \lambda^K = (1 - \lambda) \frac{1 - \lambda^K}{1 - \lambda} + \lambda^K = 1 - \lambda^K + \lambda^K \equiv 1$$
4. **Deterministic Evaluation of Trajectory Realization:** The latent states $s_t$, actions $a_t$, and predicted rewards $r_t$ are evaluated deterministically along the imagined path.

**3. Underlying Intuition & Geometric / Physical Interpretation:**
The parameter $\lambda$ provides a continuous spectrum balancing bias and variance in policy evaluation:
- At $\lambda = 0$: $V_t^{\lambda=0} = R_t^{(1)} = r_t + \gamma v(s_{t+1})$ (pure 1-step TD, zero sample variance along the trajectory, but high critic bias).
- At $\lambda = 1$: $V_t^{\lambda=1} = R_t^{(H-t)} = \sum_{i=0}^{H-t-1} \gamma^i r_{t+i} + \gamma^{H-t} v(s_{\tau+H})$ (pure truncated Monte Carlo rollout, zero bootstrap bias before $H$, but accumulating compounding model error).
- For $\lambda \in (0, 1)$: $V_t^\lambda$ averages over all planning horizons.

Direct calculation of all $n$-step returns $R_t^{(n)}$ for every $t \in \{\tau, \dots, \tau + H - 1\}$ requires evaluating $\frac{H(H+1)}{2}$ returns, an $O(H^2)$ operation. The recursive formulation reveals that the future mixture from step $t+1$ onward is mathematically identical to $V_{t+1}^\lambda$, allowing the value targets to be propagated backward in a single linear $O(H)$ pass.

**4. End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Establish the recursive relation between consecutive $n$-step returns.*
Recall the definition of the $n$-step bootstrapped return:
$$R_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n v(s_{t+n})$$
For $n = 1$:
$$R_t^{(1)} = \gamma^0 r_t + \gamma^1 v(s_{t+1}) = r_t + \gamma v(s_{t+1}) \quad \text{(Equation 1)}$$
For any $n \ge 2$:
$$R_t^{(n)} = r_t + \sum_{i=1}^{n-1} \gamma^i r_{t+i} + \gamma^n v(s_{t+n})$$
Perform a change of summation index by setting $j = i - 1$. As $i$ runs from $1$ to $n - 1$, $j$ runs from $0$ to $n - 2$:
$$R_t^{(n)} = r_t + \sum_{j=0}^{n-2} \gamma^{j+1} r_{t+1+j} + \gamma^n v(s_{t+1+(n-1)})$$
Factor out one factor of discount $\gamma$:
$$R_t^{(n)} = r_t + \gamma \left( \sum_{j=0}^{n-2} \gamma^j r_{(t+1)+j} + \gamma^{n-1} v(s_{(t+1)+(n-1)}) \right)$$
Notice that the term in parentheses is precisely the definition of the $(n-1)$-step return initiated from step $t+1$:
$$R_t^{(n)} = r_t + \gamma R_{t+1}^{(n-1)} \quad \text{for all } n \ge 2 \quad \text{(Equation 2)}$$

*Step 2: Partition the definition of the $\lambda$-return.*
Recall the definition of $V_t^\lambda$:
$$V_t^\lambda = (1 - \lambda) \sum_{n=1}^{H - t - 1} \lambda^{n-1} R_t^{(n)} + \lambda^{H - t - 1} R_t^{(H - t)}$$
Isolate the first term ($n = 1$) from the summation:
$$V_t^\lambda = (1 - \lambda) \lambda^0 R_t^{(1)} + (1 - \lambda) \sum_{n=2}^{H - t - 1} \lambda^{n-1} R_t^{(n)} + \lambda^{H - t - 1} R_t^{(H - t)}$$
Since $\lambda^0 = 1$:
$$V_t^\lambda = (1 - \lambda) R_t^{(1)} + (1 - \lambda) \sum_{n=2}^{H - t - 1} \lambda^{n-1} R_t^{(n)} + \lambda^{H - t - 1} R_t^{(H - t)}$$

*Step 3: Substitute the recursive return identities into $V_t^\lambda$.*
Substitute Equation 1 for $R_t^{(1)}$ and Equation 2 for each $R_t^{(n)}$ ($n \ge 2$):
$$V_t^\lambda = (1 - \lambda) \big[ r_t + \gamma v(s_{t+1}) \big] + (1 - \lambda) \sum_{n=2}^{H - t - 1} \lambda^{n-1} \big[ r_t + \gamma R_{t+1}^{(n-1)} \big] + \lambda^{H - t - 1} \big[ r_t + \gamma R_{t+1}^{(H - t - 1)} \big]$$

*Step 4: Collect all terms involving the immediate stage reward $r_t$.*
Group the coefficients multiplying $r_t$:
$$\text{Coeff}(r_t) = (1 - \lambda) + (1 - \lambda) \sum_{n=2}^{H - t - 1} \lambda^{n-1} + \lambda^{H - t - 1}$$
Recombine the first scalar with the summation:
$$\text{Coeff}(r_t) = (1 - \lambda) \left( 1 + \sum_{n=2}^{H - t - 1} \lambda^{n-1} \right) + \lambda^{H - t - 1} = (1 - \lambda) \sum_{n=1}^{H - t - 1} \lambda^{n-1} + \lambda^{H - t - 1}$$
Evaluate the finite geometric series for $K = H - t - 1$:
$$\sum_{n=1}^K \lambda^{n-1} = \frac{1 - \lambda^K}{1 - \lambda}$$
Multiplying by $(1 - \lambda)$:
$$(1 - \lambda) \left( \frac{1 - \lambda^K}{1 - \lambda} \right) = 1 - \lambda^K$$
Substituting $K = H - t - 1$:
$$\text{Coeff}(r_t) = (1 - \lambda^{H - t - 1}) + \lambda^{H - t - 1} = 1$$
Thus, the immediate reward $r_t$ appears with an exact coefficient of $1$:
$$V_t^\lambda = r_t + \gamma \cdot \Omega_{t+1}$$
where $\Omega_{t+1}$ collects all terms scaled by $\gamma$.

*Step 5: Factor out discount $\gamma$ and isolate $(1 - \lambda) v(s_{t+1})$.*
Writing out the bracketed terms scaled by $\gamma$:
$$\Omega_{t+1} = (1 - \lambda) v(s_{t+1}) + (1 - \lambda) \sum_{n=2}^{H - t - 1} \lambda^{n-1} R_{t+1}^{(n-1)} + \lambda^{H - t - 1} R_{t+1}^{(H - t - 1)}$$

*Step 6: Shift the summation index to align with timestep $t+1$.*
In the summation term, define index $m = n - 1$.
- When $n = 2$, $m = 2 - 1 = 1$.
- When $n = H - t - 1$, $m = (H - t - 1) - 1 = H - (t + 1) - 1$.
- The power $\lambda^{n-1} = \lambda^{(m+1)-1} = \lambda^m = \lambda \cdot \lambda^{m-1}$.
Substituting $m$ into the summation:
$$(1 - \lambda) \sum_{n=2}^{H - t - 1} \lambda^{n-1} R_{t+1}^{(n-1)} = (1 - \lambda) \sum_{m=1}^{H - (t+1) - 1} \lambda \cdot \lambda^{m-1} R_{t+1}^{(m)} = \lambda (1 - \lambda) \sum_{m=1}^{H - (t+1) - 1} \lambda^{m-1} R_{t+1}^{(m)}$$
Now examine the terminal boundary return:
$$\lambda^{H - t - 1} R_{t+1}^{(H - t - 1)} = \lambda \cdot \lambda^{(H - t - 2)} R_{t+1}^{(H - (t+1))} = \lambda \cdot \lambda^{H - (t+1) - 1} R_{t+1}^{(H - (t+1))}$$

*Step 7: Factor $\lambda$ to reveal the $(t+1)$-step $\lambda$-return $V_{t+1}^\lambda$.*
Summing the shifted series and the terminal boundary return:
$$\lambda \left[ (1 - \lambda) \sum_{m=1}^{H - (t+1) - 1} \lambda^{m-1} R_{t+1}^{(m)} + \lambda^{H - (t+1) - 1} R_{t+1}^{(H - (t+1))} \right]$$
Inspect the expression inside the large brackets. By definition, the $\lambda$-return at timestep $t+1$ over the remaining horizon $H' = H - 1$ (where remaining steps are $H - (t+1)$) is:
$$V_{t+1}^\lambda = (1 - \lambda) \sum_{m=1}^{H - (t+1) - 1} \lambda^{m-1} R_{t+1}^{(m)} + \lambda^{H - (t+1) - 1} R_{t+1}^{(H - (t+1))}$$
Therefore:
$$\Omega_{t+1} = (1 - \lambda) v(s_{t+1}) + \lambda V_{t+1}^\lambda$$
Substituting $\Omega_{t+1}$ back into the expression for $V_t^\lambda$:
$$V_t^\lambda = r_t + \gamma \left[ (1 - \lambda) v(s_{t+1}) + \lambda V_{t+1}^\lambda \right]$$

*Step 8: Verify the boundary condition at the horizon limit $t = \tau + H - 1$.*
At the penultimate step $t = \tau + H - 1$, the remaining horizon is $H - t = 1$.
The summation in the definition of $V_t^\lambda$ is empty (runs from $n=1$ to $0$), leaving only the terminal term:
$$V_{\tau+H-1}^\lambda = \lambda^{1 - 1} R_{\tau+H-1}^{(1)} = 1 \cdot \left[ r_{\tau+H-1} + \gamma v(s_{\tau+H}) \right]$$
Now test the recursive formula with boundary condition $V_{\tau+H}^\lambda \triangleq v(s_{\tau+H})$:
$$V_{\tau+H-1}^\lambda = r_{\tau+H-1} + \gamma \left[ (1 - \lambda) v(s_{\tau+H}) + \lambda V_{\tau+H}^\lambda \right]$$
$$= r_{\tau+H-1} + \gamma \left[ (1 - \lambda) v(s_{\tau+H}) + \lambda v(s_{\tau+H}) \right]$$
$$= r_{\tau+H-1} + \gamma \Big[ (1 - \lambda + \lambda) v(s_{\tau+H}) \Big] = r_{\tau+H-1} + \gamma v(s_{\tau+H})$$
The recursive formula matches the exact definition at the horizon boundary.
For generalized state-dependent discount factors $\gamma_t$, replacing $\gamma$ with $\gamma_t$ identically yields:
$$V_t^\lambda = r_t + \gamma_t \left[ (1 - \lambda) v(s_{t+1}) + \lambda V_{t+1}^\lambda \right]$$
This completes the proof. $\blacksquare$

---

#### Derivation 11.24.3: DreamerV3 Symlog Transformation and Two-Hot Categorical Cross-Entropy Loss

```
====================================================================================================
DERIVATION 11.24.3: DreamerV3 Symlog Transformation & Two-Hot Categorical Cross-Entropy Loss
====================================================================================================
Problem Statement:
Prove the algebraic and analytic properties of the symmetrical logarithmic mapping:
  symlog(x) ≜ sign(x) ln(|x| + 1),  x ∈ ℝ
its analytical inverse symexp(y) ≜ sign(y) (exp(|y|) - 1),
and prove that the two-hot categorical probability distribution target p* over discrete bins {b_k}
has expectation identically equal to symlog(x) (E_p*[b] = symlog(x)), with bounded loss gradients
preventing gradient explosion across multi-magnitude reinforcement learning reward scales.
====================================================================================================
```

**1. Problem Statement & Mathematical Goal:**
Reinforcement learning algorithms routinely fail when deployed across domains with drastically differing reward scales:
- In continuous control (DeepMind Control Suite), rewards are dense and normalized: $r_t \in [0, 1]$, with episode returns $\sim 10^2$.
- In classic arcade games (Atari), scores vary wildly: Pong yields $\pm 1$, while Ms. Pacman or Bowling yields $10^4$ to $10^5$.
- In open-ended worlds (Minecraft), finding diamonds yields sparse, massive reward spikes.

Standard mean-squared error (MSE) regression losses $\mathcal{L}_{\text{MSE}} = \frac{1}{2}(v_\theta(s) - y)^2$ generate loss gradients $\nabla_\theta \mathcal{L} = (v_\theta(s) - y) \nabla_\theta v_\theta(s)$. When returns are $10^4$, gradient magnitudes explode by four orders of magnitude, causing catastrophic numerical overflow, dead ReLUs, or network divergence.
Hafner et al. (DreamerV3, Nature 2023) resolved this foundational instability through:
1. The **symlog transformation** $\operatorname{symlog}(x) \triangleq \operatorname{sign}(x) \ln(|x| + 1)$ and its inverse $\operatorname{symexp}(y) \triangleq \operatorname{sign}(y) (\exp(|y|) - 1)$.
2. The **two-hot categorical cross-entropy loss** over a fixed discrete 1D support grid $\mathcal{B} = \{b_1, \dots, b_K\}$.

The mathematical goal is to:
1. Prove that $\operatorname{symlog}$ is strictly monotonic, odd, continuously differentiable everywhere on $\mathbb{R}$, and has a derivative universally bounded in $(0, 1]$.
2. Prove that $\operatorname{symexp}$ is the exact global inverse of $\operatorname{symlog}$.
3. Prove that the two-hot categorical target distribution $\mathbf{p}^*$ over adjacent bins preserves the exact continuous target: $\mathbb{E}_{\mathbf{p}^*}[b] = y$.
4. Derive the analytical cross-entropy loss gradient $\nabla_{\mathbf{l}} \mathcal{L}_{\text{CE}} = \mathbf{p} - \mathbf{p}^*$ and prove that all gradient components are bounded within $[-1, 1]$, establishing universal scale-invariance.

**2. Explicit Assumptions & Regularity Conditions:**
1. **Unrestricted Real Domain:** The continuous input target $x \in \mathbb{R}$ is an arbitrary real scalar (reward $r$, return $R$, or value $V$).
2. **Support Grid Span:** A pre-specified sorted array of $K \ge 2$ scalar bin centers $\mathcal{B} = \{b_1, b_2, \dots, b_K\}$ satisfies $b_1 < b_2 < \dots < b_K$, with spacing $\Delta b_k = b_{k+1} - b_k > 0$. The grid covers the transformed target: $y = \operatorname{symlog}(x) \in [b_1, b_K]$.
3. **Softmax Output Parameterization:** The value network outputs unbounded logits $\mathbf{l} \in \mathbb{R}^K$. The predicted probability distribution $\mathbf{p} \in \Delta^{K-1}$ is computed via the standard softmax function:
   $$p_i = \frac{\exp(l_i)}{\sum_{j=1}^K \exp(l_j)}, \quad i \in \{1, \dots, K\}$$

**3. Underlying Intuition & Geometric / Physical Interpretation:**
Standard logarithm $\ln(x)$ is asymmetric, undefined for non-positive numbers ($x \le 0$), and has infinite derivative as $x \to 0^+$.
The symlog function resolves this by shifting the argument by $+1$ before applying the natural logarithm, and applying the sign of $x$:
- **Infinitesimal Regime ($|x| \ll 1$):** By Taylor expansion, $\ln(1 + |x|) = |x| - \frac{|x|^2}{2} + \mathcal{O}(|x|^3)$. Thus $\operatorname{symlog}(x) \approx x$. Small signals pass through linearly without loss of resolution.
- **Macroscopic Regime ($|x| \gg 1$):** $\operatorname{symlog}(x) \approx \operatorname{sign}(x) \ln|x|$. Astronomical returns are compressed exponentially: a reward of $10,000$ becomes $9.21$.

Crucially, in the two-hot categorical formulation, target values are not predicted via unbounded scalar regression. Instead, the real target $y$ is projected onto a 1D probability simplex $\Delta^{K-1}$ via linear interpolation between the two nearest bin centers. The neural network is trained using categorical cross-entropy. Because cross-entropy gradients with respect to logits are simply $\mathbf{p} - \mathbf{p}^* \in [-1, 1]$, the gradient signal driving the neural network is completely decoupled from the physical reward unit!

**4. End-to-End Step-by-Step Algebraic Proof:**

*Step 1: Proof of symmetry and global invertibility.*
Define the symmetrical logarithm for all $x \in \mathbb{R}$:
$$\operatorname{symlog}(x) \triangleq \operatorname{sign}(x) \ln(|x| + 1)$$
where the sign function is defined as $\operatorname{sign}(x) = +1$ if $x > 0$, $-1$ if $x < 0$, and $0$ if $x = 0$.
1. **Odd Symmetry:**
   $$\operatorname{symlog}(-x) = \operatorname{sign}(-x) \ln(|-x| + 1) = -\operatorname{sign}(x) \ln(|x| + 1) = -\operatorname{symlog}(x)$$
   Thus $\operatorname{symlog}(x)$ is an odd function with $\operatorname{symlog}(0) = 0$.
2. **Sign Preservation:**
   Since $|x| + 1 > 1$ for all $x \neq 0$, $\ln(|x| + 1) > 0$.
   Therefore, $\operatorname{sign}(\operatorname{symlog}(x)) = \operatorname{sign}(x)$.
   Also, $|\operatorname{symlog}(x)| = \ln(|x| + 1)$.
3. **Analytical Inverse (symexp):**
   Let $y = \operatorname{symlog}(x)$. Then $|y| = \ln(|x| + 1)$.
   Exponentiating both sides:
   $$\exp(|y|) = |x| + 1 \implies |x| = \exp(|y|) - 1$$
   Since $\operatorname{sign}(x) = \operatorname{sign}(y)$:
   $$x = \operatorname{sign}(x) |x| = \operatorname{sign}(y) \big( \exp(|y|) - 1 \big) \triangleq \operatorname{symexp}(y)$$
   Verify identity by composition:
   $$\operatorname{symexp}(\operatorname{symlog}(x)) = \operatorname{sign}(\operatorname{symlog}(x)) \Big( \exp\big(|\operatorname{symlog}(x)|\big) - 1 \Big)$$
   $$= \operatorname{sign}(x) \Big( \exp\big(\ln(|x| + 1)\big) - 1 \Big) = \operatorname{sign}(x) \big( (|x| + 1) - 1 \big) = \operatorname{sign}(x) |x| = x$$
   $$\operatorname{symlog}(\operatorname{symexp}(y)) = \operatorname{sign}(\operatorname{symexp}(y)) \ln\Big( |\operatorname{symexp}(y)| + 1 \Big)$$
   $$= \operatorname{sign}(y) \ln\Big( (\exp(|y|) - 1) + 1 \Big) = \operatorname{sign}(y) \ln\big(\exp(|y|)\big) = \operatorname{sign}(y) |y| = y$$
   Thus $\operatorname{symexp}$ is the exact global inverse of $\operatorname{symlog}$ on all of $\mathbb{R}$.

*Step 2: Smoothness and derivative bounds.*
1. For $x > 0$: $|x| = x$, so $\operatorname{symlog}(x) = \ln(x + 1)$.
   $$\frac{d}{dx} \operatorname{symlog}(x) = \frac{1}{x + 1} = \frac{1}{|x| + 1}$$
2. For $x < 0$: $|x| = -x$, so $\operatorname{symlog}(x) = -\ln(-x + 1)$.
   By the chain rule:
   $$\frac{d}{dx} \operatorname{symlog}(x) = - \frac{1}{-x + 1} \cdot (-1) = \frac{1}{1 - x} = \frac{1}{|x| + 1}$$
3. At $x = 0$: evaluate the left and right limits of the difference quotient:
   $$\lim_{h \to 0^+} \frac{\operatorname{symlog}(h) - \operatorname{symlog}(0)}{h} = \lim_{h \to 0^+} \frac{\ln(1 + h)}{h} = 1$$
   $$\lim_{h \to 0^-} \frac{\operatorname{symlog}(h) - \operatorname{symlog}(0)}{h} = \lim_{h \to 0^-} \frac{-\ln(1 - h)}{h}$$
   Letting $u = -h \to 0^+$:
   $$= \lim_{u \to 0^+} \frac{-\ln(1 + u)}{-u} = \lim_{u \to 0^+} \frac{\ln(1 + u)}{u} = 1$$
   Since both one-sided limits exist and equal $1$:
   $$\left. \frac{d}{dx} \operatorname{symlog}(x) \right|_{x=0} = 1 = \frac{1}{|0| + 1}$$
   Therefore, $\operatorname{symlog}$ is continuously differentiable ($C^1$) across all of $\mathbb{R}$ with:
   $$\frac{d}{dx} \operatorname{symlog}(x) = \frac{1}{|x| + 1}$$
   Notice that for all $x \in \mathbb{R}$:
   $$0 < \frac{d}{dx} \operatorname{symlog}(x) \le 1$$
   Strict positivity proves that $\operatorname{symlog}(x)$ is strictly monotonically increasing.
   The upper bound of $1$ proves that the mapping is non-expansive ($1$-Lipschitz continuous):
   $$|\operatorname{symlog}(x_1) - \operatorname{symlog}(x_2)| \le |x_1 - x_2| \quad \forall x_1, x_2 \in \mathbb{R}$$

*Step 3: Two-hot categorical target distribution and expectation preservation.*
Let $y = \operatorname{symlog}(x) \in [b_1, b_K]$.
Since the grid $\mathcal{B} = \{b_1, \dots, b_K\}$ is sorted, there exists an index $k \in \{1, \dots, K-1\}$ such that $b_k \le y \le b_{k+1}$.
Define the two-hot target distribution vector $\mathbf{p}^* = [p_1^*, \dots, p_K^*]^T \in \mathbb{R}^K$ by:
$$p_k^* \triangleq \frac{b_{k+1} - y}{b_{k+1} - b_k}, \quad p_{k+1}^* \triangleq \frac{y - b_k}{b_{k+1} - b_k}, \quad p_j^* \triangleq 0 \quad (\forall j \notin \{k, k+1\})$$
1. **Probability Axiom Verification:**
   - Non-negativity: Since $b_k \le y \le b_{k+1}$, $b_{k+1} - y \ge 0$ and $y - b_k \ge 0$. Because $b_{k+1} - b_k > 0$, $p_k^* \ge 0$ and $p_{k+1}^* \ge 0$.
   - Normalization:
     $$\sum_{j=1}^K p_j^* = p_k^* + p_{k+1}^* = \frac{b_{k+1} - y + y - b_k}{b_{k+1} - b_k} = \frac{b_{k+1} - b_k}{b_{k+1} - b_k} = 1$$
   Thus $\mathbf{p}^* \in \Delta^{K-1}$ is a valid probability distribution.
2. **Expectation Preservation:**
   Compute the expected bin center under $\mathbf{p}^*$:
   $$\mathbb{E}_{\mathbf{p}^*}[b] \triangleq \sum_{j=1}^K p_j^* b_j = p_k^* b_k + p_{k+1}^* b_{k+1}$$
   Substitute the definitions of $p_k^*$ and $p_{k+1}^*$:
   $$\mathbb{E}_{\mathbf{p}^*}[b] = \left( \frac{b_{k+1} - y}{b_{k+1} - b_k} \right) b_k + \left( \frac{y - b_k}{b_{k+1} - b_k} \right) b_{k+1}$$
   $$= \frac{b_{k+1} b_k - y b_k + y b_{k+1} - b_k b_{k+1}}{b_{k+1} - b_k} = \frac{y (b_{k+1} - b_k)}{b_{k+1} - b_k} = y \equiv \operatorname{symlog}(x)$$
   The expected value under the two-hot target distribution recovers the continuous target $y$ with zero bias!

*Step 4: Cross-entropy loss gradient and universal gradient boundedness.*
Let the value network output logits $\mathbf{l} \in \mathbb{R}^K$. The predicted probability vector $\mathbf{p} = [p_1, \dots, p_K]^T$ has elements:
$$p_i = \frac{\exp(l_i)}{\sum_{m=1}^K \exp(l_m)}$$
The cross-entropy loss between the two-hot target $\mathbf{p}^*$ and predicted distribution $\mathbf{p}$ is:
$$\mathcal{L}_{\text{CE}}(\mathbf{l}, \mathbf{p}^*) = - \sum_{i=1}^K p_i^* \ln p_i$$
To compute the gradient with respect to logit $l_j$, apply the multivariate chain rule:
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial l_j} = - \sum_{i=1}^K p_i^* \frac{\partial \ln p_i}{\partial l_j} = - \sum_{i=1}^K \frac{p_i^*}{p_i} \frac{\partial p_i}{\partial l_j}$$
Recall the Jacobian of the softmax function:
$$\frac{\partial p_i}{\partial l_j} = p_i (\delta_{ij} - p_j)$$
where $\delta_{ij}$ is the Kronecker delta ($\delta_{ij} = 1$ if $i = j$, $0$ otherwise).
Substitute this into the derivative:
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial l_j} = - \sum_{i=1}^K \frac{p_i^*}{p_i} \Big[ p_i (\delta_{ij} - p_j) \Big] = - \sum_{i=1}^K p_i^* (\delta_{ij} - p_j)$$
Distribute the sum:
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial l_j} = - \left( \sum_{i=1}^K p_i^* \delta_{ij} - p_j \sum_{i=1}^K p_i^* \right)$$
Using the sifting property $\sum_i p_i^* \delta_{ij} = p_j^*$ and the normalization property $\sum_i p_i^* = 1$:
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial l_j} = - (p_j^* - p_j \cdot 1) = p_j - p_j^*$$
In vector notation:
$$\nabla_{\mathbf{l}} \mathcal{L}_{\text{CE}} = \mathbf{p} - \mathbf{p}^*$$

*Step 5: Analysis of scale-invariance.*
Because both $\mathbf{p}$ and $\mathbf{p}^*$ are valid probability vectors living in the simplex $\Delta^{K-1}$:
$$p_j \in [0, 1] \quad \text{and} \quad p_j^* \in [0, 1] \implies -1 \le p_j - p_j^* \le 1$$
Therefore, for every single coordinate $j \in \{1, \dots, K\}$:
$$\left| \frac{\partial \mathcal{L}_{\text{CE}}}{\partial l_j} \right| \le 1$$
Under standard regression MSE, the gradient with respect to predicted value $\hat{y}$ is:
$$\left| \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}} \right| = |\hat{y} - y|$$
which scales linearly with $|y|$ and diverges to $\infty$ on extreme targets.
In contrast, the two-hot cross-entropy gradient $\mathbf{p} - \mathbf{p}^*$ is strictly bounded in $[-1, 1]$ independently of whether the unnormalized physical return is $0.001$ or $100,000$.
Furthermore, when inverted back to physical space, the prediction is:
$$\hat{x} = \operatorname{symexp}\left( \sum_{j=1}^K p_j b_j \right)$$
which is continuous, monotonically increasing in expected logits, and robust to multi-magnitude reward scales.
This completes the first-principles derivation. $\blacksquare$

---


## 3. Geometric & Physical Interpretation

### 3.1 Latent Space Manifold Projection
A $64 \times 64 \times 3$ pixel frame lives in $\mathbb{R}^{12,288}$. However, the true physical degrees of freedom (e.g., robot joint angles, cart position) occupy a smooth, compact submanifold of dimension $d \le 30$.

```
High-Dimensional Observation Space (R^12288)
         \      /
          \    /  <-- Variational Encoder (Lossy Compression)
           \  /
            \/
Low-Dimensional Latent Manifold (R^32)
   *-------*-------*-------*   <-- Locally smooth, linearizable physics!
  z_0     z_1     z_2     z_3
```
In latent space, physics is smooth, predictable, and differentiable, enabling gradient-based trajectory optimization that would be impossible over raw pixels.

---

## 4. Real-World Analogy: The Flight Simulator in Your Mind

Imagine an elite stunt pilot preparing for a dangerous aerial maneuver:
- They sit in a chair with their eyes closed.
- In their mind, they execute the sequence: *"Pull stick back 3 inches, kick right rudder, tap throttle."*
- Their mental world model predicts the resulting sensation: *"G-force surges to 4G, horizon flips upside down, stall alarm sounds."*
- If the mental simulation ends in a crash, the pilot adjusts their imagined stick pressure and tries again—**all without burning a single drop of jet fuel or risking their life**.
- That is Dreamer: millions of simulated flights inside a neural world model!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace the exact numerical calculations of:
1. **The Variational Gaussian KL Divergence** between Posterior and Prior.
2. **The 2-Step Imagined $\lambda$-Return Calculation**.

---

### 5.1 System & Model Parameters

#### Component 1: Gaussian KL Calculation (1D Scalar Latent)
- **Prior Distribution:** $p(z) = \mathcal{N}(\mu_p, \sigma_p^2)$
  - $\mu_p = 0.0000$
  - $\sigma_p = 1.0000 \implies \sigma_p^2 = 1.0000$
- **Posterior Distribution:** $q(z) = \mathcal{N}(\mu_q, \sigma_q^2)$
  - $\mu_q = 0.8000$
  - $\sigma_q = 0.5000 \implies \sigma_q^2 = 0.2500$

#### Component 2: Imagined 2-Step $\lambda$-Return Rollout
- **Planning Horizon:** $H = 2$ steps ($t = 1, 2$)
- **Discount Factor:** $\gamma = 0.9000$
- **Bootstrapping Factor:** $\lambda = 0.8000$
- **Imagined Rewards:**
  - $\hat{r}_1 = 2.0000$
  - $\hat{r}_2 = 5.0000$
- **Critic Value Predictions:**
  - $v(s_1) = 4.0000$
  - $v(s_2) = 8.0000$
  - $v(s_3) = 10.0000$ (Terminal bootstrap)

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\mu_p, \sigma_p$ | `prior_mu, prior_std` | Mean and standard deviation of imagination prior $p(z \mid h)$ |
| $\mu_q, \sigma_q$ | `post_mu, post_std` | Mean and standard deviation of sensory posterior $q(z \mid h, x)$ |
| $D_{\text{KL}}(q \parallel p)$ | `kl_div` | Information divergence penalizing model hallucination |
| $\hat{r}_t$ | `reward_hat[t]` | Predicted stage reward emitted by latent state $s_t$ |
| $v(s_t)$ | `critic_val[t]` | Critic network baseline value estimate at state $s_t$ |
| $V_t^\lambda$ | `lambda_return[t]` | Recursive multi-step target return in imagination |

---

### 5.3 Step-by-Step Hand Calculations: Gaussian KL Divergence

Recall the exact formula:
$$D_{\text{KL}}(q \;\Vert\; p) = \ln\left(\frac{\sigma_p}{\sigma_q}\right) + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2 \sigma_p^2} - \frac{1}{2}$$

1. **Log-Ratio of Standard Deviations:**
   $$\frac{\sigma_p}{\sigma_q} = \frac{1.0000}{0.5000} = 2.0000$$
   $$\ln(2.0000) \approx \mathbf{0.693147}$$

2. **Variance and Mean Difference Term:**
   $$\sigma_q^2 = (0.5000)^2 = 0.2500$$
   $$(\mu_q - \mu_p)^2 = (0.8000 - 0.0000)^2 = 0.6400$$
   $$\sigma_q^2 + (\mu_q - \mu_p)^2 = 0.2500 + 0.6400 = 0.8900$$
   $$\frac{0.8900}{2 \times \sigma_p^2} = \frac{0.8900}{2.0000} = \mathbf{0.445000}$$

3. **Total KL Divergence:**
   $$D_{\text{KL}}(q \;\Vert\; p) = 0.693147 + 0.445000 - 0.500000 = \mathbf{0.638147}$$

---

### 5.4 Step-by-Step Hand Calculations: 2-Step Imagined $\lambda$-Return

The recursive formula is evaluated backward from terminal step $H = 2$ to $t = 1$:

$$V_H^\lambda = v(s_{H+1}) = v(s_3) = \mathbf{10.0000}$$

#### Timestep $t = 2$:
At the penultimate step:
$$V_2^\lambda = \hat{r}_2 + \gamma \cdot v(s_3) = 5.0000 + 0.9000 \times 10.0000 = 5.0000 + 9.0000 = \mathbf{14.0000}$$

#### Timestep $t = 1$:
Now apply the general recursive $\lambda$-combination:
$$V_1^\lambda = \hat{r}_1 + \gamma \left[ (1 - \lambda) v(s_2) + \lambda V_2^\lambda \right]$$
1. **Bracketed Next-State Value Combination:**
   $$(1 - \lambda) v(s_2) = (1.0 - 0.80) \times 8.0000 = 0.20 \times 8.0000 = \mathbf{1.6000}$$
   $$\lambda V_2^\lambda = 0.80 \times 14.0000 = \mathbf{11.2000}$$
   $$\text{Combined Target} = 1.6000 + 11.2000 = \mathbf{12.8000}$$

2. **Discount and Stage Reward Addition:**
   $$V_1^\lambda = \hat{r}_1 + \gamma \times 12.8000 = 2.0000 + 0.9000 \times 12.8000 = 2.0000 + 11.5200 = \mathbf{13.5200}$$

---

### 5.5 Summary Visual Grid: Latent World Model & Imagination Ledger

| Variable / Step | Prior Distribution $p$ | Posterior Distribution $q$ | Component Values | Step Result |
| :--- | :---: | :---: | :---: | :---: |
| **Log Ratio** | $\sigma_p = 1.0$ | $\sigma_q = 0.5$ | $\ln(1.0 / 0.5) = \ln(2)$ | **$+0.6931$** |
| **Quad Penalty** | $\mu_p = 0.0$ | $\mu_q = 0.8$ | $(0.25 + 0.64) / 2$ | **$+0.4450$** |
| **Offset** | — | — | Constant $-0.5$ | **$-0.5000$** |
| **Total KL** | — | — | $0.6931 + 0.4450 - 0.5$ | $\mathbf{D_{\text{KL}} = 0.6381}$ |
| **$V_2^\lambda$ Return** | — | — | $5.0 + 0.9 \times 10.0$ | $\mathbf{V_2^\lambda = 14.0000}$ |
| **$V_1^\lambda$ Return** | — | — | $2.0 + 0.9 \times [0.2(8) + 0.8(14)]$ | $\mathbf{V_1^\lambda = 13.5200}$ |

---

## 6. Solved Illustrations

### Illustration 1: Why DreamerV2 & V3 Use Discrete Categorical Latents
**Problem:**
Why did DreamerV2 and V3 replace continuous Gaussian latents $\mathcal{N}(\mu, \sigma^2)$ with a matrix of categorical variables ($32$ categorical distributions with $32$ classes each)? Provide a concrete numerical demonstration of Gaussian mode-averaging failure vs. categorical mode preservation.

**Solution:**
1. **The Mode-Averaging Dilemma:**
   Consider an agent navigating around an obstacle located at coordinate $x = 0$. The true transition dynamics are bimodal: the agent can either pass to the left ($x = -3.0$) with probability $0.5$, or pass to the right ($x = +3.0$) with probability $0.5$:
   $$P(X = -3.0) = 0.5, \quad P(X = +3.0) = 0.5$$
   The true statistics of this ground-truth distribution are:
   $$\mu_{\text{true}} = 0.5(-3.0) + 0.5(+3.0) = \mathbf{0.000000}$$
   $$\sigma_{\text{true}}^2 = 0.5(-3.0 - 0)^2 + 0.5(+3.0 - 0)^2 = 0.5(9.0) + 0.5(9.0) = \mathbf{9.000000} \implies \sigma_{\text{true}} = \mathbf{3.000000}$$

2. **Gaussian Distribution Failure:**
   If we fit a unimodal Gaussian $\mathcal{N}(\mu, \sigma^2)$ to this data, the maximum likelihood estimate chooses $\mu = 0.0$ and $\sigma = 3.0$.
   Evaluating the Gaussian probability density at the coordinate of the solid obstacle ($x = 0$):
   $$f_{\text{Gaussian}}(0) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(0 - 0)^2}{2\sigma^2} \right) = \frac{1}{3 \times 2.506628} = \frac{1}{7.519885} \approx \mathbf{0.132981}$$
   The peak probability density of the Gaussian sits directly inside the obstacle!
   Now compute the probability mass placed within a collision boundary $[-1.0, +1.0]$ around the obstacle:
   $$z = \frac{1.0 - 0.0}{3.0} = 0.333333$$
   $$P(|X| \le 1.0) = \Phi(0.333333) - \Phi(-0.333333) = 0.630559 - 0.369441 = \mathbf{0.261117}$$
   The unimodal Gaussian hallucinates that the agent crashes directly into the solid obstacle **$26.11\%$ of the time**!

3. **Discrete Categorical Representation:**
   In contrast, represent the latent state using a categorical distribution over discrete tokens $\mathcal{K} = \{-3.0, +3.0\}$ with probabilities $\mathbf{p} = [0.5, 0.5]^T$:
   $$P(X = -3.0) = 0.5, \quad P(X = +3.0) = 0.5, \quad P(X \in [-1.0, +1.0]) = \mathbf{0.000000}$$
   The categorical distribution allocates strictly zero probability mass to the forbidden interior $[-1.0, +1.0]$, preserving sharp, non-blurry discrete modes across long imagined rollouts. $\blacksquare$

---

### Illustration 2: RSSM Forward Roll in Latent Space: Deterministic GRU State $h_t$, Stochastic Prior $p(z_t|h_t)$, Posterior $q(z_t|h_t, x_t)$, and KL Divergence Calculation
**Problem:**
Given a 2D recurrent latent space ($d_h = 2, d_z = 2$) with previous deterministic state $h_{t-1} = [0.50, -0.20]^T$, previous sampled latent $z_{t-1} = [0.10, 0.40]^T$, executed action $a_{t-1} = [1.00, 0.00]^T$, and current observation feature $e(x_t) = [0.80, -0.50]^T$.
The model parameters are:
- Recurrent weights:
  $$W_{hh} = \begin{bmatrix} 0.60 & -0.10 \\ 0.20 & 0.50 \end{bmatrix}, \quad W_{hz} = \begin{bmatrix} 0.30 & 0.20 \\ -0.40 & 0.10 \end{bmatrix}, \quad W_{ha} = \begin{bmatrix} 0.50 & 0.00 \\ 0.10 & 0.30 \end{bmatrix}, \quad b_h = \begin{bmatrix} 0.10 \\ -0.10 \end{bmatrix}$$
- Prior network:
  $$W_\mu^p = \begin{bmatrix} 0.80 & -0.40 \\ 0.10 & 0.60 \end{bmatrix}, \quad b_\mu^p = \begin{bmatrix} 0.00 \\ 0.20 \end{bmatrix}, \quad W_\sigma^p = \begin{bmatrix} 0.20 & 0.10 \\ -0.10 & 0.30 \end{bmatrix}, \quad b_\sigma^p = \begin{bmatrix} 0.00 \\ 0.10 \end{bmatrix}$$
- Posterior network (operating on concatenated vector $[h_t; e(x_t)] \in \mathbb{R}^4$):
  $$W_\mu^q = \begin{bmatrix} 0.50 & -0.20 & 0.60 & 0.10 \\ 0.20 & 0.40 & -0.30 & 0.50 \end{bmatrix}, \quad b_\mu^q = \begin{bmatrix} 0.10 \\ 0.00 \end{bmatrix}$$
  $$W_\sigma^q = \begin{bmatrix} 0.10 & 0.00 & 0.20 & -0.10 \\ 0.00 & 0.20 & 0.10 & 0.20 \end{bmatrix}, \quad b_\sigma^q = \begin{bmatrix} -0.20 \\ -0.10 \end{bmatrix}$$
where standard deviations use the softplus activation $\sigma = \ln(1 + \exp(\text{raw}))$.
Compute:
1. The new deterministic state $h_t = \tanh(W_{hh} h_{t-1} + W_{hz} z_{t-1} + W_{ha} a_{t-1} + b_h)$.
2. The prior parameters $\mu_p, \sigma_p$ for $p(z_t \mid h_t)$.
3. The posterior parameters $\mu_q, \sigma_q$ for $q(z_t \mid h_t, x_t)$.
4. The exact analytical Gaussian KL divergence $D_{\text{KL}}(q \parallel p)$.

**Solution:**

#### Step 1: Compute Deterministic Recurrent State $h_t$
Compute each matrix-vector product:
$$W_{hh} h_{t-1} = \begin{bmatrix} 0.60(0.50) + (-0.10)(-0.20) \\ 0.20(0.50) + 0.50(-0.20) \end{bmatrix} = \begin{bmatrix} 0.30 + 0.02 \\ 0.10 - 0.10 \end{bmatrix} = \begin{bmatrix} 0.320000 \\ 0.000000 \end{bmatrix}$$

$$W_{hz} z_{t-1} = \begin{bmatrix} 0.30(0.10) + 0.20(0.40) \\ -0.40(0.10) + 0.10(0.40) \end{bmatrix} = \begin{bmatrix} 0.03 + 0.08 \\ -0.04 + 0.04 \end{bmatrix} = \begin{bmatrix} 0.110000 \\ 0.000000 \end{bmatrix}$$

$$W_{ha} a_{t-1} = \begin{bmatrix} 0.50(1.00) + 0.00(0.00) \\ 0.10(1.00) + 0.30(0.00) \end{bmatrix} = \begin{bmatrix} 0.500000 \\ 0.100000 \end{bmatrix}$$

Sum the components and bias vector:
$$u_t = \begin{bmatrix} 0.320000 \\ 0.000000 \end{bmatrix} + \begin{bmatrix} 0.110000 \\ 0.000000 \end{bmatrix} + \begin{bmatrix} 0.500000 \\ 0.100000 \end{bmatrix} + \begin{bmatrix} 0.100000 \\ -0.100000 \end{bmatrix} = \begin{bmatrix} 1.030000 \\ 0.000000 \end{bmatrix}$$

Apply the hyperbolic tangent activation:
$$h_t = \begin{bmatrix} \tanh(1.030000) \\ \tanh(0.000000) \end{bmatrix} = \begin{bmatrix} \mathbf{0.773908} \\ \mathbf{0.000000} \end{bmatrix}$$

#### Step 2: Compute Imagination Prior $p(z_t \mid h_t)$
1. Prior Mean $\mu_p$:
   $$\mu_p = W_\mu^p h_t + b_\mu^p = \begin{bmatrix} 0.80(0.773908) - 0.40(0.0) + 0.00 \\ 0.10(0.773908) + 0.60(0.0) + 0.20 \end{bmatrix} = \begin{bmatrix} 0.619127 \\ 0.077391 + 0.20 \end{bmatrix} = \begin{bmatrix} \mathbf{0.619127} \\ \mathbf{0.277391} \end{bmatrix}$$
2. Prior Standard Deviation $\sigma_p$:
   $$\text{raw}_p = W_\sigma^p h_t + b_\sigma^p = \begin{bmatrix} 0.20(0.773908) + 0.10(0.0) + 0.00 \\ -0.10(0.773908) + 0.30(0.0) + 0.10 \end{bmatrix} = \begin{bmatrix} 0.154782 \\ -0.077391 + 0.10 \end{bmatrix} = \begin{bmatrix} 0.154782 \\ 0.022609 \end{bmatrix}$$
   $$\sigma_{p, 1} = \ln(1 + \exp(0.154782)) = \ln(1 + 1.167404) = \ln(2.167404) = \mathbf{0.773530}$$
   $$\sigma_{p, 2} = \ln(1 + \exp(0.022609)) = \ln(1 + 1.022867) = \ln(2.022867) = \mathbf{0.704516}$$
   $$\sigma_p = \begin{bmatrix} \mathbf{0.773530} \\ \mathbf{0.704516} \end{bmatrix}$$

#### Step 3: Compute Sensory Posterior $q(z_t \mid h_t, x_t)$
Construct the concatenated feature vector:
$$f_t = \begin{bmatrix} h_t \\ e(x_t) \end{bmatrix} = \begin{bmatrix} 0.773908 \\ 0.000000 \\ 0.800000 \\ -0.500000 \end{bmatrix}$$
1. Posterior Mean $\mu_q$:
   $$\mu_{q, 1} = 0.50(0.773908) - 0.20(0.0) + 0.60(0.80) + 0.10(-0.50) + 0.10 = 0.386954 + 0.480000 - 0.050000 + 0.10 = \mathbf{0.916954}$$
   $$\mu_{q, 2} = 0.20(0.773908) + 0.40(0.0) - 0.30(0.80) + 0.50(-0.50) + 0.00 = 0.154782 - 0.240000 - 0.250000 = \mathbf{-0.335218}$$
   $$\mu_q = \begin{bmatrix} \mathbf{0.916954} \\ \mathbf{-0.335218} \end{bmatrix}$$
2. Posterior Standard Deviation $\sigma_q$:
   $$\text{raw}_{q, 1} = 0.10(0.773908) + 0.00(0.0) + 0.20(0.80) - 0.10(-0.50) - 0.20 = 0.077391 + 0.160000 + 0.050000 - 0.20 = 0.087391$$
   $$\text{raw}_{q, 2} = 0.00(0.773908) + 0.20(0.0) + 0.10(0.80) + 0.20(-0.50) - 0.10 = 0.080000 - 0.100000 - 0.10 = -0.120000$$
   $$\sigma_{q, 1} = \ln(1 + \exp(0.087391)) = \ln(1 + 1.091323) = \ln(2.091323) = \mathbf{0.737797}$$
   $$\sigma_{q, 2} = \ln(1 + \exp(-0.120000)) = \ln(1 + 0.886920) = \ln(1.886920) = \mathbf{0.634946}$$
   $$\sigma_q = \begin{bmatrix} \mathbf{0.737797} \\ \mathbf{0.634946} \end{bmatrix}$$

#### Step 4: Step-by-Step Analytical Gaussian KL Divergence
Recall the coordinate-wise formula:
$$D_{\text{KL}, j} = \ln\left( \frac{\sigma_{p, j}}{\sigma_{q, j}} \right) + \frac{\sigma_{q, j}^2 + (\mu_{q, j} - \mu_{p, j})^2}{2 \sigma_{p, j}^2} - 0.500000$$

1. **For Latent Dimension $j = 1$:**
   - Log-ratio: $\ln\left(\frac{0.773530}{0.737797}\right) = \ln(1.048432) = \mathbf{0.047295}$
   - Mean deviation: $\mu_{q, 1} - \mu_{p, 1} = 0.916954 - 0.619127 = 0.297828$
   - Squared deviation: $(0.297828)^2 = 0.088701$
   - Posterior variance: $\sigma_{q, 1}^2 = (0.737797)^2 = 0.544344$
   - Prior variance: $\sigma_{p, 1}^2 = (0.773530)^2 = 0.598348 \implies 2 \sigma_{p, 1}^2 = 1.196696$
   - Quadratic fraction: $\frac{0.544344 + 0.088701}{1.196696} = \frac{0.633045}{1.196696} = \mathbf{0.528994}$
   - Coordinate divergence:
     $$D_{\text{KL}, 1} = 0.047295 + 0.528994 - 0.500000 = \mathbf{0.076290}$$

2. **For Latent Dimension $j = 2$:**
   - Log-ratio: $\ln\left(\frac{0.704516}{0.634946}\right) = \ln(1.109568) = \mathbf{0.103970}$
   - Mean deviation: $\mu_{q, 2} - \mu_{p, 2} = -0.335218 - 0.277391 = -0.612609$
   - Squared deviation: $(-0.612609)^2 = 0.375290$
   - Posterior variance: $\sigma_{q, 2}^2 = (0.634946)^2 = 0.403157$
   - Prior variance: $\sigma_{p, 2}^2 = (0.704516)^2 = 0.496342 \implies 2 \sigma_{p, 2}^2 = 0.992685$
   - Quadratic fraction: $\frac{0.403157 + 0.375290}{0.992685} = \frac{0.778447}{0.992685} = \mathbf{0.784183}$
   - Coordinate divergence:
     $$D_{\text{KL}, 2} = 0.103970 + 0.784183 - 0.500000 = \mathbf{0.388154}$$

3. **Total Latent KL Divergence:**
   $$D_{\text{KL}}(q \;\Vert\; p) = D_{\text{KL}, 1} + D_{\text{KL}, 2} = 0.076290 + 0.388154 = \mathbf{0.464444} \quad \blacksquare$$

---

### Illustration 3: Latent Imagination Trajectory Rollout of Horizon $H = 4$ and Backward $\lambda$-Return Calculation ($\lambda = 0.95, \gamma = 0.99$)
**Problem:**
An agent executes an imagined latent rollout of horizon $H = 4$ starting from initial seed state $s_0$. The model generates predicted stage rewards and critic evaluations:
- Imagined Rewards:
  $$\hat{r}_0 = 1.000000, \quad \hat{r}_1 = 0.500000, \quad \hat{r}_2 = 2.000000, \quad \hat{r}_3 = 1.500000$$
- Critic Value Baseline Estimates:
  $$v(s_0) = 10.000000, \quad v(s_1) = 12.000000, \quad v(s_2) = 11.000000, \quad v(s_3) = 15.000000, \quad v(s_4) = 14.000000$$
The discount factor is $\gamma = 0.990000$ and the bootstrapping parameter is $\lambda = 0.950000$.
Compute step-by-step the backward recursive $\lambda$-return targets $V_3^\lambda, V_2^\lambda, V_1^\lambda, V_0^\lambda$ used to train the critic network.

**Solution:**

#### Boundary Condition at Horizon Limit $t = 4$:
At the terminal lookahead horizon $H = 4$, the return is initialized to the critic baseline estimate at the terminal latent state:
$$V_4^\lambda \triangleq v(s_4) = \mathbf{14.000000}$$

#### Backward Step $t = 3$:
Recall the recursive formula proven in Derivation 11.24.2:
$$V_t^\lambda = \hat{r}_t + \gamma \left[ (1 - \lambda) v(s_{t+1}) + \lambda V_{t+1}^\lambda \right]$$
1. Target combination for next state $s_4$:
   $$(1 - \lambda) v(s_4) = (1.00 - 0.95) \times 14.000000 = 0.05 \times 14.000000 = \mathbf{0.700000}$$
   $$\lambda V_4^\lambda = 0.95 \times 14.000000 = \mathbf{13.300000}$$
   $$\text{Combined Target} = 0.700000 + 13.300000 = \mathbf{14.000000}$$
2. Discounting and immediate reward addition:
   $$\text{Discounted Target} = \gamma \times 14.000000 = 0.99 \times 14.000000 = \mathbf{13.860000}$$
   $$V_3^\lambda = \hat{r}_3 + 13.860000 = 1.500000 + 13.860000 = \mathbf{15.360000}$$

#### Backward Step $t = 2$:
1. Target combination for next state $s_3$:
   $$(1 - \lambda) v(s_3) = 0.05 \times 15.000000 = \mathbf{0.750000}$$
   $$\lambda V_3^\lambda = 0.95 \times 15.360000 = \mathbf{14.592000}$$
   $$\text{Combined Target} = 0.750000 + 14.592000 = \mathbf{15.342000}$$
2. Discounting and immediate reward addition:
   $$\text{Discounted Target} = 0.99 \times 15.342000 = \mathbf{15.188580}$$
   $$V_2^\lambda = \hat{r}_2 + 15.188580 = 2.000000 + 15.188580 = \mathbf{17.188580}$$

#### Backward Step $t = 1$:
1. Target combination for next state $s_2$:
   $$(1 - \lambda) v(s_2) = 0.05 \times 11.000000 = \mathbf{0.550000}$$
   $$\lambda V_2^\lambda = 0.95 \times 17.188580 = \mathbf{16.329151}$$
   $$\text{Combined Target} = 0.550000 + 16.329151 = \mathbf{16.879151}$$
2. Discounting and immediate reward addition:
   $$\text{Discounted Target} = 0.99 \times 16.879151 = \mathbf{16.710359}$$
   $$V_1^\lambda = \hat{r}_1 + 16.710359 = 0.500000 + 16.710359 = \mathbf{17.210359}$$

#### Backward Step $t = 0$:
1. Target combination for next state $s_1$:
   $$(1 - \lambda) v(s_1) = 0.05 \times 12.000000 = \mathbf{0.600000}$$
   $$\lambda V_1^\lambda = 0.95 \times 17.210359 = \mathbf{16.349842}$$
   $$\text{Combined Target} = 0.600000 + 16.349842 = \mathbf{16.949842}$$
2. Discounting and immediate reward addition:
   $$\text{Discounted Target} = 0.99 \times 16.949842 = \mathbf{16.780343}$$
   $$V_0^\lambda = \hat{r}_0 + 16.780343 = 1.000000 + 16.780343 = \mathbf{17.780343}$$

#### Final Return Summary Vector:
$$\mathbf{V}^\lambda = [V_0^\lambda, V_1^\lambda, V_2^\lambda, V_3^\lambda, V_4^\lambda]^T = [\mathbf{17.780343}, \mathbf{17.210359}, \mathbf{17.188580}, \mathbf{15.360000}, \mathbf{14.000000}]^T \quad \blacksquare$$

---

### Illustration 4: DreamerV3 Symlog Transformation and Symexp Inversion on Extreme Reward Values ($R \in \{0.001, 1.0, 100.0, 10000.0\}$)
**Problem:**
DreamerV3 processes returns spanning 7 orders of magnitude.
1. For rewards $R \in \{0.001, 1.0, 100.0, 10000.0\}$ and negative return $R = -100.0$, compute the forward symlog transformation $y = \operatorname{symlog}(R)$, the local derivative $\frac{dy}{dR}$, and the reconstructed value $\hat{R} = \operatorname{symexp}(y)$.
2. For $R = 100.0$ ($y = 4.615121$), project $y$ onto an integer support grid $\mathcal{B} = \{-10, -9, \dots, +9, +10\}$ with bin spacing $\Delta b = 1.0$. Find the active adjacent bins $b_k, b_{k+1}$, compute the two-hot target distribution probabilities $p_k^*, p_{k+1}^*$, and verify expectation preservation.

**Solution:**

#### Part 1: Forward Symlog, Derivative, and Inverse Symexp
Recall the mathematical formulas from Derivation 11.24.3:
$$\operatorname{symlog}(R) = \operatorname{sign}(R) \ln(|R| + 1), \quad \frac{d}{dR}\operatorname{symlog}(R) = \frac{1}{|R| + 1}, \quad \operatorname{symexp}(y) = \operatorname{sign}(y) (\exp(|y|) - 1)$$

1. **For $R = 0.001000$ (Infinitesimal Regime):**
   - $|R| + 1 = 1.001000$
   - $y = +1 \times \ln(1.001000) = \mathbf{0.001000}$ (exact linear preservation: $\ln(1 + \epsilon) \approx \epsilon$)
   - Derivative: $\frac{d}{dR} = \frac{1}{1.001000} = \mathbf{0.999001}$
   - Inversion: $\operatorname{symexp}(0.001000) = +1 \times (\exp(0.0009995) - 1) = \mathbf{0.001000}$

2. **For $R = 1.000000$ (Unit Regime):**
   - $|R| + 1 = 2.000000$
   - $y = +1 \times \ln(2.000000) = \mathbf{0.693147}$
   - Derivative: $\frac{d}{dR} = \frac{1}{2.000000} = \mathbf{0.500000}$
   - Inversion: $\operatorname{symexp}(0.693147) = \exp(0.693147) - 1 = 2.000000 - 1 = \mathbf{1.000000}$

3. **For $R = 100.000000$ (Moderate Arcade Score):**
   - $|R| + 1 = 101.000000$
   - $y = +1 \times \ln(101.000000) = \mathbf{4.615121}$
   - Derivative: $\frac{d}{dR} = \frac{1}{101.000000} = \mathbf{0.009901}$ (suppresses gradients by a factor of 101)
   - Inversion: $\operatorname{symexp}(4.615121) = \exp(4.615121) - 1 = 101.000000 - 1 = \mathbf{100.000000}$

4. **For $R = 10000.000000$ (Extreme Minecraft / Atari Spike):**
   - $|R| + 1 = 10001.000000$
   - $y = +1 \times \ln(10001.000000) = \mathbf{9.210440}$
   - Derivative: $\frac{d}{dR} = \frac{1}{10001.000000} = \mathbf{0.000100}$ (gradient explosion completely arrested!)
   - Inversion: $\operatorname{symexp}(9.210440) = \exp(9.210440) - 1 = 10001.000000 - 1 = \mathbf{10000.000000}$

5. **For $R = -100.000000$ (Negative Penalty):**
   - $|R| + 1 = 101.000000$, $\operatorname{sign}(R) = -1$
   - $y = -1 \times \ln(101.000000) = \mathbf{-4.615121}$
   - Derivative: $\frac{d}{dR} = \frac{1}{|-100| + 1} = \frac{1}{101.000000} = \mathbf{0.009901}$
   - Inversion: $\operatorname{symexp}(-4.615121) = -1 \times (\exp(4.615121) - 1) = -(101.000000 - 1) = \mathbf{-100.000000}$

| Input Value $R$ | Transformed Target $y = \operatorname{symlog}(R)$ | Local Gradient Scale $\frac{dy}{dR}$ | Inverted Value $\operatorname{symexp}(y)$ | Absolute Reconstruction Error |
| :---: | :---: | :---: | :---: | :---: |
| **$0.001000$** | $+0.001000$ | $0.999001$ | $0.001000$ | $< 10^{-12}$ |
| **$1.000000$** | $+0.693147$ | $0.500000$ | $1.000000$ | $< 10^{-12}$ |
| **$100.000000$** | $+4.615121$ | $0.009901$ | $100.000000$ | $< 10^{-12}$ |
| **$10000.000000$** | $+9.210440$ | $0.000100$ | $10000.000000$ | $< 10^{-12}$ |
| **$-100.000000$** | $-4.615121$ | $0.009901$ | $-100.000000$ | $< 10^{-12}$ |

#### Part 2: Two-Hot Categorical Projection for $y = 4.615121$
On the integer support grid $\mathcal{B}$, the target $y = 4.615121$ falls between:
$$b_k = 4.000000 \quad \text{and} \quad b_{k+1} = 5.000000$$
The grid spacing is $\Delta b = 5.0 - 4.0 = 1.0$.
Apply the two-hot projection formula:
$$p_k^* = \frac{b_{k+1} - y}{\Delta b} = \frac{5.000000 - 4.615121}{1.000000} = \mathbf{0.384879}$$
$$p_{k+1}^* = \frac{y - b_k}{\Delta b} = \frac{4.615121 - 4.000000}{1.000000} = \mathbf{0.615121}$$
All other bins $j \notin \{k, k+1\}$ have $p_j^* = 0.0$.
1. Check probability normalization:
   $$\sum_j p_j^* = 0.384879 + 0.615121 = \mathbf{1.000000}$$
2. Check expectation preservation:
   $$\mathbb{E}_{\mathbf{p}^*}[b] = (0.384879 \times 4.000000) + (0.615121 \times 5.000000) = 1.539516 + 3.075605 = \mathbf{4.615121} \equiv y$$
3. Invert expectation back to physical reward:
   $$\hat{R} = \operatorname{symexp}(\mathbb{E}_{\mathbf{p}^*}[b]) = \exp(4.615121) - 1 = 101.000000 - 1 = \mathbf{100.000000} \quad \blacksquare$$

---

### Illustration 5: Actor Update in Latent Imagination Using Straight-Through Gradient Estimator Through Categorical Latents
**Problem:**
In DreamerV3, the latent state $z$ is discrete, consisting of categorical distributions.
Consider a single latent variable with $K = 3$ discrete classes. The world model's transition predictor outputs unnormalized logits:
$$\mathbf{l} = [2.000000, 1.000000, 0.000000]^T$$
These logits are parameterized as a linear function of a scalar policy action $a = 0.800000$ via $\mathbf{l} = W_\theta a$, with weight vector $W_\theta = [1.500000, 0.500000, -0.500000]^T$.
Downstream in imagination, a linear value evaluation head assigns returns based on the chosen discrete latent:
$$V(z) = \mathbf{w}^T z, \quad \text{where } \mathbf{w} = [5.000000, 2.000000, -1.000000]^T$$
1. Compute the categorical probabilities $\mathbf{p} = \operatorname{softmax}(\mathbf{l})$ and the one-hot discrete sample $z_{\text{hard}} = \operatorname{one\_hot}(\arg\max(\mathbf{l}))$.
2. Formulate the straight-through estimator $z_{\text{st}} = z_{\text{hard}} + \mathbf{p} - \operatorname{sg}[\mathbf{p}]$ and compute the forward value $V(z_{\text{hard}})$.
3. Compute the analytic backward gradient $\frac{\partial V}{\partial \mathbf{l}}$ through the straight-through estimator and verify the zero-sum property $\sum_j \frac{\partial V}{\partial l_j} = 0$.
4. Backpropagate the gradient to the action $a$, finding the exact policy gradient $\frac{\partial V}{\partial a}$.

**Solution:**

#### Step 1: Compute Softmax Probabilities and Discrete Hard Sample
Evaluate the exponentials:
$$\exp(l_1) = \exp(2.000000) \approx 7.389056$$
$$\exp(l_2) = \exp(1.000000) \approx 2.718282$$
$$\exp(l_3) = \exp(0.000000) = 1.000000$$
Denominator:
$$Z = 7.389056 + 2.718282 + 1.000000 = 11.107338$$
Probabilities:
$$p_1 = \frac{7.389056}{11.107338} = \mathbf{0.665241}$$
$$p_2 = \frac{2.718282}{11.107338} = \mathbf{0.244728}$$
$$p_3 = \frac{1.000000}{11.107338} = \mathbf{0.090031}$$
Check normalization: $0.665241 + 0.244728 + 0.090031 = 1.000000$.
One-hot argmax sample:
$$k^* = \arg\max(2.0, 1.0, 0.0) = 1 \implies z_{\text{hard}} = \begin{bmatrix} \mathbf{1.000000} \\ \mathbf{0.000000} \\ \mathbf{0.000000} \end{bmatrix}$$

#### Step 2: Straight-Through Latent & Forward Value Evaluation
The straight-through estimator is defined as:
$$z_{\text{st}} = z_{\text{hard}} + \mathbf{p} - \operatorname{sg}[\mathbf{p}]$$
On the forward pass, $z_{\text{st}} = z_{\text{hard}} = [1, 0, 0]^T$.
The imagined value outcome is:
$$V(z_{\text{hard}}) = \mathbf{w}^T z_{\text{hard}} = 5.0(1) + 2.0(0) - 1.0(0) = \mathbf{5.000000}$$

#### Step 3: Analytical Straight-Through Gradient with Respect to Logits
On the backward pass, because $\operatorname{sg}[\cdot]$ has zero gradient, the derivative of $z_{\text{st}}$ with respect to the probability vector $\mathbf{p}$ is the identity:
$$\frac{\partial z_{\text{st}}}{\partial \mathbf{p}} = \mathbf{I} \implies \frac{\partial V}{\partial \mathbf{p}} = \mathbf{w} = \begin{bmatrix} 5.000000 \\ 2.000000 \\ -1.000000 \end{bmatrix}$$
Now apply the multivariate chain rule with the softmax Jacobian $\frac{\partial p_i}{\partial l_j} = p_i (\delta_{ij} - p_j)$:
$$\frac{\partial V}{\partial l_j} = \sum_{i=1}^3 \frac{\partial V}{\partial p_i} \frac{\partial p_i}{\partial l_j} = \sum_{i=1}^3 w_i p_i (\delta_{ij} - p_j) = w_j p_j - p_j \sum_{i=1}^3 w_i p_i$$
$$= p_j \left( w_j - \mathbb{E}_{\mathbf{p}}[\mathbf{w}] \right)$$
First, compute the expected weight $\mathbb{E}_{\mathbf{p}}[\mathbf{w}]$:
$$\mathbb{E}_{\mathbf{p}}[\mathbf{w}] = (0.665241 \times 5.0) + (0.244728 \times 2.0) + (0.090031 \times -1.0)$$
$$= 3.326205 + 0.489456 - 0.090031 = \mathbf{3.725631}$$
Now compute the centered weight deviations $w_j - \mathbb{E}_{\mathbf{p}}[\mathbf{w}]$:
$$w_1 - \mathbb{E} = 5.000000 - 3.725631 = \mathbf{+1.274369}$$
$$w_2 - \mathbb{E} = 2.000000 - 3.725631 = \mathbf{-1.725631}$$
$$w_3 - \mathbb{E} = -1.000000 - 3.725631 = \mathbf{-4.725631}$$
Multiply by the softmax probabilities $p_j$:
$$\frac{\partial V}{\partial l_1} = 0.665241 \times (+1.274369) = \mathbf{+0.847762}$$
$$\frac{\partial V}{\partial l_2} = 0.244728 \times (-1.725631) = \mathbf{-0.422311}$$
$$\frac{\partial V}{\partial l_3} = 0.090031 \times (-4.725631) = \mathbf{-0.425451}$$
Vector gradient:
$$\nabla_{\mathbf{l}} V = \begin{bmatrix} \mathbf{+0.847762} \\ \mathbf{-0.422311} \\ \mathbf{-0.425451} \end{bmatrix}$$
Verify the fundamental zero-sum conservation property of softmax gradients:
$$\sum_{j=1}^3 \frac{\partial V}{\partial l_j} = 0.847762 - 0.422311 - 0.425451 = \mathbf{0.000000}$$

#### Step 4: Policy Action Gradient $\frac{\partial V}{\partial a}$
The logits were generated from action $a$ via $\mathbf{l} = W_\theta a$. By the multivariable chain rule:
$$\frac{\partial V}{\partial a} = \sum_{j=1}^3 \frac{\partial V}{\partial l_j} \frac{\partial l_j}{\partial a} = \sum_{j=1}^3 \frac{\partial V}{\partial l_j} W_{\theta, j}$$
Evaluate the inner product:
$$\frac{\partial V}{\partial a} = (0.847762 \times 1.500000) + (-0.422311 \times 0.500000) + (-0.425451 \times -0.500000)$$
$$= 1.271643 - 0.211156 + 0.212726 = \mathbf{1.273214}$$
Because $\frac{\partial V}{\partial a} = +1.273214 > 0$, increasing action $a$ shifts probability toward Category 1 (which possesses the highest value return $w_1 = 5.0$).
This demonstrates how analytic gradients flow directly through discrete categorical world model states into continuous policy parameters. $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. DreamerV3: One Agent, All Domains (Hafner et al., Nature 2023)
DreamerV3 is the decisive validation of the latent world model approach:
- **Single Hyperparameter Set Across 7 Domains:** Atari 200k, DMControl Suite (continuous locomotion), Minecraft, DMLab, Crafter, Memory Maze, BSuite — identical $(\alpha, \beta, \gamma) = (0.5, 1.0, 0.997)$ for all. No per-domain tuning.
- **Minecraft Diamond Collection:** First RL agent to collect a diamond in vanilla Minecraft from scratch (no human demonstrations), requiring 20+ sequential tool-crafting steps over horizon $\sim$30,000 environment steps.
- **Symlog Predictions:** DreamerV3 introduces symmetric log transformations of rewards and values: $\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$, enabling the same network to handle both near-zero (BSuite) and thousands-scale (Atari Pong) reward signals without clipping.

### 2. Autonomous Driving Video World Models (2023–2024)
Neural world models are reshaping self-driving development pipelines:
- **Wayve GAIA-1 (Hu et al., 2023):** A 9B-parameter autoregressive video diffusion world model conditioned on ego-vehicle actions and language descriptions. Generates photorealistic 10-second driving videos of rare edge cases (pedestrian jaywalking, sudden lane changes) for synthetic data augmentation.
- **Tesla FSD Neural Rendering:** Tesla's FSD stack uses Occupancy Networks (neural world models) to predict 3D scene evolution 2 seconds into the future, enabling the planner to simulate potential collision trajectories before executing any real maneuver.
- **UniSim (Yang et al., CVPR 2024):** A universal simulator learning a world model from 100M+ real-world driving frames, enabling zero-shot policy training for new vehicle types without physical testing.

### 3. IRIS & Transformer World Models for Sample-Efficient Game Playing
- **IRIS (Micheli et al., ICLR 2023):** Replaces the RSSM with a discrete VQ-VAE + Transformer world model, achieving human-level performance on 10 of 26 Atari games in only 100k real environment steps — orders of magnitude fewer than model-free agents.
- **TWM (Robine et al., 2023):** Transformer World Model uses causal self-attention over token sequences of $(o_t, a_t, r_t)$ triples to predict future frames, enabling arbitrarily long-range temporal dependencies beyond what LSTM-based RSSMs capture.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Closed-form Gaussian KL divergence verification matching $0.638147$ to $< 10^{-6}$.
   - Backward recursive $\lambda$-return calculation matching $V_2^\lambda = 14.0000, V_1^\lambda = 13.5200$ to $< 10^{-12}$.
2. **Complete PyTorch-Vectorized RSSM Latent Dynamics Layer:**
   - Deterministic GRU recurrent transition.
   - Stochastic Gaussian posterior and prior with reparameterization trick.
   - Vectorized KL divergence computation with KL balancing ($\alpha = 0.8$).
3. **Actor-Critic Imagination Rollout Pipeline:**
   - Generates $H$-step imagined latent rollouts and computes analytical $\lambda$-returns.
4. **Section 6 Solved Numerical Illustrations Verification:**
   - Mode-averaging collision calculation in Gaussian vs. Categorical latents ($26.11\%$ vs. $0.0\%$).
   - 2D RSSM forward state update, prior/posterior estimation, and analytical KL divergence ($0.464444$).
   - Horizon $H=4$ backward $\lambda$-return recursion matching $\mathbf{V}^\lambda = [17.780343, 17.210359, 17.188580, 15.360000, 14.000000]$ to machine precision.
   - Symlog transformation, derivative bounds, symexp exact inversion, and two-hot categorical target expectation preservation.
   - Straight-through policy gradient through categorical latents with zero-sum gradient verification ($\sum \nabla_l V = 0$) and analytic action gradient ($1.273214$).

See implementation in:
[`11_reinforcement_learning/code/24_world_models_and_dreamer.py`](./code/24_world_models_and_dreamer.py)
