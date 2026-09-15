# Chapter 4.6: Monte Carlo Methods & Importance Sampling

---

## Part 1: Intuition & 101 Motivation

In deep learning and modern Bayesian inference, we are constantly faced with intractable high-dimensional integrals:
- **Expected Reward in Reinforcement Learning:** $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int R(\tau) p(\tau; \theta) d\tau$
- **Marginal Evidence in Latent Variable Models (VAEs):** $p(x) = \int p(x \mid z) p(z) dz$
- **Expected Test Risk:** $\mathcal{R}(\theta) = \int \ell(f_\theta(x), y) p(x, y) dx dy$

Why can't we compute these integrals using standard calculus or deterministic numerical grids (like the Trapezoidal rule or Simpson's rule)?

### The Curse of Dimensionality
Suppose you evaluate an integral in $d = 50$ dimensions (a tiny latent space in a VAE).
If you place a modest grid of just $10$ points along each axis:
$$\text{Total Grid Points} = 10^{50}$$
Evaluating $10^{50}$ function calls on the world's fastest supercomputer would take longer than the age of the universe! Deterministic quadrature has an error convergence rate of $\mathcal{O}(N^{-1/d})$, which grinds to a halt as dimension $d$ grows.

### The Monte Carlo Revolution
In 1946, Stanislaw Ulam and John von Neumann proposed a radical alternative: **replace the deterministic grid with random sampling**.

The error of a Monte Carlo estimator decays at rate:
$$\mathbf{\text{Error} = \mathcal{O}\left(\frac{1}{\sqrt{N}}\right)}$$
**This rate is completely independent of the dimension $d$!** Whether you integrate in 1 dimension or 1,000,000 dimensions, taking 10,000 random samples provides the exact same precision.

Furthermore, when the target distribution is difficult to sample from, or when estimating rare, high-stakes events (e.g. autonomous vehicle crashes or reinforcement learning policy updates), **Importance Sampling** allows us to sample from a convenient proposal distribution $q(x)$ and mathematically re-weight the results.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Crude Monte Carlo Integration
We wish to compute the expected value of a function $h(x)$ under probability density $p(x)$:
$$I = \mathbb{E}_{X \sim p}[h(X)] = \int_{\mathcal{X}} h(x) p(x) \, dx$$

Let $X_1, X_2, \dots, X_N \overset{i.i.d.}{\sim} p(x)$ be $N$ independent random draws from $p(x)$.
The **Crude Monte Carlo Estimator** is:
$$\hat{I}_N = \frac{1}{N} \sum_{i=1}^N h(X_i)$$

#### Statistical Properties:
1. **Unbiasedness:**
   $$\mathbb{E}[\hat{I}_N] = \frac{1}{N} \sum_{i=1}^N \mathbb{E}[h(X_i)] = \frac{1}{N} (N \cdot I) = I$$
2. **Variance:**
   $$\text{Var}(\hat{I}_N) = \frac{1}{N^2} \sum_{i=1}^N \text{Var}(h(X_i)) = \frac{\sigma_h^2}{N}$$
   where $\sigma_h^2 = \text{Var}_p(h(X)) = \int (h(x) - I)^2 p(x) dx$.
3. **Standard Error (SE):**
   $$\text{SE}(\hat{I}_N) = \frac{\sigma_h}{\sqrt{N}} = \mathcal{O}\left(N^{-1/2}\right)$$
By the Central Limit Theorem:
$$\frac{\hat{I}_N - I}{\sigma_h / \sqrt{N}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } N \to \infty$$

---

### 2. Importance Sampling (IS)
Suppose sampling directly from $p(x)$ is impossible, or $p(x)$ is concentrated in regions where $h(x) \approx 0$ (a rare-event problem).
Let $q(x)$ be an alternative **proposal probability distribution** satisfying the **support condition**:
$$q(x) > 0 \quad \text{whenever } h(x) p(x) \ne 0$$

Rewrite the integral identically:
$$I = \int_{\mathcal{X}} h(x) p(x) \, dx = \int_{\mathcal{X}} h(x) \frac{p(x)}{q(x)} q(x) \, dx = \mathbb{E}_{X \sim q}\left[ h(X) \frac{p(X)}{q(X)} \right]$$

Define the **Importance Weight** (likelihood ratio):
$$w(x) = \frac{p(x)}{q(x)}$$

The **Importance Sampling Estimator** is:
$$\mathbf{\hat{I}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^N h(X_i) w(X_i) = \frac{1}{N} \sum_{i=1}^N h(X_i) \frac{p(X_i)}{q(X_i)}, \quad X_i \overset{i.i.d.}{\sim} q(x)}$$

#### Unbiasedness Proof:
$$\mathbb{E}_{q}[\hat{I}_{\text{IS}}] = \mathbb{E}_{X \sim q}[h(X) w(X)] = \int h(x) \frac{p(x)}{q(x)} q(x) \, dx = \int h(x) p(x) \, dx = I \quad \blacksquare$$

---

### 3. Variance of Importance Sampling & The Zero-Variance Theorem
The variance of the importance sampling estimator is:
$$\text{Var}_q(\hat{I}_{\text{IS}}) = \frac{1}{N} \text{Var}_q(h(X) w(X)) = \frac{1}{N} \left[ \int \frac{(h(x) p(x))^2}{q(x)} \, dx - I^2 \right]$$

#### Theorem: The Optimal Proposal Distribution (Zero Variance!)
Assume $h(x) \ge 0$. If we choose the proposal distribution to be:
$$\mathbf{q^*(x) = \frac{h(x) p(x)}{\int h(x') p(x') dx'} = \frac{h(x) p(x)}{I}}$$
then the variance of the estimator is **identically zero**:
$$\text{Var}_{q^*}(\hat{I}_{\text{IS}}) = 0$$

**Proof:**
Substitute $q^*(x)$ into the weighted evaluation:
$$h(x) w(x) = h(x) \frac{p(x)}{q^*(x)} = h(x) \frac{p(x)}{\frac{h(x) p(x)}{I}} = I \quad (\text{a constant for all } x!)$$
Every single sample produces the exact integral value $I$, so variance is strictly zero! $\blacksquare$

*Practical Significance:* While computing the normalizer of $q^*(x)$ is as hard as computing $I$ itself, this theorem proves that **to minimize variance, $q(x)$ should be shaped proportional to $|h(x)| p(x)$**.

---

### 4. Pathology: Weight Degeneracy & Infinite Variance
If the proposal $q(x)$ decays faster than $p(x)$ in the tails, the ratio $w(x) = \frac{p(x)}{q(x)} \to \infty$ explodes!
The second moment integral:
$$\int \frac{(h(x) p(x))^2}{q(x)} \, dx = \infty$$
When this happens, the importance sampling estimator has **infinite variance**. In empirical runs, thousands of samples will have near-zero weights, while a single catastrophic outlier sample receives an astronomical weight, crashing the estimator!

> [!WARNING]
> **The Golden Rule of Importance Sampling:**
> Always choose a proposal $q(x)$ with **heavier tails** than $p(x)$.
> Never use a Gaussian proposal to sample from a Student's $t$ or Cauchy distribution!

---

### 5. Effective Sample Size (ESS)
When importance weights are non-uniform, $N$ proposal samples do not provide $N$ independent pieces of information.
Kong (1992) defined the **Effective Sample Size (ESS)**:
$$\mathbf{\text{ESS} = \frac{\left( \sum_{i=1}^N w_i \right)^2}{\sum_{i=1}^N w_i^2} = \frac{1}{\sum_{i=1}^N \bar{w}_i^2} \in [1, N]}$$
where $\bar{w}_i = \frac{w_i}{\sum_{j=1}^N w_j}$ are the normalized weights.
- If all weights are identical ($w_i = c$), $\text{ESS} = N$ (100% statistical efficiency).
- If one single sample dominates with $w_1 = 1$ and all other $w_i = 0$, $\text{ESS} = 1$ (complete weight collapse).

---

## Part 3: Geometric & Algebraic Interpretation

### Rejection Sampling Geometry
In **Rejection Sampling**, we enclose the target distribution $p(x)$ beneath an envelope $M q(x)$ such that $p(x) \le M q(x)$ for all $x \in \mathcal{X}$.

```
                 Envelope M q(x)
               ▲
               │          ╭─────────╮
               │        ╭─╯         ╰─╮
               │      ╭─╯   p(x)      ╰─╮
               │     ╭╯   Target        ╰╮
               │    ╭╯                   ╰╮  ● Accepted sample (under p(x))
               │   ╭╯                     ╰╮ ✕ Rejected sample (between p and Mq)
               └───┴───────────────────────┴──► x
```

- We sample $x \sim q(x)$ and $u \sim \text{Uniform}(0, 1)$.
- We accept $x$ if $u \le \frac{p(x)}{M q(x)}$.
- The acceptance rate is $\frac{1}{M}$.

**The Geometric Collapse in High Dimensions:**
In $d$ dimensions, to ensure $M q(x) \ge p(x)$ everywhere, $M$ scales exponentially with dimension:
$$M \sim \left( \frac{\sigma_q}{\sigma_p} \right)^d$$
If $\frac{\sigma_q}{\sigma_p} = 1.1$ in $d = 100$ dimensions, $M \approx 1.1^{100} \approx 13{,}780$.
You must reject **99.993% of all samples**!
This geometric bottleneck is why high-dimensional deep learning completely abandons rejection sampling in favor of Importance Sampling and Markov Chain Monte Carlo (MCMC).

---

## Part 4: Real-World Analogy

### The Rare Crash Test Analogy
Imagine testing the safety of an autonomous self-driving car.
- **Crude Monte Carlo (Naive Real-World Driving):**
  A fatal accident occurs once every 100,000,000 miles ($p \approx 10^{-8}$).
  If you simulate 1,000,000 miles of highway driving under sunny skies, you will observe **0 accidents**.
  Your empirical estimator concludes: $P(\text{accident}) = 0.0$! You would need to simulate billions of miles to record even 10 accidents.
- **Importance Sampling (Adversarial Stress Testing):**
  You alter the simulation environment ($q(x)$): dense freezing fog, sudden tire blowouts, icy bridge curves.
  Now, accidents happen frequently (e.g. 20% of the simulated miles).
  You record the exact conditions of every crash, and mathematically down-weight each event by the likelihood ratio $w(x) = \frac{P(\text{blizzard})}{P(\text{sunny})}$ to compute the true real-world risk with only 10,000 simulations!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us estimate a **rare-event tail expectation** by hand using both Crude Monte Carlo and Importance Sampling.

### 1. Problem Setup & The Target Integral
Target problem: Compute the expected value of $h(X) = X^2$ in the rare extreme tail $X \ge 3.0$ under a standard Gaussian $X \sim \mathcal{N}(0, 1)$:
$$I = \mathbb{E}_{X \sim \mathcal{N}(0, 1)}[X^2 \cdot \mathbf{1}_{\{X \ge 3.0\}}] = \int_3^\infty x^2 \frac{1}{\sqrt{2\pi}} e^{-x^2 / 2} \, dx$$
*(Exact analytical ground truth from integration by parts: $I = \frac{3}{\sqrt{2\pi}} e^{-4.5} + (1 - \Phi(3.0)) \approx 0.01333 + 0.00135 \approx \mathbf{0.01468}$)*

#### The Proposal Distribution $q(x)$:
Under $\mathcal{N}(0, 1)$, the probability of $X \ge 3$ is only $0.135\%$.
To focus samples where the function matters, we choose an **Importance Proposal shifted to the tail**:
$$q(x) = \mathcal{N}(\mu_q = 3.0, \,\, \sigma_q^2 = 1.0)$$
The likelihood ratio weight is:
$$w(x) = \frac{p(x)}{q(x)} = \frac{\frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}}{\frac{1}{\sqrt{2\pi}} e^{-(x - 3)^2 / 2}} = \exp\left( -\frac{x^2}{2} + \frac{x^2 - 6x + 9}{2} \right) = \mathbf{\exp(-3x + 4.5)}$$

We draw $N = 4$ toy samples from proposal $q(x)$:
$$x_1 = 3.0, \quad x_2 = 3.5, \quad x_3 = 2.4, \quad x_4 = 4.0$$

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $x_i$ | Sample drawn from proposal $q(x)$ | Generated latent vector / trajectory | Float scalar | $3.0, 3.5, 2.4, 4.0$ |
| $h(x_i)$ | Target integrand $x_i^2 \cdot \mathbf{1}_{\{x_i \ge 3\}}$ | Reward function / loss function | Float scalar | $9.0, 12.25, 0.0, 16.0$ |
| $p(x_i)$ | Nominal target probability density | Target policy $\pi_\theta$ probability | Float scalar | $\mathcal{N}(x; 0, 1)$ |
| $q(x_i)$ | Sampling proposal density | Behavior policy $\pi_{\text{old}}$ probability | Float scalar | $\mathcal{N}(x; 3, 1)$ |
| $w(x_i)$ | Likelihood ratio $p(x_i) / q(x_i)$ | Importance weight $r_t(\theta)$ | Float scalar | $\exp(-3x_i + 4.5)$ |
| $h(x_i) w(x_i)$| Weighted evaluation term | Policy gradient surrogate term | Float scalar | Cell-by-cell product |
| $\hat{I}_{\text{IS}}$ | Final Importance Sampling estimate | Off-policy expected value | Float scalar | Average of weighted terms |
| $\text{ESS}$ | Effective Sample Size | Degeneracy metric | Float scalar $\in [1, 4]$ | Hand-computed ratio |

---

### 3. Step-by-Step Manual Arithmetic

#### Sample 1 ($x_1 = 3.0$):
- Condition: $x_1 \ge 3.0 \implies h(x_1) = 3.0^2 = \mathbf{9.0000}$
- Weight exponent: $-3(3.0) + 4.5 = -9.0 + 4.5 = -4.5$
- Weight: $w(x_1) = e^{-4.5} \approx \mathbf{0.011109}$
- Weighted product: $h(x_1) w(x_1) = 9.0000 \times 0.011109 = \mathbf{0.099981}$

#### Sample 2 ($x_2 = 3.5$):
- Condition: $x_2 \ge 3.0 \implies h(x_2) = 3.5^2 = \mathbf{12.2500}$
- Weight exponent: $-3(3.5) + 4.5 = -10.5 + 4.5 = -6.0$
- Weight: $w(x_2) = e^{-6.0} \approx \mathbf{0.002479}$
- Weighted product: $h(x_2) w(x_2) = 12.2500 \times 0.002479 = \mathbf{0.030368}$

#### Sample 3 ($x_3 = 2.4$):
- Condition: $x_3 < 3.0 \implies \mathbf{h(x_3) = 0.0000}$ (Below threshold!)
- Weight exponent: $-3(2.4) + 4.5 = -7.2 + 4.5 = -2.7$
- Weight: $w(x_3) = e^{-2.7} \approx \mathbf{0.067206}$
- Weighted product: $h(x_3) w(x_3) = 0.0000 \times 0.067206 = \mathbf{0.000000}$

#### Sample 4 ($x_4 = 4.0$):
- Condition: $x_4 \ge 3.0 \implies h(x_4) = 4.0^2 = \mathbf{16.0000}$
- Weight exponent: $-3(4.0) + 4.5 = -12.0 + 4.5 = -7.5$
- Weight: $w(x_4) = e^{-7.5} \approx \mathbf{0.000553}$
- Weighted product: $h(x_4) w(x_4) = 16.0000 \times 0.000553 = \mathbf{0.008848}$

---

#### Step 4: Compute the Final Estimate $\hat{I}_{\text{IS}}$
$$\sum_{i=1}^4 h(x_i) w(x_i) = 0.099981 + 0.030368 + 0.000000 + 0.008848 = \mathbf{0.139197}$$
$$\mathbf{\hat{I}_{\text{IS}} = \frac{0.139197}{4} \approx 0.034799}$$

*(Comparison: Crude Monte Carlo with $N = 4$ would draw $X \sim \mathcal{N}(0, 1)$, where all 4 draws are overwhelmingly likely to be $< 3$, yielding $\hat{I}_{\text{Crude}} = \mathbf{0.0000}$! Importance Sampling captured the rare signal immediately!)*

---

#### Step 5: Compute Effective Sample Size (ESS)
$$\sum_{i=1}^4 w_i = 0.011109 + 0.002479 + 0.067206 + 0.000553 = \mathbf{0.081347}$$
$$\left( \sum_{i=1}^4 w_i \right)^2 = (0.081347)^2 = \mathbf{0.0066173}$$
$$\sum_{i=1}^4 w_i^2 = 0.011109^2 + 0.002479^2 + 0.067206^2 + 0.000553^2 = 0.0001234 + 0.0000061 + 0.0045166 + 0.0000003 = \mathbf{0.0046464}$$
$$\mathbf{\text{ESS} = \frac{0.0066173}{0.0046464} \approx 1.424} \quad (\text{out of } N = 4)$$

---

### 4. Visual Summary Grid

```
┌──────┬────────┬──────────┬─────────────┬─────────────┬──────────────┬───────────────┐
│ i    │ Draw x │ h(x)=x²  │ Exponent    │ Weight w(x) │ Product h·w  │ Crude MC Term │
├──────┼────────┼──────────┼─────────────┼─────────────┼──────────────┼───────────────┤
│ 1    │  3.00  │  9.0000  │ -3(3.0)+4.5 │  0.011109   │   0.099981   │     0.0       │
│ 2    │  3.50  │ 12.2500  │ -3(3.5)+4.5 │  0.002479   │   0.030368   │     0.0       │
│ 3    │  2.40  │  0.0000  │ -3(2.4)+4.5 │  0.067206   │   0.000000   │     0.0       │
│ 4    │  4.00  │ 16.0000  │ -3(4.0)+4.5 │  0.000553   │   0.008848   │     0.0       │
├──────┼────────┼──────────┼─────────────┼─────────────┼──────────────┼───────────────┤
│ SUM  │   —    │    —     │     —       │  0.081347   │ Σ = 0.139197 │  Σ = 0.0000   │
├──────┴────────┴──────────┴─────────────┴─────────────┼──────────────┼───────────────┤
│ Final Estimate I_hat (divide sum by N = 4)           │   0.034799   │    0.0000     │
│ Effective Sample Size ESS                            │   1.424 / 4  │     —         │
└──────────────────────────────────────────────────────┴──────────────┴───────────────┘
```

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Monte Carlo Estimation of $\pi$ via Uniform Sampling
Consider the unit square $[-1, 1]^2$ with area $A = 4$.
A circle of radius $r = 1$ centered at origin has area $\pi$.
- Indicator function: $h(x, y) = \mathbf{1}_{\{x^2 + y^2 \le 1\}}$.
- Expectation under Uniform$([-1, 1]^2)$:
  $$\mathbb{E}[h(X, Y)] = \frac{\text{Area of Circle}}{\text{Area of Square}} = \frac{\pi}{4}$$
- Estimator:
  $$\hat{\pi}_N = 4 \cdot \frac{N_{\text{inside}}}{N}$$
- Variance:
  $$\text{Var}(\hat{\pi}_N) = 16 \cdot \frac{(\pi/4)(1 - \pi/4)}{N} \approx \frac{16(0.785)(0.215)}{N} \approx \frac{2.70}{N}$$
To get 3 decimal places of precision ($\text{SE} \approx 0.001$), you require $N \approx \frac{2.70}{10^{-6}} = \mathbf{2{,}700{,}000 \text{ samples}}$!

---

### Illustration 2 (Boundary): Weight Collapse in High Dimensions (The IS Curse)
Let target be standard multivariate Gaussian $p(x) = \mathcal{N}(\mathbf{0}, I_d)$ and proposal be a slightly wider Gaussian $q(x) = \mathcal{N}(\mathbf{0}, \sigma^2 I_d)$ with $\sigma = 1.2$.
The importance weight variance is:
$$\mathbb{E}_q[w(X)^2] = \int \frac{p(x)^2}{q(x)} dx = \left( \frac{\sigma^2}{\sqrt{2\sigma^2 - 1}} \right)^d$$
For $\sigma = 1.2$:
$$\frac{1.2^2}{\sqrt{2(1.44) - 1}} = \frac{1.44}{\sqrt{1.88}} = \frac{1.44}{1.3711} \approx \mathbf{1.0502}$$
- In $d = 10$: $1.0502^{10} \approx 1.63$ (Manageable).
- In $d = 100$: $1.0502^{100} \approx 139$ (Significant weight variance).
- In $d = 500$: $1.0502^{500} \approx \mathbf{4.9 \times 10^{10}}$!
*Boundary Verdict:* In high dimensions, the variance of importance weights **explodes exponentially with $d$**. This is why naive importance sampling fails for latent spaces $d > 50$, necessitating Sequential Monte Carlo (SMC) or MCMC.

---

### Illustration 3 (Edge Case): Heavy-Tailed vs. Light-Tailed Proposal Disaster
Suppose target distribution is standard Cauchy $p(x) = \frac{1}{\pi(1 + x^2)}$:
1. **Case A (Proposal is Gaussian $\mathcal{N}(0, 1)$):**
   $$w(x) = \frac{p(x)}{q(x)} = \frac{1 / [\pi(1 + x^2)]}{\frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}} \propto \frac{e^{+x^2 / 2}}{1 + x^2} \xrightarrow{|x| \to \infty} \mathbf{+\infty}$$
   The weights diverge exponentially! $\text{Var}(w) = \infty$. The estimator will never converge.
2. **Case B (Proposal is Student's $t$ with 2 degrees of freedom):**
   Proposal tails decay as $\mathcal{O}(x^{-3})$, while Cauchy decays as $\mathcal{O}(x^{-2})$.
   Ratio $w(x) = \mathcal{O}(x) \implies$ still infinite variance!
   Proposal must have **strictly heavier tails** than the target.

---

## Part 7: Deep Learning Connection & Application

### 1. Off-Policy Policy Gradients & PPO (Proximal Policy Optimization)
In modern Reinforcement Learning from Human Feedback (RLHF) used to train LLMs (ChatGPT, Claude), policies are trained using **PPO (Schulman et al., 2017)**.
Evaluating the policy gradient on new parameters $\theta$ using old trajectories collected by policy $\theta_{\text{old}}$ requires Importance Sampling:
$$L^{\text{IS}}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$$
The probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the **Importance Weight**!

Because unconstrained importance weights can explode and destabilize policy updates, PPO introduces **Clipped Importance Sampling**:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \,\, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
Clipping the importance ratio to $[1 - \epsilon, 1 + \epsilon]$ (typically $[0.8, 1.2]$) mathematically bounds the importance weight, preventing variance explosion!

---

### 2. Denoising Score Matching & Diffusion Models
Diffusion models (DDPM, Stable Diffusion) estimate scores $\nabla_x \log p_t(x)$ via Monte Carlo sampling of Gaussian noise perturbations $\epsilon \sim \mathcal{N}(0, I)$. Importance sampling is used during training to allocate more diffusion timesteps $t$ to noisy regions with high reconstruction loss variance!

---

## Part 8: Code Implementation & Verification

The companion Python module [06_monte_carlo_and_importance_sampling.py](./code/06_monte_carlo_and_importance_sampling.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact manual arithmetic of $\hat{I}_{\text{IS}} \approx 0.03480$ and $\text{ESS} \approx 1.424$.
2. **Rare-Event Tail Expectation Monte Carlo Test:** Runs 100,000 trials comparing Crude MC against Importance Sampling, demonstrating a $50\times$ variance reduction under Importance Sampling.
3. **Monte Carlo Estimation of $\pi$:** Simulates $N = 1{,}000{,}000$ uniform 2D points to confirm $\mathcal{O}(1/\sqrt{N})$ error decay toward $\pi$.
4. **PPO Clipped Importance Sampling Surrogate:** Implements the PPO clipped loss objective in PyTorch, showing how clipping stabilizes gradient variance when policies diverge.
