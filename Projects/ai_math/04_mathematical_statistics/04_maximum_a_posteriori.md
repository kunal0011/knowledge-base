# Chapter 4.4: Maximum A Posteriori (MAP) & Regularization Links (L1/L2)

---

## Part 1: Intuition & 101 Motivation

In Chapter 4.3, we studied **Maximum Likelihood Estimation (MLE)**, which treats model parameters $\theta$ as unknown fixed constants and maximizes the probability of the training data:
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(\mathcal{D} \mid \theta)$$

However, pure MLE has a fatal flaw in deep learning: **it tends to overfit**.
If a neural network has 100 million parameters and the training data is limited or noisy, MLE will enthusiastically drive weights to massive values ($10^4, 10^6$) to memorize random noise and fit every single outlier.

How do we prevent this?
We introduce **Prior Knowledge** before seeing any training data:
> *"All else being equal, smaller weights with low complexity are far more plausible than wildly exploding weights."*

This is the **Bayesian approach**. Instead of treating $\theta$ as a fixed constant, we treat $\theta$ as a **random variable** endowed with a **Prior Distribution** $P(\theta)$.

Using Bayes' Theorem:
$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})}$$
$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

**Maximum A Posteriori (MAP)** estimation finds the most probable parameter setting under the posterior distribution:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \mathcal{D})$$

This single equation establishes the grand unification of deep learning regularization:
- An isotropic **Gaussian Prior** on weights is mathematically identical to **$L_2$ Regularization (Ridge Regression / Weight Decay)**!
- An independent **Laplace Prior** on weights is mathematically identical to **$L_1$ Regularization (Lasso / Sparsity inducing)**!

---

## Part 2: Rigorous Mathematical Formulation

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be an $i.i.d.$ dataset, and let $\mathbf{w} \in \mathbb{R}^D$ be the parameter vector.

### 1. The MAP Optimization Objective
The posterior probability density of parameters given the observed data is:
$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}$$
where $p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w}) d\mathbf{w}$ is the marginal likelihood (constant with respect to $\mathbf{w}$).

Take the natural logarithm:
$$\ln p(\mathbf{w} \mid \mathcal{D}) = \ln p(\mathcal{D} \mid \mathbf{w}) + \ln p(\mathbf{w}) - \ln p(\mathcal{D})$$
Maximizing the posterior is equivalent to minimizing the Negative Log-Posterior:
$$\mathbf{\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_{\mathbf{w}} \left[ \underbrace{-\sum_{i=1}^N \ln p(y_i \mid x_i; \mathbf{w})}_{\text{Data Loss (NLL)}} \,\, \underbrace{- \ln p(\mathbf{w})}_{\text{Regularizer } \Omega(\mathbf{w})} \right]}$$

---

### 2. Theorem 1: Gaussian Prior $\iff L_2$ Regularization (Weight Decay)
**Statement:**
Under Gaussian observation noise $y_i = f_{\mathbf{w}}(x_i) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$, assuming a zero-mean isotropic Gaussian prior on weights:
$$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \sigma_0^2 I_D)$$
the MAP estimator is mathematically identical to **$L_2$ Regularization (Ridge Regression)**.

**Rigorous Proof:**
The prior probability density is:
$$p(\mathbf{w}) = \frac{1}{(2\pi \sigma_0^2)^{D/2}} \exp\left( -\frac{\|\mathbf{w}\|_2^2}{2\sigma_0^2} \right) = \frac{1}{(2\pi \sigma_0^2)^{D/2}} \exp\left( -\frac{\sum_{j=1}^D w_j^2}{2\sigma_0^2} \right)$$
Take the negative logarithm:
$$-\ln p(\mathbf{w}) = \frac{D}{2}\ln(2\pi\sigma_0^2) + \frac{1}{2\sigma_0^2} \|\mathbf{w}\|_2^2$$
Combine with the Gaussian negative log-likelihood:
$$\mathcal{L}_{\text{MAP}}(\mathbf{w}) = \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f_{\mathbf{w}}(x_i))^2 + \frac{1}{2\sigma_0^2} \|\mathbf{w}\|_2^2 + \text{constant}$$
Multiply the entire objective by $\frac{\sigma^2}{N}$ (which does not change the argmin):
$$\mathcal{L}_{\text{MAP}}(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (y_i - f_{\mathbf{w}}(x_i))^2 + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
where the regularization penalty coefficient is:
$$\mathbf{\lambda = \frac{\sigma^2}{N \sigma_0^2} = \frac{\text{Observation Noise Variance}}{N \times \text{Prior Variance}}} \quad \blacksquare$$

#### Fundamental Insights from the Proof:
1. **The Role of Prior Variance $\sigma_0^2$:**
   - If $\sigma_0^2 \to \infty$ (completely uninformative, flat prior), then $\lambda \to 0$, and MAP simplifies to pure unregularized **MLE**.
   - If $\sigma_0^2 \to 0$ (very strong prior belief that weights must be zero), then $\lambda \to \infty$, forcing $\mathbf{w} \to \mathbf{0}$.
2. **The Asymptotic Limit ($N \to \infty$):**
   - As dataset size $N \to \infty$, the penalty $\lambda = \frac{\sigma^2}{N \sigma_0^2} \to 0$.
   - **The data eventually overwhelms any fixed prior**, and MAP converges to the MLE (the Bernstein-von Mises Theorem)!

---

### 3. Theorem 2: Laplace Prior $\iff L_1$ Regularization (Lasso / Sparsity)
**Statement:**
Under Gaussian observation noise, assuming an independent zero-mean Laplace prior on weights:
$$w_j \overset{i.i.d.}{\sim} \text{Laplace}(0, b) \implies p(\mathbf{w}) = \prod_{j=1}^D \frac{1}{2b} \exp\left( -\frac{|w_j|}{b} \right)$$
the MAP estimator is mathematically identical to **$L_1$ Regularization (Lasso)**.

**Rigorous Proof:**
The joint prior density is:
$$p(\mathbf{w}) = \frac{1}{(2b)^D} \exp\left( -\frac{\sum_{j=1}^D |w_j|}{b} \right) = \frac{1}{(2b)^D} \exp\left( -\frac{\|\mathbf{w}\|_1}{b} \right)$$
Take the negative logarithm:
$$-\ln p(\mathbf{w}) = D \ln(2b) + \frac{1}{b} \|\mathbf{w}\|_1$$
Combine with Gaussian NLL and scale by $\frac{\sigma^2}{N}$:
$$\mathbf{\mathcal{L}_{\text{MAP}}(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (y_i - f_{\mathbf{w}}(x_i))^2 + \lambda \|\mathbf{w}\|_1} \quad \text{where } \mathbf{\lambda = \frac{\sigma^2}{N b}} \quad \blacksquare$$

---

### 4. Closed-Form Analytical MAP Solution for Linear Regression
For a linear model $f_{\mathbf{w}}(\mathbf{x}) = X \mathbf{w}$ with Gaussian likelihood and Gaussian prior:
$$\mathcal{L}(\mathbf{w}) = \frac{1}{2\sigma^2} \|\mathbf{y} - X\mathbf{w}\|_2^2 + \frac{1}{2\sigma_0^2} \|\mathbf{w}\|_2^2$$
Compute the gradient w.r.t. $\mathbf{w}$ and set to zero:
$$\nabla_{\mathbf{w}} \mathcal{L} = -\frac{1}{\sigma^2} X^T (\mathbf{y} - X\mathbf{w}) + \frac{1}{\sigma_0^2} \mathbf{w} = \mathbf{0}$$
Multiply by $\sigma^2$:
$$-X^T \mathbf{y} + X^T X \mathbf{w} + \frac{\sigma^2}{\sigma_0^2} \mathbf{w} = \mathbf{0} \implies \left( X^T X + \frac{\sigma^2}{\sigma_0^2} I_D \right) \mathbf{w} = X^T \mathbf{y}$$
$$\mathbf{\hat{\mathbf{w}}_{\text{MAP}} = \left( X^T X + \lambda I_D \right)^{-1} X^T \mathbf{y} \quad \text{where } \lambda = \frac{\sigma^2}{\sigma_0^2}}$$

#### Numerical Stability Guarantee:
Even if $X^T X$ is rank-deficient (e.g. collinear features or $D > N$), the regularized matrix $(X^T X + \lambda I)$ is **strictly symmetric positive definite** and always invertible because all its eigenvalues are strictly shifted by $+\lambda > 0$!

---

## Part 3: Geometric & Algebraic Interpretation

### Why Does $L_1$ Produce Exact Zeros (Sparsity) while $L_2$ Does Not?

Consider the constrained optimization formulation:
$$\min_{\mathbf{w}} \mathcal{L}_{\text{data}}(\mathbf{w}) \quad \text{subject to } \Omega(\mathbf{w}) \le C$$

```
        L_2 Regularization (Circle)                  L_1 Regularization (Diamond)
                  w_2                                          w_2
                   ▲                                            ▲
                   │                                            │      /\
             .─────┼─────.                                      │     /  \
           .'      │      `.                                    │    /    \
          /   ●────┼─────── \ ◄── Loss Contours                │   /  ●   \ ◄── Tangency at CORNER!
         │    │    │        │                                   │  /  / \   \     (w_1 = 0!)
         ───────┼────┼────────► w_1                             ┼─/──/───\───\─► w_1
         │    │    │        │                                   │ \  \   /   /
          \   └───┼────────/                                    │  \  \ /   /
           `.      │      .'                                    │   \  ●   /
             `─────┼─────'                                      │    \    /
                   │                                            │     \  /
                   │                                            │      \/
```

1. **$L_2$ Geometry (Smooth Hypersphere):**
   The constraint boundary $\|\mathbf{w}\|_2^2 \le C$ is smooth everywhere with continuous normals. The expanding loss ellipsoid typically touches the sphere at a point where **both coordinates are non-zero**. Weights are shrunk smoothly toward zero, but almost never reach exact zero.
2. **$L_1$ Geometry (Hyper-Octahedron with Sharp Corners):**
   The constraint boundary $\|\mathbf{w}\|_1 \le C$ has sharp corners (vertices) along the coordinate axes where one or more coordinates are **identically zero**. Because the corners stick out furthest, the expanding loss ellipsoid has an overwhelmingly high probability of making first contact at a corner vertex!
   Hence, $L_1$ regularization performs **automatic feature selection** and sets entire weights to **exact zeros**.

---

## Part 4: Real-World Analogy

### The Forensic Investigator (MLE vs. MAP)
Imagine a forensic investigator examining a crime scene:
- **The Evidence (Likelihood):** Muddy size-11 boot prints are found on the carpet.
- **The Pure MLE Approach:**
  The investigator concludes the suspect is a 7-foot-tall circus acrobat who wore size-11 boots, because that hypothesis fits every single odd mark in the dust with 100% precision. MLE overfits the bizarre quirks of the scene.
- **The MAP / Bayesian Approach:**
  The investigator applies Occam's razor (the Prior): the general public contains millions of average citizens with size-11 shoes, while circus acrobats are extraordinarily rare ($P(\text{acrobat}) \approx 10^{-6}$).
  The posterior heavily favors the ordinary resident unless overwhelmingly extraordinary evidence proves otherwise.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us solve a complete **Linear Regression MAP problem** by hand with concrete numbers, evaluating how Gaussian and Laplace priors alter the optimal weight and comparing cell-by-cell total losses.

### 1. Problem Setup & Toy Dataset
Model through the origin: $\hat{y} = w x$.
Observation noise: Gaussian $\epsilon \sim \mathcal{N}(0, \sigma^2 = 1.0)$.
$N = 2$ data points:
- Point 1: $(x_1, y_1) = (1.0, 3.0)$
- Point 2: $(x_2, y_2) = (2.0, 5.0)$

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Value in Walkthrough |
| :--- | :--- | :--- | :--- | :--- |
| $N$ | Sample size | Batch size | Integer scalar | $2$ |
| $(x_i, y_i)$ | Data observations | Training pairs | Float scalars | $(1.0, 3.0), (2.0, 5.0)$ |
| $\sigma^2$ | Observation noise variance | Data loss scale | Float scalar | $1.00$ |
| $\sigma_0^2$ | Gaussian prior variance | Weight decay strength denominator | Float scalar | $0.50$ |
| $b$ | Laplace prior scale | $L_1$ penalty strength denominator | Float scalar | $0.50$ |
| $w_{\text{MLE}}$ | Unregularized weight | Standard OLS weight | Float scalar | $2.6000$ |
| $w_{\text{MAP-Gauss}}$ | Gaussian MAP weight | Weight decay ($L_2$) optimal weight | Float scalar | $1.8571$ |
| $w_{\text{MAP-Laplace}}$ | Laplace MAP weight | Lasso ($L_1$) optimal weight | Float scalar | $2.2000$ |

---

### 3. Step-by-Step Manual Arithmetic

#### Step 1: Compute Data Sufficient Statistics & MLE
$$\sum_{i=1}^2 x_i^2 = 1.0^2 + 2.0^2 = 1.0 + 4.0 = \mathbf{5.0000}$$
$$\sum_{i=1}^2 x_i y_i = (1.0)(3.0) + (2.0)(5.0) = 3.0 + 10.0 = \mathbf{13.0000}$$
Unregularized MLE solution:
$$w_{\text{MLE}} = \frac{\sum x_i y_i}{\sum x_i^2} = \frac{13.0000}{5.0000} = \mathbf{2.6000}$$
Data NLL Loss as a function of $w$ (ignoring constant):
$$\mathcal{L}_{\text{data}}(w) = \frac{1}{2\sigma^2} \sum_{i=1}^2 (y_i - w x_i)^2 = \frac{1}{2} \left[ (3 - w)^2 + (5 - 2w)^2 \right] = \frac{1}{2} [ 5 w^2 - 26 w + 34 ]$$

---

#### Step 2: Gaussian Prior MAP Analytical Derivation ($L_2$)
Let prior be $\mathcal{N}(0, \sigma_0^2 = 0.50)$.
The prior penalty is:
$$\Omega_{\text{Gauss}}(w) = \frac{w^2}{2\sigma_0^2} = \frac{w^2}{2(0.50)} = \mathbf{w^2}$$
Total MAP Loss:
$$\mathcal{L}_{\text{MAP-Gauss}}(w) = \frac{1}{2}(5 w^2 - 26 w + 34) + w^2 = 3.5 w^2 - 13 w + 17$$
Take derivative and set to zero:
$$\frac{d \mathcal{L}}{dw} = 7.0 w - 13.0 = 0 \implies \mathbf{w_{\text{MAP-Gauss}} = \frac{13.0}{7.0} \approx 1.857143}$$
Notice: The weight is shrunk inward from $2.6000 \to 1.8571$!

---

#### Step 3: Laplace Prior MAP Analytical Derivation ($L_1$)
Let prior be $\text{Laplace}(0, b = 0.50)$.
The prior penalty is:
$$\Omega_{\text{Laplace}}(w) = \frac{|w|}{b} = \frac{|w|}{0.50} = \mathbf{2|w|}$$
For $w > 0$, $|w| = w$:
$$\mathcal{L}_{\text{MAP-Laplace}}(w) = \frac{1}{2}(5 w^2 - 26 w + 34) + 2 w = 2.5 w^2 - 11 w + 17$$
Take derivative and set to zero:
$$\frac{d \mathcal{L}}{dw} = 5.0 w - 11.0 = 0 \implies \mathbf{w_{\text{MAP-Laplace}} = \frac{11.0}{5.0} = 2.2000}$$
Notice: The weight is shrunk inward from $2.6000 \to 2.2000$ (by exactly $\lambda / \sum x^2 = 2/5 = 0.4$)!

---

#### Step 4: Step-by-Step Grid Evaluation for Candidate Weights

Let us evaluate the losses at candidate weights:
- $w = 0.0$
- $w = 1.8571$ (Gaussian MAP minimum)
- $w = 2.2000$ (Laplace MAP minimum)
- $w = 2.6000$ (MLE minimum)

```
Candidate w │ Data Loss L_data │ Gauss Prior (w²) │ Total Gauss MAP │ Laplace Prior (2|w|)│ Total Lapl MAP
────────────┼──────────────────┼──────────────────┼─────────────────┼─────────────────────┼───────────────
w = 0.0000  │ 17.0000          │  0.0000          │ 17.0000         │ 0.0000              │ 17.0000
w = 1.8571  │  2.9286          │  3.4490          │  6.3776 ◄── MIN │ 3.7143              │  6.6429
w = 2.2000  │  1.0000          │  4.8400          │  5.8400         │ 4.4000              │  5.4000 ◄──MIN
w = 2.6000  │  0.1000 ◄── MIN  │  6.7600          │  6.8600         │ 5.2000              │  5.3000
```
*(Note: At w=2.6, data loss is minimal at 0.1000, but heavy prior penalties make total MAP higher than at the regularized optima!)*

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Full Bayesian Linear Regression Conjugacy
When prior is Gaussian $\mathbf{w} \sim \mathcal{N}(\mathbf{m}_0, S_0)$ and likelihood is Gaussian $\mathbf{y} \mid X, \mathbf{w} \sim \mathcal{N}(X\mathbf{w}, \sigma^2 I_N)$, the posterior distribution is **also analytically Gaussian** (Conjugate):
$$p(\mathbf{w} \mid \mathcal{D}) = \mathcal{N}(\mathbf{m}_N, S_N)$$
where:
$$S_N^{-1} = S_0^{-1} + \frac{1}{\sigma^2} X^T X$$
$$\mathbf{m}_N = S_N \left( S_0^{-1} \mathbf{m}_0 + \frac{1}{\sigma^2} X^T \mathbf{y} \right)$$
Because the Gaussian is unimodal and symmetric, the **posterior mode ($\hat{\mathbf{w}}_{\text{MAP}}$) is identically equal to the posterior mean ($\mathbf{m}_N$)**!

---

### Illustration 2 (Boundary): Soft Thresholding in 1D $L_1$ MAP
Consider a 1D model with scalar $w$:
$$\min_w \frac{1}{2}(w - w_{\text{MLE}})^2 + \lambda |w|$$
Using subgradient calculus, the exact analytical solution is the **Soft-Thresholding Operator** $\mathcal{S}_\lambda(w_{\text{MLE}})$:
$$\mathbf{\hat{w}_{\text{MAP}} = \text{sign}(w_{\text{MLE}}) \max(0, \, |w_{\text{MLE}}| - \lambda) = \begin{cases} w_{\text{MLE}} - \lambda & \text{if } w_{\text{MLE}} > \lambda \\ 0 & \text{if } |w_{\text{MLE}}| \le \lambda \\ w_{\text{MLE}} + \lambda & \text{if } w_{\text{MLE}} < -\lambda \end{cases}}$$
*Boundary Verdict:* If the unregularized signal $|w_{\text{MLE}}|$ is smaller than the threshold $\lambda$, the MAP estimator snaps the parameter to **exactly zero**!

---

### Illustration 3 (Edge Case): Why $L_2$ Regularization Fails in Adam (The AdamW Discovery)
In standard SGD, adding an $L_2$ penalty to the loss function is mathematically equivalent to weight decay:
$$\nabla_\theta \left( \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2 \right) = \nabla_\theta \mathcal{L}(\theta) + \lambda \theta \implies \theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) - \eta \lambda \theta_t$$
However, in **Adam**, the update divides the gradient by $\sqrt{v_t}$:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \left( \nabla_\theta \mathcal{L}(\theta_t) + \lambda \theta_t \right)$$
- If a parameter has historically large gradients, $v_t$ is large $\implies$ the effective regularization penalty $\frac{\lambda \theta_t}{\sqrt{v_t}}$ is **artificially crushed**!
- If a parameter has historically tiny gradients, the regularization penalty is **artificially amplified**!
**AdamW (Loshchilov & Hutter, 2019):** Decouples weight decay from the adaptive gradient moment:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \nabla \mathcal{L}(\theta_t) - \mathbf{\eta \lambda \theta_t}$$
AdamW restores the true Gaussian MAP prior behavior, which is why virtually every modern Transformer (LLaMA, GPT-4, Mistral) is trained with AdamW rather than vanilla Adam!

---

## Part 7: Deep Learning Connection & Application

### 1. Bayesian Deep Learning & Dropout as Variational Inference
Training a full Bayesian neural network requires integrating over the entire intractable posterior $p(\mathbf{w} \mid \mathcal{D})$.
Gal & Ghahramani (2016) proved a landmark result:
**Applying Dropout at both training and test time (Monte Carlo Dropout) is mathematically equivalent to approximate Variational Inference over a deep Gaussian Process!**
By taking $M$ stochastic forward passes with dropout enabled during test inference:
$$\mu(x) = \frac{1}{M}\sum_{m=1}^M f_{\hat{\mathbf{w}}_m}(x), \quad \sigma^2(x) = \frac{1}{M}\sum_{m=1}^M (f_{\hat{\mathbf{w}}_m}(x) - \mu(x))^2$$
we obtain principled epistemic uncertainty estimates without training separate models!

---

## Part 8: Code Implementation & Verification

The companion Python module [04_maximum_a_posteriori.py](./code/04_maximum_a_posteriori.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact calculations of $w_{\text{MLE}} = 2.6000$, $w_{\text{MAP-Gauss}} = 1.8571$, and $w_{\text{MAP-Laplace}} = 2.2000$.
2. **Full Gaussian Conjugacy Verification:** Simulates Bayesian linear regression updating Gaussian posterior mean and covariance against analytical formulas.
3. **Soft-Thresholding Operator & Exact Sparsity:** Evaluates $L_1$ soft-thresholding across a grid of parameter values, verifying exact zero clamping when $|w| \le \lambda$.
4. **Adam vs. AdamW Decoupled Weight Decay:** Demonstrates empirically that AdamW preserves proper Gaussian shrinkage while $L_2$-penalized Adam distorts regularizer strength across varying gradient magnitudes.
