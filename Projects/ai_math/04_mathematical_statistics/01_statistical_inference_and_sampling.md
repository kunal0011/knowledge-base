# Chapter 4.1: Statistical Inference Foundations & Sampling

---

## Part 1: Intuition & 101 Motivation

What is the fundamental divide between **Probability Theory** and **Mathematical Statistics**?

- In **Probability Theory**, the world is top-down: we assume the true data-generating distribution (parameters $\theta^*$) is completely known, and we deduce what observed samples $\mathbf{X}$ should look like.
  $$\text{Probability}: \quad \text{Parameters } \theta^* \longrightarrow \text{Observed Data } \mathbf{X}$$
- In **Statistics and Machine Learning**, the direction is reversed: nature provides us only with an observed training set $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$, and we must infer the hidden true mechanism $\theta^*$ that created it!
  $$\text{Statistical Inference}: \quad \text{Observed Data } \mathcal{D} \longrightarrow \text{Estimated Model } \hat{\theta}$$

```
                ┌───────────────────────────────┐
                │   True Population / Nature    │
                │        Distribution P_θ       │
                └───────────────┬───────────────┘
                                │
               Probability      │     Statistical
               (Deduction)      │     Inference
                                │     (Induction)
                                ▼
                ┌───────────────────────────────┐
                │        Observed Sample        │
                │   D = {x_1, x_2, ..., x_N}    │
                └───────────────────────────────┘
```

Every deep learning model is a statistical estimator. When you train a ResNet or a 70B LLM on web tokens, you are performing statistical inference: using a finite sample to infer the underlying structure of visual reality or human language.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Population, Sample & The $i.i.d.$ Assumption
Let $(\Omega, \mathcal{F}, P_\theta)$ be a probability space parametrized by an unknown parameter $\theta \in \Theta$.
- The **Population** is the true, unobserved distribution $P_\theta$.
- A **Random Sample** of size $N$ is a collection of $N$ random variables:
  $$\mathbf{X} = (X_1, X_2, \dots, X_N)$$
- **The $i.i.d.$ Assumption:** In classical statistics and standard supervised deep learning, samples are assumed to be **Independent and Identically Distributed**:
  $$P(X_1, \dots, X_N) = \prod_{i=1}^N P(X_i \mid \theta)$$
- An **Observation / Realization** is the concrete numerical vector $\mathbf{x} = (x_1, \dots, x_N) \in \mathbb{R}^N$ recorded in your dataset.

---

### 2. Statistics, Estimators & Estimates
- A **Statistic** $T(\mathbf{X})$ is any measurable function of the sample that **does not depend on the unknown parameter $\theta$**:
  $$T: \mathbb{R}^N \to \mathbb{R}^k$$
- An **Estimator** $\hat{\theta} = T(\mathbf{X})$ is a statistic used to approximate an unknown population parameter $\theta$. Because $\mathbf{X}$ is random, **an estimator is itself a random variable** with its own probability distribution!
- An **Estimate** $\hat{\theta}(\mathbf{x})$ is the concrete numerical evaluation of the estimator on realized data $\mathbf{x}$.

---

### 3. The Empirical Distribution Function (EDF)
Given an $i.i.d.$ sample $X_1, \dots, X_N \sim F(x)$, the **Empirical Distribution Function (EDF)** $\hat{F}_N(x)$ is the proportion of sample points less than or equal to $x$:
$$\hat{F}_N(x) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\{X_i \le x\}}$$
where $\mathbf{1}_{\{\cdot\}}$ is the indicator function.

#### Statistical Properties of $\hat{F}_N(x)$ at any fixed $x$:
1. $N \hat{F}_N(x) \sim \text{Binomial}(N, F(x))$.
2. $\mathbb{E}[\hat{F}_N(x)] = F(x)$ (unbiased).
3. $\text{Var}(\hat{F}_N(x)) = \frac{F(x)(1 - F(x))}{N} \to 0$ as $N \to \infty$.

#### The Fundamental Theorem of Statistics (Glivenko-Cantelli Theorem):
The empirical distribution converges uniformly to the true distribution almost surely:
$$\| \hat{F}_N - F \|_\infty = \sup_{x \in \mathbb{R}} |\hat{F}_N(x) - F(x)| \xrightarrow{\text{a.s.}} 0 \quad \text{as } N \to \infty$$
*Machine Learning Consequence:* This theorem is the rigorous mathematical justification for **Empirical Risk Minimization (ERM)**: minimizing loss on a finite training dataset converges to minimizing expected loss on the entire population as dataset size $N \to \infty$.

---

### 4. Sample Mean & Sample Variance
Given $X_1, \dots, X_N$ with population mean $\mu = \mathbb{E}[X_i]$ and finite variance $\sigma^2 = \text{Var}(X_i)$:

#### Sample Mean:
$$\bar{X} = \frac{1}{N} \sum_{i=1}^N X_i$$
- $\mathbb{E}[\bar{X}] = \mu$ (unbiased)
- $\text{Var}(\bar{X}) = \frac{\sigma^2}{N}$
- **Standard Error of the Mean (SE):**
  $$\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{N}}$$

#### Sample Variance (Bessel's Correction):
If we used the naive plug-in estimator $S_n^2 = \frac{1}{N}\sum_{i=1}^N (X_i - \bar{X})^2$:
$$\mathbb{E}[S_n^2] = \frac{N - 1}{N} \sigma^2 \ne \sigma^2 \quad (\text{strictly underestimated!})$$

To obtain an **unbiased estimator**, we apply **Bessel's Correction** (dividing by degrees of freedom $N - 1$):
$$S^2 = S_{N-1}^2 = \frac{1}{N - 1} \sum_{i=1}^N (X_i - \bar{X})^2$$
$$\mathbb{E}[S^2] = \sigma^2 \quad (\text{unbiased})$$

#### Deep Derivation 4.1.1: Algebraic Proof of Bessel's Correction $\mathbb{E}[S^2] = \sigma^2$
Let $X_1, \dots, X_N$ be $i.i.d.$ random variables with mean $\mu$ and variance $\sigma^2$.
We compute the expected value of the sum of squared deviations $\sum_{i=1}^N (X_i - \bar{X})^2$:

1. **Center around population mean $\mu$:**
   $$X_i - \bar{X} = (X_i - \mu) - (\bar{X} - \mu)$$
2. **Square and expand:**
   $$(X_i - \bar{X})^2 = (X_i - \mu)^2 - 2(X_i - \mu)(\bar{X} - \mu) + (\bar{X} - \mu)^2$$
3. **Sum over all $N$ data points:**
   $$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^N (X_i - \mu) + \sum_{i=1}^N (\bar{X} - \mu)^2$$
   Since $\sum_{i=1}^N (X_i - \mu) = N(\bar{X} - \mu)$:
   $$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - 2N(\bar{X} - \mu)^2 + N(\bar{X} - \mu)^2 = \sum_{i=1}^N (X_i - \mu)^2 - N(\bar{X} - \mu)^2$$
4. **Take mathematical expectations:**
   $$\mathbb{E}\left[ \sum_{i=1}^N (X_i - \bar{X})^2 \right] = \sum_{i=1}^N \mathbb{E}[(X_i - \mu)^2] - N \mathbb{E}[(\bar{X} - \mu)^2] = \sum_{i=1}^N \text{Var}(X_i) - N \text{Var}(\bar{X})$$
   Since $\text{Var}(X_i) = \sigma^2$ and $\text{Var}(\bar{X}) = \frac{\sigma^2}{N}$:
   $$\mathbb{E}\left[ \sum_{i=1}^N (X_i - \bar{X})^2 \right] = N \sigma^2 - N \left( \frac{\sigma^2}{N} \right) = N \sigma^2 - \sigma^2 = (N - 1) \sigma^2$$
5. **Divide by $N - 1$:**
   $$\mathbf{\mathbb{E}[S^2] = \frac{1}{N - 1} \mathbb{E}\left[ \sum_{i=1}^N (X_i - \bar{X})^2 \right] = \frac{(N - 1) \sigma^2}{N - 1} = \sigma^2} \quad \blacksquare$$

---

### 5. Cochran's Theorem & Exact Sampling Distributions for Normal Populations
When samples are drawn from a Gaussian population $X_i \overset{i.i.d.}{\sim} \mathcal{N}(\mu, \sigma^2)$, Cochran's Theorem establishes three profound structural facts:

1. **Distribution of Sample Mean:**
   $$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{N}\right)$$
2. **Distribution of Sample Variance:**
   $$\frac{(N - 1)S^2}{\sigma^2} = \sum_{i=1}^N \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2(N - 1)$$
3. **Statistical Independence:**
   $$\mathbf{\bar{X} \quad \text{and} \quad S^2 \quad \text{are statistically INDEPENDENT!}}$$
   *(This independence is a unique characterizing property of Gaussian distributions!)*

#### Deep Derivation 4.1.2: Orthogonal Projection Proof of Cochran's Theorem
Let $\mathbf{X} = [X_1, \dots, X_N]^T \sim \mathcal{N}(\mu \mathbf{1}, \sigma^2 I_N)$.
Define the standardized vector $\mathbf{Z} = \frac{\mathbf{X} - \mu \mathbf{1}}{\sigma} \sim \mathcal{N}(0, I_N)$.

1. **Idempotent Projection Operators:**
   Let $P = \frac{1}{N}\mathbf{1}\mathbf{1}^T \in \mathbb{R}^{N \times N}$ be the projection matrix onto the subspace spanned by $\mathbf{1}$.
   Let $M = I_N - P = I_N - \frac{1}{N}\mathbf{1}\mathbf{1}^T$ be the projection onto the orthogonal complement $\mathbf{1}^\perp$.
   - Symmetric: $M^T = M$.
   - Idempotent: $M^2 = (I - P)(I - P) = I - 2P + P^2 = I - P = M$.
   - Trace / Rank: $\text{Tr}(M) = \text{Tr}(I_N) - \frac{1}{N}\text{Tr}(\mathbf{1}\mathbf{1}^T) = N - \frac{1}{N}N = N - 1$.

2. **Expressing Sample Variance as a Quadratic Form:**
   Note that $M \mathbf{X} = \mathbf{X} - \bar{X} \mathbf{1}$.
   $$\frac{(N - 1)S^2}{\sigma^2} = \frac{1}{\sigma^2} \| \mathbf{X} - \bar{X}\mathbf{1} \|_2^2 = \frac{1}{\sigma^2} (M \mathbf{X})^T (M \mathbf{X}) = \frac{1}{\sigma^2} \mathbf{X}^T M^2 \mathbf{X} = \mathbf{Z}^T M \mathbf{Z}$$
   Because $M$ is a symmetric idempotent matrix of rank $N - 1$, by the Spectral Theorem its eigenvalues are 1 (with multiplicity $N - 1$) and 0 (with multiplicity 1).
   Under an orthogonal change of basis $\mathbf{Z} = U \mathbf{W}$ where $\mathbf{W} \sim \mathcal{N}(0, I_N)$:
   $$\mathbf{Z}^T M \mathbf{Z} = \mathbf{W}^T \text{diag}(1, \dots, 1, 0) \mathbf{W} = \sum_{j=1}^{N-1} W_j^2 \sim \mathbf{\chi^2(N - 1)} \quad \blacksquare$$

3. **Proof of Independence between $\bar{X}$ and $S^2$:**
   The sample mean is $\bar{X} = \frac{1}{N}\mathbf{1}^T \mathbf{X}$.
   The deviation vector is $\mathbf{e} = M \mathbf{X}$.
   Both $\bar{X}$ and $\mathbf{e}$ are linear transformations of the jointly Gaussian vector $\mathbf{X}$.
   Compute their covariance matrix:
   $$\text{Cov}(\bar{X}, \mathbf{e}) = \text{Cov}\left( \frac{1}{N}\mathbf{1}^T \mathbf{X}, M \mathbf{X} \right) = \frac{1}{N} \mathbf{1}^T \text{Var}(\mathbf{X}) M^T = \frac{\sigma^2}{N} \mathbf{1}^T (I_N - P) = \frac{\sigma^2}{N} \left( \mathbf{1}^T - \frac{1}{N}\mathbf{1}^T \mathbf{1}\mathbf{1}^T \right)$$
   Since $\mathbf{1}^T \mathbf{1} = N$:
   $$\text{Cov}(\bar{X}, \mathbf{e}) = \frac{\sigma^2}{N} (\mathbf{1}^T - \mathbf{1}^T) = \mathbf{0}^T$$
   For jointly Gaussian random variables, **zero covariance implies strict statistical independence**!
   Since $S^2 = \frac{1}{N-1}\|\mathbf{e}\|_2^2$ is a deterministic function of $\mathbf{e}$, it follows that $\bar{X}$ and $S^2$ are statistically independent! $\blacksquare$

---

### 6. Student's $t$-Distribution & Confidence Intervals
In practice, the true population variance $\sigma^2$ is unknown, so we must plug in the sample standard error $\hat{\text{SE}} = \frac{S}{\sqrt{N}}$.
The standardized ratio follows **Student's $t$-distribution** with $\nu = N - 1$ degrees of freedom:
$$T = \frac{\bar{X} - \mu}{S / \sqrt{N}} \sim t(N - 1)$$

#### Deep Derivation 4.1.3: Derivation of Student's $t$-Distribution Ratio
By Cochran's Theorem, define two independent random variables:
1. $Z \triangleq \frac{\bar{X} - \mu}{\sigma / \sqrt{N}} \sim \mathcal{N}(0, 1)$
2. $V \triangleq \frac{(N - 1)S^2}{\sigma^2} \sim \chi^2(\nu)$, where $\nu = N - 1$.

Form the ratio:
$$T \triangleq \frac{Z}{\sqrt{V / \nu}} = \frac{\frac{\bar{X} - \mu}{\sigma / \sqrt{N}}}{\sqrt{\frac{(N-1)S^2}{\sigma^2 (N-1)}}} = \frac{\frac{\bar{X} - \mu}{\sigma / \sqrt{N}}}{\frac{S}{\sigma}} = \mathbf{\frac{\bar{X} - \mu}{S / \sqrt{N}}}$$
The unknown population parameter $\sigma$ cancels out perfectly from both numerator and denominator!
Because $Z$ and $V$ are independent, the joint density is $f_{Z, V}(z, v) = f_Z(z) f_V(v)$.
Applying the bivariate change of variables $(Z, V) \mapsto (T = Z/\sqrt{V/\nu}, U = V)$ and integrating out $u$ yields the exact PDF of Student's $t$-distribution:
$$\mathbf{f_T(t; \nu) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\pi \nu} \, \Gamma\left(\frac{\nu}{2}\right)} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}}} \quad \blacksquare$$
As $\nu \to \infty$, $\left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}} \to e^{-t^2 / 2}$, recovering the standard normal Gaussian density!

#### Exact Two-Sided $(1 - \alpha)$ Confidence Interval for Population Mean $\mu$:
$$\text{CI}_{1-\alpha} = \left[ \bar{X} - t_{1-\alpha/2, N-1} \frac{S}{\sqrt{N}}, \quad \bar{X} + t_{1-\alpha/2, N-1} \frac{S}{\sqrt{N}} \right]$$

---

## Part 3: Geometric & Algebraic Interpretation

### The Geometry of Bessel's Correction in $\mathbb{R}^N$
Consider the sample vector $\mathbf{X} = [X_1, X_2, \dots, X_N]^T \in \mathbb{R}^N$.
Let $\mathbf{1} = [1, 1, \dots, 1]^T \in \mathbb{R}^N$ be the all-ones vector.

1. **Subspace of the Mean:**
   The vector containing the repeated sample mean is the orthogonal projection of $\mathbf{X}$ onto the 1D subspace spanned by $\mathbf{1}$:
   $$\bar{\mathbf{X}} = \left[\begin{matrix} \bar{X} \\ \vdots \\ \bar{X} \end{matrix}\right] = \frac{\mathbf{1}^T \mathbf{X}}{\mathbf{1}^T \mathbf{1}} \mathbf{1} = \frac{1}{N} \mathbf{1}\mathbf{1}^T \mathbf{X} = P_{\mathbf{1}} \mathbf{X}$$
   where $P_{\mathbf{1}} = \frac{1}{N}\mathbf{1}\mathbf{1}^T$ has rank 1.

2. **Subspace of Residuals (Deviations):**
   The deviation vector $\mathbf{e} = \mathbf{X} - \bar{\mathbf{X}}$ is:
   $$\mathbf{e} = (I - P_{\mathbf{1}})\mathbf{X} = \left[\begin{matrix} X_1 - \bar{X} \\ \vdots \\ X_N - \bar{X} \end{matrix}\right]$$
   The projection matrix $M = I - \frac{1}{N}\mathbf{1}\mathbf{1}^T$ has rank:
   $$\text{rank}(M) = \text{tr}(M) = \text{tr}(I_N) - \text{tr}\left(\frac{1}{N}\mathbf{1}\mathbf{1}^T\right) = N - 1$$

```
            X (Sample in R^N)
            ▲
            │ \
            │  \  e = X - X_bar (Residual vector in (N-1)D subspace)
            │   \
            │    ▼
            └────► X_bar = P_1 X (Projection onto 1D line of ones)
             Origin
```

3. **Why $N - 1$ Degrees of Freedom?**
   Because $\mathbf{1}^T \mathbf{e} = \sum_{i=1}^N (X_i - \bar{X}) = 0$, the vector $\mathbf{e}$ is constrained to lie on an **$(N - 1)$-dimensional hyperplane** orthogonal to $\mathbf{1}$.
   The sample variance is the squared norm of $\mathbf{e}$:
   $$(N - 1)S^2 = \|\mathbf{e}\|_2^2 = \mathbf{X}^T (I - P_{\mathbf{1}}) \mathbf{X}$$
   Estimating the sample mean $\bar{X}$ "eats up" exactly 1 degree of freedom, leaving only $N - 1$ independent squared dimensions!

---

## Part 4: Real-World Analogy

### The Soup Tasting Analogy
Imagine a chef preparing a 50-liter vat of soup.
- **Population:** Every molecule of soup in the entire 50-liter container ($N_{\text{pop}} \to \infty$).
- **Sample:** A single 10 mL tablespoon taken by the chef ($N = 1$).
- **The $i.i.d.$ Condition (Stirring):**
  - If the soup is thoroughly stirred before tasting, each molecule in the spoon is an independent, identically distributed draw from the population. The chef can reliably assess the salt concentration $\mu$ from just 10 mL!
  - If the soup is **not stirred** (salt has sunk to the bottom), tasting from the top gives a catastrophically biased estimate (**Covariate Shift / Distribution Shift**).
- **Sample Size Scaling:**
  To halve your estimation error, you do not need half the pot—you only need to take 4 tablespoons instead of 1 ($\mathcal{O}(1/\sqrt{N})$ scaling).

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us compute the sample statistics and a 95% Student's $t$ confidence interval by hand for a toy dataset of neural network validation losses across $N = 5$ random seeds.

### 1. Problem Setup & Toy Sample Vector
A research team tests a new transformer architecture across $N = 5$ seeds, recording validation perplexity losses:
$$\mathbf{x} = [x_1, x_2, x_3, x_4, x_5] = [2.40, 3.10, 1.80, 2.90, 2.80]$$

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Value in Walkthrough |
| :--- | :--- | :--- | :--- | :--- |
| $N$ | Sample size | Number of evaluation seeds / mini-batch size | Scalar integer | $5$ |
| $x_i$ | $i$-th observed sample realization | Perplexity loss of $i$-th seed run | Scalar float | $2.40, 3.10, \dots$ |
| $\bar{x}$ | Sample mean | Average validation benchmark score | Scalar float | $2.60$ |
| $x_i - \bar{x}$ | Deviation from the mean | Centered seed residual | Scalar float | $-0.20, +0.50, \dots$ |
| $SS$ | Sum of squared deviations | Total unexplained variance numerator | Scalar float | $1.0600$ |
| $S_n^2$ | Biased sample variance ($1/N$) | Maximum likelihood variance estimate | Scalar float | $0.2120$ |
| $S_{N-1}^2$ | Unbiased sample variance ($1/(N-1)$) | Bessel-corrected sample variance | Scalar float | $0.2650$ |
| $S$ | Sample standard deviation | Empirical spread across random seeds | Scalar float | $\sqrt{0.2650} \approx 0.51478$ |
| $\text{SE}$ | Standard error of the mean | Precision of the benchmark claim | Scalar float | $S / \sqrt{5} \approx 0.23022$ |
| $t_{\text{crit}}$ | Student's $t$ critical value | Confidence interval multiplier ($df=4$) | Scalar float | $2.776$ |

---

### 3. Step-by-Step Manual Arithmetic

#### Step 1: Compute Sample Mean $\bar{x}$
$$\sum_{i=1}^5 x_i = 2.40 + 3.10 + 1.80 + 2.90 + 2.80 = 13.00$$
$$\mathbf{\bar{x} = \frac{13.00}{5} = 2.6000}$$

---

#### Step 2: Compute Deviations and Squared Deviations
- Seed 1 ($x_1 = 2.40$):
  $$x_1 - \bar{x} = 2.40 - 2.60 = \mathbf{-0.20} \implies (-0.20)^2 = \mathbf{0.0400}$$
- Seed 2 ($x_2 = 3.10$):
  $$x_2 - \bar{x} = 3.10 - 2.60 = \mathbf{+0.50} \implies (+0.50)^2 = \mathbf{0.2500}$$
- Seed 3 ($x_3 = 1.80$):
  $$x_3 - \bar{x} = 1.80 - 2.60 = \mathbf{-0.80} \implies (-0.80)^2 = \mathbf{0.6400}$$
- Seed 4 ($x_4 = 2.90$):
  $$x_4 - \bar{x} = 2.90 - 2.60 = \mathbf{+0.30} \implies (+0.30)^2 = \mathbf{0.0900}$$
- Seed 5 ($x_5 = 2.80$):
  $$x_5 - \bar{x} = 2.80 - 2.60 = \mathbf{+0.20} \implies (+0.20)^2 = \mathbf{0.0400}$$

Check sum of linear deviations:
$$\sum_{i=1}^5 (x_i - \bar{x}) = -0.20 + 0.50 - 0.80 + 0.30 + 0.20 = \mathbf{0.0000} \quad \checkmark$$

Sum of squared deviations ($SS$):
$$\mathbf{SS = 0.0400 + 0.2500 + 0.6400 + 0.0900 + 0.0400 = 1.0600}$$

---

#### Step 3: Compute Sample Variances (Biased vs. Unbiased)
- **Biased Variance ($S_n^2$):**
  $$S_n^2 = \frac{SS}{N} = \frac{1.0600}{5} = \mathbf{0.2120}$$
- **Unbiased Variance ($S^2$):**
  $$S^2 = \frac{SS}{N - 1} = \frac{1.0600}{4} = \mathbf{0.2650}$$
- **Sample Standard Deviation ($S$):**
  $$S = \sqrt{0.2650} \approx \mathbf{0.514781}$$

---

#### Step 4: Compute Standard Error of the Mean ($\text{SE}$)
$$\text{SE}(\bar{x}) = \frac{S}{\sqrt{N}} = \frac{0.514781}{\sqrt{5}} = \frac{0.514781}{2.236068} \approx \mathbf{0.230217}$$

---

#### Step 5: Compute 95% Student's $t$ Confidence Interval
For $\alpha = 0.05$ (two-tailed, $1 - \alpha = 0.95$) with degrees of freedom $\nu = N - 1 = 4$:
From Student's $t$-table:
$$t_{0.975, 4} \approx \mathbf{2.776}$$

Margin of Error ($ME$):
$$ME = t_{\text{crit}} \times \text{SE} = 2.776 \times 0.230217 \approx \mathbf{0.639082}$$

Confidence Interval:
$$\text{CI}_{0.95} = [2.6000 - 0.6391, \quad 2.6000 + 0.6391] = [\mathbf{1.9609}, \quad \mathbf{3.2391}]$$

---

### 4. Visual Summary Grid

```
┌──────┬──────────┬──────────────┬──────────────────┬─────────────────────────────────┐
│ Seed │ Loss x_i │ Dev (x_i - μ)│ Sq Dev (x_i - μ)²│ Intermediate Totals             │
├──────┼──────────┼──────────────┼──────────────────┼─────────────────────────────────┤
│ 1    │  2.40    │    -0.20     │      0.0400      │                                 │
│ 2    │  3.10    │    +0.50     │      0.2500      │ Σ x_i = 13.00                   │
│ 3    │  1.80    │    -0.80     │      0.6400      │ Mean x_bar = 13.00 / 5 = 2.6000 │
│ 4    │  2.90    │    +0.30     │      0.0900      │                                 │
│ 5    │  2.80    │    +0.20     │      0.0400      │ Σ (x_i - x_bar) = 0.0000        │
├──────┼──────────┼──────────────┼──────────────────┼─────────────────────────────────┤
│ SUM  │ 13.00    │     0.00     │   SS = 1.0600    │ SS = 1.0600                     │
├──────┴──────────┴──────────────┴──────────────────┼─────────────────────────────────┤
│ Biased Variance S_n² (divide by N = 5)            │ S_n² = 1.0600 / 5 = 0.2120      │
│ Unbiased Variance S² (divide by N - 1 = 4)        │ S²   = 1.0600 / 4 = 0.2650      │
│ Sample Std Dev S                                  │ S = sqrt(0.2650) = 0.5148       │
│ Standard Error SE(x_bar) = S / sqrt(5)            │ SE = 0.5148 / 2.2361 = 0.2302   │
│ 95% Confidence Interval (t_crit = 2.776)          │ [1.9609, 3.2391]                │
└───────────────────────────────────────────────────┴─────────────────────────────────┘
```

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Empirical Distribution Function (EDF) Step Evaluation
Using the sorted toy dataset $\mathbf{x}_{(i)} = [1.80, 2.40, 2.80, 2.90, 3.10]$:
Evaluate the EDF $\hat{F}_5(t)$ at three test queries:
1. At $t = 1.50$: Since no points $\le 1.50$, $\hat{F}_5(1.50) = \frac{0}{5} = \mathbf{0.00}$.
2. At $t = 2.85$: Points $\le 2.85$ are $\{1.80, 2.40, 2.80\}$ (3 points), so $\hat{F}_5(2.85) = \frac{3}{5} = \mathbf{0.60}$.
3. At $t = 3.50$: All points $\le 3.50$, so $\hat{F}_5(3.50) = \frac{5}{5} = \mathbf{1.00}$.

---

### Illustration 2 (Boundary): The Sample Size $N = 1$ Singularity
What happens if you have only a single training sample or batch ($N = 1$)?
- $\bar{X} = X_1$ (sample mean exists).
- Bessel's sample variance formula:
  $$S^2 = \frac{1}{1 - 1} (X_1 - \bar{X})^2 = \frac{0}{0} \quad (\text{UNDEFINED!})$$
*Deep Learning Meaning:* A single data point contains zero information about population variance. This is why **Batch Normalization** fails catastrophically when batch size $B = 1$ during training (the batch variance $\sigma_B^2 = 0$, causing division by zero $\frac{x - \mu}{\sqrt{0 + \epsilon}}$), necessitating **Layer Normalization** or **Group Normalization** in small-batch regimes.

---

### Illustration 3 (Edge Case): Violation of the $i.i.d.$ Assumption in Autoregressive LLMs
Suppose you evaluate perplexity on a consecutive sequence of $N = 1000$ tokens from an autoregressive language model.
Tokens are strongly auto-correlated: $\text{Cov}(X_t, X_{t+k}) = \rho^k \sigma^2$ with correlation $\rho = 0.8$.
The true variance of the sample mean is:
$$\text{Var}(\bar{X}) = \frac{\sigma^2}{N} \left( 1 + 2 \sum_{k=1}^{N-1} \left(1 - \frac{k}{N}\right) \rho^k \right) \approx \frac{\sigma^2}{N} \left( \frac{1 + \rho}{1 - \rho} \right)$$
For $\rho = 0.8$:
$$\frac{1 + 0.8}{1 - 0.8} = \frac{1.8}{0.2} = \mathbf{9.0}$$
*The Consequence:* The true variance is **9 times larger** than the naive formula $\sigma^2 / N$, so the naive standard error is **underestimated by $3\times$**! Naive confidence intervals on non-$i.i.d.$ data yield false confidence and invalid benchmark claims.

---

### Illustration 4 (Numerical): Two-Sample Pooled Student's $t$-Test for Model Benchmarking
An AI lab tests whether a new FlashAttention kernel (Model B) achieves higher BLEU translation scores than the baseline model (Model A).
Both models are evaluated across $N_A = 5$ and $N_B = 5$ independent random training runs:
- Model A: $\mathbf{x}_A = [30.0, 32.0, 31.0, 33.0, 29.0]$
- Model B: $\mathbf{x}_B = [34.0, 36.0, 35.0, 37.0, 33.0]$

We test the null hypothesis $H_0: \mu_B - \mu_A = 0$ against $H_1: \mu_B > \mu_A$ at significance level $\alpha = 0.01$.

1. **Step 1: Compute Sample Means & Sum of Squares:**
   - Model A:
     $$\bar{x}_A = \frac{30 + 32 + 31 + 33 + 29}{5} = \frac{155.0}{5} = \mathbf{31.000000}$$
     $$SS_A = (30-31)^2 + (32-31)^2 + (31-31)^2 + (33-31)^2 + (29-31)^2 = 1 + 1 + 0 + 4 + 4 = \mathbf{10.000000}$$
     $$s_A^2 = \frac{SS_A}{N_A - 1} = \frac{10.0}{4} = \mathbf{2.500000}$$
   - Model B:
     $$\bar{x}_B = \frac{34 + 36 + 35 + 37 + 33}{5} = \frac{175.0}{5} = \mathbf{35.000000}$$
     $$SS_B = (34-35)^2 + (36-35)^2 + (35-35)^2 + (37-35)^2 + (33-35)^2 = 1 + 1 + 0 + 4 + 4 = \mathbf{10.000000}$$
     $$s_B^2 = \frac{SS_B}{N_B - 1} = \frac{10.0}{4} = \mathbf{2.500000}$$

2. **Step 2: Compute Pooled Sample Variance $s_p^2$:**
   Degrees of freedom $\nu = N_A + N_B - 2 = 5 + 5 - 2 = \mathbf{8}$.
   $$s_p^2 = \frac{(N_A - 1)s_A^2 + (N_B - 1)s_B^2}{N_A + N_B - 2} = \frac{4(2.50) + 4(2.50)}{8} = \frac{20.0}{8} = \mathbf{2.500000}$$

3. **Step 3: Compute Standard Error of the Difference:**
   $$\text{SE}(\bar{x}_B - \bar{x}_A) = \sqrt{s_p^2 \left( \frac{1}{N_A} + \frac{1}{N_B} \right)} = \sqrt{2.50 \left( \frac{1}{5} + \frac{1}{5} \right)} = \sqrt{2.50(0.40)} = \sqrt{1.000000} = \mathbf{1.000000}$$

4. **Step 4: Compute Test Statistic $t$:**
   $$t = \frac{\bar{x}_B - \bar{x}_A - 0}{\text{SE}} = \frac{35.00 - 31.00}{1.00} = \mathbf{+4.000000}$$

5. **Step 5: Decision & Confidence Interval:**
   From Student's $t$-table with $\nu = 8$ degrees of freedom:
   - Critical value at $\alpha = 0.01$ (one-tailed): $t_{0.99, 8} \approx \mathbf{2.896}$.
   - Critical value at $\alpha = 0.05$ (two-tailed): $t_{0.975, 8} \approx \mathbf{2.306}$.
   Since $t_{\text{calc}} = 4.000 > 2.896$, we **reject $H_0$** at the $1\%$ significance level ($p < 0.005$). The new FlashAttention kernel provides a statistically genuine improvement!
   Exact $95\%$ Confidence Interval for the true performance gain $\mu_B - \mu_A$:
   $$(\bar{x}_B - \bar{x}_A) \pm t_{0.975, 8} \times \text{SE} = 4.00 \pm 2.306(1.00) = [\mathbf{1.694000, 6.306000}]$$

---

### Illustration 5 (Numerical): Non-Parametric Bootstrap Resampling by Hand
In many deep learning metrics (e.g., median token generation latency or BLEU scores), the underlying sampling distribution is unknown or non-Gaussian, invalidating Student's $t$ normality assumptions.
The **Bootstrap** (Efron, 1979) estimates sampling distributions by resampling with replacement from the empirical dataset $\mathcal{D}$.

Consider a toy dataset of $N = 4$ inference request latencies (in milliseconds):
$$\mathcal{D} = \{10.0, 20.0, 30.0, 40.0\}$$
Target statistic: **Sample Median** $\hat{\theta} = \text{Median}(\mathbf{x}) = \frac{20.0 + 30.0}{2} = \mathbf{25.000000\text{ ms}}$.

1. **Generate $B = 5$ Bootstrap Resamples (Sampling $N = 4$ with Replacement):**
   - Resample 1: $\mathbf{x}^{*(1)} = [10.0, 10.0, 20.0, 40.0] \implies \hat{\theta}^{*(1)} = \frac{10 + 20}{2} = \mathbf{15.000000}$
   - Resample 2: $\mathbf{x}^{*(2)} = [20.0, 20.0, 30.0, 40.0] \implies \hat{\theta}^{*(2)} = \frac{20 + 30}{2} = \mathbf{25.000000}$
   - Resample 3: $\mathbf{x}^{*(3)} = [10.0, 30.0, 30.0, 40.0] \implies \hat{\theta}^{*(3)} = \frac{30 + 30}{2} = \mathbf{30.000000}$
   - Resample 4: $\mathbf{x}^{*(4)} = [20.0, 30.0, 40.0, 40.0] \implies \hat{\theta}^{*(4)} = \frac{30 + 40}{2} = \mathbf{35.000000}$
   - Resample 5: $\mathbf{x}^{*(5)} = [10.0, 20.0, 30.0, 30.0] \implies \hat{\theta}^{*(5)} = \frac{20 + 30}{2} = \mathbf{25.000000}$

2. **Compute Bootstrap Mean & Standard Error:**
   $$\bar{\theta}^* = \frac{15.0 + 25.0 + 30.0 + 35.0 + 25.0}{5} = \frac{130.0}{5} = \mathbf{26.000000}$$
   Compute squared deviations from $\bar{\theta}^*$:
   $$(15 - 26)^2 = 121, \quad (25 - 26)^2 = 1, \quad (30 - 26)^2 = 16, \quad (35 - 26)^2 = 81, \quad (25 - 26)^2 = 1$$
   $$\sum_{b=1}^5 (\hat{\theta}^{*(b)} - \bar{\theta}^*)^2 = 121 + 1 + 16 + 81 + 1 = \mathbf{220.000000}$$
   $$\widehat{\text{SE}}_{\text{boot}}(\hat{\theta}) = \sqrt{\frac{1}{B - 1} \sum_{b=1}^B (\hat{\theta}^{*(b)} - \bar{\theta}^*)^2} = \sqrt{\frac{220.0}{4}} = \sqrt{55.0} \approx \mathbf{7.416198\text{ ms}}$$

3. **Bootstrap Bias Estimate:**
   $$\widehat{\text{Bias}}(\hat{\theta}) = \bar{\theta}^* - \hat{\theta}_{\text{orig}} = 26.00 - 25.00 = \mathbf{+1.000000\text{ ms}}$$
   *Deep Learning Meaning:* Bootstrap allows us to estimate standard errors and confidence intervals for arbitrary black-box neural metrics (F1 score, AUC-ROC, perplexity) directly from data, without requiring Gaussian assumptions!

---

## Part 7: Deep Learning Connection & Application

### 1. Mini-Batch Stochastic Gradient Descent as a Sampling Estimator
In deep learning, the true objective is the **Expected Risk** over the unknown data distribution:
$$\mathcal{R}(\theta) = \mathbb{E}_{(x, y) \sim P_{\text{data}}}[\ell(f_\theta(x), y)]$$
Because $P_{\text{data}}$ is unknown, we draw a mini-batch sample $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^B$ of size $B$.
The mini-batch gradient $\mathbf{g}_B(\theta)$ is a **statistical estimator** of the true population gradient $\mathbf{g}^*(\theta) = \nabla_\theta \mathcal{R}(\theta)$:
$$\mathbf{g}_B(\theta) = \frac{1}{B} \sum_{i=1}^B \nabla_\theta \ell(f_\theta(x_i), y_i)$$
- **Unbiasedness:** $\mathbb{E}_{\mathcal{B}}[\mathbf{g}_B(\theta)] = \nabla_\theta \mathcal{R}(\theta)$.
- **Variance:** $\text{Var}(\mathbf{g}_B(\theta)) = \frac{1}{B} \Sigma(\theta)$, decaying at rate $\mathcal{O}(1/B)$.
This exact $\mathcal{O}(1/B)$ variance scaling governs the critical batch size and learning rate warmup schedules in large-scale transformer pretraining!

---

### 2. Empirical Risk Minimization (ERM) & Generalization Gap
Let $\hat{\mathcal{R}}_N(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$ be the Empirical Risk.
The **Generalization Gap** is the difference between true population risk and empirical sample risk:
$$\text{GenGap}(\theta) = \mathcal{R}(\theta) - \hat{\mathcal{R}}_N(\theta)$$
Statistical learning theory (Vapnik-Chervonenkis) bounds this gap using concentration inequalities:
$$P\left( \sup_{\theta \in \mathcal{H}} |\mathcal{R}(\theta) - \hat{\mathcal{R}}_N(\theta)| \le \mathcal{O}\left(\sqrt{\frac{\text{Capacity}(\mathcal{H})}{N}}\right) \right) \ge 1 - \delta$$
This shows that as the sample size $N \to \infty$, the sample risk reliably reflects population risk.

---

## Part 8: Code Implementation & Verification

The companion Python module [01_statistical_inference_and_sampling.py](./code/01_statistical_inference_and_sampling.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates $\bar{x} = 2.60$, $SS = 1.06$, $S_n^2 = 0.212$, $S^2 = 0.265$, $S = 0.51478$, $\text{SE} = 0.23022$, and 95% Student's $t$ CI $[1.9609, 3.2391]$.
2. **Bessel's Correction Simulation:** Demonstrates across 100,000 trials that $S_n^2$ underestimates $\sigma^2$ by $\frac{N-1}{N}$, while $S^2$ is perfectly unbiased.
3. **Cochran's Theorem Independence Test:** Numerically verifies the statistical independence between $\bar{X}$ and $S^2$ for Gaussian vs. non-Gaussian samples.
4. **Glivenko-Cantelli EDF Convergence:** Tracks Kolmogorov-Smirnov supremum distance $\|\hat{F}_N - F\|_\infty$ decaying to zero as $N \to 100{,}000$.
5. **Coverage Probability of Student's $t$ Intervals:** Confirms that 95% confidence intervals achieve exact $0.950$ empirical coverage under repeated Gaussian sampling.
