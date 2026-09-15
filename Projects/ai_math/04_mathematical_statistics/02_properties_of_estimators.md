# Chapter 4.2: Properties of Estimators (Bias, Variance, Consistency, Cramér-Rao)

---

## Part 1: Intuition & 101 Motivation

When training a machine learning model, you have dozens of design choices: loss functions, normalization layers, regularization penalties, and gradient estimators. How do we rigorously judge whether one estimator is superior to another?

Mathematical statistics provides four fundamental criteria to evaluate any estimator $\hat{\theta}$:
1. **Bias:** On average, does the estimator hit the true parameter $\theta^*$, or is it systematically skewed?
2. **Variance:** If you train the model on different random subsets of data, how much does the estimate fluctuate?
3. **Consistency:** If you feed the model infinite training data ($N \to \infty$), are you mathematically guaranteed to recover the exact ground truth?
4. **Efficiency:** Does the estimator extract every last drop of information from the data, reaching the physical lower bound on variance (the **Cramér-Rao Lower Bound**)?

```
     High Bias, Low Variance         Low Bias, High Variance         Low Bias, Low Variance
         (Underfitting)                   (Overfitting)                  (The Ideal Model)
            ┌───────┐                        ┌───────┐                        ┌───────┐
            │   🎯   │                        │ 🎯    │                        │   🎯   │
            │  ***  │                        │ *   * │                        │   *   │
            │  ***  │                        │   *   │                        │       │
            └───────┘                        └───────┘                        └───────┘
     Systematically off target         Scattered everywhere             Dead center on bullseye
```

Understanding these properties is what allows deep learning practitioners to master the **Bias-Variance Tradeoff**, understand why $L_2$ weight decay improves generalization, and utilize **Fisher Information** for second-order optimization (e.g., Natural Gradient Descent and K-FAC).

---

## Part 2: Rigorous Mathematical Formulation

Let $\mathbf{X} = (X_1, X_2, \dots, X_N) \overset{i.i.d.}{\sim} P_\theta$ be a sample from a distribution parametrized by unknown $\theta \in \Theta \subseteq \mathbb{R}^k$.
Let $\hat{\theta} = T(\mathbf{X})$ be an estimator of $\theta$.

### 1. Bias of an Estimator
The **bias** of $\hat{\theta}$ is the difference between its expected value and the true parameter $\theta$:
$$\text{Bias}_\theta(\hat{\theta}) = \mathbb{E}_{\mathbf{X} \sim P_\theta}[\hat{\theta}] - \theta$$

- **Unbiased Estimator:** $\hat{\theta}$ is unbiased if $\text{Bias}_\theta(\hat{\theta}) = 0$ for all $\theta \in \Theta$, i.e.:
  $$\mathbb{E}[\hat{\theta}] = \theta$$
- **Asymptotically Unbiased:** $\hat{\theta}_N$ is asymptotically unbiased if:
  $$\lim_{N \to \infty} \mathbb{E}[\hat{\theta}_N] = \theta$$

---

### 2. Variance of an Estimator
The **variance** measures the dispersion of the estimator around its own expected value across repeated independent draws of the sample $\mathbf{X}$:
$$\text{Var}(\hat{\theta}) = \mathbb{E}\left[ (\hat{\theta} - \mathbb{E}[\hat{\theta}])^2 \right]$$
For vector-valued estimators $\hat{\theta} \in \mathbb{R}^k$, the variance is represented by the **Covariance Matrix**:
$$\text{Cov}(\hat{\theta}) = \mathbb{E}\left[ (\hat{\theta} - \mathbb{E}[\hat{\theta}])(\hat{\theta} - \mathbb{E}[\hat{\theta}])^T \right]$$

---

### 3. Mean Squared Error (MSE) & Its Decomposition
The **Mean Squared Error (MSE)** measures the total expected squared Euclidean distance from the true parameter:
$$\text{MSE}(\hat{\theta}) = \mathbb{E}\left[ (\hat{\theta} - \theta)^2 \right]$$

#### Theorem: The MSE Decomposition
For any scalar estimator $\hat{\theta}$:
$$\mathbf{\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})}$$

**Rigorous Proof:**
Add and subtract the expected value $\mathbb{E}[\hat{\theta}]$ inside the square:
$$\begin{aligned}
\text{MSE}(\hat{\theta}) &= \mathbb{E}\left[ \left( (\hat{\theta} - \mathbb{E}[\hat{\theta}]) + (\mathbb{E}[\hat{\theta}] - \theta) \right)^2 \right] \\
&= \mathbb{E}\left[ (\hat{\theta} - \mathbb{E}[\hat{\theta}])^2 + 2(\hat{\theta} - \mathbb{E}[\hat{\theta}])(\mathbb{E}[\hat{\theta}] - \theta) + (\mathbb{E}[\hat{\theta}] - \theta)^2 \right]
\end{aligned}$$
Distribute the expectation operator $\mathbb{E}[\cdot]$:
1. $\mathbb{E}\left[ (\hat{\theta} - \mathbb{E}[\hat{\theta}])^2 \right] = \text{Var}(\hat{\theta})$.
2. The term $(\mathbb{E}[\hat{\theta}] - \theta) = \text{Bias}(\hat{\theta})$ is a deterministic scalar constant, so:
   $$\mathbb{E}\left[ 2(\hat{\theta} - \mathbb{E}[\hat{\theta}])(\mathbb{E}[\hat{\theta}] - \theta) \right] = 2 \text{Bias}(\hat{\theta}) \cdot \mathbb{E}[\hat{\theta} - \mathbb{E}[\hat{\theta}]] = 2 \text{Bias}(\hat{\theta}) \cdot (\mathbb{E}[\hat{\theta}] - \mathbb{E}[\hat{\theta}]) = 0$$
3. $\mathbb{E}\left[ (\mathbb{E}[\hat{\theta}] - \theta)^2 \right] = \text{Bias}(\hat{\theta})^2$.

Combining these yields:
$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2 \quad \blacksquare$$

> [!IMPORTANT]
> **Why Unbiasedness is Overrated in Deep Learning:**
> An unbiased estimator does *not* necessarily have lower error than a biased one! A biased estimator with dramatically smaller variance can achieve substantially lower total MSE. This is the exact principle behind $L_2$ regularization (Weight Decay): we accept a small non-zero bias in exchange for a massive drop in weight variance.

---

### 4. Consistency
An estimator sequence $\hat{\theta}_N$ is **weakly consistent** if it converges in probability to the true parameter as sample size $N \to \infty$:
$$\hat{\theta}_N \xrightarrow{P} \theta^* \iff \lim_{N \to \infty} P\left( |\hat{\theta}_N - \theta^*| \ge \epsilon \right) = 0 \quad \forall \epsilon > 0$$

An estimator is **strongly consistent** if it converges almost surely:
$$\hat{\theta}_N \xrightarrow{\text{a.s.}} \theta^* \iff P\left( \lim_{N \to \infty} \hat{\theta}_N = \theta^* \right) = 1$$

#### Sufficient Condition for Consistency:
If the bias and the variance of an estimator both vanish asymptotically:
$$\lim_{N \to \infty} \text{Bias}(\hat{\theta}_N) = 0 \quad \text{and} \quad \lim_{N \to \infty} \text{Var}(\hat{\theta}_N) = 0 \implies \lim_{N \to \infty} \text{MSE}(\hat{\theta}_N) = 0$$
By Chebyshev's inequality, $\text{MSE} \to 0$ guarantees that $\hat{\theta}_N \xrightarrow{P} \theta^*$ (**consistent**).

---

### 5. The Score Function & Fisher Information

#### The Log-Likelihood Function:
Let $f(\mathbf{x}; \theta)$ be the joint density of the sample. The log-likelihood is:
$$\ell(\theta; \mathbf{x}) = \ln f(\mathbf{x}; \theta) = \sum_{i=1}^N \ln f(x_i; \theta)$$

#### The Score Function $S(\theta)$:
The **Score** is the gradient of the log-likelihood with respect to the parameter $\theta$:
$$S(\theta; \mathbf{X}) = \nabla_\theta \ln f(\mathbf{X}; \theta) = \frac{\nabla_\theta f(\mathbf{X}; \theta)}{f(\mathbf{X}; \theta)}$$

#### Theorem: The Expected Score is Always Zero
$$\mathbb{E}_{\mathbf{X} \sim P_\theta}[S(\theta; \mathbf{X})] = \mathbf{0}$$

**Proof:**
Assuming regularity conditions allowing interchange of differentiation and integration:
$$\begin{aligned}
\mathbb{E}[S(\theta; \mathbf{X})] &= \int_{\mathcal{X}} \frac{\nabla_\theta f(\mathbf{x}; \theta)}{f(\mathbf{x}; \theta)} f(\mathbf{x}; \theta) \, d\mathbf{x} = \int_{\mathcal{X}} \nabla_\theta f(\mathbf{x}; \theta) \, d\mathbf{x} \\
&= \nabla_\theta \int_{\mathcal{X}} f(\mathbf{x}; \theta) \, d\mathbf{x} = \nabla_\theta (1) = \mathbf{0} \quad \blacksquare
\end{aligned}$$

#### Fisher Information $I(\theta)$:
The **Fisher Information** is the variance of the score function:
$$I_N(\theta) = \text{Var}(S(\theta; \mathbf{X})) = \mathbb{E}\left[ S(\theta; \mathbf{X}) S(\theta; \mathbf{X})^T \right]$$

#### Theorem: Second-Derivative Formulation of Fisher Information
Under standard regularity conditions:
$$I_N(\theta) = -\mathbb{E}\left[ \nabla_\theta^2 \ln f(\mathbf{X}; \theta) \right] = -\mathbb{E}[\mathcal{H}_\ell(\theta)]$$
Fisher information equals the **negative expected Hessian** (curvature) of the log-likelihood function!

For $N$ $i.i.d.$ observations, Fisher information is strictly additive:
$$I_N(\theta) = N \cdot I_1(\theta)$$

---

### 6. The Cramér-Rao Lower Bound (CRLB)

#### Theorem (Cramér-Rao Inequality):
Let $\hat{\theta}$ be any unbiased estimator of a scalar parameter $\theta$ ($\mathbb{E}[\hat{\theta}] = \theta$).
Under regularity conditions, the variance of $\hat{\theta}$ is bounded from below:
$$\mathbf{\text{Var}(\hat{\theta}) \ge \frac{1}{I_N(\theta)} = \frac{1}{N \cdot I_1(\theta)}}$$

For a biased estimator with bias $b(\theta) = \mathbb{E}[\hat{\theta}] - \theta$:
$$\text{Var}(\hat{\theta}) \ge \frac{(1 + b'(\theta))^2}{I_N(\theta)}$$

**Rigorous Proof (via Cauchy-Schwarz Inequality):**
Consider the covariance between the estimator $\hat{\theta}$ and the score $S(\theta) = \frac{\partial \ell}{\partial \theta}$:
$$\text{Cov}(\hat{\theta}, S(\theta)) = \mathbb{E}[\hat{\theta} S(\theta)] - \mathbb{E}[\hat{\theta}]\underbrace{\mathbb{E}[S(\theta)]}_{=0} = \mathbb{E}[\hat{\theta} S(\theta)]$$
Expand the expectation:
$$\mathbb{E}[\hat{\theta} S(\theta)] = \int \hat{\theta}(\mathbf{x}) \frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta} f(\mathbf{x}; \theta) \, d\mathbf{x} = \int \hat{\theta}(\mathbf{x}) \frac{\partial f(\mathbf{x}; \theta)}{\partial \theta} \, d\mathbf{x}$$
Interchange integration and differentiation:
$$\mathbb{E}[\hat{\theta} S(\theta)] = \frac{\partial}{\partial \theta} \int \hat{\theta}(\mathbf{x}) f(\mathbf{x}; \theta) \, d\mathbf{x} = \frac{\partial}{\partial \theta} \mathbb{E}[\hat{\theta}]$$
Since $\hat{\theta}$ is unbiased, $\mathbb{E}[\hat{\theta}] = \theta$, so $\frac{\partial}{\partial \theta} \theta = 1$.
Thus:
$$\text{Cov}(\hat{\theta}, S(\theta)) = 1$$

Now apply the **Cauchy-Schwarz Inequality** for random variables:
$$\left( \text{Cov}(\hat{\theta}, S(\theta)) \right)^2 \le \text{Var}(\hat{\theta}) \cdot \text{Var}(S(\theta))$$
Substitute $\text{Cov}(\hat{\theta}, S(\theta)) = 1$ and $\text{Var}(S(\theta)) = I_N(\theta)$:
$$1^2 \le \text{Var}(\hat{\theta}) \cdot I_N(\theta) \implies \text{Var}(\hat{\theta}) \ge \frac{1}{I_N(\theta)} \quad \blacksquare$$

---

### 7. Statistical Efficiency
The **efficiency** of an unbiased estimator $\hat{\theta}$ is the ratio of the Cramér-Rao lower bound to its actual variance:
$$\text{eff}(\hat{\theta}) = \frac{\text{CRLB}}{\text{Var}(\hat{\theta})} = \frac{1}{I_N(\theta) \cdot \text{Var}(\hat{\theta})} \in [0, 1]$$

- An estimator is **efficient** if $\text{eff}(\hat{\theta}) = 1$ for all $\theta$.
- An efficient estimator is automatically the **Minimum Variance Unbiased Estimator (MVUE)**.

---

## Part 3: Geometric & Algebraic Interpretation

### The Geometry of Cauchy-Schwarz & The Score Direction
In the Hilbert space $\mathcal{L}^2(P_\theta)$ of zero-mean random variables with inner product $\langle U, V \rangle = \mathbb{E}[U V]$:
- The score $S(\theta)$ is a vector in this Hilbert space.
- The centered estimator $\hat{\theta} - \theta$ is another vector.
- The Cramér-Rao inequality is simply:
  $$\cos^2(\alpha) = \frac{\langle \hat{\theta} - \theta, S(\theta) \rangle^2}{\|\hat{\theta} - \theta\|^2 \|S(\theta)\|^2} \le 1$$

```
            Hilbert Space of Random Variables L^2
                    ▲
                    │     ^
                    │    /  Centered Estimator (θ - θ)
                    │   /
                    │  /  α (Angle between Estimator & Score)
                    │ /
                    └───────────► Score Vector S(θ)
```

**When is an estimator 100% efficient ($\cos(\alpha) = 1$)?**
Equality holds in Cauchy-Schwarz if and only if the estimator is perfectly collinear with the score vector:
$$\hat{\theta}(\mathbf{x}) - \theta = c(\theta) \cdot S(\theta; \mathbf{x})$$
This condition is satisfied if and only if the underlying probability distribution belongs to the **1-parameter Exponential Family** (Gaussian, Bernoulli, Poisson, Exponential)!

---

## Part 4: Real-World Analogy

### The Sniper vs. The Shotgun (Bias vs. Variance)
Imagine two marksmen shooting at a bullseye:
- **Marksman A (High Bias, Low Variance — The Misaligned Sniper):**
  Uses a high-powered sniper rifle with a sight knocked 10 cm to the left. Every shot lands within a tight 1 mm cluster, but always 10 cm away from the bullseye.
- **Marksman B (Low Bias, High Variance — The Untrained Novice):**
  Aims dead center, but has shaking hands. The shots are scattered across a 50 cm circle, but their mathematical average is centered exactly on the bullseye.
- **Marksman C (The Shrinkage Solution):**
  Recognizes the sniper's sight is biased by 10 cm, but rather than trying to fix it completely (which might introduce wild wobble), adjusts the sight 8 cm closer. The combination of minimal spread and near-zero drift achieves the lowest average miss distance (**Minimum MSE**).

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us evaluate and compare **three competing estimators** for the mean $\mu$ of a Gaussian population $X \sim \mathcal{N}(\mu, \sigma^2)$ with known variance $\sigma^2 = 4.0$ using a sample of size $N = 4$.

### 1. Problem Setup & The Three Estimators
Let $X_1, X_2, X_3, X_4 \overset{i.i.d.}{\sim} \mathcal{N}(\mu, \sigma^2 = 4.0)$.
- **Estimator 1 (Sample Mean $\hat{\mu}_1$):**
  $$\hat{\mu}_1 = \frac{1}{4}(X_1 + X_2 + X_3 + X_4)$$
- **Estimator 2 (First Observation Only $\hat{\mu}_2$):**
  $$\hat{\mu}_2 = X_1$$
- **Estimator 3 (Shrinkage / Regularized Estimator $\hat{\mu}_3$):**
  $$\hat{\mu}_3 = 0.80 \cdot \hat{\mu}_1 = \frac{1}{5}(X_1 + X_2 + X_3 + X_4)$$

Let the true population mean be $\mu = 2.0$.

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $N$ | Sample size | Batch size | Integer scalar | $4$ |
| $\sigma^2$ | Population variance | Inherent observation noise | Float scalar | $4.00$ |
| $\mu$ | True unknown parameter | True target weight / center | Float scalar | $2.00$ |
| $I_1(\mu)$ | Single-sample Fisher Information | Per-sample gradient curvature | Float scalar | $1 / \sigma^2 = 0.25$ |
| $I_N(\mu)$ | Total Fisher Information | Total batch Hessian magnitude | Float scalar | $N \cdot I_1 = 1.00$ |
| $\text{CRLB}$ | Cramér-Rao Lower Bound | Theoretical minimum variance floor | Float scalar | $1 / I_N = 1.00$ |
| $\text{Bias}(\hat{\mu})$ | Expected estimation offset | Model underfitting error | Float scalar | Calculated per estimator |
| $\text{Var}(\hat{\mu})$ | Estimator variance | Overfitting fluctuation error | Float scalar | Calculated per estimator |
| $\text{MSE}(\hat{\mu})$ | Total Mean Squared Error | Validation test loss | Float scalar | $\text{Bias}^2 + \text{Var}$ |
| $\text{eff}(\hat{\mu})$ | Statistical efficiency | Information utilization ratio | Percentage | $\text{CRLB} / \text{Var}$ |

---

### 3. Step-by-Step Manual Arithmetic

#### Step 1: Compute Fisher Information & CRLB Floor
For $X \sim \mathcal{N}(\mu, \sigma^2)$:
$$\ln f(x; \mu) = -\frac{1}{2} \ln(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$$
Score function:
$$S(\mu; x) = \frac{\partial \ln f}{\partial \mu} = \frac{x - \mu}{\sigma^2}$$
Second derivative:
$$\frac{\partial^2 \ln f}{\partial \mu^2} = -\frac{1}{\sigma^2}$$
Single-sample Fisher Information:
$$I_1(\mu) = -\mathbb{E}\left[ -\frac{1}{\sigma^2} \right] = \frac{1}{\sigma^2} = \frac{1}{4.0} = \mathbf{0.2500}$$
Total Fisher Information for $N = 4$:
$$I_4(\mu) = 4 \times I_1(\mu) = 4 \times 0.25 = \mathbf{1.0000}$$
Cramér-Rao Lower Bound for any unbiased estimator:
$$\mathbf{\text{CRLB} = \frac{1}{I_4(\mu)} = \frac{1}{1.0000} = 1.0000}$$

---

#### Step 2: Evaluate Estimator 1 (Sample Mean $\hat{\mu}_1 = \bar{X}$)
- **Expectation & Bias:**
  $$\mathbb{E}[\hat{\mu}_1] = \frac{1}{4}(4\mu) = \mu = 2.00 \implies \mathbf{\text{Bias}(\hat{\mu}_1) = 2.00 - 2.00 = 0.0000} \quad (\text{Unbiased})$$
- **Variance:**
  $$\text{Var}(\hat{\mu}_1) = \text{Var}\left( \frac{1}{4}\sum_{i=1}^4 X_i \right) = \frac{1}{16} \sum_{i=1}^4 \text{Var}(X_i) = \frac{4 \times 4.0}{16} = \mathbf{1.0000}$$
- **Mean Squared Error:**
  $$\text{MSE}(\hat{\mu}_1) = \text{Bias}^2 + \text{Var} = 0.00^2 + 1.0000 = \mathbf{1.0000}$$
- **Efficiency:**
  $$\text{eff}(\hat{\mu}_1) = \frac{\text{CRLB}}{\text{Var}(\hat{\mu}_1)} = \frac{1.0000}{1.0000} = \mathbf{100.0\%} \quad (\mathbf{\text{Efficient / MVUE}})$$

---

#### Step 3: Evaluate Estimator 2 (First Sample Only $\hat{\mu}_2 = X_1$)
- **Expectation & Bias:**
  $$\mathbb{E}[\hat{\mu}_2] = \mathbb{E}[X_1] = \mu = 2.00 \implies \mathbf{\text{Bias}(\hat{\mu}_2) = 0.0000} \quad (\text{Unbiased})$$
- **Variance:**
  $$\text{Var}(\hat{\mu}_2) = \text{Var}(X_1) = \sigma^2 = \mathbf{4.0000}$$
- **Mean Squared Error:**
  $$\text{MSE}(\hat{\mu}_2) = \text{Bias}^2 + \text{Var} = 0.00^2 + 4.0000 = \mathbf{4.0000}$$
- **Efficiency:**
  $$\text{eff}(\hat{\mu}_2) = \frac{\text{CRLB}}{\text{Var}(\hat{\mu}_2)} = \frac{1.0000}{4.0000} = \mathbf{25.0\%}$$
*Verdict:* Completely unbiased, but discards 75% of the data, resulting in $4\times$ larger error!

---

#### Step 4: Evaluate Estimator 3 (Shrinkage Estimator $\hat{\mu}_3 = 0.80 \bar{X}$)
- **Expectation & Bias:**
  $$\mathbb{E}[\hat{\mu}_3] = 0.80 \mathbb{E}[\bar{X}] = 0.80 \times 2.00 = 1.6000$$
  $$\mathbf{\text{Bias}(\hat{\mu}_3) = 1.6000 - 2.0000 = -0.4000} \quad (\mathbf{\text{Biased}})$$
- **Variance:**
  $$\text{Var}(\hat{\mu}_3) = (0.80)^2 \text{Var}(\bar{X}) = 0.64 \times 1.0000 = \mathbf{0.6400}$$
- **Mean Squared Error:**
  $$\text{MSE}(\hat{\mu}_3) = \text{Bias}^2 + \text{Var} = (-0.4000)^2 + 0.6400 = 0.1600 + 0.6400 = \mathbf{0.8000}$$
- **Comparison to CRLB:**
  $$\mathbf{\text{MSE}(\hat{\mu}_3) = 0.8000 < \text{MSE}(\hat{\mu}_1) = 1.0000!}$$

> [!TIP]
> **The Deep Learning Epiphany:**
> Estimator 3 is **biased**, yet its total Mean Squared Error is **20% lower** than the best possible unbiased estimator!
> By intentionally shrinking our estimate toward zero, the variance dropped from $1.00$ to $0.64$ ($\Delta = -0.36$), while squared bias only increased by $+0.16$, yielding a net win of $-0.20$ in total error.
> **This is the exact mathematical foundation of Weight Decay ($L_2$ regularization) in neural networks!**

---

### 4. Visual Summary Grid

```
┌─────────────────────────┬──────────────┬──────────────┬──────────────┬─────────────┐
│ Estimator               │ Bias (μ=2.0) │ Variance     │ Total MSE    │ Efficiency  │
├─────────────────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ CRLB Theoretical Floor  │  0.0000      │  1.0000      │  1.0000      │ 100.0%      │
├─────────────────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ 1. Sample Mean (X_bar)  │  0.0000      │  1.0000      │  1.0000      │ 100.0% (MVUE│
│ 2. Single Sample (X_1)  │  0.0000      │  4.0000      │  4.0000      │  25.0%      │
│ 3. Shrinkage (0.8 X_bar)│ -0.4000      │  0.6400      │  0.8000      │  — (Biased) │
└─────────────────────────┴──────────────┴──────────────┴──────────────┴─────────────┘
```

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Bernoulli Trials & Binary Cross-Entropy CRLB
Let $X_1, \dots, X_N \overset{i.i.d.}{\sim} \text{Bernoulli}(p)$.
1. **Log-Likelihood:**
   $$\ell(p) = \sum_{i=1}^N \left[ X_i \ln p + (1 - X_i) \ln(1 - p) \right]$$
2. **Score Function:**
   $$S(p) = \frac{\sum X_i}{p} - \frac{N - \sum X_i}{1 - p} = \frac{\sum X_i - N p}{p(1 - p)}$$
3. **Fisher Information:**
   $$I_N(p) = \text{Var}(S(p)) = \frac{\text{Var}(\sum X_i)}{p^2 (1 - p)^2} = \frac{N p (1 - p)}{p^2 (1 - p)^2} = \frac{N}{p(1 - p)}$$
4. **CRLB:**
   $$\text{CRLB} = \frac{1}{I_N(p)} = \frac{p(1 - p)}{N}$$
5. **Sample Proportion $\hat{p} = \frac{1}{N}\sum X_i$:**
   $$\text{Var}(\hat{p}) = \frac{p(1 - p)}{N} = \text{CRLB}$$
*Conclusion:* The standard empirical frequency estimator for binary classification achieves 100% efficiency.

---

### Illustration 2 (Boundary): Uniform Distribution & Failure of CRLB (Super-Efficiency)
Let $X_1, \dots, X_N \overset{i.i.d.}{\sim} \text{Uniform}(0, \theta)$.
The density is $f(x; \theta) = \frac{1}{\theta} \mathbf{1}_{\{0 \le x \le \theta\}}$.
- The support $[0, \theta]$ **depends directly on the parameter $\theta$**.
- This violates the fundamental Leibniz regularity condition allowing interchange of integration and differentiation ($\frac{d}{d\theta} \int = \int \frac{\partial}{\partial \theta}$).
- **The Cramér-Rao Lower Bound does NOT apply!**
- The maximum sample value $X_{(N)} = \max(X_1, \dots, X_N)$ has distribution:
  $$P(X_{(N)} \le t) = \left(\frac{t}{\theta}\right)^N \implies \mathbb{E}[X_{(N)}] = \frac{N}{N + 1}\theta$$
- Unbiased estimator: $\hat{\theta}^* = \frac{N + 1}{N} X_{(N)}$.
- Variance of $\hat{\theta}^*$:
  $$\text{Var}(\hat{\theta}^*) = \frac{\theta^2}{N(N + 2)} = \mathbf{\mathcal{O}\left(\frac{1}{N^2}\right)!}$$
*Significance:* Because CRLB regularity fails, this estimator achieves a variance decaying at $\mathcal{O}(1/N^2)$ ("super-efficiency"), beating the standard parametric rate $\mathcal{O}(1/N)$!

---

### Illustration 3 (Edge Case): An Estimator that is Consistent but Biased
Consider the standard plug-in sample variance $S_n^2 = \frac{1}{N}\sum_{i=1}^N (X_i - \bar{X})^2$.
- **Bias:**
  $$\mathbb{E}[S_n^2] = \frac{N - 1}{N} \sigma^2 = \sigma^2 - \frac{\sigma^2}{N} \implies \text{Bias} = -\frac{\sigma^2}{N} \ne 0 \quad (\text{Biased for all finite } N)$$
- **Asymptotic Limit:**
  $$\lim_{N \to \infty} \text{Bias}(S_n^2) = \lim_{N \to \infty} \left(-\frac{\sigma^2}{N}\right) = 0$$
- **Variance:**
  $$\text{Var}(S_n^2) = \mathcal{O}\left(\frac{1}{N}\right) \to 0 \quad \text{as } N \to \infty$$
*Conclusion:* Since both Bias and Variance vanish as $N \to \infty$, $S_n^2$ is **strictly consistent**, despite being biased at every finite sample size.

---

## Part 7: Deep Learning Connection & Application

### 1. Natural Gradient Descent & The Fisher Information Matrix
In standard deep learning, Gradient Descent updates parameters in Euclidean parameter space:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$
However, parameter space Euclidean distance $\|\theta_1 - \theta_2\|_2$ is arbitrary and depends on network parametrization. What truly matters is the **distance between the resulting output probability distributions** $P_{\theta_1}$ and $P_{\theta_2}$!

By the second-order Taylor expansion of KL Divergence:
$$D_{\text{KL}}(P_\theta \parallel P_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F(\theta) d\theta$$
where $F(\theta) = \mathbb{E}[\nabla_\theta \ln p(x; \theta) \nabla_\theta \ln p(x; \theta)^T]$ is the **Fisher Information Matrix**.

**Natural Gradient Descent (Amari, 1998):**
$$\theta_{t+1} = \theta_t - \eta F(\theta_t)^{-1} \nabla_\theta \mathcal{L}(\theta_t)$$
Preconditioning the gradient with the inverse Fisher Information Matrix ensures steepest descent along the **statistical Riemannian manifold**, invariant to network reparameterization!

---

### 2. K-FAC (Kronecker-Factored Approximate Curvature)
For modern neural networks with millions of parameters, the full Fisher Information Matrix $F(\theta) \in \mathbb{R}^{P \times P}$ is impossibly massive to invert ($P \approx 10^8 \implies F$ requires petabytes of memory).
**K-FAC** approximates the layer-wise Fisher blocks as the Kronecker product of two much smaller matrices:
$$F_l \approx A_{l-1} \otimes S_l$$
where $A_{l-1}$ is the covariance of input activations, and $S_l$ is the covariance of output pre-activation gradient derivatives. This reduces inversion cost from $\mathcal{O}(d_{\text{in}}^3 d_{\text{out}}^3)$ to $\mathcal{O}(d_{\text{in}}^3 + d_{\text{out}}^3)$, bringing second-order Fisher efficiency into practical deep learning!

---

## Part 8: Code Implementation & Verification

The companion Python module [02_properties_of_estimators.py](./code/02_properties_of_estimators.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact numerical calculations of Bias, Variance, and MSE for Estimators 1, 2, and 3 under Gaussian sampling.
2. **MSE vs. Shrinkage Factor Curve:** Computes empirical MSE across 100,000 trials as a function of the shrinkage parameter $\alpha \in [0.5, 1.2]$, confirming that the minimum MSE occurs at $\alpha < 1.0$ (proving the mathematical benefit of weight decay).
3. **Cramér-Rao Lower Bound Verification:** Simulates Bernoulli trials to confirm that the variance of the sample mean reaches the theoretical CRLB $\frac{p(1-p)}{N}$ with 100% efficiency.
4. **Uniform Distribution Super-Efficiency:** Demonstrates that the unbiased sample maximum estimator for $\text{Uniform}(0, \theta)$ achieves variance decay at $\mathcal{O}(1/N^2)$, verifying CRLB regularity breakdown.
5. **Fisher Information Preconditioned Natural Gradient Step:** Implements a toy 2-parameter logistic regression model comparing standard Euclidean gradient descent against the Natural Gradient step.
