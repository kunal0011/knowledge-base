# Chapter 4.3: Maximum Likelihood Estimation (MLE) & Loss Function Derivations

---

## Part 1: Intuition & 101 Motivation

Why is **Maximum Likelihood Estimation (MLE)** the beating heart of modern deep learning?

Consider how every neural network is trained:
- You pass an image into a ResNet or a prompt into an LLM.
- The model outputs raw real numbers (logits).
- You pass the logits through a Softmax or Sigmoid layer.
- You compute `torch.nn.CrossEntropyLoss()` or `torch.nn.MSELoss()`.
- You run `loss.backward()` and update parameters with Adam.

Where did those loss functions come from? Why do we use Mean Squared Error for regression, Binary Cross-Entropy for two classes, and Softmax Cross-Entropy for multiple classes?

**They are all Maximum Likelihood Estimators in disguise.**

The principle of MLE is intuitive:
> *"Given the training data $\mathcal{D} = \{x_1, \dots, x_N\}$ that nature actually produced, choose the model parameters $\theta$ that make the observed data as probable as possible."*

```
                 Parametric Probability Model P(y | x; θ)
                                    ▲
                                    │  Adjust θ to maximize
                                    │  the height of the curve
                                    │  at the observed points!
                         ╭──────────┴──────────╮
                         │                     │
                       ┌─┴─┐                 ┌─┴─┐
                       │ * │ (y_1)           │ * │ (y_2)
                   ────┴───┴─────────────────┴───┴────► Output Space y
```

By choosing different probability distributions for the output noise—Gaussian, Bernoulli, Multinomial, or Laplace—the exact same MLE principle automatically derives **MSE**, **BCE**, **Categorical Cross-Entropy**, and **MAE**.

---

## Part 2: Rigorous Mathematical Formulation

Let $\mathbf{X} = (X_1, X_2, \dots, X_N)$ be a sample of $N$ observations drawn $i.i.d.$ from a probability distribution with probability density (or mass) function $f(x; \theta)$, where $\theta \in \Theta \subseteq \mathbb{R}^d$ is unknown.

### 1. The Likelihood and Log-Likelihood Functions
- The **Likelihood Function** $L(\theta; \mathbf{x})$ is the joint probability density viewed as a function of the parameter $\theta$, with the observed data $\mathbf{x}$ held fixed:
  $$L(\theta; \mathbf{x}) = f(\mathbf{x}; \theta) = \prod_{i=1}^N f(x_i; \theta)$$
- Because the product of small probabilities causes severe numerical underflow on computers, we take the natural logarithm to obtain the **Log-Likelihood Function**:
  $$\ell(\theta; \mathbf{x}) = \ln L(\theta; \mathbf{x}) = \sum_{i=1}^N \ln f(x_i; \theta)$$
Since $\ln(\cdot)$ is a strictly monotonic increasing function on $(0, \infty)$, maximizing $L(\theta)$ is mathematically identical to maximizing $\ell(\theta)$.

---

### 2. The Maximum Likelihood Estimator (MLE)
The **Maximum Likelihood Estimator** $\hat{\theta}_{\text{MLE}}$ is defined as:
$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} \ell(\theta; \mathbf{x}) = \arg\min_{\theta \in \Theta} \left[ -\frac{1}{N} \sum_{i=1}^N \ln f(x_i; \theta) \right]$$

In deep learning, the term inside the minimization is defined as the **Negative Log-Likelihood (NLL) Loss**:
$$\mathcal{L}_{\text{NLL}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \ln f(x_i; \theta)$$

#### Optimization Conditions:
1. **First-Order Necessary Condition (The Likelihood Equations):**
   $$\nabla_\theta \ell(\hat{\theta}_{\text{MLE}}; \mathbf{x}) = \sum_{i=1}^N \nabla_\theta \ln f(x_i; \hat{\theta}_{\text{MLE}}) = \mathbf{0}$$
2. **Second-Order Sufficiency Condition:**
   The Hessian of the log-likelihood (the negative observed Fisher information matrix) must be strictly **negative definite**:
   $$\mathcal{H}_\ell(\hat{\theta}_{\text{MLE}}) = \nabla_\theta^2 \ell(\hat{\theta}_{\text{MLE}}; \mathbf{x}) \prec 0$$

---

### 3. Asymptotic Properties of the MLE
Under standard Cramér-Rao regularity conditions (smoothness, open parameter space, identifiable $\theta$, support independent of $\theta$):

1. **Consistency:** The MLE converges in probability to the true parameter $\theta^*$:
   $$\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta^* \quad \text{as } N \to \infty$$
2. **Asymptotic Normality:**
   $$\sqrt{N}(\hat{\theta}_{\text{MLE}} - \theta^*) \xrightarrow{d} \mathcal{N}\left(\mathbf{0}, I_1(\theta^*)^{-1}\right)$$
   where $I_1(\theta^*)$ is the single-sample Fisher Information Matrix.
3. **Asymptotic Efficiency:** As $N \to \infty$, the variance of the MLE attains the Cramér-Rao Lower Bound:
   $$\text{Cov}(\hat{\theta}_{\text{MLE}}) \to \frac{1}{N} I_1(\theta^*)^{-1} = I_N(\theta^*)^{-1}$$
   *(No other consistent estimator has lower asymptotic variance!)*
4. **Functional Invariance (Zeinna Theorem):** If $\hat{\theta}$ is the MLE of $\theta$, and $g(\theta)$ is any continuous transformation, then $g(\hat{\theta})$ is the MLE of $g(\theta)$.

#### Deep Derivation 4.3.1: Complete Proof of Asymptotic Normality and Efficiency of the MLE
Let $\hat{\theta} \triangleq \hat{\theta}_{\text{MLE}}$. By definition, $\hat{\theta}$ satisfies the score equation:
$$\nabla_\theta \ell(\hat{\theta}) = \mathbf{0}$$

1. **Mean-Value Taylor Expansion of the Score Function:**
   Expand the score $\nabla_\theta \ell(\hat{\theta})$ in a multivariate Taylor series around the true population parameter $\theta^*$:
   $$\mathbf{0} = \nabla_\theta \ell(\hat{\theta}) = \nabla_\theta \ell(\theta^*) + \nabla_\theta^2 \ell(\tilde{\theta}) (\hat{\theta} - \theta^*)$$
   where $\tilde{\theta}$ is a point on the line segment connecting $\hat{\theta}$ and $\theta^*$.

2. **Isolating the Estimation Error $\hat{\theta} - \theta^*$:**
   Multiply through by $\sqrt{N}$ and rearrange:
   $$-\frac{1}{N} \nabla_\theta^2 \ell(\tilde{\theta}) \cdot \sqrt{N}(\hat{\theta} - \theta^*) = \frac{1}{\sqrt{N}} \nabla_\theta \ell(\theta^*)$$
   Inverting the observed Hessian matrix:
   $$\sqrt{N}(\hat{\theta} - \theta^*) = \left( -\frac{1}{N} \nabla_\theta^2 \ell(\tilde{\theta}) \right)^{-1} \left( \frac{1}{\sqrt{N}} \nabla_\theta \ell(\theta^*) \right)$$

3. **Asymptotic Limit of the Numerator (via Central Limit Theorem):**
   The score is a sum of $N$ independent, identically distributed zero-mean random vectors:
   $$\frac{1}{\sqrt{N}} \nabla_\theta \ell(\theta^*) = \frac{1}{\sqrt{N}} \sum_{i=1}^N \nabla_\theta \ln f(X_i; \theta^*)$$
   Since $\mathbb{E}[\nabla_\theta \ln f(X_i; \theta^*)] = \mathbf{0}$ and $\text{Var}(\nabla_\theta \ln f(X_i; \theta^*)) = I_1(\theta^*)$, by the Lindeberg-Lévy Central Limit Theorem (Theorem 3.6.3):
   $$\frac{1}{\sqrt{N}} \nabla_\theta \ell(\theta^*) \xrightarrow{d} \mathcal{N}\left( \mathbf{0}, I_1(\theta^*) \right)$$

4. **Asymptotic Limit of the Denominator (via Weak Law of Large Numbers):**
   By consistency of the MLE ($\hat{\theta} \xrightarrow{P} \theta^*$), we have $\tilde{\theta} \xrightarrow{P} \theta^*$.
   By the Weak Law of Large Numbers and uniform continuity of the second derivative:
   $$-\frac{1}{N} \nabla_\theta^2 \ell(\tilde{\theta}) = -\frac{1}{N} \sum_{i=1}^N \nabla_\theta^2 \ln f(X_i; \tilde{\theta}) \xrightarrow{P} -\mathbb{E}\left[ \nabla_\theta^2 \ln f(X; \theta^*) \right] = I_1(\theta^*)$$

5. **Application of Slutsky's Theorem:**
   By Slutsky's Theorem, if $\mathbf{Y}_N \xrightarrow{d} \mathcal{N}(\mathbf{0}, \Sigma)$ and $A_N \xrightarrow{P} A$ (where $A$ is invertible), then $A_N^{-1} \mathbf{Y}_N \xrightarrow{d} A^{-1} \mathcal{N}(\mathbf{0}, \Sigma)$.
   Here $A = I_1(\theta^*)$ and $\Sigma = I_1(\theta^*)$:
   $$\sqrt{N}(\hat{\theta}_{\text{MLE}} - \theta^*) \xrightarrow{d} I_1(\theta^*)^{-1} \mathcal{N}\left(\mathbf{0}, I_1(\theta^*)\right) = \mathcal{N}\left(\mathbf{0}, I_1(\theta^*)^{-1} I_1(\theta^*) I_1(\theta^*)^{-1}\right)$$
   $$\mathbf{\sqrt{N}(\hat{\theta}_{\text{MLE}} - \theta^*) \xrightarrow{d} \mathcal{N}\left(\mathbf{0}, I_1(\theta^*)^{-1}\right)} \quad \blacksquare$$
   The asymptotic covariance matrix is $\frac{1}{N} I_1(\theta^*)^{-1} = I_N(\theta^*)^{-1}$, proving that the MLE attains the Cramér-Rao Lower Bound and achieves **100% asymptotic efficiency**!

#### Deep Derivation 4.3.2: First-Principles Proof of the Functional Invariance Property (Zeinna Theorem)
Let $\hat{\theta}$ be the MLE of $\theta \in \Theta$. Let $g: \Theta \to \Phi$ be an arbitrary function (not necessarily invertible).
We want to prove that the MLE of $\phi = g(\theta)$ is $\hat{\phi} = g(\hat{\theta})$.

1. **Definition of Profile Likelihood:**
   For any $\phi \in \Phi$, define the pre-image set $\Theta_\phi \triangleq \{ \theta \in \Theta \mid g(\theta) = \phi \}$.
   Define the **profile likelihood function** for $\phi$:
   $$L^*(\phi) \triangleq \sup_{\theta \in \Theta_\phi} L(\theta)$$

2. **Maximizing the Profile Likelihood:**
   Taking the supremum of $L^*(\phi)$ over all possible $\phi \in \Phi$:
   $$\sup_{\phi \in \Phi} L^*(\phi) = \sup_{\phi \in \Phi} \left( \sup_{\theta \in \Theta_\phi} L(\theta) \right) = \sup_{\theta \in \Theta} L(\theta)$$
   Since $\hat{\theta}$ maximizes $L(\theta)$ over $\Theta$:
   $$\sup_{\theta \in \Theta} L(\theta) = L(\hat{\theta})$$

3. **Evaluating at $\hat{\phi} = g(\hat{\theta})$:**
   Note that $\hat{\theta} \in \Theta_{g(\hat{\theta})}$. Therefore:
   $$L^*(g(\hat{\theta})) = \sup_{\theta \in \Theta_{g(\hat{\theta})}} L(\theta) \ge L(\hat{\theta})$$
   Since $L^*(g(\hat{\theta}))$ cannot exceed the global supremum $\sup_{\phi \in \Phi} L^*(\phi) = L(\hat{\theta})$, we must have:
   $$L^*(g(\hat{\theta})) = L(\hat{\theta}) = \sup_{\phi \in \Phi} L^*(\phi)$$
   Therefore, $\hat{\phi} = g(\hat{\theta})$ maximizes the profile likelihood $L^*(\phi)$, proving the invariance property for all transformations! $\blacksquare$

---

### 4. Mathematical Derivations of Deep Learning Loss Functions

#### Derivation 1: Gaussian Noise Model $\implies$ Mean Squared Error (MSE Loss)
Assume a regression setting where target $y \in \mathbb{R}$ is generated by a deterministic neural network $f_\theta(x)$ corrupted by additive zero-mean Gaussian noise:
$$y = f_\theta(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
The conditional probability distribution is:
$$p(y \mid x; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y - f_\theta(x))^2}{2\sigma^2} \right)$$
For an $i.i.d.$ training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, write the log-likelihood:
$$\ell(\theta, \sigma^2) = \sum_{i=1}^N \ln p(y_i \mid x_i; \theta) = -\frac{N}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f_\theta(x_i))^2$$
Take the Negative Log-Likelihood and discard constant terms independent of $\theta$:
$$\mathcal{L}_{\text{NLL}}(\theta) \propto \frac{1}{N} \sum_{i=1}^N (y_i - f_\theta(x_i))^2 = \mathbf{\mathcal{L}_{\text{MSE}}(\theta)} \quad \blacksquare$$
*Conclusion:* Minimizing Mean Squared Error is **identical to Maximum Likelihood under the assumption of homoscedastic Gaussian noise**!

---

#### Derivation 2: Bernoulli Model $\implies$ Binary Cross-Entropy (BCE Loss)
Assume a binary classification setting where label $y \in \{0, 1\}$ follows a Bernoulli distribution conditioned on input $x$:
$$y \mid x \sim \text{Bernoulli}(p = \hat{y}_\theta(x)), \quad \hat{y}_\theta(x) = \sigma(f_\theta(x)) = \frac{1}{1 + e^{-f_\theta(x)}}$$
The Probability Mass Function is:
$$P(y \mid x; \theta) = \hat{y}^y (1 - \hat{y})^{1 - y}$$
Log-likelihood for sample $i$:
$$\ln P(y_i \mid x_i; \theta) = y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i)$$
Taking the average negative log-likelihood:
$$\mathbf{\mathcal{L}_{\text{BCE}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \right]} \quad \blacksquare$$

---

#### Derivation 3: Categorical / Multinomial Model $\implies$ Categorical Cross-Entropy
Assume a multi-class setting with $K$ mutually exclusive classes.
The target vector $\mathbf{y} \in \{0, 1\}^K$ is one-hot ($\sum_{k=1}^K y_k = 1$).
The network outputs class probabilities via the Softmax activation:
$$\hat{y}_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}, \quad \mathbf{z} = f_\theta(x)$$
The Categorical likelihood is:
$$P(\mathbf{y} \mid x; \theta) = \prod_{k=1}^K \hat{y}_k^{y_k}$$
Log-likelihood for sample $i$:
$$\ln P(\mathbf{y}_i \mid x_i; \theta) = \sum_{k=1}^K y_{ik} \ln \hat{y}_{ik}$$
Taking the average negative log-likelihood:
$$\mathbf{\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \ln \hat{y}_{ik}} \quad \blacksquare$$

---

#### Derivation 4: Laplace Noise Model $\implies$ Mean Absolute Error (MAE / $L_1$ Loss)
Suppose the observation noise follows a **Laplace distribution** (heavy-tailed, robust to outliers):
$$\epsilon \sim \text{Laplace}(0, b) \implies p(y \mid x; \theta) = \frac{1}{2b} \exp\left( -\frac{|y - f_\theta(x)|}{b} \right)$$
Log-likelihood:
$$\ell(\theta) = -N \ln(2b) - \frac{1}{b} \sum_{i=1}^N |y_i - f_\theta(x_i)|$$
Minimizing NLL w.r.t. $\theta$ yields:
$$\mathbf{\mathcal{L}_{\text{MAE}}(\theta) = \frac{1}{N} \sum_{i=1}^N |y_i - f_\theta(x_i)|} \quad \blacksquare$$

---

## Part 3: Geometric & Algebraic Interpretation

### 1. Likelihood as a Potential Energy Landscape
On the parameter manifold $\Theta$, the Negative Log-Likelihood $\mathcal{L}(\theta) = -\ell(\theta)$ forms a potential well:
- The bottom of the well is $\hat{\theta}_{\text{MLE}}$.
- The curvature at the minimum is governed by the **Observed Fisher Information Matrix**:
  $$J(\hat{\theta}) = -\nabla_\theta^2 \ell(\hat{\theta}) = N \cdot \hat{I}_N(\hat{\theta})$$
- By the asymptotic normality theorem, the level sets of the log-likelihood approximate ellipsoids:
  $$\ell(\theta) \approx \ell(\hat{\theta}) - \frac{1}{2} (\theta - \hat{\theta})^T J(\hat{\theta}) (\theta - \hat{\theta})$$
  The eigenvectors of $J(\hat{\theta})$ determine the principal axes of parameter uncertainty!

```
     Negative Log-Likelihood Potential Well
     L(θ) ▲
          │      \               /
          │       \             /
          │        \     *     /  High Curvature = Sharp Well (High Information)
          │         \  θ_MLE  /
          └──────────┴───────┴────────► Parameter θ
```

---

### 2. MLE as Empirical KL Divergence Minimization
Let $P_{\text{data}}$ be the true empirical data distribution:
$$P_{\text{data}}(x) = \frac{1}{N} \sum_{i=1}^N \delta(x - x_i)$$
The KL divergence between true data and parameterized model $P_\theta$ is:
$$D_{\text{KL}}(P_{\text{data}} \parallel P_\theta) = \mathbb{E}_{x \sim P_{\text{data}}}\left[ \ln \frac{P_{\text{data}}(x)}{p_\theta(x)} \right] = \underbrace{\mathbb{E}_{x \sim P_{\text{data}}}[\ln P_{\text{data}}(x)]}_{-H(P_{\text{data}}) \text{ (Constant)}} - \underbrace{\frac{1}{N} \sum_{i=1}^N \ln p_\theta(x_i)}_{\text{Average Log-Likelihood}}$$
Therefore:
$$\mathbf{\arg\max_\theta \ell(\theta; \mathbf{x}) \equiv \arg\min_\theta D_{\text{KL}}(P_{\text{data}} \parallel P_\theta)}$$
**Maximum Likelihood Estimation is the projection of the empirical data distribution onto the model family under the KL divergence!**

---

## Part 4: Real-World Analogy

### The Radio Tuner Analogy
Imagine you are sitting in a car late at night, searching for an unknown radio broadcast.
- The dial represents the unknown parameter $\theta \in \mathbb{R}$ (frequency).
- When the dial is set far away from the station ($\theta = 90.1$ MHz), you hear pure static ($L(\theta) \approx 0$).
- As you turn the knob toward $\theta = 101.5$ MHz, the music begins to emerge through the noise ($L(\theta)$ rises).
- Exactly at $\theta = 101.5$ MHz, the reception is crisp, crystal clear, and maximum volume ($L(\theta)$ reaches its global maximum).
- **MLE is simply the rule that tells the engineer: stop turning the knob where the signal is loudest!**

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us solve a complete **Linear Regression MLE problem** by hand with concrete numbers, comparing likelihood values across candidate parameters and deriving the exact closed-form MLE.

### 1. Problem Setup & Toy Dataset
Consider a 1-parameter linear regression model passing through the origin:
$$\hat{y} = w x$$
with Gaussian observation noise $\epsilon \sim \mathcal{N}(0, \sigma^2 = 1.0)$:
$$p(y \mid x; w) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(y - w x)^2}{2} \right)$$

We observe $N = 3$ data points:
- Point 1: $(x_1, y_1) = (1.0, 1.6)$
- Point 2: $(x_2, y_2) = (2.0, 4.2)$
- Point 3: $(x_3, y_3) = (3.0, 6.0)$

Let us compare four candidate weights:
- $w_A = 1.0$ (Severe under-estimate)
- $w_B = 1.5$ (Moderate under-estimate)
- $w_C = 2.0$ (Candidate optimal)
- $w_D = 2.5$ (Over-estimate)

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $N$ | Number of samples | Batch size | Integer scalar | $3$ |
| $x_i, y_i$ | Input feature and target label | Training feature and target | Float scalars | $(1, 1.6), (2, 4.2), (3, 6.0)$ |
| $w$ | Model parameter to estimate | Linear layer weight | Float scalar | Evaluated at $1.0, 1.5, 2.0, 2.5$ |
| $\hat{y}_i(w)$ | Predicted value $w x_i$ | Forward pass activation | Float scalar | $w \cdot x_i$ |
| $e_i(w)$ | Residual $y_i - \hat{y}_i$ | Prediction error | Float scalar | $y_i - w x_i$ |
| $e_i^2$ | Squared residual | Quadratic loss term | Float scalar | $(y_i - w x_i)^2$ |
| $\sum e_i^2$ | Sum of squared errors ($SSE$) | Total Mean Squared Error numerator | Float scalar | Calculated per candidate $w$ |
| $\ell(w)$ | Total Log-Likelihood | Objective to maximize | Float scalar | $-\frac{3}{2}\ln(2\pi) - \frac{1}{2}\sum e_i^2$ |
| $w_{\text{MLE}}$ | Analytical MLE weight | Optimal trained weight | Float scalar | $2.0000$ exactly |

---

### 3. Step-by-Step Manual Arithmetic

#### Step 1: Compute Data Sufficient Statistics
$$\sum_{i=1}^3 x_i^2 = 1.0^2 + 2.0^2 + 3.0^2 = 1.0 + 4.0 + 9.0 = \mathbf{14.0000}$$
$$\sum_{i=1}^3 x_i y_i = (1.0)(1.6) + (2.0)(4.2) + (3.0)(6.0) = 1.60 + 8.40 + 18.00 = \mathbf{28.0000}$$

---

#### Step 2: Analytical Solution via Score Equation
Write the log-likelihood as a function of $w$:
$$\ell(w) = -\frac{3}{2}\ln(2\pi) - \frac{1}{2} \sum_{i=1}^3 (y_i - w x_i)^2$$
Compute the Score function (gradient w.r.t. $w$):
$$S(w) = \frac{d\ell}{dw} = \sum_{i=1}^3 (y_i - w x_i) x_i = \sum_{i=1}^3 x_i y_i - w \sum_{i=1}^3 x_i^2$$
Set the Score to zero:
$$28.0000 - w (14.0000) = 0 \implies \mathbf{w_{\text{MLE}} = \frac{28.0000}{14.0000} = 2.0000} \quad \checkmark$$

Check Second Derivative (Concavity):
$$\frac{d^2\ell}{dw^2} = -\sum_{i=1}^3 x_i^2 = -14.0000 < 0 \quad (\text{Strict global maximum!})$$

---

#### Step 3: Cell-by-Cell Arithmetic for All 4 Candidates

**Candidate A ($w = 1.0$):**
- $\hat{y} = [1.0, 2.0, 3.0]$
- $e = [1.6 - 1.0, \, 4.2 - 2.0, \, 6.0 - 3.0] = [0.60, \, 2.20, \, 3.00]$
- $e^2 = [0.36, \, 4.84, \, 9.00] \implies \sum e^2 = \mathbf{14.2000}$
- $\ell(w_A) = -\frac{3}{2}\ln(2\pi) - 0.5(14.20) = -2.7568 - 7.1000 = \mathbf{-9.8568}$

**Candidate B ($w = 1.5$):**
- $\hat{y} = [1.5, 3.0, 4.5]$
- $e = [1.6 - 1.5, \, 4.2 - 3.0, \, 6.0 - 4.5] = [0.10, \, 1.20, \, 1.50]$
- $e^2 = [0.01, \, 1.44, \, 2.25] \implies \sum e^2 = \mathbf{3.7000}$
- $\ell(w_B) = -2.7568 - 0.5(3.70) = -2.7568 - 1.8500 = \mathbf{-4.6068}$

**Candidate C ($w = 2.0$ — The MLE!):**
- $\hat{y} = [2.0, 4.0, 6.0]$
- $e = [1.6 - 2.0, \, 4.2 - 4.0, \, 6.0 - 6.0] = [-0.40, \, +0.20, \, 0.00]$
- $e^2 = [0.16, \, 0.04, \, 0.00] \implies \sum e^2 = \mathbf{0.2000}$
- $\ell(w_C) = -2.7568 - 0.5(0.20) = -2.7568 - 0.1000 = \mathbf{-2.8568} \quad (\mathbf{Maximum!})$

**Candidate D ($w = 2.5$):**
- $\hat{y} = [2.5, 5.0, 7.5]$
- $e = [1.6 - 2.5, \, 4.2 - 5.0, \, 6.0 - 7.5] = [-0.90, \, -0.80, \, -1.50]$
- $e^2 = [0.81, \, 0.64, \, 2.25] \implies \sum e^2 = \mathbf{3.7000}$
- $\ell(w_D) = -2.7568 - 0.5(3.70) = \mathbf{-4.6068}$

---

### 4. Visual Summary Grid

```
┌───────────┬─────────────┬─────────────┬─────────────┬─────────────┬──────────────┬──────────────┐
│ Candidate │ e_1 (x=1)   │ e_2 (x=2)   │ e_3 (x=3)   │ Σ e_i²(SSE) │ Log-Lik ℓ(w) │ Rank / Status│
├───────────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────────┼──────────────┤
│ w = 1.0   │   +0.60     │   +2.20     │   +3.00     │   14.2000   │   -9.8568    │ 4th (Worst)  │
│ w = 1.5   │   +0.10     │   +1.20     │   +1.50     │    3.7000   │   -4.6068    │ 2nd (Tied)   │
│ w = 2.0   │   -0.40     │   +0.20     │    0.00     │    0.2000   │   -2.8568    │ 1st (MLE! 🎯)│
│ w = 2.5   │   -0.90     │   -0.80     │   -1.50     │    3.7000   │   -4.6068    │ 2nd (Tied)   │
└───────────┴─────────────┴─────────────┴─────────────┴─────────────┴──────────────┴──────────────┘
```

Notice the beautiful parabolic symmetry: $w = 1.5$ and $w = 2.5$ are both distance $\Delta = 0.5$ away from $w^* = 2.0$, and both have the exact same log-likelihood $\ell = -4.6068$!

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Joint MLE of Gaussian Mean $\mu$ and Variance $\sigma^2$
Let $X_1, \dots, X_N \overset{i.i.d.}{\sim} \mathcal{N}(\mu, \sigma^2)$ where *both* parameters are unknown.
$$\ell(\mu, \sigma^2) = -\frac{N}{2} \ln(2\pi) - \frac{N}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (X_i - \mu)^2$$

1. **Derivative w.r.t. $\mu$:**
   $$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^N (X_i - \mu) = 0 \implies \mathbf{\hat{\mu}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N X_i = \bar{X}}$$
2. **Derivative w.r.t. $\sigma^2$ (treat $\theta = \sigma^2$ as a single variable):**
   $$\frac{\partial \ell}{\partial \sigma^2} = -\frac{N}{2\sigma^2} + \frac{1}{2(\sigma^2)^2} \sum_{i=1}^N (X_i - \hat{\mu})^2 = 0$$
   Multiply by $2(\sigma^2)^2 / N$:
   $$\mathbf{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N (X_i - \bar{X})^2}$$
*Key Insight:* The MLE variance is the **biased** sample variance ($S_n^2$), NOT the Bessel-corrected unbiased variance ($S_{N-1}^2$)!
This proves that **the MLE is not guaranteed to be unbiased in finite samples**, although it is asymptotically unbiased ($\frac{N-1}{N} \to 1$).

---

### Illustration 2 (Boundary): Complete Separation in Logistic Regression (Weight Explosion)
Suppose you train a logistic regression model on binary classification data that happens to be **linearly separable** (there exists a hyperplane separating all positive and negative samples perfectly).
- For every sample $i$, $y_i(w^T x_i) > 0$.
- The likelihood is:
  $$L(w) = \prod_{i=1}^N \sigma(y_i w^T x_i)$$
- As we scale the weights by a constant $c \to \infty$:
  $$\lim_{c \to \infty} \sigma(c \cdot y_i w^T x_i) = 1.0 \implies L(c \cdot w) \to 1.0, \quad \mathcal{L}_{\text{BCE}} \to 0.0$$
- **The MLE does not exist in finite parameter space!** The optimal weights are $\|w\| \to \infty$!
*Deep Learning Meaning:* In overparameterized neural networks, unregularized cross-entropy loss causes the norm of the final classification layer weights to explode toward infinity to push softmax probabilities to exactly 1.0. This is why **weight decay ($L_2$)** or **label smoothing** is strictly necessary!

---

### Illustration 3 (Edge Case): Heteroscedastic Uncertainty Modeling in Deep Networks
Standard regression assumes homoscedastic (constant) noise $\sigma^2$. But in autonomous driving or medical diagnosis, certain inputs have much higher uncertainty than others (e.g., driving in dense fog vs. clear daylight).

We modify the neural network to output **two values per sample**:
$$f_\theta(x) = [\hat{\mu}(x), \,\, \hat{s}(x)]$$
where $\hat{s}(x) = \ln \hat{\sigma}^2(x)$ (log-variance, guaranteed positive when exponentiated).
The Gaussian NLL loss becomes:
$$\mathbf{\mathcal{L}_{\text{Hetero}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{(y_i - \hat{\mu}(x_i))^2}{2 \exp(\hat{s}(x_i))} + \frac{1}{2} \hat{s}(x_i) \right]}$$
- The first term penalizes prediction error, but scales it by the model's self-reported uncertainty ($\exp(\hat{s})$).
- The second term acts as a **regularizer** preventing the model from cheating by setting $\hat{s} \to \infty$ everywhere!

---

### Illustration 4 (Numerical): Multi-Class Softmax Cross-Entropy Loss & Gradient Step by Hand
In a multi-class image classification model ($K = 3$ classes: `[Cat, Dog, Bird]`), an input sample has ground-truth label $y = [0, 1, 0]^T$ (Class 2: Dog).
The unnormalized output logits from the final linear layer are:
$$\mathbf{z} = [z_1, z_2, z_3]^T = [2.000000, 1.000000, 0.100000]^T$$

1. **Step 1: Compute Softmax Probabilities:**
   $$e^{z_1} = e^{2.0} \approx \mathbf{7.389056}$$
   $$e^{z_2} = e^{1.0} \approx \mathbf{2.718282}$$
   $$e^{z_3} = e^{0.1} \approx \mathbf{1.105171}$$
   Denominator sum:
   $$\sum_{j=1}^3 e^{z_j} = 7.389056 + 2.718282 + 1.105171 = \mathbf{11.212509}$$
   Compute class probabilities $\hat{y}_k = \frac{e^{z_k}}{\sum e^{z_j}}$:
   $$\hat{y}_1 = \frac{7.389056}{11.212509} \approx \mathbf{0.658998 \quad (65.90\%)}$$
   $$\hat{y}_2 = \frac{2.718282}{11.212509} \approx \mathbf{0.242433 \quad (24.24\%)}$$
   $$\hat{y}_3 = \frac{1.105171}{11.212509} \approx \mathbf{0.098569 \quad (9.86\%)}$$
   *Check:* $0.658998 + 0.242433 + 0.098569 = 1.000000 \quad \checkmark$

2. **Step 2: Compute Categorical Negative Log-Likelihood (Cross-Entropy Loss):**
   $$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^3 y_k \ln \hat{y}_k = -\ln(\hat{y}_2) = -\ln(0.242433) \approx \mathbf{1.417036\text{ nats}}$$

3. **Step 3: Analytical Gradient Vector w.r.t. Logits $\mathbf{z}$:**
   The gradient of the softmax cross-entropy loss with respect to logits is the exact prediction residual $\nabla_{\mathbf{z}} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$:
   $$\nabla_{\mathbf{z}} \mathcal{L} = \begin{bmatrix} 0.658998 - 0 \\ 0.242433 - 1 \\ 0.098569 - 0 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.658998} \\ \mathbf{-0.757567} \\ \mathbf{+0.098569} \end{bmatrix}$$
   *Notice:* The correct class $y_2$ receives a strong negative gradient $-0.758$ (pushing logit $z_2$ upward), while the false classes receive positive gradients (pushing logits $z_1$ and $z_3$ downward)!

4. **Step 4: Execute a Single Gradient Descent Step ($\eta = 0.50$):**
   $$\mathbf{z}_{\text{new}} = \mathbf{z} - \eta \nabla_{\mathbf{z}} \mathcal{L} = \begin{bmatrix} 2.000000 - 0.5(0.658998) \\ 1.000000 - 0.5(-0.757567) \\ 0.100000 - 0.5(0.098569) \end{bmatrix} = \begin{bmatrix} 2.000000 - 0.329499 \\ 1.000000 + 0.378784 \\ 0.100000 - 0.049285 \end{bmatrix} = \begin{bmatrix} \mathbf{1.670501} \\ \mathbf{1.378784} \\ \mathbf{0.050715} \end{bmatrix}$$
   New exponentials: $e^{1.6705} \approx 5.3148, e^{1.3788} \approx 3.9699, e^{0.0507} \approx 1.0520$. Sum $= 10.3367$.
   New Dog probability: $\hat{y}_{2, \text{new}} = \frac{3.9699}{10.3367} \approx \mathbf{0.384059}$ (up from $24.2\%$).
   New loss: $\mathcal{L}_{\text{new}} = -\ln(0.384059) \approx \mathbf{0.956960\text{ nats}}$ (dropped by $32.5\%$ in a single step!).

---

### Illustration 5 (Numerical): Exponential Distribution MLE for GPU Server Reliability
In a cloud cluster, GPU hardware node failure times $X$ (in days) follow an exponential distribution:
$$f(x; \lambda) = \lambda e^{-\lambda x}, \quad x \ge 0$$
Across $N = 5$ server racks, recorded failure times are:
$$\mathbf{x} = [12.0, 24.0, 6.0, 18.0, 40.0]\text{ days}$$

1. **Step 1: Formulate Likelihood and Score Equation:**
   $$L(\lambda) = \prod_{i=1}^5 \lambda e^{-\lambda x_i} = \lambda^5 \exp\left( -\lambda \sum_{i=1}^5 x_i \right)$$
   $$\ell(\lambda) = 5 \ln \lambda - \lambda \sum_{i=1}^5 x_i$$
   $$\sum_{i=1}^5 x_i = 12.0 + 24.0 + 6.0 + 18.0 + 40.0 = \mathbf{100.000000\text{ days}}$$
   Differentiating w.r.t. $\lambda$:
   $$\frac{d\ell}{d\lambda} = \frac{5}{\lambda} - 100.0 = 0 \implies \mathbf{\hat{\lambda}_{\text{MLE}} = \frac{5}{100.0} = 0.050000\text{ failures/day}}$$

2. **Step 2: Mean Time Between Failures (MTBF) via Functional Invariance:**
   The parameter of interest to cloud operators is the Mean Time Between Failures $\theta = g(\lambda) = \frac{1}{\lambda}$.
   By the Zeinna Invariance Theorem (Section 4.3.3):
   $$\mathbf{\hat{\theta}_{\text{MLE}} = g(\hat{\lambda}_{\text{MLE}}) = \frac{1}{0.050000} = 20.000000\text{ days}}$$

3. **Step 3: Fisher Information & Asymptotic Confidence Interval:**
   $$\frac{d^2\ell}{d\lambda^2} = -\frac{5}{\lambda^2} \implies I_5(\lambda) = \frac{5}{\lambda^2} = \frac{5}{(0.05)^2} = \frac{5}{0.0025} = \mathbf{2000.000000}$$
   $$\text{CRLB} = \frac{1}{I_5(\lambda)} = \frac{1}{2000.0} = \mathbf{0.000500}$$
   $$\text{SE}(\hat{\lambda}) = \sqrt{0.000500} \approx \mathbf{0.022361}$$
   Exact asymptotic 95% Confidence Interval for failure rate $\lambda$:
   $$\hat{\lambda} \pm 1.96 \, \text{SE} = 0.050000 \pm 1.96(0.022361) = 0.050000 \pm 0.043828 = [\mathbf{0.006172, 0.093828}]$$

---

## Part 7: Deep Learning Connection & Application

### 1. Unification of Deep Learning Architectures via MLE

| Deep Learning Domain | Output Noise Model | Output Activation | Loss Function |
| :--- | :--- | :--- | :--- |
| **Standard Regression** | Gaussian $\mathcal{N}(\mu, \sigma^2 I)$ | Linear (None) | Mean Squared Error (MSE) |
| **Robust Regression** | Laplace $(0, b)$ | Linear (None) | Mean Absolute Error (MAE / $L_1$) |
| **Binary Classification** | Bernoulli $(p)$ | Sigmoid $\sigma(z)$ | Binary Cross-Entropy (BCE) |
| **Multi-Class Classification** | Categorical $(\pi_1, \dots, \pi_K)$ | Softmax | Categorical Cross-Entropy |
| **Count / Frequency Modeling** | Poisson $(\lambda)$ | Exponential $\exp(z)$ | Poisson NLL Loss |
| **Autoregressive LLMs** | Categorical over Vocabulary | Softmax | Token-level Cross-Entropy |

Every single state-of-the-art Large Language Model (GPT-4, LLaMA, Gemini) is trained under the exact MLE objective:
$$\mathcal{L}_{\text{LLM}}(\theta) = -\sum_{t=1}^T \ln P(w_t \mid w_1, \dots, w_{t-1}; \theta)$$
Pre-training an LLM is literally maximum likelihood estimation applied to billions of sequential categorical draws.

---

## Part 8: Code Implementation & Verification

The companion Python module [03_maximum_likelihood_estimation.py](./code/03_maximum_likelihood_estimation.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact calculations of $\sum e_i^2$, $\ell(w)$, and confirms $w_{\text{MLE}} = 2.0000$ analytically and via numerical optimization.
2. **Joint Gaussian MLE Verification:** Proves empirically that $\hat{\sigma}^2_{\text{MLE}}$ exhibits exact $\frac{N-1}{N}$ downward bias across 100,000 trials.
3. **Bernoulli MLE & Cross-Entropy Equivalence:** Verifies that minimizing PyTorch `BCELoss` matches the closed-form MLE frequency $\hat{p} = \frac{1}{N}\sum y_i$.
4. **Separable Logistic Regression Weight Explosion:** Simulates training on linearly separable data to demonstrate gradient descent weights diverging ($\|w\| \to \infty$) unless $L_2$ regularization is added.
5. **Heteroscedastic Aleatoric Loss PyTorch Module:** Implements a toy neural network predicting both mean and variance, trained using the heteroscedastic Gaussian NLL loss.
