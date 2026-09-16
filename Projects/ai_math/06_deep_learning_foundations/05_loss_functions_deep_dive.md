# 6.5 Loss Functions Deep-Dive (MSE, BCE, Cross-Entropy, Focal, Triplet Loss)

---

## Part 1: Intuition & 101 Motivation

An optimization algorithm (SGD, AdamW) possesses no inherent common sense. It is a blind mathematical engine that relentlessly follows a single command: **minimize the scalar loss function $\mathcal{L}(\theta)$**.

The loss function is the ultimate mathematical specification of what our model *should care about*. If the loss function is misaligned with the real-world problem:
- It can cause the model to learn degenerate, trivial shortcuts (e.g., predicting the majority class 100% of the time).
- It can cause gradients to saturate and vanish to zero when the model is most wrong.
- It can cause extreme sensitivity to rare, corrupted outliers.

Loss functions in deep learning fall into three overarching paradigms:
1. **Regression Losses (Continuous Targets)**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Huber Loss. Grounded in Maximum Likelihood Estimation under different environmental noise distributions (Gaussian vs. Laplace).
2. **Classification Losses (Discrete Probability Distributions)**: Binary Cross-Entropy (BCE) and Categorical Cross-Entropy (CCE). Grounded in Information Theory and Kullback-Leibler (KL) divergence.
3. **Advanced & Modern Objective Functions**:
   - **Focal Loss (Lin et al. 2017)**: Solves extreme class imbalance (e.g., 1 object per 100,000 background tiles in object detection) by dynamically adding a modulating factor $(1 - p_t)^\gamma$ that suppresses the gradient from easy, well-classified examples.
   - **Metric Learning / Triplet Loss (Schroff et al. 2015)**: Instead of predicting discrete classes, shapes high-dimensional embedding spaces so that semantically similar items (anchor and positive) are clustered together while dissimilar items (negative) are pushed apart beyond a geometric margin.

---

## Part 2: Rigorous Mathematical Formulation

```
                     PARADIGMS OF LOSS FUNCTIONS IN DEEP LEARNING
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ Regression (Continuous) │ Classification (Prob)   │ Metric & Representation │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • MSE (Gaussian Noise)  │ • BCE (Bernoulli MLE)   │ • Triplet Loss (Margin) │
│ • MAE (Laplace Noise)   │ • CCE (Multinomial MLE) │ • Contrastive / InfoNCE │
│ • Huber (Smooth L1)     │ • Focal Loss (Imbalance)│ • Cosine / ArcFace      │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

### 1. Regression Losses: MSE, MAE, and Huber Loss

Let $y \in \mathbb{R}$ be the ground truth target, and $\hat{y} \in \mathbb{R}$ be the model prediction. Let residual error be $r = \hat{y} - y$.

#### Mean Squared Error (MSE / $L_2$ Loss)
$$\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2} r^2$$
- Gradient: $\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}} = \hat{y} - y = r$.
- **Probabilistic Origin**: In Chapter 4.3, we proved that minimizing MSE is strictly mathematically identical to **Maximum Likelihood Estimation under additive Gaussian noise**:
  $$y = f(x; \theta) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
- **Vulnerability**: Because the penalty scales quadratically ($r^2$), large outliers exert catastrophic leverage on the gradient ($\nabla \sim r$), often pulling the regression line far away from clean data.

#### Mean Absolute Error (MAE / $L_1$ Loss)
$$\mathcal{L}_{\text{MAE}}(\hat{y}, y) = |\hat{y} - y| = |r|$$
- Gradient: $\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}} = \text{sign}(\hat{y} - y) = \text{sign}(r)$.
- **Probabilistic Origin**: Equivalent to MLE under heavy-tailed **Laplace noise**: $p(\epsilon) = \frac{1}{2b} e^{-|\epsilon|/b}$.
- **Advantage & Flaw**: Robust to outliers because gradient magnitude is constantly $1.0$, regardless of error size. However, the derivative is discontinuous at $r = 0$, causing optimization to bounce erratically near the minimum.

#### Huber Loss (Smooth $L_1$ Loss)
Huber loss bridges MSE and MAE by applying a quadratic penalty for small errors ($|r| \le \delta$) and a linear penalty for large errors ($|r| > \delta$):
$$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta \left( |r| - \frac{1}{2} \delta \right) & \text{if } |r| > \delta \end{cases}$$
- Gradient:
  $$\frac{\partial \mathcal{L}_\delta}{\partial r} = \begin{cases} r & \text{if } |r| \le \delta \\ \delta \cdot \text{sign}(r) & \text{if } |r| > \delta \end{cases}$$
- Continuously differentiable ($C^1$) everywhere, combining quadratic smoothness at the origin with linear robustness to extreme outliers.

---

### 2. Binary Cross-Entropy (BCE)

For binary classification with target label $y \in \{0, 1\}$ and predicted probability $p = \sigma(z) \in (0, 1)$:
$$\mathcal{L}_{\text{BCE}}(p, y) = -\left[ y \log(p) + (1 - y) \log(1 - p) \right]$$

#### Probabilistic Derivation from Bernoulli Likelihood
Assume $Y \sim \text{Bernoulli}(p)$ where $P(Y=1) = p$ and $P(Y=0) = 1-p$.
The likelihood of a single observation is:
$$P(Y=y | p) = p^y (1 - p)^{1 - y}$$
Taking the negative log-likelihood directly yields BCE:
$$-\log P(Y=y | p) = -\log\left( p^y (1 - p)^{1 - y} \right) = -\left[ y \log p + (1 - y) \log(1 - p) \right] = \mathcal{L}_{\text{BCE}}$$

#### Numerically Stable BCE with Logits
Evaluating $\log(\sigma(z))$ naively in floating point arithmetic causes catastrophic underflow when $z < -50$.
Expressing BCE directly in terms of raw logit $z$:
$$p = \frac{1}{1 + e^{-z}}, \quad 1 - p = \frac{e^{-z}}{1 + e^{-z}} = \frac{1}{1 + e^z}$$
$$\mathcal{L}_{\text{BCE}}(z, y) = -y \log\left( \frac{1}{1 + e^{-z}} \right) - (1 - y) \log\left( \frac{e^{-z}}{1 + e^{-z}} \right)$$
$$= y \log(1 + e^{-z}) + (1 - y) \left[ z + \log(1 + e^{-z}) \right] = (1 - y)z + \log(1 + e^{-z})$$
For numerical stability across both positive and negative $z$:
$$\mathbf{\mathcal{L}_{\text{BCE}}(z, y) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)}$$

- **Gradient with Respect to Logit $z$**:
  $$\mathbf{\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \sigma(z) - y = p - y}$$

---

### 3. Categorical Cross-Entropy (CCE)

For multi-class classification with $K$ classes, one-hot ground-truth target vector $y \in \{0, 1\}^K$ ($\sum_{k=1}^K y_k = 1$), and predicted class probability vector $S = \text{softmax}(z) \in \Delta^{K-1}$:
$$\mathcal{L}_{\text{CCE}}(S, y) = -\sum_{k=1}^K y_k \log S_k$$
Since $y$ is one-hot with true class index $c$ ($y_c = 1, y_{k \ne c} = 0$):
$$\mathcal{L}_{\text{CCE}} = -\log S_c = -\log\left( \frac{e^{z_c}}{\sum_{j=1}^K e^{z_j}} \right) = -z_c + \log\left( \sum_{j=1}^K e^{z_j} \right)$$

- **Gradient with Respect to Logit $z_k$**:
  $$\mathbf{\frac{\partial \mathcal{L}_{\text{CCE}}}{\partial z_k} = S_k - y_k}$$
  In vector form: $\nabla_z \mathcal{L} = S - y$.

---

### 4. Focal Loss (Lin, Goyal, Girshick, He, Dollár 2017)

In dense object detection (e.g., RetinaNet), a detector evaluates $\sim 100{,}000$ candidate bounding box locations per image. Only $\sim 10$ contain actual objects (foreground); the remaining $99{,}990$ are easy background tiles.
Even though each easy background tile produces a tiny loss ($\sim 0.001$), summing over $100{,}000$ tiles yields a cumulative gradient that **completely overwhelms the foreground gradient**, causing training to fail.

To resolve this, Lin et al. introduced **Focal Loss**:
Define $p_t$ as the model's estimated probability for the correct class:
$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$
The standard Cross-Entropy loss is simply $\mathcal{L}_{\text{CE}} = -\log(p_t)$.

**Focal Loss introduces a dynamic modulating factor $(1 - p_t)^\gamma$**:
$$\mathbf{\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)}$$
where:
- $\gamma \ge 0$ is the **focusing parameter** (typically $\gamma = 2.0$).
- $\alpha_t \in [0, 1]$ is an optional weighting factor balancing class prevalence.

#### The Mathematical Filtering Mechanism
- **When an example is easy / well-classified ($p_t = 0.99$)**:
  The modulating factor is:
  $$(1 - p_t)^\gamma = (1 - 0.99)^2 = 0.01^2 = \mathbf{0.0001}$$
  The loss and gradient for this example are **suppressed by $10{,}000\times$**!
- **When an example is hard / misclassified ($p_t = 0.10$)**:
  The modulating factor is:
  $$(1 - p_t)^\gamma = (1 - 0.10)^2 = 0.90^2 = \mathbf{0.81}$$
  The loss and gradient are virtually unaffected ($81\%$ retained)!

```
Loss Magnitude vs. Probability of Ground Truth Class p_t:
  Loss
   5 +
     | \      Cross-Entropy (\gamma = 0)
   4 +  \
     |   \
   3 +    \
     |     \
   2 +      \        Focal Loss (\gamma = 2):
     |       \       Suppresses easy examples (p_t > 0.5) to near-zero!
   1 +        \ _ _ _ _
   0 +---------+-------+-----------------------------> p_t
    0.0       0.2     0.4     0.6     0.8     1.0
```

#### Analytical Gradient of Focal Loss with Respect to Logit $z$
For binary classification with $y \in \{0, 1\}$, $p = \sigma(z)$, and $\alpha = 1$:
$$\mathcal{L}_{\text{FL}} = -y (1 - p)^\gamma \log(p) - (1 - y) p^\gamma \log(1 - p)$$

Differentiating with respect to logit $z$ (using $\frac{\partial p_t}{\partial z} = (2y - 1) p_t (1 - p_t)$):
$$\mathbf{\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z} = (2y - 1) \left[ \gamma p_t (1 - p_t)^\gamma \log(p_t) - (1 - p_t)^{\gamma + 1} \right]}$$

---

### 5. Triplet Loss (Schroff, Kalenichenko, Philbin 2015)

Used for **deep metric learning, face recognition (FaceNet), and vector embeddings**.

Instead of classifying an image into fixed categories, a neural network $f_\theta: \mathcal{X} \to \mathbb{R}^d$ maps inputs onto a unit hypersphere ($\|f(x)\|_2 = 1$).
We construct a **triplet** consisting of:
1. **Anchor ($a$)**: A reference image (e.g., Face A, photo 1).
2. **Positive ($p$)**: Another image of the same identity (Face A, photo 2).
3. **Negative ($n$)**: An image of a completely different person (Face B).

We desire the distance between anchor and positive to be smaller than the distance between anchor and negative by at least a margin $\alpha > 0$:
$$\|f(a) - f(p)\|_2^2 + \alpha < \|f(a) - f(n)\|_2^2$$

#### The Triplet Loss Formula:
$$\mathbf{\mathcal{L}_{\text{triplet}}(a, p, n) = \max\left(0, \; \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha\right)}$$

```
Geometric Triplet Configuration:
                        Positive p
                         (Face A)
                          /
                         /  d(a, p)^2
                        /
                       v
        Anchor a <---------- Margin alpha ----------> Negative n
        (Face A)                                      (Face B)
```

#### Analytical Gradients of Triplet Loss
When the loss is active ($\mathcal{L} > 0$):
$$\frac{\partial \mathcal{L}}{\partial f(a)} = 2 (f(n) - f(p))$$
$$\frac{\partial \mathcal{L}}{\partial f(p)} = 2 (f(p) - f(a)) = -2 (f(a) - f(p))$$
$$\frac{\partial \mathcal{L}}{\partial f(n)} = 2 (f(a) - f(n))$$
- The gradient on $f(p)$ pulls the positive directly toward the anchor.
- The gradient on $f(n)$ pushes the negative directly away from the anchor.

#### Triplet Mining Strategies
1. **Easy Triplets**: $\|f(a) - f(p)\|^2 + \alpha < \|f(a) - f(n)\|^2 \implies \mathcal{L} = 0$. Provides zero gradient signal.
2. **Hard Triplets**: $\|f(a) - f(n)\|^2 < \|f(a) - f(p)\|^2$. Negative is closer to anchor than positive. Produces large gradients, but can destabilize early training.
3. **Semi-Hard Triplets**: $\|f(a) - f(p)\|^2 < \|f(a) - f(n)\|^2 < \|f(a) - f(p)\|^2 + \alpha$. Negative is farther than positive, but still inside the margin $\alpha$. This provides the most stable and effective gradient signal for training!

---

### 6. Deep Derivation 6.5.1: Fisher Consistency, Proper Scoring Rules, and Bayes-Optimal Predictors

#### Context & Theoretical Foundation
In classification, our ultimate operational goal is minimizing 0-1 classification risk:
$$\mathcal{R}_{0-1}(f) = \mathbb{E}_{(X, Y)}\left[ \mathbb{I}(f(X) \ne Y) \right]$$
However, the 0-1 step function is non-convex and has zero gradient almost everywhere, making direct gradient optimization impossible. We optimize a continuous, differentiable **surrogate loss** $\ell(\hat{y}, y)$ instead.

A surrogate loss is defined as **Fisher consistent** (or classification-calibrated) if the minimizer of the surrogate population risk:
$$f^* = \arg\min_f \mathbb{E}_{(X, Y)}\left[ \ell(f(X), Y) \right]$$
preserves the exact decision boundary of the **Bayes-optimal classifier**:
$$f_{\text{Bayes}}(x) = \begin{cases} 1 & \text{if } \eta(x) \ge \frac{1}{2} \\ 0 & \text{if } \eta(x) < \frac{1}{2} \end{cases} \quad \text{where } \eta(x) \equiv P(Y = 1 \mid X = x)$$

---

#### 1. Binary Cross-Entropy as a Strictly Proper Scoring Rule
Let $p \in (0, 1)$ denote the model's predicted probability for $Y=1$.
Conditioned on feature $X=x$, the conditional expected risk under BCE is:
$$L_{\text{BCE}}(p; \eta) = \mathbb{E}_{Y \mid X=x}\left[ \mathcal{L}_{\text{BCE}}(p, Y) \right] = \eta [-\ln(p)] + (1 - \eta) [-\ln(1 - p)]$$

To find the optimal prediction $p^*$, we set the first derivative with respect to $p$ to zero:
$$\frac{\partial L_{\text{BCE}}}{\partial p} = -\frac{\eta}{p} + \frac{1 - \eta}{1 - p} = 0$$
$$\frac{1 - \eta}{1 - p} = \frac{\eta}{p} \implies \eta(1 - p) = p(1 - \eta) \implies \eta - \eta p = p - \eta p \implies \mathbf{p^* = \eta}$$

Checking the second derivative for strict convexity:
$$\frac{\partial^2 L_{\text{BCE}}}{\partial p^2} = \frac{\eta}{p^2} + \frac{1 - \eta}{(1 - p)^2} > 0 \quad \forall p \in (0, 1), \; \eta \in (0, 1)$$
Because the second derivative is strictly positive everywhere, $p^* = \eta(x) = P(Y = 1 \mid X = x)$ is the **unique global minimizer**.
By definition (Gneiting & Raftery, 2007), a scoring rule where the expected penalty is uniquely minimized if and only if the forecast equals the true posterior distribution is a **strictly proper scoring rule**.
Minimizing Cross-Entropy forces the model to produce **calibrated posterior probabilities**, not merely arbitrary ranking scores!

---

#### 2. Why MSE on Probabilities Fails in Deep Learning (Gradient Starvation)
Now consider using Mean Squared Error to train a classification network:
$$\mathcal{L}_{\text{MSE}}(p, y) = \frac{1}{2} (p - y)^2$$
The conditional expected MSE loss is:
$$L_{\text{MSE}}(p; \eta) = \frac{1}{2} \eta (p - 1)^2 + \frac{1}{2} (1 - \eta) p^2$$
Differentiating with respect to $p$:
$$\frac{\partial L_{\text{MSE}}}{\partial p} = \eta (p - 1) + (1 - \eta) p = p - \eta = 0 \implies \mathbf{p^* = \eta}$$
At the infinite-data limit, MSE is also Fisher consistent! Why then is MSE virtually never used for classification in deep neural networks?

The difference lies entirely in the **gradient dynamics with respect to the network logits $z$**, where $p = \sigma(z)$:

##### Gradient under BCE with Logits:
$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial p} \frac{dp}{dz} = \left( -\frac{y}{p} + \frac{1-y}{1-p} \right) \cdot p(1 - p) = \mathbf{p - y}$$
The gradient is linear in the classification error $(p - y)$, completely bounded and non-saturating.

##### Gradient under MSE with Logits:
$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial p} \frac{dp}{dz} = (p - y) \cdot \sigma'(z) = \mathbf{(p - y) \cdot p (1 - p)}$$

Suppose a training example has true label $y = 1$, but the network is currently completely wrong and confident: $z = -10 \implies p = \sigma(-10) \approx 4.54 \times 10^{-5}$.
- **BCE Gradient**: $\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = 4.54 \times 10^{-5} - 1.0 \approx \mathbf{-1.0000}$ (Strong, immediate corrective gradient!).
- **MSE Gradient**: $\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = (4.54 \times 10^{-5} - 1.0) \cdot (4.54 \times 10^{-5})(1 - 4.54 \times 10^{-5}) \approx \mathbf{-4.54 \times 10^{-5}}$ (The gradient vanishes by a factor of $22{,}000\times$!).

Under MSE, the more confidently incorrect the network is, the smaller the gradient becomes! The network experiences **gradient starvation** and freezes, unable to escape catastrophic misclassifications. BCE completely eliminates this pathology.

---

### 7. Deep Derivation 6.5.2: Analytical Gradient & Curvature of Focal Loss vs. Cross-Entropy

#### Step-by-Step Derivation of the Focal Loss Logit Gradient
Recall the definition of Binary Focal Loss:
$$\mathcal{L}_{\text{FL}}(p, y) = -y (1 - p)^\gamma \ln(p) - (1 - y) p^\gamma \ln(1 - p)$$
where $p = \sigma(z) = \frac{1}{1 + e^{-z}}$ and $\frac{dp}{dz} = p(1 - p)$.

We derive the exact analytical gradient $\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z}$ by parts for both classes:

##### Case 1: Positive Ground Truth ($y = 1$)
Here, $\mathcal{L}_{\text{FL}} = -(1 - p)^\gamma \ln(p)$.
Applying the calculus product rule:
$$\frac{d\mathcal{L}_{\text{FL}}}{dp} = -\left[ \frac{d}{dp}(1 - p)^\gamma \cdot \ln(p) + (1 - p)^\gamma \cdot \frac{d}{dp}\ln(p) \right]$$
Using $\frac{d}{dp}(1 - p)^\gamma = -\gamma (1 - p)^{\gamma - 1}$:
$$\frac{d\mathcal{L}_{\text{FL}}}{dp} = \gamma (1 - p)^{\gamma - 1} \ln(p) - \frac{(1 - p)^\gamma}{p}$$

Now apply the chain rule to differentiate with respect to logit $z$, using $\frac{dp}{dz} = p(1 - p)$:
$$\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z} = \frac{d\mathcal{L}_{\text{FL}}}{dp} \cdot p(1 - p) = \left[ \gamma (1 - p)^{\gamma - 1} \ln(p) - \frac{(1 - p)^\gamma}{p} \right] p(1 - p)$$
Multiplying through:
$$\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z}\Big|_{y=1} = \gamma p (1 - p)^\gamma \ln(p) - (1 - p)^{\gamma + 1}$$
Factoring out $(1 - p)^\gamma$:
$$\mathbf{\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z}\Big|_{y=1} = (1 - p)^\gamma \left[ \gamma p \ln(p) - (1 - p) \right]}$$

Notice that setting $\gamma = 0$ yields:
$$\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z}\Big|_{y=1, \gamma=0} = (1 - p)^0 [0 - (1 - p)] = p - 1 = p - y$$
recovering the standard BCE gradient exactly!

##### Case 2: Negative Ground Truth ($y = 0$)
Here, $\mathcal{L}_{\text{FL}} = -p^\gamma \ln(1 - p)$.
Differentiating with respect to $p$:
$$\frac{d\mathcal{L}_{\text{FL}}}{dp} = -\left[ \gamma p^{\gamma - 1} \ln(1 - p) + p^\gamma \left(-\frac{1}{1 - p}\right) \right] = -\gamma p^{\gamma - 1} \ln(1 - p) + \frac{p^\gamma}{1 - p}$$

Applying the chain rule with $\frac{dp}{dz} = p(1 - p)$:
$$\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z}\Big|_{y=0} = \left[ -\gamma p^{\gamma - 1} \ln(1 - p) + \frac{p^\gamma}{1 - p} \right] p(1 - p) = -\gamma p^\gamma (1 - p) \ln(1 - p) + p^{\gamma + 1}$$
Factoring out $p^\gamma$:
$$\mathbf{\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z}\Big|_{y=0} = p^\gamma \left[ p - \gamma (1 - p) \ln(1 - p) \right]}$$
Setting $\gamma = 0$ recovers $p - 0 = p - y$.

##### Unified Formula in Terms of $p_t$:
Defining $p_t = p$ if $y = 1$ and $p_t = 1 - p$ if $y = 0$:
$$\mathbf{\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z} = (2y - 1) \left[ \gamma p_t (1 - p_t)^\gamma \ln(p_t) - (1 - p_t)^{\gamma + 1} \right]}$$

---

#### Curvature Analysis (Second Derivative / Hessian Landscape)
Consider the asymptotic behavior as an example becomes well-classified ($p_t \to 1$):
- For Cross-Entropy ($\gamma = 0$):
  $$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = -(1 - p_t) \implies \frac{\partial^2 \mathcal{L}_{\text{CE}}}{\partial z^2} = p_t(1 - p_t) \approx \mathcal{O}(1 - p_t)$$
  The curvature vanishes **linearly**.
- For Focal Loss ($\gamma = 2$):
  $$(1 - p_t)^{\gamma + 1} = (1 - p_t)^3, \quad (1 - p_t)^\gamma = (1 - p_t)^2$$
  The gradient and curvature scale as $\mathcal{O}((1 - p_t)^2)$.
  The loss surface around well-classified points forms an **extremely flat plateau (basin of near-zero curvature)**. This topological flattening guarantees that thousands of easy background gradients sum to negligible magnitude, leaving gradient space completely unpolluted for hard foreground samples.

---

### 8. Deep Derivation 6.5.3: InfoNCE Loss as a Lower Bound on Mutual Information

#### Context & Mathematical Definition
In self-supervised representation learning (CLIP, SimCLR, CPC), we seek to learn feature representations $f(x)$ without human labels by maximizing the mutual information between different views of the same underlying data:
$$I(X; Y) = \mathbb{E}_{p(x, y)}\left[ \log \frac{p(x, y)}{p(x) p(y)} \right] = \mathbb{E}_{p(x, y)}\left[ \log \frac{p(y \mid x)}{p(y)} \right]$$

Given an anchor sample $x$, let $\{y_1, y_2, \dots, y_K\}$ be a set of $K$ candidate representations containing:
- Exactly one **positive sample** $y_1 \sim p(y \mid x)$
- $K - 1$ **negative samples** $y_2, \dots, y_K \sim p(y)$ drawn independently from the proposal distribution.

The **InfoNCE loss** (van den Oord, Li, & Vinyals, 2018) is defined as:
$$\mathbf{\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}_{X, Y}\left[ \log \frac{\exp(s(x, y_1) / \tau)}{\sum_{j=1}^K \exp(s(x, y_j) / \tau)} \right]}$$
where $s(x, y) = \frac{f(x)^T g(y)}{\|f(x)\| \|g(y)\|}$ is cosine similarity and $\tau > 0$ is a temperature hyperparameter.

---

#### Proof: InfoNCE Bound on Mutual Information
Let $C \in \{1, 2, \dots, K\}$ be a random variable indicating which of the $K$ candidates is the true positive sample.
Assume a uniform prior across candidates: $P(C = j) = \frac{1}{K}$.
The generative distribution of the tuple $(x, y_1, \dots, y_K)$ given $C = 1$ is:
$$p(x, y_1, \dots, y_K \mid C = 1) = p(x, y_1) \prod_{j=2}^K p(y_j) = p(x) p(y_1 \mid x) \prod_{j=2}^K p(y_j)$$

Applying Bayes' Theorem to compute the posterior probability that candidate 1 is the true match:
$$P(C = 1 \mid x, y_1, \dots, y_K) = \frac{P(C=1) p(x, y_1, \dots, y_K \mid C=1)}{\sum_{k=1}^K P(C=k) p(x, y_1, \dots, y_K \mid C=k)}$$
Canceling the uniform prior $P(C=k) = 1/K$ and dividing the numerator and denominator by $\prod_{j=1}^K p(y_j)$:
$$P(C = 1 \mid x, y_1, \dots, y_K) = \frac{\frac{p(x, y_1)}{p(x) p(y_1)}}{\sum_{k=1}^K \frac{p(x, y_k)}{p(x) p(y_k)}} = \frac{\frac{p(y_1 \mid x)}{p(y_1)}}{\sum_{k=1}^K \frac{p(y_k \mid x)}{p(y_k)}}$$

Notice that the InfoNCE objective:
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[ \log \frac{\exp(s(x, y_1)/\tau)}{\sum_{k=1}^K \exp(s(x, y_k)/\tau)} \right]$$
is precisely the multi-class categorical cross-entropy of predicting the correct positive index $C=1$, where the scoring function models the log-density ratio $\log \frac{p(y \mid x)}{p(y)} \approx s(x, y)/\tau$.

Now consider the optimal scoring function $f^*(x, y) = \frac{p(y \mid x)}{p(y)}$. Under this optimal critic:
$$\mathcal{L}_{\text{InfoNCE}}^* = -\mathbb{E}\left[ \log P(C = 1 \mid x, y_1, \dots, y_K) \right]$$

Expanding the internal logarithm:
$$\log P(C = 1 \mid x, y_1, \dots, y_K) = \log\left( \frac{\frac{p(y_1 \mid x)}{p(y_1)}}{\frac{p(y_1 \mid x)}{p(y_1)} + \sum_{k=2}^K \frac{p(y_k \mid x)}{p(y_k)}} \right) = -\log\left( 1 + \frac{p(y_1)}{p(y_1 \mid x)} \sum_{k=2}^K \frac{p(y_k \mid x)}{p(y_k)} \right)$$

Taking the expectation over the $K - 1$ independent negative samples $y_2, \dots, y_K \sim p(y)$:
$$\mathbb{E}_{y_2, \dots, y_K}\left[ \sum_{k=2}^K \frac{p(y_k \mid x)}{p(y_k)} \right] = \sum_{k=2}^K \int p(y_k) \frac{p(y_k \mid x)}{p(y_k)} dy_k = \sum_{k=2}^K \int p(y_k \mid x) dy_k = \sum_{k=2}^K 1 = K - 1$$

Applying **Jensen's inequality** to the strictly concave function $\phi(t) = -\log(1 + c \cdot t)$:
$$\mathbb{E}_{y_2, \dots, y_K}\left[ \log P(C = 1 \mid x, Y) \right] \le -\log\left( 1 + \frac{p(y_1)}{p(y_1 \mid x)} \cdot \mathbb{E}\left[ \sum_{k=2}^K \frac{p(y_k \mid x)}{p(y_k)} \right] \right)$$
$$= -\log\left( 1 + \frac{p(y_1)}{p(y_1 \mid x)} (K - 1) \right) \le -\log\left( \frac{p(y_1)}{p(y_1 \mid x)} K \right) = \log \frac{p(y_1 \mid x)}{p(y_1)} - \log(K)$$

Taking the outer expectation over $(x, y_1) \sim p(x, y)$:
$$\mathbb{E}_{(x, y_1)}\left[ \mathbb{E}_{y_{2:K}}[\log P(C = 1 \mid x, Y)] \right] \le \mathbb{E}_{(x, y_1)}\left[ \log \frac{p(y_1 \mid x)}{p(y_1)} \right] - \log(K) = I(X; Y) - \log(K)$$

Multiplying through by $-1$:
$$\mathcal{L}_{\text{InfoNCE}} \ge \log(K) - I(X; Y)$$
Rearranging terms yields the fundamental **InfoNCE Mutual Information Bound**:
$$\mathbf{I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}}$$
$\blacksquare$ **Q.E.D.**

#### Theoretical Insights from the Bound:
1. **Capacity Limit**: The maximum mutual information that InfoNCE can extract is strictly capped at $\log(K)$. To learn rich representations that capture high mutual information, models must scale the number of negative samples $K$ (e.g., CLIP's batch size of $K = 32{,}768$ yields $\log_2(32768) = 15 \text{ bits}$ of information capacity).
2. **Temperature Parameter $\tau$**: The scale $\tau$ controls the hardness of negatives. As $\tau \to 0$, the softmax approaches an argmax, causing the gradient to focus exclusively on the single hardest negative in the mini-batch.

---

## Part 3: Geometric & Information-Theoretic Interpretation

### 1. Cross-Entropy as Kullback-Leibler (KL) Divergence
Let $P$ be the true data distribution and $Q_\theta$ be the model's predicted distribution.
The KL divergence is:
$$D_{\text{KL}}(P \parallel Q_\theta) = \sum_{x} P(x) \log\left( \frac{P(x)}{Q_\theta(x)} \right) = \sum_x P(x) \log P(x) - \sum_x P(x) \log Q_\theta(x)$$
$$D_{\text{KL}}(P \parallel Q_\theta) = -H(P) + H(P, Q_\theta)$$
where $H(P)$ is the entropy of the true data, and $H(P, Q_\theta)$ is the **Cross-Entropy**.

Because the ground-truth distribution $P$ is fixed, its entropy $H(P)$ is a constant independent of model parameters $\theta$:
$$\nabla_\theta D_{\text{KL}}(P \parallel Q_\theta) = \nabla_\theta H(P, Q_\theta) = \nabla_\theta \mathcal{L}_{\text{CE}}$$
**Minimizing Cross-Entropy is strictly equivalent to minimizing the statistical distance (KL divergence) between the model and reality!**

---

## Part 4: Real-World Analogy

### 1. The University Grading Scale
- **MSE (Grading on strict numerical deviation)**: If you miss a question by 5 points, your grade drops by 25 points. If you miss by 20 points, your grade drops by 400 points. Minor arithmetic slips are brutally penalized.
- **Cross-Entropy (Grading on confidence & conviction)**: If you are uncertain between A and B, the penalty is small. But if you state with 99.9% certainty that the answer is A when it is actually B, the penalty is nearly infinite ($-\log(0.001) \approx 6.9$). Cross-Entropy ruthlessly punishes arrogant mistakes.
- **Focal Loss (An elite coach reviewing exam questions)**: The coach skips all the basic questions that the student answered easily with 99% accuracy. He forces the student to spend 100% of study time reviewing the 3 difficult questions that were missed.
- **Triplet Loss (The Police Mugshot Lineup)**: A witness looks at a suspect (Anchor). The detective brings in a known family photo of the suspect (Positive) and an innocent citizen (Negative). The goal is not to assign a numerical score, but to ensure that the suspect's two photos look closer to each other than to any other person in the database by at least a clear margin.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us calculate by hand the exact loss values and logit gradients across **BCE Loss** vs. **Focal Loss** on three representative classification samples under severe class imbalance ($\gamma = 2, \alpha = 1$).

### Three Test Samples
1. **Sample 1 (Easy Positive)**: True $y = 1$, logit $z = +3.0$.
2. **Sample 2 (Hard Negative)**: True $y = 0$, logit $z = +2.0$ (confident false alarm!).
3. **Sample 3 (Borderline)**: True $y = 1$, logit $z = 0.0$.

---

### Step-by-Step Calculations

#### 1. Forward Probabilities $p = \sigma(z) = \frac{1}{1 + e^{-z}}$:
- Sample 1: $p_1 = \sigma(3.0) = \frac{1}{1 + e^{-3}} \approx \frac{1}{1.049787} \approx \mathbf{0.9526}$.
  True class probability: $p_{t, 1} = 0.9526$.
- Sample 2: $p_2 = \sigma(2.0) = \frac{1}{1 + e^{-2}} \approx \frac{1}{1.135335} \approx \mathbf{0.8808}$.
  Since true $y = 0$, true class probability is $p_{t, 2} = 1 - p_2 = 1 - 0.8808 = \mathbf{0.1192}$.
- Sample 3: $p_3 = \sigma(0.0) = \frac{1}{1 + 1} = \mathbf{0.5000}$.
  True class probability: $p_{t, 3} = 0.5000$.

---

#### 2. Cross-Entropy Loss: $\mathcal{L}_{\text{BCE}} = -\log(p_t)$
- Sample 1: $\mathcal{L}_{\text{BCE}, 1} = -\log(0.9526) \approx \mathbf{0.0486}$.
- Sample 2: $\mathcal{L}_{\text{BCE}, 2} = -\log(0.1192) \approx \mathbf{2.1270}$.
- Sample 3: $\mathcal{L}_{\text{BCE}, 3} = -\log(0.5000) \approx \mathbf{0.6931}$.

---

#### 3. Focal Loss Modulating Factor $(1 - p_t)^2$ and Loss:
- Sample 1:
  $$(1 - p_{t, 1})^2 = (1 - 0.9526)^2 = (0.0474)^2 = \mathbf{0.002247}$$
  $$\mathcal{L}_{\text{FL}, 1} = 0.002247 \times 0.0486 \approx \mathbf{0.000109}$$
  *(Notice: The loss dropped from $0.0486$ to $0.000109$—a **$445\times$ suppression!**)*
- Sample 2:
  $$(1 - p_{t, 2})^2 = (1 - 0.1192)^2 = (0.8808)^2 = \mathbf{0.7758}$$
  $$\mathcal{L}_{\text{FL}, 2} = 0.7758 \times 2.1270 \approx \mathbf{1.6501}$$
  *(Hard negative retained $78\%$ of its original loss!)*
- Sample 3:
  $$(1 - p_{t, 3})^2 = (1 - 0.5000)^2 = 0.5^2 = \mathbf{0.2500}$$
  $$\mathcal{L}_{\text{FL}, 3} = 0.2500 \times 0.6931 \approx \mathbf{0.1733}$$

---

#### 4. Logit Gradients $\frac{\partial \mathcal{L}}{\partial z}$:
- **Standard BCE Gradient: $\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = p - y$**:
  - Sample 1 ($y=1$): $g_{\text{BCE}, 1} = 0.9526 - 1 = \mathbf{-0.0474}$.
  - Sample 2 ($y=0$): $g_{\text{BCE}, 2} = 0.8808 - 0 = \mathbf{+0.8808}$.
  - Sample 3 ($y=1$): $g_{\text{BCE}, 3} = 0.5000 - 1 = \mathbf{-0.5000}$.
- **Focal Loss Gradient: $\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z}$**:
  Formula for $y=1$: $(p - 1)(1 - p)^\gamma + \gamma p (1 - p)^\gamma \log(p) = -(1 - p)^{\gamma + 1} + \gamma p (1 - p)^\gamma \log(p)$.
  - Sample 1 ($y=1, p=0.9526, 1-p=0.0474$):
    Term 1: $-(0.0474)^3 \approx -0.000106$.
    Term 2: $2(0.9526)(0.0474)^2 \log(0.9526) \approx 2(0.9526)(0.002247)(-0.0486) \approx -0.000208$.
    $$g_{\text{FL}, 1} \approx -0.000106 - 0.000208 = \mathbf{-0.000314}$$
    *(Notice: Gradient collapsed from $-0.0474$ to $-0.000314$—a **$150\times$ reduction!**)*
  - Sample 2 ($y=0, p=0.8808, 1-p=0.1192$):
    Term 1: $p^{\gamma + 1} = 0.8808^3 \approx 0.6833$.
    Term 2: $-2(0.1192)(0.8808)^2 \log(0.1192) \approx -2(0.1192)(0.7758)(-2.1270) \approx +0.3934$.
    $$g_{\text{FL}, 2} \approx 0.6833 + 0.3934 = \mathbf{+1.0767}$$
    *(The hard negative dominates the training update!)*

---

### Comparative Visual Grid: BCE vs. Focal Loss

```
+----------------------------------------------------------------------------------------------------+
|                                    BCE VS. FOCAL LOSS COMPARISON (\gamma = 2)                      |
+-------+-----+-------+--------+--------+-------------+-------------+---------------+----------------+
| Sample| y   | Logit | Prob p | p_t    | BCE Loss    | Focal Loss  | BCE Grad dL/dz| Focal Grad     |
+-------+-----+-------+--------+--------+-------------+-------------+---------------+----------------+
| 1 (Ez)| 1   | +3.0  | 0.9526 | 0.9526 | 0.0486      | 0.000109    | -0.0474       | -0.000314      |
|       |     |       |        |        |             | (445x drop!)|               | (150x drop!)   |
+-------+-----+-------+--------+--------+-------------+-------------+---------------+----------------+
| 2 (Hd)| 0   | +2.0  | 0.8808 | 0.1192 | 2.1270      | 1.6501      | +0.8808       | +1.0767        |
|       |     |       |        |        |             | (Active!)   |               | (DOMINANT!)    |
+-------+-----+-------+--------+--------+-------------+-------------+---------------+----------------+
| 3 (Bd)| 1   |  0.0  | 0.5000 | 0.5000 | 0.6931      | 0.1733      | -0.5000       | -0.2201        |
+-------+-----+-------+--------+--------+-------------+-------------+---------------+----------------+
| TAKEAWAY: Focal Loss dynamically suppresses easy samples, focusing 99.9% of gradients on hard cases!|
+----------------------------------------------------------------------------------------------------+
```

---

### Step-by-Step Triplet Loss Manual Calculation

Let Anchor $a = [0.0, 0.0]^T$, Positive $p = [0.5, 0.5]^T$, Negative $n = [0.6, 0.6]^T$, with margin $\alpha = 0.5$.
1. **Squared Euclidean Distances**:
   $$d(a, p)^2 = (0.0 - 0.5)^2 + (0.0 - 0.5)^2 = 0.25 + 0.25 = \mathbf{0.50}$$
   $$d(a, n)^2 = (0.0 - 0.6)^2 + (0.0 - 0.6)^2 = 0.36 + 0.36 = \mathbf{0.72}$$
2. **Margin Condition Check**:
   $$d(a, p)^2 - d(a, n)^2 + \alpha = 0.50 - 0.72 + 0.50 = -0.22 + 0.50 = \mathbf{+0.28}$$
   Since $+0.28 > 0$, this is a **Semi-Hard Triplet**! The loss is active!
   $$\mathcal{L}_{\text{triplet}} = \max(0, 0.28) = \mathbf{0.2800}$$
3. **Gradients**:
   $$\frac{\partial \mathcal{L}}{\partial p} = 2 (p - a) = 2 \begin{bmatrix} 0.5 - 0 \\ 0.5 - 0 \end{bmatrix} = \begin{bmatrix} \mathbf{+1.0} \\ \mathbf{+1.0} \end{bmatrix}$$
   $$\frac{\partial \mathcal{L}}{\partial n} = 2 (a - n) = 2 \begin{bmatrix} 0 - 0.6 \\ 0 - 0.6 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.2} \\ \mathbf{-1.2} \end{bmatrix}$$
   $$\frac{\partial \mathcal{L}}{\partial a} = 2 (n - p) = 2 \begin{bmatrix} 0.6 - 0.5 \\ 0.6 - 0.5 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.2} \\ \mathbf{+0.2} \end{bmatrix}$$

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning Analog |
| :--- | :--- | :--- | :--- |
| $\hat{y}$ | Continuous scalar/tensor | Model predicted target | Network regression output |
| $y$ | Continuous or discrete | Ground-truth supervision signal | Training label tensor |
| $z$ | $\mathbb{R}$ or $\mathbb{R}^K$ | Unnormalized logit score | Output of final `nn.Linear` layer |
| $p_t$ | $[0, 1]$ scalar | Probability assigned to ground truth | Model confidence on true class |
| $\gamma \ge 0$ | Scalar hyperparameter | Focusing curvature parameter | Exponent in Focal Loss (`gamma=2.0`) |
| $(1 - p_t)^\gamma$ | $[0, 1]$ scalar | Modulating suppression factor | Dynamic sample-weight multiplier |
| $f(a), f(p), f(n)$ | $\mathbb{R}^d$ embedding | Normalized representation vectors | Output of embedding encoder |
| $\alpha > 0$ | Scalar hyperparameter | Metric separation margin | Separation buffer in Triplet Loss |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Equivalence of Maximum Likelihood and Cross-Entropy
**Statement**: Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be an i.i.d. dataset where $y_i \in \{1, \dots, K\}$.
Prove that finding the Maximum Likelihood parameters $\theta_{\text{MLE}}$ is strictly identical to minimizing the empirical Categorical Cross-Entropy loss.

**Proof**:
1. The likelihood of the dataset under parameter vector $\theta$ is:
   $$\mathcal{L}(\theta) = \prod_{i=1}^N P(Y = y_i | x_i; \theta) = \prod_{i=1}^N S_{y_i}(x_i; \theta)$$
2. The log-likelihood is:
   $$\log \mathcal{L}(\theta) = \sum_{i=1}^N \log S_{y_i}(x_i; \theta)$$
3. The Maximum Likelihood Estimator maximizes this quantity:
   $$\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log S_{y_i}(x_i; \theta)$$
4. Multiplying by $-\frac{1}{N}$ turns the maximization into an equivalent minimization:
   $$\theta_{\text{MLE}} = \arg\min_\theta \left( -\frac{1}{N} \sum_{i=1}^N \log S_{y_i}(x_i; \theta) \right)$$
5. Notice that $-\log S_{y_i}(x_i; \theta)$ is exactly the Categorical Cross-Entropy loss $\mathcal{L}_{\text{CCE}}(S(x_i), y_i)$!
   $$\theta_{\text{MLE}} = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{\text{CCE}}(S(x_i), y_i)$$
$\blacksquare$ Minimizing Cross-Entropy is mathematically identical to Maximum Likelihood Estimation!

---

### Problem 2: Triplet Loss Representation Collapse Pathology
**Statement**: In Triplet Loss, what happens to the optimal solution if embeddings $f(x)$ are *not* normalized to unit norm ($\|f(x)\|_2 = 1$)?

**Analysis**:
The loss is:
$$\mathcal{L} = \max(0, \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha)$$
- **Degenerate Collapse to Zero**:
  If the network sets all weights to zero, then $f(x) = \vec{0}$ for all inputs.
  $$\mathcal{L} = \max(0, 0 - 0 + \alpha) = \alpha$$
- **Degenerate Explosion to Infinity**:
  If the network scales all embeddings by a huge scalar $C \to \infty$, then for any configuration where the negative is slightly farther than the positive ($\|f(a)-f(n)\| > \|f(a)-f(p)\|$):
  $$\|C f(a) - C f(p)\|^2 - \|C f(a) - C f(n)\|^2 + \alpha = C^2 (d_{ap}^2 - d_{an}^2) + \alpha \to -\infty$$
  The loss becomes $\max(0, -\infty) = 0$ everywhere without learning any useful semantic structure!
*Resolution*: Embeddings must strictly be projected onto the unit hypersphere:
$$f(x) \leftarrow \frac{f(x)}{\|f(x)\|_2}$$
This bounds all pairwise squared distances to $[0, 4]$, preventing both collapse to zero and explosion to infinity.

---

### Problem 3: MSE vs. Cross-Entropy Gradient Starvation on Confident Misclassification

**Statement**:
Consider a single output neuron with Sigmoid activation $p = \sigma(z) = \frac{1}{1 + e^{-z}}$.
A training example has true binary label $y = 1.0$. The model currently outputs logit $z = -4.0$ (confident misclassification).
1. Calculate the predicted probability $p$.
2. Calculate the forward loss under MSE $\mathcal{L}_{\text{MSE}} = \frac{1}{2}(p - y)^2$ and BCE $\mathcal{L}_{\text{BCE}} = -\ln(p)$.
3. Calculate the exact analytical gradient with respect to logit $z$ for both losses:
   $$g_{\text{MSE}} = \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = (p - y) \cdot p(1 - p)$$
   $$g_{\text{BCE}} = \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = p - y$$
4. Compute the gradient ratio $\frac{|g_{\text{BCE}}|}{|g_{\text{MSE}}|}$ and explain why MSE stalls gradient descent.

---

#### Solution

##### 1. Forward Predicted Probability
Using $z = -4.0$:
$$e^{-(-4.0)} = e^{4.0} \approx 54.5981500$$
$$p = \frac{1}{1 + e^{4.0}} = \frac{1}{1 + 54.5981500} = \frac{1}{55.5981500} \approx \mathbf{0.0179862}$$
$$1 - p \approx 0.9820138$$

##### 2. Forward Loss Values
- **Mean Squared Error**:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{2}(0.0179862 - 1.0000000)^2 = \frac{1}{2}(-0.9820138)^2 = \frac{1}{2}(0.9643511) \approx \mathbf{0.4821756}$$
- **Binary Cross-Entropy**:
  $$\mathcal{L}_{\text{BCE}} = -\ln(0.0179862) \approx \mathbf{4.0181500}$$

##### 3. Gradients with Respect to Logit $z$
- **BCE Gradient**:
  $$g_{\text{BCE}} = p - y = 0.0179862 - 1.0000000 = \mathbf{-0.9820138}$$
- **MSE Gradient**:
  $$g_{\text{MSE}} = (p - y) \cdot p(1 - p) = (-0.9820138) \cdot (0.0179862)(0.9820138) \approx (-0.9820138) \cdot (0.0176627) \approx \mathbf{-0.0173449}$$

##### 4. Ratio & Mathematical Interpretation
$$\frac{|g_{\text{BCE}}|}{|g_{\text{MSE}}|} = \frac{0.9820138}{0.0173449} \approx \mathbf{56.617 \times}$$
- The BCE gradient is **over $56\times$ stronger**!
- **Core Insight**: Under MSE, the gradient contains the multiplicative factor $\sigma'(z) = p(1 - p)$. When the network is confidently wrong ($z = -4.0$), $\sigma'(z) \approx 0.01766$, which almost completely extinguishes the error signal. The parameters receive virtually zero update, trapping the network in a severe local plateau.
- Under BCE, the denominator $p$ in the log-likelihood derivative $\frac{\partial \mathcal{L}}{\partial p} = -\frac{1}{p}$ **exactly cancels** the $p$ term in $\sigma'(z) = p(1-p)$, preserving the unattenuated error signal $p - y \approx -0.982$.

---

### Problem 4: Huber Loss (Smooth $L_1$) Numerical Trace Across Regimes

**Statement**:
Let the Huber loss threshold parameter be $\delta = 1.0$:
$$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta \left( |r| - \frac{1}{2} \delta \right) & \text{if } |r| > \delta \end{cases}$$
For three residual errors $r = \hat{y} - y$:
1. $r_1 = +0.500$ (inlier)
2. $r_2 = +1.000$ (boundary point)
3. $r_3 = +5.000$ (extreme outlier)

Compute the exact loss values and gradients $\frac{\partial \mathcal{L}}{\partial r}$ under MSE ($\frac{1}{2} r^2$), MAE ($|r|$), and Huber Loss ($\mathcal{L}_\delta$). Verify the continuous differentiability ($C^1$) at the boundary and explain the influence function boundedness.

---

#### Solution

##### 1. Residual $r_1 = +0.500$ (Quadratic Regime: $|r_1| \le \delta$)
- **MSE**:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{2}(0.5)^2 = \mathbf{0.125000}, \quad g_{\text{MSE}} = r_1 = \mathbf{+0.500000}$$
- **MAE**:
  $$\mathcal{L}_{\text{MAE}} = |0.5| = \mathbf{0.500000}, \quad g_{\text{MAE}} = \text{sign}(r_1) = \mathbf{+1.000000}$$
- **Huber**:
  $$\mathcal{L}_\delta = \frac{1}{2}(0.5)^2 = \mathbf{0.125000}, \quad g_\delta = r_1 = \mathbf{+0.500000}$$
*(Huber matches MSE identically in the small error regime)*.

---

##### 2. Residual $r_2 = +1.000$ (Boundary: $|r_2| = \delta = 1.0$)
- Approaching from left ($|r| \le 1.0$):
  $$\mathcal{L}_{\text{left}} = \frac{1}{2}(1.0)^2 = \mathbf{0.500000}, \quad g_{\text{left}} = 1.0 = \mathbf{+1.000000}$$
- Approaching from right ($|r| > 1.0$):
  $$\mathcal{L}_{\text{right}} = 1.0 \left( 1.0 - \frac{1}{2}(1.0) \right) = 1.0(0.5) = \mathbf{0.500000}, \quad g_{\text{right}} = 1.0 \cdot \text{sign}(1.0) = \mathbf{+1.000000}$$
Both loss and gradient agree from both sides: $\mathcal{L}_\delta(1.0) = \mathbf{0.500000}$ and $g_\delta(1.0) = \mathbf{+1.000000}$. The transition is smoothly $C^1$ differentiable!

---

##### 3. Residual $r_3 = +5.000$ (Linear Regime: $|r_3| > \delta$)
- **MSE**:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{2}(5.0)^2 = \mathbf{12.500000}, \quad g_{\text{MSE}} = r_3 = \mathbf{+5.000000}$$
- **MAE**:
  $$\mathcal{L}_{\text{MAE}} = |5.0| = \mathbf{5.000000}, \quad g_{\text{MAE}} = \text{sign}(r_3) = \mathbf{+1.000000}$$
- **Huber**:
  $$\mathcal{L}_\delta = 1.0 \left( 5.0 - \frac{1}{2}(1.0) \right) = 5.0 - 0.5 = \mathbf{4.500000}, \quad g_\delta = 1.0 \cdot \text{sign}(5.0) = \mathbf{+1.000000}$$

##### Comparison & Robust Statistics Takeaway
```
+----------+---------------+----------------+----------------+----------------+
| Residual | Property      | MSE            | MAE            | Huber (delta=1)|
+----------+---------------+----------------+----------------+----------------+
| r = 0.5  | Loss / Grad   | 0.125 / +0.500 | 0.500 / +1.000 | 0.125 / +0.500 |
| r = 1.0  | Loss / Grad   | 0.500 / +1.000 | 1.000 / +1.000 | 0.500 / +1.000 |
| r = 5.0  | Loss / Grad   | 12.50 / +5.000 | 5.000 / +1.000 | 4.500 / +1.000 |
+----------+---------------+----------------+----------------+----------------+
```
Under MSE, an outlier with error $r = 5.0$ exerts $5\times$ the gradient pull of a unit error, dominating parameter updates. Under Huber loss, the influence function $\psi(r) = \frac{\partial \mathcal{L}}{\partial r}$ is **strictly bounded**:
$$\sup_{r \in \mathbb{R}} |\psi(r)| = \delta = 1.0$$
This guarantees finite sample robustness against corrupted labels and heavy-tailed measurement noise.

---

### Problem 5: InfoNCE Contrastive Loss & Temperature Scaling Arithmetic

**Statement**:
In self-supervised contrastive learning (CLIP, SimCLR), an anchor embedding $a \in \mathbb{R}^2$ is compared against a positive embedding $p \in \mathbb{R}^2$ and two negative embeddings $n_1, n_2 \in \mathbb{R}^2$:
$$a = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad p = \begin{bmatrix} 0.8 \\ 0.6 \end{bmatrix}, \quad n_1 = \begin{bmatrix} 0.6 \\ -0.8 \end{bmatrix}, \quad n_2 = \begin{bmatrix} -1.0 \\ 0.0 \end{bmatrix}$$
All vectors are normalized to unit length ($\|v\|_2 = 1.0$).
1. Calculate the cosine similarity scores $s(a, p), s(a, n_1), s(a, n_2)$.
2. Calculate the InfoNCE loss $\mathcal{L}_{\text{InfoNCE}}$ and candidate posterior probabilities under temperature $\tau = 1.0$.
3. Calculate the InfoNCE loss $\mathcal{L}_{\text{InfoNCE}}$ and candidate posterior probabilities under temperature $\tau = 0.2$.
4. Analyze how temperature $\tau$ modulates the gradient focus on hard negatives.

---

#### Solution

##### 1. Pairwise Cosine Similarities $s(a, v) = a^T v$
$$s(a, p) = (1.0)(0.8) + (0.0)(0.6) = \mathbf{+0.800000}$$
$$s(a, n_1) = (1.0)(0.6) + (0.0)(-0.8) = \mathbf{+0.600000} \quad (\text{Hard Negative!})$$
$$s(a, n_2) = (1.0)(-1.0) + (0.0)(0.0) = \mathbf{-1.000000} \quad (\text{Easy Negative!})$$

---

##### 2. InfoNCE under Standard Temperature $\tau = 1.0$
Scaled logits $z_i = s(a, \cdot) / \tau$:
$$z_p = \frac{0.8}{1.0} = 0.8000, \quad z_{n_1} = \frac{0.6}{1.0} = 0.6000, \quad z_{n_2} = \frac{-1.0}{1.0} = -1.0000$$

Exponentials:
$$e^{0.8000} \approx 2.2255409$$
$$e^{0.6000} \approx 1.8221188$$
$$e^{-1.0000} \approx 0.3678794$$
$$\sum = 2.2255409 + 1.8221188 + 0.3678794 = 4.4155391$$

Softmax Probabilities:
$$P(p) = \frac{2.2255409}{4.4155391} \approx \mathbf{0.504025} \quad (50.4\%)$$
$$P(n_1) = \frac{1.8221188}{4.4155391} \approx \mathbf{0.412661} \quad (41.3\%)$$
$$P(n_2) = \frac{0.3678794}{4.4155391} \approx \mathbf{0.083314} \quad (8.3\%)$$

InfoNCE Loss:
$$\mathcal{L}_{\text{InfoNCE}}(\tau = 1.0) = -\ln(0.504025) \approx \mathbf{0.685130}$$

---

##### 3. InfoNCE under Sharp Temperature $\tau = 0.2$
Scaled logits $z_i = s(a, \cdot) / 0.2$:
$$z_p = \frac{0.8}{0.2} = 4.0000, \quad z_{n_1} = \frac{0.6}{0.2} = 3.0000, \quad z_{n_2} = \frac{-1.0}{0.2} = -5.0000$$

Exponentials:
$$e^{4.0000} \approx 54.5981500$$
$$e^{3.0000} \approx 20.0855369$$
$$e^{-5.0000} \approx 0.0067379$$
$$\sum = 54.5981500 + 20.0855369 + 0.0067379 = 74.6904248$$

Softmax Probabilities:
$$P(p) = \frac{54.5981500}{74.6904248} \approx \mathbf{0.730993} \quad (73.1\%)$$
$$P(n_1) = \frac{20.0855369}{74.6904248} \approx \mathbf{0.268917} \quad (26.9\%)$$
$$P(n_2) = \frac{0.0067379}{74.6904248} \approx \mathbf{0.000090} \quad (0.009\%)$$

InfoNCE Loss:
$$\mathcal{L}_{\text{InfoNCE}}(\tau = 0.2) = -\ln(0.730993) \approx \mathbf{0.313351}$$

---

##### 4. Comparative Analysis: The Temperature Sharpening Mechanism
```
+------------------------------------------------------------------------------------+
|                       TEMPERATURE EFFECT ON INFONCE PROBABILITIES                  |
+-------------------+-------------+-------------------+------------------------------+
| Candidate         | Cosine Sim  | Prob (tau = 1.0)  | Prob (tau = 0.2)             |
+-------------------+-------------+-------------------+------------------------------+
| Positive p        | +0.8000     | 50.40%            | 73.10%                       |
| Hard Negative n_1 | +0.6000     | 41.27%            | 26.89% (Takes 99.9% of negs!)|
| Easy Negative n_2 | -1.0000     |  8.33%            |  0.009% (Completely ignored!)|
+-------------------+-------------+-------------------+------------------------------+
| InfoNCE Loss      | -           | 0.685130          | 0.313351                     |
+-------------------+-------------+-------------------+------------------------------+
```
- **Temperature Scaling Role**: Dividing by $\tau < 1$ acts as an inverse temperature (in Boltzmann distribution terms). It amplifies differences between logits: the difference $0.8 - 0.6 = 0.2$ becomes $\frac{0.2}{0.2} = 1.0$ in log-space, corresponding to an odds ratio of $e^1 \approx 2.718$.
- **Negative Filtering**: Under $\tau = 0.2$, the easy negative $n_2$ receives probability $0.009\%$ and produces zero gradient, while the hard negative $n_1$ accounts for $99.96\%$ of all negative competition. Temperature $\tau$ acts as a **soft mining mechanism**!

---

## Part 7: Deep Learning Connection & Application

### 1. RetinaNet & Conquering Extreme Class Imbalance
Prior to Focal Loss, single-stage object detectors (YOLOv1, SSD) suffered lower accuracy than two-stage detectors (Faster R-CNN) because two-stage detectors used a Region Proposal Network (RPN) to filter out 99% of easy background candidates.
Lin et al. demonstrated that simply replacing standard Cross-Entropy with **Focal Loss ($\alpha = 0.25, \gamma = 2.0$)** allowed a single-stage detector to surpass state-of-the-art two-stage detectors while running at real-time speeds!

### 2. Contrastive Learning & Modern Foundation Models (CLIP, SimCLR)
Modern multimodal AI (OpenAI's CLIP) generalizes Triplet Loss to a multi-negative softmax loss known as **InfoNCE**:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(a, p) / \tau)}{\exp(\text{sim}(a, p) / \tau) + \sum_{j} \exp(\text{sim}(a, n_j) / \tau)}$$
CLIP computes cosine similarity between image embeddings and text embeddings across a batch of $32{,}768$ samples, pulling paired text-image vectors together while pushing all other $32{,}767$ negative pairings apart!

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `06_deep_learning_foundations/code/05_loss_functions_deep_dive.py`. It implements:
1. Exact visual grid verification of BCE vs. Focal Loss forward loss and logit gradients matching Part 5 hand calculations.
2. Triplet loss semi-hard calculation and analytical gradient verification.
3. Strict parity check between scratch loss implementations (MSE, Huber, BCEWithLogits, CrossEntropy, TripletMarginLoss) and PyTorch's native `torch.nn` loss modules to machine precision ($< 10^{-7}$).
4. Empirical class imbalance experiment demonstrating Focal Loss achieving higher average precision on an imbalanced dataset (99% negative, 1% positive) compared to standard BCE.

Save the code and run it directly in Python 3.
