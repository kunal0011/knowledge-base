# 5.6 Adaptive Learning Rates (AdaGrad, RMSprop, Adam, AdamW)

---

## Part 1: Intuition & 101 Motivation

In standard Gradient Descent and Momentum, a single global learning rate $\alpha$ scales the update vector across all $d$ dimensions:
$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

In deep neural networks, this scalar step-size assumption causes immediate failure due to **extreme parameter heterogeneity**:
1. **Feature Frequency Disparity**: In Natural Language Processing and tabular data, certain embeddings (e.g., the word *"the"*) receive gradient updates on almost every mini-batch, while rare tokens (e.g., *"phenomenology"*) receive non-zero gradients only once every thousand batches. With a uniform learning rate, frequent weights update too rapidly and diverge, while rare weights barely move.
2. **Layer Curvature Disparity**: In deep networks, gradients in early layers are often orders of magnitude smaller than gradients in the final classification layer. A learning rate that is stable for the head causes vanishing progress in the stem.

To solve this, **adaptive learning rate algorithms** assign an individualized, coordinate-wise step size to every single parameter in the model:
$$\theta_{t+1, i} = \theta_{t, i} - \frac{\alpha}{\sqrt{s_{t, i}} + \epsilon} \cdot \text{direction}_{t, i}$$
where $s_{t, i}$ accumulates historical gradient magnitude along dimension $i$.

The historical evolution of adaptive optimization proceeded through four foundational breakthroughs:
1. **AdaGrad (2011)**: Sums all historical squared gradients ($G_t = G_{t-1} + g_t^2$). *Flaw*: Since $g_t^2 \ge 0$, $G_t$ grows monotonically, driving the effective learning rate $\alpha / \sqrt{G_t}$ to zero, causing premature learning starvation.
2. **RMSprop (2012)**: Replaces the infinite sum with an Exponentially Weighted Moving Average ($v_t = \beta v_{t-1} + (1-\beta) g_t^2$). This "leaky" accumulator discards ancient history, allowing continuous learning.
3. **Adam (2014)**: Combines the first moment (momentum $m_t$) and the second uncentered moment (RMSprop $v_t$), augmented with analytical **bias corrections** ($\hat{m}_t, \hat{v}_t$) to eliminate initialization artifacts at early time steps.
4. **AdamW (2017)**: Fixes a severe flaw in Adam's interaction with $L_2$ regularization. Loshchilov & Hutter proved that standard $L_2$ regularization in Adam inadvertently penalizes weights with large gradients far *less* than weights with small gradients. AdamW **decouples weight decay** from the adaptive gradient moment, becoming the undisputed default optimizer for modern Transformers and Large Language Models.

---

## Part 2: Rigorous Mathematical Formulation

Let $\theta \in \mathbb{R}^d$ denote the model parameters, $f(\theta)$ the objective function, and $g_t = \nabla_\theta f_t(\theta_t)$ the stochastic mini-batch gradient at step $t \ge 1$.

```
                    CHRONOLOGICAL EVOLUTION OF ADAPTIVE OPTIMIZERS
┌──────────────┐      Leaky EWMA      ┌──────────────┐    Add Momentum &    ┌──────────────┐
│   AdaGrad    │ ───────────────────> │   RMSprop    │ ───────────────────> │     Adam     │
│ (Duchi 2011) │  (Hinton 2012 Coursera)│ (Hinton 2012) │  Bias Correction   │(Kingma & Ba) │
└──────────────┘                      └──────────────┘  (Kingma & Ba 2014)  └──────────────┘
                                                                                   │
                                                                         Decouple  │ Weight
                                                                          Decay    │ (2017)
                                                                                   v
                                                                            ┌──────────────┐
                                                                            │    AdamW     │
                                                                            │(Loshchilov & │
                                                                            │ Hutter 2017) │
                                                                            └──────────────┘
```

---

### 1. AdaGrad (Adaptive Subgradient Method)

At each step $t \ge 1$:
$$G_t = G_{t-1} + g_t \odot g_t$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} \odot g_t$$
where $\odot$ denotes element-wise Hadamard product, and $\epsilon > 0$ (typically $10^{-8}$) prevents division by zero.

- **Limitation**: $G_{t, i} = \sum_{\tau=1}^t g_{\tau, i}^2$. As $t \to \infty$, $G_{t, i} \to \infty$, which guarantees that $\frac{\alpha}{\sqrt{G_{t, i}} + \epsilon} \to 0$. In non-convex deep learning, the optimizer permanently freezes long before discovering good local minima.

---

### 2. RMSprop (Root Mean Square Propagation)

RMSprop replaces the monotonic sum with an Exponential Moving Average (EMA) parameterized by smoothing factor $\beta \in [0, 1)$ (typically $0.9$ or $0.99$):
$$v_t = \beta v_{t-1} + (1 - \beta) (g_t \odot g_t)$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot g_t$$

The effective memory window is approximately $\frac{1}{1-\beta}$ steps. The denominator $\sqrt{v_t}$ acts as a local estimate of the Root Mean Square of recent gradients:
$$\sqrt{v_{t, i}} \approx \text{RMS}[g_i]$$
Thus, the step size along coordinate $i$ is normalized by its recent magnitude:
$$\Delta \theta_{t, i} \approx -\alpha \cdot \frac{g_{t, i}}{\text{RMS}[g_{t, i}]} \approx -\alpha \cdot \text{sign}(g_{t, i})$$

---

### 3. Adam (Adaptive Moment Estimation)

Adam maintains running estimates of both the **first uncentered moment** (the mean $m_t$) and the **second uncentered moment** (the variance / energy $v_t$):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t \odot g_t)$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\alpha = 10^{-3}$.

#### Rigorous Derivation of Bias Corrections
Since $m_0 = \vec{0}$ and $v_0 = \vec{0}$, both vectors are heavily biased toward zero during the initial steps of training.

Expanding $m_t$ recursively:
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

Taking the mathematical expectation $\mathbb{E}[m_t]$ and assuming the true gradient expectation $\mathbb{E}[g_i] \approx g$ is locally stationary:
$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] = g (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$

The finite geometric series evaluates to:
$$\sum_{i=1}^t \beta_1^{t-i} = \sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$$

Substituting this back:
$$\mathbb{E}[m_t] = g (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = g \cdot (1 - \beta_1^t)$$

Because $\mathbb{E}[m_t] \ne g$, $m_t$ is a **biased estimator** of the true gradient mean $g$. To obtain an **unbiased estimator** $\hat{m}_t$ such that $\mathbb{E}[\hat{m}_t] = g$, we must divide by the factor $(1 - \beta_1^t)$:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

By identical algebraic expansion on the second moment:
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

At step $t=1$ with $\beta_2 = 0.999$:
$$1 - \beta_2^1 = 1 - 0.999 = 0.001$$
Without bias correction, $v_1$ would be artificially scaled down by a factor of $1000$, causing $\frac{\alpha}{\sqrt{v_1}}$ to explode by a factor of $\sqrt{1000} \approx 31.6\times$! Bias correction eliminates this initial explosion.

#### The Adam Parameter Update:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

---

### 4. AdamW (Decoupled Weight Decay)

In standard Stochastic Gradient Descent, $L_2$ regularization and weight decay are mathematically identical:
$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2$$
$$\nabla_\theta \mathcal{L}_{\text{reg}}(\theta) = \nabla \mathcal{L}(\theta) + \lambda \theta$$
$$\theta_{t+1} = \theta_t - \alpha (\nabla \mathcal{L}(\theta_t) + \lambda \theta_t) = \underbrace{(1 - \alpha \lambda) \theta_t}_{\text{weight decay}} - \underbrace{\alpha \nabla \mathcal{L}(\theta_t)}_{\text{loss gradient}}$$

#### The Fatal Flaw of $L_2$ Regularization in Adam
When $L_2$ regularization is naively plugged into Adam, the regularizer gradient $\lambda \theta_t$ is added directly into $g_t$:
$$g_t = \nabla \mathcal{L}(\theta_t) + \lambda \theta_t$$
The second moment $v_t$ now accumulates the squared magnitude of this combined vector:
$$v_{t, i} \sim \mathbb{E}[(g_{t, i} + \lambda \theta_{t, i})^2]$$

The resulting parameter update becomes:
$$\theta_{t+1, i} = \theta_{t, i} - \frac{\alpha}{\sqrt{\hat{v}_{t, i}} + \epsilon} \left( \hat{m}_{t, i}(\nabla \mathcal{L}) + \lambda \theta_{t, i} \right)$$

Look at the effective regularization applied to coordinate $i$:
$$\text{Effective Weight Decay}_i = \frac{\alpha \lambda}{\sqrt{\hat{v}_{t, i}} + \epsilon} \theta_{t, i}$$

This produces an **inverted, catastrophic regularization penalty**:
- **Weights with large, frequent gradients**: $\hat{v}_{t, i}$ is huge $\implies \frac{\alpha \lambda}{\sqrt{\hat{v}_{t, i}}}$ shrinks to near zero. *Weights that need regularization the most receive almost NO decay!*
- **Weights with small, rare gradients**: $\hat{v}_{t, i}$ is tiny $\implies \frac{\alpha \lambda}{\sqrt{\hat{v}_{t, i}}}$ becomes massive. *Rare weights are aggressively crushed toward zero!*

#### The AdamW Solution (Loshchilov & Hutter, 2017)
AdamW restores proper regularization by **decoupling** weight decay from the gradient moments:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla \mathcal{L}(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla \mathcal{L}(\theta_t) \odot \nabla \mathcal{L}(\theta_t))$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \underbrace{\theta_t - \alpha \lambda_{\text{decay}} \theta_t}_{\text{Decoupled Weight Decay}} - \underbrace{\frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t}_{\text{Standard Adam Gradient Step}}$$

Every parameter shrinks by the exact same proportion $(1 - \alpha \lambda_{\text{decay}})$, regardless of the gradient scale along that coordinate!

---

## Part 3: Geometric & Coordinate-Wise Rescaling Interpretation

### 1. Diagonal Preconditioning
Adam can be viewed geometrically as a **diagonal preconditioned gradient descent**:
$$\theta_{t+1} = \theta_t - \alpha P_t^{-1} \hat{m}_t$$
where the preconditioner is the diagonal matrix:
$$P_t = \text{diag}\left( \sqrt{\hat{v}_{t, 1}} + \epsilon, \dots, \sqrt{\hat{v}_{t, d}} + \epsilon \right)$$

- Geometrically, $P_t^{-1}$ reshapes the elliptical, ill-conditioned iso-contours into a spherical geometry where all axes have roughly equal effective curvature.
- The step vector along coordinate $i$ is bounded by:
  $$\left| \frac{\hat{m}_{t, i}}{\sqrt{\hat{v}_{t, i}} + \epsilon} \right| \le 1$$
- Therefore, the maximum step size along any coordinate is strictly capped by $\alpha$:
  $$|\Delta \theta_{t, i}| \lesssim \alpha$$

### 2. Scale Invariance
Suppose we rescale the loss function by a scalar $c > 0$: $\tilde{f}(\theta) = c \cdot f(\theta)$.
- The gradient scales by $c$: $\tilde{g}_t = c \cdot g_t$.
- The first moment scales by $c$: $\tilde{m}_t = c \cdot m_t$.
- The second moment scales by $c^2$: $\tilde{v}_t = c^2 \cdot v_t$.
- The square root scales by $c$: $\sqrt{\tilde{v}_t} = c \cdot \sqrt{v_t}$.
- In the Adam step:
  $$\frac{\tilde{m}_t}{\sqrt{\tilde{v}_t}} = \frac{c \cdot m_t}{c \cdot \sqrt{v_t}} = \frac{m_t}{\sqrt{v_t}}$$

**Adam's step size is completely invariant to global gradient rescaling!** Whether gradients are $10^{-6}$ or $10^4$, Adam automatically takes steps of magnitude $\sim \alpha$.

---

## Part 4: Real-World Analogy

### 1. The Adaptive Personal Trainer vs. The Drill Sergeant
Imagine a group fitness bootcamp with runners of vastly different physical builds:
- **Standard GD is a Drill Sergeant with a megaphone**: He shouts *"Everyone run forward at 10 miles per hour!"* The lightweight sprinter can easily handle it, but the heavyweight lifter pulls a muscle and collapses (gradient explosion), while the elite marathoner is held back (slow convergence).
- **AdaGrad is a Trainer with a strict lifetime fatigue rule**: He measures total miles run since day one. Over time, every runner accumulates so much recorded mileage that the trainer forces everyone to crawl at 0.01 mph, halting all fitness progress prematurely.
- **Adam is an AI Biometric Smartwatch on every athlete**:
  - It tracks instantaneous momentum ($m_t$) and recent heart-rate / fatigue ($v_t$).
  - For an athlete facing a steep hill (low gradient, rare feature), it increases assistance.
  - For an athlete sprinting down an icy cliff (high gradient, steep ravine), it applies regenerative brakes.
- **AdamW is the same Smartwatch plus a universal caloric deficit**: Regardless of whether a runner ran uphill or downhill, every runner burns off a fixed fraction of body fat ($\lambda \theta_t$) at the end of every day, completely decoupled from momentary heart rate.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us perform a complete, hand-calculated trace of **Adam** and **AdamW** over 2 iterations on a 2-parameter model experiencing a **$100\times$ gradient disparity**.

### Setup & Hyperparameters
- Initial parameters: $\theta_0 = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$
- Initial moments: $m_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$, $v_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$
- Hyperparameters: $\alpha = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.99$, $\epsilon = 10^{-6}$, weight decay $\lambda = 0.05$
- Gradient inputs:
  - Step 1: $g_1 = \begin{bmatrix} 10.0 \\ 0.1 \end{bmatrix}$ (Coordinate 1 is $100\times$ larger than Coordinate 2)
  - Step 2: $g_2 = \begin{bmatrix} 10.0 \\ 0.2 \end{bmatrix}$

---

### Step-by-Step Manual Calculations

#### Iteration 1 ($t = 1$):
1. **Raw First Moment ($m_1$)**:
   $$m_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = 0.9 \begin{bmatrix} 0 \\ 0 \end{bmatrix} + 0.1 \begin{bmatrix} 10.0 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 1.0000 \\ 0.0100 \end{bmatrix}$$
2. **Raw Second Moment ($v_1$)**:
   $$g_1^2 = \begin{bmatrix} 100.00 \\ 0.01 \end{bmatrix}$$
   $$v_1 = \beta_2 v_0 + (1 - \beta_2) g_1^2 = 0.99 \begin{bmatrix} 0 \\ 0 \end{bmatrix} + 0.01 \begin{bmatrix} 100.00 \\ 0.01 \end{bmatrix} = \begin{bmatrix} 1.0000 \\ 0.0001 \end{bmatrix}$$
3. **Bias Corrections ($1 - \beta_1^1 = 0.1$, $1 - \beta_2^1 = 0.01$)**:
   $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{1}{0.1} \begin{bmatrix} 1.0000 \\ 0.0100 \end{bmatrix} = \begin{bmatrix} 10.0000 \\ 0.1000 \end{bmatrix}$$
   $$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{1}{0.01} \begin{bmatrix} 1.0000 \\ 0.0001 \end{bmatrix} = \begin{bmatrix} 100.0000 \\ 0.0100 \end{bmatrix}$$
   *(Notice: At step 1, bias correction restores $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$ perfectly!)*
4. **Adaptive Scaling Factor ($\sqrt{\hat{v}_1}$)**:
   $$\sqrt{\hat{v}_1} = \begin{bmatrix} \sqrt{100.0} \\ \sqrt{0.01} \end{bmatrix} = \begin{bmatrix} 10.0000 \\ 0.1000 \end{bmatrix}$$
5. **Gradient Step Direction ($\frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon}$)**:
   $$\frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = \begin{bmatrix} \frac{10.0}{10.0} \\ \frac{0.1}{0.1} \end{bmatrix} = \begin{bmatrix} 1.0000 \\ 1.0000 \end{bmatrix}$$
   *(Remarkable insight: Despite a $100\times$ difference in raw gradient, both coordinates take an equal, normalized step!)*
6. **Adam Update ($\lambda = 0$)**:
   $$\theta_1^{\text{Adam}} = \theta_0 - \alpha \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 1.0 - 0.10 \\ 1.0 - 0.10 \end{bmatrix} = \begin{bmatrix} 0.9000 \\ 0.9000 \end{bmatrix}$$
7. **AdamW Update ($\lambda = 0.05$, decoupled decay $(1 - \alpha \lambda) = 1 - 0.005 = 0.995$)**:
   $$\theta_1^{\text{AdamW}} = 0.995 \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} - 0.1 \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.995 - 0.10 \\ 0.995 - 0.10 \end{bmatrix} = \begin{bmatrix} 0.8950 \\ 0.8950 \end{bmatrix}$$

---

#### Iteration 2 ($t = 2$, with $g_2 = [10.0, 0.2]^T$):
1. **Raw First Moment ($m_2$)**:
   $$m_2 = 0.9 \begin{bmatrix} 1.0 \\ 0.01 \end{bmatrix} + 0.1 \begin{bmatrix} 10.0 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 0.90 + 1.00 \\ 0.009 + 0.020 \end{bmatrix} = \begin{bmatrix} 1.9000 \\ 0.0290 \end{bmatrix}$$
2. **Raw Second Moment ($v_2$)**:
   $$g_2^2 = \begin{bmatrix} 100.00 \\ 0.04 \end{bmatrix}$$
   $$v_2 = 0.99 \begin{bmatrix} 1.0000 \\ 0.0001 \end{bmatrix} + 0.01 \begin{bmatrix} 100.00 \\ 0.04 \end{bmatrix} = \begin{bmatrix} 0.99 + 1.00 \\ 0.000099 + 0.0004 \end{bmatrix} = \begin{bmatrix} 1.9900 \\ 0.000499 \end{bmatrix}$$
3. **Bias Corrections ($1 - 0.9^2 = 0.19$, $1 - 0.99^2 = 0.0199$)**:
   $$\hat{m}_2 = \frac{1}{0.19} \begin{bmatrix} 1.9000 \\ 0.0290 \end{bmatrix} = \begin{bmatrix} 10.0000 \\ 0.15263 \end{bmatrix}$$
   $$\hat{v}_2 = \frac{1}{0.0199} \begin{bmatrix} 1.9900 \\ 0.000499 \end{bmatrix} = \begin{bmatrix} 100.0000 \\ 0.025075 \end{bmatrix}$$
4. **Adaptive Denominator ($\sqrt{\hat{v}_2}$)**:
   $$\sqrt{\hat{v}_2} = \begin{bmatrix} 10.0000 \\ \sqrt{0.025075} \end{bmatrix} = \begin{bmatrix} 10.0000 \\ 0.15835 \end{bmatrix}$$
5. **Gradient Step Direction**:
   $$\frac{\hat{m}_2}{\sqrt{\hat{v}_2}} = \begin{bmatrix} \frac{10.0000}{10.0000} \\ \frac{0.15263}{0.15835} \end{bmatrix} = \begin{bmatrix} 1.0000 \\ 0.96387 \end{bmatrix}$$
6. **Adam Update**:
   $$\theta_2^{\text{Adam}} = \begin{bmatrix} 0.9000 \\ 0.9000 \end{bmatrix} - 0.1 \begin{bmatrix} 1.0000 \\ 0.96387 \end{bmatrix} = \begin{bmatrix} 0.8000 \\ 0.80361 \end{bmatrix}$$
7. **AdamW Update**:
   $$\theta_2^{\text{AdamW}} = 0.995 \begin{bmatrix} 0.8950 \\ 0.8950 \end{bmatrix} - 0.1 \begin{bmatrix} 1.0000 \\ 0.96387 \end{bmatrix} = \begin{bmatrix} 0.890525 - 0.1000 \\ 0.890525 - 0.096387 \end{bmatrix} = \begin{bmatrix} 0.79053 \\ 0.79414 \end{bmatrix}$$

---

### Comparative Visual Grid: Coordinate 1 vs. Coordinate 2

```
+--------------------------------------------------------------------------------------------------------+
|                                    ITERATION 1 (Raw Gradients: g_1 = [10.0, 0.1])                      |
+-------------------+--------------------+--------------------+--------------------+---------------------+
| Coordinate        | Raw Moment m_1     | Raw Moment v_1     | Bias-Corrected     | Final Step Delta    |
|                   |                    |                    | [m_hat, v_hat]     | (alpha * m / sqrt(v)|
+-------------------+--------------------+--------------------+--------------------+---------------------+
| Coord 1 (High)    | 1.0000             | 1.0000             | [10.0000, 100.000] | -0.1000             |
| Coord 2 (Low)     | 0.0100             | 0.0001             | [ 0.1000,   0.010] | -0.1000             |
+-------------------+--------------------+--------------------+--------------------+---------------------+
| CONCLUSION: Despite 100x gradient gap, BOTH coordinates take an identical step of -0.1000!           |
+--------------------------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------------------------+
|                                    ITERATION 2 (Raw Gradients: g_2 = [10.0, 0.2])                      |
+-------------------+--------------------+--------------------+--------------------+---------------------+
| Coordinate        | Raw Moment m_2     | Raw Moment v_2     | Bias-Corrected     | Final Step Delta    |
|                   |                    |                    | [m_hat, v_hat]     | (alpha * m / sqrt(v)|
+-------------------+--------------------+--------------------+--------------------+---------------------+
| Coord 1 (High)    | 1.9000             | 1.9900             | [10.0000, 100.000] | -0.1000             |
| Coord 2 (Low)     | 0.0290             | 0.000499           | [ 0.1526,  0.0251] | -0.0964             |
+-------------------+--------------------+--------------------+--------------------+---------------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning & LLM Implementation |
| :--- | :--- | :--- | :--- |
| $\theta_t$ | $\mathbb{R}^d$ vector | Model parameter state at iteration $t$ | Neural network weights/biases (`p.data`) |
| $g_t$ | $\mathbb{R}^d$ vector | Instantaneous loss gradient $\nabla_\theta \mathcal{L}_t$ | Backpropagated gradient tensor (`p.grad`) |
| $m_t$ | $\mathbb{R}^d$ vector | Exponential moving average of first moment | Momentum buffer (`state['exp_avg']`) |
| $v_t$ | $\mathbb{R}^d$ vector | Exponential moving average of uncentered variance | Squared gradient buffer (`state['exp_avg_sq']`) |
| $\beta_1$ | $[0, 1)$ scalar | First-moment decay factor (default: $0.9$) | Hyperparameter `betas[0]` |
| $\beta_2$ | $[0, 1)$ scalar | Second-moment decay factor (default: $0.999$) | Hyperparameter `betas[1]` |
| $\hat{m}_t, \hat{v}_t$ | $\mathbb{R}^d$ vectors | Bias-corrected moment estimates | Moments scaled by $1/(1-\beta_1^t)$ and $1/(1-\beta_2^t)$ |
| $\epsilon$ | $\mathbb{R}_{>0}$ scalar | Numerical stabilization term (default: $10^{-8}$) | Prevents division by zero (`eps=1e-8`) |
| $\lambda_{\text{decay}}$ | $\mathbb{R}_{\ge 0}$ scalar | Decoupled weight decay coefficient | Hyperparameter `weight_decay=0.01` in AdamW |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Why Adam Fails Without Bias Correction
**Statement**: Suppose $\beta_2 = 0.999$, $\alpha = 0.001$, and a constant gradient $g_t = 1.0$ is observed for all $t$.
1. Compute the raw second moment $v_1$ at step $t=1$.
2. Compute the uncorrected parameter update $\Delta \theta_1^{\text{raw}} = \frac{\alpha}{\sqrt{v_1}} m_1$ (assume $m_1 = 0.1$).
3. Compute the bias-corrected update $\Delta \theta_1^{\text{corrected}}$.
4. Evaluate the ratio $\Delta \theta_1^{\text{raw}} / \Delta \theta_1^{\text{corrected}}$.

**Solution**:
1. **Raw second moment**:
   $$v_1 = (1 - \beta_2) g_1^2 = (1 - 0.999) \cdot (1.0)^2 = 0.001 \cdot 1.0 = 0.001$$
2. **Uncorrected step**:
   $$\Delta \theta_1^{\text{raw}} = \frac{\alpha}{\sqrt{v_1}} m_1 = \frac{0.001}{\sqrt{0.001}} \cdot 0.1 = \frac{0.001}{0.0316228} \cdot 0.1 = 0.0316228 \cdot 0.1 = \mathbf{0.003162}$$
3. **Bias-corrected step**:
   $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.1}{0.1} = 1.0$$
   $$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{0.001}{1 - 0.999} = \frac{0.001}{0.001} = 1.0$$
   $$\Delta \theta_1^{\text{corrected}} = \frac{\alpha}{\sqrt{\hat{v}_1}} \hat{m}_1 = \frac{0.001}{\sqrt{1.0}} \cdot 1.0 = \mathbf{0.001000}$$
4. **Ratio**:
   $$\frac{\Delta \theta_1^{\text{raw}}}{\Delta \theta_1^{\text{corrected}}} = \frac{0.003162}{0.001000} = \mathbf{3.162\times}$$
   If $m_1$ had also not been scaled by $(1-\beta_1)$, the uncorrected update would be $\frac{0.001}{\sqrt{0.001}} \cdot 1.0 \approx 0.03162$—a **$31.6\times$ overshoot**!
   Without bias correction, initial gradient steps violently destabilize neural network initialization, triggering NaN activations or catastrophic early weight divergence.

---

### Problem 2: AdaGrad Learning Rate Starvation
**Statement**: In AdaGrad, let $g_t = 1.0$ be a steady gradient at each step $t$, with base learning rate $\alpha = 1.0$.
1. Derive the closed-form effective learning rate $\alpha_t^{\text{eff}}$ as a function of iteration $t$.
2. Compute $\alpha_t^{\text{eff}}$ at $t = 1, 100, 10{,}000, 1{,}000{,}000$.
3. Compute the cumulative distance traveled $\sum_{t=1}^T \alpha_t^{\text{eff}}$. Does it diverge or converge?

**Solution**:
1. In AdaGrad:
   $$G_t = \sum_{\tau=1}^t g_\tau^2 = \sum_{\tau=1}^t 1.0 = t$$
   $$\alpha_t^{\text{eff}} = \frac{\alpha}{\sqrt{G_t}} = \frac{1}{\sqrt{t}}$$
2. Numerical values:
   - $t = 1$: $\alpha_1^{\text{eff}} = \frac{1}{\sqrt{1}} = \mathbf{1.0}$
   - $t = 100$: $\alpha_{100}^{\text{eff}} = \frac{1}{\sqrt{100}} = \mathbf{0.10}$
   - $t = 10{,}000$: $\alpha_{10^4}^{\text{eff}} = \frac{1}{100} = \mathbf{0.01}$
   - $t = 1{,}000{,}000$: $\alpha_{10^6}^{\text{eff}} = \frac{1}{1000} = \mathbf{0.001}$
3. Cumulative distance traveled:
   $$\sum_{t=1}^T \frac{1}{\sqrt{t}} \approx \int_1^T t^{-1/2} \, dt = 2(\sqrt{T} - 1) = \mathcal{O}(\sqrt{T})$$
   Although the series diverges as $T \to \infty$, the incremental step size $\frac{1}{\sqrt{t}}$ decreases so rapidly that late-stage training makes virtually zero progress. If the model enters a complex loss valley late in training, AdaGrad is completely incapable of adapting.

---

### Problem 3: The Adam Non-Convergence Proof (Reddi, Kale, Kumar 2018)
**Statement**: Kingma & Ba (2014) originally claimed that Adam converges for all convex optimization problems. In 2018, Reddi et al. proved this claim is false by providing an explicit 1D convex counterexample. Explain the mathematical mechanism of this failure and the solution (AMSGrad).

**Explanation**:
1. **The Mechanism**:
   Adam's second-moment update is $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$.
   Because $\beta_2 < 1$, when a large gradient is followed by several small gradients, $v_t$ **decreases** exponentially:
   $$v_t < v_{t-1} \implies \frac{1}{\sqrt{v_t}} > \frac{1}{\sqrt{v_{t-1}}}$$
   This means the effective learning rate can actually **increase**! Reddi et al. constructed a periodic 1D sequence of gradients where occasional large gradients point toward the optimum, but frequent small gradients point away. Because $v_t$ "forgets" the large correct gradients, the optimizer takes larger steps in the wrong direction and diverges.
2. **The AMSGrad Fix**:
   To guarantee that the effective learning rate is monotonically non-increasing, AMSGrad enforces a maximum over historical second moments:
   $$\hat{v}_t^{\max} = \max(\hat{v}_{t-1}^{\max}, \hat{v}_t)$$
   $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t^{\max}} + \epsilon} \hat{m}_t$$
   This restores the theoretical convergence guarantee for general convex functions while retaining adaptive scaling.

---

## Part 7: Deep Learning Connection & Application

### 1. Why AdamW is the Undisputed Standard for LLMs
Every modern Large Language Model—including **GPT-4, LLaMA 3, Claude, Gemini, Mistral, and DeepSeek**—is trained using **AdamW**.
- **The Decoupling Impact**: Transformers contain hundreds of billions of parameters spanning multi-head attention projections, feed-forward layers, and vocabulary embeddings. Vocabulary embeddings for rare words receive sparse gradients. Under regular Adam ($L_2$ penalty), these rare token embeddings receive monstrously high weight decay penalties, degrading language fluency. AdamW applies uniform weight decay across all parameters, preserving vocabulary representations.
- **Typical Hyperparameter Recipes for LLM Pretraining**:
  - Learning rate: $\alpha \in [1 \times 10^{-4}, 6 \times 10^{-4}]$ (scaled inversely with model parameter count: $\alpha \propto 1/\sqrt{d_{\text{model}}}$).
  - First moment: $\beta_1 = 0.9$.
  - Second moment: $\beta_2 = 0.95$ (used by LLaMA and GPT-3) instead of the default $0.999$, allowing faster tracking of non-stationary loss landscapes.
  - Epsilon: $\epsilon = 10^{-8}$ (for fp32/bf16) or $\epsilon = 10^{-6}$ (to prevent underflow in pure fp16).
  - Weight decay: $\lambda = 0.1$.

### 2. The Absolute Necessity of Learning Rate Warmup
At step $t=1$, the parameter gradients $g_1$ are computed from random initial weights and contain high variance. If a full learning rate $\alpha$ is used immediately, the momentum buffers $m_t$ become contaminated with random exploratory noise, warping the model's initialization.
A **warmup schedule** ramps $\alpha_t$ linearly from $0$ to $\alpha_{\max}$ over the first $N_{\text{warmup}}$ steps (typically $1\%$ to $5\%$ of total steps):
$$\alpha_t = \alpha_{\max} \cdot \min\left(1, \frac{t}{N_{\text{warmup}}}\right)$$
This ensures that the adaptive preconditioner $\sqrt{v_t}$ and momentum $m_t$ stabilize before the model takes large steps.

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `05_optimization/code/06_adaptive_learning_rates.py`. It implements:
1. Exact visual grid verification of Adam and AdamW across 2 iterations against hand-computed values.
2. Verification of bias correction factors $\hat{m}_t, \hat{v}_t$ eliminating early-step variance artifacts.
3. Strict parity check between scratch Adam/AdamW implementations and PyTorch's native `torch.optim.Adam` and `torch.optim.AdamW` to float64 machine precision ($< 10^{-7}$).
4. An empirical demonstration comparing Adam vs. AdamW on disparate gradient scales, proving that decoupled weight decay applies uniform shrinkage.

Save the code and run it directly in Python 3.
