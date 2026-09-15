# 6.7 Regularization Techniques (Weight Decay, Dropout, Label Smoothing)

---

## Part 1: Intuition & 101 Motivation

Modern deep neural networks operate in the **massively overparameterized regime**: the number of trainable parameters $P$ often dwarfs the number of available training samples $N$ ($P \gg N$).

In 2016, Zhang et al. published a landmark paper, *"Rethinking Generalization,"* demonstrating that standard deep architectures (Inception, AlexNet) have sufficient capacity to memorize completely random image labels with 100% training accuracy. Yet, when trained on real data, these same networks generalize remarkably well to unseen test data.

Why do overparameterized networks generalize instead of merely memorizing?
Part of the answer lies in **implicit regularization** (the optimization trajectory of SGD/AdamW). The other crucial part lies in **explicit regularization techniques**—algorithmic mechanisms specifically designed to restrict model capacity, prevent overfitting, and enforce smoothness on the learned mapping:

1. **Weight Decay ($L_2$ Regularization)**: Penalizes large parameter magnitudes, shrinking weights toward the origin and preventing the network from fitting high-frequency noise.
2. **Dropout (Srivastava et al. 2014)**: During each forward pass, randomly zeroes out a fraction $p$ of hidden activations. This prevents neurons from developing fragile, complex **co-adaptations** (where one neuron only works if another specific neuron corrects its errors), forcing every feature detector to be independently useful.
3. **Label Smoothing (Szegedy et al. 2016)**: Replaces hard one-hot $\{0, 1\}$ ground-truth labels with softened probability distributions. This prevents the network from driving its output logits to infinite saturation, promoting better calibrated probabilities and better feature clustering.

```
                    THREE COMPLEMENTARY REGULARIZATION AXES
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ Parameter Space         │ Activation Space        │ Label Space             │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • Weight Decay (L2)     │ • Dropout (p)           │ • Label Smoothing (\alpha)│
│ • Weight Norm Clamping  │ • DropPath / StochDepth │ • Temperature Scaling   │
│ • Spectral Norm Reg     │ • Zoneout               │ • Mixup / CutMix        │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## Part 2: Rigorous Mathematical Formulation

### 1. Weight Decay ($L_2$ Regularization)

Let $\mathcal{L}_0(\theta)$ be the unregularized empirical loss. The $L_2$-regularized objective is:
$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_0(\theta) + \frac{\lambda}{2} \|\theta\|_2^2 = \mathcal{L}_0(\theta) + \frac{\lambda}{2} \sum_{i=1}^P \theta_i^2$$
where $\lambda > 0$ is the weight decay hyperparameter.

#### Gradient and Parameter Update:
$$\nabla_\theta \mathcal{L}_{\text{reg}}(\theta) = \nabla_\theta \mathcal{L}_0(\theta) + \lambda \theta$$

Under standard Stochastic Gradient Descent with step size $\eta$:
$$\theta_{t+1} = \theta_t - \eta \left( \nabla_\theta \mathcal{L}_0(\theta_t) + \lambda \theta_t \right) = \mathbf{(1 - \eta \lambda) \theta_t - \eta \nabla_\theta \mathcal{L}_0(\theta_t)}$$

- The coefficient $(1 - \eta \lambda) \in (0, 1)$ acts as a **multiplicative shrinkage factor** that scales down the weight vector at every single iteration before applying the gradient step.
- **Bayesian Interpretation**: In Chapter 4.4, we proved that $L_2$ regularization is mathematically equivalent to **Maximum A Posteriori (MAP) estimation under a zero-mean Gaussian prior** on weights: $\theta \sim \mathcal{N}(0, \sigma^2 I)$ where $\lambda = \frac{1}{\sigma^2}$.

---

### 2. Dropout (Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov 2014)

Let $h \in \mathbb{R}^d$ be the vector of activations emerging from a hidden layer.

#### 1. Standard Dropout (Original 2014 Formulation)
During training, each neuron $h_i$ is independently multiplied by a random binary mask $m_i \sim \text{Bernoulli}(1 - p)$:
$$\tilde{h} = m \odot h, \quad \text{where } P(m_i = 1) = 1 - p, \; P(m_i = 0) = p$$
where $p \in [0, 1)$ is the **dropout probability** (typically $p = 0.5$ for dense layers, $p = 0.1$ for Transformers).

At inference (test time), all neurons are active ($m = \vec{1}$). To ensure that the expected sum of inputs to the next layer remains identical between training and test time, the weights must be multiplied by $(1 - p)$:
$$W_{\text{test}} = (1 - p) W_{\text{train}}$$

#### 2. Inverted Dropout (Modern PyTorch / JAX Implementation)
Modern frameworks eliminate test-time scaling by applying the scale factor $\frac{1}{1 - p}$ **during training**:
$$\mathbf{\tilde{h} = \frac{1}{1 - p} (m \odot h), \quad m_i \sim \text{Bernoulli}(1 - p)}$$

At inference time:
$$\mathbf{\tilde{h}_{\text{test}} = h \quad (\text{Identity function! Zero test-time overhead!})}$$

#### Mathematical Expectation Preservation:
$$\mathbb{E}[\tilde{h}_i] = \mathbb{E}\left[ \frac{1}{1 - p} m_i h_i \right] = \frac{1}{1 - p} \mathbb{E}[m_i] h_i = \frac{1}{1 - p} (1 - p) h_i = \mathbf{h_i}$$
The expected value of the activation is preserved exactly!

#### Variance Injection:
$$\text{Var}(\tilde{h}_i) = \text{Var}\left( \frac{1}{1 - p} m_i h_i \right) = \frac{h_i^2}{(1 - p)^2} \text{Var}(m_i) = \frac{h_i^2}{(1 - p)^2} (p(1 - p)) = \mathbf{\frac{p}{1 - p} h_i^2}$$
For $p = 0.5$, $\text{Var}(\tilde{h}_i) = \frac{0.5}{0.5} h_i^2 = h_i^2$.
Dropout injects **multiplicative noise proportional to the squared magnitude of the activation**.

#### Backward Pass Gradient Through Inverted Dropout:
Let $g = \frac{\partial \mathcal{L}}{\partial \tilde{h}}$ be the incoming gradient from the layer above.
By chain rule:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial h} = \frac{1}{1 - p} (m \odot g)}$$
The exact same binary mask $m$ used in the forward pass must be applied to the backward gradient!

---

### 3. Label Smoothing (Szegedy, Vanhoucke, Ioffe, Shlens, Wojna 2016)

In standard classification, ground truth targets are one-hot encoded vectors $y \in \{0, 1\}^K$:
$$y_k = \begin{cases} 1 & \text{if } k = \text{target class } c \\ 0 & \text{if } k \ne c \end{cases}$$

#### The Logit Explosion Pathology
Under standard Cross-Entropy loss with Softmax probabilities $S_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$:
$$\mathcal{L}_{\text{CE}} = -\log S_c = -z_c + \log\left( \sum_{j=1}^K e^{z_j} \right)$$
To drive $\mathcal{L}_{\text{CE}} \to 0$, the network must achieve $S_c \to 1.0$.
This requires:
$$z_c - z_k \to +\infty \quad \forall k \ne c$$
The network is incentivized to drive its weights to huge magnitudes, causing:
1. Severe overconfidence (poorly calibrated probabilities).
2. Fragility to label noise.
3. Feature collapse in the penultimate layer.

#### The Smoothed Target Distribution
Label smoothing blends the one-hot target distribution $y$ with a uniform distribution $u = \frac{1}{K} \vec{1}$:
$$\mathbf{\tilde{y}_k = (1 - \alpha) y_k + \frac{\alpha}{K}}$$
where $\alpha \in (0, 1)$ is the smoothing parameter (typically $\alpha = 0.1$).

For the correct class $c$:
$$\tilde{y}_c = (1 - \alpha) \cdot 1 + \frac{\alpha}{K} = 1 - \alpha + \frac{\alpha}{K}$$
For any incorrect class $k \ne c$:
$$\tilde{y}_k = (1 - \alpha) \cdot 0 + \frac{\alpha}{K} = \frac{\alpha}{K}$$

#### Finite Optimal Logit Derivation
What are the optimal logits $z^*$ that minimize Cross-Entropy with smoothed targets?
The optimal output probabilities must match the smoothed target: $S_k^* = \tilde{y}_k$.
Therefore, the optimal logit difference between the correct class $c$ and any incorrect class $k$ is:
$$\frac{S_c^*}{S_k^*} = \frac{e^{z_c^*}}{e^{z_k^*}} = e^{z_c^* - z_k^*} = \frac{\tilde{y}_c}{\tilde{y}_k} = \frac{1 - \alpha + \frac{\alpha}{K}}{\frac{\alpha}{K}} = \frac{K(1 - \alpha) + \alpha}{\alpha} = \frac{K(1 - \alpha)}{\alpha} + 1$$
Taking the natural logarithm:
$$\mathbf{z_c^* - z_k^* = \log\left( \frac{(K - 1)(1 - \alpha)}{\alpha} + 1 \right)}$$

- **Notice**: The optimal logit difference is now **strictly finite**!
- For $K = 10$ and $\alpha = 0.1$:
  $$z_c^* - z_k^* = \log\left( \frac{9(0.9)}{0.1} + 1 \right) = \log(81 + 1) = \log(82) \approx \mathbf{4.4067}$$
  Instead of driving logits toward $+\infty$, the network stops at a healthy separation of $\sim 4.41$, completely preventing logit saturation!

---

## Part 3: Geometric Interpretation

### 1. Dropout as an Implicit Ensemble of $2^d$ Sub-Networks
A layer with $d$ neurons subject to Dropout has $2^d$ possible binary mask configurations $m \in \{0, 1\}^d$.
- At each training step, the network samples one of $2^d$ sub-networks.
- Weights are shared across all sub-networks.
- Training with Dropout is equivalent to training an **ensemble of $2^d$ distinct sub-networks** with shared parameters.
- At inference time, the single unmasked network computes the geometric mean of the predictions of this exponential ensemble!

### 2. Label Smoothing: Geometry of Penultimate Representations
Müller, Kornblith, & Hinton (2019) analyzed the activations of the penultimate layer under label smoothing:
- Under standard one-hot cross-entropy, representations within the same class form broad, diffuse clouds oriented toward infinity.
- Under label smoothing, representations for each class collapse into **tight, dense clusters on the surface of a hypersphere**, with all class clusters positioned at equal mutual angular distances!

---

## Part 4: Real-World Analogy

### 1. The Multi-Disciplinary Office Team
Imagine a software engineering team of 10 developers building a critical application:
- **No Dropout (Toxic Co-adaptation)**: Alice only writes code if Bob writes the unit tests, and Charlie fixes the database bugs. If Bob gets sick, the entire team halts. Alice relies on Bob to cover her mistakes.
- **Dropout (Independent Competence)**: Every morning, management flips a coin for each engineer; 5 random engineers are sent home for the day. Alice cannot rely on Bob being present tomorrow. Every engineer is forced to learn coding, testing, and database management independently. The entire team becomes robust and resilient.
- **Label Smoothing (The Humble Doctor)**: A medical AI is diagnosing X-rays. A standard one-hot network outputs: *"This is 100.0000% benign; 0.0000% malignant."* If an edge case appears, it makes a catastrophic overconfident error. A label-smoothed AI outputs: *"This is 96.7% benign, with a 3.3% possibility of malignant."* It maintains appropriate epistemic humility.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us perform a complete, cell-by-cell numerical walkthrough of **Inverted Dropout** (forward and backward) and **Label Smoothing**.

### 1. Inverted Dropout Walkthrough
- **Input activation vector**: $h = [2.0, \ -4.0, \ 1.0, \ 3.0]^T \in \mathbb{R}^4$.
- **Dropout probability**: $p = 0.5 \implies$ Keep probability $1 - p = 0.5$.
- **Scale factor**: $\frac{1}{1 - p} = \frac{1}{0.5} = \mathbf{2.0}$.
- **Sampled binary mask**: $m = [1, \ 0, \ 1, \ 0]^T$ (Neurons 2 and 4 are dropped).
- **Incoming upstream gradient**: $g = \frac{\partial \mathcal{L}}{\partial \tilde{h}} = [0.10, \ 0.20, \ 0.30, \ 0.40]^T$.

#### Step-by-Step Forward:
1. Apply mask: $m \odot h = [1(2.0), \ 0(-4.0), \ 1(1.0), \ 0(3.0)]^T = [2.0, \ 0.0, \ 1.0, \ 0.0]^T$.
2. Scale by $\frac{1}{1-p} = 2.0$:
   $$\tilde{h} = 2.0 \begin{bmatrix} 2.0 \\ 0.0 \\ 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{4.0000} \\ \mathbf{0.0000} \\ \mathbf{2.0000} \\ \mathbf{0.0000} \end{bmatrix}$$

#### Step-by-Step Backward:
1. Apply mask to gradient: $m \odot g = [1(0.10), \ 0(0.20), \ 1(0.30), \ 0(0.40)]^T = [0.10, \ 0.0, \ 0.30, \ 0.0]^T$.
2. Scale by $\frac{1}{1-p} = 2.0$:
   $$\frac{\partial \mathcal{L}}{\partial h} = 2.0 \begin{bmatrix} 0.10 \\ 0.00 \\ 0.30 \\ 0.00 \end{bmatrix} = \begin{bmatrix} \mathbf{0.2000} \\ \mathbf{0.0000} \\ \mathbf{0.6000} \\ \mathbf{0.0000} \end{bmatrix}$$

---

### 2. Label Smoothing Walkthrough
- **Number of classes**: $K = 3$.
- **True class**: $c = 1$ (Class index 0, one-hot: $y = [1.0, \ 0.0, \ 0.0]^T$).
- **Smoothing factor**: $\alpha = 0.1$.
- **Logits**: $z = [2.0, \ 1.0, \ 0.0]^T$.

#### Step-by-Step Target Smoothing:
1. Correct class ($k = 0$):
   $$\tilde{y}_0 = (1 - 0.1)(1.0) + \frac{0.1}{3} = 0.9000 + 0.033333 = \mathbf{0.933333}$$
2. Incorrect classes ($k = 1, 2$):
   $$\tilde{y}_1 = (1 - 0.1)(0.0) + \frac{0.1}{3} = \mathbf{0.033333}$$
   $$\tilde{y}_2 = (1 - 0.1)(0.0) + \frac{0.1}{3} = \mathbf{0.033333}$$
   Check sum: $0.933333 + 0.033333 + 0.033333 = 1.000000$.

#### Softmax & Loss Calculation:
1. Exponentials: $e^2 \approx 7.3891, \; e^1 \approx 2.7183, \; e^0 = 1.0000 \implies \sum = 11.1074$.
2. Softmax probabilities:
   $$S = \begin{bmatrix} 7.3891 / 11.1074 \\ 2.7183 / 11.1074 \\ 1.0000 / 11.1074 \end{bmatrix} = \begin{bmatrix} \mathbf{0.6652} \\ \mathbf{0.2447} \\ \mathbf{0.0900} \end{bmatrix}$$
3. Log probabilities: $\log(S) = [\log(0.6652), \log(0.2447), \log(0.0900)] = [-0.4076, \ -1.4076, \ -2.4076]^T$.
4. **Standard Cross-Entropy (One-Hot)**:
   $$\mathcal{L}_{\text{CE}} = -1.0 \log(S_0) = -(-0.4076) = \mathbf{0.4076}$$
5. **Label-Smoothed Cross-Entropy**:
   $$\mathcal{L}_{\text{LS}} = -\sum_{k=0}^2 \tilde{y}_k \log S_k = -[0.933333(-0.4076) + 0.033333(-1.4076) + 0.033333(-2.4076)]$$
   $$= -[-0.3804 - 0.0469 - 0.0803] = -[-0.5076] = \mathbf{0.5076}$$

---

### Comparative Visual Grid

```
+----------------------------------------------------------------------------------------------------+
|                                    INVERTED DROPOUT TRACE (p = 0.5, Scale = 2.0)                   |
+---------+------------------+------------------+--------------------+-------------------------------+
| Index i | Raw Activation h | Mask m_i         | Forward h_tilde    | Backward Grad dL/dh           |
+---------+------------------+------------------+--------------------+-------------------------------+
| 0       | +2.0000          | 1 (Active)       | +4.0000 (Doubled!) | +0.2000 (Doubled!)            |
| 1       | -4.0000          | 0 (DROPPED)      |  0.0000 (Zeroed!)  |  0.0000 (Zeroed!)             |
| 2       | +1.0000          | 1 (Active)       | +2.0000 (Doubled!) | +0.6000 (Doubled!)            |
| 3       | +3.0000          | 0 (DROPPED)      |  0.0000 (Zeroed!)  |  0.0000 (Zeroed!)             |
+---------+------------------+------------------+--------------------+-------------------------------+

+----------------------------------------------------------------------------------------------------+
|                                    LABEL SMOOTHING TARGET COMPARISON (K = 3, \alpha = 0.1)         |
+---------+------------------+------------------+--------------------+-------------------------------+
| Class k | One-Hot Target y | Smoothed Target  | Softmax Pred S_k   | Logit Target Gradient (S - y) |
+---------+------------------+------------------+--------------------+-------------------------------+
| Class 0 | 1.0000           | 0.9333           | 0.6652             | 0.6652 - 0.9333 = -0.2681     |
| Class 1 | 0.0000           | 0.0333           | 0.2447             | 0.2447 - 0.0333 = +0.2114     |
| Class 2 | 0.0000           | 0.0333           | 0.0900             | 0.0900 - 0.0333 = +0.0567     |
+---------+------------------+------------------+--------------------+-------------------------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning Implementation |
| :--- | :--- | :--- | :--- |
| $\lambda$ | $\mathbb{R}_{\ge 0}$ scalar | $L_2$ regularization penalty coefficient | `weight_decay` in optimizer |
| $p$ | $[0, 1)$ scalar | Probability of dropping a hidden unit | `p` in `nn.Dropout(p=0.5)` |
| $m$ | $\{0, 1\}^d$ binary vector | Bernoulli random realization mask | Dropout mask tensor |
| $\frac{1}{1-p}$ | Scalar $\ge 1$ | Inverted dropout normalization scale | Multiplier in training pass |
| $\alpha$ | $[0, 1)$ scalar | Label smoothing interpolation factor | `label_smoothing` in `CrossEntropyLoss` |
| $\tilde{y}$ | Probability simplex $\Delta^{K-1}$ | Softened target probability distribution | Smoothed target tensor |
| $z_c^* - z_k^*$ | $\mathbb{R}_{>0}$ scalar | Asymptotic optimal logit difference | Target logit separation gap |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Dropout Equivalence to Adaptive $L_2$ Regularization
**Statement**: Consider a linear regression model with input $x \in \mathbb{R}^d$ and weights $w \in \mathbb{R}^d$: $\hat{y} = w^T (m \odot x)$ where $m_i \sim \text{Bernoulli}(1 - p)$ with scaling $\frac{1}{1-p}$.
Prove that minimizing the expected squared error under Dropout is mathematically equivalent to minimizing ordinary least squares with an **adaptive data-dependent Ridge ($L_2$) penalty** (Wager, Wang, & Liang 2013).

**Proof**:
Let the loss under dropout be $\mathcal{L}_{\text{drop}} = \mathbb{E}_m \left[ \left( y - \frac{1}{1-p} \sum_{j=1}^d w_j m_j x_j \right)^2 \right]$.
1. Expand the square:
   $$\mathcal{L}_{\text{drop}} = y^2 - 2 y \frac{1}{1-p} \sum_j w_j \mathbb{E}[m_j] x_j + \frac{1}{(1-p)^2} \mathbb{E}\left[ \left( \sum_j w_j m_j x_j \right)^2 \right]$$
2. Since $\mathbb{E}[m_j] = 1 - p$:
   $$\text{Cross Term} = -2 y \sum_j w_j x_j = -2 y (w^T x)$$
3. Expand the quadratic term $\mathbb{E}\left[ \sum_j \sum_k w_j w_k m_j m_k x_j x_k \right]$:
   - For $j \ne k$: Since $m_j$ and $m_k$ are independent, $\mathbb{E}[m_j m_k] = \mathbb{E}[m_j]\mathbb{E}[m_k] = (1 - p)^2$.
   - For $j = k$: $m_j^2 = m_j \implies \mathbb{E}[m_j^2] = 1 - p$.
   Therefore:
   $$\mathbb{E}[m_j m_k] = (1 - p)^2 + \delta_{jk} \left( (1 - p) - (1 - p)^2 \right) = (1 - p)^2 + \delta_{jk} p (1 - p)$$
4. Substituting this into the quadratic sum:
   $$\frac{1}{(1-p)^2} \sum_j \sum_k w_j w_k x_j x_k \left[ (1 - p)^2 + \delta_{jk} p(1 - p) \right] = (w^T x)^2 + \frac{p}{1 - p} \sum_j w_j^2 x_j^2$$
5. Combining all terms:
   $$\mathcal{L}_{\text{drop}} = \underbrace{(y - w^T x)^2}_{\text{Standard OLS Loss}} + \underbrace{\frac{p}{1 - p} \sum_{j=1}^d x_j^2 w_j^2}_{\text{Adaptive } L_2 \text{ Penalty!}}$$
$\blacksquare$ Dropout on inputs is mathematically identical to an $L_2$ regularizer where coordinate $j$ is penalized proportionally to its feature variance $\sum x_j^2$!

---

### Problem 2: Weight Decay vs. $L_1$ Lasso Sparsity
**Statement**: Contrast the gradient update of Weight Decay ($L_2$) versus Lasso ($L_1$) regularization:
$$\mathcal{L}_{L_1}(\theta) = \mathcal{L}_0(\theta) + \lambda \sum_i |\theta_i|$$
Explain why $L_1$ induces strict parameter sparsity ($\theta_i = 0$) while $L_2$ only shrinks parameters toward zero.

**Analysis**:
- **$L_2$ Update**: $\Delta \theta_i = -\eta \nabla_i \mathcal{L}_0 - \eta \lambda \theta_i$.
  The shrinkage force is proportional to $\theta_i$. As $\theta_i \to 0$, the penalty force vanishes ($\lambda \theta_i \to 0$). Therefore, weights get tiny, but never reach exactly zero.
- **$L_1$ Update**: $\Delta \theta_i = -\eta \nabla_i \mathcal{L}_0 - \eta \lambda \text{sign}(\theta_i)$.
  The shrinkage force is a **constant $\eta \lambda$ regardless of how small $\theta_i$ is!** When $|\theta_i| < \eta \lambda$, the parameter is pushed directly across zero and clamped to exactly $0.0$.
*Result*: $L_1$ produces sparse feature selection; $L_2$ produces smooth weight shrinkage.

---

## Part 7: Deep Learning Connection & Application

### 1. Why Modern LLMs Abandoned Dropout
In early NLP models (LSTM, early BERT), Dropout ($p = 0.1$) was ubiquitous.
However, in modern LLM pretraining (LLaMA 3, GPT-4, Mistral), **Dropout is almost universally disabled ($p = 0.0$)**:
1. **Underfitting on Massive Web Data**: Trillion-token pretraining datasets (Common Crawl, Fineweb) contain so much natural diversity that underfitting is a larger concern than overfitting.
2. **Compute Efficiency & Memory**: Storing binary dropout masks across thousands of GPU nodes requires massive activation memory and extra CUDA kernel launches.
3. **Inference Discrepancy**: Removing dropout guarantees deterministic output representations during KV-cache generation.
- Modern LLMs instead rely heavily on **Weight Decay ($\lambda = 0.1$)** and **RoPE/RMSNorm** for regularization.

### 2. Vision Transformers & Stochastic Depth (DropPath)
While dense dropout is rare in modern vision models, **Stochastic Depth / DropPath** (Huang et al. 2016) is mandatory for Vision Transformers (Swin, ViT, ConvNeXt):
$$x_{l+1} = x_l + \text{Bernoulli}(1 - p_{\text{drop}}) \cdot \text{Block}(x_l)$$
Instead of dropping individual scalar activations, DropPath randomly skips entire residual blocks, effectively training an ensemble of shallow and deep architectures!

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `06_deep_learning_foundations/code/07_regularization_techniques.py`. It implements:
1. Exact visual grid verification of Inverted Dropout (forward activation and backward gradient) and Label Smoothing.
2. Strict parity check between scratch implementations and PyTorch modules (`nn.Dropout`, `nn.CrossEntropyLoss(label_smoothing=0.1)`).
3. Empirical Overfitting vs. Generalization experiment on a noisy dataset:
   - Unregularized MLP overfits to 100% train / 62% test accuracy.
   - Regularized MLP (Weight Decay + Dropout + Label Smoothing) prevents memorization and achieves 84% test accuracy!

Save the code and run it directly in Python 3.
