# Chapter 7.3: Classic to Modern ConvNets (LeNet, AlexNet, VGG, ResNet, ConvNeXt)

---

## 1. Intuition & 101 Motivation

For nearly three decades—from Yann LeCun's handwritten digit recognizer in 1998 to Zhuang Liu and Kaiming He's ConvNeXt in 2022—Convolutional Neural Networks (CNNs) have defined the architecture of computer vision.

Yet, building deeper and more powerful vision backbones was not merely a matter of stacking more layers on GPUs. The architectural progression was forged by overcoming severe mathematical bottlenecks:
1. **The Representation Gap (LeNet $\to$ AlexNet):** Shifting from small saturating activations (Sigmoid/Tanh) on low-resolution inputs to rectified linear units (ReLU), spatial dropout, and data augmentation capable of scaling to ImageNet-1k ($1.2$ million images).
2. **Receptive Field Factorization (AlexNet $\to$ VGG):** AlexNet used large $11 \times 11$ and $5 \times 5$ filters. VGG proved that stacking two $3 \times 3$ filters achieves the exact same $5 \times 5$ receptive field with **$28\%$ fewer parameters** and an extra non-linearity. Stacking three $3 \times 3$ filters matches a $7 \times 7$ receptive field with **$45\%$ fewer parameters**.
3. **The Degradation Problem (VGG $\to$ ResNet):** Prior to 2015, stacking more than 20–30 layers caused training error (not just test error) to increase drastically—even with BatchNorm preventing vanishing gradients. Kaiming He et al. diagnosed this not as optimization failure, but as representational hardness: fitting an identity mapping $\mathcal{H}(\mathbf{x}) = \mathbf{x}$ using non-linear layers $\mathcal{F}(\mathbf{x})$ is notoriously difficult. By reformulating the layer to learn a residual $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$, the network provides an unimpeded **gradient highway** ($\mathbf{x}_{L} = \mathbf{x}_l + \sum \mathcal{F}$), enabling training of 152 to 1,000+ layers.
4. **The Modern Renaissance (ResNet $\to$ ConvNeXt):** When Vision Transformers (ViTs) emerged in 2020 and outperformed CNNs, researchers questioned whether convolutions were obsolete. In 2022, ConvNeXt systematically modernized ResNet using ViT design principles—inverted bottlenecks, depthwise separable $7 \times 7$ convolutions, LayerNorm, and GELU—proving that standard pure CNNs match or surpass Swin Transformers at identical FLOP and throughput budgets.

---

## 2. Rigorous Mathematical Formulation

```
                                  CONVNET EVOLUTION (1998 - 2022)
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   LeNet-5    │      │   AlexNet    │      │    VGG-16    │      │  ResNet-50   │      │  ConvNeXt-T  │
│    (1998)    │ ───► │    (2012)    │ ───► │    (2014)    │ ───► │    (2015)    │ ───► │    (2022)    │
│  5x5 Conv    │      │  11x11, 5x5  │      │ Stacked 3x3  │      │ Residual     │      │ 7x7 Depthwise│
│  AvgPool     │      │  ReLU, LRN   │      │ Homogeneous  │      │ Shortcuts    │      │ Inverted     │
│  Tanh / Sigm │      │  Dropout 0.5 │      │ Deep (16-19) │      │ Bottlenecks  │      │ Bottleneck   │
│  ~60k Params │      │  ~60M Params │      │ ~138M Params │      │ ~25M Params  │      │ ~28M Params  │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
```

---

### 2.1 The Architectural Milestones

#### 1. LeNet-5 (LeCun et al., 1998)
Designed for $32 \times 32$ grayscale digit recognition (MNIST):
$$\mathbf{X} \in \mathbb{R}^{1 \times 32 \times 32} \xrightarrow{\text{Conv } 5 \times 5} \mathbf{C}_1 (6@28 \times 28) \xrightarrow{\text{Subsampling } 2 \times 2} \mathbf{S}_2 (6@14 \times 14) \xrightarrow{\text{Conv } 5 \times 5} \mathbf{C}_3 (16@10 \times 10) \dots$$
- **Subsampling mechanism:** $y = \tanh(w \cdot \text{average}(x) + b)$ where $w, b \in \mathbb{R}$ were learnable per-channel scales and biases.
- Total parameters: $\approx 60,000$.

#### 2. AlexNet (Krizhevsky, Sutskever, Hinton, 2012)
ImageNet breakthrough ($224 \times 224 \times 3$ RGB):
- First layer: $11 \times 11$ kernel, stride $4$ ($96$ filters).
- Second layer: $5 \times 5$ kernel, stride $1$ ($256$ filters).
- Subsequent layers: $3 \times 3$ kernels.
- **Local Response Normalization (LRN):** Lateral inhibition across adjacent filter maps $k$:
  $$b_{x, y}^i = \frac{a_{x, y}^i}{\left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x, y}^j)^2\right)^\beta}$$
  *(Later discarded across the field in favor of Batch Normalization).*
- Fully connected head: Two $4096$-dim dense layers with Dropout ($p=0.5$).
- Total parameters: $\approx 61$ million.

---

### 2.2 VGG & The Mathematical Principle of Receptive Field Factorization (Simonyan & Zisserman, 2014)

Why replace a single $k \times k$ convolution with a cascade of $3 \times 3$ convolutions?

Let an input feature map have $C$ channels, and consider maintaining $C$ channels throughout.

#### Theorem: Equivalence of Receptive Field
A stack of two $3 \times 3$ convolutional layers (stride $1$, padding $1$) has an effective receptive field of:
$$r_2 = r_1 + (k_2 - 1) = 3 + (3 - 1) = 5$$
identical to a single $5 \times 5$ convolution.
A stack of three $3 \times 3$ convolutional layers has an effective receptive field of:
$$r_3 = r_2 + (k_3 - 1) = 5 + (3 - 1) = 7$$
identical to a single $7 \times 7$ convolution.

#### Parameter Count Comparison:
- Single $5 \times 5$ conv:
  $$\text{Params}(5 \times 5) = 5 \times 5 \times C \times C = 25 C^2$$
- Two stacked $3 \times 3$ convs:
  $$\text{Params}(2 \times 3 \times 3) = 2 \times (3 \times 3 \times C \times C) = 18 C^2$$
  $$\text{Savings} = \frac{25 C^2 - 18 C^2}{25 C^2} = \mathbf{28\% \text{ reduction in parameters}}$$

- Single $7 \times 7$ conv:
  $$\text{Params}(7 \times 7) = 7 \times 7 \times C \times C = 49 C^2$$
- Three stacked $3 \times 3$ convs:
  $$\text{Params}(3 \times 3 \times 3) = 3 \times (3 \times 3 \times C \times C) = 27 C^2$$
  $$\text{Savings} = \frac{49 C^2 - 27 C^2}{49 C^2} = \mathbf{44.9\% \text{ reduction in parameters}}$$

#### Expressive Power Advantage:
Beyond parameter and FLOP savings, two stacked $3 \times 3$ layers incorporate **two non-linear activation functions** (e.g., $\text{ReLU} \circ \text{Conv} \circ \text{ReLU} \circ \text{Conv}$) instead of one, rendering the decision boundary strictly more expressive.

---

### 2.3 ResNet & The Residual Gradient Highway (He et al., 2015)

#### The Degradation Problem
When deeper networks (e.g., 56 layers vs. 20 layers) are trained without residual connections, the 56-layer network displays **higher training error** than the 20-layer network.
This is not overfitting (overfitting causes low training error and high test error). Nor is it vanishing gradients (BatchNorm and proper initialization guarantee healthy forward activations and non-zero gradient variances).

It indicates that deep stacks of standard non-linear transformations are **structurally difficult to optimize toward identity mappings**.

#### The Residual Formulation
Instead of tasking a cascade of stacked layers with approximating an underlying target mapping $\mathcal{H}(\mathbf{x})$, we explicitly split the mapping into:
$$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x}$$
where $\mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\})$ is the residual mapping to be learned:
$$\mathcal{F} \equiv \mathcal{H}(\mathbf{x}) - \mathbf{x}$$

If identity mapping were optimal, optimizing $\mathcal{F}(\mathbf{x}) \to \mathbf{0}$ by pushing weights $\mathbf{W}_i \to \mathbf{0}$ is trivially achievable with standard weight decay regularization, whereas driving stacked non-linear layers $\sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x})) \to \mathbf{x}$ is profoundly non-convex and difficult.

```
       Residual Block Architecture
              x ────┐
              │     │ (Identity Shortcut)
              ▼     │
         ┌────────┐ │
         │ Weight │ │
         └────────┘ │
              ▼     │
         ┌────────┐ │
         │  ReLU  │ │
         └────────┘ │
              ▼     │
         ┌────────┐ │
         │ Weight │ │
         └────────┘ │
              ▼     │
            F(x)    │
              │     │
              ▼     ▼
             [ + ] ◄┘  (Element-wise Addition)
              │
              ▼
             ReLU
              │
              ▼
         Output y = ReLU(F(x) + x)
```

---

#### Rigorous Derivation of the Gradient Highway
Consider a general deep residual network where each block $l$ computes:
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$
Recursively expanding this relation from block $l$ to any deeper block $L$ ($L > l$):
$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$$

Now, apply the vector chain rule to compute the gradient of the total scalar loss $\mathcal{E}$ with respect to the activation $\mathbf{x}_l$:
$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right)$$

Look carefully at the term inside the parentheses:
$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \underbrace{\frac{\partial \mathcal{E}}{\partial \mathbf{x}_L}}_{\text{Unimpeded Gradient Highway}} + \underbrace{\frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \left( \sum_{i=l}^{L-1} \frac{\partial \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)}{\partial \mathbf{x}_l} \right)}_{\text{Residual Modulation}}$$

**Mathematical Guarantees:**
1. The term $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_L}$ is transferred directly to $\mathbf{x}_l$ without being multiplied by weight matrices $\mathbf{W}$.
2. Even if the Jacobian term $\sum_{i=l}^{L-1} \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l}$ vanishes (due to small weights or saturating regions), the gradient $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l}$ **cannot vanish**, because the identity matrix $\mathbf{I}$ guarantees a floor on gradient flow.
3. The gradient cannot vanish for all mini-batch samples simultaneously unless $\sum \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l} = -\mathbf{I}$, which is an exceptionally rare pathological singularity in high dimensions.

---

### 2.4 BasicBlock vs. Bottleneck Block

1. **BasicBlock (ResNet-18 / ResNet-34):**
   - Two $3 \times 3$ convolutions: $\text{Conv}(3 \times 3, C) \to \text{BN} \to \text{ReLU} \to \text{Conv}(3 \times 3, C) \to \text{BN}$.
   - Add identity $\mathbf{x}$, then apply final $\text{ReLU}$.

2. **Bottleneck Block (ResNet-50 / 101 / 152):**
   - For high channel counts (e.g., $C = 256$), a $3 \times 3$ convolution is computationally expensive.
   - ResNet introduces a $1 \times 1 \to 3 \times 3 \to 1 \times 1$ sandwich:
     1. $1 \times 1 \text{ Conv}$ to reduce channels: $4C \to C$ (compression by $4\times$).
     2. $3 \times 3 \text{ Conv}$ operating on the bottleneck: $C \to C$.
     3. $1 \times 1 \text{ Conv}$ to restore channels: $C \to 4C$ (expansion by $4\times$).
   - **FLOP Comparison:**
     - Standard $3 \times 3$ on $4C$ channels: $9 \times (4C)^2 = 144 C^2$.
     - Bottleneck: $(1 \times 1 \times 4C \times C) + (3 \times 3 \times C \times C) + (1 \times 1 \times C \times 4C) = 4C^2 + 9C^2 + 4C^2 = \mathbf{17 C^2}$.
     - The bottleneck reduces computation by **$88.2\%$**!

---

### 2.5 ConvNeXt: Modernizing ConvNets for the 2020s (Liu et al., 2022)

Starting from a standard ResNet-50, ConvNeXt progressively adopts Vision Transformer design choices:

| Modification Step | ResNet Baseline | ConvNeXt Modernization | Justification / ViT Parity |
| :--- | :--- | :--- | :--- |
| **Macro Design** | Stage compute $(3, 4, 6, 3)$ | Stage compute $(3, 3, 9, 3)$ | Matches Swin-T compute distribution |
| **Stem Cell** | $7 \times 7, s=2 \text{ Conv} + \text{MaxPool}$ | $4 \times 4, s=4 \text{ Conv}$ | Non-overlapping patchification like ViT |
| **ResNeXt Strategy** | Standard dense conv | **Depthwise Conv ($7 \times 7$)** | Separates spatial filtering from channel mixing (like self-attention) |
| **Inverted Bottleneck** | Narrow bottleneck ($4C \to C \to 4C$) | **Inverted Bottleneck ($C \to 4C \to C$)** | Matches MLP block in Transformers |
| **Large Kernel Size** | $3 \times 3$ spatial kernels | **$7 \times 7$ depthwise kernels** | Expands effective receptive field |
| **Activation Choice** | ReLU after every conv | **GELU**, only *one* per block | Matches ViT Transformer block |
| **Normalization** | BatchNorm after every conv | **LayerNorm**, only *one* per block | Stabilizes training, removes batch statistics dependence |

```
    ResNet Bottleneck                    ConvNeXt Block
       Input (256-d)                      Input (96-d)
            │                                  │
      1x1 Conv, 64-d                     7x7 Depthwise Conv, 96-d
      BatchNorm + ReLU                         │
            │                              LayerNorm
      3x3 Conv, 64-d                           │
      BatchNorm + ReLU                    1x1 Conv, 384-d (4x expand)
            │                                  │
     1x1 Conv, 256-d                         GELU
        BatchNorm                              │
            │                             1x1 Conv, 96-d (project)
         [ + ] ◄── Identity Shortcut           │
            │                              LayerScale (gamma)
          ReLU                                 │
            │                               [ + ] ◄── Identity Shortcut
      Output (256-d)                           │
                                         Output (96-d)
```

---

### 2.6 Deep Derivation 7.3.1: Asymptotic Gradient Preservation and Spectral Analysis of Residual Networks

In this derivation, we prove why residual networks prevent exponential gradient decay and explosion, establishing the mathematical foundation of the skip-connection identity highway.

#### Step 1: Mathematical Formulation of Plain vs. Residual Architectures
Let a deep feedforward network consist of $L$ sequential layers with hidden state vectors $\mathbf{x}_l \in \mathbb{R}^{d}$ for $l \in \{0, 1, \dots, L\}$.
In a **Plain Network**, transitions are governed by pure non-linear composition:
$$\mathbf{x}_{l+1} = \sigma(\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l)$$
where $\mathbf{W}_l \in \mathbb{R}^{d \times d}$ and $\sigma: \mathbb{R} \to \mathbb{R}$ is an element-wise activation function.

In a **Residual Network**, transitions are governed by an affine residual mapping added to the identity:
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}_l(\mathbf{x}_l, \mathcal{W}_l)$$
where $\mathcal{F}_l(\mathbf{x}_l, \mathcal{W}_l) = \mathbf{W}_{l, 2} \, \sigma(\mathbf{W}_{l, 1} \mathbf{x}_l + \mathbf{b}_{l, 1}) + \mathbf{b}_{l, 2}$.

#### Step 2: Exponential Gradient Decay in Plain Networks
Let $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ be a scalar loss computed at layer $L$. In a Plain Network, applying the multivariable chain rule from layer $L$ to layer $l$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \prod_{k=l}^{L-1} \frac{\partial \mathbf{x}_{k+1}}{\partial \mathbf{x}_k} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \prod_{k=l}^{L-1} \left( \mathbf{D}_k \mathbf{W}_k \right)$$
where $\mathbf{D}_k = \operatorname{diag}(\sigma'(\mathbf{W}_k \mathbf{x}_k + \mathbf{b}_k)) \in \mathbb{R}^{d \times d}$ is the diagonal derivative matrix.

Taking the Euclidean operator norm (spectral norm $\|\cdot\|_2$):
$$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \right\|_2 \le \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\|_2 \prod_{k=l}^{L-1} \|\mathbf{D}_k\|_2 \|\mathbf{W}_k\|_2$$

Let $\lambda_{\max} = \sup_k \|\mathbf{D}_k \mathbf{W}_k\|_2$ denote the maximum singular value across layers:
1. **Vanishing Gradient Regime:** If $\lambda_{\max} < 1 - \epsilon$ for some $\epsilon > 0$:
   $$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \right\|_2 \le \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\|_2 (1 - \epsilon)^{L - l} \xrightarrow{L - l \to \infty} 0$$
   The gradient vanishes exponentially with depth $L - l$.
2. **Exploding Gradient Regime:** If $\lambda_{\min} > 1 + \epsilon$:
   $$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \right\|_2 \ge \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\|_2 (1 + \epsilon)^{L - l} \xrightarrow{L - l \to \infty} \infty$$

Unless every matrix product satisfies $\|\mathbf{D}_k \mathbf{W}_k\|_2 \equiv 1$ exactly (a set of measure zero in parameter space), training Plain Networks beyond 20–30 layers fails catastrophically.

#### Step 3: Exact Gradient Formulation of Residual Networks
By telescoping the residual recursion $\mathbf{x}_{k+1} - \mathbf{x}_k = \mathcal{F}_k(\mathbf{x}_k)$, the state at layer $L$ is:
$$\mathbf{x}_L = \mathbf{x}_l + \sum_{k=l}^{L-1} \mathcal{F}_k(\mathbf{x}_k, \mathcal{W}_k)$$

Differentiating $\mathcal{L}$ with respect to $\mathbf{x}_l$ using the total derivative:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \sum_{k=l}^{L-1} \frac{\partial \mathcal{F}_k(\mathbf{x}_k, \mathcal{W}_k)}{\partial \mathbf{x}_l} \right)$$

Define the composite residual Jacobian:
$$\mathbf{A}_{L, l} \equiv \sum_{k=l}^{L-1} \frac{\partial \mathcal{F}_k(\mathbf{x}_k, \mathcal{W}_k)}{\partial \mathbf{x}_l} \in \mathbb{R}^{d \times d}$$

Then:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} (\mathbf{I} + \mathbf{A}_{L, l}) = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} + \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \mathbf{A}_{L, l}$$

#### Step 4: Spectral Lower Bound on Gradient Norm
By the reverse triangle inequality for operator norms:
$$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \right\|_2 = \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} (\mathbf{I} + \mathbf{A}_{L, l}) \right\|_2 \ge \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\|_2 \cdot \sigma_{\min}(\mathbf{I} + \mathbf{A}_{L, l})$$
where $\sigma_{\min}(\mathbf{M})$ is the smallest singular value of matrix $\mathbf{M}$.

By Weyl's perturbation inequality for singular values:
$$\sigma_{\min}(\mathbf{I} + \mathbf{A}_{L, l}) \ge |1 - \|\mathbf{A}_{L, l}\|_2|$$

**Critical Initial State Analysis:**
At the start of training, weights are initialized with small variances (or zero-initialized in the final BN of each block, i.e., Fixup/ReZero initialization):
$$\|\mathbf{A}_{L, l}\|_2 \approx 0 \implies \sigma_{\min}(\mathbf{I} + \mathbf{A}_{L, l}) \approx 1$$
Consequently:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \approx \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \mathbf{I} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}$$

The gradient flows backward through 1,000 layers with **zero exponential decay or growth**. Gradient decay can only occur if $\mathbf{A}_{L, l} = -\mathbf{I}$, which requires the residual branch to actively invert the identity mapping—an unstable, non-isolated singular manifold that gradient descent readily avoids. $\blacksquare$

---

### 2.7 Deep Derivation 7.3.2: Dimensionality Reduction via $1 \times 1$ Convolutions and Asymmetric Kernel Factorization

In this derivation, we analyze the exact parameter and computational complexity reductions introduced by Inception architectures (Szegedy et al., 2015, 2016).

#### Step 1: Algebraic Structure of the $1 \times 1$ Cross-Channel Projection
Let an input feature tensor be $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$. A $1 \times 1$ convolution with $C_{\text{out}}$ filters parameterized by $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times 1 \times 1}$ and bias $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$ performs:
$$Y_{c, h, w} = \sum_{c'=0}^{C_{\text{in}}-1} W_{c, c', 0, 0} X_{c', h, w} + b_c$$

At every individual spatial coordinate $(h, w)$, this operation is mathematically identical to a dense linear projection:
$$\mathbf{y}_{h, w} = \mathbf{W} \, \mathbf{x}_{h, w} + \mathbf{b}, \qquad \mathbf{x}_{h, w} \in \mathbb{R}^{C_{\text{in}}}, \, \mathbf{y}_{h, w} \in \mathbb{R}^{C_{\text{out}}}$$

It pools features across channels without changing the spatial geometry ($H \times W$).

#### Step 2: FLOP Reduction in the Inception Bottleneck
Consider applying a $K \times K$ convolution to produce $C_{\text{out}}$ channels from $C_{\text{in}}$ channels.
The direct Multiply-Accumulate (MAC) count on an $H \times W$ feature map is:
$$\text{MAC}_{\text{direct}} = K^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W$$

Now insert a $1 \times 1$ bottleneck convolution that compresses $C_{\text{in}} \to C_{\text{mid}}$ channels (where $C_{\text{mid}} \ll C_{\text{in}}$), followed by the $K \times K$ convolution:
1. Stage 1 ($1 \times 1$ conv): $\text{MAC}_1 = 1^2 \cdot C_{\text{in}} \cdot C_{\text{mid}} \cdot H \cdot W$
2. Stage 2 ($K \times K$ conv): $\text{MAC}_2 = K^2 \cdot C_{\text{mid}} \cdot C_{\text{out}} \cdot H \cdot W$
$$\text{MAC}_{\text{bottleneck}} = (C_{\text{in}} C_{\text{mid}} + K^2 C_{\text{mid}} C_{\text{out}}) \cdot H \cdot W$$

The FLOP ratio between the bottleneck and direct approaches is:
$$\rho = \frac{\text{MAC}_{\text{bottleneck}}}{\text{MAC}_{\text{direct}}} = \frac{C_{\text{in}} C_{\text{mid}} + K^2 C_{\text{mid}} C_{\text{out}}}{K^2 C_{\text{in}} C_{\text{out}}} = \frac{C_{\text{mid}}}{K^2 C_{\text{out}}} + \frac{C_{\text{mid}}}{C_{\text{in}}}$$

**Numerical Evaluation:** For standard Inception-v1 values ($K = 5, C_{\text{in}} = 192, C_{\text{out}} = 192, C_{\text{mid}} = 32$):
$$\rho = \frac{32}{25 \times 192} + \frac{32}{192} = \frac{32}{4800} + \frac{1}{6} \approx 0.00667 + 0.16667 \approx 0.1733 \implies \mathbf{82.7\% \text{ reduction in computation!}}$$

#### Step 3: Asymmetric Spatial Factorization ($K \times K \to 1 \times K + K \times 1$)
In Inception-v2/v3, any 2D symmetric convolution $K \times K$ is factored into a cascade of two 1D asymmetric convolutions: a $1 \times K$ horizontal convolution followed by a $K \times 1$ vertical convolution.

Let the 2D spatial convolution of continuous or discrete kernel $k(x, y)$ be separable:
$$k(x, y) = g(x) \cdot h(y)$$
The 2D discrete cross-correlation factors as:
$$(X \star k)_{i, j} = \sum_{u} \sum_{v} X_{i+u, j+v} \, g_u h_v = \sum_{u} g_u \left( \sum_{v} X_{i+u, j+v} \, h_v \right)$$

Parameter and FLOP Comparison for $C$ channels:
- Symmetric $K \times K$:
  $$\text{Params}_{\text{sym}} = K^2 \cdot C^2$$
- Asymmetric $1 \times K$ followed by $K \times 1$:
  $$\text{Params}_{\text{asym}} = (1 \times K \times C^2) + (K \times 1 \times C^2) = 2K \cdot C^2$$

The asymptotic savings ratio is:
$$\frac{\text{Params}_{\text{asym}}}{\text{Params}_{\text{sym}}} = \frac{2K C^2}{K^2 C^2} = \frac{2}{K}$$

For $K = 7$ (as used extensively in Inception-v3):
$$\text{Ratio} = \frac{2}{7} \approx 0.2857 \implies \mathbf{71.4\% \text{ reduction in parameters and FLOPs}}$$
The receptive field remains identically $r = 1 + (K - 1) + (K - 1) = 2K - 1$ or $K \times K$ spatially, but introduces an intermediate ReLU non-linearity that boosts model expressivity. $\blacksquare$

---

### 2.8 Deep Derivation 7.3.3: MobileNet Depthwise Separable Convolutions and ConvNeXt Inverted Bottleneck Mechanics

Here we prove the computational scaling laws of depthwise separable convolutions and derive the algebraic properties of inverted bottleneck designs.

#### Step 1: Mathematical Factorization of Standard Convolution
A standard convolutional layer maps an input tensor $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ to an output tensor $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times H \times W}$ by simultaneously filtering spatial dimensions and combining channel dimensions:
$$Y_{c_{\text{out}}, h, w} = \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{u=0}^{D_K-1} \sum_{v=0}^{D_K-1} K_{c_{\text{out}}, c_{\text{in}}, u, v} \, X_{c_{\text{in}}, h+u, w+v}$$
Computational cost (Multiply-Accumulate operations):
$$\text{Cost}_{\text{standard}} = D_K \cdot D_K \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W$$

#### Step 2: Depthwise Separable Decomposition
Depthwise separable convolution splits this joint operation into two decoupled stages:
1. **Depthwise Convolution (Spatial Filtering):** Applies a single convolutional filter per input channel without cross-channel communication:
   $$\hat{Y}_{c, h, w} = \sum_{u=0}^{D_K-1} \sum_{v=0}^{D_K-1} K^{\text{DW}}_{c, u, v} \, X_{c, h+u, w+v}, \qquad c \in \{0, \dots, C_{\text{in}}-1\}$$
   Cost: $\text{Cost}_{\text{DW}} = D_K \cdot D_K \cdot C_{\text{in}} \cdot H \cdot W$.
2. **Pointwise Convolution (Channel Mixing):** A $1 \times 1$ convolution computes linear combinations across channels:
   $$Y_{c_{\text{out}}, h, w} = \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} K^{\text{PW}}_{c_{\text{out}}, c_{\text{in}}} \, \hat{Y}_{c_{\text{in}}, h, w}$$
   Cost: $\text{Cost}_{\text{PW}} = 1 \cdot 1 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W$.

Summing both stages:
$$\text{Cost}_{\text{separable}} = \left( D_K^2 \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}} \right) \cdot H \cdot W$$

#### Step 3: Theoretical Acceleration Factor
The ratio of computation between depthwise separable and standard convolution is:
$$\frac{\text{Cost}_{\text{separable}}}{\text{Cost}_{\text{standard}}} = \frac{(D_K^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}) H W}{D_K^2 C_{\text{in}} C_{\text{out}} H W} = \frac{1}{C_{\text{out}}} + \frac{1}{D_K^2}$$

In modern vision backbones where $C_{\text{out}} \gg 1$ (e.g., $C_{\text{out}} \ge 64$) and $D_K = 3$:
$$\frac{1}{C_{\text{out}}} \approx 0 \implies \frac{\text{Cost}_{\text{separable}}}{\text{Cost}_{\text{standard}}} \approx \frac{1}{D_K^2} = \frac{1}{3^2} = \frac{1}{9} \approx 0.111$$
Depthwise separable convolution delivers an **$8$ to $9\times$ reduction in computation** with negligible degradation in classification accuracy.

#### Step 4: The Inverted Bottleneck and Manifold of Interest
In classic ResNet bottlenecks, feature channels follow a Wide $\to$ Narrow $\to$ Wide structure:
$$C \xrightarrow{1 \times 1 \text{ reduce}} \frac{C}{4} \xrightarrow{3 \times 3 \text{ conv}} \frac{C}{4} \xrightarrow{1 \times 1 \text{ expand}} C$$

MobileNet-v2 (Sandler et al., 2018) and ConvNeXt (Liu et al., 2022) reverse this paradigm, adopting an **Inverted Bottleneck**:
$$C \xrightarrow{\text{DW Conv } 7 \times 7} C \xrightarrow{1 \times 1 \text{ expand}} t C \xrightarrow{\text{GELU}} t C \xrightarrow{1 \times 1 \text{ project}} C$$
where $t \ge 4$ is the expansion factor.

**Mathematical Justification (Manifold of Interest):**
Let feature representations reside in a low-dimensional manifold embedded within $\mathbb{R}^C$.
Applying non-linear activation $\sigma(z) = \max(0, z)$ (ReLU) or $z \cdot \Phi(z)$ (GELU) to a low-dimensional representation $d \le C$ projects all negative coordinates to $0$, collapsing the manifold volume and irreversibly destroying information (the "manifold collapse").
By projecting features into a high-dimensional expansion space $t C$ ($t = 4$ or $6$), the non-linearity operates with minimal subspace destruction. The final linear projection $t C \to C$ compresses the enriched representation back into the low-dimensional transmission channel without applying an activation function. $\blacksquare$

---

## 3. Geometric & Algebraic Interpretation

### Loss Landscape Smoothing (Li et al., 2018)
In "Filter Normalization for Visualizing Loss Landscapes", Li et al. demonstrated that without skip connections, deep networks (e.g., 56-layer PlainNet) possess chaotic, non-convex landscapes riddled with dramatic ridges, saddle plateaus, and shattered gradients.
The addition of identity shortcuts $\mathbf{x} + \mathcal{F}(\mathbf{x})$ mathematically prevents eigenvalues of the Hessian from exploding or collapsing, transforming the non-convex geometry into an extraordinarily smooth, nearly convex bowl.

### The Exponential Ensemble View (Veit et al., 2016)
A residual network of $N$ blocks can be algebraically expanded into a sum of $2^N$ distinct paths:
$$\mathbf{x}_N = \mathbf{x}_0 + \sum_i \mathcal{F}_i(\mathbf{x}_{i-1}) + \sum_{i < j} \mathcal{F}_j(\mathcal{F}_i(\dots)) + \dots$$
ResNet behaves like an ensemble of exponentially many shallow networks. When individual layers are randomly dropped during testing, performance degrades smoothly (unlike VGG, which collapses completely if any single layer is excised).

---

## 4. Real-World Analogy

### The Corporate Escalator vs. The Bureaucratic Memo Chain
- **Plain ConvNet (Bureaucracy):** An urgent operational directive is sent from the front-line branch office to headquarters across 50 bureaucratic management tiers. Each manager rewrites the memo in their own words. If any single manager misunderstands, distorts, or deletes the memo, the signal is destroyed; headquarters receives incoherent noise.
- **ResNet (The Express Escalator):** An open escalator runs continuously from the front desk directly to the CEO's office. Anyone can step on the escalator and deliver the raw message unchanged (the identity shortcut $\mathbf{I}$). Middle managers sit along the escalator and hand supplemental sticky notes (the residual $\mathcal{F}(\mathbf{x})$) to the messenger as they glide past. Even if a manager falls asleep or writes nonsense, the primary message arrives intact.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace forward and backward propagation through a complete Residual Block with concrete hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Residual Block |
| :--- | :--- | :--- | :--- |
| $\mathbf{x}$ | Input Vector | $(2,)$ | Input features entering the residual block |
| $\mathbf{W}_1, \mathbf{b}_1$ | Layer 1 Parameters | $(2, 2), (2,)$ | Weights and bias of first linear transformation |
| $\mathbf{z}_1$ | Pre-activation 1 | $(2,)$ | $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$ |
| $\mathbf{a}_1$ | Activation 1 | $(2,)$ | $\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$ |
| $\mathbf{W}_2, \mathbf{b}_2$ | Layer 2 Parameters | $(2, 2), (2,)$ | Weights and bias of second linear transformation |
| $\mathcal{F}(\mathbf{x})$ | Residual Output | $(2,)$ | Non-linear mapping $\mathcal{F}(\mathbf{x}) = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$ |
| $\mathbf{y}$ | Block Output | $(2,)$ | Combined output $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ | Upstream Gradient | $(2,)$ | Sensitivity of objective loss w.r.t. block output |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ | Input Gradient | $(2,)$ | Gradient routed through shortcut + residual path |

---

### 5.2 Concrete Toy Numbers

#### Input Vector:
$$\mathbf{x} = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix}$$

#### Layer 1 Weights and Bias:
$$\mathbf{W}_1 = \begin{bmatrix} 1.0 & -1.0 \\ 0.0 & 2.0 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$

#### Layer 2 Weights and Bias:
$$\mathbf{W}_2 = \begin{bmatrix} 0.5 & 1.0 \\ -1.0 & 0.5 \end{bmatrix}, \quad \mathbf{b}_2 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

#### Upstream Loss Gradient:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$

---

### 5.3 Forward Pass Arithmetic

1. **Pre-activation $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$:**
   $$z_{1, 1} = (1.0)(2.0) + (-1.0)(-1.0) + 0.0 = 2.0 + 1.0 + 0.0 = 3.0$$
   $$z_{1, 2} = (0.0)(2.0) + (2.0)(-1.0) + 1.0 = 0.0 - 2.0 + 1.0 = -1.0$$
   $$\mathbf{z}_1 = \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix}$$

2. **Activation $\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$:**
   $$a_{1, 1} = \max(0, 3.0) = 3.0$$
   $$a_{1, 2} = \max(0, -1.0) = 0.0$$
   $$\mathbf{a}_1 = \begin{bmatrix} 3.0 \\ 0.0 \end{bmatrix}$$

3. **Residual Mapping $\mathcal{F}(\mathbf{x}) = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$:**
   $$\mathcal{F}_1 = (0.5)(3.0) + (1.0)(0.0) + 0.0 = 1.5$$
   $$\mathcal{F}_2 = (-1.0)(3.0) + (0.5)(0.0) + 0.0 = -3.0$$
   $$\mathcal{F}(\mathbf{x}) = \begin{bmatrix} 1.5 \\ -3.0 \end{bmatrix}$$

4. **Residual Addition $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$:**
   $$y_1 = 1.5 + 2.0 = \mathbf{3.5}$$
   $$y_2 = -3.0 + (-1.0) = \mathbf{-4.0}$$
   $$\mathbf{y} = \begin{bmatrix} 3.5 \\ -4.0 \end{bmatrix}$$

---

### 5.4 Backward Pass Arithmetic (The Gradient Highway in Action)

The gradient with respect to $\mathbf{x}$ splits into two additive terms:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{y}}}_{\text{Via Shortcut}} + \underbrace{\left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} \right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}}_{\text{Via Residual Path}}$$

1. **Upstream Gradient through Residual Path:**
   $$\frac{\partial \mathcal{L}}{\partial \mathcal{F}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$

2. **Gradient w.r.t. $\mathbf{a}_1$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = \mathbf{W}_2^T \frac{\partial \mathcal{L}}{\partial \mathcal{F}} = \begin{bmatrix} 0.5 & -1.0 \\ 1.0 & 0.5 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$
   $$\frac{\partial \mathcal{L}}{\partial a_{1, 1}} = (0.5)(1.0) + (-1.0)(2.0) = 0.5 - 2.0 = -1.5$$
   $$\frac{\partial \mathcal{L}}{\partial a_{1, 2}} = (1.0)(1.0) + (0.5)(2.0) = 1.0 + 1.0 = 2.0$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = \begin{bmatrix} -1.5 \\ 2.0 \end{bmatrix}$$

3. **Gradient through ReLU $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}$:**
   Since $z_{1, 1} = 3.0 > 0$, ReLU gate is open ($1.0$).
   Since $z_{1, 2} = -1.0 < 0$, ReLU gate is closed ($0.0$).
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \begin{bmatrix} -1.5 \times 1.0 \\ 2.0 \times 0.0 \end{bmatrix} = \begin{bmatrix} -1.5 \\ 0.0 \end{bmatrix}$$

4. **Gradient w.r.t. $\mathbf{x}$ from Residual Path:**
   $$\left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} \right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \mathbf{W}_1^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \begin{bmatrix} 1.0 & 0.0 \\ -1.0 & 2.0 \end{bmatrix} \begin{bmatrix} -1.5 \\ 0.0 \end{bmatrix}$$
   $$= \begin{bmatrix} (1.0)(-1.5) + (0.0)(0.0) \\ (-1.0)(-1.5) + (2.0)(0.0) \end{bmatrix} = \begin{bmatrix} -1.5 \\ 1.5 \end{bmatrix}$$

5. **Accumulation with Identity Shortcut:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}_{\text{Shortcut}} + \begin{bmatrix} -1.5 \\ 1.5 \end{bmatrix}_{\text{Residual}} = \begin{bmatrix} 1.0 - 1.5 \\ 2.0 + 1.5 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.5 \\ 3.5 \end{bmatrix}}$$

Even if the second neuron in layer 1 was dead ($a_{1, 2} = 0$), the second component of the upstream gradient was safely preserved and transmitted by the shortcut ($2.0 \to 3.5$)!

---

## 6. Solved Illustrations

### Illustration 1: Bottleneck Block Dimension Projection

**Problem:**
In ResNet, when spatial resolution is halved ($s=2$) and channels are doubled (e.g., $128 \to 256$), the input tensor $\mathbf{x} \in \mathbb{R}^{B \times 128 \times 28 \times 28}$ cannot be directly added to $\mathcal{F}(\mathbf{x}) \in \mathbb{R}^{B \times 256 \times 14 \times 14}$.
How does ResNet resolve this dimension mismatch? What is the parameter cost of the shortcut projection?

**Solution:**
ResNet applies a **Projection Shortcut** using a $1 \times 1$ convolution with stride $s=2$:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{W}_s \mathbf{x}$$
where $\mathbf{W}_s \in \mathbb{R}^{256 \times 128 \times 1 \times 1}$ with stride $2$.
1. **Spatial downsampling:** The stride of $2$ downsamples the spatial resolution from $28 \times 28 \to 14 \times 14$.
2. **Channel matching:** The $256$ kernels expand the channel dimension from $128 \to 256$.
3. **Parameter cost:**
   $$\text{Params}(\mathbf{W}_s) = 256 \times 128 \times 1 \times 1 = 32,768 \text{ parameters}$$
Across the entire ResNet-50, projection shortcuts are only used at the beginning of Stages 2, 3, and 4, keeping the remaining shortcuts completely parameter-free.

---

### Illustration 2: Why ConvNeXt Adopts $7 \times 7$ Depthwise Convolutions

**Problem:**
Calculate the parameter count and computational FLOPs for a $7 \times 7$ depthwise convolution compared to a standard $3 \times 3$ dense convolution, assuming $C = 96$ channels and feature map size $56 \times 56$.

**Solution:**
1. **Standard Dense $3 \times 3$ Conv ($C \to C$):**
   - Parameters: $K_H \times K_W \times C_{\text{in}} \times C_{\text{out}} = 3 \times 3 \times 96 \times 96 = \mathbf{82,944}$
   - FLOPs: $2 \times \text{Params} \times H \times W = 2 \times 82,944 \times 56 \times 56 \approx \mathbf{520.2 \text{ MFLOPs}}$

2. **Depthwise $7 \times 7$ Conv ($C \to C$, groups=$C$):**
   - In a depthwise convolution, each of the $96$ channels is convolved with its own independent $7 \times 7$ kernel:
   - Parameters: $K_H \times K_W \times C = 7 \times 7 \times 96 = \mathbf{4,704}$
   - FLOPs: $2 \times \text{Params} \times H \times W = 2 \times 4,704 \times 56 \times 56 \approx \mathbf{29.5 \text{ MFLOPs}}$

**Conclusion:**
The $7 \times 7$ depthwise convolution has **$17.6\times$ fewer parameters** and **$17.6\times$ fewer FLOPs** than a standard dense $3 \times 3$ conv, while providing a dramatically larger receptive field ($7 \times 7 = 49$ pixels vs. $3 \times 3 = 9$ pixels)!

---

### Illustration 3: ResNet Bottleneck Block FLOP and Parameter Derivation

**Problem:**
Consider a ResNet-50 Bottleneck block in Stage 2 operating on feature maps of spatial dimensions $56 \times 56$ with $C_{\text{in}} = 256$, bottleneck width $C_{\text{mid}} = 64$, and output channels $C_{\text{out}} = 256$ (stride $1$, no downsampling).
1. Compute the exact parameter count and Multiply-Accumulate (MAC) count for each of the three constituent convolutional layers ($1 \times 1 \to 3 \times 3 \to 1 \times 1$).
2. Compare the total block parameters and MACs against a hypothetical "Plain" block consisting of two standard $3 \times 3$ convolutions with $256$ channels.
3. Compute the percentage reduction in computation and parameters achieved by the bottleneck design.

**Solution:**

#### Step 1: Bottleneck Layer-by-Layer Breakdown
- **Layer 1 ($1 \times 1 \text{ Conv, } 256 \to 64$):**
  - Parameters: $1 \times 1 \times C_{\text{in}} \times C_{\text{mid}} = 1 \times 1 \times 256 \times 64 = \mathbf{16,384}$
  - MACs: $\text{Params} \times H \times W = 16,384 \times 56 \times 56 = \mathbf{51,380,224}$ ($51.38 \text{ MMACs}$)
- **Layer 2 ($3 \times 3 \text{ Conv, } 64 \to 64$, padding $1$):**
  - Parameters: $3 \times 3 \times C_{\text{mid}} \times C_{\text{mid}} = 9 \times 64 \times 64 = \mathbf{36,864}$
  - MACs: $\text{Params} \times H \times W = 36,864 \times 56 \times 56 = \mathbf{115,605,504}$ ($115.61 \text{ MMACs}$)
- **Layer 3 ($1 \times 1 \text{ Conv, } 64 \to 256$):**
  - Parameters: $1 \times 1 \times C_{\text{mid}} \times C_{\text{out}} = 1 \times 1 \times 64 \times 256 = \mathbf{16,384}$
  - MACs: $\text{Params} \times H \times W = 16,384 \times 56 \times 56 = \mathbf{51,380,224}$ ($51.38 \text{ MMACs}$)

**Total Bottleneck Cost:**
- Total Parameters: $16,384 + 36,864 + 16,384 = \mathbf{69,632}$
- Total MACs: $51,380,224 + 115,605,504 + 51,380,224 = \mathbf{218,365,952}$ ($218.37 \text{ MMACs} \approx 436.73 \text{ MFLOPs}$)

#### Step 2: Comparison with Plain Two-Layer $3 \times 3$ Block ($256 \to 256$)
In a plain block without bottlenecking:
- Layer 1: $3 \times 3 \times 256 \times 256 = 589,824$ parameters; $\text{MACs} = 589,824 \times 56^2 = 1,849,688,064$ ($1.85 \text{ GMACs}$)
- Layer 2: $3 \times 3 \times 256 \times 256 = 589,824$ parameters; $\text{MACs} = 589,824 \times 56^2 = 1,849,688,064$ ($1.85 \text{ GMACs}$)
- Total Plain Parameters: $2 \times 589,824 = \mathbf{1,179,648}$
- Total Plain MACs: $2 \times 1,849,688,064 = \mathbf{3,699,376,128}$ ($3.70 \text{ GMACs}$)

#### Step 3: Computational and Parameter Savings
- **Parameter Savings:**
  $$\frac{1,179,648 - 69,632}{1,179,648} = \frac{1,110,016}{1,179,648} = \mathbf{94.1\% \text{ parameter reduction}}$$
- **Computational (FLOP) Savings:**
  $$\frac{3,699,376,128 - 218,365,952}{3,699,376,128} = \frac{3,481,010,176}{3,699,376,128} = \mathbf{94.1\% \text{ FLOP reduction}}$$
The bottleneck formulation enables nearly a **$17\times$ reduction** in computational load per block while preserving full 256-channel input and output capacity.

---

### Illustration 4: Inception $7 \times 7 \to 1 \times 7 + 7 \times 1$ Asymmetric Factorization

**Problem:**
In Inception-v3, a $7 \times 7$ convolution operating on $C = 128$ channels at spatial resolution $14 \times 14$ is factored into a sequential cascade of $1 \times 7$ and $7 \times 1$ convolutions.
1. Compute the parameter count and MACs for the direct $7 \times 7$ convolution.
2. Compute the parameter count and MACs for the factorized $(1 \times 7) \to (7 \times 1)$ cascade.
3. Verify that the effective receptive field is identical to $7 \times 7$.

**Solution:**

#### Step 1: Direct $7 \times 7$ Convolution
- Parameters:
  $$\text{Params}_{\text{direct}} = 7 \times 7 \times C \times C = 49 \times 128 \times 128 = 49 \times 16,384 = \mathbf{802,816}$$
- Computational Cost:
  $$\text{MAC}_{\text{direct}} = 802,816 \times 14 \times 14 = \mathbf{157,351,936} \approx \mathbf{157.35 \text{ MMACs}}$$

#### Step 2: Factorized $(1 \times 7) + (7 \times 1)$ Cascade
- **Stage 1 ($1 \times 7 \text{ Conv, horizontal, padding }(0, 3)$):**
  - Parameters: $1 \times 7 \times 128 \times 128 = 7 \times 16,384 = \mathbf{114,688}$
  - MACs: $114,688 \times 14 \times 14 = \mathbf{22,478,848}$ ($22.48 \text{ MMACs}$)
- **Stage 2 ($7 \times 1 \text{ Conv, vertical, padding }(3, 0)$):**
  - Parameters: $7 \times 1 \times 128 \times 128 = 7 \times 16,384 = \mathbf{114,688}$
  - MACs: $114,688 \times 14 \times 14 = \mathbf{22,478,848}$ ($22.48 \text{ MMACs}$)

**Total Factorized Cost:**
- Parameters: $114,688 + 114,688 = \mathbf{229,376}$
- MACs: $22,478,848 + 22,478,848 = \mathbf{44,957,696} \approx \mathbf{44.96 \text{ MMACs}}$

**Exact Savings:**
$$\text{Reduction} = \frac{802,816 - 229,376}{802,816} = \frac{573,440}{802,816} = 1 - \frac{2}{7} = \frac{5}{7} \approx \mathbf{71.43\% \text{ reduction in compute and memory}}$$

#### Step 3: Receptive Field Verification
- After $1 \times 7$ horizontal convolution with unit stride: $r_H = 1$, $r_W = 1 + (7 - 1) = 7$.
- After $7 \times 1$ vertical convolution with unit stride: $r_H = 1 + (7 - 1) = 7$, $r_W = 7 + (1 - 1) = 7$.
The composite receptive field is exactly $7 \times 7$, matching the unfactored filter while executing $3.5\times$ faster.

---

### Illustration 5: MobileNet-v2 Inverted Residual Block ($t=6$) Arithmetic Trace

**Problem:**
Let an Inverted Residual Block with expansion factor $t = 6$ process an input $\mathbf{X} \in \mathbb{R}^{64 \times 28 \times 28}$ to produce output $\mathbf{Y} \in \mathbb{R}^{64 \times 28 \times 28}$ ($C_{\text{in}} = 64, C_{\text{out}} = 64, s=1$).
The block executes:
1. $1 \times 1 \text{ pointwise expansion } (64 \to 64 \times 6 = 384) + \text{ReLU6}$
2. $3 \times 3 \text{ depthwise conv } (384 \to 384, \text{groups}=384) + \text{ReLU6}$
3. $1 \times 1 \text{ pointwise linear projection } (384 \to 64, \text{no non-linearity})$
4. Element-wise shortcut addition $\mathbf{Y} = \mathbf{X} + \text{residual}$.

Compute:
1. Parameter count and MAC count for each of the three stages.
2. The total block FLOP count.
3. Contrast this against a direct $3 \times 3$ dense convolution with $384$ intermediate channels.

**Solution:**

#### Step 1: Layer-by-Layer Arithmetic
1. **Pointwise Expansion ($1 \times 1 \text{ Conv, } 64 \to 384$):**
   - Parameters: $1 \times 1 \times 64 \times 384 = \mathbf{24,576}$
   - MACs: $24,576 \times 28 \times 28 = \mathbf{19,267,584}$ ($19.27 \text{ MMACs}$)
2. **Depthwise Convolution ($3 \times 3 \text{ Conv, } 384 \to 384, \text{groups}=384$):**
   - Parameters: $3 \times 3 \times 384 = \mathbf{3,456}$
   - MACs: $3,456 \times 28 \times 28 = \mathbf{2,709,504}$ ($2.71 \text{ MMACs}$)
3. **Linear Pointwise Projection ($1 \times 1 \text{ Conv, } 384 \to 64$):**
   - Parameters: $1 \times 1 \times 384 \times 64 = \mathbf{24,576}$
   - MACs: $24,576 \times 28 \times 28 = \mathbf{19,267,584}$ ($19.27 \text{ MMACs}$)

#### Step 2: Total Block Totals
- Total Parameters: $24,576 + 3,456 + 24,576 = \mathbf{52,608}$
- Total MACs: $19,267,584 + 2,709,504 + 19,267,584 = \mathbf{41,244,672} \approx \mathbf{41.24 \text{ MMACs}}$ ($82.49 \text{ MFLOPs}$)

#### Step 3: Comparison with Dense 384-Channel Convolution
If the spatial filtering had been performed with a standard dense $3 \times 3$ convolution on the expanded 384 channels ($384 \to 384$):
- Parameters: $3 \times 3 \times 384 \times 384 = 1,327,104$ parameters!
- MACs: $1,327,104 \times 28^2 = 1,040,449,536 \approx \mathbf{1,040.45 \text{ MMACs}}$
- The depthwise layer consumes only $2.71$ MMACs compared to $1,040.45$ MMACs—an astonishing **$384\times$ reduction** in spatial filtering compute, proving why inverted bottlenecks are the foundation of edge computer vision.

---

## 7. Deep Learning Connection & Application

### Architecture Selection Guidelines: ResNet vs. ConvNeXt vs. ViT

| Criteria | ResNet-50 | ConvNeXt-T | ViT-B/16 |
| :--- | :--- | :--- | :--- |
| **Primary Inductive Bias** | Strong (Local translation equivariance) | Strong (Local translation equivariance) | Weak (Global self-attention) |
| **Data Efficiency** | High (Trains well from scratch on small datasets) | High (Excels on ImageNet-1k without massive pretraining) | Low (Requires massive datasets e.g., ImageNet-21k, JFT-300M) |
| **Inference Latency** | Ultra-fast on edge/mobile hardware | Extremely fast, optimized for GPU memory access | High memory footprint; quadratic cost with resolution |
| **Modern Downstream Use** | Legacy embedded devices, simple baselines | State-of-the-art vision backbone for detection/segmentation | Multimodal foundation models (CLIP, LLaVA, SAM) |

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Pure PyTorch implementations of:
   - Factorized VGG Block ($2 \times 3 \times 3$ vs. $1 \times 5 \times 5$).
   - ResNet BasicBlock and Bottleneck with identity and projection shortcuts.
   - ConvNeXt Block (7x7 depthwise, LayerNorm2d, 1x1 conv expansion 4x, GELU, 1x1 projection, LayerScale).
2. Exact numerical verification of the Part 5 Visual Grid hand arithmetic.
3. The **Gradient Norm Preservation Experiment**:
   - Compare backpropagated gradient magnitudes through a 50-layer PlainNet vs. a 50-layer ResNet.
   - Proves that PlainNet gradients decay exponentially toward $0$, whereas ResNet preserves healthy gradient magnitude across all 50 blocks.

See implementation in:
[`07_convolutional_networks/code/03_classic_to_modern_convnets.py`](./code/03_classic_to_modern_convnets.py)
