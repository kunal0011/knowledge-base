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
