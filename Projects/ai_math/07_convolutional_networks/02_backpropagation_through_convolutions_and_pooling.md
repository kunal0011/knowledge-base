# Chapter 7.2: Backpropagation Through Convolutions and Pooling Layers

---

## 1. Intuition & 101 Motivation

In dense multilayer perceptrons, every scalar weight connects exactly one input neuron to one output neuron. Deriving backpropagation for a dense layer reduces to standard vector-matrix calculus: upstream gradients multiply the weight transpose $\mathbf{W}^T$.

In Convolutional Neural Networks (CNNs), the computational paradigm changes radically:
1. **Weight Sharing:** A single kernel element $K_{u, v}$ participates in thousands of scalar multiplications across spatial locations in an image.
2. **Local Connectivity:** Each input pixel $X_{i, j}$ influences multiple adjacent output activations within overlapping receptive windows.
3. **Non-Invertible Spatial Downsampling:** Max-pooling discards $75\%$ or more of activations, dynamically selecting maximal values whose identities vary batch-to-batch and iteration-to-iteration.

How does backpropagation route errors through shared kernel weights and spatial pooling operations?

The central insights of this chapter are:
- **Gradient w.r.t. Weights ($\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$):** Is itself a convolution/cross-correlation between the input feature map $\mathbf{X}$ and the upstream loss gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$.
- **Gradient w.r.t. Input ($\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$):** Is a *full convolution* between the zero-padded upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$ and the spatial **$180^\circ$ rotated kernel** ($\text{rot}_{180}(\mathbf{K})$).
- **Matrix Adjoint Formulation:** When 2D convolution is expressed as a matrix-vector product $\mathbf{y} = \mathbf{C}\mathbf{x}$ (where $\mathbf{C}$ is a Doubly Block Toeplitz matrix), the backward pass w.r.t. $\mathbf{x}$ is the linear operator adjoint $\mathbf{C}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$.
- **Max-Pooling Backward:** Acts as an *argmax routing switch*—the upstream gradient flows unimpeded through the winning index, while non-maximal cells receive exactly zero gradient.

---

## 2. Rigorous Mathematical Formulation

Let the forward cross-correlation (standard deep learning convolution) for a single input channel and single kernel be defined over spatial coordinates $(h, w)$ as:
$$Y_{h, w} = \sum_{u=0}^{K_H - 1} \sum_{v=0}^{K_W - 1} X_{h \cdot s + u, \, w \cdot s + v} \, K_{u, v} + b$$
where:
- $\mathbf{X} \in \mathbb{R}^{H_{\text{in}} \times W_{\text{in}}}$ is the input feature map (zero-padded as necessary),
- $\mathbf{K} \in \mathbb{R}^{K_H \times K_W}$ is the convolutional kernel,
- $b \in \mathbb{R}$ is the scalar bias,
- $s \in \mathbb{Z}_{\ge 1}$ is the stride,
- $\mathbf{Y} \in \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$ is the output feature map,
- $\mathcal{L} \in \mathbb{R}$ is the scalar scalar objective loss function.

We are given the upstream gradient tensor:
$$\delta_{h, w} \equiv \frac{\partial \mathcal{L}}{\partial Y_{h, w}} \in \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$$

---

### 2.1 Gradient with Respect to Kernel Weights ($\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$)

By multivariable chain rule, the partial derivative of $\mathcal{L}$ with respect to a specific kernel parameter $K_{u, v}$ accumulates the gradient contributions across all spatial positions $(h, w)$ where $K_{u, v}$ was evaluated:
$$\frac{\partial \mathcal{L}}{\partial K_{u, v}} = \sum_{h=0}^{H_{\text{out}} - 1} \sum_{w=0}^{W_{\text{out}} - 1} \frac{\partial \mathcal{L}}{\partial Y_{h, w}} \frac{\partial Y_{h, w}}{\partial K_{u, v}}$$

From the forward equation:
$$\frac{\partial Y_{h, w}}{\partial K_{u, v}} = X_{h \cdot s + u, \, w \cdot s + v}$$

Therefore:
$$\frac{\partial \mathcal{L}}{\partial K_{u, v}} = \sum_{h=0}^{H_{\text{out}} - 1} \sum_{w=0}^{W_{\text{out}} - 1} \delta_{h, w} \, X_{h \cdot s + u, \, w \cdot s + v}$$

**Structural Interpretation:**
For stride $s = 1$, this equation represents the cross-correlation between the input $\mathbf{X}$ and the upstream gradient $\boldsymbol{\delta}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \mathbf{X} \star \boldsymbol{\delta}$$
The kernel gradient has spatial dimension $K_H \times K_W$.

For multi-channel, multi-filter tensors ($\mathbf{X} \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$, $\mathbf{K} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W}$):
$$\frac{\partial \mathcal{L}}{\partial K_{c_{\text{out}}, c_{\text{in}}, u, v}} = \sum_{b=0}^{B - 1} \sum_{h=0}^{H_{\text{out}} - 1} \sum_{w=0}^{W_{\text{out}} - 1} \delta_{b, c_{\text{out}}, h, w} \, X_{b, c_{\text{in}}, h \cdot s + u, \, w \cdot s + v}$$

---

### 2.2 Gradient with Respect to Bias ($\frac{\partial \mathcal{L}}{\partial b}$)

Since the bias $b_{c_{\text{out}}}$ is added to every spatial location of channel $c_{\text{out}}$ across the entire mini-batch:
$$\frac{\partial Y_{b, c_{\text{out}}, h, w}}{\partial b_{c_{\text{out}}}} = 1$$
$$\frac{\partial \mathcal{L}}{\partial b_{c_{\text{out}}}} = \sum_{b=0}^{B - 1} \sum_{h=0}^{H_{\text{out}} - 1} \sum_{w=0}^{W_{\text{out}} - 1} \delta_{b, c_{\text{out}}, h, w}$$

The bias gradient is simply the sum of upstream gradient activations over batch and spatial dimensions.

---

### 2.3 Gradient with Respect to Input Feature Map ($\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$)

An input pixel $X_{i, j}$ influences any output pixel $Y_{h, w}$ where the kernel window covers $(i, j)$.
From $Y_{h, w} = \sum_{u} \sum_{v} X_{h \cdot s + u, \, w \cdot s + v} K_{u, v}$, an index match occurs when:
$$i = h \cdot s + u \implies u = i - h \cdot s$$
$$j = w \cdot s + v \implies v = j - w \cdot s$$

Subject to the kernel boundary constraints $0 \le u < K_H$ and $0 \le v < K_W$:
$$\frac{\partial \mathcal{L}}{\partial X_{i, j}} = \sum_{h} \sum_{w} \delta_{h, w} \frac{\partial Y_{h, w}}{\partial X_{i, j}} = \sum_{h} \sum_{w} \delta_{h, w} \, K_{i - h \cdot s, \, j - w \cdot s}$$

#### The 180° Kernel Rotation ($\text{rot}_{180}$)
For unit stride $s = 1$, let $u' = K_H - 1 - u$ and $v' = K_W - 1 - v$. This spatial reflection yields:
$$K^{\text{rot}}_{u', v'} = K_{K_H - 1 - u', \, K_W - 1 - v'}$$
Substituting variables transforms the equation into a standard convolution of the zero-padded gradient $\boldsymbol{\delta}$ with the flipped kernel $\mathbf{K}^{\text{rot}}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \boldsymbol{\delta} * \mathbf{K}^{\text{rot}} = \boldsymbol{\delta} \star \mathbf{K}$$
where $*$ denotes mathematical convolution (which reflects the kernel) and $\star$ denotes cross-correlation.

#### Boundary Padding for the Backward Pass
To obtain $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ of shape $H_{\text{in}} \times W_{\text{in}}$ from an upstream gradient of shape $H_{\text{out}} \times W_{\text{out}}$ (where $H_{\text{out}} = H_{\text{in}} - K_H + 1$ with $p=0, s=1$):
$$\text{Output Shape} = H_{\text{out}} + 2 \cdot p_{\text{backward}} - K_H + 1 = (H_{\text{in}} - K_H + 1) + 2(K_H - 1) - K_H + 1 = H_{\text{in}}$$
Thus, the backward pass on the gradient requires **full padding**:
$$p_{\text{backward}} = K_H - 1 - p_{\text{forward}}$$

---

### 2.4 Strided Convolutions and Transposed Convolutions

When the forward stride $s > 1$, output pixels are spaced apart on the input grid.
In the backward pass:
1. The upstream gradient $\boldsymbol{\delta}$ must be **dilated** by inserting $s - 1$ zeros between adjacent rows and columns.
2. The dilated gradient is padded with $p_{\text{backward}} = K - 1 - p_{\text{forward}}$ zeros.
3. A unit-stride cross-correlation is performed with the $180^\circ$ flipped kernel $\mathbf{K}^{\text{rot}}$.

This exact operation defines the **Transposed Convolution** (`torch.nn.ConvTranspose2d`, historically termed *deconvolution* or *fractionally strided convolution*).

$$\text{Forward}(\text{ConvTranspose2d}) \equiv \text{Backward}_{\mathbf{X}}(\text{Conv2d})$$

---

### 2.5 Backpropagation Through Pooling Layers

Pooling layers contain no learnable parameters ($\mathbf{K}$ does not exist), so backpropagation only computes $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$.

#### Max-Pooling Backward Pass (Argmax Routing Switch)
Let the forward max-pooling operation on window $\Omega(h, w)$ of size $P_H \times P_W$ with stride $s_p$ be:
$$Y_{h, w} = \max_{(i, j) \in \Omega(h, w)} X_{i, j}$$

Define the indicator mask:
$$\mathbf{1}_{\text{argmax}}(i, j; h, w) = \begin{cases} 1 & \text{if } (i, j) = \operatorname{argmax}_{(i', j') \in \Omega(h, w)} X_{i', j'} \\ 0 & \text{otherwise} \end{cases}$$

The derivative of $Y_{h, w}$ with respect to $X_{i, j}$ is piecewise differentiable:
$$\frac{\partial Y_{h, w}}{\partial X_{i, j}} = \mathbf{1}_{\text{argmax}}(i, j; h, w)$$

Accumulating upstream gradient contributions over all windows covering $(i, j)$:
$$\frac{\partial \mathcal{L}}{\partial X_{i, j}} = \sum_{h, w} \delta_{h, w} \cdot \mathbf{1}_{\text{argmax}}(i, j; h, w)$$

**Tie-Breaking Rule:** If multiple inputs inside a pooling window achieve the exact maximum value, subgradient convention routes the gradient either uniformly divided among ties or entirely to the first occurrence in row-major order (PyTorch convention routes to the first index).

#### Average Pooling Backward Pass (Uniform Gradient Distribution)
Let the forward average pooling operation be:
$$Y_{h, w} = \frac{1}{P_H \cdot P_W} \sum_{(i, j) \in \Omega(h, w)} X_{i, j}$$

The partial derivative is uniform across all elements in the pooling window:
$$\frac{\partial Y_{h, w}}{\partial X_{i, j}} = \frac{1}{P_H \cdot P_W} \cdot \mathbf{1}_{(i, j) \in \Omega(h, w)}$$

The input gradient distributes the upstream error uniformly:
$$\frac{\partial \mathcal{L}}{\partial X_{i, j}} = \sum_{(h, w): (i, j) \in \Omega(h, w)} \frac{1}{P_H \cdot P_W} \delta_{h, w}$$

---

### 2.6 Deep Derivation 7.2.1: Rigorous Vectorized Backpropagation for Conv2D (Doubly Block Toeplitz & Adjoint Operators)

In this derivation, we establish the exact coordinate-free and tensor-algebraic formulation of 2D convolutional backpropagation from first principles using Fréchet differentials and inner-product adjoints.

#### Step 1: Definition of the Forward Operator and Inner Product Spaces
Let the input space $\mathcal{X} = \mathbb{R}^{H \times W}$ and output space $\mathcal{Y} = \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$ be equipped with the standard Frobenius inner product:
$$\langle \mathbf{A}, \mathbf{B} \rangle = \operatorname{Tr}(\mathbf{A}^T \mathbf{B}) = \sum_{i, j} A_{i, j} B_{i, j}$$

For a fixed kernel $\mathbf{K} \in \mathbb{R}^{K_H \times K_W}$, unit stride $s = 1$, and valid padding $p = 0$, the forward cross-correlation linear operator $\mathcal{T}_{\mathbf{K}}: \mathcal{X} \to \mathcal{Y}$ is defined coordinate-wise by:
$$(\mathcal{T}_{\mathbf{K}} \mathbf{X})_{h, w} = \sum_{u=0}^{K_H - 1} \sum_{v=0}^{K_W - 1} K_{u, v} X_{h+u, w+v}, \quad 0 \le h < H_{\text{out}}, \, 0 \le w < W_{\text{out}}$$
where $H_{\text{out}} = H - K_H + 1$ and $W_{\text{out}} = W - K_W + 1$.

Let $\mathcal{L}: \mathcal{Y} \to \mathbb{R}$ be a continuously differentiable scalar loss function. The Fréchet differential of $\mathcal{L}$ with respect to $\mathbf{Y} = \mathcal{T}_{\mathbf{K}} \mathbf{X}$ is:
$$d\mathcal{L} = \langle \nabla_{\mathbf{Y}} \mathcal{L}, d\mathbf{Y} \rangle = \langle \boldsymbol{\delta}, d\mathbf{Y} \rangle = \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta_{h, w} \, dY_{h, w}$$
where $\boldsymbol{\delta} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \in \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$ is the upstream gradient.

#### Step 2: Exact Fréchet Derivation of Weight Gradient $\nabla_{\mathbf{K}} \mathcal{L}$
Holding $\mathbf{X}$ constant and perturbing $\mathbf{K}$ by an infinitesimal variation $d\mathbf{K} \in \mathbb{R}^{K_H \times K_W}$:
$$dY_{h, w} = \sum_{u=0}^{K_H - 1} \sum_{v=0}^{K_W - 1} X_{h+u, w+v} \, dK_{u, v}$$

Substituting $d\mathbf{Y}$ into the Fréchet differential $d\mathcal{L}$:
$$d\mathcal{L} = \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta_{h, w} \left( \sum_{u=0}^{K_H - 1} \sum_{v=0}^{K_W - 1} X_{h+u, w+v} \, dK_{u, v} \right)$$

By Fubini's theorem (interchanging finite summations):
$$d\mathcal{L} = \sum_{u=0}^{K_H - 1} \sum_{v=0}^{K_W - 1} dK_{u, v} \left( \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} X_{h+u, w+v} \, \delta_{h, w} \right)$$

By Riesz representation theorem on $(\mathbb{R}^{K_H \times K_W}, \langle \cdot, \cdot \rangle)$, $d\mathcal{L} = \langle \nabla_{\mathbf{K}} \mathcal{L}, d\mathbf{K} \rangle$. Identifying the kernel of the linear form gives:
$$\left( \frac{\partial \mathcal{L}}{\partial \mathbf{K}} \right)_{u, v} = \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} X_{h+u, w+v} \, \delta_{h, w} = (\mathbf{X} \star \boldsymbol{\delta})_{u, v}$$

Thus, the gradient with respect to the filter is precisely the valid cross-correlation between the input feature map $\mathbf{X}$ and the upstream gradient $\boldsymbol{\delta}$.

#### Step 3: Exact Adjoint Derivation of Input Gradient $\nabla_{\mathbf{X}} \mathcal{L}$
Holding $\mathbf{K}$ constant and perturbing $\mathbf{X}$ by $d\mathbf{X} \in \mathbb{R}^{H \times W}$:
$$dY_{h, w} = \sum_{u=0}^{K_H - 1} \sum_{v=0}^{K_W - 1} K_{u, v} \, dX_{h+u, w+v}$$

Substituting into $d\mathcal{L}$:
$$d\mathcal{L} = \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta_{h, w} \sum_{u=0}^{K_H - 1} \sum_{v=0}^{K_W - 1} K_{u, v} \, dX_{h+u, w+v}$$

To isolate $dX_{i, j}$, we perform a change of index coordinates. Define:
$$i = h + u \implies u = i - h, \qquad j = w + v \implies v = j - w$$

Under this change of summation, $(i, j)$ ranges over the full spatial grid $0 \le i < H$ and $0 \le j < W$. The indicators $0 \le i - h < K_H$ and $0 \le j - w < K_W$ define the active kernel bounds:
$$d\mathcal{L} = \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} dX_{i, j} \left( \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta_{h, w} \, K_{i-h, j-w} \cdot \mathbf{1}_{\{0 \le i-h < K_H, \, 0 \le j-w < K_W\}} \right)$$

Equating with $d\mathcal{L} = \langle \nabla_{\mathbf{X}} \mathcal{L}, d\mathbf{X} \rangle$:
$$\left( \frac{\partial \mathcal{L}}{\partial \mathbf{X}} \right)_{i, j} = \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta_{h, w} \, K_{i-h, j-w}$$

Now let $\mathbf{K}^{\text{rot}} \in \mathbb{R}^{K_H \times K_W}$ be the spatial $180^\circ$ reflection of $\mathbf{K}$:
$$K^{\text{rot}}_{u', v'} = K_{K_H - 1 - u', \, K_W - 1 - v'}$$
Let $u' = K_H - 1 - (i - h)$ and $v' = K_W - 1 - (j - w)$, which implies $h = i - (K_H - 1 - u')$ and $w = j - (K_W - 1 - v')$.
Embedding $\boldsymbol{\delta}$ in a zero-padded array $\boldsymbol{\delta}^{\text{pad}} \in \mathbb{R}^{(H_{\text{out}} + 2(K_H - 1)) \times (W_{\text{out}} + 2(K_W - 1))}$:
$$\left( \frac{\partial \mathcal{L}}{\partial \mathbf{X}} \right)_{i, j} = \sum_{u'=0}^{K_H-1} \sum_{v'=0}^{K_W-1} \delta^{\text{pad}}_{i + u', \, j + v'} \, K^{\text{rot}}_{u', v'} = (\boldsymbol{\delta}^{\text{pad}} \star \mathbf{K}^{\text{rot}})_{i, j} = \text{FullConv}(\boldsymbol{\delta}, \mathbf{K}^{\text{rot}})_{i, j}$$

#### Step 4: Matrix Representation via Doubly Block Toeplitz Operators
Let $\mathbf{x} = \operatorname{vec}(\mathbf{X}) \in \mathbb{R}^{HW}$ and $\mathbf{y} = \operatorname{vec}(\mathbf{Y}) \in \mathbb{R}^{H_{\text{out}} W_{\text{out}}}$ denote vectorized feature maps in lexicographic (row-major) order. The 2D convolution is an exact matrix-vector multiplication:
$$\mathbf{y} = \mathbf{C}_{\mathbf{K}} \mathbf{x}$$
where $\mathbf{C}_{\mathbf{K}}$ is a **Doubly Block Toeplitz Matrix** structured as:
$$\mathbf{C}_{\mathbf{K}} = \begin{bmatrix}
\mathbf{T}_0 & \mathbf{T}_1 & \dots & \mathbf{T}_{K_H-1} & \mathbf{0} & \dots & \mathbf{0} \\
\mathbf{0} & \mathbf{T}_0 & \mathbf{T}_1 & \dots & \mathbf{T}_{K_H-1} & \dots & \mathbf{0} \\
\vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots \\
\mathbf{0} & \dots & \mathbf{0} & \mathbf{T}_0 & \mathbf{T}_1 & \dots & \mathbf{T}_{K_H-1}
\end{bmatrix} \in \mathbb{R}^{(H_{\text{out}}W_{\text{out}}) \times (HW)}$$
Each block $\mathbf{T}_u \in \mathbb{R}^{W_{\text{out}} \times W}$ is itself a banded Toeplitz matrix formed from the $u$-th row of $\mathbf{K}$:
$$\mathbf{T}_u = \begin{bmatrix}
K_{u, 0} & K_{u, 1} & \dots & K_{u, K_W-1} & 0 & \dots & 0 \\
0 & K_{u, 0} & K_{u, 1} & \dots & K_{u, K_W-1} & \dots & 0 \\
\vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots \\
0 & \dots & 0 & K_{u, 0} & K_{u, 1} & \dots & K_{u, K_W-1}
\end{bmatrix}$$

Using vector calculus, the gradient with respect to $\mathbf{x}$ is the adjoint (transpose):
$$\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{C}_{\mathbf{K}}^T \boldsymbol{\delta}_{\mathbf{y}}$$
Because the transpose of a Toeplitz matrix reverses the diagonals ($\mathbf{T}_u^T$ shifts down instead of right and has reversed elements), the block transpose $\mathbf{C}_{\mathbf{K}}^T$ precisely mirrors the kernel horizontally and vertically ($180^\circ$ rotation) and expands the spatial domain to full convolution:
$$\operatorname{vec}^{-1}(\mathbf{C}_{\mathbf{K}}^T \boldsymbol{\delta}_{\mathbf{y}}) \equiv \text{FullConv}(\boldsymbol{\delta}, \mathbf{K}^{\text{rot}})$$
$\blacksquare$

---

### 2.7 Deep Derivation 7.2.2: Exact Backpropagation Through Pooling Operators and Global Average Pooling Adjoints

Pooling operations reduce spatial resolution. Here we derive the exact subgradients and adjoint operators for Max-Pooling, Average-Pooling, and Global Average Pooling (GAP).

#### Step 1: Max-Pooling Subdifferential
Let the pooling window covering location $(h, w)$ with kernel size $(P_H, P_W)$ and stride $s_p$ be:
$$\Omega(h, w) = \{ (i, j) \in \mathbb{Z}^2 : h \cdot s_p \le i < h \cdot s_p + P_H, \, w \cdot s_p \le j < w \cdot s_p + P_W \}$$
The forward activation is $Y_{h, w} = \max_{(i, j) \in \Omega(h, w)} X_{i, j}$.

Since the maximum function $f(\mathbf{z}) = \max_k z_k$ is convex and non-smooth at points with non-unique maxima, its Clarke generalized subdifferential is the convex hull of the standard basis vectors for the maximal coordinates:
$$\partial Y_{h, w} = \operatorname{conv}\left\{ \mathbf{e}_{i^*, j^*} : (i^*, j^*) \in \operatorname{Argmax}_{(i, j) \in \Omega(h, w)} X_{i, j} \right\}$$

Under the standard deep learning tie-breaking rule (selecting the first occurring maximal index in lexicographic order):
$$\mathbf{1}_{\text{argmax}}(i, j; h, w) = \begin{cases} 1 & \text{if } (i, j) = \min_{\text{lex}} \operatorname{Argmax}_{(i', j') \in \Omega(h, w)} X_{i', j'} \\ 0 & \text{otherwise} \end{cases}$$

Applying the chain rule for the scalar loss $\mathcal{L}$:
$$\frac{\partial \mathcal{L}}{\partial X_{i, j}} = \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \frac{\partial \mathcal{L}}{\partial Y_{h, w}} \frac{\partial Y_{h, w}}{\partial X_{i, j}} = \sum_{h, w} \delta_{h, w} \, \mathbf{1}_{\text{argmax}}(i, j; h, w)$$

**Key Properties:**
1. **Sparsity:** For non-overlapping pooling ($s_p = P_H = P_W$), exactly one input in each window receives a nonzero gradient; the remaining $P_H P_W - 1$ inputs receive strictly zero gradient. The gradient tensor $\nabla_{\mathbf{X}} \mathcal{L}$ has a sparsity ratio of at least $1 - \frac{1}{P_H P_W}$ (e.g., $75\%$ zeros for $2 \times 2$ pooling).
2. **Accumulation on Overlap:** When $s_p < P_H$, windows overlap. If a single pixel $X_{i, j}$ dominates multiple overlapping windows, it receives the algebraic sum of all corresponding upstream errors $\sum \delta_{h, w}$.

#### Step 2: Average Pooling Adjoint
For average pooling, the forward operation is linear:
$$Y_{h, w} = \frac{1}{P_H P_W} \sum_{(i, j) \in \Omega(h, w)} X_{i, j}$$
The Jacobian entry is independent of the input values:
$$\frac{\partial Y_{h, w}}{\partial X_{i, j}} = \frac{1}{P_H P_W} \mathbf{1}_{\{(i, j) \in \Omega(h, w)\}}$$
Accumulating upstream contributions yields:
$$\frac{\partial \mathcal{L}}{\partial X_{i, j}} = \frac{1}{P_H P_W} \sum_{(h, w): (i, j) \in \Omega(h, w)} \delta_{h, w}$$
Average pooling acts as a uniform spatial diffuser of the error gradient.

#### Step 3: Global Average Pooling (GAP) as the Exact Adjoint of Spatial Broadcasting
Global Average Pooling collapses each 2D channel map $\mathbf{X}_c \in \mathbb{R}^{H \times W}$ into a single scalar $y_c \in \mathbb{R}$:
$$y_c = \frac{1}{HW} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} X_{c, i, j} = \frac{1}{HW} \mathbf{1}^T \operatorname{vec}(\mathbf{X}_c)$$
where $\mathbf{1} \in \mathbb{R}^{HW}$ is a vector of ones.

Let $\delta_c = \frac{\partial \mathcal{L}}{\partial y_c} \in \mathbb{R}$ be the scalar upstream gradient for channel $c$. The Fréchet differential is:
$$d\mathcal{L} = \delta_c \, dy_c = \delta_c \left( \frac{1}{HW} \sum_{i, j} dX_{c, i, j} \right) = \sum_{i, j} \left( \frac{\delta_c}{HW} \right) dX_{c, i, j}$$
Identifying the gradient tensor for channel $c$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}_c} = \frac{\delta_c}{HW} \mathbf{J}_{H \times W}$$
where $\mathbf{J}_{H \times W}$ is the $H \times W$ all-ones matrix.

**Adjoint Interpretation:**
In the forward pass, GAP is an orthogonal projection onto the subspace of spatially constant feature maps followed by scaling by $\frac{1}{HW}$. In the backward pass, the adjoint operator is the **uniform spatial broadcast** of the scalar gradient $\delta_c$ across all $HW$ spatial pixels, divided by $HW$. $\blacksquare$

---

### 2.8 Deep Derivation 7.2.3: Transposed Convolution as the Exact Matrix Adjoint Operator and Checkerboard Periodicity

Transposed convolution (`torch.nn.ConvTranspose2d`) is frequently termed "deconvolution," though it does not invert the convolution mathematically. Here we prove its exact equivalence to the adjoint operator and analyze the origin of checkerboard artifacts.

#### Step 1: Definition of the Adjoint Linear Operator
Let $\mathbf{C} \in \mathbb{R}^{N_{\text{out}} \times N_{\text{in}}}$ be the linear transformation matrix representing a standard strided convolution $\mathbf{y} = \mathbf{C} \mathbf{x}$, with stride $s > 1$, kernel size $K$, and zero padding $p$.
By definition of the inner-product adjoint in Euclidean space:
$$\langle \mathbf{C} \mathbf{x}, \mathbf{y} \rangle_{\mathcal{Y}} = \mathbf{y}^T (\mathbf{C} \mathbf{x}) = (\mathbf{C}^T \mathbf{y})^T \mathbf{x} = \langle \mathbf{x}, \mathbf{C}^T \mathbf{y} \rangle_{\mathcal{X}}$$

The **Transposed Convolution** with parameters $(K, s, p)$ is defined as the linear map whose matrix representation is precisely the transpose $\mathbf{C}^T$:
$$\tilde{\mathbf{x}} = \mathbf{C}^T \mathbf{y}, \quad \mathbf{C}^T \in \mathbb{R}^{N_{\text{in}} \times N_{\text{out}}}$$

#### Step 2: Operational Equivalence to Dilated Unit-Stride Convolution
In a standard strided convolution with stride $s$, the matrix $\mathbf{C}$ is obtained by selecting every $s$-th row of the unstrided convolution matrix $\mathbf{C}_1$:
$$\mathbf{C} = \mathbf{S}_s \mathbf{C}_1$$
where $\mathbf{S}_s$ is the spatial downsampling (row-decimation) matrix.
Transposing this relation:
$$\mathbf{C}^T = \mathbf{C}_1^T \mathbf{S}_s^T$$

The operator $\mathbf{S}_s^T$ acts on the vector $\mathbf{y}$ by inserting $s - 1$ zeros between each adjacent spatial entry of $\mathbf{y}$. This is known as **spatial dilation** of the feature map:
$$\tilde{y}_{i \cdot s, j \cdot s} = y_{i, j}, \qquad \tilde{y}_{h, w} = 0 \quad \text{if } h \not\equiv 0 \pmod s \text{ or } w \not\equiv 0 \pmod s$$

Subsequently, $\mathbf{C}_1^T$ performs a unit-stride full cross-correlation with the spatially flipped kernel $\mathbf{K}^{\text{rot}}$.
Therefore:
$$\text{ConvTranspose2d}(\mathbf{Y}, \mathbf{K}, \text{stride}=s) \equiv \text{Conv2d}(\text{Dilation}_s(\mathbf{Y}), \mathbf{K}^{\text{rot}}, \text{stride}=1, \text{padding}=K-1-p)$$

#### Step 3: Spatial Output Dimension Formula
In the forward strided convolution, the output size satisfies $H_{\text{out}} = \lfloor \frac{H_{\text{in}} + 2p - K}{s} \rfloor + 1$.
Inverting this map for transposed convolution:
$$H_{\text{out}}^{\text{trans}} = (H_{\text{in}} - 1) \cdot s - 2p + K + p_{\text{out}}$$
where $p_{\text{out}} \in \{0, 1, \dots, s-1\}$ is the output padding parameter required to resolve the ambiguity of the floor function $\lfloor \cdot \rfloor$.

#### Step 4: Mathematical Origin of Checkerboard Artifacts
Consider a 1D transposed convolution with stride $s$ and kernel size $K$. Each output coordinate $n$ receives contributions from all input locations $m$ whose receptive fields cover $n$:
$$x_n = \sum_{m} y_m \, K_{n - m \cdot s} \cdot \mathbf{1}_{\{0 \le n - m \cdot s < K\}}$$

$$\mathcal{N}(n) = \left\{ m \in \mathbb{Z} : 0 \le n - m \cdot s < K \right\} = \left\{ m \in \mathbb{Z} : \frac{n - K + 1}{s} \le m \le \frac{n}{s} \right\}$$

The effective weight magnitude (or "stamping overlap") received by pixel $n$ is:
$$\Omega(n) = \sum_{m \in \mathcal{N}(n)} |K_{n - m \cdot s}|$$

If $K$ is **not an integer multiple of $s$** (e.g., $K = 3, s = 2$):
- For even $n = 2k$: $n \pmod 2 = 0 \implies m \in \{k, k-1\}$. Number of contributors $= 2$.
- For odd $n = 2k + 1$: $n \pmod 2 = 1 \implies m \in \{k\}$. Number of contributors $= 1$.

The number of contributing kernel weights alternates with period $s$:
$$|\mathcal{N}(n)| = \begin{cases} 2 & \text{if } n \text{ is even} \\ 1 & \text{if } n \text{ is odd} \end{cases}$$
In 2D, the overlap cardinality $|\mathcal{N}(i, j)| = |\mathcal{N}(i)| \cdot |\mathcal{N}(j)|$ creates a 2D checkerboard pattern with values alternating between $4$, $2$, and $1$ across adjacent pixels.
Even with uniform weights $K_{u, v} = 1$, the forward output oscillates sharply between $4$ and $1$, introducing high-frequency spatial noise into generated images.
This proves why models avoiding checkerboards set $K$ to an exact multiple of $s$ (e.g., $K = 4, s = 2$) or replace transposed convolutions with bilinear upsampling followed by standard convolution. $\blacksquare$

---

## 3. Geometric & Algebraic Interpretation

### The Adjoint Matrix View of Convolution

Consider a 1D convolution of input $\mathbf{x} = [x_0, x_1, x_2, x_3]^T$ with kernel $\mathbf{k} = [k_0, k_1]^T$, stride $1$, padding $0$.
The forward pass is the matrix-vector multiplication $\mathbf{y} = \mathbf{C}\mathbf{x}$:

$$\begin{bmatrix} y_0 \\ y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} k_0 & k_1 & 0 & 0 \\ 0 & k_0 & k_1 & 0 \\ 0 & 0 & k_0 & k_1 \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

Here, $\mathbf{C} \in \mathbb{R}^{3 \times 4}$ is a banded Toeplitz matrix.

By the chain rule of matrix calculus, given upstream gradient vector $\boldsymbol{\delta} = \left[\frac{\partial \mathcal{L}}{\partial y_0}, \frac{\partial \mathcal{L}}{\partial y_1}, \frac{\partial \mathcal{L}}{\partial y_2}\right]^T$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{C}^T \boldsymbol{\delta}$$

Transposing $\mathbf{C}$:
$$\mathbf{C}^T = \begin{bmatrix} k_0 & 0 & 0 \\ k_1 & k_0 & 0 \\ 0 & k_1 & k_0 \\ 0 & 0 & k_1 \end{bmatrix} \in \mathbb{R}^{4 \times 3}$$

Computing $\mathbf{C}^T \boldsymbol{\delta}$:
$$\begin{bmatrix} \frac{\partial \mathcal{L}}{\partial x_0} \\ \frac{\partial \mathcal{L}}{\partial x_1} \\ \frac{\partial \mathcal{L}}{\partial x_2} \\ \frac{\partial \mathcal{L}}{\partial x_3} \end{bmatrix} = \begin{bmatrix} k_0 \delta_0 \\ k_1 \delta_0 + k_0 \delta_1 \\ k_1 \delta_1 + k_0 \delta_2 \\ k_1 \delta_2 \end{bmatrix}$$

Notice what this matrix multiplication represents:
- It is a convolution of $\boldsymbol{\delta}$ with the flipped kernel $[k_1, k_0]^T$!
- $\delta$ is zero-padded on both sides: $[\dots, 0, \delta_0, \delta_1, \delta_2, 0, \dots]$.
- When the flipped kernel $[k_1, k_0]$ slides over the padded $\boldsymbol{\delta}$:
  - At position 0: $[k_1, k_0] \cdot [0, \delta_0] = k_0 \delta_0$.
  - At position 1: $[k_1, k_0] \cdot [\delta_0, \delta_1] = k_1 \delta_0 + k_0 \delta_1$.
  - At position 2: $[k_1, k_0] \cdot [\delta_1, \delta_2] = k_1 \delta_1 + k_0 \delta_2$.
  - At position 3: $[k_1, k_0] \cdot [\delta_2, 0] = k_1 \delta_2$.

The $180^\circ$ rotation is not an arbitrary rule—it is the algebraic necessity of the matrix transpose $\mathbf{C}^T$ on a Toeplitz operator!

---

## 4. Real-World Analogy

### 1. The Stage Spotlight vs. The Shadow Traceback
- **Forward Convolution (Spotlight):** A conical spotlight (the kernel) sweeps across a stage floor (input pixels). At each position, it integrates whatever actors/props stand within its beam into a single light intensity reading on an overhead camera sensor (output feature map).
- **Backward Input Gradient (Shadow Traceback):** A director on the lighting truss spots a flaw in the camera image (upstream loss gradient $\boldsymbol{\delta}$). To figure out which actor on stage contributed to that flaw, the light beam is projected *backwards* through the same cone aperture. Because the light travels in the opposite direction, the cone is inverted ($180^\circ$ flip), and actors where beams overlap receive illumination proportional to all camera sensors they hit.

### 2. The Railway Switching Junction (Max-Pooling)
- **Forward Max-Pooling:** Multiple train tracks enter a switching yard. A sensor picks the heaviest train (argmax) and routes it onto the main outbound line; all other trains are halted.
- **Backward Max-Pooling:** A telemetry command (gradient $\delta$) travels back along the outbound line. Because the physical switch remains thrown toward whichever track sent the heaviest train forward, the entire error signal routes back along that single track. The halted tracks receive zero signal.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical example with verified hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Convolution Backward |
| :--- | :--- | :--- | :--- |
| $\mathbf{X}$ | Input Matrix | $(3, 3)$ | Forward input feature map activations |
| $\mathbf{K}$ | Convolution Kernel | $(2, 2)$ | Learnable spatial filter weights |
| $\mathbf{Y}$ | Forward Output | $(2, 2)$ | Convolution output with $s=1, p=0$ |
| $\boldsymbol{\delta} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$ | Upstream Gradient | $(2, 2)$ | Sensitivity of scalar loss $\mathcal{L}$ w.r.t. output activations |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$ | Kernel Gradient | $(2, 2)$ | Accumulation of $\delta \cdot X$ patches over spatial receptive fields |
| $\mathbf{K}^{\text{rot}}$ | Rotated Kernel | $(2, 2)$ | Kernel rotated $180^\circ$ ($K^{\text{rot}}_{u, v} = K_{1-u, 1-v}$) |
| $\boldsymbol{\delta}^{\text{pad}}$ | Padded Gradient | $(4, 4)$ | Upstream gradient zero-padded with $p=1$ for full convolution |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ | Input Gradient | $(3, 3)$ | Sensitivity of scalar loss $\mathcal{L}$ w.r.t. input activations |
| $\mathbf{M}$ | Max-Pool Mask | $(4, 4)$ | Boolean mask where $M_{i,j}=1$ if $X_{i,j}$ was the maximum in window |

---

### 5.2 Concrete Toy Numbers

#### Input Map $\mathbf{X}$ ($3 \times 3$):
$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 0 \\ 3 & 1 & 4 \\ 0 & 2 & 1 \end{bmatrix}$$

#### Kernel $\mathbf{K}$ ($2 \times 2$):
$$\mathbf{K} = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix}$$

---

### 5.3 Forward Pass Computation ($s=1, p=0$)

$$Y_{h, w} = \sum_{u=0}^1 \sum_{v=0}^1 X_{h+u, w+v} \, K_{u, v}$$

1. **Top-Left ($h=0, w=0$):**
   $$Y_{0, 0} = (1)(2) + (2)(-1) + (3)(1) + (1)(3) = 2 - 2 + 3 + 3 = 6$$
2. **Top-Right ($h=0, w=1$):**
   $$Y_{0, 1} = (2)(2) + (0)(-1) + (1)(1) + (4)(3) = 4 - 0 + 1 + 12 = 17$$
3. **Bottom-Left ($h=1, w=0$):**
   $$Y_{1, 0} = (3)(2) + (1)(-1) + (0)(1) + (2)(3) = 6 - 1 + 0 + 6 = 11$$
4. **Bottom-Right ($h=1, w=1$):**
   $$Y_{1, 1} = (1)(2) + (4)(-1) + (2)(1) + (1)(3) = 2 - 4 + 2 + 3 = 3$$

$$\mathbf{Y} = \begin{bmatrix} 6 & 17 \\ 11 & 3 \end{bmatrix}$$

---

### 5.4 Given Upstream Gradient $\boldsymbol{\delta} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$

Suppose the upstream loss gradient is:
$$\boldsymbol{\delta} = \begin{bmatrix} 1 & -1 \\ 2 & 0 \end{bmatrix}$$

---

### 5.5 Backward Pass 1: Weight Gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$

$$\frac{\partial \mathcal{L}}{\partial K_{u, v}} = \sum_{h=0}^1 \sum_{w=0}^1 \delta_{h, w} \, X_{h+u, w+v}$$

1. **Cell $(0, 0)$ ($K_{0, 0}$):**
   $$\frac{\partial \mathcal{L}}{\partial K_{0, 0}} = \delta_{0, 0} X_{0, 0} + \delta_{0, 1} X_{0, 1} + \delta_{1, 0} X_{1, 0} + \delta_{1, 1} X_{1, 1}$$
   $$= (1)(1) + (-1)(2) + (2)(3) + (0)(1) = 1 - 2 + 6 + 0 = \mathbf{5}$$

2. **Cell $(0, 1)$ ($K_{0, 1}$):**
   $$\frac{\partial \mathcal{L}}{\partial K_{0, 1}} = \delta_{0, 0} X_{0, 1} + \delta_{0, 1} X_{0, 2} + \delta_{1, 0} X_{1, 1} + \delta_{1, 1} X_{1, 2}$$
   $$= (1)(2) + (-1)(0) + (2)(1) + (0)(4) = 2 - 0 + 2 + 0 = \mathbf{4}$$

3. **Cell $(1, 0)$ ($K_{1, 0}$):**
   $$\frac{\partial \mathcal{L}}{\partial K_{1, 0}} = \delta_{0, 0} X_{1, 0} + \delta_{0, 1} X_{1, 1} + \delta_{1, 0} X_{2, 0} + \delta_{1, 1} X_{2, 1}$$
   $$= (1)(3) + (-1)(1) + (2)(0) + (0)(2) = 3 - 1 + 0 + 0 = \mathbf{2}$$

4. **Cell $(1, 1)$ ($K_{1, 1}$):**
   $$\frac{\partial \mathcal{L}}{\partial K_{1, 1}} = \delta_{0, 0} X_{1, 1} + \delta_{0, 1} X_{1, 2} + \delta_{1, 0} X_{2, 1} + \delta_{1, 1} X_{2, 2}$$
   $$= (1)(1) + (-1)(4) + (2)(2) + (0)(1) = 1 - 4 + 4 + 0 = \mathbf{1}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \begin{bmatrix} 5 & 4 \\ 2 & 1 \end{bmatrix}$$

---

### 5.6 Backward Pass 2: Input Gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$

Using the rotated kernel $\mathbf{K}^{\text{rot}}$ and zero-padded upstream gradient $\boldsymbol{\delta}^{\text{pad}}$:

$$\mathbf{K} = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix} \implies \mathbf{K}^{\text{rot}} = \begin{bmatrix} 3 & 1 \\ -1 & 2 \end{bmatrix}$$

Padded gradient with $p=1$ ($4 \times 4$):
$$\boldsymbol{\delta}^{\text{pad}} = \begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 1 & -1 & 0 \\
0 & 2 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}$$

Convolving $\boldsymbol{\delta}^{\text{pad}}$ with $\mathbf{K}^{\text{rot}}$ gives the $3 \times 3$ gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$:

1. **Position $(0, 0)$:**
   Window: $\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} \implies (3)(0) + (1)(0) + (-1)(0) + (2)(1) = \mathbf{2}$
2. **Position $(0, 1)$:**
   Window: $\begin{bmatrix} 0 & 0 \\ 1 & -1 \end{bmatrix} \implies (3)(0) + (1)(0) + (-1)(1) + (2)(-1) = 0 + 0 - 1 - 2 = \mathbf{-3}$
3. **Position $(0, 2)$:**
   Window: $\begin{bmatrix} 0 & 0 \\ -1 & 0 \end{bmatrix} \implies (3)(0) + (1)(0) + (-1)(-1) + (2)(0) = \mathbf{1}$
4. **Position $(1, 0)$:**
   Window: $\begin{bmatrix} 0 & 1 \\ 0 & 2 \end{bmatrix} \implies (3)(0) + (1)(1) + (-1)(0) + (2)(2) = 0 + 1 + 0 + 4 = \mathbf{5}$
5. **Position $(1, 1)$:**
   Window: $\begin{bmatrix} 1 & -1 \\ 2 & 0 \end{bmatrix} \implies (3)(1) + (1)(-1) + (-1)(2) + (2)(0) = 3 - 1 - 2 + 0 = \mathbf{0}$
6. **Position $(1, 2)$:**
   Window: $\begin{bmatrix} -1 & 0 \\ 0 & 0 \end{bmatrix} \implies (3)(-1) + (1)(0) + (-1)(0) + (2)(0) = \mathbf{-3}$
7. **Position $(2, 0)$:**
   Window: $\begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix} \implies (3)(0) + (1)(2) + (-1)(0) + (2)(0) = \mathbf{2}$
8. **Position $(2, 1)$:**
   Window: $\begin{bmatrix} 2 & 0 \\ 0 & 0 \end{bmatrix} \implies (3)(2) + (1)(0) + (-1)(0) + (2)(0) = \mathbf{6}$
9. **Position $(2, 2)$:**
   Window: $\begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix} \implies (3)(0) + (1)(0) + (-1)(0) + (2)(0) = \mathbf{0}$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \begin{bmatrix} 2 & -3 & 1 \\ 5 & 0 & -3 \\ 2 & 6 & 0 \end{bmatrix}$$

---

### 5.7 Max-Pooling $2 \times 2$ Forward & Backward Grid

Consider a $4 \times 4$ feature map $\mathbf{Z}$:
$$\mathbf{Z} = \begin{bmatrix}
1 & 5 & 2 & 3 \\
4 & 2 & 8 & 1 \\
3 & 7 & 4 & 6 \\
0 & 2 & 9 & 5
\end{bmatrix}$$

Applying $2 \times 2$ Max-Pooling with stride $2$:
- **Top-Left Window:** $\begin{bmatrix} 1 & 5 \\ 4 & 2 \end{bmatrix} \implies \max = 5$ at index $(0, 1)$
- **Top-Right Window:** $\begin{bmatrix} 2 & 3 \\ 8 & 1 \end{bmatrix} \implies \max = 8$ at index $(1, 2)$
- **Bottom-Left Window:** $\begin{bmatrix} 3 & 7 \\ 0 & 2 \end{bmatrix} \implies \max = 7$ at index $(2, 1)$
- **Bottom-Right Window:** $\begin{bmatrix} 4 & 6 \\ 9 & 5 \end{bmatrix} \implies \max = 9$ at index $(3, 2)$

Forward Output $\mathbf{P} \in \mathbb{R}^{2 \times 2}$:
$$\mathbf{P} = \begin{bmatrix} 5 & 8 \\ 7 & 9 \end{bmatrix}$$

Argmax Boolean Mask $\mathbf{M} \in \{0, 1\}^{4 \times 4}$:
$$\mathbf{M} = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}$$

Given upstream gradient $\boldsymbol{\delta}_P = \begin{bmatrix} 10 & -4 \\ 3 & 7 \end{bmatrix}$:
The backward pass distributes the gradients solely to the masked winning locations:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} = \begin{bmatrix}
0 & 10 & 0 & 0 \\
0 & 0 & -4 & 0 \\
0 & 3 & 0 & 0 \\
0 & 0 & 7 & 0
\end{bmatrix}$$

---

## 6. Solved Illustrations

### Illustration 1: Multi-Channel Gradient Accumulation

**Problem:**
Suppose an input has $C_{\text{in}} = 2$ channels, and a layer produces $C_{\text{out}} = 1$ channel using two $2 \times 2$ kernels $\mathbf{K}_0$ and $\mathbf{K}_1$. Given upstream gradient $\boldsymbol{\delta}$, explain why $\frac{\partial \mathcal{L}}{\partial \mathbf{X}_0}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{X}_1}$ are decoupled, while kernel updates depend on their respective channels.

**Solution:**
The forward equation is:
$$Y_{h, w} = (\mathbf{X}_0 \star \mathbf{K}_0)_{h, w} + (\mathbf{X}_1 \star \mathbf{K}_1)_{h, w} + b$$
Taking partial derivatives:
1. With respect to $\mathbf{X}_0$:
   $$\frac{\partial Y_{h, w}}{\partial X_{0, i, j}} = K_{0, i - h, j - w}$$
   Channel 1 ($\mathbf{X}_1$) has identically zero derivative w.r.t. $X_{0, i, j}$.
   Therefore:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{X}_0} = \boldsymbol{\delta} * \mathbf{K}_0^{\text{rot}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{X}_1} = \boldsymbol{\delta} * \mathbf{K}_1^{\text{rot}}$$
   The backward pass splits across input channels independently.

2. With respect to kernels:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{K}_0} = \mathbf{X}_0 \star \boldsymbol{\delta}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{K}_1} = \mathbf{X}_1 \star \boldsymbol{\delta}$$
   Each kernel receives gradient from its dedicated input channel correlated with the shared upstream error.

---

### Illustration 2: Overlapping Max-Pooling Windows (Stride < Filter Size)

**Problem:**
Let $X = [1, 5, 3]$, and apply $1\text{D}$ max-pooling with pool size $2$ and stride $1$.
1. Compute forward outputs $Y_0, Y_1$.
2. Given upstream gradients $\delta = [\delta_0, \delta_1] = [4, 6]$, compute $\frac{\partial \mathcal{L}}{\partial X}$.

**Solution:**
1. Forward pass:
   - Window 0: $\max(X_0, X_1) = \max(1, 5) = 5$ at index $1$. Thus $Y_0 = 5$.
   - Window 1: $\max(X_1, X_2) = \max(5, 3) = 5$ at index $1$. Thus $Y_1 = 5$.
2. Backward pass:
   Index $1$ ($X_1$) was the maximum in *both* windows!
   - Window 0 routes $\delta_0 = 4$ to $X_1$.
   - Window 1 routes $\delta_1 = 6$ to $X_1$.
   Accumulating across all windows covering index $i$:
   $$\frac{\partial \mathcal{L}}{\partial X_0} = 0$$
   $$\frac{\partial \mathcal{L}}{\partial X_1} = \delta_0 + \delta_1 = 4 + 6 = 10$$
   $$\frac{\partial \mathcal{L}}{\partial X_2} = 0$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = [0, 10, 0]$$
When strides are smaller than the pooling window size, overlapping argmax selections **accumulate additively**.

---

### Illustration 3: Complete Hand-Calculated Conv2D Backward Pass with Boundary Reflections

**Problem:**
Let the input $\mathbf{X} \in \mathbb{R}^{3 \times 3}$ and convolutional filter $\mathbf{K} \in \mathbb{R}^{2 \times 2}$ be:
$$\mathbf{X} = \begin{bmatrix} 2 & 1 & 3 \\ 0 & 4 & 1 \\ 1 & 2 & 0 \end{bmatrix}, \qquad \mathbf{K} = \begin{bmatrix} 1 & -2 \\ 3 & 1 \end{bmatrix}$$
with stride $s = 1$ and valid padding $p = 0$.
1. Compute the forward output map $\mathbf{Y} \in \mathbb{R}^{2 \times 2}$.
2. Given the upstream loss gradient:
$$\boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix}$$
compute the exact kernel gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{K}} \in \mathbb{R}^{2 \times 2}$.
3. Construct the $180^\circ$ rotated filter $\mathbf{K}^{\text{rot}}$ and zero-padded gradient $\boldsymbol{\delta}^{\text{pad}}$, and compute all $9$ elements of the input gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{X}} \in \mathbb{R}^{3 \times 3}$.
4. Verify the conservation identity $\langle \boldsymbol{\delta}, \mathbf{Y} \rangle = \langle \frac{\partial \mathcal{L}}{\partial \mathbf{K}}, \mathbf{K} \rangle = \langle \frac{\partial \mathcal{L}}{\partial \mathbf{X}}, \mathbf{X} \rangle$.

**Solution:**

#### Step 1: Forward Pass
Using $Y_{h, w} = \sum_{u=0}^1 \sum_{v=0}^1 X_{h+u, w+v} K_{u, v}$:
- $Y_{0, 0} = (2)(1) + (1)(-2) + (0)(3) + (4)(1) = 2 - 2 + 0 + 4 = 4$
- $Y_{0, 1} = (1)(1) + (3)(-2) + (4)(3) + (1)(1) = 1 - 6 + 12 + 1 = 8$
- $Y_{1, 0} = (0)(1) + (4)(-2) + (1)(3) + (2)(1) = 0 - 8 + 3 + 2 = -3$
- $Y_{1, 1} = (4)(1) + (1)(-2) + (2)(3) + (0)(1) = 4 - 2 + 6 + 0 = 8$

$$\mathbf{Y} = \begin{bmatrix} 4 & 8 \\ -3 & 8 \end{bmatrix}$$

#### Step 2: Weight Gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \mathbf{X} \star \boldsymbol{\delta}$
Using $\frac{\partial \mathcal{L}}{\partial K_{u, v}} = \sum_{h=0}^1 \sum_{w=0}^1 \delta_{h, w} X_{h+u, w+v}$:
- Cell $(0, 0)$:
  $$\frac{\partial \mathcal{L}}{\partial K_{0, 0}} = \delta_{0, 0} X_{0, 0} + \delta_{0, 1} X_{0, 1} + \delta_{1, 0} X_{1, 0} + \delta_{1, 1} X_{1, 1} = 2(2) + (-1)(1) + 1(0) + 3(4) = 4 - 1 + 0 + 12 = \mathbf{15}$$
- Cell $(0, 1)$:
  $$\frac{\partial \mathcal{L}}{\partial K_{0, 1}} = \delta_{0, 0} X_{0, 1} + \delta_{0, 1} X_{0, 2} + \delta_{1, 0} X_{1, 1} + \delta_{1, 1} X_{1, 2} = 2(1) + (-1)(3) + 1(4) + 3(1) = 2 - 3 + 4 + 3 = \mathbf{6}$$
- Cell $(1, 0)$:
  $$\frac{\partial \mathcal{L}}{\partial K_{1, 0}} = \delta_{0, 0} X_{1, 0} + \delta_{0, 1} X_{1, 1} + \delta_{1, 0} X_{2, 0} + \delta_{1, 1} X_{2, 1} = 2(0) + (-1)(4) + 1(1) + 3(2) = 0 - 4 + 1 + 6 = \mathbf{3}$$
- Cell $(1, 1)$:
  $$\frac{\partial \mathcal{L}}{\partial K_{1, 1}} = \delta_{0, 0} X_{1, 1} + \delta_{0, 1} X_{1, 2} + \delta_{1, 0} X_{2, 1} + \delta_{1, 1} X_{2, 2} = 2(4) + (-1)(1) + 1(2) + 3(0) = 8 - 1 + 2 + 0 = \mathbf{9}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \begin{bmatrix} 15 & 6 \\ 3 & 9 \end{bmatrix}$$

#### Step 3: Input Gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \text{FullConv}(\boldsymbol{\delta}, \mathbf{K}^{\text{rot}})$
Rotating $\mathbf{K}$ by $180^\circ$:
$$\mathbf{K}^{\text{rot}} = \begin{bmatrix} K_{1, 1} & K_{1, 0} \\ K_{0, 1} & K_{0, 0} \end{bmatrix} = \begin{bmatrix} 1 & 3 \\ -2 & 1 \end{bmatrix}$$

Zero-padding $\boldsymbol{\delta}$ with $p = K - 1 = 1$ gives the $4 \times 4$ padded array:
$$\boldsymbol{\delta}^{\text{pad}} = \begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 2 & -1 & 0 \\
0 & 1 & 3 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}$$

Computing the 9 spatial outputs by sliding $\mathbf{K}^{\text{rot}}$ over $\boldsymbol{\delta}^{\text{pad}}$:
1. $(0, 0)$: $1(0) + 3(0) - 2(0) + 1(2) = \mathbf{2}$
2. $(0, 1)$: $1(0) + 3(0) - 2(2) + 1(-1) = 0 - 4 - 1 = \mathbf{-5}$
3. $(0, 2)$: $1(0) + 3(0) - 2(-1) + 1(0) = 0 + 2 + 0 = \mathbf{2}$
4. $(1, 0)$: $1(0) + 3(2) - 2(0) + 1(1) = 0 + 6 + 0 + 1 = \mathbf{7}$
5. $(1, 1)$: $1(2) + 3(-1) - 2(1) + 1(3) = 2 - 3 - 2 + 3 = \mathbf{0}$
6. $(1, 2)$: $1(-1) + 3(0) - 2(3) + 1(0) = -1 + 0 - 6 + 0 = \mathbf{-7}$
7. $(2, 0)$: $1(0) + 3(1) - 2(0) + 1(0) = 0 + 3 - 0 + 0 = \mathbf{3}$
8. $(2, 1)$: $1(1) + 3(3) - 2(0) + 1(0) = 1 + 9 - 0 + 0 = \mathbf{10}$
9. $(2, 2)$: $1(3) + 3(0) - 2(0) + 1(0) = 3 + 0 - 0 + 0 = \mathbf{3}$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \begin{bmatrix} 2 & -5 & 2 \\ 7 & 0 & -7 \\ 3 & 10 & 3 \end{bmatrix}$$

#### Step 4: Conservation Identity Verification
Because the forward convolution is bilinear, Euler's homogeneous function theorem requires that the inner products match:
1. $\langle \boldsymbol{\delta}, \mathbf{Y} \rangle = 2(4) + (-1)(8) + 1(-3) + 3(8) = 8 - 8 - 3 + 24 = \mathbf{21}$
2. $\langle \frac{\partial \mathcal{L}}{\partial \mathbf{K}}, \mathbf{K} \rangle = 15(1) + 6(-2) + 3(3) + 9(1) = 15 - 12 + 9 + 9 = \mathbf{21}$
3. $\langle \frac{\partial \mathcal{L}}{\partial \mathbf{X}}, \mathbf{X} \rangle = 2(2) + (-5)(1) + 2(3) + 7(0) + 0(4) + (-7)(1) + 3(1) + 10(2) + 3(0) = 4 - 5 + 6 + 0 + 0 - 7 + 3 + 20 + 0 = \mathbf{21}$

All three inner products evaluate to exactly $21$, verifying mathematical consistency.

---

### Illustration 4: Max-Pooling vs. Average-Pooling Backward Pass on a 4x4 Feature Map

**Problem:**
Consider the $4 \times 4$ activation map:
$$\mathbf{X} = \begin{bmatrix}
3 & 1 & 2 & 4 \\
5 & 2 & 1 & 6 \\
4 & 8 & 3 & 2 \\
1 & 7 & 5 & 9
\end{bmatrix}$$
1. Apply $2 \times 2$ Max-Pooling with stride $2$. Record the winning spatial argmax coordinates and the forward output $\mathbf{Y}_{\text{max}} \in \mathbb{R}^{2 \times 2}$.
2. Apply $2 \times 2$ Average-Pooling with stride $2$. Compute the forward output $\mathbf{Y}_{\text{avg}} \in \mathbb{R}^{2 \times 2}$.
3. Given the upstream gradient:
$$\boldsymbol{\delta} = \begin{bmatrix} 4 & -2 \\ 6 & 2 \end{bmatrix}$$
compute the input gradients $\nabla_{\mathbf{X}}^{\text{max}} \mathcal{L}$ and $\nabla_{\mathbf{X}}^{\text{avg}} \mathcal{L}$. Compare their sparsity and gradient sum conservation.

**Solution:**

#### Step 1: Forward Max-Pooling
- Window $(0, 0)$ covers $\{X_{0,0}=3, X_{0,1}=1, X_{1,0}=5, X_{1,1}=2\} \implies \max = 5$ at $(1, 0)$.
- Window $(0, 1)$ covers $\{X_{0,2}=2, X_{0,3}=4, X_{1,2}=1, X_{1,3}=6\} \implies \max = 6$ at $(1, 3)$.
- Window $(1, 0)$ covers $\{X_{2,0}=4, X_{2,1}=8, X_{3,0}=1, X_{3,1}=7\} \implies \max = 8$ at $(2, 1)$.
- Window $(1, 1)$ covers $\{X_{2,2}=3, X_{2,3}=2, X_{3,2}=5, X_{3,3}=9\} \implies \max = 9$ at $(3, 3)$.

$$\mathbf{Y}_{\text{max}} = \begin{bmatrix} 5 & 6 \\ 8 & 9 \end{bmatrix}$$

#### Step 2: Forward Average Pooling
- Window $(0, 0)$: $\frac{3 + 1 + 5 + 2}{4} = \frac{11}{4} = \mathbf{2.75}$
- Window $(0, 1)$: $\frac{2 + 4 + 1 + 6}{4} = \frac{13}{4} = \mathbf{3.25}$
- Window $(1, 0)$: $\frac{4 + 8 + 1 + 7}{4} = \frac{20}{4} = \mathbf{5.00}$
- Window $(1, 1)$: $\frac{3 + 2 + 5 + 9}{4} = \frac{19}{4} = \mathbf{4.75}$

$$\mathbf{Y}_{\text{avg}} = \begin{bmatrix} 2.75 & 3.25 \\ 5.00 & 4.75 \end{bmatrix}$$

#### Step 3: Backward Pass Comparison
- **Max-Pooling Backward:** Routes each $\delta_{h, w}$ exclusively to the recorded argmax coordinates:
  - $(1, 0)$ receives $\delta_{0, 0} = 4$
  - $(1, 3)$ receives $\delta_{0, 1} = -2$
  - $(2, 1)$ receives $\delta_{1, 0} = 6$
  - $(3, 3)$ receives $\delta_{1, 1} = 2$
  All other $12$ positions are $0$:
  $$\nabla_{\mathbf{X}}^{\text{max}} \mathcal{L} = \begin{bmatrix}
  0 & 0 & 0 & 0 \\
  4 & 0 & 0 & -2 \\
  0 & 6 & 0 & 0 \\
  0 & 0 & 0 & 2
  \end{bmatrix}$$

- **Average-Pooling Backward:** Distributes $\frac{\delta_{h, w}}{4}$ uniformly across each $2 \times 2$ block:
  - Block $(0, 0)$: $\frac{4}{4} = 1.0$
  - Block $(0, 1)$: $\frac{-2}{4} = -0.5$
  - Block $(1, 0)$: $\frac{6}{4} = 1.5$
  - Block $(1, 1)$: $\frac{2}{4} = 0.5$
  $$\nabla_{\mathbf{X}}^{\text{avg}} \mathcal{L} = \begin{bmatrix}
  1.0 & 1.0 & -0.5 & -0.5 \\
  1.0 & 1.0 & -0.5 & -0.5 \\
  1.5 & 1.5 & 0.5 & 0.5 \\
  1.5 & 1.5 & 0.5 & 0.5
  \end{bmatrix}$$

- **Comparison:**
  - **Sparsity:** Max-pooling produces $12/16 = 75\%$ zero gradients, focusing learning on peak feature detectors. Average-pooling produces $0\%$ sparsity.
  - **Total Gradient Conservation:**
    $$\sum_{i, j} (\nabla_{\mathbf{X}}^{\text{max}} \mathcal{L})_{i, j} = 4 + (-2) + 6 + 2 = \mathbf{10}$$
    $$\sum_{i, j} (\nabla_{\mathbf{X}}^{\text{avg}} \mathcal{L})_{i, j} = 4(1.0) + 4(-0.5) + 4(1.5) + 4(0.5) = 4 - 2 + 6 + 2 = \mathbf{10}$$
    Both operations conserve total gradient sum $\sum \delta_{h, w} = 10$.

---

### Illustration 5: Transposed Convolution (Fractionally Strided) Forward Pass as Adjoint Operator

**Problem:**
Let an input feature map $\mathbf{Z} \in \mathbb{R}^{2 \times 2}$ and kernel $\mathbf{K} \in \mathbb{R}^{2 \times 2}$ be:
$$\mathbf{Z} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad \mathbf{K} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$$
We perform a transposed convolution (`ConvTranspose2d`) with stride $s = 2$ and padding $p = 0$.
1. Determine the exact output spatial dimensions $H_{\text{out}} \times W_{\text{out}}$.
2. Trace the output computation via the "stamped kernel patch" formulation.
3. Verify the output using the dilated convolution equivalence $\text{Conv2d}(\text{Dilation}_2(\mathbf{Z}), \mathbf{K}^{\text{rot}}, s=1, p=1)$.

**Solution:**

#### Step 1: Spatial Dimensions
Using the transposed convolution output size formula with $p=0, p_{\text{out}}=0$:
$$H_{\text{out}} = (H_{\text{in}} - 1) \cdot s - 2p + K_H = (2 - 1) \cdot 2 - 0 + 2 = 2 + 2 = 4$$
$$W_{\text{out}} = (W_{\text{in}} - 1) \cdot s - 2p + K_W = (2 - 1) \cdot 2 - 0 + 2 = 4$$
The output map $\mathbf{Y}$ has shape $4 \times 4$.

#### Step 2: Stamped Kernel Patch Formulation
In transposed convolution, each input cell $Z_{i, j}$ scales the kernel $\mathbf{K}$, and stamps the resulting $2 \times 2$ patch onto the output canvas starting at top-left index $(i \cdot s, j \cdot s) = (2i, 2j)$:
1. Input $Z_{0, 0} = 1$ stamps $1 \cdot \mathbf{K}$ at rows $0:2$, cols $0:2$:
   $$\mathbf{P}_{0, 0} = \begin{bmatrix} 2 & 1 & 0 & 0 \\ 0 & 3 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$
2. Input $Z_{0, 1} = 2$ stamps $2 \cdot \mathbf{K} = \begin{bmatrix} 4 & 2 \\ 0 & 6 \end{bmatrix}$ at rows $0:2$, cols $2:4$:
   $$\mathbf{P}_{0, 1} = \begin{bmatrix} 0 & 0 & 4 & 2 \\ 0 & 0 & 0 & 6 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$
3. Input $Z_{1, 0} = 3$ stamps $3 \cdot \mathbf{K} = \begin{bmatrix} 6 & 3 \\ 0 & 9 \end{bmatrix}$ at rows $2:4$, cols $0:2$:
   $$\mathbf{P}_{1, 0} = \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 6 & 3 & 0 & 0 \\ 0 & 9 & 0 & 0 \end{bmatrix}$$
4. Input $Z_{1, 1} = 4$ stamps $4 \cdot \mathbf{K} = \begin{bmatrix} 8 & 4 \\ 0 & 12 \end{bmatrix}$ at rows $2:4$, cols $2:4$:
   $$\mathbf{P}_{1, 1} = \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 8 & 4 \\ 0 & 0 & 0 & 12 \end{bmatrix}$$

Because stride $s = 2 \ge K = 2$, there is zero spatial overlap between adjacent stamped patches. Summing $\mathbf{Y} = \sum \mathbf{P}_{i, j}$:
$$\mathbf{Y} = \begin{bmatrix}
2 & 1 & 4 & 2 \\
0 & 3 & 0 & 6 \\
6 & 3 & 8 & 4 \\
0 & 9 & 0 & 12
\end{bmatrix}$$

#### Step 3: Verification via Dilation and Rotated Convolution
Dilating $\mathbf{Z}$ by inserting $s - 1 = 1$ zero between elements yields a $3 \times 3$ grid:
$$\tilde{\mathbf{Z}} = \begin{bmatrix}
1 & 0 & 2 \\
0 & 0 & 0 \\
3 & 0 & 4
\end{bmatrix}$$
The rotated kernel is:
$$\mathbf{K}^{\text{rot}} = \begin{bmatrix} 3 & 0 \\ 1 & 2 \end{bmatrix}$$
Applying full padding $p = K - 1 = 1$ to $\tilde{\mathbf{Z}}$ yields a $5 \times 5$ padded grid, and convolving with $\mathbf{K}^{\text{rot}}$ gives output size $5 - 2 + 1 = 4$:
- Position $(0, 0)$: $\mathbf{K}^{\text{rot}}$ overlaps only $\tilde{Z}_{0, 0} = 1$ at its bottom-right cell ($K^{\text{rot}}_{1, 1} = 2$): $(1)(2) = \mathbf{2}$.
- Position $(0, 1)$: $\mathbf{K}^{\text{rot}}$ bottom-left cell ($K^{\text{rot}}_{1, 0} = 1$) covers $\tilde{Z}_{0, 0}=1$: $(1)(1) = \mathbf{1}$.
- Position $(0, 2)$: $\mathbf{K}^{\text{rot}}$ bottom-right cell covers $\tilde{Z}_{0, 2}=2$: $(2)(2) = \mathbf{4}$.
- Position $(0, 3)$: $\mathbf{K}^{\text{rot}}$ bottom-left cell covers $\tilde{Z}_{0, 2}=2$: $(2)(1) = \mathbf{2}$.
This precisely reproduces the first row $[2, 1, 4, 2]$, confirming the exact algebraic equivalence of the adjoint operator.

---

## 7. Deep Learning Connection & Application

### 1. High-Performance Implementations in cuDNN
In production deep learning frameworks (PyTorch, TensorFlow, JAX), convolution backward passes are not implemented as nested Python loops. NVIDIA cuDNN provides specialized primitives:
- `cudnnConvolutionForward`: Computes $\mathbf{Y}$ via im2col GEMM or Winograd $F(2 \times 2, 3 \times 3)$.
- `cudnnConvolutionBackwardData`: Computes $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ via transposed GEMM:
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{\text{col}}} = \mathbf{K}^T \cdot \boldsymbol{\delta}_{\text{col}} \xrightarrow{\text{col2im}} \frac{\partial \mathcal{L}}{\partial \mathbf{X}}$$
- `cudnnConvolutionBackwardFilter`: Computes $\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$:
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \boldsymbol{\delta}_{\text{col}} \cdot \mathbf{X}_{\text{col}}^T$$

The `col2im` transformation inverts `im2col` by accumulating overlapping receptive field patches back into the full input gradient tensor.

### 2. Transposed Convolution Artifacts (Checkerboard Artifacts)
When designing Generative Adversarial Networks (e.g., DCGAN) or image super-resolution networks, generators employ transposed convolutions (`nn.ConvTranspose2d`) to upsample feature maps.
If kernel size is not divisible by the stride (e.g., $3 \times 3$ kernel with stride $2$), receptive fields in the backward/transposed pass overlap unevenly. Some pixels receive contributions from $4$ kernel cells, while adjacent pixels receive contributions from only $1$ or $2$, producing notorious **checkerboard artifacts**. Modern generative vision models (e.g., StyleGAN, Stable Diffusion) mitigate this by replacing transposed convolutions with bilinear/nearest-neighbor upsampling followed by standard unit-stride convolutions.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Pure NumPy convolution backward pass for weights, biases, and inputs using vectorized im2col and col2im.
2. Max-pooling and Average-pooling forward and backward passes with argmax mask caching.
3. Verification against the Part 5 hand-calculated numbers to machine precision.
4. Exact numerical gradient checks via finite differences.
5. End-to-end parity validation against PyTorch's `torch.autograd.grad` across multi-channel, mini-batched tensors.

See implementation in:
[`07_convolutional_networks/code/02_backpropagation_through_convolutions_and_pooling.py`](./code/02_backpropagation_through_convolutions_and_pooling.py)
