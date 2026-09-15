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
