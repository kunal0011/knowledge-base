# 7.1 The Convolution Operation (Strides, Padding, Dilation, Receptive Field)

---

## Part 1: Intuition & 101 Motivation

In Module 6, we studied dense Multilayer Perceptrons (MLPs). While MLPs are universal function approximators, applying them directly to images, audio spectrograms, and spatial video data results in immediate computational and architectural failure:

1. **Parameter Explosion**: A modest color image of resolution $1024 \times 1024 \times 3$ contains over 3.1 million input values. Connecting this input to a single hidden layer of just 1,000 neurons requires:
   $$3{,}145{,}728 \times 1{,}000 \approx \mathbf{3.14\text{ Billion Parameters}}$$
   for that single layer alone! Storing and training such a network exceeds GPU memory and causes severe overfitting.
2. **Spatial Topology Blindness**: An MLP flattens a 2D image grid into an unstructured 1D vector. If you randomly permute the pixel locations across the entire image, an MLP's capacity to fit the data is completely unchanged. It is fundamentally blind to the fact that adjacent pixels share high spatial correlation.
3. **Translation Non-Invariance**: In an MLP, a cat positioned in the top-left corner activates an entirely different set of weights than the exact same cat positioned in the bottom-right corner. The network must re-learn what a cat looks like at every single pixel coordinate.

```
MLP (Dense): Every pixel connected to every neuron.   CNN (Convolution): Local kernel sweeps with shared weights.
===================================================   =========================================================
Image (1D flattened) --------> Dense Weights (Billions)   Image (2D Grid)        Kernel (e.g. 3x3)
[ p_1, p_2, ..., p_M ]       W \in R^{N \times M}         [ .  .  .  . ]         [ w_1 w_2 w_3 ]
No spatial structure.        Translation blind.           [ . [x  x  x] . ]  *   [ w_4 w_5 w_6 ]
                                                          [ . [x  x  x] . ]      [ w_7 w_8 w_9 ]
                                                          [ . [x  x  x] . ]   (Shared across all patches!)
```

To solve this, Yann LeCun (1989, 1998) introduced **Convolutional Neural Networks (CNNs)**, embedding two profound **inductive biases** directly into the architecture:
- **Local Connectivity**: Each neuron connects only to a small local spatial neighborhood (e.g., $3 \times 3$ or $5 \times 5$ pixels) known as its **receptive field**.
- **Weight Sharing (Translation Equivariance)**: The exact same parameter filter (kernel) sweeps across the entire image grid. If a filter learns to detect a horizontal edge or a cat's ear, it detects that feature identically anywhere it appears in the image!

---

## Part 2: Rigorous Mathematical Formulation

### 1. Discrete 2D Cross-Correlation vs. Convolution

In continuous functional analysis, the 2D convolution of an image $I$ with a filter kernel $K$ is defined by:
$$(I * K)(x, y) = \int_{-\infty}^\infty \int_{-\infty}^\infty I(\tau_1, \tau_2) K(x - \tau_1, y - \tau_2) \, d\tau_1 d\tau_2$$

Notice the minus signs: mathematical convolution **flips the kernel** along both horizontal and vertical axes before computing the inner product ($K(-u, -v)$).

In deep learning libraries (PyTorch, TensorFlow, JAX), what is called "convolution" is mathematically **discrete 2D cross-correlation**:
$$S(i, j) = (I \star K)(i, j) = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} I(i + m, j + n) K(m, n)$$
- **Why no flip?** Because filter weights $K(m, n)$ are learned from scratch via backpropagation. Learning an unflipped kernel is mathematically isomorphic to learning a flipped kernel. Omitting the flip saves unnecessary index operations.

---

### 2. Multi-Channel 2D Convolution (Batched Formulation)

In real-world computer vision, inputs and hidden representations are multi-channel 3D tensors:
- **Input Tensor**: $X \in \mathbb{R}^{C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$ (e.g., $C_{\text{in}} = 3$ for RGB color channels).
- **Filter Bank (Weights)**: $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$, containing $C_{\text{out}}$ distinct 3D filters, each having depth $C_{\text{in}}$.
- **Bias Vector**: $b \in \mathbb{R}^{C_{\text{out}}}$.
- **Output Feature Map**: $Y \in \mathbb{R}^{C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}}$.

The forward operation for output channel $c_{\text{out}}$ at spatial coordinate $(i, j)$ is:
$$\mathbf{Y(c_{\text{out}}, i, j) = b(c_{\text{out}}) + \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} X(c_{\text{in}}, \; i \cdot s_h + m, \; j \cdot s_w + n) \cdot W(c_{\text{out}}, c_{\text{in}}, m, n)}$$
where $s_h, s_w$ denote vertical and horizontal **strides**.

---

### 3. Output Dimension Formula (Padding, Stride, Dilation)

Let:
- $H_{\text{in}}, W_{\text{in}}$ be the input height and width.
- $k_h, k_w$ be the kernel height and width.
- $p_h, p_w$ be the padding added to top/bottom and left/right.
- $s_h, s_w$ be the stride (step size of filter movement).
- $d_h, d_w$ be the dilation rate (spacing between kernel elements).

The effective kernel size under dilation is:
$$\tilde{k} = d(k - 1) + 1$$

The exact spatial dimensions of the output feature map are:
$$\mathbf{H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p_h - d_h(k_h - 1) - 1}{s_h} \right\rfloor + 1}$$
$$\mathbf{W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2p_w - d_w(k_w - 1) - 1}{s_w} \right\rfloor + 1}$$

#### Standard Padding Modes:
1. **Valid Padding ($p = 0, s = 1, d = 1$)**:
   $$H_{\text{out}} = H_{\text{in}} - k + 1$$
   No zero-padding. The feature map shrinks at every layer.
2. **Same Padding ($s = 1, d = 1$, odd $k$)**:
   To ensure $H_{\text{out}} = H_{\text{in}}$, we choose padding:
   $$p = \frac{k - 1}{2}$$
   For a $3 \times 3$ kernel, $p = 1$. For a $5 \times 5$ kernel, $p = 2$. Spatial resolution is perfectly preserved!

---

### 4. The Receptive Field (RF) Recurrence

The **Receptive Field** $r_l$ of a neuron in layer $l$ is the spatial diameter of the region in the original input image $X^{(0)}$ that directly influences that neuron's activation.

```
Receptive Field Expansion Across Layers:
Layer 2 (Neuron):         [ o ]          Receptive Field = 5x5
                         /  |  \
Layer 1 (Neurons):     [o] [o] [o]       Receptive Field = 3x3
                      / | \
Layer 0 (Input Image):[x][x][x][x][x]    Receptive Field = 1x1 (Raw Pixels)
```

#### The General Recursive Receptive Field Equations:
For layer $l \ge 1$ with kernel size $k_l$, stride $s_l$, and cumulative stride $j_{l-1}$:

1. **Cumulative Stride (Jump $j_l$)**:
   $$j_0 = 1, \quad \mathbf{j_l = j_{l-1} \cdot s_l}$$
   The distance between adjacent receptive field centers in the input image.

2. **Receptive Field Diameter ($r_l$)**:
   $$r_0 = 1, \quad \mathbf{r_l = r_{l-1} + (k_l - 1) \cdot j_{l-1}}$$

3. **Effective Receptive Field with Dilation $d_l$**:
   $$r_l = r_{l-1} + (d_l(k_l - 1)) \cdot j_{l-1}$$

#### Profound Insight: Stacking $3 \times 3$ Convolutions
- Stacking two $3 \times 3$ conv layers with stride 1 yields:
  $$r_1 = 1 + (3 - 1)(1) = 3$$
  $$r_2 = 3 + (3 - 1)(1) = 5$$
- A stack of two $3 \times 3$ layers has the **exact same receptive field ($5 \times 5$) as a single $5 \times 5$ conv layer!**
- Parameter cost:
  - Single $5 \times 5$ layer: $5 \times 5 \times C^2 = \mathbf{25 C^2}$.
  - Two $3 \times 3$ layers: $2 \times (3 \times 3 \times C^2) = \mathbf{18 C^2}$ (**28% fewer parameters!**).
  - Plus: Two layers incorporate **two non-linear activations** instead of one, exponentially increasing expressive capacity! This is the architectural secret of **VGG and ResNet**.

---

### 5. Efficient GPU Execution: `im2col` & GEMM

On modern GPUs (NVIDIA Tensor Cores), 2D convolutions are never executed via naive nested `for` loops.
Instead, they are transformed into a single **General Matrix Multiply (GEMM)** using the **`im2col` (Image-to-Column)** operator (Chetlur et al., cuDNN 2014):

```
                               THE im2col TRANSFORMATION
                               
   Input Image (C x H x W)                     Unrolled Patch Matrix (im2col)
   ┌───┬───┬───┐                              ┌───────────────────────────────┐
   │ 1 │ 2 │ 3 │                              │ Patch 1: [ 1, 2, 4, 5 ]       │
   ├───┼───┼───┤   Slide 2x2 window           │ Patch 2: [ 2, 3, 5, 6 ]       │
   │ 4 │ 5 │ 6 │ ───────────────────────────> │ Patch 3: [ 4, 5, 7, 8 ]       │
   ├───┼───┼───┤                              │ Patch 4: [ 5, 6, 8, 9 ]       │
   │ 7 │ 8 │ 9 │                              └───────────────────────────────┘
   └───┴───┴───┘                                              |
                                                              v
   Flattened Filter Weights (C_out x K^2)     GEMM:  Y = W_flat @ X_col
```

1. **Unroll Patches (`im2col`)**: Every $K_h \times K_w \times C_{\text{in}}$ patch in the input tensor is flattened into a single column of matrix $X_{\text{col}} \in \mathbb{R}^{(C_{\text{in}} K_h K_w) \times (H_{\text{out}} W_{\text{out}})}$.
2. **Flatten Filters**: The filter bank is reshaped into $W_{\text{flat}} \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} K_h K_w)}$.
3. **Execute GEMM**:
   $$Y_{\text{flat}} = W_{\text{flat}} \times X_{\text{col}} \in \mathbb{R}^{C_{\text{out}} \times (H_{\text{out}} W_{\text{out}})}$$
4. **Reshape (`col2im`)**: Reshape $Y_{\text{flat}}$ into the 3D output tensor $C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$.

---

### 6. Deep Derivation 7.1.1: Exact Closed-Form Proof of the Convolution Output Dimension Formula

#### Context & Mathematical Formulation
Consider a 1D discrete sequence of length $L_{\text{in}}$ (generalizing trivially to 2D height and width).
Let:
- $p \in \mathbb{N}_0$ be the symmetric padding applied to both ends.
- $k \in \mathbb{N}_{\ge 1}$ be the kernel size.
- $d \in \mathbb{N}_{\ge 1}$ be the dilation rate.
- $s \in \mathbb{N}_{\ge 1}$ be the stride.

We desire to find the exact number of valid, discrete kernel placements $L_{\text{out}}$.

---

#### 1. Effective Kernel Footprint Under Dilation
In standard convolution ($d = 1$), a kernel of size $k$ occupies exactly $k$ consecutive cells.
Under dilation $d \ge 1$, adjacent kernel elements are separated by $d - 1$ blank spaces (holes):
```
d = 1:  [ * ] [ * ] [ * ]                 --> Footprint = 3
d = 2:  [ * ] [ . ] [ * ] [ . ] [ * ]     --> Footprint = 5
d = 3:  [ * ] [ . ] [ . ] [ * ] [ . ] [ . ] [ * ] --> Footprint = 7
```
The number of elements is $k$, so there are $k - 1$ intervals between elements, each of length $d$.
The total spatial span (effective kernel size $\tilde{k}$) is:
$$\tilde{k} = (k - 1) \cdot d + 1$$

---

#### 2. Padded Canvas Length and Feasible Anchor Positions
Adding padding $p$ to both ends expands the input sequence length from $L_{\text{in}}$ to:
$$L_{\text{padded}} = L_{\text{in}} + 2p$$
Index the padded cells from $0$ to $L_{\text{padded}} - 1$.
Let $i$ denote the starting (leftmost) index of the kernel window:
- The first possible placement is at the very beginning: $i_0 = 0$.
- For placement $i$, the rightmost element of the kernel occupies index:
  $$i_{\text{right}} = i + \tilde{k} - 1$$
- For the kernel to fit completely within the padded boundary, the rightmost index must satisfy:
  $$i_{\text{right}} \le L_{\text{padded}} - 1 \implies i + \tilde{k} - 1 \le L_{\text{in}} + 2p - 1$$
  Subtracting $\tilde{k} - 1$ from both sides gives the maximum valid starting index $i_{\max}$:
  $$i_{\max} = L_{\text{in}} + 2p - \tilde{k}$$

---

#### 3. Counting Valid Strided Placements
Because the filter advances in discrete increments of stride $s \ge 1$, the sequence of valid starting positions is an arithmetic progression:
$$i_n = n \cdot s, \quad \text{for } n = 0, 1, 2, \dots, N-1$$
For placement $n$ to be valid, we must have:
$$i_n \le i_{\max} \iff n \cdot s \le L_{\text{in}} + 2p - \tilde{k}$$
Dividing by $s$:
$$n \le \frac{L_{\text{in}} + 2p - \tilde{k}}{s}$$
Since $n$ must be an integer, the largest integer $n_{\max}$ satisfying this inequality is the floor function:
$$n_{\max} = \left\lfloor \frac{L_{\text{in}} + 2p - \tilde{k}}{s} \right\rfloor$$
Because the indices start at $n = 0$, the total number of valid placements $L_{\text{out}}$ is $n_{\max} + 1$:
$$\mathbf{L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} + 2p - \tilde{k}}{s} \right\rfloor + 1 = \left\lfloor \frac{L_{\text{in}} + 2p - d(k - 1) - 1}{s} \right\rfloor + 1}$$
$\blacksquare$ **Q.E.D.**

---

### 7. Deep Derivation 7.1.2: Translation Equivariance & Lie Group Shift Operators

#### Context & Formal Definition
A mapping $f: \mathcal{X} \to \mathcal{Y}$ is defined as **equivariant** with respect to a transformation group $\mathcal{G}$ if transforming the input and then applying $f$ produces the exact same result as applying $f$ and then transforming the output:
$$f(g \cdot x) = g \cdot f(x) \quad \forall g \in \mathcal{G}$$
In computer vision, the transformation group of interest is the **2D Spatial Translation Group** $(\mathbb{R}^2, +)$.

---

#### 1. The Translation Operator $T_\Delta$
For any continuous spatial shift vector $\Delta = (\Delta_x, \Delta_y) \in \mathbb{R}^2$, define the translation operator $T_\Delta$ acting on an image $I: \mathbb{R}^2 \to \mathbb{R}$ by:
$$(T_\Delta I)(u) = I(u - \Delta)$$
where $u = (x, y)$ denotes spatial coordinates.

---

#### 2. Theorem: Convolution is Translation Equivariant
Let $K$ be an arbitrary filter kernel.
The continuous convolution operation is $(I * K)(u) = \int_{\mathbb{R}^2} I(\tau) K(u - \tau) \, d\tau$.

##### Proof:
Apply the convolution operator to the translated image $\tilde{I} = T_\Delta I$:
$$\left( (T_\Delta I) * K \right)(u) = \int_{\mathbb{R}^2} (T_\Delta I)(\tau) K(u - \tau) \, d\tau$$
By definition of $T_\Delta$, $(T_\Delta I)(\tau) = I(\tau - \Delta)$:
$$\left( (T_\Delta I) * K \right)(u) = \int_{\mathbb{R}^2} I(\tau - \Delta) K(u - \tau) \, d\tau$$

Perform the change of variables: let $\nu = \tau - \Delta \implies \tau = \nu + \Delta$. The differential measure is invariant: $d\tau = d\nu$.
The integration domain remains all of $\mathbb{R}^2$:
$$\left( (T_\Delta I) * K \right)(u) = \int_{\mathbb{R}^2} I(\nu) K(u - (\nu + \Delta)) \, d\nu = \int_{\mathbb{R}^2} I(\nu) K((u - \Delta) - \nu) \, d\nu$$
Notice that by definition of convolution evaluated at the shifted coordinate $(u - \Delta)$:
$$\int_{\mathbb{R}^2} I(\nu) K((u - \Delta) - \nu) \, d\nu = (I * K)(u - \Delta)$$
Finally, applying the definition of the translation operator to the feature map $Y = I * K$:
$$(I * K)(u - \Delta) = T_\Delta(I * K)(u)$$
Therefore:
$$\mathbf{(T_\Delta I) * K = T_\Delta(I * K)}$$
$\blacksquare$ **Q.E.D.**

#### Equivariance vs. Invariance Distinction
- **Convolution is Translation Equivariant**: If an object shifts by 10 pixels to the right in the input, its feature representation in the convolutional output shifts by exactly 10 pixels to the right.
- **Classification Requires Translation Invariance**: A classifier should output "Cat" regardless of where the cat is located ($f(T_\Delta I) = f(I)$).
- **The CNN Architecture Bridge**: Stacking translation **equivariant** convolutional layers followed by a **Global Average Pooling (GAP)** layer computes the spatial integral:
  $$\text{GAP}(T_\Delta Y) = \frac{1}{|\Omega|} \int_{\Omega} Y(u - \Delta) \, du = \frac{1}{|\Omega|} \int_{\Omega} Y(\nu) \, d\nu = \text{GAP}(Y)$$
  This spatial integration converts translation equivariance into strict translation **invariance** at the classification head!

---

### 8. Deep Derivation 7.1.3: Continuous to Discrete Receptive Field & Gaussian Effective Receptive Field (ERF)

#### 1. Theoretical Receptive Field (TRF) vs. Effective Receptive Field (ERF)
The standard recursive equation:
$$r_l = r_{l-1} + (k_l - 1) j_{l-1}$$
computes the **Theoretical Receptive Field (TRF)**: the set of all input pixels that have a non-zero mathematical path to the output neuron.
However, in 2016, Luo, Peng, Huang, Lu, & Philips proved that pixels inside the TRF do **not** contribute equally to the output.
In reality, the impact of an input pixel follows a 2D Gaussian distribution, and the **Effective Receptive Field (ERF)** occupies only a fraction of the theoretical window.

---

#### 2. Derivation via Central Limit Theorem
Consider a deep linear convolutional network of $L$ layers with filter size $k$ and stride $1$.
The output pixel at layer $L$ is a linear combination of input pixels:
$$y = \sum_{p} w_{\text{eff}}(p) \cdot x(p)$$
where $w_{\text{eff}}$ is the effective filter obtained by iteratively convolving the layer filters:
$$w_{\text{eff}} = K_1 * K_2 * \dots * K_L$$

Assume each layer's filter weights are initialized independently from a distribution with mean 0 and finite variance $\sigma_K^2$.
By the **Central Limit Theorem for Iterated Convolutions**:
As depth $L \to \infty$, the composition of $L$ convolutions of bounded kernel functions converges asymptotically to a **Gaussian function**:
$$\mathbf{w_{\text{eff}}(x, y) \propto \exp\left( -\frac{x^2 + y^2}{2 \sigma_{\text{eff}}^2} \right)}$$
where the effective variance scales linearly with depth $L$:
$$\sigma_{\text{eff}}^2 = \sum_{l=1}^L \sigma_l^2 = L \cdot \frac{k^2 - 1}{12}$$

Taking the square root, the effective receptive field radius $\sigma_{\text{eff}}$ scales as:
$$\mathbf{\text{ERF} \propto \sigma_{\text{eff}} = \mathcal{O}(k \sqrt{L})}$$

#### Fundamental Takeaways:
1. **Gaussian Concentration**: Pixels near the center of the receptive field have thousands of alternative paths through the network DAG to reach the output neuron, while border pixels have only a single path. Thus, the center has exponentially higher influence!
2. **Sublinear Growth**: While the theoretical window grows linearly ($\text{TRF} \sim \mathcal{O}(L)$), the effective window grows only as the square root of depth ($\text{ERF} \sim \mathcal{O}(\sqrt{L})$).
3. **Architectural Motivation for Dilated Convolutions and Transformers**: In tasks requiring dense global context (semantic segmentation, object detection), standard deep convolutions suffer from a decaying ERF. Dilated convolutions and self-attention mechanisms are essential to expand the ERF to cover the entire image canvas.

---

## Part 3: Geometric & Signal Processing Interpretation

### 1. Spatial Frequency Filtering
In classical computer vision, 2D convolutions with hand-crafted kernels act as spatial frequency filters:
- **Sobel Horizontal Edge Detector**:
  $$K_{\text{Sobel}} = \begin{bmatrix} +1 & 0 & -1 \\ +2 & 0 & -2 \\ +1 & 0 & -1 \end{bmatrix}$$
  Computes the discrete directional spatial derivative $\frac{\partial I}{\partial x}$, firing strongly on vertical edges where color transitions abruptly.
- **Gaussian Blur / Low-Pass Filter**:
  $$K_{\text{Gauss}} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
  Attenuates high-frequency noise, smoothing the spatial image.
- **In deep learning, the network learns the optimal linear combination of edge, corner, texture, and semantic filters automatically via gradient descent!**

---

## Part 4: Real-World Analogy

### 1. The Crime Scene Investigator with a Magnifying Glass
Imagine an investigator inspecting a large room for clues:
- **The MLP Approach**: The investigator stares at the entire room all at once from a helicopter, trying to process every millimeter of the carpet simultaneously with a billion separate mental connections.
- **The CNN Approach**: The investigator holds a $3 \times 3$ inch magnifying glass (kernel) and systematically walks across the room step-by-step (stride).
  - Everywhere he points the glass, he checks for the exact same pattern: fingerprints or scratches (shared weights).
  - When he finds a fingerprint in the kitchen or the bedroom, his magnifying glass recognizes it equally well (translation equivariance).
  - He notes down a summary map of where fingerprints were found (feature map).
  - The next investigator reviews this summary map using a larger magnifying glass, connecting fingerprints to door handles (hierarchical receptive fields).

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us perform a complete, cell-by-cell manual convolution calculation on a concrete $4 \times 4$ single-channel image using a $3 \times 3$ Sobel-like edge detector kernel.

### Input Data & Filter Kernel

Input Image $X \in \mathbb{R}^{4 \times 4}$:
$$X = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 3 & 2 \\ 2 & 0 & 1 & 0 \\ 1 & 3 & 2 & 1 \end{bmatrix}$$

Filter Kernel $K \in \mathbb{R}^{3 \times 3}$ (Bias $b = 0$):
$$K = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$$

---

### Step-by-Step Manual Calculations

#### Scenario 1: Valid Padding ($p = 0$, Stride $s = 1$)
Output dimensions:
$$H_{\text{out}} = \frac{4 - 3 + 2(0)}{1} + 1 = 1 + 1 = 2, \quad W_{\text{out}} = 2$$
The output feature map $Y$ has shape $2 \times 2$.

1. **Output Position $(0, 0)$**: Top-left $3 \times 3$ patch:
   $$\text{Patch}(0, 0) = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \\ 2 & 0 & 1 \end{bmatrix}$$
   Element-wise Hadamard multiplication with $K$:
   $$\text{Row 0: } 1(1) + 2(0) + 0(-1) = 1 + 0 + 0 = 1$$
   $$\text{Row 1: } 0(2) + 1(0) + 3(-2) = 0 + 0 - 6 = -6$$
   $$\text{Row 2: } 2(1) + 0(0) + 1(-1) = 2 + 0 - 1 = 1$$
   Sum:
   $$Y(0, 0) = 1 + (-6) + 1 = \mathbf{-4}$$

2. **Output Position $(0, 1)$**: Shift patch right by stride $s = 1$:
   $$\text{Patch}(0, 1) = \begin{bmatrix} 2 & 0 & 1 \\ 1 & 3 & 2 \\ 0 & 1 & 0 \end{bmatrix}$$
   $$\text{Row 0: } 2(1) + 0(0) + 1(-1) = 2 - 1 = 1$$
   $$\text{Row 1: } 1(2) + 3(0) + 2(-2) = 2 - 4 = -2$$
   $$\text{Row 2: } 0(1) + 1(0) + 0(-1) = 0 - 0 = 0$$
   Sum:
   $$Y(0, 1) = 1 + (-2) + 0 = \mathbf{-1}$$

3. **Output Position $(1, 0)$**: Shift patch down by stride $s = 1$:
   $$\text{Patch}(1, 0) = \begin{bmatrix} 0 & 1 & 3 \\ 2 & 0 & 1 \\ 1 & 3 & 2 \end{bmatrix}$$
   $$\text{Row 0: } 0(1) + 1(0) + 3(-1) = 0 - 3 = -3$$
   $$\text{Row 1: } 2(2) + 0(0) + 1(-2) = 4 - 2 = +2$$
   $$\text{Row 2: } 1(1) + 3(0) + 2(-1) = 1 - 2 = -1$$
   Sum:
   $$Y(1, 0) = -3 + 2 + (-1) = \mathbf{-2}$$

4. **Output Position $(1, 1)$**: Bottom-right $3 \times 3$ patch:
   $$\text{Patch}(1, 1) = \begin{bmatrix} 1 & 3 & 2 \\ 0 & 1 & 0 \\ 3 & 2 & 1 \end{bmatrix}$$
   $$\text{Row 0: } 1(1) + 3(0) + 2(-1) = 1 - 2 = -1$$
   $$\text{Row 1: } 0(2) + 1(0) + 0(-2) = 0 - 0 = 0$$
   $$\text{Row 2: } 3(1) + 2(0) + 1(-1) = 3 - 1 = +2$$
   Sum:
   $$Y(1, 1) = -1 + 0 + 2 = \mathbf{+1}$$

**Final $2 \times 2$ Feature Map**:
$$Y = \begin{bmatrix} -4 & -1 \\ -2 & +1 \end{bmatrix}$$

---

#### Scenario 2: Strided Convolution ($p = 0$, Stride $s = 2$)
Output dimensions:
$$H_{\text{out}} = \left\lfloor \frac{4 - 3}{2} \right\rfloor + 1 = 0 + 1 = 1, \quad W_{\text{out}} = 1$$
Only position $(0, 0)$ can be computed before the window steps past the image boundary!
$$Y = \begin{bmatrix} -4 \end{bmatrix}$$

---

### Comparative Visual Grid: Convolution Walkthrough

```
+----------------------------------------------------------------------------------------------------+
|                                    INPUT IMAGE X (4 x 4)                                           |
|                                    [ 1, 2, 0, 1 ]                                                  |
|                                    [ 0, 1, 3, 2 ]                                                  |
|                                    [ 2, 0, 1, 0 ]                                                  |
|                                    [ 1, 3, 2, 1 ]                                                  |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Output Coordinate | Extracted Patch    | Filter Kernel K    | Sum of Products    | Feature Output  |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Y(0, 0)           | [[1, 2, 0],        | [[ 1, 0, -1],      | 1(1) + 3(-2)       | -4.0000         |
|                   |  [0, 1, 3],        |  [ 2, 0, -2],      | + 2(1) + 1(-1)     |                 |
|                   |  [2, 0, 1]]        |  [ 1, 0, -1]]      | = 1 - 6 + 2 - 1    |                 |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Y(0, 1)           | [[2, 0, 1],        | [[ 1, 0, -1],      | 2(1) + 1(-1)       | -1.0000         |
|                   |  [1, 3, 2],        |  [ 2, 0, -2],      | + 1(2) + 2(-2)     |                 |
|                   |  [0, 1, 0]]        |  [ 1, 0, -1]]      | = 2 - 1 + 2 - 4    |                 |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Y(1, 0)           | [[0, 1, 3],        | [[ 1, 0, -1],      | 3(-1) + 2(2)       | -2.0000         |
|                   |  [2, 0, 1],        |  [ 2, 0, -2],      | + 1(-2) + 1(1)     |                 |
|                   |  [1, 3, 2]]        |  [ 1, 0, -1]]      | = -3 + 4 - 2 + 1-2 |                 |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Y(1, 1)           | [[1, 3, 2],        | [[ 1, 0, -1],      | 1(1) + 2(-1)       | +1.0000         |
|                   |  [0, 1, 0],        |  [ 2, 0, -2],      | + 3(1) + 1(-1)     |                 |
|                   |  [3, 2, 1]]        |  [ 1, 0, -1]]      | = 1 - 2 + 3 - 1    |                 |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| FINAL OUTPUT MATRIX: Y = [[-4, -1], [-2, +1]] (100% Hand Computed Match!)                          |
+----------------------------------------------------------------------------------------------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning Implementation |
| :--- | :--- | :--- | :--- |
| $X$ | $\mathbb{R}^{C_{\text{in}} \times H \times W}$ | Spatial multi-channel input tensor | Input feature tensor (`x.shape = [B, C, H, W]`) |
| $W$ | $\mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$ | Learnable 4D filter bank | Weight tensor in `nn.Conv2d` |
| $s$ | Integer $\ge 1$ | Stride: Spatial step size | `stride` argument in Conv2d |
| $p$ | Integer $\ge 0$ | Padding: Border zero-padding width | `padding` argument in Conv2d |
| $d$ | Integer $\ge 1$ | Dilation rate: Kernel element spacing | `dilation` argument in Conv2d |
| $r_l$ | Integer $\ge 1$ | Receptive Field diameter in pixels | Spatial footprint in raw image |
| $j_l$ | Integer $\ge 1$ | Cumulative stride (jump) | Distance between feature centers |
| $X_{\text{col}}$ | 2D Matrix | Unrolled patches from `im2col` | Intermediate matrix passed to cuBLAS |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Receptive Field Calculation across a 4-Layer VGG Network
**Statement**: Consider a 4-layer CNN feature extractor:
- Layer 1: Conv2D ($k_1 = 3, s_1 = 1$)
- Layer 2: Conv2D ($k_2 = 3, s_2 = 1$)
- Layer 3: MaxPool2D ($k_3 = 2, s_3 = 2$)
- Layer 4: Conv2D ($k_4 = 3, s_4 = 1$)

Trace the cumulative stride $j_l$ and the receptive field $r_l$ at every layer from the raw image ($r_0 = 1, j_0 = 1$).

**Solution**:
We apply the recurrence formulas:
$$j_l = j_{l-1} \cdot s_l, \quad r_l = r_{l-1} + (k_l - 1) j_{l-1}$$

1. **Input (Layer 0)**:
   $$r_0 = 1, \quad j_0 = 1$$
2. **Layer 1 (Conv2D, $k_1 = 3, s_1 = 1$)**:
   $$r_1 = r_0 + (k_1 - 1) j_0 = 1 + (3 - 1)(1) = 1 + 2 = \mathbf{3}$$
   $$j_1 = j_0 \cdot s_1 = 1 \cdot 1 = \mathbf{1}$$
3. **Layer 2 (Conv2D, $k_2 = 3, s_2 = 1$)**:
   $$r_2 = r_1 + (k_2 - 1) j_1 = 3 + (3 - 1)(1) = 3 + 2 = \mathbf{5}$$
   $$j_2 = j_1 \cdot s_2 = 1 \cdot 1 = \mathbf{1}$$
4. **Layer 3 (MaxPool2D, $k_3 = 2, s_3 = 2$)**:
   $$r_3 = r_2 + (k_3 - 1) j_2 = 5 + (2 - 1)(1) = 5 + 1 = \mathbf{6}$$
   $$j_3 = j_2 \cdot s_3 = 1 \cdot 2 = \mathbf{2}$$
5. **Layer 4 (Conv2D, $k_4 = 3, s_4 = 1$)**:
   $$r_4 = r_3 + (k_4 - 1) j_3 = 6 + (3 - 1)(2) = 6 + 4 = \mathbf{10}$$
   $$j_4 = j_3 \cdot s_4 = 2 \cdot 1 = \mathbf{2}$$

*Summary Table*:
| Layer | Operation | Kernel $k$ | Stride $s$ | Cumulative Stride $j$ | Receptive Field $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 0 | Input Image | - | - | 1 | 1 |
| 1 | Conv2D | 3 | 1 | 1 | **3** |
| 2 | Conv2D | 3 | 1 | 1 | **5** |
| 3 | MaxPool2D | 2 | 2 | 2 | **6** |
| 4 | Conv2D | 3 | 1 | 2 | **10** |

A neuron in Layer 4 sees a **$10 \times 10$ patch** of the original image!

---

### Problem 2: Parameter Savings: MLP vs. ConvNet
**Statement**: Compare the parameter count required to connect an image of size $224 \times 224 \times 3$ to 64 output channels using:
1. A dense fully-connected layer (`nn.Linear`).
2. A convolutional layer with $3 \times 3$ filters (`nn.Conv2d(3, 64, kernel_size=3)`).

**Solution**:
1. **Fully Connected Layer**:
   Input dimension: $n_{\text{in}} = 224 \times 224 \times 3 = 150{,}528$.
   Output dimension: $n_{\text{out}} = 224 \times 224 \times 64 = 3{,}211{,}264$.
   Total parameters:
   $$P_{\text{FC}} = n_{\text{in}} \times n_{\text{out}} = 150{,}528 \times 3{,}211{,}264 \approx \mathbf{483.39\text{ Billion Parameters!}}$$
   *(Impossibly large for any single GPU!)*

2. **Convolutional Layer**:
   The $3 \times 3$ filter connects locally across all 3 input channels:
   $$P_{\text{Conv}} = C_{\text{out}} \times (C_{\text{in}} \times k_h \times k_w) + C_{\text{out}} = 64 \times (3 \times 3 \times 3) + 64$$
   $$P_{\text{Conv}} = 64 \times 27 + 64 = 1728 + 64 = \mathbf{1{,}792\text{ Parameters!}}$$

*Comparison*:
$$\frac{P_{\text{FC}}}{P_{\text{Conv}}} \approx \frac{4.83 \times 10^{11}}{1.79 \times 10^3} \approx \mathbf{270{,}000{,}000\times\text{ Reduction!}}$$
Convolution reduces parameters by over **270 million times**, turning an intractable problem into an instantaneous computation!

---

### Problem 3: Multi-Channel Convolution Manual Calculation with 2 Input and 2 Output Channels

**Statement**:
Consider a multi-channel input tensor $X \in \mathbb{R}^{C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$ with $C_{\text{in}} = 2$ and spatial dimensions $3 \times 3$:
$$X_0 = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 2 & 1 & 0 \end{bmatrix}, \quad X_1 = \begin{bmatrix} 0 & 1 & 0 \\ 2 & 0 & 1 \\ 1 & 2 & 1 \end{bmatrix}$$
We convolve $X$ with a 4D filter bank $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times 2 \times 2}$ with $C_{\text{out}} = 2$ output channels:
- **Output Channel 0 ($W_0$)**:
  $$W_{0, 0} = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}, \quad W_{0, 1} = \begin{bmatrix} 2 & 0 \\ -1 & 1 \end{bmatrix}, \quad b_0 = 1.0$$
- **Output Channel 1 ($W_1$)**:
  $$W_{1, 0} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad W_{1, 1} = \begin{bmatrix} 1 & 1 \\ 0 & -1 \end{bmatrix}, \quad b_1 = -0.5$$
Assume stride $s = 1$ and padding $p = 0$.
Compute the full 3D output feature tensor $Y \in \mathbb{R}^{2 \times 2 \times 2}$ step-by-step.

---

#### Solution

Output spatial dimensions:
$$H_{\text{out}} = \frac{3 - 2 + 2(0)}{1} + 1 = 2, \quad W_{\text{out}} = 2$$

##### 1. Output Channel $c_{\text{out}} = 0$ (Bias $b_0 = 1.0$)
- **Position $(0, 0)$**:
  $$X_0 \text{ patch: } \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \implies (1)(1) + (0)(-1) + (0)(0) + (1)(1) = 1 + 0 + 0 + 1 = \mathbf{2}$$
  $$X_1 \text{ patch: } \begin{bmatrix} 0 & 1 \\ 2 & 0 \end{bmatrix} \implies (0)(2) + (1)(0) + (2)(-1) + (0)(1) = 0 + 0 - 2 + 0 = \mathbf{-2}$$
  $$Y_0(0, 0) = 2 + (-2) + b_0 = 0 + 1.0 = \mathbf{1.0}$$

- **Position $(0, 1)$**:
  $$X_0 \text{ patch: } \begin{bmatrix} 0 & 2 \\ 1 & 1 \end{bmatrix} \implies (0)(1) + (2)(-1) + (1)(0) + (1)(1) = 0 - 2 + 0 + 1 = \mathbf{-1}$$
  $$X_1 \text{ patch: } \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \implies (1)(2) + (0)(0) + (0)(-1) + (1)(1) = 2 + 0 + 0 + 1 = \mathbf{3}$$
  $$Y_0(0, 1) = -1 + 3 + b_0 = 2 + 1.0 = \mathbf{3.0}$$

- **Position $(1, 0)$**:
  $$X_0 \text{ patch: } \begin{bmatrix} 0 & 1 \\ 2 & 1 \end{bmatrix} \implies (0)(1) + (1)(-1) + (2)(0) + (1)(1) = 0 - 1 + 0 + 1 = \mathbf{0}$$
  $$X_1 \text{ patch: } \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix} \implies (2)(2) + (0)(0) + (1)(-1) + (2)(1) = 4 + 0 - 1 + 2 = \mathbf{5}$$
  $$Y_0(1, 0) = 0 + 5 + b_0 = 5 + 1.0 = \mathbf{6.0}$$

- **Position $(1, 1)$**:
  $$X_0 \text{ patch: } \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} \implies (1)(1) + (1)(-1) + (1)(0) + (0)(1) = 1 - 1 + 0 + 0 = \mathbf{0}$$
  $$X_1 \text{ patch: } \begin{bmatrix} 0 & 1 \\ 2 & 1 \end{bmatrix} \implies (0)(2) + (1)(0) + (2)(-1) + (1)(1) = 0 + 0 - 2 + 1 = \mathbf{-1}$$
  $$Y_0(1, 1) = 0 + (-1) + b_0 = -1 + 1.0 = \mathbf{0.0}$$

$$Y_0 = \begin{bmatrix} 1.0 & 3.0 \\ 6.0 & 0.0 \end{bmatrix}$$

---

##### 2. Output Channel $c_{\text{out}} = 1$ (Bias $b_1 = -0.5$)
- **Position $(0, 0)$**:
  $$X_0 \star W_{1, 0} = (1)(0) + (0)(1) + (0)(1) + (1)(0) = \mathbf{0}$$
  $$X_1 \star W_{1, 1} = (0)(1) + (1)(1) + (2)(0) + (0)(-1) = \mathbf{1}$$
  $$Y_1(0, 0) = 0 + 1 + (-0.5) = \mathbf{0.5}$$

- **Position $(0, 1)$**:
  $$X_0 \star W_{1, 0} = (0)(0) + (2)(1) + (1)(1) + (1)(0) = 0 + 2 + 1 + 0 = \mathbf{3}$$
  $$X_1 \star W_{1, 1} = (1)(1) + (0)(1) + (0)(0) + (1)(-1) = 1 + 0 + 0 - 1 = \mathbf{0}$$
  $$Y_1(0, 1) = 3 + 0 + (-0.5) = \mathbf{2.5}$$

- **Position $(1, 0)$**:
  $$X_0 \star W_{1, 0} = (0)(0) + (1)(1) + (2)(1) + (1)(0) = 0 + 1 + 2 + 0 = \mathbf{3}$$
  $$X_1 \star W_{1, 1} = (2)(1) + (0)(1) + (1)(0) + (2)(-1) = 2 + 0 + 0 - 2 = \mathbf{0}$$
  $$Y_1(1, 0) = 3 + 0 + (-0.5) = \mathbf{2.5}$$

- **Position $(1, 1)$**:
  $$X_0 \star W_{1, 0} = (1)(0) + (1)(1) + (1)(1) + (0)(0) = 0 + 1 + 1 + 0 = \mathbf{2}$$
  $$X_1 \star W_{1, 1} = (0)(1) + (1)(1) + (2)(0) + (1)(-1) = 0 + 1 + 0 - 1 = \mathbf{0}$$
  $$Y_1(1, 1) = 2 + 0 + (-0.5) = \mathbf{1.5}$$

$$Y_1 = \begin{bmatrix} 0.5 & 2.5 \\ 2.5 & 1.5 \end{bmatrix}$$

**Final 3D Output Tensor $Y \in \mathbb{R}^{2 \times 2 \times 2}$**:
$$Y = \left[ \begin{bmatrix} 1.0 & 3.0 \\ 6.0 & 0.0 \end{bmatrix}, \; \begin{bmatrix} 0.5 & 2.5 \\ 2.5 & 1.5 \end{bmatrix} \right]$$

---

### Problem 4: Dilated (Atrous) Convolution & Exponential Receptive Field Expansion

**Statement**:
Consider a 5-layer 1D convolutional neural network with constant filter size $k = 3$, stride $s = 1$, and exponentially increasing dilation rates:
$$d_1 = 1, \quad d_2 = 2, \quad d_3 = 4, \quad d_4 = 8, \quad d_5 = 16$$
(as implemented in DeepMind's WaveNet audio architecture).
1. Calculate the effective kernel footprint $\tilde{k}_l = d_l(k - 1) + 1$ for each layer.
2. Trace the exact receptive field $r_l$ at every layer starting from $r_0 = 1, j_0 = 1$.
3. Compare the receptive field at Layer 5 against a standard non-dilated CNN ($d \equiv 1$).

---

#### Solution

##### 1. Recurrence Equations
Cumulative stride with $s_l = 1$:
$$j_l = j_{l-1} \cdot s_l = 1 \cdot 1 = 1 \quad \forall l$$

Receptive field recurrence:
$$r_l = r_{l-1} + (\tilde{k}_l - 1) \cdot j_{l-1} = r_{l-1} + 2 d_l$$

##### 2. Layer-by-Layer Calculation
- **Layer 0 (Input)**: $r_0 = 1, \quad j_0 = 1$
- **Layer 1 ($d_1 = 1$)**:
  $$\tilde{k}_1 = 1(3 - 1) + 1 = \mathbf{3}$$
  $$r_1 = 1 + (3 - 1)(1) = 1 + 2 = \mathbf{3}$$
- **Layer 2 ($d_2 = 2$)**:
  $$\tilde{k}_2 = 2(3 - 1) + 1 = \mathbf{5}$$
  $$r_2 = 3 + (5 - 1)(1) = 3 + 4 = \mathbf{7}$$
- **Layer 3 ($d_3 = 4$)**:
  $$\tilde{k}_3 = 4(3 - 1) + 1 = \mathbf{9}$$
  $$r_3 = 7 + (9 - 1)(1) = 7 + 8 = \mathbf{15}$$
- **Layer 4 ($d_4 = 8$)**:
  $$\tilde{k}_4 = 8(3 - 1) + 1 = \mathbf{17}$$
  $$r_4 = 15 + (17 - 1)(1) = 15 + 16 = \mathbf{31}$$
- **Layer 5 ($d_5 = 16$)**:
  $$\tilde{k}_5 = 16(3 - 1) + 1 = \mathbf{33}$$
  $$r_5 = 31 + (33 - 1)(1) = 31 + 32 = \mathbf{63}$$

##### 3. Comparison with Non-Dilated Architecture
For standard convolution ($d_l = 1$ for all layers):
$$r_l = r_{l-1} + 2 \implies r_5 = 1 + 5 \times 2 = \mathbf{11}$$

```
+-------+--------------+------------------+-----------------------+---------------------+
| Layer | Dilation d_l | Effective Span k | Dilated RF r_l        | Standard RF (d = 1) |
+-------+--------------+------------------+-----------------------+---------------------+
| 1     | 1            | 3                | 3                     | 3                   |
| 2     | 2            | 5                | 7                     | 5                   |
| 3     | 4            | 9                | 15                    | 7                   |
| 4     | 8            | 17               | 31                    | 9                   |
| 5     | 16           | 33               | 63 (EXPONENTIAL: 2^6-1)| 11 (LINEAR: 1 + 2L) |
+-------+--------------+------------------+-----------------------+---------------------+
```
*Conclusion*: Dilated convolution achieves a receptive field of **63 samples** compared to only **11 samples** for standard convolution—a **$5.7\times$ expansion** with zero additional parameters and zero pooling loss of temporal resolution!

---

### Problem 5: Manual `im2col` Matrix Unrolling & GEMM Execution

**Statement**:
Given a single-channel $3 \times 3$ image $X$:
$$X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$
and two $2 \times 2$ filters $W_1, W_2$:
$$W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$
with stride $s = 1$, padding $p = 0$, and bias $b = [0, 0]^T$.
1. Construct the unrolled patch matrix $X_{\text{col}} \in \mathbb{R}^{4 \times 4}$.
2. Construct the flattened filter matrix $W_{\text{flat}} \in \mathbb{R}^{2 \times 4}$.
3. Compute the GEMM product $Y_{\text{flat}} = W_{\text{flat}} X_{\text{col}}$.
4. Reshape $Y_{\text{flat}}$ into the output tensor $Y \in \mathbb{R}^{2 \times 2 \times 2}$ and verify against spatial convolution.

---

#### Solution

##### 1. `im2col` Patch Extraction
Output spatial size: $H_{\text{out}} = 3 - 2 + 1 = 2, W_{\text{out}} = 2 \implies 4$ patches.
- **Patch $(0, 0)$**: $\begin{bmatrix} 1 & 2 \\ 4 & 5 \end{bmatrix} \longrightarrow \text{Column 0: } [1, 2, 4, 5]^T$
- **Patch $(0, 1)$**: $\begin{bmatrix} 2 & 3 \\ 5 & 6 \end{bmatrix} \longrightarrow \text{Column 1: } [2, 3, 5, 6]^T$
- **Patch $(1, 0)$**: $\begin{bmatrix} 4 & 5 \\ 7 & 8 \end{bmatrix} \longrightarrow \text{Column 2: } [4, 5, 7, 8]^T$
- **Patch $(1, 1)$**: $\begin{bmatrix} 5 & 6 \\ 8 & 9 \end{bmatrix} \longrightarrow \text{Column 3: } [5, 6, 8, 9]^T$

$$\mathbf{X_{\text{col}} = \begin{bmatrix} 1 & 2 & 4 & 5 \\ 2 & 3 & 5 & 6 \\ 4 & 5 & 7 & 8 \\ 5 & 6 & 8 & 9 \end{bmatrix} \in \mathbb{R}^{4 \times 4}}$$

---

##### 2. Filter Bank Flattening
- **Filter 1**: $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \longrightarrow \text{Row 0: } [1, 0, 0, 1]$
- **Filter 2**: $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \longrightarrow \text{Row 1: } [0, 1, 1, 0]$

$$\mathbf{W_{\text{flat}} = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}}$$

---

##### 3. General Matrix Multiplication (GEMM)
$$Y_{\text{flat}} = W_{\text{flat}} \times X_{\text{col}} = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 2 & 4 & 5 \\ 2 & 3 & 5 & 6 \\ 4 & 5 & 7 & 8 \\ 5 & 6 & 8 & 9 \end{bmatrix}$$

- **Row 0 (Filter 1)**:
  - Col 0: $(1)(1) + (0)(2) + (0)(4) + (1)(5) = 1 + 5 = \mathbf{6}$
  - Col 1: $(1)(2) + (0)(3) + (0)(5) + (1)(6) = 2 + 6 = \mathbf{8}$
  - Col 2: $(1)(4) + (0)(5) + (0)(7) + (1)(8) = 4 + 8 = \mathbf{12}$
  - Col 3: $(1)(5) + (0)(6) + (0)(8) + (1)(9) = 5 + 9 = \mathbf{14}$
- **Row 1 (Filter 2)**:
  - Col 0: $(0)(1) + (1)(2) + (1)(4) + (0)(5) = 2 + 4 = \mathbf{6}$
  - Col 1: $(0)(2) + (1)(3) + (1)(5) + (0)(6) = 3 + 5 = \mathbf{8}$
  - Col 2: $(0)(4) + (1)(5) + (1)(7) + (0)(8) = 5 + 7 = \mathbf{12}$
  - Col 3: $(0)(5) + (1)(6) + (1)(8) + (0)(9) = 6 + 8 = \mathbf{14}$

$$\mathbf{Y_{\text{flat}} = \begin{bmatrix} 6 & 8 & 12 & 14 \\ 6 & 8 & 12 & 14 \end{bmatrix} \in \mathbb{R}^{2 \times 4}}$$

---

##### 4. `col2im` Reshaping & Verification
Reshaping each row of length 4 into a $2 \times 2$ spatial map:
$$Y_0 = \begin{bmatrix} 6 & 8 \\ 12 & 14 \end{bmatrix}, \quad Y_1 = \begin{bmatrix} 6 & 8 \\ 12 & 14 \end{bmatrix}$$

**Verification via Direct Spatial Convolution**:
For Filter 1 ($W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$):
- Position $(0, 0)$: $1(1) + 2(0) + 4(0) + 5(1) = 6$
- Position $(0, 1)$: $2(1) + 3(0) + 5(0) + 6(1) = 8$
- Position $(1, 0)$: $4(1) + 5(0) + 7(0) + 8(1) = 12$
- Position $(1, 1)$: $5(1) + 6(0) + 8(0) + 9(1) = 14$
Matches the GEMM result identically!

---

## Part 7: Deep Learning Connection & Application

### 1. Depthwise Separable Convolutions (MobileNet)
Standard Conv2D computes spatial filtering and channel cross-talk simultaneously ($K^2 \cdot C_{\text{in}} \cdot C_{\text{out}}$ FLOPS).
Howard et al. (2017) factored this into two distinct steps:
1. **Depthwise Convolution**: Applies a single $K \times K$ filter per input channel ($K^2 \cdot C_{\text{in}}$).
2. **Pointwise Convolution ($1 \times 1$)**: Linearly combines channel outputs ($1 \times 1 \times C_{\text{in}} \cdot C_{\text{out}}$).
- Total compute reduction:
  $$\frac{K^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}}{K^2 C_{\text{in}} C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{K^2} \approx \frac{1}{9} \approx \mathbf{88\%\text{ to } 90\%\text{ compute savings!}}$$
  Enabled deep learning to run smoothly on edge smartphones (MobileNet).

### 2. Dilated Convolutions in WaveNet & Semantic Segmentation
To capture long-range audio context (16,000 samples per second) without pooling (which degrades resolution), DeepMind's **WaveNet** stacks dilated convolutions with exponentially increasing dilation rates:
$$d \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512\}$$
The receptive field expands **exponentially** ($r \sim 2^L$) while keeping compute and parameter count strictly linear!

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `07_convolutional_networks/code/01_convolution_operation_and_receptive_field.py`. It implements:
1. Exact visual grid verification of $4 \times 4$ image with $3 \times 3$ Sobel filter matching Part 5 hand calculations.
2. A pure NumPy vectorized `conv2d_im2col` engine implementing the `im2col` GEMM algorithm.
3. Strict parity check against PyTorch's native `torch.nn.functional.conv2d` across multi-channel inputs with padding, stride, and dilation to machine precision ($< 10^{-7}$).
4. Recursive analytical Receptive Field calculator verifying VGG network theoretical numbers.

Save the code and run it directly in Python 3.
