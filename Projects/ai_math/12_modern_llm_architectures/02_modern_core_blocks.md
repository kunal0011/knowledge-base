# Chapter 02: Modern Core Blocks: Pre-RMSNorm, SwiGLU & Residual Scaling

---

## 1. Intuition & 101 Motivation

When the original Transformer was published in 2017 (*"Attention Is All You Need"*), it relied on three foundational components:
1. **Post-LayerNorm:** Normalization applied *after* the residual addition: $x_{l+1} = \operatorname{LN}(x_l + f(x_l))$.
2. **Full LayerNorm:** Computing both the batch mean $\mu$ and variance $\sigma^2$, with both learnable gain $\gamma$ and bias $\beta$.
3. **Standard Feed-Forward Network (FFN):** A two-layer MLP with ReLU or GELU activation: $\operatorname{FFN}(x) = \operatorname{GELU}(x W_1 + b_1) W_2 + b_2$.

Over the past decade of training multi-billion parameter foundation models, researchers discovered severe limitations in this original blueprint:
- **Post-LN Instability:** Post-LN causes gradient vanishing near the input layers. Deep networks (50+ layers) suffered immediate training divergence unless babysat with fragile learning rate warmup schedules.
- **LayerNorm Redundancy:** Subtracting the mean vector $\mu$ accounts for only negligible shift invariance but incurs significant GPU memory bandwidth overhead.
- **GELU Bottleneck:** Standard MLPs lack a multiplicative gating mechanism, limiting expressivity per parameter.

Today, foundation models across the entire AI ecosystem—including **LLaMA 1/2/3, Mistral, Gemma, Qwen, and DeepSeek**—have reached an extraordinary consensus. They converge on the **Modern Transformer Block**:
- **Pre-Normalization:** Placing normalization *before* each sub-layer: $x_{l+1} = x_l + f(\operatorname{Norm}(x_l))$.
- **Root Mean Square Normalization (RMSNorm):** Dropping mean subtraction to boost throughput by $15–20\%$.
- **SwiGLU Gated Activation:** Introducing a bilinear Swish-gated branch that dramatically enhances gradient flow and representation capacity.

---

## 2. Rigorous Mathematical Formulation

### 2.1 LayerNorm vs. RMSNorm

#### Standard LayerNorm (Ba, Kiros, & Hinton, 2016):
Given an input vector $x \in \mathbb{R}^d$:
$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$
$$\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
where $\gamma, \beta \in \mathbb{R}^d$ are learnable affine gain and bias parameters.

#### Root Mean Square Normalization (RMSNorm, Zhang & Sennrich, 2019):
Zhang & Sennrich demonstrated that the shift-invariance property ($\mu$) in LayerNorm contributes virtually nothing to training stability. The primary benefit of normalization is **scaling invariance**: keeping activation magnitudes within a stable bounded sphere.

RMSNorm defines the root-mean-square statistic:
$$\operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

The normalized output is computed by scaling directly, dispensing with mean subtraction and bias:
$$\operatorname{RMSNorm}(x) = \frac{x}{\operatorname{RMS}(x)} \odot \gamma$$

where $\gamma \in \mathbb{R}^d$ is a learnable scale vector initialized to $\mathbf{1}$.

#### Key Mathematical Properties of RMSNorm:
1. **Scale Invariance:** For any scalar $\alpha > 0$:
   $$\operatorname{RMSNorm}(\alpha x) = \frac{\alpha x}{\sqrt{\frac{1}{d} \sum (\alpha x_i)^2 + \epsilon}} \odot \gamma \approx \frac{\alpha x}{\alpha \operatorname{RMS}(x)} \odot \gamma = \operatorname{RMSNorm}(x)$$
2. **Computational Savings:** Eliminates the two-pass algorithm required by mean-centering, cutting GPU memory reads and writes and saving $15–20\%$ of normalization runtime.

---

### 2.2 The SwiGLU Gated Activation Function

In a classical Transformer MLP, the feed-forward network uses two weight matrices:
$$\operatorname{FFN}_{\text{GELU}}(x) = \operatorname{GELU}(x W_1) W_2$$
where $W_1 \in \mathbb{R}^{d \times d_{\text{ffn}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ffn}} \times d}$ (traditionally $d_{\text{ffn}} = 4d$).

#### Gated Linear Units (GLU, Dauphin et al., 2017):
A GLU splits the transformation into two parallel projections and takes their element-wise product:
$$\operatorname{GLU}(x, W_1, W_3) = (x W_1) \odot \sigma(x W_3)$$
The branch $\sigma(x W_3)$ acts as an adaptive multiplicative **gate**, dynamically attenuating or magnifying features.

#### SwiGLU (Noam Shazeer, 2020):
Shazeer evaluated multiple activation variants and showed that **SwiGLU**—using the Swish (SiLU) activation function—outperformed all other architectures across language modeling benchmarks:

$$\operatorname{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

$$\operatorname{SwiGLU}(x) = \Big( \operatorname{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \Big) W_{\text{down}}$$

where:
- $W_{\text{gate}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$ (projects to the gating branch)
- $W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$ (projects to the linear feature branch)
- $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$ (projects the modulated features back to model dimension $d$)
- Note: Modern LLMs omit bias vectors entirely ($b = 0$).

#### Parameter Matching: The $\frac{8}{3} d$ Sizing Rule:
A standard GELU MLP has two matrices totaling $2 \times d \times (4d) = 8 d^2$ parameters.
Because SwiGLU introduces a third matrix ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$), setting $d_{\text{ffn}} = 4d$ would increase parameters by $50\%$ ($3 \times 4 d^2 = 12 d^2$).

To maintain identical FLOPs and parameter counts:
$$3 \times d \times d_{\text{ffn}} = 8 d^2 \implies d_{\text{ffn}} \approx \frac{8}{3} d \approx 2.67 d$$

In LLaMA 1/2/3, Meta rounds this to the nearest multiple of 256 for maximum GPU Tensor Core utilization:
$$d_{\text{ffn}} = 256 \times \left\lfloor \frac{2 \times 4 d}{3 \times 256} + 0.5 \right\rfloor$$
*(e.g., for LLaMA-7B with $d = 4096$, $d_{\text{ffn}} = 11,008$).*

---

### 2.3 Pre-LN vs. Post-LN Signal Propagation

```
Post-LN (Original 2017 Transformer)          Pre-LN (Modern LLM Consensus)
          x                                            x
          |                                            +------------+
      [ SubLayer ]                                     |            |
          |                                            |        [ RMSNorm ]
          v                                            |            |
       (  +  ) <--- Identity Residual                  |       [ SubLayer ]
          |                                            |            |
      [ LayerNorm ]                                    v            v
          |                                         (  +  ) <-------+
          v                                            |
        x_{l+1}                                     x_{l+1}
```

#### Theorem 12.2 (The Pre-LN Identity Highway):
In a Pre-LN Transformer, unrolling across $L$ layers yields:
$$x_L = x_0 + \sum_{l=0}^{L-1} f_l(\operatorname{RMSNorm}(x_l))$$

The gradient of the final output $x_L$ with respect to the input embedding $x_0$ is:
$$\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \sum_{l=0}^{L-1} \frac{\partial f_l}{\partial x_0}$$

Because the identity matrix $\mathbf{I}$ passes through unattenuated, **gradients flow directly from the loss back to the input embeddings without vanishing**, completely eliminating the need for delicate learning rate warmup gymnastics!

---

## 3. Geometric & Physical Interpretation

### 3.1 Hypersphere Projection
RMSNorm projects any arbitrary vector $x \in \mathbb{R}^d$ onto the surface of a centered hypersphere of radius $\sqrt{d}$:
$$\|\hat{x}\|_2 = \sqrt{\sum_{i=1}^d \left( \frac{x_i}{\sqrt{\frac{1}{d} \sum x_j^2}} \right)^2} = \sqrt{\frac{\sum x_i^2}{\frac{1}{d} \sum x_j^2}} = \sqrt{d}$$

```
Activation Space (R^d)
      \        /
       \  x   /   <-- Large magnitude activation vector
        \ |  /
         \| /
      +-------+
     /    *    \  <-- Projected vector x_hat on hypersphere (radius = sqrt(d))
    |     |     |
     \    *    /
      +-------+
```
This guarantees that regardless of how large or small intermediate activations become, the inputs entering subsequent linear attention layers always have bounded variance, preventing numerical overflow and activation explosion.

---

## 4. Real-World Analogy: The Express Train

Consider high-speed commuter rail:
- **Post-LN:** Every passenger train must pass through a crowded downtown security terminal (normalization) at every single station before proceeding. If there are 80 stations, traffic grinds to a halt.
- **Pre-LN:** There is an open **high-speed express track** that runs non-stop from the suburbs directly to the city center ($x_L = x_0 + \sum f$). Passengers who wish to visit local intermediate stations exit via side-turnoffs that have their own entry turnstiles (RMSNorm). The main line remains perpetually unobstructed!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace the complete numerical calculations of:
1. **RMSNorm** on a concrete 4-dimensional vector.
2. **SwiGLU** forward pass with exact gating arithmetic.

---

### 5.1 System & Parameter Setup

#### Component 1: RMSNorm
- Input vector: $x = [2.0, -1.0, 3.0, -2.0]^T \in \mathbb{R}^4$ ($d = 4$)
- Learnable scale gain: $\gamma = [1.0, 0.5, 2.0, 1.0]^T$
- Epsilon: $\epsilon = 0.0$

#### Component 2: SwiGLU Gated MLP
- Input vector: $z = [1.0, -2.0]^T \in \mathbb{R}^2$ ($d = 2, d_{\text{ffn}} = 2$)
- Gate Projection Matrix:
  $$W_{\text{gate}} = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}$$
- Up Projection Matrix:
  $$W_{\text{up}} = \begin{bmatrix} 0.5 & 1.0 \\ -1.0 & 2.0 \end{bmatrix}$$
- Down Projection Matrix:
  $$W_{\text{down}} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} \in \mathbb{R}^{2 \times 1}$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $x$ | `input_x` | Raw activation vector before normalization $[2, -1, 3, -2]$ |
| $\operatorname{RMS}(x)$ | `rms_val` | Root mean square scalar: $\sqrt{\frac{1}{d} \sum x_i^2}$ |
| $\hat{x}$ | `x_norm` | Unit-variance normalized vector: $x / \operatorname{RMS}(x)$ |
| $\gamma$ | `scale_gamma` | Learnable affine gain scaling factor |
| $y$ | `rmsnorm_out` | Final normalized output: $\hat{x} \odot \gamma$ |
| $g$ | `gate_proj` | Gating branch projection: $z W_{\text{gate}}$ |
| $u$ | `up_proj` | Up branch projection: $z W_{\text{up}}$ |
| $\operatorname{Swish}(g)$ | `swish_gate` | Swish-activated gate: $g \odot \sigma(g)$ |
| $h$ | `modulated_h` | Hadamard product: $\operatorname{Swish}(g) \odot u$ |
| $\text{Out}$ | `swiglu_out` | Final SwiGLU output: $h W_{\text{down}}$ |

---

### 5.3 Step-by-Step Hand Calculations: RMSNorm

#### Step 1: Calculate Sum of Squares & Mean Square
$$x_1^2 = (2.0)^2 = 4.0$$
$$x_2^2 = (-1.0)^2 = 1.0$$
$$x_3^2 = (3.0)^2 = 9.0$$
$$x_4^2 = (-2.0)^2 = 4.0$$
$$\sum_{i=1}^4 x_i^2 = 4.0 + 1.0 + 9.0 + 4.0 = \mathbf{18.0}$$
$$\text{Mean Square} = \frac{18.0}{4} = \mathbf{4.500000}$$

#### Step 2: Compute RMS Scalar
$$\operatorname{RMS}(x) = \sqrt{4.500000} \approx \mathbf{2.121320}$$

#### Step 3: Compute Scaled Normalized Vector $\hat{x}$
$$\hat{x}_1 = \frac{2.0}{2.121320} \approx \mathbf{0.942809}$$
$$\hat{x}_2 = \frac{-1.0}{2.121320} \approx \mathbf{-0.471405}$$
$$\hat{x}_3 = \frac{3.0}{2.121320} \approx \mathbf{1.414214}$$
$$\hat{x}_4 = \frac{-2.0}{2.121320} \approx \mathbf{-0.942809}$$

#### Step 4: Multiply by Learnable Scale Vector $\gamma$
$$y_1 = \hat{x}_1 \times \gamma_1 = 0.942809 \times 1.0 = \mathbf{0.942809}$$
$$y_2 = \hat{x}_2 \times \gamma_2 = -0.471405 \times 0.5 = \mathbf{-0.235702}$$
$$y_3 = \hat{x}_3 \times \gamma_3 = 1.414214 \times 2.0 = \mathbf{2.828427}$$
$$y_4 = \hat{x}_4 \times \gamma_4 = -0.942809 \times 1.0 = \mathbf{-0.942809}$$

---

### 5.4 Step-by-Step Hand Calculations: SwiGLU Gated MLP

Input $z = [1.0, -2.0]$.

#### Step 1: Gate Projection $g = z W_{\text{gate}}$
$$g_1 = 1.0 \times 2.0 + (-2.0) \times 0.0 = \mathbf{2.000000}$$
$$g_2 = 1.0 \times 0.0 + (-2.0) \times 0.5 = \mathbf{-1.000000}$$
$$g = \begin{bmatrix} 2.000000 & -1.000000 \end{bmatrix}$$

#### Step 2: Up Projection $u = z W_{\text{up}}$
$$u_1 = 1.0 \times 0.5 + (-2.0) \times (-1.0) = 0.5 + 2.0 = \mathbf{2.500000}$$
$$u_2 = 1.0 \times 1.0 + (-2.0) \times 2.0 = 1.0 - 4.0 = \mathbf{-3.000000}$$
$$u = \begin{bmatrix} 2.500000 & -3.000000 \end{bmatrix}$$

#### Step 3: Swish (SiLU) Activation on Gate $g$
Recall $\operatorname{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$:
1. **For $g_1 = 2.0$:**
   $$\sigma(2.0) = \frac{1}{1 + e^{-2.0}} \approx \frac{1}{1 + 0.135335} = \frac{1}{1.135335} \approx 0.880797$$
   $$\operatorname{Swish}(g_1) = 2.0 \times 0.880797 \approx \mathbf{1.761594}$$

2. **For $g_2 = -1.0$:**
   $$\sigma(-1.0) = \frac{1}{1 + e^{1.0}} \approx \frac{1}{1 + 2.718282} = \frac{1}{3.718282} \approx 0.268941$$
   $$\operatorname{Swish}(g_2) = -1.0 \times 0.268941 \approx \mathbf{-0.268941}$$

#### Step 4: Element-Wise Modulation $h = \operatorname{Swish}(g) \odot u$
$$h_1 = \operatorname{Swish}(g_1) \times u_1 = 1.761594 \times 2.500000 = \mathbf{4.403985}$$
$$h_2 = \operatorname{Swish}(g_2) \times u_2 = -0.268941 \times (-3.000000) = \mathbf{+0.806824}$$
$$h = \begin{bmatrix} 4.403985 & 0.806824 \end{bmatrix}$$

#### Step 5: Down Projection $\text{Out} = h W_{\text{down}}$
$$\text{Out} = 4.403985 \times 1.0 + 0.806824 \times 2.0 = 4.403985 + 1.613648 = \mathbf{6.017633}$$

---

### 5.5 Summary Visual Grid: Modern Block Computations Ledger

| Component | Operation | Formula | Intermediate Vector | Final Scalar / Vector |
| :---: | :---: | :---: | :---: | :---: |
| **RMSNorm** | Sum of Squares | $\sum x_i^2$ | $[4, 1, 9, 4]$ | $18.0000$ |
| **RMSNorm** | RMS Divisor | $\sqrt{\text{mean}(x^2)}$ | $\sqrt{18 / 4} = \sqrt{4.5}$ | $\mathbf{2.121320}$ |
| **RMSNorm** | Normalized $\hat{x}$ | $x / \text{RMS}$ | $[2, -1, 3, -2] / 2.1213$ | $[0.9428, -0.4714, 1.4142, -0.9428]$ |
| **RMSNorm** | Scaled Output $y$ | $\hat{x} \odot \gamma$ | $[1.0, 0.5, 2.0, 1.0]$ | $\mathbf{[0.9428, -0.2357, 2.8284, -0.9428]}$ |
| **SwiGLU** | Gate Projection | $z W_{\text{gate}}$ | $[1, -2] \cdot W_g$ | $[2.0000, -1.0000]$ |
| **SwiGLU** | Up Projection | $z W_{\text{up}}$ | $[1, -2] \cdot W_u$ | $[2.5000, -3.0000]$ |
| **SwiGLU** | Swish Gate | $g \odot \sigma(g)$ | $[2.0(0.8808), -1.0(0.2689)]$ | $[1.7616, -0.2689]$ |
| **SwiGLU** | Gated Product $h$ | $\text{Swish}(g) \odot u$ | $[1.7616(2.5), -0.2689(-3.0)]$ | $[4.4040, 0.8068]$ |
| **SwiGLU** | Down Output | $h W_{\text{down}}$ | $[4.4040, 0.8068] \cdot [1, 2]^T$ | $\mathbf{6.017633}$ |

---

## 6. Solved Illustrations

### Illustration 1: Invariance to Scale in RMSNorm
**Problem:**
Verify algebraically and numerically that if $x$ is multiplied by an arbitrary scalar $\alpha = 10.0$, the normalized output $\hat{x}_{\text{new}}$ is identical to $\hat{x}_{\text{orig}}$.
**Solution:**
$$x_{\text{new}} = 10 \cdot [2.0, -1.0, 3.0, -2.0] = [20.0, -10.0, 30.0, -20.0]$$
$$\operatorname{RMS}(x_{\text{new}}) = \sqrt{\frac{400 + 100 + 900 + 400}{4}} = \sqrt{\frac{1800}{4}} = \sqrt{450} = 10 \sqrt{4.5} \approx 21.213203$$
$$\hat{x}_{\text{new}} = \frac{[20.0, -10.0, 30.0, -20.0]}{21.213203} = [0.942809, -0.471405, 1.414214, -0.942809] \equiv \mathbf{\hat{x}_{\text{orig}}} \quad \blacksquare$$

---

## 7. Deep Learning Connection & Modern Applications

### Modern Architectural Matrix (Sebastian Raschka Taxonomy)
The following table summarizes the foundational building blocks chosen across premier open-weights models:

| Model Family | Normalization | Activation | Attention Mechanism | Positional Encoding | Bias Terms |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LLaMA 1 / 2** | Pre-RMSNorm | SwiGLU | MHA / GQA | RoPE | No |
| **LLaMA 3 / 3.1** | Pre-RMSNorm | SwiGLU | GQA (8 KV heads) | RoPE (scaled) | No |
| **Mistral / Mixtral** | Pre-RMSNorm | SwiGLU | GQA / Sparse MoE | RoPE | No |
| **Gemma 1 / 2** | Pre-RMSNorm (w/ offset) | GeGLU | MHA / GQA | RoPE | No |
| **DeepSeek-V2 / V3**| Pre-RMSNorm | SwiGLU | MLA + DeepSeekMoE | RoPE | No |
| **Qwen 2.5** | Pre-RMSNorm | SwiGLU | GQA | RoPE | Yes (QKV only) |

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - RMSNorm forward pass matching $y = [0.942809, -0.235702, 2.828427, -0.942809]$ to $< 10^{-6}$.
   - SwiGLU forward pass matching $\text{Out} = 6.017633$ to $< 10^{-6}$.
2. **Production-Ready PyTorch Pre-RMSNorm Transformer Layer:**
   - Vectorized `RMSNorm` module.
   - Vectorized `SwiGLU` MLP module with $\frac{8}{3}d$ dimension calculation.
   - Complete Pre-LN residual block verified with backward gradient propagation.

See implementation in:
[`12_modern_llm_architectures/code/02_modern_core_blocks.py`](./code/02_modern_core_blocks.py)
