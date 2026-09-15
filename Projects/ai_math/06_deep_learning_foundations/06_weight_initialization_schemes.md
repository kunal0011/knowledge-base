# 6.6 Weight Initialization Schemes (Xavier/Glorot, He/Kaiming, Orthogonal)

---

## Part 1: Intuition & 101 Motivation

Before a neural network sees its very first training sample, its parameters must be assigned starting numerical values.

In the early days of deep learning, weight initialization was treated as an arbitrary afterthought—practitioners typically sampled weights from a standard normal distribution $\mathcal{N}(0, 1)$ or a small Gaussian $\mathcal{N}(0, 0.01^2)$. This naive choice caused deep networks with more than a few layers to fail completely.

To understand why, observe what happens when an input signal passes through an $L$-layer network:
$$a^{(L)} = W^{(L)} \sigma\left( W^{(L-1)} \dots \sigma\left( W^{(1)} x \right) \dots \right)$$
At its core, this is a sequence of $L$ consecutive matrix multiplications. If the spectral radius of each matrix deviates even slightly from unity:
1. **If weights are too large ($\|W\| > 1$)**: Activations compound exponentially at every layer:
   $$\text{Var}(a^{(l)}) \propto \alpha^l \to \infty$$
   Within 20 layers, activations explode to floating-point overflow (`inf` or `NaN`).
2. **If weights are too small ($\|W\| < 1$)**: Activations contract exponentially at every layer:
   $$\text{Var}(a^{(l)}) \propto \beta^l \to 0$$
   Within 20 layers, activations shrink to sub-normal floating-point zeros ($10^{-30}$), and backward gradients vanish entirely.
3. **The All-Zeros Trap**: Setting all weights to zero ($W = 0$) causes every neuron in a layer to compute the exact same activation and receive the exact same gradient. The neurons remain permanently identical throughout training—a catastrophic failure known as the **symmetry breaking failure**.

```
                THE GOLDILOCKS PRINCIPLE OF WEIGHT INITIALIZATION
                
  Too Small (Var << 1/n):       Just Right (Kaiming / Xavier):       Too Large (Var >> 1/n):
  ======================       =============================       ======================
  Layer 1:  Var = 1.0000       Layer 1:  Var = 1.0000              Layer 1:  Var = 1.0000
  Layer 10: Var = 0.0009       Layer 10: Var = 1.0420              Layer 10: Var = 1,024.0
  Layer 50: Var = 1e-25        Layer 50: Var = 0.9850              Layer 50: Var = 1.1e15
  (Signal Vanishes to 0!)      (Signal Preserved Perfectly!)       (Signal Explodes to NaN!)
```

The mathematical solution is to enforce the **Goldilocks Condition**: tune the variance of the initial weight distribution such that both **forward activation variance** and **backward gradient variance** remain constant from layer 1 to layer $L$.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Forward Variance Propagation in a Linear Layer

Consider a single neuron $i$ in layer $l$, computing the linear combination of $n_{\text{in}}$ inputs from layer $l-1$:
$$z_i = \sum_{j=1}^{n_{\text{in}}} w_{ij} x_j + b_i$$
We assume biases are initialized to zero ($b_i = 0$).

#### Statistical Assumptions:
1. The weights $w_{ij}$ are independent and identically distributed (i.i.d.) with zero mean: $\mathbb{E}[w_{ij}] = 0$ and variance $\text{Var}(w)$.
2. The input activations $x_j$ are independent of the weights $w_{ij}$, with zero mean: $\mathbb{E}[x_j] = 0$ and variance $\text{Var}(x)$.

#### Variance of a Product of Independent Random Variables:
For any two independent random variables $U$ and $V$:
$$\text{Var}(U V) = \mathbb{E}[U^2 V^2] - (\mathbb{E}[U V])^2 = \mathbb{E}[U^2] \mathbb{E}[V^2] - (\mathbb{E}[U]\mathbb{E}[V])^2$$
Since $\mathbb{E}[w_{ij}] = 0$ and $\mathbb{E}[x_j] = 0$:
$$\text{Var}(w_{ij} x_j) = \mathbb{E}[w_{ij}^2] \mathbb{E}[x_j^2] - 0 = \text{Var}(w) \text{Var}(x)$$

#### Variance of the Sum of $n_{\text{in}}$ Independent Terms:
$$\text{Var}(z_i) = \text{Var}\left( \sum_{j=1}^{n_{\text{in}}} w_{ij} x_j \right) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(w_{ij} x_j) = n_{\text{in}} \text{Var}(w) \text{Var}(x)$$

To ensure that the output variance equals the input variance ($\text{Var}(z) = \text{Var}(x)$), we must enforce:
$$n_{\text{in}} \text{Var}(w) = 1 \implies \mathbf{\text{Var}(w) = \frac{1}{n_{\text{in}}}}$$

---

### 2. Backward Variance Propagation in Backpropagation

During backpropagation, error sensitivities propagate backward through the transpose weight matrix:
$$\Delta_j = \sum_{i=1}^{n_{\text{out}}} w_{ij} \delta_i$$
Under identical statistical assumptions:
$$\text{Var}(\Delta_j) = n_{\text{out}} \text{Var}(w) \text{Var}(\delta)$$

To ensure that gradient variance does not explode or vanish backward ($\text{Var}(\Delta) = \text{Var}(\delta)$):
$$n_{\text{out}} \text{Var}(w) = 1 \implies \mathbf{\text{Var}(w) = \frac{1}{n_{\text{out}}}}$$

---

### 3. Xavier / Glorot Initialization (Glorot & Bengio 2010)

In general, layers have different input and output dimensions ($n_{\text{in}} \ne n_{\text{out}}$). Glorot & Bengio proposed taking the **harmonic mean** of the forward and backward constraints:
$$\frac{1}{\text{Var}(w)} = \frac{n_{\text{in}} + n_{\text{out}}}{2} \implies \mathbf{\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

#### 1. Xavier Normal Distribution:
$$w_{ij} \sim \mathcal{N}\left(0, \; \sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

#### 2. Xavier Uniform Distribution:
For a uniform distribution $\mathcal{U}(-a, a)$, its variance is $\frac{(a - (-a))^2}{12} = \frac{(2a)^2}{12} = \frac{4a^2}{12} = \frac{a^2}{3}$.
Setting $\frac{a^2}{3} = \frac{2}{n_{\text{in}} + n_{\text{out}}} \implies a^2 = \frac{6}{n_{\text{in}} + n_{\text{out}}}$.
$$\mathbf{w_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \; +\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)}$$

- **Applicability**: Optimal for symmetric, linear-regime activations: **Sigmoid, Tanh, and Linear layers**.
- **Fatal Flaw for ReLU**: Glorot assumed the activation function has derivative $\approx 1.0$ at the origin ($\tanh'(0) = 1$). But ReLU zeroes out half the real line!

---

### 4. He / Kaiming Initialization (He, Zhang, Ren, & Sun 2015)

When Kaiming He et al. analyzed deep networks with **ReLU activations** ($\max(0, z)$), they made a startling discovery:
Because ReLU sets all negative values to zero, **it cuts the activation energy in half at every layer!**

#### The Half-Variance Proof for ReLU:
Assume pre-activation $z \sim \mathcal{N}(0, \sigma_z^2)$. Its probability density function $p(z)$ is symmetric around zero:
$$\mathbb{E}[a^2] = \mathbb{E}[\text{ReLU}(z)^2] = \int_{-\infty}^\infty \max(0, z)^2 p(z) \, dz = \int_0^\infty z^2 p(z) \, dz$$
By symmetry, the integral over $[0, \infty)$ is exactly half the integral over $(-\infty, \infty)$:
$$\mathbb{E}[a^2] = \frac{1}{2} \int_{-\infty}^\infty z^2 p(z) \, dz = \frac{1}{2} \mathbb{E}[z^2] = \frac{1}{2} \text{Var}(z)$$

Now, propagate this through the next linear layer:
$$\text{Var}(z^{(l+1)}) = n_{\text{in}} \text{Var}(w) \cdot \mathbb{E}[(a^{(l)})^2] = n_{\text{in}} \text{Var}(w) \cdot \left( \frac{1}{2} \text{Var}(z^{(l)}) \right) = \frac{1}{2} n_{\text{in}} \text{Var}(w) \text{Var}(z^{(l)})$$

To maintain constant variance ($\text{Var}(z^{(l+1)}) = \text{Var}(z^{(l)})$), the factor must equal 1:
$$\frac{1}{2} n_{\text{in}} \text{Var}(w) = 1 \implies \mathbf{\text{Var}(w) = \frac{2}{n_{\text{in}}}}$$

#### 1. Kaiming Normal Distribution:
$$\mathbf{w_{ij} \sim \mathcal{N}\left(0, \; \sigma^2 = \frac{2}{n_{\text{in}}}\right)}$$

#### 2. Kaiming Uniform Distribution:
$$\text{Var} = \frac{a^2}{3} = \frac{2}{n_{\text{in}}} \implies a = \sqrt{\frac{6}{n_{\text{in}}}}$$
$$\mathbf{w_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \; +\sqrt{\frac{6}{n_{\text{in}}}}\right)}$$

#### Generalization to Leaky ReLU:
For Leaky ReLU with slope $\alpha$ for $z < 0$:
$$\mathbb{E}[\text{LeakyReLU}(z)^2] = \frac{1 + \alpha^2}{2} \text{Var}(z) \implies \mathbf{\text{Var}(w) = \frac{2}{(1 + \alpha^2) n_{\text{in}}}}$$

---

### 5. Orthogonal Initialization (Saxe, McClelland, & Ganguli 2013)

Instead of sampling independent random numbers, **Orthogonal Initialization** forces the weight matrix to be an orthogonal matrix:
$$W^T W = I \implies \|W x\|_2 = \|x\|_2$$
- **Construction**: Sample a random Gaussian matrix $M \in \mathbb{R}^{n \times n}$ and compute its QR decomposition: $M = Q R$. The orthogonal matrix $Q$ is assigned to $W$.
- **Singular Value Spectrum**: All singular values of $W$ are identically $1.0$ ($\sigma_i(W) = 1$).
- **Dynamical Isometry**: The entire network achieves dynamical isometry—all singular values of the input-output Jacobian remain bounded away from $0$ and $\infty$, enabling training of networks with **over 1,000 layers without normalization layers!**

---

## Part 3: Geometric Interpretation

### 1. Geometric Sphere Preservation
Consider an input vector $x$ residing on a hypersphere of radius $R$: $\|x\|_2 = R$.
- **Under standard Gaussian $\mathcal{N}(0, 1)$**:
  The expected squared norm of $W x$ is:
  $$\mathbb{E}[\|W x\|_2^2] = n_{\text{out}} n_{\text{in}} \text{Var}(w) \|x\|_2^2 = n_{\text{out}} \cdot n_{\text{in}} \cdot 1 \cdot R^2$$
  The radius expands by a factor of $\sqrt{n_{\text{out}} n_{\text{in}}}$. In high dimensions ($n = 1024$), the signal expands by $1024\times$ per layer!
- **Under Kaiming Initialization**:
  $W$ acts as an approximate isometry on the positive orthant, preserving the mean radius of the feature cloud from layer to layer.

---

## Part 4: Real-World Analogy

### 1. The Multi-Stage Public Address (PA) Audio System
Imagine a stadium audio system with 50 amplifiers connected in series:
- **Gain $g = 2.0$ (Weights Too Large)**: The first amplifier doubles the volume ($2\times$), the second quadruples it ($4\times$), the 50th multiplies it by $2^{50} \approx 10^{15}$. A gentle whisper becomes a catastrophic, speaker-shredding acoustic feedback explosion (`inf` / `NaN`).
- **Gain $g = 0.5$ (Weights Too Small)**: The audio signal is halved at every stage ($0.5^{50} \approx 10^{-15}$). The speaker at the other end emits dead silence (vanishing gradients).
- **Gain $g = 1.0$ (Kaiming / Xavier)**: Every amplifier perfectly compensates for cable resistance. The voice spoken into the microphone is reproduced with crystal clarity across the stadium.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us compute by hand the exact variance, standard deviation, and forward signal propagation for a 3-layer neural network with dimensions:
$$n_0 = 4, \quad n_1 = 4, \quad n_2 = 4, \quad n_3 = 4$$
under four distinct initialization regimes.

### 1. Theoretical Parameter Comparison

| Initialization Method | Mathematical Variance Formula | Value for $n_{\text{in}} = 4, n_{\text{out}} = 4$ | Standard Deviation $\sigma$ | Uniform Bound $a = \sqrt{3 \text{Var}}$ |
| :--- | :--- | :--- | :--- | :--- |
| **All-Zeros** | $0$ | $0.0000$ | $0.0000$ | $0.0000$ |
| **Standard Normal** | $1$ | $1.0000$ | $1.0000$ | $\sqrt{3} \approx 1.7321$ |
| **Xavier Normal** | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | $\frac{2}{4 + 4} = \frac{2}{8} = \mathbf{0.2500}$ | $\sqrt{0.25} = \mathbf{0.5000}$ | $\sqrt{0.75} \approx \mathbf{0.8660}$ |
| **Kaiming Normal (ReLU)**| $\frac{2}{n_{\text{in}}}$ | $\frac{2}{4} = \mathbf{0.5000}$ | $\sqrt{0.50} \approx \mathbf{0.7071}$ | $\sqrt{1.50} \approx \mathbf{1.2247}$ |

---

### 2. Forward Pass Activation Variance Tracking (ReLU Network)

Let the initial input have unit variance: $\text{Var}(a^{(0)}) = 1.0000$.
At each layer $l$, the expected variance of the pre-activation $z^{(l)}$ is:
$$\text{Var}(z^{(l)}) = n_{\text{in}} \text{Var}(w) \cdot \mathbb{E}[(a^{(l-1)})^2]$$
For ReLU, $\mathbb{E}[(a^{(l-1)})^2] = \frac{1}{2} \text{Var}(z^{(l-1)})$.
Therefore, the variance recurrence from layer to layer is:
$$\text{Var}(z^{(l)}) = \left( \frac{1}{2} n_{\text{in}} \text{Var}(w) \right) \text{Var}(z^{(l-1)})$$

Let us trace the variance across 3 consecutive layers ($l = 1, 2, 3$):

#### Case 1: Standard Normal ($\text{Var}(w) = 1.0$)
- Layer 1: $\text{Var}(z^{(1)}) = \left(\frac{1}{2} \times 4 \times 1.0\right) (1.0) = 2 \times 1.0 = \mathbf{2.0000}$
- Layer 2: $\text{Var}(z^{(2)}) = 2 \times \text{Var}(z^{(1)}) = 2 \times 2.0 = \mathbf{4.0000}$
- Layer 3: $\text{Var}(z^{(3)}) = 2 \times \text{Var}(z^{(2)}) = 2 \times 4.0 = \mathbf{8.0000}$
*(Variance doubles at every layer: $\text{Var}(z^{(l)}) = 2^l \to \infty$!)*

#### Case 2: Xavier Normal ($\text{Var}(w) = 0.25$)
- Layer 1: $\text{Var}(z^{(1)}) = \left(\frac{1}{2} \times 4 \times 0.25\right) (1.0) = 0.5 \times 1.0 = \mathbf{0.5000}$
- Layer 2: $\text{Var}(z^{(2)}) = 0.5 \times 0.5000 = \mathbf{0.2500}$
- Layer 3: $\text{Var}(z^{(3)}) = 0.5 \times 0.2500 = \mathbf{0.1250}$
*(Because ReLU zeroes out half the variance, Xavier causes the signal to halve at every layer: $\text{Var}(z^{(l)}) = 0.5^l \to 0$!)*

#### Case 3: Kaiming Normal ($\text{Var}(w) = 0.50$)
- Layer 1: $\text{Var}(z^{(1)}) = \left(\frac{1}{2} \times 4 \times 0.50\right) (1.0) = 1.0 \times 1.0 = \mathbf{1.0000}$
- Layer 2: $\text{Var}(z^{(2)}) = 1.0 \times 1.0000 = \mathbf{1.0000}$
- Layer 3: $\text{Var}(z^{(3)}) = 1.0 \times 1.0000 = \mathbf{1.0000}$
*(Variance remains EXACTLY 1.0000 at every layer! Perfect equilibrium!)*

---

### Comparative Visual Grid: Variance Evolution Across Depth

```
+----------------------------------------------------------------------------------------------------+
|                                    FORWARD ACTIVATION VARIANCE EVOLUTION (ReLU)                    |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Method            | Layer 1 Variance   | Layer 2 Variance   | Layer 3 Variance   | Layer 50 (Extr) |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| All-Zeros (W = 0) | 0.0000             | 0.0000             | 0.0000             | 0.0000 (Dead)   |
| Standard Normal   | 2.0000             | 4.0000             | 8.0000             | 1.12e15 (NaN!)  |
| Xavier Normal     | 0.5000             | 0.2500             | 0.1250             | 8.88e-16 (Zero!)|
| Kaiming Normal    | 1.0000 (PERFECT!)  | 1.0000 (PERFECT!)  | 1.0000 (PERFECT!)  | 1.0000 (STABLE!)|
+-------------------+--------------------+--------------------+--------------------+-----------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning Implementation |
| :--- | :--- | :--- | :--- |
| $n_{\text{in}}$ | Integer $\ge 1$ | Fan-in: Number of input connections | Input feature dimension (`layer.in_features`) |
| $n_{\text{out}}$ | Integer $\ge 1$ | Fan-out: Number of output connections | Output feature dimension (`layer.out_features`) |
| $\text{Var}(w)$ | $\mathbb{R}_{>0}$ scalar | Variance of weight distribution | Scaling parameter in `torch.nn.init` |
| $\sigma$ | $\mathbb{R}_{>0}$ scalar | Standard deviation $\sqrt{\text{Var}(w)}$ | Std argument in Gaussian sampler |
| $a$ | $\mathbb{R}_{>0}$ scalar | Half-width of uniform distribution $\mathcal{U}(-a, a)$ | Bound in uniform initializers |
| $Q$ | Orthogonal matrix | Isometry satisfying $Q^T Q = I$ | Weight tensor from `init.orthogonal_` |
| $\text{gain}$ | $\mathbb{R}_{>0}$ scalar | Activation-specific scaling constant | Value returned by `torch.nn.init.calculate_gain` |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Mathematical Proof of Symmetry Breaking Failure (All-Zeros)
**Statement**: Suppose all weights in an $L$-layer neural network are initialized to zero: $W^{(l)} = 0$ and $b^{(l)} = 0$ for all $l$.
Prove that regardless of training time or data, all hidden neurons within any given layer compute identical activations and receive identical gradients, preventing the network from learning distinct features.

**Proof**:
1. **Forward Pass**:
   At layer 1:
   $$z_i^{(1)} = \sum_j W_{ij}^{(1)} x_j + b_i^{(1)} = \sum_j (0) x_j + 0 = 0 \quad \forall i \in \{1, \dots, n_1\}$$
   $$a_i^{(1)} = \sigma(z_i^{(1)}) = \sigma(0) = c \quad (\text{a constant across all } i)$$
   By induction, for any layer $l$:
   $$z_i^{(l)} = \sum_j (0) c + 0 = 0 \implies a_i^{(l)} = \sigma(0) = c \quad \forall i$$
   All neurons within layer $l$ output the exact same scalar $c$.

2. **Backward Pass**:
   The output error $\delta^{(L)}$ is identical across any symmetric outputs.
   The error propagated to layer $l$ is:
   $$\delta_j^{(l)} = \left( \sum_k W_{kj}^{(l+1)} \delta_k^{(l+1)} \right) \sigma'(z_j^{(l)}) = 0 \cdot \sigma'(0) = 0$$
   For the final layer weights $W_{ik}^{(L)}$:
   $$\frac{\partial \mathcal{L}}{\partial W_{ik}^{(L)}} = \delta_i^{(L)} a_k^{(L-1)} = \delta_i^{(L)} \cdot c$$
   Because $a_k^{(L-1)} = c$ is independent of index $k$, every weight entering neuron $i$ receives the exact same gradient update:
   $$\Delta W_{ik}^{(L)} = -\eta \delta_i^{(L)} c$$
   After the update, all weights connected to neuron $i$ remain strictly equal:
   $$W_{ik}^{(L), \text{new}} = W_{ik'}^{(L), \text{new}} \quad \forall k, k'$$
3. Therefore, the neurons can never diverge from one another. A network with 10,000 neurons behaves identically to a network with 1 single neuron!
$\blacksquare$ **Random symmetry breaking is an absolute prerequisite for neural learning.**

---

### Problem 2: Scaled Residual Initialization in Transformers
**Statement**: In deep Transformers with $L$ residual layers:
$$x_{l+1} = x_l + \text{MLP}(x_l)$$
If each residual sub-layer adds a block with variance $\text{Var}(\text{MLP}(x_l)) \approx \text{Var}(x_l)$, show that the variance of the final activation $x_L$ scales as $\mathcal{O}(L)$. How does modern LLM architecture (GPT-3, LLaMA) fix this?

**Solution**:
1. Assuming the residual branch output $\text{MLP}(x_l)$ is uncorrelated with the skip connection $x_l$:
   $$\text{Var}(x_{l+1}) = \text{Var}(x_l) + \text{Var}(\text{MLP}(x_l))$$
   If $\text{Var}(\text{MLP}(x_l)) = \sigma_0^2$:
   $$\text{Var}(x_L) = \text{Var}(x_0) + L \cdot \sigma_0^2 = \mathcal{O}(L)$$
   In a 96-layer Transformer (GPT-3 175B), the activation variance explodes by nearly **$100\times$**, destabilizing attention softmax scores and causing fp16 overflow!
2. **The Modern Fix (Scaled Residual Initialization)**:
   Modern LLMs scale the weights of the *final projection* in each residual block (e.g., $W_{\text{down}}$ in MLP, $W_O$ in Multi-Head Attention) by an extra factor of $\frac{1}{\sqrt{2 L}}$:
   $$W_{\text{proj}} \sim \mathcal{N}\left(0, \; \frac{2}{n_{\text{in}} \cdot 2L}\right)$$
   This ensures $\text{Var}(\text{MLP}(x_l)) = \frac{1}{2L} \text{Var}(x_l)$, yielding:
   $$\text{Var}(x_L) \approx \text{Var}(x_0) \left( 1 + \frac{1}{2L} \right)^L \approx \text{Var}(x_0) \sqrt{e} \sim \mathcal{O}(1)$$
   The activation variance remains strictly $\mathcal{O}(1)$ regardless of depth!

---

## Part 7: Deep Learning Connection & Application

### 1. PyTorch `torch.nn.init` Implementation
PyTorch provides dedicated initializers:
```python
# Kaiming Normal (for ReLU / LeakyReLU):
torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Xavier Normal (for Tanh / Sigmoid):
torch.nn.init.xavier_normal_(layer.weight, gain=torch.nn.init.calculate_gain('tanh'))

# Orthogonal:
torch.nn.init.orthogonal_(layer.weight, gain=1.0)
```

### 2. Why PyTorch `nn.Linear` Uses LeCun Initialization by Default
Inside `torch.nn.modules.linear.py`, PyTorch does *not* initialize weights with Kaiming or Xavier normal. Instead, it uses **LeCun Uniform Initialization** (He et al. 2015 mode `fan_in` with uniform distribution):
$$W \sim \mathcal{U}\left(-\frac{1}{\sqrt{n_{\text{in}}}}, \; +\frac{1}{\sqrt{n_{\text{in}}}}\right)$$
$$b \sim \mathcal{U}\left(-\frac{1}{\sqrt{n_{\text{in}}}}, \; +\frac{1}{\sqrt{n_{\text{in}}}}\right)$$
This provides a safe, robust baseline across arbitrary layer widths and activations.

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `06_deep_learning_foundations/code/06_weight_initialization_schemes.py`. It implements:
1. Exact visual grid verification comparing theoretical variances and standard deviations for Xavier vs. Kaiming.
2. 50-layer deep network forward variance propagation simulation demonstrating:
   - Standard Normal ($\sigma=1.0$) exploding to $\gg 10^{10}$.
   - Xavier with ReLU collapsing to $\ll 10^{-10}$.
   - Kaiming Normal maintaining stable variance $\sim 1.0$ throughout all 50 layers.
3. Strict parity check against PyTorch native initializers (`init.xavier_normal_`, `init.kaiming_normal_`, `init.orthogonal_`).
4. Orthogonal initialization singular value spectrum verification ($\sigma_i \equiv 1.0$).

Save the code and run it directly in Python 3.
