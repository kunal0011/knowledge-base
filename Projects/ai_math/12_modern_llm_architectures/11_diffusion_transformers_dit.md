# 12.11 Diffusion Transformers (DiT): Replacing UNet with Transformers for Generative Modeling

---

## 1. Intuition & 101 Motivation

For the first five years of the diffusion revolution (DDPM, Latent Diffusion, Stable Diffusion 1.5/2.1), generative modeling was dominated by **convolutional UNets** augmented with intermittent cross-attention layers. While UNets provided local inductive biases suitable for pixel grids, they suffered from fundamental limitations:
1. **Poor Scaling Laws:** Increasing UNet capacity resulted in diminishing returns and severe training instabilities.
2. **Fixed 2D Inductive Bias:** UNets struggled to extend natively to 3D video volumes, variable aspect ratios, and multimodal token streams.

In 2023, Peebles & Xie introduced **Diffusion Transformers (DiT)**. DiT discards the convolutional UNet entirely in favor of a standard **Vision Transformer (ViT)** architecture operating on latent image patches. DiT demonstrated that diffusion models adhere to the exact same power-law scaling laws as Large Language Models: larger compute budgets and higher model capacities monotonically produce higher sample quality (lower FID).

Today, DiT is the undisputed backbone of frontier visual generative AI, powering:
- **OpenAI Sora** (video generation via space-time patch tokens)
- **Stable Diffusion 3** (Multimodal DiT - MMDiT)
- **Black Forest Labs Flux.1** (12B parameter flow-matching DiT)

```
                       CONVOLUTIONAL UNET vs. DiT
                       
      Traditional UNet                        Diffusion Transformer (DiT)
  ┌───────────────────────┐                    ┌─────────────────────────┐
  │ Downsampling Convol.  │                    │ Patchify Latent (p x p) │
  │        ▼              │                    │           ▼             │
  │ Bottleneck Attention  │                    │ adaLN-Zero Modulated    │
  │        ▼              │                    │ Transformer Blocks      │
  │ Upsampling Convol.    │                    │           ▼             │
  │ + Skip Connections    │                    │ Unpatchify to Latent    │
  └───────────────────────┘                    └─────────────────────────┘
  Inductive Bias: Local Convolutions           Inductive Bias: Global Self-Attention
  Scaling: Diminishing returns                 Scaling: Pure Power-Law Compute Scaling
```

---

## 2. Rigorous Mathematical Formulation

The DiT pipeline consists of three sequential stages:
1. **Patchification & Linear Embedding**
2. **adaLN-Zero Modulated Transformer Blocks**
3. **Unpatchification & Linear Projection**

---

### 2.1 Patchification & Positional Embedding

Given an input latent tensor $z \in \mathbb{R}^{C \times H \times W}$ (e.g., from a pre-trained VAE encoder where $C = 4$ or $C = 16$):
1. The spatial grid is partitioned into non-overlapping patches of size $p \times p$.
2. The number of visual tokens (sequence length) is:
   $$T = \left(\frac{H}{p}\right) \times \left(\frac{W}{p}\right)$$
3. Each patch has dimension $d_{\text{patch}} = C \cdot p^2$.
4. A linear projection maps each flattened patch to the Transformer hidden dimension $d$:
   $$x_0 = \operatorname{Linear}(z_{\text{patches}}) + E_{\text{pos}} \in \mathbb{R}^{T \times d}$$
   where $E_{\text{pos}}$ is a standard 2D frequency sinusoidal positional embedding.

---

### 2.2 Conditioning Injection: adaLN-Zero

In diffusion models, the network must be conditioned on:
- Continuous diffusion timestep $t \in \mathbb{R}$ (noise level)
- Optional conditioning vector $c$ (class label or text embedding)

#### 1. Conditioning Representation:
Timestep $t$ is mapped via frequency embedding into $\mathbb{R}^{d_{\text{time}}}$. Label $c$ is embedded into $\mathbb{R}^{d_{\text{label}}}$.
They are combined and projected via a 2-layer MLP to form conditioning vector $y \in \mathbb{R}^d$:
$$y = \operatorname{MLP}(e_t + e_c)$$

#### 2. Adaptive Layer Normalization with Zero Initialization (adaLN-Zero):
Rather than inserting cross-attention layers, DiT modulates standard Layer Normalization using $y$.
A linear projection maps $y$ to **6 modulation parameters** per Transformer block:
$$[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2] = \operatorname{Linear}_{\text{mod}}(y) \in \mathbb{R}^{6d}$$

where for each sub-layer (Self-Attention and MLP):
- $\gamma \in \mathbb{R}^d$: Dimension-wise scale factor
- $\beta \in \mathbb{R}^d$: Dimension-wise shift factor
- $\alpha \in \mathbb{R}^d$: Dimension-wise residual gating factor

#### 3. Modulated Transformer Block Forward Pass:
Given input token representations $x \in \mathbb{R}^{T \times d}$:

**Step A: Modulated Self-Attention:**
$$\hat{x} = \operatorname{adaLN}(x; \gamma_1, \beta_1) = (1 + \gamma_1) \odot \left( \frac{x - \mu}{\sigma + \epsilon} \right) + \beta_1$$
$$x' = x + \alpha_1 \odot \operatorname{MHA}(\hat{x})$$

**Step B: Modulated Feed-Forward Network (FFN):**
$$\hat{x}' = \operatorname{adaLN}(x'; \gamma_2, \beta_2) = (1 + \gamma_2) \odot \left( \frac{x' - \mu'}{\sigma' + \epsilon} \right) + \beta_2$$
$$x'' = x' + \alpha_2 \odot \operatorname{FFN}(\hat{x}')$$

---

### 2.3 Theorem: The Zero-Initialization Stability Property

#### Theorem:
If the weight and bias of the final linear layer in $\operatorname{Linear}_{\text{mod}}$ are initialized to **all zeros**, then at step $t=0$:
$$\gamma_1 = 0, \quad \beta_1 = 0, \quad \alpha_1 = 0$$
$$\gamma_2 = 0, \quad \beta_2 = 0, \quad \alpha_2 = 0$$
Under this condition, the entire DiT block acts as a pure **identity operator**:
$$x'' = x$$

#### Proof:
Substitute $\gamma_1 = 0, \beta_1 = 0, \alpha_1 = 0$:
$$\hat{x} = (1 + 0) \odot \operatorname{LN}(x) + 0 = \operatorname{LN}(x)$$
$$x' = x + 0 \odot \operatorname{MHA}(\hat{x}) = x + 0 = x$$
Substitute $\gamma_2 = 0, \beta_2 = 0, \alpha_2 = 0$:
$$\hat{x}' = (1 + 0) \odot \operatorname{LN}(x') + 0 = \operatorname{LN}(x)$$
$$x'' = x' + 0 \odot \operatorname{FFN}(\hat{x}') = x + 0 = x \quad \blacksquare$$

**Significance:** Just like Fixup initialization in ResNets, adaLN-Zero prevents exploding or vanishing signals in deep Transformer stacks at initialization, allowing models with 28+ layers (DiT-XL) to train smoothly without warmup hacks.

---

### 2.4 Unpatchification

After passing through $L$ blocks, the final representation is modulated via a final scale and shift:
$$x_{\text{out}} = (1 + \gamma_{\text{final}}) \odot \operatorname{LN}(x^{(L)}) + \beta_{\text{final}}$$
A linear projection decodes $x_{\text{out}} \in \mathbb{R}^{T \times d}$ to $\mathbb{R}^{T \times (p^2 \cdot C)}$.
The tokens are reshaped back to the latent grid $\mathbb{R}^{C \times H \times W}$, predicting the diffusion noise $\epsilon_\theta$ or flow vector $v_\theta$.

---

## 3. Geometric & Physical Interpretation

### 3.1 Global Receptive Field from Layer 1
In a convolutional UNet, pixel $(0, 0)$ can only communicate with pixel $(255, 255)$ after passing through multiple downsampling stages (the bottleneck).
In DiT, every spatial patch $i$ attends to every other patch $j$ in **Layer 1** through dense dot-product attention $Q K^T / \sqrt{d}$. The effective receptive field is instantaneous and global across the entire image canvas.

```
   CNN Receptive Field (Grows Linearly)         DiT Receptive Field (Instant Global)
   Layer 3: [     *     ]                       Layer 1: [*****************]
   Layer 2: [   *****   ]                       Layer 2: [*****************]
   Layer 1: [ ******* ]                         Layer 3: [*****************]
```

---

## 4. Real-World Analogy: The Movie Studio Lighting Rig

Imagine a film studio where 100 actors (patch tokens) are performing on stage:
- **Convolutional UNet:** Each actor can only whisper to their immediate neighbor on stage. Information takes 5 minutes to travel from stage left to stage right.
- **DiT:** Every actor wears an open radio headset (Self-Attention) and can hear and talk to any other actor instantly.
- **adaLN-Zero:** The studio director sits at a central lighting console (conditioning vector $y$). At every take (timestep $t$), the director dials a master knob that simultaneously alters the color tint ($\beta$), intensity contrast ($\gamma$), and microphone volume ($\alpha$) of all actors on stage without changing the script.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us execute a complete, step-by-step numerical trace of:
1. **LayerNorm Normalization Statistics**
2. **adaLN Scale, Shift, and Modulated Activation**
3. **Gated Attention Residual Update ($\alpha_1$)**
4. **Zero-Init Identity Preservation**

---

### 5.1 Concrete Input Values

- **Token dimension:** $d = 2$
- **Input token activation vector:**
  $$x = \begin{bmatrix} 2.0 \\ 4.0 \end{bmatrix} \in \mathbb{R}^2$$
- **LayerNorm Epsilon:** $\epsilon = 0.0$ (for clean hand arithmetic)
- **Regressed adaLN parameters from conditioning $y$:**
  - Scale factor $\gamma_1 = [0.20, -0.10]^T$
  - Shift factor $\beta_1 = [0.50, 0.50]^T$
  - Gate factor $\alpha_1 = [0.10, 0.10]^T$
- **Self-Attention Sub-Layer Output on modulated input:**
  $$\operatorname{MHA}(\hat{x}) = \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix}$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\mu$ | `mean` | Mean of token activation across dimension $d$ |
| $\sigma$ | `std` | Standard deviation of token activation |
| $\hat{x}_{\text{norm}}$ | `norm_x` | Standardized token vector $\frac{x - \mu}{\sigma}$ |
| $\gamma_1$ | `scale_gamma` | Conditioning scale vector $[0.20, -0.10]^T$ |
| $\beta_1$ | `shift_beta` | Conditioning shift vector $[0.50, 0.50]^T$ |
| $\hat{x}$ | `modulated_x` | Modulated input $(1 + \gamma_1) \odot \hat{x}_{\text{norm}} + \beta_1$ |
| $\alpha_1$ | `gate_alpha` | Residual gating multiplier $[0.10, 0.10]^T$ |
| $x'$ | `output_x` | Gated residual output $x + \alpha_1 \odot \operatorname{MHA}(\hat{x})$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute LayerNorm Statistics for $x = [2.0, 4.0]^T$
$$\mu = \frac{2.0 + 4.0}{2} = \frac{6.0}{2} = \mathbf{3.000000}$$
$$\sigma^2 = \frac{(2.0 - 3.0)^2 + (4.0 - 3.0)^2}{2} = \frac{(-1.0)^2 + (1.0)^2}{2} = \frac{1.0 + 1.0}{2} = \mathbf{1.000000}$$
$$\sigma = \sqrt{1.0} = \mathbf{1.000000}$$

$$\hat{x}_{\text{norm}} = \frac{x - \mu}{\sigma} = \begin{bmatrix} \frac{2.0 - 3.0}{1.0} \\ \frac{4.0 - 3.0}{1.0} \end{bmatrix} = \begin{bmatrix} \mathbf{-1.000000} \\ \mathbf{1.000000} \end{bmatrix}$$

---

#### Step 2: Apply adaLN Modulation
$$\hat{x} = (1 + \gamma_1) \odot \hat{x}_{\text{norm}} + \beta_1$$

- Component 1:
  $$1 + \gamma_{1, 1} = 1 + 0.20 = 1.20$$
  $$\hat{x}_1 = 1.20 \times (-1.000000) + 0.50 = -1.20 + 0.50 = \mathbf{-0.700000}$$
- Component 2:
  $$1 + \gamma_{1, 2} = 1 + (-0.10) = 0.90$$
  $$\hat{x}_2 = 0.90 \times (1.000000) + 0.50 = 0.90 + 0.50 = \mathbf{1.400000}$$

$$\hat{x} = \begin{bmatrix} \mathbf{-0.700000} \\ \mathbf{1.400000} \end{bmatrix}$$

---

#### Step 3: Compute Gated Residual Attention Update
$$\operatorname{MHA}(\hat{x}) = \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix}$$
$$\alpha_1 = \begin{bmatrix} 0.10 \\ 0.10 \end{bmatrix}$$

- Gated delta:
  $$\Delta x = \alpha_1 \odot \operatorname{MHA}(\hat{x}) = \begin{bmatrix} 0.10 \times 3.0 \\ 0.10 \times (-1.0) \end{bmatrix} = \begin{bmatrix} \mathbf{0.300000} \\ \mathbf{-0.100000} \end{bmatrix}$$

- Residual addition:
  $$x' = x + \Delta x = \begin{bmatrix} 2.0 \\ 4.0 \end{bmatrix} + \begin{bmatrix} 0.30 \\ -0.10 \end{bmatrix} = \begin{bmatrix} \mathbf{2.300000} \\ \mathbf{3.900000} \end{bmatrix}$$

---

#### Step 4: Verify Zero-Initialization Identity Preservation
If initialization has $\alpha_1 = [0.0, 0.0]^T$:
$$\Delta x_{\text{init}} = \begin{bmatrix} 0.0 \times 3.0 \\ 0.0 \times (-1.0) \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$
$$x'_{\text{init}} = x + \Delta x_{\text{init}} = \begin{bmatrix} 2.0 \\ 4.0 \end{bmatrix} = x \implies \mathbf{\text{Exact Identity Verified!}} \quad \blacksquare$$

---

### 5.4 Summary Visual Grid: adaLN-Zero Execution Ledger

| Step | Operation | Formula | Component 1 | Component 2 |
| :---: | :--- | :--- | :---: | :---: |
| **0** | Input | $x$ | $2.0000$ | $4.0000$ |
| **1** | Mean & Std | $\mu, \sigma$ | $\mu = 3.0000$ | $\sigma = 1.0000$ |
| **2** | Normalized | $\hat{x}_{\text{norm}}$ | $-1.0000$ | $+1.0000$ |
| **3** | Scale Multiplier | $1 + \gamma_1$ | $1.2000$ | $0.9000$ |
| **4** | Modulated Token | $\hat{x}$ | $\mathbf{-0.7000}$ | $\mathbf{+1.4000}$ |
| **5** | Attention Out | $\operatorname{MHA}(\hat{x})$ | $3.0000$ | $-1.0000$ |
| **6** | Gated Update | $\alpha_1 \odot \operatorname{MHA}$ | $+0.3000$ | $-0.1000$ |
| **7** | Final Block Out | $x'$ | $\mathbf{2.3000}$ | $\mathbf{3.9000}$ |

---

## 6. Solved Illustrations

### Illustration 1: Patchification Arithmetic for Latent Space
**Problem:**
A high-resolution image of size $1024 \times 1024 \times 3$ is encoded by a $8\times$ downsampling VAE into latent $z \in \mathbb{R}^{4 \times 128 \times 128}$.
Using a DiT with patch size $p = 2$:
1. What is the total token sequence length $T$?
2. What is the flattened dimension of each patch $d_{\text{patch}}$ before linear projection?
3. How does this compare to patch size $p = 4$?

**Solution:**
1. **Latent dimensions:** $H_{\text{lat}} = 128, W_{\text{lat}} = 128, C = 4$.
   For patch size $p = 2$:
   $$T_{p=2} = \left(\frac{128}{2}\right) \times \left(\frac{128}{2}\right) = 64 \times 64 = \mathbf{4,096 \text{ tokens}}$$
2. **Flattened patch dimension:**
   $$d_{\text{patch}} = C \cdot p^2 = 4 \times 2^2 = 4 \times 4 = \mathbf{16 \text{ features}}$$
3. **For patch size $p = 4$:**
   $$T_{p=4} = \left(\frac{128}{4}\right) \times \left(\frac{128}{4}\right) = 32 \times 32 = \mathbf{1,024 \text{ tokens}}$$
   $$d_{\text{patch}} = 4 \times 4^2 = 4 \times 16 = \mathbf{64 \text{ features}}$$
   *(Notice: Halving patch size quadruples sequence length $T$, which increases attention memory and compute by $16\times$ ($O(T^2)$), but captures significantly sharper high-frequency visual details!).* $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **OpenAI Sora & LTX-Video:** Rather than 2D spatial patches, Sora uses **3D Space-Time Patches** ($p_t \times p_h \times p_w$). A video volume $C \times T \times H \times W$ is flattened into spatio-temporal tokens, allowing DiT to generate consistent physical motion over time.
- **Flux.1 (Black Forest Labs):** Employs a 12-billion parameter hybrid Flow-Matching DiT with dual-stream attention blocks for text and visual tokens, surpassing Midjourney v6 in photorealism and typography rendering.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - LayerNorm normalization matching $[-1.0, 1.0]^T$.
   - adaLN modulation matching $[-0.700000, 1.400000]^T$.
   - Gated residual update matching $[2.300000, 3.900000]^T$.
   - Zero-init identity check.
2. **Production-Ready PyTorch `DiTBlock`:**
   - Modulated adaLN-Zero layer with 6 regressed parameters.
   - Multi-head self-attention and MLP with residual scaling.
   - Forward pass, invertible patchify/unpatchify functions, and gradient backpropagation.

See implementation in:
[`12_modern_llm_architectures/code/11_diffusion_transformers_dit.py`](./code/11_diffusion_transformers_dit.py)
