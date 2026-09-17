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

### Illustration 2: AdaLN Conditioning Computation
**Problem:**
Compute the Adaptive Layer Normalization parameters and modulated output.
Given a class embedding $c = [1.5, -0.5]^T$, project it to a scale $\gamma$ and shift $\beta$ using:
- $W_s = \begin{bmatrix} 2 & 1 \\ -1 & 3 \end{bmatrix}, b_s = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$
- $W_{sh} = \begin{bmatrix} 1 & -1 \\ 0.5 & 2 \end{bmatrix}, b_{sh} = \begin{bmatrix} -0.2 \\ 0.3 \end{bmatrix}$
Apply the modulation: $\text{output} = \gamma \odot \text{RMSNorm}(x) + \beta$ for $x = [0.8, -0.3]^T$ (Assume RMSNorm gives $[0.936, -0.351]^T$).

**Solution:**
1. **Compute Scale $\gamma$:**
   $$\gamma = W_s \cdot c + b_s = \begin{bmatrix} 2 & 1 \\ -1 & 3 \end{bmatrix} \begin{bmatrix} 1.5 \\ -0.5 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$
   $$\gamma = \begin{bmatrix} 2(1.5) + 1(-0.5) \\ -1(1.5) + 3(-0.5) \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 3 - 0.5 \\ -1.5 - 1.5 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 2.5 \\ -3.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \mathbf{\begin{bmatrix} 2.6 \\ -3.1 \end{bmatrix}}$$

2. **Compute Shift $\beta$:**
   $$\beta = W_{sh} \cdot c + b_{sh} = \begin{bmatrix} 1 & -1 \\ 0.5 & 2 \end{bmatrix} \begin{bmatrix} 1.5 \\ -0.5 \end{bmatrix} + \begin{bmatrix} -0.2 \\ 0.3 \end{bmatrix}$$
   $$\beta = \begin{bmatrix} 1.5 - (-0.5) \\ 0.75 - 1.0 \end{bmatrix} + \begin{bmatrix} -0.2 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 2.0 \\ -0.25 \end{bmatrix} + \begin{bmatrix} -0.2 \\ 0.3 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.8 \\ 0.05 \end{bmatrix}}$$

3. **Apply Modulation:**
   $$\text{output} = \gamma \odot \text{RMSNorm}(x) + \beta = \begin{bmatrix} 2.6 \\ -3.1 \end{bmatrix} \odot \begin{bmatrix} 0.936 \\ -0.351 \end{bmatrix} + \begin{bmatrix} 1.8 \\ 0.05 \end{bmatrix}$$
   $$\text{output} = \begin{bmatrix} 2.4336 \\ 1.0881 \end{bmatrix} + \begin{bmatrix} 1.8 \\ 0.05 \end{bmatrix} = \mathbf{\begin{bmatrix} 4.2336 \\ 1.1381 \end{bmatrix}} \quad \blacksquare$$

---

### Illustration 3: Forward Diffusion Process
**Problem:**
A 2D latent image $x_0 = [1.0, -1.0]^T$ is corrupted by a forward diffusion process. 
The cumulative noise schedule is: $\bar{\alpha}_1 = 0.9, \bar{\alpha}_{10} = 0.5, \bar{\alpha}_{100} = 0.05$.
Given sampled noise $\epsilon = [0.5, 0.7]^T$, compute $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ and the signal-to-noise ratio multiplier $\sqrt{\bar{\alpha}_t / (1 - \bar{\alpha}_t)}$ at each $t$.

**Solution:**
1. **At $t=1$:**
   - $x_1 = \sqrt{0.9}[1, -1]^T + \sqrt{0.1}[0.5, 0.7]^T$
   - $x_1 = 0.9487[1, -1]^T + 0.3162[0.5, 0.7]^T = [0.9487, -0.9487]^T + [0.1581, 0.2214]^T = \mathbf{[1.1068, -0.7273]^T}$
   - $\text{SNR multiplier} = \sqrt{\frac{0.9}{0.1}} = \sqrt{9} = \mathbf{3.00}$ (Strong signal)

2. **At $t=10$:**
   - $x_{10} = \sqrt{0.5}[1, -1]^T + \sqrt{0.5}[0.5, 0.7]^T$
   - $x_{10} = 0.7071[1, -1]^T + 0.7071[0.5, 0.7]^T = [0.7071, -0.7071]^T + [0.3536, 0.4950]^T = \mathbf{[1.0607, -0.2121]^T}$
   - $\text{SNR multiplier} = \sqrt{\frac{0.5}{0.5}} = \sqrt{1} = \mathbf{1.00}$ (Equal signal and noise)

3. **At $t=100$:**
   - $x_{100} = \sqrt{0.05}[1, -1]^T + \sqrt{0.95}[0.5, 0.7]^T$
   - $x_{100} = 0.2236[1, -1]^T + 0.9747[0.5, 0.7]^T = [0.2236, -0.2236]^T + [0.4873, 0.6823]^T = \mathbf{[0.7109, 0.4587]^T}$
   - $\text{SNR multiplier} = \sqrt{\frac{0.05}{0.95}} = \sqrt{0.0526} = \mathbf{0.2294}$ (Signal destroyed) $\blacksquare$

---

### Illustration 4: DiT Parameter Count Breakdown
**Problem:**
Estimate the parameter budget for a **DiT-XL/2** model.
Hyperparameters: patch\_size = 2, img\_size = 256, embedding\_dim $d = 1152$, depth = 28, n\_heads = 16. SwiGLU FFN expansion = $4 \times d = 4608$.

**Solution:**
1. **Input Tokens:**
   - $\text{Patches} = (256/2)^2 = 128^2 = \mathbf{16,384 \text{ tokens}}$

2. **Patch Embedding Layer:**
   - $W_{\text{embed}} \in \mathbb{R}^{d_{\text{patch}} \times d}$, where $d_{\text{patch}} = 3 \times 2 \times 2 = 12$.
   - Params = $12 \times 1152 = \mathbf{13,824}$

3. **Per-Layer Parameters:**
   - **Self-Attention:** 4 weight matrices ($W_Q, W_K, W_V, W_O$).
     Params = $4 \times 1152^2 = \mathbf{5.31 \text{ M}}$
   - **SwiGLU FFN:** 3 weight matrices ($W_1, W_2, W_3$ mapping $1152 \to 4608$ or vice versa).
     Params = $3 \times 1152 \times 4608 = \mathbf{15.93 \text{ M}}$
   - **Total per layer** $\approx 5.31 + 15.93 = \mathbf{21.24 \text{ M}}$

4. **Total Core Transformer Budget:**
   - For 28 layers: $28 \times 21.24\text{ M} \approx \mathbf{594.7 \text{ M parameters}}$

*(Compared to SD 1.5's UNet at $\sim 860\text{M}$ params, DiT achieves vastly superior results with a highly structured, dense parameter allocation without convolutions).* $\blacksquare$

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
