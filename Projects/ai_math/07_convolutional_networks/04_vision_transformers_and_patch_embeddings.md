# Chapter 7.4: Vision Transformers (ViT) & Patch Embeddings

---

## 1. Intuition & 101 Motivation

For over a decade, computer vision was anchored on the conviction that translation equivariance and local connectivity (the core inductive biases of convolutions) were indispensable prerequisites for visual processing.

In 2020, Alexey Dosovitskiy and the Google Brain team challenged this foundational dogma with **ViT (Vision Transformer)** in the landmark paper:
> *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*

Instead of sliding small convolutional kernels across pixel grids:
1. The image is diced into a grid of non-overlapping square patches (e.g., $16 \times 16$ pixels).
2. Each 2D patch is flattened and linearly projected into a 1D vector embedding—analogous to a "token" in Natural Language Processing (NLP).
3. A learnable `[CLS]` token is prepended to aggregate global representation.
4. Position embeddings are added to restore spatial coordinates.
5. The sequence of patch tokens is passed directly into a standard, unmodified **Transformer Encoder**.

```
           Vision Transformer (ViT) Architecture Overview
┌────────────────────────────────────────────────────────────────────────┐
│                        Classification Head (MLP)                       │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ [CLS] Token
┌───────────────────────────────────┴────────────────────────────────────┐
│                    Transformer Encoder Stack (L layers)                │
│             ┌──────────────────────────────────────────────┐           │
│             │  LayerNorm ──► Multi-Head Attention ──► [ + ]│           │
│             │  LayerNorm ──► MLP (GELU)           ──► [ + ]│           │
│             └──────────────────────────────────────────────┘           │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │
               Sequence: [ [CLS], E_1, E_2, E_3, ... , E_N ]
                                    ▲
                                    │ + Position Embeddings (1D)
    Linear Projection: [ E_cls, x_p1*E, x_p2*E, x_p3*E, ... , x_pN*E ]
                                    ▲
                                    │
          Flattened Patches: [ x_p1, x_p2, x_p3, ... , x_pN ]
                                    ▲
                                    │ Patchify (P x P = 16 x 16)
┌───────────────────────────────────┴────────────────────────────────────┐
│                       Input Image (H x W x C)                          │
└────────────────────────────────────────────────────────────────────────┘
```

### The Inductive Bias Trade-off: CNN vs. ViT
- **CNNs have Strong Hardcoded Inductive Bias:**
  - *Locality:* Neurons only interact with neighboring pixels in receptive windows.
  - *Translation Equivariance:* Shifting an object in the input shifts the feature representation identically.
  - *Data Requirement:* Low to medium. CNNs learn competitive representations from scratch on small datasets (e.g., CIFAR-10, ImageNet-1k).
- **ViTs have Weak Inductive Bias:**
  - *Global Connectivity:* From layer 1, any patch can attend to any other patch across the entire canvas ($O(N^2)$ global self-attention).
  - *Permutation Equivariant:* The model has zero built-in knowledge of 2D geometry; spatial arrangement must be learned purely through position embeddings.
  - *Data Requirement:* High. On ImageNet-1k, plain ViT underperforms ResNet. But when pre-trained on web-scale datasets (ImageNet-21k, JFT-300M), ViT shatters CNN performance limits and scales monotonically with compute.

---

## 2. Rigorous Mathematical Formulation

### 2.1 Patch Extraction & Linear Projection

Let the input image be:
$$\mathbf{x} \in \mathbb{R}^{H \times W \times C}$$
where $H$ is height, $W$ is width, and $C$ is the number of channels (typically $3$).

We partition $\mathbf{x}$ into a sequence of non-overlapping 2D patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 C)}$, where:
- $P$ is the patch spatial resolution (typically $P = 16$ or $P = 14$),
- $N = \frac{HW}{P^2}$ is the resulting number of image patches (the effective sequence length).

Each flattened patch vector $\mathbf{x}_p^i \in \mathbb{R}^{P^2 C}$ is linearly mapped to the model's hidden dimension $D$ using a learnable projection matrix $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$:
$$\mathbf{e}_i = \mathbf{x}_p^i \mathbf{E} \in \mathbb{R}^D, \quad \forall i \in \{1, \dots, N\}$$

#### Equivalence to 2D Convolution:
This patch projection is mathematically identical to a 2D convolution with:
- Kernel size: $K_H = P, K_W = P$,
- Stride: $s_H = P, s_W = P$,
- Number of output filters: $C_{\text{out}} = D$.

The convolution outputs a tensor of shape $(B, D, H/P, W/P)$, which is flattened along spatial dimensions to $(B, N, D)$.

---

### 2.2 Class Token (`[CLS]`) & Position Embeddings

#### The Class Token
Borrowing BERT’s convention, a learnable token $\mathbf{x}_{\text{class}} \in \mathbb{R}^{1 \times D}$ is prepended to the sequence of patch projections. Its state at the output of the Transformer encoder serves as the global image representation $\mathbf{y}$:
$$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}}; \, \mathbf{x}_p^1 \mathbf{E}; \, \mathbf{x}_p^2 \mathbf{E}; \, \dots; \, \mathbf{x}_p^N \mathbf{E} \right] \in \mathbb{R}^{(N + 1) \times D}$$

Why use a `[CLS]` token instead of average pooling over patches?
Global average pooling forces the network to distribute classification features uniformly across all patches. The `[CLS]` token allows self-attention to selectively attend to task-relevant patches while leaving background/texture patches unburdened.

#### Position Embeddings
Because self-attention is permutation-invariant ($\text{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P} \text{Attention}(\mathbf{X})$ for any permutation matrix $\mathbf{P}$), spatial order must be injected:
$$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}}; \, \mathbf{x}_p \mathbf{E} \right] + \mathbf{E}_{\text{pos}}$$
where $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N + 1) \times D}$ is a learnable 1D position embedding matrix.

Remarkably, even though 1D position indices $0, 1, \dots, N$ are assigned sequentially, the trained embedding vectors learn 2D spatial grid coordinates: patches in the same row or column exhibit high cosine similarity!

---

### 2.3 Transformer Encoder Equations (Pre-LN Formulation)

The sequence $\mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}$ passes through $L$ identical Transformer encoder blocks. Each block consists of Multi-Head Self-Attention (MSA) and a Multi-Layer Perceptron (MLP), with Layer Normalization applied **before** each sub-block (Pre-LN):

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}, \quad l = 1, \dots, L$$
$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l, \quad l = 1, \dots, L$$

#### Multi-Head Self-Attention (MSA)
For a single head with dimension $d_k = D / h$:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$
$$\text{MSA}(\mathbf{X}) = \left[ \text{head}_1; \dots; \text{head}_h \right] \mathbf{W}_O$$

#### Multi-Layer Perceptron (MLP)
The MLP consists of two linear layers with GELU non-linearity, expanding the dimension by $4\times$:
$$\text{MLP}(\mathbf{u}) = \text{GELU}(\mathbf{u} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$
where $\mathbf{W}_1 \in \mathbb{R}^{D \times 4D}$ and $\mathbf{W}_2 \in \mathbb{R}^{4D \times D}$.

---

### 2.4 Classification Head

After $L$ layers, the representation of the `[CLS]` token at index $0$ is normalized and mapped to class logits:
$$\mathbf{y} = \mathbf{W}_{\text{head}} \, \text{LN}(\mathbf{z}_L^0) + \mathbf{b}_{\text{head}}$$
where $\mathbf{W}_{\text{head}} \in \mathbb{R}^{K \times D}$ for $K$ target classes.

---

### 2.5 2D Bicubic Position Interpolation for Higher Resolutions

When a model pre-trained at resolution $H \times W$ (e.g., $224 \times 224$ with $N = 14 \times 14 = 196$ patches) is fine-tuned at higher resolution $H' \times W'$ (e.g., $384 \times 384$ with $N' = 24 \times 24 = 576$ patches):
- Patch size $P$ remains constant ($16 \times 16$).
- The sequence length increases from $N \to N'$.
- The learned position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ must be adapted.

**Interpolation Algorithm:**
1. Separate `[CLS]` position embedding $\mathbf{E}_{\text{pos}}^0 \in \mathbb{R}^{1 \times D}$.
2. Reshape the remaining $N$ spatial embeddings into a 2D grid:
   $$\mathbf{E}_{\text{grid}} \in \mathbb{R}^{\sqrt{N} \times \sqrt{N} \times D}$$
3. Perform **2D bicubic interpolation** from grid size $\sqrt{N} \times \sqrt{N}$ to $\sqrt{N'} \times \sqrt{N'}$:
   $$\mathbf{E}'_{\text{grid}} = \text{BicubicInterpolate}(\mathbf{E}_{\text{grid}}, (\sqrt{N'}, \sqrt{N'}))$$
4. Flatten back to $\mathbb{R}^{N' \times D}$ and concatenate the original `[CLS]` embedding.

This allows pre-trained ViTs to seamlessly process images of arbitrarily high resolutions without discarding learned spatial structure.

---

## 3. Geometric & Algebraic Interpretation

### Graph Geometry: CNN vs. ViT

| Property | Convolutional Neural Network (CNN) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| **Underlying Graph** | Fixed lattice grid (each pixel connects only to $3 \times 3$ neighbors) | **Complete Graph $K_N$** (every patch connects to every other patch) |
| **Edge Weights** | Fixed spatial kernels $\mathbf{K}$ (independent of input content) | **Dynamic, Content-Dependent:** $A_{i, j} = \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d})}{\sum_m \exp(\mathbf{q}_i^T \mathbf{k}_m / \sqrt{d})}$ |
| **Effective Receptive Field** | Grows linearly with depth: $r_l = r_{l-1} + (k-1)$ | **Infinite ($100\%$ canvas)** at Layer 1 |
| **Computational Complexity** | Linear in pixels: $O(H W \cdot C^2)$ | Quadratic in patches: $O(N^2 D) = O\left(\frac{(HW)^2}{P^4} D\right)$ |

In a CNN, long-range dependencies (e.g., relating a dog's head in the top-left to its tail in the bottom-right) require dozens of stacked layers to expand the receptive field.
In a ViT, the head and tail attend to each other in a single hop at layer 1.

---

## 4. Real-World Analogy

### The Magnifying Glass Inspector vs. The Puzzle Assembly Board
- **CNN (Magnifying Glass Inspector):** An investigator inspects a massive mural by peering through a $3 \times 3$ inch magnifying glass, sliding it methodically across every inch of the wall. To understand the whole scene, they must synthesize notes across multiple passes.
- **ViT (Puzzle Assembly Board):** A team cuts the mural into square puzzle tiles (patches). They throw all tiles onto a large conference table at once, label each tile with its original coordinate (position embedding), and place an empty clipboard in the center (the `[CLS]` token). Every tile simultaneously looks around the room, identifies the tiles it relates to (self-attention), and writes relevant discoveries onto the central clipboard.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace patch projection, sequence construction, and self-attention on a concrete toy example.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Vision Transformer |
| :--- | :--- | :--- | :--- |
| $\mathbf{X}$ | Input Image | $(4, 4)$ | Single-channel toy image canvas |
| $P$ | Patch Size | Scalar ($2$) | Spatial dimension of each patch ($2 \times 2$) |
| $N$ | Number of Patches | Scalar ($4$) | $N = (4 \times 4)/(2 \times 2) = 4$ image patches |
| $\mathbf{x}_p$ | Flattened Patches | $(4, 4)$ | Matrix of $N=4$ rows, each row of length $P^2 C = 4$ |
| $\mathbf{E}$ | Projection Matrix | $(4, 2)$ | Linear projection from patch dimension $4 \to D=2$ |
| $\mathbf{x}_{\text{class}}$ | Class Token | $(1, 2)$ | Learnable classification token |
| $\mathbf{E}_{\text{pos}}$ | Position Embeddings | $(5, 2)$ | 1D spatial embeddings for tokens $0, \dots, 4$ |
| $\mathbf{z}_0$ | Input Sequence | $(5, 2)$ | Embedded sequence fed into Transformer encoder |

---

### 5.2 Concrete Toy Numbers

#### Input Image $\mathbf{X}$ ($4 \times 4$ pixels):
$$\mathbf{X} = \begin{bmatrix}
1 & 2 & 0 & 1 \\
3 & 0 & 2 & 1 \\
1 & 1 & 4 & 0 \\
0 & 2 & 1 & 3
\end{bmatrix}$$

---

### 5.3 Step 1: Patch Extraction & Flattening ($P = 2$)

Partition $\mathbf{X}$ into four non-overlapping $2 \times 2$ patches:

1. **Patch 1 (Top-Left):**
   $$\mathbf{P}_1 = \begin{bmatrix} 1 & 2 \\ 3 & 0 \end{bmatrix} \xrightarrow{\text{Flatten}} \mathbf{x}_p^1 = \begin{bmatrix} 1 & 2 & 3 & 0 \end{bmatrix}$$

2. **Patch 2 (Top-Right):**
   $$\mathbf{P}_2 = \begin{bmatrix} 0 & 1 \\ 2 & 1 \end{bmatrix} \xrightarrow{\text{Flatten}} \mathbf{x}_p^2 = \begin{bmatrix} 0 & 1 & 2 & 1 \end{bmatrix}$$

3. **Patch 3 (Bottom-Left):**
   $$\mathbf{P}_3 = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix} \xrightarrow{\text{Flatten}} \mathbf{x}_p^3 = \begin{bmatrix} 1 & 1 & 0 & 2 \end{bmatrix}$$

4. **Patch 4 (Bottom-Right):**
   $$\mathbf{P}_4 = \begin{bmatrix} 4 & 0 \\ 1 & 3 \end{bmatrix} \xrightarrow{\text{Flatten}} \mathbf{x}_p^4 = \begin{bmatrix} 4 & 0 & 1 & 3 \end{bmatrix}$$

Stacked Flattened Patches $\mathbf{x}_p \in \mathbb{R}^{4 \times 4}$:
$$\mathbf{x}_p = \begin{bmatrix}
1 & 2 & 3 & 0 \\
0 & 1 & 2 & 1 \\
1 & 1 & 0 & 2 \\
4 & 0 & 1 & 3
\end{bmatrix}$$

---

### 5.4 Step 2: Linear Patch Projection ($D = 2$)

Let the projection matrix $\mathbf{E} \in \mathbb{R}^{4 \times 2}$ be:
$$\mathbf{E} = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & -1 \\
-1 & 1
\end{bmatrix}$$

Compute $\mathbf{e}_i = \mathbf{x}_p^i \mathbf{E}$:

1. **Patch 1:**
   $$\mathbf{e}_1 = [1(1) + 2(0) + 3(1) + 0(-1), \, 1(0) + 2(1) + 3(-1) + 0(1)] = [1 + 3, \, 2 - 3] = \mathbf{[4, -1]}$$
2. **Patch 2:**
   $$\mathbf{e}_2 = [0(1) + 1(0) + 2(1) + 1(-1), \, 0(0) + 1(1) + 2(-1) + 1(1)] = [2 - 1, \, 1 - 2 + 1] = \mathbf{[1, 0]}$$
3. **Patch 3:**
   $$\mathbf{e}_3 = [1(1) + 1(0) + 0(1) + 2(-1), \, 1(0) + 1(1) + 0(-1) + 2(1)] = [1 - 2, \, 1 + 2] = \mathbf{[-1, 3]}$$
4. **Patch 4:**
   $$\mathbf{e}_4 = [4(1) + 0(0) + 1(1) + 3(-1), \, 4(0) + 0(1) + 1(-1) + 3(1)] = [4 + 1 - 3, \, -1 + 3] = \mathbf{[2, 2]}$$

$$\mathbf{x}_p \mathbf{E} = \begin{bmatrix}
4 & -1 \\
1 & 0 \\
-1 & 3 \\
2 & 2
\end{bmatrix}$$

---

### 5.5 Step 3: Prepending `[CLS]` Token and Adding Position Embeddings

Let the learnable class token be:
$$\mathbf{x}_{\text{class}} = \begin{bmatrix} 0 & 1 \end{bmatrix}$$

The sequence before position embeddings is:
$$\mathbf{z}_{\text{unposed}} = \begin{bmatrix}
0 & 1 \\
4 & -1 \\
1 & 0 \\
-1 & 3 \\
2 & 2
\end{bmatrix} \quad \begin{array}{l} \text{(Token 0: [CLS])} \\ \text{(Token 1: Patch 1)} \\ \text{(Token 2: Patch 2)} \\ \text{(Token 3: Patch 3)} \\ \text{(Token 4: Patch 4)} \end{array}$$

Let the learned position embedding matrix $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{5 \times 2}$ be:
$$\mathbf{E}_{\text{pos}} = \begin{bmatrix}
0.5 & 0.5 \\
0.1 & -0.1 \\
-0.2 & 0.2 \\
0.3 & -0.3 \\
-0.4 & 0.4
\end{bmatrix}$$

The input sequence $\mathbf{z}_0 = \mathbf{z}_{\text{unposed}} + \mathbf{E}_{\text{pos}}$ is:
$$\mathbf{z}_0 = \begin{bmatrix}
0 + 0.5 & 1 + 0.5 \\
4 + 0.1 & -1 - 0.1 \\
1 - 0.2 & 0 + 0.2 \\
-1 + 0.3 & 3 - 0.3 \\
2 - 0.4 & 2 + 0.4
\end{bmatrix} = \mathbf{\begin{bmatrix}
0.5 & 1.5 \\
4.1 & -1.1 \\
0.8 & 0.2 \\
-0.7 & 2.7 \\
1.6 & 2.4
\end{bmatrix}}$$

This $(5, 2)$ matrix enters the Transformer encoder. Every token now carries both semantic patch features and spatial location coordinates.

---

## 6. Solved Illustrations

### Illustration 1: Quadratic Complexity Scaling ($N$ vs. $P$)

**Problem:**
Consider processing a $512 \times 512$ image through ViT with hidden dimension $D = 768$.
Compare the sequence length $N$ and the attention matrix memory footprint ($N \times N$) for:
1. Standard patch size $P = 16$.
2. Fine-grained patch size $P = 8$.

**Solution:**
1. **Patch size $P = 16$:**
   $$N_{16} = \frac{512 \times 512}{16 \times 16} = 32 \times 32 = 1,024 \text{ patches}$$
   Attention matrix size: $N \times N = 1,024 \times 1,024 = 1,048,576 \text{ elements}$ ($1 \text{ M elements}$).

2. **Patch size $P = 8$:**
   $$N_8 = \frac{512 \times 512}{8 \times 8} = 64 \times 64 = 4,096 \text{ patches}$$
   Attention matrix size: $N \times N = 4,096 \times 4,096 = 16,777,216 \text{ elements}$ ($16.8 \text{ M elements}$).

**Conclusion:**
Halving the patch size from $16 \to 8$ increases the sequence length by **$4\times$**, and increases the self-attention memory and compute by **$16\times$** ($4^2$). This quadratic scaling motivated hierarchical Vision Transformers like Swin Transformer, which compute self-attention within shifted local windows.

---

### Illustration 2: Inductive Bias Comparison Across Vision Paradigms

**Problem:**
Synthesize the architectural properties, inductive biases, and trade-offs between CNNs, Vanilla ViT, and Swin Transformers.

| Feature / Property | ConvNet (ResNet / ConvNeXt) | Vanilla ViT (Dosovitskiy 2020) | Swin Transformer (Liu 2021) |
| :--- | :--- | :--- | :--- |
| **Tokenization** | None (Continuous feature map) | $16 \times 16$ Non-overlapping patches | $4 \times 4$ Patches + Hierarchical merging |
| **Feature Hierarchy** | Multi-scale pyramid ($4\times, 8\times, 16\times, 32\times$) | Single-scale columnar ($16\times$ throughout) | Multi-scale pyramid ($4\times, 8\times, 16\times, 32\times$) |
| **Self-Attention Scope** | None (Local kernel convolution) | **Global** across entire image | **Local** inside $M \times M$ shifted windows |
| **Complexity w.r.t Image Size** | Linear $O(H W)$ | Quadratic $O((HW)^2)$ | **Linear $O(HW \cdot M^2)$** |
| **Primary Advantage** | Efficient inference, small training data | Boundless scaling on massive data | Drop-in backbone for dense prediction |

---

## 7. Deep Learning Connection & Application

### 1. Vision Foundation Models
The ViT architecture is the computational bedrock of modern multimodal and computer vision foundation models:
- **CLIP (Contrastive Language-Image Pretraining):** Uses ViT-B/16 or ViT-L/14 as its visual encoder, projecting the `[CLS]` token into a shared embedding space with text.
- **MAE (Masked Autoencoders, He et al., 2021):** Masks $75\%$ of image patches and passes only the remaining $25\%$ visible patches through a ViT encoder, achieving self-supervised visual pre-training at unprecedented scale.
- **DINOv2 (Oquab et al., 2023):** Self-supervised ViT producing dense, general-purpose visual representations that work out-of-the-box for segmentation, depth estimation, and retrieval without fine-tuning.
- **Multimodal LLMs (GPT-4V, Gemini, Claude 3, LLaVA):** Use ViT encoders to convert visual inputs into sequences of visual tokens directly ingested by Transformer autoregressive decoders.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `PatchEmbedding`: Pure PyTorch module implementing patch flattening and linear projection via both `nn.Conv2d` and `torch.nn.functional.unfold`, proving mathematical equivalence.
2. `VisionTransformer`: Complete, clean implementation including `[CLS]` token prepending, learnable 1D position embeddings, Pre-LN Multi-Head Self-Attention, and MLP classification head.
3. Verification of the Part 5 Visual Grid hand arithmetic to machine precision.
4. Position embedding 2D bicubic interpolation test verifying resolution adaptation from $224 \times 224 \to 384 \times 384$.

See implementation in:
[`07_convolutional_networks/code/04_vision_transformers_and_patch_embeddings.py`](./code/04_vision_transformers_and_patch_embeddings.py)
