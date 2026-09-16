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

### 2.6 Deep Derivation 7.4.1: Mathematical Equivalence of Patch Partitioning and Strided 2D Convolution

In this derivation, we prove that dicing an image into non-overlapping patches and linearly projecting each patch to dimension $D$ is algebraically isomorphic to a standard 2D convolution with matching kernel and stride.

#### Step 1: Formulation of Patch Flattening and Linear Projection
Let an input image be represented as a 3D tensor $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ (in channel-first format).
Let the patch size be $P \times P$, where $P$ divides both $H$ and $W$.
The spatial grid of patches has dimensions:
$$H_P = \frac{H}{P}, \qquad W_P = \frac{W}{P}, \qquad N = H_P \cdot W_P$$

For each patch coordinate $(h_p, w_p) \in \{0, \dots, H_P - 1\} \times \{0, \dots, W_P - 1\}$, the spatial sub-array is:
$$\mathbf{P}_{(h_p, w_p)} \in \mathbb{R}^{C \times P \times P}, \quad \text{where } (\mathbf{P}_{(h_p, w_p)})_{c, u, v} = X_{c, \, h_p \cdot P + u, \, w_p \cdot P + v}$$
for $0 \le c < C$, $0 \le u < P$, $0 \le v < P$.

Flattening this sub-array into a 1D vector $\mathbf{x}_p^{(h_p, w_p)} \in \mathbb{R}^{C P^2}$ using the canonical row-major index bijection $\phi(c, u, v) = c P^2 + u P + v \in \{0, \dots, C P^2 - 1\}$:
$$(\mathbf{x}_p^{(h_p, w_p)})_{\phi(c, u, v)} = X_{c, \, h_p \cdot P + u, \, w_p \cdot P + v}$$

Given a learnable linear projection matrix $\mathbf{E} \in \mathbb{R}^{(C P^2) \times D}$ and bias $\mathbf{b} \in \mathbb{R}^D$, the embedded patch token $\mathbf{e}_{(h_p, w_p)} \in \mathbb{R}^D$ is:
$$(\mathbf{e}_{(h_p, w_p)})_d = \sum_{k=0}^{C P^2 - 1} (\mathbf{x}_p^{(h_p, w_p)})_k \, E_{k, d} + b_d = \sum_{c=0}^{C-1} \sum_{u=0}^{P-1} \sum_{v=0}^{P-1} X_{c, \, h_p \cdot P + u, \, w_p \cdot P + v} \, E_{\phi(c, u, v), \, d} + b_d$$

#### Step 2: Formulation of Strided 2D Convolution
Now consider a standard 2D convolutional layer with:
- Kernel tensor $\mathbf{K} \in \mathbb{R}^{D \times C \times P \times P}$, where $D$ is the number of output channels,
- Stride $s = P$ in both height and width,
- Zero padding $p = 0$,
- Bias vector $\mathbf{b} \in \mathbb{R}^D$.

The forward cross-correlation operation produces an output tensor $\mathbf{Y} \in \mathbb{R}^{D \times H_P \times W_P}$ whose entry at channel $d$ and spatial index $(h_p, w_p)$ is:
$$Y_{d, h_p, w_p} = \sum_{c=0}^{C-1} \sum_{u=0}^{P-1} \sum_{v=0}^{P-1} K_{d, c, u, v} \, X_{c, \, h_p \cdot P + u, \, w_p \cdot P + v} + b_d$$

#### Step 3: Bijective Parameter Identification
Define the parameter mapping $\Psi: \mathbb{R}^{(C P^2) \times D} \to \mathbb{R}^{D \times C \times P \times P}$ by:
$$K_{d, c, u, v} \equiv E_{\phi(c, u, v), \, d} = E_{c P^2 + u P + v, \, d}$$

Under this isomorphism, the summations in Step 1 and Step 2 are term-for-term identical for all $(d, h_p, w_p)$:
$$Y_{d, h_p, w_p} \equiv (\mathbf{e}_{(h_p, w_p)})_d$$

Therefore, flattening the spatial dimensions of $\mathbf{Y}$ from $(B, D, H_P, W_P)$ to $(B, H_P W_P, D) = (B, N, D)$ yields the exact same patch embedding sequence $\mathbf{E}_{\text{tokens}}$.
In PyTorch and modern hardware accelerators, `nn.Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)` is universally preferred over explicit image slicing and flattening, as it leverages heavily optimized Tensor Core GEMM kernels. $\blacksquare$

---

### 2.7 Deep Derivation 7.4.2: Self-Attention Inductive Bias and Permutation Equivariance Proof

In this derivation, we prove that the core mechanism of Vision Transformers (Multi-Head Self-Attention) possesses zero intrinsic awareness of 2D image topology and establish why position embeddings are strictly required.

#### Step 1: Definition of Permutation Equivariance
Let $\mathcal{S}_N$ be the symmetric group of all permutations on $N$ elements. Any permutation $\pi \in \mathcal{S}_N$ can be uniquely represented by an orthogonal permutation matrix $\mathbf{P} \in \{0, 1\}^{N \times N}$ where $P_{i, j} = 1$ if and only if $\pi(i) = j$.
A sequence transformation operator $\mathcal{T}: \mathbb{R}^{N \times D} \to \mathbb{R}^{N \times D}$ is **permutation equivariant** if:
$$\mathcal{T}(\mathbf{P} \mathbf{X}) = \mathbf{P} \, \mathcal{T}(\mathbf{X}), \quad \forall \mathbf{P} \in \mathcal{S}_N$$

#### Step 2: Proof of Permutation Equivariance for Scaled Dot-Product Attention
Let an input sequence be $\mathbf{X} \in \mathbb{R}^{N \times D}$.
Let the linear projections for Query, Key, and Value be $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times d}$.
Under permutation $\mathbf{P}$, the transformed inputs are $\mathbf{X}' = \mathbf{P} \mathbf{X}$.
The permuted projections satisfy:
$$\mathbf{Q}' = \mathbf{X}' \mathbf{W}_Q = \mathbf{P} \mathbf{X} \mathbf{W}_Q = \mathbf{P} \mathbf{Q}$$
$$\mathbf{K}' = \mathbf{X}' \mathbf{W}_K = \mathbf{P} \mathbf{X} \mathbf{W}_K = \mathbf{P} \mathbf{K}$$
$$\mathbf{V}' = \mathbf{X}' \mathbf{W}_V = \mathbf{P} \mathbf{X} \mathbf{W}_V = \mathbf{P} \mathbf{V}$$

Now compute the pre-softmax attention score matrix:
$$\mathbf{S}' = \frac{\mathbf{Q}' (\mathbf{K}')^T}{\sqrt{d}} = \frac{(\mathbf{P} \mathbf{Q}) (\mathbf{P} \mathbf{K})^T}{\sqrt{d}} = \frac{\mathbf{P} \mathbf{Q} \mathbf{K}^T \mathbf{P}^T}{\sqrt{d}} = \mathbf{P} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} \right) \mathbf{P}^T = \mathbf{P} \mathbf{S} \mathbf{P}^T$$

Applying the row-wise softmax operator $\operatorname{softmax}(\cdot)$:
Since $\mathbf{P}$ permutes rows and $\mathbf{P}^T$ permutes columns, the scalar exponential $\exp(S'_{i, j}) = \exp(S_{\pi(i), \pi(j)})$. The row denominator is:
$$\sum_{j=1}^N \exp(S'_{i, j}) = \sum_{j=1}^N \exp(S_{\pi(i), \pi(j)}) = \sum_{k=1}^N \exp(S_{\pi(i), k})$$
Therefore:
$$\mathbf{A}' = \operatorname{softmax}(\mathbf{S}') = \mathbf{P} \, \operatorname{softmax}(\mathbf{S}) \, \mathbf{P}^T = \mathbf{P} \mathbf{A} \mathbf{P}^T$$

Now compute the output context:
$$\mathbf{A}' \mathbf{V}' = (\mathbf{P} \mathbf{A} \mathbf{P}^T) (\mathbf{P} \mathbf{V})$$
Because $\mathbf{P}$ is an orthogonal matrix ($\mathbf{P}^T \mathbf{P} = \mathbf{I}$):
$$\mathbf{A}' \mathbf{V}' = \mathbf{P} \mathbf{A} (\mathbf{P}^T \mathbf{P}) \mathbf{V} = \mathbf{P} (\mathbf{A} \mathbf{V})$$

Finally, applying the output projection $\mathbf{W}_O$:
$$\text{Attention}(\mathbf{X}') \mathbf{W}_O = \mathbf{P} (\mathbf{A} \mathbf{V}) \mathbf{W}_O = \mathbf{P} \left( \text{Attention}(\mathbf{X}) \mathbf{W}_O \right)$$

This completes the proof: **Self-Attention is strictly permutation equivariant**.
It treats the sequence of patches as an unordered bag of nodes on a complete graph $K_N$. It cannot distinguish an upright face from a jumbled collage where the nose, eyes, and mouth patches are swapped at random.

#### Step 3: Symmetry Breaking via Position Embeddings
To inject geometric inductive bias, ViT adds position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{N \times D}$:
$$\mathbf{Z} = \mathbf{X} + \mathbf{E}_{\text{pos}}$$

If the input is permuted while retaining the positional coordinate assignment:
$$\mathcal{T}(\mathbf{P}\mathbf{X} + \mathbf{E}_{\text{pos}}) \neq \mathbf{P} \, \mathcal{T}(\mathbf{X} + \mathbf{E}_{\text{pos}})$$
Because $\mathbf{E}_{\text{pos}}$ is anchored to spatial indices, the permutation symmetry is broken, allowing the model to condition attention on spatial adjacency and distance. $\blacksquare$

---

### 2.8 Deep Derivation 7.4.3: Computational Complexity and FLOP Scaling: ViT vs. ConvNet as a Function of Resolution

In this derivation, we derive the exact Multiply-Accumulate (MAC) and Floating Point Operations (FLOPs) for a Vision Transformer layer and compute the asymptotic crossover point where ViT exceeds CNN computational cost.

#### Step 1: Detailed FLOP Breakdown of a ViT Encoder Layer
Let the input sequence have length $N = \frac{HW}{P^2}$ (omitting the single `[CLS]` token for asymptotic analysis) and hidden dimension $D$.
Each Transformer layer consists of:

1. **Layer Normalization (LN):**
   - Mean and variance: $2ND$ FLOPs; Normalization and affine scale/shift: $2ND$ FLOPs. Total: $4ND$ FLOPs (negligible).
2. **QKV Linear Projections:**
   - Multiplies $\mathbf{X} \in \mathbb{R}^{N \times D}$ by three weight matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}$:
   - MACs: $3 \times (N \cdot D \cdot D) = 3 N D^2$
   - FLOPs ($1 \text{ MAC} = 2 \text{ FLOPs}$): $6 N D^2$
3. **Scaled Dot-Product Attention:**
   - **Score computation:** $\mathbf{Q} \mathbf{K}^T$ where $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{N \times D}$:
     - MACs: $N \times D \times N = N^2 D$
     - FLOPs: $2 N^2 D$
   - **Softmax:** Row-wise exponentiation and normalization: $3 N^2$ FLOPs.
   - **Context aggregation:** $\mathbf{A} \mathbf{V}$ where $\mathbf{A} \in \mathbb{R}^{N \times N}, \mathbf{V} \in \mathbb{R}^{N \times D}$:
     - MACs: $N \times N \times D = N^2 D$
     - FLOPs: $2 N^2 D$
4. **Attention Output Projection ($\mathbf{W}_O$):**
   - Multiplies context by $\mathbf{W}_O \in \mathbb{R}^{D \times D}$:
   - MACs: $N \cdot D \cdot D = N D^2$
   - FLOPs: $2 N D^2$
5. **Multi-Layer Perceptron (MLP):**
   - Two linear transformations with expansion factor $4$ ($D \to 4D \to D$):
   - Layer 1 ($\mathbf{W}_1 \in \mathbb{R}^{D \times 4D}$): $N \times D \times 4D = 4 N D^2$ MACs ($8 N D^2$ FLOPs).
   - Layer 2 ($\mathbf{W}_2 \in \mathbb{R}^{4D \times D}$): $N \times 4D \times D = 4 N D^2$ MACs ($8 N D^2$ FLOPs).
   - Total MLP: $8 N D^2$ MACs ($16 N D^2$ FLOPs).

#### Step 2: Total ViT Layer Complexity
Summing all terms:
$$\text{MACs}_{\text{ViT}} = \underbrace{4 N D^2}_{\text{MSA Projections}} + \underbrace{2 N^2 D}_{\text{Attention Maps}} + \underbrace{8 N D^2}_{\text{MLP}} = 12 N D^2 + 2 N^2 D$$
$$\text{FLOPs}_{\text{ViT}} = 24 N D^2 + 4 N^2 D$$

Substituting $N = \frac{HW}{P^2}$:
$$\text{FLOPs}_{\text{ViT}}(H, W) = \underbrace{24 \left( \frac{D^2}{P^2} \right) HW}_{\text{Linear in Pixels } (HW)} + \underbrace{4 \left( \frac{D}{P^4} \right) (HW)^2}_{\text{Quadratic in Pixels } (HW)^2}$$

#### Step 3: Complexity of a Convolutional Layer (e.g., ResNet)
A standard $K \times K$ convolutional layer mapping $C \to C$ channels over spatial size $H \times W$ requires:
$$\text{FLOPs}_{\text{CNN}}(H, W) = 2 \cdot K^2 \cdot C^2 \cdot H W$$
This cost is **strictly linear** in the number of pixels $HW$ for all resolutions.

#### Step 4: The High-Resolution Quadratic Crossover Point
Equating the quadratic attention term to the linear projection terms:
$$4 \frac{D}{P^4} (HW)^2 \gg 24 \frac{D^2}{P^2} HW \iff HW \gg 6 P^2 D$$

For ViT-Base ($P = 16, D = 768$):
$$HW \gg 6 \times 256 \times 768 = 1,179,648 \text{ pixels} \approx 1.18 \text{ Megapixels} \approx (1086 \times 1086)$$

- At standard ImageNet resolution ($224 \times 224 \approx 50,000$ pixels), $N = 196$. The linear term $24 N D^2$ dominates ($96\%$ of compute).
- At high resolution for object detection or segmentation ($1024 \times 1024 \approx 1.05 \text{M}$ pixels), $N = 4,096$. The quadratic term $4 N^2 D$ surges to $67.1 \times 10^9$ FLOPs per layer.
- At $2048 \times 2048$ ($4.19\text{M}$ pixels), $N = 16,384$, requiring over $1.07 \text{ TFLOPs}$ per single attention layer!
This quadratic barrier mathematically proves why Vanilla ViT cannot be directly applied to dense high-resolution vision tasks without local windowing (Swin Transformer) or multi-scale feature pyramids. $\blacksquare$

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

### Illustration 3: Complete Hand Trace of Single-Head Self-Attention on Patch Tokens

**Problem:**
Consider a sequence of $N = 3$ token vectors in $\mathbb{R}^2$ entering a single-head self-attention module ($D = 2, d_k = 2$):
$$\mathbf{Z} = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}$$
The learnable projection matrices are:
$$\mathbf{W}_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{W}_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad \mathbf{W}_V = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$
1. Compute the Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) matrices.
2. Compute the scaled pre-softmax score matrix $\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}$.
3. Compute the attention probability matrix $\mathbf{A} = \operatorname{softmax}(\mathbf{S})$ row-by-row with explicit exponentials.
4. Compute the final attention output context $\mathbf{O} = \mathbf{A} \mathbf{V}$.

**Solution:**

#### Step 1: Query, Key, Value Projections
$$\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}$$
$$\mathbf{K} = \mathbf{Z} \mathbf{W}_K = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 2 & 0 \\ 1 & 1 \end{bmatrix}$$
$$\mathbf{V} = \mathbf{Z} \mathbf{W}_V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 2 & 1 \end{bmatrix}$$

#### Step 2: Scaled Dot-Product Scores
Compute $\mathbf{Q} \mathbf{K}^T$:
- Row 1:
  - $S_{1, 1} = [1, 0] \cdot [0, 1] = 0$
  - $S_{1, 2} = [1, 0] \cdot [2, 0] = 2$
  - $S_{1, 3} = [1, 0] \cdot [1, 1] = 1$
- Row 2:
  - $S_{2, 1} = [0, 2] \cdot [0, 1] = 2$
  - $S_{2, 2} = [0, 2] \cdot [2, 0] = 0$
  - $S_{2, 3} = [0, 2] \cdot [1, 1] = 2$
- Row 3:
  - $S_{3, 1} = [1, 1] \cdot [0, 1] = 1$
  - $S_{3, 2} = [1, 1] \cdot [2, 0] = 2$
  - $S_{3, 3} = [1, 1] \cdot [1, 1] = 2$

$$\mathbf{Q} \mathbf{K}^T = \begin{bmatrix} 0 & 2 & 1 \\ 2 & 0 & 2 \\ 1 & 2 & 2 \end{bmatrix}$$

Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.4142$:
$$\mathbf{S} = \frac{1}{\sqrt{2}} \begin{bmatrix} 0 & 2 & 1 \\ 2 & 0 & 2 \\ 1 & 2 & 2 \end{bmatrix} \approx \begin{bmatrix} 0.000 & 1.414 & 0.707 \\ 1.414 & 0.000 & 1.414 \\ 0.707 & 1.414 & 1.414 \end{bmatrix}$$

#### Step 3: Row-by-Row Softmax Computation
- **Row 1:** $e^0 = 1.000$, $e^{1.4142} \approx 4.113$, $e^{0.7071} \approx 2.028$.
  Denominator: $\Sigma_1 = 1.000 + 4.113 + 2.028 = 7.141$.
  $$A_{1, 1} = \frac{1.000}{7.141} \approx \mathbf{0.140}, \quad A_{1, 2} = \frac{4.113}{7.141} \approx \mathbf{0.576}, \quad A_{1, 3} = \frac{2.028}{7.141} \approx \mathbf{0.284}$$

- **Row 2:** $e^{1.4142} \approx 4.113$, $e^0 = 1.000$, $e^{1.4142} \approx 4.113$.
  Denominator: $\Sigma_2 = 4.113 + 1.000 + 4.113 = 9.226$.
  $$A_{2, 1} = \frac{4.113}{9.226} \approx \mathbf{0.446}, \quad A_{2, 2} = \frac{1.000}{9.226} \approx \mathbf{0.108}, \quad A_{2, 3} = \frac{4.113}{9.226} \approx \mathbf{0.446}$$

- **Row 3:** $e^{0.7071} \approx 2.028$, $e^{1.4142} \approx 4.113$, $e^{1.4142} \approx 4.113$.
  Denominator: $\Sigma_3 = 2.028 + 4.113 + 4.113 = 10.254$.
  $$A_{3, 1} = \frac{2.028}{10.254} \approx \mathbf{0.198}, \quad A_{3, 2} = \frac{4.113}{10.254} \approx \mathbf{0.401}, \quad A_{3, 3} = \frac{4.113}{10.254} \approx \mathbf{0.401}$$

$$\mathbf{A} \approx \begin{bmatrix}
0.140 & 0.576 & 0.284 \\
0.446 & 0.108 & 0.446 \\
0.198 & 0.401 & 0.401
\end{bmatrix}$$

#### Step 4: Output Context $\mathbf{O} = \mathbf{A} \mathbf{V}$
Using $\mathbf{V} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 2 & 1 \end{bmatrix}$:
- Row 1:
  $$O_{1, 1} = 0.140(2) + 0.576(0) + 0.284(2) = 0.280 + 0.568 = \mathbf{0.848}$$
  $$O_{1, 2} = 0.140(0) + 0.576(2) + 0.284(1) = 1.152 + 0.284 = \mathbf{1.436}$$
- Row 2:
  $$O_{2, 1} = 0.446(2) + 0.108(0) + 0.446(2) = 0.892 + 0.892 = \mathbf{1.784}$$
  $$O_{2, 2} = 0.446(0) + 0.108(2) + 0.446(1) = 0.216 + 0.446 = \mathbf{0.662}$$
- Row 3:
  $$O_{3, 1} = 0.198(2) + 0.401(0) + 0.401(2) = 0.396 + 0.802 = \mathbf{1.198}$$
  $$O_{3, 2} = 0.198(0) + 0.401(2) + 0.401(1) = 0.802 + 0.401 = \mathbf{1.203}$$

$$\mathbf{O} \approx \begin{bmatrix} 0.848 & 1.436 \\ 1.784 & 0.662 \\ 1.198 & 1.203 \end{bmatrix}$$

---

### Illustration 4: ViT-Base/16 ImageNet FLOP and Parameter Calculation

**Problem:**
Calculate the exact parameter count and total inference FLOPs for the canonical **ViT-Base/16** model processing a standard ImageNet RGB image $\mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$ for a 1,000-class classification head.
Architecture hyper-parameters:
- Patch size $P = 16$
- Hidden dimension $D = 768$
- Number of layers $L = 12$
- Number of attention heads $h = 12$ ($d_k = D / h = 64$)
- MLP expansion dimension $D_{\text{mlp}} = 4D = 3,072$

**Solution:**

#### Step 1: Sequence Length and Patch Embedding
- Patches along spatial grid: $H_P = 224 / 16 = 14$, $W_P = 224 / 16 = 14$.
- Patch count: $N = 14 \times 14 = 196$.
- Total sequence length with `[CLS]` token: $N_{\text{seq}} = N + 1 = 197$.
- **Patch Embedding Parameters:**
  $$\text{Params}_{\text{patch}} = (P^2 \cdot C \cdot D) + D = (16 \times 16 \times 3 \times 768) + 768 = 589,824 + 768 = \mathbf{590,592}$$
- **Embeddings:**
  - Learnable `[CLS]` token: $1 \times 768 = \mathbf{768}$
  - Learnable 1D Position Embeddings: $N_{\text{seq}} \times D = 197 \times 768 = \mathbf{151,296}$

#### Step 2: Transformer Encoder Layer Parameters
For a single encoder block:
1. **Multi-Head Self-Attention (MSA):**
   - $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$: Each has $(D \times D) + D = (768 \times 768) + 768 = 590,592$ parameters.
   - Total MSA params: $4 \times 590,592 = \mathbf{2,362,368}$.
2. **LayerNorm 1 & 2:**
   - Each LN has scale $\gamma$ and shift $\beta$: $2 \times 768 = 1,536$ parameters.
   - Two LNs: $2 \times 1,536 = \mathbf{3,072}$.
3. **MLP Block:**
   - Linear 1 ($D \to 4D$): $(768 \times 3,072) + 3,072 = 2,359,296 + 3,072 = 2,362,368$.
   - Linear 2 ($4D \to D$): $(3,072 \times 768) + 768 = 2,359,296 + 768 = 2,360,064$.
   - Total MLP params: $2,362,368 + 2,360,064 = \mathbf{4,722,432}$.

**Per-Block Parameters:**
$$\text{Params}_{\text{block}} = 2,362,368 + 3,072 + 4,722,432 = \mathbf{7,087,872}$$
**All 12 Blocks:**
$$\text{Params}_{12 \text{ blocks}} = 12 \times 7,087,872 = \mathbf{85,054,464}$$

#### Step 3: Classification Head Parameters
- Linear head ($768 \to 1,000$):
  $$\text{Params}_{\text{head}} = (768 \times 1,000) + 1,000 = \mathbf{769,000}$$
- Final LayerNorm: $2 \times 768 = \mathbf{1,536}$.

**Total Model Parameters:**
$$\text{Params}_{\text{total}} = 590,592 + 768 + 151,296 + 85,054,464 + 1,536 + 769,000 = \mathbf{86,567,656} \approx \mathbf{86.57 \text{ M}}$$

#### Step 4: Inference FLOP Breakdown
For $N_{\text{seq}} = 197$:
1. **Patch Embedding:** $2 \times (196 \times 768 \times 768 / \dots) = 2 \times 196 \times (16^2 \times 3) \times 768 \approx \mathbf{0.23 \text{ GFLOPs}}$.
2. **Per-Block FLOPs:**
   - MSA QKV + Output Projections: $2 \times (4 \times N_{\text{seq}} \times D^2) = 8 \times 197 \times 768^2 \approx \mathbf{929.6 \text{ MFLOPs}}$.
   - Attention Maps ($\mathbf{Q}\mathbf{K}^T + \mathbf{A}\mathbf{V}$): $2 \times (2 \times N_{\text{seq}}^2 \times D) = 4 \times 197^2 \times 768 \approx \mathbf{119.2 \text{ MFLOPs}}$.
   - MLP ($D \to 4D \to D$): $2 \times (2 \times N_{\text{seq}} \times D \times 4D) = 16 \times 197 \times 768^2 \approx \mathbf{1,859.3 \text{ MFLOPs}}$.
   - Block Total: $929.6 + 119.2 + 1,859.3 = \mathbf{2,908.1 \text{ MFLOPs}} \approx \mathbf{2.91 \text{ GFLOPs}}$.
3. **Total for 12 Blocks:**
   $$12 \times 2.9081 \text{ GFLOPs} \approx \mathbf{34.90 \text{ GFLOPs}}$$
4. **Overall Model FLOPs:**
   $$\text{FLOPs}_{\text{total}} \approx 0.23 + 34.90 + \text{Head}(0.0015) \approx \mathbf{35.13 \text{ GFLOPs}} \quad (\approx \mathbf{17.57 \text{ GMACs}})$$

---

### Illustration 5: 2D Position Embedding Interpolation ($2 \times 2 \to 3 \times 3$ Grid)

**Problem:**
A small ViT is pre-trained with $N = 4$ patches on a $2 \times 2$ grid. The learned 1D spatial position embeddings for hidden dimension $D = 1$ are:
$$\mathbf{E}_{\text{spatial}} = \begin{bmatrix} 1.0 \\ 3.0 \\ 2.0 \\ 4.0 \end{bmatrix}$$
corresponding to coordinates $(0, 0), (0, 1), (1, 0), (1, 1)$.
We wish to fine-tune this model on an image yielding $N' = 9$ patches ($3 \times 3$ grid). Trace the bilinear interpolation arithmetic to compute the new $3 \times 3$ interpolated position embedding matrix $\mathbf{E}'_{\text{spatial}} \in \mathbb{R}^{3 \times 3}$.

**Solution:**

#### Step 1: Reshape to 2D Spatial Grid
$$\mathbf{G} = \begin{bmatrix} 1.0 & 3.0 \\ 2.0 & 4.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$
where rows correspond to $y \in \{0, 1\}$ and columns to $x \in \{0, 1\}$.

#### Step 2: Coordinate Mapping
Let the target $3 \times 3$ grid coordinates be indexed by $(i, j) \in \{0, 1, 2\} \times \{0, 1, 2\}$.
Under continuous alignment without corner shift, the normalized continuous coordinates in the source $2 \times 2$ grid are:
$$y_i = i \cdot \frac{2 - 1}{3 - 1} = i \cdot 0.5 \in \{0.0, 0.5, 1.0\}$$
$$x_j = j \cdot \frac{2 - 1}{3 - 1} = j \cdot 0.5 \in \{0.0, 0.5, 1.0\}$$

#### Step 3: Bilinear Interpolation Equation
For any query point $(y, x)$ with $y \in [0, 1]$ and $x \in [0, 1]$:
$$G'(y, x) = (1 - y)(1 - x) G_{0, 0} + (1 - y) x G_{0, 1} + y (1 - x) G_{1, 0} + y x G_{1, 1}$$

Evaluating all $9$ target points:
1. $(0.0, 0.0)$: $G_{0, 0} = \mathbf{1.0}$
2. $(0.0, 0.5)$: $0.5(1.0) + 0.5(3.0) = \mathbf{2.0}$
3. $(0.0, 1.0)$: $G_{0, 1} = \mathbf{3.0}$
4. $(0.5, 0.0)$: $0.5(1.0) + 0.5(2.0) = \mathbf{1.5}$
5. $(0.5, 0.5)$: $0.25(1.0 + 3.0 + 2.0 + 4.0) = 0.25(10.0) = \mathbf{2.5}$
6. $(0.5, 1.0)$: $0.5(3.0) + 0.5(4.0) = \mathbf{3.5}$
7. $(1.0, 0.0)$: $G_{1, 0} = \mathbf{2.0}$
8. $(1.0, 0.5)$: $0.5(2.0) + 0.5(4.0) = \mathbf{3.0}$
9. $(1.0, 1.0)$: $G_{1, 1} = \mathbf{4.0}$

#### Step 4: Target 2D Grid and Flattened Sequence
$$\mathbf{G}' = \begin{bmatrix}
1.0 & 2.0 & 3.0 \\
1.5 & 2.5 & 3.5 \\
2.0 & 3.0 & 4.0
\end{bmatrix}$$

Flattening $\mathbf{G}'$ back into a 1D sequence for the 9 patch tokens:
$$\mathbf{E}'_{\text{spatial}} = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 1.5 & 2.5 & 3.5 & 2.0 & 3.0 & 4.0 \end{bmatrix}^T \in \mathbb{R}^{9 \times 1}$$
Prepending the un-interpolated `[CLS]` position embedding completes the adapted position embedding matrix for the higher-resolution input.

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
