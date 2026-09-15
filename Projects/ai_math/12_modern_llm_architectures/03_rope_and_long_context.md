# Chapter 03: Advanced Positional Encodings: RoPE, YaRN & NTK-Aware Scaling

---

## 1. Intuition & 101 Motivation

The standard Transformer self-attention operation is fundamentally **permutation-equivariant**:
$$\operatorname{Attention}(\mathbf{P} X) = \mathbf{P} \operatorname{Attention}(X)$$
where $\mathbf{P}$ is any permutation matrix. If we shuffle the words in a prompt, self-attention computes the exact same attention weights on the permuted tokens. Natural language, however, is deeply sensitive to word order: *"Dog bites man"* carries a completely different meaning than *"Man bites dog"*.

To inject word order, early architectures used:
1. **Absolute Sinusoidal Encodings (Vaswani et al., 2017):** Added directly to the word embeddings: $x_t \leftarrow x_t + p_t$.
2. **Learned Absolute Encodings (GPT-2, BERT):** A trainable embedding lookup table.

Both absolute approaches suffer from a fatal flaw: **Inability to Generalize Beyond Training Length**. If a model is trained on sequences of length $L = 2048$, position embedding $p_{2049}$ is completely untrained. When tested on longer documents, the model's perplexity catastrophically explodes.

Enter **Rotary Position Embedding (RoPE)** (Jianlin Su et al., 2021; RoFormer):
- Instead of *adding* positional vectors to input embeddings, RoPE **rotates** the Query and Key vectors in high-dimensional 2D orthogonal planes.
- Crucially, the inner product between a rotated query at position $m$ and a rotated key at position $n$ depends **strictly on their relative distance $(m - n)$**!
- Today, RoPE is the undisputed gold standard used by virtually every premier foundation model: **LLaMA 1/2/3, Mistral, Gemma, Qwen, and DeepSeek**.

Furthermore, through modern frequency scaling techniques—such as **NTK-Aware Scaling** and **YaRN**—RoPE allows a model trained on 4,000 tokens to seamlessly extend its context window to **128,000+ tokens**!

---

## 2. Rigorous Mathematical Formulation

### 2.1 The RoPE Invariance Criterion

Let $q_m \in \mathbb{R}^d$ be a query vector at sequence position $m$, and $k_n \in \mathbb{R}^d$ be a key vector at sequence position $n$. We seek a transformation function $\mathbf{R}(x, m)$ such that their inner product is a function *only* of the vectors themselves and their relative positional offset $(m - n)$:

$$\langle \mathbf{R}(q, m), \; \mathbf{R}(k, n) \rangle = g(q, k, m - n)$$

---

### 2.2 The 2D Complex Plane Solution

To see how rotation solves this, consider a 2-dimensional vector represented as a complex number $z = x_1 + i x_2 \in \mathbb{C}$.

Rotation by an angle $m \theta$ corresponds to multiplication by the unit complex exponential:
$$\mathbf{R}(z, m) = z \cdot e^{i m \theta}$$

Now compute the complex Hermitian inner product of query $q$ at position $m$ and key $k$ at position $n$:
$$\langle \mathbf{R}(q, m), \mathbf{R}(k, n) \rangle = \operatorname{Re}\left[ (q e^{i m \theta}) (k e^{i n \theta})^* \right]$$
$$= \operatorname{Re}\left[ q k^* e^{i m \theta} e^{-i n \theta} \right] = \operatorname{Re}\left[ q k^* e^{i (m - n) \theta} \right]$$

The absolute positions $m$ and $n$ completely vanish! The inner product depends strictly on the relative distance $\Delta = m - n$.

---

### 2.3 General $d$-Dimensional Rotary Matrix

For a $d$-dimensional head embedding vector (where $d$ is even), RoPE partitions the vector into $d/2$ orthogonal 2D subspaces. Each subspace $i \in \{0, 1, \dots, d/2 - 1\}$ is assigned a distinct base frequency $\theta_i$:

$$\theta_i = b^{-2i / d}, \quad \text{where } b = 10,000 \text{ (standard) or } 500,000 \text{ (LLaMA 3)}$$

The full rotation operator is represented as a block-diagonal orthogonal matrix $\mathbf{R}_{\Theta, m}^d$:

$$\mathbf{R}_{\Theta, m}^d = \begin{bmatrix}
\cos(m \theta_0) & -\sin(m \theta_0) & 0 & 0 & \dots & 0 & 0 \\
\sin(m \theta_0) & \cos(m \theta_0) & 0 & 0 & \dots & 0 & 0 \\
0 & 0 & \cos(m \theta_1) & -\sin(m \theta_1) & \dots & 0 & 0 \\
0 & 0 & \sin(m \theta_1) & \cos(m \theta_1) & \dots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \dots & \cos(m \theta_{d/2-1}) & -\sin(m \theta_{d/2-1}) \\
0 & 0 & 0 & 0 & \dots & \sin(m \theta_{d/2-1}) & \cos(m \theta_{d/2-1})
\end{bmatrix}$$

#### Computational Implementation (Fast Element-Wise Formulation):
In practice, one never instantiates the sparse matrix $\mathbf{R}$. Instead, the rotation is computed using vectorized Hadamard products:

$$\mathbf{R}_{\Theta, m}^d x = x \odot \cos(m \Theta) + \tilde{x} \odot \sin(m \Theta)$$

where $\tilde{x} = [-x_2, x_1, -x_4, x_3, \dots, -x_d, x_{d-1}]^T$ is the vector obtained by swapping adjacent pairs and negating the odd indices.

> **Crucial Implementation Rule:** RoPE is applied **ONLY to the Query and Key vectors** ($q$ and $k$). It is **NEVER applied to Value vectors ($v$)**! Values represent semantic content and must not be rotated.

---

### 2.4 Context Window Extension: NTK-Aware & YaRN

When a model pre-trained on length $L$ (e.g., 4096) is prompted with a longer sequence $L' = s \cdot L$ (e.g., 16384, scale $s = 4$):

#### 1. Linear Position Interpolation (PI):
Compress positions uniformly: $m' = m / s$.
- **Flaw:** High-frequency dimensions (small $i$) oscillate very fast; compressing them destroys local grammar and token-level order.

#### 2. NTK-Aware Scaling (Neural Tangent Kernel):
Instead of scaling positions $m$, scale the base frequency $b$:
$$b' = b \cdot s^{d / (d - 2)}$$
- Low frequencies (large $i$) are stretched heavily to handle long-distance context.
- High frequencies (small $i$) remain almost unchanged, perfectly preserving short-range token resolution!

#### 3. YaRN (Yet another RoPE extensioN, Peng et al., 2023):
Divides dimensions into three distinct regimes:
- **High-frequency bands ($\lambda_i < r_{\text{low}}$):** Zero interpolation (pure extrapolation).
- **Low-frequency bands ($\lambda_i > r_{\text{high}}$):** Full linear interpolation.
- **Medium-frequency bands:** Smoothly blended using a ramp function $\alpha_i \in [0, 1]$.
- **Temperature Scaling:** Multiplies attention logits by $1/t$ where $\sqrt{t} \approx 0.1 \ln s + 1$, preventing the attention entropy from collapsing as context length scales to $128\text{k}+$.

---

## 3. Geometric & Physical Interpretation

### 3.1 The Multiscale Clockwork Mechanism
Think of the $d/2$ dimensions of RoPE as hands on a multiscale clock:
```
Subspace i=0 (High Frequency):
   Rotates 1 full turn every 6 tokens!  <---> Like the Second Hand (Detects immediate adjacent words: "New" + "York")

Subspace i=d/4 (Medium Frequency):
   Rotates 1 full turn every 200 tokens! <---> Like the Minute Hand (Detects sentence and paragraph structure)

Subspace i=d/2-1 (Low Frequency):
   Rotates 1 full turn every 10,000 tokens! <---> Like the Hour Hand (Detects overall document theme & topic)
```
Because the relative dot product combines all $d/2$ frequencies:
$$\langle q_m, k_n \rangle = \sum_{i=0}^{d/2-1} \Big( (q_{2i} k_{2i} + q_{2i+1} k_{2i+1}) \cos((m - n)\theta_i) + (q_{2i} k_{2i+1} - q_{2i+1} k_{2i}) \sin((m - n)\theta_i) \Big)$$
attention naturally possesses **multiscale temporal resolution**: sharp local decay for adjacent tokens combined with long-wavelength persistence for global themes!

---

## 4. Real-World Analogy: Spiral Notebook Binding

Imagine binding a massive manuscript into a spiral notebook:
- You don't write absolute page numbers at the bottom of the page.
- Instead, each page is punched along a spiral coil.
- The relative angle between page 5 and page 8 along the wire coil is determined entirely by the pitch of the spiral multiplied by their separation ($\Delta = 3$ loops).
- Any reader holding the coil can instantly gauge how far apart two sections are simply by measuring the angular twist between them!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete 2D RoPE calculation by hand and verify the relative dot-product equivalence down to machine precision.

---

### 5.1 System & Parameter Setup
- **Subspace Dimension:** $d = 2$
- **Base Angle:** $\theta = 0.500000$ radians
- **Query at Position $m = 1$:**
  $$q = \begin{bmatrix} 1.000000 \\ 0.000000 \end{bmatrix}$$
- **Key at Position $n = 3$:**
  $$k = \begin{bmatrix} 0.000000 \\ 2.000000 \end{bmatrix}$$
- **Relative Distance:** $\Delta = m - n = 1 - 3 = \mathbf{-2}$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $m, n$ | `pos_q, pos_k` | Integer sequence positions ($m = 1, n = 3$) |
| $\theta_m, \theta_n$ | `rot_angle_q, rot_angle_k` | Rotation angles: $m \cdot \theta = 0.5$ rad, $n \cdot \theta = 1.5$ rad |
| $\mathbf{R}_m, \mathbf{R}_n$ | `rot_mat_q, rot_mat_k` | $2 \times 2$ orthogonal Givens rotation matrices |
| $q_{\text{rot}}, k_{\text{rot}}$ | `q_rotated, k_rotated` | Spatially rotated query and key vectors |
| $\text{Score}_{\text{rot}}$ | `dot_product` | Attention logit: $\langle q_{\text{rot}}, k_{\text{rot}} \rangle$ |
| $\text{Score}_{\text{rel}}$ | `rel_formula` | Direct evaluation via relative distance $(m - n) = -2$ |

---

### 5.3 Step-by-Step Hand Calculations: Rotations & Dot Product

#### Step 1: Rotate Query $q$ at Position $m = 1$
Angle: $\theta_m = 1 \times 0.500000 = \mathbf{0.500000}$ rad.
$$\cos(0.500000) \approx \mathbf{0.877583}$$
$$\sin(0.500000) \approx \mathbf{0.479426}$$

The 2D rotation matrix is:
$$\mathbf{R}_1 = \begin{bmatrix} \cos(0.5) & -\sin(0.5) \\ \sin(0.5) & \cos(0.5) \end{bmatrix} = \begin{bmatrix} 0.877583 & -0.479426 \\ 0.479426 & 0.877583 \end{bmatrix}$$

Apply to $q = [1.0, 0.0]^T$:
$$q_{\text{rot}} = \mathbf{R}_1 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.877583 \times 1.0 - 0.479426 \times 0.0 \\ 0.479426 \times 1.0 + 0.877583 \times 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.877583} \\ \mathbf{0.479426} \end{bmatrix}$$

---

#### Step 2: Rotate Key $k$ at Position $n = 3$
Angle: $\theta_n = 3 \times 0.500000 = \mathbf{1.500000}$ rad.
$$\cos(1.500000) \approx \mathbf{0.070737}$$
$$\sin(1.500000) \approx \mathbf{0.997495}$$

The 2D rotation matrix is:
$$\mathbf{R}_3 = \begin{bmatrix} \cos(1.5) & -\sin(1.5) \\ \sin(1.5) & \cos(1.5) \end{bmatrix} = \begin{bmatrix} 0.070737 & -0.997495 \\ 0.997495 & 0.070737 \end{bmatrix}$$

Apply to $k = [0.0, 2.0]^T$:
$$k_{\text{rot}} = \mathbf{R}_3 \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 0.070737 \times 0.0 - 0.997495 \times 2.0 \\ 0.997495 \times 0.0 + 0.070737 \times 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.994990} \\ \mathbf{0.141474} \end{bmatrix}$$

---

#### Step 3: Compute Dot Product of Rotated Vectors
$$\langle q_{\text{rot}}, k_{\text{rot}} \rangle = (0.877583) \times (-1.994990) + (0.479426) \times (0.141474)$$
$$= -1.750769 + 0.067827 = \mathbf{-1.682942}$$

---

#### Step 4: Verification via Relative Angle Formula ($\Delta = -2$)
Now let us compute the dot product *without ever knowing absolute positions $m$ and $n$*, using strictly the relative offset $\Delta = m - n = -2$:

$$\Delta \theta = -2 \times 0.500000 = \mathbf{-1.000000} \text{ rad}$$
$$\cos(-1.000000) = \cos(1.000000) \approx \mathbf{0.540302}$$
$$\sin(-1.000000) = -\sin(1.000000) \approx \mathbf{-0.841471}$$

Recall the relative identity:
$$\langle q_{\text{rot}}, k_{\text{rot}} \rangle = (q_1 k_1 + q_2 k_2) \cos(\Delta \theta) + (q_1 k_2 - q_2 k_1) \sin(\Delta \theta)$$

1. Symmetric term:
   $$q_1 k_1 + q_2 k_2 = (1.0 \times 0.0) + (0.0 \times 2.0) = \mathbf{0.0}$$
2. Skew-symmetric term:
   $$q_1 k_2 - q_2 k_1 = (1.0 \times 2.0) - (0.0 \times 0.0) = \mathbf{2.0}$$
3. Relative dot product:
   $$\text{Score}_{\text{rel}} = 0.0 \times 0.540302 + 2.0 \times (-0.841471) = \mathbf{-1.682942}$$

$$\mathbf{\langle q_{\text{rot}}, k_{\text{rot}} \rangle \equiv \text{Score}_{\text{rel}} = -1.682942} \quad \text{(Exact Match!)}$$

---

### 5.4 Summary Visual Grid: RoPE Rotation & Equivalence Ledger

| Vector | Original | Position | Rotation Angle | $\cos(\theta)$ | $\sin(\theta)$ | Rotated Vector | Dot Product |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Query $q$** | $[1.0, 0.0]^T$ | $m = 1$ | $0.5$ rad | $0.8776$ | $0.4794$ | $[0.8776, 0.4794]^T$ | — |
| **Key $k$** | $[0.0, 2.0]^T$ | $n = 3$ | $1.5$ rad | $0.0707$ | $0.9975$ | $[-1.9950, 0.1415]^T$ | — |
| **Direct Dot** | — | — | — | — | — | $\langle q_{\text{rot}}, k_{\text{rot}} \rangle$ | **$-1.682942$** |
| **Relative Form**| — | $\Delta = -2$ | $-1.0$ rad | $0.5403$ | $-0.8415$ | $0(\cos) + 2(\sin)$ | **$-1.682942$** |

---

## 6. Solved Illustrations

### Illustration 1: NTK-Aware Scaling Calculation
**Problem:**
A model is pre-trained with base frequency $b = 10,000$ on context length $L = 4096$ with head dimension $d = 64$.
We wish to extend its context window by $4\times$ to $L' = 16,384$ using NTK-Aware Scaling.
Calculate the scaled base frequency $b'$.

**Solution:**
Scale factor $s = \frac{16384}{4096} = 4.0$.
The NTK base scaling exponent is:
$$\alpha = \frac{d}{d - 2} = \frac{64}{64 - 2} = \frac{64}{62} \approx 1.032258$$
$$s^{\alpha} = 4^{1.032258} \approx 4.182937$$
The new base frequency is:
$$b' = b \cdot s^\alpha = 10,000 \times 4.182937 = \mathbf{41,829.37}$$
By increasing the base from $10,000$ to $41,829$, low frequencies rotate slower, allowing position signals to span $16\text{k}$ tokens without high-frequency degradation! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **LLaMA 3 / 3.1 128k Context:** Meta set the base frequency $\theta_{\text{base}} = 500,000$ (up from LLaMA 2's $10,000$), enabling the model to retain needle-in-a-haystack retrieval across 128,000 tokens with $100\%$ accuracy.
- **ALiBi vs. RoPE:** While ALiBi (Attention with Linear Biases) offers good extrapolation, it imposes a strictly monotonically decaying bias, preventing the model from attending strongly to middle-distance tokens. RoPE's periodic trigonometric basis allows arbitrary attention patterns, dominating state-of-the-art benchmarks.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - 2D rotation of $q$ and $k$ matching $[-1.994990, 0.141474]$ and dot product $-1.682942$ to $< 10^{-6}$.
   - Exact mathematical equivalence between rotated dot product and relative distance formula.
2. **Production-Ready PyTorch RoPE Module:**
   - Vectorized frequency calculation: $\theta_i = b^{-2i/d}$.
   - Fast complex-number implementation `apply_rotary_emb(x, freqs_cis)` used in LLaMA 3.
   - Relative invariance property test on batch tensor inputs.

See implementation in:
[`12_modern_llm_architectures/code/03_rope_and_long_context.py`](./code/03_rope_and_long_context.py)
