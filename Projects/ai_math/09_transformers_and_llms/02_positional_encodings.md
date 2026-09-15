# Chapter 9.2: Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi)

---

## 1. Intuition & 101 Motivation

The standard scaled dot-product attention operation $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}\mathbf{K}^T / \sqrt{d_k})\mathbf{V}$ exhibits a fundamental algebraic property: **Permutation Equivariance**.

For any arbitrary $T \times T$ permutation matrix $\mathbf{P}$:
$$\text{Attention}(\mathbf{P}\mathbf{Q}, \, \mathbf{P}\mathbf{K}, \, \mathbf{P}\mathbf{V}) = \mathbf{P} \, \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$

If you shuffle the input sentence:
$$\text{"The dog bit the man"} \quad \longleftrightarrow \quad \text{"The man bit the dog"}$$
a Transformer encoder with no positional information processes both sentences with **identically permuted hidden states**. The model reduces to an unordered Bag-of-Words!

Because natural syntax and semantics are strictly ordered, spatial coordinates must be explicitly injected.

```
                  EVOLUTION OF POSITIONAL ENCODING
┌───────────────────────────┐      ┌───────────────────────────┐
│ Sinusoidal (Vaswani 2017) │ ───► │  Learned Absolute (BERT)  │
│  Deterministic sin/cos    │      │  Embedding table (T_max)  │
│  Additive: x + PE         │      │  Fails beyond T_max       │
└─────────────┬─────────────┘      └─────────────┬─────────────┘
              │                                  │
              ▼                                  ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│     ALiBi (Press 2021)    │ ◄─── │      RoPE (Su 2021)       │
│  Static linear bias slope │      │  Rotary Position Embedding│
│  q*k - m*|i - j|          │      │  R_m * q, R_n * k in SO(2)│
│  Zero embeddings          │      │  The Modern LLM Standard  │
└───────────────────────────┘      └───────────────────────────┘
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Sinusoidal Positional Encoding (Vaswani et al., 2017)

Vaswani et al. proposed fixed deterministic sinusoidal functions across geometric frequencies:
$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$
$$PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$
where $pos \in \{0, 1, \dots, T-1\}$ is the token index, and $i \in \{0, \dots, d_{\text{model}}/2 - 1\}$ indexes the frequency channel.

The wavelengths form a geometric progression from $2\pi$ to $10,000 \cdot 2\pi$.

#### The Linear Relative Offset Property
For any fixed temporal offset $k$, there exists a linear transformation matrix $\mathbf{M}_k \in \mathbb{R}^{2 \times 2}$ such that:
$$\begin{bmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$
where $\omega_i = \frac{1}{10000^{2i / d_{\text{model}}}}$.
*Proof:* Direct application of angle sum trigonometric identities:
$$\sin(\omega_i (pos + k)) = \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)$$
$$\cos(\omega_i (pos + k)) = \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)$$
This allows the attention mechanism to learn to attend to relative positions $k = pos_q - pos_k$ via linear transformations.

---

### 2.2 Learned Absolute Positional Embeddings (BERT, GPT-2, GPT-3)

Instead of handcrafting trigonometric functions, models allocate a trainable parameter matrix:
$$\mathbf{E}_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}$$
and compute input embeddings additively:
$$\mathbf{z}_0 = \mathbf{X}_{\text{token}} + \mathbf{E}_{\text{pos}}[0:T]$$

#### The Fundamental Failure of Learned Embeddings: Context Extrapolation
- Learned embeddings are bounded by $T_{\max}$ (e.g., $2,048$ in GPT-3).
- If the model encounters an input of length $T = 2,049$, position $2,048$ has no embedding vector; the model physically crashes.
- Position coordinates that were rarely observed during pre-training (e.g., positions $1,900 - 2,048$) suffer from high estimation variance and poor generalization.

---

### 2.3 Rotary Position Embedding (RoPE) (Su et al., 2021)

Jianlin Su et al. proposed **RoPE (Rotary Position Embedding)**, currently the standard in modern LLMs (LLaMA 1/2/3, Mistral, Gemma, Qwen).

#### The Foundational Objective:
We seek an encoding function that transforms Query $\mathbf{q}$ at position $m$ and Key $\mathbf{k}$ at position $n$ such that their inner product depends **only on the relative displacement $m - n$**:
$$\langle f_q(\mathbf{q}, m), \, f_k(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$

#### The 2D Derivation:
In a 2-dimensional space, treating vectors as complex numbers $\mathbf{q} = q_1 + i q_2 \in \mathbb{C}$:
$$f_q(\mathbf{q}, m) = \mathbf{q} \, e^{i m \theta}$$
$$f_k(\mathbf{k}, n) = \mathbf{k} \, e^{i n \theta}$$

Taking the complex inner product $\langle \mathbf{u}, \mathbf{v} \rangle = \operatorname{Re}(\mathbf{u} \mathbf{v}^*)$:
$$\langle f_q(\mathbf{q}, m), \, f_k(\mathbf{k}, n) \rangle = \operatorname{Re}\left( (\mathbf{q} e^{i m \theta}) (\mathbf{k} e^{i n \theta})^* \right) = \operatorname{Re}\left( \mathbf{q} \mathbf{k}^* e^{i (m - n) \theta} \right)$$
The inner product depends **strictly on the relative distance $m - n$**!

#### Matrix Formulation in $d$-Dimensions:
In real matrix calculus, multiplying by $e^{i m \theta}$ is an orthogonal 2D rotation matrix in $\mathrm{SO}(2)$:
$$\mathbf{R}_{\theta, m}^{(i)} = \begin{bmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{bmatrix}$$
where $\theta_i = 10000^{-2(i-1)/d}$ for $i \in \{1, \dots, d/2\}$.

In $d$-dimensions, RoPE splits the vector into $d/2$ consecutive pairs and applies block-diagonal rotation:
$$\mathbf{R}_{\Theta, m} = \begin{bmatrix}
\mathbf{R}_{\theta_1, m}^{(1)} & & & 0 \\
& \mathbf{R}_{\theta_2, m}^{(2)} & & \\
& & \ddots & \\
0 & & & \mathbf{R}_{\theta_{d/2}, m}^{(d/2)}
\end{bmatrix} \in \mathbb{R}^{d \times d}$$

$$\tilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m} \mathbf{q}_m, \quad \tilde{\mathbf{k}}_n = \mathbf{R}_{\Theta, n} \mathbf{k}_n$$

#### The Relative Shift Invariance Property:
Because rotation matrices are orthogonal ($\mathbf{R}_m^T = \mathbf{R}_{-m}$ and $\mathbf{R}_m \mathbf{R}_n = \mathbf{R}_{m+n}$):
$$\tilde{\mathbf{q}}_m^T \tilde{\mathbf{k}}_n = (\mathbf{R}_m \mathbf{q})^T (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{k} = \mathbf{q}^T \mathbf{R}_{n - m} \mathbf{k}$$
For any arbitrary temporal shift $s \in \mathbb{Z}$:
$$\tilde{\mathbf{q}}_{m+s}^T \tilde{\mathbf{k}}_{n+s} = \mathbf{q}^T \mathbf{R}_{(n+s) - (m+s)} \mathbf{k} = \mathbf{q}^T \mathbf{R}_{n - m} \mathbf{k} \equiv \tilde{\mathbf{q}}_m^T \tilde{\mathbf{k}}_n$$
The attention score between two tokens is strictly invariant to shifting the entire sequence across time.

---

### 2.4 ALiBi (Attention with Linear Biases) (Press et al., 2021)

Ofir Press et al. introduced **ALiBi**, which completely removes positional embeddings from token representations:
$$\mathbf{X} = \mathbf{X}_{\text{token}} \quad (\text{No } PE \text{ added!})$$

Instead, ALiBi injects position directly into the attention score matrix by penalizing token distance linearly:
$$\text{score}(q_i, k_j) = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} - m \cdot |i - j|$$
where $m \in \mathbb{R}^+$ is a head-specific scalar slope fixed geometrically across the $h$ attention heads:
$$m \in \left\{ 2^{-8/h}, \, 2^{-16/h}, \, 2^{-24/h}, \, \dots, \, 2^{-8} \right\}$$
For $h = 8$ heads:
$$m \in \left\{ \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256} \right\}$$

- Heads with large slopes (e.g., $m = 1/2$) focus strictly on local immediate context.
- Heads with tiny slopes (e.g., $m = 1/256$) attend globally across thousands of tokens.
- **Extreme Extrapolation:** Models trained on sequence length $T = 1,024$ extrapolate seamlessly to $T = 8,192$ with zero fine-tuning.

---

## 3. Geometric & Algebraic Interpretation

```
              ROPE ROTATION IN 2D SUBSPACE (SO(2))
                              q_2, k_2
                                 ▲
                                 │       * k_tilde (Rotated by 2*theta)
                                 │      /
     q_tilde (Rotated by 1*theta)│ *   /
                               \ │/   /
                                \|   /
                                 O──/────────► q_1, k_1
                                   /
                                  /
            Relative Angle between q_tilde and k_tilde = (2 - 1)*theta = theta
```

### The Long-Term Decay Property of RoPE
By the Riemann-Lebesgue lemma, the expected inner product under RoPE decays gracefully as relative distance $|m - n|$ increases:
$$\lim_{|m - n| \to \infty} \mathbb{E}\left[ \mathbf{q}_m^T \mathbf{k}_n \right] = 0$$
Tokens naturally attend most strongly to nearby syntactic neighbors, while retaining the capacity to form long-range connections when Query and Key alignments are exceptionally strong.

---

## 4. Real-World Analogy

### The Analog Clock Face (RoPE)
- Imagine a massive 12-hour analog clock face.
- A token placed at position 1 sets its clock hand to **1:00** (rotates by $30^\circ$).
- A token placed at position 4 sets its clock hand to **4:00** (rotates by $120^\circ$).
- To compute their similarity, the attention mechanism measures the **angle between the two clock hands**:
  $$4:00 - 1:00 = 3 \text{ hours} \quad (90^\circ \text{ difference})$$
- If the entire document is shifted forward by 5 pages, the tokens now sit at positions 6 and 9 (hands pointing to 6:00 and 9:00).
- The angle between them is still $9:00 - 6:00 = 3 \text{ hours}$ ($90^\circ$)!
- The model evaluates their relationship identically, regardless of where they sit in a 100,000-word book.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete 2D RoPE calculation with concrete hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in RoPE |
| :--- | :--- | :--- | :--- |
| $d$ | Head Dimension | Scalar ($2$) | Single 2D rotational subspace |
| $\theta$ | Base Rotation Angle | Scalar ($\frac{\pi}{4} = 45^\circ$) | Frequency angle for this 2D pair |
| $m$ | Query Position | Scalar ($1$) | Token coordinate of Query |
| $n$ | Key Position | Scalar ($2$) | Token coordinate of Key |
| $\mathbf{q}$ | Raw Query Vector | $(2,)$ | Query before positional rotation |
| $\mathbf{k}$ | Raw Key Vector | $(2,)$ | Key before positional rotation |
| $\mathbf{R}_m$ | Rotation Matrix at $m$ | $(2, 2)$ | $\mathrm{SO}(2)$ rotation by angle $m \theta$ |
| $\mathbf{R}_n$ | Rotation Matrix at $n$ | $(2, 2)$ | $\mathrm{SO}(2)$ rotation by angle $n \theta$ |
| $\tilde{\mathbf{q}}_m$ | Rotated Query | $(2,)$ | $\mathbf{R}_m \mathbf{q}$ |
| $\tilde{\mathbf{k}}_n$ | Rotated Key | $(2,)$ | $\mathbf{R}_n \mathbf{k}$ |

---

### 5.2 Concrete Toy Numbers

#### Base Angle:
$$\theta = \frac{\pi}{4} = 45^\circ \implies \cos(45^\circ) = \frac{\sqrt{2}}{2} \approx 0.707107, \quad \sin(45^\circ) = \frac{\sqrt{2}}{2} \approx 0.707107$$

#### Query at Position $m = 1$:
$$\text{Angle } m \theta = 1 \times 45^\circ = 45^\circ$$
$$\mathbf{q} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$

#### Key at Position $n = 2$:
$$\text{Angle } n \theta = 2 \times 45^\circ = 90^\circ$$
$$\cos(90^\circ) = 0.0, \quad \sin(90^\circ) = 1.0$$
$$\mathbf{k} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$

---

### 5.3 Step 1: Construct Rotation Matrices

$$\mathbf{R}_1 = \begin{bmatrix} \cos(45^\circ) & -\sin(45^\circ) \\ \sin(45^\circ) & \cos(45^\circ) \end{bmatrix} = \begin{bmatrix} 0.707107 & -0.707107 \\ 0.707107 & 0.707107 \end{bmatrix}$$

$$\mathbf{R}_2 = \begin{bmatrix} \cos(90^\circ) & -\sin(90^\circ) \\ \sin(90^\circ) & \cos(90^\circ) \end{bmatrix} = \begin{bmatrix} 0.0 & -1.0 \\ 1.0 & 0.0 \end{bmatrix}$$

---

### 5.4 Step 2: Apply Rotation to Query and Key

1. **Rotated Query $\tilde{\mathbf{q}}_1 = \mathbf{R}_1 \mathbf{q}$:**
   $$\tilde{\mathbf{q}}_1 = \begin{bmatrix} 0.707107 & -0.707107 \\ 0.707107 & 0.707107 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.707107 \\ 0.707107 \end{bmatrix}}$$

2. **Rotated Key $\tilde{\mathbf{k}}_2 = \mathbf{R}_2 \mathbf{k}$:**
   $$\tilde{\mathbf{k}}_2 = \begin{bmatrix} 0.0 & -1.0 \\ 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} -1.0 \\ 0.0 \end{bmatrix}}$$

---

### 5.5 Step 3: Compute Attention Score

$$\text{Score} = \tilde{\mathbf{q}}_1^T \tilde{\mathbf{k}}_2 = [0.707107, \, 0.707107] \begin{bmatrix} -1.0 \\ 0.0 \end{bmatrix} = (0.707107)(-1.0) + (0.707107)(0.0) = \mathbf{-0.707107}$$

---

### 5.6 Step 4: Verification of Relative Shift Invariance

Now compute via the relative displacement formula:
$$\Delta pos = n - m = 2 - 1 = 1 \implies \text{Angle } 1 \times 45^\circ = 45^\circ$$
$$\mathbf{R}_{n - m} = \mathbf{R}_1 = \begin{bmatrix} 0.707107 & -0.707107 \\ 0.707107 & 0.707107 \end{bmatrix}$$

Multiply $\mathbf{q}^T \mathbf{R}_1 \mathbf{k}$:
$$\mathbf{R}_1 \mathbf{k} = \begin{bmatrix} 0.707107 & -0.707107 \\ 0.707107 & 0.707107 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} -0.707107 \\ 0.707107 \end{bmatrix}$$
$$\mathbf{q}^T (\mathbf{R}_1 \mathbf{k}) = [1.0, \, 0.0] \begin{bmatrix} -0.707107 \\ 0.707107 \end{bmatrix} = (1.0)(-0.707107) = \mathbf{-0.707107}$$

The results match identically to machine precision:
$$\tilde{\mathbf{q}}_1^T \tilde{\mathbf{k}}_2 \equiv \mathbf{q}^T \mathbf{R}_{n - m} \mathbf{k} = -0.707107$$

---

## 6. Solved Illustrations

### Illustration 1: RoPE Base Frequency Scaling for Long Context (YaRN & NTK-Aware)

**Problem:**
LLaMA-2 was pre-trained with RoPE base frequency $b = 10,000$ on context length $4,096$.
When expanding LLaMA-3 to $8,192$ and $128,000$ tokens, Meta scaled the base frequency to $b = 500,000$.
Why does increasing the base frequency $b$ expand the effective context window?

**Solution:**
The rotation frequency for dimension pair $i$ is:
$$\theta_i = b^{-2(i-1)/d}$$
The period (wavelength) of the lowest frequency component ($i = d/2$) is:
$$\lambda_{\max} = \frac{2\pi}{\theta_{d/2}} = 2\pi \cdot b$$
For $b = 10,000$:
$$\lambda_{\max} = 2\pi \times 10,000 \approx 62,831$$
However, high-frequency components have very short wavelengths. If context length expands beyond the training horizon, high-frequency coordinates rotate through multiple full cycles, creating out-of-distribution phase angles.
By increasing $b \to 500,000$:
- All frequencies are scaled down ($\theta_i \downarrow$).
- Rotation angles $m \theta_i$ accumulate much more slowly across token positions $m$.
- A sequence length of $128,000$ produces rotational phases that remain within the familiar range seen during pre-training, preventing catastrophic perplexity spikes.

---

### Illustration 2: Summary Matrix of Modern Positional Encoding Schemes

| Method | Type | Context Extrapolation | Where Used | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Sinusoidal** | Absolute | Moderate | Original Transformer, ViT | Parameter-free, deterministic |
| **Learned Absolute** | Absolute | Zero ($T \le T_{\max}$) | BERT, GPT-2, GPT-3 | Highly expressive on fixed lengths |
| **RoPE** | Relative (Rotary) | High (with frequency scaling) | LLaMA 1/2/3, Mistral, Gemma | Strictly relative, zero extra parameters |
| **ALiBi** | Relative (Bias) | **Extraordinary ($10\times$)** | MPT-7B, BLOOM | Zero embedding parameters, fastest training |

---

## 7. Deep Learning Connection & Application

### 1. Vectorized Implementation of RoPE Without Explicit Matrices
In frameworks like PyTorch and Triton, one never creates full block-diagonal matrices $\mathbf{R}_{\Theta, m} \in \mathbb{R}^{d \times d}$.
Instead, the 2D rotation of pair $[x_1, x_2]$:
$$\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1 \cos\theta - x_2 \sin\theta \\ x_1 \sin\theta + x_2 \cos\theta \end{bmatrix}$$
is vectorized across all $d$ channels via:
$$\mathbf{R}_{\Theta, m} \mathbf{x} = (\mathbf{x} \odot \cos(m \boldsymbol{\theta})) + (\operatorname{rotate\_half}(\mathbf{x}) \odot \sin(m \boldsymbol{\theta}))$$
where:
$$\operatorname{rotate\_half}(\mathbf{x}) = [-x_{d/2+1:d}, \, x_{1:d/2}] \quad \text{or} \quad [-x_2, x_1, -x_4, x_3, \dots]$$
This requires only **two element-wise multiplications and one addition**, executing at memory bandwidth speeds in a single fused GPU kernel.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `SinusoidalPositionalEncoding`: Pure PyTorch module implementing Vaswani's formula.
2. `RotaryPositionalEmbedding`: Vectorized implementation using `rotate_half` and cosine/sine caching.
3. `ALiBiAttention`: Multi-head attention with linear bias slopes.
4. Exact numerical verification of the Part 5 Visual Grid RoPE hand arithmetic.
5. Proof of the RoPE Relative Shift Invariance:
   $$\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle \equiv \langle \tilde{\mathbf{q}}_{m+s}, \tilde{\mathbf{k}}_{n+s} \rangle, \quad \forall s \in \mathbb{Z}$$
6. ALiBi context extrapolation verification.

See implementation in:
[`09_transformers_and_llms/code/02_positional_encodings.py`](./code/02_positional_encodings.py)
