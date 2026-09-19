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

### 2.5 Deep Derivation 9.2.1: Functional Equation Solution for RoPE and Complex Uniqueness

#### The Postulate of Relative Position Invariance
We require an encoding function $\mathbf{f}_q(\mathbf{q}, m)$ and $\mathbf{f}_k(\mathbf{k}, n)$ acting on $\mathbb{R}^2$ such that their inner product is a function solely of their vectors and relative displacement $m - n$:
$$\langle \mathbf{f}_q(\mathbf{q}, m), \, \mathbf{f}_k(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$
with boundary initial conditions:
$$\mathbf{f}_q(\mathbf{q}, 0) = \mathbf{q}, \quad \mathbf{f}_k(\mathbf{k}, 0) = \mathbf{k}$$

#### Theorem: Complex Exponential Representation
Under the assumption of linearity and norm preservation ($\|\mathbf{f}(\mathbf{x}, m)\|_2 = \|\mathbf{x}\|_2$), the unique solution in 2D is:
$$\mathbf{f}_q(\mathbf{q}, m) = \mathbf{q} \, e^{i m \theta}, \quad \mathbf{f}_k(\mathbf{k}, n) = \mathbf{k} \, e^{i n \theta}$$
where $\mathbf{q}, \mathbf{k} \in \mathbb{C} \cong \mathbb{R}^2$ and $\theta \in \mathbb{R}$.

#### Proof:
1. **Polar Decomposition in $\mathbb{C}$:**
   Represent vectors as complex numbers $\mathbf{q} = r_q e^{i \phi_q}$ and $\mathbf{k} = r_k e^{i \phi_k}$.
   Assume the transformation takes the form:
   $$\mathbf{f}_q(\mathbf{q}, m) = R_q(\mathbf{q}, m) e^{i \Theta_q(\mathbf{q}, m)}$$
   $$\mathbf{f}_k(\mathbf{k}, n) = R_k(\mathbf{k}, n) e^{i \Theta_k(\mathbf{k}, n)}$$
   From the initial conditions at $m=0, n=0$:
   $$R_q(\mathbf{q}, 0) = r_q, \quad \Theta_q(\mathbf{q}, 0) = \phi_q$$
   $$R_k(\mathbf{k}, 0) = r_k, \quad \Theta_k(\mathbf{k}, 0) = \phi_k$$

2. **Inner Product Formulation:**
   The inner product in $\mathbb{C}$ is given by $\operatorname{Re}(\mathbf{u} \mathbf{v}^*)$:
   $$\langle \mathbf{f}_q, \mathbf{f}_k \rangle = R_q(\mathbf{q}, m) R_k(\mathbf{k}, n) \cos\left( \Theta_q(\mathbf{q}, m) - \Theta_k(\mathbf{k}, n) \right) = g(\mathbf{q}, \mathbf{k}, m - n)$$

3. **Separation of Radial and Phase Terms:**
   Setting $m = n$ gives:
   $$R_q(\mathbf{q}, m) R_k(\mathbf{k}, m) \cos\left( \Theta_q(\mathbf{q}, m) - \Theta_k(\mathbf{k}, m) \right) = g(\mathbf{q}, \mathbf{k}, 0) = r_q r_k \cos(\phi_q - \phi_k)$$
   For this identity to hold for all $m$, the radial magnitudes must be position-invariant:
   $$R_q(\mathbf{q}, m) = r_q = \|\mathbf{q}\|, \quad R_k(\mathbf{k}, n) = r_k = \|\mathbf{k}\|$$
   and the phase difference must satisfy:
   $$\Theta_q(\mathbf{q}, m) - \Theta_k(\mathbf{k}, m) = \phi_q - \phi_k$$

4. **Cauchy's Additive Functional Equation:**
   The phase condition requires:
   $$\Theta_q(\mathbf{q}, m) - \Theta_k(\mathbf{k}, n) = \Phi(m - n) + (\phi_q - \phi_k)$$
   For $m=n$, $\Phi(0) = 0$.
   Setting $\phi_q = \phi_k = 0$, let $\theta(m) = \Theta(m)$. The equation becomes:
   $$\theta(m) - \theta(n) = \Phi(m - n)$$
   Setting $n = 0 \implies \Phi(m) = \theta(m) - \theta(0) = \theta(m)$.
   Therefore:
   $$\theta(m) - \theta(n) = \theta(m - n) \implies \theta(m - n) + \theta(n) = \theta(m)$$
   This is **Cauchy's fundamental functional equation**: $f(x + y) = f(x) + f(y)$.
   For continuous functions, the unique non-trivial solution is strictly linear:
   $$\theta(m) = m \cdot \theta$$
   where $\theta \in \mathbb{R}$ is an arbitrary constant base frequency.
   Converting back to the Cartesian plane $\mathbb{R}^2$:
   $$\mathbf{f}_q(\mathbf{q}, m) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_1 \\ q_2 \end{bmatrix}$$
   which proves that rotation in $\mathrm{SO}(2)$ is the unique norm-preserving linear relative position mapping in 2D. $\blacksquare$

---

### 2.6 Deep Derivation 9.2.2: Analytical Backpropagation and Isometry Conservation of RoPE

#### Theorem: Adjoint of Rotary Position Embedding
Let $\tilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m} \mathbf{q}_m \in \mathbb{R}^d$ be the rotated query at sequence index $m$, where $\mathbf{R}_{\Theta, m} \in \mathrm{SO}(2)^{d/2}$ is orthogonal.
Given the upstream loss sensitivity $\delta_{\tilde{q}} = \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{q}}_m} \in \mathbb{R}^d$:

1. The exact analytical gradient with respect to the raw unrotated query $\mathbf{q}_m$ is:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{q}_m} = \mathbf{R}_{\Theta, m}^T \delta_{\tilde{q}} = \mathbf{R}_{\Theta, -m} \delta_{\tilde{q}}$$
2. RoPE preserves gradient magnitude identically across all sequence positions:
   $$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{q}_m} \right\|_2 \equiv \|\delta_{\tilde{q}}\|_2, \quad \forall m \in \mathbb{N}$$

#### Proof:
1. **Differential Derivation:**
   The variation of loss is:
   $$d\mathcal{L} = \left\langle \delta_{\tilde{q}}, \, d\tilde{\mathbf{q}}_m \right\rangle = \delta_{\tilde{q}}^T d\tilde{\mathbf{q}}_m = \delta_{\tilde{q}}^T (\mathbf{R}_{\Theta, m} \, d\mathbf{q}_m) = (\mathbf{R}_{\Theta, m}^T \delta_{\tilde{q}})^T d\mathbf{q}_m$$
   By definition of the gradient:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{q}_m} = \mathbf{R}_{\Theta, m}^T \delta_{\tilde{q}}$$

2. **Orthogonality and Inverse Rotation:**
   Each $2 \times 2$ block of $\mathbf{R}_{\Theta, m}$ is:
   $$\mathbf{R}_{\theta_i, m} = \begin{bmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{bmatrix}$$
   Taking the transpose:
   $$\mathbf{R}_{\theta_i, m}^T = \begin{bmatrix} \cos(m \theta_i) & \sin(m \theta_i) \\ -\sin(m \theta_i) & \cos(m \theta_i) \end{bmatrix} = \begin{bmatrix} \cos(-m \theta_i) & -\sin(-m \theta_i) \\ \sin(-m \theta_i) & \cos(-m \theta_i) \end{bmatrix} = \mathbf{R}_{\theta_i, -m}$$
   Thus, backpropagating through RoPE is mathematically identical to applying forward RoPE with the **negated positional coordinate $-m$**!

3. **Norm Conservation (Isometry):**
   Because $\mathbf{R}_{\Theta, m} \in \mathrm{O}(d)$ is an orthogonal transformation:
   $$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{q}_m} \right\|_2^2 = \left( \mathbf{R}_{\Theta, -m} \delta_{\tilde{q}} \right)^T \left( \mathbf{R}_{\Theta, -m} \delta_{\tilde{q}} \right) = \delta_{\tilde{q}}^T \left( \mathbf{R}_{\Theta, -m}^T \mathbf{R}_{\Theta, -m} \right) \delta_{\tilde{q}} = \delta_{\tilde{q}}^T \mathbf{I} \delta_{\tilde{q}} = \|\delta_{\tilde{q}}\|_2^2$$
   Unlike additive positional encodings (which distort gradient norm depending on $\|PE_m\|$) or learned embeddings (which fragment gradients into separate embedding table rows), RoPE is an **exact isometry**. It introduces **zero gradient vanishing or explosion** regardless of how deep or long the sequence is. $\blacksquare$

---

### 2.7 Deep Derivation 9.2.3: Context Window Extension: Linear Scaling vs. NTK-Aware vs. YaRN

#### 1. Linear RoPE Position Interpolation (PI) (Chen et al., 2023)
To extend the context window from original limit $L$ to new target $L' > L$, define scale factor $s = \frac{L'}{L} > 1$.
Position Interpolation scales position indices down uniformly:
$$m' = \frac{m}{s}$$
Equivalently, all base frequencies are divided by $s$:
$$\theta_i' = \frac{\theta_i}{s} = \frac{10000^{-2(i-1)/d}}{s}$$
- **Limitation:** While it successfully limits maximum phase angles to pre-training values, scaling all frequencies uniformly compresses **high-frequency components**, causing the model to lose fine-grained local syntactic discrimination (catastrophic local attention degradation).

#### 2. NTK-Aware RoPE Scaling (bloc97, 2023)
Neural Tangent Kernel (NTK) theory establishes that deep networks struggle to learn high frequencies if they are stretched.
Instead of scaling all frequencies by $s$, NTK-Aware scaling changes the **base $b = 10,000$** to a larger effective base $b'$:
$$b' = b \cdot s^{d / (d - 2)}$$
The modified frequency is:
$$\theta_i^{\text{NTK}} = (b')^{-2(i-1)/d} = \left( b \cdot s^{d/(d-2)} \right)^{-2(i-1)/d} = \theta_i \cdot s^{-\frac{2(i-1)}{d-2}}$$
Let us inspect the boundary behavior:
- **Highest frequency ($i = 1$):**
  $$s^{-\frac{2(1-1)}{d-2}} = s^0 = 1 \implies \theta_1^{\text{NTK}} = \theta_1$$
  High frequencies are **completely unscaled**, preserving local token precision!
- **Lowest frequency ($i = d/2$):**
  $$s^{-\frac{2(d/2 - 1)}{d-2}} = s^{-\frac{d-2}{d-2}} = s^{-1} = \frac{1}{s} \implies \theta_{d/2}^{\text{NTK}} = \frac{\theta_{d/2}}{s}$$
  Low frequencies are **fully interpolated by $1/s$**, extending the global context horizon!

#### 3. YaRN (Yet another RoPE extensioN) (Peng et al., 2023)
YaRN defines three frequency regimes based on wavelength $\lambda_i = \frac{2\pi}{\theta_i}$:
1. If $\lambda_i < \beta L$: High frequency (local context). Do **not** interpolate ($s_i = 1$).
2. If $\lambda_i > \alpha L$: Low frequency (global context). Fully interpolate ($s_i = s$).
3. If $\beta L \le \lambda_i \le \alpha L$: Smooth ramp interpolation using ramp function $\gamma(r) = \frac{r - \beta}{\alpha - \beta}$:
   $$\theta_i^{\text{YaRN}} = (1 - \gamma_i) \frac{\theta_i}{s} + \gamma_i \theta_i$$
YaRN delivers superior context extension with minimal perplexity degradation. $\blacksquare$

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

### Illustration 3: Complete 4D RoPE Forward and Backward Hand Trace across 2 Frequency Channels

**Problem:**
Consider a 4-dimensional attention head ($d = 4$) with two 2D frequency subspaces:
- Subspace 1 ($i = 0$): Base angle $\theta_0 = 45^\circ = \frac{\pi}{4} \implies \cos(45^\circ) = \frac{\sqrt{2}}{2} \approx 0.707107, \, \sin(45^\circ) \approx 0.707107$.
- Subspace 2 ($i = 1$): Base angle $\theta_1 = 30^\circ = \frac{\pi}{6} \implies \cos(30^\circ) = \frac{\sqrt{3}}{2} \approx 0.866025, \, \sin(30^\circ) = 0.500000$.

Let Query vector at position $m = 3$ and Key vector at position $n = 1$ be:
$$\mathbf{q} = \begin{bmatrix} 1.0 \\ 0.0 \\ 2.0 \\ -1.0 \end{bmatrix}, \quad \mathbf{k} = \begin{bmatrix} 0.0 \\ 1.0 \\ 1.0 \\ 1.0 \end{bmatrix}$$
1. Evaluate rotated Query $\tilde{\mathbf{q}}_3$ and rotated Key $\tilde{\mathbf{k}}_1$.
2. Compute the attention inner product $\tilde{\mathbf{q}}_3^T \tilde{\mathbf{k}}_1$.
3. Verify that the result equals $\mathbf{q}^T \mathbf{R}_{\Theta, n - m} \mathbf{k}$ using relative displacement $\Delta = n - m = -2$.
4. Given upstream gradient $\delta_{\tilde{q}} = \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{q}}_3} = [1.0, 1.0, 0.0, -1.0]^T$, calculate the raw parameter gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{q}}$ and verify exact $\ell_2$-norm conservation.

**Solution:**

#### Step 1: Forward Rotation

1. **Query Rotation ($m = 3$):**
   - **Subspace 1 ($3 \times 45^\circ = 135^\circ$):**
     $$\cos(135^\circ) = -0.707107, \quad \sin(135^\circ) = +0.707107$$
     $$\tilde{q}_1 = (1.0)(-0.707107) - (0.0)(0.707107) = \mathbf{-0.707107}$$
     $$\tilde{q}_2 = (1.0)(0.707107) + (0.0)(-0.707107) = \mathbf{+0.707107}$$
   - **Subspace 2 ($3 \times 30^\circ = 90^\circ$):**
     $$\cos(90^\circ) = 0.000000, \quad \sin(90^\circ) = 1.000000$$
     $$\tilde{q}_3 = (2.0)(0.0) - (-1.0)(1.0) = 0.0 + 1.0 = \mathbf{+1.000000}$$
     $$\tilde{q}_4 = (2.0)(1.0) + (-1.0)(0.0) = 2.0 + 0.0 = \mathbf{+2.000000}$$
   $$\tilde{\mathbf{q}}_3 = \begin{bmatrix} -0.707107 \\ 0.707107 \\ 1.000000 \\ 2.000000 \end{bmatrix}$$

2. **Key Rotation ($n = 1$):**
   - **Subspace 1 ($1 \times 45^\circ = 45^\circ$):**
     $$\tilde{k}_1 = (0.0)(0.707107) - (1.0)(0.707107) = \mathbf{-0.707107}$$
     $$\tilde{k}_2 = (0.0)(0.707107) + (1.0)(0.707107) = \mathbf{+0.707107}$$
   - **Subspace 2 ($1 \times 30^\circ = 30^\circ$):**
     $$\tilde{k}_3 = (1.0)(0.866025) - (1.0)(0.500000) = 0.866025 - 0.500000 = \mathbf{+0.366025}$$
     $$\tilde{k}_4 = (1.0)(0.500000) + (1.0)(0.866025) = 0.500000 + 0.866025 = \mathbf{+1.366025}$$
   $$\tilde{\mathbf{k}}_1 = \begin{bmatrix} -0.707107 \\ 0.707107 \\ 0.366025 \\ 1.366025 \end{bmatrix}$$

---

#### Step 2: Inner Product Computation

$$\tilde{\mathbf{q}}_3^T \tilde{\mathbf{k}}_1 = (-0.707107)(-0.707107) + (0.707107)(0.707107) + (1.000000)(0.366025) + (2.000000)(1.366025)$$
$$= 0.500000 + 0.500000 + 0.366025 + 2.732050 = 1.000000 + 3.098075 = \mathbf{4.098075}$$

---

#### Step 3: Relative Shift Verification ($\Delta = n - m = 1 - 3 = -2$)

- **Subspace 1:** Angle is $-2 \times 45^\circ = -90^\circ \implies \cos(-90^\circ) = 0, \sin(-90^\circ) = -1$.
  $$\mathbf{R}_{-90^\circ} = \begin{bmatrix} 0.0 & 1.0 \\ -1.0 & 0.0 \end{bmatrix}$$
  $$[q_1, q_2] \mathbf{R}_{-90^\circ} \begin{bmatrix} k_1 \\ k_2 \end{bmatrix} = [1.0, 0.0] \begin{bmatrix} 0.0 & 1.0 \\ -1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = [1.0, 0.0] \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \mathbf{1.000000}$$

- **Subspace 2:** Angle is $-2 \times 30^\circ = -60^\circ \implies \cos(-60^\circ) = 0.500000, \sin(-60^\circ) = -0.866025$.
  $$\mathbf{R}_{-60^\circ} = \begin{bmatrix} 0.500000 & 0.866025 \\ -0.866025 & 0.500000 \end{bmatrix}$$
  $$\mathbf{R}_{-60^\circ} \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.500000 + 0.866025 \\ -0.866025 + 0.500000 \end{bmatrix} = \begin{bmatrix} 1.366025 \\ -0.366025 \end{bmatrix}$$
  $$[q_3, q_4] \begin{bmatrix} 1.366025 \\ -0.366025 \end{bmatrix} = (2.0)(1.366025) + (-1.0)(-0.366025) = 2.732050 + 0.366025 = \mathbf{3.098075}$$

- **Total Relative Inner Product:**
  $$1.000000 + 3.098075 = \mathbf{4.098075}$$
  *(Identity holds identically: $\tilde{\mathbf{q}}_3^T \tilde{\mathbf{k}}_1 \equiv \mathbf{q}^T \mathbf{R}_{\Theta, -2} \mathbf{k}$)*.

---

#### Step 4: Backward Pass and Isometry Check

Given upstream loss gradient $\delta_{\tilde{q}} = [1.0, 1.0, 0.0, -1.0]^T$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{q}} = \mathbf{R}_{\Theta, -3} \delta_{\tilde{q}}$$

- **Subspace 1 (angle $-135^\circ$):** $\cos(-135^\circ) = -0.707107, \sin(-135^\circ) = -0.707107$.
  $$\mathbf{R}_{-135^\circ} = \begin{bmatrix} -0.707107 & 0.707107 \\ -0.707107 & -0.707107 \end{bmatrix}$$
  $$\frac{\partial \mathcal{L}}{\partial [q_1, q_2]^T} = \begin{bmatrix} -0.707107 & 0.707107 \\ -0.707107 & -0.707107 \end{bmatrix} \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{0.000000} \\ \mathbf{-1.414214} \end{bmatrix}$$

- **Subspace 2 (angle $-90^\circ$):** $\cos(-90^\circ) = 0.0, \sin(-90^\circ) = -1.0$.
  $$\mathbf{R}_{-90^\circ} = \begin{bmatrix} 0.0 & 1.0 \\ -1.0 & 0.0 \end{bmatrix}$$
  $$\frac{\partial \mathcal{L}}{\partial [q_3, q_4]^T} = \begin{bmatrix} 0.0 & 1.0 \\ -1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 0.0 \\ -1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.000000} \\ \mathbf{0.000000} \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{q}} = \begin{bmatrix} 0.000000 \\ -1.414214 \\ -1.000000 \\ 0.000000 \end{bmatrix}$$

- **Norm Preservation Verification:**
  $$\|\delta_{\tilde{q}}\|_2^2 = (1.0)^2 + (1.0)^2 + (0.0)^2 + (-1.0)^2 = 1 + 1 + 0 + 1 = \mathbf{3.000000}$$
  $$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{q}} \right\|_2^2 = (0.0)^2 + (-1.414214)^2 + (-1.0)^2 + (0.0)^2 = 0 + 2.000000 + 1.0 + 0 = \mathbf{3.000000}$$
  Gradient magnitude is perfectly preserved without damping or amplification!

---

### Illustration 4: ALiBi Multi-Head Attention Score Matrix Hand Calculation

**Problem:**
A sequence of $T = 3$ tokens produces the unnormalized scaled attention score matrix:
$$\mathbf{S} = \begin{bmatrix}
2.0 & 1.0 & 0.0 \\
1.5 & 3.0 & 2.0 \\
1.0 & 2.5 & 3.5
\end{bmatrix}$$
Consider two attention heads:
- Head 1: Local slope $m_1 = 0.5$
- Head 2: Global slope $m_2 = 0.125$

1. Write out the causal distance penalty matrix $\mathbf{B}$ where $B_{i, j} = -m \cdot (i - j)$ for $j \le i$, and $-\infty$ for $j > i$.
2. Compute the effective pre-softmax logit matrix $\mathbf{S}' = \mathbf{S} + \mathbf{B}$ for both heads.
3. Compute the resulting softmax attention probability distributions for row 3 ($i = 3$) for both heads, demonstrating how slope $m$ modulates token attention locality.

**Solution:**

#### Step 1: Distance Matrix and Bias Construction

The causal relative distances $(i - j)$ are:
$$\mathbf{D} = \begin{bmatrix}
0 & \infty & \infty \\
1 & 0 & \infty \\
2 & 1 & 0
\end{bmatrix}$$

1. **Head 1 ($m_1 = 0.5$):**
   $$B_{i, j}^{(1)} = -0.5 \times (i - j) \implies \mathbf{B}^{(1)} = \begin{bmatrix}
   0.0 & -\infty & -\infty \\
   -0.5 & 0.0 & -\infty \\
   -1.0 & -0.5 & 0.0
   \end{bmatrix}$$

2. **Head 2 ($m_2 = 0.125$):**
   $$B_{i, j}^{(2)} = -0.125 \times (i - j) \implies \mathbf{B}^{(2)} = \begin{bmatrix}
   0.0 & -\infty & -\infty \\
   -0.125 & 0.0 & -\infty \\
   -0.250 & -0.125 & 0.0
   \end{bmatrix}$$

---

#### Step 2: Penalized Logit Matrices

1. **Head 1 Logits $\mathbf{S}'_{(1)} = \mathbf{S} + \mathbf{B}^{(1)}$:**
   $$\mathbf{S}'_{(1)} = \begin{bmatrix}
   2.0 + 0.0 & -\infty & -\infty \\
   1.5 - 0.5 & 3.0 + 0.0 & -\infty \\
   1.0 - 1.0 & 2.5 - 0.5 & 3.5 + 0.0
   \end{bmatrix} = \begin{bmatrix}
   2.0 & -\infty & -\infty \\
   1.0 & 3.0 & -\infty \\
   0.0 & 2.0 & 3.5
   \end{bmatrix}$$

2. **Head 2 Logits $\mathbf{S}'_{(2)} = \mathbf{S} + \mathbf{B}^{(2)}$:**
   $$\mathbf{S}'_{(2)} = \begin{bmatrix}
   2.0 + 0.0 & -\infty & -\infty \\
   1.5 - 0.125 & 3.0 + 0.0 & -\infty \\
   1.0 - 0.250 & 2.5 - 0.125 & 3.5 + 0.0
   \end{bmatrix} = \begin{bmatrix}
   2.000 & -\infty & -\infty \\
   1.375 & 3.000 & -\infty \\
   0.750 & 2.375 & 3.500
   \end{bmatrix}$$

---

#### Step 3: Softmax Probabilities for Token 3 ($i = 3$)

- **Head 1 ($m_1 = 0.5$):**
  Logits: $[0.0, 2.0, 3.5]$.
  $$\exp(0.0) = 1.000000, \quad \exp(2.0) = 7.389056, \quad \exp(3.5) = 33.115452$$
  $$\text{Sum} = 1.000000 + 7.389056 + 33.115452 = 41.504508$$
  $$A_{3, 1}^{(1)} = \frac{1.000000}{41.504508} = \mathbf{0.024094} \ (2.41\%)$$
  $$A_{3, 2}^{(1)} = \frac{7.389056}{41.504508} = \mathbf{0.178030} \ (17.80\%)$$
  $$A_{3, 3}^{(1)} = \frac{33.115452}{41.504508} = \mathbf{0.797876} \ (79.79\%)$$

- **Head 2 ($m_2 = 0.125$):**
  Logits: $[0.750, 2.375, 3.500]$.
  $$\exp(0.750) = 2.117000, \quad \exp(2.375) = 10.751012, \quad \exp(3.500) = 33.115452$$
  $$\text{Sum} = 2.117000 + 10.751012 + 33.115452 = 45.983464$$
  $$A_{3, 1}^{(2)} = \frac{2.117000}{45.983464} = \mathbf{0.046038} \ (4.60\%)$$
  $$A_{3, 2}^{(2)} = \frac{10.751012}{45.983464} = \mathbf{0.233802} \ (23.38\%)$$
  $$A_{3, 3}^{(2)} = \frac{33.115452}{45.983464} = \mathbf{0.720160} \ (72.02\%)$$

**Analysis:**
In Head 1 ($m = 0.5$), the penalty heavily suppresses the distant Token 1 ($2.41\%$). In Head 2 ($m = 0.125$), the flatter slope nearly doubles Token 1's attention weight ($4.60\%$), allowing the head to form long-range cross-token dependencies without needing positional parameter tuning.

---

### Illustration 5: NTK-Aware vs Linear Context Window Extension Wavelength and Phase Shift

**Problem:**
An LLM is pre-trained with context length $L = 2,048$, head dimension $d = 64$, and RoPE base $b = 10,000$.
To extend the context window to $L' = 8,192$ (scaling ratio $s = \frac{8192}{2048} = 4.0$):
1. Compute the modified base frequency $b_{\text{NTK}}$ under NTK-Aware RoPE scaling.
2. For the highest-frequency channel ($i = 1$) and the lowest-frequency channel ($i = 32 = d/2$), calculate the rotational frequencies $\theta_i$ and corresponding wavelengths $\lambda_i = \frac{2\pi}{\theta_i}$ under:
   - Original Pre-trained RoPE
   - Linear Position Interpolation (PI)
   - NTK-Aware RoPE
3. Compare the phase accumulated at position $m = 8,192$ and explain why Linear PI degrades local syntactic resolution while NTK-Aware succeeds.

**Solution:**

#### Step 1: Compute $b_{\text{NTK}}$

Under NTK-Aware scaling:
$$b_{\text{NTK}} = b \cdot s^{\frac{d}{d - 2}} = 10000 \times 4^{\frac{64}{64 - 2}} = 10000 \times 4^{\frac{64}{62}} = 10000 \times 4^{\frac{32}{31}}$$
$$\frac{32}{31} \approx 1.032258 \implies 4^{1.032258} \approx 4.182558$$
$$b_{\text{NTK}} = 10000 \times 4.182558 = \mathbf{41,825.58}$$

---

#### Step 2: Frequency and Wavelength Calculations

1. **Highest-Frequency Channel ($i = 1$):**
   - **Original ($b = 10000$):**
     $$\theta_1 = 10000^{-2(0)/64} = 1.000000, \quad \lambda_1 = \frac{2\pi}{1.0} \approx \mathbf{6.283 \text{ tokens}}$$
   - **Linear PI ($s = 4$):**
     $$\theta_1^{\text{PI}} = \frac{1.000000}{4} = 0.250000, \quad \lambda_1^{\text{PI}} = \frac{2\pi}{0.25} \approx \mathbf{25.133 \text{ tokens}}$$
   - **NTK-Aware ($b = 41825.58$):**
     $$\theta_1^{\text{NTK}} = (41825.58)^0 = \mathbf{1.000000}, \quad \lambda_1^{\text{NTK}} = \frac{2\pi}{1.0} \approx \mathbf{6.283 \text{ tokens}}$$

2. **Lowest-Frequency Channel ($i = 32$):**
   - Exponent: $-\frac{2(31)}{64} = -\frac{31}{32} = -0.968750$.
   - **Original ($b = 10000$):**
     $$\theta_{32} = 10000^{-0.968750} \approx 0.000133352, \quad \lambda_{32} = \frac{2\pi}{0.000133352} \approx \mathbf{47,117 \text{ tokens}}$$
   - **Linear PI ($s = 4$):**
     $$\theta_{32}^{\text{PI}} = \frac{0.000133352}{4} = \mathbf{0.000033338}, \quad \lambda_{32}^{\text{PI}} = 4 \times 47117 \approx \mathbf{188,469 \text{ tokens}}$$
   - **NTK-Aware ($b = 41825.58$):**
     $$\theta_{32}^{\text{NTK}} = (41825.58)^{-0.968750} = \left(10000 \times 4^{32/31}\right)^{-31/32} = 10000^{-31/32} \times 4^{-1} = \frac{\theta_{32}}{4} = \mathbf{0.000033338}$$
     $$\lambda_{32}^{\text{NTK}} = \mathbf{188,469 \text{ tokens}}$$

---

#### Step 3: Phase Accumulation and Architectural Tradeoff Analysis

Let us calculate the total rotation angle $\Phi = m \cdot \theta$ at target horizon $m = 8,192$:

| Method | Channel $i = 1$ Phase at $m=8192$ | Channel $i = 32$ Phase at $m=8192$ | Impact on Representation |
| :--- | :--- | :--- | :--- |
| **Original RoPE** | $8192 \times 1.0 = 8,192 \text{ rad}$ | $8192 \times 0.00013335 = 1.092 \text{ rad}$ | $1.092 > 0.273 \implies$ OOD phase angles! Perplexity explodes. |
| **Linear PI** | $8192 \times 0.25 = 2,048 \text{ rad}$ | $8192 \times 0.00003334 = \mathbf{0.273 \text{ rad}}$ | High-frequency wavelength quadrupled to $25$ tokens $\to$ **blurs local word order**! |
| **NTK-Aware** | $8192 \times 1.0 = 8,192 \text{ rad}$ | $8192 \times 0.00003334 = \mathbf{0.273 \text{ rad}}$ | **Best of both worlds**: channel 1 preserves $\lambda = 6.28$, while channel 32 matches training phase $0.273$ rad! |

**Conclusion:**
Linear Position Interpolation stretches all frequencies uniformly, destroying the model's ability to distinguish immediately adjacent words (like "not guilty" vs "guilty"). NTK-Aware RoPE dynamically interpolates low frequencies to handle long context while keeping high frequencies untouched to preserve sharp grammatical and semantic precision.

---

## 7. Deep Learning Connection & Application

### 1. Vectorized Implementation of RoPE Without Explicit Matrices
In frameworks like PyTorch and Triton, one never creates full block-diagonal matrices $\mathbf{R}_{\Theta, m} \in \mathbb{R}^{d \times d}$.
Instead, the 2D rotation of pair $[x_1, x_2]$:
$$\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1 \cos\theta - x_2 \sin\theta \\ x_1 \sin\theta + x_2 \cos\theta \end{bmatrix}$$
is vectorized across all $d$ channels via:
$$\mathbf{R}_{\Theta, m} \mathbf{x} = (\mathbf{x} \odot \cos(m \theta)) + (\operatorname{rotate\_half}(\mathbf{x}) \odot \sin(m \theta))$$
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
