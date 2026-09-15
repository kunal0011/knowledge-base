# Chapter 8.2: The Exploding and Vanishing Gradient Problem in RNNs

---

## 1. Intuition & 101 Motivation

In Chapter 8.1, we saw that Backpropagation Through Time (BPTT) unrolls a Recurrent Neural Network across $T$ temporal steps, forming a deep computational graph of depth $T$.
However, unlike a standard feedforward deep network where each layer has its own distinct weight matrix $\mathbf{W}_l$, an RNN applies the **exact same weight matrix $\mathbf{W}_{hh}$** at every single time step.

This structural repetition leads directly to the defining pathological challenge of recurrent sequence modeling:
- When calculating the influence of an early hidden state $\mathbf{h}_t$ on a distant future loss $\mathcal{L}_T$, the chain rule computes a product of $(T - t)$ Jacobian matrices.
- Just as repeatedly multiplying a scalar $w$ yields $w^{T-t}$:
  - If $|w| < 1$, $w^{100} \to 0$ (exponential decay $\to$ **Vanishing Gradient**).
  - If $|w| > 1$, $w^{100} \to \infty$ (exponential explosion $\to$ **Exploding Gradient**).

### The Fundamental Information Trade-off (Bengio et al., 1994)
Why couldn't early practitioners easily tune $\mathbf{W}_{hh}$ to avoid both extremes?
Yoshua Bengio et al. proved a mathematical dilemma:
- To store information robustly against input noise, an autonomous dynamical system must possess **attracting fixed points**, which requires the system map to be **contractive** ($\|\mathbf{J}\| < 1$).
- But contracting dynamics guarantee that backward error signals contract at the exact same exponential rate!
- Therefore, a Vanilla RNN cannot simultaneously be robust to input perturbations and capable of learning long-term temporal dependencies ($> 10-15$ steps).

---

## 2. Rigorous Mathematical Formulation

```
                 THE TEMPORAL JACOBIAN CHAIN PRODUCT
    dL/dh_T ──► [ J_T ] ──► [ J_T-1 ] ──► ... ──► [ J_t+1 ] ──► dL/dh_t
                 │            │                     │
               W_hh         W_hh                  W_hh
               
    If ||W_hh|| < 1 : Signal shrinks by factor of ||W_hh|| at every hop (Vanishing)
    If ||W_hh|| > 1 : Signal multiplies exponentially at every hop     (Exploding)
```

---

### 2.1 The Long-Term Jacobian Chain Product

Recall from Chapter 8.1 that the gradient of the loss $\mathcal{L}_T$ at time $T$ with respect to the hidden state $\mathbf{h}_t$ at an earlier time step $t < T$ is:
$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t}$$

By expanding the intermediate hidden state transitions:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^T \mathbf{J}_k$$

where each temporal Jacobian matrix $\mathbf{J}_k \in \mathbb{R}^{d_h \times d_h}$ is given by:
$$\mathbf{J}_k \equiv \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \operatorname{diag}\left(1 - \mathbf{h}_k^2\right) \mathbf{W}_{hh}$$
Letting $\mathbf{D}_k = \operatorname{diag}\left(1 - \mathbf{h}_k^2\right)$ be the diagonal matrix of $\tanh$ derivatives:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \mathbf{D}_T \mathbf{W}_{hh} \mathbf{D}_{T-1} \mathbf{W}_{hh} \dots \mathbf{D}_{t+1} \mathbf{W}_{hh} = \prod_{k=t+1}^T \left( \mathbf{D}_k \mathbf{W}_{hh} \right)$$

---

### 2.2 Upper and Lower Bounds: The Vanishing and Exploding Theorems

Let $\|\cdot\|$ denote the induced $L_2$ operator matrix norm (the maximum singular value $\sigma_{\max}$).

#### Theorem 1: Sufficient Condition for Vanishing Gradients
Using the submultiplicative property of matrix norms ($\|\mathbf{A}\mathbf{B}\| \le \|\mathbf{A}\| \|\mathbf{B}\|$):
$$\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\| \le \prod_{k=t+1}^T \|\mathbf{D}_k \mathbf{W}_{hh}\| \le \prod_{k=t+1}^T \|\mathbf{D}_k\| \|\mathbf{W}_{hh}\|$$

For the hyperbolic tangent activation function $\tanh(z)$:
$$|\tanh'(z)| = 1 - \tanh^2(z) \in (0, 1] \implies \|\mathbf{D}_k\| \le 1, \quad \forall k$$

Therefore:
$$\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\| \le \|\mathbf{W}_{hh}\|^{T - t} = \sigma_{\max}(\mathbf{W}_{hh})^{T - t}$$

**Conclusion:**
If the largest singular value of the recurrent transition matrix satisfies $\sigma_{\max}(\mathbf{W}_{hh}) < 1$:
$$\lim_{(T - t) \to \infty} \left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\| = 0$$
The gradient vanishes exponentially fast with temporal separation $(T - t)$. Early input tokens receive identically zero gradient, paralyzing long-term memory acquisition.

---

#### Theorem 2: Condition for Exploding Gradients
Now consider the case where the spectral radius $\rho(\mathbf{W}_{hh}) = \max_i |\lambda_i| > 1$.
If the pre-activations lie in the linear regime of $\tanh$ ($|a_k| \approx 0 \implies \tanh'(a_k) \approx 1 \implies \mathbf{D}_k \approx \mathbf{I}$), then:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \approx \mathbf{W}_{hh}^{T - t}$$
Let $\mathbf{v}$ be an eigenvector of $\mathbf{W}_{hh}$ corresponding to eigenvalue $\lambda$ with $|\lambda| > 1$. Then:
$$\mathbf{W}_{hh}^{T - t} \mathbf{v} = \lambda^{T - t} \mathbf{v}$$
Taking norms:
$$\left\| \mathbf{W}_{hh}^{T - t} \mathbf{v} \right\| = |\lambda|^{T - t} \|\mathbf{v}\| \xrightarrow{(T - t) \to \infty} \infty$$

**Conclusion:**
If $\rho(\mathbf{W}_{hh}) > 1$ and states do not heavily saturate, gradient norms grow exponentially:
$$\lim_{(T - t) \to \infty} \left\| \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_t} \right\| = \infty$$
During SGD optimization, an exploding gradient causes parameter updates $\Delta \mathbf{W} = -\eta \mathbf{g}$ to take gigantic steps, completely destroying learned representations and causing floating-point overflow (`NaN` or `Inf`).

---

### 2.3 Pascanu's Gradient Clipping by Norm (Pascanu et al., 2013)

To stabilize training in the presence of exploding gradients without artificially limiting sequence length, Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio introduced **Gradient Clipping by Norm**.

Let $\mathbf{g} = \nabla_{\boldsymbol{\theta}} \mathcal{L}$ be the concatenated gradient vector of all trainable parameters.
Given a maximum allowable gradient norm threshold $\theta_{\text{clip}} > 0$:

$$\mathbf{g}_{\text{clipped}} = \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \le \theta_{\text{clip}} \\ \theta_{\text{clip}} \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \theta_{\text{clip}} \end{cases}$$

#### Why Norm-based Clipping is Superior to Element-wise Value Clipping:
1. **Direction Preservation:**
   $$\frac{\mathbf{g}_{\text{clipped}}}{\|\mathbf{g}_{\text{clipped}}\|} = \frac{\mathbf{g}}{\|\mathbf{g}\|}$$
   Norm clipping rescales the step size while keeping the descent direction **$100\%$ identical** to the true steepest descent direction.
2. **Value Clipping Ruin:**
   If one instead clips component-wise ($\text{clip}(g_i, -c, c)$), the angle between the clipped vector and the true gradient can approach $90^\circ$, causing the optimizer to move in an arbitrary, incorrect direction.

```
       Norm Clipping vs. Element-wise Value Clipping
                   g (True Gradient)
                  ▲
                 /
                /  <-- Same direction, scaled magnitude
               /
              * g_clipped (Norm Clipping)
             /
            /
           O ──────────────► g_value_clipped (Direction altered!)
```

---

### 2.4 Orthogonal & Unitary Weight Initialization

Can we prevent vanishing gradients at initialization?
If we initialize $\mathbf{W}_{hh}$ as an **orthogonal matrix**:
$$\mathbf{W}_{hh}^T \mathbf{W}_{hh} = \mathbf{I}$$
Then all singular values are identically equal to one:
$$\sigma_i(\mathbf{W}_{hh}) = 1, \quad \forall i \in \{1, \dots, d_h\}$$

An orthogonal transformation preserves Euclidean vector norms:
$$\|\mathbf{W}_{hh} \mathbf{v}\|_2 = \|\mathbf{v}\|_2$$
When backpropagating through the linear component:
$$\|\mathbf{W}_{hh}^{T - t} \boldsymbol{\delta}\|_2 = \|\boldsymbol{\delta}\|_2$$
Gradients neither explode nor vanish through the linear transition!
*(Note: While orthogonal initialization mitigates gradient decay at step 0, as training progresses and non-linearities saturate, gating mechanisms like LSTMs/GRUs become necessary to maintain constant gradient highways).*

---

## 3. Geometric & Algebraic Interpretation

### The Eigenspectrum on the Complex Unit Circle

Consider the eigendecomposition of the recurrent matrix $\mathbf{W}_{hh} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}$, where $\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_{d_h})$ with $\lambda_i \in \mathbb{C}$:

$$\mathbf{W}_{hh}^k = \mathbf{Q} \begin{bmatrix} \lambda_1^k & & 0 \\ & \ddots & \\ 0 & & \lambda_{d_h}^k \end{bmatrix} \mathbf{Q}^{-1}$$

The complex plane partitions the dynamical behavior into three distinct regimes relative to the **unit circle** $\{z \in \mathbb{C} : |z| = 1\}$:
1. **Inside the Unit Circle ($|\lambda_i| < 1$):**
   $\lambda_i^k \to 0$ as $k \to \infty$. Defines the *stable subspace*. Gradients in this subspace vanish exponentially.
2. **Outside the Unit Circle ($|\lambda_i| > 1$):**
   $\lambda_i^k \to \infty$ as $k \to \infty$. Defines the *unstable subspace*. Gradients in this subspace explode exponentially.
3. **On the Unit Circle ($|\lambda_i| = 1$):**
   $|\lambda_i^k| = 1$ for all $k$. Defines the *center manifold*. Memory is preserved without amplification or decay (pure rotation in complex space).

```
                 COMPLEX EIGENSPECTRUM PLANE
                           Im(λ)
                             ▲
                             │   * (Unstable: |λ| > 1 -> Explodes)
                         ┌───┼───┐
                       ┌─┘   │   └─┐
                      ┌┘     │     └┐
                     ─┼──────┼──────┼─► Re(λ)
                      └┐  *  │     ┌┘
                       └─┐(Stable)─┘
                         └───┼───┘
                             │  Unit Circle (|λ| = 1)
```

---

## 4. Real-World Analogy

### The Public Address Feedback Loop vs. The Whispering Gallery
- **Exploding Gradient (The Acoustic Feedback Loop):**
  A singer accidentally points a live microphone at an amplifier speaker. The sound enters the mic, gets amplified by factor $G > 1$, exits the speaker, re-enters the mic, and multiplies again: $G \to G^2 \to G^3 \dots$ Within a second, the sound system produces a deafening, system-destroying acoustic screech (`NaN`). Gradient clipping is the electronic limiter that instantly clamps the audio signal before the speakers blow up.
- **Vanishing Gradient (The Telephone Game):**
  A message is whispered through a line of 100 people. Each person speaks at $90\%$ of the volume of the person before them ($0.9$). By person 50, the volume is $0.9^{50} \approx 0.005$; the original signal has dropped below the threshold of human hearing. No matter how important the first person's message was, the 100th person hears only silence.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace concrete numerical walkthroughs of both vanishing/exploding dynamics and gradient clipping arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Gradient Stability |
| :--- | :--- | :--- | :--- |
| $T$ | Temporal Horizon | Scalar ($4$) | Number of time steps across which gradients backpropagate |
| $w$ | Scalar Recurrent Weight | Scalar | $1$D recurrent transition parameter ($h_t = \tanh(w \cdot h_{t-1})$) |
| $J_k$ | Temporal Jacobian | Scalar | $J_k = (1 - h_k^2) \cdot w$ |
| $\frac{\partial h_4}{\partial h_0}$ | Accumulated Gradient | Scalar | $\prod_{k=1}^4 J_k = \text{sensitivity of } h_4 \text{ w.r.t. } h_0$ |
| $\mathbf{g}$ | Unclipped Gradient Vector | $(2,)$ | Raw parameter gradient vector entering optimizer |
| $\|\mathbf{g}\|_2$ | Gradient Norm | Scalar | Euclidean magnitude $\sqrt{g_1^2 + g_2^2}$ |
| $\theta_{\text{clip}}$ | Clipping Threshold | Scalar ($2.0$) | Maximum permitted Euclidean norm |
| $\mathbf{g}_{\text{clipped}}$ | Clipped Gradient Vector | $(2,)$ | Rescaled gradient vector ($\theta \frac{\mathbf{g}}{\|\mathbf{g}\|}$) |

---

### 5.2 Concrete Toy Numbers: Vanishing vs. Exploding Tracing ($T=4$)

Assume inputs $\mathbf{x} = \mathbf{0}$ and initial state $h_0 = 0$, so that all hidden states remain at the origin:
$$h_t = 0 \implies \tanh'(0) = 1 - (0)^2 = 1.0, \quad \forall t$$
Thus the Jacobian at each step simplifies to $J_k = 1.0 \times w = w$.

The gradient backpropagated across 4 time steps is:
$$\frac{\partial h_4}{\partial h_0} = \prod_{k=1}^4 J_k = w^4$$

#### Case A: Sub-unitary Weight $w = 0.5$ (Vanishing Regime)
- Step 4 to 3: $J_4 = 0.5$
- Step 4 to 2: $J_4 \cdot J_3 = (0.5)^2 = \mathbf{0.25}$
- Step 4 to 1: $J_4 \cdot J_3 \cdot J_2 = (0.5)^3 = \mathbf{0.125}$
- Step 4 to 0: $J_4 \cdot J_3 \cdot J_2 \cdot J_1 = (0.5)^4 = \mathbf{0.0625}$
Across just 4 steps, the gradient has decayed by **$93.75\%$**!
For $T = 20$: $(0.5)^{20} \approx 9.53 \times 10^{-7}$ (effective extinction).

#### Case B: Super-unitary Weight $w = 2.0$ (Exploding Regime)
- Step 4 to 3: $J_4 = 2.0$
- Step 4 to 2: $J_4 \cdot J_3 = (2.0)^2 = \mathbf{4.0}$
- Step 4 to 1: $J_4 \cdot J_3 \cdot J_2 = (2.0)^3 = \mathbf{8.0}$
- Step 4 to 0: $J_4 \cdot J_3 \cdot J_2 \cdot J_1 = (2.0)^4 = \mathbf{16.0}$
Across just 4 steps, the gradient has amplified by **$1,600\%$**!
For $T = 20$: $(2.0)^{20} \approx 1,048,576$ (massive gradient explosion).

---

### 5.3 Norm-based Gradient Clipping Arithmetic

Suppose backpropagation on a mini-batch yields an exploding parameter gradient vector:
$$\mathbf{g} = \begin{bmatrix} 6.0 \\ 8.0 \end{bmatrix}$$
Let the maximum allowed norm threshold be $\theta_{\text{clip}} = 2.0$.

1. **Compute Euclidean Norm $\|\mathbf{g}\|_2$:**
   $$\|\mathbf{g}\|_2 = \sqrt{(6.0)^2 + (8.0)^2} = \sqrt{36.0 + 64.0} = \sqrt{100.0} = \mathbf{10.0}$$

2. **Check Clipping Condition:**
   $$\|\mathbf{g}\|_2 = 10.0 > \theta_{\text{clip}} = 2.0 \implies \text{Condition triggered!}$$

3. **Compute Scaling Factor $\alpha$:**
   $$\alpha = \frac{\theta_{\text{clip}}}{\|\mathbf{g}\|_2} = \frac{2.0}{10.0} = \mathbf{0.2}$$

4. **Compute Clipped Gradient $\mathbf{g}_{\text{clipped}} = \alpha \mathbf{g}$:**
   $$g_{\text{clipped}, 1} = 0.2 \times 6.0 = \mathbf{1.2}$$
   $$g_{\text{clipped}, 2} = 0.2 \times 8.0 = \mathbf{1.6}$$
   $$\mathbf{g}_{\text{clipped}} = \begin{bmatrix} 1.2 \\ 1.6 \end{bmatrix}$$

5. **Verify Clipped Norm:**
   $$\|\mathbf{g}_{\text{clipped}}\|_2 = \sqrt{(1.2)^2 + (1.6)^2} = \sqrt{1.44 + 2.56} = \sqrt{4.00} = \mathbf{2.0} \equiv \theta_{\text{clip}}$$

The direction angle $\theta = \arctan(8/6) \approx 53.13^\circ$ is preserved exactly, while the magnitude is strictly restrained to $2.0$.

---

## 6. Solved Illustrations

### Illustration 1: Why ReLU in Recurrent Layers Causes Severe Catastrophic Explosion

**Problem:**
ReLU ($\sigma(z) = \max(0, z)$) revolutionized feedforward ConvNets by solving vanishing gradients. Why does replacing $\tanh$ with standard ReLU in a Vanilla RNN frequently cause catastrophic numerical explosion?

**Solution:**
For $\tanh(z)$, the derivative satisfies $|\tanh'(z)| \le 1$, and as activations grow large ($|z| \gg 1$), $\tanh'(z) \to 0$, providing a natural saturating "brake" against explosive growth.
For ReLU:
$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
Whenever neurons are active ($z > 0$), $\sigma'(z) \equiv 1.0$.
Thus:
$$\mathbf{J}_k = \mathbf{I} \cdot \mathbf{W}_{hh} = \mathbf{W}_{hh}$$
If $\sigma_{\max}(\mathbf{W}_{hh}) > 1$, there is **zero saturation damping**. Activations in the forward pass grow without bound ($h_t \sim \mathcal{O}(\lambda^t)$), and gradients in the backward pass explode without bound.
*(To use ReLU safely in an RNN, Quoc Le et al., 2015 introduced the IRNN, requiring $\mathbf{W}_{hh}$ to be initialized strictly to the Identity matrix $\mathbf{I}$ with tiny learning rates).*

---

### Illustration 2: Value Clipping vs. Norm Clipping Directional Distortion

**Problem:**
Let $\mathbf{g} = [100.0, 1.0]^T$.
Compare the resulting vector under:
1. Element-wise value clipping with bounds $[-5.0, 5.0]$.
2. Norm clipping with threshold $\theta = 5.0$.
Compute the cosine similarity of each clipped vector with the original gradient $\mathbf{g}$.

**Solution:**
1. **Original Direction:**
   $$\|\mathbf{g}\| = \sqrt{100^2 + 1^2} \approx 100.005$$
   $$\mathbf{u}_{\text{orig}} = \frac{\mathbf{g}}{\|\mathbf{g}\|} \approx [0.99995, 0.0099995]^T$$

2. **Norm Clipping ($\theta = 5.0$):**
   $$\mathbf{g}_{\text{norm}} = 5.0 \frac{\mathbf{g}}{\|\mathbf{g}\|} \approx [4.99975, 0.0499975]^T$$
   $$\text{Cosine Similarity} = \frac{\mathbf{g}^T \mathbf{g}_{\text{norm}}}{\|\mathbf{g}\| \|\mathbf{g}_{\text{norm}}\|} = \mathbf{1.0000} \quad (\text{Perfect preservation})$$

3. **Value Clipping (clamp between $[-5, 5]$):**
   $$\mathbf{g}_{\text{value}} = [\text{clip}(100, -5, 5), \text{clip}(1, -5, 5)]^T = [5.0, 1.0]^T$$
   $$\|\mathbf{g}_{\text{value}}\| = \sqrt{5^2 + 1^2} = \sqrt{26} \approx 5.099$$
   $$\text{Cosine Similarity} = \frac{100(5) + 1(1)}{100.005 \times 5.099} = \frac{501}{509.925} \approx \mathbf{0.9825}$$
   Angle of deflection: $\theta = \arccos(0.9825) \approx 10.74^\circ$.

Value clipping shifted the direction by over $10^\circ$, penalizing large components while leaving small components unscaled. In high dimensions ($d = 10,000$), value clipping can rotate the gradient by up to $80^\circ$ away from the true steepest descent direction!

---

## 7. Deep Learning Connection & Application

### 1. The Historical Catalyst for Gated Networks (LSTM & GRU)
The mathematical impossibility of training Vanilla RNNs across long horizons directly drove Sepp Hochreiter and Jürgen Schmidhuber (1997) to invent the **Long Short-Term Memory (LSTM)** network.
The core architectural insight of LSTM is to replace the multiplicative Jacobian chain:
$$\prod_{k=t+1}^T \mathbf{D}_k \mathbf{W}_{hh}$$
with an additive **Constant Error Carousel (CEC)**:
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
where the cell state $\mathbf{C}_t$ updates via linear addition. When the forget gate $\mathbf{f}_t \approx \mathbf{1}$, the Jacobian $\frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_t} \approx \mathbf{I}$, creating an unimpeded gradient superhighway spanning hundreds of time steps.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Pure NumPy simulations comparing gradient norm decay ($w < 1$) and explosion ($w > 1$) across sequence lengths $T \in [2, 5, 10, 20, 50]$.
2. Exact numerical verification of the Part 5 Visual Grid hand arithmetic and gradient clipping equations.
3. Scratch implementation of Pascanu's Norm-based Gradient Clipping compared against PyTorch's `torch.nn.utils.clip_grad_norm_`.
4. Long-term gradient preservation experiment comparing Random Gaussian Initialization vs. Orthogonal Initialization.

See implementation in:
[`08_recurrent_networks/code/02_exploding_and_vanishing_gradients.py`](./code/02_exploding_and_vanishing_gradients.py)
