# Chapter 8.1: Sequence Processing, Autoregressive Models & Vanilla RNN (BPTT)

---

## 1. Intuition & 101 Motivation

Feedforward networks and Convolutional Neural Networks operate under the foundational assumption that input samples are **independent and identically distributed (I.I.D.)**:
$$\mathcal{D} = \{(\mathbf{x}^{(1)}, \mathbf{y}^{(1)}), (\mathbf{x}^{(2)}, \mathbf{y}^{(2)}), \dots, (\mathbf{x}^{(N)}, \mathbf{y}^{(N)})\}$$
Processing sample $\mathbf{x}^{(i)}$ has zero mathematical dependence on sample $\mathbf{x}^{(i-1)}$.

However, natural signals in the physical universe—human speech, written language, DNA nucleotides, musical compositions, algorithmic trades, and robot sensory streams—are inherently **temporal sequences**:
$$\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$$
where the meaning, syntax, and truth value of token $\mathbf{x}_t$ depend inextricably on the preceding context $(\mathbf{x}_1, \dots, \mathbf{x}_{t-1})$.

### The Challenges of Sequential Modeling
1. **Variable Input Length:** Sentences vary from 3 to 1,000 words; feedforward networks require fixed-dimensional vector inputs.
2. **Long-Range Dependencies:** A grammatical subject introduced at $t=1$ dictates the pluralization of a verb at $t=50$.
3. **Temporal Invariance:** The phrase *"the cat sat"* has the same semantic syntax whether it appears at the start, middle, or end of a paragraph.

### The Autoregressive Factorization
By the probability chain rule, any joint distribution over sequences factorizes without loss of generality into an autoregressive product of conditional probabilities:
$$p(\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T) = \prod_{t=1}^T p(\mathbf{x}_t \mid \mathbf{x}_1, \dots, \mathbf{x}_{t-1}) = \prod_{t=1}^T p(\mathbf{x}_t \mid \mathbf{x}_{<t})$$

To compute this without maintaining an exponentially growing history of past inputs, the Recurrent Neural Network (RNN) introduces a **hidden state vector** $\mathbf{h}_t \in \mathbb{R}^{d_h}$:
$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$$
The hidden state $\mathbf{h}_t$ acts as a lossy, fixed-dimensional dynamical memory that recursively updates with each incoming token.

---

## 2. Rigorous Mathematical Formulation

```
      UNFOLDED RECURRENT NEURAL NETWORK (COMPUTATIONAL GRAPH)
       L_1                  L_2                  L_T
        ▲                    ▲                    ▲
        │                    │                    │
       y_hat_1              y_hat_2              y_hat_T
        ▲                    ▲                    ▲
        │ W_hy               │ W_hy               │ W_hy
   ┌────┴─────┐  W_hh   ┌────┴─────┐  W_hh   ┌────┴─────┐
h_0│   h_1    │────────►│   h_2    │── ... ─►│   h_T    │
──►│          │         │          │         │          │
   └────▲─────┘         └────▲─────┘         └────▲─────┘
        │ W_xh               │ W_xh               │ W_xh
        │                    │                    │
       x_1                  x_2                  x_T
```

---

### 2.1 The Forward Pass (Elman RNN Formulation)

Let $\mathbf{x} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ be a sequence of input vectors $\mathbf{x}_t \in \mathbb{R}^{d_x}$.
Let $\mathbf{h}_t \in \mathbb{R}^{d_h}$ be the hidden state vector at time step $t$, initialized with $\mathbf{h}_0 = \mathbf{0}$.

At each time step $t \in \{1, \dots, T\}$:
1. **Pre-activation Linear Combination:**
   $$\mathbf{a}_t = \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h$$
   where:
   - $\mathbf{W}_{hh} \in \mathbb{R}^{d_h \times d_h}$ is the recurrent hidden-to-hidden transition weight matrix,
   - $\mathbf{W}_{xh} \in \mathbb{R}^{d_h \times d_x}$ is the input-to-hidden projection weight matrix,
   - $\mathbf{b}_h \in \mathbb{R}^{d_h}$ is the hidden bias vector.

2. **Non-linear Activation:**
   $$\mathbf{h}_t = \tanh(\mathbf{a}_t)$$

3. **Output Emission (for sequence-to-sequence or language modeling):**
   $$\mathbf{z}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$
   $$\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{z}_t) \in \mathbb{R}^{d_y}$$
   where $\mathbf{W}_{hy} \in \mathbb{R}^{d_y \times d_h}$ and $\mathbf{b}_y \in \mathbb{R}^{d_y}$.

4. **Sequence Objective Loss Function:**
   $$\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t(\hat{\mathbf{y}}_t, \mathbf{y}_t)$$
   For multi-class classification / language modeling, $\mathcal{L}_t$ is the cross-entropy loss:
   $$\mathcal{L}_t = - \sum_{k=1}^{d_y} y_{t, k} \log \hat{y}_{t, k}$$

---

### 2.2 Backpropagation Through Time (BPTT) Derivation

Because parameters $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{b}_h, \mathbf{W}_{hy}, \mathbf{b}_y$ are **shared identically across all time steps**, the total gradient w.r.t. any parameter $\theta$ is the sum of its partial gradients across time:
$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \theta}$$

---

#### 1. Output Layer Gradients ($\mathbf{W}_{hy}, \mathbf{b}_y$)
For cross-entropy loss with softmax activation, the error at time step $t$ is:
$$\boldsymbol{\delta}_t^y \equiv \frac{\partial \mathcal{L}_t}{\partial \mathbf{z}_t} = \hat{\mathbf{y}}_t - \mathbf{y}_t \in \mathbb{R}^{d_y}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hy}} = \sum_{t=1}^T \boldsymbol{\delta}_t^y \, \mathbf{h}_t^T \in \mathbb{R}^{d_y \times d_h}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_y} = \sum_{t=1}^T \boldsymbol{\delta}_t^y \in \mathbb{R}^{d_y}$$

---

#### 2. Hidden State Recurrent Gradients ($\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$)
At any time step $t$, the hidden state $\mathbf{h}_t$ influences the loss in two distinct pathways:
1. **Immediately:** through the output prediction $\hat{\mathbf{y}}_t$ and loss $\mathcal{L}_t$.
2. **Into the Future:** through the next hidden state $\mathbf{h}_{t+1}$, which influences all future losses $\sum_{k=t+1}^T \mathcal{L}_k$.

Applying the multivariable chain rule backwards in time from $t = T, T-1, \dots, 1$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \underbrace{\mathbf{W}_{hy}^T \boldsymbol{\delta}_t^y}_{\text{Direct loss contribution at } t} + \underbrace{\left( \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}}_{\text{Recurrent gradient from future } t+1}$$

To compute the Jacobian $\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$:
Recall: $\mathbf{h}_{t+1} = \tanh(\mathbf{a}_{t+1})$, so $\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{a}_{t+1}} = \operatorname{diag}\left(1 - \mathbf{h}_{t+1}^2\right)$.
Since $\mathbf{a}_{t+1} = \mathbf{W}_{hh} \mathbf{h}_t + \mathbf{W}_{xh} \mathbf{x}_{t+1} + \mathbf{b}_h$, we have $\frac{\partial \mathbf{a}_{t+1}}{\partial \mathbf{h}_t} = \mathbf{W}_{hh}$.

By the chain rule:
$$\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \operatorname{diag}\left(1 - \mathbf{h}_{t+1}^2\right) \mathbf{W}_{hh}$$

Therefore, the backward recurrence relation for the hidden gradient is:
$$\boldsymbol{\delta}_t^h \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \mathbf{W}_{hy}^T \boldsymbol{\delta}_t^y + \mathbf{W}_{hh}^T \left( \boldsymbol{\delta}_{t+1}^h \odot (1 - \mathbf{h}_{t+1}^2) \right)$$
with the terminal boundary condition at $t = T$:
$$\boldsymbol{\delta}_T^h = \mathbf{W}_{hy}^T \boldsymbol{\delta}_T^y$$

---

#### 3. Pre-activation Gradient ($\frac{\partial \mathcal{L}}{\partial \mathbf{a}_t}$)
Define:
$$\boldsymbol{\delta}_t^a \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_t} = \boldsymbol{\delta}_t^h \odot (1 - \mathbf{h}_t^2) \in \mathbb{R}^{d_h}$$

---

#### 4. Recurrent and Input Weight Gradients ($\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{b}_h$)
Accumulating across all time steps:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \boldsymbol{\delta}_t^a \, \mathbf{h}_{t-1}^T \in \mathbb{R}^{d_h \times d_h}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \sum_{t=1}^T \boldsymbol{\delta}_t^a \, \mathbf{x}_t^T \in \mathbb{R}^{d_h \times d_x}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_h} = \sum_{t=1}^T \boldsymbol{\delta}_t^a \in \mathbb{R}^{d_h}$$

---

### 2.3 Truncated Backpropagation Through Time (TBPTT)

For sequence lengths $T \ge 1,000$, standard BPTT suffers from:
1. **Memory Exhaustion:** Storing all $T$ hidden state vectors $\{\mathbf{h}_1, \dots, \mathbf{h}_T\}$ in memory consumes $O(T \cdot B \cdot d_h)$ GPU RAM.
2. **Severe Gradient Instability:** Backpropagating across 1,000 time steps entails 1,000 successive matrix multiplications with $\mathbf{W}_{hh}^T$, causing gradients to vanish or explode exponentially.

**TBPTT Protocol:**
- Unroll the forward pass continuously across long text, maintaining the hidden state $\mathbf{h}_t$.
- Truncate the backward pass to a fixed horizon of $k_1$ steps.
- Update weights after every $k_2$ steps.
- While TBPTT solves the memory bottleneck, it destroys the network's mathematical capacity to learn statistical dependencies spanning longer than $k_1$ time steps.

---

## 3. Geometric & Algebraic Interpretation

### The Discrete-Time Non-Linear Dynamical System
An RNN is fundamentally an autonomous non-linear dynamical map:
$$\mathbf{h}_t = \Phi(\mathbf{h}_{t-1}; \mathbf{x}_t)$$
Operating in hidden state phase space $\mathbb{R}^{d_h}$:
- **Fixed Points:** Coordinates $\mathbf{h}^*$ satisfying $\mathbf{h}^* = \tanh(\mathbf{W}_{hh} \mathbf{h}^* + \mathbf{b}_h)$ when input $\mathbf{x} = \mathbf{0}$.
- **Attractors:** Valleys in phase space where trajectories converge. In memory models, distinct attractors represent distinct memorized concepts (Hopfield network analogy).
- **Jacobian Spectrum:** The stability of the dynamical trajectory is governed by the eigenvalues of the Jacobian matrix:
  $$\mathbf{J}_t = \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \operatorname{diag}(1 - \mathbf{h}_t^2) \mathbf{W}_{hh}$$
  If the spectral radius $\rho(\mathbf{J}) > 1$, trajectories diverge (chaotic regime / exploding gradients). If $\rho(\mathbf{J}) < 1$, trajectories contract toward the origin (dissipative regime / vanishing gradients).

---

## 4. Real-World Analogy

### The Courtroom Stenographer vs. The Snapshot Photographer
- **Feedforward / CNN (The Photographer):** Takes a single instant photograph of a static room. Every detail must be visible in that single frame. The camera has no memory of what happened 5 minutes ago.
- **RNN (The Courtroom Stenographer):** The stenographer listens to continuous spoken testimony over 6 hours.
  - They cannot record every raw audio waveform in short-term working memory.
  - Instead, after every spoken word $\mathbf{x}_t$, they update their running mental model $\mathbf{h}_t$ ("The witness just contradicted their earlier statement").
  - When the judge asks at 5:00 PM "Why did the witness pause at 10:00 AM?", the error signal $\boldsymbol{\delta}$ must trace backwards through thousands of sequential mental states. If their memory degraded at 11:00 AM, the historical connection is permanently severed.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a full forward pass and complete BPTT backward pass on a concrete toy sequence of length $T = 2$.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Vanilla RNN |
| :--- | :--- | :--- | :--- |
| $T$ | Sequence Length | Scalar ($2$) | Number of temporal time steps |
| $\mathbf{x}_1, \mathbf{x}_2$ | Input Vectors | $(2,)$ | Inputs at time steps $t=1$ and $t=2$ |
| $\mathbf{h}_0$ | Initial Hidden State | $(2,)$ | Starting state (initialized to $[0, 0]^T$) |
| $\mathbf{W}_{xh}$ | Input-to-Hidden Weights | $(2, 2)$ | Projects input $\mathbf{x}_t$ into hidden space |
| $\mathbf{W}_{hh}$ | Hidden-to-Hidden Weights | $(2, 2)$ | Recurrent state transition matrix |
| $\mathbf{a}_1, \mathbf{a}_2$ | Pre-activations | $(2,)$ | Linear combinations before $\tanh$ |
| $\mathbf{h}_1, \mathbf{h}_2$ | Hidden State Activations | $(2,)$ | Hidden memories $\mathbf{h}_t = \tanh(\mathbf{a}_t)$ |
| $\mathbf{W}_{hy}$ | Hidden-to-Output Weights | $(2, 2)$ | Maps hidden state to output logits |
| $\mathbf{z}_1, \mathbf{z}_2$ | Output Logits | $(2,)$ | Unnormalized scores for prediction |
| $\boldsymbol{\delta}_t^z \equiv \hat{\mathbf{y}}_t - \mathbf{y}_t$ | Output Loss Gradient | $(2,)$ | Error at output time step $t$ |
| $\boldsymbol{\delta}_t^h \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$ | Hidden Gradient | $(2,)$ | Sensitivity of total loss w.r.t. hidden state at $t$ |
| $\boldsymbol{\delta}_t^a \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_t}$ | Pre-activation Gradient | $(2,)$ | $\boldsymbol{\delta}_t^h \odot (1 - \mathbf{h}_t^2)$ |

---

### 5.2 Concrete Toy Numbers

#### Dimensions:
$d_x = 2, \quad d_h = 2, \quad d_y = 2, \quad T = 2$

#### Inputs:
$$\mathbf{x}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}, \quad \mathbf{h}_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

#### Model Weights (biases $= \mathbf{0}$ for crystal clarity):
$$\mathbf{W}_{xh} = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}, \quad \mathbf{W}_{hh} = \begin{bmatrix} 0.4 & 0.1 \\ 0.0 & 0.4 \end{bmatrix}, \quad \mathbf{W}_{hy} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$$

#### Targets:
At each time step, suppose downstream targets and softmax produce output errors:
$$\boldsymbol{\delta}_1^z \equiv \frac{\partial \mathcal{L}_1}{\partial \mathbf{z}_1} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}, \quad \boldsymbol{\delta}_2^z \equiv \frac{\partial \mathcal{L}_2}{\partial \mathbf{z}_2} = \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix}$$

---

### 5.3 Forward Pass Arithmetic

#### Time Step $t = 1$:
1. **Pre-activation $\mathbf{a}_1 = \mathbf{W}_{hh} \mathbf{h}_0 + \mathbf{W}_{xh} \mathbf{x}_1$:**
   $$\mathbf{W}_{hh} \mathbf{h}_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
   $$\mathbf{W}_{xh} \mathbf{x}_1 = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix}$$
   $$\mathbf{a}_1 = \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix}$$

2. **Hidden State $\mathbf{h}_1 = \tanh(\mathbf{a}_1)$:**
   Using $\tanh(0.5) \approx 0.462117$, $\tanh(0.0) = 0.0$:
   $$\mathbf{h}_1 = \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix}$$

3. **Output Logits $\mathbf{z}_1 = \mathbf{W}_{hy} \mathbf{h}_1$:**
   $$\mathbf{z}_1 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix}$$

---

#### Time Step $t = 2$:
1. **Pre-activation $\mathbf{a}_2 = \mathbf{W}_{hh} \mathbf{h}_1 + \mathbf{W}_{xh} \mathbf{x}_2$:**
   $$\mathbf{W}_{hh} \mathbf{h}_1 = \begin{bmatrix} 0.4 & 0.1 \\ 0.0 & 0.4 \end{bmatrix} \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.4 \times 0.462117 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.184847 \\ 0.0 \end{bmatrix}$$
   $$\mathbf{W}_{xh} \mathbf{x}_2 = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.5 \end{bmatrix}$$
   $$\mathbf{a}_2 = \begin{bmatrix} 0.184847 + 0.0 \\ 0.0 + 0.5 \end{bmatrix} = \begin{bmatrix} 0.184847 \\ 0.5 \end{bmatrix}$$

2. **Hidden State $\mathbf{h}_2 = \tanh(\mathbf{a}_2)$:**
   Using $\tanh(0.184847) \approx 0.182773$, $\tanh(0.5) \approx 0.462117$:
   $$\mathbf{h}_2 = \begin{bmatrix} 0.182773 \\ 0.462117 \end{bmatrix}$$

3. **Output Logits $\mathbf{z}_2 = \mathbf{W}_{hy} \mathbf{h}_2$:**
   $$\mathbf{z}_2 = \begin{bmatrix} 0.182773 \\ 0.462117 \end{bmatrix}$$

---

### 5.4 Backward Pass (BPTT) Arithmetic

#### 1. Backward at Time Step $t = 2$:
- **Hidden Gradient at $t = 2$:**
  Since $t=2$ is the terminal step ($T=2$), there is no future gradient:
  $$\boldsymbol{\delta}_2^h = \mathbf{W}_{hy}^T \boldsymbol{\delta}_2^z = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix}$$

- **Pre-activation Gradient at $t = 2$ ($\boldsymbol{\delta}_2^a = \boldsymbol{\delta}_2^h \odot (1 - \mathbf{h}_2^2)$):**
  $$1 - h_{2, 1}^2 = 1 - (0.182773)^2 = 1 - 0.033406 = 0.966594$$
  $$1 - h_{2, 2}^2 = 1 - (0.462117)^2 = 1 - 0.213552 = 0.786448$$
  $$\boldsymbol{\delta}_2^a = \begin{bmatrix} 0.2 \times 0.966594 \\ 0.1 \times 0.786448 \end{bmatrix} = \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix}$$

- **Parameter Gradient Contributions at $t = 2$:**
  $$\frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{hh}} = \boldsymbol{\delta}_2^a \, \mathbf{h}_1^T = \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix} \begin{bmatrix} 0.462117 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.089336 & 0.0 \\ 0.036343 & 0.0 \end{bmatrix}$$
  $$\frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{xh}} = \boldsymbol{\delta}_2^a \, \mathbf{x}_2^T = \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix} \begin{bmatrix} 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.0 & 0.193319 \\ 0.0 & 0.078645 \end{bmatrix}$$

---

#### 2. Backward at Time Step $t = 1$ (The Temporal Recurrence):
- **Hidden Gradient at $t = 1$ ($\boldsymbol{\delta}_1^h$):**
  $$\boldsymbol{\delta}_1^h = \underbrace{\mathbf{W}_{hy}^T \boldsymbol{\delta}_1^z}_{\text{Direct from } \mathcal{L}_1} + \underbrace{\mathbf{W}_{hh}^T \boldsymbol{\delta}_2^a}_{\text{Recurrent from } t=2}$$
  $$\text{Direct Term} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$
  $$\text{Recurrent Term} = \begin{bmatrix} 0.4 & 0.0 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix} = \begin{bmatrix} 0.4(0.193319) \\ 0.1(0.193319) + 0.4(0.078645) \end{bmatrix} = \begin{bmatrix} 0.077328 \\ 0.019332 + 0.031458 \end{bmatrix} = \begin{bmatrix} 0.077328 \\ 0.050790 \end{bmatrix}$$
  $$\boldsymbol{\delta}_1^h = \begin{bmatrix} 0.5 + 0.077328 \\ -0.5 + 0.050790 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.577328 \\ -0.449210 \end{bmatrix}}$$

- **Pre-activation Gradient at $t = 1$ ($\boldsymbol{\delta}_1^a = \boldsymbol{\delta}_1^h \odot (1 - \mathbf{h}_1^2)$):**
  $$1 - h_{1, 1}^2 = 1 - (0.462117)^2 = 0.786448$$
  $$1 - h_{1, 2}^2 = 1 - 0.0 = 1.0$$
  $$\boldsymbol{\delta}_1^a = \begin{bmatrix} 0.577328 \times 0.786448 \\ -0.449210 \times 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.454038 \\ -0.449210 \end{bmatrix}}$$

- **Parameter Gradient Contributions at $t = 1$:**
  $$\frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{hh}} = \boldsymbol{\delta}_1^a \, \mathbf{h}_0^T = \boldsymbol{\delta}_1^a \begin{bmatrix} 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$$
  $$\frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{xh}} = \boldsymbol{\delta}_1^a \, \mathbf{x}_1^T = \begin{bmatrix} 0.454038 \\ -0.449210 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.454038 & 0.0 \\ -0.449210 & 0.0 \end{bmatrix}$$

---

#### 3. Total Parameter Gradients (Accumulation Over Time):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{hh}} + \frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{hh}} = \begin{bmatrix} 0.089336 & 0.0 \\ 0.036343 & 0.0 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{xh}} + \frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{xh}} = \begin{bmatrix} 0.454038 & 0.193319 \\ -0.449210 & 0.078645 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hy}} = \boldsymbol{\delta}_1^z \mathbf{h}_1^T + \boldsymbol{\delta}_2^z \mathbf{h}_2^T = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} [0.462117, 0] + \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix} [0.182773, 0.462117]$$
$$= \begin{bmatrix} 0.231059 & 0 \\ -0.231059 & 0 \end{bmatrix} + \begin{bmatrix} 0.036555 & 0.092423 \\ 0.018277 & 0.046212 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.267614 & 0.092423 \\ -0.212782 & 0.046212 \end{bmatrix}}$$

Every calculation above is exact and mathematically transparent.

---

## 6. Solved Illustrations

### Illustration 1: Teacher Forcing vs. Free-Running Autoregression

**Problem:**
Explain the mathematical distinction between **Teacher Forcing** during training and **Free-Running Generation** during inference for an autoregressive sequence model. What is **Exposure Bias**?

**Solution:**
1. **Teacher Forcing (Training Phase):**
   At time step $t$, the ground-truth previous token $\mathbf{y}_{t-1}^*$ is fed as input $\mathbf{x}_t$, regardless of what the network predicted at step $t-1$:
   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{y}_{t-1}^*)$$
   - *Advantage:* Fast, stable parallel training; errors do not cascade early in training.
2. **Free-Running (Inference / Sampling Phase):**
   Ground-truth future tokens do not exist. The model must feed its own sampled output $\hat{\mathbf{y}}_{t-1}$ as input to step $t$:
   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \hat{\mathbf{y}}_{t-1})$$
3. **Exposure Bias:**
   Because the model was trained exclusively on pristine ground-truth history ($p(\mathbf{x}_t \mid \mathbf{y}_{<t}^*)$), it never learned to recover from mistakes. During free-running inference, an early minor error puts the hidden state into an out-of-distribution regime, causing errors to compound uncontrollably down the sequence.

---

### Illustration 2: Why RNNs Cannot Parallelize Training (The $O(T)$ Sequential Bottleneck)

**Problem:**
Contrast the computational graph depth and training parallelizability of an RNN vs. a Transformer on a sequence of length $T$.

**Solution:**
- **RNN:**
  The forward recurrence $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t)$ forms an inherently sequential chain. Step $t$ cannot begin until step $t-1$ finishes.
  - Compute graph depth across time: $\mathcal{O}(T)$ sequential operations.
  - Modern GPUs have thousands of parallel cores that remain underutilized because time cannot be unrolled in parallel.
- **Transformer:**
  All $T$ tokens are projected and attended simultaneously via matrix multiplications:
  $$\mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{T \times T}$$
  - Compute graph depth across time: $\mathcal{O}(1)$ parallel operations.
  - Maximizes GPU hardware occupancy, which is the primary reason Transformers replaced RNNs for large foundation models.

---

## 7. Deep Learning Connection & Application

### 1. Modern Roles of RNNs / State Space Models
While Transformers dominate language modeling, the recurrent formulation $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$ has experienced a massive resurgence through **Linear Recurrent Networks and State Space Models (SSMs)**:
- **Mamba & S4 (Gu & Dao, 2023):** Modern continuous-time linear dynamical systems that eliminate non-linear $\tanh$ recurrences, allowing training to be formulated as a parallel 1D convolution ($\mathcal{O}(\log T)$ parallel scan) while retaining $\mathcal{O}(1)$ inference memory per step.
- **Ultra-low Latency Audio & Biosensors:** Microcontrollers and edge wearables running on milliwatt power budgets rely on lightweight RNNs because $O(1)$ streaming inference requires no multi-gigabyte KV-cache.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `VanillaRNNScratch`: Pure NumPy implementation of forward pass and backward BPTT.
2. Exact numerical verification of the Part 5 Visual Grid hand arithmetic.
3. High-precision finite-difference numerical gradient checking for $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy}$.
4. Full parity comparison against PyTorch's `nn.RNN` module with gradient matching to $< 10^{-12}$.

See implementation in:
[`08_recurrent_networks/code/01_sequence_processing_and_vanilla_rnn.py`](./code/01_sequence_processing_and_vanilla_rnn.py)
