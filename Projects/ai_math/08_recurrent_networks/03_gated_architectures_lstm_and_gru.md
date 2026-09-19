# Chapter 8.3: Gated Architectures (LSTM & GRU)

---

## 1. Intuition & 101 Motivation

In Chapter 8.2, we uncovered the fundamental mathematical dilemma of Vanilla Recurrent Neural Networks: repeatedly applying the non-linear recurrent transformation $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t)$ creates a long-term Jacobian chain product:
$$\prod_{k=t+1}^T \operatorname{diag}(1 - \mathbf{h}_k^2) \mathbf{W}_{hh}$$
whose norm contracts exponentially to zero whenever singular values are strictly bounded to prevent explosion.

In 1997, Sepp Hochreiter and Jürgen Schmidhuber solved this long-standing crisis with the **Long Short-Term Memory (LSTM)** network.

Their radical architectural innovation was:
1. **Separating Memory from Output:** The model decouples the internal linear memory accumulator—the **Cell State** $\mathbf{c}_t$—from the exposed, non-linearly bounded hidden state $\mathbf{h}_t$.
2. **The Constant Error Carousel (CEC):** Instead of overwriting memory through non-linear matrix multiplication, the cell state updates **additively**:
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
   When the forget gate $\mathbf{f}_t = \mathbf{1}$, the Jacobian $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{I}$, creating a pristine **gradient highway** through time.
3. **Multiplicative Gating Valves:** Three continuous, differentiable sigmoid gates $\in (0, 1)$ regulate information traffic:
   - **Forget Gate ($\mathbf{f}_t$):** What proportion of the old cell state to discard.
   - **Input Gate ($\mathbf{i}_t$):** What proportion of the candidate memory to write.
   - **Output Gate ($\mathbf{o}_t$):** What proportion of the internal memory to expose to the network.

In 2014, Kyunghyun Cho et al. introduced the **Gated Recurrent Unit (GRU)**, which compresses the 4-gate LSTM into an elegant 2-gate architecture without a separate cell state, reducing parameter count and compute by **$25\%$**.

---

## 2. Rigorous Mathematical Formulation

```
                     LONG SHORT-TERM MEMORY (LSTM) CELL
           c_{t-1} ──────[ x ]───────────────────[ + ]─────────────────► c_t
                          ▲                       ▲            │
                          │ f_t                   │ i_t * c~   │
                     ┌────┴────┐             ┌────┴────┐     [tanh]
                     │  sigma  │             │    *    │       │
                     └────▲────┘             └────▲────┘       ▼
                          │                  ┌────┴────┐     [ x ] ◄── o_t (sigma)
                          │             ┌───►│  sigma  │(i_t)  │
                          │             │    └─────────┘       │
                          │             │    ┌─────────┐       │
                          │             ├───►│  tanh   │(c~)   │
                          │             │    └─────────┘       │
                          │             │    ┌─────────┐       │
                          ├─────────────┴───►│  sigma  │(o_t)  ▼
                          │                  └─────────┘      h_t
                   [ h_{t-1}, x_t ]
```

---

### 2.1 The Long Short-Term Memory (LSTM) Formulation

Let $\mathbf{x}_t \in \mathbb{R}^{d_x}$ be the input vector at time step $t$.
Let $\mathbf{h}_{t-1} \in \mathbb{R}^{d_h}$ be the previous hidden state, and $\mathbf{c}_{t-1} \in \mathbb{R}^{d_h}$ be the previous cell state.

We concatenate input and recurrent activations into:
$$\mathbf{v}_t = \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} \in \mathbb{R}^{(d_h + d_x)}$$

#### The Six Governing Equations:
1. **Forget Gate:**
   $$\mathbf{f}_t = \sigma\left(\mathbf{W}_f \mathbf{v}_t + \mathbf{b}_f\right) \in (0, 1)^{d_h}$$
2. **Input / Write Gate:**
   $$\mathbf{i}_t = \sigma\left(\mathbf{W}_i \mathbf{v}_t + \mathbf{b}_i\right) \in (0, 1)^{d_h}$$
3. **Candidate Cell State:**
   $$\tilde{\mathbf{c}}_t = \tanh\left(\mathbf{W}_c \mathbf{v}_t + \mathbf{b}_c\right) \in (-1, 1)^{d_h}$$
4. **Cell State Update (The Constant Error Carousel):**
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \in \mathbb{R}^{d_h}$$
5. **Output / Read Gate:**
   $$\mathbf{o}_t = \sigma\left(\mathbf{W}_o \mathbf{v}_t + \mathbf{b}_o\right) \in (0, 1)^{d_h}$$
6. **Hidden State Emission:**
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \in (-1, 1)^{d_h}$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the element-wise logistic sigmoid function, and $\odot$ represents the Hadamard (element-wise) product.

---

### 2.2 The Constant Error Carousel (CEC) Mathematical Proof

Consider the temporal Jacobian of the cell state from step $t-1$ to $t$:
$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \operatorname{diag}(\mathbf{f}_t) + \underbrace{\frac{\partial \mathbf{f}_t}{\partial \mathbf{c}_{t-1}} \odot \mathbf{c}_{t-1} + \frac{\partial \mathbf{i}_t}{\partial \mathbf{c}_{t-1}} \odot \tilde{\mathbf{c}}_t + \frac{\partial \tilde{\mathbf{c}}_t}{\partial \mathbf{c}_{t-1}} \odot \mathbf{i}_t}_{\text{Indirect terms through } \mathbf{h}_{t-1}}$$

Because $\mathbf{f}_t, \mathbf{i}_t, \tilde{\mathbf{c}}_t$ depend on $\mathbf{h}_{t-1}$ (which depends on $\mathbf{c}_{t-1}$ via $\mathbf{o}_{t-1} \odot \tanh(\mathbf{c}_{t-1})$), the direct error propagation along the cell state backbone is governed by:
$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \operatorname{diag}(\mathbf{f}_t)$$

Over an extended horizon of $T - t$ time steps:
$$\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_t} = \prod_{k=t+1}^T \operatorname{diag}(\mathbf{f}_k) = \operatorname{diag}\left( \prod_{k=t+1}^T \mathbf{f}_k \right)$$

#### Key Theoretical Insights:
1. **Absence of Matrix Multiplications:** The Jacobian product does **not** involve repeated multiplication by weight matrices $\mathbf{W}$. It is a diagonal product of scalar gate activations!
2. **Gradient Preservation:** If the network learns to keep the forget gate open ($\mathbf{f}_k \approx \mathbf{1}$), the product $\prod_{k} \mathbf{f}_k \approx \mathbf{1}$. Gradients flow backwards across hundreds of steps with **zero exponential decay**.
3. **Selective Amnesia:** If an old memory is deemed obsolete (e.g., closing a parentheses or ending a sentence), setting $\mathbf{f}_k \to \mathbf{0}$ instantly zeroes both the forward memory and the backward gradient for that specific feature coordinate.

---

### 2.3 The Forget Gate Bias Initialization Trick (Jozefowicz et al., 2015)

In an exhaustive empirical search of over 10,000 RNN architectures, Rafal Jozefowicz, Wojciech Zaremba, and Ilya Sutskever demonstrated that the single most critical hyperparameter for LSTM stability is **initializing the forget gate bias to a positive constant**:
$$\mathbf{b}_f \sim \mathcal{U}(1.0, 2.0)$$
- If $\mathbf{b}_f = 0$, at initialization $\mathbf{f}_t \approx \sigma(0) = 0.5$. Over 20 steps, the gradient decays by $0.5^{20} \approx 10^{-6}$, re-introducing vanishing gradients on day one!
- If $\mathbf{b}_f = 1.0$ or $2.0$, at initialization $\mathbf{f}_t \approx \sigma(2.0) \approx 0.88$. The gates default to wide open, allowing gradients to propagate deeply from the very first training epoch.

---

### 2.4 The Gated Recurrent Unit (GRU) (Cho et al., 2014)

```
                     GATED RECURRENT UNIT (GRU) CELL
           h_{t-1} ──────[ x ]───────────────────[ + ]─────────────────► h_t
                          ▲                       ▲
                          │ (1 - z_t)             │ z_t * h~
                          │                  ┌────┴────┐
                          │             ┌───►│  sigma  │(z_t)
                          │             │    └─────────┘
                          │             │    ┌─────────┐
                          │             ├───►│  tanh   │(h~) ◄── r_t (sigma)
                          │             │    └─────────┘
                   [ h_{t-1}, x_t ] ────┴─────────────────────────────
```

The GRU eliminates the separate cell state $\mathbf{c}_t$, utilizing only the hidden state $\mathbf{h}_t$. It merges the input and forget gates into a single **Update Gate** $\mathbf{z}_t$, and controls historical context through a **Reset Gate** $\mathbf{r}_t$.

#### The Four Governing Equations:
1. **Reset Gate:**
   $$\mathbf{r}_t = \sigma\left(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r\right) \in (0, 1)^{d_h}$$
2. **Update Gate:**
   $$\mathbf{z}_t = \sigma\left(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z\right) \in (0, 1)^{d_h}$$
3. **Candidate Hidden State:**
   $$\tilde{\mathbf{h}}_t = \tanh\left(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \, \mathbf{x}_t] + \mathbf{b}_h\right) \in (-1, 1)^{d_h}$$
4. **Interpolation State Update:**
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \in (-1, 1)^{d_h}$$

#### Architectural Trade-offs:
- **Convex Combination:** The update equation is an exact linear interpolation between the previous state $\mathbf{h}_{t-1}$ and candidate state $\tilde{\mathbf{h}}_t$.
- When $\mathbf{z}_t = \mathbf{0}$, $\mathbf{h}_t = \mathbf{h}_{t-1}$ (pure identity bypass, zero gradient vanishing).
- When $\mathbf{z}_t = \mathbf{1}$, $\mathbf{h}_t = \tilde{\mathbf{h}}_t$ (complete overwrite).
- **Parameter Efficiency:** GRU requires $3$ gate matrices vs. $4$ for LSTM. For identical hidden dimensions, GRU has **$25\%$ fewer parameters** and trains significantly faster on small-to-medium datasets.

---

### 2.5 Deep Derivation 8.3.1: Full Analytical Backpropagation Through Time for LSTM Cells

In this derivation, we derive the exact, complete analytical gradient equations for an LSTM cell at time step $t$, showing how upstream errors decompose across gates, the cell state, and historical hidden states.

#### Step 1: Upstream Gradient Ingestion
At time step $t$, the LSTM receives two gradient signals:
1. $\delta_t^h \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \in \mathbb{R}^{d_h}$ (from current loss $\ell_t$ and future recurrence $\mathbf{h}_{t+1}$),
2. $\delta_{t+1}^c \equiv \frac{\partial \mathcal{L}_{\text{future}}}{\partial \mathbf{c}_t} \in \mathbb{R}^{d_h}$ (from future cell state $\mathbf{c}_{t+1}$).

#### Step 2: Total Cell State Gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t}$
The current cell state $\mathbf{c}_t$ affects the loss through two paths:
1. Immediately through the hidden state $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$,
2. Into the future through $\mathbf{c}_{t+1}$.
Applying the multivariable chain rule:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} = \delta_{t+1}^c + \delta_t^h \odot \mathbf{o}_t \odot \left(1 - \tanh^2(\mathbf{c}_t)\right)$$

#### Step 3: Gate Sensitivities and Pre-Activation Deltas
Using the forward equations:
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t), \qquad \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

1. **Output Gate Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{o}_t} = \delta_t^h \odot \tanh(\mathbf{c}_t)$$
   Since $\mathbf{o}_t = \sigma(\mathbf{a}_o)$, where $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:
   $$\delta_t^{a_o} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_o} = \frac{\partial \mathcal{L}}{\partial \mathbf{o}_t} \odot \mathbf{o}_t \odot (1 - \mathbf{o}_t) = \delta_t^h \odot \tanh(\mathbf{c}_t) \odot \mathbf{o}_t \odot (1 - \mathbf{o}_t)$$

2. **Candidate Cell State Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{c}}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \mathbf{i}_t$$
   Since $\tilde{\mathbf{c}}_t = \tanh(\mathbf{a}_c)$, where $\tanh'(z) = 1 - \tanh^2(z)$:
   $$\delta_t^{a_c} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_c} = \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{c}}_t} \odot \left(1 - \tilde{\mathbf{c}}_t^2\right) = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \mathbf{i}_t \odot \left(1 - \tilde{\mathbf{c}}_t^2\right)$$

3. **Input Gate Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{i}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \tilde{\mathbf{c}}_t$$
   Since $\mathbf{i}_t = \sigma(\mathbf{a}_i)$:
   $$\delta_t^{a_i} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{i}_t} \odot \mathbf{i}_t \odot (1 - \mathbf{i}_t) = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \tilde{\mathbf{c}}_t \odot \mathbf{i}_t \odot (1 - \mathbf{i}_t)$$

4. **Forget Gate Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{f}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \mathbf{c}_{t-1}$$
   Since $\mathbf{f}_t = \sigma(\mathbf{a}_f)$:
   $$\delta_t^{a_f} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_f} = \frac{\partial \mathcal{L}}{\partial \mathbf{f}_t} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t) = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)$$

#### Step 4: Backward Recurrence to Previous States
1. **To Previous Cell State $\mathbf{c}_{t-1}$ (The CEC Highway):**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_{t-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \mathbf{f}_t$$
2. **To Previous Hidden State $\mathbf{h}_{t-1}$ and Input $\mathbf{x}_t$:**
   Stack all four pre-activation deltas into $\delta_t^{\text{all}} = [\delta_t^{a_i}; \, \delta_t^{a_f}; \, \delta_t^{a_c}; \, \delta_t^{a_o}] \in \mathbb{R}^{4d_h}$.
   Recall the concatenated forward projection: $\mathbf{a}_{\text{all}} = \mathbf{W}_{\text{all}} \mathbf{v}_t + \mathbf{b}_{\text{all}}$, where $\mathbf{v}_t = [\mathbf{h}_{t-1}; \, \mathbf{x}_t]$.
   Then:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_t} = \mathbf{W}_{\text{all}}^T \delta_t^{\text{all}} \in \mathbb{R}^{d_h + d_x}$$
   Splitting the vector:
   $$\frac{\partial \mathcal{L}_{\text{recurrent}}}{\partial \mathbf{h}_{t-1}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{v}_t} \right)_{1:d_h}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{x}_t} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{v}_t} \right)_{d_h+1 : d_h+d_x}$$
3. **Parameter Gradient Accumulation:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{all}}} = \sum_{t=1}^T \delta_t^{\text{all}} \, \mathbf{v}_t^T \in \mathbb{R}^{4d_h \times (d_h + d_x)}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_{\text{all}}} = \sum_{t=1}^T \delta_t^{\text{all}} \in \mathbb{R}^{4d_h}$$
This completes the exact, closed-form BPTT derivation for LSTM cells. $\blacksquare$

---

### 2.6 Deep Derivation 8.3.2: Full Analytical Backpropagation Through Time for GRU Cells

In this derivation, we derive the exact BPTT gradient equations for the Gated Recurrent Unit (GRU).

#### Step 1: Upstream Gradient and Intermediate Derivatives
Given upstream gradient $\delta_t^h \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \in \mathbb{R}^{d_h}$.
From the GRU update equation:
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

Taking partial derivatives:
1. **Candidate Hidden State:**
   $$\frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{h}}_t} = \delta_t^h \odot \mathbf{z}_t$$
2. **Update Gate:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_t} = \delta_t^h \odot (\tilde{\mathbf{h}}_t - \mathbf{h}_{t-1})$$

#### Step 2: Pre-Activation Deltas for Candidate and Update Gate
1. **Candidate Pre-activation $\mathbf{a}_h$:**
   Since $\tilde{\mathbf{h}}_t = \tanh(\mathbf{a}_h)$:
   $$\delta_t^{a_h} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_h} = \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{h}}_t} \odot (1 - \tilde{\mathbf{h}}_t^2) = \delta_t^h \odot \mathbf{z}_t \odot (1 - \tilde{\mathbf{h}}_t^2)$$

2. **Update Gate Pre-activation $\mathbf{a}_z$:**
   Since $\mathbf{z}_t = \sigma(\mathbf{a}_z)$:
   $$\delta_t^{a_z} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_z} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_t} \odot \mathbf{z}_t \odot (1 - \mathbf{z}_t) = \delta_t^h \odot (\tilde{\mathbf{h}}_t - \mathbf{h}_{t-1}) \odot \mathbf{z}_t \odot (1 - \mathbf{z}_t)$$

#### Step 3: Backpropagation Through the Reset Gate
The candidate pre-activation is $\mathbf{a}_h = \mathbf{W}_{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h$.
Differentiating with respect to the gated recurrent activation $\tilde{\mathbf{r}}_t \equiv \mathbf{r}_t \odot \mathbf{h}_{t-1}$:
$$\frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{r}}_t} = \mathbf{W}_{hh}^T \delta_t^{a_h}$$

Now differentiate with respect to the reset gate $\mathbf{r}_t$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{r}_t} = \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{r}}_t} \odot \mathbf{h}_{t-1} = \left( \mathbf{W}_{hh}^T \delta_t^{a_h} \right) \odot \mathbf{h}_{t-1}$$
Since $\mathbf{r}_t = \sigma(\mathbf{a}_r)$:
$$\delta_t^{a_r} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_r} = \frac{\partial \mathcal{L}}{\partial \mathbf{r}_t} \odot \mathbf{r}_t \odot (1 - \mathbf{r}_t) = \left( \mathbf{W}_{hh}^T \delta_t^{a_h} \right) \odot \mathbf{h}_{t-1} \odot \mathbf{r}_t \odot (1 - \mathbf{r}_t)$$

#### Step 4: Total Recurrent Gradient w.r.t. $\mathbf{h}_{t-1}$
The previous hidden state $\mathbf{h}_{t-1}$ influences the loss along three distinct pathways:
1. Directly through the linear interpolation shortcut: $(1 - \mathbf{z}_t) \odot \delta_t^h$
2. Through the gated candidate state: $\mathbf{r}_t \odot \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{r}}_t} = \mathbf{r}_t \odot (\mathbf{W}_{hh}^T \delta_t^{a_h})$
3. Through the gate projections $[\mathbf{W}_{zr}; \, \mathbf{W}_{rr}]^T [\delta_t^{a_z}; \, \delta_t^{a_r}]$

Summing all contributions:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t-1}} = \delta_t^h \odot (1 - \mathbf{z}_t) + \mathbf{r}_t \odot \left( \mathbf{W}_{hh}^T \delta_t^{a_h} \right) + \mathbf{W}_{z, h}^T \delta_t^{a_z} + \mathbf{W}_{r, h}^T \delta_t^{a_r}$$
When $\mathbf{z}_t \to \mathbf{0}$, the term $\delta_t^h \odot (1 - \mathbf{z}_t) \to \delta_t^h$, guaranteeing that the gradient bypasses all non-linearities and matrix weights unimpeded. $\blacksquare$

---

### 2.7 Deep Derivation 8.3.3: The Gradient Highway Equivalence: Constant Error Carousel vs. Residual Connections

Here we establish the formal mathematical equivalence between the Constant Error Carousel (CEC) of LSTMs and the Identity Shortcuts of Residual Networks (ResNet).

#### Step 1: Comparison of Dynamical State Equations
Consider the state transition equations:
- **LSTM Cell State:**
  $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t(\mathbf{h}_{t-1}, \mathbf{x}_t)$$
- **ResNet Layer Activation:**
  $$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$

Notice that the ResNet residual connection is an exact special case of the LSTM Cell State where:
$$\mathbf{f}_t \equiv \mathbf{1}, \qquad \mathbf{i}_t \equiv \mathbf{1}, \qquad \tilde{\mathbf{c}}_t \equiv \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$
ResNet is mathematically an unrolled LSTM across depth with gates clamped permanently open!

#### Step 2: Backward Adjoint Comparison
Differentiating the state recursion from step $l$ to $L$:
- **ResNet Adjoint Operator:**
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \sum_{k=l}^{L-1} \frac{\partial \mathcal{F}_k}{\partial \mathbf{x}_l} \right)$$
- **LSTM Adjoint Operator (holding gates constant):**
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_T} \left( \prod_{k=t+1}^T \operatorname{diag}(\mathbf{f}_k) \right) + \sum_{k=t+1}^T \frac{\partial \ell_k}{\partial \mathbf{c}_t}$$

In both architectures:
1. The forward state is additive: $\mathbf{s}_{\text{new}} = \mathbf{s}_{\text{old}} + \Delta \mathbf{s}$.
2. The backward gradient contains an additive identity component that avoids repeated multiplication by weight matrices.
3. While ResNet provides an unconditional identity highway ($\mathbf{I}$), LSTM provides a **parameter-controlled conditional highway** ($\operatorname{diag}(\mathbf{f}_k)$), allowing the network to dynamically open the highway to remember or close it to forget. $\blacksquare$

---

## 3. Geometric & Algebraic Interpretation

### The Gated Phase Space Topology
In a Vanilla RNN, the state transformation $\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h}_{t-1})$ warps, rotates, and squashes the entire phase space simultaneously.
In an LSTM:
- The cell state $\mathbf{c}_t$ evolves in an unbounded linear vector space $\mathbb{R}^{d_h}$.
- The gating signals $\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t$ act as **coordinate-wise dynamic projection operators** restricted to the hypercube $[0, 1]^{d_h}$.
- Instead of twisting the entire coordinate frame, gating scales individual orthogonal coordinate axes independently. A neuron tracking a specific long-term attribute (e.g., whether a quotation mark is open) can freeze its axis by keeping $f_k = 1, i_k = 0$ while all other 511 dimensions actively process short-term grammatical tokens.

---

## 4. Real-World Analogy

### The Industrial Factory Assembly Line (LSTM)
- **Cell State ($\mathbf{c}_t$):** A continuous, heavy-duty industrial conveyor belt running through a massive factory. Items placed on the belt remain on the belt by inertia unless acted upon.
- **Forget Gate ($\mathbf{f}_t$):** A robotic scraper arm above the belt. When an obsolete part passes by, it pushes it into a recycling bin ($f \to 0$). If the part is still needed, the arm lifts up and lets it pass untouched ($f \to 1$).
- **Input Gate ($\mathbf{i}_t$) & Candidate ($\tilde{\mathbf{c}}_t$):** A stamping machine that inspects incoming raw supplies ($\mathbf{x}_t$), manufactures a new component ($\tilde{\mathbf{c}}_t$), and places it onto the belt if permitted ($i_t \to 1$).
- **Output Gate ($\mathbf{o}_t$):** A camera over the belt that photographs the current assembly, applies an optical filter ($\tanh$), and transmits the image to the factory dashboard ($\mathbf{h}_t$) without altering the physical belt itself.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace forward and backward propagation through a complete LSTM cell with concrete hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in LSTM Cell |
| :--- | :--- | :--- | :--- |
| $x_t$ | Input Token | Scalar ($1.0$) | Input activation at time step $t$ |
| $h_{t-1}$ | Previous Hidden State | Scalar ($0.5$) | Hidden state from time step $t-1$ |
| $c_{t-1}$ | Previous Cell State | Scalar ($2.0$) | Memory accumulator from time step $t-1$ |
| $f_t$ | Forget Gate | Scalar $\in (0, 1)$ | $\sigma(W_f x_t + U_f h_{t-1} + b_f)$ |
| $i_t$ | Input Gate | Scalar $\in (0, 1)$ | $\sigma(W_i x_t + U_i h_{t-1} + b_i)$ |
| $\tilde{c}_t$ | Candidate Cell State | Scalar $\in (-1, 1)$ | $\tanh(W_c x_t + U_c h_{t-1} + b_c)$ |
| $c_t$ | Updated Cell State | Scalar | $c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t$ |
| $o_t$ | Output Gate | Scalar $\in (0, 1)$ | $\sigma(W_o x_t + U_o h_{t-1} + b_o)$ |
| $h_t$ | Updated Hidden State | Scalar $\in (-1, 1)$ | $h_t = o_t \cdot \tanh(c_t)$ |
| $\frac{\partial \mathcal{L}}{\partial h_t}$ | Upstream Hidden Grad | Scalar ($1.0$) | Sensitivity of objective loss w.r.t. current hidden state |
| $\frac{\partial \mathcal{L}}{\partial c_t}$ | Cell State Gradient | Scalar | Direct upstream gradient + gradient through $h_t$ |
| $\frac{\partial \mathcal{L}}{\partial c_{t-1}}$ | Prior Cell Gradient | Scalar | Error routed backwards along Constant Error Carousel |

---

### 5.2 Concrete Toy Numbers

#### State Inputs:
$$x_t = 1.0, \quad h_{t-1} = 0.5, \quad c_{t-1} = 2.0$$

#### Model Weights and Biases:
- **Forget Gate:** $W_f = 0.0, \quad U_f = 0.0, \quad b_f = 1.0 \implies z_f = 1.0$
- **Input Gate:** $W_i = 1.0, \quad U_i = -1.0, \quad b_i = 0.5 \implies z_i = 1.0(1.0) - 1.0(0.5) + 0.5 = 1.0$
- **Candidate State:** $W_c = 0.5, \quad U_c = 1.0, \quad b_c = -1.0 \implies z_c = 0.5(1.0) + 1.0(0.5) - 1.0 = 0.0$
- **Output Gate:** $W_o = 0.0, \quad U_o = 2.0, \quad b_o = -1.0 \implies z_o = 0.0(1.0) + 2.0(0.5) - 1.0 = 0.0$

---

### 5.3 Forward Pass Arithmetic

1. **Forget Gate Activation:**
   $$f_t = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} \approx \mathbf{0.731059}$$

2. **Input Gate Activation:**
   $$i_t = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} \approx \mathbf{0.731059}$$

3. **Candidate Memory:**
   $$\tilde{c}_t = \tanh(0.0) = \mathbf{0.0}$$

4. **Cell State Update ($c_t = f_t c_{t-1} + i_t \tilde{c}_t$):**
   $$c_t = (0.731059)(2.0) + (0.731059)(0.0) = 1.462118 + 0.0 = \mathbf{1.462118}$$

5. **Output Gate Activation:**
   $$o_t = \sigma(0.0) = \frac{1}{1 + e^{0}} = \frac{1}{2} = \mathbf{0.5}$$

6. **Hidden State Emission ($h_t = o_t \tanh(c_t)$):**
   $$\tanh(c_t) = \tanh(1.462118) \approx 0.898144$$
   $$h_t = (0.5)(0.898144) = \mathbf{0.449072}$$

---

### 5.4 Backward Pass Arithmetic (The Constant Error Carousel)

Suppose the upstream loss gradient at step $t$ is:
$$\frac{\partial \mathcal{L}}{\partial h_t} = 1.0, \quad \frac{\partial \mathcal{L}_{\text{direct}}}{\partial c_t} = 0.0$$

1. **Gradient w.r.t. Current Cell State ($\frac{\partial \mathcal{L}}{\partial c_t}$):**
   $$h_t = o_t \tanh(c_t)$$
   $$\frac{\partial \mathcal{L}}{\partial c_t} = \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial c_t} = \frac{\partial \mathcal{L}}{\partial h_t} \cdot o_t \cdot \left(1 - \tanh^2(c_t)\right)$$
   $$1 - \tanh^2(1.462118) = 1 - (0.898144)^2 = 1 - 0.806663 = 0.193337$$
   $$\frac{\partial \mathcal{L}}{\partial c_t} = (1.0)(0.5)(0.193337) = \mathbf{0.096669}$$

2. **Gradient w.r.t. Output Gate Pre-activation ($\delta^{z_o}$):**
   $$\frac{\partial \mathcal{L}}{\partial o_t} = \frac{\partial \mathcal{L}}{\partial h_t} \tanh(c_t) = (1.0)(0.898144) = 0.898144$$
   $$\delta^{z_o} = \frac{\partial \mathcal{L}}{\partial o_t} \cdot o_t(1 - o_t) = (0.898144)(0.5)(1 - 0.5) = (0.898144)(0.25) = \mathbf{0.224536}$$

3. **Gradient w.r.t. Previous Cell State ($\frac{\partial \mathcal{L}}{\partial c_{t-1}}$ - The CEC in Action):**
   $$c_t = f_t c_{t-1} + i_t \tilde{c}_t$$
   $$\frac{\partial \mathcal{L}}{\partial c_{t-1}} = \frac{\partial \mathcal{L}}{\partial c_t} \frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial \mathcal{L}}{\partial c_t} \cdot f_t$$
   $$\frac{\partial \mathcal{L}}{\partial c_{t-1}} = (0.096669)(0.731059) = \mathbf{0.070671}$$

Look at this calculation: the gradient flows directly from $c_t$ to $c_{t-1}$ scaled cleanly by the forget gate $f_t = 0.731059$, without encountering any saturating matrix multiplications!

---

## 6. Solved Illustrations

### Illustration 1: Parameter Count Derivation: LSTM vs. GRU

**Problem:**
Derive the exact formula for the total number of trainable parameters in:
1. A standard LSTM cell with input dimension $d_x$ and hidden dimension $d_h$.
2. A standard GRU cell with input dimension $d_x$ and hidden dimension $d_h$.
Calculate the parameter counts for $d_x = 512, d_h = 512$.

**Solution:**
1. **LSTM Cell:**
   An LSTM contains 4 transformations ($f, i, c, o$), each taking concatenated input $[\mathbf{h}_{t-1}, \mathbf{x}_t] \in \mathbb{R}^{d_h + d_x}$ to an output $\in \mathbb{R}^{d_h}$:
   $$\text{Params}(\text{LSTM}) = 4 \times \left[ (d_h + d_x) \times d_h + d_h \right] = 4 d_h (d_h + d_x + 1)$$
   For $d_x = 512, d_h = 512$:
   $$\text{Params}(\text{LSTM}) = 4 \times 512 \times (512 + 512 + 1) = 2,048 \times 1,025 = \mathbf{2,099,200 \text{ parameters}}$$

2. **GRU Cell:**
   A GRU contains 3 transformations ($r, z, h$):
   $$\text{Params}(\text{GRU}) = 3 \times \left[ (d_h + d_x) \times d_h + d_h \right] = 3 d_h (d_h + d_x + 1)$$
   For $d_x = 512, d_h = 512$:
   $$\text{Params}(\text{GRU}) = 3 \times 512 \times (512 + 512 + 1) = 1,536 \times 1,025 = \mathbf{1,574,400 \text{ parameters}}$$

**Comparison:**
$$\frac{\text{Params}(\text{GRU})}{\text{Params}(\text{LSTM})} = \frac{3}{4} = \mathbf{75\%}$$
The GRU provides an exact **$25\%$ reduction in parameter memory and matrix multiplication FLOPs**.

---

### Illustration 2: Peephole Connections (Gers & Schmidhuber, 2000)

**Problem:**
What are **peephole connections** in an LSTM? Why were they introduced, and why did modern deep learning frameworks (PyTorch default `nn.LSTM`) largely abandon them?

**Solution:**
In standard LSTM, gates $f_t, i_t, o_t$ only observe the previous hidden state $\mathbf{h}_{t-1}$ and input $\mathbf{x}_t$; they cannot inspect the current cell state $\mathbf{c}_{t-1}$ or $\mathbf{c}_t$.
Felix Gers and Jürgen Schmidhuber introduced peephole connections by adding diagonal weights from the cell state directly into the gates:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{v}_t + \mathbf{p}_f \odot \mathbf{c}_{t-1} + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{v}_t + \mathbf{p}_i \odot \mathbf{c}_{t-1} + \mathbf{b}_i)$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{v}_t + \mathbf{p}_o \odot \mathbf{c}_t + \mathbf{b}_o)$$
- *Theoretical Motivation:* Allows gates to time precise intervals by directly monitoring the accumulation level of the linear integrator $\mathbf{c}_t$.
- *Why Abandoned:* Benchmarks across large-scale speech recognition and machine translation revealed negligible accuracy gains, while adding memory access synchronization overhead that prevents fusing all 4 gate matrix multiplications into a single cuBLAS GEMM call (`[W_f; W_i; W_c; W_o]`).

---

### Illustration 3: Hand Arithmetic for a Complete GRU Forward and Backward Step

**Problem:**
Let a 1D scalar GRU ($d_h = 1, d_x = 1$) process a single time step with inputs:
$$x_t = 1.0, \quad h_{t-1} = 0.5$$
The model parameters are:
- **Reset Gate:** $W_{rx} = 0.0, \quad W_{rh} = 0.0, \quad b_r = 0.0$
- **Update Gate:** $W_{zx} = 1.0, \quad W_{zh} = -1.0, \quad b_z = 0.5$
- **Candidate State:** $W_{hx} = 0.5, \quad W_{hh} = 1.0, \quad b_h = -0.5$
1. Compute the forward activations $r_t, z_t, \tilde{h}_t, h_t$.
2. Given upstream loss gradient $\frac{\partial \mathcal{L}}{\partial h_t} = 1.0$, compute the gate deltas $\delta^{a_h}, \delta^{a_z}, \delta^{a_r}$.
3. Compute the backpropagated gradient with respect to the previous hidden state $\frac{\partial \mathcal{L}}{\partial h_{t-1}}$.

**Solution:**

#### Step 1: Forward Pass Arithmetic
1. **Reset Gate:**
   $$a_r = W_{rx} x_t + W_{rh} h_{t-1} + b_r = 0.0(1.0) + 0.0(0.5) + 0.0 = 0.0$$
   $$r_t = \sigma(0.0) = \mathbf{0.5}$$

2. **Update Gate:**
   $$a_z = W_{zx} x_t + W_{zh} h_{t-1} + b_z = 1.0(1.0) - 1.0(0.5) + 0.5 = 1.0$$
   $$z_t = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} \approx \mathbf{0.731059}$$

3. **Candidate Hidden State:**
   Gated state: $\tilde{r}_t = r_t h_{t-1} = (0.5)(0.5) = 0.25$
   $$a_h = W_{hh} \tilde{r}_t + W_{hx} x_t + b_h = 1.0(0.25) + 0.5(1.0) - 0.5 = 0.25$$
   $$\tilde{h}_t = \tanh(0.25) \approx \mathbf{0.244919}$$

4. **Updated Hidden State:**
   $$h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t = (1 - 0.731059)(0.5) + (0.731059)(0.244919)$$
   $$= (0.268941)(0.5) + 0.179049 = 0.134471 + 0.179049 = \mathbf{0.313520}$$

#### Step 2: Backward Pass Arithmetic
Given $\frac{\partial \mathcal{L}}{\partial h_t} = 1.0$:

1. **Candidate Pre-activation Delta ($\delta^{a_h}$):**
   $$\frac{\partial \mathcal{L}}{\partial \tilde{h}_t} = \frac{\partial \mathcal{L}}{\partial h_t} z_t = 1.0(0.731059) = 0.731059$$
   $$1 - \tilde{h}_t^2 = 1 - (0.244919)^2 = 1 - 0.059985 = 0.940015$$
   $$\delta^{a_h} = \frac{\partial \mathcal{L}}{\partial \tilde{h}_t} (1 - \tilde{h}_t^2) = (0.731059)(0.940015) = \mathbf{0.687206}$$

2. **Update Gate Pre-activation Delta ($\delta^{a_z}$):**
   $$\frac{\partial \mathcal{L}}{\partial z_t} = \frac{\partial \mathcal{L}}{\partial h_t} (\tilde{h}_t - h_{t-1}) = 1.0(0.244919 - 0.5) = -0.255081$$
   $$z_t (1 - z_t) = (0.731059)(0.268941) \approx 0.196612$$
   $$\delta^{a_z} = \frac{\partial \mathcal{L}}{\partial z_t} z_t (1 - z_t) = (-0.255081)(0.196612) = \mathbf{-0.050152}$$

3. **Reset Gate Pre-activation Delta ($\delta^{a_r}$):**
   $$\frac{\partial \mathcal{L}}{\partial \tilde{r}_t} = W_{hh} \delta^{a_h} = 1.0(0.687206) = 0.687206$$
   $$\frac{\partial \mathcal{L}}{\partial r_t} = \frac{\partial \mathcal{L}}{\partial \tilde{r}_t} h_{t-1} = (0.687206)(0.5) = 0.343603$$
   $$r_t (1 - r_t) = (0.5)(0.5) = 0.25$$
   $$\delta^{a_r} = \frac{\partial \mathcal{L}}{\partial r_t} r_t (1 - r_t) = (0.343603)(0.25) = \mathbf{0.085901}$$

#### Step 3: Gradient Flow into Previous Hidden State ($h_{t-1}$)
$$\frac{\partial \mathcal{L}}{\partial h_{t-1}} = \underbrace{(1 - z_t) \frac{\partial \mathcal{L}}{\partial h_t}}_{\text{Linear Shortcut}} + \underbrace{r_t \frac{\partial \mathcal{L}}{\partial \tilde{r}_t}}_{\text{Candidate Branch}} + \underbrace{W_{zh} \delta^{a_z}}_{\text{Update Gate}} + \underbrace{W_{rh} \delta^{a_r}}_{\text{Reset Gate}}$$
$$= (0.268941)(1.0) + (0.5)(0.687206) + (-1.0)(-0.050152) + 0.0(0.085901)$$
$$= 0.268941 + 0.343603 + 0.050152 + 0.0 = \mathbf{0.662696}$$

---

### Illustration 4: Fused GEMM Weight Packing and FLOP Count for LSTM vs. GRU

**Problem:**
A sequence modeling task processes mini-batches of size $B = 64$ across sequence length $T = 100$, with input feature dimension $d_x = 256$ and recurrent hidden dimension $d_h = 512$.
1. Compute the total number of trainable parameters for an LSTM layer vs. a GRU layer.
2. Compute the total Multiply-Accumulate (MAC) count and FLOPs per sequence for the fused matrix multiplications of both architectures.
3. Compute the memory footprint savings of the GRU.

**Solution:**

#### Step 1: Trainable Parameter Comparison
Input-to-hidden concatenated dimension: $D_{\text{concat}} = d_h + d_x = 512 + 256 = 768$.
- **LSTM Parameters (4 gates: $i, f, c, o$):**
  $$\text{Params}_{\text{LSTM}} = 4 \times \left( D_{\text{concat}} \cdot d_h + d_h \right) = 4 \times (768 \times 512 + 512) = 4 \times (393,216 + 512) = 4 \times 393,728 = \mathbf{1,574,912}$$
- **GRU Parameters (3 gates: $r, z, h$):**
  $$\text{Params}_{\text{GRU}} = 3 \times \left( D_{\text{concat}} \cdot d_h + d_h \right) = 3 \times 393,728 = \mathbf{1,181,184}$$
- **Parameter Savings:**
  $$\frac{1,574,912 - 1,181,184}{1,574,912} = \mathbf{25.0\% \text{ reduction in weights}}$$

#### Step 2: Fused GEMM Computation (per Sequence of Length $T=100$)
At each time step $t$, the forward pass multiplies the concatenated batch matrix $\mathbf{V}_t \in \mathbb{R}^{B \times (d_h + d_x)} = \mathbb{R}^{64 \times 768}$ by the fused weight matrix:
- **LSTM GEMM ($\mathbf{W}_{\text{all}} \in \mathbb{R}^{768 \times 2048}$):**
  $$\text{MACs}_{\text{step}} = B \cdot D_{\text{concat}} \cdot (4d_h) = 64 \times 768 \times 2048 = 100,663,296 \text{ MACs}$$
  Across $T = 100$ steps:
  $$\text{MACs}_{\text{total}}^{\text{LSTM}} = 100 \times 100,663,296 = \mathbf{10,066,329,600} \approx \mathbf{10.07 \text{ GMACs}} \quad (\mathbf{20.13 \text{ GFLOPs}})$$

- **GRU GEMM ($\mathbf{W}_{\text{all}} \in \mathbb{R}^{768 \times 1536}$):**
  $$\text{MACs}_{\text{step}} = B \cdot D_{\text{concat}} \cdot (3d_h) = 64 \times 768 \times 1536 = 75,497,472 \text{ MACs}$$
  Across $T = 100$ steps:
  $$\text{MACs}_{\text{total}}^{\text{GRU}} = 100 \times 75,497,472 = \mathbf{7,549,747,200} \approx \mathbf{7.55 \text{ GMACs}} \quad (\mathbf{15.10 \text{ GFLOPs}})$$

#### Step 3: Performance Summary
The GRU saves over **$5.03$ GFLOPs** per forward pass while eliminating the storage buffer for the cell state $\mathbf{c}_t \in \mathbb{R}^{B \times T \times d_h} = \mathbb{R}^{64 \times 100 \times 512} \implies 3,276,800$ floating-point activations ($13.1 \text{ MB}$ per batch).

---

### Illustration 5: Multi-Step Forget Gate Attenuation Trace on a Long-Term Memory

**Problem:**
An LSTM cell stores an essential scalar memory token $c_0 = 10.0$ at time step $t = 0$.
Over the subsequent $T = 100$ time steps, no new information is added ($\tilde{c}_t = 0$), so the cell state evolves purely via:
$$c_t = f_t \, c_{t-1} \implies c_{100} = c_0 \prod_{t=1}^{100} f_t$$
Similarly, an upstream gradient $\frac{\partial \mathcal{L}}{\partial c_{100}} = 1.0$ backpropagates along the CEC to $c_0$:
$$\frac{\partial \mathcal{L}}{\partial c_0} = \frac{\partial \mathcal{L}}{\partial c_{100}} \prod_{t=1}^{100} f_t = \prod_{t=1}^{100} f_t$$
Compare the retained memory $c_{100}$ and gradient transmission ratio $\frac{\partial \mathcal{L}}{\partial c_0}$ under four forget gate bias settings:
1. Zero bias ($b_f = 0.0 \implies f_t = \sigma(0) = 0.5$)
2. Default positive bias ($b_f = 1.0 \implies f_t = \sigma(1.0) \approx 0.731059$)
3. High positive bias ($b_f = 3.0 \implies f_t = \sigma(3.0) \approx 0.952574$)
4. Very high positive bias ($b_f = 5.0 \implies f_t = \sigma(5.0) \approx 0.993307$)

**Solution:**

#### Numerical Evaluations:
1. **Case 1: $b_f = 0.0 \implies f = 0.5$:**
   - Retained Memory: $c_{100} = 10.0 \times (0.5)^{100} = 10.0 \times 7.8886 \times 10^{-31} \approx \mathbf{7.89 \times 10^{-30}}$
   - Gradient Flow: $\frac{\partial \mathcal{L}}{\partial c_0} = (0.5)^{100} \approx \mathbf{7.89 \times 10^{-31}}$ (Complete extinction).

2. **Case 2: $b_f = 1.0 \implies f = 0.731059$:**
   - Retained Memory: $c_{100} = 10.0 \times (0.731059)^{100} = 10.0 \times 2.625 \times 10^{-14} \approx \mathbf{2.63 \times 10^{-13}}$
   - Gradient Flow: $\frac{\partial \mathcal{L}}{\partial c_0} \approx \mathbf{2.63 \times 10^{-14}}$ (Severe attenuation over 100 steps).

3. **Case 3: $b_f = 3.0 \implies f = 0.952574$:**
   - Retained Memory: $c_{100} = 10.0 \times (0.952574)^{100} \approx 10.0 \times 0.00762 = \mathbf{0.0762}$
   - Gradient Flow: $\frac{\partial \mathcal{L}}{\partial c_0} \approx \mathbf{0.00762}$ ($0.76\%$ signal preserved).

4. **Case 4: $b_f = 5.0 \implies f = 0.993307$:**
   - Retained Memory: $c_{100} = 10.0 \times (0.993307)^{100} \approx 10.0 \times 0.5103 = \mathbf{5.103}$
   - Gradient Flow: $\frac{\partial \mathcal{L}}{\partial c_0} \approx \mathbf{0.5103}$ (**$51.0\%$ of the gradient survives across 100 steps!**)

**Conclusion:**
This analytical trace proves why initializing forget gate biases to large positive values ($b_f \ge 1.0-2.0$) or learning near-identity gates ($f \approx 0.99$) is strictly required for the Constant Error Carousel to bridge century-scale sequence dependencies.

---

## 7. Deep Learning Connection & Application

### 1. Fused Gate Kernel Optimization in cuDNN
In production implementations (`torch.nn.LSTM`), PyTorch does not execute 4 separate matrix-vector products. Instead, it stacks all 4 weight matrices into a single monolithic weight tensor:
$$\mathbf{W}_{\text{all}} = \begin{bmatrix} \mathbf{W}_i \\ \mathbf{W}_f \\ \mathbf{W}_c \\ \mathbf{W}_o \end{bmatrix} \in \mathbb{R}^{4d_h \times (d_h + d_x)}$$
A single massive cuBLAS matrix multiplication computes all 4 pre-activations simultaneously:
$$\begin{bmatrix} \mathbf{a}_i \\ \mathbf{a}_f \\ \mathbf{a}_c \\ \mathbf{a}_o \end{bmatrix} = \mathbf{W}_{\text{all}} \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b}_{\text{all}}$$
Then, a fused CUDA element-wise kernel applies $\sigma$ and $\tanh$ activations in high-speed GPU registers without round-tripping to DRAM.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `LSTMCellScratch`: Pure NumPy implementation of forward pass and analytical backward pass.
2. `GRUCellScratch`: Pure NumPy implementation of forward pass and analytical backward pass.
3. Exact numerical verification of the Part 5 Visual Grid hand arithmetic.
4. Finite-difference numerical gradient checks for LSTM and GRU parameters.
5. End-to-end parity validation against PyTorch's `torch.nn.LSTMCell` and `torch.nn.GRUCell` with discrepancies $< 10^{-12}$.

See implementation in:
[`08_recurrent_networks/code/03_gated_architectures_lstm_and_gru.py`](./code/03_gated_architectures_lstm_and_gru.py)
